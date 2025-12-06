"""
Redis Client for Trading Infrastructure

UPGRADE-015 Phase 7: Redis Infrastructure

Provides a Redis client wrapper with connection management,
health checks, and trading-specific utilities.

Features:
- Connection pooling
- Health monitoring
- Automatic reconnection
- Trading-specific operations
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


logger = logging.getLogger(__name__)

# Optional import - gracefully degrade if not installed
try:
    import redis
    from redis import ConnectionPool, Redis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore
    Redis = None  # type: ignore
    ConnectionPool = None  # type: ignore
    REDIS_AVAILABLE = False


@dataclass
class RedisConfig:
    """Redis connection configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    max_connections: int = 10
    decode_responses: bool = True
    health_check_interval: int = 30


@dataclass
class HealthStatus:
    """Redis health status."""

    connected: bool = False
    latency_ms: float = 0.0
    last_check: datetime = field(default_factory=datetime.utcnow)
    error: str | None = None


class RedisClient:
    """Redis client wrapper with trading-specific features."""

    def __init__(self, config: RedisConfig | None = None):
        """
        Initialize Redis client.

        Args:
            config: Redis configuration (uses defaults if not provided)
        """
        self.config = config or RedisConfig()
        self._pool: Any = None
        self._client: Any = None
        self._health_status = HealthStatus()

        if not REDIS_AVAILABLE:
            self._health_status.error = "Redis library not installed"
            return

        self._create_pool()

    def _create_pool(self) -> None:
        """Create connection pool."""
        if not REDIS_AVAILABLE:
            return

        self._pool = ConnectionPool(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            max_connections=self.config.max_connections,
            decode_responses=self.config.decode_responses,
            health_check_interval=self.config.health_check_interval,
        )
        self._client = Redis(connection_pool=self._pool)

    def ping(self) -> bool:
        """Check if Redis is available."""
        if not REDIS_AVAILABLE or not self._client:
            return False

        try:
            start = datetime.utcnow()
            result = self._client.ping()
            latency = (datetime.utcnow() - start).total_seconds() * 1000

            self._health_status.connected = result
            self._health_status.latency_ms = latency
            self._health_status.last_check = datetime.utcnow()
            self._health_status.error = None

            return result
        except (redis.RedisError, OSError) as e:
            logger.warning(f"Redis health check failed: {e}")
            self._health_status.connected = False
            self._health_status.error = str(e)
            self._health_status.last_check = datetime.utcnow()
            return False

    def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        return self._health_status

    # ==========================================================================
    # Basic Operations
    # ==========================================================================

    def get(self, key: str) -> str | None:
        """Get a value by key."""
        if not self._client:
            return None
        try:
            return self._client.get(key)
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Redis get failed for key '{key}': {e}")
            return None

    def set(
        self,
        key: str,
        value: str,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """
        Set a value.

        Args:
            key: Key name
            value: Value to set
            ex: Expire time in seconds
            px: Expire time in milliseconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists
        """
        if not self._client:
            return False
        try:
            return bool(self._client.set(key, value, ex=ex, px=px, nx=nx, xx=xx))
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Redis set failed for key '{key}': {e}")
            return False

    def delete(self, *keys: str) -> int:
        """Delete keys."""
        if not self._client or not keys:
            return 0
        try:
            return self._client.delete(*keys)
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Redis delete failed for keys {keys}: {e}")
            return 0

    def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        if not self._client or not keys:
            return 0
        try:
            return self._client.exists(*keys)
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Redis exists check failed for keys {keys}: {e}")
            return 0

    # ==========================================================================
    # JSON Operations
    # ==========================================================================

    def get_json(self, key: str) -> dict[str, Any] | None:
        """Get a JSON value."""
        value = self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return None

    def set_json(
        self,
        key: str,
        value: dict[str, Any],
        ex: int | None = None,
    ) -> bool:
        """Set a JSON value."""
        try:
            json_str = json.dumps(value, default=str)
            return self.set(key, json_str, ex=ex)
        except (TypeError, ValueError):
            return False

    # ==========================================================================
    # Trading-Specific Operations
    # ==========================================================================

    def cache_quote(
        self,
        symbol: str,
        quote: dict[str, Any],
        ttl_seconds: int = 60,
    ) -> bool:
        """Cache a market quote."""
        key = f"quote:{symbol.upper()}"
        quote["cached_at"] = datetime.utcnow().isoformat()
        return self.set_json(key, quote, ex=ttl_seconds)

    def get_cached_quote(self, symbol: str) -> dict[str, Any] | None:
        """Get a cached market quote."""
        key = f"quote:{symbol.upper()}"
        return self.get_json(key)

    def cache_position(
        self,
        symbol: str,
        position: dict[str, Any],
        ttl_seconds: int = 300,
    ) -> bool:
        """Cache a position."""
        key = f"position:{symbol.upper()}"
        position["cached_at"] = datetime.utcnow().isoformat()
        return self.set_json(key, position, ex=ttl_seconds)

    def get_cached_position(self, symbol: str) -> dict[str, Any] | None:
        """Get a cached position."""
        key = f"position:{symbol.upper()}"
        return self.get_json(key)

    def set_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """
        Check and update rate limit.

        Returns:
            (allowed, remaining) tuple
        """
        if not self._client:
            return True, max_requests

        try:
            rate_key = f"ratelimit:{key}"
            current = self._client.incr(rate_key)

            if current == 1:
                self._client.expire(rate_key, window_seconds)

            allowed = current <= max_requests
            remaining = max(0, max_requests - current)
            return allowed, remaining
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Redis rate limit check failed for key '{key}': {e}")
            return True, max_requests

    def acquire_lock(
        self,
        name: str,
        timeout: int = 10,
        blocking: bool = True,
        blocking_timeout: float | None = None,
    ) -> Any:
        """
        Acquire a distributed lock.

        Args:
            name: Lock name
            timeout: Lock timeout in seconds
            blocking: Whether to wait for lock
            blocking_timeout: Max time to wait for lock

        Returns:
            Lock object if acquired, None otherwise
        """
        if not self._client:
            return None

        try:
            lock = self._client.lock(
                f"lock:{name}",
                timeout=timeout,
                blocking=blocking,
                blocking_timeout=blocking_timeout,
            )
            if lock.acquire():
                return lock
            return None
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Redis acquire lock failed for '{name}': {e}")
            return None

    def release_lock(self, lock: Any) -> bool:
        """Release a distributed lock."""
        if not lock:
            return False
        try:
            lock.release()
            return True
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Redis release lock failed: {e}")
            return False

    # ==========================================================================
    # List Operations (for queues)
    # ==========================================================================

    def push_to_queue(self, queue_name: str, *values: str) -> int:
        """Push values to a queue (list)."""
        if not self._client or not values:
            return 0
        try:
            return self._client.rpush(queue_name, *values)
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Redis push to queue '{queue_name}' failed: {e}")
            return 0

    def pop_from_queue(
        self,
        queue_name: str,
        timeout: int = 0,
    ) -> str | None:
        """Pop a value from a queue (blocking)."""
        if not self._client:
            return None
        try:
            result = self._client.blpop(queue_name, timeout=timeout)
            return result[1] if result else None
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Redis pop from queue '{queue_name}' failed: {e}")
            return None

    def queue_length(self, queue_name: str) -> int:
        """Get queue length."""
        if not self._client:
            return 0
        try:
            return self._client.llen(queue_name)
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Redis queue length for '{queue_name}' failed: {e}")
            return 0

    # ==========================================================================
    # Cleanup
    # ==========================================================================

    def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            try:
                self._pool.disconnect()
            except (redis.RedisError, OSError) as e:
                logger.debug(f"Redis connection pool close failed: {e}")


def create_redis_client(
    host: str | None = None,
    port: int | None = None,
    password: str | None = None,
) -> RedisClient:
    """
    Factory function to create a Redis client.

    Args:
        host: Redis host (or use REDIS_HOST env var)
        port: Redis port (or use REDIS_PORT env var)
        password: Redis password (or use REDIS_PASSWORD env var)

    Returns:
        Configured RedisClient
    """
    config = RedisConfig(
        host=host or os.environ.get("REDIS_HOST", "localhost"),
        port=port or int(os.environ.get("REDIS_PORT", "6379")),
        password=password or os.environ.get("REDIS_PASSWORD"),
    )
    return RedisClient(config)
