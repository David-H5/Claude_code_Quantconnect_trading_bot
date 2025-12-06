"""
Tests for Redis Client

UPGRADE-015 Phase 7: Redis Infrastructure

Tests cover:
- Basic operations (get, set, delete)
- JSON operations
- Trading-specific operations
- Rate limiting
- Distributed locks
- Queue operations
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.redis_client import (
    HealthStatus,
    RedisClient,
    RedisConfig,
    create_redis_client,
)


class MockRedis:
    """Mock Redis client for testing."""

    def __init__(self):
        self.data = {}
        self.expire_times = {}

    def ping(self):
        return True

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None, px=None, nx=False, xx=False):
        if nx and key in self.data:
            return False
        if xx and key not in self.data:
            return False
        self.data[key] = value
        if ex:
            self.expire_times[key] = ex
        return True

    def delete(self, *keys):
        count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                count += 1
        return count

    def exists(self, *keys):
        return sum(1 for k in keys if k in self.data)

    def incr(self, key):
        if key not in self.data:
            self.data[key] = 0
        self.data[key] = int(self.data[key]) + 1
        return self.data[key]

    def expire(self, key, seconds):
        self.expire_times[key] = seconds
        return True

    def rpush(self, key, *values):
        if key not in self.data:
            self.data[key] = []
        self.data[key].extend(values)
        return len(self.data[key])

    def blpop(self, key, timeout=0):
        if self.data.get(key):
            return (key, self.data[key].pop(0))
        return None

    def llen(self, key):
        if key in self.data and isinstance(self.data[key], list):
            return len(self.data[key])
        return 0

    def lock(self, name, timeout=10, blocking=True, blocking_timeout=None):
        return MockLock(name)


class MockLock:
    """Mock Redis lock."""

    def __init__(self, name):
        self.name = name
        self.acquired = False

    def acquire(self):
        self.acquired = True
        return True

    def release(self):
        self.acquired = False


class TestRedisConfig:
    """Test RedisConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RedisConfig()

        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.socket_timeout == 5.0
        assert config.max_connections == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = RedisConfig(
            host="redis.example.com",
            port=6380,
            db=1,
            password="secret",
            max_connections=20,
        )

        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.db == 1
        assert config.password == "secret"
        assert config.max_connections == 20


class TestHealthStatus:
    """Test HealthStatus dataclass."""

    def test_default_status(self):
        """Test default health status."""
        status = HealthStatus()

        assert status.connected is False
        assert status.latency_ms == 0.0
        assert status.error is None
        assert isinstance(status.last_check, datetime)

    def test_custom_status(self):
        """Test custom health status."""
        status = HealthStatus(
            connected=True,
            latency_ms=5.5,
            error=None,
        )

        assert status.connected is True
        assert status.latency_ms == 5.5


class TestRedisClient:
    """Test RedisClient class."""

    def test_client_initialization(self):
        """Test client initialization without Redis."""
        with patch("infrastructure.redis_client.REDIS_AVAILABLE", False):
            client = RedisClient()
            assert client._client is None
            assert "not installed" in client._health_status.error

    def test_ping_without_client(self):
        """Test ping when client is not available."""
        client = RedisClient.__new__(RedisClient)
        client._client = None
        client._health_status = HealthStatus()

        assert client.ping() is False

    def test_ping_with_mock(self):
        """Test ping with mock Redis."""
        client = RedisClient.__new__(RedisClient)
        client._client = MockRedis()
        client._health_status = HealthStatus()

        assert client.ping() is True
        assert client._health_status.connected is True

    def test_get_set(self):
        """Test basic get and set operations."""
        client = RedisClient.__new__(RedisClient)
        client._client = MockRedis()

        # Set a value
        assert client.set("test_key", "test_value") is True

        # Get the value
        assert client.get("test_key") == "test_value"

        # Get non-existent key
        assert client.get("nonexistent") is None

    def test_set_with_expiry(self):
        """Test set with expiry time."""
        client = RedisClient.__new__(RedisClient)
        client._client = MockRedis()

        assert client.set("expiring_key", "value", ex=60) is True
        assert client._client.expire_times.get("expiring_key") == 60

    def test_set_nx(self):
        """Test set with NX flag (only if not exists)."""
        client = RedisClient.__new__(RedisClient)
        client._client = MockRedis()

        # First set should succeed
        assert client.set("nx_key", "value1", nx=True) is True

        # Second set should fail
        assert client.set("nx_key", "value2", nx=True) is False

        # Value should be original
        assert client.get("nx_key") == "value1"

    def test_delete(self):
        """Test delete operation."""
        client = RedisClient.__new__(RedisClient)
        client._client = MockRedis()

        client.set("key1", "value1")
        client.set("key2", "value2")

        # Delete one key
        assert client.delete("key1") == 1
        assert client.get("key1") is None

        # Delete non-existent key
        assert client.delete("nonexistent") == 0

    def test_exists(self):
        """Test exists operation."""
        client = RedisClient.__new__(RedisClient)
        client._client = MockRedis()

        client.set("key1", "value1")

        assert client.exists("key1") == 1
        assert client.exists("nonexistent") == 0
        assert client.exists("key1", "nonexistent") == 1


class TestRedisClientJson:
    """Test JSON operations."""

    def test_get_set_json(self):
        """Test JSON get and set."""
        client = RedisClient.__new__(RedisClient)
        client._client = MockRedis()
        client.config = RedisConfig()

        data = {"name": "test", "value": 123, "nested": {"a": 1}}

        # Set JSON
        assert client.set_json("json_key", data) is True

        # Get JSON
        result = client.get_json("json_key")
        assert result == data

    def test_get_json_invalid(self):
        """Test getting invalid JSON."""
        client = RedisClient.__new__(RedisClient)
        client._client = MockRedis()

        # Set non-JSON value
        client.set("invalid_json", "not json {")

        # Should return None for invalid JSON
        assert client.get_json("invalid_json") is None


class TestRedisClientTrading:
    """Test trading-specific operations."""

    def test_cache_quote(self):
        """Test caching a market quote."""
        client = RedisClient.__new__(RedisClient)
        client._client = MockRedis()
        client.config = RedisConfig()

        quote = {
            "bid": 100.50,
            "ask": 100.55,
            "last": 100.52,
            "volume": 10000,
        }

        assert client.cache_quote("AAPL", quote, ttl_seconds=30) is True

        # Retrieve the quote
        cached = client.get_cached_quote("AAPL")
        assert cached is not None
        assert cached["bid"] == 100.50
        assert "cached_at" in cached

    def test_cache_position(self):
        """Test caching a position."""
        client = RedisClient.__new__(RedisClient)
        client._client = MockRedis()
        client.config = RedisConfig()

        position = {
            "quantity": 100,
            "avg_price": 150.25,
            "market_value": 15025.00,
        }

        assert client.cache_position("SPY", position) is True

        # Retrieve position
        cached = client.get_cached_position("SPY")
        assert cached is not None
        assert cached["quantity"] == 100

    def test_rate_limit(self):
        """Test rate limiting."""
        client = RedisClient.__new__(RedisClient)
        client._client = MockRedis()

        # First request
        allowed, remaining = client.set_rate_limit("api_calls", 5, 60)
        assert allowed is True
        assert remaining == 4

        # More requests
        for _ in range(4):
            allowed, remaining = client.set_rate_limit("api_calls", 5, 60)

        # Should still be allowed on 5th
        assert allowed is True
        assert remaining == 0

        # 6th should exceed
        allowed, remaining = client.set_rate_limit("api_calls", 5, 60)
        assert allowed is False

    def test_acquire_release_lock(self):
        """Test distributed lock."""
        client = RedisClient.__new__(RedisClient)
        client._client = MockRedis()

        # Acquire lock
        lock = client.acquire_lock("test_lock", timeout=10)
        assert lock is not None
        assert lock.acquired is True

        # Release lock
        assert client.release_lock(lock) is True


class TestRedisClientQueue:
    """Test queue operations."""

    def test_push_pop_queue(self):
        """Test queue push and pop."""
        client = RedisClient.__new__(RedisClient)
        client._client = MockRedis()

        # Push items
        assert client.push_to_queue("test_queue", "item1", "item2") == 2

        # Pop item
        item = client.pop_from_queue("test_queue", timeout=0)
        assert item == "item1"

        # Check length
        assert client.queue_length("test_queue") == 1

    def test_queue_empty(self):
        """Test popping from empty queue."""
        client = RedisClient.__new__(RedisClient)
        client._client = MockRedis()

        # Pop from empty queue
        item = client.pop_from_queue("empty_queue", timeout=0)
        assert item is None


class TestCreateRedisClient:
    """Test factory function."""

    def test_create_with_defaults(self):
        """Test creating client with defaults."""
        with patch("infrastructure.redis_client.REDIS_AVAILABLE", False):
            client = create_redis_client()
            assert client.config.host == "localhost"
            assert client.config.port == 6379

    def test_create_with_env_vars(self):
        """Test creating client with environment variables."""
        with (
            patch("infrastructure.redis_client.REDIS_AVAILABLE", False),
            patch.dict(
                "os.environ",
                {
                    "REDIS_HOST": "redis.test.com",
                    "REDIS_PORT": "6380",
                },
            ),
        ):
            client = create_redis_client()
            assert client.config.host == "redis.test.com"
            assert client.config.port == 6380

    def test_create_with_params(self):
        """Test creating client with explicit parameters."""
        with patch("infrastructure.redis_client.REDIS_AVAILABLE", False):
            client = create_redis_client(
                host="custom.redis.com",
                port=6381,
                password="test_password",
            )
            assert client.config.host == "custom.redis.com"
            assert client.config.port == 6381
            assert client.config.password == "test_password"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
