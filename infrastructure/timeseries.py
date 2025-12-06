"""
Time Series Store for Trading Infrastructure

UPGRADE-015 Phase 7: Redis Infrastructure

Provides time-series data storage using Redis sorted sets.
Optimized for OHLCV data, indicators, and metrics.

Features:
- Sorted set storage for time-ordered data
- Automatic data retention/trimming
- Range queries by timestamp
- Aggregation support
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


logger = logging.getLogger(__name__)


# Optional import - gracefully degrade if not installed
try:
    import redis
    from redis import Redis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore
    Redis = None  # type: ignore
    REDIS_AVAILABLE = False


@dataclass
class TimeSeriesConfig:
    """Configuration for time series store."""

    key_prefix: str = "ts"
    max_points: int = 100000
    default_retention_hours: int = 168  # 7 days
    auto_trim: bool = True


@dataclass
class DataPoint:
    """A single time series data point."""

    timestamp: datetime
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataPoint":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            value=float(data["value"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OHLCVBar:
    """OHLCV bar data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OHLCVBar":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=int(data["volume"]),
        )


class TimeSeriesStore:
    """Time series data store using Redis sorted sets."""

    def __init__(
        self,
        redis_client: Any,
        config: TimeSeriesConfig | None = None,
    ):
        """
        Initialize time series store.

        Args:
            redis_client: Redis client instance
            config: Store configuration
        """
        self.config = config or TimeSeriesConfig()
        self._client = redis_client

    def _make_key(self, series_name: str) -> str:
        """Create Redis key for a series."""
        return f"{self.config.key_prefix}:{series_name}"

    def _timestamp_to_score(self, ts: datetime) -> float:
        """Convert timestamp to Redis score."""
        return ts.timestamp()

    def _score_to_timestamp(self, score: float) -> datetime:
        """Convert Redis score to timestamp."""
        return datetime.fromtimestamp(score)

    # ==========================================================================
    # Basic Operations
    # ==========================================================================

    def add(
        self,
        series_name: str,
        value: float,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Add a data point to a series.

        Args:
            series_name: Name of the series
            value: Numeric value
            timestamp: Timestamp (default: now)
            metadata: Optional metadata

        Returns:
            True if successful
        """
        if not self._client:
            return False

        timestamp = timestamp or datetime.utcnow()
        key = self._make_key(series_name)
        score = self._timestamp_to_score(timestamp)

        data = {
            "value": value,
            "metadata": metadata or {},
        }

        try:
            # Use score as timestamp, data as JSON member
            member = json.dumps(data, default=str)
            self._client.zadd(key, {member: score})

            if self.config.auto_trim:
                self._trim_series(key)

            return True
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to add point to series '{series_name}': {e}")
            return False

    def add_batch(
        self,
        series_name: str,
        points: list[DataPoint],
    ) -> int:
        """
        Add multiple data points efficiently.

        Args:
            series_name: Name of the series
            points: List of data points

        Returns:
            Number of points added
        """
        if not self._client or not points:
            return 0

        key = self._make_key(series_name)
        mapping = {}

        for point in points:
            score = self._timestamp_to_score(point.timestamp)
            data = {"value": point.value, "metadata": point.metadata}
            member = json.dumps(data, default=str)
            mapping[member] = score

        try:
            added = self._client.zadd(key, mapping)

            if self.config.auto_trim:
                self._trim_series(key)

            return added
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to add batch to series '{series_name}': {e}")
            return 0

    def get_range(
        self,
        series_name: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[DataPoint]:
        """
        Get data points in a time range.

        Args:
            series_name: Name of the series
            start: Start timestamp (default: 24 hours ago)
            end: End timestamp (default: now)
            limit: Maximum points to return

        Returns:
            List of data points
        """
        if not self._client:
            return []

        end = end or datetime.utcnow()
        start = start or (end - timedelta(hours=24))

        key = self._make_key(series_name)
        start_score = self._timestamp_to_score(start)
        end_score = self._timestamp_to_score(end)

        try:
            results = self._client.zrangebyscore(
                key,
                min=start_score,
                max=end_score,
                withscores=True,
                start=0,
                num=limit or -1,
            )

            points = []
            for member, score in results:
                data = json.loads(member)
                points.append(
                    DataPoint(
                        timestamp=self._score_to_timestamp(score),
                        value=data["value"],
                        metadata=data.get("metadata", {}),
                    )
                )

            return points
        except (redis.RedisError, OSError, json.JSONDecodeError) as e:
            logger.debug(f"Failed to get range from series '{series_name}': {e}")
            return []

    def get_latest(
        self,
        series_name: str,
        count: int = 1,
    ) -> list[DataPoint]:
        """
        Get the latest data points.

        Args:
            series_name: Name of the series
            count: Number of points to return

        Returns:
            List of data points (newest first)
        """
        if not self._client:
            return []

        key = self._make_key(series_name)

        try:
            results = self._client.zrevrange(
                key,
                start=0,
                end=count - 1,
                withscores=True,
            )

            points = []
            for member, score in results:
                data = json.loads(member)
                points.append(
                    DataPoint(
                        timestamp=self._score_to_timestamp(score),
                        value=data["value"],
                        metadata=data.get("metadata", {}),
                    )
                )

            return points
        except (redis.RedisError, OSError, json.JSONDecodeError) as e:
            logger.debug(f"Failed to get latest from series '{series_name}': {e}")
            return []

    # ==========================================================================
    # OHLCV Operations
    # ==========================================================================

    def add_ohlcv(
        self,
        symbol: str,
        bar: OHLCVBar,
        timeframe: str = "1m",
    ) -> bool:
        """
        Add an OHLCV bar.

        Args:
            symbol: Stock symbol
            bar: OHLCV bar data
            timeframe: Timeframe (1m, 5m, 1h, 1d, etc.)

        Returns:
            True if successful
        """
        if not self._client:
            return False

        series_name = f"ohlcv:{symbol.upper()}:{timeframe}"
        key = self._make_key(series_name)
        score = self._timestamp_to_score(bar.timestamp)

        try:
            member = json.dumps(bar.to_dict(), default=str)
            self._client.zadd(key, {member: score})

            if self.config.auto_trim:
                self._trim_series(key)

            return True
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to add OHLCV bar for {symbol}: {e}")
            return False

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[OHLCVBar]:
        """
        Get OHLCV bars.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe
            start: Start timestamp
            end: End timestamp
            limit: Maximum bars to return

        Returns:
            List of OHLCV bars
        """
        if not self._client:
            return []

        series_name = f"ohlcv:{symbol.upper()}:{timeframe}"
        key = self._make_key(series_name)

        end = end or datetime.utcnow()
        start = start or (end - timedelta(days=1))

        start_score = self._timestamp_to_score(start)
        end_score = self._timestamp_to_score(end)

        try:
            results = self._client.zrangebyscore(
                key,
                min=start_score,
                max=end_score,
                withscores=True,
                start=0,
                num=limit or -1,
            )

            bars = []
            for member, _score in results:
                data = json.loads(member)
                bars.append(OHLCVBar.from_dict(data))

            return bars
        except (redis.RedisError, OSError, json.JSONDecodeError) as e:
            logger.debug(f"Failed to get OHLCV for {symbol}: {e}")
            return []

    def get_latest_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        count: int = 1,
    ) -> list[OHLCVBar]:
        """
        Get the latest OHLCV bars.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe
            count: Number of bars

        Returns:
            List of bars (newest first)
        """
        if not self._client:
            return []

        series_name = f"ohlcv:{symbol.upper()}:{timeframe}"
        key = self._make_key(series_name)

        try:
            results = self._client.zrevrange(
                key,
                start=0,
                end=count - 1,
                withscores=True,
            )

            bars = []
            for member, _score in results:
                data = json.loads(member)
                bars.append(OHLCVBar.from_dict(data))

            return bars
        except (redis.RedisError, OSError, json.JSONDecodeError) as e:
            logger.debug(f"Failed to get latest OHLCV for {symbol}: {e}")
            return []

    # ==========================================================================
    # Indicator Storage
    # ==========================================================================

    def add_indicator(
        self,
        symbol: str,
        indicator_name: str,
        value: float,
        timestamp: datetime | None = None,
        params: dict[str, Any] | None = None,
    ) -> bool:
        """
        Store an indicator value.

        Args:
            symbol: Stock symbol
            indicator_name: Indicator name (e.g., "RSI_14")
            value: Indicator value
            timestamp: Timestamp
            params: Indicator parameters

        Returns:
            True if successful
        """
        series_name = f"indicator:{symbol.upper()}:{indicator_name}"
        return self.add(
            series_name,
            value,
            timestamp,
            metadata={"params": params or {}},
        )

    def get_indicator(
        self,
        symbol: str,
        indicator_name: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[DataPoint]:
        """
        Get indicator values.

        Args:
            symbol: Stock symbol
            indicator_name: Indicator name
            start: Start timestamp
            end: End timestamp
            limit: Maximum points

        Returns:
            List of data points
        """
        series_name = f"indicator:{symbol.upper()}:{indicator_name}"
        return self.get_range(series_name, start, end, limit)

    # ==========================================================================
    # Aggregation
    # ==========================================================================

    def get_stats(
        self,
        series_name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, float]:
        """
        Get statistics for a series in a time range.

        Args:
            series_name: Name of the series
            start: Start timestamp
            end: End timestamp

        Returns:
            Statistics dictionary
        """
        points = self.get_range(series_name, start, end)

        if not points:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "sum": 0.0,
                "avg": 0.0,
            }

        values = [p.value for p in points]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
        }

    # ==========================================================================
    # Management
    # ==========================================================================

    def _trim_series(self, key: str) -> int:
        """Trim series to max points."""
        if not self._client:
            return 0

        try:
            current_count = self._client.zcard(key)
            if current_count > self.config.max_points:
                # Remove oldest entries
                to_remove = current_count - self.config.max_points
                return self._client.zremrangebyrank(key, 0, to_remove - 1)
            return 0
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to trim series '{key}': {e}")
            return 0

    def trim_by_time(
        self,
        series_name: str,
        retention_hours: int | None = None,
    ) -> int:
        """
        Remove data older than retention period.

        Args:
            series_name: Name of the series
            retention_hours: Hours to retain (default from config)

        Returns:
            Number of removed points
        """
        if not self._client:
            return 0

        retention_hours = retention_hours or self.config.default_retention_hours
        key = self._make_key(series_name)
        cutoff = datetime.utcnow() - timedelta(hours=retention_hours)
        cutoff_score = self._timestamp_to_score(cutoff)

        try:
            return self._client.zremrangebyscore(key, "-inf", cutoff_score)
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to trim by time for series '{series_name}': {e}")
            return 0

    def get_series_count(self, series_name: str) -> int:
        """Get the number of points in a series."""
        if not self._client:
            return 0

        key = self._make_key(series_name)
        try:
            return self._client.zcard(key)
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to get series count for '{series_name}': {e}")
            return 0

    def delete_series(self, series_name: str) -> bool:
        """Delete an entire series."""
        if not self._client:
            return False

        key = self._make_key(series_name)
        try:
            self._client.delete(key)
            return True
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to delete series '{series_name}': {e}")
            return False

    def list_series(self, pattern: str = "*") -> list[str]:
        """
        List all series matching a pattern.

        Args:
            pattern: Glob pattern

        Returns:
            List of series names
        """
        if not self._client:
            return []

        full_pattern = f"{self.config.key_prefix}:{pattern}"
        try:
            keys = self._client.keys(full_pattern)
            prefix_len = len(self.config.key_prefix) + 1
            return [k[prefix_len:] for k in keys]
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to list series with pattern '{pattern}': {e}")
            return []


def create_timeseries_store(
    redis_client: Any = None,
    key_prefix: str = "ts",
    max_points: int = 100000,
) -> TimeSeriesStore:
    """
    Factory function to create a time series store.

    Args:
        redis_client: Redis client (required)
        key_prefix: Prefix for Redis keys
        max_points: Maximum points per series

    Returns:
        Configured TimeSeriesStore
    """
    config = TimeSeriesConfig(
        key_prefix=key_prefix,
        max_points=max_points,
    )
    return TimeSeriesStore(redis_client, config)
