"""
Infrastructure Module

UPGRADE-015 Phase 7: Redis Infrastructure

Provides infrastructure components for the trading system:
- Redis client for caching
- Market data streams
- Time series storage
- Pub/Sub messaging

Usage:
    from infrastructure import (
        RedisClient,
        create_redis_client,
        MarketDataStream,
        TimeSeriesStore,
        PubSubManager,
    )
"""

from infrastructure.market_stream import (
    MarketDataStream,
    StreamConfig,
    create_market_stream,
)
from infrastructure.pubsub import (
    PubSubManager,
    create_pubsub_manager,
)
from infrastructure.redis_client import (
    RedisClient,
    RedisConfig,
    create_redis_client,
)
from infrastructure.timeseries import (
    TimeSeriesStore,
    create_timeseries_store,
)


__all__ = [
    # Redis Client
    "RedisClient",
    "RedisConfig",
    "create_redis_client",
    # Market Stream
    "MarketDataStream",
    "StreamConfig",
    "create_market_stream",
    # Time Series
    "TimeSeriesStore",
    "create_timeseries_store",
    # Pub/Sub
    "PubSubManager",
    "create_pubsub_manager",
]
