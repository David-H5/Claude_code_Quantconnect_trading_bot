"""
Market Data Stream for Trading Infrastructure

UPGRADE-015 Phase 7: Redis Infrastructure

Provides real-time market data streaming using Redis Streams.
Supports publishing and consuming market data with consumer groups.

Features:
- Redis Streams for ordered, persistent data
- Consumer groups for scalable processing
- Automatic stream trimming
- Reconnection handling
"""

import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
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
class StreamConfig:
    """Configuration for market data stream."""

    stream_name: str = "market:data"
    consumer_group: str = "trading_bot"
    consumer_name: str = "consumer_1"
    max_stream_length: int = 10000
    block_ms: int = 1000
    batch_size: int = 100
    auto_create_group: bool = True


@dataclass
class MarketTick:
    """A single market data tick."""

    symbol: str
    price: float
    volume: int
    bid: float | None = None
    ask: float | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "symbol": self.symbol,
            "price": str(self.price),
            "volume": str(self.volume),
            "bid": str(self.bid) if self.bid else "",
            "ask": str(self.ask) if self.ask else "",
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarketTick":
        """Create from dictionary."""
        return cls(
            symbol=data.get("symbol", ""),
            price=float(data.get("price", 0)),
            volume=int(data.get("volume", 0)),
            bid=float(data["bid"]) if data.get("bid") else None,
            ask=float(data["ask"]) if data.get("ask") else None,
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            source=data.get("source", "unknown"),
        )


class MarketDataStream:
    """Market data streaming using Redis Streams."""

    def __init__(
        self,
        redis_client: Any,
        config: StreamConfig | None = None,
    ):
        """
        Initialize market data stream.

        Args:
            redis_client: Redis client instance
            config: Stream configuration
        """
        self.config = config or StreamConfig()
        self._client = redis_client
        self._running = False
        self._callbacks: list[Callable[[MarketTick], None]] = []
        self._last_id = ">"  # Start from new messages

        if self._client and self.config.auto_create_group:
            self._ensure_consumer_group()

    def _ensure_consumer_group(self) -> None:
        """Create consumer group if it doesn't exist."""
        if not self._client:
            return

        try:
            self._client.xgroup_create(
                self.config.stream_name,
                self.config.consumer_group,
                id="0",
                mkstream=True,
            )
        except (redis.RedisError, OSError) as e:
            # Group already exists - that's fine
            if "BUSYGROUP" not in str(e):
                logger.debug(f"Failed to create consumer group: {e}")

    # ==========================================================================
    # Publishing
    # ==========================================================================

    def publish(self, tick: MarketTick) -> str | None:
        """
        Publish a market tick to the stream.

        Args:
            tick: Market tick to publish

        Returns:
            Message ID if successful, None otherwise
        """
        if not self._client:
            return None

        try:
            msg_id = self._client.xadd(
                self.config.stream_name,
                tick.to_dict(),
                maxlen=self.config.max_stream_length,
                approximate=True,
            )
            return msg_id
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to publish tick for {tick.symbol}: {e}")
            return None

    def publish_batch(self, ticks: list[MarketTick]) -> list[str]:
        """
        Publish multiple ticks efficiently.

        Args:
            ticks: List of market ticks

        Returns:
            List of message IDs
        """
        if not self._client or not ticks:
            return []

        ids = []
        pipe = self._client.pipeline()

        for tick in ticks:
            pipe.xadd(
                self.config.stream_name,
                tick.to_dict(),
                maxlen=self.config.max_stream_length,
                approximate=True,
            )

        try:
            results = pipe.execute()
            ids = [r for r in results if r]
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to publish batch of {len(ticks)} ticks: {e}")

        return ids

    def publish_quote(
        self,
        symbol: str,
        price: float,
        volume: int = 0,
        bid: float | None = None,
        ask: float | None = None,
        source: str = "api",
    ) -> str | None:
        """
        Convenience method to publish a quote.

        Args:
            symbol: Stock symbol
            price: Current price
            volume: Volume
            bid: Bid price
            ask: Ask price
            source: Data source

        Returns:
            Message ID if successful
        """
        tick = MarketTick(
            symbol=symbol.upper(),
            price=price,
            volume=volume,
            bid=bid,
            ask=ask,
            source=source,
        )
        return self.publish(tick)

    # ==========================================================================
    # Consuming
    # ==========================================================================

    def subscribe(self, callback: Callable[[MarketTick], None]) -> None:
        """
        Subscribe to market data updates.

        Args:
            callback: Function to call for each tick
        """
        self._callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[MarketTick], None]) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def read_new(self, count: int | None = None) -> list[MarketTick]:
        """
        Read new messages from the stream.

        Args:
            count: Maximum messages to read (default from config)

        Returns:
            List of market ticks
        """
        if not self._client:
            return []

        count = count or self.config.batch_size

        try:
            messages = self._client.xreadgroup(
                groupname=self.config.consumer_group,
                consumername=self.config.consumer_name,
                streams={self.config.stream_name: ">"},
                count=count,
                block=self.config.block_ms,
            )

            ticks = []
            if messages:
                for _stream_name, stream_messages in messages:
                    for msg_id, data in stream_messages:
                        tick = MarketTick.from_dict(data)
                        ticks.append(tick)
                        # Acknowledge message
                        self._client.xack(
                            self.config.stream_name,
                            self.config.consumer_group,
                            msg_id,
                        )

            return ticks
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to consume from stream: {e}")
            return []

    def read_range(
        self,
        start: str = "-",
        end: str = "+",
        count: int | None = None,
    ) -> list[MarketTick]:
        """
        Read a range of messages from the stream.

        Args:
            start: Start ID ("-" for oldest)
            end: End ID ("+" for newest)
            count: Maximum messages to read

        Returns:
            List of market ticks
        """
        if not self._client:
            return []

        try:
            messages = self._client.xrange(
                self.config.stream_name,
                min=start,
                max=end,
                count=count,
            )

            return [MarketTick.from_dict(data) for msg_id, data in messages]
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to read range from stream: {e}")
            return []

    def read_latest(self, symbol: str | None = None) -> MarketTick | None:
        """
        Read the latest tick, optionally for a specific symbol.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            Latest tick or None
        """
        ticks = self.read_range(start="-", end="+", count=100)

        if not ticks:
            return None

        if symbol:
            symbol = symbol.upper()
            ticks = [t for t in ticks if t.symbol == symbol]

        return ticks[-1] if ticks else None

    # ==========================================================================
    # Stream Management
    # ==========================================================================

    def start_consuming(self) -> None:
        """Start consuming messages in a loop."""
        self._running = True

        while self._running:
            ticks = self.read_new()

            for tick in ticks:
                for callback in self._callbacks:
                    with contextlib.suppress(Exception):
                        callback(tick)

            if not ticks:
                time.sleep(0.01)  # Small sleep to prevent busy loop

    def stop_consuming(self) -> None:
        """Stop the consuming loop."""
        self._running = False

    def get_stream_info(self) -> dict[str, Any]:
        """Get stream information."""
        if not self._client:
            return {}

        try:
            info = self._client.xinfo_stream(self.config.stream_name)
            return {
                "length": info.get("length", 0),
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
                "groups": info.get("groups", 0),
            }
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to get stream info: {e}")
            return {}

    def get_pending_count(self) -> int:
        """Get count of pending (unacknowledged) messages."""
        if not self._client:
            return 0

        try:
            pending = self._client.xpending(
                self.config.stream_name,
                self.config.consumer_group,
            )
            return pending.get("pending", 0) if pending else 0
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to get pending count: {e}")
            return 0

    def trim_stream(self, maxlen: int | None = None) -> int:
        """
        Trim stream to maximum length.

        Args:
            maxlen: Maximum length (default from config)

        Returns:
            Number of trimmed messages
        """
        if not self._client:
            return 0

        maxlen = maxlen or self.config.max_stream_length

        try:
            return self._client.xtrim(
                self.config.stream_name,
                maxlen=maxlen,
                approximate=True,
            )
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to trim stream: {e}")
            return 0

    def delete_stream(self) -> bool:
        """Delete the entire stream."""
        if not self._client:
            return False

        try:
            self._client.delete(self.config.stream_name)
            return True
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to delete stream: {e}")
            return False


def create_market_stream(
    redis_client: Any = None,
    stream_name: str = "market:data",
    consumer_group: str = "trading_bot",
) -> MarketDataStream:
    """
    Factory function to create a market data stream.

    Args:
        redis_client: Redis client (creates new if not provided)
        stream_name: Name of the Redis stream
        consumer_group: Consumer group name

    Returns:
        Configured MarketDataStream
    """
    config = StreamConfig(
        stream_name=stream_name,
        consumer_group=consumer_group,
    )
    return MarketDataStream(redis_client, config)
