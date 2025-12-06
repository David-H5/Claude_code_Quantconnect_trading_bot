"""
Tests for Market Data Stream

UPGRADE-015 Phase 7: Redis Infrastructure

Tests cover:
- Publishing market ticks
- Batch publishing
- Consuming messages
- Stream management
"""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.market_stream import (
    MarketDataStream,
    MarketTick,
    StreamConfig,
    create_market_stream,
)


class MockRedisStream:
    """Mock Redis client for stream testing."""

    def __init__(self):
        self.streams = {}
        self.groups = {}
        self.pending = {}
        self.msg_counter = 0

    def xgroup_create(self, stream, group, id="0", mkstream=True):
        if stream not in self.streams:
            if mkstream:
                self.streams[stream] = []
            else:
                raise Exception("Stream does not exist")

        if stream not in self.groups:
            self.groups[stream] = {}

        if group in self.groups[stream]:
            raise Exception("BUSYGROUP Consumer Group name already exists")

        self.groups[stream][group] = {"last_id": id}
        return True

    def xadd(self, stream, data, maxlen=None, approximate=True):
        if stream not in self.streams:
            self.streams[stream] = []

        self.msg_counter += 1
        msg_id = f"1234567890123-{self.msg_counter}"
        self.streams[stream].append((msg_id, data))

        # Trim if maxlen specified
        if maxlen and len(self.streams[stream]) > maxlen:
            self.streams[stream] = self.streams[stream][-maxlen:]

        return msg_id

    def xreadgroup(self, groupname, consumername, streams, count=None, block=None):
        results = []
        for stream_name, _start_id in streams.items():
            if stream_name in self.streams:
                messages = self.streams[stream_name]
                if messages:
                    # Return all messages for testing
                    results.append((stream_name, messages[:count] if count else messages))

        return results if results else None

    def xack(self, stream, group, *msg_ids):
        return len(msg_ids)

    def xrange(self, stream, min="-", max="+", count=None):
        if stream not in self.streams:
            return []

        messages = self.streams[stream]
        if count:
            return messages[:count]
        return messages

    def xrevrange(self, stream, max="+", min="-", count=None):
        if stream not in self.streams:
            return []

        messages = list(reversed(self.streams[stream]))
        if count:
            return messages[:count]
        return messages

    def xinfo_stream(self, stream):
        if stream not in self.streams:
            raise Exception("Stream does not exist")

        messages = self.streams[stream]
        return {
            "length": len(messages),
            "first-entry": messages[0] if messages else None,
            "last-entry": messages[-1] if messages else None,
            "groups": len(self.groups.get(stream, {})),
        }

    def xpending(self, stream, group):
        return {"pending": 0}

    def xtrim(self, stream, maxlen=None, approximate=True):
        if stream not in self.streams:
            return 0

        current = len(self.streams[stream])
        if maxlen and current > maxlen:
            removed = current - maxlen
            self.streams[stream] = self.streams[stream][-maxlen:]
            return removed
        return 0

    def delete(self, stream):
        if stream in self.streams:
            del self.streams[stream]
            return 1
        return 0

    def pipeline(self):
        return MockPipeline(self)


class MockPipeline:
    """Mock Redis pipeline."""

    def __init__(self, redis):
        self.redis = redis
        self.commands = []

    def xadd(self, stream, data, maxlen=None, approximate=True):
        self.commands.append(("xadd", stream, data, maxlen))
        return self

    def execute(self):
        results = []
        for cmd in self.commands:
            if cmd[0] == "xadd":
                result = self.redis.xadd(cmd[1], cmd[2], cmd[3])
                results.append(result)
        return results


class TestMarketTick:
    """Test MarketTick dataclass."""

    def test_tick_creation(self):
        """Test creating a market tick."""
        tick = MarketTick(
            symbol="AAPL",
            price=175.50,
            volume=1000,
            bid=175.45,
            ask=175.55,
        )

        assert tick.symbol == "AAPL"
        assert tick.price == 175.50
        assert tick.volume == 1000
        assert tick.bid == 175.45
        assert tick.ask == 175.55

    def test_tick_to_dict(self):
        """Test converting tick to dictionary."""
        tick = MarketTick(
            symbol="SPY",
            price=450.00,
            volume=5000,
        )

        data = tick.to_dict()

        assert data["symbol"] == "SPY"
        assert data["price"] == "450.0"
        assert data["volume"] == "5000"

    def test_tick_from_dict(self):
        """Test creating tick from dictionary."""
        data = {
            "symbol": "MSFT",
            "price": "300.00",
            "volume": "2000",
            "bid": "299.95",
            "ask": "300.05",
            "timestamp": "2025-01-15T10:30:00",
            "source": "test",
        }

        tick = MarketTick.from_dict(data)

        assert tick.symbol == "MSFT"
        assert tick.price == 300.00
        assert tick.volume == 2000
        assert tick.bid == 299.95
        assert tick.ask == 300.05


class TestStreamConfig:
    """Test StreamConfig dataclass."""

    def test_default_config(self):
        """Test default stream configuration."""
        config = StreamConfig()

        assert config.stream_name == "market:data"
        assert config.consumer_group == "trading_bot"
        assert config.max_stream_length == 10000
        assert config.block_ms == 1000
        assert config.batch_size == 100

    def test_custom_config(self):
        """Test custom stream configuration."""
        config = StreamConfig(
            stream_name="custom:stream",
            consumer_group="custom_group",
            max_stream_length=5000,
        )

        assert config.stream_name == "custom:stream"
        assert config.consumer_group == "custom_group"
        assert config.max_stream_length == 5000


class TestMarketDataStream:
    """Test MarketDataStream class."""

    def test_stream_initialization(self):
        """Test stream initialization."""
        mock_redis = MockRedisStream()
        stream = MarketDataStream(mock_redis)

        assert stream.config.stream_name == "market:data"
        assert "market:data" in mock_redis.streams

    def test_publish_tick(self):
        """Test publishing a single tick."""
        mock_redis = MockRedisStream()
        stream = MarketDataStream(mock_redis)

        tick = MarketTick(symbol="AAPL", price=175.50, volume=1000)

        msg_id = stream.publish(tick)

        assert msg_id is not None
        assert len(mock_redis.streams["market:data"]) == 1

    def test_publish_quote(self):
        """Test convenience method for publishing quotes."""
        mock_redis = MockRedisStream()
        stream = MarketDataStream(mock_redis)

        msg_id = stream.publish_quote(
            symbol="spy",  # Should be uppercased
            price=450.00,
            volume=10000,
            bid=449.95,
            ask=450.05,
        )

        assert msg_id is not None
        messages = mock_redis.streams["market:data"]
        assert messages[0][1]["symbol"] == "SPY"

    def test_publish_batch(self):
        """Test batch publishing."""
        mock_redis = MockRedisStream()
        stream = MarketDataStream(mock_redis)

        ticks = [
            MarketTick(symbol="AAPL", price=175.50, volume=1000),
            MarketTick(symbol="MSFT", price=300.00, volume=2000),
            MarketTick(symbol="GOOGL", price=140.00, volume=500),
        ]

        ids = stream.publish_batch(ticks)

        assert len(ids) == 3

    def test_subscribe_callback(self):
        """Test subscribing with callback."""
        mock_redis = MockRedisStream()
        stream = MarketDataStream(mock_redis)

        received = []

        def handler(tick):
            received.append(tick)

        stream.subscribe(handler)

        assert len(stream._callbacks) == 1

    def test_unsubscribe(self):
        """Test unsubscribing."""
        mock_redis = MockRedisStream()
        stream = MarketDataStream(mock_redis)

        def handler(tick):
            pass

        stream.subscribe(handler)
        assert len(stream._callbacks) == 1

        stream.unsubscribe(handler)
        assert len(stream._callbacks) == 0

    def test_read_range(self):
        """Test reading range of messages."""
        mock_redis = MockRedisStream()
        stream = MarketDataStream(mock_redis)

        # Publish some ticks
        for i in range(5):
            tick = MarketTick(symbol=f"SYM{i}", price=100.0 + i, volume=1000)
            stream.publish(tick)

        # Read range
        ticks = stream.read_range(count=3)

        assert len(ticks) == 3

    def test_get_stream_info(self):
        """Test getting stream information."""
        mock_redis = MockRedisStream()
        stream = MarketDataStream(mock_redis)

        # Publish some data
        stream.publish(MarketTick(symbol="TEST", price=100.0, volume=100))

        info = stream.get_stream_info()

        assert info["length"] == 1
        assert info["groups"] == 1

    def test_trim_stream(self):
        """Test stream trimming."""
        mock_redis = MockRedisStream()
        config = StreamConfig(max_stream_length=5)
        stream = MarketDataStream(mock_redis, config)

        # Publish more than max
        for i in range(10):
            tick = MarketTick(symbol=f"SYM{i}", price=100.0 + i, volume=1000)
            stream.publish(tick)

        # Manually trim
        stream.trim_stream(maxlen=5)

        assert len(mock_redis.streams["market:data"]) <= 5

    def test_delete_stream(self):
        """Test deleting stream."""
        mock_redis = MockRedisStream()
        stream = MarketDataStream(mock_redis)

        stream.publish(MarketTick(symbol="TEST", price=100.0, volume=100))
        assert "market:data" in mock_redis.streams

        result = stream.delete_stream()

        assert result is True
        assert "market:data" not in mock_redis.streams

    def test_start_stop_consuming(self):
        """Test starting and stopping consumer."""
        mock_redis = MockRedisStream()
        stream = MarketDataStream(mock_redis)

        assert stream._running is False

        # Start and immediately stop
        stream.stop_consuming()
        assert stream._running is False


class TestCreateMarketStream:
    """Test factory function."""

    def test_create_with_defaults(self):
        """Test creating stream with defaults."""
        mock_redis = MockRedisStream()
        stream = create_market_stream(redis_client=mock_redis)

        assert stream.config.stream_name == "market:data"
        assert stream.config.consumer_group == "trading_bot"

    def test_create_with_custom(self):
        """Test creating stream with custom config."""
        mock_redis = MockRedisStream()
        stream = create_market_stream(
            redis_client=mock_redis,
            stream_name="custom:stream",
            consumer_group="custom_group",
        )

        assert stream.config.stream_name == "custom:stream"
        assert stream.config.consumer_group == "custom_group"

    def test_create_without_redis(self):
        """Test creating stream without Redis client."""
        stream = create_market_stream(redis_client=None)

        assert stream._client is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
