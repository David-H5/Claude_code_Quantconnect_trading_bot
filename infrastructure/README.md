# Infrastructure Module

**UPGRADE-015 Phase 7: Redis Infrastructure**

This module provides Redis-based infrastructure components for the trading system, including caching, streaming, time series storage, and pub/sub messaging.

## Components

### 1. Redis Client (`redis_client.py`)

Core Redis client with connection pooling and trading-specific utilities.

```python
from infrastructure import RedisClient, create_redis_client

# Create client
client = create_redis_client(
    host="localhost",
    port=6379,
    password="optional",
)

# Basic operations
client.set("key", "value", ex=300)  # 5 min TTL
value = client.get("key")

# JSON operations
client.set_json("config", {"setting": "value"})
config = client.get_json("config")

# Trading-specific
client.cache_quote("AAPL", {"bid": 175.50, "ask": 175.55})
quote = client.get_cached_quote("AAPL")

# Rate limiting
allowed, remaining = client.set_rate_limit("api_calls", max=100, window=60)

# Distributed locks
lock = client.acquire_lock("order_processing")
try:
    # Critical section
    pass
finally:
    client.release_lock(lock)
```

### 2. Market Data Stream (`market_stream.py`)

Real-time market data streaming using Redis Streams.

```python
from infrastructure import MarketDataStream, create_market_stream

# Create stream
stream = create_market_stream(
    redis_client=client._client,
    stream_name="market:data",
)

# Publishing
stream.publish_quote("AAPL", price=175.50, bid=175.45, ask=175.55)

# Batch publishing
from infrastructure.market_stream import MarketTick
ticks = [
    MarketTick(symbol="AAPL", price=175.50, volume=1000),
    MarketTick(symbol="MSFT", price=300.00, volume=2000),
]
stream.publish_batch(ticks)

# Subscribing
def on_tick(tick):
    print(f"{tick.symbol}: {tick.price}")

stream.subscribe(on_tick)
stream.start_consuming()
```

### 3. Time Series Store (`timeseries.py`)

Time-series data storage for OHLCV data, indicators, and metrics.

```python
from infrastructure import TimeSeriesStore, create_timeseries_store

# Create store
ts = create_timeseries_store(redis_client=client._client)

# Add data points
ts.add("metrics:latency", 45.5)
ts.add("metrics:latency", 52.3)

# Add OHLCV bars
from infrastructure.timeseries import OHLCVBar
bar = OHLCVBar(
    timestamp=datetime.utcnow(),
    open=175.00,
    high=176.50,
    low=174.50,
    close=175.25,
    volume=1000000,
)
ts.add_ohlcv("AAPL", bar, timeframe="1m")

# Query data
from datetime import timedelta
start = datetime.utcnow() - timedelta(hours=1)
points = ts.get_range("metrics:latency", start=start)
bars = ts.get_ohlcv("AAPL", timeframe="1m", limit=60)

# Statistics
stats = ts.get_stats("metrics:latency")
print(f"Avg: {stats['avg']}, Max: {stats['max']}")
```

### 4. Pub/Sub Manager (`pubsub.py`)

Publish/subscribe messaging for trading events and alerts.

```python
from infrastructure import PubSubManager, create_pubsub_manager

# Create manager
pubsub = create_pubsub_manager(redis_client=client._client)

# Publishing
pubsub.publish_trade_signal("AAPL", "BUY", confidence=0.85)
pubsub.publish_order_update("ORD001", "FILLED", "AAPL")
pubsub.publish_alert("risk", "Daily loss limit approaching", severity="warning")

# Subscribing
def on_signal(message):
    print(f"Signal: {message.data['symbol']} - {message.data['signal']}")

pubsub.on_trade_signal(on_signal)
pubsub.start_listening()

# Cleanup
pubsub.close()
```

## Docker Deployment

Use `docker-compose.redis.yml` to run Redis:

```bash
# Start Redis
docker-compose -f docker-compose.redis.yml up -d

# Start with web UI (Redis Commander)
docker-compose -f docker-compose.redis.yml --profile ui up -d

# Stop
docker-compose -f docker-compose.redis.yml down
```

Access Redis Commander at: http://localhost:8081

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | localhost | Redis server hostname |
| `REDIS_PORT` | 6379 | Redis server port |
| `REDIS_PASSWORD` | None | Redis password |

### Redis Configuration

The `redis.conf` file configures:

- Memory limit: 256MB
- Eviction policy: allkeys-lru
- Persistence: RDB snapshots
- Stream node limits for market data

## Standard Channels

| Channel | Purpose | Message Type |
|---------|---------|--------------|
| `trading:signals` | Trade signals | `TRADE_SIGNAL` |
| `trading:orders` | Order updates | `ORDER_UPDATE` |
| `trading:positions` | Position changes | `POSITION_UPDATE` |
| `trading:alerts` | System alerts | `PRICE_ALERT`, `RISK_ALERT` |
| `trading:system` | System events | `SYSTEM_EVENT`, `HEARTBEAT` |

## Testing

```bash
# Run infrastructure tests
pytest tests/infrastructure/ -v

# Run with coverage
pytest tests/infrastructure/ --cov=infrastructure
```

## Graceful Degradation

All components gracefully degrade if Redis is unavailable:

```python
from infrastructure import RedisClient

client = RedisClient()
if client.ping():
    # Redis available - use caching
    pass
else:
    # Redis unavailable - fallback logic
    pass
```

## Dependencies

- `redis>=5.0.0` - Redis client
- `hiredis>=2.3.0` - High-performance parser (optional but recommended)

Install with:

```bash
pip install redis hiredis
```
