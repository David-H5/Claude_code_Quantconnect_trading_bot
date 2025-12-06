# Two-Part Spread Strategy

## Strategy Overview

This is the core trading strategy for legging into butterfly and iron condor positions in two separate parts to achieve net-credit or break-even multi-leg options positions.

### Philosophy

Unlike traditional spread trading that avoids wide bid-ask spreads, this strategy **treats wide spreads as opportunities**. Wide spreads indicate:
- Lower liquidity where patient orders can get filled below mid-price
- Market maker pricing inefficiencies
- Potential for price improvement on limit orders

### Strategy Steps

1. **Find Underpriced Debit Spread**
   - Scan for debit spreads that are cheaper than usual
   - Wide bid-ask spreads = opportunity (can get filled below mid)
   - Target fills at 35% of spread width from bid (not mid-price)
   - Factors: Low liquidity, moderate spread width, elevated volatility

2. **Execute Debit Spread**
   - Place limit order below mid-price
   - Quick cancel if not filled within 2-3 seconds
   - Record actual fill location for session statistics

3. **Find Matching Credit Spread**
   - Search for credit spread further out-of-the-money
   - Credit received must be >= debit cost paid
   - Target fills at 65% of spread width from bid (above mid)
   - Result: Net-credit or break-even butterfly/iron condor

4. **Execute Credit Spread**
   - Place limit order for credit spread
   - Quick cancel if not filled
   - Track position balance per option chain

---

## Implementation Guide

### Core Modules

| Module | Purpose |
|--------|---------|
| `execution/two_part_spread.py` | Base strategy logic, fill tracking |
| `execution/arbitrage_executor.py` | Full execution system with optimization |

### Key Classes

```python
from execution import (
    TwoPartSpreadStrategy,      # Basic strategy
    ArbitrageExecutor,          # Full execution system
    create_arbitrage_executor,  # Factory function
    TimingParameters,           # Delay/cancel settings
    SizingParameters,           # Contract sizing
)
```

### Basic Setup

```python
from execution import create_two_part_strategy

strategy = create_two_part_strategy()

# Scan for debit opportunities
opportunities = strategy.scan_debit_opportunities(quotes)

# Find credit matches
if opportunities:
    matches = strategy.find_credit_matches(opportunities[0], credit_quotes)
```

### Full Execution Setup

```python
from execution import create_arbitrage_executor

executor = create_arbitrage_executor(
    order_callback=broker.submit_order,
    quote_callback=broker.get_quotes,
    underlying_callback=broker.get_price,
    min_fill_rate=0.25,          # 25% minimum fill rate
    cancel_after_seconds=2.5,     # Quick cancel
    min_delay=3.0,                # Min delay between attempts
    max_delay=15.0,               # Max delay between attempts
    max_contracts=10              # Max contracts per order
)

# Add symbols with expirations to trade
executor.add_symbol("AAPL", expirations)
executor.add_symbol("SPY", expirations)

# Start concurrent trading across all expirations
executor.start()
```

### Configuration Parameters

#### Timing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cancel_after_seconds` | 2.5 | Cancel unfilled orders after this time |
| `min_delay_between_attempts` | 3.0 | Minimum seconds between order attempts |
| `max_delay_between_attempts` | 15.0 | Maximum seconds between order attempts |
| `random_delay_factor` | 0.3 | Randomization factor for delays |
| `open_auction_delay_multiplier` | 2.0 | Extra caution at market open |
| `close_auction_delay_multiplier` | 1.5 | Extra caution at market close |

#### Sizing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_contracts` | 1 | Minimum contracts per order |
| `max_contracts` | 10 | Maximum contracts per order |
| `current_contracts` | 1 | Starting contract size |
| `target_fill_rate` | 0.50 | Target 50% fill rate |
| `size_increase_threshold` | 0.70 | Increase size if fills > 70% |
| `size_decrease_threshold` | 0.30 | Decrease size if fills < 30% |

---

## Success Metrics

### Fill Rate Tracking

The strategy tracks **actual fill rates** during the trading session, NOT hypothetical factors.

```python
# Get current session fill rate
fill_rate = strategy.get_session_fill_rate()

# Check if should attempt trade (25% minimum)
should_trade, reason = strategy.should_attempt_trade(min_fill_rate=0.25)
```

### Key Metrics to Monitor

| Metric | Target | Description |
|--------|--------|-------------|
| Session Fill Rate | >= 25% | Actual fills / attempts this session |
| Fill Location | Below Mid | Where fills occur relative to bid/ask |
| Net Premium | >= 0 | Credit received minus debit paid |
| Position Balance | 0 | Debit contracts should equal credit contracts |
| Time to Fill | < 2.5s | Successful fills should be quick |

### Fill Location Categories

| Location | Description | Implication |
|----------|-------------|-------------|
| `AT_BID` | Filled at bid price | Excellent - maximum savings |
| `BELOW_MID` | Filled below mid | Good - price improvement |
| `AT_MID` | Filled at mid price | Acceptable |
| `ABOVE_MID` | Filled above mid | Suboptimal |
| `AT_ASK` | Filled at ask price | Poor - no price improvement |

### Autonomous Optimization Metrics

The `ArbitrageExecutor` tracks and optimizes:

1. **Fill Rate by Contract Size** - Which sizes get filled most
2. **Fill Rate by Hour** - Best times of day to trade
3. **Fill Rate by Delay** - Optimal delay between attempts
4. **Time to Fill Distribution** - How long successful fills take

```python
# Get optimization recommendations
recommendations = executor.optimize_parameters()
# Returns: optimal_contracts, optimal_hours, optimal_cancel_ms

# Apply recommendations
executor.apply_optimization(recommendations)
```

---

## Trading Rules

### Order Execution Rules

1. **Quick Cancel**: If order not filled within ~2.5 seconds, cancel immediately
   - Unfilled orders interfere with order book
   - Market makers adjust quotes when they see resting orders
   - If it's not filling quickly, it won't fill at all

2. **Delay Between Attempts**: Random delay 3-15 seconds
   - Prevents triggering market maker algorithms
   - Avoids pattern detection
   - Varies by trading phase (longer at open/close)

3. **Contract Sizing**: Start with 1 contract
   - Highest chance of quick fill
   - Avoid triggering size-based algorithms
   - Scale up only if fill rate > 70%

### Position Balance Rules

1. **Balance Per Chain**: Debit contracts must roughly equal credit contracts
   - Maximum imbalance: 5 contracts
   - Prevents holding excess long positions
   - Protects buying power

2. **Chain Isolation**: Each symbol/expiration tracked separately
   - Imbalance in one chain doesn't affect others
   - Clear accounting per position

### Expiration Selection Rules

1. **Optimal Range**: 30-180 days to expiration
   - Best liquidity and fill rates
   - Sufficient time value for strategy

2. **Acceptable Outside Range**: If meeting criteria:
   - Reasonable daily volume
   - Acceptable bid-ask spreads
   - Recent elevated volatility
   - Successful fill rates in session

3. **Avoid**: < 7 days to expiration
   - Too illiquid
   - Gamma risk too high

---

## Key Observations from Live Trading

### What Works

1. **Wide spreads ARE opportunities**
   - Can get filled 35% from bid instead of mid
   - Patient limit orders get price improvement

2. **1 contract at a time**
   - Highest fill probability
   - Avoids detection by algorithms
   - Trade-off: Lower daily volume

3. **Quick cancels (2-3 seconds)**
   - If not filled quickly, won't fill at all
   - Resting orders attract adverse selection
   - Clean order book = better subsequent fills

4. **Random delays**
   - Market makers detect patterns
   - Randomization avoids triggering adjustments

### What Doesn't Work

1. **Waiting for fills** - Orders that don't fill in seconds won't fill
2. **Large size orders** - Alert algorithms, get worse fills
3. **Predictable timing** - MMs adjust to patterns
4. **Ignoring balance** - Holding unmatched positions = risk

---

## Integration Points

### With LLM Analysis

The strategy provides LLM-ready summaries:

```python
summary = executor.get_llm_summary()
# Returns formatted text with:
# - Active trader count
# - Net premium
# - Position imbalance
# - Performance by contract size
# - Performance by hour
# - Top performing expirations
```

### With Risk Manager

```python
from models import RiskManager, RiskLimits

limits = RiskLimits(
    max_position_size=0.25,
    max_daily_loss=0.03,
    max_drawdown=0.10,
)

# Check before executing
if risk_manager.can_open_position(position_value):
    executor.start()
```

### With Circuit Breaker

```python
from models import create_circuit_breaker

breaker = create_circuit_breaker()

# Stop execution on trip
if breaker.is_tripped():
    executor.stop()
```

---

## Files Reference

| File | Description |
|------|-------------|
| `execution/two_part_spread.py` | Core strategy classes |
| `execution/arbitrage_executor.py` | Full execution system |
| `execution/fill_predictor.py` | Fill rate prediction (supplementary) |
| `execution/spread_analysis.py` | Spread quality analysis |
| `execution/spread_anomaly.py` | Market maker manipulation detection |
| `models/enhanced_volatility.py` | IV Rank/Percentile for opportunity detection |

---

## Changelog

| Date | Change |
|------|--------|
| 2024-11 | Initial strategy documentation |
| 2024-11 | Added arbitrage executor with concurrent trading |
| 2024-11 | Added autonomous optimization system |
| 2024-11 | Added position balancing per option chain |
