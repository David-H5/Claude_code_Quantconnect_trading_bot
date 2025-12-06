# Arbitrage Executor System

## Overview

The Arbitrage Executor is the autonomous trading engine that implements the two-part spread strategy across multiple symbols and expirations concurrently. It handles order execution, timing optimization, position balancing, and autonomous parameter learning.

---

## Architecture

```
ArbitrageExecutor (Main Orchestrator)
├── PositionBalancer (Balance tracking per chain)
├── CreditMaximizer (Find best credit spreads)
├── OptimizationMetrics (Learning system)
└── ExpirationTrader[] (One per symbol/expiration)
    ├── TimingParameters (Delays, cancels)
    ├── SizingParameters (Contract sizing)
    └── Trade execution loop
```

---

## Implementation Guide

### Quick Start

```python
from execution import create_arbitrage_executor

# Define broker callbacks
def submit_order(symbol, legs, quantity, limit_price, expiration):
    """Submit order to broker, return order_id."""
    return broker.place_spread_order(symbol, legs, quantity, limit_price)

def get_quotes(symbol, expiration):
    """Get current option quotes for symbol/expiration."""
    return broker.get_option_chain(symbol, expiration)

def get_price(symbol):
    """Get current underlying price."""
    return broker.get_quote(symbol).last

# Create executor
executor = create_arbitrage_executor(
    order_callback=submit_order,
    quote_callback=get_quotes,
    underlying_callback=get_price,
    min_fill_rate=0.25,
    cancel_after_seconds=2.5,
    min_delay=3.0,
    max_delay=15.0,
    max_contracts=10
)

# Add symbols to trade
from datetime import datetime, timedelta

expirations = [
    datetime.now() + timedelta(days=30),
    datetime.now() + timedelta(days=60),
    datetime.now() + timedelta(days=90),
]

executor.add_symbol("AAPL", expirations)
executor.add_symbol("MSFT", expirations)
executor.add_symbol("SPY", expirations)

# Start trading
executor.start()

# Monitor status
status = executor.get_status()
print(f"Active traders: {len(status['traders'])}")
print(f"Net premium: ${status['position_summary']['net_premium']:.2f}")

# Stop when done
executor.stop()
```

### Advanced Configuration

```python
from execution import (
    ArbitrageExecutor,
    TimingParameters,
    SizingParameters,
    PositionBalancer,
    CreditMaximizer,
)

# Custom timing - very conservative
timing = TimingParameters(
    cancel_after_seconds=2.0,        # Quick cancel
    min_delay_between_attempts=5.0,  # Longer min delay
    max_delay_between_attempts=20.0, # Longer max delay
    random_delay_factor=0.4,         # More randomization
    open_auction_delay_multiplier=3.0,  # Very cautious at open
    close_auction_delay_multiplier=2.0,
)

# Custom sizing - aggressive
sizing = SizingParameters(
    min_contracts=1,
    max_contracts=20,
    current_contracts=1,
    target_fill_rate=0.40,           # Lower target
    size_increase_threshold=0.60,    # Increase earlier
    size_decrease_threshold=0.20,    # Decrease later
)

# Custom position balancing
balancer = PositionBalancer(
    max_imbalance=3,                 # Tighter balance requirement
    force_balance_at_close=True,
)

# Custom credit maximization
credit_max = CreditMaximizer(
    min_credit_threshold=0.05,       # Minimum $0.05 net credit
    max_spread_width=15,             # Allow wider spreads
    prefer_atm=True,                 # Prefer ATM for liquidity
)

# Build executor manually
executor = ArbitrageExecutor(
    order_callback=submit_order,
    quote_callback=get_quotes,
    underlying_callback=get_price,
    timing=timing,
    sizing=sizing,
    min_fill_rate=0.25,
    optimal_expiry_range=(30, 180),
)

# Override components
executor.balancer = balancer
executor.credit_maximizer = credit_max
```

---

## Components

### PositionBalancer

Tracks position balance per option chain to ensure debit and credit contracts stay matched.

```python
from execution import PositionBalancer

balancer = PositionBalancer(max_imbalance=5)

# Check if can add positions
can_add = balancer.can_add_debit("AAPL", expiration, contracts=2)

# Record fills
balancer.record_debit_fill("AAPL", expiration, contracts=2, cost=1.50)
balancer.record_credit_fill("AAPL", expiration, contracts=2, credit=1.60)

# Get balance status
balance = balancer.get_balance("AAPL", expiration)
print(f"Net contracts: {balance.net_contracts}")
print(f"Net premium: ${balance.net_premium:.2f}")

# Get all imbalanced chains
imbalanced = balancer.get_all_imbalanced()

# Summary
summary = balancer.get_summary()
# {chains, total_debit_paid, total_credit_received, net_premium, total_imbalance}
```

### CreditMaximizer

Finds the credit spread that provides maximum credit to offset debit cost.

```python
from execution import CreditMaximizer, SpreadLeg

maximizer = CreditMaximizer(
    min_credit_threshold=0.0,   # Accept break-even
    max_spread_width=10,        # Max strike width
    prefer_atm=True,            # Prefer liquid ATM options
)

# Find best credit spread for a debit opportunity
debit_legs = (debit_long_leg, debit_short_leg)
result = maximizer.find_best_credit_spread(
    debit_opportunity=debit_legs,
    available_options=option_chain,
    underlying_price=150.00
)

if result:
    credit_long, credit_short, credit_amount = result
    print(f"Best credit: ${credit_amount:.2f}")
```

### OptimizationMetrics

Collects data for autonomous optimization of trading parameters.

```python
from execution import OptimizationMetrics

metrics = OptimizationMetrics()

# Record attempt results
metrics.record_attempt(
    contracts=2,
    hour=10,
    delay_category="medium",
    filled=True,
    fill_time_ms=1500,
    profit=0.10
)

# Get optimal parameters
optimal_size = metrics.get_optimal_size()
optimal_hours = metrics.get_optimal_hours()
optimal_cancel_ms = metrics.get_optimal_cancel_time_ms()
```

### ExpirationTrader

Handles trading for a single expiration date with its own trading loop.

```python
from execution import ExpirationTrader

trader = ExpirationTrader(
    symbol="AAPL",
    expiration=expiration,
    timing=timing,
    sizing=sizing,
    balancer=balancer,
    credit_maximizer=maximizer,
    order_callback=submit_order,
    quote_callback=get_quotes,
)

# Properties
print(f"Days to expiry: {trader.days_to_expiry}")
print(f"Category: {trader.expiration_category}")  # WEEKLY, MONTHLY, etc.
print(f"Is optimal: {trader.is_optimal_expiration}")

# Get fill rate
fill_rate = trader.get_fill_rate()

# Scan for opportunities
opportunities = trader.scan_opportunities(options, underlying_price)
```

---

## Trading Phases

The executor adjusts behavior based on market trading phase:

| Phase | Time (ET) | Delay Multiplier | Description |
|-------|-----------|------------------|-------------|
| PRE_MARKET | < 9:30 | N/A | No trading |
| OPEN_AUCTION | 9:30-9:45 | 2.0x | High caution, longer delays |
| MORNING | 9:45-11:30 | 1.0x | Normal trading |
| MIDDAY | 11:30-14:00 | 1.0x | Normal trading |
| AFTERNOON | 14:00-15:45 | 1.0x | Normal trading |
| CLOSE_AUCTION | 15:45-16:00 | 1.5x | Increased caution |
| AFTER_HOURS | > 16:00 | N/A | No trading |

---

## Autonomous Optimization

The system learns optimal parameters from actual trading results.

### Data Collection

```python
# The executor automatically collects:
# - Fill rate by contract size
# - Fill rate by hour of day
# - Fill rate by delay category (short/medium/long)
# - Time to fill for successful orders
```

### Optimization Cycle

```python
# After sufficient data (e.g., end of day)
recommendations = executor.optimize_parameters()

# Returns:
# {
#     "optimal_contracts": 2,
#     "optimal_hours": [10, 11, 14, 15],
#     "optimal_cancel_ms": 2200
# }

# Apply recommendations
executor.apply_optimization(recommendations)
```

### Learning Logic

**Contract Size Optimization:**
- Calculates fill_rate * size for each contract size
- Maximizes total filled volume
- Requires 10+ samples per size for reliability

**Hour Optimization:**
- Filters hours with < 25% fill rate
- Ranks remaining by fill rate
- Requires 5+ samples per hour

**Cancel Time Optimization:**
- Uses 95th percentile of successful fill times
- Adds 500ms buffer
- Caps at 5000ms maximum

---

## Success Metrics

### Real-Time Monitoring

```python
status = executor.get_status()

# Overall status
print(f"Active: {status['active']}")
print(f"Total traders: {len(status['traders'])}")

# Position summary
pos = status['position_summary']
print(f"Net premium: ${pos['net_premium']:.2f}")
print(f"Total imbalance: {pos['total_imbalance']} contracts")
print(f"Balanced chains: {pos['balanced_chains']}/{pos['chains']}")

# Per-trader metrics
for key, trader in status['traders'].items():
    print(f"\n{trader['symbol']} {trader['expiration'][:10]}:")
    print(f"  Fill rate: {trader['fill_rate']:.1%}")
    print(f"  Attempts: {trader['session_attempts']}")
    print(f"  Fills: {trader['session_fills']}")
    print(f"  Contracts: {trader['current_contracts']}")
```

### LLM Summary

```python
summary = executor.get_llm_summary()
# Returns formatted text for LLM analysis
```

Example output:
```
Arbitrage Executor Status:

Active Traders: 9
Total Net Premium: $127.50
Position Imbalance: 3 contracts
Balanced Chains: 7/9

Performance by Contract Size:
  1 contracts: 45.2% fill rate
  2 contracts: 38.1% fill rate
  3 contracts: 22.4% fill rate

Performance by Hour:
  10:00: 52.3% fill rate
  11:00: 48.1% fill rate
  14:00: 44.2% fill rate
  15:00: 41.8% fill rate

Top Performing Expirations:
  AAPL 2024-02-16: 48.5% (32/66)
  SPY 2024-03-15: 45.2% (28/62)
  MSFT 2024-02-16: 42.1% (24/57)
```

---

## Error Handling

### Order Failures

```python
# Orders can fail with these results:
# - FILLED: Success
# - PARTIAL: Partial fill (rare)
# - CANCELLED: Manually cancelled
# - TIMEOUT: Not filled before cancel time
# - REJECTED: Broker rejected order

# The executor automatically:
# - Counts timeouts toward fill rate
# - Skips rejected orders
# - Handles partial fills
```

### Fill Rate Threshold

```python
# If fill rate drops below minimum (25% default):
# - Trader pauses for 60 seconds
# - Then retries
# - Continues until rate improves or day ends
```

### Thread Safety

```python
# The executor uses threading.Lock for:
# - Position balance updates
# - Trader state changes
# - Metrics recording

# Each ExpirationTrader also has its own lock
```

---

## Files Reference

| File | Classes |
|------|---------|
| `execution/arbitrage_executor.py` | ArbitrageExecutor, ExpirationTrader, PositionBalancer, CreditMaximizer, OptimizationMetrics, TimingParameters, SizingParameters |
| `execution/two_part_spread.py` | TwoPartSpreadStrategy (simpler version) |

---

## Changelog

| Date | Change |
|------|--------|
| 2024-11 | Initial implementation |
| 2024-11 | Added concurrent expiration trading |
| 2024-11 | Added autonomous optimization |
| 2024-11 | Added position balancing |
