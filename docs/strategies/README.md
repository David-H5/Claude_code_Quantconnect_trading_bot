# Trading Strategies Documentation

This directory contains detailed documentation for all trading strategies implemented in this project.

## Strategy Index

| Strategy | Type | Status | Documentation |
|----------|------|--------|---------------|
| [Two-Part Spread](TWO_PART_SPREAD_STRATEGY.md) | Options Arbitrage | Active | Full |
| [Arbitrage Executor](ARBITRAGE_EXECUTOR.md) | Execution System | Active | Full |
| [Wheel Strategy](../algorithms/WHEEL_STRATEGY.md) | Options Income | Reference | Basic |

---

## Core Strategy: Two-Part Spread

The primary trading strategy for this project. See [TWO_PART_SPREAD_STRATEGY.md](TWO_PART_SPREAD_STRATEGY.md) for full details.

### Quick Summary

**Goal:** Leg into net-credit or break-even butterflies/iron condors in two parts.

**Process:**
1. Find underpriced debit spread (wide spreads = opportunity)
2. Execute debit spread below mid-price with quick cancel
3. Find credit spread further OTM with credit >= debit cost
4. Execute credit spread to complete position
5. Repeat with delays to avoid market maker detection

**Key Insights:**
- Wide bid-ask spreads are opportunities, not risks
- Orders that don't fill in 2-3 seconds won't fill at all
- 1 contract at a time offers best fill rate
- Random delays prevent pattern detection
- Position balance per chain is critical

---

## Implementation Components

### Execution Layer

| Module | Purpose |
|--------|---------|
| `execution/arbitrage_executor.py` | Full autonomous execution system |
| `execution/two_part_spread.py` | Core strategy logic |
| `execution/smart_execution.py` | Order management |
| `execution/fill_predictor.py` | Fill rate tracking |
| `execution/spread_analysis.py` | Spread quality analysis |
| `execution/spread_anomaly.py` | MM manipulation detection |

### Analysis Layer

| Module | Purpose |
|--------|---------|
| `models/enhanced_volatility.py` | IV Rank, IV Percentile, regimes |
| `models/volatility_surface.py` | Vol surface analysis |
| `models/pnl_attribution.py` | P&L breakdown by Greeks |
| `models/portfolio_hedging.py` | Delta/gamma hedging |

### Risk Layer

| Module | Purpose |
|--------|---------|
| `models/risk_manager.py` | Position limits, drawdown |
| `models/circuit_breaker.py` | Safety cutoff |
| `models/multi_leg_strategy.py` | Multi-leg Greeks |

---

## User-Provided Trading Insights

These insights come directly from the strategy developer's live trading experience:

### Fill Rate Observations

1. **Quick fills or no fills**: If an order doesn't fill within 2-3 seconds, it won't fill at all
2. **Cancel quickly**: Unfilled orders interfere with order book and alert market makers
3. **Actual fill rates matter**: Track real session fills, not hypothetical factors

### Sizing Observations

1. **1 contract = highest fill rate**: But low daily volume
2. **Larger sizes alert algorithms**: Market makers detect and adjust
3. **Need to experiment**: Find balance between fill rate and volume

### Timing Observations

1. **Random delays essential**: Predictable patterns get exploited
2. **Optimal expirations**: 30-180 days offers best liquidity
3. **Other expirations**: Can work with good volume/spreads/volatility

### Market Maker Behavior

1. **Wide spreads = opportunity**: Not a signal to avoid
2. **Algorithms detect patterns**: Must randomize timing and sizing
3. **Price improvement possible**: Patient orders get filled below mid

---

## Strategy Parameters Summary

### Timing Defaults

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Cancel after | 2.5s | Orders won't fill after this |
| Min delay | 3.0s | Avoid detection |
| Max delay | 15.0s | Avoid detection |
| Random factor | 30% | Add unpredictability |

### Sizing Defaults

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Starting size | 1 | Highest fill probability |
| Max size | 10 | Avoid large-order detection |
| Increase threshold | 70% | Scale up when working |
| Decrease threshold | 30% | Scale down when not |

### Fill Rate Thresholds

| Threshold | Value | Action |
|-----------|-------|--------|
| Minimum | 25% | Below = pause trading |
| Target | 50% | Optimal operation |
| Excellent | 70% | Can increase size |

---

## Adding New Strategies

When adding a new strategy:

1. Create `docs/strategies/STRATEGY_NAME.md`
2. Include sections:
   - Strategy Overview
   - Implementation Guide
   - Success Metrics
   - Trading Rules
   - Key Observations
   - Files Reference
   - Changelog

3. Update this index file
4. Update `CLAUDE.md` if special handling needed

---

## Documentation Maintenance

These documentation files are automatically maintained by Claude Code based on:
- User-provided trading insights
- Implementation changes
- Performance observations
- New features

See `CLAUDE.md` for instructions on how to provide updates.

---

## Changelog

| Date | Change |
|------|--------|
| 2024-11 | Initial documentation structure |
| 2024-11 | Added Two-Part Spread strategy docs |
| 2024-11 | Added Arbitrage Executor docs |
