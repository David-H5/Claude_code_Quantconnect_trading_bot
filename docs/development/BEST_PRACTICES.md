# Trading Bot Development Best Practices

This document establishes permanent guidelines for developing, testing, and deploying trading algorithms in this project. All contributors must follow these practices.

---

## Table of Contents

1. [Safety-First Development](#1-safety-first-development)
2. [Risk Management](#2-risk-management)
3. [Backtesting Standards](#3-backtesting-standards)
4. [Live Trading Safety](#4-live-trading-safety)
5. [Code Quality Standards](#5-code-quality-standards)
6. [Options Trading Specifics](#6-options-trading-specifics)
7. [Pre-Deployment Checklist](#7-pre-deployment-checklist)
8. [Common Pitfalls](#8-common-pitfalls)

---

## 1. Safety-First Development

### Core Principle

**Never deploy untested code to live trading.** All algorithms must pass through the complete testing pipeline before any live deployment.

### Mandatory Testing Pipeline

```
Local Tests → Validation → Backtest → Paper Trading → Live
```

| Stage | Requirements | Duration |
|-------|--------------|----------|
| Local Tests | All unit tests pass, 70%+ coverage | Immediate |
| Validation | Algorithm validator passes | Immediate |
| Cloud Backtest | Sharpe > 1.0, Drawdown < 20% | 1-2 hours |
| Paper Trading | No errors, fills match backtest | 2+ weeks |
| Live (Phase 1) | Micro capital (1% of intended) | 1 week |
| Live (Phase 2) | Small capital (10% of intended) | 2 weeks |
| Live (Full) | Full allocation | Ongoing |

### Before Making Any Code Changes

```python
from scripts.backup_manager import create_pre_change_backup
create_pre_change_backup("algorithms/my_algo.py", "description of change")
```

---

## 2. Risk Management

### Required Risk Limits

Every algorithm MUST implement these limits:

| Limit | Recommended Value | Purpose |
|-------|-------------------|---------|
| Max Position Size | 10-25% | Prevent single-stock concentration |
| Max Daily Loss | 2-3% | Stop trading after significant loss |
| Max Drawdown | 10-15% | Halt before catastrophic loss |
| Max Risk Per Trade | 1-2% | Limit damage from single trade |
| Max Consecutive Losses | 5-7 | Detect strategy failure |

### Implementation Pattern

```python
from models import RiskManager, RiskLimits

limits = RiskLimits(
    max_position_size=0.25,
    max_daily_loss=0.03,
    max_drawdown=0.10,
    max_risk_per_trade=0.02,
    max_consecutive_losses=5,
)

risk_manager = RiskManager(starting_equity=100000, limits=limits)

# In trading loop - ALWAYS check before trading
if not risk_manager.check_can_trade():
    return
```

### Circuit Breaker Requirements

```python
from models.circuit_breaker import TradingCircuitBreaker, create_circuit_breaker

breaker = create_circuit_breaker(
    max_daily_loss=0.03,
    max_drawdown=0.10,
    max_consecutive_losses=5,
    require_human_reset=True,  # ALWAYS True for live trading
)

# In trading loop
if not breaker.can_trade():
    self.Liquidate()  # Close all positions
    return

# After each trade
breaker.record_trade_result(is_winner=trade.profitable)
breaker.check_daily_loss(portfolio.daily_pnl_pct)
breaker.check_drawdown(current_equity, peak_equity)
```

### Stop Loss Requirements

- **EVERY position must have a stop loss**
- Calculate stops based on risk per trade, not arbitrary percentages
- Place stops immediately upon entry
- For options: use both profit-taking AND stop losses

### Graduated Profit-Taking

```python
PROFIT_THRESHOLDS = [
    {"gain_pct": 1.00, "sell_pct": 0.50},  # 50% at +100%
    {"gain_pct": 2.00, "sell_pct": 0.25},  # 25% at +200%
    {"gain_pct": 4.00, "sell_pct": 0.15},  # 15% at +400%
    {"gain_pct": 10.00, "sell_pct": 1.00}, # Rest at +1000%
]
```

---

## 3. Backtesting Standards

### Avoiding Look-Ahead Bias

**NEVER use future data in current decisions.**

```python
# WRONG - Look-ahead bias
def on_data(self, data):
    future_prices = self.History(self.symbol, 5, Resolution.Daily)[-1]  # FUTURE!
    if future_prices > current_price:
        self.Buy()

# CORRECT - Only use past data
def on_data(self, data):
    if not data.ContainsKey(self.symbol):
        return
    current_price = data[self.symbol].Close
    past_prices = self.History(self.symbol, 20, Resolution.Daily)
    if len(past_prices) < 20:
        return
    sma = past_prices['Close'].mean()
    if current_price > sma:
        self.Buy()
```

### Look-Ahead Prevention Checklist

- [ ] Never access data timestamped in the future
- [ ] Never access indicators requiring future data
- [ ] Use only completed bars from history
- [ ] Verify all lookback windows have sufficient history
- [ ] For options: use volatility data from same timestamp

### Walk-Forward Analysis

```
Training Period: 2 years
Testing Period: 6 months
Step Size: 3 months
Reoptimization: Every 3 months

Example:
- Train 2018-2019 → Test 2020 Q1
- Train 2018-2020 Q1 → Test 2020 Q2
- Continue quarterly...
```

### Out-of-Sample Testing Protocol

| Data Split | Usage |
|------------|-------|
| 60% | Training/optimization |
| 20% | Validation/tuning |
| 20% | Out-of-sample (NEVER optimize on this) |

### Required Performance Metrics

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Sharpe Ratio | > 1.0 | > 1.5 | > 2.0 |
| Sortino Ratio | > 1.0 | > 1.5 | > 3.0 |
| Max Drawdown | < 25% | < 15% | < 10% |
| Win Rate | > 40% | > 50% | > 60% |
| Profit Factor | > 1.2 | > 1.5 | > 2.0 |

---

## 4. Live Trading Safety

### Paper Trading Validation (Minimum 2 Weeks)

Before moving to live:

- [ ] Order submission/cancellation works correctly
- [ ] Position management executes as expected
- [ ] Risk limits trigger at proper thresholds
- [ ] Profit-taking executes on schedule
- [ ] Circuit breaker halts when expected
- [ ] Fill rates within 10% of backtest
- [ ] No system errors for 2 weeks

### Gradual Deployment Strategy

| Phase | Capital | Position Limit | Daily Loss Limit | Duration |
|-------|---------|----------------|------------------|----------|
| Micro | 1% | 0.5% | 0.5% | 1 week |
| Small | 10% | 2% | 1% | 2 weeks |
| Full | 100% | Full limits | 3% | Ongoing |

### Monitoring Requirements

**Real-Time Dashboard Must Show:**
- Current equity and peak equity
- Daily P&L and weekly P&L
- Active positions and unrealized P&L
- Order queue and recent fills
- Circuit breaker state
- Risk utilization vs limits

**Critical Alerts (Immediate Notification):**
1. Circuit breaker trip
2. Order rejection or system error
3. Drawdown > 75% of limit
4. Daily loss > 75% of limit
5. Unusual slippage (> 2x expected)
6. Data freshness issue (> 5 min lag)

### Failsafe Pattern

```python
try:
    if not breaker.can_trade():
        self.Liquidate()
        return

    # Main trading logic

except Exception as e:
    self.Log(f"ERROR: {e}")
    self.Liquidate()  # ALWAYS close on error
    breaker.halt_all_trading(f"System error: {e}")
```

---

## 5. Code Quality Standards

### Deterministic Execution

- Same input must produce same output
- Use `random.seed()` if randomness needed
- Use UTC timestamps, never local timezone
- Cache external data (don't re-fetch same data)
- Version dependencies explicitly

### Idempotent Operations

All critical operations must be safe to call multiple times:

```python
def close_position(self, symbol):
    """Idempotent: safe to call multiple times."""
    if self.Portfolio[symbol].Invested:
        self.Liquidate(symbol)
    # If already closed, does nothing

def record_trade(self, trade):
    """Idempotent: safe to record same trade twice."""
    if trade.id in self.recorded_trades:
        return  # Already recorded
    self.recorded_trades.add(trade.id)
    self.update_metrics(trade)
```

### State Management

```python
class AlgorithmState:
    """Always track state explicitly."""

    def __init__(self):
        self.positions = {}
        self.pending_orders = {}
        self.risk_state = RiskState()
        self.circuit_breaker_state = CircuitBreakerState()

    def checkpoint(self):
        """Save state for recovery."""
        return json.dumps(self.to_dict())

    def restore(self, checkpoint):
        """Restore from checkpoint."""
        self.from_dict(json.loads(checkpoint))
```

### Logging Requirements

Every algorithm MUST log:

```python
# Orders
self.Log(f"ORDER_SUBMITTED: {symbol} {side} {qty} @ ${price}")
self.Log(f"ORDER_FILLED: {order_id} @ ${fill_price}")
self.Log(f"ORDER_CANCELLED: {order_id} - {reason}")

# Risk
self.Log(f"RISK_CHECK: daily_loss={daily_loss:.2%} limit={limit:.2%}")
self.Log(f"RISK_ACTION: {action} - {reason}")

# Circuit Breaker
self.Log(f"CIRCUIT_BREAKER_TRIP: {reason}")
self.Log(f"CIRCUIT_BREAKER_RESET: by={user}")

# State
self.Log(f"STATE: positions={len(positions)} exposure={exposure:.0%}")
```

### Data Validation

**ALWAYS validate before using:**

```python
def on_data(self, data):
    # Check data exists
    if not data.ContainsKey(self.symbol):
        return

    # Check warmup complete
    if self.IsWarmingUp:
        return

    # Check indicators ready
    if not self.indicator.IsReady:
        return

    # Now safe to trade
    bar = data[self.symbol]
    # ...
```

---

## 6. Options Trading Specifics

### IV-Based Greeks (LEAN PR #6720)

**CRITICAL UPDATE**: As of LEAN PR #6720, Greeks are now calculated using implied volatility.

**Key Changes:**

- **NO warmup period required** for Greeks calculations
- Greeks values match Interactive Brokers and major brokerages
- Uses implied volatility from option prices directly
- Available immediately upon contract data arrival

**Default Pricing Models:**

- European options: Black-Scholes model
- American options: Bjerksund-Stensland model

**New Properties:**

```python
# Immediate access to IV-based Greeks (no warmup needed)
delta = contract.Greeks.Delta
gamma = contract.Greeks.Gamma
vega = contract.Greeks.Vega
theta = contract.Greeks.Theta
rho = contract.Greeks.Rho
theta_per_day = contract.Greeks.ThetaPerDay  # Daily theta decay

# Implied volatility directly from market prices
iv = contract.ImpliedVolatility
```

### Greeks Limitations

- Delta ≠ probability of ITM at expiration (approximation only)
- Greeks assume flat volatility surface (real surface has smile/skew)
- Theta varies by contract and model
- ThetaPerDay = Theta / 365 (annual theta divided by days)
- Greeks are recalculated on every price update using current IV

### IV Surface Considerations

```python
# Don't assume flat IV
# Real surface has smile/skew patterns

iv_percentile = calculate_iv_percentile(symbol, current_iv)

if iv_percentile < 0.30:  # Low IV
    prefer_selling = True   # Sell premium
elif iv_percentile > 0.70:  # High IV
    prefer_buying = True    # Buy premium
```

### Spread Execution Parameters

| Parameter | Recommended Value |
|-----------|-------------------|
| Cancel unfilled orders | 2-3 seconds |
| Min delay between attempts | 3 seconds |
| Max delay between attempts | 15 seconds |
| Min fill rate threshold | 25% |
| Starting contract size | 1 contract |

### Multi-Leg Strategies with ComboOrders

**CRITICAL**: Use ComboOrders for atomic execution of multi-leg strategies.

LEAN includes 24 dedicated files for multi-leg strategy matching:

- Automatic strategy detection (butterflies, condors, iron condors, spreads)
- Atomic order execution (all legs fill together or none fill)
- Single commission calculation per combo
- Support for complex multi-leg positions

**Available ComboOrder Types:**

```python
# ComboMarket - Execute at market prices
legs = [
    Leg.Create(call1_symbol, 1),
    Leg.Create(call2_symbol, -1),
]
self.ComboMarketOrder(legs, quantity=1)

# ComboLimit - Execute at net limit price
self.ComboLimitOrder(legs, quantity=1, limit_price=net_debit)

# ComboLegLimit - Individual leg limits
leg_limits = {call1_symbol: 2.50, call2_symbol: 1.75}
self.ComboLegLimitOrder(legs, quantity=1, leg_limits=leg_limits)
```

**Butterfly Example:**

```python
# Long call butterfly: Buy 1 lower, Sell 2 middle, Buy 1 upper
atm_strike = self.get_atm_strike(underlying_price)
legs = [
    Leg.Create(self.get_call(atm_strike - 5), 1),   # Buy lower
    Leg.Create(self.get_call(atm_strike), -2),      # Sell ATM
    Leg.Create(self.get_call(atm_strike + 5), 1),   # Buy upper
]
self.ComboLimitOrder(legs, quantity=1, limit_price=net_debit)
```

### Charles Schwab Brokerage Specifics

**CRITICAL LIMITATION**: Charles Schwab allows **ONLY ONE algorithm per account**.

**Important Notes:**

- Deploying a second algorithm automatically stops the first
- All trading strategies must be combined into a single algorithm
- Cannot run separate algorithms for different strategies simultaneously
- Schwab integration is built into LEAN core (not a separate plugin)
- OAuth re-authentication required approximately weekly
- Uses QuantConnect cloud OAuth infrastructure (not direct Schwab API)

**Deployment Strategy:**

```python
# WRONG - Multiple algorithms (second will stop first)
# Algorithm 1: Options scanner
# Algorithm 2: Equity momentum

# CORRECT - Combined into single algorithm
class UnifiedTradingBot(QCAlgorithm):
    def Initialize(self):
        self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)

        # Combine all strategies
        self.options_strategy = OptionsScanner()
        self.equity_strategy = EquityMomentum()
        # Both run in same algorithm
```

### Assignment Risk

- **Long options**: No assignment risk
- **Short options**: Can be assigned anytime when ITM
- Assume 60-90% of deep ITM options will be assigned
- Monitor dividend dates for early assignment risk

---

## 7. Pre-Deployment Checklist

### Before ANY Deployment

**Safety:**
- [ ] Circuit breaker implemented and tested
- [ ] Risk manager integrated with all limits
- [ ] Stop losses on all positions
- [ ] Daily/drawdown limits configured
- [ ] Manual kill switch available

**Code Quality:**
- [ ] All tests pass (70%+ coverage)
- [ ] Algorithm validator passes (no errors)
- [ ] Type hints on all functions
- [ ] Docstrings complete (Google style)
- [ ] Code reviewed

**Backtesting:**
- [ ] No look-ahead bias
- [ ] Walk-forward validation done
- [ ] Out-of-sample testing completed
- [ ] Sharpe > 1.0, Sortino > 1.5
- [ ] Max drawdown < 20%

**Paper Trading:**
- [ ] 2+ weeks with no errors
- [ ] Fill rates within 10% of backtest
- [ ] Risk limits triggered correctly
- [ ] Monitoring and alerts working

---

## 8. Common Pitfalls

### Critical Mistakes to Avoid

| Mistake | Prevention |
|---------|------------|
| **Look-ahead bias** | Never use future-timestamped data |
| **Survivorship bias** | Include delisted securities |
| **Insufficient warmup** | Always check `IsWarmingUp` |
| **Unsafe data access** | Always check `ContainsKey()` |
| **No stop losses** | Every position needs a stop |
| **Overfitting** | Use walk-forward validation |
| **Order failures** | Handle rejections/partial fills |
| **State loss** | Checkpoint state regularly |

### QuantConnect-Specific Patterns

```python
# ALWAYS do this:
def on_data(self, data):
    # 1. Check warmup
    if self.IsWarmingUp:
        return

    # 2. Check data exists
    if not data.ContainsKey(self.symbol):
        return

    # 3. Check indicator ready
    if not self.indicator.IsReady:
        return

    # 4. Check risk limits
    if not self.risk_manager.can_trade():
        return

    # 5. Check circuit breaker
    if not self.circuit_breaker.can_trade():
        return

    # NOW safe to trade
```

---

## Related Documentation

- [QuantConnect GitHub Resources Guide](QUANTCONNECT_GITHUB_GUIDE.md) - LEAN architecture, Algorithm Framework patterns, risk management examples from official code
- [Coding Standards](CODING_STANDARDS.md) - Python style guide and conventions
- [Strategy Documentation](../strategies/README.md) - Trading strategy details
- [Infrastructure Setup](../infrastructure/SETUP_SUMMARY.md) - Compute nodes and Object Store
- [QuantConnect Documentation](https://www.quantconnect.com/docs) - Official API reference

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-29 | Initial release |

---

*These guidelines are mandatory for all development on this project. Violations may result in rejected PRs and reverted commits.*
