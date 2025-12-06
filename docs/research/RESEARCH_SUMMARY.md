# QuantConnect GitHub Research Summary

**Research Phases Completed**: 1, 2, 3
**Date**: 2025-11-30
**Repository**: https://github.com/QuantConnect/Lean

---

## Executive Summary

This research analyzed the QuantConnect LEAN engine repository to identify production-ready patterns for live options trading. The findings validate and enhance the existing two-part spread strategy while providing critical implementation guidance for multi-leg options execution, risk management, and data handling.

---

## Key Findings

### 1. Multi-Leg Options Strategies - VALIDATED

**Your Two-Part Spread Strategy is FULLY SUPPORTED** on Charles Schwab using QuantConnect's infrastructure:

✅ **ComboOrders on Charles Schwab**:
- `ComboMarketOrder()` - SUPPORTED
- `ComboLimitOrder()` with NET pricing - SUPPORTED
- `ComboLegLimitOrder()` with individual leg limits - NOT SUPPORTED

**Critical Implementation Note**: Your strategy should use `ComboLimitOrder` with net debit/credit pricing (NOT individual leg limits).

**Example from Research**:
```python
# Your two-part spread pattern - SCHWAB COMPATIBLE
legs = [
    Leg.create(lower_call, 1),   # Buy lower strike
    Leg.create(upper_call, -1),  # Sell upper strike
]

# Net limit price for entire combo (Schwab supported)
net_debit = 1.90
tickets = self.combo_limit_order(legs, quantity=1, limit_price=net_debit)
```

### 2. Strategy Detection - 37+ Pre-Defined Strategies

QuantConnect automatically detects and groups multi-leg positions using `OptionStrategyDefinitions`:

**Your Strategies Included**:
- Butterfly Call/Put (3-leg)
- Iron Condor (4-leg)
- All spread types (2-leg)

**Benefits**:
- Automatic position grouping
- Correct margin calculations as a unit
- Simplified exit: close entire strategy in one call

### 3. Greeks Access - NO WARMUP REQUIRED

**CRITICAL UPDATE** (LEAN PR #6720):
- Greeks now use implied volatility
- Available IMMEDIATELY upon option data arrival
- NO warmup period required
- Values match Interactive Brokers and major brokerages

**Your Options Scanner** can access Greeks without any initialization delay:
```python
for contract in chain:
    delta = contract.greeks.delta          # Immediate access
    gamma = contract.greeks.gamma
    theta_per_day = contract.greeks.theta_per_day
    iv = contract.implied_volatility

    # Use in scanner logic immediately
    if 0.25 < abs(delta) < 0.35 and iv > 0.20:
        # Underpriced option detection
        pass
```

### 4. Risk Management - NO Built-In Circuit Breaker

**Finding**: LEAN does NOT have a built-in circuit breaker system.

**Your Implementation is CORRECT**: The `TradingCircuitBreaker` in `models/circuit_breaker.py` is the right approach.

**Available in LEAN**:
- Margin call events (`OnMarginCall`, `OnMarginCallWarning`)
- Framework risk models (drawdown limits, position limits)
- Buying power models for position groups

**Recommendation**: Continue using your custom circuit breaker, integrate with LEAN's margin events.

### 5. Universe Selection - Extensive Filtering

**Greeks and IV Filtering Available**:
```python
option.set_filter(lambda u: u
    .strikes(-10, +10)              # Strike range
    .expiration(30, 90)             # DTE range
    .delta(0.25, 0.35)              # Delta range
    .implied_volatility(0.20, 0.50) # IV range
    .open_interest(100, 999999)     # OI minimum
)
```

**Performance Optimizations in LEAN**:
- Strike caching (only regenerates on date change)
- Binary search for strike filtering
- Lazy evaluation
- Aggressive inlining for validation

### 6. Data Handling - Critical Patterns

**Defensive Access Pattern**:
```python
# Always use .get() or TryGetValue pattern
chain = slice.option_chains.get(self.option_symbol)
if not chain:
    return

# Always validate data before use
for contract in chain:
    if contract.bid_price == 0 or contract.ask_price == 0:
        continue  # Invalid data
```

**Warmup Best Practice**:
```python
# Enable globally
self.settings.automatic_indicator_warm_up = True

# Always check status
if self.is_warming_up or not self.indicator.is_ready:
    return
```

---

## Critical Implementation Recommendations

### For Your Two-Part Spread Strategy

**From Research File**: `/home/dshooter/projects/Claude_code_Quantconnect_trading_bot/docs/research/PHASE3_ADVANCED_FEATURES_RESEARCH.md`

**Section 5.1** provides a complete implementation example for your strategy:

1. ✅ Use `ComboLimitOrder` with net pricing
2. ✅ Track pending orders by order ID
3. ✅ Implement 2.5s cancellation in scheduled event
4. ✅ Random delays between 3-15 seconds
5. ✅ Balance positions per option chain
6. ✅ Start with 1 contract for highest fill rate

**Key Pattern Validated**:
```python
# Your philosophy: Wide spreads = opportunities
lower_bid = lower.bid_price
lower_mid = (lower.bid_price + lower.ask_price) / 2
lower_target = lower_bid + (lower_mid - lower_bid) * 0.35  # 35% from bid

# Execute atomically
tickets = self.combo_limit_order(legs, quantity=1, limit_price=net_debit)
```

### Integration with Existing Components

**Risk Manager** (`models/risk_manager.py`):
```python
# Check before each trade
if not self.risk_manager.can_trade():
    return

# Calculate appropriate size
position_size = self.risk_manager.calculate_position_size(
    symbol=symbol,
    entry_price=entry_price,
    stop_loss_price=stop_price
)
```

**Circuit Breaker** (`models/circuit_breaker.py`):
```python
# Check daily loss
daily_pnl_pct = (current_equity - daily_start) / daily_start
self.breaker.check_daily_loss(daily_pnl_pct)

# Check drawdown
self.breaker.check_drawdown(current_equity, peak_equity)

# Halt if needed
if not self.breaker.can_trade():
    return
```

**Options Scanner** (`scanners/options_scanner.py`):
```python
# Use LEAN's filter for initial screening
option.set_filter(lambda u: u
    .delta(config.min_delta, config.max_delta)
    .implied_volatility(config.min_iv, config.max_iv)
    .expiration(config.min_dte, config.max_dte)
)

# Then apply your scanner for deeper analysis
opportunities = self.scanner.scan_chain(
    underlying="SPY",
    spot_price=price,
    chain=list(chain)
)
```

---

## Common Pitfalls to Avoid

### 1. ComboOrder Mistakes for Schwab

❌ **WRONG**:
```python
# Individual leg limits - NOT supported on Schwab
legs = [
    Leg.create(call1, 1, order_price=5.00),  # order_price NOT supported!
]
tickets = self.combo_leg_limit_order(legs, 1)  # NOT on Schwab!
```

✅ **CORRECT**:
```python
# Net limit price - Schwab supported
legs = [Leg.create(call1, 1)]  # No order_price
net_limit = 2.00
tickets = self.combo_limit_order(legs, 1, limit_price=net_limit)
```

### 2. Data Access Errors

❌ **WRONG**:
```python
chain = slice.option_chains[symbol]  # KeyError if no data!
```

✅ **CORRECT**:
```python
chain = slice.option_chains.get(symbol)
if not chain:
    return
```

### 3. Indicator Warmup Mistakes

❌ **WRONG**:
```python
self.sma = self.sma(symbol, 50)  # Missing warmup!
if self.sma.current.value > 100:  # Could be wrong!
    pass
```

✅ **CORRECT**:
```python
self.settings.automatic_indicator_warm_up = True
self.sma = self.sma(symbol, 50)

if self.is_warming_up or not self.sma.is_ready:
    return
```

### 4. Individual Leg Execution

❌ **WRONG**:
```python
# Executing legs individually
self.market_order(call1, 1)   # Leg 1
self.market_order(call2, -2)  # Leg 2
# Risk: Unbalanced positions if only some fill
```

✅ **CORRECT**:
```python
# Execute atomically
legs = [Leg.create(call1, 1), Leg.create(call2, -2)]
tickets = self.combo_limit_order(legs, 1, limit_price=net_limit)
# All legs fill together or none fill
```

---

## Quick Reference Card

### Strategy Factory Methods

```python
# Use for standard strategies
butterfly = OptionStrategies.butterfly_call(symbol, upper, middle, lower, expiry)
self.buy(butterfly, 2)

condor = OptionStrategies.iron_condor(symbol, lp, sp, sc, lc, expiry)
self.buy(condor, 1)
```

### ComboOrders for Custom Strategies

```python
# Create legs
legs = [Leg.create(symbol, quantity)]  # No order_price for Schwab

# Execute atomically
tickets = self.combo_limit_order(legs, qty, limit_price=net_limit)
```

### Option Filtering

```python
option.set_filter(lambda u: u
    .strikes(-10, +10)
    .expiration(30, 90)
    .delta(0.25, 0.35)
    .implied_volatility(0.20, 0.50)
)
```

### Data Access

```python
# Safe pattern
chain = slice.option_chains.get(symbol)
if not chain:
    return

for contract in chain:
    if contract.bid_price == 0:
        continue
    # Use contract
```

### Greeks (No Warmup)

```python
# Available immediately
delta = contract.greeks.delta
gamma = contract.greeks.gamma
theta_per_day = contract.greeks.theta_per_day
iv = contract.implied_volatility
```

---

## Research Files

| File | Purpose |
|------|---------|
| `/docs/research/PHASE3_ADVANCED_FEATURES_RESEARCH.md` | Complete detailed research report |
| `/docs/research/RESEARCH_SUMMARY.md` | This executive summary |

---

## Next Steps

### 1. Update Strategy Implementation

✅ **Validated**: Your two-part spread strategy is fully compatible with Schwab
✅ **Use**: `ComboLimitOrder` with net pricing for atomic execution
✅ **Pattern**: Implement scheduled event for 2.5s cancellation

**Implementation Location**: Update `algorithms/options_trading_bot.py`

### 2. Enhance Options Scanner

✅ **Add**: LEAN's built-in Greeks/IV filtering for initial screening
✅ **Keep**: Your detailed scanner for deeper opportunity analysis
✅ **Benefit**: Reduce data processing, faster execution

**Implementation Location**: Update `scanners/options_scanner.py`

### 3. Integrate Risk Management

✅ **Use**: LEAN's margin call events
✅ **Keep**: Your custom circuit breaker
✅ **Add**: Framework risk models for additional safety layers

**Implementation Location**: Update `models/risk_manager.py` and `models/circuit_breaker.py`

### 4. Add Defensive Data Patterns

✅ **Review**: All data access in algorithms
✅ **Replace**: Direct dictionary access with `.get()` pattern
✅ **Add**: Validation checks before using contract data

**Implementation Location**: All files in `algorithms/`

### 5. Enable Automatic Indicator Warmup

✅ **Add**: `self.settings.automatic_indicator_warm_up = True`
✅ **Add**: `IsWarmingUp` and `IsReady` checks
✅ **Remove**: Manual warmup code (no longer needed for Greeks)

**Implementation Location**: All algorithm files

---

## Validation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Two-Part Spread Strategy | ✅ VALIDATED | Fully supported on Schwab with ComboLimitOrder |
| ComboOrders | ✅ CONFIRMED | ComboLimitOrder with net pricing works on Schwab |
| Greeks Access | ✅ UPDATED | No warmup needed (LEAN PR #6720) |
| Risk Management | ✅ CUSTOM | Use existing circuit breaker, add margin events |
| Universe Selection | ✅ ENHANCED | Add Greeks/IV filtering to existing scanner |
| Data Handling | ✅ PATTERNS IDENTIFIED | Implement defensive access patterns |

---

## Conclusion

The research confirms that your two-part spread strategy is production-ready for QuantConnect with Charles Schwab. The key findings:

1. **ComboOrders work on Schwab** with net pricing (not individual leg limits)
2. **Greeks are available immediately** (no warmup required)
3. **Strategy detection is automatic** for 37+ pre-defined patterns
4. **Custom circuit breaker is correct** (LEAN doesn't have one built-in)
5. **Defensive data patterns are critical** for production reliability

All research findings align with and enhance your existing implementation. The detailed research report provides complete code examples and best practices for integrating these patterns into your trading bot.

**Next Action**: Review the detailed research report (`PHASE3_ADVANCED_FEATURES_RESEARCH.md`) and implement the recommended patterns in your algorithm files.

---

**End of Summary**
