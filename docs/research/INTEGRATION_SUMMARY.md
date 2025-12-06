# QuantConnect Integration Research - Executive Summary

**Date**: 2025-11-30
**Researcher**: Claude Code
**Phase**: 2 - Critical Integration Points

---

## 🎯 Mission Critical Findings

### 1. Charles Schwab ComboOrders - FULLY SUPPORTED ✅

**Your CLAUDE.md is OUTDATED** - It says combo orders are "on the way", but they are **FULLY SUPPORTED** as of 2024.

| Order Type | Schwab Status | Use Case |
|-----------|--------------|----------|
| `ComboMarketOrder` | ✅ **SUPPORTED** | Execute multi-leg at market |
| `ComboLimitOrder` | ✅ **SUPPORTED** | Execute multi-leg with net limit price |
| `ComboLegLimitOrder` | ❌ **NOT SUPPORTED** | Individual leg limits (IB only) |

**Critical Distinction**:
- `ComboLimitOrder` uses **net debit/credit** pricing across all legs
- `ComboLegLimitOrder` uses **individual limit prices** per leg (NOT available on Schwab)

**Your Two-Part Spread Strategy**: ✅ Works with `ComboLimitOrder` on Schwab!

---

### 2. Greeks Calculation - NO WARMUP REQUIRED ✅

**PR #6720 MERGED** - Greeks now use implied volatility, available immediately.

**Before PR #6720**:
```python
# OLD: Required warmup for historical volatility
self.SetWarmup(self.lookback_period)
if self.IsWarmingUp:
    return  # Skip until warmed up
```

**After PR #6720** (Current):
```python
# NEW: Greeks available immediately (no warmup)
for contract in option_chain:
    delta = contract.Greeks.Delta           # Available immediately
    theta_per_day = contract.Greeks.ThetaPerDay  # Use this, not Theta
    iv = contract.ImpliedVolatility

    # Filter by Greeks with NO delay
    if 0.25 < abs(delta) < 0.35 and iv > 0.20:
        self.MarketOrder(contract.Symbol, 1)
```

**Key Changes**:
- Greeks calculated using **IV from option prices** (not historical volatility)
- Values **match Interactive Brokers** and major brokerages
- **NO warmup period** required for Greeks
- Use `ThetaPerDay` instead of `Theta` to match IB format

**Pricing Models** (automatic):
- European options: Black-Scholes
- American options: Bjerksund-Stensland

---

### 3. Charles Schwab Critical Limitations ⚠️

#### ONE Algorithm Per Account (CRITICAL)

**Your CLAUDE.md is CORRECT** - Only one algorithm allowed.

```python
# WRONG - Second algorithm stops first
# Algorithm 1: Options scanner → STOPPED when Algorithm 2 deploys
# Algorithm 2: Equity momentum → Stops Algorithm 1

# CORRECT - Combine all strategies
class UnifiedTradingBot(QCAlgorithm):
    def Initialize(self):
        self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)
        self.options_strategy = OptionsScanner()
        self.equity_strategy = EquityMomentum()
        # All strategies in ONE algorithm
```

#### Other Schwab Limitations

| Limitation | Impact | Workaround |
|-----------|--------|------------|
| Weekly OAuth renewal | Trading halts if not renewed | Set calendar reminder |
| No paper trading API | Can't test with real Schwab API | Use QC paper modeling |
| Account activity stream error | Daily errors after market close | Ignore activity stream errors |
| Trading permissions setup | 48-hour delay for changes | Configure via thinkorswim first |

---

## 📝 Required CLAUDE.md Updates

### High Priority - Update Immediately

#### 1. Update ComboOrder Section

**Change FROM**:
> "Combo Market and Combo Limit orders on the way to further enhance options trading"

**Change TO**:
```markdown
✅ **CONFIRMED: ComboOrders are FULLY SUPPORTED on Charles Schwab** (as of 2025-11-30)

**Available ComboOrder Types**:

| Order Type | Schwab Support | Description |
|-----------|---------------|-------------|
| `ComboMarketOrder()` | ✅ SUPPORTED | Execute at market |
| `ComboLimitOrder()` | ✅ SUPPORTED | Net limit price across all legs |
| `ComboLegLimitOrder()` | ❌ NOT supported | Individual leg limits (not available on Schwab) |

**Important for Charles Schwab**:
- Use `ComboLimitOrder()` with net debit/credit pricing
- Do NOT use `ComboLegLimitOrder()` - individual leg limits not supported
- Do NOT specify `order_price` parameter in `Leg.Create()` calls
```

#### 2. Update Greeks Section

**Change FROM**:
> "NOTE: As of LEAN PR #6720, Greeks use IV and require NO warmup"

**Change TO**:
```markdown
**CRITICAL UPDATE**: Greeks now use implied volatility - **NO warmup required**.

```python
# Immediate access to Greeks (no warmup needed)
for contract in option_chain:
    delta = contract.Greeks.Delta
    gamma = contract.Greeks.Gamma
    theta_per_day = contract.Greeks.ThetaPerDay  # Use this instead of Theta
    iv = contract.ImpliedVolatility

    # Greeks available immediately upon data arrival
    if 0.25 < abs(delta) < 0.35 and iv > 0.20:
        # Trade logic
```

**Key Changes**:
- Greeks calculated using IV from option prices
- Values match Interactive Brokers and major brokerages
- Default models: Black-Scholes (European), Bjerksund-Stensland (American)
- **Use `ThetaPerDay` not `Theta`** to match IB data format
- No warmup period required for Greeks calculations
```

#### 3. Add Known Schwab Issues Section

**ADD NEW SECTION**:
```markdown
### Known Issues (2024-2025)

**Account Activity Stream Error**:
- Charles Schwab does not support account activity streams
- Attempting to subscribe causes brokerage errors
- Occurs daily after market close and can terminate strategies
- **Workaround**: Configure algorithm to ignore activity stream errors

**No Paper Trading API**:
- Schwab does not provide paper trading endpoints
- Must use QuantConnect's built-in paper modeling
- Paper trading validates execution logic, not real-world Schwab behavior

**OAuth Re-authentication**:
- Required approximately weekly
- QuantConnect sends reminder emails
- Failure to renew halts trading until re-authenticated
```

---

## 🔧 Implementation Patterns

### Modern Iron Condor with ComboLimitOrder

```python
# Modern approach: Atomic execution with net pricing
legs = [
    Leg.Create(put_buy.Symbol, 1),      # Buy put protection
    Leg.Create(put_sell.Symbol, -1),    # Sell put for income
    Leg.Create(call_sell.Symbol, -1),   # Sell call for income
    Leg.Create(call_buy.Symbol, 1),     # Buy call protection
]

# Execute as combo limit order
# Uses NET credit pricing (not individual leg limits)
self.ComboLimitOrder(legs, quantity=1, limit_price=net_credit * 0.9)
```

**Benefits**:
- ✅ All legs fill together or none fill (atomic)
- ✅ Single commission per combo
- ✅ Prevents holding unbalanced positions
- ✅ Works on Charles Schwab

### Using OptionStrategies Helpers

```python
# Even easier: Use built-in strategy constructor
strategy = OptionStrategies.IronCondor(
    self.symbol,
    put_buy_strike=100,    # Protection
    put_sell_strike=105,   # Income
    call_sell_strike=115,  # Income
    call_buy_strike=120,   # Protection
    expiration=expiry
)

# Buy strategy (creates ComboOrder automatically)
self.Buy(strategy, 1)
```

**26+ Built-in Strategies**:
- Butterflies (Call, Put, Iron)
- Condors (Iron, Short Iron)
- Spreads (Bull/Bear Call/Put, Calendar, Box)
- Straddles, Strangles, Covered Calls
- Protective Puts, Collars, Conversions

**Source**: [OptionStrategies.cs](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/Option/OptionStrategies.cs)

---

## 📊 Portfolio & Risk Management

### Position Groups (Margin Benefits)

**Implemented**: April 2021 (Issue #4065)

**What it does**:
- Recognizes option strategies as units (not individual legs)
- Reduces margin for hedged strategies
- FINRA Rule 2360 compliant

**Example**:
```python
# Iron condor treated as single position group
# Margin calculated for entire strategy
# MUCH lower margin than 4 separate positions
```

**Limitations**:
- Some complex strategies not fully optimized
- Ongoing work in Issue #5693

### Risk Management Models

```python
# Trailing stop: Liquidate if drops 20% from highest profit
self.SetRiskManagement(TrailingStopRiskManagementModel(0.20))

# Maximum drawdown per security
self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.15))

# Maximum sector exposure
self.SetRiskManagement(MaximumSectorExposureRiskManagementModel(0.30))
```

**Source**: [Risk Management Models](https://github.com/QuantConnect/Lean/tree/master/Algorithm.Framework/Risk)

### Portfolio Construction Models

```python
# Equal weighting across all insights
self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())

# Confidence-weighted positions
self.SetPortfolioConstruction(ConfidenceWeightedPortfolioConstructionModel())

# Mean-variance optimization (minimize volatility)
self.SetPortfolioConstruction(MeanVarianceOptimizationPortfolioConstructionModel())

# Black-Litterman (multi-alpha sources)
self.SetPortfolioConstruction(BlackLittermanOptimizationPortfolioConstructionModel())
```

---

## 🎓 Learning Resources

### Official Documentation

- [Charles Schwab Brokerage](https://www.quantconnect.com/docs/v2/cloud-platform/live-trading/brokerages/charles-schwab)
- [Combo Limit Orders](https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/order-types/combo-limit-orders)
- [Option Strategies](https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/option-strategies)
- [Portfolio Construction](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/portfolio-construction/key-concepts)
- [Risk Management](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/risk-management/key-concepts)

### GitHub Examples

- [QuantConnect/Lean](https://github.com/QuantConnect/Lean) - Main LEAN engine
- [jboesen/iron-condor](https://github.com/Jboesen/iron-condor) - Real-world iron condor algorithm
- [QuantConnect/Tutorials](https://github.com/QuantConnect/Tutorials) - Official tutorials

### Key Pull Requests

- [#6720 - Greeks with IV](https://github.com/QuantConnect/Lean/pull/6720) - ✅ MERGED
- [#4065 - Options Margin](https://github.com/QuantConnect/Lean/issues/4065) - ✅ CLOSED (2021)

---

## ✅ Action Items Checklist

### Immediate (High Priority)

- [ ] Update CLAUDE.md: Change "combo orders on the way" to "FULLY SUPPORTED"
- [ ] Update CLAUDE.md: Add distinction between ComboLimitOrder vs ComboLegLimitOrder
- [ ] Update CLAUDE.md: Remove Greeks warmup requirement
- [ ] Update CLAUDE.md: Add note to use ThetaPerDay instead of Theta
- [ ] Update CLAUDE.md: Add Known Schwab Issues section (activity stream error)

### Soon (Medium Priority)

- [ ] Add OptionStrategies helper examples to docs
- [ ] Document position groups margin benefits
- [ ] Add portfolio construction model examples
- [ ] Add risk management framework patterns

### Eventually (Low Priority)

- [ ] Create custom buying power model example
- [ ] Add real-world iron condor algorithm to examples
- [ ] Document SetHoldings order sorting behavior
- [ ] Create Schwab-specific best practices guide

---

## 🚨 Critical Reminders

1. **ComboOrders ARE SUPPORTED on Schwab** - Your CLAUDE.md needs updating
2. **Greeks require NO warmup** - PR #6720 merged, IV-based calculation
3. **Only ONE algorithm per Schwab account** - Combine all strategies
4. **Use ComboLimitOrder not ComboLegLimitOrder** - Schwab uses net pricing
5. **Use ThetaPerDay not Theta** - Matches Interactive Brokers format
6. **Weekly OAuth renewal required** - Set calendar reminder

---

**Full Research Report**: `/home/dshooter/projects/Claude_code_Quantconnect_trading_bot/docs/research/PHASE_2_INTEGRATION_RESEARCH.md`

**Next Steps**: Review detailed report and update CLAUDE.md with confirmed integration details.
