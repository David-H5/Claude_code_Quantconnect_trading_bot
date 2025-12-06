---
title: "QuantConnect Integration Research"
topic: quantconnect
related_upgrades: []
related_docs: []
tags: [quantconnect, options]
created: 2025-12-01
updated: 2025-12-02
---

# Phase 2: Critical Integration Points from QuantConnect GitHub

**Research Date**: 2025-11-30
**Focus**: Charles Schwab brokerage integration, ComboOrders, Greeks calculation, and portfolio management patterns

---

## Executive Summary

This research phase focused on four critical integration areas for the QuantConnect trading bot using Charles Schwab brokerage. Key findings:

1. **Charles Schwab ComboOrder Support**: ✅ **CONFIRMED SUPPORTED** - Combo Market and Combo Limit orders are officially supported for Charles Schwab
2. **Greeks Calculation**: Major improvement in PR #6720 - NO warmup required, uses IV directly
3. **Options Margin**: Position groups implemented (2021) but still has limitations for complex strategies
4. **Critical Limitation**: Only ONE algorithm per Schwab account allowed

---

## 1. Charles Schwab Brokerage Integration

### Supported Order Types

| Order Type | Equity | Equity Options | Index Options | Status |
|-----------|--------|---------------|---------------|--------|
| Market | ✅ | ✅ | ✅ | Fully Supported |
| Limit | ✅ | ✅ | ✅ | Fully Supported |
| Stop Market | ✅ | ✅ | ❌ | Equity only |
| Market on Open | ✅ | ❌ | ❌ | Equity only |
| Market on Close | ✅ | ❌ | ❌ | Equity only |
| **Combo Market** | ✅ | ✅ | ✅ | **SUPPORTED** |
| **Combo Limit** | ✅ | ✅ | ✅ | **SUPPORTED** |
| Combo Leg Limit | ❌ | ❌ | ❌ | NOT supported |

**Source**: [Charles Schwab Documentation](https://www.quantconnect.com/docs/v2/cloud-platform/live-trading/brokerages/charles-schwab)

### Supported Asset Classes

- ✅ US Equities
- ✅ Equity Options
- ✅ Index Options
- ❌ Futures (not yet supported)
- ❌ Forex (not supported)

### Order Properties

**TimeInForce Options**:
- `Day` - Order valid for trading day only
- `GoodTilCanceled` - Order persists until filled or cancelled
- `GoodTilDate` - Order valid until specified date

**Extended Trading Hours**:
- Use `ExtendedRegularTradingHours` property to enable after-hours trading
- Only available for equity orders

### CRITICAL LIMITATIONS

#### 1. Single Algorithm Restriction

**⚠️ CRITICAL**: Charles Schwab only supports authenticating **ONE account at a time per user**.

- Deploying a second algorithm **automatically stops the first**
- All trading strategies **MUST** be combined into a single algorithm
- Cannot run separate algorithms for different strategies simultaneously

**Example - WRONG Approach**:
```python
# Algorithm 1: Options scanner (will be stopped)
# Algorithm 2: Equity momentum (stops first algorithm)
```

**Example - CORRECT Approach**:
```python
class UnifiedTradingBot(QCAlgorithm):
    def Initialize(self):
        self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)
        # Combine all strategies in one algorithm
        self.options_strategy = OptionsScanner()
        self.equity_strategy = EquityMomentum()
```

**Source**: [Charles Schwab Announcement](https://www.quantconnect.com/announcements/18559/introducing-the-charles-schwab-integration-on-quantconnect/)

#### 2. Weekly OAuth Re-authentication

- Schwab API requires OAuth token renewal approximately weekly
- QuantConnect sends reminder emails
- Failure to re-authenticate halts trading

#### 3. Account Setup Requirements

- Trading permissions must be enabled via **thinkorswim platform**
- Changes may require up to **48 hours** to take effect
- Pattern day trading requires $25,000+ account balance

#### 4. Daily Cash Sync

- Account cash holdings synchronize at **7:45 AM ET daily**
- Settlement is immediate for margin accounts

### Known Issues (2024)

#### Account Activity Stream Error

**Issue**: Charles Schwab does not support account activity streams
**Impact**: Daily errors after market close causing strategy termination
**Workaround**: Configure algorithm to ignore activity stream errors

**Source**: [Forum Discussion](https://www.quantconnect.com/forum/discussion/19744/when-deploying-charles-schwab-locally-an-error-will-occur-after-the-market-closes-every-day-causing-the-strategy-to-terminate/)

#### No Paper Trading

- Schwab does **not** provide paper trading API
- Must use QuantConnect's built-in paper modeling for testing
- Paper trading validates execution logic but not real-world Schwab behavior

### Implementation Repository

**Official Repository**: [QuantConnect/Lean.Brokerages.CharlesSchwab](https://github.com/QuantConnect/Lean.Brokerages.CharlesSchwab)
**Last Updated**: November 30, 2024
**Language**: C#

---

## 2. ComboOrder Implementation

### Order Types Available

#### ComboMarketOrder

**Definition**: Execute multiple option legs simultaneously at market prices

**Example**:
```python
# Long call butterfly - all legs at market
legs = [
    Leg.Create(lower_call, 1),    # Buy lower strike
    Leg.Create(atm_call, -2),     # Sell 2 ATM
    Leg.Create(upper_call, 1),    # Buy upper strike
]

self.ComboMarketOrder(legs, quantity=1)
```

**Brokerages**: Schwab ✅, Interactive Brokers ✅, TD Ameritrade ✅

#### ComboLimitOrder

**Definition**: Execute multiple legs with net limit price (debit/credit)

**Example**:
```python
# Iron condor with net credit limit
legs = [
    Leg.Create(put_buy, 1),       # Buy put (lower)
    Leg.Create(put_sell, -1),     # Sell put (higher)
    Leg.Create(call_sell, -1),    # Sell call (lower)
    Leg.Create(call_buy, 1),      # Buy call (higher)
]

# Net credit pricing (NOT individual leg limits)
self.ComboLimitOrder(legs, quantity=1, limit_price=net_credit)
```

**Key Feature**: Uses **net debit/credit** across all legs, not individual leg limits

**Brokerages**: Schwab ✅, Interactive Brokers ✅

**Source**: [Combo Limit Orders Documentation](https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/order-types/combo-limit-orders)

#### ComboLegLimitOrder

**Definition**: Execute multiple legs with **individual** limit prices per leg

**Example**:
```python
# Each leg has its own limit price
legs = [
    Leg.Create(option1, 1, limit_price=2.50),
    Leg.Create(option2, -1, limit_price=3.75),
]

self.ComboLegLimitOrder(legs, quantity=1)
```

**⚠️ SCHWAB STATUS**: **NOT SUPPORTED** on Charles Schwab
**Alternative**: Use `ComboLimitOrder` with net pricing instead

**Brokerages**: Interactive Brokers ✅, Schwab ❌

### ComboOrder Architecture

All combo orders receive a `GroupOrderManager` parameter that coordinates execution:

```csharp
// From Order.cs CreateOrder method
case OrderType.ComboLimit:
    order = new ComboLimitOrder(symbol, quantity, limitPrice, time,
        groupOrderManager, tag, properties);
    break;

case OrderType.ComboMarket:
    order = new ComboMarketOrder(symbol, quantity, time,
        groupOrderManager, tag, properties);
    break;
```

**Source**: [Common/Orders/Order.cs](https://github.com/QuantConnect/Lean/blob/master/Common/Orders/Order.cs)

### Benefits of ComboOrders

1. **Atomic Execution**: All legs fill together or none fill
2. **Single Commission**: One commission per combo (not per leg)
3. **Strategy Detection**: LEAN has 24 files for multi-leg matching
4. **No Legging Risk**: Prevents holding unbalanced positions
5. **Net Pricing**: Better fills on wide bid-ask spreads

### Real-World Iron Condor Example

From [jboesen/iron-condor](https://github.com/Jboesen/iron-condor):

```python
class IronCondorAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2019, 11, 1)
        self.SetEndDate(2020, 11, 1)
        self.SetCash(5000)

        # Add equity and options
        equity = self.AddEquity("SPY", Resolution.Minute)
        option = self.AddOption("SPY", Resolution.Minute)
        option.SetFilter(lambda u: u.IncludeWeeklys().Strikes(-15, 15).Expiration(0, 50))

    def AddToPortfolio(self, option_chain):
        """Construct iron condor with four legs"""
        # Sort contracts
        calls = sorted([x for x in option_chain if x.Right == OptionRight.Call],
                      key=lambda x: x.Strike)
        puts = sorted([x for x in option_chain if x.Right == OptionRight.Put],
                     key=lambda x: x.Strike)

        # Build iron condor: Buy lower put, Sell higher put, Sell lower call, Buy higher call
        otm_put_lower = puts[0]       # Buy OTM put (protection)
        otm_put = puts[15]            # Sell OTM put (income)
        otm_call = calls[-15]         # Sell OTM call (income)
        otm_call_higher = calls[-1]   # Buy OTM call (protection)

        # Execute (uses individual orders, could use ComboMarketOrder)
        self.Buy(otm_put_lower.Symbol, 1)
        self.Sell(otm_put.Symbol, 1)
        self.Sell(otm_call.Symbol, 1)
        self.Buy(otm_call_higher.Symbol, 1)
```

**Modern Approach with ComboOrder**:
```python
# Better: Atomic execution with ComboMarketOrder
legs = [
    Leg.Create(otm_put_lower.Symbol, 1),
    Leg.Create(otm_put.Symbol, -1),
    Leg.Create(otm_call.Symbol, -1),
    Leg.Create(otm_call_higher.Symbol, 1),
]
self.ComboMarketOrder(legs, quantity=1)
```

### Option Strategy Helpers

QuantConnect provides built-in strategy constructors:

**Source**: [Common/Securities/Option/OptionStrategies.cs](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/Option/OptionStrategies.cs)

#### Available Strategies

**Spreads**:
- `OptionStrategies.BullCallSpread(symbol, lower_strike, upper_strike, expiration)`
- `OptionStrategies.BearCallSpread(symbol, lower_strike, upper_strike, expiration)`
- `OptionStrategies.BullPutSpread(symbol, lower_strike, upper_strike, expiration)`
- `OptionStrategies.BearPutSpread(symbol, lower_strike, upper_strike, expiration)`

**Butterflies** (requires equidistant strikes):
- `OptionStrategies.CallButterfly(symbol, lower_strike, middle_strike, upper_strike, expiration)`
- `OptionStrategies.PutButterfly(symbol, lower_strike, middle_strike, upper_strike, expiration)`
- `OptionStrategies.ShortButterflyCall(symbol, lower_strike, middle_strike, upper_strike, expiration)`
- `OptionStrategies.ShortButterflyPut(symbol, lower_strike, middle_strike, upper_strike, expiration)`

**Iron Condors**:
- `OptionStrategies.IronCondor(symbol, put_buy_strike, put_sell_strike, call_sell_strike, call_buy_strike, expiration)`
- `OptionStrategies.ShortIronCondor(symbol, put_buy_strike, put_sell_strike, call_sell_strike, call_buy_strike, expiration)`

**Iron Butterflies**:
- `OptionStrategies.IronButterfly(symbol, lower_strike, middle_strike, upper_strike, expiration)`
- `OptionStrategies.ShortIronButterfly(symbol, lower_strike, middle_strike, upper_strike, expiration)`

**Calendar Spreads**:
- `OptionStrategies.CallCalendarSpread(symbol, strike, near_expiration, far_expiration)`
- `OptionStrategies.PutCalendarSpread(symbol, strike, near_expiration, far_expiration)`

**Straddles & Strangles**:
- `OptionStrategies.Straddle(symbol, strike, expiration)` - Same strike, both call and put
- `OptionStrategies.Strangle(symbol, call_strike, put_strike, expiration)` - Different strikes

**Other Advanced Strategies**:
- `OptionStrategies.BoxSpread()` / `ShortBoxSpread()`
- `OptionStrategies.CoveredCall()` / `CoveredPut()`
- `OptionStrategies.ProtectiveCall()` / `ProtectivePut()`
- `OptionStrategies.Conversion()` / `ReverseConversion()`

**Example Usage**:
```python
# Create iron condor strategy
strategy = OptionStrategies.IronCondor(
    canonical_symbol,
    put_buy_strike=100,    # Long put (protection)
    put_sell_strike=105,   # Short put (income)
    call_sell_strike=115,  # Short call (income)
    call_buy_strike=120,   # Long call (protection)
    expiration=expiry_date
)

# Execute as combo order
self.Buy(strategy, 1)  # Automatically creates ComboOrder
```

**Documentation**: [Option Strategies](https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/option-strategies)

---

## 3. Greeks and Options Pricing

### LEAN PR #6720: IV-Based Greeks (MERGED)

**Pull Request**: [#6720 - Calculate Option Greeks with Implied Volatility](https://github.com/QuantConnect/Lean/pull/6720)
**Status**: ✅ **MERGED** (Production)
**Impact**: Major improvement in Greeks accuracy

#### Key Changes

**Before PR #6720**:
- Greeks calculated using **historical volatility**
- Required **warmup period** for volatility estimation
- Values **differed significantly** from Interactive Brokers and other brokerages
- Example discrepancies: Delta signs opposite (0.25 vs -0.09)

**After PR #6720**:
- Greeks calculated using **implied volatility** from option prices
- **NO warmup required** for Greeks (still needed for theoretical price)
- Values **match Interactive Brokers** and major brokerages
- Uses Fed primary interest rate as risk-free rate

#### Implementation Details

**Pricing Models (Automatic Selection)**:
- **European Options**: Black-Scholes model
- **American Options**: Bjerksund-Stensland model

**Default Assignment**:
```python
# Automatically assigned when options are added
option = self.AddOption("SPY")
# option.PriceModel is now set automatically
# No need to manually configure pricing model
```

**Override Pricing Model (Optional)**:
```python
# Optional: Use different pricing model
option.PriceModel = OptionPriceModels.CrankNicolsonFD()
```

#### Accessing Greeks (NO WARMUP REQUIRED)

```python
def OnData(self, slice):
    chain = slice.OptionChains.get(self.option_symbol)
    if not chain:
        return

    # Greeks available IMMEDIATELY (no warmup needed)
    for contract in chain:
        delta = contract.Greeks.Delta
        gamma = contract.Greeks.Gamma
        theta = contract.Greeks.Theta
        theta_per_day = contract.Greeks.ThetaPerDay  # Matches IB format
        vega = contract.Greeks.Vega
        rho = contract.Greeks.Rho
        iv = contract.ImpliedVolatility

        # Filter by Greeks (no warmup delay)
        if 0.25 < abs(delta) < 0.35 and iv > 0.20:
            self.MarketOrder(contract.Symbol, 1)
```

**Source**: [Greeks.cs Implementation](https://github.com/QuantConnect/Lean/blob/master/Common/Data/Market/Greeks.cs)

#### ThetaPerDay Implementation

```csharp
// From Greeks.cs
public decimal ThetaPerDay => Theta / 365m;
```

**Note**: Use `ThetaPerDay` instead of `Theta` to match Interactive Brokers data format

#### Risk-Free Rate

- Default: **Federal Reserve's primary credit rate**
- Automatically updated
- More accurate than previous assumptions

#### Validation

- Compared against Interactive Brokers data
- Values now **close match** to IB and other brokerages
- Resolves GitHub issues #5838, #4009, #5289

### Greeks Usage Examples

**Filter Options by Delta Range**:
```python
# Target delta-neutral positions
for contract in option_chain:
    if 0.45 <= abs(contract.Greeks.Delta) <= 0.55:
        # Near ATM options
        self.MarketOrder(contract.Symbol, 1)
```

**Gamma Scalping**:
```python
# High gamma for volatility trading
for contract in option_chain:
    if contract.Greeks.Gamma > 0.05:
        # High convexity positions
        self.MarketOrder(contract.Symbol, 1)
```

**Theta Decay Strategy**:
```python
# Target high theta decay
for contract in option_chain:
    if contract.Greeks.ThetaPerDay < -0.10:  # Use ThetaPerDay not Theta
        # Sell high theta decay options
        self.Sell(contract.Symbol, 1)
```

**Vega Exposure Management**:
```python
# Monitor portfolio vega exposure
portfolio_vega = sum(
    holding.Quantity * option.Greeks.Vega
    for symbol, holding in self.Portfolio.items()
    if holding.Type == SecurityType.Option
    for option in [self.Securities[symbol]]
)

if portfolio_vega > 100:  # Risk limit
    # Reduce vega exposure
    pass
```

---

## 4. Portfolio Management and Margin

### Position Groups (Implemented 2021)

**GitHub Issue**: [#4065 - Options margin modelling](https://github.com/QuantConnect/Lean/issues/4065)
**Status**: ✅ **CLOSED/COMPLETED** (April 2021)

#### What's Implemented

**Position Grouping System**:
- `IPositionGroup` interface for grouped positions
- Strategies recognized as units (not individual legs)
- Reduced margin for hedged strategies

**FINRA Rule 2360 Compliance**:
- Short options covered by underlying securities
- Offsetting long positions reduce margin
- Hedging strategies (conversions, collars, box spreads)

**Event-Driven Architecture**:
- `SecurityHolding.QuantityChanged` events
- Automatic position group resolution
- Dynamic margin recalculation

**Example**:
```python
# Iron condor treated as single position group
# Margin calculated for entire strategy, not individual legs
# Significantly lower margin than 4 separate positions
```

**Source**: [OptionMarginModel.cs](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/Option/OptionMarginModel.cs)

#### Limitations

**Current Gaps** (as of 2024):
- Some complex multi-leg strategies not fully optimized
- Margin models may still treat certain orders separately
- Custom strategies may require manual buying power models

**Related Issues**:
- [#2709 - Combo buying power model](https://github.com/QuantConnect/Lean/issues/2709) - Closed, superseded by #5693
- [#5693 - Can't Maximize Position Size of OptionStrategies](https://github.com/QuantConnect/Lean/issues/5693) - Higher priority ongoing work

### Buying Power Models

**Base Class**: [BuyingPowerModel.cs](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/BuyingPowerModel.cs)

#### Available Models

**Cash Buying Power Model**:
- No leverage allowed
- Full payment required upfront
- Source: [CashBuyingPowerModel.cs](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/CashBuyingPowerModel.cs)

**Security Margin Model** (Default for equities):
- Standard margin requirements
- Pattern day trading rules applied
- Regulation T margin

**Option Margin Model**:
- Single long/short positions supported
- Covered calls and covered puts
- Position groups for recognized strategies

#### Custom Buying Power Models

**Interface**:
```python
class MyCustomBuyingPowerModel(BuyingPowerModel):
    def GetBuyingPower(self, parameters):
        """Calculate available buying power"""
        pass

    def HasSufficientBuyingPowerForOrder(self, parameters):
        """Validate order can be placed"""
        pass

    def GetMaximumOrderQuantityForTargetBuyingPower(self, parameters):
        """Calculate max position size"""
        pass
```

**Apply Custom Model**:
```python
def Initialize(self):
    equity = self.AddEquity("SPY")
    equity.SetBuyingPowerModel(MyCustomBuyingPowerModel())
```

### Portfolio Construction Models

**Documentation**: [Portfolio Construction](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/portfolio-construction/key-concepts)

#### Pre-Built Models

**Equal Weighting Portfolio Construction Model**:
```python
from AlgorithmImports import *

self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
```
- Assigns equal share to all insights
- Useful for universe rotation strategies
- Rebalances on new insights

**Confidence Weighted Portfolio Construction Model**:
```python
self.SetPortfolioConstruction(ConfidenceWeightedPortfolioConstructionModel())
```
- Weights based on insight confidence levels
- Creates long/short positions accordingly
- Normalizes weights if total exceeds 1

**Mean Variance Optimization**:
```python
self.SetPortfolioConstruction(MeanVarianceOptimizationPortfolioConstructionModel())
```
- Minimizes portfolio volatility
- Uses 252 days of log returns
- Maximizes Sharpe ratio

**Black-Litterman Model**:
```python
self.SetPortfolioConstruction(BlackLittermanOptimizationPortfolioConstructionModel())
```
- Combines multiple alpha sources
- Treats alphas as "investor views"
- Optimal for multi-strategy portfolios

**Source**: [PortfolioConstructionModel.cs](https://github.com/QuantConnect/Lean/blob/master/Algorithm/Portfolio/PortfolioConstructionModel.cs)

#### Custom Portfolio Construction

**Base Pattern**:
```python
class MyPortfolioConstructionModel(PortfolioConstructionModel):
    def CreateTargets(self, algorithm, insights):
        """Convert insights to portfolio targets"""
        targets = []
        for insight in insights:
            # Calculate position size based on custom logic
            weight = self.CalculateWeight(insight)
            targets.append(PortfolioTarget(insight.Symbol, weight))
        return targets
```

### Position Sizing Best Practices

**SetHoldings Method**:
```python
# Set 50% of portfolio to SPY
self.SetHoldings("SPY", 0.50)

# Liquidate entire position
self.SetHoldings("SPY", 0)

# Multiple positions
targets = [
    PortfolioTarget(self.spy, 0.3),
    PortfolioTarget(self.qqq, 0.3),
    PortfolioTarget(self.iwm, 0.2),
]
self.SetHoldings(targets)
```

**Key Features**:
- Accounts for lot sizes
- Includes pre-calculated order fees
- Orders sorted by position delta
- Reduces positions before increasing (prevents insufficient buying power)

**Source**: [Position Sizing Documentation](https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/position-sizing)

### Risk Management Framework

**Base Class**: [RiskManagementModel](https://github.com/QuantConnect/Lean/blob/master/Algorithm.Framework/Risk/TrailingStopRiskManagementModel.py)

#### Implementation Pattern

```python
class MyRiskManagementModel(RiskManagementModel):
    def ManageRisk(self, algorithm, targets):
        """
        Evaluate positions and return adjusted targets

        Returns:
            List of PortfolioTarget to liquidate or adjust
        """
        risk_adjusted_targets = []

        for symbol, security in algorithm.Securities.items():
            if not security.Invested:
                continue

            # Monitor position
            position_value = security.Holdings.HoldingsValue

            # Check risk limits
            if self.ShouldLiquidate(security):
                # Cancel insights
                algorithm.Insights.Cancel([symbol])
                # Liquidate position
                risk_adjusted_targets.append(PortfolioTarget(symbol, 0))

        return risk_adjusted_targets
```

#### Pre-Built Risk Models

**Maximum Drawdown Per Security**:
```python
from AlgorithmImports import *

# Liquidate if security drops 5% from highest unrealized profit
self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.05))
```

**Maximum Sector Exposure**:
```python
# Limit sector exposure to 30% of portfolio
self.SetRiskManagement(MaximumSectorExposureRiskManagementModel(0.30))
```

**Trailing Stop**:
```python
# 10% trailing stop from highest profit
self.SetRiskManagement(TrailingStopRiskManagementModel(0.10))
```

**Source**: [Risk Management Models](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/risk-management/supported-models)

---

## 5. Recommended Updates to CLAUDE.md

### Section: "Multi-Leg Options with ComboOrders"

**Current Status**: Document states "on the way to further enhance options trading"

**UPDATE TO**:
```markdown
### Multi-Leg Options with ComboOrders

**✅ CONFIRMED: ComboOrders are FULLY SUPPORTED on Charles Schwab** (as of 2025-11-30)

```python
# Long call butterfly - atomic execution
atm_strike = self.get_atm_strike(underlying_price)
legs = [
    Leg.Create(self.get_call(atm_strike - 5), 1),   # Buy lower
    Leg.Create(self.get_call(atm_strike), -2),      # Sell ATM
    Leg.Create(self.get_call(atm_strike + 5), 1),   # Buy upper
]

# All legs fill together or none fill
# Uses net debit/credit pricing (NOT individual leg limits)
self.ComboLimitOrder(legs, quantity=1, limit_price=net_debit)
```

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

**Benefits**:
- Automatic strategy detection (LEAN has 24 files for multi-leg matching)
- Atomic execution (all-or-nothing fills)
- Single commission per combo
- Prevents holding unbalanced positions
- **Your two-part spread strategy works with Schwab using ComboLimitOrder!**
```

### Section: "IV-Based Greeks (LEAN PR #6720)"

**ADD NOTE**:
```markdown
**CRITICAL UPDATE**: Greeks now use implied volatility - **NO warmup required**.

```python
# Immediate access to Greeks (no warmup needed)
for contract in option_chain:
    delta = contract.Greeks.Delta
    gamma = contract.Greeks.Gamma
    theta_per_day = contract.Greeks.ThetaPerDay  # Daily theta
    iv = contract.ImpliedVolatility

    # Greeks available immediately upon data arrival
    if 0.25 < abs(delta) < 0.35 and iv > 0.20:
        # Trade logic
```

**Key Changes**:
- Greeks calculated using IV from option prices
- Values match Interactive Brokers and major brokerages
- Default models: Black-Scholes (European), Bjerksund-Stensland (American)
- No warmup period required for Greeks calculations
```

### Section: "CRITICAL: Platform-Specific Limitations"

**ADD**:
```markdown
#### Known Issues (2024-2025)

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

### Section: "Options Trading Patterns"

**UPDATE Initialize Section**:
```python
def initialize(self):
    equity = self.add_equity("SPY")
    option = self.add_option("SPY")
    option.set_filter(-10, +10, 0, 90)  # ±10 strikes, 0-90 days

    # Optional: Specify pricing model (defaults to Black-Scholes/Bjerksund-Stensland)
    # option.price_model = OptionPriceModels.crank_nicolson_fd()

    # NOTE: As of LEAN PR #6720, Greeks use IV and require NO warmup
    # Greeks are available immediately upon option data arrival
```

**UPDATE OnData Section**:
```python
def on_data(self, slice):
    chain = slice.option_chains.get(self.option_symbol)
    if not chain:
        return

    # Greeks available immediately (IV-based, no warmup required)
    for contract in chain:
        delta = contract.greeks.delta
        gamma = contract.greeks.gamma
        theta = contract.greeks.theta
        theta_per_day = contract.greeks.theta_per_day  # Daily theta decay
        vega = contract.greeks.vega
        rho = contract.greeks.rho
        iv = contract.implied_volatility

        # Greeks values match Interactive Brokers and major brokerages
        if 0.25 < abs(delta) < 0.35 and iv > 0.20:
            self.market_order(contract.symbol, 1)
```

---

## 6. Action Items for CLAUDE.md

### High Priority

1. ✅ **Update ComboOrder Status**: Change from "on the way" to "FULLY SUPPORTED"
2. ✅ **Add ComboLimitOrder vs ComboLegLimitOrder Distinction**: Clarify Schwab supports net pricing, not individual leg limits
3. ✅ **Update Greeks Warmup Requirements**: Remove warmup requirement, add PR #6720 details
4. ✅ **Add ThetaPerDay Usage**: Document to use `ThetaPerDay` instead of `Theta`
5. ⚠️ **Document Known Schwab Issues**: Add account activity stream error and workarounds

### Medium Priority

6. 📝 **Add OptionStrategies Helper Examples**: Document built-in strategy constructors
7. 📝 **Add Position Groups Section**: Explain margin benefits for multi-leg strategies
8. 📝 **Add Portfolio Construction Patterns**: Document Equal Weighting, Mean Variance, Black-Litterman models
9. 📝 **Add Risk Management Framework**: Document TrailingStop, MaximumDrawdown models

### Low Priority

10. 📝 **Add Custom Buying Power Model Pattern**: For advanced margin management
11. 📝 **Add Real-World Iron Condor Example**: Include full algorithm from jboesen repository
12. 📝 **Document SetHoldings Best Practices**: Order sorting, fee calculation details

---

## 7. Critical Code Examples

### Modern Iron Condor with ComboLimitOrder

```python
from AlgorithmImports import *

class ModernIronCondorAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)

        # Charles Schwab brokerage
        self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)

        # Add SPY options
        equity = self.AddEquity("SPY", Resolution.Minute)
        option = self.AddOption("SPY", Resolution.Minute)
        option.SetFilter(lambda u: u.IncludeWeeklys().Strikes(-15, 15).Expiration(30, 60))
        self.symbol = option.Symbol

        # Greeks available immediately (no warmup needed)

    def OnData(self, slice):
        # Get option chain
        if slice.OptionChains.ContainsKey(self.symbol):
            chain = slice.OptionChains[self.symbol]
        else:
            return

        # Skip if already invested
        if self.Portfolio.Invested:
            return

        # Separate calls and puts
        calls = [x for x in chain if x.Right == OptionRight.Call]
        puts = [x for x in chain if x.Right == OptionRight.Put]

        # Sort by strike
        calls = sorted(calls, key=lambda x: x.Strike)
        puts = sorted(puts, key=lambda x: x.Strike)

        if len(calls) < 20 or len(puts) < 20:
            return

        # Select strikes for iron condor
        # Buy lower put, Sell higher put, Sell lower call, Buy higher call
        put_buy = puts[5]      # OTM protection
        put_sell = puts[10]    # OTM income
        call_sell = calls[-10] # OTM income
        call_buy = calls[-5]   # OTM protection

        # Filter by Greeks (no warmup required!)
        if abs(put_sell.Greeks.Delta) > 0.30:
            return  # Too close to ATM
        if abs(call_sell.Greeks.Delta) > 0.30:
            return  # Too close to ATM

        # Calculate net credit
        put_spread_credit = put_sell.BidPrice - put_buy.AskPrice
        call_spread_credit = call_sell.BidPrice - call_buy.AskPrice
        net_credit = put_spread_credit + call_spread_credit

        if net_credit <= 0:
            return  # Must receive credit

        # Create combo order legs
        legs = [
            Leg.Create(put_buy.Symbol, 1),      # Buy put protection
            Leg.Create(put_sell.Symbol, -1),    # Sell put for income
            Leg.Create(call_sell.Symbol, -1),   # Sell call for income
            Leg.Create(call_buy.Symbol, 1),     # Buy call protection
        ]

        # Execute as atomic combo limit order
        # Uses NET credit pricing (not individual leg limits)
        self.ComboLimitOrder(legs, quantity=1, limit_price=net_credit * 0.9)

        self.Debug(f"Iron Condor: Net Credit = ${net_credit:.2f}")
        self.Debug(f"Put Spread Delta: {put_sell.Greeks.Delta:.3f}")
        self.Debug(f"Call Spread Delta: {call_sell.Greeks.Delta:.3f}")
```

### Using OptionStrategies Helper

```python
from AlgorithmImports import *

class IronCondorWithHelperAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)

        option = self.AddOption("SPY", Resolution.Minute)
        option.SetFilter(lambda u: u.IncludeWeeklys().Strikes(-15, 15).Expiration(30, 60))
        self.symbol = option.Symbol

    def OnData(self, slice):
        if self.Portfolio.Invested:
            return

        if slice.OptionChains.ContainsKey(self.symbol):
            chain = slice.OptionChains[self.symbol]
        else:
            return

        # Get ATM strike
        underlying_price = self.Securities[self.symbol.Underlying].Price
        expiry = sorted(set([x.Expiry for x in chain]))[0]

        # Use OptionStrategies helper to create iron condor
        # Automatically creates proper leg structure
        strategy = OptionStrategies.IronCondor(
            self.symbol,
            put_buy_strike=underlying_price - 10,   # Protection
            put_sell_strike=underlying_price - 5,   # Income
            call_sell_strike=underlying_price + 5,  # Income
            call_buy_strike=underlying_price + 10,  # Protection
            expiration=expiry
        )

        # Buy strategy (creates ComboOrder automatically)
        self.Buy(strategy, 1)
```

### Risk Management with TrailingStop

```python
from AlgorithmImports import *

class RiskManagedIronCondorAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)

        # Set trailing stop risk management
        # Liquidate if position drops 20% from highest profit
        self.SetRiskManagement(TrailingStopRiskManagementModel(0.20))

        # Alternative: Maximum drawdown per security
        # self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.15))

        option = self.AddOption("SPY")
        self.symbol = option.Symbol

    # ... rest of algorithm
```

---

## 8. Resources and References

### Official Documentation

- [Charles Schwab Brokerage](https://www.quantconnect.com/docs/v2/cloud-platform/live-trading/brokerages/charles-schwab)
- [Combo Limit Orders](https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/order-types/combo-limit-orders)
- [Combo Market Orders](https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/order-types/combo-market-orders)
- [Option Strategies](https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/option-strategies)
- [Portfolio Construction](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/portfolio-construction/key-concepts)
- [Risk Management](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/risk-management/key-concepts)
- [Buying Power Models](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/buying-power)

### GitHub Repositories

- [QuantConnect/Lean](https://github.com/QuantConnect/Lean) - Main LEAN engine
- [QuantConnect/Lean.Brokerages.CharlesSchwab](https://github.com/QuantConnect/Lean.Brokerages.CharlesSchwab) - Schwab integration
- [QuantConnect/Tutorials](https://github.com/QuantConnect/Tutorials) - Official tutorials
- [jboesen/iron-condor](https://github.com/Jboesen/iron-condor) - Real-world iron condor example

### Key Source Files

- [Order.cs](https://github.com/QuantConnect/Lean/blob/master/Common/Orders/Order.cs) - ComboOrder creation
- [Greeks.cs](https://github.com/QuantConnect/Lean/blob/master/Common/Data/Market/Greeks.cs) - Greeks implementation
- [OptionStrategies.cs](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/Option/OptionStrategies.cs) - Strategy helpers
- [OptionMarginModel.cs](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/Option/OptionMarginModel.cs) - Margin calculation
- [BuyingPowerModel.cs](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/BuyingPowerModel.cs) - Buying power base
- [PortfolioConstructionModel.cs](https://github.com/QuantConnect/Lean/blob/master/Algorithm/Portfolio/PortfolioConstructionModel.cs) - Portfolio construction
- [TrailingStopRiskManagementModel.py](https://github.com/QuantConnect/Lean/blob/master/Algorithm.Framework/Risk/TrailingStopRiskManagementModel.py) - Risk management example

### GitHub Issues

- [#6720 - Calculate Option Greeks with Implied Volatility](https://github.com/QuantConnect/Lean/pull/6720) - Greeks PR (MERGED)
- [#4065 - Options margin modelling](https://github.com/QuantConnect/Lean/issues/4065) - Position groups (CLOSED)
- [#2709 - Combo buying power model](https://github.com/QuantConnect/Lean/issues/2709) - Superseded by #5693
- [#5693 - Can't Maximize Position Size of OptionStrategies](https://github.com/QuantConnect/Lean/issues/5693) - Ongoing
- [#2900 - Option strategy limit order](https://github.com/QuantConnect/Lean/issues/2900) - Historical reference

### Learning Resources

- [Iron Condor Tutorial](https://www.quantconnect.com/learning/articles/applied-options/iron-condor)
- [Iron Butterfly Tutorial](https://www.quantconnect.com/learning/articles/applied-options/iron-butterfly)
- [Applied Options Series](https://www.quantconnect.com/learning/articles/applied-options)

### Forum Discussions

- [ComboLimitOrder for Iron Condor](https://www.quantconnect.com/forum/discussion/16501/combolimitorder-limit-price-for-open-and-close-of-iron-condor-strategy/p1/comment-46383)
- [Multi-Leg Options Atomically](https://www.quantconnect.com/forum/discussion/16109/multi-leg-option-strategies-atomically/)
- [Multiple Algorithms per Schwab Account](https://www.quantconnect.com/forum/discussion/19210/is-it-possible-to-deploy-multiple-quantconnect-algorithms-to-one-schwab-account/)
- [Schwab Daily Errors](https://www.quantconnect.com/forum/discussion/19744/when-deploying-charles-schwab-locally-an-error-will-occur-after-the-market-closes-every-day-causing-the-strategy-to-terminate/)

---

## 9. Summary of Critical Findings

### ✅ Confirmed Capabilities

1. **ComboOrders Fully Supported on Schwab** - ComboMarketOrder and ComboLimitOrder work with Charles Schwab
2. **Greeks No Warmup Required** - PR #6720 merged, Greeks available immediately using IV
3. **Position Groups Implemented** - Margin benefits for multi-leg strategies (2021)
4. **26+ Option Strategies Available** - Built-in helpers for common strategies
5. **Risk Management Framework** - Pre-built trailing stop, drawdown, sector exposure models

### ⚠️ Critical Limitations

1. **ONE Algorithm Per Schwab Account** - Second algorithm stops first
2. **ComboLegLimitOrder NOT Supported on Schwab** - Use ComboLimitOrder with net pricing instead
3. **Weekly OAuth Re-authentication Required** - API tokens expire approximately weekly
4. **No Paper Trading API** - Must use QuantConnect's paper modeling
5. **Account Activity Stream Errors** - Daily errors after market close (workaround available)

### 🔧 Implementation Recommendations

1. **Use ComboLimitOrder for Multi-Leg Orders** - Atomic execution, net pricing
2. **Use OptionStrategies Helpers** - Simplifies strategy construction
3. **Access Greeks Immediately** - No warmup needed after PR #6720
4. **Use ThetaPerDay Not Theta** - Matches Interactive Brokers format
5. **Implement TrailingStop or MaxDrawdown** - Pre-built risk management
6. **Combine All Strategies in One Algorithm** - Schwab limitation workaround

### 📊 Documentation Quality

- **Excellent**: ComboOrder documentation, Greeks implementation, Option strategies
- **Good**: Risk management, portfolio construction, buying power models
- **Limited**: Schwab-specific quirks, ComboLegLimitOrder limitations, margin model edge cases
- **Missing**: Detailed Schwab ComboOrder examples, account activity stream workaround code

---

**Report Prepared**: 2025-11-30
**Research Focus**: Charles Schwab integration, ComboOrders, Greeks, Portfolio Management
**Recommendation**: Update CLAUDE.md with confirmed ComboOrder support and Greeks changes
