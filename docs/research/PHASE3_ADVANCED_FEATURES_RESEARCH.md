---
title: "QuantConnect Advanced Features Research"
topic: quantconnect
related_upgrades: []
related_docs: []
tags: [quantconnect, options]
created: 2025-12-01
updated: 2025-12-02
---

# Phase 3: Advanced Features Research - QuantConnect GitHub Repository

**Research Date**: 2025-11-30
**Repository**: https://github.com/QuantConnect/Lean
**Focus**: Production-ready patterns for live options trading

---

## Table of Contents

1. [Multi-Leg Options Strategies](#1-multi-leg-options-strategies)
2. [Risk Management Patterns](#2-risk-management-patterns)
3. [Universe Selection for Options](#3-universe-selection-for-options)
4. [Data Handling Best Practices](#4-data-handling-best-practices)
5. [Implementation Recommendations](#5-implementation-recommendations)
6. [Common Pitfalls to Avoid](#6-common-pitfalls-to-avoid)
7. [Quick Reference](#7-quick-reference)

---

## 1. Multi-Leg Options Strategies

### 1.1 Strategy Detection and Recognition

QuantConnect includes **37+ pre-defined option strategies** with automatic detection via the `OptionStrategyDefinitions` class.

**Location**: `Common/Securities/Option/StrategyMatcher/OptionStrategyDefinitions.cs`

#### Complete Strategy List

**Single-Leg Strategies** (2):
- Naked Call, Naked Put

**Underlying + Single Option** (4):
- Covered Call, Protective Call
- Covered Put, Protective Put

**Underlying + Dual Options** (3):
- Protective Collar
- Conversion, Reverse Conversion

**Two-Leg Spreads** (8):
- Bull Call Spread, Bear Call Spread
- Bull Put Spread, Bear Put Spread
- Call Calendar Spread, Short Call Calendar Spread
- Put Calendar Spread, Short Put Calendar Spread

**Straddles/Strangles** (4):
- Straddle, Short Straddle
- Strangle, Short Strangle

**Three-Leg Butterflies** (6):
- Butterfly Call, Short Butterfly Call
- Butterfly Put, Short Butterfly Put
- Iron Butterfly, Short Iron Butterfly

**Four-Leg Complex** (6):
- Iron Condor, Short Iron Condor
- Box Spread, Short Box Spread
- Jelly Roll, Short Jelly Roll

**Ladder Strategies** (4):
- Bear Call Ladder, Bull Call Ladder
- Bear Put Ladder, Bull Put Ladder

### 1.2 Strategy Detection Mechanism

**Automatic Recognition Features**:

1. **Strike validation**: Verifies relative strike positions using predicates
2. **Expiration matching**: Ensures all legs share identical expiration dates
3. **Quantity validation**: Confirms leg quantities match strategy definitions
4. **Inversion handling**: Automatically generates inverted strategies (reversed quantities)

**Example Detection Logic** (from source):

```csharp
// Butterfly validation pattern
predicates.Add(p => p.Strike > legs[0].Strike);  // Middle strike check
predicates.Add(p => p.Expiration == legs[0].Expiration);  // Same expiry
// Quantity multiples validated against definition
```

### 1.3 Using OptionStrategies Factory (Recommended Approach)

**Python Implementation Pattern**:

```python
from AlgorithmImports import *

class MyOptionsAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        # Subscribe to options
        option = self.add_option("SPY")
        self.option_symbol = option.symbol
        option.set_filter(-10, +10, 0, 90)

    def on_data(self, slice):
        if self.portfolio.invested:
            return

        chain = slice.option_chains.get(self.option_symbol)
        if not chain:
            return

        # Get contracts sorted by expiration and strike
        contracts = sorted(chain, key=lambda x: (x.expiry, x.strike))
        calls = [c for c in contracts if c.right == OptionRight.CALL]

        if len(calls) < 3:
            return

        # Select strikes for butterfly
        atm_strike = self.find_atm_strike(calls)
        lower_strike = atm_strike - 5
        upper_strike = atm_strike + 5
        expiry = calls[0].expiry

        # Create butterfly using factory method
        butterfly = OptionStrategies.butterfly_call(
            self.option_symbol,
            upper_strike,
            atm_strike,
            lower_strike,
            expiry
        )

        # Execute as single atomic order
        self.buy(butterfly, 2)  # Buy 2 units of the strategy

    def find_atm_strike(self, calls):
        underlying_price = self.securities[self.option_symbol.underlying].price
        return min(calls, key=lambda x: abs(x.strike - underlying_price)).strike
```

**Key Benefits**:
- ✅ Automatic strategy detection and position grouping
- ✅ Single execution call for all legs
- ✅ Proper margin calculations for the strategy as a unit
- ✅ Simplified exit: `self.sell(butterfly, 2)` closes entire position

### 1.4 Using ComboOrders for Custom Multi-Leg Strategies

**When to Use ComboOrders**:
- Custom strategies not in the factory (e.g., ratio spreads)
- Need precise control over leg execution
- Building dynamic strategies based on market conditions

**Python ComboOrder Pattern**:

```python
from AlgorithmImports import *

def execute_custom_spread(self, contracts):
    """
    Execute a custom multi-leg strategy using ComboOrders.
    CRITICAL: Use ComboLimitOrder with NET pricing (not ComboLegLimitOrder).
    """
    # Define legs with quantities
    legs = [
        Leg.create(contracts[0].symbol, 1),   # Buy 1 lower strike
        Leg.create(contracts[1].symbol, -2),  # Sell 2 middle strike
        Leg.create(contracts[2].symbol, 1),   # Buy 1 upper strike
    ]

    # Calculate net debit/credit for the combo
    # DO NOT specify order_price in Leg.create() - not supported on Schwab
    net_debit = 1.90  # Target net price for entire combo

    # Execute atomically - all legs fill together or none fill
    tickets = self.combo_limit_order(legs, quantity=10, limit_price=net_debit)

    return tickets
```

**CRITICAL for Charles Schwab**:

✅ **SUPPORTED**:
- `ComboMarketOrder()` - Execute at market
- `ComboLimitOrder()` - Net limit price across all legs

❌ **NOT SUPPORTED**:
- `ComboLegLimitOrder()` - Individual leg limits (Schwab doesn't support this)
- Do NOT use `order_price` parameter in `Leg.create()` calls

**ComboOrder Benefits**:
- ✅ Atomic execution (all-or-nothing fills)
- ✅ Single commission per combo
- ✅ Prevents holding unbalanced positions
- ✅ Automatic strategy detection (if matches known pattern)

### 1.5 Complete Example: Iron Condor Implementation

**Source**: `Algorithm.Python/IronCondorStrategyAlgorithm.py`

```python
from AlgorithmImports import *

class IronCondorExample(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2020, 3, 1)
        self.set_cash(100000)

        option = self.add_option("SPY")
        self.option_symbol = option.symbol
        option.set_filter(-10, +10, 0, 90)

        self.iron_condor = None

    def on_data(self, slice):
        if self.portfolio.invested:
            return

        chain = slice.option_chains.get(self.option_symbol)
        if not chain:
            return

        # Separate calls and puts
        calls = [c for c in chain if c.right == OptionRight.CALL]
        puts = [c for c in chain if c.right == OptionRight.PUT]

        if len(calls) < 2 or len(puts) < 2:
            return

        # Get same expiration
        expiry = calls[0].expiry
        calls = [c for c in calls if c.expiry == expiry]
        puts = [c for c in puts if c.expiry == expiry]

        # Sort by strike
        calls = sorted(calls, key=lambda x: x.strike)
        puts = sorted(puts, key=lambda x: x.strike)

        # Select strikes for iron condor
        # Put side: two lowest strikes
        long_put_strike = puts[0].strike
        short_put_strike = puts[1].strike

        # Call side: find calls above short put strike
        calls_above = [c for c in calls if c.strike > short_put_strike]
        if len(calls_above) < 2:
            return

        short_call_strike = calls_above[0].strike
        long_call_strike = calls_above[1].strike

        # Create iron condor using factory
        self.iron_condor = OptionStrategies.iron_condor(
            self.option_symbol,
            long_put_strike,
            short_put_strike,
            short_call_strike,
            long_call_strike,
            expiry
        )

        # Execute all 4 legs atomically
        self.buy(self.iron_condor, 2)

    def on_order_event(self, order_event):
        self.debug(f"{order_event}")
```

**Strike Selection Logic**:
1. Separate chain into calls and puts
2. Filter for same expiration
3. Put side: Select 2 lowest strikes (long lower, short higher)
4. Call side: Select 2 strikes above short put (short lower, long higher)
5. Validate ascending order: long_put < short_put < short_call < long_call

**Position Validation** (from source):
```python
# Verify position group has exactly 4 positions
assert len(position_group.positions) == 4

# Verify quantities
for position in position_group.positions:
    if position.symbol.id.option_right == OptionRight.PUT:
        if position.symbol.id.strike_price == long_put_strike:
            assert position.quantity == 2  # Long put
        else:
            assert position.quantity == -2  # Short put
    else:  # Calls
        if position.symbol.id.strike_price == short_call_strike:
            assert position.quantity == -2  # Short call
        else:
            assert position.quantity == 2  # Long call
```

---

## 2. Risk Management Patterns

### 2.1 Margin Call Handling

**Source**: `Algorithm.CSharp/EquityMarginCallAlgorithm.cs`

QuantConnect provides built-in margin call events that trigger when portfolio margin falls below requirements.

**Implementation Pattern**:

```python
from AlgorithmImports import *

class MarginSafeAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        # Track margin events
        self.margin_warning_fired = False
        self.margin_call_fired = False

    def on_margin_call_warning(self):
        """
        Called when margin is approaching critical levels.
        Use this to reduce positions before forced liquidation.
        """
        self.margin_warning_fired = True
        self.debug("MARGIN WARNING: Consider reducing positions")

        # Proactive position reduction
        for symbol, security in self.portfolio.items():
            if security.invested:
                # Reduce position by 50%
                current_quantity = security.quantity
                self.market_order(symbol, -current_quantity * 0.5)

    def on_margin_call(self, requests):
        """
        Called when margin call is triggered.
        Receives list of liquidation requests.

        Args:
            requests: List of SubmitOrderRequest objects for liquidation
        """
        self.margin_call_fired = True

        for request in requests:
            self.debug(f"Margin call liquidation: {request.symbol} "
                      f"quantity={request.quantity}")

        # Default behavior: LEAN will execute these liquidations
        # Can customize by returning modified requests
        return requests

    def on_end_of_algorithm(self):
        # Verify margin restored after liquidations
        if self.portfolio.margin_remaining < 0:
            raise Exception("Margin not restored after liquidations")
```

**Key Monitoring Properties**:
- `Portfolio.MarginRemaining` - Available margin for new positions
- `Portfolio.TotalMarginUsed` - Current margin in use
- `Portfolio.TotalPortfolioValue` - Total account value
- `Portfolio[symbol].Holdings.UnrealizedProfit` - Per-position P&L

**Best Practices**:
1. ✅ Monitor `MarginRemaining` before each trade
2. ✅ Implement `OnMarginCallWarning()` for proactive management
3. ✅ Keep margin buffer (don't use 100% of available margin)
4. ✅ Validate margin calls only happen when exchange is open

### 2.2 Drawdown Tracking (Framework Implementation)

**Source**: `Algorithm.CSharp/MaximumDrawdownPercentPerSecurityFrameworkRegressionAlgorithm.cs`

While LEAN doesn't have a built-in circuit breaker, it provides risk management models in the Algorithm Framework.

**Framework Risk Management Pattern**:

```python
from AlgorithmImports import *

class DrawdownManagedAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        # Use Algorithm Framework
        self.set_universe_selection(ManualUniverseSelectionModel([Symbol.create("SPY", SecurityType.EQUITY, Market.USA)]))

        # Risk management: Maximum drawdown per security
        # This will liquidate positions that exceed drawdown threshold
        self.set_risk_management(MaximumDrawdownPercentPerSecurity(0.03))  # 3% max drawdown

        # Track equity peaks
        self.peak_equity = self.portfolio.total_portfolio_value

    def on_data(self, slice):
        # Update peak equity
        current_equity = self.portfolio.total_portfolio_value
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Calculate current drawdown
        drawdown = (self.peak_equity - current_equity) / self.peak_equity

        if drawdown > 0.10:  # 10% portfolio drawdown
            self.debug(f"WARNING: Portfolio drawdown at {drawdown:.2%}")
            # Could halt trading here
```

**Available Risk Management Models**:

| Model | Purpose | Parameters |
|-------|---------|------------|
| `MaximumDrawdownPercentPerSecurity` | Limit per-security drawdown | `max_drawdown_percent` |
| `MaximumDrawdownPercentPortfolio` | Limit portfolio drawdown | `max_drawdown_percent` |
| `MaximumSectorExposureRiskManagementModel` | Limit sector concentration | `max_exposure_percent` |
| `MaximumUnrealizedProfitPercentPerSecurity` | Take profits at threshold | `max_profit_percent` |

### 2.3 Custom Circuit Breaker Implementation

Since LEAN doesn't have a built-in circuit breaker, implement one using the existing `TradingCircuitBreaker` in your project.

**Location**: `models/circuit_breaker.py`

**Integration Pattern**:

```python
from AlgorithmImports import *
from models.circuit_breaker import create_circuit_breaker

class CircuitBreakerAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        # Create circuit breaker
        self.breaker = create_circuit_breaker(
            max_daily_loss=0.03,         # 3% daily loss limit
            max_drawdown=0.10,           # 10% max drawdown
            max_consecutive_losses=5,     # Halt after 5 losses
            require_human_reset=True      # Require manual reset
        )

        self.starting_equity = self.portfolio.cash
        self.peak_equity = self.starting_equity
        self.daily_starting_equity = self.starting_equity

    def on_data(self, slice):
        # Check if trading is allowed
        if not self.breaker.can_trade():
            self.debug("Trading halted by circuit breaker")
            return

        # Check daily loss
        current_equity = self.portfolio.total_portfolio_value
        daily_pnl_pct = (current_equity - self.daily_starting_equity) / self.daily_starting_equity
        self.breaker.check_daily_loss(daily_pnl_pct)

        # Check drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        self.breaker.check_drawdown(current_equity, self.peak_equity)

        # Your trading logic here
        if not self.breaker.can_trade():
            return

        # ... place orders ...

    def on_order_event(self, order_event):
        if order_event.status == OrderStatus.FILLED:
            # Determine if trade was winner or loser
            is_winner = order_event.fill_quantity * order_event.fill_price > 0
            self.breaker.record_trade_result(is_winner)

    def on_end_of_day(self, symbol):
        # Reset daily tracking
        self.daily_starting_equity = self.portfolio.total_portfolio_value
```

### 2.4 Position Sizing and Portfolio Limits

**Buying Power Model** for Options:

**Source**: `Common/Securities/Positions/PositionGroupBuyingPowerModel.cs`

```python
from AlgorithmImports import *

class PositionSizedAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        # Position sizing limits
        self.max_position_size = 0.25  # 25% max per position
        self.max_open_positions = 10

    def calculate_position_size(self, symbol, target_percent=0.10):
        """
        Calculate appropriate position size based on portfolio value.

        Args:
            symbol: Symbol to trade
            target_percent: Target allocation (default 10%)

        Returns:
            Number of contracts to trade
        """
        # Don't exceed max positions
        open_positions = sum(1 for s in self.portfolio.keys() if self.portfolio[s].invested)
        if open_positions >= self.max_open_positions:
            return 0

        # Calculate target dollar amount
        portfolio_value = self.portfolio.total_portfolio_value
        target_percent = min(target_percent, self.max_position_size)
        target_value = portfolio_value * target_percent

        # Get current price
        security = self.securities[symbol]
        price = security.price

        if price == 0:
            return 0

        # Calculate quantity
        # For options: price is per-share, contract is 100 shares
        multiplier = security.symbol_properties.contract_multiplier
        quantity = int(target_value / (price * multiplier))

        return max(1, quantity)  # At least 1 contract

    def check_buying_power(self, symbol, quantity):
        """
        Verify sufficient buying power before placing order.

        Returns:
            True if sufficient buying power, False otherwise
        """
        # Get security
        security = self.securities[symbol]

        # Calculate required margin
        order_value = abs(quantity) * security.price * security.symbol_properties.contract_multiplier

        # Check against available margin
        if self.portfolio.margin_remaining < order_value * 0.5:  # 50% margin buffer
            self.debug(f"Insufficient buying power for {symbol}: "
                      f"Required ~{order_value * 0.5}, Available {self.portfolio.margin_remaining}")
            return False

        return True
```

**Position Group Concepts**:

For multi-leg strategies, LEAN groups positions together and calculates margin as a unit:

- **Unit-based stepping**: Position quantities maintain consistent ratios
- **Iterative convergence**: Adjusts quantities in unit increments to meet margin targets
- **Maintenance vs Initial margin**: Distinguishes between current usage and new trade requirements

**Key Properties**:
- `Portfolio.TotalMarginUsed` - Current margin in use
- `Portfolio.MarginRemaining` - Available for new positions
- `Securities[symbol].BuyingPowerModel` - Per-security margin model

---

## 3. Universe Selection for Options

### 3.1 Basic Option Universe Selection

**Source**: `Algorithm.Framework/Selection/OptionUniverseSelectionModel.cs`

```python
from AlgorithmImports import *

class BasicOptionUniverse(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        # Method 1: Simple filter
        option = self.add_option("SPY")
        self.option_symbol = option.symbol

        # Filter by strikes and expiration
        option.set_filter(
            min_strike=-10,     # 10 strikes below current price
            max_strike=+10,     # 10 strikes above current price
            min_expiry=0,       # 0 days minimum
            max_expiry=90       # 90 days maximum
        )

        # Method 2: Lambda filter with custom logic
        option.set_filter(lambda u: u.strikes(-5, +5).expiration(30, 60))
```

### 3.2 Advanced Filtering with Greeks and IV

**Source**: `Common/Securities/Option/OptionFilterUniverse.cs`

QuantConnect provides extensive filtering capabilities including Greeks and implied volatility.

**Available Filter Methods**:

```python
from AlgorithmImports import *

class GreeksFilteredUniverse(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        option = self.add_option("SPY")
        self.option_symbol = option.symbol

        # Filter by Greeks and IV
        option.set_filter(lambda universe: universe
            .strikes(-10, +10)              # Strike range
            .expiration(30, 90)             # 30-90 days to expiration
            .calls_only()                   # Only call options
            .delta(0.25, 0.35)              # Delta between 0.25 and 0.35
            .implied_volatility(0.20, 0.50) # IV between 20% and 50%
            .open_interest(100, 999999)     # Minimum 100 open interest
        )

    def on_data(self, slice):
        chain = slice.option_chains.get(self.option_symbol)
        if not chain:
            return

        # Further filtering in OnData
        for contract in chain:
            # Access Greeks (available immediately as of LEAN PR #6720)
            delta = contract.greeks.delta
            gamma = contract.greeks.gamma
            theta = contract.greeks.theta_per_day  # Daily theta
            vega = contract.greeks.vega
            iv = contract.implied_volatility

            # Custom filtering logic
            if abs(delta) > 0.30 and iv > 0.25:
                self.debug(f"Opportunity: {contract.symbol} "
                          f"Delta={delta:.3f} IV={iv:.2%}")
```

**Complete Filter Method Reference**:

| Filter Method | Alias | Description | Example |
|---------------|-------|-------------|---------|
| `strikes(min, max)` | - | Relative strike offsets | `.strikes(-5, +5)` |
| `expiration(min, max)` | - | Days to expiration range | `.expiration(30, 90)` |
| `calls_only()` | - | Only call options | `.calls_only()` |
| `puts_only()` | - | Only put options | `.puts_only()` |
| `delta(min, max)` | `.d(min, max)` | Delta range | `.delta(0.25, 0.35)` |
| `gamma(min, max)` | `.g(min, max)` | Gamma range | `.gamma(0.01, 0.10)` |
| `theta(min, max)` | `.t(min, max)` | Theta range | `.theta(-0.05, -0.01)` |
| `vega(min, max)` | `.v(min, max)` | Vega range | `.vega(0.10, 0.50)` |
| `rho(min, max)` | `.r(min, max)` | Rho range | `.rho(-0.05, 0.05)` |
| `implied_volatility(min, max)` | `.iv(min, max)` | IV range | `.iv(0.20, 0.50)` |
| `open_interest(min, max)` | `.oi(min, max)` | Open interest range | `.oi(100, 999999)` |

**IMPORTANT**: Greeks filters are NOT supported for future options (will throw exception).

### 3.3 Strategy-Based Universe Selection

**Pre-built strategy filters** automatically select contracts for specific strategies:

```python
from AlgorithmImports import *

class StrategyUniverse(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        option = self.add_option("SPY")

        # Filter for iron condor opportunities
        option.set_filter(lambda u: u.iron_condor(
            days_to_expiry=45,
            wing_width=5
        ))

        # Other strategy filters available:
        # .call_spread(days_to_expiry, strike_width)
        # .put_spread(days_to_expiry, strike_width)
        # .straddle(days_to_expiry)
        # .strangle(days_to_expiry)
        # .butterfly_call(days_to_expiry)
        # .butterfly_put(days_to_expiry)
        # ... and more
```

### 3.4 Performance Optimization

**Source**: `Common/Securities/Option/OptionFilterUniverse.cs`

**Best Practices for High-Performance Filtering**:

```python
from AlgorithmImports import *

class OptimizedFilterAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        option = self.add_option("SPY")

        # 1. Use narrow filters to reduce data
        option.set_filter(lambda u: u
            .strikes(-5, +5)       # Narrow strike range
            .expiration(30, 60)    # Narrow expiration window
            .delta(0.25, 0.40)     # Specific delta range
        )

    def on_data(self, slice):
        chain = slice.option_chains.get(self.option_symbol)
        if not chain:
            return

        # 2. Cache expensive calculations
        if not hasattr(self, '_cached_strikes'):
            self._cached_strikes = {}

        # 3. Use sorted() once, then index
        contracts = sorted(chain, key=lambda x: (x.expiry, x.strike))

        # 4. Break early when possible
        for contract in contracts:
            if contract.greeks.delta > 0.35:
                break  # Stop processing once delta exceeds threshold

            # Process contract
            pass

        # 5. Batch orders when possible (use combo orders)
        # This reduces API calls and improves execution
```

**Performance Tips from Source Code**:

1. **Strike caching**: LEAN caches unique strikes and only regenerates when exchange date changes
2. **Binary search**: Uses binary search on ordered lists for strike filtering
3. **Lazy evaluation**: Computation defers until enumeration
4. **Aggressive inlining**: Validation methods use aggressive inlining for speed
5. **Deterministic ordering**: Always orders by symbol ID for consistency

---

## 4. Data Handling Best Practices

### 4.1 Defensive Data Access Patterns

**Source**: `Common/Data/Slice.cs`, `Algorithm.CSharp/BasicTemplateOptionsAlgorithm.cs`

#### Pattern 1: TryGetValue (Recommended)

```python
from AlgorithmImports import *

class DefensiveDataAlgorithm(QCAlgorithm):
    def on_data(self, slice):
        # Pattern 1: TryGetValue for option chains (Python uses .get())
        chain = slice.option_chains.get(self.option_symbol)
        if not chain:
            return  # No data available, skip

        # Pattern 2: ContainsKey check (available but less Pythonic)
        if not slice.option_chains.contains_key(self.option_symbol):
            return

        # Pattern 3: Check for specific contract data
        for contract in chain:
            # Verify quote data exists
            if not slice.quote_bars.contains_key(contract.symbol):
                continue  # Skip contracts without quote data

            quote = slice.quote_bars[contract.symbol]

            # Always validate data before use
            if quote.bid.close == 0 or quote.ask.close == 0:
                continue  # Invalid pricing data

            # Safe to use
            mid_price = (quote.bid.close + quote.ask.close) / 2
```

#### Pattern 2: Portfolio State Validation

```python
def on_data(self, slice):
    # Check market is open before trading
    if not self.is_market_open(self.option_symbol):
        return

    # Check portfolio state
    if self.portfolio.invested:
        return  # Already have position

    # Check sufficient buying power
    if self.portfolio.margin_remaining < 10000:
        self.debug("Insufficient buying power")
        return

    # Proceed with trading logic
    pass
```

#### Pattern 3: Indicator Readiness Checks

```python
from AlgorithmImports import *

class IndicatorSafeAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        # Enable automatic indicator warmup
        self.settings.automatic_indicator_warm_up = True

        spy = self.add_equity("SPY")
        self.spy = spy.symbol

        # Create indicators (will auto-warmup)
        self.sma = self.sma(self.spy, 20)
        self.rsi = self.rsi(self.spy, 14)

    def on_data(self, slice):
        # Always check IsWarmingUp
        if self.is_warming_up:
            return

        # Always check indicator IsReady
        if not self.sma.is_ready or not self.rsi.is_ready:
            return

        # Safe to use indicator values
        if self.sma.current.value > self.securities[self.spy].price:
            # Trading logic
            pass
```

### 4.2 Warmup Best Practices

**Source**: `Algorithm.CSharp/AutomaticIndicatorWarmupRegressionAlgorithm.cs`

#### Automatic Warmup (Recommended)

```python
from AlgorithmImports import *

class AutoWarmupAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        # Method 1: Enable automatic warmup globally
        self.settings.automatic_indicator_warm_up = True

        spy = self.add_equity("SPY", Resolution.DAILY)
        self.spy = spy.symbol

        # Indicators created after this will auto-warmup
        self.sma = self.sma(self.spy, 50)  # Automatically warmed up
        self.rsi = self.rsi(self.spy, 14)  # Automatically warmed up

        # Verify warmup completed (for custom indicators)
        if not self.sma.is_ready:
            raise Exception("SMA should be ready after auto-warmup")

    def on_data(self, slice):
        # Check warming up status
        if self.is_warming_up:
            return  # Skip during warmup period

        # Indicators are ready
        sma_value = self.sma.current.value
        rsi_value = self.rsi.current.value
```

#### Manual Warmup

```python
from AlgorithmImports import *
from QuantConnect.Indicators import *

class ManualWarmupAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        spy = self.add_equity("SPY", Resolution.DAILY)
        self.spy = spy.symbol

        # Create custom indicator
        self.custom_indicator = SimpleMovingAverage(50)

        # Register indicator manually
        self.register_indicator(self.spy, self.custom_indicator, Resolution.DAILY)

        # Warm up the indicator explicitly
        self.warm_up_indicator(self.spy, self.custom_indicator)

        # Verify it's ready
        if not self.custom_indicator.is_ready:
            raise Exception("Indicator should be warmed up")

    def on_data(self, slice):
        if self.is_warming_up:
            return

        if not self.custom_indicator.is_ready:
            return

        # Use indicator
        value = self.custom_indicator.current.value
```

#### SetWarmup for Algorithm-Level Warmup

```python
from AlgorithmImports import *

class AlgorithmWarmupExample(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 2, 1)  # Start date
        self.set_cash(100000)

        # Warm up algorithm for 50 days before start date
        self.set_warm_up(50)

        # OR: Warm up for specific time span
        # self.set_warm_up(timedelta(days=50))

        spy = self.add_equity("SPY", Resolution.DAILY)
        self.spy = spy.symbol

        self.sma = self.sma(self.spy, 20)

    def on_data(self, slice):
        # IsWarmingUp property tracks warmup status
        if self.is_warming_up:
            # Still warming up - indicators receiving historical data
            return

        # Warmup complete - begin live trading logic
        if self.sma.is_ready:
            # Trading logic here
            pass
```

### 4.3 Greeks Access (Updated for LEAN PR #6720)

**CRITICAL**: As of LEAN PR #6720, Greeks use implied volatility and require **NO warmup**.

```python
from AlgorithmImports import *

class GreeksAccessAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        option = self.add_option("SPY")
        self.option_symbol = option.symbol
        option.set_filter(-10, +10, 0, 90)

        # Optional: Set custom pricing model (defaults to Black-Scholes/Bjerksund-Stensland)
        # option.price_model = OptionPriceModels.bjerksund_stensland()

        # NO WARMUP REQUIRED FOR GREEKS (as of PR #6720)

    def on_data(self, slice):
        chain = slice.option_chains.get(self.option_symbol)
        if not chain:
            return

        for contract in chain:
            # Greeks available immediately (IV-based calculation)
            delta = contract.greeks.delta
            gamma = contract.greeks.gamma
            theta = contract.greeks.theta
            theta_per_day = contract.greeks.theta_per_day  # Daily theta decay
            vega = contract.greeks.vega
            rho = contract.greeks.rho
            iv = contract.implied_volatility

            # Greeks values match Interactive Brokers and major brokerages
            self.debug(f"{contract.symbol}: "
                      f"Delta={delta:.3f}, "
                      f"Gamma={gamma:.4f}, "
                      f"Theta/day={theta_per_day:.4f}, "
                      f"IV={iv:.2%}")

            # Use in trading logic
            if 0.25 < abs(delta) < 0.35 and iv > 0.20:
                # Trade logic for delta-neutral strategies
                pass
```

**Available Pricing Models**:

**Source**: `Common/Securities/Option/OptionPriceModels.cs`

```python
# Analytical models
option.price_model = OptionPriceModels.black_scholes()  # European (default)

# Approximation models
option.price_model = OptionPriceModels.bjerksund_stensland()  # American (default)
option.price_model = OptionPriceModels.barone_adesi_whaley()  # American

# Numerical models
option.price_model = OptionPriceModels.crank_nicolson_fd()  # Finite differences
option.price_model = OptionPriceModels.binomial_cox_ross_rubinstein()
option.price_model = OptionPriceModels.binomial_jarrow_rudd()
option.price_model = OptionPriceModels.binomial_joshi()
# ... many other binomial models available
```

### 4.4 Look-Ahead Bias Prevention

While no specific documentation was found in LEAN for "look-ahead bias", here are essential patterns:

**Critical Rules**:

```python
from AlgorithmImports import *

class NoBiasAlgorithm(QCAlgorithm):
    def on_data(self, slice):
        # ❌ WRONG: Using future data
        # Don't access data beyond current slice
        # Don't peek at next bar's open/close

        # ✅ CORRECT: Only use current and past data
        chain = slice.option_chains.get(self.option_symbol)
        if not chain:
            return

        for contract in chain:
            # Only use data available at this moment
            current_price = contract.bid_price  # Current bid

            # Don't use contract.close if it's not yet available
            # Don't use future expiration outcomes in current decision

            # Use historical data from self.history() safely
            history = self.history(contract.symbol, 20, Resolution.DAILY)

            # Make decisions based only on past/current data
            if current_price < history['close'].mean():
                # This is safe - using historical mean
                pass

    def on_order_event(self, order_event):
        # ❌ WRONG: Don't use fill price to generate new signals
        # This would create look-ahead bias in backtesting

        # ✅ CORRECT: Use fills for position management only
        if order_event.status == OrderStatus.FILLED:
            # Record fill for tracking
            self.debug(f"Filled at {order_event.fill_price}")
```

**Warmup Prevents Look-Ahead Bias**:
- Indicators receive historical data during warmup
- Algorithm starts after warmup completes
- No future data leakage into indicator calculations

---

## 5. Implementation Recommendations

### 5.1 For Your Two-Part Spread Strategy

Based on your strategy documentation in `docs/strategies/TWO_PART_SPREAD_STRATEGY.md`:

#### Use ComboOrders for Your Strategy

```python
from AlgorithmImports import *

class TwoPartSpreadStrategy(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        option = self.add_option("SPY")
        self.option_symbol = option.symbol

        # Wide strikes for opportunities
        option.set_filter(-20, +20, 30, 180)

        # Track pending orders
        self.pending_debit_spreads = {}
        self.pending_credit_spreads = {}

    def on_data(self, slice):
        chain = slice.option_chains.get(self.option_symbol)
        if not chain:
            return

        # Find underpriced debit spread
        debit_spread = self.find_debit_spread(chain)
        if debit_spread:
            self.execute_debit_spread(debit_spread)

    def find_debit_spread(self, chain):
        """
        Find debit spread with wide bid-ask (opportunity for below-mid fill).
        """
        contracts = sorted(chain, key=lambda x: (x.expiry, x.strike))
        calls = [c for c in contracts if c.right == OptionRight.CALL]

        for i in range(len(calls) - 1):
            lower = calls[i]
            upper = calls[i + 1]

            # Calculate spread metrics
            lower_mid = (lower.bid_price + lower.ask_price) / 2
            upper_mid = (upper.bid_price + upper.ask_price) / 2

            # Wide spread = opportunity
            lower_spread_width = lower.ask_price - lower.bid_price
            if lower_spread_width < 0.50:  # Need wide spread
                continue

            # Return debit spread opportunity
            return {
                'lower': lower,
                'upper': upper,
                'expected_debit': lower_mid - upper_mid,
            }

        return None

    def execute_debit_spread(self, spread):
        """
        Execute debit spread at 35% from bid with 2.5s quick cancel.
        """
        lower = spread['lower']
        upper = spread['upper']

        # Create legs
        legs = [
            Leg.create(lower.symbol, 1),   # Buy lower strike
            Leg.create(upper.symbol, -1),  # Sell upper strike
        ]

        # Calculate aggressive limit (35% from bid toward mid)
        lower_bid = lower.bid_price
        lower_mid = (lower.bid_price + lower.ask_price) / 2
        lower_target = lower_bid + (lower_mid - lower_bid) * 0.35

        upper_bid = upper.bid_price
        upper_mid = (upper.bid_price + upper.ask_price) / 2
        upper_target = upper_bid + (upper_mid - upper_bid) * 0.35

        net_debit = lower_target - upper_target

        # Execute combo order
        tickets = self.combo_limit_order(legs, quantity=1, limit_price=net_debit)

        # Track for 2.5s cancellation
        self.pending_debit_spreads[tickets[0].order_id] = {
            'tickets': tickets,
            'submit_time': self.time,
            'spread': spread,
        }

        self.debug(f"Debit spread submitted: net_debit={net_debit:.2f}")

    def on_order_event(self, order_event):
        """Handle fills and implement 2.5s cancellation."""
        # Check for debit spread fills
        if order_event.order_id in self.pending_debit_spreads:
            spread_info = self.pending_debit_spreads[order_event.order_id]

            if order_event.status == OrderStatus.FILLED:
                # Debit spread filled - now find credit spread
                self.debug(f"Debit spread filled at {order_event.fill_price}")
                self.find_credit_spread(spread_info['spread'])
                del self.pending_debit_spreads[order_event.order_id]

            elif order_event.status == OrderStatus.CANCELED:
                # Canceled - clean up
                del self.pending_debit_spreads[order_event.order_id]

    def scheduled_event(self):
        """Run every second to check for 2.5s cancellations."""
        current_time = self.time

        to_cancel = []
        for order_id, info in self.pending_debit_spreads.items():
            elapsed = (current_time - info['submit_time']).total_seconds()

            if elapsed > 2.5:
                # Cancel unfilled orders after 2.5 seconds
                for ticket in info['tickets']:
                    if ticket.status == OrderStatus.SUBMITTED or ticket.status == OrderStatus.PARTIALLY_FILLED:
                        ticket.cancel("2.5s timeout")

                to_cancel.append(order_id)

        for order_id in to_cancel:
            if order_id in self.pending_debit_spreads:
                del self.pending_debit_spreads[order_id]
```

**Key Implementation Points**:

1. ✅ Use `ComboLimitOrder` with net pricing (Schwab supported)
2. ✅ Wide bid-ask spreads are opportunities (can get filled below mid)
3. ✅ 2.5 second quick cancel for unfilled orders
4. ✅ Random delays between 3-15 seconds to avoid detection
5. ✅ 1 contract at a time for highest fill probability
6. ✅ Balance positions per option chain

### 5.2 Risk Management Integration

Integrate existing `RiskManager` and `TradingCircuitBreaker`:

```python
from AlgorithmImports import *
from models.risk_manager import RiskManager, RiskLimits
from models.circuit_breaker import create_circuit_breaker

class ProductionAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        # Risk management
        limits = RiskLimits(
            max_position_size=0.25,      # 25% max per position
            max_daily_loss=0.03,         # 3% daily loss limit
            max_drawdown=0.10,           # 10% max drawdown
            max_risk_per_trade=0.02,     # 2% risk per trade
        )
        self.risk_manager = RiskManager(
            starting_equity=self.portfolio.cash,
            limits=limits
        )

        # Circuit breaker
        self.breaker = create_circuit_breaker(
            max_daily_loss=0.03,
            max_drawdown=0.10,
            max_consecutive_losses=5,
            require_human_reset=True
        )

        # Initialize options
        option = self.add_option("SPY")
        self.option_symbol = option.symbol
        option.set_filter(-10, +10, 30, 90)

    def on_data(self, slice):
        # Check circuit breaker
        if not self.breaker.can_trade():
            return

        # Update risk tracking
        current_equity = self.portfolio.total_portfolio_value
        self.risk_manager.update(current_equity)

        # Check risk limits
        if not self.risk_manager.can_trade():
            self.debug("Risk limits exceeded")
            return

        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            symbol=self.option_symbol,
            entry_price=100.0,  # Example
            stop_loss_price=95.0  # Example
        )

        if position_size == 0:
            return

        # Execute strategy
        # ... trading logic ...
```

### 5.3 Options Scanner Integration

Integrate your existing scanners with LEAN patterns:

```python
from AlgorithmImports import *
from scanners.options_scanner import create_options_scanner
from config import get_config

class ScannerIntegratedAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        # Load config
        config = get_config()

        # Create scanner
        self.options_scanner = create_options_scanner(
            config.get_options_scanner_config()
        )

        # Add options universe
        option = self.add_option("SPY")
        self.option_symbol = option.symbol

        # Use scanner's filter criteria
        scanner_config = config.get("options_scanner")
        option.set_filter(lambda u: u
            .delta(
                scanner_config.get("min_delta"),
                scanner_config.get("max_delta")
            )
            .implied_volatility(
                scanner_config.get("min_iv"),
                scanner_config.get("max_iv")
            )
            .expiration(
                scanner_config.get("min_dte"),
                scanner_config.get("max_dte")
            )
        )

    def on_data(self, slice):
        chain = slice.option_chains.get(self.option_symbol)
        if not chain:
            return

        # Convert chain to scanner format
        underlying_price = self.securities[self.option_symbol.underlying].price

        # Use scanner to find opportunities
        opportunities = self.options_scanner.scan_chain(
            underlying="SPY",
            spot_price=underlying_price,
            chain=list(chain),  # Convert to list
        )

        for opp in opportunities:
            self.debug(f"Opportunity: {opp.contract.symbol} "
                      f"Underpriced by {opp.underpriced_pct:.1%}")

            # Execute trade
            self.market_order(opp.contract.symbol, 1)
```

---

## 6. Common Pitfalls to Avoid

### 6.1 Data Access Errors

❌ **WRONG**:
```python
def on_data(self, slice):
    # Direct access without checking
    chain = slice.option_chains[self.option_symbol]  # KeyError if no data!

    for contract in chain:
        # Using data without validation
        price = contract.bid_price  # Could be 0 or invalid
```

✅ **CORRECT**:
```python
def on_data(self, slice):
    # Safe access with validation
    chain = slice.option_chains.get(self.option_symbol)
    if not chain:
        return

    for contract in chain:
        # Validate data before use
        if contract.bid_price == 0 or contract.ask_price == 0:
            continue

        # Safe to use
        mid_price = (contract.bid_price + contract.ask_price) / 2
```

### 6.2 Indicator Warmup Mistakes

❌ **WRONG**:
```python
def initialize(self):
    self.sma = self.sma(self.spy, 50)
    # Missing warmup!

def on_data(self, slice):
    # Using indicator before ready
    if self.sma.current.value > 100:  # Could be wrong!
        pass
```

✅ **CORRECT**:
```python
def initialize(self):
    self.settings.automatic_indicator_warm_up = True
    self.sma = self.sma(self.spy, 50)

def on_data(self, slice):
    if self.is_warming_up or not self.sma.is_ready:
        return

    # Safe to use
    if self.sma.current.value > 100:
        pass
```

### 6.3 ComboOrder Mistakes for Charles Schwab

❌ **WRONG**:
```python
# Individual leg limits - NOT supported on Schwab
legs = [
    Leg.create(call1, 1, order_price=5.00),  # order_price NOT supported!
    Leg.create(call2, -1, order_price=3.00),
]
tickets = self.combo_leg_limit_order(legs, 1)  # ComboLegLimitOrder NOT on Schwab!
```

✅ **CORRECT**:
```python
# Net limit price - Schwab supported
legs = [
    Leg.create(call1, 1),   # No order_price parameter
    Leg.create(call2, -1),
]
net_limit = 2.00  # Net debit/credit for entire combo
tickets = self.combo_limit_order(legs, 1, limit_price=net_limit)
```

### 6.4 Greeks Access Mistakes

❌ **WRONG** (before PR #6720):
```python
def on_data(self, slice):
    chain = slice.option_chains.get(self.option_symbol)

    for contract in chain:
        # Using Greeks without checking IsReady (old pattern)
        delta = contract.greeks.delta  # Might be 0 without warmup
```

✅ **CORRECT** (after PR #6720):
```python
def on_data(self, slice):
    chain = slice.option_chains.get(self.option_symbol)

    for contract in chain:
        # Greeks available immediately (IV-based, no warmup needed)
        delta = contract.greeks.delta
        iv = contract.implied_volatility

        # Can use immediately
        if abs(delta) > 0.30:
            pass
```

### 6.5 Position Sizing Errors

❌ **WRONG**:
```python
def on_data(self, slice):
    # No buying power check
    self.market_order(symbol, 100)  # Might fail!
```

✅ **CORRECT**:
```python
def on_data(self, slice):
    # Check buying power first
    if self.portfolio.margin_remaining < 10000:
        return

    # Calculate appropriate size
    quantity = self.calculate_position_size(symbol)

    if quantity > 0:
        self.market_order(symbol, quantity)
```

### 6.6 Multi-Leg Strategy Errors

❌ **WRONG**:
```python
# Executing legs individually
self.market_order(call1, 1)   # Leg 1
self.market_order(call2, -2)  # Leg 2
self.market_order(call3, 1)   # Leg 3
# Risk: Legs might fill at different prices or partially
```

✅ **CORRECT**:
```python
# Execute atomically with combo order
legs = [
    Leg.create(call1, 1),
    Leg.create(call2, -2),
    Leg.create(call3, 1),
]
tickets = self.combo_limit_order(legs, quantity=1, limit_price=net_debit)
# All legs fill together or none fill
```

---

## 7. Quick Reference

### 7.1 Option Strategy Factory Methods

```python
# Single-leg
OptionStrategies.naked_call(option_symbol, strike, expiry)
OptionStrategies.covered_call(option_symbol, strike, expiry)

# Two-leg spreads
OptionStrategies.bull_call_spread(option_symbol, lower_strike, upper_strike, expiry)
OptionStrategies.bear_put_spread(option_symbol, higher_strike, lower_strike, expiry)

# Three-leg
OptionStrategies.butterfly_call(option_symbol, upper, middle, lower, expiry)
OptionStrategies.straddle(option_symbol, strike, expiry)
OptionStrategies.strangle(option_symbol, call_strike, put_strike, expiry)

# Four-leg
OptionStrategies.iron_condor(option_symbol, lp_strike, sp_strike, sc_strike, lc_strike, expiry)
```

### 7.2 ComboOrder Methods

```python
# Market execution
tickets = self.combo_market_order(legs, quantity)

# Limit order with net pricing (Schwab supported)
tickets = self.combo_limit_order(legs, quantity, limit_price=net_limit)

# Creating legs
leg = Leg.create(symbol, quantity)  # No order_price for Schwab
```

### 7.3 Filter Methods

```python
option.set_filter(lambda u: u
    .strikes(-10, +10)               # Strike range
    .expiration(30, 90)              # DTE range
    .calls_only()                    # or .puts_only()
    .delta(0.25, 0.35)               # Delta range
    .implied_volatility(0.20, 0.50)  # IV range
    .open_interest(100, 999999)      # OI minimum
)
```

### 7.4 Data Access Patterns

```python
# Option chains
chain = slice.option_chains.get(symbol)
if not chain:
    return

# Individual contracts
for contract in chain:
    if contract.bid_price == 0:
        continue

    # Greeks (available immediately)
    delta = contract.greeks.delta
    iv = contract.implied_volatility
```

### 7.5 Indicator Patterns

```python
# Enable auto-warmup
self.settings.automatic_indicator_warm_up = True

# Create indicator
self.sma = self.sma(symbol, period)

# Check in OnData
if self.is_warming_up or not self.sma.is_ready:
    return
```

### 7.6 Risk Management Checks

```python
# Circuit breaker
if not self.breaker.can_trade():
    return

# Buying power
if self.portfolio.margin_remaining < threshold:
    return

# Position limits
if len([s for s in self.portfolio.keys() if self.portfolio[s].invested]) >= max_positions:
    return
```

---

## Key GitHub Repository Links

### Algorithm Examples

**Multi-Leg Strategies**:
- [C# Examples](https://github.com/QuantConnect/Lean/tree/master/Algorithm.CSharp)
- [Python Examples](https://github.com/QuantConnect/Lean/tree/master/Algorithm.Python)
- [Butterfly Implementation](https://github.com/QuantConnect/Lean/blob/master/Algorithm.Python/LongAndShortButterflyCallStrategiesAlgorithm.py)
- [Iron Condor Implementation](https://github.com/QuantConnect/Lean/blob/master/Algorithm.Python/IronCondorStrategyAlgorithm.py)
- [ComboOrder Demo](https://github.com/QuantConnect/Lean/blob/master/Algorithm.Python/ComboOrderTicketDemoAlgorithm.py)

### Core Framework Files

**Options Trading**:
- [OptionStrategies Factory](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/Option/OptionStrategies.cs)
- [OptionStrategyDefinitions](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/Option/StrategyMatcher/OptionStrategyDefinitions.cs)
- [OptionFilterUniverse](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/Option/OptionFilterUniverse.cs)
- [OptionPriceModels](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/Option/OptionPriceModels.cs)

**Data Handling**:
- [Slice.cs](https://github.com/QuantConnect/Lean/blob/master/Common/Data/Slice.cs)
- [QCAlgorithm.cs](https://github.com/QuantConnect/Lean/blob/master/Algorithm/QCAlgorithm.cs)

**Orders**:
- [ComboLimitOrder](https://github.com/QuantConnect/Lean/blob/master/Common/Orders/ComboLimitOrder.cs)
- [ComboMarketOrder](https://github.com/QuantConnect/Lean/blob/master/Common/Orders/ComboMarketOrder.cs)

**Risk Management**:
- [PositionGroupBuyingPowerModel](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/Positions/PositionGroupBuyingPowerModel.cs)
- [EquityMarginCallAlgorithm](https://github.com/QuantConnect/Lean/blob/master/Algorithm.CSharp/EquityMarginCallAlgorithm.cs)

### Testing Files

**Warmup Examples**:
- [AutomaticIndicatorWarmup](https://github.com/QuantConnect/Lean/blob/master/Algorithm.CSharp/AutomaticIndicatorWarmupRegressionAlgorithm.cs)

**Data Validation**:
- [BasicTemplateOptionsAlgorithm.cs](https://github.com/QuantConnect/Lean/blob/master/Algorithm.CSharp/BasicTemplateOptionsAlgorithm.cs)
- [BasicTemplateOptionsAlgorithm.py](https://github.com/QuantConnect/Lean/blob/master/Algorithm.Python/BasicTemplateOptionsAlgorithm.py)

---

## Summary

### Key Findings

1. **Multi-Leg Strategies**: QuantConnect has 37+ pre-defined strategies with automatic detection and atomic execution via `OptionStrategies` factory or `ComboOrders`.

2. **Charles Schwab Support**: Confirmed support for `ComboMarketOrder` and `ComboLimitOrder` with NET pricing. Individual leg limits (`ComboLegLimitOrder`) NOT supported.

3. **Greeks Access**: As of LEAN PR #6720, Greeks use IV and require NO warmup. Available immediately upon option data arrival.

4. **Risk Management**: No built-in circuit breaker, but margin call events and framework risk models available. Use custom `TradingCircuitBreaker` from your project.

5. **Universe Selection**: Extensive filtering capabilities including Greeks, IV, open interest, with performance optimizations (caching, binary search).

6. **Data Handling**: Critical patterns include `TryGetValue`/`.get()`, `IsWarmingUp`, `IsReady` checks, and defensive validation.

### Production-Ready Recommendations

1. ✅ Use `OptionStrategies` factory for standard strategies (butterflies, condors)
2. ✅ Use `ComboLimitOrder` with net pricing for custom strategies on Schwab
3. ✅ Enable `automatic_indicator_warm_up` globally
4. ✅ Always check `IsWarmingUp` and `IsReady` before using indicators
5. ✅ Implement custom circuit breaker using existing `TradingCircuitBreaker`
6. ✅ Use defensive data access (`chain.get()` instead of `chain[]`)
7. ✅ Filter options by Greeks and IV for better opportunity selection
8. ✅ Monitor `MarginRemaining` and implement `OnMarginCall` handlers
9. ✅ Validate position groups after multi-leg strategy execution
10. ✅ Use atomic combo orders to prevent unbalanced positions

---

**End of Report**
