# QuantConnect Quick Reference Guide

**Last Updated**: 2025-11-30

This is a quick reference for common patterns when developing algorithms for QuantConnect with Charles Schwab brokerage.

---

## Table of Contents

1. [Multi-Leg Options Execution](#multi-leg-options-execution)
2. [Data Access Patterns](#data-access-patterns)
3. [Greeks and Indicators](#greeks-and-indicators)
4. [Risk Management](#risk-management)
5. [Common Mistakes](#common-mistakes)

---

## Multi-Leg Options Execution

### ComboOrders for Schwab (RECOMMENDED)

**Schwab supports**: `ComboMarketOrder` and `ComboLimitOrder` with NET pricing
**Schwab does NOT support**: `ComboLegLimitOrder` with individual leg limits

```python
from AlgorithmImports import *

# Create legs (no order_price parameter for Schwab)
legs = [
    Leg.create(lower_call, 1),   # Buy lower strike
    Leg.create(upper_call, -1),  # Sell upper strike
]

# Calculate net debit/credit for entire combo
net_debit = 1.90

# Execute atomically - all legs fill together or none fill
tickets = self.combo_limit_order(legs, quantity=1, limit_price=net_debit)

# Market order alternative
tickets = self.combo_market_order(legs, quantity=1)
```

### Strategy Factory (ALTERNATIVE)

Use for standard strategies (butterflies, condors, spreads):

```python
from AlgorithmImports import *

# Butterfly spread
butterfly = OptionStrategies.butterfly_call(
    option_symbol,
    upper_strike,
    middle_strike,
    lower_strike,
    expiry
)
self.buy(butterfly, 2)  # Execute all legs atomically

# Iron condor
condor = OptionStrategies.iron_condor(
    option_symbol,
    long_put_strike,
    short_put_strike,
    short_call_strike,
    long_call_strike,
    expiry
)
self.buy(condor, 1)

# Exit entire strategy
self.sell(butterfly, 2)
```

---

## Data Access Patterns

### Safe Option Chain Access

```python
from AlgorithmImports import *

def on_data(self, slice):
    # ALWAYS use .get() or check existence
    chain = slice.option_chains.get(self.option_symbol)
    if not chain:
        return  # No data available

    # Check market is open
    if not self.is_market_open(self.option_symbol):
        return

    # Validate each contract
    for contract in chain:
        # Skip invalid pricing data
        if contract.bid_price == 0 or contract.ask_price == 0:
            continue

        # Safe to use
        mid_price = (contract.bid_price + contract.ask_price) / 2
```

### Defensive Data Validation

```python
# Check portfolio state before trading
if self.portfolio.invested:
    return

# Verify sufficient buying power
if self.portfolio.margin_remaining < 10000:
    self.debug("Insufficient buying power")
    return

# Validate quote data exists
if not slice.quote_bars.contains_key(contract.symbol):
    continue

quote = slice.quote_bars[contract.symbol]
```

---

## Greeks and Indicators

### Greeks Access (No Warmup Required)

**As of LEAN PR #6720**: Greeks use implied volatility and are available IMMEDIATELY.

```python
from AlgorithmImports import *

def on_data(self, slice):
    chain = slice.option_chains.get(self.option_symbol)
    if not chain:
        return

    for contract in chain:
        # Greeks available immediately (no warmup needed)
        delta = contract.greeks.delta
        gamma = contract.greeks.gamma
        theta = contract.greeks.theta
        theta_per_day = contract.greeks.theta_per_day  # Daily theta decay
        vega = contract.greeks.vega
        rho = contract.greeks.rho
        iv = contract.implied_volatility

        # Use in trading logic immediately
        if 0.25 < abs(delta) < 0.35 and iv > 0.20:
            # Trade logic
            pass
```

### Indicator Warmup (Automatic)

```python
from AlgorithmImports import *

class MyAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        # Enable automatic warmup globally
        self.settings.automatic_indicator_warm_up = True

        spy = self.add_equity("SPY", Resolution.DAILY)
        self.spy = spy.symbol

        # Indicators will auto-warmup
        self.sma = self.sma(self.spy, 50)
        self.rsi = self.rsi(self.spy, 14)

    def on_data(self, slice):
        # Always check warmup status
        if self.is_warming_up:
            return

        # Always check indicator ready
        if not self.sma.is_ready or not self.rsi.is_ready:
            return

        # Safe to use
        sma_value = self.sma.current.value
        rsi_value = self.rsi.current.value
```

---

## Risk Management

### Circuit Breaker Integration

```python
from AlgorithmImports import *
from models.circuit_breaker import create_circuit_breaker

class SafeAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_cash(100000)

        # Create circuit breaker
        self.breaker = create_circuit_breaker(
            max_daily_loss=0.03,         # 3% daily loss limit
            max_drawdown=0.10,           # 10% max drawdown
            max_consecutive_losses=5,    # Halt after 5 losses
            require_human_reset=True
        )

        self.starting_equity = self.portfolio.cash
        self.peak_equity = self.starting_equity
        self.daily_starting_equity = self.starting_equity

    def on_data(self, slice):
        # Check if trading allowed
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

        # Your trading logic
        if not self.breaker.can_trade():
            return

        # ... place orders ...

    def on_end_of_day(self, symbol):
        # Reset daily tracking
        self.daily_starting_equity = self.portfolio.total_portfolio_value
```

### Margin Call Handling

```python
from AlgorithmImports import *

class MarginSafeAlgorithm(QCAlgorithm):
    def on_margin_call_warning(self):
        """Called when margin approaches critical levels."""
        self.debug("MARGIN WARNING: Reducing positions")

        # Reduce positions by 50%
        for symbol, security in self.portfolio.items():
            if security.invested:
                current_quantity = security.quantity
                self.market_order(symbol, -current_quantity * 0.5)

    def on_margin_call(self, requests):
        """Called when margin call is triggered."""
        for request in requests:
            self.debug(f"Margin call: {request.symbol} qty={request.quantity}")

        # LEAN will execute these liquidations
        return requests

    def on_data(self, slice):
        # Always check margin before trading
        if self.portfolio.margin_remaining < 10000:
            self.debug("Insufficient margin")
            return

        # Trading logic
        pass
```

### Position Sizing

```python
from AlgorithmImports import *

class PositionSizedAlgorithm(QCAlgorithm):
    def initialize(self):
        self.max_position_size = 0.25  # 25% max per position
        self.max_open_positions = 10

    def calculate_position_size(self, symbol, target_percent=0.10):
        """Calculate appropriate position size."""
        # Check max positions
        open_positions = sum(1 for s in self.portfolio.keys()
                           if self.portfolio[s].invested)
        if open_positions >= self.max_open_positions:
            return 0

        # Calculate target value
        portfolio_value = self.portfolio.total_portfolio_value
        target_percent = min(target_percent, self.max_position_size)
        target_value = portfolio_value * target_percent

        # Get current price
        security = self.securities[symbol]
        price = security.price
        if price == 0:
            return 0

        # Calculate quantity (for options: account for multiplier)
        multiplier = security.symbol_properties.contract_multiplier
        quantity = int(target_value / (price * multiplier))

        return max(1, quantity)
```

---

## Common Mistakes

### ❌ WRONG: Individual Leg Limits on Schwab

```python
# Individual leg limits - NOT supported on Schwab
legs = [
    Leg.create(call1, 1, order_price=5.00),  # order_price NOT supported!
    Leg.create(call2, -1, order_price=3.00),
]
tickets = self.combo_leg_limit_order(legs, 1)  # ComboLegLimitOrder NOT on Schwab!
```

### ✅ CORRECT: Net Limit Price

```python
# Net limit price - Schwab supported
legs = [
    Leg.create(call1, 1),   # No order_price parameter
    Leg.create(call2, -1),
]
net_limit = 2.00  # Net debit/credit for entire combo
tickets = self.combo_limit_order(legs, 1, limit_price=net_limit)
```

### ❌ WRONG: Direct Dictionary Access

```python
# Can throw KeyError if no data
chain = slice.option_chains[self.option_symbol]
```

### ✅ CORRECT: Safe Access

```python
chain = slice.option_chains.get(self.option_symbol)
if not chain:
    return
```

### ❌ WRONG: Using Indicators Without Checks

```python
self.sma = self.sma(self.spy, 50)  # Missing warmup!

def on_data(self, slice):
    if self.sma.current.value > 100:  # Could be wrong!
        pass
```

### ✅ CORRECT: Warmup and Ready Checks

```python
self.settings.automatic_indicator_warm_up = True
self.sma = self.sma(self.spy, 50)

def on_data(self, slice):
    if self.is_warming_up or not self.sma.is_ready:
        return

    if self.sma.current.value > 100:
        pass
```

### ❌ WRONG: Individual Leg Execution

```python
# Legs execute separately - risk of unbalanced positions
self.market_order(call1, 1)
self.market_order(call2, -2)
self.market_order(call3, 1)
```

### ✅ CORRECT: Atomic Combo Execution

```python
# All legs fill together or none fill
legs = [
    Leg.create(call1, 1),
    Leg.create(call2, -2),
    Leg.create(call3, 1),
]
tickets = self.combo_limit_order(legs, 1, limit_price=net_limit)
```

---

## Option Universe Selection

### Basic Filter

```python
from AlgorithmImports import *

def initialize(self):
    option = self.add_option("SPY")
    self.option_symbol = option.symbol

    # Basic filter: strikes and expiration
    option.set_filter(-10, +10, 0, 90)  # ±10 strikes, 0-90 days
```

### Advanced Filter with Greeks

```python
def initialize(self):
    option = self.add_option("SPY")
    self.option_symbol = option.symbol

    # Filter by Greeks and IV
    option.set_filter(lambda u: u
        .strikes(-10, +10)              # Strike range
        .expiration(30, 90)             # 30-90 days to expiration
        .calls_only()                   # Only calls (or .puts_only())
        .delta(0.25, 0.35)              # Delta range
        .implied_volatility(0.20, 0.50) # IV range
        .open_interest(100, 999999)     # Minimum OI
    )
```

### Strategy-Based Filter

```python
def initialize(self):
    option = self.add_option("SPY")

    # Filter for specific strategy opportunities
    option.set_filter(lambda u: u.iron_condor(
        days_to_expiry=45,
        wing_width=5
    ))

    # Other strategy filters:
    # .butterfly_call(days_to_expiry)
    # .straddle(days_to_expiry)
    # .strangle(days_to_expiry)
```

---

## Two-Part Spread Strategy Template

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
        self.pending_orders = {}

        # Schedule 2.5s cancellation checks every second
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.every(timedelta(seconds=1)),
            self.check_order_timeouts
        )

    def on_data(self, slice):
        chain = slice.option_chains.get(self.option_symbol)
        if not chain:
            return

        # Find debit spread opportunity
        debit_spread = self.find_debit_spread(chain)
        if debit_spread:
            self.execute_debit_spread(debit_spread)

    def find_debit_spread(self, chain):
        """Find underpriced debit spread with wide bid-ask."""
        contracts = sorted(chain, key=lambda x: (x.expiry, x.strike))
        calls = [c for c in contracts if c.right == OptionRight.CALL]

        for i in range(len(calls) - 1):
            lower = calls[i]
            upper = calls[i + 1]

            # Wide spread = opportunity
            spread_width = lower.ask_price - lower.bid_price
            if spread_width < 0.50:
                continue

            return {'lower': lower, 'upper': upper}

        return None

    def execute_debit_spread(self, spread):
        """Execute at 35% from bid with quick cancel."""
        lower = spread['lower']
        upper = spread['upper']

        # Create combo legs
        legs = [
            Leg.create(lower.symbol, 1),
            Leg.create(upper.symbol, -1),
        ]

        # Calculate aggressive limit (35% from bid)
        lower_bid = lower.bid_price
        lower_mid = (lower.bid_price + lower.ask_price) / 2
        lower_target = lower_bid + (lower_mid - lower_bid) * 0.35

        upper_bid = upper.bid_price
        upper_mid = (upper.bid_price + upper.ask_price) / 2
        upper_target = upper_bid + (upper_mid - upper_bid) * 0.35

        net_debit = lower_target - upper_target

        # Execute combo
        tickets = self.combo_limit_order(legs, quantity=1, limit_price=net_debit)

        # Track for 2.5s cancellation
        self.pending_orders[tickets[0].order_id] = {
            'tickets': tickets,
            'submit_time': self.time,
        }

    def check_order_timeouts(self):
        """Cancel orders not filled within 2.5 seconds."""
        current_time = self.time
        to_cancel = []

        for order_id, info in self.pending_orders.items():
            elapsed = (current_time - info['submit_time']).total_seconds()

            if elapsed > 2.5:
                # Cancel unfilled orders
                for ticket in info['tickets']:
                    if ticket.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                        ticket.cancel("2.5s timeout")
                to_cancel.append(order_id)

        for order_id in to_cancel:
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]

    def on_order_event(self, order_event):
        """Handle fills and clean up tracking."""
        if order_event.order_id in self.pending_orders:
            if order_event.status == OrderStatus.FILLED:
                self.debug(f"Filled at {order_event.fill_price}")
                del self.pending_orders[order_event.order_id]
            elif order_event.status == OrderStatus.CANCELED:
                del self.pending_orders[order_event.order_id]
```

---

## Additional Resources

- **Detailed Research**: `/docs/research/PHASE3_ADVANCED_FEATURES_RESEARCH.md`
- **Strategy Documentation**: `/docs/strategies/TWO_PART_SPREAD_STRATEGY.md`
- **Project Instructions**: `/CLAUDE.md`
- **QuantConnect Docs**: https://www.quantconnect.com/docs
- **LEAN GitHub**: https://github.com/QuantConnect/Lean

---

**Last Updated**: 2025-11-30
