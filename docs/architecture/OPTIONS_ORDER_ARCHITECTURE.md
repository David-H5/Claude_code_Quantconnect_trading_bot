# Options Order Architecture Guide

**Date**: November 30, 2025
**Purpose**: Comprehensive guide for autonomous + manual UI-driven options trading
**Scope**: OptionStrategies factory methods vs manual Leg.Create() + UI integration

---

## 📋 Table of Contents

1. [Understanding the Two Approaches](#understanding-the-two-approaches)
2. [OptionStrategies Factory Methods (37+)](#optionstrategies-factory-methods)
3. [Manual Leg.Create() Approach](#manual-legcreate-approach)
4. [Comparison Matrix](#comparison-matrix)
5. [Hybrid Architecture](#hybrid-architecture-recommended)
6. [UI Integration](#ui-integration)
7. [Recurring Orders System](#recurring-orders-system)
8. [Bot-Managed Manual Orders](#bot-managed-manual-orders)
9. [Implementation Examples](#implementation-examples)

---

## Understanding the Two Approaches

### OptionStrategies Factory Methods

**What**: Pre-built constructors for 37+ standard multi-leg option strategies

**How it works**:
```python
# Single line creates entire multi-leg position
strategy = OptionStrategies.butterfly_call(
    option_symbol, lower_strike, middle_strike, upper_strike, expiry
)
self.buy(strategy, 1)  # Atomic execution
```

**Benefits**:
- ✅ **Automatic position grouping** for accurate margin
- ✅ **Strategy recognition** by LEAN (37 known patterns)
- ✅ **Cleaner code** (no manual leg construction)
- ✅ **Single command** to enter/exit entire position
- ✅ **Correct multi-leg margin** calculation

**Limitations**:
- ❌ **Less price control** (uses market or net pricing)
- ❌ **Limited customization** (standard strategies only)
- ❌ **No leg-by-leg timing** (all legs execute together)

---

### Manual Leg.Create() Approach

**What**: Manually construct each leg with precise control

**How it works**:
```python
# Construct each leg manually
legs = [
    Leg.Create(call1_symbol, 1),    # Buy lower strike
    Leg.Create(call2_symbol, -2),   # Sell 2x middle strike
    Leg.Create(call3_symbol, 1),    # Buy upper strike
]

# Execute with precise pricing control
self.ComboLimitOrder(legs, quantity=1, limit_price=net_debit)
```

**Benefits**:
- ✅ **Precise price control** (limit orders, net pricing)
- ✅ **Custom strategies** (non-standard spreads)
- ✅ **Two-part execution** (leg into positions gradually)
- ✅ **Dynamic adjustment** (modify legs on the fly)
- ✅ **Fill tracking** per leg

**Limitations**:
- ❌ **More code complexity** (manual leg construction)
- ❌ **Manual margin tracking** (no automatic grouping)
- ❌ **Risk of unbalanced positions** (if some legs don't fill)

---

## OptionStrategies Factory Methods

### Complete List of 37+ Strategies

QuantConnect provides factory methods for all these strategies:

#### Single Leg Strategies (8)

| Strategy | Method | Description |
|----------|--------|-------------|
| **Long Call** | `bare_call()` | Buy call option |
| **Short Call** | `naked_call()` | Sell call option |
| **Long Put** | `bare_put()` | Buy put option |
| **Short Put** | `naked_put()` | Sell put option |
| **Covered Call** | `covered_call()` | Long stock + short call |
| **Protective Put** | `protective_put()` | Long stock + long put |
| **Covered Put** | `covered_put()` | Short stock + short put |
| **Protective Call** | `protective_call()` | Short stock + long call |

#### Vertical Spreads (8)

| Strategy | Method | Description |
|----------|--------|-------------|
| **Bull Call Spread** | `bull_call_spread()` | Buy lower call, sell higher call |
| **Bear Call Spread** | `bear_call_spread()` | Sell lower call, buy higher call |
| **Bull Put Spread** | `bull_put_spread()` | Sell higher put, buy lower put |
| **Bear Put Spread** | `bear_put_spread()` | Buy higher put, sell lower put |
| **Call Calendar Spread** | `call_calendar_spread()` | Sell near-term, buy far-term call |
| **Put Calendar Spread** | `put_calendar_spread()` | Sell near-term, buy far-term put |
| **Short Call Calendar** | `short_call_calendar_spread()` | Buy near-term, sell far-term call |
| **Short Put Calendar** | `short_put_calendar_spread()` | Buy near-term, sell far-term put |

#### Butterfly & Condor Strategies (10)

| Strategy | Method | Description |
|----------|--------|-------------|
| **Long Butterfly Call** | `butterfly_call()` | Buy-Sell-Sell-Buy calls |
| **Short Butterfly Call** | `short_butterfly_call()` | Sell-Buy-Buy-Sell calls |
| **Long Butterfly Put** | `butterfly_put()` | Buy-Sell-Sell-Buy puts |
| **Short Butterfly Put** | `short_butterfly_put()` | Sell-Buy-Buy-Sell puts |
| **Iron Butterfly** | `iron_butterfly()` | Call + put butterfly combo |
| **Short Iron Butterfly** | `short_iron_butterfly()` | Reverse iron butterfly |
| **Long Iron Condor** | `iron_condor()` | Sell OTM put/call spreads |
| **Short Iron Condor** | `short_iron_condor()` | Buy OTM put/call spreads |
| **Long Condor** | `condor_call()` or `condor_put()` | 4-strike butterfly |
| **Short Condor** | `short_condor_call()` or `short_condor_put()` | Reverse condor |

#### Straddle & Strangle Strategies (8)

| Strategy | Method | Description |
|----------|--------|-------------|
| **Long Straddle** | `straddle()` | Buy call + put same strike |
| **Short Straddle** | `short_straddle()` | Sell call + put same strike |
| **Long Strangle** | `strangle()` | Buy OTM call + OTM put |
| **Short Strangle** | `short_strangle()` | Sell OTM call + OTM put |
| **Call Ladder** | `call_ladder()` | Buy 1, sell 2 calls different strikes |
| **Put Ladder** | `put_ladder()` | Buy 1, sell 2 puts different strikes |
| **Short Call Ladder** | `short_call_ladder()` | Sell 1, buy 2 calls |
| **Short Put Ladder** | `short_put_ladder()` | Sell 1, buy 2 puts |

#### Advanced Strategies (3+)

| Strategy | Method | Description |
|----------|--------|-------------|
| **Box Spread** | `box_spread()` | Arbitrage - call spread + put spread |
| **Conversion** | `conversion()` | Long stock + synthetic short stock |
| **Reversal** | `reversal()` | Short stock + synthetic long stock |

**Source**: [OptionStrategies.cs](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/Option/OptionStrategies.cs)

---

### OptionStrategies Code Examples

#### Example 1: Long Butterfly Call

```python
from AlgorithmImports import *

class ButterflyBot(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        option = self.AddOption("SPY", Resolution.Minute)
        self.option_symbol = option.Symbol
        option.SetFilter(-10, 10, 30, 90)

    def OnData(self, data: Slice):
        chain = data.OptionChains.get(self.option_symbol)
        if not chain:
            return

        # Find strikes for butterfly
        expiry = min([x.Expiry for x in chain])
        calls = sorted([c for c in chain
                       if c.Expiry == expiry and c.Right == OptionRight.Call],
                      key=lambda x: x.Strike)

        if len(calls) < 5:
            return

        # Create butterfly using factory method
        strategy = OptionStrategies.butterfly_call(
            self.option_symbol,
            calls[0].Strike,   # Lower: Buy 1
            calls[2].Strike,   # Middle: Sell 2
            calls[4].Strike,   # Upper: Buy 1
            expiry
        )

        # Execute entire strategy atomically
        if not self.Portfolio[strategy].Invested:
            self.Buy(strategy, 1)  # Opens entire butterfly
            self.Debug(f"Opened butterfly: {calls[0].Strike}/{calls[2].Strike}/{calls[4].Strike}")
```

#### Example 2: Iron Condor

```python
def OnData(self, data: Slice):
    chain = data.OptionChains.get(self.option_symbol)
    if not chain:
        return

    # Find strikes
    atm = chain.Underlying.Price
    expiry = min([x.Expiry for x in chain])
    strikes = sorted(set(c.Strike for c in chain if c.Expiry == expiry))

    # Select strikes (example: ~5% OTM)
    put_buy = min([s for s in strikes if s < atm * 0.95], default=None)
    put_sell = min([s for s in strikes if s < atm * 0.97], default=None)
    call_sell = max([s for s in strikes if s > atm * 1.03], default=None)
    call_buy = max([s for s in strikes if s > atm * 1.05], default=None)

    if not all([put_buy, put_sell, call_sell, call_buy]):
        return

    # Create iron condor
    strategy = OptionStrategies.iron_condor(
        self.option_symbol,
        put_buy,      # Protective put (buy)
        put_sell,     # Income put (sell)
        call_sell,    # Income call (sell)
        call_buy,     # Protective call (buy)
        expiry
    )

    # Execute with automatic margin grouping
    if not self.Portfolio[strategy].Invested:
        self.Buy(strategy, 1)
        self.Debug(f"Iron Condor: P{put_buy}/{put_sell} C{call_sell}/{call_buy}")
```

---

## Manual Leg.Create() Approach

### When to Use Manual Legs

**Use manual Leg.Create() when you need**:

1. **Precise price control**: Your two-part spread strategy with 35%/65% fill targets
2. **Custom timing**: Leg into positions separately (debit first, credit later)
3. **Non-standard strategies**: Ratio spreads, custom butterflies
4. **Dynamic adjustments**: Modify leg quantities based on fills
5. **Fill tracking**: Monitor each leg independently

### ComboOrder with Manual Legs

**ComboLimitOrder** (Schwab Supported):

```python
from AlgorithmImports import *

class ManualLegsBot(QCAlgorithm):
    def OnData(self, data: Slice):
        chain = data.OptionChains.get(self.option_symbol)
        if not chain:
            return

        # Find contracts manually
        calls = sorted([c for c in chain if c.Right == OptionRight.Call],
                      key=lambda x: x.Strike)

        if len(calls) < 5:
            return

        # Calculate net debit/credit
        lower = calls[0]
        middle = calls[2]
        upper = calls[4]

        net_debit = (lower.AskPrice - middle.BidPrice * 2 + upper.AskPrice) * 100

        # Construct legs manually
        legs = [
            Leg.Create(lower.Symbol, 1),    # Buy lower
            Leg.Create(middle.Symbol, -2),  # Sell 2x middle
            Leg.Create(upper.Symbol, 1),    # Buy upper
        ]

        # Execute with net limit price
        tickets = self.ComboLimitOrder(
            legs,
            quantity=1,
            limit_price=net_debit * 0.9  # Try to fill at 90% of mid
        )

        # Track tickets
        for ticket in tickets:
            self.Debug(f"Order ID: {ticket.OrderId}, Status: {ticket.Status}")
```

**Two-Part Execution** (Your Current Strategy):

```python
class TwoPartSpreadBot(QCAlgorithm):
    def __init__(self):
        self.pending_debit_spreads = {}

    def ExecuteDebitSpread(self, chain, underlying_price):
        """Execute debit spread at 35% from bid."""
        calls = sorted([c for c in chain if c.Right == OptionRight.Call],
                      key=lambda x: x.Strike)

        if len(calls) < 3:
            return

        # Debit spread legs
        long_call = calls[0]
        short_call = calls[1]

        # Calculate limit price (35% from bid)
        bid_price = (long_call.BidPrice - short_call.AskPrice) * 100
        ask_price = (long_call.AskPrice - short_call.BidPrice) * 100
        limit_price = bid_price + (ask_price - bid_price) * 0.35

        # Create legs
        legs = [
            Leg.Create(long_call.Symbol, 1),
            Leg.Create(short_call.Symbol, -1),
        ]

        # Execute with 2.5s timeout
        tickets = self.ComboLimitOrder(legs, quantity=1, limit_price=limit_price)

        # Track for credit spread matching
        order_id = tickets[0].OrderId
        self.pending_debit_spreads[order_id] = {
            'tickets': tickets,
            'submit_time': self.Time,
            'long_strike': long_call.Strike,
            'short_strike': short_call.Strike,
        }

        # Schedule cancel check
        self.Schedule.On(
            self.DateRules.Today,
            self.TimeRules.AfterMarketOpen("SPY", 2.5 / 60),  # 2.5 seconds
            lambda: self.CheckCancelDebitSpread(order_id)
        )

    def CheckCancelDebitSpread(self, order_id):
        """Cancel unfilled debit spread after 2.5 seconds."""
        if order_id not in self.pending_debit_spreads:
            return

        info = self.pending_debit_spreads[order_id]

        for ticket in info['tickets']:
            if ticket.Status in [OrderStatus.Submitted, OrderStatus.PartiallyFilled]:
                ticket.Cancel("2.5s timeout - resubmit")

        # Resubmit with adjusted price
        # ... your two-part logic ...
```

---

## Comparison Matrix

| Feature | OptionStrategies | Manual Leg.Create() |
|---------|-----------------|---------------------|
| **Code Simplicity** | ✅ Very clean (1 line) | ❌ More complex (manual construction) |
| **Price Control** | ❌ Limited (market/net pricing) | ✅ Precise (leg-by-leg limits) |
| **Two-Part Execution** | ❌ Not supported | ✅ Fully supported |
| **Margin Calculation** | ✅ Automatic multi-leg margin | ❌ Manual tracking needed |
| **Strategy Recognition** | ✅ LEAN recognizes 37 patterns | ❌ No automatic recognition |
| **Custom Strategies** | ❌ Standard strategies only | ✅ Any custom combination |
| **Exit Simplicity** | ✅ `Liquidate(strategy)` | ❌ Close each leg manually |
| **Fill Tracking** | ❌ All-or-nothing atomic | ✅ Per-leg tracking |
| **Recurring Orders** | ✅ Easy to template | ✅ Flexible templates |
| **UI Integration** | ✅ Simple (dropdown selection) | ✅ Flexible (custom builder) |
| **QuantConnect Compatibility** | ✅ Yes | ✅ Yes |
| **Schwab Compatibility** | ✅ Yes (via ComboOrders) | ✅ Yes (ComboLimitOrder) |
| **Best For** | Standard strategies, simplicity | Custom strategies, price control |

---

## Hybrid Architecture (RECOMMENDED)

**Use both approaches based on order source and requirements.**

### Strategy Decision Matrix

```python
class HybridOptionsBot(QCAlgorithm):
    def DetermineExecutionMethod(self, order_request):
        """Choose OptionStrategies or manual legs based on requirements."""

        # Use OptionStrategies if:
        if (order_request['strategy_type'] in STANDARD_STRATEGIES and
            order_request['execution_type'] == 'market' and
            not order_request.get('two_part_execution', False)):
            return "option_strategies"

        # Use manual legs if:
        elif (order_request.get('two_part_execution', False) or
              order_request.get('custom_pricing', False) or
              order_request['strategy_type'] == 'custom'):
            return "manual_legs"

        # Default to OptionStrategies for simplicity
        return "option_strategies"


# Standard strategies list
STANDARD_STRATEGIES = [
    'butterfly_call', 'butterfly_put', 'iron_butterfly',
    'iron_condor', 'bull_call_spread', 'bear_put_spread',
    'straddle', 'strangle', 'covered_call', 'protective_put'
]
```

---

## UI Integration

### UI Architecture

```
┌────────────────────────────────────────────────┐
│              Trading Dashboard UI              │
│  (PySide6 - Local Desktop Application)        │
├────────────────────────────────────────────────┤
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │  Manual Order Panel                      │ │
│  │  - Strategy Selector (dropdown)          │ │
│  │  - Strike/Expiry Selector                │ │
│  │  - Execution Type (Market/Limit/Two-Part)│ │
│  │  - Quantity Input                        │ │
│  │  - [ Submit Order ]                      │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │  Recurring Orders Panel                  │ │
│  │  - Schedule Selector (Daily/Weekly/etc)  │ │
│  │  - Conditions (Greeks thresholds, etc)   │ │
│  │  - [ Enable Recurring ]                  │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │  Active Positions                        │ │
│  │  - Position P&L                          │ │
│  │  - Greeks Summary                        │ │
│  │  - [ Close Position ]                    │ │
│  └──────────────────────────────────────────┘ │
│                                                │
└────────────────────────────────────────────────┘
         ↕ JSON-RPC or REST API ↕
┌────────────────────────────────────────────────┐
│       QuantConnect Algorithm (Cloud)           │
│  - Autonomous trading logic                    │
│  - Receives manual orders from UI              │
│  - Executes recurring orders                   │
│  - Reports position status to UI               │
└────────────────────────────────────────────────┘
```

### UI Order Data Structure

```python
# Order request from UI to algorithm
manual_order = {
    "source": "ui_manual",
    "order_id": "ui_001",
    "timestamp": "2025-11-30T10:30:00",
    "strategy_type": "iron_condor",  # or "custom"
    "execution_method": "option_strategies",  # or "manual_legs"
    "execution_type": "limit",  # market, limit, two_part
    "underlying": "SPY",
    "expiry": "2025-12-20",
    "strikes": {
        "put_buy": 540,
        "put_sell": 545,
        "call_sell": 560,
        "call_buy": 565,
    },
    "quantity": 1,
    "limit_price": 200.0,  # net credit/debit
    "two_part_config": {
        "enabled": True,
        "debit_fill_target": 0.35,  # 35% from bid
        "credit_fill_target": 0.65,  # 65% from bid
        "cancel_timeout_seconds": 2.5,
    },
    "recurring": False,
}
```

---

## Recurring Orders System

### Architecture

```python
@dataclass
class RecurringOrderTemplate:
    """Template for recurring orders."""
    template_id: str
    name: str
    strategy_type: str  # "iron_condor", "butterfly_call", etc.
    execution_method: str  # "option_strategies" or "manual_legs"

    # Schedule
    schedule_type: str  # "daily", "weekly", "monthly", "conditional"
    schedule_config: Dict[str, Any]

    # Strike selection rules
    strike_selection: Dict[str, Any]  # Example: {"delta_target": 0.30}

    # Entry conditions
    entry_conditions: Dict[str, Any]  # Example: {"iv_rank_min": 50}

    # Execution config
    execution_config: Dict[str, Any]

    # Position management
    profit_target_pct: float = 0.50
    loss_limit_pct: float = 2.00
    max_dte: int = 45

    # Bot management
    bot_managed: bool = True  # Let bot manage after entry

    enabled: bool = True


class RecurringOrderManager:
    """Manages recurring order execution."""

    def __init__(self, algorithm: QCAlgorithm):
        self.algorithm = algorithm
        self.templates: Dict[str, RecurringOrderTemplate] = {}
        self.active_orders: Dict[str, Any] = {}

    def add_template(self, template: RecurringOrderTemplate):
        """Add recurring order template from UI."""
        self.templates[template.template_id] = template

        # Schedule execution based on template
        if template.schedule_type == "daily":
            self.algorithm.Schedule.On(
                self.algorithm.DateRules.EveryDay(),
                self.algorithm.TimeRules.AfterMarketOpen("SPY", 30),
                lambda: self.execute_template(template.template_id)
            )
        elif template.schedule_type == "weekly":
            self.algorithm.Schedule.On(
                self.algorithm.DateRules.Every(DayOfWeek.Monday),
                self.algorithm.TimeRules.AfterMarketOpen("SPY", 30),
                lambda: self.execute_template(template.template_id)
            )
        # ... other schedule types

    def execute_template(self, template_id: str):
        """Execute recurring order if conditions met."""
        template = self.templates.get(template_id)
        if not template or not template.enabled:
            return

        # Check entry conditions
        if not self.check_entry_conditions(template):
            self.algorithm.Debug(f"Template {template_id}: Entry conditions not met")
            return

        # Select strikes based on rules
        strikes = self.select_strikes(template)
        if not strikes:
            self.algorithm.Debug(f"Template {template_id}: Could not find suitable strikes")
            return

        # Execute order
        if template.execution_method == "option_strategies":
            self.execute_via_factory(template, strikes)
        else:
            self.execute_via_manual_legs(template, strikes)

    def check_entry_conditions(self, template: RecurringOrderTemplate) -> bool:
        """Check if entry conditions are met."""
        conditions = template.entry_conditions

        # Example: Check IV Rank
        if "iv_rank_min" in conditions:
            iv_rank = self.get_iv_rank(template.strategy_type)
            if iv_rank < conditions["iv_rank_min"]:
                return False

        # Example: Check Greeks
        if "delta_neutral" in conditions and conditions["delta_neutral"]:
            portfolio_delta = self.get_portfolio_delta()
            if abs(portfolio_delta) > 100:  # Not delta neutral
                return False

        # All conditions met
        return True

    def select_strikes(self, template: RecurringOrderTemplate) -> Dict[str, float]:
        """Select strikes based on template rules."""
        selection = template.strike_selection
        chain = self.get_option_chain(template.strategy_type)

        if "delta_target" in selection:
            # Find strikes by delta
            return self.find_strikes_by_delta(chain, selection["delta_target"])
        elif "atm_offset" in selection:
            # Find strikes by ATM offset
            return self.find_strikes_by_atm_offset(chain, selection["atm_offset"])

        return {}

    def execute_via_factory(self, template: RecurringOrderTemplate, strikes: Dict):
        """Execute using OptionStrategies factory method."""
        strategy_method = getattr(OptionStrategies, template.strategy_type)

        # Build strategy
        if template.strategy_type == "iron_condor":
            strategy = strategy_method(
                self.algorithm.option_symbol,
                strikes['put_buy'],
                strikes['put_sell'],
                strikes['call_sell'],
                strikes['call_buy'],
                strikes['expiry']
            )
        elif template.strategy_type == "butterfly_call":
            strategy = strategy_method(
                self.algorithm.option_symbol,
                strikes['lower'],
                strikes['middle'],
                strikes['upper'],
                strikes['expiry']
            )

        # Execute
        self.algorithm.Buy(strategy, template.execution_config.get('quantity', 1))

        # Track for bot management
        if template.bot_managed:
            self.active_orders[strategy] = {
                'template_id': template.template_id,
                'entry_time': self.algorithm.Time,
                'profit_target': template.profit_target_pct,
                'loss_limit': template.loss_limit_pct,
            }
```

---

## Bot-Managed Manual Orders

### Concept

**User initiates order via UI** → **Bot manages the position** (profit-taking, stop-loss, adjustments)

```python
class BotManagedPosition:
    """Bot management of user-initiated positions."""

    def __init__(self, position_id: str, config: Dict[str, Any]):
        self.position_id = position_id
        self.entry_time = datetime.now()
        self.strategy_symbol = config['strategy_symbol']

        # Management rules
        self.profit_targets = config.get('profit_targets', [0.50, 1.00, 2.00])
        self.profit_percentages = config.get('profit_percentages', [0.30, 0.50, 0.20])
        self.loss_limit_pct = config.get('loss_limit_pct', 2.00)
        self.max_dte = config.get('max_dte', 7)  # Roll if < 7 DTE

        # State
        self.quantity_remaining = config['initial_quantity']
        self.entry_price = config['entry_price']

    def check_profit_taking(self, current_price: float) -> Optional[float]:
        """Check if should take profits."""
        pnl_pct = (current_price - self.entry_price) / self.entry_price

        for i, target in enumerate(self.profit_targets):
            if pnl_pct >= target:
                # Take profit on this tier
                quantity_to_close = self.quantity_remaining * self.profit_percentages[i]
                return quantity_to_close

        return None

    def check_stop_loss(self, current_price: float) -> bool:
        """Check if should exit on stop loss."""
        pnl_pct = (current_price - self.entry_price) / self.entry_price
        return pnl_pct <= -self.loss_limit_pct

    def check_roll_needed(self, days_to_expiry: int) -> bool:
        """Check if position should be rolled."""
        return days_to_expiry <= self.max_dte


class BotManagedOrdersModule:
    """Module for bot management of UI orders."""

    def __init__(self, algorithm: QCAlgorithm):
        self.algorithm = algorithm
        self.managed_positions: Dict[str, BotManagedPosition] = {}

    def add_position_from_ui(self, order_result: Dict):
        """Start managing a position created from UI."""
        position = BotManagedPosition(
            position_id=order_result['order_id'],
            config={
                'strategy_symbol': order_result['strategy_symbol'],
                'initial_quantity': order_result['filled_quantity'],
                'entry_price': order_result['fill_price'],
                'profit_targets': [0.50, 1.00, 2.00],  # +50%, +100%, +200%
                'profit_percentages': [0.30, 0.50, 0.20],  # Close 30%, 50%, 20%
                'loss_limit_pct': 2.00,  # -200%
                'max_dte': 7,
            }
        )

        self.managed_positions[order_result['order_id']] = position
        self.algorithm.Debug(f"Now managing position {order_result['order_id']}")

    def on_data(self, data: Slice):
        """Check all managed positions for management actions."""
        for position_id, position in list(self.managed_positions.items()):
            # Get current price
            if position.strategy_symbol not in self.algorithm.Portfolio:
                continue

            holding = self.algorithm.Portfolio[position.strategy_symbol]
            current_price = holding.Price

            # Check profit taking
            quantity_to_close = position.check_profit_taking(current_price)
            if quantity_to_close and quantity_to_close > 0:
                self.algorithm.Liquidate(position.strategy_symbol, quantity_to_close)
                position.quantity_remaining -= quantity_to_close
                self.algorithm.Debug(
                    f"Profit-taking: Closed {quantity_to_close} contracts of {position_id}"
                )

            # Check stop loss
            if position.check_stop_loss(current_price):
                self.algorithm.Liquidate(position.strategy_symbol)
                self.algorithm.Debug(f"Stop loss: Closed position {position_id}")
                del self.managed_positions[position_id]
                continue

            # Check roll needed
            # (Implementation depends on your rolling strategy)
```

---

## Implementation Examples

### Complete Hybrid Bot Example

```python
from AlgorithmImports import *
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

class HybridOptionsBot(QCAlgorithm):
    """
    Complete bot with:
    - Autonomous trading via OptionStrategies
    - Manual UI orders via Leg.Create()
    - Recurring orders
    - Bot-managed positions
    """

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        # Options setup
        option = self.AddOption("SPY", Resolution.Minute)
        self.option_symbol = option.Symbol
        option.SetFilter(-10, 10, 30, 90)

        # Modules
        self.recurring_manager = RecurringOrderManager(self)
        self.bot_managed = BotManagedOrdersModule(self)

        # UI communication (simulated - replace with actual API)
        self.ui_orders_queue = []

        # Autonomous trading state
        self.autonomous_enabled = True

    def OnData(self, data: Slice):
        """Main trading loop."""

        # 1. Process UI orders
        self.process_ui_orders(data)

        # 2. Autonomous trading (if enabled)
        if self.autonomous_enabled:
            self.autonomous_trading(data)

        # 3. Manage bot-managed positions
        self.bot_managed.on_data(data)

    def process_ui_orders(self, data: Slice):
        """Process manual orders from UI."""
        while self.ui_orders_queue:
            order = self.ui_orders_queue.pop(0)

            if order['execution_method'] == "option_strategies":
                result = self.execute_via_option_strategies(order, data)
            else:
                result = self.execute_via_manual_legs(order, data)

            # If bot management enabled, track position
            if order.get('bot_managed', False) and result:
                self.bot_managed.add_position_from_ui(result)

    def autonomous_trading(self, data: Slice):
        """Autonomous trading logic using OptionStrategies."""
        chain = data.OptionChains.get(self.option_symbol)
        if not chain:
            return

        # Example: Enter iron condor when IV Rank > 50
        iv_rank = self.calculate_iv_rank()

        if iv_rank > 50 and not self.Portfolio[self.option_symbol].Invested:
            # Find strikes
            strikes = self.find_iron_condor_strikes(chain)
            if strikes:
                strategy = OptionStrategies.iron_condor(
                    self.option_symbol,
                    strikes['put_buy'],
                    strikes['put_sell'],
                    strikes['call_sell'],
                    strikes['call_buy'],
                    strikes['expiry']
                )

                self.Buy(strategy, 1)
                self.Debug(f"Autonomous: Entered iron condor at IV Rank {iv_rank:.1f}")

    def execute_via_option_strategies(self, order: Dict, data: Slice) -> Optional[Dict]:
        """Execute order using OptionStrategies factory method."""
        strategy_type = order['strategy_type']
        strikes = order['strikes']
        expiry = order['expiry']

        # Get factory method
        if strategy_type == "iron_condor":
            strategy = OptionStrategies.iron_condor(
                self.option_symbol,
                strikes['put_buy'],
                strikes['put_sell'],
                strikes['call_sell'],
                strikes['call_buy'],
                expiry
            )
        elif strategy_type == "butterfly_call":
            strategy = OptionStrategies.butterfly_call(
                self.option_symbol,
                strikes['lower'],
                strikes['middle'],
                strikes['upper'],
                expiry
            )
        # ... other strategies

        # Execute
        if order['execution_type'] == 'market':
            ticket = self.Buy(strategy, order['quantity'])
        else:
            ticket = self.LimitOrder(strategy, order['quantity'], order['limit_price'])

        return {
            'order_id': order['order_id'],
            'strategy_symbol': strategy,
            'filled_quantity': order['quantity'],
            'fill_price': order.get('limit_price', 0),
        }

    def execute_via_manual_legs(self, order: Dict, data: Slice) -> Optional[Dict]:
        """Execute order using manual Leg.Create()."""
        # Build legs based on order
        legs = []

        if order['strategy_type'] == "custom":
            # Custom leg construction from UI
            for leg_config in order['legs']:
                legs.append(Leg.Create(leg_config['symbol'], leg_config['quantity']))

        # Execute with two-part logic if enabled
        if order.get('two_part_config', {}).get('enabled', False):
            return self.execute_two_part_spread(order, legs)
        else:
            # Standard combo order
            tickets = self.ComboLimitOrder(
                legs,
                quantity=order['quantity'],
                limit_price=order['limit_price']
            )

            return {
                'order_id': order['order_id'],
                'strategy_symbol': legs[0],  # Simplified
                'filled_quantity': order['quantity'],
                'fill_price': order['limit_price'],
            }

    # ... Additional methods for strike finding, IV rank calculation, etc.
```

---

## Conclusion

### Recommended Approach

1. **Autonomous Trading**: Use **OptionStrategies** factory methods for clean, simple code
2. **Manual UI Orders (Standard)**: Use **OptionStrategies** with dropdown selection
3. **Manual UI Orders (Custom/Two-Part)**: Use **manual Leg.Create()** for full control
4. **Recurring Orders**: Support both methods based on user preference
5. **Bot Management**: Support for both execution methods

### Next Steps

1. **Implement UI Order Queue**: JSON-RPC or REST API for UI communication
2. **Build Strategy Templates**: Pre-configured templates for common strategies
3. **Add Position Tracking**: Dashboard showing all positions (autonomous + manual)
4. **Implement Recurring System**: Schedule engine for recurring orders
5. **Test Both Approaches**: Backtest with OptionStrategies and manual legs

Both approaches work perfectly with QuantConnect. Your choice depends on whether you prioritize **simplicity** (OptionStrategies) or **control** (manual legs).

For your two-part spread strategy, **manual Leg.Create() with ComboLimitOrder** is the correct choice!
