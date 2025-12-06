# QuantConnect Python API Reference

**Critical**: Python vs C# Naming Conventions

---

## Executive Summary

QuantConnect's LEAN engine is written in **C#**, but the **Python API uses snake_case** for methods and properties. Mixing C# and Python conventions will cause **runtime AttributeErrors**.

**This document clarifies the correct Python API patterns based on official QuantConnect GitHub examples.**

---

## Critical API Naming Differences

### ❌ WRONG (C# Convention - PascalCase)

```python
# DON'T USE C# naming in Python!
chain = slice.OptionChains[self.option_symbol]  # ❌ AttributeError
equity = self.AddEquity("SPY")                   # ❌ AttributeError
option = self.AddOption("SPY")                   # ❌ AttributeError
option.SetFilter(lambda u: u.Strikes(-2, +2))   # ❌ AttributeError
```

### ✅ CORRECT (Python Convention - snake_case)

```python
# Use Python snake_case naming
chain = slice.option_chains.get(self.option_symbol)  # ✅ Correct
equity = self.add_equity("SPY")                      # ✅ Correct
option = self.add_option("SPY")                      # ✅ Correct
option.set_filter(lambda u: u.strikes(-2, +2))      # ✅ Correct
```

**Source**: [BasicTemplateOptionsAlgorithm.py](https://github.com/QuantConnect/Lean/blob/master/Algorithm.Python/BasicTemplateOptionsAlgorithm.py)

---

## Complete API Reference

### Slice Data Access

| C# (WRONG in Python) | Python (CORRECT) | Notes |
|----------------------|------------------|-------|
| `slice.OptionChains[symbol]` | `slice.option_chains.get(symbol)` | Get option chain |
| `slice.OptionChains.Values` | `slice.option_chains.values()` | Iterate all chains |
| `slice.Bars[symbol]` | `slice.Bars[symbol]` | **Exception**: Bars uses PascalCase |
| `slice.QuoteBars[symbol]` | `slice.QuoteBars[symbol]` | **Exception**: QuoteBars uses PascalCase |

**Note**: `Bars` and `QuoteBars` are exceptions that use PascalCase in Python API.

---

### Algorithm Methods

| C# (WRONG in Python) | Python (CORRECT) | Description |
|----------------------|------------------|-------------|
| `self.AddEquity()` | `self.add_equity()` | Subscribe to equity data |
| `self.AddOption()` | `self.add_option()` | Subscribe to option data |
| `self.AddForex()` | `self.add_forex()` | Subscribe to forex data |
| `self.AddFuture()` | `self.add_future()` | Subscribe to futures data |
| `self.SetStartDate()` | `self.set_start_date()` | Set backtest start |
| `self.SetEndDate()` | `self.set_end_date()` | Set backtest end |
| `self.SetCash()` | `self.set_cash()` | Set starting cash |
| `self.SetWarmUp()` | `self.set_warm_up()` | Set warmup period |

**Pattern**: All algorithm setup methods use snake_case in Python.

---

### Option Chain Methods

| C# (WRONG in Python) | Python (CORRECT) | Description |
|----------------------|------------------|-------------|
| `option.SetFilter()` | `option.set_filter()` | Set option filter |
| `u.Strikes()` | `u.strikes()` | Filter by strike range |
| `u.Expiration()` | `u.expiration()` | Filter by expiration |
| `u.Calls()` | `u.calls()` | Filter calls only |
| `u.Puts()` | `u.puts()` | Filter puts only |
| `u.CallsOnly()` | `u.calls_only()` | **Deprecated**: use `calls()` |
| `u.PutsOnly()` | `u.puts_only()` | **Deprecated**: use `puts()` |

**Source**: [SetFilterUniverseEx.py](https://github.com/QuantConnect/Lean/blob/master/Algorithm.Python/SetFilterUniverseEx.py)

---

### Exceptions: Framework Properties (PascalCase)

**These properties REMAIN PascalCase in Python:**

```python
# Framework properties (PascalCase - CORRECT)
if self.IsWarmingUp:                           # ✅ Correct
    return

if symbol in self.Portfolio:                   # ✅ Correct
    holding = self.Portfolio[symbol]

if symbol in data.Bars:                        # ✅ Correct
    bar = data.Bars[symbol]

current_time = self.Time                       # ✅ Correct
total_value = self.Portfolio.TotalPortfolioValue  # ✅ Correct
```

**Why**: These are framework-level properties that maintain C# naming for consistency with documentation.

---

### Greeks and Contract Properties (PascalCase)

**Contract properties use PascalCase:**

```python
# Contract properties (PascalCase - CORRECT)
delta = contract.Greeks.Delta                  # ✅ Correct
gamma = contract.Greeks.Gamma                  # ✅ Correct
theta = contract.Greeks.Theta                  # ✅ Correct
theta_daily = contract.Greeks.ThetaPerDay      # ✅ Correct
vega = contract.Greeks.Vega                    # ✅ Correct
rho = contract.Greeks.Rho                      # ✅ Correct
iv = contract.ImpliedVolatility                # ✅ Correct
strike = contract.Strike                       # ✅ Correct
expiry = contract.Expiry                       # ✅ Correct
right = contract.Right                         # ✅ Correct (OptionRight.Call or Put)
```

**Why**: These are data properties, not methods.

---

### Order Methods

| C# (WRONG in Python) | Python (CORRECT) | Description |
|----------------------|------------------|-------------|
| `self.MarketOrder()` | `self.market_order()` | Submit market order |
| `self.LimitOrder()` | `self.limit_order()` | Submit limit order |
| `self.StopMarketOrder()` | `self.stop_market_order()` | Submit stop order |
| `self.ComboLimitOrder()` | `self.combo_limit_order()` | **Actually**: Uses PascalCase! |
| `self.ComboMarketOrder()` | `self.combo_market_order()` | **Actually**: Uses PascalCase! |
| `self.SetHoldings()` | `self.set_holdings()` | Set target holdings |
| `self.Liquidate()` | `self.liquidate()` | Liquidate positions |

**CRITICAL EXCEPTION**: ComboOrders use **PascalCase** even in Python API!

```python
# ComboOrders are exception - use PascalCase
from AlgorithmImports import Leg

legs = [
    Leg.Create(long_symbol, 1),
    Leg.Create(short_symbol, -1),
]

# PascalCase for ComboOrder methods
ticket = self.ComboLimitOrder(legs, 1, net_price)   # ✅ Correct (PascalCase!)
ticket = self.ComboMarketOrder(legs, 1)             # ✅ Correct (PascalCase!)
```

---

### Complete Example (Correct Python API)

```python
from AlgorithmImports import *

class MyOptionsAlgorithm(QCAlgorithm):
    def initialize(self):
        # Setup methods: snake_case
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)

        # Add data: snake_case
        equity = self.add_equity("SPY", Resolution.Minute)
        option = self.add_option("SPY", Resolution.Minute)

        # Set filter: snake_case
        option.set_filter(lambda u: u.strikes(-5, +5).expiration(0, 30))

        self.option_symbol = option.Symbol

    def on_data(self, data):
        # Framework properties: PascalCase
        if self.IsWarmingUp:
            return

        # Slice data: snake_case for collections
        chain = data.option_chains.get(self.option_symbol)
        if chain is None:
            return

        # Contract properties: PascalCase
        for contract in chain:
            if contract.Greeks:
                delta = contract.Greeks.Delta
                iv = contract.ImpliedVolatility

                # Order methods: snake_case (except ComboOrders!)
                if abs(delta) > 0.3:
                    self.market_order(contract.Symbol, 1)

        # ComboOrders: PascalCase exception
        if len(list(chain)) >= 2:
            contracts = list(chain)
            legs = [
                Leg.Create(contracts[0].Symbol, 1),
                Leg.Create(contracts[1].Symbol, -1),
            ]
            self.ComboLimitOrder(legs, 1, 0.50)  # PascalCase!
```

---

## Common Mistakes

### Mistake 1: Using C# Naming for Slice Access

```python
# ❌ WRONG - Will cause AttributeError
chain = slice.OptionChains[self.option_symbol]

# ✅ CORRECT
chain = slice.option_chains.get(self.option_symbol)
```

### Mistake 2: Using snake_case for Framework Properties

```python
# ❌ WRONG
if self.is_warming_up:  # AttributeError
    return

# ✅ CORRECT
if self.IsWarmingUp:
    return
```

### Mistake 3: Using snake_case for ComboOrders

```python
# ❌ WRONG
ticket = self.combo_limit_order(legs, 1, 0.50)  # AttributeError

# ✅ CORRECT - ComboOrders use PascalCase
ticket = self.ComboLimitOrder(legs, 1, 0.50)
```

### Mistake 4: Using C# Naming for Algorithm Methods

```python
# ❌ WRONG
equity = self.AddEquity("SPY")  # AttributeError

# ✅ CORRECT
equity = self.add_equity("SPY")
```

---

## Quick Reference Table

| Category | Convention | Examples |
|----------|------------|----------|
| Algorithm Setup Methods | snake_case | `add_equity()`, `set_start_date()` |
| Algorithm Trading Methods | snake_case | `market_order()`, `limit_order()` |
| **ComboOrder Methods** | **PascalCase** | `ComboLimitOrder()`, `ComboMarketOrder()` |
| Framework Properties | PascalCase | `IsWarmingUp`, `Portfolio`, `Time` |
| Slice Collections | snake_case | `option_chains.get()`, `option_chains.values()` |
| **Slice Data** | **PascalCase** | `Bars[]`, `QuoteBars[]` |
| Contract Properties | PascalCase | `Greeks.Delta`, `ImpliedVolatility`, `Strike` |
| Option Filter Methods | snake_case | `set_filter()`, `strikes()`, `expiration()` |

---

## Official Sources

All Python API conventions are from official QuantConnect GitHub repository:

1. [BasicTemplateOptionsAlgorithm.py](https://github.com/QuantConnect/Lean/blob/master/Algorithm.Python/BasicTemplateOptionsAlgorithm.py) - Option chain access patterns
2. [SetFilterUniverseEx.py](https://github.com/QuantConnect/Lean/blob/master/Algorithm.Python/SetFilterUniverseEx.py) - Filter examples
3. [OptionChainedUniverseFilteringAlgorithm.py](https://github.com/QuantConnect/Lean/blob/master/Algorithm.Python/OptionChainedUniverseFilteringAlgorithm.py) - Universe selection

---

## Summary

✅ **Default Rule**: Use **snake_case** for methods in Python API

⚠️ **Exceptions**:
- Framework properties (IsWarmingUp, Portfolio, Time, etc.) - **PascalCase**
- Slice data dictionaries (Bars, QuoteBars) - **PascalCase**
- ComboOrder methods (ComboLimitOrder, ComboMarketOrder) - **PascalCase**
- Contract properties (Greeks, Strike, Expiry, etc.) - **PascalCase**

**When in doubt**: Check official Python examples in QuantConnect GitHub repository.

---

**Last Updated**: 2025-11-30
**Author**: Claude Code Integration Team
**Status**: ✅ Authoritative Python API Reference
