# QuantConnect Framework Updates Summary

**Date**: 2025-11-30
**Version**: Based on QuantConnect GitHub analysis (LEAN PR #6720 and latest repository state)

---

## Critical Discoveries from QuantConnect GitHub Analysis

This document summarizes the critical updates made to the project based on comprehensive analysis of QuantConnect's official GitHub repositories.

### Source Documentation

- [QuantConnect GitHub Resources Guide](../development/QUANTCONNECT_GITHUB_GUIDE.md) - Complete LEAN architecture and patterns (Version 2.0.0, 1,683 lines)
- [GitHub Retry Results](GITHUB_RETRY_RESULTS.md) - Interface definitions and architectural details

---

## 1. IV-Based Greeks (LEAN PR #6720)

### What Changed

**CRITICAL UPDATE**: Greeks calculations now use implied volatility instead of historical volatility.

### Key Impact

- **NO warmup period required** for Greeks calculations
- Greeks values match Interactive Brokers and major brokerages
- Values available immediately upon option data arrival
- More accurate than previous historical-based approach

### Default Pricing Models

- **European options**: Black-Scholes model
- **American options**: Bjerksund-Stensland model

### New Properties

```python
# Immediate access to IV-based Greeks (no warmup needed)
delta = contract.Greeks.Delta
gamma = contract.Greeks.Gamma
vega = contract.Greeks.Vega
theta = contract.Greeks.Theta
rho = contract.Greeks.Rho
theta_per_day = contract.Greeks.ThetaPerDay  # Daily theta decay (Theta / 365)

# Implied volatility directly from market prices
iv = contract.ImpliedVolatility
```

### Files Updated

- [algorithms/options_trading_bot.py](../../algorithms/options_trading_bot.py#L191-L194) - Added comment explaining warmup is for indicators only
- [algorithms/wheel_strategy.py](../../algorithms/wheel_strategy.py#L590-L612) - Added docstring noting IV-based Greeks
- [CLAUDE.md](../../CLAUDE.md#L280-L302) - Added dedicated IV-Based Greeks section
- [CLAUDE.md](../../CLAUDE.md#L696-L734) - Updated Options Trading Patterns section
- [docs/development/BEST_PRACTICES.md](../development/BEST_PRACTICES.md#L353-L387) - Added comprehensive IV-Based Greeks section

---

## 2. Charles Schwab Single Algorithm Limitation

### What Changed

**CRITICAL LIMITATION**: Charles Schwab allows **ONLY ONE algorithm per account**.

### Key Impact

- Deploying a second algorithm automatically stops the first
- All trading strategies must be combined into a single algorithm
- Cannot run separate algorithms for different strategies simultaneously
- Major architectural constraint for multi-strategy deployments

### Additional Schwab Details

- Built into LEAN core (not a separate plugin like Interactive Brokers)
- OAuth re-authentication required approximately weekly
- Uses QuantConnect cloud OAuth infrastructure (not direct Schwab API)
- Requires QuantConnect cloud for live trading (cannot use LEAN CLI locally with Schwab)

### Deployment Strategy

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

### Files Updated

- [algorithms/options_trading_bot.py](../../algorithms/options_trading_bot.py#L88-L92) - Added prominent warning comment
- [CLAUDE.md](../../CLAUDE.md#L257-L278) - Added Charles Schwab Brokerage section
- [docs/development/BEST_PRACTICES.md](../development/BEST_PRACTICES.md#L452-L479) - Added Charles Schwab Brokerage Specifics section

---

## 3. Multi-Leg Options with ComboOrders

### What Changed

**NEW CAPABILITY**: LEAN includes comprehensive multi-leg strategy support with atomic execution.

### Key Impact

- 24 dedicated files in LEAN for multi-leg strategy matching
- Atomic execution (all legs fill together or none fill)
- Single commission calculation per combo
- Prevents holding unbalanced positions
- Automatic strategy detection (butterflies, condors, iron condors, spreads)

### Available ComboOrder Types

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

### Butterfly Example

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

### Files Updated

- [algorithms/options_trading_bot.py](../../algorithms/options_trading_bot.py#L257-L278) - Added comprehensive ComboOrder documentation in option filter method
- [CLAUDE.md](../../CLAUDE.md#L304-L332) - Added Multi-Leg Options with ComboOrders section
- [docs/development/BEST_PRACTICES.md](../development/BEST_PRACTICES.md#L413-L450) - Added Multi-Leg Strategies with ComboOrders section

---

## 4. LEAN Engine Handler Lifecycle

### What Changed

**NEW UNDERSTANDING**: Complete LEAN engine initialization and execution lifecycle documented.

### Handler Initialization Sequence

```
Setup Phase:
1. IResultHandler.Initialize()
2. IDataFeed.Initialize()
3. ITransactionHandler.Initialize()
4. IRealTimeHandler.Initialize()

Execution Phase:
5. AlgorithmManager.Run() - Main trading loop
6. IDataFeed.Run() - Streams TimeSlice data
7. Algorithm.OnData(slice) - User algorithm processes data
8. ITransactionHandler - Processes order submissions

Shutdown Phase:
9. IDataFeed.Exit()
10. IResultHandler.Exit()
11. 30-second timeout for graceful shutdown
```

### Key Insights

- Result handler starts before data feed
- Data feed runs in separate thread, yields TimeSlice objects
- Transaction handler manages order submission and brokerage communication
- Real-time handler manages scheduled events and market hours
- Shutdown has 30-second timeout; use OnEndOfAlgorithm() for cleanup

### Documentation

- [QuantConnect GitHub Resources Guide](../development/QUANTCONNECT_GITHUB_GUIDE.md) - Complete handler lifecycle documentation
- [GitHub Retry Results](GITHUB_RETRY_RESULTS.md) - Interface definitions for all handlers

---

## Files Modified Summary

### Algorithm Files

| File | Changes | Lines |
|------|---------|-------|
| [algorithms/options_trading_bot.py](../../algorithms/options_trading_bot.py) | Added Schwab limitation warning, Greeks warmup clarification, ComboOrder documentation | 88-92, 191-194, 257-278 |
| [algorithms/wheel_strategy.py](../../algorithms/wheel_strategy.py) | Added IV-based Greeks note | 590-612 |

### Documentation Files

| File | Changes | Lines |
|------|---------|-------|
| [CLAUDE.md](../../CLAUDE.md) | Added Platform-Specific Limitations section, updated Options Trading Patterns | 255-332, 696-734 |
| [docs/development/BEST_PRACTICES.md](../development/BEST_PRACTICES.md) | Added IV-Based Greeks, ComboOrders, Schwab Specifics sections | 353-487 |
| [docs/development/QUANTCONNECT_GITHUB_GUIDE.md](../development/QUANTCONNECT_GITHUB_GUIDE.md) | Complete rewrite from web search to repository analysis (v2.0.0) | 1-1683 |
| [docs/quantconnect/GITHUB_RETRY_RESULTS.md](GITHUB_RETRY_RESULTS.md) | New file with interface definitions and architectural details | NEW |
| [docs/quantconnect/UPDATES_SUMMARY.md](UPDATES_SUMMARY.md) | This file | NEW |

---

## Action Items for Developers

### Immediate Actions

1. **Review IV-Based Greeks**: Remove any warmup logic for Greeks calculations
2. **Schwab Constraint**: Ensure all strategies are combined in single algorithm
3. **ComboOrders**: Update multi-leg strategies to use ComboOrders for atomic execution

### Code Review Checklist

- [ ] Remove Greeks warmup code (if present)
- [ ] Verify all strategies combined in single algorithm for Schwab
- [ ] Update multi-leg strategies to use ComboOrders
- [ ] Use `ThetaPerDay` property for daily theta calculations
- [ ] Test that Greeks are available immediately without warmup

### Testing Requirements

- [ ] Verify Greeks values match broker Greeks (Interactive Brokers, Schwab)
- [ ] Test ComboOrder execution for butterflies and condors
- [ ] Confirm single algorithm constraint for Schwab deployment
- [ ] Validate atomic execution of multi-leg positions

---

## Related Documentation

- [QuantConnect GitHub Resources Guide](../development/QUANTCONNECT_GITHUB_GUIDE.md) - **START HERE** - Complete LEAN architecture, Algorithm Framework patterns, working examples
- [GitHub Retry Results](GITHUB_RETRY_RESULTS.md) - Interface definitions, architectural diagrams, direct GitHub links
- [Development Best Practices](../development/BEST_PRACTICES.md) - Trading safety, risk management, backtesting standards
- [Compute Nodes Setup](../infrastructure/SETUP_SUMMARY.md) - Resource management and node configuration

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-30 | Initial documentation of QuantConnect framework updates |

---

**Status**: ✓ Complete

All project files updated with critical QuantConnect discoveries from GitHub repository analysis.
