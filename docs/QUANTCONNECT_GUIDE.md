# QuantConnect Development Guide

This guide covers QuantConnect/LEAN platform essentials for the trading bot.

## Compute Node Selection

This project is configured for optimal QuantConnect compute nodes:

| Node Type | Model | Cores | RAM | Cost/Month | Purpose |
|-----------|-------|-------|-----|------------|---------|
| **Backtesting** | **B8-16** | 8 @ 4.9GHz | 16GB | $28 | Options data + multi-chain spreads |
| **Research** | **R8-16** | 8 @ 2.4GHz | 16GB | $14 | LLM ensemble + strategy exploration |
| **Live Trading** | **L2-4** | 2 @ 2.6GHz | 4GB | $50 | Real-time options trading |

Total monthly cost: $92/month

### Why These Nodes?

- **B8-16**: High RAM for 500+ option contracts, 8 cores for Greeks calculations
- **R8-16**: Sufficient for LLM API calls (not local inference), best value for research
- **L2-4**: Dual-core for trading loop + scanners, colocated for <100ms latency

### Analyze Algorithm Requirements

```bash
# Get automatic node recommendations
python scripts/deploy_with_nodes.py algorithms/options_trading_bot.py --analyze-only
```

### Deploy with Node Selection

```bash
# Backtest deployment
python scripts/deploy_with_nodes.py algorithms/options_trading_bot.py --type backtest

# Live deployment
python scripts/deploy_with_nodes.py algorithms/options_trading_bot.py --type live --node L2-4
```

### Resource Monitoring

The algorithm automatically monitors resources every 30 seconds:

```python
# Configured in config/settings.json
"quantconnect": {
  "monitoring": {
    "enabled": true,
    "check_interval_seconds": 30,
    "memory_warning_pct": 80,
    "memory_critical_pct": 90
  }
}
```

See [Compute Nodes Documentation](infrastructure/COMPUTE_NODES.md) for complete guide.

## Algorithm Structure

```python
from AlgorithmImports import *

class MyAlgorithm(QCAlgorithm):
    def Initialize(self) -> None:
        # Python API uses snake_case for methods
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        self.symbol = self.add_equity("SPY", Resolution.Daily).Symbol

    def OnData(self, data: Slice) -> None:
        # ContainsKey is a framework method (PascalCase exception)
        if not data.ContainsKey(self.symbol):
            return
        # Trading logic here
```

## Common APIs

**Note**: Python API uses snake_case for methods (not C# PascalCase)

| Method | Purpose |
|--------|---------|
| `self.add_equity(ticker, resolution)` | Subscribe to stock data |
| `self.rsi(symbol, period)` | Create RSI indicator |
| `self.set_holdings(symbol, weight)` | Set position size |
| `self.liquidate(symbol)` | Close position |
| `self.Portfolio[symbol].Invested` | Check if holding (Portfolio is PascalCase) |
| `self.set_warm_up(periods)` | Warm up indicators |

## Critical Patterns

**Always validate data:**

```python
# ContainsKey and IsReady are framework properties (PascalCase exceptions)
if data.ContainsKey(self.symbol) and self.indicator.IsReady:
    # Safe to proceed
```

**Always warm up indicators:**

```python
# Python API uses snake_case for methods
self.set_warm_up(self.lookback_period)
# In OnData:
# IsWarmingUp is a framework property (PascalCase exception)
if self.IsWarmingUp:
    return
```

## Platform-Specific Limitations

### Charles Schwab Brokerage

**CRITICAL LIMITATION**: Charles Schwab allows **ONLY ONE algorithm per account**.

- Deploying a second algorithm automatically stops the first
- All trading strategies must be combined into a single algorithm
- Cannot run separate algorithms for different strategies simultaneously
- OAuth re-authentication required approximately weekly

```python
# WRONG - Second algorithm will stop first
# Algorithm 1: Options scanner
# Algorithm 2: Equity momentum

# CORRECT - Combine into single algorithm
class UnifiedTradingBot(QCAlgorithm):
    def Initialize(self):
        self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)
        # Combine all strategies in one algorithm
        self.options_strategy = OptionsScanner()
        self.equity_strategy = EquityMomentum()
```

### IV-Based Greeks (LEAN PR #6720)

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

**Key Changes:**

- Greeks calculated using IV from option prices
- Values match Interactive Brokers and major brokerages
- Default models: Black-Scholes (European), Bjerksund-Stensland (American)
- No warmup period required for Greeks calculations

### Multi-Leg Options with ComboOrders

**For butterflies, condors, spreads - use ComboOrders for atomic execution.**

ComboOrders are FULLY SUPPORTED on Charles Schwab (as of 2025-11-30)

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

**Available ComboOrder Types:**

| Order Type | Schwab Support | Description |
|-----------|---------------|-------------|
| `ComboMarketOrder()` | SUPPORTED | Execute at market |
| `ComboLimitOrder()` | SUPPORTED | Net limit price across all legs |
| `ComboLegLimitOrder()` | NOT supported | Individual leg limits (not available on Schwab) |

**Important for Charles Schwab:**

- Use `ComboLimitOrder()` with net debit/credit pricing
- Do NOT use `ComboLegLimitOrder()` - individual leg limits not supported
- Do NOT specify `order_price` parameter in `Leg.Create()` calls

**Benefits:**

- Automatic strategy detection (LEAN has 24 files for multi-leg matching)
- Atomic execution (all-or-nothing fills)
- Single commission per combo
- Prevents holding unbalanced positions

## Risk Management Integration

Use the RiskManager class from `models/risk_manager.py`:

```python
from models import RiskManager, RiskLimits

# Configure limits
limits = RiskLimits(
    max_position_size=0.25,      # 25% max per position
    max_daily_loss=0.03,         # 3% daily loss limit
    max_drawdown=0.10,           # 10% max drawdown
    max_risk_per_trade=0.02,     # 2% risk per trade
)

risk_manager = RiskManager(starting_equity=100000, limits=limits)
```

## Resources

- [QuantConnect Documentation](https://www.quantconnect.com/docs)
- [LEAN Engine GitHub](https://github.com/QuantConnect/Lean)
- [QuantConnect Forum](https://www.quantconnect.com/forum)
- [Charles Schwab API](https://developer.schwab.com/)
- [QuantConnect GitHub Resources Guide](development/QUANTCONNECT_GITHUB_GUIDE.md)
