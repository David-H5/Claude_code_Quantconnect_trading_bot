# QuantConnect Documentation

Comprehensive documentation for building algorithmic trading strategies on the QuantConnect LEAN platform.

## Quick Navigation

### Core Documentation

| Document | Description |
|----------|-------------|
| [LEAN Engine](LEAN_ENGINE.md) | Core platform setup, Python API, algorithm structure |
| [Options Trading](OPTIONS_TRADING.md) | Options contracts, Greeks, pricing models |
| [Universe Selection](UNIVERSE_SELECTION.md) | Dynamic security selection and filtering |
| [Risk Management](RISK_MANAGEMENT.md) | Position sizing, portfolio protection, drawdown limits |
| [Brokerage Integration](BROKERAGE_INTEGRATION.md) | Charles Schwab and other broker connections |
| [Backtesting](BACKTESTING.md) | Best practices, avoiding pitfalls, validation |

### Advanced Topics

| Document | Description |
|----------|-------------|
| [Algorithm Framework](ALGORITHM_FRAMEWORK.md) | Alpha models, portfolio construction, execution models |
| [Custom Data](CUSTOM_DATA.md) | Alternative data, custom data sources, Object Store |
| [Machine Learning](MACHINE_LEARNING.md) | Scikit-learn, TensorFlow, Keras integration |
| [Futures, Forex, Crypto](FUTURES_FOREX_CRYPTO.md) | Multi-asset trading, continuous contracts |
| [Research Environment](RESEARCH_ENVIRONMENT.md) | Jupyter notebooks, QuantBook API, data analysis |
| [Data Consolidation](DATA_CONSOLIDATION.md) | Consolidators, scheduled events, warm-up, logging |

## What is QuantConnect?

QuantConnect is a cloud-based algorithmic trading platform that provides:

- **LEAN Engine**: Open-source backtesting and live trading engine
- **Multi-Asset Support**: Equities, options, futures, forex, crypto
- **Cloud Infrastructure**: Run backtests in the cloud with historical data
- **Live Trading**: Connect to brokerages for automated trading
- **Research Environment**: Jupyter notebooks for strategy development

## Platform Status (2025)

| Component | Current Status |
|-----------|----------------|
| **Python Version** | 3.11.7 |
| **LEAN Engine** | v2.5.17269 (Aug 2025) |
| **TensorFlow** | 2.16.1 |
| **Contributors** | 180+ engineers |
| **Hedge Funds** | 300+ powered by LEAN |
| **Brokerages** | 15+ including Schwab, IB, Alpaca |

### Key 2025 Updates

- **Combo Orders**: Multi-leg options orders now fully supported
- **GPU Nodes**: Out of beta, 100x+ acceleration for ML
- **PDT Rule**: FINRA approved elimination of $25K minimum (pending SEC approval)
- **Crypto Futures**: Binance/Bybit perpetuals with 700+ pairs
- **Schwab Integration**: Full OAuth with email refresh notifications

## Algorithm Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    Algorithm Lifecycle                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Initialize()     → Set dates, cash, add securities      │
│         │                                                   │
│         ▼                                                   │
│  2. set_warm_up()    → Warm up indicators with history      │
│         │                                                   │
│         ▼                                                   │
│  3. OnData()         → Process incoming market data         │
│         │            → Execute trading logic                │
│         │                                                   │
│         ▼                                                   │
│  4. OnEndOfDay()     → Daily portfolio rebalancing          │
│         │                                                   │
│         ▼                                                   │
│  5. OnEndOfAlgorithm() → Final cleanup and reporting        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Basic Algorithm Template

```python
from AlgorithmImports import *

class MyTradingAlgorithm(QCAlgorithm):
    """
    Basic algorithm template following QuantConnect best practices.
    """

    def Initialize(self) -> None:
        """Set up the algorithm parameters and subscriptions."""
        # Backtest period (Python API uses snake_case)
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)

        # Add securities
        self.symbol = self.add_equity("SPY", Resolution.Daily).Symbol

        # Create indicators (methods use snake_case)
        self.rsi = self.rsi(self.symbol, 14, MovingAverageType.Wilders, Resolution.Daily)
        self.sma = self.sma(self.symbol, 50, Resolution.Daily)

        # Warm up indicators
        self.set_warm_up(50)

        # Risk management
        self.set_risk_management(MaximumDrawdownPercentPerSecurity(0.05))

    def OnData(self, data: Slice) -> None:
        """Process incoming data and execute trading logic."""
        # Skip if warming up or data not ready
        # IsWarmingUp is a framework property (PascalCase exception)
        if self.IsWarmingUp:
            return

        # ContainsKey is a framework method (PascalCase exception)
        if not data.ContainsKey(self.symbol):
            return

        if not self.rsi.IsReady or not self.sma.IsReady:
            return

        # Trading logic
        price = data[self.symbol].Close

        if self.rsi.Current.Value < 30 and price > self.sma.Current.Value:
            # Oversold + above SMA = buy signal (Python API uses snake_case)
            self.set_holdings(self.symbol, 0.95)
        elif self.rsi.Current.Value > 70:
            # Overbought = exit signal
            self.liquidate(self.symbol)

    def OnEndOfDay(self, symbol: Symbol) -> None:
        """Called at the end of each trading day."""
        self.Log(f"EOD - Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
```

## Key Concepts

### Resolution Types

| Resolution | Description | Use Case |
|------------|-------------|----------|
| `Resolution.Tick` | Every trade | High-frequency strategies |
| `Resolution.Second` | Every second | Scalping |
| `Resolution.Minute` | Every minute | Intraday trading |
| `Resolution.Hour` | Every hour | Swing trading |
| `Resolution.Daily` | Daily bars | Position trading |

### Data Access Patterns

```python
# Safe data access
if data.ContainsKey(self.symbol):
    bar = data[self.symbol]
    close_price = bar.Close
    volume = bar.Volume

# Option chain access (Python API uses snake_case option_chains)
chain = data.option_chains.get(self.option_symbol)
if chain:
    for contract in chain:
        # Process contracts
        pass

# Slice properties (PascalCase exceptions for data dictionaries)
data.Bars           # TradeBar data
data.QuoteBars      # Quote data
data.option_chains  # Options data (use snake_case .option_chains in Python)
data.future_chains  # Futures data (use snake_case .future_chains in Python)
```

### Order Types

```python
# Python API uses snake_case for order methods

# Market order - immediate execution at market price
self.market_order(symbol, quantity)

# Limit order - execute at specified price or better
self.limit_order(symbol, quantity, limit_price)

# Stop market order - triggered when price reaches stop
self.stop_market_order(symbol, quantity, stop_price)

# Stop limit order - stop trigger + limit execution
self.stop_limit_order(symbol, quantity, stop_price, limit_price)

# Trailing stop - follows price movement
self.trailing_stop_order(symbol, quantity, trailing_amount)

# Set portfolio allocation (% of portfolio)
self.set_holdings(symbol, 0.25)  # 25% of portfolio
```

## Project Structure Recommendation

```
my_algorithm/
├── main.py                 # Algorithm entry point
├── config/
│   └── settings.py         # Configuration parameters
├── indicators/
│   ├── __init__.py
│   └── custom_indicators.py
├── models/
│   ├── __init__.py
│   ├── risk_model.py
│   └── signal_model.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
└── tests/
    └── test_strategy.py
```

## Common Mistakes to Avoid

1. **Look-Ahead Bias**: Using future data in current decisions
2. **Survivorship Bias**: Not accounting for delisted securities
3. **Overfitting**: Optimizing too many parameters on historical data
4. **Ignoring Costs**: Not accounting for commissions and slippage
5. **Missing Warmup**: Using indicators before they're ready
6. **Unchecked Data**: Accessing data without checking existence

## Resources

- [QuantConnect Documentation](https://www.quantconnect.com/docs)
- [LEAN GitHub Repository](https://github.com/QuantConnect/Lean)
- [QuantConnect Forum](https://www.quantconnect.com/forum)
- [QuantConnect Tutorials](https://www.quantconnect.com/tutorials)
- [Algorithm Examples](https://www.quantconnect.com/project)

---

*Last Updated: November 2025*
