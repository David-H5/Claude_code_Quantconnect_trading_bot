# LEAN Engine Documentation

The LEAN Algorithmic Trading Engine is the open-source foundation of QuantConnect. This document covers setup, Python API, and core functionality.

## Table of Contents

- [What is LEAN?](#what-is-lean)
- [Installation](#installation)
- [Algorithm Structure](#algorithm-structure)
- [Event Handlers](#event-handlers)
- [Python API Reference](#python-api-reference)
- [Data Handling](#data-handling)
- [Data Normalization](#data-normalization)
- [Reality Modeling](#reality-modeling)
- [Indicators](#indicators)
- [Scheduling](#scheduling)
- [Portfolio Statistics](#portfolio-statistics)
- [Logging and Debugging](#logging-and-debugging)
- [Compute Resources](#compute-resources)

## What is LEAN?

LEAN is a fully-featured algorithmic trading engine that:

- Processes market data in real-time or historical mode
- Supports multiple asset classes (equities, options, futures, forex, crypto)
- Handles order management and portfolio tracking
- Provides built-in technical indicators
- Integrates with multiple brokerages

> **2025 Update**: LEAN is powered by a global community of 180+ engineers and powers more than 300+ hedge funds. The latest version is **2.5.17315** (as of October 2025), running **Python 3.11.7** with updated libraries including pandas 2.1.4, TensorFlow 2.16.1, OpenAI 1.14.3, LangChain 0.1.12, and new additions like TPOT 0.12.2, llama-index 0.10.19, and PyCaret 3.3.0.

## Installation

### Local Development with LEAN CLI

```bash
# Install LEAN CLI
pip install lean

# Initialize a new project
lean init my-project

# Pull historical data
lean data download --data-type equity/usa/minute/spy

# Run a backtest locally
lean backtest my-algorithm.py

# Deploy to cloud
lean cloud push --project my-project
```

### Docker Installation

```bash
# Pull LEAN image
docker pull quantconnect/lean

# Run backtest in container
docker run -v $(pwd):/Lean/Launcher/bin/Debug \
    quantconnect/lean mono QuantConnect.Lean.Launcher.exe \
    --algorithm-type-name MyAlgorithm \
    --algorithm-language Python
```

## Algorithm Structure

### Basic Algorithm Class

```python
from AlgorithmImports import *

class MyAlgorithm(QCAlgorithm):
    """
    All algorithms inherit from QCAlgorithm base class.
    """

    def Initialize(self) -> None:
        """
        Called once at the start. Set up everything here.

        Required:
        - Start/End dates (backtest) or SetStartDate only (live)
        - Starting cash
        - Security subscriptions

        Optional:
        - Indicators
        - Scheduled events
        - Risk management models
        - Benchmark
        """
        # Required setup
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)

        # Add securities
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol

        # Set benchmark
        self.SetBenchmark("SPY")

        # Set brokerage model
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)

    def OnData(self, data: Slice) -> None:
        """
        Called for every data point subscribed.
        Main trading logic goes here.
        """
        pass

    def OnOrderEvent(self, orderEvent: OrderEvent) -> None:
        """Called when order status changes."""
        if orderEvent.Status == OrderStatus.Filled:
            self.Log(f"Order filled: {orderEvent}")

    def OnEndOfDay(self, symbol: Symbol) -> None:
        """Called at market close for each symbol."""
        pass

    def OnEndOfAlgorithm(self) -> None:
        """Called once when algorithm terminates."""
        self.Log(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
```

### Lifecycle Methods

| Method | When Called | Use Case |
|--------|-------------|----------|
| `Initialize()` | Once at start | Setup |
| `OnWarmupFinished()` | After warmup completes | Start trading |
| `OnData(data)` | Every data point | Main logic |
| `OnOrderEvent(event)` | Order status change | Order tracking |
| `OnEndOfDay(symbol)` | Market close | Daily rebalancing |
| `OnEndOfAlgorithm()` | Algorithm ends | Cleanup |
| `OnSecuritiesChanged(changes)` | Universe changes | Dynamic updates |

## Event Handlers

### OnSecuritiesChanged

Called when securities are added or removed from the universe:

```python
def OnSecuritiesChanged(self, changes: SecurityChanges) -> None:
    """
    Handle universe changes - securities added or removed.

    Triggered by:
    - Universe selection changes
    - Security delisting
    - Contract expiration
    - Explicit RemoveSecurity() calls
    """
    # Process new securities
    for security in changes.AddedSecurities:
        symbol = security.Symbol
        self.Log(f"Added: {symbol}")

        # Initialize indicators for new security
        self.indicators[symbol] = self.RSI(symbol, 14)

        # Warm up with history
        history = self.History(symbol, 20, Resolution.Daily)
        for bar in history.itertuples():
            self.indicators[symbol].Update(bar.Index[1], bar.close)

    # Process removed securities
    for security in changes.RemovedSecurities:
        symbol = security.Symbol
        self.Log(f"Removed: {symbol}")

        # Liquidate position
        if self.Portfolio[symbol].Invested:
            self.Liquidate(symbol)

        # Cleanup
        if symbol in self.indicators:
            del self.indicators[symbol]
```

**Best Practice**: Avoid placing trades directly in `OnSecuritiesChanged`. Instead, set flags and handle trading in `OnData`.

### OnOrderEvent

Called when order status changes:

```python
def OnOrderEvent(self, orderEvent: OrderEvent) -> None:
    """
    Handle order status updates.

    Called for: Submitted, PartiallyFilled, Filled, Canceled, Invalid
    Note: In live trading, events are asynchronous
    """
    order = self.Transactions.GetOrderById(orderEvent.OrderId)

    if orderEvent.Status == OrderStatus.Submitted:
        self.Log(f"Order submitted: {order.Symbol}")

    elif orderEvent.Status == OrderStatus.PartiallyFilled:
        self.Log(f"Partial fill: {orderEvent.FillQuantity} @ ${orderEvent.FillPrice}")

    elif orderEvent.Status == OrderStatus.Filled:
        self.Log(f"Filled: {orderEvent.FillQuantity} @ ${orderEvent.FillPrice}")
        self.Log(f"Fees: ${orderEvent.OrderFee.Value.Amount}")

        # Implement OCO (One-Cancels-Other) logic
        if orderEvent.Symbol in self.stop_orders:
            self.Transactions.CancelOrder(self.stop_orders[orderEvent.Symbol])

    elif orderEvent.Status == OrderStatus.Canceled:
        self.Log(f"Order canceled: {order.Symbol}")

    elif orderEvent.Status == OrderStatus.Invalid:
        self.Log(f"Order INVALID: {orderEvent.Message}")
```

**Warning**: Avoid placing orders in `OnOrderEvent` to prevent infinite loops.

### OnDividends and OnSplits

```python
def OnDividends(self, dividends: Dividends) -> None:
    """Called when dividends are paid."""
    for symbol, dividend in dividends.items():
        self.Log(f"Dividend: {symbol} - ${dividend.Distribution}/share")

def OnSplits(self, splits: Splits) -> None:
    """Called when stock splits occur."""
    for symbol, split in splits.items():
        self.Log(f"Split: {symbol} - {split.SplitFactor}:1")

        # Reset indicators after split
        if symbol in self.indicators:
            self.indicators[symbol].Reset()
            # Re-warm up with adjusted data
```

### Algorithm Engine Execution Order

```text
1. Initialize()
2. For each time step:
   a. OnSecuritiesChanged() - universe updates
   b. Apply dividends to portfolio
   c. Handle splits (adjust holdings, open orders)
   d. Update consolidators with new data
   e. OnData() - main data handler
   f. Universe selection (if scheduled)
   g. Alpha model Update()
   h. Portfolio Construction model
   i. Risk Management model
   j. Execution model
```

## Python API Reference

### Adding Securities

```python
# Equities
equity = self.AddEquity("AAPL", Resolution.Minute)
symbol = equity.Symbol

# Options (adds underlying automatically)
option = self.AddOption("SPY", Resolution.Minute)
option.SetFilter(-5, 5, 0, 30)  # strikes, min DTE, max DTE

# Futures
future = self.AddFuture(Futures.Indices.SP500EMini, Resolution.Minute)
future.SetFilter(0, 90)  # days to expiration

# Forex
forex = self.AddForex("EURUSD", Resolution.Hour)

# Crypto
crypto = self.AddCrypto("BTCUSD", Resolution.Hour)
```

### Portfolio Management

```python
# Check if invested
if self.Portfolio[symbol].Invested:
    quantity = self.Portfolio[symbol].Quantity
    avg_price = self.Portfolio[symbol].AveragePrice
    unrealized_pnl = self.Portfolio[symbol].UnrealizedProfit

# Portfolio-level properties
total_value = self.Portfolio.TotalPortfolioValue
cash = self.Portfolio.Cash
margin_remaining = self.Portfolio.MarginRemaining
total_unrealized = self.Portfolio.TotalUnrealizedProfit
total_fees = self.Portfolio.TotalFees

# Set holdings as percentage of portfolio
self.SetHoldings(symbol, 0.5)  # 50% of portfolio

# Liquidate positions
self.Liquidate(symbol)      # Single position
self.Liquidate()            # All positions
```

### Order Management

```python
# Submit orders
ticket = self.MarketOrder(symbol, 100)
ticket = self.LimitOrder(symbol, 100, 150.00)
ticket = self.StopMarketOrder(symbol, -100, 145.00)
ticket = self.StopLimitOrder(symbol, -100, 145.00, 144.50)

# Order ticket properties
order_id = ticket.OrderId
status = ticket.Status
quantity_filled = ticket.QuantityFilled
average_fill_price = ticket.AverageFillPrice

# Update order
response = ticket.Update(UpdateOrderFields())
response.IsSuccess

# Cancel order
response = ticket.Cancel()

# Get order by ID
order = self.Transactions.GetOrderById(order_id)

# Get all open orders
open_orders = self.Transactions.GetOpenOrders(symbol)
```

### History Requests

```python
# Get historical data
history = self.History(symbol, 30, Resolution.Daily)

# Multiple symbols
history = self.History([symbol1, symbol2], 30, Resolution.Daily)

# Specific time period
history = self.History(symbol,
    datetime(2023, 1, 1),
    datetime(2023, 6, 30),
    Resolution.Daily)

# Working with history DataFrame
for bar in history.itertuples():
    close = bar.close
    volume = bar.volume

# Get specific columns
closes = history['close']
highs = history['high']
```

## Data Handling

### Slice Object

```python
def OnData(self, data: Slice) -> None:
    # Check data availability
    if not data.ContainsKey(self.symbol):
        return

    # Access TradeBar data
    if data.Bars.ContainsKey(self.symbol):
        bar = data.Bars[self.symbol]
        open_price = bar.Open
        high_price = bar.High
        low_price = bar.Low
        close_price = bar.Close
        volume = bar.Volume

    # Access QuoteBar data
    if data.QuoteBars.ContainsKey(self.symbol):
        quote = data.QuoteBars[self.symbol]
        bid = quote.Bid.Close
        ask = quote.Ask.Close
        spread = ask - bid

    # Access tick data
    if data.Ticks.ContainsKey(self.symbol):
        for tick in data.Ticks[self.symbol]:
            price = tick.Price
            quantity = tick.Quantity
```

### Consolidators

```python
def Initialize(self):
    # Consolidate minute data into 15-minute bars
    fifteen_minute = TradeBarConsolidator(timedelta(minutes=15))
    fifteen_minute.DataConsolidated += self.OnFifteenMinuteBar
    self.SubscriptionManager.AddConsolidator(self.symbol, fifteen_minute)

    # Daily consolidator
    daily = TradeBarConsolidator(timedelta(days=1))
    daily.DataConsolidated += self.OnDailyBar
    self.SubscriptionManager.AddConsolidator(self.symbol, daily)

def OnFifteenMinuteBar(self, sender, bar):
    """Called when 15-minute bar completes."""
    self.Log(f"15-min bar: {bar.Close}")

def OnDailyBar(self, sender, bar):
    """Called when daily bar completes."""
    self.Log(f"Daily bar: {bar.Close}")
```

## Data Normalization

> **2025 Verification**: QuantConnect's data normalization modes remain current with **four primary modes**: `Adjusted` (default), `Raw`, `SplitAdjusted`, and `TotalReturn`. The `ScaledRaw` mode is available for advanced use cases.

QuantConnect offers several modes to handle splits and dividends:

### Normalization Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `Adjusted` | Splits and dividends factored into price (default) | Smooth price curve for indicators |
| `Raw` | Prices as-is, dividends as cash, quantity adjusted on splits | Live trading simulation |
| `SplitAdjusted` | Only splits adjusted, dividends paid as cash | Charting with dividend income |
| `TotalReturn` | Dividends added to price | Total return analysis |
| `ScaledRaw` | Adjusted based on data before current algorithm time | Warm-up with adjusted data |

### Setting Data Normalization

```python
def Initialize(self):
    # Set normalization mode when adding security
    self.symbol = self.AddEquity(
        "TSLA",
        Resolution.Daily,
        dataNormalizationMode=DataNormalizationMode.Raw
    ).Symbol

    # Or set default for all securities
    self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Adjusted
```

### Normalization Behavior

```python
# With Adjusted or TotalReturn: dividends in price
# With other modes: dividends added to cash

# With Raw mode: splits adjust quantity automatically
def OnSplits(self, splits):
    for symbol, split in splits.items():
        # LEAN automatically adjusts Portfolio[symbol].Quantity
        self.Log(f"Split {split.SplitFactor}:1, new qty: {self.Portfolio[symbol].Quantity}")
```

**Important**:

- Cannot switch normalization during backtest
- Data normalization is NOT applied in live trading
- Use `ScaledRaw` to warm up indicators when using `Raw` data

## Reality Modeling

Reality models simulate real-world trading conditions in backtests.

### Slippage Models

> **2025 Verification**: All three built-in slippage models remain current: `ConstantSlippageModel`, `VolumeShareSlippageModel`, and `MarketImpactSlippageModel`. Note that `MarketImpactSlippageModel` parameters were calibrated ~20 years ago and may need recalibration for modern markets.

```python
def Initialize(self):
    self.symbol = self.AddEquity("SPY", Resolution.Minute).Symbol

    # Constant percentage slippage
    self.Securities[self.symbol].SetSlippageModel(
        ConstantSlippageModel(0.001)  # 0.1% slippage
    )

    # Volume-based slippage (default: volumeLimit=0.025, priceImpact=0.1)
    self.Securities[self.symbol].SetSlippageModel(
        VolumeShareSlippageModel(0.01, 0.10)  # price impact * (order/volume)^2
    )

    # Market impact slippage (consumption of order book)
    # WARNING: Default parameters calibrated ~2 decades ago
    self.Securities[self.symbol].SetSlippageModel(
        MarketImpactSlippageModel(self)
    )
```

### Custom Slippage Model

```python
class MySlippageModel:
    """Custom slippage based on volatility."""

    def GetSlippageApproximation(self, asset, order):
        # Higher slippage for larger orders
        order_value = abs(order.Quantity * asset.Price)
        base_slippage = 0.0005  # 0.05%

        # Scale by order size
        if order_value > 100000:
            return asset.Price * base_slippage * 2
        return asset.Price * base_slippage

# Usage
self.Securities[symbol].SetSlippageModel(MySlippageModel())
```

### Fee Models

```python
def Initialize(self):
    # Constant fee
    self.Securities[symbol].SetFeeModel(ConstantFeeModel(1.00))  # $1 per trade

    # Interactive Brokers fees
    self.Securities[symbol].SetFeeModel(InteractiveBrokersFeeModel())

    # No fees
    self.Securities[symbol].SetFeeModel(ConstantFeeModel(0))
```

### Custom Fee Model

```python
class MyFeeModel(FeeModel):
    """Tiered fee structure."""

    def GetOrderFee(self, parameters):
        order = parameters.Order
        security = parameters.Security

        quantity = abs(order.Quantity)
        price = security.Price

        # Per-share pricing with minimum
        per_share = 0.005
        fee = max(1.0, quantity * per_share)

        return OrderFee(CashAmount(fee, "USD"))

self.Securities[symbol].SetFeeModel(MyFeeModel())
```

### Fill Models

```python
# Immediate fill (default)
self.Securities[symbol].SetFillModel(ImmediateFillModel())

# Partial fills based on volume
self.Securities[symbol].SetFillModel(PartialFillModel())
```

### Using SetSecurityInitializer

```python
def Initialize(self):
    # Apply models to all securities
    self.SetSecurityInitializer(self.CustomSecurityInitializer)

    self.AddUniverse(self.CoarseSelection)

def CustomSecurityInitializer(self, security):
    """Called for each security added."""
    security.SetFeeModel(ConstantFeeModel(0))
    security.SetSlippageModel(ConstantSlippageModel(0.001))
    security.SetFillModel(ImmediateFillModel())
```

## Indicators

> **2025 Update**: LEAN provides **100+ pre-built technical indicators** and candlestick patterns. Popular indicators include SMA, EMA, RSI, MACD, BB (Bollinger Bands), ATR, VWAP, ADX, AROON, OBV, CCI, Stochastic, and many more. For the complete list, see the [Supported Indicators documentation](https://www.quantconnect.com/docs/v2/writing-algorithms/indicators/supported-indicators).

### Built-in Indicators

```python
def Initialize(self):
    # Moving Averages
    self.sma = self.SMA(self.symbol, 20, Resolution.Daily)
    self.ema = self.EMA(self.symbol, 20, Resolution.Daily)

    # Momentum Indicators
    self.rsi = self.RSI(self.symbol, 14, MovingAverageType.Wilders, Resolution.Daily)
    self.macd = self.MACD(self.symbol, 12, 26, 9, MovingAverageType.Exponential, Resolution.Daily)
    self.cci = self.CCI(self.symbol, 20, MovingAverageType.Simple, Resolution.Daily)
    self.stoch = self.STO(self.symbol, 14, 3, 3, Resolution.Daily)

    # Volatility Indicators
    self.bb = self.BB(self.symbol, 20, 2, MovingAverageType.Simple, Resolution.Daily)
    self.atr = self.ATR(self.symbol, 14, MovingAverageType.Simple, Resolution.Daily)
    self.std = self.STD(self.symbol, 20, Resolution.Daily)

    # Volume Indicators
    self.obv = self.OBV(self.symbol, Resolution.Daily)
    self.vwap = self.VWAP(self.symbol, Resolution.Daily)
    self.ad = self.AD(self.symbol, Resolution.Daily)

    # Trend Indicators
    self.adx = self.ADX(self.symbol, 14, Resolution.Daily)
    self.aroon = self.AROON(self.symbol, 25, Resolution.Daily)

    # Warm up indicators
    self.SetWarmUp(50)

def OnData(self, data: Slice) -> None:
    if self.IsWarmingUp:
        return

    # Check if indicators are ready
    if not self.sma.IsReady or not self.rsi.IsReady:
        return

    # Access indicator values
    sma_value = self.sma.Current.Value
    rsi_value = self.rsi.Current.Value

    # MACD components
    macd_value = self.macd.Current.Value
    signal = self.macd.Signal.Current.Value
    histogram = self.macd.Histogram.Current.Value

    # Bollinger Bands components
    upper = self.bb.UpperBand.Current.Value
    middle = self.bb.MiddleBand.Current.Value
    lower = self.bb.LowerBand.Current.Value
```

### Custom Indicators

```python
class CustomMomentum(PythonIndicator):
    """Custom momentum indicator example."""

    def __init__(self, name, period):
        super().__init__()
        self.Name = name
        self.period = period
        self.queue = deque(maxlen=period)
        self.Value = 0

    def Update(self, input) -> bool:
        """Called with each new data point."""
        self.queue.append(input.Close)

        if len(self.queue) == self.period:
            self.Value = (self.queue[-1] - self.queue[0]) / self.queue[0]
            return True
        return False

# Register custom indicator
def Initialize(self):
    self.custom = CustomMomentum("MyMomentum", 20)
    self.RegisterIndicator(self.symbol, self.custom, Resolution.Daily)
```

## Scheduling

### Scheduled Events

```python
def Initialize(self):
    # Schedule function at specific time
    self.Schedule.On(
        self.DateRules.EveryDay(self.symbol),
        self.TimeRules.AfterMarketOpen(self.symbol, 30),
        self.RebalancePortfolio
    )

    # Schedule at market close
    self.Schedule.On(
        self.DateRules.EveryDay(self.symbol),
        self.TimeRules.BeforeMarketClose(self.symbol, 15),
        self.EndOfDayReport
    )

    # Weekly schedule
    self.Schedule.On(
        self.DateRules.Every(DayOfWeek.Monday),
        self.TimeRules.At(10, 0),
        self.WeeklyRebalance
    )

def RebalancePortfolio(self):
    """Called 30 minutes after market open."""
    self.Log("Rebalancing portfolio...")

def EndOfDayReport(self):
    """Called 15 minutes before market close."""
    self.Log(f"EOD Portfolio: ${self.Portfolio.TotalPortfolioValue:,.2f}")

def WeeklyRebalance(self):
    """Called every Monday at 10:00 AM."""
    self.Log("Weekly rebalance...")
```

### Date Rules

| Rule | Description |
|------|-------------|
| `DateRules.EveryDay()` | Every trading day |
| `DateRules.Every(DayOfWeek.Monday)` | Every Monday |
| `DateRules.MonthStart()` | First trading day of month |
| `DateRules.MonthEnd()` | Last trading day of month |
| `DateRules.WeekStart()` | First trading day of week |
| `DateRules.WeekEnd()` | Last trading day of week |
| `DateRules.On(year, month, day)` | Specific date |

### Time Rules

| Rule | Description |
|------|-------------|
| `TimeRules.AfterMarketOpen(symbol, minutes)` | After market opens |
| `TimeRules.BeforeMarketClose(symbol, minutes)` | Before market closes |
| `TimeRules.At(hour, minute)` | Specific time |
| `TimeRules.Every(timedelta)` | Repeating interval |
| `TimeRules.Midnight` | At midnight |
| `TimeRules.Noon` | At noon |

## Portfolio Statistics

QuantConnect calculates comprehensive performance metrics for backtest analysis.

### Accessing Statistics in Algorithm

```python
def OnEndOfAlgorithm(self):
    """Access performance statistics at end of backtest."""
    # Basic metrics
    total_return = self.Portfolio.TotalPortfolioValue / self.StartingCash - 1
    self.Log(f"Total Return: {total_return:.2%}")

    # Win/loss tracking (manual implementation)
    win_rate = self.wins / max(1, self.wins + self.losses)
    self.Log(f"Win Rate: {win_rate:.1%}")
```

### Key Performance Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Sharpe Ratio** | Risk-adjusted return (excess return / volatility) | > 1.0 |
| **Sortino Ratio** | Like Sharpe but uses downside deviation only | > 1.5 |
| **Alpha** | Excess return vs benchmark | > 0 |
| **Beta** | Correlation/volatility vs benchmark | Depends on strategy |
| **Max Drawdown** | Largest peak-to-trough decline | < 20% |
| **CAGR** | Compound Annual Growth Rate | > risk-free rate |
| **Profit Factor** | Gross profits / gross losses | > 1.5 |
| **Win Rate** | Winning trades / total trades | > 50% |

### Custom Statistics

```python
def Initialize(self):
    self.wins = 0
    self.losses = 0
    self.gross_profit = 0
    self.gross_loss = 0
    self.peak_equity = self.StartingCash
    self.max_drawdown = 0

def OnOrderEvent(self, orderEvent):
    if orderEvent.Status == OrderStatus.Filled:
        # Track trade results
        if orderEvent.Direction == OrderDirection.Sell:
            pnl = orderEvent.FillQuantity * (orderEvent.FillPrice - self.entry_price)
            if pnl > 0:
                self.wins += 1
                self.gross_profit += pnl
            else:
                self.losses += 1
                self.gross_loss += abs(pnl)

def OnData(self, data):
    # Track drawdown
    current_equity = self.Portfolio.TotalPortfolioValue
    if current_equity > self.peak_equity:
        self.peak_equity = current_equity

    drawdown = (self.peak_equity - current_equity) / self.peak_equity
    self.max_drawdown = max(self.max_drawdown, drawdown)

def OnEndOfAlgorithm(self):
    # Calculate and log custom statistics
    total_trades = self.wins + self.losses
    win_rate = self.wins / max(1, total_trades)
    profit_factor = self.gross_profit / max(0.01, self.gross_loss)

    self.SetRuntimeStatistic("Custom Win Rate", f"{win_rate:.1%}")
    self.SetRuntimeStatistic("Profit Factor", f"{profit_factor:.2f}")
    self.SetRuntimeStatistic("Max Drawdown", f"{self.max_drawdown:.2%}")
    self.SetRuntimeStatistic("Total Trades", str(total_trades))
```

### Benchmark Comparison

```python
def Initialize(self):
    # Set benchmark for comparison
    self.SetBenchmark("SPY")

    # Or use a custom benchmark
    self.SetBenchmark(lambda dt: self.History(self.spy, 1, Resolution.Daily)['close'].iloc[-1])

def OnEndOfAlgorithm(self):
    # Log benchmark comparison
    self.Log("See backtest results for Alpha and Beta vs benchmark")
```

### Sharpe and Sortino Calculation

```python
import numpy as np

class StatisticsAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)
        self.symbol = self.AddEquity("SPY").Symbol
        self.daily_returns = []
        self.previous_value = self.Portfolio.TotalPortfolioValue

        # Schedule daily return tracking
        self.Schedule.On(
            self.DateRules.EveryDay(self.symbol),
            self.TimeRules.BeforeMarketClose(self.symbol, 1),
            self.RecordDailyReturn
        )

    def RecordDailyReturn(self):
        current_value = self.Portfolio.TotalPortfolioValue
        daily_return = (current_value - self.previous_value) / self.previous_value
        self.daily_returns.append(daily_return)
        self.previous_value = current_value

    def OnEndOfAlgorithm(self):
        if len(self.daily_returns) < 30:
            return

        returns = np.array(self.daily_returns)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate

        # Sharpe Ratio (annualized)
        excess_returns = returns - risk_free_rate
        sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

        # Sortino Ratio (uses downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else np.std(returns)
        sortino = np.sqrt(252) * np.mean(excess_returns) / downside_std

        self.SetRuntimeStatistic("Sharpe Ratio", f"{sharpe:.2f}")
        self.SetRuntimeStatistic("Sortino Ratio", f"{sortino:.2f}")
```

### Tracking Error and Information Ratio

```python
def CalculateTrackingMetrics(self, portfolio_returns, benchmark_returns):
    """Calculate tracking error and information ratio."""
    # Tracking Error: std of excess returns
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = np.std(excess_returns) * np.sqrt(252)

    # Information Ratio: excess return / tracking error
    annualized_excess = np.mean(excess_returns) * 252
    information_ratio = annualized_excess / tracking_error if tracking_error > 0 else 0

    return tracking_error, information_ratio
```

## Logging and Debugging

### Logging

```python
# Standard logging
self.Log("This is a log message")
self.Log(f"Price: {price}, RSI: {rsi_value:.2f}")

# Debug logging (more verbose)
self.Debug("Debug message - only in debug mode")

# Error logging
self.Error("Something went wrong!")

# Plotting (shows in backtest results)
self.Plot("Indicators", "RSI", self.rsi.Current.Value)
self.Plot("Indicators", "SMA", self.sma.Current.Value)
self.Plot("Portfolio", "Value", self.Portfolio.TotalPortfolioValue)
```

### Runtime Statistics

```python
# Add custom statistics to results
self.SetRuntimeStatistic("Win Rate", f"{win_rate:.1%}")
self.SetRuntimeStatistic("Profit Factor", f"{profit_factor:.2f}")
self.SetRuntimeStatistic("Max Drawdown", f"{max_drawdown:.2%}")
```

### Debugging Tips

1. **Use `self.Log()` liberally** during development
2. **Check `IsWarmingUp`** before accessing indicators
3. **Verify data exists** with `data.ContainsKey()`
4. **Check `IsReady`** on all indicators
5. **Use smaller date ranges** for faster iteration
6. **Plot key metrics** for visual debugging

## Compute Resources

QuantConnect provides various compute nodes for backtesting, research, and live trading.

### Backtesting Nodes

| Node Type | RAM | CPU | Use Case | Free Tier |
|-----------|-----|-----|----------|-----------|
| **B-MICRO** | ~4 GB | Shared | Simple strategies | Yes (20s delay) |
| **B2-8** | 8 GB | 2 cores | Standard backtests | No |
| **B4-12** | 12 GB | 4 cores | Options/Futures | No |
| **B4-16** | 16 GB | 4 cores | Large universes | No |
| **B4-16-GPU** | 16 GB | 4 cores + GPU | ML strategies | No |

> **2025 Note**: Backtesting nodes range from 512MB RAM / 1 vCPU to 64GB RAM / 8 vCPU. Research nodes can scale up to 24 CPUs and 128GB RAM for intensive computations.

### Research Nodes

| Node Type | RAM | CPU | Use Case |
|-----------|-----|-----|----------|
| **R2-8** | 8 GB | 2 cores | Jupyter notebooks |
| **R4-12** | 12 GB | 4 cores | Data analysis |
| **R4-16-GPU** | 16 GB | 4 cores + GPU | ML research |

### Live Trading Nodes

| Node Type | RAM | CPU | Use Case |
|-----------|-----|-----|----------|
| **L-MICRO** | 512 MB | 1 CPU @ 2.4GHz | Simple live algos |
| **L-Small** | 1 GB | 1 CPU | Light trading |
| **L-Medium** | 2 GB | 2 CPU | Standard live trading |
| **L-Large** | 4 GB | 4 CPU | Options live trading |

> **2025 Note**: Live trading nodes range from 512MB to 4GB RAM. Each security subscription requires approximately 5MB of RAM. Nodes are deployed to low-latency, colocated racks in New York (NY7).

### Platform Constraints

| Constraint | Limit | Notes |
|------------|-------|-------|
| **Max Backtest Runtime** | 12 hours | Hard limit for all node types |
| **Single Event Loop** | 10 minutes | Individual OnData/scheduled event limit |
| **Free Tier Delay** | 20 seconds | Removed with paid subscription ($24/mo minimum) |
| **Results Upload** | 700 MB | Max backtest results size |
| **Log Limit (Free)** | 1 KB per backtest | Increased to 100 KB with Researcher tier |
| **Daily Log Limit** | 3 MB | Applies to all tiers |
| **Custom Data Files (Free)** | 25 files | Each up to 200 MB |
| **Concurrent Backtests** | By node count | 1 backtest per node |
| **Coding Sessions (Free)** | 1 global | One active session across all free orgs |

### Pricing Tiers (as of 2025)

| Tier | Monthly Cost | Live Nodes | Backtest Nodes | Notes |
|------|--------------|------------|----------------|-------|
| **Free** | $0 | 0 | 1 (B-MICRO, 20s delay) | Limited data |
| **Organization** | $20/mo | Varies | Varies | Most algorithmic traders |
| **Professional** | $40/mo | More | More | Advanced features |
| **Team** | $80/mo | 10 | 10 | Up to 10 users |
| **Trading Firm** | Custom | Unlimited | Unlimited | Enterprise support |

> **Note**: Pricing is customizable with seat types (Researcher, Team, Trading Firm, Institution). Visit [QuantConnect Pricing](https://www.quantconnect.com/pricing/) for current rates. Additional costs may apply for specialized datasets and live trading infrastructure.

### GPU Nodes

GPU nodes are available for machine learning workflows:

```python
# GPU-optimized algorithm for ML
class GPUAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)

        # GPU nodes support TensorFlow/PyTorch
        # Ensure your algorithm uses GPU-compatible libraries

        # Note: GPU resources are shared (max 1/3 GPU per user)
        # Best for training, not inference during live trading
```

**GPU Node Specifications (2025)**:

- GPU nodes are now out of beta and fully supported across research, backtesting, and live trading
- Approximately 1/3 GPU allocation per concurrent user (max 3 members sharing)
- RAM and CPU are dedicated (not shared)
- Monthly lease: ~$400 for GPU-enabled nodes (B4-16-GPU, R4-16-GPU)
- ML strategies can experience acceleration of 100x+ (e.g., 24-hour CPU task completing in 17 minutes on GPU)
- Best for repetitive, highly-parallel tasks like training ML models

### Selecting the Right Node

```python
# For options backtesting - use larger nodes
# lean backtest --node B4-12 my_options_algo.py

# For simple equity strategies - B-MICRO is sufficient
# lean backtest my_simple_algo.py

# For ML training - use GPU nodes
# lean backtest --node B4-16-GPU my_ml_algo.py
```

### Node Selection Guidelines

| Strategy Type | Recommended Node | Why |
|---------------|------------------|-----|
| Single equity, daily | B-MICRO | Low memory needs |
| Single equity, minute | B2-8 | Moderate data volume |
| Options (1 underlying) | B4-12 | High memory for chains |
| Options (multiple) | B4-16 | Very high memory |
| Large universe (500+) | B4-12+ | Universe data overhead |
| ML/Deep Learning | B4-16-GPU | GPU acceleration |

### Memory Management Tips

```python
class MemoryEfficientAlgorithm(QCAlgorithm):
    """Tips for staying within node memory limits."""

    def Initialize(self):
        # 1. Use restrictive filters for options
        option = self.AddOption("SPY")
        option.SetFilter(-3, 3, 30, 45)  # Tight filter

        # 2. Limit universe size
        self.UniverseSettings.Resolution = Resolution.Daily  # Not minute
        self.AddUniverse(self.CoarseFilter)

        # 3. Don't store historical data unnecessarily
        # BAD: self.history_cache = []
        # GOOD: Process and discard

    def CoarseFilter(self, coarse):
        # Limit to top 100 by volume
        return [x.Symbol for x in sorted(coarse,
                key=lambda x: x.DollarVolume, reverse=True)[:100]]

    def OnData(self, data):
        # 4. Avoid memory leaks in OnData
        # BAD: self.all_prices.append(data["SPY"].Close)
        # GOOD: Use rolling windows or indicators

        # 5. Clear large objects when done
        if hasattr(self, 'temp_data'):
            del self.temp_data
```

### Monitoring Resource Usage

```python
def OnEndOfDay(self, symbol):
    """Monitor resource usage (approximately)."""
    import sys

    # Log approximate memory usage
    # Note: This is Python-side only, not total LEAN memory
    self.Log(f"Approx Python objects: {len(gc.get_objects())}")

    # Track portfolio size
    positions = sum(1 for s in self.Portfolio.Keys
                   if self.Portfolio[s].Invested)
    self.Log(f"Active positions: {positions}")
```

### Cost Optimization

1. **Start with B-MICRO**: Test logic before using paid nodes
2. **Use daily resolution first**: Switch to minute only when needed
3. **Limit backtest date range**: 1 year instead of 10 for iteration
4. **Batch backtests**: Run multiple parameter sets efficiently
5. **Use local LEAN**: Free unlimited backtests on your hardware

```bash
# Run backtest locally (free, unlimited)
lean backtest my_algorithm.py

# Only use cloud nodes for final validation
lean cloud backtest --node B4-12 my_algorithm.py
```

---

*Last Updated: November 2025*
