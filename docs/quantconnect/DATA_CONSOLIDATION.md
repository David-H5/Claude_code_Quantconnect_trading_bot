# Data Consolidation and Scheduling on QuantConnect

Guide to consolidating data into custom time periods, scheduling events, and warming up indicators.

## Table of Contents

- [Data Consolidation](#data-consolidation)
- [Scheduled Events](#scheduled-events)
- [Indicator Warm-Up](#indicator-warm-up)
- [Logging and Debugging](#logging-and-debugging)
- [Runtime Statistics](#runtime-statistics)

---

## Data Consolidation

> **2025 Verification**: The consolidator API remains stable. Available consolidator types include: `TradeBarConsolidator`, `QuoteBarConsolidator`, `TickConsolidator`, `RenkoConsolidator`, `VolumeRenkoConsolidator`, and `ClassicRenkoConsolidator`. Calendar consolidators (`Calendar.Weekly`, `Calendar.Monthly`) continue to work as documented.

Consolidators aggregate data from one resolution to another (e.g., minute bars to hourly bars).

### Basic Consolidation with Helper Method

```python
from AlgorithmImports import *

class ConsolidationAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        # Subscribe to minute data
        self.symbol = self.AddEquity("SPY", Resolution.Minute).Symbol

        # Create 15-minute consolidator using helper
        self.Consolidate(self.symbol, timedelta(minutes=15), self.OnFifteenMinuteBar)

        # Create hourly consolidator
        self.Consolidate(self.symbol, Resolution.Hour, self.OnHourlyBar)

        # Create daily consolidator
        self.Consolidate(self.symbol, Resolution.Daily, self.OnDailyBar)

    def OnFifteenMinuteBar(self, bar):
        """Called when 15-minute bar completes."""
        self.Log(f"15-min bar: O={bar.Open} H={bar.High} L={bar.Low} C={bar.Close}")

    def OnHourlyBar(self, bar):
        """Called when hourly bar completes."""
        self.Log(f"Hourly bar: {bar.Close}")

    def OnDailyBar(self, bar):
        """Called when daily bar completes."""
        self.Log(f"Daily bar: {bar.Close}")

    def OnData(self, data):
        # Minute data still flows here
        pass
```

### Manual Consolidator Setup

```python
def Initialize(self):
    self.symbol = self.AddEquity("SPY", Resolution.Minute).Symbol

    # Create trade bar consolidator
    self.fifteen_min_consolidator = TradeBarConsolidator(timedelta(minutes=15))
    self.fifteen_min_consolidator.DataConsolidated += self.OnConsolidatedBar

    # Register consolidator with subscription manager
    self.SubscriptionManager.AddConsolidator(self.symbol, self.fifteen_min_consolidator)

def OnConsolidatedBar(self, sender, bar):
    """Handler for consolidated bars."""
    self.Log(f"Consolidated: {bar.Time} - Close: {bar.Close}")
```

### Quote Bar Consolidation

```python
def Initialize(self):
    # For bid/ask data
    self.symbol = self.AddEquity("SPY", Resolution.Second).Symbol

    # Quote bar consolidator
    quote_consolidator = QuoteBarConsolidator(timedelta(minutes=5))
    quote_consolidator.DataConsolidated += self.OnQuoteBar
    self.SubscriptionManager.AddConsolidator(self.symbol, quote_consolidator)

def OnQuoteBar(self, sender, bar):
    """Handler for consolidated quote bars."""
    self.Log(f"Bid: {bar.Bid.Close}, Ask: {bar.Ask.Close}")
```

### Using Consolidated Data with Indicators

```python
def Initialize(self):
    self.symbol = self.AddEquity("SPY", Resolution.Minute).Symbol

    # Create indicators
    self.rsi_15min = RelativeStrengthIndex(14)
    self.sma_hourly = SimpleMovingAverage(20)

    # 15-minute consolidator for RSI
    fifteen_min = TradeBarConsolidator(timedelta(minutes=15))
    fifteen_min.DataConsolidated += self.OnFifteenMinute
    self.SubscriptionManager.AddConsolidator(self.symbol, fifteen_min)

    # Register RSI with consolidator
    self.RegisterIndicator(self.symbol, self.rsi_15min, fifteen_min)

    # Hourly consolidator for SMA
    hourly = TradeBarConsolidator(Resolution.Hour)
    hourly.DataConsolidated += self.OnHourly
    self.SubscriptionManager.AddConsolidator(self.symbol, hourly)

    # Register SMA with consolidator
    self.RegisterIndicator(self.symbol, self.sma_hourly, hourly)

def OnFifteenMinute(self, sender, bar):
    if self.rsi_15min.IsReady:
        self.Log(f"15-min RSI: {self.rsi_15min.Current.Value:.2f}")

def OnHourly(self, sender, bar):
    if self.sma_hourly.IsReady:
        self.Log(f"Hourly SMA: {self.sma_hourly.Current.Value:.2f}")
```

### Calendar Consolidators

For custom time periods based on calendar rules:

```python
def Initialize(self):
    self.symbol = self.AddEquity("SPY", Resolution.Minute).Symbol

    # Weekly bars
    weekly = TradeBarConsolidator(Calendar.Weekly)
    weekly.DataConsolidated += self.OnWeeklyBar
    self.SubscriptionManager.AddConsolidator(self.symbol, weekly)

    # Monthly bars
    monthly = TradeBarConsolidator(Calendar.Monthly)
    monthly.DataConsolidated += self.OnMonthlyBar
    self.SubscriptionManager.AddConsolidator(self.symbol, monthly)

def OnWeeklyBar(self, sender, bar):
    self.Log(f"Weekly: {bar.Time.strftime('%Y-%m-%d')} Close: {bar.Close}")

def OnMonthlyBar(self, sender, bar):
    self.Log(f"Monthly: {bar.Time.strftime('%Y-%m')} Close: {bar.Close}")
```

### Mixed-Mode Consolidators

Consolidate based on both time AND count:

```python
def Initialize(self):
    self.symbol = self.AddEquity("SPY", Resolution.Minute).Symbol

    # Bar closes at 15 minutes OR 100 trades, whichever first
    mixed = TradeBarConsolidator(timedelta(minutes=15))
    # Additional count-based logic would go in handler
```

---

## Scheduled Events

Schedule code to run at specific times, regardless of data events.

### Date Rules

```python
def Initialize(self):
    self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

    # Every day
    self.Schedule.On(
        self.DateRules.EveryDay(self.symbol),
        self.TimeRules.AfterMarketOpen(self.symbol, 30),
        self.DailyTask
    )

    # Every Monday
    self.Schedule.On(
        self.DateRules.Every(DayOfWeek.Monday),
        self.TimeRules.At(10, 0),
        self.WeeklyTask
    )

    # First day of month
    self.Schedule.On(
        self.DateRules.MonthStart(self.symbol),
        self.TimeRules.AfterMarketOpen(self.symbol, 60),
        self.MonthlyRebalance
    )

    # Last day of month
    self.Schedule.On(
        self.DateRules.MonthEnd(self.symbol),
        self.TimeRules.BeforeMarketClose(self.symbol, 30),
        self.MonthEndTask
    )

    # Specific date
    self.Schedule.On(
        self.DateRules.On(2024, 12, 31),
        self.TimeRules.At(12, 0),
        self.YearEndTask
    )

    # First and last day of week
    self.Schedule.On(
        self.DateRules.WeekStart(),
        self.TimeRules.AfterMarketOpen(self.symbol, 30),
        self.WeekStartTask
    )

    self.Schedule.On(
        self.DateRules.WeekEnd(),
        self.TimeRules.BeforeMarketClose(self.symbol, 30),
        self.WeekEndTask
    )
```

### Time Rules

```python
def Initialize(self):
    self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

    # Minutes after market open
    self.Schedule.On(
        self.DateRules.EveryDay(self.symbol),
        self.TimeRules.AfterMarketOpen(self.symbol, 30),  # 30 minutes after open
        self.MarketOpenTask
    )

    # Minutes before market close
    self.Schedule.On(
        self.DateRules.EveryDay(self.symbol),
        self.TimeRules.BeforeMarketClose(self.symbol, 15),  # 15 minutes before close
        self.MarketCloseTask
    )

    # Specific time
    self.Schedule.On(
        self.DateRules.EveryDay(),
        self.TimeRules.At(14, 30),  # 2:30 PM
        self.AfternoonTask
    )

    # Every N minutes/hours
    self.Schedule.On(
        self.DateRules.EveryDay(self.symbol),
        self.TimeRules.Every(timedelta(hours=1)),
        self.HourlyTask
    )

    # At midnight
    self.Schedule.On(
        self.DateRules.EveryDay(),
        self.TimeRules.Midnight,
        self.MidnightTask
    )

    # At noon
    self.Schedule.On(
        self.DateRules.EveryDay(),
        self.TimeRules.Noon,
        self.NoonTask
    )
```

### Custom Date and Time Rules

```python
def Initialize(self):
    # Custom date rule: 10th of each month
    def TenthOfMonth(dates):
        return [d for d in dates if d.day == 10]

    custom_date_rule = self.DateRules.On(FuncDateRule("TenthOfMonth", TenthOfMonth))

    # Custom time rule
    def MarketMiddle(dates):
        # Return midday for each date
        return [d.replace(hour=12, minute=30) for d in dates]

    custom_time_rule = FuncTimeRule("MarketMiddle", MarketMiddle)

    self.Schedule.On(custom_date_rule, custom_time_rule, self.CustomTask)
```

### Scheduled Rebalancing

```python
class RebalancingAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        self.symbols = [
            self.AddEquity("SPY", Resolution.Daily).Symbol,
            self.AddEquity("TLT", Resolution.Daily).Symbol,
            self.AddEquity("GLD", Resolution.Daily).Symbol,
        ]

        # Monthly rebalancing
        self.Schedule.On(
            self.DateRules.MonthStart("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.Rebalance
        )

    def Rebalance(self):
        """Equal weight rebalancing."""
        target_weight = 1.0 / len(self.symbols)

        for symbol in self.symbols:
            self.SetHoldings(symbol, target_weight)

        self.Log(f"Rebalanced at {self.Time}")
```

---

## Indicator Warm-Up

Warm-up ensures indicators have enough historical data before trading begins.

### SetWarmUp Method

```python
def Initialize(self):
    self.SetStartDate(2024, 1, 1)
    self.SetCash(100000)

    self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

    # Create indicators
    self.sma = self.SMA(self.symbol, 50, Resolution.Daily)
    self.rsi = self.RSI(self.symbol, 14, Resolution.Daily)

    # Warm up for 50 bars (covers longest indicator)
    self.SetWarmUp(50, Resolution.Daily)

    # Or by time period
    # self.SetWarmUp(timedelta(days=60))

def OnData(self, data):
    # Check if still warming up
    if self.IsWarmingUp:
        return

    # Now indicators are ready
    if self.sma.IsReady and self.rsi.IsReady:
        # Trading logic
        pass
```

### Automatic Indicator Warm-Up

```python
def Initialize(self):
    # Enable automatic warm-up for indicators
    self.Settings.AutomaticIndicatorWarmUp = True

    self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

    # Indicators will automatically warm up
    self.sma = self.SMA(self.symbol, 50, Resolution.Daily)
    self.rsi = self.RSI(self.symbol, 14, Resolution.Daily)
```

### Manual Warm-Up with History

```python
def Initialize(self):
    self.SetStartDate(2024, 1, 1)
    self.SetCash(100000)

    self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

    # Create indicators manually
    self.sma = SimpleMovingAverage(50)
    self.rsi = RelativeStrengthIndex(14)

    # Warm up immediately with history
    history = self.History(self.symbol, 60, Resolution.Daily)

    for bar in history.itertuples():
        trade_bar = TradeBar()
        trade_bar.Close = bar.close
        trade_bar.Time = bar.Index[1]

        self.sma.Update(bar.Index[1], bar.close)
        self.rsi.Update(bar.Index[1], bar.close)

    self.Log(f"SMA Ready: {self.sma.IsReady}, RSI Ready: {self.rsi.IsReady}")
```

### WarmUpIndicator Method

```python
def Initialize(self):
    self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

    # Create indicator
    self.sma = self.SMA(self.symbol, 50, Resolution.Daily)

    # Warm up specific indicator
    self.WarmUpIndicator(self.symbol, self.sma, Resolution.Daily)
```

### Universe Warm-Up

```python
def OnSecuritiesChanged(self, changes):
    """Warm up indicators for new securities."""
    for security in changes.AddedSecurities:
        symbol = security.Symbol

        # Create indicator
        self.indicators[symbol] = self.RSI(symbol, 14, Resolution.Daily)

        # Warm up with history
        history = self.History(symbol, 20, Resolution.Daily)

        for bar in history.itertuples():
            self.indicators[symbol].Update(bar.Index[1], bar.close)
```

---

## Logging and Debugging

### Logging Methods

```python
def OnData(self, data):
    # Standard logging (saved to log file)
    self.Log("This is a log message")
    self.Log(f"Price: {data[self.symbol].Close}")

    # Debug logging (shown in console)
    self.Debug("Debug message")

    # Error logging
    self.Error("Something went wrong!")

    # Conditional logging
    if self.Portfolio[self.symbol].UnrealizedProfitPercent > 0.10:
        self.Log(f"10% profit reached!")
```

### Plotting Charts

```python
def OnData(self, data):
    if not data.ContainsKey(self.symbol):
        return

    price = data[self.symbol].Close

    # Plot to named chart and series
    self.Plot("Price", "Close", price)
    self.Plot("Price", "SMA", self.sma.Current.Value)

    # Plot indicators
    self.Plot("Indicators", "RSI", self.rsi.Current.Value)
    self.Plot("Indicators", "Overbought", 70)
    self.Plot("Indicators", "Oversold", 30)

    # Plot portfolio
    self.Plot("Portfolio", "Value", self.Portfolio.TotalPortfolioValue)
    self.Plot("Portfolio", "Cash", self.Portfolio.Cash)
```

### Custom Chart Setup

```python
def Initialize(self):
    # Create custom chart
    price_chart = Chart("Custom Price")
    price_chart.AddSeries(Series("Close", SeriesType.Line, "$"))
    price_chart.AddSeries(Series("SMA", SeriesType.Line, "$"))
    self.AddChart(price_chart)

    # Create indicator chart
    indicator_chart = Chart("Signals")
    indicator_chart.AddSeries(Series("RSI", SeriesType.Line, "%"))
    indicator_chart.AddSeries(Series("Signal", SeriesType.Scatter, ""))
    self.AddChart(indicator_chart)
```

### Debugging Techniques

```python
def OnData(self, data):
    # Log algorithm state
    self.Debug(f"Time: {self.Time}")
    self.Debug(f"Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
    self.Debug(f"Cash: ${self.Portfolio.Cash:,.2f}")

    # Log indicator states
    for symbol, indicator in self.indicators.items():
        self.Debug(f"{symbol}: Ready={indicator.IsReady}, Value={indicator.Current.Value:.2f}")

    # Log order events
    for order in self.Transactions.GetOpenOrders():
        self.Debug(f"Open Order: {order.Symbol} - {order.Quantity} @ {order.LimitPrice}")
```

---

## Runtime Statistics

Display custom statistics in the backtest interface.

### Setting Runtime Statistics

```python
def OnEndOfDay(self, symbol):
    # Update runtime statistics
    self.SetRuntimeStatistic("Portfolio Value",
                            f"${self.Portfolio.TotalPortfolioValue:,.2f}")
    self.SetRuntimeStatistic("Daily P&L",
                            f"${self.Portfolio.TotalProfit:,.2f}")
    self.SetRuntimeStatistic("Open Positions",
                            str(sum(1 for s in self.Portfolio.Values if s.Invested)))

def OnOrderEvent(self, orderEvent):
    if orderEvent.Status == OrderStatus.Filled:
        # Track trade statistics
        self.trade_count += 1
        self.SetRuntimeStatistic("Total Trades", str(self.trade_count))

        if orderEvent.FillPrice > 0:
            self.SetRuntimeStatistic("Last Fill Price",
                                    f"${orderEvent.FillPrice:.2f}")
```

### Custom Performance Metrics

```python
class MetricsAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.trade_results = []
        self.peak_value = 0
        self.current_drawdown = 0

    def OnOrderEvent(self, orderEvent):
        if orderEvent.Status == OrderStatus.Filled:
            # Track trade P&L
            pnl = orderEvent.FillPrice * orderEvent.FillQuantity
            self.trade_results.append(pnl)

            # Update win rate
            winners = sum(1 for t in self.trade_results if t > 0)
            win_rate = winners / len(self.trade_results) if self.trade_results else 0
            self.SetRuntimeStatistic("Win Rate", f"{win_rate:.1%}")

    def OnEndOfDay(self, symbol):
        # Track drawdown
        current_value = self.Portfolio.TotalPortfolioValue
        self.peak_value = max(self.peak_value, current_value)
        self.current_drawdown = (self.peak_value - current_value) / self.peak_value

        self.SetRuntimeStatistic("Max Drawdown", f"{self.current_drawdown:.1%}")

        # Sharpe approximation
        if len(self.trade_results) > 10:
            returns = np.array(self.trade_results)
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            self.SetRuntimeStatistic("Sharpe (approx)", f"{sharpe:.2f}")
```

---

**Sources:**
- [Consolidating Data](https://www.quantconnect.com/docs/algorithm-reference/consolidating-data)
- [Time Period Consolidators](https://www.quantconnect.com/docs/v2/writing-algorithms/consolidating-data/consolidator-types/time-period-consolidators)
- [Calendar Consolidators](https://www.quantconnect.com/docs/v2/writing-algorithms/consolidating-data/consolidator-types/calendar-consolidators)
- [Scheduled Events](https://www.quantconnect.com/docs/v2/writing-algorithms/scheduled-events)
- [Warm Up Periods](https://www.quantconnect.com/docs/v2/writing-algorithms/historical-data/warm-up-periods)
- [Logging](https://www.quantconnect.com/docs/v2/writing-algorithms/logging)
- [Runtime Statistics](https://www.quantconnect.com/docs/v2/writing-algorithms/statistics/runtime-statistics)
- [Debugging Tools](https://www.quantconnect.com/docs/v2/writing-algorithms/key-concepts/debugging-tools)

*Last Updated: November 2025*
