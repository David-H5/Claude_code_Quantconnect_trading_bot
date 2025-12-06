# Algorithm Framework on QuantConnect

The Algorithm Framework provides a modular, plug-and-play architecture for building sophisticated trading strategies. This document covers Alpha Models, Portfolio Construction, Risk Management, and Execution Models.

## Table of Contents

- [Overview](#overview)
- [Framework Architecture](#framework-architecture)
- [Alpha Models](#alpha-models)
- [Portfolio Construction Models](#portfolio-construction-models)
- [Risk Management Models](#risk-management-models)
- [Execution Models](#execution-models)
- [Hybrid Algorithms](#hybrid-algorithms)
- [Multiple Alpha Models](#multiple-alpha-models)

## Overview

The Algorithm Framework separates trading logic into modular components:

1. **Universe Selection** → Which securities to trade
2. **Alpha Model** → Generate trading signals (Insights)
3. **Portfolio Construction** → Convert signals to position sizes
4. **Risk Management** → Apply risk limits and adjustments
5. **Execution Model** → Place orders efficiently

## Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Algorithm Framework Flow                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Universe Selection Model                                   │
│  ├── Selects tradeable securities                          │
│  └── Output: List of Symbols                               │
│           │                                                 │
│           ▼                                                 │
│  Alpha Model                                                │
│  ├── Generates trading signals                             │
│  └── Output: Insight[] (direction, magnitude, confidence)  │
│           │                                                 │
│           ▼                                                 │
│  Portfolio Construction Model                               │
│  ├── Converts insights to target positions                 │
│  └── Output: PortfolioTarget[] (symbol, quantity)          │
│           │                                                 │
│           ▼                                                 │
│  Risk Management Model                                      │
│  ├── Adjusts targets based on risk limits                  │
│  └── Output: Adjusted PortfolioTarget[]                    │
│           │                                                 │
│           ▼                                                 │
│  Execution Model                                            │
│  ├── Places orders to reach targets                        │
│  └── Output: Order submissions                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Alpha Models

Alpha Models predict market trends and generate Insight objects. An Insight contains:
- **Direction**: Up, Down, or Flat
- **Magnitude**: Expected price movement percentage
- **Confidence**: Probability of correct prediction (0-1)
- **Weight**: Suggested portfolio weight

### Built-in Alpha Models

> **2025 Verification**: The built-in alpha models remain current as of 2025. Available models include: `EmaCrossAlphaModel`, `RsiAlphaModel`, `MacdAlphaModel`, `ConstantAlphaModel`, and `BasePairsTradingAlphaModel`. No new alpha models were added in 2024-2025, but existing models continue to be maintained and improved.

```python
from AlgorithmImports import *

def Initialize(self):
    # EMA Cross Alpha - trades on EMA crossovers
    self.SetAlpha(EmaCrossAlphaModel(
        fast_period=12,
        slow_period=26,
        resolution=Resolution.Daily
    ))

    # RSI Alpha - trades on RSI signals
    self.SetAlpha(RsiAlphaModel(
        period=14,
        resolution=Resolution.Daily
    ))

    # MACD Alpha - trades on MACD crossovers
    self.SetAlpha(MacdAlphaModel(
        fast_period=12,
        slow_period=26,
        signal_period=9,
        moving_average_type=MovingAverageType.Exponential,
        resolution=Resolution.Daily
    ))

    # Constant Alpha - always bullish (for testing)
    self.SetAlpha(ConstantAlphaModel(
        InsightType.Price,
        InsightDirection.Up,
        timedelta(days=1)
    ))
```

### Custom Alpha Model

```python
class MyAlphaModel(AlphaModel):
    """Custom alpha model using RSI and SMA."""

    def __init__(self, rsi_period=14, sma_period=50):
        self.rsi_period = rsi_period
        self.sma_period = sma_period
        self.indicators = {}

    def Update(self, algorithm, data):
        """
        Generate insights based on current data.

        Parameters:
            algorithm: Algorithm instance
            data: Current Slice

        Returns:
            List of Insight objects
        """
        insights = []

        for symbol in self.indicators:
            rsi = self.indicators[symbol]['rsi']
            sma = self.indicators[symbol]['sma']

            if not rsi.IsReady or not sma.IsReady:
                continue

            price = algorithm.Securities[symbol].Price

            # Generate insight based on conditions
            if rsi.Current.Value < 30 and price > sma.Current.Value:
                # Oversold + above SMA = bullish
                insights.append(Insight.Price(
                    symbol,
                    timedelta(days=5),
                    InsightDirection.Up,
                    0.02,  # 2% expected move
                    0.75   # 75% confidence
                ))
            elif rsi.Current.Value > 70:
                # Overbought = bearish/exit
                insights.append(Insight.Price(
                    symbol,
                    timedelta(days=5),
                    InsightDirection.Down,
                    0.02,
                    0.60
                ))

        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        """Handle universe changes."""
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            self.indicators[symbol] = {
                'rsi': algorithm.RSI(symbol, self.rsi_period),
                'sma': algorithm.SMA(symbol, self.sma_period)
            }

        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.indicators:
                del self.indicators[symbol]

# Usage
def Initialize(self):
    self.SetAlpha(MyAlphaModel(rsi_period=14, sma_period=50))
```

### Insight Properties

```python
# Create an insight
insight = Insight.Price(
    symbol,                      # Symbol
    timedelta(days=5),           # Duration
    InsightDirection.Up,         # Direction: Up, Down, Flat
    0.02,                        # Magnitude (2%)
    0.75,                        # Confidence (75%)
    None,                        # Source model
    0.10                         # Weight (10% portfolio)
)

# Access insight properties
direction = insight.Direction
magnitude = insight.Magnitude
confidence = insight.Confidence
period = insight.Period
weight = insight.Weight
close_time = insight.CloseTimeUtc
```

## Portfolio Construction Models

Portfolio Construction Models convert Insights into PortfolioTargets, determining how many shares to hold.

### Built-in Portfolio Construction Models

```python
def Initialize(self):
    # Equal Weighting - equal weight to all insights
    self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())

    # Insight Weighting - weight by insight confidence
    self.SetPortfolioConstruction(InsightWeightingPortfolioConstructionModel())

    # Mean-Variance - optimize using mean-variance
    self.SetPortfolioConstruction(MeanVarianceOptimizationPortfolioConstructionModel())

    # Black-Litterman - combine views with market equilibrium
    self.SetPortfolioConstruction(BlackLittermanOptimizationPortfolioConstructionModel())

    # Risk Parity - equal risk contribution
    self.SetPortfolioConstruction(RiskParityPortfolioConstructionModel())

    # Null - pass through (for manual control)
    self.SetPortfolioConstruction(NullPortfolioConstructionModel())
```

### Custom Portfolio Construction Model

```python
class MyPortfolioConstructionModel(PortfolioConstructionModel):
    """Custom PCM with position limits."""

    def __init__(self, max_weight=0.10, rebalance_days=5):
        self.max_weight = max_weight
        self.rebalance_days = rebalance_days
        self.last_rebalance = datetime.min

    def CreateTargets(self, algorithm, insights):
        """
        Create portfolio targets from insights.

        Parameters:
            algorithm: Algorithm instance
            insights: List of Insight objects

        Returns:
            List of PortfolioTarget objects
        """
        # Only rebalance periodically
        if (algorithm.Time - self.last_rebalance).days < self.rebalance_days:
            return []

        self.last_rebalance = algorithm.Time
        targets = []

        # Filter active insights
        active_insights = [i for i in insights
                         if i.IsActive(algorithm.UtcTime)]

        # Group by symbol, take latest
        symbol_insights = {}
        for insight in active_insights:
            symbol_insights[insight.Symbol] = insight

        # Calculate weights
        if not symbol_insights:
            return targets

        equal_weight = min(self.max_weight, 1.0 / len(symbol_insights))

        for symbol, insight in symbol_insights.items():
            direction = 1 if insight.Direction == InsightDirection.Up else -1
            weight = direction * equal_weight * insight.Confidence

            targets.append(PortfolioTarget(symbol, weight))

        return targets

# Usage
def Initialize(self):
    self.SetPortfolioConstruction(MyPortfolioConstructionModel(
        max_weight=0.10,
        rebalance_days=7
    ))
```

### Rebalancing Control

```python
def Initialize(self):
    # Rebalance based on insight changes only
    self.SetPortfolioConstruction(
        EqualWeightingPortfolioConstructionModel(
            rebalance=Resolution.Daily
        )
    )

    # Custom rebalance function
    def RebalanceFunction(time):
        # Rebalance on first trading day of month
        return time.day == 1

    self.SetPortfolioConstruction(
        EqualWeightingPortfolioConstructionModel(
            rebalance=RebalanceFunction
        )
    )
```

## Risk Management Models

Risk Management Models adjust portfolio targets to ensure they're within acceptable risk parameters.

### Built-in Risk Management Models

```python
def Initialize(self):
    # Maximum Drawdown per Security
    self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.05))

    # Maximum Drawdown for Portfolio
    self.SetRiskManagement(MaximumDrawdownPercentPortfolio(0.10))

    # Maximum Sector Exposure
    self.SetRiskManagement(MaximumSectorExposureRiskManagementModel(0.30))

    # Trailing Stop
    self.SetRiskManagement(TrailingStopRiskManagementModel(0.05))

    # Null - no risk adjustment
    self.SetRiskManagement(NullRiskManagementModel())

    # Composite - combine multiple models
    self.SetRiskManagement(CompositeRiskManagementModel(
        MaximumDrawdownPercentPerSecurity(0.05),
        MaximumDrawdownPercentPortfolio(0.15)
    ))
```

### Custom Risk Management Model

```python
class MyRiskManagementModel(RiskManagementModel):
    """Custom risk management with volatility scaling."""

    def __init__(self, max_portfolio_std=0.02):
        self.max_portfolio_std = max_portfolio_std

    def ManageRisk(self, algorithm, targets):
        """
        Adjust targets based on risk.

        Parameters:
            algorithm: Algorithm instance
            targets: List of PortfolioTarget objects

        Returns:
            Adjusted list of PortfolioTarget objects
        """
        risk_adjusted_targets = []

        for target in targets:
            symbol = target.Symbol
            quantity = target.Quantity

            # Get historical volatility
            history = algorithm.History(symbol, 20, Resolution.Daily)
            if history.empty:
                risk_adjusted_targets.append(target)
                continue

            returns = history['close'].pct_change().dropna()
            volatility = returns.std()

            # Scale position by inverse volatility
            if volatility > self.max_portfolio_std:
                scale_factor = self.max_portfolio_std / volatility
                adjusted_quantity = int(quantity * scale_factor)
                risk_adjusted_targets.append(
                    PortfolioTarget(symbol, adjusted_quantity)
                )
            else:
                risk_adjusted_targets.append(target)

        return risk_adjusted_targets

# Usage
def Initialize(self):
    self.SetRiskManagement(MyRiskManagementModel(max_portfolio_std=0.02))
```

## Execution Models

Execution Models place orders to reach the portfolio targets efficiently.

### Built-in Execution Models

```python
def Initialize(self):
    # Immediate Execution - market orders immediately
    self.SetExecution(ImmediateExecutionModel())

    # VWAP Execution - split orders over time
    self.SetExecution(VolumeWeightedAveragePriceExecutionModel())

    # Standard Deviation Execution - execute when volatility is low
    self.SetExecution(StandardDeviationExecutionModel(
        period=60,
        deviations=2,
        resolution=Resolution.Minute
    ))

    # Null - no execution (for manual orders)
    self.SetExecution(NullExecutionModel())
```

### Custom Execution Model

```python
class MyExecutionModel(ExecutionModel):
    """Custom execution with limit orders."""

    def __init__(self, limit_offset_pct=0.001):
        self.limit_offset_pct = limit_offset_pct
        self.pending_orders = {}

    def Execute(self, algorithm, targets):
        """
        Execute orders to reach targets.

        Parameters:
            algorithm: Algorithm instance
            targets: List of PortfolioTarget objects
        """
        for target in targets:
            symbol = target.Symbol
            quantity = target.Quantity

            # Calculate current position
            current = algorithm.Portfolio[symbol].Quantity
            order_quantity = quantity - current

            if order_quantity == 0:
                continue

            # Cancel existing pending order
            if symbol in self.pending_orders:
                algorithm.Transactions.CancelOrder(self.pending_orders[symbol])

            # Calculate limit price
            price = algorithm.Securities[symbol].Price
            if order_quantity > 0:
                # Buying - place below market
                limit_price = price * (1 - self.limit_offset_pct)
            else:
                # Selling - place above market
                limit_price = price * (1 + self.limit_offset_pct)

            # Place limit order
            ticket = algorithm.LimitOrder(symbol, order_quantity, limit_price)
            self.pending_orders[symbol] = ticket.OrderId

    def OnOrderEvent(self, algorithm, orderEvent):
        """Handle order events."""
        if orderEvent.Status in [OrderStatus.Filled, OrderStatus.Canceled]:
            symbol = orderEvent.Symbol
            if symbol in self.pending_orders:
                del self.pending_orders[symbol]

# Usage
def Initialize(self):
    self.SetExecution(MyExecutionModel(limit_offset_pct=0.001))
```

## Hybrid Algorithms

Hybrid algorithms combine framework components with classic algorithm methods.

```python
class HybridAlgorithm(QCAlgorithm):
    """Hybrid algorithm using framework + classic methods."""

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        # Framework components
        self.SetUniverseSelection(QC500UniverseSelectionModel())
        self.SetAlpha(EmaCrossAlphaModel())
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())

        # Classic scheduling
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.BeforeMarketClose("SPY", 30),
            self.DailyReport
        )

    def OnData(self, data):
        """Classic OnData for additional logic."""
        # Can add custom logic here
        if self.Portfolio.TotalUnrealizedProfit > 10000:
            self.Log("Significant unrealized profit!")

    def DailyReport(self):
        """Scheduled daily report."""
        self.Log(f"EOD Portfolio: ${self.Portfolio.TotalPortfolioValue:,.2f}")
```

## Multiple Alpha Models

You can combine multiple Alpha Models to generate signals from different strategies.

```python
def Initialize(self):
    self.SetStartDate(2024, 1, 1)
    self.SetCash(100000)

    # Add universe
    self.SetUniverseSelection(QC500UniverseSelectionModel())

    # Add multiple alphas - insights are combined
    self.AddAlpha(EmaCrossAlphaModel(12, 26, Resolution.Daily))
    self.AddAlpha(RsiAlphaModel(14, Resolution.Daily))
    self.AddAlpha(MacdAlphaModel(12, 26, 9))

    # Insight weighting considers all alpha sources
    self.SetPortfolioConstruction(InsightWeightingPortfolioConstructionModel())

    self.SetExecution(ImmediateExecutionModel())
```

### Handling Combined Insights

```python
class MultiAlphaPortfolioConstruction(PortfolioConstructionModel):
    """PCM that combines insights from multiple alphas."""

    def CreateTargets(self, algorithm, insights):
        # Group insights by symbol
        symbol_insights = {}
        for insight in insights:
            if insight.Symbol not in symbol_insights:
                symbol_insights[insight.Symbol] = []
            symbol_insights[insight.Symbol].append(insight)

        targets = []
        for symbol, symbol_insight_list in symbol_insights.items():
            # Average the directions weighted by confidence
            weighted_direction = sum(
                (1 if i.Direction == InsightDirection.Up else -1) * i.Confidence
                for i in symbol_insight_list
            ) / len(symbol_insight_list)

            # Only trade if majority agree
            if abs(weighted_direction) > 0.5:
                direction = 1 if weighted_direction > 0 else -1
                weight = direction * 0.10 * abs(weighted_direction)
                targets.append(PortfolioTarget(symbol, weight))

        return targets
```

---

**Sources:**
- [Algorithm Framework Overview](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/overview)
- [Alpha Model Key Concepts](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/alpha/key-concepts)
- [Portfolio Construction](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/portfolio-construction/key-concepts)
- [Hybrid Algorithms](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/hybrid-algorithms)

*Last Updated: November 2025*
