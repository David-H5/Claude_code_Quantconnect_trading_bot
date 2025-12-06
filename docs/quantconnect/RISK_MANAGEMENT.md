# Risk Management on QuantConnect

Comprehensive guide to implementing risk management in QuantConnect algorithms, including position sizing, portfolio protection, drawdown limits, and built-in risk models.

## Table of Contents

- [Overview](#overview)
- [Position Sizing](#position-sizing)
- [Built-in Risk Models](#built-in-risk-models)
- [Custom Risk Management](#custom-risk-management)
- [Drawdown Protection](#drawdown-protection)
- [Stop Loss and Take Profit](#stop-loss-and-take-profit)
- [Portfolio Risk Metrics](#portfolio-risk-metrics)
- [Sector and Correlation Risk](#sector-and-correlation-risk)
- [Shorting and Margin](#shorting-and-margin)
- [Best Practices](#best-practices)

## Overview

### Why Risk Management Matters

Risk management is crucial for:
- Preserving capital during drawdowns
- Preventing catastrophic losses
- Ensuring consistent returns
- Maintaining emotional discipline

### Risk Management Framework

```
┌─────────────────────────────────────────────────────────────┐
│                Risk Management Framework                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Pre-Trade                                                  │
│  ├── Position Sizing      → How much to buy                │
│  ├── Entry Criteria       → When to enter                  │
│  └── Portfolio Limits     → Max positions, sectors         │
│                                                             │
│  During Trade                                               │
│  ├── Stop Loss            → Exit on loss                   │
│  ├── Take Profit          → Exit on gain                   │
│  └── Trailing Stop        → Lock in profits                │
│                                                             │
│  Portfolio Level                                            │
│  ├── Drawdown Limit       → Max portfolio decline          │
│  ├── Daily Loss Limit     → Max daily loss                 │
│  └── Correlation Risk     → Diversification                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Position Sizing

### Fixed Fractional Position Sizing

```python
def CalculatePositionSize(self, symbol, risk_per_trade=0.02):
    """
    Calculate position size based on fixed percentage risk.

    Parameters:
        symbol: Security symbol
        risk_per_trade: Max portfolio % to risk (default 2%)

    Returns:
        Number of shares to buy
    """
    portfolio_value = self.Portfolio.TotalPortfolioValue
    risk_amount = portfolio_value * risk_per_trade

    price = self.Securities[symbol].Price

    # Simple: position size = risk amount / price
    shares = int(risk_amount / price)

    return shares
```

### Risk-Based Position Sizing with Stop Loss

```python
def CalculatePositionSizeWithStop(self, symbol, stop_loss_pct=0.05, risk_per_trade=0.02):
    """
    Calculate position size using stop loss to define risk.

    Parameters:
        symbol: Security symbol
        stop_loss_pct: Stop loss percentage (e.g., 0.05 = 5%)
        risk_per_trade: Max portfolio % to risk

    Returns:
        Number of shares to buy
    """
    portfolio_value = self.Portfolio.TotalPortfolioValue
    risk_amount = portfolio_value * risk_per_trade

    price = self.Securities[symbol].Price
    stop_price = price * (1 - stop_loss_pct)
    risk_per_share = price - stop_price

    # Position size = risk amount / risk per share
    shares = int(risk_amount / risk_per_share)

    return shares
```

### ATR-Based Position Sizing

```python
def Initialize(self):
    self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
    self.atr = self.ATR(self.symbol, 14, MovingAverageType.Simple, Resolution.Daily)
    self.SetWarmUp(14)

def CalculatePositionSizeATR(self, symbol, atr_multiplier=2.0, risk_per_trade=0.02):
    """
    Calculate position size using ATR for volatility-adjusted sizing.

    Parameters:
        symbol: Security symbol
        atr_multiplier: Multiple of ATR for stop distance
        risk_per_trade: Max portfolio % to risk

    Returns:
        Number of shares to buy
    """
    if not self.atr.IsReady:
        return 0

    portfolio_value = self.Portfolio.TotalPortfolioValue
    risk_amount = portfolio_value * risk_per_trade

    price = self.Securities[symbol].Price
    atr_value = self.atr.Current.Value
    risk_per_share = atr_value * atr_multiplier

    shares = int(risk_amount / risk_per_share)

    return shares
```

### Kelly Criterion Position Sizing

```python
def CalculateKellySize(self, win_rate, avg_win, avg_loss, kelly_fraction=0.5):
    """
    Calculate position size using Kelly Criterion.

    Parameters:
        win_rate: Historical win rate (0-1)
        avg_win: Average winning trade %
        avg_loss: Average losing trade % (positive number)
        kelly_fraction: Fraction of full Kelly to use (0.5 = half Kelly)

    Returns:
        Portfolio weight (0-1)
    """
    if avg_loss == 0:
        return 0

    # Kelly formula: f = (bp - q) / b
    # Where b = odds ratio (avg_win / avg_loss)
    # p = probability of win
    # q = probability of loss (1 - p)

    b = avg_win / avg_loss
    p = win_rate
    q = 1 - win_rate

    kelly = (b * p - q) / b

    # Apply fraction and bound
    position_size = max(0, min(kelly * kelly_fraction, 0.25))  # Cap at 25%

    return position_size
```

## Built-in Risk Models

### Maximum Drawdown Per Security

```python
def Initialize(self):
    self.SetStartDate(2020, 1, 1)
    self.SetCash(100000)

    self.AddEquity("SPY", Resolution.Daily)

    # Liquidate if any position drops 5%
    self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.05))
```

### Maximum Drawdown Per Portfolio

```python
def Initialize(self):
    # Liquidate all if portfolio drops 10%
    self.SetRiskManagement(MaximumDrawdownPercentPortfolio(0.10))
```

### Maximum Sector Exposure

```python
def Initialize(self):
    # Limit exposure to any sector to 30%
    self.SetRiskManagement(MaximumSectorExposureRiskManagementModel(0.30))
```

### Trailing Stop

```python
def Initialize(self):
    # 5% trailing stop on all positions
    self.SetRiskManagement(TrailingStopRiskManagementModel(0.05))
```

### Combining Risk Models

```python
from QuantConnect.Algorithm.Framework.Risk import *

def Initialize(self):
    # Use composite risk model
    self.SetRiskManagement(CompositeRiskManagementModel(
        MaximumDrawdownPercentPerSecurity(0.05),
        MaximumDrawdownPercentPortfolio(0.15),
        TrailingStopRiskManagementModel(0.03)
    ))
```

## Custom Risk Management

### Custom Risk Management Model

```python
from QuantConnect.Algorithm.Framework.Risk import RiskManagementModel

class CustomRiskManagement(RiskManagementModel):
    """Custom risk management with multiple rules."""

    def __init__(self,
                 max_drawdown_per_security=0.05,
                 max_portfolio_drawdown=0.15,
                 max_daily_loss=0.03):

        self.max_drawdown_per_security = max_drawdown_per_security
        self.max_portfolio_drawdown = max_portfolio_drawdown
        self.max_daily_loss = max_daily_loss

        self.peak_portfolio_value = 0
        self.daily_start_value = 0
        self.last_date = None

    def ManageRisk(self, algorithm, targets):
        """
        Evaluate portfolio and return risk-adjusted targets.

        Parameters:
            algorithm: Algorithm instance
            targets: Current portfolio targets

        Returns:
            List of risk-adjusted targets
        """
        current_value = algorithm.Portfolio.TotalPortfolioValue

        # Track peak value for drawdown
        self.peak_portfolio_value = max(self.peak_portfolio_value, current_value)

        # Track daily starting value
        current_date = algorithm.Time.date()
        if self.last_date != current_date:
            self.daily_start_value = current_value
            self.last_date = current_date

        risk_targets = []

        # Check portfolio drawdown
        portfolio_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        if portfolio_drawdown > self.max_portfolio_drawdown:
            algorithm.Log(f"RISK: Portfolio drawdown {portfolio_drawdown:.2%} exceeds limit")
            # Liquidate everything
            for symbol in algorithm.Portfolio.Keys:
                if algorithm.Portfolio[symbol].Invested:
                    risk_targets.append(PortfolioTarget(symbol, 0))
            return risk_targets

        # Check daily loss
        daily_loss = (self.daily_start_value - current_value) / self.daily_start_value
        if daily_loss > self.max_daily_loss:
            algorithm.Log(f"RISK: Daily loss {daily_loss:.2%} exceeds limit")
            # Liquidate everything
            for symbol in algorithm.Portfolio.Keys:
                if algorithm.Portfolio[symbol].Invested:
                    risk_targets.append(PortfolioTarget(symbol, 0))
            return risk_targets

        # Check individual position drawdowns
        for symbol in algorithm.Portfolio.Keys:
            holding = algorithm.Portfolio[symbol]
            if not holding.Invested:
                continue

            # Calculate position drawdown
            if holding.AveragePrice > 0:
                position_return = (holding.Price - holding.AveragePrice) / holding.AveragePrice
                if position_return < -self.max_drawdown_per_security:
                    algorithm.Log(f"RISK: {symbol} drawdown {position_return:.2%}")
                    risk_targets.append(PortfolioTarget(symbol, 0))

        return risk_targets

# Usage
def Initialize(self):
    self.SetRiskManagement(CustomRiskManagement(
        max_drawdown_per_security=0.05,
        max_portfolio_drawdown=0.15,
        max_daily_loss=0.03
    ))
```

## Drawdown Protection

### Drawdown Tracker

```python
class DrawdownTracker:
    """Track and manage drawdown metrics."""

    def __init__(self, max_drawdown=0.20):
        self.max_drawdown = max_drawdown
        self.peak_value = 0
        self.current_drawdown = 0
        self.max_observed_drawdown = 0
        self.in_drawdown = False
        self.drawdown_start_date = None

    def update(self, current_value, current_time):
        """Update drawdown metrics with new value."""

        if current_value > self.peak_value:
            self.peak_value = current_value
            self.in_drawdown = False
            self.drawdown_start_date = None
        else:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
            self.max_observed_drawdown = max(self.max_observed_drawdown, self.current_drawdown)

            if not self.in_drawdown and self.current_drawdown > 0.01:
                self.in_drawdown = True
                self.drawdown_start_date = current_time

        return self.current_drawdown

    def is_exceeded(self):
        """Check if max drawdown exceeded."""
        return self.current_drawdown > self.max_drawdown

    def get_recovery_needed(self):
        """Calculate % gain needed to recover from drawdown."""
        if self.current_drawdown == 0:
            return 0
        return self.current_drawdown / (1 - self.current_drawdown)

# Usage in algorithm
class MyAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.drawdown_tracker = DrawdownTracker(max_drawdown=0.20)
        self.trading_halted = False

    def OnData(self, data):
        # Update drawdown tracker
        current_value = self.Portfolio.TotalPortfolioValue
        self.drawdown_tracker.update(current_value, self.Time)

        # Check if trading should halt
        if self.drawdown_tracker.is_exceeded():
            if not self.trading_halted:
                self.Log(f"HALT: Max drawdown exceeded ({self.drawdown_tracker.current_drawdown:.2%})")
                self.Liquidate()
                self.trading_halted = True
            return

        # Resume trading if recovered
        if self.trading_halted and self.drawdown_tracker.current_drawdown < 0.10:
            self.Log("RESUME: Recovered from drawdown")
            self.trading_halted = False

        # Normal trading logic
        if not self.trading_halted:
            # ... trading logic
            pass
```

### Progressive Position Reduction

```python
def AdjustPositionForDrawdown(self, target_weight):
    """
    Reduce position sizes as drawdown increases.

    Parameters:
        target_weight: Desired portfolio weight (0-1)

    Returns:
        Adjusted weight based on drawdown
    """
    current_dd = self.drawdown_tracker.current_drawdown

    if current_dd < 0.05:
        # No adjustment below 5% drawdown
        return target_weight
    elif current_dd < 0.10:
        # Reduce to 75% at 5-10% drawdown
        return target_weight * 0.75
    elif current_dd < 0.15:
        # Reduce to 50% at 10-15% drawdown
        return target_weight * 0.50
    elif current_dd < 0.20:
        # Reduce to 25% at 15-20% drawdown
        return target_weight * 0.25
    else:
        # No new positions above 20%
        return 0
```

## Stop Loss and Take Profit

### Simple Stop Loss and Take Profit

```python
class MyAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.symbol = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.entry_price = {}
        self.stop_loss_pct = 0.05    # 5% stop loss
        self.take_profit_pct = 0.10  # 10% take profit

    def OnOrderEvent(self, orderEvent):
        """Track entry prices."""
        if orderEvent.Status == OrderStatus.Filled:
            symbol = orderEvent.Symbol
            if orderEvent.Direction == OrderDirection.Buy:
                self.entry_price[symbol] = orderEvent.FillPrice
            elif symbol in self.entry_price:
                del self.entry_price[symbol]

    def OnData(self, data):
        for symbol, entry in list(self.entry_price.items()):
            if not data.ContainsKey(symbol):
                continue

            current_price = data[symbol].Close
            pnl_pct = (current_price - entry) / entry

            # Check stop loss
            if pnl_pct <= -self.stop_loss_pct:
                self.Log(f"STOP LOSS: {symbol} at {pnl_pct:.2%}")
                self.Liquidate(symbol)

            # Check take profit
            elif pnl_pct >= self.take_profit_pct:
                self.Log(f"TAKE PROFIT: {symbol} at {pnl_pct:.2%}")
                self.Liquidate(symbol)
```

### Trailing Stop Implementation

```python
class TrailingStopManager:
    """Manage trailing stops for positions."""

    def __init__(self, trailing_pct=0.05):
        self.trailing_pct = trailing_pct
        self.highest_prices = {}

    def update(self, symbol, current_price):
        """Update highest price and check stop."""
        if symbol not in self.highest_prices:
            self.highest_prices[symbol] = current_price
            return False

        # Update highest price
        if current_price > self.highest_prices[symbol]:
            self.highest_prices[symbol] = current_price

        # Calculate stop level
        stop_price = self.highest_prices[symbol] * (1 - self.trailing_pct)

        # Check if stop triggered
        if current_price <= stop_price:
            return True

        return False

    def reset(self, symbol):
        """Reset tracking for symbol."""
        if symbol in self.highest_prices:
            del self.highest_prices[symbol]

    def get_stop_price(self, symbol):
        """Get current stop price for symbol."""
        if symbol in self.highest_prices:
            return self.highest_prices[symbol] * (1 - self.trailing_pct)
        return None

# Usage
class MyAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.trailing_stop = TrailingStopManager(trailing_pct=0.05)

    def OnData(self, data):
        for symbol in self.Portfolio.Keys:
            if not self.Portfolio[symbol].Invested:
                continue

            if not data.ContainsKey(symbol):
                continue

            current_price = data[symbol].Close

            if self.trailing_stop.update(symbol, current_price):
                self.Log(f"TRAILING STOP: {symbol}")
                self.Liquidate(symbol)
                self.trailing_stop.reset(symbol)
```

### Graduated Take Profit

```python
def CheckGraduatedTakeProfit(self, symbol, current_price, entry_price):
    """
    Take partial profits at multiple levels.

    Returns:
        Quantity to sell (0 if no action)
    """
    pnl_pct = (current_price - entry_price) / entry_price
    holding = self.Portfolio[symbol]
    current_qty = holding.Quantity

    # Define profit levels and sell percentages
    profit_levels = [
        (1.00, 0.25),  # +100%: sell 25%
        (2.00, 0.25),  # +200%: sell 25%
        (4.00, 0.25),  # +400%: sell 25%
        (10.0, 0.25),  # +1000%: sell remaining 25%
    ]

    # Track which levels have been hit (need persistent storage)
    if symbol not in self.profit_levels_hit:
        self.profit_levels_hit[symbol] = set()

    for level, sell_pct in profit_levels:
        if pnl_pct >= level and level not in self.profit_levels_hit[symbol]:
            self.profit_levels_hit[symbol].add(level)
            sell_qty = int(current_qty * sell_pct)
            self.Log(f"GRADUATED PROFIT: {symbol} +{pnl_pct:.0%}, selling {sell_qty}")
            return sell_qty

    return 0
```

## Portfolio Risk Metrics

### Risk Metrics Calculator

```python
import numpy as np

class RiskMetrics:
    """Calculate portfolio risk metrics."""

    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0

        excess_returns = returns - risk_free_rate / periods_per_year
        if np.std(excess_returns) == 0:
            return 0

        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe * np.sqrt(periods_per_year)

    @staticmethod
    def calculate_sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) < 2:
            return 0

        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0

        sortino = np.mean(excess_returns) / np.std(downside_returns)
        return sortino * np.sqrt(periods_per_year)

    @staticmethod
    def calculate_max_drawdown(equity_curve):
        """Calculate maximum drawdown from equity curve."""
        if len(equity_curve) < 2:
            return 0

        peak = equity_curve[0]
        max_dd = 0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)

        return max_dd

    @staticmethod
    def calculate_var(returns, confidence=0.95):
        """Calculate Value at Risk."""
        if len(returns) < 10:
            return 0
        return np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def calculate_cvar(returns, confidence=0.95):
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = RiskMetrics.calculate_var(returns, confidence)
        cvar = np.mean(returns[returns <= var])
        return cvar if not np.isnan(cvar) else var

# Usage in algorithm
class MyAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.returns = []
        self.equity_curve = []

    def OnEndOfDay(self, symbol):
        # Track daily returns
        portfolio_value = self.Portfolio.TotalPortfolioValue
        self.equity_curve.append(portfolio_value)

        if len(self.equity_curve) >= 2:
            daily_return = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            self.returns.append(daily_return)

        # Calculate metrics weekly
        if len(self.returns) >= 5 and len(self.returns) % 5 == 0:
            returns_array = np.array(self.returns)
            equity_array = np.array(self.equity_curve)

            sharpe = RiskMetrics.calculate_sharpe_ratio(returns_array)
            max_dd = RiskMetrics.calculate_max_drawdown(equity_array)
            var = RiskMetrics.calculate_var(returns_array)

            self.Log(f"Metrics - Sharpe: {sharpe:.2f}, MaxDD: {max_dd:.2%}, VaR(95%): {var:.2%}")
```

## Sector and Correlation Risk

### Sector Exposure Tracking

```python
class SectorExposureTracker:
    """Track and limit sector exposure."""

    def __init__(self, max_sector_exposure=0.30):
        self.max_sector_exposure = max_sector_exposure
        self.sector_map = {}  # symbol -> sector

    def set_sector(self, symbol, sector):
        """Set sector for symbol."""
        self.sector_map[symbol] = sector

    def get_sector_exposure(self, portfolio):
        """Calculate current sector exposures."""
        total_value = portfolio.TotalPortfolioValue
        sector_values = {}

        for symbol in portfolio.Keys:
            holding = portfolio[symbol]
            if not holding.Invested:
                continue

            position_value = holding.HoldingsValue
            sector = self.sector_map.get(symbol, "Unknown")

            if sector not in sector_values:
                sector_values[sector] = 0
            sector_values[sector] += position_value

        # Convert to percentages
        return {sector: value / total_value for sector, value in sector_values.items()}

    def can_add_position(self, portfolio, symbol, position_value):
        """Check if adding position would exceed sector limit."""
        sector = self.sector_map.get(symbol, "Unknown")
        current_exposure = self.get_sector_exposure(portfolio)
        current_sector = current_exposure.get(sector, 0)

        new_exposure = current_sector + position_value / portfolio.TotalPortfolioValue
        return new_exposure <= self.max_sector_exposure

# Usage with fine fundamental data
def OnSecuritiesChanged(self, changes):
    for security in changes.AddedSecurities:
        symbol = security.Symbol

        # Get sector from fundamental data
        if security.Fundamentals:
            sector_code = security.Fundamentals.AssetClassification.MorningstarSectorCode
            self.sector_tracker.set_sector(symbol, sector_code)
```

### Correlation-Based Position Limits

```python
def CalculateCorrelationMatrix(self, symbols, lookback=60):
    """Calculate correlation matrix for symbols."""

    # Get historical data
    history = self.History(symbols, lookback, Resolution.Daily)

    if history.empty:
        return None

    # Pivot to get returns for each symbol
    returns = history['close'].unstack(level=0).pct_change().dropna()

    # Calculate correlation matrix
    corr_matrix = returns.corr()

    return corr_matrix

def CheckCorrelationRisk(self, new_symbol, max_correlation=0.70):
    """Check if new position is too correlated with existing."""

    existing_symbols = [s for s in self.Portfolio.Keys
                       if self.Portfolio[s].Invested]

    if not existing_symbols:
        return True  # OK to add

    symbols = existing_symbols + [new_symbol]
    corr_matrix = self.CalculateCorrelationMatrix(symbols)

    if corr_matrix is None:
        return True  # Allow if can't calculate

    # Check correlation of new symbol with all existing
    for existing in existing_symbols:
        if existing in corr_matrix.columns and new_symbol in corr_matrix.index:
            correlation = abs(corr_matrix.loc[new_symbol, existing])
            if correlation > max_correlation:
                self.Log(f"HIGH CORRELATION: {new_symbol} and {existing}: {correlation:.2f}")
                return False

    return True
```

## Shorting and Margin

### Short Selling Basics

```python
class ShortSellingAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)

        # Add equity with margin account
        self.symbol = self.AddEquity("SPY", Resolution.Minute).Symbol

        # Set brokerage model for realistic margin/shorting
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

    def OnData(self, data):
        if not data.ContainsKey(self.symbol):
            return

        # Short sell (negative quantity)
        if not self.Portfolio[self.symbol].Invested:
            self.MarketOrder(self.symbol, -100)  # Short 100 shares

        # Check short position
        if self.Portfolio[self.symbol].IsShort:
            quantity = self.Portfolio[self.symbol].Quantity  # Negative
            self.Log(f"Short position: {quantity} shares")

        # Cover short (buy to close)
        if self.Portfolio[self.symbol].IsShort:
            self.Liquidate(self.symbol)  # Buys back shares
```

### Margin Requirements

```python
def Initialize(self):
    self.SetStartDate(2020, 1, 1)
    self.SetCash(100000)

    equity = self.AddEquity("SPY", Resolution.Minute)
    self.symbol = equity.Symbol

    # Set leverage (affects margin requirements)
    equity.SetLeverage(2.0)  # 2:1 leverage (50% margin)

    # Or use brokerage model for realistic margin
    self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

def OnData(self, data):
    # Check margin available
    margin_remaining = self.Portfolio.MarginRemaining
    total_margin_used = self.Portfolio.TotalMarginUsed
    margin_call_warning = self.Portfolio.MarginRemaining < self.Portfolio.TotalPortfolioValue * 0.10

    self.Log(f"Margin Remaining: ${margin_remaining:,.2f}")
    self.Log(f"Total Margin Used: ${total_margin_used:,.2f}")

    if margin_call_warning:
        self.Log("WARNING: Low margin - consider reducing positions")
```

### Margin Models

| Account Type | Initial Margin | Maintenance Margin |
|--------------|----------------|-------------------|
| **Cash** | 100% | N/A |
| **Reg T Margin** | 50% | 25% |
| **Portfolio Margin** | 15-25% | 10-15% |

> **2025 Margin Rules**: Regulation T sets the initial margin at 50% of purchase price. FINRA Rule 4210 sets maintenance margin at 25% of current market value. Brokers may require higher "house" margins. Concentrated accounts (one position ≥ 60% of portfolio) typically require 50% maintenance margin.

```python
def Initialize(self):
    # Different margin requirements by security type
    equity = self.AddEquity("SPY")
    equity.SetLeverage(4.0)  # Requires 25% margin

    # Options have different margin
    option = self.AddOption("SPY")
    # Option margin is calculated based on strategy

    # Futures use different margin model
    future = self.AddFuture(Futures.Indices.SP500EMini)
    # Futures margin is set by exchange
```

### Borrowing Costs (Short Interest)

```python
class ShortingCostModel(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)

        self.symbol = self.AddEquity("GME", Resolution.Minute).Symbol

        # Track borrowing costs
        self.daily_borrow_rate = 0.10 / 365  # 10% annual rate
        self.borrow_costs_paid = 0

    def OnEndOfDay(self, symbol):
        """Calculate daily borrowing costs for short positions."""
        if not self.Portfolio[symbol].IsShort:
            return

        # Short position value (positive number)
        short_value = abs(self.Portfolio[symbol].HoldingsValue)

        # Daily borrowing cost
        daily_cost = short_value * self.daily_borrow_rate
        self.borrow_costs_paid += daily_cost

        self.Log(f"Daily borrow cost for {symbol}: ${daily_cost:.2f}")
        self.Log(f"Total borrow costs: ${self.borrow_costs_paid:.2f}")

    def GetBorrowRate(self, symbol):
        """
        Estimate borrow rate based on characteristics.
        In practice, this varies by broker and stock availability.
        """
        # Easy to borrow (large caps, high float)
        easy_to_borrow_rate = 0.003  # 0.3% annual

        # General to borrow
        general_rate = 0.02  # 2% annual

        # Hard to borrow (high short interest, low float)
        hard_to_borrow_rate = 0.10  # 10%+ annual

        # Very hard to borrow (meme stocks, squeeze candidates)
        extreme_rate = 0.50  # 50%+ annual

        # Would need external data to determine actual rate
        return general_rate
```

### Short Selling Risk Management

```python
class ShortRiskManager:
    """Manage risks specific to short selling."""

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.max_short_exposure = 0.30  # 30% of portfolio
        self.max_single_short = 0.05    # 5% per position
        self.short_squeeze_threshold = 0.10  # 10% daily move

    def can_short(self, symbol, quantity, price):
        """Check if short position is within risk limits."""
        portfolio = self.algorithm.Portfolio

        # Calculate proposed short exposure
        proposed_value = abs(quantity * price)
        proposed_pct = proposed_value / portfolio.TotalPortfolioValue

        # Check single position limit
        if proposed_pct > self.max_single_short:
            self.algorithm.Log(f"Short {symbol} exceeds single position limit")
            return False

        # Calculate total short exposure
        current_short_exposure = sum(
            abs(portfolio[s].HoldingsValue)
            for s in portfolio.Keys
            if portfolio[s].IsShort
        )

        new_total_short = (current_short_exposure + proposed_value) / portfolio.TotalPortfolioValue

        if new_total_short > self.max_short_exposure:
            self.algorithm.Log(f"Total short exposure would exceed limit: {new_total_short:.2%}")
            return False

        return True

    def check_short_squeeze_risk(self, symbol, current_price, previous_price):
        """Monitor for potential short squeeze."""
        if previous_price == 0:
            return False

        daily_move = (current_price - previous_price) / previous_price

        if daily_move > self.short_squeeze_threshold:
            self.algorithm.Log(f"SHORT SQUEEZE WARNING: {symbol} up {daily_move:.2%}")
            return True

        return False

# Usage
def Initialize(self):
    self.short_risk = ShortRiskManager(self)

def OnData(self, data):
    if self.short_risk.can_short(self.symbol, -100, data[self.symbol].Close):
        self.MarketOrder(self.symbol, -100)
```

### Locate and Borrow Availability

```python
def CheckShortAvailability(self, symbol):
    """
    Check if shares are available to borrow.
    Note: QuantConnect doesn't have real-time locate data in backtests.
    In live trading, this is handled by the broker.
    """
    # In backtests, assume shares are available
    # In live trading, broker will reject if unavailable

    # Could implement custom logic based on:
    # - Short interest data
    # - Market cap (larger = easier to borrow)
    # - Float (higher float = easier to borrow)

    security = self.Securities[symbol]

    # Example heuristic
    if security.Fundamentals:
        market_cap = security.Fundamentals.MarketCap
        if market_cap > 10e9:  # $10B+
            return True, 0.003  # Available, low rate
        elif market_cap > 1e9:  # $1B+
            return True, 0.02   # Available, moderate rate
        else:
            return True, 0.10   # May be hard to borrow

    return True, 0.05  # Default assumption
```

### Margin Call Handling

```python
class MarginCallHandler:
    """Handle margin calls and forced liquidations."""

    def __init__(self, algorithm, maintenance_margin_pct=0.25):
        self.algorithm = algorithm
        self.maintenance_margin_pct = maintenance_margin_pct
        self.warning_threshold = 0.30  # Warn at 30%

    def check_margin_status(self):
        """Check margin status and take action if needed."""
        portfolio = self.algorithm.Portfolio

        total_value = portfolio.TotalPortfolioValue
        margin_used = portfolio.TotalMarginUsed
        margin_remaining = portfolio.MarginRemaining

        if total_value <= 0:
            return "CRITICAL"

        margin_ratio = margin_remaining / total_value

        if margin_ratio < self.maintenance_margin_pct:
            # Margin call - must reduce positions
            self.algorithm.Log("MARGIN CALL: Reducing positions")
            self.reduce_positions_for_margin()
            return "MARGIN_CALL"

        elif margin_ratio < self.warning_threshold:
            self.algorithm.Log(f"MARGIN WARNING: {margin_ratio:.2%} remaining")
            return "WARNING"

        return "OK"

    def reduce_positions_for_margin(self):
        """Reduce positions to meet margin requirements."""
        portfolio = self.algorithm.Portfolio

        # Find largest positions to reduce
        positions = [
            (s, abs(portfolio[s].HoldingsValue))
            for s in portfolio.Keys
            if portfolio[s].Invested
        ]
        positions.sort(key=lambda x: x[1], reverse=True)

        # Reduce positions until margin is OK
        for symbol, value in positions:
            holding = portfolio[symbol]
            reduce_qty = int(holding.Quantity * 0.25)  # Reduce by 25%

            if reduce_qty != 0:
                self.algorithm.MarketOrder(symbol, -reduce_qty)
                self.algorithm.Log(f"Margin reduction: {symbol} by {reduce_qty}")

            # Check if margin is now OK
            if portfolio.MarginRemaining / portfolio.TotalPortfolioValue > self.warning_threshold:
                break

# Usage
def Initialize(self):
    self.margin_handler = MarginCallHandler(self)

def OnData(self, data):
    status = self.margin_handler.check_margin_status()
    if status == "MARGIN_CALL":
        return  # Don't trade during margin call
```

## Best Practices

### 1. Always Use Position Limits

```python
def Initialize(self):
    # Set maximum position size
    self.max_position_size = 0.10  # 10% max per position
    self.max_total_positions = 20

def BeforeTradeCheck(self, symbol, target_weight):
    """Validate trade before execution."""

    # Check position size
    if target_weight > self.max_position_size:
        self.Log(f"Reducing {symbol} position from {target_weight:.2%} to {self.max_position_size:.2%}")
        target_weight = self.max_position_size

    # Check total positions
    current_positions = sum(1 for s in self.Portfolio.Keys if self.Portfolio[s].Invested)
    if not self.Portfolio[symbol].Invested and current_positions >= self.max_total_positions:
        self.Log(f"Max positions reached, skipping {symbol}")
        return 0

    return target_weight
```

### 2. Log All Risk Events

```python
def LogRiskEvent(self, event_type, symbol, details):
    """Standardized risk event logging."""
    timestamp = self.Time.strftime("%Y-%m-%d %H:%M:%S")
    self.Log(f"[RISK][{event_type}][{timestamp}] {symbol}: {details}")

    # Could also store for analysis
    if not hasattr(self, 'risk_events'):
        self.risk_events = []

    self.risk_events.append({
        'time': self.Time,
        'type': event_type,
        'symbol': str(symbol),
        'details': details
    })
```

### 3. Test Risk Management Separately

```python
def TestRiskManagement(self):
    """Unit test risk management logic."""

    # Test position sizing
    size = self.CalculatePositionSize(self.symbol, risk_per_trade=0.02)
    assert size > 0, "Position size should be positive"
    assert size * self.Securities[self.symbol].Price < self.Portfolio.TotalPortfolioValue * 0.02, \
        "Position size exceeds risk limit"

    # Test drawdown calculation
    test_equity = [100, 110, 105, 95, 100]
    max_dd = RiskMetrics.calculate_max_drawdown(test_equity)
    assert abs(max_dd - 0.136) < 0.01, f"Max drawdown calculation error: {max_dd}"

    self.Log("Risk management tests passed")
```

### 4. Use Multiple Layers of Protection

```python
def Initialize(self):
    # Layer 1: Position-level stops
    self.stop_loss_pct = 0.05
    self.take_profit_pct = 0.15

    # Layer 2: Portfolio-level limits
    self.max_daily_loss = 0.03
    self.max_drawdown = 0.15

    # Layer 3: Built-in risk model
    self.SetRiskManagement(CompositeRiskManagementModel(
        MaximumDrawdownPercentPerSecurity(0.07),
        MaximumDrawdownPercentPortfolio(0.20)
    ))
```

### 5. Backtest with Different Market Conditions

```python
# Test across multiple periods
test_periods = [
    ("Bull Market", 2017, 2019),
    ("Bear Market", 2008, 2009),
    ("High Volatility", 2020, 2020),
    ("Low Volatility", 2014, 2015),
]

# For each period, verify:
# - Max drawdown within limits
# - Risk metrics acceptable
# - No blown accounts
```

## Pattern Day Trader (PDT) Rule

### Current Rule (2025)

Under FINRA rules, you're classified as a pattern day trader if you execute 4+ day trades within 5 business days in a margin account (where day trades exceed 6% of total trades).

**Current Requirements:**

- Minimum equity: **$25,000** in margin account
- If account falls below, no day trading until restored
- Rule applies only to margin accounts (cash accounts exempt)
- Futures and Forex are exempt (regulated by CFTC/NFA, not FINRA)

### 2025 Rule Changes (Pending SEC Approval)

> **Major Update**: FINRA Board approved PDT rule amendments on **September 24, 2025** to eliminate the $25,000 minimum. Pending SEC approval (expected late 2025 or early 2026).

**Proposed Changes:**

- Remove fixed $25,000 minimum requirement
- Replace with intraday maintenance margin requirement
- Pattern day traders must maintain enough to avoid margin calls (typically 25% of position value)
- No more blanket restrictions on day trading

**Until Approved:** The current $25,000 PDT requirement remains in force.

### PDT Workarounds

```python
def Initialize(self):
    # Option 1: Use cash account (no PDT restrictions)
    self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Cash)
    # Note: Must wait for settlement (T+1 for stocks)

    # Option 2: Trade futures (CFTC regulated, no PDT)
    self.AddFuture(Futures.Indices.SP500EMini)

    # Option 3: Monitor day trade count
    self.day_trade_count = 0
    self.day_trade_limit = 3  # Stay under 4

def OnOrderEvent(self, orderEvent):
    if orderEvent.Status == OrderStatus.Filled:
        # Track if this is a day trade (same-day round trip)
        if self.IsDayTrade(orderEvent):
            self.day_trade_count += 1
            if self.day_trade_count >= self.day_trade_limit:
                self.Log("WARNING: Approaching PDT limit")
```

---

*Last Updated: November 2025*
