# Brokerage Integration on QuantConnect

Guide to connecting QuantConnect algorithms to brokerages for live trading, with focus on Charles Schwab integration.

## Table of Contents

- [Overview](#overview)
- [Supported Brokerages](#supported-brokerages)
- [Charles Schwab Integration](#charles-schwab-integration)
- [Brokerage Models](#brokerage-models)
- [Order Handling](#order-handling)
- [Live Trading Setup](#live-trading-setup)
- [Paper Trading](#paper-trading)
- [Monitoring and Alerts](#monitoring-and-alerts)
- [Notifications](#notifications)
- [Live Deployment](#live-deployment)
- [Common Issues](#common-issues)

## Overview

### Live Trading Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Live Trading Flow                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  QuantConnect Cloud                                         │
│  ├── Your Algorithm                                         │
│  ├── LEAN Engine                                            │
│  └── Brokerage Handler                                      │
│           │                                                 │
│           ▼                                                 │
│  Brokerage API                                              │
│  ├── Authentication                                         │
│  ├── Market Data (optional)                                 │
│  ├── Order Routing                                          │
│  └── Account Management                                     │
│           │                                                 │
│           ▼                                                 │
│  Exchange / Market                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Supported Brokerages

### Full Brokerage List (2025)

| Brokerage | Asset Classes | Notes |
|-----------|---------------|-------|
| **Charles Schwab** | Equities, Options, Index Options | Primary US retail broker |
| **Interactive Brokers** | All | Most comprehensive API, requires IBKR PRO |
| **TradeStation** | Equities, Options, Futures | Full-featured platform |
| **Tastytrade** | Equities, Options | Options-focused |
| **Tradier** | Equities, Options | Commission-free options |
| **Alpaca** | Equities, Crypto | Commission-free |
| **Coinbase** | Crypto | Crypto only, 0.6%/0.8% maker/taker |
| **Binance** | Crypto (Spot, Futures) | Includes BinanceUS and Futures |
| **Bybit** | Crypto (Spot, Futures) | 0.1% maker/taker at VIP 0 |
| **Kraken** | Crypto | Crypto only |
| **Bitfinex** | Crypto | Crypto only |
| **OANDA** | Forex | Forex focused |
| **Bloomberg EMSX** | Institutional | Enterprise only |
| **Trading Technologies** | Futures | Professional trading |
| **Wolverine** | Institutional | Enterprise only |

> **Note**: TD Ameritrade API was disabled on May 10, 2024 following Schwab's acquisition. All former TD Ameritrade users should use Charles Schwab.
>
> **2025 Update**: For most crypto trading, Coinbase is recommended. For Forex, OANDA is popular and trustworthy. Interactive Brokers requires an IBKR PRO plan (LITE not supported).

## Charles Schwab Integration

### Overview

Charles Schwab acquired TD Ameritrade and now provides API access for algorithmic trading. QuantConnect supports Schwab through their API.

### Setting Up Schwab Connection

```python
def Initialize(self):
    # Set brokerage model
    self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)

    # Alternative: explicit model
    # self.SetBrokerageModel(CharlesSchwabBrokerageModel())

    # Configure for live trading
    self.SetCash(0)  # Use broker's cash balance

    # Add securities
    self.AddEquity("SPY", Resolution.Minute)
```

### Schwab API Requirements

1. **Developer Account**: Register at [developer.schwab.com](https://developer.schwab.com)
2. **API Keys**: Create application to get client ID and secret
3. **OAuth 2.0**: Schwab uses OAuth for authentication
4. **Permissions**: Request appropriate scopes for trading

### Configuration in QuantConnect

Add credentials to your QuantConnect account:

1. Go to Algorithm Lab → Live Trading
2. Select Charles Schwab
3. Enter credentials:
   - Client ID
   - Client Secret
   - Refresh Token (obtained via OAuth flow)
   - Account Number

### Schwab-Specific Considerations

```python
def Initialize(self):
    self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)

    # Schwab market hours
    # Regular: 9:30 AM - 4:00 PM ET
    # Extended hours available for some orders

    # Schwab order types supported:
    # - Market
    # - Limit
    # - Stop
    # - Stop Limit
    # - Trailing Stop

    # Options trading requires approval
    self.AddOption("SPY", Resolution.Minute)
```

> **2025 Note**: Schwab supports Equities, Options, and Index Options. **Futures are NOT supported** - this was removed from the QuantConnect roadmap as the Schwab API does not support futures trading. Schwab also does not offer paper trading; use QuantConnect's built-in paper trading for testing.

### Schwab Fee Structure

```python
# Schwab fees (as of 2025)
# Stocks: $0 commission
# Options: $0.65 per contract
# ETFs: $0 commission
# Index Options: $0.65 per contract

# Account for fees in backtesting
self.SetSecurityInitializer(lambda x: x.SetFeeModel(ConstantFeeModel(0)))

# Or use built-in Schwab fee model
# self.SetSecurityInitializer(lambda x: x.SetFeeModel(CharlesSchwabFeeModel()))
```

### Schwab API Rate Limits

Charles Schwab enforces strict API rate limits that critically impact algorithmic trading throughput.

#### Rate Limit Structure

| Limit Type | Rate | Notes |
|------------|------|-------|
| **Order Operations** | 0-120 requests/min | Configurable per app |
| **Non-Order Operations** | 120 requests/min | Market data, quotes |
| **Burst Limit** | 2-4 requests/sec | Short-term throttling |
| **GET Order Status** | Unthrottled | Check order status freely |

> **Critical Understanding**: The 120/minute limit refers to **API requests**, not orders. Since Schwab does not support batch order submission, each individual order consumes one API request.

#### HTTP 429 Rate Limit Errors

```python
def OnBrokerageMessage(self, message):
    """Handle broker messages including rate limit errors."""

    if "429" in message.Message:
        # Rate limit hit - back off for 60 seconds
        self.rate_limited = True
        self.rate_limit_until = self.Time + timedelta(seconds=60)
        self.Log(f"RATE LIMITED: {message.Message}")

        # Error codes:
        # 429-001: General rate limit exceeded
        # 429-005: Burst limit exceeded

        if "429-001" in message.Message:
            self.Log("General rate limit - reduce order frequency")
        elif "429-005" in message.Message:
            self.Log("Burst limit - add delays between orders")

def OnData(self, data):
    # Check rate limit status before trading
    if hasattr(self, 'rate_limited') and self.rate_limited:
        if self.Time < self.rate_limit_until:
            return  # Wait for rate limit to clear
        self.rate_limited = False

    # Trading logic here
```

> **Important**: When rate limits are exceeded, orders are **rejected immediately**, not queued. Your algorithm must handle this gracefully.

#### Daily Order Capacity

Based on 120 orders/minute limit:

| Trading Window | Max Orders | Notes |
|----------------|------------|-------|
| **Per Minute** | 120 | Hard limit |
| **Per Hour** | 7,200 | 120 × 60 |
| **Regular Hours (6.5h)** | 46,800 | 9:30 AM - 4:00 PM ET |
| **Extended Hours (+4h)** | 24,000 | Pre-market + After-hours |
| **Full Day (10.5h)** | 70,800 | Maximum theoretical |

#### Order Rate Management

```python
class SchwabRateLimiter:
    """Manage Schwab API rate limits."""

    def __init__(self, algorithm, max_per_minute=100):  # Leave headroom
        self.algorithm = algorithm
        self.max_per_minute = max_per_minute
        self.order_timestamps = []
        self.min_order_interval = timedelta(milliseconds=500)  # Avoid burst limit
        self.last_order_time = None

    def can_place_order(self):
        """Check if we can place an order without hitting rate limits."""
        now = self.algorithm.Time

        # Clean old timestamps (older than 1 minute)
        cutoff = now - timedelta(minutes=1)
        self.order_timestamps = [t for t in self.order_timestamps if t > cutoff]

        # Check minute limit
        if len(self.order_timestamps) >= self.max_per_minute:
            return False

        # Check burst limit (500ms between orders)
        if self.last_order_time:
            if now - self.last_order_time < self.min_order_interval:
                return False

        return True

    def record_order(self):
        """Record an order placement."""
        now = self.algorithm.Time
        self.order_timestamps.append(now)
        self.last_order_time = now

    def orders_remaining(self):
        """Get remaining orders this minute."""
        now = self.algorithm.Time
        cutoff = now - timedelta(minutes=1)
        recent = len([t for t in self.order_timestamps if t > cutoff])
        return self.max_per_minute - recent

# Usage in algorithm
def Initialize(self):
    self.rate_limiter = SchwabRateLimiter(self, max_per_minute=100)

def OnData(self, data):
    if not self.rate_limiter.can_place_order():
        self.Log(f"Rate limited - {self.rate_limiter.orders_remaining()} orders remaining")
        return

    # Place order
    ticket = self.MarketOrder("SPY", 100)
    self.rate_limiter.record_order()
```

### Schwab OAuth Authentication

Schwab uses OAuth 2.0 with specific token expiration rules that require careful management.

#### Token Expiration

| Token Type | Expiration | Notes |
|------------|------------|-------|
| **Access Token** | 30 minutes | Required for every API request |
| **Refresh Token** | 7 days | Used to get new access tokens |

#### Authentication Lifecycle

```text
Initial OAuth Flow (Browser Required)
         │
         ▼
┌─────────────────────┐
│   Access Token      │──── Valid for 30 minutes
│   Refresh Token     │──── Valid for 7 days
└─────────────────────┘
         │
         │ Every 28-29 minutes
         ▼
┌─────────────────────┐
│ Token Refresh       │──── Automatic via refresh token
│ (No browser needed) │
└─────────────────────┘
         │
         │ Every 7 days
         ▼
┌─────────────────────┐
│ Full Re-auth        │──── Browser login required
│ (Manual step)       │
└─────────────────────┘
```

#### Best Practices

1. **Proactive Token Refresh**: Refresh access tokens every 28-29 minutes, before expiration
2. **Monitor for 401 Errors**: Indicates expired or invalid token
3. **7-Day Re-authentication**: Schedule manual re-authentication before refresh token expires
4. **QuantConnect Handling**: QuantConnect's LEAN engine handles token refresh automatically
5. **Email Notifications (2025)**: QuantConnect will send you an email when OAuth needs refresh, allowing you to restart the connection from your phone or computer

```python
def Initialize(self):
    # Set reminder for manual re-authentication
    self.last_oauth_refresh = self.Time
    self.oauth_refresh_interval = timedelta(days=6)  # 1 day buffer before 7-day expiry

    self.Schedule.On(
        self.DateRules.EveryDay(),
        self.TimeRules.At(9, 0),
        self.CheckOAuthStatus
    )

def CheckOAuthStatus(self):
    """Remind about OAuth re-authentication."""
    days_since_auth = (self.Time - self.last_oauth_refresh).days

    if days_since_auth >= 6:
        self.Notify.Email(
            "trader@example.com",
            "Schwab OAuth Refresh Required",
            f"Your Schwab refresh token will expire in {7 - days_since_auth} day(s). "
            "Please re-authenticate via the QuantConnect dashboard."
        )
```

### Single Algorithm Per Account

> **Critical Limitation**: Charles Schwab only supports authenticating **one account at a time per user**. If you deploy a second algorithm to the same Schwab account, the first algorithm **stops running**.

```python
# INCORRECT: Two algorithms on same Schwab account
# Algorithm 1 - Will STOP when Algorithm 2 deploys
class Algorithm1(QCAlgorithm):
    def Initialize(self):
        self.SetBrokerageModel(BrokerageName.CharlesSchwab)  # Uses account X

# Algorithm 2 - Deploying this kills Algorithm 1
class Algorithm2(QCAlgorithm):
    def Initialize(self):
        self.SetBrokerageModel(BrokerageName.CharlesSchwab)  # Also uses account X
```

**Solutions**:

1. **Multiple Schwab Accounts**: Use separate accounts for different strategies
2. **Single Multi-Strategy Algorithm**: Combine strategies into one algorithm
3. **Strategy Selection**: Choose which strategy runs at any given time

```python
class MultiStrategyAlgorithm(QCAlgorithm):
    """Combine multiple strategies in single algorithm for Schwab."""

    def Initialize(self):
        self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)

        # Initialize multiple strategies
        self.momentum_strategy = MomentumStrategy(self)
        self.mean_reversion = MeanReversionStrategy(self)
        self.options_strategy = OptionsStrategy(self)

        # Allocate capital to each
        self.strategy_allocations = {
            'momentum': 0.40,
            'mean_reversion': 0.30,
            'options': 0.30
        }

    def OnData(self, data):
        # Run all strategies with their allocation
        portfolio_value = self.Portfolio.TotalPortfolioValue

        self.momentum_strategy.Execute(data, portfolio_value * 0.40)
        self.mean_reversion.Execute(data, portfolio_value * 0.30)
        self.options_strategy.Execute(data, portfolio_value * 0.30)
```

### Schwab Order Latency

Expect 100ms to 5 seconds latency for order execution through Schwab's API:

| Operation | Typical Latency | Notes |
|-----------|-----------------|-------|
| **Order Submission** | 100-500ms | Market conditions dependent |
| **Order Confirmation** | 500ms-2s | Includes exchange routing |
| **Fill Notification** | 1-5s | Depends on order type |

```python
def OnData(self, data):
    # Don't expect instant fills - design for latency
    if self.pending_order and not self.pending_order.Status.IsFill():
        # Order still pending - don't submit new orders
        order_age = (self.Time - self.pending_order.Time).total_seconds()
        if order_age > 30:
            self.Log(f"Order pending for {order_age}s - may need attention")
        return

    # Safe to submit new order
    self.pending_order = self.MarketOrder("SPY", 100)
```

### TD Ameritrade API Status (2025)

> **PERMANENTLY DISCONTINUED**: The TD Ameritrade API was **permanently discontinued** following Charles Schwab's acquisition. As of 2025, **Schwab has not released a replacement retail API**, despite earlier mentions of plans. The TD Ameritrade developer platform is closed to new registrations.
>
> **Impact**: While QuantConnect documentation still lists TD Ameritrade integration, **new users cannot use it**. Existing users with active developer accounts may continue using the legacy API temporarily, but this is unsupported and may cease functioning at any time.

**What to Do**:

1. **For New Users**: Use Charles Schwab's integration via QuantConnect (no direct retail API from Schwab)
2. **For Existing TD API Users**: Migrate to Interactive Brokers, Schwab (via QC), or other supported brokerages
3. **Do NOT wait** for Schwab to release a retail API - no concrete timeline exists as of 2025

## Extended Hours Trading by Broker

> **2025 Update**: Extended hours trading support varies significantly by brokerage. Only **Interactive Brokers** has confirmed support for extended hours trading through QuantConnect's integration.

### Broker Support Matrix

| Brokerage | Pre-Market | After-Hours | Implementation | Notes |
|-----------|------------|-------------|----------------|-------|
| **Interactive Brokers** | ✅ Yes | ✅ Yes | `InteractiveBrokersOrderProperties` | Full support with limit orders |
| **Charles Schwab** | ⚠️ Limited | ⚠️ Limited | Via QuantConnect integration | Check with broker |
| **Tradier** | ❌ No | ❌ No | Orders queued until market open | Blocked through QC integration |
| **Other Brokers** | Varies | Varies | Check documentation | Confirm before deploying |

### Implementing Extended Hours Trading (Interactive Brokers)

```python
from AlgorithmImports import *

class ExtendedHoursAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)

        # Enable extended market hours data
        self.symbol = self.AddEquity(
            "SPY",
            Resolution.Minute,
            extendedMarketHours=True  # CRITICAL: Must enable
        ).Symbol

    def OnData(self, data):
        if not data.ContainsKey(self.symbol):
            return

        # Only receive extended hours data if minute/second/tick resolution
        # Daily resolution will NOT receive extended hours data

        # Use limit orders (NOT market orders) during extended hours
        if not self.Portfolio.Invested:
            # Set order properties for extended hours
            order_properties = InteractiveBrokersOrderProperties()
            order_properties.OutsideRegularTradingHours = True

            # Place limit order (market orders not supported after-hours)
            self.LimitOrder(
                self.symbol,
                100,
                data[self.symbol].Close * 1.001,  # Slightly above market
                orderProperties=order_properties
            )
```

### Extended Hours Limitations

**Important Notes:**

- **Market orders NOT supported** during extended hours (limited liquidity)
- **Use limit orders only** to avoid poor fills
- **Data resolution** must be minute, second, or tick (NOT daily)
- **Extended hours** include:
  - **Pre-market**: 4:00 AM - 9:30 AM ET
  - **After-hours**: 4:00 PM - 8:00 PM ET
- **Liquidity is lower** outside regular hours, expect wider spreads
- **Tradier** will queue orders and execute at market open (not true extended hours trading)

## Brokerage Models

### Setting Brokerage Model

```python
def Initialize(self):
    # Using enum
    self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

    # Using explicit model
    self.SetBrokerageModel(InteractiveBrokersBrokerageModel())

    # With account type
    self.SetBrokerageModel(
        InteractiveBrokersBrokerageModel(AccountType.Cash)
    )
```

### Brokerage Model Components

| Component | Description |
|-----------|-------------|
| **Fee Model** | Commission and fee calculation |
| **Fill Model** | Order fill simulation |
| **Slippage Model** | Price slippage simulation |
| **Buying Power Model** | Margin/leverage calculation |
| **Settlement Model** | T+1, T+2 settlement |

### Custom Brokerage Model

```python
from QuantConnect.Brokerages import DefaultBrokerageModel

class MyBrokerageModel(DefaultBrokerageModel):
    """Custom brokerage model with specific rules."""

    def __init__(self):
        super().__init__()

    def GetFeeModel(self, security):
        """Custom fee structure."""
        if security.Type == SecurityType.Option:
            return ConstantFeeModel(0.65)  # $0.65 per contract
        return ConstantFeeModel(0)  # Free stock trades

    def GetFillModel(self, security):
        """Custom fill model."""
        return ImmediateFillModel()

    def GetSlippageModel(self, security):
        """Custom slippage model."""
        return ConstantSlippageModel(0.001)  # 0.1% slippage

    def GetBuyingPowerModel(self, security):
        """Custom buying power."""
        if security.Type == SecurityType.Equity:
            return SecurityMarginModel(2.0)  # 2x leverage
        return super().GetBuyingPowerModel(security)

# Usage
def Initialize(self):
    self.SetBrokerageModel(MyBrokerageModel())
```

## Short Selling and Borrow Costs

> **2025 Update**: The US Equity Short Availability dataset provides available shares for short positions and borrowing costs for **10,500+ US Equities** starting **January 2018** (daily frequency). In live trading, borrowing rates are set by your brokerage. LEAN currently does **not model borrowing costs in backtests**, but this is tracked in GitHub Issue #4563.

### Short Selling by Brokerage

| Brokerage | Short Selling Support | Borrow Cost Data | Notes |
|-----------|----------------------|------------------|-------|
| **Interactive Brokers** | ✅ Full support | ✅ Real-time via API | Gold standard for professional short sellers |
| **Charles Schwab** | ✅ Equities only | ⚠️ Account-dependent | Check with broker for HTB stocks |
| **Other Brokerages** | Varies | Varies | Check brokerage documentation |

### Borrow Cost Modeling

```python
from AlgorithmImports import *

class ShortSellingAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        # Add equity
        self.symbol = self.AddEquity("GME", Resolution.Daily).Symbol

        # NOTE: Borrow costs are NOT modeled in backtests by default
        # For live trading, use InteractiveBrokersShortableProvider

    def OnData(self, data):
        if not self.Portfolio.Invested:
            # Short sale (negative quantity)
            self.SetHoldings(self.symbol, -0.25)  # 25% short

        # Monitor position
        if self.Portfolio[self.symbol].IsShort:
            unrealized_pnl = self.Portfolio[self.symbol].UnrealizedProfit
            self.Log(f"Short position P&L: ${unrealized_pnl:.2f}")
```

### Interactive Brokers Borrow Rates

```python
# For live trading with Interactive Brokers
def Initialize(self):
    self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)

    # Access shortable provider (live trading only)
    # self.shortable_provider = InteractiveBrokersShortableProvider()

def OnData(self, data):
    # Check if shares are available to short (live trading)
    # fee_rate = self.Securities[self.symbol].ShortableProvider.FeeRate(self.symbol, self.Time)
    # rebate_rate = self.Securities[self.symbol].ShortableProvider.RebateRate(self.symbol, self.Time)

    # In live trading, IB provides real-time borrow costs
    pass
```

### Hard-to-Borrow (HTB) Considerations

> **2025 Margin Rules**: Under Regulation T, short sales require a deposit equal to **150%** of the position value (100% value + 50% margin requirement). Interactive Brokers starting margin rate is **5.58% APR** (as of November 2025).

**Important Notes:**

- **Easy-to-borrow stocks**: Readily available, low/no borrow fees
- **Hard-to-borrow (HTB) stocks**: Limited inventory, elevated borrow fees (can exceed 100% annually in extreme cases)
- **Borrow fees are NOT fixed**: Rates change intraday based on supply/demand in the securities lending market
- **Extreme example**: SharpLink Gaming (SBET) reached **1,000% borrow fee** in May 2025 during a price surge

### Short Availability Dataset

```python
# Using QuantConnect's Short Availability dataset (backtest only)
from QuantConnect.DataSource import *

def Initialize(self):
    self.symbol = self.AddEquity("GME", Resolution.Daily).Symbol

    # Add short availability data
    self.short_data = self.AddData(
        USEquityShortAvailability,
        self.symbol,
        Resolution.Daily
    ).Symbol

def OnData(self, data):
    # Access short availability info
    if data.ContainsKey(self.short_data):
        short_info = data[self.short_data]
        # available_shares = short_info.Quantity
        # borrow_fee_rate = short_info.FeeRate

        # Only short if borrow cost is reasonable
        # if borrow_fee_rate < 0.10:  # Less than 10% annual
        #     self.SetHoldings(self.symbol, -0.25)
```

## Order Handling

### Order Types by Brokerage

```python
def Initialize(self):
    self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)

def OnData(self, data):
    # Market Order - immediate execution
    self.MarketOrder("SPY", 100)

    # Limit Order - specify price
    self.LimitOrder("SPY", 100, 400.00)

    # Stop Market - trigger at stop price
    self.StopMarketOrder("SPY", -100, 395.00)

    # Stop Limit - stop trigger + limit execution
    self.StopLimitOrder("SPY", -100, 395.00, 394.50)

    # Trailing Stop (if supported)
    # self.TrailingStopOrder("SPY", -100, 5.00)  # $5 trail
```

### Order Events

```python
def OnOrderEvent(self, orderEvent):
    """Handle order status changes."""

    order = self.Transactions.GetOrderById(orderEvent.OrderId)

    if orderEvent.Status == OrderStatus.Submitted:
        self.Log(f"Order submitted: {order.Symbol}")

    elif orderEvent.Status == OrderStatus.PartiallyFilled:
        self.Log(f"Partial fill: {orderEvent.FillQuantity} @ {orderEvent.FillPrice}")

    elif orderEvent.Status == OrderStatus.Filled:
        self.Log(f"Order filled: {orderEvent.FillQuantity} @ {orderEvent.FillPrice}")
        self.Log(f"Fees: ${orderEvent.OrderFee.Value.Amount}")

    elif orderEvent.Status == OrderStatus.Canceled:
        self.Log(f"Order canceled: {order.Symbol}")

    elif orderEvent.Status == OrderStatus.Invalid:
        self.Log(f"Order invalid: {orderEvent.Message}")
```

### Handling Partial Fills

```python
def Initialize(self):
    self.pending_orders = {}

def OnData(self, data):
    if "SPY" not in self.pending_orders:
        ticket = self.LimitOrder("SPY", 1000, data["SPY"].Close - 0.10)
        self.pending_orders["SPY"] = ticket

def OnOrderEvent(self, orderEvent):
    symbol = orderEvent.Symbol

    if orderEvent.Status == OrderStatus.PartiallyFilled:
        ticket = self.pending_orders.get(str(symbol))
        if ticket:
            filled = ticket.QuantityFilled
            remaining = ticket.Quantity - filled
            self.Log(f"Partial fill: {filled}, remaining: {remaining}")

    elif orderEvent.Status in [OrderStatus.Filled, OrderStatus.Canceled]:
        if str(symbol) in self.pending_orders:
            del self.pending_orders[str(symbol)]
```

## Live Trading Setup

### Pre-Launch Checklist

```python
class LiveTradingAlgorithm(QCAlgorithm):
    """Algorithm configured for live trading."""

    def Initialize(self):
        # 1. Set brokerage model
        self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)

        # 2. Use broker's cash (not SetCash)
        # self.SetCash(0)  # For live trading

        # 3. Add securities
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol

        # 4. Set warmup period
        self.SetWarmUp(timedelta(days=30))

        # 5. Configure risk management
        self.SetRiskManagement(MaximumDrawdownPercentPortfolio(0.10))

        # 6. Set realistic position limits
        self.max_position_value = 10000  # $10K max per position

        # 7. Configure notifications
        self.Notify.Email("your@email.com", "Algo Started", "Algorithm is now live")

    def OnData(self, data):
        if self.IsWarmingUp:
            return

        # Live trading logic
        pass

    def OnEndOfAlgorithm(self):
        self.Notify.Email("your@email.com", "Algo Stopped",
                         f"Final value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
```

### Live Trading Best Practices

```python
def Initialize(self):
    # Always log important state
    self.Log(f"Starting algorithm - Cash: ${self.Portfolio.Cash:,.2f}")

    # Use scheduled functions for time-sensitive operations
    self.Schedule.On(
        self.DateRules.EveryDay(self.spy),
        self.TimeRules.AfterMarketOpen(self.spy, 5),
        self.MarketOpenRoutine
    )

    # Health check
    self.Schedule.On(
        self.DateRules.EveryDay(),
        self.TimeRules.Every(timedelta(hours=1)),
        self.HealthCheck
    )

def MarketOpenRoutine(self):
    """Run every day 5 minutes after market open."""
    self.Log(f"Market open - Portfolio: ${self.Portfolio.TotalPortfolioValue:,.2f}")

    # Cancel any stale orders
    for order in self.Transactions.GetOpenOrders():
        if (self.Time - order.Time).total_seconds() > 3600:  # 1 hour old
            self.Transactions.CancelOrder(order.Id)
            self.Log(f"Canceled stale order: {order.Id}")

def HealthCheck(self):
    """Periodic health check."""
    portfolio_value = self.Portfolio.TotalPortfolioValue
    cash = self.Portfolio.Cash
    margin_used = self.Portfolio.TotalMarginUsed

    self.Log(f"Health: Value=${portfolio_value:,.2f}, Cash=${cash:,.2f}, Margin=${margin_used:,.2f}")

    # Alert if unusual
    if cash < 0:
        self.Notify.Email("your@email.com", "ALERT: Negative Cash",
                         f"Cash balance is negative: ${cash:,.2f}")
```

## Paper Trading

### Setting Up Paper Trading

```python
def Initialize(self):
    # Paper trading uses same code as live
    # Difference is in deployment settings

    self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)

    # For paper trading, set starting cash
    if self.LiveMode:
        pass  # Use broker cash
    else:
        self.SetCash(100000)

    # Enable detailed logging for paper trading
    self.Settings.FreePortfolioValuePercentage = 0.05  # Keep 5% cash buffer
```

### Paper vs Live Comparison

| Aspect | Paper Trading | Live Trading |
|--------|--------------|--------------|
| Fills | Simulated (instant) | Real (may partial fill) |
| Slippage | Model-based | Real market impact |
| Fees | Model-based | Real broker fees |
| Data | Same as live | Same |
| Risk | None | Real capital |

## Monitoring and Alerts

### Built-in Notifications

```python
def Initialize(self):
    # Email notifications
    self.email = "your@email.com"

def OnOrderEvent(self, orderEvent):
    if orderEvent.Status == OrderStatus.Filled:
        # Notify on fills
        self.Notify.Email(
            self.email,
            f"Order Filled: {orderEvent.Symbol}",
            f"Filled {orderEvent.FillQuantity} @ ${orderEvent.FillPrice:.2f}"
        )

def OnData(self, data):
    # Alert on significant events
    daily_pnl = self.Portfolio.TotalProfit

    if daily_pnl < -1000:  # $1000 loss threshold
        self.Notify.Email(
            self.email,
            "ALERT: Significant Loss",
            f"Daily P&L: ${daily_pnl:,.2f}"
        )
```

### Custom Monitoring

```python
class TradingMonitor:
    """Monitor algorithm health and performance."""

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.daily_start_value = 0
        self.order_count = 0
        self.error_count = 0

    def start_of_day(self):
        """Reset daily metrics."""
        self.daily_start_value = self.algorithm.Portfolio.TotalPortfolioValue
        self.order_count = 0

    def record_order(self):
        """Track orders."""
        self.order_count += 1

        # Alert if too many orders
        if self.order_count > 100:
            self.algorithm.Notify.Email(
                "your@email.com",
                "ALERT: High Order Count",
                f"Orders today: {self.order_count}"
            )

    def record_error(self, error_msg):
        """Track errors."""
        self.error_count += 1
        self.algorithm.Log(f"ERROR: {error_msg}")

        if self.error_count > 10:
            self.algorithm.Notify.Email(
                "your@email.com",
                "ALERT: Multiple Errors",
                f"Error count: {self.error_count}"
            )

    def get_daily_pnl(self):
        """Calculate daily P&L."""
        current_value = self.algorithm.Portfolio.TotalPortfolioValue
        return current_value - self.daily_start_value

# Usage
def Initialize(self):
    self.monitor = TradingMonitor(self)

    self.Schedule.On(
        self.DateRules.EveryDay(),
        self.TimeRules.AfterMarketOpen(self.spy, 1),
        self.monitor.start_of_day
    )
```

### Runtime Statistics

```python
def OnEndOfDay(self, symbol):
    """Update runtime statistics for monitoring."""

    # Portfolio metrics
    self.SetRuntimeStatistic("Portfolio Value",
                            f"${self.Portfolio.TotalPortfolioValue:,.2f}")
    self.SetRuntimeStatistic("Cash",
                            f"${self.Portfolio.Cash:,.2f}")
    self.SetRuntimeStatistic("Invested",
                            f"{self.Portfolio.Invested}")

    # Today's metrics
    if hasattr(self, 'monitor'):
        daily_pnl = self.monitor.get_daily_pnl()
        self.SetRuntimeStatistic("Daily P&L", f"${daily_pnl:,.2f}")
        self.SetRuntimeStatistic("Orders Today", str(self.monitor.order_count))
```

## Notifications

QuantConnect provides multiple notification channels for alerts and status updates.

### Email Notifications

```python
def Initialize(self):
    self.email = "trader@example.com"

def OnOrderEvent(self, orderEvent):
    if orderEvent.Status == OrderStatus.Filled:
        self.Notify.Email(
            self.email,
            f"Order Filled: {orderEvent.Symbol}",
            f"""
            Symbol: {orderEvent.Symbol}
            Quantity: {orderEvent.FillQuantity}
            Price: ${orderEvent.FillPrice:.2f}
            Time: {self.Time}
            """
        )

def OnEndOfDay(self, symbol):
    # Daily summary email
    self.Notify.Email(
        self.email,
        f"Daily Summary - {self.Time.strftime('%Y-%m-%d')}",
        f"""
        Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}
        Daily P&L: ${self.Portfolio.TotalProfit:,.2f}
        Cash: ${self.Portfolio.Cash:,.2f}
        Positions: {sum(1 for s in self.Portfolio.Keys if self.Portfolio[s].Invested)}
        """
    )
```

### SMS Notifications

```python
def Initialize(self):
    # Phone number with country code
    self.phone = "+1234567890"

def OnData(self, data):
    # Send SMS for urgent alerts only (rate limited)
    daily_loss = self.Portfolio.TotalProfit
    if daily_loss < -5000:  # $5K loss threshold
        self.Notify.Sms(
            self.phone,
            f"ALERT: Daily loss ${abs(daily_loss):,.0f}"
        )
```

### Webhook Notifications

```python
def Initialize(self):
    # Webhook URL (Slack, Discord, custom endpoint)
    self.webhook_url = "https://hooks.slack.com/services/xxx/yyy/zzz"

def SendWebhook(self, message, data=None):
    """Send notification to webhook endpoint."""
    import json

    payload = {
        "text": message,
        "data": data or {}
    }

    self.Notify.Web(
        self.webhook_url,
        json.dumps(payload)
    )

def OnOrderEvent(self, orderEvent):
    if orderEvent.Status == OrderStatus.Filled:
        self.SendWebhook(
            f":chart_with_upwards_trend: Order Filled",
            {
                "symbol": str(orderEvent.Symbol),
                "quantity": orderEvent.FillQuantity,
                "price": orderEvent.FillPrice
            }
        )
```

### Telegram Integration

```python
def Initialize(self):
    # Telegram bot token and chat ID
    self.telegram_token = "your_bot_token"
    self.telegram_chat_id = "your_chat_id"
    self.telegram_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"

def SendTelegram(self, message):
    """Send message to Telegram."""
    import json

    payload = {
        "chat_id": self.telegram_chat_id,
        "text": message,
        "parse_mode": "HTML"
    }

    self.Notify.Web(self.telegram_url, json.dumps(payload))

def OnData(self, data):
    # Send Telegram alerts
    if self.Portfolio.TotalProfit > 1000:
        self.SendTelegram(
            f"<b>Profit Alert!</b>\n"
            f"Portfolio P&L: ${self.Portfolio.TotalProfit:,.2f}"
        )
```

### Discord Integration

```python
def Initialize(self):
    self.discord_webhook = "https://discord.com/api/webhooks/xxx/yyy"

def SendDiscord(self, title, description, color=0x00ff00):
    """Send embed message to Discord."""
    import json

    payload = {
        "embeds": [{
            "title": title,
            "description": description,
            "color": color,
            "timestamp": self.Time.isoformat()
        }]
    }

    self.Notify.Web(self.discord_webhook, json.dumps(payload))

def OnOrderEvent(self, orderEvent):
    if orderEvent.Status == OrderStatus.Filled:
        color = 0x00ff00 if orderEvent.Direction == OrderDirection.Buy else 0xff0000
        self.SendDiscord(
            f"Order Filled: {orderEvent.Symbol}",
            f"**Qty:** {orderEvent.FillQuantity}\n**Price:** ${orderEvent.FillPrice:.2f}",
            color
        )
```

### Notification Rate Limiting

```python
class NotificationManager:
    """Manage notification rate limiting."""

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.last_notification = {}
        self.min_interval = timedelta(minutes=5)  # Minimum 5 minutes between same type

    def can_notify(self, notification_type):
        """Check if notification can be sent."""
        now = self.algorithm.Time

        if notification_type not in self.last_notification:
            self.last_notification[notification_type] = now
            return True

        elapsed = now - self.last_notification[notification_type]
        if elapsed >= self.min_interval:
            self.last_notification[notification_type] = now
            return True

        return False

    def send_email(self, subject, body, notification_type="general"):
        """Send rate-limited email."""
        if self.can_notify(notification_type):
            self.algorithm.Notify.Email("trader@example.com", subject, body)

    def send_sms(self, message, notification_type="sms"):
        """Send rate-limited SMS."""
        if self.can_notify(notification_type):
            self.algorithm.Notify.Sms("+1234567890", message)

# Usage
def Initialize(self):
    self.notifications = NotificationManager(self)

def OnData(self, data):
    if self.Portfolio.TotalProfit < -1000:
        self.notifications.send_email(
            "Loss Alert",
            f"Loss: ${abs(self.Portfolio.TotalProfit):,.2f}",
            notification_type="loss_alert"
        )
```

## Live Deployment

### Deploying to QuantConnect Cloud

#### Via Web Interface

1. **Navigate to Algorithm Lab**
2. **Select your algorithm**
3. **Click "Go Live"**
4. **Configure settings:**
   - Select brokerage
   - Enter credentials
   - Set data feeds
   - Configure notifications
5. **Review and deploy**

#### Via LEAN CLI

```bash
# Install LEAN CLI
pip install lean

# Login to QuantConnect
lean login

# Deploy to live trading
lean cloud live deploy --brokerage "Charles Schwab" --project "MyAlgorithm"

# Check live algorithm status
lean cloud live list

# Stop live algorithm
lean cloud live stop --project "MyAlgorithm"
```

### Deployment Configuration

```python
class DeploymentReadyAlgorithm(QCAlgorithm):
    """Algorithm configured for production deployment."""

    def Initialize(self):
        # Determine environment
        self.is_live = self.LiveMode

        # Configure based on environment
        if self.is_live:
            self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)
            # Don't set cash - use broker's balance
        else:
            self.SetCash(100000)

        # Common setup
        self.SetStartDate(2020, 1, 1)
        self.symbol = self.AddEquity("SPY", Resolution.Minute).Symbol

        # Risk management (critical for live)
        self.SetRiskManagement(MaximumDrawdownPercentPortfolio(0.15))

        # Warmup
        self.SetWarmUp(timedelta(days=30))

        # Logging
        if self.is_live:
            self.Log("LIVE TRADING STARTED")
            self.Notify.Email("trader@example.com", "Algorithm Started",
                            f"Live trading started at {self.Time}")

    def OnData(self, data):
        if self.IsWarmingUp:
            return

        # Your trading logic
        pass

    def OnEndOfAlgorithm(self):
        if self.is_live:
            self.Notify.Email("trader@example.com", "Algorithm Stopped",
                            f"Final value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
```

### Pre-Deployment Checklist

```python
def RunPreDeploymentChecks(self):
    """Validate algorithm before going live."""
    checks = []

    # 1. Brokerage model set
    checks.append(("Brokerage Model", self.BrokerageModel is not None))

    # 2. Risk management configured
    checks.append(("Risk Management", hasattr(self, 'RiskManagement')))

    # 3. Warmup period set
    checks.append(("Warmup Period", self.Settings.WarmupResolution is not None))

    # 4. Position limits defined
    checks.append(("Position Limits", hasattr(self, 'max_position_size')))

    # 5. Notifications configured
    checks.append(("Notifications", hasattr(self, 'email') or hasattr(self, 'phone')))

    # Report
    all_passed = True
    for name, passed in checks:
        status = "✓" if passed else "✗"
        self.Log(f"{status} {name}")
        if not passed:
            all_passed = False

    return all_passed
```

### Monitoring Live Algorithm

```python
def Initialize(self):
    # Schedule health checks
    self.Schedule.On(
        self.DateRules.EveryDay(),
        self.TimeRules.Every(timedelta(minutes=30)),
        self.PerformHealthCheck
    )

    # Track metrics
    self.trades_today = 0
    self.errors_today = 0
    self.last_data_time = None

def PerformHealthCheck(self):
    """Periodic health check for live trading."""

    # Check data freshness
    if self.last_data_time:
        data_age = self.Time - self.last_data_time
        if data_age > timedelta(minutes=5):
            self.Log(f"WARNING: Data may be stale ({data_age})")
            self.Notify.Email("trader@example.com", "Data Freshness Warning",
                            f"Last data: {self.last_data_time}")

    # Check portfolio health
    portfolio_value = self.Portfolio.TotalPortfolioValue
    margin_remaining = self.Portfolio.MarginRemaining

    if margin_remaining < portfolio_value * 0.20:
        self.Notify.Email("trader@example.com", "Low Margin Warning",
                        f"Margin remaining: ${margin_remaining:,.2f}")

    # Log status
    self.Log(f"Health Check - Value: ${portfolio_value:,.2f}, "
            f"Trades: {self.trades_today}, Errors: {self.errors_today}")

def OnData(self, data):
    self.last_data_time = self.Time
    # ... trading logic
```

### Rollback Strategy

```python
def Initialize(self):
    # Store initial state for potential rollback
    self.deployment_time = self.Time
    self.initial_positions = {}

def StoreInitialState(self):
    """Store positions at deployment for rollback."""
    for symbol in self.Portfolio.Keys:
        holding = self.Portfolio[symbol]
        if holding.Invested:
            self.initial_positions[symbol] = {
                'quantity': holding.Quantity,
                'avg_price': holding.AveragePrice
            }

def EmergencyLiquidate(self, reason):
    """Emergency liquidation with notification."""
    self.Log(f"EMERGENCY LIQUIDATE: {reason}")

    # Notify immediately
    self.Notify.Sms("+1234567890", f"EMERGENCY: {reason}")
    self.Notify.Email("trader@example.com", "Emergency Liquidation",
                     f"Reason: {reason}\nTime: {self.Time}")

    # Liquidate all
    self.Liquidate()

    # Cancel all pending orders
    for order in self.Transactions.GetOpenOrders():
        self.Transactions.CancelOrder(order.Id)

def OnData(self, data):
    # Emergency conditions
    daily_loss = self.Portfolio.TotalProfit
    if daily_loss < -10000:  # $10K loss threshold
        self.EmergencyLiquidate("Daily loss limit exceeded")
        return

    # Normal trading
    pass
```

## Common Issues

### Connection Issues

```python
def OnBrokerageDisconnect(self):
    """Handle broker disconnection."""
    self.Log("BROKER DISCONNECTED")
    self.Notify.Email(
        "your@email.com",
        "ALERT: Broker Disconnected",
        "Algorithm has lost connection to broker"
    )

    # Cancel pending orders
    for order in self.Transactions.GetOpenOrders():
        self.Transactions.CancelOrder(order.Id)

def OnBrokerageReconnect(self):
    """Handle broker reconnection."""
    self.Log("BROKER RECONNECTED")

    # Verify portfolio state
    self.Log(f"Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")

    # Check for any orphaned positions
    for symbol in self.Portfolio.Keys:
        holding = self.Portfolio[symbol]
        if holding.Invested:
            self.Log(f"Position: {symbol} - {holding.Quantity} shares")
```

### Order Rejection

```python
def OnOrderEvent(self, orderEvent):
    if orderEvent.Status == OrderStatus.Invalid:
        self.Log(f"ORDER REJECTED: {orderEvent.Message}")

        # Common rejection reasons:
        # - Insufficient buying power
        # - Invalid order type for security
        # - Market closed
        # - Pattern day trader restrictions

        # Handle specific rejections
        if "buying power" in orderEvent.Message.lower():
            self.Log("Insufficient margin - reducing position sizes")
            # Could implement position reduction logic

        elif "market closed" in orderEvent.Message.lower():
            self.Log("Market is closed - queuing for next open")
            # Could implement order queue
```

### Troubleshooting Checklist

1. **Authentication Issues**
   - Verify API credentials are correct
   - Check OAuth token hasn't expired
   - Ensure account has API access enabled

2. **Order Issues**
   - Verify sufficient buying power
   - Check order type is supported
   - Confirm market hours
   - Verify symbol is valid

3. **Data Issues**
   - Ensure symbol is subscribed
   - Check resolution is appropriate
   - Verify data is flowing

4. **Performance Issues**
   - Monitor algorithm CPU/memory
   - Check for infinite loops
   - Review logging volume

---

*Last Updated: November 2025*
