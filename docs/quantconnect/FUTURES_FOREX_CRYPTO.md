# Futures, Forex, and Crypto Trading on QuantConnect

Guide to trading futures, forex, and cryptocurrency on QuantConnect, including contract handling, continuous futures, and crypto exchange integration.

## Table of Contents

- [Futures Trading](#futures-trading)
- [Forex Trading](#forex-trading)
- [Cryptocurrency Trading](#cryptocurrency-trading)
- [Multi-Asset Strategies](#multi-asset-strategies)

---

## Futures Trading

### Adding Futures

```python
from AlgorithmImports import *

class FuturesAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        # Add continuous futures contract
        self.future = self.AddFuture(
            Futures.Indices.SP500EMini,
            Resolution.Minute,
            extendedMarketHours=True,
            dataNormalizationMode=DataNormalizationMode.BackwardsRatio,
            dataMappingMode=DataMappingMode.OpenInterest,
            contractDepthOffset=0  # Front month
        )

        # Set filter for contracts
        self.future.SetFilter(0, 182)  # 0-182 days to expiration
```

### Available Futures

| Category | Symbol | Example |
|----------|--------|---------|
| **Indices** | `Futures.Indices.SP500EMini` | E-mini S&P 500 |
| **Indices** | `Futures.Indices.NASDAQ100EMini` | E-mini Nasdaq |
| **Indices** | `Futures.Indices.Dow30EMini` | E-mini Dow |
| **Metals** | `Futures.Metals.Gold` | Gold |
| **Metals** | `Futures.Metals.Silver` | Silver |
| **Energy** | `Futures.Energies.CrudeOilWTI` | Crude Oil |
| **Energy** | `Futures.Energies.NaturalGas` | Natural Gas |
| **Grains** | `Futures.Grains.Corn` | Corn |
| **Currencies** | `Futures.Currencies.EUR` | Euro |

### Continuous Contracts

> **2025 Update**: QuantConnect provides **continuous futures support with proprietary mappings** (free, built into the cloud). The US Futures dataset by AlgoSeek covers the **157 most liquid contracts** starting from **May 2009** with tick-to-daily frequency. Supports **3 contract rolling methods** and **4 price scaling adjustments** with back month support.
>
> **SymbolChangedEvent Timing**: In backtesting, contract rollovers occur at **midnight Eastern Time (ET)**. In live trading, live data for continuous contract mapping arrives at **6-7 AM ET**.

Continuous contracts stitch together multiple contract months for backtesting:

```python
def Initialize(self):
    self.future = self.AddFuture(
        Futures.Indices.SP500EMini,
        Resolution.Minute,
        # When to roll to next contract
        dataMappingMode=DataMappingMode.OpenInterest,
        # How to adjust prices at rollover
        dataNormalizationMode=DataNormalizationMode.BackwardsRatio,
        # Which contract (0=front month, 1=next, etc.)
        contractDepthOffset=0  # 0=front, 1=first back, 2=second back
    )
```

#### Data Mapping Modes (When to Roll)

| Mode | Description | VIX Support |
|------|-------------|-------------|
| `OpenInterest` | Roll when back month has higher open interest | **No** (VIX limitation) |
| `OpenInterestAnnual` | Annual rollover based on open interest | **No** (VIX limitation) |
| `LastTradingDay` | Roll on last trading day | Yes |
| `FirstDayMonth` | Roll on first day of delivery month | Yes |

> **VIX Futures Limitation**: VIX Futures (VX) do **not** support continuous contract rolling with `OpenInterest` or `OpenInterestAnnual` mapping modes. Use `LastTradingDay` or `FirstDayMonth` for VIX.

#### Data Normalization Modes (Price Adjustment)

| Mode | Description | Use Case |
|------|-------------|----------|
| `Raw` | No adjustment (gaps at rollover) | Order execution, real prices |
| `ForwardPanamaCanal` | Additive adjustment, first contract is true | Price trend analysis |
| `BackwardsPanamaCanal` | Additive adjustment, last contract is true | Backtesting continuity |
| `BackwardsRatio` | Multiplicative adjustment, last contract is true | Technical indicators (recommended) |

### Handling Rollovers

```python
def OnData(self, data):
    # Check for rollover event
    for changed in data.SymbolChangedEvents.Values:
        self.Log(f"Rollover: {changed.OldSymbol} -> {changed.NewSymbol}")

        # Close old contract position
        if self.Portfolio[changed.OldSymbol].Invested:
            self.Liquidate(changed.OldSymbol)

    # Get the mapped (current) contract
    if self.future.Mapped:
        mapped_symbol = self.future.Mapped
        self.Log(f"Current contract: {mapped_symbol}")

        # Trade the mapped contract
        if not self.Portfolio[mapped_symbol].Invested:
            self.MarketOrder(mapped_symbol, 1)
```

### Front Month Selection

```python
def Initialize(self):
    self.future = self.AddFuture(Futures.Indices.SP500EMini)

    # Filter to front month only
    self.future.SetFilter(lambda universe: universe.FrontMonth())

    # Or specific expiration range
    self.future.SetFilter(0, 90)  # Next 90 days

def OnData(self, data):
    # Access the futures chain
    chain = data.FutureChains.get(self.future.Symbol)
    if not chain:
        return

    # Get front month contract
    contracts = sorted(chain, key=lambda x: x.Expiry)
    if contracts:
        front_month = contracts[0]
        self.Log(f"Front month: {front_month.Symbol}, Expiry: {front_month.Expiry}")
```

### Futures Contract Properties

```python
for contract in chain:
    symbol = contract.Symbol
    expiry = contract.Expiry
    open_interest = contract.OpenInterest
    last_price = contract.LastPrice
    bid = contract.BidPrice
    ask = contract.AskPrice
    volume = contract.Volume

    # Calculate DTE
    dte = (expiry - self.Time).days
```

---

## Forex Trading

> **2025 Update**: QuantConnect provides **71 forex pairs** from OANDA with data starting from various dates (earliest from **January 2007**). Available in tick, second, minute, hourly, and daily resolutions. Default leverage is **50:1** (max 50x with OANDA margin accounts). Market.OANDA is the only market available for forex pairs.

### Adding Currency Pairs

```python
class ForexAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        # Add forex pairs (Market.OANDA is implicit)
        self.eurusd = self.AddForex("EURUSD", Resolution.Hour).Symbol
        self.gbpusd = self.AddForex("GBPUSD", Resolution.Hour).Symbol
        self.usdjpy = self.AddForex("USDJPY", Resolution.Hour).Symbol

        # Set account currency (default is USD)
        self.SetAccountCurrency("USD")
```

### Available Pairs

> **2025 Verification**: QuantConnect provides **71 forex pairs** from OANDA, covering majors, crosses, and exotics.

- **Major Pairs**: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD
- **Cross Pairs**: EURGBP, EURJPY, GBPJPY, AUDNZD
- **Exotic Pairs**: USDMXN, USDZAR, EURTRY

### Leverage and Position Sizing

```python
def Initialize(self):
    # Add forex with custom leverage (default is 50:1)
    forex = self.AddForex("EURUSD", Resolution.Hour)
    forex.SetLeverage(35)  # Custom leverage (max 50:1 with OANDA)

    # Access leverage
    leverage = self.Securities["EURUSD"].Leverage

def OnData(self, data):
    # Calculate position size based on margin
    margin_remaining = self.Portfolio.MarginRemaining
    price = self.Securities["EURUSD"].Price
    leverage = self.Securities["EURUSD"].Leverage

    max_position = margin_remaining * leverage / price
    self.Log(f"Max position: {max_position:,.0f}")

    # Order in lots (standard lot = 100,000 units)
    self.MarketOrder("EURUSD", 10000)  # 0.1 lot
```

### Pip Calculations

```python
def Initialize(self):
    self.eurusd = self.AddForex("EURUSD", Resolution.Hour).Symbol
    self.usdjpy = self.AddForex("USDJPY", Resolution.Hour).Symbol

def OnData(self, data):
    # Get minimum price variation (pipette)
    pip_eurusd = self.Securities["EURUSD"].SymbolProperties.MinimumPriceVariation
    # pip_eurusd = 0.00001 (pipette), pip = 0.0001

    pip_usdjpy = self.Securities["USDJPY"].SymbolProperties.MinimumPriceVariation
    # pip_usdjpy = 0.001 (pipette), pip = 0.01

    # Calculate pip value
    position_size = 100000  # Standard lot
    pip_value_eurusd = position_size * 0.0001  # $10 per pip

def CalculateStopLoss(self, symbol, entry_price, pips):
    """Calculate stop loss price based on pips."""
    pip_size = self.Securities[symbol].SymbolProperties.MinimumPriceVariation * 10

    if "JPY" in str(symbol):
        pip_size = 0.01
    else:
        pip_size = 0.0001

    # For buy order, stop is below entry
    stop_price = entry_price - (pips * pip_size)
    return stop_price
```

### Multi-Pair Strategy

```python
class MultiPairForexAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        # Major pairs
        self.pairs = [
            self.AddForex("EURUSD", Resolution.Hour).Symbol,
            self.AddForex("GBPUSD", Resolution.Hour).Symbol,
            self.AddForex("USDJPY", Resolution.Hour).Symbol,
            self.AddForex("AUDUSD", Resolution.Hour).Symbol,
        ]

        # Create indicators for each pair
        self.rsi = {}
        self.sma = {}

        for pair in self.pairs:
            self.rsi[pair] = self.RSI(pair, 14, Resolution.Hour)
            self.sma[pair] = self.SMA(pair, 50, Resolution.Hour)

        self.SetWarmUp(50)

    def OnData(self, data):
        if self.IsWarmingUp:
            return

        for pair in self.pairs:
            if not data.ContainsKey(pair):
                continue

            if not self.rsi[pair].IsReady:
                continue

            rsi = self.rsi[pair].Current.Value
            sma = self.sma[pair].Current.Value
            price = data[pair].Close

            # Simple strategy
            if rsi < 30 and price > sma:
                self.SetHoldings(pair, 0.2)
            elif rsi > 70:
                self.Liquidate(pair)
```

---

## Cryptocurrency Trading

### Adding Crypto

```python
class CryptoAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)

        # Set account currency for crypto
        self.SetAccountCurrency("USDT")  # Tether for crypto exchanges
        self.SetCash("USDT", 100000)

        # Set brokerage model
        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Cash)

        # Add crypto pairs
        self.btc = self.AddCrypto("BTCUSDT", Resolution.Hour).Symbol
        self.eth = self.AddCrypto("ETHUSDT", Resolution.Hour).Symbol
```

### Supported Exchanges

| Exchange | Brokerage Model | Data Coverage |
|----------|-----------------|---------------|
| Binance | `BrokerageName.Binance` | 718 Crypto pairs (Spot + Futures) |
| Binance Futures | `BrokerageName.BINANCE_FUTURES` | USDT/COIN perpetuals since Aug 2020 |
| Bybit | `BrokerageName.Bybit` | 631 Crypto pairs since Aug 2020 |
| Coinbase | `BrokerageName.Coinbase` | Spot trading only |
| Kraken | `BrokerageName.Kraken` | Spot trading |
| Bitfinex | `BrokerageName.Bitfinex` | Spot trading |

> **2025 Update**: Crypto Perpetual Futures are now supported via Binance and Bybit. Binance Crypto Future Price Data (by CoinAPI) covers 718 pairs; Bybit covers 631 pairs. Both start from August 2020 with tick-to-daily frequency. Bybit fees are 0.1% maker/taker at VIP 0 level. When using Binance, you must set account currency to USDT (not USD).

### Trading Crypto

```python
def OnData(self, data):
    if not data.ContainsKey(self.btc):
        return

    btc_price = data[self.btc].Close

    # Place orders
    self.MarketOrder(self.btc, 0.1)  # Buy 0.1 BTC

    # Check holdings
    btc_holdings = self.Portfolio[self.btc].Quantity
    btc_value = self.Portfolio[self.btc].HoldingsValue

    # Set holdings as percentage
    self.SetHoldings(self.btc, 0.5)  # 50% in BTC
```

### Cross-Pair Trading

```python
class CryptoCrossPairAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetAccountCurrency("USDT")
        self.SetCash("USDT", 100000)

        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Cash)

        # Direct pairs
        self.btcusdt = self.AddCrypto("BTCUSDT", Resolution.Hour).Symbol
        self.ethusdt = self.AddCrypto("ETHUSDT", Resolution.Hour).Symbol

        # Cross pair for ratio analysis
        self.ethbtc = self.AddCrypto("ETHBTC", Resolution.Hour).Symbol

        # EMA for ETH/BTC ratio
        self.eth_btc_ema = self.EMA(self.ethbtc, 20, Resolution.Hour)

    def OnData(self, data):
        if not self.eth_btc_ema.IsReady:
            return

        # Trade based on ETH/BTC ratio
        eth_btc_price = data[self.ethbtc].Close
        eth_btc_ema = self.eth_btc_ema.Current.Value

        if eth_btc_price < eth_btc_ema * 0.95:
            # ETH undervalued vs BTC - buy ETH
            self.SetHoldings(self.ethusdt, 0.5)
            self.Liquidate(self.btcusdt)
        elif eth_btc_price > eth_btc_ema * 1.05:
            # BTC undervalued vs ETH - buy BTC
            self.SetHoldings(self.btcusdt, 0.5)
            self.Liquidate(self.ethusdt)
```

### 24/7 Trading Considerations

```python
def Initialize(self):
    # Crypto trades 24/7 - schedule events carefully
    self.Schedule.On(
        self.DateRules.EveryDay(),
        self.TimeRules.Every(timedelta(hours=4)),  # Every 4 hours
        self.Rebalance
    )

def Rebalance(self):
    """Rebalance every 4 hours."""
    # No market open/close for crypto
    pass

def OnData(self, data):
    # Check for weekend data (crypto trades weekends)
    if self.Time.weekday() >= 5:
        self.Log("Weekend trading active")
```

---

## Multi-Asset Strategies

### Stocks + Crypto Strategy

```python
class MultiAssetAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        # Equities
        self.spy = self.AddEquity("SPY", Resolution.Hour).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Hour).Symbol

        # Crypto (separate cash)
        self.SetCash("USDT", 50000)
        self.btc = self.AddCrypto("BTCUSDT", Resolution.Hour,
                                  Market.Binance).Symbol

    def OnData(self, data):
        # Trade both asset classes
        if data.ContainsKey(self.spy):
            pass  # Stock logic

        if data.ContainsKey(self.btc):
            pass  # Crypto logic
```

### Futures + Options Hedging

```python
class HedgedFuturesAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        # S&P 500 futures
        self.es = self.AddFuture(Futures.Indices.SP500EMini)
        self.es.SetFilter(0, 90)

        # SPY options for hedging
        self.spy = self.AddEquity("SPY", Resolution.Minute)
        self.spy_options = self.AddOption("SPY", Resolution.Minute)
        self.spy_options.SetFilter(-5, 5, 30, 60)

    def OnData(self, data):
        # Long futures, hedge with puts
        pass
```

### Risk Parity Across Assets

```python
class RiskParityMultiAsset(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        # Different asset classes
        self.assets = {
            'equity': self.AddEquity("SPY", Resolution.Daily).Symbol,
            'bond': self.AddEquity("TLT", Resolution.Daily).Symbol,
            'gold': self.AddEquity("GLD", Resolution.Daily).Symbol,
            'commodity': self.AddEquity("DBC", Resolution.Daily).Symbol,
        }

        self.target_vol = 0.10  # 10% target portfolio volatility

        self.Schedule.On(
            self.DateRules.MonthStart(),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.Rebalance
        )

    def Rebalance(self):
        """Risk parity rebalancing."""
        # Get historical volatility for each asset
        weights = {}

        for name, symbol in self.assets.items():
            history = self.History(symbol, 60, Resolution.Daily)
            if history.empty:
                continue

            returns = history['close'].pct_change().dropna()
            vol = returns.std() * np.sqrt(252)  # Annualized

            # Inverse volatility weight
            weights[symbol] = 1 / vol if vol > 0 else 0

        # Normalize weights
        total = sum(weights.values())
        for symbol in weights:
            weights[symbol] /= total
            self.SetHoldings(symbol, weights[symbol])

        self.Log(f"Rebalanced: {weights}")
```

---

**Sources:**
- [Futures Documentation](https://www.quantconnect.com/docs/v2/writing-algorithms/universes/futures)
- [Continuous Futures Support](https://www.quantconnect.com/forum/discussion/12644/continuous-futures-support/)
- [FOREX Data](https://www.quantconnect.com/data/quantconnect-forex)
- [Crypto Trades](https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/crypto-trades)
- [Crypto Requesting Data](https://www.quantconnect.com/docs/v2/writing-algorithms/securities/asset-classes/crypto/requesting-data)

*Last Updated: November 2025*
