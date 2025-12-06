# Universe Selection on QuantConnect

Dynamic security selection allows algorithms to automatically subscribe to and trade securities that match specific criteria. This is essential for strategies that trade many securities or need to adapt to changing market conditions.

## Table of Contents

- [Overview](#overview)
- [Universe Types](#universe-types)
- [Coarse Universe Selection](#coarse-universe-selection)
- [Fine Universe Selection](#fine-universe-selection)
- [Combined Selection](#combined-selection)
- [ETF Constituents](#etf-constituents)
- [Options Universe](#options-universe)
- [Custom Universes](#custom-universes)
- [Best Practices](#best-practices)

## Overview

### What is Universe Selection?

Universe selection is the process of dynamically choosing which securities to trade. Instead of hardcoding symbols, your algorithm can:

- Filter stocks by market cap, volume, price
- Select based on fundamental data (P/E, revenue, etc.)
- Track index constituents
- Add/remove securities as conditions change

> **2025 Update**: QuantConnect now offers `FundamentalUniverseSelectionModel` as the modern approach to universe selection. The legacy Coarse/Fine two-step process is still supported but the new model simplifies filtering. Universe selection runs once per day (at midnight in backtests). The fundamental data is powered by Morningstar® and covers approximately 8,000 US Equities with 900+ properties each. Note: ETFs, ADRs, and OTC equities are not included in fundamental data.

### Universe Selection Flow

```
┌─────────────────────────────────────────────────────────────┐
│                 Universe Selection Flow                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Coarse Filter    → Price, Volume, HasFundamental       │
│         │                                                   │
│         ▼                                                   │
│  2. Fine Filter      → Market Cap, P/E, Sector, Industry   │
│         │                                                   │
│         ▼                                                   │
│  3. OnSecuritiesChanged() → Handle additions/removals      │
│         │                                                   │
│         ▼                                                   │
│  4. OnData()         → Trade selected securities           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Universe Types

| Type | Description | Data Available |
|------|-------------|----------------|
| **Coarse** | Basic price/volume data | Price, Volume, Dollar Volume |
| **Fine** | Fundamental data | All financial metrics |
| **ETF** | ETF constituents | Holdings, weights |
| **Options** | Option contracts | Strikes, expiries |
| **Futures** | Future contracts | Expiries |

## Coarse Universe Selection

### Basic Coarse Selection

```python
def Initialize(self):
    self.SetStartDate(2020, 1, 1)
    self.SetEndDate(2023, 12, 31)
    self.SetCash(100000)

    # Add universe with coarse filter only
    self.AddUniverse(self.CoarseSelectionFunction)

    # Control how often universe updates
    self.UniverseSettings.Resolution = Resolution.Daily

def CoarseSelectionFunction(self, coarse):
    """
    Filter stocks based on price and volume.

    Parameters:
        coarse: List of CoarseFundamental objects

    Returns:
        List of Symbol objects to include in universe
    """
    # Filter criteria
    filtered = [x for x in coarse
                if x.HasFundamentalData           # Has fundamental data
                and x.Price > 10                   # Price > $10
                and x.DollarVolume > 1000000]      # $1M+ daily volume

    # Sort by dollar volume and take top 100
    sorted_by_volume = sorted(filtered,
                              key=lambda x: x.DollarVolume,
                              reverse=True)

    return [x.Symbol for x in sorted_by_volume[:100]]
```

### Coarse Fundamental Properties

| Property | Description |
|----------|-------------|
| `Symbol` | Security symbol |
| `Price` | Current price |
| `Volume` | Trading volume |
| `DollarVolume` | Price × Volume |
| `HasFundamentalData` | Has Fine data available |
| `Market` | Market (USA, etc.) |
| `PriceFactor` | Split/dividend adjustment |

### Momentum-Based Coarse Selection

```python
def Initialize(self):
    self.AddUniverse(self.CoarseSelectionFunction)

    # Store momentum data
    self.momentum = {}
    self.momentum_period = 252  # 1 year

def CoarseSelectionFunction(self, coarse):
    """Select stocks with highest momentum."""

    # Basic filters
    filtered = [x for x in coarse
                if x.HasFundamentalData
                and x.Price > 5
                and x.DollarVolume > 500000]

    # Calculate momentum for each
    for stock in filtered:
        symbol = stock.Symbol
        if symbol not in self.momentum:
            self.momentum[symbol] = MomentumPercent(self.momentum_period)

        # Update momentum indicator
        self.momentum[symbol].Update(self.Time, stock.Price)

    # Get stocks with ready momentum indicators
    ready = [x for x in filtered
             if x.Symbol in self.momentum
             and self.momentum[x.Symbol].IsReady]

    # Sort by momentum
    sorted_by_momentum = sorted(ready,
        key=lambda x: self.momentum[x.Symbol].Current.Value,
        reverse=True)

    return [x.Symbol for x in sorted_by_momentum[:50]]
```

## Fine Universe Selection

### Basic Fine Selection

```python
def Initialize(self):
    # Add universe with both coarse and fine filters
    self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)

def CoarseSelectionFunction(self, coarse):
    """Initial filter - price and volume."""
    filtered = [x for x in coarse
                if x.HasFundamentalData
                and x.Price > 10
                and x.DollarVolume > 1000000]

    return [x.Symbol for x in filtered]

def FineSelectionFunction(self, fine):
    """
    Filter based on fundamental data.

    Parameters:
        fine: List of FineFundamental objects

    Returns:
        List of Symbol objects
    """
    # Filter by fundamentals
    filtered = [x for x in fine
                if x.MarketCap > 2e9                    # Market cap > $2B
                and x.ValuationRatios.PERatio > 0       # Positive P/E
                and x.ValuationRatios.PERatio < 25      # P/E < 25
                and x.OperationRatios.ROE.Value > 0.10] # ROE > 10%

    # Sort by market cap
    sorted_stocks = sorted(filtered,
                          key=lambda x: x.MarketCap,
                          reverse=True)

    return [x.Symbol for x in sorted_stocks[:50]]
```

### Fine Fundamental Properties

#### Company Reference

| Property | Description |
|----------|-------------|
| `CompanyReference.IndustryTemplateCode` | Industry classification |
| `CompanyReference.CountryId` | Country code |
| `CompanyReference.PrimaryExchangeID` | Exchange |

#### Asset Classification

| Property | Description |
|----------|-------------|
| `AssetClassification.MorningstarSectorCode` | Sector code |
| `AssetClassification.MorningstarIndustryCode` | Industry code |
| `AssetClassification.MorningstarIndustryGroupCode` | Industry group |
| `AssetClassification.StockType` | Stock classification |

#### Morningstar Sector Codes

> **Note**: Morningstar revised their sector, industry group, and industry codes in 2019. LEAN adopted these updated codes. For the latest classification structure, refer to [Morningstar Global Equity Class Structure 2019](http://advisor.morningstar.com/Enterprise/VTC/MorningstarGlobalEquityClassStructure2019v3.pdf).

| Code | Sector | Value |
|------|--------|-------|
| `MorningstarSectorCode.BasicMaterials` | Basic Materials | 101 |
| `MorningstarSectorCode.ConsumerCyclical` | Consumer Cyclical | 102 |
| `MorningstarSectorCode.FinancialServices` | Financial Services | 103 |
| `MorningstarSectorCode.RealEstate` | Real Estate | 104 |
| `MorningstarSectorCode.ConsumerDefensive` | Consumer Defensive | 205 |
| `MorningstarSectorCode.Healthcare` | Healthcare | 206 |
| `MorningstarSectorCode.Utilities` | Utilities | 207 |
| `MorningstarSectorCode.CommunicationServices` | Communication Services | 308 |
| `MorningstarSectorCode.Energy` | Energy | 309 |
| `MorningstarSectorCode.Industrials` | Industrials | 310 |
| `MorningstarSectorCode.Technology` | Technology | 311 |

#### Morningstar Industry Group Codes (Examples)

| Code | Industry Group |
|------|---------------|
| 10101 | Agricultural Inputs |
| 10217 | Auto Manufacturers |
| 10320 | Banks - Regional |
| 20539 | Drug Manufacturers |
| 30888 | Internet Content & Information |
| 31167 | Software - Application |
| 31169 | Semiconductors |

#### Stock Type Classifications

```python
from QuantConnect.Data.Fundamental import StockType

def FineSelectionFunction(self, fine):
    """Filter by stock type."""
    # Common stock types
    common_stocks = [x for x in fine
                     if x.AssetClassification.StockType == StockType.Common]

    # ADRs (American Depositary Receipts)
    adrs = [x for x in fine
            if x.AssetClassification.StockType == StockType.ADR]

    # REITs
    reits = [x for x in fine
             if x.AssetClassification.StockType == StockType.REIT]

    return [x.Symbol for x in common_stocks[:50]]
```

| Stock Type | Description |
|------------|-------------|
| `StockType.Common` | Common shares |
| `StockType.ADR` | American Depositary Receipt |
| `StockType.REIT` | Real Estate Investment Trust |
| `StockType.MLP` | Master Limited Partnership |
| `StockType.PreferredStock` | Preferred shares |
| `StockType.Unit` | Unit (stock + warrant) |

#### Valuation Ratios

| Property | Description |
|----------|-------------|
| `ValuationRatios.PERatio` | Price-to-Earnings |
| `ValuationRatios.PBRatio` | Price-to-Book |
| `ValuationRatios.PSRatio` | Price-to-Sales |
| `ValuationRatios.PEGRatio` | P/E to Growth |
| `ValuationRatios.DividendYield` | Dividend yield |
| `ValuationRatios.EVToEBITDA` | EV/EBITDA |

#### Operation Ratios

| Property | Description |
|----------|-------------|
| `OperationRatios.ROE` | Return on Equity |
| `OperationRatios.ROA` | Return on Assets |
| `OperationRatios.ROIC` | Return on Invested Capital |
| `OperationRatios.GrossMargin` | Gross Margin |
| `OperationRatios.NetMargin` | Net Margin |
| `OperationRatios.CurrentRatio` | Current Ratio |
| `OperationRatios.QuickRatio` | Quick Ratio |
| `OperationRatios.DebtToEquity` | Debt/Equity |

#### Financial Statements

| Property | Description |
|----------|-------------|
| `EarningReports.BasicEPS` | Earnings per Share |
| `EarningReports.DilutedEPS` | Diluted EPS |
| `FinancialStatements.IncomeStatement.TotalRevenue` | Revenue |
| `FinancialStatements.BalanceSheet.TotalAssets` | Total Assets |
| `FinancialStatements.CashFlowStatement.FreeCashFlow` | Free Cash Flow |

### Sector-Based Selection

```python
from QuantConnect.Data.Fundamental import MorningstarSectorCode

def FineSelectionFunction(self, fine):
    """Select stocks from specific sectors."""

    # Technology sector only
    tech = [x for x in fine
            if x.AssetClassification.MorningstarSectorCode == MorningstarSectorCode.Technology]

    # Or multiple sectors
    target_sectors = [
        MorningstarSectorCode.Technology,
        MorningstarSectorCode.Healthcare,
        MorningstarSectorCode.FinancialServices
    ]

    filtered = [x for x in fine
                if x.AssetClassification.MorningstarSectorCode in target_sectors
                and x.MarketCap > 1e9]

    # Equal weight from each sector
    by_sector = {}
    for stock in filtered:
        sector = stock.AssetClassification.MorningstarSectorCode
        if sector not in by_sector:
            by_sector[sector] = []
        by_sector[sector].append(stock)

    # Take top 10 by market cap from each sector
    result = []
    for sector, stocks in by_sector.items():
        sorted_stocks = sorted(stocks, key=lambda x: x.MarketCap, reverse=True)
        result.extend([x.Symbol for x in sorted_stocks[:10]])

    return result
```

## Combined Selection

### Value + Quality Screen

```python
def Initialize(self):
    self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
    self.num_stocks = 30

def CoarseSelectionFunction(self, coarse):
    """Pre-filter for liquidity."""
    filtered = [x for x in coarse
                if x.HasFundamentalData
                and x.Price > 5
                and x.DollarVolume > 2000000]

    # Return more than needed for fine filter
    return [x.Symbol for x in filtered[:500]]

def FineSelectionFunction(self, fine):
    """Value + Quality screen."""

    # Quality filters
    quality = [x for x in fine
               if x.OperationRatios.ROE.Value > 0.15        # ROE > 15%
               and x.OperationRatios.DebtToEquity < 1.0      # D/E < 1
               and x.EarningReports.BasicEPS.Value > 0]      # Positive EPS

    # Value filters
    value = [x for x in quality
             if x.ValuationRatios.PERatio > 0
             and x.ValuationRatios.PERatio < 20              # P/E < 20
             and x.ValuationRatios.PBRatio > 0
             and x.ValuationRatios.PBRatio < 3]              # P/B < 3

    # Score by combined metrics
    scored = []
    for stock in value:
        # Lower P/E and P/B = better value
        value_score = (1 / stock.ValuationRatios.PERatio +
                      1 / stock.ValuationRatios.PBRatio)
        # Higher ROE = better quality
        quality_score = stock.OperationRatios.ROE.Value

        combined_score = value_score * quality_score
        scored.append((stock.Symbol, combined_score))

    # Sort by combined score
    sorted_stocks = sorted(scored, key=lambda x: x[1], reverse=True)

    return [x[0] for x in sorted_stocks[:self.num_stocks]]
```

## ETF Constituents

> **2025 Update**: ETF Constituents universe selection is **free for all users** on QuantConnect Cloud. The dataset provides an excellent source of tradable universes without selection bias. Use cases include index tracking, passive portfolio management, and statistical arbitrage with the base ETF.

### Using ETFConstituentsUniverseSelectionModel (Framework)

```python
from AlgorithmImports import *

def Initialize(self):
    self.SetStartDate(2024, 1, 1)
    self.SetCash(100000)

    # Using framework model (recommended)
    self.AddUniverseSelection(
        ETFConstituentsUniverseSelectionModel("SPY")
    )

    # Or with custom filter function
    self.AddUniverseSelection(
        ETFConstituentsUniverseSelectionModel(
            "QQQ",
            universeFilterFunc=self.ETFConstituentsFilter
        )
    )

def ETFConstituentsFilter(self, constituents):
    """Filter top holdings by weight."""
    # Filter constituents based on weight
    top_holdings = [c for c in constituents if c.Weight >= 0.01]  # >= 1%

    # Sort by weight and take top 30
    sorted_constituents = sorted(
        top_holdings,
        key=lambda x: x.Weight,
        reverse=True
    )

    return [c.Symbol for c in sorted_constituents[:30]]
```

### Classic Universe Selection with ETF

```python
def Initialize(self):
    # Track S&P 500 constituents (classic approach)
    self.AddUniverse(
        self.Universe.ETF("SPY", Market.USA, self.UniverseSettings, self.ETFSelection)
    )

def ETFSelection(self, constituents):
    """
    Filter ETF constituents.

    Parameters:
        constituents: List of ETFConstituentData

    Returns:
        List of Symbol objects
    """
    # Each constituent has weight information
    for constituent in constituents:
        symbol = constituent.Symbol
        weight = constituent.Weight

    # Get top 50 by weight
    sorted_by_weight = sorted(constituents,
                              key=lambda x: x.Weight,
                              reverse=True)

    return [x.Symbol for x in sorted_by_weight[:50]]
```

### Popular ETF Tickers

| ETF Ticker | Description | Use Case |
|------------|-------------|----------|
| **SPY** | S&P 500 | Large-cap US equities |
| **QQQ** | Nasdaq 100 | Tech-heavy portfolio |
| **DIA** | Dow Jones | Blue-chip stocks |
| **IWM** | Russell 2000 | Small-cap equities |
| **EFA** | EAFE Index | International developed markets |

### ETF Constituent Properties

| Property | Description | Use Case |
|----------|-------------|----------|
| `Symbol` | Security symbol | Identify constituent |
| `Weight` | Portfolio weight in ETF | Filter top holdings (e.g., >= 1%) |

### Use Cases

**1. Index Tracking**:

```python
# Replicate SPY with top holdings
self.AddUniverseSelection(ETFConstituentsUniverseSelectionModel("SPY"))
```

**2. Statistical Arbitrage**:

```python
# Trade constituents vs ETF
self.AddEquity("SPY")  # Benchmark
self.AddUniverseSelection(ETFConstituentsUniverseSelectionModel("SPY"))
```

**3. Sector Rotation**:

```python
# Rotate between sector ETFs
for ticker in ["XLK", "XLF", "XLE", "XLV"]:
    self.AddUniverseSelection(ETFConstituentsUniverseSelectionModel(ticker))
```

## Options Universe

### Option Chain Universe

```python
def Initialize(self):
    # Add equity first
    equity = self.AddEquity("AAPL", Resolution.Minute)
    self.underlying = equity.Symbol

    # Add options universe
    option = self.AddOption("AAPL", Resolution.Minute)
    self.option_symbol = option.Symbol

    # Set option filter
    option.SetFilter(self.OptionFilterFunction)

def OptionFilterFunction(self, universe: OptionFilterUniverse):
    """Dynamic option filtering."""
    return universe \
        .Strikes(-5, 5) \
        .Expiration(7, 45) \
        .IncludeWeeklys()
```

### Programmatic Options Selection

```python
def Initialize(self):
    self.SetStartDate(2020, 1, 1)
    self.SetCash(100000)

    # Add underlying
    self.underlying = self.AddEquity("SPY", Resolution.Minute).Symbol

    # Add option with programmatic selection
    option = self.AddOption("SPY", Resolution.Minute)
    option.SetFilter(self.OptionFilter)

def OptionFilter(self, universe):
    """Select specific option contracts."""

    # Get current underlying price (use history if needed)
    price = self.Securities[self.underlying].Price if self.Securities.ContainsKey(self.underlying) else 400

    # Calculate target strikes
    target_strikes = [
        int(price * 0.95),  # 5% OTM put
        int(price),          # ATM
        int(price * 1.05),   # 5% OTM call
    ]

    return universe \
        .Strikes(-10, 10) \
        .Expiration(25, 35) \
        .IncludeWeeklys()
```

## Custom Universes

### Manual Universe

```python
def Initialize(self):
    # Create manual universe from list of symbols
    symbols = [
        Symbol.Create("AAPL", SecurityType.Equity, Market.USA),
        Symbol.Create("MSFT", SecurityType.Equity, Market.USA),
        Symbol.Create("GOOGL", SecurityType.Equity, Market.USA),
        Symbol.Create("AMZN", SecurityType.Equity, Market.USA),
        Symbol.Create("META", SecurityType.Equity, Market.USA),
    ]

    self.SetUniverseSelection(ManualUniverseSelectionModel(symbols))
```

### QC500 Universe Selection Model

The QC500 universe closely mirrors the S&P 500 composition and is a convenient way to trade large-cap US stocks.

```python
from QuantConnect.Algorithm.Framework.Selection import QC500UniverseSelectionModel

def Initialize(self):
    # Use built-in QC500 universe
    self.SetUniverseSelection(QC500UniverseSelectionModel())

    # Alternative: extend QC500 with custom filtering
    # self.SetUniverseSelection(MyQC500Model())

class MyQC500Model(QC500UniverseSelectionModel):
    """Custom extension of QC500 with additional filters."""

    def SelectCoarse(self, algorithm, coarse):
        # Call parent selection
        selected = super().SelectCoarse(algorithm, coarse)
        # Add custom logic here
        return selected

    def SelectFine(self, algorithm, fine):
        # QC500 fine selection available
        selected = super().SelectFine(algorithm, fine)
        # Filter by additional criteria (e.g., ROIC, momentum)
        return selected
```

**QC500 Selection Criteria:**

- **Coarse**: Has fundamental data, positive price, positive volume
- **Fine**: US-based company, listed on NYSE or NASDAQ, IPO > 180 days ago
- **Size**: Filters 1,000 coarse → 500 fine selection
- **Rebalance**: Monthly at month start

### Scheduled Universe Selection

```python
def Initialize(self):
    self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)

    # Only rebalance monthly
    self.Schedule.On(
        self.DateRules.MonthStart(),
        self.TimeRules.AfterMarketOpen("SPY", 30),
        self.TriggerUniverseSelection
    )

    self.rebalance_flag = False

def TriggerUniverseSelection(self):
    """Set flag to trigger universe rebalance."""
    self.rebalance_flag = True

def CoarseSelectionFunction(self, coarse):
    """Only reselect when flag is set."""
    if not self.rebalance_flag:
        return Universe.Unchanged

    self.rebalance_flag = False

    # Normal selection logic
    filtered = [x for x in coarse
                if x.HasFundamentalData
                and x.Price > 10
                and x.DollarVolume > 1000000]

    return [x.Symbol for x in filtered[:100]]
```

## Best Practices

### 1. Handle Securities Changed

```python
def OnSecuritiesChanged(self, changes: SecurityChanges) -> None:
    """Handle universe changes."""

    # Process additions
    for security in changes.AddedSecurities:
        symbol = security.Symbol
        self.Log(f"Added: {symbol}")

        # Initialize any tracking data
        if symbol not in self.indicators:
            self.indicators[symbol] = self.RSI(symbol, 14)

        # Warm up new securities
        history = self.History(symbol, 20, Resolution.Daily)
        for bar in history.itertuples():
            self.indicators[symbol].Update(bar.Index[1], bar.close)

    # Process removals
    for security in changes.RemovedSecurities:
        symbol = security.Symbol
        self.Log(f"Removed: {symbol}")

        # Liquidate position
        if self.Portfolio[symbol].Invested:
            self.Liquidate(symbol)

        # Cleanup tracking data
        if symbol in self.indicators:
            del self.indicators[symbol]
```

### 2. Limit Universe Size

```python
def CoarseSelectionFunction(self, coarse):
    """Limit universe to manageable size."""
    filtered = [x for x in coarse
                if x.HasFundamentalData
                and x.Price > 10
                and x.DollarVolume > 1000000]

    # Sort by liquidity
    sorted_by_volume = sorted(filtered,
                              key=lambda x: x.DollarVolume,
                              reverse=True)

    # Limit to top N
    max_stocks = 100
    return [x.Symbol for x in sorted_by_volume[:max_stocks]]
```

### 3. Use Universe Settings

```python
def Initialize(self):
    # Configure universe behavior
    self.UniverseSettings.Resolution = Resolution.Daily
    self.UniverseSettings.Leverage = 1.0
    self.UniverseSettings.FillForward = True
    self.UniverseSettings.ExtendedMarketHours = False
    self.UniverseSettings.MinimumTimeInUniverse = timedelta(days=1)

    self.AddUniverse(self.CoarseSelectionFunction)
```

### 4. Avoid Look-Ahead Bias

```python
def FineSelectionFunction(self, fine):
    """Use only point-in-time data."""

    # WRONG: This might use future data
    # filtered = [x for x in fine if x.EarningReports.BasicEPS.Value > 0]

    # RIGHT: Use data with appropriate lag
    filtered = [x for x in fine
                if x.EarningReports.BasicEPS.ThreeMonths > 0]  # Lagged data

    return [x.Symbol for x in filtered]
```

### 5. Handle Empty Universes

```python
def CoarseSelectionFunction(self, coarse):
    """Handle case when no stocks meet criteria."""
    filtered = [x for x in coarse
                if x.HasFundamentalData
                and x.Price > 10
                and x.DollarVolume > 10000000]  # Strict filter

    if len(filtered) < 10:
        self.Log("Warning: Less than 10 stocks meet criteria")
        # Could relax criteria or return previous universe
        return Universe.Unchanged

    return [x.Symbol for x in filtered[:50]]
```

---

*Last Updated: November 2025*
