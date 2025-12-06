# Options Trading on QuantConnect

Comprehensive guide to options trading, Greeks, pricing models, and option chain management in QuantConnect.

## Table of Contents

- [Options Basics](#options-basics)
- [Adding Options](#adding-options)
- [Options Data Limitations](#options-data-limitations)
- [Option Chain Filtering](#option-chain-filtering)
- [Accessing Greeks](#accessing-greeks)
- [Implied Volatility Analysis](#implied-volatility-analysis)
- [Pricing Models](#pricing-models)
- [Options Strategies](#options-strategies)
- [Multi-Leg Orders](#multi-leg-orders)
- [Exercise and Assignment](#exercise-and-assignment)
- [Memory Management](#memory-management)
- [Best Practices](#best-practices)

## Options Basics

### Option Contract Properties

| Property | Description |
|----------|-------------|
| `Symbol` | Unique identifier for the contract |
| `UnderlyingSymbol` | Symbol of the underlying asset |
| `Strike` | Strike price |
| `Expiry` | Expiration date |
| `Right` | Call or Put (`OptionRight.Call`, `OptionRight.Put`) |
| `BidPrice` | Current bid |
| `AskPrice` | Current ask |
| `LastPrice` | Last traded price |
| `OpenInterest` | Open interest |
| `ImpliedVolatility` | Implied volatility |

## Adding Options

### Basic Option Subscription

```python
def Initialize(self):
    # Add equity (required for options)
    self.equity = self.AddEquity("SPY", Resolution.Minute)
    self.underlying = self.equity.Symbol

    # Add option
    option = self.AddOption("SPY", Resolution.Minute)
    self.option_symbol = option.Symbol

    # Set filter for option chain
    option.SetFilter(self.OptionFilterFunction)

    # Or use simple filter
    option.SetFilter(-5, 5, 0, 30)  # ±5 strikes, 0-30 DTE

def OptionFilterFunction(self, universe: OptionFilterUniverse) -> OptionFilterUniverse:
    """Custom option filter function."""
    return universe \
        .Strikes(-10, 10) \
        .Expiration(7, 60) \
        .IncludeWeeklys() \
        .OnlyApplyFilterAtMarketOpen()
```

### Filter Methods

| Method | Description |
|--------|-------------|
| `.Strikes(min, max)` | Filter by strike distance from ATM |
| `.Expiration(minDays, maxDays)` | Filter by days to expiration |
| `.IncludeWeeklys()` | Include weekly options |
| `.OnlyApplyFilterAtMarketOpen()` | Only filter at market open |
| `.CallsOnly()` | Only call options |
| `.PutsOnly()` | Only put options |

### Greeks-Based Universe Filtering (NEW)

**Since PR #6720, you can filter options by Greeks and IV before they reach your algorithm**, significantly reducing data processing and memory usage.

```python
def OptionFilterFunction(self, universe: OptionFilterUniverse) -> OptionFilterUniverse:
    """Filter options using Greeks and IV (available immediately, no warmup)."""
    return (universe
        .Strikes(-10, 10)                    # Strike range
        .Expiration(30, 180)                 # DTE range
        .Delta(0.25, 0.35)                   # Delta range (or shortcut: .d())
        .ImpliedVolatility(0.20, None)       # Min IV threshold (or: .iv())
        .Gamma(0.01, 0.5)                    # Gamma range (or: .g())
        .Theta(-1.0, -0.1)                   # Theta range (or: .t())
        .Vega(0.5, 1.5)                      # Vega range (or: .v())
        .OpenInterest(100, None)             # Min open interest (or: .oi())
        .IncludeWeeklys())
```

**Available Greeks Filter Methods:**

| Method | Shortcut | Description |
|--------|----------|-------------|
| `.Delta(min, max)` | `.d(min, max)` | Filter by delta |
| `.Gamma(min, max)` | `.g(min, max)` | Filter by gamma |
| `.Vega(min, max)` | `.v(min, max)` | Filter by vega |
| `.Theta(min, max)` | `.t(min, max)` | Filter by theta |
| `.Rho(min, max)` | `.r(min, max)` | Filter by rho |
| `.ImpliedVolatility(min, max)` | `.iv(min, max)` | Filter by IV |
| `.OpenInterest(min, max)` | `.oi(min, max)` | Filter by open interest |

**Benefits:**

- **Reduces data volume**: Pre-filter before detailed analysis in scanners
- **Faster backtests**: Less data to process in OnData
- **Lower memory**: Fewer contracts loaded into memory
- **No warmup needed**: Greeks available immediately (PR #6720)

**Example - Scanner-Optimized Filter:**

```python
def OptionFilterFunction(self, universe: OptionFilterUniverse) -> OptionFilterUniverse:
    """Pre-filter for underpriced options scanner."""
    return (universe
        .d(0.25, 0.35)          # Delta range for scanner
        .iv(0.20, None)         # Minimum IV threshold
        .Expiration(30, 180)    # Optimal expiration range
        .oi(100, None)          # Liquidity filter
        .IncludeWeeklys())
```

## Options Data Limitations

Understanding QuantConnect's options data infrastructure is critical for setting realistic expectations.

### Data Resolution

> **Critical Limitation (2025)**: QuantConnect officially supports options data at **Minute, Hour, and Daily** resolutions only. Second resolution technically exists but is **not officially supported** - QuantConnect will not provide support tickets for issues encountered with second resolution options data. Tick resolution is not available.
>
> **Upcoming**: Plans for a native **OPRA feed** built into QuantConnect were mentioned for early 2025. Currently, for higher resolution options data (tick/second), use external providers like **Interactive Brokers** or **Polygon**.

| Resolution | Available | Notes |
|------------|-----------|-------|
| **Tick** | No | Not supported for options |
| **Second** | ⚠️ Unsupported | Technically possible but no official support, slow in Python |
| **Minute** | ✅ Yes | Primary trading resolution (officially supported) |
| **Hour** | ✅ Yes | Aggregated from minute data |
| **Daily** | ✅ Yes | End-of-day summaries |

```python
def Initialize(self):
    # CORRECT: Minute resolution (officially supported)
    option = self.AddOption("SPY", Resolution.Minute)

    # NOT RECOMMENDED: Second resolution (unsupported, may be slow)
    # option = self.AddOption("SPY", Resolution.Second)  # No support!

    # INCORRECT: Tick resolution (not available)
    # option = self.AddOption("SPY", Resolution.Tick)  # Will fail!
```

### Data Source: AlgoSeek OPRA Feed

QuantConnect sources options data from **AlgoSeek**, which monitors the **Options Price Reporting Authority (OPRA)** data feed.

| Attribute | Details |
|-----------|---------|
| **Source** | AlgoSeek / OPRA consolidated feed |
| **Coverage Start** | January 2012 |
| **Underlying Symbols** | ~4,000 symbols |
| **Index Options** | 7 indices (SPX, VIX, etc.) |
| **Data Type** | Trades, Quotes, Open Interest |
| **Timestamp Precision** | Millisecond |
| **Timezone** | Eastern Time (New York) |

### Market Depth Limitations

> **Important**: QuantConnect provides **NBBO (Level 1) data only**. Level 2 market depth (order book) is **not available** for options.

| Data Level | Available | Description |
|------------|-----------|-------------|
| **Level 1 (NBBO)** | Yes | Best bid/ask prices and sizes |
| **Level 2 (Depth)** | No | Full order book not available |
| **Time & Sales** | Yes | Via minute aggregation |

**Implications for HFT Strategies**:

- Cannot see order book depth beyond best bid/ask
- Cannot detect large resting orders
- Cannot implement order flow analysis
- Market making strategies severely limited

```python
def OnData(self, data):
    chain = data.OptionChains.get(self.option_symbol)
    if not chain:
        return

    for contract in chain:
        # Available (Level 1)
        bid = contract.BidPrice
        ask = contract.AskPrice
        bid_size = contract.BidSize
        ask_size = contract.AskSize

        # NOT available (would need Level 2)
        # order_book_depth = contract.OrderBook  # Does not exist!
```

### Greeks Source

> **Note**: Greeks displayed in QuantConnect are **calculated by QuantConnect's pricing models**, not provided directly from exchanges. The Greeks values are comparable to Interactive Brokers data (e.g., similar IV, Delta values). However, American put options with discrete dividends before expiry cannot be accurately priced by existing QuantLib models.

```python
def OnData(self, data):
    chain = data.OptionChains.get(self.option_symbol)
    if not chain:
        return

    for contract in chain:
        # Greeks are CALCULATED, not market data
        # Accuracy depends on pricing model selection
        delta = contract.Greeks.Delta
        gamma = contract.Greeks.Gamma
        theta = contract.Greeks.Theta
        vega = contract.Greeks.Vega

        # IV is also calculated (model-derived)
        iv = contract.ImpliedVolatility
```

**Pricing Model Impact on Greeks**:

| Model | Greeks Accuracy | Speed |
|-------|-----------------|-------|
| Black-Scholes | Good for European | Fast |
| Binomial (CRR) | Better for American | Medium |
| Crank-Nicolson FD | Best accuracy | Slow |

### Data Subscription Tiers

QuantConnect options data requires appropriate subscription tier:

| Tier | Options Data | Notes |
|------|--------------|-------|
| **Free** | Limited backtest | Basic access, 8 hours/month |
| **Quant Researcher (~$10/mo)** | Full backtest | Individual researchers |
| **Organization (~$20/mo)** | Full + Live | Most traders |
| **Trading Firm** | Enterprise | Custom data feeds |

> **2025 Update**: QuantConnect has plans for a native OPRA feed built into the platform. For now, Polygon and Interactive Brokers are alternatives for live OPRA data access.

### Latency Considerations

For live trading with Schwab brokerage:

| Operation | Expected Latency |
|-----------|------------------|
| **Data Feed** | 100-500ms |
| **Order Submission** | 100-500ms |
| **Order Fill** | 500ms - 5s |
| **Total Round Trip** | 1-6 seconds |

> **HFT Reality Check**: Sub-second options trading strategies are **not feasible** on QuantConnect due to data resolution and execution latency constraints.

### Python vs C# Performance

For options backtesting, language choice significantly impacts performance:

| Language | Relative Speed | Best For |
|----------|----------------|----------|
| **C#** | 1x (baseline) | Production, large backtests |
| **Python** | 10-50x slower | Prototyping, research |

```python
# Python is convenient but slower for options
# Consider C# for production algorithms with heavy options processing

# Python example (slower)
class PythonOptionsAlgo(QCAlgorithm):
    def OnData(self, data):
        chain = data.OptionChains.get(self.option_symbol)
        # Processing here is 10-50x slower than C#
```

## Option Chain Filtering

### Accessing the Option Chain

```python
def OnData(self, data: Slice) -> None:
    # Get option chain
    chain = data.OptionChains.get(self.option_symbol)
    if not chain:
        return

    # Get underlying price
    underlying_price = self.Securities[self.underlying].Price

    # Filter for specific contracts
    calls = [x for x in chain if x.Right == OptionRight.Call]
    puts = [x for x in chain if x.Right == OptionRight.Put]

    # Find ATM call
    atm_call = min(calls, key=lambda x: abs(x.Strike - underlying_price))

    # Find OTM puts (strike below current price)
    otm_puts = [x for x in puts if x.Strike < underlying_price]

    # Filter by Greeks
    low_delta_calls = [x for x in calls if 0.25 <= abs(x.Greeks.Delta) <= 0.35]
    high_theta_options = [x for x in chain if x.Greeks.Theta < -0.05]
```

### Advanced Filtering Examples

```python
def FindIdealContract(self, chain, underlying_price):
    """Find contract matching specific criteria."""

    # Filter criteria
    min_dte = 30
    max_dte = 60
    target_delta = 0.30
    min_open_interest = 100
    max_spread_pct = 0.10  # 10% bid-ask spread

    candidates = []

    for contract in chain:
        # Days to expiration
        dte = (contract.Expiry - self.Time).days
        if not (min_dte <= dte <= max_dte):
            continue

        # Delta filter
        delta = abs(contract.Greeks.Delta)
        if not (target_delta - 0.05 <= delta <= target_delta + 0.05):
            continue

        # Liquidity filters
        if contract.OpenInterest < min_open_interest:
            continue

        # Spread filter
        if contract.BidPrice > 0:
            spread_pct = (contract.AskPrice - contract.BidPrice) / contract.BidPrice
            if spread_pct > max_spread_pct:
                continue

        candidates.append(contract)

    # Sort by volume
    candidates.sort(key=lambda x: x.Volume, reverse=True)

    return candidates[0] if candidates else None
```

## Accessing Greeks

### 🔥 **CRITICAL UPDATE: PR #6720 - Greeks Now Use Implied Volatility**

**As of November 2022 (LEAN PR #6720), Greeks calculation was fundamentally changed:**

- **Old Behavior**: Greeks used historical volatility (unreliable, didn't match broker values)
- **New Behavior**: Greeks now use **implied volatility** from option prices
- **Impact**: Values now **match Interactive Brokers and major brokerages exactly**
- **NO WARMUP REQUIRED**: Greeks are available immediately upon data arrival

**Key Improvements:**

- ✅ Accurate Greeks matching real broker values
- ✅ No warmup period needed (immediate availability)
- ✅ Default models: Black-Scholes (European), Bjerksund-Stensland (American)
- ✅ Greeks calculated using IV from option market prices

### Available Greeks

```python
def OnData(self, data: Slice) -> None:
    chain = data.OptionChains.get(self.option_symbol)
    if not chain:
        return

    # Greeks are available IMMEDIATELY (no warmup needed since PR #6720)
    for contract in chain:
        greeks = contract.Greeks

        # Delta: Rate of change of option price vs underlying
        # Range: -1 to 1 (calls: 0 to 1, puts: -1 to 0)
        delta = greeks.Delta

        # Gamma: Rate of change of delta vs underlying
        # Higher for ATM options near expiration
        gamma = greeks.Gamma

        # Theta: Time decay (total theta over option lifetime)
        # Typically negative (options lose value over time)
        theta = greeks.Theta

        # ThetaPerDay: Daily theta decay - USE THIS for IB compatibility
        # Matches Interactive Brokers theta values
        theta_per_day = greeks.ThetaPerDay  # RECOMMENDED

        # Vega: Sensitivity to implied volatility
        # Higher for longer-dated options
        vega = greeks.Vega

        # Rho: Sensitivity to interest rates
        rho = greeks.Rho

        # Implied Volatility (used to calculate Greeks since PR #6720)
        iv = contract.ImpliedVolatility

        # Theoretical price from pricing model
        theoretical_price = greeks.TheoreticalPrice
```

### **Theta vs ThetaPerDay - IMPORTANT**

| Property | Description | Use Case |
|----------|-------------|----------|
| `greeks.Theta` | Total theta over option lifetime | Historical calculation |
| `greeks.ThetaPerDay` | **Daily theta decay** | **Use this for IB compatibility** |

**Recommendation**: Always use `ThetaPerDay` instead of `Theta` for accurate daily time decay matching Interactive Brokers.

### Greeks Interpretation Guide

| Greek | Meaning | Trading Use |
|-------|---------|-------------|
| **Delta** | $ change per $1 underlying move | Position sizing, hedge ratio |
| **Gamma** | Delta change per $1 underlying move | Risk of delta changes |
| **Theta** | $ lost per day (time decay) | Income strategies |
| **Vega** | $ change per 1% IV change | Volatility trading |
| **Rho** | $ change per 1% rate change | Usually minor |

### Portfolio Greeks

```python
def CalculatePortfolioGreeks(self):
    """Calculate aggregate portfolio Greeks."""
    total_delta = 0
    total_gamma = 0
    total_theta = 0
    total_vega = 0

    for symbol, holding in self.Portfolio.items():
        if not holding.Invested:
            continue

        security = self.Securities[symbol]

        if security.Type == SecurityType.Option:
            # Get current Greeks
            option = security
            quantity = holding.Quantity
            multiplier = option.ContractMultiplier  # Usually 100

            greeks = option.Greeks
            total_delta += greeks.Delta * quantity * multiplier
            total_gamma += greeks.Gamma * quantity * multiplier
            total_theta += greeks.Theta * quantity * multiplier
            total_vega += greeks.Vega * quantity * multiplier

        elif security.Type == SecurityType.Equity:
            # Stock has delta of 1
            total_delta += holding.Quantity

    return {
        'delta': total_delta,
        'gamma': total_gamma,
        'theta': total_theta,
        'vega': total_vega
    }
```

## Implied Volatility Analysis

### IV Rank and IV Percentile

IV Rank and IV Percentile help determine if current implied volatility is high or low relative to historical levels.

```python
class IVAnalysis(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        self.equity = self.AddEquity("SPY", Resolution.Daily)
        self.option = self.AddOption("SPY", Resolution.Minute)
        self.option.SetFilter(-5, 5, 30, 60)

        # Track IV history
        self.iv_history = []
        self.iv_lookback = 252  # 1 year of trading days

    def OnData(self, data):
        chain = data.OptionChains.get(self.option.Symbol)
        if not chain:
            return

        # Get ATM option IV as proxy for underlying IV
        underlying_price = self.Securities[self.equity.Symbol].Price
        atm_options = sorted(chain, key=lambda x: abs(x.Strike - underlying_price))

        if not atm_options:
            return

        current_iv = atm_options[0].ImpliedVolatility
        self.iv_history.append(current_iv)

        # Keep only lookback period
        if len(self.iv_history) > self.iv_lookback:
            self.iv_history.pop(0)

        if len(self.iv_history) < 30:  # Need minimum history
            return

        # Calculate IV Rank
        iv_rank = self.CalculateIVRank(current_iv)

        # Calculate IV Percentile
        iv_percentile = self.CalculateIVPercentile(current_iv)

        self.Log(f"Current IV: {current_iv:.2%}, IV Rank: {iv_rank:.1%}, IV Percentile: {iv_percentile:.1%}")

    def CalculateIVRank(self, current_iv):
        """
        IV Rank = (Current IV - 52-week Low) / (52-week High - 52-week Low)
        Range: 0% to 100%
        """
        iv_min = min(self.iv_history)
        iv_max = max(self.iv_history)

        if iv_max == iv_min:
            return 0.5

        return (current_iv - iv_min) / (iv_max - iv_min)

    def CalculateIVPercentile(self, current_iv):
        """
        IV Percentile = % of days IV was lower than current
        More accurate for skewed distributions
        """
        lower_count = sum(1 for iv in self.iv_history if iv < current_iv)
        return lower_count / len(self.iv_history)
```

### IV Rank vs IV Percentile

| Metric | Calculation | Best Use |
|--------|-------------|----------|
| **IV Rank** | Position between high/low | Quick assessment |
| **IV Percentile** | % of days below current | More statistically robust |

**Trading Implications:**

- **High IV (>50%)**: Consider selling premium (iron condors, credit spreads)
- **Low IV (<25%)**: Consider buying premium (straddles, strangles)
- **Mean-reverting**: IV tends to revert to historical averages

### Volatility Smile and Skew

The volatility smile shows how IV varies across strike prices:

```python
def AnalyzeVolatilitySmile(self, chain, underlying_price):
    """
    Analyze the volatility smile/skew pattern.

    Typical patterns:
    - Smile: Higher IV at both OTM puts and calls (common in forex)
    - Skew: Higher IV for OTM puts (common in equities - "fear premium")
    - Smirk: Asymmetric smile favoring one side
    """
    calls = sorted([x for x in chain if x.Right == OptionRight.Call],
                   key=lambda x: x.Strike)
    puts = sorted([x for x in chain if x.Right == OptionRight.Put],
                  key=lambda x: x.Strike)

    # Calculate moneyness and IV for each strike
    smile_data = []

    for contract in chain:
        moneyness = contract.Strike / underlying_price
        iv = contract.ImpliedVolatility

        if iv > 0:
            smile_data.append({
                'strike': contract.Strike,
                'moneyness': moneyness,
                'iv': iv,
                'right': 'call' if contract.Right == OptionRight.Call else 'put'
            })

    # Sort by moneyness
    smile_data.sort(key=lambda x: x['moneyness'])

    # Calculate skew metrics
    atm_iv = self.GetATMImpliedVolatility(chain, underlying_price)
    otm_put_iv = self.GetOTMPutIV(puts, underlying_price, delta_target=-0.25)
    otm_call_iv = self.GetOTMCallIV(calls, underlying_price, delta_target=0.25)

    # Skew = OTM Put IV - OTM Call IV (positive = put skew)
    skew = otm_put_iv - otm_call_iv if otm_put_iv and otm_call_iv else None

    return {
        'atm_iv': atm_iv,
        'otm_put_iv': otm_put_iv,
        'otm_call_iv': otm_call_iv,
        'skew': skew,
        'smile_data': smile_data
    }

def GetATMImpliedVolatility(self, chain, underlying_price):
    """Get IV of nearest ATM option."""
    atm = min(chain, key=lambda x: abs(x.Strike - underlying_price))
    return atm.ImpliedVolatility

def GetOTMPutIV(self, puts, underlying_price, delta_target=-0.25):
    """Get IV of OTM put at target delta."""
    candidates = [p for p in puts if p.Strike < underlying_price and p.Greeks.Delta]
    if not candidates:
        return None
    target = min(candidates, key=lambda x: abs(x.Greeks.Delta - delta_target))
    return target.ImpliedVolatility

def GetOTMCallIV(self, calls, underlying_price, delta_target=0.25):
    """Get IV of OTM call at target delta."""
    candidates = [c for c in calls if c.Strike > underlying_price and c.Greeks.Delta]
    if not candidates:
        return None
    target = min(candidates, key=lambda x: abs(x.Greeks.Delta - delta_target))
    return target.ImpliedVolatility
```

### Volatility Surface

The volatility surface extends the smile across multiple expirations:

```python
def BuildVolatilitySurface(self, chain):
    """
    Build a volatility surface (strike x expiration).

    Returns a dictionary of IV values indexed by (strike, expiry).
    """
    surface = {}
    expirations = set()
    strikes = set()

    for contract in chain:
        if contract.ImpliedVolatility <= 0:
            continue

        strike = contract.Strike
        expiry = contract.Expiry
        iv = contract.ImpliedVolatility

        surface[(strike, expiry)] = iv
        expirations.add(expiry)
        strikes.add(strike)

    return {
        'surface': surface,
        'expirations': sorted(expirations),
        'strikes': sorted(strikes)
    }

def InterpolateIV(self, surface_data, target_strike, target_expiry):
    """
    Interpolate IV for a specific strike/expiry not in chain.
    Uses simple linear interpolation.
    """
    import numpy as np
    from scipy import interpolate

    surface = surface_data['surface']
    strikes = surface_data['strikes']
    expirations = surface_data['expirations']

    # Build grid
    iv_grid = np.zeros((len(strikes), len(expirations)))
    for i, strike in enumerate(strikes):
        for j, expiry in enumerate(expirations):
            iv_grid[i, j] = surface.get((strike, expiry), np.nan)

    # Create interpolator
    strike_indices = np.arange(len(strikes))
    expiry_indices = np.arange(len(expirations))

    f = interpolate.interp2d(expiry_indices, strike_indices, iv_grid, kind='linear')

    # Find indices for target
    target_strike_idx = np.interp(target_strike, strikes, strike_indices)
    target_expiry_days = [(e - self.Time).days for e in expirations]
    target_days = (target_expiry - self.Time).days
    target_expiry_idx = np.interp(target_days, target_expiry_days, expiry_indices)

    return f(target_expiry_idx, target_strike_idx)[0]
```

### Term Structure Analysis

Analyze how IV changes across expirations:

```python
def AnalyzeTermStructure(self, chain, underlying_price):
    """
    Analyze IV term structure (IV vs time to expiration).

    Patterns:
    - Contango: Near-term IV < Far-term IV (normal)
    - Backwardation: Near-term IV > Far-term IV (event/fear)
    """
    # Group by expiration
    by_expiry = {}
    for contract in chain:
        expiry = contract.Expiry
        if expiry not in by_expiry:
            by_expiry[expiry] = []
        by_expiry[expiry].append(contract)

    # Get ATM IV for each expiration
    term_structure = []
    for expiry, contracts in sorted(by_expiry.items()):
        atm = min(contracts, key=lambda x: abs(x.Strike - underlying_price))
        dte = (expiry - self.Time).days

        term_structure.append({
            'expiry': expiry,
            'dte': dte,
            'atm_iv': atm.ImpliedVolatility
        })

    # Determine structure type
    if len(term_structure) >= 2:
        near_iv = term_structure[0]['atm_iv']
        far_iv = term_structure[-1]['atm_iv']

        if near_iv < far_iv:
            structure_type = 'contango'
        else:
            structure_type = 'backwardation'
    else:
        structure_type = 'unknown'

    return {
        'term_structure': term_structure,
        'structure_type': structure_type
    }
```

### Trading Based on IV Analysis

```python
def OnData(self, data):
    chain = data.OptionChains.get(self.option.Symbol)
    if not chain:
        return

    underlying_price = self.Securities[self.underlying].Price

    # Get IV metrics
    current_iv = self.GetATMImpliedVolatility(chain, underlying_price)
    iv_rank = self.CalculateIVRank(current_iv)
    smile_analysis = self.AnalyzeVolatilitySmile(chain, underlying_price)
    term_structure = self.AnalyzeTermStructure(chain, underlying_price)

    # High IV Rank - Sell premium
    if iv_rank > 0.5 and not self.Portfolio.Invested:
        self.Log(f"High IV Rank ({iv_rank:.1%}) - Selling iron condor")
        self.ExecuteIronCondor(chain, underlying_price)

    # Low IV Rank - Buy premium
    elif iv_rank < 0.25 and not self.Portfolio.Invested:
        self.Log(f"Low IV Rank ({iv_rank:.1%}) - Buying straddle")
        self.ExecuteStraddle(chain, underlying_price)

    # High skew - Sell put spreads (collect fear premium)
    if smile_analysis['skew'] and smile_analysis['skew'] > 0.05:
        self.Log(f"High put skew ({smile_analysis['skew']:.2%}) - Put spreads attractive")

    # Backwardation - Near-term event expected
    if term_structure['structure_type'] == 'backwardation':
        self.Log("IV Backwardation detected - potential event risk")
```

## Pricing Models

> **2025 Update**: The default pricing model is **BinomialCoxRossRubinstein** for American options and **BlackScholes** for European options. All Greeks and theoretical prices are calculated by the pricing model, but **BlackScholes currently only populates underlying/implied volatility** (not Delta, Gamma, Vega, Rho, Theta). Use **BjerksundStensland** or **CrankNicolsonFD** for full Greeks support.
>
> **Known Limitation**: American put options with discrete dividends before expiry cannot be priced with closed-form solutions. In these cases, IV and Greeks may not be calculated.

### Setting Pricing Models

```python
def Initialize(self):
    option = self.AddOption("SPY", Resolution.Minute)

    # Black-Scholes-Merton (default for European)
    # WARNING: Only populates volatility, NOT Greeks
    option.PriceModel = OptionPriceModels.BlackScholes()

    # Binomial model (better for American options)
    option.PriceModel = OptionPriceModels.BinomialCoxRossRubinstein()

    # Crank-Nicolson finite difference (full Greeks support)
    option.PriceModel = OptionPriceModels.CrankNicolsonFD()

    # Barone-Adesi-Whaley (American approximation, 1987)
    option.PriceModel = OptionPriceModels.BaroneAdesiWhaley()

    # Bjerksund-Stensland (American, 1993, RECOMMENDED for full Greeks)
    option.PriceModel = OptionPriceModels.BjerksundStensland()
```

### Pricing Model Comparison

| Model | Best For | Speed | Accuracy | Greeks Support |
|-------|----------|-------|----------|----------------|
| Black-Scholes | European options | Fast | Good | **No** (volatility only) |
| Binomial | American options | Medium | Very Good | Yes |
| Crank-Nicolson FD | Complex payoffs, accurate Greeks | Slow | Excellent | **Yes** (recommended) |
| Barone-Adesi-Whaley | American approx (1987) | Fast | Good | Yes |
| Bjerksund-Stensland | American options (1993) | Fast | Very Good | **Yes** (recommended) |

> **Recommendation**: For production algorithms requiring Greeks, use **`OptionPriceModels.BjerksundStensland()`** or **`OptionPriceModels.CrankNicolsonFD()`**. Avoid BlackScholes unless you only need implied volatility.

### IV Models

```python
def Initialize(self):
    option = self.AddOption("SPY", Resolution.Minute)

    # Set IV smoothing model
    option.SetOptionAssignmentModel(
        DefaultOptionAssignmentModel()
    )

    # Custom IV calculation
    option.VolatilityModel = StandardDeviationOfReturnsVolatilityModel(30)
```

## Options Strategies

### Single Leg Strategies

```python
def BuyCall(self, chain, underlying_price):
    """Buy a call option (bullish)."""
    calls = [x for x in chain if x.Right == OptionRight.Call]
    otm_calls = [x for x in calls if x.Strike > underlying_price]

    if not otm_calls:
        return

    # Select first OTM call
    contract = min(otm_calls, key=lambda x: x.Strike)
    self.MarketOrder(contract.Symbol, 1)

def BuyPut(self, chain, underlying_price):
    """Buy a put option (bearish/protective)."""
    puts = [x for x in chain if x.Right == OptionRight.Put]
    otm_puts = [x for x in puts if x.Strike < underlying_price]

    if not otm_puts:
        return

    # Select first OTM put
    contract = max(otm_puts, key=lambda x: x.Strike)
    self.MarketOrder(contract.Symbol, 1)
```

### Vertical Spreads

```python
def BullCallSpread(self, chain, underlying_price):
    """Bull call spread (debit spread, bullish)."""
    calls = [x for x in chain if x.Right == OptionRight.Call]
    calls = sorted(calls, key=lambda x: x.Strike)

    # Find ATM and OTM calls
    atm_calls = [x for x in calls if x.Strike >= underlying_price]
    if len(atm_calls) < 2:
        return

    long_call = atm_calls[0]   # Buy lower strike
    short_call = atm_calls[1]  # Sell higher strike

    # Execute spread
    self.MarketOrder(long_call.Symbol, 1)   # Buy
    self.MarketOrder(short_call.Symbol, -1)  # Sell

def BearPutSpread(self, chain, underlying_price):
    """Bear put spread (debit spread, bearish)."""
    puts = [x for x in chain if x.Right == OptionRight.Put]
    puts = sorted(puts, key=lambda x: x.Strike, reverse=True)

    # Find ATM and OTM puts
    atm_puts = [x for x in puts if x.Strike <= underlying_price]
    if len(atm_puts) < 2:
        return

    long_put = atm_puts[0]   # Buy higher strike
    short_put = atm_puts[1]  # Sell lower strike

    # Execute spread
    self.MarketOrder(long_put.Symbol, 1)   # Buy
    self.MarketOrder(short_put.Symbol, -1)  # Sell
```

### Iron Condor

```python
def IronCondor(self, chain, underlying_price):
    """
    Iron condor: Sell OTM strangle, buy further OTM strangle.
    Profit from low volatility / range-bound market.
    """
    calls = sorted([x for x in chain if x.Right == OptionRight.Call],
                   key=lambda x: x.Strike)
    puts = sorted([x for x in chain if x.Right == OptionRight.Put],
                  key=lambda x: x.Strike, reverse=True)

    # Find strikes relative to underlying
    otm_calls = [x for x in calls if x.Strike > underlying_price]
    otm_puts = [x for x in puts if x.Strike < underlying_price]

    if len(otm_calls) < 2 or len(otm_puts) < 2:
        return

    # Short strangle (inner legs)
    short_call = otm_calls[0]
    short_put = otm_puts[0]

    # Long strangle (outer legs - protection)
    long_call = otm_calls[1]
    long_put = otm_puts[1]

    # Execute all four legs
    self.MarketOrder(short_call.Symbol, -1)  # Sell call
    self.MarketOrder(long_call.Symbol, 1)    # Buy call (protection)
    self.MarketOrder(short_put.Symbol, -1)   # Sell put
    self.MarketOrder(long_put.Symbol, 1)     # Buy put (protection)
```

### Butterfly Spread

```python
def ButterflySpread(self, chain, underlying_price):
    """
    Long butterfly: Buy 1 ITM, sell 2 ATM, buy 1 OTM.
    Profit from low volatility, price staying near ATM strike.
    """
    calls = sorted([x for x in chain if x.Right == OptionRight.Call],
                   key=lambda x: x.Strike)

    # Find ATM strike
    atm_call = min(calls, key=lambda x: abs(x.Strike - underlying_price))
    atm_strike = atm_call.Strike

    # Find equal-distance strikes
    strike_width = 5  # Adjust based on underlying

    lower_calls = [x for x in calls if x.Strike == atm_strike - strike_width]
    atm_calls = [x for x in calls if x.Strike == atm_strike]
    upper_calls = [x for x in calls if x.Strike == atm_strike + strike_width]

    if not (lower_calls and atm_calls and upper_calls):
        return

    # Execute butterfly
    self.MarketOrder(lower_calls[0].Symbol, 1)   # Buy 1 lower
    self.MarketOrder(atm_calls[0].Symbol, -2)    # Sell 2 ATM
    self.MarketOrder(upper_calls[0].Symbol, 1)   # Buy 1 upper
```

### Covered Call

```python
def CoveredCall(self, chain, underlying_price):
    """
    Covered call: Own stock + sell OTM call.
    Income strategy, limits upside.
    """
    # First, ensure we own the underlying
    if not self.Portfolio[self.underlying].Invested:
        self.MarketOrder(self.underlying, 100)  # Buy 100 shares

    calls = [x for x in chain if x.Right == OptionRight.Call]
    otm_calls = [x for x in calls if x.Strike > underlying_price * 1.05]

    if not otm_calls:
        return

    # Select call ~5% OTM with 30-45 DTE
    suitable = [x for x in otm_calls
                if 30 <= (x.Expiry - self.Time).days <= 45]

    if not suitable:
        return

    contract = suitable[0]
    self.MarketOrder(contract.Symbol, -1)  # Sell call
```

## Multi-Leg Orders

QuantConnect supports combo orders for multi-leg options strategies, which can improve execution and reduce slippage.

### OptionStrategies Factory Methods (RECOMMENDED)

**QuantConnect provides 37+ pre-built strategy constructors** that automatically create multi-leg positions with proper grouping and margin calculation.

**Why Use Factory Methods:**

- ✅ Automatic position grouping for accurate margin calculations
- ✅ Cleaner code (no manual `Leg.Create()` calls)
- ✅ Automatic strategy recognition by LEAN
- ✅ Compatible with `Buy()` and `Sell()` methods
- ✅ Single command to enter/exit entire positions

**Common Strategy Factory Methods:**

| Strategy | Method | Example |
|----------|--------|---------|
| Butterfly Call | `OptionStrategies.butterfly_call()` | Buy-Sell-Sell-Buy calls |
| Butterfly Put | `OptionStrategies.butterfly_put()` | Buy-Sell-Sell-Buy puts |
| Iron Butterfly | `OptionStrategies.iron_butterfly()` | Combo of call + put butterflies |
| Iron Condor | `OptionStrategies.iron_condor()` | Sell OTM put/call spreads |
| Bull Call Spread | `OptionStrategies.bull_call_spread()` | Buy lower, sell higher call |
| Bear Put Spread | `OptionStrategies.bear_put_spread()` | Buy higher, sell lower put |
| Straddle | `OptionStrategies.straddle()` | Buy call + put same strike |
| Strangle | `OptionStrategies.strangle()` | Buy OTM call + OTM put |
| Covered Call | `OptionStrategies.covered_call()` | Long stock + sell call |
| Protective Put | `OptionStrategies.protective_put()` | Long stock + buy put |

**37+ total strategies available** - see [OptionStrategies.cs](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/Option/OptionStrategies.cs)

**Example - Butterfly Call:**

```python
def OnData(self, data: Slice) -> None:
    chain = data.OptionChains.get(self.option_symbol)
    if not chain:
        return

    # Find strikes
    expiry = min([x.Expiry for x in chain])
    calls = sorted([c for c in chain if c.Expiry == expiry and c.Right == OptionRight.Call],
                   key=lambda x: x.Strike)

    if len(calls) < 5:
        return

    # Use factory method to create butterfly
    strategy = OptionStrategies.butterfly_call(
        self.option_symbol,
        calls[0].Strike,   # Lower strike (buy)
        calls[2].Strike,   # Middle strike (sell 2x)
        calls[4].Strike,   # Upper strike (buy)
        expiry
    )

    # Execute entire strategy atomically
    self.buy(strategy, 1)      # Long butterfly
    # OR
    # self.sell(strategy, 1)   # Short butterfly
```

**Example - Iron Condor:**

```python
# Get OTM strikes
atm = chain.Underlying.Price
strikes = sorted(set(c.Strike for c in chain if c.Expiry == expiry))

put_buy = [s for s in strikes if s < atm * 0.95][0]
put_sell = [s for s in strikes if s < atm * 0.97][0]
call_sell = [s for s in strikes if s > atm * 1.03][0]
call_buy = [s for s in strikes if s > atm * 1.05][0]

# Create iron condor
strategy = OptionStrategies.iron_condor(
    self.option_symbol,
    put_buy,      # Lower put protection
    put_sell,     # Put income
    call_sell,    # Call income
    call_buy,     # Upper call protection
    expiry
)

# Execute atomically with automatic position grouping
self.buy(strategy, 1)
```

**All Factory Method Benefits:**

1. **Position Grouping**: LEAN automatically groups legs for margin
2. **Strategy Recognition**: Matches to 37 known multi-leg patterns
3. **Margin Accuracy**: Correct multi-leg margin (not single-leg margin)
4. **Clean Exit**: `Liquidate(strategy_symbol)` closes entire position
5. **Compatible with ComboOrders**: Can still use for price control

### Combo Order Types (2025 Update)

**✅ CONFIRMED: ComboOrders are FULLY SUPPORTED** (as of 2025)

| Order Type | QuantConnect | Charles Schwab | Notes |
|------------|--------------|----------------|-------|
| **ComboMarketOrder** | ✅ Yes | ✅ SUPPORTED | Execute all legs at market |
| **ComboLimitOrder** | ✅ Yes | ✅ SUPPORTED | Net limit price across all legs |
| **ComboLegLimitOrder** | ✅ Yes | ❌ NOT supported | Individual leg limits (IB only) |

**CRITICAL for Charles Schwab Users:**

- ✅ Use `ComboMarketOrder()` for market execution
- ✅ Use `ComboLimitOrder()` with **net debit/credit pricing**
- ❌ Do NOT use `ComboLegLimitOrder()` - individual leg limits not supported on Schwab
- ❌ Do NOT specify `order_price` parameter in `Leg.Create()` calls for Schwab

**Benefits of ComboOrders:**

- ✅ Automatic multi-leg margin calculation (not single-leg margin)
- ✅ Atomic execution (all-or-nothing fills)
- ✅ Single commission per combo
- ✅ Prevents holding unbalanced positions
- ✅ Automatic strategy recognition (37+ patterns)

### Current Multi-Leg Execution

```python
def ExecuteIronCondor(self, chain, underlying_price):
    """Execute iron condor with individual leg orders."""
    calls = sorted([x for x in chain if x.Right == OptionRight.Call],
                   key=lambda x: x.Strike)
    puts = sorted([x for x in chain if x.Right == OptionRight.Put],
                  key=lambda x: x.Strike, reverse=True)

    otm_calls = [x for x in calls if x.Strike > underlying_price]
    otm_puts = [x for x in puts if x.Strike < underlying_price]

    if len(otm_calls) < 2 or len(otm_puts) < 2:
        return None

    # Define legs
    short_call = otm_calls[0]
    long_call = otm_calls[1]
    short_put = otm_puts[0]
    long_put = otm_puts[1]

    # Execute individual legs (current method)
    # Risk: Legs may fill at different times/prices
    tickets = []
    tickets.append(self.MarketOrder(short_call.Symbol, -1))
    tickets.append(self.MarketOrder(long_call.Symbol, 1))
    tickets.append(self.MarketOrder(short_put.Symbol, -1))
    tickets.append(self.MarketOrder(long_put.Symbol, 1))

    return tickets
```

### Leg Risk Management

When executing multi-leg strategies with individual orders:

```python
class MultiLegOrderManager:
    """Manage multi-leg order execution and risk."""

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.pending_legs = []
        self.completed_legs = []

    def execute_spread(self, legs):
        """
        Execute multi-leg spread with risk management.

        Args:
            legs: List of (symbol, quantity) tuples
        """
        self.pending_legs = []
        self.completed_legs = []

        for symbol, quantity in legs:
            ticket = self.algorithm.MarketOrder(symbol, quantity)
            self.pending_legs.append({
                'ticket': ticket,
                'symbol': symbol,
                'quantity': quantity
            })

        return self.pending_legs

    def check_completion(self):
        """Check if all legs are filled."""
        for leg in self.pending_legs:
            if leg['ticket'].Status == OrderStatus.Filled:
                if leg not in self.completed_legs:
                    self.completed_legs.append(leg)

        all_filled = len(self.completed_legs) == len(self.pending_legs)
        return all_filled

    def handle_partial_fill(self):
        """Handle case where some legs filled but others didn't."""
        filled = [l for l in self.pending_legs
                  if l['ticket'].Status == OrderStatus.Filled]
        unfilled = [l for l in self.pending_legs
                    if l['ticket'].Status != OrderStatus.Filled]

        if filled and unfilled:
            self.algorithm.Log(f"WARNING: Partial spread fill - "
                             f"{len(filled)} filled, {len(unfilled)} pending")

            # Option 1: Cancel unfilled and unwind filled
            for leg in unfilled:
                self.algorithm.Transactions.CancelOrder(leg['ticket'].OrderId)

            # Unwind filled legs
            for leg in filled:
                self.algorithm.MarketOrder(leg['symbol'], -leg['quantity'])

            return False

        return True

# Usage
def OnData(self, data):
    manager = MultiLegOrderManager(self)

    # Define iron condor legs
    legs = [
        (short_call.Symbol, -1),
        (long_call.Symbol, 1),
        (short_put.Symbol, -1),
        (long_put.Symbol, 1)
    ]

    manager.execute_spread(legs)
```

### Combo Order Implementation (2025)

Combo orders are now fully supported:

```python
def ExecuteIronCondorCombo(self, chain, underlying_price):
    """Execute iron condor as single combo order."""
    calls = sorted([x for x in chain if x.Right == OptionRight.Call],
                   key=lambda x: x.Strike)
    puts = sorted([x for x in chain if x.Right == OptionRight.Put],
                  key=lambda x: x.Strike, reverse=True)

    otm_calls = [x for x in calls if x.Strike > underlying_price]
    otm_puts = [x for x in puts if x.Strike < underlying_price]

    if len(otm_calls) < 2 or len(otm_puts) < 2:
        return None

    short_call = otm_calls[0]
    long_call = otm_calls[1]
    short_put = otm_puts[0]
    long_put = otm_puts[1]

    # Create legs - at least one must be positive, one negative
    legs = [
        Leg.Create(short_call.Symbol, -1),
        Leg.Create(long_call.Symbol, 1),
        Leg.Create(short_put.Symbol, -1),
        Leg.Create(long_put.Symbol, 1)
    ]

    # Option 1: Combo Market Order
    tickets = self.ComboMarketOrder(legs)

    # Option 2: Combo Limit Order (same limit for all legs)
    # tickets = self.ComboLimitOrder(legs, net_limit_price)

    # Option 3: Combo Leg Limit Order (different limits per leg - recommended)
    # tickets = self.ComboLegLimitOrder(legs)

    return tickets

def ExecuteWithOptionStrategies(self, chain, underlying_price):
    """Use built-in OptionStrategies helper."""
    # Find appropriate expiration
    expiry = min(set(c.Expiry for c in chain),
                 key=lambda x: abs((x - self.Time).days - 30))

    # Use OptionStrategies.IronCondor helper
    iron_condor = OptionStrategies.IronCondor(
        self.option_symbol,
        put_strike_lower=underlying_price * 0.95,
        put_strike=underlying_price * 0.97,
        call_strike=underlying_price * 1.03,
        call_strike_higher=underlying_price * 1.05,
        expiry=expiry
    )

    # Execute as combo market order
    self.Buy(iron_condor, 1)
```

### Iron Condor Filter

Use the built-in iron condor filter to narrow the universe:

```python
def Initialize(self):
    option = self.AddOption("SPY", Resolution.Minute)
    # Filter for iron condor contracts: 30 DTE, 5 strikes for shorts, 10 for longs
    option.SetFilter(lambda u: u.IncludeWeeklys()
                               .Strikes(-20, 20)
                               .Expiration(0, 30)
                               .IronCondor(30, 5, 10))
```

## Exercise and Assignment

### Handling Exercise

```python
def Initialize(self):
    option = self.AddOption("SPY", Resolution.Minute)

    # Set exercise model
    option.SetOptionExerciseModel(
        DefaultOptionExerciseModel()
    )

def OnData(self, data: Slice) -> None:
    # Check for options near expiration
    for symbol, holding in self.Portfolio.items():
        if not holding.Invested:
            continue

        security = self.Securities[symbol]
        if security.Type != SecurityType.Option:
            continue

        option = security
        dte = (option.Expiry - self.Time).days

        # Close positions before expiration to avoid assignment
        if dte <= 1:
            self.Liquidate(symbol)
            self.Log(f"Closed {symbol} to avoid assignment")
```

### Assignment Handling

```python
def OnOrderEvent(self, orderEvent: OrderEvent) -> None:
    """Handle option assignment/exercise events."""
    if orderEvent.Status != OrderStatus.Filled:
        return

    # Check if this is an option exercise
    if "Option Exercise" in orderEvent.Message:
        self.Log(f"Option exercised: {orderEvent}")

        # May need to manage resulting stock position
        symbol = orderEvent.Symbol
        if symbol.SecurityType == SecurityType.Equity:
            # Stock was assigned from short option
            self.Log(f"Stock position from assignment: {self.Portfolio[symbol].Quantity}")
```

## Memory Management

Options backtesting is extremely memory-intensive. Understanding and managing memory is critical for successful options algorithms.

### Platform Constraints

| Constraint | Limit | Notes |
|------------|-------|-------|
| **Max Backtest Runtime** | 12 hours | Hard limit |
| **B-MICRO RAM** | ~4 GB | Free tier |
| **Standard Node RAM** | 8 GB | Paid tier |
| **Large Node RAM** | 12+ GB | For options |
| **Data Upload Limit** | 700 MB | Results upload cap |

### Common Memory Issues

Options algorithms commonly encounter `System.OutOfMemoryException`:

```python
# PROBLEMATIC: Loading full option chain every bar
def OnData(self, data):
    chain = data.OptionChains.get(self.option_symbol)
    if chain:
        # This loads ALL contracts in chain into memory
        all_contracts = list(chain)  # Memory spike!

        # Processing many contracts per bar quickly exhausts RAM
        for contract in all_contracts:
            self.analyze_contract(contract)
```

### Memory-Efficient Option Chain Access

Use `OptionChainProvider.GetOptionContractList()` for lighter-weight chain access:

```python
def Initialize(self):
    self.equity = self.AddEquity("SPY", Resolution.Minute)
    self.underlying = self.equity.Symbol

    # DON'T subscribe to full option chain for memory efficiency
    # self.option = self.AddOption("SPY")  # Heavy memory usage

    # Instead, query contracts on-demand
    self.contract_symbols = []

def OnData(self, data):
    # Get contract list without full subscription
    contracts = self.OptionChainProvider.GetOptionContractList(
        self.underlying,
        self.Time
    )

    # Filter contracts manually (lighter than full chain)
    underlying_price = self.Securities[self.underlying].Price

    filtered = [c for c in contracts
                if abs(c.ID.StrikePrice - underlying_price) < 20
                and 30 <= (c.ID.Date - self.Time).days <= 60]

    # Only subscribe to contracts you need
    for contract in filtered[:10]:  # Limit contracts
        if contract not in self.contract_symbols:
            self.AddOptionContract(contract, Resolution.Minute)
            self.contract_symbols.append(contract)
```

### Memory Optimization Strategies

```python
class MemoryEfficientOptionsAlgo(QCAlgorithm):
    """Options algorithm optimized for memory usage."""

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)

        self.equity = self.AddEquity("SPY", Resolution.Minute)
        self.underlying = self.equity.Symbol

        # 1. Use restrictive filter
        option = self.AddOption("SPY", Resolution.Minute)
        option.SetFilter(self.conservative_filter)

        # 2. Track subscribed contracts for cleanup
        self.active_contracts = set()
        self.max_contracts = 20  # Limit concurrent contracts

        # 3. Schedule periodic cleanup
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(16, 0),
            self.cleanup_expired_contracts
        )

    def conservative_filter(self, universe):
        """Restrictive filter to limit memory usage."""
        return universe \
            .Strikes(-3, 3) \
            .Expiration(30, 45) \
            .OnlyApplyFilterAtMarketOpen()  # Reduces re-filtering

    def cleanup_expired_contracts(self):
        """Remove expired contracts to free memory."""
        expired = []

        for contract in self.active_contracts:
            security = self.Securities.get(contract)
            if security and security.Expiry <= self.Time:
                expired.append(contract)

        for contract in expired:
            self.RemoveSecurity(contract)
            self.active_contracts.discard(contract)
            self.Log(f"Removed expired contract: {contract}")

    def OnData(self, data):
        chain = data.OptionChains.get(self.option.Symbol)
        if not chain:
            return

        # 4. Process only subset of contracts
        contracts = list(chain)[:self.max_contracts]

        # 5. Avoid storing large data structures
        # BAD: self.historical_chains.append(chain)
        # GOOD: Store only essential data
        best_contract = self.select_best_contract(contracts)

        if best_contract:
            self.active_contracts.add(best_contract.Symbol)

    def select_best_contract(self, contracts):
        """Select single best contract without storing all."""
        if not contracts:
            return None

        # Filter and select in one pass
        underlying_price = self.Securities[self.underlying].Price

        best = None
        best_score = float('-inf')

        for c in contracts:
            if c.Right != OptionRight.Call:
                continue

            delta = abs(c.Greeks.Delta) if c.Greeks else 0
            if 0.25 <= delta <= 0.35:
                score = c.OpenInterest  # Or other criteria
                if score > best_score:
                    best_score = score
                    best = c

        return best
```

### Backtest Node Selection

Choose appropriate node size for options backtesting:

| Data Complexity | Recommended Node | RAM |
|-----------------|------------------|-----|
| Single underlying, tight filter | B-MICRO | 4 GB |
| Single underlying, wide filter | Standard | 8 GB |
| Multiple underlyings | Large | 12 GB |
| Full chain analysis | Large/GPU | 12+ GB |

```python
# In QuantConnect UI or CLI, select appropriate node:
# lean backtest --node B4-8  # 8GB node
# lean backtest --node B4-12 # 12GB node
```

### Reducing Backtest Data Size

Prevent "results too large to upload" errors:

```python
def Initialize(self):
    # Reduce logging frequency
    self.log_frequency = 100  # Log every 100th bar
    self.bar_count = 0

    # Limit chart data points
    self.SetPandasConverter(None)  # Disable pandas overhead

def OnData(self, data):
    self.bar_count += 1

    # Sparse logging
    if self.bar_count % self.log_frequency == 0:
        self.Log(f"Bar {self.bar_count}: Portfolio ${self.Portfolio.TotalPortfolioValue:,.0f}")

    # Avoid excessive Plot() calls
    if self.bar_count % 1000 == 0:  # Plot less frequently
        self.Plot("Portfolio", "Value", self.Portfolio.TotalPortfolioValue)
```

## Best Practices

### 1. Always Check Liquidity

```python
def IsLiquid(self, contract, min_oi=100, max_spread_pct=0.10):
    """Check if contract is liquid enough to trade."""
    if contract.OpenInterest < min_oi:
        return False

    if contract.BidPrice <= 0:
        return False

    spread_pct = (contract.AskPrice - contract.BidPrice) / contract.BidPrice
    if spread_pct > max_spread_pct:
        return False

    return True
```

### 2. Manage Position Size

```python
def CalculatePositionSize(self, contract, max_risk_pct=0.02):
    """Calculate position size based on risk."""
    portfolio_value = self.Portfolio.TotalPortfolioValue
    max_risk = portfolio_value * max_risk_pct

    # Maximum loss for long option is premium paid
    option_cost = contract.AskPrice * 100  # Standard multiplier

    max_contracts = int(max_risk / option_cost)
    return max(1, min(max_contracts, 10))  # 1-10 contracts
```

### 3. Monitor Greeks Exposure

```python
def CheckGreeksLimits(self):
    """Ensure portfolio Greeks are within limits."""
    greeks = self.CalculatePortfolioGreeks()

    # Define limits
    max_delta = self.Portfolio.TotalPortfolioValue * 0.50  # 50% delta exposure
    max_gamma = self.Portfolio.TotalPortfolioValue * 0.10

    if abs(greeks['delta']) > max_delta:
        self.Log(f"WARNING: Delta exposure too high: {greeks['delta']}")
        # Consider hedging

    if abs(greeks['gamma']) > max_gamma:
        self.Log(f"WARNING: Gamma exposure too high: {greeks['gamma']}")
```

### 4. Roll Positions Before Expiration

```python
def ManageExpiringOptions(self):
    """Roll or close options approaching expiration."""
    for symbol, holding in self.Portfolio.items():
        if not holding.Invested:
            continue

        security = self.Securities[symbol]
        if security.Type != SecurityType.Option:
            continue

        option = security
        dte = (option.Expiry - self.Time).days

        if dte <= 5:  # 5 days to expiration
            self.Log(f"Rolling {symbol} - {dte} DTE remaining")

            # Close current position
            self.Liquidate(symbol)

            # Open new position with later expiration
            # (implementation depends on strategy)
```

### 5. Use Appropriate Pricing Models

```python
def Initialize(self):
    option = self.AddOption("SPY", Resolution.Minute)

    # For US equity options (American style)
    option.PriceModel = OptionPriceModels.BinomialCoxRossRubinstein()

    # For index options (often European style)
    # option.PriceModel = OptionPriceModels.BlackScholes()
```

---

*Last Updated: November 2025*
