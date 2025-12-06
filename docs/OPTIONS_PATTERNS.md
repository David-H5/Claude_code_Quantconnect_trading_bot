# Options Trading Patterns

This guide covers options trading patterns and implementations for QuantConnect.

## Initialize Options with Greeks

```python
def initialize(self):
    equity = self.add_equity("SPY")
    option = self.add_option("SPY")
    option.set_filter(-10, +10, 0, 90)  # +/-10 strikes, 0-90 days

    # Optional: Specify pricing model (defaults to Black-Scholes/Bjerksund-Stensland)
    # option.price_model = OptionPriceModels.crank_nicolson_fd()

    # NOTE: As of LEAN PR #6720, Greeks use IV and require NO warmup
    # Greeks are available immediately upon option data arrival
```

## Access Greeks in OnData

```python
def on_data(self, slice):
    chain = slice.option_chains.get(self.option_symbol)
    if not chain:
        return

    # Greeks available immediately (IV-based, no warmup required)
    for contract in chain:
        delta = contract.greeks.delta
        gamma = contract.greeks.gamma
        theta = contract.greeks.theta
        theta_per_day = contract.greeks.theta_per_day  # Daily theta decay
        vega = contract.greeks.vega
        rho = contract.greeks.rho
        iv = contract.implied_volatility

        # Greeks values match Interactive Brokers and major brokerages
        if 0.25 < abs(delta) < 0.35 and iv > 0.20:
            self.market_order(contract.symbol, 1)
```

## Strategy Filters

```python
# Iron condor filter
option.set_filter(lambda u: u.iron_condor(30, 5, 10))

# Straddle filter
option.set_filter(lambda u: u.straddle(30))
```

## Greeks-Based Universe Filtering

**Recommended**: Filter options using Greeks **before** they enter your scanner to reduce data processing:

```python
def initialize(self):
    option = self.add_option("SPY")
    option.set_filter(self.option_filter)

def option_filter(self, universe):
    """Filter options by Greeks and IV before detailed analysis"""
    return (universe
        .delta(0.25, 0.35)               # Delta range for scanner
        .implied_volatility(0.20, None)  # Minimum IV threshold
        .expiration(30, 180)             # DTE range
        .strikes(-10, 10)                # Strike range from ATM
        .include_weeklys())              # Include weekly options
```

**Available Chainable Filter Methods:**

- `.delta(min, max)` or `.d(min, max)` - Filter by delta
- `.gamma(min, max)` or `.g(min, max)` - Filter by gamma
- `.vega(min, max)` or `.v(min, max)` - Filter by vega
- `.theta(min, max)` or `.t(min, max)` - Filter by theta
- `.rho(min, max)` or `.r(min, max)` - Filter by rho
- `.implied_volatility(min, max)` or `.iv(min, max)` - Filter by IV
- `.open_interest(min, max)` or `.oi(min, max)` - Filter by open interest

**Benefits:**

- Reduces data to process in scanners
- Faster backtests and live execution
- Lower memory usage with large option chains

## OptionStrategies Factory Methods

**QuantConnect provides 37+ pre-built strategy constructors** that automatically create multi-leg positions with proper grouping and margin calculation.

### Butterfly Call Pattern

```python
def on_data(self, slice):
    chain = slice.option_chains.get(self.option_symbol)
    if not chain:
        return

    # Get strikes
    expiry = min([x.expiry for x in chain])
    strikes = sorted([c.strike for c in chain if c.expiry == expiry])

    # Use factory method instead of manual Leg.Create()
    strategy = OptionStrategies.butterfly_call(
        self.option_symbol,
        strikes[0],   # Lower strike (buy)
        strikes[2],   # Middle strike (sell 2x)
        strikes[4],   # Upper strike (buy)
        expiry
    )

    # Buy or sell the entire strategy atomically
    self.buy(strategy, 1)      # Long butterfly
    # OR
    self.sell(strategy, 1)     # Short butterfly
```

### Iron Condor Pattern

```python
strategy = OptionStrategies.iron_condor(
    self.option_symbol,
    put_buy_strike,    # Lower put protection
    put_sell_strike,   # Put income
    call_sell_strike,  # Call income
    call_buy_strike,   # Upper call protection
    expiry
)
self.buy(strategy, 1)  # Execute entire condor atomically
```

### Common Strategy Factory Methods

| Strategy Type | Method | Legs |
|---------------|--------|------|
| Butterfly Call | `OptionStrategies.butterfly_call()` | Buy 1, Sell 2, Buy 1 |
| Butterfly Put | `OptionStrategies.butterfly_put()` | Buy 1, Sell 2, Buy 1 |
| Iron Butterfly | `OptionStrategies.iron_butterfly()` | Buy put, Sell put+call, Buy call |
| Iron Condor | `OptionStrategies.iron_condor()` | Buy put, Sell put, Sell call, Buy call |
| Bull Call Spread | `OptionStrategies.bull_call_spread()` | Buy lower, Sell upper |
| Bear Put Spread | `OptionStrategies.bear_put_spread()` | Buy upper, Sell lower |
| Straddle | `OptionStrategies.straddle()` | Buy call + put at same strike |
| Strangle | `OptionStrategies.strangle()` | Buy OTM call + OTM put |
| Covered Call | `OptionStrategies.covered_call()` | Long stock + sell call |

**37+ strategies available** - see [OptionStrategies.cs](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/Option/OptionStrategies.cs) for complete list.

### Benefits of Factory Methods

- Automatic position grouping for correct margin calculations
- Cleaner code (no manual `Leg.Create()` calls)
- Automatic strategy detection by LEAN
- Compatible with `ComboLimitOrder()` execution
- Single command to enter/exit entire position

**Note:** You can still use manual `Leg.Create()` with `ComboLimitOrder()` for custom strategies or precise price control.

## Two-Part Spread Strategy (Primary)

**Philosophy**: Leg into butterflies/iron condors in two parts to achieve net-credit positions.

**Key Insight**: Wide bid-ask spreads are OPPORTUNITIES (can get filled below mid), not risks to avoid.

**Process**:

1. Find underpriced debit spread (wide spreads = opportunity)
2. Execute debit at 35% from bid with 2.5s quick cancel
3. Find credit spread further OTM with credit >= debit cost
4. Execute credit at 65% from bid to complete position

**Critical Parameters**:

- Cancel unfilled orders after: **2.5 seconds**
- Minimum delay between attempts: **3 seconds**
- Maximum delay between attempts: **15 seconds**
- Minimum fill rate threshold: **25%**
- Starting contract size: **1 contract** (highest fill probability)
- Optimal expiration range: **30-180 days**

**User Observations (from live trading)**:

- Orders that don't fill in 2-3 seconds won't fill at all
- 1 contract at a time offers highest fill rate but low volume
- Random delays prevent market maker algorithm detection
- Wide spreads mean price improvement opportunity, not risk
- Position balance per option chain prevents holding excess longs

**Implementation Files**:

- `execution/two_part_spread.py` - Core strategy logic
- `execution/arbitrage_executor.py` - Full autonomous executor
- `execution/fill_predictor.py` - Fill rate tracking
- `execution/spread_analysis.py` - Spread quality analysis
- `models/enhanced_volatility.py` - IV analysis for opportunities
