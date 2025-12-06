# Analytics Module

**UPGRADE-015 Phase 8: Options Analytics Engine**

This module provides comprehensive options analytics capabilities for the trading system, including IV surface modeling, term structure analysis, volatility skew calculations, Greeks computation, and options pricing.

## Components

### 1. IV Surface (`iv_surface.py`)

Models the implied volatility surface across strikes and expirations.

```python
from analytics import IVSurface, create_iv_surface

# Create surface
surface = create_iv_surface(underlying_price=450.0)

# Add data points
surface.add_point(strike=440, expiry_days=30, iv=0.28)
surface.add_point(strike=450, expiry_days=30, iv=0.25)
surface.add_point(strike=460, expiry_days=30, iv=0.26)

# Interpolate IV
iv = surface.get_iv(moneyness=0.98, expiry_days=30)

# Get volatility smile
smile = surface.get_smile(expiry_days=30)

# Get term structure
term_structure = surface.get_iv_term_structure(moneyness=1.0)

# Detect arbitrage
arbitrage = surface.detect_arbitrage()
```

### 2. Term Structure (`term_structure.py`)

Analyzes ATM implied volatility across time.

```python
from analytics import TermStructure, create_term_structure

# Create term structure
ts = create_term_structure(underlying="SPY")

# Add points
ts.add_points([
    (7, 0.20),   # 7 DTE, 20% IV
    (14, 0.21),
    (30, 0.23),
    (60, 0.25),
])

# Analyze shape
shape = ts.get_shape()  # contango, backwardation, flat, humped

# Get forward volatility
forward_iv = ts.get_forward_iv(start_dte=30, end_dte=60)

# Get VIX-equivalent
vix_30d = ts.get_vix_equivalent()
```

### 3. Volatility Skew (`volatility_skew.py`)

Analyzes put/call volatility skew.

```python
from analytics import VolatilitySkew, create_volatility_skew

# Create skew analyzer
skew = create_volatility_skew(underlying_price=450.0, expiry_days=30)

# Add option data
skew.add_point(strike=430, iv=0.32, delta=-0.25, option_type="put")
skew.add_point(strike=450, iv=0.25, delta=0.50, option_type="call")
skew.add_point(strike=470, iv=0.27, delta=0.25, option_type="call")

# Calculate metrics
metrics = skew.calculate_metrics()
print(f"Risk Reversal 25D: {metrics.risk_reversal_25d:.2%}")
print(f"Butterfly 25D: {metrics.butterfly_25d:.2%}")
print(f"Skew Type: {metrics.skew_type.value}")

# Estimate tail risk
tail_risk = skew.estimate_tail_risk()
```

### 4. Greeks Calculator (`greeks_calculator.py`)

Calculates all option Greeks.

```python
from analytics import GreeksCalculator, create_greeks_calculator

# Create calculator
calc = create_greeks_calculator(risk_free_rate=0.05)

# Calculate Greeks
greeks = calc.calculate(
    spot=450,
    strike=450,
    time_to_expiry=30/365,
    iv=0.25,
    option_type="call",
    quantity=10,
)

print(f"Delta: {greeks.delta:.4f}")
print(f"Gamma: {greeks.gamma:.4f}")
print(f"Theta: {greeks.theta:.4f}")
print(f"Vega: {greeks.vega:.4f}")
print(f"Dollar Delta: ${greeks.dollar_delta:,.2f}")

# Aggregate position Greeks
position = calc.aggregate_position([
    {"spot": 450, "strike": 440, "dte": 30, "iv": 0.25, "type": "call", "quantity": 5},
    {"spot": 450, "strike": 460, "dte": 30, "iv": 0.25, "type": "call", "quantity": -5},
])

print(f"Net Delta: {position.net_delta:.2f}")
print(f"Net Gamma: {position.net_gamma:.4f}")

# Scenario analysis
scenarios = calc.scenario_analysis(
    option={"spot": 450, "strike": 450, "dte": 30, "iv": 0.25, "type": "call"},
    spot_changes=[-0.10, -0.05, 0, 0.05, 0.10],
    iv_changes=[-0.05, 0, 0.05],
)
```

### 5. Pricing Models (`pricing_models.py`)

Options pricing using Black-Scholes and Binomial Tree models.

```python
from analytics import BlackScholes, BinomialTree, create_pricer, compare_prices

# Black-Scholes (European)
bs = BlackScholes(risk_free_rate=0.05)
result = bs.price(
    spot=450,
    strike=450,
    time_to_expiry=30/365,
    iv=0.25,
    option_type="call",
)
print(f"BS Price: ${result.price:.2f}")
print(f"Intrinsic: ${result.intrinsic_value:.2f}")
print(f"Time Value: ${result.time_value:.2f}")

# Implied Volatility
iv = bs.implied_volatility(
    market_price=12.50,
    spot=450,
    strike=450,
    time_to_expiry=30/365,
    option_type="call",
)
print(f"IV: {iv:.2%}")

# Binomial Tree (American)
bt = BinomialTree(risk_free_rate=0.05, steps=100)
result = bt.price(
    spot=450,
    strike=450,
    time_to_expiry=30/365,
    iv=0.25,
    option_type="put",
    exercise_style="american",
)
print(f"American Put: ${result.price:.2f}")
print(f"Early Exercise Premium: ${result.early_exercise_premium:.2f}")

# Compare models
comparison = compare_prices(
    spot=450, strike=450, time_to_expiry=30/365, iv=0.25, option_type="put"
)
print(comparison)
```

## Greeks Reference

### First-Order Greeks

| Greek | Description | Typical Range |
|-------|-------------|---------------|
| **Delta** | Price change per $1 spot move | Call: 0 to 1, Put: -1 to 0 |
| **Gamma** | Delta change per $1 spot move | Always positive |
| **Theta** | Price decay per day | Usually negative for longs |
| **Vega** | Price change per 1% IV change | Always positive |
| **Rho** | Price change per 1% rate change | Call: positive, Put: negative |

### Second-Order Greeks

| Greek | Description |
|-------|-------------|
| **Vanna** | Delta change per 1% IV change |
| **Volga** | Vega change per 1% IV change |
| **Charm** | Delta change per day |

## Testing

```bash
# Run analytics tests
pytest tests/analytics/ -v

# Run with coverage
pytest tests/analytics/ --cov=analytics
```

## Mathematical Background

### Black-Scholes Formula

For a call option:
```
C = S*N(d1) - K*e^(-rT)*N(d2)

d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d2 = d1 - σ√T
```

Where:
- S = Spot price
- K = Strike price
- r = Risk-free rate
- T = Time to expiry
- σ = Implied volatility
- N() = Standard normal CDF

### IV Surface

The IV surface is a 3D surface where:
- X-axis: Moneyness (Strike / Spot)
- Y-axis: Time to expiry
- Z-axis: Implied volatility

Key features:
- **Smile**: IV higher for OTM options
- **Skew**: Asymmetric smile (puts typically higher IV)
- **Term Structure**: IV variation across expirations
