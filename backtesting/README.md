# Backtesting Robustness Module

UPGRADE-015 Phase 9: Backtesting Robustness Tools

This module provides comprehensive tools for validating trading strategy robustness beyond simple backtesting.

## Components

### 1. Walk-Forward Analysis (`walk_forward.py`)

Implements walk-forward optimization to detect overfitting and validate out-of-sample performance.

```python
from backtesting import WalkForwardAnalyzer, create_walk_forward_analyzer

# Create analyzer with rolling windows
analyzer = create_walk_forward_analyzer(
    method="rolling",
    in_sample_days=252,  # 1 year optimization
    out_sample_days=63,  # 3 months validation
    num_windows=5,
)

# Run analysis with returns data
result = analyzer.analyze(returns)

# Check robustness
print(f"Robustness Score: {result.robustness_score:.2f}")
print(f"Is Robust: {result.is_robust}")
print(f"Performance Ratio: {result.avg_performance_ratio:.2f}")
```

**Methods Available:**
- `ROLLING` - Fixed window moves forward
- `ANCHORED` - Start always at beginning, end expands
- `EXPANDING` - Same as anchored (window grows over time)

### 2. Monte Carlo Simulation (`monte_carlo.py`)

Bootstrap-based Monte Carlo simulation for confidence interval estimation and risk analysis.

```python
from backtesting import MonteCarloSimulator, create_monte_carlo_simulator

# Create simulator
simulator = create_monte_carlo_simulator(
    num_simulations=10000,
    num_periods=252,  # 1 year
    bootstrap_method="block",  # Preserves autocorrelation
    random_seed=42,
)

# Run simulation
result = simulator.simulate(historical_returns)

# Analyze results
print(f"Mean Return: {result.mean_return:.2%}")
print(f"VaR 95%: {result.var_95:.2%}")
print(f"CVaR 95%: {result.cvar_95:.2%}")
print(f"Probability of Profit: {result.prob_profit:.0%}")
```

**Bootstrap Methods:**
- `SIMPLE` - IID resampling (assumes independence)
- `BLOCK` - Block bootstrap (preserves autocorrelation)
- `STATIONARY` - Random block lengths (more robust)

### 3. Parameter Sensitivity (`parameter_sensitivity.py`)

Analyze how sensitive strategy performance is to parameter changes.

```python
from backtesting import (
    ParameterSensitivity,
    ParameterRange,
    create_parameter_sensitivity,
)

# Create analyzer
analyzer = create_parameter_sensitivity(
    stability_threshold=0.20,  # 20% deviation for stability
    cliff_threshold=0.50,      # 50% drop = cliff
)

# Define parameter range
param = ParameterRange(
    name="lookback",
    min_value=10,
    max_value=50,
    num_points=20,
)

# Analyze sensitivity
def evaluate(lookback):
    # Your strategy evaluation here
    return calculate_sharpe(lookback)

result = analyzer.analyze_parameter(param, evaluate)

print(f"Sensitivity: {result.sensitivity_level.value}")
print(f"Stable Range: {result.stable_range}")
print(f"Cliff Points: {result.cliff_points}")
```

**Features:**
- Single parameter sensitivity
- Multi-parameter grid search
- 2D heatmap generation
- Robustness scoring

### 4. Regime Detection (`regime_detector.py`)

Detect market regimes and analyze strategy performance across different conditions.

```python
from backtesting import RegimeDetector, MarketRegime, create_regime_detector

# Create detector
detector = create_regime_detector(
    volatility_window=20,
    trend_window=60,
    low_vol_threshold=25.0,
    high_vol_threshold=75.0,
)

# Run analysis
analysis = detector.analyze(
    returns=returns,
    prices=prices,
    strategy_returns=strategy_returns,  # Optional
)

# Check current regime
print(f"Current: {analysis.current_regime.combined_regime.value}")
print(f"Confidence: {analysis.current_regime.confidence:.0%}")

# Performance by regime
for regime, perf in analysis.performance_by_regime.items():
    print(f"{regime}: Sharpe={perf['sharpe']:.2f}")
```

**Regime Types:**
- Volatility: LOW_VOL, NORMAL_VOL, HIGH_VOL, CRISIS
- Trend: STRONG_UPTREND, WEAK_UPTREND, SIDEWAYS, WEAK_DOWNTREND, STRONG_DOWNTREND
- Combined: BULL_LOW_VOL, BULL_HIGH_VOL, BEAR_LOW_VOL, BEAR_HIGH_VOL

### 5. Overfitting Guard (`overfitting_guard.py`)

Detect and quantify overfitting risk using statistical methods.

```python
from backtesting import OverfittingGuard, create_overfitting_guard

# Create guard
guard = create_overfitting_guard(
    min_dof_ratio=10,  # 10 data points per parameter
    significance_level=0.05,
)

# Analyze strategy
result = guard.analyze(
    returns=returns,
    num_parameters=5,
    num_trials=100,  # How many variations tested
    in_sample_returns=in_sample,
    out_sample_returns=out_sample,
)

# Check results
print(f"Overfit Risk: {result.overfit_risk.value}")
print(f"Overfit Probability: {result.overfit_probability:.0%}")
print(f"Original Sharpe: {result.original_sharpe:.2f}")
print(f"Adjusted Sharpe: {result.adjusted_sharpe:.2f}")
print(f"Haircut: {result.sharpe_haircut:.0%}")

# Recommendations
for rec in result.recommendations:
    print(f"- {rec}")
```

**Analysis Methods:**
- Degrees of Freedom analysis
- Deflated Sharpe Ratio (Bailey & Lopez de Prado)
- In-sample vs Out-of-sample comparison
- Sharpe ratio haircut calculation

## Best Practices

### 1. Walk-Forward Validation
- Use at least 3-5 windows
- Out-of-sample should be 20-30% of in-sample
- Check performance ratio (out/in) > 0.5

### 2. Monte Carlo Analysis
- Use block bootstrap for autocorrelated returns
- Run at least 10,000 simulations
- Focus on tail risk (VaR/CVaR), not mean

### 3. Sensitivity Analysis
- Test all tunable parameters
- Look for flat "plateaus" (stable regions)
- Avoid parameters near "cliffs"

### 4. Regime Analysis
- Ensure strategy works across regimes
- Beware of strategies that only work in bull markets
- Consider regime-specific position sizing

### 5. Overfitting Prevention
- Maintain DOF ratio > 10 (data points per parameter)
- Use adjusted Sharpe for realistic expectations
- Extend track record before trusting results

## Integration Example

```python
from backtesting import (
    create_walk_forward_analyzer,
    create_monte_carlo_simulator,
    create_parameter_sensitivity,
    create_regime_detector,
    create_overfitting_guard,
)

# 1. Walk-Forward Validation
wf_analyzer = create_walk_forward_analyzer(method="rolling")
wf_result = wf_analyzer.analyze(returns)

# 2. Monte Carlo Risk Analysis
mc_simulator = create_monte_carlo_simulator(num_simulations=10000)
mc_result = mc_simulator.simulate(returns)

# 3. Parameter Sensitivity
sens_analyzer = create_parameter_sensitivity()
sens_result = sens_analyzer.analyze_parameter(param_range, evaluator)

# 4. Regime Analysis
regime_detector = create_regime_detector()
regime_result = regime_detector.analyze(returns, prices, strategy_returns)

# 5. Overfitting Check
overfit_guard = create_overfitting_guard()
overfit_result = overfit_guard.analyze(returns, num_parameters=5)

# Comprehensive validation report
print("=" * 50)
print("STRATEGY VALIDATION REPORT")
print("=" * 50)
print(f"Walk-Forward Robust: {wf_result.is_robust}")
print(f"Monte Carlo P(Profit): {mc_result.prob_profit:.0%}")
print(f"Sensitivity Robustness: {sens_analyzer.calculate_robustness_score():.2f}")
print(f"Regime Coverage: {len(regime_result.regime_frequencies)} regimes")
print(f"Overfit Risk: {overfit_result.overfit_risk.value}")
```

## Requirements

- numpy
- scipy (for statistical tests)

## References

- Bailey & Lopez de Prado: "The Deflated Sharpe Ratio"
- Harvey & Liu: "Backtesting"
- Pardo: "The Evaluation and Optimization of Trading Strategies"
