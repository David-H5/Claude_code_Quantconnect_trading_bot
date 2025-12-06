"""
Backtesting Module

UPGRADE-015 Phase 9: Backtesting Robustness

Provides robust backtesting tools including:
- Walk-forward optimization
- Monte Carlo simulation
- Parameter sensitivity analysis
- Market regime detection
- Overfitting detection

Usage:
    from backtesting import (
        WalkForwardAnalyzer,
        MonteCarloSimulator,
        ParameterSensitivity,
        RegimeDetector,
        OverfittingGuard,
    )
"""

from backtesting.monte_carlo import (
    MonteCarloResult,
    MonteCarloSimulator,
    create_monte_carlo_simulator,
)
from backtesting.overfitting_guard import (
    OverfittingGuard,
    OverfittingResult,
    create_overfitting_guard,
)
from backtesting.parameter_sensitivity import (
    ParameterSensitivity,
    SensitivityResult,
    create_parameter_sensitivity,
)
from backtesting.regime_detector import (
    MarketRegime,
    RegimeDetector,
    create_regime_detector,
)
from backtesting.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardResult,
    create_walk_forward_analyzer,
)


__all__ = [
    # Walk Forward
    "WalkForwardAnalyzer",
    "WalkForwardResult",
    "create_walk_forward_analyzer",
    # Monte Carlo
    "MonteCarloSimulator",
    "MonteCarloResult",
    "create_monte_carlo_simulator",
    # Parameter Sensitivity
    "ParameterSensitivity",
    "SensitivityResult",
    "create_parameter_sensitivity",
    # Regime Detection
    "RegimeDetector",
    "MarketRegime",
    "create_regime_detector",
    # Overfitting Guard
    "OverfittingGuard",
    "OverfittingResult",
    "create_overfitting_guard",
]
