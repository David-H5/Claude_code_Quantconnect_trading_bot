"""
Analytics Pipeline Integration Tests

UPGRADE-015 Phase 12: Integration & Final Validation

Tests analytics and backtesting pipeline:
- IV surface calculations
- Greeks calculator
- Walk-forward optimization
- Monte Carlo simulation
- Parameter sensitivity
"""

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parent.parent))


class TestIVSurfaceIntegration:
    """Integration tests for IV surface module."""

    def test_iv_surface_import(self):
        """Test IV surface module imports correctly."""
        from analytics import create_iv_surface

        surface = create_iv_surface(underlying_price=450.0)
        assert surface is not None

    def test_iv_surface_data_loading(self):
        """Test IV surface can accept data."""
        from analytics import create_iv_surface

        surface = create_iv_surface(underlying_price=450.0)

        # Add sample data points (expiry_days is days to expiration as int)
        surface.add_point(
            strike=450.0,
            expiry_days=30,
            iv=0.20,
        )

        surface.add_point(
            strike=455.0,
            expiry_days=30,
            iv=0.22,
        )

        # Should have data
        assert surface.get_stats()["num_points"] >= 2


class TestGreeksCalculatorIntegration:
    """Integration tests for Greeks calculator."""

    def test_greeks_calculator_import(self):
        """Test Greeks calculator imports correctly."""
        from analytics import create_greeks_calculator

        calc = create_greeks_calculator()
        assert calc is not None

    def test_greeks_calculation(self):
        """Test Greeks calculations produce valid results."""
        from analytics import create_greeks_calculator

        calc = create_greeks_calculator(risk_free_rate=0.05)

        greeks = calc.calculate(
            spot=450.0,
            strike=455.0,
            time_to_expiry=30 / 365,  # 30 days
            iv=0.20,
            option_type="call",
        )

        # All Greeks should be calculated (GreeksResult is a dataclass)
        assert greeks.delta is not None
        assert greeks.gamma is not None
        assert greeks.theta is not None
        assert greeks.vega is not None
        assert greeks.rho is not None

        # Delta should be between -1 and 1
        assert -1 <= greeks.delta <= 1


class TestPricingModelsIntegration:
    """Integration tests for pricing models."""

    def test_black_scholes_import(self):
        """Test Black-Scholes model imports correctly."""
        from analytics import BlackScholes, create_pricer

        model = create_pricer(model="black_scholes")
        assert model is not None
        assert isinstance(model, BlackScholes)

    def test_black_scholes_pricing(self):
        """Test Black-Scholes pricing produces valid results."""
        from analytics import create_pricer

        model = create_pricer(model="black_scholes", risk_free_rate=0.05)

        result = model.price(
            spot=450.0,
            strike=455.0,
            time_to_expiry=30 / 365,
            iv=0.20,
            option_type="call",
        )

        # Price should be positive (PricingResult has .price attribute)
        assert result.price > 0


class TestWalkForwardIntegration:
    """Integration tests for walk-forward optimization."""

    def test_walk_forward_import(self):
        """Test walk-forward module imports correctly."""
        from backtesting import create_walk_forward_analyzer

        analyzer = create_walk_forward_analyzer()
        assert analyzer is not None

    def test_walk_forward_window_generation(self):
        """Test walk-forward window generation."""
        from datetime import datetime

        from backtesting import create_walk_forward_analyzer

        analyzer = create_walk_forward_analyzer(
            method="rolling",
            in_sample_days=60,
            out_sample_days=15,
            num_windows=3,
        )

        # Generate windows for a date range
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)

        windows = analyzer.generate_windows(start_date, end_date)

        # Windows are tuples of (in_start, in_end, out_start, out_end)
        assert len(windows) > 0
        for window in windows:
            assert len(window) == 4  # (in_start, in_end, out_start, out_end)
            in_start, in_end, out_start, out_end = window
            assert in_start < in_end
            assert out_start < out_end


class TestMonteCarloIntegration:
    """Integration tests for Monte Carlo simulation."""

    def test_monte_carlo_import(self):
        """Test Monte Carlo module imports correctly."""
        from backtesting import create_monte_carlo_simulator

        simulator = create_monte_carlo_simulator()
        assert simulator is not None

    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation produces valid results."""
        from backtesting import create_monte_carlo_simulator

        simulator = create_monte_carlo_simulator(
            num_simulations=100,
            num_periods=252,
        )

        # Sample returns
        returns = list(np.random.normal(0.001, 0.02, 252))  # 1 year daily returns

        result = simulator.simulate(returns)

        assert result is not None
        assert result.num_simulations == 100
        assert result.var_95 is not None
        assert result.cvar_95 is not None


class TestParameterSensitivityIntegration:
    """Integration tests for parameter sensitivity."""

    def test_parameter_sensitivity_import(self):
        """Test parameter sensitivity module imports correctly."""
        from backtesting import (
            create_parameter_sensitivity,
        )

        analyzer = create_parameter_sensitivity()
        assert analyzer is not None

    def test_parameter_sensitivity_analysis(self):
        """Test parameter sensitivity analysis."""
        from backtesting import create_parameter_sensitivity
        from backtesting.parameter_sensitivity import ParameterRange

        analyzer = create_parameter_sensitivity()

        # Define parameter range
        param_range = ParameterRange(
            name="stop_loss",
            min_value=0.01,
            max_value=0.10,
            num_points=10,
        )

        # Simple evaluator function
        def evaluator(param_value: float) -> float:
            # Higher stop loss = lower performance (simple example)
            return 1.0 - param_value * 5

        result = analyzer.analyze_parameter(
            param_range=param_range,
            evaluator=evaluator,
            base_value=0.05,
        )

        assert result is not None
        assert result.parameter_name == "stop_loss"
        assert result.optimal_value is not None


class TestRegimeDetectorIntegration:
    """Integration tests for regime detection."""

    def test_regime_detector_import(self):
        """Test regime detector module imports correctly."""
        from backtesting import create_regime_detector

        detector = create_regime_detector()
        assert detector is not None

    def test_regime_detection(self):
        """Test regime detection produces valid results."""
        from backtesting import create_regime_detector

        detector = create_regime_detector()

        # Generate sample data
        returns = list(np.random.normal(0.001, 0.02, 252))

        regime, volatility, percentile = detector.detect_volatility_regime(returns)

        assert regime is not None
        assert volatility > 0
        assert 0 <= percentile <= 100


class TestOverfittingGuardIntegration:
    """Integration tests for overfitting guard."""

    def test_overfitting_guard_import(self):
        """Test overfitting guard module imports correctly."""
        from backtesting import create_overfitting_guard

        guard = create_overfitting_guard()
        assert guard is not None

    def test_deflated_sharpe_calculation(self):
        """Test deflated Sharpe ratio calculation."""
        from backtesting import create_overfitting_guard

        guard = create_overfitting_guard()

        result = guard.calculate_deflated_sharpe(
            sharpe_ratio=1.5,
            num_trials=100,
            track_record_months=12,  # 12 months (1 year)
            skewness=0.0,
            kurtosis=3.0,
        )

        assert result is not None
        assert result.deflated_sharpe <= result.original_sharpe


class TestFullAnalyticsPipeline:
    """Integration tests for complete analytics pipeline."""

    def test_options_analytics_pipeline(self):
        """Test complete options analytics pipeline."""
        from analytics import create_greeks_calculator, create_iv_surface

        # Create components
        iv_surface = create_iv_surface(underlying_price=450.0)
        greeks_calc = create_greeks_calculator()

        # Add IV surface data (expiry_days is days to expiration as int)
        for strike in range(440, 460, 5):
            iv_surface.add_point(
                strike=float(strike),
                expiry_days=30,
                iv=0.20 + (strike - 450) * 0.001,
            )

        # Calculate Greeks using surface IV
        greeks = greeks_calc.calculate(
            spot=450.0,
            strike=450.0,
            time_to_expiry=30 / 365,
            iv=0.20,
            option_type="call",
        )

        assert greeks.delta is not None

    def test_backtesting_pipeline(self):
        """Test complete backtesting pipeline."""

        from backtesting import (
            create_monte_carlo_simulator,
            create_overfitting_guard,
            create_regime_detector,
        )

        # Create components
        simulator = create_monte_carlo_simulator(num_simulations=50)
        detector = create_regime_detector()
        guard = create_overfitting_guard()

        # Generate sample strategy returns
        returns = list(np.random.normal(0.001, 0.02, 252))

        # Detect regime
        regime, vol, pct = detector.detect_volatility_regime(returns)

        # Run Monte Carlo
        mc_result = simulator.simulate(returns)

        # Check for overfitting
        guard_result = guard.calculate_deflated_sharpe(
            sharpe_ratio=1.5,
            num_trials=10,
            track_record_months=12,  # 12 months (1 year)
        )

        assert regime is not None
        assert mc_result is not None
        assert guard_result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
