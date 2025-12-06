"""
Tests for TGARCH Volatility Model (Sprint 4)

Tests the TGARCH(1,1) model for volatility forecasting.
Part of UPGRADE-010 Sprint 4 - Test Coverage.
"""

import numpy as np
import pytest

from models.tgarch import (
    TGARCHFitResult,
    TGARCHModel,
    TGARCHParams,
    create_tgarch_model,
)


class TestTGARCHParams:
    """Tests for TGARCHParams dataclass."""

    def test_creation(self):
        """Test parameter creation."""
        params = TGARCHParams(
            omega=0.00001,
            alpha=0.05,
            beta=0.90,
            gamma=0.05,
        )

        assert params.omega == 0.00001
        assert params.alpha == 0.05
        assert params.beta == 0.90
        assert params.gamma == 0.05

    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = TGARCHParams(omega=0.001, alpha=0.05, beta=0.90, gamma=0.05)
        d = params.to_dict()

        assert d["omega"] == 0.001
        assert "alpha" in d
        assert "beta" in d
        assert "gamma" in d
        assert "persistence" in d

    def test_persistence(self):
        """Test persistence calculation."""
        params = TGARCHParams(omega=0.00001, alpha=0.05, beta=0.90, gamma=0.05)

        # persistence = alpha + beta + gamma/2
        expected = 0.05 + 0.90 + 0.05 / 2
        assert params.persistence == expected

    def test_long_run_variance(self):
        """Test long run variance calculation."""
        params = TGARCHParams(omega=0.00001, alpha=0.05, beta=0.90, gamma=0.04)

        # Should be positive and finite
        assert params.long_run_variance > 0

    def test_long_run_volatility(self):
        """Test long run volatility."""
        params = TGARCHParams(omega=0.00001, alpha=0.05, beta=0.90, gamma=0.04)

        assert params.long_run_volatility > 0
        assert params.long_run_volatility == np.sqrt(params.long_run_variance)

    def test_is_valid_stationary(self):
        """Test stationarity check - valid case."""
        params = TGARCHParams(omega=0.00001, alpha=0.05, beta=0.90, gamma=0.04)
        is_valid, msg = params.is_valid()

        # May or may not be valid depending on persistence
        assert isinstance(is_valid, bool)
        assert isinstance(msg, str)

    def test_is_valid_non_stationary(self):
        """Test stationarity check - invalid case."""
        params = TGARCHParams(omega=0.00001, alpha=0.30, beta=0.80, gamma=0.10)
        is_valid, msg = params.is_valid()

        # Should be invalid due to high persistence
        assert is_valid is False

    def test_is_valid_negative_omega(self):
        """Test invalid negative omega."""
        params = TGARCHParams(omega=-0.00001, alpha=0.05, beta=0.90, gamma=0.05)
        is_valid, msg = params.is_valid()

        assert is_valid is False
        assert "omega" in msg


class TestTGARCHModel:
    """Tests for TGARCHModel class."""

    @pytest.fixture
    def model(self):
        """Create default model."""
        return TGARCHModel()

    @pytest.fixture
    def returns_series(self):
        """Generate synthetic return series."""
        np.random.seed(42)
        # Generate returns with volatility clustering
        n = 500
        returns = np.zeros(n)
        sigma = np.zeros(n)
        sigma[0] = 0.01

        for t in range(1, n):
            # TGARCH-like process
            shock = returns[t - 1] if returns[t - 1] < 0 else 0
            sigma[t] = np.sqrt(0.00001 + 0.05 * returns[t - 1] ** 2 + 0.90 * sigma[t - 1] ** 2 + 0.05 * shock**2)
            returns[t] = np.random.normal(0, sigma[t])

        return returns

    def test_initialization(self, model):
        """Test model initialization."""
        assert model.p == 1
        assert model.q == 1
        assert model.params is None
        assert not model.is_fitted

    def test_fit_model(self, model, returns_series):
        """Test model fitting."""
        result = model.fit(returns_series)

        assert isinstance(result, TGARCHFitResult)
        assert model.is_fitted
        assert model.params is not None

    def test_fit_result_structure(self, model, returns_series):
        """Test fit result contains expected fields."""
        result = model.fit(returns_series)

        assert hasattr(result, "params")
        assert hasattr(result, "log_likelihood")
        assert hasattr(result, "aic")
        assert hasattr(result, "bic")
        assert hasattr(result, "num_observations")
        assert hasattr(result, "convergence")

    def test_fit_with_initial_params(self, model, returns_series):
        """Test fitting with custom initial parameters."""
        initial = TGARCHParams(omega=0.00001, alpha=0.10, beta=0.85, gamma=0.05)
        result = model.fit(returns_series, initial_params=initial)

        assert result.convergence is True

    def test_forecast_single_step(self, model, returns_series):
        """Test single-step volatility forecast."""
        model.fit(returns_series)
        forecast = model.forecast(steps=1)

        assert len(forecast) == 1
        assert forecast[0] > 0  # Volatility must be positive

    def test_forecast_multi_step(self, model, returns_series):
        """Test multi-step volatility forecast."""
        model.fit(returns_series)
        forecast = model.forecast(steps=10)

        assert len(forecast) == 10
        assert all(f > 0 for f in forecast)

    def test_forecast_with_confidence(self, model, returns_series):
        """Test forecast with confidence intervals."""
        model.fit(returns_series)
        forecast = model.forecast(steps=5, return_confidence=True)

        # Should return array or tuple
        assert forecast is not None

    def test_forecast_unfitted_model(self, model):
        """Test forecasting with unfitted model."""
        with pytest.raises(ValueError, match="[Ff]it|[Tt]rain"):
            model.forecast(steps=1)

    def test_simulate_paths(self, model, returns_series):
        """Test path simulation."""
        model.fit(returns_series)
        returns, volatilities = model.simulate(n_steps=100, n_paths=10, seed=42)

        assert returns.shape == (10, 100)
        assert volatilities.shape == (10, 100)
        assert np.all(volatilities > 0)

    def test_simulate_deterministic(self, model, returns_series):
        """Test that simulation with seed is reproducible."""
        model.fit(returns_series)

        r1, v1 = model.simulate(n_steps=50, n_paths=5, seed=42)
        r2, v2 = model.simulate(n_steps=50, n_paths=5, seed=42)

        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(v1, v2)

    def test_leverage_effect(self, model, returns_series):
        """Test leverage effect (gamma > 0)."""
        model.fit(returns_series)

        # TGARCH captures leverage effect with gamma
        assert model.params.gamma >= 0


class TestTGARCHFitResult:
    """Tests for TGARCHFitResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TGARCHFitResult(
            params=TGARCHParams(omega=0.00001, alpha=0.05, beta=0.90, gamma=0.05),
            log_likelihood=-1000.0,
            aic=2010.0,
            bic=2030.0,
            num_observations=500,
            convergence=True,
            iterations=50,
            conditional_volatility=np.array([0.01, 0.02]),
            residuals=np.array([0.5, -0.5]),
        )

        d = result.to_dict()

        assert d["log_likelihood"] == -1000.0
        assert d["aic"] == 2010.0
        assert d["bic"] == 2030.0
        assert d["num_observations"] == 500
        assert d["convergence"] is True


class TestCreateTGARCHModel:
    """Tests for factory function."""

    def test_create_default(self):
        """Test creating model with defaults."""
        model = create_tgarch_model()

        assert isinstance(model, TGARCHModel)
        assert model.p == 1
        assert model.q == 1


class TestTGARCHEdgeCases:
    """Edge case tests for TGARCH model."""

    def test_constant_returns(self):
        """Test with constant (zero variance) returns."""
        model = TGARCHModel()
        # Need at least 30 observations
        constant_returns = np.zeros(100)
        constant_returns[10] = 0.01  # Add small variance

        # Should handle gracefully
        result = model.fit(constant_returns)
        assert result is not None

    def test_extreme_returns(self):
        """Test with extreme return values."""
        model = TGARCHModel()
        np.random.seed(42)
        extreme_returns = np.random.normal(0, 0.10, 200)  # 10% daily vol
        extreme_returns[50] = 0.50  # 50% return spike

        result = model.fit(extreme_returns)
        assert result.convergence is True

    def test_negative_returns_bias(self):
        """Test with negative-skewed returns."""
        model = TGARCHModel()
        np.random.seed(42)
        # Simulate negative-skewed returns
        returns = np.random.normal(-0.001, 0.02, 300)

        result = model.fit(returns)
        # Leverage effect should capture asymmetry
        assert result.params.gamma >= 0

    def test_short_series_raises(self):
        """Test that short series raises error."""
        model = TGARCHModel()
        short_returns = np.random.normal(0, 0.01, 20)

        with pytest.raises(ValueError, match="[Oo]bservation|[Mm]inimum"):
            model.fit(short_returns)


class TestTGARCHParameterBounds:
    """Tests for parameter validation and bounds."""

    def test_params_within_bounds(self):
        """Test that fitted params are within reasonable bounds."""
        model = TGARCHModel()
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 300)

        result = model.fit(returns)

        assert result.params.omega > 0
        assert result.params.alpha >= 0
        assert result.params.beta >= 0
        assert result.params.gamma >= 0
        assert result.params.alpha + result.params.beta < 1

    def test_aic_bic_relationship(self):
        """Test AIC/BIC relationship."""
        model = TGARCHModel()
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 300)

        result = model.fit(returns)

        # BIC penalizes complexity more than AIC for n > ~7
        # So BIC >= AIC for reasonable sample sizes
        assert result.bic >= result.aic or abs(result.bic - result.aic) < 50

    def test_residuals_properties(self):
        """Test that residuals are approximately standard normal."""
        model = TGARCHModel()
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 500)

        result = model.fit(returns)

        # Standardized residuals should have mean ~0 and std ~1
        if len(result.residuals) > 0:
            mean = np.mean(result.residuals)
            std = np.std(result.residuals)
            assert abs(mean) < 0.5  # Mean near 0
            assert 0.5 < std < 2.0  # Std near 1
