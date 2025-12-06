"""
Tests for Monte Carlo Simulation Module

UPGRADE-015 Phase 9: Backtesting Robustness

Tests cover:
- Bootstrap methods
- Simulation execution
- Statistics calculation
- Specialized simulations
"""

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtesting.monte_carlo import (
    BootstrapMethod,
    MonteCarloConfig,
    MonteCarloResult,
    MonteCarloSimulator,
    SimulationPath,
    create_monte_carlo_simulator,
)


class TestMonteCarloConfig:
    """Test MonteCarloConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MonteCarloConfig()

        assert config.num_simulations == 10000
        assert config.num_periods == 252
        assert config.initial_equity == 100000.0
        assert config.bootstrap_method == BootstrapMethod.BLOCK
        assert config.block_size == 20

    def test_custom_config(self):
        """Test custom configuration."""
        config = MonteCarloConfig(
            num_simulations=1000,
            num_periods=126,
            initial_equity=50000.0,
            bootstrap_method=BootstrapMethod.SIMPLE,
        )

        assert config.num_simulations == 1000
        assert config.num_periods == 126
        assert config.initial_equity == 50000.0
        assert config.bootstrap_method == BootstrapMethod.SIMPLE


class TestSimulationPath:
    """Test SimulationPath dataclass."""

    def test_path_creation(self):
        """Test creating a simulation path."""
        path = SimulationPath(
            path_id=0,
            returns=[0.01, 0.02, -0.01],
            cumulative_return=0.02,
            max_drawdown=0.01,
            sharpe_ratio=1.5,
            final_equity=102000.0,
        )

        assert path.path_id == 0
        assert len(path.returns) == 3
        assert path.final_equity == 102000.0

    def test_path_to_dict(self):
        """Test converting path to dictionary."""
        path = SimulationPath(
            path_id=0,
            returns=[0.01],
            cumulative_return=0.01,
            max_drawdown=0.005,
            sharpe_ratio=1.0,
            final_equity=101000.0,
        )

        data = path.to_dict()

        assert data["path_id"] == 0
        assert data["cumulative_return"] == 0.01
        assert data["sharpe_ratio"] == 1.0


class TestMonteCarloResult:
    """Test MonteCarloResult dataclass."""

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = MonteCarloResult(
            num_simulations=1000,
            num_periods=252,
            initial_equity=100000.0,
            mean_return=0.10,
            var_95=-0.15,
            prob_profit=0.75,
        )

        data = result.to_dict()

        assert data["num_simulations"] == 1000
        assert data["mean_return"] == 0.10
        assert data["var_95"] == -0.15
        assert data["prob_profit"] == 0.75


class TestMonteCarloSimulator:
    """Test MonteCarloSimulator class."""

    def test_initialization(self):
        """Test simulator initialization."""
        simulator = MonteCarloSimulator()

        assert simulator.config.num_simulations == 10000

    def test_custom_config_initialization(self):
        """Test simulator with custom config."""
        config = MonteCarloConfig(num_simulations=500)
        simulator = MonteCarloSimulator(config)

        assert simulator.config.num_simulations == 500

    def test_simple_bootstrap(self):
        """Test simple bootstrap resampling."""
        config = MonteCarloConfig(
            bootstrap_method=BootstrapMethod.SIMPLE,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config)

        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
        resampled = simulator._simple_bootstrap(returns, 10)

        assert len(resampled) == 10
        # All values should be from original
        assert all(r in returns for r in resampled)

    def test_block_bootstrap(self):
        """Test block bootstrap resampling."""
        config = MonteCarloConfig(
            bootstrap_method=BootstrapMethod.BLOCK,
            block_size=5,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config)

        returns = np.array([0.01 * i for i in range(20)])
        resampled = simulator._block_bootstrap(returns, 15)

        assert len(resampled) == 15

    def test_stationary_bootstrap(self):
        """Test stationary bootstrap resampling."""
        config = MonteCarloConfig(
            bootstrap_method=BootstrapMethod.STATIONARY,
            block_size=5,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config)

        returns = np.array([0.01 * i for i in range(20)])
        resampled = simulator._stationary_bootstrap(returns, 15)

        assert len(resampled) == 15

    def test_simulate_empty_returns(self):
        """Test simulation with empty returns."""
        simulator = MonteCarloSimulator()
        result = simulator.simulate([])

        assert result.num_simulations == 0
        assert result.num_periods == 0

    def test_simulate_insufficient_returns(self):
        """Test simulation with too few returns."""
        simulator = MonteCarloSimulator()
        result = simulator.simulate([0.01] * 5)  # Less than 10

        assert result.num_simulations == 0

    def test_simulate_with_returns(self):
        """Test simulation with valid returns."""
        config = MonteCarloConfig(
            num_simulations=100,  # Small for testing
            num_periods=50,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config)

        # Generate synthetic returns
        np.random.seed(42)
        returns = list(np.random.normal(0.0005, 0.02, 200))

        result = simulator.simulate(returns)

        assert result.num_simulations == 100
        assert result.num_periods == 50
        assert result.mean_return is not None
        assert result.var_95 is not None
        assert result.prob_profit >= 0 and result.prob_profit <= 1

    def test_simulate_with_paths(self):
        """Test simulation storing paths."""
        config = MonteCarloConfig(
            num_simulations=10,
            num_periods=20,
            store_paths=True,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config)

        returns = list(np.random.normal(0.001, 0.01, 50))
        result = simulator.simulate(returns)

        assert len(result.paths) == 10
        assert result.paths[0].path_id == 0
        assert len(result.paths[0].returns) == 20

    def test_calculate_path_metrics(self):
        """Test path metrics calculation."""
        simulator = MonteCarloSimulator()

        returns = np.array([0.01, 0.02, -0.01, 0.01, -0.005])
        cum_ret, max_dd, sharpe, final = simulator._calculate_path_metrics(returns, 100000.0)

        assert cum_ret > 0  # Net positive returns
        assert max_dd >= 0  # Drawdown is non-negative
        assert final > 100000  # Ended higher

    def test_statistics_calculation(self):
        """Test statistics calculation."""
        config = MonteCarloConfig(
            num_simulations=1000,
            num_periods=100,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config)

        returns = list(np.random.normal(0.001, 0.02, 500))
        result = simulator.simulate(returns)

        # Check percentiles exist
        assert 5 in result.return_percentiles
        assert 50 in result.return_percentiles
        assert 95 in result.return_percentiles

        # Median should be between 5th and 95th percentile
        assert result.return_percentiles[5] <= result.median_return
        assert result.median_return <= result.return_percentiles[95]

    def test_var_cvar_calculation(self):
        """Test VaR and CVaR calculation."""
        config = MonteCarloConfig(
            num_simulations=1000,
            num_periods=100,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config)

        returns = list(np.random.normal(0.001, 0.02, 500))
        result = simulator.simulate(returns)

        # VaR 99 should be worse than VaR 95
        assert result.var_99 <= result.var_95

        # CVaR should be worse than VaR (conditional on tail)
        assert result.cvar_95 <= result.var_95
        assert result.cvar_99 <= result.var_99

    def test_simulate_equity_paths(self):
        """Test equity path generation."""
        config = MonteCarloConfig(
            num_periods=50,
            initial_equity=100000.0,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config)

        returns = list(np.random.normal(0.001, 0.01, 100))
        paths = simulator.simulate_equity_paths(returns, num_paths=5)

        assert len(paths) == 5
        # Each path should have num_periods + 1 points (including initial)
        assert len(paths[0]) == 51
        # Should start at initial equity
        assert paths[0][0] == 100000.0

    def test_simulate_with_regime(self):
        """Test regime-based simulation."""
        config = MonteCarloConfig(
            num_simulations=50,
            num_periods=100,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config)

        returns_by_regime = {
            "bull": [0.01, 0.02, 0.015, 0.005],
            "bear": [-0.01, -0.02, -0.015, 0.005],
            "neutral": [0.001, -0.001, 0.002, -0.002],
        }
        regime_probs = {
            "bull": 0.4,
            "bear": 0.2,
            "neutral": 0.4,
        }

        result = simulator.simulate_with_regime(returns_by_regime, regime_probs)

        assert result.num_simulations > 0

    def test_calculate_required_capital(self):
        """Test required capital calculation."""
        config = MonteCarloConfig(
            num_simulations=100,
            num_periods=50,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config)

        # High variance returns
        returns = list(np.random.normal(0.001, 0.05, 200))
        multiplier = simulator.calculate_required_capital(returns, target_survival_prob=0.95, max_loss_pct=0.50)

        assert multiplier >= 1.0


class TestCreateMonteCarloSimulator:
    """Test factory function."""

    def test_create_with_defaults(self):
        """Test creating simulator with defaults."""
        simulator = create_monte_carlo_simulator()

        assert simulator.config.num_simulations == 10000
        assert simulator.config.bootstrap_method == BootstrapMethod.BLOCK

    def test_create_with_custom_settings(self):
        """Test creating simulator with custom settings."""
        simulator = create_monte_carlo_simulator(
            num_simulations=500,
            num_periods=126,
            bootstrap_method="simple",
            random_seed=123,
        )

        assert simulator.config.num_simulations == 500
        assert simulator.config.num_periods == 126
        assert simulator.config.bootstrap_method == BootstrapMethod.SIMPLE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
