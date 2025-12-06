"""
Tests for Monte Carlo Stress Tester (Sprint 4)

Tests the Sprint 4 Monte Carlo stress testing module.
Part of UPGRADE-010 Sprint 4 - Test Coverage.
"""

import numpy as np
import pytest

from models.monte_carlo import (
    STRESS_SCENARIOS,
    DrawdownAnalysis,
    MonteCarloStressTester,
    ScenarioType,
    SimulationConfig,
    SimulationResult,
    StressScenario,
    create_monte_carlo_tester,
)


class TestScenarioTypeSprint4:
    """Tests for ScenarioType enum."""

    def test_types_exist(self):
        """Test all scenario types exist."""
        assert ScenarioType.NORMAL.value == "normal"
        assert ScenarioType.HIGH_VOLATILITY.value == "high_volatility"
        assert ScenarioType.CRASH_2008.value == "crash_2008"
        assert ScenarioType.COVID_CRASH.value == "covid_crash"
        assert ScenarioType.FLASH_CRASH.value == "flash_crash"


class TestStressScenarioSprint4:
    """Tests for StressScenario dataclass."""

    def test_creation(self):
        """Test scenario creation."""
        scenario = StressScenario(
            name="Test Scenario",
            scenario_type=ScenarioType.HIGH_VOLATILITY,
            volatility_multiplier=2.0,
        )

        assert scenario.name == "Test Scenario"
        assert scenario.volatility_multiplier == 2.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scenario = StressScenario(
            name="Custom",
            scenario_type=ScenarioType.CUSTOM,
            volatility_multiplier=3.0,
            mean_return_override=-0.005,
        )

        d = scenario.to_dict()

        assert d["name"] == "Custom"
        assert d["scenario_type"] == "custom"
        assert d["volatility_multiplier"] == 3.0


class TestPreDefinedScenarios:
    """Tests for pre-defined stress scenarios."""

    def test_scenarios_defined(self):
        """Test all scenarios are defined."""
        assert "crash_2008" in STRESS_SCENARIOS
        assert "covid_crash" in STRESS_SCENARIOS
        assert "flash_crash" in STRESS_SCENARIOS
        assert "black_monday" in STRESS_SCENARIOS
        assert "high_vol" in STRESS_SCENARIOS

    def test_crash_2008_params(self):
        """Test 2008 crash parameters."""
        scenario = STRESS_SCENARIOS["crash_2008"]

        assert scenario.volatility_multiplier == 3.0
        assert scenario.mean_return_override < 0
        assert scenario.correlation_shock > 0


class TestSimulationConfigSprint4:
    """Tests for SimulationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = SimulationConfig()

        assert config.num_simulations == 1000
        assert config.time_horizon_days == 21
        assert config.use_tgarch is True
        assert config.ruin_threshold_pct == 0.25

    def test_custom_values(self):
        """Test custom configuration."""
        config = SimulationConfig(
            num_simulations=5000,
            time_horizon_days=63,
            use_tgarch=False,
        )

        assert config.num_simulations == 5000
        assert config.time_horizon_days == 63
        assert config.use_tgarch is False


class TestMonteCarloStressTesterSprint4:
    """Tests for MonteCarloStressTester class."""

    @pytest.fixture
    def tester(self):
        """Create default tester."""
        config = SimulationConfig(
            num_simulations=100,
            time_horizon_days=10,
            seed=42,
        )
        return MonteCarloStressTester(config)

    @pytest.fixture
    def returns_history(self):
        """Create sample returns history."""
        np.random.seed(42)
        return np.random.normal(0.0005, 0.015, 60)

    def test_initialization(self, tester):
        """Test tester initialization."""
        assert tester.config is not None
        assert tester.config.num_simulations == 100

    def test_run_simulation_basic(self, tester, returns_history):
        """Test basic simulation run."""
        result = tester.run_simulation(
            portfolio_value=100000.0,
            returns_history=returns_history,
        )

        assert isinstance(result, SimulationResult)
        assert result.initial_value == 100000.0
        assert result.var_95 > 0
        assert result.var_99 > result.var_95

    def test_simulation_paths_shape(self, tester, returns_history):
        """Test simulation paths have correct shape."""
        result = tester.run_simulation(
            portfolio_value=100000.0,
            returns_history=returns_history,
        )

        n_sims = tester.config.num_simulations
        n_days = tester.config.time_horizon_days

        assert result.paths.shape == (n_sims, n_days + 1)
        assert len(result.final_values) == n_sims

    def test_stress_test_crash_2008(self, tester, returns_history):
        """Test 2008 crash stress scenario."""
        result = tester.stress_test(
            portfolio_value=100000.0,
            returns_history=returns_history,
            scenario_name="crash_2008",
        )

        assert result.scenario is not None
        assert result.scenario.name == "2008 Financial Crisis"

    def test_stress_test_invalid_scenario(self, tester, returns_history):
        """Test invalid scenario name."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            tester.stress_test(100000.0, returns_history, "invalid_scenario")

    def test_scenario_comparison(self, tester, returns_history):
        """Test running multiple scenarios."""
        results = tester.run_scenario_comparison(
            portfolio_value=100000.0,
            returns_history=returns_history,
            scenarios=["crash_2008", "high_vol"],
        )

        assert "base" in results
        assert "crash_2008" in results
        assert "high_vol" in results

    def test_analyze_drawdowns(self, tester, returns_history):
        """Test detailed drawdown analysis."""
        tester.run_simulation(100000.0, returns_history)
        analysis = tester.analyze_drawdowns()

        assert isinstance(analysis, DrawdownAnalysis)
        assert analysis.mean_max_drawdown >= 0

    def test_probability_of_target(self, tester, returns_history):
        """Test probability of achieving target return."""
        tester.run_simulation(100000.0, returns_history)

        prob_5pct = tester.probability_of_target(0.05)
        prob_50pct = tester.probability_of_target(0.50)

        assert prob_5pct > prob_50pct
        assert 0 <= prob_5pct <= 1


class TestSimulationResultSprint4:
    """Tests for SimulationResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SimulationConfig()
        result = SimulationResult(
            final_values=np.array([100000, 105000, 95000]),
            paths=np.zeros((3, 22)),
            var_95=3000.0,
            var_99=5000.0,
            cvar_95=4000.0,
            cvar_99=6000.0,
            probability_of_ruin=0.05,
            expected_return=0.02,
            return_std=0.05,
            max_drawdown_95=0.08,
            max_drawdown_median=0.04,
            percentiles={5: 95000, 50: 100000, 95: 105000},
            equity_curve_percentiles={},
            simulation_time_seconds=1.5,
            config=config,
        )

        d = result.to_dict()

        assert d["var_95"] == 3000.0
        assert d["probability_of_ruin"] == 0.05


class TestCreateMonteCarloTesterSprint4:
    """Tests for factory function."""

    def test_create_default(self):
        """Test creating with defaults."""
        tester = create_monte_carlo_tester()

        assert isinstance(tester, MonteCarloStressTester)
        assert tester.config.num_simulations == 1000

    def test_create_with_params(self):
        """Test creating with custom params."""
        tester = create_monte_carlo_tester(
            num_simulations=500,
            time_horizon_days=63,
            use_tgarch=False,
        )

        assert tester.config.num_simulations == 500
        assert tester.config.time_horizon_days == 63
        assert tester.config.use_tgarch is False
