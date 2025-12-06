"""
Monte Carlo Stress Tester

Implements Monte Carlo simulation for portfolio stress testing:
- 1000+ scenario simulation
- TGARCH volatility modeling
- Probability of ruin calculation
- Equity curve distributions
- Historical stress scenarios

Part of UPGRADE-010 Sprint 4: Risk & Execution.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from models.exceptions import DataMissingError, StrategyExecutionError
from models.tgarch import TGARCHModel


logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of stress scenarios."""

    NORMAL = "normal"  # Normal market conditions
    HIGH_VOLATILITY = "high_volatility"  # Elevated volatility
    CRASH_2008 = "crash_2008"  # 2008 financial crisis
    COVID_CRASH = "covid_crash"  # 2020 COVID crash
    FLASH_CRASH = "flash_crash"  # Flash crash scenario
    BLACK_MONDAY = "black_monday"  # 1987 Black Monday
    CUSTOM = "custom"  # User-defined scenario


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""

    num_simulations: int = 1000  # Number of paths to simulate
    time_horizon_days: int = 21  # 1 month default
    confidence_levels: list[float] = field(default_factory=lambda: [0.95, 0.99])
    use_tgarch: bool = True  # Use TGARCH for volatility
    ruin_threshold_pct: float = 0.25  # Portfolio loss threshold for "ruin"
    seed: int | None = None  # Random seed for reproducibility
    annual_trading_days: int = 252

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_simulations": self.num_simulations,
            "time_horizon_days": self.time_horizon_days,
            "confidence_levels": self.confidence_levels,
            "use_tgarch": self.use_tgarch,
            "ruin_threshold_pct": self.ruin_threshold_pct,
            "seed": self.seed,
        }


@dataclass
class StressScenario:
    """Definition of a stress scenario."""

    name: str
    scenario_type: ScenarioType
    volatility_multiplier: float  # Multiply normal volatility by this
    mean_return_override: float | None = None  # Override mean return
    correlation_shock: float = 0.0  # Increase correlations by this
    max_daily_move: float | None = None  # Cap daily moves
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "scenario_type": self.scenario_type.value,
            "volatility_multiplier": self.volatility_multiplier,
            "mean_return_override": self.mean_return_override,
            "correlation_shock": self.correlation_shock,
            "max_daily_move": self.max_daily_move,
            "description": self.description,
        }


# Pre-defined stress scenarios based on historical events
STRESS_SCENARIOS = {
    "crash_2008": StressScenario(
        name="2008 Financial Crisis",
        scenario_type=ScenarioType.CRASH_2008,
        volatility_multiplier=3.0,
        mean_return_override=-0.003,  # -0.3% daily
        correlation_shock=0.3,
        description="Lehman collapse and credit crisis",
    ),
    "covid_crash": StressScenario(
        name="COVID-19 Crash (Mar 2020)",
        scenario_type=ScenarioType.COVID_CRASH,
        volatility_multiplier=4.0,
        mean_return_override=-0.004,  # -0.4% daily
        max_daily_move=0.12,  # 12% daily limit
        description="Pandemic market crash",
    ),
    "flash_crash": StressScenario(
        name="Flash Crash",
        scenario_type=ScenarioType.FLASH_CRASH,
        volatility_multiplier=5.0,
        mean_return_override=-0.01,  # -1% daily during flash
        max_daily_move=0.10,
        description="Sudden market flash crash",
    ),
    "black_monday": StressScenario(
        name="Black Monday (1987)",
        scenario_type=ScenarioType.BLACK_MONDAY,
        volatility_multiplier=6.0,
        mean_return_override=-0.08,  # -8% single day
        description="1987 Black Monday crash",
    ),
    "high_vol": StressScenario(
        name="High Volatility Regime",
        scenario_type=ScenarioType.HIGH_VOLATILITY,
        volatility_multiplier=2.0,
        description="Sustained high volatility period",
    ),
}


@dataclass
class SimulationResult:
    """Result of Monte Carlo simulation."""

    final_values: np.ndarray  # Distribution of final portfolio values
    paths: np.ndarray  # All simulation paths (num_sims x time_horizon)
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # 95% CVaR (expected shortfall)
    cvar_99: float  # 99% CVaR
    probability_of_ruin: float  # P(portfolio < threshold)
    expected_return: float  # Mean final return
    return_std: float  # Std of final returns
    max_drawdown_95: float  # 95th percentile max drawdown
    max_drawdown_median: float  # Median max drawdown
    percentiles: dict[int, float]  # Percentile values of final portfolio
    equity_curve_percentiles: dict[int, np.ndarray]  # Percentile equity curves
    simulation_time_seconds: float
    config: SimulationConfig
    scenario: StressScenario | None = None
    initial_value: float = 100000.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "probability_of_ruin": self.probability_of_ruin,
            "expected_return": self.expected_return,
            "return_std": self.return_std,
            "max_drawdown_95": self.max_drawdown_95,
            "max_drawdown_median": self.max_drawdown_median,
            "percentiles": self.percentiles,
            "simulation_time_seconds": self.simulation_time_seconds,
            "num_simulations": self.config.num_simulations,
            "time_horizon_days": self.config.time_horizon_days,
            "scenario": self.scenario.to_dict() if self.scenario else None,
        }


@dataclass
class DrawdownAnalysis:
    """Maximum drawdown analysis."""

    max_drawdowns: np.ndarray  # Max DD for each simulation
    mean_max_drawdown: float
    median_max_drawdown: float
    percentile_95_max_drawdown: float
    worst_drawdown: float
    recovery_times: np.ndarray  # Time to recover from max DD

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_max_drawdown": self.mean_max_drawdown,
            "median_max_drawdown": self.median_max_drawdown,
            "percentile_95_max_drawdown": self.percentile_95_max_drawdown,
            "worst_drawdown": self.worst_drawdown,
            "mean_recovery_time": float(np.mean(self.recovery_times)),
        }


class MonteCarloStressTester:
    """
    Monte Carlo stress testing for portfolio.

    Simulates portfolio paths under various market conditions
    using TGARCH volatility dynamics.
    """

    def __init__(self, config: SimulationConfig | None = None):
        """
        Initialize stress tester.

        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        self.tgarch = TGARCHModel()
        self._last_result: SimulationResult | None = None

    def run_simulation(
        self,
        portfolio_value: float,
        returns_history: np.ndarray,
        scenario: StressScenario | None = None,
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Args:
            portfolio_value: Starting portfolio value
            returns_history: Historical daily returns for calibration
            scenario: Optional stress scenario to apply

        Returns:
            SimulationResult with simulation outcomes
        """
        start_time = time.time()

        # Set seed if specified
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        returns_history = np.asarray(returns_history)

        # Fit TGARCH if enabled and sufficient data
        if self.config.use_tgarch and len(returns_history) >= 30:
            try:
                fit_result = self.tgarch.fit(returns_history)
                use_tgarch = True
            except Exception as e:
                logger.warning(f"TGARCH fitting failed: {e}, using simple model")
                use_tgarch = False
        else:
            use_tgarch = False

        # Calculate base parameters
        mean_return = np.mean(returns_history)
        std_return = np.std(returns_history)

        # Apply scenario adjustments
        if scenario:
            std_return *= scenario.volatility_multiplier
            if scenario.mean_return_override is not None:
                mean_return = scenario.mean_return_override

        # Run simulations
        n_sims = self.config.num_simulations
        n_days = self.config.time_horizon_days

        if use_tgarch:
            # Simulate using TGARCH dynamics
            paths = self._simulate_tgarch(portfolio_value, n_sims, n_days, mean_return, scenario)
        else:
            # Simple geometric Brownian motion
            paths = self._simulate_gbm(portfolio_value, mean_return, std_return, n_sims, n_days, scenario)

        # Calculate statistics
        final_values = paths[:, -1]
        returns = (final_values - portfolio_value) / portfolio_value

        # VaR and CVaR
        sorted_returns = np.sort(returns)
        var_95 = -np.percentile(sorted_returns, 5) * portfolio_value
        var_99 = -np.percentile(sorted_returns, 1) * portfolio_value

        cvar_95_idx = int(n_sims * 0.05)
        cvar_99_idx = int(n_sims * 0.01)
        cvar_95 = -np.mean(sorted_returns[: max(1, cvar_95_idx)]) * portfolio_value
        cvar_99 = -np.mean(sorted_returns[: max(1, cvar_99_idx)]) * portfolio_value

        # Probability of ruin
        ruin_threshold = portfolio_value * (1 - self.config.ruin_threshold_pct)
        probability_of_ruin = np.mean(final_values < ruin_threshold)

        # Max drawdown analysis
        max_drawdowns = self._calculate_max_drawdowns(paths)
        max_dd_95 = np.percentile(max_drawdowns, 95)
        max_dd_median = np.median(max_drawdowns)

        # Percentiles
        percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            percentiles[p] = float(np.percentile(final_values, p))

        # Equity curve percentiles
        equity_percentiles = {}
        for p in [5, 25, 50, 75, 95]:
            equity_percentiles[p] = np.percentile(paths, p, axis=0)

        sim_time = time.time() - start_time

        result = SimulationResult(
            final_values=final_values,
            paths=paths,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            probability_of_ruin=probability_of_ruin,
            expected_return=float(np.mean(returns)),
            return_std=float(np.std(returns)),
            max_drawdown_95=max_dd_95,
            max_drawdown_median=max_dd_median,
            percentiles=percentiles,
            equity_curve_percentiles=equity_percentiles,
            simulation_time_seconds=sim_time,
            config=self.config,
            scenario=scenario,
            initial_value=portfolio_value,
        )

        self._last_result = result
        return result

    def stress_test(
        self,
        portfolio_value: float,
        returns_history: np.ndarray,
        scenario_name: str,
    ) -> SimulationResult:
        """
        Run stress test with pre-defined historical scenario.

        Args:
            portfolio_value: Starting portfolio value
            returns_history: Historical returns for calibration
            scenario_name: Name of scenario (e.g., "crash_2008", "covid_crash")

        Returns:
            SimulationResult under stress conditions
        """
        if scenario_name not in STRESS_SCENARIOS:
            available = ", ".join(STRESS_SCENARIOS.keys())
            raise DataMissingError(
                field="scenario_name",
                source=f"STRESS_SCENARIOS. Available: {available}",
            )

        scenario = STRESS_SCENARIOS[scenario_name]
        return self.run_simulation(portfolio_value, returns_history, scenario)

    def run_scenario_comparison(
        self,
        portfolio_value: float,
        returns_history: np.ndarray,
        scenarios: list[str] | None = None,
    ) -> dict[str, SimulationResult]:
        """
        Run multiple scenarios for comparison.

        Args:
            portfolio_value: Starting portfolio value
            returns_history: Historical returns
            scenarios: List of scenario names (default: all)

        Returns:
            Dictionary of scenario name to result
        """
        if scenarios is None:
            scenarios = list(STRESS_SCENARIOS.keys())

        results = {}

        # Run base case (no scenario)
        results["base"] = self.run_simulation(portfolio_value, returns_history)

        # Run each scenario
        for scenario_name in scenarios:
            results[scenario_name] = self.stress_test(portfolio_value, returns_history, scenario_name)

        return results

    def _simulate_tgarch(
        self,
        initial_value: float,
        n_sims: int,
        n_days: int,
        mean_return: float,
        scenario: StressScenario | None = None,
    ) -> np.ndarray:
        """Simulate paths using TGARCH volatility dynamics."""
        # Use TGARCH to simulate returns and volatilities
        returns, volatilities = self.tgarch.simulate(n_days, n_sims)

        # Apply scenario adjustments
        if scenario:
            returns *= scenario.volatility_multiplier
            if scenario.mean_return_override is not None:
                returns += scenario.mean_return_override - np.mean(returns)
            if scenario.max_daily_move is not None:
                returns = np.clip(
                    returns,
                    -scenario.max_daily_move,
                    scenario.max_daily_move,
                )

        # Convert returns to prices
        paths = np.zeros((n_sims, n_days + 1))
        paths[:, 0] = initial_value

        for t in range(n_days):
            paths[:, t + 1] = paths[:, t] * (1 + returns[:, t])

        return paths

    def _simulate_gbm(
        self,
        initial_value: float,
        mean_return: float,
        std_return: float,
        n_sims: int,
        n_days: int,
        scenario: StressScenario | None = None,
    ) -> np.ndarray:
        """Simulate paths using geometric Brownian motion."""
        # Generate random returns
        z = np.random.standard_normal((n_sims, n_days))
        returns = mean_return + std_return * z

        # Apply scenario adjustments
        if scenario:
            returns *= scenario.volatility_multiplier
            if scenario.mean_return_override is not None:
                returns += scenario.mean_return_override - mean_return
            if scenario.max_daily_move is not None:
                returns = np.clip(
                    returns,
                    -scenario.max_daily_move,
                    scenario.max_daily_move,
                )

        # Convert to prices
        paths = np.zeros((n_sims, n_days + 1))
        paths[:, 0] = initial_value

        for t in range(n_days):
            paths[:, t + 1] = paths[:, t] * (1 + returns[:, t])

        return paths

    def _calculate_max_drawdowns(self, paths: np.ndarray) -> np.ndarray:
        """Calculate maximum drawdown for each simulation path."""
        n_sims = paths.shape[0]
        max_drawdowns = np.zeros(n_sims)

        for i in range(n_sims):
            path = paths[i]
            peak = path[0]
            max_dd = 0

            for value in path:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd

            max_drawdowns[i] = max_dd

        return max_drawdowns

    def analyze_drawdowns(
        self,
        result: SimulationResult | None = None,
    ) -> DrawdownAnalysis:
        """
        Detailed drawdown analysis.

        Args:
            result: Simulation result (uses last if None)

        Returns:
            DrawdownAnalysis with detailed metrics
        """
        result = result or self._last_result
        if result is None:
            raise StrategyExecutionError(
                strategy_name="MonteCarloStressTester",
                phase="analyze_drawdowns",
                reason="No simulation result available. Run run_simulation() first.",
            )

        max_drawdowns = self._calculate_max_drawdowns(result.paths)

        # Calculate recovery times (simplified)
        recovery_times = np.zeros(len(max_drawdowns))

        return DrawdownAnalysis(
            max_drawdowns=max_drawdowns,
            mean_max_drawdown=float(np.mean(max_drawdowns)),
            median_max_drawdown=float(np.median(max_drawdowns)),
            percentile_95_max_drawdown=float(np.percentile(max_drawdowns, 95)),
            worst_drawdown=float(np.max(max_drawdowns)),
            recovery_times=recovery_times,
        )

    def get_equity_curve_distribution(
        self,
        result: SimulationResult | None = None,
        percentiles: list[int] = [5, 25, 50, 75, 95],
    ) -> dict[int, np.ndarray]:
        """
        Get percentile equity curves for visualization.

        Args:
            result: Simulation result (uses last if None)
            percentiles: List of percentiles to calculate

        Returns:
            Dictionary of percentile to equity curve array
        """
        result = result or self._last_result
        if result is None:
            raise StrategyExecutionError(
                strategy_name="MonteCarloStressTester",
                phase="get_equity_curve_distribution",
                reason="No simulation result available. Run run_simulation() first.",
            )

        curves = {}
        for p in percentiles:
            curves[p] = np.percentile(result.paths, p, axis=0)

        return curves

    def probability_of_target(
        self,
        target_return: float,
        result: SimulationResult | None = None,
    ) -> float:
        """
        Calculate probability of achieving target return.

        Args:
            target_return: Target return (e.g., 0.10 for 10%)
            result: Simulation result (uses last if None)

        Returns:
            Probability of achieving target
        """
        result = result or self._last_result
        if result is None:
            raise StrategyExecutionError(
                strategy_name="MonteCarloStressTester",
                phase="probability_of_target",
                reason="No simulation result available. Run run_simulation() first.",
            )

        target_value = result.initial_value * (1 + target_return)
        return float(np.mean(result.final_values >= target_value))

    def get_summary(self) -> dict[str, Any]:
        """Get tester summary."""
        return {
            "config": self.config.to_dict(),
            "tgarch_fitted": self.tgarch.is_fitted,
            "available_scenarios": list(STRESS_SCENARIOS.keys()),
            "last_result": self._last_result.to_dict() if self._last_result else None,
        }


def create_monte_carlo_tester(
    num_simulations: int = 1000,
    time_horizon_days: int = 21,
    use_tgarch: bool = True,
) -> MonteCarloStressTester:
    """Factory function to create Monte Carlo stress tester."""
    config = SimulationConfig(
        num_simulations=num_simulations,
        time_horizon_days=time_horizon_days,
        use_tgarch=use_tgarch,
    )
    return MonteCarloStressTester(config=config)


__all__ = [
    "STRESS_SCENARIOS",
    "DrawdownAnalysis",
    "MonteCarloStressTester",
    "ScenarioType",
    "SimulationConfig",
    "SimulationResult",
    "StressScenario",
    "create_monte_carlo_tester",
]
