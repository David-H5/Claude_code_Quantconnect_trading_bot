"""
Monte Carlo Simulation Module

UPGRADE-015 Phase 9: Backtesting Robustness

Provides Monte Carlo simulation for strategy validation:
- Bootstrap resampling of returns
- Path-dependent simulations
- Confidence interval estimation
- Risk metric distributions

Features:
- Block bootstrap for autocorrelation
- Parallel simulation support
- Distribution analysis
- VaR/CVaR calculations
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class BootstrapMethod(Enum):
    """Bootstrap resampling methods."""

    SIMPLE = "simple"  # IID resampling
    BLOCK = "block"  # Block bootstrap for autocorrelation
    STATIONARY = "stationary"  # Random block lengths


@dataclass
class SimulationPath:
    """A single Monte Carlo simulation path."""

    path_id: int
    returns: list[float]
    cumulative_return: float
    max_drawdown: float
    sharpe_ratio: float
    final_equity: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path_id": self.path_id,
            "cumulative_return": self.cumulative_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "final_equity": self.final_equity,
        }


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""

    num_simulations: int
    num_periods: int
    initial_equity: float

    # Return statistics
    mean_return: float = 0.0
    median_return: float = 0.0
    std_return: float = 0.0
    min_return: float = 0.0
    max_return: float = 0.0

    # Risk metrics
    var_95: float = 0.0  # Value at Risk 95%
    var_99: float = 0.0  # Value at Risk 99%
    cvar_95: float = 0.0  # Conditional VaR 95%
    cvar_99: float = 0.0  # Conditional VaR 99%

    # Drawdown statistics
    mean_max_drawdown: float = 0.0
    median_max_drawdown: float = 0.0
    worst_drawdown: float = 0.0

    # Sharpe statistics
    mean_sharpe: float = 0.0
    median_sharpe: float = 0.0

    # Probability estimates
    prob_profit: float = 0.0
    prob_loss_10pct: float = 0.0
    prob_loss_20pct: float = 0.0

    # Confidence intervals (95%)
    return_ci_lower: float = 0.0
    return_ci_upper: float = 0.0
    drawdown_ci_lower: float = 0.0
    drawdown_ci_upper: float = 0.0

    # Distribution percentiles
    return_percentiles: dict[int, float] = field(default_factory=dict)

    # Paths (optional, for detailed analysis)
    paths: list[SimulationPath] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_simulations": self.num_simulations,
            "num_periods": self.num_periods,
            "initial_equity": self.initial_equity,
            "mean_return": self.mean_return,
            "median_return": self.median_return,
            "std_return": self.std_return,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "mean_max_drawdown": self.mean_max_drawdown,
            "prob_profit": self.prob_profit,
            "return_ci_lower": self.return_ci_lower,
            "return_ci_upper": self.return_ci_upper,
            "return_percentiles": self.return_percentiles,
        }


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""

    num_simulations: int = 10000
    num_periods: int = 252  # 1 year of trading days
    initial_equity: float = 100000.0
    bootstrap_method: BootstrapMethod = BootstrapMethod.BLOCK
    block_size: int = 20  # For block bootstrap
    random_seed: int | None = None
    store_paths: bool = False  # Store individual paths
    confidence_level: float = 0.95


class MonteCarloSimulator:
    """Monte Carlo simulation for strategy validation."""

    def __init__(
        self,
        config: MonteCarloConfig | None = None,
    ):
        """
        Initialize Monte Carlo simulator.

        Args:
            config: Simulation configuration
        """
        self.config = config or MonteCarloConfig()

        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

    # ==========================================================================
    # Bootstrap Methods
    # ==========================================================================

    def _simple_bootstrap(
        self,
        returns: np.ndarray,
        num_periods: int,
    ) -> np.ndarray:
        """Simple IID bootstrap resampling."""
        indices = np.random.randint(0, len(returns), num_periods)
        return returns[indices]

    def _block_bootstrap(
        self,
        returns: np.ndarray,
        num_periods: int,
    ) -> np.ndarray:
        """Block bootstrap for autocorrelated returns."""
        block_size = self.config.block_size
        num_blocks = (num_periods // block_size) + 1
        result = []

        for _ in range(num_blocks):
            # Random starting point
            start = np.random.randint(0, len(returns) - block_size + 1)
            result.extend(returns[start : start + block_size])

        return np.array(result[:num_periods])

    def _stationary_bootstrap(
        self,
        returns: np.ndarray,
        num_periods: int,
    ) -> np.ndarray:
        """Stationary bootstrap with random block lengths."""
        p = 1.0 / self.config.block_size  # Probability of starting new block
        result = []

        idx = np.random.randint(0, len(returns))

        for _ in range(num_periods):
            result.append(returns[idx])

            # With probability p, start new block
            if np.random.random() < p:
                idx = np.random.randint(0, len(returns))
            else:
                idx = (idx + 1) % len(returns)

        return np.array(result)

    def _resample(
        self,
        returns: np.ndarray,
        num_periods: int,
    ) -> np.ndarray:
        """Resample returns using configured method."""
        if self.config.bootstrap_method == BootstrapMethod.SIMPLE:
            return self._simple_bootstrap(returns, num_periods)
        elif self.config.bootstrap_method == BootstrapMethod.BLOCK:
            return self._block_bootstrap(returns, num_periods)
        else:  # STATIONARY
            return self._stationary_bootstrap(returns, num_periods)

    # ==========================================================================
    # Simulation
    # ==========================================================================

    def simulate(
        self,
        historical_returns: list[float],
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.

        Args:
            historical_returns: List of historical returns

        Returns:
            MonteCarloResult with statistics
        """
        if not historical_returns or len(historical_returns) < 10:
            return MonteCarloResult(
                num_simulations=0,
                num_periods=0,
                initial_equity=self.config.initial_equity,
            )

        returns_array = np.array(historical_returns)
        num_sims = self.config.num_simulations
        num_periods = self.config.num_periods
        initial = self.config.initial_equity

        # Storage for results
        final_returns = []
        max_drawdowns = []
        sharpe_ratios = []
        paths = []

        for i in range(num_sims):
            # Resample returns
            sim_returns = self._resample(returns_array, num_periods)

            # Calculate path metrics
            cum_return, max_dd, sharpe, final_equity = self._calculate_path_metrics(sim_returns, initial)

            final_returns.append(cum_return)
            max_drawdowns.append(max_dd)
            sharpe_ratios.append(sharpe)

            if self.config.store_paths:
                paths.append(
                    SimulationPath(
                        path_id=i,
                        returns=sim_returns.tolist(),
                        cumulative_return=cum_return,
                        max_drawdown=max_dd,
                        sharpe_ratio=sharpe,
                        final_equity=final_equity,
                    )
                )

        # Convert to arrays for calculations
        final_returns = np.array(final_returns)
        max_drawdowns = np.array(max_drawdowns)
        sharpe_ratios = np.array(sharpe_ratios)

        # Calculate statistics
        result = self._calculate_statistics(final_returns, max_drawdowns, sharpe_ratios, initial)
        result.paths = paths

        return result

    def _calculate_path_metrics(
        self,
        returns: np.ndarray,
        initial_equity: float,
    ) -> tuple[float, float, float, float]:
        """Calculate metrics for a single simulation path."""
        # Cumulative return
        equity_curve = initial_equity * np.cumprod(1 + returns)
        final_equity = equity_curve[-1]
        cum_return = (final_equity - initial_equity) / initial_equity

        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (peak - equity_curve) / peak
        max_dd = np.max(drawdowns)

        # Sharpe ratio (annualized)
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        return cum_return, max_dd, sharpe, final_equity

    def _calculate_statistics(
        self,
        final_returns: np.ndarray,
        max_drawdowns: np.ndarray,
        sharpe_ratios: np.ndarray,
        initial_equity: float,
    ) -> MonteCarloResult:
        """Calculate comprehensive statistics from simulation results."""
        # Return statistics
        mean_return = np.mean(final_returns)
        median_return = np.median(final_returns)
        std_return = np.std(final_returns)

        # VaR and CVaR (as losses, so negative returns)
        var_95 = np.percentile(final_returns, 5)
        var_99 = np.percentile(final_returns, 1)
        cvar_95 = np.mean(final_returns[final_returns <= var_95])
        cvar_99 = np.mean(final_returns[final_returns <= var_99])

        # Drawdown statistics
        mean_max_dd = np.mean(max_drawdowns)
        median_max_dd = np.median(max_drawdowns)
        worst_dd = np.max(max_drawdowns)

        # Sharpe statistics
        mean_sharpe = np.mean(sharpe_ratios)
        median_sharpe = np.median(sharpe_ratios)

        # Probability estimates
        prob_profit = np.mean(final_returns > 0)
        prob_loss_10 = np.mean(final_returns < -0.10)
        prob_loss_20 = np.mean(final_returns < -0.20)

        # Confidence intervals
        ci_level = self.config.confidence_level
        alpha = (1 - ci_level) / 2
        return_ci_lower = np.percentile(final_returns, alpha * 100)
        return_ci_upper = np.percentile(final_returns, (1 - alpha) * 100)
        dd_ci_lower = np.percentile(max_drawdowns, alpha * 100)
        dd_ci_upper = np.percentile(max_drawdowns, (1 - alpha) * 100)

        # Percentiles
        percentiles = {}
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            percentiles[p] = float(np.percentile(final_returns, p))

        return MonteCarloResult(
            num_simulations=len(final_returns),
            num_periods=self.config.num_periods,
            initial_equity=initial_equity,
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            min_return=float(np.min(final_returns)),
            max_return=float(np.max(final_returns)),
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            mean_max_drawdown=mean_max_dd,
            median_max_drawdown=median_max_dd,
            worst_drawdown=worst_dd,
            mean_sharpe=mean_sharpe,
            median_sharpe=median_sharpe,
            prob_profit=prob_profit,
            prob_loss_10pct=prob_loss_10,
            prob_loss_20pct=prob_loss_20,
            return_ci_lower=return_ci_lower,
            return_ci_upper=return_ci_upper,
            drawdown_ci_lower=dd_ci_lower,
            drawdown_ci_upper=dd_ci_upper,
            return_percentiles=percentiles,
        )

    # ==========================================================================
    # Specialized Simulations
    # ==========================================================================

    def simulate_equity_paths(
        self,
        historical_returns: list[float],
        num_paths: int = 100,
    ) -> list[list[float]]:
        """
        Generate equity curve paths for visualization.

        Args:
            historical_returns: Historical returns
            num_paths: Number of paths to generate

        Returns:
            List of equity curves
        """
        returns_array = np.array(historical_returns)
        num_periods = self.config.num_periods
        initial = self.config.initial_equity

        equity_paths = []
        for _ in range(num_paths):
            sim_returns = self._resample(returns_array, num_periods)
            equity_curve = initial * np.cumprod(1 + sim_returns)
            equity_paths.append([initial, *equity_curve.tolist()])

        return equity_paths

    def simulate_with_regime(
        self,
        returns_by_regime: dict[str, list[float]],
        regime_probs: dict[str, float],
        transitions: dict[str, dict[str, float]] | None = None,
    ) -> MonteCarloResult:
        """
        Simulate with market regime switching.

        Args:
            returns_by_regime: Returns for each regime
            regime_probs: Probability of each regime
            transitions: Optional regime transition matrix

        Returns:
            MonteCarloResult
        """
        regimes = list(returns_by_regime.keys())
        probs = [regime_probs.get(r, 1 / len(regimes)) for r in regimes]

        # Normalize probabilities
        total = sum(probs)
        probs = [p / total for p in probs]

        # Create combined returns by sampling regimes
        combined_returns = []
        num_periods = self.config.num_periods

        current_regime = np.random.choice(regimes, p=probs)

        for _ in range(num_periods):
            # Get return from current regime
            regime_returns = returns_by_regime[current_regime]
            if regime_returns:
                ret = random.choice(regime_returns)
                combined_returns.append(ret)

            # Transition to next regime
            if transitions and current_regime in transitions:
                trans_probs = transitions[current_regime]
                trans_regimes = list(trans_probs.keys())
                trans_vals = [trans_probs.get(r, 0) for r in trans_regimes]
                total = sum(trans_vals)
                if total > 0:
                    trans_vals = [v / total for v in trans_vals]
                    current_regime = np.random.choice(trans_regimes, p=trans_vals)
            else:
                # Random regime selection
                current_regime = np.random.choice(regimes, p=probs)

        return self.simulate(combined_returns)

    def calculate_required_capital(
        self,
        historical_returns: list[float],
        target_survival_prob: float = 0.95,
        max_loss_pct: float = 0.50,
    ) -> float:
        """
        Calculate required capital to survive with given probability.

        Args:
            historical_returns: Historical returns
            target_survival_prob: Target probability of survival
            max_loss_pct: Maximum acceptable loss percentage

        Returns:
            Capital multiplier needed
        """
        result = self.simulate(historical_returns)

        # Find capital multiplier where P(loss > max_loss) < (1 - target)
        # Use CVaR as proxy for tail risk
        if result.cvar_99 >= -max_loss_pct:
            return 1.0

        # Need to increase capital proportionally
        return abs(result.cvar_99) / max_loss_pct


def create_monte_carlo_simulator(
    num_simulations: int = 10000,
    num_periods: int = 252,
    bootstrap_method: str = "block",
    random_seed: int | None = None,
) -> MonteCarloSimulator:
    """
    Factory function to create a Monte Carlo simulator.

    Args:
        num_simulations: Number of simulations to run
        num_periods: Number of periods per simulation
        bootstrap_method: "simple", "block", or "stationary"
        random_seed: Optional random seed for reproducibility

    Returns:
        Configured MonteCarloSimulator
    """
    method = BootstrapMethod(bootstrap_method.lower())
    config = MonteCarloConfig(
        num_simulations=num_simulations,
        num_periods=num_periods,
        bootstrap_method=method,
        random_seed=random_seed,
    )
    return MonteCarloSimulator(config)
