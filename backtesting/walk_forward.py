"""
Walk-Forward Analysis Module

UPGRADE-015 Phase 9: Backtesting Robustness

Provides walk-forward optimization and analysis:
- Rolling window optimization
- Out-of-sample validation
- Parameter stability analysis
- Anchored vs rolling methods

Features:
- Configurable window sizes
- Multiple optimization methods
- Robustness metrics
- Visualization support
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np


class WalkForwardMethod(Enum):
    """Walk-forward analysis methods."""

    ROLLING = "rolling"  # Fixed window moves forward
    ANCHORED = "anchored"  # Start always at beginning
    EXPANDING = "expanding"  # Window grows over time


@dataclass
class WindowResult:
    """Results from a single walk-forward window."""

    window_id: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_sample_start: datetime
    out_sample_end: datetime

    # Performance metrics
    in_sample_return: float = 0.0
    out_sample_return: float = 0.0
    in_sample_sharpe: float = 0.0
    out_sample_sharpe: float = 0.0

    # Optimized parameters
    optimal_params: dict[str, Any] = field(default_factory=dict)

    # Degradation ratio (out-sample / in-sample)
    performance_ratio: float = 0.0


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis results."""

    method: WalkForwardMethod
    num_windows: int
    window_results: list[WindowResult]

    # Aggregate metrics
    avg_in_sample_return: float = 0.0
    avg_out_sample_return: float = 0.0
    avg_performance_ratio: float = 0.0
    parameter_stability: float = 0.0

    # Pass/Fail indicators
    is_robust: bool = False
    robustness_score: float = 0.0
    warnings: list[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method.value,
            "num_windows": self.num_windows,
            "avg_in_sample_return": self.avg_in_sample_return,
            "avg_out_sample_return": self.avg_out_sample_return,
            "avg_performance_ratio": self.avg_performance_ratio,
            "parameter_stability": self.parameter_stability,
            "is_robust": self.is_robust,
            "robustness_score": self.robustness_score,
            "warnings": self.warnings,
            "windows": [
                {
                    "id": w.window_id,
                    "in_sample_return": w.in_sample_return,
                    "out_sample_return": w.out_sample_return,
                    "performance_ratio": w.performance_ratio,
                }
                for w in self.window_results
            ],
        }


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""

    method: WalkForwardMethod = WalkForwardMethod.ROLLING
    in_sample_days: int = 252  # 1 year
    out_sample_days: int = 63  # 3 months
    num_windows: int = 5
    min_performance_ratio: float = 0.5
    max_parameter_variance: float = 0.3
    robustness_threshold: float = 0.7


class WalkForwardAnalyzer:
    """Walk-forward analysis for strategy validation."""

    def __init__(
        self,
        config: WalkForwardConfig | None = None,
    ):
        """
        Initialize walk-forward analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or WalkForwardConfig()
        self._results: list[WindowResult] = []

    # ==========================================================================
    # Window Generation
    # ==========================================================================

    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate walk-forward windows.

        Args:
            start_date: Data start date
            end_date: Data end date

        Returns:
            List of (in_start, in_end, out_start, out_end) tuples
        """
        windows = []

        if self.config.method == WalkForwardMethod.ROLLING:
            windows = self._rolling_windows(start_date, end_date)
        elif self.config.method == WalkForwardMethod.ANCHORED:
            windows = self._anchored_windows(start_date, end_date)
        else:  # EXPANDING
            windows = self._expanding_windows(start_date, end_date)

        return windows

    def _rolling_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[tuple[datetime, datetime, datetime, datetime]]:
        """Generate rolling windows."""
        windows = []
        window_size = self.config.in_sample_days + self.config.out_sample_days
        step_size = self.config.out_sample_days

        current_start = start_date

        for i in range(self.config.num_windows):
            in_start = current_start
            in_end = in_start + timedelta(days=self.config.in_sample_days)
            out_start = in_end
            out_end = out_start + timedelta(days=self.config.out_sample_days)

            if out_end > end_date:
                break

            windows.append((in_start, in_end, out_start, out_end))
            current_start = current_start + timedelta(days=step_size)

        return windows

    def _anchored_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[tuple[datetime, datetime, datetime, datetime]]:
        """Generate anchored windows (always start from beginning)."""
        windows = []
        step_size = self.config.out_sample_days

        for i in range(self.config.num_windows):
            in_start = start_date
            in_end = start_date + timedelta(days=self.config.in_sample_days + i * step_size)
            out_start = in_end
            out_end = out_start + timedelta(days=self.config.out_sample_days)

            if out_end > end_date:
                break

            windows.append((in_start, in_end, out_start, out_end))

        return windows

    def _expanding_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[tuple[datetime, datetime, datetime, datetime]]:
        """Generate expanding windows."""
        # Same as anchored but with expanding in-sample
        return self._anchored_windows(start_date, end_date)

    # ==========================================================================
    # Analysis
    # ==========================================================================

    def analyze(
        self,
        returns: list[tuple[datetime, float]],
        optimizer: Callable[[list[tuple[datetime, float]]], dict[str, Any]] | None = None,
        evaluator: Callable[[list[tuple[datetime, float]], dict[str, Any]], dict[str, float]] | None = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis on returns data.

        Args:
            returns: List of (date, return) tuples
            optimizer: Function to optimize parameters on in-sample data
            evaluator: Function to evaluate performance given data and params

        Returns:
            WalkForwardResult with analysis
        """
        if not returns:
            return WalkForwardResult(
                method=self.config.method,
                num_windows=0,
                window_results=[],
                warnings=["No return data provided"],
            )

        # Convert to date-indexed dict for easy slicing
        returns_dict = {r[0]: r[1] for r in returns}
        dates = sorted(returns_dict.keys())
        start_date = dates[0]
        end_date = dates[-1]

        # Generate windows
        windows = self.generate_windows(start_date, end_date)

        if not windows:
            return WalkForwardResult(
                method=self.config.method,
                num_windows=0,
                window_results=[],
                warnings=["Could not generate windows for date range"],
            )

        # Analyze each window
        window_results = []
        all_params = []

        for i, (in_start, in_end, out_start, out_end) in enumerate(windows):
            # Split data
            in_sample = [(d, returns_dict[d]) for d in dates if in_start <= d < in_end]
            out_sample = [(d, returns_dict[d]) for d in dates if out_start <= d < out_end]

            if not in_sample or not out_sample:
                continue

            # Optimize on in-sample (or use default)
            if optimizer:
                optimal_params = optimizer(in_sample)
            else:
                optimal_params = {}

            all_params.append(optimal_params)

            # Evaluate on both samples
            if evaluator:
                in_metrics = evaluator(in_sample, optimal_params)
                out_metrics = evaluator(out_sample, optimal_params)
            else:
                in_metrics = self._default_evaluate(in_sample)
                out_metrics = self._default_evaluate(out_sample)

            # Calculate performance ratio
            in_ret = in_metrics.get("return", 0)
            out_ret = out_metrics.get("return", 0)

            if in_ret != 0:
                perf_ratio = out_ret / in_ret if in_ret > 0 else 0
            else:
                perf_ratio = 1.0 if out_ret >= 0 else 0

            window_results.append(
                WindowResult(
                    window_id=i,
                    in_sample_start=in_start,
                    in_sample_end=in_end,
                    out_sample_start=out_start,
                    out_sample_end=out_end,
                    in_sample_return=in_ret,
                    out_sample_return=out_ret,
                    in_sample_sharpe=in_metrics.get("sharpe", 0),
                    out_sample_sharpe=out_metrics.get("sharpe", 0),
                    optimal_params=optimal_params,
                    performance_ratio=perf_ratio,
                )
            )

        # Calculate aggregate metrics
        result = self._aggregate_results(window_results, all_params)
        return result

    def _default_evaluate(
        self,
        returns: list[tuple[datetime, float]],
    ) -> dict[str, float]:
        """Default evaluation function."""
        if not returns:
            return {"return": 0, "sharpe": 0}

        rets = np.array([r[1] for r in returns])
        total_return = np.prod(1 + rets) - 1
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0

        return {"return": total_return, "sharpe": sharpe}

    def _aggregate_results(
        self,
        window_results: list[WindowResult],
        all_params: list[dict[str, Any]],
    ) -> WalkForwardResult:
        """Aggregate window results into final result."""
        if not window_results:
            return WalkForwardResult(
                method=self.config.method,
                num_windows=0,
                window_results=[],
                warnings=["No valid windows"],
            )

        # Calculate averages
        avg_in_ret = np.mean([w.in_sample_return for w in window_results])
        avg_out_ret = np.mean([w.out_sample_return for w in window_results])
        avg_ratio = np.mean([w.performance_ratio for w in window_results])

        # Calculate parameter stability
        param_stability = self._calculate_param_stability(all_params)

        # Generate warnings
        warnings = []

        if avg_ratio < self.config.min_performance_ratio:
            warnings.append(f"Low performance ratio ({avg_ratio:.2f}) suggests overfitting")

        if param_stability < 1 - self.config.max_parameter_variance:
            warnings.append(f"Low parameter stability ({param_stability:.2f}) suggests sensitivity")

        # Count positive out-of-sample windows
        positive_windows = sum(1 for w in window_results if w.out_sample_return > 0)
        win_rate = positive_windows / len(window_results)

        if win_rate < 0.5:
            warnings.append(f"Only {win_rate:.0%} of windows had positive out-of-sample returns")

        # Calculate robustness score (0-1)
        robustness = (
            0.3 * min(avg_ratio / self.config.min_performance_ratio, 1.0)
            + 0.3 * param_stability
            + 0.2 * win_rate
            + 0.2 * (avg_out_ret > 0)
        )

        is_robust = robustness >= self.config.robustness_threshold

        return WalkForwardResult(
            method=self.config.method,
            num_windows=len(window_results),
            window_results=window_results,
            avg_in_sample_return=avg_in_ret,
            avg_out_sample_return=avg_out_ret,
            avg_performance_ratio=avg_ratio,
            parameter_stability=param_stability,
            is_robust=is_robust,
            robustness_score=robustness,
            warnings=warnings,
        )

    def _calculate_param_stability(
        self,
        all_params: list[dict[str, Any]],
    ) -> float:
        """Calculate stability of optimized parameters across windows."""
        if len(all_params) < 2:
            return 1.0

        # Get common numeric parameters
        common_params = set(all_params[0].keys())
        for params in all_params[1:]:
            common_params &= set(params.keys())

        if not common_params:
            return 1.0

        # Calculate coefficient of variation for each parameter
        cvs = []
        for param in common_params:
            values = []
            for params in all_params:
                val = params.get(param)
                if isinstance(val, (int, float)):
                    values.append(val)

            if len(values) >= 2 and np.mean(values) != 0:
                cv = np.std(values) / abs(np.mean(values))
                cvs.append(min(cv, 1.0))  # Cap at 1.0

        if not cvs:
            return 1.0

        # Stability is 1 - average CV
        return max(0, 1 - np.mean(cvs))


def create_walk_forward_analyzer(
    method: str = "rolling",
    in_sample_days: int = 252,
    out_sample_days: int = 63,
    num_windows: int = 5,
) -> WalkForwardAnalyzer:
    """
    Factory function to create a walk-forward analyzer.

    Args:
        method: "rolling", "anchored", or "expanding"
        in_sample_days: In-sample period in days
        out_sample_days: Out-of-sample period in days
        num_windows: Number of windows

    Returns:
        Configured WalkForwardAnalyzer
    """
    method_enum = WalkForwardMethod(method.lower())
    config = WalkForwardConfig(
        method=method_enum,
        in_sample_days=in_sample_days,
        out_sample_days=out_sample_days,
        num_windows=num_windows,
    )
    return WalkForwardAnalyzer(config)
