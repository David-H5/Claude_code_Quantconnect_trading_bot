"""
Parameter Sensitivity Analysis Module

UPGRADE-015 Phase 9: Backtesting Robustness

Provides parameter sensitivity analysis for strategies:
- Grid search across parameters
- Sensitivity coefficients
- Stability regions identification
- Cliff detection

Features:
- Multi-dimensional parameter sweeps
- Heatmap generation
- Optimal region finding
- Robustness scoring
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class SensitivityLevel(Enum):
    """Sensitivity classification."""

    LOW = "low"  # < 10% change
    MEDIUM = "medium"  # 10-30% change
    HIGH = "high"  # 30-50% change
    CRITICAL = "critical"  # > 50% change


@dataclass
class ParameterRange:
    """Definition of a parameter range."""

    name: str
    min_value: float
    max_value: float
    step: float | None = None
    num_points: int = 10
    log_scale: bool = False

    def get_values(self) -> list[float]:
        """Get list of parameter values to test."""
        if self.step is not None:
            values = []
            v = self.min_value
            while v <= self.max_value:
                values.append(v)
                v += self.step
            return values

        if self.log_scale:
            return np.logspace(np.log10(self.min_value), np.log10(self.max_value), self.num_points).tolist()

        return np.linspace(self.min_value, self.max_value, self.num_points).tolist()


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis."""

    parameter_name: str
    base_value: float
    base_metric: float

    # Sensitivity metrics
    sensitivity_coefficient: float = 0.0
    elasticity: float = 0.0
    sensitivity_level: SensitivityLevel = SensitivityLevel.LOW

    # Range results
    values_tested: list[float] = field(default_factory=list)
    metrics: list[float] = field(default_factory=list)

    # Stability
    stable_range: tuple[float, float] | None = None
    cliff_points: list[float] = field(default_factory=list)

    # Best value
    optimal_value: float = 0.0
    optimal_metric: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "parameter_name": self.parameter_name,
            "base_value": self.base_value,
            "base_metric": self.base_metric,
            "sensitivity_coefficient": self.sensitivity_coefficient,
            "elasticity": self.elasticity,
            "sensitivity_level": self.sensitivity_level.value,
            "stable_range": self.stable_range,
            "cliff_points": self.cliff_points,
            "optimal_value": self.optimal_value,
            "optimal_metric": self.optimal_metric,
        }


@dataclass
class MultiDimensionalResult:
    """Results from multi-dimensional sensitivity analysis."""

    parameters: list[str]
    dimensions: list[int]

    # Results grid
    results_grid: dict[tuple[float, ...], float] = field(default_factory=dict)

    # Optimal point
    optimal_params: dict[str, float] = field(default_factory=dict)
    optimal_metric: float = 0.0

    # Stability
    stable_region_volume: float = 0.0
    total_volume: float = 0.0

    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "parameters": self.parameters,
            "optimal_params": self.optimal_params,
            "optimal_metric": self.optimal_metric,
            "stable_region_pct": (self.stable_region_volume / self.total_volume if self.total_volume > 0 else 0),
        }


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis."""

    metric_name: str = "sharpe_ratio"
    stability_threshold: float = 0.20  # 20% deviation from base
    cliff_threshold: float = 0.50  # 50% drop = cliff
    min_improvement: float = 0.05  # 5% improvement = significant


class ParameterSensitivity:
    """Parameter sensitivity analysis for trading strategies."""

    def __init__(
        self,
        config: SensitivityConfig | None = None,
    ):
        """
        Initialize sensitivity analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or SensitivityConfig()
        self._results: dict[str, SensitivityResult] = {}

    # ==========================================================================
    # Single Parameter Analysis
    # ==========================================================================

    def analyze_parameter(
        self,
        param_range: ParameterRange,
        evaluator: Callable[[float], float],
        base_value: float | None = None,
    ) -> SensitivityResult:
        """
        Analyze sensitivity to a single parameter.

        Args:
            param_range: Parameter range definition
            evaluator: Function that takes parameter value and returns metric
            base_value: Base parameter value (default: middle of range)

        Returns:
            SensitivityResult
        """
        values = param_range.get_values()
        base_value = base_value or values[len(values) // 2]

        # Evaluate at all points
        metrics = []
        for v in values:
            try:
                metric = evaluator(v)
                metrics.append(metric)
            except Exception:
                metrics.append(float("nan"))

        # Get base metric
        base_metric = evaluator(base_value)

        # Calculate sensitivity coefficient
        sensitivity = self._calculate_sensitivity(values, metrics, base_value, base_metric)

        # Calculate elasticity
        elasticity = self._calculate_elasticity(values, metrics, base_value, base_metric)

        # Find optimal
        valid_indices = [i for i, m in enumerate(metrics) if not np.isnan(m)]
        if valid_indices:
            best_idx = max(valid_indices, key=lambda i: metrics[i])
            optimal_value = values[best_idx]
            optimal_metric = metrics[best_idx]
        else:
            optimal_value = base_value
            optimal_metric = base_metric

        # Find stable range
        stable_range = self._find_stable_range(values, metrics, base_metric)

        # Find cliff points
        cliff_points = self._find_cliff_points(values, metrics)

        # Classify sensitivity
        level = self._classify_sensitivity(sensitivity)

        result = SensitivityResult(
            parameter_name=param_range.name,
            base_value=base_value,
            base_metric=base_metric,
            sensitivity_coefficient=sensitivity,
            elasticity=elasticity,
            sensitivity_level=level,
            values_tested=values,
            metrics=metrics,
            stable_range=stable_range,
            cliff_points=cliff_points,
            optimal_value=optimal_value,
            optimal_metric=optimal_metric,
        )

        self._results[param_range.name] = result
        return result

    def _calculate_sensitivity(
        self,
        values: list[float],
        metrics: list[float],
        base_value: float,
        base_metric: float,
    ) -> float:
        """Calculate sensitivity coefficient (∂metric/∂param)."""
        if len(values) < 2 or base_metric == 0:
            return 0.0

        # Use finite differences
        valid_pairs = []
        for i in range(len(values) - 1):
            if not np.isnan(metrics[i]) and not np.isnan(metrics[i + 1]):
                dm = metrics[i + 1] - metrics[i]
                dp = values[i + 1] - values[i]
                if dp != 0:
                    valid_pairs.append(dm / dp)

        if not valid_pairs:
            return 0.0

        return np.mean(valid_pairs)

    def _calculate_elasticity(
        self,
        values: list[float],
        metrics: list[float],
        base_value: float,
        base_metric: float,
    ) -> float:
        """Calculate elasticity (% change in metric / % change in param)."""
        if base_value == 0 or base_metric == 0:
            return 0.0

        # Find points near base
        near_base = []
        for v, m in zip(values, metrics):
            if not np.isnan(m) and v != base_value:
                pct_param_change = (v - base_value) / base_value
                pct_metric_change = (m - base_metric) / base_metric
                if pct_param_change != 0:
                    near_base.append(pct_metric_change / pct_param_change)

        if not near_base:
            return 0.0

        return np.mean(near_base)

    def _find_stable_range(
        self,
        values: list[float],
        metrics: list[float],
        base_metric: float,
    ) -> tuple[float, float] | None:
        """Find range where metric stays within threshold of base."""
        threshold = self.config.stability_threshold

        stable_values = []
        for v, m in zip(values, metrics):
            if not np.isnan(m) and base_metric != 0:
                deviation = abs(m - base_metric) / abs(base_metric)
                if deviation <= threshold:
                    stable_values.append(v)

        if not stable_values:
            return None

        return (min(stable_values), max(stable_values))

    def _find_cliff_points(
        self,
        values: list[float],
        metrics: list[float],
    ) -> list[float]:
        """Find parameter values where performance drops sharply."""
        cliffs = []
        threshold = self.config.cliff_threshold

        for i in range(len(metrics) - 1):
            if np.isnan(metrics[i]) or np.isnan(metrics[i + 1]):
                continue

            if metrics[i] != 0:
                change = (metrics[i + 1] - metrics[i]) / abs(metrics[i])
                if change < -threshold:
                    cliffs.append(values[i + 1])

        return cliffs

    def _classify_sensitivity(self, sensitivity: float) -> SensitivityLevel:
        """Classify sensitivity level."""
        abs_sens = abs(sensitivity)
        if abs_sens > 0.50:
            return SensitivityLevel.CRITICAL
        elif abs_sens > 0.30:
            return SensitivityLevel.HIGH
        elif abs_sens > 0.10:
            return SensitivityLevel.MEDIUM
        return SensitivityLevel.LOW

    # ==========================================================================
    # Multi-Parameter Analysis
    # ==========================================================================

    def analyze_multi_parameter(
        self,
        param_ranges: list[ParameterRange],
        evaluator: Callable[[dict[str, float]], float],
    ) -> MultiDimensionalResult:
        """
        Analyze sensitivity across multiple parameters.

        Args:
            param_ranges: List of parameter ranges
            evaluator: Function that takes dict of params and returns metric

        Returns:
            MultiDimensionalResult
        """
        # Generate all combinations
        param_values = [pr.get_values() for pr in param_ranges]
        param_names = [pr.name for pr in param_ranges]

        results_grid: dict[tuple[float, ...], float] = {}
        best_params = {}
        best_metric = float("-inf")

        # Grid search
        for combo in self._generate_combinations(param_values):
            params = dict(zip(param_names, combo))
            try:
                metric = evaluator(params)
                results_grid[combo] = metric

                if metric > best_metric:
                    best_metric = metric
                    best_params = params.copy()
            except Exception:
                results_grid[combo] = float("nan")

        # Calculate stable region
        stable_count = 0
        total_count = len(results_grid)

        if best_metric != float("-inf"):
            threshold = abs(best_metric) * self.config.stability_threshold
            for metric in results_grid.values():
                if not np.isnan(metric) and abs(metric - best_metric) <= threshold:
                    stable_count += 1

        return MultiDimensionalResult(
            parameters=param_names,
            dimensions=[len(v) for v in param_values],
            results_grid=results_grid,
            optimal_params=best_params,
            optimal_metric=best_metric,
            stable_region_volume=stable_count,
            total_volume=total_count,
        )

    def _generate_combinations(
        self,
        param_values: list[list[float]],
    ) -> list[tuple[float, ...]]:
        """Generate all parameter combinations."""
        if not param_values:
            return [()]

        result = []
        first_values = param_values[0]
        rest_combinations = self._generate_combinations(param_values[1:])

        for v in first_values:
            for rest in rest_combinations:
                result.append((v, *rest))

        return result

    # ==========================================================================
    # Heatmap Generation
    # ==========================================================================

    def generate_heatmap(
        self,
        param1: ParameterRange,
        param2: ParameterRange,
        evaluator: Callable[[dict[str, float]], float],
    ) -> dict[str, Any]:
        """
        Generate 2D heatmap data for visualization.

        Args:
            param1: First parameter range
            param2: Second parameter range
            evaluator: Evaluation function

        Returns:
            Heatmap data dictionary
        """
        values1 = param1.get_values()
        values2 = param2.get_values()

        heatmap = np.zeros((len(values1), len(values2)))

        for i, v1 in enumerate(values1):
            for j, v2 in enumerate(values2):
                try:
                    metric = evaluator({param1.name: v1, param2.name: v2})
                    heatmap[i, j] = metric
                except Exception:
                    heatmap[i, j] = float("nan")

        return {
            "x_param": param1.name,
            "y_param": param2.name,
            "x_values": values1,
            "y_values": values2,
            "heatmap": heatmap.tolist(),
            "min_value": float(np.nanmin(heatmap)),
            "max_value": float(np.nanmax(heatmap)),
        }

    # ==========================================================================
    # Robustness Scoring
    # ==========================================================================

    def calculate_robustness_score(
        self,
        results: list[SensitivityResult] | None = None,
    ) -> float:
        """
        Calculate overall robustness score (0-1).

        Args:
            results: Optional list of results (uses stored if None)

        Returns:
            Robustness score
        """
        results = results or list(self._results.values())
        if not results:
            return 0.0

        scores = []
        for r in results:
            # Score based on sensitivity level
            if r.sensitivity_level == SensitivityLevel.LOW:
                level_score = 1.0
            elif r.sensitivity_level == SensitivityLevel.MEDIUM:
                level_score = 0.7
            elif r.sensitivity_level == SensitivityLevel.HIGH:
                level_score = 0.4
            else:
                level_score = 0.1

            # Score based on stable range
            if r.stable_range and r.values_tested:
                range_width = r.stable_range[1] - r.stable_range[0]
                total_width = max(r.values_tested) - min(r.values_tested)
                if total_width > 0:
                    range_score = range_width / total_width
                else:
                    range_score = 0.0
            else:
                range_score = 0.0

            # Score based on cliff points
            cliff_score = 1.0 - min(len(r.cliff_points) * 0.2, 1.0)

            # Combine scores
            param_score = 0.4 * level_score + 0.4 * range_score + 0.2 * cliff_score
            scores.append(param_score)

        return np.mean(scores)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all sensitivity analyses."""
        if not self._results:
            return {"num_parameters": 0}

        critical_params = [
            r.parameter_name for r in self._results.values() if r.sensitivity_level == SensitivityLevel.CRITICAL
        ]

        high_sens_params = [
            r.parameter_name for r in self._results.values() if r.sensitivity_level == SensitivityLevel.HIGH
        ]

        return {
            "num_parameters": len(self._results),
            "critical_parameters": critical_params,
            "high_sensitivity_parameters": high_sens_params,
            "robustness_score": self.calculate_robustness_score(),
            "results": {name: r.to_dict() for name, r in self._results.items()},
        }


def create_parameter_sensitivity(
    stability_threshold: float = 0.20,
    cliff_threshold: float = 0.50,
) -> ParameterSensitivity:
    """
    Factory function to create a parameter sensitivity analyzer.

    Args:
        stability_threshold: Deviation threshold for stability
        cliff_threshold: Drop threshold for cliff detection

    Returns:
        Configured ParameterSensitivity
    """
    config = SensitivityConfig(
        stability_threshold=stability_threshold,
        cliff_threshold=cliff_threshold,
    )
    return ParameterSensitivity(config)
