"""
Walk-Forward Analysis for Trading Algorithm Evaluation.

Implements walk-forward analysis to prevent overfitting by optimizing parameters on
one data segment, then testing on the next segment. This is more robust than traditional
backtesting as it simulates real-world conditions where future data is unknown.

Enhanced with Monte Carlo simulation (December 2025) to:
- Generate random combinations of in-sample/out-of-sample periods
- Account for randomness in data ordering
- Provide confidence intervals for performance estimates
- Better assess parameter stability under different data orderings

Reference: https://3commas.io/blog/comprehensive-2025-guide-to-backtesting-ai-trading
Reference: https://www.utradealgos.com/blog/5-key-metrics-to-evaluate-the-performance-of-your-trading-algorithms
Reference: docs/research/EVALUATION_FRAMEWORK_RESEARCH.md

Walk-Forward Process:
1. Split data into sequential windows (e.g., train 6 months, test 1 month)
2. Optimize parameters on training window
3. Test with optimized parameters on next out-of-sample window
4. Roll forward and repeat
5. Aggregate results across all windows
6. (Optional) Run Monte Carlo simulation for confidence intervals

Version: 2.0 (December 2025) - Added Monte Carlo simulation
"""

import math
import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass
class WalkForwardWindow:
    """Single walk-forward analysis window."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    optimized_parameters: dict[str, Any]
    train_performance: dict[str, float]
    test_performance: dict[str, float]
    degradation_pct: float  # (train - test) / train * 100


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis results."""

    total_windows: int
    windows: list[WalkForwardWindow]
    avg_train_performance: dict[str, float]
    avg_test_performance: dict[str, float]
    avg_degradation_pct: float
    consistency_score: float  # 0-1, how consistent parameters were across windows
    robustness_score: float  # 0-1, how well test performed relative to train
    production_ready: bool


@dataclass
class MonteCarloIteration:
    """Single Monte Carlo iteration result."""

    iteration_id: int
    window_selection: list[int]  # Which windows were used
    train_performance: dict[str, float]
    test_performance: dict[str, float]
    degradation_pct: float
    robustness_score: float


@dataclass
class MonteCarloResult:
    """
    Monte Carlo simulation results for walk-forward analysis.

    Provides confidence intervals and distribution statistics for
    performance metrics across random window combinations.
    """

    num_iterations: int
    iterations: list[MonteCarloIteration]

    # Performance statistics (mean, std, percentiles)
    sharpe_mean: float
    sharpe_std: float
    sharpe_5th_percentile: float
    sharpe_95th_percentile: float

    # Degradation statistics
    degradation_mean: float
    degradation_std: float
    degradation_5th_percentile: float
    degradation_95th_percentile: float

    # Robustness statistics
    robustness_mean: float
    robustness_std: float

    # Confidence assessment
    confidence_level: float  # 0-1, overall confidence in results
    production_ready_pct: float  # % of iterations that were production ready
    recommendation: str


def run_walk_forward_analysis(
    data_start: datetime,
    data_end: datetime,
    train_window_months: int = 6,
    test_window_months: int = 1,
    optimization_func: Callable | None = None,
    evaluation_func: Callable | None = None,
    parameter_space: dict[str, list[Any]] | None = None,
) -> WalkForwardResult:
    """
    Run walk-forward analysis on trading algorithm.

    Args:
        data_start: Start date of available data
        data_end: End date of available data
        train_window_months: Training window size in months (default: 6)
        test_window_months: Test window size in months (default: 1)
        optimization_func: Function to optimize parameters (train_data) -> best_params
        evaluation_func: Function to evaluate performance (test_data, params) -> metrics
        parameter_space: Dict of parameter ranges to optimize

    Returns:
        WalkForwardResult with aggregated analysis
    """
    if optimization_func is None or evaluation_func is None:
        raise ValueError("optimization_func and evaluation_func are required")

    windows = []
    current_date = data_start

    window_id = 0

    while current_date + timedelta(days=train_window_months * 30) + timedelta(days=test_window_months * 30) <= data_end:
        train_start = current_date
        train_end = train_start + timedelta(days=train_window_months * 30)
        test_start = train_end
        test_end = test_start + timedelta(days=test_window_months * 30)

        # 1. Optimize parameters on training window
        optimized_params = optimization_func(train_start, train_end, parameter_space)

        # 2. Evaluate on training window (in-sample)
        train_performance = evaluation_func(train_start, train_end, optimized_params)

        # 3. Evaluate on test window (out-of-sample)
        test_performance = evaluation_func(test_start, test_end, optimized_params)

        # 4. Calculate degradation
        train_sharpe = train_performance.get("sharpe_ratio", 0)
        test_sharpe = test_performance.get("sharpe_ratio", 0)
        degradation = ((train_sharpe - test_sharpe) / train_sharpe * 100) if train_sharpe > 0 else 0

        window = WalkForwardWindow(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            optimized_parameters=optimized_params,
            train_performance=train_performance,
            test_performance=test_performance,
            degradation_pct=degradation,
        )

        windows.append(window)

        # Roll forward by test window size
        current_date = test_start
        window_id += 1

    # Aggregate results
    avg_train_performance = _aggregate_performance([w.train_performance for w in windows])
    avg_test_performance = _aggregate_performance([w.test_performance for w in windows])

    avg_degradation = statistics.mean([w.degradation_pct for w in windows])

    # Consistency score: How similar were parameters across windows?
    consistency_score = _calculate_parameter_consistency(windows)

    # Robustness score: test performance / train performance
    train_sharpe_avg = avg_train_performance.get("sharpe_ratio", 0)
    test_sharpe_avg = avg_test_performance.get("sharpe_ratio", 0)
    robustness_score = test_sharpe_avg / train_sharpe_avg if train_sharpe_avg > 0 else 0

    # Production ready if:
    # - Average degradation < 15%
    # - Robustness score > 0.80
    # - Test Sharpe > 1.0 (for conservative) or > 0.8 (for moderate/aggressive)
    production_ready = avg_degradation < 15.0 and robustness_score > 0.80 and test_sharpe_avg > 0.8

    return WalkForwardResult(
        total_windows=len(windows),
        windows=windows,
        avg_train_performance=avg_train_performance,
        avg_test_performance=avg_test_performance,
        avg_degradation_pct=avg_degradation,
        consistency_score=consistency_score,
        robustness_score=robustness_score,
        production_ready=production_ready,
    )


def _aggregate_performance(performances: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate performance metrics across windows."""
    if not performances:
        return {}

    keys = performances[0].keys()
    aggregated = {}

    for key in keys:
        values = [p.get(key, 0) for p in performances]
        aggregated[key] = statistics.mean(values)

    return aggregated


def _calculate_parameter_consistency(windows: list[WalkForwardWindow]) -> float:
    """
    Calculate parameter consistency across windows.

    Returns:
        Consistency score 0-1 (1 = identical parameters, 0 = completely different)
    """
    if len(windows) < 2:
        return 1.0

    # Compare each window's parameters to the next
    param_keys = windows[0].optimized_parameters.keys()
    consistencies = []

    for i in range(len(windows) - 1):
        window_a = windows[i].optimized_parameters
        window_b = windows[i + 1].optimized_parameters

        matches = 0
        total = 0

        for key in param_keys:
            val_a = window_a.get(key)
            val_b = window_b.get(key)

            if val_a is not None and val_b is not None:
                total += 1
                # For numeric values, check if within 10% of each other
                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                    if abs(val_a - val_b) / max(abs(val_a), abs(val_b), 0.0001) < 0.10:
                        matches += 1
                elif val_a == val_b:
                    matches += 1

        if total > 0:
            consistencies.append(matches / total)

    return statistics.mean(consistencies) if consistencies else 0.0


def generate_walk_forward_report(result: WalkForwardResult) -> str:
    """
    Generate walk-forward analysis report.

    Args:
        result: WalkForwardResult object

    Returns:
        Formatted markdown report
    """
    report = []
    report.append("# Walk-Forward Analysis Report\n")
    report.append(f"**Total Windows**: {result.total_windows}")
    report.append(f"**Production Ready**: {'✅ YES' if result.production_ready else '⚠️ NO'}\n")

    # Summary metrics
    report.append("## Summary Metrics\n")
    report.append(
        f"- Average Degradation: {result.avg_degradation_pct:.1f}% {'✅' if result.avg_degradation_pct < 15 else '⚠️'}"
    )
    report.append(
        f"- Robustness Score: {result.robustness_score:.2f} {'✅' if result.robustness_score > 0.80 else '⚠️'}"
    )
    report.append(f"- Consistency Score: {result.consistency_score:.2f}\n")

    # Training performance
    report.append("## Average Training Performance (In-Sample)\n")
    for key, value in result.avg_train_performance.items():
        report.append(f"- {key.replace('_', ' ').title()}: {value:.3f}")

    # Test performance
    report.append("\n## Average Test Performance (Out-of-Sample)\n")
    for key, value in result.avg_test_performance.items():
        report.append(f"- {key.replace('_', ' ').title()}: {value:.3f}")

    # Window details
    report.append("\n## Window Details\n")
    report.append("| Window | Train Period | Test Period | Train Sharpe | Test Sharpe | Degradation |")
    report.append("|--------|--------------|-------------|--------------|-------------|-------------|")

    for window in result.windows:
        train_period = f"{window.train_start.strftime('%Y-%m')} to {window.train_end.strftime('%Y-%m')}"
        test_period = f"{window.test_start.strftime('%Y-%m')} to {window.test_end.strftime('%Y-%m')}"
        train_sharpe = window.train_performance.get("sharpe_ratio", 0)
        test_sharpe = window.test_performance.get("sharpe_ratio", 0)

        report.append(
            f"| {window.window_id + 1} | {train_period} | {test_period} | "
            f"{train_sharpe:.2f} | {test_sharpe:.2f} | {window.degradation_pct:.1f}% |"
        )

    # Assessment
    report.append("\n## Assessment\n")
    if result.production_ready:
        report.append("✅ **Strategy is robust and production-ready**")
        report.append("- Out-of-sample degradation is acceptable (<15%)")
        report.append("- Test performance is consistent across windows")
        report.append("- Parameters show reasonable stability")
    else:
        report.append("⚠️ **Strategy requires improvements**")
        if result.avg_degradation_pct >= 15:
            report.append(f"- ❌ Excessive degradation ({result.avg_degradation_pct:.1f}% > 15%)")
        if result.robustness_score <= 0.80:
            report.append(f"- ❌ Low robustness score ({result.robustness_score:.2f} < 0.80)")
        if result.avg_test_performance.get("sharpe_ratio", 0) < 0.8:
            report.append(
                f"- ❌ Low test Sharpe ratio ({result.avg_test_performance.get('sharpe_ratio', 0):.2f} < 0.8)"
            )

    return "\n".join(report)


def create_walk_forward_schedule(
    data_start: datetime,
    data_end: datetime,
    train_window_months: int = 6,
    test_window_months: int = 1,
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """
    Create walk-forward analysis schedule without running analysis.

    Args:
        data_start: Start date of available data
        data_end: End date of available data
        train_window_months: Training window size in months
        test_window_months: Test window size in months

    Returns:
        List of (train_start, train_end, test_start, test_end) tuples
    """
    schedule = []
    current_date = data_start

    while current_date + timedelta(days=train_window_months * 30) + timedelta(days=test_window_months * 30) <= data_end:
        train_start = current_date
        train_end = train_start + timedelta(days=train_window_months * 30)
        test_start = train_end
        test_end = test_start + timedelta(days=test_window_months * 30)

        schedule.append((train_start, train_end, test_start, test_end))

        # Roll forward by test window size
        current_date = test_start

    return schedule


def validate_walk_forward_configuration(
    data_start: datetime,
    data_end: datetime,
    train_window_months: int,
    test_window_months: int,
    min_windows: int = 3,
) -> tuple[bool, str]:
    """
    Validate walk-forward configuration before running analysis.

    Args:
        data_start: Start date of available data
        data_end: End date of available data
        train_window_months: Training window size in months
        test_window_months: Test window size in months
        min_windows: Minimum number of windows required (default: 3)

    Returns:
        (is_valid, message) tuple
    """
    # Check if we have enough data
    total_days = (data_end - data_start).days
    window_days = (train_window_months + test_window_months) * 30

    if total_days < window_days:
        return False, f"Insufficient data: {total_days} days available, {window_days} days required for one window"

    # Calculate number of windows
    schedule = create_walk_forward_schedule(data_start, data_end, train_window_months, test_window_months)
    num_windows = len(schedule)

    if num_windows < min_windows:
        return False, f"Insufficient windows: {num_windows} windows possible, {min_windows} required"

    return True, f"Valid configuration: {num_windows} windows will be created"


# =============================================================================
# Monte Carlo Simulation (NEW - December 2025)
# =============================================================================


def run_monte_carlo_walk_forward(
    walk_forward_result: WalkForwardResult,
    num_iterations: int = 1000,
    window_subset_size: int | None = None,
    random_seed: int | None = None,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation on walk-forward analysis results.

    Randomly samples and permutes windows to assess the stability of results
    and provide confidence intervals for performance metrics.

    Args:
        walk_forward_result: Completed walk-forward analysis result
        num_iterations: Number of Monte Carlo iterations (default: 1000)
        window_subset_size: Number of windows to sample per iteration
                           (default: all windows)
        random_seed: Random seed for reproducibility

    Returns:
        MonteCarloResult with distribution statistics and confidence intervals
    """
    if random_seed is not None:
        random.seed(random_seed)

    windows = walk_forward_result.windows
    num_windows = len(windows)

    if num_windows < 2:
        raise ValueError("Need at least 2 windows for Monte Carlo simulation")

    # Default to using all windows
    if window_subset_size is None:
        window_subset_size = num_windows

    window_subset_size = min(window_subset_size, num_windows)

    iterations = []
    sharpe_values = []
    degradation_values = []
    robustness_values = []
    production_ready_count = 0

    for i in range(num_iterations):
        # Randomly sample windows
        selected_indices = random.sample(range(num_windows), window_subset_size)
        selected_windows = [windows[idx] for idx in selected_indices]

        # Aggregate performance for this iteration
        train_perf = _aggregate_performance([w.train_performance for w in selected_windows])
        test_perf = _aggregate_performance([w.test_performance for w in selected_windows])

        # Calculate metrics
        train_sharpe = train_perf.get("sharpe_ratio", 0)
        test_sharpe = test_perf.get("sharpe_ratio", 0)
        degradation = ((train_sharpe - test_sharpe) / train_sharpe * 100) if train_sharpe > 0 else 0
        robustness = test_sharpe / train_sharpe if train_sharpe > 0 else 0

        # Check production readiness
        is_production_ready = degradation < 15.0 and robustness > 0.80 and test_sharpe > 0.8
        if is_production_ready:
            production_ready_count += 1

        iteration = MonteCarloIteration(
            iteration_id=i,
            window_selection=selected_indices,
            train_performance=train_perf,
            test_performance=test_perf,
            degradation_pct=degradation,
            robustness_score=robustness,
        )
        iterations.append(iteration)

        sharpe_values.append(test_sharpe)
        degradation_values.append(degradation)
        robustness_values.append(robustness)

    # Calculate statistics
    sharpe_mean = statistics.mean(sharpe_values)
    sharpe_std = statistics.stdev(sharpe_values) if len(sharpe_values) > 1 else 0

    degradation_mean = statistics.mean(degradation_values)
    degradation_std = statistics.stdev(degradation_values) if len(degradation_values) > 1 else 0

    robustness_mean = statistics.mean(robustness_values)
    robustness_std = statistics.stdev(robustness_values) if len(robustness_values) > 1 else 0

    # Calculate percentiles
    sorted_sharpe = sorted(sharpe_values)
    sorted_degradation = sorted(degradation_values)

    sharpe_5th = _percentile(sorted_sharpe, 5)
    sharpe_95th = _percentile(sorted_sharpe, 95)
    degradation_5th = _percentile(sorted_degradation, 5)
    degradation_95th = _percentile(sorted_degradation, 95)

    # Calculate confidence level based on variance
    # Lower variance = higher confidence
    sharpe_cv = sharpe_std / abs(sharpe_mean) if sharpe_mean != 0 else float("inf")
    confidence_level = max(0, min(1, 1 - sharpe_cv))

    production_ready_pct = production_ready_count / num_iterations * 100

    # Generate recommendation
    recommendation = _generate_monte_carlo_recommendation(
        sharpe_mean=sharpe_mean,
        sharpe_5th=sharpe_5th,
        degradation_mean=degradation_mean,
        degradation_95th=degradation_95th,
        production_ready_pct=production_ready_pct,
        confidence_level=confidence_level,
    )

    return MonteCarloResult(
        num_iterations=num_iterations,
        iterations=iterations,
        sharpe_mean=sharpe_mean,
        sharpe_std=sharpe_std,
        sharpe_5th_percentile=sharpe_5th,
        sharpe_95th_percentile=sharpe_95th,
        degradation_mean=degradation_mean,
        degradation_std=degradation_std,
        degradation_5th_percentile=degradation_5th,
        degradation_95th_percentile=degradation_95th,
        robustness_mean=robustness_mean,
        robustness_std=robustness_std,
        confidence_level=confidence_level,
        production_ready_pct=production_ready_pct,
        recommendation=recommendation,
    )


def _percentile(sorted_data: list[float], percentile: float) -> float:
    """Calculate percentile from sorted data."""
    if not sorted_data:
        return 0.0

    k = (len(sorted_data) - 1) * (percentile / 100)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_data[int(k)]

    return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)


def _generate_monte_carlo_recommendation(
    sharpe_mean: float,
    sharpe_5th: float,
    degradation_mean: float,
    degradation_95th: float,
    production_ready_pct: float,
    confidence_level: float,
) -> str:
    """Generate Monte Carlo recommendation based on simulation results."""
    recommendations = []

    # Sharpe assessment
    if sharpe_5th >= 1.0:
        recommendations.append("✅ 95% confidence: Sharpe > 1.0")
    elif sharpe_mean >= 1.0:
        recommendations.append(f"⚠️ Mean Sharpe={sharpe_mean:.2f} but 5th percentile={sharpe_5th:.2f}")
    else:
        recommendations.append(f"❌ Low Sharpe: mean={sharpe_mean:.2f}, 5th percentile={sharpe_5th:.2f}")

    # Degradation assessment
    if degradation_95th < 15:
        recommendations.append("✅ 95% confidence: Degradation < 15%")
    elif degradation_mean < 15:
        recommendations.append(
            f"⚠️ Mean degradation={degradation_mean:.1f}% but 95th percentile={degradation_95th:.1f}%"
        )
    else:
        recommendations.append(f"❌ High degradation: mean={degradation_mean:.1f}%")

    # Production readiness assessment
    if production_ready_pct >= 95:
        recommendations.append(f"✅ {production_ready_pct:.0f}% iterations production-ready")
    elif production_ready_pct >= 80:
        recommendations.append(f"⚠️ {production_ready_pct:.0f}% iterations production-ready (target: 95%)")
    else:
        recommendations.append(f"❌ Only {production_ready_pct:.0f}% iterations production-ready")

    # Confidence assessment
    if confidence_level >= 0.8:
        recommendations.append(f"✅ High confidence: {confidence_level:.0%}")
    elif confidence_level >= 0.6:
        recommendations.append(f"⚠️ Moderate confidence: {confidence_level:.0%}")
    else:
        recommendations.append(f"❌ Low confidence: {confidence_level:.0%} - results highly variable")

    return "\n".join(recommendations)


def generate_monte_carlo_report(mc_result: MonteCarloResult) -> str:
    """
    Generate Monte Carlo simulation report.

    Args:
        mc_result: MonteCarloResult from run_monte_carlo_walk_forward()

    Returns:
        Formatted markdown report
    """
    report = []
    report.append("# Monte Carlo Walk-Forward Analysis Report\n")
    report.append(f"**Iterations**: {mc_result.num_iterations}")
    report.append(f"**Confidence Level**: {mc_result.confidence_level:.0%}")
    report.append(f"**Production Ready**: {mc_result.production_ready_pct:.1f}% of iterations\n")

    # Summary Statistics
    report.append("## Summary Statistics\n")
    report.append("### Sharpe Ratio (Out-of-Sample)\n")
    report.append(f"- Mean: {mc_result.sharpe_mean:.3f}")
    report.append(f"- Std Dev: {mc_result.sharpe_std:.3f}")
    report.append(f"- 5th Percentile: {mc_result.sharpe_5th_percentile:.3f}")
    report.append(f"- 95th Percentile: {mc_result.sharpe_95th_percentile:.3f}")
    report.append(
        f"- 90% Confidence Interval: [{mc_result.sharpe_5th_percentile:.3f}, {mc_result.sharpe_95th_percentile:.3f}]\n"
    )

    report.append("### Degradation (%)\n")
    report.append(f"- Mean: {mc_result.degradation_mean:.1f}%")
    report.append(f"- Std Dev: {mc_result.degradation_std:.1f}%")
    report.append(f"- 5th Percentile: {mc_result.degradation_5th_percentile:.1f}%")
    report.append(f"- 95th Percentile: {mc_result.degradation_95th_percentile:.1f}%")
    report.append(
        f"- 90% Confidence Interval: [{mc_result.degradation_5th_percentile:.1f}%, {mc_result.degradation_95th_percentile:.1f}%]\n"
    )

    report.append("### Robustness Score\n")
    report.append(f"- Mean: {mc_result.robustness_mean:.3f}")
    report.append(f"- Std Dev: {mc_result.robustness_std:.3f}\n")

    # Assessment
    report.append("## Assessment\n")
    report.append(mc_result.recommendation)

    # Interpretation Guide
    report.append("\n## Interpretation Guide\n")
    report.append("- **5th Percentile**: Worst-case scenario (5% chance of worse)")
    report.append("- **95th Percentile**: Best-case scenario (5% chance of better)")
    report.append("- **Confidence Level**: Based on result variance (higher = more stable)")
    report.append("- **Production Ready %**: % of random samplings meeting all thresholds\n")

    # Thresholds Reference
    report.append("## Thresholds Reference\n")
    report.append("| Metric | Threshold | Target |")
    report.append("|--------|-----------|--------|")
    report.append("| Sharpe Ratio | > 0.8 | > 1.0 |")
    report.append("| Degradation | < 15% | < 10% |")
    report.append("| Robustness | > 0.80 | > 0.90 |")
    report.append("| Production Ready | > 80% | > 95% |")

    return "\n".join(report)
