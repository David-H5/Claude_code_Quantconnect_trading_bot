"""
Population Stability Index (PSI) Drift Detection.

Implements PSI-based drift detection for monitoring AI trading strategy performance
degradation over time. Based on 2025 research findings showing:
- AI strategy half-life: 11 months (2025) vs 18 months (2020)
- Models unchanged for 6+ months saw 35% error rate increase
- PSI thresholds: <0.1 no drift, 0.1-0.25 growing drift, >0.25 significant shift

References:
- https://labelyourdata.com/articles/machine-learning/data-drift (Published: 2025)
- https://orq.ai/blog/model-vs-data-drift (Published: 2025)
- docs/research/EVALUATION_FRAMEWORK_RESEARCH.md
"""

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class DriftLevel(Enum):
    """Drift severity levels based on PSI thresholds."""

    NONE = "none"  # PSI < 0.10
    MODERATE = "moderate"  # 0.10 <= PSI < 0.25
    SIGNIFICANT = "significant"  # PSI >= 0.25


@dataclass
class PSIResult:
    """Result of PSI calculation for a single metric."""

    metric_name: str
    psi_score: float
    drift_level: DriftLevel
    expected_distribution: list[float]
    actual_distribution: list[float]
    bin_contributions: list[float]  # PSI contribution per bin
    interpretation: str


@dataclass
class StrategyHealthMetrics:
    """Comprehensive strategy health metrics including half-life tracking."""

    # PSI scores by metric
    psi_results: dict[str, PSIResult]

    # Overall drift assessment
    overall_psi: float
    overall_drift_level: DriftLevel
    metrics_with_drift: list[str]

    # Strategy age and decay
    strategy_age_months: float
    estimated_decay_pct: float  # Based on 11-month half-life
    days_until_review: int  # Recommended review date

    # Action recommendations
    action_required: bool
    recommendations: list[str]


# PSI Thresholds from 2025 research
PSI_THRESHOLDS = {
    "none": 0.10,
    "moderate": 0.25,
    "significant": float("inf"),
}

# Strategy half-life (months) - from 2025 research
STRATEGY_HALF_LIFE_MONTHS = 11

# Error rate increase threshold for unchanged models (6+ months)
UNCHANGED_MODEL_ERROR_INCREASE = 0.35  # 35%


def calculate_psi(
    expected: list[float],
    actual: list[float],
    bins: int = 10,
    epsilon: float = 0.0001,
) -> tuple[float, list[float]]:
    """
    Calculate Population Stability Index between expected and actual distributions.

    PSI measures how much a variable has shifted over time. It's widely used
    in credit scoring and increasingly in ML model monitoring.

    Formula: PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)

    Args:
        expected: Baseline/training distribution values
        actual: Current/production distribution values
        bins: Number of bins for histogram (default: 10)
        epsilon: Small value to avoid division by zero

    Returns:
        Tuple of (total_psi, bin_contributions)

    Thresholds (from 2025 research):
        PSI < 0.10: No significant population shift
        0.10 <= PSI < 0.25: Moderate shift, investigate
        PSI >= 0.25: Significant shift, action required
    """
    if not expected or not actual:
        return 0.0, []

    # Calculate bin edges from expected distribution
    min_val = min(min(expected), min(actual))
    max_val = max(max(expected), max(actual))
    bin_edges = [min_val + i * (max_val - min_val) / bins for i in range(bins + 1)]

    # Calculate percentages in each bin
    expected_counts = [0] * bins
    actual_counts = [0] * bins

    for val in expected:
        for i in range(bins):
            if bin_edges[i] <= val < bin_edges[i + 1]:
                expected_counts[i] += 1
                break
        else:
            expected_counts[-1] += 1  # Last bin includes max value

    for val in actual:
        for i in range(bins):
            if bin_edges[i] <= val < bin_edges[i + 1]:
                actual_counts[i] += 1
                break
        else:
            actual_counts[-1] += 1

    # Convert to percentages
    expected_pcts = [(c / len(expected)) + epsilon for c in expected_counts]
    actual_pcts = [(c / len(actual)) + epsilon for c in actual_counts]

    # Calculate PSI contribution for each bin
    bin_contributions = []
    for exp_pct, act_pct in zip(expected_pcts, actual_pcts):
        contribution = (act_pct - exp_pct) * math.log(act_pct / exp_pct)
        bin_contributions.append(contribution)

    total_psi = sum(bin_contributions)

    return total_psi, bin_contributions


def get_drift_level(psi_score: float) -> DriftLevel:
    """
    Determine drift level based on PSI score.

    Args:
        psi_score: Calculated PSI score

    Returns:
        DriftLevel enum value
    """
    if psi_score < PSI_THRESHOLDS["none"]:
        return DriftLevel.NONE
    elif psi_score < PSI_THRESHOLDS["moderate"]:
        return DriftLevel.MODERATE
    else:
        return DriftLevel.SIGNIFICANT


def calculate_psi_for_metric(
    metric_name: str,
    expected_values: list[float],
    actual_values: list[float],
    bins: int = 10,
) -> PSIResult:
    """
    Calculate PSI for a specific trading metric.

    Args:
        metric_name: Name of the metric (e.g., "sharpe_ratio", "win_rate")
        expected_values: Baseline values from training/backtest
        actual_values: Current values from production
        bins: Number of bins for histogram

    Returns:
        PSIResult with full analysis
    """
    psi_score, bin_contributions = calculate_psi(expected_values, actual_values, bins)
    drift_level = get_drift_level(psi_score)

    # Generate interpretation
    if drift_level == DriftLevel.NONE:
        interpretation = f"{metric_name} shows stable distribution (PSI={psi_score:.3f} < 0.10)"
    elif drift_level == DriftLevel.MODERATE:
        interpretation = f"{metric_name} shows moderate shift (PSI={psi_score:.3f}). Investigate cause."
    else:
        interpretation = (
            f"{metric_name} shows significant shift (PSI={psi_score:.3f} >= 0.25). Immediate action required."
        )

    return PSIResult(
        metric_name=metric_name,
        psi_score=psi_score,
        drift_level=drift_level,
        expected_distribution=expected_values,
        actual_distribution=actual_values,
        bin_contributions=bin_contributions,
        interpretation=interpretation,
    )


def estimate_strategy_decay(strategy_start_date: datetime) -> tuple[float, float]:
    """
    Estimate strategy decay based on 11-month half-life research.

    From 2025 research: Average half-life of profitable AI strategy is 11 months,
    down from 18 months in 2020.

    Args:
        strategy_start_date: When the strategy was deployed

    Returns:
        Tuple of (age_months, estimated_decay_pct)
    """
    age_days = (datetime.now() - strategy_start_date).days
    age_months = age_days / 30.0

    # Exponential decay: remaining_effectiveness = 0.5 ^ (age / half_life)
    remaining_effectiveness = 0.5 ** (age_months / STRATEGY_HALF_LIFE_MONTHS)
    decay_pct = (1.0 - remaining_effectiveness) * 100

    return age_months, decay_pct


def calculate_strategy_health(
    baseline_metrics: dict[str, list[float]],
    current_metrics: dict[str, list[float]],
    strategy_start_date: datetime | None = None,
) -> StrategyHealthMetrics:
    """
    Calculate comprehensive strategy health including PSI for all metrics.

    Args:
        baseline_metrics: Dict of metric_name -> list of baseline values
        current_metrics: Dict of metric_name -> list of current values
        strategy_start_date: When strategy was deployed (for half-life calc)

    Returns:
        StrategyHealthMetrics with full analysis
    """
    psi_results = {}
    metrics_with_drift = []

    # Calculate PSI for each metric
    for metric_name in baseline_metrics:
        if metric_name in current_metrics:
            result = calculate_psi_for_metric(
                metric_name=metric_name,
                expected_values=baseline_metrics[metric_name],
                actual_values=current_metrics[metric_name],
            )
            psi_results[metric_name] = result

            if result.drift_level != DriftLevel.NONE:
                metrics_with_drift.append(metric_name)

    # Calculate overall PSI (average of all metrics)
    if psi_results:
        overall_psi = sum(r.psi_score for r in psi_results.values()) / len(psi_results)
    else:
        overall_psi = 0.0

    overall_drift_level = get_drift_level(overall_psi)

    # Calculate strategy age and decay
    if strategy_start_date:
        age_months, decay_pct = estimate_strategy_decay(strategy_start_date)
    else:
        age_months, decay_pct = 0.0, 0.0

    # Calculate days until recommended review (at half-life)
    days_until_review = max(0, int((STRATEGY_HALF_LIFE_MONTHS - age_months) * 30))

    # Generate recommendations
    recommendations = []
    action_required = False

    # PSI-based recommendations
    if overall_drift_level == DriftLevel.SIGNIFICANT:
        action_required = True
        recommendations.append(
            "CRITICAL: Significant distribution shift detected. " "Retrain or recalibrate strategy immediately."
        )
    elif overall_drift_level == DriftLevel.MODERATE:
        recommendations.append(
            "WARNING: Moderate distribution shift detected. " "Monitor closely and plan for recalibration."
        )

    # Age-based recommendations
    if age_months >= STRATEGY_HALF_LIFE_MONTHS:
        action_required = True
        recommendations.append(
            f"CRITICAL: Strategy age ({age_months:.1f} months) exceeds half-life "
            f"({STRATEGY_HALF_LIFE_MONTHS} months). Recommend full review and update."
        )
    elif age_months >= STRATEGY_HALF_LIFE_MONTHS * 0.75:
        recommendations.append(
            f"WARNING: Strategy approaching half-life. " f"Plan review within {days_until_review} days."
        )

    # 6+ month unchanged model warning (from research: 35% error increase)
    if age_months >= 6 and not recommendations:
        recommendations.append(
            f"INFO: Strategy unchanged for {age_months:.1f} months. " f"Research shows 35% error rate increase risk."
        )

    if not recommendations:
        recommendations.append("Strategy health is good. Continue monitoring.")

    return StrategyHealthMetrics(
        psi_results=psi_results,
        overall_psi=overall_psi,
        overall_drift_level=overall_drift_level,
        metrics_with_drift=metrics_with_drift,
        strategy_age_months=age_months,
        estimated_decay_pct=decay_pct,
        days_until_review=days_until_review,
        action_required=action_required,
        recommendations=recommendations,
    )


def generate_psi_report(health: StrategyHealthMetrics) -> str:
    """
    Generate comprehensive PSI drift detection report.

    Args:
        health: StrategyHealthMetrics from calculate_strategy_health()

    Returns:
        Formatted markdown report
    """
    report = []
    report.append("# Strategy Health Report (PSI Drift Detection)\n")

    # Overall status
    status_emoji = {
        DriftLevel.NONE: "✅",
        DriftLevel.MODERATE: "⚠️",
        DriftLevel.SIGNIFICANT: "🔴",
    }

    report.append("## Overall Status\n")
    report.append(
        f"**Drift Level**: {status_emoji[health.overall_drift_level]} {health.overall_drift_level.value.upper()}"
    )
    report.append(f"**Overall PSI**: {health.overall_psi:.3f}")
    report.append(f"**Action Required**: {'YES' if health.action_required else 'No'}\n")

    # Strategy age
    report.append("## Strategy Age Analysis\n")
    report.append(f"- Strategy Age: {health.strategy_age_months:.1f} months")
    report.append(f"- Half-Life: {STRATEGY_HALF_LIFE_MONTHS} months (2025 benchmark)")
    report.append(f"- Estimated Decay: {health.estimated_decay_pct:.1f}%")
    report.append(f"- Days Until Review: {health.days_until_review}\n")

    # PSI thresholds reference
    report.append("## PSI Thresholds Reference\n")
    report.append("| Level | PSI Range | Interpretation |")
    report.append("|-------|-----------|----------------|")
    report.append("| ✅ None | < 0.10 | No significant shift |")
    report.append("| ⚠️ Moderate | 0.10 - 0.25 | Investigate cause |")
    report.append("| 🔴 Significant | >= 0.25 | Action required |\n")

    # Per-metric PSI
    report.append("## Metric-Level PSI Analysis\n")
    report.append("| Metric | PSI Score | Drift Level | Status |")
    report.append("|--------|-----------|-------------|--------|")

    for metric_name, result in health.psi_results.items():
        emoji = status_emoji[result.drift_level]
        report.append(f"| {metric_name} | {result.psi_score:.3f} | " f"{result.drift_level.value} | {emoji} |")

    report.append("")

    # Metrics with drift
    if health.metrics_with_drift:
        report.append("## Metrics Requiring Attention\n")
        for metric in health.metrics_with_drift:
            result = health.psi_results[metric]
            report.append(f"- **{metric}**: {result.interpretation}")
        report.append("")

    # Recommendations
    report.append("## Recommendations\n")
    for rec in health.recommendations:
        report.append(f"- {rec}")

    return "\n".join(report)


# Convenience function for integration with continuous_monitoring.py
def check_drift_with_psi(
    baseline_values: list[float],
    current_values: list[float],
    metric_name: str = "metric",
) -> dict[str, Any]:
    """
    Quick PSI check for integration with existing monitoring.

    Args:
        baseline_values: Expected distribution from training
        current_values: Current distribution from production
        metric_name: Name of the metric

    Returns:
        Dict with psi_score, drift_level, and action_required
    """
    result = calculate_psi_for_metric(metric_name, baseline_values, current_values)

    return {
        "psi_score": result.psi_score,
        "drift_level": result.drift_level.value,
        "action_required": result.drift_level == DriftLevel.SIGNIFICANT,
        "interpretation": result.interpretation,
    }
