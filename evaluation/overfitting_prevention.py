"""
QuantConnect-Specific Overfitting Prevention Metrics.

Implements best practices from QuantConnect's Research Guide to detect and prevent
overfitting in algorithmic trading strategies. Monitors parameter count, backtest count,
time invested, and out-of-sample validation.

References:
- https://www.quantconnect.com/docs/v2/cloud-platform/backtesting/research-guide
- https://www.quantconnect.com/blog/7-tips-for-fixing-your-strategy-backtesting-a-qa-with-top-quants/
- https://algotrading101.com/learn/quantconnect-guide/

QuantConnect Best Practices:
- Time Investment: ≤16 hours per experiment (for proficient coders)
- Backtest Count: Minimize to prevent curve-fitting
- Parameter Count: Keep to minimum (≤5 recommended)
- Out-of-Sample Period: Enforce 6-12 months before current date
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class OverfitRiskLevel(Enum):
    """Risk level for overfitting."""

    LOW = "low"  # Green - safe to proceed
    MEDIUM = "medium"  # Yellow - proceed with caution
    HIGH = "high"  # Orange - significant risk
    CRITICAL = "critical"  # Red - do not deploy


@dataclass
class OverfittingMetrics:
    """Metrics for detecting overfitting in trading strategies."""

    # Development metrics
    parameter_count: int
    backtest_count: int
    time_invested_hours: float
    hypothesis_documented: bool

    # Out-of-sample validation
    oos_period_months: int
    oos_sharpe: float
    is_sharpe: float
    sharpe_degradation_pct: float

    # Parameter stability
    parameter_sensitivity: float  # 0-1, how much results change with small param changes
    parameter_clustering: float  # 0-1, how many param sets give similar results

    # Statistical robustness
    monte_carlo_confidence: float  # 0-1, confidence from Monte Carlo simulation
    bootstrap_stability: float  # 0-1, stability across bootstrap samples

    # Overall overfitting risk
    overfit_risk_level: OverfitRiskLevel
    overfit_risk_score: float  # 0-100, higher = more risk
    production_ready: bool


def calculate_overfitting_metrics(
    parameter_count: int,
    backtest_count: int,
    time_invested_hours: float,
    hypothesis_documented: bool,
    in_sample_sharpe: float,
    out_of_sample_sharpe: float,
    oos_period_months: int,
    parameter_tests: list[dict[str, Any]] | None = None,
    monte_carlo_results: list[float] | None = None,
    bootstrap_results: list[float] | None = None,
) -> OverfittingMetrics:
    """
    Calculate overfitting risk metrics following QuantConnect best practices.

    Args:
        parameter_count: Number of optimizable parameters
        backtest_count: Number of backtests run
        time_invested_hours: Total hours spent on strategy development
        hypothesis_documented: Whether strategy has documented hypothesis
        in_sample_sharpe: Sharpe ratio on training data
        out_of_sample_sharpe: Sharpe ratio on OOS data
        oos_period_months: Out-of-sample period in months
        parameter_tests: Optional list of parameter test results
        monte_carlo_results: Optional Monte Carlo simulation Sharpe ratios
        bootstrap_results: Optional bootstrap sample Sharpe ratios

    Returns:
        OverfittingMetrics with risk assessment
    """
    # ========== DEVELOPMENT METRICS ==========

    # QuantConnect thresholds
    max_time_hours = 16  # For proficient coders
    max_params = 5
    max_backtests = 20

    # Time investment score (0-100, higher = more risk)
    time_score = min(100, (time_invested_hours / max_time_hours) * 100)

    # Parameter count score
    param_score = min(100, (parameter_count / max_params) * 100)

    # Backtest count score
    backtest_score = min(100, (backtest_count / max_backtests) * 100)

    # Hypothesis penalty (-20 if not documented)
    hypothesis_penalty = 0 if hypothesis_documented else 20

    # ========== OUT-OF-SAMPLE VALIDATION ==========

    # Sharpe degradation
    sharpe_degradation = (
        ((in_sample_sharpe - out_of_sample_sharpe) / in_sample_sharpe * 100) if in_sample_sharpe > 0 else 0
    )

    # OOS period score (higher is better, ≥12 months ideal)
    oos_period_score = min(100, (oos_period_months / 12) * 100)

    # ========== PARAMETER STABILITY ==========

    if parameter_tests:
        # Parameter sensitivity: How much results change with small parameter changes
        # Lower sensitivity = more robust
        sharpe_values = [t.get("sharpe", 0) for t in parameter_tests]
        if len(sharpe_values) > 1:
            import statistics

            param_std = statistics.stdev(sharpe_values)
            param_mean = statistics.mean(sharpe_values)
            parameter_sensitivity = param_std / param_mean if param_mean > 0 else 1.0
        else:
            parameter_sensitivity = 0.5
    else:
        parameter_sensitivity = 0.5  # Assume medium if no data

    # Parameter clustering: How many parameter sets give similar results
    # Higher clustering = more robust (not curve-fit to specific params)
    if parameter_tests and len(parameter_tests) > 5:
        # Count how many tests have Sharpe within 10% of median
        sharpe_values = [t.get("sharpe", 0) for t in parameter_tests]
        import statistics

        median_sharpe = statistics.median(sharpe_values)
        within_10pct = sum(1 for s in sharpe_values if abs(s - median_sharpe) / median_sharpe < 0.10)
        parameter_clustering = within_10pct / len(sharpe_values)
    else:
        parameter_clustering = 0.5  # Assume medium if no data

    # ========== STATISTICAL ROBUSTNESS ==========

    if monte_carlo_results and len(monte_carlo_results) > 100:
        # Confidence: What % of Monte Carlo runs beat OOS Sharpe
        beat_oos = sum(1 for mc in monte_carlo_results if mc >= out_of_sample_sharpe)
        monte_carlo_confidence = beat_oos / len(monte_carlo_results)
    else:
        monte_carlo_confidence = 0.5  # Assume medium if no data

    if bootstrap_results and len(bootstrap_results) > 100:
        # Stability: Coefficient of variation of bootstrap Sharpe ratios
        import statistics

        boot_mean = statistics.mean(bootstrap_results)
        boot_std = statistics.stdev(bootstrap_results)
        cv = boot_std / boot_mean if boot_mean > 0 else 1.0
        bootstrap_stability = max(0, 1.0 - cv)  # Lower CV = higher stability
    else:
        bootstrap_stability = 0.5  # Assume medium if no data

    # ========== OVERALL OVERFITTING RISK ==========

    # Calculate risk score (0-100, higher = more risk)
    weights = {
        "time": 0.15,
        "params": 0.20,
        "backtests": 0.15,
        "sharpe_degradation": 0.25,
        "oos_period": -0.10,  # Negative weight (longer OOS reduces risk)
        "param_sensitivity": 0.10,
        "hypothesis": 0.05,
    }

    # Normalize degradation to 0-100 scale (>30% = 100)
    degradation_score = min(100, (sharpe_degradation / 30) * 100)

    overfit_risk_score = (
        weights["time"] * time_score
        + weights["params"] * param_score
        + weights["backtests"] * backtest_score
        + weights["sharpe_degradation"] * degradation_score
        + weights["oos_period"] * oos_period_score
        + weights["param_sensitivity"] * (parameter_sensitivity * 100)
        + weights["hypothesis"] * hypothesis_penalty
    )

    # Clamp to 0-100
    overfit_risk_score = max(0, min(100, overfit_risk_score))

    # Determine risk level
    if overfit_risk_score < 25:
        risk_level = OverfitRiskLevel.LOW
    elif overfit_risk_score < 50:
        risk_level = OverfitRiskLevel.MEDIUM
    elif overfit_risk_score < 75:
        risk_level = OverfitRiskLevel.HIGH
    else:
        risk_level = OverfitRiskLevel.CRITICAL

    # Production ready if:
    # - Risk level LOW or MEDIUM
    # - OOS period >= 6 months
    # - Sharpe degradation < 20%
    # - OOS Sharpe > 1.0
    production_ready = (
        risk_level in [OverfitRiskLevel.LOW, OverfitRiskLevel.MEDIUM]
        and oos_period_months >= 6
        and sharpe_degradation < 20
        and out_of_sample_sharpe > 1.0
    )

    return OverfittingMetrics(
        parameter_count=parameter_count,
        backtest_count=backtest_count,
        time_invested_hours=time_invested_hours,
        hypothesis_documented=hypothesis_documented,
        oos_period_months=oos_period_months,
        oos_sharpe=out_of_sample_sharpe,
        is_sharpe=in_sample_sharpe,
        sharpe_degradation_pct=sharpe_degradation,
        parameter_sensitivity=parameter_sensitivity,
        parameter_clustering=parameter_clustering,
        monte_carlo_confidence=monte_carlo_confidence,
        bootstrap_stability=bootstrap_stability,
        overfit_risk_level=risk_level,
        overfit_risk_score=overfit_risk_score,
        production_ready=production_ready,
    )


def generate_overfitting_report(metrics: OverfittingMetrics) -> str:
    """
    Generate overfitting risk assessment report.

    Args:
        metrics: OverfittingMetrics object

    Returns:
        Formatted markdown report
    """
    report = []
    report.append("# Overfitting Risk Assessment (QuantConnect Best Practices)\n")

    # Risk level indicator
    risk_icon = {
        OverfitRiskLevel.LOW: "🟢",
        OverfitRiskLevel.MEDIUM: "🟡",
        OverfitRiskLevel.HIGH: "🟠",
        OverfitRiskLevel.CRITICAL: "🔴",
    }[metrics.overfit_risk_level]

    report.append(f"**Overfitting Risk**: {risk_icon} {metrics.overfit_risk_level.value.upper()}")
    report.append(f"**Risk Score**: {metrics.overfit_risk_score:.1f}/100")
    report.append(f"**Production Ready**: {'✅ YES' if metrics.production_ready else '⚠️ NO'}\n")

    # Development metrics
    report.append("## 🔨 Development Metrics\n")
    report.append(
        f"- Parameter Count: {metrics.parameter_count} {'✅' if metrics.parameter_count <= 5 else '⚠️'} (Target: ≤5)"
    )
    report.append(
        f"- Backtest Count: {metrics.backtest_count} {'✅' if metrics.backtest_count <= 20 else '⚠️'} (Target: ≤20)"
    )
    report.append(
        f"- Time Invested: {metrics.time_invested_hours:.1f}h {'✅' if metrics.time_invested_hours <= 16 else '⚠️'} (Target: ≤16h)"
    )
    report.append(f"- Hypothesis Documented: {'✅ YES' if metrics.hypothesis_documented else '❌ NO'}\n")

    # Out-of-sample validation
    report.append("## 📊 Out-of-Sample Validation\n")
    report.append(
        f"- OOS Period: {metrics.oos_period_months} months {'✅' if metrics.oos_period_months >= 12 else '⚠️'} (Target: ≥12 months)"
    )
    report.append(f"- In-Sample Sharpe: {metrics.is_sharpe:.2f}")
    report.append(f"- Out-of-Sample Sharpe: {metrics.oos_sharpe:.2f}")
    report.append(
        f"- Sharpe Degradation: {metrics.sharpe_degradation_pct:.1f}% {'✅' if metrics.sharpe_degradation_pct < 20 else '⚠️'} (Target: <20%)\n"
    )

    # Parameter stability
    report.append("## 🎯 Parameter Stability\n")
    report.append(
        f"- Parameter Sensitivity: {metrics.parameter_sensitivity:.2f} {'✅' if metrics.parameter_sensitivity < 0.30 else '⚠️'} (Lower is better)"
    )
    report.append(
        f"- Parameter Clustering: {metrics.parameter_clustering:.2f} {'✅' if metrics.parameter_clustering > 0.60 else '⚠️'} (Higher is better)\n"
    )

    # Statistical robustness
    report.append("## 📈 Statistical Robustness\n")
    report.append(f"- Monte Carlo Confidence: {metrics.monte_carlo_confidence:.1%}")
    report.append(f"- Bootstrap Stability: {metrics.bootstrap_stability:.1%}\n")

    # Recommendations
    report.append("## 💡 Recommendations\n")

    recommendations = []

    if metrics.parameter_count > 5:
        recommendations.append("❌ **Reduce parameter count** to ≤5 to avoid curve-fitting")

    if metrics.backtest_count > 20:
        recommendations.append("❌ **Too many backtests** - high risk of finding patterns by chance")

    if metrics.time_invested_hours > 16:
        recommendations.append("⚠️ **Excessive time invested** - may indicate over-optimization")

    if not metrics.hypothesis_documented:
        recommendations.append("❌ **Document trading hypothesis** before further development")

    if metrics.oos_period_months < 12:
        recommendations.append(f"⚠️ **Extend OOS period** to 12 months (currently {metrics.oos_period_months} months)")

    if metrics.sharpe_degradation_pct > 20:
        recommendations.append(
            f"❌ **Excessive Sharpe degradation** ({metrics.sharpe_degradation_pct:.1f}%) - strategy may not generalize"
        )

    if metrics.parameter_sensitivity > 0.30:
        recommendations.append("⚠️ **High parameter sensitivity** - small changes significantly affect results")

    if metrics.parameter_clustering < 0.60:
        recommendations.append("⚠️ **Low parameter clustering** - results depend too much on specific parameters")

    if recommendations:
        for rec in recommendations:
            report.append(rec)
    else:
        report.append("✅ **All checks passed** - strategy shows good statistical robustness")

    # Overall verdict
    report.append("\n## 🏁 Verdict\n")

    if metrics.overfit_risk_level == OverfitRiskLevel.LOW:
        report.append(
            "✅ **LOW RISK** - Strategy is well-developed with minimal overfitting risk. Safe to proceed to paper trading."
        )
    elif metrics.overfit_risk_level == OverfitRiskLevel.MEDIUM:
        report.append(
            "🟡 **MEDIUM RISK** - Strategy shows some overfitting indicators. Address recommendations before deployment."
        )
    elif metrics.overfit_risk_level == OverfitRiskLevel.HIGH:
        report.append(
            "🟠 **HIGH RISK** - Significant overfitting detected. Major improvements required before deployment."
        )
    else:
        report.append(
            "🔴 **CRITICAL RISK** - Severe overfitting. Do NOT deploy. Restart development with hypothesis-driven approach."
        )

    return "\n".join(report)


def validate_quantconnect_best_practices(
    parameter_count: int,
    backtest_count: int,
    time_invested_hours: float,
    oos_period_months: int,
) -> dict[str, bool]:
    """
    Validate against QuantConnect best practices.

    Args:
        parameter_count: Number of parameters
        backtest_count: Number of backtests
        time_invested_hours: Time invested
        oos_period_months: OOS period

    Returns:
        Dict of validation results
    """
    return {
        "parameter_count_ok": parameter_count <= 5,
        "backtest_count_ok": backtest_count <= 20,
        "time_invested_ok": time_invested_hours <= 16,
        "oos_period_ok": oos_period_months >= 12,
    }
