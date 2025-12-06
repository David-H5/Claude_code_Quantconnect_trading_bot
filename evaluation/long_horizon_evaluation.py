"""
Long-Horizon Backtesting Evaluation Module.

Implements long-horizon backtesting inspired by FINSABER (Financial Safety-Aware
Backtesting Environment) research findings that LLM strategy advantages deteriorate
significantly over longer time periods.

Key Research Findings (FINSABER 2025):
- LLM strategies that look promising in 2-year tests often fail in 20-year tests
- Strategy degradation accelerates after 5 years
- Regime-specific performance varies significantly
- Long-horizon testing is essential for robust validation

Testing Horizons:
- Short-term: 2 years (standard backtest)
- Medium-term: 5 years (extended validation)
- Long-term: 10 years (robust validation)
- Ultra-long: 15-20 years (stress testing)

References:
- https://arxiv.org/abs/2502.17979 (FINSABER)
- docs/research/EVALUATION_UPGRADE_GUIDE.md
- docs/research/EVALUATION_FRAMEWORK_RESEARCH.md

Version: 1.0 (December 2025)
"""

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class DegradationSeverity(Enum):
    """Severity of performance degradation over time."""

    NONE = "none"  # < 10% degradation
    MINOR = "minor"  # 10-25% degradation
    MODERATE = "moderate"  # 25-50% degradation
    SEVERE = "severe"  # 50-75% degradation
    CRITICAL = "critical"  # > 75% degradation


class LongTermViability(Enum):
    """Long-term strategy viability classification."""

    ROBUST = "robust"  # Viable for 10+ years
    ACCEPTABLE = "acceptable"  # Viable for 5-10 years
    LIMITED = "limited"  # Viable for 2-5 years
    UNSTABLE = "unstable"  # < 2 years viability
    UNKNOWN = "unknown"  # Insufficient data


@dataclass
class HorizonResult:
    """Results for a specific time horizon."""

    horizon_years: int
    start_date: datetime
    end_date: datetime

    # Performance metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    cagr: float
    win_rate: float
    profit_factor: float
    total_return: float

    # Trade statistics
    total_trades: int
    avg_trade_duration_days: float

    # Degradation metrics
    degradation_vs_short: float  # % degradation vs 2-year baseline
    degradation_vs_previous: float  # % degradation vs previous horizon

    # Market regime breakdown
    regime_performance: dict[str, float]  # Performance by regime

    # Quality assessment
    is_profitable: bool
    meets_threshold: bool  # Sharpe > threshold


@dataclass
class LongHorizonMetrics:
    """Comprehensive long-horizon backtesting metrics."""

    # Horizon results
    horizon_results: list[HorizonResult]
    horizons_tested: list[int]

    # Degradation analysis
    degradation_rate_annual: float  # Annual degradation percentage
    degradation_acceleration: float  # Rate of degradation increase
    stable_horizon_years: int  # Years before significant degradation
    half_life_years: float  # Estimated performance half-life

    # Long-term viability
    long_term_viability: LongTermViability
    viability_confidence: float  # 0-1 confidence in assessment

    # Regime consistency
    regime_consistency: float  # 0-1, performance consistency across regimes
    worst_regime: str  # Regime with worst performance
    best_regime: str  # Regime with best performance

    # Risk metrics
    max_consecutive_losses: int
    recovery_time_months: float  # Avg time to recover from drawdown
    tail_risk_ratio: float  # Worst 5% vs avg returns

    # Recommendations
    recommendations: list[str]
    overall_assessment: str


@dataclass
class RegimePerformance:
    """Performance metrics for a specific market regime."""

    regime_name: str
    start_date: datetime
    end_date: datetime
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_return: float


# FINSABER-inspired thresholds
LONG_HORIZON_THRESHOLDS = {
    "max_degradation_rate_annual": 0.05,  # < 5% annual degradation
    "min_stable_horizon_years": 10,  # At least 10 years stable
    "min_sharpe_at_10_years": 0.5,  # Sharpe > 0.5 at 10+ years
    "min_sharpe_at_5_years": 0.8,  # Sharpe > 0.8 at 5 years
    "min_regime_consistency": 0.70,  # 70% consistency across regimes
    "max_degradation_acceleration": 0.02,  # < 2% acceleration per year
    "min_viability_confidence": 0.80,  # 80% confidence threshold
}

# Market regime definitions
MARKET_REGIMES = {
    "bull_market": {"description": "Sustained uptrend", "typical_duration_months": 24},
    "bear_market": {"description": "Sustained downtrend", "typical_duration_months": 12},
    "high_volatility": {"description": "VIX > 25", "typical_duration_months": 6},
    "low_volatility": {"description": "VIX < 15", "typical_duration_months": 18},
    "recession": {"description": "Economic contraction", "typical_duration_months": 12},
    "recovery": {"description": "Post-recession growth", "typical_duration_months": 18},
    "sideways": {"description": "Range-bound market", "typical_duration_months": 12},
    "crisis": {"description": "Black swan event", "typical_duration_months": 3},
}


def run_long_horizon_backtest(
    algorithm_path: str,
    data_start: datetime,
    data_end: datetime,
    horizons: list[int] | None = None,
    backtest_func: Callable | None = None,
    baseline_sharpe: float | None = None,
) -> LongHorizonMetrics:
    """
    Run backtests across multiple time horizons.

    Per FINSABER research, LLM strategies often show:
    - Good 2-year performance
    - Degraded 5-year performance
    - Significantly worse 10+ year performance

    Args:
        algorithm_path: Path to algorithm file
        data_start: Earliest data date available
        data_end: Latest data date available
        horizons: List of horizon years to test (default: [2, 5, 10, 15, 20])
        backtest_func: Function to run backtest (algorithm_path, start, end) -> metrics
        baseline_sharpe: Baseline Sharpe for degradation calculation

    Returns:
        LongHorizonMetrics with comprehensive analysis
    """
    if horizons is None:
        horizons = [2, 5, 10, 15, 20]

    # Filter horizons to available data
    available_years = (data_end - data_start).days / 365.25
    horizons = [h for h in horizons if h <= available_years]

    if not horizons:
        raise ValueError(f"No valid horizons for {available_years:.1f} years of data")

    results: list[HorizonResult] = []
    base_sharpe = baseline_sharpe

    for years in sorted(horizons):
        # Calculate date range for this horizon
        end_date = data_end
        start_date = end_date - timedelta(days=int(years * 365.25))

        # Run backtest (or use mock for testing)
        if backtest_func:
            metrics = backtest_func(algorithm_path, start_date, end_date)
        else:
            # Mock backtest with degradation model
            metrics = _mock_backtest_with_degradation(years, base_sharpe)

        # Store baseline for degradation calculation
        if base_sharpe is None and years == min(horizons):
            base_sharpe = metrics.get("sharpe_ratio", 1.5)

        # Calculate degradation
        current_sharpe = metrics.get("sharpe_ratio", 0)
        degradation_vs_short = 0.0
        if base_sharpe and base_sharpe > 0:
            degradation_vs_short = (base_sharpe - current_sharpe) / base_sharpe

        # Degradation vs previous horizon
        degradation_vs_prev = 0.0
        if results:
            prev_sharpe = results[-1].sharpe_ratio
            if prev_sharpe > 0:
                degradation_vs_prev = (prev_sharpe - current_sharpe) / prev_sharpe

        # Create horizon result
        result = HorizonResult(
            horizon_years=years,
            start_date=start_date,
            end_date=end_date,
            sharpe_ratio=current_sharpe,
            sortino_ratio=metrics.get("sortino_ratio", current_sharpe * 1.2),
            max_drawdown=metrics.get("max_drawdown", 0.15 + years * 0.01),
            cagr=metrics.get("cagr", 0.10 - years * 0.005),
            win_rate=metrics.get("win_rate", 0.55 - years * 0.005),
            profit_factor=metrics.get("profit_factor", 1.5 - years * 0.02),
            total_return=metrics.get("total_return", (1.10**years) - 1),
            total_trades=metrics.get("total_trades", int(250 * years)),
            avg_trade_duration_days=metrics.get("avg_trade_duration_days", 5.0),
            degradation_vs_short=degradation_vs_short,
            degradation_vs_previous=degradation_vs_prev,
            regime_performance=metrics.get("regime_performance", {}),
            is_profitable=current_sharpe > 0,
            meets_threshold=current_sharpe >= 0.5,
        )
        results.append(result)

    # Calculate aggregate metrics
    return _calculate_long_horizon_metrics(results, horizons)


def _mock_backtest_with_degradation(years: int, base_sharpe: float | None) -> dict[str, Any]:
    """
    Mock backtest with realistic degradation model.

    Based on FINSABER findings that LLM strategy performance degrades over time.
    """
    base = base_sharpe or 1.5

    # Degradation model: exponential decay
    # ~10% degradation at 5 years, ~30% at 10 years, ~50% at 20 years
    degradation_factor = math.exp(-0.035 * years)
    sharpe = base * degradation_factor

    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sharpe * 1.2,
        "max_drawdown": 0.12 + 0.01 * years,
        "cagr": 0.12 - 0.005 * years,
        "win_rate": 0.58 - 0.005 * years,
        "profit_factor": 1.6 - 0.02 * years,
        "total_return": (1.10**years) - 1,
        "total_trades": int(250 * years),
        "avg_trade_duration_days": 5.0,
        "regime_performance": {
            "bull_market": sharpe * 1.2,
            "bear_market": sharpe * 0.6,
            "high_volatility": sharpe * 0.8,
            "sideways": sharpe * 0.9,
        },
    }


def _calculate_long_horizon_metrics(
    results: list[HorizonResult],
    horizons: list[int],
) -> LongHorizonMetrics:
    """Calculate comprehensive long-horizon metrics from individual results."""

    if not results:
        return _empty_long_horizon_metrics()

    # Calculate degradation rate using linear regression
    sharpes = [r.sharpe_ratio for r in results]
    years_list = [r.horizon_years for r in results]

    if len(results) >= 2:
        # Simple linear regression for degradation rate
        n = len(years_list)
        sum_x = sum(years_list)
        sum_y = sum(sharpes)
        sum_xy = sum(years_list[i] * sharpes[i] for i in range(n))
        sum_xx = sum(x**2 for x in years_list)

        denom = n * sum_xx - sum_x**2
        if denom != 0:
            slope = (n * sum_xy - sum_x * sum_y) / denom
            # Annual degradation rate (negative slope means degradation)
            base_sharpe = results[0].sharpe_ratio
            degradation_rate = abs(slope / base_sharpe) if base_sharpe > 0 else 0
        else:
            degradation_rate = 0.0
    else:
        degradation_rate = 0.0

    # Calculate degradation acceleration
    if len(results) >= 3:
        # Second derivative approximation
        degradation_values = [r.degradation_vs_short for r in results]
        accelerations = []
        for i in range(1, len(degradation_values) - 1):
            accel = degradation_values[i + 1] - 2 * degradation_values[i] + degradation_values[i - 1]
            accelerations.append(accel)
        degradation_acceleration = statistics.mean(accelerations) if accelerations else 0.0
    else:
        degradation_acceleration = 0.0

    # Find stable horizon (where Sharpe stays > 0.5)
    stable_horizon = 0
    for r in results:
        if r.sharpe_ratio >= 0.5:
            stable_horizon = r.horizon_years
        else:
            break

    # Calculate half-life (when Sharpe drops to half of initial)
    base_sharpe = results[0].sharpe_ratio if results else 1.0
    half_life = 0.0
    for r in results:
        if r.sharpe_ratio <= base_sharpe / 2:
            half_life = r.horizon_years
            break
    if half_life == 0 and results:
        # Extrapolate based on degradation rate
        if degradation_rate > 0:
            half_life = 0.5 / degradation_rate  # 50% / annual rate
        else:
            half_life = 100  # Effectively no degradation

    # Assess long-term viability
    viability = _assess_viability(results, degradation_rate, stable_horizon)

    # Calculate viability confidence
    viability_confidence = _calculate_viability_confidence(results, degradation_rate)

    # Regime consistency analysis
    regime_consistency, worst_regime, best_regime = _analyze_regime_consistency(results)

    # Risk metrics
    max_consecutive_losses = max((r.total_trades * (1 - r.win_rate) * 0.1) for r in results) if results else 0
    recovery_time = statistics.mean([r.max_drawdown / (r.cagr + 0.01) * 12 for r in results]) if results else 0
    tail_risk = 1.5  # Placeholder

    # Generate recommendations
    recommendations = _generate_long_horizon_recommendations(
        results, degradation_rate, stable_horizon, regime_consistency, viability
    )

    # Overall assessment
    overall = _generate_overall_assessment(viability, degradation_rate, stable_horizon, regime_consistency)

    return LongHorizonMetrics(
        horizon_results=results,
        horizons_tested=horizons,
        degradation_rate_annual=degradation_rate,
        degradation_acceleration=degradation_acceleration,
        stable_horizon_years=stable_horizon,
        half_life_years=half_life,
        long_term_viability=viability,
        viability_confidence=viability_confidence,
        regime_consistency=regime_consistency,
        worst_regime=worst_regime,
        best_regime=best_regime,
        max_consecutive_losses=int(max_consecutive_losses),
        recovery_time_months=recovery_time,
        tail_risk_ratio=tail_risk,
        recommendations=recommendations,
        overall_assessment=overall,
    )


def _empty_long_horizon_metrics() -> LongHorizonMetrics:
    """Return empty long-horizon metrics."""
    return LongHorizonMetrics(
        horizon_results=[],
        horizons_tested=[],
        degradation_rate_annual=0.0,
        degradation_acceleration=0.0,
        stable_horizon_years=0,
        half_life_years=0.0,
        long_term_viability=LongTermViability.UNKNOWN,
        viability_confidence=0.0,
        regime_consistency=0.0,
        worst_regime="unknown",
        best_regime="unknown",
        max_consecutive_losses=0,
        recovery_time_months=0.0,
        tail_risk_ratio=0.0,
        recommendations=["No data available for analysis"],
        overall_assessment="Unable to assess - insufficient data",
    )


def _assess_viability(
    results: list[HorizonResult],
    degradation_rate: float,
    stable_horizon: int,
) -> LongTermViability:
    """Assess long-term strategy viability."""

    # Check if we have 10+ year data with positive Sharpe
    has_long_term_data = any(r.horizon_years >= 10 for r in results)
    long_term_sharpe = next((r.sharpe_ratio for r in results if r.horizon_years >= 10), None)

    if has_long_term_data and long_term_sharpe:
        if long_term_sharpe >= 0.5 and stable_horizon >= 10:
            return LongTermViability.ROBUST
        elif long_term_sharpe >= 0.3 and stable_horizon >= 5:
            return LongTermViability.ACCEPTABLE
        elif long_term_sharpe > 0:
            return LongTermViability.LIMITED
        else:
            return LongTermViability.UNSTABLE

    # Without long-term data, use degradation rate
    if degradation_rate < 0.03:  # < 3% annual degradation
        return LongTermViability.ROBUST
    elif degradation_rate < 0.07:  # < 7% annual
        return LongTermViability.ACCEPTABLE
    elif degradation_rate < 0.15:  # < 15% annual
        return LongTermViability.LIMITED
    else:
        return LongTermViability.UNSTABLE


def _calculate_viability_confidence(
    results: list[HorizonResult],
    degradation_rate: float,
) -> float:
    """Calculate confidence in viability assessment."""

    # Factors affecting confidence:
    # 1. Number of horizons tested
    # 2. Data consistency
    # 3. Degradation predictability

    # Base confidence from data coverage
    max_horizon = max(r.horizon_years for r in results) if results else 0
    coverage_score = min(1.0, max_horizon / 20)  # Max confidence at 20 years

    # Consistency score
    if len(results) >= 3:
        sharpes = [r.sharpe_ratio for r in results]
        mean_sharpe = statistics.mean(sharpes)
        std_sharpe = statistics.stdev(sharpes) if len(sharpes) > 1 else 0
        consistency_score = max(0, 1 - std_sharpe / (mean_sharpe + 0.1))
    else:
        consistency_score = 0.5

    # Predictability score (lower degradation acceleration = more predictable)
    predictability_score = max(0, 1 - abs(degradation_rate) * 5)

    # Weighted average
    confidence = coverage_score * 0.4 + consistency_score * 0.3 + predictability_score * 0.3

    return min(1.0, max(0.0, confidence))


def _analyze_regime_consistency(
    results: list[HorizonResult],
) -> tuple:
    """Analyze performance consistency across market regimes."""

    # Aggregate regime performance
    regime_sharpes: dict[str, list[float]] = {}

    for result in results:
        for regime, sharpe in result.regime_performance.items():
            if regime not in regime_sharpes:
                regime_sharpes[regime] = []
            regime_sharpes[regime].append(sharpe)

    if not regime_sharpes:
        return 0.5, "unknown", "unknown"

    # Calculate average Sharpe per regime
    avg_regime_sharpes = {regime: statistics.mean(sharpes) for regime, sharpes in regime_sharpes.items()}

    # Consistency = 1 - (std / mean) across regimes
    regime_values = list(avg_regime_sharpes.values())
    if len(regime_values) > 1:
        mean_val = statistics.mean(regime_values)
        std_val = statistics.stdev(regime_values)
        consistency = max(0, 1 - std_val / (abs(mean_val) + 0.1))
    else:
        consistency = 0.5

    # Find worst and best regimes
    worst_regime = min(avg_regime_sharpes, key=avg_regime_sharpes.get)
    best_regime = max(avg_regime_sharpes, key=avg_regime_sharpes.get)

    return consistency, worst_regime, best_regime


def _generate_long_horizon_recommendations(
    results: list[HorizonResult],
    degradation_rate: float,
    stable_horizon: int,
    regime_consistency: float,
    viability: LongTermViability,
) -> list[str]:
    """Generate recommendations based on long-horizon analysis."""

    recommendations = []
    thresholds = LONG_HORIZON_THRESHOLDS

    # Degradation rate assessment
    if degradation_rate > thresholds["max_degradation_rate_annual"]:
        recommendations.append(
            f"HIGH: Annual degradation rate ({degradation_rate:.1%}) exceeds threshold "
            f"({thresholds['max_degradation_rate_annual']:.1%}). Strategy may need regular recalibration."
        )
    else:
        recommendations.append(f"OK: Degradation rate ({degradation_rate:.1%}) is within acceptable limits.")

    # Stable horizon assessment
    if stable_horizon < thresholds["min_stable_horizon_years"]:
        recommendations.append(
            f"WARNING: Stable horizon ({stable_horizon} years) below target "
            f"({thresholds['min_stable_horizon_years']} years). Strategy viability is limited."
        )
    else:
        recommendations.append(f"OK: Strategy remains stable for {stable_horizon}+ years.")

    # Regime consistency
    if regime_consistency < thresholds["min_regime_consistency"]:
        recommendations.append(
            f"MEDIUM: Regime consistency ({regime_consistency:.1%}) below threshold. "
            "Consider regime-adaptive strategies."
        )

    # Long-term Sharpe check
    long_term_results = [r for r in results if r.horizon_years >= 10]
    if long_term_results:
        lt_sharpe = long_term_results[0].sharpe_ratio
        if lt_sharpe < thresholds["min_sharpe_at_10_years"]:
            recommendations.append(
                f"WARNING: 10-year Sharpe ({lt_sharpe:.2f}) below minimum threshold "
                f"({thresholds['min_sharpe_at_10_years']}). Long-term profitability at risk."
            )

    # Viability-based recommendations
    if viability == LongTermViability.UNSTABLE:
        recommendations.append(
            "CRITICAL: Strategy shows unstable long-term behavior. "
            "Major architectural changes recommended before production deployment."
        )
    elif viability == LongTermViability.LIMITED:
        recommendations.append(
            "WARNING: Strategy has limited long-term viability. " "Plan for regular strategy updates (every 2-3 years)."
        )

    # FINSABER-specific recommendation
    if results and results[0].horizon_years <= 2:
        if results[0].sharpe_ratio > 2.0:
            recommendations.append(
                "CAUTION: High 2-year Sharpe may indicate overfitting. "
                "Per FINSABER research, verify performance on longer horizons."
            )

    return recommendations


def _generate_overall_assessment(
    viability: LongTermViability,
    degradation_rate: float,
    stable_horizon: int,
    regime_consistency: float,
) -> str:
    """Generate overall assessment summary."""

    if viability == LongTermViability.ROBUST:
        return (
            f"ROBUST - Strategy shows strong long-term viability. "
            f"Stable for {stable_horizon}+ years with {degradation_rate:.1%} annual degradation. "
            f"Regime consistency: {regime_consistency:.1%}. Suitable for long-term deployment."
        )
    elif viability == LongTermViability.ACCEPTABLE:
        return (
            f"ACCEPTABLE - Strategy viable for medium-term deployment. "
            f"Stable for {stable_horizon} years. Plan recalibration every 3-5 years. "
            f"Monitor regime-specific performance."
        )
    elif viability == LongTermViability.LIMITED:
        return (
            f"LIMITED - Strategy has short-term viability only. "
            f"Significant degradation ({degradation_rate:.1%}/year) observed. "
            f"Requires frequent updates (every 1-2 years) to maintain profitability."
        )
    elif viability == LongTermViability.UNSTABLE:
        return (
            "UNSTABLE - Strategy not suitable for production deployment. "
            "Severe degradation and/or regime sensitivity detected. "
            "Major redesign recommended before use."
        )
    else:
        return (
            "UNKNOWN - Insufficient data for assessment. "
            "Extend backtest period to at least 5 years for meaningful analysis."
        )


def generate_long_horizon_report(metrics: LongHorizonMetrics) -> str:
    """
    Generate formatted long-horizon analysis report.

    Args:
        metrics: LongHorizonMetrics from analysis

    Returns:
        Formatted markdown report
    """
    lines = []
    lines.append("# Long-Horizon Backtesting Report\n")
    lines.append("*Based on FINSABER methodology*\n")

    # Summary
    lines.append("## Executive Summary\n")
    lines.append(f"**Long-Term Viability**: {metrics.long_term_viability.value.upper()}")
    lines.append(f"**Confidence Level**: {metrics.viability_confidence:.0%}")
    lines.append(f"**Stable Horizon**: {metrics.stable_horizon_years} years")
    lines.append(f"**Performance Half-Life**: {metrics.half_life_years:.1f} years")
    lines.append(f"**Annual Degradation Rate**: {metrics.degradation_rate_annual:.1%}\n")

    lines.append(f"**Assessment**: {metrics.overall_assessment}\n")

    # Horizon Results Table
    lines.append("## Performance by Horizon\n")
    lines.append("| Horizon | Sharpe | Max DD | Win Rate | CAGR | Degradation |")
    lines.append("|---------|--------|--------|----------|------|-------------|")

    for r in metrics.horizon_results:
        lines.append(
            f"| {r.horizon_years} years | {r.sharpe_ratio:.2f} | "
            f"{r.max_drawdown:.1%} | {r.win_rate:.1%} | "
            f"{r.cagr:.1%} | {r.degradation_vs_short:.1%} |"
        )
    lines.append("")

    # Degradation Analysis
    lines.append("## Degradation Analysis\n")
    lines.append(f"- **Annual Degradation Rate**: {metrics.degradation_rate_annual:.2%}")
    lines.append(f"- **Degradation Acceleration**: {metrics.degradation_acceleration:.3f}")
    lines.append(f"- **Half-Life**: {metrics.half_life_years:.1f} years")
    lines.append(f"- **Stable Horizon**: {metrics.stable_horizon_years} years\n")

    # FINSABER thresholds
    lines.append("### FINSABER Thresholds\n")
    lines.append("| Metric | Value | Threshold | Status |")
    lines.append("|--------|-------|-----------|--------|")

    thresholds = LONG_HORIZON_THRESHOLDS
    deg_status = "OK" if metrics.degradation_rate_annual < thresholds["max_degradation_rate_annual"] else "ALERT"
    stable_status = "OK" if metrics.stable_horizon_years >= thresholds["min_stable_horizon_years"] else "ALERT"

    lines.append(
        f"| Degradation Rate | {metrics.degradation_rate_annual:.1%} | < {thresholds['max_degradation_rate_annual']:.1%} | {deg_status} |"
    )
    lines.append(
        f"| Stable Horizon | {metrics.stable_horizon_years} years | >= {thresholds['min_stable_horizon_years']} years | {stable_status} |"
    )
    lines.append(
        f"| Regime Consistency | {metrics.regime_consistency:.1%} | >= {thresholds['min_regime_consistency']:.0%} | {'OK' if metrics.regime_consistency >= thresholds['min_regime_consistency'] else 'ALERT'} |"
    )
    lines.append("")

    # Regime Analysis
    lines.append("## Market Regime Analysis\n")
    lines.append(f"- **Regime Consistency**: {metrics.regime_consistency:.1%}")
    lines.append(f"- **Best Performing Regime**: {metrics.best_regime}")
    lines.append(f"- **Worst Performing Regime**: {metrics.worst_regime}\n")

    # Risk Metrics
    lines.append("## Risk Metrics\n")
    lines.append(f"- **Max Consecutive Losses**: {metrics.max_consecutive_losses}")
    lines.append(f"- **Avg Recovery Time**: {metrics.recovery_time_months:.1f} months")
    lines.append(f"- **Tail Risk Ratio**: {metrics.tail_risk_ratio:.2f}\n")

    # Recommendations
    lines.append("## Recommendations\n")
    for rec in metrics.recommendations:
        lines.append(f"- {rec}")

    # Interpretation Guide
    lines.append("\n## Interpretation Guide\n")
    lines.append("### Viability Levels\n")
    lines.append("- **ROBUST**: Viable for 10+ years, suitable for long-term deployment")
    lines.append("- **ACCEPTABLE**: Viable for 5-10 years, requires periodic review")
    lines.append("- **LIMITED**: Viable for 2-5 years, requires frequent updates")
    lines.append("- **UNSTABLE**: Not recommended for production\n")

    lines.append("### Key Insights from FINSABER Research\n")
    lines.append("- LLM strategies often show 2-year outperformance but fail over 20 years")
    lines.append("- Strategies with > 5% annual degradation need regular recalibration")
    lines.append("- Regime-dependent strategies may appear robust but fail in new regimes")
    lines.append("- Always validate short-term winners with long-horizon testing")

    return "\n".join(lines)


def check_long_horizon_thresholds(metrics: LongHorizonMetrics) -> dict[str, Any]:
    """
    Check long-horizon metrics against FINSABER thresholds.

    Args:
        metrics: LongHorizonMetrics to check

    Returns:
        Dict with compliance status and details
    """
    thresholds = LONG_HORIZON_THRESHOLDS
    issues = []
    warnings = []

    # Check degradation rate
    if metrics.degradation_rate_annual > thresholds["max_degradation_rate_annual"] * 2:
        issues.append(f"Critical: Degradation rate ({metrics.degradation_rate_annual:.1%}) " f"exceeds 2x threshold")
    elif metrics.degradation_rate_annual > thresholds["max_degradation_rate_annual"]:
        warnings.append(
            f"Warning: Degradation rate ({metrics.degradation_rate_annual:.1%}) "
            f"exceeds threshold ({thresholds['max_degradation_rate_annual']:.1%})"
        )

    # Check stable horizon
    if metrics.stable_horizon_years < 5:
        issues.append(f"Critical: Stable horizon ({metrics.stable_horizon_years} years) too short")
    elif metrics.stable_horizon_years < thresholds["min_stable_horizon_years"]:
        warnings.append(
            f"Warning: Stable horizon ({metrics.stable_horizon_years} years) "
            f"below target ({thresholds['min_stable_horizon_years']} years)"
        )

    # Check regime consistency
    if metrics.regime_consistency < thresholds["min_regime_consistency"]:
        warnings.append(
            f"Warning: Regime consistency ({metrics.regime_consistency:.1%}) "
            f"below threshold ({thresholds['min_regime_consistency']:.0%})"
        )

    # Check 10-year Sharpe
    long_term = next((r for r in metrics.horizon_results if r.horizon_years >= 10), None)
    if long_term and long_term.sharpe_ratio < thresholds["min_sharpe_at_10_years"]:
        warnings.append(
            f"Warning: 10-year Sharpe ({long_term.sharpe_ratio:.2f}) "
            f"below threshold ({thresholds['min_sharpe_at_10_years']})"
        )

    return {
        "passes": len(issues) == 0,
        "viability": metrics.long_term_viability.value,
        "viability_confidence": metrics.viability_confidence,
        "critical_issues": issues,
        "warnings": warnings,
        "stable_horizon_years": metrics.stable_horizon_years,
        "degradation_rate": metrics.degradation_rate_annual,
        "recommendations": metrics.recommendations,
    }
