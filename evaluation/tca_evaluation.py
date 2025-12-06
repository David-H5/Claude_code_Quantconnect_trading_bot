"""
Transaction Cost Analysis (TCA) Evaluation Module.

Evaluates execution quality by comparing actual fill prices against market benchmarks.
Critical for identifying hidden costs that erode strategy profitability.

Key Benchmarks:
- VWAP (Volume-Weighted Average Price): Compare execution vs market average
- PWP (Participation-Weighted Price): Adjusted for your own market impact
- Implementation Shortfall: Decision price - Execution price (total delay cost)
- Arrival Price: Slippage from order receipt to execution
- Market Impact: Price movement caused by your order

Professional Standards (2025):
- VWAP deviation: < 5 bps for liquid assets
- Implementation shortfall: < 10 bps for optimal execution
- Market impact: < 3 bps for small orders, < 15 bps for large orders

MiFID II Compliance:
- Best execution reporting required
- Pre-trade/post-trade analysis mandatory
- Audit trail for all executions

References:
- https://www.lseg.com/en/data-analytics/pre-trade-post-trade-analytics
- https://www.esma.europa.eu/trading/mifid-ii
- https://www.babelfish.ai/blog/transaction-cost-analysis-tca-evaluating-algo-performance
- docs/research/EVALUATION_UPGRADE_GUIDE.md

Version: 1.0 (December 2025)
"""

import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class ExecutionQuality(Enum):
    """Execution quality classification based on total cost."""

    EXCELLENT = "excellent"  # < 2 bps
    GOOD = "good"  # 2-5 bps
    FAIR = "fair"  # 5-10 bps
    POOR = "poor"  # > 10 bps


class OrderSize(Enum):
    """Order size classification for threshold selection."""

    SMALL = "small"  # < 1% of ADV
    MEDIUM = "medium"  # 1-5% of ADV
    LARGE = "large"  # > 5% of ADV


@dataclass
class ExecutionRecord:
    """
    Single execution record for TCA analysis.

    Captures all price points needed for comprehensive TCA:
    - Decision price: When the trading decision was made
    - Arrival price: When the order reached the exchange
    - Execution price: Actual fill price
    - VWAP: Market VWAP during execution window
    """

    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    decision_price: float  # Price when decision made
    arrival_price: float  # Price when order received
    execution_price: float  # Actual fill price
    vwap: float  # Market VWAP during execution
    decision_time: datetime
    arrival_time: datetime
    execution_time: datetime
    market_volume: float  # Total market volume during execution
    order_volume: float  # Our order volume
    asset_class: str = "equity"  # equity, option, futures
    exchange: str = ""
    order_type: str = "limit"  # market, limit, etc.


@dataclass
class TCAMetrics:
    """
    Comprehensive Transaction Cost Analysis metrics.

    All values in basis points (bps) for standardization.
    1 bp = 0.01% = 0.0001
    """

    # Primary benchmarks
    vwap_deviation_bps: float  # vs Volume-Weighted Average Price
    implementation_shortfall_bps: float  # Decision price - Execution price
    arrival_cost_bps: float  # Arrival price - Execution price

    # Market impact analysis
    market_impact_bps: float  # Price movement caused by our order
    realized_spread_bps: float  # Actual spread paid
    effective_spread_bps: float  # 2 * |execution - mid|

    # Timing analysis
    timing_cost_bps: float  # Cost of waiting to execute
    opportunity_cost_bps: float  # Cost of not executing immediately
    delay_cost_bps: float  # Cost from decision to arrival

    # Aggregate metrics
    total_cost_bps: float  # All-in execution cost
    execution_quality: ExecutionQuality

    # Per-execution statistics
    num_executions: int
    total_volume: float
    avg_execution_time_ms: float

    # Compliance
    mifid_compliant: bool  # Meets best execution requirements
    audit_trail_complete: bool  # Full documentation available
    best_execution_score: float  # 0-100 score

    # Recommendations
    recommendations: list[str]


@dataclass
class TCAReport:
    """Full TCA report with breakdowns."""

    overall_metrics: TCAMetrics
    by_symbol: dict[str, TCAMetrics]
    by_order_size: dict[str, TCAMetrics]
    by_time_of_day: dict[str, TCAMetrics]
    executions_analyzed: int
    report_period_start: datetime
    report_period_end: datetime
    generated_at: datetime


# TCA Thresholds (Professional Standards 2025)
TCA_THRESHOLDS = {
    "vwap_deviation_bps": {
        "excellent": 2.0,
        "good": 5.0,
        "acceptable": 10.0,
        "warning": 15.0,
        "critical": 25.0,
    },
    "implementation_shortfall_bps": {
        "excellent": 3.0,
        "good": 10.0,
        "acceptable": 20.0,
        "warning": 30.0,
        "critical": 50.0,
    },
    "market_impact_bps": {
        "small_order": {
            "excellent": 1.0,
            "good": 3.0,
            "acceptable": 5.0,
        },
        "medium_order": {
            "excellent": 3.0,
            "good": 8.0,
            "acceptable": 15.0,
        },
        "large_order": {
            "excellent": 5.0,
            "good": 15.0,
            "acceptable": 25.0,
        },
    },
    "total_cost_bps": {
        "excellent": 2.0,
        "good": 5.0,
        "fair": 10.0,
        "poor": 25.0,
    },
}


def calculate_vwap_deviation(
    execution_price: float,
    vwap: float,
    side: str,
) -> float:
    """
    Calculate deviation from VWAP in basis points.

    Positive value = worse than VWAP (paid more for buys, received less for sells)
    Negative value = better than VWAP (price improvement)

    Args:
        execution_price: Actual fill price
        vwap: Market VWAP during execution window
        side: "buy" or "sell"

    Returns:
        VWAP deviation in basis points
    """
    if vwap <= 0:
        return 0.0

    if side.lower() == "buy":
        # For buys: positive deviation means we paid more than VWAP
        return ((execution_price - vwap) / vwap) * 10000
    else:
        # For sells: positive deviation means we received less than VWAP
        return ((vwap - execution_price) / vwap) * 10000


def calculate_implementation_shortfall(
    decision_price: float,
    execution_price: float,
    side: str,
) -> float:
    """
    Calculate implementation shortfall in basis points.

    Measures total cost from decision to execution, including:
    - Delay cost (price moved while waiting)
    - Market impact
    - Execution slippage

    Args:
        decision_price: Price when trading decision was made
        execution_price: Actual fill price
        side: "buy" or "sell"

    Returns:
        Implementation shortfall in basis points
    """
    if decision_price <= 0:
        return 0.0

    if side.lower() == "buy":
        # For buys: positive means we paid more than decision price
        return ((execution_price - decision_price) / decision_price) * 10000
    else:
        # For sells: positive means we received less than decision price
        return ((decision_price - execution_price) / decision_price) * 10000


def calculate_arrival_cost(
    arrival_price: float,
    execution_price: float,
    side: str,
) -> float:
    """
    Calculate arrival cost (slippage) in basis points.

    Measures cost from order arrival to execution.

    Args:
        arrival_price: Price when order was received
        execution_price: Actual fill price
        side: "buy" or "sell"

    Returns:
        Arrival cost in basis points
    """
    if arrival_price <= 0:
        return 0.0

    if side.lower() == "buy":
        return ((execution_price - arrival_price) / arrival_price) * 10000
    else:
        return ((arrival_price - execution_price) / arrival_price) * 10000


def calculate_market_impact(
    order_volume: float,
    market_volume: float,
    price_change_pct: float,
) -> float:
    """
    Estimate market impact in basis points using square-root model.

    Square-root market impact model is standard in TCA:
    Impact = sigma * sqrt(V/ADV) * sgn(order)

    Args:
        order_volume: Our order volume
        market_volume: Total market volume during execution
        price_change_pct: Price change during execution (as decimal)

    Returns:
        Estimated market impact in basis points
    """
    if market_volume <= 0:
        return 0.0

    participation_rate = order_volume / market_volume
    # Square-root model for market impact
    impact = abs(price_change_pct) * math.sqrt(participation_rate) * 10000
    return impact


def calculate_delay_cost(
    decision_price: float,
    arrival_price: float,
    side: str,
) -> float:
    """
    Calculate delay cost (decision to arrival) in basis points.

    Captures cost of latency from decision to order submission.

    Args:
        decision_price: Price when decision made
        arrival_price: Price when order received
        side: "buy" or "sell"

    Returns:
        Delay cost in basis points
    """
    if decision_price <= 0:
        return 0.0

    if side.lower() == "buy":
        return ((arrival_price - decision_price) / decision_price) * 10000
    else:
        return ((decision_price - arrival_price) / decision_price) * 10000


def classify_order_size(
    order_volume: float,
    average_daily_volume: float,
) -> OrderSize:
    """
    Classify order size relative to average daily volume.

    Args:
        order_volume: Order quantity
        average_daily_volume: Asset's ADV

    Returns:
        OrderSize classification
    """
    if average_daily_volume <= 0:
        return OrderSize.MEDIUM

    participation = order_volume / average_daily_volume

    if participation < 0.01:  # < 1% of ADV
        return OrderSize.SMALL
    elif participation < 0.05:  # 1-5% of ADV
        return OrderSize.MEDIUM
    else:  # > 5% of ADV
        return OrderSize.LARGE


def classify_execution_quality(total_cost_bps: float) -> ExecutionQuality:
    """
    Classify execution quality based on total cost.

    Args:
        total_cost_bps: Total execution cost in basis points

    Returns:
        ExecutionQuality classification
    """
    thresholds = TCA_THRESHOLDS["total_cost_bps"]

    if abs(total_cost_bps) < thresholds["excellent"]:
        return ExecutionQuality.EXCELLENT
    elif abs(total_cost_bps) < thresholds["good"]:
        return ExecutionQuality.GOOD
    elif abs(total_cost_bps) < thresholds["fair"]:
        return ExecutionQuality.FAIR
    else:
        return ExecutionQuality.POOR


def calculate_tca_metrics(
    executions: list[ExecutionRecord],
) -> TCAMetrics:
    """
    Calculate comprehensive TCA metrics from execution records.

    Args:
        executions: List of execution records to analyze

    Returns:
        TCAMetrics with full TCA analysis
    """
    if not executions:
        return _empty_tca_metrics()

    # Volume-weighted calculations
    total_volume = sum(e.quantity for e in executions)
    if total_volume <= 0:
        return _empty_tca_metrics()

    # Calculate metrics for each execution
    vwap_devs = []
    impl_shortfalls = []
    arrival_costs = []
    market_impacts = []
    delay_costs = []
    execution_times = []

    for e in executions:
        weight = e.quantity / total_volume

        # VWAP deviation
        vwap_dev = calculate_vwap_deviation(e.execution_price, e.vwap, e.side)
        vwap_devs.append(vwap_dev * weight)

        # Implementation shortfall
        impl_sf = calculate_implementation_shortfall(e.decision_price, e.execution_price, e.side)
        impl_shortfalls.append(impl_sf * weight)

        # Arrival cost
        arr_cost = calculate_arrival_cost(e.arrival_price, e.execution_price, e.side)
        arrival_costs.append(arr_cost * weight)

        # Market impact
        price_change = abs(e.execution_price - e.arrival_price) / e.arrival_price
        impact = calculate_market_impact(e.order_volume, e.market_volume, price_change)
        market_impacts.append(impact * weight)

        # Delay cost
        delay = calculate_delay_cost(e.decision_price, e.arrival_price, e.side)
        delay_costs.append(delay * weight)

        # Execution time
        exec_time_ms = (e.execution_time - e.arrival_time).total_seconds() * 1000
        execution_times.append(exec_time_ms)

    # Aggregate weighted metrics
    vwap_deviation = sum(vwap_devs)
    impl_shortfall = sum(impl_shortfalls)
    arrival_cost = sum(arrival_costs)
    market_impact = sum(market_impacts)
    delay_cost = sum(delay_costs)

    # Derived metrics
    timing_cost = impl_shortfall - arrival_cost
    realized_spread = arrival_cost - market_impact
    effective_spread = arrival_cost * 2  # Approximation
    opportunity_cost = delay_cost  # Cost of waiting

    # Total cost is implementation shortfall
    total_cost = impl_shortfall

    # Quality classification
    quality = classify_execution_quality(total_cost)

    # Average execution time
    avg_exec_time = statistics.mean(execution_times) if execution_times else 0.0

    # MiFID II compliance check
    mifid_compliant = abs(total_cost) < TCA_THRESHOLDS["implementation_shortfall_bps"]["acceptable"]

    # Best execution score (0-100)
    # Lower cost = higher score
    max_cost = TCA_THRESHOLDS["implementation_shortfall_bps"]["critical"]
    best_exec_score = max(0, min(100, 100 * (1 - abs(total_cost) / max_cost)))

    # Generate recommendations
    recommendations = _generate_tca_recommendations(
        vwap_deviation=vwap_deviation,
        impl_shortfall=impl_shortfall,
        market_impact=market_impact,
        arrival_cost=arrival_cost,
        quality=quality,
    )

    return TCAMetrics(
        vwap_deviation_bps=vwap_deviation,
        implementation_shortfall_bps=impl_shortfall,
        arrival_cost_bps=arrival_cost,
        market_impact_bps=market_impact,
        realized_spread_bps=realized_spread,
        effective_spread_bps=effective_spread,
        timing_cost_bps=timing_cost,
        opportunity_cost_bps=opportunity_cost,
        delay_cost_bps=delay_cost,
        total_cost_bps=total_cost,
        execution_quality=quality,
        num_executions=len(executions),
        total_volume=total_volume,
        avg_execution_time_ms=avg_exec_time,
        mifid_compliant=mifid_compliant,
        audit_trail_complete=True,
        best_execution_score=best_exec_score,
        recommendations=recommendations,
    )


def _empty_tca_metrics() -> TCAMetrics:
    """Return empty TCA metrics."""
    return TCAMetrics(
        vwap_deviation_bps=0.0,
        implementation_shortfall_bps=0.0,
        arrival_cost_bps=0.0,
        market_impact_bps=0.0,
        realized_spread_bps=0.0,
        effective_spread_bps=0.0,
        timing_cost_bps=0.0,
        opportunity_cost_bps=0.0,
        delay_cost_bps=0.0,
        total_cost_bps=0.0,
        execution_quality=ExecutionQuality.EXCELLENT,
        num_executions=0,
        total_volume=0.0,
        avg_execution_time_ms=0.0,
        mifid_compliant=True,
        audit_trail_complete=True,
        best_execution_score=100.0,
        recommendations=["No executions to analyze"],
    )


def _generate_tca_recommendations(
    vwap_deviation: float,
    impl_shortfall: float,
    market_impact: float,
    arrival_cost: float,
    quality: ExecutionQuality,
) -> list[str]:
    """Generate TCA recommendations based on metrics."""
    recommendations = []

    thresholds = TCA_THRESHOLDS

    # VWAP analysis
    if abs(vwap_deviation) > thresholds["vwap_deviation_bps"]["warning"]:
        recommendations.append(
            f"HIGH: VWAP deviation ({vwap_deviation:.1f} bps) exceeds warning threshold. "
            "Consider using VWAP algorithms for better execution."
        )
    elif abs(vwap_deviation) > thresholds["vwap_deviation_bps"]["acceptable"]:
        recommendations.append(
            f"MEDIUM: VWAP deviation ({vwap_deviation:.1f} bps) above acceptable. " "Review execution timing."
        )

    # Implementation shortfall
    if abs(impl_shortfall) > thresholds["implementation_shortfall_bps"]["warning"]:
        recommendations.append(
            f"HIGH: Implementation shortfall ({impl_shortfall:.1f} bps) is significant. "
            "Reduce latency between decision and execution."
        )
    elif abs(impl_shortfall) > thresholds["implementation_shortfall_bps"]["acceptable"]:
        recommendations.append(
            f"MEDIUM: Implementation shortfall ({impl_shortfall:.1f} bps) above acceptable. " "Review order routing."
        )

    # Market impact
    if market_impact > thresholds["market_impact_bps"]["large_order"]["acceptable"]:
        recommendations.append(
            f"HIGH: Market impact ({market_impact:.1f} bps) is high. "
            "Consider splitting large orders or using dark pools."
        )
    elif market_impact > thresholds["market_impact_bps"]["medium_order"]["acceptable"]:
        recommendations.append(
            f"MEDIUM: Market impact ({market_impact:.1f} bps) above threshold. " "Consider reducing order size."
        )

    # Arrival cost
    if abs(arrival_cost) > 10:
        recommendations.append(
            f"MEDIUM: Arrival cost ({arrival_cost:.1f} bps) suggests slippage issues. "
            "Consider limit orders or better execution venues."
        )

    # Quality-based recommendations
    if quality == ExecutionQuality.POOR:
        recommendations.append(
            "CRITICAL: Overall execution quality is POOR. " "Comprehensive review of execution strategy required."
        )
    elif quality == ExecutionQuality.FAIR:
        recommendations.append("WARNING: Execution quality is FAIR. " "Improvements recommended for cost reduction.")

    if not recommendations:
        recommendations.append("Execution quality is good. Continue monitoring for consistency.")

    return recommendations


def calculate_tca_by_symbol(
    executions: list[ExecutionRecord],
) -> dict[str, TCAMetrics]:
    """
    Calculate TCA metrics grouped by symbol.

    Args:
        executions: List of execution records

    Returns:
        Dict mapping symbol to TCA metrics
    """
    by_symbol: dict[str, list[ExecutionRecord]] = {}

    for e in executions:
        if e.symbol not in by_symbol:
            by_symbol[e.symbol] = []
        by_symbol[e.symbol].append(e)

    return {symbol: calculate_tca_metrics(execs) for symbol, execs in by_symbol.items()}


def calculate_tca_by_time_of_day(
    executions: list[ExecutionRecord],
) -> dict[str, TCAMetrics]:
    """
    Calculate TCA metrics grouped by time of day.

    Groups: morning (9:30-11:00), midday (11:00-14:00),
    afternoon (14:00-15:30), close (15:30-16:00)

    Args:
        executions: List of execution records

    Returns:
        Dict mapping time period to TCA metrics
    """
    by_time: dict[str, list[ExecutionRecord]] = {
        "morning": [],
        "midday": [],
        "afternoon": [],
        "close": [],
    }

    for e in executions:
        hour = e.execution_time.hour
        minute = e.execution_time.minute
        time_decimal = hour + minute / 60

        if time_decimal < 11.0:
            by_time["morning"].append(e)
        elif time_decimal < 14.0:
            by_time["midday"].append(e)
        elif time_decimal < 15.5:
            by_time["afternoon"].append(e)
        else:
            by_time["close"].append(e)

    return {
        period: calculate_tca_metrics(execs)
        for period, execs in by_time.items()
        if execs  # Only include periods with executions
    }


def generate_tca_report(
    executions: list[ExecutionRecord],
    period_start: datetime | None = None,
    period_end: datetime | None = None,
) -> TCAReport:
    """
    Generate comprehensive TCA report.

    Args:
        executions: List of execution records
        period_start: Report period start (defaults to earliest execution)
        period_end: Report period end (defaults to latest execution)

    Returns:
        TCAReport with full analysis
    """
    if not executions:
        now = datetime.now()
        return TCAReport(
            overall_metrics=_empty_tca_metrics(),
            by_symbol={},
            by_order_size={},
            by_time_of_day={},
            executions_analyzed=0,
            report_period_start=period_start or now,
            report_period_end=period_end or now,
            generated_at=now,
        )

    # Determine period
    if period_start is None:
        period_start = min(e.execution_time for e in executions)
    if period_end is None:
        period_end = max(e.execution_time for e in executions)

    # Calculate metrics
    overall = calculate_tca_metrics(executions)
    by_symbol = calculate_tca_by_symbol(executions)
    by_time = calculate_tca_by_time_of_day(executions)

    # Group by order size (simplified - would need ADV data for real classification)
    by_size: dict[str, list[ExecutionRecord]] = {
        "small": [],
        "medium": [],
        "large": [],
    }
    for e in executions:
        if e.quantity < 100:
            by_size["small"].append(e)
        elif e.quantity < 1000:
            by_size["medium"].append(e)
        else:
            by_size["large"].append(e)

    by_order_size = {size: calculate_tca_metrics(execs) for size, execs in by_size.items() if execs}

    return TCAReport(
        overall_metrics=overall,
        by_symbol=by_symbol,
        by_order_size=by_order_size,
        by_time_of_day=by_time,
        executions_analyzed=len(executions),
        report_period_start=period_start,
        report_period_end=period_end,
        generated_at=datetime.now(),
    )


def format_tca_report(report: TCAReport) -> str:
    """
    Format TCA report as markdown.

    Args:
        report: TCAReport to format

    Returns:
        Formatted markdown report
    """
    lines = []
    lines.append("# Transaction Cost Analysis (TCA) Report\n")
    lines.append(f"**Report Generated**: {report.generated_at.strftime('%Y-%m-%d %H:%M')}")
    lines.append(
        f"**Period**: {report.report_period_start.strftime('%Y-%m-%d')} to {report.report_period_end.strftime('%Y-%m-%d')}"
    )
    lines.append(f"**Executions Analyzed**: {report.executions_analyzed}\n")

    # Overall metrics
    m = report.overall_metrics
    lines.append("## Overall Execution Quality\n")
    lines.append(f"**Quality Rating**: {m.execution_quality.value.upper()}")
    lines.append(f"**Best Execution Score**: {m.best_execution_score:.1f}/100")
    lines.append(f"**MiFID II Compliant**: {'YES' if m.mifid_compliant else 'NO'}\n")

    # Cost breakdown
    lines.append("## Cost Breakdown (basis points)\n")
    lines.append("| Metric | Value | Threshold | Status |")
    lines.append("|--------|-------|-----------|--------|")

    def status_emoji(value: float, threshold: float) -> str:
        return "OK" if abs(value) < threshold else "ALERT"

    lines.append(
        f"| VWAP Deviation | {m.vwap_deviation_bps:+.2f} | < 5.0 | {status_emoji(m.vwap_deviation_bps, 5.0)} |"
    )
    lines.append(
        f"| Implementation Shortfall | {m.implementation_shortfall_bps:+.2f} | < 10.0 | {status_emoji(m.implementation_shortfall_bps, 10.0)} |"
    )
    lines.append(f"| Arrival Cost | {m.arrival_cost_bps:+.2f} | < 5.0 | {status_emoji(m.arrival_cost_bps, 5.0)} |")
    lines.append(f"| Market Impact | {m.market_impact_bps:+.2f} | < 3.0 | {status_emoji(m.market_impact_bps, 3.0)} |")
    lines.append(f"| Delay Cost | {m.delay_cost_bps:+.2f} | < 5.0 | {status_emoji(m.delay_cost_bps, 5.0)} |")
    lines.append(
        f"| **Total Cost** | **{m.total_cost_bps:+.2f}** | **< 10.0** | **{status_emoji(m.total_cost_bps, 10.0)}** |"
    )
    lines.append("")

    # Execution statistics
    lines.append("## Execution Statistics\n")
    lines.append(f"- Total Volume: {m.total_volume:,.0f}")
    lines.append(f"- Avg Execution Time: {m.avg_execution_time_ms:.1f} ms")
    lines.append(f"- Effective Spread: {m.effective_spread_bps:.2f} bps")
    lines.append(f"- Realized Spread: {m.realized_spread_bps:.2f} bps\n")

    # By symbol breakdown
    if report.by_symbol:
        lines.append("## Analysis by Symbol\n")
        lines.append("| Symbol | Executions | VWAP Dev | Impl Shortfall | Quality |")
        lines.append("|--------|------------|----------|----------------|---------|")
        for symbol, metrics in sorted(report.by_symbol.items()):
            lines.append(
                f"| {symbol} | {metrics.num_executions} | "
                f"{metrics.vwap_deviation_bps:+.2f} | "
                f"{metrics.implementation_shortfall_bps:+.2f} | "
                f"{metrics.execution_quality.value} |"
            )
        lines.append("")

    # By time of day
    if report.by_time_of_day:
        lines.append("## Analysis by Time of Day\n")
        lines.append("| Period | Executions | VWAP Dev | Impl Shortfall | Quality |")
        lines.append("|--------|------------|----------|----------------|---------|")
        for period, metrics in report.by_time_of_day.items():
            lines.append(
                f"| {period.capitalize()} | {metrics.num_executions} | "
                f"{metrics.vwap_deviation_bps:+.2f} | "
                f"{metrics.implementation_shortfall_bps:+.2f} | "
                f"{metrics.execution_quality.value} |"
            )
        lines.append("")

    # Recommendations
    lines.append("## Recommendations\n")
    for rec in m.recommendations:
        lines.append(f"- {rec}")

    # Thresholds reference
    lines.append("\n## Threshold Reference\n")
    lines.append("| Metric | Excellent | Good | Acceptable | Warning |")
    lines.append("|--------|-----------|------|------------|---------|")
    lines.append("| VWAP Deviation | < 2 bps | < 5 bps | < 10 bps | > 15 bps |")
    lines.append("| Impl Shortfall | < 3 bps | < 10 bps | < 20 bps | > 30 bps |")
    lines.append("| Market Impact (small) | < 1 bps | < 3 bps | < 5 bps | - |")
    lines.append("| Market Impact (large) | < 5 bps | < 15 bps | < 25 bps | - |")
    lines.append("| Total Cost | < 2 bps | < 5 bps | < 10 bps | > 25 bps |")

    return "\n".join(lines)


def check_tca_compliance(metrics: TCAMetrics) -> dict[str, Any]:
    """
    Check TCA metrics against compliance thresholds.

    Args:
        metrics: TCA metrics to check

    Returns:
        Dict with compliance status and details
    """
    issues = []
    warnings = []

    thresholds = TCA_THRESHOLDS

    # Check VWAP deviation
    if abs(metrics.vwap_deviation_bps) > thresholds["vwap_deviation_bps"]["critical"]:
        issues.append(f"VWAP deviation ({metrics.vwap_deviation_bps:.1f} bps) exceeds critical threshold")
    elif abs(metrics.vwap_deviation_bps) > thresholds["vwap_deviation_bps"]["warning"]:
        warnings.append(f"VWAP deviation ({metrics.vwap_deviation_bps:.1f} bps) exceeds warning threshold")

    # Check implementation shortfall
    if abs(metrics.implementation_shortfall_bps) > thresholds["implementation_shortfall_bps"]["critical"]:
        issues.append(
            f"Implementation shortfall ({metrics.implementation_shortfall_bps:.1f} bps) exceeds critical threshold"
        )
    elif abs(metrics.implementation_shortfall_bps) > thresholds["implementation_shortfall_bps"]["warning"]:
        warnings.append(
            f"Implementation shortfall ({metrics.implementation_shortfall_bps:.1f} bps) exceeds warning threshold"
        )

    # MiFID II compliance
    mifid_compliant = len(issues) == 0 and metrics.audit_trail_complete

    return {
        "compliant": len(issues) == 0,
        "mifid_compliant": mifid_compliant,
        "critical_issues": issues,
        "warnings": warnings,
        "best_execution_score": metrics.best_execution_score,
        "execution_quality": metrics.execution_quality.value,
        "recommendations": metrics.recommendations,
    }
