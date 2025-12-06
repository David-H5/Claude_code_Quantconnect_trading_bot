"""
Realistic Thresholds Calibration Module.

Calibrates evaluation thresholds based on 2025 benchmark findings that set
realistic expectations for LLM-based trading agents.

Key Research Findings (December 2025):

1. Finance Agent Benchmark:
   - Even OpenAI o3 achieved only 46.8% accuracy at $3.79/query
   - Higher cost does NOT correlate with better performance
   - Most LLMs struggle with complex financial decisions

2. InvestorBench (ACL 2025):
   - Best LLM achieved only 58.7% success rate on complex decisions
   - Significant variance across asset classes

3. StockBench:
   - Most LLMs fail to beat buy-and-hold during downturns
   - Model rankings shift between market periods

4. FINSABER:
   - LLM advantages deteriorate over longer time periods
   - 2-year performance doesn't predict 20-year performance

Implications:
- Don't expect > 60% accuracy on complex decisions
- Focus on risk management over prediction accuracy
- Cost-adjusted performance is critical
- Ensemble approaches help but won't solve fundamental limits

References:
- https://paperswithcode.com/sota/financial-agent-benchmark
- https://arxiv.org/abs/2501.00174 (InvestorBench)
- https://arxiv.org/abs/2510.02209 (StockBench)
- docs/research/EVALUATION_UPGRADE_GUIDE.md

Version: 1.0 (December 2025)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExpectationLevel(Enum):
    """Expectation level for LLM performance."""

    REALISTIC = "realistic"  # Based on 2025 benchmarks
    OPTIMISTIC = "optimistic"  # Higher expectations
    CONSERVATIVE = "conservative"  # Lower expectations (safer)


class DecisionComplexity(Enum):
    """Complexity level of financial decisions."""

    SIMPLE = "simple"  # Single-factor decisions
    MODERATE = "moderate"  # Multi-factor decisions
    COMPLEX = "complex"  # Full portfolio decisions
    ADVERSARIAL = "adversarial"  # Edge cases, adversarial scenarios


@dataclass
class AccuracyThresholds:
    """Accuracy thresholds by decision complexity."""

    excellent: float
    good: float
    acceptable: float
    poor: float


@dataclass
class CostThresholds:
    """Cost efficiency thresholds ($ per correct decision)."""

    excellent: float
    good: float
    acceptable: float
    poor: float


@dataclass
class RealisticExpectations:
    """
    Realistic expectations for LLM trading performance.

    Based on December 2025 benchmark findings.
    """

    # Decision accuracy by complexity
    accuracy_by_complexity: dict[DecisionComplexity, AccuracyThresholds]

    # Cost efficiency
    cost_per_correct_decision: CostThresholds

    # Risk-adjusted returns
    sharpe_target: float
    max_drawdown_limit: float
    win_rate_target: float
    profit_factor_target: float

    # Benchmark references
    o3_accuracy: float  # 46.8% - state of the art
    o3_cost_per_query: float  # $3.79

    # Notes
    notes: list[str]


# Updated 2025 Realistic Thresholds
# Based on Finance Agent Benchmark, InvestorBench, StockBench findings

REALISTIC_THRESHOLDS_2025 = RealisticExpectations(
    accuracy_by_complexity={
        DecisionComplexity.SIMPLE: AccuracyThresholds(
            excellent=0.75,  # > 75% is exceptional for simple decisions
            good=0.65,
            acceptable=0.55,
            poor=0.45,
        ),
        DecisionComplexity.MODERATE: AccuracyThresholds(
            excellent=0.65,  # > 65% is exceptional
            good=0.55,
            acceptable=0.48,  # o3 level
            poor=0.40,
        ),
        DecisionComplexity.COMPLEX: AccuracyThresholds(
            excellent=0.55,  # > 55% is exceptional for LLMs
            good=0.48,  # o3 level (46.8%)
            acceptable=0.40,
            poor=0.35,  # Below random with bias
        ),
        DecisionComplexity.ADVERSARIAL: AccuracyThresholds(
            excellent=0.45,  # Adversarial is hard
            good=0.38,
            acceptable=0.30,
            poor=0.25,
        ),
    },
    cost_per_correct_decision=CostThresholds(
        excellent=2.00,  # < $2 per correct decision
        good=4.00,  # $2-4
        acceptable=8.00,  # $4-8 (o3 at $3.79/query ÷ 0.468 = $8.10)
        poor=15.00,  # > $8
    ),
    sharpe_target=1.5,  # Lower than "leading" bots (2.5-3.2)
    max_drawdown_limit=0.15,  # 15% max (conservative)
    win_rate_target=0.50,  # 50% is acceptable with good R:R
    profit_factor_target=1.3,  # 1.3 is realistic for LLM strategies
    o3_accuracy=0.468,  # Reference: best LLM as of Dec 2025
    o3_cost_per_query=3.79,  # Reference: o3 cost
    notes=[
        "Even o3 achieves only 46.8% accuracy on complex financial decisions",
        "Cost per correct decision is critical - cheap wrong answers are worthless",
        "Focus on risk management over prediction accuracy",
        "50% accuracy with 2:1 R:R is profitable; 60% with 1:1 breaks even",
        "Ensemble approaches improve consistency but not accuracy ceiling",
    ],
)

# Optimistic thresholds (pre-2025 expectations)
OPTIMISTIC_THRESHOLDS = RealisticExpectations(
    accuracy_by_complexity={
        DecisionComplexity.SIMPLE: AccuracyThresholds(
            excellent=0.85,
            good=0.75,
            acceptable=0.65,
            poor=0.55,
        ),
        DecisionComplexity.MODERATE: AccuracyThresholds(
            excellent=0.75,
            good=0.65,
            acceptable=0.55,
            poor=0.45,
        ),
        DecisionComplexity.COMPLEX: AccuracyThresholds(
            excellent=0.70,
            good=0.60,
            acceptable=0.50,
            poor=0.40,
        ),
        DecisionComplexity.ADVERSARIAL: AccuracyThresholds(
            excellent=0.60,
            good=0.50,
            acceptable=0.40,
            poor=0.30,
        ),
    },
    cost_per_correct_decision=CostThresholds(
        excellent=1.00,
        good=2.00,
        acceptable=5.00,
        poor=10.00,
    ),
    sharpe_target=2.5,
    max_drawdown_limit=0.10,
    win_rate_target=0.60,
    profit_factor_target=1.75,
    o3_accuracy=0.468,
    o3_cost_per_query=3.79,
    notes=[
        "WARNING: These thresholds may be unrealistic based on 2025 benchmarks",
        "High accuracy expectations may indicate test contamination",
        "Consider using REALISTIC_THRESHOLDS_2025 instead",
    ],
)

# Conservative thresholds (for production safety)
CONSERVATIVE_THRESHOLDS = RealisticExpectations(
    accuracy_by_complexity={
        DecisionComplexity.SIMPLE: AccuracyThresholds(
            excellent=0.70,
            good=0.60,
            acceptable=0.50,
            poor=0.40,
        ),
        DecisionComplexity.MODERATE: AccuracyThresholds(
            excellent=0.60,
            good=0.50,
            acceptable=0.42,
            poor=0.35,
        ),
        DecisionComplexity.COMPLEX: AccuracyThresholds(
            excellent=0.50,
            good=0.42,
            acceptable=0.35,
            poor=0.30,
        ),
        DecisionComplexity.ADVERSARIAL: AccuracyThresholds(
            excellent=0.40,
            good=0.33,
            acceptable=0.28,
            poor=0.22,
        ),
    },
    cost_per_correct_decision=CostThresholds(
        excellent=1.50,
        good=3.00,
        acceptable=6.00,
        poor=12.00,
    ),
    sharpe_target=1.0,
    max_drawdown_limit=0.20,
    win_rate_target=0.45,
    profit_factor_target=1.2,
    o3_accuracy=0.468,
    o3_cost_per_query=3.79,
    notes=[
        "Conservative thresholds for production safety",
        "Lower accuracy expectations = less overfitting risk",
        "Focus on consistent returns over peak performance",
    ],
)


def get_thresholds(level: ExpectationLevel = ExpectationLevel.REALISTIC) -> RealisticExpectations:
    """
    Get thresholds for the specified expectation level.

    Args:
        level: ExpectationLevel to use

    Returns:
        RealisticExpectations for the level
    """
    if level == ExpectationLevel.REALISTIC:
        return REALISTIC_THRESHOLDS_2025
    elif level == ExpectationLevel.OPTIMISTIC:
        return OPTIMISTIC_THRESHOLDS
    else:
        return CONSERVATIVE_THRESHOLDS


def get_accuracy_threshold(
    complexity: DecisionComplexity,
    level: ExpectationLevel = ExpectationLevel.REALISTIC,
) -> AccuracyThresholds:
    """
    Get accuracy thresholds for a specific decision complexity.

    Args:
        complexity: DecisionComplexity level
        level: ExpectationLevel to use

    Returns:
        AccuracyThresholds for the complexity level
    """
    thresholds = get_thresholds(level)
    return thresholds.accuracy_by_complexity[complexity]


def calibrate_expectations(
    observed_accuracy: float,
    decision_complexity: DecisionComplexity = DecisionComplexity.COMPLEX,
    cost_per_query: float | None = None,
) -> dict[str, Any]:
    """
    Calibrate expectations based on observed performance.

    Compares observed metrics against 2025 benchmarks and provides assessment.

    Args:
        observed_accuracy: Observed accuracy rate (0-1)
        decision_complexity: Complexity of decisions being made
        cost_per_query: Optional cost per query in dollars

    Returns:
        Dict with assessment and recommendations
    """
    thresholds = get_accuracy_threshold(decision_complexity)

    # Percentile vs o3 benchmark
    o3_accuracy = REALISTIC_THRESHOLDS_2025.o3_accuracy
    percentile_vs_o3 = (
        "above" if observed_accuracy > o3_accuracy else "at" if abs(observed_accuracy - o3_accuracy) < 0.02 else "below"
    )

    # Quality assessment
    if observed_accuracy >= thresholds.excellent:
        quality = "excellent"
    elif observed_accuracy >= thresholds.good:
        quality = "good"
    elif observed_accuracy >= thresholds.acceptable:
        quality = "acceptable"
    else:
        quality = "poor"

    # Overfitting check
    is_suspicious = observed_accuracy > 0.60  # > 60% accuracy is suspicious

    # Cost efficiency (if provided)
    cost_per_correct = None
    cost_quality = None
    if cost_per_query is not None and observed_accuracy > 0:
        cost_per_correct = cost_per_query / observed_accuracy
        cost_thresholds = REALISTIC_THRESHOLDS_2025.cost_per_correct_decision
        if cost_per_correct <= cost_thresholds.excellent:
            cost_quality = "excellent"
        elif cost_per_correct <= cost_thresholds.good:
            cost_quality = "good"
        elif cost_per_correct <= cost_thresholds.acceptable:
            cost_quality = "acceptable"
        else:
            cost_quality = "poor"

    # Generate recommendations
    recommendations = []

    if is_suspicious:
        recommendations.append(
            f"WARNING: Accuracy ({observed_accuracy:.1%}) exceeds 60%, which is unusual. "
            f"Finance Agent Benchmark shows even o3 achieves only {o3_accuracy:.1%}. "
            "Verify test cases are truly out-of-sample and not contaminated."
        )

    if percentile_vs_o3 == "above":
        recommendations.append(
            f"Accuracy ({observed_accuracy:.1%}) exceeds o3 benchmark ({o3_accuracy:.1%}). "
            "This is exceptional but verify with additional validation."
        )

    if quality == "poor":
        recommendations.append(
            f"Accuracy ({observed_accuracy:.1%}) below acceptable threshold "
            f"({thresholds.acceptable:.1%}). Consider ensemble approach or simpler decisions."
        )

    if cost_quality == "poor" and cost_per_correct:
        recommendations.append(
            f"Cost efficiency poor: ${cost_per_correct:.2f} per correct decision. "
            "Consider cheaper models or reducing query complexity."
        )

    # Focus on risk management
    if observed_accuracy < 0.55:
        recommendations.append(
            "With sub-55% accuracy, focus on risk management: "
            "Strict position sizing, stop losses, and favorable risk:reward ratios (2:1+)."
        )

    if not recommendations:
        recommendations.append(
            f"Performance at {quality} level for {decision_complexity.value} decisions. "
            "Continue monitoring and maintain realistic expectations."
        )

    return {
        "observed_accuracy": observed_accuracy,
        "decision_complexity": decision_complexity.value,
        "quality_assessment": quality,
        "percentile_vs_o3": percentile_vs_o3,
        "is_suspicious": is_suspicious,
        "cost_per_correct_decision": cost_per_correct,
        "cost_quality": cost_quality,
        "thresholds": {
            "excellent": thresholds.excellent,
            "good": thresholds.good,
            "acceptable": thresholds.acceptable,
            "poor": thresholds.poor,
        },
        "o3_benchmark": o3_accuracy,
        "recommendations": recommendations,
    }


def calculate_profitability_requirements(
    accuracy: float,
    avg_win: float = 1.0,
    avg_loss: float = 1.0,
) -> dict[str, Any]:
    """
    Calculate profitability requirements given accuracy.

    Shows what risk:reward ratio is needed to be profitable at a given accuracy.

    Args:
        accuracy: Win rate (0-1)
        avg_win: Average win amount
        avg_loss: Average loss amount

    Returns:
        Dict with profitability analysis
    """
    if accuracy <= 0 or accuracy >= 1:
        return {"error": "Accuracy must be between 0 and 1"}

    # Expected value = (accuracy * avg_win) - ((1 - accuracy) * avg_loss)
    # Breakeven: accuracy * avg_win = (1 - accuracy) * avg_loss
    # Required R:R for breakeven: avg_win / avg_loss = (1 - accuracy) / accuracy

    breakeven_rr = (1 - accuracy) / accuracy
    current_ev = (accuracy * avg_win) - ((1 - accuracy) * avg_loss)
    current_rr = avg_win / avg_loss if avg_loss > 0 else 0

    # Profit factor
    gross_profit = accuracy * avg_win
    gross_loss = (1 - accuracy) * avg_loss
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    is_profitable = current_ev > 0

    # Calculate required accuracy for different R:R ratios
    rr_scenarios = {}
    for rr in [0.5, 1.0, 1.5, 2.0, 3.0]:
        # Breakeven accuracy = loss / (win + loss) = 1 / (rr + 1)
        required_acc = 1 / (rr + 1)
        rr_scenarios[f"rr_{rr}"] = {
            "risk_reward": rr,
            "breakeven_accuracy": required_acc,
            "profitable_at_current_accuracy": accuracy > required_acc,
        }

    return {
        "accuracy": accuracy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "current_risk_reward": current_rr,
        "breakeven_rr_required": breakeven_rr,
        "expected_value_per_trade": current_ev,
        "profit_factor": profit_factor,
        "is_profitable": is_profitable,
        "rr_scenarios": rr_scenarios,
        "insight": (
            f"At {accuracy:.1%} accuracy, you need {breakeven_rr:.2f}:1 R:R to break even. "
            f"Current R:R is {current_rr:.2f}:1, making strategy "
            f"{'profitable' if is_profitable else 'unprofitable'}."
        ),
    }


def generate_realistic_expectations_report() -> str:
    """
    Generate report on realistic expectations for LLM trading.

    Returns:
        Formatted markdown report
    """
    lines = []
    lines.append("# Realistic Expectations for LLM Trading Agents\n")
    lines.append("*Based on December 2025 Benchmark Research*\n")

    # Key findings
    lines.append("## Key Research Findings\n")
    lines.append("### Finance Agent Benchmark (November 2025)\n")
    lines.append("- **OpenAI o3 Accuracy**: 46.8% on complex financial decisions")
    lines.append("- **Cost**: $3.79 per query")
    lines.append("- **Cost per Correct Decision**: $8.10 ($3.79 / 0.468)")
    lines.append("- **Key Insight**: Higher cost does NOT correlate with better performance\n")

    lines.append("### InvestorBench (ACL 2025)\n")
    lines.append("- **Best LLM Success Rate**: 58.7% on complex decisions")
    lines.append("- **Tests 13 Different LLMs**: GPT-4, Claude-3.5, Gemini, etc.")
    lines.append("- **Key Insight**: Significant variance across asset classes\n")

    lines.append("### StockBench (October 2025)\n")
    lines.append("- **Key Finding**: Most LLMs fail to beat buy-and-hold during downturns")
    lines.append("- **Key Insight**: Model rankings shift between market periods\n")

    # Threshold tables
    lines.append("## Accuracy Thresholds by Decision Complexity\n")
    lines.append("### Realistic (Recommended)\n")
    lines.append("| Complexity | Excellent | Good | Acceptable | Poor |")
    lines.append("|------------|-----------|------|------------|------|")

    thresholds = REALISTIC_THRESHOLDS_2025
    for complexity, acc in thresholds.accuracy_by_complexity.items():
        lines.append(
            f"| {complexity.value.capitalize()} | "
            f">{acc.excellent:.0%} | >{acc.good:.0%} | "
            f">{acc.acceptable:.0%} | <{acc.poor:.0%} |"
        )
    lines.append("")

    # Cost efficiency
    lines.append("## Cost Efficiency Thresholds\n")
    lines.append("| Quality | Cost per Correct Decision |")
    lines.append("|---------|---------------------------|")
    cost = thresholds.cost_per_correct_decision
    lines.append(f"| Excellent | < ${cost.excellent:.2f} |")
    lines.append(f"| Good | < ${cost.good:.2f} |")
    lines.append(f"| Acceptable | < ${cost.acceptable:.2f} |")
    lines.append(f"| Poor | > ${cost.poor:.2f} |")
    lines.append("")

    # Risk-adjusted targets
    lines.append("## Risk-Adjusted Performance Targets\n")
    lines.append("| Metric | Realistic Target | Note |")
    lines.append("|--------|------------------|------|")
    lines.append(f"| Sharpe Ratio | {thresholds.sharpe_target} | Lower than 'leading' bots (2.5-3.2) |")
    lines.append(f"| Max Drawdown | {thresholds.max_drawdown_limit:.0%} | Conservative limit |")
    lines.append(f"| Win Rate | {thresholds.win_rate_target:.0%} | Acceptable with good R:R |")
    lines.append(f"| Profit Factor | {thresholds.profit_factor_target} | Realistic for LLM strategies |")
    lines.append("")

    # Profitability math
    lines.append("## Profitability Requirements by Accuracy\n")
    lines.append("| Accuracy | Breakeven R:R | Comment |")
    lines.append("|----------|---------------|---------|")
    for acc in [0.40, 0.45, 0.50, 0.55, 0.60]:
        rr = (1 - acc) / acc
        comment = "Needs favorable R:R" if acc < 0.50 else "More flexibility"
        lines.append(f"| {acc:.0%} | {rr:.2f}:1 | {comment} |")
    lines.append("")

    # Implications
    lines.append("## Key Implications\n")
    for note in thresholds.notes:
        lines.append(f"- {note}")
    lines.append("")

    # Warning signs
    lines.append("## Warning Signs of Unrealistic Results\n")
    lines.append("- Accuracy > 60% on complex decisions (may indicate overfitting)")
    lines.append("- Sharpe > 3.0 consistently (verify with out-of-sample)")
    lines.append("- Win rate > 65% without clear edge (check for look-ahead bias)")
    lines.append("- Profit factor > 2.5 (validate methodology)")
    lines.append("- Results significantly better than o3 benchmark")

    return "\n".join(lines)


def validate_against_benchmarks(
    metrics: dict[str, float],
) -> dict[str, Any]:
    """
    Validate strategy metrics against 2025 benchmarks.

    Args:
        metrics: Dict with accuracy, sharpe, win_rate, profit_factor, etc.

    Returns:
        Validation results with warnings
    """
    warnings = []
    red_flags = []

    accuracy = metrics.get("accuracy", 0.5)
    sharpe = metrics.get("sharpe_ratio", 1.0)
    win_rate = metrics.get("win_rate", 0.5)
    profit_factor = metrics.get("profit_factor", 1.5)

    # Check against o3 benchmark
    if accuracy > 0.60:
        red_flags.append(
            f"Accuracy ({accuracy:.1%}) exceeds 60% - unusual for complex decisions. " f"o3 achieves only 46.8%."
        )

    if accuracy > 0.55 and accuracy < 0.60:
        warnings.append(f"Accuracy ({accuracy:.1%}) exceeds o3 benchmark (46.8%). Verify methodology.")

    # Check Sharpe
    if sharpe > 3.5:
        red_flags.append(
            f"Sharpe ({sharpe:.2f}) is suspiciously high. " "May indicate overfitting. Validate with longer horizon."
        )
    elif sharpe > 2.5:
        warnings.append(f"Sharpe ({sharpe:.2f}) in top tier (leading bots: 2.5-3.2). Verify robustness.")

    # Check profit factor
    if profit_factor > 4.0:
        red_flags.append(f"Profit factor ({profit_factor:.2f}) > 4.0 is a classic overfitting signal.")
    elif profit_factor > 2.5:
        warnings.append(f"Profit factor ({profit_factor:.2f}) is high. Verify with walk-forward.")

    # Check win rate
    if win_rate > 0.85:
        red_flags.append(
            f"Win rate ({win_rate:.1%}) > 85% is suspicious. " "Very few strategies maintain this long-term."
        )

    # Overall assessment
    if red_flags:
        status = "SUSPICIOUS"
        recommendation = "Investigate potential overfitting or test contamination"
    elif warnings:
        status = "CAUTION"
        recommendation = "Results need additional validation"
    else:
        status = "OK"
        recommendation = "Results within realistic expectations"

    return {
        "status": status,
        "red_flags": red_flags,
        "warnings": warnings,
        "recommendation": recommendation,
        "benchmarks_used": {
            "o3_accuracy": 0.468,
            "leading_bot_sharpe": "2.5-3.2",
            "profit_factor_warning": 4.0,
            "win_rate_warning": 0.85,
        },
    }
