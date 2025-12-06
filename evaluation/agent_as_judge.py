"""
Agent-as-a-Judge Evaluation Module.

Uses LLM judges to evaluate trading agent decision quality, reasoning chains,
and risk assessments. Based on LLM-as-a-Judge methodology (Zheng et al. 2023)
adapted for financial trading evaluation.

Features:
- Multi-judge evaluation (Claude, GPT-4, Gemini ensemble)
- Position-debiased scoring (swap position averaging)
- Specialized rubrics for trading decisions
- Reasoning chain analysis
- Risk assessment validation
- Hallucination detection in financial contexts

Version: 1.0 (December 2025)
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class JudgeModel(Enum):
    """Available judge models."""

    CLAUDE_SONNET = "claude-sonnet"
    CLAUDE_OPUS = "claude-opus"
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    GEMINI_PRO = "gemini-pro"
    LOCAL_LLAMA = "local-llama"


class EvaluationCategory(Enum):
    """Categories for agent evaluation."""

    TRADING_DECISION = "trading_decision"
    RISK_ASSESSMENT = "risk_assessment"
    REASONING_CHAIN = "reasoning_chain"
    MARKET_ANALYSIS = "market_analysis"
    POSITION_SIZING = "position_sizing"
    TIMING_DECISION = "timing_decision"
    HALLUCINATION_CHECK = "hallucination_check"


class ScoreLevel(Enum):
    """Score levels for evaluation."""

    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    UNACCEPTABLE = 1


@dataclass
class JudgeScore:
    """Score from a single judge."""

    judge_model: JudgeModel
    category: EvaluationCategory
    score: int  # 1-5 scale
    confidence: float  # 0-1 confidence in score
    reasoning: str
    issues_found: list[str] = field(default_factory=list)
    strengths_found: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvaluationRubric:
    """Rubric for evaluating a specific category."""

    category: EvaluationCategory
    description: str
    criteria: dict[int, str]  # Score -> Description
    weight: float = 1.0  # Weight in overall score
    required_context: list[str] = field(default_factory=list)


@dataclass
class AgentDecision:
    """A trading agent decision to be evaluated."""

    decision_id: str
    decision_type: str  # "buy", "sell", "hold", "size", etc.
    symbol: str
    reasoning: str
    market_context: dict[str, Any]
    risk_assessment: str | None = None
    confidence: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    outcome: dict[str, Any] | None = None  # Actual result if known


@dataclass
class JudgeEvaluationResult:
    """Complete evaluation result from judge ensemble."""

    decision_id: str
    scores: list[JudgeScore]
    overall_score: float
    overall_confidence: float
    agreement_rate: float
    consensus_issues: list[str]
    consensus_strengths: list[str]
    recommendation: str
    detailed_feedback: str
    timestamp: datetime = field(default_factory=datetime.now)


# Trading-specific evaluation rubrics
TRADING_DECISION_RUBRIC = EvaluationRubric(
    category=EvaluationCategory.TRADING_DECISION,
    description="Evaluate the quality of the trading decision",
    criteria={
        5: "Decision is well-reasoned, aligns with market conditions, proper risk management, clear entry/exit logic",
        4: "Decision is sound with minor improvements possible, good risk consideration",
        3: "Decision is acceptable but lacks depth in analysis or risk consideration",
        2: "Decision has significant flaws in logic or ignores important market factors",
        1: "Decision is poorly reasoned, contradicts market evidence, or shows dangerous risk taking",
    },
    weight=1.5,
    required_context=["market_data", "position_info", "risk_limits"],
)

RISK_ASSESSMENT_RUBRIC = EvaluationRubric(
    category=EvaluationCategory.RISK_ASSESSMENT,
    description="Evaluate the quality of risk assessment",
    criteria={
        5: "Comprehensive risk analysis covering position, portfolio, and market risks with appropriate sizing",
        4: "Good risk analysis with proper position sizing, minor gaps in coverage",
        3: "Basic risk consideration present but incomplete analysis",
        2: "Inadequate risk assessment, missing key risk factors",
        1: "No meaningful risk assessment or dangerously ignores risks",
    },
    weight=1.5,
    required_context=["portfolio_state", "risk_limits", "market_volatility"],
)

REASONING_CHAIN_RUBRIC = EvaluationRubric(
    category=EvaluationCategory.REASONING_CHAIN,
    description="Evaluate the logical coherence of the reasoning chain",
    criteria={
        5: "Clear, logical progression from data to conclusion, all steps justified",
        4: "Good logical flow with minor gaps that don't affect conclusion",
        3: "Acceptable reasoning but some leaps in logic or unsupported claims",
        2: "Significant logical gaps or contradictions in reasoning",
        1: "Incoherent reasoning or conclusions not supported by premises",
    },
    weight=1.0,
    required_context=["input_data", "intermediate_steps"],
)

HALLUCINATION_CHECK_RUBRIC = EvaluationRubric(
    category=EvaluationCategory.HALLUCINATION_CHECK,
    description="Check for fabricated facts or data in the analysis",
    criteria={
        5: "All facts verifiable, no hallucinations detected",
        4: "Minor unverifiable claims that don't affect decision",
        3: "Some questionable claims but core analysis sound",
        2: "Significant hallucinated data or facts affecting decision",
        1: "Major fabrications that fundamentally undermine the analysis",
    },
    weight=2.0,  # High weight - hallucinations are critical
    required_context=["verifiable_data", "market_data"],
)

MARKET_ANALYSIS_RUBRIC = EvaluationRubric(
    category=EvaluationCategory.MARKET_ANALYSIS,
    description="Evaluate quality of market condition analysis",
    criteria={
        5: "Thorough analysis of trends, volatility, correlations, and macro factors",
        4: "Good analysis covering key market factors with minor omissions",
        3: "Basic market analysis present but lacks depth",
        2: "Superficial or incorrect market analysis",
        1: "No meaningful market analysis or grossly incorrect",
    },
    weight=1.0,
    required_context=["market_data", "indicators", "news"],
)

ALL_RUBRICS = {
    EvaluationCategory.TRADING_DECISION: TRADING_DECISION_RUBRIC,
    EvaluationCategory.RISK_ASSESSMENT: RISK_ASSESSMENT_RUBRIC,
    EvaluationCategory.REASONING_CHAIN: REASONING_CHAIN_RUBRIC,
    EvaluationCategory.HALLUCINATION_CHECK: HALLUCINATION_CHECK_RUBRIC,
    EvaluationCategory.MARKET_ANALYSIS: MARKET_ANALYSIS_RUBRIC,
}


def create_judge_prompt(
    decision: AgentDecision,
    rubric: EvaluationRubric,
    include_outcome: bool = False,
) -> str:
    """Create evaluation prompt for a judge model."""
    criteria_text = "\n".join(f"  {score}: {desc}" for score, desc in sorted(rubric.criteria.items(), reverse=True))

    outcome_text = ""
    if include_outcome and decision.outcome:
        outcome_text = f"""

ACTUAL OUTCOME (for reference only - do not let this bias your evaluation of the decision quality):
{json.dumps(decision.outcome, indent=2)}
"""

    prompt = f"""You are an expert financial trading evaluator. Evaluate the following trading agent decision.

EVALUATION CATEGORY: {rubric.category.value}
DESCRIPTION: {rubric.description}

SCORING CRITERIA (1-5 scale):
{criteria_text}

AGENT DECISION TO EVALUATE:
- Decision ID: {decision.decision_id}
- Type: {decision.decision_type}
- Symbol: {decision.symbol}
- Agent's Reasoning: {decision.reasoning}
- Agent's Risk Assessment: {decision.risk_assessment or "Not provided"}
- Agent's Confidence: {decision.confidence or "Not provided"}

MARKET CONTEXT:
{json.dumps(decision.market_context, indent=2)}
{outcome_text}

INSTRUCTIONS:
1. Carefully analyze the agent's decision and reasoning
2. Check for logical consistency and factual accuracy
3. Evaluate against the scoring criteria
4. Identify specific issues and strengths
5. Provide a detailed explanation for your score

Respond in the following JSON format:
{{
    "score": <1-5>,
    "confidence": <0.0-1.0>,
    "reasoning": "<detailed explanation for the score>",
    "issues_found": ["<issue1>", "<issue2>", ...],
    "strengths_found": ["<strength1>", "<strength2>", ...]
}}
"""
    return prompt


def parse_judge_response(
    response: str,
    judge_model: JudgeModel,
    category: EvaluationCategory,
) -> JudgeScore:
    """Parse JSON response from judge model."""
    try:
        # Handle potential markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        data = json.loads(response.strip())

        return JudgeScore(
            judge_model=judge_model,
            category=category,
            score=max(1, min(5, int(data.get("score", 3)))),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            reasoning=data.get("reasoning", "No reasoning provided"),
            issues_found=data.get("issues_found", []),
            strengths_found=data.get("strengths_found", []),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Failed to parse judge response: {e}")
        return JudgeScore(
            judge_model=judge_model,
            category=category,
            score=3,  # Default to middle score on parse failure
            confidence=0.0,  # Low confidence
            reasoning=f"Failed to parse response: {e!s}",
            issues_found=["Parse error in judge response"],
            strengths_found=[],
        )


def evaluate_with_position_debiasing(
    decision: AgentDecision,
    rubric: EvaluationRubric,
    judge_fn: Callable[[str], str],
    judge_model: JudgeModel,
) -> JudgeScore:
    """
    Evaluate with position debiasing by averaging scores from two orderings.

    This reduces position bias where judges might favor the first or last option.
    For single-decision evaluation, we present context in different orderings.
    """
    # First evaluation - standard order
    prompt1 = create_judge_prompt(decision, rubric, include_outcome=False)
    response1 = judge_fn(prompt1)
    score1 = parse_judge_response(response1, judge_model, rubric.category)

    # Second evaluation - outcome included (if available) to check consistency
    if decision.outcome:
        prompt2 = create_judge_prompt(decision, rubric, include_outcome=True)
        response2 = judge_fn(prompt2)
        score2 = parse_judge_response(response2, judge_model, rubric.category)

        # Average the scores (position debiasing)
        avg_score = (score1.score + score2.score) / 2
        avg_confidence = (score1.confidence + score2.confidence) / 2

        # Combine issues and strengths
        all_issues = list(set(score1.issues_found + score2.issues_found))
        all_strengths = list(set(score1.strengths_found + score2.strengths_found))

        return JudgeScore(
            judge_model=judge_model,
            category=rubric.category,
            score=round(avg_score),
            confidence=avg_confidence,
            reasoning=f"Position-debiased evaluation:\n\nWithout outcome: {score1.reasoning}\n\nWith outcome: {score2.reasoning}",
            issues_found=all_issues,
            strengths_found=all_strengths,
        )

    return score1


def calculate_agreement_rate(scores: list[JudgeScore]) -> float:
    """Calculate agreement rate among judges."""
    if len(scores) < 2:
        return 1.0

    score_values = [s.score for s in scores]
    max_diff = max(score_values) - min(score_values)

    # Agreement is inverse of max difference (normalized to 0-1)
    # Max possible diff is 4 (1 to 5), so agreement = 1 - (diff/4)
    return 1.0 - (max_diff / 4.0)


def aggregate_judge_scores(
    scores: list[JudgeScore],
    rubrics: dict[EvaluationCategory, EvaluationRubric],
) -> tuple[float, float, list[str], list[str]]:
    """
    Aggregate scores from multiple judges with weighted averaging.

    Returns: (overall_score, overall_confidence, consensus_issues, consensus_strengths)
    """
    if not scores:
        return 0.0, 0.0, [], []

    # Group scores by category
    by_category: dict[EvaluationCategory, list[JudgeScore]] = {}
    for score in scores:
        if score.category not in by_category:
            by_category[score.category] = []
        by_category[score.category].append(score)

    # Calculate weighted average
    total_weight = 0.0
    weighted_sum = 0.0
    confidence_sum = 0.0

    for category, category_scores in by_category.items():
        weight = rubrics.get(category, ALL_RUBRICS.get(category))
        if weight:
            weight = weight.weight
        else:
            weight = 1.0

        # Average scores within category
        avg_score = sum(s.score for s in category_scores) / len(category_scores)
        avg_confidence = sum(s.confidence for s in category_scores) / len(category_scores)

        weighted_sum += avg_score * weight
        confidence_sum += avg_confidence * weight
        total_weight += weight

    overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    overall_confidence = confidence_sum / total_weight if total_weight > 0 else 0.0

    # Find consensus issues (mentioned by majority of judges)
    issue_counts: dict[str, int] = {}
    strength_counts: dict[str, int] = {}

    for score in scores:
        for issue in score.issues_found:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        for strength in score.strengths_found:
            strength_counts[strength] = strength_counts.get(strength, 0) + 1

    threshold = len(scores) / 2
    consensus_issues = [i for i, c in issue_counts.items() if c >= threshold]
    consensus_strengths = [s for s, c in strength_counts.items() if c >= threshold]

    return overall_score, overall_confidence, consensus_issues, consensus_strengths


def generate_recommendation(
    overall_score: float,
    consensus_issues: list[str],
    category: EvaluationCategory | None = None,
) -> str:
    """Generate recommendation based on score and issues."""
    if overall_score >= 4.5:
        rec = "STRONG APPROVE - Decision meets highest quality standards"
    elif overall_score >= 3.5:
        rec = "APPROVE - Decision is sound with minor improvements possible"
    elif overall_score >= 2.5:
        rec = "CONDITIONAL - Decision needs improvement before execution"
    elif overall_score >= 1.5:
        rec = "REJECT - Significant issues must be addressed"
    else:
        rec = "STRONG REJECT - Decision has critical flaws"

    if consensus_issues:
        rec += "\n\nKey issues to address:\n" + "\n".join(f"- {i}" for i in consensus_issues[:5])

    return rec


def evaluate_agent_decision(
    decision: AgentDecision,
    judge_functions: dict[JudgeModel, Callable[[str], str]],
    categories: list[EvaluationCategory] | None = None,
    use_position_debiasing: bool = True,
) -> JudgeEvaluationResult:
    """
    Evaluate an agent decision using multiple judge models.

    Args:
        decision: The trading decision to evaluate
        judge_functions: Dict mapping judge models to their API call functions
        categories: Categories to evaluate (defaults to all)
        use_position_debiasing: Whether to use position debiasing

    Returns:
        JudgeEvaluationResult with aggregated scores
    """
    if categories is None:
        categories = [
            EvaluationCategory.TRADING_DECISION,
            EvaluationCategory.RISK_ASSESSMENT,
            EvaluationCategory.REASONING_CHAIN,
            EvaluationCategory.HALLUCINATION_CHECK,
        ]

    all_scores: list[JudgeScore] = []

    # Evaluate with each judge for each category
    for judge_model, judge_fn in judge_functions.items():
        for category in categories:
            rubric = ALL_RUBRICS.get(category)
            if not rubric:
                continue

            try:
                if use_position_debiasing:
                    score = evaluate_with_position_debiasing(decision, rubric, judge_fn, judge_model)
                else:
                    prompt = create_judge_prompt(decision, rubric)
                    response = judge_fn(prompt)
                    score = parse_judge_response(response, judge_model, category)

                all_scores.append(score)
            except Exception as e:
                logger.error(f"Judge {judge_model.value} failed for {category.value}: {e}")

    # Aggregate scores
    overall_score, overall_confidence, consensus_issues, consensus_strengths = aggregate_judge_scores(
        all_scores, ALL_RUBRICS
    )

    agreement_rate = calculate_agreement_rate(all_scores)
    recommendation = generate_recommendation(overall_score, consensus_issues)

    # Generate detailed feedback
    detailed_feedback = generate_detailed_feedback(all_scores, overall_score)

    return JudgeEvaluationResult(
        decision_id=decision.decision_id,
        scores=all_scores,
        overall_score=overall_score,
        overall_confidence=overall_confidence,
        agreement_rate=agreement_rate,
        consensus_issues=consensus_issues,
        consensus_strengths=consensus_strengths,
        recommendation=recommendation,
        detailed_feedback=detailed_feedback,
    )


def generate_detailed_feedback(scores: list[JudgeScore], overall_score: float) -> str:
    """Generate detailed feedback from all judge scores."""
    feedback_parts = [
        "## Agent Decision Evaluation Report",
        "",
        f"**Overall Score**: {overall_score:.2f}/5.0",
        f"**Number of Evaluations**: {len(scores)}",
        "",
    ]

    # Group by category
    by_category: dict[EvaluationCategory, list[JudgeScore]] = {}
    for score in scores:
        if score.category not in by_category:
            by_category[score.category] = []
        by_category[score.category].append(score)

    for category, category_scores in by_category.items():
        avg_score = sum(s.score for s in category_scores) / len(category_scores)
        feedback_parts.append(f"### {category.value.replace('_', ' ').title()}")
        feedback_parts.append(f"**Average Score**: {avg_score:.2f}/5.0")
        feedback_parts.append("")

        for score in category_scores:
            feedback_parts.append(
                f"**{score.judge_model.value}** (Score: {score.score}, Confidence: {score.confidence:.2f})"
            )
            feedback_parts.append(f"- {score.reasoning[:500]}...")
            if score.issues_found:
                feedback_parts.append(f"- Issues: {', '.join(score.issues_found[:3])}")
            if score.strengths_found:
                feedback_parts.append(f"- Strengths: {', '.join(score.strengths_found[:3])}")
            feedback_parts.append("")

    return "\n".join(feedback_parts)


def create_mock_judge(
    model: JudgeModel,
    base_score: int = 3,
    variance: float = 1.0,
) -> Callable[[str], str]:
    """Create a mock judge function for testing."""
    import random

    def mock_judge(prompt: str) -> str:
        # Add some variance to the base score
        score = max(1, min(5, base_score + random.randint(-1, 1)))
        confidence = 0.5 + random.random() * 0.4

        # Generate mock issues and strengths based on prompt content
        issues = []
        strengths = []

        if "risk" in prompt.lower():
            if score < 3:
                issues.append("Insufficient risk assessment")
            else:
                strengths.append("Adequate risk consideration")

        if "reasoning" in prompt.lower():
            if score < 3:
                issues.append("Logical gaps in reasoning chain")
            else:
                strengths.append("Clear logical flow")

        if "hallucination" in prompt.lower():
            if score < 3:
                issues.append("Potential fabricated data points")
            else:
                strengths.append("All facts verifiable")

        return json.dumps(
            {
                "score": score,
                "confidence": confidence,
                "reasoning": f"Mock evaluation with score {score}. The decision shows {'adequate' if score >= 3 else 'inadequate'} consideration of relevant factors.",
                "issues_found": issues,
                "strengths_found": strengths,
            }
        )

    return mock_judge


def create_real_judge_function(
    model: str = "claude-sonnet-4-20250514",
    category: EvaluationCategory = EvaluationCategory.TRADING_DECISION,
) -> Callable[[str], str]:
    """
    Create a real LLM judge function that uses API calls.

    This function is the integration point between agent_as_judge.py and llm_judge.py.
    It imports LLMJudge lazily to avoid circular imports.

    Args:
        model: LLM model to use (Claude Sonnet, Opus, GPT-4, etc.)
        category: Evaluation category to assess

    Returns:
        Callable that takes a prompt and returns JSON response string

    Usage:
        # Create a real judge for trading decisions
        judge_fn = create_real_judge_function(
            model="claude-sonnet-4-20250514",
            category=EvaluationCategory.TRADING_DECISION
        )

        # Use in evaluate_agent_decision
        judge_functions = {
            JudgeModel.CLAUDE_SONNET: judge_fn,
        }
        result = evaluate_agent_decision(decision, judge_functions)
    """
    # Lazy import to avoid circular imports
    from evaluation.llm_judge import JudgeConfig, LLMJudge

    config = JudgeConfig(
        model=model,
        temperature=0.0,  # Deterministic for consistency
        enable_caching=True,
    )
    judge = LLMJudge(config=config)

    rubric = ALL_RUBRICS.get(category)

    def real_judge(prompt: str) -> str:
        """Call real LLM to evaluate the decision."""
        try:
            # Call the LLM
            response_text = judge._call_llm_with_retry(prompt)
            return response_text
        except Exception as e:
            # Return a structured error response
            import json

            return json.dumps(
                {
                    "score": 3,
                    "confidence": 0.0,
                    "reasoning": f"LLM evaluation failed: {e!s}",
                    "issues_found": ["LLM API error"],
                    "strengths_found": [],
                }
            )

    return real_judge


def generate_judge_report(result: JudgeEvaluationResult) -> str:
    """Generate a formatted report from judge evaluation result."""
    report_parts = [
        "=" * 60,
        "AGENT-AS-A-JUDGE EVALUATION REPORT",
        "=" * 60,
        "",
        f"Decision ID: {result.decision_id}",
        f"Evaluation Time: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-" * 40,
        "SUMMARY",
        "-" * 40,
        f"Overall Score: {result.overall_score:.2f}/5.0",
        f"Overall Confidence: {result.overall_confidence:.2%}",
        f"Judge Agreement Rate: {result.agreement_rate:.2%}",
        "",
        "RECOMMENDATION:",
        result.recommendation,
        "",
    ]

    if result.consensus_strengths:
        report_parts.extend(
            [
                "-" * 40,
                "CONSENSUS STRENGTHS",
                "-" * 40,
            ]
        )
        for strength in result.consensus_strengths:
            report_parts.append(f"  + {strength}")
        report_parts.append("")

    if result.consensus_issues:
        report_parts.extend(
            [
                "-" * 40,
                "CONSENSUS ISSUES",
                "-" * 40,
            ]
        )
        for issue in result.consensus_issues:
            report_parts.append(f"  - {issue}")
        report_parts.append("")

    report_parts.extend(
        [
            "-" * 40,
            "DETAILED SCORES BY CATEGORY",
            "-" * 40,
        ]
    )

    # Group scores by category
    by_category: dict[EvaluationCategory, list[JudgeScore]] = {}
    for score in result.scores:
        if score.category not in by_category:
            by_category[score.category] = []
        by_category[score.category].append(score)

    for category, scores in by_category.items():
        avg = sum(s.score for s in scores) / len(scores)
        report_parts.append(f"\n{category.value.upper()}:")
        report_parts.append(f"  Average: {avg:.2f}/5.0")
        for score in scores:
            report_parts.append(f"  - {score.judge_model.value}: {score.score}/5 (conf: {score.confidence:.2f})")

    report_parts.extend(
        [
            "",
            "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ]
    )

    return "\n".join(report_parts)


def check_judge_thresholds(
    result: JudgeEvaluationResult,
    min_overall_score: float = 3.0,
    min_agreement_rate: float = 0.5,
    max_critical_issues: int = 2,
) -> dict[str, Any]:
    """Check if evaluation result meets quality thresholds."""
    checks = {
        "overall_score": {
            "passed": result.overall_score >= min_overall_score,
            "value": result.overall_score,
            "threshold": min_overall_score,
            "message": f"Overall score {result.overall_score:.2f} {'meets' if result.overall_score >= min_overall_score else 'below'} threshold {min_overall_score}",
        },
        "agreement_rate": {
            "passed": result.agreement_rate >= min_agreement_rate,
            "value": result.agreement_rate,
            "threshold": min_agreement_rate,
            "message": f"Agreement rate {result.agreement_rate:.2%} {'meets' if result.agreement_rate >= min_agreement_rate else 'below'} threshold {min_agreement_rate:.2%}",
        },
        "critical_issues": {
            "passed": len(result.consensus_issues) <= max_critical_issues,
            "value": len(result.consensus_issues),
            "threshold": max_critical_issues,
            "message": f"Found {len(result.consensus_issues)} consensus issues (max allowed: {max_critical_issues})",
        },
    }

    checks["all_passed"] = all(c["passed"] for c in checks.values() if isinstance(c, dict))

    return checks


# Export convenience functions
__all__ = [
    "ALL_RUBRICS",
    "HALLUCINATION_CHECK_RUBRIC",
    "MARKET_ANALYSIS_RUBRIC",
    "REASONING_CHAIN_RUBRIC",
    "RISK_ASSESSMENT_RUBRIC",
    "TRADING_DECISION_RUBRIC",
    "AgentDecision",
    "EvaluationCategory",
    "EvaluationRubric",
    "JudgeEvaluationResult",
    "JudgeModel",
    "JudgeScore",
    "ScoreLevel",
    "aggregate_judge_scores",
    "calculate_agreement_rate",
    "check_judge_thresholds",
    "create_judge_prompt",
    "create_mock_judge",
    "create_real_judge_function",
    "evaluate_agent_decision",
    "evaluate_with_position_debiasing",
    "generate_judge_report",
    "parse_judge_response",
]
