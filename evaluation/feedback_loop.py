"""
Evaluator-Optimizer Feedback Loop

Implements a closed-loop evaluation system that automatically improves
AI agent performance through iterative evaluation and prompt refinement.

Architecture:
1. Evaluate: Score agent outputs against criteria
2. Identify: Find weak areas and patterns
3. Refine: Generate prompt improvements
4. Apply: Update agent prompts
5. Repeat: Until convergence or max iterations

References:
- Agent Evaluation Best Practices (2025)
- Self-Improving AI Systems patterns
- UPGRADE-003 specification
"""

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ConvergenceReason(Enum):
    """Reasons for loop termination."""

    TARGET_REACHED = "target_reached"
    SCORE_CONVERGED = "score_converged"
    MAX_ITERATIONS = "max_iterations"
    NO_IMPROVEMENT = "no_improvement"
    ERROR = "error"


class WeaknessCategory(Enum):
    """Categories of identified weaknesses."""

    ACCURACY = "accuracy"
    REASONING = "reasoning"
    CONSISTENCY = "consistency"
    HALLUCINATION = "hallucination"
    RISK_ASSESSMENT = "risk_assessment"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    RESPONSE_FORMAT = "response_format"


@dataclass
class Weakness:
    """Identified weakness in agent performance."""

    category: WeaknessCategory
    description: str
    severity: float  # 0.0 to 1.0
    affected_test_cases: list[str]
    suggested_fix: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptRefinement:
    """A refinement to apply to an agent's prompt."""

    target_section: str  # e.g., "system_prompt", "few_shot_examples"
    original_text: str
    refined_text: str
    weakness_addressed: WeaknessCategory
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackCycle:
    """Record of a single feedback cycle."""

    iteration: int
    timestamp: datetime
    evaluation_scores: dict[str, float]
    aggregate_score: float
    identified_weaknesses: list[Weakness]
    applied_refinements: list[PromptRefinement]
    improved_scores: dict[str, float] | None = None
    improvement_delta: float = 0.0
    duration_seconds: float = 0.0


@dataclass
class FeedbackResult:
    """Result of running the feedback loop."""

    converged: bool
    reason: ConvergenceReason
    iterations_completed: int
    initial_scores: dict[str, float]
    final_scores: dict[str, float]
    initial_aggregate: float
    final_aggregate: float
    total_improvement: float
    cycles: list[FeedbackCycle]
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "converged": self.converged,
            "reason": self.reason.value,
            "iterations_completed": self.iterations_completed,
            "initial_scores": self.initial_scores,
            "final_scores": self.final_scores,
            "initial_aggregate": self.initial_aggregate,
            "final_aggregate": self.final_aggregate,
            "total_improvement": self.total_improvement,
            "duration_seconds": self.duration_seconds,
            "cycle_count": len(self.cycles),
        }


@dataclass
class EvaluationScore:
    """Score from evaluating an agent on test cases."""

    test_case_id: str
    passed: bool
    score: float
    category: str
    reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)


class EvaluatorOptimizerLoop:
    """
    Closed-loop evaluation with automatic agent improvement.

    Implements the Evaluator-Optimizer pattern:
    - Evaluates agent performance on test cases
    - Identifies weaknesses and patterns
    - Generates and applies prompt refinements
    - Iterates until convergence or max iterations

    Usage:
        loop = EvaluatorOptimizerLoop(
            evaluator=my_evaluator,
            max_iterations=5,
            target_score=0.85,
        )
        result = loop.run(agent, test_cases)
        if result.converged:
            print(f"Improved from {result.initial_aggregate:.2f} to {result.final_aggregate:.2f}")
    """

    def __init__(
        self,
        evaluator: Any | None = None,
        evaluate_fn: Callable | None = None,
        max_iterations: int = 5,
        convergence_threshold: float = 0.02,
        target_score: float = 0.80,
        no_improvement_patience: int = 2,
        min_improvement_delta: float = 0.01,
        score_weights: dict[str, float] | None = None,
    ):
        """
        Initialize the feedback loop.

        Args:
            evaluator: AgentEvaluator instance (optional)
            evaluate_fn: Custom evaluation function (alternative to evaluator)
            max_iterations: Maximum feedback cycles
            convergence_threshold: Score delta to consider converged
            target_score: Target aggregate score to achieve
            no_improvement_patience: Iterations without improvement before stopping
            min_improvement_delta: Minimum score improvement to count as progress
            score_weights: Weights for aggregating scores by category
        """
        self.evaluator = evaluator
        self.evaluate_fn = evaluate_fn
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.target_score = target_score
        self.no_improvement_patience = no_improvement_patience
        self.min_improvement_delta = min_improvement_delta
        self.score_weights = score_weights or {}

        # State
        self.history: list[FeedbackCycle] = []
        self._no_improvement_count = 0
        self._best_score = 0.0

    def run(
        self,
        agent: Any,
        test_cases: list[Any],
        context: dict[str, Any] | None = None,
    ) -> FeedbackResult:
        """
        Run the feedback loop until convergence.

        Args:
            agent: Agent to improve (must have modifiable prompts)
            test_cases: Test cases to evaluate against
            context: Additional context for evaluation

        Returns:
            FeedbackResult with convergence status and improvement metrics
        """
        start_time = datetime.utcnow()
        context = context or {}

        # Initial evaluation
        initial_scores = self._evaluate(agent, test_cases, context)
        initial_aggregate = self._aggregate_scores(initial_scores)
        self._best_score = initial_aggregate

        current_scores = initial_scores.copy()

        for iteration in range(self.max_iterations):
            cycle_start = datetime.utcnow()

            # Check if target already reached
            current_aggregate = self._aggregate_scores(current_scores)
            if current_aggregate >= self.target_score:
                return self._create_result(
                    converged=True,
                    reason=ConvergenceReason.TARGET_REACHED,
                    iterations=iteration,
                    initial_scores=initial_scores,
                    final_scores=current_scores,
                    start_time=start_time,
                )

            # Identify weaknesses
            weaknesses = self._identify_weaknesses(current_scores, test_cases)

            if not weaknesses:
                # No weaknesses found but target not reached
                return self._create_result(
                    converged=True,
                    reason=ConvergenceReason.SCORE_CONVERGED,
                    iterations=iteration,
                    initial_scores=initial_scores,
                    final_scores=current_scores,
                    start_time=start_time,
                )

            # Generate refinements
            refinements = self._generate_refinements(weaknesses, agent)

            # Apply refinements
            if refinements:
                self._apply_refinements(agent, refinements)

            # Re-evaluate
            new_scores = self._evaluate(agent, test_cases, context)
            new_aggregate = self._aggregate_scores(new_scores)

            # Record cycle
            cycle_end = datetime.utcnow()
            cycle = FeedbackCycle(
                iteration=iteration,
                timestamp=cycle_start,
                evaluation_scores=current_scores.copy(),
                aggregate_score=current_aggregate,
                identified_weaknesses=weaknesses,
                applied_refinements=refinements,
                improved_scores=new_scores.copy(),
                improvement_delta=new_aggregate - current_aggregate,
                duration_seconds=(cycle_end - cycle_start).total_seconds(),
            )
            self.history.append(cycle)

            # Check for improvement
            if new_aggregate > self._best_score + self.min_improvement_delta:
                self._best_score = new_aggregate
                self._no_improvement_count = 0
            else:
                self._no_improvement_count += 1

            # Check for no improvement patience
            if self._no_improvement_count >= self.no_improvement_patience:
                return self._create_result(
                    converged=False,
                    reason=ConvergenceReason.NO_IMPROVEMENT,
                    iterations=iteration + 1,
                    initial_scores=initial_scores,
                    final_scores=new_scores,
                    start_time=start_time,
                )

            # Check for convergence (small but positive delta)
            # Only trigger SCORE_CONVERGED if there's positive improvement that's small
            # If delta is 0 or negative, let NO_IMPROVEMENT patience handle it
            improvement = new_aggregate - current_aggregate
            if 0 < improvement < self.convergence_threshold:
                return self._create_result(
                    converged=True,
                    reason=ConvergenceReason.SCORE_CONVERGED,
                    iterations=iteration + 1,
                    initial_scores=initial_scores,
                    final_scores=new_scores,
                    start_time=start_time,
                )

            current_scores = new_scores

        # Max iterations reached
        return self._create_result(
            converged=False,
            reason=ConvergenceReason.MAX_ITERATIONS,
            iterations=self.max_iterations,
            initial_scores=initial_scores,
            final_scores=current_scores,
            start_time=start_time,
        )

    def _evaluate(
        self,
        agent: Any,
        test_cases: list[Any],
        context: dict[str, Any],
    ) -> dict[str, float]:
        """
        Evaluate agent on test cases.

        Args:
            agent: Agent to evaluate
            test_cases: Test cases
            context: Evaluation context

        Returns:
            Dictionary of scores by category
        """
        if self.evaluate_fn:
            return self.evaluate_fn(agent, test_cases, context)

        if self.evaluator:
            # Use provided evaluator
            result = self.evaluator.evaluate(agent, test_cases)
            return result.scores if hasattr(result, "scores") else {"overall": 0.5}

        # Default evaluation - simple pass rate
        scores: dict[str, list[float]] = {}
        for tc in test_cases:
            category = getattr(tc, "category", "overall")
            if category not in scores:
                scores[category] = []

            # Simple evaluation based on expected vs actual
            if hasattr(tc, "expected") and hasattr(tc, "actual"):
                score = 1.0 if tc.expected == tc.actual else 0.0
            else:
                score = 0.5  # Unknown

            scores[category].append(score)

        # Aggregate by category
        return {category: statistics.mean(values) for category, values in scores.items()}

    def _aggregate_scores(self, scores: dict[str, float]) -> float:
        """
        Calculate aggregate score from category scores.

        Args:
            scores: Scores by category

        Returns:
            Weighted aggregate score
        """
        if not scores:
            return 0.0

        if self.score_weights:
            weighted_sum = 0.0
            total_weight = 0.0
            for category, score in scores.items():
                weight = self.score_weights.get(category, 1.0)
                weighted_sum += score * weight
                total_weight += weight
            return weighted_sum / total_weight if total_weight > 0 else 0.0

        return statistics.mean(scores.values())

    def _identify_weaknesses(
        self,
        scores: dict[str, float],
        test_cases: list[Any],
    ) -> list[Weakness]:
        """
        Identify weaknesses from evaluation scores.

        Args:
            scores: Current scores by category
            test_cases: Test cases evaluated

        Returns:
            List of identified weaknesses
        """
        weaknesses = []

        # Map categories to weakness types
        category_mapping = {
            "accuracy": WeaknessCategory.ACCURACY,
            "reasoning": WeaknessCategory.REASONING,
            "consistency": WeaknessCategory.CONSISTENCY,
            "hallucination": WeaknessCategory.HALLUCINATION,
            "risk": WeaknessCategory.RISK_ASSESSMENT,
            "confidence": WeaknessCategory.CONFIDENCE_CALIBRATION,
            "format": WeaknessCategory.RESPONSE_FORMAT,
        }

        # Identify weak categories (below threshold)
        weakness_threshold = 0.7

        for category, score in scores.items():
            if score < weakness_threshold:
                severity = 1.0 - score  # Higher severity for lower scores

                # Map to weakness category
                weakness_cat = category_mapping.get(category.lower(), WeaknessCategory.ACCURACY)

                # Find affected test cases
                affected = []
                for tc in test_cases:
                    tc_category = getattr(tc, "category", "overall")
                    if tc_category == category:
                        tc_id = getattr(tc, "id", getattr(tc, "name", str(tc)))
                        affected.append(tc_id)

                weakness = Weakness(
                    category=weakness_cat,
                    description=f"Low score in {category}: {score:.2f}",
                    severity=severity,
                    affected_test_cases=affected[:5],  # Limit to 5
                    suggested_fix=self._suggest_fix(weakness_cat, score),
                )
                weaknesses.append(weakness)

        # Sort by severity
        weaknesses.sort(key=lambda w: w.severity, reverse=True)

        return weaknesses

    def _suggest_fix(self, category: WeaknessCategory, score: float) -> str:
        """Generate a suggested fix for a weakness category."""
        suggestions = {
            WeaknessCategory.ACCURACY: (
                "Add more specific examples in few-shot prompts. " "Include explicit decision criteria."
            ),
            WeaknessCategory.REASONING: (
                "Add chain-of-thought prompting. " "Request step-by-step reasoning in system prompt."
            ),
            WeaknessCategory.CONSISTENCY: (
                "Add response format templates. " "Include consistency examples in prompts."
            ),
            WeaknessCategory.HALLUCINATION: (
                "Add explicit instruction to only use provided data. " "Include 'if uncertain, say so' directive."
            ),
            WeaknessCategory.RISK_ASSESSMENT: (
                "Add risk evaluation checklist to prompt. " "Include risk scenarios in examples."
            ),
            WeaknessCategory.CONFIDENCE_CALIBRATION: (
                "Add confidence calibration examples. " "Include instruction to express uncertainty levels."
            ),
            WeaknessCategory.RESPONSE_FORMAT: (
                "Add explicit format specification. " "Include format validation examples."
            ),
        }
        return suggestions.get(category, "Review and improve related prompt sections.")

    def _generate_refinements(
        self,
        weaknesses: list[Weakness],
        agent: Any,
    ) -> list[PromptRefinement]:
        """
        Generate prompt refinements for identified weaknesses.

        Args:
            weaknesses: Identified weaknesses
            agent: Agent to refine

        Returns:
            List of prompt refinements
        """
        refinements = []

        # Get current system prompt
        system_prompt = getattr(agent, "system_prompt", "")

        for weakness in weaknesses[:3]:  # Limit to top 3 weaknesses
            refinement = self._create_refinement(weakness, system_prompt)
            if refinement:
                refinements.append(refinement)

        return refinements

    def _create_refinement(
        self,
        weakness: Weakness,
        current_prompt: str,
    ) -> PromptRefinement | None:
        """Create a refinement for a specific weakness."""
        # Define refinement patterns for each category
        refinement_patterns = {
            WeaknessCategory.ACCURACY: (
                "\n\nIMPORTANT: Ensure accuracy by:\n"
                "- Double-checking all numerical values\n"
                "- Verifying data before making decisions\n"
                "- Using only provided information\n"
            ),
            WeaknessCategory.REASONING: (
                "\n\nReasoning Guidelines:\n"
                "- Think step by step before reaching conclusions\n"
                "- Explicitly state your reasoning chain\n"
                "- Consider multiple perspectives\n"
            ),
            WeaknessCategory.CONSISTENCY: (
                "\n\nConsistency Requirements:\n"
                "- Maintain consistent terminology\n"
                "- Use the same format for similar responses\n"
                "- Ensure conclusions match reasoning\n"
            ),
            WeaknessCategory.HALLUCINATION: (
                "\n\nData Integrity:\n"
                "- ONLY use data explicitly provided\n"
                "- If information is missing, state 'insufficient data'\n"
                "- Never invent or assume values\n"
            ),
            WeaknessCategory.RISK_ASSESSMENT: (
                "\n\nRisk Assessment Framework:\n"
                "- Identify potential risks before decisions\n"
                "- Quantify risk when possible\n"
                "- Consider worst-case scenarios\n"
            ),
            WeaknessCategory.CONFIDENCE_CALIBRATION: (
                "\n\nConfidence Calibration:\n"
                "- Express uncertainty honestly (0-100%)\n"
                "- Lower confidence when data is limited\n"
                "- Provide confidence intervals when relevant\n"
            ),
            WeaknessCategory.RESPONSE_FORMAT: (
                "\n\nResponse Format:\n"
                "- Follow the specified output format exactly\n"
                "- Include all required fields\n"
                "- Use consistent structure\n"
            ),
        }

        addition = refinement_patterns.get(weakness.category)
        if not addition:
            return None

        return PromptRefinement(
            target_section="system_prompt",
            original_text=current_prompt,
            refined_text=current_prompt + addition,
            weakness_addressed=weakness.category,
            confidence=0.7,
            metadata={"severity": weakness.severity},
        )

    def _apply_refinements(
        self,
        agent: Any,
        refinements: list[PromptRefinement],
    ) -> None:
        """
        Apply refinements to agent.

        Args:
            agent: Agent to update
            refinements: Refinements to apply
        """
        for refinement in refinements:
            if refinement.target_section == "system_prompt":
                if hasattr(agent, "system_prompt"):
                    agent.system_prompt = refinement.refined_text
                elif hasattr(agent, "update_system_prompt"):
                    agent.update_system_prompt(refinement.refined_text)

    def _create_result(
        self,
        converged: bool,
        reason: ConvergenceReason,
        iterations: int,
        initial_scores: dict[str, float],
        final_scores: dict[str, float],
        start_time: datetime,
    ) -> FeedbackResult:
        """Create a FeedbackResult from loop state."""
        end_time = datetime.utcnow()
        initial_agg = self._aggregate_scores(initial_scores)
        final_agg = self._aggregate_scores(final_scores)

        return FeedbackResult(
            converged=converged,
            reason=reason,
            iterations_completed=iterations,
            initial_scores=initial_scores,
            final_scores=final_scores,
            initial_aggregate=initial_agg,
            final_aggregate=final_agg,
            total_improvement=final_agg - initial_agg,
            cycles=self.history.copy(),
            duration_seconds=(end_time - start_time).total_seconds(),
        )

    def reset(self) -> None:
        """Reset loop state for reuse."""
        self.history = []
        self._no_improvement_count = 0
        self._best_score = 0.0


def create_feedback_loop(
    evaluator: Any | None = None,
    max_iterations: int = 5,
    target_score: float = 0.80,
    **kwargs,
) -> EvaluatorOptimizerLoop:
    """
    Factory function to create a feedback loop.

    Args:
        evaluator: AgentEvaluator instance
        max_iterations: Maximum iterations
        target_score: Target score to achieve
        **kwargs: Additional arguments for EvaluatorOptimizerLoop

    Returns:
        Configured EvaluatorOptimizerLoop instance
    """
    return EvaluatorOptimizerLoop(
        evaluator=evaluator,
        max_iterations=max_iterations,
        target_score=target_score,
        **kwargs,
    )


def generate_feedback_report(result: FeedbackResult) -> str:
    """
    Generate a human-readable report from feedback loop result.

    Args:
        result: FeedbackResult from loop.run()

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "EVALUATOR-OPTIMIZER FEEDBACK LOOP REPORT",
        "=" * 60,
        "",
        f"Status: {'CONVERGED' if result.converged else 'NOT CONVERGED'}",
        f"Reason: {result.reason.value}",
        f"Iterations: {result.iterations_completed}",
        f"Duration: {result.duration_seconds:.2f}s",
        "",
        "SCORE IMPROVEMENT",
        "-" * 40,
        f"Initial Score: {result.initial_aggregate:.3f}",
        f"Final Score:   {result.final_aggregate:.3f}",
        f"Improvement:   {result.total_improvement:+.3f} "
        f"({result.total_improvement/max(result.initial_aggregate, 0.001)*100:+.1f}%)",
        "",
        "CATEGORY BREAKDOWN",
        "-" * 40,
    ]

    # Show category scores
    for category in result.final_scores:
        initial = result.initial_scores.get(category, 0)
        final = result.final_scores[category]
        delta = final - initial
        lines.append(f"  {category}: {initial:.3f} -> {final:.3f} ({delta:+.3f})")

    # Show cycle history
    if result.cycles:
        lines.extend(
            [
                "",
                "CYCLE HISTORY",
                "-" * 40,
            ]
        )
        for cycle in result.cycles:
            lines.append(
                f"  Iteration {cycle.iteration}: "
                f"score={cycle.aggregate_score:.3f}, "
                f"delta={cycle.improvement_delta:+.3f}, "
                f"weaknesses={len(cycle.identified_weaknesses)}, "
                f"refinements={len(cycle.applied_refinements)}"
            )

    lines.extend(["", "=" * 60])

    return "\n".join(lines)
