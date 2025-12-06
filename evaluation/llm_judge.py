"""
Real LLM Judge Implementation.

Replaces mock judges with actual LLM API calls for production-quality
evaluation of trading agent decisions.

Features:
- Claude Sonnet/Opus judge via Anthropic API
- GPT-4 judge via OpenAI API (optional)
- Position debiasing through multiple evaluations
- Configurable rubrics and thresholds
- Cost tracking and rate limiting
- Caching for repeated evaluations

Based on Phase 6 research findings:
- Mock judges (random scores) replaced with real LLM evaluation
- Agent-as-a-Judge paradigm for trajectory-level assessment
- Multi-judge ensemble for robust scoring

Version: 1.0 (December 2025)
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from evaluation.agent_as_judge import (
    ALL_RUBRICS,
    AgentDecision,
    EvaluationCategory,
    EvaluationRubric,
    JudgeEvaluationResult,
    JudgeModel,
    JudgeScore,
    aggregate_judge_scores,
    calculate_agreement_rate,
    create_judge_prompt,
    generate_recommendation,
    parse_judge_response,
)


logger = logging.getLogger(__name__)


@dataclass
class JudgeConfig:
    """Configuration for LLM judge."""

    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0  # Deterministic for consistency
    max_tokens: int = 1024
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    enable_caching: bool = True
    track_costs: bool = True


@dataclass
class JudgeCostTracker:
    """Track costs for judge evaluations."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    total_cost_usd: float = 0.0

    # Cost per 1M tokens (approximate, update as needed)
    INPUT_COST_PER_M: float = 3.0  # Sonnet input
    OUTPUT_COST_PER_M: float = 15.0  # Sonnet output

    def add_request(self, input_tokens: int, output_tokens: int) -> None:
        """Add a request's token counts."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1

        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_M
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_M
        self.total_cost_usd += input_cost + output_cost

    def get_summary(self) -> dict[str, Any]:
        """Get cost summary."""
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "avg_cost_per_request": round(self.total_cost_usd / max(1, self.total_requests), 4),
        }


class LLMJudge:
    """
    Real LLM judge using Claude or GPT-4 for evaluation.

    Replaces mock judges with actual API calls for production-quality
    assessment of trading agent decisions.

    Usage:
        # Create judge
        judge = LLMJudge(config=JudgeConfig(model="claude-sonnet-4-20250514"))

        # Evaluate a decision
        score = judge.evaluate(decision, rubric=TRADING_DECISION_RUBRIC)

        # Evaluate with position debiasing
        score = judge.evaluate_with_debiasing(decision, rubric, num_permutations=3)
    """

    def __init__(
        self,
        config: JudgeConfig | None = None,
        anthropic_client: Any | None = None,
        openai_client: Any | None = None,
    ):
        """
        Initialize LLM judge.

        Args:
            config: Judge configuration
            anthropic_client: Pre-configured Anthropic client (optional)
            openai_client: Pre-configured OpenAI client (optional)
        """
        self.config = config or JudgeConfig()
        self.cost_tracker = JudgeCostTracker()
        self._cache: dict[str, JudgeScore] = {}

        # Initialize clients lazily
        self._anthropic_client = anthropic_client
        self._openai_client = openai_client

    @property
    def anthropic_client(self):
        """Lazily initialize Anthropic client."""
        if self._anthropic_client is None:
            try:
                from anthropic import Anthropic

                self._anthropic_client = Anthropic()
            except ImportError:
                logger.warning("anthropic package not installed. Install with: pip install anthropic")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                return None
        return self._anthropic_client

    @property
    def openai_client(self):
        """Lazily initialize OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import OpenAI

                self._openai_client = OpenAI()
            except ImportError:
                logger.warning("openai package not installed. Install with: pip install openai")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                return None
        return self._openai_client

    def evaluate(
        self,
        decision: AgentDecision,
        rubric: EvaluationRubric | None = None,
        category: EvaluationCategory | None = None,
    ) -> JudgeScore:
        """
        Evaluate an agent decision using real LLM.

        Args:
            decision: AgentDecision to evaluate
            rubric: Evaluation rubric (defaults to TRADING_DECISION_RUBRIC)
            category: Evaluation category (inferred from rubric if not provided)

        Returns:
            JudgeScore with evaluation result
        """
        if rubric is None:
            rubric = ALL_RUBRICS[EvaluationCategory.TRADING_DECISION]

        if category is None:
            category = rubric.category

        # Check cache
        cache_key = self._get_cache_key(decision, rubric)
        if self.config.enable_caching and cache_key in self._cache:
            logger.debug(f"Using cached evaluation for {decision.decision_id}")
            return self._cache[cache_key]

        # Create prompt
        prompt = create_judge_prompt(decision, rubric, include_outcome=False)

        # Call LLM with retries
        response_text = self._call_llm_with_retry(prompt)

        # Parse response
        judge_model = self._get_judge_model()
        score = parse_judge_response(response_text, judge_model, category)

        # Cache result
        if self.config.enable_caching:
            self._cache[cache_key] = score

        return score

    def evaluate_with_debiasing(
        self,
        decision: AgentDecision,
        rubric: EvaluationRubric | None = None,
        num_permutations: int = 3,
    ) -> JudgeScore:
        """
        Evaluate with position debiasing through multiple evaluations.

        Position debiasing reduces bias where judges might favor certain
        orderings or phrasings by averaging scores across multiple evaluations.

        Args:
            decision: AgentDecision to evaluate
            rubric: Evaluation rubric
            num_permutations: Number of evaluations to average

        Returns:
            JudgeScore with debiased score (median of evaluations)
        """
        if rubric is None:
            rubric = ALL_RUBRICS[EvaluationCategory.TRADING_DECISION]

        scores: list[JudgeScore] = []

        for i in range(num_permutations):
            # Vary the prompt slightly for each evaluation
            include_outcome = (i % 2 == 1) and decision.outcome is not None
            prompt = create_judge_prompt(decision, rubric, include_outcome=include_outcome)

            response_text = self._call_llm_with_retry(prompt)
            judge_model = self._get_judge_model()
            score = parse_judge_response(response_text, judge_model, rubric.category)
            scores.append(score)

        # Calculate median score
        score_values = sorted([s.score for s in scores])
        median_score = score_values[len(score_values) // 2]

        # Average confidence
        avg_confidence = sum(s.confidence for s in scores) / len(scores)

        # Combine issues and strengths (deduplicated)
        all_issues = list(set(issue for s in scores for issue in s.issues_found))
        all_strengths = list(set(strength for s in scores for strength in s.strengths_found))

        # Combine reasoning
        reasoning_parts = [f"Evaluation {i+1}: {s.reasoning[:200]}..." for i, s in enumerate(scores)]
        combined_reasoning = f"Position-debiased evaluation (median of {num_permutations}):\n\n" + "\n\n".join(
            reasoning_parts
        )

        return JudgeScore(
            judge_model=self._get_judge_model(),
            category=rubric.category,
            score=median_score,
            confidence=avg_confidence,
            reasoning=combined_reasoning,
            issues_found=all_issues,
            strengths_found=all_strengths,
        )

    def evaluate_full(
        self,
        decision: AgentDecision,
        categories: list[EvaluationCategory] | None = None,
        use_debiasing: bool = False,
    ) -> JudgeEvaluationResult:
        """
        Full evaluation across multiple categories.

        Args:
            decision: AgentDecision to evaluate
            categories: Categories to evaluate (defaults to standard set)
            use_debiasing: Whether to use position debiasing

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

        for category in categories:
            rubric = ALL_RUBRICS.get(category)
            if rubric is None:
                continue

            try:
                if use_debiasing:
                    score = self.evaluate_with_debiasing(decision, rubric)
                else:
                    score = self.evaluate(decision, rubric, category)
                all_scores.append(score)
            except Exception as e:
                logger.error(f"Evaluation failed for {category.value}: {e}")

        # Aggregate scores
        overall_score, overall_confidence, consensus_issues, consensus_strengths = aggregate_judge_scores(
            all_scores, ALL_RUBRICS
        )

        agreement_rate = calculate_agreement_rate(all_scores)
        recommendation = generate_recommendation(overall_score, consensus_issues)

        return JudgeEvaluationResult(
            decision_id=decision.decision_id,
            scores=all_scores,
            overall_score=overall_score,
            overall_confidence=overall_confidence,
            agreement_rate=agreement_rate,
            consensus_issues=consensus_issues,
            consensus_strengths=consensus_strengths,
            recommendation=recommendation,
            detailed_feedback=self._generate_detailed_feedback(all_scores, overall_score),
        )

    def _call_llm_with_retry(self, prompt: str) -> str:
        """
        Call LLM with retry logic.

        Args:
            prompt: Prompt to send

        Returns:
            LLM response text
        """
        last_error = None

        for attempt in range(self.config.retry_attempts):
            try:
                return self._call_llm(prompt)
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")

                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay_seconds * (attempt + 1))

        # All retries failed, return error response
        logger.error(f"All LLM call attempts failed: {last_error}")
        return json.dumps(
            {
                "score": 3,
                "confidence": 0.0,
                "reasoning": f"LLM call failed after {self.config.retry_attempts} attempts: {last_error!s}",
                "issues_found": ["LLM evaluation failed"],
                "strengths_found": [],
            }
        )

    def _call_llm(self, prompt: str) -> str:
        """
        Call the configured LLM.

        Args:
            prompt: Prompt to send

        Returns:
            LLM response text
        """
        model = self.config.model

        # Route to appropriate API
        if "claude" in model.lower() or "anthropic" in model.lower():
            return self._call_anthropic(prompt)
        elif "gpt" in model.lower() or "openai" in model.lower():
            return self._call_openai(prompt)
        else:
            # Default to Anthropic
            return self._call_anthropic(prompt)

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        client = self.anthropic_client
        if client is None:
            raise RuntimeError("Anthropic client not available")

        response = client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        # Track costs
        if self.config.track_costs:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            self.cost_tracker.add_request(input_tokens, output_tokens)

        return response.content[0].text

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        client = self.openai_client
        if client is None:
            raise RuntimeError("OpenAI client not available")

        response = client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        # Track costs
        if self.config.track_costs:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            self.cost_tracker.add_request(input_tokens, output_tokens)

        return response.choices[0].message.content

    def _get_judge_model(self) -> JudgeModel:
        """Get JudgeModel enum from config model string."""
        model = self.config.model.lower()

        if "opus" in model:
            return JudgeModel.CLAUDE_OPUS
        elif "sonnet" in model:
            return JudgeModel.CLAUDE_SONNET
        elif "gpt-4-turbo" in model:
            return JudgeModel.GPT4_TURBO
        elif "gpt-4" in model:
            return JudgeModel.GPT4
        elif "gemini" in model:
            return JudgeModel.GEMINI_PRO
        else:
            return JudgeModel.CLAUDE_SONNET

    def _get_cache_key(self, decision: AgentDecision, rubric: EvaluationRubric) -> str:
        """Generate cache key for decision+rubric combination."""
        return f"{decision.decision_id}_{rubric.category.value}_{self.config.model}"

    def _generate_detailed_feedback(self, scores: list[JudgeScore], overall_score: float) -> str:
        """Generate detailed feedback report."""
        parts = [
            "## LLM Judge Evaluation Report",
            "",
            f"**Overall Score**: {overall_score:.2f}/5.0",
            f"**Judge Model**: {self.config.model}",
            f"**Number of Evaluations**: {len(scores)}",
            "",
        ]

        for score in scores:
            parts.append(f"### {score.category.value.replace('_', ' ').title()}")
            parts.append(f"**Score**: {score.score}/5.0 (Confidence: {score.confidence:.2f})")
            parts.append(f"**Reasoning**: {score.reasoning[:500]}...")

            if score.issues_found:
                parts.append(f"**Issues**: {', '.join(score.issues_found[:5])}")
            if score.strengths_found:
                parts.append(f"**Strengths**: {', '.join(score.strengths_found[:5])}")
            parts.append("")

        return "\n".join(parts)

    def get_cost_summary(self) -> dict[str, Any]:
        """Get cost tracking summary."""
        return self.cost_tracker.get_summary()

    def clear_cache(self) -> None:
        """Clear evaluation cache."""
        self._cache.clear()


def create_production_judge(
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.0,
    enable_caching: bool = True,
) -> LLMJudge:
    """
    Factory function to create a production LLM judge.

    Args:
        model: LLM model to use
        temperature: Temperature for generation (0.0 for deterministic)
        enable_caching: Whether to cache evaluations

    Returns:
        Configured LLMJudge instance
    """
    config = JudgeConfig(
        model=model,
        temperature=temperature,
        enable_caching=enable_caching,
    )
    return LLMJudge(config=config)


def create_judge_function(
    judge: LLMJudge,
    category: EvaluationCategory = EvaluationCategory.TRADING_DECISION,
) -> Callable[[AgentDecision], JudgeScore]:
    """
    Create a judge function compatible with evaluate_agent_decision().

    Args:
        judge: LLMJudge instance
        category: Category to evaluate

    Returns:
        Callable that takes AgentDecision and returns JudgeScore
    """
    rubric = ALL_RUBRICS.get(category)

    def judge_fn(decision: AgentDecision) -> JudgeScore:
        return judge.evaluate(decision, rubric, category)

    return judge_fn


# Export all public API
__all__ = [
    "JudgeConfig",
    "JudgeCostTracker",
    "LLMJudge",
    "create_judge_function",
    "create_production_judge",
]
