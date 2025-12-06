"""
Tests for Evaluator-Optimizer Feedback Loop

Tests verify the feedback loop correctly:
- Evaluates agents against test cases
- Identifies weaknesses from scores
- Generates appropriate prompt refinements
- Tracks convergence and improvement
"""

from dataclasses import dataclass
from typing import Any

import pytest

from evaluation.feedback_loop import (
    ConvergenceReason,
    EvaluatorOptimizerLoop,
    FeedbackCycle,
    PromptRefinement,
    Weakness,
    WeaknessCategory,
    create_feedback_loop,
    generate_feedback_report,
)


@dataclass
class MockTestCase:
    """Mock test case for testing."""

    id: str
    category: str
    expected: str
    actual: str = ""


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.system_prompt = system_prompt
        self.name = "mock_agent"

    def analyze(self, query: str, context: dict) -> dict:
        """Mock analysis."""
        return {"answer": "test response", "confidence": 0.7}


class MockEvaluator:
    """Mock evaluator that returns configurable scores."""

    def __init__(self, scores: dict[str, float], improve_rate: float = 0.1):
        self.base_scores = scores.copy()
        self.current_scores = scores.copy()
        self.improve_rate = improve_rate
        self.call_count = 0

    def evaluate(self, agent: Any, test_cases: list) -> Any:
        """Return scores, improving each call."""
        self.call_count += 1

        # Simulate improvement with each evaluation
        if self.call_count > 1:
            for key in self.current_scores:
                self.current_scores[key] = min(1.0, self.current_scores[key] + self.improve_rate)

        @dataclass
        class Result:
            scores: dict[str, float]

        return Result(scores=self.current_scores.copy())


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    return MockAgent()


@pytest.fixture
def mock_test_cases():
    """Create mock test cases."""
    return [
        MockTestCase(id="tc1", category="accuracy", expected="A", actual="A"),
        MockTestCase(id="tc2", category="accuracy", expected="B", actual="C"),
        MockTestCase(id="tc3", category="reasoning", expected="X", actual="X"),
        MockTestCase(id="tc4", category="reasoning", expected="Y", actual="Z"),
        MockTestCase(id="tc5", category="consistency", expected="P", actual="P"),
    ]


class TestFeedbackLoopCreation:
    """Tests for feedback loop creation."""

    def test_create_with_defaults(self):
        """Loop can be created with default settings."""
        loop = EvaluatorOptimizerLoop()

        assert loop.max_iterations == 5
        assert loop.target_score == 0.80
        assert loop.convergence_threshold == 0.02

    def test_create_with_custom_settings(self):
        """Loop can be created with custom settings."""
        loop = EvaluatorOptimizerLoop(
            max_iterations=10,
            target_score=0.90,
            convergence_threshold=0.01,
        )

        assert loop.max_iterations == 10
        assert loop.target_score == 0.90
        assert loop.convergence_threshold == 0.01

    def test_factory_function(self):
        """Factory function creates loop correctly."""
        loop = create_feedback_loop(
            max_iterations=3,
            target_score=0.75,
        )

        assert loop.max_iterations == 3
        assert loop.target_score == 0.75


class TestEvaluation:
    """Tests for agent evaluation."""

    def test_evaluate_with_custom_function(self, mock_agent, mock_test_cases):
        """Loop can use custom evaluation function."""

        def custom_eval(agent, cases, context):
            return {"accuracy": 0.7, "reasoning": 0.6}

        loop = EvaluatorOptimizerLoop(evaluate_fn=custom_eval)
        scores = loop._evaluate(mock_agent, mock_test_cases, {})

        assert scores["accuracy"] == 0.7
        assert scores["reasoning"] == 0.6

    def test_evaluate_with_evaluator(self, mock_agent, mock_test_cases):
        """Loop can use provided evaluator."""
        evaluator = MockEvaluator({"accuracy": 0.8, "reasoning": 0.7})
        loop = EvaluatorOptimizerLoop(evaluator=evaluator)

        scores = loop._evaluate(mock_agent, mock_test_cases, {})

        assert scores["accuracy"] == 0.8
        assert scores["reasoning"] == 0.7

    def test_aggregate_scores_unweighted(self):
        """Aggregate scores calculates mean without weights."""
        loop = EvaluatorOptimizerLoop()
        scores = {"accuracy": 0.8, "reasoning": 0.6, "consistency": 0.7}

        aggregate = loop._aggregate_scores(scores)

        assert aggregate == pytest.approx(0.7, abs=0.01)

    def test_aggregate_scores_weighted(self):
        """Aggregate scores respects weights."""
        loop = EvaluatorOptimizerLoop(score_weights={"accuracy": 2.0, "reasoning": 1.0})
        scores = {"accuracy": 0.8, "reasoning": 0.6}

        # Weighted: (0.8 * 2 + 0.6 * 1) / 3 = 2.2 / 3 = 0.733
        aggregate = loop._aggregate_scores(scores)

        assert aggregate == pytest.approx(0.733, abs=0.01)


class TestWeaknessIdentification:
    """Tests for weakness identification."""

    def test_identify_low_scores(self, mock_test_cases):
        """Identifies weaknesses from low scores."""
        loop = EvaluatorOptimizerLoop()
        scores = {"accuracy": 0.5, "reasoning": 0.85}

        weaknesses = loop._identify_weaknesses(scores, mock_test_cases)

        assert len(weaknesses) == 1
        assert weaknesses[0].category == WeaknessCategory.ACCURACY
        assert weaknesses[0].severity == pytest.approx(0.5, abs=0.01)

    def test_identify_multiple_weaknesses(self, mock_test_cases):
        """Identifies multiple weak categories."""
        loop = EvaluatorOptimizerLoop()
        scores = {"accuracy": 0.4, "reasoning": 0.5, "consistency": 0.3}

        weaknesses = loop._identify_weaknesses(scores, mock_test_cases)

        assert len(weaknesses) == 3
        # Should be sorted by severity (lowest score = highest severity)
        assert weaknesses[0].severity > weaknesses[2].severity

    def test_no_weaknesses_when_all_high(self, mock_test_cases):
        """No weaknesses identified when all scores are high."""
        loop = EvaluatorOptimizerLoop()
        scores = {"accuracy": 0.9, "reasoning": 0.85, "consistency": 0.8}

        weaknesses = loop._identify_weaknesses(scores, mock_test_cases)

        assert len(weaknesses) == 0

    def test_weakness_includes_suggested_fix(self, mock_test_cases):
        """Weaknesses include suggested fixes."""
        loop = EvaluatorOptimizerLoop()
        scores = {"accuracy": 0.4}

        weaknesses = loop._identify_weaknesses(scores, mock_test_cases)

        assert len(weaknesses) == 1
        assert weaknesses[0].suggested_fix is not None
        assert len(weaknesses[0].suggested_fix) > 0


class TestRefinementGeneration:
    """Tests for prompt refinement generation."""

    def test_generate_refinements_for_weaknesses(self, mock_agent):
        """Generates refinements for identified weaknesses."""
        loop = EvaluatorOptimizerLoop()
        weaknesses = [
            Weakness(
                category=WeaknessCategory.ACCURACY,
                description="Low accuracy",
                severity=0.5,
                affected_test_cases=["tc1"],
                suggested_fix="Improve accuracy",
            )
        ]

        refinements = loop._generate_refinements(weaknesses, mock_agent)

        assert len(refinements) == 1
        assert refinements[0].weakness_addressed == WeaknessCategory.ACCURACY
        assert "accuracy" in refinements[0].refined_text.lower()

    def test_limits_refinements_to_top_weaknesses(self, mock_agent):
        """Limits refinements to top 3 weaknesses."""
        loop = EvaluatorOptimizerLoop()
        weaknesses = [
            Weakness(
                category=WeaknessCategory.ACCURACY,
                description="Test",
                severity=0.8,
                affected_test_cases=[],
                suggested_fix="Fix",
            )
            for _ in range(5)
        ]

        refinements = loop._generate_refinements(weaknesses, mock_agent)

        assert len(refinements) <= 3

    def test_apply_refinements_updates_prompt(self, mock_agent):
        """Applying refinements updates agent's system prompt."""
        loop = EvaluatorOptimizerLoop()
        original_prompt = mock_agent.system_prompt

        refinements = [
            PromptRefinement(
                target_section="system_prompt",
                original_text=original_prompt,
                refined_text=original_prompt + "\n\nNew instruction.",
                weakness_addressed=WeaknessCategory.ACCURACY,
                confidence=0.8,
            )
        ]

        loop._apply_refinements(mock_agent, refinements)

        assert "New instruction" in mock_agent.system_prompt


class TestFeedbackLoopExecution:
    """Tests for full feedback loop execution."""

    def test_converges_when_target_reached(self, mock_agent, mock_test_cases):
        """Loop converges when target score is reached."""
        # Start above target
        evaluator = MockEvaluator({"accuracy": 0.9, "reasoning": 0.85})
        loop = EvaluatorOptimizerLoop(
            evaluator=evaluator,
            target_score=0.80,
        )

        result = loop.run(mock_agent, mock_test_cases)

        assert result.converged
        assert result.reason == ConvergenceReason.TARGET_REACHED
        assert result.iterations_completed == 0

    def test_converges_on_score_plateau(self, mock_agent, mock_test_cases):
        """Loop converges when scores stop improving."""
        # Very slow improvement
        evaluator = MockEvaluator(
            {"accuracy": 0.75, "reasoning": 0.75},
            improve_rate=0.005,  # Very slow
        )
        loop = EvaluatorOptimizerLoop(
            evaluator=evaluator,
            target_score=0.95,  # Unreachable
            convergence_threshold=0.02,
        )

        result = loop.run(mock_agent, mock_test_cases)

        assert result.converged
        assert result.reason == ConvergenceReason.SCORE_CONVERGED

    def test_stops_after_max_iterations(self, mock_agent, mock_test_cases):
        """Loop stops after max iterations."""
        # Keep improving but slowly
        evaluator = MockEvaluator(
            {"accuracy": 0.3, "reasoning": 0.3},
            improve_rate=0.05,
        )
        loop = EvaluatorOptimizerLoop(
            evaluator=evaluator,
            target_score=0.99,  # Hard to reach
            max_iterations=3,
            convergence_threshold=0.001,  # Won't trigger
        )

        result = loop.run(mock_agent, mock_test_cases)

        assert not result.converged
        assert result.reason == ConvergenceReason.MAX_ITERATIONS
        assert result.iterations_completed == 3

    def test_tracks_improvement(self, mock_agent, mock_test_cases):
        """Loop tracks improvement over iterations."""
        evaluator = MockEvaluator(
            {"accuracy": 0.5, "reasoning": 0.5},
            improve_rate=0.15,
        )
        loop = EvaluatorOptimizerLoop(
            evaluator=evaluator,
            target_score=0.85,
            max_iterations=5,
        )

        result = loop.run(mock_agent, mock_test_cases)

        assert result.total_improvement > 0
        assert result.final_aggregate > result.initial_aggregate

    def test_records_cycle_history(self, mock_agent, mock_test_cases):
        """Loop records cycle history."""
        evaluator = MockEvaluator(
            {"accuracy": 0.6, "reasoning": 0.6},
            improve_rate=0.1,
        )
        loop = EvaluatorOptimizerLoop(
            evaluator=evaluator,
            target_score=0.90,
            max_iterations=3,
        )

        result = loop.run(mock_agent, mock_test_cases)

        assert len(result.cycles) > 0
        for cycle in result.cycles:
            assert isinstance(cycle, FeedbackCycle)
            assert cycle.iteration >= 0

    def test_result_to_dict(self, mock_agent, mock_test_cases):
        """Result can be converted to dictionary."""
        evaluator = MockEvaluator({"accuracy": 0.9})
        loop = EvaluatorOptimizerLoop(evaluator=evaluator)

        result = loop.run(mock_agent, mock_test_cases)
        result_dict = result.to_dict()

        assert "converged" in result_dict
        assert "reason" in result_dict
        assert "total_improvement" in result_dict


class TestFeedbackLoopReset:
    """Tests for loop reset functionality."""

    def test_reset_clears_history(self, mock_agent, mock_test_cases):
        """Reset clears loop history."""
        evaluator = MockEvaluator({"accuracy": 0.6})
        loop = EvaluatorOptimizerLoop(evaluator=evaluator, max_iterations=2)

        # Run once
        loop.run(mock_agent, mock_test_cases)
        assert len(loop.history) > 0

        # Reset
        loop.reset()
        assert len(loop.history) == 0


class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_report(self, mock_agent, mock_test_cases):
        """Generates readable report from result."""
        evaluator = MockEvaluator({"accuracy": 0.7, "reasoning": 0.6})
        loop = EvaluatorOptimizerLoop(evaluator=evaluator, max_iterations=2)

        result = loop.run(mock_agent, mock_test_cases)
        report = generate_feedback_report(result)

        assert "FEEDBACK LOOP REPORT" in report
        assert "SCORE IMPROVEMENT" in report
        assert "Initial Score" in report
        assert "Final Score" in report

    def test_report_includes_category_breakdown(self, mock_agent, mock_test_cases):
        """Report includes category breakdown."""
        evaluator = MockEvaluator({"accuracy": 0.7, "reasoning": 0.6})
        loop = EvaluatorOptimizerLoop(evaluator=evaluator, max_iterations=1)

        result = loop.run(mock_agent, mock_test_cases)
        report = generate_feedback_report(result)

        assert "CATEGORY BREAKDOWN" in report
        assert "accuracy" in report
        assert "reasoning" in report


class TestCustomEvaluationFunction:
    """Tests for custom evaluation functions."""

    def test_with_custom_eval_function(self, mock_agent, mock_test_cases):
        """Loop works with custom evaluation function."""
        call_count = [0]

        def custom_eval(agent, cases, context):
            call_count[0] += 1
            # Start low and improve each call
            base = 0.3 + (call_count[0] * 0.1)
            return {"accuracy": min(base, 1.0)}

        loop = EvaluatorOptimizerLoop(
            evaluate_fn=custom_eval,
            target_score=0.85,
            max_iterations=5,
            convergence_threshold=0.01,  # Small threshold to force iterations
        )

        result = loop.run(mock_agent, mock_test_cases)

        assert call_count[0] >= 1
        # The loop should make progress
        assert result.final_aggregate >= result.initial_aggregate


class TestNoImprovementPatience:
    """Tests for no-improvement patience."""

    def test_stops_when_no_improvement(self, mock_agent, mock_test_cases):
        """Stops after patience iterations without improvement."""
        # No improvement
        evaluator = MockEvaluator({"accuracy": 0.6}, improve_rate=0.0)
        loop = EvaluatorOptimizerLoop(
            evaluator=evaluator,
            target_score=0.95,
            max_iterations=10,
            no_improvement_patience=2,
        )

        result = loop.run(mock_agent, mock_test_cases)

        assert not result.converged
        assert result.reason == ConvergenceReason.NO_IMPROVEMENT
        assert result.iterations_completed <= 4  # Should stop early
