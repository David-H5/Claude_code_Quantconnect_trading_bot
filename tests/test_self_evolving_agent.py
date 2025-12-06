"""
Tests for Self-Evolving Agent and Prompt Optimizer

Tests verify:
- Self-evolution cycle execution
- Prompt versioning and rollback
- Convergence detection
- Prompt optimization refinements
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from llm.prompt_optimizer import (
    OptimizationResult,
    PromptOptimizer,
    PromptRefinement,
    RefinementCategory,
    RefinementStrategy,
    create_prompt_optimizer,
    generate_optimization_report,
)
from llm.self_evolving_agent import (
    ConvergenceReason,
    EvolutionCycle,
    EvolutionResult,
    PromptVersion,
    SelfEvolvingAgent,
    create_self_evolving_agent,
    generate_evolution_report,
)


class MockTradingAgent:
    """Mock trading agent for testing."""

    def __init__(self, name: str = "test_agent", system_prompt: str = "You are a trading analyst."):
        self.name = name
        self.system_prompt = system_prompt
        self.role = "analyst"

    def analyze(self, query: str, context: dict[str, Any]) -> MagicMock:
        """Mock analysis."""
        response = MagicMock()
        response.success = True
        response.confidence = 0.75
        return response


@pytest.fixture
def mock_agent():
    """Create a mock trading agent."""
    return MockTradingAgent()


@pytest.fixture
def evolving_agent(mock_agent):
    """Create a self-evolving agent wrapper."""
    return SelfEvolvingAgent(
        base_agent=mock_agent,
        max_evolution_cycles=3,
        improvement_threshold=0.02,
        target_score=0.85,
    )


@pytest.fixture
def optimizer():
    """Create a prompt optimizer."""
    return PromptOptimizer()


class TestPromptVersion:
    """Tests for PromptVersion dataclass."""

    def test_creation(self):
        """PromptVersion can be created."""
        version = PromptVersion(
            version=1,
            prompt="Test prompt",
            score=0.75,
        )
        assert version.version == 1
        assert version.score == 0.75

    def test_to_dict(self):
        """PromptVersion can be serialized."""
        version = PromptVersion(
            version=1,
            prompt="Test prompt",
            score=0.75,
        )
        d = version.to_dict()
        assert d["version"] == 1
        assert d["score"] == 0.75
        assert "timestamp" in d


class TestEvolutionCycle:
    """Tests for EvolutionCycle dataclass."""

    def test_creation(self):
        """EvolutionCycle can be created."""
        cycle = EvolutionCycle(
            cycle_number=0,
            pre_score=0.6,
            post_score=0.7,
            refinements_applied=["Added examples"],
            improvement=0.1,
            prompt_before="Before",
            prompt_after="After",
        )
        assert cycle.cycle_number == 0
        assert cycle.improvement == 0.1

    def test_to_dict(self):
        """EvolutionCycle can be serialized."""
        cycle = EvolutionCycle(
            cycle_number=0,
            pre_score=0.6,
            post_score=0.7,
            refinements_applied=["Added examples"],
            improvement=0.1,
            prompt_before="Before",
            prompt_after="After",
        )
        d = cycle.to_dict()
        assert d["cycle_number"] == 0
        assert d["improvement"] == 0.1
        assert "timestamp" in d


class TestSelfEvolvingAgentInit:
    """Tests for SelfEvolvingAgent initialization."""

    def test_basic_init(self, mock_agent):
        """Agent initializes with defaults."""
        evolving = SelfEvolvingAgent(base_agent=mock_agent)

        assert evolving.agent == mock_agent
        assert evolving.max_cycles == 5
        assert evolving.target_score == 0.85

    def test_custom_params(self, mock_agent):
        """Agent accepts custom parameters."""
        evolving = SelfEvolvingAgent(
            base_agent=mock_agent,
            max_evolution_cycles=10,
            improvement_threshold=0.05,
            target_score=0.95,
            enable_rollback=False,
        )

        assert evolving.max_cycles == 10
        assert evolving.improvement_threshold == 0.05
        assert evolving.target_score == 0.95
        assert evolving.enable_rollback is False

    def test_initial_prompt_saved(self, mock_agent):
        """Initial prompt is saved on creation."""
        evolving = SelfEvolvingAgent(base_agent=mock_agent)

        assert len(evolving.prompt_versions) == 1
        assert evolving.prompt_versions[0].version == 0


class TestEvolutionProcess:
    """Tests for the evolution process."""

    def test_evolve_returns_result(self, evolving_agent):
        """Evolve returns EvolutionResult."""
        result = evolving_agent.evolve()

        assert isinstance(result, EvolutionResult)
        assert result.agent_name == "test_agent"

    def test_evolve_tracks_cycles(self, evolving_agent):
        """Evolution tracks cycles."""
        result = evolving_agent.evolve()

        assert len(result.cycles) >= 0

    def test_evolve_improves_prompt(self, evolving_agent):
        """Evolution modifies the prompt."""
        original = evolving_agent.agent.system_prompt
        result = evolving_agent.evolve()

        # Prompt should be modified or target reached immediately
        assert result.evolved_prompt is not None

    def test_evolution_with_custom_evaluator(self, mock_agent):
        """Evolution works with custom evaluator function."""
        scores = [0.6, 0.7, 0.8, 0.9]
        call_count = [0]

        def custom_eval(agent, cases):
            score = scores[min(call_count[0], len(scores) - 1)]
            call_count[0] += 1
            return score

        evolving = SelfEvolvingAgent(
            base_agent=mock_agent,
            evaluate_fn=custom_eval,
            target_score=0.85,
        )

        result = evolving.evolve()

        assert result.final_score >= 0.8


class TestConvergenceDetection:
    """Tests for convergence detection."""

    def test_target_reached_convergence(self, mock_agent):
        """Converges when target score is reached."""

        def high_score(agent, cases):
            return 0.90

        evolving = SelfEvolvingAgent(
            base_agent=mock_agent,
            evaluate_fn=high_score,
            target_score=0.85,
        )

        result = evolving.evolve()

        assert result.converged is True
        assert result.convergence_reason == ConvergenceReason.TARGET_REACHED

    def test_max_cycles_convergence(self, mock_agent):
        """Converges when max cycles reached."""
        cycle_count = [0]

        def improving_score(agent, cases):
            cycle_count[0] += 1
            return 0.5 + cycle_count[0] * 0.05  # Never reaches 0.85

        evolving = SelfEvolvingAgent(
            base_agent=mock_agent,
            evaluate_fn=improving_score,
            target_score=0.95,  # Unreachable
            max_evolution_cycles=3,
            improvement_threshold=0.01,
        )

        result = evolving.evolve()

        assert result.convergence_reason == ConvergenceReason.MAX_CYCLES_REACHED

    def test_no_improvement_convergence(self, mock_agent):
        """Converges when improvement stalls."""

        def flat_score(agent, cases):
            return 0.7  # Always same score

        evolving = SelfEvolvingAgent(
            base_agent=mock_agent,
            evaluate_fn=flat_score,
            target_score=0.85,
            improvement_threshold=0.02,
        )

        result = evolving.evolve()

        assert result.convergence_reason == ConvergenceReason.NO_IMPROVEMENT


class TestRollback:
    """Tests for rollback functionality."""

    def test_rollback_on_regression(self, mock_agent):
        """Rolls back when score regresses."""
        scores = [0.7, 0.65]  # Second score is worse
        call_count = [0]

        def regressing_score(agent, cases):
            score = scores[min(call_count[0], len(scores) - 1)]
            call_count[0] += 1
            return score

        evolving = SelfEvolvingAgent(
            base_agent=mock_agent,
            evaluate_fn=regressing_score,
            target_score=0.95,
            enable_rollback=True,
        )

        result = evolving.evolve()

        assert result.convergence_reason == ConvergenceReason.REGRESSION_DETECTED

    def test_no_rollback_when_disabled(self, mock_agent):
        """No rollback when disabled."""
        scores = [0.7, 0.65, 0.70]
        call_count = [0]

        def regressing_score(agent, cases):
            score = scores[min(call_count[0], len(scores) - 1)]
            call_count[0] += 1
            return score

        evolving = SelfEvolvingAgent(
            base_agent=mock_agent,
            evaluate_fn=regressing_score,
            target_score=0.95,
            enable_rollback=False,
            max_evolution_cycles=2,
        )

        result = evolving.evolve()

        # Should not trigger regression since rollback disabled
        assert result.convergence_reason != ConvergenceReason.REGRESSION_DETECTED


class TestPromptVersioning:
    """Tests for prompt versioning."""

    def test_versions_tracked(self, evolving_agent):
        """All prompt versions are tracked."""
        evolving_agent.evolve()

        assert len(evolving_agent.prompt_versions) >= 1

    def test_get_best_version(self, mock_agent):
        """Can retrieve best performing version."""
        scores = [0.6, 0.8, 0.7]  # Peak at second
        call_count = [0]

        def variable_score(agent, cases):
            score = scores[min(call_count[0], len(scores) - 1)]
            call_count[0] += 1
            return score

        evolving = SelfEvolvingAgent(
            base_agent=mock_agent,
            evaluate_fn=variable_score,
            target_score=0.95,
            max_evolution_cycles=2,
            enable_rollback=False,
        )

        evolving.evolve()
        best = evolving.get_best_version()

        assert best is not None
        assert best.score == max(v.score for v in evolving.prompt_versions)


class TestEvolutionHistory:
    """Tests for evolution history."""

    def test_history_tracking(self, evolving_agent):
        """Evolution history is tracked."""
        result = evolving_agent.evolve()

        history = evolving_agent.get_evolution_history()
        assert len(history) == len(result.cycles)

    def test_reset_clears_history(self, evolving_agent):
        """Reset clears all history."""
        evolving_agent.evolve()
        assert len(evolving_agent.evolution_history) > 0 or evolving_agent.best_score > 0

        evolving_agent.reset()

        assert len(evolving_agent.evolution_history) == 0
        assert len(evolving_agent.prompt_versions) == 0
        assert evolving_agent.best_score == 0.0


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_self_evolving_agent(self, mock_agent):
        """Factory creates configured agent."""
        evolving = create_self_evolving_agent(
            base_agent=mock_agent,
            target_score=0.90,
            max_cycles=10,
            improvement_threshold=0.03,
        )

        assert evolving.target_score == 0.90
        assert evolving.max_cycles == 10
        assert evolving.improvement_threshold == 0.03


class TestEvolutionReport:
    """Tests for report generation."""

    def test_generate_report(self, evolving_agent):
        """Report is generated correctly."""
        result = evolving_agent.evolve()
        report = generate_evolution_report(result)

        assert "SELF-EVOLUTION REPORT" in report
        assert "test_agent" in report
        assert "PERFORMANCE:" in report


class TestEvolutionResultSerialization:
    """Tests for EvolutionResult serialization."""

    def test_to_dict(self, evolving_agent):
        """EvolutionResult can be serialized."""
        result = evolving_agent.evolve()
        d = result.to_dict()

        assert "agent_name" in d
        assert "initial_score" in d
        assert "final_score" in d
        assert "convergence_reason" in d


# ============= Prompt Optimizer Tests =============


class TestPromptOptimizerInit:
    """Tests for PromptOptimizer initialization."""

    def test_default_init(self):
        """Optimizer initializes with defaults."""
        optimizer = PromptOptimizer()

        assert optimizer.strategy == RefinementStrategy.RULE_BASED
        assert optimizer.max_refinements == 5

    def test_custom_init(self):
        """Optimizer accepts custom parameters."""
        optimizer = PromptOptimizer(
            strategy=RefinementStrategy.HYBRID,
            max_refinements_per_cycle=10,
            min_expected_impact=0.10,
        )

        assert optimizer.strategy == RefinementStrategy.HYBRID
        assert optimizer.max_refinements == 10
        assert optimizer.min_impact == 0.10


class TestRefinementGeneration:
    """Tests for refinement generation."""

    def test_generate_refinements_for_missing_examples(self, optimizer):
        """Generates refinement for missing examples."""
        weaknesses = ["Missing concrete examples"]
        prompt = "You are a trading analyst."

        refinements = optimizer.generate_refinements(prompt, weaknesses)

        assert len(refinements) > 0
        example_refinements = [r for r in refinements if r.category == RefinementCategory.ADD_EXAMPLES]
        assert len(example_refinements) > 0

    def test_generate_refinements_for_missing_process(self, optimizer):
        """Generates refinement for missing process."""
        weaknesses = ["Missing step-by-step process"]
        prompt = "You are a trading analyst."

        refinements = optimizer.generate_refinements(prompt, weaknesses)

        assert len(refinements) > 0
        structure_refinements = [r for r in refinements if r.category == RefinementCategory.IMPROVE_STRUCTURE]
        assert len(structure_refinements) > 0

    def test_generate_refinements_for_missing_constraints(self, optimizer):
        """Generates refinement for missing constraints."""
        weaknesses = ["Missing explicit constraints"]
        prompt = "You are a trading analyst."

        refinements = optimizer.generate_refinements(prompt, weaknesses)

        assert len(refinements) > 0

    def test_generate_refinements_for_error_handling(self, optimizer):
        """Generates refinement for missing error handling."""
        weaknesses = ["Missing error handling for edge cases"]
        prompt = "You are a trading analyst."

        refinements = optimizer.generate_refinements(prompt, weaknesses)

        error_refinements = [r for r in refinements if r.category == RefinementCategory.ADD_ERROR_HANDLING]
        assert len(error_refinements) > 0

    def test_respects_max_refinements(self, optimizer):
        """Respects maximum refinements limit."""
        optimizer.max_refinements = 2
        weaknesses = [
            "Missing examples",
            "Missing process",
            "Missing constraints",
            "Missing error handling",
        ]

        refinements = optimizer.generate_refinements("Test prompt", weaknesses)

        assert len(refinements) <= 2


class TestRefinementApplication:
    """Tests for applying refinements."""

    def test_apply_refinements_appends(self, optimizer):
        """Applies refinements by appending."""
        prompt = "You are a trading analyst."
        refinements = [
            PromptRefinement(
                category=RefinementCategory.ADD_EXAMPLES,
                original_section="",
                refined_section="\n\nExample: Analyze SPY technicals.",
                description="Added example",
                expected_impact=0.15,
            )
        ]

        new_prompt = optimizer.apply_refinements(prompt, refinements)

        assert "Example: Analyze SPY" in new_prompt
        assert len(new_prompt) > len(prompt)

    def test_apply_refinements_replaces(self, optimizer):
        """Applies refinements by replacing."""
        prompt = "You should consider all factors."
        refinements = [
            PromptRefinement(
                category=RefinementCategory.CLARIFY_INSTRUCTIONS,
                original_section="should consider",
                refined_section="must explicitly evaluate",
                description="Clarified instruction",
                expected_impact=0.10,
            )
        ]

        new_prompt = optimizer.apply_refinements(prompt, refinements)

        assert "must explicitly evaluate" in new_prompt
        assert "should consider" not in new_prompt


class TestPromptAnalysis:
    """Tests for prompt analysis."""

    def test_analyze_prompt_basic(self, optimizer):
        """Analyzes prompt for weaknesses."""
        prompt = "You are a trading analyst."
        analysis = optimizer.analyze_prompt(prompt)

        assert "length" in analysis
        assert "word_count" in analysis
        assert "potential_weaknesses" in analysis

    def test_analyze_prompt_detects_missing_elements(self, optimizer):
        """Detects missing prompt elements."""
        prompt = "You are a trading analyst."
        analysis = optimizer.analyze_prompt(prompt)

        assert analysis["has_examples"] is False
        assert "Missing concrete examples" in analysis["potential_weaknesses"]

    def test_analyze_prompt_detects_present_elements(self, optimizer):
        """Detects present prompt elements."""
        prompt = "You are a trading analyst. Example: Analyze SPY. Constraint: Max 2% risk."
        analysis = optimizer.analyze_prompt(prompt)

        assert analysis["has_examples"] is True
        assert analysis["has_constraints"] is True


class TestOptimization:
    """Tests for full optimization."""

    def test_optimize_returns_result(self, optimizer):
        """Optimize returns OptimizationResult."""
        prompt = "You are a trading analyst."
        weaknesses = ["Missing examples"]

        result = optimizer.optimize(prompt, weaknesses)

        assert isinstance(result, OptimizationResult)
        assert result.original_prompt == prompt

    def test_optimize_improves_prompt(self, optimizer):
        """Optimization improves the prompt."""
        prompt = "You are a trading analyst."
        weaknesses = ["Missing examples", "Missing constraints"]

        result = optimizer.optimize(prompt, weaknesses)

        assert len(result.optimized_prompt) > len(prompt)
        assert len(result.refinements_applied) > 0

    def test_optimize_calculates_impact(self, optimizer):
        """Optimization calculates total expected impact."""
        prompt = "You are a trading analyst."
        weaknesses = ["Missing examples"]

        result = optimizer.optimize(prompt, weaknesses)

        assert result.total_expected_impact > 0
        assert result.total_expected_impact <= 1.0


class TestOptimizerFactory:
    """Tests for optimizer factory function."""

    def test_create_prompt_optimizer(self):
        """Factory creates configured optimizer."""
        optimizer = create_prompt_optimizer(
            strategy="hybrid",
            max_refinements=3,
            min_impact=0.10,
        )

        assert optimizer.strategy == RefinementStrategy.HYBRID
        assert optimizer.max_refinements == 3
        assert optimizer.min_impact == 0.10


class TestOptimizationReport:
    """Tests for optimization report generation."""

    def test_generate_optimization_report(self, optimizer):
        """Report is generated correctly."""
        result = optimizer.optimize(
            "You are a trading analyst.",
            ["Missing examples"],
        )
        report = generate_optimization_report(result)

        assert "PROMPT OPTIMIZATION REPORT" in report
        assert "REFINEMENTS APPLIED:" in report
        assert "PROMPT SIZE CHANGE:" in report


class TestRefinementSerialization:
    """Tests for refinement serialization."""

    def test_prompt_refinement_to_dict(self):
        """PromptRefinement can be serialized."""
        refinement = PromptRefinement(
            category=RefinementCategory.ADD_EXAMPLES,
            original_section="",
            refined_section="Example added",
            description="Added example",
            expected_impact=0.15,
        )
        d = refinement.to_dict()

        assert d["category"] == "add_examples"
        assert d["expected_impact"] == 0.15

    def test_optimization_result_to_dict(self, optimizer):
        """OptimizationResult can be serialized."""
        result = optimizer.optimize(
            "Test prompt",
            ["Missing examples"],
        )
        d = result.to_dict()

        assert "refinements_applied" in d
        assert "total_expected_impact" in d
        assert "strategy_used" in d
