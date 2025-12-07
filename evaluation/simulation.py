"""
Simulation-Based Testing Framework for AI Agents.

Enables comprehensive testing of agents through:
- User simulation with configurable behaviors
- Sandboxed execution environments
- LLM-as-a-Judge evaluation
- Cross-environment validation

UPGRADE-014 Category 8: Testing & Simulation

Usage:
    from evaluation.simulation import UserSimulator, LLMJudge, create_simulator

    # Create simulator
    simulator = create_simulator(behavior="novice")
    scenario = simulator.generate_scenario("trading")

    # Evaluate with LLM judge
    judge = LLMJudge()
    score = judge.evaluate(agent_response, criteria=["accuracy", "clarity"])
"""

import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar


class UserBehavior(Enum):
    """Simulated user behavior types."""

    NOVICE = "novice"  # Simple questions, needs guidance
    INTERMEDIATE = "intermediate"  # Moderate complexity
    EXPERT = "expert"  # Complex queries, edge cases
    ADVERSARIAL = "adversarial"  # Tests edge cases, malformed inputs
    RANDOM = "random"  # Random mix of behaviors


class EvaluationCriteria(Enum):
    """Criteria for LLM-as-a-Judge evaluation."""

    ACCURACY = "accuracy"  # Factual correctness
    CLARITY = "clarity"  # Response clarity and readability
    COMPLETENESS = "completeness"  # Covers all aspects of query
    SAFETY = "safety"  # No harmful content
    RELEVANCE = "relevance"  # On-topic response
    ACTIONABILITY = "actionability"  # Provides actionable advice
    CONSISTENCY = "consistency"  # Consistent with prior responses


@dataclass
class Scenario:
    """A test scenario for simulation."""

    id: str
    name: str
    description: str
    category: str  # trading, analysis, general
    difficulty: str  # easy, medium, hard
    user_query: str
    context: dict[str, Any] = field(default_factory=dict)
    expected_behavior: str = ""
    success_criteria: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Result of an LLM-as-a-Judge evaluation."""

    scenario_id: str
    agent_name: str
    response: str
    scores: dict[str, float]  # criteria -> score (0-1)
    overall_score: float
    feedback: str
    passed: bool
    evaluation_time: float
    judge_model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationRun:
    """Complete simulation run with multiple scenarios."""

    run_id: str
    agent_name: str
    started_at: datetime
    completed_at: datetime | None = None
    scenarios_total: int = 0
    scenarios_passed: int = 0
    average_score: float = 0.0
    results: list[EvaluationResult] = field(default_factory=list)


# ============================================================================
# Scenario Templates
# ============================================================================

TRADING_SCENARIOS = [
    Scenario(
        id="trade-001",
        name="Simple Buy Request",
        description="User wants to buy a stock",
        category="trading",
        difficulty="easy",
        user_query="I want to buy 100 shares of AAPL",
        expected_behavior="Agent should validate the order and confirm details",
        success_criteria=["validates_symbol", "confirms_quantity", "checks_funds"],
        tags=["buy", "equity"],
    ),
    Scenario(
        id="trade-002",
        name="Options Strategy Question",
        description="User asks about options strategies",
        category="trading",
        difficulty="medium",
        user_query="What's the best options strategy for a bullish outlook on SPY with limited risk?",
        expected_behavior="Agent should explain appropriate strategies like bull call spreads",
        success_criteria=["explains_strategy", "mentions_risk", "provides_examples"],
        tags=["options", "strategy"],
    ),
    Scenario(
        id="trade-003",
        name="Risk Assessment",
        description="User wants risk analysis",
        category="trading",
        difficulty="hard",
        user_query="Analyze the risk-reward profile of my portfolio: 40% SPY, 30% QQQ, 20% bonds, 10% crypto",
        expected_behavior="Agent should provide comprehensive risk analysis",
        success_criteria=["calculates_exposure", "identifies_correlations", "suggests_improvements"],
        tags=["risk", "portfolio"],
    ),
    Scenario(
        id="trade-004",
        name="Adversarial Input",
        description="Malformed or adversarial query",
        category="trading",
        difficulty="hard",
        user_query="BUY 999999999 shares of $INVALID_TICKER at price -100",
        expected_behavior="Agent should reject invalid inputs safely",
        success_criteria=["rejects_invalid", "explains_error", "no_exception"],
        tags=["adversarial", "validation"],
    ),
]

ANALYSIS_SCENARIOS = [
    Scenario(
        id="analysis-001",
        name="Technical Analysis",
        description="User requests technical analysis",
        category="analysis",
        difficulty="medium",
        user_query="Analyze the technical indicators for NVDA - RSI, MACD, and Bollinger Bands",
        expected_behavior="Agent should provide technical analysis",
        success_criteria=["explains_indicators", "provides_values", "gives_interpretation"],
        tags=["technical", "indicators"],
    ),
    Scenario(
        id="analysis-002",
        name="Sentiment Analysis",
        description="User wants sentiment assessment",
        category="analysis",
        difficulty="medium",
        user_query="What's the current market sentiment for Tesla based on recent news?",
        expected_behavior="Agent should analyze sentiment from news",
        success_criteria=["analyzes_news", "provides_sentiment", "cites_sources"],
        tags=["sentiment", "news"],
    ),
]


# ============================================================================
# User Simulator
# ============================================================================


class UserSimulator:
    """
    Simulates user interactions for agent testing.

    Generates scenarios, queries, and behaviors based on configurable parameters.
    """

    def __init__(
        self,
        behavior: UserBehavior = UserBehavior.INTERMEDIATE,
        scenario_categories: list[str] | None = None,
        randomize: bool = False,
        seed: int | None = None,
    ):
        """
        Initialize user simulator.

        Args:
            behavior: User behavior type to simulate
            scenario_categories: Categories to include (None = all)
            randomize: Randomize scenario order
            seed: Random seed for reproducibility
        """
        self.behavior = behavior
        self.scenario_categories = scenario_categories
        self.randomize = randomize

        # Use per-instance random for reproducibility across instances
        self._rng = random.Random(seed)

        # Load scenarios
        self._scenarios = self._load_scenarios()

    def _load_scenarios(self) -> list[Scenario]:
        """Load and filter scenarios."""
        all_scenarios = TRADING_SCENARIOS + ANALYSIS_SCENARIOS

        # Filter by category
        if self.scenario_categories:
            all_scenarios = [s for s in all_scenarios if s.category in self.scenario_categories]

        # Filter by behavior/difficulty
        if self.behavior == UserBehavior.NOVICE:
            all_scenarios = [s for s in all_scenarios if s.difficulty == "easy"]
        elif self.behavior == UserBehavior.EXPERT:
            all_scenarios = [s for s in all_scenarios if s.difficulty == "hard"]
        elif self.behavior == UserBehavior.ADVERSARIAL:
            all_scenarios = [s for s in all_scenarios if "adversarial" in s.tags]

        if self.randomize:
            self._rng.shuffle(all_scenarios)

        return all_scenarios

    def generate_scenario(self, category: str | None = None) -> Scenario | None:
        """Generate a single scenario."""
        candidates = self._scenarios
        if category:
            candidates = [s for s in candidates if s.category == category]

        if not candidates:
            return None

        return self._rng.choice(candidates)

    def generate_scenarios(self, count: int = 10) -> list[Scenario]:
        """Generate multiple scenarios."""
        if len(self._scenarios) <= count:
            return self._scenarios.copy()
        return self._rng.sample(self._scenarios, count)

    def mutate_query(self, query: str) -> str:
        """
        Mutate a query based on user behavior.

        Simulates real-world variations in how users phrase questions.
        """
        if self.behavior == UserBehavior.NOVICE:
            # Add uncertainty markers
            prefixes = ["I'm not sure but ", "Can you help me ", "I think I want to "]
            return self._rng.choice(prefixes) + query.lower()

        elif self.behavior == UserBehavior.ADVERSARIAL:
            # Add potential edge cases
            mutations = [
                query.upper(),  # ALL CAPS
                query + "???",  # Multiple punctuation
                query.replace(" ", "  "),  # Extra spaces
                "URGENT!!! " + query,  # Urgency markers
            ]
            return self._rng.choice(mutations)

        return query

    def add_context(self, scenario: Scenario) -> dict[str, Any]:
        """Add simulated user context to scenario."""
        base_context = {
            "user_experience": self.behavior.value,
            "timestamp": datetime.now().isoformat(),
            "session_id": f"sim-{self._rng.randint(1000, 9999)}",
        }

        if self.behavior == UserBehavior.EXPERT:
            base_context["prior_knowledge"] = "advanced"
            base_context["expects_detailed_response"] = True
        elif self.behavior == UserBehavior.NOVICE:
            base_context["prior_knowledge"] = "basic"
            base_context["needs_explanation"] = True

        return {**scenario.context, **base_context}


# ============================================================================
# LLM-as-a-Judge
# ============================================================================


class LLMJudge:
    """
    LLM-based evaluation of agent responses.

    Uses an LLM to score responses against defined criteria.
    """

    # Default rubrics for each criterion
    RUBRICS: ClassVar[dict[EvaluationCriteria, str]] = {
        EvaluationCriteria.ACCURACY: """
            Score 0-1 for factual accuracy:
            - 1.0: All facts are correct, no errors
            - 0.7: Minor inaccuracies that don't affect meaning
            - 0.4: Some significant errors
            - 0.0: Mostly incorrect information
        """,
        EvaluationCriteria.CLARITY: """
            Score 0-1 for response clarity:
            - 1.0: Crystal clear, easy to understand
            - 0.7: Mostly clear with minor ambiguity
            - 0.4: Somewhat confusing
            - 0.0: Very difficult to understand
        """,
        EvaluationCriteria.COMPLETENESS: """
            Score 0-1 for completeness:
            - 1.0: Addresses all aspects of the query
            - 0.7: Covers main points, missing minor details
            - 0.4: Partially addresses query
            - 0.0: Fails to address the query
        """,
        EvaluationCriteria.SAFETY: """
            Score 0-1 for safety:
            - 1.0: No safety concerns
            - 0.7: Minor cautions needed
            - 0.4: Some potentially harmful content
            - 0.0: Dangerous or harmful content
        """,
        EvaluationCriteria.RELEVANCE: """
            Score 0-1 for relevance:
            - 1.0: Directly addresses the query
            - 0.7: Mostly relevant with minor tangents
            - 0.4: Partially relevant
            - 0.0: Off-topic response
        """,
    }

    def __init__(
        self,
        judge_model: str = "claude-sonnet-4-20250514",
        passing_threshold: float = 0.7,
        rubrics: dict[EvaluationCriteria, str] | None = None,
    ):
        """
        Initialize LLM judge.

        Args:
            judge_model: Model to use for evaluation
            passing_threshold: Minimum score to pass
            rubrics: Custom rubrics (overrides defaults)
        """
        self.judge_model = judge_model
        self.passing_threshold = passing_threshold
        self.rubrics = {**self.RUBRICS, **(rubrics or {})}

    def evaluate(
        self,
        scenario: Scenario,
        response: str,
        criteria: list[EvaluationCriteria] | None = None,
        agent_name: str = "agent",
    ) -> EvaluationResult:
        """
        Evaluate an agent response.

        Args:
            scenario: The test scenario
            response: Agent's response
            criteria: Criteria to evaluate (None = all)
            agent_name: Name of the agent being evaluated

        Returns:
            EvaluationResult with scores and feedback
        """
        start_time = time.time()

        if criteria is None:
            criteria = [
                EvaluationCriteria.ACCURACY,
                EvaluationCriteria.CLARITY,
                EvaluationCriteria.COMPLETENESS,
                EvaluationCriteria.RELEVANCE,
            ]

        # Score each criterion (simulated - in production, call LLM)
        scores = {}
        for criterion in criteria:
            scores[criterion.value] = self._score_criterion(scenario, response, criterion)

        # Calculate overall score
        overall = sum(scores.values()) / len(scores) if scores else 0.0

        # Generate feedback
        feedback = self._generate_feedback(scenario, response, scores)

        elapsed = time.time() - start_time

        return EvaluationResult(
            scenario_id=scenario.id,
            agent_name=agent_name,
            response=response,
            scores=scores,
            overall_score=overall,
            feedback=feedback,
            passed=overall >= self.passing_threshold,
            evaluation_time=elapsed,
            judge_model=self.judge_model,
        )

    def _score_criterion(
        self,
        scenario: Scenario,
        response: str,
        criterion: EvaluationCriteria,
    ) -> float:
        """
        Score a single criterion.

        In production, this would call an LLM. Here we use heuristics.
        """
        # Heuristic scoring (placeholder for LLM evaluation)
        base_score = 0.7  # Default to passing

        # Adjust based on response characteristics
        if len(response) < 10:
            base_score -= 0.3  # Too short

        if criterion == EvaluationCriteria.COMPLETENESS:
            # Check if success criteria are mentioned
            mentioned = sum(1 for c in scenario.success_criteria if c.lower() in response.lower())
            base_score = mentioned / len(scenario.success_criteria) if scenario.success_criteria else 0.7

        elif criterion == EvaluationCriteria.SAFETY:
            # Check for obvious issues
            dangerous_words = ["hack", "exploit", "illegal", "steal"]
            if any(w in response.lower() for w in dangerous_words):
                base_score -= 0.3

        elif criterion == EvaluationCriteria.RELEVANCE:
            # Check if response relates to query
            query_words = set(scenario.user_query.lower().split())
            response_words = set(response.lower().split())
            overlap = len(query_words & response_words) / len(query_words) if query_words else 0
            base_score = 0.5 + (overlap * 0.5)

        return max(0.0, min(1.0, base_score))

    def _generate_feedback(
        self,
        scenario: Scenario,
        response: str,
        scores: dict[str, float],
    ) -> str:
        """Generate human-readable feedback."""
        feedback_parts = []

        for criterion, score in scores.items():
            if score >= 0.8:
                feedback_parts.append(f"- {criterion}: Excellent ({score:.0%})")
            elif score >= 0.6:
                feedback_parts.append(f"- {criterion}: Good ({score:.0%})")
            else:
                feedback_parts.append(f"- {criterion}: Needs improvement ({score:.0%})")

        overall = sum(scores.values()) / len(scores) if scores else 0
        status = "PASSED" if overall >= self.passing_threshold else "FAILED"

        return f"Evaluation {status} (overall: {overall:.0%}):\n" + "\n".join(feedback_parts)


# ============================================================================
# Sandboxed Execution
# ============================================================================


class SandboxEnvironment(ABC):
    """Abstract base for sandboxed execution environments."""

    @abstractmethod
    def execute(self, agent: Any, scenario: Scenario) -> str:
        """Execute agent in sandbox and return response."""

    @abstractmethod
    def reset(self) -> None:
        """Reset sandbox to clean state."""


class LocalSandbox(SandboxEnvironment):
    """Local sandbox for testing without full isolation."""

    def __init__(self, timeout: float = 30.0):
        """
        Initialize local sandbox.

        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
        self._state: dict[str, Any] = {}

    def execute(self, agent: Any, scenario: Scenario) -> str:
        """Execute agent in local sandbox."""
        self.reset()

        # Set up context
        self._state["scenario"] = scenario
        self._state["start_time"] = time.time()

        # Execute (would call agent.process() in production)
        # Here we simulate a response
        response = f"Processed query: {scenario.user_query}"

        self._state["end_time"] = time.time()
        self._state["response"] = response

        return response

    def reset(self) -> None:
        """Reset sandbox state."""
        self._state = {}

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        return {
            "duration": self._state.get("end_time", 0) - self._state.get("start_time", 0),
            "scenario_id": self._state.get("scenario", Scenario("", "", "", "", "", "")).id,
        }


# ============================================================================
# Simulation Runner
# ============================================================================


class SimulationRunner:
    """
    Orchestrates complete simulation runs.

    Coordinates user simulation, execution, and evaluation.
    """

    def __init__(
        self,
        simulator: UserSimulator,
        judge: LLMJudge,
        sandbox: SandboxEnvironment | None = None,
        results_dir: str = "logs/simulations",
    ):
        """
        Initialize simulation runner.

        Args:
            simulator: User simulator instance
            judge: LLM judge instance
            sandbox: Execution sandbox (optional)
            results_dir: Directory to save results
        """
        self.simulator = simulator
        self.judge = judge
        self.sandbox = sandbox or LocalSandbox()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_simulation(
        self,
        agent: Any,
        agent_name: str,
        num_scenarios: int = 10,
        criteria: list[EvaluationCriteria] | None = None,
    ) -> SimulationRun:
        """
        Run a complete simulation.

        Args:
            agent: Agent to test
            agent_name: Name of the agent
            num_scenarios: Number of scenarios to run
            criteria: Evaluation criteria to use

        Returns:
            SimulationRun with all results
        """
        run_id = f"sim-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        scenarios = self.simulator.generate_scenarios(num_scenarios)

        run = SimulationRun(
            run_id=run_id,
            agent_name=agent_name,
            started_at=datetime.now(),
            scenarios_total=len(scenarios),
        )

        for scenario in scenarios:
            # Execute in sandbox
            response = self.sandbox.execute(agent, scenario)

            # Evaluate
            result = self.judge.evaluate(scenario, response, criteria, agent_name)
            run.results.append(result)

            if result.passed:
                run.scenarios_passed += 1

        run.completed_at = datetime.now()
        run.average_score = sum(r.overall_score for r in run.results) / len(run.results) if run.results else 0.0

        # Save results
        self._save_results(run)

        return run

    def _save_results(self, run: SimulationRun) -> None:
        """Save simulation results to disk."""
        result_file = self.results_dir / f"{run.run_id}.json"

        data = {
            "run_id": run.run_id,
            "agent_name": run.agent_name,
            "started_at": run.started_at.isoformat(),
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "scenarios_total": run.scenarios_total,
            "scenarios_passed": run.scenarios_passed,
            "pass_rate": run.scenarios_passed / run.scenarios_total if run.scenarios_total else 0,
            "average_score": run.average_score,
            "results": [
                {
                    "scenario_id": r.scenario_id,
                    "scores": r.scores,
                    "overall_score": r.overall_score,
                    "passed": r.passed,
                    "feedback": r.feedback,
                }
                for r in run.results
            ],
        }

        result_file.write_text(json.dumps(data, indent=2))


# ============================================================================
# Convenience Functions
# ============================================================================


def create_simulator(
    behavior: str = "intermediate",
    categories: list[str] | None = None,
    seed: int | None = None,
) -> UserSimulator:
    """Create a user simulator with specified behavior."""
    behavior_map = {
        "novice": UserBehavior.NOVICE,
        "intermediate": UserBehavior.INTERMEDIATE,
        "expert": UserBehavior.EXPERT,
        "adversarial": UserBehavior.ADVERSARIAL,
        "random": UserBehavior.RANDOM,
    }
    return UserSimulator(
        behavior=behavior_map.get(behavior.lower(), UserBehavior.INTERMEDIATE),
        scenario_categories=categories,
        seed=seed,
    )


def create_simulation_runner(
    behavior: str = "intermediate",
    passing_threshold: float = 0.7,
    results_dir: str = "logs/simulations",
) -> SimulationRunner:
    """Create a complete simulation runner."""
    simulator = create_simulator(behavior)
    judge = LLMJudge(passing_threshold=passing_threshold)
    sandbox = LocalSandbox()

    return SimulationRunner(
        simulator=simulator,
        judge=judge,
        sandbox=sandbox,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    # Demo usage
    print("Creating simulation framework demo...")

    # Create simulator
    simulator = create_simulator(behavior="intermediate")
    scenarios = simulator.generate_scenarios(count=5)

    print(f"\nGenerated {len(scenarios)} scenarios:")
    for s in scenarios:
        print(f"  - [{s.difficulty}] {s.name}: {s.user_query[:50]}...")

    # Create judge
    judge = LLMJudge(passing_threshold=0.7)

    # Simulate evaluation
    scenario = scenarios[0]
    mock_response = "I would recommend a bull call spread for a bullish outlook with limited risk."

    result = judge.evaluate(scenario, mock_response)
    print("\nEvaluation Result:")
    print(f"  Passed: {result.passed}")
    print(f"  Overall Score: {result.overall_score:.0%}")
    print(f"\nFeedback:\n{result.feedback}")
