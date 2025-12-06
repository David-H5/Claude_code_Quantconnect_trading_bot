"""Tests for simulation-based testing framework."""

# Import directly to avoid numpy dependency issues
import sys


sys.path.insert(0, ".")

import pytest

from evaluation.simulation import (
    EvaluationCriteria,
    EvaluationResult,
    LLMJudge,
    LocalSandbox,
    Scenario,
    SimulationRun,
    SimulationRunner,
    UserBehavior,
    UserSimulator,
    create_simulation_runner,
    create_simulator,
)


class TestUserBehavior:
    """Tests for UserBehavior enum."""

    def test_behavior_values(self):
        """Test behavior enum values."""
        assert UserBehavior.NOVICE.value == "novice"
        assert UserBehavior.EXPERT.value == "expert"
        assert UserBehavior.ADVERSARIAL.value == "adversarial"


class TestEvaluationCriteria:
    """Tests for EvaluationCriteria enum."""

    def test_criteria_values(self):
        """Test criteria enum values."""
        assert EvaluationCriteria.ACCURACY.value == "accuracy"
        assert EvaluationCriteria.SAFETY.value == "safety"
        assert EvaluationCriteria.RELEVANCE.value == "relevance"


class TestScenario:
    """Tests for Scenario dataclass."""

    def test_scenario_creation(self):
        """Test creating a scenario."""
        scenario = Scenario(
            id="test-001",
            name="Test Scenario",
            description="A test scenario",
            category="testing",
            difficulty="easy",
            user_query="Test query",
        )
        assert scenario.id == "test-001"
        assert scenario.difficulty == "easy"

    def test_scenario_with_optional_fields(self):
        """Test scenario with all optional fields."""
        scenario = Scenario(
            id="test-002",
            name="Full Scenario",
            description="Complete scenario",
            category="trading",
            difficulty="hard",
            user_query="Complex query",
            context={"key": "value"},
            expected_behavior="Expected response",
            success_criteria=["criterion1", "criterion2"],
            tags=["tag1", "tag2"],
        )
        assert len(scenario.success_criteria) == 2
        assert "tag1" in scenario.tags


class TestUserSimulator:
    """Tests for UserSimulator class."""

    def test_create_simulator_default(self):
        """Test creating default simulator."""
        simulator = UserSimulator()
        assert simulator.behavior == UserBehavior.INTERMEDIATE

    def test_create_simulator_novice(self):
        """Test novice behavior filters easy scenarios."""
        simulator = UserSimulator(behavior=UserBehavior.NOVICE)
        scenarios = simulator.generate_scenarios(count=10)
        for s in scenarios:
            assert s.difficulty == "easy"

    def test_create_simulator_expert(self):
        """Test expert behavior filters hard scenarios."""
        simulator = UserSimulator(behavior=UserBehavior.EXPERT)
        scenarios = simulator.generate_scenarios(count=10)
        for s in scenarios:
            assert s.difficulty == "hard"

    def test_create_simulator_adversarial(self):
        """Test adversarial behavior filters adversarial scenarios."""
        simulator = UserSimulator(behavior=UserBehavior.ADVERSARIAL)
        scenarios = simulator.generate_scenarios(count=10)
        for s in scenarios:
            assert "adversarial" in s.tags

    def test_generate_single_scenario(self):
        """Test generating a single scenario."""
        simulator = UserSimulator()
        scenario = simulator.generate_scenario()
        assert scenario is not None
        assert isinstance(scenario, Scenario)

    def test_generate_scenario_by_category(self):
        """Test generating scenario by category."""
        simulator = UserSimulator()
        scenario = simulator.generate_scenario(category="trading")
        if scenario:
            assert scenario.category == "trading"

    def test_generate_multiple_scenarios(self):
        """Test generating multiple scenarios."""
        simulator = UserSimulator(seed=42)
        scenarios = simulator.generate_scenarios(count=5)
        assert len(scenarios) == 5

    def test_mutate_query_novice(self):
        """Test query mutation for novice."""
        simulator = UserSimulator(behavior=UserBehavior.NOVICE)
        mutated = simulator.mutate_query("Buy AAPL")
        assert mutated.islower() or "help" in mutated.lower() or "not sure" in mutated.lower()

    def test_mutate_query_adversarial(self):
        """Test query mutation for adversarial."""
        simulator = UserSimulator(behavior=UserBehavior.ADVERSARIAL)
        original = "Buy AAPL"
        mutated = simulator.mutate_query(original)
        # Should be different from original
        assert mutated != original

    def test_add_context(self):
        """Test adding context to scenario."""
        simulator = UserSimulator(behavior=UserBehavior.EXPERT)
        scenario = Scenario(
            id="test",
            name="Test",
            description="Test",
            category="test",
            difficulty="easy",
            user_query="Query",
        )
        context = simulator.add_context(scenario)
        assert "user_experience" in context
        assert context["user_experience"] == "expert"

    def test_reproducibility_with_seed(self):
        """Test that seed provides reproducibility."""
        sim1 = UserSimulator(seed=12345, randomize=True)
        sim2 = UserSimulator(seed=12345, randomize=True)

        scenarios1 = sim1.generate_scenarios(count=3)
        scenarios2 = sim2.generate_scenarios(count=3)

        assert [s.id for s in scenarios1] == [s.id for s in scenarios2]


class TestLLMJudge:
    """Tests for LLMJudge class."""

    @pytest.fixture
    def judge(self):
        """Create judge instance."""
        return LLMJudge(passing_threshold=0.7)

    @pytest.fixture
    def sample_scenario(self):
        """Create sample scenario."""
        return Scenario(
            id="test-001",
            name="Test",
            description="Test scenario",
            category="trading",
            difficulty="easy",
            user_query="What is a stock?",
            success_criteria=["explains_stock", "gives_example"],
        )

    def test_evaluate_passing(self, judge, sample_scenario):
        """Test evaluation that passes."""
        response = "A stock explains_stock is ownership in a company. For example gives_example..."
        result = judge.evaluate(sample_scenario, response)
        assert isinstance(result, EvaluationResult)
        assert result.overall_score > 0

    def test_evaluate_short_response(self, judge, sample_scenario):
        """Test evaluation penalizes short responses."""
        result = judge.evaluate(sample_scenario, "Yes")
        assert result.scores.get("accuracy", 0) < 0.7

    def test_evaluate_with_specific_criteria(self, judge, sample_scenario):
        """Test evaluation with specific criteria."""
        criteria = [EvaluationCriteria.ACCURACY, EvaluationCriteria.SAFETY]
        result = judge.evaluate(sample_scenario, "Test response", criteria)
        assert "accuracy" in result.scores
        assert "safety" in result.scores
        assert "clarity" not in result.scores

    def test_evaluate_feedback_format(self, judge, sample_scenario):
        """Test feedback is generated."""
        result = judge.evaluate(sample_scenario, "Test response")
        assert "Evaluation" in result.feedback
        assert "%" in result.feedback

    def test_safety_scoring(self, judge, sample_scenario):
        """Test safety scoring detects dangerous content."""
        dangerous_response = "You should hack into the system and exploit vulnerabilities"
        result = judge.evaluate(
            sample_scenario,
            dangerous_response,
            [EvaluationCriteria.SAFETY],
        )
        assert result.scores["safety"] < 0.7


class TestLocalSandbox:
    """Tests for LocalSandbox class."""

    @pytest.fixture
    def sandbox(self):
        """Create sandbox instance."""
        return LocalSandbox(timeout=30.0)

    @pytest.fixture
    def sample_scenario(self):
        """Create sample scenario."""
        return Scenario(
            id="sandbox-test",
            name="Sandbox Test",
            description="Test",
            category="test",
            difficulty="easy",
            user_query="Test query",
        )

    def test_execute(self, sandbox, sample_scenario):
        """Test basic execution."""
        response = sandbox.execute(None, sample_scenario)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_reset(self, sandbox, sample_scenario):
        """Test reset clears state."""
        sandbox.execute(None, sample_scenario)
        sandbox.reset()
        assert sandbox._state == {}

    def test_execution_stats(self, sandbox, sample_scenario):
        """Test execution statistics."""
        sandbox.execute(None, sample_scenario)
        stats = sandbox.get_execution_stats()
        assert "duration" in stats
        assert stats["duration"] >= 0


class TestSimulationRunner:
    """Tests for SimulationRunner class."""

    @pytest.fixture
    def runner(self, tmp_path):
        """Create runner instance."""
        return SimulationRunner(
            simulator=UserSimulator(seed=42),
            judge=LLMJudge(),
            sandbox=LocalSandbox(),
            results_dir=str(tmp_path / "results"),
        )

    def test_run_simulation(self, runner):
        """Test running a simulation."""
        run = runner.run_simulation(
            agent=None,
            agent_name="test_agent",
            num_scenarios=3,
        )
        assert isinstance(run, SimulationRun)
        assert run.scenarios_total == 3
        assert len(run.results) == 3

    def test_run_saves_results(self, runner, tmp_path):
        """Test simulation saves results to disk."""
        runner.run_simulation(
            agent=None,
            agent_name="test_agent",
            num_scenarios=2,
        )

        result_files = list((tmp_path / "results").glob("*.json"))
        assert len(result_files) == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_simulator_default(self):
        """Test create_simulator with defaults."""
        simulator = create_simulator()
        assert simulator.behavior == UserBehavior.INTERMEDIATE

    def test_create_simulator_with_behavior(self):
        """Test create_simulator with behavior."""
        simulator = create_simulator(behavior="expert")
        assert simulator.behavior == UserBehavior.EXPERT

    def test_create_simulator_with_categories(self):
        """Test create_simulator with categories."""
        simulator = create_simulator(categories=["trading"])
        scenarios = simulator.generate_scenarios(count=10)
        for s in scenarios:
            assert s.category == "trading"

    def test_create_simulation_runner(self, tmp_path):
        """Test create_simulation_runner."""
        runner = create_simulation_runner(
            behavior="novice",
            passing_threshold=0.5,
            results_dir=str(tmp_path),
        )
        assert runner.simulator.behavior == UserBehavior.NOVICE
        assert runner.judge.passing_threshold == 0.5


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_scenarios(self):
        """Test handling when no scenarios match."""
        simulator = UserSimulator(
            behavior=UserBehavior.NOVICE,
            scenario_categories=["nonexistent"],
        )
        scenarios = simulator.generate_scenarios(count=5)
        assert len(scenarios) == 0

    def test_evaluation_empty_response(self):
        """Test evaluation of empty response."""
        judge = LLMJudge()
        scenario = Scenario(
            id="test",
            name="Test",
            description="Test",
            category="test",
            difficulty="easy",
            user_query="Query",
        )
        result = judge.evaluate(scenario, "")
        assert result.overall_score < 0.7

    def test_long_response(self):
        """Test evaluation of very long response."""
        judge = LLMJudge()
        scenario = Scenario(
            id="test",
            name="Test",
            description="Test",
            category="test",
            difficulty="easy",
            user_query="Query",
        )
        long_response = "This is a test response. " * 1000
        result = judge.evaluate(scenario, long_response)
        assert result.overall_score >= 0  # Should not crash
