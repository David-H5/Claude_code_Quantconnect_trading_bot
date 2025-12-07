"""
Tests for Dual LLM Model Router (UPGRADE-010 Sprint 2)

Tests task classification, model routing, and cost tracking.
"""

from datetime import datetime, timedelta

import pytest

from llm.model_router import (
    CostReport,
    CostTracker,
    LLMRouter,
    ModelConfig,
    ModelTier,
    RouteDecision,
    TaskClassifier,
    TaskType,
    UsageRecord,
    create_router,
)


class TestTaskType:
    """Tests for TaskType enum."""

    @pytest.mark.unit
    def test_task_types_exist(self):
        """Test all task types exist."""
        assert TaskType.REASONING is not None
        assert TaskType.STANDARD is not None
        assert TaskType.TOOL_USE is not None
        assert TaskType.SIMPLE is not None

    @pytest.mark.unit
    def test_task_type_values(self):
        """Test task type values."""
        assert TaskType.REASONING.value == "reasoning"
        assert TaskType.TOOL_USE.value == "tool_use"


class TestModelTier:
    """Tests for ModelTier enum."""

    @pytest.mark.unit
    def test_model_tiers_exist(self):
        """Test all tiers exist."""
        assert ModelTier.REASONING is not None
        assert ModelTier.STANDARD is not None
        assert ModelTier.FAST is not None


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    @pytest.mark.unit
    def test_model_config_creation(self):
        """Test creating model config."""
        config = ModelConfig(
            provider="anthropic",
            model_id="claude-3-opus",
            tier=ModelTier.REASONING,
            cost_per_1k_input=15.0,
            cost_per_1k_output=75.0,
            avg_latency_ms=5000,
        )

        assert config.provider == "anthropic"
        assert config.model_id == "claude-3-opus"
        assert config.tier == ModelTier.REASONING
        assert config.cost_per_1k_input == 15.0

    @pytest.mark.unit
    def test_model_config_defaults(self):
        """Test default values."""
        config = ModelConfig(
            provider="test",
            model_id="test-model",
            tier=ModelTier.FAST,
            cost_per_1k_input=0.1,
            cost_per_1k_output=0.2,
            avg_latency_ms=100,
        )

        assert config.max_tokens == 4096
        assert config.supports_tools is True
        assert config.supports_vision is False


class TestTaskClassifier:
    """Tests for TaskClassifier."""

    @pytest.fixture
    def classifier(self) -> TaskClassifier:
        """Create classifier for testing."""
        return TaskClassifier()

    @pytest.mark.unit
    def test_classify_reasoning_task(self, classifier):
        """Test classifying reasoning tasks."""
        prompt = "Analyze the implications of this trade strategy and explain why it might fail"

        task_type, confidence = classifier.classify(prompt)

        assert task_type == TaskType.REASONING
        assert confidence > 0.3

    @pytest.mark.unit
    def test_classify_tool_use_task(self, classifier):
        """Test classifying tool use tasks."""
        prompt = "Execute the order to buy 100 shares of SPY"

        task_type, confidence = classifier.classify(prompt)

        assert task_type == TaskType.TOOL_USE
        assert confidence >= 0.3

    @pytest.mark.unit
    def test_classify_simple_task(self, classifier):
        """Test classifying simple tasks."""
        prompt = "Format this list of tickers"

        task_type, confidence = classifier.classify(prompt)

        assert task_type == TaskType.SIMPLE

    @pytest.mark.unit
    def test_classify_standard_task(self, classifier):
        """Test classifying standard tasks."""
        prompt = "What is the current price of SPY?"

        task_type, confidence = classifier.classify(prompt)

        # Should default to standard for ambiguous prompts
        assert task_type in [TaskType.STANDARD, TaskType.SIMPLE]

    @pytest.mark.unit
    def test_long_prompt_biases_reasoning(self, classifier):
        """Test that long prompts bias toward reasoning."""
        short_prompt = "Buy SPY"
        long_prompt = " ".join(["word"] * 150)

        short_type, _ = classifier.classify(short_prompt)
        long_type, _ = classifier.classify(long_prompt)

        # Long prompts should lean toward reasoning
        # Short prompts should lean toward simple/tool_use
        assert short_type != TaskType.REASONING or long_type == TaskType.REASONING

    @pytest.mark.unit
    def test_question_complexity(self, classifier):
        """Test that question type affects classification."""
        complex_q = "Why would this strategy underperform in high volatility?"
        simple_q = "Is SPY up today?"

        complex_type, _ = classifier.classify(complex_q)
        simple_type, _ = classifier.classify(simple_q)

        assert complex_type == TaskType.REASONING
        # Simple yes/no questions
        assert simple_type in [TaskType.SIMPLE, TaskType.STANDARD]


class TestCostTracker:
    """Tests for CostTracker."""

    @pytest.fixture
    def tracker(self) -> CostTracker:
        """Create tracker for testing."""
        return CostTracker()

    @pytest.mark.unit
    def test_record_usage(self, tracker):
        """Test recording usage."""
        record = tracker.record_usage(
            provider="anthropic",
            model_id="claude-3-opus",
            tier=ModelTier.REASONING,
            input_tokens=1000,
            output_tokens=500,
            latency_ms=2500,
            cost_usd=0.15,
            task_type=TaskType.REASONING,
        )

        assert record.provider == "anthropic"
        assert record.input_tokens == 1000
        assert record.cost_usd == 0.15

    @pytest.mark.unit
    def test_get_cost_report_empty(self, tracker):
        """Test cost report with no records."""
        report = tracker.get_cost_report()

        assert report.total_cost_usd == 0.0
        assert report.total_requests == 0

    @pytest.mark.unit
    def test_get_cost_report(self, tracker):
        """Test cost report with records."""
        # Add some records
        tracker.record_usage(
            "anthropic",
            "claude-3-opus",
            ModelTier.REASONING,
            1000,
            500,
            2500,
            0.15,
            TaskType.REASONING,
        )
        tracker.record_usage(
            "anthropic",
            "claude-3-haiku",
            ModelTier.FAST,
            1000,
            500,
            500,
            0.01,
            TaskType.TOOL_USE,
        )

        report = tracker.get_cost_report()

        assert report.total_cost_usd == 0.16
        assert report.total_requests == 2
        assert "reasoning" in report.cost_by_tier
        assert "fast" in report.cost_by_tier

    @pytest.mark.unit
    def test_get_cost_report_time_filter(self, tracker):
        """Test cost report with time filtering."""
        # Add a record
        tracker.record_usage(
            "anthropic",
            "claude-3-opus",
            ModelTier.REASONING,
            1000,
            500,
            2500,
            0.15,
            TaskType.REASONING,
        )

        # Filter to future
        future = datetime.utcnow() + timedelta(hours=1)
        report = tracker.get_cost_report(start_time=future)

        assert report.total_requests == 0

    @pytest.mark.unit
    def test_history_limit(self):
        """Test that history is limited."""
        tracker = CostTracker(max_history=10)

        for _ in range(20):
            tracker.record_usage(
                "test",
                "model",
                ModelTier.FAST,
                100,
                50,
                100,
                0.01,
                TaskType.SIMPLE,
            )

        stats = tracker.get_statistics()
        assert stats["total_records"] == 10


class TestLLMRouter:
    """Tests for LLMRouter."""

    @pytest.fixture
    def router(self) -> LLMRouter:
        """Create router for testing."""
        return LLMRouter()

    @pytest.mark.unit
    def test_router_creation(self, router):
        """Test router creation."""
        assert router is not None
        assert len(router.models) > 0

    @pytest.mark.unit
    def test_route_reasoning_task(self, router):
        """Test routing reasoning tasks."""
        decision = router.route("Analyze and explain the implications of this complex strategy")

        assert decision.tier == ModelTier.REASONING
        assert decision.task_type == TaskType.REASONING

    @pytest.mark.unit
    def test_route_tool_use_task(self, router):
        """Test routing tool use tasks."""
        decision = router.route(
            "Execute order to buy SPY",
            task_hint=TaskType.TOOL_USE,
        )

        assert decision.tier == ModelTier.FAST
        assert decision.task_type == TaskType.TOOL_USE

    @pytest.mark.unit
    def test_route_with_task_hint(self, router):
        """Test routing with explicit task hint."""
        decision = router.route(
            "Hello",
            task_hint=TaskType.REASONING,
        )

        assert decision.task_type == TaskType.REASONING
        assert decision.tier == ModelTier.REASONING

    @pytest.mark.unit
    def test_route_with_override(self, router):
        """Test routing with override pattern."""
        router.add_override(r"urgent", ModelTier.REASONING)

        decision = router.route("This is an urgent request")

        assert decision.tier == ModelTier.REASONING
        assert decision.override_used is True

    @pytest.mark.unit
    def test_remove_override(self, router):
        """Test removing override."""
        router.add_override(r"test", ModelTier.FAST)

        result = router.remove_override(r"test")
        assert result is True

        result = router.remove_override(r"nonexistent")
        assert result is False

    @pytest.mark.unit
    def test_route_with_tool_requirement(self, router):
        """Test routing with tool requirement."""
        decision = router.route(
            "Simple task",
            require_tools=True,
        )

        # Should get a model that supports tools
        model = router.get_model_info(decision.provider, decision.model_id)
        assert model is not None
        assert model.supports_tools is True

    @pytest.mark.unit
    def test_route_with_vision_requirement(self, router):
        """Test routing with vision requirement."""
        decision = router.route(
            "Analyze this chart",
            require_vision=True,
        )

        model = router.get_model_info(decision.provider, decision.model_id)
        assert model is not None
        assert model.supports_vision is True

    @pytest.mark.unit
    def test_classify_task(self, router):
        """Test task classification through router."""
        task_type, confidence = router.classify_task("Analyze and compare these strategies")

        assert task_type == TaskType.REASONING
        assert confidence > 0

    @pytest.mark.unit
    def test_record_usage(self, router):
        """Test recording usage through router."""
        decision = router.route("Test prompt")

        record = router.record_usage(
            decision=decision,
            input_tokens=100,
            output_tokens=50,
            latency_ms=1000,
        )

        assert record is not None
        assert record.provider == decision.provider

    @pytest.mark.unit
    def test_get_cost_summary(self, router):
        """Test getting cost summary."""
        # Make a request and record usage
        decision = router.route("Test")
        router.record_usage(decision, 100, 50, 1000)

        report = router.get_cost_summary()

        assert report.total_requests == 1

    @pytest.mark.unit
    def test_get_model_info(self, router):
        """Test getting model info."""
        model = router.get_model_info("anthropic", "claude-3-opus")

        assert model is not None
        assert model.provider == "anthropic"
        assert model.tier == ModelTier.REASONING

    @pytest.mark.unit
    def test_get_model_info_not_found(self, router):
        """Test getting non-existent model."""
        model = router.get_model_info("unknown", "model")
        assert model is None

    @pytest.mark.unit
    def test_list_models(self, router):
        """Test listing all models."""
        models = router.list_models()
        assert len(models) > 0

    @pytest.mark.unit
    def test_list_models_by_tier(self, router):
        """Test listing models by tier."""
        reasoning_models = router.list_models(tier=ModelTier.REASONING)

        assert len(reasoning_models) > 0
        assert all(m.tier == ModelTier.REASONING for m in reasoning_models)

    @pytest.mark.unit
    def test_preferred_provider(self):
        """Test preferred provider selection."""
        router = LLMRouter(preferred_provider="openai")

        decision = router.route("Test prompt")

        # Should prefer OpenAI when possible
        assert decision.provider == "openai"

    @pytest.mark.unit
    def test_get_statistics(self, router):
        """Test getting statistics."""
        stats = router.get_statistics()

        assert "total_models" in stats
        assert "models_by_tier" in stats
        assert stats["total_models"] > 0


class TestRouteDecision:
    """Tests for RouteDecision dataclass."""

    @pytest.mark.unit
    def test_route_decision_creation(self):
        """Test creating route decision."""
        decision = RouteDecision(
            provider="anthropic",
            model_id="claude-3-opus",
            tier=ModelTier.REASONING,
            task_type=TaskType.REASONING,
            reason="Classified as reasoning",
        )

        assert decision.provider == "anthropic"
        assert decision.tier == ModelTier.REASONING
        assert decision.override_used is False


class TestUsageRecord:
    """Tests for UsageRecord dataclass."""

    @pytest.mark.unit
    def test_usage_record_creation(self):
        """Test creating usage record."""
        record = UsageRecord(
            timestamp=datetime.utcnow(),
            provider="anthropic",
            model_id="claude-3-opus",
            tier=ModelTier.REASONING,
            input_tokens=1000,
            output_tokens=500,
            latency_ms=2500,
            cost_usd=0.15,
            task_type=TaskType.REASONING,
        )

        assert record.provider == "anthropic"
        assert record.input_tokens == 1000
        assert record.cost_usd == 0.15


class TestCostReport:
    """Tests for CostReport dataclass."""

    @pytest.mark.unit
    def test_cost_report_creation(self):
        """Test creating cost report."""
        report = CostReport(
            total_cost_usd=1.50,
            total_requests=10,
            total_input_tokens=10000,
            total_output_tokens=5000,
            cost_by_tier={"reasoning": 1.0, "fast": 0.5},
            cost_by_model={"anthropic/claude-3-opus": 1.5},
            requests_by_tier={"reasoning": 5, "fast": 5},
            avg_latency_by_tier={"reasoning": 3000, "fast": 500},
        )

        assert report.total_cost_usd == 1.50
        assert report.total_requests == 10
        assert len(report.cost_by_tier) == 2


class TestCreateRouter:
    """Tests for factory function."""

    @pytest.mark.unit
    def test_create_with_defaults(self):
        """Test factory with defaults."""
        router = create_router()

        assert router is not None
        assert router.default_tier == ModelTier.STANDARD

    @pytest.mark.unit
    def test_create_with_preferred_provider(self):
        """Test factory with preferred provider."""
        router = create_router(preferred_provider="openai")

        assert router.preferred_provider == "openai"

    @pytest.mark.unit
    def test_create_with_custom_tier(self):
        """Test factory with custom default tier."""
        router = create_router(default_tier=ModelTier.FAST)

        assert router.default_tier == ModelTier.FAST

    @pytest.mark.unit
    def test_create_with_custom_models(self):
        """Test factory with custom models."""
        custom = {
            "custom/model": ModelConfig(
                provider="custom",
                model_id="model",
                tier=ModelTier.STANDARD,
                cost_per_1k_input=1.0,
                cost_per_1k_output=2.0,
                avg_latency_ms=1000,
            ),
        }

        router = create_router(custom_models=custom)

        assert "custom/model" in router.models


class TestModelRouterIntegration:
    """Integration tests for router."""

    @pytest.mark.unit
    def test_full_routing_workflow(self):
        """Test complete routing workflow."""
        router = create_router()

        # Route a prompt
        decision = router.route("Analyze the market conditions and recommend a strategy")

        assert decision.tier == ModelTier.REASONING

        # Record usage
        record = router.record_usage(
            decision=decision,
            input_tokens=500,
            output_tokens=300,
            latency_ms=2000,
        )

        assert record.cost_usd > 0

        # Get report
        report = router.get_cost_summary()

        assert report.total_requests == 1
        assert report.total_cost_usd > 0

    @pytest.mark.unit
    def test_multiple_routes_cost_tracking(self):
        """Test cost tracking across multiple routes."""
        router = create_router()

        # Route several prompts
        prompts = [
            "Analyze this in depth",
            "Execute buy order",
            "Format this list",
        ]

        for prompt in prompts:
            decision = router.route(prompt)
            router.record_usage(decision, 100, 50, 500)

        report = router.get_cost_summary()

        assert report.total_requests == 3
        assert len(report.requests_by_tier) > 0
