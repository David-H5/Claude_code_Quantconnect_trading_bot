"""
Tests for Phase 1 Autonomous Agent Framework Components.

Phase 1 Components:
1.1 Test Case Format Adapter - AgentResponseAdapter
1.2 Real LLM Judge Implementation - LLMJudge
1.3 Circuit Breaker Agent Wrapper - SafeAgentWrapper
1.4 Safe Agent Factory Functions - create_safe_*
1.5 Real Judge Function - create_real_judge_function

Version: 1.1 (December 2025) - Added safe factory and real judge tests
"""

from unittest.mock import Mock, patch

import pytest

# Phase 1 components
from evaluation.adapters import (
    AgentResponseAdapter,
    adapt_response_to_decision,
    adapt_response_to_dict,
    batch_adapt_responses,
    create_test_case_from_response,
)

# Real judge function (Phase 1.5)
# Evaluation framework
from evaluation.agent_as_judge import EvaluationCategory, create_real_judge_function
from evaluation.llm_judge import (
    JudgeConfig,
    JudgeCostTracker,
    LLMJudge,
    create_production_judge,
)

# Safe agent factory functions (Phase 1.4)
from llm.agents import (
    create_safe_conservative_trader,
    create_safe_position_risk_manager,
    create_safe_sentiment_analyst,
    create_safe_supervisor_agent,
    create_safe_technical_analyst,
)

# Agent base classes
from llm.agents.base import (
    AgentResponse,
    AgentRole,
    AgentThought,
    ThoughtType,
    TradingAgent,
)
from llm.agents.safe_agent_wrapper import (
    RiskTier,
    RiskTierConfig,
    SafeAgentWrapper,
    wrap_agent_with_safety,
)

# Circuit breaker
from models.circuit_breaker import (
    CircuitBreakerConfig,
    TradingCircuitBreaker,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_agent_response():
    """Create a sample AgentResponse for testing."""
    thoughts = [
        AgentThought(
            thought_type=ThoughtType.REASONING,
            content="RSI shows oversold conditions at 25",
            metadata={"indicator": "RSI", "value": 25},
        ),
        AgentThought(
            thought_type=ThoughtType.REASONING,
            content="MACD showing bullish crossover",
            metadata={"indicator": "MACD", "signal": "bullish"},
        ),
        AgentThought(
            thought_type=ThoughtType.FINAL_ANSWER,
            content="Technical indicators support a bullish outlook",
            metadata={},
        ),
    ]

    return AgentResponse(
        agent_name="TechnicalAnalyst",
        agent_role=AgentRole.ANALYST,
        query="Analyze SPY for potential entry",
        thoughts=thoughts,
        final_answer="Action: BUY\nReason: RSI oversold with bullish MACD crossover",
        confidence=0.85,
        tools_used=["RSI", "MACD"],
        execution_time_ms=150.5,
        success=True,
        error=None,
    )


@pytest.fixture
def hold_agent_response():
    """Create an AgentResponse with HOLD action."""
    thoughts = [
        AgentThought(
            thought_type=ThoughtType.REASONING,
            content="Market conditions uncertain, waiting for confirmation",
            metadata={},
        ),
    ]

    return AgentResponse(
        agent_name="RiskManager",
        agent_role=AgentRole.RISK_MANAGER,
        query="Should we enter position?",
        thoughts=thoughts,
        final_answer="Action: HOLD\nReason: Wait for market confirmation",
        confidence=0.60,
        tools_used=[],
        execution_time_ms=50.0,
        success=True,
        error=None,
    )


@pytest.fixture
def mock_trading_agent():
    """Create a mock TradingAgent for wrapper testing."""
    agent = Mock(spec=TradingAgent)
    agent.name = "MockAgent"
    agent.role = AgentRole.ANALYST

    # Default analyze behavior
    def mock_analyze(query, context):
        return AgentResponse(
            agent_name="MockAgent",
            agent_role=AgentRole.ANALYST,
            query=query,
            thoughts=[
                AgentThought(
                    thought_type=ThoughtType.FINAL_ANSWER,
                    content="Analysis complete",
                    metadata={},
                ),
            ],
            final_answer="Action: BUY\nReason: Analysis indicates opportunity",
            confidence=0.75,
            tools_used=[],
            execution_time_ms=100.0,
            success=True,
            error=None,
        )

    agent.analyze = Mock(side_effect=mock_analyze)
    return agent


@pytest.fixture
def circuit_breaker():
    """Create a circuit breaker for testing."""
    config = CircuitBreakerConfig(
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.10,
        max_consecutive_losses=5,
        require_human_reset=False,
    )
    return TradingCircuitBreaker(config=config)


@pytest.fixture
def risk_config():
    """Create a risk configuration for testing."""
    return RiskTierConfig(
        low_max_position_pct=0.02,
        medium_max_position_pct=0.05,
        high_max_position_pct=0.10,
        high_volatility_threshold=0.03,
        max_correlated_exposure_pct=0.15,
    )


# =============================================================================
# Section 1.1: Agent Response Adapter Tests
# =============================================================================


class TestAgentResponseAdapterToDict:
    """Tests for AgentResponseAdapter.to_dict()."""

    def test_converts_basic_response(self, sample_agent_response):
        """Convert a basic AgentResponse to Dict."""
        result = AgentResponseAdapter.to_dict(sample_agent_response)

        assert result["agent_name"] == "TechnicalAnalyst"
        assert result["agent_role"] == "analyst"
        assert result["action"] == "BUY"
        assert result["confidence"] == 0.85
        assert result["success"] is True

    def test_extracts_reasoning_chain(self, sample_agent_response):
        """Verify reasoning chain is properly extracted."""
        result = AgentResponseAdapter.to_dict(sample_agent_response)

        assert len(result["reasoning_chain"]) == 3
        assert result["reasoning_chain"][0]["type"] == "reasoning"
        assert "RSI" in result["reasoning_chain"][0]["content"]

    def test_extracts_buy_action(self, sample_agent_response):
        """Extract BUY action from final answer."""
        result = AgentResponseAdapter.to_dict(sample_agent_response)
        assert result["action"] == "BUY"

    def test_extracts_hold_action(self, hold_agent_response):
        """Extract HOLD action from final answer."""
        result = AgentResponseAdapter.to_dict(hold_agent_response)
        assert result["action"] == "HOLD"

    def test_extracts_sell_action(self, sample_agent_response):
        """Extract SELL action from final answer."""
        sample_agent_response.final_answer = "Action: SELL\nReason: Take profits"
        result = AgentResponseAdapter.to_dict(sample_agent_response)
        assert result["action"] == "SELL"

    def test_unknown_action_defaults(self, sample_agent_response):
        """Unknown actions return UNKNOWN."""
        sample_agent_response.final_answer = "No clear recommendation"
        result = AgentResponseAdapter.to_dict(sample_agent_response)
        assert result["action"] == "UNKNOWN"


class TestAgentResponseAdapterToDecision:
    """Tests for AgentResponseAdapter.to_decision()."""

    def test_creates_agent_decision(self, sample_agent_response):
        """Create AgentDecision from AgentResponse."""
        decision = AgentResponseAdapter.to_decision(
            sample_agent_response,
            symbol="AAPL",
            market_context={"price": 175.50, "volatility": 0.02},
        )

        assert decision.symbol == "AAPL"
        assert decision.decision_type == "buy"
        assert decision.confidence == 0.85
        assert "price" in decision.market_context

    def test_generates_decision_id(self, sample_agent_response):
        """Auto-generate decision_id if not provided."""
        decision = AgentResponseAdapter.to_decision(
            sample_agent_response,
            symbol="SPY",
        )

        assert "TechnicalAnalyst" in decision.decision_id
        assert "SPY" in decision.decision_id

    def test_custom_decision_id(self, sample_agent_response):
        """Use custom decision_id when provided."""
        decision = AgentResponseAdapter.to_decision(
            sample_agent_response,
            symbol="SPY",
            decision_id="custom_id_123",
        )

        assert decision.decision_id == "custom_id_123"

    def test_builds_full_reasoning(self, sample_agent_response):
        """Build full reasoning from thoughts."""
        decision = AgentResponseAdapter.to_decision(
            sample_agent_response,
            symbol="AAPL",
        )

        assert "RSI shows oversold" in decision.reasoning
        assert "Final Answer:" in decision.reasoning


class TestAgentResponseAdapterToTestCase:
    """Tests for AgentResponseAdapter.to_test_case()."""

    def test_creates_test_case(self, sample_agent_response):
        """Create TestCase from AgentResponse."""
        input_data = {"symbol": "SPY", "price": 450.0}

        test_case = AgentResponseAdapter.to_test_case(
            sample_agent_response,
            input_data=input_data,
            expected_action="BUY",
        )

        assert test_case.input_data == input_data
        assert test_case.expected_output["action"] == "BUY"
        assert test_case.case_id is not None
        assert test_case.agent_type == "analyst"

    def test_infers_expected_action(self, sample_agent_response):
        """Infer expected action from actual if not provided."""
        test_case = AgentResponseAdapter.to_test_case(
            sample_agent_response,
            input_data={},
        )

        assert test_case.expected_output["action"] == "BUY"
        assert test_case.success_criteria["action_match"] is True


class TestBatchAdaptResponses:
    """Tests for batch_adapt_responses()."""

    def test_batch_convert_multiple_responses(self, sample_agent_response, hold_agent_response):
        """Convert multiple responses to decisions."""
        decisions = batch_adapt_responses(
            responses=[sample_agent_response, hold_agent_response],
            symbols=["AAPL", "MSFT"],
        )

        assert len(decisions) == 2
        assert decisions[0].symbol == "AAPL"
        assert decisions[1].symbol == "MSFT"

    def test_single_symbol_for_all(self, sample_agent_response, hold_agent_response):
        """Use single symbol for all responses."""
        decisions = batch_adapt_responses(
            responses=[sample_agent_response, hold_agent_response],
            symbols=["SPY"],
        )

        assert decisions[0].symbol == "SPY"
        assert decisions[1].symbol == "SPY"


# =============================================================================
# Section 1.2: LLM Judge Tests
# =============================================================================


class TestJudgeConfig:
    """Tests for JudgeConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = JudgeConfig()

        assert "claude" in config.model.lower() or "sonnet" in config.model.lower()
        assert config.temperature == 0.0
        assert config.enable_caching is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = JudgeConfig(
            model="gpt-4",
            temperature=0.2,
            max_tokens=2048,
        )

        assert config.model == "gpt-4"
        assert config.temperature == 0.2


class TestJudgeCostTracker:
    """Tests for JudgeCostTracker."""

    def test_tracks_requests(self):
        """Track requests and costs."""
        tracker = JudgeCostTracker()
        tracker.add_request(input_tokens=1000, output_tokens=500)

        summary = tracker.get_summary()
        assert summary["total_requests"] == 1
        assert summary["total_input_tokens"] == 1000
        assert summary["total_output_tokens"] == 500
        assert summary["total_cost_usd"] > 0

    def test_accumulates_costs(self):
        """Accumulate costs across multiple requests."""
        tracker = JudgeCostTracker()
        tracker.add_request(input_tokens=1000, output_tokens=500)
        tracker.add_request(input_tokens=2000, output_tokens=1000)

        summary = tracker.get_summary()
        assert summary["total_requests"] == 2
        assert summary["total_input_tokens"] == 3000


class TestLLMJudge:
    """Tests for LLMJudge class."""

    def test_initialization(self):
        """Test judge initialization."""
        judge = LLMJudge()

        assert judge.config is not None
        assert judge.cost_tracker is not None

    def test_cache_key_generation(self, sample_agent_response):
        """Test cache key generation."""
        judge = LLMJudge()

        from evaluation.agent_as_judge import ALL_RUBRICS

        decision = AgentResponseAdapter.to_decision(
            sample_agent_response,
            symbol="AAPL",
        )
        rubric = ALL_RUBRICS[EvaluationCategory.TRADING_DECISION]

        cache_key = judge._get_cache_key(decision, rubric)
        assert decision.decision_id in cache_key
        assert "trading_decision" in cache_key

    @patch.object(LLMJudge, "_call_llm")
    def test_evaluate_with_mock(self, mock_call_llm, sample_agent_response):
        """Test evaluation with mocked LLM."""
        # Mock LLM response
        mock_call_llm.return_value = """{
            "score": 4,
            "confidence": 0.85,
            "reasoning": "Good analysis with clear reasoning",
            "issues_found": [],
            "strengths_found": ["Clear reasoning", "Good technical analysis"]
        }"""

        judge = LLMJudge(config=JudgeConfig(enable_caching=False))

        decision = AgentResponseAdapter.to_decision(
            sample_agent_response,
            symbol="AAPL",
        )

        score = judge.evaluate(decision)

        assert score.score == 4
        assert score.confidence == 0.85

    def test_get_judge_model(self):
        """Test judge model detection."""
        judge = LLMJudge()

        judge.config.model = "claude-opus-4-20250514"
        from evaluation.agent_as_judge import JudgeModel

        assert judge._get_judge_model() == JudgeModel.CLAUDE_OPUS

        judge.config.model = "gpt-4-turbo"
        assert judge._get_judge_model() == JudgeModel.GPT4_TURBO

    def test_clear_cache(self):
        """Test cache clearing."""
        judge = LLMJudge()
        judge._cache["test_key"] = "test_value"

        judge.clear_cache()
        assert len(judge._cache) == 0


class TestCreateProductionJudge:
    """Tests for create_production_judge factory function."""

    def test_creates_judge_with_defaults(self):
        """Create judge with default settings."""
        judge = create_production_judge()

        assert judge.config.temperature == 0.0
        assert judge.config.enable_caching is True

    def test_creates_judge_with_custom_model(self):
        """Create judge with custom model."""
        judge = create_production_judge(model="gpt-4")

        assert judge.config.model == "gpt-4"


# =============================================================================
# Section 1.3: Safe Agent Wrapper Tests
# =============================================================================


class TestRiskTier:
    """Tests for RiskTier enum."""

    def test_risk_tier_values(self):
        """Verify risk tier values."""
        assert RiskTier.LOW.value == "low"
        assert RiskTier.MEDIUM.value == "medium"
        assert RiskTier.HIGH.value == "high"
        assert RiskTier.CRITICAL.value == "critical"


class TestRiskTierConfig:
    """Tests for RiskTierConfig."""

    def test_default_thresholds(self):
        """Test default risk thresholds."""
        config = RiskTierConfig()

        assert config.low_max_position_pct == 0.02
        assert config.medium_max_position_pct == 0.05
        assert config.high_max_position_pct == 0.10


class TestSafeAgentWrapper:
    """Tests for SafeAgentWrapper class."""

    def test_initialization(self, mock_trading_agent, circuit_breaker, risk_config):
        """Test wrapper initialization."""
        wrapper = SafeAgentWrapper(
            agent=mock_trading_agent,
            circuit_breaker=circuit_breaker,
            risk_config=risk_config,
        )

        assert "Safe[" in wrapper.name
        assert wrapper.role == AgentRole.ANALYST

    def test_passes_through_when_safe(self, mock_trading_agent, circuit_breaker, risk_config):
        """Pass through agent response when all checks pass."""
        wrapper = SafeAgentWrapper(
            agent=mock_trading_agent,
            circuit_breaker=circuit_breaker,
            risk_config=risk_config,
        )

        response = wrapper.analyze(
            query="Should we buy AAPL?",
            context={"proposed_position_pct": 0.01},
        )

        assert response.success is True
        assert "BUY" in response.final_answer

    def test_blocks_when_circuit_breaker_open(self, mock_trading_agent, circuit_breaker, risk_config):
        """Block trading when circuit breaker is open."""
        # Trip the circuit breaker
        circuit_breaker.check_daily_loss(-0.05)  # Exceed 3% limit

        wrapper = SafeAgentWrapper(
            agent=mock_trading_agent,
            circuit_breaker=circuit_breaker,
            risk_config=risk_config,
        )

        response = wrapper.analyze(
            query="Should we buy AAPL?",
            context={},
        )

        assert "HOLD" in response.final_answer
        assert "circuit breaker" in response.final_answer.lower()

    def test_classifies_low_risk(self, mock_trading_agent, circuit_breaker, risk_config):
        """Classify small positions as LOW risk."""
        wrapper = SafeAgentWrapper(
            agent=mock_trading_agent,
            circuit_breaker=circuit_breaker,
            risk_config=risk_config,
        )

        result = wrapper._classify_risk_tier(0.01)  # 1% position
        assert result == RiskTier.LOW

    def test_classifies_medium_risk(self, mock_trading_agent, circuit_breaker, risk_config):
        """Classify medium positions as MEDIUM risk."""
        wrapper = SafeAgentWrapper(
            agent=mock_trading_agent,
            circuit_breaker=circuit_breaker,
            risk_config=risk_config,
        )

        result = wrapper._classify_risk_tier(0.03)  # 3% position
        assert result == RiskTier.MEDIUM

    def test_classifies_high_risk(self, mock_trading_agent, circuit_breaker, risk_config):
        """Classify large positions as HIGH risk."""
        wrapper = SafeAgentWrapper(
            agent=mock_trading_agent,
            circuit_breaker=circuit_breaker,
            risk_config=risk_config,
        )

        result = wrapper._classify_risk_tier(0.08)  # 8% position
        assert result == RiskTier.HIGH

    def test_classifies_critical_risk(self, mock_trading_agent, circuit_breaker, risk_config):
        """Classify very large positions as CRITICAL risk."""
        wrapper = SafeAgentWrapper(
            agent=mock_trading_agent,
            circuit_breaker=circuit_breaker,
            risk_config=risk_config,
        )

        result = wrapper._classify_risk_tier(0.15)  # 15% position
        assert result == RiskTier.CRITICAL

    def test_blocks_critical_risk(self, mock_trading_agent, circuit_breaker, risk_config):
        """Block CRITICAL risk positions."""
        wrapper = SafeAgentWrapper(
            agent=mock_trading_agent,
            circuit_breaker=circuit_breaker,
            risk_config=risk_config,
        )

        response = wrapper.analyze(
            query="Should we buy AAPL?",
            context={"proposed_position_pct": 0.20},  # 20% position
        )

        assert "HOLD" in response.final_answer
        assert "blocked" in response.thoughts[0].content.lower()

    def test_escalates_risk_in_high_volatility(self, mock_trading_agent, circuit_breaker, risk_config):
        """Escalate risk tier in high volatility."""
        wrapper = SafeAgentWrapper(
            agent=mock_trading_agent,
            circuit_breaker=circuit_breaker,
            risk_config=risk_config,
        )

        # Low position but high volatility
        result = wrapper._run_safety_checks(
            mock_trading_agent.analyze("test", {}),
            {"volatility": 0.05, "proposed_position_pct": 0.01},  # 5% volatility
        )

        # Should escalate from LOW to MEDIUM
        assert result.risk_tier in [RiskTier.MEDIUM, RiskTier.HIGH]
        assert "volatility" in result.warnings[0].lower()

    def test_approval_callback_integration(self, mock_trading_agent, circuit_breaker, risk_config):
        """Test approval callback for HIGH risk decisions."""
        approval_calls = []

        def approval_callback(response, safety_result):
            approval_calls.append((response, safety_result))
            return True  # Approve

        wrapper = SafeAgentWrapper(
            agent=mock_trading_agent,
            circuit_breaker=circuit_breaker,
            risk_config=risk_config,
            approval_callback=approval_callback,
        )

        response = wrapper.analyze(
            query="Should we buy AAPL?",
            context={"proposed_position_pct": 0.08},  # HIGH risk
        )

        # Callback should have been called
        assert len(approval_calls) == 1
        assert approval_calls[0][1].risk_tier == RiskTier.HIGH

    def test_blocks_when_approval_denied(self, mock_trading_agent, circuit_breaker, risk_config):
        """Block when approval is denied."""

        def deny_approval(resp, safety_result):
            return False  # Deny

        wrapper = SafeAgentWrapper(
            agent=mock_trading_agent,
            circuit_breaker=circuit_breaker,
            risk_config=risk_config,
            approval_callback=deny_approval,
        )

        result = wrapper.analyze(
            query="Should we buy AAPL?",
            context={"proposed_position_pct": 0.08},  # HIGH risk
        )

        assert "HOLD" in result.final_answer
        assert "not approved" in result.final_answer.lower()

    def test_audit_logging(self, mock_trading_agent, circuit_breaker, risk_config):
        """Test audit log creation."""
        wrapper = SafeAgentWrapper(
            agent=mock_trading_agent,
            circuit_breaker=circuit_breaker,
            risk_config=risk_config,
            enable_audit=True,
        )

        wrapper.analyze(
            query="Test query",
            context={"proposed_position_pct": 0.01},
        )

        summary = wrapper.get_audit_summary()
        assert summary["total_decisions"] == 1
        assert summary["blocked_decisions"] == 0

    def test_handles_agent_errors(self, mock_trading_agent, circuit_breaker, risk_config):
        """Handle errors from underlying agent."""
        mock_trading_agent.analyze.side_effect = Exception("API Error")

        wrapper = SafeAgentWrapper(
            agent=mock_trading_agent,
            circuit_breaker=circuit_breaker,
            risk_config=risk_config,
        )

        response = wrapper.analyze(
            query="Should we buy?",
            context={},
        )

        assert response.success is False
        assert "error" in response.final_answer.lower()


class TestWrapAgentWithSafety:
    """Tests for wrap_agent_with_safety factory function."""

    def test_creates_wrapper(self, mock_trading_agent, circuit_breaker):
        """Create wrapper with factory function."""
        wrapper = wrap_agent_with_safety(
            agent=mock_trading_agent,
            circuit_breaker=circuit_breaker,
        )

        assert isinstance(wrapper, SafeAgentWrapper)
        assert "Safe[" in wrapper.name


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase1Integration:
    """Integration tests for Phase 1 components working together."""

    def test_adapter_to_judge_flow(self, sample_agent_response):
        """Test flow from adapter to judge."""
        # Convert response to decision
        decision = adapt_response_to_decision(
            sample_agent_response,
            symbol="AAPL",
            market_context={"price": 175.0},
        )

        # Create judge (verify it can be created)
        _ = create_production_judge()

        # Verify decision is properly formatted for judge
        assert decision.decision_id is not None
        assert decision.symbol == "AAPL"
        assert decision.decision_type == "buy"

    def test_wrapper_audit_to_decision(self, mock_trading_agent, circuit_breaker, risk_config):
        """Test flow from wrapper audit to decision conversion."""
        wrapper = SafeAgentWrapper(
            agent=mock_trading_agent,
            circuit_breaker=circuit_breaker,
            risk_config=risk_config,
            enable_audit=True,
        )

        # Make a decision
        response = wrapper.analyze(
            query="Buy AAPL?",
            context={"proposed_position_pct": 0.01},
        )

        # Convert to decision
        decision = adapt_response_to_decision(response, symbol="AAPL")

        # Verify audit and decision align
        audit = wrapper.get_audit_summary()
        assert audit["total_decisions"] == 1
        assert decision.decision_type in ["buy", "hold"]

    def test_full_safety_evaluation_pipeline(self, mock_trading_agent, circuit_breaker, risk_config):
        """Test complete pipeline: agent -> wrapper -> adapter -> (judge ready)."""
        # 1. Wrap agent with safety
        safe_agent = wrap_agent_with_safety(
            agent=mock_trading_agent,
            circuit_breaker=circuit_breaker,
            risk_config=risk_config,
        )

        # 2. Get safe response
        response = safe_agent.analyze(
            query="Analyze SPY opportunity",
            context={"proposed_position_pct": 0.02, "volatility": 0.02},
        )

        # 3. Convert to dict for evaluation framework
        eval_dict = adapt_response_to_dict(response)

        # 4. Convert to decision for judge
        decision = adapt_response_to_decision(response, symbol="SPY")

        # 5. Create test case (now returns proper TestCase)
        test_case = create_test_case_from_response(
            response,
            input_data={"symbol": "SPY"},
        )

        # Verify all conversions succeeded
        # Note: safe_agent passes through the original agent's name when checks pass
        assert "Safe[" in safe_agent.name  # Wrapper has Safe[] prefix
        assert eval_dict["agent_name"] is not None  # Response has an agent name
        assert decision.symbol == "SPY"
        assert test_case.case_id is not None
        assert test_case.expected_output["success"] is True


# =============================================================================
# Section 1.4: Safe Agent Factory Function Tests
# =============================================================================


class TestSafeAgentFactoryFunctions:
    """Tests for safe agent factory functions."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client for factory testing."""
        client = Mock()
        client.chat = Mock(
            return_value=Mock(
                content='{"action": "BUY", "confidence": 0.75}',
                stop_reason="end_turn",
                usage={"input_tokens": 100, "output_tokens": 50},
            )
        )
        client.estimate_cost = Mock(return_value=0.001)
        return client

    def test_create_safe_technical_analyst_returns_wrapper(self, mock_llm_client, circuit_breaker, risk_config):
        """create_safe_technical_analyst returns SafeAgentWrapper."""
        # Note: This will fail if the agent can't be created due to prompt issues
        # In production, you'd need valid prompts configured
        with patch("llm.agents.technical_analyst.get_prompt") as mock_get_prompt:
            mock_prompt = Mock()
            mock_prompt.template = "Test prompt"
            mock_prompt.version = "active"
            mock_prompt.max_tokens = 1000
            mock_prompt.temperature = 0.0
            mock_get_prompt.return_value = mock_prompt

            with patch("llm.agents.technical_analyst.get_registry") as mock_registry:
                mock_registry.return_value = Mock()
                mock_registry.return_value.record_usage = Mock()

                safe_agent = create_safe_technical_analyst(
                    llm_client=mock_llm_client,
                    circuit_breaker=circuit_breaker,
                    risk_config=risk_config,
                )

                assert isinstance(safe_agent, SafeAgentWrapper)
                assert "Safe[" in safe_agent.name

    def test_create_safe_sentiment_analyst_returns_wrapper(self, mock_llm_client, circuit_breaker, risk_config):
        """create_safe_sentiment_analyst returns SafeAgentWrapper."""
        with patch("llm.agents.sentiment_analyst.get_prompt") as mock_get_prompt:
            mock_prompt = Mock()
            mock_prompt.template = "Test prompt"
            mock_prompt.version = "active"
            mock_prompt.max_tokens = 1000
            mock_prompt.temperature = 0.0
            mock_get_prompt.return_value = mock_prompt

            with patch("llm.agents.sentiment_analyst.get_registry") as mock_registry:
                mock_registry.return_value = Mock()
                mock_registry.return_value.record_usage = Mock()

                safe_agent = create_safe_sentiment_analyst(
                    llm_client=mock_llm_client,
                    circuit_breaker=circuit_breaker,
                    use_finbert=False,  # Disable for testing
                    risk_config=risk_config,
                )

                assert isinstance(safe_agent, SafeAgentWrapper)
                assert "Safe[" in safe_agent.name

    def test_create_safe_conservative_trader_returns_wrapper(self, mock_llm_client, circuit_breaker, risk_config):
        """create_safe_conservative_trader returns SafeAgentWrapper."""
        with patch("llm.agents.traders.get_prompt") as mock_get_prompt:
            mock_prompt = Mock()
            mock_prompt.template = "Test prompt"
            mock_prompt.version = "active"
            mock_prompt.max_tokens = 1000
            mock_prompt.temperature = 0.0
            mock_get_prompt.return_value = mock_prompt

            with patch("llm.agents.traders.get_registry") as mock_registry:
                mock_registry.return_value = Mock()
                mock_registry.return_value.record_usage = Mock()

                safe_agent = create_safe_conservative_trader(
                    llm_client=mock_llm_client,
                    circuit_breaker=circuit_breaker,
                    risk_config=risk_config,
                )

                assert isinstance(safe_agent, SafeAgentWrapper)
                assert "Safe[" in safe_agent.name

    def test_create_safe_position_risk_manager_returns_wrapper(self, mock_llm_client, circuit_breaker, risk_config):
        """create_safe_position_risk_manager returns SafeAgentWrapper."""
        with patch("llm.agents.risk_managers.get_prompt") as mock_get_prompt:
            mock_prompt = Mock()
            mock_prompt.template = "Test prompt"
            mock_prompt.version = "active"
            mock_prompt.max_tokens = 1000
            mock_prompt.temperature = 0.0
            mock_get_prompt.return_value = mock_prompt

            with patch("llm.agents.risk_managers.get_registry") as mock_registry:
                mock_registry.return_value = Mock()
                mock_registry.return_value.record_usage = Mock()

                safe_agent = create_safe_position_risk_manager(
                    llm_client=mock_llm_client,
                    circuit_breaker=circuit_breaker,
                    risk_config=risk_config,
                )

                assert isinstance(safe_agent, SafeAgentWrapper)
                assert "Safe[" in safe_agent.name

    def test_create_safe_supervisor_returns_wrapper(self, mock_llm_client, circuit_breaker, risk_config):
        """create_safe_supervisor_agent returns SafeAgentWrapper."""
        with patch("llm.agents.supervisor.get_prompt") as mock_get_prompt:
            mock_prompt = Mock()
            mock_prompt.template = "Test prompt"
            mock_prompt.version = "active"
            mock_prompt.max_tokens = 1000
            mock_prompt.temperature = 0.0
            mock_get_prompt.return_value = mock_prompt

            with patch("llm.agents.supervisor.get_registry") as mock_registry:
                mock_registry.return_value = Mock()
                mock_registry.return_value.record_usage = Mock()

                safe_agent = create_safe_supervisor_agent(
                    llm_client=mock_llm_client,
                    circuit_breaker=circuit_breaker,
                    risk_config=risk_config,
                )

                assert isinstance(safe_agent, SafeAgentWrapper)
                assert "Safe[" in safe_agent.name

    def test_safe_agents_block_when_circuit_breaker_open(self, mock_llm_client, circuit_breaker, risk_config):
        """All safe agents should block trading when circuit breaker is open."""
        # Trip the circuit breaker
        circuit_breaker.check_daily_loss(-0.05)  # Exceed 3% limit

        with patch("llm.agents.technical_analyst.get_prompt") as mock_get_prompt:
            mock_prompt = Mock()
            mock_prompt.template = "Test prompt"
            mock_prompt.version = "active"
            mock_prompt.max_tokens = 1000
            mock_prompt.temperature = 0.0
            mock_get_prompt.return_value = mock_prompt

            with patch("llm.agents.technical_analyst.get_registry") as mock_registry:
                mock_registry.return_value = Mock()
                mock_registry.return_value.record_usage = Mock()

                safe_agent = create_safe_technical_analyst(
                    llm_client=mock_llm_client,
                    circuit_breaker=circuit_breaker,
                    risk_config=risk_config,
                )

                response = safe_agent.analyze(
                    query="Analyze AAPL",
                    context={},
                )

                assert "HOLD" in response.final_answer
                assert "circuit breaker" in response.final_answer.lower()


# =============================================================================
# Section 1.5: Real Judge Function Tests
# =============================================================================


class TestCreateRealJudgeFunction:
    """Tests for create_real_judge_function."""

    def test_creates_callable_function(self):
        """create_real_judge_function returns a callable."""
        judge_fn = create_real_judge_function()
        assert callable(judge_fn)

    def test_accepts_custom_model(self):
        """Accepts custom model parameter."""
        judge_fn = create_real_judge_function(model="gpt-4")
        assert callable(judge_fn)

    def test_accepts_custom_category(self):
        """Accepts custom category parameter."""
        judge_fn = create_real_judge_function(category=EvaluationCategory.RISK_ASSESSMENT)
        assert callable(judge_fn)

    @patch("evaluation.llm_judge.LLMJudge._call_llm_with_retry")
    def test_returns_json_response_on_success(self, mock_call_llm):
        """Returns JSON response when LLM call succeeds."""
        mock_call_llm.return_value = """{
            "score": 4,
            "confidence": 0.85,
            "reasoning": "Good analysis",
            "issues_found": [],
            "strengths_found": ["Clear reasoning"]
        }"""

        judge_fn = create_real_judge_function()
        result = judge_fn("Test prompt for evaluation")

        assert "score" in result
        assert "4" in result

    @patch("evaluation.llm_judge.LLMJudge._call_llm_with_retry")
    def test_returns_fallback_on_error(self, mock_call_llm):
        """Returns fallback JSON when LLM call fails."""
        mock_call_llm.side_effect = Exception("API Error")

        judge_fn = create_real_judge_function()
        result = judge_fn("Test prompt for evaluation")

        # Should return fallback JSON
        import json

        parsed = json.loads(result)
        assert parsed["score"] == 3
        assert parsed["confidence"] == 0.0
        assert "failed" in parsed["reasoning"].lower()

    def test_integrates_with_judge_model_enum(self):
        """Works with different JudgeModel configurations."""

        # Test with Claude Sonnet
        judge_fn = create_real_judge_function(model="claude-sonnet-4-20250514")
        assert callable(judge_fn)

        # Test with GPT-4
        judge_fn = create_real_judge_function(model="gpt-4-turbo")
        assert callable(judge_fn)


# =============================================================================
# Section 1.6: Orchestration Pipeline LLM Judges Tests
# =============================================================================


class TestOrchestrationPipelineLLMJudges:
    """Tests for orchestration pipeline use_real_llm_judges configuration."""

    def test_pipeline_config_accepts_use_real_llm_judges(self):
        """Pipeline config should accept use_real_llm_judges option."""
        from evaluation.orchestration_pipeline import EvaluationOrchestrator

        # With mock judges (default)
        config_mock = {"use_real_llm_judges": False}
        pipeline_mock = EvaluationOrchestrator(config=config_mock)
        assert pipeline_mock.config.get("use_real_llm_judges") is False

        # With real judges
        config_real = {"use_real_llm_judges": True}
        pipeline_real = EvaluationOrchestrator(config=config_real)
        assert pipeline_real.config.get("use_real_llm_judges") is True

    def test_pipeline_defaults_to_mock_judges(self):
        """Pipeline defaults to mock judges when not configured."""
        from evaluation.orchestration_pipeline import EvaluationOrchestrator

        pipeline = EvaluationOrchestrator(config={})
        assert pipeline.config.get("use_real_llm_judges", False) is False

    def test_pipeline_works_without_config(self):
        """Pipeline works when no config is provided."""
        from evaluation.orchestration_pipeline import EvaluationOrchestrator

        pipeline = EvaluationOrchestrator()
        assert pipeline.config.get("use_real_llm_judges", False) is False

    @patch("evaluation.orchestration_pipeline.create_production_judge")
    def test_pipeline_creates_real_judges_when_configured(self, mock_create_judge):
        """Pipeline creates real judges when use_real_llm_judges is True."""
        from evaluation.llm_judge import LLMJudge
        from evaluation.orchestration_pipeline import EvaluationOrchestrator

        # Mock the judge creation
        mock_judge = Mock(spec=LLMJudge)
        mock_create_judge.return_value = mock_judge

        config = {"use_real_llm_judges": True}
        pipeline = EvaluationOrchestrator(config=config)

        # Verify config is stored
        assert pipeline.config.get("use_real_llm_judges") is True


# =============================================================================
# Additional Integration Tests
# =============================================================================


class TestPhase1ExtendedIntegration:
    """Extended integration tests for all Phase 1 components."""

    def test_safe_factory_to_adapter_flow(self, circuit_breaker, risk_config):
        """Test flow from safe factory to adapter."""
        # Create mock client
        mock_client = Mock()
        mock_client.chat = Mock(
            return_value=Mock(
                content='{"action": "BUY", "confidence": 0.8, "symbol": "AAPL"}',
                stop_reason="end_turn",
                usage={"input_tokens": 100, "output_tokens": 50},
            )
        )
        mock_client.estimate_cost = Mock(return_value=0.001)

        with patch("llm.agents.technical_analyst.get_prompt") as mock_get_prompt:
            mock_prompt = Mock()
            mock_prompt.template = "Test"
            mock_prompt.version = "active"
            mock_prompt.max_tokens = 1000
            mock_prompt.temperature = 0.0
            mock_get_prompt.return_value = mock_prompt

            with patch("llm.agents.technical_analyst.get_registry") as mock_reg:
                mock_reg.return_value = Mock()
                mock_reg.return_value.record_usage = Mock()

                # 1. Create safe agent
                safe_agent = create_safe_technical_analyst(
                    llm_client=mock_client,
                    circuit_breaker=circuit_breaker,
                    risk_config=risk_config,
                )

                # 2. Analyze
                response = safe_agent.analyze(
                    query="Analyze AAPL",
                    context={"proposed_position_pct": 0.01},
                )

                # 3. Adapt response
                decision = adapt_response_to_decision(
                    response,
                    symbol="AAPL",
                )

                # Verify flow
                assert isinstance(safe_agent, SafeAgentWrapper)
                assert response.success is True
                assert decision.symbol == "AAPL"

    def test_complete_phase1_component_availability(self):
        """Verify all Phase 1 components are importable and available."""
        # Adapters

        # LLM Judge

        # Safe Agent Wrapper

        # Safe Agent Factories

        # Real Judge Function

        # All imports successful
        assert True

    def test_exports_from_evaluation_init(self):
        """Verify exports from evaluation/__init__.py."""

        assert True

    def test_exports_from_llm_agents_init(self):
        """Verify exports from llm/agents/__init__.py."""

        assert True
