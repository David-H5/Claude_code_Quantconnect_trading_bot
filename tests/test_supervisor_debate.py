"""
Tests for Supervisor Debate Integration

Tests verify the supervisor correctly:
- Triggers debates based on criteria
- Integrates debate results into analysis
- Tracks debate history
- Works with factory functions
"""

from unittest.mock import MagicMock, patch

import pytest

from llm.agents.debate_mechanism import (
    BullBearDebate,
    DebateConfig,
)
from llm.agents.supervisor import (
    DebateTriggerReason,
    SupervisorAgent,
    create_supervisor_agent,
)


class MockLLMClient:
    """Mock LLM client for testing."""

    def chat(self, model, messages, system, max_tokens, temperature):
        """Mock chat response."""
        response = MagicMock()
        response.content = '{"action": "BUY", "confidence": 0.75, "reasoning": "Test"}'
        response.stop_reason = "end_turn"
        response.usage = {"input_tokens": 100, "output_tokens": 50}
        return response

    def estimate_cost(self, model, input_tokens, output_tokens):
        """Mock cost estimation."""
        return 0.001


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def debate_mechanism():
    """Create a debate mechanism for testing."""
    config = DebateConfig(max_rounds=2)
    return BullBearDebate(config=config)


@pytest.fixture
def supervisor_with_debate(mock_llm_client, debate_mechanism):
    """Create a supervisor with debate mechanism."""
    with patch("llm.agents.supervisor.get_prompt") as mock_get_prompt:
        mock_prompt = MagicMock()
        mock_prompt.template = "You are a supervisor."
        mock_prompt.version = "v1.0"
        mock_prompt.max_tokens = 1000
        mock_prompt.temperature = 0.7
        mock_get_prompt.return_value = mock_prompt

        with patch("llm.agents.supervisor.get_registry") as mock_registry:
            mock_registry.return_value = MagicMock()

            return SupervisorAgent(
                llm_client=mock_llm_client,
                debate_mechanism=debate_mechanism,
                debate_threshold=0.10,
                min_debate_confidence=0.70,
            )


@pytest.fixture
def supervisor_without_debate(mock_llm_client):
    """Create a supervisor without debate mechanism."""
    with patch("llm.agents.supervisor.get_prompt") as mock_get_prompt:
        mock_prompt = MagicMock()
        mock_prompt.template = "You are a supervisor."
        mock_prompt.version = "v1.0"
        mock_prompt.max_tokens = 1000
        mock_prompt.temperature = 0.7
        mock_get_prompt.return_value = mock_prompt

        with patch("llm.agents.supervisor.get_registry") as mock_registry:
            mock_registry.return_value = MagicMock()

            return SupervisorAgent(
                llm_client=mock_llm_client,
                debate_mechanism=None,
            )


class TestDebateTriggerLogic:
    """Tests for debate triggering logic."""

    def test_no_debate_when_disabled(self, supervisor_without_debate):
        """No debate when mechanism is None."""
        opportunity = {"position_size_pct": 0.50}  # 50% would normally trigger
        context = {}

        should, reason = supervisor_without_debate.should_debate(opportunity, context)

        assert should is False
        assert reason is None

    def test_trigger_on_high_position_size(self, supervisor_with_debate):
        """Triggers debate for high position size."""
        opportunity = {"position_size_pct": 0.15}  # 15% > 10% threshold
        context = {}

        should, reason = supervisor_with_debate.should_debate(opportunity, context)

        assert should is True
        assert reason == DebateTriggerReason.HIGH_POSITION_SIZE

    def test_trigger_on_low_confidence(self, supervisor_with_debate):
        """Triggers debate for low confidence."""
        opportunity = {"position_size_pct": 0.05}
        context = {}

        should, reason = supervisor_with_debate.should_debate(opportunity, context, initial_confidence=0.60)

        assert should is True
        assert reason == DebateTriggerReason.LOW_CONFIDENCE

    def test_trigger_on_conflicting_signals(self, supervisor_with_debate):
        """Triggers debate for conflicting analyst signals."""
        opportunity = {"position_size_pct": 0.05}
        context = {
            "analyst_signals": {
                "technical": 0.85,
                "fundamental": 0.40,  # 0.45 difference > 0.30 threshold
            }
        }

        should, reason = supervisor_with_debate.should_debate(opportunity, context, initial_confidence=0.80)

        assert should is True
        assert reason == DebateTriggerReason.CONFLICTING_SIGNALS

    def test_trigger_on_earnings(self, supervisor_with_debate):
        """Triggers debate for earnings events."""
        opportunity = {"position_size_pct": 0.05, "has_earnings": True}
        context = {}

        should, reason = supervisor_with_debate.should_debate(opportunity, context, initial_confidence=0.80)

        assert should is True
        assert reason == DebateTriggerReason.HIGH_IMPACT_EVENT

    def test_trigger_on_major_news(self, supervisor_with_debate):
        """Triggers debate for major news events."""
        opportunity = {"position_size_pct": 0.05, "has_major_news": True}
        context = {}

        should, reason = supervisor_with_debate.should_debate(opportunity, context, initial_confidence=0.80)

        assert should is True
        assert reason == DebateTriggerReason.HIGH_IMPACT_EVENT

    def test_trigger_on_unusual_volatility(self, supervisor_with_debate):
        """Triggers debate for unusual volatility."""
        opportunity = {"position_size_pct": 0.05}
        context = {"unusual_volatility": True}

        should, reason = supervisor_with_debate.should_debate(opportunity, context, initial_confidence=0.80)

        assert should is True
        assert reason == DebateTriggerReason.UNUSUAL_MARKET

    def test_trigger_on_market_stress(self, supervisor_with_debate):
        """Triggers debate for market stress."""
        opportunity = {"position_size_pct": 0.05}
        context = {"market_stress": True}

        should, reason = supervisor_with_debate.should_debate(opportunity, context, initial_confidence=0.80)

        assert should is True
        assert reason == DebateTriggerReason.UNUSUAL_MARKET

    def test_trigger_on_manual_request(self, supervisor_with_debate):
        """Triggers debate on manual request."""
        opportunity = {"position_size_pct": 0.05}
        context = {"force_debate": True}

        should, reason = supervisor_with_debate.should_debate(opportunity, context, initial_confidence=0.80)

        assert should is True
        assert reason == DebateTriggerReason.MANUAL_REQUEST

    def test_no_trigger_normal_conditions(self, supervisor_with_debate):
        """Does not trigger for normal conditions."""
        opportunity = {"position_size_pct": 0.05}
        context = {}

        should, reason = supervisor_with_debate.should_debate(opportunity, context, initial_confidence=0.85)

        assert should is False
        assert reason is None


class TestAnalyzeWithDebate:
    """Tests for analyze_with_debate method."""

    def test_no_debate_when_not_needed(self, supervisor_with_debate):
        """Returns initial analysis when debate not needed."""
        context = {"opportunity": {"position_size_pct": 0.05}}

        response = supervisor_with_debate.analyze_with_debate(
            "Should we buy SPY?",
            context,
        )

        assert response.success
        # No debate history added
        assert len(supervisor_with_debate.debate_history) == 0

    def test_debate_when_forced(self, supervisor_with_debate):
        """Runs debate when forced."""
        context = {"opportunity": {"symbol": "SPY"}}

        response = supervisor_with_debate.analyze_with_debate(
            "Should we buy SPY?",
            context,
            force_debate=True,
        )

        assert response.success
        # Debate history added
        assert len(supervisor_with_debate.debate_history) == 1
        assert supervisor_with_debate.debate_history[0]["trigger_reason"] == "manual_request"

    def test_debate_adds_insights_to_response(self, supervisor_with_debate):
        """Debate adds insights to response."""
        context = {"opportunity": {"symbol": "SPY"}}

        response = supervisor_with_debate.analyze_with_debate(
            "Should we buy SPY?",
            context,
            force_debate=True,
        )

        # Should have debate-related content
        assert "debate" in response.final_answer.lower() or len(response.thoughts) > 1


class TestDebateHistory:
    """Tests for debate history tracking."""

    def test_history_empty_initially(self, supervisor_with_debate):
        """Debate history is empty initially."""
        assert len(supervisor_with_debate.debate_history) == 0

    def test_history_records_debates(self, supervisor_with_debate):
        """Debate history records debates."""
        context = {"opportunity": {"symbol": "SPY"}}

        supervisor_with_debate.analyze_with_debate("Query 1", context, force_debate=True)
        supervisor_with_debate.analyze_with_debate("Query 2", context, force_debate=True)

        history = supervisor_with_debate.get_debate_history()
        assert len(history) == 2
        assert history[0]["query"] == "Query 1"
        assert history[1]["query"] == "Query 2"

    def test_history_includes_metadata(self, supervisor_with_debate):
        """Debate history includes relevant metadata."""
        context = {"opportunity": {"symbol": "SPY"}}

        supervisor_with_debate.analyze_with_debate("Test query", context, force_debate=True)

        history = supervisor_with_debate.get_debate_history()
        record = history[0]

        assert "timestamp" in record
        assert "query" in record
        assert "trigger_reason" in record
        assert "debate_id" in record
        assert "outcome" in record
        assert "consensus_confidence" in record

    def test_clear_history(self, supervisor_with_debate):
        """Can clear debate history."""
        context = {"opportunity": {"symbol": "SPY"}}

        supervisor_with_debate.analyze_with_debate("Test query", context, force_debate=True)
        assert len(supervisor_with_debate.debate_history) > 0

        supervisor_with_debate.clear_debate_history()
        assert len(supervisor_with_debate.debate_history) == 0


class TestCustomThresholds:
    """Tests for custom debate thresholds."""

    def test_custom_position_threshold(self, mock_llm_client, debate_mechanism):
        """Custom position size threshold works."""
        with patch("llm.agents.supervisor.get_prompt") as mock_get_prompt:
            mock_prompt = MagicMock()
            mock_prompt.template = "You are a supervisor."
            mock_prompt.version = "v1.0"
            mock_prompt.max_tokens = 1000
            mock_prompt.temperature = 0.7
            mock_get_prompt.return_value = mock_prompt

            with patch("llm.agents.supervisor.get_registry") as mock_registry:
                mock_registry.return_value = MagicMock()

                supervisor = SupervisorAgent(
                    llm_client=mock_llm_client,
                    debate_mechanism=debate_mechanism,
                    debate_threshold=0.20,  # 20% threshold
                )

                # 15% should NOT trigger with 20% threshold
                should, _ = supervisor.should_debate({"position_size_pct": 0.15}, {}, 0.80)
                assert should is False

                # 25% should trigger
                should, reason = supervisor.should_debate({"position_size_pct": 0.25}, {}, 0.80)
                assert should is True
                assert reason == DebateTriggerReason.HIGH_POSITION_SIZE

    def test_custom_confidence_threshold(self, mock_llm_client, debate_mechanism):
        """Custom confidence threshold works."""
        with patch("llm.agents.supervisor.get_prompt") as mock_get_prompt:
            mock_prompt = MagicMock()
            mock_prompt.template = "You are a supervisor."
            mock_prompt.version = "v1.0"
            mock_prompt.max_tokens = 1000
            mock_prompt.temperature = 0.7
            mock_get_prompt.return_value = mock_prompt

            with patch("llm.agents.supervisor.get_registry") as mock_registry:
                mock_registry.return_value = MagicMock()

                supervisor = SupervisorAgent(
                    llm_client=mock_llm_client,
                    debate_mechanism=debate_mechanism,
                    min_debate_confidence=0.50,  # 50% threshold
                )

                # 60% confidence should NOT trigger with 50% threshold
                should, _ = supervisor.should_debate({"position_size_pct": 0.05}, {}, 0.60)
                assert should is False

                # 40% confidence should trigger
                should, reason = supervisor.should_debate({"position_size_pct": 0.05}, {}, 0.40)
                assert should is True
                assert reason == DebateTriggerReason.LOW_CONFIDENCE


class TestSupervisorDebateFactoryFunctions:
    """Tests for factory functions."""

    def test_create_supervisor_agent_without_debate(self, mock_llm_client):
        """Factory creates supervisor without debate."""
        with patch("llm.agents.supervisor.get_prompt") as mock_get_prompt:
            mock_prompt = MagicMock()
            mock_prompt.template = "You are a supervisor."
            mock_prompt.version = "v1.0"
            mock_prompt.max_tokens = 1000
            mock_prompt.temperature = 0.7
            mock_get_prompt.return_value = mock_prompt

            with patch("llm.agents.supervisor.get_registry") as mock_registry:
                mock_registry.return_value = MagicMock()

                supervisor = create_supervisor_agent(
                    llm_client=mock_llm_client,
                )

                assert supervisor.debate is None

    def test_create_supervisor_agent_with_debate(self, mock_llm_client, debate_mechanism):
        """Factory creates supervisor with debate."""
        with patch("llm.agents.supervisor.get_prompt") as mock_get_prompt:
            mock_prompt = MagicMock()
            mock_prompt.template = "You are a supervisor."
            mock_prompt.version = "v1.0"
            mock_prompt.max_tokens = 1000
            mock_prompt.temperature = 0.7
            mock_get_prompt.return_value = mock_prompt

            with patch("llm.agents.supervisor.get_registry") as mock_registry:
                mock_registry.return_value = MagicMock()

                supervisor = create_supervisor_agent(
                    llm_client=mock_llm_client,
                    debate_mechanism=debate_mechanism,
                    debate_threshold=0.15,
                    min_debate_confidence=0.65,
                )

                assert supervisor.debate is not None
                assert supervisor.debate_threshold == 0.15
                assert supervisor.min_debate_confidence == 0.65


class TestDebateTriggerReasonEnum:
    """Tests for DebateTriggerReason enum."""

    def test_all_reasons_have_values(self):
        """All trigger reasons have string values."""
        for reason in DebateTriggerReason:
            assert isinstance(reason.value, str)
            assert len(reason.value) > 0

    def test_reason_values_unique(self):
        """All trigger reason values are unique."""
        values = [r.value for r in DebateTriggerReason]
        assert len(values) == len(set(values))
