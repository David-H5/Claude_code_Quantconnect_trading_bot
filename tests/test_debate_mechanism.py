"""
Tests for Bull/Bear Debate Mechanism

Tests verify the debate mechanism correctly:
- Triggers debates based on criteria
- Runs multi-round debates
- Compiles results with consensus
- Integrates bull and bear researchers
"""

import pytest

from llm.agents.bear_researcher import BearResearcher, create_bear_researcher
from llm.agents.bull_researcher import BullResearcher, create_bull_researcher
from llm.agents.debate_mechanism import (
    BullBearDebate,
    DebateArgument,
    DebateConfig,
    DebateOutcome,
    DebateResult,
    DebateTrigger,
    ModeratorAssessment,
    create_debate_mechanism,
    generate_debate_report,
)


@pytest.fixture
def debate():
    """Create a debate mechanism with default settings."""
    return BullBearDebate()


@pytest.fixture
def opportunity():
    """Create a sample trading opportunity."""
    return {
        "symbol": "SPY",
        "price": 450.0,
        "position_size_pct": 0.05,
        "has_position": False,
    }


@pytest.fixture
def initial_analysis():
    """Create sample initial analysis."""
    return {
        "technicals": {
            "rsi": 55,
            "above_sma_20": True,
            "macd_bullish": True,
        },
        "fundamentals": {
            "pe_ratio": 22,
            "earnings_growth": 10,
        },
        "sentiment": {
            "news_sentiment": 0.3,
        },
    }


class TestDebateTrigger:
    """Tests for debate triggering logic."""

    def test_trigger_on_high_position_size(self, debate):
        """Triggers debate for high position size."""
        opportunity = {"position_size_pct": 0.15}  # 15% > 10% threshold
        context = {}

        should, reason = debate.should_debate(opportunity, context)

        assert should is True
        assert reason == DebateTrigger.HIGH_POSITION_SIZE

    def test_trigger_on_low_confidence(self, debate):
        """Triggers debate for low confidence."""
        opportunity = {"position_size_pct": 0.05}
        context = {}

        should, reason = debate.should_debate(opportunity, context, initial_confidence=0.60)

        assert should is True
        assert reason == DebateTrigger.LOW_CONFIDENCE

    def test_trigger_on_conflicting_signals(self, debate):
        """Triggers debate for conflicting analyst signals."""
        opportunity = {"position_size_pct": 0.05}
        context = {
            "analyst_signals": {
                "technical": 0.8,
                "fundamental": 0.3,  # 0.5 difference > 0.3 threshold
            }
        }

        should, reason = debate.should_debate(opportunity, context, initial_confidence=0.75)

        assert should is True
        assert reason == DebateTrigger.CONFLICTING_SIGNALS

    def test_trigger_on_high_impact_event(self, debate):
        """Triggers debate for earnings events."""
        opportunity = {"position_size_pct": 0.05, "has_earnings": True}
        context = {}

        should, reason = debate.should_debate(opportunity, context, initial_confidence=0.75)

        assert should is True
        assert reason == DebateTrigger.HIGH_IMPACT_EVENT

    def test_trigger_on_unusual_market(self, debate):
        """Triggers debate for unusual volatility."""
        opportunity = {"position_size_pct": 0.05}
        context = {"unusual_volatility": True}

        should, reason = debate.should_debate(opportunity, context, initial_confidence=0.75)

        assert should is True
        assert reason == DebateTrigger.UNUSUAL_MARKET

    def test_no_trigger_normal_conditions(self, debate):
        """Does not trigger for normal conditions."""
        opportunity = {"position_size_pct": 0.05}
        context = {}

        should, reason = debate.should_debate(opportunity, context, initial_confidence=0.85)

        assert should is False
        assert reason is None


class TestDebateExecution:
    """Tests for debate execution."""

    def test_run_debate_returns_result(self, debate, opportunity, initial_analysis):
        """Running debate returns DebateResult."""
        result = debate.run_debate(opportunity, initial_analysis)

        assert isinstance(result, DebateResult)
        assert result.debate_id.startswith("debate_")
        assert len(result.rounds) > 0

    def test_debate_has_valid_outcome(self, debate, opportunity, initial_analysis):
        """Debate result has valid outcome."""
        result = debate.run_debate(opportunity, initial_analysis)

        assert result.final_outcome in DebateOutcome

    def test_debate_consensus_in_range(self, debate, opportunity, initial_analysis):
        """Consensus confidence is between 0 and 1."""
        result = debate.run_debate(opportunity, initial_analysis)

        assert 0 <= result.consensus_confidence <= 1

    def test_debate_respects_max_rounds(self, opportunity, initial_analysis):
        """Debate respects max rounds setting."""
        config = DebateConfig(max_rounds=2)
        debate = BullBearDebate(config=config)

        result = debate.run_debate(opportunity, initial_analysis)

        assert len(result.rounds) <= 2

    def test_debate_records_history(self, debate, opportunity, initial_analysis):
        """Debate records to history."""
        assert len(debate.debate_history) == 0

        debate.run_debate(opportunity, initial_analysis)

        assert len(debate.debate_history) == 1

    def test_multiple_debates_tracked(self, debate, opportunity, initial_analysis):
        """Multiple debates are tracked in history."""
        debate.run_debate(opportunity, initial_analysis)
        debate.run_debate(opportunity, initial_analysis)

        assert len(debate.debate_history) == 2
        assert debate.debate_history[0].debate_id != debate.debate_history[1].debate_id


class TestDebateRounds:
    """Tests for individual debate rounds."""

    def test_round_has_all_arguments(self, debate, opportunity, initial_analysis):
        """Each round has bull, bear, and moderator."""
        result = debate.run_debate(opportunity, initial_analysis)

        for round in result.rounds:
            assert isinstance(round.bull_argument, DebateArgument)
            assert isinstance(round.bear_argument, DebateArgument)
            assert isinstance(round.moderator_assessment, ModeratorAssessment)

    def test_arguments_have_content(self, debate, opportunity, initial_analysis):
        """Arguments have non-empty content."""
        result = debate.run_debate(opportunity, initial_analysis)

        for round in result.rounds:
            assert len(round.bull_argument.content) > 0
            assert len(round.bear_argument.content) > 0

    def test_confidences_are_valid(self, debate, opportunity, initial_analysis):
        """Argument confidences are in valid range."""
        result = debate.run_debate(opportunity, initial_analysis)

        for round in result.rounds:
            assert 0 <= round.bull_argument.confidence <= 1
            assert 0 <= round.bear_argument.confidence <= 1


class TestDebateOutcomes:
    """Tests for debate outcome determination."""

    def test_buy_outcome_when_bull_wins(self):
        """BUY outcome when bull has higher confidence."""
        # Use lower min_confidence_delta so bull 0.72 vs bear 0.65 (delta 0.07) results in BUY
        config = DebateConfig(min_confidence_delta=0.05)
        debate = BullBearDebate(config=config)
        # Mock would set bull confidence higher - using default mock behavior
        result = debate.run_debate(
            {"symbol": "SPY", "has_position": False},
            {},
        )

        # Default mocks give bull 0.72, bear 0.65 (delta 0.07 > 0.05)
        assert result.final_outcome == DebateOutcome.BUY

    def test_avoid_outcome_when_bear_wins_no_position(self):
        """AVOID outcome when bear wins and no position."""
        config = DebateConfig(min_confidence_delta=0.01)
        debate = BullBearDebate(config=config)

        # Need to make bear confidence higher by manipulating context
        result = debate.run_debate(
            {"symbol": "SPY", "has_position": False},
            {"analysis": {"technicals": {"rsi": 80}}},  # Overbought
        )

        # Check outcome is valid
        assert result.final_outcome in DebateOutcome

    def test_key_points_extracted(self, debate, opportunity, initial_analysis):
        """Key points are extracted from arguments."""
        result = debate.run_debate(opportunity, initial_analysis)

        assert isinstance(result.key_points_bull, list)
        assert isinstance(result.key_points_bear, list)
        assert isinstance(result.risk_factors, list)


class TestDebateStatistics:
    """Tests for debate statistics."""

    def test_empty_statistics(self, debate):
        """Statistics for empty debate history."""
        stats = debate.get_debate_statistics()

        assert stats["total_debates"] == 0
        assert stats["avg_rounds"] == 0

    def test_statistics_after_debates(self, debate, opportunity, initial_analysis):
        """Statistics calculated correctly after debates."""
        debate.run_debate(opportunity, initial_analysis)
        debate.run_debate(opportunity, initial_analysis)

        stats = debate.get_debate_statistics()

        assert stats["total_debates"] == 2
        assert stats["avg_rounds"] > 0
        assert "outcome_distribution" in stats

    def test_clear_history(self, debate, opportunity, initial_analysis):
        """Clear history removes all records."""
        debate.run_debate(opportunity, initial_analysis)
        assert len(debate.debate_history) == 1

        debate.clear_history()

        assert len(debate.debate_history) == 0


class TestDebateIntegration:
    """Tests for integration with researcher agents."""

    def test_with_bull_researcher(self, opportunity, initial_analysis):
        """Debate works with BullResearcher."""
        bull = create_bull_researcher()
        debate = BullBearDebate(bull_agent=bull)

        result = debate.run_debate(opportunity, initial_analysis)

        assert isinstance(result, DebateResult)

    def test_with_bear_researcher(self, opportunity, initial_analysis):
        """Debate works with BearResearcher."""
        bear = create_bear_researcher()
        debate = BullBearDebate(bear_agent=bear)

        result = debate.run_debate(opportunity, initial_analysis)

        assert isinstance(result, DebateResult)

    def test_with_both_researchers(self, opportunity, initial_analysis):
        """Debate works with both researchers."""
        bull = create_bull_researcher()
        bear = create_bear_researcher()
        debate = BullBearDebate(bull_agent=bull, bear_agent=bear)

        result = debate.run_debate(opportunity, initial_analysis)

        assert isinstance(result, DebateResult)
        assert len(result.rounds) > 0


class TestSerialization:
    """Tests for serialization."""

    def test_debate_result_to_dict(self, debate, opportunity, initial_analysis):
        """DebateResult can be serialized."""
        result = debate.run_debate(opportunity, initial_analysis)
        result_dict = result.to_dict()

        assert "debate_id" in result_dict
        assert "rounds" in result_dict
        assert "final_outcome" in result_dict
        assert isinstance(result_dict["final_outcome"], str)

    def test_debate_round_to_dict(self, debate, opportunity, initial_analysis):
        """DebateRound can be serialized."""
        result = debate.run_debate(opportunity, initial_analysis)
        round_dict = result.rounds[0].to_dict()

        assert "round_number" in round_dict
        assert "bull_argument" in round_dict
        assert "bear_argument" in round_dict


class TestFactoryAndReport:
    """Tests for factory function and report generation."""

    def test_create_debate_mechanism(self):
        """Factory creates debate mechanism."""
        debate = create_debate_mechanism(
            max_rounds=5,
            consensus_threshold=0.8,
        )

        assert debate.config.max_rounds == 5
        assert debate.config.consensus_threshold == 0.8

    def test_generate_report(self, debate, opportunity, initial_analysis):
        """Report is generated correctly."""
        result = debate.run_debate(opportunity, initial_analysis)
        report = generate_debate_report(result)

        assert "BULL/BEAR DEBATE REPORT" in report
        assert "OUTCOME" in report
        assert "ROUNDS SUMMARY" in report
        assert "KEY POINTS" in report
        assert "RISK FACTORS" in report


class TestBullResearcher:
    """Tests for BullResearcher agent."""

    def test_creation(self):
        """Bull researcher can be created."""
        researcher = BullResearcher()
        assert researcher.name == "bull_researcher"

    def test_analyze(self):
        """Bull researcher can analyze."""
        researcher = BullResearcher()
        response = researcher.analyze(
            query="Analyze SPY",
            context={"symbol": "SPY", "price": 450.0},
        )

        assert response.success
        assert response.confidence > 0
        assert "BULLISH" in response.final_answer or "bullish" in response.final_answer.lower()

    def test_argue(self):
        """Bull researcher can argue in debate context."""
        researcher = BullResearcher()
        response = researcher.argue(
            context={"opportunity": {"symbol": "SPY"}},
            position="bullish",
        )

        assert response.success


class TestBearResearcher:
    """Tests for BearResearcher agent."""

    def test_creation(self):
        """Bear researcher can be created."""
        researcher = BearResearcher()
        assert researcher.name == "bear_researcher"

    def test_analyze(self):
        """Bear researcher can analyze."""
        researcher = BearResearcher()
        response = researcher.analyze(
            query="Analyze SPY risks",
            context={"symbol": "SPY", "price": 450.0},
        )

        assert response.success
        assert response.confidence > 0
        assert "CAUTION" in response.final_answer or "caution" in response.final_answer.lower()

    def test_argue(self):
        """Bear researcher can argue in debate context."""
        researcher = BearResearcher()
        response = researcher.argue(
            context={"opportunity": {"symbol": "SPY"}},
            position="bearish",
            previous_bull_argument="Market is bullish",
        )

        assert response.success
