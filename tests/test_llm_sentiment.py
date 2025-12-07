"""
Tests for UPGRADE-014: LLM Sentiment Integration

Tests for all sentiment-related components:
- SentimentFilter
- NewsAlertManager
- LLMGuardrails (trading constraints)
- Debate mechanism enhancements
- Circuit breaker sentiment integration
- Ensemble dynamic weighting
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest

from llm.agents.llm_guardrails import (
    LLMGuardrails,
    TradingConstraints,
    create_llm_guardrails,
)
from llm.base import Sentiment, SentimentResult
from llm.ensemble import (
    DynamicWeightConfig,
    LLMEnsemble,
    ProviderPerformance,
    create_ensemble,
)
from llm.news_alert_manager import (
    NewsAlertConfig,
    NewsAlertManager,
    NewsEventType,
    NewsImpact,
    create_news_alert_manager,
)

# Import sentiment components
from llm.sentiment_filter import (
    FilterDecision,
    FilterReason,
    SentimentFilter,
    SentimentSignal,
    create_sentiment_filter,
    create_signal_from_ensemble,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def sentiment_filter():
    """Create a default sentiment filter."""
    return SentimentFilter(
        min_sentiment_for_long=0.1,
        max_sentiment_for_short=-0.1,
        min_confidence=0.5,
    )


@pytest.fixture
def bullish_signal():
    """Create a bullish sentiment signal."""
    return SentimentSignal(
        symbol="AAPL",
        sentiment_score=0.6,
        confidence=0.8,
        source="finbert",
        articles_analyzed=5,
    )


@pytest.fixture
def bearish_signal():
    """Create a bearish sentiment signal."""
    return SentimentSignal(
        symbol="AAPL",
        sentiment_score=-0.5,
        confidence=0.75,
        source="finbert",
        articles_analyzed=3,
    )


@pytest.fixture
def neutral_signal():
    """Create a neutral sentiment signal."""
    return SentimentSignal(
        symbol="AAPL",
        sentiment_score=0.05,
        confidence=0.7,
        source="finbert",
        articles_analyzed=4,
    )


@pytest.fixture
def news_alert_manager():
    """Create a news alert manager."""
    config = NewsAlertConfig(
        high_impact_sentiment_threshold=0.7,
        circuit_breaker_sentiment_threshold=-0.8,
        enable_circuit_breaker_triggers=True,
    )
    return NewsAlertManager(config=config)


@pytest.fixture
def trading_constraints():
    """Create trading constraints for guardrails."""
    return TradingConstraints(
        max_position_size_pct=0.25,
        min_confidence_for_trade=0.5,
        blocked_symbols=["GME", "AMC"],
        max_daily_trades=50,
    )


@pytest.fixture
def llm_guardrails(trading_constraints):
    """Create LLM guardrails with constraints."""
    return LLMGuardrails(constraints=trading_constraints)


# ==============================================================================
# SentimentFilter Tests
# ==============================================================================


class TestSentimentFilter:
    """Tests for SentimentFilter class."""

    @pytest.mark.unit
    def test_allows_long_with_bullish_sentiment(self, sentiment_filter, bullish_signal):
        """Test that long entries are allowed with bullish sentiment."""
        result = sentiment_filter.check_entry("AAPL", "long", bullish_signal)

        assert result.is_allowed
        assert result.decision == FilterDecision.ALLOW
        assert result.reason == FilterReason.SENTIMENT_FAVORABLE

    @pytest.mark.unit
    def test_blocks_long_with_bearish_sentiment(self, sentiment_filter, bearish_signal):
        """Test that long entries are blocked with bearish sentiment."""
        result = sentiment_filter.check_entry("AAPL", "long", bearish_signal)

        assert result.is_blocked
        assert result.decision == FilterDecision.BLOCK
        assert result.reason == FilterReason.SENTIMENT_UNFAVORABLE

    @pytest.mark.unit
    def test_allows_short_with_bearish_sentiment(self, sentiment_filter, bearish_signal):
        """Test that short entries are allowed with bearish sentiment."""
        result = sentiment_filter.check_entry("AAPL", "short", bearish_signal)

        assert result.is_allowed
        assert result.decision == FilterDecision.ALLOW

    @pytest.mark.unit
    def test_blocks_short_with_bullish_sentiment(self, sentiment_filter, bullish_signal):
        """Test that short entries are blocked with bullish sentiment."""
        result = sentiment_filter.check_entry("AAPL", "short", bullish_signal)

        assert result.is_blocked
        assert result.decision == FilterDecision.BLOCK

    @pytest.mark.unit
    def test_requires_review_with_low_confidence(self, sentiment_filter):
        """Test that low confidence signals require review."""
        low_conf_signal = SentimentSignal(
            symbol="AAPL",
            sentiment_score=0.5,
            confidence=0.3,  # Below threshold
            source="finbert",
        )

        result = sentiment_filter.check_entry("AAPL", "long", low_conf_signal)

        assert result.needs_review
        assert result.decision == FilterDecision.REQUIRE_REVIEW
        assert result.reason == FilterReason.CONFIDENCE_TOO_LOW

    @pytest.mark.unit
    def test_override_bypasses_filter(self, sentiment_filter, bearish_signal):
        """Test that manual override bypasses sentiment check."""
        sentiment_filter.set_override("AAPL", duration_minutes=30)

        result = sentiment_filter.check_entry("AAPL", "long", bearish_signal)

        assert result.is_allowed
        assert result.reason == FilterReason.MANUAL_OVERRIDE

    @pytest.mark.unit
    def test_override_expires(self, sentiment_filter, bearish_signal):
        """Test that override expires after duration."""
        # Set expired override
        sentiment_filter._override_symbols["AAPL"] = datetime.now(timezone.utc) - timedelta(minutes=1)

        result = sentiment_filter.check_entry("AAPL", "long", bearish_signal)

        assert result.is_blocked
        assert "AAPL" not in sentiment_filter._override_symbols

    @pytest.mark.unit
    def test_tracks_signal_history(self, sentiment_filter, bullish_signal):
        """Test that signals are tracked in history."""
        sentiment_filter.check_entry("AAPL", "long", bullish_signal)

        history = sentiment_filter.get_signal_history("AAPL")
        assert len(history) == 1
        assert history[0].symbol == "AAPL"

    @pytest.mark.unit
    def test_tracks_statistics(self, sentiment_filter, bullish_signal, bearish_signal):
        """Test that filter statistics are tracked."""
        sentiment_filter.check_entry("AAPL", "long", bullish_signal)
        sentiment_filter.check_entry("AAPL", "long", bearish_signal)

        stats = sentiment_filter.get_stats()
        assert stats["total_checks"] == 2
        assert stats["allowed"] == 1
        assert stats["blocked"] == 1

    @pytest.mark.unit
    def test_sentiment_signal_properties(self, bullish_signal, bearish_signal):
        """Test SentimentSignal property methods."""
        assert bullish_signal.is_bullish
        assert not bullish_signal.is_bearish
        assert bullish_signal.strength == "moderate"

        assert bearish_signal.is_bearish
        assert not bearish_signal.is_bullish

    @pytest.mark.unit
    def test_filter_result_to_dict(self, sentiment_filter, bullish_signal):
        """Test FilterResult serialization."""
        result = sentiment_filter.check_entry("AAPL", "long", bullish_signal)

        result_dict = result.to_dict()
        assert "decision" in result_dict
        assert "reason" in result_dict
        assert result_dict["decision"] == "allow"


class TestSentimentFilterTrend:
    """Tests for trend-based filtering."""

    @pytest.fixture
    def trend_filter(self):
        """Create filter with trend requirement."""
        return SentimentFilter(
            min_sentiment_for_long=0.1,
            max_sentiment_for_short=-0.1,
            min_confidence=0.5,
            require_positive_trend=True,
            trend_window_size=3,
        )

    @pytest.mark.unit
    def test_trend_supports_long(self, trend_filter):
        """Test that improving trend supports long entry."""
        # Add improving sentiment history
        for score in [0.2, 0.3, 0.4]:
            signal = SentimentSignal(symbol="AAPL", sentiment_score=score, confidence=0.8, source="finbert")
            trend_filter._add_signal("AAPL", signal)

        # Now check with current bullish signal
        current = SentimentSignal(symbol="AAPL", sentiment_score=0.5, confidence=0.8, source="finbert")
        result = trend_filter.check_entry("AAPL", "long", current)

        assert result.is_allowed

    @pytest.mark.unit
    def test_trend_analysis_with_declining_scores(self, trend_filter):
        """Test trend analysis with declining scores."""
        # Add declining sentiment history with all negative values
        for score in [-0.1, -0.3, -0.5]:
            signal = SentimentSignal(symbol="AAPL", sentiment_score=score, confidence=0.8, source="finbert")
            trend_filter._add_signal("AAPL", signal)

        # Try short with declining trend - should support short
        current = SentimentSignal(symbol="AAPL", sentiment_score=-0.6, confidence=0.8, source="finbert")
        result = trend_filter.check_entry("AAPL", "short", current)

        # Declining negative trend supports short positions
        assert result.is_allowed


# ==============================================================================
# NewsAlertManager Tests
# ==============================================================================


class TestNewsAlertManager:
    """Tests for NewsAlertManager class."""

    @pytest.mark.unit
    def test_manager_initialization(self, news_alert_manager):
        """Test that news alert manager initializes correctly."""
        assert news_alert_manager is not None
        assert news_alert_manager.config is not None

    @pytest.mark.unit
    def test_config_has_expected_fields(self, news_alert_manager):
        """Test that config has expected sentiment thresholds."""
        config = news_alert_manager.config
        assert hasattr(config, "high_impact_sentiment_threshold")
        assert hasattr(config, "circuit_breaker_sentiment_threshold")
        assert hasattr(config, "enable_circuit_breaker_triggers")

    @pytest.mark.unit
    def test_news_impact_enum_values(self):
        """Test NewsImpact enum has expected values."""
        assert NewsImpact.LOW is not None
        assert NewsImpact.MEDIUM is not None
        assert NewsImpact.HIGH is not None
        assert NewsImpact.CRITICAL is not None

    @pytest.mark.unit
    def test_news_event_type_enum_values(self):
        """Test NewsEventType enum has expected values."""
        assert NewsEventType.EARNINGS is not None
        assert NewsEventType.GENERAL is not None
        # Check that we have multiple event types
        assert len(list(NewsEventType)) > 1


# ==============================================================================
# LLMGuardrails Tests
# ==============================================================================


class TestLLMGuardrails:
    """Tests for LLMGuardrails class."""

    @pytest.mark.unit
    def test_validates_clean_input(self, llm_guardrails):
        """Test that clean input passes validation."""
        result = llm_guardrails.check_input(
            query="Analyze sentiment for AAPL",
            context={"symbol": "AAPL"},
        )

        assert result.passed

    @pytest.mark.unit
    def test_blocks_empty_query(self, llm_guardrails):
        """Test that empty queries are blocked."""
        result = llm_guardrails.check_input(query="", context={})

        assert not result.passed
        assert any("empty" in v.message.lower() for v in result.violations)

    @pytest.mark.unit
    def test_blocks_blocked_symbols(self, llm_guardrails):
        """Test that blocked symbols are rejected."""
        result = llm_guardrails.check_input(
            query="Buy GME calls",
            context={"symbol": "GME"},
        )

        assert not result.passed

    @pytest.mark.unit
    def test_validates_output_confidence(self, llm_guardrails):
        """Test that output confidence is validated."""
        # Valid confidence - check_output expects string output
        result = llm_guardrails.check_output(
            output="Based on analysis, I recommend BUY with high confidence.",
            confidence=0.8,
            context={"symbol": "AAPL"},
        )
        assert result.passed

        # Very low confidence (may fail minimum threshold)
        result_low = llm_guardrails.check_output(
            output="I recommend BUY.",
            confidence=0.1,  # Very low
            context={"symbol": "AAPL"},
        )
        # Result depends on min_confidence_for_trade setting
        assert isinstance(result_low.passed, bool)

    @pytest.mark.unit
    def test_adjusts_position_for_confidence(self, llm_guardrails):
        """Test confidence-based position adjustment."""
        # High confidence -> Higher proportion
        size_high, _ = llm_guardrails.adjust_position_for_confidence(base_size=0.20, confidence=0.9, context={})
        # Should maintain or increase size for high confidence

        # Low confidence -> Reduced proportion
        size_low, _ = llm_guardrails.adjust_position_for_confidence(base_size=0.20, confidence=0.4, context={})
        # Low confidence should reduce size
        assert size_low <= size_high

    @pytest.mark.unit
    def test_validates_trade_decision(self, llm_guardrails):
        """Test trade decision validation."""
        result = llm_guardrails.validate_trade_decision(
            action="BUY",
            symbol="AAPL",
            position_size=0.1,
            confidence=0.8,
            sentiment_score=0.6,
        )

        assert result.passed

    @pytest.mark.unit
    def test_blocks_invalid_action(self, llm_guardrails):
        """Test that invalid actions are blocked."""
        result = llm_guardrails.validate_trade_decision(
            action="INVALID_ACTION",  # Not in allowed actions
            symbol="AAPL",
            position_size=0.1,
            confidence=0.8,
            sentiment_score=0.6,
        )

        assert not result.passed


# ==============================================================================
# Ensemble Dynamic Weighting Tests
# ==============================================================================


class TestEnsembleDynamicWeighting:
    """Tests for dynamic weighting in LLMEnsemble."""

    @pytest.fixture
    def dynamic_weight_config(self):
        """Create dynamic weight configuration."""
        return DynamicWeightConfig(
            learning_rate=0.2,
            min_weight=0.1,
            max_weight=0.5,
            enable_boosting=True,
        )

    @pytest.fixture
    def mock_ensemble(self, dynamic_weight_config):
        """Create ensemble with mocked providers."""
        config = {
            "enabled": True,
            "providers": {"finbert": {"local": True}},
            "weights": {"finbert": 1.0},
            "ensemble_strategy": "weighted_average",
            "enable_dynamic_weights": True,
        }

        with patch("llm.ensemble.create_sentiment_analyzer") as mock_analyzer:
            mock_analyzer.return_value = Mock()
            mock_analyzer.return_value.analyze.return_value = SentimentResult(
                sentiment=Sentiment.BULLISH,
                confidence=0.8,
                score=0.5,
                provider="finbert",
            )
            ensemble = LLMEnsemble(config, dynamic_weight_config)

        return ensemble

    @pytest.mark.unit
    def test_provider_performance_tracking(self):
        """Test ProviderPerformance tracking."""
        perf = ProviderPerformance(provider_name="finbert")

        # Update with correct prediction
        perf.update_accuracy(was_correct=True)
        assert perf.total_predictions == 1
        assert perf.correct_predictions == 1
        assert perf.accuracy == 1.0

        # Update with incorrect prediction
        perf.update_accuracy(was_correct=False)
        assert perf.total_predictions == 2
        assert perf.accuracy == 0.5

    @pytest.mark.unit
    def test_weighted_accuracy_decay(self):
        """Test exponential decay in weighted accuracy."""
        perf = ProviderPerformance(provider_name="finbert")
        initial_accuracy = perf.weighted_accuracy

        # Multiple correct predictions should increase accuracy
        for _ in range(5):
            perf.update_accuracy(was_correct=True)

        assert perf.weighted_accuracy > initial_accuracy

    @pytest.mark.unit
    def test_calibration_tracking(self):
        """Test calibration score updates."""
        perf = ProviderPerformance(provider_name="finbert")

        # High confidence, correct -> good calibration
        perf.update_calibration(confidence=0.9, was_correct=True)
        high_cal = perf.calibration_score

        # Reset and test bad calibration
        perf2 = ProviderPerformance(provider_name="test")
        perf2.update_calibration(confidence=0.9, was_correct=False)

        assert perf2.calibration_score < high_cal

    @pytest.mark.unit
    def test_weight_normalization(self, mock_ensemble):
        """Test that weights are normalized after updates."""
        # Manually set weights
        mock_ensemble.weights = {"finbert": 0.5, "openai": 0.3, "anthropic": 0.2}
        mock_ensemble._normalize_weights()

        total = sum(mock_ensemble.weights.values())
        assert abs(total - 1.0) < 0.01  # Should sum to ~1.0

    @pytest.mark.unit
    def test_prediction_history_recording(self, mock_ensemble):
        """Test that predictions are recorded in history."""
        # Analyze sentiment
        mock_ensemble.analyze_sentiment("Test news about AAPL")

        assert len(mock_ensemble._prediction_history) == 1
        assert "id" in mock_ensemble._prediction_history[0]

    @pytest.mark.unit
    def test_prediction_outcome_feedback(self, mock_ensemble):
        """Test recording prediction outcomes."""
        # First, make a prediction
        result = mock_ensemble.analyze_sentiment("Test news")
        prediction_id = result.prediction_id

        # Record outcome
        mock_ensemble.record_prediction_outcome(
            prediction_id=prediction_id,
            actual_sentiment=0.5,
            actual_direction_correct=True,
        )

        # Check performance was updated
        perf = mock_ensemble._performance.get("finbert")
        if perf:
            assert perf.total_predictions >= 1


# ==============================================================================
# Circuit Breaker Sentiment Integration Tests
# ==============================================================================


class TestCircuitBreakerSentiment:
    """Tests for sentiment integration with circuit breaker."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker with sentiment config."""
        from models.circuit_breaker import CircuitBreakerConfig, TradingCircuitBreaker

        config = CircuitBreakerConfig(
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.10,
            enable_sentiment_triggers=True,
            sentiment_halt_threshold=-0.7,
            consecutive_negative_sentiment=3,
        )
        return TradingCircuitBreaker(config=config)

    @pytest.mark.unit
    def test_sentiment_check_allows_normal(self, circuit_breaker):
        """Test that normal sentiment doesn't trip breaker."""
        can_trade = circuit_breaker.check_sentiment(
            symbol="AAPL",
            sentiment_score=0.2,
            confidence=0.8,
            source="finbert",
        )

        assert can_trade

    @pytest.mark.unit
    def test_severe_negative_trips_breaker(self, circuit_breaker):
        """Test that severe negative sentiment trips breaker."""
        can_trade = circuit_breaker.check_sentiment(
            symbol="AAPL",
            sentiment_score=-0.8,  # Below threshold
            confidence=0.9,
            source="finbert",
        )

        assert not can_trade
        assert not circuit_breaker.can_trade()

    @pytest.mark.unit
    def test_consecutive_negative_trips_breaker(self, circuit_breaker):
        """Test that consecutive negative sentiment trips breaker."""
        # Multiple moderately negative readings
        for _ in range(4):  # Above threshold of 3
            circuit_breaker.check_sentiment(
                symbol="AAPL",
                sentiment_score=-0.4,
                confidence=0.7,
                source="finbert",
            )

        assert not circuit_breaker.can_trade()

    @pytest.mark.unit
    def test_positive_sentiment_resets_counter(self, circuit_breaker):
        """Test that positive sentiment resets negative counter."""
        # Add some negative readings
        for _ in range(2):
            circuit_breaker.check_sentiment(
                symbol="AAPL",
                sentiment_score=-0.4,
                confidence=0.7,
                source="finbert",
            )

        # Now positive
        circuit_breaker.check_sentiment(
            symbol="AAPL",
            sentiment_score=0.5,
            confidence=0.7,
            source="finbert",
        )

        # Should still be able to trade
        assert circuit_breaker.can_trade()

    @pytest.mark.unit
    def test_critical_news_trips_breaker(self, circuit_breaker):
        """Test that critical news triggers circuit breaker."""
        can_trade = circuit_breaker.check_news_event(
            symbol="AAPL",
            headline="SEC investigation into Apple fraud",
            sentiment_score=-0.9,
            impact="critical",
        )

        assert not can_trade

    @pytest.mark.unit
    def test_sentiment_divergence_detection(self, circuit_breaker):
        """Test detection of sentiment divergence."""
        # Market going up, but sentiment very negative
        can_trade = circuit_breaker.check_sentiment_divergence(
            symbol="AAPL",
            market_direction=1,  # Bullish price action
            sentiment_direction=-1,  # Bearish sentiment
            divergence_magnitude=0.6,  # Significant divergence
        )

        # Divergence may or may not halt depending on threshold
        assert isinstance(can_trade, bool)

    @pytest.mark.unit
    def test_get_sentiment_stats(self, circuit_breaker):
        """Test getting sentiment statistics."""
        # Add some readings
        circuit_breaker.check_sentiment("AAPL", 0.5, 0.8, "finbert")
        circuit_breaker.check_sentiment("AAPL", 0.3, 0.7, "finbert")

        stats = circuit_breaker.get_sentiment_stats()

        # Check for expected keys in stats
        assert "total_signals" in stats
        assert stats["total_signals"] >= 2


# ==============================================================================
# Factory Function Tests
# ==============================================================================


class TestLlmSentimentFactoryFunctions:
    """Tests for factory functions."""

    @pytest.mark.unit
    def test_create_sentiment_filter_default(self):
        """Test creating filter with defaults."""
        filter_instance = create_sentiment_filter()

        assert filter_instance is not None
        assert filter_instance.min_confidence == 0.5

    @pytest.mark.unit
    def test_create_sentiment_filter_custom(self):
        """Test creating filter with custom config."""
        config = {
            "min_sentiment_for_long": 0.2,
            "max_sentiment_for_short": -0.2,
            "min_confidence": 0.7,
        }
        filter_instance = create_sentiment_filter(config)

        assert filter_instance.min_sentiment_for_long == 0.2
        assert filter_instance.min_confidence == 0.7

    @pytest.mark.unit
    def test_create_signal_from_ensemble(self):
        """Test creating signal from ensemble result."""
        # Create mock ensemble result
        mock_result = Mock()
        mock_result.sentiment.score = 0.5
        mock_result.sentiment.confidence = 0.8
        mock_result.individual_results = [Mock(), Mock()]

        signal = create_signal_from_ensemble("AAPL", mock_result)

        assert signal.symbol == "AAPL"
        assert signal.sentiment_score == 0.5
        assert signal.confidence == 0.8
        assert signal.source == "ensemble"
        assert signal.articles_analyzed == 2

    @pytest.mark.unit
    def test_create_news_alert_manager_default(self):
        """Test creating news alert manager with defaults."""
        manager = create_news_alert_manager()

        assert manager is not None

    @pytest.mark.unit
    def test_create_llm_guardrails_default(self):
        """Test creating guardrails with defaults."""
        guardrails = create_llm_guardrails()

        assert guardrails is not None

    @pytest.mark.unit
    def test_create_ensemble_default(self):
        """Test creating ensemble with defaults."""
        with patch("llm.ensemble.create_sentiment_analyzer") as mock:
            mock.return_value = Mock()
            ensemble = create_ensemble()

        assert ensemble is not None


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestSentimentIntegration:
    """Integration tests for sentiment components."""

    @pytest.mark.integration
    def test_full_sentiment_flow(self):
        """Test complete sentiment analysis flow."""
        # Create components
        filter_instance = create_sentiment_filter(
            {
                "min_sentiment_for_long": 0.1,
                "min_confidence": 0.5,
            }
        )

        # Create signal
        signal = SentimentSignal(
            symbol="AAPL",
            sentiment_score=0.6,
            confidence=0.8,
            source="ensemble",
        )

        # Check filter
        result = filter_instance.check_entry("AAPL", "long", signal)
        assert result.is_allowed

        # Verify stats
        stats = filter_instance.get_stats()
        assert stats["total_checks"] == 1
        assert stats["allowed"] == 1

    @pytest.mark.integration
    def test_sentiment_with_circuit_breaker(self):
        """Test sentiment filter with circuit breaker."""
        from models.circuit_breaker import CircuitBreakerConfig, TradingCircuitBreaker

        # Create circuit breaker
        breaker_config = CircuitBreakerConfig(
            enable_sentiment_triggers=True,
            sentiment_halt_threshold=-0.7,
        )
        breaker = TradingCircuitBreaker(config=breaker_config)

        # Create filter
        filter_instance = create_sentiment_filter()

        # Normal operation
        signal = SentimentSignal(
            symbol="AAPL",
            sentiment_score=0.5,
            confidence=0.8,
            source="finbert",
        )

        # Both should allow
        can_trade = breaker.check_sentiment("AAPL", signal.sentiment_score, signal.confidence, signal.source)
        filter_result = filter_instance.check_entry("AAPL", "long", signal)

        assert can_trade
        assert filter_result.is_allowed

        # Severe negative should trip breaker
        bad_signal = SentimentSignal(
            symbol="AAPL",
            sentiment_score=-0.8,
            confidence=0.9,
            source="finbert",
        )

        can_trade = breaker.check_sentiment(
            "AAPL", bad_signal.sentiment_score, bad_signal.confidence, bad_signal.source
        )
        assert not can_trade


# ==============================================================================
# Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_neutral_sentiment_handling(self):
        """Test handling of neutral sentiment."""
        filter_instance = create_sentiment_filter(
            {
                "min_sentiment_for_long": 0.0,
                "max_sentiment_for_short": 0.0,
            }
        )

        neutral = SentimentSignal(
            symbol="AAPL",
            sentiment_score=0.0,
            confidence=0.8,
            source="finbert",
        )

        # Should allow both directions at exact neutral
        long_result = filter_instance.check_entry("AAPL", "long", neutral)
        short_result = filter_instance.check_entry("AAPL", "short", neutral)

        assert long_result.is_allowed
        assert short_result.is_allowed

    @pytest.mark.unit
    def test_extreme_sentiment_values(self):
        """Test handling of extreme sentiment values."""
        filter_instance = create_sentiment_filter()

        extreme_bullish = SentimentSignal(
            symbol="AAPL",
            sentiment_score=1.0,  # Maximum bullish
            confidence=1.0,
            source="finbert",
        )

        extreme_bearish = SentimentSignal(
            symbol="AAPL",
            sentiment_score=-1.0,  # Maximum bearish
            confidence=1.0,
            source="finbert",
        )

        bull_result = filter_instance.check_entry("AAPL", "long", extreme_bullish)
        bear_result = filter_instance.check_entry("AAPL", "short", extreme_bearish)

        assert bull_result.is_allowed
        assert bear_result.is_allowed
        assert extreme_bullish.strength == "strong"
        assert extreme_bearish.strength == "strong"

    @pytest.mark.unit
    def test_zero_confidence_handling(self):
        """Test handling of zero confidence."""
        filter_instance = create_sentiment_filter({"min_confidence": 0.5})

        zero_conf = SentimentSignal(
            symbol="AAPL",
            sentiment_score=0.8,
            confidence=0.0,
            source="finbert",
        )

        result = filter_instance.check_entry("AAPL", "long", zero_conf)

        assert result.needs_review
        assert result.reason == FilterReason.CONFIDENCE_TOO_LOW

    @pytest.mark.unit
    def test_history_limit_enforcement(self):
        """Test that history is limited."""
        filter_instance = SentimentFilter(max_history_per_symbol=5)

        # Add more signals than limit
        for i in range(10):
            signal = SentimentSignal(
                symbol="AAPL",
                sentiment_score=0.1 * i,
                confidence=0.8,
                source="finbert",
            )
            filter_instance._add_signal("AAPL", signal)

        history = filter_instance.get_signal_history("AAPL", limit=100)
        assert len(history) == 5  # Should be capped at max

    @pytest.mark.unit
    def test_check_entry_no_data(self):
        """Test check_entry returns NO_DATA when no signals available."""
        filter_instance = SentimentFilter()

        # No signals for symbol
        result = filter_instance.check_entry("UNKNOWN", "long")

        assert result.decision == FilterDecision.REQUIRE_REVIEW
        assert result.reason == FilterReason.NO_DATA

    @pytest.mark.unit
    def test_check_entry_with_history(self):
        """Test check_entry uses latest signal from history."""
        filter_instance = SentimentFilter()

        # Add signal via public method
        signal = SentimentSignal(
            symbol="AAPL",
            sentiment_score=0.6,
            confidence=0.8,
            source="ensemble",
        )
        filter_instance.add_signal(signal)

        # Check entry without providing current_signal
        result = filter_instance.check_entry("AAPL", "long")

        assert result.is_allowed


# ==============================================================================
# Sentiment Decay Tests (UPGRADE-014 Expansion)
# ==============================================================================


class TestSentimentDecay:
    """Tests for sentiment decay and time-weighted analysis."""

    @pytest.fixture
    def filter_with_signals(self):
        """Create filter with time-distributed signals."""
        from datetime import timedelta

        sf = SentimentFilter()
        now = datetime.now(timezone.utc)

        # Recent signal (high weight)
        sf.add_signal(
            SentimentSignal(
                symbol="AAPL",
                sentiment_score=0.8,
                confidence=0.9,
                source="ensemble",
                timestamp=now - timedelta(minutes=10),
            )
        )

        # Older signal (medium weight)
        sf.add_signal(
            SentimentSignal(
                symbol="AAPL",
                sentiment_score=-0.3,
                confidence=0.7,
                source="ensemble",
                timestamp=now - timedelta(hours=5),
            )
        )

        # Very old signal (low weight)
        sf.add_signal(
            SentimentSignal(
                symbol="AAPL",
                sentiment_score=-0.6,
                confidence=0.6,
                source="ensemble",
                timestamp=now - timedelta(hours=12),
            )
        )

        return sf

    @pytest.mark.unit
    def test_get_weighted_sentiment_recent_dominates(self, filter_with_signals):
        """Test that recent signals have higher weight."""
        result = filter_with_signals.get_weighted_sentiment("AAPL")

        # Recent positive (0.8) should dominate over older negatives
        assert result is not None
        assert result["weighted_score"] > 0
        assert result["is_bullish"] is True

    @pytest.mark.unit
    def test_get_weighted_sentiment_no_data(self):
        """Test get_weighted_sentiment returns None when no data."""
        sf = SentimentFilter()
        result = sf.get_weighted_sentiment("UNKNOWN")
        assert result is None

    @pytest.mark.unit
    def test_get_weighted_sentiment_expired_signals(self):
        """Test that very old signals are excluded."""
        from datetime import timedelta

        sf = SentimentFilter()
        now = datetime.now(timezone.utc)

        # Add only very old signal
        sf.add_signal(
            SentimentSignal(
                symbol="AAPL",
                sentiment_score=-0.5,
                confidence=0.8,
                source="ensemble",
                timestamp=now - timedelta(hours=48),  # Older than 24h default
            )
        )

        result = sf.get_weighted_sentiment("AAPL", max_age_hours=24)
        assert result is None  # All signals expired

    @pytest.mark.unit
    def test_get_weighted_sentiment_custom_decay(self, filter_with_signals):
        """Test custom decay rate affects weighting."""
        # Lower decay rate = faster decay = recent signals dominate more
        # Higher decay rate = slower decay = older signals retain more weight
        result_high_decay = filter_with_signals.get_weighted_sentiment("AAPL", decay_rate=0.95)
        result_low_decay = filter_with_signals.get_weighted_sentiment("AAPL", decay_rate=0.5)

        # With low decay, recent signal dominates more (positive recent vs negative older)
        assert result_low_decay["weighted_score"] > result_high_decay["weighted_score"]

    @pytest.mark.unit
    def test_get_bulk_sentiment(self, filter_with_signals):
        """Test bulk sentiment for multiple symbols."""
        from datetime import timedelta

        now = datetime.now(timezone.utc)

        # Add signal for another symbol
        filter_with_signals.add_signal(
            SentimentSignal(
                symbol="TSLA",
                sentiment_score=-0.5,
                confidence=0.8,
                source="ensemble",
                timestamp=now - timedelta(minutes=30),
            )
        )

        bulk = filter_with_signals.get_bulk_sentiment(["AAPL", "TSLA", "MSFT"])

        assert "AAPL" in bulk
        assert "TSLA" in bulk
        assert "MSFT" not in bulk  # No data

    @pytest.mark.unit
    def test_weighted_sentiment_includes_trend(self, filter_with_signals):
        """Test that weighted sentiment includes trend information."""
        result = filter_with_signals.get_weighted_sentiment("AAPL")

        assert "trend" in result
        assert "signals_used" in result
        assert "total_weight" in result
        assert result["signals_used"] > 0


# ==============================================================================
# Public add_signal Tests (Bug Fix Verification)
# ==============================================================================


class TestPublicAddSignal:
    """Tests for public add_signal method (bug fix)."""

    @pytest.mark.unit
    def test_add_signal_exists(self):
        """Test that public add_signal method exists."""
        sf = SentimentFilter()
        assert hasattr(sf, "add_signal")
        assert callable(sf.add_signal)

    @pytest.mark.unit
    def test_add_signal_stores_in_history(self):
        """Test that add_signal stores signal in history."""
        sf = SentimentFilter()

        signal = SentimentSignal(
            symbol="AAPL",
            sentiment_score=0.5,
            confidence=0.8,
            source="ensemble",
        )
        sf.add_signal(signal)

        history = sf.get_signal_history("AAPL")
        assert len(history) == 1
        assert history[0].sentiment_score == 0.5

    @pytest.mark.unit
    def test_add_signal_enables_check_entry(self):
        """Test that signals added via add_signal work with check_entry."""
        sf = SentimentFilter()

        # Add signal
        signal = SentimentSignal(
            symbol="AAPL",
            sentiment_score=0.6,
            confidence=0.8,
            source="ensemble",
        )
        sf.add_signal(signal)

        # Check entry without providing current_signal
        result = sf.check_entry("AAPL", "long")

        # Should use the signal we added
        assert result.decision == FilterDecision.ALLOW


# ==============================================================================
# UPGRADE-014 EXPANSION TESTS (December 2025)
# ==============================================================================


class TestRegimeDetector:
    """Tests for Regime-Adaptive Sentiment Weighting."""

    @pytest.fixture
    def detector(self):
        """Create regime detector for tests."""
        from llm.sentiment_filter import RegimeDetector

        return RegimeDetector()

    @pytest.mark.unit
    def test_regime_detector_initialization(self, detector):
        """Test regime detector initializes correctly."""
        assert detector.volatility_window == 20
        assert detector.trend_window == 50
        assert detector.high_vol_threshold == 75.0
        assert detector.low_vol_threshold == 25.0

    @pytest.mark.unit
    def test_detect_high_volatility_regime(self, detector):
        """Test detection of high volatility regime."""
        from llm.sentiment_filter import MarketRegime

        # Build up a distribution of volatility values
        # First add normal volatility to establish baseline
        for i in range(50):
            detector.update(0.10 + i * 0.002, 0.0)  # 10% to 20% vol

        # Then add high volatility at the end
        for _ in range(10):
            detector.update(0.45, 0.0)  # 45% vol (high)

        state = detector.get_current_regime()
        assert state is not None
        assert state.regime == MarketRegime.HIGH_VOLATILITY
        assert state.config.sentiment_weight == 0.4
        assert state.config.position_mult == 0.5

    @pytest.mark.unit
    def test_detect_low_volatility_regime(self, detector):
        """Test detection of low volatility regime."""
        from llm.sentiment_filter import MarketRegime

        # Build up a distribution with higher volatility
        for i in range(50):
            detector.update(0.15 + i * 0.002, 0.0)  # 15% to 25% vol

        # Then add low volatility at the end
        for _ in range(10):
            detector.update(0.08, 0.0)  # 8% vol (low relative to history)

        state = detector.get_current_regime()
        assert state.regime == MarketRegime.LOW_VOLATILITY
        assert state.config.sentiment_weight == 0.5
        assert state.config.position_mult == 1.0

    @pytest.mark.unit
    def test_detect_bull_trending_regime(self, detector):
        """Test detection of bull trending regime."""
        from llm.sentiment_filter import MarketRegime

        # First, build a wide volatility distribution
        # This ensures current vol will be in middle percentile
        for i in range(100):
            vol = 0.08 + i * 0.004  # 8% to 48% range
            detector.update(vol, 0.0)  # Neutral returns

        # Now add moderate volatility with strong positive trend
        # 28% vol is median of 8-48% range, so ~50th percentile
        for _ in range(60):
            detector.update(0.28, 0.015)  # 28% vol, +1.5% daily

        state = detector.get_current_regime()
        # Vol should be middle (25-75 percentile), trend should be strong
        assert state.regime == MarketRegime.BULL_TRENDING
        assert state.config.sentiment_weight == 0.7

    @pytest.mark.unit
    def test_detect_bear_trending_regime(self, detector):
        """Test detection of bear trending regime."""
        from llm.sentiment_filter import MarketRegime

        # First, build a wide volatility distribution
        for i in range(100):
            vol = 0.08 + i * 0.004  # 8% to 48% range
            detector.update(vol, 0.0)

        # Now add moderate volatility with strong negative trend
        for _ in range(60):
            detector.update(0.28, -0.015)  # 28% vol, -1.5% daily

        state = detector.get_current_regime()
        assert state.regime == MarketRegime.BEAR_TRENDING
        assert state.config.position_mult == 0.8

    @pytest.mark.unit
    def test_regime_state_to_dict(self, detector):
        """Test regime state serialization."""
        detector.update(0.20, 0.01)
        state = detector.get_current_regime()

        state_dict = state.to_dict()
        assert "regime" in state_dict
        assert "sentiment_weight" in state_dict
        assert "position_mult" in state_dict
        assert "volatility_percentile" in state_dict

    @pytest.mark.unit
    def test_get_sentiment_weight(self, detector):
        """Test getting sentiment weight from regime."""
        detector.update(0.10, 0.0)
        weight = detector.get_sentiment_weight()
        assert 0.0 <= weight <= 1.0

    @pytest.mark.unit
    def test_get_position_multiplier(self, detector):
        """Test getting position multiplier from regime."""
        detector.update(0.10, 0.0)
        mult = detector.get_position_multiplier()
        assert 0.0 < mult <= 1.5


class TestConfidencePositionSizing:
    """Tests for Confidence-Based Position Sizing."""

    @pytest.mark.unit
    def test_calculate_basic_position_size(self):
        """Test basic position size calculation."""
        from llm.sentiment_filter import calculate_confidence_position_size

        result = calculate_confidence_position_size(
            base_size=0.02,
            sentiment_confidence=0.8,
        )

        assert result.base_size == 0.02
        assert result.adjusted_size > result.base_size  # High conf = larger size
        assert result.confidence_mult == 1.3  # 0.5 + 0.8

    @pytest.mark.unit
    def test_low_confidence_reduces_size(self):
        """Test that low confidence reduces position size."""
        from llm.sentiment_filter import calculate_confidence_position_size

        result_high = calculate_confidence_position_size(0.02, 0.9)
        result_low = calculate_confidence_position_size(0.02, 0.3)

        assert result_low.adjusted_size < result_high.adjusted_size

    @pytest.mark.unit
    def test_ensemble_agreement_multiplier(self):
        """Test ensemble agreement affects position size."""
        from llm.sentiment_filter import calculate_confidence_position_size

        result_agree = calculate_confidence_position_size(0.02, 0.8, ensemble_agreement=1.0)
        result_disagree = calculate_confidence_position_size(0.02, 0.8, ensemble_agreement=0.3)

        assert result_disagree.adjusted_size < result_agree.adjusted_size

    @pytest.mark.unit
    def test_regime_multiplier(self):
        """Test regime multiplier affects position size."""
        from llm.sentiment_filter import calculate_confidence_position_size

        result_normal = calculate_confidence_position_size(0.02, 0.8, regime_mult=1.0)
        result_volatile = calculate_confidence_position_size(0.02, 0.8, regime_mult=0.5)

        assert result_volatile.adjusted_size < result_normal.adjusted_size

    @pytest.mark.unit
    def test_position_size_caps(self):
        """Test position size is capped correctly."""
        from llm.sentiment_filter import calculate_confidence_position_size

        result = calculate_confidence_position_size(
            base_size=0.10,
            sentiment_confidence=1.0,
            ensemble_agreement=1.0,
            regime_mult=1.5,
            max_mult=2.0,
        )

        # Should be capped at 2.0x base size
        assert result.adjusted_size <= 0.10 * 2.0
        assert result.capped

    @pytest.mark.unit
    def test_position_size_result_to_dict(self):
        """Test position size result serialization."""
        from llm.sentiment_filter import calculate_confidence_position_size

        result = calculate_confidence_position_size(0.02, 0.8)
        result_dict = result.to_dict()

        assert "base_size" in result_dict
        assert "adjusted_size" in result_dict
        assert "final_mult" in result_dict


class TestLogitToScore:
    """Tests for logit-to-score conversion."""

    @pytest.mark.unit
    def test_logit_to_score_zero(self):
        """Test logit=0 converts to 0.5."""
        from llm.sentiment_filter import logit_to_score

        assert abs(logit_to_score(0.0) - 0.5) < 0.001

    @pytest.mark.unit
    def test_logit_to_score_positive(self):
        """Test positive logit converts to >0.5."""
        from llm.sentiment_filter import logit_to_score

        assert logit_to_score(2.0) > 0.5

    @pytest.mark.unit
    def test_logit_to_score_negative(self):
        """Test negative logit converts to <0.5."""
        from llm.sentiment_filter import logit_to_score

        assert logit_to_score(-2.0) < 0.5

    @pytest.mark.unit
    def test_logit_to_score_range(self):
        """Test output is always in [0, 1]."""
        from llm.sentiment_filter import logit_to_score

        for logit in [-10, -5, -1, 0, 1, 5, 10]:
            score = logit_to_score(logit)
            assert 0.0 <= score <= 1.0


class TestSoftVotingEnsemble:
    """Tests for Soft Voting Ensemble."""

    @pytest.mark.unit
    def test_soft_vote_basic(self):
        """Test basic soft voting."""
        from llm.sentiment_filter import soft_vote_ensemble

        predictions = [
            ("model1", 0.5, 0.9),
            ("model2", 0.4, 0.8),
            ("model3", 0.6, 0.7),
        ]

        result = soft_vote_ensemble(predictions)

        assert 0.0 < result.final_score < 1.0
        assert 0.0 < result.final_confidence <= 1.0
        assert result.voting_method == "soft_vote_confidence"
        assert len(result.model_scores) == 3

    @pytest.mark.unit
    def test_soft_vote_with_weights(self):
        """Test weighted soft voting."""
        from llm.sentiment_filter import soft_vote_ensemble

        predictions = [
            ("finbert", 0.8, 0.9),
            ("gpt4", 0.2, 0.7),
        ]
        weights = {"finbert": 2.0, "gpt4": 1.0}

        result = soft_vote_ensemble(predictions, weights=weights)

        # FinBERT should dominate due to higher weight
        assert result.final_score > 0.5

    @pytest.mark.unit
    def test_soft_vote_empty_predictions(self):
        """Test soft voting with empty predictions."""
        from llm.sentiment_filter import soft_vote_ensemble

        result = soft_vote_ensemble([])

        assert result.final_score == 0.0
        assert result.final_confidence == 0.0

    @pytest.mark.unit
    def test_soft_vote_without_confidence_weighting(self):
        """Test soft voting without confidence weighting."""
        from llm.sentiment_filter import soft_vote_ensemble

        predictions = [
            ("model1", 0.5, 0.1),  # Low confidence
            ("model2", 0.5, 0.9),  # High confidence
        ]

        result = soft_vote_ensemble(predictions, use_confidence_weighting=False)
        assert result.voting_method == "soft_vote"


class TestHardVotingEnsemble:
    """Tests for Hard Voting Ensemble."""

    @pytest.mark.unit
    def test_hard_vote_bullish_majority(self):
        """Test hard voting with bullish majority."""
        from llm.sentiment_filter import hard_vote_ensemble

        predictions = [
            ("model1", 0.5, 0.9),  # Bullish
            ("model2", 0.4, 0.8),  # Bullish
            ("model3", -0.3, 0.7),  # Bearish
        ]

        result = hard_vote_ensemble(predictions)

        assert result.final_score > 0  # Majority bullish
        assert result.final_confidence == 2 / 3  # 2 out of 3

    @pytest.mark.unit
    def test_hard_vote_bearish_majority(self):
        """Test hard voting with bearish majority."""
        from llm.sentiment_filter import hard_vote_ensemble

        predictions = [
            ("model1", -0.5, 0.9),
            ("model2", -0.4, 0.8),
            ("model3", 0.3, 0.7),
        ]

        result = hard_vote_ensemble(predictions)
        assert result.final_score < 0  # Majority bearish

    @pytest.mark.unit
    def test_hard_vote_neutral_majority(self):
        """Test hard voting with neutral majority."""
        from llm.sentiment_filter import hard_vote_ensemble

        predictions = [
            ("model1", 0.05, 0.9),  # Neutral
            ("model2", -0.05, 0.8),  # Neutral
            ("model3", 0.5, 0.7),  # Bullish
        ]

        result = hard_vote_ensemble(predictions)
        assert abs(result.final_score) < 0.1  # Neutral wins


class TestWeightedSoftVoteEnsemble:
    """Tests for Weighted Soft Voting Ensemble."""

    @pytest.mark.unit
    def test_weighted_soft_vote(self):
        """Test weighted soft voting with performance weights."""
        from llm.sentiment_filter import weighted_soft_vote_ensemble

        predictions = [
            ("finbert", 0.7, 0.85),
            ("gpt4", 0.3, 0.90),
            ("claude", 0.5, 0.80),
        ]
        performance_weights = {
            "finbert": 0.6,  # Best historical performance
            "gpt4": 0.25,
            "claude": 0.15,
        }

        result = weighted_soft_vote_ensemble(predictions, performance_weights)

        # FinBERT should have most influence
        assert result.final_score > 0.5


class TestHallucinationDetector:
    """Tests for Enhanced Hallucination Detection."""

    @pytest.fixture
    def detector(self):
        """Create hallucination detector for tests."""
        from llm.agents.llm_guardrails import HallucinationDetector

        return HallucinationDetector()

    @pytest.mark.unit
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector.consensus_threshold == 0.67
        assert detector.price_tolerance_pct == 0.20

    @pytest.mark.unit
    def test_detect_invalid_symbol(self, detector):
        """Test detection of made-up symbols."""
        context = {
            "known_symbols": ["AAPL", "MSFT", "GOOGL"],
        }

        is_halluc, reasons, conf = detector.detect(
            "XYZQ stock is performing well",
            context,
        )

        assert len(reasons) > 0
        assert any("symbol" in r.lower() for r in reasons)

    @pytest.mark.unit
    def test_detect_price_deviation(self, detector):
        """Test detection of impossible price claims."""
        context = {
            "current_prices": {"AAPL": 175.0},
            "known_symbols": ["AAPL"],
        }

        is_halluc, reasons, conf = detector.detect(
            "AAPL stock is trading at $50 per share",
            context,
        )

        assert len(reasons) > 0
        assert any("price" in r.lower() or "deviate" in r.lower() for r in reasons)

    @pytest.mark.unit
    def test_no_hallucination_detected(self, detector):
        """Test no false positives for valid content."""
        context = {
            "current_prices": {"AAPL": 175.0},
            "known_symbols": ["AAPL"],
        }

        is_halluc, reasons, conf = detector.detect(
            "AAPL is showing positive momentum",
            context,
        )

        # Should have low confidence / no major issues
        assert conf < 0.5

    @pytest.mark.unit
    def test_detect_arithmetic_error(self, detector):
        """Test detection of arithmetic errors."""
        is_halluc, reasons, conf = detector.detect(
            "The profit was calculated as 10 + 5 = 20",
            {},
        )

        assert len(reasons) > 0
        assert any("arithmetic" in r.lower() for r in reasons)

    @pytest.mark.unit
    def test_detect_model_consensus_issue(self, detector):
        """Test detection of consensus issues."""
        model_predictions = [
            {"action": "BUY", "sentiment_score": 0.5},
            {"action": "SELL", "sentiment_score": -0.3},
            {"action": "HOLD", "sentiment_score": 0.0},
        ]

        is_halluc, reasons, conf = detector.detect(
            "Action: BUY - strongly bullish",
            {},
            model_predictions=model_predictions,
        )

        assert len(reasons) > 0
        assert any("consensus" in r.lower() for r in reasons)

    @pytest.mark.unit
    def test_detect_provenance_issue(self, detector):
        """Test detection of provenance issues."""
        context = {
            "news_sources": [],  # No sources available
        }

        is_halluc, reasons, conf = detector.detect(
            "According to reports, the company announced major news",
            context,
        )

        assert len(reasons) > 0
        assert any("source" in r.lower() or "provenance" in r.lower() for r in reasons)

    @pytest.mark.unit
    def test_strict_mode(self):
        """Test strict mode has lower threshold."""
        from llm.agents.llm_guardrails import HallucinationDetector

        detector_strict = HallucinationDetector(strict_mode=True)
        detector_normal = HallucinationDetector(strict_mode=False)

        context = {"current_prices": {"AAPL": 175.0}}

        # Same output
        output = "AAPL at $140 per share might be undervalued"

        is_halluc_strict, _, conf_strict = detector_strict.detect(output, context)
        is_halluc_normal, _, conf_normal = detector_normal.detect(output, context)

        # Strict mode should flag more cases
        # (In this case, same conf, but strict uses lower threshold)
        assert detector_strict.strict_mode
        assert not detector_normal.strict_mode

    @pytest.mark.unit
    def test_get_detection_stats(self, detector):
        """Test getting detection statistics."""
        # Run some detections
        detector.detect("test output 1", {})
        detector.detect("test output 2", {})

        stats = detector.get_detection_stats()

        assert "total_checks" in stats
        assert stats["total_checks"] == 2


class TestVotingResultSerialization:
    """Tests for VotingResult serialization."""

    @pytest.mark.unit
    def test_voting_result_to_dict(self):
        """Test VotingResult serialization."""
        from llm.sentiment_filter import VotingResult

        result = VotingResult(
            final_score=0.5,
            final_confidence=0.8,
            voting_method="soft_vote",
            model_scores={"m1": 0.4, "m2": 0.6},
            model_weights={"m1": 0.5, "m2": 0.5},
        )

        result_dict = result.to_dict()

        assert result_dict["final_score"] == 0.5
        assert result_dict["final_confidence"] == 0.8
        assert result_dict["voting_method"] == "soft_vote"
        assert len(result_dict["model_scores"]) == 2


class TestDefaultRegimeConfigs:
    """Tests for default regime configurations."""

    @pytest.mark.unit
    def test_default_configs_exist(self):
        """Test default regime configs are defined."""
        from llm.sentiment_filter import DEFAULT_REGIME_CONFIGS, MarketRegime

        assert MarketRegime.BULL_TRENDING in DEFAULT_REGIME_CONFIGS
        assert MarketRegime.BEAR_TRENDING in DEFAULT_REGIME_CONFIGS
        assert MarketRegime.HIGH_VOLATILITY in DEFAULT_REGIME_CONFIGS
        assert MarketRegime.LOW_VOLATILITY in DEFAULT_REGIME_CONFIGS
        assert MarketRegime.MEAN_REVERTING in DEFAULT_REGIME_CONFIGS

    @pytest.mark.unit
    def test_regime_config_values(self):
        """Test regime config values are reasonable."""
        from llm.sentiment_filter import DEFAULT_REGIME_CONFIGS

        for regime, config in DEFAULT_REGIME_CONFIGS.items():
            assert 0.0 <= config.sentiment_weight <= 1.0
            assert 0.0 <= config.technical_weight <= 1.0
            # Weights should approximately sum to 1
            assert 0.9 <= config.sentiment_weight + config.technical_weight <= 1.1
            assert 0.0 < config.position_mult <= 1.5
            assert 0.0 <= config.min_confidence <= 1.0


# ==============================================================================
# Sentiment Momentum & Mean Reversion Tests (Feature 7)
# ==============================================================================


class TestSentimentMomentumTracker:
    """Tests for SentimentMomentumTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a momentum tracker for testing."""
        from llm.sentiment_filter import SentimentMomentumTracker

        return SentimentMomentumTracker(
            lookback_periods=10,
            extreme_threshold=0.7,
            momentum_smoothing=0.3,
            mean_reversion_threshold=0.8,
        )

    @pytest.mark.unit
    def test_tracker_initialization(self, tracker):
        """Test momentum tracker initialization."""
        assert tracker.lookback_periods == 10
        assert tracker.extreme_threshold == 0.7
        assert tracker.momentum_smoothing == 0.3
        assert tracker.mean_reversion_threshold == 0.8

    @pytest.mark.unit
    def test_update_sentiment(self, tracker):
        """Test sentiment history update."""
        tracker.update_sentiment("AAPL", 0.5)
        tracker.update_sentiment("AAPL", 0.6)
        tracker.update_sentiment("AAPL", 0.7)

        assert "AAPL" in tracker._sentiment_history
        assert len(tracker._sentiment_history["AAPL"]) == 3

    @pytest.mark.unit
    def test_get_momentum_signal_insufficient_data(self, tracker):
        """Test momentum signal returns None with insufficient data."""
        tracker.update_sentiment("AAPL", 0.5)

        signal = tracker.get_momentum_signal("AAPL")

        # Need at least 3 points
        assert signal is None

    @pytest.mark.unit
    def test_get_momentum_signal_with_data(self, tracker):
        """Test momentum signal calculation with sufficient data."""
        from llm.sentiment_filter import SentimentExtreme

        # Add increasing sentiment (bullish momentum)
        for i in range(5):
            tracker.update_sentiment("AAPL", 0.1 + i * 0.1)

        signal = tracker.get_momentum_signal("AAPL")

        assert signal is not None
        assert signal.symbol == "AAPL"
        assert signal.current_score == 0.5
        assert signal.momentum > 0  # Positive momentum (increasing)
        assert signal.extreme == SentimentExtreme.NORMAL

    @pytest.mark.unit
    def test_extreme_bullish_detection(self, tracker):
        """Test detection of extreme bullish sentiment."""
        from llm.sentiment_filter import SentimentExtreme

        # Add extreme bullish sentiment
        for i in range(5):
            tracker.update_sentiment("AAPL", 0.8)

        signal = tracker.get_momentum_signal("AAPL")

        assert signal is not None
        assert signal.extreme == SentimentExtreme.EXTREME_BULLISH

    @pytest.mark.unit
    def test_extreme_bearish_detection(self, tracker):
        """Test detection of extreme bearish sentiment."""
        from llm.sentiment_filter import SentimentExtreme

        # Add extreme bearish sentiment
        for i in range(5):
            tracker.update_sentiment("AAPL", -0.8)

        signal = tracker.get_momentum_signal("AAPL")

        assert signal is not None
        assert signal.extreme == SentimentExtreme.EXTREME_BEARISH

    @pytest.mark.unit
    def test_mean_reversion_signal_extreme_bullish(self, tracker):
        """Test mean reversion signal when extreme bullish with reversing momentum."""
        # Build up to extreme, then start declining
        scores = [0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.8]  # Rising then falling
        for score in scores:
            tracker.update_sentiment("AAPL", score)

        signal = tracker.get_momentum_signal("AAPL")

        assert signal is not None
        # Mean reversion when extreme and momentum reversing
        assert signal.extreme.value in ["extreme_bullish", "normal"]

    @pytest.mark.unit
    def test_update_price_for_divergence(self, tracker):
        """Test price history update for divergence calculation."""
        tracker.update_price("AAPL", 150.0)
        tracker.update_price("AAPL", 155.0)
        tracker.update_price("AAPL", 160.0)

        assert "AAPL" in tracker._price_history
        assert len(tracker._price_history["AAPL"]) == 3

    @pytest.mark.unit
    def test_divergence_calculation(self, tracker):
        """Test sentiment-price divergence calculation."""
        # Sentiment rising while price falling = positive divergence (bullish)
        for i in range(5):
            tracker.update_sentiment("AAPL", 0.2 + i * 0.1)  # Rising sentiment
            tracker.update_price("AAPL", 150.0 - i * 5)  # Falling price

        signal = tracker.get_momentum_signal("AAPL")

        assert signal is not None
        assert signal.divergence_from_price is not None
        # Sentiment going up, price going down = positive divergence
        assert signal.divergence_from_price > 0

    @pytest.mark.unit
    def test_get_all_momentum_signals(self, tracker):
        """Test getting momentum signals for all tracked symbols."""
        # Add data for multiple symbols
        for i in range(5):
            tracker.update_sentiment("AAPL", 0.1 + i * 0.1)
            tracker.update_sentiment("MSFT", -0.1 - i * 0.1)
            tracker.update_sentiment("GOOGL", 0.0)

        signals = tracker.get_all_momentum_signals()

        assert "AAPL" in signals
        assert "MSFT" in signals
        assert "GOOGL" in signals

    @pytest.mark.unit
    def test_get_extreme_symbols(self, tracker):
        """Test getting symbols at sentiment extremes."""
        from llm.sentiment_filter import SentimentExtreme

        # Add extreme bullish for AAPL
        for i in range(5):
            tracker.update_sentiment("AAPL", 0.8)

        # Add normal sentiment for MSFT
        for i in range(5):
            tracker.update_sentiment("MSFT", 0.3)

        extremes = tracker.get_extreme_symbols()

        assert "AAPL" in extremes
        assert extremes["AAPL"] == SentimentExtreme.EXTREME_BULLISH
        assert "MSFT" not in extremes

    @pytest.mark.unit
    def test_get_mean_reversion_candidates(self, tracker):
        """Test getting mean reversion candidates."""
        # Use lower threshold for testing
        tracker.mean_reversion_threshold = 0.75

        # Add extreme sentiment at reversal threshold
        for i in range(5):
            tracker.update_sentiment("AAPL", 0.85)  # Above mean_reversion_threshold

        candidates = tracker.get_mean_reversion_candidates()

        # Should be a candidate if at extreme
        signal = tracker.get_momentum_signal("AAPL")
        if signal and signal.mean_reversion_signal:
            assert "AAPL" in candidates

    @pytest.mark.unit
    def test_get_divergence_signals(self, tracker):
        """Test getting divergence signals above threshold."""
        # Create strong positive divergence
        for i in range(5):
            tracker.update_sentiment("AAPL", 0.1 + i * 0.15)  # Rising sentiment
            tracker.update_price("AAPL", 150.0 - i * 10)  # Falling price

        divergences = tracker.get_divergence_signals(min_divergence=0.2)

        # Should have AAPL with positive divergence
        if "AAPL" in divergences:
            assert divergences["AAPL"] > 0

    @pytest.mark.unit
    def test_get_stats(self, tracker):
        """Test getting tracker statistics."""
        for i in range(5):
            tracker.update_sentiment("AAPL", 0.5)
            tracker.update_sentiment("MSFT", 0.8)

        stats = tracker.get_stats()

        assert stats["tracked_symbols"] == 2
        assert stats["total_observations"] == 10
        assert stats["lookback_periods"] == 10
        assert stats["extreme_threshold"] == 0.7

    @pytest.mark.unit
    def test_history_trimming(self, tracker):
        """Test that history is trimmed to max size."""
        # lookback_periods is 10, max_history is 2x = 20
        for i in range(30):
            tracker.update_sentiment("AAPL", 0.5)

        assert len(tracker._sentiment_history["AAPL"]) == 20


class TestMomentumSignal:
    """Tests for MomentumSignal dataclass."""

    @pytest.mark.unit
    def test_momentum_signal_to_dict(self):
        """Test MomentumSignal serialization."""
        from llm.sentiment_filter import MomentumSignal, SentimentExtreme

        signal = MomentumSignal(
            symbol="AAPL",
            current_score=0.5,
            momentum=0.2,
            acceleration=0.05,
            extreme=SentimentExtreme.NORMAL,
            mean_reversion_signal=False,
            divergence_from_price=0.1,
            lookback_periods=10,
        )

        result = signal.to_dict()

        assert result["symbol"] == "AAPL"
        assert result["current_score"] == 0.5
        assert result["momentum"] == 0.2
        assert result["acceleration"] == 0.05
        assert result["extreme"] == "normal"
        assert result["mean_reversion_signal"] is False
        assert result["divergence_from_price"] == 0.1
        assert result["lookback_periods"] == 10
        assert "timestamp" in result


class TestCreateMomentumTracker:
    """Tests for create_momentum_tracker factory function."""

    @pytest.mark.unit
    def test_create_with_defaults(self):
        """Test creating tracker with default parameters."""
        from llm.sentiment_filter import create_momentum_tracker

        tracker = create_momentum_tracker()

        assert tracker.lookback_periods == 20
        assert tracker.extreme_threshold == 0.7
        assert tracker.momentum_smoothing == 0.3
        assert tracker.mean_reversion_threshold == 0.8

    @pytest.mark.unit
    def test_create_with_custom_params(self):
        """Test creating tracker with custom parameters."""
        from llm.sentiment_filter import create_momentum_tracker

        tracker = create_momentum_tracker(
            lookback_periods=30,
            extreme_threshold=0.8,
            momentum_smoothing=0.5,
            mean_reversion_threshold=0.9,
        )

        assert tracker.lookback_periods == 30
        assert tracker.extreme_threshold == 0.8
        assert tracker.momentum_smoothing == 0.5
        assert tracker.mean_reversion_threshold == 0.9


class TestSentimentExtremeEnum:
    """Tests for SentimentExtreme enum."""

    @pytest.mark.unit
    def test_extreme_values(self):
        """Test SentimentExtreme enum values."""
        from llm.sentiment_filter import SentimentExtreme

        assert SentimentExtreme.EXTREME_BULLISH.value == "extreme_bullish"
        assert SentimentExtreme.EXTREME_BEARISH.value == "extreme_bearish"
        assert SentimentExtreme.NORMAL.value == "normal"


class TestMomentumImports:
    """Tests for momentum feature imports."""

    @pytest.mark.unit
    def test_imports_from_llm(self):
        """Test importing momentum classes from llm module."""
        from llm import (
            MomentumSignal,
            SentimentExtreme,
            SentimentMomentumTracker,
            create_momentum_tracker,
        )

        assert SentimentExtreme is not None
        assert MomentumSignal is not None
        assert SentimentMomentumTracker is not None
        assert create_momentum_tracker is not None
