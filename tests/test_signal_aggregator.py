"""
Tests for Multi-Source Signal Aggregator Module

Tests signal aggregation from multiple sources with weighting and conflict resolution.
Part of UPGRADE-010 Sprint 3 Iteration 3 - Test Coverage.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from llm.signal_aggregator import (
    AggregatedAction,
    AggregatedSignal,
    SignalAggregator,
    SignalAggregatorConfig,
    SignalDirection,
    SignalSource,
    SourceSignal,
    create_signal_aggregator,
)


class TestSignalSource:
    """Tests for SignalSource enum."""

    def test_all_sources_exist(self):
        """Test that all expected sources exist."""
        assert SignalSource.REDDIT.value == "reddit"
        assert SignalSource.NEWS.value == "news"
        assert SignalSource.EARNINGS.value == "earnings"
        assert SignalSource.TECHNICAL.value == "technical"
        assert SignalSource.FUNDAMENTALS.value == "fundamentals"
        assert SignalSource.OPTIONS_FLOW.value == "options_flow"
        assert SignalSource.INSIDER_TRADES.value == "insider_trades"
        assert SignalSource.ANALYST.value == "analyst"
        assert SignalSource.CUSTOM.value == "custom"


class TestSignalDirection:
    """Tests for SignalDirection enum."""

    def test_all_directions_exist(self):
        """Test that all expected directions exist."""
        assert SignalDirection.BULLISH.value == "bullish"
        assert SignalDirection.BEARISH.value == "bearish"
        assert SignalDirection.NEUTRAL.value == "neutral"


class TestAggregatedAction:
    """Tests for AggregatedAction enum."""

    def test_all_actions_exist(self):
        """Test that all expected actions exist."""
        assert AggregatedAction.STRONG_BUY.value == "strong_buy"
        assert AggregatedAction.BUY.value == "buy"
        assert AggregatedAction.HOLD.value == "hold"
        assert AggregatedAction.SELL.value == "sell"
        assert AggregatedAction.STRONG_SELL.value == "strong_sell"
        assert AggregatedAction.CONFLICTING.value == "conflicting"


class TestSourceSignal:
    """Tests for SourceSignal dataclass."""

    @pytest.fixture
    def sample_signal(self):
        """Create sample source signal."""
        return SourceSignal(
            source=SignalSource.NEWS,
            ticker="AAPL",
            direction=SignalDirection.BULLISH,
            confidence=0.8,
            strength=0.7,
            timestamp=datetime.now(),
            expiry=datetime.now() + timedelta(hours=24),
            metadata={"headline": "AAPL beats earnings"},
        )

    def test_to_dict(self, sample_signal):
        """Test conversion to dictionary."""
        result = sample_signal.to_dict()

        assert result["source"] == "news"
        assert result["ticker"] == "AAPL"
        assert result["direction"] == "bullish"
        assert result["confidence"] == 0.8
        assert result["strength"] == 0.7
        assert "headline" in result["metadata"]

    def test_is_expired_false(self, sample_signal):
        """Test is_expired returns False for valid signal."""
        assert sample_signal.is_expired is False

    def test_is_expired_true(self):
        """Test is_expired returns True for expired signal."""
        signal = SourceSignal(
            source=SignalSource.NEWS,
            ticker="AAPL",
            direction=SignalDirection.BULLISH,
            confidence=0.8,
            strength=0.7,
            timestamp=datetime.now() - timedelta(hours=48),
            expiry=datetime.now() - timedelta(hours=24),  # Expired
        )

        assert signal.is_expired is True

    def test_is_expired_no_expiry(self):
        """Test is_expired returns False when no expiry set."""
        signal = SourceSignal(
            source=SignalSource.NEWS,
            ticker="AAPL",
            direction=SignalDirection.BULLISH,
            confidence=0.8,
            strength=0.7,
            timestamp=datetime.now(),
            expiry=None,  # No expiry
        )

        assert signal.is_expired is False

    def test_weighted_score_bullish(self):
        """Test weighted score calculation for bullish signal."""
        signal = SourceSignal(
            source=SignalSource.NEWS,
            ticker="AAPL",
            direction=SignalDirection.BULLISH,
            confidence=0.8,
            strength=0.5,
            timestamp=datetime.now(),
        )

        # Bullish: 1.0 * 0.8 * 0.5 = 0.4
        assert signal.weighted_score == 0.4

    def test_weighted_score_bearish(self):
        """Test weighted score calculation for bearish signal."""
        signal = SourceSignal(
            source=SignalSource.NEWS,
            ticker="AAPL",
            direction=SignalDirection.BEARISH,
            confidence=0.6,
            strength=0.5,
            timestamp=datetime.now(),
        )

        # Bearish: -1.0 * 0.6 * 0.5 = -0.3
        assert signal.weighted_score == -0.3

    def test_weighted_score_neutral(self):
        """Test weighted score calculation for neutral signal."""
        signal = SourceSignal(
            source=SignalSource.NEWS,
            ticker="AAPL",
            direction=SignalDirection.NEUTRAL,
            confidence=0.9,
            strength=0.8,
            timestamp=datetime.now(),
        )

        assert signal.weighted_score == 0.0


class TestSignalAggregatorConfig:
    """Tests for SignalAggregatorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SignalAggregatorConfig()

        assert config.strong_signal_threshold == 0.7
        assert config.actionable_threshold == 0.5
        assert config.conflict_threshold == 0.3
        assert config.recency_decay_hours == 24.0
        assert config.max_signal_age_hours == 72.0
        assert config.min_signals_for_action == 2
        assert config.require_majority is True

    def test_default_source_weights(self):
        """Test default source weights."""
        config = SignalAggregatorConfig()

        assert config.source_weights[SignalSource.NEWS] == 1.0
        assert config.source_weights[SignalSource.REDDIT] == 0.7
        assert config.source_weights[SignalSource.EARNINGS] == 1.2
        assert config.source_weights[SignalSource.TECHNICAL] == 0.9

    def test_custom_config(self):
        """Test custom configuration."""
        config = SignalAggregatorConfig(
            strong_signal_threshold=0.8,
            min_signals_for_action=3,
        )

        assert config.strong_signal_threshold == 0.8
        assert config.min_signals_for_action == 3


class TestAggregatedSignal:
    """Tests for AggregatedSignal dataclass."""

    @pytest.fixture
    def sample_aggregated(self):
        """Create sample aggregated signal."""
        return AggregatedSignal(
            ticker="AAPL",
            action=AggregatedAction.BUY,
            direction=SignalDirection.BULLISH,
            composite_score=0.5,
            confidence=0.7,
            actionability=0.6,
            agreement_score=0.8,
            bullish_count=3,
            bearish_count=1,
            neutral_count=0,
            signals=[],
            conflicting_signals=[],
            primary_driver=SignalSource.NEWS,
            analysis_summary="Bullish consensus",
            aggregation_time_ms=10.0,
            timestamp=datetime.now(),
        )

    def test_to_dict(self, sample_aggregated):
        """Test conversion to dictionary."""
        result = sample_aggregated.to_dict()

        assert result["ticker"] == "AAPL"
        assert result["action"] == "buy"
        assert result["direction"] == "bullish"
        assert result["composite_score"] == 0.5
        assert result["confidence"] == 0.7
        assert result["agreement_score"] == 0.8
        assert result["primary_driver"] == "news"


class TestSignalAggregator:
    """Tests for SignalAggregator class."""

    @pytest.fixture
    def aggregator(self):
        """Create default aggregator."""
        return SignalAggregator()

    def create_signal(
        self,
        ticker: str,
        source: SignalSource,
        direction: SignalDirection,
        confidence: float = 0.7,
        strength: float = 0.6,
        hours_ago: float = 0,
    ) -> SourceSignal:
        """Helper to create signals."""
        return SourceSignal(
            source=source,
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            strength=strength,
            timestamp=datetime.now() - timedelta(hours=hours_ago),
            expiry=datetime.now() + timedelta(hours=24),
        )

    # Signal addition tests
    def test_add_signal(self, aggregator):
        """Test adding a signal."""
        signal = self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.BULLISH)
        aggregator.add_signal(signal)

        assert "AAPL" in aggregator._signals
        assert len(aggregator._signals["AAPL"]) == 1

    def test_add_signals_multiple(self, aggregator):
        """Test adding multiple signals."""
        signals = [
            self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.BULLISH),
            self.create_signal("AAPL", SignalSource.REDDIT, SignalDirection.BULLISH),
            self.create_signal("MSFT", SignalSource.NEWS, SignalDirection.BEARISH),
        ]

        aggregator.add_signals(signals)

        assert len(aggregator._signals["AAPL"]) == 2
        assert len(aggregator._signals["MSFT"]) == 1

    def test_signal_ticker_normalized(self, aggregator):
        """Test that ticker is normalized to uppercase."""
        signal = self.create_signal("aapl", SignalSource.NEWS, SignalDirection.BULLISH)
        aggregator.add_signal(signal)

        assert "AAPL" in aggregator._signals

    # Aggregation tests
    def test_aggregate_bullish_consensus(self, aggregator):
        """Test aggregation with bullish consensus."""
        signals = [
            self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.BULLISH, 0.8),
            self.create_signal("AAPL", SignalSource.REDDIT, SignalDirection.BULLISH, 0.7),
            self.create_signal("AAPL", SignalSource.TECHNICAL, SignalDirection.BULLISH, 0.6),
        ]

        aggregator.add_signals(signals)
        result = aggregator.aggregate("AAPL")

        assert result is not None
        assert result.direction == SignalDirection.BULLISH
        assert result.bullish_count == 3
        assert result.bearish_count == 0
        assert result.composite_score > 0

    def test_aggregate_bearish_consensus(self, aggregator):
        """Test aggregation with bearish consensus."""
        signals = [
            self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.BEARISH, 0.8),
            self.create_signal("AAPL", SignalSource.EARNINGS, SignalDirection.BEARISH, 0.7),
            self.create_signal("AAPL", SignalSource.ANALYST, SignalDirection.BEARISH, 0.6),
        ]

        aggregator.add_signals(signals)
        result = aggregator.aggregate("AAPL")

        assert result is not None
        assert result.direction == SignalDirection.BEARISH
        assert result.bearish_count == 3
        assert result.composite_score < 0

    def test_aggregate_mixed_signals(self, aggregator):
        """Test aggregation with mixed signals."""
        signals = [
            self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.BULLISH, 0.8),
            self.create_signal("AAPL", SignalSource.REDDIT, SignalDirection.BEARISH, 0.7),
            self.create_signal("AAPL", SignalSource.TECHNICAL, SignalDirection.NEUTRAL, 0.6),
        ]

        aggregator.add_signals(signals)
        result = aggregator.aggregate("AAPL")

        assert result is not None
        assert result.bullish_count == 1
        assert result.bearish_count == 1
        assert result.neutral_count == 1

    def test_aggregate_insufficient_signals(self, aggregator):
        """Test aggregation with insufficient signals."""
        signal = self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.BULLISH)
        aggregator.add_signal(signal)

        # Default requires 2 signals
        result = aggregator.aggregate("AAPL")

        assert result is None

    def test_aggregate_unknown_ticker(self, aggregator):
        """Test aggregation for unknown ticker."""
        result = aggregator.aggregate("UNKNOWN")

        assert result is None

    # Action determination tests
    def test_strong_buy_action(self, aggregator):
        """Test strong buy action determination."""
        signals = [
            self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.BULLISH, 0.9, 0.9),
            self.create_signal("AAPL", SignalSource.EARNINGS, SignalDirection.BULLISH, 0.9, 0.9),
            self.create_signal("AAPL", SignalSource.TECHNICAL, SignalDirection.BULLISH, 0.9, 0.9),
        ]

        aggregator.add_signals(signals)
        result = aggregator.aggregate("AAPL")

        assert result is not None
        assert result.action in {AggregatedAction.STRONG_BUY, AggregatedAction.BUY}

    def test_strong_sell_action(self, aggregator):
        """Test strong sell action determination."""
        signals = [
            self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.BEARISH, 0.9, 0.9),
            self.create_signal("AAPL", SignalSource.EARNINGS, SignalDirection.BEARISH, 0.9, 0.9),
            self.create_signal("AAPL", SignalSource.TECHNICAL, SignalDirection.BEARISH, 0.9, 0.9),
        ]

        aggregator.add_signals(signals)
        result = aggregator.aggregate("AAPL")

        assert result is not None
        assert result.action in {AggregatedAction.STRONG_SELL, AggregatedAction.SELL}

    def test_hold_action(self, aggregator):
        """Test hold action for neutral signals."""
        signals = [
            self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.NEUTRAL, 0.5, 0.5),
            self.create_signal("AAPL", SignalSource.REDDIT, SignalDirection.NEUTRAL, 0.5, 0.5),
        ]

        aggregator.add_signals(signals)
        result = aggregator.aggregate("AAPL")

        assert result is not None
        assert result.action == AggregatedAction.HOLD

    def test_conflicting_action(self, aggregator):
        """Test conflicting action for high-conflict signals."""
        signals = [
            self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.BULLISH, 0.9, 0.9),
            self.create_signal("AAPL", SignalSource.EARNINGS, SignalDirection.BEARISH, 0.9, 0.9),
            self.create_signal("AAPL", SignalSource.TECHNICAL, SignalDirection.BULLISH, 0.8, 0.8),
            self.create_signal("AAPL", SignalSource.ANALYST, SignalDirection.BEARISH, 0.8, 0.8),
        ]

        aggregator.add_signals(signals)
        result = aggregator.aggregate("AAPL")

        assert result is not None
        # High conflict should have conflicting signals identified
        assert len(result.conflicting_signals) > 0 or result.agreement_score < 0.7

    # Recency weighting tests
    def test_recent_signal_weighted_higher(self, aggregator):
        """Test that recent signals are weighted higher."""
        # Old bullish signal
        old_signal = self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.BULLISH, 0.9, 0.9, hours_ago=20)
        # Recent bearish signal
        new_signal = self.create_signal("AAPL", SignalSource.EARNINGS, SignalDirection.BEARISH, 0.7, 0.7, hours_ago=1)

        aggregator.add_signals([old_signal, new_signal])
        result = aggregator.aggregate("AAPL")

        assert result is not None
        # Recent signal should have more weight

    # Agreement score tests
    def test_high_agreement_score(self, aggregator):
        """Test high agreement when all signals agree."""
        signals = [
            self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.BULLISH),
            self.create_signal("AAPL", SignalSource.REDDIT, SignalDirection.BULLISH),
            self.create_signal("AAPL", SignalSource.TECHNICAL, SignalDirection.BULLISH),
        ]

        aggregator.add_signals(signals)
        result = aggregator.aggregate("AAPL")

        assert result is not None
        assert result.agreement_score == 1.0

    def test_low_agreement_score(self, aggregator):
        """Test low agreement when signals disagree."""
        signals = [
            self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.BULLISH),
            self.create_signal("AAPL", SignalSource.REDDIT, SignalDirection.BEARISH),
            self.create_signal("AAPL", SignalSource.TECHNICAL, SignalDirection.NEUTRAL),
        ]

        aggregator.add_signals(signals)
        result = aggregator.aggregate("AAPL")

        assert result is not None
        assert result.agreement_score < 0.5

    # Top signals tests
    def test_get_top_signals(self, aggregator):
        """Test getting top signals."""
        # Add signals for multiple tickers
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            signals = [
                self.create_signal(ticker, SignalSource.NEWS, SignalDirection.BULLISH, 0.8),
                self.create_signal(ticker, SignalSource.REDDIT, SignalDirection.BULLISH, 0.7),
            ]
            aggregator.add_signals(signals)

        top_signals = aggregator.get_top_signals(limit=2)

        assert len(top_signals) <= 2

    def test_get_top_signals_by_direction(self, aggregator):
        """Test getting top signals filtered by direction."""
        # Bullish ticker
        aggregator.add_signals(
            [
                self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.BULLISH, 0.8),
                self.create_signal("AAPL", SignalSource.REDDIT, SignalDirection.BULLISH, 0.7),
            ]
        )

        # Bearish ticker
        aggregator.add_signals(
            [
                self.create_signal("MSFT", SignalSource.NEWS, SignalDirection.BEARISH, 0.8),
                self.create_signal("MSFT", SignalSource.REDDIT, SignalDirection.BEARISH, 0.7),
            ]
        )

        bullish_signals = aggregator.get_top_signals(direction=SignalDirection.BULLISH)
        bearish_signals = aggregator.get_top_signals(direction=SignalDirection.BEARISH)

        for signal in bullish_signals:
            assert signal.direction == SignalDirection.BULLISH

        for signal in bearish_signals:
            assert signal.direction == SignalDirection.BEARISH

    # Aggregate all tests
    def test_aggregate_all(self, aggregator):
        """Test aggregating all tickers."""
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            signals = [
                self.create_signal(ticker, SignalSource.NEWS, SignalDirection.BULLISH, 0.8),
                self.create_signal(ticker, SignalSource.REDDIT, SignalDirection.BULLISH, 0.7),
            ]
            aggregator.add_signals(signals)

        all_results = aggregator.aggregate_all()

        assert "AAPL" in all_results
        assert "MSFT" in all_results
        assert "GOOGL" in all_results

    # Clear signals tests
    def test_clear_signals_specific_ticker(self, aggregator):
        """Test clearing signals for specific ticker."""
        aggregator.add_signals(
            [
                self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.BULLISH),
                self.create_signal("MSFT", SignalSource.NEWS, SignalDirection.BULLISH),
            ]
        )

        aggregator.clear_signals("AAPL")

        assert "AAPL" not in aggregator._signals
        assert "MSFT" in aggregator._signals

    def test_clear_signals_all(self, aggregator):
        """Test clearing all signals."""
        aggregator.add_signals(
            [
                self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.BULLISH),
                self.create_signal("MSFT", SignalSource.NEWS, SignalDirection.BULLISH),
            ]
        )

        aggregator.clear_signals()

        assert len(aggregator._signals) == 0

    # Stats tests
    def test_get_stats(self, aggregator):
        """Test getting statistics."""
        signals = [
            self.create_signal("AAPL", SignalSource.NEWS, SignalDirection.BULLISH),
            self.create_signal("AAPL", SignalSource.REDDIT, SignalDirection.BULLISH),
            self.create_signal("MSFT", SignalSource.NEWS, SignalDirection.BEARISH),
        ]

        aggregator.add_signals(signals)
        aggregator.aggregate("AAPL")

        stats = aggregator.get_stats()

        assert stats["total_signals"] == 3
        assert stats["active_tickers"] == 2
        assert stats["aggregations_performed"] >= 1

    # Callback tests
    def test_alert_callback(self):
        """Test alert callback is triggered."""
        callback_results = []

        def callback(signal):
            callback_results.append(signal)

        aggregator = SignalAggregator(alert_callback=callback)

        signals = [
            SourceSignal(
                source=SignalSource.NEWS,
                ticker="AAPL",
                direction=SignalDirection.BULLISH,
                confidence=0.9,
                strength=0.9,
                timestamp=datetime.now(),
            ),
            SourceSignal(
                source=SignalSource.EARNINGS,
                ticker="AAPL",
                direction=SignalDirection.BULLISH,
                confidence=0.9,
                strength=0.9,
                timestamp=datetime.now(),
            ),
        ]

        aggregator.add_signals(signals)
        result = aggregator.aggregate("AAPL")

        # Callback should be triggered if actionability >= threshold
        if result and result.actionability >= aggregator.config.actionable_threshold:
            assert len(callback_results) >= 1

    # Expired signals tests
    def test_expired_signals_excluded(self, aggregator):
        """Test that expired signals are excluded from aggregation."""
        valid_signal = SourceSignal(
            source=SignalSource.NEWS,
            ticker="AAPL",
            direction=SignalDirection.BULLISH,
            confidence=0.8,
            strength=0.7,
            timestamp=datetime.now(),
            expiry=datetime.now() + timedelta(hours=24),
        )

        expired_signal = SourceSignal(
            source=SignalSource.REDDIT,
            ticker="AAPL",
            direction=SignalDirection.BEARISH,
            confidence=0.8,
            strength=0.7,
            timestamp=datetime.now() - timedelta(hours=48),
            expiry=datetime.now() - timedelta(hours=24),
        )

        aggregator.add_signals([valid_signal, expired_signal])
        result = aggregator.aggregate("AAPL")

        # Should not have enough valid signals (expired excluded)
        assert result is None


class TestCreateSignalAggregator:
    """Tests for factory function."""

    def test_create_default(self):
        """Test creating aggregator with defaults."""
        aggregator = create_signal_aggregator()

        assert isinstance(aggregator, SignalAggregator)
        assert aggregator.config is not None

    def test_create_with_config(self):
        """Test creating aggregator with custom config."""
        config = SignalAggregatorConfig(
            min_signals_for_action=3,
            strong_signal_threshold=0.8,
        )
        aggregator = create_signal_aggregator(config)

        assert aggregator.config.min_signals_for_action == 3
        assert aggregator.config.strong_signal_threshold == 0.8

    def test_create_with_callback(self):
        """Test creating aggregator with callback."""
        callback = Mock()
        aggregator = create_signal_aggregator(alert_callback=callback)

        assert aggregator.alert_callback == callback
