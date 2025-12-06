"""
Tests for Adaptive Cancel Timing Optimizer

Tests the cancel timing optimization module.
Part of UPGRADE-010 Sprint 4 - Test Coverage.
"""

import pytest

from execution.cancel_optimizer import (
    CancelDecision,
    CancelOptimizer,
    CancelReason,
    CancelTimingFeatures,
    TimingStatistics,
    create_cancel_optimizer,
)
from execution.fill_predictor import FillOutcome


class TestCancelTimingFeatures:
    """Tests for CancelTimingFeatures dataclass."""

    def test_creation(self):
        """Test feature creation."""
        features = CancelTimingFeatures(
            spread_bps=20.0,
            fill_probability=0.5,
            time_since_submit=1.5,
            partial_fill_pct=0.0,
            volatility_regime="normal",
            order_age_percentile=0.3,
            price_movement_since_submit=0.001,
            spread_change_since_submit=2.0,
            volume_since_submit=50,
        )

        assert features.spread_bps == 20.0
        assert features.fill_probability == 0.5
        assert features.time_since_submit == 1.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        features = CancelTimingFeatures(
            spread_bps=15.0,
            fill_probability=0.6,
            time_since_submit=2.0,
            partial_fill_pct=0.25,
            volatility_regime="high",
            order_age_percentile=0.5,
            price_movement_since_submit=-0.002,
            spread_change_since_submit=5.0,
            volume_since_submit=100,
        )

        d = features.to_dict()

        assert d["spread_bps"] == 15.0
        assert d["fill_probability"] == 0.6
        assert d["volatility_regime"] == "high"


class TestCancelDecision:
    """Tests for CancelDecision dataclass."""

    def test_creation(self):
        """Test decision creation."""
        decision = CancelDecision(
            should_cancel=True,
            optimal_wait_seconds=2.5,
            confidence=0.8,
            reason=CancelReason.TIMEOUT_EXCEEDED,
            reasoning="Order exceeded timeout",
        )

        assert decision.should_cancel is True
        assert decision.optimal_wait_seconds == 2.5
        assert decision.confidence == 0.8

    def test_to_dict(self):
        """Test conversion to dictionary."""
        decision = CancelDecision(
            should_cancel=False,
            optimal_wait_seconds=3.0,
            confidence=0.7,
            reason=CancelReason.LOW_FILL_PROBABILITY,
            reasoning="Low fill probability",
            suggested_price_adjustment=0.05,
        )

        d = decision.to_dict()

        assert d["should_cancel"] is False
        assert d["reason"] == "low_fill_probability"
        assert d["suggested_price_adjustment"] == 0.05


class TestCancelOptimizer:
    """Tests for CancelOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create default optimizer."""
        return CancelOptimizer(base_timeout=2.5)

    @pytest.fixture
    def sample_features(self):
        """Create sample features."""
        return CancelTimingFeatures(
            spread_bps=20.0,
            fill_probability=0.5,
            time_since_submit=1.0,
            partial_fill_pct=0.0,
            volatility_regime="normal",
            order_age_percentile=0.3,
            price_movement_since_submit=0.001,
            spread_change_since_submit=2.0,
            volume_since_submit=50,
        )

    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.base_timeout == 2.5
        assert optimizer.learning_rate == 0.1
        assert optimizer.min_samples == 20

    def test_get_optimal_timeout_base(self, optimizer, sample_features):
        """Test basic timeout calculation."""
        decision = optimizer.get_optimal_timeout(sample_features)

        assert isinstance(decision, CancelDecision)
        assert decision.optimal_wait_seconds > 0
        assert decision.optimal_wait_seconds <= CancelOptimizer.MAX_TIMEOUT

    def test_timeout_adjusts_for_volatility(self, optimizer):
        """Test timeout adjusts for volatility regime."""
        low_vol = CancelTimingFeatures(
            spread_bps=20.0,
            fill_probability=0.5,
            time_since_submit=1.0,
            partial_fill_pct=0.0,
            volatility_regime="low",
            order_age_percentile=0.3,
            price_movement_since_submit=0.001,
            spread_change_since_submit=0.0,
            volume_since_submit=50,
        )

        high_vol = CancelTimingFeatures(
            spread_bps=20.0,
            fill_probability=0.5,
            time_since_submit=1.0,
            partial_fill_pct=0.0,
            volatility_regime="high",
            order_age_percentile=0.3,
            price_movement_since_submit=0.001,
            spread_change_since_submit=0.0,
            volume_since_submit=50,
        )

        low_decision = optimizer.get_optimal_timeout(low_vol)
        high_decision = optimizer.get_optimal_timeout(high_vol)

        # Low volatility should have longer timeout
        assert low_decision.optimal_wait_seconds > high_decision.optimal_wait_seconds

    def test_timeout_adjusts_for_spread(self, optimizer):
        """Test timeout adjusts for spread width."""
        tight = CancelTimingFeatures(
            spread_bps=5.0,
            fill_probability=0.5,
            time_since_submit=1.0,
            partial_fill_pct=0.0,
            volatility_regime="normal",
            order_age_percentile=0.3,
            price_movement_since_submit=0.001,
            spread_change_since_submit=0.0,
            volume_since_submit=50,
        )

        wide = CancelTimingFeatures(
            spread_bps=100.0,
            fill_probability=0.5,
            time_since_submit=1.0,
            partial_fill_pct=0.0,
            volatility_regime="normal",
            order_age_percentile=0.3,
            price_movement_since_submit=0.001,
            spread_change_since_submit=0.0,
            volume_since_submit=50,
        )

        tight_decision = optimizer.get_optimal_timeout(tight)
        wide_decision = optimizer.get_optimal_timeout(wide)

        # Tight spread should allow longer wait
        assert tight_decision.optimal_wait_seconds > wide_decision.optimal_wait_seconds

    def test_timeout_adjusts_for_fill_probability(self, optimizer):
        """Test timeout adjusts for fill probability."""
        high_prob = CancelTimingFeatures(
            spread_bps=20.0,
            fill_probability=0.8,
            time_since_submit=1.0,
            partial_fill_pct=0.0,
            volatility_regime="normal",
            order_age_percentile=0.3,
            price_movement_since_submit=0.001,
            spread_change_since_submit=0.0,
            volume_since_submit=50,
        )

        low_prob = CancelTimingFeatures(
            spread_bps=20.0,
            fill_probability=0.2,
            time_since_submit=1.0,
            partial_fill_pct=0.0,
            volatility_regime="normal",
            order_age_percentile=0.3,
            price_movement_since_submit=0.001,
            spread_change_since_submit=0.0,
            volume_since_submit=50,
        )

        high_decision = optimizer.get_optimal_timeout(high_prob)
        low_decision = optimizer.get_optimal_timeout(low_prob)

        # High probability should allow longer wait
        assert high_decision.optimal_wait_seconds > low_decision.optimal_wait_seconds

    def test_should_cancel_when_timeout_exceeded(self, optimizer):
        """Test should_cancel is True when timeout exceeded."""
        features = CancelTimingFeatures(
            spread_bps=20.0,
            fill_probability=0.5,
            time_since_submit=10.0,  # Long time
            partial_fill_pct=0.0,
            volatility_regime="normal",
            order_age_percentile=0.9,
            price_movement_since_submit=0.001,
            spread_change_since_submit=0.0,
            volume_since_submit=50,
        )

        decision = optimizer.get_optimal_timeout(features)

        assert decision.should_cancel is True
        assert decision.reason in [
            CancelReason.TIMEOUT_EXCEEDED,
            CancelReason.LOW_FILL_PROBABILITY,
        ]

    def test_should_not_cancel_early(self, optimizer, sample_features):
        """Test should_cancel is False early in order life."""
        decision = optimizer.get_optimal_timeout(sample_features)

        # With 1 second elapsed and base 2.5s timeout, shouldn't cancel
        assert decision.should_cancel is False

    def test_record_outcome(self, optimizer, sample_features):
        """Test recording fill outcome."""
        optimizer.record_outcome(
            order_id="TEST001",
            symbol="SPY",
            outcome=FillOutcome.FILLED,
            time_to_outcome=2.0,
            features=sample_features,
            fill_pct=1.0,
        )

        assert len(optimizer.history["SPY"]) == 1
        assert len(optimizer.fill_times) == 1

    def test_statistics_update(self, optimizer, sample_features):
        """Test statistics update after recording outcomes."""
        # Record multiple outcomes
        for i in range(10):
            filled = i % 3 != 0
            optimizer.record_outcome(
                order_id=f"TEST{i:03d}",
                symbol="SPY",
                outcome=FillOutcome.FILLED if filled else FillOutcome.NOT_FILLED,
                time_to_outcome=2.0 + i * 0.2,
                features=sample_features,
                fill_pct=1.0 if filled else 0.0,
            )

        stats = optimizer.get_statistics("SPY")

        assert stats is not None
        assert stats.total_orders == 10
        assert stats.filled_count > 0

    def test_partial_fill_extends_timeout(self, optimizer):
        """Test partial fills extend the timeout."""
        no_fill = CancelTimingFeatures(
            spread_bps=20.0,
            fill_probability=0.5,
            time_since_submit=2.0,
            partial_fill_pct=0.0,
            volatility_regime="normal",
            order_age_percentile=0.5,
            price_movement_since_submit=0.001,
            spread_change_since_submit=0.0,
            volume_since_submit=50,
        )

        partial_fill = CancelTimingFeatures(
            spread_bps=20.0,
            fill_probability=0.5,
            time_since_submit=2.0,
            partial_fill_pct=0.5,  # 50% filled
            volatility_regime="normal",
            order_age_percentile=0.5,
            price_movement_since_submit=0.001,
            spread_change_since_submit=0.0,
            volume_since_submit=50,
        )

        no_decision = optimizer.get_optimal_timeout(no_fill)
        partial_decision = optimizer.get_optimal_timeout(partial_fill)

        # Partial fill should extend timeout
        assert partial_decision.optimal_wait_seconds > no_decision.optimal_wait_seconds

    def test_analyze_timing_patterns(self, optimizer, sample_features):
        """Test timing pattern analysis."""
        # Record enough data
        for i in range(25):
            optimizer.record_outcome(
                order_id=f"TEST{i:03d}",
                symbol="SPY",
                outcome=FillOutcome.FILLED,
                time_to_outcome=1.5 + i * 0.1,
                features=sample_features,
                fill_pct=1.0,
            )

        analysis = optimizer.analyze_timing_patterns()

        assert "mean_fill_time" in analysis
        assert "recommended_timeout" in analysis
        assert analysis["total_fills"] == 25

    def test_should_reprice(self, optimizer, sample_features):
        """Test reprice recommendation."""
        should, new_price = optimizer.should_reprice(
            features=sample_features,
            current_price=450.0,
            best_bid=449.8,
            best_ask=450.2,
        )

        # At mid, shouldn't need reprice
        assert should is False

    def test_global_statistics(self, optimizer, sample_features):
        """Test global statistics across symbols."""
        # Record for multiple symbols
        for symbol in ["SPY", "QQQ", "IWM"]:
            for i in range(10):
                optimizer.record_outcome(
                    order_id=f"{symbol}_{i:03d}",
                    symbol=symbol,
                    outcome=FillOutcome.FILLED,
                    time_to_outcome=2.0 + i * 0.1,
                    features=sample_features,
                    fill_pct=1.0,
                )

        stats = optimizer.get_global_statistics()

        assert stats.total_orders == 30
        assert stats.filled_count == 30


class TestTimingStatistics:
    """Tests for TimingStatistics dataclass."""

    def test_default_values(self):
        """Test default values."""
        stats = TimingStatistics()

        assert stats.total_orders == 0
        assert stats.filled_count == 0
        assert stats.optimal_cancel_time == 2.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = TimingStatistics(
            total_orders=100,
            filled_count=75,
            avg_fill_time=2.1,
            median_fill_time=1.9,
            percentile_90_fill_time=3.5,
            optimal_cancel_time=3.5,
        )

        d = stats.to_dict()

        assert d["total_orders"] == 100
        assert d["fill_rate"] == 0.75
        assert d["optimal_cancel_time"] == 3.5


class TestCancelReason:
    """Tests for CancelReason enum."""

    def test_reasons_exist(self):
        """Test all reasons exist."""
        assert CancelReason.TIMEOUT_EXCEEDED.value == "timeout_exceeded"
        assert CancelReason.LOW_FILL_PROBABILITY.value == "low_fill_probability"
        assert CancelReason.MARKET_MOVED.value == "market_moved"
        assert CancelReason.SPREAD_WIDENED.value == "spread_widened"


class TestCreateCancelOptimizer:
    """Tests for factory function."""

    def test_create_default(self):
        """Test creating with defaults."""
        optimizer = create_cancel_optimizer()

        assert isinstance(optimizer, CancelOptimizer)
        assert optimizer.base_timeout == 2.5

    def test_create_with_timeout(self):
        """Test creating with custom timeout."""
        optimizer = create_cancel_optimizer(base_timeout=3.0)

        assert optimizer.base_timeout == 3.0
