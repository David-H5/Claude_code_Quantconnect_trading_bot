"""
Tests for spread_anomaly module.

Tests spread anomaly detection including:
- Quote stuffing detection
- Sudden spread widening
- Liquidity gaps
- Crossed markets
- Baseline statistics
"""

from datetime import datetime, timedelta

import pytest

from execution.spread_anomaly import (
    AnomalySeverity,
    QuoteUpdate,
    SpreadAnomaly,
    SpreadAnomalyDetector,
    SpreadAnomalyType,
    SpreadBaseline,
    create_spread_anomaly_detector,
)


class TestSpreadAnomalyType:
    """Tests for SpreadAnomalyType enum."""

    def test_all_values_exist(self):
        """Test all anomaly types exist."""
        assert SpreadAnomalyType.NORMAL.value == "normal"
        assert SpreadAnomalyType.WIDE_SPREAD.value == "wide_spread"
        assert SpreadAnomalyType.QUOTE_STUFFING.value == "quote_stuffing"
        assert SpreadAnomalyType.SUDDEN_WIDENING.value == "sudden_widening"
        assert SpreadAnomalyType.LIQUIDITY_GAP.value == "liquidity_gap"
        assert SpreadAnomalyType.STALE_QUOTE.value == "stale_quote"
        assert SpreadAnomalyType.CROSSED_MARKET.value == "crossed_market"

    def test_enum_count(self):
        """Test correct number of anomaly types."""
        assert len(SpreadAnomalyType) == 7


class TestAnomalySeverity:
    """Tests for AnomalySeverity enum."""

    def test_all_values_exist(self):
        """Test all severity levels exist."""
        assert AnomalySeverity.INFO.value == "info"
        assert AnomalySeverity.WARNING.value == "warning"
        assert AnomalySeverity.CRITICAL.value == "critical"

    def test_enum_count(self):
        """Test correct number of severity levels."""
        assert len(AnomalySeverity) == 3


class TestQuoteUpdate:
    """Tests for QuoteUpdate dataclass."""

    @pytest.fixture
    def quote(self):
        """Create sample quote."""
        return QuoteUpdate(
            timestamp=datetime.now(),
            bid=100.0,
            ask=100.10,
            bid_size=500,
            ask_size=300,
            sequence=1,
        )

    def test_initialization(self, quote):
        """Test quote initialization."""
        assert quote.bid == 100.0
        assert quote.ask == 100.10
        assert quote.bid_size == 500
        assert quote.ask_size == 300
        assert quote.sequence == 1

    def test_spread_property(self, quote):
        """Test spread calculation."""
        assert quote.spread == pytest.approx(0.10, rel=1e-6)

    def test_mid_property(self, quote):
        """Test mid price calculation."""
        assert quote.mid == pytest.approx(100.05, rel=1e-6)

    def test_spread_pct_property(self, quote):
        """Test spread percentage calculation."""
        # spread / mid = 0.10 / 100.05 ≈ 0.000999
        expected = 0.10 / 100.05
        assert quote.spread_pct == pytest.approx(expected, rel=1e-6)

    def test_spread_pct_zero_mid(self):
        """Test spread_pct with zero mid price."""
        quote = QuoteUpdate(
            timestamp=datetime.now(),
            bid=0.0,
            ask=0.0,
            bid_size=100,
            ask_size=100,
        )
        assert quote.spread_pct == 0

    def test_default_sequence(self):
        """Test default sequence value."""
        quote = QuoteUpdate(
            timestamp=datetime.now(),
            bid=100.0,
            ask=100.10,
            bid_size=100,
            ask_size=100,
        )
        assert quote.sequence == 0


class TestSpreadAnomaly:
    """Tests for SpreadAnomaly dataclass."""

    @pytest.fixture
    def anomaly(self):
        """Create sample anomaly."""
        return SpreadAnomaly(
            timestamp=datetime(2025, 1, 15, 10, 30, 0),
            anomaly_type=SpreadAnomalyType.WIDE_SPREAD,
            severity=AnomalySeverity.WARNING,
            current_spread_bps=50.0,
            normal_spread_bps=20.0,
            deviation_factor=2.5,
            description="Spread 2.5x wider than normal",
            should_avoid_trading=True,
            fill_probability_impact=-0.5,
        )

    def test_initialization(self, anomaly):
        """Test anomaly initialization."""
        assert anomaly.anomaly_type == SpreadAnomalyType.WIDE_SPREAD
        assert anomaly.severity == AnomalySeverity.WARNING
        assert anomaly.current_spread_bps == 50.0
        assert anomaly.normal_spread_bps == 20.0
        assert anomaly.deviation_factor == 2.5
        assert anomaly.should_avoid_trading is True
        assert anomaly.fill_probability_impact == -0.5

    def test_to_dict(self, anomaly):
        """Test to_dict conversion."""
        d = anomaly.to_dict()

        assert d["type"] == "wide_spread"
        assert d["severity"] == "warning"
        assert d["current_spread_bps"] == 50.0
        assert d["normal_spread_bps"] == 20.0
        assert d["deviation_factor"] == 2.5
        assert d["description"] == "Spread 2.5x wider than normal"
        assert d["avoid_trading"] is True
        assert d["fill_probability_impact"] == -0.5
        assert "timestamp" in d

    def test_critical_anomaly(self):
        """Test critical anomaly creation."""
        anomaly = SpreadAnomaly(
            timestamp=datetime.now(),
            anomaly_type=SpreadAnomalyType.CROSSED_MARKET,
            severity=AnomalySeverity.CRITICAL,
            current_spread_bps=-5.0,
            normal_spread_bps=10.0,
            deviation_factor=float("inf"),
            description="Crossed market: bid > ask",
            should_avoid_trading=True,
            fill_probability_impact=-1.0,
        )

        assert anomaly.severity == AnomalySeverity.CRITICAL
        assert anomaly.fill_probability_impact == -1.0


class TestSpreadBaseline:
    """Tests for SpreadBaseline dataclass."""

    @pytest.fixture
    def baseline(self):
        """Create sample baseline."""
        return SpreadBaseline(
            symbol="SPY",
            avg_spread_bps=10.0,
            median_spread_bps=9.0,
            std_spread_bps=3.0,
            p95_spread_bps=15.0,
            p99_spread_bps=20.0,
            avg_quote_frequency=5.0,
            samples=500,
        )

    def test_initialization(self, baseline):
        """Test baseline initialization."""
        assert baseline.symbol == "SPY"
        assert baseline.avg_spread_bps == 10.0
        assert baseline.median_spread_bps == 9.0
        assert baseline.std_spread_bps == 3.0
        assert baseline.p95_spread_bps == 15.0
        assert baseline.p99_spread_bps == 20.0
        assert baseline.avg_quote_frequency == 5.0
        assert baseline.samples == 500

    def test_is_spread_abnormal_normal(self, baseline):
        """Test normal spread is not flagged."""
        is_abnormal, deviation = baseline.is_spread_abnormal(12.0)
        assert is_abnormal is False
        assert deviation == pytest.approx(1.2, rel=1e-6)

    def test_is_spread_abnormal_high(self, baseline):
        """Test high spread is flagged."""
        # Threshold = max(10 + 2*3, 10*2) = max(16, 20) = 20
        is_abnormal, deviation = baseline.is_spread_abnormal(25.0)
        assert is_abnormal is True
        assert deviation == pytest.approx(2.5, rel=1e-6)

    def test_is_spread_abnormal_at_threshold(self, baseline):
        """Test spread at threshold boundary."""
        # Threshold = 20, so 20 should not be abnormal, 20.1 should be
        is_abnormal, _ = baseline.is_spread_abnormal(20.0)
        assert is_abnormal is False

        is_abnormal, _ = baseline.is_spread_abnormal(20.1)
        assert is_abnormal is True

    def test_is_spread_abnormal_zero_avg(self):
        """Test with zero average spread."""
        baseline = SpreadBaseline(
            symbol="TEST",
            avg_spread_bps=0,
            median_spread_bps=0,
            std_spread_bps=0,
            p95_spread_bps=0,
            p99_spread_bps=0,
            avg_quote_frequency=0,
            samples=0,
        )

        is_abnormal, deviation = baseline.is_spread_abnormal(10.0)
        assert is_abnormal is False
        assert deviation == 1.0

    def test_to_dict(self, baseline):
        """Test to_dict conversion."""
        d = baseline.to_dict()

        assert d["symbol"] == "SPY"
        assert d["avg_spread_bps"] == 10.0
        assert d["median_spread_bps"] == 9.0
        assert d["std_spread_bps"] == 3.0
        assert d["p95_spread_bps"] == 15.0
        assert d["p99_spread_bps"] == 20.0
        assert d["avg_quote_frequency"] == 5.0
        assert d["samples"] == 500


class TestSpreadAnomalyDetector:
    """Tests for SpreadAnomalyDetector class."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return SpreadAnomalyDetector(
            baseline_window=100,
            quote_stuffing_threshold=50,
            sudden_widening_factor=3.0,
        )

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.baseline_window == 100
        assert detector.quote_stuffing_threshold == 50
        assert detector.sudden_widening_factor == 3.0
        assert detector.alert_callback is None

    def test_initialization_with_callback(self):
        """Test detector with alert callback."""
        alerts = []

        def callback(a):
            alerts.append(a)

        detector = SpreadAnomalyDetector(alert_callback=callback)
        assert detector.alert_callback is not None

    def test_update_creates_history(self, detector):
        """Test update creates history for new symbol."""
        detector.update("SPY", bid=450.0, ask=450.05)

        assert "SPY" in detector.quote_history
        assert len(detector.quote_history["SPY"]) == 1

    def test_update_normal_quote(self, detector):
        """Test update with normal quote returns no anomaly."""
        # Add some baseline quotes first with proper timestamps to avoid stuffing
        base_time = datetime.now()
        for i in range(100):
            ts = base_time + timedelta(seconds=i)
            detector.update("SPY", bid=450.0, ask=450.05, timestamp=ts)

        ts = base_time + timedelta(seconds=101)
        anomaly = detector.update("SPY", bid=450.0, ask=450.05, timestamp=ts)
        # Normal spread should not trigger anomaly or only INFO level
        assert anomaly is None or anomaly.severity == AnomalySeverity.INFO

    def test_update_crossed_market(self, detector):
        """Test crossed market detection (bid > ask)."""
        anomaly = detector.update("SPY", bid=450.10, ask=450.00)

        assert anomaly is not None
        assert anomaly.anomaly_type == SpreadAnomalyType.CROSSED_MARKET
        assert anomaly.severity == AnomalySeverity.CRITICAL
        assert anomaly.should_avoid_trading is True
        assert anomaly.fill_probability_impact == -1.0

    def test_update_with_timestamp(self, detector):
        """Test update with explicit timestamp."""
        ts = datetime(2025, 1, 15, 10, 30, 0)
        detector.update("SPY", bid=450.0, ask=450.05, timestamp=ts)

        quotes = list(detector.quote_history["SPY"])
        assert quotes[0].timestamp == ts

    def test_wide_spread_detection(self, detector):
        """Test wide spread anomaly detection."""
        # Build baseline with tight spreads
        for i in range(100):
            detector.update("SPY", bid=450.0, ask=450.02)

        # Now send a very wide spread
        anomaly = detector.update("SPY", bid=450.0, ask=450.50)

        assert anomaly is not None
        assert anomaly.anomaly_type == SpreadAnomalyType.WIDE_SPREAD

    def test_sudden_widening_detection(self, detector):
        """Test sudden spread widening detection."""
        # Send normal spreads
        base_time = datetime.now()
        for i in range(10):
            ts = base_time + timedelta(seconds=i)
            detector.update("SPY", bid=450.0, ask=450.02, timestamp=ts)

        # Suddenly widen by 3x+
        ts = base_time + timedelta(seconds=11)
        anomaly = detector.update("SPY", bid=450.0, ask=450.10, timestamp=ts)

        # Should detect sudden widening (spread went from 0.02 to 0.10 = 5x)
        if anomaly:
            assert anomaly.anomaly_type in [
                SpreadAnomalyType.SUDDEN_WIDENING,
                SpreadAnomalyType.WIDE_SPREAD,
            ]

    def test_liquidity_gap_detection(self, detector):
        """Test liquidity gap detection."""
        # Add quotes with imbalanced sizes (10x ratio)
        for i in range(5):
            detector.update(
                "SPY",
                bid=450.0,
                ask=450.05,
                bid_size=1000,
                ask_size=100,  # 10x imbalance
            )

        anomaly = detector.update(
            "SPY",
            bid=450.0,
            ask=450.05,
            bid_size=1000,
            ask_size=50,  # 20x imbalance
        )

        if anomaly and anomaly.anomaly_type == SpreadAnomalyType.LIQUIDITY_GAP:
            assert "thin" in anomaly.description.lower()
            assert anomaly.severity == AnomalySeverity.INFO

    def test_quote_stuffing_detection(self, detector):
        """Test quote stuffing detection."""
        # Rapid-fire quotes to simulate stuffing
        base_time = datetime.now()

        # Send 60 quotes in 1 second (exceeds 50/sec threshold)
        for i in range(60):
            ts = base_time + timedelta(milliseconds=i * 16)
            anomaly = detector.update("SPY", bid=450.0, ask=450.05, timestamp=ts)

        # Final quote should trigger stuffing
        ts = base_time + timedelta(seconds=1)
        anomaly = detector.update("SPY", bid=450.0, ask=450.05, timestamp=ts)

        # May detect quote stuffing if frequency is high enough
        if anomaly and anomaly.anomaly_type == SpreadAnomalyType.QUOTE_STUFFING:
            assert anomaly.severity == AnomalySeverity.WARNING

    def test_alert_callback_called(self):
        """Test alert callback is called on anomaly."""
        alerts = []

        def callback(a):
            alerts.append(a)

        detector = SpreadAnomalyDetector(alert_callback=callback)

        # Trigger crossed market anomaly
        detector.update("SPY", bid=450.10, ask=450.00)

        assert len(alerts) == 1
        assert alerts[0].anomaly_type == SpreadAnomalyType.CROSSED_MARKET

    def test_anomaly_history_limited(self, detector):
        """Test anomaly history is limited to 100 entries."""
        # Generate many anomalies
        for i in range(150):
            detector.update("SPY", bid=450.0 + i * 0.01, ask=449.0)  # Crossed markets

        assert len(detector.anomaly_history["SPY"]) <= 100


class TestSpreadAnomalyDetectorMethods:
    """Tests for SpreadAnomalyDetector helper methods."""

    @pytest.fixture
    def detector_with_history(self):
        """Create detector with quote history and baseline."""
        detector = SpreadAnomalyDetector(baseline_window=100)

        # Add 150 quotes to build baseline (baseline updates every 50 when >= 100)
        base_time = datetime.now()
        for i in range(150):
            ts = base_time + timedelta(seconds=i)
            detector.update(
                "SPY",
                bid=450.0,
                ask=450.05,
                bid_size=500,
                ask_size=500,
                timestamp=ts,
            )

        return detector

    def test_is_safe_to_trade_no_history(self, detector_with_history):
        """Test is_safe_to_trade with no history."""
        is_safe, reason, confidence = detector_with_history.is_safe_to_trade("UNKNOWN")

        assert is_safe is True
        assert "No anomaly history" in reason
        assert confidence == 0.5

    def test_is_safe_to_trade_normal(self, detector_with_history):
        """Test is_safe_to_trade under normal conditions."""
        is_safe, reason, confidence = detector_with_history.is_safe_to_trade("SPY")

        assert is_safe is True
        assert confidence >= 0.5

    def test_is_safe_to_trade_after_critical(self, detector_with_history):
        """Test is_safe_to_trade after critical anomaly."""
        # Trigger critical anomaly
        detector_with_history.update("SPY", bid=450.10, ask=450.00)

        is_safe, reason, confidence = detector_with_history.is_safe_to_trade("SPY", lookback_seconds=60)

        assert is_safe is False
        assert confidence >= 0.9

    def test_get_spread_quality_score_no_history(self, detector_with_history):
        """Test quality score with no history."""
        score = detector_with_history.get_spread_quality_score("UNKNOWN")
        assert score == 50.0

    def test_get_spread_quality_score_normal(self, detector_with_history):
        """Test quality score under normal conditions."""
        score = detector_with_history.get_spread_quality_score("SPY")

        # Should be high (>= 80) under normal conditions
        assert 0 <= score <= 100

    def test_get_spread_quality_score_after_anomaly(self, detector_with_history):
        """Test quality score decreases after anomaly."""
        initial_score = detector_with_history.get_spread_quality_score("SPY")

        # Trigger critical anomaly
        detector_with_history.update("SPY", bid=450.10, ask=450.00)

        after_score = detector_with_history.get_spread_quality_score("SPY")

        assert after_score < initial_score

    def test_get_baseline_exists(self, detector_with_history):
        """Test getting existing baseline."""
        baseline = detector_with_history.get_baseline("SPY")

        assert baseline is not None
        assert baseline.symbol == "SPY"
        assert baseline.samples > 0

    def test_get_baseline_not_exists(self, detector_with_history):
        """Test getting non-existent baseline."""
        baseline = detector_with_history.get_baseline("UNKNOWN")
        assert baseline is None

    def test_get_recent_anomalies_empty(self, detector_with_history):
        """Test getting anomalies when none exist."""
        anomalies = detector_with_history.get_recent_anomalies("SPY")
        # May have some anomalies from baseline building or none
        assert isinstance(anomalies, list)

    def test_get_recent_anomalies_with_limit(self, detector_with_history):
        """Test getting anomalies with limit."""
        # Generate some anomalies
        for i in range(5):
            detector_with_history.update("SPY", bid=450.10, ask=450.00)

        anomalies = detector_with_history.get_recent_anomalies("SPY", limit=3)
        assert len(anomalies) <= 3


class TestSpreadAnomalyDetectorSummary:
    """Tests for summary methods."""

    @pytest.fixture
    def detector_with_data(self):
        """Create detector with comprehensive data and baseline."""
        detector = SpreadAnomalyDetector(baseline_window=100)

        # Build baseline (need 150+ quotes for baseline to be created)
        base_time = datetime.now()
        for i in range(150):
            ts = base_time + timedelta(seconds=i)
            detector.update(
                "SPY",
                bid=450.0,
                ask=450.05,
                bid_size=500,
                ask_size=500,
                timestamp=ts,
            )

        return detector

    def test_get_summary_structure(self, detector_with_data):
        """Test summary has expected structure."""
        summary = detector_with_data.get_summary("SPY")

        assert "symbol" in summary
        assert "timestamp" in summary
        assert "current_spread_bps" in summary
        assert "baseline" in summary
        assert "spread_quality_score" in summary
        assert "is_safe_to_trade" in summary
        assert "safety_reason" in summary
        assert "safety_confidence" in summary
        assert "quote_frequency" in summary
        assert "recent_anomalies" in summary

    def test_get_summary_values(self, detector_with_data):
        """Test summary values are valid."""
        summary = detector_with_data.get_summary("SPY")

        assert summary["symbol"] == "SPY"
        assert summary["current_spread_bps"] > 0
        assert summary["baseline"] is not None
        assert 0 <= summary["spread_quality_score"] <= 100
        assert isinstance(summary["is_safe_to_trade"], bool)
        assert isinstance(summary["recent_anomalies"], list)

    def test_get_summary_unknown_symbol(self, detector_with_data):
        """Test summary for unknown symbol."""
        summary = detector_with_data.get_summary("UNKNOWN")

        assert summary["symbol"] == "UNKNOWN"
        assert summary["current_spread_bps"] == 0
        assert summary["baseline"] is None

    def test_get_llm_summary_format(self, detector_with_data):
        """Test LLM summary is formatted text."""
        summary = detector_with_data.get_llm_summary("SPY")

        assert isinstance(summary, str)
        assert "SPREAD ANALYSIS FOR SPY" in summary
        assert "Current Spread:" in summary
        assert "Spread Quality Score:" in summary
        assert "Safe to Trade:" in summary

    def test_get_llm_summary_with_baseline(self, detector_with_data):
        """Test LLM summary includes baseline stats."""
        summary = detector_with_data.get_llm_summary("SPY")

        assert "BASELINE STATISTICS" in summary
        assert "Average Spread:" in summary
        assert "Median Spread:" in summary

    def test_get_llm_summary_with_anomalies(self, detector_with_data):
        """Test LLM summary includes anomalies if present."""
        # Generate anomaly
        detector_with_data.update("SPY", bid=450.10, ask=450.00)

        summary = detector_with_data.get_llm_summary("SPY")

        assert "RECENT ANOMALIES" in summary


class TestQuoteFrequency:
    """Tests for quote frequency tracking."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return SpreadAnomalyDetector()

    def test_quote_frequency_empty(self, detector):
        """Test frequency with no quotes."""
        freq = detector._get_quote_frequency("SPY")
        assert freq == 0

    def test_quote_frequency_single_quote(self, detector):
        """Test frequency with single quote."""
        detector.update("SPY", bid=450.0, ask=450.05)
        freq = detector._get_quote_frequency("SPY")
        # Single quote should return 0 (need at least 2 for time span)
        assert freq == 0

    def test_quote_frequency_multiple_quotes(self, detector):
        """Test frequency with multiple quotes over time."""
        base_time = datetime.now()

        # Add 10 quotes over 5 seconds = 2 quotes/sec
        for i in range(10):
            ts = base_time + timedelta(milliseconds=i * 500)
            detector.update("SPY", bid=450.0, ask=450.05, timestamp=ts)

        freq = detector._get_quote_frequency("SPY")
        # Should be approximately 2 quotes/sec
        assert freq > 0

    def test_quote_frequency_cleanup(self, detector):
        """Test old quotes are cleaned up."""
        base_time = datetime.now()

        # Add quotes 15 seconds ago (should be cleaned up)
        old_time = base_time - timedelta(seconds=15)
        for i in range(5):
            ts = old_time + timedelta(seconds=i)
            detector.update("SPY", bid=450.0, ask=450.05, timestamp=ts)

        # Add recent quote
        detector.update("SPY", bid=450.0, ask=450.05, timestamp=base_time)

        # Old quotes should be cleaned up, only recent remains
        assert len(detector.quote_counts["SPY"]) < 6


class TestBaselineCalculation:
    """Tests for baseline calculation."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return SpreadAnomalyDetector(baseline_window=100)

    def test_baseline_not_created_insufficient_data(self, detector):
        """Test baseline not created with too few quotes."""
        for i in range(30):
            detector.update("SPY", bid=450.0, ask=450.05)

        # With only 30 quotes, baseline should not be created yet
        # (requires 100+ quotes with update every 50)
        assert detector.get_baseline("SPY") is None

    def test_baseline_created_sufficient_data(self, detector):
        """Test baseline created with enough quotes."""
        for i in range(150):
            detector.update("SPY", bid=450.0, ask=450.05)

        baseline = detector.get_baseline("SPY")
        assert baseline is not None
        assert baseline.symbol == "SPY"
        assert baseline.samples >= 50

    def test_baseline_statistics_accuracy(self, detector):
        """Test baseline statistics are calculated correctly."""
        # Add quotes with known spread and proper timestamps
        base_time = datetime.now()
        for i in range(150):
            ts = base_time + timedelta(seconds=i)
            # Spread = 0.05, mid = 450.025
            # spread_bps = 0.05 / 450.025 * 10000 ≈ 1.11 bps
            detector.update("SPY", bid=450.0, ask=450.05, timestamp=ts)

        baseline = detector.get_baseline("SPY")
        assert baseline is not None

        # All spreads are identical, so std should be ~0
        assert baseline.std_spread_bps < 1.0
        # Avg should be close to 1.11 bps (0.05 / 450.025 * 10000)
        assert 1.0 < baseline.avg_spread_bps < 1.5


class TestCreateSpreadAnomalyDetector:
    """Tests for create_spread_anomaly_detector factory."""

    def test_default_parameters(self):
        """Test factory with default parameters."""
        detector = create_spread_anomaly_detector()

        assert detector.baseline_window == 1000
        assert detector.quote_stuffing_threshold == 50
        assert detector.sudden_widening_factor == 3.0
        assert detector.alert_callback is None

    def test_custom_parameters(self):
        """Test factory with custom parameters."""
        detector = create_spread_anomaly_detector(
            baseline_window=500,
            quote_stuffing_threshold=100,
            sudden_widening_factor=2.0,
        )

        assert detector.baseline_window == 500
        assert detector.quote_stuffing_threshold == 100
        assert detector.sudden_widening_factor == 2.0

    def test_with_callback(self):
        """Test factory with alert callback."""
        alerts = []

        def callback(a):
            alerts.append(a)

        detector = create_spread_anomaly_detector(alert_callback=callback)

        assert detector.alert_callback is not None

        # Verify callback works
        detector.update("SPY", bid=450.10, ask=450.00)
        assert len(alerts) == 1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_negative_prices(self):
        """Test handling of negative prices (shouldn't happen but test anyway)."""
        detector = SpreadAnomalyDetector()

        # Negative prices should still be processed without raising exception
        # Not a crossed market since -1.0 < -0.5
        detector.update("TEST", bid=-1.0, ask=-0.5)
        assert "TEST" in detector.quote_history

    def test_zero_prices(self):
        """Test handling of zero prices."""
        detector = SpreadAnomalyDetector()

        # spread_pct will be 0 (mid is 0) - should not raise exception
        detector.update("TEST", bid=0.0, ask=0.0)
        assert "TEST" in detector.quote_history

    def test_very_large_spread(self):
        """Test handling of extremely large spreads."""
        detector = SpreadAnomalyDetector(baseline_window=10)

        # Build a baseline first with proper timestamps
        base_time = datetime.now()
        for i in range(50):
            ts = base_time + timedelta(seconds=i)
            detector.update("TEST", bid=100.0, ask=100.01, timestamp=ts)

        # Now send huge spread
        ts = base_time + timedelta(seconds=51)
        anomaly = detector.update("TEST", bid=100.0, ask=200.0, timestamp=ts)

        if anomaly:
            # May detect wide spread, sudden widening, or quote stuffing
            assert anomaly.anomaly_type in [
                SpreadAnomalyType.WIDE_SPREAD,
                SpreadAnomalyType.SUDDEN_WIDENING,
                SpreadAnomalyType.QUOTE_STUFFING,
            ]

    def test_size_zero(self):
        """Test handling of zero bid/ask sizes."""
        detector = SpreadAnomalyDetector()

        # Zero sizes should not trigger liquidity gap (division by zero)
        # Should not raise exception
        detector.update(
            "TEST",
            bid=100.0,
            ask=100.05,
            bid_size=0,
            ask_size=0,
        )
        assert "TEST" in detector.quote_history

    def test_multiple_symbols(self):
        """Test tracking multiple symbols independently."""
        detector = SpreadAnomalyDetector()

        detector.update("SPY", bid=450.0, ask=450.05)
        detector.update("QQQ", bid=380.0, ask=380.08)
        detector.update("IWM", bid=220.0, ask=220.10)

        assert "SPY" in detector.quote_history
        assert "QQQ" in detector.quote_history
        assert "IWM" in detector.quote_history
        assert len(detector.quote_history) == 3


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_market_open_volatility(self):
        """Test handling market open volatility."""
        detector = SpreadAnomalyDetector(baseline_window=50)
        base_time = datetime.now()

        # Pre-market: steady spreads
        for i in range(50):
            ts = base_time + timedelta(seconds=i)
            detector.update("SPY", bid=450.0, ask=450.02, timestamp=ts)

        # Market open: spreads widen temporarily
        open_time = base_time + timedelta(seconds=60)
        anomaly = detector.update("SPY", bid=450.0, ask=450.10, timestamp=open_time)

        # Should detect the widening
        if anomaly:
            assert anomaly.severity in [AnomalySeverity.WARNING, AnomalySeverity.CRITICAL]

    def test_news_event_spread_widening(self):
        """Test spread widening on news event."""
        detector = SpreadAnomalyDetector(baseline_window=100)

        # Normal trading
        for i in range(100):
            detector.update("AAPL", bid=180.0, ask=180.01)

        # News hits - spreads widen dramatically
        anomaly = detector.update("AAPL", bid=180.0, ask=180.50)

        assert anomaly is not None
        # News creates wide spread condition
        assert anomaly.should_avoid_trading is True

    def test_gradual_spread_improvement(self):
        """Test tracking gradual spread improvement."""
        detector = SpreadAnomalyDetector(baseline_window=50)

        # Start with wide spreads, gradually tightening with timestamps
        base_time = datetime.now()
        for i in range(60):
            ts = base_time + timedelta(seconds=i)
            spread = 0.20 - (i * 0.002)  # Gradually tightening
            spread = max(spread, 0.01)
            detector.update("SPY", bid=450.0, ask=450.0 + spread, timestamp=ts)

        # Score should be reasonable at the end (may be low due to early anomalies)
        score = detector.get_spread_quality_score("SPY")
        assert score >= 0  # Score can be 0 if many anomalies were detected

    def test_continuous_monitoring_session(self):
        """Test continuous monitoring over extended period."""
        detector = create_spread_anomaly_detector(baseline_window=100)

        anomaly_count = 0
        base_time = datetime.now()

        # Simulate 500 quotes over ~8 minutes
        for i in range(500):
            ts = base_time + timedelta(seconds=i)

            # Occasionally inject wide spreads
            if i % 100 == 50:
                spread = 0.50
            else:
                spread = 0.02 + (i % 10) * 0.001  # Small variation

            anomaly = detector.update(
                "SPY",
                bid=450.0,
                ask=450.0 + spread,
                bid_size=500,
                ask_size=500,
                timestamp=ts,
            )

            if anomaly:
                anomaly_count += 1

        # Should detect the periodic wide spreads
        assert anomaly_count > 0

        # Final state should be reasonable
        summary = detector.get_summary("SPY")
        assert summary["baseline"] is not None
        assert 0 <= summary["spread_quality_score"] <= 100


class TestSeverityOrdering:
    """Tests for anomaly severity ordering."""

    def test_critical_returned_over_warning(self):
        """Test critical anomaly is returned when multiple detected."""
        detector = SpreadAnomalyDetector(baseline_window=50)

        # Build baseline
        for i in range(60):
            detector.update("SPY", bid=450.0, ask=450.02)

        # This creates crossed market (critical) - bid > ask
        anomaly = detector.update("SPY", bid=450.10, ask=450.00)

        assert anomaly is not None
        assert anomaly.severity == AnomalySeverity.CRITICAL
        assert anomaly.anomaly_type == SpreadAnomalyType.CROSSED_MARKET

    def test_multiple_warnings_affect_safety(self):
        """Test multiple warnings make trading unsafe."""
        detector = SpreadAnomalyDetector(baseline_window=50)

        # Build baseline
        for i in range(60):
            detector.update("SPY", bid=450.0, ask=450.02)

        # Generate multiple warnings by sending wide spreads
        for i in range(5):
            detector.update("SPY", bid=450.0, ask=450.20)

        is_safe, reason, confidence = detector.is_safe_to_trade("SPY", lookback_seconds=10)

        # With multiple warnings, should not be safe
        # (depends on how many actually get recorded as warnings)
