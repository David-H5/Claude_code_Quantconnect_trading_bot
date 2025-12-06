"""
Tests for Real-Time Anomaly Detector (UPGRADE-010 Sprint 1)

Tests anomaly detection including flash crash, volume spike,
volatility spike, price gap, and Isolation Forest detection.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from models.anomaly_detector import (
    AnomalyDetector,
    AnomalyDetectorConfig,
    AnomalyResult,
    AnomalySeverity,
    AnomalyType,
    MarketDataPoint,
    create_anomaly_detector,
)


class TestMarketDataPoint:
    """Tests for MarketDataPoint class."""

    @pytest.mark.unit
    def test_data_point_creation(self):
        """Test creating a market data point."""
        dp = MarketDataPoint(
            timestamp=datetime.utcnow(),
            price=450.0,
            volume=1000000,
            high=452.0,
            low=448.0,
        )

        assert dp.price == 450.0
        assert dp.volume == 1000000
        assert dp.high == 452.0
        assert dp.low == 448.0

    @pytest.mark.unit
    def test_true_range(self):
        """Test true range calculation."""
        dp = MarketDataPoint(
            timestamp=datetime.utcnow(),
            price=450.0,
            volume=1000000,
            high=455.0,
            low=445.0,
        )

        assert dp.true_range == 10.0

    @pytest.mark.unit
    def test_true_range_no_high_low(self):
        """Test true range with no high/low data."""
        dp = MarketDataPoint(
            timestamp=datetime.utcnow(),
            price=450.0,
            volume=1000000,
        )

        assert dp.true_range == 0.0


class TestAnomalyDetectorConfig:
    """Tests for AnomalyDetectorConfig."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default configuration values."""
        config = AnomalyDetectorConfig()

        assert config.contamination == 0.01
        assert config.flash_crash_threshold_pct == 0.05
        assert config.volume_spike_threshold == 3.0
        assert config.price_gap_threshold_pct == 0.03

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom configuration."""
        config = AnomalyDetectorConfig(
            contamination=0.02,
            flash_crash_threshold_pct=0.10,
            volume_spike_threshold=5.0,
        )

        assert config.contamination == 0.02
        assert config.flash_crash_threshold_pct == 0.10
        assert config.volume_spike_threshold == 5.0


class TestAnomalyResult:
    """Tests for AnomalyResult class."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating an anomaly result."""
        result = AnomalyResult(
            is_anomaly=True,
            anomaly_type=AnomalyType.FLASH_CRASH,
            severity=AnomalySeverity.CRITICAL,
            score=-0.8,
            threshold=-0.5,
            timestamp=datetime.utcnow(),
            description="Flash crash detected",
            recommended_action="HALT_TRADING",
        )

        assert result.is_anomaly is True
        assert result.anomaly_type == AnomalyType.FLASH_CRASH
        assert result.severity == AnomalySeverity.CRITICAL

    @pytest.mark.unit
    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = AnomalyResult(
            is_anomaly=True,
            anomaly_type=AnomalyType.VOLUME_SPIKE,
            severity=AnomalySeverity.HIGH,
            score=-0.6,
            threshold=-0.5,
            timestamp=datetime.utcnow(),
            feature_values={"volume_ratio": 4.5},
        )

        data = result.to_dict()

        assert data["is_anomaly"] is True
        assert data["anomaly_type"] == "volume_spike"
        assert data["severity"] == "high"
        assert "volume_ratio" in data["feature_values"]


class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""

    @pytest.fixture
    def detector(self) -> AnomalyDetector:
        """Create a detector for testing."""
        return AnomalyDetector()

    @pytest.fixture
    def detector_with_data(self) -> AnomalyDetector:
        """Create detector with sample data."""
        detector = AnomalyDetector()

        # Add normal data points
        base_time = datetime.utcnow() - timedelta(hours=1)
        for i in range(60):
            dp = MarketDataPoint(
                timestamp=base_time + timedelta(minutes=i),
                price=450.0 + np.random.normal(0, 0.5),
                volume=1000000 + np.random.normal(0, 50000),
                high=451.0,
                low=449.0,
            )
            detector.add_data(dp)

        return detector

    @pytest.mark.unit
    def test_detector_creation(self, detector):
        """Test detector creation."""
        assert detector is not None
        assert len(detector._data_points) == 0
        assert len(detector._anomaly_history) == 0

    @pytest.mark.unit
    def test_add_data(self, detector):
        """Test adding data points."""
        dp = MarketDataPoint(
            timestamp=datetime.utcnow(),
            price=450.0,
            volume=1000000,
        )
        detector.add_data(dp)

        assert len(detector._data_points) == 1

    @pytest.mark.unit
    def test_detect_insufficient_data(self, detector):
        """Test detection with insufficient data."""
        # Add just a few points
        for _ in range(5):
            detector.add_data(
                MarketDataPoint(
                    timestamp=datetime.utcnow(),
                    price=450.0,
                    volume=1000000,
                )
            )

        result = detector.detect()

        assert result.is_anomaly is False
        assert "Insufficient" in result.description

    @pytest.mark.unit
    def test_detect_flash_crash(self, detector_with_data):
        """Test flash crash detection."""
        # Flash crash detection window is 60 seconds, so we need at least
        # 2 data points within that window. Add a "pre-crash" point first.
        pre_crash_time = datetime.utcnow() - timedelta(seconds=30)
        detector_with_data.add_data(
            MarketDataPoint(
                timestamp=pre_crash_time,
                price=450.0,
                volume=1000000,
                high=451.0,
                low=449.0,
            )
        )

        # Now add the crash point
        crash_time = datetime.utcnow()
        crash_price = 450.0 * (1 - 0.08)  # 8% drop

        detector_with_data.add_data(
            MarketDataPoint(
                timestamp=crash_time,
                price=crash_price,
                volume=2000000,
                high=450.0,
                low=crash_price,
            )
        )

        result = detector_with_data.detect()

        # Should detect the crash
        assert result.is_anomaly is True
        assert result.anomaly_type == AnomalyType.FLASH_CRASH

    @pytest.mark.unit
    def test_detect_volume_spike(self):
        """Test volume spike detection."""
        detector = AnomalyDetector()

        # Add normal volume data
        base_time = datetime.utcnow() - timedelta(hours=1)
        for i in range(50):
            detector.add_data(
                MarketDataPoint(
                    timestamp=base_time + timedelta(minutes=i),
                    price=450.0,
                    volume=1000000,  # Normal volume
                    high=451.0,
                    low=449.0,
                )
            )

        # Add volume spike
        detector.add_data(
            MarketDataPoint(
                timestamp=datetime.utcnow(),
                price=450.0,
                volume=5000000,  # 5x normal volume
                high=451.0,
                low=449.0,
            )
        )

        result = detector.detect()

        # Should detect volume spike
        assert result.is_anomaly is True
        assert result.anomaly_type == AnomalyType.VOLUME_SPIKE

    @pytest.mark.unit
    def test_detect_volatility_spike(self):
        """Test volatility spike detection."""
        detector = AnomalyDetector()

        # Add low volatility data
        base_time = datetime.utcnow() - timedelta(hours=1)
        for i in range(50):
            detector.add_data(
                MarketDataPoint(
                    timestamp=base_time + timedelta(minutes=i),
                    price=450.0,
                    volume=1000000,
                    high=451.0,  # 0.4% range
                    low=449.0,
                )
            )

        # Add high volatility point
        detector.add_data(
            MarketDataPoint(
                timestamp=datetime.utcnow(),
                price=450.0,
                volume=1000000,
                high=460.0,  # 4.4% range - much higher
                low=440.0,
            )
        )

        result = detector.detect()

        # Should detect volatility spike
        assert result.is_anomaly is True
        assert result.anomaly_type == AnomalyType.VOLATILITY_SPIKE

    @pytest.mark.unit
    def test_detect_price_gap(self):
        """Test price gap detection."""
        detector = AnomalyDetector()

        # Add initial data
        base_time = datetime.utcnow() - timedelta(hours=1)
        for i in range(50):
            detector.add_data(
                MarketDataPoint(
                    timestamp=base_time + timedelta(minutes=i),
                    price=450.0,
                    volume=1000000,
                )
            )

        # Add gap (5% gap)
        detector.add_data(
            MarketDataPoint(
                timestamp=datetime.utcnow(),
                price=450.0 * 1.05,  # 5% gap up
                volume=1000000,
            )
        )

        result = detector.detect()

        assert result.is_anomaly is True
        assert result.anomaly_type == AnomalyType.PRICE_GAP

    @pytest.mark.unit
    def test_no_anomaly_normal_data(self):
        """Test no anomaly with normal data."""
        detector = AnomalyDetector()

        # Add consistent normal data
        base_time = datetime.utcnow() - timedelta(hours=1)
        for i in range(60):
            detector.add_data(
                MarketDataPoint(
                    timestamp=base_time + timedelta(minutes=i),
                    price=450.0 + np.random.normal(0, 0.2),  # Very small variation
                    volume=1000000 + np.random.normal(0, 10000),
                    high=450.5,
                    low=449.5,
                )
            )

        result = detector.detect()

        # Should not detect anomaly
        assert result.is_anomaly is False

    @pytest.mark.unit
    def test_anomaly_callback(self):
        """Test anomaly callback is triggered."""
        callback_results = []

        def on_anomaly(result: AnomalyResult):
            callback_results.append(result)

        detector = AnomalyDetector(on_anomaly_callback=on_anomaly)

        # Add normal data
        base_time = datetime.utcnow() - timedelta(hours=1)
        for i in range(50):
            detector.add_data(
                MarketDataPoint(
                    timestamp=base_time + timedelta(minutes=i),
                    price=450.0,
                    volume=1000000,
                )
            )

        # Add crash
        detector.add_data(
            MarketDataPoint(
                timestamp=datetime.utcnow(),
                price=400.0,  # Big drop
                volume=1000000,
            )
        )

        detector.detect()

        assert len(callback_results) > 0

    @pytest.mark.unit
    def test_get_anomaly_history(self, detector_with_data):
        """Test getting anomaly history."""
        # Trigger an anomaly
        detector_with_data.add_data(
            MarketDataPoint(
                timestamp=datetime.utcnow(),
                price=400.0,  # Big drop
                volume=1000000,
            )
        )
        detector_with_data.detect()

        history = detector_with_data.get_anomaly_history()

        assert len(history) > 0

    @pytest.mark.unit
    def test_get_statistics(self, detector_with_data):
        """Test getting detector statistics."""
        stats = detector_with_data.get_statistics()

        assert "data_points" in stats
        assert stats["data_points"] == 60
        assert "anomalies_detected" in stats
        assert "is_fitted" in stats

    @pytest.mark.unit
    def test_reset(self, detector_with_data):
        """Test detector reset."""
        detector_with_data.reset()

        assert len(detector_with_data._data_points) == 0
        assert len(detector_with_data._returns) == 0
        assert detector_with_data._is_fitted is False

    @pytest.mark.unit
    def test_severity_calculation(self, detector):
        """Test severity calculation from score."""
        # Critical severity
        severity = detector._calculate_severity_from_score(-0.8)
        assert severity == AnomalySeverity.CRITICAL

        # High severity
        severity = detector._calculate_severity_from_score(-0.6)
        assert severity == AnomalySeverity.HIGH

        # Medium severity
        severity = detector._calculate_severity_from_score(-0.4)
        assert severity == AnomalySeverity.MEDIUM

        # Low severity
        severity = detector._calculate_severity_from_score(-0.1)
        assert severity == AnomalySeverity.LOW

    @pytest.mark.unit
    def test_false_positive_rate(self, detector):
        """Test false positive rate calculation."""
        # Initially 0
        assert detector.get_false_positive_rate() == 0.0


class TestCreateAnomalyDetector:
    """Tests for factory function."""

    @pytest.mark.unit
    def test_create_with_defaults(self):
        """Test factory with defaults."""
        detector = create_anomaly_detector()

        assert detector is not None
        assert detector.config.contamination == 0.01

    @pytest.mark.unit
    def test_create_with_custom_config(self):
        """Test factory with custom config."""
        config = AnomalyDetectorConfig(contamination=0.05)
        detector = create_anomaly_detector(config=config)

        assert detector.config.contamination == 0.05

    @pytest.mark.unit
    def test_create_with_callback(self):
        """Test factory with callback."""
        callback_called = []

        def on_anomaly(result):
            callback_called.append(True)

        detector = create_anomaly_detector(on_anomaly_callback=on_anomaly)

        assert detector.on_anomaly_callback is not None


class TestAnomalyTypeEnum:
    """Tests for AnomalyType enum."""

    @pytest.mark.unit
    def test_all_types_exist(self):
        """Test all expected anomaly types exist."""
        assert AnomalyType.FLASH_CRASH is not None
        assert AnomalyType.VOLUME_SPIKE is not None
        assert AnomalyType.VOLATILITY_SPIKE is not None
        assert AnomalyType.PRICE_GAP is not None
        assert AnomalyType.LIQUIDITY_DROP is not None
        assert AnomalyType.REGIME_SHIFT is not None

    @pytest.mark.unit
    def test_type_values(self):
        """Test anomaly type values."""
        assert AnomalyType.FLASH_CRASH.value == "flash_crash"
        assert AnomalyType.VOLUME_SPIKE.value == "volume_spike"


class TestAnomalySeverityEnum:
    """Tests for AnomalySeverity enum."""

    @pytest.mark.unit
    def test_all_severities_exist(self):
        """Test all severity levels exist."""
        assert AnomalySeverity.LOW is not None
        assert AnomalySeverity.MEDIUM is not None
        assert AnomalySeverity.HIGH is not None
        assert AnomalySeverity.CRITICAL is not None

    @pytest.mark.unit
    def test_severity_values(self):
        """Test severity values."""
        assert AnomalySeverity.CRITICAL.value == "critical"
        assert AnomalySeverity.LOW.value == "low"


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration."""

    @pytest.mark.unit
    def test_trip_reason_exists(self):
        """Test ANOMALY_DETECTED trip reason exists."""
        from models.circuit_breaker import TripReason

        assert TripReason.ANOMALY_DETECTED is not None
        assert TripReason.ANOMALY_DETECTED.value == "anomaly_detected"
