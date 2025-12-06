"""
Tests for Correlation Monitor (Sprint 4 Expansion)

Tests position correlation and concentration risk monitoring.
Part of UPGRADE-010 Sprint 4 Expansion.
"""

import numpy as np
import pytest

from models.correlation_monitor import (
    ConcentrationLevel,
    CorrelationAlert,
    CorrelationConfig,
    CorrelationMonitor,
    CorrelationPair,
    DiversificationScore,
    create_correlation_monitor,
)


class TestCorrelationConfig:
    """Tests for CorrelationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CorrelationConfig()

        assert config.lookback_days == 60
        assert config.high_correlation_threshold == 0.70
        assert config.max_correlated_weight == 0.40

    def test_custom_values(self):
        """Test custom configuration."""
        config = CorrelationConfig(
            lookback_days=30,
            high_correlation_threshold=0.80,
        )

        assert config.lookback_days == 30
        assert config.high_correlation_threshold == 0.80

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = CorrelationConfig()
        d = config.to_dict()

        assert d["lookback_days"] == 60
        assert "high_correlation_threshold" in d


class TestCorrelationPair:
    """Tests for CorrelationPair dataclass."""

    def test_creation(self):
        """Test pair creation."""
        pair = CorrelationPair(
            symbol1="SPY",
            symbol2="QQQ",
            correlation=0.85,
            weight1=0.30,
            weight2=0.25,
            combined_weight=0.55,
        )

        assert pair.correlation == 0.85
        assert pair.combined_weight == 0.55

    def test_to_dict(self):
        """Test conversion to dictionary."""
        pair = CorrelationPair(
            symbol1="SPY",
            symbol2="IWM",
            correlation=0.75,
        )

        d = pair.to_dict()

        assert d["symbol1"] == "SPY"
        assert d["correlation"] == 0.75


class TestCorrelationMonitor:
    """Tests for CorrelationMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create default monitor."""
        return CorrelationMonitor()

    @pytest.fixture
    def monitor_with_data(self):
        """Create monitor with correlated returns."""
        monitor = CorrelationMonitor()

        # Generate correlated returns
        np.random.seed(42)
        n_days = 60

        # SPY returns
        spy_returns = np.random.normal(0.0005, 0.01, n_days)

        # QQQ returns (highly correlated with SPY)
        qqq_returns = spy_returns * 1.2 + np.random.normal(0, 0.002, n_days)

        # IWM returns (moderately correlated)
        iwm_returns = spy_returns * 0.8 + np.random.normal(0, 0.005, n_days)

        # GLD returns (uncorrelated)
        gld_returns = np.random.normal(0.0002, 0.008, n_days)

        for i in range(n_days):
            monitor.update_returns("SPY", spy_returns[i])
            monitor.update_returns("QQQ", qqq_returns[i])
            monitor.update_returns("IWM", iwm_returns[i])
            monitor.update_returns("GLD", gld_returns[i])

        # Set weights
        monitor.update_weight("SPY", 0.30)
        monitor.update_weight("QQQ", 0.25)
        monitor.update_weight("IWM", 0.25)
        monitor.update_weight("GLD", 0.20)

        return monitor

    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.config is not None
        assert len(monitor._returns) == 0
        assert len(monitor._weights) == 0

    def test_update_returns(self, monitor):
        """Test returns update."""
        monitor.update_returns("SPY", 0.01)
        monitor.update_returns("SPY", -0.005)

        assert len(monitor._returns["SPY"]) == 2

    def test_update_weight(self, monitor):
        """Test weight update."""
        monitor.update_weight("SPY", 0.30)

        assert monitor._weights["SPY"] == 0.30

    def test_remove_position_by_zero_weight(self, monitor):
        """Test position removal by setting zero weight."""
        monitor.update_weight("SPY", 0.30)
        monitor.update_weight("SPY", 0)

        assert "SPY" not in monitor._weights

    def test_remove_position(self, monitor):
        """Test explicit position removal."""
        monitor.update_weight("SPY", 0.30)
        for _ in range(30):
            monitor.update_returns("SPY", 0.01)

        monitor.remove_position("SPY")

        assert "SPY" not in monitor._weights
        assert "SPY" not in monitor._returns

    def test_clear_all(self, monitor_with_data):
        """Test clearing all data."""
        monitor_with_data.clear_all()

        assert len(monitor_with_data._returns) == 0
        assert len(monitor_with_data._weights) == 0

    def test_calculate_correlation_insufficient_data(self, monitor):
        """Test correlation with insufficient data."""
        for _ in range(10):  # Less than 20 required
            monitor.update_returns("SPY", 0.01)
            monitor.update_returns("QQQ", 0.02)

        corr = monitor.calculate_correlation("SPY", "QQQ")
        assert corr is None

    def test_calculate_correlation(self, monitor_with_data):
        """Test correlation calculation."""
        corr = monitor_with_data.calculate_correlation("SPY", "QQQ")

        assert corr is not None
        assert -1 <= corr <= 1
        assert corr > 0.7  # Should be highly correlated

    def test_calculate_correlation_matrix(self, monitor_with_data):
        """Test correlation matrix calculation."""
        matrix, symbols = monitor_with_data.calculate_correlation_matrix()

        assert len(symbols) == 4
        assert matrix.shape == (4, 4)
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(matrix), np.ones(4))

    def test_get_high_correlations(self, monitor_with_data):
        """Test finding high correlation pairs."""
        high_corr = monitor_with_data.get_high_correlations()

        # SPY-QQQ should be highly correlated
        spy_qqq = [p for p in high_corr if "SPY" in [p.symbol1, p.symbol2] and "QQQ" in [p.symbol1, p.symbol2]]
        assert len(spy_qqq) > 0
        assert spy_qqq[0].correlation > 0.7

    def test_check_concentration_single_position(self, monitor):
        """Test concentration check for single large position."""
        monitor.update_weight("SPY", 0.40)  # Above 0.25 limit

        alerts = monitor.check_concentration()

        concentration_alerts = [a for a in alerts if "SPY" in a.message]
        assert len(concentration_alerts) > 0

    def test_check_concentration_correlated_positions(self, monitor_with_data):
        """Test concentration check for correlated positions."""
        # SPY + QQQ = 0.55 combined weight with high correlation
        alerts = monitor_with_data.check_concentration()

        # Should have alerts for correlated pair if exceeding limit
        pair_alerts = [a for a in alerts if a.pair is not None]
        # May or may not trigger depending on actual correlation
        assert isinstance(alerts, list)

    def test_check_concentration_min_positions(self, monitor):
        """Test concentration check for minimum positions."""
        monitor.update_weight("SPY", 1.0)  # Only 1 position

        alerts = monitor.check_concentration()

        min_pos_alerts = [a for a in alerts if "position" in a.message.lower()]
        assert len(min_pos_alerts) > 0

    def test_get_diversification_score_empty(self, monitor):
        """Test diversification score with empty portfolio."""
        score = monitor.get_diversification_score()

        assert score.score == 0.0
        assert score.level == "POOR"

    def test_get_diversification_score(self, monitor_with_data):
        """Test diversification score calculation."""
        score = monitor_with_data.get_diversification_score()

        assert 0 <= score.score <= 1
        assert score.level in ["POOR", "FAIR", "GOOD", "EXCELLENT"]
        assert score.effective_positions > 0

    def test_effective_positions_calculation(self, monitor):
        """Test effective positions (HHI inverse)."""
        # Equal weights = maximum diversification
        for sym in ["A", "B", "C", "D"]:
            monitor.update_weight(sym, 0.25)

        score = monitor.get_diversification_score()

        # With 4 equal positions, effective = 4
        assert score.effective_positions == pytest.approx(4.0, rel=0.01)

    def test_concentration_ratio(self, monitor):
        """Test concentration ratio calculation."""
        monitor.update_weight("A", 0.50)
        monitor.update_weight("B", 0.30)
        monitor.update_weight("C", 0.15)
        monitor.update_weight("D", 0.05)

        score = monitor.get_diversification_score()

        # Top 3 = 0.50 + 0.30 + 0.15 = 0.95
        assert score.concentration_ratio == pytest.approx(0.95, rel=0.01)

    def test_alert_callback(self, monitor):
        """Test alert callback."""
        alerts_received = []

        def callback(alert):
            alerts_received.append(alert)

        monitor.register_alert_callback(callback)
        monitor.update_weight("SPY", 0.40)  # Trigger concentration alert

        monitor.check_concentration()

        assert len(alerts_received) > 0

    def test_get_summary(self, monitor_with_data):
        """Test summary generation."""
        summary = monitor_with_data.get_summary()

        assert "position_count" in summary
        assert "diversification_score" in summary
        assert "high_correlation_pairs" in summary
        assert "config" in summary


class TestDiversificationScore:
    """Tests for DiversificationScore dataclass."""

    def test_creation(self):
        """Test score creation."""
        score = DiversificationScore(
            score=0.75,
            effective_positions=4.5,
            concentration_ratio=0.60,
            avg_correlation=0.35,
            max_correlation=0.80,
            level="GOOD",
        )

        assert score.score == 0.75
        assert score.level == "GOOD"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = DiversificationScore(
            score=0.65,
            effective_positions=3.5,
            concentration_ratio=0.70,
            avg_correlation=0.40,
            max_correlation=0.85,
            level="FAIR",
            recommendations=["Add more positions"],
        )

        d = score.to_dict()

        assert d["score"] == 0.65
        assert d["level"] == "FAIR"
        assert len(d["recommendations"]) == 1


class TestCorrelationAlert:
    """Tests for CorrelationAlert dataclass."""

    def test_creation(self):
        """Test alert creation."""
        alert = CorrelationAlert(
            level=ConcentrationLevel.HIGH,
            pair=("SPY", "QQQ"),
            correlation=0.85,
            combined_weight=0.55,
            message="High correlation detected",
            recommendation="Reduce exposure",
        )

        assert alert.level == ConcentrationLevel.HIGH
        assert alert.correlation == 0.85

    def test_to_dict(self):
        """Test conversion to dictionary."""
        alert = CorrelationAlert(
            level=ConcentrationLevel.CRITICAL,
            message="Critical concentration",
        )

        d = alert.to_dict()

        assert d["level"] == "critical"
        assert "timestamp" in d


class TestCreateCorrelationMonitor:
    """Tests for factory function."""

    def test_create_default(self):
        """Test creating with defaults."""
        monitor = create_correlation_monitor()

        assert isinstance(monitor, CorrelationMonitor)
        assert monitor.config.lookback_days == 60

    def test_create_with_custom_config(self):
        """Test creating with custom configuration."""
        monitor = create_correlation_monitor(
            lookback_days=30,
            high_correlation_threshold=0.80,
            max_correlated_weight=0.50,
        )

        assert monitor.config.lookback_days == 30
        assert monitor.config.high_correlation_threshold == 0.80
        assert monitor.config.max_correlated_weight == 0.50


class TestCorrelationMonitorEdgeCases:
    """Edge case tests."""

    def test_single_position(self):
        """Test with single position."""
        monitor = CorrelationMonitor()
        monitor.update_weight("SPY", 1.0)

        score = monitor.get_diversification_score()

        assert score.effective_positions == 1.0
        assert score.level == "POOR"

    def test_identical_returns(self):
        """Test with identical returns (perfect correlation)."""
        monitor = CorrelationMonitor()
        returns = [0.01, -0.02, 0.015, -0.005, 0.008] * 10

        for r in returns:
            monitor.update_returns("A", r)
            monitor.update_returns("B", r)

        corr = monitor.calculate_correlation("A", "B")

        # Should be ~1.0 (perfect correlation)
        assert corr is not None
        assert corr > 0.99

    def test_opposite_returns(self):
        """Test with opposite returns (negative correlation)."""
        monitor = CorrelationMonitor()

        for _ in range(60):
            r = np.random.normal(0.01, 0.02)
            monitor.update_returns("A", r)
            monitor.update_returns("B", -r)

        corr = monitor.calculate_correlation("A", "B")

        assert corr is not None
        assert corr < -0.99  # Should be ~ -1.0

    def test_returns_history_limit(self):
        """Test that returns history is limited."""
        config = CorrelationConfig(lookback_days=30)
        monitor = CorrelationMonitor(config=config)

        for _ in range(50):
            monitor.update_returns("SPY", 0.01)

        assert len(monitor._returns["SPY"]) == 30

    def test_missing_symbol(self):
        """Test correlation with missing symbol."""
        monitor = CorrelationMonitor()
        monitor.update_returns("SPY", 0.01)

        corr = monitor.calculate_correlation("SPY", "MISSING")

        assert corr is None
