"""
Tests for Greeks Risk Monitor (Sprint 4 Expansion)

Tests real-time Greeks exposure monitoring with limits and alerts.
Part of UPGRADE-010 Sprint 4 Expansion.
"""

import pytest

from models.greeks_monitor import (
    GreeksAlertLevel,
    GreeksLimits,
    GreeksMonitor,
    GreeksType,
    HedgeRecommendation,
    PortfolioGreeksExposure,
    PositionGreeksSnapshot,
    RiskProfile,
    create_greeks_monitor,
)


class TestGreeksLimits:
    """Tests for GreeksLimits dataclass."""

    def test_default_values(self):
        """Test default limit values."""
        limits = GreeksLimits()

        assert limits.max_delta == 50.0
        assert limits.min_delta == -50.0
        assert limits.max_gamma == 10.0
        assert limits.max_vega == 1000.0
        assert limits.warning_threshold == 0.75

    def test_from_conservative_profile(self):
        """Test conservative risk profile."""
        limits = GreeksLimits.from_profile(RiskProfile.CONSERVATIVE)

        assert limits.max_delta == 30.0
        assert limits.max_gamma == 5.0
        assert limits.max_vega == 500.0

    def test_from_moderate_profile(self):
        """Test moderate risk profile."""
        limits = GreeksLimits.from_profile(RiskProfile.MODERATE)

        assert limits.max_delta == 50.0
        assert limits.max_gamma == 10.0
        assert limits.max_vega == 1000.0

    def test_from_aggressive_profile(self):
        """Test aggressive risk profile."""
        limits = GreeksLimits.from_profile(RiskProfile.AGGRESSIVE)

        assert limits.max_delta == 100.0
        assert limits.max_gamma == 20.0
        assert limits.max_vega == 2000.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        limits = GreeksLimits()
        d = limits.to_dict()

        assert d["max_delta"] == 50.0
        assert "warning_threshold" in d
        assert "max_position_delta" in d


class TestPositionGreeksSnapshot:
    """Tests for PositionGreeksSnapshot dataclass."""

    def test_creation(self):
        """Test snapshot creation."""
        snapshot = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.45,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=10,
        )

        assert snapshot.delta == 0.45
        assert snapshot.quantity == 10

    def test_net_delta(self):
        """Test net delta calculation."""
        snapshot = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.50,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=10,
            contract_multiplier=100,
        )

        # net_delta = delta * quantity * multiplier
        assert snapshot.net_delta == 0.50 * 10 * 100  # 500

    def test_net_gamma(self):
        """Test net gamma calculation."""
        snapshot = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.50,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=10,
        )

        assert snapshot.net_gamma == 0.02 * 10 * 100  # 20

    def test_net_vega(self):
        """Test net vega calculation."""
        snapshot = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.50,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=10,
        )

        assert snapshot.net_vega == 0.15 * 10 * 100  # 150

    def test_net_theta(self):
        """Test net theta calculation."""
        snapshot = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.50,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=10,
        )

        assert snapshot.net_theta == -0.03 * 10 * 100  # -30

    def test_to_dict(self):
        """Test conversion to dictionary."""
        snapshot = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.45,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=10,
        )

        d = snapshot.to_dict()

        assert d["symbol"] == "SPY_241220C450"
        assert d["net_delta"] == 450.0
        assert "timestamp" in d


class TestGreeksMonitor:
    """Tests for GreeksMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create default monitor."""
        return GreeksMonitor()

    @pytest.fixture
    def long_call_position(self):
        """Create long call position."""
        return PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.45,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=1,
        )

    @pytest.fixture
    def short_put_position(self):
        """Create short put position."""
        return PositionGreeksSnapshot(
            symbol="SPY_241220P440",
            underlying="SPY",
            delta=0.30,  # Short put has positive delta
            gamma=0.015,
            vega=0.10,
            theta=0.02,  # Positive theta from short position
            quantity=-1,  # Short
        )

    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.limits is not None
        assert monitor.risk_profile == RiskProfile.MODERATE
        assert len(monitor._positions) == 0

    def test_update_position(self, monitor, long_call_position):
        """Test position update."""
        monitor.update_position(long_call_position)

        assert "SPY_241220C450" in monitor._positions
        assert monitor._positions["SPY_241220C450"].delta == 0.45

    def test_remove_position(self, monitor, long_call_position):
        """Test position removal."""
        monitor.update_position(long_call_position)
        monitor.remove_position("SPY_241220C450")

        assert "SPY_241220C450" not in monitor._positions

    def test_remove_position_by_zero_quantity(self, monitor, long_call_position):
        """Test position removal by setting quantity to 0."""
        monitor.update_position(long_call_position)

        zero_quantity = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.45,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=0,
        )
        monitor.update_position(zero_quantity)

        assert "SPY_241220C450" not in monitor._positions

    def test_clear_positions(self, monitor, long_call_position, short_put_position):
        """Test clearing all positions."""
        monitor.update_position(long_call_position)
        monitor.update_position(short_put_position)
        monitor.clear_positions()

        assert len(monitor._positions) == 0

    def test_get_portfolio_exposure_empty(self, monitor):
        """Test portfolio exposure with no positions."""
        exposure = monitor.get_portfolio_exposure()

        assert exposure.total_delta == 0.0
        assert exposure.total_gamma == 0.0
        assert exposure.position_count == 0

    def test_get_portfolio_exposure(self, monitor, long_call_position):
        """Test portfolio exposure calculation."""
        monitor.update_position(long_call_position)
        exposure = monitor.get_portfolio_exposure()

        assert exposure.total_delta == 45.0  # 0.45 * 1 * 100
        assert exposure.total_gamma == 2.0  # 0.02 * 1 * 100
        assert exposure.total_vega == 15.0
        assert exposure.total_theta == -3.0
        assert exposure.position_count == 1

    def test_get_portfolio_exposure_multiple(self, monitor, long_call_position, short_put_position):
        """Test portfolio exposure with multiple positions."""
        monitor.update_position(long_call_position)
        monitor.update_position(short_put_position)
        exposure = monitor.get_portfolio_exposure()

        # Long call: delta=45, Short put: delta=-30
        expected_delta = 45.0 + (-30.0)
        assert exposure.total_delta == expected_delta
        assert exposure.position_count == 2

    def test_check_limits_within(self, monitor, long_call_position):
        """Test limits check when within limits."""
        monitor.update_position(long_call_position)
        within_limits, alerts = monitor.check_limits()

        assert within_limits is True
        # Small position should not trigger alerts
        breach_alerts = [a for a in alerts if a.level == GreeksAlertLevel.BREACH]
        assert len(breach_alerts) == 0

    def test_check_limits_delta_breach(self, monitor):
        """Test limits check with delta breach."""
        # Create position that exceeds delta limit
        large_position = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.60,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=10,  # 10 contracts = 600 delta (exceeds 50)
        )
        monitor.update_position(large_position)
        within_limits, alerts = monitor.check_limits()

        assert within_limits is False
        delta_alerts = [a for a in alerts if a.greeks_type == GreeksType.DELTA]
        assert len(delta_alerts) > 0

    def test_check_limits_gamma_breach(self, monitor):
        """Test limits check with gamma breach."""
        # Create position with high gamma
        high_gamma = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.50,
            gamma=0.10,  # High gamma
            vega=0.15,
            theta=-0.03,
            quantity=5,  # 50 gamma (exceeds 10)
        )
        monitor.update_position(high_gamma)
        within_limits, alerts = monitor.check_limits()

        gamma_alerts = [a for a in alerts if a.greeks_type == GreeksType.GAMMA]
        assert len(gamma_alerts) > 0

    def test_check_limits_warning_threshold(self, monitor):
        """Test warning threshold detection."""
        # Create position at ~80% of delta limit
        warning_position = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.40,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=1,  # 40 delta (80% of 50 limit)
        )
        monitor.update_position(warning_position)
        within_limits, alerts = monitor.check_limits()

        assert within_limits is True  # Still within limits
        warning_alerts = [a for a in alerts if a.level == GreeksAlertLevel.WARNING]
        assert len(warning_alerts) >= 1

    def test_get_hedging_recommendation_no_hedge(self, monitor):
        """Test hedging recommendation when not needed."""
        # Use small position (net_delta=20, which is 40% of limit 50)
        # This stays below warning_threshold (75%)
        small_position = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.20,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=1,  # net_delta = 20 (40% of 50)
        )
        monitor.update_position(small_position)
        recommendation = monitor.get_hedging_recommendation()

        assert recommendation.needed is False
        assert recommendation.priority == "LOW"

    def test_get_hedging_recommendation_hedge_needed(self, monitor):
        """Test hedging recommendation when needed."""
        # Create position near limit
        large_position = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.50,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=8,  # 400 delta (80% of 500 limit... wait limits are 50)
        )
        # Actually with default limits max_delta=50, 8 contracts = 400 delta
        # This far exceeds the limit, so hedge should be urgently needed
        monitor.update_position(large_position)
        recommendation = monitor.get_hedging_recommendation()

        assert recommendation.needed is True
        assert recommendation.priority in ["HIGH", "URGENT"]
        assert recommendation.delta_adjustment < 0  # Need to reduce delta

    def test_alert_callback(self, monitor):
        """Test alert callback registration and triggering."""
        alerts_received = []

        def callback(alert):
            alerts_received.append(alert)

        monitor.register_alert_callback(callback)

        # Create breach position
        breach_position = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.60,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=10,
        )
        monitor.update_position(breach_position)
        monitor.check_limits()

        assert len(alerts_received) > 0

    def test_get_exposure_by_underlying(self, monitor):
        """Test exposure grouping by underlying."""
        # SPY position
        spy_call = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.45,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=1,
        )

        # QQQ position
        qqq_call = PositionGreeksSnapshot(
            symbol="QQQ_241220C380",
            underlying="QQQ",
            delta=0.50,
            gamma=0.025,
            vega=0.12,
            theta=-0.025,
            quantity=2,
        )

        monitor.update_position(spy_call)
        monitor.update_position(qqq_call)

        by_underlying = monitor.get_exposure_by_underlying()

        assert "SPY" in by_underlying
        assert "QQQ" in by_underlying
        assert by_underlying["SPY"].total_delta == 45.0
        assert by_underlying["QQQ"].total_delta == 100.0

    def test_get_alert_history(self, monitor):
        """Test alert history retrieval."""
        # Create breach to generate alerts
        breach_position = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.60,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=10,
        )
        monitor.update_position(breach_position)
        monitor.check_limits()

        history = monitor.get_alert_history()
        assert len(history) > 0

    def test_get_summary(self, monitor, long_call_position):
        """Test summary generation."""
        monitor.update_position(long_call_position)
        summary = monitor.get_summary()

        assert "within_limits" in summary
        assert "exposure" in summary
        assert "limits" in summary
        assert "utilization" in summary
        assert "position_count" in summary
        assert summary["position_count"] == 1


class TestPortfolioGreeksExposure:
    """Tests for PortfolioGreeksExposure dataclass."""

    def test_creation(self):
        """Test exposure creation."""
        exposure = PortfolioGreeksExposure(
            total_delta=100.0,
            total_gamma=5.0,
            total_vega=500.0,
            total_theta=-50.0,
            position_count=5,
        )

        assert exposure.total_delta == 100.0
        assert exposure.position_count == 5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        exposure = PortfolioGreeksExposure(
            total_delta=100.0,
            total_gamma=5.0,
            total_vega=500.0,
            total_theta=-50.0,
            position_count=5,
        )

        d = exposure.to_dict()

        assert d["total_delta"] == 100.0
        assert "timestamp" in d


class TestHedgeRecommendation:
    """Tests for HedgeRecommendation dataclass."""

    def test_creation(self):
        """Test recommendation creation."""
        rec = HedgeRecommendation(
            needed=True,
            priority="HIGH",
            delta_adjustment=-50.0,
            suggested_hedge="Sell 50 shares of SPY",
            estimated_cost=25.0,
            reason="Delta at 90% of limit",
        )

        assert rec.needed is True
        assert rec.priority == "HIGH"
        assert rec.delta_adjustment == -50.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rec = HedgeRecommendation(
            needed=True,
            priority="MEDIUM",
            delta_adjustment=-30.0,
            suggested_hedge="Sell 30 shares",
        )

        d = rec.to_dict()

        assert d["needed"] is True
        assert d["priority"] == "MEDIUM"


class TestGreeksAlert:
    """Tests for GreeksAlert dataclass."""

    def test_creation(self):
        """Test alert creation."""
        from models.greeks_monitor import GreeksAlert

        alert = GreeksAlert(
            level=GreeksAlertLevel.WARNING,
            greeks_type=GreeksType.DELTA,
            current_value=40.0,
            limit_value=50.0,
            utilization_pct=0.8,
            message="Delta at 80% of limit",
        )

        assert alert.level == GreeksAlertLevel.WARNING
        assert alert.utilization_pct == 0.8

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from models.greeks_monitor import GreeksAlert

        alert = GreeksAlert(
            level=GreeksAlertLevel.CRITICAL,
            greeks_type=GreeksType.GAMMA,
            current_value=9.0,
            limit_value=10.0,
            utilization_pct=0.9,
            message="Gamma at 90% of limit",
        )

        d = alert.to_dict()

        assert d["level"] == "critical"
        assert d["greeks_type"] == "gamma"


class TestCreateGreeksMonitor:
    """Tests for factory function."""

    def test_create_default(self):
        """Test creating with defaults."""
        monitor = create_greeks_monitor()

        assert isinstance(monitor, GreeksMonitor)
        assert monitor.limits.max_delta == 50.0

    def test_create_conservative(self):
        """Test creating with conservative profile."""
        monitor = create_greeks_monitor(risk_profile="conservative")

        assert monitor.limits.max_delta == 30.0

    def test_create_aggressive(self):
        """Test creating with aggressive profile."""
        monitor = create_greeks_monitor(risk_profile="aggressive")

        assert monitor.limits.max_delta == 100.0

    def test_create_with_custom_limits(self):
        """Test creating with custom limit overrides."""
        monitor = create_greeks_monitor(
            risk_profile="moderate",
            max_delta=75.0,
            max_gamma=15.0,
        )

        assert monitor.limits.max_delta == 75.0
        assert monitor.limits.min_delta == -75.0
        assert monitor.limits.max_gamma == 15.0


class TestGreeksMonitorEdgeCases:
    """Edge case tests."""

    def test_empty_portfolio_check(self):
        """Test checking limits with empty portfolio."""
        monitor = GreeksMonitor()
        within_limits, alerts = monitor.check_limits()

        assert within_limits is True
        assert len(alerts) == 0

    def test_short_positions(self):
        """Test handling of short positions."""
        monitor = GreeksMonitor()

        short_call = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=-0.45,  # Short call has negative delta
            gamma=-0.02,
            vega=-0.15,
            theta=0.03,  # Positive theta
            quantity=-1,
        )

        monitor.update_position(short_call)
        exposure = monitor.get_portfolio_exposure()

        # Short 1 contract = positive delta exposure (double negative)
        assert exposure.total_delta == 45.0  # -0.45 * -1 * 100
        assert exposure.total_theta == -3.0

    def test_position_level_limits(self):
        """Test position-level limit checking."""
        # Use limits with tight position limits
        limits = GreeksLimits(
            max_position_delta=20.0,
            max_position_gamma=2.0,
        )
        monitor = GreeksMonitor(limits=limits)

        # Position exceeds position-level delta limit
        large_delta = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.30,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=1,  # 30 delta > 20 limit
        )

        monitor.update_position(large_delta)
        _, alerts = monitor.check_limits()

        position_alerts = [a for a in alerts if a.position is not None]
        assert len(position_alerts) > 0

    def test_theta_positive_allowed(self):
        """Test that positive theta (income) doesn't trigger alerts."""
        monitor = GreeksMonitor()

        # Credit spread with positive theta
        credit_spread = PositionGreeksSnapshot(
            symbol="SPY_SPREAD",
            underlying="SPY",
            delta=0.10,
            gamma=0.005,
            vega=0.05,
            theta=0.05,  # Positive theta
            quantity=1,
        )

        monitor.update_position(credit_spread)
        _, alerts = monitor.check_limits()

        theta_alerts = [a for a in alerts if a.greeks_type == GreeksType.THETA]
        assert len(theta_alerts) == 0

    def test_multiple_callbacks(self):
        """Test multiple alert callbacks."""
        monitor = GreeksMonitor()
        callback1_calls = []
        callback2_calls = []

        def callback1(alert):
            callback1_calls.append(alert)

        def callback2(alert):
            callback2_calls.append(alert)

        monitor.register_alert_callback(callback1)
        monitor.register_alert_callback(callback2)

        # Trigger alerts
        breach_position = PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.60,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=10,
        )
        monitor.update_position(breach_position)
        monitor.check_limits()

        assert len(callback1_calls) > 0
        assert len(callback2_calls) > 0
        assert len(callback1_calls) == len(callback2_calls)
