"""
Tests for Real-Time VaR Monitor

Tests the Value at Risk monitoring module.
Part of UPGRADE-010 Sprint 4 - Test Coverage.
"""

import numpy as np
import pytest

from models.var_monitor import (
    PositionRisk,
    RiskLevel,
    VaRAlert,
    VaRLimits,
    VaRMethod,
    VaRMonitor,
    VaRResult,
    create_var_monitor,
)


class TestVaRMethod:
    """Tests for VaRMethod enum."""

    def test_methods_exist(self):
        """Test all methods exist."""
        assert VaRMethod.PARAMETRIC.value == "parametric"
        assert VaRMethod.HISTORICAL.value == "historical"
        assert VaRMethod.MONTE_CARLO.value == "monte_carlo"


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_levels_exist(self):
        """Test all levels exist."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.ELEVATED.value == "elevated"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


class TestVaRLimits:
    """Tests for VaRLimits dataclass."""

    def test_default_values(self):
        """Test default limit values."""
        limits = VaRLimits()

        assert limits.max_var_pct == 0.05
        assert limits.max_cvar_pct == 0.08
        assert limits.warning_threshold == 0.7
        assert limits.critical_threshold == 0.9

    def test_custom_values(self):
        """Test custom limit values."""
        limits = VaRLimits(
            max_var_pct=0.03,
            max_cvar_pct=0.05,
        )

        assert limits.max_var_pct == 0.03
        assert limits.max_cvar_pct == 0.05

    def test_to_dict(self):
        """Test conversion to dictionary."""
        limits = VaRLimits()
        d = limits.to_dict()

        assert d["max_var_pct"] == 0.05
        assert "warning_threshold" in d


class TestVaRResult:
    """Tests for VaRResult dataclass."""

    def test_creation(self):
        """Test result creation."""
        result = VaRResult(
            var_95=5000.0,
            var_99=7500.0,
            cvar_95=6000.0,
            cvar_99=9000.0,
            var_95_pct=0.05,
            var_99_pct=0.075,
            method=VaRMethod.HISTORICAL,
            confidence=0.9,
            calculation_time_ms=15.0,
            positions_included=5,
            portfolio_value=100000.0,
        )

        assert result.var_95 == 5000.0
        assert result.var_99 == 7500.0
        assert result.method == VaRMethod.HISTORICAL

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = VaRResult(
            var_95=5000.0,
            var_99=7500.0,
            cvar_95=6000.0,
            cvar_99=9000.0,
            var_95_pct=0.05,
            var_99_pct=0.075,
            method=VaRMethod.PARAMETRIC,
            confidence=0.85,
            calculation_time_ms=10.0,
            positions_included=3,
            portfolio_value=100000.0,
        )

        d = result.to_dict()

        assert d["var_95"] == 5000.0
        assert d["method"] == "parametric"
        assert "timestamp" in d


class TestVaRMonitor:
    """Tests for VaRMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create default monitor."""
        return VaRMonitor()

    @pytest.fixture
    def monitor_with_data(self):
        """Create monitor with sample data."""
        monitor = VaRMonitor()

        # Add positions
        monitor.update_position("SPY", 50000.0)
        monitor.update_position("QQQ", 30000.0)
        monitor.update_position("IWM", 20000.0)

        # Add returns history
        np.random.seed(42)
        for symbol in ["SPY", "QQQ", "IWM"]:
            returns = np.random.normal(0.0005, 0.015, 60)  # 60 days
            for r in returns:
                monitor.update_returns(symbol, r)

        return monitor

    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.limits is not None
        assert monitor.lookback_days == 252
        assert monitor.min_history_days == 30

    def test_update_position(self, monitor):
        """Test position updates."""
        monitor.update_position("SPY", 50000.0)
        monitor.update_position("QQQ", 30000.0)

        assert monitor._positions["SPY"] == 50000.0
        assert monitor._positions["QQQ"] == 30000.0

    def test_remove_position(self, monitor):
        """Test position removal."""
        monitor.update_position("SPY", 50000.0)
        monitor.update_position("SPY", 0)

        assert "SPY" not in monitor._positions

    def test_update_returns(self, monitor):
        """Test returns history updates."""
        monitor.update_returns("SPY", 0.01)
        monitor.update_returns("SPY", -0.005)

        assert len(monitor.returns_history["SPY"]) == 2

    def test_returns_history_limit(self, monitor):
        """Test returns history is limited."""
        for _ in range(300):
            monitor.update_returns("SPY", 0.001)

        assert len(monitor.returns_history["SPY"]) == monitor.lookback_days

    def test_calculate_var_empty_portfolio(self, monitor):
        """Test VaR with empty portfolio."""
        result = monitor.calculate_var()

        assert result.var_95 == 0.0
        assert result.portfolio_value == 0.0

    def test_calculate_var_parametric(self, monitor_with_data):
        """Test parametric VaR calculation."""
        result = monitor_with_data.calculate_var(method=VaRMethod.PARAMETRIC)

        assert result.var_95 > 0
        assert result.var_99 > result.var_95  # 99% VaR should be higher
        assert result.method == VaRMethod.PARAMETRIC
        assert result.calculation_time_ms > 0

    def test_calculate_var_historical(self, monitor_with_data):
        """Test historical VaR calculation."""
        result = monitor_with_data.calculate_var(method=VaRMethod.HISTORICAL)

        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.method == VaRMethod.HISTORICAL

    def test_calculate_var_monte_carlo(self, monitor_with_data):
        """Test Monte Carlo VaR calculation."""
        result = monitor_with_data.calculate_var(method=VaRMethod.MONTE_CARLO)

        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.method == VaRMethod.MONTE_CARLO

    def test_cvar_greater_than_var(self, monitor_with_data):
        """Test CVaR is greater than or equal to VaR."""
        result = monitor_with_data.calculate_var()

        assert result.cvar_95 >= result.var_95
        assert result.cvar_99 >= result.var_99

    def test_var_percentage(self, monitor_with_data):
        """Test VaR as percentage of portfolio."""
        result = monitor_with_data.calculate_var()

        expected_pct = result.var_95 / result.portfolio_value
        assert result.var_95_pct == pytest.approx(expected_pct, rel=0.01)

    def test_check_limits_within(self, monitor_with_data):
        """Test limits check when within limits."""
        # Use high limits
        monitor_with_data.limits = VaRLimits(max_var_pct=0.20)

        result = monitor_with_data.calculate_var()
        within, msg = monitor_with_data.check_limits(result)

        assert within is True
        assert "Within limits" in msg

    def test_check_limits_exceeded(self, monitor_with_data):
        """Test limits check when exceeded."""
        # Use very low limits
        monitor_with_data.limits = VaRLimits(max_var_pct=0.001)

        result = monitor_with_data.calculate_var()
        within, msg = monitor_with_data.check_limits(result)

        assert within is False
        assert "exceeds limit" in msg

    def test_var_contribution(self, monitor_with_data):
        """Test individual position VaR contribution."""
        contribution = monitor_with_data.get_var_contribution("SPY")

        assert contribution > 0

    def test_var_contribution_unknown_symbol(self, monitor):
        """Test VaR contribution for unknown symbol."""
        contribution = monitor.get_var_contribution("UNKNOWN")

        assert contribution == 0.0

    def test_alert_callback(self, monitor_with_data):
        """Test alert callback registration."""
        alerts_received = []

        def alert_handler(alert):
            alerts_received.append(alert)

        monitor_with_data.register_alert_callback(alert_handler)

        # Set very low limit to trigger alert
        monitor_with_data.limits = VaRLimits(max_var_pct=0.001)
        monitor_with_data.calculate_var()

        # Should have received at least one alert
        assert len(alerts_received) >= 1

    def test_risk_summary(self, monitor_with_data):
        """Test risk summary generation."""
        monitor_with_data.calculate_var()
        summary = monitor_with_data.get_risk_summary()

        assert "portfolio_value" in summary
        assert "positions" in summary
        assert "latest_var" in summary
        assert "top_contributors" in summary

    def test_estimated_var_insufficient_history(self, monitor):
        """Test estimated VaR with insufficient history."""
        monitor.update_position("SPY", 100000.0)

        # Only 5 days of history
        for _ in range(5):
            monitor.update_returns("SPY", 0.01)

        result = monitor.calculate_var()

        # Should still return a result with low confidence
        assert result.var_95 > 0
        assert result.confidence < 0.5  # Low confidence


class TestVaRAlert:
    """Tests for VaRAlert dataclass."""

    def test_creation(self):
        """Test alert creation."""
        alert = VaRAlert(
            level=RiskLevel.HIGH,
            message="VaR at 85% of limit",
            current_var_pct=0.0425,
            limit_var_pct=0.05,
            utilization_pct=0.85,
            positions_contributing=["SPY", "QQQ"],
        )

        assert alert.level == RiskLevel.HIGH
        assert len(alert.positions_contributing) == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        alert = VaRAlert(
            level=RiskLevel.CRITICAL,
            message="Critical risk level",
            current_var_pct=0.048,
            limit_var_pct=0.05,
            utilization_pct=0.96,
            positions_contributing=["SPY"],
        )

        d = alert.to_dict()

        assert d["level"] == "critical"
        assert d["utilization_pct"] == 0.96


class TestPositionRisk:
    """Tests for PositionRisk dataclass."""

    def test_creation(self):
        """Test position risk creation."""
        risk = PositionRisk(
            symbol="SPY",
            market_value=50000.0,
            weight=0.5,
            volatility=0.18,
            var_contribution=2000.0,
            var_contribution_pct=0.4,
        )

        assert risk.symbol == "SPY"
        assert risk.var_contribution == 2000.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        risk = PositionRisk(
            symbol="QQQ",
            market_value=30000.0,
            weight=0.3,
            volatility=0.22,
            var_contribution=1500.0,
            var_contribution_pct=0.3,
            beta=1.1,
            correlation=0.85,
        )

        d = risk.to_dict()

        assert d["symbol"] == "QQQ"
        assert d["beta"] == 1.1
        assert d["correlation"] == 0.85


class TestCreateVarMonitor:
    """Tests for factory function."""

    def test_create_default(self):
        """Test creating with defaults."""
        monitor = create_var_monitor()

        assert isinstance(monitor, VaRMonitor)
        assert monitor.limits.max_var_pct == 0.05

    def test_create_with_custom_limit(self):
        """Test creating with custom limit."""
        monitor = create_var_monitor(max_var_pct=0.03)

        assert monitor.limits.max_var_pct == 0.03

    def test_create_with_lookback(self):
        """Test creating with custom lookback."""
        monitor = create_var_monitor(lookback_days=126)

        assert monitor.lookback_days == 126
