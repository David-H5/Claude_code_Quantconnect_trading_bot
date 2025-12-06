"""
Tests for Alerting Service and System Monitor (UPGRADE-013)

Tests multi-channel alerting, rate limiting, aggregation, and health monitoring.
"""

import time
from unittest.mock import MagicMock

import pytest

from utils.alerting_service import (
    Alert,
    AlertAggregator,
    AlertCategory,
    AlertingService,
    AlertSeverity,
    ConsoleChannel,
    DiscordChannel,
    EmailChannel,
    RateLimiter,
    SlackChannel,
    create_alerting_service,
)
from utils.system_monitor import (
    HealthStatus,
    ServiceState,
    SystemMonitor,
    create_disk_check,
    create_memory_check,
    create_system_monitor,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def alerting_service():
    """Create an alerting service for testing."""
    return AlertingService()


@pytest.fixture
def system_monitor():
    """Create a system monitor for testing."""
    return SystemMonitor()


@pytest.fixture
def mock_config():
    """Create mock alerting config."""
    config = MagicMock()
    config.console_alerts = True
    config.email_enabled = False
    config.discord_enabled = False
    config.slack_enabled = False
    config.min_severity = "WARNING"
    config.rate_limit_per_minute = 10
    config.aggregate_similar = True
    config.aggregation_window_seconds = 60
    return config


# ==============================================================================
# Test Alert Data Types
# ==============================================================================


class TestAlertSeverity:
    """Test AlertSeverity enum."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert AlertSeverity.DEBUG.value == "debug"
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_severity_levels(self):
        """Test severity numeric levels."""
        assert AlertSeverity.DEBUG.level < AlertSeverity.INFO.level
        assert AlertSeverity.INFO.level < AlertSeverity.WARNING.level
        assert AlertSeverity.WARNING.level < AlertSeverity.ERROR.level
        assert AlertSeverity.ERROR.level < AlertSeverity.CRITICAL.level


class TestAlert:
    """Test Alert dataclass."""

    def test_create_alert(self):
        """Test creating an alert."""
        alert = Alert(
            title="Test Alert",
            message="This is a test",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
        )
        assert alert.title == "Test Alert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.alert_id != ""

    def test_to_dict(self):
        """Test converting alert to dict."""
        alert = Alert(
            title="Test",
            message="Test message",
            data={"key": "value"},
        )
        data = alert.to_dict()

        assert data["title"] == "Test"
        assert data["message"] == "Test message"
        assert data["data"]["key"] == "value"

    def test_format_console(self):
        """Test console formatting."""
        alert = Alert(
            title="Test",
            message="Test message",
            severity=AlertSeverity.ERROR,
        )
        formatted = alert.format_console()

        assert "[ERROR]" in formatted
        assert "Test" in formatted

    def test_format_markdown(self):
        """Test markdown formatting."""
        alert = Alert(
            title="Test",
            message="Test message",
            severity=AlertSeverity.WARNING,
        )
        formatted = alert.format_markdown()

        assert "**Test**" in formatted
        assert "WARNING" in formatted.upper()


# ==============================================================================
# Test Alert Channels
# ==============================================================================


class TestConsoleChannel:
    """Test ConsoleChannel."""

    def test_send_alert(self, capsys):
        """Test sending alert to console."""
        channel = ConsoleChannel(enabled=True)
        alert = Alert(title="Test", message="Hello")

        result = channel.send(alert)

        assert result is True
        captured = capsys.readouterr()
        assert "Test" in captured.out

    def test_disabled_channel(self):
        """Test disabled channel doesn't send."""
        channel = ConsoleChannel(enabled=False)
        alert = Alert(title="Test", message="Hello")

        result = channel.send(alert)

        assert result is False

    def test_channel_stats(self, capsys):
        """Test channel statistics."""
        channel = ConsoleChannel(enabled=True)
        alert = Alert(title="Test", message="Hello")

        channel.send(alert)
        stats = channel.get_stats()

        assert stats["send_count"] == 1
        assert stats["name"] == "console"


class TestEmailChannel:
    """Test EmailChannel."""

    def test_disabled_without_config(self):
        """Test email channel fails without config."""
        channel = EmailChannel(
            smtp_host="",
            recipients=[],
            enabled=True,
        )
        alert = Alert(title="Test", message="Hello")

        result = channel.send(alert)

        assert result is False


class TestDiscordChannel:
    """Test DiscordChannel."""

    def test_disabled_without_webhook(self):
        """Test Discord fails without webhook."""
        channel = DiscordChannel(webhook_url="", enabled=True)
        alert = Alert(title="Test", message="Hello")

        result = channel.send(alert)

        assert result is False


class TestSlackChannel:
    """Test SlackChannel."""

    def test_disabled_without_webhook(self):
        """Test Slack fails without webhook."""
        channel = SlackChannel(webhook_url="", enabled=True)
        alert = Alert(title="Test", message="Hello")

        result = channel.send(alert)

        assert result is False


# ==============================================================================
# Test Rate Limiting
# ==============================================================================


class TestRateLimiter:
    """Test RateLimiter."""

    def test_allows_initial_request(self):
        """Test first request is allowed."""
        limiter = RateLimiter(max_per_minute=5)

        assert limiter.check("test") is True

    def test_blocks_after_limit(self):
        """Test blocks after limit reached."""
        limiter = RateLimiter(max_per_minute=3, window_seconds=60)

        limiter.check("test")
        limiter.check("test")
        limiter.check("test")

        # Fourth should be blocked
        assert limiter.check("test") is False

    def test_different_keys_independent(self):
        """Test different keys have independent limits."""
        limiter = RateLimiter(max_per_minute=2)

        limiter.check("key1")
        limiter.check("key1")

        # key1 is at limit, but key2 should be allowed
        assert limiter.check("key2") is True

    def test_reset_key(self):
        """Test resetting a key."""
        limiter = RateLimiter(max_per_minute=1)

        limiter.check("test")
        assert limiter.check("test") is False

        limiter.reset("test")
        assert limiter.check("test") is True


# ==============================================================================
# Test Alert Aggregation
# ==============================================================================


class TestAlertAggregator:
    """Test AlertAggregator."""

    def test_first_alert_sent(self):
        """Test first alert is sent immediately."""
        aggregator = AlertAggregator(min_count_to_aggregate=3)
        alert = Alert(title="Test", message="Hello")

        result = aggregator.add(alert)

        assert result is not None
        assert result.title == "Test"

    def test_aggregates_repeated_alerts(self):
        """Test similar alerts are aggregated."""
        aggregator = AlertAggregator(min_count_to_aggregate=3)
        alert = Alert(title="Test", message="Hello")

        # First is sent
        result1 = aggregator.add(alert)
        assert result1 is not None

        # Second is suppressed
        result2 = aggregator.add(alert)
        assert result2 is None

        # Third triggers summary
        result3 = aggregator.add(alert)
        assert result3 is not None
        assert "Repeated" in result3.title

    def test_different_alerts_not_aggregated(self):
        """Test different alerts are not aggregated."""
        aggregator = AlertAggregator()

        alert1 = Alert(title="Alert 1", message="First")
        alert2 = Alert(title="Alert 2", message="Second")

        result1 = aggregator.add(alert1)
        result2 = aggregator.add(alert2)

        # Both should be sent
        assert result1 is not None
        assert result2 is not None


# ==============================================================================
# Test AlertingService
# ==============================================================================


class TestAlertingService:
    """Test AlertingService."""

    def test_service_creation(self, alerting_service):
        """Test service creation."""
        assert alerting_service is not None

    def test_send_alert(self, alerting_service, capsys):
        """Test sending an alert."""
        result = alerting_service.send_alert(
            title="Test Alert",
            message="This is a test",
            severity=AlertSeverity.WARNING,
        )

        assert result is True
        captured = capsys.readouterr()
        assert "Test Alert" in captured.out

    def test_severity_filtering(self, alerting_service):
        """Test alerts below min severity are filtered."""
        # Default min severity is WARNING
        result = alerting_service.send_alert(
            title="Debug Alert",
            message="Should be filtered",
            severity=AlertSeverity.DEBUG,
        )

        # DEBUG is below WARNING, should be filtered
        assert result is False

    def test_send_trading_alert(self, alerting_service, capsys):
        """Test trading alert helper."""
        result = alerting_service.send_trading_alert(
            title="Trade Executed",
            message="SPY call bought",
        )

        # INFO is below WARNING (default min), so filtered
        # Change min severity to test
        alerting_service._min_severity = AlertSeverity.INFO
        result = alerting_service.send_trading_alert(
            title="Trade Executed",
            message="SPY call bought",
        )

        assert result is True

    def test_send_risk_alert(self, alerting_service, capsys):
        """Test risk alert helper."""
        result = alerting_service.send_risk_alert(
            title="High Exposure",
            message="Portfolio exposure at 80%",
        )

        assert result is True
        captured = capsys.readouterr()
        assert "High Exposure" in captured.out

    def test_send_circuit_breaker_alert(self, alerting_service, capsys):
        """Test circuit breaker alert."""
        result = alerting_service.send_circuit_breaker_alert(
            reason="Daily loss limit exceeded",
        )

        assert result is True
        captured = capsys.readouterr()
        assert "Circuit Breaker" in captured.out

    def test_add_channel(self, alerting_service):
        """Test adding a channel."""
        channel = ConsoleChannel(enabled=True)
        alerting_service.add_channel(channel)

        assert "console" in alerting_service._channels

    def test_remove_channel(self, alerting_service):
        """Test removing a channel."""
        alerting_service.remove_channel("console")

        assert "console" not in alerting_service._channels

    def test_alert_listener(self, alerting_service):
        """Test alert listener."""
        received = []

        def listener(alert):
            received.append(alert)

        alerting_service.add_listener(listener)
        alerting_service.send_alert(
            title="Test",
            message="Hello",
            severity=AlertSeverity.WARNING,
        )

        assert len(received) == 1
        assert received[0].title == "Test"

    def test_get_recent_alerts(self, alerting_service):
        """Test getting recent alerts."""
        alerting_service.send_alert("Alert 1", "First", severity=AlertSeverity.WARNING)
        alerting_service.send_alert("Alert 2", "Second", severity=AlertSeverity.ERROR)

        alerts = alerting_service.get_recent_alerts(limit=10)

        assert len(alerts) == 2

    def test_get_stats(self, alerting_service):
        """Test getting service stats."""
        stats = alerting_service.get_stats()

        assert "channels" in stats
        assert "total_alerts" in stats


class TestAlertingServiceWithConfig:
    """Test AlertingService with configuration."""

    def test_config_integration(self, mock_config):
        """Test service with config."""
        service = AlertingService(config=mock_config)

        assert service._min_severity == AlertSeverity.WARNING
        assert "console" in service._channels


# ==============================================================================
# Test SystemMonitor
# ==============================================================================


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_status_values(self):
        """Test health status values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestServiceState:
    """Test ServiceState dataclass."""

    def test_create_state(self):
        """Test creating service state."""
        state = ServiceState(name="test_service")

        assert state.name == "test_service"
        assert state.status == HealthStatus.UNKNOWN
        assert state.total_checks == 0

    def test_to_dict(self):
        """Test converting to dict."""
        state = ServiceState(
            name="test",
            status=HealthStatus.HEALTHY,
            total_checks=10,
            total_failures=1,
        )
        data = state.to_dict()

        assert data["name"] == "test"
        assert data["status"] == "healthy"
        assert data["uptime_pct"] == 90.0


class TestSystemMonitor:
    """Test SystemMonitor."""

    def test_monitor_creation(self, system_monitor):
        """Test monitor creation."""
        assert system_monitor is not None
        assert not system_monitor.is_running()

    def test_register_service(self, system_monitor):
        """Test registering a service."""
        system_monitor.register_service(
            "test_service",
            lambda: True,
            interval_seconds=30,
        )

        assert "test_service" in system_monitor._services

    def test_unregister_service(self, system_monitor):
        """Test unregistering a service."""
        system_monitor.register_service("test", lambda: True)
        system_monitor.unregister_service("test")

        assert "test" not in system_monitor._services

    def test_check_healthy_service(self, system_monitor):
        """Test checking a healthy service."""
        # recovery_threshold=1 means service is healthy after 1 success
        system_monitor.register_service("healthy", lambda: True, recovery_threshold=1)

        result = system_monitor.check_service("healthy")

        assert result is True

    def test_check_unhealthy_service(self, system_monitor):
        """Test checking an unhealthy service."""
        system_monitor.register_service(
            "unhealthy",
            lambda: False,
            failure_threshold=1,
        )

        result = system_monitor.check_service("unhealthy")

        assert result is False
        state = system_monitor.get_service_status("unhealthy")
        assert state.status == HealthStatus.UNHEALTHY

    def test_check_failing_service(self, system_monitor):
        """Test service that throws exception."""

        def failing_check():
            raise Exception("Service error")

        system_monitor.register_service(
            "failing",
            failing_check,
            failure_threshold=1,
        )

        result = system_monitor.check_service("failing")

        assert result is False
        state = system_monitor.get_service_status("failing")
        assert "Service error" in state.error_message

    def test_check_all(self, system_monitor):
        """Test checking all services."""
        # recovery_threshold=1 for healthy services to be marked healthy after 1 check
        system_monitor.register_service("s1", lambda: True, recovery_threshold=1)
        system_monitor.register_service("s2", lambda: True, recovery_threshold=1)
        system_monitor.register_service("s3", lambda: False, failure_threshold=1)

        results = system_monitor.check_all()

        assert results["s1"] is True
        assert results["s2"] is True
        assert results["s3"] is False

    def test_get_status_summary(self, system_monitor):
        """Test status summary."""
        system_monitor.register_service("healthy", lambda: True)
        system_monitor.check_service("healthy")

        summary = system_monitor.get_status_summary()

        assert "overall_status" in summary
        assert "services" in summary

    def test_status_listener(self, system_monitor):
        """Test status change listener."""
        changes = []

        def listener(name, old, new):
            changes.append((name, old, new))

        system_monitor.add_status_listener(listener)
        system_monitor.register_service("test", lambda: False, failure_threshold=1)
        system_monitor.check_service("test")

        assert len(changes) >= 1

    def test_start_stop(self, system_monitor):
        """Test starting and stopping monitor."""
        system_monitor.register_service("test", lambda: True, interval_seconds=1)

        system_monitor.start()
        assert system_monitor.is_running()

        time.sleep(0.1)

        system_monitor.stop()
        assert not system_monitor.is_running()


# ==============================================================================
# Test Built-in Health Checks
# ==============================================================================


class TestBuiltinChecks:
    """Test built-in health check functions."""

    def test_memory_check_creation(self):
        """Test memory check creation."""
        check = create_memory_check(threshold_pct=95.0)
        # Should return callable
        assert callable(check)

    def test_disk_check_creation(self):
        """Test disk check creation."""
        check = create_disk_check(path="/", threshold_pct=95.0)
        assert callable(check)


# ==============================================================================
# Test Factory Functions
# ==============================================================================


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_alerting_service(self):
        """Test alerting service factory."""
        service = create_alerting_service()

        assert isinstance(service, AlertingService)

    def test_create_system_monitor(self):
        """Test system monitor factory."""
        monitor = create_system_monitor(check_interval_seconds=30)

        assert isinstance(monitor, SystemMonitor)
        assert monitor.default_interval == 30


# ==============================================================================
# Test Integration
# ==============================================================================


class TestIntegration:
    """Test integration between AlertingService and SystemMonitor."""

    def test_monitor_with_alerting(self):
        """Test monitor sends alerts on status change."""
        alerting = AlertingService()
        monitor = SystemMonitor(alerting_service=alerting)

        # Track alerts
        alerts = []
        alerting.add_listener(lambda a: alerts.append(a))

        # Register failing service
        monitor.register_service(
            "failing_service",
            lambda: False,
            failure_threshold=1,
        )

        # Check service (should trigger alert)
        monitor.check_service("failing_service")

        # Should have received alert
        assert len(alerts) >= 1
        assert "failing_service" in alerts[0].title.lower() or "failing_service" in str(alerts[0].data)
