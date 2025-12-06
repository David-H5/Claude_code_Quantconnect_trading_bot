"""
Tests for Observability Agent

UPGRADE-014 Category 2: Observability & Debugging
Updated to match actual API implementation.
"""

import time
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from llm.agents.base import AgentRole
from llm.agents.observability_agent import (
    AgentMetricsSnapshot,
    Alert,
    AlertSeverity,
    AlertType,
    MonitoringThresholds,
    ObservabilityAgent,
    create_observability_agent,
)


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_all_severities_exist(self):
        """Test all expected severities exist."""
        severities = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL,
        ]
        assert len(severities) == 4

    def test_severity_values(self):
        """Test severity string values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestAlertType:
    """Tests for AlertType enum."""

    def test_all_types_exist(self):
        """Test all expected alert types exist."""
        types = [
            AlertType.HIGH_ERROR_RATE,
            AlertType.HIGH_LATENCY,
            AlertType.AGENT_UNHEALTHY,
            AlertType.TOKEN_BUDGET_WARNING,
            AlertType.TOKEN_BUDGET_EXCEEDED,
            AlertType.NO_RESPONSE,
            AlertType.ANOMALY_DETECTED,
            AlertType.AGENT_RECOVERED,
        ]
        assert len(types) == 8


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity=AlertSeverity.ERROR,
            message="Error rate exceeded threshold",
            agent_name="analyst",
        )

        assert alert.alert_type == AlertType.HIGH_ERROR_RATE
        assert alert.severity == AlertSeverity.ERROR
        assert alert.agent_name == "analyst"

    def test_alert_has_id(self):
        """Test alert has unique ID."""
        alert = Alert(
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity=AlertSeverity.WARNING,
            message="Test alert",
        )

        assert alert.alert_id is not None
        assert len(alert.alert_id) > 0

    def test_alert_timestamp(self):
        """Test alert has timestamp."""
        alert = Alert(
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity=AlertSeverity.WARNING,
            message="Test alert",
        )

        assert alert.timestamp is not None
        assert isinstance(alert.timestamp, datetime)

    def test_resolve_alert(self):
        """Test resolving an alert."""
        alert = Alert(
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity=AlertSeverity.WARNING,
            message="Test alert",
        )

        assert not alert.resolved
        assert alert.resolved_at is None

        alert.resolve()

        assert alert.resolved
        assert alert.resolved_at is not None

    def test_to_dict(self):
        """Test serialization."""
        alert = Alert(
            alert_type=AlertType.HIGH_LATENCY,
            severity=AlertSeverity.ERROR,
            message="Latency too high",
            agent_name="analyst",
        )

        d = alert.to_dict()

        assert d["alert_type"] == "high_latency"
        assert d["severity"] == "error"
        assert d["agent_name"] == "analyst"


class TestMonitoringThresholds:
    """Tests for MonitoringThresholds dataclass."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = MonitoringThresholds()

        # Updated to match actual defaults (percentages)
        assert thresholds.error_rate_warning == 10.0
        assert thresholds.error_rate_critical == 25.0
        assert thresholds.latency_warning_ms == 2000.0
        assert thresholds.latency_critical_ms == 5000.0
        assert thresholds.token_budget_warning_pct == 80.0
        assert thresholds.token_budget_critical_pct == 95.0

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = MonitoringThresholds(
            error_rate_warning=5.0,
            latency_warning_ms=500.0,
        )

        assert thresholds.error_rate_warning == 5.0
        assert thresholds.latency_warning_ms == 500.0


class TestAgentMetricsSnapshot:
    """Tests for AgentMetricsSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating a metrics snapshot."""
        snapshot = AgentMetricsSnapshot(
            agent_name="analyst",
            is_healthy=True,
            total_calls=100,
            failed_calls=5,
            avg_latency_ms=150.0,
        )

        assert snapshot.agent_name == "analyst"
        assert snapshot.is_healthy
        assert snapshot.total_calls == 100
        assert snapshot.failed_calls == 5

    def test_error_rate_field(self):
        """Test error rate is a field."""
        snapshot = AgentMetricsSnapshot(
            agent_name="analyst",
            total_calls=100,
            failed_calls=10,
            error_rate=0.1,  # 10%
        )

        assert snapshot.error_rate == 0.1

    def test_to_dict(self):
        """Test serialization."""
        snapshot = AgentMetricsSnapshot(
            agent_name="analyst",
            is_healthy=True,
            total_calls=50,
            failed_calls=2,
            avg_latency_ms=200.0,
            error_rate=0.04,
        )

        d = snapshot.to_dict()

        assert d["agent_name"] == "analyst"
        assert d["is_healthy"]
        assert d["error_rate"] == 0.04


class TestObservabilityAgent:
    """Tests for ObservabilityAgent class."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock agent registry."""
        registry = MagicMock()
        registry.list_all.return_value = []
        registry.get_healthy_agents.return_value = []
        return registry

    @pytest.fixture
    def agent(self, mock_registry):
        """Create an observability agent."""
        return ObservabilityAgent(
            registry=mock_registry,
            thresholds=MonitoringThresholds(),
            token_budget=10000,
        )

    def test_agent_creation(self, mock_registry):
        """Test creating an observability agent."""
        agent = create_observability_agent(
            registry=mock_registry,
            token_budget=5000,
        )

        assert agent is not None
        assert agent.role == AgentRole.SUPERVISOR

    def test_agent_has_correct_role(self, agent):
        """Test agent has supervisor role."""
        assert agent.role == AgentRole.SUPERVISOR

    def test_analyze_returns_response(self, agent):
        """Test analyze method returns valid response."""
        response = agent.analyze(
            query="Check system health",
            context={},
        )

        assert response is not None
        assert response.final_answer is not None

    def test_get_active_alerts_empty(self, agent):
        """Test getting alerts when none exist."""
        alerts = agent.get_active_alerts()
        assert len(alerts) == 0

    def test_add_alert(self, agent):
        """Test adding an alert."""
        alert = Alert(
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity=AlertSeverity.WARNING,
            message="Test alert",
        )

        agent._add_alert(alert)

        alerts = agent.get_active_alerts()
        assert len(alerts) == 1
        assert alerts[0].message == "Test alert"

    def test_resolve_alert(self, agent):
        """Test resolving an alert."""
        alert = Alert(
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity=AlertSeverity.WARNING,
            message="Test alert",
        )

        agent._add_alert(alert)
        agent.resolve_alert(alert.alert_id)

        active = agent.get_active_alerts()
        assert len(active) == 0

    def test_get_alerts_by_severity(self, agent):
        """Test filtering alerts by severity."""
        agent._add_alert(
            Alert(
                alert_type=AlertType.HIGH_ERROR_RATE,
                severity=AlertSeverity.WARNING,
                message="Warning alert",
            )
        )
        agent._add_alert(
            Alert(
                alert_type=AlertType.HIGH_LATENCY,
                severity=AlertSeverity.ERROR,
                message="Error alert",
            )
        )
        agent._add_alert(
            Alert(
                alert_type=AlertType.AGENT_UNHEALTHY,
                severity=AlertSeverity.ERROR,
                message="Another error",
            )
        )

        warnings = agent.get_alerts_by_severity(AlertSeverity.WARNING)
        errors = agent.get_alerts_by_severity(AlertSeverity.ERROR)

        assert len(warnings) == 1
        assert len(errors) == 2


class TestObservabilityAgentCallbacks:
    """Tests for alert callbacks."""

    def test_on_alert_callback(self):
        """Test alert callback is called."""
        mock_registry = MagicMock()
        mock_registry.list_all.return_value = []
        mock_registry.get_healthy_agents.return_value = []

        alerts_received = []

        def on_alert(alert):
            alerts_received.append(alert)

        agent = ObservabilityAgent(
            registry=mock_registry,
            on_alert=on_alert,
        )

        alert = Alert(
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity=AlertSeverity.ERROR,
            message="Test alert",
        )

        agent._add_alert(alert)

        assert len(alerts_received) == 1
        assert alerts_received[0].message == "Test alert"


class TestObservabilityAgentMonitoring:
    """Tests for background monitoring."""

    def test_monitoring_not_started_by_default(self):
        """Test monitoring is not started by default."""
        mock_registry = MagicMock()
        mock_registry.list_all.return_value = []

        agent = ObservabilityAgent(registry=mock_registry)

        # Check that monitoring thread is not active
        assert agent._monitor_thread is None or not agent._monitor_thread.is_alive()

    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        mock_registry = MagicMock()
        mock_registry.list_all.return_value = []
        mock_registry.get_healthy_agents.return_value = []

        agent = ObservabilityAgent(
            registry=mock_registry,
            thresholds=MonitoringThresholds(health_check_interval=0.1),
        )

        agent.start_monitoring()
        assert agent._monitor_thread is not None

        time.sleep(0.2)  # Let it run one cycle

        agent.stop_monitoring()
        # After stopping, the thread should be dead or None
        assert agent._monitor_thread is None or not agent._monitor_thread.is_alive()


class TestAlertDeduplication:
    """Tests for alert deduplication."""

    def test_different_agents_get_separate_alerts(self):
        """Test different agents can have same alert type."""
        mock_registry = MagicMock()
        mock_registry.list_all.return_value = []

        agent = ObservabilityAgent(registry=mock_registry)

        alert1 = Alert(
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity=AlertSeverity.WARNING,
            message="Error rate high",
            agent_name="agent_a",
        )

        alert2 = Alert(
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity=AlertSeverity.WARNING,
            message="Error rate high",
            agent_name="agent_b",
        )

        agent._add_alert(alert1)
        agent._add_alert(alert2)

        # Should have two alerts for different agents
        alerts = agent.get_active_alerts()
        assert len(alerts) == 2
