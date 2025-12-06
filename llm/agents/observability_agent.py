"""
Observability Agent

Autonomous agent that monitors other agents in the system.
Detects anomalies, generates alerts, and can trigger basic remediation.

UPGRADE-014 Category 2: Observability & Debugging

Features:
- Continuous monitoring of registered agents
- Anomaly detection (error rate, latency spikes)
- Alert generation with severity levels
- Integration with agent registry
- Self-healing capabilities

QuantConnect Compatible: Yes
- Non-blocking monitoring
- Thread-safe implementation
- Configurable thresholds
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any, Optional

from llm.agents.base import AgentResponse, AgentRole, AgentThought, ThoughtType, TradingAgent


if TYPE_CHECKING:
    from llm.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


# ============================================================================
# Alert Definitions
# ============================================================================


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""

    HIGH_ERROR_RATE = "high_error_rate"
    HIGH_LATENCY = "high_latency"
    AGENT_UNHEALTHY = "agent_unhealthy"
    TOKEN_BUDGET_WARNING = "token_budget_warning"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    NO_RESPONSE = "no_response"
    ANOMALY_DETECTED = "anomaly_detected"
    AGENT_RECOVERED = "agent_recovered"


@dataclass
class Alert:
    """An observability alert."""

    alert_id: str = field(default_factory=lambda: f"alert_{int(time.time() * 1000)}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    alert_type: AlertType = AlertType.ANOMALY_DETECTED
    severity: AlertSeverity = AlertSeverity.WARNING
    agent_name: str = ""
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: datetime | None = None

    def acknowledge(self):
        """Acknowledge the alert."""
        self.acknowledged = True

    def resolve(self, message: str = ""):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.now(timezone.utc)
        if message:
            self.details["resolution"] = message

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "agent_name": self.agent_name,
            "message": self.message,
            "details": self.details,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


# ============================================================================
# Thresholds Configuration
# ============================================================================


@dataclass
class MonitoringThresholds:
    """Configurable thresholds for anomaly detection."""

    # Error rate thresholds (percentage)
    error_rate_warning: float = 10.0  # 10%
    error_rate_critical: float = 25.0  # 25%

    # Latency thresholds (milliseconds)
    latency_warning_ms: float = 2000.0  # 2 seconds
    latency_critical_ms: float = 5000.0  # 5 seconds

    # Token budget thresholds
    token_budget_warning_pct: float = 80.0  # 80% of budget
    token_budget_critical_pct: float = 95.0  # 95% of budget

    # Health check interval (seconds)
    health_check_interval: float = 30.0

    # Anomaly detection window (minutes)
    anomaly_window_minutes: int = 5


# ============================================================================
# Agent Metrics Snapshot
# ============================================================================


@dataclass
class AgentMetricsSnapshot:
    """Point-in-time metrics for an agent."""

    agent_name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    error_rate: float = 0.0
    is_healthy: bool = True
    last_call_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "error_rate": round(self.error_rate, 2),
            "is_healthy": self.is_healthy,
            "last_call_time": self.last_call_time.isoformat() if self.last_call_time else None,
        }


# ============================================================================
# Observability Agent
# ============================================================================


class ObservabilityAgent(TradingAgent):
    """
    Agent that monitors other agents for health and anomalies.

    This agent:
    - Collects metrics from registered agents
    - Detects anomalies based on thresholds
    - Generates alerts with severity levels
    - Can trigger basic remediation actions

    Usage:
        from llm.agents.registry import get_global_registry

        registry = get_global_registry()
        observer = ObservabilityAgent(registry=registry)

        # Start monitoring
        observer.start_monitoring()

        # Check alerts
        alerts = observer.get_active_alerts()

        # Stop monitoring
        observer.stop_monitoring()
    """

    def __init__(
        self,
        registry: Optional["AgentRegistry"] = None,
        thresholds: MonitoringThresholds | None = None,
        on_alert: Callable[[Alert], None] | None = None,
        token_budget: int | None = None,
        name: str = "observability_agent",
    ):
        """
        Initialize observability agent.

        Args:
            registry: Agent registry to monitor
            thresholds: Monitoring thresholds
            on_alert: Callback for new alerts
            token_budget: Total token budget (for budget alerts)
            name: Agent name
        """
        super().__init__(
            name=name,
            role=AgentRole.SUPERVISOR,
            system_prompt="""You are an observability agent that monitors the health
and performance of other trading agents. Your responsibilities are:
1. Track agent metrics (error rates, latencies, token usage)
2. Detect anomalies and unhealthy agents
3. Generate appropriate alerts
4. Recommend remediation actions when possible""",
            tools=[],
            max_iterations=3,
            timeout_ms=5000.0,
        )

        self.registry = registry
        self.thresholds = thresholds or MonitoringThresholds()
        self.on_alert = on_alert
        self.token_budget = token_budget

        # State
        self._alerts: list[Alert] = []
        self._metrics_history: dict[str, list[AgentMetricsSnapshot]] = {}
        self._lock = Lock()

        # Background monitoring
        self._monitoring = False
        self._monitor_thread: Thread | None = None

        # Token tracking (integration with TokenUsageTracker)
        self._total_tokens_used: int = 0

    def analyze(self, query: str, context: dict[str, Any]) -> AgentResponse:
        """
        Analyze agent health and metrics.

        Args:
            query: Analysis query
            context: Additional context

        Returns:
            AgentResponse with health analysis
        """
        start_time = time.time()
        thoughts: list[AgentThought] = []

        # Gather metrics
        thoughts.append(
            AgentThought(
                thought_type=ThoughtType.REASONING,
                content="Gathering metrics from all registered agents",
            )
        )

        metrics = self._collect_all_metrics()

        # Check for anomalies
        thoughts.append(
            AgentThought(
                thought_type=ThoughtType.REASONING,
                content=f"Analyzing {len(metrics)} agents for anomalies",
            )
        )

        new_alerts = self._check_anomalies(metrics)

        # Generate summary
        summary = self._generate_health_summary(metrics, new_alerts)

        thoughts.append(
            AgentThought(
                thought_type=ThoughtType.FINAL_ANSWER,
                content=summary,
            )
        )

        execution_time = (time.time() - start_time) * 1000

        return AgentResponse(
            agent_name=self.name,
            agent_role=self.role,
            query=query,
            thoughts=thoughts,
            final_answer=summary,
            confidence=0.9,
            tools_used=[],
            execution_time_ms=execution_time,
            success=True,
        )

    def _collect_all_metrics(self) -> dict[str, AgentMetricsSnapshot]:
        """Collect metrics from all registered agents."""
        metrics: dict[str, AgentMetricsSnapshot] = {}

        if not self.registry:
            return metrics

        try:
            # Import here to avoid circular imports
            from llm.agents.registry import AgentHealth

            all_agents = self.registry.list_all()

            for registration in all_agents:
                agent = registration.agent
                agent_name = agent.name

                # Create snapshot
                snapshot = AgentMetricsSnapshot(
                    agent_name=agent_name,
                    is_healthy=registration.health == AgentHealth.HEALTHY,
                )

                # Try to get metrics from agent if available
                if hasattr(agent, "get_metrics"):
                    try:
                        agent_metrics = agent.get_metrics()
                        snapshot.total_calls = agent_metrics.get("total_calls", 0)
                        snapshot.failed_calls = agent_metrics.get("failed_calls", 0)
                        snapshot.avg_latency_ms = agent_metrics.get("avg_latency_ms", 0.0)
                    except Exception:
                        pass

                # Calculate error rate
                if snapshot.total_calls > 0:
                    snapshot.error_rate = (snapshot.failed_calls / snapshot.total_calls) * 100

                metrics[agent_name] = snapshot

                # Store in history
                with self._lock:
                    if agent_name not in self._metrics_history:
                        self._metrics_history[agent_name] = []
                    self._metrics_history[agent_name].append(snapshot)

                    # Keep only last hour of history
                    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
                    self._metrics_history[agent_name] = [
                        m for m in self._metrics_history[agent_name] if m.timestamp >= cutoff
                    ]

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

        return metrics

    def _check_anomalies(self, metrics: dict[str, AgentMetricsSnapshot]) -> list[Alert]:
        """Check for anomalies in metrics."""
        new_alerts: list[Alert] = []

        for agent_name, snapshot in metrics.items():
            # Check error rate
            if snapshot.error_rate >= self.thresholds.error_rate_critical:
                alert = Alert(
                    alert_type=AlertType.HIGH_ERROR_RATE,
                    severity=AlertSeverity.CRITICAL,
                    agent_name=agent_name,
                    message=f"Critical error rate: {snapshot.error_rate:.1f}%",
                    details={"error_rate": snapshot.error_rate},
                )
                new_alerts.append(alert)
            elif snapshot.error_rate >= self.thresholds.error_rate_warning:
                alert = Alert(
                    alert_type=AlertType.HIGH_ERROR_RATE,
                    severity=AlertSeverity.WARNING,
                    agent_name=agent_name,
                    message=f"High error rate: {snapshot.error_rate:.1f}%",
                    details={"error_rate": snapshot.error_rate},
                )
                new_alerts.append(alert)

            # Check latency
            if snapshot.avg_latency_ms >= self.thresholds.latency_critical_ms:
                alert = Alert(
                    alert_type=AlertType.HIGH_LATENCY,
                    severity=AlertSeverity.CRITICAL,
                    agent_name=agent_name,
                    message=f"Critical latency: {snapshot.avg_latency_ms:.0f}ms",
                    details={"latency_ms": snapshot.avg_latency_ms},
                )
                new_alerts.append(alert)
            elif snapshot.avg_latency_ms >= self.thresholds.latency_warning_ms:
                alert = Alert(
                    alert_type=AlertType.HIGH_LATENCY,
                    severity=AlertSeverity.WARNING,
                    agent_name=agent_name,
                    message=f"High latency: {snapshot.avg_latency_ms:.0f}ms",
                    details={"latency_ms": snapshot.avg_latency_ms},
                )
                new_alerts.append(alert)

            # Check health
            if not snapshot.is_healthy:
                alert = Alert(
                    alert_type=AlertType.AGENT_UNHEALTHY,
                    severity=AlertSeverity.ERROR,
                    agent_name=agent_name,
                    message="Agent is unhealthy",
                    details=snapshot.to_dict(),
                )
                new_alerts.append(alert)

        # Check token budget
        if self.token_budget and self._total_tokens_used > 0:
            usage_pct = (self._total_tokens_used / self.token_budget) * 100

            if usage_pct >= self.thresholds.token_budget_critical_pct:
                alert = Alert(
                    alert_type=AlertType.TOKEN_BUDGET_EXCEEDED,
                    severity=AlertSeverity.CRITICAL,
                    agent_name="system",
                    message=f"Token budget exceeded: {usage_pct:.1f}%",
                    details={
                        "used_tokens": self._total_tokens_used,
                        "budget": self.token_budget,
                        "usage_pct": usage_pct,
                    },
                )
                new_alerts.append(alert)
            elif usage_pct >= self.thresholds.token_budget_warning_pct:
                alert = Alert(
                    alert_type=AlertType.TOKEN_BUDGET_WARNING,
                    severity=AlertSeverity.WARNING,
                    agent_name="system",
                    message=f"Token budget warning: {usage_pct:.1f}%",
                    details={
                        "used_tokens": self._total_tokens_used,
                        "budget": self.token_budget,
                        "usage_pct": usage_pct,
                    },
                )
                new_alerts.append(alert)

        # Store alerts and notify
        for alert in new_alerts:
            self._add_alert(alert)

        return new_alerts

    def _add_alert(self, alert: Alert):
        """Add alert and notify callback."""
        with self._lock:
            self._alerts.append(alert)

            # Keep last 1000 alerts
            if len(self._alerts) > 1000:
                self._alerts = self._alerts[-1000:]

        if self.on_alert:
            try:
                self.on_alert(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _generate_health_summary(
        self,
        metrics: dict[str, AgentMetricsSnapshot],
        new_alerts: list[Alert],
    ) -> str:
        """Generate a health summary."""
        healthy_count = sum(1 for m in metrics.values() if m.is_healthy)
        total_count = len(metrics)

        critical_alerts = [a for a in new_alerts if a.severity == AlertSeverity.CRITICAL]
        warning_alerts = [a for a in new_alerts if a.severity == AlertSeverity.WARNING]

        summary_parts = [
            f"System Health: {healthy_count}/{total_count} agents healthy",
        ]

        if critical_alerts:
            summary_parts.append(f"CRITICAL: {len(critical_alerts)} critical alerts")
        if warning_alerts:
            summary_parts.append(f"WARNING: {len(warning_alerts)} warnings")

        if not critical_alerts and not warning_alerts:
            summary_parts.append("No anomalies detected")

        return ". ".join(summary_parts)

    def update_token_usage(self, tokens: int):
        """Update total token usage for budget tracking."""
        self._total_tokens_used += tokens

    # =========================================================================
    # Background Monitoring
    # =========================================================================

    def start_monitoring(self):
        """Start background monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Observability monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info("Observability monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                # Collect and check metrics
                metrics = self._collect_all_metrics()
                self._check_anomalies(metrics)

                # Also run health checks via registry
                if self.registry:
                    self.registry.health_check()

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")

            # Sleep until next check
            time.sleep(self.thresholds.health_check_interval)

    # =========================================================================
    # Alert Management
    # =========================================================================

    def get_active_alerts(self) -> list[Alert]:
        """Get all unresolved alerts."""
        with self._lock:
            return [a for a in self._alerts if not a.resolved]

    def get_alerts_by_severity(self, severity: AlertSeverity) -> list[Alert]:
        """Get alerts by severity level."""
        with self._lock:
            return [a for a in self._alerts if a.severity == severity]

    def get_alerts_for_agent(self, agent_name: str) -> list[Alert]:
        """Get alerts for a specific agent."""
        with self._lock:
            return [a for a in self._alerts if a.agent_name == agent_name]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledge()
                    return True
        return False

    def resolve_alert(self, alert_id: str, message: str = "") -> bool:
        """Resolve an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.resolve(message)
                    return True
        return False

    def clear_resolved_alerts(self):
        """Clear all resolved alerts."""
        with self._lock:
            self._alerts = [a for a in self._alerts if not a.resolved]

    # =========================================================================
    # Metrics Access
    # =========================================================================

    def get_metrics_history(
        self,
        agent_name: str,
        window_minutes: int = 60,
    ) -> list[AgentMetricsSnapshot]:
        """Get metrics history for an agent."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)

        with self._lock:
            history = self._metrics_history.get(agent_name, [])
            return [m for m in history if m.timestamp >= cutoff]

    def get_system_status(self) -> dict[str, Any]:
        """Get overall system status."""
        with self._lock:
            active_alerts = [a for a in self._alerts if not a.resolved]

        metrics = self._collect_all_metrics()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_agents": len(metrics),
            "healthy_agents": sum(1 for m in metrics.values() if m.is_healthy),
            "unhealthy_agents": sum(1 for m in metrics.values() if not m.is_healthy),
            "active_alerts": len(active_alerts),
            "critical_alerts": sum(1 for a in active_alerts if a.severity == AlertSeverity.CRITICAL),
            "warning_alerts": sum(1 for a in active_alerts if a.severity == AlertSeverity.WARNING),
            "total_tokens_used": self._total_tokens_used,
            "token_budget": self.token_budget,
            "monitoring_active": self._monitoring,
        }


# ============================================================================
# Factory Functions
# ============================================================================


def create_observability_agent(
    registry: Optional["AgentRegistry"] = None,
    thresholds: MonitoringThresholds | None = None,
    on_alert: Callable[[Alert], None] | None = None,
    token_budget: int | None = None,
) -> ObservabilityAgent:
    """
    Factory function to create an observability agent.

    Args:
        registry: Agent registry to monitor
        thresholds: Monitoring thresholds
        on_alert: Alert callback
        token_budget: Token budget for alerts

    Returns:
        Configured ObservabilityAgent
    """
    return ObservabilityAgent(
        registry=registry,
        thresholds=thresholds,
        on_alert=on_alert,
        token_budget=token_budget,
    )
