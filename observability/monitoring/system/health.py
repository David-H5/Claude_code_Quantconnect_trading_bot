"""
System Monitor for Trading Bot

Provides:
- Health checks for critical services
- Performance monitoring
- Automatic alerting on issues
- Status dashboard data
- Resource usage tracking

UPGRADE-013: Monitoring & Alerting (December 2025)
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


# ==============================================================================
# Data Types
# ==============================================================================


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceCheck:
    """Service health check configuration."""

    name: str
    check_fn: Callable[[], bool]
    interval_seconds: int = 60
    timeout_seconds: int = 10
    failure_threshold: int = 3
    recovery_threshold: int = 2


@dataclass
class ServiceState:
    """Current state of a monitored service."""

    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: datetime | None = None
    last_success: datetime | None = None
    last_failure: datetime | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_checks: int = 0
    total_failures: int = 0
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "total_checks": self.total_checks,
            "total_failures": self.total_failures,
            "error_message": self.error_message,
            "uptime_pct": (
                100.0 * (self.total_checks - self.total_failures) / self.total_checks
                if self.total_checks > 0
                else 100.0
            ),
        }


@dataclass
class SystemMetrics:
    """System-wide metrics."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    healthy_services: int = 0
    degraded_services: int = 0
    unhealthy_services: int = 0
    total_services: int = 0
    alerts_sent: int = 0
    errors_last_hour: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "healthy_services": self.healthy_services,
            "degraded_services": self.degraded_services,
            "unhealthy_services": self.unhealthy_services,
            "total_services": self.total_services,
            "overall_health": (
                "healthy"
                if self.unhealthy_services == 0 and self.degraded_services == 0
                else "degraded"
                if self.unhealthy_services == 0
                else "unhealthy"
            ),
            "alerts_sent": self.alerts_sent,
            "errors_last_hour": self.errors_last_hour,
        }


# ==============================================================================
# System Monitor
# ==============================================================================


class SystemMonitor:
    """
    Monitor system health and performance.

    Features:
    - Register services for health monitoring
    - Automatic periodic checks
    - Alert on status changes
    - Metrics and status dashboard
    - Integration with AlertingService

    Example:
        >>> monitor = SystemMonitor(alerting_service=alerting_service)
        >>> monitor.register_service(
        ...     "broker_api",
        ...     lambda: check_broker_connection(),
        ...     interval_seconds=30,
        ... )
        >>> monitor.start()
        >>> # ... later
        >>> status = monitor.get_status_summary()
    """

    def __init__(
        self,
        alerting_service: Any | None = None,
        check_interval_seconds: int = 60,
    ):
        """
        Initialize system monitor.

        Args:
            alerting_service: AlertingService for alerts
            check_interval_seconds: Default interval between checks
        """
        self.alerting_service = alerting_service
        self.default_interval = check_interval_seconds

        self._services: dict[str, ServiceCheck] = {}
        self._states: dict[str, ServiceState] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._metrics_history: list[SystemMetrics] = []
        self._max_history = 60  # Keep 1 hour of metrics at 1-minute intervals
        self._listeners: list[Callable[[str, HealthStatus, HealthStatus], None]] = []

    def register_service(
        self,
        name: str,
        check_fn: Callable[[], bool],
        interval_seconds: int | None = None,
        timeout_seconds: int = 10,
        failure_threshold: int = 3,
        recovery_threshold: int = 2,
    ) -> None:
        """
        Register a service for health monitoring.

        Args:
            name: Service name
            check_fn: Function that returns True if healthy
            interval_seconds: Check interval (None = default)
            timeout_seconds: Timeout for check
            failure_threshold: Consecutive failures before unhealthy
            recovery_threshold: Consecutive successes before healthy
        """
        with self._lock:
            self._services[name] = ServiceCheck(
                name=name,
                check_fn=check_fn,
                interval_seconds=interval_seconds or self.default_interval,
                timeout_seconds=timeout_seconds,
                failure_threshold=failure_threshold,
                recovery_threshold=recovery_threshold,
            )
            self._states[name] = ServiceState(name=name)

    def unregister_service(self, name: str) -> None:
        """Unregister a service."""
        with self._lock:
            self._services.pop(name, None)
            self._states.pop(name, None)

    def check_service(self, name: str) -> bool:
        """
        Run health check for a specific service.

        Args:
            name: Service name

        Returns:
            True if healthy
        """
        with self._lock:
            if name not in self._services:
                return False

            service = self._services[name]
            state = self._states[name]

        now = datetime.now(timezone.utc)
        old_status = state.status

        try:
            # Run check with timeout
            result = self._run_check_with_timeout(
                service.check_fn,
                service.timeout_seconds,
            )

            state.last_check = now
            state.total_checks += 1

            if result:
                state.last_success = now
                state.consecutive_successes += 1
                state.consecutive_failures = 0
                state.error_message = ""

                # Check for recovery
                if state.consecutive_successes >= service.recovery_threshold:
                    state.status = HealthStatus.HEALTHY

            else:
                state.last_failure = now
                state.consecutive_failures += 1
                state.consecutive_successes = 0
                state.total_failures += 1
                state.error_message = "Check returned False"

                # Check for degraded/unhealthy
                if state.consecutive_failures >= service.failure_threshold:
                    state.status = HealthStatus.UNHEALTHY
                elif state.consecutive_failures >= 1:
                    state.status = HealthStatus.DEGRADED

        except Exception as e:
            state.last_check = now
            state.last_failure = now
            state.total_checks += 1
            state.total_failures += 1
            state.consecutive_failures += 1
            state.consecutive_successes = 0
            state.error_message = str(e)

            if state.consecutive_failures >= service.failure_threshold:
                state.status = HealthStatus.UNHEALTHY
            else:
                state.status = HealthStatus.DEGRADED

        # Notify on status change
        if old_status != state.status:
            self._on_status_change(name, old_status, state.status)

        return state.status == HealthStatus.HEALTHY

    def _run_check_with_timeout(
        self,
        check_fn: Callable[[], bool],
        timeout: int,
    ) -> bool:
        """Run check function with timeout."""
        result = [False]
        exception = [None]

        def run() -> None:
            try:
                result[0] = check_fn()
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            raise TimeoutError(f"Check timed out after {timeout}s")

        if exception[0]:
            raise exception[0]

        return result[0]

    def _on_status_change(
        self,
        name: str,
        old_status: HealthStatus,
        new_status: HealthStatus,
    ) -> None:
        """Handle service status change."""
        logger.info(f"Service {name} status changed: {old_status.value} -> {new_status.value}")

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(name, old_status, new_status)
            except Exception:
                pass

        # Send alert
        if self.alerting_service:
            from utils.alerting_service import AlertCategory, AlertSeverity

            severity_map = {
                HealthStatus.HEALTHY: AlertSeverity.INFO,
                HealthStatus.DEGRADED: AlertSeverity.WARNING,
                HealthStatus.UNHEALTHY: AlertSeverity.ERROR,
            }

            severity = severity_map.get(new_status, AlertSeverity.WARNING)

            # Only alert on degradation
            if new_status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY):
                self.alerting_service.send_alert(
                    title=f"Service {new_status.value}: {name}",
                    message=f"Service {name} changed from {old_status.value} to {new_status.value}",
                    severity=severity,
                    category=AlertCategory.SERVICE_HEALTH,
                    data={"service": name, "old_status": old_status.value, "new_status": new_status.value},
                )

    def check_all(self) -> dict[str, bool]:
        """
        Run health checks for all services.

        Returns:
            Dictionary of service name to health status
        """
        results = {}
        with self._lock:
            services = list(self._services.keys())

        for name in services:
            results[name] = self.check_service(name)

        return results

    def get_service_status(self, name: str) -> ServiceState | None:
        """Get current status of a service."""
        with self._lock:
            return self._states.get(name)

    def get_all_statuses(self) -> dict[str, ServiceState]:
        """Get status of all services."""
        with self._lock:
            return dict(self._states)

    def get_status_summary(self) -> dict[str, Any]:
        """
        Get overall system status summary.

        Returns:
            Summary dictionary with counts and overall health
        """
        with self._lock:
            states = list(self._states.values())

        healthy = sum(1 for s in states if s.status == HealthStatus.HEALTHY)
        degraded = sum(1 for s in states if s.status == HealthStatus.DEGRADED)
        unhealthy = sum(1 for s in states if s.status == HealthStatus.UNHEALTHY)
        unknown = sum(1 for s in states if s.status == HealthStatus.UNKNOWN)

        overall = "healthy" if unhealthy == 0 and degraded == 0 else "degraded" if unhealthy == 0 else "unhealthy"

        return {
            "overall_status": overall,
            "healthy_count": healthy,
            "degraded_count": degraded,
            "unhealthy_count": unhealthy,
            "unknown_count": unknown,
            "total_services": len(states),
            "services": {s.name: s.to_dict() for s in states},
        }

    def add_status_listener(
        self,
        callback: Callable[[str, HealthStatus, HealthStatus], None],
    ) -> None:
        """Add listener for status changes."""
        self._listeners.append(callback)

    def remove_status_listener(
        self,
        callback: Callable[[str, HealthStatus, HealthStatus], None],
    ) -> None:
        """Remove status change listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("System monitor started")

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("System monitor stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        last_check: dict[str, datetime] = {}

        while self._running:
            now = datetime.now(timezone.utc)

            with self._lock:
                services = dict(self._services)

            for name, service in services.items():
                last = last_check.get(name)
                interval = timedelta(seconds=service.interval_seconds)

                if last is None or now - last >= interval:
                    try:
                        self.check_service(name)
                    except Exception as e:
                        logger.error(f"Error checking {name}: {e}")
                    last_check[name] = now

            # Record metrics
            self._record_metrics()

            # Sleep for a bit
            time.sleep(1)

    def _record_metrics(self) -> None:
        """Record current system metrics."""
        with self._lock:
            states = list(self._states.values())

        metrics = SystemMetrics(
            healthy_services=sum(1 for s in states if s.status == HealthStatus.HEALTHY),
            degraded_services=sum(1 for s in states if s.status == HealthStatus.DEGRADED),
            unhealthy_services=sum(1 for s in states if s.status == HealthStatus.UNHEALTHY),
            total_services=len(states),
        )

        with self._lock:
            self._metrics_history.append(metrics)
            if len(self._metrics_history) > self._max_history:
                self._metrics_history.pop(0)

    def get_metrics_history(self, limit: int = 60) -> list[dict[str, Any]]:
        """Get recent metrics history."""
        with self._lock:
            return [m.to_dict() for m in self._metrics_history[-limit:]]

    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running


# ==============================================================================
# Built-in Health Checks
# ==============================================================================


def create_memory_check(threshold_pct: float = 90.0) -> Callable[[], bool]:
    """Create a memory usage health check."""
    try:
        import psutil

        def check() -> bool:
            return psutil.virtual_memory().percent < threshold_pct

        return check
    except ImportError:
        return lambda: True  # Always healthy if psutil not available


def create_disk_check(
    path: str = "/",
    threshold_pct: float = 90.0,
) -> Callable[[], bool]:
    """Create a disk usage health check."""
    try:
        import psutil

        def check() -> bool:
            return psutil.disk_usage(path).percent < threshold_pct

        return check
    except ImportError:
        return lambda: True


def create_http_check(
    url: str,
    timeout: int = 5,
    expected_status: int = 200,
) -> Callable[[], bool]:
    """Create an HTTP endpoint health check."""
    import urllib.request

    def check() -> bool:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status == expected_status
        except Exception:
            return False

    return check


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_system_monitor(
    alerting_service: Any | None = None,
    check_interval_seconds: int = 60,
) -> SystemMonitor:
    """
    Factory function to create system monitor.

    Args:
        alerting_service: AlertingService for alerts
        check_interval_seconds: Default check interval

    Returns:
        Configured SystemMonitor instance
    """
    return SystemMonitor(
        alerting_service=alerting_service,
        check_interval_seconds=check_interval_seconds,
    )
