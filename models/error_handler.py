"""
Enhanced Error Handling Infrastructure for Trading Bot

Provides:
- Error classification and categorization
- Centralized error management
- Graceful degradation support
- Integration with logging and circuit breaker
- Error history and statistics

UPGRADE-012: Error Handling (December 2025)
"""

from __future__ import annotations

import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


# ==============================================================================
# Error Classification
# ==============================================================================


class ErrorSeverity(Enum):
    """Error severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    TRANSIENT = "transient"  # Network timeouts, temporary failures
    PERMANENT = "permanent"  # Invalid data, logic errors
    RECOVERABLE = "recoverable"  # Can retry with different params
    CRITICAL = "critical"  # Requires immediate attention
    DEGRADED = "degraded"  # Non-critical service failure
    DATA = "data"  # Data quality or validation errors
    EXECUTION = "execution"  # Order execution failures
    SYSTEM = "system"  # System resource issues


class ServiceStatus(Enum):
    """Service health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class TradingError:
    """Structured trading error."""

    error_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    category: ErrorCategory = ErrorCategory.TRANSIENT
    severity: ErrorSeverity = ErrorSeverity.ERROR
    message: str = ""
    exception_type: str = ""
    exception_message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    recoverable: bool = True
    retry_count: int = 0
    max_retries: int = 3
    component: str = ""
    operation: str = ""

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        category: ErrorCategory = ErrorCategory.TRANSIENT,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: dict[str, Any] | None = None,
        component: str = "",
        operation: str = "",
    ) -> TradingError:
        """Create TradingError from an exception."""
        return cls(
            category=category,
            severity=severity,
            message=str(exception),
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            stack_trace=traceback.format_exc(),
            context=context or {},
            recoverable=category
            in (
                ErrorCategory.TRANSIENT,
                ErrorCategory.RECOVERABLE,
            ),
            component=component,
            operation=operation,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "stack_trace": self.stack_trace,
            "recoverable": self.recoverable,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "component": self.component,
            "operation": self.operation,
        }


@dataclass
class ServiceHealth:
    """Health status of a service."""

    name: str
    status: ServiceStatus = ServiceStatus.HEALTHY
    last_error: TradingError | None = None
    error_count: int = 0
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    degraded_since: datetime | None = None
    recovery_attempts: int = 0

    def is_available(self) -> bool:
        """Check if service is available (healthy or degraded)."""
        return self.status in (ServiceStatus.HEALTHY, ServiceStatus.DEGRADED)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "error_count": self.error_count,
            "last_check": self.last_check.isoformat(),
            "degraded_since": self.degraded_since.isoformat() if self.degraded_since else None,
            "recovery_attempts": self.recovery_attempts,
        }


@dataclass
class ErrorAggregation:
    """
    Aggregated error data for alerting (UPGRADE-012 expansion).

    Groups similar errors within a time window to reduce alert noise.
    """

    key: str  # Aggregation key (e.g., "category:component:operation")
    category: ErrorCategory
    severity: ErrorSeverity
    count: int = 0
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sample_errors: list[TradingError] = field(default_factory=list)
    max_samples: int = 5
    component: str = ""
    operation: str = ""
    alerted: bool = False

    def add_error(self, error: TradingError) -> None:
        """Add an error to the aggregation."""
        self.count += 1
        self.last_seen = datetime.now(timezone.utc)
        if len(self.sample_errors) < self.max_samples:
            self.sample_errors.append(error)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "category": self.category.value,
            "severity": self.severity.value,
            "count": self.count,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "component": self.component,
            "operation": self.operation,
            "duration_seconds": (self.last_seen - self.first_seen).total_seconds(),
        }


@dataclass
class AlertTrigger:
    """Alert trigger configuration and state (UPGRADE-012 expansion)."""

    name: str
    condition: str  # "spike", "threshold", "service_degraded"
    threshold: float = 0.0
    window_minutes: int = 5
    enabled: bool = True
    last_triggered: datetime | None = None
    trigger_count: int = 0
    cooldown_minutes: int = 5

    def can_trigger(self) -> bool:
        """Check if alert can trigger (respects cooldown)."""
        if not self.enabled:
            return False
        if self.last_triggered is None:
            return True
        cooldown = timedelta(minutes=self.cooldown_minutes)
        return datetime.now(timezone.utc) - self.last_triggered > cooldown

    def trigger(self) -> None:
        """Mark alert as triggered."""
        self.last_triggered = datetime.now(timezone.utc)
        self.trigger_count += 1


# ==============================================================================
# Error Handler
# ==============================================================================


class ErrorHandler:
    """
    Centralized error handler for trading operations.

    Features:
    - Error classification and logging
    - Error history and statistics
    - Service health tracking
    - Graceful degradation support
    - Circuit breaker integration
    - Event listeners for error notifications

    Example:
        >>> handler = ErrorHandler(logger=structured_logger)
        >>> try:
        ...     execute_order(order)
        ... except Exception as e:
        ...     error = handler.handle_error(
        ...         e,
        ...         category=ErrorCategory.EXECUTION,
        ...         context={"order_id": order.id}
        ...     )
        ...     if error.recoverable:
        ...         # Retry logic
    """

    def __init__(
        self,
        logger: Any | None = None,
        circuit_breaker: Any | None = None,
        max_history: int = 1000,
        aggregation_window_seconds: int = 60,
        spike_threshold: int = 10,
    ):
        """
        Initialize error handler.

        Args:
            logger: StructuredLogger instance for logging errors
            circuit_breaker: TradingCircuitBreaker for safety integration
            max_history: Maximum number of errors to keep in history
            aggregation_window_seconds: Window for aggregating similar errors
            spike_threshold: Error count threshold for spike detection
        """
        self.logger = logger
        self.circuit_breaker = circuit_breaker
        self.max_history = max_history
        self.aggregation_window_seconds = aggregation_window_seconds
        self.spike_threshold = spike_threshold

        self._error_history: list[TradingError] = []
        self._services: dict[str, ServiceHealth] = {}
        self._error_counts: dict[ErrorCategory, int] = dict.fromkeys(ErrorCategory, 0)
        self._listeners: list[Callable[[TradingError], None]] = []
        self._lock = threading.Lock()

        # Aggregation and alerting (UPGRADE-012 expansion)
        self._aggregations: dict[str, ErrorAggregation] = {}
        self._alert_triggers: dict[str, AlertTrigger] = {}
        self._alert_listeners: list[Callable[[str, dict[str, Any]], None]] = []
        self._setup_default_alerts()

    # ==========================================================================
    # Error Handling
    # ==========================================================================

    def handle_error(
        self,
        exception: Exception,
        category: ErrorCategory | None = None,
        severity: ErrorSeverity | None = None,
        context: dict[str, Any] | None = None,
        component: str = "",
        operation: str = "",
    ) -> TradingError:
        """
        Handle and classify an error.

        Args:
            exception: The exception that occurred
            category: Error category (auto-classified if not provided)
            severity: Error severity (auto-classified if not provided)
            context: Additional context information
            component: Component where error occurred
            operation: Operation that was being performed

        Returns:
            TradingError with full classification
        """
        # Auto-classify if not provided
        if category is None:
            category = self._classify_category(exception)
        if severity is None:
            severity = self._classify_severity(exception, category)

        # Create trading error
        error = TradingError.from_exception(
            exception,
            category=category,
            severity=severity,
            context=context,
            component=component,
            operation=operation,
        )

        with self._lock:
            # Add to history
            self._error_history.append(error)
            if len(self._error_history) > self.max_history:
                self._error_history.pop(0)

            # Update counts
            self._error_counts[category] += 1

        # Log the error
        self._log_error(error)

        # Check circuit breaker
        self._check_circuit_breaker(error)

        # Notify listeners
        self._notify_listeners(error)

        return error

    def _classify_category(self, exception: Exception) -> ErrorCategory:
        """Auto-classify error category based on exception type."""
        exception_name = type(exception).__name__.lower()
        exception_msg = str(exception).lower()

        # Network/timeout errors
        if any(
            term in exception_name or term in exception_msg for term in ["timeout", "connection", "network", "socket"]
        ):
            return ErrorCategory.TRANSIENT

        # Data validation errors
        if any(
            term in exception_name or term in exception_msg for term in ["validation", "invalid", "parse", "format"]
        ):
            return ErrorCategory.DATA

        # Value errors might be recoverable
        if "value" in exception_name:
            return ErrorCategory.RECOVERABLE

        # Type errors are usually permanent
        if "type" in exception_name:
            return ErrorCategory.PERMANENT

        # Memory/resource errors
        if any(term in exception_name or term in exception_msg for term in ["memory", "resource", "overflow"]):
            return ErrorCategory.SYSTEM

        # Default to transient (retry-able)
        return ErrorCategory.TRANSIENT

    def _classify_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Auto-classify error severity."""
        if category == ErrorCategory.CRITICAL:
            return ErrorSeverity.CRITICAL
        if category == ErrorCategory.PERMANENT:
            return ErrorSeverity.ERROR
        if category == ErrorCategory.TRANSIENT:
            return ErrorSeverity.WARNING
        if category == ErrorCategory.DEGRADED:
            return ErrorSeverity.WARNING
        return ErrorSeverity.ERROR

    def _log_error(self, error: TradingError) -> None:
        """Log error to structured logger."""
        if self.logger is None:
            return

        # Map severity to log level
        from observability.logging.structured import LogCategory, LogLevel

        level_map = {
            ErrorSeverity.DEBUG: LogLevel.DEBUG,
            ErrorSeverity.INFO: LogLevel.INFO,
            ErrorSeverity.WARNING: LogLevel.WARNING,
            ErrorSeverity.ERROR: LogLevel.ERROR,
            ErrorSeverity.CRITICAL: LogLevel.CRITICAL,
        }

        self.logger.log(
            category=LogCategory.ERROR,
            level=level_map.get(error.severity, LogLevel.ERROR),
            event_type="error_handled",
            message=error.message,
            data=error.to_dict(),
            component=error.component,
            operation=error.operation,
        )

    def _check_circuit_breaker(self, error: TradingError) -> None:
        """Check if error should trigger circuit breaker."""
        if self.circuit_breaker is None:
            return

        # Critical errors should halt trading
        if error.severity == ErrorSeverity.CRITICAL:
            self.circuit_breaker.halt_all_trading(reason=f"Critical error: {error.message}")

    # ==========================================================================
    # Service Health / Graceful Degradation
    # ==========================================================================

    def register_service(self, name: str) -> None:
        """Register a service for health tracking."""
        with self._lock:
            if name not in self._services:
                self._services[name] = ServiceHealth(name=name)

    def mark_service_healthy(self, name: str) -> None:
        """Mark a service as healthy."""
        with self._lock:
            if name in self._services:
                service = self._services[name]
                service.status = ServiceStatus.HEALTHY
                service.degraded_since = None
                service.last_check = datetime.now(timezone.utc)

    def mark_service_degraded(self, name: str, error: TradingError | None = None) -> None:
        """Mark a service as degraded."""
        with self._lock:
            if name not in self._services:
                self._services[name] = ServiceHealth(name=name)

            service = self._services[name]
            service.status = ServiceStatus.DEGRADED
            service.last_error = error
            service.error_count += 1
            service.last_check = datetime.now(timezone.utc)
            if service.degraded_since is None:
                service.degraded_since = datetime.now(timezone.utc)

    def mark_service_failed(self, name: str, error: TradingError | None = None) -> None:
        """Mark a service as failed."""
        with self._lock:
            if name not in self._services:
                self._services[name] = ServiceHealth(name=name)

            service = self._services[name]
            service.status = ServiceStatus.FAILED
            service.last_error = error
            service.error_count += 1
            service.last_check = datetime.now(timezone.utc)

    def is_service_available(self, name: str) -> bool:
        """Check if a service is available."""
        with self._lock:
            if name not in self._services:
                return True  # Unknown services assumed healthy
            return self._services[name].is_available()

    def get_service_health(self, name: str) -> ServiceHealth | None:
        """Get health status of a service."""
        with self._lock:
            return self._services.get(name)

    def get_all_service_health(self) -> dict[str, ServiceHealth]:
        """Get health status of all services."""
        with self._lock:
            return dict(self._services)

    def get_degraded_services(self) -> list[str]:
        """Get list of degraded services."""
        with self._lock:
            return [
                name
                for name, health in self._services.items()
                if health.status in (ServiceStatus.DEGRADED, ServiceStatus.FAILED)
            ]

    # ==========================================================================
    # Error History & Statistics
    # ==========================================================================

    def get_recent_errors(
        self,
        limit: int = 10,
        category: ErrorCategory | None = None,
        severity: ErrorSeverity | None = None,
    ) -> list[TradingError]:
        """Get recent errors, optionally filtered."""
        with self._lock:
            errors = self._error_history[-limit * 2 :]  # Get more than needed

            if category:
                errors = [e for e in errors if e.category == category]
            if severity:
                errors = [e for e in errors if e.severity == severity]

            return errors[-limit:]

    def get_error_counts(self) -> dict[str, int]:
        """Get error counts by category."""
        with self._lock:
            return {cat.value: count for cat, count in self._error_counts.items()}

    def get_error_rate(self, window_minutes: int = 5) -> float:
        """Get error rate (errors per minute) in time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        with self._lock:
            recent = [e for e in self._error_history if e.timestamp > cutoff]
        return len(recent) / window_minutes if window_minutes > 0 else 0

    def clear_history(self) -> None:
        """Clear error history."""
        with self._lock:
            self._error_history.clear()
            self._error_counts = dict.fromkeys(ErrorCategory, 0)

    # ==========================================================================
    # Event Listeners
    # ==========================================================================

    def add_listener(self, callback: Callable[[TradingError], None]) -> None:
        """Add error listener callback."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[TradingError], None]) -> None:
        """Remove error listener callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_listeners(self, error: TradingError) -> None:
        """Notify all listeners of an error."""
        for listener in self._listeners:
            try:
                listener(error)
            except Exception:
                pass  # Don't let listener errors propagate

    # ==========================================================================
    # Error Aggregation (UPGRADE-012 expansion)
    # ==========================================================================

    def _setup_default_alerts(self) -> None:
        """Set up default alert triggers."""
        self._alert_triggers = {
            "error_spike": AlertTrigger(
                name="error_spike",
                condition="spike",
                threshold=float(self.spike_threshold),
                window_minutes=5,
                cooldown_minutes=5,
            ),
            "critical_error": AlertTrigger(
                name="critical_error",
                condition="severity",
                threshold=0,  # Any critical error
                window_minutes=1,
                cooldown_minutes=1,
            ),
            "service_degraded": AlertTrigger(
                name="service_degraded",
                condition="service_degraded",
                threshold=0,
                window_minutes=1,
                cooldown_minutes=5,
            ),
        }

    def _get_aggregation_key(self, error: TradingError) -> str:
        """Generate aggregation key for error."""
        return f"{error.category.value}:{error.component}:{error.operation}"

    def aggregate_error(self, error: TradingError) -> ErrorAggregation:
        """
        Aggregate an error with similar errors.

        Args:
            error: The error to aggregate

        Returns:
            ErrorAggregation containing the grouped error data
        """
        key = self._get_aggregation_key(error)
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.aggregation_window_seconds)

        with self._lock:
            # Clean up old aggregations
            expired_keys = [k for k, agg in self._aggregations.items() if agg.last_seen < cutoff]
            for k in expired_keys:
                del self._aggregations[k]

            # Get or create aggregation
            if key not in self._aggregations:
                self._aggregations[key] = ErrorAggregation(
                    key=key,
                    category=error.category,
                    severity=error.severity,
                    component=error.component,
                    operation=error.operation,
                )

            aggregation = self._aggregations[key]
            aggregation.add_error(error)

            # Check for spike
            self._check_spike_alert(aggregation)

            return aggregation

    def _check_spike_alert(self, aggregation: ErrorAggregation) -> None:
        """Check if aggregation triggers a spike alert."""
        trigger = self._alert_triggers.get("error_spike")
        if trigger and trigger.can_trigger():
            if aggregation.count >= int(trigger.threshold):
                self._fire_alert(
                    "error_spike",
                    {
                        "aggregation": aggregation.to_dict(),
                        "threshold": trigger.threshold,
                        "message": (
                            f"Error spike detected: {aggregation.count} errors "
                            f"in {self.aggregation_window_seconds}s for "
                            f"{aggregation.key}"
                        ),
                    },
                )
                trigger.trigger()
                aggregation.alerted = True

    def get_aggregations(self, active_only: bool = True) -> dict[str, ErrorAggregation]:
        """
        Get current error aggregations.

        Args:
            active_only: Only return aggregations within the window

        Returns:
            Dictionary of aggregation key to ErrorAggregation
        """
        with self._lock:
            if not active_only:
                return dict(self._aggregations)

            cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.aggregation_window_seconds)
            return {k: agg for k, agg in self._aggregations.items() if agg.last_seen >= cutoff}

    def get_spike_candidates(self, threshold: int | None = None) -> list[ErrorAggregation]:
        """
        Get aggregations that are approaching or at spike threshold.

        Args:
            threshold: Custom threshold (defaults to spike_threshold)

        Returns:
            List of aggregations at or above threshold
        """
        threshold = threshold or self.spike_threshold
        with self._lock:
            return [agg for agg in self._aggregations.values() if agg.count >= threshold]

    def check_error_spike(self, window_minutes: int = 5, threshold: int | None = None) -> bool:
        """
        Check if there's an error spike in the time window.

        Args:
            window_minutes: Time window to check
            threshold: Error count threshold

        Returns:
            True if error count exceeds threshold
        """
        threshold = threshold or self.spike_threshold
        rate = self.get_error_rate(window_minutes)
        return rate * window_minutes >= threshold

    # ==========================================================================
    # Alert Management (UPGRADE-012 expansion)
    # ==========================================================================

    def add_alert_listener(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """
        Add alert listener callback.

        Args:
            callback: Function called with (alert_name, alert_data)
        """
        self._alert_listeners.append(callback)

    def remove_alert_listener(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """Remove alert listener callback."""
        if callback in self._alert_listeners:
            self._alert_listeners.remove(callback)

    def _fire_alert(self, name: str, data: dict[str, Any]) -> None:
        """Fire an alert to all listeners."""
        for listener in self._alert_listeners:
            try:
                listener(name, data)
            except Exception:
                pass  # Don't let listener errors propagate

        # Log alert
        if self.logger:
            try:
                from observability.logging.structured import LogCategory, LogLevel

                self.logger.log(
                    category=LogCategory.SYSTEM,
                    level=LogLevel.WARNING,
                    event_type="alert_fired",
                    message=data.get("message", f"Alert: {name}"),
                    data={"alert_name": name, **data},
                )
            except Exception:
                pass

    def configure_alert(
        self,
        name: str,
        enabled: bool | None = None,
        threshold: float | None = None,
        window_minutes: int | None = None,
        cooldown_minutes: int | None = None,
    ) -> None:
        """
        Configure an alert trigger.

        Args:
            name: Alert name
            enabled: Whether alert is enabled
            threshold: Alert threshold
            window_minutes: Time window for alert
            cooldown_minutes: Cooldown between alerts
        """
        with self._lock:
            if name not in self._alert_triggers:
                self._alert_triggers[name] = AlertTrigger(
                    name=name,
                    condition="custom",
                )

            trigger = self._alert_triggers[name]
            if enabled is not None:
                trigger.enabled = enabled
            if threshold is not None:
                trigger.threshold = threshold
            if window_minutes is not None:
                trigger.window_minutes = window_minutes
            if cooldown_minutes is not None:
                trigger.cooldown_minutes = cooldown_minutes

    def get_alert_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all alert triggers."""
        with self._lock:
            return {
                name: {
                    "name": trigger.name,
                    "condition": trigger.condition,
                    "enabled": trigger.enabled,
                    "threshold": trigger.threshold,
                    "trigger_count": trigger.trigger_count,
                    "last_triggered": (trigger.last_triggered.isoformat() if trigger.last_triggered else None),
                }
                for name, trigger in self._alert_triggers.items()
            }

    def trigger_manual_alert(self, name: str, message: str, data: dict[str, Any] | None = None) -> bool:
        """
        Manually trigger an alert.

        Args:
            name: Alert name
            message: Alert message
            data: Additional alert data

        Returns:
            True if alert was fired, False if on cooldown
        """
        with self._lock:
            if name not in self._alert_triggers:
                self._alert_triggers[name] = AlertTrigger(
                    name=name,
                    condition="manual",
                )

            trigger = self._alert_triggers[name]
            if not trigger.can_trigger():
                return False

            trigger.trigger()

        alert_data = {"message": message, **(data or {})}
        self._fire_alert(name, alert_data)
        return True


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_error_handler(
    logger: Any | None = None,
    circuit_breaker: Any | None = None,
    max_history: int = 1000,
    aggregation_window_seconds: int = 60,
    spike_threshold: int = 10,
) -> ErrorHandler:
    """
    Factory function to create an error handler.

    Args:
        logger: StructuredLogger instance
        circuit_breaker: TradingCircuitBreaker instance
        max_history: Maximum error history size
        aggregation_window_seconds: Window for aggregating similar errors
        spike_threshold: Error count threshold for spike detection

    Returns:
        Configured ErrorHandler instance
    """
    return ErrorHandler(
        logger=logger,
        circuit_breaker=circuit_breaker,
        max_history=max_history,
        aggregation_window_seconds=aggregation_window_seconds,
        spike_threshold=spike_threshold,
    )
