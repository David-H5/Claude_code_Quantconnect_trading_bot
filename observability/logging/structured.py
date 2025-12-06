"""
Structured Logging Infrastructure for Trading Bot

Provides JSON-structured logging with:
- Trade execution events
- Risk management alerts
- Strategy decisions
- Audit trail for compliance

UPGRADE-009: Structured Logging (December 2025)
Refactored: Phase 2 - Consolidated Logging Infrastructure
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# Import base types from unified logging infrastructure
from observability.logging.base import (
    AbstractLogger,
)
from observability.logging.base import (
    LogCategory as BaseLogCategory,
)
from observability.logging.base import (
    LogLevel as BaseLogLevel,
)


# ============================================================================
# Enums and Data Classes (backwards-compatible with base types)
# ============================================================================


class LogCategory(Enum):
    """Log event categories for filtering and routing."""

    EXECUTION = "execution"  # Order submissions, fills, cancellations
    RISK = "risk"  # Circuit breaker, position limits, drawdown
    STRATEGY = "strategy"  # Entry/exit signals, strategy decisions
    SYSTEM = "system"  # API requests, WebSocket, resources
    AUDIT = "audit"  # All trading decisions with context
    ERROR = "error"  # Errors and exceptions
    PERFORMANCE = "performance"  # P&L, metrics, analytics

    def to_base(self) -> BaseLogCategory:
        """Convert to base LogCategory."""
        mapping = {
            "execution": BaseLogCategory.EXECUTION,
            "risk": BaseLogCategory.RISK,
            "strategy": BaseLogCategory.STRATEGY,
            "system": BaseLogCategory.SYSTEM,
            "audit": BaseLogCategory.AUDIT,
            "error": BaseLogCategory.ERROR,
            "performance": BaseLogCategory.PERFORMANCE,
        }
        return mapping.get(self.value, BaseLogCategory.SYSTEM)


class LogLevel(Enum):
    """Log severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def to_base(self) -> BaseLogLevel:
        """Convert to base LogLevel."""
        mapping = {
            "debug": BaseLogLevel.DEBUG,
            "info": BaseLogLevel.INFO,
            "warning": BaseLogLevel.WARNING,
            "error": BaseLogLevel.ERROR,
            "critical": BaseLogLevel.CRITICAL,
        }
        return mapping.get(self.value, BaseLogLevel.INFO)


class ExecutionEventType(Enum):
    """Execution-specific event types."""

    ORDER_SUBMITTED = "order_submitted"
    ORDER_PENDING = "order_pending"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIAL_FILL = "order_partial_fill"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_EXPIRED = "order_expired"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_ADJUSTED = "position_adjusted"
    PROFIT_TARGET_HIT = "profit_target_hit"
    STOP_LOSS_HIT = "stop_loss_hit"


class RiskEventType(Enum):
    """Risk-specific event types."""

    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    CIRCUIT_BREAKER_RESET = "circuit_breaker_reset"
    POSITION_LIMIT_WARNING = "position_limit_warning"
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    DAILY_LOSS_WARNING = "daily_loss_warning"
    DAILY_LOSS_EXCEEDED = "daily_loss_exceeded"
    DRAWDOWN_WARNING = "drawdown_warning"
    DRAWDOWN_EXCEEDED = "drawdown_exceeded"
    CONSECUTIVE_LOSS_WARNING = "consecutive_loss_warning"


class StrategyEventType(Enum):
    """Strategy-specific event types."""

    SIGNAL_GENERATED = "signal_generated"
    ENTRY_APPROVED = "entry_approved"
    ENTRY_REJECTED = "entry_rejected"
    EXIT_SIGNAL = "exit_signal"
    STRATEGY_SELECTED = "strategy_selected"
    IV_RANK_UPDATE = "iv_rank_update"
    SENTIMENT_UPDATE = "sentiment_update"


@dataclass
class LogEvent:
    """Structured log event with full context."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    category: LogCategory = LogCategory.SYSTEM
    level: LogLevel = LogLevel.INFO
    event_type: str = ""
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    correlation_id: str | None = None
    duration_ms: float | None = None

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "level": self.level.value,
            "event_type": self.event_type,
            "message": self.message,
        }

        if self.data:
            result["data"] = self.data
        if self.context:
            result["context"] = self.context
        if self.source:
            result["source"] = self.source
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms

        return result


# ============================================================================
# Structured Logger
# ============================================================================


class StructuredLogger(AbstractLogger):
    """
    Structured logger for trading operations.

    Implements AbstractLogger interface from observability.logging.base.

    Features:
    - JSON-formatted log events
    - Category-based filtering
    - Multiple output handlers
    - Context propagation
    - Correlation ID tracking
    - Thread-safe operations

    Example:
        >>> logger = create_structured_logger("trading_bot")
        >>> logger.log_order_submitted("ORD-001", "SPY", quantity=10)
        >>> logger.log_circuit_breaker(True, "Daily loss exceeded 3%")
    """

    def __init__(
        self,
        name: str = "trading_bot",
        min_level: LogLevel = LogLevel.INFO,
        handlers: list[logging.Handler] | None = None,
    ):
        """Initialize structured logger.

        Args:
            name: Logger name
            min_level: Minimum log level to output
            handlers: List of logging handlers
        """
        self.name = name
        self.min_level = min_level
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, min_level.value.upper()))
        self._context: dict[str, Any] = {}
        self._correlation_id: str | None = None
        self._lock = threading.Lock()
        self._event_listeners: list[Callable[[LogEvent], None]] = []

        # Configure handlers
        if handlers:
            for handler in handlers:
                self._logger.addHandler(handler)

    def set_context(self, **kwargs) -> None:
        """Set persistent context for all logs.

        Args:
            **kwargs: Context key-value pairs
        """
        with self._lock:
            self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear persistent context."""
        with self._lock:
            self._context.clear()

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for request tracing.

        Args:
            correlation_id: Unique ID to trace related events
        """
        self._correlation_id = correlation_id

    def clear_correlation_id(self) -> None:
        """Clear correlation ID."""
        self._correlation_id = None

    @contextmanager
    def correlation_scope(self, correlation_id: str | None = None):
        """Context manager for correlation ID scope.

        Args:
            correlation_id: Optional ID, auto-generated if None

        Yields:
            The correlation ID
        """
        cid = correlation_id or str(uuid.uuid4())[:12]
        old_id = self._correlation_id
        self._correlation_id = cid
        try:
            yield cid
        finally:
            self._correlation_id = old_id

    def add_listener(self, listener: Callable[[LogEvent], None]) -> None:
        """Add event listener for real-time processing.

        Args:
            listener: Callback function for log events
        """
        self._event_listeners.append(listener)

    def remove_listener(self, listener: Callable[[LogEvent], None]) -> None:
        """Remove event listener.

        Args:
            listener: Callback function to remove
        """
        if listener in self._event_listeners:
            self._event_listeners.remove(listener)

    def log(
        self,
        category: LogCategory,
        level: LogLevel,
        event_type: str,
        message: str,
        data: dict[str, Any] | None = None,
        source: str = "",
        duration_ms: float | None = None,
        **context,
    ) -> LogEvent:
        """Log a structured event.

        Args:
            category: Event category
            level: Log level
            event_type: Specific event type
            message: Human-readable message
            data: Event-specific data
            source: Source module/component
            duration_ms: Operation duration
            **context: Additional context

        Returns:
            The created LogEvent
        """
        with self._lock:
            merged_context = {**self._context, **context}

        event = LogEvent(
            category=category,
            level=level,
            event_type=event_type,
            message=message,
            data=data or {},
            context=merged_context,
            source=source,
            correlation_id=self._correlation_id,
            duration_ms=duration_ms,
        )

        # Output to Python logger
        log_level = getattr(logging, level.value.upper())
        self._logger.log(log_level, event.to_json())

        # Notify listeners
        for listener in self._event_listeners:
            try:
                listener(event)
            except Exception:
                pass  # Don't let listener errors break logging

        return event

    def audit(
        self,
        action: str,
        resource: str,
        outcome: str,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> LogEvent:
        """
        Log an audit trail entry for compliance.

        Implements AbstractLogger.audit() interface.

        Args:
            action: What action was performed
            resource: What resource was affected
            outcome: Result (SUCCESS, FAILED, etc.)
            actor: Who/what performed the action
            details: Additional details

        Returns:
            The created LogEvent
        """
        return self.log(
            LogCategory.AUDIT,
            LogLevel.INFO,
            action,
            f"{actor} {action} on {resource}: {outcome}",
            data={
                "actor": actor,
                "resource": resource,
                "outcome": outcome,
                **(details or {}),
            },
            source="audit",
        )

    # =========================================================================
    # Execution Log Methods
    # =========================================================================

    def log_order_submitted(
        self,
        order_id: str,
        symbol: str,
        order_type: str = "limit",
        side: str = "buy",
        quantity: int = 1,
        limit_price: float | None = None,
        strategy: str | None = None,
        source: str = "execution",
        **extra,
    ) -> LogEvent:
        """Log order submission."""
        return self.log(
            LogCategory.EXECUTION,
            LogLevel.INFO,
            ExecutionEventType.ORDER_SUBMITTED.value,
            f"Order {order_id} submitted: {side.upper()} {quantity} {symbol}",
            {
                "order_id": order_id,
                "symbol": symbol,
                "order_type": order_type,
                "side": side,
                "quantity": quantity,
                "limit_price": limit_price,
                "strategy": strategy,
                **extra,
            },
            source=source,
        )

    def log_order_filled(
        self,
        order_id: str,
        symbol: str,
        fill_price: float,
        quantity: int,
        commission: float = 0.0,
        slippage_bps: float = 0.0,
        source: str = "execution",
        **extra,
    ) -> LogEvent:
        """Log order fill."""
        return self.log(
            LogCategory.EXECUTION,
            LogLevel.INFO,
            ExecutionEventType.ORDER_FILLED.value,
            f"Order {order_id} filled: {quantity} @ {fill_price:.2f}",
            {
                "order_id": order_id,
                "symbol": symbol,
                "fill_price": fill_price,
                "quantity": quantity,
                "commission": commission,
                "slippage_bps": slippage_bps,
                **extra,
            },
            source=source,
        )

    def log_order_cancelled(
        self, order_id: str, symbol: str, reason: str = "", source: str = "execution", **extra
    ) -> LogEvent:
        """Log order cancellation."""
        return self.log(
            LogCategory.EXECUTION,
            LogLevel.WARNING,
            ExecutionEventType.ORDER_CANCELLED.value,
            f"Order {order_id} cancelled: {reason}",
            {"order_id": order_id, "symbol": symbol, "reason": reason, **extra},
            source=source,
        )

    def log_position_opened(
        self,
        position_id: str,
        symbol: str,
        quantity: int,
        entry_price: float,
        strategy: str | None = None,
        source: str = "execution",
        **extra,
    ) -> LogEvent:
        """Log position opening."""
        return self.log(
            LogCategory.EXECUTION,
            LogLevel.INFO,
            ExecutionEventType.POSITION_OPENED.value,
            f"Position opened: {quantity} {symbol} @ {entry_price:.2f}",
            {
                "position_id": position_id,
                "symbol": symbol,
                "quantity": quantity,
                "entry_price": entry_price,
                "strategy": strategy,
                **extra,
            },
            source=source,
        )

    def log_position_closed(
        self,
        position_id: str,
        symbol: str,
        quantity: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        hold_time_hours: float = 0.0,
        exit_reason: str = "",
        source: str = "execution",
        **extra,
    ) -> LogEvent:
        """Log position closing."""
        level = LogLevel.INFO if pnl >= 0 else LogLevel.WARNING
        return self.log(
            LogCategory.EXECUTION,
            level,
            ExecutionEventType.POSITION_CLOSED.value,
            f"Position closed: {symbol} P&L ${pnl:+.2f} ({pnl_pct:+.1%})",
            {
                "position_id": position_id,
                "symbol": symbol,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "hold_time_hours": hold_time_hours,
                "exit_reason": exit_reason,
                **extra,
            },
            source=source,
        )

    # =========================================================================
    # Risk Log Methods
    # =========================================================================

    def log_circuit_breaker(
        self, is_halted: bool, reason: str, metrics: dict[str, Any] | None = None, source: str = "risk", **extra
    ) -> LogEvent:
        """Log circuit breaker event."""
        level = LogLevel.CRITICAL if is_halted else LogLevel.INFO
        event_type = (
            RiskEventType.CIRCUIT_BREAKER_TRIGGERED.value if is_halted else RiskEventType.CIRCUIT_BREAKER_RESET.value
        )
        return self.log(
            LogCategory.RISK,
            level,
            event_type,
            f"Circuit breaker {'HALTED' if is_halted else 'RESET'}: {reason}",
            {"is_halted": is_halted, "reason": reason, "metrics": metrics or {}, **extra},
            source=source,
        )

    def log_risk_warning(
        self,
        warning_type: str,
        current_value: float,
        threshold: float,
        message: str = "",
        source: str = "risk",
        **extra,
    ) -> LogEvent:
        """Log risk warning."""
        return self.log(
            LogCategory.RISK,
            LogLevel.WARNING,
            warning_type,
            message or f"{warning_type}: {current_value:.2%} (threshold: {threshold:.2%})",
            {"warning_type": warning_type, "current_value": current_value, "threshold": threshold, **extra},
            source=source,
        )

    def log_risk_breach(
        self,
        breach_type: str,
        current_value: float,
        threshold: float,
        action_taken: str = "",
        source: str = "risk",
        **extra,
    ) -> LogEvent:
        """Log risk limit breach."""
        return self.log(
            LogCategory.RISK,
            LogLevel.ERROR,
            breach_type,
            f"RISK BREACH: {breach_type} - {current_value:.2%} exceeds {threshold:.2%}",
            {
                "breach_type": breach_type,
                "current_value": current_value,
                "threshold": threshold,
                "action_taken": action_taken,
                **extra,
            },
            source=source,
        )

    # =========================================================================
    # Strategy Log Methods
    # =========================================================================

    def log_signal(
        self,
        signal_type: str,
        symbol: str,
        direction: str,
        strength: float,
        strategy: str,
        indicators: dict[str, float] | None = None,
        source: str = "strategy",
        **extra,
    ) -> LogEvent:
        """Log strategy signal."""
        return self.log(
            LogCategory.STRATEGY,
            LogLevel.INFO,
            StrategyEventType.SIGNAL_GENERATED.value,
            f"Signal: {direction.upper()} {symbol} (strength: {strength:.2f})",
            {
                "signal_type": signal_type,
                "symbol": symbol,
                "direction": direction,
                "strength": strength,
                "strategy": strategy,
                "indicators": indicators or {},
                **extra,
            },
            source=source,
        )

    def log_strategy_decision(
        self,
        decision: str,
        symbol: str,
        strategy: str,
        reason: str = "",
        confidence: float = 0.0,
        source: str = "strategy",
        **extra,
    ) -> LogEvent:
        """Log strategy decision (entry/exit approved or rejected)."""
        return self.log(
            LogCategory.STRATEGY,
            LogLevel.INFO,
            decision,
            f"Strategy decision: {decision} for {symbol}",
            {
                "decision": decision,
                "symbol": symbol,
                "strategy": strategy,
                "reason": reason,
                "confidence": confidence,
                **extra,
            },
            source=source,
        )

    # =========================================================================
    # System Log Methods
    # =========================================================================

    def log_system_event(
        self, event_type: str, message: str, level: LogLevel = LogLevel.INFO, source: str = "system", **data
    ) -> LogEvent:
        """Log system event."""
        return self.log(
            LogCategory.SYSTEM,
            level,
            event_type,
            message,
            data,
            source=source,
        )

    def log_error(self, error: Exception, context: str = "", source: str = "error", **extra) -> LogEvent:
        """Log error with exception details."""
        import traceback

        return self.log(
            LogCategory.ERROR,
            LogLevel.ERROR,
            "exception",
            f"Error in {context}: {type(error).__name__}: {error!s}",
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "traceback": traceback.format_exc(),
                **extra,
            },
            source=source,
        )

    # =========================================================================
    # Performance Log Methods
    # =========================================================================

    def log_performance(
        self, metrics: dict[str, float], period: str = "daily", source: str = "performance", **extra
    ) -> LogEvent:
        """Log performance metrics."""
        pnl = metrics.get("pnl", 0)
        return self.log(
            LogCategory.PERFORMANCE,
            LogLevel.INFO,
            "performance_update",
            f"{period.capitalize()} P&L: ${pnl:+.2f}",
            {"period": period, "metrics": metrics, **extra},
            source=source,
        )


# ============================================================================
# Factory Functions
# ============================================================================


def create_structured_logger(
    name: str = "trading_bot",
    log_file: str | None = None,
    console: bool = True,
    min_level: LogLevel = LogLevel.INFO,
) -> StructuredLogger:
    """Factory function to create configured logger.

    Args:
        name: Logger name
        log_file: Optional file path for log output
        console: Whether to output to console
        min_level: Minimum log level

    Returns:
        Configured StructuredLogger
    """
    handlers = []

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(console_handler)

    if log_file:
        from .log_handlers import create_rotating_file_handler

        handlers.append(create_rotating_file_handler(log_file))

    return StructuredLogger(name=name, min_level=min_level, handlers=handlers)


# Global logger instance
_default_logger: StructuredLogger | None = None


def get_logger() -> StructuredLogger:
    """Get the default structured logger instance.

    Returns:
        The default StructuredLogger
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = create_structured_logger()
    return _default_logger


def set_logger(logger: StructuredLogger) -> None:
    """Set the default structured logger instance.

    Args:
        logger: StructuredLogger to use as default
    """
    global _default_logger
    _default_logger = logger
