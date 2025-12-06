"""
Unified Logging Base Infrastructure

Provides the abstract base interface for all logging implementations.
Part of Phase 2 refactoring: Consolidate Logging Infrastructure.

All loggers should inherit from AbstractLogger to ensure consistent interface.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class LogLevel(Enum):
    """Log severity levels (unified across all loggers)."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    AUDIT = "audit"  # Special level for compliance logging


class LogCategory(Enum):
    """Log event categories for filtering and routing."""

    # Trading categories
    EXECUTION = "execution"  # Order submissions, fills, cancellations
    RISK = "risk"  # Circuit breaker, position limits, drawdown
    STRATEGY = "strategy"  # Entry/exit signals, strategy decisions
    POSITION = "position"  # Position changes
    TRADE = "trade"  # Trade executions

    # System categories
    SYSTEM = "system"  # API requests, WebSocket, resources
    ERROR = "error"  # Errors and exceptions
    PERFORMANCE = "performance"  # P&L, metrics, analytics

    # AI/Agent categories
    AGENT = "agent"  # Agent decisions and reasoning
    REASONING = "reasoning"  # Chain-of-thought reasoning

    # Compliance categories
    AUDIT = "audit"  # All trading decisions with context
    AUTHENTICATION = "authentication"  # Login, logout events
    CONFIGURATION = "configuration"  # Config changes

    # Infrastructure categories
    HOOK = "hook"  # Claude Code hook events


@dataclass
class LogEntry:
    """
    Unified log entry structure.

    All logging implementations should produce LogEntry instances
    for consistent processing and storage.
    """

    # Identity
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Classification
    level: LogLevel = LogLevel.INFO
    category: LogCategory = LogCategory.SYSTEM
    event_type: str = ""

    # Content
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    # Tracing
    source: str = ""
    correlation_id: str | None = None
    session_id: str | None = None

    # Metadata
    duration_ms: float | None = None
    actor: str | None = None
    resource: str | None = None
    outcome: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "category": self.category.value,
            "event_type": self.event_type,
            "message": self.message,
        }

        # Add optional fields if present
        if self.data:
            result["data"] = self.data
        if self.context:
            result["context"] = self.context
        if self.source:
            result["source"] = self.source
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.session_id:
            result["session_id"] = self.session_id
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.actor:
            result["actor"] = self.actor
        if self.resource:
            result["resource"] = self.resource
        if self.outcome:
            result["outcome"] = self.outcome

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogEntry:
        """Create from dictionary."""
        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())[:12]),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(timezone.utc),
            level=LogLevel(data.get("level", "info")),
            category=LogCategory(data.get("category", "system")),
            event_type=data.get("event_type", ""),
            message=data.get("message", ""),
            data=data.get("data", {}),
            context=data.get("context", {}),
            source=data.get("source", ""),
            correlation_id=data.get("correlation_id"),
            session_id=data.get("session_id"),
            duration_ms=data.get("duration_ms"),
            actor=data.get("actor"),
            resource=data.get("resource"),
            outcome=data.get("outcome"),
        )


class AbstractLogger(ABC):
    """
    Abstract base class for all loggers.

    Provides a consistent interface for logging across:
    - Structured logging (JSON)
    - Audit logging (compliance)
    - Decision logging (AI agents)
    - Reasoning logging (chain-of-thought)
    - Hook logging (Claude Code)
    """

    @abstractmethod
    def log(
        self,
        level: LogLevel,
        category: LogCategory,
        event_type: str,
        message: str,
        data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LogEntry:
        """
        Log a structured event.

        Args:
            level: Log severity level
            category: Event category
            event_type: Specific event type
            message: Human-readable message
            data: Event-specific data
            **kwargs: Additional context

        Returns:
            The created LogEntry
        """
        pass

    @abstractmethod
    def audit(
        self,
        action: str,
        resource: str,
        outcome: str,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> LogEntry:
        """
        Log an audit trail entry for compliance.

        Args:
            action: What action was performed
            resource: What resource was affected
            outcome: Result (SUCCESS, FAILED, etc.)
            actor: Who/what performed the action
            details: Additional details

        Returns:
            The created LogEntry
        """
        pass

    def debug(self, message: str, **kwargs: Any) -> LogEntry:
        """Log debug message."""
        return self.log(LogLevel.DEBUG, LogCategory.SYSTEM, "debug", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> LogEntry:
        """Log info message."""
        return self.log(LogLevel.INFO, LogCategory.SYSTEM, "info", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> LogEntry:
        """Log warning message."""
        return self.log(LogLevel.WARNING, LogCategory.SYSTEM, "warning", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> LogEntry:
        """Log error message."""
        return self.log(LogLevel.ERROR, LogCategory.ERROR, "error", message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> LogEntry:
        """Log critical message."""
        return self.log(LogLevel.CRITICAL, LogCategory.ERROR, "critical", message, **kwargs)


class LoggerAdapter(AbstractLogger):
    """
    Adapter base class for wrapping existing loggers.

    Use this to create adapters that make existing logging
    implementations conform to the AbstractLogger interface.
    """

    def __init__(self, wrapped_logger: Any):
        """
        Initialize adapter with wrapped logger.

        Args:
            wrapped_logger: The existing logger to wrap
        """
        self._wrapped = wrapped_logger

    @property
    def wrapped(self) -> Any:
        """Get the wrapped logger."""
        return self._wrapped


class CompositeLogger(AbstractLogger):
    """
    Logger that forwards to multiple loggers.

    Useful for sending logs to multiple destinations
    (e.g., console + file + audit trail).
    """

    def __init__(self, loggers: list[AbstractLogger] | None = None):
        """
        Initialize composite logger.

        Args:
            loggers: List of loggers to forward to
        """
        self._loggers: list[AbstractLogger] = loggers or []

    def add_logger(self, logger: AbstractLogger) -> None:
        """Add a logger to the composite."""
        self._loggers.append(logger)

    def remove_logger(self, logger: AbstractLogger) -> None:
        """Remove a logger from the composite."""
        if logger in self._loggers:
            self._loggers.remove(logger)

    def log(
        self,
        level: LogLevel,
        category: LogCategory,
        event_type: str,
        message: str,
        data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LogEntry:
        """Forward log to all loggers."""
        entry = None
        for logger in self._loggers:
            result = logger.log(level, category, event_type, message, data, **kwargs)
            if entry is None:
                entry = result
        return entry or LogEntry(level=level, category=category, event_type=event_type, message=message)

    def audit(
        self,
        action: str,
        resource: str,
        outcome: str,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> LogEntry:
        """Forward audit to all loggers."""
        entry = None
        for logger in self._loggers:
            result = logger.audit(action, resource, outcome, actor, details)
            if entry is None:
                entry = result
        return entry or LogEntry(
            level=LogLevel.AUDIT,
            category=LogCategory.AUDIT,
            event_type=action,
            message=f"{action} on {resource}",
            actor=actor,
            resource=resource,
            outcome=outcome,
        )


class FilteredLogger(AbstractLogger):
    """
    Logger that filters entries based on criteria.

    Wraps another logger and only forwards entries
    that pass the filter function.
    """

    def __init__(
        self,
        logger: AbstractLogger,
        filter_fn: Callable[[LogEntry], bool] | None = None,
        min_level: LogLevel = LogLevel.DEBUG,
        categories: list[LogCategory] | None = None,
    ):
        """
        Initialize filtered logger.

        Args:
            logger: Logger to wrap
            filter_fn: Custom filter function
            min_level: Minimum level to pass
            categories: Categories to include (None = all)
        """
        self._logger = logger
        self._filter_fn = filter_fn
        self._min_level = min_level
        self._categories = set(categories) if categories else None

        # Level ordering for comparison
        self._level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4,
            LogLevel.AUDIT: 5,
        }

    def _should_log(self, entry: LogEntry) -> bool:
        """Check if entry should be logged."""
        # Level filter
        if self._level_order.get(entry.level, 0) < self._level_order.get(self._min_level, 0):
            return False

        # Category filter
        if self._categories and entry.category not in self._categories:
            return False

        # Custom filter
        if self._filter_fn and not self._filter_fn(entry):
            return False

        return True

    def log(
        self,
        level: LogLevel,
        category: LogCategory,
        event_type: str,
        message: str,
        data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LogEntry:
        """Log if passes filter."""
        entry = LogEntry(
            level=level,
            category=category,
            event_type=event_type,
            message=message,
            data=data or {},
            **{k: v for k, v in kwargs.items() if k in LogEntry.__dataclass_fields__},
        )

        if self._should_log(entry):
            return self._logger.log(level, category, event_type, message, data, **kwargs)

        return entry

    def audit(
        self,
        action: str,
        resource: str,
        outcome: str,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> LogEntry:
        """Audit entries always pass filter."""
        return self._logger.audit(action, resource, outcome, actor, details)
