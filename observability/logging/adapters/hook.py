"""
Hook Logger Adapter

Wraps .claude/hooks/core/hook_utils logging functions to provide AbstractLogger interface.
"""

from __future__ import annotations

from typing import Any

from observability.logging.base import (
    AbstractLogger,
    LogCategory,
    LogEntry,
    LogLevel,
)


class HookLoggerAdapter(AbstractLogger):
    """
    Adapter for hook_utils logging that implements AbstractLogger interface.

    This adapter wraps the functional logging from hook_utils to provide
    an object-oriented interface consistent with other loggers.

    Usage:
        from observability.logging.adapters import HookLoggerAdapter

        logger = HookLoggerAdapter("my_hook")

        # Use AbstractLogger interface
        logger.log(LogLevel.INFO, LogCategory.HOOK, "check", "Performing check")

        # Or use convenience methods
        logger.log_activity("check", {"result": "pass"})
        logger.log_error("Something failed", {"context": "data"})
    """

    def __init__(self, hook_name: str):
        """
        Initialize adapter.

        Args:
            hook_name: Name of the hook for logging context
        """
        self.hook_name = hook_name

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
        Log using AbstractLogger interface.

        Also logs to hook_utils for backwards compatibility.
        """
        entry = LogEntry(
            level=level,
            category=category,
            event_type=event_type,
            message=message,
            data=data or {},
            source=self.hook_name,
            **{k: v for k, v in kwargs.items() if k in LogEntry.__dataclass_fields__},
        )

        # Also log to hook_utils for backwards compatibility
        try:
            from claude.hooks.core.hook_utils import log_error, log_hook_activity

            level_str = level.value.upper()

            if level in (LogLevel.ERROR, LogLevel.CRITICAL):
                log_error(
                    self.hook_name,
                    message,
                    data or {},
                    recoverable=(level != LogLevel.CRITICAL),
                )
            else:
                log_hook_activity(
                    self.hook_name,
                    event_type,
                    {"message": message, **(data or {})},
                    level=level_str,
                )
        except ImportError:
            pass  # hook_utils not available

        return entry

    def audit(
        self,
        action: str,
        resource: str,
        outcome: str,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> LogEntry:
        """
        Log an audit trail entry for hooks.
        """
        return self.log(
            LogLevel.AUDIT,
            LogCategory.HOOK,
            action,
            f"{actor} {action} on {resource}: {outcome}",
            data={
                "actor": actor,
                "resource": resource,
                "outcome": outcome,
                **(details or {}),
            },
        )

    def log_activity(
        self,
        event_type: str,
        details: dict[str, Any],
        level: str = "INFO",
    ) -> LogEntry:
        """
        Log hook activity (convenience method).

        Args:
            event_type: Type of event (e.g., "check", "warn", "auto_fix")
            details: Event details
            level: Log level string

        Returns:
            LogEntry
        """
        log_level = LogLevel(level.lower()) if level.lower() in [l.value for l in LogLevel] else LogLevel.INFO

        return self.log(
            log_level,
            LogCategory.HOOK,
            event_type,
            f"Hook {self.hook_name}: {event_type}",
            data=details,
        )

    def log_error(
        self,
        error: str,
        context: dict[str, Any],
        recoverable: bool = True,
    ) -> LogEntry:
        """
        Log hook error (convenience method).

        Args:
            error: Error message
            context: Error context
            recoverable: Whether the error was recovered from

        Returns:
            LogEntry
        """
        level = LogLevel.ERROR if recoverable else LogLevel.CRITICAL

        return self.log(
            level,
            LogCategory.HOOK,
            "error",
            error,
            data={"context": context, "recoverable": recoverable},
        )
