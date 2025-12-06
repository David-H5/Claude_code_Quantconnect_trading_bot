"""
Structured Exception Logging

Provides utilities for logging exceptions with rich context
and integration with the TradingError hierarchy.

Part of Exception Refactoring (Phase 5).

Usage:
    from observability.exception_logger import log_exception, exception_handler

    # Log any exception with context
    try:
        do_something()
    except Exception as e:
        log_exception(e, context={"operation": "fetch_data"})
        raise

    # Auto-log exceptions from functions
    @exception_handler
    def process_order(order):
        ...
"""

import functools
import json
import logging
import traceback
from collections.abc import Callable
from datetime import datetime
from typing import Any


logger = logging.getLogger(__name__)


def log_exception(
    e: Exception,
    context: dict[str, Any] | None = None,
    level: str = "error",
    include_traceback: bool = True,
) -> dict[str, Any]:
    """
    Log exception with structured context.

    Automatically detects TradingError instances and uses their
    built-in structured logging. For other exceptions, creates
    a structured log entry.

    Args:
        e: The exception to log.
        context: Additional context to include in the log.
        level: Log level (error, warning, critical, info, debug).
        include_traceback: Whether to include full traceback.

    Returns:
        The structured log data dictionary.

    Usage:
        try:
            submit_order(order)
        except OrderSubmissionError as e:
            log_exception(e, context={"attempt": 3})
            raise
        except Exception as e:
            log_data = log_exception(e, level="critical")
            send_alert(log_data)
            raise
    """
    # Import here to avoid circular imports
    from models.exceptions import TradingError

    log_func = getattr(logger, level)

    # Build structured log data
    if isinstance(e, TradingError):
        # Use built-in structured logging from TradingError
        log_data = e.to_dict()
        if context:
            log_data["additional_context"] = context
    else:
        # Wrap generic exceptions in structured format
        log_data = {
            "error_type": type(e).__name__,
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
            "recoverable": _is_recoverable(e),
            "context": context or {},
        }

        if include_traceback:
            log_data["traceback"] = traceback.format_exc()

    # Log the exception
    try:
        # Try to use structured logging with JSON extra
        log_func(
            f"{log_data['error_type']}: {log_data['message']}",
            extra={"structured": log_data},
        )
    except Exception:
        # Fallback to simple logging if structured logging fails
        log_func(f"{log_data['error_type']}: {log_data['message']}")

    return log_data


def _is_recoverable(e: Exception) -> bool:
    """
    Determine if an exception is recoverable.

    Args:
        e: The exception to check.

    Returns:
        True if the exception is likely recoverable.
    """
    # Import here to avoid circular imports
    from models.exceptions import TradingError

    if isinstance(e, TradingError):
        return e.recoverable

    # Non-recoverable exception types
    non_recoverable = (
        SystemExit,
        KeyboardInterrupt,
        MemoryError,
        AssertionError,
    )

    # Typically recoverable exception types
    recoverable = (
        TimeoutError,
        ConnectionError,
        IOError,
        OSError,
    )

    if isinstance(e, non_recoverable):
        return False
    if isinstance(e, recoverable):
        return True

    # Default: assume recoverable for safety
    return True


def exception_handler(
    func: Callable | None = None,
    *,
    level: str = "error",
    reraise: bool = True,
    include_args: bool = True,
    max_arg_length: int = 200,
) -> Callable:
    """
    Decorator to automatically log exceptions from a function.

    Logs structured exception information including function name,
    arguments, and full context before optionally reraising.

    Args:
        func: The function to decorate.
        level: Log level for exceptions (default: error).
        reraise: Whether to reraise the exception after logging.
        include_args: Whether to include function arguments in log.
        max_arg_length: Maximum length for arg string representation.

    Returns:
        Decorated function.

    Usage:
        @exception_handler
        def process_order(order):
            ...

        @exception_handler(level="warning", reraise=False)
        def optional_task():
            ...

        @exception_handler(include_args=False)  # For sensitive data
        def process_credentials(api_key):
            ...
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                # Build context with function info
                context: dict[str, Any] = {
                    "function": fn.__name__,
                    "module": fn.__module__,
                }

                if include_args:
                    # Truncate args to avoid huge log entries
                    args_str = str(args)[:max_arg_length]
                    kwargs_str = str(kwargs)[:max_arg_length]
                    context["args"] = args_str
                    context["kwargs"] = kwargs_str

                log_exception(e, context=context, level=level)

                if reraise:
                    raise

                return None

        return wrapper

    # Support both @exception_handler and @exception_handler() syntax
    if func is not None:
        return decorator(func)
    return decorator


class ExceptionAggregator:
    """
    Aggregate multiple exceptions for batch reporting.

    Useful for collecting errors during batch operations
    and generating a summary report.

    Usage:
        aggregator = ExceptionAggregator()

        for item in items:
            try:
                process(item)
            except Exception as e:
                aggregator.add(e, context={"item_id": item.id})

        if aggregator.has_errors:
            report = aggregator.get_report()
            send_alert(report)
    """

    def __init__(self, max_errors: int = 100) -> None:
        """
        Initialize exception aggregator.

        Args:
            max_errors: Maximum errors to store (prevents memory issues).
        """
        self.max_errors = max_errors
        self.errors: list[dict[str, Any]] = []
        self._error_counts: dict[str, int] = {}

    def add(
        self,
        e: Exception,
        context: dict[str, Any] | None = None,
        log: bool = True,
    ) -> None:
        """
        Add an exception to the aggregator.

        Args:
            e: The exception to add.
            context: Additional context.
            log: Whether to also log the exception.
        """
        if log:
            log_data = log_exception(e, context=context, level="warning")
        else:
            # Import here to avoid circular imports
            from models.exceptions import TradingError

            if isinstance(e, TradingError):
                log_data = e.to_dict()
                if context:
                    log_data["additional_context"] = context
            else:
                log_data = {
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "context": context or {},
                }

        # Track error counts by type
        error_type = log_data["error_type"]
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1

        # Store error (with limit)
        if len(self.errors) < self.max_errors:
            self.errors.append(log_data)

    @property
    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self.errors) > 0

    @property
    def error_count(self) -> int:
        """Get total error count."""
        return sum(self._error_counts.values())

    @property
    def error_types(self) -> dict[str, int]:
        """Get count of each error type."""
        return dict(self._error_counts)

    def get_report(self, include_details: bool = True) -> dict[str, Any]:
        """
        Generate a summary report of all errors.

        Args:
            include_details: Include individual error details.

        Returns:
            Report dictionary with summary and optional details.
        """
        report: dict[str, Any] = {
            "total_errors": self.error_count,
            "error_types": self.error_types,
            "timestamp": datetime.now().isoformat(),
        }

        if include_details:
            report["errors"] = self.errors

        # Calculate recoverable vs non-recoverable
        recoverable_count = sum(1 for e in self.errors if e.get("recoverable", True))
        report["recoverable_count"] = recoverable_count
        report["non_recoverable_count"] = len(self.errors) - recoverable_count

        return report

    def get_report_json(self, include_details: bool = True) -> str:
        """Get report as JSON string."""
        return json.dumps(self.get_report(include_details), indent=2)

    def clear(self) -> None:
        """Clear all collected errors."""
        self.errors.clear()
        self._error_counts.clear()

    def raise_if_errors(self, message: str = "Multiple errors occurred") -> None:
        """
        Raise an exception if any errors were collected.

        Args:
            message: Message for the raised exception.

        Raises:
            RuntimeError with error summary.
        """
        if self.has_errors:
            error_summary = ", ".join(f"{k}: {v}" for k, v in self._error_counts.items())
            raise RuntimeError(f"{message}: {error_summary}")


def create_exception_aggregator(max_errors: int = 100) -> ExceptionAggregator:
    """
    Factory function to create an exception aggregator.

    Args:
        max_errors: Maximum errors to store.

    Returns:
        ExceptionAggregator instance.
    """
    return ExceptionAggregator(max_errors=max_errors)


__all__ = [
    "ExceptionAggregator",
    "create_exception_aggregator",
    "exception_handler",
    "log_exception",
]
