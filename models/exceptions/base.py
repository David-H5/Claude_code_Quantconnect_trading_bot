"""Base exception classes with rich context."""

import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ErrorContext:
    """Rich context for debugging exceptions."""

    timestamp: datetime = field(default_factory=datetime.now)
    component: str = ""
    operation: str = ""
    symbol: str | None = None
    order_id: str | None = None
    agent_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "operation": self.operation,
        }
        if self.symbol:
            result["symbol"] = self.symbol
        if self.order_id:
            result["order_id"] = self.order_id
        if self.agent_name:
            result["agent_name"] = self.agent_name
        if self.extra:
            result.update(self.extra)
        return result


class TradingError(Exception):
    """Base exception for all trading-related errors.

    Provides rich error context, automatic timestamps, chaining support,
    and structured logging capabilities.

    Usage:
        try:
            submit_order(order)
        except TradingError as e:
            logger.error(f"Trading error: {e}", extra=e.to_dict())
            if e.recoverable:
                retry_order(order)
    """

    def __init__(
        self,
        message: str,
        context: ErrorContext | None = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.context = context or ErrorContext()
        self.recoverable = recoverable
        self.timestamp = datetime.now()

    def with_context(self, **kwargs: Any) -> "TradingError":
        """Add context and return self for chaining."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.extra[key] = value
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to structured dict for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context.to_dict(),
            "traceback": traceback.format_exc(),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"message={str(self)!r}, " f"recoverable={self.recoverable})"


__all__ = ["ErrorContext", "TradingError"]
