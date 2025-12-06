"""
Decision Logger Adapter

Wraps llm.decision_logger.DecisionLogger to provide AbstractLogger interface.
"""

from __future__ import annotations

from typing import Any

from observability.logging.base import (
    LogCategory,
    LogEntry,
    LoggerAdapter,
    LogLevel,
)


class DecisionLoggerAdapter(LoggerAdapter):
    """
    Adapter for DecisionLogger that implements AbstractLogger interface.

    Usage:
        from llm.decision_logger import DecisionLogger
        from observability.logging.adapters import DecisionLoggerAdapter

        decision_logger = DecisionLogger()
        adapter = DecisionLoggerAdapter(decision_logger)

        # Use AbstractLogger interface
        adapter.log(LogLevel.INFO, LogCategory.AGENT, "decision", "Agent made decision")

        # Or access wrapped logger directly
        adapter.wrapped.log_decision(...)
    """

    def __init__(self, decision_logger: Any):
        """
        Initialize adapter.

        Args:
            decision_logger: DecisionLogger instance to wrap
        """
        super().__init__(decision_logger)

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

        For decision logging, this creates a basic log entry.
        For full decision logging, use the wrapped logger directly.
        """
        entry = LogEntry(
            level=level,
            category=category,
            event_type=event_type,
            message=message,
            data=data or {},
            source="decision_logger",
            **{k: v for k, v in kwargs.items() if k in LogEntry.__dataclass_fields__},
        )

        # If this is an agent/decision category, we could potentially
        # create a decision log entry, but the DecisionLogger has a
        # much more complex interface, so we just return the basic entry
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
        Log an audit trail entry.

        For decision auditing, this creates a basic audit entry.
        For full decision logging with reasoning chains, use the wrapped logger.
        """
        return LogEntry(
            level=LogLevel.AUDIT,
            category=LogCategory.AGENT,
            event_type=action,
            message=f"{actor} {action} on {resource}: {outcome}",
            data=details or {},
            actor=actor,
            resource=resource,
            outcome=outcome,
            source="decision_logger",
        )

    def log_decision(
        self,
        agent_name: str,
        decision: str,
        confidence: float,
        context: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """
        Convenience method to log a decision using the wrapped logger.

        Args:
            agent_name: Name of the agent making the decision
            decision: The decision made
            confidence: Confidence level (0-1)
            context: Context for the decision
            **kwargs: Additional arguments for DecisionLogger.log_decision

        Returns:
            AgentDecisionLog from the wrapped logger
        """
        return self._wrapped.log_decision(
            agent_name=agent_name,
            decision=decision,
            confidence=confidence,
            context=context,
            **kwargs,
        )
