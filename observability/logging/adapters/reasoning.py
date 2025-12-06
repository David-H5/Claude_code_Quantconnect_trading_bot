"""
Reasoning Logger Adapter

Wraps llm.reasoning_logger.ReasoningLogger to provide AbstractLogger interface.
"""

from __future__ import annotations

from typing import Any

from observability.logging.base import (
    LogCategory,
    LogEntry,
    LoggerAdapter,
    LogLevel,
)


class ReasoningLoggerAdapter(LoggerAdapter):
    """
    Adapter for ReasoningLogger that implements AbstractLogger interface.

    Usage:
        from llm.reasoning_logger import ReasoningLogger
        from observability.logging.adapters import ReasoningLoggerAdapter

        reasoning_logger = ReasoningLogger()
        adapter = ReasoningLoggerAdapter(reasoning_logger)

        # Use AbstractLogger interface
        adapter.log(LogLevel.INFO, LogCategory.REASONING, "step", "Reasoning step")

        # Or access wrapped logger directly
        chain = adapter.wrapped.start_chain("agent", "task")
        chain.add_step("thought", confidence=0.8)
    """

    def __init__(self, reasoning_logger: Any):
        """
        Initialize adapter.

        Args:
            reasoning_logger: ReasoningLogger instance to wrap
        """
        super().__init__(reasoning_logger)

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

        For reasoning logging, this creates a basic log entry.
        For full chain-of-thought logging, use the wrapped logger directly.
        """
        entry = LogEntry(
            level=level,
            category=category,
            event_type=event_type,
            message=message,
            data=data or {},
            source="reasoning_logger",
            **{k: v for k, v in kwargs.items() if k in LogEntry.__dataclass_fields__},
        )

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

        For reasoning auditing, this creates a basic audit entry.
        For full reasoning chains, use the wrapped logger.
        """
        return LogEntry(
            level=LogLevel.AUDIT,
            category=LogCategory.REASONING,
            event_type=action,
            message=f"{actor} {action} on {resource}: {outcome}",
            data=details or {},
            actor=actor,
            resource=resource,
            outcome=outcome,
            source="reasoning_logger",
        )

    def start_chain(
        self,
        agent_name: str,
        task: str,
        **kwargs: Any,
    ) -> Any:
        """
        Convenience method to start a reasoning chain.

        Args:
            agent_name: Name of the agent
            task: Description of the task
            **kwargs: Additional arguments

        Returns:
            ReasoningChain from the wrapped logger
        """
        return self._wrapped.start_chain(agent_name, task, **kwargs)

    def complete_chain(
        self,
        chain_id: str,
        decision: str,
        confidence: float,
    ) -> Any:
        """
        Convenience method to complete a reasoning chain.

        Args:
            chain_id: ID of the chain to complete
            decision: Final decision
            confidence: Confidence in decision

        Returns:
            Completed ReasoningChain
        """
        return self._wrapped.complete_chain(chain_id, decision, confidence)
