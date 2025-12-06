"""
Logging Adapters

Provides adapters that wrap existing specialized loggers to conform
to the AbstractLogger interface while preserving their unique functionality.

Available adapters:
- DecisionLoggerAdapter: Wraps llm.decision_logger.DecisionLogger
- ReasoningLoggerAdapter: Wraps llm.reasoning_logger.ReasoningLogger
- HookLoggerAdapter: Wraps .claude/hooks/core/hook_utils logging functions
"""

from observability.logging.adapters.decision import DecisionLoggerAdapter
from observability.logging.adapters.hook import HookLoggerAdapter
from observability.logging.adapters.reasoning import ReasoningLoggerAdapter


__all__ = [
    "DecisionLoggerAdapter",
    "HookLoggerAdapter",
    "ReasoningLoggerAdapter",
]
