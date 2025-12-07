"""Unified Logging Infrastructure.

Provides consolidated logging for the trading bot:
- Structured JSON logging for trading operations
- Compliance audit logging with hash chains
- Agent decision and reasoning logging adapters
- Hook activity logging

Usage:
    # Structured logging (trading operations)
    from observability.logging import StructuredLogger, create_structured_logger

    logger = create_structured_logger("trading_bot")
    logger.log_order_submitted("ORD-001", "SPY", quantity=10)

    # Audit logging (compliance)
    from observability.logging import AuditLogger, create_audit_logger

    audit = create_audit_logger()
    audit.log_trade(trade_id="TRD-001", symbol="SPY", ...)

    # Using adapters for specialized loggers
    from observability.logging.adapters import DecisionLoggerAdapter
    from llm.decision_logger import DecisionLogger

    decision_logger = DecisionLogger()
    adapter = DecisionLoggerAdapter(decision_logger)

Refactored: Phase 2 - Consolidated Logging Infrastructure
"""

# Agent logger
from observability.logging.agent import (
    AgentEventType,
    AgentLogEntry,
    AgentLogger,
    create_agent_logger,
    get_current_cwd,
    is_external_session,
)

# Base types
# Audit logger (compliance)
from observability.logging.audit import (
    AuditCategory,
    AuditEntry,
    AuditLevel,
    AuditLogger,
    AuditTrail,
    create_audit_logger,
)
from observability.logging.base import (
    AbstractLogger,
    CompositeLogger,
    FilteredLogger,
    LogCategory,
    LogEntry,
    LoggerAdapter,
    LogLevel,
)

# Structured logger (trading operations)
from observability.logging.structured import (
    ExecutionEventType,
    LogEvent,
    RiskEventType,
    StrategyEventType,
    StructuredLogger,
    create_structured_logger,
    get_logger,
    set_logger,
)
from observability.logging.structured import (
    LogCategory as StructuredLogCategory,
)
from observability.logging.structured import (
    LogLevel as StructuredLogLevel,
)


__all__ = [
    # Agent logger
    "AgentEventType",
    "AgentLogEntry",
    "AgentLogger",
    "create_agent_logger",
    "get_current_cwd",
    "is_external_session",
    # Base types
    "AbstractLogger",
    "LogEntry",
    "LogLevel",
    "LogCategory",
    "LoggerAdapter",
    "CompositeLogger",
    "FilteredLogger",
    # Structured logger
    "StructuredLogger",
    "LogEvent",
    "StructuredLogCategory",
    "StructuredLogLevel",
    "ExecutionEventType",
    "RiskEventType",
    "StrategyEventType",
    "create_structured_logger",
    "get_logger",
    "set_logger",
    # Audit logger
    "AuditLogger",
    "AuditEntry",
    "AuditTrail",
    "AuditLevel",
    "AuditCategory",
    "create_audit_logger",
]
