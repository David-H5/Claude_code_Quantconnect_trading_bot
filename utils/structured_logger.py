"""
Structured Logging Infrastructure for Trading Bot

DEPRECATED: This module has been moved to observability.logging.structured.
This file provides backwards-compatible re-exports.

Use the new import path:
    from observability.logging import StructuredLogger, create_structured_logger

Original: UPGRADE-009: Structured Logging (December 2025)
Refactored: Phase 2 - Consolidated Logging Infrastructure
"""

# Re-export everything from new location for backwards compatibility
from observability.logging.structured import (
    ExecutionEventType,
    # Enums
    LogCategory,
    # Data classes
    LogEvent,
    LogLevel,
    RiskEventType,
    StrategyEventType,
    # Main class
    StructuredLogger,
    # Factory functions
    create_structured_logger,
    get_logger,
    set_logger,
)


__all__ = [
    "ExecutionEventType",
    "LogCategory",
    "LogEvent",
    "LogLevel",
    "RiskEventType",
    "StrategyEventType",
    "StructuredLogger",
    "create_structured_logger",
    "get_logger",
    "set_logger",
]
