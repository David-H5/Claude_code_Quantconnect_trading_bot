"""
Execution Quality Metrics

DEPRECATED: This module has been moved to observability.metrics.collectors.execution.
This file provides backwards-compatible re-exports.

Use the new import path:
    from observability.metrics.collectors.execution import ExecutionQualityTracker
    from observability.metrics.collectors.execution import create_execution_tracker

Original: QuantConnect Compatible execution quality metrics
Refactored: Phase 3 - Consolidated Metrics Infrastructure
"""

# Re-export everything from new location for backwards compatibility
from observability.metrics.collectors.execution import (
    ExecutionDashboard,
    # Main class
    ExecutionQualityTracker,
    # Data classes
    OrderRecord,
    # Enums
    OrderStatus,
    QualityThresholds,
    # Factory functions
    create_execution_tracker,
    # Report generation
    generate_execution_report,
)


__all__ = [
    # Enums
    "OrderStatus",
    # Data classes
    "OrderRecord",
    "ExecutionDashboard",
    "QualityThresholds",
    # Main class
    "ExecutionQualityTracker",
    # Factory functions
    "create_execution_tracker",
    # Report generation
    "generate_execution_report",
]
