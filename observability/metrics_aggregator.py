"""
Real-Time Metrics Aggregator

DEPRECATED: This module has been moved to observability.metrics.aggregator.
This file provides backwards-compatible re-exports.

Use the new import path:
    from observability.metrics.aggregator import MetricsAggregator
    from observability.metrics.aggregator import create_aggregator

Original: UPGRADE-014 Category 2: Observability & Debugging
Refactored: Phase 3 - Consolidated Metrics Infrastructure
"""

# Re-export everything from new location for backwards compatibility
from observability.metrics.aggregator import (
    AggregatedMetric,
    # Data classes
    MetricPoint,
    # Main class
    MetricsAggregator,
    MetricSeries,
    # Enums
    MetricType,
    SystemHealth,
    WindowSize,
    create_aggregator,
    # Factory functions
    get_global_aggregator,
    record_agent_decision,
    record_llm_request,
    # Convenience functions
    record_metric,
    record_tool_call,
)


__all__ = [
    # Enums
    "MetricType",
    "WindowSize",
    # Data classes
    "MetricPoint",
    "MetricSeries",
    "AggregatedMetric",
    "SystemHealth",
    # Main class
    "MetricsAggregator",
    # Factory functions
    "get_global_aggregator",
    "create_aggregator",
    # Convenience functions
    "record_metric",
    "record_llm_request",
    "record_agent_decision",
    "record_tool_call",
]
