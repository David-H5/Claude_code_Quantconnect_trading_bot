"""
Agent Performance Metrics

DEPRECATED: This module has been moved to observability.metrics.collectors.agent.
This file provides backwards-compatible re-exports.

Use the new import path:
    from observability.metrics.collectors.agent import AgentMetricsTracker
    from observability.metrics.collectors.agent import create_metrics_tracker

Original: QuantConnect Compatible agent metrics
Refactored: Phase 3 - Consolidated Metrics Infrastructure
"""

# Re-export everything from new location for backwards compatibility
from observability.metrics.collectors.agent import (
    AgentComparison,
    AgentMetrics,
    # Main class
    AgentMetricsTracker,
    # Data classes
    DecisionRecord,
    # Enums
    MetricCategory,
    PerformanceTrend,
    # Factory functions
    create_metrics_tracker,
    # Report generation
    generate_metrics_report,
)


__all__ = [
    # Enums
    "MetricCategory",
    # Data classes
    "DecisionRecord",
    "AgentMetrics",
    "PerformanceTrend",
    "AgentComparison",
    # Main class
    "AgentMetricsTracker",
    # Factory functions
    "create_metrics_tracker",
    # Report generation
    "generate_metrics_report",
]
