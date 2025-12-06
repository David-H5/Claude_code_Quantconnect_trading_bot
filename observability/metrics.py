"""
Metrics Collection for AI Trading Agent

DEPRECATED: This module has been moved to observability.metrics.base.
This file provides backwards-compatible re-exports.

Use the new import path:
    from observability.metrics import Counter, Gauge, Histogram, Timer
    from observability.metrics import MetricRegistry, get_default_registry

Original: UPGRADE-015 Phase 6: Observability Setup
Refactored: Phase 3 - Consolidated Metrics Infrastructure
"""

# Re-export everything from new location for backwards compatibility
from observability.metrics.base import (
    # Metric classes
    BaseMetric,
    Counter,
    Gauge,
    Histogram,
    MetricDefinition,
    # Data classes
    MetricPoint,
    # Registry
    MetricRegistry,
    MetricSnapshot,
    # Enums
    MetricType,
    MetricUnit,
    Timer,
    TimerContext,
    # Factory functions
    counter,
    gauge,
    get_default_registry,
    histogram,
    timer,
)


# Legacy compatibility: MetricsRegistry -> MetricRegistry
MetricsRegistry = MetricRegistry

# Legacy compatibility: MetricValue -> MetricPoint
MetricValue = MetricPoint


def get_trading_metrics():
    """
    Get pre-defined trading metrics.

    DEPRECATED: Use observability.metrics.collectors.trading instead.
    """
    from observability.metrics.collectors.trading import (
        get_trading_metrics as _get_trading_metrics,
    )

    return _get_trading_metrics()


def create_metrics_registry() -> MetricRegistry:
    """
    Create a new metrics registry.

    DEPRECATED: Use MetricRegistry() directly.
    """
    return MetricRegistry()


__all__ = [
    # Enums
    "MetricType",
    "MetricUnit",
    # Data classes
    "MetricValue",  # Legacy alias
    "MetricPoint",
    "MetricDefinition",
    "MetricSnapshot",
    # Metric classes
    "BaseMetric",
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    "TimerContext",
    # Registry
    "MetricRegistry",
    "MetricsRegistry",  # Legacy alias
    "get_default_registry",
    # Factory functions
    "counter",
    "gauge",
    "histogram",
    "timer",
    # Legacy functions
    "get_trading_metrics",
    "create_metrics_registry",
]
