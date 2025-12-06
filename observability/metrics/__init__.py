"""
Unified Metrics Infrastructure

Provides consolidated metrics collection and export for the trading bot:
- Base metric types (Counter, Gauge, Histogram, Timer)
- Domain-specific collectors (trading, execution, agent, token)
- Real-time aggregation with sliding windows
- Multiple export formats (JSON, CSV, Prometheus)

Usage:
    # Base metrics
    from observability.metrics import counter, gauge, histogram, timer

    requests = counter("api_requests", "Total API requests", labels=["endpoint"])
    requests.inc(endpoint="/orders")

    # Trading metrics
    from observability.metrics.collectors import calculate_advanced_trading_metrics

    metrics = calculate_advanced_trading_metrics(trades, balance)

    # Export
    from observability.metrics.exporters import export_to_json, export_to_prometheus

    json_data = export_to_json()

Refactored: Phase 3 - Consolidated Metrics Infrastructure
"""

# Base types
from observability.metrics.aggregator import (
    MetricPoint as AggregatorMetricPoint,
)

# Aggregator
from observability.metrics.aggregator import (
    MetricsAggregator,
    MetricSeries,
    WindowSize,
    create_aggregator,
)
from observability.metrics.aggregator import (
    MetricType as AggregatorMetricType,
)
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


# Legacy aliases for backwards compatibility
MetricsRegistry = MetricRegistry


def create_metrics_registry() -> MetricRegistry:
    """Create a new metrics registry. DEPRECATED: Use MetricRegistry() directly."""
    return MetricRegistry()


def get_trading_metrics():
    """Get pre-defined trading metrics. DEPRECATED: Use collectors.trading instead."""
    from observability.metrics.collectors.trading import (
        get_trading_metrics as _get_trading_metrics,
    )

    return _get_trading_metrics()


__all__ = [
    # Base types
    "MetricType",
    "MetricUnit",
    "MetricPoint",
    "MetricDefinition",
    "MetricSnapshot",
    "BaseMetric",
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    "TimerContext",
    "MetricRegistry",
    "MetricsRegistry",  # Legacy alias
    "get_default_registry",
    "counter",
    "gauge",
    "histogram",
    "timer",
    # Aggregator
    "MetricsAggregator",
    "MetricSeries",
    "AggregatorMetricPoint",
    "WindowSize",
    "AggregatorMetricType",
    "create_aggregator",
    # Legacy functions
    "create_metrics_registry",
    "get_trading_metrics",
]
