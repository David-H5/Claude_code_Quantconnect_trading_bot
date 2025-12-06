"""
Real-Time Metrics Aggregator

Aggregates metrics from multiple sources with sliding window calculations.
Provides unified view of system health and performance.

UPGRADE-014 Category 2: Observability & Debugging
Refactored: Phase 3 - Consolidated Metrics Infrastructure

Location: observability/metrics/aggregator.py
Old location: observability/metrics_aggregator.py (re-exports for compatibility)

QuantConnect Compatible: Yes
- Non-blocking aggregation
- Thread-safe operations
- Memory-bounded storage
"""

import logging
import statistics
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Lock, RLock
from typing import Any


logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"


class WindowSize(Enum):
    """Standard sliding window sizes."""

    ONE_MINUTE = 60
    FIVE_MINUTES = 300
    FIFTEEN_MINUTES = 900
    ONE_HOUR = 3600
    ONE_DAY = 86400


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class MetricPoint:
    """Single metric data point."""

    timestamp: datetime
    value: float
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric points with sliding window."""

    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    points: deque = field(default_factory=lambda: deque(maxlen=10000))
    _lock: Lock = field(default_factory=Lock, repr=False)

    def add_point(self, value: float, labels: dict[str, str] | None = None):
        """Add a data point."""
        with self._lock:
            self.points.append(
                MetricPoint(
                    timestamp=datetime.now(timezone.utc),
                    value=value,
                    labels=labels or {},
                )
            )

    def get_points(
        self,
        window_seconds: int | None = None,
        labels: dict[str, str] | None = None,
    ) -> list[MetricPoint]:
        """Get points within window, optionally filtered by labels."""
        with self._lock:
            points = list(self.points)

        now = datetime.now(timezone.utc)

        # Filter by time window
        if window_seconds:
            cutoff = now - timedelta(seconds=window_seconds)
            points = [p for p in points if p.timestamp >= cutoff]

        # Filter by labels
        if labels:
            points = [p for p in points if all(p.labels.get(k) == v for k, v in labels.items())]

        return points

    def get_latest(self) -> MetricPoint | None:
        """Get most recent point."""
        with self._lock:
            if self.points:
                return self.points[-1]
        return None


@dataclass
class AggregatedMetric:
    """Aggregated metric values over a window."""

    name: str
    window_seconds: int
    count: int = 0
    sum_value: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    avg_value: float = 0.0
    median_value: float = 0.0
    p95_value: float = 0.0
    p99_value: float = 0.0
    rate_per_second: float = 0.0
    latest_value: float | None = None
    latest_timestamp: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "window_seconds": self.window_seconds,
            "count": self.count,
            "sum": round(self.sum_value, 4),
            "min": round(self.min_value, 4) if self.min_value != float("inf") else None,
            "max": round(self.max_value, 4) if self.max_value != float("-inf") else None,
            "avg": round(self.avg_value, 4),
            "median": round(self.median_value, 4),
            "p95": round(self.p95_value, 4),
            "p99": round(self.p99_value, 4),
            "rate_per_second": round(self.rate_per_second, 4),
            "latest_value": round(self.latest_value, 4) if self.latest_value else None,
            "latest_timestamp": self.latest_timestamp.isoformat() if self.latest_timestamp else None,
        }


@dataclass
class SystemHealth:
    """Overall system health summary."""

    timestamp: datetime
    status: str  # healthy, degraded, unhealthy
    score: float  # 0.0 to 1.0

    # Component health
    agents_healthy: int = 0
    agents_total: int = 0
    active_alerts: int = 0
    error_rate_1m: float = 0.0

    # Token metrics
    tokens_1h: int = 0
    cost_1h: float = 0.0
    tokens_per_minute: float = 0.0

    # Performance
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Details
    components: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "score": round(self.score, 2),
            "agents_healthy": self.agents_healthy,
            "agents_total": self.agents_total,
            "active_alerts": self.active_alerts,
            "error_rate_1m": round(self.error_rate_1m, 4),
            "tokens_1h": self.tokens_1h,
            "cost_1h": round(self.cost_1h, 4),
            "tokens_per_minute": round(self.tokens_per_minute, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "components": self.components,
        }


# ============================================================================
# Metrics Aggregator
# ============================================================================


class MetricsAggregator:
    """
    Aggregates metrics from multiple sources.

    Provides:
    - Real-time metric collection
    - Sliding window calculations
    - Unified health view
    - Exportable metrics format

    Usage:
        aggregator = MetricsAggregator()

        # Register metrics
        aggregator.register_metric("llm_latency_ms", MetricType.HISTOGRAM)
        aggregator.register_metric("llm_errors", MetricType.COUNTER)

        # Record values
        aggregator.record("llm_latency_ms", 150.0, {"agent": "analyst"})
        aggregator.record("llm_errors", 1, {"agent": "analyst"})

        # Get aggregations
        stats = aggregator.get_aggregation("llm_latency_ms", WindowSize.FIVE_MINUTES)
        print(f"P95 latency: {stats.p95_value}ms")

        # Get system health
        health = aggregator.get_system_health()
        print(f"Status: {health.status}, Score: {health.score}")
    """

    def __init__(
        self,
        max_points_per_metric: int = 10000,
        default_windows: list[WindowSize] | None = None,
    ):
        """
        Initialize aggregator.

        Args:
            max_points_per_metric: Maximum points per metric series
            default_windows: Default windows for aggregation
        """
        self.max_points = max_points_per_metric
        self.default_windows = default_windows or [
            WindowSize.ONE_MINUTE,
            WindowSize.FIVE_MINUTES,
            WindowSize.FIFTEEN_MINUTES,
        ]

        self._metrics: dict[str, MetricSeries] = {}
        # Use RLock (reentrant lock) to allow nested locking - record() calls register_metric()
        self._lock = RLock()

        # Health check callbacks
        self._health_checkers: list[Callable[[], dict[str, Any]]] = []

        # Initialize standard metrics
        self._init_standard_metrics()

    def _init_standard_metrics(self):
        """Initialize standard system metrics."""
        standard_metrics = [
            ("llm_request_duration_ms", MetricType.HISTOGRAM, "LLM request latency", "ms"),
            ("llm_requests_total", MetricType.COUNTER, "Total LLM requests", ""),
            ("llm_errors_total", MetricType.COUNTER, "Total LLM errors", ""),
            ("llm_tokens_input", MetricType.COUNTER, "Input tokens", "tokens"),
            ("llm_tokens_output", MetricType.COUNTER, "Output tokens", "tokens"),
            ("llm_cost_usd", MetricType.COUNTER, "LLM cost", "USD"),
            ("agent_decision_duration_ms", MetricType.HISTOGRAM, "Agent decision time", "ms"),
            ("agent_decisions_total", MetricType.COUNTER, "Total agent decisions", ""),
            ("agent_errors_total", MetricType.COUNTER, "Agent errors", ""),
            ("tool_call_duration_ms", MetricType.HISTOGRAM, "Tool call latency", "ms"),
            ("tool_calls_total", MetricType.COUNTER, "Total tool calls", ""),
            ("tool_failures_total", MetricType.COUNTER, "Tool failures", ""),
        ]

        for name, metric_type, description, unit in standard_metrics:
            self.register_metric(name, metric_type, description, unit)

    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        unit: str = "",
    ) -> MetricSeries:
        """Register a new metric series."""
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]

            series = MetricSeries(
                name=name,
                metric_type=metric_type,
                description=description,
                unit=unit,
                points=deque(maxlen=self.max_points),
            )
            self._metrics[name] = series

        return series

    def record(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ):
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels for filtering
        """
        with self._lock:
            if name not in self._metrics:
                # Auto-register as gauge
                self.register_metric(name, MetricType.GAUGE)

            series = self._metrics[name]

        series.add_point(value, labels)

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ):
        """Increment a counter metric."""
        self.record(name, value, labels)

    def observe(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ):
        """Observe a histogram value."""
        self.record(name, value, labels)

    def get_aggregation(
        self,
        name: str,
        window: WindowSize,
        labels: dict[str, str] | None = None,
    ) -> AggregatedMetric:
        """
        Get aggregated metrics for a window.

        Args:
            name: Metric name
            window: Time window
            labels: Optional label filter

        Returns:
            AggregatedMetric with statistics
        """
        with self._lock:
            if name not in self._metrics:
                return AggregatedMetric(name=name, window_seconds=window.value)
            series = self._metrics[name]

        points = series.get_points(window.value, labels)

        if not points:
            return AggregatedMetric(name=name, window_seconds=window.value)

        values = [p.value for p in points]
        sorted_values = sorted(values)

        count = len(values)
        sum_value = sum(values)
        min_value = min(values)
        max_value = max(values)
        avg_value = sum_value / count if count > 0 else 0.0

        # Percentiles
        median_value = statistics.median(values) if values else 0.0
        p95_idx = int(0.95 * count)
        p99_idx = int(0.99 * count)
        p95_value = sorted_values[min(p95_idx, count - 1)] if values else 0.0
        p99_value = sorted_values[min(p99_idx, count - 1)] if values else 0.0

        # Rate calculation
        rate_per_second = count / window.value if window.value > 0 else 0.0

        # Latest value
        latest = series.get_latest()

        return AggregatedMetric(
            name=name,
            window_seconds=window.value,
            count=count,
            sum_value=sum_value,
            min_value=min_value,
            max_value=max_value,
            avg_value=avg_value,
            median_value=median_value,
            p95_value=p95_value,
            p99_value=p99_value,
            rate_per_second=rate_per_second,
            latest_value=latest.value if latest else None,
            latest_timestamp=latest.timestamp if latest else None,
        )

    def get_all_aggregations(
        self,
        window: WindowSize,
    ) -> dict[str, AggregatedMetric]:
        """Get aggregations for all metrics."""
        with self._lock:
            metric_names = list(self._metrics.keys())

        return {name: self.get_aggregation(name, window) for name in metric_names}

    def register_health_checker(self, checker: Callable[[], dict[str, Any]]):
        """Register a health check callback."""
        self._health_checkers.append(checker)

    def get_system_health(self) -> SystemHealth:
        """
        Get overall system health.

        Returns:
            SystemHealth with status and metrics
        """
        now = datetime.now(timezone.utc)
        components = {}
        issues = []

        # Collect component health from registered checkers
        for checker in self._health_checkers:
            try:
                result = checker()
                if result:
                    components.update(result)
            except Exception as e:
                logger.warning(f"Health checker failed: {e}")
                issues.append(str(e))

        # Get error metrics
        errors_1m = self.get_aggregation("llm_errors_total", WindowSize.ONE_MINUTE)
        requests_1m = self.get_aggregation("llm_requests_total", WindowSize.ONE_MINUTE)

        error_rate = 0.0
        if requests_1m.count > 0:
            error_rate = errors_1m.sum_value / requests_1m.count

        # Get latency metrics
        latency_5m = self.get_aggregation("llm_request_duration_ms", WindowSize.FIVE_MINUTES)

        # Get token metrics (1 hour window)
        tokens_input_1h = self.get_aggregation("llm_tokens_input", WindowSize.ONE_HOUR)
        tokens_output_1h = self.get_aggregation("llm_tokens_output", WindowSize.ONE_HOUR)
        cost_1h = self.get_aggregation("llm_cost_usd", WindowSize.ONE_HOUR)

        total_tokens_1h = int(tokens_input_1h.sum_value + tokens_output_1h.sum_value)
        tokens_per_minute = total_tokens_1h / 60 if total_tokens_1h > 0 else 0.0

        # Calculate health score (0.0 to 1.0)
        score = 1.0

        # Error rate penalty (up to 0.3)
        if error_rate > 0.1:
            score -= 0.3
        elif error_rate > 0.05:
            score -= 0.15
        elif error_rate > 0.01:
            score -= 0.05

        # Latency penalty (up to 0.2)
        if latency_5m.p99_value > 5000:  # > 5s
            score -= 0.2
        elif latency_5m.p99_value > 2000:  # > 2s
            score -= 0.1
        elif latency_5m.p99_value > 1000:  # > 1s
            score -= 0.05

        # Issues penalty
        score -= len(issues) * 0.1

        score = max(0.0, min(1.0, score))

        # Determine status
        if score >= 0.9:
            status = "healthy"
        elif score >= 0.7:
            status = "degraded"
        else:
            status = "unhealthy"

        # Count healthy agents from components
        agents_healthy = sum(1 for c in components.values() if isinstance(c, dict) and c.get("healthy", False))
        agents_total = len(components)

        return SystemHealth(
            timestamp=now,
            status=status,
            score=score,
            agents_healthy=agents_healthy,
            agents_total=agents_total,
            active_alerts=len(issues),
            error_rate_1m=error_rate,
            tokens_1h=total_tokens_1h,
            cost_1h=cost_1h.sum_value,
            tokens_per_minute=tokens_per_minute,
            avg_latency_ms=latency_5m.avg_value,
            p99_latency_ms=latency_5m.p99_value,
            components=components,
        )

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-compatible metrics text
        """
        lines = []
        now = datetime.now(timezone.utc)

        with self._lock:
            for name, series in self._metrics.items():
                # HELP line
                if series.description:
                    lines.append(f"# HELP {name} {series.description}")

                # TYPE line
                type_str = "gauge"
                if series.metric_type == MetricType.COUNTER:
                    type_str = "counter"
                elif series.metric_type == MetricType.HISTOGRAM:
                    type_str = "histogram"
                lines.append(f"# TYPE {name} {type_str}")

                # Get latest values grouped by labels
                points = series.get_points(window_seconds=60)
                label_values: dict[str, float] = {}

                for p in points:
                    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(p.labels.items())) if p.labels else ""
                    key = label_str
                    # For counters, sum; for gauges/histograms, take latest
                    if series.metric_type == MetricType.COUNTER:
                        label_values[key] = label_values.get(key, 0) + p.value
                    else:
                        label_values[key] = p.value

                for label_str, value in label_values.items():
                    if label_str:
                        lines.append(f"{name}{{{label_str}}} {value}")
                    else:
                        lines.append(f"{name} {value}")

        return "\n".join(lines)

    def export_json(
        self,
        windows: list[WindowSize] | None = None,
    ) -> dict[str, Any]:
        """
        Export all metrics as JSON.

        Args:
            windows: Windows to export (default: standard windows)

        Returns:
            JSON-serializable metrics dict
        """
        windows = windows or self.default_windows

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health": self.get_system_health().to_dict(),
            "metrics": {},
        }

        with self._lock:
            metric_names = list(self._metrics.keys())

        for name in metric_names:
            result["metrics"][name] = {}
            for window in windows:
                agg = self.get_aggregation(name, window)
                result["metrics"][name][f"{window.value}s"] = agg.to_dict()

        return result

    def get_metric_names(self) -> list[str]:
        """Get all registered metric names."""
        with self._lock:
            return list(self._metrics.keys())

    def clear(self):
        """Clear all metric data (but keep registrations)."""
        with self._lock:
            for series in self._metrics.values():
                series.points.clear()

    def to_dict(self) -> dict[str, Any]:
        """Export current state as dictionary."""
        return self.export_json()


# ============================================================================
# Global Aggregator Instance
# ============================================================================

_global_aggregator: MetricsAggregator | None = None
_aggregator_lock = Lock()


def get_global_aggregator() -> MetricsAggregator:
    """Get the global metrics aggregator singleton."""
    global _global_aggregator

    if _global_aggregator is None:
        with _aggregator_lock:
            if _global_aggregator is None:
                _global_aggregator = MetricsAggregator()

    return _global_aggregator


def create_aggregator(
    max_points_per_metric: int = 10000,
    default_windows: list[WindowSize] | None = None,
) -> MetricsAggregator:
    """Factory function to create a new aggregator."""
    return MetricsAggregator(
        max_points_per_metric=max_points_per_metric,
        default_windows=default_windows,
    )


# ============================================================================
# Convenience Functions
# ============================================================================


def record_metric(name: str, value: float, labels: dict[str, str] | None = None):
    """Record a metric to the global aggregator."""
    get_global_aggregator().record(name, value, labels)


def record_llm_request(
    agent_name: str,
    model: str,
    duration_ms: float,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    success: bool = True,
):
    """Convenience function to record all LLM request metrics."""
    labels = {"agent": agent_name, "model": model}
    aggregator = get_global_aggregator()

    aggregator.observe("llm_request_duration_ms", duration_ms, labels)
    aggregator.increment("llm_requests_total", 1, labels)
    aggregator.increment("llm_tokens_input", input_tokens, labels)
    aggregator.increment("llm_tokens_output", output_tokens, labels)
    aggregator.increment("llm_cost_usd", cost_usd, labels)

    if not success:
        aggregator.increment("llm_errors_total", 1, labels)


def record_agent_decision(
    agent_name: str,
    duration_ms: float,
    success: bool = True,
):
    """Convenience function to record agent decision metrics."""
    labels = {"agent": agent_name}
    aggregator = get_global_aggregator()

    aggregator.observe("agent_decision_duration_ms", duration_ms, labels)
    aggregator.increment("agent_decisions_total", 1, labels)

    if not success:
        aggregator.increment("agent_errors_total", 1, labels)


def record_tool_call(
    tool_name: str,
    agent_name: str,
    duration_ms: float,
    success: bool = True,
):
    """Convenience function to record tool call metrics."""
    labels = {"tool": tool_name, "agent": agent_name}
    aggregator = get_global_aggregator()

    aggregator.observe("tool_call_duration_ms", duration_ms, labels)
    aggregator.increment("tool_calls_total", 1, labels)

    if not success:
        aggregator.increment("tool_failures_total", 1, labels)
