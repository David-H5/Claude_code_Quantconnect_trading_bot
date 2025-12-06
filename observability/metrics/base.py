"""
Unified Metrics Base Infrastructure

Provides the abstract base interface for all metrics implementations.
Part of Phase 3 refactoring: Consolidate Metrics Infrastructure.

All metric collectors should use these base types for consistency.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"  # Can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"  # Duration measurements
    RATE = "rate"  # Rate of change


class MetricUnit(Enum):
    """Standard metric units."""

    COUNT = "count"
    PERCENT = "percent"
    MILLISECONDS = "ms"
    SECONDS = "seconds"
    BYTES = "bytes"
    DOLLARS = "dollars"
    BPS = "bps"  # Basis points
    RATIO = "ratio"


@dataclass
class MetricPoint:
    """Single metric data point."""

    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
        }


@dataclass
class MetricDefinition:
    """Definition of a metric."""

    name: str
    metric_type: MetricType
    description: str = ""
    unit: MetricUnit = MetricUnit.COUNT
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "description": self.description,
            "unit": self.unit.value,
            "labels": self.labels,
        }


@dataclass
class MetricSnapshot:
    """Snapshot of metric values at a point in time."""

    definition: MetricDefinition
    values: dict[tuple, float]  # label_values -> value
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    statistics: dict[str, float] | None = None  # For histograms: min, max, avg, p50, p95, p99

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "metric": self.definition.to_dict(),
            "values": {str(k): v for k, v in self.values.items()},
            "timestamp": self.timestamp.isoformat(),
        }
        if self.statistics:
            result["statistics"] = self.statistics
        return result


class BaseMetric(ABC):
    """Abstract base class for all metric types."""

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: MetricUnit = MetricUnit.COUNT,
        labels: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.unit = unit
        self.labels = labels or []
        self._lock = threading.Lock()

    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """Return the metric type."""
        pass

    @abstractmethod
    def get_snapshot(self) -> MetricSnapshot:
        """Get current metric snapshot."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset metric values."""
        pass

    def get_definition(self) -> MetricDefinition:
        """Get metric definition."""
        return MetricDefinition(
            name=self.name,
            metric_type=self.metric_type,
            description=self.description,
            unit=self.unit,
            labels=self.labels,
        )

    def _make_key(self, **label_values: str) -> tuple:
        """Create key from label values."""
        return tuple(label_values.get(label, "") for label in self.labels)


class Counter(BaseMetric):
    """Counter metric that only goes up."""

    def __init__(self, name: str, description: str = "", labels: list[str] | None = None):
        super().__init__(name, description, MetricUnit.COUNT, labels)
        self._values: dict[tuple, float] = defaultdict(float)

    @property
    def metric_type(self) -> MetricType:
        return MetricType.COUNTER

    def inc(self, value: float = 1.0, **label_values: str) -> None:
        """Increment the counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented")
        with self._lock:
            key = self._make_key(**label_values)
            self._values[key] += value

    def get(self, **label_values: str) -> float:
        """Get current counter value."""
        with self._lock:
            key = self._make_key(**label_values)
            return self._values[key]

    def get_snapshot(self) -> MetricSnapshot:
        """Get current snapshot."""
        with self._lock:
            return MetricSnapshot(
                definition=self.get_definition(),
                values=dict(self._values),
            )

    def reset(self) -> None:
        """Reset counter (use sparingly)."""
        with self._lock:
            self._values.clear()


class Gauge(BaseMetric):
    """Gauge metric that can go up or down."""

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: MetricUnit = MetricUnit.COUNT,
        labels: list[str] | None = None,
    ):
        super().__init__(name, description, unit, labels)
        self._values: dict[tuple, float] = defaultdict(float)

    @property
    def metric_type(self) -> MetricType:
        return MetricType.GAUGE

    def set(self, value: float, **label_values: str) -> None:
        """Set the gauge value."""
        with self._lock:
            key = self._make_key(**label_values)
            self._values[key] = value

    def inc(self, value: float = 1.0, **label_values: str) -> None:
        """Increment the gauge."""
        with self._lock:
            key = self._make_key(**label_values)
            self._values[key] += value

    def dec(self, value: float = 1.0, **label_values: str) -> None:
        """Decrement the gauge."""
        with self._lock:
            key = self._make_key(**label_values)
            self._values[key] -= value

    def get(self, **label_values: str) -> float:
        """Get current gauge value."""
        with self._lock:
            key = self._make_key(**label_values)
            return self._values[key]

    def get_snapshot(self) -> MetricSnapshot:
        """Get current snapshot."""
        with self._lock:
            return MetricSnapshot(
                definition=self.get_definition(),
                values=dict(self._values),
            )

    def reset(self) -> None:
        """Reset gauge."""
        with self._lock:
            self._values.clear()


class Histogram(BaseMetric):
    """Histogram metric for tracking distributions."""

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: MetricUnit = MetricUnit.COUNT,
        labels: list[str] | None = None,
        buckets: list[float] | None = None,
    ):
        super().__init__(name, description, unit, labels)
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._values: dict[tuple, list[float]] = defaultdict(list)
        self._max_samples = 10000  # Limit memory usage

    @property
    def metric_type(self) -> MetricType:
        return MetricType.HISTOGRAM

    def observe(self, value: float, **label_values: str) -> None:
        """Record an observation."""
        with self._lock:
            key = self._make_key(**label_values)
            values = self._values[key]
            values.append(value)
            # Limit memory usage
            if len(values) > self._max_samples:
                self._values[key] = values[-self._max_samples :]

    def get_statistics(self, **label_values: str) -> dict[str, float]:
        """Get statistics for a label combination."""
        with self._lock:
            key = self._make_key(**label_values)
            values = self._values[key]
            if not values:
                return {}

            sorted_values = sorted(values)
            n = len(sorted_values)

            return {
                "count": n,
                "sum": sum(values),
                "min": sorted_values[0],
                "max": sorted_values[-1],
                "avg": sum(values) / n,
                "p50": sorted_values[n // 2],
                "p95": sorted_values[int(n * 0.95)] if n > 20 else sorted_values[-1],
                "p99": sorted_values[int(n * 0.99)] if n > 100 else sorted_values[-1],
            }

    def get_snapshot(self) -> MetricSnapshot:
        """Get current snapshot with statistics."""
        with self._lock:
            # Calculate statistics for each label combination
            stats = {}
            for key, values in self._values.items():
                if values:
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    stats[key] = sum(values) / n  # Use average as the value

            # Get overall statistics
            all_values = []
            for v in self._values.values():
                all_values.extend(v)

            overall_stats = None
            if all_values:
                sorted_all = sorted(all_values)
                n = len(sorted_all)
                overall_stats = {
                    "count": n,
                    "min": sorted_all[0],
                    "max": sorted_all[-1],
                    "avg": sum(all_values) / n,
                    "p50": sorted_all[n // 2],
                    "p95": sorted_all[int(n * 0.95)] if n > 20 else sorted_all[-1],
                    "p99": sorted_all[int(n * 0.99)] if n > 100 else sorted_all[-1],
                }

            return MetricSnapshot(
                definition=self.get_definition(),
                values=stats,
                statistics=overall_stats,
            )

    def reset(self) -> None:
        """Reset histogram."""
        with self._lock:
            self._values.clear()


class Timer(Histogram):
    """Timer metric for measuring durations."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ):
        # Timer buckets in milliseconds
        buckets = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
        super().__init__(name, description, MetricUnit.MILLISECONDS, labels, buckets)

    @property
    def metric_type(self) -> MetricType:
        return MetricType.TIMER

    def time(self, **label_values: str) -> TimerContext:
        """Context manager for timing operations."""
        return TimerContext(self, label_values)


class TimerContext:
    """Context manager for Timer metric."""

    def __init__(self, timer: Timer, label_values: dict[str, str]):
        self.timer = timer
        self.label_values = label_values
        self.start_time: float | None = None

    def __enter__(self) -> TimerContext:
        import time

        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        import time

        if self.start_time is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            self.timer.observe(duration_ms, **self.label_values)


class MetricRegistry:
    """Registry for managing metrics."""

    def __init__(self):
        self._metrics: dict[str, BaseMetric] = {}
        self._lock = threading.Lock()

    def register(self, metric: BaseMetric) -> BaseMetric:
        """Register a metric."""
        with self._lock:
            if metric.name in self._metrics:
                raise ValueError(f"Metric {metric.name} already registered")
            self._metrics[metric.name] = metric
        return metric

    def get(self, name: str) -> BaseMetric | None:
        """Get a metric by name."""
        with self._lock:
            return self._metrics.get(name)

    def get_or_create(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        **kwargs: Any,
    ) -> BaseMetric:
        """Get existing metric or create new one."""
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]

            if metric_type == MetricType.COUNTER:
                metric = Counter(name, description, kwargs.get("labels"))
            elif metric_type == MetricType.GAUGE:
                metric = Gauge(name, description, kwargs.get("unit", MetricUnit.COUNT), kwargs.get("labels"))
            elif metric_type == MetricType.HISTOGRAM:
                metric = Histogram(name, description, kwargs.get("unit", MetricUnit.COUNT), kwargs.get("labels"))
            elif metric_type == MetricType.TIMER:
                metric = Timer(name, description, kwargs.get("labels"))
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")

            self._metrics[name] = metric
            return metric

    def get_all_snapshots(self) -> list[MetricSnapshot]:
        """Get snapshots of all metrics."""
        with self._lock:
            return [m.get_snapshot() for m in self._metrics.values()]

    def list_metrics(self) -> list[MetricDefinition]:
        """List all registered metrics."""
        with self._lock:
            return [m.get_definition() for m in self._metrics.values()]

    def reset_all(self) -> None:
        """Reset all metrics."""
        with self._lock:
            for metric in self._metrics.values():
                metric.reset()


# Global default registry
_default_registry = MetricRegistry()


def get_default_registry() -> MetricRegistry:
    """Get the default metric registry."""
    return _default_registry


def counter(name: str, description: str = "", labels: list[str] | None = None) -> Counter:
    """Create and register a counter metric."""
    metric = Counter(name, description, labels)
    return _default_registry.register(metric)


def gauge(
    name: str, description: str = "", unit: MetricUnit = MetricUnit.COUNT, labels: list[str] | None = None
) -> Gauge:
    """Create and register a gauge metric."""
    metric = Gauge(name, description, unit, labels)
    return _default_registry.register(metric)


def histogram(
    name: str, description: str = "", unit: MetricUnit = MetricUnit.COUNT, labels: list[str] | None = None
) -> Histogram:
    """Create and register a histogram metric."""
    metric = Histogram(name, description, unit, labels)
    return _default_registry.register(metric)


def timer(name: str, description: str = "", labels: list[str] | None = None) -> Timer:
    """Create and register a timer metric."""
    metric = Timer(name, description, labels)
    return _default_registry.register(metric)
