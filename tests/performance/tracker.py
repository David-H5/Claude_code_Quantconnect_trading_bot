"""
Performance Regression Tracker

Tracks performance metrics across test runs to detect regressions.

UPGRADE-015: Advanced Test Framework - Performance Testing
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import pytest


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    name: str
    duration_ms: float
    timestamp: str
    threshold_ms: float
    iterations: int = 1
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "threshold_ms": self.threshold_ms,
            "iterations": self.iterations,
            "metadata": self.metadata,
        }


class PerformanceTracker:
    """
    Track performance metrics and detect regressions.

    Usage:
        tracker = PerformanceTracker()

        with tracker.measure("order_placement"):
            place_order(order)

        # Or as decorator
        @tracker.track("risk_calculation", threshold_ms=50)
        def calculate_risk():
            ...
    """

    def __init__(
        self,
        history_file: Path | str = "tests/performance/history.json",
        regression_threshold_pct: float = 0.20,
    ):
        self.history_file = Path(history_file)
        self.regression_threshold_pct = regression_threshold_pct
        self.current_metrics: list[PerformanceMetric] = []
        self.history: list[dict] = []
        self._load_history()

    def _load_history(self):
        """Load historical metrics."""
        if self.history_file.exists():
            try:
                self.history = json.loads(self.history_file.read_text())
            except (json.JSONDecodeError, OSError):
                self.history = []

    def save(self):
        """Save current metrics to history."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        for metric in self.current_metrics:
            self.history.append(metric.to_dict())

        # Keep last 1000 entries
        self.history = self.history[-1000:]
        self.history_file.write_text(json.dumps(self.history, indent=2))

    def record(
        self,
        name: str,
        duration_ms: float,
        threshold_ms: float = 100,
        iterations: int = 1,
        **metadata,
    ) -> PerformanceMetric:
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            duration_ms=duration_ms,
            timestamp=datetime.now().isoformat(),
            threshold_ms=threshold_ms,
            iterations=iterations,
            metadata=metadata,
        )
        self.current_metrics.append(metric)

        # Check for regression
        regression = self._check_regression(metric)
        if regression:
            print(f"\n{'='*60}")
            print(f"PERFORMANCE REGRESSION DETECTED: {name}")
            print(f"  Previous avg: {regression['previous_avg']:.2f}ms")
            print(f"  Current:      {duration_ms:.2f}ms")
            print(f"  Regression:   {regression['pct_change']:.1f}%")
            print(f"{'='*60}\n")

        return metric

    def _check_regression(self, metric: PerformanceMetric) -> dict | None:
        """Check if metric shows regression vs history."""
        previous = [
            h for h in self.history
            if h["name"] == metric.name
        ][-10:]  # Last 10 measurements

        if len(previous) < 3:
            return None

        previous_avg = sum(h["duration_ms"] for h in previous) / len(previous)
        pct_change = (metric.duration_ms - previous_avg) / previous_avg

        if pct_change > self.regression_threshold_pct:
            return {
                "previous_avg": previous_avg,
                "current": metric.duration_ms,
                "pct_change": pct_change * 100,
            }
        return None

    def measure(self, name: str, threshold_ms: float = 100):
        """Context manager for measuring execution time."""
        return _PerformanceMeasureContext(self, name, threshold_ms)

    def track(
        self,
        name: str,
        threshold_ms: float = 100,
        iterations: int = 1,
    ):
        """Decorator for tracking function performance."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000

                self.record(
                    name=name,
                    duration_ms=elapsed_ms,
                    threshold_ms=threshold_ms,
                    iterations=iterations,
                    function=func.__name__,
                )
                return result
            return wrapper
        return decorator

    def get_summary(self) -> dict:
        """Get summary of current session metrics."""
        if not self.current_metrics:
            return {"count": 0, "metrics": []}

        return {
            "count": len(self.current_metrics),
            "total_ms": sum(m.duration_ms for m in self.current_metrics),
            "slowest": max(self.current_metrics, key=lambda m: m.duration_ms).name,
            "metrics": [m.to_dict() for m in self.current_metrics],
        }


class _PerformanceMeasureContext:
    """Context manager for performance measurement."""

    def __init__(self, tracker: PerformanceTracker, name: str, threshold_ms: float):
        self.tracker = tracker
        self.name = name
        self.threshold_ms = threshold_ms
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        self.tracker.record(
            name=self.name,
            duration_ms=elapsed_ms,
            threshold_ms=self.threshold_ms,
        )
        return False


# Global tracker instance
_global_tracker: PerformanceTracker | None = None


def get_performance_tracker() -> PerformanceTracker:
    """Get global performance tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker


@pytest.fixture(scope="session")
def perf_tracker():
    """Pytest fixture for performance tracking."""
    tracker = PerformanceTracker()
    yield tracker
    tracker.save()


def benchmark(
    name: str,
    threshold_ms: float = 100,
    iterations: int = 1,
):
    """
    Decorator to benchmark a test function.

    Usage:
        @benchmark("order_placement", threshold_ms=50)
        def test_order_placement_speed():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracker = get_performance_tracker()

            if iterations > 1:
                start = time.perf_counter()
                for _ in range(iterations):
                    result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
            else:
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000

            tracker.record(
                name=name,
                duration_ms=elapsed_ms,
                threshold_ms=threshold_ms,
                iterations=iterations,
            )

            # Fail test if over threshold
            assert elapsed_ms < threshold_ms, \
                f"Performance regression: {name} took {elapsed_ms:.2f}ms (threshold: {threshold_ms}ms)"

            return result
        return wrapper
    return decorator


__all__ = [
    "PerformanceMetric",
    "PerformanceTracker",
    "get_performance_tracker",
    "perf_tracker",
    "benchmark",
]
