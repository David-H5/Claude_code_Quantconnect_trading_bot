"""
Tests for Metrics Collection Module

UPGRADE-015 Phase 6: Observability Setup
Updated to match actual API implementation.

Tests cover:
- Counter metrics
- Gauge metrics
- Histogram metrics
- Timer metrics
- Registry operations
"""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricType,
    Timer,
    create_metrics_registry,
    get_default_registry,
)


# =============================================================================
# Counter Tests
# =============================================================================


class TestCounter:
    """Test counter metrics."""

    def test_counter_basic(self):
        """Test basic counter operations."""
        counter = Counter("test_counter", "Test counter")

        counter.inc()
        assert counter.get() == 1.0

        counter.inc(5)
        assert counter.get() == 6.0

    def test_counter_with_labels(self):
        """Test counter with labels."""
        counter = Counter("test_counter", labels=["method", "status"])

        counter.inc(method="GET", status="200")
        counter.inc(method="POST", status="200")
        counter.inc(method="GET", status="200")

        assert counter.get(method="GET", status="200") == 2.0
        assert counter.get(method="POST", status="200") == 1.0
        assert counter.get(method="PUT", status="200") == 0.0

    def test_counter_get_snapshot(self):
        """Test getting counter snapshot."""
        counter = Counter("test_counter", labels=["method"])

        counter.inc(method="GET")
        counter.inc(method="POST")
        counter.inc(method="GET")

        snapshot = counter.get_snapshot()
        assert len(snapshot.values) == 2
        assert snapshot.values[("GET",)] == 2.0
        assert snapshot.values[("POST",)] == 1.0

    def test_counter_reset(self):
        """Test counter reset."""
        counter = Counter("test_counter")
        counter.inc(5)
        assert counter.get() == 5.0

        counter.reset()
        assert counter.get() == 0.0

    def test_counter_negative_increment_fails(self):
        """Test that negative increment raises error."""
        counter = Counter("test_counter")
        with pytest.raises(ValueError):
            counter.inc(-1)


# =============================================================================
# Gauge Tests
# =============================================================================


class TestGauge:
    """Test gauge metrics."""

    def test_gauge_set(self):
        """Test gauge set operation."""
        gauge = Gauge("test_gauge", "Test gauge")

        gauge.set(100.0)
        assert gauge.get() == 100.0

        gauge.set(50.0)
        assert gauge.get() == 50.0

    def test_gauge_inc_dec(self):
        """Test gauge increment/decrement."""
        gauge = Gauge("test_gauge")

        gauge.set(10.0)
        gauge.inc(5)
        assert gauge.get() == 15.0

        gauge.dec(3)
        assert gauge.get() == 12.0

    def test_gauge_with_labels(self):
        """Test gauge with labels."""
        gauge = Gauge("test_gauge", labels=["host"])

        gauge.set(100.0, host="host1")
        gauge.set(200.0, host="host2")

        assert gauge.get(host="host1") == 100.0
        assert gauge.get(host="host2") == 200.0

    def test_gauge_reset(self):
        """Test gauge reset."""
        gauge = Gauge("test_gauge")
        gauge.set(100.0)
        assert gauge.get() == 100.0

        gauge.reset()
        assert gauge.get() == 0.0


# =============================================================================
# Histogram Tests
# =============================================================================


class TestHistogram:
    """Test histogram metrics."""

    def test_histogram_observe(self):
        """Test histogram observations."""
        histogram = Histogram("test_histogram")

        histogram.observe(0.1)
        histogram.observe(0.2)
        histogram.observe(0.3)

        stats = histogram.get_statistics()
        assert stats["count"] == 3
        assert stats["min"] == 0.1
        assert stats["max"] == 0.3
        assert stats["avg"] == pytest.approx(0.2, rel=0.01)

    def test_histogram_with_labels(self):
        """Test histogram with labels."""
        histogram = Histogram("test_histogram", labels=["endpoint"])

        histogram.observe(0.1, endpoint="/api/v1")
        histogram.observe(0.2, endpoint="/api/v1")
        histogram.observe(0.5, endpoint="/api/v2")

        stats_v1 = histogram.get_statistics(endpoint="/api/v1")
        stats_v2 = histogram.get_statistics(endpoint="/api/v2")

        assert stats_v1["count"] == 2
        assert stats_v2["count"] == 1

    def test_histogram_empty(self):
        """Test histogram with no observations."""
        histogram = Histogram("test_histogram")

        stats = histogram.get_statistics()
        assert stats == {}

    def test_histogram_snapshot(self):
        """Test histogram snapshot with statistics."""
        histogram = Histogram("test_histogram")

        histogram.observe(1.0)
        histogram.observe(2.0)
        histogram.observe(3.0)

        snapshot = histogram.get_snapshot()
        assert snapshot.statistics is not None
        assert snapshot.statistics["count"] == 3
        assert snapshot.statistics["min"] == 1.0
        assert snapshot.statistics["max"] == 3.0

    def test_histogram_reset(self):
        """Test histogram reset."""
        histogram = Histogram("test_histogram")
        histogram.observe(1.0)
        assert histogram.get_statistics()["count"] == 1

        histogram.reset()
        assert histogram.get_statistics() == {}


# =============================================================================
# Timer Tests
# =============================================================================


class TestTimer:
    """Test timer metrics."""

    def test_timer_context_manager(self):
        """Test timer as context manager."""
        timer = Timer("test_timer")

        import time

        with timer.time():
            time.sleep(0.01)  # 10ms

        stats = timer.get_statistics()
        assert stats["count"] == 1
        assert stats["min"] >= 10  # At least 10ms (timer records in ms)

    def test_timer_observe(self):
        """Test timer observe method (in milliseconds)."""
        timer = Timer("test_timer")

        timer.observe(500)  # 500ms
        timer.observe(1000)  # 1000ms
        timer.observe(1500)  # 1500ms

        stats = timer.get_statistics()
        assert stats["count"] == 3
        assert stats["avg"] == pytest.approx(1000.0, rel=0.01)

    def test_timer_with_labels(self):
        """Test timer with labels."""
        timer = Timer("test_timer", labels=["operation"])

        timer.observe(100, operation="read")
        timer.observe(200, operation="write")

        read_stats = timer.get_statistics(operation="read")
        write_stats = timer.get_statistics(operation="write")

        assert read_stats["count"] == 1
        assert write_stats["count"] == 1


# =============================================================================
# Registry Tests
# =============================================================================


class TestMetricRegistry:
    """Test metrics registry."""

    def test_registry_get_or_create_counter(self):
        """Test getting counter from registry."""
        registry = create_metrics_registry()

        counter = registry.get_or_create("test_counter", MetricType.COUNTER, "Test")
        counter.inc()

        # Should return same counter
        same_counter = registry.get_or_create("test_counter", MetricType.COUNTER)
        assert same_counter.get() == 1.0

    def test_registry_get_or_create_gauge(self):
        """Test getting gauge from registry."""
        registry = create_metrics_registry()

        gauge = registry.get_or_create("test_gauge", MetricType.GAUGE)
        gauge.set(42.0)

        same_gauge = registry.get_or_create("test_gauge", MetricType.GAUGE)
        assert same_gauge.get() == 42.0

    def test_registry_get_or_create_histogram(self):
        """Test getting histogram from registry."""
        registry = create_metrics_registry()

        histogram = registry.get_or_create("test_histogram", MetricType.HISTOGRAM)
        histogram.observe(1.0)

        same_histogram = registry.get_or_create("test_histogram", MetricType.HISTOGRAM)
        assert same_histogram.get_statistics()["count"] == 1

    def test_registry_get_or_create_timer(self):
        """Test getting timer from registry."""
        registry = create_metrics_registry()

        timer = registry.get_or_create("test_timer", MetricType.TIMER)
        timer.observe(1.0)

        same_timer = registry.get_or_create("test_timer", MetricType.TIMER)
        assert same_timer.get_statistics()["count"] == 1

    def test_registry_register(self):
        """Test registering a metric."""
        registry = create_metrics_registry()

        counter = Counter("my_counter", "My counter")
        registry.register(counter)

        retrieved = registry.get("my_counter")
        assert retrieved is counter

    def test_registry_duplicate_registration_fails(self):
        """Test that duplicate registration raises error."""
        registry = create_metrics_registry()

        counter1 = Counter("duplicate_counter", "First")
        counter2 = Counter("duplicate_counter", "Second")

        registry.register(counter1)
        with pytest.raises(ValueError):
            registry.register(counter2)

    def test_registry_get_all_snapshots(self):
        """Test getting all metric snapshots."""
        registry = create_metrics_registry()

        registry.get_or_create("requests", MetricType.COUNTER).inc(10)
        registry.get_or_create("temperature", MetricType.GAUGE).set(25.5)

        snapshots = registry.get_all_snapshots()
        assert len(snapshots) == 2

    def test_registry_list_metrics(self):
        """Test listing registered metrics."""
        registry = create_metrics_registry()

        registry.get_or_create("counter1", MetricType.COUNTER, "Counter 1")
        registry.get_or_create("gauge1", MetricType.GAUGE, "Gauge 1")

        definitions = registry.list_metrics()
        assert len(definitions) == 2

        names = [d.name for d in definitions]
        assert "counter1" in names
        assert "gauge1" in names

    def test_registry_reset_all(self):
        """Test resetting all metrics."""
        registry = create_metrics_registry()

        counter = registry.get_or_create("counter", MetricType.COUNTER)
        counter.inc(100)

        gauge = registry.get_or_create("gauge", MetricType.GAUGE)
        gauge.set(50)

        registry.reset_all()

        assert counter.get() == 0.0
        assert gauge.get() == 0.0


# =============================================================================
# Default Registry Tests
# =============================================================================


class TestDefaultRegistry:
    """Test default registry."""

    def test_get_default_registry(self):
        """Test getting default registry."""
        registry = get_default_registry()
        assert registry is not None

        # Should return same instance
        same_registry = get_default_registry()
        assert registry is same_registry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
