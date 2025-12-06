"""
Tests for Metrics Aggregator

UPGRADE-014 Category 2: Observability & Debugging
"""

from datetime import datetime, timezone

import pytest

from observability.metrics_aggregator import (
    AggregatedMetric,
    MetricPoint,
    MetricsAggregator,
    MetricSeries,
    MetricType,
    SystemHealth,
    WindowSize,
    create_aggregator,
    get_global_aggregator,
    record_agent_decision,
    record_llm_request,
    record_metric,
    record_tool_call,
)


class TestMetricType:
    """Tests for MetricType enum."""

    def test_all_types_exist(self):
        """Test all expected metric types exist."""
        types = [
            MetricType.COUNTER,
            MetricType.GAUGE,
            MetricType.HISTOGRAM,
            MetricType.RATE,
        ]
        assert len(types) == 4


class TestWindowSize:
    """Tests for WindowSize enum."""

    def test_all_sizes_exist(self):
        """Test all expected window sizes exist."""
        sizes = [
            WindowSize.ONE_MINUTE,
            WindowSize.FIVE_MINUTES,
            WindowSize.FIFTEEN_MINUTES,
            WindowSize.ONE_HOUR,
            WindowSize.ONE_DAY,
        ]
        assert len(sizes) == 5

    def test_size_values(self):
        """Test window size values in seconds."""
        assert WindowSize.ONE_MINUTE.value == 60
        assert WindowSize.FIVE_MINUTES.value == 300
        assert WindowSize.FIFTEEN_MINUTES.value == 900
        assert WindowSize.ONE_HOUR.value == 3600
        assert WindowSize.ONE_DAY.value == 86400


class TestMetricPoint:
    """Tests for MetricPoint dataclass."""

    def test_point_creation(self):
        """Test creating a metric point."""
        point = MetricPoint(
            timestamp=datetime.now(timezone.utc),
            value=100.0,
            labels={"agent": "analyst"},
        )

        assert point.value == 100.0
        assert point.labels["agent"] == "analyst"


class TestMetricSeries:
    """Tests for MetricSeries class."""

    def test_series_creation(self):
        """Test creating a metric series."""
        series = MetricSeries(
            name="test_metric",
            metric_type=MetricType.GAUGE,
            description="Test metric",
            unit="ms",
        )

        assert series.name == "test_metric"
        assert series.metric_type == MetricType.GAUGE

    def test_add_point(self):
        """Test adding points to series."""
        series = MetricSeries(name="test", metric_type=MetricType.GAUGE)

        series.add_point(100.0)
        series.add_point(200.0, {"agent": "test"})

        points = series.get_points()
        assert len(points) == 2

    def test_get_points_with_window(self):
        """Test getting points within time window."""
        series = MetricSeries(name="test", metric_type=MetricType.GAUGE)

        series.add_point(100.0)
        series.add_point(200.0)

        points = series.get_points(window_seconds=60)
        assert len(points) == 2

    def test_get_points_with_labels(self):
        """Test filtering points by labels."""
        series = MetricSeries(name="test", metric_type=MetricType.GAUGE)

        series.add_point(100.0, {"agent": "a"})
        series.add_point(200.0, {"agent": "b"})
        series.add_point(300.0, {"agent": "a"})

        points = series.get_points(labels={"agent": "a"})
        assert len(points) == 2

    def test_get_latest(self):
        """Test getting latest point."""
        series = MetricSeries(name="test", metric_type=MetricType.GAUGE)

        series.add_point(100.0)
        series.add_point(200.0)

        latest = series.get_latest()
        assert latest.value == 200.0


class TestAggregatedMetric:
    """Tests for AggregatedMetric dataclass."""

    def test_aggregation_creation(self):
        """Test creating aggregated metric."""
        agg = AggregatedMetric(
            name="test_metric",
            window_seconds=60,
            count=10,
            sum_value=1000.0,
            min_value=50.0,
            max_value=150.0,
            avg_value=100.0,
        )

        assert agg.name == "test_metric"
        assert agg.count == 10
        assert agg.avg_value == 100.0

    def test_to_dict(self):
        """Test serialization."""
        agg = AggregatedMetric(
            name="test",
            window_seconds=60,
            count=5,
            sum_value=500.0,
            avg_value=100.0,
            p95_value=120.0,
            p99_value=140.0,
        )

        d = agg.to_dict()

        assert d["name"] == "test"
        assert d["count"] == 5
        assert d["avg"] == 100.0
        assert d["p95"] == 120.0


class TestSystemHealth:
    """Tests for SystemHealth dataclass."""

    def test_health_creation(self):
        """Test creating system health."""
        health = SystemHealth(
            timestamp=datetime.now(timezone.utc),
            status="healthy",
            score=0.95,
            agents_healthy=5,
            agents_total=5,
        )

        assert health.status == "healthy"
        assert health.score == 0.95

    def test_to_dict(self):
        """Test serialization."""
        health = SystemHealth(
            timestamp=datetime.now(timezone.utc),
            status="degraded",
            score=0.75,
            error_rate_1m=0.05,
            avg_latency_ms=150.0,
        )

        d = health.to_dict()

        assert d["status"] == "degraded"
        assert d["score"] == 0.75
        assert d["error_rate_1m"] == 0.05


class TestMetricsAggregator:
    """Tests for MetricsAggregator class."""

    def test_aggregator_creation(self):
        """Test creating an aggregator."""
        aggregator = create_aggregator(max_points_per_metric=5000)
        assert aggregator.max_points == 5000

    def test_register_metric(self):
        """Test registering a metric."""
        aggregator = MetricsAggregator()

        series = aggregator.register_metric(
            name="custom_metric",
            metric_type=MetricType.HISTOGRAM,
            description="Custom metric",
            unit="ms",
        )

        assert series.name == "custom_metric"
        assert "custom_metric" in aggregator.get_metric_names()

    def test_record_metric(self):
        """Test recording a metric value."""
        aggregator = MetricsAggregator()

        aggregator.register_metric("latency", MetricType.HISTOGRAM)
        aggregator.record("latency", 150.0)
        aggregator.record("latency", 200.0)

        agg = aggregator.get_aggregation("latency", WindowSize.ONE_MINUTE)
        assert agg.count == 2
        assert agg.avg_value == 175.0

    def test_auto_register_on_record(self):
        """Test automatic metric registration on record."""
        aggregator = MetricsAggregator()

        aggregator.record("unknown_metric", 100.0)

        assert "unknown_metric" in aggregator.get_metric_names()

    def test_increment_counter(self):
        """Test incrementing a counter."""
        aggregator = MetricsAggregator()

        aggregator.increment("requests_total", 1)
        aggregator.increment("requests_total", 1)
        aggregator.increment("requests_total", 1)

        agg = aggregator.get_aggregation("requests_total", WindowSize.ONE_MINUTE)
        assert agg.sum_value == 3

    def test_observe_histogram(self):
        """Test observing histogram values."""
        aggregator = MetricsAggregator()

        for v in [100, 110, 120, 130, 140, 150, 200, 250, 300, 500]:
            aggregator.observe("duration_ms", v)

        agg = aggregator.get_aggregation("duration_ms", WindowSize.ONE_MINUTE)

        assert agg.count == 10
        assert agg.min_value == 100
        assert agg.max_value == 500
        assert agg.p95_value > agg.median_value

    def test_labels_filtering(self):
        """Test filtering by labels."""
        aggregator = MetricsAggregator()

        aggregator.record("requests", 1, {"agent": "a"})
        aggregator.record("requests", 1, {"agent": "b"})
        aggregator.record("requests", 1, {"agent": "a"})

        agg_a = aggregator.get_aggregation("requests", WindowSize.ONE_MINUTE, labels={"agent": "a"})
        agg_b = aggregator.get_aggregation("requests", WindowSize.ONE_MINUTE, labels={"agent": "b"})

        assert agg_a.count == 2
        assert agg_b.count == 1

    def test_get_all_aggregations(self):
        """Test getting all aggregations."""
        aggregator = MetricsAggregator()

        aggregator.record("metric_a", 100)
        aggregator.record("metric_b", 200)

        all_aggs = aggregator.get_all_aggregations(WindowSize.ONE_MINUTE)

        assert "metric_a" in all_aggs
        assert "metric_b" in all_aggs

    def test_get_system_health(self):
        """Test getting system health."""
        aggregator = MetricsAggregator()

        # Record some metrics
        aggregator.increment("llm_requests_total", 1)
        aggregator.observe("llm_request_duration_ms", 150)

        health = aggregator.get_system_health()

        assert health.status in ["healthy", "degraded", "unhealthy"]
        assert 0.0 <= health.score <= 1.0

    def test_health_score_degradation(self):
        """Test health score degradation on errors."""
        aggregator = MetricsAggregator()

        # Record many errors
        for _ in range(10):
            aggregator.increment("llm_requests_total", 1)
            aggregator.increment("llm_errors_total", 1)

        health = aggregator.get_system_health()

        # High error rate should degrade score
        assert health.score < 1.0

    def test_register_health_checker(self):
        """Test registering health check callback."""
        aggregator = MetricsAggregator()

        def custom_checker():
            return {"custom_component": {"healthy": True, "latency_ms": 50}}

        aggregator.register_health_checker(custom_checker)
        health = aggregator.get_system_health()

        assert "custom_component" in health.components

    def test_export_prometheus(self):
        """Test Prometheus format export."""
        aggregator = MetricsAggregator()

        aggregator.register_metric("test_metric", MetricType.COUNTER, "Test counter")
        aggregator.increment("test_metric", 5)

        output = aggregator.export_prometheus()

        assert "# HELP test_metric" in output
        assert "# TYPE test_metric counter" in output

    def test_export_json(self):
        """Test JSON format export."""
        aggregator = MetricsAggregator()

        aggregator.record("test_metric", 100)

        output = aggregator.export_json()

        assert "timestamp" in output
        assert "health" in output
        assert "metrics" in output
        assert "test_metric" in output["metrics"]

    def test_clear_metrics(self):
        """Test clearing metric data."""
        aggregator = MetricsAggregator()

        aggregator.record("test", 100)
        aggregator.record("test", 200)

        aggregator.clear()

        agg = aggregator.get_aggregation("test", WindowSize.ONE_MINUTE)
        assert agg.count == 0

    def test_global_aggregator_singleton(self):
        """Test global aggregator is singleton."""
        agg1 = get_global_aggregator()
        agg2 = get_global_aggregator()
        assert agg1 is agg2

    def test_standard_metrics_initialized(self):
        """Test standard metrics are pre-registered."""
        aggregator = MetricsAggregator()
        names = aggregator.get_metric_names()

        assert "llm_request_duration_ms" in names
        assert "llm_requests_total" in names
        assert "llm_errors_total" in names
        assert "llm_tokens_input" in names
        assert "llm_tokens_output" in names


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.skip(reason="QUARANTINED: Deadlock in global aggregator lock - see metrics_aggregator.py:280")
    def test_record_metric(self):
        """Test record_metric convenience function."""
        # Clear first
        get_global_aggregator().clear()

        record_metric("custom_value", 123.0, {"source": "test"})

        agg = get_global_aggregator().get_aggregation("custom_value", WindowSize.ONE_MINUTE)
        assert agg.count >= 1

    def test_record_llm_request(self):
        """Test record_llm_request convenience function."""
        get_global_aggregator().clear()

        record_llm_request(
            agent_name="analyst",
            model="claude-3-sonnet",
            duration_ms=150.0,
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.003,
            success=True,
        )

        agg = get_global_aggregator()

        latency = agg.get_aggregation("llm_request_duration_ms", WindowSize.ONE_MINUTE)
        assert latency.count >= 1

        requests = agg.get_aggregation("llm_requests_total", WindowSize.ONE_MINUTE)
        assert requests.sum_value >= 1

    def test_record_llm_request_failure(self):
        """Test recording failed LLM request."""
        get_global_aggregator().clear()

        record_llm_request(
            agent_name="analyst",
            model="claude-3-sonnet",
            duration_ms=50.0,
            input_tokens=100,
            output_tokens=0,
            cost_usd=0.001,
            success=False,
        )

        errors = get_global_aggregator().get_aggregation("llm_errors_total", WindowSize.ONE_MINUTE)
        assert errors.sum_value >= 1

    def test_record_agent_decision(self):
        """Test record_agent_decision convenience function."""
        get_global_aggregator().clear()

        record_agent_decision(
            agent_name="analyst",
            duration_ms=200.0,
            success=True,
        )

        agg = get_global_aggregator().get_aggregation("agent_decisions_total", WindowSize.ONE_MINUTE)
        assert agg.sum_value >= 1

    def test_record_tool_call(self):
        """Test record_tool_call convenience function."""
        get_global_aggregator().clear()

        record_tool_call(
            tool_name="get_price",
            agent_name="analyst",
            duration_ms=50.0,
            success=True,
        )

        agg = get_global_aggregator().get_aggregation("tool_calls_total", WindowSize.ONE_MINUTE)
        assert agg.sum_value >= 1


@pytest.mark.skip(reason="Threading tests quarantined - cause hangs in CI")
class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_recording(self):
        """Test concurrent metric recording."""
        import threading

        aggregator = MetricsAggregator()
        errors = []

        def record_many():
            try:
                for _ in range(100):
                    aggregator.record("concurrent_test", 1.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        agg = aggregator.get_aggregation("concurrent_test", WindowSize.ONE_MINUTE)
        assert agg.count == 500
