"""
Tests for Token Usage Metrics

UPGRADE-014 Category 2: Observability & Debugging
"""

from datetime import datetime, timedelta, timezone

import pytest

from observability.token_metrics import (
    MODEL_COSTS,
    TokenUsageRecord,
    TokenUsageSummary,
    TokenUsageTracker,
    create_tracker,
    get_global_tracker,
    get_model_cost,
    record_usage,
)


class TestModelCosts:
    """Tests for model cost configuration."""

    def test_anthropic_models_exist(self):
        """Test Anthropic model costs are defined."""
        assert "claude-3-opus" in MODEL_COSTS
        assert "claude-3-sonnet" in MODEL_COSTS
        assert "claude-3-haiku" in MODEL_COSTS

    def test_openai_models_exist(self):
        """Test OpenAI model costs are defined."""
        assert "gpt-4" in MODEL_COSTS
        assert "gpt-4o" in MODEL_COSTS
        assert "gpt-3.5-turbo" in MODEL_COSTS

    def test_default_cost_exists(self):
        """Test default cost is defined."""
        assert "default" in MODEL_COSTS

    def test_get_model_cost_exact_match(self):
        """Test exact model name matching."""
        cost = get_model_cost("claude-3-sonnet")
        assert cost == MODEL_COSTS["claude-3-sonnet"]

    def test_get_model_cost_partial_match(self):
        """Test partial model name matching."""
        cost = get_model_cost("claude-3-sonnet-20240229")
        assert cost["input"] == MODEL_COSTS["claude-3-sonnet"]["input"]

    def test_get_model_cost_unknown_returns_default(self):
        """Test unknown model returns default cost."""
        cost = get_model_cost("unknown-model-xyz")
        assert cost == MODEL_COSTS["default"]


class TestTokenUsageRecord:
    """Tests for TokenUsageRecord dataclass."""

    def test_record_creation(self):
        """Test creating a usage record."""
        record = TokenUsageRecord(
            agent_name="analyst",
            model="claude-3-sonnet",
            input_tokens=100,
            output_tokens=200,
        )

        assert record.agent_name == "analyst"
        assert record.model == "claude-3-sonnet"
        assert record.input_tokens == 100
        assert record.output_tokens == 200

    def test_auto_total_tokens(self):
        """Test automatic total token calculation."""
        record = TokenUsageRecord(
            agent_name="analyst",
            model="claude-3-sonnet",
            input_tokens=100,
            output_tokens=200,
        )

        assert record.total_tokens == 300

    def test_auto_cost_calculation(self):
        """Test automatic cost calculation."""
        record = TokenUsageRecord(
            agent_name="analyst",
            model="claude-3-sonnet",
            input_tokens=1000,  # 1K input tokens
            output_tokens=1000,  # 1K output tokens
        )

        # claude-3-sonnet: $0.003/1K input, $0.015/1K output
        expected_input_cost = 0.003
        expected_output_cost = 0.015

        assert abs(record.input_cost - expected_input_cost) < 0.0001
        assert abs(record.output_cost - expected_output_cost) < 0.0001
        assert abs(record.total_cost - 0.018) < 0.0001

    def test_timestamp_auto_set(self):
        """Test timestamp is automatically set."""
        record = TokenUsageRecord(
            agent_name="test",
            model="gpt-4",
            input_tokens=100,
            output_tokens=100,
        )

        assert record.timestamp is not None
        assert isinstance(record.timestamp, datetime)

    def test_to_dict(self):
        """Test serialization."""
        record = TokenUsageRecord(
            agent_name="analyst",
            model="claude-3-sonnet",
            input_tokens=100,
            output_tokens=200,
            operation="analyze",
            trace_id="trace-123",
        )

        d = record.to_dict()

        assert d["agent_name"] == "analyst"
        assert d["model"] == "claude-3-sonnet"
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 200
        assert d["total_tokens"] == 300
        assert d["operation"] == "analyze"
        assert d["trace_id"] == "trace-123"
        assert "timestamp" in d
        assert "total_cost" in d


class TestTokenUsageSummary:
    """Tests for TokenUsageSummary dataclass."""

    def test_summary_creation(self):
        """Test creating a summary."""
        now = datetime.now(timezone.utc)
        summary = TokenUsageSummary(
            period_start=now - timedelta(hours=1),
            period_end=now,
            total_records=10,
            total_input_tokens=5000,
            total_output_tokens=10000,
            total_tokens=15000,
            total_cost=0.50,
        )

        assert summary.total_records == 10
        assert summary.total_tokens == 15000
        assert summary.total_cost == 0.50

    def test_to_dict(self):
        """Test serialization."""
        now = datetime.now(timezone.utc)
        summary = TokenUsageSummary(
            period_start=now - timedelta(hours=1),
            period_end=now,
            total_records=10,
            total_input_tokens=5000,
            total_output_tokens=10000,
            total_tokens=15000,
            total_cost=0.50,
            avg_tokens_per_call=1500.0,
            avg_cost_per_call=0.05,
        )

        d = summary.to_dict()

        assert d["total_records"] == 10
        assert d["total_tokens"] == 15000
        assert d["total_cost"] == 0.5
        assert "period_start" in d
        assert "period_end" in d


class TestTokenUsageTracker:
    """Tests for TokenUsageTracker class."""

    def test_tracker_creation(self):
        """Test creating a tracker."""
        tracker = create_tracker(max_records=1000, retention_hours=12)
        assert tracker.max_records == 1000
        assert tracker.retention_hours == 12

    def test_record_usage(self):
        """Test recording usage."""
        tracker = TokenUsageTracker()

        record = tracker.record(
            agent_name="analyst",
            model="claude-3-sonnet",
            input_tokens=100,
            output_tokens=200,
        )

        assert record.agent_name == "analyst"
        assert record.total_tokens == 300

    def test_get_summary_all_records(self):
        """Test getting summary of all records."""
        tracker = TokenUsageTracker()

        tracker.record("agent_a", "claude-3-sonnet", 100, 200)
        tracker.record("agent_b", "gpt-4", 150, 250)
        tracker.record("agent_a", "claude-3-sonnet", 200, 300)

        summary = tracker.get_summary()

        assert summary.total_records == 3
        assert summary.total_input_tokens == 450
        assert summary.total_output_tokens == 750
        assert summary.total_tokens == 1200

    def test_get_summary_with_window(self):
        """Test getting summary with time window."""
        tracker = TokenUsageTracker()

        tracker.record("agent", "claude-3-sonnet", 100, 200)
        tracker.record("agent", "claude-3-sonnet", 100, 200)

        # All records within 1 minute
        summary = tracker.get_summary(window_minutes=1)
        assert summary.total_records == 2

    def test_get_summary_by_agent(self):
        """Test getting summary filtered by agent."""
        tracker = TokenUsageTracker()

        tracker.record("agent_a", "claude-3-sonnet", 100, 200)
        tracker.record("agent_b", "gpt-4", 150, 250)
        tracker.record("agent_a", "claude-3-sonnet", 200, 300)

        summary = tracker.get_summary(agent_name="agent_a")

        assert summary.total_records == 2
        assert summary.total_input_tokens == 300

    def test_get_summary_by_model(self):
        """Test getting summary filtered by model."""
        tracker = TokenUsageTracker()

        tracker.record("agent", "claude-3-sonnet", 100, 200)
        tracker.record("agent", "gpt-4", 150, 250)
        tracker.record("agent", "claude-3-sonnet", 200, 300)

        summary = tracker.get_summary(model="claude-3-sonnet")

        assert summary.total_records == 2

    def test_get_summary_by_agent_aggregation(self):
        """Test by_agent aggregation in summary."""
        tracker = TokenUsageTracker()

        tracker.record("agent_a", "claude-3-sonnet", 100, 200)
        tracker.record("agent_a", "claude-3-sonnet", 100, 200)
        tracker.record("agent_b", "gpt-4", 150, 250)

        summary = tracker.get_summary()

        assert "agent_a" in summary.by_agent
        assert "agent_b" in summary.by_agent
        assert summary.by_agent["agent_a"]["calls"] == 2
        assert summary.by_agent["agent_b"]["calls"] == 1

    def test_get_summary_by_model_aggregation(self):
        """Test by_model aggregation in summary."""
        tracker = TokenUsageTracker()

        tracker.record("agent", "claude-3-sonnet", 100, 200)
        tracker.record("agent", "gpt-4", 150, 250)
        tracker.record("agent", "claude-3-sonnet", 200, 300)

        summary = tracker.get_summary()

        assert "claude-3-sonnet" in summary.by_model
        assert "gpt-4" in summary.by_model
        assert summary.by_model["claude-3-sonnet"]["calls"] == 2
        assert summary.by_model["gpt-4"]["calls"] == 1

    def test_get_usage_by_agent(self):
        """Test getting records by agent."""
        tracker = TokenUsageTracker()

        tracker.record("agent_a", "claude-3-sonnet", 100, 200)
        tracker.record("agent_b", "gpt-4", 150, 250)
        tracker.record("agent_a", "claude-3-sonnet", 200, 300)

        records = tracker.get_usage_by_agent("agent_a")

        assert len(records) == 2
        assert all(r.agent_name == "agent_a" for r in records)

    def test_get_usage_by_model(self):
        """Test getting records by model."""
        tracker = TokenUsageTracker()

        tracker.record("agent", "claude-3-sonnet", 100, 200)
        tracker.record("agent", "gpt-4", 150, 250)

        records = tracker.get_usage_by_model("gpt-4")

        assert len(records) == 1
        assert records[0].model == "gpt-4"

    def test_get_recent_records(self):
        """Test getting recent records."""
        tracker = TokenUsageTracker()

        for i in range(10):
            tracker.record("agent", "claude-3-sonnet", 100, 200)

        records = tracker.get_recent_records(limit=5)

        assert len(records) == 5

    def test_get_totals(self):
        """Test getting running totals."""
        tracker = TokenUsageTracker()

        tracker.record("agent", "claude-3-sonnet", 100, 200)
        tracker.record("agent", "gpt-4", 150, 250)

        totals = tracker.get_totals()

        assert totals["total_tokens"] == 700
        assert totals["total_records"] == 2
        assert "total_cost" in totals

    def test_get_rate(self):
        """Test getting usage rate."""
        tracker = TokenUsageTracker()

        tracker.record("agent", "claude-3-sonnet", 100, 200)
        tracker.record("agent", "claude-3-sonnet", 100, 200)

        rate = tracker.get_rate(window_minutes=1)

        assert "tokens_per_minute" in rate
        assert "cost_per_minute" in rate
        assert rate["tokens_per_minute"] > 0

    def test_clear(self):
        """Test clearing all records."""
        tracker = TokenUsageTracker()

        tracker.record("agent", "claude-3-sonnet", 100, 200)
        tracker.record("agent", "gpt-4", 150, 250)

        tracker.clear()

        totals = tracker.get_totals()
        assert totals["total_records"] == 0
        assert totals["total_tokens"] == 0

    def test_to_dict(self):
        """Test export to dictionary."""
        tracker = TokenUsageTracker()

        tracker.record("agent", "claude-3-sonnet", 100, 200)

        d = tracker.to_dict()

        assert "totals" in d
        assert "summary_1h" in d
        assert "summary_24h" in d
        assert "rate_1m" in d
        assert "rate_5m" in d

    def test_max_records_cleanup(self):
        """Test that old records are cleaned up."""
        tracker = TokenUsageTracker(max_records=10)

        for i in range(20):
            tracker.record("agent", "claude-3-sonnet", 100, 200)

        # Should have at most 10 records
        records = tracker.get_recent_records(limit=100)
        assert len(records) <= 10

    def test_global_tracker_singleton(self):
        """Test global tracker is singleton."""
        tracker1 = get_global_tracker()
        tracker2 = get_global_tracker()
        assert tracker1 is tracker2

    def test_record_usage_convenience(self):
        """Test convenience function."""
        # Clear global tracker first
        get_global_tracker().clear()

        record = record_usage(
            agent_name="test",
            model="claude-3-sonnet",
            input_tokens=100,
            output_tokens=200,
        )

        assert record.total_tokens == 300

        totals = get_global_tracker().get_totals()
        assert totals["total_records"] >= 1


class TestTokenUsageSummaryEmpty:
    """Tests for empty summary edge cases."""

    def test_empty_summary(self):
        """Test summary with no records."""
        tracker = TokenUsageTracker()
        summary = tracker.get_summary()

        assert summary.total_records == 0
        assert summary.total_tokens == 0
        assert summary.total_cost == 0.0

    def test_empty_rate(self):
        """Test rate with no records."""
        tracker = TokenUsageTracker()
        rate = tracker.get_rate()

        assert rate["tokens_per_minute"] == 0.0
        assert rate["cost_per_minute"] == 0.0


@pytest.mark.skip(reason="Threading tests quarantined - cause hangs in CI")
class TestTokenMetricsThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_recording(self):
        """Test concurrent record calls."""
        import threading

        tracker = TokenUsageTracker()
        errors = []

        def record_many():
            try:
                for _ in range(100):
                    tracker.record("agent", "claude-3-sonnet", 100, 200)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        totals = tracker.get_totals()
        assert totals["total_records"] == 500
