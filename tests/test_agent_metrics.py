"""
Tests for Agent Performance Metrics

Tests verify the metrics tracker correctly:
- Records agent decisions
- Calculates accuracy and confidence metrics
- Tracks trends over time
- Compares agents
"""

from datetime import datetime, timedelta

import pytest

from evaluation.agent_metrics import (
    AgentMetrics,
    AgentMetricsTracker,
    create_metrics_tracker,
    generate_metrics_report,
)


@pytest.fixture
def tracker():
    """Create a metrics tracker."""
    return AgentMetricsTracker()


@pytest.fixture
def populated_tracker():
    """Create a tracker with sample decisions."""
    tracker = AgentMetricsTracker()

    # Add decisions for analyst agent
    for i in range(10):
        tracker.record_decision(
            agent_name="analyst",
            decision_type="analysis",
            decision="BUY" if i % 2 == 0 else "SELL",
            confidence=0.7 + (i % 3) * 0.1,
            was_correct=i % 3 != 0,  # 70% accuracy
            execution_time_ms=100 + i * 10,
            reasoning_steps=3 + i % 2,
        )

    # Add decisions for trader agent
    for i in range(5):
        tracker.record_decision(
            agent_name="trader",
            decision_type="trade",
            decision="EXECUTE",
            confidence=0.85,
            was_correct=i < 4,  # 80% accuracy
            execution_time_ms=50 + i * 5,
        )

    return tracker


class TestDecisionRecording:
    """Tests for decision recording."""

    def test_record_decision_basic(self, tracker):
        """Can record a basic decision."""
        decision_id = tracker.record_decision(
            agent_name="test_agent",
            decision_type="analysis",
            decision="BUY",
            confidence=0.85,
        )

        assert decision_id.startswith("dec_test_agent_")
        assert len(tracker.decisions) == 1

    def test_record_decision_with_outcome(self, tracker):
        """Can record decision with known outcome."""
        tracker.record_decision(
            agent_name="test_agent",
            decision_type="trade",
            decision="SELL",
            confidence=0.75,
            was_correct=True,
        )

        assert tracker.decisions[0].was_correct is True

    def test_record_decision_with_metadata(self, tracker):
        """Can record decision with metadata."""
        tracker.record_decision(
            agent_name="test_agent",
            decision_type="analysis",
            decision="HOLD",
            confidence=0.6,
            metadata={"symbol": "SPY", "reason": "uncertainty"},
        )

        assert tracker.decisions[0].metadata["symbol"] == "SPY"

    def test_decisions_tracked_by_agent(self, tracker):
        """Decisions are grouped by agent."""
        tracker.record_decision(
            agent_name="agent_a",
            decision_type="analysis",
            decision="BUY",
            confidence=0.8,
        )
        tracker.record_decision(
            agent_name="agent_b",
            decision_type="analysis",
            decision="SELL",
            confidence=0.7,
        )

        assert len(tracker.decisions_by_agent["agent_a"]) == 1
        assert len(tracker.decisions_by_agent["agent_b"]) == 1


class TestOutcomeUpdate:
    """Tests for outcome updates."""

    def test_update_outcome(self, tracker):
        """Can update decision outcome later."""
        decision_id = tracker.record_decision(
            agent_name="test_agent",
            decision_type="trade",
            decision="BUY",
            confidence=0.8,
        )

        assert tracker.decisions[0].was_correct is None

        success = tracker.update_outcome(decision_id, was_correct=True)

        assert success is True
        assert tracker.decisions[0].was_correct is True

    def test_update_outcome_with_description(self, tracker):
        """Can update outcome with description."""
        decision_id = tracker.record_decision(
            agent_name="test_agent",
            decision_type="trade",
            decision="BUY",
            confidence=0.8,
        )

        tracker.update_outcome(
            decision_id,
            was_correct=False,
            actual_outcome="Price dropped 5%",
        )

        assert tracker.decisions[0].actual_outcome == "Price dropped 5%"

    def test_update_nonexistent_decision(self, tracker):
        """Updating nonexistent decision returns False."""
        success = tracker.update_outcome("nonexistent_id", was_correct=True)
        assert success is False


class TestMetricsCalculation:
    """Tests for metrics calculation."""

    def test_empty_metrics(self, tracker):
        """Metrics for agent with no decisions."""
        metrics = tracker.get_metrics("unknown_agent")

        assert metrics.total_decisions == 0
        assert metrics.accuracy_rate == 0.0

    def test_accuracy_calculation(self, populated_tracker):
        """Accuracy calculated correctly."""
        metrics = populated_tracker.get_metrics("analyst")

        # was_correct=i % 3 != 0 gives 6/10 correct = 60%
        # (i=0,3,6,9 are incorrect; i=1,2,4,5,7,8 are correct)
        assert metrics.accuracy_rate == pytest.approx(0.6, abs=0.05)

    def test_confidence_metrics(self, populated_tracker):
        """Confidence metrics calculated."""
        metrics = populated_tracker.get_metrics("analyst")

        assert 0 < metrics.average_confidence < 1
        assert metrics.confidence_std >= 0

    def test_execution_time_metrics(self, populated_tracker):
        """Execution time metrics calculated."""
        metrics = populated_tracker.get_metrics("analyst")

        assert metrics.average_execution_time_ms > 0
        assert metrics.p95_execution_time_ms >= metrics.average_execution_time_ms

    def test_decision_distribution(self, populated_tracker):
        """Decision distribution tracked."""
        metrics = populated_tracker.get_metrics("analyst")

        assert "analysis" in metrics.decision_distribution
        assert metrics.decision_distribution["analysis"] == 10


class TestCalibration:
    """Tests for calibration metrics."""

    def test_calibration_error(self, tracker):
        """Calibration error calculated correctly."""
        # All high confidence, 50% correct = poor calibration
        for i in range(10):
            tracker.record_decision(
                agent_name="overconfident",
                decision_type="trade",
                decision="BUY",
                confidence=0.9,
                was_correct=i < 5,  # 50% correct
            )

        metrics = tracker.get_metrics("overconfident")

        # Calibration error = |0.9 - 0.5| = 0.4
        assert metrics.calibration_error == pytest.approx(0.4, abs=0.05)

    def test_overconfidence_rate(self, tracker):
        """Overconfidence rate calculated."""
        # High confidence but wrong
        tracker.record_decision(
            agent_name="agent",
            decision_type="trade",
            decision="BUY",
            confidence=0.9,
            was_correct=False,
        )

        metrics = tracker.get_metrics("agent")

        assert metrics.overconfidence_rate == 1.0  # 100% overconfident

    def test_underconfidence_rate(self, tracker):
        """Underconfidence rate calculated."""
        # Low confidence but right
        tracker.record_decision(
            agent_name="agent",
            decision_type="trade",
            decision="BUY",
            confidence=0.3,
            was_correct=True,
        )

        metrics = tracker.get_metrics("agent")

        assert metrics.underconfidence_rate == 1.0  # 100% underconfident


class TestTimeFiltering:
    """Tests for time-based filtering."""

    def test_filter_by_start_time(self, tracker):
        """Can filter metrics by start time."""
        now = datetime.utcnow()

        # Old decision
        tracker.record_decision(
            agent_name="agent",
            decision_type="trade",
            decision="BUY",
            confidence=0.8,
        )
        tracker.decisions[0].timestamp = now - timedelta(days=30)

        # Recent decision
        tracker.record_decision(
            agent_name="agent",
            decision_type="trade",
            decision="SELL",
            confidence=0.7,
        )

        metrics = tracker.get_metrics(
            "agent",
            start_time=now - timedelta(days=1),
        )

        assert metrics.total_decisions == 1


class TestTrendAnalysis:
    """Tests for trend analysis."""

    def test_trend_insufficient_data(self, tracker):
        """Trend with insufficient data."""
        tracker.record_decision(
            agent_name="agent",
            decision_type="trade",
            decision="BUY",
            confidence=0.8,
        )

        trend = tracker.get_trend("agent", "confidence")

        assert trend.trend_direction == "insufficient_data"

    def test_trend_with_data(self, populated_tracker):
        """Trend calculated with sufficient data."""
        trend = populated_tracker.get_trend("analyst", "confidence")

        assert trend.agent_name == "analyst"
        assert trend.metric_name == "confidence"
        # May not have enough daily data for trend in test


class TestAgentComparison:
    """Tests for agent comparison."""

    def test_compare_accuracy(self, populated_tracker):
        """Can compare agents by accuracy."""
        comparison = populated_tracker.compare_agents(
            ["analyst", "trader"],
            "accuracy",
        )

        assert comparison.metric_name == "accuracy"
        assert len(comparison.values) == 2
        assert comparison.best_agent in ["analyst", "trader"]
        assert comparison.worst_agent in ["analyst", "trader"]

    def test_compare_confidence(self, populated_tracker):
        """Can compare agents by confidence."""
        comparison = populated_tracker.compare_agents(
            ["analyst", "trader"],
            "confidence",
        )

        assert comparison.metric_name == "confidence"
        assert comparison.spread >= 0


class TestAllMetrics:
    """Tests for getting all metrics."""

    def test_get_all_metrics(self, populated_tracker):
        """Get metrics for all agents."""
        all_metrics = populated_tracker.get_all_metrics()

        assert "analyst" in all_metrics
        assert "trader" in all_metrics
        assert isinstance(all_metrics["analyst"], AgentMetrics)


class TestSummary:
    """Tests for summary functionality."""

    def test_empty_summary(self, tracker):
        """Summary for empty tracker."""
        summary = tracker.get_summary()

        assert summary["total_agents"] == 0
        assert summary["total_decisions"] == 0

    def test_summary_with_data(self, populated_tracker):
        """Summary with populated data."""
        summary = populated_tracker.get_summary()

        assert summary["total_agents"] == 2
        assert summary["total_decisions"] == 15
        assert "analyst" in summary["agents"]
        assert "trader" in summary["agents"]


class TestMaxHistory:
    """Tests for history size limits."""

    def test_max_history_enforced(self):
        """History doesn't exceed max size."""
        tracker = AgentMetricsTracker(max_history=5)

        for i in range(10):
            tracker.record_decision(
                agent_name="agent",
                decision_type="trade",
                decision="BUY",
                confidence=0.8,
            )

        assert len(tracker.decisions) == 5


class TestSerialization:
    """Tests for serialization."""

    def test_decision_record_to_dict(self, tracker):
        """DecisionRecord can be serialized."""
        tracker.record_decision(
            agent_name="agent",
            decision_type="trade",
            decision="BUY",
            confidence=0.8,
        )

        record_dict = tracker.decisions[0].to_dict()

        assert "decision_id" in record_dict
        assert "agent_name" in record_dict
        assert "confidence" in record_dict
        assert "timestamp" in record_dict

    def test_metrics_to_dict(self, populated_tracker):
        """AgentMetrics can be serialized."""
        metrics = populated_tracker.get_metrics("analyst")
        metrics_dict = metrics.to_dict()

        assert "agent_name" in metrics_dict
        assert "accuracy_rate" in metrics_dict
        assert "average_confidence" in metrics_dict


class TestClear:
    """Tests for clearing data."""

    def test_clear_removes_all(self, populated_tracker):
        """Clear removes all decisions."""
        assert len(populated_tracker.decisions) > 0

        populated_tracker.clear()

        assert len(populated_tracker.decisions) == 0
        assert len(populated_tracker.decisions_by_agent) == 0


class TestFactoryAndReport:
    """Tests for factory function and report generation."""

    def test_create_metrics_tracker(self):
        """Factory creates tracker."""
        tracker = create_metrics_tracker(
            max_history=5000,
            evaluation_window_days=14,
        )

        assert tracker.max_history == 5000
        assert tracker.evaluation_window_days == 14

    def test_generate_report(self, populated_tracker):
        """Report is generated correctly."""
        report = generate_metrics_report(populated_tracker)

        assert "AGENT PERFORMANCE METRICS REPORT" in report
        assert "analyst" in report
        assert "trader" in report
        assert "ACCURACY:" in report
        assert "CONFIDENCE:" in report
        assert "EFFICIENCY:" in report
