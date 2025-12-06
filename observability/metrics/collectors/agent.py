"""
Agent Performance Metrics

Comprehensive tracking and analysis of trading agent performance.
Provides insights into decision quality, accuracy, and efficiency.

Refactored: Phase 3 - Consolidated Metrics Infrastructure

Location: observability/metrics/collectors/agent.py
Old location: evaluation/agent_metrics.py (re-exports for compatibility)

QuantConnect Compatible: Yes
- Lightweight data structures
- Non-blocking operations
- Configurable persistence
"""

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class MetricCategory(Enum):
    """Categories of agent metrics."""

    ACCURACY = "accuracy"
    CONFIDENCE = "confidence"
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    RELIABILITY = "reliability"


@dataclass
class DecisionRecord:
    """Record of an agent decision for metrics tracking."""

    decision_id: str
    agent_name: str
    decision_type: str
    decision: str
    confidence: float
    was_correct: bool | None = None
    actual_outcome: str | None = None
    execution_time_ms: float = 0.0
    reasoning_steps: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "agent_name": self.agent_name,
            "decision_type": self.decision_type,
            "decision": self.decision,
            "confidence": self.confidence,
            "was_correct": self.was_correct,
            "actual_outcome": self.actual_outcome,
            "execution_time_ms": self.execution_time_ms,
            "reasoning_steps": self.reasoning_steps,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class AgentMetrics:
    """Performance metrics for a single agent."""

    agent_name: str
    total_decisions: int
    decisions_evaluated: int

    # Accuracy metrics
    accuracy_rate: float  # % of correct decisions (when evaluated)
    calibration_error: float  # Difference between confidence and actual accuracy

    # Confidence metrics
    average_confidence: float
    confidence_std: float
    overconfidence_rate: float  # High confidence but wrong
    underconfidence_rate: float  # Low confidence but right

    # Efficiency metrics
    average_execution_time_ms: float
    p95_execution_time_ms: float
    average_reasoning_steps: float

    # Quality metrics
    decision_distribution: dict[str, int]  # Distribution of decision types
    time_period_start: datetime
    time_period_end: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "total_decisions": self.total_decisions,
            "decisions_evaluated": self.decisions_evaluated,
            "accuracy_rate": self.accuracy_rate,
            "calibration_error": self.calibration_error,
            "average_confidence": self.average_confidence,
            "confidence_std": self.confidence_std,
            "overconfidence_rate": self.overconfidence_rate,
            "underconfidence_rate": self.underconfidence_rate,
            "average_execution_time_ms": self.average_execution_time_ms,
            "p95_execution_time_ms": self.p95_execution_time_ms,
            "average_reasoning_steps": self.average_reasoning_steps,
            "decision_distribution": self.decision_distribution,
            "time_period_start": self.time_period_start.isoformat(),
            "time_period_end": self.time_period_end.isoformat(),
        }


@dataclass
class PerformanceTrend:
    """Trend analysis for agent performance over time."""

    agent_name: str
    metric_name: str
    values: list[float]
    timestamps: list[datetime]
    trend_direction: str  # "improving", "declining", "stable"
    change_pct: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "metric_name": self.metric_name,
            "values": self.values,
            "timestamps": [t.isoformat() for t in self.timestamps],
            "trend_direction": self.trend_direction,
            "change_pct": self.change_pct,
        }


@dataclass
class AgentComparison:
    """Comparison between multiple agents."""

    agents: list[str]
    metric_name: str
    values: dict[str, float]
    best_agent: str
    worst_agent: str
    spread: float  # Difference between best and worst

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agents": self.agents,
            "metric_name": self.metric_name,
            "values": self.values,
            "best_agent": self.best_agent,
            "worst_agent": self.worst_agent,
            "spread": self.spread,
        }


class AgentMetricsTracker:
    """
    Track and analyze agent performance over time.

    Usage:
        tracker = AgentMetricsTracker()

        # Record decisions
        tracker.record_decision(
            agent_name="technical_analyst",
            decision_type="trade",
            decision="BUY",
            confidence=0.85,
        )

        # Update with outcome later
        tracker.update_outcome("decision_123", was_correct=True)

        # Get metrics
        metrics = tracker.get_metrics("technical_analyst")
        print(f"Accuracy: {metrics.accuracy_rate:.1%}")
    """

    def __init__(
        self,
        max_history: int = 10000,
        evaluation_window_days: int = 30,
    ):
        """
        Initialize metrics tracker.

        Args:
            max_history: Maximum decision records to keep
            evaluation_window_days: Window for trend analysis
        """
        self.max_history = max_history
        self.evaluation_window_days = evaluation_window_days

        # Storage
        self.decisions: list[DecisionRecord] = []
        self.decisions_by_agent: dict[str, list[DecisionRecord]] = defaultdict(list)
        self._decision_count = 0

    def record_decision(
        self,
        agent_name: str,
        decision_type: str,
        decision: str,
        confidence: float,
        was_correct: bool | None = None,
        execution_time_ms: float = 0.0,
        reasoning_steps: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Record an agent decision.

        Args:
            agent_name: Name of the agent
            decision_type: Type of decision (trade, analysis, etc.)
            decision: The actual decision made
            confidence: Confidence level (0-1)
            was_correct: Whether decision was correct (if known)
            execution_time_ms: Time taken to make decision
            reasoning_steps: Number of reasoning steps
            metadata: Additional metadata

        Returns:
            Decision ID for later reference
        """
        self._decision_count += 1
        decision_id = f"dec_{agent_name}_{self._decision_count}"

        record = DecisionRecord(
            decision_id=decision_id,
            agent_name=agent_name,
            decision_type=decision_type,
            decision=decision,
            confidence=confidence,
            was_correct=was_correct,
            execution_time_ms=execution_time_ms,
            reasoning_steps=reasoning_steps,
            metadata=metadata or {},
        )

        self.decisions.append(record)
        self.decisions_by_agent[agent_name].append(record)

        # Enforce max history
        if len(self.decisions) > self.max_history:
            oldest = self.decisions.pop(0)
            agent_decisions = self.decisions_by_agent[oldest.agent_name]
            if agent_decisions and agent_decisions[0].decision_id == oldest.decision_id:
                agent_decisions.pop(0)

        return decision_id

    def update_outcome(
        self,
        decision_id: str,
        was_correct: bool,
        actual_outcome: str | None = None,
    ) -> bool:
        """
        Update a decision with its actual outcome.

        Args:
            decision_id: ID of the decision to update
            was_correct: Whether the decision was correct
            actual_outcome: Description of actual outcome

        Returns:
            True if decision was found and updated
        """
        for record in self.decisions:
            if record.decision_id == decision_id:
                record.was_correct = was_correct
                record.actual_outcome = actual_outcome
                return True
        return False

    def get_metrics(
        self,
        agent_name: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> AgentMetrics:
        """
        Get metrics for a specific agent.

        Args:
            agent_name: Name of the agent
            start_time: Filter start time
            end_time: Filter end time

        Returns:
            AgentMetrics with computed performance data
        """
        records = self.decisions_by_agent.get(agent_name, [])

        # Apply time filters
        if start_time:
            records = [r for r in records if r.timestamp >= start_time]
        if end_time:
            records = [r for r in records if r.timestamp <= end_time]

        if not records:
            return self._empty_metrics(agent_name, start_time, end_time)

        # Calculate metrics
        total = len(records)
        evaluated = [r for r in records if r.was_correct is not None]
        correct = [r for r in evaluated if r.was_correct]

        # Accuracy
        accuracy_rate = len(correct) / len(evaluated) if evaluated else 0.0

        # Confidence metrics
        confidences = [r.confidence for r in records]
        avg_confidence = statistics.mean(confidences)
        confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0.0

        # Calibration error
        calibration_error = abs(avg_confidence - accuracy_rate)

        # Over/under confidence
        overconfident = [r for r in evaluated if r.confidence > 0.7 and not r.was_correct]
        underconfident = [r for r in evaluated if r.confidence < 0.5 and r.was_correct]
        overconfidence_rate = len(overconfident) / len(evaluated) if evaluated else 0.0
        underconfidence_rate = len(underconfident) / len(evaluated) if evaluated else 0.0

        # Efficiency
        exec_times = [r.execution_time_ms for r in records if r.execution_time_ms > 0]
        avg_exec_time = statistics.mean(exec_times) if exec_times else 0.0
        p95_exec_time = (
            sorted(exec_times)[int(len(exec_times) * 0.95)]
            if len(exec_times) >= 20
            else max(exec_times)
            if exec_times
            else 0.0
        )

        reasoning_steps = [r.reasoning_steps for r in records]
        avg_reasoning = statistics.mean(reasoning_steps) if reasoning_steps else 0.0

        # Decision distribution
        distribution: dict[str, int] = defaultdict(int)
        for r in records:
            distribution[r.decision_type] += 1

        return AgentMetrics(
            agent_name=agent_name,
            total_decisions=total,
            decisions_evaluated=len(evaluated),
            accuracy_rate=accuracy_rate,
            calibration_error=calibration_error,
            average_confidence=avg_confidence,
            confidence_std=confidence_std,
            overconfidence_rate=overconfidence_rate,
            underconfidence_rate=underconfidence_rate,
            average_execution_time_ms=avg_exec_time,
            p95_execution_time_ms=p95_exec_time,
            average_reasoning_steps=avg_reasoning,
            decision_distribution=dict(distribution),
            time_period_start=min(r.timestamp for r in records),
            time_period_end=max(r.timestamp for r in records),
        )

    def get_all_metrics(self) -> dict[str, AgentMetrics]:
        """
        Get metrics for all tracked agents.

        Returns:
            Dictionary mapping agent names to their metrics
        """
        return {agent_name: self.get_metrics(agent_name) for agent_name in self.decisions_by_agent.keys()}

    def get_trend(
        self,
        agent_name: str,
        metric_name: str,
        window_days: int | None = None,
    ) -> PerformanceTrend:
        """
        Get performance trend for an agent metric.

        Args:
            agent_name: Name of the agent
            metric_name: Name of the metric to trend
            window_days: Days to analyze

        Returns:
            PerformanceTrend with direction and change
        """
        window = window_days or self.evaluation_window_days
        cutoff = datetime.utcnow() - timedelta(days=window)

        records = [r for r in self.decisions_by_agent.get(agent_name, []) if r.timestamp >= cutoff]

        if len(records) < 10:
            return PerformanceTrend(
                agent_name=agent_name,
                metric_name=metric_name,
                values=[],
                timestamps=[],
                trend_direction="insufficient_data",
                change_pct=0.0,
            )

        # Group by day
        daily_values: dict[str, list[float]] = defaultdict(list)
        for r in records:
            day = r.timestamp.strftime("%Y-%m-%d")
            if metric_name == "confidence":
                daily_values[day].append(r.confidence)
            elif metric_name == "accuracy" and r.was_correct is not None:
                daily_values[day].append(1.0 if r.was_correct else 0.0)
            elif metric_name == "execution_time":
                daily_values[day].append(r.execution_time_ms)

        # Average per day
        dates = sorted(daily_values.keys())
        values = [statistics.mean(daily_values[d]) for d in dates]
        timestamps = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

        # Determine trend
        if len(values) >= 2:
            first_half = statistics.mean(values[: len(values) // 2])
            second_half = statistics.mean(values[len(values) // 2 :])
            change_pct = ((second_half - first_half) / first_half * 100) if first_half != 0 else 0

            if change_pct > 5:
                direction = "improving" if metric_name != "execution_time" else "declining"
            elif change_pct < -5:
                direction = "declining" if metric_name != "execution_time" else "improving"
            else:
                direction = "stable"
        else:
            direction = "stable"
            change_pct = 0.0

        return PerformanceTrend(
            agent_name=agent_name,
            metric_name=metric_name,
            values=values,
            timestamps=timestamps,
            trend_direction=direction,
            change_pct=change_pct,
        )

    def compare_agents(
        self,
        agent_names: list[str],
        metric_name: str,
    ) -> AgentComparison:
        """
        Compare multiple agents on a specific metric.

        Args:
            agent_names: List of agents to compare
            metric_name: Metric to compare

        Returns:
            AgentComparison with rankings
        """
        values: dict[str, float] = {}

        for agent in agent_names:
            metrics = self.get_metrics(agent)
            if metric_name == "accuracy":
                values[agent] = metrics.accuracy_rate
            elif metric_name == "confidence":
                values[agent] = metrics.average_confidence
            elif metric_name == "execution_time":
                values[agent] = metrics.average_execution_time_ms
            elif metric_name == "calibration":
                values[agent] = 1.0 - metrics.calibration_error  # Invert so higher is better
            else:
                values[agent] = 0.0

        if not values:
            return AgentComparison(
                agents=agent_names,
                metric_name=metric_name,
                values={},
                best_agent="",
                worst_agent="",
                spread=0.0,
            )

        best_agent = max(values.keys(), key=lambda k: values[k])
        worst_agent = min(values.keys(), key=lambda k: values[k])
        spread = values[best_agent] - values[worst_agent]

        return AgentComparison(
            agents=agent_names,
            metric_name=metric_name,
            values=values,
            best_agent=best_agent,
            worst_agent=worst_agent,
            spread=spread,
        )

    def get_summary(self) -> dict[str, Any]:
        """Get overall summary of all agents."""
        agents = list(self.decisions_by_agent.keys())

        if not agents:
            return {
                "total_agents": 0,
                "total_decisions": 0,
                "agents": {},
            }

        summary = {
            "total_agents": len(agents),
            "total_decisions": len(self.decisions),
            "agents": {},
        }

        for agent in agents:
            metrics = self.get_metrics(agent)
            summary["agents"][agent] = {
                "decisions": metrics.total_decisions,
                "accuracy": f"{metrics.accuracy_rate:.1%}",
                "avg_confidence": f"{metrics.average_confidence:.1%}",
                "calibration_error": f"{metrics.calibration_error:.2f}",
            }

        return summary

    def _empty_metrics(
        self,
        agent_name: str,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> AgentMetrics:
        """Create empty metrics for agent with no data."""
        now = datetime.utcnow()
        return AgentMetrics(
            agent_name=agent_name,
            total_decisions=0,
            decisions_evaluated=0,
            accuracy_rate=0.0,
            calibration_error=0.0,
            average_confidence=0.0,
            confidence_std=0.0,
            overconfidence_rate=0.0,
            underconfidence_rate=0.0,
            average_execution_time_ms=0.0,
            p95_execution_time_ms=0.0,
            average_reasoning_steps=0.0,
            decision_distribution={},
            time_period_start=start_time or now,
            time_period_end=end_time or now,
        )

    def clear(self) -> None:
        """Clear all tracked decisions."""
        self.decisions = []
        self.decisions_by_agent = defaultdict(list)


def create_metrics_tracker(
    max_history: int = 10000,
    evaluation_window_days: int = 30,
) -> AgentMetricsTracker:
    """
    Factory function to create a metrics tracker.

    Args:
        max_history: Maximum decisions to keep
        evaluation_window_days: Window for trend analysis

    Returns:
        Configured AgentMetricsTracker instance
    """
    return AgentMetricsTracker(
        max_history=max_history,
        evaluation_window_days=evaluation_window_days,
    )


def generate_metrics_report(tracker: AgentMetricsTracker) -> str:
    """
    Generate a human-readable metrics report.

    Args:
        tracker: AgentMetricsTracker instance

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "AGENT PERFORMANCE METRICS REPORT",
        "=" * 60,
        "",
    ]

    summary = tracker.get_summary()
    lines.extend(
        [
            f"Total Agents: {summary['total_agents']}",
            f"Total Decisions: {summary['total_decisions']}",
            "",
        ]
    )

    for agent_name, metrics in tracker.get_all_metrics().items():
        lines.extend(
            [
                "-" * 40,
                f"AGENT: {agent_name}",
                "-" * 40,
                f"  Total Decisions: {metrics.total_decisions}",
                f"  Evaluated: {metrics.decisions_evaluated}",
                "",
                "  ACCURACY:",
                f"    Accuracy Rate: {metrics.accuracy_rate:.1%}",
                f"    Calibration Error: {metrics.calibration_error:.3f}",
                f"    Overconfidence Rate: {metrics.overconfidence_rate:.1%}",
                f"    Underconfidence Rate: {metrics.underconfidence_rate:.1%}",
                "",
                "  CONFIDENCE:",
                f"    Average: {metrics.average_confidence:.1%}",
                f"    Std Dev: {metrics.confidence_std:.3f}",
                "",
                "  EFFICIENCY:",
                f"    Avg Execution Time: {metrics.average_execution_time_ms:.0f}ms",
                f"    P95 Execution Time: {metrics.p95_execution_time_ms:.0f}ms",
                f"    Avg Reasoning Steps: {metrics.average_reasoning_steps:.1f}",
                "",
            ]
        )

    lines.append("=" * 60)

    return "\n".join(lines)
