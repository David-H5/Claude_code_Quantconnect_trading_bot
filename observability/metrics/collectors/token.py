"""
Token Usage Metrics Tracker

Tracks token usage and cost estimates for LLM operations.
Provides aggregation by agent, model, and time windows.

UPGRADE-014 Category 2: Observability & Debugging
Refactored: Phase 3 - Consolidated Metrics Infrastructure

Location: observability/metrics/collectors/token.py
Old location: observability/token_metrics.py (re-exports for compatibility)

QuantConnect Compatible: Yes
- No blocking operations
- Thread-safe implementation
- Memory-bounded history
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any


logger = logging.getLogger(__name__)


# ============================================================================
# Cost Configuration
# ============================================================================

# Token costs per model (per 1K tokens)
# Updated December 2025
MODEL_COSTS = {
    # Anthropic
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3.5-haiku": {"input": 0.0008, "output": 0.004},
    # OpenAI
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Default for unknown models
    "default": {"input": 0.001, "output": 0.002},
}


def get_model_cost(model: str) -> dict[str, float]:
    """Get cost per 1K tokens for a model."""
    # Try exact match
    if model in MODEL_COSTS:
        return MODEL_COSTS[model]

    # Try partial match
    model_lower = model.lower()
    for key in MODEL_COSTS:
        if key in model_lower:
            return MODEL_COSTS[key]

    return MODEL_COSTS["default"]


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class TokenUsageRecord:
    """Single token usage record."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    operation: str = ""
    trace_id: str | None = None

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens

        # Calculate costs
        costs = get_model_cost(self.model)
        self.input_cost = (self.input_tokens / 1000) * costs["input"]
        self.output_cost = (self.output_tokens / 1000) * costs["output"]
        self.total_cost = self.input_cost + self.output_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": round(self.input_cost, 6),
            "output_cost": round(self.output_cost, 6),
            "total_cost": round(self.total_cost, 6),
            "operation": self.operation,
            "trace_id": self.trace_id,
        }


@dataclass
class TokenUsageSummary:
    """Aggregated token usage summary."""

    period_start: datetime
    period_end: datetime
    total_records: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_tokens_per_call: float = 0.0
    avg_cost_per_call: float = 0.0
    by_agent: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_records": self.total_records,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "avg_tokens_per_call": round(self.avg_tokens_per_call, 2),
            "avg_cost_per_call": round(self.avg_cost_per_call, 6),
            "by_agent": self.by_agent,
            "by_model": self.by_model,
        }


# ============================================================================
# Token Usage Tracker
# ============================================================================


class TokenUsageTracker:
    """
    Tracks token usage across agents and models.

    Thread-safe implementation with configurable retention.

    Usage:
        tracker = TokenUsageTracker()

        # Record usage
        tracker.record(
            agent_name="analyst",
            model="claude-3-sonnet",
            input_tokens=150,
            output_tokens=300,
        )

        # Get summary
        summary = tracker.get_summary(window_minutes=60)
        print(f"Total cost: ${summary.total_cost:.4f}")

        # Get by agent
        agent_usage = tracker.get_usage_by_agent("analyst")
    """

    def __init__(
        self,
        max_records: int = 100000,
        retention_hours: int = 24,
    ):
        """
        Initialize tracker.

        Args:
            max_records: Maximum records to retain
            retention_hours: Hours to retain records
        """
        self.max_records = max_records
        self.retention_hours = retention_hours

        self._records: list[TokenUsageRecord] = []
        self._lock = Lock()

        # Running totals for efficiency
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._total_records: int = 0

    def record(
        self,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "",
        trace_id: str | None = None,
    ) -> TokenUsageRecord:
        """
        Record token usage.

        Args:
            agent_name: Name of the agent
            model: LLM model name
            input_tokens: Input/prompt tokens
            output_tokens: Output/completion tokens
            operation: Operation name
            trace_id: Trace ID for correlation

        Returns:
            Created TokenUsageRecord
        """
        record = TokenUsageRecord(
            agent_name=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            operation=operation,
            trace_id=trace_id,
        )

        with self._lock:
            self._records.append(record)
            self._total_tokens += record.total_tokens
            self._total_cost += record.total_cost
            self._total_records += 1

            # Cleanup old records
            self._cleanup_old_records()

        return record

    def _cleanup_old_records(self):
        """Remove old records beyond retention period."""
        if len(self._records) <= self.max_records:
            return

        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
        old_len = len(self._records)

        self._records = [r for r in self._records if r.timestamp >= cutoff]

        # If still over limit, trim oldest
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]

        if len(self._records) < old_len:
            logger.debug(f"Cleaned up {old_len - len(self._records)} old token records")

    def get_summary(
        self,
        window_minutes: int | None = None,
        agent_name: str | None = None,
        model: str | None = None,
    ) -> TokenUsageSummary:
        """
        Get aggregated usage summary.

        Args:
            window_minutes: Time window in minutes (None for all)
            agent_name: Filter by agent
            model: Filter by model

        Returns:
            TokenUsageSummary with aggregated metrics
        """
        with self._lock:
            records = list(self._records)

        now = datetime.now(timezone.utc)

        # Apply time filter
        if window_minutes:
            cutoff = now - timedelta(minutes=window_minutes)
            records = [r for r in records if r.timestamp >= cutoff]
            period_start = cutoff
        else:
            period_start = records[0].timestamp if records else now

        # Apply agent filter
        if agent_name:
            records = [r for r in records if r.agent_name == agent_name]

        # Apply model filter
        if model:
            records = [r for r in records if r.model == model]

        if not records:
            return TokenUsageSummary(
                period_start=period_start,
                period_end=now,
            )

        # Aggregate
        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        total_tokens = sum(r.total_tokens for r in records)
        total_cost = sum(r.total_cost for r in records)

        # By agent
        by_agent: dict[str, dict[str, Any]] = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "calls": 0})
        for r in records:
            by_agent[r.agent_name]["tokens"] += r.total_tokens
            by_agent[r.agent_name]["cost"] += r.total_cost
            by_agent[r.agent_name]["calls"] += 1

        # By model
        by_model: dict[str, dict[str, Any]] = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "calls": 0})
        for r in records:
            by_model[r.model]["tokens"] += r.total_tokens
            by_model[r.model]["cost"] += r.total_cost
            by_model[r.model]["calls"] += 1

        return TokenUsageSummary(
            period_start=period_start,
            period_end=now,
            total_records=len(records),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_tokens,
            total_cost=total_cost,
            avg_tokens_per_call=total_tokens / len(records),
            avg_cost_per_call=total_cost / len(records),
            by_agent=dict(by_agent),
            by_model=dict(by_model),
        )

    def get_usage_by_agent(
        self,
        agent_name: str,
        window_minutes: int | None = None,
    ) -> list[TokenUsageRecord]:
        """Get all records for a specific agent."""
        with self._lock:
            records = [r for r in self._records if r.agent_name == agent_name]

        if window_minutes:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
            records = [r for r in records if r.timestamp >= cutoff]

        return records

    def get_usage_by_model(
        self,
        model: str,
        window_minutes: int | None = None,
    ) -> list[TokenUsageRecord]:
        """Get all records for a specific model."""
        with self._lock:
            records = [r for r in self._records if r.model == model]

        if window_minutes:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
            records = [r for r in records if r.timestamp >= cutoff]

        return records

    def get_recent_records(self, limit: int = 100) -> list[TokenUsageRecord]:
        """Get most recent records."""
        with self._lock:
            return list(self._records[-limit:])

    def get_totals(self) -> dict[str, Any]:
        """Get running totals."""
        return {
            "total_tokens": self._total_tokens,
            "total_cost": round(self._total_cost, 4),
            "total_records": self._total_records,
        }

    def get_rate(self, window_minutes: int = 1) -> dict[str, float]:
        """
        Get usage rate per minute.

        Args:
            window_minutes: Window for rate calculation

        Returns:
            Dict with tokens_per_minute and cost_per_minute
        """
        summary = self.get_summary(window_minutes=window_minutes)

        if summary.total_records == 0:
            return {"tokens_per_minute": 0.0, "cost_per_minute": 0.0}

        return {
            "tokens_per_minute": summary.total_tokens / window_minutes,
            "cost_per_minute": summary.total_cost / window_minutes,
        }

    def clear(self):
        """Clear all records."""
        with self._lock:
            self._records.clear()
            self._total_tokens = 0
            self._total_cost = 0.0
            self._total_records = 0

    def to_dict(self) -> dict[str, Any]:
        """Export tracker state."""
        return {
            "totals": self.get_totals(),
            "summary_1h": self.get_summary(window_minutes=60).to_dict(),
            "summary_24h": self.get_summary(window_minutes=1440).to_dict(),
            "rate_1m": self.get_rate(window_minutes=1),
            "rate_5m": self.get_rate(window_minutes=5),
        }


# ============================================================================
# Global Tracker Instance
# ============================================================================

_global_tracker: TokenUsageTracker | None = None
_tracker_lock = Lock()


def get_global_tracker() -> TokenUsageTracker:
    """Get the global token usage tracker singleton."""
    global _global_tracker

    if _global_tracker is None:
        with _tracker_lock:
            if _global_tracker is None:
                _global_tracker = TokenUsageTracker()

    return _global_tracker


def create_tracker(
    max_records: int = 100000,
    retention_hours: int = 24,
) -> TokenUsageTracker:
    """Factory function to create a new tracker."""
    return TokenUsageTracker(
        max_records=max_records,
        retention_hours=retention_hours,
    )


def record_usage(
    agent_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    operation: str = "",
    trace_id: str | None = None,
) -> TokenUsageRecord:
    """Convenience function to record usage to global tracker."""
    tracker = get_global_tracker()
    return tracker.record(
        agent_name=agent_name,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        operation=operation,
        trace_id=trace_id,
    )
