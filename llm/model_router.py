"""
Dual LLM Model Router (UPGRADE-010 Sprint 2)

Task-based model selection for cost/latency optimization.
Routes prompts to appropriate model tiers based on task complexity.

Features:
- Task classification (reasoning vs tool-use vs simple)
- Automatic tier selection
- Cost tracking per model
- Latency metrics
- Override capability

QuantConnect Compatible: Yes

Usage:
    from llm.model_router import (
        LLMRouter,
        create_router,
        TaskType,
        ModelTier,
    )

    # Create router
    router = create_router()

    # Route a prompt
    provider, model = router.route("Analyze SPY technicals in depth")
    # Returns: ("anthropic", "claude-3-opus")

    # Route with task type hint
    provider, model = router.route(
        "Execute trailing stop order",
        task_hint=TaskType.TOOL_USE,
    )
    # Returns: ("anthropic", "claude-3-haiku")

    # Get cost summary
    report = router.get_cost_summary()
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from models.exceptions import ConfigurationError, DataValidationError


class TaskType(Enum):
    """Type of task for routing."""

    REASONING = "reasoning"  # Complex analysis, multi-step thinking
    STANDARD = "standard"  # General tasks, summaries
    TOOL_USE = "tool_use"  # Function calls, simple extraction
    SIMPLE = "simple"  # Basic responses, formatting


class ModelTier(Enum):
    """Model tier for cost/capability tradeoff."""

    REASONING = "reasoning"  # Highest capability, highest cost
    STANDARD = "standard"  # Balanced
    FAST = "fast"  # Lowest cost, fastest


@dataclass
class ModelConfig:
    """Configuration for a model."""

    provider: str
    model_id: str
    tier: ModelTier
    cost_per_1k_input: float  # USD per 1000 input tokens
    cost_per_1k_output: float  # USD per 1000 output tokens
    avg_latency_ms: float  # Average response latency
    max_tokens: int = 4096
    supports_tools: bool = True
    supports_vision: bool = False


@dataclass
class RouteDecision:
    """Result of routing decision."""

    provider: str
    model_id: str
    tier: ModelTier
    task_type: TaskType
    reason: str
    override_used: bool = False


@dataclass
class UsageRecord:
    """Record of model usage."""

    timestamp: datetime
    provider: str
    model_id: str
    tier: ModelTier
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    task_type: TaskType


@dataclass
class CostReport:
    """Cost summary report."""

    total_cost_usd: float
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    cost_by_tier: dict[str, float]
    cost_by_model: dict[str, float]
    requests_by_tier: dict[str, int]
    avg_latency_by_tier: dict[str, float]
    period_start: datetime | None = None
    period_end: datetime | None = None


class TaskClassifier:
    """
    Classifies prompts to determine appropriate task type.

    Uses keyword analysis and heuristics to determine complexity.
    """

    # Keywords indicating reasoning tasks
    REASONING_KEYWORDS = [
        "analyze",
        "explain",
        "why",
        "compare",
        "evaluate",
        "assess",
        "strategy",
        "plan",
        "recommend",
        "consider",
        "trade-off",
        "implications",
        "reasoning",
        "think through",
        "deep dive",
        "comprehensive",
        "in depth",
        "thoroughly",
    ]

    # Keywords indicating tool use
    TOOL_KEYWORDS = [
        "execute",
        "run",
        "call",
        "fetch",
        "get",
        "set",
        "create",
        "delete",
        "update",
        "submit",
        "order",
        "cancel",
        "place",
    ]

    # Keywords indicating simple tasks
    SIMPLE_KEYWORDS = [
        "format",
        "convert",
        "list",
        "extract",
        "summarize briefly",
        "yes or no",
        "true or false",
        "short answer",
    ]

    def __init__(
        self,
        reasoning_threshold: float = 0.3,
        tool_threshold: float = 0.3,
    ):
        """
        Initialize classifier.

        Args:
            reasoning_threshold: Score threshold for reasoning classification
            tool_threshold: Score threshold for tool use classification
        """
        self.reasoning_threshold = reasoning_threshold
        self.tool_threshold = tool_threshold

    def classify(self, prompt: str) -> tuple[TaskType, float]:
        """
        Classify a prompt.

        Args:
            prompt: The prompt to classify

        Returns:
            Tuple of (TaskType, confidence score)
        """
        prompt_lower = prompt.lower()
        prompt_length = len(prompt.split())

        # Calculate scores for each type
        reasoning_score = self._calculate_score(
            prompt_lower,
            self.REASONING_KEYWORDS,
        )
        tool_score = self._calculate_score(
            prompt_lower,
            self.TOOL_KEYWORDS,
        )
        simple_score = self._calculate_score(
            prompt_lower,
            self.SIMPLE_KEYWORDS,
        )

        # Length-based adjustments
        if prompt_length > 100:
            reasoning_score += 0.2
        elif prompt_length < 20:
            simple_score += 0.2

        # Question complexity
        if "?" in prompt:
            if any(w in prompt_lower for w in ["how", "why", "what if"]):
                reasoning_score += 0.15
            elif any(w in prompt_lower for w in ["is", "are", "can"]):
                simple_score += 0.1

        # Determine type
        if reasoning_score >= self.reasoning_threshold and reasoning_score > tool_score:
            return TaskType.REASONING, min(reasoning_score, 1.0)
        elif tool_score >= self.tool_threshold and tool_score > simple_score:
            return TaskType.TOOL_USE, min(tool_score, 1.0)
        elif simple_score > 0.2:
            return TaskType.SIMPLE, min(simple_score, 1.0)
        else:
            return TaskType.STANDARD, 0.5

    def _calculate_score(self, text: str, keywords: list[str]) -> float:
        """Calculate keyword match score."""
        matches = sum(1 for kw in keywords if kw in text)
        return min(matches * 0.15, 1.0)


class CostTracker:
    """
    Tracks LLM usage costs and metrics.
    """

    def __init__(self, max_history: int = 10000):
        """
        Initialize cost tracker.

        Args:
            max_history: Maximum usage records to keep
        """
        self.max_history = max_history
        self._records: list[UsageRecord] = []

    def record_usage(
        self,
        provider: str,
        model_id: str,
        tier: ModelTier,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost_usd: float,
        task_type: TaskType,
    ) -> UsageRecord:
        """
        Record model usage.

        Args:
            provider: Provider name
            model_id: Model identifier
            tier: Model tier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Response latency in milliseconds
            cost_usd: Cost in USD
            task_type: Type of task

        Returns:
            UsageRecord
        """
        record = UsageRecord(
            timestamp=datetime.utcnow(),
            provider=provider,
            model_id=model_id,
            tier=tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            task_type=task_type,
        )

        self._records.append(record)

        # Trim history
        if len(self._records) > self.max_history:
            self._records = self._records[-self.max_history :]

        return record

    def get_cost_report(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> CostReport:
        """
        Generate cost report.

        Args:
            start_time: Filter start time
            end_time: Filter end time

        Returns:
            CostReport
        """
        records = self._records

        # Filter by time
        if start_time:
            records = [r for r in records if r.timestamp >= start_time]
        if end_time:
            records = [r for r in records if r.timestamp <= end_time]

        if not records:
            return CostReport(
                total_cost_usd=0.0,
                total_requests=0,
                total_input_tokens=0,
                total_output_tokens=0,
                cost_by_tier={},
                cost_by_model={},
                requests_by_tier={},
                avg_latency_by_tier={},
                period_start=start_time,
                period_end=end_time,
            )

        # Aggregate metrics
        cost_by_tier: dict[str, float] = {}
        cost_by_model: dict[str, float] = {}
        requests_by_tier: dict[str, int] = {}
        latency_by_tier: dict[str, list[float]] = {}

        for record in records:
            tier_name = record.tier.value

            # Cost by tier
            cost_by_tier[tier_name] = cost_by_tier.get(tier_name, 0) + record.cost_usd

            # Cost by model
            model_key = f"{record.provider}/{record.model_id}"
            cost_by_model[model_key] = cost_by_model.get(model_key, 0) + record.cost_usd

            # Requests by tier
            requests_by_tier[tier_name] = requests_by_tier.get(tier_name, 0) + 1

            # Latency by tier
            if tier_name not in latency_by_tier:
                latency_by_tier[tier_name] = []
            latency_by_tier[tier_name].append(record.latency_ms)

        # Calculate average latencies
        avg_latency_by_tier = {tier: sum(lats) / len(lats) for tier, lats in latency_by_tier.items()}

        return CostReport(
            total_cost_usd=sum(r.cost_usd for r in records),
            total_requests=len(records),
            total_input_tokens=sum(r.input_tokens for r in records),
            total_output_tokens=sum(r.output_tokens for r in records),
            cost_by_tier=cost_by_tier,
            cost_by_model=cost_by_model,
            requests_by_tier=requests_by_tier,
            avg_latency_by_tier=avg_latency_by_tier,
            period_start=start_time or (records[0].timestamp if records else None),
            period_end=end_time or (records[-1].timestamp if records else None),
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get tracker statistics."""
        if not self._records:
            return {
                "total_records": 0,
                "total_cost_usd": 0.0,
                "oldest_record": None,
                "newest_record": None,
            }

        return {
            "total_records": len(self._records),
            "total_cost_usd": sum(r.cost_usd for r in self._records),
            "oldest_record": self._records[0].timestamp.isoformat(),
            "newest_record": self._records[-1].timestamp.isoformat(),
        }


class BudgetPeriod(Enum):
    """Budget period type."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class AlertLevel(Enum):
    """Alert severity level."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class BudgetConfig:
    """Configuration for a budget limit."""

    period: BudgetPeriod
    limit_usd: float
    warning_threshold: float = 0.80  # 80% of limit
    critical_threshold: float = 0.95  # 95% of limit
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate thresholds."""
        if not 0 < self.warning_threshold < self.critical_threshold <= 1.0:
            raise DataValidationError(
                field="thresholds",
                value={"warning": self.warning_threshold, "critical": self.critical_threshold},
                reason="must be: 0 < warning < critical <= 1.0",
            )


@dataclass
class BudgetAlert:
    """A budget alert event."""

    timestamp: datetime
    period: BudgetPeriod
    level: AlertLevel
    current_spend: float
    limit_usd: float
    percentage: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "period": self.period.value,
            "level": self.level.value,
            "current_spend": self.current_spend,
            "limit_usd": self.limit_usd,
            "percentage": self.percentage,
            "message": self.message,
        }


class BudgetManager:
    """
    Manages budget limits and alerts for LLM usage.

    Supports daily, weekly, and monthly budget limits with
    configurable warning and critical thresholds.
    """

    def __init__(
        self,
        cost_tracker: CostTracker,
        alert_callback: Callable[[BudgetAlert], None] | None = None,
    ):
        """
        Initialize budget manager.

        Args:
            cost_tracker: CostTracker instance to monitor
            alert_callback: Optional callback for alert notifications
        """
        self.cost_tracker = cost_tracker
        self.alert_callback = alert_callback
        self._budgets: dict[BudgetPeriod, BudgetConfig] = {}
        self._alerts: list[BudgetAlert] = []
        self._last_check: dict[BudgetPeriod, datetime] = {}

    def set_budget(self, config: BudgetConfig) -> None:
        """
        Set a budget limit.

        Args:
            config: Budget configuration
        """
        self._budgets[config.period] = config

    def remove_budget(self, period: BudgetPeriod) -> None:
        """Remove a budget limit."""
        self._budgets.pop(period, None)

    def get_budget(self, period: BudgetPeriod) -> BudgetConfig | None:
        """Get budget configuration for a period."""
        return self._budgets.get(period)

    def check_budgets(self) -> list[BudgetAlert]:
        """
        Check all budgets and return any new alerts.

        Returns:
            List of new alerts generated
        """
        new_alerts: list[BudgetAlert] = []
        now = datetime.utcnow()

        for period, config in self._budgets.items():
            if not config.enabled:
                continue

            # Get period boundaries
            start_time = self._get_period_start(now, period)

            # Get cost report for period
            report = self.cost_tracker.get_cost_report(start_time=start_time)
            current_spend = report.total_cost_usd
            percentage = current_spend / config.limit_usd if config.limit_usd > 0 else 0

            # Check thresholds
            alert = self._check_thresholds(period, config, current_spend, percentage, now)

            if alert:
                new_alerts.append(alert)
                self._alerts.append(alert)

                if self.alert_callback:
                    self.alert_callback(alert)

        return new_alerts

    def _get_period_start(self, now: datetime, period: BudgetPeriod) -> datetime:
        """Get start of budget period."""
        if period == BudgetPeriod.DAILY:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == BudgetPeriod.WEEKLY:
            # Start of week (Monday)
            days_since_monday = now.weekday()
            start = now - timedelta(days=days_since_monday)
            return start.replace(hour=0, minute=0, second=0, microsecond=0)
        else:  # MONTHLY
            return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def _check_thresholds(
        self,
        period: BudgetPeriod,
        config: BudgetConfig,
        current_spend: float,
        percentage: float,
        now: datetime,
    ) -> BudgetAlert | None:
        """Check if thresholds are exceeded."""
        # Determine alert level
        level: AlertLevel | None = None
        if percentage >= config.critical_threshold:
            level = AlertLevel.CRITICAL
        elif percentage >= config.warning_threshold:
            level = AlertLevel.WARNING

        if level is None:
            return None

        # Avoid duplicate alerts in same period
        last_check = self._last_check.get(period)
        if last_check and self._same_period(last_check, now, period):
            # Check if we already alerted at this level
            recent_alerts = [
                a for a in self._alerts if a.period == period and self._same_period(a.timestamp, now, period)
            ]
            if any(a.level == level for a in recent_alerts):
                return None

        self._last_check[period] = now

        message = (
            f"{level.value.upper()}: {period.value} spend ${current_spend:.2f} "
            f"is {percentage:.1%} of ${config.limit_usd:.2f} limit"
        )

        return BudgetAlert(
            timestamp=now,
            period=period,
            level=level,
            current_spend=current_spend,
            limit_usd=config.limit_usd,
            percentage=percentage,
            message=message,
        )

    def _same_period(self, time1: datetime, time2: datetime, period: BudgetPeriod) -> bool:
        """Check if two times are in the same budget period."""
        start1 = self._get_period_start(time1, period)
        start2 = self._get_period_start(time2, period)
        return start1 == start2

    def get_alerts(
        self,
        period: BudgetPeriod | None = None,
        level: AlertLevel | None = None,
        limit: int = 100,
    ) -> list[BudgetAlert]:
        """
        Get recent alerts.

        Args:
            period: Filter by period
            level: Filter by level
            limit: Maximum alerts to return

        Returns:
            List of alerts (newest first)
        """
        alerts = self._alerts.copy()

        if period:
            alerts = [a for a in alerts if a.period == period]
        if level:
            alerts = [a for a in alerts if a.level == level]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

    def get_budget_status(self) -> dict[str, Any]:
        """
        Get current status of all budgets.

        Returns:
            Dictionary with budget status per period
        """
        now = datetime.utcnow()
        status: dict[str, Any] = {}

        for period, config in self._budgets.items():
            start_time = self._get_period_start(now, period)
            report = self.cost_tracker.get_cost_report(start_time=start_time)
            current_spend = report.total_cost_usd
            percentage = current_spend / config.limit_usd if config.limit_usd > 0 else 0
            remaining = max(0, config.limit_usd - current_spend)

            status[period.value] = {
                "enabled": config.enabled,
                "limit_usd": config.limit_usd,
                "current_spend": current_spend,
                "remaining": remaining,
                "percentage": percentage,
                "warning_threshold": config.warning_threshold,
                "critical_threshold": config.critical_threshold,
                "status": (
                    "critical"
                    if percentage >= config.critical_threshold
                    else "warning"
                    if percentage >= config.warning_threshold
                    else "ok"
                ),
            }

        return status


class LLMRouter:
    """
    Routes prompts to appropriate LLM models.

    Selects model tier based on task complexity, cost constraints,
    and latency requirements.
    """

    # Default model configurations
    DEFAULT_MODELS: dict[str, ModelConfig] = {
        "anthropic/claude-3-opus": ModelConfig(
            provider="anthropic",
            model_id="claude-3-opus",
            tier=ModelTier.REASONING,
            cost_per_1k_input=15.0,
            cost_per_1k_output=75.0,
            avg_latency_ms=5000,
            max_tokens=4096,
            supports_vision=True,
        ),
        "anthropic/claude-3-sonnet": ModelConfig(
            provider="anthropic",
            model_id="claude-3-sonnet",
            tier=ModelTier.STANDARD,
            cost_per_1k_input=3.0,
            cost_per_1k_output=15.0,
            avg_latency_ms=2000,
            max_tokens=4096,
            supports_vision=True,
        ),
        "anthropic/claude-3-haiku": ModelConfig(
            provider="anthropic",
            model_id="claude-3-haiku",
            tier=ModelTier.FAST,
            cost_per_1k_input=0.25,
            cost_per_1k_output=1.25,
            avg_latency_ms=500,
            max_tokens=4096,
            supports_vision=True,
        ),
        "openai/gpt-4o": ModelConfig(
            provider="openai",
            model_id="gpt-4o",
            tier=ModelTier.REASONING,
            cost_per_1k_input=5.0,
            cost_per_1k_output=15.0,
            avg_latency_ms=3000,
            max_tokens=4096,
            supports_vision=True,
        ),
        "openai/gpt-4o-mini": ModelConfig(
            provider="openai",
            model_id="gpt-4o-mini",
            tier=ModelTier.STANDARD,
            cost_per_1k_input=0.15,
            cost_per_1k_output=0.60,
            avg_latency_ms=1000,
            max_tokens=4096,
            supports_vision=True,
        ),
        "openai/gpt-3.5-turbo": ModelConfig(
            provider="openai",
            model_id="gpt-3.5-turbo",
            tier=ModelTier.FAST,
            cost_per_1k_input=0.0015,
            cost_per_1k_output=0.002,
            avg_latency_ms=300,
            max_tokens=4096,
        ),
    }

    def __init__(
        self,
        classifier: TaskClassifier | None = None,
        cost_tracker: CostTracker | None = None,
        models: dict[str, ModelConfig] | None = None,
        default_tier: ModelTier = ModelTier.STANDARD,
        preferred_provider: str | None = None,
    ):
        """
        Initialize router.

        Args:
            classifier: Task classifier
            cost_tracker: Cost tracker
            models: Model configurations
            default_tier: Default tier when uncertain
            preferred_provider: Preferred provider name
        """
        self.classifier = classifier or TaskClassifier()
        self.cost_tracker = cost_tracker or CostTracker()
        self.models = models or self.DEFAULT_MODELS.copy()
        self.default_tier = default_tier
        self.preferred_provider = preferred_provider

        # Override rules (task_type or keyword -> forced tier)
        self._overrides: dict[str, ModelTier] = {}

        # Organize models by tier
        self._models_by_tier: dict[ModelTier, list[ModelConfig]] = {}
        for config in self.models.values():
            if config.tier not in self._models_by_tier:
                self._models_by_tier[config.tier] = []
            self._models_by_tier[config.tier].append(config)

    def add_override(
        self,
        pattern: str,
        tier: ModelTier,
    ) -> None:
        """
        Add routing override for pattern.

        Args:
            pattern: Regex pattern to match
            tier: Tier to use when pattern matches
        """
        self._overrides[pattern] = tier

    def remove_override(self, pattern: str) -> bool:
        """Remove an override."""
        if pattern in self._overrides:
            del self._overrides[pattern]
            return True
        return False

    def classify_task(self, prompt: str) -> tuple[TaskType, float]:
        """
        Classify a prompt's task type.

        Args:
            prompt: The prompt to classify

        Returns:
            Tuple of (TaskType, confidence)
        """
        return self.classifier.classify(prompt)

    def route(
        self,
        prompt: str,
        task_hint: TaskType | None = None,
        require_tools: bool = False,
        require_vision: bool = False,
        max_tokens: int | None = None,
    ) -> RouteDecision:
        """
        Route a prompt to appropriate model.

        Args:
            prompt: The prompt to route
            task_hint: Optional task type hint
            require_tools: Require tool support
            require_vision: Require vision support
            max_tokens: Required max tokens

        Returns:
            RouteDecision with provider and model
        """
        override_used = False
        reason = ""

        # Check overrides first
        for pattern, tier in self._overrides.items():
            if re.search(pattern, prompt, re.IGNORECASE):
                task_type = TaskType.STANDARD  # Default for overrides
                selected_tier = tier
                override_used = True
                reason = f"Override matched: {pattern}"
                break
        else:
            # Classify task
            if task_hint:
                task_type = task_hint
                confidence = 1.0
                reason = f"Task hint provided: {task_hint.value}"
            else:
                task_type, confidence = self.classify_task(prompt)
                reason = f"Classified as {task_type.value} (confidence: {confidence:.2f})"

            # Map task type to tier
            selected_tier = self._task_to_tier(task_type)

        # Get models for tier
        candidates = self._models_by_tier.get(selected_tier, [])

        # Filter by requirements
        if require_tools:
            candidates = [m for m in candidates if m.supports_tools]
        if require_vision:
            candidates = [m for m in candidates if m.supports_vision]
        if max_tokens:
            candidates = [m for m in candidates if m.max_tokens >= max_tokens]

        # Fallback to lower tiers if no candidates
        if not candidates:
            for fallback_tier in [ModelTier.STANDARD, ModelTier.REASONING]:
                candidates = self._models_by_tier.get(fallback_tier, [])
                if require_tools:
                    candidates = [m for m in candidates if m.supports_tools]
                if require_vision:
                    candidates = [m for m in candidates if m.supports_vision]
                if candidates:
                    selected_tier = fallback_tier
                    reason += f" (fallback to {fallback_tier.value})"
                    break

        # Select best model
        if not candidates:
            # Last resort: any model
            candidates = list(self.models.values())

        model = self._select_model(candidates)

        return RouteDecision(
            provider=model.provider,
            model_id=model.model_id,
            tier=selected_tier,
            task_type=task_type if not override_used else TaskType.STANDARD,
            reason=reason,
            override_used=override_used,
        )

    def _task_to_tier(self, task_type: TaskType) -> ModelTier:
        """Map task type to model tier."""
        mapping = {
            TaskType.REASONING: ModelTier.REASONING,
            TaskType.STANDARD: ModelTier.STANDARD,
            TaskType.TOOL_USE: ModelTier.FAST,
            TaskType.SIMPLE: ModelTier.FAST,
        }
        return mapping.get(task_type, self.default_tier)

    def _select_model(self, candidates: list[ModelConfig]) -> ModelConfig:
        """Select best model from candidates."""
        if not candidates:
            raise ConfigurationError(
                key="model_candidates",
                reason="No model candidates available for selection",
            )

        # Prefer specified provider
        if self.preferred_provider:
            provider_models = [m for m in candidates if m.provider == self.preferred_provider]
            if provider_models:
                candidates = provider_models

        # Sort by cost (cheapest first)
        candidates.sort(key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)

        return candidates[0]

    def record_usage(
        self,
        decision: RouteDecision,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> UsageRecord:
        """
        Record usage for a routed request.

        Args:
            decision: The routing decision
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Response latency

        Returns:
            UsageRecord
        """
        model_key = f"{decision.provider}/{decision.model_id}"
        config = self.models.get(model_key)

        if config:
            cost = (input_tokens / 1000) * config.cost_per_1k_input + (output_tokens / 1000) * config.cost_per_1k_output
        else:
            cost = 0.0

        return self.cost_tracker.record_usage(
            provider=decision.provider,
            model_id=decision.model_id,
            tier=decision.tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            task_type=decision.task_type,
        )

    def get_cost_summary(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> CostReport:
        """
        Get cost summary report.

        Args:
            start_time: Filter start time
            end_time: Filter end time

        Returns:
            CostReport
        """
        return self.cost_tracker.get_cost_report(start_time, end_time)

    def get_model_info(self, provider: str, model_id: str) -> ModelConfig | None:
        """Get model configuration."""
        key = f"{provider}/{model_id}"
        return self.models.get(key)

    def list_models(self, tier: ModelTier | None = None) -> list[ModelConfig]:
        """List available models."""
        if tier:
            return self._models_by_tier.get(tier, [])
        return list(self.models.values())

    def get_statistics(self) -> dict[str, Any]:
        """Get router statistics."""
        return {
            "total_models": len(self.models),
            "models_by_tier": {tier.value: len(models) for tier, models in self._models_by_tier.items()},
            "overrides_active": len(self._overrides),
            "preferred_provider": self.preferred_provider,
            "cost_tracker": self.cost_tracker.get_statistics(),
        }


def create_router(
    preferred_provider: str | None = None,
    default_tier: ModelTier = ModelTier.STANDARD,
    reasoning_threshold: float = 0.3,
    custom_models: dict[str, ModelConfig] | None = None,
) -> LLMRouter:
    """
    Factory function to create an LLMRouter.

    Args:
        preferred_provider: Preferred provider name
        default_tier: Default tier when uncertain
        reasoning_threshold: Threshold for reasoning classification
        custom_models: Additional model configurations

    Returns:
        Configured LLMRouter
    """
    classifier = TaskClassifier(reasoning_threshold=reasoning_threshold)
    cost_tracker = CostTracker()

    models = LLMRouter.DEFAULT_MODELS.copy()
    if custom_models:
        models.update(custom_models)

    return LLMRouter(
        classifier=classifier,
        cost_tracker=cost_tracker,
        models=models,
        default_tier=default_tier,
        preferred_provider=preferred_provider,
    )
