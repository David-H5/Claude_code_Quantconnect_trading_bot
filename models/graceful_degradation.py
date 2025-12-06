"""
Graceful Degradation System

Provides model fallback chains and service degradation for fault tolerance.
When primary models/services fail, automatically falls back to simpler alternatives.

UPGRADE-014 Category 3: Fault Tolerance

Features:
- Model fallback chains (Claude -> GPT -> local)
- Health-based automatic routing
- Service level degradation
- Recovery detection

QuantConnect Compatible: Yes
- Non-blocking fallback decisions
- Configurable timeout handling
- Memory-efficient state tracking
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Enums
# ============================================================================


class ServiceLevel(Enum):
    """Service degradation levels."""

    FULL = "full"  # All features available
    DEGRADED = "degraded"  # Reduced features, slower processing
    MINIMAL = "minimal"  # Basic functionality only
    OFFLINE = "offline"  # Service unavailable


class HealthStatus(Enum):
    """Health status for services/models."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class FallbackReason(Enum):
    """Reason for fallback activation."""

    TIMEOUT = "timeout"
    ERROR = "error"
    RATE_LIMIT = "rate_limit"
    UNHEALTHY = "unhealthy"
    MANUAL = "manual"
    COST = "cost"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ModelInfo:
    """Information about an LLM model."""

    name: str
    provider: str
    tier: int  # 1=premium, 2=standard, 3=budget
    cost_per_1k_tokens: float
    max_tokens: int
    latency_ms_avg: float = 500.0
    reliability: float = 0.99


@dataclass
class ServiceHealth:
    """Health status for a service/model."""

    name: str
    status: HealthStatus
    last_check: datetime
    consecutive_failures: int = 0
    last_error: str | None = None
    latency_ms: float | None = None
    success_rate: float = 1.0

    def is_available(self) -> bool:
        """Check if service is available for use."""
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


@dataclass
class FallbackEvent:
    """Record of a fallback event."""

    timestamp: datetime
    from_model: str
    to_model: str
    reason: FallbackReason
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class DegradationConfig:
    """Configuration for degradation behavior."""

    # Health check thresholds
    unhealthy_threshold: int = 3  # Consecutive failures before unhealthy
    degraded_threshold: int = 1  # Failures before degraded
    recovery_threshold: int = 2  # Successes before recovery

    # Timing
    health_check_interval_seconds: int = 30
    timeout_seconds: float = 30.0

    # Cost limits
    max_cost_per_request: float = 0.10  # Max cost before forcing cheaper model

    # Fallback behavior
    auto_fallback: bool = True
    fallback_on_timeout: bool = True
    fallback_on_rate_limit: bool = True


# ============================================================================
# Model Fallback Chain
# ============================================================================


# Default model chain (premium -> standard -> budget)
DEFAULT_MODEL_CHAIN: list[ModelInfo] = [
    ModelInfo(
        name="claude-3-opus",
        provider="anthropic",
        tier=1,
        cost_per_1k_tokens=0.075,
        max_tokens=200000,
        latency_ms_avg=2000.0,
        reliability=0.995,
    ),
    ModelInfo(
        name="claude-3-5-sonnet",
        provider="anthropic",
        tier=1,
        cost_per_1k_tokens=0.015,
        max_tokens=200000,
        latency_ms_avg=1000.0,
        reliability=0.99,
    ),
    ModelInfo(
        name="gpt-4o",
        provider="openai",
        tier=2,
        cost_per_1k_tokens=0.015,
        max_tokens=128000,
        latency_ms_avg=800.0,
        reliability=0.98,
    ),
    ModelInfo(
        name="gpt-4o-mini",
        provider="openai",
        tier=3,
        cost_per_1k_tokens=0.00015,
        max_tokens=128000,
        latency_ms_avg=500.0,
        reliability=0.99,
    ),
    ModelInfo(
        name="claude-3-haiku",
        provider="anthropic",
        tier=3,
        cost_per_1k_tokens=0.00125,
        max_tokens=200000,
        latency_ms_avg=400.0,
        reliability=0.99,
    ),
]


class ModelFallbackChain:
    """
    Manages model fallback chain with automatic degradation.

    Tracks health of each model and automatically falls back to
    next available model when primary fails.
    """

    def __init__(
        self,
        models: list[ModelInfo] | None = None,
        config: DegradationConfig | None = None,
    ):
        """Initialize fallback chain."""
        self.models = models or DEFAULT_MODEL_CHAIN.copy()
        self.config = config or DegradationConfig()

        # Health tracking
        self._health: dict[str, ServiceHealth] = {}
        for model in self.models:
            self._health[model.name] = ServiceHealth(
                name=model.name,
                status=HealthStatus.HEALTHY,
                last_check=datetime.now(timezone.utc),
            )

        # State
        self._current_index = 0
        self._fallback_history: list[FallbackEvent] = []
        self._lock = Lock()

    @property
    def current_model(self) -> ModelInfo:
        """Get currently active model."""
        return self.models[self._current_index]

    def get_available_model(self) -> ModelInfo | None:
        """Get first available model in chain."""
        with self._lock:
            for i, model in enumerate(self.models):
                health = self._health.get(model.name)
                if health and health.is_available():
                    if i != self._current_index:
                        self._record_fallback(
                            from_model=self.models[self._current_index].name,
                            to_model=model.name,
                            reason=FallbackReason.UNHEALTHY,
                        )
                        self._current_index = i
                    return model
            return None

    def record_success(self, model_name: str, latency_ms: float) -> None:
        """Record successful model call."""
        with self._lock:
            if model_name not in self._health:
                return

            health = self._health[model_name]
            health.consecutive_failures = 0
            health.latency_ms = latency_ms
            health.last_check = datetime.now(timezone.utc)
            health.last_error = None

            # Update success rate (exponential moving average)
            health.success_rate = 0.9 * health.success_rate + 0.1

            # Recovery from degraded/unhealthy
            if health.status != HealthStatus.HEALTHY:
                health.status = HealthStatus.HEALTHY
                logger.info(f"Model {model_name} recovered to healthy status")

                # Try to move back up the chain
                self._try_upgrade()

    def record_failure(
        self,
        model_name: str,
        reason: FallbackReason,
        error: str | None = None,
    ) -> ModelInfo | None:
        """Record failed model call and get next available model."""
        with self._lock:
            if model_name not in self._health:
                return None

            health = self._health[model_name]
            health.consecutive_failures += 1
            health.last_check = datetime.now(timezone.utc)
            health.last_error = error

            # Update success rate
            health.success_rate = 0.9 * health.success_rate

            # Update status based on failures
            if health.consecutive_failures >= self.config.unhealthy_threshold:
                health.status = HealthStatus.UNHEALTHY
                logger.warning(f"Model {model_name} marked unhealthy " f"after {health.consecutive_failures} failures")
            elif health.consecutive_failures >= self.config.degraded_threshold:
                health.status = HealthStatus.DEGRADED
                logger.info(f"Model {model_name} marked degraded")

            # Auto-fallback if enabled
            if self.config.auto_fallback:
                return self._fallback(reason)

            return None

    def _fallback(self, reason: FallbackReason) -> ModelInfo | None:
        """Fall back to next available model."""
        current = self.models[self._current_index].name

        for i in range(self._current_index + 1, len(self.models)):
            health = self._health.get(self.models[i].name)
            if health and health.is_available():
                self._record_fallback(
                    from_model=current,
                    to_model=self.models[i].name,
                    reason=reason,
                )
                self._current_index = i
                return self.models[i]

        logger.error("No available models in fallback chain")
        return None

    def _try_upgrade(self) -> None:
        """Try to move back up the chain to a better model."""
        for i in range(self._current_index):
            health = self._health.get(self.models[i].name)
            if health and health.status == HealthStatus.HEALTHY:
                logger.info(f"Upgrading from {self.current_model.name} " f"to {self.models[i].name}")
                self._current_index = i
                return

    def _record_fallback(
        self,
        from_model: str,
        to_model: str,
        reason: FallbackReason,
    ) -> None:
        """Record a fallback event."""
        event = FallbackEvent(
            timestamp=datetime.now(timezone.utc),
            from_model=from_model,
            to_model=to_model,
            reason=reason,
        )
        self._fallback_history.append(event)
        logger.warning(f"Fallback: {from_model} -> {to_model} (reason: {reason.value})")

    def get_health_summary(self) -> dict[str, dict[str, Any]]:
        """Get health summary for all models."""
        with self._lock:
            return {
                name: {
                    "status": health.status.value,
                    "failures": health.consecutive_failures,
                    "success_rate": round(health.success_rate, 3),
                    "latency_ms": health.latency_ms,
                    "last_error": health.last_error,
                }
                for name, health in self._health.items()
            }

    def get_fallback_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent fallback history."""
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "from": e.from_model,
                "to": e.to_model,
                "reason": e.reason.value,
            }
            for e in self._fallback_history[-limit:]
        ]

    def reset(self) -> None:
        """Reset chain to initial state."""
        with self._lock:
            self._current_index = 0
            for health in self._health.values():
                health.status = HealthStatus.HEALTHY
                health.consecutive_failures = 0
                health.last_error = None


# ============================================================================
# Service Degradation Manager
# ============================================================================


class ServiceDegradationManager:
    """
    Manages service level degradation across the system.

    Provides coordinated degradation when resources are constrained
    or failures cascade.
    """

    def __init__(self, config: DegradationConfig | None = None):
        """Initialize degradation manager."""
        self.config = config or DegradationConfig()
        self._service_level = ServiceLevel.FULL
        self._services: dict[str, ServiceHealth] = {}
        self._degradation_callbacks: list[Callable[[ServiceLevel], None]] = []
        self._lock = Lock()

    @property
    def service_level(self) -> ServiceLevel:
        """Get current service level."""
        return self._service_level

    def register_service(self, name: str) -> None:
        """Register a service for health tracking."""
        with self._lock:
            self._services[name] = ServiceHealth(
                name=name,
                status=HealthStatus.HEALTHY,
                last_check=datetime.now(timezone.utc),
            )

    def update_service_health(
        self,
        name: str,
        status: HealthStatus,
        error: str | None = None,
    ) -> None:
        """Update health of a specific service."""
        with self._lock:
            if name in self._services:
                self._services[name].status = status
                self._services[name].last_check = datetime.now(timezone.utc)
                self._services[name].last_error = error

            # Recalculate system service level
            self._update_service_level()

    def _update_service_level(self) -> None:
        """Update overall service level based on individual service health."""
        if not self._services:
            return

        healthy_count = sum(1 for s in self._services.values() if s.status == HealthStatus.HEALTHY)
        total = len(self._services)

        healthy_ratio = healthy_count / total

        old_level = self._service_level

        if healthy_ratio >= 0.9:
            self._service_level = ServiceLevel.FULL
        elif healthy_ratio >= 0.5:
            self._service_level = ServiceLevel.DEGRADED
        elif healthy_ratio > 0:
            self._service_level = ServiceLevel.MINIMAL
        else:
            self._service_level = ServiceLevel.OFFLINE

        if old_level != self._service_level:
            logger.warning(f"Service level changed: {old_level.value} -> " f"{self._service_level.value}")
            self._notify_degradation(self._service_level)

    def _notify_degradation(self, level: ServiceLevel) -> None:
        """Notify registered callbacks of degradation change."""
        for callback in self._degradation_callbacks:
            try:
                callback(level)
            except Exception as e:
                logger.error(f"Degradation callback error: {e}")

    def register_callback(self, callback: Callable[[ServiceLevel], None]) -> None:
        """Register callback for service level changes."""
        self._degradation_callbacks.append(callback)

    def get_feature_availability(self) -> dict[str, bool]:
        """Get feature availability based on current service level."""
        base_features = {
            "llm_analysis": True,
            "real_time_data": True,
            "advanced_analytics": True,
            "multi_agent_consensus": True,
            "historical_backtest": True,
        }

        if self._service_level == ServiceLevel.DEGRADED:
            base_features["multi_agent_consensus"] = False
            base_features["advanced_analytics"] = False
        elif self._service_level == ServiceLevel.MINIMAL:
            base_features["llm_analysis"] = False
            base_features["multi_agent_consensus"] = False
            base_features["advanced_analytics"] = False
            base_features["historical_backtest"] = False
        elif self._service_level == ServiceLevel.OFFLINE:
            return dict.fromkeys(base_features, False)

        return base_features


# ============================================================================
# Factory Functions
# ============================================================================


def create_fallback_chain(
    models: list[ModelInfo] | None = None,
    config: DegradationConfig | None = None,
) -> ModelFallbackChain:
    """Create a model fallback chain."""
    return ModelFallbackChain(models=models, config=config)


def create_degradation_manager(
    config: DegradationConfig | None = None,
) -> ServiceDegradationManager:
    """Create a service degradation manager."""
    return ServiceDegradationManager(config=config)


# Global instances
_global_fallback_chain: ModelFallbackChain | None = None
_global_degradation_manager: ServiceDegradationManager | None = None
_global_lock = Lock()


def get_global_fallback_chain() -> ModelFallbackChain:
    """Get or create the global fallback chain."""
    global _global_fallback_chain

    if _global_fallback_chain is None:
        with _global_lock:
            if _global_fallback_chain is None:
                _global_fallback_chain = create_fallback_chain()

    return _global_fallback_chain


def get_global_degradation_manager() -> ServiceDegradationManager:
    """Get or create the global degradation manager."""
    global _global_degradation_manager

    if _global_degradation_manager is None:
        with _global_lock:
            if _global_degradation_manager is None:
                _global_degradation_manager = create_degradation_manager()

    return _global_degradation_manager
