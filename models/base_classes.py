"""
Base Classes for Trading Bot Components

Provides abstract base classes for consistent component architecture:
- ExecutorBase: Order execution components
- ScannerBase: Market/opportunity scanners
- MonitorBase: Monitoring and tracking components
- AnalyzerBase: Analysis and evaluation components

Phase 4 Refactoring: Structural improvements for maintainability.

Usage:
    from models.base_classes import ExecutorBase, ScannerBase

    class MyScanner(ScannerBase):
        def scan(self, data):
            # Implementation
            pass

        def get_opportunities(self):
            return self._opportunities
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar


logger = logging.getLogger(__name__)

# Type variables for generic base classes
T = TypeVar("T")  # Result type
C = TypeVar("C")  # Config type


# =============================================================================
# Configuration Base
# =============================================================================


@dataclass
class BaseConfig:
    """
    Base configuration class for all components.

    Provides common configuration patterns that all components should inherit.
    """

    enabled: bool = True
    log_level: str = "INFO"
    component_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "enabled": self.enabled,
            "log_level": self.log_level,
            "component_name": self.component_name,
        }


# =============================================================================
# Component Statistics Base
# =============================================================================


@dataclass
class ComponentStats:
    """
    Base statistics class for component monitoring.

    Tracks common metrics across all component types.
    """

    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    last_operation_time: datetime | None = None
    total_processing_time_ms: float = 0.0
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_operations == 0:
            return 0.0
        return (self.successful_operations / self.total_operations) * 100

    @property
    def avg_processing_time_ms(self) -> float:
        """Calculate average processing time."""
        if self.total_operations == 0:
            return 0.0
        return self.total_processing_time_ms / self.total_operations

    def record_success(self, processing_time_ms: float = 0.0) -> None:
        """Record a successful operation."""
        self.total_operations += 1
        self.successful_operations += 1
        self.total_processing_time_ms += processing_time_ms
        self.last_operation_time = datetime.now()

    def record_failure(self, error: Exception, processing_time_ms: float = 0.0) -> None:
        """Record a failed operation."""
        self.total_operations += 1
        self.failed_operations += 1
        self.total_processing_time_ms += processing_time_ms
        self.last_operation_time = datetime.now()
        self.errors.append(
            {
                "timestamp": datetime.now().isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.success_rate,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "last_operation_time": self.last_operation_time.isoformat() if self.last_operation_time else None,
            "recent_errors": self.errors[-5:] if self.errors else [],
        }


# =============================================================================
# Base Component Class
# =============================================================================


class BaseComponent(ABC):
    """
    Abstract base class for all trading bot components.

    Provides common functionality:
    - Logging setup
    - Statistics tracking
    - Enable/disable toggling
    - Lifecycle management
    """

    def __init__(self, name: str, enabled: bool = True):
        """
        Initialize base component.

        Args:
            name: Component name for logging and identification
            enabled: Whether component is enabled
        """
        self.name = name
        self.enabled = enabled
        self._logger = logging.getLogger(f"{__name__}.{name}")
        self._stats = ComponentStats()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the component. Override for custom initialization."""
        self._initialized = True
        self._logger.info(f"{self.name} initialized")

    def shutdown(self) -> None:
        """Shutdown the component. Override for custom cleanup."""
        self._initialized = False
        self._logger.info(f"{self.name} shutdown")

    @property
    def is_ready(self) -> bool:
        """Check if component is initialized and enabled."""
        return self._initialized and self.enabled

    def get_stats(self) -> dict[str, Any]:
        """Get component statistics."""
        return {"name": self.name, "enabled": self.enabled, "initialized": self._initialized, **self._stats.to_dict()}


# =============================================================================
# Executor Base Class
# =============================================================================


@dataclass
class ExecutionResult:
    """Result of an execution operation."""

    success: bool
    order_id: str | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "order_id": self.order_id,
            "message": self.message,
            "details": self.details,
        }


class ExecutorBase(BaseComponent, ABC):
    """
    Base class for order execution components.

    Provides:
    - Order submission interface
    - Order tracking
    - Execution statistics
    - Error handling

    Subclasses should implement:
    - execute(): Main execution logic
    - cancel(): Order cancellation
    """

    def __init__(
        self,
        name: str,
        algorithm: Any = None,
        enabled: bool = True,
    ):
        """
        Initialize executor.

        Args:
            name: Executor name
            algorithm: QuantConnect algorithm instance (if applicable)
            enabled: Whether executor is enabled
        """
        super().__init__(name, enabled)
        self.algorithm = algorithm
        self._pending_orders: dict[str, Any] = {}
        self._completed_orders: list[dict[str, Any]] = []

        # Callbacks
        self.on_execution: Callable[[ExecutionResult], None] | None = None
        self.on_error: Callable[[Exception], None] | None = None

    @abstractmethod
    def execute(self, *args, **kwargs) -> ExecutionResult:
        """
        Execute the primary operation.

        Returns:
            ExecutionResult with operation outcome
        """
        pass

    def cancel(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order to cancel

        Returns:
            True if cancellation successful
        """
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]
            self._logger.info(f"Order {order_id} cancelled")
            return True
        return False

    def get_pending_orders(self) -> dict[str, Any]:
        """Get all pending orders."""
        return self._pending_orders.copy()

    def _record_execution(self, result: ExecutionResult) -> None:
        """Record execution result in statistics."""
        if result.success:
            self._stats.record_success()
        else:
            self._stats.record_failure(Exception(result.message))

        if self.on_execution:
            self.on_execution(result)


# =============================================================================
# Scanner Base Class
# =============================================================================


@dataclass
class ScanResult(Generic[T]):
    """Result of a scan operation."""

    opportunities: list[T]
    scan_time_ms: float = 0.0
    symbols_scanned: int = 0
    filters_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "opportunity_count": len(self.opportunities),
            "scan_time_ms": self.scan_time_ms,
            "symbols_scanned": self.symbols_scanned,
            "filters_applied": self.filters_applied,
        }


class ScannerBase(BaseComponent, ABC):
    """
    Base class for market/opportunity scanners.

    Provides:
    - Scan interface
    - Opportunity tracking
    - Filter management
    - Scan statistics

    Subclasses should implement:
    - scan(): Main scanning logic
    - get_opportunities(): Return found opportunities
    """

    def __init__(
        self,
        name: str,
        enabled: bool = True,
    ):
        """
        Initialize scanner.

        Args:
            name: Scanner name
            enabled: Whether scanner is enabled
        """
        super().__init__(name, enabled)
        self._opportunities: list[Any] = []
        self._filters: list[Callable] = []
        self._last_scan_result: ScanResult | None = None

        # Callbacks
        self.on_opportunity: Callable[[Any], None] | None = None

    @abstractmethod
    def scan(self, *args, **kwargs) -> ScanResult:
        """
        Execute the scan operation.

        Returns:
            ScanResult with found opportunities
        """
        pass

    def add_filter(self, filter_func: Callable[[Any], bool], name: str = "") -> None:
        """Add a filter function to the scanner."""
        self._filters.append(filter_func)
        self._logger.debug(f"Filter added: {name or 'unnamed'}")

    def clear_filters(self) -> None:
        """Remove all filters."""
        self._filters.clear()

    def get_opportunities(self) -> list[Any]:
        """Get current opportunities."""
        return self._opportunities.copy()

    def _apply_filters(self, items: list[Any]) -> list[Any]:
        """Apply all filters to a list of items."""
        filtered = items
        for filter_func in self._filters:
            filtered = [item for item in filtered if filter_func(item)]
        return filtered


# =============================================================================
# Monitor Base Class
# =============================================================================


class MonitorBase(BaseComponent, ABC):
    """
    Base class for monitoring and tracking components.

    Provides:
    - Metric recording
    - Alert thresholds
    - History tracking
    - Report generation

    Subclasses should implement:
    - record(): Record a new data point
    - get_metrics(): Get current metrics
    """

    def __init__(
        self,
        name: str,
        max_history: int = 10000,
        enabled: bool = True,
    ):
        """
        Initialize monitor.

        Args:
            name: Monitor name
            max_history: Maximum history entries to keep
            enabled: Whether monitor is enabled
        """
        super().__init__(name, enabled)
        self.max_history = max_history
        self._history: list[dict[str, Any]] = []

        # Alert thresholds
        self._alert_thresholds: dict[str, float] = {}

        # Callbacks
        self.on_alert: Callable[[str, Any], None] | None = None

    @abstractmethod
    def record(self, *args, **kwargs) -> None:
        """Record a new data point."""
        pass

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        pass

    def set_threshold(self, metric: str, value: float) -> None:
        """Set an alert threshold for a metric."""
        self._alert_thresholds[metric] = value
        self._logger.debug(f"Threshold set: {metric} = {value}")

    def _check_thresholds(self, metrics: dict[str, Any]) -> None:
        """Check metrics against thresholds and trigger alerts."""
        for metric, threshold in self._alert_thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                self._trigger_alert(metric, metrics[metric], threshold)

    def _trigger_alert(self, metric: str, value: Any, threshold: float) -> None:
        """Trigger an alert."""
        message = f"{metric} ({value}) exceeded threshold ({threshold})"
        self._logger.warning(f"ALERT: {message}")
        if self.on_alert:
            self.on_alert(metric, value)

    def _add_to_history(self, entry: dict[str, Any]) -> None:
        """Add entry to history with size management."""
        self._history.append({"timestamp": datetime.now().isoformat(), **entry})
        # Trim history if needed
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

    def get_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent history entries."""
        return self._history[-limit:]

    def clear_history(self) -> None:
        """Clear all history."""
        self._history.clear()


# =============================================================================
# Analyzer Base Class
# =============================================================================


@dataclass
class AnalysisResult:
    """Result of an analysis operation."""

    analysis_type: str
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    recommendations: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analysis_type": self.analysis_type,
            "score": self.score,
            "confidence": self.confidence,
            "recommendations": self.recommendations,
            "details": self.details,
        }


class AnalyzerBase(BaseComponent, ABC):
    """
    Base class for analysis and evaluation components.

    Provides:
    - Analysis interface
    - Caching
    - Multi-source aggregation
    - Confidence tracking

    Subclasses should implement:
    - analyze(): Main analysis logic
    """

    def __init__(
        self,
        name: str,
        cache_ttl_seconds: int = 300,
        enabled: bool = True,
    ):
        """
        Initialize analyzer.

        Args:
            name: Analyzer name
            cache_ttl_seconds: Cache time-to-live in seconds
            enabled: Whether analyzer is enabled
        """
        super().__init__(name, enabled)
        self.cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[datetime, AnalysisResult]] = {}

    @abstractmethod
    def analyze(self, *args, **kwargs) -> AnalysisResult:
        """
        Perform analysis.

        Returns:
            AnalysisResult with analysis outcome
        """
        pass

    def get_cached(self, cache_key: str) -> AnalysisResult | None:
        """Get cached analysis result if still valid."""
        if cache_key in self._cache:
            timestamp, result = self._cache[cache_key]
            age = (datetime.now() - timestamp).total_seconds()
            if age < self.cache_ttl:
                self._logger.debug(f"Cache hit for {cache_key}")
                return result
            # Expired
            del self._cache[cache_key]
        return None

    def cache_result(self, cache_key: str, result: AnalysisResult) -> None:
        """Cache an analysis result."""
        self._cache[cache_key] = (datetime.now(), result)

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Configurations
    "BaseConfig",
    "ComponentStats",
    # Base components
    "BaseComponent",
    # Executor
    "ExecutionResult",
    "ExecutorBase",
    # Scanner
    "ScanResult",
    "ScannerBase",
    # Monitor
    "MonitorBase",
    # Analyzer
    "AnalysisResult",
    "AnalyzerBase",
]
