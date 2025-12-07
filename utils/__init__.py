"""
Utility Functions and Helpers

Layer: 0 (Utilities)
This is the base layer. All other layers may import from utils.

This package contains general utility functions used across the project.
"""

from .alerting_service import (
    Alert,
    AlertAggregator,
    AlertCategory,
    AlertChannel,
    AlertingService,
    AlertSeverity,
    ConsoleChannel,
    DiscordChannel,
    EmailChannel,
    RateLimiter,
    SlackChannel,
    create_alerting_service,
)
from .calculations import (
    calculate_cagr,
    calculate_kelly_fraction,
    calculate_max_drawdown,
    calculate_position_size,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_volatility,
    calculate_win_rate,
)
from .error_handling import (
    ErrorAccumulator,
    fallback,
    retry,
    with_retry,
)
from .log_handlers import (
    AsyncHandler,
    CallbackHandler,
    CompressedRotatingFileHandler,
    ObjectStoreHandler,
    create_async_file_handler,
    create_callback_handler,
    create_object_store_handler,
    create_rotating_file_handler,
)
from .node_optimizer import (
    AlgorithmRequirements,
    NodeOptimizer,
    NodeSpec,
    NodeType,
    analyze_algorithm_requirements,
    print_recommendations,
)
from .object_store import (
    ObjectStoreManager,
    SentimentPersistence,
    StorageCategory,
    StoredObject,
    create_object_store_manager,
    create_sentiment_persistence,
)
from observability.monitoring.system.resource import (
    ResourceAlert,
    ResourceMetrics,
    ResourceMonitor,
    create_resource_monitor,
)
from .storage_monitor import (
    StorageAlert,
    StorageMonitor,
    create_storage_monitor,
)
from observability.logging.structured import (
    ExecutionEventType,
    LogCategory,
    LogEvent,
    LogLevel,
    RiskEventType,
    StrategyEventType,
    StructuredLogger,
    create_structured_logger,
    get_logger,
    set_logger,
)
from observability.monitoring.system.health import (
    HealthStatus,
    ServiceCheck,
    ServiceState,
    SystemMetrics,
    SystemMonitor,
    create_disk_check,
    create_http_check,
    create_memory_check,
    create_system_monitor,
)


__all__ = [
    "calculate_position_size",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "calculate_sortino_ratio",
    "calculate_win_rate",
    "calculate_profit_factor",
    "calculate_cagr",
    "calculate_volatility",
    "calculate_kelly_fraction",
    "ResourceMonitor",
    "ResourceMetrics",
    "ResourceAlert",
    "create_resource_monitor",
    "NodeOptimizer",
    "NodeSpec",
    "AlgorithmRequirements",
    "NodeType",
    "analyze_algorithm_requirements",
    "print_recommendations",
    "ObjectStoreManager",
    "StorageCategory",
    "StoredObject",
    "create_object_store_manager",
    # Sentiment Persistence (UPGRADE-014)
    "SentimentPersistence",
    "create_sentiment_persistence",
    "StorageMonitor",
    "StorageAlert",
    "create_storage_monitor",
    # Structured Logging (UPGRADE-009)
    "StructuredLogger",
    "LogEvent",
    "LogCategory",
    "LogLevel",
    "ExecutionEventType",
    "RiskEventType",
    "StrategyEventType",
    "create_structured_logger",
    "get_logger",
    "set_logger",
    # Log Handlers
    "CompressedRotatingFileHandler",
    "ObjectStoreHandler",
    "AsyncHandler",
    "CallbackHandler",
    "create_rotating_file_handler",
    "create_async_file_handler",
    "create_object_store_handler",
    "create_callback_handler",
    # Alerting Service (UPGRADE-013)
    "AlertSeverity",
    "AlertCategory",
    "Alert",
    "AlertChannel",
    "ConsoleChannel",
    "EmailChannel",
    "DiscordChannel",
    "SlackChannel",
    "RateLimiter",
    "AlertAggregator",
    "AlertingService",
    "create_alerting_service",
    # System Monitor (UPGRADE-013)
    "HealthStatus",
    "ServiceCheck",
    "ServiceState",
    "SystemMetrics",
    "SystemMonitor",
    "create_memory_check",
    "create_disk_check",
    "create_http_check",
    "create_system_monitor",
    # Error Handling (Exception Refactor Phase 4)
    "retry",
    "fallback",
    "ErrorAccumulator",
    "with_retry",
]
