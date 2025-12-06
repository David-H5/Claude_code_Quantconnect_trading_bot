"""
Metrics Collectors

Specialized metric collectors for different domains:
- token: LLM token usage and cost tracking
- agent: AI agent performance metrics
- trading: Advanced trading performance metrics
- execution: Order execution quality metrics
- system: System resource metrics

Each collector provides domain-specific metrics that integrate
with the unified metrics infrastructure.
"""

from observability.metrics.collectors.token import (
    MODEL_COSTS,
    TokenUsageRecord,
    TokenUsageSummary,
    TokenUsageTracker,
    create_tracker,
    get_global_tracker,
    get_model_cost,
    record_usage,
)


# Legacy alias
TokenMetricsTracker = TokenUsageTracker
create_token_tracker = create_tracker

from observability.metrics.collectors.agent import (
    AgentMetrics,
    AgentMetricsTracker,
    DecisionRecord,
    MetricCategory,
    create_metrics_tracker,
    generate_metrics_report,
)
from observability.metrics.collectors.execution import (
    ExecutionDashboard,
    ExecutionQualityTracker,
    OrderRecord,
    OrderStatus,
    create_execution_tracker,
)
from observability.metrics.collectors.trading import (
    AdvancedTradingMetrics,
    Trade,
    calculate_advanced_trading_metrics,
)


__all__ = [
    # Token metrics
    "TokenUsageTracker",
    "TokenMetricsTracker",  # Legacy alias
    "TokenUsageRecord",
    "TokenUsageSummary",
    "MODEL_COSTS",
    "get_model_cost",
    "create_tracker",
    "create_token_tracker",  # Legacy alias
    "get_global_tracker",
    "record_usage",
    # Agent metrics
    "AgentMetricsTracker",
    "AgentMetrics",
    "DecisionRecord",
    "MetricCategory",
    "create_metrics_tracker",
    "generate_metrics_report",
    # Trading metrics
    "Trade",
    "AdvancedTradingMetrics",
    "calculate_advanced_trading_metrics",
    # Execution metrics
    "ExecutionQualityTracker",
    "ExecutionDashboard",
    "OrderRecord",
    "OrderStatus",
    "create_execution_tracker",
]
