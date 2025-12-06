"""Observability Module.

Layer: 1 (Infrastructure)
May import from: Layer 0 (utils)
May be imported by: Layers 2-4

Provides comprehensive observability infrastructure for LLM agents:
- OpenTelemetry-compatible tracing
- Token usage metrics and cost tracking
- Real-time metrics aggregation
- System health monitoring
- AgentOps integration (UPGRADE-015)
- Decision tracing (UPGRADE-015)
- Metrics collection (UPGRADE-015)

UPGRADE-014 Category 2: Observability & Debugging
UPGRADE-015 Phase 6: Extended Observability

Usage:
    from observability import (
        # Tracing
        LLMTracer,
        GenAISpan,
        GenAISystem,
        SpanKind,
        create_tracer,
        get_global_tracer,

        # Token Metrics
        TokenUsageTracker,
        TokenUsageRecord,
        TokenUsageSummary,
        create_tracker,
        get_global_tracker,
        record_usage,

        # Metrics Aggregation
        MetricsAggregator,
        MetricType,
        WindowSize,
        AggregatedMetric,
        SystemHealth,
        create_aggregator,
        get_global_aggregator,
        record_metric,
        record_llm_request,
        record_agent_decision,
        record_tool_call,
    )

    # Quick start - trace an LLM call
    tracer = get_global_tracer()
    with tracer.start_span("analyze", SpanKind.CLIENT, "analyst", "claude-3-sonnet"):
        # LLM call here
        pass

    # Record token usage
    record_usage(
        agent_name="analyst",
        model="claude-3-sonnet",
        input_tokens=500,
        output_tokens=1000,
    )

    # Get system health
    aggregator = get_global_aggregator()
    health = aggregator.get_system_health()
    print(f"Status: {health.status}, Score: {health.score}")
"""

# OTel Tracer
# UPGRADE-015: AgentOps Client
# Agent Performance Leaderboard (Phase 6)
from observability.agent_leaderboard import (
    AgentRanking,
    TrendDirection,
    get_leaderboard,
    print_leaderboard,
)
from observability.agentops_client import (
    AgentEvent,
    AgentOpsClient,
    EventType,
    SessionStats,
    agent_session,
    create_agentops_client,
)

# Anomaly Detection (Phase 6)
from observability.anomaly_detector import (
    AlertSeverity,
    Anomaly,
    AnomalyDetector,
    AnomalyType,
    run_anomaly_detection,
)

# UPGRADE-015: Decision Tracer
from observability.decision_tracer import (
    Decision,
    DecisionCategory,
    DecisionOutcome,
    DecisionTracer,
    ReasoningStep,
    create_decision_tracer,
)

# Exception Logging (Exception Refactor Phase 5)
from observability.exception_logger import (
    ExceptionAggregator,
    create_exception_aggregator,
    exception_handler,
    log_exception,
)

# UPGRADE-015: Metrics Collection
from observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    Timer,
    create_metrics_registry,
    get_default_registry,
    get_trading_metrics,
)

# Metrics Aggregator
from observability.metrics_aggregator import (
    AggregatedMetric,
    # Data Classes
    MetricPoint,
    # Main Classes
    MetricsAggregator,
    MetricSeries,
    # Enums
    MetricType,
    SystemHealth,
    WindowSize,
    # Factory Functions
    create_aggregator,
    get_global_aggregator,
    record_agent_decision,
    record_llm_request,
    # Convenience Functions
    record_metric,
    record_tool_call,
)
from observability.otel_tracer import (
    GenAISpan,
    # Enums
    GenAISystem,
    # Main Classes
    LLMTracer,
    SpanKind,
    SpanStatus,
    # Data Classes
    TokenUsage,
    # Factory Functions
    create_tracer,
    get_global_tracer,
)

# Token Metrics
from observability.token_metrics import (
    # Cost Configuration
    MODEL_COSTS,
    # Data Classes
    TokenUsageRecord,
    TokenUsageSummary,
    # Main Classes
    TokenUsageTracker,
    # Factory Functions
    create_tracker,
    get_global_tracker,
    get_model_cost,
    record_usage,
)


__all__ = [
    # OTel Tracer
    "GenAISystem",
    "SpanKind",
    "SpanStatus",
    "TokenUsage",
    "GenAISpan",
    "LLMTracer",
    "create_tracer",
    "get_global_tracer",
    # Token Metrics
    "MODEL_COSTS",
    "get_model_cost",
    "TokenUsageRecord",
    "TokenUsageSummary",
    "TokenUsageTracker",
    "create_tracker",
    "get_global_tracker",
    "record_usage",
    # Metrics Aggregator
    "MetricType",
    "WindowSize",
    "MetricPoint",
    "MetricSeries",
    "AggregatedMetric",
    "SystemHealth",
    "MetricsAggregator",
    "create_aggregator",
    "get_global_aggregator",
    "record_metric",
    "record_llm_request",
    "record_agent_decision",
    "record_tool_call",
    # UPGRADE-015: AgentOps Client
    "AgentOpsClient",
    "AgentEvent",
    "EventType",
    "SessionStats",
    "create_agentops_client",
    "agent_session",
    # UPGRADE-015: Decision Tracer
    "DecisionTracer",
    "Decision",
    "DecisionCategory",
    "DecisionOutcome",
    "ReasoningStep",
    "create_decision_tracer",
    # UPGRADE-015: Metrics Collection
    "MetricsRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    "create_metrics_registry",
    "get_default_registry",
    "get_trading_metrics",
    # Phase 6: Agent Performance Leaderboard
    "AgentRanking",
    "TrendDirection",
    "get_leaderboard",
    "print_leaderboard",
    # Phase 6: Anomaly Detection
    "AlertSeverity",
    "Anomaly",
    "AnomalyDetector",
    "AnomalyType",
    "run_anomaly_detection",
    # Exception Logging (Exception Refactor Phase 5)
    "log_exception",
    "exception_handler",
    "ExceptionAggregator",
    "create_exception_aggregator",
]
