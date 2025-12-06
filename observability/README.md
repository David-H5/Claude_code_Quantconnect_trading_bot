# Observability Module

UPGRADE-015 Phase 6: Comprehensive observability for AI trading agents.

## Overview

This module provides observability features for monitoring AI agent behavior,
tracking performance, and ensuring system health.

## Components

### AgentOps Client (`agentops_client.py`)

Integration with AgentOps for AI agent monitoring and analytics.

```python
from observability import create_agentops_client, agent_session

# Create client
client = create_agentops_client()

# Use session context manager
with agent_session(client, "trading_session") as session_id:
    client.record_tool_call("get_quote", {"symbol": "SPY"})
    client.record_decision("trade", "BUY", 0.85)
```

**Features:**
- Session management
- Event recording (tool calls, LLM calls, decisions, trades)
- Cost tracking
- Performance analytics

### Decision Tracer (`decision_tracer.py`)

Traces and records AI agent decision-making processes.

```python
from observability import create_decision_tracer, DecisionCategory, DecisionOutcome

tracer = create_decision_tracer()

# Start tracing a decision
decision_id = tracer.start_decision(
    category=DecisionCategory.TRADE,
    description="Should we buy SPY?",
    agent_name="technical_analyst",
)

# Add reasoning steps
tracer.add_reasoning_step(
    decision_id,
    "Analyzed RSI indicator",
    inputs={"rsi": 35},
    outputs={"signal": "oversold"},
    confidence=0.8,
)

# Complete decision
tracer.complete_decision(
    decision_id,
    outcome=DecisionOutcome.EXECUTED,
    final_confidence=0.85,
)
```

### Metrics Collection (`metrics.py`)

Collects and aggregates metrics for monitoring.

```python
from observability import create_metrics_registry, get_trading_metrics

# Get pre-defined trading metrics
metrics = get_trading_metrics()

# Record metrics
metrics["orders_total"].inc(symbol="SPY", side="buy", order_type="limit")
metrics["portfolio_value"].set(100000.0)
metrics["fill_latency"].observe(0.5)

# Time operations
with metrics["decision_latency"].time(agent="technical_analyst"):
    # Do decision making
    pass
```

### Token Metrics (`token_metrics.py`)

Tracks LLM token usage and costs (from UPGRADE-014).

### OpenTelemetry Tracer (`otel_tracer.py`)

Distributed tracing with OpenTelemetry (from UPGRADE-014).

### Metrics Aggregator (`metrics_aggregator.py`)

Real-time metrics aggregation (from UPGRADE-014).

## Configuration

### Environment Variables

```bash
# AgentOps API key
AGENTOPS_API_KEY=your_key_here

# Optional: OpenTelemetry
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=quantconnect-trading-bot
```

## Usage Examples

### Full Observability Setup

```python
from observability import (
    create_agentops_client,
    create_decision_tracer,
    create_metrics_registry,
    get_trading_metrics,
    agent_session,
)

# Initialize components
agentops = create_agentops_client()
tracer = create_decision_tracer()
registry = create_metrics_registry()
metrics = get_trading_metrics()

# Start session
with agent_session(agentops, "trading_session"):
    # Track a decision
    decision_id = tracer.start_decision(
        category=DecisionCategory.TRADE,
        description="Evaluate SPY trade",
        agent_name="strategy_agent",
    )

    # Record metrics
    metrics["decisions_total"].inc(agent="strategy_agent", decision_type="trade")

    # Make decision...
    tracer.complete_decision(decision_id, DecisionOutcome.EXECUTED, 0.85)

    # Record the trade
    agentops.record_trade("SPY", "buy", 100, 450.0)
    metrics["orders_total"].inc(symbol="SPY", side="buy", order_type="market")
```

### Export Metrics

```python
# Export to JSON
registry.export_json("metrics_export.json")

# Get summary
summary = registry.get_summary()
print(summary)

# Export decisions
tracer.export_decisions("decisions_export.json")
```

## Pre-defined Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `orders_total` | Counter | Total orders by symbol/side/type |
| `orders_filled` | Counter | Filled orders by symbol/side |
| `orders_cancelled` | Counter | Cancelled orders |
| `portfolio_value` | Gauge | Current portfolio value |
| `cash_balance` | Gauge | Current cash balance |
| `positions_count` | Gauge | Number of open positions |
| `daily_pnl` | Gauge | Daily profit/loss |
| `max_drawdown` | Gauge | Maximum drawdown |
| `fill_latency` | Histogram | Order fill latency |
| `slippage` | Histogram | Execution slippage (bps) |
| `decisions_total` | Counter | Agent decisions |
| `decision_latency` | Timer | Decision making time |
| `llm_calls_total` | Counter | LLM API calls |
| `llm_tokens_total` | Counter | LLM tokens used |

## Testing

```bash
pytest tests/observability/ -v
```

## Dependencies

Required:
- Python 3.10+

Optional (for full functionality):
- `agentops>=0.1.0` - AgentOps integration
- `opentelemetry-api>=1.20.0` - OpenTelemetry support
- `opentelemetry-sdk>=1.20.0`
- `opentelemetry-exporter-otlp>=1.20.0`

## Related Documentation

- [CLAUDE.md](../CLAUDE.md) - Project instructions
- [Token Metrics](token_metrics.py) - LLM cost tracking
- [Decision Logger](../llm/decision_logger.py) - Agent decision logging
