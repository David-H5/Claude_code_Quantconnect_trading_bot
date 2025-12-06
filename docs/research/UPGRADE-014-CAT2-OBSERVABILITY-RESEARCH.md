# UPGRADE-014 Category 2: Observability & Debugging Research

## Phase 0 Research - December 3, 2025

**Search Date**: December 3, 2025 at ~12:00 PM EST

---

## Research Summary

### 1. OpenTelemetry GenAI Semantic Conventions

**Search Query**: "OpenTelemetry GenAI semantic conventions LLM observability Python 2025"

**Key Sources**:
1. [OpenTelemetry for Generative AI (Published: 2024)](https://opentelemetry.io/blog/2024/otel-generative-ai/)
2. [OpenLLMetry GitHub (Published: 2024-2025)](https://github.com/traceloop/openllmetry) - 20k+ stars
3. [AI Agent Observability - OpenTelemetry (Published: March 2025)](https://opentelemetry.io/blog/2025/ai-agent-observability/)
4. [Semantic Conventions for GenAI (Published: 2025)](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
5. [GenAI Agentic Systems Issue #2664 (Published: 2025)](https://github.com/open-telemetry/semantic-conventions/issues/2664)

**Key Findings**:
- OpenTelemetry now has official GenAI semantic conventions
- OpenLLMetry conventions are now part of OpenTelemetry standard
- Key attributes: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
- Python instrumentation library available in `opentelemetry-python-contrib`
- Technology-specific conventions for Azure AI, OpenAI, AWS Bedrock

### 2. AI Agent Observability Tools & Patterns

**Search Query**: "autonomous AI agent monitoring observability dashboard real-time metrics Python 2025"

**Key Sources**:
1. [AgentOps GitHub (Published: 2024-2025)](https://github.com/AgentOps-AI/agentops) - Python SDK
2. [Langfuse - LLM Observability Platform (Published: 2024-2025)](https://langfuse.com/blog/2024-07-ai-agent-observability-with-langfuse)
3. [15 AI Agent Observability Tools (Published: 2025)](https://research.aimultiple.com/agentic-monitoring/)
4. [Datadog LLM Observability (Published: 2025)](https://www.datadoghq.com/blog/openai-agents-llm-observability/)
5. [Arize LLM Observability Platform (Published: 2025)](https://arize.com/)

**Key Findings**:
- AgentOps: One-line Python SDK for agent observability, session replays, cost tracking
- Langfuse: Open-source, captures inputs/outputs/tool usage/latencies/costs
- Key metrics: latency, error rates, token usage, cost, tool usage statistics
- Dashboard essentials: total requests, costs, error rates, model usage, latency trends
- Multi-agent interaction tracking is critical

### 3. Self-Monitoring & Auto-Remediation Agents

**Search Query**: "AI agent self-monitoring watchdog autonomous remediation error detection Python"

**Key Sources**:
1. [Agentic AI for DevOps (Published: July 2025)](https://digitalthoughtdisruption.com/2025/07/31/agentic-ai-devops-real-world-implementation/)
2. [Watchdog Elite GitHub (Published: 2025)](https://github.com/SovArcNeo/watchdog_elite_v2.0)
3. [AI-Powered Auto-Remediation (Published: 2025)](https://medium.com/@cortilliusmckinney/building-an-agentic-devops-pipeline-on-gcp-with-ai-powered-auto-remediation-eccd48e513ec)
4. [Datadog Watchdog AI (Published: 2025)](https://docs.datadoghq.com/watchdog/)

**Key Findings**:
- Monitor Agent pattern: watches metrics, detects anomalies, triggers remediations
- Event-driven architecture for real-time reactions
- AI safety reviewer for assessing proposed fixes
- SuperAGI: scalable, self-monitoring agent ecosystem with dashboard
- Built-in tracing/logging enables faster error detection

---

## Codebase Analysis

### What Already Exists:

| Component | Status | Location |
|-----------|--------|----------|
| **Decision Logger** | ✅ Implemented | `llm/decision_logger.py` (728 lines) |
| **Reasoning Logger** | ✅ Implemented | `llm/reasoning_logger.py` (571 lines) |
| **Structured Logger** | ✅ Implemented | `utils/structured_logger.py` |
| **Agent Metrics** | ✅ Implemented | `evaluation/agent_metrics.py` |
| **Agent Metrics Widget** | ✅ Implemented | `ui/agent_metrics_widget.py` |
| **Dashboard** | ✅ Implemented | `ui/dashboard.py` |
| **OpenTelemetry** | ❌ Not implemented | No OTel integration |
| **Autonomous Monitor Agent** | ❌ Not implemented | No self-monitoring agent |
| **Real-time Metrics Dashboard** | ⚠️ Partial | Widget exists but not real-time streaming |

### Gaps Identified:

1. **OpenTelemetry Integration**:
   - No OTel semantic conventions for LLM calls
   - No trace/span infrastructure
   - No metrics export to OTel collectors

2. **Agent Decision Logging**:
   - ✅ Already exists but needs enhancement
   - ⚠️ Missing: tool call parameter logging at OTel level
   - ⚠️ Missing: state change tracking with spans

3. **Autonomous Observability Agent**:
   - ❌ No self-monitoring agent exists
   - ❌ No anomaly detection on agent metrics
   - ❌ No auto-remediation capability

4. **Real-time Dashboard**:
   - ⚠️ Widget exists but needs streaming updates
   - ❌ Missing: token usage tracking display
   - ❌ Missing: live error rate monitoring

---

## Enhancement Opportunities

### Priority 1: OpenTelemetry GenAI Integration

Create OTel instrumentation following semantic conventions:

```python
# Key attributes to implement
gen_ai.system = "anthropic"  # or "openai"
gen_ai.request.model = "claude-3-opus"
gen_ai.request.max_tokens = 4096
gen_ai.response.id = "msg_123"
gen_ai.usage.input_tokens = 150
gen_ai.usage.output_tokens = 300
gen_ai.usage.total_tokens = 450
```

### Priority 2: Observability Agent

Create a monitoring agent that:
- Continuously monitors other agents' health
- Detects anomalies in metrics (error rates, latencies)
- Diagnoses issues using pattern matching
- Triggers alerts or auto-remediation

### Priority 3: Enhanced Dashboard

Add real-time capabilities:
- WebSocket/SSE for live updates
- Token usage charts
- Error rate trends
- Agent health status grid

---

## Research Gate: PASSED

- [x] Searched for OpenTelemetry GenAI conventions
- [x] Searched for agent observability tools
- [x] Searched for self-monitoring patterns
- [x] Analyzed existing codebase
- [x] Identified gaps vs research
- [x] Documented findings with timestamps

**Proceed to Phase 1: Upgrade Path Definition**

---

## Phase 1: Upgrade Path

### Target State

Create a comprehensive observability layer for AI agents with:
1. **OpenTelemetry GenAI integration** - Standardized telemetry for LLM calls
2. **Enhanced agent decision logging** - Full trace of all agent actions
3. **Autonomous observability agent** - Self-monitoring and anomaly detection
4. **Real-time metrics streaming** - Live dashboard updates

### Scope

| Item | In Scope | Notes |
|------|----------|-------|
| OTel GenAI Tracer | ✅ | Trace spans for LLM calls |
| Token/Cost Metrics | ✅ | Usage tracking |
| Agent Monitor | ✅ | Watchdog agent |
| Anomaly Detection | ✅ | Simple threshold-based |
| Dashboard Streaming | ❌ | Deferred (UI complex) |
| External OTel Export | ❌ | Future enhancement |
| Full auto-remediation | ❌ | Basic alerts only |

### Success Criteria

- [x] LLM calls emit OTel-compatible spans
- [x] Token usage tracked per agent/call
- [x] Monitor agent detects anomalies
- [x] Tests cover all new components

---

## Phase 2: Implementation Checklist

### 2.1 OpenTelemetry GenAI Tracer (P0) - COMPLETE

- [x] Create `observability/otel_tracer.py` - 607 lines
- [x] Implement `GenAISpan` dataclass with semantic conventions
- [x] Create `LLMTracer` class for tracking LLM calls
- [x] Add span attributes: model, tokens, latency, status
- [x] Integrate with existing `TradingAgent.react_loop()`
- [x] Create `tests/test_otel_tracer.py`

### 2.2 Token Usage Metrics (P0) - COMPLETE

- [x] Create `observability/token_metrics.py` - 466 lines
- [x] Implement `TokenUsageTracker` class
- [x] Track input/output/total tokens per call
- [x] Track cost estimates per model
- [x] Add aggregation by agent, time window
- [x] Create `tests/test_token_metrics.py`

### 2.3 Observability Agent (P1) - COMPLETE

- [x] Create `llm/agents/observability_agent.py` - 637 lines
- [x] Implement `ObservabilityAgent` class extending TradingAgent
- [x] Add metric collection from all registered agents
- [x] Implement anomaly detection (error rate, latency spikes)
- [x] Add alert generation with severity levels
- [x] Create `tests/test_observability_agent.py`

### 2.4 Metrics Aggregator (P1) - COMPLETE

- [x] Create `observability/metrics_aggregator.py` - 701 lines
- [x] Implement real-time metric aggregation
- [x] Add sliding windows for rates (1m, 5m, 15m)
- [x] Create exportable metrics format
- [x] Create `tests/test_metrics_aggregator.py`

### 2.5 Package Integration (P1) - COMPLETE

- [x] Create `observability/__init__.py` with exports
- [x] Update `llm/agents/__init__.py` for observability agent
- [x] Add documentation in research file

### Total: 2562 lines implemented + 1896 lines tests

---

## Related Documents

- [Main Upgrade Document](UPGRADE-014-AUTONOMOUS-AGENT-ENHANCEMENTS.md)
- [Progress Tracker](../../claude-progress.txt)
