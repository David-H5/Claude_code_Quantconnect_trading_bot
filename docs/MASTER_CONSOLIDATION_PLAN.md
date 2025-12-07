# Master Consolidation Plan

A comprehensive plan to deduplicate, consolidate, and enhance the QuantConnect Trading Bot by merging existing systems with the Multi-Agent Enhancement Plan.

---

## Executive Summary

### Current State Analysis

| Category | Existing Systems | Duplicates Found | Action Required |
|----------|-----------------|------------------|-----------------|
| **Agent Framework** | 15+ agents in llm/agents/ | Moderate overlap | Consolidate, add Claude SDK subagents |
| **Orchestration** | 2 systems (hooks + evaluation) | Low overlap | Merge into unified orchestrator |
| **Sentiment Analysis** | 5 separate systems | HIGH duplication | Consolidate to single interface |
| **Risk Management** | 4+ enforcement points | HIGH duplication | Unify enforcement chain |
| **Metrics** | 3+ systems | HIGH duplication | Consolidate to observability/ |
| **News Analysis** | 5 modules | HIGH duplication | Centralize |
| **Anomaly Detection** | 4+ implementations | HIGH duplication | Create unified interface |
| **Alerts** | 4 systems | Moderate overlap | Consolidate to observability/alerting/ |

### Key Findings

1. **Already Exists (No New Code Needed):**
   - TradingAgent base class with ReAct loop
   - AgentRegistry with capability discovery
   - Supervisor + Risk + Sentiment + Technical agents
   - Circuit breaker with 3-state design
   - Comprehensive orchestration in agent_orchestrator.py

2. **Needs Consolidation (Deduplicate):**
   - Sentiment analysis (5 → 1 system)
   - News analysis (5 → 1 system)
   - Metrics collection (3 → 1 system)
   - Anomaly detection (4 → 1 interface)

3. **Needs Addition (From Enhancement Plan):**
   - Claude SDK subagent definitions (.claude/agents/*.md)
   - MCP server integration
   - Agent skills system (.claude/skills/*.md)

---

## Part 1: System Mapping

### Enhancement Plan → Existing Code

| Enhancement Plan Item | Existing Implementation | Gap Analysis |
|----------------------|------------------------|--------------|
| **Market Analyst Subagent** | `llm/agents/technical_analyst.py` | Add Claude SDK .md wrapper |
| **Sentiment Scanner Subagent** | `llm/agents/sentiment_analyst.py` + `llm/sentiment.py` | Consolidate, add .md wrapper |
| **Risk Guardian Subagent** | `llm/agents/risk_managers.py` + `models/risk_manager.py` | Consolidate, add .md wrapper |
| **Execution Manager Subagent** | `execution/smart_execution.py` | Add .md wrapper |
| **Research Compiler Subagent** | `evaluation/orchestration_pipeline.py` | Add .md wrapper |
| **Orchestrator (Opus)** | `llm/agents/supervisor.py` + `.claude/hooks/agents/agent_orchestrator.py` | Already exists, needs unification |
| **MCP Integration** | Not implemented | **NEW** - add to .claude/settings.json |
| **Skills System** | Partial in orchestrator templates | **NEW** - create .claude/skills/*.md |
| **Circuit Breaker** | `models/circuit_breaker.py` | EXISTS - already comprehensive |

---

## Part 2: Consolidation Strategy

### 2.1 Sentiment Analysis Consolidation

**Current State (5 systems):**
```
llm/sentiment.py                    # Core FinBERT analyzer
llm/sentiment_filter.py             # Post-processing (55KB)
llm/reddit_sentiment.py             # Reddit-specific
llm/agents/sentiment_analyst.py     # Agent wrapper
llm/emotion_detector.py             # Emotion detection
```

**Target State (1 unified system):**
```
llm/sentiment/
├── __init__.py                     # Public API
├── core.py                         # FinBERT + unified interface
├── providers/
│   ├── finbert.py                  # FinBERT backend
│   ├── reddit.py                   # Reddit backend
│   └── news.py                     # News backend
├── filters.py                      # Consolidated filtering
└── agent.py                        # Agent wrapper (uses core.py)
```

**Migration Steps:**
1. Create `llm/sentiment/` package
2. Move core FinBERT logic to `providers/finbert.py`
3. Create unified `SentimentResult` dataclass
4. Update `llm/agents/sentiment_analyst.py` to use new package
5. Add deprecation warnings to old modules
6. Update all imports across codebase

### 2.2 News Analysis Consolidation

**Current State (5 modules):**
```
llm/news_analyzer.py                # General analysis
llm/news_processor.py               # Pre-processing
llm/news_alert_manager.py           # Alerts
llm/agents/news_analyst.py          # Agent wrapper
scanners/movement_scanner.py        # Inline news analysis
```

**Target State (1 unified system):**
```
llm/news/
├── __init__.py                     # Public API
├── analyzer.py                     # Core analysis
├── processor.py                    # Pre-processing
├── alerts.py                       # Alert generation
└── agent.py                        # Agent wrapper
```

**Migration Steps:**
1. Create `llm/news/` package
2. Consolidate analysis logic
3. Remove inline news analysis from movement_scanner.py
4. Create `NewsSignal` dataclass for unified output

### 2.3 Metrics Consolidation

**Current State (3+ systems):**
```
evaluation/metrics.py               # Agent metrics
evaluation/agent_metrics.py         # Extended metrics
evaluation/advanced_trading_metrics.py  # STOCKBENCH metrics
observability/metrics.py            # DEPRECATED re-export
observability/metrics_aggregator.py # DEPRECATED re-export
execution/execution_quality_metrics.py  # DEPRECATED re-export
```

**Target State (1 canonical location):**
```
observability/metrics/
├── base.py                         # Counter, Gauge, Histogram, Timer
├── aggregator.py                   # Real-time aggregation
├── collectors/
│   ├── agent.py                    # Agent metrics (move from evaluation/)
│   ├── trading.py                  # Trading metrics
│   ├── execution.py                # Execution metrics
│   └── token.py                    # LLM token metrics
└── exporters/
    ├── prometheus.py
    ├── json_exporter.py
    └── csv_exporter.py
```

**Migration Steps:**
1. Move `evaluation/metrics.py` logic to `observability/metrics/collectors/agent.py`
2. Delete deprecated re-export files
3. Update all imports to use `observability.metrics`
4. Remove `evaluation/agent_metrics.py` (merge into collectors)

### 2.4 Anomaly Detection Unification

**Current State (4 different types):**
```
models/anomaly_detector.py          # Returns AnomalyResult
observability/anomaly_detector.py   # Returns Anomaly
execution/spread_anomaly.py         # Returns SpreadAnomaly
scanners/unusual_activity_scanner.py # Returns UnusualActivityAlert
```

**Target State (unified interface):**
```
models/anomaly/
├── __init__.py                     # Unified AnomalyEvent type
├── base.py                         # Base detector interface
├── market.py                       # Market regime anomalies
├── agent.py                        # Agent behavior anomalies
├── spread.py                       # Spread anomalies
└── activity.py                     # Unusual activity
```

**Unified Type:**
```python
@dataclass
class AnomalyEvent:
    type: AnomalyType  # MARKET, AGENT, SPREAD, ACTIVITY
    severity: Severity  # LOW, MEDIUM, HIGH, CRITICAL
    source: str
    timestamp: datetime
    details: dict
    recommended_action: str | None
```

### 2.5 Risk Management Unification

**Current State (4 enforcement points):**
```
models/circuit_breaker.py           # Hard stops
models/risk_manager.py              # Portfolio limits
llm/agents/risk_managers.py         # Agent vetoes
execution/pre_trade_validator.py    # Pre-trade checks
```

**Issue:** Unclear execution order, potential for bypassing

**Target State (clear chain):**
```
┌─────────────────────────────────────────────────────────────┐
│                    RISK ENFORCEMENT CHAIN                   │
├─────────────────────────────────────────────────────────────┤
│ 1. Circuit Breaker Check (models/circuit_breaker.py)       │
│    └─ If OPEN → Block all trades immediately               │
│                                                             │
│ 2. Portfolio Risk Check (models/risk_manager.py)           │
│    └─ Daily loss, drawdown, position limits                │
│                                                             │
│ 3. Pre-Trade Validation (execution/pre_trade_validator.py) │
│    └─ Order-specific checks, liquidity, data freshness     │
│                                                             │
│ 4. Agent Risk Review (llm/agents/risk_managers.py)         │
│    └─ LLM-based contextual review (cannot be overridden)   │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**
Create `models/risk_chain.py`:
```python
class RiskEnforcementChain:
    def __init__(self):
        self.circuit_breaker = get_circuit_breaker()
        self.portfolio_risk = RiskManager()
        self.pre_trade = PreTradeValidator()
        self.agent_risk = PositionRiskManager()

    def validate_trade(self, order: Order) -> RiskResult:
        # Executes all checks in order
        # Returns first failure or final approval
```

---

## Part 3: New Features (From Enhancement Plan)

### 3.1 Claude SDK Subagent Definitions

Create `.claude/agents/` directory with subagent definitions that wrap existing code:

```markdown
# .claude/agents/market-analyst.md

## Role
Technical analysis and pattern recognition using existing TechnicalAnalystAgent.

## Backend
Uses: llm/agents/technical_analyst.py

## Tools
- Read indicator data (indicators/technical_alpha.py)
- Access chart patterns
- Query historical support/resistance

## Output Format
{
  "symbol": "SPY",
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": 0.0-1.0,
  "patterns": ["double_bottom", "rsi_oversold"],
  "key_levels": {"support": 580, "resistance": 595}
}

## Constraints
- Read-only market access
- Must include confidence scores
- 5 second response SLA
```

**Files to Create:**
- `.claude/agents/market-analyst.md` → wraps `technical_analyst.py`
- `.claude/agents/sentiment-scanner.md` → wraps consolidated `llm/sentiment/`
- `.claude/agents/risk-guardian.md` → wraps `risk_managers.py`
- `.claude/agents/execution-manager.md` → wraps `smart_execution.py`
- `.claude/agents/research-compiler.md` → wraps `evaluation/orchestration_pipeline.py`

### 3.2 MCP Server Integration

Add to `.claude/settings.json`:

```json
{
  "mcpServers": {
    "market-data": {
      "type": "http",
      "url": "http://localhost:8080/mcp",
      "description": "Real-time market quotes and options chains",
      "tools": ["get_quote", "get_options_chain", "get_historical"]
    },
    "postgres": {
      "type": "stdio",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-postgres"],
      "env": {"DATABASE_URL": "${TRADING_DB_URL}"},
      "description": "Trade history and backtest results"
    },
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-github"],
      "description": "Code versioning and issue tracking"
    }
  }
}
```

**MCP Server Priority:**
| Server | Purpose | Priority | Implementation |
|--------|---------|----------|----------------|
| Custom Market Data | Real-time quotes | P0 | Create `mcp_servers/market_data/` |
| Postgres | Trade history | P0 | Use existing @modelcontextprotocol |
| GitHub | Code tracking | P1 | Use existing @modelcontextprotocol |
| Slack | Alerts | P2 | Use existing @modelcontextprotocol |

### 3.3 Skills System

Create `.claude/skills/` with reusable operations:

```markdown
# .claude/skills/SKILL_backtest.md

## Trigger
"backtest", "evaluate strategy", "historical test", "walk forward"

## Backend
Uses: evaluation/orchestration_pipeline.py, evaluation/walk_forward_analysis.py

## Steps
1. Parse strategy parameters
2. Validate date range and symbols
3. Run evaluation pipeline
4. Generate performance report

## Required Tools
- Read, Write, Bash

## Output
{
  "sharpe_ratio": 1.45,
  "sortino_ratio": 1.82,
  "max_drawdown": -0.12,
  "win_rate": 0.58,
  "total_return": 0.24,
  "recommendation": "Strategy meets targets"
}
```

**Skills to Create:**
- `SKILL_backtest.md` → evaluation pipeline
- `SKILL_options_analysis.md` → options scanner + Greeks
- `SKILL_risk_check.md` → risk chain validation
- `SKILL_report_generator.md` → performance reports
- `SKILL_sentiment_scan.md` → consolidated sentiment

---

## Part 4: Implementation Roadmap

### Phase 1: Consolidation (Weeks 1-2)

| Day | Task | Files Affected |
|-----|------|----------------|
| 1-2 | Consolidate sentiment to `llm/sentiment/` | 5 files → 1 package |
| 3-4 | Consolidate news to `llm/news/` | 5 files → 1 package |
| 5 | Unify anomaly types in `models/anomaly/` | 4 files → 1 package |
| 6-7 | Consolidate metrics to `observability/metrics/` | 6 files → canonical |
| 8 | Create risk enforcement chain | New file |
| 9-10 | Delete deprecated re-exports | 3 files |

### Phase 2: Claude SDK Integration (Weeks 3-4)

| Day | Task | Files Affected |
|-----|------|----------------|
| 11-12 | Create `.claude/agents/` subagent definitions | 5 new files |
| 13-14 | Create `.claude/skills/` skill definitions | 5 new files |
| 15-16 | Setup MCP server configs | .claude/settings.json |
| 17-18 | Create market data MCP server | New directory |
| 19-20 | Integration testing | - |

### Phase 3: Enhancement (Weeks 5-6)

| Day | Task | Files Affected |
|-----|------|----------------|
| 21-23 | Parallel subagent execution | agent_orchestrator.py |
| 24-26 | Database MCP server setup | New directory |
| 27-28 | Agent monitoring dashboard | New files |
| 29-30 | Documentation updates | docs/*.md |

---

## Part 5: File-by-File Action Plan

### Files to DELETE (After Migration)
```
# After consolidation is complete and imports updated:
llm/sentiment.py              → Merged into llm/sentiment/
llm/sentiment_filter.py       → Merged into llm/sentiment/
llm/reddit_sentiment.py       → Merged into llm/sentiment/
llm/emotion_detector.py       → Merged into llm/sentiment/
llm/news_analyzer.py          → Merged into llm/news/
llm/news_processor.py         → Merged into llm/news/
llm/news_alert_manager.py     → Merged into llm/news/
evaluation/metrics.py         → Moved to observability/metrics/
evaluation/agent_metrics.py   → Moved to observability/metrics/

# Deprecated re-exports (delete after updating callers):
observability/metrics.py
observability/metrics_aggregator.py
execution/execution_quality_metrics.py
```

### Files to CREATE
```
# Consolidated packages:
llm/sentiment/__init__.py
llm/sentiment/core.py
llm/sentiment/providers/finbert.py
llm/sentiment/providers/reddit.py
llm/sentiment/providers/news.py
llm/sentiment/filters.py
llm/sentiment/agent.py

llm/news/__init__.py
llm/news/analyzer.py
llm/news/processor.py
llm/news/alerts.py
llm/news/agent.py

models/anomaly/__init__.py
models/anomaly/base.py
models/anomaly/market.py
models/anomaly/agent.py
models/anomaly/spread.py
models/anomaly/activity.py

models/risk_chain.py

# Claude SDK integration:
.claude/agents/market-analyst.md
.claude/agents/sentiment-scanner.md
.claude/agents/risk-guardian.md
.claude/agents/execution-manager.md
.claude/agents/research-compiler.md

.claude/skills/SKILL_backtest.md
.claude/skills/SKILL_options_analysis.md
.claude/skills/SKILL_risk_check.md
.claude/skills/SKILL_report_generator.md
.claude/skills/SKILL_sentiment_scan.md

# MCP servers:
mcp_servers/market_data/__init__.py
mcp_servers/market_data/server.py
mcp_servers/market_data/tools.py
```

### Files to MODIFY
```
# Update imports after consolidation:
llm/agents/sentiment_analyst.py   # Use llm.sentiment
llm/agents/news_analyst.py        # Use llm.news
scanners/movement_scanner.py      # Use llm.news (remove inline)
models/circuit_breaker.py         # Use models.anomaly
evaluation/orchestration_pipeline.py  # Use observability.metrics

# Update orchestrator:
.claude/hooks/agents/agent_orchestrator.py  # Add subagent routing

# Update settings:
.claude/settings.json             # Add MCP servers
```

---

## Part 6: Success Metrics

### Consolidation Metrics

| Metric | Before | Target | Measurement |
|--------|--------|--------|-------------|
| Sentiment analysis files | 5 | 1 package | File count |
| News analysis files | 5 | 1 package | File count |
| Metrics locations | 6 | 1 canonical | Import paths |
| Anomaly types | 4 different | 1 unified | Type definitions |
| Deprecated re-exports | 3 | 0 | File count |

### Enhancement Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Claude SDK subagents | 5 defined | .claude/agents/*.md count |
| Skills defined | 5+ | .claude/skills/*.md count |
| MCP servers | 3+ configured | .claude/settings.json |
| Test coverage | >80% | pytest --cov |

### Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Agent response latency | <5s | Average response time |
| Sentiment analysis time | <2s | Per-symbol analysis |
| Risk check time | <1s | Full chain validation |
| Import time | <3s | Module load time |

---

## Part 7: Risk Mitigation

### Migration Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking imports | HIGH | Add deprecation warnings first, maintain compatibility |
| Test failures | HIGH | Run full test suite after each consolidation step |
| Performance regression | MEDIUM | Benchmark before/after each phase |
| Lost functionality | HIGH | Create feature checklist, verify each feature migrated |

### Rollback Plan

1. **Git tags** before each phase: `pre-consolidation-phase-1`
2. **Feature flags** for new systems
3. **Parallel operation** during transition
4. **Automated tests** gate each merge

---

## Appendix A: Existing System Inventory

### Agent Systems (llm/agents/)
| File | Lines | Purpose | Keep/Merge |
|------|-------|---------|------------|
| base.py | ~800 | TradingAgent base class | KEEP |
| registry.py | ~400 | Agent discovery | KEEP |
| supervisor.py | ~500 | Orchestration | KEEP |
| risk_managers.py | ~400 | Risk approval | KEEP |
| sentiment_analyst.py | ~300 | Sentiment agent | KEEP (use consolidated) |
| technical_analyst.py | ~300 | Technical agent | KEEP |
| news_analyst.py | ~300 | News agent | KEEP (use consolidated) |
| traders.py | ~400 | Trader agents | KEEP |
| debate_mechanism.py | ~300 | Bull/Bear debate | KEEP |
| safe_agent_wrapper.py | ~200 | Safety wrapper | KEEP |

### Orchestration Systems
| File | Lines | Purpose | Keep/Merge |
|------|-------|---------|------------|
| .claude/hooks/agents/agent_orchestrator.py | ~2500 | Full orchestration | KEEP (primary) |
| evaluation/orchestration_pipeline.py | ~600 | Evaluation pipeline | KEEP (specialized) |

### Sentiment Systems (TO CONSOLIDATE)
| File | Lines | Purpose | Action |
|------|-------|---------|--------|
| llm/sentiment.py | ~200 | FinBERT core | → llm/sentiment/providers/finbert.py |
| llm/sentiment_filter.py | ~1500 | Filtering | → llm/sentiment/filters.py |
| llm/reddit_sentiment.py | ~300 | Reddit | → llm/sentiment/providers/reddit.py |
| llm/emotion_detector.py | ~200 | Emotions | → llm/sentiment/providers/emotion.py |
| llm/agents/sentiment_analyst.py | ~300 | Agent | → llm/sentiment/agent.py |

---

## Appendix B: Import Update Script

```python
# scripts/update_imports.py
IMPORT_MIGRATIONS = {
    # Old import → New import
    "from llm.sentiment import": "from llm.sentiment.core import",
    "from llm.news_analyzer import": "from llm.news.analyzer import",
    "from evaluation.metrics import": "from observability.metrics.collectors.agent import",
    "from observability.metrics import": "from observability.metrics.base import",
}

# Run with: python scripts/update_imports.py --dry-run
```

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Create Phase 1 branch**: `feature/consolidation-phase-1`
3. **Start with sentiment consolidation** (highest duplication)
4. **Run tests after each step**
5. **Tag releases**: `v2.0.0-consolidated`

---

## References

- [MULTI_AGENT_ENHANCEMENT_PLAN.md](./MULTI_AGENT_ENHANCEMENT_PLAN.md) - Original enhancement plan
- [GIT_MULTI_AGENT_WORKFLOW.md](./GIT_MULTI_AGENT_WORKFLOW.md) - Git workflow documentation
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Current architecture documentation
