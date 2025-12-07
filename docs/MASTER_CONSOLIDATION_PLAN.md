# Master Consolidation Plan

A comprehensive plan to deduplicate, consolidate, and enhance the QuantConnect Trading Bot by merging existing systems with the Multi-Agent Enhancement Plan.

---

## Executive Summary

### Current State Analysis (Updated: 2025-12-06)

| Category | Existing Systems | Duplicates Found | Action Required | Status |
|----------|-----------------|------------------|-----------------|--------|
| **Agent Framework** | 15+ agents in llm/agents/ | Moderate overlap | Consolidate, add Claude SDK subagents | TODO |
| **Orchestration** | 2 systems (hooks + evaluation) | Low overlap | Merge into unified orchestrator | TODO |
| **Sentiment Analysis** | 5 separate systems | HIGH duplication | Consolidate to single interface | TODO |
| **Risk Management** | 4+ enforcement points | HIGH duplication | Unify enforcement chain | TODO |
| **Trading Monitors** | 4 scattered monitors | HIGH duplication | Consolidate to observability/ | ✅ DONE |
| **Deprecated Wrappers** | 4 wrapper files | Full duplication | Remove after migration | ✅ DONE |
| **News Analysis** | 5 modules | HIGH duplication | Centralize | TODO |
| **Anomaly Detection** | 4+ implementations | HIGH duplication | Create unified interface | TODO |
| **Config Documentation** | 2 config systems | None (different purposes) | Document boundaries | ✅ DONE |
| **Execution Module** | 4 overlapping modules | Moderate overlap | Review after patterns stabilize | ⏸️ DEFERRED |

### Recently Completed Work (2025-12-06)

The following items from the original plan have been completed by a parallel consolidation effort:

#### Phase 1-6 Completed Items:

1. **✅ Monitoring Consolidation** → `observability/monitoring/trading/` (VERIFIED 2025-12-06)
   - `execution/slippage_monitor.py` → `observability/monitoring/trading/slippage.py`
   - `models/greeks_monitor.py` → `observability/monitoring/trading/greeks.py`
   - `models/correlation_monitor.py` → `observability/monitoring/trading/correlation.py`
   - `models/var_monitor.py` → `observability/monitoring/trading/var.py`
   - Original files converted to deprecation wrappers with re-exports for backwards compatibility
   - Package `__init__.py` exports all monitors via public API

2. **✅ Deprecated Wrapper Removal** (VERIFIED 2025-12-06)
   - `utils/structured_logger.py` → REMOVED (canonical: `observability.logging.structured`)
   - `compliance/audit_logger.py` → REMOVED (canonical: `observability.logging.audit`)
   - `utils/system_monitor.py` → REMOVED (canonical: `observability.monitoring.system.health`)
   - `utils/resource_monitor.py` → REMOVED (canonical: `observability.monitoring.system.resource`)
   - Note: These files were fully deleted, not converted to wrappers (unlike trading monitors)

3. **✅ Config Documentation**
   - Documented that `config/` = Trading configuration
   - Documented that `utils/overnight_config.py` = Claude Code session configuration
   - Added cross-reference documentation

4. **✅ Decision/Reasoning Logger Integration** (VERIFIED 2025-12-06)
   - Core loggers: `llm/decision_logger.py` and `llm/reasoning_logger.py`
   - Adapters verified: `observability/logging/adapters/decision.py`, `observability/logging/adapters/reasoning.py`
   - DecisionLoggerAdapter and ReasoningLoggerAdapter implement AbstractLogger interface

5. **✅ Deprecated Directories Cleanup** (VERIFIED 2025-12-06)
   - `.claude/hooks/deprecated/` → REMOVED (directory does not exist)
   - `.claude/deprecated/` → REMOVED (directory does not exist)

6. **⏸️ Execution Module Review** - DEFERRED
   - Project in early development, execution patterns haven't stabilized
   - Deferred files: `smart_execution.py`, `spread_analysis.py`, `two_part_spread.py`, `option_strategies_executor.py`

### Key Findings

1. **Already Exists (No New Code Needed):** *(Verified 2025-12-06)*
   - TradingAgent base class with ReAct loop
   - AgentRegistry with capability discovery
   - Supervisor + Risk + Sentiment + Technical agents
   - Circuit breaker with 3-state design
   - Comprehensive orchestration in agent_orchestrator.py
   - Decision/Reasoning logger integration (adapters in `observability/logging/adapters/`)
   - Trading monitors → `observability/monitoring/trading/` (slippage, greeks, correlation, var)
   - Logging infrastructure → `observability/logging/` (structured, audit)

2. **Still Needs Consolidation (Deduplicate):**
   - Sentiment analysis (5 → 1 system)
   - News analysis (5 → 1 system)
   - Anomaly detection (4 → 1 interface)

3. **Needs Addition (From Enhancement Plan):**
   - Claude SDK subagent definitions (.claude/agents/*.md)
   - MCP server integration
   - Agent skills system (.claude/skills/*.md)
   - Risk enforcement chain (models/risk_chain.py)

### Related Documents & Coordination

| Document | Scope | Relationship |
|----------|-------|--------------|
| [FIX_GUIDE.md](FIX_GUIDE.md) | Algorithm bug fixes & code quality | **Parallel work** - do P0 first |
| [CONSOLIDATION_PLAN.md](CONSOLIDATION_PLAN.md) | Completed consolidation phases | ✅ Done - reference only |
| [CONSOLIDATION_CHANGELOG.md](CONSOLIDATION_CHANGELOG.md) | What was completed | ✅ Done - verified |
| [MULTI_AGENT_ENHANCEMENT_PLAN.md](MULTI_AGENT_ENHANCEMENT_PLAN.md) | Claude SDK vision | Input for Phase 2 |

**Execution Order with FIX_GUIDE:**
1. **FIX_GUIDE P0-1** (timestamp bug) → Do FIRST (critical)
2. **This plan Phase 1** (sentiment/news consolidation) → Can parallel with FIX_GUIDE P1
3. **FIX_GUIDE P1** (base class) → After or parallel with consolidation
4. **This plan Phase 2** (Claude SDK) → After consolidation complete

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

### 2.3 Metrics & Monitoring Consolidation (PARTIAL)

> **Note:** Trading monitors consolidation is ✅ COMPLETE. See `observability/monitoring/trading/`.
> Metrics collectors consolidation is still TODO.

#### ✅ COMPLETED: Trading Monitors

**New Canonical Location:** `observability/monitoring/trading/`

```python
# New canonical imports
from observability.monitoring.trading import SlippageMonitor, create_slippage_monitor
from observability.monitoring.trading import GreeksMonitor, create_greeks_monitor
from observability.monitoring.trading import CorrelationMonitor, create_correlation_monitor
from observability.monitoring.trading import VaRMonitor, create_var_monitor
```

**Migration Mapping:**
| Original | New Location | Status |
|----------|--------------|--------|
| `execution/slippage_monitor.py` | `observability/monitoring/trading/slippage.py` | ✅ DONE |
| `models/greeks_monitor.py` | `observability/monitoring/trading/greeks.py` | ✅ DONE |
| `models/correlation_monitor.py` | `observability/monitoring/trading/correlation.py` | ✅ DONE |
| `models/var_monitor.py` | `observability/monitoring/trading/var.py` | ✅ DONE |

Original files are now deprecation wrappers maintaining backwards compatibility.

#### TODO: Metrics Collectors

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

## Part 4: Implementation Roadmap (Updated: 2025-12-06)

### ✅ Completed Phases (Pre-requisite Work Done)

| Task | Status | Notes |
|------|--------|-------|
| Monitoring Consolidation | ✅ DONE | → `observability/monitoring/trading/` |
| Deprecated Wrapper Removal | ✅ DONE | Utils wrappers now point to observability/ |
| Config Documentation | ✅ DONE | config/ vs utils/overnight_config.py documented |
| Decision/Reasoning Logger | ✅ EXISTS | Already integrated in llm/ |
| Deprecated Directories | ✅ DONE | .claude/hooks/deprecated/ removed |
| Test Suite Verification | ✅ DONE | 3548 tests passing |

### Phase 1: Remaining Consolidation (Next Priority)

| Priority | Task | Files Affected | Effort |
|----------|------|----------------|--------|
| P0 | Consolidate sentiment to `llm/sentiment/` | 5 files → 1 package | 2 days |
| P0 | Consolidate news to `llm/news/` | 5 files → 1 package | 2 days |
| P1 | Unify anomaly types in `models/anomaly/` | 4 files → 1 package | 1 day |
| P1 | Create risk enforcement chain | New file | 1 day |
| P2 | Consolidate metrics collectors | 6 files → canonical | 2 days |

### Phase 2: Claude SDK Integration

| Priority | Task | Files Affected | Effort |
|----------|------|----------------|--------|
| P1 | Create `.claude/agents/` subagent definitions | 5 new files | 2 days |
| P1 | Create `.claude/skills/` skill definitions | 5 new files | 2 days |
| P2 | Setup MCP server configs | .claude/settings.json | 1 day |
| P2 | Create market data MCP server | New directory | 2 days |
| P2 | Integration testing | - | 2 days |

### Phase 3: Enhancement (Deferred until Phase 1-2 Complete)

| Priority | Task | Files Affected | Effort |
|----------|------|----------------|--------|
| P3 | Parallel subagent execution | agent_orchestrator.py | 3 days |
| P3 | Database MCP server setup | New directory | 2 days |
| P3 | Agent monitoring dashboard | New files | 2 days |
| P3 | Documentation updates | docs/*.md | 2 days |

### ⏸️ Deferred: Execution Module Review

**Reason:** Project in early development, execution patterns haven't stabilized

| Deferred Task | Size | Reason |
|--------------|------|--------|
| `smart_execution.py` review | 27KB | Patterns unstable |
| `spread_analysis.py` review | 18KB | Patterns unstable |
| `two_part_spread.py` review | 42KB | Patterns unstable |
| `option_strategies_executor.py` review | 26KB | Patterns unstable |

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

## Part 6: Success Metrics (Updated: 2025-12-06)

### ✅ Completed Metrics

| Metric | Before | After | Status |
|--------|--------|-------|--------|
| Trading monitor locations | 4 scattered | 1 canonical package | ✅ DONE |
| Deprecated wrappers | 4 | 0 (converted) | ✅ DONE |
| Config documentation | Undocumented | Fully documented | ✅ DONE |
| Test pass rate | - | 3548 passing | ✅ DONE |

### Remaining Consolidation Metrics

| Metric | Current | Target | Priority |
|--------|---------|--------|----------|
| Sentiment analysis files | 5 | 1 package | P0 |
| News analysis files | 5 | 1 package | P0 |
| Anomaly types | 4 different | 1 unified | P1 |
| Risk enforcement chain | None | 1 unified | P1 |
| Metrics collectors | 6 scattered | 1 canonical | P2 |

### Enhancement Metrics

| Metric | Current | Target | Priority |
|--------|---------|--------|----------|
| Claude SDK subagents | 0 | 5 defined | P1 |
| Skills defined | 0 | 5+ | P1 |
| MCP servers | 0 | 3+ configured | P2 |
| Test coverage | Unknown | >80% | P2 |

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

## Part 8: Detailed Implementation Guides

### 8.1 Sentiment Analysis Consolidation - Step by Step

#### Prerequisites
```bash
# Ensure all tests pass before starting
pytest tests/ -v -m "not slow" --tb=short
git checkout -b feature/consolidation-sentiment
git tag pre-sentiment-consolidation
```

#### Step 1: Create Package Structure
```bash
mkdir -p llm/sentiment/providers
touch llm/sentiment/__init__.py
touch llm/sentiment/core.py
touch llm/sentiment/filters.py
touch llm/sentiment/providers/__init__.py
touch llm/sentiment/providers/finbert.py
touch llm/sentiment/providers/reddit.py
touch llm/sentiment/providers/news.py
touch llm/sentiment/providers/emotion.py
touch llm/sentiment/agent.py
```

#### Step 2: Create Core Data Types (llm/sentiment/core.py)
```python
"""
Unified sentiment analysis core module.

This module provides the central interface for all sentiment analysis,
consolidating previously separate systems into a single API.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, Any

class SentimentDirection(Enum):
    """Sentiment direction classification."""
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2

class SentimentSource(Enum):
    """Source of sentiment data."""
    FINBERT = "finbert"
    REDDIT = "reddit"
    NEWS = "news"
    TWITTER = "twitter"
    EMOTION = "emotion"
    AGGREGATE = "aggregate"

@dataclass
class SentimentResult:
    """
    Unified sentiment analysis result.

    All sentiment providers must return this type, enabling
    consistent downstream processing regardless of source.
    """
    symbol: str
    direction: SentimentDirection
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    source: SentimentSource
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Whether sentiment is strong enough to act on."""
        return abs(self.score) > 0.3 and self.confidence > 0.6

    @property
    def signal_strength(self) -> float:
        """Combined score and confidence."""
        return self.score * self.confidence

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "direction": self.direction.name,
            "score": self.score,
            "confidence": self.confidence,
            "source": self.source.value,
            "timestamp": self.timestamp.isoformat(),
            "is_actionable": self.is_actionable,
            "signal_strength": self.signal_strength,
            "metadata": self.metadata,
        }

class SentimentProvider(Protocol):
    """Protocol for sentiment analysis providers."""

    def analyze(self, text: str, symbol: str | None = None) -> SentimentResult:
        """Analyze text and return sentiment result."""
        ...

    @property
    def source(self) -> SentimentSource:
        """The source type for this provider."""
        ...

class SentimentAggregator:
    """
    Aggregates sentiment from multiple providers.

    Combines FinBERT, Reddit, news, and emotion sentiment
    into a single weighted score.
    """

    def __init__(self, weights: dict[SentimentSource, float] | None = None):
        self.weights = weights or {
            SentimentSource.FINBERT: 0.4,
            SentimentSource.NEWS: 0.3,
            SentimentSource.REDDIT: 0.2,
            SentimentSource.EMOTION: 0.1,
        }
        self._providers: dict[SentimentSource, SentimentProvider] = {}

    def register_provider(self, provider: SentimentProvider) -> None:
        """Register a sentiment provider."""
        self._providers[provider.source] = provider

    def aggregate(self, results: list[SentimentResult]) -> SentimentResult:
        """Aggregate multiple sentiment results into one."""
        if not results:
            raise ValueError("Cannot aggregate empty results")

        total_weight = 0.0
        weighted_score = 0.0
        weighted_confidence = 0.0

        for result in results:
            weight = self.weights.get(result.source, 0.1)
            weighted_score += result.score * weight
            weighted_confidence += result.confidence * weight
            total_weight += weight

        if total_weight > 0:
            final_score = weighted_score / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_score = 0.0
            final_confidence = 0.0

        # Determine direction from score
        if final_score <= -0.6:
            direction = SentimentDirection.VERY_BEARISH
        elif final_score <= -0.2:
            direction = SentimentDirection.BEARISH
        elif final_score >= 0.6:
            direction = SentimentDirection.VERY_BULLISH
        elif final_score >= 0.2:
            direction = SentimentDirection.BULLISH
        else:
            direction = SentimentDirection.NEUTRAL

        return SentimentResult(
            symbol=results[0].symbol,
            direction=direction,
            score=final_score,
            confidence=final_confidence,
            source=SentimentSource.AGGREGATE,
            metadata={"sources": [r.source.value for r in results]},
        )
```

#### Step 3: Migrate FinBERT Provider (llm/sentiment/providers/finbert.py)
```python
"""
FinBERT sentiment analysis provider.

Migrated from: llm/sentiment.py
"""
from typing import Any
from datetime import datetime

# Import from existing implementation
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from ..core import (
    SentimentResult,
    SentimentDirection,
    SentimentSource,
    SentimentProvider,
)

class FinBERTProvider:
    """
    FinBERT-based financial sentiment analysis.

    Uses the ProsusAI/finbert model for finance-specific
    sentiment classification.
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: str | None = None):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers and torch required. "
                "Install with: pip install transformers torch"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None

    @property
    def source(self) -> SentimentSource:
        return SentimentSource.FINBERT

    def _load_model(self) -> None:
        """Lazy load model on first use."""
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_NAME
            ).to(self.device)
            self._model.eval()

    def analyze(self, text: str, symbol: str | None = None) -> SentimentResult:
        """
        Analyze text using FinBERT.

        Args:
            text: Text to analyze
            symbol: Optional stock symbol for context

        Returns:
            SentimentResult with FinBERT classification
        """
        self._load_model()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # FinBERT outputs: [negative, neutral, positive]
        neg, neu, pos = probs[0].cpu().numpy()

        # Calculate composite score (-1 to 1)
        score = float(pos - neg)
        confidence = float(max(neg, neu, pos))

        # Determine direction
        if score <= -0.3:
            direction = SentimentDirection.BEARISH
        elif score >= 0.3:
            direction = SentimentDirection.BULLISH
        else:
            direction = SentimentDirection.NEUTRAL

        return SentimentResult(
            symbol=symbol or "UNKNOWN",
            direction=direction,
            score=score,
            confidence=confidence,
            source=SentimentSource.FINBERT,
            metadata={
                "probabilities": {
                    "negative": float(neg),
                    "neutral": float(neu),
                    "positive": float(pos),
                },
                "text_length": len(text),
            },
        )
```

#### Step 4: Migrate Reddit Provider (llm/sentiment/providers/reddit.py)
```python
"""
Reddit sentiment analysis provider.

Migrated from: llm/reddit_sentiment.py
"""
from datetime import datetime, timedelta
from typing import Any
import re

from ..core import (
    SentimentResult,
    SentimentDirection,
    SentimentSource,
)

class RedditProvider:
    """
    Reddit-based sentiment analysis.

    Analyzes posts and comments from financial subreddits
    like r/wallstreetbets, r/stocks, r/options.
    """

    SUBREDDITS = ["wallstreetbets", "stocks", "options", "investing"]

    # Keywords that boost sentiment signal
    BULLISH_KEYWORDS = [
        "buy", "calls", "moon", "rocket", "diamond hands",
        "bullish", "long", "squeeze", "yolo", "tendies",
    ]
    BEARISH_KEYWORDS = [
        "sell", "puts", "crash", "dump", "paper hands",
        "bearish", "short", "drill", "rip", "loss",
    ]

    def __init__(self, finbert_provider=None):
        """
        Initialize Reddit provider.

        Args:
            finbert_provider: Optional FinBERT provider for text analysis
        """
        self._finbert = finbert_provider

    @property
    def source(self) -> SentimentSource:
        return SentimentSource.REDDIT

    def analyze(self, text: str, symbol: str | None = None) -> SentimentResult:
        """
        Analyze Reddit-style text.

        Uses keyword matching plus optional FinBERT for
        nuanced understanding of WSB-style language.
        """
        text_lower = text.lower()

        # Count keyword occurrences
        bullish_count = sum(
            1 for kw in self.BULLISH_KEYWORDS if kw in text_lower
        )
        bearish_count = sum(
            1 for kw in self.BEARISH_KEYWORDS if kw in text_lower
        )

        # Calculate keyword-based score
        total_keywords = bullish_count + bearish_count
        if total_keywords > 0:
            keyword_score = (bullish_count - bearish_count) / total_keywords
            keyword_confidence = min(total_keywords / 5, 1.0)  # Cap at 5 keywords
        else:
            keyword_score = 0.0
            keyword_confidence = 0.2

        # Optionally blend with FinBERT
        if self._finbert:
            finbert_result = self._finbert.analyze(text, symbol)
            # Blend: 60% keywords, 40% FinBERT (Reddit slang needs keyword help)
            final_score = keyword_score * 0.6 + finbert_result.score * 0.4
            final_confidence = (keyword_confidence + finbert_result.confidence) / 2
        else:
            final_score = keyword_score
            final_confidence = keyword_confidence

        # Determine direction
        if final_score <= -0.3:
            direction = SentimentDirection.BEARISH
        elif final_score >= 0.3:
            direction = SentimentDirection.BULLISH
        else:
            direction = SentimentDirection.NEUTRAL

        return SentimentResult(
            symbol=symbol or "UNKNOWN",
            direction=direction,
            score=final_score,
            confidence=final_confidence,
            source=SentimentSource.REDDIT,
            metadata={
                "bullish_keywords": bullish_count,
                "bearish_keywords": bearish_count,
                "keyword_score": keyword_score,
            },
        )
```

#### Step 5: Create Package Init (llm/sentiment/__init__.py)
```python
"""
Consolidated sentiment analysis package.

This package provides unified sentiment analysis, combining:
- FinBERT financial sentiment
- Reddit social sentiment
- News article sentiment
- Emotion detection

Usage:
    from llm.sentiment import analyze_sentiment, SentimentResult

    result = analyze_sentiment("AAPL is showing strong growth", symbol="AAPL")
    print(result.direction, result.confidence)

Migration Notes:
    Previous imports should be updated:
    - from llm.sentiment import ... → from llm.sentiment.providers.finbert import ...
    - from llm.reddit_sentiment import ... → from llm.sentiment.providers.reddit import ...
"""

from .core import (
    SentimentResult,
    SentimentDirection,
    SentimentSource,
    SentimentProvider,
    SentimentAggregator,
)
from .providers.finbert import FinBERTProvider
from .providers.reddit import RedditProvider

__all__ = [
    # Core types
    "SentimentResult",
    "SentimentDirection",
    "SentimentSource",
    "SentimentProvider",
    "SentimentAggregator",
    # Providers
    "FinBERTProvider",
    "RedditProvider",
    # Convenience functions
    "analyze_sentiment",
    "get_aggregator",
]

# Module-level aggregator instance
_aggregator: SentimentAggregator | None = None

def get_aggregator() -> SentimentAggregator:
    """Get or create the global sentiment aggregator."""
    global _aggregator
    if _aggregator is None:
        _aggregator = SentimentAggregator()
        # Register available providers
        try:
            _aggregator.register_provider(FinBERTProvider())
        except ImportError:
            pass  # FinBERT not available
        _aggregator.register_provider(RedditProvider())
    return _aggregator

def analyze_sentiment(
    text: str,
    symbol: str | None = None,
    sources: list[SentimentSource] | None = None,
) -> SentimentResult:
    """
    Analyze sentiment using configured providers.

    Args:
        text: Text to analyze
        symbol: Optional stock symbol
        sources: Optional list of sources to use (default: all available)

    Returns:
        Aggregated SentimentResult
    """
    aggregator = get_aggregator()
    results = []

    for source, provider in aggregator._providers.items():
        if sources is None or source in sources:
            results.append(provider.analyze(text, symbol))

    if len(results) == 1:
        return results[0]
    return aggregator.aggregate(results)
```

#### Step 6: Add Deprecation Warnings to Old Modules
```python
# Add to top of llm/sentiment.py (old file)
import warnings
warnings.warn(
    "llm.sentiment is deprecated. Use llm.sentiment.providers.finbert instead. "
    "This module will be removed in v3.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Add re-export for backwards compatibility
from llm.sentiment.providers.finbert import FinBERTProvider
from llm.sentiment.core import SentimentResult, SentimentDirection

# Keep old class name working
SentimentAnalyzer = FinBERTProvider
```

#### Step 7: Update Tests
```python
# tests/test_sentiment_consolidation.py
"""Tests for consolidated sentiment package."""
import pytest
from llm.sentiment import (
    analyze_sentiment,
    SentimentResult,
    SentimentDirection,
    SentimentSource,
    FinBERTProvider,
    RedditProvider,
    SentimentAggregator,
)

class TestSentimentResult:
    """Test SentimentResult dataclass."""

    def test_is_actionable_strong_signal(self):
        result = SentimentResult(
            symbol="AAPL",
            direction=SentimentDirection.BULLISH,
            score=0.5,
            confidence=0.8,
            source=SentimentSource.FINBERT,
        )
        assert result.is_actionable is True

    def test_is_actionable_weak_signal(self):
        result = SentimentResult(
            symbol="AAPL",
            direction=SentimentDirection.NEUTRAL,
            score=0.1,
            confidence=0.4,
            source=SentimentSource.FINBERT,
        )
        assert result.is_actionable is False

    def test_signal_strength(self):
        result = SentimentResult(
            symbol="AAPL",
            direction=SentimentDirection.BULLISH,
            score=0.6,
            confidence=0.8,
            source=SentimentSource.FINBERT,
        )
        assert result.signal_strength == pytest.approx(0.48)

class TestRedditProvider:
    """Test Reddit sentiment provider."""

    def test_bullish_keywords(self):
        provider = RedditProvider()
        result = provider.analyze("AAPL to the moon! 🚀 Diamond hands!", symbol="AAPL")
        assert result.direction in [SentimentDirection.BULLISH, SentimentDirection.VERY_BULLISH]
        assert result.score > 0

    def test_bearish_keywords(self):
        provider = RedditProvider()
        result = provider.analyze("Selling everything, this is going to crash", symbol="SPY")
        assert result.direction in [SentimentDirection.BEARISH, SentimentDirection.VERY_BEARISH]
        assert result.score < 0

class TestSentimentAggregator:
    """Test sentiment aggregation."""

    def test_aggregate_multiple_sources(self):
        aggregator = SentimentAggregator()
        results = [
            SentimentResult(
                symbol="AAPL",
                direction=SentimentDirection.BULLISH,
                score=0.6,
                confidence=0.8,
                source=SentimentSource.FINBERT,
            ),
            SentimentResult(
                symbol="AAPL",
                direction=SentimentDirection.NEUTRAL,
                score=0.1,
                confidence=0.6,
                source=SentimentSource.REDDIT,
            ),
        ]
        aggregated = aggregator.aggregate(results)
        assert aggregated.source == SentimentSource.AGGREGATE
        assert "finbert" in aggregated.metadata["sources"]
        assert "reddit" in aggregated.metadata["sources"]
```

#### Step 8: Verify and Commit
```bash
# Run new tests
pytest tests/test_sentiment_consolidation.py -v

# Run full test suite
pytest tests/ -v -m "not slow" --tb=short

# If tests pass, commit
git add llm/sentiment/
git add tests/test_sentiment_consolidation.py
git commit -m "feat(sentiment): consolidate sentiment analysis into unified package

- Create llm/sentiment/ package with unified API
- Migrate FinBERT provider from llm/sentiment.py
- Migrate Reddit provider from llm/reddit_sentiment.py
- Add SentimentAggregator for multi-source analysis
- Add deprecation warnings to old modules

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### 8.2 News Analysis Consolidation - Step by Step

#### Prerequisites
```bash
git checkout -b feature/consolidation-news
git tag pre-news-consolidation
```

#### Step 1: Create Package Structure
```bash
mkdir -p llm/news
touch llm/news/__init__.py
touch llm/news/analyzer.py
touch llm/news/processor.py
touch llm/news/alerts.py
touch llm/news/agent.py
```

#### Step 2: Create Core Types (llm/news/analyzer.py)
```python
"""
Unified news analysis module.

Migrated and consolidated from:
- llm/news_analyzer.py
- llm/news_processor.py
- scanners/movement_scanner.py (news portions)
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

class NewsImpact(Enum):
    """Estimated market impact of news."""
    NEGLIGIBLE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class NewsCategory(Enum):
    """News category classification."""
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    PRODUCT_LAUNCH = "product_launch"
    EXECUTIVE_CHANGE = "executive_change"
    REGULATORY = "regulatory"
    MACRO_ECONOMIC = "macro_economic"
    SECTOR_NEWS = "sector_news"
    ANALYST_RATING = "analyst_rating"
    INSIDER_TRADING = "insider_trading"
    OTHER = "other"

@dataclass
class NewsSignal:
    """
    Unified news analysis signal.

    All news analysis components return this type,
    enabling consistent downstream processing.
    """
    symbol: str
    headline: str
    category: NewsCategory
    impact: NewsImpact
    sentiment_score: float  # -1.0 to 1.0
    relevance_score: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    url: str = ""
    summary: str = ""
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Whether news is significant enough to act on."""
        return (
            self.impact.value >= NewsImpact.MEDIUM.value
            and self.relevance_score > 0.5
        )

    @property
    def urgency_score(self) -> float:
        """Combined urgency based on impact and recency."""
        age_hours = (datetime.utcnow() - self.timestamp).total_seconds() / 3600
        recency_factor = max(0, 1 - (age_hours / 24))  # Decay over 24 hours
        return (self.impact.value / 4) * recency_factor

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "headline": self.headline,
            "category": self.category.value,
            "impact": self.impact.name,
            "sentiment_score": self.sentiment_score,
            "relevance_score": self.relevance_score,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "url": self.url,
            "summary": self.summary,
            "entities": self.entities,
            "is_actionable": self.is_actionable,
            "urgency_score": self.urgency_score,
        }

class NewsAnalyzer:
    """
    Central news analysis engine.

    Consolidates all news analysis functionality into
    a single, consistent interface.
    """

    # Category detection keywords
    CATEGORY_KEYWORDS = {
        NewsCategory.EARNINGS: [
            "earnings", "revenue", "eps", "quarterly results",
            "profit", "loss", "guidance", "forecast",
        ],
        NewsCategory.MERGER_ACQUISITION: [
            "merger", "acquisition", "acquire", "buyout",
            "takeover", "deal", "bid",
        ],
        NewsCategory.REGULATORY: [
            "sec", "fda", "regulation", "compliance",
            "lawsuit", "investigation", "fine", "penalty",
        ],
        NewsCategory.ANALYST_RATING: [
            "upgrade", "downgrade", "price target",
            "rating", "analyst", "outperform", "underperform",
        ],
    }

    def __init__(self, sentiment_provider=None):
        """
        Initialize news analyzer.

        Args:
            sentiment_provider: Optional sentiment provider for text analysis
        """
        self._sentiment = sentiment_provider

    def analyze(self, headline: str, body: str = "", symbol: str = "") -> NewsSignal:
        """
        Analyze a news article.

        Args:
            headline: News headline
            body: Optional article body
            symbol: Stock symbol the news relates to

        Returns:
            NewsSignal with analysis results
        """
        full_text = f"{headline} {body}".lower()

        # Detect category
        category = self._detect_category(full_text)

        # Estimate impact
        impact = self._estimate_impact(category, full_text)

        # Get sentiment
        if self._sentiment:
            sentiment_result = self._sentiment.analyze(headline, symbol)
            sentiment_score = sentiment_result.score
        else:
            sentiment_score = self._simple_sentiment(full_text)

        # Calculate relevance
        relevance_score = self._calculate_relevance(symbol, full_text)

        return NewsSignal(
            symbol=symbol,
            headline=headline,
            category=category,
            impact=impact,
            sentiment_score=sentiment_score,
            relevance_score=relevance_score,
            summary=body[:500] if body else "",
        )

    def _detect_category(self, text: str) -> NewsCategory:
        """Detect news category from text."""
        max_matches = 0
        best_category = NewsCategory.OTHER

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text)
            if matches > max_matches:
                max_matches = matches
                best_category = category

        return best_category

    def _estimate_impact(self, category: NewsCategory, text: str) -> NewsImpact:
        """Estimate market impact based on category and content."""
        # High-impact categories
        if category in [
            NewsCategory.EARNINGS,
            NewsCategory.MERGER_ACQUISITION,
            NewsCategory.REGULATORY,
        ]:
            # Check for strong language
            if any(word in text for word in ["beat", "miss", "surprise", "shock"]):
                return NewsImpact.HIGH
            return NewsImpact.MEDIUM

        if category == NewsCategory.ANALYST_RATING:
            return NewsImpact.MEDIUM

        return NewsImpact.LOW

    def _simple_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment when no provider available."""
        positive = ["beat", "exceed", "upgrade", "growth", "profit", "gain"]
        negative = ["miss", "below", "downgrade", "loss", "decline", "drop"]

        pos_count = sum(1 for w in positive if w in text)
        neg_count = sum(1 for w in negative if w in text)

        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total

    def _calculate_relevance(self, symbol: str, text: str) -> float:
        """Calculate how relevant news is to the symbol."""
        if not symbol:
            return 0.5

        # Direct mention is highly relevant
        if symbol.lower() in text:
            return 0.9

        # Check for company name variations
        # This would be enhanced with a company name database
        return 0.5
```

#### Step 3: Create Package Init (llm/news/__init__.py)
```python
"""
Consolidated news analysis package.

This package provides unified news analysis, combining:
- News article classification
- Impact estimation
- Sentiment extraction
- Alert generation

Usage:
    from llm.news import analyze_news, NewsSignal

    signal = analyze_news("AAPL beats earnings expectations", symbol="AAPL")
    print(signal.impact, signal.category)
"""

from .analyzer import (
    NewsSignal,
    NewsImpact,
    NewsCategory,
    NewsAnalyzer,
)

__all__ = [
    "NewsSignal",
    "NewsImpact",
    "NewsCategory",
    "NewsAnalyzer",
    "analyze_news",
]

# Module-level analyzer instance
_analyzer: NewsAnalyzer | None = None

def get_analyzer() -> NewsAnalyzer:
    """Get or create the global news analyzer."""
    global _analyzer
    if _analyzer is None:
        # Try to use sentiment provider if available
        try:
            from llm.sentiment import FinBERTProvider
            _analyzer = NewsAnalyzer(sentiment_provider=FinBERTProvider())
        except ImportError:
            _analyzer = NewsAnalyzer()
    return _analyzer

def analyze_news(
    headline: str,
    body: str = "",
    symbol: str = "",
) -> NewsSignal:
    """
    Analyze a news article.

    Args:
        headline: News headline
        body: Optional article body text
        symbol: Stock symbol the news relates to

    Returns:
        NewsSignal with analysis results
    """
    return get_analyzer().analyze(headline, body, symbol)
```

---

### 8.3 Anomaly Detection Unification - Step by Step

#### Create Unified Anomaly Types (models/anomaly/__init__.py)
```python
"""
Unified anomaly detection package.

Consolidates all anomaly detection into a consistent interface:
- Market regime anomalies
- Agent behavior anomalies
- Spread anomalies
- Unusual activity detection

Usage:
    from models.anomaly import detect_anomalies, AnomalyEvent, AnomalyType

    events = detect_anomalies(market_data)
    for event in events:
        if event.severity >= Severity.HIGH:
            handle_critical_anomaly(event)
"""

from .base import (
    AnomalyEvent,
    AnomalyType,
    Severity,
    AnomalyDetector,
)

__all__ = [
    "AnomalyEvent",
    "AnomalyType",
    "Severity",
    "AnomalyDetector",
]
```

#### Create Base Types (models/anomaly/base.py)
```python
"""
Base anomaly detection types and interfaces.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, Any

class AnomalyType(Enum):
    """Type of anomaly detected."""
    MARKET_REGIME = "market_regime"
    VOLATILITY_SPIKE = "volatility_spike"
    VOLUME_ANOMALY = "volume_anomaly"
    PRICE_ANOMALY = "price_anomaly"
    SPREAD_ANOMALY = "spread_anomaly"
    AGENT_BEHAVIOR = "agent_behavior"
    CORRELATION_BREAK = "correlation_break"
    LIQUIDITY_CRISIS = "liquidity_crisis"

class Severity(Enum):
    """Anomaly severity level."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AnomalyEvent:
    """
    Unified anomaly event representation.

    All anomaly detectors return this type, enabling
    consistent handling regardless of anomaly source.
    """
    type: AnomalyType
    severity: Severity
    source: str  # Detector that identified the anomaly
    symbol: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: dict[str, Any] = field(default_factory=dict)
    recommended_action: str | None = None
    expires_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        """Whether the anomaly is still active."""
        if self.expires_at is None:
            return True
        return datetime.utcnow() < self.expires_at

    @property
    def requires_immediate_action(self) -> bool:
        """Whether this anomaly requires immediate intervention."""
        return self.severity in [Severity.HIGH, Severity.CRITICAL]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "type": self.type.value,
            "severity": self.severity.name,
            "source": self.source,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "recommended_action": self.recommended_action,
            "is_active": self.is_active,
            "requires_immediate_action": self.requires_immediate_action,
        }

class AnomalyDetector(Protocol):
    """Protocol for anomaly detectors."""

    @property
    def detector_name(self) -> str:
        """Name of this detector."""
        ...

    def detect(self, data: Any) -> list[AnomalyEvent]:
        """
        Detect anomalies in the provided data.

        Args:
            data: Input data to analyze (type varies by detector)

        Returns:
            List of detected anomaly events
        """
        ...
```

---

### 8.4 Risk Chain Implementation - Step by Step

#### Create Risk Enforcement Chain (models/risk_chain.py)
```python
"""
Unified risk enforcement chain.

Executes all risk checks in the correct order:
1. Circuit Breaker (hard stop)
2. Portfolio Risk (position limits)
3. Pre-Trade Validation (order checks)
4. Agent Risk Review (LLM contextual)

Usage:
    from models.risk_chain import RiskEnforcementChain, RiskResult

    chain = RiskEnforcementChain()
    result = chain.validate_trade(order)

    if not result.approved:
        print(f"Trade blocked: {result.rejection_reason}")
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol
import logging

logger = logging.getLogger(__name__)

class RiskCheckStage(Enum):
    """Stages in the risk check pipeline."""
    CIRCUIT_BREAKER = 1
    PORTFOLIO_RISK = 2
    PRE_TRADE = 3
    AGENT_REVIEW = 4

@dataclass
class RiskResult:
    """Result of risk chain validation."""
    approved: bool
    stage_reached: RiskCheckStage
    rejection_reason: str | None = None
    rejecting_stage: RiskCheckStage | None = None
    warnings: list[str] = field(default_factory=list)
    checks_performed: dict[str, bool] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "approved": self.approved,
            "stage_reached": self.stage_reached.name,
            "rejection_reason": self.rejection_reason,
            "rejecting_stage": self.rejecting_stage.name if self.rejecting_stage else None,
            "warnings": self.warnings,
            "checks_performed": self.checks_performed,
            "timestamp": self.timestamp.isoformat(),
        }

class RiskCheck(Protocol):
    """Protocol for individual risk checks."""

    @property
    def stage(self) -> RiskCheckStage:
        """Stage in the pipeline."""
        ...

    def check(self, order: Any) -> tuple[bool, str | None]:
        """
        Perform the risk check.

        Returns:
            Tuple of (passed, rejection_reason)
        """
        ...

class RiskEnforcementChain:
    """
    Unified risk enforcement chain.

    Executes all risk checks in sequence, stopping
    at the first rejection.
    """

    def __init__(self):
        self._circuit_breaker = None
        self._portfolio_risk = None
        self._pre_trade = None
        self._agent_risk = None
        self._initialized = False

    def _lazy_init(self) -> None:
        """Lazy initialization of risk components."""
        if self._initialized:
            return

        try:
            from models.circuit_breaker import get_circuit_breaker
            self._circuit_breaker = get_circuit_breaker()
        except ImportError:
            logger.warning("Circuit breaker not available")

        try:
            from models.risk_manager import RiskManager
            self._portfolio_risk = RiskManager()
        except ImportError:
            logger.warning("Portfolio risk manager not available")

        try:
            from execution.pre_trade_validator import PreTradeValidator
            self._pre_trade = PreTradeValidator()
        except ImportError:
            logger.warning("Pre-trade validator not available")

        try:
            from llm.agents.risk_managers import PositionRiskManager
            self._agent_risk = PositionRiskManager()
        except ImportError:
            logger.warning("Agent risk manager not available")

        self._initialized = True

    def validate_trade(self, order: Any) -> RiskResult:
        """
        Validate a trade through all risk stages.

        Args:
            order: Order to validate

        Returns:
            RiskResult with approval status and details
        """
        self._lazy_init()

        checks_performed = {}
        warnings = []

        # Stage 1: Circuit Breaker
        if self._circuit_breaker:
            can_trade = self._circuit_breaker.can_trade()
            checks_performed["circuit_breaker"] = can_trade

            if not can_trade:
                return RiskResult(
                    approved=False,
                    stage_reached=RiskCheckStage.CIRCUIT_BREAKER,
                    rejection_reason="Circuit breaker is OPEN - all trading halted",
                    rejecting_stage=RiskCheckStage.CIRCUIT_BREAKER,
                    checks_performed=checks_performed,
                )
        else:
            warnings.append("Circuit breaker check skipped - not available")

        # Stage 2: Portfolio Risk
        if self._portfolio_risk:
            try:
                risk_ok = self._portfolio_risk.check_order(order)
                checks_performed["portfolio_risk"] = risk_ok

                if not risk_ok:
                    return RiskResult(
                        approved=False,
                        stage_reached=RiskCheckStage.PORTFOLIO_RISK,
                        rejection_reason="Order exceeds portfolio risk limits",
                        rejecting_stage=RiskCheckStage.PORTFOLIO_RISK,
                        checks_performed=checks_performed,
                        warnings=warnings,
                    )
            except Exception as e:
                logger.error(f"Portfolio risk check error: {e}")
                warnings.append(f"Portfolio risk check error: {e}")
        else:
            warnings.append("Portfolio risk check skipped - not available")

        # Stage 3: Pre-Trade Validation
        if self._pre_trade:
            try:
                validation_result = self._pre_trade.validate(order)
                checks_performed["pre_trade"] = validation_result.is_valid

                if not validation_result.is_valid:
                    return RiskResult(
                        approved=False,
                        stage_reached=RiskCheckStage.PRE_TRADE,
                        rejection_reason=validation_result.reason,
                        rejecting_stage=RiskCheckStage.PRE_TRADE,
                        checks_performed=checks_performed,
                        warnings=warnings,
                    )
            except Exception as e:
                logger.error(f"Pre-trade validation error: {e}")
                warnings.append(f"Pre-trade validation error: {e}")
        else:
            warnings.append("Pre-trade validation skipped - not available")

        # Stage 4: Agent Risk Review (optional, for large orders)
        if self._agent_risk and self._should_agent_review(order):
            try:
                agent_approval = self._agent_risk.approve(order)
                checks_performed["agent_review"] = agent_approval

                if not agent_approval:
                    return RiskResult(
                        approved=False,
                        stage_reached=RiskCheckStage.AGENT_REVIEW,
                        rejection_reason="Agent risk review rejected the order",
                        rejecting_stage=RiskCheckStage.AGENT_REVIEW,
                        checks_performed=checks_performed,
                        warnings=warnings,
                    )
            except Exception as e:
                logger.error(f"Agent risk review error: {e}")
                warnings.append(f"Agent risk review error: {e}")

        # All checks passed
        return RiskResult(
            approved=True,
            stage_reached=RiskCheckStage.AGENT_REVIEW,
            checks_performed=checks_performed,
            warnings=warnings,
        )

    def _should_agent_review(self, order: Any) -> bool:
        """Determine if order needs agent review."""
        # Large orders or orders during volatile periods need agent review
        try:
            order_value = getattr(order, 'value', 0) or getattr(order, 'quantity', 0) * getattr(order, 'price', 0)
            return order_value > 10000
        except:
            return False
```

---

## Part 9: Testing Procedures

### 9.1 Pre-Migration Testing Checklist

Before starting any consolidation phase, verify:

```bash
# 1. All existing tests pass
pytest tests/ -v --tb=short

# 2. No lint errors in affected files
ruff check llm/ models/ execution/

# 3. Type checking passes
mypy llm/ models/ execution/ --ignore-missing-imports

# 4. Git is clean
git status  # Should show nothing to commit

# 5. Create safety tag
git tag pre-migration-$(date +%Y%m%d)
```

### 9.2 Post-Migration Testing Checklist

After each consolidation step:

```bash
# 1. New package tests pass
pytest tests/test_<package>_consolidation.py -v

# 2. Integration tests pass
pytest tests/integration/ -v --tb=short

# 3. Deprecation warnings work
python -c "import llm.sentiment" 2>&1 | grep -i deprecation

# 4. Old imports still work (backwards compatibility)
python -c "from llm.sentiment import SentimentAnalyzer"

# 5. Performance regression check
python scripts/benchmark_sentiment.py  # Should not be >20% slower
```

### 9.3 Test Templates for Consolidated Packages

```python
# tests/conftest.py additions
import pytest

@pytest.fixture
def sentiment_provider():
    """Provide test sentiment provider."""
    from llm.sentiment import FinBERTProvider
    return FinBERTProvider()

@pytest.fixture
def news_analyzer():
    """Provide test news analyzer."""
    from llm.news import NewsAnalyzer
    return NewsAnalyzer()

@pytest.fixture
def risk_chain():
    """Provide test risk chain."""
    from models.risk_chain import RiskEnforcementChain
    return RiskEnforcementChain()

@pytest.fixture
def mock_order():
    """Provide mock order for testing."""
    from unittest.mock import MagicMock
    order = MagicMock()
    order.symbol = "SPY"
    order.quantity = 100
    order.price = 450.0
    order.value = 45000.0
    return order
```

---

## Part 10: Rollback Procedures

### 10.1 Quick Rollback (Git Reset)

If issues are found immediately after migration:

```bash
# Rollback to pre-migration tag
git reset --hard pre-migration-YYYYMMDD

# Or rollback specific commit
git revert <commit-hash>

# Force push if already pushed (coordinate with team)
git push origin main --force
```

### 10.2 Selective Rollback (Keep Some Changes)

If only part of the migration needs rollback:

```bash
# Restore specific files from tag
git checkout pre-migration-YYYYMMDD -- llm/sentiment/
git checkout pre-migration-YYYYMMDD -- llm/sentiment.py

# Commit the restoration
git add -A
git commit -m "rollback: restore sentiment module to pre-migration state"
```

### 10.3 Backwards Compatibility Layer

If rollback isn't feasible, add compatibility shim:

```python
# llm/sentiment_compat.py
"""Backwards compatibility layer for old sentiment imports."""
import warnings

warnings.warn(
    "Direct imports from llm.sentiment_compat are deprecated. "
    "Update to llm.sentiment package.",
    DeprecationWarning,
)

# Re-export everything from new location
from llm.sentiment import *
from llm.sentiment.providers.finbert import FinBERTProvider as SentimentAnalyzer
```

---

## Part 11: Validation Scripts

### 11.1 Import Migration Validator

```python
#!/usr/bin/env python
# scripts/validate_imports.py
"""Validate that all imports have been migrated correctly."""

import ast
import sys
from pathlib import Path

DEPRECATED_IMPORTS = {
    "llm.sentiment": "llm.sentiment.providers.finbert",
    "llm.reddit_sentiment": "llm.sentiment.providers.reddit",
    "llm.news_analyzer": "llm.news.analyzer",
    "llm.news_processor": "llm.news.processor",
    "evaluation.metrics": "observability.metrics.collectors.agent",
}

def check_file(filepath: Path) -> list[str]:
    """Check a file for deprecated imports."""
    issues = []

    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())
    except SyntaxError:
        return [f"{filepath}: Syntax error, cannot parse"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in DEPRECATED_IMPORTS:
                    issues.append(
                        f"{filepath}:{node.lineno}: "
                        f"Deprecated import '{alias.name}' → "
                        f"Use '{DEPRECATED_IMPORTS[alias.name]}'"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module in DEPRECATED_IMPORTS:
                issues.append(
                    f"{filepath}:{node.lineno}: "
                    f"Deprecated import from '{node.module}' → "
                    f"Use '{DEPRECATED_IMPORTS[node.module]}'"
                )

    return issues

def main():
    """Run import validation on codebase."""
    root = Path(".")
    all_issues = []

    for py_file in root.rglob("*.py"):
        if "venv" in str(py_file) or "__pycache__" in str(py_file):
            continue
        issues = check_file(py_file)
        all_issues.extend(issues)

    if all_issues:
        print("Deprecated imports found:")
        for issue in all_issues:
            print(f"  {issue}")
        sys.exit(1)
    else:
        print("No deprecated imports found.")
        sys.exit(0)

if __name__ == "__main__":
    main()
```

### 11.2 Consolidation Progress Tracker

```python
#!/usr/bin/env python
# scripts/consolidation_progress.py
"""Track consolidation progress."""

from pathlib import Path

CONSOLIDATION_STATUS = {
    "sentiment": {
        "old_files": [
            "llm/sentiment.py",
            "llm/sentiment_filter.py",
            "llm/reddit_sentiment.py",
            "llm/emotion_detector.py",
        ],
        "new_package": "llm/sentiment/",
        "tests": "tests/test_sentiment_consolidation.py",
    },
    "news": {
        "old_files": [
            "llm/news_analyzer.py",
            "llm/news_processor.py",
            "llm/news_alert_manager.py",
        ],
        "new_package": "llm/news/",
        "tests": "tests/test_news_consolidation.py",
    },
    "anomaly": {
        "old_files": [
            "models/anomaly_detector.py",
            "observability/anomaly_detector.py",
            "execution/spread_anomaly.py",
        ],
        "new_package": "models/anomaly/",
        "tests": "tests/test_anomaly_consolidation.py",
    },
    "risk_chain": {
        "old_files": [],  # Integration, not replacement
        "new_package": "models/risk_chain.py",
        "tests": "tests/test_risk_chain.py",
    },
}

def check_status():
    """Check consolidation status."""
    print("=" * 60)
    print("CONSOLIDATION PROGRESS")
    print("=" * 60)

    for name, config in CONSOLIDATION_STATUS.items():
        print(f"\n{name.upper()}")
        print("-" * 40)

        # Check new package exists
        new_path = Path(config["new_package"])
        new_exists = new_path.exists()
        print(f"  New package: {'✓' if new_exists else '✗'} {config['new_package']}")

        # Check old files removed
        old_remaining = [f for f in config["old_files"] if Path(f).exists()]
        if old_remaining:
            print(f"  Old files remaining: {len(old_remaining)}")
            for f in old_remaining:
                print(f"    - {f}")
        else:
            print(f"  Old files: ✓ All removed")

        # Check tests exist
        test_path = Path(config["tests"])
        tests_exist = test_path.exists()
        print(f"  Tests: {'✓' if tests_exist else '✗'} {config['tests']}")

        # Status
        if new_exists and not old_remaining and tests_exist:
            print(f"  Status: ✓ COMPLETE")
        elif new_exists:
            print(f"  Status: ⚠ IN PROGRESS")
        else:
            print(f"  Status: ✗ NOT STARTED")

if __name__ == "__main__":
    check_status()
```

---

## Part 12: Claude SDK Integration Guide

### 12.1 Creating Subagent Definitions

Create `.claude/agents/` directory and add agent markdown files:

```bash
mkdir -p .claude/agents
```

#### Market Analyst Agent (.claude/agents/market-analyst.md)
```markdown
# Market Analyst Agent

## Role
Technical analysis specialist that identifies trading opportunities using chart patterns, indicators, and support/resistance levels.

## Backend Integration
- Primary: `llm/agents/technical_analyst.py`
- Indicators: `indicators/technical_alpha.py`
- Patterns: `llm/pattern_recognition.py`

## Available Tools
- Read market data files
- Execute indicator calculations (RSI, MACD, Bollinger)
- Query historical price levels
- Access support/resistance database

## Input Format
```json
{
  "symbol": "SPY",
  "timeframe": "1D",
  "lookback_days": 30,
  "indicators": ["RSI", "MACD", "BB"]
}
```

## Output Format
```json
{
  "symbol": "SPY",
  "signal": "bullish",
  "confidence": 0.75,
  "patterns_detected": ["double_bottom", "rsi_oversold"],
  "key_levels": {
    "support": [580, 575],
    "resistance": [595, 600]
  },
  "indicators": {
    "rsi": 28.5,
    "macd_signal": "bullish_crossover"
  },
  "reasoning": "RSI oversold with price at major support level"
}
```

## Constraints
- Read-only access to market data
- Must include confidence scores (0.0-1.0)
- Response time SLA: <5 seconds
- Cannot execute trades directly

## Escalation
If confidence < 0.5, escalate to orchestrator for additional analysis sources.
```

#### Risk Guardian Agent (.claude/agents/risk-guardian.md)
```markdown
# Risk Guardian Agent

## Role
Risk management gatekeeper that validates all trading decisions against portfolio limits, circuit breaker state, and market conditions.

## Backend Integration
- Primary: `llm/agents/risk_managers.py`
- Circuit Breaker: `models/circuit_breaker.py`
- Risk Chain: `models/risk_chain.py`
- Portfolio: `models/risk_manager.py`

## Available Tools
- Check circuit breaker state
- Query portfolio exposure
- Calculate position sizing
- Access risk metrics

## Input Format
```json
{
  "action": "validate_trade",
  "order": {
    "symbol": "SPY",
    "quantity": 100,
    "direction": "BUY",
    "price": 590.0
  },
  "portfolio_state": {
    "cash": 50000,
    "positions": [...]
  }
}
```

## Output Format
```json
{
  "approved": true,
  "checks_passed": [
    "circuit_breaker",
    "daily_loss_limit",
    "position_concentration",
    "liquidity"
  ],
  "warnings": ["Approaching daily trade limit (8/10)"],
  "adjusted_quantity": 100,
  "risk_score": 0.35,
  "reasoning": "Trade within all risk parameters"
}
```

## Constraints
- CANNOT be overridden by other agents
- Must check circuit breaker FIRST
- Logs all decisions to audit trail
- Response time SLA: <2 seconds

## Circuit Breaker Integration
```python
from models.circuit_breaker import get_circuit_breaker

breaker = get_circuit_breaker()
if not breaker.can_trade():
    return {"approved": False, "reason": "Circuit breaker OPEN"}
```
```

### 12.2 Creating Skills

Create `.claude/skills/` directory:

```bash
mkdir -p .claude/skills
```

#### Backtest Skill (.claude/skills/SKILL_backtest.md)
```markdown
# Backtest Skill

## Trigger Phrases
- "backtest this strategy"
- "evaluate historical performance"
- "run walk-forward analysis"
- "test on historical data"

## Backend Integration
- Pipeline: `evaluation/orchestration_pipeline.py`
- Walk-Forward: `evaluation/walk_forward_analysis.py`
- Metrics: `observability/metrics/collectors/agent.py`

## Required Parameters
| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| strategy_name | string | Yes | - |
| symbols | list[str] | Yes | - |
| start_date | date | No | 1 year ago |
| end_date | date | No | today |
| initial_capital | float | No | 100000 |

## Execution Steps
1. Validate strategy exists in `algorithms/`
2. Load historical data for specified period
3. Initialize backtest engine
4. Run strategy simulation
5. Calculate performance metrics
6. Generate report

## Output Format
```json
{
  "strategy": "IronCondorStrategy",
  "period": "2024-01-01 to 2024-12-01",
  "metrics": {
    "total_return": 0.24,
    "sharpe_ratio": 1.45,
    "sortino_ratio": 1.82,
    "max_drawdown": -0.12,
    "win_rate": 0.58,
    "profit_factor": 1.8
  },
  "trades": 156,
  "avg_trade_duration": "3.2 days",
  "recommendation": "Strategy meets performance targets",
  "warnings": ["High correlation with SPY during drawdowns"]
}
```

## Error Handling
- Missing data: Request data download or reduce date range
- Strategy error: Return compilation errors with line numbers
- Insufficient capital: Warn and suggest minimum capital

## Example Usage

```
User: "Backtest the iron condor strategy on SPY for 2024"

Agent Response:
1. Loading IronCondorStrategy from algorithms/iron_condor.py
2. Fetching SPY data for 2024-01-01 to 2024-12-01
3. Running backtest with $100,000 initial capital
4. Calculating performance metrics...

Results: [JSON output as specified above]
```
```

#### Options Analysis Skill (.claude/skills/SKILL_options_analysis.md)

```markdown
# Options Analysis Skill

## Trigger Phrases
- "analyze options for"
- "show me the options chain"
- "what's the IV for"
- "calculate Greeks for"

## Backend Integration
- Scanner: `scanners/options_scanner.py`
- Greeks: `indicators/options_greeks.py`
- IV: `indicators/implied_volatility.py`

## Required Parameters

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| symbol | string | Yes | - |
| expiration | date | No | nearest monthly |
| strike_range | float | No | 0.1 (10% from ATM) |

## Output Format

```json
{
  "symbol": "SPY",
  "underlying_price": 590.50,
  "iv_rank": 45,
  "iv_percentile": 52,
  "chain_summary": {
    "calls": {
      "total_volume": 125000,
      "total_oi": 450000,
      "avg_iv": 0.18
    },
    "puts": {
      "total_volume": 98000,
      "total_oi": 380000,
      "avg_iv": 0.22
    }
  },
  "notable_strikes": [
    {"strike": 590, "type": "call", "iv": 0.16, "volume": 15000},
    {"strike": 580, "type": "put", "iv": 0.24, "volume": 12000}
  ],
  "recommendation": "Put IV elevated - consider selling put spreads"
}
```
```

### 12.3 MCP Server Setup Guide

#### Creating Custom Market Data MCP Server

```bash
mkdir -p mcp_servers/market_data
cd mcp_servers/market_data
```

**Server Implementation (mcp_servers/market_data/server.py):**

```python
"""
Custom MCP server for market data.

Provides real-time quotes, options chains, and historical data
to Claude agents via the Model Context Protocol.
"""
import asyncio
import json
from datetime import datetime
from typing import Any

# MCP SDK imports (install with: pip install mcp)
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Your existing market data client
from data.market_data_client import MarketDataClient

server = Server("market-data")
client = MarketDataClient()

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available market data tools."""
    return [
        Tool(
            name="get_quote",
            description="Get real-time quote for a symbol",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol"},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="get_options_chain",
            description="Get options chain for a symbol",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "expiration": {"type": "string", "format": "date"},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="get_historical",
            description="Get historical OHLCV data",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "start_date": {"type": "string", "format": "date"},
                    "end_date": {"type": "string", "format": "date"},
                    "interval": {"type": "string", "enum": ["1m", "5m", "1h", "1d"]},
                },
                "required": ["symbol"],
            },
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    if name == "get_quote":
        quote = await client.get_quote(arguments["symbol"])
        return [TextContent(type="text", text=json.dumps(quote))]

    elif name == "get_options_chain":
        chain = await client.get_options_chain(
            arguments["symbol"],
            arguments.get("expiration"),
        )
        return [TextContent(type="text", text=json.dumps(chain))]

    elif name == "get_historical":
        data = await client.get_historical(
            arguments["symbol"],
            arguments.get("start_date"),
            arguments.get("end_date"),
            arguments.get("interval", "1d"),
        )
        return [TextContent(type="text", text=json.dumps(data))]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

**Configuration (.claude/settings.json):**

```json
{
  "mcpServers": {
    "market-data": {
      "type": "stdio",
      "command": "python",
      "args": ["mcp_servers/market_data/server.py"],
      "description": "Real-time market data, quotes, and options chains"
    },
    "postgres": {
      "type": "stdio",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "${TRADING_DB_URL}"
      },
      "description": "Trade history and backtest results database"
    }
  }
}
```

---

## Part 13: Troubleshooting Guide

### 13.1 Common Migration Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Import errors after consolidation | `ModuleNotFoundError: No module named 'llm.sentiment.core'` | Ensure `__init__.py` files exist in all package directories |
| Circular imports | `ImportError: cannot import name 'X' from partially initialized module` | Move imports inside functions or use `TYPE_CHECKING` guard |
| Missing dependencies | `ImportError: No module named 'transformers'` | Run `pip install -r requirements.txt` |
| Type errors | `mypy: Incompatible types` | Update type hints to use new unified types |
| Test failures | `AssertionError: expected SentimentResult, got dict` | Update tests to use new dataclass types |

### 13.2 Debugging Commands

```bash
# Check import structure
python -c "from llm.sentiment import analyze_sentiment; print('OK')"

# Verify package discovery
python -c "import llm.sentiment; print(dir(llm.sentiment))"

# Check deprecation warnings
python -W default::DeprecationWarning -c "import llm.sentiment"

# Validate type hints
mypy llm/sentiment/ --show-error-codes

# Run specific test with verbose output
pytest tests/test_sentiment_consolidation.py -v -s --tb=long

# Check for circular imports
python -c "
import sys
sys.setrecursionlimit(50)
try:
    import llm.sentiment
    print('No circular imports')
except RecursionError:
    print('Circular import detected!')
"
```

### 13.3 Performance Troubleshooting

```python
# scripts/benchmark_consolidation.py
"""Benchmark consolidated packages against original."""
import time
from statistics import mean, stdev

def benchmark_sentiment(iterations: int = 100):
    """Benchmark sentiment analysis."""
    # New consolidated package
    from llm.sentiment import analyze_sentiment

    test_text = "AAPL shows strong earnings growth potential"

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        analyze_sentiment(test_text, symbol="AAPL")
        times.append(time.perf_counter() - start)

    print(f"Sentiment Analysis ({iterations} iterations):")
    print(f"  Mean: {mean(times)*1000:.2f}ms")
    print(f"  Std:  {stdev(times)*1000:.2f}ms")
    print(f"  Min:  {min(times)*1000:.2f}ms")
    print(f"  Max:  {max(times)*1000:.2f}ms")

if __name__ == "__main__":
    benchmark_sentiment()
```

### 13.4 Circuit Breaker Troubleshooting

```python
# Debug circuit breaker state
from models.circuit_breaker import get_circuit_breaker

breaker = get_circuit_breaker()
print(f"State: {breaker.state}")
print(f"Can trade: {breaker.can_trade()}")
print(f"Daily loss: ${breaker.daily_loss:.2f}")
print(f"Trip count today: {breaker.trip_count}")

# Force reset (use with caution!)
# breaker.reset()
```

### 13.5 Agent Communication Debugging

```python
# Debug agent orchestration
import logging
logging.basicConfig(level=logging.DEBUG)

from llm.agents.supervisor import SupervisorAgent

supervisor = SupervisorAgent()
# Now all agent communication will be logged
```

---

## Part 14: Quick Reference

### 14.1 Consolidation Checklist

```markdown
## Pre-Migration
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Git clean: `git status`
- [ ] Create safety tag: `git tag pre-migration-YYYYMMDD`
- [ ] Create feature branch: `git checkout -b feature/consolidation-X`

## During Migration
- [ ] Create new package structure
- [ ] Migrate core types first
- [ ] Migrate providers/implementations
- [ ] Add deprecation warnings to old files
- [ ] Update imports in dependent files
- [ ] Write tests for new package

## Post-Migration
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Deprecation warnings work
- [ ] Backwards compatibility verified
- [ ] Performance benchmark acceptable
- [ ] Documentation updated
- [ ] Commit and push
```

### 14.2 File Location Quick Reference

| What You Need | Old Location | New Location |
|--------------|--------------|--------------|
| Sentiment analysis | `llm/sentiment.py` | `llm/sentiment/` |
| Reddit sentiment | `llm/reddit_sentiment.py` | `llm/sentiment/providers/reddit.py` |
| News analysis | `llm/news_analyzer.py` | `llm/news/analyzer.py` |
| Anomaly detection | `models/anomaly_detector.py` | `models/anomaly/` |
| Agent metrics | `evaluation/metrics.py` | `observability/metrics/collectors/agent.py` |
| Risk chain | (scattered) | `models/risk_chain.py` |

### 14.3 Import Migration Quick Reference

```python
# OLD → NEW

# Sentiment
from llm.sentiment import SentimentAnalyzer
# →
from llm.sentiment import FinBERTProvider

# Reddit sentiment
from llm.reddit_sentiment import RedditSentiment
# →
from llm.sentiment.providers.reddit import RedditProvider

# News
from llm.news_analyzer import NewsAnalyzer
# →
from llm.news import NewsAnalyzer, analyze_news

# Metrics
from evaluation.metrics import AgentMetrics
# →
from observability.metrics.collectors.agent import AgentMetrics

# Anomaly
from models.anomaly_detector import AnomalyDetector
# →
from models.anomaly import AnomalyDetector, AnomalyEvent
```

### 14.4 Command Quick Reference

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run specific consolidation tests
pytest tests/test_*_consolidation.py -v

# Check for deprecated imports
python scripts/validate_imports.py

# Check consolidation progress
python scripts/consolidation_progress.py

# Benchmark performance
python scripts/benchmark_consolidation.py

# Run type checking
mypy llm/ models/ --ignore-missing-imports

# Run linting
ruff check llm/ models/ execution/

# Create migration tag
git tag -a v2.0.0-pre-consolidation -m "Before consolidation migration"
```

---

## Part 15: Next Steps (Updated: 2025-12-06)

### ✅ Completed (Pre-requisite Work)

- [x] Monitoring consolidation → `observability/monitoring/trading/`
- [x] Deprecated wrapper removal
- [x] Config documentation (config/ vs utils/overnight_config.py)
- [x] Decision/Reasoning logger verification
- [x] Deprecated directories cleanup
- [x] Test suite verification (3548 tests passing)

### Immediate Actions (Next Priority)

1. **Start sentiment consolidation** - Highest duplication, best ROI
   ```bash
   git checkout -b feature/consolidation-sentiment
   git tag pre-sentiment-consolidation
   ```

2. **Start news consolidation** - Can run in parallel with sentiment
   ```bash
   git checkout -b feature/consolidation-news
   git tag pre-news-consolidation
   ```

### Phase 1 Goals (Remaining Consolidation)

| Task | Priority | Status |
|------|----------|--------|
| Complete sentiment package consolidation | P0 | TODO |
| Complete news package consolidation | P0 | TODO |
| Create unified anomaly types | P1 | TODO |
| Implement risk enforcement chain | P1 | TODO |
| Consolidate metrics collectors | P2 | TODO |

### Phase 2 Goals (Claude SDK Integration)

| Task | Priority | Status |
|------|----------|--------|
| Create Claude SDK subagent definitions | P1 | TODO |
| Create skill definitions | P1 | TODO |
| Set up MCP server configs | P2 | TODO |
| Build custom market data MCP server | P2 | TODO |

### Phase 3 Goals (Enhancement - Deferred)

| Task | Priority | Status |
|------|----------|--------|
| Enable parallel subagent execution | P3 | TODO |
| Set up database MCP server | P3 | TODO |
| Create agent monitoring dashboard | P3 | TODO |
| Complete documentation updates | P3 | TODO |

### ⏸️ Deferred Tasks

| Task | Reason |
|------|--------|
| Execution module review | Project in early development, patterns unstable |

### Success Criteria

| Metric | Target | Current | Verification |
|--------|--------|---------|--------------|
| Trading monitors consolidated | 1 package | ✅ DONE | `observability/monitoring/trading/` |
| Sentiment consolidated | 1 package | TODO | `llm/sentiment/` |
| News consolidated | 1 package | TODO | `llm/news/` |
| Test pass rate | 100% | ✅ 3548 passing | `pytest tests/` |
| Import deprecations | 0 | Partial | `python scripts/validate_imports.py` |
| Performance regression | <10% | N/A | Benchmark scripts |

---

## References

### Internal Documents
- [PARALLEL_AGENT_COORDINATION.md](./PARALLEL_AGENT_COORDINATION.md) - **Multi-agent parallel execution system**
- [FIX_GUIDE.md](./FIX_GUIDE.md) - Algorithm bug fixes (P0-1 timestamp bug)
- [CONSOLIDATION_CHANGELOG.md](./CONSOLIDATION_CHANGELOG.md) - Completed work tracker
- [CONSOLIDATION_PLAN.md](./CONSOLIDATION_PLAN.md) - Phase-by-phase execution plan
- [MULTI_AGENT_ENHANCEMENT_PLAN.md](./MULTI_AGENT_ENHANCEMENT_PLAN.md) - Original enhancement plan
- [GIT_MULTI_AGENT_WORKFLOW.md](./GIT_MULTI_AGENT_WORKFLOW.md) - Git workflow documentation
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Current architecture documentation

### External Resources
- [Anthropic: Building agents with Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
- [Anthropic: Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [Claude Flow Framework](https://github.com/ruvnet/claude-flow)
- [AI Native Dev: Parallelizing AI Coding Agents](https://ainativedev.io/news/how-to-parallelize-ai-coding-agents)
- [Simon Willison: Parallel coding agents](https://simonwillison.net/2025/Oct/5/parallel-coding-agents/)
