# Next Refactoring Plan: Technical Debt Remediation

**Created:** 2025-12-05
**Status:** Complete (Phase 6 Deferred)
**Prerequisite:** REFACTOR_PLAN.md Complete ✅
**Note:** SEPARATION_PLAN.md execution was Phase 6, now DEFERRED (see line 480)
**Estimated Effort:** 8-14 weeks

---

## Completed Phases

| Phase | Status | Commit | LOC Impact |
|-------|--------|--------|------------|
| Phase 1 | ✅ DONE | `3e71f07` | -19,449 LOC |
| Phase 2 | ✅ DONE | `b069b6a` | Consolidated logging |
| Phase 3 | ✅ DONE | `982bf77` | -3,164 LOC (from re-exports) |
| Phase 4 | ✅ DONE | `8ab5c9a` | Unified monitoring & alerting |
| Phase 5 | ✅ DONE | `4c981bc` | Clarified module boundaries |
| Phase 6 | ⏸️ DEFERRED | | See notes below |

---

## Research Sources

This plan is based on industry best practices from:

- [Tweag: Python Monorepo Structure](https://www.tweag.io/blog/2023-04-04-python-monorepo-1/)
- [Python Workspaces (Monorepos) 2025](https://tomasrepcik.dev/blog/2025/2025-10-26-python-workspaces/)
- [NautilusTrader Architecture](https://github.com/nautechsystems/nautilus_trader)
- [Architectural Design Patterns for HFT Bots](https://medium.com/@halljames9963/architectural-design-patterns-for-high-frequency-algo-trading-bots-c84f5083d704)
- [From Trading Bot to Trading Agent](https://medium.com/@gwrx2005/from-trading-bot-to-trading-agent-how-to-build-an-ai-based-investment-system-313d4c370c60)
- [Monorepo Best Practices](https://monorepo.tools/)

---

## Executive Summary

**Codebase Statistics:**
- **Total:** 163,219 lines of code
- **Python Files:** 23,790 files
- **Technical Debt:** ~12,000+ LOC of redundant/deprecated code

**Key Issues Identified:**

| Issue | Impact | LOC Affected |
|-------|--------|--------------|
| Deprecated code not removed | High | 8,333+ |
| Duplicate logging systems | High | 1,500+ |
| Duplicate metrics implementations | High | 2,500+ |
| Scattered monitoring/alerting | Medium | 1,500+ |
| Unclear module boundaries | Medium | N/A |

---

## Priority Matrix

### P0 - Critical (Do First)

1. **Remove Deprecated Code** - 8,333+ LOC cluttering codebase
2. **Consolidate Logging** - 5+ parallel logging systems

### P1 - High Priority

3. **Consolidate Metrics** - 8+ metrics implementations
4. **Unify Monitoring/Alerting** - 8+ scattered modules

### P2 - Medium Priority

5. **Clarify Module Boundaries** - Circular dependencies
6. **Execute SEPARATION_PLAN.md** - Trading vs Dev tools

### P3 - Low Priority (Future)

7. **Modern Tooling Migration** - UV workspaces, Pants/Bazel
8. **Full Test Coverage** - Target 80%+

---

## Phase 1: Remove Deprecated Code

**Goal:** Reduce codebase by ~8,333+ LOC of unused code

### 1.1 Clean Deprecated Hooks

**Location:** `.claude/hooks/deprecated/`

| File | LOC | Action |
|------|-----|--------|
| `ric_v45.py` | ~3,000 | DELETE |
| `ric_v50_dev.py` | ~1,500 | DELETE |
| `ric_hooks.py` | ~500 | DELETE |
| `ric_enforcer.py` | ~800 | DELETE |
| `ric_prompts.py` | ~300 | DELETE |
| `ric_state_manager.py` | ~600 | DELETE |
| `enforce_ric_compliance.py` | ~400 | DELETE |
| `session_stop_v1.py` | ~300 | DELETE |

**Commands:**
```bash
cd /home/dshooter/projects/Claude_code_Quantconnect_trading_bot

# Verify no imports from deprecated
grep -r "from.*deprecated" --include="*.py" | grep -v "deprecated/"
grep -r "import.*ric_v4\|ric_v50" --include="*.py" | grep -v deprecated

# If clean, remove
rm -rf .claude/hooks/deprecated/*.py
touch .claude/hooks/deprecated/.gitkeep  # Keep directory for structure
```

### 1.2 Clean Deprecated Agents

**Location:** `agents_deprecated/`

```bash
# Verify no imports
grep -r "from agents_deprecated\|import agents_deprecated" --include="*.py"

# If clean, remove directory
rm -rf agents_deprecated/
```

### 1.3 Clean Archive Directory

**Location:** `.claude/archive/`

```bash
# Review contents
ls -la .claude/archive/

# Archive to git history, then remove
git add .claude/archive/
git commit -m "archive: Preserve deprecated archive files before removal"

# Remove after commit
rm -rf .claude/archive/
```

### 1.4 Clean Stale Pycache

```bash
# Remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove .pyc files
find . -name "*.pyc" -delete
```

### 1.5 Verification

```bash
# Count remaining LOC
find . -name "*.py" -not -path "./venv/*" -not -path "./.venv/*" | xargs wc -l | tail -1

# Should show ~155,000 LOC (down from 163,000)
```

### 1.6 Commit Phase 1

```bash
git add -A
git commit -m "refactor: Remove deprecated code (Phase 1)

- Removed deprecated hooks (8,333+ LOC)
- Removed agents_deprecated/ directory
- Removed .claude/archive/ directory
- Cleaned __pycache__ directories

Reduces codebase by ~8,000+ lines of unused code.

$(cat <<'EOF'
Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Phase 2: Consolidate Logging Infrastructure

**Goal:** Replace 5+ logging implementations with unified system

### 2.1 Current State (Scattered)

| File | Purpose | LOC |
|------|---------|-----|
| `utils/structured_logger.py` | JSON structured logging | ~300 |
| `compliance/audit_logger.py` | Compliance audit trail | ~250 |
| `llm/decision_logger.py` | Agent decision logging | ~400 |
| `llm/reasoning_logger.py` | Reasoning chain logging | ~300 |
| `.claude/hooks/core/hook_utils.py` | Hook logging | ~200 |

### 2.2 Target Architecture

```
observability/
├── __init__.py
├── logging/
│   ├── __init__.py
│   ├── base.py              # AbstractLogger interface
│   ├── structured.py        # JSON structured logging
│   ├── audit.py             # Compliance audit (FINRA, SEC)
│   └── adapters/
│       ├── __init__.py
│       ├── decision.py      # Agent decision adapter
│       ├── reasoning.py     # Reasoning chain adapter
│       ├── hook.py          # Hook logging adapter
│       └── trading.py       # Trading event adapter
└── ... (existing metrics, tracing)
```

### 2.3 Implementation Steps

**Step 1: Create base interface**
```python
# observability/logging/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from enum import Enum

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    AUDIT = "audit"  # For compliance

@dataclass
class LogEntry:
    level: LogLevel
    category: str
    message: str
    context: Dict[str, Any]
    timestamp: str
    correlation_id: Optional[str] = None

class AbstractLogger(ABC):
    @abstractmethod
    def log(self, entry: LogEntry) -> None: ...

    @abstractmethod
    def audit(self, action: str, details: Dict[str, Any]) -> None: ...
```

**Step 2: Migrate each logger**

```bash
# Create new structure
mkdir -p observability/logging/adapters

# Create migration script
python scripts/migrate_loggers.py --dry-run
python scripts/migrate_loggers.py --execute
```

**Step 3: Update imports across codebase**

```bash
# Find all logging imports
grep -r "from utils.structured_logger\|from compliance.audit_logger\|from llm.decision_logger\|from llm.reasoning_logger" --include="*.py" -l

# Update each file to use new imports
# from observability.logging import StructuredLogger, AuditLogger
```

### 2.4 Verification

```bash
# Ensure all tests pass
pytest tests/ -x -q

# Ensure no old imports remain
grep -r "from utils.structured_logger" --include="*.py"
# Should return empty
```

### 2.5 Commit Phase 2

```bash
git add -A
git commit -m "refactor: Consolidate logging infrastructure (Phase 2)

- Created unified logging framework in observability/logging/
- Migrated structured_logger, audit_logger, decision_logger, reasoning_logger
- All logging now uses consistent interface
- Reduced redundant code by ~1,000+ LOC

$(cat <<'EOF'
Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Phase 3: Consolidate Metrics Infrastructure

**Goal:** Replace 8+ metrics implementations with unified framework

### 3.1 Current State (Scattered)

| File | Purpose | LOC |
|------|---------|-----|
| `observability/metrics.py` | Generic metrics | ~300 |
| `observability/metrics_aggregator.py` | Aggregation | ~400 |
| `observability/token_metrics.py` | Token/cost tracking | ~300 |
| `evaluation/metrics.py` | Agent metrics | ~300 |
| `evaluation/agent_metrics.py` | Agent-specific | ~400 |
| `evaluation/advanced_trading_metrics.py` | Trading metrics | ~500 |
| `execution/execution_quality_metrics.py` | Execution quality | ~400 |
| `models/performance_tracker.py` | Performance | ~400 |

### 3.2 Target Architecture

```
observability/
├── __init__.py
├── logging/          # From Phase 2
├── metrics/
│   ├── __init__.py
│   ├── base.py       # AbstractMetrics, MetricType enum
│   ├── aggregator.py # Unified aggregation
│   ├── exporters/
│   │   ├── __init__.py
│   │   ├── prometheus.py
│   │   ├── json.py
│   │   └── csv.py
│   └── collectors/
│       ├── __init__.py
│       ├── trading.py    # From advanced_trading_metrics
│       ├── execution.py  # From execution_quality_metrics
│       ├── agent.py      # From agent_metrics
│       ├── token.py      # From token_metrics
│       └── system.py     # System resource metrics
└── tracing/          # Existing
```

### 3.3 Implementation Steps

Similar to Phase 2:
1. Create base interface in `observability/metrics/base.py`
2. Create collector modules
3. Create aggregator
4. Migrate imports across codebase
5. Remove old implementations

### 3.4 Commit Phase 3

```bash
git add -A
git commit -m "refactor: Consolidate metrics infrastructure (Phase 3)

- Created unified metrics framework in observability/metrics/
- Consolidated 8 metrics implementations into collectors/
- Added exporters for Prometheus, JSON, CSV
- Reduced redundant code by ~2,000+ LOC

$(cat <<'EOF'
Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Phase 4: Unify Monitoring & Alerting

**Goal:** Consolidate 8+ monitoring modules

### 4.1 Current State (Scattered)

| File | Purpose |
|------|---------|
| `evaluation/continuous_monitoring.py` | Continuous monitoring |
| `evaluation/anomaly_alerting_bridge.py` | Anomaly alerts |
| `utils/alerting_service.py` | Alert service |
| `utils/resource_monitor.py` | Resource monitoring |
| `utils/system_monitor.py` | System monitoring |
| `utils/storage_monitor.py` | Storage monitoring |
| `llm/news_alert_manager.py` | News alerts |
| `models/anomaly_detector.py` | Anomaly detection |

### 4.2 Target Architecture

```
observability/
├── logging/          # From Phase 2
├── metrics/          # From Phase 3
├── monitoring/
│   ├── __init__.py
│   ├── base.py       # AbstractMonitor interface
│   ├── continuous.py # Unified continuous monitoring
│   ├── anomaly.py    # Anomaly detection & alerting
│   ├── system/
│   │   ├── __init__.py
│   │   ├── resource.py   # CPU, memory, etc.
│   │   ├── storage.py    # Disk usage
│   │   └── network.py    # Network health
│   └── domain/
│       ├── __init__.py
│       ├── trading.py    # Trading-specific monitors
│       └── news.py       # News & sentiment monitors
├── alerting/
│   ├── __init__.py
│   ├── base.py       # AbstractAlertHandler
│   ├── channels/
│   │   ├── email.py
│   │   ├── slack.py
│   │   ├── discord.py
│   │   └── webhook.py
│   └── rules.py      # Alert rule definitions
└── tracing/
```

### 4.3 Commit Phase 4

```bash
git add -A
git commit -m "refactor: Unify monitoring and alerting (Phase 4)

- Created observability/monitoring/ for all monitors
- Created observability/alerting/ for alert channels
- Consolidated 8 scattered monitoring modules
- Unified alert channel configuration

$(cat <<'EOF'
Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Phase 5: Clarify Module Boundaries

**Goal:** Eliminate circular dependencies, establish clear layering

### 5.1 Current Dependency Issues

```
evaluation/ → observability/, compliance/, execution/, models/, llm/
llm/        → utils/, compliance/, observability/, models/
models/     → utils/, config/, infrastructure/
execution/  → utils/, models/, observability/, compliance/
```

### 5.2 Target Layered Architecture

```
Layer 4: Applications (algorithms/, api/, ui/)
    ↓
Layer 3: Domain Logic (execution/, llm/, evaluation/)
    ↓
Layer 2: Core Models (models/, compliance/)
    ↓
Layer 1: Infrastructure (observability/, infrastructure/, config/)
    ↓
Layer 0: Utilities (utils/)
```

**Rules:**
- Each layer can only import from layers below it
- No circular dependencies within a layer
- Cross-cutting concerns (logging, metrics) use dependency injection

### 5.3 Implementation

1. Add `# Layer: N` comment to each module's `__init__.py`
2. Create `scripts/check_layer_violations.py` to enforce boundaries
3. Add to pre-commit hooks

### 5.4 Commit Phase 5

```bash
git add -A
git commit -m "refactor: Establish module layer boundaries (Phase 5)

- Defined 5-layer architecture
- Added layer annotations to all modules
- Created layer violation checker
- Documented dependency rules

$(cat <<'EOF'
Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Phase 6: Execute SEPARATION_PLAN.md (DEFERRED)

**Goal:** Separate trading infrastructure from development tools

**Status:** ⏸️ DEFERRED - Not required given Phase 5 layer architecture

### Why Deferred

The SEPARATION_PLAN.md was written before Phase 5's 5-layer architecture was established.
Analysis revealed significant conflicts:

1. **Layer Architecture Conflict**: The plan classifies `observability/` and `utils/` as dev tools
   to move to `src/claude_dev/`, but Phase 5 established them as shared infrastructure
   (Layer 0-1) that trading code depends on.

2. **Import Impact**: Migration would require changing 334+ imports across the codebase:
   - llm: 182 imports
   - execution: 44 imports
   - ui: 41 imports
   - api: 33 imports
   - scanners: 15 imports
   - indicators: 14 imports
   - algorithms: 5 imports

3. **Circular Import Risk**: Simple sed-based import updates caused cascading failures
   due to nested relative imports (e.g., `from .agents.base` vs `from ..prompts`).

4. **Phase 5 Already Achieves Goals**: The 5-layer architecture with:
   - Layer annotations in all `__init__.py` files
   - `scripts/check_layer_violations.py` enforcement
   - Pre-commit hook for layer checking

   Already provides clear module boundaries without physical file moves.

### Recommendation

Keep the current flat structure with layer annotations. The benefits of physical
separation don't justify the risk and effort of 334+ import changes.

If separation is still desired in the future:
1. Use a proper AST-based import migration tool (not sed)
2. Plan for 2-3 days of focused migration work
3. Consider keeping Layer 0-2 at root (shared infrastructure)
4. Only move Layer 3-4 (domain logic and applications)

Original plan preserved below for reference:

---

**Original Goal:** Separate trading infrastructure from development tools

The original SEPARATION_PLAN.md proposed moving:
- Trading code → `src/trading/`
- Dev tools → `src/claude_dev/`

See [SEPARATION_PLAN.md](SEPARATION_PLAN.md) for the original detailed steps.

---

## Post-Refactor Verification

### Automated Checks

```bash
# 1. All Python files compile
find . -name "*.py" -not -path "./venv/*" -exec python3 -m py_compile {} \;

# 2. All tests pass
pytest tests/ -x -q

# 3. No circular imports
python scripts/check_circular_imports.py

# 4. Layer violations
python scripts/check_layer_violations.py

# 5. No deprecated imports
grep -r "from.*deprecated\|import.*deprecated" --include="*.py"

# 6. Hooks functional
python .claude/hooks/core/ric.py status
```

### Manual Verification

- [ ] Claude Code hooks respond correctly
- [ ] RIC loop workflow functions
- [ ] Trading algorithms backtest successfully
- [ ] API endpoints respond
- [ ] Dashboard loads

---

## Timeline Estimate

| Phase | Description | Effort |
|-------|-------------|--------|
| Phase 1 | Remove deprecated code | 1-2 days |
| Phase 2 | Consolidate logging | 1-2 weeks |
| Phase 3 | Consolidate metrics | 1-2 weeks |
| Phase 4 | Unify monitoring | 1-2 weeks |
| Phase 5 | Module boundaries | 1 week |
| Phase 6 | SEPARATION_PLAN.md | 1-2 weeks |

**Total:** 6-10 weeks (assuming focused effort)

---

## Quick Start

```bash
# To execute this plan, tell Claude:
"Read .claude/NEXT_REFACTOR_PLAN.md and execute Phase 1"
```

---

## Notes

1. **Execute phases in order** - Each builds on previous
2. **Commit after each phase** - Easy rollback
3. **Test after each phase** - Catch regressions early
4. **Phase 1 is safe** - Only removes unused code
5. **Phases 2-4 are riskier** - Require careful import migration
6. **Phase 6 is major** - Consider doing after stabilization

---

## Appendix: Files to Delete (Phase 1)

```
.claude/hooks/deprecated/ric_v45.py
.claude/hooks/deprecated/ric_v50_dev.py
.claude/hooks/deprecated/ric_hooks.py
.claude/hooks/deprecated/ric_enforcer.py
.claude/hooks/deprecated/ric_prompts.py
.claude/hooks/deprecated/ric_state_manager.py
.claude/hooks/deprecated/enforce_ric_compliance.py
.claude/hooks/deprecated/session_stop_v1.py
.claude/archive/ (entire directory)
agents_deprecated/ (entire directory)
```

**Estimated removal:** ~8,333+ LOC
