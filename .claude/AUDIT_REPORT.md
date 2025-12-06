# Project Audit Report

**Date:** 2025-12-05
**Updated:** 2025-12-06
**Status:** REFACTOR_PLAN Complete ✅, NEXT_REFACTOR_PLAN Phases 1-5 Complete ✅
**Refactor Plan:** See [REFACTOR_PLAN.md](REFACTOR_PLAN.md) (All 6 phases complete)
**Separation Plan:** See [SEPARATION_PLAN.md](SEPARATION_PLAN.md) (⏸️ DEFERRED)

---

## Executive Summary

This project has grown organically through multiple upgrade cycles (UPGRADE-001 through UPGRADE-019). After refactoring:

### Resolved
- ~~3 separate state management implementations~~ → ✅ Consolidated
- ~~2 duplicate audit logger implementations~~ → ✅ Removed duplicate
- ~~27 hooks with scattered functionality~~ → ✅ Organized into 6 categories
- ~~2 incompatible agent architectures~~ → ✅ Consolidated to `llm/agents/`
- ~~RIC version sprawl~~ → ✅ Consolidated to `.claude/hooks/core/ric.py`

### Deferred
- **Trading/dev separation** - SEPARATION_PLAN.md deferred (5-layer architecture sufficient)

**Next Steps:** Execute [EXCEPTION_REFACTOR_PLAN.md](EXCEPTION_REFACTOR_PLAN.md) or [AUTONOMOUS_ARCHITECTURE_PLAN.md](AUTONOMOUS_ARCHITECTURE_PLAN.md)

---

## Critical Issues (P0) ✅ RESOLVED

### 1. Session State Management Duplication ✅ FIXED

| File | Purpose | State File |
|------|---------|------------|
| `scripts/session_state_manager.py` | Session persistence | `logs/session_state.json` |
| `.claude/hooks/session_stop.py` | Stop hook with state | `logs/session_state.json` |
| `.claude/hooks/hook_utils.py` | Hook utilities | `session_state.json`, `recovery_state.json` |

**Resolution:** Consolidated to `hook_utils.py` with separate state file. `session_stop.py` now merges state properly.

---

### 2. Audit Logger Duplication ✅ FIXED

| File | Lines | Purpose |
|------|-------|---------|
| `compliance/audit_logger.py` | ~500 | Compliance audit logging |
| ~~`models/audit_logger.py`~~ | ~~500~~ | ~~DELETED~~ |

**Resolution:** Kept `compliance/audit_logger.py`, removed duplicate from `models/`.

---

## High Priority Issues (P1)

### 3. Two Incompatible Agent Architectures

**Architecture 1: `agents/`**
```
agents/
├── base_agent.py       # BaseAgent with AgentState enum
├── orchestrator.py     # AgentOrchestrator
├── market_agent.py
├── risk_agent.py
└── strategy_agent.py
```

**Architecture 2: `llm/agents/`**
```
llm/agents/
├── base.py             # TradingAgent (different base)
├── supervisor.py       # SupervisorAgent
├── registry.py         # AgentRegistry
├── technical_analyst.py
├── sentiment_analyst.py
├── traders.py
└── risk_managers.py
```

**Conflicts:**
- Different base classes
- Different state enums (AgentState vs AgentRole)
- Different message passing (AgentMessage vs AgentResponse)
- Different orchestration patterns

**Fix:** Deprecate `agents/`, use `llm/agents/` as primary.

---

### 4. RIC Version Sprawl ✅ CONSOLIDATED

| Location | Version | Status |
|----------|---------|--------|
| `.claude/hooks/core/ric.py` | v5.1 | **ACTIVE** |
| ~~`.claude/hooks/deprecated/ric_v50_dev.py`~~ | ~~v5.0-dev~~ | Removed |
| ~~`.claude/hooks/deprecated/ric_v45.py`~~ | ~~v4.5~~ | Removed |

**Resolution:** Consolidated to `.claude/hooks/core/ric.py`. Deprecated versions removed.

---

### 5. Hook System Sprawl (27 hooks) ✅ ORGANIZED

**Resolution:** Hooks organized into subdirectories (Option A chosen - keep separate files):

```
.claude/hooks/
├── core/           # ric.py, session_stop.py, protect_files.py, hook_utils.py, pre_compact.py
├── validation/     # validate_algorithm.py, validate_research.py, validate_category_docs.py, qa_auto_check.py, algo_change_guard.py
├── research/       # document_research.py, research_saver.py, thorough_research.py, auto_research_trigger.py
├── trading/        # risk_validator.py, log_trade.py, parse_backtest.py, load_trading_context.py
├── formatting/     # auto_format.py, update_cross_refs.py, template_detector.py
├── agents/         # agent_orchestrator.py, multi_agent.py
└── deprecated/     # Legacy hooks
```

---

## Medium Priority Issues (P2)

### 6. Documentation with Version Numbers

**Current (problematic):**
```
docs/
├── PROMPT_V4_FRAMEWORK.md
├── PROMPT_V5_FRAMEWORK.md
├── PROMPT_V6_FRAMEWORK.md
```

**Proposed:**
```
docs/prompts/
├── FRAMEWORK.md              # Current active version
└── versions/
    ├── v4_framework.md
    ├── v5_framework.md
    └── v6_framework.md
```

---

### 7. Config File Proliferation

| File | Purpose |
|------|---------|
| `.claude/settings.json` | Claude Code settings |
| `.claude/settings.local.json` | Local overrides |
| `.claude/agent_config.json` | Agent orchestration |
| `config/settings.json` | Trading config |
| `config/__init__.py` | Python config |

**Fix:** Consolidate into:
- `.claude/config/settings.json` - Claude Code
- `config/trading.json` - Trading

---

## Architecture Separation Proposal

### Current State (Mixed)

```
project/
├── .claude/          # Dev tools (hooks, commands, templates)
├── algorithms/       # Trading
├── agents/           # Trading (deprecated)
├── llm/agents/       # Trading
├── execution/        # Trading
├── mcp/              # Trading
├── scripts/          # MIXED (dev + trading)
├── models/           # MIXED (dev + trading)
└── evaluation/       # Dev tools
```

### Proposed Separation

```
project/
├── .claude/                    # Claude Code development tools
│   ├── hooks/                  # Consolidated hooks
│   ├── commands/               # Slash commands
│   ├── config/                 # Dev tool configs
│   └── state/                  # State files
│
├── trading/                    # Live trading infrastructure
│   ├── algorithms/             # Trading algorithms
│   ├── agents/                 # LLM trading agents (from llm/agents/)
│   ├── execution/              # Order execution
│   ├── mcp/                    # MCP servers
│   ├── models/                 # Trading models
│   └── config/                 # Trading configs
│
├── evaluation/                 # Dev tools (keep)
├── docs/                       # Documentation
├── tests/                      # Tests
└── scripts/                    # Renamed dev scripts only
```

---

## State File Rationalization

### Current (Scattered)
```
.claude/ric_state.json
.claude/research_state.json
.claude/agent_config.json
logs/session_state.json
logs/recovery_state.json
session_state.json
recovery_state.json
```

### Proposed (Organized)
```
.claude/state/
├── ric.json          # RIC loop state
├── session.json      # Session state
├── research.json     # Research state
└── recovery.json     # Recovery state

.claude/config/
├── settings.json     # Main settings
├── agents.json       # Agent config
└── hooks.json        # Hook config
```

---

## Recommended Refactoring Phases

### Phase 1: P0 Fixes (Immediate)
1. Consolidate session state management into single implementation
2. Remove duplicate audit logger (keep `compliance/audit_logger.py`)
3. Update all imports to use consolidated versions

### Phase 2: P1 Fixes (Week 1)
1. Deprecate `agents/` directory
2. Consolidate hooks (validators, research, agents)
3. Clean up RIC version references
4. Organize state files

### Phase 3: P2 Fixes (Week 2)
1. Separate trading infrastructure from dev tools
2. Organize documentation versions
3. Consolidate config files
4. Update all imports and references

### Phase 4: Cleanup (Week 3)
1. Remove deprecated directories
2. Update documentation
3. Run full test suite
4. Final verification

---

## Files to Delete After Refactoring

```
# Duplicate audit logger
models/audit_logger.py

# Deprecated agent system
agents/base_agent.py
agents/orchestrator.py
agents/market_agent.py
agents/risk_agent.py
agents/strategy_agent.py

# Deprecated RIC versions (already in deprecated/)
.claude/hooks/deprecated/*

# Archive after extracting useful info
.claude/archive/ric_v4*.py
```

---

## Questions for User

Before proceeding with refactoring:

1. **Agent Architecture:** Confirm deprecating `agents/` in favor of `llm/agents/`?

2. **Trading Separation:** Should we create a `trading/` subdirectory for all trading code?

3. **State Files:** Confirm new state file organization under `.claude/state/`?

4. **Hook Consolidation:** Which hooks should remain separate vs consolidated?

5. **Documentation:** Should versioned docs (PROMPT_V4, V5, V6) be moved to `docs/versions/`?

---

## Next Steps

1. Review this audit report
2. Approve refactoring phases
3. Create backup branch before changes
4. Execute Phase 1 (P0 fixes)
5. Test and verify
6. Continue with subsequent phases
