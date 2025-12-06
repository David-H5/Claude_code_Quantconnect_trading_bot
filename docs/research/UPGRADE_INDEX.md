# Upgrade to Research Document Index

**Purpose**: Maps upgrade numbers to their associated research documents, avoiding the need to name documents by upgrade number while maintaining traceability.

**Principle**: Research documents are named by SUBJECT (for findability), but linked to upgrades here (for traceability).

---

## Quick Lookup

| Upgrade | Title | Primary Research | Implementation Guide |
|---------|-------|------------------|---------------------|
| UPGRADE-008 | Enhanced RIC Loop | [AUTONOMOUS_WORKFLOW_RESEARCH.md](AUTONOMOUS_WORKFLOW_RESEARCH.md) | [UPGRADE-008-ENHANCED-RIC-LOOP.md](UPGRADE-008-ENHANCED-RIC-LOOP.md) |
| UPGRADE-009 | Workflow Enhancements | [WORKFLOW_ENHANCEMENT_RESEARCH.md](WORKFLOW_ENHANCEMENT_RESEARCH.md) | [UPGRADE-009-WORKFLOW-ENHANCEMENTS.md](UPGRADE-009-WORKFLOW-ENHANCEMENTS.md) |
| UPGRADE-010 | Advanced AI Features | [ADVANCED_FEATURES_RESEARCH.md](ADVANCED_FEATURES_RESEARCH.md) | [UPGRADE-010-ADVANCED-FEATURES.md](UPGRADE-010-ADVANCED-FEATURES.md) |
| UPGRADE-010-S5 | Quality & Coverage (Core) | Sprint 5 | [UPGRADE-010-SPRINT5-QUALITY-COVERAGE.md](UPGRADE-010-SPRINT5-QUALITY-COVERAGE.md) |
| UPGRADE-010-S6 | Quality & Coverage (Exec) | Sprint 6 | [UPGRADE-010-SPRINT6-TEST-COVERAGE.md](UPGRADE-010-SPRINT6-TEST-COVERAGE.md) |
| UPGRADE-011 | Overnight Sessions | [UPGRADE-011-OVERNIGHT-SESSIONS.md](UPGRADE-011-OVERNIGHT-SESSIONS.md) | See research doc |
| UPGRADE-012 | Hierarchical Prompts | [UPGRADE-012-HIERARCHICAL-PROMPTS.md](UPGRADE-012-HIERARCHICAL-PROMPTS.md) | See sub-upgrades |
| UPGRADE-012.1 | Depth+Width Classification | [UPGRADE-012.1-DEPTH-WIDTH-CLASSIFICATION.md](UPGRADE-012.1-DEPTH-WIDTH-CLASSIFICATION.md) | See research doc |
| UPGRADE-012.2 | ACE Reflector & Semantic Routing | [UPGRADE-012.2-ACE-REFLECTOR-SEMANTIC-ROUTING.md](UPGRADE-012.2-ACE-REFLECTOR-SEMANTIC-ROUTING.md) | See research doc |
| UPGRADE-012.3 | Meta-RIC Loop v3.0 | [UPGRADE-012.3-META-RIC-LOOP.md](UPGRADE-012.3-META-RIC-LOOP.md) | See research doc |
| UPGRADE-014 | LLM Sentiment Integration | [LLM_SENTIMENT_RESEARCH.md](LLM_SENTIMENT_RESEARCH.md) | See research doc |
| UPGRADE-014-EXP | LLM Sentiment Expansion | [LLM_SENTIMENT_EXPANSION_RESEARCH.md](LLM_SENTIMENT_EXPANSION_RESEARCH.md) | See research doc |

---

## Detailed Upgrade Mapping

### UPGRADE-008: Enhanced RIC Loop

**Status**: ✅ Implemented

**Research Documents**:

- Primary: [AUTONOMOUS_WORKFLOW_RESEARCH.md](AUTONOMOUS_WORKFLOW_RESEARCH.md)
- Related: [WORKFLOW_MANAGEMENT_RESEARCH.md](WORKFLOW_MANAGEMENT_RESEARCH.md)

**Implementation**:

- Checklist: [UPGRADE-008-ENHANCED-RIC-LOOP.md](UPGRADE-008-ENHANCED-RIC-LOOP.md)
- Workflow: [../development/ENHANCED_RIC_WORKFLOW.md](../development/ENHANCED_RIC_WORKFLOW.md)

**Tags**: `workflow`, `ric-loop`, `autonomous`, `convergence`

---

### UPGRADE-009: Workflow Enhancements

**Status**: 🔄 In Progress

**Research Documents**:

- Primary: [WORKFLOW_ENHANCEMENT_RESEARCH.md](WORKFLOW_ENHANCEMENT_RESEARCH.md)
- Related: [WORKFLOW_MANAGEMENT_RESEARCH.md](WORKFLOW_MANAGEMENT_RESEARCH.md)
- Related: [INSTRUCTION_FILES_UPGRADE_GUIDE.md](INSTRUCTION_FILES_UPGRADE_GUIDE.md)

**Implementation**:

- Checklist: [UPGRADE-009-WORKFLOW-ENHANCEMENTS.md](UPGRADE-009-WORKFLOW-ENHANCEMENTS.md)

**Tags**: `workflow`, `ci-cd`, `testing`, `security`, `code-quality`

---

### UPGRADE-010: Advanced AI Trading Features

**Status**: ✅ Implemented (6 Sprints Complete)

**Research Documents**:

- Primary: [ADVANCED_FEATURES_RESEARCH.md](ADVANCED_FEATURES_RESEARCH.md)
- Related: [LLM_TRADING_RESEARCH.md](LLM_TRADING_RESEARCH.md)
- Related: [EVALUATION_FRAMEWORK_RESEARCH.md](EVALUATION_FRAMEWORK_RESEARCH.md)

**Implementation**:

- Checklist: [UPGRADE-010-ADVANCED-FEATURES.md](UPGRADE-010-ADVANCED-FEATURES.md)
- Sprint 5: [UPGRADE-010-SPRINT5-QUALITY-COVERAGE.md](UPGRADE-010-SPRINT5-QUALITY-COVERAGE.md)
- Sprint 6: [UPGRADE-010-SPRINT6-TEST-COVERAGE.md](UPGRADE-010-SPRINT6-TEST-COVERAGE.md)
- 98 features across P0-P3 priorities
- 6-sprint implementation completed

**Feature Categories**:

| Category | P0 | P1 | P2 | P3 | Total |
|----------|----|----|----|----|-------|
| Multi-Agent LLM | 4 | 5 | 1 | 0 | 10 |
| Reinforcement Learning | 2 | 3 | 1 | 1 | 7 |
| Alternative Data | 2 | 4 | 4 | 0 | 10 |
| Execution Optimization | 2 | 4 | 3 | 0 | 9 |
| Risk Management | 2 | 3 | 4 | 0 | 9 |
| Explainable AI | 2 | 2 | 3 | 0 | 7 |
| Graph/Transformer | 0 | 3 | 3 | 0 | 6 |
| Options Trading | 2 | 4 | 4 | 1 | 11 |
| Backtesting | 2 | 4 | 4 | 0 | 10 |
| Signal Aggregation | 1 | 4 | 4 | 1 | 10 |
| Other | 0 | 4 | 5 | 0 | 9 |
| **Total** | **19** | **40** | **36** | **3** | **98** |

**Tags**: `ai-features`, `rl`, `xai`, `multi-agent`, `execution`, `risk`, `options`

---

### UPGRADE-011: Enhanced Overnight Autonomous Sessions

**Status**: ✅ Implemented

**Research Documents**:

- Primary: [UPGRADE-011-OVERNIGHT-SESSIONS.md](UPGRADE-011-OVERNIGHT-SESSIONS.md)
- Related: [AUTONOMOUS_WORKFLOW_RESEARCH.md](AUTONOMOUS_WORKFLOW_RESEARCH.md)
- Related: [../autonomous-agents/README.md](../autonomous-agents/README.md)

**Implementation**:

- Stop Hook: `.claude/hooks/session_stop.py` (continuous mode)
- PreCompact Hook: `.claude/hooks/pre_compact.py` (transcript backup)
- Auto-Resume: `scripts/auto-resume.sh` (jitter in backoff)
- Overnight Script: `scripts/run_overnight.sh` (--continuous flag)
- Session Notes: `scripts/templates/session-notes-template.md` (relay-race pattern)

**Key Features**:

| Feature | Description |
|---------|-------------|
| Continuous Mode | `--continuous` flag blocks Claude from stopping until all tasks complete |
| Relay-Race Pattern | `claude-session-notes.md` persists context across restarts |
| Hook Timeouts | Explicit 5-60s timeouts prevent blocking |
| Jitter in Backoff | ±25% randomization prevents thundering herd |
| Transcript Backup | PreCompact uses `CLAUDE_TOOL_INPUT` for path |

**Tags**: `overnight`, `autonomous`, `hooks`, `crash-recovery`, `continuous-mode`

---

### UPGRADE-012: Hierarchical Prompts & Meta-RIC

**Status**: ✅ v3.0 Implemented (Meta-RIC Loop)

**Research Documents**:

- Parent: [UPGRADE-012-HIERARCHICAL-PROMPTS.md](UPGRADE-012-HIERARCHICAL-PROMPTS.md)
- Sub-upgrade 1: [UPGRADE-012.1-DEPTH-WIDTH-CLASSIFICATION.md](UPGRADE-012.1-DEPTH-WIDTH-CLASSIFICATION.md)
- Sub-upgrade 2: [UPGRADE-012.2-ACE-REFLECTOR-SEMANTIC-ROUTING.md](UPGRADE-012.2-ACE-REFLECTOR-SEMANTIC-ROUTING.md)
- Sub-upgrade 3: [UPGRADE-012.3-META-RIC-LOOP.md](UPGRADE-012.3-META-RIC-LOOP.md)

**Implementation**:

- Task Router: `scripts/select_prompts.py` (depth+width classification, semantic routing)
- ACE Reflector: `scripts/ace_reflector.py` (pattern extraction from session outcomes)
- Semantic Config: `config/semantic-routes.yaml` (embedding model, thresholds, candidates)
- Meta-RIC Workflow: `prompts/complexity/L1_complex.md` (insight-driven iteration)
- CLAUDE.md: Updated with Meta-RIC Loop v3.0

**Key Features**:

| Version | Feature | Description |
|---------|---------|-------------|
| v1.0 | Task Router | Rule-based keyword matching for complexity/domain |
| v1.1 | Depth+Width | Two-dimensional complexity scoring (12 indicators each) |
| v1.2 | ACE Reflector | Pattern extraction from session outcomes |
| v1.2 | Semantic Routing | Hybrid keyword→semantic→fallback cascade |
| v2.0 | Meta-RIC Loop | Insight-driven min-max iteration (replaces score convergence) |
| v3.0 | Meta-RIC Loop Enhanced | Strict sequential, ALL loops to Phase 0, min 3 iterations, P2 required |

**Research Sources**:

- [DeepWideSearch Benchmark (arXiv, Oct 2025)](https://arxiv.org/pdf/2510.20168)
- [ACE Framework - Stanford (arXiv, Oct 2025)](https://arxiv.org/abs/2510.04618)
- [Signal-Decision Architecture (vLLM, Nov 2025)](https://blog.vllm.ai/2025/11/19/signal-decision.html)
- [MetaAgent: Self-Evolving via Meta-Learning (arXiv, Aug 2025)](https://arxiv.org/html/2508.00271v2)
- [Self-Refine: Iterative Refinement (arXiv, Mar 2023)](https://arxiv.org/abs/2303.17651)

**Tags**: `prompts`, `routing`, `complexity`, `semantic`, `ace-framework`, `meta-ric`, `iteration`

---

### UPGRADE-014: LLM Sentiment Integration

**Status**: ✅ 7/8 Features Implemented

**Research Documents**:

- Primary: [LLM_SENTIMENT_RESEARCH.md](LLM_SENTIMENT_RESEARCH.md)
- Expansion: [LLM_SENTIMENT_EXPANSION_RESEARCH.md](LLM_SENTIMENT_EXPANSION_RESEARCH.md)
- Background: [LLM_TRADING_RESEARCH.md](LLM_TRADING_RESEARCH.md)

**Implementation**:

- Sentiment Filter: `llm/sentiment_filter.py`
- Guardrails: `llm/agents/llm_guardrails.py`
- PPO Optimizer: `llm/ppo_weight_optimizer.py`

**Tags**: `llm`, `sentiment`, `trading`, `multi-agent`, `ppo`

---

## Evaluation & Testing Upgrades

### Evaluation Framework (No Upgrade Number)

**Status**: ✅ v2.2 Implemented

**Research Documents**:

- Primary: [EVALUATION_FRAMEWORK_RESEARCH.md](EVALUATION_FRAMEWORK_RESEARCH.md)
- Guide: [EVALUATION_UPGRADE_GUIDE.md](EVALUATION_UPGRADE_GUIDE.md)
- Agent Integration: [AUTONOMOUS_AGENT_UPGRADE_GUIDE.md](AUTONOMOUS_AGENT_UPGRADE_GUIDE.md)

**Implementation**: `evaluation/` directory (7 frameworks)

**Tags**: `evaluation`, `testing`, `stockbench`, `classic`, `walk-forward`

---

## QuantConnect Platform Upgrades

### QuantConnect Integration (No Upgrade Number)

**Status**: ✅ Validated

**Research Documents**:

- Phase 2: [PHASE_2_INTEGRATION_RESEARCH.md](PHASE_2_INTEGRATION_RESEARCH.md)
- Phase 3: [PHASE3_ADVANCED_FEATURES_RESEARCH.md](PHASE3_ADVANCED_FEATURES_RESEARCH.md)
- Master: [../quantconnect/MASTER_RESEARCH_REPORT.md](../quantconnect/MASTER_RESEARCH_REPORT.md)

**Summaries**:

- [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)
- [RESEARCH_SUMMARY.md](RESEARCH_SUMMARY.md)

**Tags**: `quantconnect`, `schwab`, `greeks`, `combo-orders`, `options`

---

## Prompt Engineering Upgrades

### Prompt Enhancements (v2.0-v6.1)

**Status**: ✅ v6.1 Implemented

**Research Documents**:

- Primary: [SPECIALIZED_PROMPT_RESEARCH.md](SPECIALIZED_PROMPT_RESEARCH.md)
- Implementation: [PROMPT_ENHANCEMENTS_RESEARCH.md](PROMPT_ENHANCEMENTS_RESEARCH.md)
- Background: [LLM_TRADING_RESEARCH.md](LLM_TRADING_RESEARCH.md)

**Tags**: `prompts`, `agents`, `react`, `multi-agent`

---

## Search by Tag

| Tag | Related Documents |
|-----|-------------------|
| `workflow` | AUTONOMOUS_WORKFLOW_RESEARCH, WORKFLOW_ENHANCEMENT_RESEARCH, WORKFLOW_MANAGEMENT_RESEARCH |
| `llm` | LLM_TRADING_RESEARCH, LLM_SENTIMENT_RESEARCH, PROMPT_ENHANCEMENTS_RESEARCH |
| `evaluation` | EVALUATION_FRAMEWORK_RESEARCH, EVALUATION_UPGRADE_GUIDE, AUTONOMOUS_AGENT_UPGRADE_GUIDE |
| `quantconnect` | PHASE_2_INTEGRATION_RESEARCH, PHASE3_ADVANCED_FEATURES_RESEARCH, MASTER_RESEARCH_REPORT |
| `agents` | AUTONOMOUS_AGENT_UPGRADE_GUIDE, SPECIALIZED_PROMPT_RESEARCH, LLM_TRADING_RESEARCH |
| `testing` | EVALUATION_FRAMEWORK_RESEARCH, WORKFLOW_ENHANCEMENT_RESEARCH |
| `ai-features` | ADVANCED_FEATURES_RESEARCH_DEC2025, UPGRADE-010-ADVANCED-FEATURES |
| `rl` | ADVANCED_FEATURES_RESEARCH_DEC2025 (Phases 3, 18-19) |
| `xai` | ADVANCED_FEATURES_RESEARCH_DEC2025 (Phase 11) |
| `execution` | ADVANCED_FEATURES_RESEARCH_DEC2025 (Phase 5) |
| `risk` | ADVANCED_FEATURES_RESEARCH_DEC2025 (Phase 6) |
| `options` | ADVANCED_FEATURES_RESEARCH_DEC2025, PHASE3_ADVANCED_FEATURES_RESEARCH |
| `overnight` | UPGRADE-011-OVERNIGHT-SESSIONS, autonomous-agents/README |
| `autonomous` | AUTONOMOUS_WORKFLOW_RESEARCH, UPGRADE-011-OVERNIGHT-SESSIONS |
| `hooks` | UPGRADE-011-OVERNIGHT-SESSIONS (Stop, PreCompact hooks) |
| `crash-recovery` | UPGRADE-011-OVERNIGHT-SESSIONS (auto-resume, jitter) |

---

## How to Use This Index

### Finding Research for an Upgrade

1. Search this document for the upgrade number (e.g., `UPGRADE-014`)
2. Follow the "Primary Research" link for main documentation
3. Check "Related" links for additional context

### Adding a New Upgrade

1. Create research document with subject-based name: `[TOPIC]_RESEARCH.md`
2. Add frontmatter with `related_upgrades: [UPGRADE-NNN]`
3. Add entry to this index under appropriate section
4. Create implementation checklist if needed: `UPGRADE-NNN-[TOPIC].md`

### Searching by Topic

1. Use the "Search by Tag" table above
2. Or search the main [README.md](README.md) quick reference

---

## Related Documents

- [NAMING_CONVENTION.md](NAMING_CONVENTION.md) - Naming rules
- [README.md](README.md) - Full research index
- [../development/ENHANCED_RIC_WORKFLOW.md](../development/ENHANCED_RIC_WORKFLOW.md) - RIC integration

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-03 | Added UPGRADE-010 Sprint 6 (180 tests, 3 modules >90% coverage) |
| 2025-12-03 | Added UPGRADE-011 Enhanced Overnight Sessions (continuous mode, relay-race, hooks) |
| 2025-12-02 | Initial upgrade index created |
| 2025-12-02 | Added UPGRADE-010 Advanced AI Features (98 features, 30 research phases) |
