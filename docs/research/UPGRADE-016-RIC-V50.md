# UPGRADE-016: RIC v5.0 Complete Implementation

**Created**: December 4, 2025
**Status**: In Progress
**Goal**: Create fully standalone ric_v50.py with all v4.5 code plus new features

---

## Research Phase (P0)

### v4.5 Architecture Analysis

**File**: `.claude/hooks/ric_v45.py`
**Size**: 4,419 lines

#### Classes (8 total)

| Class | Line | Purpose |
|-------|------|---------|
| `IterationMetrics` | 1405 | Convergence tracking per iteration |
| `ThrottleState` | 1547 | Safety throttle management |
| `ConfidenceRecord` | 2786 | Confidence calibration per phase |
| `Phase` | 3059 | Enum for P0-P4 phases |
| `Priority` | 3069 | Enum for P0/P1/P2 priorities |
| `Insight` | 3078 | Individual insight/gap tracking |
| `RICState` | 3091 | Full session state dataclass |
| `RICStateManager` | 3112 | State persistence and management |

#### Major Constants/Configs

| Constant | Line | Purpose |
|----------|------|---------|
| `RIC_VERSION` | 57 | Version string |
| `PHASES` | 63 | Phase definitions |
| `ITERATION_LIMITS` | 75 | Min/max iterations |
| `FIX_LIMITS` | 85 | Fix attempt limits |
| `SAFETY_THROTTLES` | 98 | Safety limits |
| `RESEARCH_ENFORCEMENT` | 111 | Research rules |
| `TIMESTAMP_PATTERNS` | 138 | Regex for timestamps |
| `DOC_ENFORCEMENT` | 293 | Documentation rules |
| `CONVERGENCE_THRESHOLDS` | 451 | Convergence criteria |
| `HALLUCINATION_CATEGORIES` | 463 | 5-category taxonomy |
| `GATES` | 1268 | Phase gate criteria |

#### Prompts (~15 major prompts)

| Prompt | Line | Purpose |
|--------|------|---------|
| `RESEARCH_AUTOPERSIST_NOTICE` | 160 | Auto-persist warning |
| `RESEARCH_PERSIST_WARNING` | 181 | Persist reminder |
| `RESEARCH_TIMESTAMP_WARNING` | 210 | Timestamp warning |
| `DOC_NAMING_WARNING` | 367 | Doc naming warning |
| `DOC_COMPLETION_BLOCK_WARNING` | 385 | Completion block |
| `DOC_SESSION_START_PROMPT` | 406 | Session start |
| `DOC_STALENESS_WARNING` | 428 | Stale doc warning |
| `CONTEXT_MANAGEMENT_PROMPT` | 984 | Context tips |
| `DECISION_TRACE_FORMAT` | 1030 | Decision logging |
| `AUTONOMOUS_START` | 1069 | Autonomous start |
| `AUTONOMOUS_STATUS` | 1105 | Status template |
| `TEST_FAILURE_PROMPT` | 1121 | Test failure guide |
| `HALLUCINATION_FIX_PROMPT` | 1173 | Hallucination fix |
| `STUCK_PROMPT` | 1204 | Stuck guidance |
| `CHECKPOINT_PROMPT` | 1229 | Checkpoint guide |
| `THROTTLE_WARNING_PROMPT` | 1245 | Throttle warning |

#### Functions (88 total)

**Core Functions (35)**:
- Phase management: `get_header`, `get_commit_example`, `get_phase_prompt`
- Gate checking: `check_gate`, `check_throttles`
- Convergence: `calculate_convergence`, `check_hallucination`
- Research: `record_web_search`, `record_research_persist`, `auto_persist_research`
- Documentation: `validate_doc_naming`, `validate_doc_sections`, `validate_cross_references`
- Security: `check_for_secrets`, `security_gate_check`

**CLI Functions (19)**:
| Function | Purpose |
|----------|---------|
| `cli_init` | Initialize session |
| `cli_status` | Show status |
| `cli_advance` | Advance phase |
| `cli_add_insight` | Add insight |
| `cli_confidence` | Set confidence |
| `cli_decision` | Log decision |
| `cli_convergence` | Show convergence |
| `cli_can_exit` | Check exit |
| `cli_end` | End session |
| `cli_json` | JSON output |
| `cli_sync_progress` | Sync progress file |
| `cli_check_gate` | Check phase gate |
| `cli_security_check` | Security check |
| `cli_summary` | Show summary |
| `cli_resolve` | Resolve insight |
| `cli_insights` | List insights |
| `cli_throttles` | Show throttles |
| `cli_research_status` | Research status |
| `cli_decisions` | Show decisions |

**Hook Handlers (3)**:
- `handle_pretool_use` - PreToolUse hook
- `handle_posttool_use` - PostToolUse hook
- `handle_user_prompt` - UserPromptSubmit hook

---

### v5.0 New Features (from ric_v50_dev.py)

| Feature | Source | Lines | Status |
|---------|--------|-------|--------|
| Drift Detection | AEGIS Framework (Forrester 2025) | ~150 | Ready |
| Guardian Mode | Gartner 2025 | ~150 | Ready |
| Structured Memory | Anthropic 2025 | ~200 | Ready |
| Candidate Ranking | SEIDR Lexicase | ~100 | Ready |
| Replace/Repair Tracking | SEIDR Paper | ~50 | Ready |

#### New Classes

| Class | Purpose |
|-------|---------|
| `DriftMetrics` | Track scope drift |
| `GuardianReview` | Guardian review results |
| `FixCandidate` | Fix candidate for ranking |
| `RepairStats` | Replace/repair tracking |

#### New CLI Commands

| Command | Purpose |
|---------|---------|
| `drift` | Check scope drift |
| `guardian` | Run guardian review |
| `notes` | View RIC_NOTES.md |
| `features` | List enabled features |

---

### Integration Plan

**Approach**: Create fully standalone ric_v50.py

1. Copy ALL v4.5 code (4,419 lines)
2. Add new feature sections (~650 lines)
3. Extend RICState with new fields
4. Add new CLI commands
5. Update version to 5.0

**Estimated Size**: ~5,100 lines

**File Structure**:
```
ric_v50.py
├── Header & Version (50 lines)
├── Configuration & Constants (400 lines)
├── Prompts & Templates (900 lines)
├── Core Functions (1,200 lines)
├── Documentation Functions (800 lines)
├── Classes (400 lines)
├── Hook Handlers (400 lines)
├── CLI Commands (600 lines)
├── NEW: Drift Detection (150 lines)
├── NEW: Guardian Mode (150 lines)
├── NEW: Structured Memory (200 lines)
├── NEW: Candidate Ranking (100 lines)
├── NEW: Replace/Repair Tracking (50 lines)
├── Main & Entry Point (50 lines)
└── Total: ~5,500 lines
```

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-04 | Initial research completed |
