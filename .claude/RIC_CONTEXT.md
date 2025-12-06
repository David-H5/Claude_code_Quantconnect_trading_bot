# RIC Loop v5.1 "Guardian" Quick Reference

<!--
╔══════════════════════════════════════════════════════════════════════════════╗
║  CORE RIC FILE - v5.1 Guardian                                               ║
║                                                                              ║
║  Main implementation: .claude/hooks/core/ric.py                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
-->

**When to Use**: Multi-file changes, research needed, architecture decisions, complex bugs, upgrades

## 5-Phase Workflow (v5.1)

```
[I{iter}/{max}] P0 → P1 → P2 → P3 → P4 → Loop/Exit
               ↓     ↓     ↓     ↓     ↓
           RESEARCH PLAN BUILD VERIFY REFLECT
```

| Phase | Name | Key Action | Gate |
|-------|------|------------|------|
| **P0** | Research | WebSearch + persist every 3 | 3+ timestamped sources |
| **P1** | Plan | P0/P1/P2/[OUT] tasks | All tasks prioritized |
| **P2** | Build | Atomic commits (1-5 files) | Tests with each change |
| **P3** | Verify | Tests → Coverage → Lint | All pass, >70% coverage |
| **P4** | Reflect | Classify gaps → Decide | Loop decision made |

## Critical Rules (v5.1)

1. **STRICT SEQUENTIAL**: Execute P0→P1→P2→P3→P4 in order, NO SKIPPING
2. **LOG EVERY PHASE**: Format: `[I{iter}/{max}][P{phase}] PHASE_NAME`
3. **Minimum 3 Iterations**: Cannot exit before 3 full loops
4. **Complete ALL P0-P2**: P2 is REQUIRED, not optional
5. **ALL loops go to P0**: Research first, then implement
6. **Persist Research**: Save to docs/research/ every 3 searches

## Priority Levels

| Level | Name | Definition | RIC Rule |
|-------|------|------------|----------|
| **[P0]** | Critical | Would hold release | Must resolve to exit |
| **[P1]** | Important | Should complete | Must resolve to exit |
| **[P2]** | Polish | Quality improvement | **REQUIRED** (not optional) |
| **[OUT]** | Out of Scope | Explicitly excluded | Prevents scope creep |

## Loop Decision (Phase 4) - v5.1

```text
1. IF iteration < 3         → LOOP (minimum 3 iterations)
2. ELIF P0 insights > 0     → LOOP (critical gaps)
3. ELIF P1 insights > 0     → LOOP (important items)
4. ELIF P2 insights > 0     → LOOP (P2 is REQUIRED)
5. ELIF iteration >= 5      → FORCED EXIT (max reached)
6. ELSE                     → EXIT ALLOWED
```

## v5.1 Safety Features

| Feature | Description |
|---------|-------------|
| **Hallucination Detection** | 5-category taxonomy check before commits |
| **Convergence Detection** | Multi-metric tracking (insight rate, fix success, code churn) |
| **Confidence Calibration** | Per-phase 0-100% confidence ratings |
| **Safety Throttles** | Tool call limits, time limits, failure limits |
| **Decision Tracing** | Structured JSON logs for meta-debugging |
| **Research Enforcement** | Timestamp validation, auto-persist every 3 searches |

## CLI Commands

```bash
# Initialize
python3 .claude/hooks/core/ric.py init 5          # Start session (5 iterations)

# Status & Navigation
python3 .claude/hooks/core/ric.py status          # Show current status
python3 .claude/hooks/core/ric.py advance         # Advance to next phase (enforced)
python3 .claude/hooks/core/ric.py advance --force # Force advance (override)
python3 .claude/hooks/core/ric.py can-exit        # Check if can exit

# Phase Enforcement (v5.1 NEW)
python3 .claude/hooks/core/ric.py p0-status       # P0 RESEARCH requirements
python3 .claude/hooks/core/ric.py p4-status       # P4 REFLECT requirements
python3 .claude/hooks/core/ric.py record-keywords "kw1,kw2,kw3"
python3 .claude/hooks/core/ric.py record-search "query"
python3 .claude/hooks/core/ric.py record-findings
python3 .claude/hooks/core/ric.py record-introspection
python3 .claude/hooks/core/ric.py upgrade-ideas "idea1,idea2"
python3 .claude/hooks/core/ric.py loop-decision LOOP|EXIT "reason"

# Insights
python3 .claude/hooks/core/ric.py add-insight P0 "Description"
python3 .claude/hooks/core/ric.py resolve INS-001 "Resolution"
python3 .claude/hooks/core/ric.py insights        # List all

# Quality & Safety
python3 .claude/hooks/core/ric.py check-gate 0    # Check gate criteria
python3 .claude/hooks/core/ric.py convergence     # Convergence status
python3 .claude/hooks/core/ric.py throttles       # Throttle status
python3 .claude/hooks/core/ric.py security        # Security check

# v5.1 Features
python3 .claude/hooks/core/ric.py drift           # Check scope drift
python3 .claude/hooks/core/ric.py guardian        # Run guardian review
python3 .claude/hooks/core/ric.py features        # Show v5.1 features
python3 .claude/hooks/core/ric.py v50-status      # Comprehensive status

# Machine-Parseable
python3 .claude/hooks/core/ric.py json            # JSON output
python3 .claude/hooks/core/ric.py help            # All commands
```

## Commit Format

```bash
git commit -m "[I2/5][P2] Add user validation with tests"
```

- Use `[I{iter}/{max}][P2]` prefix (P2 = Build phase)
- Pass **AND-test**: Can describe without "AND"
- 1-5 related files per commit
- Code + test together is OK

## Iteration Limits

| Limit | Value |
|-------|-------|
| Minimum | 3 iterations |
| Maximum | 5 iterations |
| Plateau | 2 consecutive loops with no insights |

## Enforcement Modes

```bash
export RIC_MODE=SUGGESTED   # Default - suggest for complex tasks
export RIC_MODE=ENFORCED    # Inject mandatory actions
export RIC_MODE=DISABLED    # No RIC
```

## Key Files

| File | Purpose |
|------|---------|
| `.claude/hooks/core/ric.py` | Main RIC v5.1 Guardian (~6700 lines) |
| `.claude/RIC_CONTEXT.md` | This quick reference |
| `.claude/state/ric.json` | Session state (auto-managed) |
| `docs/research/` | Research documents |
| `claude-progress.txt` | Task tracking |

## v5.1 Phase Enforcement (NEW)

**P0 RESEARCH** blocks advancement until:
- 3+ keywords extracted
- 3+ web searches completed
- 3+ sources documented
- Findings persisted to docs/research/

**P4 REFLECT** blocks advancement until:
- Introspection completed
- Insights reviewed
- Convergence checked
- Upgrade ideas generated (1+)
- LOOP/EXIT decision made with reason
