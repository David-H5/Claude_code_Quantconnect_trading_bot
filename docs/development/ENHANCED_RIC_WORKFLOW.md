# RIC Loop Workflow (v5.1 Guardian)

> **Current Implementation**: `.claude/hooks/ric_v50.py`
>
> **Quick Reference**: `.claude/RIC_CONTEXT.md`
>
> This document provides detailed workflow documentation for the RIC v5.1 system.

**Version**: 5.1 Guardian
**Updated**: December 5, 2025
**Research**: [RIC-V46-RESEARCH-DEC2025.md](../research/RIC-V46-RESEARCH-DEC2025.md) (35+ sources)

---

## Executive Summary

The RIC Loop v5.1 "Guardian" is a 5-phase workflow for complex tasks with:

1. **STRICT SEQUENTIAL EXECUTION** - Phases P0→P1→P2→P3→P4, NO SKIPPING
2. **Research Phase (P0)** - Online research before planning, IMMEDIATELY write to file
3. **Insight Classification** - P0/P1/P2/[OUT] priority levels
4. **ALL Loops Go to P0** - Research first, then implement
5. **Minimum 3 Iterations** - Ensures thoroughness before exit
6. **P2 is REQUIRED** - Cannot defer P2 items

---

## RIC Loop v5.1 Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              RIC LOOP v5.1 (5 Phases)                               │
│                         STRICT SEQUENTIAL EXECUTION                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   [I{iter}/{max}] P0 → P1 → P2 → P3 → P4 → Loop/Exit                               │
│                    ↓     ↓     ↓     ↓     ↓                                        │
│                RESEARCH PLAN BUILD VERIFY REFLECT                                    │
│                                                                                      │
│   ┌──────────────────────────────────────────────────────────────────────────────┐  │
│   │ P0/P1/P2 REMAIN?     │                        │ ALL RESOLVED +               │  │
│   │ OR iteration < 3?    │                        │ iteration >= 3?              │  │
│   │                      │                        │                              │  │
│   │ → LOOP to Phase 0    │                        │ → EXIT ALLOWED               │  │
│   └──────────────────────┘                        └──────────────────────────────┘  │
│                                                                                      │
│   Iteration: [1] [2] [3] [4] [5]   ← **Min 3**, Max 5, ALL P0-P2 REQUIRED          │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Details

### Phase 0: RESEARCH

**Purpose**: Frame problem measurably, conduct online research

**Actions**:

- WebSearch for current best practices
- Persist findings to `docs/research/` every 3 searches
- Timestamp all sources

**Gate Criteria**:

- 3+ timestamped web searches
- Findings documented to file

**CLI Commands**:

```bash
python3 .claude/hooks/ric_v50.py record-keywords "kw1,kw2,kw3"
python3 .claude/hooks/ric_v50.py record-search "query"
python3 .claude/hooks/ric_v50.py record-findings
python3 .claude/hooks/ric_v50.py p0-status
```

### Phase 1: PLAN

**Purpose**: Define scope, success criteria, task breakdown

**Actions**:

- Classify tasks as P0/P1/P2/[OUT]
- Define success criteria
- Create implementation plan

**Gate Criteria**:

- All tasks prioritized
- [OUT] scope defined

### Phase 2: BUILD

**Purpose**: Implementation with atomic commits

**Actions**:

- Single-component changes (1-5 files per commit)
- Hallucination check before commits
- Tests with each change

**Gate Criteria**:

- Atomic commits
- No batched multi-component changes

### Phase 3: VERIFY

**Purpose**: Quality gates validation

**Actions**:

- Run tests
- Check coverage (>70%)
- Run linting
- Security check

**Gate Criteria**:

- All tests pass
- Coverage >70%
- No security issues

**CLI Commands**:

```bash
python3 .claude/hooks/ric_v50.py check-gate 3
python3 .claude/hooks/ric_v50.py security
```

### Phase 4: REFLECT

**Purpose**: Gap analysis and loop decision

**Actions**:

- Introspection on completed work
- Classify remaining gaps as P0/P1/P2
- Decide: LOOP or EXIT

**Gate Criteria**:

- Introspection completed
- All insights classified
- Loop decision made

**CLI Commands**:

```bash
python3 .claude/hooks/ric_v50.py record-introspection
python3 .claude/hooks/ric_v50.py upgrade-ideas "idea1,idea2"
python3 .claude/hooks/ric_v50.py loop-decision LOOP|EXIT "reason"
python3 .claude/hooks/ric_v50.py p4-status
```

---

## Priority Levels

| Level | Name | Definition | RIC Rule |
|-------|------|------------|----------|
| **[P0]** | Critical | Would block functionality | Must resolve to exit |
| **[P1]** | Important | Affects quality | Must resolve to exit |
| **[P2]** | Polish | Improves UX | **REQUIRED** (not optional) |
| **[OUT]** | Out of Scope | Explicitly excluded | Prevents scope creep |

---

## Loop Decision Rules (Phase 4)

```text
1. IF iteration < 3         → LOOP (minimum 3 iterations)
2. ELIF P0 insights > 0     → LOOP (critical gaps)
3. ELIF P1 insights > 0     → LOOP (important items)
4. ELIF P2 insights > 0     → LOOP (P2 is REQUIRED)
5. ELIF iteration >= 5      → FORCED EXIT (max reached)
6. ELSE                     → EXIT ALLOWED
```

---

## v5.1 Safety Features

| Feature | Description | Source |
|---------|-------------|--------|
| **Hallucination Detection** | 5-category taxonomy check | Research 2025 |
| **Convergence Detection** | Multi-metric tracking | SAGE Paper |
| **Confidence Calibration** | Per-phase 0-100% ratings | MetaQA |
| **Safety Throttles** | Tool/time/failure limits | AEGIS Framework |
| **Decision Tracing** | Structured JSON logs | Anthropic 2025 |
| **Drift Detection** | Scope creep >20% alerts | AEGIS Framework |
| **Guardian Mode** | Independent verification | Gartner 2025 |
| **SEIDR Debug Loop** | Multi-candidate fixes | ACM TELO 2025 |

---

## CLI Reference

```bash
# Session Management
python3 .claude/hooks/ric_v50.py init 5          # Start session
python3 .claude/hooks/ric_v50.py status          # Current status
python3 .claude/hooks/ric_v50.py advance         # Next phase
python3 .claude/hooks/ric_v50.py can-exit        # Exit check
python3 .claude/hooks/ric_v50.py end             # End session

# Insights
python3 .claude/hooks/ric_v50.py add-insight P0 "Description"
python3 .claude/hooks/ric_v50.py resolve INS-001 "Resolution"
python3 .claude/hooks/ric_v50.py insights

# Quality & Safety
python3 .claude/hooks/ric_v50.py check-gate 0    # Gate criteria
python3 .claude/hooks/ric_v50.py convergence     # Convergence status
python3 .claude/hooks/ric_v50.py throttles       # Throttle status
python3 .claude/hooks/ric_v50.py security        # Security check

# Monitoring
python3 .claude/hooks/ric_v50.py json            # JSON output
python3 .claude/hooks/ric_v50.py sync            # Sync progress
python3 .claude/hooks/ric_v50.py summary         # Iteration summary
python3 .claude/hooks/ric_v50.py help            # All commands
```

---

## Related Documentation

- [RIC Quick Reference](../../.claude/RIC_CONTEXT.md) - Compact guide
- [RIC Enforcement](RIC_LOOP_ENFORCEMENT.md) - System details
- [RIC v5.0 Research](../research/RIC-V46-RESEARCH-DEC2025.md) - Source research
- [RIC v5.1 Quality Gates](../research/RIC-V51-QUALITY-GATES-RESEARCH.md) - Latest
