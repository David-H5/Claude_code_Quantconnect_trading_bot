# UPGRADE-012.3: Meta-RIC Loop with Insight-Driven Iteration

> **⚠️ RESEARCH DOCUMENT - DO NOT EDIT FOR PROMPT CHANGES**
>
> To change RIC prompts/rules, edit ONLY:
>
> - `.claude/hooks/ric_prompts.py` - Prompt text, warnings, templates
> - `.claude/RIC_CONTEXT.md` - Workflow rules, phase definitions

## Research Overview

**Date**: December 3, 2025
**Scope**: Replace score-based convergence with insight-driven min-max iteration loop
**Status**: ✅ Complete (v3.0)
**Version**: v3.0 - 7 phases, min 3 iterations, complete ALL P0-P2, strict sequential, ALL loops to Phase 0
**Parent**: [UPGRADE-012 Hierarchical Prompts](UPGRADE-012-HIERARCHICAL-PROMPTS.md)
**Predecessor**: [UPGRADE-012.2 ACE Reflector](UPGRADE-012.2-ACE-REFLECTOR-SEMANTIC-ROUTING.md)

---

## Phase 0: Research Summary

**Search Date**: December 3, 2025 at ~11:30 AM EST

### Research Topic 1: Iterative Refinement Patterns

**Search Query**: "iterative refinement loop self-improvement AI agent plateau detection minimum iterations 2025"

**Key Sources**:
1. [Self-Refine: Iterative Refinement with Self-Feedback (Published: Mar 2023, Updated 2024)](https://arxiv.org/abs/2303.17651)
2. [Multi-AI Agent System for Autonomous Optimization (Published: Dec 2024)](https://arxiv.org/html/2412.17149)
3. [Google ADK Loop Agents Documentation (Published: 2025)](https://google.github.io/adk-docs/agents/workflow-agents/loop-agents/)

**Key Findings**:
- Self-Refine: Same LLM provides feedback and refines iteratively
- Default max iterations: 5 (configurable)
- `is_refinement_sufficient` defines task-dependent stopping criteria
- "STOP" signal from Critic Agent when quality satisfied
- Plateau detection is uncommon - quality thresholds preferred

### Research Topic 2: Meta-Cognitive Action Planning

**Search Query**: "meta-cognition action planning AI agent self-reflection insight aggregation 2025"

**Key Sources**:
1. [MetaAgent: Self-Evolving via Tool Meta-Learning (Published: Aug 2025)](https://arxiv.org/html/2508.00271v2)
2. [ICML 2025 Position: Intrinsic Metacognitive Learning (Published: 2025)](https://openreview.net/forum?id=4KhDd0Ozqe)
3. [Microsoft AI Agents: Metacognition (Published: 2025)](https://microsoft.github.io/ai-agents-for-beginners/09-metacognition/)
4. [Metagent-P: Neuro-Symbolic Planning (Published: 2025)](https://aclanthology.org/2025.findings-acl.1169.pdf)

**Key Findings**:

**MetaAgent Pattern** (Most Relevant):
- After each task: active self-reflection and answer verification
- Reviews accuracy, reasoning patterns, tool selection effectiveness
- **Abstracts broadly applicable insights from each experience**
- Insights dynamically incorporated into subsequent contexts

**ICML 2025 Framework**:
- Three components: metacognitive knowledge, planning, evaluation
- Agents reflect on what they know, how they learn, how well strategies work

**Metagent-P Framework**:
- "Planning-verification-execution-**reflection**" cycle
- Four components: Planner, Verifier, Controller, Reflector

---

## Problem Statement

**Current State (RIC Loop v1.0)**:
- Phase 6 (Metacognition) generates valuable insights about gaps
- Phase 8 (Convergence) calculates a score and exits if >= 0.80
- Insights from Phase 6 are documented but NOT acted upon
- Score-based exit allows skipping missing features and necessary updates

**Example of the Problem** (from UPGRADE-012.2):
```
Phase 6 identified:
- "Keyword-only classification misses complex tasks like OAuth2 auth"
- "Unit tests still pending (P1)"
- "Would add more L1_complex keyword patterns"

Phase 8 calculated:
- Score: 90.7% >= 80% threshold
- Decision: EXIT

Result: Gaps were documented but not fixed
```

**Target State (Meta-RIC Loop v3.0)**:
- **STRICT SEQUENTIAL EXECUTION** - Phases 0→1→2→3→4→5→6→7, NO SKIPPING
- Phase 6 insights are **classified and triaged** into actionable categories
- Phase 7 becomes **Integration Phase** that generates incremental update plan (7 phases total)
- **Min-max iteration loop** ensures thorough completion (min 3, max 5)
- **Complete ALL P0-P2**: AI assistance makes this feasible - P2 is REQUIRED, not optional
- **ALL loops go to Phase 0**: Research first, then implement
- **Plateau detection** stops when no new insights emerge for 2 consecutive loops
- **Compaction Protection**: IMMEDIATELY write research to file after EVERY search

---

## Phase 1: Upgrade Path

### Success Criteria

1. [x] Phase 6 insights are classified into P0/P1/P2 action categories
2. [x] Phase 7 generates incremental update plan from Phase 6 insights (7 phases total)
3. [x] Min-max loop structure enforced (min 2, max 5 iterations)
4. [x] Plateau detection stops loop when no new insights for 2 consecutive iterations
5. [x] ALL P0-P2 insights must be addressed (AI makes this feasible)
6. [x] If no insights & below minimum → Loop to Phase 0 (Research)
7. [x] Documentation updated with new workflow (L1_complex.md, CLAUDE.md)

### Scope

**In Scope**:
- Redesign Phase 6 to output classified insights
- Replace Phase 8 (Convergence) with Phase 7 (Integration) - 7 phases total
- Add min-max iteration tracking (min 3, max 5)
- Add research fallback when no insights & below minimum
- Add plateau detection logic (2 consecutive iterations)
- Update L1_complex.md prompt with new workflow
- Update CLAUDE.md with new RIC loop

**Out of Scope**:
- Automated insight classification (manual for v2.1)
- Integration with ACE Curator (defer to v2.2)
- Multi-agent debate on insights (defer to v2.2)

---

## Design: Meta-RIC Loop v3.0

### Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              META-RIC LOOP v3.0                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │ 0. RESEARCH │────▶│ 1. UPGRADE  │────▶│ 2. CHECKLIST│────▶│ 3. CODING   │       │
│   │             │     │    PATH     │     │             │     │  (1 at a    │       │
│   └──────▲──────┘     └─────────────┘     └─────────────┘     │   time)     │       │
│          │                                                     └──────┬──────┘       │
│          │ No insights?                                               │              │
│          │ Research more!    ┌────────────────────────────────────────┘              │
│          │                   │                                                       │
│          │                   ▼                                                       │
│          │           ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│          │           │ 4. DOUBLE-  │────▶│ 5. INTRO-   │────▶│ 6. META-    │        │
│          │           │   CHECK     │     │   SPECTION  │     │   COGNITION │        │
│          │           └─────────────┘     └─────────────┘     └──────┬──────┘        │
│          │                                                          │               │
│          │                                          ┌───────────────┴───────────┐   │
│          │                                          │                           │   │
│          │                                          ▼                           ▼   │
│          │                                  ┌───────────────┐          ┌────────────┐│
│          └──────────────────────────────────│ NO INSIGHTS & │          │ 7. INTEGRATE│
│                                             │ iter < min    │          │  & DECIDE  ││
│     ┌───────────────────────────────────────└───────────────┘          └─────┬──────┘│
│     │                                                                        │       │
│     │    ┌──────────────────────────────────────────────────────────────────┤       │
│     │    │                              │                                    │       │
│     │    ▼                              ▼                                    ▼       │
│     │ ┌──────────────┐          ┌──────────────┐                   ┌──────────────┐ │
│     │ │ P0/P1/P2     │          │ PLATEAU      │                   │ ALL DONE     │ │
│     └─│ Loop → Ph 0  │          │ EXIT         │                   │ EXIT         │ │
│       └──────────────┘          └──────────────┘                   └──────────────┘ │
│                                                                                      │
│   Iteration: [1] [2] [3] [4] [5]   ← Min 3, Max 5, complete ALL P0-P2               │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Phase 6: Enhanced Metacognition

**Current**: 5 self-reflection questions with confidence scores

**New**: 5 questions + **Classified Insight Output**

```markdown
## Phase 6: Metacognition

### Self-Reflection Questions
1. Did implementation achieve research goals? [Confidence: X%]
2. Are there known gaps? [Confidence: X%]
3. What assumptions need validation? [Confidence: X%]
4. What would I do differently? [Confidence: X%]
5. Is the implementation maintainable? [Confidence: X%]

### Classified Insights (NEW)

| ID | Insight | Classification | Action Required |
|----|---------|----------------|-----------------|
| I1 | Keyword patterns miss complex tasks | P0-GAP | Add more patterns |
| I2 | Unit tests pending | P1-DEBT | Create tests |
| I3 | Could add batch mode | P2-ENHANCEMENT | Future work |
| I4 | Documentation complete | ADDRESSED | None |

### Classification Legend
- **P0-GAP**: Critical gap that blocks completion
- **P0-BUG**: Bug that must be fixed
- **P1-DEBT**: Technical debt that should be addressed
- **P1-IMPROVEMENT**: Improvement worth making
- **P2-ENHANCEMENT**: Nice to have
- **ADDRESSED**: Already handled
- **DEFERRED**: Explicitly skipped with justification
```

### Phase 7: Integration & Decision (Replaces Convergence)

**Current**: Score calculation → Exit if >= 0.80

**New**: Insight triage → Incremental plan → Loop or exit decision

```markdown
## Phase 7: Integration & Decision

### Iteration Status
- Current iteration: [N]
- Previous iterations: [N-1] insights addressed
- New insights this iteration: [count]

### Insight Triage

**P0 Insights (Must Address)**:
| ID | Insight | Plan | Estimated Effort |
|----|---------|------|------------------|
| I1 | [from Phase 6] | [specific fix] | [30min/1hr/2hr] |

**P1 Insights (Should Address)**:
| ID | Insight | Plan | Priority for Next Iteration |
|----|---------|------|-----------------------------|

**P2/Deferred (Document Only)**:
| ID | Insight | Justification for Deferral |
|----|---------|---------------------------|

### Loop Decision (Updated v3.0)

```text
IF iteration < 3 (minimum):
    → MANDATORY LOOP to Phase 0 (Research)
    → Do more research before continuing

ELIF P0 OR P1 OR P2 insights exist:
    → LOOP BACK to Phase 0 (Research)
    → Research solutions, then complete ALL insights

ELIF no new insights for 2 consecutive iterations AND iteration >= 3:
    → PLATEAU DETECTED: EXIT

ELIF iteration >= 5:
    → MAX ITERATIONS: EXIT (document remaining work)

ELSE:
    → EXIT: All work complete, upgrade is polished
```

### Exit Finalization Process (v2.1 Addition)

**REQUIRED before exit - thorough completion checklist:**

| Category | Checklist Items |
|----------|-----------------|
| **4a. Bug Finding** | Run tests, check syntax, review TODOs, remove debug code |
| **4b. Missing Files** | Verify files exist, imports work, dependencies in requirements.txt |
| **4c. Test Coverage** | Tests for new code, edge cases covered, >70% coverage |
| **4d. Documentation** | Upgrade doc complete, CLAUDE.md updated, docstrings added |
| **4e. Cross-Reference** | Related files updated, configs match, no stale references |
| **4f. Cleanup** | Git status clean, no temp files, branch ready for merge |
| **4g. Integration** | Scope-appropriate integration check (see below) |

**Integration Check Tiers (v2.2 Addition)**:

| Scope | Check Type | What to Verify |
|-------|-----------|----------------|
| **SUB-UPGRADE** (UPGRADE-XXX.Y) | Basic | Changed feature + immediate deps, affected tests, no import errors |
| **MAJOR UPGRADE** (UPGRADE-XXX) | Complete | Full test suite, cross-component, end-to-end, architecture alignment |

**This ensures each upgrade is production-ready before moving to the next.**

### Exit Report

**Completion Status**:

- Total iterations: [N]
- P0 insights: [X addressed]
- P1 insights: [X addressed]
- P2 insights: [X addressed]
- Exit reason: [COMPLETE | PLATEAU | MAX_ITER]

**Finalization Checklist**:

- Bug finding: [PASS/FAIL]
- Missing files check: [PASS/FAIL]
- Test coverage: [PASS/FAIL - X%]
- Documentation: [PASS/FAIL]
- Cross-references: [PASS/FAIL]
- Cleanup: [PASS/FAIL]
- Integration: [PASS/FAIL - BASIC|COMPLETE]

**Files Modified/Created**:
[List all files touched in this upgrade]

**Next Steps (if any)**:
[Deferred items for future work]
```

### Min-Max Iteration Rules

| Rule | Value | Rationale |
|------|-------|-----------|
| Minimum iterations | 3 | Ensures polished output |
| Maximum iterations | 5 | From Self-Refine research (default max) |
| Plateau threshold | 2 | No new insights for 2 consecutive loops |
| Complete ALL P0-P2 | Yes | AI assistance makes this feasible, P2 is REQUIRED |
| ALL loops → Phase 0 | Yes | Research first, then implement |
| Strict Sequential | Yes | Phases 0→1→2→3→4→5→6→7, NO SKIPPING |
| Compaction Protection | Yes | IMMEDIATELY write research to file |

### Iteration State Tracking

```json
{
  "iteration": 3,
  "insights_history": [
    {"iteration": 1, "p0": 2, "p1": 3, "p2": 1, "addressed": 0},
    {"iteration": 2, "p0": 0, "p1": 2, "p2": 1, "addressed": 3},
    {"iteration": 3, "p0": 0, "p1": 0, "p2": 2, "addressed": 2}
  ],
  "plateau_counter": 2,
  "exit_decision": "PLATEAU"
}
```

---

## Phase 2: Implementation Checklist

### P0 - Critical

- [x] Update Phase 6 template in L1_complex.md with classified insights
- [x] Replace Phase 8 (Convergence) with Phase 7 (Integration) in L1_complex.md
- [x] Add iteration tracking structure to RIC loop templates
- [x] Update CLAUDE.md with new Meta-RIC Loop v2.1 diagram

### P1 - Important

- [x] Create `/ric-integrate` slash command for Phase 7
- [x] Update `/ric-converge` to redirect to new behavior
- [x] Add plateau detection to session notes template
- [x] Create insight classification guide (in L1_complex.md)

### P2 - Nice to Have

- [ ] Add iteration visualization to session notes
- [ ] Create RIC loop state persistence across sessions
- [ ] Integrate with ACE Reflector for insight learning

---

## Files to Modify/Create

| File | Action | Status | Purpose |
|------|--------|--------|---------|
| `prompts/complexity/L1_complex.md` | MODIFY | ✅ DONE | New Phase 6 + Phase 7 templates (v2.1) |
| `CLAUDE.md` | MODIFY | ✅ DONE | Update RIC loop documentation (v2.1) |
| `.claude/commands/ric-integrate.md` | CREATE | ✅ EXISTS | Phase 7 slash command |
| `.claude/commands/ric-converge.md` | MODIFY | ✅ EXISTS | Redirect to new behavior |
| `config/task-router.yaml` | MODIFY | ✅ DONE | Enhanced RIC loop triggers (v1.2.0) |
| `docs/development/ENHANCED_RIC_WORKFLOW.md` | MODIFY | ⏳ PENDING | Full workflow documentation |

### Enhanced RIC Loop Triggers (v1.2.0)

The task-router.yaml now includes robust triggers to force RIC loop usage:

| Trigger Category | Patterns | Weight |
|------------------|----------|--------|
| **Explicit RIC requests** | `ric loop`, `ric workflow`, `ric loop workflow` | 5 |
| **Meta-RIC references** | `meta-ric`, `enhanced ric`, `full ric`, `use ric` | 5 |
| **Slash commands** | `/ric`, `ric-start`, `ric-research`, `ric-introspect`, `ric-converge` | 5 |
| **UPGRADE-XXX references** | `UPGRADE-012`, `upgrade-010`, etc. | 4 |
| **Upgrade keywords** | `upgrade`, `sub-upgrade`, `major upgrade` | 3 |
| **Comprehensive work** | `comprehensive`, `systematic`, `thorough implement` | 3 |
| **Full workflow** | `full workflow`, `complete workflow`, `end-to-end implement` | 3 |
| **Research-first** | `research first`, `investigate before`, `explore then implement` | 3 |
| **Multi-phase work** | `multi-phase`, `multi-iteration`, `iterative refin` | 3 |

**Threshold**: Score >= 5 triggers L1_complex (RIC loop required).

---

## Example: Applying Meta-RIC to UPGRADE-012.2

**What should have happened**:

### Iteration 1 (Original)

**Phase 6 Classified Insights**:
| ID | Insight | Classification |
|----|---------|----------------|
| I1 | Keyword-only misses OAuth2 tasks | P1-IMPROVEMENT |
| I2 | Unit tests pending | P1-DEBT |
| I3 | More L1_complex patterns needed | P1-IMPROVEMENT |

**Phase 8 Decision**:
- P0 insights: 0
- P1 insights: 3
- Iteration: 1 (< 3 minimum)
- Decision: **RECOMMEND CONTINUE** (loop back to Phase 0 for research)

### Iteration 2

**Phase 3**: Add more L1_complex keyword patterns to task-router.yaml

**Phase 6 Classified Insights**:
| ID | Insight | Classification |
|----|---------|----------------|
| I1 | Patterns now catch OAuth2 | ADDRESSED |
| I2 | Unit tests still pending | P1-DEBT |
| I3 | Could add more domain patterns | P2-ENHANCEMENT |

**Phase 8 Decision**:
- P0 insights: 0
- P1 insights: 1 (tests)
- New P0/P1 this iteration: 0 (I2 is repeat)
- Plateau counter: 1
- Decision: **CONTINUE** (plateau not yet reached)

### Iteration 3

**Phase 3**: Create unit tests for ACE Reflector and SemanticRouter

**Phase 6 Classified Insights**:
| ID | Insight | Classification |
|----|---------|----------------|
| I2 | Unit tests created | ADDRESSED |
| I3 | Could add more domain patterns | P2-ENHANCEMENT |

**Phase 7 Decision**:
- P0 insights: 0
- P1 insights: 0
- P2 insights: 0 (all addressed)
- Plateau counter: 2 (threshold reached)
- Decision: **EXIT - PLATEAU DETECTED**

**Result**: More thorough completion with tests written and patterns improved.

---

## Changelog

| Date | Change |
|------|--------|
| 2025-12-03 | Initial UPGRADE-012.3 research document created |
| 2025-12-03 | Phase 0 research completed with 2 topic areas |
| 2025-12-03 | Meta-RIC Loop v2.0 architecture designed |
| 2025-12-03 | Phase 6 enhanced metacognition template created |
| 2025-12-03 | Phase 8 integration template created |
| 2025-12-03 | **v2.1 UPGRADE**: Renumbered to 7 phases (Phase 8 → Phase 7) |
| 2025-12-03 | **v2.1 UPGRADE**: Min 2 iterations (was min 1) |
| 2025-12-03 | **v2.1 UPGRADE**: Complete ALL P0-P2 (not just P0) |
| 2025-12-03 | **v2.1 UPGRADE**: No insights → Loop to Phase 0 (Research) |
| 2025-12-03 | Updated L1_complex.md with v2.1 workflow |
| 2025-12-03 | Updated CLAUDE.md with v2.1 diagram |
| 2025-12-03 | **v2.1 UPGRADE**: Added Exit Finalization Process (6-step checklist) |
| 2025-12-03 | Enhanced RIC loop triggers in task-router.yaml (v1.2.0) |
| 2025-12-03 | **v2.2 UPGRADE**: Added tiered integration check (Step 4g) - basic vs complete |
| 2025-12-03 | **P0+P1 COMPLETE** - UPGRADE-012.3 v2.2 implemented |
| 2025-12-03 | **v3.0 UPGRADE**: Strict sequential execution, ALL loops to Phase 0, min 3 iterations |
| 2025-12-03 | **v3.0 UPGRADE**: P2 is REQUIRED (not optional), compaction protection added |
| 2025-12-03 | **P0+P1+P2 COMPLETE** - UPGRADE-012.3 v3.0 implemented |

---

## Research Sources

- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
- [MetaAgent: Self-Evolving via Tool Meta-Learning](https://arxiv.org/html/2508.00271v2)
- [ICML 2025: Intrinsic Metacognitive Learning](https://openreview.net/forum?id=4KhDd0Ozqe)
- [Microsoft AI Agents: Metacognition](https://microsoft.github.io/ai-agents-for-beginners/09-metacognition/)
- [Google ADK Loop Agents](https://google.github.io/adk-docs/agents/workflow-agents/loop-agents/)
- [Metagent-P: Neuro-Symbolic Planning](https://aclanthology.org/2025.findings-acl.1169.pdf)
