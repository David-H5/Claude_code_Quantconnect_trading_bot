# UPGRADE-013: Autonomous Agent Prompt Gap Analysis

## Research Overview

**Date**: December 3, 2025
**Scope**: Evaluate hierarchical prompt system for overnight autonomous operation
**Status**: ✅ P0+P1+P2 COMPLETE
**Predecessor**: [UPGRADE-012 Hierarchical Prompts](UPGRADE-012-HIERARCHICAL-PROMPTS.md)

---

## Phase 0: Research Summary

**Search Date**: December 3, 2025 at ~11:00 AM EST

### Research Topic 1: Autonomous Coding Agent Best Practices

**Search Query**: "autonomous coding agent prompt engineering edge cases 2025"

**Key Sources**:
1. [Anthropic Claude Code Best Practices (Published: 2025)](https://www.anthropic.com/engineering/claude-code-best-practices)
2. [Simon Willison - Designing Agentic Loops (Published: Sep 2025)](https://simonwillison.net/2025/Sep/30/designing-agentic-loops/)
3. [Weaviate Agentic AI Patterns (Published: Nov 2025)](https://weaviate.io/blog/what-is-agentic-ai)

**Key Findings**:
- Context engineering is critical: CLAUDE.md files, git for state tracking
- Designing agentic loops: recovery from mistakes, graceful fallbacks
- Multi-agent patterns: orchestrators with fallback modules, specialization

### Research Topic 2: LLM Agent Self-Improvement Patterns

**Search Query**: "LLM agent self-improvement metacognition reflexion 2025"

**Key Sources**:
1. [Intrinsic Metacognitive Learning Framework (Published: 2025)](https://openreview.net/forum?id=4KhDd0Ozqe)
2. [LLM Self-Improvement Comprehensive Survey (Published: May 2025)](https://arxiv.org/abs/2505.08228)
3. [Verbal Reinforcement Learning (Published: 2025)](https://arxiv.org/abs/2502.00773)

**Key Findings**:
- **Three metacognitive components**:
  1. Metacognitive Knowledge (understanding task difficulty)
  2. Metacognitive Planning (strategy selection)
  3. Metacognitive Evaluation (self-monitoring progress)
- Reflexion framework: linguistic feedback for self-improvement
- Self-generated training: agents improve by learning from mistakes

### Research Topic 3: Error Recovery & Stuck Loop Detection

**Search Query**: "LLM agent error recovery stuck loop detection graceful fallback 2025"

**Key Sources**:
1. [OpenHands PR #5500 - Graceful Recovery (Published: Nov 2025)](https://github.com/All-Hands-AI/OpenHands/pull/5500)
2. [Agentic AI Error Handling Patterns (Published: 2025)](https://weaviate.io/blog/what-is-agentic-ai)
3. [Multi-LLM Routing Strategies - AWS (Published: 2025)](https://aws.amazon.com/blogs/machine-learning/multi-llm-routing-strategies-for-generative-ai-applications-on-aws/)

**Key Findings**:
- **Stuck Loop Detection**: Monitor for repeating patterns, maxIterations limits
- **Graceful Recovery**: Orchestrator pattern with fallback modules
- **Error Classification**: Distinguish recoverable vs unrecoverable errors
- **State Preservation**: Checkpoint before risky operations

### Research Topic 4: Hierarchical Task Planning

**Search Query**: "hierarchical task decomposition LLM agent MCTS planning 2025"

**Key Sources**:
1. [HALO Framework - Hierarchical Reasoning (Published: May 2025)](https://arxiv.org/html/2505.13516v1)
2. [Hierarchical Task DAG Research (Published: 2025)](https://arxiv.org/html/2505.08228v1)
3. [Task Mining for Intelligent Automation (Published: 2025)](https://link.springer.com/article/10.1007/s41019-025-00296-9)

**Key Findings**:
- **HTDAG (Hierarchical Task DAG)**: Break complex tasks into dependency graphs
- **MCTS for subtask execution**: Monte Carlo Tree Search for exploration
- **Task difficulty estimation**: Predict complexity before execution
- **Dependency management**: Track task prerequisites and sequencing

---

## Phase 5: Introspection Results

### Current Prompt System Inventory

| Category | Count | Purpose |
|----------|-------|---------|
| **Complexity Prompts** | 3 | L1_simple (direct), L1_moderate (3-5 steps), L1_complex (Meta-RIC 7 phases) |
| **Domain Prompts** | 6 | algorithm, llm_agent, evaluation, infrastructure, documentation, general |
| **Slash Commands** | 11 | /ric-start, /ric-research, /ric-introspect, /ric-integrate, etc. |
| **Task Router** | 1 | config/task-router.yaml (keyword+semantic+depth/width) |
| **Overnight Scripts** | 6 | run_overnight.sh, watchdog.py, auto-resume.sh, checkpoint.sh, etc. |

### Coverage Analysis

| Capability | Research Requirement | Current Implementation | Gap Level |
|------------|---------------------|------------------------|-----------|
| **Task Complexity Classification** | Depth × Width scoring | ✅ task-router.yaml v1.1 | None |
| **Complexity-Based Workflow** | Different workflows per level | ✅ L1_simple/moderate/complex | None |
| **Domain Specialization** | Domain-specific guidance | ✅ 6 domain prompts | None |
| **Loop Prevention** | Max iterations, plateau detection | ✅ RIC Loop has limits | None |
| **Crash Recovery** | Auto-resume, checkpointing | ✅ auto-resume.sh, checkpoint.sh | None |
| **Context Persistence** | Session notes, relay-race | ✅ claude-session-notes.md | None |
| **Stuck Detection** | Recognize no-progress patterns | ❌ No prompt guidance | **P0** |
| **Graceful Fallback** | Alternative approaches when blocked | ❌ No prompt guidance | **P0** |
| **Self-Monitoring** | Check if making progress | ❌ No prompt guidance | **P1** |
| **Error Recovery Prompts** | How to handle runtime errors | ❌ No prompt guidance | **P1** |
| **Time/Budget Awareness** | Track remaining resources | ⚠️ Watchdog only, not in prompts | **P2** |
| **Edge Case Library** | Known failure patterns | ❌ Not documented | **P2** |

---

## Phase 6: Classified Insights

### P0 - Critical (Required for Overnight Operation)

#### I1: Stuck Detection Prompt Missing

**Problem**: No explicit guidance on detecting when the agent is stuck in a non-productive loop (different from RIC iteration limits).

**Evidence**:
- OpenHands PR #5500 specifically addresses "graceful recovery from stuck loops"
- Current prompts only have iteration limits, not pattern detection

**Recommendation**: Create `prompts/autonomous/stuck_detection.md` with:
- Signs of being stuck (repeating same actions, no file changes, circular reasoning)
- When to declare "stuck" state
- Required actions: log state, try alternative, escalate

**Impact**: Without this, overnight sessions may burn context on unproductive loops.

---

#### I2: Graceful Fallback Prompt Missing

**Problem**: No guidance on what to do when a task/approach fails or is blocked.

**Evidence**:
- Research shows "orchestrator pattern with fallback modules" is critical
- L1_complex escalates to Phase 0, but no mid-task fallback guidance

**Recommendation**: Create `prompts/autonomous/graceful_fallback.md` with:
- Fallback hierarchy (alternative approach → simplify scope → skip and document → escalate)
- When to abandon vs persist
- How to preserve partial progress

**Impact**: Without this, agent may waste hours on blocked paths instead of pivoting.

---

### P1 - Important (Significantly Improves Robustness)

#### I3: Self-Monitoring Prompt Missing

**Problem**: No explicit "check if you're making progress" directive in prompts.

**Evidence**:
- Research shows metacognitive evaluation is one of three critical components
- Current RIC has Phase 6 metacognition but it's task-focused, not self-focused

**Recommendation**: Add self-monitoring section to complexity prompts:
- Every N minutes, assess: "Am I closer to the goal than before?"
- Track: files changed, tests passing, blockers resolved
- Trigger: If no progress for 30 min, enter stuck_detection protocol

**Impact**: Enables early detection of unproductive patterns.

---

#### I4: Error Recovery Prompt Missing

**Problem**: Prompts have "Before Committing" checklists but no runtime error handling guidance.

**Evidence**:
- Research distinguishes recoverable vs unrecoverable errors
- Current prompts assume happy path execution

**Recommendation**: Add `prompts/autonomous/error_recovery.md` with:
- Common error categories (import, syntax, API, test failure)
- Recovery actions per category
- When to rollback vs fix forward

**Impact**: Reduces time lost to repeated errors.

---

#### I5: Task Escalation Path Incomplete

**Problem**: L1_moderate says "escalate to COMPLEX workflow" but no criteria defined.

**Evidence**:
- HALO research shows task difficulty estimation is important
- Current escalation is mentioned but triggers undefined

**Recommendation**: Add explicit escalation triggers to L1_moderate.md:
- Task taking 2x expected time
- More than 3 unexpected blockers
- Scope creep detected (new requirements discovered)

**Impact**: Prevents underestimation of complex tasks.

---

### P2 - Nice to Have (Polish for Production)

#### I6: Time/Budget Awareness Not in Prompts

**Problem**: Watchdog tracks time/cost externally but prompts don't mention it.

**Evidence**:
- Overnight sessions have finite context budget
- Agent should plan work accordingly

**Recommendation**: Add to L1_complex.md:
- "Check remaining session time before starting new phase"
- "Prioritize high-value tasks when time limited"

**Impact**: Better resource utilization.

---

#### I7: Edge Case Library Missing

**Problem**: No documented library of known edge cases and failure patterns.

**Evidence**:
- Research shows "self-generated training from mistakes" is valuable
- ACE Reflector analyzes patterns but doesn't feed them into prompts

**Recommendation**: Create `prompts/autonomous/edge_cases.md`:
- Compile known failure patterns from ACE Reflector
- Document: Pattern, Detection, Recovery

**Impact**: Prevents repeated failures.

---

#### I8: Context Overflow Handling Missing

**Problem**: No guidance on what to do when context window is filling up.

**Evidence**:
- Long overnight sessions may approach context limits
- Agent should summarize/checkpoint before overflow

**Recommendation**: Add to overnight prompts:
- "Monitor context usage indicators"
- "Before context limit: commit checkpoint, summarize progress, update session notes"

**Impact**: Prevents loss of work from context exhaustion.

---

## Phase 7: Integration & Decision

### Summary of Gaps

| Priority | Count | Impact |
|----------|-------|--------|
| **P0** | 2 | Critical for overnight operation |
| **P1** | 3 | Significantly improves robustness |
| **P2** | 3 | Polish for production |

### Recommended Implementation Order

1. **P0-I1**: Create `prompts/autonomous/stuck_detection.md`
2. **P0-I2**: Create `prompts/autonomous/graceful_fallback.md`
3. **P1-I3**: Add self-monitoring to L1_complex.md
4. **P1-I4**: Create `prompts/autonomous/error_recovery.md`
5. **P1-I5**: Add escalation triggers to L1_moderate.md
6. **P2-I6**: Add time/budget awareness to L1_complex.md
7. **P2-I7**: Create `prompts/autonomous/edge_cases.md`
8. **P2-I8**: Add context overflow handling

### Files to Create/Modify

| File | Action | Priority |
|------|--------|----------|
| `prompts/autonomous/stuck_detection.md` | CREATE | P0 |
| `prompts/autonomous/graceful_fallback.md` | CREATE | P0 |
| `prompts/autonomous/error_recovery.md` | CREATE | P1 |
| `prompts/autonomous/edge_cases.md` | CREATE | P2 |
| `prompts/complexity/L1_complex.md` | MODIFY | P1/P2 |
| `prompts/complexity/L1_moderate.md` | MODIFY | P1 |
| `config/task-router.yaml` | MODIFY | P1 |

### Decision

**RECOMMENDATION**: Implement P0 items immediately for safe overnight operation. P1 items should follow in next session. P2 items can be deferred.

**Answer to User Question**: The current hierarchical prompt system has **good structural coverage** (complexity levels, domains, task routing) but is **missing critical autonomous operation guidance**:
- ❌ No stuck detection guidance
- ❌ No graceful fallback guidance
- ❌ No self-monitoring prompts
- ❌ No runtime error recovery prompts

**For safe overnight autonomous operation, P0 items must be implemented.**

---

## Changelog

| Date | Change |
|------|--------|
| 2025-12-03 | Initial UPGRADE-013 research document created |
| 2025-12-03 | Phase 0 research completed (4 topics) |
| 2025-12-03 | Phase 5 introspection completed |
| 2025-12-03 | Phase 6 insights classified (2 P0, 3 P1, 3 P2) |
| 2025-12-03 | Phase 7 recommendations documented |
| 2025-12-03 | P0-I1: Created prompts/autonomous/stuck_detection.md |
| 2025-12-03 | P0-I2: Created prompts/autonomous/graceful_fallback.md |
| 2025-12-03 | P1-I3: Added self-monitoring to L1_complex.md |
| 2025-12-03 | P1-I5: Added escalation triggers to L1_moderate.md |
| 2025-12-03 | Updated task-router.yaml to v1.3.0 with autonomous prompts |
| 2025-12-03 | **P0+P1 COMPLETE** - UPGRADE-013 implemented |
| 2025-12-03 | **Meta-RIC v3.0**: Rewrote L1_complex.md with strict enforcement |
| 2025-12-03 | v3.0: Mandatory loop counter, min 3 iterations, P2 required |
| 2025-12-03 | Updated CLAUDE.md to v3.0 with UPGRADE-013 reference |
| 2025-12-03 | **RIC LOOP RESTART**: Starting iteration 1/5 for P2 completion |
| 2025-12-03 | T1: Created prompts/autonomous/error_recovery.md (7 error categories) |
| 2025-12-03 | T2: Created prompts/autonomous/edge_cases.md (10 failure patterns) |
| 2025-12-03 | T3: Added Time & Budget Awareness section to L1_complex.md |
| 2025-12-03 | T4: Added Context Overflow Recovery section to L1_complex.md |
| 2025-12-03 | T5: Updated task-router.yaml v1.3.0 with all autonomous prompts |
| 2025-12-03 | **P0+P1+P2 COMPLETE** - UPGRADE-013 fully implemented |

---

## RIC Loop Iteration 1/5 - Phase 0 Research

**Search Date**: December 3, 2025 at ~3:30 PM EST

### Gap Verification

**Files that EXIST**:
- `prompts/autonomous/stuck_detection.md` ✅
- `prompts/autonomous/graceful_fallback.md` ✅

**Files that DO NOT EXIST (P2 remaining)**:
- `prompts/autonomous/edge_cases.md` ❌
- `prompts/autonomous/error_recovery.md` ❌ (P1-I4 was NOT completed)

**Missing in L1_complex.md**:
- Time/budget awareness (I6) ❌
- Context overflow handling (I8) ❌

### P2 Items Now Required (v3.0 Rule)

| ID | Item | Status | Action Required |
|----|------|--------|-----------------|
| I4 | Error Recovery Prompt | NOT DONE | Create `prompts/autonomous/error_recovery.md` |
| I6 | Time/Budget Awareness | NOT DONE | Add to L1_complex.md |
| I7 | Edge Case Library | NOT DONE | Create `prompts/autonomous/edge_cases.md` |
| I8 | Context Overflow Handling | NOT DONE | Add to overnight prompts |

---

## RIC Loop Iteration 1/5 - Phase 1 Upgrade Path

### Target State

Complete all remaining P2 items (now required per v3.0) plus the missing P1 item (I4):

1. **Error Recovery Prompt** (P1-I4): Create comprehensive error handling guide
2. **Time/Budget Awareness** (P2-I6): Add resource awareness to L1_complex.md
3. **Edge Case Library** (P2-I7): Create pattern library from known failures
4. **Context Overflow Handling** (P2-I8): Add compaction/overflow guidance

### Success Criteria

| ID | Criteria | Verification |
|----|----------|--------------|
| SC1 | `prompts/autonomous/error_recovery.md` exists with ≥5 error categories | File exists, content check |
| SC2 | `prompts/autonomous/edge_cases.md` exists with ≥5 patterns | File exists, content check |
| SC3 | L1_complex.md has time/budget awareness section | Grep for "time budget" or "session time" |
| SC4 | L1_complex.md has context overflow section | Grep for "context overflow" or "compaction" |
| SC5 | task-router.yaml references new autonomous prompts | Config updated |
| SC6 | UPGRADE-013 status updated to "P0+P1+P2 COMPLETE" | Research doc updated |

### Scope

**In Scope**:

- Create error_recovery.md with common error categories
- Create edge_cases.md with known failure patterns
- Add time/budget awareness section to L1_complex.md
- Add context overflow handling to L1_complex.md
- Update task-router.yaml if needed
- Update UPGRADE-013 research document

**Out of Scope**:

- ACE Reflector integration (separate upgrade)
- Automated pattern extraction (future enhancement)
- External monitoring integration (UPGRADE-011 covers this)

---

## RIC Loop Iteration 1/5 - Phase 2 Checklist

### P0 - Critical (All Previously Completed)

- [x] I1: stuck_detection.md ✅ EXISTS
- [x] I2: graceful_fallback.md ✅ EXISTS

### P1 - Important

- [x] T1: Create `prompts/autonomous/error_recovery.md` (I4) ✅
  - 7 error categories with recovery actions
  - Recoverable vs unrecoverable distinction
  - Rollback vs fix-forward guidance

### P2 - Polish (REQUIRED per v3.0)

- [x] T2: Create `prompts/autonomous/edge_cases.md` (I7) ✅
  - 10 known failure patterns
  - Pattern → Detection → Recovery format

- [x] T3: Add time/budget awareness to L1_complex.md (I6) ✅
  - Session time monitoring section added
  - Time-limited prioritization table
  - Budget/cost awareness section

- [x] T4: Add context overflow handling to L1_complex.md (I8) ✅
  - Context overflow recovery section added
  - Emergency overflow procedure
  - Session continuation protocol
  - Prevention strategies

- [x] T5: Update task-router.yaml with new prompts ✅
  - Added error_recovery and edge_cases references

- [x] T6: Update UPGRADE-013 status to complete ✅
  - Status: P0+P1+P2 COMPLETE

### Execution Order

1. T1 (error_recovery.md) - highest impact P1
2. T2 (edge_cases.md) - P2 file creation
3. T3 (time/budget in L1_complex) - P2 modification
4. T4 (context overflow in L1_complex) - P2 modification
5. T5 (task-router update) - integration
6. T6 (status update) - finalization
