# Complex Task Workflow (L3) - Meta-RIC Loop v3.0

You are working on a **complex task** that requires the Meta-RIC Loop workflow with STRICT sequential phase execution.

## CRITICAL: Workflow Enforcement Rules

**RULE 1: STRICT SEQUENTIAL EXECUTION**
- You MUST execute phases in order: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7
- You CANNOT skip ANY phase
- You MUST log each phase transition with: `=== PHASE X: [Name] ===`
- If you find yourself in Phase 5 without going through 1-4, STOP and restart

**RULE 2: MANDATORY LOOP COUNTER**
- You MUST track and display the current iteration number
- Format: `[ITERATION X/5]` at the start of each phase
- Example: `[ITERATION 1/5] === PHASE 0: RESEARCH ===`

**RULE 3: MINIMUM 3 ITERATIONS**
- You CANNOT exit before completing 3 full iterations
- Even if no insights found, you MUST loop back and look harder
- Iteration 1: Initial implementation
- Iteration 2: Refinement based on insights
- Iteration 3: Polish and verification

**RULE 4: ALL P0/P1/P2 MUST BE RESOLVED**
- You CANNOT defer P2 items - they MUST be completed
- Exit is BLOCKED if ANY P0/P1/P2 insights remain unaddressed
- "Nice to have" is still REQUIRED with AI assistance

---

## Required Workflow: Meta-RIC Loop v3.0

```text
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              META-RIC LOOP v3.0                                      │
│                         STRICT SEQUENTIAL EXECUTION                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   [ITERATION X/5]                                                                    │
│                                                                                      │
│   Phase 0 ──▶ Phase 1 ──▶ Phase 2 ──▶ Phase 3 ──▶ Phase 4 ──▶ Phase 5 ──▶ Phase 6  │
│   RESEARCH    UPGRADE     CHECKLIST   CODING      DOUBLE-     INTRO-      META-     │
│               PATH                    (1 file     CHECK       SPECTION    COGNITION │
│                                       at a time)                                     │
│                                                                                      │
│                                                                           │          │
│                                                                           ▼          │
│                                                                    ┌─────────────┐   │
│                                                                    │  Phase 7    │   │
│                                                                    │ INTEGRATION │   │
│                                                                    └──────┬──────┘   │
│                                                                           │          │
│   ┌───────────────────────────────────────────────────────────────────────┤          │
│   │                                                                       │          │
│   ▼                                                                       ▼          │
│   ┌──────────────────────┐                                    ┌──────────────────┐  │
│   │ P0/P1/P2 REMAIN?     │                                    │ ALL RESOLVED +   │  │
│   │ OR iteration < 3?    │                                    │ iteration >= 3?  │  │
│   │                      │                                    │                  │  │
│   │ → LOOP to Phase 0    │                                    │ → EXIT ALLOWED   │  │
│   └──────────────────────┘                                    └──────────────────┘  │
│                                                                                      │
│   Iteration: [1] [2] [3] [4] [5]   ← Min 3, Max 5, ALL P0-P2 REQUIRED              │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Execution Format (REQUIRED)

Each phase MUST be logged with this exact format:

```text
[ITERATION X/5] === PHASE N: PHASE_NAME ===

[Phase content here]

=== PHASE N COMPLETE - Gate Check ===
- [ ] Gate criteria 1
- [ ] Gate criteria 2
...
All gates passed: [YES/NO]

→ PROCEEDING TO PHASE N+1
```

---

## Phase 0: Research

```text
[ITERATION X/5] === PHASE 0: RESEARCH ===
```

**Purpose**: Online research BEFORE any planning or coding

**Required Actions**:

1. Conduct online searches for relevant topics
2. Document at least 3 sources with timestamps
3. Identify approach options with trade-offs
4. Create or update research document in `docs/research/`

**CRITICAL: Compaction Protection**

⚠️ **IMMEDIATELY write research to file after EVERY search!**

- Context compaction can occur at ~70% of 200K tokens
- Research held only in context WILL BE LOST
- Write findings to `docs/research/UPGRADE-XXX.md` within SAME response as search
- Pattern: WebSearch → IMMEDIATELY Write to file → Next search

**Output Format**:

```markdown
### Research Results

**Search Date**: [Date and time]
**Search Queries**: [List queries used]

**Key Sources**:
1. [Source Title (Published: Date)](URL)
2. [Source Title (Published: Date)](URL)
3. [Source Title (Published: Date)](URL)

**Key Findings**:
- Finding 1
- Finding 2

**Approach Options**:
| Option | Pros | Cons |
|--------|------|------|
| A | ... | ... |
| B | ... | ... |

**Recommended Approach**: [Choice with justification]
```

**GATE 0 Checklist**:

- [ ] Online research conducted with timestamped sources
- [ ] At least 3 relevant sources found
- [ ] Key insights documented in research file
- [ ] Approach options identified with trade-offs

**All gates passed? → PROCEED TO PHASE 1**

---

## Phase 1: Upgrade Path

```text
[ITERATION X/5] === PHASE 1: UPGRADE PATH ===
```

**Purpose**: Define WHAT will be built based on research

**Required Actions**:

1. Write a formal Upgrade Path section in research doc
2. Define target state clearly
3. Set scope boundaries (in/out of scope)
4. List measurable success criteria

**Output Format**:

```markdown
## Upgrade Path

### Problem Statement

**Current State**: [What exists now]
**Gap**: [What's missing]
**Target State**: [What we're building]

### Scope

**In Scope**:
- Item 1
- Item 2

**Out of Scope**:
- Item A (reason)
- Item B (reason)

### Success Criteria

1. [ ] Criterion 1 (measurable)
2. [ ] Criterion 2 (measurable)
3. [ ] Criterion 3 (measurable)

### Dependencies

- Dependency 1
- Dependency 2
```

**GATE 1 Checklist**:

- [ ] Target state clearly defined
- [ ] Scope boundaries set (in/out of scope)
- [ ] Success criteria are measurable
- [ ] Dependencies identified

**All gates passed? → PROCEED TO PHASE 2**

---

## Phase 2: Checklist

```text
[ITERATION X/5] === PHASE 2: CHECKLIST ===
```

**Purpose**: Break work into prioritized tasks

**Required Actions**:

1. Break work into 30min-4hr tasks
2. Assign P0/P1/P2 priority to EVERY task
3. Map dependencies between tasks
4. Identify ALL P2 polish items upfront

**Output Format**:

```markdown
## Implementation Checklist

### P0 - Critical (Must complete for basic functionality)

- [ ] P0-1: [Task description] (~Xhr)
- [ ] P0-2: [Task description] (~Xhr)

### P1 - Important (Required for robustness)

- [ ] P1-1: [Task description] (~Xhr)
- [ ] P1-2: [Task description] (~Xhr)

### P2 - Polish (Required for production-ready)

- [ ] P2-1: [Task description] (~Xhr)
- [ ] P2-2: [Task description] (~Xhr)

### Task Dependencies

P0-1 → P0-2 → P1-1
P1-1 → P2-1
```

**GATE 2 Checklist**:

- [ ] Tasks broken into 30min-4hr chunks
- [ ] P0 (critical), P1 (important), P2 (polish) assigned to ALL
- [ ] Dependencies between tasks mapped
- [ ] ALL P0-P2 tasks identified (not just P0)

**All gates passed? → PROCEED TO PHASE 3**

---

## Phase 3: Coding

```text
[ITERATION X/5] === PHASE 3: CODING ===
```

**Purpose**: Execute tasks ONE AT A TIME

**CRITICAL RULES**:

1. **ONE FILE PER CHANGE** - Do not modify multiple files at once
2. **COMMIT AFTER EACH** - `git commit` after every file change
3. **TEST AS YOU GO** - Write tests alongside implementation
4. **FOLLOW CHECKLIST ORDER** - P0 first, then P1, then P2

**Execution Format**:

```text
--- Task P0-1: [Description] ---
File: [path/to/file.py]
Action: [CREATE/MODIFY]

[Make the change]

Commit: git commit -m "feat: [description]"
Tests: [PASS/FAIL]
---
```

**GATE 3 Checklist**:

- [ ] All P0 tasks completed
- [ ] All P1 tasks completed
- [ ] All P2 tasks completed
- [ ] Tests written for new functionality
- [ ] All tests pass locally
- [ ] Git commits made for each component

**All gates passed? → PROCEED TO PHASE 4**

---

## Phase 4: Double-Check

```text
[ITERATION X/5] === PHASE 4: DOUBLE-CHECK ===
```

**Purpose**: Verify implementation completeness

**Required Actions**:

1. Run full test suite
2. Check test coverage
3. Verify documentation updated
4. Review for critical gaps

**Output Format**:

```text
### Verification Results

**Test Suite**: pytest tests/ -v
Result: [PASS/FAIL]
Tests passed: X/Y

**Coverage**: pytest --cov=.
Coverage: X%
Threshold: 70%
Status: [PASS/FAIL]

**Documentation**:
- [ ] Docstrings added
- [ ] README updated (if needed)
- [ ] Research doc updated

**Critical Gaps**:
- [None / List gaps]
```

**GATE 4 Checklist**:

- [ ] Functional requirements met
- [ ] Test coverage > 70%
- [ ] Documentation updated
- [ ] No critical gaps remaining

**All gates passed? → PROCEED TO PHASE 5**

---

## Phase 5: Introspection

```text
[ITERATION X/5] === PHASE 5: INTROSPECTION ===
```

**Purpose**: Find ALL gaps, bugs, and improvements

**Required Actions**:

1. Check for missing files/components
2. Document known bugs
3. Capture expansion ideas
4. Note technical debt

**Look For**:

- Files mentioned but not created
- Functions referenced but not implemented
- Edge cases not handled
- Error handling missing
- Performance concerns
- Security considerations

**Output Format**:

```markdown
### Introspection Results

**Missing Files/Components**:
- [ ] [File/component that should exist]

**Known Bugs**:
- [ ] [Bug description]

**Expansion Ideas**:
- [ ] [Idea that would improve the feature]

**Technical Debt**:
- [ ] [Debt item]

**Total Issues Found**: X
```

**GATE 5 Checklist**:

- [ ] Missing files/components identified
- [ ] Known bugs documented
- [ ] Expansion ideas captured
- [ ] Technical debt noted

**All gates passed? → PROCEED TO PHASE 6**

---

## Phase 6: Metacognition

```text
[ITERATION X/5] === PHASE 6: METACOGNITION ===
```

**Purpose**: Self-reflection + CLASSIFY all insights as P0/P1/P2

**Part A: Self-Reflection Questions**

Answer with confidence scores:

1. Did implementation achieve research goals? [Confidence: X%]
2. Are there known gaps in the approach? [Confidence: X%]
3. What assumptions need validation? [Confidence: X%]
4. What would I do differently? [Confidence: X%]
5. Is the implementation maintainable? [Confidence: X%]

**Part B: Classified Insights (REQUIRED)**

Create a table of ALL insights from Phase 5:

| ID | Insight | Classification | Action Required | Est. Effort |
|----|---------|----------------|-----------------|-------------|
| I1 | [Description] | [P0-GAP/P1-DEBT/P2-POLISH] | [Action] | [Xhr] |
| I2 | [Description] | [Classification] | [Action] | [Xhr] |

**Classification Legend**:

- **P0-GAP**: Critical gap that blocks completion
- **P0-BUG**: Bug that must be fixed
- **P1-DEBT**: Technical debt that should be addressed
- **P1-IMPROVEMENT**: Improvement worth making
- **P2-POLISH**: Makes it production-ready
- **P2-ENHANCEMENT**: Adds completeness

**Summary**:

```text
Total insights: X
- P0 (Critical): X
- P1 (Important): X
- P2 (Polish): X
```

**GATE 6 Checklist**:

- [ ] Self-reflection questions answered
- [ ] ALL insights from Phase 5 classified
- [ ] Each insight has action and effort estimate

**All gates passed? → PROCEED TO PHASE 7**

---

## Phase 7: Integration & Loop Decision

```text
[ITERATION X/5] === PHASE 7: INTEGRATION & DECISION ===
```

**Step 1: Iteration Status**

```text
Current iteration: [X]
Minimum required: 3
Maximum allowed: 5

P0 insights remaining: [count]
P1 insights remaining: [count]
P2 insights remaining: [count]
Total unresolved: [count]
```

**Step 2: Exit Eligibility Check**

```text
EXIT ELIGIBILITY:
- [ ] Iteration >= 3? [YES/NO]
- [ ] P0 insights = 0? [YES/NO]
- [ ] P1 insights = 0? [YES/NO]
- [ ] P2 insights = 0? [YES/NO]

ALL CHECKS PASSED? [YES/NO]
```

**Step 3: Loop Decision**

Apply these rules IN ORDER:

```text
1. IF iteration < 3:
   → MANDATORY LOOP to Phase 0
   → Reason: Minimum iterations not met, do more research

2. ELIF P0 insights > 0:
   → MANDATORY LOOP to Phase 0
   → Reason: Critical gaps remain, research solutions

3. ELIF P1 insights > 0:
   → MANDATORY LOOP to Phase 0
   → Reason: Important items remain, research approach

4. ELIF P2 insights > 0:
   → MANDATORY LOOP to Phase 0
   → Reason: Polish items remain (P2 is REQUIRED, not optional)

5. ELIF iteration >= 5:
   → FORCED EXIT (max iterations)
   → Document any remaining work

6. ELSE (all resolved AND iteration >= 3):
   → EXIT ALLOWED
```

**Decision**: [LOOP_TO_PHASE_0 | EXIT_ALLOWED | FORCED_EXIT]
**Reason**: [Explanation]

**If Looping**:

```text
=== LOOPING TO PHASE 0 (RESEARCH) ===
Iteration: [X] → [X+1]
Items to research/address this iteration:
- [Item 1]
- [Item 2]
...
```

Then return to Phase 0 with `[ITERATION X+1/5] === PHASE 0: RESEARCH ===`

---

## Exit Finalization (Only when EXIT_ALLOWED)

**Complete ONLY if Decision = EXIT_ALLOWED**

### Finalization Checklist

**4a. Final Bug Finding**:

- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Check for syntax errors in all modified files
- [ ] Review any TODO/FIXME comments
- [ ] Verify no debug prints left behind

**4b. Missing Files Check**:

- [ ] All files mentioned in checklist exist
- [ ] All new imports work (no ImportError)
- [ ] Dependencies added to `requirements.txt`

**4c. Test Coverage**:

- [ ] Tests exist for all new functionality
- [ ] All tests pass
- [ ] Coverage > 70%

**4d. Documentation**:

- [ ] Research doc marked complete
- [ ] CLAUDE.md updated if needed
- [ ] Docstrings added

**4e. Cross-References**:

- [ ] Related files updated
- [ ] Config files match code

**4f. Cleanup**:

- [ ] Git status clean
- [ ] No temp files

### Exit Report

```text
=== RIC LOOP EXIT REPORT ===

Completion Status:
- Total iterations: [X]
- P0 insights addressed: [X]
- P1 insights addressed: [X]
- P2 insights addressed: [X]
- Exit reason: [COMPLETE | FORCED_EXIT]

Finalization:
- Bug finding: [PASS/FAIL]
- Missing files: [PASS/FAIL]
- Test coverage: [X% - PASS/FAIL]
- Documentation: [PASS/FAIL]
- Cross-references: [PASS/FAIL]
- Cleanup: [PASS/FAIL]

Files Modified/Created:
- [file1]
- [file2]

Quality Assessment:
- Production-ready: [YES/NO]
- All P2 polish complete: [YES/NO]
```

---

## Iteration Limits

| Limit | Value | Rationale |
|-------|-------|-----------|
| **Minimum** | **3 iterations** | Ensures thorough implementation |
| Maximum | 5 iterations | Prevents infinite loops |

---

## Critical Rules Summary

1. **STRICT SEQUENTIAL**: Phases 0→1→2→3→4→5→6→7 in order, NO skipping
2. **LOG EVERY PHASE**: `[ITERATION X/5] === PHASE N: NAME ===`
3. **MINIMUM 3 ITERATIONS**: Cannot exit before 3 complete loops
4. **ALL P0/P1/P2 REQUIRED**: P2 is NOT optional, must be completed
5. **ONE FILE AT A TIME**: Single-component changes only
6. **COMMIT EACH CHANGE**: Git checkpoint after every modification
7. **GATE ENFORCEMENT**: Do NOT proceed until gate criteria met

---

## Time & Budget Awareness (UPGRADE-013)

### Session Time Monitoring

Before starting each phase, estimate remaining session time:

```text
TIME CHECK (Every Phase Transition):
- Estimated remaining time: [X hours/minutes]
- Current phase: [N]
- Phases remaining: [7 - N]
- Avg time per phase: [~15-30 min]

Status: [ON_TRACK | TIME_LIMITED | URGENT]
```

### Time-Limited Prioritization

When session time is limited, adjust approach:

| Time Remaining | Strategy |
|---------------|----------|
| > 2 hours | Normal execution, full P0/P1/P2 |
| 1-2 hours | Complete current P0/P1, defer P2 to next session |
| 30min-1hr | Complete current P0, checkpoint P1/P2 |
| < 30 min | Emergency checkpoint, document state |

**Time-Limited Actions**:

```text
IF time_remaining < 1 hour:
  1. Immediately commit current work
  2. Update progress file with incomplete items
  3. Prioritize:
     - P0: Must complete (functional requirement)
     - P1: Try to complete (robustness)
     - P2: Document for next session (polish)
  4. Create handoff notes in session-notes.md
```

### Context Budget Monitoring

Monitor context window usage to prevent research loss:

```text
CONTEXT CHECK (Every ~30 minutes):
- Estimated context usage: [~X% of 200K tokens]
- Research held in context: [List]
- Files modified not committed: [List]

IF context > 60%:
  → Write ALL research to files immediately
  → Commit current work
  → Consider using /compact

IF context > 80%:
  → CRITICAL: Compaction imminent
  → Emergency checkpoint all work
  → Update session-notes.md with full state
```

### Budget/Cost Awareness

For cost-sensitive sessions:

```text
COST ESTIMATE:
- Session duration: [X hours]
- Estimated tokens used: [~Y tokens]
- Model: [Sonnet/Opus]
- Estimated cost: [$Z]

IF approaching budget limit:
  1. Prioritize high-value completions
  2. Reduce exploratory research
  3. Focus on P0 tasks only
  4. Document P1/P2 for future sessions
```

### Graceful Degradation Protocol

When resources (time/context/budget) are constrained:

```text
RESOURCE CONSTRAINT DETECTED

Level 1 - Yellow (Caution):
- Time: 1-2 hours remaining
- Context: 60-70% used
- Action: Accelerate, focus on priorities

Level 2 - Orange (Warning):
- Time: 30min-1hr remaining
- Context: 70-80% used
- Action: Complete current task, checkpoint

Level 3 - Red (Critical):
- Time: < 30 min remaining
- Context: > 80% used
- Action: Emergency checkpoint, handoff

Emergency Checkpoint Steps:
1. git add -A && git commit -m "WIP: [current task] - resource constraint"
2. Update claude-session-notes.md with:
   - Current goal
   - Work completed
   - Work remaining
   - Next steps
3. Update claude-progress.txt with accurate status
4. Create HANDOFF.md if significant context to transfer
```

---

## Autonomous Operation (UPGRADE-013)

When running autonomously, apply self-monitoring every 15-20 minutes:

```text
PROGRESS CHECK:
- [ ] Have I created/modified files since last check?
- [ ] Have tests/lints improved?
- [ ] Is my approach different from before?
- [ ] Am I closer to the goal?

If 2+ unchecked → Check for stuck patterns
```

**Stuck Patterns** → See [stuck_detection.md](../autonomous/stuck_detection.md)
**Fallback Protocol** → See [graceful_fallback.md](../autonomous/graceful_fallback.md)
**Error Recovery** → See [error_recovery.md](../autonomous/error_recovery.md)
**Edge Cases** → See [edge_cases.md](../autonomous/edge_cases.md)

---

## Context Overflow Recovery (UPGRADE-013)

### Detecting Impending Overflow

Context overflow typically occurs when:

- Multiple large file reads have accumulated
- Extensive code generation without commits
- Long debugging sessions with many tool calls
- Research with multiple web searches held in context

**Warning Signs**:

```text
OVERFLOW WARNING SIGNS:
- Response time noticeably slower
- Responses becoming shorter/truncated
- "Let me summarize..." appearing unprompted
- System indicating context limit approaching
- /compact command becoming available
```

### Emergency Overflow Procedure

When context overflow is imminent or occurring:

```text
=== EMERGENCY CONTEXT OVERFLOW PROCEDURE ===

STEP 1: STOP current task (do not start new operations)

STEP 2: COMMIT immediately
  git add -A
  git commit -m "WIP: [task] - context overflow checkpoint"

STEP 3: CREATE session handoff file

  File: claude-session-notes.md (or HANDOFF.md if complex)

  Contents:
  ---
  ## Session Handoff - [Date/Time]

  ### Current RIC Loop State
  - Iteration: [X/5]
  - Current Phase: [N]
  - Task: [Current task ID]

  ### Work Completed This Session
  - [List of completed items]
  - [Files created/modified]

  ### Work In Progress
  - [Current task details]
  - [What was being attempted]
  - [Last successful state]

  ### Next Steps (for continuing session)
  1. [Immediate next action]
  2. [Following action]
  3. [...]

  ### Key Context (preserve this information)
  - [Important discoveries]
  - [Decisions made and why]
  - [Gotchas encountered]
  ---

STEP 4: UPDATE progress tracking
  - claude-progress.txt: Mark current task status
  - UPGRADE-XXX.md: Update checklist

STEP 5: If time permits, use /compact
  - This summarizes context and frees space
  - Review summary to ensure nothing critical lost
```

### Session Continuation Protocol

When starting a new session after overflow:

```text
=== NEW SESSION STARTUP ===

STEP 1: READ handoff files
  - claude-session-notes.md
  - claude-progress.txt
  - Relevant UPGRADE-XXX.md

STEP 2: VERIFY git state
  git status
  git log --oneline -5

STEP 3: IDENTIFY current position
  - Which RIC iteration?
  - Which phase?
  - Which task?

STEP 4: RESUME from last checkpoint
  - Re-read only necessary files (not everything)
  - Continue from documented next step
  - Log: "[ITERATION X/5] === PHASE N: NAME === (RESUMED)"

STEP 5: AVOID re-reading large files
  - Use targeted reads (specific line ranges)
  - Reference handoff notes instead of re-researching
  - Trust committed code state
```

### Preventing Context Overflow

Best practices to avoid overflow:

```text
PREVENTION STRATEGIES:

1. COMMIT FREQUENTLY
   - After every file change
   - After every test pass
   - Before starting new research

2. WRITE RESEARCH IMMEDIATELY
   - WebSearch → Immediately write to file
   - Never hold 3+ search results in context only
   - Pattern: Search → Write → Search → Write

3. MINIMIZE FILE READS
   - Read only what's needed
   - Use line ranges for large files
   - Don't re-read files already understood

4. USE STRUCTURED CHECKPOINTS
   - Every 20-30 minutes
   - At each phase transition
   - Before any risky operation

5. PROGRESSIVE SUMMARIZATION
   - Summarize completed work in handoff notes
   - Don't rely on full context for decisions
   - Document key facts for future reference
```

---

## Start Checklist

Before beginning, verify:

- [ ] I understand this is a complex task requiring full RIC loop
- [ ] I will execute ALL 7 phases in order
- [ ] I will complete MINIMUM 3 iterations
- [ ] I will NOT skip P2 items
- [ ] I will log each phase with `[ITERATION X/5] === PHASE N ===`

**Ready? Begin with:**

```text
[ITERATION 1/5] === PHASE 0: RESEARCH ===
```
