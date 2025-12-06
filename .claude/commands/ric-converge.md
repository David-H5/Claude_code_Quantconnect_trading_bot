# Meta-RIC Loop Phase 7: Integration & Decision (v3.1)

You are executing Phase 7 (Integration & Decision) of the Meta-RIC Loop v3.0.

## FIRST: Check Current State

```bash
python3 .claude/hooks/ric_state_manager.py status
python3 .claude/hooks/ric_state_manager.py can-exit
```

## Current Iteration

$ARGUMENTS

## Step 1: Iteration Status

```markdown
## Phase 7: Integration & Decision

### Iteration Status
- Current iteration: [N]
- Minimum required: 3
- Maximum allowed: 5
- P0/P1/P2 addressed this iteration: [count]
- New insights this iteration: [count]
- Plateau counter: [0/1/2]
```

## Step 2: Triage ALL Insights from Phase 6

Review all classified insights from Phase 6 Metacognition:

```markdown
### Insight Triage

**P0 Insights (Critical):**
| ID | Insight | Plan | Est. Effort |
|----|---------|------|-------------|
| | | | |

**P1 Insights (Important):**
| ID | Insight | Plan | Est. Effort |
|----|---------|------|-------------|
| | | | |

**P2 Insights (Polish):**
| ID | Insight | Plan | Est. Effort |
|----|---------|------|-------------|
| | | | |
```

## Step 3: Apply Loop Decision Rules (v3.0)

Apply these rules IN ORDER:

```text
1. IF iteration < 3 (minimum):
   → MANDATORY LOOP to Phase 0 (Research)
   → Do more research before continuing

2. ELIF P0 OR P1 OR P2 insights exist:
   → LOOP BACK to Phase 0 (Research)
   → Research solutions, then complete ALL insights

3. ELIF no new insights for 2 consecutive iterations AND iteration >= 3:
   → PLATEAU DETECTED: EXIT

4. ELIF iteration >= 5:
   → MAX ITERATIONS: EXIT (document remaining work)

5. ELSE:
   → EXIT: All work complete, upgrade is polished
```

**Decision**: [LOOP_TO_PHASE_0 | EXIT]
**Reason**: [explanation]

## Step 4: Finalization Process (REQUIRED before EXIT)

**If exiting, complete this thorough finalization checklist:**

### 4a. Final Bug Finding
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Check for syntax errors in all modified files
- [ ] Review any TODO/FIXME comments added during implementation
- [ ] Verify no debug prints or temporary code left behind

### 4b. Missing Files & Dependencies Check
- [ ] All files mentioned in upgrade checklist exist
- [ ] All new imports work (no ImportError)
- [ ] New dependencies added to `requirements.txt` if needed
- [ ] No circular import issues

### 4c. Test Coverage
- [ ] Tests exist for all new functionality
- [ ] Tests cover edge cases identified during implementation
- [ ] All tests pass: `pytest tests/ -v --tb=short`
- [ ] Coverage meets minimum threshold (>70%)

### 4d. Documentation Update
- [ ] Upgrade research document marked complete with changelog
- [ ] CLAUDE.md updated if workflow/commands changed
- [ ] README.md updated if user-facing features added
- [ ] Docstrings added to new functions/classes
- [ ] Upgrade index updated with new entry

### 4e. Cross-Reference Verification
- [ ] All files that reference the changed feature are updated
- [ ] Config files match code expectations
- [ ] Related prompts/commands reference correct phase numbers
- [ ] No stale references to old implementations

### 4f. Final Cleanup
- [ ] Remove any temporary/debug files
- [ ] Git status clean (all changes committed)
- [ ] No merge conflicts pending
- [ ] Branch ready for merge if applicable

### 4g. Integration Check (Scope-Appropriate)

**Determine upgrade scope:**
- **SUB-UPGRADE** (UPGRADE-XXX.Y): Basic integration
- **MAJOR UPGRADE** (UPGRADE-XXX): Complete integration

**For SUB-UPGRADES (basic):**
- [ ] Changed feature works with immediate dependencies
- [ ] Affected tests pass (not necessarily full suite)
- [ ] No import errors in modified files
- [ ] Config files match code expectations

**For MAJOR UPGRADES (complete):**
- [ ] Full test suite passes: `pytest tests/ -v`
- [ ] Cross-component integration verified
- [ ] End-to-end workflow tested
- [ ] Architecture alignment reviewed
- [ ] Documentation reflects holistic vision

## Step 5: Exit Report

```text
Completion Status:
- Total iterations: [N]
- P0 insights: [X addressed]
- P1 insights: [X addressed]
- P2 insights: [X addressed]
- Exit reason: [COMPLETE | PLATEAU | MAX_ITER]

Finalization Checklist:
- Bug finding: [PASS/FAIL]
- Missing files check: [PASS/FAIL]
- Test coverage: [PASS/FAIL - X%]
- Documentation: [PASS/FAIL]
- Cross-references: [PASS/FAIL]
- Cleanup: [PASS/FAIL]
- Integration: [PASS/FAIL - BASIC|COMPLETE]

Quality Assessment:
- Is this upgrade production-ready? [Yes/No]
- What makes it polished? [list features]

Files Modified/Created:
- [list all files touched in this upgrade]

Next Steps (if any):
- [deferred items for future work]
```

---

## Decision Actions (v3.0)

### If LOOP_TO_PHASE_0 → Return to Phase 0 (Research)

**ALL loops go to Phase 0 first for additional research:**

1. Identify what research is needed for the remaining P0/P1/P2 insights
2. Conduct online research with timestamps
3. **IMMEDIATELY** write findings to research file (compaction protection)
4. Return to Phase 1 to plan implementation
5. Then proceed through phases 2-7 again

### If EXIT

1. Complete Step 4 Finalization checklist
2. Generate Exit Report (Step 5)
3. Create final checkpoint commit
4. Mark upgrade as complete in research document
5. Update IMPLEMENTATION_TRACKER.md if applicable

---

## CRITICAL: Iteration Limits

| Limit | Value | Rationale |
|-------|-------|-----------|
| Minimum | 3 iterations | Ensures polished output |
| Maximum | 5 iterations | Prevents infinite loops |
| Plateau | 2 consecutive | Exit when no new insights for 2 loops |

**If at iteration 5 and insights remain**: You MUST exit with a detailed report explaining what would be needed to complete and recommendations for next steps.

---

## Session Complete

**If LOOP_TO_PHASE_0**:
```bash
# Start next iteration
python3 .claude/hooks/ric_state_manager.py advance
# This loops back to Phase 0 if insights remain
```

**If EXIT allowed**:
```bash
# Verify exit eligibility
python3 .claude/hooks/ric_state_manager.py can-exit
# Should show "Can exit: True"
```

Mark RIC Loop session as complete and update TodoWrite accordingly.
