# Meta-RIC Loop Phases 5-6: Introspection & Metacognition (v3.0)

You are executing Phases 5 and 6 of the Meta-RIC Loop v3.0.

## Current Context

$ARGUMENTS

## Phase 5: Introspection Mode

### 5.1 Systematic Completeness Check

Complete ALL categories:

```markdown
## Introspection Report

### Date: [YYYY-MM-DD]
### Iteration: [N]

### A. Missing Files Check
| Expected File | Exists? | Status | Notes |
|---------------|---------|--------|-------|
| [file.py] | [Yes/No] | [Complete/Partial/Missing] | [Notes] |

### B. Bug Detection
| Location | Bug Type | Severity | Status |
|----------|----------|----------|--------|
| [file:line] | [Type] | [P0/P1/P2] | [Open/Fixed] |

### C. Code Health Check
- [ ] All imports resolve
- [ ] All tests pass
- [ ] No linting errors
- [ ] No type errors
- [ ] No circular dependencies

### D. Incomplete Features
| Feature | Completion % | Missing Parts | Priority |
|---------|--------------|---------------|----------|
| [Feature] | [XX%] | [What's missing] | [P0/P1/P2] |
```

### 5.2 Feature Expansion Prompts

Answer EACH of these questions:

1. **Adjacent Features**: "What related functionality would users need next?"
2. **Edge Cases**: "What edge cases haven't been handled?"
3. **Integration Points**: "What else could this connect to?"
4. **Performance**: "Where could this be optimized?"
5. **Usability**: "How could this be easier to use?"

Document expansion ideas:

```markdown
### E. Expansion Ideas
| Idea | Type | Effort | Value | Priority |
|------|------|--------|-------|----------|
| [Idea] | [Enhancement/Feature/Refactor] | [S/M/L] | [Description] | [P0/P1/P2] |
```

---

## Phase 6: Metacognition with Classified Insights (v3.0)

### 6.1 Five Reflection Questions

Answer ALL of these with confidence scores:

```markdown
## Metacognitive Reflection

### 1. Did implementation achieve research goals? [Confidence: X%]
- [Assessment]

### 2. Are there known gaps in the approach? [Confidence: X%]
- [Gap 1]
- [Gap 2]

### 3. What assumptions need validation? [Confidence: X%]
- [Assumption 1]: [Could be wrong because...]
- [Assumption 2]: [Could be wrong because...]

### 4. What would I do differently? [Confidence: X%]
- [Change 1]: [Why]
- [Change 2]: [Why]

### 5. Is the implementation maintainable? [Confidence: X%]
- [Assessment]
```

### 6.2 CRITICAL: Classified Insights (REQUIRED)

**Create a table of ALL insights discovered in this iteration:**

```markdown
### Classified Insights

| ID | Insight | Classification | Action Required |
|----|---------|----------------|-----------------|
| I1 | [Description] | [Classification] | [Specific action] |
| I2 | [Description] | [Classification] | [Specific action] |
| I3 | [Description] | [Classification] | [Specific action] |
```

**Classification Legend:**

- **P0-GAP**: Critical gap that blocks completion
- **P0-BUG**: Bug that must be fixed
- **P1-DEBT**: Technical debt that should be addressed
- **P1-IMPROVEMENT**: Improvement worth making
- **P2-POLISH**: Nice to have, makes it production-ready
- **P2-ENHANCEMENT**: Future feature idea
- **ADDRESSED**: Already handled in this iteration

**IMPORTANT**: With AI assistance, ALL P0-P2 items should be completed. P2 items make the upgrade polished and production-ready.

### 6.3 Reasoning Quality Check

```markdown
### Reasoning Quality Self-Assessment
- [ ] Did I follow the single-component principle?
- [ ] Did I checkpoint appropriately?
- [ ] Did I attribute changes correctly?
- [ ] Did I miss any obvious solutions?
- [ ] Am I over-engineering?
- [ ] Am I under-engineering?
```

---

## Gate: Proceed to Phase 7 When

- [ ] All 5 introspection categories completed
- [ ] All 5 metacognition questions answered with confidence scores
- [ ] **Classified Insights table created with P0/P1/P2 classifications**
- [ ] Reasoning quality self-assessed

## Next Step

Proceed to Phase 7 (Integration & Decision) by running `/ric-converge`.

Phase 7 will:
1. Triage all insights from this phase
2. Apply loop decision rules (ALL loops go to Phase 0 for more research, or exit)
3. If exiting, complete finalization checklist (Steps 4a-4g)
4. Generate exit report
