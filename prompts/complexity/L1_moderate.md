# Moderate Task Workflow (L2)

You are working on a **moderate complexity task** that requires planning before execution.

## Approach

1. **Plan**: Create a brief plan (3-5 steps)
2. **Execute**: Implement ONE component at a time
3. **Verify**: Test after each component
4. **Checkpoint**: Commit after each component
5. **Review**: Double-check completeness

## Guidelines

- Plan before coding, but keep the plan concise
- Modify ONE file per change when possible
- Commit after each logical unit of work
- If the task becomes more complex, escalate to COMPLEX workflow (full RIC loop)

## Escalation Triggers (UPGRADE-013)

**Escalate to L3 (Complex/Meta-RIC) when ANY of:**

| Trigger | Detection | Action |
|---------|-----------|--------|
| **Time Overflow** | Task taking 2x+ expected time | Stop, escalate to L3 for research phase |
| **Multiple Blockers** | 3+ unexpected issues encountered | Stop, escalate for deeper analysis |
| **Scope Creep** | New requirements discovered mid-task | Stop, replan with full RIC loop |
| **Research Needed** | Need to look up external docs/patterns | Stop, escalate for Phase 0 research |
| **Multi-Module Impact** | Change affects 5+ files | Stop, escalate for architectural review |

**How to escalate:**

1. Commit current progress: `git commit -m "WIP: escalating to L3"`
2. Document what was learned
3. Start L3 workflow with Phase 0 research
4. Reference the moderate task that led to escalation

**When NOT to escalate:**

- Single error that has a clear fix
- Expected complexity within plan
- Minor scope adjustment (1 additional step)

## Workflow with Checkpoint Gates

```text
Task → Plan (3-5 steps) → [GATE 1] → Execute Step 1 → Verify → [GATE 2] → Commit
                                           ↓
                                    Execute Step 2 → Verify → [GATE 2] → Commit
                                           ↓
                                         ...
                                           ↓
                                    [GATE 3] → Final Review → Done
```

### GATE 1: Plan Validation
**STOP and verify before proceeding:**
- [ ] Plan has 3+ concrete steps
- [ ] Each step modifies at most 2-3 files
- [ ] Success criteria defined for each step
- [ ] No step requires research (escalate to L3 if so)

### GATE 2: Step Completion
**STOP and verify before moving to next step:**
- [ ] Code change is complete and functional
- [ ] Tests written OR existing tests updated
- [ ] Tests pass locally
- [ ] Git commit made with descriptive message

### GATE 3: Task Completion
**STOP and verify before marking done:**
- [ ] All planned steps completed
- [ ] All tests pass
- [ ] No obvious gaps in implementation
- [ ] Documentation updated if API changed

## Planning Template

Before starting, outline your plan:

1. **Step 1**: [What you'll do first]
2. **Step 2**: [What you'll do second]
3. **Step 3**: [What you'll do third]
4. **Tests**: [How you'll verify]

## Quality Checks

- [ ] Plan is clear and achievable
- [ ] Each step is independently verifiable
- [ ] Tests cover new functionality
- [ ] Documentation updated if needed
- [ ] All checkpoint commits made
- [ ] All gates passed
