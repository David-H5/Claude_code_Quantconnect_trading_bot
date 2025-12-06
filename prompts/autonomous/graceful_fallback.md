# Graceful Fallback Protocol (P0 - UPGRADE-013)

When you detect a stuck state or blocker, follow this fallback hierarchy.

## Fallback Hierarchy

```
Blocker Detected
      │
      ▼
┌─────────────────┐
│ LEVEL 1:        │ ← Try first (90% of cases)
│ Alternative     │
│ Approach        │
└────────┬────────┘
         │ Failed after 2 attempts?
         ▼
┌─────────────────┐
│ LEVEL 2:        │ ← Reduce scope
│ Simplify        │
│ Scope           │
└────────┬────────┘
         │ Still blocked?
         ▼
┌─────────────────┐
│ LEVEL 3:        │ ← Move on
│ Skip &          │
│ Document        │
└────────┬────────┘
         │ Critical blocker?
         ▼
┌─────────────────┐
│ LEVEL 4:        │ ← Rare, last resort
│ Escalate        │
│ to Human        │
└─────────────────┘
```

---

## Level 1: Alternative Approach

**When**: First fallback when current approach isn't working

**Actions**:
1. Identify what specifically is failing
2. List 2-3 alternative approaches
3. Choose the most different one from current
4. Try for max 2 attempts

**Time Budget**: 15-20 minutes per alternative

**Examples**:

| Blocker | Alternative Approaches |
|---------|----------------------|
| Import not found | 1. Check spelling 2. Find correct module 3. Use different library |
| Test failing | 1. Fix test 2. Fix code 3. Mock the dependency |
| API error | 1. Check auth 2. Try different endpoint 3. Use fallback service |
| Complex refactor | 1. Smaller steps 2. Copy-modify-delete pattern 3. Feature flag |

**Template**:
```
LEVEL 1 FALLBACK:
Current approach: [what isn't working]
Why it's failing: [root cause if known]

Alternative 1: [description]
Alternative 2: [description]

Trying Alternative 1...
```

---

## Level 2: Simplify Scope

**When**: Alternative approaches also failed

**Actions**:
1. Identify the core requirement
2. Remove optional features/edge cases
3. Implement minimal viable version
4. Note deferred items for later

**Time Budget**: 10-15 minutes

**Simplification Strategies**:

| Original Task | Simplified Version |
|--------------|-------------------|
| Full error handling | Handle most common error only |
| All test cases | Core happy path only |
| Configurable feature | Hardcoded first, configure later |
| Multi-file refactor | Single file first |
| Complete integration | Stub/mock for now |

**Template**:
```
LEVEL 2 FALLBACK - Simplifying Scope:
Original scope: [full description]
Core requirement: [essential part only]

Simplifications:
- Removing: [feature/case 1] - reason: [why deferrable]
- Removing: [feature/case 2] - reason: [why deferrable]

Proceeding with minimal version...

DEFERRED (document in session notes):
- [ ] [Deferred item 1]
- [ ] [Deferred item 2]
```

---

## Level 3: Skip & Document

**When**: Task is blocked but not critical to session goals

**Actions**:
1. Document what was attempted
2. Document why it's blocked
3. Document what needs to happen for unblocking
4. Add to claude-progress.txt as blocked
5. Move to next task

**Time Budget**: 5 minutes (just documentation)

**What to Document**:
```markdown
## Blocked Task: [Task Name]

**Status**: Skipped (blocked)
**Time Spent**: [X minutes]

### What Was Tried
1. [Approach 1] - Result: [what happened]
2. [Approach 2] - Result: [what happened]

### Why It's Blocked
[Root cause if known, otherwise symptoms]

### Unblocking Requirements
- [ ] [What needs to happen to unblock]
- [ ] [Dependencies that need resolution]

### Recommendation
[Should human address? Wait for upstream fix? Defer to later sprint?]
```

**Add to progress file**:
```
## Blockers
- [Task Name]: [One-line description of block]
```

---

## Level 4: Escalate to Human

**When**: Critical blocker that prevents session progress

**Use ONLY for**:
- Security-sensitive operations needing human approval
- Production deployments/changes
- Irreversible operations
- External system access issues (auth, API keys)
- Decisions requiring business context

**Actions**:
1. Document everything attempted
2. Clear description of what's needed
3. Update progress file with ESCALATION block
4. If continuous mode: exit gracefully

**Template**:
```
ESCALATION REQUIRED

**Task**: [What you were trying to do]
**Blocker**: [What's preventing progress]
**Impact**: [What can't proceed without this]

**Attempted Solutions**:
1. [Approach] - [Result]
2. [Approach] - [Result]

**Human Action Needed**:
[Specific action requested]

**Session Status**:
Can continue with other tasks: [Yes/No]
Tasks that can proceed: [List if yes]
```

---

## Fallback Decision Matrix

Use this to decide which level to jump to:

| Situation | Recommended Level |
|-----------|------------------|
| "This approach isn't working" | Level 1 |
| "I've tried 3+ approaches" | Level 2 |
| "This task is blocked by external factor" | Level 3 |
| "This requires human decision/auth" | Level 4 |
| "Time budget exceeded for this task" | Level 2 or 3 |
| "This would take 10x expected time" | Level 2 |
| "Security-sensitive operation" | Level 4 |

---

## Preserving Progress

**Before any fallback**, ensure:

1. **Commit changes**: Even partial work
   ```bash
   git add -A && git commit -m "WIP: [task] - entering fallback"
   ```

2. **Update progress file**: Mark what's done, what's blocked

3. **Update session notes**: Document learnings
   ```markdown
   ## Important Discoveries
   - [What you learned that might help future attempts]
   ```

---

## Integration with Stuck Detection

```
Stuck Detected (stuck_detection.md)
         │
         ▼
   Enter Fallback Protocol
         │
    ┌────┴────┐
    │ Success │ → Resume normal work
    └────┬────┘
         │ All levels exhausted?
         ▼
   Document & Continue
   with other tasks
```

---

## Anti-Patterns to Avoid

### DON'T: Infinite Retry
```
# BAD
while not working:
    try_same_thing()
```

### DO: Bounded Fallback
```
# GOOD
for attempt in range(2):
    try_alternative[attempt]()
if still_blocked:
    simplify_scope()
```

### DON'T: Silent Failure
```
# BAD
if blocked:
    pass  # Just move on
```

### DO: Document & Defer
```
# GOOD
if blocked:
    document_blocker()
    add_to_deferred_list()
    notify_via_progress_file()
```

### DON'T: Premature Escalation
```
# BAD (first obstacle → escalate)
if error:
    escalate_to_human()
```

### DO: Exhaust Alternatives First
```
# GOOD
if error:
    try_alternative()
    if still_error:
        simplify()
        if still_error:
            skip_and_document()
```
