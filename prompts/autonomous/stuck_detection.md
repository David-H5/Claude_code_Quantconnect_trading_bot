# Stuck Detection Protocol (P0 - UPGRADE-013)

You are working autonomously and need to monitor for stuck states.

## Signs You May Be Stuck

### Pattern 1: Circular Reasoning
- Repeatedly considering the same options without choosing
- Revisiting decisions already made
- Asking the same questions multiple times

**Detection**: If you recognize you're reconsidering something you already decided, STOP.

### Pattern 2: No File Changes
- More than 15 minutes of reasoning without any tool calls
- Multiple "planning" cycles without execution
- Reading the same files repeatedly

**Detection**: If you haven't modified or created any files in the last 3-4 reasoning blocks, STOP.

### Pattern 3: Repeating Actions
- Running the same command that just failed
- Making the same edit that was just reverted
- Retrying without changing approach

**Detection**: If the same action appears 3+ times with same result, STOP.

### Pattern 4: Error Loop
- Fixing an error creates a new error
- New error fix re-introduces original error
- Oscillating between two broken states

**Detection**: If errors are cycling (A → B → A), STOP.

### Pattern 5: Scope Creep Loop
- Task keeps expanding with each step
- "One more thing" pattern repeating
- Never reaching completion

**Detection**: If task scope has grown 3x from original, STOP.

---

## When You Detect Stuck State

### Step 1: Declare Stuck State

Log clearly:
```
STUCK STATE DETECTED
- Pattern: [which pattern from above]
- Evidence: [what triggered detection]
- Time in current state: [estimate]
- Progress made: [what was accomplished before stuck]
```

### Step 2: Preserve State

Before any recovery action:
1. Commit current changes (even partial): `git add -A && git commit -m "WIP: checkpoint before stuck recovery"`
2. Update claude-progress.txt with current state
3. Update claude-session-notes.md with what you've learned

### Step 3: Enter Fallback Protocol

**See**: [Graceful Fallback Protocol](graceful_fallback.md)

Choose from fallback hierarchy:
1. Try alternative approach (most common)
2. Simplify scope
3. Skip and document
4. Escalate to human

---

## Prevention Strategies

### During Long Tasks

Every 15-20 minutes, do a quick self-check:

```
PROGRESS CHECK:
- [ ] Have I created/modified files since last check?
- [ ] Have tests/lints improved or stayed same?
- [ ] Is my current approach different from 20 min ago?
- [ ] Am I closer to the goal than before?

If 2+ boxes unchecked → Evaluate if stuck
```

### Before Starting Complex Work

Ask yourself:
- What does "done" look like?
- What's my exit criteria if this doesn't work?
- What's my time budget for this attempt?

### After Errors

Before retrying:
- What specifically failed?
- Is my retry actually different?
- Should I try a different approach instead?

---

## Integration with RIC Loop

This protocol is **complementary** to Meta-RIC Loop:
- RIC Loop handles task-level iteration (max 5)
- Stuck Detection handles moment-to-moment monitoring
- Both work together for robust autonomous operation

**When to use which**:
- RIC iteration limit reached → Exit to Phase 7
- Stuck pattern detected → Enter Fallback Protocol first
- After fallback, continue RIC if progress resumed

---

## Examples

### Example 1: Circular Reasoning
```
User asked: "Add error handling to the API"

BAD (stuck pattern):
- "Should I use try/except or error codes?"
- "Let me think about try/except vs error codes..."
- "Actually, should I use try/except or error codes?"
[CIRCULAR - same question 3 times]

GOOD (detect and break):
- "Should I use try/except or error codes?"
- "DECISION: Using try/except (Python idiomatic)"
- [Execute the decision]
```

### Example 2: Error Loop
```
BAD (stuck pattern):
- Add import → NameError
- Fix NameError → ImportError
- Fix ImportError → NameError (same original error!)
[ERROR LOOP - cycling between same two errors]

GOOD (detect and break):
- STUCK DETECTED: Error loop A → B → A
- FALLBACK: Step back, check if module even exists
- RESOLUTION: Wrong module name, fixed at source
```

### Example 3: No Progress
```
BAD (stuck pattern):
- Read file A
- Read file B
- "Let me think about the architecture..."
- Read file A again
- "Considering the options..."
[NO CHANGES for 20+ minutes]

GOOD (detect and break):
- PROGRESS CHECK: No files created/modified
- ACTION: Make smallest possible change to start
- "Creating skeleton of function first, will refine"
```
