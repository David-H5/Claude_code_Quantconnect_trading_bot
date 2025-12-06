# Start Meta-RIC Loop Session (v3.1)

You are starting a new Meta-RIC Loop session for a complex task.

## MANDATORY: Initialize RIC State

**FIRST**, initialize the RIC Loop state using the state manager:

```bash
python3 .claude/hooks/ric_state_manager.py init --upgrade-id "$ARGUMENTS" --title "RIC Loop Session"
```

If $ARGUMENTS is empty, ask the user for an upgrade ID (e.g., UPGRADE-016, FEATURE-001).

## Quick Reference Files

- `.claude/RIC_CONTEXT.md` - Quick reference with 7 phases
- `docs/development/ENHANCED_RIC_WORKFLOW.md` - Full guide
- `ric-progress.md` - Your session tracking file (auto-created)
- `.claude/state/ric.json` - State persistence (auto-managed)

## Task Description

The user wants to work on: $ARGUMENTS

## Step 1: Verify RIC Loop is Required

**USE Meta-RIC Loop** if ANY of these apply:
- Multi-file changes required
- Research needed (uncertain approach)
- Architectural decisions involved
- Complex bug with unknown root cause
- Upgrade work (any UPGRADE-XXX reference)
- Major trade-offs to evaluate

**SKIP** if ALL of these apply:
- Single file change
- Known solution
- Documentation only
- Configuration only

If skipping, run: `python3 .claude/hooks/ric_state_manager.py status` and note "No active session".

## Step 2: Begin Phase 0 (Research)

After initialization, you are in Phase 0 - Research.

**Output the phase header**:
```
[ITERATION 1/5] === PHASE 0: RESEARCH ===
```

**Research Protocol**:
1. Identify 3-5 search queries related to the task
2. Execute each search with WebSearch tool
3. **IMMEDIATELY** document findings in `ric-progress.md` (compaction protection!)
4. Include timestamps: Search Date AND Source Publication Date
5. After all searches, move to Phase 1

**Research Template** (add to ric-progress.md):
```markdown
## Phase 0: Research

### Search 1
**Search Date**: [Current date/time]
**Query**: "[query]"
**Sources**:
- [Source Title (Published: Month Year)](URL)

**Key Findings**:
- Finding 1
- Finding 2

**Applied**: [How this will be used]
```

## Step 3: Advance Phases

Use the state manager to advance phases:

```bash
# Check current state
python3 .claude/hooks/ric_state_manager.py status

# Advance to next phase
python3 .claude/hooks/ric_state_manager.py advance
```

## Step 4: Track Insights

During Phases 5-6, add classified insights:

```bash
# Add a P0 (critical) insight
python3 .claude/hooks/ric_state_manager.py add-insight --priority P0 --description "Missing error handling"

# Add a P1 (important) insight
python3 .claude/hooks/ric_state_manager.py add-insight --priority P1 --description "Should add retry logic"

# Add a P2 (polish) insight - STILL REQUIRED!
python3 .claude/hooks/ric_state_manager.py add-insight --priority P2 --description "Clean up comments"
```

## Step 5: Resolve Insights

Before exit, resolve all insights:

```bash
python3 .claude/hooks/ric_state_manager.py resolve --insight-id INS-01-001 --resolution "Added try/catch in broker_server.py"
```

## Step 6: Check Exit Eligibility

```bash
python3 .claude/hooks/ric_state_manager.py can-exit
```

Exit is allowed ONLY when:
- Iteration >= 3 (minimum)
- All P0 insights resolved
- All P1 insights resolved
- All P2 insights resolved (P2 is REQUIRED!)

## CRITICAL RULES (v3.1)

1. **INITIALIZE FIRST**: Always run `ric_state_manager.py init` before starting
2. **STRICT SEQUENTIAL EXECUTION**: Phases 0→1→2→3→4→5→6→7 in order
3. **LOG EVERY PHASE**: Format: `[ITERATION X/5] === PHASE N: NAME ===`
4. **Single-Component Changes**: In Phase 3, modify ONE file per commit
5. **Checkpoint After Each Change**: `git commit` after every change
6. **Complete ALL P0-P2**: P2 is REQUIRED, not optional
7. **Minimum 3 Iterations**: Ensures thoroughness
8. **ALL Loops Go to Phase 0**: Research first, then implement
9. **Compaction Protection**: IMMEDIATELY write research to file
10. **Use State Manager**: Track state with ric_state_manager.py

## Loop Decision Rules (Phase 7)

```text
1. IF iteration < 3 → MANDATORY LOOP to Phase 0
2. ELIF any P0 insights open → LOOP to Phase 0
3. ELIF any P1 insights open → LOOP to Phase 0
4. ELIF any P2 insights open → LOOP to Phase 0 (P2 is REQUIRED!)
5. ELIF iteration >= 5 → FORCED EXIT (max iterations)
6. ELSE → EXIT ALLOWED
```

## Environment Variables

Set these for overnight sessions:
```bash
export RIC_MODE=ENFORCED    # ENFORCED | SUGGESTED | DISABLED
export CONTINUOUS_MODE=1    # Enable work-until-complete
```

## Begin Now

1. Initialize state with the command above
2. Check `python3 .claude/hooks/ric_state_manager.py status`
3. Output `[ITERATION 1/5] === PHASE 0: RESEARCH ===`
4. Begin your research searches
