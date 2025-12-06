# RIC Loop Phase Management

Manage phase transitions in your Meta-RIC Loop session.

## Current State

First, check your current state:

```bash
python3 .claude/hooks/ric_state_manager.py status
```

## Phase Commands

### Advance to Next Phase

```bash
python3 .claude/hooks/ric_state_manager.py advance
```

This will:
1. Record completion of current phase
2. Move to the next phase
3. If at Phase 7 (Integration), handle loop decision

### Get Phase Header

After advancing, output the phase header:

```
[ITERATION X/5] === PHASE N: NAME ===
```

Where:
- X = current iteration (1-5)
- N = phase number (0-7)
- NAME = phase name in caps

## Phase Reference

| Phase | Name | Purpose | Key Deliverables |
|-------|------|---------|------------------|
| 0 | Research | Online research | Timestamped findings in ric-progress.md |
| 1 | Upgrade Path | Define target state | Scope, success criteria |
| 2 | Checklist | Break into tasks | P0/P1/P2 classified tasks |
| 3 | Coding | Implement changes | One file per commit |
| 4 | Double-Check | Verify completeness | Tests pass, >70% coverage |
| 5 | Introspection | Find gaps | Missing files, bugs, debt |
| 6 | Metacognition | Classify insights | P0/P1/P2 insights |
| 7 | Integration | Loop decision | Exit or loop to Phase 0 |

## Phase-Specific Actions

### Phase 0: Research
- Use WebSearch for each query
- Document findings IMMEDIATELY (compaction protection)
- Include timestamps on all sources

### Phase 1: Upgrade Path
- Define what success looks like
- Set clear scope boundaries
- List measurable success criteria

### Phase 2: Checklist
- Classify ALL tasks as P0/P1/P2
- P0 = Critical (blocks exit)
- P1 = Important (should complete)
- P2 = Polish (STILL REQUIRED)

### Phase 3: Coding
- ONE file per commit
- Write tests for each change
- Use commit message format: `[ITERATION X/5] [single-component] message`

### Phase 4: Double-Check
- Run full test suite
- Verify >70% coverage
- Check for lint errors

### Phase 5: Introspection
- Find missing files
- Identify potential bugs
- Note technical debt

### Phase 6: Metacognition
- Add insights with priority:
```bash
python3 .claude/hooks/ric_state_manager.py add-insight --priority P0 --description "..."
```

### Phase 7: Integration
- Check exit eligibility:
```bash
python3 .claude/hooks/ric_state_manager.py can-exit
```
- If can exit: Complete finalization checklist
- If cannot exit: Loop back to Phase 0

## Command Arguments

If $ARGUMENTS provided:

- `advance` - Advance to next phase
- `status` - Show current state
- `check` - Verify gate criteria for current phase
- `reset` - Reset to Phase 0 (new iteration)

Run the appropriate command based on: $ARGUMENTS
