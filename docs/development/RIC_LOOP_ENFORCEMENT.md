# RIC Loop Enforcement System (v5.1 Guardian)

> **Current Implementation**: `.claude/hooks/ric_v50.py`
>
> **Quick Reference**: `.claude/RIC_CONTEXT.md`
>
> This document describes how the RIC v5.1 Guardian enforcement system works.

## Overview

The RIC (Research-Implement-Converge) Loop Enforcement System ensures that complex tasks follow a structured methodology with minimum 3 iterations. The v5.1 Guardian system is a standalone implementation (~6700 lines) that consolidates all enforcement logic into a single file.

## Why Enforcement?

Without enforcement, autonomous sessions tend to:
- Skip Phase 0 (Research) - jumping straight to coding
- Skip Phase 4 (Reflect) - missing gap analysis
- Complete only 1 iteration instead of minimum 3
- Batch commits instead of single-component changes
- Exit without resolving P0/P1/P2 insights (treating them as optional)

## System Components

### 1. RIC v5.0 Guardian (`ric_v50.py`)

**Location**: `.claude/hooks/ric_v50.py`

The single consolidated implementation containing:
- State management
- Hook handlers (PreToolUse, UserPromptSubmit)
- Phase enforcement
- Convergence detection
- Safety throttles
- Decision tracing

**Commands**:
```bash
# Initialize new session
python3 .claude/hooks/ric_v50.py init 5

# Check current state
python3 .claude/hooks/ric_v50.py status

# Advance to next phase
python3 .claude/hooks/ric_v50.py advance

# Add insight with priority
python3 .claude/hooks/ric_v50.py add-insight P0 "Missing error handling"

# Resolve insight
python3 .claude/hooks/ric_v50.py resolve INS-001 "Added try/catch"

# Check exit eligibility
python3 .claude/hooks/ric_v50.py can-exit
```

**State File**: `.claude/ric_state.json`

### 2. Quick Reference (`RIC_CONTEXT.md`)

**Location**: `.claude/RIC_CONTEXT.md`

Compact reference for workflow rules and phase definitions.

### 3. Session Hooks

| Hook | File | Purpose |
|------|------|---------|
| PreToolUse | `ric_v50.py` | Compliance checking before edits |
| UserPromptSubmit | `ric_v50.py` | Status display and enforcement |
| SessionStart | `ric_v50.py` | State display on session start |
| SessionStop | `session_stop_v2.py` | Exit validation (continuous mode) |

### 4. Slash Commands

| Command | Description |
|---------|-------------|
| `/ric-start <task>` | Initialize RIC session with templates |
| `/ric-research <topic>` | Phase 0 research protocol |
| `/ric-introspect` | Run Phase 4 introspection |
| `/ric-converge` | Check convergence and exit eligibility |

## RIC Loop Phases (v5.0 - 5 Phases)

| Phase | Name | Purpose | Gate Criteria |
|-------|------|---------|---------------|
| **P0** | Research | Online research | 3+ timestamped sources |
| **P1** | Plan | Define scope | P0/P1/P2/[OUT] tasks |
| **P2** | Build | Implementation | Atomic commits (1-5 files) |
| **P3** | Verify | Quality gates | Tests pass, >70% coverage |
| **P4** | Reflect | Gap analysis | Insights classified |

## Exit Rules

```text
1. IF iteration < 3         → LOOP (minimum 3 iterations)
2. ELIF P0 insights > 0     → LOOP (critical gaps)
3. ELIF P1 insights > 0     → LOOP (important items)
4. ELIF P2 insights > 0     → LOOP (P2 is REQUIRED)
5. ELIF iteration >= 5      → FORCED EXIT (max reached)
6. ELSE                     → EXIT ALLOWED
```

## Priority Classification

| Priority | Description | Exit Requirement |
|----------|-------------|------------------|
| **[P0]** | Critical - blocks functionality | MUST resolve |
| **[P1]** | Important - affects quality | MUST resolve |
| **[P2]** | Polish - improves UX | MUST resolve (not optional!) |
| **[OUT]** | Out of Scope | Explicitly excluded |

## v5.0 Safety Features

| Feature | Description |
|---------|-------------|
| **Hallucination Detection** | 5-category taxonomy check before commits |
| **Convergence Detection** | Multi-metric tracking (insight rate, fix success, churn) |
| **Confidence Calibration** | Per-phase 0-100% confidence ratings |
| **Safety Throttles** | Tool call limits, time limits, failure limits |
| **Decision Tracing** | Structured JSON logs for meta-debugging |
| **Research Enforcement** | Timestamp validation, auto-persist every 3 searches |
| **Drift Detection** | Scope creep alerts >20% expansion (AEGIS) |
| **Guardian Mode** | Independent verification pass (Gartner 2025) |

## Usage Examples

### Starting a New RIC Session

```bash
# Via slash command (recommended)
/ric-start "Implement new feature"

# Via CLI
python3 .claude/hooks/ric_v50.py init 5

# Via overnight script
./scripts/run_overnight.sh --continuous --with-recovery "Implement feature"
```

### During Development

```bash
# Check state before major changes
python3 .claude/hooks/ric_v50.py status

# Record phase actions
python3 .claude/hooks/ric_v50.py record-search "query"
python3 .claude/hooks/ric_v50.py record-findings

# Advance phase when ready
python3 .claude/hooks/ric_v50.py advance
```

### Managing Insights

```bash
# Add insights by priority
python3 .claude/hooks/ric_v50.py add-insight P0 "Missing input validation"
python3 .claude/hooks/ric_v50.py add-insight P1 "Add retry logic"
python3 .claude/hooks/ric_v50.py add-insight P2 "Clean up imports"

# Resolve insights
python3 .claude/hooks/ric_v50.py resolve INS-001 "Added validation in api/validators.py"

# List all insights
python3 .claude/hooks/ric_v50.py insights
```

### Checking Exit Eligibility

```bash
python3 .claude/hooks/ric_v50.py can-exit
# Output: EXIT: ALLOWED or EXIT: BLOCKED (reason)
```

## Troubleshooting

### "No active RIC session"

Initialize a session first:
```bash
python3 .claude/hooks/ric_v50.py init 5
```

### Session won't exit

Check why:
```bash
python3 .claude/hooks/ric_v50.py status
python3 .claude/hooks/ric_v50.py can-exit
```

Common blockers:
- Iteration < 3 (need more iterations)
- Open P0/P1/P2 insights (need to resolve)
- Phase < 4 (need to reach Reflect phase)

### Clear RIC state

To reset and start fresh:
```bash
python3 .claude/hooks/ric_v50.py end
# or manually
rm -f .claude/ric_state.json
```

## Configuration Reference

### settings.json Hooks

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Edit|Write|Bash",
      "hooks": [{
        "type": "command",
        "command": "python3 .claude/hooks/ric_v50.py",
        "timeout": 10
      }]
    }],
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "python3 .claude/hooks/ric_v50.py",
        "timeout": 10
      }]
    }]
  }
}
```

## Core Files

| File | Purpose |
|------|---------|
| `.claude/hooks/ric_v50.py` | **Main implementation** (~6700 lines) |
| `.claude/RIC_CONTEXT.md` | Quick reference |
| `.claude/ric_state.json` | State persistence |
| `.claude/RIC_NOTES.md` | Structured memory (auto-generated) |

## Related Documentation

- [RIC Quick Reference](./.claude/RIC_CONTEXT.md) - Compact workflow guide
- [RIC v5.0 Research](../research/RIC-V46-RESEARCH-DEC2025.md) - Source research
- [RIC v5.1 Quality Gates](../research/RIC-V51-QUALITY-GATES-RESEARCH.md) - Latest research
- [Overnight Sessions](../autonomous-agents/README.md) - Autonomous operation

## Deprecated Files

The following files have been superseded by `ric_v50.py`:
- `.claude/hooks/deprecated/ric_v45.py` - v4.5 implementation
- `.claude/hooks/deprecated/ric_v50_dev.py` - Development container
- `.claude/hooks/deprecated/ric_state_manager.py` - Old state manager
- `.claude/hooks/deprecated/ric_prompts.py` - Old prompts
- `.claude/hooks/deprecated/ric_hooks.py` - Old hooks
- `.claude/hooks/deprecated/ric_enforcer.py` - Old enforcer
- `.claude/hooks/deprecated/enforce_ric_compliance.py` - Old compliance
