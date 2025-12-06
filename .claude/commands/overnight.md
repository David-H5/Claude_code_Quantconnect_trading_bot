# Overnight Autonomous Session Setup

You are preparing to start or continue an autonomous overnight development session.

## Activation Keywords

This command can be triggered by:
- `/overnight` - Start overnight session setup
- "start overnight session" - Natural language trigger
- "autonomous session" - Alternative trigger
- "long running task" - When task requires extended work

## Pre-Session Checklist

Before starting, verify:

1. **Read Current State**
   - Read `claude-progress.txt` for pending tasks
   - Read `claude-session-notes.md` for context from previous sessions
   - Read `claude-recovery-context.md` if it exists (post-compaction)

2. **Check Prerequisites**
   - Virtual environment should be active (check with `echo $VIRTUAL_ENV`)
   - Run `python3 -c "import psutil"` to verify dependencies

3. **Identify Target Upgrade**
   - Look for UPGRADE-XXX references in progress file
   - Check `docs/research/UPGRADE-*.md` for implementation details

## Session Configuration

Set these based on task complexity:

| Task Type | Recommended Flags |
|-----------|------------------|
| **Simple feature** | `./scripts/run_overnight.sh "Task description"` |
| **Complex refactor** | `./scripts/run_overnight.sh --continuous --with-recovery "Task"` |
| **Multi-day project** | `./scripts/run_overnight.sh --model opus --continuous --with-recovery "Task"` |

## Quick Start Commands

```bash
# Basic overnight session
./scripts/run_overnight.sh "Complete UPGRADE-XXX implementation"

# Full autonomous with recovery (recommended)
./scripts/run_overnight.sh --continuous --with-recovery "Complete all pending tasks in claude-progress.txt"

# Attach to running session
tmux attach -t overnight-dev

# Check session status
tmux ls
```

## Your Task

Based on the current context:

1. **Read the progress file** to identify pending tasks
2. **Summarize what needs to be done** (P0, P1, P2 categories)
3. **Recommend the appropriate session command**
4. **Ask if user wants to start the session or needs modifications**

Start by reading `claude-progress.txt` and `claude-session-notes.md`.
