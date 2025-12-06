# Check Overnight Session Status

You are checking the status of an overnight autonomous session.

## Quick Status Check

Run these commands to check session status:

```bash
# Check if tmux session exists
tmux ls 2>/dev/null | grep overnight-dev || echo "No overnight session running"

# Check watchdog status
ps aux | grep watchdog.py | grep -v grep || echo "Watchdog not running"

# Check recent progress
tail -30 claude-progress.txt

# Check continuation history
cat logs/continuation_history.jsonl 2>/dev/null | tail -5

# Check compaction history
cat logs/compaction-history.jsonl 2>/dev/null | tail -3
```

## Status Report Template

After running the checks, provide a status report:

### Session Status
- **Tmux Session**: Running/Not running
- **Watchdog**: Active/Inactive
- **Last Activity**: [timestamp from progress file]

### Progress Summary
- **P0 Tasks**: X/Y complete
- **P1 Tasks**: X/Y complete
- **P2 Tasks**: X/Y complete
- **Overall**: X% complete

### Recent Activity
- Last checkpoint: [from git log]
- Continuations triggered: [from history]
- Context compactions: [from history]

### Recommendations
- [Based on status, suggest next actions]

## Commands for User

| Action | Command |
|--------|---------|
| Attach to session | `tmux attach -t overnight-dev` |
| View logs | `tail -f logs/watchdog-*.log` |
| Force checkpoint | `./scripts/checkpoint.sh auto` |
| Stop session | `tmux kill-session -t overnight-dev` |

Execute the status checks now and provide the report.
