# UPGRADE-011: Enhanced Overnight Autonomous Sessions

## Research Overview

**Search Date**: December 3, 2025 at 01:15 AM EST
**Scope**: Autonomous AI agent overnight sessions, crash recovery, watchdog patterns
**Focus**: Claude Code hooks, continuous development loops, safety patterns
**Result**: Multiple improvements identified for current overnight infrastructure

**Related Documentation**:

- [CLAUDE.md - Overnight Sessions](../../CLAUDE.md#overnight-autonomous-sessions) - Main instructions
- [Autonomous Agents Guide](../autonomous-agents/README.md) - Comprehensive autonomous guide
- [Session Notes Template](../../scripts/templates/session-notes-template.md) - Relay-race context template

**Implementation Files**:

| Component | File | Description |
|-----------|------|-------------|
| Stop Hook | [.claude/hooks/session_stop.py](../../.claude/hooks/session_stop.py) | Continuous mode, task completion check |
| PreCompact Hook | [.claude/hooks/pre_compact.py](../../.claude/hooks/pre_compact.py) | Transcript backup, checkpoint commits |
| Overnight Script | [scripts/run_overnight.sh](../../scripts/run_overnight.sh) | Session launcher with --continuous flag |
| Auto-Resume | [scripts/auto-resume.sh](../../scripts/auto-resume.sh) | Crash recovery with jitter |
| Watchdog | [scripts/watchdog.py](../../scripts/watchdog.py) | External process monitoring |
| Settings | [.claude/settings.json](../../.claude/settings.json) | Hook timeouts configuration |

---

## Phase 0: Research Findings

### Search Queries Used

1. "Claude Code autonomous overnight sessions best practices 2025"
2. "AI coding agents running autonomously overnight safety patterns 2025"
3. "autonomous AI agent crash recovery exponential backoff patterns"
4. "Claude Code hooks PreCompact Stop session transcript backup"
5. "continuous-claude autonomous overnight development GitHub"
6. "AI agent watchdog process monitoring idle timeout budget limits"
7. "Claude Code Stop hook exit code 2 continue session force restart"

---

## Key Sources

### Official Documentation

1. [Claude Code: Best practices for agentic coding](https://www.anthropic.com/engineering/claude-code-best-practices) (Anthropic, 2025)
2. [Hooks reference - Claude Code Docs](https://code.claude.com/docs/en/hooks) (Anthropic, 2025)
3. [Claude Code 2.0: Checkpoints, Subagents, and Autonomous Coding](https://skywork.ai/blog/claude-code-2-0-checkpoints-subagents-autonomous-coding/) (Skywork, 2025)

### Industry Research

4. [Safe ways to let your coding agent work autonomously](https://ericmjl.github.io/blog/2025/11/8/safe-ways-to-let-your-coding-agent-work-autonomously/) (Eric Ma, Nov 2025)
5. [Coding for the Future Agentic World](https://addyo.substack.com/p/coding-for-the-future-agentic-world) (Addy Osmani, 2025)
6. [9 Agentic AI Workflow Patterns](https://www.marktechpost.com/2025/08/09/9-agentic-ai-workflow-patterns-transforming-ai-agents-in-2025/) (MarkTechPost, Aug 2025)

### Tools & Implementations

7. [continuous-claude GitHub](https://github.com/AnandChowdhary/continuous-claude) (AnandChowdhary, 2025)
8. [claude-loop GitHub](https://github.com/DeprecatedLuke/claude-loop) (DeprecatedLuke, 2025)
9. [claude-code-hooks-mastery GitHub](https://github.com/disler/claude-code-hooks-mastery) (disler, 2025)

### Error Recovery & Monitoring

10. [Multi-Agent AI Failure Recovery That Actually Works](https://galileo.ai/blog/multi-agent-ai-system-failure-recovery) (Galileo, 2025)
11. [Error Recovery and Fallback Strategies in AI Agent Development](https://www.gocodeo.com/post/error-recovery-and-fallback-strategies-in-ai-agent-development) (GoCodeo, 2025)
12. [AI Agent Monitoring: Best Practices](https://uptimerobot.com/knowledge-hub/monitoring/ai-agent-monitoring-best-practices-tools-and-metrics/) (UptimeRobot, 2025)

---

## Critical Discoveries

### 1. Stop Hook Exit Code 2 - Force Continuation

**Discovery**: Stop hooks can prevent Claude from stopping using exit code 2.

```json
{
  "decision": "block",
  "reason": "Continue working on pending tasks from todo list"
}
```

**Impact**: Can implement "continuous mode" where Claude keeps working until explicitly stopped.

**Current Gap**: Our session_stop.py exits with code 0, allowing normal stop.

### 2. Continuous-Claude Pattern

**Discovery**: The [continuous-claude](https://github.com/AnandChowdhary/continuous-claude) project implements a relay-race pattern:
- Shared markdown file for context persistence between iterations
- Key instruction: "don't complete entire goal in one iteration, make meaningful progress"
- Supports `--max-runs` and `--max-cost` budget limits
- GitHub Next team validated this approach

**Impact**: We can implement similar relay-race pattern for overnight sessions.

### 3. Exponential Backoff with Jitter

**Discovery**: Basic exponential backoff without jitter causes "thundering herd" problems:

```python
# Current (basic)
delay = base * (2 ** retry_count)

# Better (with jitter)
delay = (base * (2 ** retry_count)) + random.uniform(-interval, interval)
```

**Current Gap**: Our auto-resume.sh uses basic exponential backoff without jitter.

### 4. Transcript Path in PreCompact Hook

**Discovery**: PreCompact hook receives `transcript_path` in its payload:

```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/session.jsonl",
  "permission_mode": "default",
  "hook_event_name": "PreCompact"
}
```

**Current Gap**: Our pre_compact.py searches for transcript files instead of using the provided path.

### 5. Stop Hook Active Flag

**Discovery**: `stop_hook_active` flag indicates Claude is continuing due to a Stop hook.

**Impact**: Must check this to prevent infinite loops when using exit code 2.

### 6. 60-Second Hook Timeout

**Discovery**: Hooks have a 60-second execution limit by default, configurable per command.

**Current Gap**: Our hooks don't specify timeout, relying on defaults.

### 7. Context Management at 80%

**Discovery**: Best practice is to run `/status` often and `/clear` at 80% context usage.

**Impact**: Could implement automatic context monitoring in watchdog.

### 8. AWS Frontier Agents Pattern

**Discovery**: Amazon's frontier agents (Kiro) can work autonomously for hours/days:
- Learn customer preferences over time
- "3-6 months in, agents behave like part of your team"
- Key: building trust through consistent, predictable behavior

**Impact**: Long-term goal for our autonomous system.

---

## Gap Analysis: Current vs Best Practices

| Feature | Current State | Best Practice | Priority |
|---------|--------------|---------------|----------|
| Stop hook continuation | Exits with code 0 | Exit code 2 + JSON block | HIGH |
| Exponential backoff | Basic | With jitter | MEDIUM |
| Transcript backup | File search | Use provided path | MEDIUM |
| Relay-race pattern | Not implemented | Shared context file | HIGH |
| Context monitoring | Manual | Auto-clear at 80% | LOW |
| Hook timeouts | Default | Explicit per hook | LOW |
| Budget limits | In watchdog | Also in continuous loop | MEDIUM |

---

## Phase 1: Upgrade Path

### Success Criteria

1. ✅ Stop hook can force continuation for multi-task sessions
2. ✅ Auto-resume uses exponential backoff with jitter
3. ✅ PreCompact hook uses provided transcript path
4. ✅ Relay-race pattern implemented with shared context
5. ✅ All tests pass after implementation

### Scope

- Modify: `.claude/hooks/session_stop.py`
- Modify: `.claude/hooks/pre_compact.py`
- Modify: `scripts/auto-resume.sh`
- Create: `scripts/continuous_mode.py` (optional)

---

## Phase 2: Implementation Checklist

### P0 - Critical

- [x] Update auto-resume.sh with jitter in exponential backoff ✅ (2025-12-03)
- [x] Update pre_compact.py to use CLAUDE_TOOL_INPUT for transcript path ✅ (2025-12-03)
- [x] Add optional continuation mode to session_stop.py ✅ (2025-12-03)

### P1 - Important

- [x] Implement relay-race pattern with shared notes file ✅ (2025-12-03)
- [x] Add explicit timeouts to all hooks ✅ (2025-12-03)
- [ ] Add context usage monitoring to watchdog
- [x] Update run_overnight.sh with --continuous flag ✅ (2025-12-03)

### P2 - Nice to Have

- [ ] Implement continuous-claude style loop wrapper
- [ ] Add --max-cost budget tracking to overnight launcher
- [ ] Create agent preference learning over time

---

## Implementation Details

### 1. Exponential Backoff with Jitter

```bash
# In auto-resume.sh
calculate_backoff() {
    local restart_count="$1"
    local base_delay=$((BACKOFF_BASE * (2 ** restart_count)))
    # Add jitter: ±25% of base delay
    local jitter=$(( (RANDOM % (base_delay / 2)) - (base_delay / 4) ))
    local delay=$((base_delay + jitter))
    # Cap at 10 minutes
    if [ $delay -gt 600 ]; then delay=600; fi
    if [ $delay -lt 10 ]; then delay=10; fi  # Minimum 10 seconds
    echo $delay
}
```

### 2. PreCompact Using Provided Path

```python
# In pre_compact.py
def get_transcript_from_input():
    """Get transcript path from hook input."""
    tool_input = os.environ.get("CLAUDE_TOOL_INPUT", "")
    if tool_input:
        try:
            data = json.loads(tool_input)
            return data.get("transcript_path")
        except json.JSONDecodeError:
            pass
    return None
```

### 3. Stop Hook Continuation Mode

```python
# In session_stop.py
def should_continue():
    """Check if there are pending tasks."""
    progress_file = Path("claude-progress.txt")
    if progress_file.exists():
        content = progress_file.read_text()
        # Check for incomplete tasks
        if "- [ ]" in content:
            return True
    return False

def main():
    if os.environ.get("CONTINUOUS_MODE") == "1" and should_continue():
        # Output JSON to continue
        result = {
            "decision": "block",
            "reason": "Pending tasks in claude-progress.txt. Continue working."
        }
        print(json.dumps(result))
        return 2  # Exit code 2 blocks stopping

    # Normal stop behavior
    # ... existing code ...
    return 0
```

---

## Research Deliverables

| Deliverable | Status |
|-------------|--------|
| Research document | ✅ Created |
| Gap analysis | ✅ Complete |
| Implementation checklist | ✅ Created |
| Code examples | ✅ Provided |

---

## Changelog

| Date | Change |
|------|--------|
| 2025-12-03 | Initial research document created |
| 2025-12-03 | P0 Implementation complete: jitter in backoff, transcript_path, continuation mode |
| 2025-12-03 | P1 Implementation: hook timeouts, --continuous flag, relay-race pattern |

---

## References

- [Anthropic Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Claude Code Hooks Reference](https://code.claude.com/docs/en/hooks)
- [continuous-claude GitHub](https://github.com/AnandChowdhary/continuous-claude)
- [Safe Autonomous Coding Agents](https://ericmjl.github.io/blog/2025/11/8/safe-ways-to-let-your-coding-agent-work-autonomously/)
- [Exponential Backoff Wikipedia](https://en.wikipedia.org/wiki/Exponential_backoff)
