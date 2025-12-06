# Overnight Infrastructure Additions (Expanded)

**Purpose**: Supplement to existing workflow — adds WSL2 persistence, Docker sandboxing, tmux, notifications, autonomous hooks, and operational tooling.
**Integration Point**: Works alongside `scripts/run_overnight.sh`, `scripts/watchdog.py`, and existing RIC Loop.
**Does NOT Replace**: RIC Loop, two-agent architecture, research documentation, or existing hooks.

---

## 1. WSL2 Persistence Configuration

Your watchdog assumes the WSL2 session survives overnight. Without these settings, WSL2 terminates ~15 seconds after the last process exits.

### Windows Host: `%UserProfile%\.wslconfig`

```ini
[wsl2]
memory=8GB
processors=4
swap=4GB
localhostForwarding=true

[experimental]
sparseVhd=true
```

### Inside Ubuntu: `/etc/wsl.conf`

```ini
[boot]
systemd=true
command="service cron start"

[automount]
enabled=true
options="metadata,umask=22,fmask=11"

[user]
default=your_username
```

**Apply changes**: Run `wsl --shutdown` from PowerShell, then reopen Ubuntu.

---

## 2. tmux Session Persistence

Ensures your overnight session survives terminal disconnections and VS Code restarts.

### Install and Configure

```bash
sudo apt update && sudo apt install -y tmux
```

Create `~/.tmux.conf`:

```bash
set -g history-limit 50000
set -g mouse on
set -sg escape-time 0
setw -g mode-keys vi

# Status bar for session health monitoring
set -g status on
set -g status-interval 5
set -g status-position bottom
set -g status-right '#{pane_current_path} | %H:%M'

# Easy pane management
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"

# Don't rename windows automatically
set-option -g allow-rename off

# Activity monitoring
setw -g monitor-activity on
set -g visual-activity on
```

### Integration with run_overnight.sh

Add to beginning of `scripts/run_overnight.sh`:

```bash
#!/bin/bash
SESSION="overnight-dev"

# Ensure tmux session exists
if [ -z "$TMUX" ]; then
    if ! tmux has-session -t $SESSION 2>/dev/null; then
        tmux new-session -d -s $SESSION -c "$(pwd)"
    fi
    tmux send-keys -t $SESSION "$0 $*" C-m
    echo "Started in tmux session '$SESSION'"
    echo "Attach with: tmux attach -t $SESSION"
    exit 0
fi

# Rest of your existing run_overnight.sh continues here...
```

### Useful Commands

| Command | Purpose |
|---------|---------|
| `tmux attach -t overnight-dev` | Reconnect to session |
| `tmux ls` | List active sessions |
| `Ctrl+b d` | Detach without stopping |
| `Ctrl+b [` | Scroll mode (q to exit) |

---

## 3. Docker Sandboxing

Enables safe use of `--dangerously-skip-permissions` by isolating the coding agent. Your external watchdog remains on the host.

### Dockerfile

Save as `docker/Dockerfile.claude`:

```dockerfile
FROM ubuntu:22.04

# Create non-root user
RUN groupadd -r claude && useradd -r -g claude -m claude

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git curl nodejs npm jq bc \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code
RUN npm install -g @anthropic-ai/claude-code@latest

# Install LEAN CLI for backtesting
RUN pip install lean --break-system-packages

# Setup Python environment
WORKDIR /workspace
RUN python3 -m venv /home/claude/.venv \
    && chown -R claude:claude /home/claude
ENV PATH="/home/claude/.venv/bin:$PATH"

# Install your trading dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

USER claude
ENTRYPOINT ["claude"]
```

### docker-compose.yml

Save as `docker/docker-compose.yml`:

```yaml
version: '3.8'

services:
  claude-coding-agent:
    build:
      context: .
      dockerfile: Dockerfile.claude
    container_name: claude-overnight
    volumes:
      # Mount project (read-write for coding)
      - ../:/workspace
      # Mount Claude config (read-only)
      - ~/.claude:/home/claude/.claude:ro
      # Mount progress file location for watchdog communication
      - ../claude-progress.txt:/workspace/claude-progress.txt
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
    # Security hardening
    read_only: true
    tmpfs:
      - /tmp
      - /var/tmp
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true
    # Network isolation (no external access)
    networks:
      - isolated
    # Health check for watchdog integration
    healthcheck:
      test: ["CMD", "pgrep", "-f", "claude"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  isolated:
    driver: bridge
    internal: true
```

### Modified run_overnight.sh (Docker Version)

Replace the Claude Code invocation in your existing script:

```bash
# Instead of running claude directly, run in container
start_coding_agent() {
    local task="$1"

    cd docker
    docker-compose up -d --build

    docker exec -it claude-overnight claude \
        --dangerously-skip-permissions \
        -p "$task" 2>&1 | tee -a "../logs/overnight-$(date +%Y%m%d).log"
}

# Your existing watchdog.py remains external on the host
# It monitors claude-progress.txt which is bind-mounted
```

### Build and Test

```bash
cd docker
docker-compose build
docker-compose run --rm claude-coding-agent --version
```

---

## 4. CLAUDE.md Autonomous Directives

Add these directives to your existing CLAUDE.md to enhance overnight operation. These complement your RIC Loop instructions.

### Autonomous Operation Section (Add to CLAUDE.md)

```markdown
## Autonomous Overnight Operation

IMPORTANT: When running in autonomous overnight mode, follow these directives:

### Session Management
- Use /compact at 60% context usage (your existing threshold)
- Document progress in claude-progress.txt before any /clear
- Create checkpoint commits before major architectural changes
- If context exceeds 70% after compact, /clear and resume from progress file

### Error Recovery Protocol
If you encounter repeated failures (3+ consecutive):
1. Document the error in ERRORS.md with timestamp and stack trace
2. Create checkpoint: `git commit -am "checkpoint: before recovery $(date +%s)"`
3. Try alternative approach with simpler implementation
4. If still failing after 2 recovery attempts, document findings and halt
5. Update claude-progress.txt with status: BLOCKED

### Commit Protocol for Autonomous Work
- Commit after each successful component change (aligns with your single-component rule)
- Use conventional commit format: `type(scope): description [AI-GENERATED]`
- Types: feat, fix, refactor, test, docs, chore
- Always run tests before commit (your existing gates handle this)

### Context Window Awareness
- Every 500 input tokens adds ~0.53 seconds latency
- Performance degrades sharply above 80% capacity
- Prefer multiple focused sessions over one long session
```

---

## 5. Extended Hooks for Autonomous Operation

These hooks complement your existing RIC Loop hooks. Add to `.claude/settings.json`:

### Autonomous Operation Hooks

```json
{
  "env": {
    "BASH_DEFAULT_TIMEOUT_MS": "1800000",
    "BASH_MAX_TIMEOUT_MS": "7200000"
  },
  "permissions": {
    "allow": [
      "Read(**)",
      "Write(src/**)",
      "Write(tests/**)",
      "Write(docs/**)",
      "Write(claude-progress.txt)",
      "Write(ERRORS.md)",
      "Bash(python *)",
      "Bash(pytest *)",
      "Bash(mypy *)",
      "Bash(ruff *)",
      "Bash(git status)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(git diff *)",
      "Bash(lean backtest *)",
      "Bash(lean cloud backtest *)"
    ],
    "deny": [
      "Read(.env*)",
      "Read(**/secrets/**)",
      "Read(**/*.key)",
      "Write(.git/**)",
      "Write(.env*)",
      "Write(**/*.key)",
      "Bash(rm -rf *)",
      "Bash(sudo *)",
      "Bash(curl * | bash)",
      "Bash(wget * | bash)",
      "Bash(pip install *)",
      "Bash(npm install *)",
      "Bash(lean live *)",
      "WebFetch(*)"
    ]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.file_path // empty' | { read fp; if [ -n \"$fp\" ] && echo \"$fp\" | grep -q '\\.py$'; then ruff check --fix \"$fp\" 2>/dev/null; ruff format \"$fp\" 2>/dev/null; fi; }",
            "timeout": 30
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "jq -c '{timestamp: (now | strftime(\"%Y-%m-%dT%H:%M:%S\")), tool: \"Bash\", command: .tool_input.command}' >> ~/.claude/activity-log.jsonl"
          }
        ]
      },
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "jq -c '{timestamp: (now | strftime(\"%Y-%m-%dT%H:%M:%S\")), tool: .tool_name, file: .tool_input.file_path}' >> ~/.claude/activity-log.jsonl"
          }
        ]
      }
    ]
  }
}
```

### Token Burn Control Hook

Prevents runaway iterations by adding deliberate pauses before file modifications:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "sleep 5",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

For more aggressive throttling during overnight runs (prevents rapid token consumption):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "sleep 30",
            "timeout": 35
          }
        ]
      }
    ]
  }
}
```

### Auto-Commit After Successful Test Hook

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.command' | grep -q '^pytest' && [ \"$EXIT_CODE\" = \"0\" ] && git add -A && git commit -m \"test: passing tests [AI-GENERATED]\" 2>/dev/null || true"
          }
        ]
      }
    ]
  }
}
```

---

## 6. Self-Recovery Checkpoint Scripts

### scripts/checkpoint.sh

```bash
#!/bin/bash
# Checkpoint management for autonomous sessions

CHECKPOINT_PREFIX="checkpoint"

create_checkpoint() {
    local name="${1:-$(date +%Y%m%d_%H%M%S)}"
    local message="${2:-Automated checkpoint}"

    git add -A
    if git diff --cached --quiet; then
        echo "No changes to checkpoint"
        return 0
    fi

    git commit -m "${CHECKPOINT_PREFIX}: ${name} - ${message}"
    git tag -a "${CHECKPOINT_PREFIX}-${name}" -m "${message}"
    echo "✅ Created ${CHECKPOINT_PREFIX}-${name}"
}

restore_checkpoint() {
    local checkpoint="$1"

    if [ -z "$checkpoint" ]; then
        echo "Available checkpoints:"
        git tag -l "${CHECKPOINT_PREFIX}-*" | tail -10
        return 1
    fi

    # Stash current work
    git stash push -m "Before restoring ${checkpoint}"

    # Reset to checkpoint
    if git rev-parse "$checkpoint" >/dev/null 2>&1; then
        git reset --hard "$checkpoint"
        echo "✅ Restored to ${checkpoint}"
    else
        echo "❌ Checkpoint not found: ${checkpoint}"
        git stash pop
        return 1
    fi
}

list_checkpoints() {
    echo "Recent checkpoints:"
    git tag -l "${CHECKPOINT_PREFIX}-*" --sort=-creatordate | head -20
}

# Get last known good checkpoint (tests passed)
get_last_good() {
    git log --oneline --all --grep="test: passing" | head -1 | cut -d' ' -f1
}

# Main
case "${1:-}" in
    create)  create_checkpoint "$2" "$3" ;;
    restore) restore_checkpoint "$2" ;;
    list)    list_checkpoints ;;
    last-good) get_last_good ;;
    *)
        echo "Usage: $0 {create|restore|list|last-good} [args]"
        echo "  create [name] [message] - Create named checkpoint"
        echo "  restore <checkpoint>    - Restore to checkpoint"
        echo "  list                    - List recent checkpoints"
        echo "  last-good               - Get last commit with passing tests"
        ;;
esac
```

### scripts/auto-recover.sh

```bash
#!/bin/bash
# Automatic recovery from failed state

PROJECT_DIR="${1:-$(pwd)}"
MAX_RETRIES=3

cd "$PROJECT_DIR"

check_health() {
    # Run quick validation
    if pytest tests/ -x -q --tb=no 2>/dev/null; then
        return 0
    fi
    return 1
}

attempt_recovery() {
    local attempt=$1
    echo "Recovery attempt $attempt of $MAX_RETRIES"

    # Strategy 1: Reset to last good checkpoint
    if [ "$attempt" -eq 1 ]; then
        local last_good=$(./scripts/checkpoint.sh last-good)
        if [ -n "$last_good" ]; then
            echo "Resetting to last good commit: $last_good"
            git stash push -m "Failed state before recovery"
            git reset --hard "$last_good"
            return $?
        fi
    fi

    # Strategy 2: Reset to last checkpoint tag
    if [ "$attempt" -eq 2 ]; then
        local last_checkpoint=$(git tag -l "checkpoint-*" --sort=-creatordate | head -1)
        if [ -n "$last_checkpoint" ]; then
            echo "Resetting to checkpoint: $last_checkpoint"
            git reset --hard "$last_checkpoint"
            return $?
        fi
    fi

    # Strategy 3: Clean and reinstall dependencies
    if [ "$attempt" -eq 3 ]; then
        echo "Cleaning and reinstalling..."
        rm -rf __pycache__ .pytest_cache .mypy_cache
        pip install -r requirements.txt --quiet
        return $?
    fi

    return 1
}

main() {
    if check_health; then
        echo "✅ Project healthy"
        exit 0
    fi

    echo "❌ Project unhealthy, attempting recovery..."

    for i in $(seq 1 $MAX_RETRIES); do
        attempt_recovery $i
        if check_health; then
            echo "✅ Recovery successful on attempt $i"
            exit 0
        fi
    done

    echo "❌ Recovery failed after $MAX_RETRIES attempts"
    echo "Manual intervention required"
    exit 1
}

main
```

---

## 7. Git Automation for AI-Generated Commits

### Conventional Commit Conventions

```
type(scope): description [AI-GENERATED]

[optional body with details]

[optional footer]
```

**Types for autonomous work:**
- `feat`: New feature or indicator
- `fix`: Bug fix
- `refactor`: Code restructuring
- `test`: Adding or updating tests
- `docs`: Documentation changes
- `chore`: Maintenance tasks

### .gitattributes for AI Tracking

```gitattributes
# Track AI-generated content
*.py diff=python
*.md diff=markdown

# AI commit marker
**/AI-GENERATED linguist-generated
```

### Pre-push Hook with Rollback

Save as `.git/hooks/pre-push`:

```bash
#!/bin/bash
# Validate before push, rollback on failure

echo "Running pre-push validation..."

# Run full test suite
if ! pytest tests/ -v --tb=short; then
    echo "❌ Tests failed - push blocked"

    # Find last good state
    LAST_GOOD=$(git log --oneline --grep="test: passing" -1 --format="%H")

    if [ -n "$LAST_GOOD" ]; then
        echo "Last known good commit: $LAST_GOOD"
        echo "To rollback: git reset --hard $LAST_GOOD"
    fi

    exit 1
fi

# Run type checking
if ! mypy src/ --ignore-missing-imports; then
    echo "⚠️ Type errors detected - push blocked"
    exit 1
fi

# Create checkpoint on successful push
git tag -a "push-$(date +%Y%m%d-%H%M%S)" -m "Pre-push checkpoint" 2>/dev/null || true

echo "✅ Validation passed"
exit 0
```

Make executable: `chmod +x .git/hooks/pre-push`

---

## 8. LEAN CLI Integration for QuantConnect

### scripts/validate-strategy.sh

```bash
#!/bin/bash
# Validate trading strategy with backtesting

PROJECT="${1:-TradingBot}"
MIN_SHARPE="${2:-0.5}"
MAX_DRAWDOWN="${3:-0.20}"
START_DATE="${4:-2023-01-01}"
END_DATE="${5:-2024-01-01}"

echo "🧪 Running backtest for $PROJECT..."
echo "   Period: $START_DATE to $END_DATE"
echo "   Min Sharpe: $MIN_SHARPE | Max Drawdown: $MAX_DRAWDOWN"

# Run local backtest
if ! lean backtest "$PROJECT" \
    --data-provider-historical Local \
    --start "$START_DATE" \
    --end "$END_DATE"; then
    echo "❌ Backtest execution failed"
    exit 1
fi

# Find most recent results
RESULT_FILE=$(find "$PROJECT/backtests" -name "*.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2)

if [ -z "$RESULT_FILE" ]; then
    echo "❌ No backtest results found"
    exit 1
fi

# Extract metrics
SHARPE=$(jq -r '.Statistics.SharpeRatio // "0"' "$RESULT_FILE")
DRAWDOWN=$(jq -r '.Statistics.Drawdown // "1"' "$RESULT_FILE" | tr -d '%' | awk '{print $1/100}')
TOTAL_TRADES=$(jq -r '.Statistics.TotalNumberOfTrades // "0"' "$RESULT_FILE")
WIN_RATE=$(jq -r '.Statistics.WinRate // "0"' "$RESULT_FILE" | tr -d '%')

echo ""
echo "📊 Results:"
echo "   Sharpe Ratio: $SHARPE"
echo "   Max Drawdown: $(echo "$DRAWDOWN * 100" | bc)%"
echo "   Total Trades: $TOTAL_TRADES"
echo "   Win Rate: $WIN_RATE%"

# Validate against thresholds
PASSED=true

if (( $(echo "$SHARPE < $MIN_SHARPE" | bc -l) )); then
    echo "⚠️  Sharpe ratio $SHARPE below threshold $MIN_SHARPE"
    PASSED=false
fi

if (( $(echo "$DRAWDOWN > $MAX_DRAWDOWN" | bc -l) )); then
    echo "⚠️  Drawdown $(echo "$DRAWDOWN * 100" | bc)% exceeds threshold $(echo "$MAX_DRAWDOWN * 100" | bc)%"
    PASSED=false
fi

if [ "$TOTAL_TRADES" -lt 10 ]; then
    echo "⚠️  Insufficient trades ($TOTAL_TRADES) for statistical significance"
    PASSED=false
fi

if [ "$PASSED" = true ]; then
    echo ""
    echo "✅ Strategy validation PASSED"
    exit 0
else
    echo ""
    echo "❌ Strategy validation FAILED"
    exit 1
fi
```

### Integration with Pre-commit

Add to `.pre-commit-config.yaml`:

```yaml
  - repo: local
    hooks:
      - id: backtest-validation
        name: backtest validation
        entry: ./scripts/validate-strategy.sh TradingBot 0.5 0.20
        language: script
        files: ^src/algorithms/.*\.py$
        pass_filenames: false
        stages: [push]  # Only on push, not every commit
```

---

## 9. Discord/Slack Notifications

Wire into your existing `watchdog.py` for real-time alerts.

### scripts/notify.py

```python
#!/usr/bin/env python3
"""
Notification module for watchdog.py integration.
Supports Discord and Slack webhooks.
"""

import os
import json
import requests
from datetime import datetime
from typing import Optional

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")

def send_discord(title: str, message: str, color: int = 0x00FF00) -> bool:
    if not DISCORD_WEBHOOK:
        return False
    try:
        requests.post(DISCORD_WEBHOOK, json={
            "embeds": [{
                "title": title,
                "description": message[:4000],
                "color": color,
                "timestamp": datetime.utcnow().isoformat()
            }]
        }, timeout=10)
        return True
    except Exception as e:
        print(f"Discord notification failed: {e}")
        return False

def send_slack(title: str, message: str, color: str = "good") -> bool:
    if not SLACK_WEBHOOK:
        return False
    try:
        requests.post(SLACK_WEBHOOK, json={
            "attachments": [{
                "color": color,
                "title": title,
                "text": message[:3000],
                "ts": datetime.utcnow().timestamp()
            }]
        }, timeout=10)
        return True
    except Exception as e:
        print(f"Slack notification failed: {e}")
        return False

def notify(title: str, message: str, level: str = "info") -> None:
    """
    Send notification to all configured channels.

    Args:
        title: Notification title
        message: Notification body
        level: "info" (green), "warning" (yellow), "error" (red)
    """
    colors = {
        "info": (0x00FF00, "good"),
        "warning": (0xFFFF00, "warning"),
        "error": (0xFF0000, "danger")
    }
    discord_color, slack_color = colors.get(level, colors["info"])

    send_discord(title, message, discord_color)
    send_slack(title, message, slack_color)

# Convenience functions for watchdog.py integration
def notify_session_start(task: str) -> None:
    notify("🚀 Overnight Session Started", f"**Task**: {task}", "info")

def notify_checkpoint(phase: str, progress: str) -> None:
    notify(f"📍 Checkpoint: {phase}", progress, "info")

def notify_context_warning(usage_pct: float) -> None:
    notify(
        f"⚠️ Context at {usage_pct:.0f}%",
        "Approaching context limit. Consider /compact or /clear.",
        "warning"
    )

def notify_warning(issue: str) -> None:
    notify("⚠️ Warning", issue, "warning")

def notify_error(error: str) -> None:
    notify("❌ Error - Session Halted", f"```\n{error[:1500]}\n```", "error")

def notify_completion(summary: str, success: bool = True) -> None:
    if success:
        notify("✅ Session Complete", summary, "info")
    else:
        notify("⚠️ Session Ended with Issues", summary, "warning")

def notify_cost_alert(current: float, limit: float) -> None:
    pct = (current / limit) * 100
    notify(
        f"💰 Cost Alert: ${current:.2f} ({pct:.0f}% of limit)",
        f"Approaching ${limit:.2f} budget limit",
        "warning" if pct < 90 else "error"
    )

def notify_backtest_result(sharpe: float, drawdown: float, passed: bool) -> None:
    status = "✅ PASSED" if passed else "❌ FAILED"
    notify(
        f"📊 Backtest {status}",
        f"Sharpe: {sharpe:.2f} | Drawdown: {drawdown:.1%}",
        "info" if passed else "warning"
    )

def notify_rate_limit(wait_minutes: int) -> None:
    notify(
        "⏳ Rate Limit Hit",
        f"Waiting {wait_minutes} minutes before resuming...",
        "warning"
    )
```

### Integration with watchdog.py

Add to your existing `scripts/watchdog.py`:

```python
# Add import at top
from notify import (
    notify_session_start, notify_checkpoint, notify_warning,
    notify_error, notify_completion, notify_cost_alert,
    notify_context_warning, notify_rate_limit, notify_backtest_result
)

# Add to your existing check functions:

def on_session_start(self, task: str):
    notify_session_start(task)

def check_idle_timeout(self):
    # Your existing logic...
    if idle_exceeded:
        notify_warning(f"Idle timeout: No activity for {idle_minutes} minutes")
        # existing termination logic...

def check_context_usage(self, usage_pct: float):
    if usage_pct >= 60:  # Your 60% threshold
        notify_context_warning(usage_pct)

def check_cost_limit(self):
    # Your existing logic...
    if cost > self.config['max_cost_usd'] * 0.8:
        notify_cost_alert(cost, self.config['max_cost_usd'])
    if cost >= self.config['max_cost_usd']:
        notify_error(f"Cost limit reached: ${cost:.2f}")
        # existing halt logic...

def on_rate_limit(self, wait_minutes: int):
    notify_rate_limit(wait_minutes)

def on_session_complete(self, success: bool):
    summary = self.generate_summary()  # Your existing summary logic
    notify_completion(summary, success)
```

### Environment Variables

Add to your shell profile or `.env`:

```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/YOUR_WEBHOOK"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR_WEBHOOK"
```

---

## 10. Activity Logging and Diagnostics

### Activity Log Analysis

The hooks create `~/.claude/activity-log.jsonl`. Analyze with:

```bash
# Recent activity
tail -50 ~/.claude/activity-log.jsonl | jq .

# Count operations by type
cat ~/.claude/activity-log.jsonl | jq -r '.tool' | sort | uniq -c | sort -rn

# Find all file edits
grep '"tool":"Edit"' ~/.claude/activity-log.jsonl | jq -r '.file'

# Activity timeline
cat ~/.claude/activity-log.jsonl | jq -r '.timestamp + " " + .tool' | tail -100
```

### Diagnostic Tools Reference

| Tool | Purpose | Usage |
|------|---------|-------|
| `claude doctor` | Built-in diagnostics | `claude doctor` |
| `status.claude.com` | Service status | Check before overnight |
| `ccusage` | Token usage monitoring | `npx ccusage@latest blocks --live` |
| `/cost` | Session cost (API only) | Within Claude Code |
| `/context` | Context window usage | Within Claude Code |

### scripts/diagnose.sh

```bash
#!/bin/bash
# Pre-overnight diagnostic check

echo "=== Claude Code Diagnostics ==="
echo ""

# Check Claude Code version
echo "📦 Claude Code Version:"
claude --version
echo ""

# Check API connectivity
echo "🌐 API Status:"
curl -s https://status.anthropic.com/api/v2/status.json | jq -r '.status.description'
echo ""

# Check disk space
echo "💾 Disk Space:"
df -h . | tail -1 | awk '{print "  Used: " $3 " / " $2 " (" $5 ")"}'
echo ""

# Check memory
echo "🧠 Memory:"
free -h | grep Mem | awk '{print "  Used: " $3 " / " $2}'
echo ""

# Check git status
echo "📁 Git Status:"
if git status --porcelain | head -5 | grep -q .; then
    echo "  ⚠️  Uncommitted changes:"
    git status --porcelain | head -5 | sed 's/^/    /'
else
    echo "  ✅ Clean working directory"
fi
echo ""

# Check tests
echo "🧪 Quick Test:"
if pytest tests/ -x -q --tb=no 2>/dev/null; then
    echo "  ✅ Tests passing"
else
    echo "  ❌ Tests failing"
fi
echo ""

# Check activity log size
if [ -f ~/.claude/activity-log.jsonl ]; then
    LINES=$(wc -l < ~/.claude/activity-log.jsonl)
    SIZE=$(du -h ~/.claude/activity-log.jsonl | cut -f1)
    echo "📝 Activity Log: $LINES entries ($SIZE)"
    if [ "$LINES" -gt 10000 ]; then
        echo "  ⚠️  Consider rotating: mv ~/.claude/activity-log.jsonl ~/.claude/activity-log.jsonl.bak"
    fi
fi
echo ""

echo "=== Ready for overnight session ==="
```

---

## 11. VS Code Remote-WSL Settings

Ensures VS Code maintains connection during overnight sessions.

### `.vscode/settings.json` (Project Level)

Add these to your existing settings:

```json
{
    "terminal.integrated.defaultProfile.linux": "tmux",
    "terminal.integrated.profiles.linux": {
        "tmux": {
            "path": "/usr/bin/tmux",
            "args": ["new-session", "-A", "-s", "vscode-dev"]
        }
    },
    "files.autoSave": "afterDelay",
    "files.autoSaveDelay": 1000,
    "remote.WSL.useShellEnvironment": true,
    "remote.autoForwardPorts": false,
    "files.watcherExclude": {
        "**/logs/**": true,
        "**/.claude/**": true,
        "**/backtests/**": true
    }
}
```

---

## 12. Pre-commit Quality Gates

Automates your existing quality checks before any commit.

### `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
        additional_dependencies: []

  - repo: local
    hooks:
      - id: pytest-quick
        name: pytest (quick)
        entry: pytest tests/ -x -q --tb=short
        language: system
        types: [python]
        pass_filenames: false
        stages: [commit]

      - id: validate-research-docs
        name: validate research docs
        entry: python -c "from scripts.validate_docs import main; main()"
        language: system
        files: ^docs/research/.*\.md$
        pass_filenames: false

      - id: backtest-validation
        name: backtest validation
        entry: ./scripts/validate-strategy.sh TradingBot 0.5 0.20
        language: script
        files: ^src/algorithms/.*\.py$
        pass_filenames: false
        stages: [push]
```

### Install

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

---

## 13. Community Session Management Tools

Optional tools that complement your existing watchdog.

### CCAutoRenew

Monitors rate limits and auto-resumes when they lift:

```bash
git clone https://github.com/aniketkarne/CCAutoRenew.git
cd CCAutoRenew
chmod +x claude-daemon-manager.sh

# Start daemon for overnight session
./claude-daemon-manager.sh start --at "22:00" --message "continue per CLAUDE.md"
```

### Token Usage Monitoring

```bash
# Live monitoring
npx ccusage@latest blocks --live

# Historical usage
npx ccusage@latest history --days 7
```

### claude-auto-resume

Detects rate limits and auto-resumes:

```bash
npm install -g claude-auto-resume
claude-auto-resume "implement the RSI divergence indicator"
```

---

## 14. Context Window Performance Reference

### Latency Impact

| Context Size | Approximate Latency Added |
|--------------|---------------------------|
| 10K tokens | +10 seconds |
| 50K tokens | +53 seconds |
| 100K tokens | +106 seconds |
| 150K tokens | +159 seconds |

**Rule**: Every 500 input tokens adds ~0.53 seconds latency.

### Prompt Caching Benefits

When using API directly (for custom tooling):

| Metric | Without Caching | With Caching | Improvement |
|--------|-----------------|--------------|-------------|
| Latency (100K context) | 11.5s | 2.4s | 79% faster |
| Cost | Full price | 10% of input cost | 90% cheaper |

Cache TTL: 5 minutes (default) or 1 hour (2x write cost)

### Optimal Session Strategy

```
Session Start → Work → 60% Context → /compact → Continue
                                  ↓
                            Still >70%?
                                  ↓
                      Document → /clear → Resume
```

---

## 15. CLI vs VS Code Extension Comparison

For overnight autonomous work, CLI generally performs better:

| Aspect | CLI | VS Code Extension |
|--------|-----|-------------------|
| Resource footprint | Lightweight | Higher (renderer overhead) |
| Commands available | ~40 | ~18 |
| Session recovery | `/rewind`, double-ESC | Limited |
| Long session stability | Better | CPU spikes reported |
| Headless operation | Native | Requires display |

**Recommendation**: Use CLI for overnight sessions, extension for interactive development.

---

## 16. Complete Project Directory Template

```
your-trading-bot/
├── .claude/
│   ├── settings.json              # Permissions + hooks (Section 5)
│   ├── settings.local.json        # Personal overrides (gitignored)
│   ├── commands/                  # Your existing slash commands
│   │   ├── ric-start.md
│   │   ├── ric-research.md
│   │   └── ...
│   └── hooks/                     # Your existing Python hooks
│       ├── protect_files.py
│       ├── check_ric_loop.py
│       └── ...
├── .github/
│   └── workflows/                 # Your existing CI/CD
├── .vscode/
│   └── settings.json              # Section 11
├── config/
│   └── watchdog.json              # Your existing config
├── docker/
│   ├── Dockerfile.claude          # Section 3
│   └── docker-compose.yml         # Section 3
├── docs/
│   ├── development/
│   │   └── ENHANCED_RIC_WORKFLOW.md
│   └── research/
├── logs/                          # Overnight session logs
├── scripts/
│   ├── run_overnight.sh           # Your existing + tmux wrapper
│   ├── watchdog.py                # Your existing + notify integration
│   ├── checkpoint.sh              # Section 6
│   ├── auto-recover.sh            # Section 6
│   ├── validate-strategy.sh       # Section 8
│   ├── notify.py                  # Section 9
│   └── diagnose.sh                # Section 10
├── src/
│   └── algorithms/
├── tests/
├── .gitattributes                 # Section 7
├── .gitignore
├── .pre-commit-config.yaml        # Section 12
├── CLAUDE.md                      # + Section 4 additions
├── CLAUDE.local.md                # Personal (gitignored)
├── claude-progress.txt            # Watchdog communication
├── ERRORS.md                      # Error documentation
├── pyproject.toml
└── requirements.txt
```

---

## 17. Integration Checklist

### WSL2 & Session Persistence
- [ ] Create `.wslconfig` on Windows host
- [ ] Create `/etc/wsl.conf` in Ubuntu
- [ ] Run `wsl --shutdown` and restart
- [ ] Install tmux, create `~/.tmux.conf`
- [ ] Add tmux wrapper to `run_overnight.sh`

### Docker Sandboxing (Optional but Recommended)
- [ ] Create `docker/Dockerfile.claude`
- [ ] Create `docker/docker-compose.yml`
- [ ] Build container: `docker-compose build`
- [ ] Test: `docker-compose run --rm claude-coding-agent --version`

### Hooks & Permissions
- [ ] Merge autonomous hooks into `.claude/settings.json`
- [ ] Add token burn control hook (if desired)
- [ ] Add activity logging hooks
- [ ] Test hooks: `claude --print-config`

### CLAUDE.md Updates
- [ ] Add autonomous operation directives (Section 4)
- [ ] Add error recovery protocol
- [ ] Add commit protocol for AI work

### Scripts
- [ ] Create `scripts/checkpoint.sh`
- [ ] Create `scripts/auto-recover.sh`
- [ ] Create `scripts/validate-strategy.sh`
- [ ] Create `scripts/notify.py`
- [ ] Create `scripts/diagnose.sh`
- [ ] Make all scripts executable: `chmod +x scripts/*.sh`

### Notifications
- [ ] Set up Discord/Slack webhook
- [ ] Add webhook URLs to environment
- [ ] Integrate notify calls into `watchdog.py`
- [ ] Test: `python scripts/notify.py` (add test function)

### Git Automation
- [ ] Create `.git/hooks/pre-push`
- [ ] Make executable: `chmod +x .git/hooks/pre-push`
- [ ] Install pre-commit: `pip install pre-commit && pre-commit install`

### VS Code
- [ ] Update `.vscode/settings.json`

### Final Validation
- [ ] Run `./scripts/diagnose.sh`
- [ ] Test overnight startup in tmux
- [ ] Verify watchdog communication via `claude-progress.txt`
- [ ] Test notification delivery

---

## 18. Quick Reference: Overnight Startup

```bash
# 1. Run diagnostics
./scripts/diagnose.sh

# 2. Start tmux session
tmux new-session -s overnight -c ~/projects/quantconnect-bot

# 3. Create pre-session checkpoint
./scripts/checkpoint.sh create "pre-overnight"

# 4. Verify watchdog config
cat config/watchdog.json

# 5. Launch (choose one)

# Option A: Native sandbox (simpler)
./scripts/run_overnight.sh "Task description"

# Option B: Docker sandbox (more isolated)
cd docker && docker-compose up -d
docker exec -it claude-overnight claude \
    --dangerously-skip-permissions \
    -p "Task description"

# 6. Detach from tmux
# Press Ctrl+b, then d

# 7. Reconnect later
tmux attach -t overnight

# 8. Monitor remotely
tail -f logs/overnight-$(date +%Y%m%d).log
```

---

**Document End** — Expanded infrastructure supplement ready for integration with existing RIC Loop workflow.
