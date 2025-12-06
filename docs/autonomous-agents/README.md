# Autonomous Claude Code Development Guide

**Version**: 1.4.0 | **Last Updated**: December 3, 2025 | **Status**: Production Ready

This guide documents best practices for running Claude Code autonomously for extended (8+ hour) development sessions on this QuantConnect trading bot project.

**Related Documentation**:

- [CLAUDE.md](../../CLAUDE.md) - Main project instructions
- [UPGRADE-011 Research](../research/UPGRADE-011-OVERNIGHT-SESSIONS.md) - Overnight session enhancements
- [Session Notes Template](../../scripts/templates/session-notes-template.md) - Relay-race context template
- [Multi-Agent Commands](MULTI-AGENT-COMMANDS.md) - **NEW**: Quick reference for agent orchestration

---

## ⭐ December 2025 Updates - Enhanced Overnight Sessions (UPGRADE-011)

**Key enhancements for autonomous overnight development:**

| Feature | Description | Impact |
|---------|-------------|--------|
| **Continuous Mode** | `--continuous` flag blocks Claude from stopping until all tasks complete | Full overnight task completion |
| **Relay-Race Pattern** | `claude-session-notes.md` persists context across session restarts | No context loss on crashes |
| **Hook Timeouts** | Explicit timeouts (5-60s) on all hooks prevent blocking | Robust session lifecycle |
| **Jitter in Backoff** | ±25% randomization prevents thundering herd on restarts | Reliable auto-recovery |

### Quick Start (December 2025)

```bash
# NEW: Start continuous overnight session with crash recovery
./scripts/run_overnight.sh --continuous --with-recovery "Implement feature X"

# Monitor session context (relay-race file)
tail -f claude-session-notes.md

# Full autonomous overnight with Opus for complex tasks
./scripts/run_overnight.sh --continuous --with-recovery --model opus "Major refactoring"
```

**See**: [UPGRADE-011 Implementation Checklist](../research/UPGRADE-011-OVERNIGHT-SESSIONS.md) for full feature details.

---

## ⭐ November 2025 Updates - Start Here

Major releases in November 2025 have significantly improved autonomous agent workflows. **Prioritize these tools:**

| Priority | Tool | What's New | Impact |
|----------|------|-----------|--------|
| **#1** | Claude Code Sandbox | Native sandbox, 84% fewer prompts | Essential for overnight runs |
| **#2** | Claude Code on Web | Browser-based, GitHub integration | Start tasks remotely |
| **#3** | Kiro GA | Spec-driven, property-based testing | Structured development |
| **#4** | GitHub Copilot Subagents | Background PR generation | CI/CD automation |

### Quick Start (November 2025)

```bash
# NEW: Start overnight session with native sandbox (recommended)
claude --sandbox -p "Implement feature X" --output-format stream-json

# Enable sandbox in interactive mode
/sandbox

# Or with specific directory and network access
claude --sandbox --sandbox-allow-dirs "$(pwd)" \
  --sandbox-allow-network "api.quantconnect.com,github.com"
```

**Key Changes**:
- **Native sandbox** reduces permission prompts by 84%
- **No Docker required** for basic sandboxing (uses OS-level bubblewrap/seatbelt)
- **Built-in checkpointing** for session recovery
- **Web access** available at claude.ai for remote session initiation

See [COMPARISON.md - November 2025 Release Comparison](COMPARISON.md#november-2025-release-comparison) for full analysis.

---

## Table of Contents

1. [Overview](#overview)
2. [Checkpoint Recovery](#checkpoint-recovery)
3. [Claude Code Configuration](#claude-code-configuration)
4. [Sandbox Architecture](#sandbox-architecture)
5. [Session Management](#session-management)
6. [Safety Infrastructure](#safety-infrastructure)
7. [Multi-Agent Patterns](#multi-agent-patterns)
8. [Trading-Specific MCP Servers](#trading-specific-mcp-servers)
9. [Plan Limits and Rate Management](#plan-limits-and-rate-management)
10. [Durable Execution with Temporal](#durable-execution-with-temporal)
11. [Quick Start](#quick-start)

---

## Overview

### Why Autonomous Development?

- **Extended Sessions**: 8+ hour overnight coding without human supervision
- **Iterative Refinement**: Agent can run tests, fix issues, and iterate autonomously
- **Cost Efficiency**: Batch development during off-hours
- **Consistency**: Follows documented patterns without fatigue

### Real-World Success Stories

| Project | Duration | Result | Source |
|---------|----------|--------|--------|
| Rakuten ML Feature | 7 hours | 99.9% numerical accuracy | [Anthropic](https://www.anthropic.com/news/enabling-claude-code-to-work-more-autonomously) |
| FinovateApps Refactor | 7 nights | Weeks of work completed | Industry reports |
| Rackspace Modernization | 3 weeks | 52 weeks of work (using Kiro) | AWS Kiro |
| SaaS Platform (echodocs.ai) | Multi-session | Full NextJS + PostgreSQL | Developer blogs |

**Industry Adoption**: 2025 Stack Overflow Survey shows 78% of developers use or plan to use AI tools, 23% employ AI agents at least weekly.

### Key Principles

1. **Defense in Depth**: Multiple isolation layers (sandbox + watchdog + circuit breaker)
2. **Fail-Safe Defaults**: Always halt on unexpected states
3. **Auditability**: Complete logging of all actions
4. **Human Gates**: Critical operations require explicit approval

### Research Findings

**AIDev Dataset** (July 2025, [arXiv](https://arxiv.org/abs/2507.15003)): First large-scale study of autonomous coding agents - 456,000 PRs by 5 leading agents across 61,000 repositories.

**Speed vs Trust Gap** ([arXiv](https://arxiv.org/html/2509.06216v1)): Over 68% of agent-generated PRs face long delays or remain unreviewed, creating urgent need for review automation.

**SWE-bench Performance**: Top agents now resolve 70-74% of SWE-bench Verified issues, but only 23% on harder SWE-bench Pro tasks.

### Why Sandboxing Matters

NIST research (January 2025) found that AI agent hijacking attacks achieved **81% success rate** against unsandboxed agents (up from 11% baseline) ([source](https://www.nist.gov/news-events/news/2025/01/technical-blog-strengthening-ai-agent-hijacking-evaluations)). Sandboxing with network restrictions and least-privilege configurations is a critical defense.

---

## Checkpoint Recovery

Claude Code automatically tracks file edits, allowing instant rollback ([docs](https://docs.claude.com/en/docs/claude-code/checkpointing)).

### Quick Restore

- **Press Esc twice** (Esc + Esc) - Opens rewind menu
- **Use `/rewind` command** - Same menu access

### Restore Options

| Option | Behavior |
|--------|----------|
| **Conversation only** | Rewind messages, keep code changes |
| **Code only** | Revert files, keep conversation |
| **Both** | Restore to prior point completely |

### Limitations

- Only tracks Claude's direct file edits (not bash commands like `rm`, `mv`)
- Cannot undo external changes (git push, API calls, database changes)
- Retained for 30 days

**Best Practice**: Use checkpoints as "local undo" and Git as "permanent history".

---

## AGENTS.md Standard

The AGENTS.md standard is an emerging open format (adopted by 20,000+ GitHub repositories) for guiding AI coding agents ([spec](https://agents.md/), [GitHub](https://github.com/openai/agents.md)).

### Why Use AGENTS.md

- **Universal**: Works with Claude Code, OpenAI Codex, Cursor, Kiro, GitHub Copilot, and others
- **Machine-readable**: Provides context that complements human-facing README.md
- **Hierarchical**: Nested files in subdirectories provide package-level instructions

### Creating AGENTS.md for This Project

Create `AGENTS.md` at project root:

```markdown
# AGENTS.md - QuantConnect Trading Bot

## Project Overview
This is a Python algorithmic trading project for QuantConnect's LEAN platform.

## Build & Test
- Install: `pip install -r requirements.txt`
- Test: `pytest tests/ -v`
- Lint: `flake8 && mypy .`
- Backtest: `lean backtest algorithms/<name>.py`

## Code Style
- Python 3.10+ with type hints
- Google-style docstrings
- Max 100 chars/line
- Use `from AlgorithmImports import *` for QC algorithms

## Security
- NEVER commit .env or credentials
- NEVER deploy untested code to live trading
- Always validate data before trading decisions

## Testing Requirements
- Minimum 70% coverage
- All algorithms must have unit tests
- Run validation before backtest

## Architecture
- algorithms/ - Trading algorithms (QC compatible)
- models/ - Risk management, circuit breakers
- llm/ - Sentiment analysis ensemble
- execution/ - Order execution strategies
```

**Supported by**: OpenAI Codex, Cursor, GitHub Copilot, Kiro, Factory, and more.

---

## Claude Code Configuration

### Permissions System

Configure in `.claude/settings.json`:

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "Bash(pytest:*)",
      "Bash(python:*)",
      "Bash(git status:*)",
      "Bash(git diff:*)",
      "Bash(git log:*)",
      "Bash(lean backtest:*)"
    ],
    "deny": [
      "Bash(curl:*)",
      "Bash(wget:*)",
      "Bash(rm -rf:*)",
      "Read(./.env)",
      "Read(./config/credentials*)"
    ],
    "ask": [
      "Write",
      "Edit",
      "Bash(git push:*)",
      "Bash(git commit:*)"
    ]
  }
}
```

### CRITICAL BUG: Deny Rules Not Enforced

**Known Issue**: As of November 2025, `deny` rules in settings.json are NOT reliably enforced (GitHub issues #6699, #6631).

**Workaround**: Use PreToolUse hooks with exit code 2 to block operations:

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Edit|Write|Read",
      "hooks": [{
        "type": "command",
        "command": "python3 .claude/hooks/protect_files.py \"$TOOL_INPUT\""
      }]
    }]
  }
}
```

See [.claude/hooks/protect_files.py](../../.claude/hooks/protect_files.py) for implementation.

### CLI Flags for Autonomous Mode

Use these CLI flags for controlled autonomous operation:

```bash
# Full autonomous mode with tool allowlist
claude --dangerously-skip-permissions \
  --allowedTools "Edit,Write,Read,Glob,Grep,Bash(pytest:*),Bash(python:*),Bash(lean:*),Bash(git:*)" \
  --disallowedTools "Bash(rm -rf:*),Bash(curl:*),Bash(wget:*)"

# Alternative: Persistent configuration
claude config set allowedTools "Bash(git:*),Write,Read"
claude config add disallowedTools "Bash(rm:*)" "Bash(sudo:*)"
```

**Note**: Use colon before wildcard: `Bash(git:*)` NOT `Bash(git*)`

### Headless Mode (Automation & CI/CD)

Run Claude Code without interactive UI for automation ([docs](https://docs.claude.com/en/docs/claude-code/headless)):

```bash
# Basic headless execution
claude -p "implement feature X" --output-format stream-json

# With allowed tools for unattended operation
claude -p "run tests and fix failures" \
  --output-format stream-json \
  --allowedTools "Read,Write,Edit,Bash(pytest:*)"

# Multi-turn via stdin (jsonl format)
echo '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"Run tests"}]}}' \
  | claude -p --output-format=stream-json --input-format=stream-json
```

**Output Formats**:
- `text` (default): Human-readable
- `json`: Structured JSON with metadata
- `stream-json`: Real-time JSON streaming (recommended for automation)

**Monitoring Live Sessions**:
```bash
# Watch what Claude is doing in real-time
claude --dangerously-skip-permissions --output-format stream-json \
  | tee /var/log/claude-session.json \
  | jq '.type'
```

### Sandbox Activation

**Native Sandbox** (Beta - [source](https://www.anthropic.com/engineering/claude-code-sandboxing)):

```bash
# Enable sandbox mode
claude --sandbox

# With specific directory access
claude --sandbox --sandbox-allow-dirs "$(pwd)"

# With network access to specific hosts
claude --sandbox --sandbox-allow-network "api.quantconnect.com,github.com"
```

**Key Benefits**:
- **84% fewer permission prompts** in Anthropic's internal testing
- OS-level isolation via Linux bubblewrap / macOS Seatbelt
- No container overhead

**Other Methods**:
1. **Slash Command** (interactive): `/sandbox` in Claude Code session
2. **Docker Integration**: `docker sandbox run claude`
3. **Settings Configuration**: Set in `~/.claude/settings.json`

### Verified Sandbox Settings

```json
{
  "sandbox": {
    "enabled": true,
    "autoAllowBashIfSandboxed": true,
    "excludedCommands": ["docker"],
    "network": {
      "allowUnixSockets": ["/var/run/docker.sock"],
      "allowLocalBinding": true
    }
  }
}
```

**Note**: Network domain allowlists are managed via WebFetch permission rules, not sandbox.network.allowedDomains.

---

## Sandbox Architecture

### Isolation Hierarchy (Strongest to Weakest)

| Level | Technology | Isolation | Performance | Use Case |
|-------|------------|-----------|-------------|----------|
| 1 | Full VM (VMware/Hyper-V) | Complete | Slowest | Maximum security |
| 2 | Firecracker (E2B) | Near-VM | Fast | Cloud sandboxes |
| 3 | libkrun (Microsandbox) | MicroVM | Very fast | Self-hosted agents |
| 4 | gVisor/Kata (K8s) | Container+ | Fast | Enterprise Kubernetes |
| 5 | Docker Sandboxes | Container | Fastest | Quick development |
| 6 | WSL2 | Shared kernel | Native | NOT recommended |

### Docker Sandbox Commands

```bash
# Basic sandbox session
docker sandbox run claude

# With workspace mount
docker sandbox run -w ~/Projects/trading-bot claude

# Continue previous session
docker sandbox run claude -c

# With resource limits
docker sandbox run --memory=4g --cpus=2 claude
```

### Manual Network Isolation (iptables)

For maximum security without Docker, use iptables to restrict network access:

```bash
# Allow only essential hosts
sudo iptables -A OUTPUT -d api.quantconnect.com -j ACCEPT
sudo iptables -A OUTPUT -d github.com -j ACCEPT
sudo iptables -A OUTPUT -d api.anthropic.com -j ACCEPT
sudo iptables -A OUTPUT -j DROP

# Restore after session
sudo iptables -F OUTPUT
```

**Note**: This provides host-level isolation. For process-level isolation, use Docker or sandbox mode.

### ⚠️ WSL2 Security Warning

**WSL2 is NOT recommended for production autonomous agents**:
- Shares kernel with Windows host
- Network isolation is weaker than containers
- Process isolation is not as robust
- Escape attacks are easier than Docker/microVMs

If using WSL2 for development:
1. Run inside Docker Desktop (WSL2 backend) for better isolation
2. Use `--network=none` for Docker containers
3. Never store production credentials in WSL2 environment
4. Consider Microsandbox or E2B for production workloads

### Current Status (November 2025)

- **Docker Desktop 4.50**: Sandboxes use containers (NOT microVMs yet)
- **MicroVM Support**: On roadmap, not yet released
- **Recommendation**: Use Docker sandboxes for development, Microsandbox for production

### Microsandbox Setup (Self-Hosted MicroVMs)

```bash
# Install microsandbox
curl -fsSL https://get.microsandbox.dev | sh

# Run Claude Code in microVM
microsandbox run --image claude-code:latest \
  --memory 4096 \
  --cpus 2 \
  --network-policy deny-all \
  --mount ./workspace:/workspace
```

---

## Session Management

### Multi-Session Pattern (Recommended)

For 8+ hour sessions, use the Initializer + Coding Agent pattern:

```
┌─────────────────────┐
│  Initializer Agent  │ ← Starts session, writes plan
│  (Short-lived)      │
└─────────┬───────────┘
          │ writes claude-progress.txt
          ▼
┌─────────────────────┐
│   Coding Agent      │ ← Executes plan autonomously
│   (Long-running)    │
└─────────┬───────────┘
          │ updates progress file
          ▼
┌─────────────────────┐
│   Watchdog Process  │ ← External monitoring
│   (Separate PID)    │
└─────────────────────┘
```

### Progress File Protocol

Create `claude-progress.txt` for session continuity:

```markdown
# Session: 2025-11-27-overnight
# Started: 2025-11-27T22:00:00Z
# Goal: Implement two-part spread execution improvements

## Completed
- [x] Added position balance tracking
- [x] Implemented quick cancel (2.5s timeout)
- [x] Created fill rate predictor

## In Progress
- [ ] Autonomous parameter optimization

## Next Steps
- [ ] Add IV percentile filtering
- [ ] Implement credit maximizer

## Blockers
None

## Notes
Fill rate observed at 32% with 1-contract orders.
```

### Context Management

**60% Rule**: Anthropic recommends never exceeding 60% context utilization. Beyond this, agents become less effective.

When context reaches 60% capacity:
1. Summarize current state to progress file
2. Commit work in progress
3. Spawn new session with context handoff

### Phase-Based Workflow

Use `/clear` between phases to reset context:

```
Phase 1: Research (explore codebase, understand requirements)
/clear

Phase 2: Plan (design approach, identify files to modify)
/clear

Phase 3: Implement (write code, one feature at a time)
/clear

Phase 4: Validate (run tests, fix issues)
```

### Git Worktrees for Parallel Sessions

Run multiple Claude sessions on different features simultaneously:

```bash
# Create separate worktrees for parallel Claude sessions
git worktree add ../trading-bot-feature-a -b feature/credit-maximizer
git worktree add ../trading-bot-feature-b -b feature/position-balancer

# Run Claude in each worktree
cd ../trading-bot-feature-a && claude
cd ../trading-bot-feature-b && claude
```

### Stuck Detection

Detect if an agent is stuck in a loop using similarity analysis:

```python
from difflib import SequenceMatcher

def detect_stuck_agent(messages: list, threshold: int = 3, similarity_cutoff: float = 0.85) -> bool:
    """
    Returns True if last N messages are too similar.

    Uses SequenceMatcher for fuzzy matching to detect near-identical responses
    that indicate the agent is stuck in a loop.

    Args:
        messages: List of message objects with .content attribute
        threshold: Number of recent messages to check
        similarity_cutoff: Ratio above which messages are considered "same" (0.0-1.0)

    Returns:
        True if agent appears stuck, False otherwise
    """
    if len(messages) < threshold:
        return False

    last_messages = [msg.content for msg in messages[-threshold:]]

    # Check pairwise similarity
    similar_count = 0
    for i in range(len(last_messages) - 1):
        ratio = SequenceMatcher(None, last_messages[i], last_messages[i + 1]).ratio()
        if ratio > similarity_cutoff:
            similar_count += 1

    # Stuck if most consecutive pairs are similar
    return similar_count >= threshold - 1
```

**When stuck is detected**:
1. Log warning with last few messages for debugging
2. Try injecting a "step back and reassess" prompt
3. If still stuck after 2 attempts, gracefully terminate and save state
4. Alert via watchdog webhook (if configured)

### Session Startup Script

See [scripts/run_overnight.sh](../../scripts/run_overnight.sh) for the complete overnight startup script.

---

## Safety Infrastructure

### External Watchdog (Required)

The watchdog runs as a separate process monitoring the Claude Code session:

```python
# See scripts/watchdog.py for full implementation
watchdog = Watchdog(
    max_runtime_hours=10,
    max_idle_minutes=30,
    max_cost_usd=50.0,
    checkpoint_interval_minutes=15
)
watchdog.start()
```

### Overnight Session Configuration

Use this Python dictionary format for programmatic session configuration:

```python
overnight_config = {
    # Safety limits
    "max_runtime_hours": 10,
    "max_idle_minutes": 30,
    "max_cost_usd": 50.0,
    "checkpoint_interval_minutes": 15,

    # Model preferences
    "model": "sonnet",  # or "opus" for complex tasks
    "thinking_mode": "normal",  # or "ultrathink" for deep analysis

    # Tool permissions
    "allowed_tools": [
        "Read", "Write", "Edit", "Glob", "Grep",
        "Bash(pytest:*)", "Bash(python:*)", "Bash(git:*)",
        "Bash(lean:*)"
    ],
    "disallowed_tools": [
        "Bash(rm -rf:*)", "Bash(curl:*)", "Bash(wget:*)"
    ],

    # Progress tracking
    "progress_file": "claude-progress.txt",
    "checkpoint_commits": True,
    "commit_prefix": "[AUTO]",

    # Alerts
    "slack_webhook": None,  # Optional: "https://hooks.slack.com/..."
    "alert_email": None,    # Optional: "alerts@example.com"

    # Convergence criteria
    "max_iterations": 50,
    "success_threshold": 0.95,  # 95% test pass rate
    "stuck_threshold": 3,       # Consecutive similar responses
}
```

Save as `config/overnight.py` and import in startup scripts.

### Circuit Breaker Integration

Use the project's circuit breaker for trading-specific safety:

```python
from models.circuit_breaker import create_circuit_breaker

breaker = create_circuit_breaker(
    max_daily_loss=0.03,
    max_drawdown=0.10,
    max_consecutive_losses=5,
    require_human_reset=True
)

# In autonomous loop
if not breaker.can_trade():
    log.warning("Circuit breaker tripped - halting")
    sys.exit(1)
```

### Budget Control with LiteLLM

Configure via YAML for production deployments:

```yaml
# litellm_config.yaml
general_settings:
  master_key: sk-your-master-key

litellm_settings:
  max_budget: 50  # USD per period
  budget_duration: 24h

model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      max_budget: 40
      budget_duration: 24h

alerting:
  webhook: "https://your-slack-webhook"
  alert_types: ["budget"]
  budget_alerts:
    - 50   # Alert at 50%
    - 75   # Alert at 75%
    - 90   # Alert at 90%
    - 100  # Alert at 100%
```

Run the proxy:
```bash
litellm --config litellm_config.yaml --port 4000
```

**LiteLLM Features**: Per-key/team/model budgets, 8ms P95 latency, 100+ LLM providers, Datadog integration.

### AWS Agentic AI Security Scoping Matrix

Use this framework to classify your autonomous agent's risk level:

| Scope | Name | Description | This Project |
|-------|------|-------------|--------------|
| 1 | No Agency | Read-only, human-triggered, predefined paths | Basic querying |
| 2 | Prescribed Agency | Can recommend actions, requires HITL approval | Supervised sessions |
| 3 | Supervised Agency | Autonomous execution after human initiation | **Overnight runs** |
| 4 | Full Agency | Self-initiating, minimal oversight | Not recommended |

**Key Concepts**:
- **Agency**: What the system is *allowed* to do (capabilities/permissions)
- **Autonomy**: How *independently* it can decide and act

**Recommendation**: Use Scope 3 (Supervised Agency) for overnight sessions - autonomous execution within bounded parameters after human initiation.

**Graceful Degradation**: Design systems to automatically reduce autonomy levels when security events are detected.

### Required Safety Checklist

- [ ] External watchdog running (separate process)
- [ ] Cost budget configured
- [ ] Circuit breaker active
- [ ] Progress file initialized
- [ ] Checkpoint commits enabled
- [ ] Network restrictions applied (if sandbox)

### Enterprise Security Best Practices

From industry analysis ([Martin Fowler](https://martinfowler.com/articles/agentic-ai-security.html), [Skywork](https://skywork.ai/blog/agentic-ai-safety-best-practices-2025-enterprise/)):

**Multi-Layer Defense**:

| Plane | Implementation |
|-------|----------------|
| **Execution** | gVisor/GKE Sandbox; Firecracker for code-exec; WASI for plugins |
| **Data/Memory** | Vector stores on private subnets; PII redaction; memory TTLs |
| **Observability** | OpenTelemetry GenAI spans; SIEM aggregation; SOAR automations |
| **Assurance** | Sigstore/Cosign verification; SLSA attestations; SBOMs in CI/CD |

**Critical Practices**:

1. **Default-Deny Tool Permissions**: Treat every tool as potential escalation path
2. **Egress Allowlists**: Block default outbound internet; allow specific domains only
3. **Resource Limits**: CPU/memory quotas, ulimits, wall-clock timeboxing
4. **Ephemeral Environments**: Read-only root, temporary work directories
5. **No Credentials in Files**: Use environment variables, 1Password CLI for secrets
6. **Temporary Privilege Escalation**: Short-lived access tokens, read-only defaults
7. **Human Confirmation**: Gate irreversible actions ("I will create X with Y - proceed?")

### Defense-in-Depth Layers

| Layer | Implementation | Purpose |
|-------|----------------|---------|
| 1. Identity | SPIFFE/SPIRE, short-lived secrets | Unique workload identities |
| 2. Sandboxing | gVisor/Firecracker, network allowlists | Code isolation |
| 3. Observability | OpenTelemetry GenAI conventions | Runtime monitoring |
| 4. RAG Safety | Content sanitization, PII/DLP controls | Memory protection |
| 5. Human Oversight | Risk-tier actions, playbooks | Emergency intervention |

### OpenTelemetry GenAI Observability

OpenTelemetry has semantic conventions for AI agent observability ([docs](https://opentelemetry.io/docs/specs/semconv/gen-ai/)):

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("agent_task") as span:
    span.set_attribute("gen_ai.system", "anthropic")
    span.set_attribute("gen_ai.request.model", "claude-sonnet-4")
    span.set_attribute("gen_ai.agent.id", "overnight-trader-001")
    span.set_attribute("gen_ai.agent.name", "Trading Bot Developer")
    span.set_attribute("gen_ai.operation.name", "invoke_agent")

    # Your agent code here
    result = execute_agent_action()

    span.set_attribute("gen_ai.response.token_count", result.tokens)
```

**Key Agent Attributes**:
- `gen_ai.agent.id` - Unique identifier
- `gen_ai.agent.name` - Human-readable name
- `gen_ai.agent.description` - Free-form description
- `gen_ai.operation.name` - `create_agent`, `invoke_agent`

---

## Claude Agent SDK

The Claude Agent SDK (formerly Claude Code SDK) provides the same agent harness that powers Claude Code for building custom agents ([docs](https://docs.claude.com/en/docs/agent-sdk/python)).

### Installation

```bash
# Python
pip install claude-agent-sdk

# TypeScript/Node.js
npm install @anthropic-ai/claude-agent-sdk
```

**Note**: The Claude Code CLI is bundled - no separate installation required.

### Basic Usage

```python
from claude_agent_sdk import ClaudeAgent, ClaudeAgentOptions

options = ClaudeAgentOptions(
    allowed_tools=["Read", "Write", "Edit", "Bash(pytest:*)"],
    max_turns=50,
    auto_compact=True  # Context management
)

agent = ClaudeAgent(options)

# Run autonomous task
result = await agent.run(
    "Run all tests, fix any failures, then implement the TODO items"
)
```

### Key Features

- **Context Management**: Auto-compaction when context fills
- **Rich Tool Ecosystem**: Same tools as Claude Code
- **MCP Extensibility**: Connect to MCP servers
- **Session Management**: Persist and resume sessions
- **Subagent Spawning**: Create specialized sub-agents for parallel work

### Migration from Claude Code SDK

```bash
pip uninstall claude-code-sdk
pip install claude-agent-sdk
```

Update imports: `claude_code_sdk` → `claude_agent_sdk`

---

## Multi-Agent Patterns

### Linear Pipeline

```
Scanner Agent → Analysis Agent → Execution Agent → Validation Agent
```

Best for: Simple workflows with clear handoffs

### Supervisor Pattern

```
        ┌─────────────────┐
        │   Supervisor    │
        │   (Orchestrator)│
        └───────┬─────────┘
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌───────┐  ┌───────┐  ┌───────┐
│Scanner│  │Analyst│  │Tester │
└───────┘  └───────┘  └───────┘
```

Best for: Complex workflows needing coordination

### Recommended Frameworks

| Framework | Best For | Trading Suitability |
|-----------|----------|---------------------|
| Claude Agent SDK | Native Claude integration | Excellent |
| LangGraph | State machine workflows | Good |
| CrewAI | Role-based collaboration | Good |
| Temporal.io | Durable execution | Excellent |
| AutoGen | Multi-agent debate | Moderate |

---

## Trading-Specific MCP Servers

### QuantConnect MCP (Primary)

```json
{
  "mcpServers": {
    "quantconnect": {
      "command": "docker",
      "args": ["run", "-i", "--rm",
               "-e", "QC_USER_ID",
               "-e", "QC_API_TOKEN",
               "quantconnect/mcp-server:latest"]
    }
  }
}
```

**Features**: 60+ API endpoints, full trading lifecycle, backtesting, live deployment

### Alpaca MCP (Official)

```json
{
  "mcpServers": {
    "alpaca": {
      "command": "npx",
      "args": ["-y", "@anthropic/alpaca-mcp"],
      "env": {
        "ALPACA_API_KEY": "${ALPACA_API_KEY}",
        "ALPACA_SECRET_KEY": "${ALPACA_SECRET_KEY}"
      }
    }
  }
}
```

**Features**: Stocks, ETFs, crypto, options trading

### Finviz MCP (Community)

```json
{
  "mcpServers": {
    "finviz": {
      "command": "uvx",
      "args": ["finviz-mcp"],
      "env": {
        "FINVIZ_API_KEY": "${FINVIZ_API_KEY}"
      }
    }
  }
}
```

**Features**: Stock screening, SEC filings, insider trading data (requires Elite subscription)

### Alpha Vantage MCP (Official)

```json
{
  "mcpServers": {
    "alphavantage": {
      "command": "npx",
      "args": ["-y", "@anthropic/alphavantage-mcp"],
      "env": {
        "ALPHA_VANTAGE_API_KEY": "${ALPHA_VANTAGE_API_KEY}"
      }
    }
  }
}
```

**Features**: Stock data, forex, crypto, technical indicators, fundamental data

---

## Plan Limits and Rate Management

### Claude Code Plan Tiers (Verified November 2025)

| Plan | Cost | 5-Hour Prompts | Weekly Sonnet 4 | Weekly Opus 4 |
|------|------|----------------|-----------------|---------------|
| Pro | $20/mo | 10-40 | 40-80 hrs | None |
| Max 5x | $100/mo | 50-200 | 140-280 hrs | 15-35 hrs |
| Max 20x | $200/mo | 200-800 | 240-480 hrs | 24-40 hrs |

### Model Switching Behavior

- **Max 5x**: Switches from Opus 4 to Sonnet 4 at 20% usage
- **Max 20x**: Switches from Opus 4 to Sonnet 4 at 50% usage
- Manual control via `/model` command, but Opus consumes capacity faster

**Value Note**: Max 20x costs 2x Max 5x but only provides ~1.7x Sonnet hours and ~1.3x Opus hours. Dollar-for-dollar, Max 5x may be better value for moderate usage.

### Rate Limit Architecture

```
5-Hour Rolling Window
├── Prompt counter (resets every 5 hours)
└── Weekly Ceiling
    ├── Sonnet 4 hours (usage × duration)
    └── Opus 4 hours (separate quota)
```

### Optimization Strategies

1. **Use Sonnet 4 for routine tasks**: 10x more weekly hours than Opus
2. **Reserve Opus for complex analysis**: Architecture, deep debugging
3. **Monitor rolling window**: Pace requests to avoid 5-hour limit
4. **Batch similar operations**: Reduce prompt count
5. **Enable auto-compact**: Reduce context size automatically

### Monitoring Usage

```bash
# Check current usage (in Claude Code)
/usage

# API usage tracking
curl -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
  https://api.anthropic.com/v1/usage
```

---

## Durable Execution with Temporal

### Why Temporal?

- **Fault Tolerance**: Automatic retry with state preservation
- **Long-Running**: Hours/days without losing progress
- **Visibility**: Full execution history and debugging
- **Production Proven**: Used by Replit Agent, OpenAI Codex

### Production Case Studies

**Replit Agent** ([case study](https://temporal.io/resources/case-studies/replit-uses-temporal-to-power-replit-agent-reliably-at-scale)):
- Every Agent is its own Temporal Workflow
- Workflow IDs ensure only one active agent per user session
- "Temporal has never been the bottleneck" as usage scaled massively

**OpenAI Codex**: Uses Temporal in production, handling millions of requests.

### OpenAI Agents SDK + Temporal (July 2025)

New official integration ([announcement](https://temporal.io/blog/announcing-openai-agents-sdk-integration)):

```python
from temporalio import workflow
from openai_agents import Agent

@workflow.defn
class AgentWorkflow:
    @workflow.run
    async def run(self, task: str):
        # Agent calls automatically get durable execution
        agent = Agent(name="coding-agent")
        result = await workflow.execute_activity(
            agent.run,
            task,
            start_to_close_timeout=timedelta(hours=2),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        return result
```

**Benefits**: Built-in retry, state persistence, crash recovery, deterministic replay.

### Basic Workflow

```python
from temporalio import workflow, activity
from datetime import timedelta

@activity.defn
async def run_backtest(algorithm: str) -> dict:
    # Execute backtest
    result = await lean_cli.backtest(algorithm)
    return result

@activity.defn
async def analyze_results(results: dict) -> dict:
    # Analyze with Claude
    analysis = await claude.analyze(results)
    return analysis

@workflow.defn
class TradingResearchWorkflow:
    @workflow.run
    async def run(self, params: dict) -> dict:
        # Run backtest with retry
        results = await workflow.execute_activity(
            run_backtest,
            params["algorithm"],
            start_to_close_timeout=timedelta(hours=2),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )

        # Analyze results
        analysis = await workflow.execute_activity(
            analyze_results,
            results,
            start_to_close_timeout=timedelta(minutes=30)
        )

        return analysis
```

### Temporal Setup

```bash
# Install Temporal CLI
curl -sSf https://temporal.download/cli.sh | sh

# Start local development server
temporal server start-dev

# Run worker
python -m trading_workflows.worker
```

---

## Quick Start

### 1. Configure Claude Code

Settings are already configured in `.claude/settings.json` with:

- Extended bash timeouts (30 min default, 2 hr max)
- Auto-format hooks for Python files
- File protection hooks
- Session lifecycle hooks (Stop, PreCompact)
- RIC Loop workflow integration

### 2. Install Dependencies

```bash
pip install psutil litellm requests
```

### 3. Start Overnight Session

```bash
# Basic overnight session
./scripts/run_overnight.sh "Implement feature X"

# With crash recovery enabled
./scripts/run_overnight.sh --with-recovery "Implement feature X"

# With Opus model for complex work
./scripts/run_overnight.sh --model opus --with-recovery "Major refactoring"

# View available options
./scripts/run_overnight.sh --help
```

### 4. Schedule Automated Sessions

```bash
# Interactive cron setup
./scripts/setup_cron.sh add

# Quick setup: 11 PM weekdays
./scripts/setup_cron.sh add 23 1-5 "Continue development"

# View example configurations
./scripts/setup_cron.sh examples
```

### 5. Recovery Commands

```bash
# Create test-validated checkpoint
./scripts/checkpoint.sh auto

# Recover from failed session
./scripts/checkpoint.sh recover

# Periodic checkpoint (for cron)
./scripts/checkpoint.sh periodic 30
```

### Continuous Loop Workflows

Several open-source projects enable autonomous overnight development:

#### continuous-claude ([GitHub](https://github.com/AnandChowdhary/continuous-claude))

Orchestrates Claude Code in a continuous loop, creating PRs, waiting for CI, and merging:

```bash
# Clone and configure
git clone https://github.com/AnandChowdhary/continuous-claude
cd continuous-claude

# Configure your goal in GOAL.md
echo "Implement two-part spread execution" > GOAL.md

# Run overnight (defaults to 50 iterations)
./run.sh
```

**Key insight**: Uses shared markdown as external memory - Claude records what it did and what's next.

#### ralph-claude-code ([GitHub](https://github.com/frankbria/ralph-claude-code))

Autonomous development with intelligent exit detection:

```bash
# Live dashboard monitoring
# Configurable timeouts (1-120 minutes)
# Prevents infinite loops and API overuse
```

#### claude-flow ([GitHub](https://github.com/ruvnet/claude-flow))

Enterprise multi-agent orchestration with 64-agent swarm architecture:

```bash
# Install
npx claude-flow@alpha init --force

# Features: 25 Claude Skills, Hive-Mind Intelligence, 100+ MCP tools
```

#### claude-code-workflow ([GitHub](https://github.com/doodledood/claude-code-workflow))

Spec-driven overnight automation:

```
10 PM: /spec "Build user management system"
[Accept plan] → [Sleep]
7 AM: [Review implementation] → Feature complete
```

**Warning**: From real-world experience:
- "Reviewing 40-50 overnight runs is not fun"
- "1/4 of the time you may just throw it away"
- Always run in sandbox with iteration limits

### Alternative: Autonomous Runner Script

Create a dedicated autonomous runner for customized sessions:

```bash
#!/bin/bash
# scripts/autonomous_dev.sh

export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"

claude --dangerously-skip-permissions \
  --allowedTools "Read,Write,Edit,Glob,Grep,Bash(pytest:*),Bash(git:*),Bash(lean:*)" \
  -p "$(cat <<'EOF'
You are working on a QuantConnect trading bot.

Tasks:
1. Run pytest and fix any failing tests
2. Review code quality with flake8
3. Implement any TODO items in the codebase
4. Create git commits for completed work

Stop conditions:
- Stop after 4 hours
- Stop if test pass rate drops below 70%
- Create checkpoint commits every 30 minutes
EOF
)"
```

Run in isolated environment:
```bash
# With Container Use (parallel agents)
container-use run ./scripts/autonomous_dev.sh

# Or with Docker (single agent)
docker-compose run --rm sandbox ./scripts/autonomous_dev.sh
```

### 4. Monitor Progress

```bash
# Watch progress file
tail -f claude-progress.txt

# Check watchdog logs
tail -f logs/watchdog.log
```

### 5. Review Results

```bash
# Check git log for checkpoint commits
git log --oneline -20

# Review any generated reports
ls -la reports/
```

---

## Related Documentation

- [COMPARISON.md](COMPARISON.md) - Detailed tool comparisons
- [INSTALLATION.md](INSTALLATION.md) - Step-by-step installation
- [TODO.md](TODO.md) - Implementation checklist
- [../strategies/README.md](../strategies/README.md) - Trading strategies
- [../../CLAUDE.md](../../CLAUDE.md) - Project-wide Claude instructions

---

## Advanced Hooks for Overnight Sessions

The project includes specialized hooks for autonomous operation (December 2025, enhanced with UPGRADE-011):

### Stop Hook (Session Completion)

Runs when Claude Code session ends - `.claude/hooks/session_stop.py`:

- **Continuous Mode Support**: When `CONTINUOUS_MODE=1`, checks for pending tasks (`- [ ]` in progress file)
- **Exit Code 2**: Blocks Claude from stopping if tasks remain (tells Claude to continue)
- Calculates session statistics (runtime, commits, files changed)
- Creates final checkpoint commit
- Sends completion notification via Discord/Slack
- Logs session summary to `logs/session-history.jsonl`

**Continuous Mode Flow**:
```
Stop Hook Triggered
       │
       ├── CONTINUOUS_MODE=1?
       │   ├── YES → Check pending tasks
       │   │         ├── Tasks remain → Exit code 2 (block stop, continue working)
       │   │         └── No tasks → Normal exit (cleanup, commit, notify)
       │   └── NO → Normal exit
```

### PreCompact Hook (Context Backup)

Runs before context compaction (~70% of 200K tokens) - `.claude/hooks/pre_compact.py`:

- **Transcript Path from Hook Input**: Reads `transcript_path` from `CLAUDE_TOOL_INPUT` (preferred)
- Falls back to searching `~/.claude/` if hook input not available
- Backs up full transcript to `logs/transcripts/`
- Creates checkpoint commit before summarization
- Logs compaction events to `logs/compaction-history.jsonl`
- Sends context warning notification

### Relay-Race Pattern (Context Persistence)

The relay-race pattern ensures context survives session restarts:

**File**: `claude-session-notes.md` (project root)

```markdown
# Claude Session Notes

## Current Goal
Implement feature X with full test coverage

## Key Decisions Made
- Chose approach A over B because of performance
- Using library X for Y functionality

## Important Discoveries
- Found existing code in module Z that can be reused
- API limit is 100 requests/minute

## Next Steps
- [ ] Complete unit tests for module A
- [ ] Update documentation

---
**Last Updated**: 2025-12-03T22:00:00Z
**Session ID**: 20251203-220000
```

**How It Works**:
1. `run_overnight.sh` initializes notes file (preserves if exists)
2. Claude reads notes at session start for context
3. Claude updates notes with key decisions/discoveries
4. On crash/restart, next session reads accumulated context
5. Context persists across multiple session restarts

### Auto-Resume Script

Automatic crash recovery - `scripts/auto-resume.sh`:

```bash
# Start auto-resume monitoring
./scripts/auto-resume.sh --max-restarts 5 --backoff-base 30

# Environment configuration
AUTO_RESUME_MAX_RESTARTS=5      # Maximum restart attempts
AUTO_RESUME_BACKOFF_BASE=30     # Base backoff seconds
AUTO_RESUME_CHECK_INTERVAL=60   # Health check interval
```

**Features**:

- Monitors Claude process health
- Auto-restarts on crash with exponential backoff **with jitter** (±25% randomization)
- Jitter prevents "thundering herd" problem when multiple processes restart
- Reads progress from `claude-progress.txt` to provide context
- Tracks restart attempts and prevents infinite loops
- Sends notifications on restart attempts

**Backoff Formula**:

```text
delay = base_delay × 2^restart_count ± 25% jitter
        (capped at 600 seconds, minimum 10 seconds)
```

### Hook Configuration

All hooks are configured in `.claude/settings.json` with **explicit timeouts** (UPGRADE-011):

```json
{
  "hooks": {
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "python3 .claude/hooks/session_stop.py",
        "statusMessage": "Running session cleanup and notification",
        "timeout": 60
      }]
    }],
    "PreCompact": [{
      "hooks": [{
        "type": "command",
        "command": "python3 .claude/hooks/pre_compact.py",
        "statusMessage": "Backing up transcript before context compaction",
        "timeout": 60
      }]
    }]
  }
}
```

**Hook Timeout Guidelines**:

| Hook Type | Recommended Timeout | Purpose |
|-----------|---------------------|---------|
| PreToolUse (validation) | 5-15 seconds | Quick checks before tool execution |
| PostToolUse (formatting) | 15-30 seconds | Code formatting, validation |
| Stop (cleanup) | 60 seconds | Git commits, notifications |
| PreCompact (backup) | 60 seconds | Transcript backup, checkpoints |
| SessionStart | 5 seconds | Quick context loading |
| UserPromptSubmit | 5 seconds | Quick analysis |

---

## Changelog

| Date | Version | Change |
|------|---------|--------|
| 2025-12-03 | 1.4.0 | **UPGRADE-011**: Added continuous mode, relay-race pattern, hook timeouts, jitter in backoff |
| 2025-12-03 | 1.3.0 | Added Stop hook continuous mode support, transcript path from hook input |
| 2025-12-03 | 1.2.0 | Added Stop hook, PreCompact hook, and auto-resume script (PDF guide alignment) |
| 2025-11-28 | 1.1.0 | Added November 2025 release priorities (sandbox, Kiro GA, Copilot subagents) |
| 2025-11 | 1.0.0 | Initial comprehensive guide |
