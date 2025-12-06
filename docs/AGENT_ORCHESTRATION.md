# Multi-Agent Orchestration Suite

A comprehensive system for spawning, coordinating, and managing multiple Claude agents with intelligent model selection and autonomous capabilities.

## Quick Start Commands

| Command | Description | Agents |
|---------|-------------|--------|
| `/agents auto <task>` | **Intelligent auto-routing** - analyzes task and selects best agents | Auto |
| `/agent-quick <task>` | Single fast agent dispatch | 1 |
| `/agent-swarm <topic>` | Massive parallel exploration | 8 |
| `/agent-consensus <decision>` | Multi-agent voting/decision | 3 |
| `/agent-implement <feature>` | Full implementation pipeline | 4 |
| `/agent-compare <options>` | Compare approaches | 2-3 |

## Model Selection Guide

| Model | Speed | Cost | Best For |
|-------|-------|------|----------|
| **haiku** | Fastest | Lowest | File search, grep, simple checks, quick tasks |
| **sonnet** | Medium | Medium | Code review, implementation, balanced analysis |
| **opus** | Slowest | Highest | Architecture design, deep reasoning, critical decisions |

## Intelligent Auto-Selection

The `/agents auto` command automatically analyzes tasks and selects optimal agents:

```bash
/agents auto find all error handling code          # -> 3 haiku search agents
/agents auto review the execution module           # -> 4 review agents (haiku+sonnet)
/agents auto design caching strategy               # -> 2 agents (haiku research + sonnet design)
/agents auto should we refactor authentication     # -> 3 consensus agents (sonnet)
```

### Auto-Classification

| Task Pattern | Classification | Model | Agent Count |
|--------------|----------------|-------|-------------|
| find, search, where | SEARCH | haiku | 3 |
| review, check, audit | REVIEW | haiku+sonnet | 4 |
| implement, create, build | BUILD | sonnet | 2-4 |
| design, architect, plan | DESIGN | sonnet | 2 |
| trading, risk, execution | TRADING | sonnet | 3 |
| decide, should, choose | CONSENSUS | sonnet | 3 |

## Pre-Built Workflows

| Workflow | Pattern | Agents | Use Case |
|----------|---------|--------|----------|
| `code_review` | Parallel | 4 | Security + Types + Tests + Architecture |
| `multi_search` | Parallel | 3 | Code + Docs + Tests search |
| `trading_review` | Parallel | 3 | Risk + Execution + Backtest |
| `critical_decision` | Consensus | 3 | Multi-perspective voting |
| `implementation_pipeline` | Sequential | 4 | Research -> Plan -> Implement -> Review |

## Workflow Patterns

### Parallel Pattern

All agents run simultaneously:

```python
Task(model="haiku", prompt="Security scan...")
Task(model="haiku", prompt="Type check...")
Task(model="sonnet", prompt="Architecture review...")
# All three start immediately
```

### Sequential Pattern

Output chains to next agent:

```text
Research Agent -> Plan Agent -> Implement Agent -> Review Agent
```

### Consensus Pattern

Multiple agents vote independently:

```text
Agent 1: APPROVE (85% confidence)
Agent 2: APPROVE (70% confidence)
Agent 3: NEEDS_MORE_INFO
Result: 66% consensus -> APPROVED
```

## Using the Task Tool

```python
Task(
    subagent_type="Explore",    # or "Plan", "general-purpose"
    model="haiku",              # or "sonnet", "opus"
    description="Short desc",   # 5-10 words
    prompt="Detailed task..."   # Full instructions
)
```

## Agent Templates

### Search Agents (haiku)

- `CodeFinder` - Search source code
- `DocFinder` - Search documentation
- `TestFinder` - Search test files

### Review Agents (haiku/sonnet)

- `SecurityScanner` - Vulnerability scanning
- `TypeChecker` - Type hint validation
- `TestAnalyzer` - Coverage analysis
- `Architect` - Architecture review

### Trading Agents (sonnet)

- `RiskReviewer` - Risk management check
- `ExecutionReviewer` - Order execution review
- `BacktestReviewer` - Look-ahead bias check

### Deep Analysis (opus)

- `DeepArchitect` - Comprehensive architecture
- `CriticalReviewer` - Production code review

## Cost-Effective Strategies

1. **Search tasks**: Always use `haiku` (fastest, cheapest)
2. **Batch searches**: Spawn 3-8 haiku agents in parallel
3. **Code review**: `haiku` for checks, `sonnet` for recommendations
4. **Critical decisions**: Use `sonnet` for consensus, `opus` only when required
5. **Implementation**: `haiku` research + `sonnet` implementation

## Files

| File | Purpose |
|------|---------|
| `.claude/hooks/agents/agent_orchestrator.py` | **Main orchestration engine** (1000+ lines) |
| `.claude/config/agents.json` | Configuration and preferences |
| `.claude/commands/agents.md` | Master `/agents` command |
| `.claude/commands/agent-auto.md` | Intelligent auto-routing |
| `.claude/commands/agent-quick.md` | Single agent dispatch |
| `.claude/commands/agent-swarm.md` | Massive parallel exploration |
| `.claude/commands/agent-consensus.md` | Multi-agent voting |
| `.claude/commands/agent-implement.md` | Implementation pipeline |
| `.claude/commands/agent-compare.md` | Option comparison |
| `.claude/commands/agent-status.md` | System status |

## CLI Usage

```bash
# Show help
python3 .claude/hooks/agents/agent_orchestrator.py help

# List all workflows and agents
python3 .claude/hooks/agents/agent_orchestrator.py list

# Auto-select agents for a task
python3 .claude/hooks/agents/agent_orchestrator.py auto "find all auth code"

# Generate Task calls for a workflow
python3 .claude/hooks/agents/agent_orchestrator.py generate code_review

# Show statistics
python3 .claude/hooks/agents/agent_orchestrator.py status
```
