# Multi-Agent Command Reference

Quick reference for the multi-agent orchestration suite.

---

## Command Summary

| Command | Agents | Pattern | Use Case |
|---------|--------|---------|----------|
| `/agents auto <task>` | Auto | Auto | **Best for most tasks** - intelligent routing |
| `/agent-quick <task>` | 1 | Single | Fast, simple tasks |
| `/agent-swarm <topic>` | 8 | Parallel | Deep exploration |
| `/agent-consensus <decision>` | 3 | Voting | Critical decisions |
| `/agent-implement <feature>` | 4 | Pipeline | Full implementation |
| `/agent-compare A vs B` | 2-3 | Parallel | Option comparison |
| `/agent-status` | - | - | System statistics |
| `/parallel-review <files>` | 4 | Parallel | Code review |
| `/multi-search <query>` | 3 | Parallel | Codebase search |
| `/trading-review <files>` | 3 | Parallel | Trading safety |

---

## Model Selection

| Model | Speed | Cost | Best For |
|-------|-------|------|----------|
| **haiku** | Fast | Low | Search, grep, simple checks |
| **sonnet** | Medium | Medium | Implementation, review |
| **opus** | Slow | High | Architecture, critical |

---

## Quick Examples

### Search Tasks

```bash
/agents auto find all error handling code
/agent-swarm authentication
/multi-search "circuit breaker"
```

### Review Tasks

```bash
/agents auto review the execution module
/parallel-review algorithms/
/trading-review execution/
```

### Implementation Tasks

```bash
/agent-implement add IV rank filter
/agents auto implement volume spike detector
```

### Decision Tasks

```bash
/agent-consensus should we refactor auth
/agent-compare Redis vs Memcached
```

---

## Auto-Classification

The `/agents auto` command classifies tasks automatically:

| Task Pattern | Classification | Model | Agents |
|--------------|----------------|-------|--------|
| find, search, where | SEARCH | haiku | 3 |
| review, check, audit | REVIEW | haiku+sonnet | 4 |
| implement, create, build | BUILD | sonnet | 2-4 |
| design, architect, plan | DESIGN | sonnet | 2 |
| trading, risk, execution | TRADING | sonnet | 3 |
| decide, should, choose | CONSENSUS | sonnet | 3 |

---

## CLI Access

```bash
# Help
python3 .claude/hooks/agent_orchestrator.py help

# Auto-select
python3 .claude/hooks/agent_orchestrator.py auto "task"

# List resources
python3 .claude/hooks/agent_orchestrator.py list

# Generate Task calls
python3 .claude/hooks/agent_orchestrator.py generate workflow_name

# Statistics
python3 .claude/hooks/agent_orchestrator.py status
```

---

## Workflow Patterns

### Parallel (Default)

All agents run simultaneously. Results aggregated at end.

```
Agent 1 ─┐
Agent 2 ─┼─→ Aggregate Results
Agent 3 ─┘
```

### Sequential (Pipeline)

Output chains to next agent.

```
Research → Plan → Implement → Review
```

### Consensus (Voting)

Multiple agents vote independently. 66% threshold.

```
Agent 1: APPROVE
Agent 2: APPROVE   → 66% = APPROVED
Agent 3: REJECT
```

---

## Files

| File | Purpose |
|------|---------|
| `.claude/hooks/agent_orchestrator.py` | Main engine |
| `.claude/agent_config.json` | Configuration |
| `.claude/commands/agents.md` | Master command |
| `.claude/commands/agent-*.md` | Individual commands |

---

## See Also

- [MULTI-AGENT-ORCHESTRATION-RESEARCH.md](../research/MULTI-AGENT-ORCHESTRATION-RESEARCH.md) - Full research
- [CLAUDE.md](../../CLAUDE.md#multi-agent-orchestration-suite) - Main documentation
