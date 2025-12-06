# Agent Status - Orchestration Statistics

Show the status and statistics of the agent orchestration system.

## Arguments
- `$ARGUMENTS`: Optional filter (e.g., "recent", "stats", "config")

## Examples
```
/agent-status
/agent-status stats
/agent-status config
```

## Instructions

Run the orchestrator status command and display results:

```bash
python3 .claude/hooks/agent_orchestrator.py status
```

Also display available resources:

### Status Output Format

```markdown
## Agent Orchestrator Status

### System Info
- **Version**: 1.0.0
- **Config**: .claude/config/agents.json
- **State**: .claude/state/agent_state.json

### Statistics
- Total Runs: {count}
- Total Agents Spawned: {count}
- Average Success Rate: {percent}%
- Average Execution Time: {time}ms
- Last Run: {timestamp}

### Available Workflows
| Workflow | Agents | Pattern | Tags |
|----------|--------|---------|------|
| code_review | 4 | parallel | review, quality |
| multi_search | 3 | parallel | search, fast |
| trading_review | 3 | parallel | trading, safety |
| critical_decision | 3 | consensus | decision |
| implementation_pipeline | 4 | sequential | build |

### Available Agents
| Agent | Model | Type | Purpose |
|-------|-------|------|---------|
| CodeFinder | haiku | Explore | Search code |
| DocFinder | haiku | Explore | Search docs |
| TestFinder | haiku | Explore | Search tests |
| SecurityScanner | haiku | Explore | Security check |
| TypeChecker | haiku | Explore | Type hints |
| Architect | sonnet | Plan | Architecture |
| RiskReviewer | sonnet | Explore | Trading risk |
| CriticalReviewer | opus | General | Production review |

### Quick Commands
| Command | Description |
|---------|-------------|
| `/agents auto <task>` | Auto-select agents |
| `/agent-quick <task>` | Single fast agent |
| `/agent-swarm <topic>` | 8-agent exploration |
| `/agent-consensus <decision>` | 3-agent voting |
| `/agent-implement <feature>` | 4-stage pipeline |
| `/agent-compare <options>` | Multi-option analysis |
```
