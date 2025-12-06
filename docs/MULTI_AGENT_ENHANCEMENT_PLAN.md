# Multi-Agent Enhancement Plan

Based on research into Claude Code multi-agent frameworks, MCP integration, and autonomous trading architectures.

## Research Summary

### Key Findings

1. **Claude Agent SDK** (formerly Claude Code SDK) provides native subagent support with parallelization and isolated context windows
2. **MCP (Model Context Protocol)** enables integration with thousands of external tools and data sources
3. **Multi-agent trading systems** using specialized agents (Data, Prediction, Trading) outperform single-agent approaches
4. **Anthropic's research** shows multi-agent systems outperform single-agent by 90.2% on complex tasks

### Relevant Frameworks

| Framework | Key Feature | Relevance |
|-----------|-------------|-----------|
| [Claude Flow](https://github.com/ruvnet/claude-flow) | Swarm orchestration, MCP protocol | High - production-ready orchestration |
| [Claude Code Agentrooms](https://claudecode.run/) | @mentions routing, local/remote agents | Medium - collaborative development |
| [wshobson/agents](https://github.com/wshobson/agents) | 85 agents, 15 orchestrators, 47 skills | High - proven patterns |

---

## Proposed Architecture

### Phase 1: Subagent Infrastructure (Foundation)

Create specialized subagents in `.claude/agents/`:

```
.claude/agents/
├── market-analyst.md       # Technical analysis, chart patterns
├── sentiment-scanner.md    # News, social media sentiment
├── risk-guardian.md        # Position sizing, circuit breaker
├── execution-manager.md    # Order routing, fill optimization
└── research-compiler.md    # Strategy backtesting, reports
```

#### Example Subagent Definition

```markdown
# Market Analyst Agent

## Role
Analyze technical indicators and identify trading opportunities.

## Tools
- Read market data files
- Execute indicator calculations
- Query historical patterns

## Constraints
- Read-only access to trading systems
- Must include confidence scores
- Report to orchestrator only
```

### Phase 2: MCP Server Integration

Connect to external data sources via MCP:

```json
// .claude/settings.json
{
  "mcpServers": {
    "market-data": {
      "type": "http",
      "url": "http://localhost:8080/mcp",
      "tools": ["get_quotes", "get_options_chain", "get_news"]
    },
    "database": {
      "type": "stdio",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-postgres"],
      "env": {"DATABASE_URL": "${TRADING_DB_URL}"}
    }
  }
}
```

#### Recommended MCP Servers

| Server | Purpose | Priority |
|--------|---------|----------|
| Postgres | Trade history, backtests | High |
| GitHub | Code versioning, issues | Medium |
| Slack/Discord | Alerts, notifications | Medium |
| Custom Market Data | Real-time quotes | High |

### Phase 3: Orchestration Pipeline

Implement the proven orchestrator pattern:

```
                    ┌─────────────────┐
                    │   Orchestrator  │
                    │  (Claude Opus)  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Market Analyst  │ │ Sentiment Agent │ │  Risk Guardian  │
│  (Sonnet 4)     │ │   (Sonnet 4)    │ │   (Sonnet 4)    │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Execution Agent │
                    │  (Sonnet 4)     │
                    └─────────────────┘
```

#### Orchestration Rules

1. **Orchestrator** (Opus): Global planning, delegation, state management
2. **Subagents** (Sonnet): Single specialized job each
3. **Communication**: Return only relevant excerpts, not full context
4. **Parallelization**: Analyst + Sentiment run in parallel; Risk gates Execution

### Phase 4: Agent Skills System

Define reusable skills in `.claude/skills/`:

```
.claude/skills/
├── SKILL_backtest.md           # Run strategy backtests
├── SKILL_options_analysis.md   # Greeks, IV analysis
├── SKILL_risk_check.md         # Pre-trade validation
└── SKILL_report_generator.md   # Create trading reports
```

#### Example Skill Definition

```markdown
# Backtest Skill

## Trigger
When user asks to "backtest", "evaluate strategy", or "historical test"

## Steps
1. Validate strategy parameters
2. Load historical data
3. Run QuantConnect backtest
4. Parse results
5. Generate performance report

## Required Tools
- Read, Write, Bash

## Output Format
Return structured JSON with Sharpe, Sortino, MaxDrawdown
```

---

## Implementation Priorities

### High Priority (Week 1-2)

| Task | Description | Effort |
|------|-------------|--------|
| Create agent definitions | Define 5 core subagents | 2 days |
| Setup MCP market data | Real-time quote integration | 3 days |
| Implement orchestrator | Basic task routing | 2 days |
| Add circuit breaker hooks | Agent safety integration | 1 day |

### Medium Priority (Week 3-4)

| Task | Description | Effort |
|------|-------------|--------|
| Database MCP server | Historical trade analysis | 2 days |
| Parallel execution | Multiple subagents at once | 2 days |
| Skill library | Reusable trading operations | 3 days |
| Monitoring dashboard | Agent activity visualization | 2 days |

### Lower Priority (Week 5+)

| Task | Description | Effort |
|------|-------------|--------|
| Slack/Discord alerts | Trading notifications | 1 day |
| Auto-strategy evolution | Learning from results | 5 days |
| Multi-model ensemble | Opus + Sonnet + Haiku mix | 3 days |

---

## Safety Considerations

### Circuit Breaker Integration

All subagents must respect trading halts:

```python
# In each agent's initialization
from models.circuit_breaker import get_circuit_breaker

def before_action(action_type: str) -> bool:
    breaker = get_circuit_breaker()
    if not breaker.can_trade():
        return False  # Abort action
    return True
```

### Human-in-the-Loop Gates

| Action | Requires Human Approval |
|--------|------------------------|
| New strategy deployment | Yes |
| Position > $10,000 | Yes |
| After circuit breaker trip | Yes |
| Algorithm changes | Yes |
| Read-only analysis | No |
| Paper trading | No |

### Audit Trail

Log all agent decisions to `logs/agent_decisions.jsonl`:

```json
{
  "timestamp": "2025-12-06T14:30:00Z",
  "agent": "market-analyst",
  "action": "signal_generated",
  "symbol": "SPY",
  "direction": "bullish",
  "confidence": 0.78,
  "reasoning": "RSI oversold + support bounce"
}
```

---

## Git Workflow Integration

### Agent Branch Strategy

```
main
├── develop
├── agent/overnight       # Overnight analysis agent
├── agent/market-analyst  # Dedicated analysis branch
├── agent/backtest        # Strategy testing agent
└── human/feature/*       # Human development
```

### Commit Conventions for Agents

```
<agent>(<scope>): <description>

Examples:
analyst(SPY): identified support at 580
risk(circuit): triggered daily loss halt
orchestrator(handoff): completed overnight scan
```

---

## Metrics & Success Criteria

### Agent Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Signal accuracy | >60% | Backtest validation |
| Response latency | <5s | Average agent response time |
| Context efficiency | <50% | Tokens used vs available |
| Uptime | >99% | Agent availability |

### Trading Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Sharpe Ratio | >1.5 | Baseline TBD |
| Max Drawdown | <15% | Baseline TBD |
| Win Rate | >55% | Baseline TBD |
| Risk-Adjusted Return | >20% annual | Baseline TBD |

---

## Next Steps

1. **Review this plan** - Approve priorities and timeline
2. **Create agent definitions** - Start with market-analyst.md
3. **Setup MCP server** - Begin with market data integration
4. **Implement orchestrator** - Basic routing and state management
5. **Add monitoring** - Agent activity dashboard

---

## References

- [Anthropic: Building agents with Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
- [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Claude Code MCP Integration](https://docs.anthropic.com/en/docs/claude-code/mcp)
- [GitHub: Agentic Workflows](https://github.blog/ai-and-ml/github-copilot/how-to-build-reliable-ai-workflows-with-agentic-primitives-and-context-engineering/)
- [AI-Powered Multi-Agent Trading Workflow](https://medium.com/@bijit211987/ai-powered-multi-agent-trading-workflow-90722a2ada3b)
- [Claude Flow Framework](https://github.com/ruvnet/claude-flow)
