# Agents - Multi-Agent Orchestration Suite

The master command for the multi-agent orchestration system. Spawn, coordinate, and manage multiple Claude agents with intelligent model selection.

## Arguments
- `$ARGUMENTS`: Subcommand and options

## Subcommands

| Command | Description |
|---------|-------------|
| `auto <task>` | Auto-select best agents for any task |
| `run <workflow>` | Run a pre-built workflow |
| `search <query>` | Quick 3-agent parallel search |
| `review <files>` | Quick 4-agent code review |
| `trading <files>` | Quick 3-agent trading safety review |
| `list` | List all workflows and agents |
| `status` | Show orchestration statistics |
| `help` | Show detailed help |

## Quick Examples

```
/agents auto find all error handling code
/agents search "circuit breaker"
/agents review algorithms/
/agents trading execution/
/agents run code_review target=scanners/
```

## Instructions

Parse `$ARGUMENTS` to determine the subcommand:

### `auto <task>` - Intelligent Auto-Selection

When user provides a task description, automatically:

1. **Analyze Task Complexity**
   - TRIVIAL: Simple search/grep → Use haiku
   - SIMPLE: Single-file analysis → Use haiku
   - MODERATE: Multi-file changes → Use sonnet
   - COMPLEX: Architecture decisions → Use sonnet
   - CRITICAL: Trading/security/production → Use opus

2. **Select Best Agents** based on task keywords:
   - "find/search/where" → CodeFinder, DocFinder, TestFinder
   - "review/check/audit" → SecurityScanner, TypeChecker, Architect
   - "trading/risk/order" → RiskReviewer, ExecutionReviewer, BacktestReviewer
   - "implement/create/build" → Implementer with research support
   - "decide/choose/evaluate" → Consensus workflow

3. **Spawn Agents in Parallel** (all in ONE response):

```python
# Example for "find all error handling"
Task(subagent_type="Explore", model="haiku",
     description="Find error handling in code",
     prompt="Search source files for error handling patterns...")

Task(subagent_type="Explore", model="haiku",
     description="Find error handling in tests",
     prompt="Search test files for error handling tests...")

Task(subagent_type="Explore", model="haiku",
     description="Find error handling docs",
     prompt="Search documentation for error handling guidelines...")
```

### `run <workflow>` - Execute Pre-Built Workflow

Available workflows:
- `code_review` - 4-agent: security + types + tests + architecture
- `multi_search` - 3-agent: code + docs + tests search
- `trading_review` - 3-agent: risk + execution + backtest
- `critical_decision` - 3-agent consensus voting
- `implementation_pipeline` - Sequential: research → plan → implement → review

For `code_review`:
```python
Task(subagent_type="Explore", model="haiku", description="Security scan", prompt="...")
Task(subagent_type="Explore", model="haiku", description="Type check", prompt="...")
Task(subagent_type="Explore", model="haiku", description="Test analysis", prompt="...")
Task(subagent_type="Plan", model="sonnet", description="Architecture review", prompt="...")
```

### `search <query>` - Quick Multi-Search

Spawn 3 haiku agents in parallel:

```python
Task(subagent_type="Explore", model="haiku",
     description="Search code for query",
     prompt="Search source code (*.py) for: {query}. Return file:line and snippets.")

Task(subagent_type="Explore", model="haiku",
     description="Search docs for query",
     prompt="Search docs/, README, CLAUDE.md for: {query}. Return excerpts.")

Task(subagent_type="Explore", model="haiku",
     description="Search tests for query",
     prompt="Search tests/ for: {query}. Return test cases and fixtures.")
```

### `review <files>` - Quick Code Review

Spawn 4 agents in parallel:

```python
Task(subagent_type="Explore", model="haiku",
     description="Security scan",
     prompt="Scan {files} for: SQL injection, hardcoded secrets, OWASP Top 10...")

Task(subagent_type="Explore", model="haiku",
     description="Type check",
     prompt="Check type hints in {files}: missing hints, wrong types, Optional handling...")

Task(subagent_type="Explore", model="haiku",
     description="Test coverage",
     prompt="Analyze test coverage for {files}: untested functions, missing edge cases...")

Task(subagent_type="Plan", model="sonnet",
     description="Architecture review",
     prompt="Review architecture of {files}: SOLID, patterns, coupling, recommendations...")
```

### `trading <files>` - Trading Safety Review

Spawn 3 agents in parallel:

```python
Task(subagent_type="Explore", model="sonnet",
     description="Risk management review",
     prompt="Review {files} for: circuit breaker, position limits (25%), daily loss (3%), drawdown (10%), order validation...")

Task(subagent_type="Explore", model="haiku",
     description="Execution review",
     prompt="Review {files} for: order validation, fill handling, cancel/replace, duplicates...")

Task(subagent_type="Explore", model="haiku",
     description="Backtest integrity",
     prompt="Check {files} for: look-ahead bias, survivorship bias, unrealistic fills, missing costs...")
```

### `list` - Show Available Resources

Display:
```
=== WORKFLOWS ===
- code_review (4 agents): Security, types, tests, architecture
- multi_search (3 agents): Code, docs, tests search
- trading_review (3 agents): Risk, execution, backtest
- critical_decision (3 agents): Consensus voting
- implementation_pipeline (4 agents): Research → Plan → Implement → Review

=== AGENT TEMPLATES ===
Search: code_finder, doc_finder, test_finder
Review: security_scanner, type_checker, test_analyzer, architect
Trading: risk_reviewer, execution_reviewer, backtest_reviewer
Implementation: implementer, refactorer
Deep: deep_architect, critical_reviewer
```

### `status` - Show Statistics

Run:
```bash
python3 .claude/hooks/agent_orchestrator.py status
```

### `help` - Detailed Help

Run:
```bash
python3 .claude/hooks/agent_orchestrator.py help
```

## Model Selection Guide

| Task Type | Model | Agents |
|-----------|-------|--------|
| File search, grep | haiku | CodeFinder, DocFinder, TestFinder |
| Simple analysis | haiku | SecurityScanner, TypeChecker |
| Code review | sonnet | Architect, RiskReviewer |
| Implementation | sonnet | Implementer, Refactorer |
| Architecture | opus | DeepArchitect |
| Critical decisions | opus | CriticalReviewer |

## After Agents Complete

Always aggregate results into a structured report:

```markdown
## Agent Results Summary

### Task: {original task}
### Agents Spawned: {count}
### Pattern: {parallel/sequential/consensus}

### Findings by Agent
[Each agent's key findings]

### Aggregated Recommendations
1. [Priority 1 item]
2. [Priority 2 item]
...

### Action Items
- [ ] [Specific action]
- [ ] [Specific action]
```
