# Agent Auto - Intelligent Task Routing

Automatically analyze a task and spawn the best agents to handle it.

## Arguments
- `$ARGUMENTS`: Any task description in natural language

## Examples
```
/agent-auto find all places where we handle authentication
/agent-auto review the execution module for bugs
/agent-auto what options strategies are implemented
/agent-auto implement a new scanner for volume spikes
```

## Instructions

This is the **intelligent auto-routing** command. Analyze `$ARGUMENTS` and:

### Step 1: Classify Task Type

| Pattern | Type | Model | Agents |
|---------|------|-------|--------|
| find, search, where, locate | SEARCH | haiku | 3 search agents |
| review, check, audit, analyze | REVIEW | haiku+sonnet | 4 review agents |
| implement, create, build, add | BUILD | sonnet | research + implement |
| fix, debug, solve, troubleshoot | DEBUG | sonnet | finder + fixer |
| design, architect, plan | DESIGN | sonnet/opus | planner + architect |
| trading, risk, order, execution | TRADING | sonnet | 3 trading agents |
| security, vulnerability, secret | SECURITY | haiku+sonnet | security focused |
| test, coverage, verify | TESTING | haiku | test focused |

### Step 2: Select Agent Configuration

**For SEARCH tasks** (3 haiku agents parallel):
```python
Task(subagent_type="Explore", model="haiku", description="Search source code",
     prompt="Search all Python files for: {query}. Return file:line and code snippets.")
Task(subagent_type="Explore", model="haiku", description="Search documentation",
     prompt="Search docs/, README files, CLAUDE.md for: {query}. Return excerpts.")
Task(subagent_type="Explore", model="haiku", description="Search tests",
     prompt="Search tests/ for: {query}. Return test cases and related fixtures.")
```

**For REVIEW tasks** (4 agents parallel):
```python
Task(subagent_type="Explore", model="haiku", description="Security check",
     prompt="Security scan of {target}: injection, secrets, OWASP issues...")
Task(subagent_type="Explore", model="haiku", description="Code quality",
     prompt="Quality check of {target}: type hints, error handling, patterns...")
Task(subagent_type="Explore", model="haiku", description="Test coverage",
     prompt="Test coverage analysis for {target}: missing tests, edge cases...")
Task(subagent_type="Plan", model="sonnet", description="Architecture review",
     prompt="Architecture review of {target}: SOLID, design, recommendations...")
```

**For BUILD tasks** (2 agents sequential):
```python
# First: Research
Task(subagent_type="Explore", model="haiku", description="Research patterns",
     prompt="Find existing patterns for: {task}. Return code examples and best practices.")
# Then: Implement (based on research)
Task(subagent_type="general-purpose", model="sonnet", description="Implement",
     prompt="Implement {task} following project patterns. Include tests and docs.")
```

**For TRADING tasks** (3 agents parallel):
```python
Task(subagent_type="Explore", model="sonnet", description="Risk review",
     prompt="Risk management check: circuit breakers, position limits, stop losses...")
Task(subagent_type="Explore", model="haiku", description="Execution review",
     prompt="Order execution check: validation, fills, cancel/replace logic...")
Task(subagent_type="Explore", model="haiku", description="Backtest check",
     prompt="Backtesting integrity: look-ahead bias, costs, realistic assumptions...")
```

**For DESIGN tasks** (2 agents):
```python
Task(subagent_type="Explore", model="haiku", description="Research existing",
     prompt="Research existing architecture for: {topic}. Map dependencies and patterns.")
Task(subagent_type="Plan", model="sonnet", description="Design proposal",
     prompt="Design proposal for: {topic}. Include trade-offs, alternatives, recommendations.")
```

### Step 3: Execute and Aggregate

1. Spawn all selected agents IN PARALLEL (single response)
2. Wait for all results
3. Aggregate into structured summary:

```markdown
## Auto-Agent Results: {task}

### Task Classification
- Type: {SEARCH/REVIEW/BUILD/etc}
- Complexity: {TRIVIAL/SIMPLE/MODERATE/COMPLEX/CRITICAL}
- Agents: {count} ({models used})

### Findings
{Aggregated findings from all agents}

### Recommendations
{Prioritized action items}
```
