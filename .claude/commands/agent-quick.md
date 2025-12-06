# Agent Quick - Fast Single-Agent Dispatch

Quickly dispatch a single agent for a focused task. Fastest way to get agent help.

## Arguments
- `$ARGUMENTS`: Format: `<model>:<task>` or just `<task>` (defaults to haiku)

## Examples
```
/agent-quick find circuit breaker usage
/agent-quick haiku:count all test files
/agent-quick sonnet:review the risk manager
/agent-quick opus:design caching strategy
```

## Model Shortcuts
- `h` or `haiku` - Fast, cheap (default)
- `s` or `sonnet` - Balanced
- `o` or `opus` - Deep analysis

## Instructions

Parse `$ARGUMENTS`:
- If format is `model:task`, extract model and task
- If no model specified, default to `haiku`
- If task contains "review", "design", "architect" → use `sonnet`
- If task contains "critical", "production", "security risk" → use `opus`

### Quick Dispatch

```python
Task(
    subagent_type="Explore",  # Use "Plan" for design tasks
    model="{model}",
    description="Quick task: {task_summary}",
    prompt="""{task}

Be concise and direct. Return:
1. Key findings or results
2. File:line references where relevant
3. Brief recommendations if applicable"""
)
```

### Smart Model Selection

If no model specified, auto-select based on task:

| Task Contains | Model |
|---------------|-------|
| find, search, list, count | haiku |
| check, verify, validate | haiku |
| review, analyze | sonnet |
| design, architect, plan | sonnet |
| critical, production, deploy | opus |
| security vulnerability | opus |

### Response Format

```markdown
## Quick Agent: {task}
**Model**: {model} | **Type**: {agent_type}

### Result
{Agent output}

### References
{File:line references if any}
```
