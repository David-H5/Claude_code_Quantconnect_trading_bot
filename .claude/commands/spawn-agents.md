# Spawn Multiple Agents

Spawn multiple specialized agents in parallel for a custom task.

## Arguments
- `$ARGUMENTS`: Agent specification in format: `<model>:<type>:<task>` separated by `|`

Example: `haiku:Explore:find auth code | haiku:Explore:find tests for auth | sonnet:Plan:design auth improvements`

## Models Available
- `haiku` - Fastest, cheapest (use for search, simple checks)
- `sonnet` - Balanced (use for implementation, review)
- `opus` - Deep reasoning (use for architecture, complex analysis)

## Agent Types Available
- `Explore` - Fast codebase exploration (recommended for searches)
- `Plan` - Architecture and planning
- `general-purpose` - Complex multi-step tasks

## Instructions

Parse the `$ARGUMENTS` and spawn one Task for each specification.

For example, if `$ARGUMENTS` is:
```
haiku:Explore:find circuit breaker usage | haiku:Explore:find risk management code | sonnet:Plan:improve risk coverage
```

Then spawn 3 agents IN PARALLEL:

```
Task(
  subagent_type="Explore",
  model="haiku",
  description="Find circuit breaker usage",
  prompt="Find circuit breaker usage in the codebase. Return file paths and code snippets."
)
```

```
Task(
  subagent_type="Explore",
  model="haiku",
  description="Find risk management code",
  prompt="Find risk management code in the codebase. Return file paths and code snippets."
)
```

```
Task(
  subagent_type="Plan",
  model="sonnet",
  description="Improve risk coverage",
  prompt="Based on the codebase, design improvements for risk coverage. Provide specific recommendations."
)
```

## Quick Patterns

**3x Haiku Search:**
```
haiku:Explore:search 1 | haiku:Explore:search 2 | haiku:Explore:search 3
```

**Review Team:**
```
haiku:Explore:security scan | haiku:Explore:type check | sonnet:Plan:architecture review
```

**Research Squad:**
```
haiku:Explore:find code | haiku:Explore:find docs | haiku:Explore:find tests
```

## After Completion

Combine results and present a unified summary.
