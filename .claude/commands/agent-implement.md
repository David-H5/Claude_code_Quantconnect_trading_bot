# Agent Implement - Multi-Agent Implementation Pipeline

Run a full implementation pipeline: Research → Plan → Implement → Review

## Arguments
- `$ARGUMENTS`: Feature or change to implement

## Examples
```
/agent-implement add IV rank filter to options scanner
/agent-implement create volume spike detector
/agent-implement add websocket reconnection logic
/agent-implement implement order deduplication
```

## Instructions

This runs a SEQUENTIAL 4-stage pipeline where each agent's output feeds the next.

### Stage 1: Research (Haiku)

```python
Task(
    subagent_type="Explore",
    model="haiku",
    description="Research existing patterns",
    prompt="""Research existing codebase patterns for implementing: $ARGUMENTS

Find:
1. Similar implementations in the codebase
2. Related utility functions
3. Configuration patterns used
4. Test patterns for similar features
5. Documentation standards

Return a research summary that will guide implementation."""
)
```

**Wait for Research results before proceeding**

### Stage 2: Plan (Sonnet)

```python
Task(
    subagent_type="Plan",
    model="sonnet",
    description="Create implementation plan",
    prompt="""Create an implementation plan for: $ARGUMENTS

Based on research findings:
{research_output}

Create a detailed plan including:
1. Files to create/modify
2. Classes/functions to implement
3. Integration points
4. Configuration changes needed
5. Tests to write
6. Documentation updates

Format as actionable checklist."""
)
```

**Wait for Plan results before proceeding**

### Stage 3: Implement (Sonnet)

```python
Task(
    subagent_type="general-purpose",
    model="sonnet",
    description="Implement the feature",
    prompt="""Implement: $ARGUMENTS

Following this plan:
{plan_output}

Requirements:
- Follow project coding standards
- Type hints on all functions
- Google-style docstrings
- Defensive error handling
- Create corresponding tests

Write the actual code changes needed."""
)
```

**Wait for Implementation before proceeding**

### Stage 4: Review (Sonnet)

```python
Task(
    subagent_type="general-purpose",
    model="sonnet",
    description="Review implementation",
    prompt="""Review this implementation of: $ARGUMENTS

Code:
{implementation_output}

Review for:
1. Correctness - Does it do what's intended?
2. Safety - Any security or trading risks?
3. Quality - Code style, error handling, types?
4. Tests - Are tests comprehensive?
5. Integration - Will it work with existing code?

Provide specific feedback and any required fixes."""
)
```

### Pipeline Output

```markdown
## Implementation Pipeline: $ARGUMENTS

### Stage 1: Research
{Research findings summary}

### Stage 2: Plan
{Implementation plan}

### Stage 3: Implementation
{Code changes made}

### Stage 4: Review
{Review feedback}

### Status: {COMPLETE/NEEDS_FIXES}

### Files Changed
- {file1}
- {file2}

### Tests Added
- {test1}
- {test2}

### Next Steps
- [ ] {Any remaining items}
```
