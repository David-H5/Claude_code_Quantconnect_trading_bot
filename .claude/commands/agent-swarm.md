# Agent Swarm - Massive Parallel Exploration

Deploy a swarm of fast agents to thoroughly explore a topic or codebase area.

## Arguments
- `$ARGUMENTS`: Topic or area to explore comprehensively

## Examples
```
/agent-swarm error handling
/agent-swarm options trading implementation
/agent-swarm authentication and authorization
/agent-swarm database access patterns
```

## Instructions

This command spawns 5-8 haiku agents simultaneously for comprehensive exploration.

### Swarm Configuration

For topic `$ARGUMENTS`, spawn these agents IN PARALLEL (all in ONE response):

```python
# Agent 1: Core implementation search
Task(
    subagent_type="Explore",
    model="haiku",
    description="Find core implementation",
    prompt="""Find the core implementation of: $ARGUMENTS

Search in:
- algorithms/
- execution/
- models/
- scanners/

Return: Main files, key classes/functions, entry points."""
)

# Agent 2: Supporting code search
Task(
    subagent_type="Explore",
    model="haiku",
    description="Find supporting code",
    prompt="""Find supporting/utility code for: $ARGUMENTS

Search in:
- utils/
- indicators/
- llm/

Return: Helper functions, utilities, integrations."""
)

# Agent 3: Configuration search
Task(
    subagent_type="Explore",
    model="haiku",
    description="Find configuration",
    prompt="""Find configuration related to: $ARGUMENTS

Search in:
- config/
- settings.json
- .env files
- Constants/defaults

Return: Configurable parameters and their defaults."""
)

# Agent 4: Test coverage search
Task(
    subagent_type="Explore",
    model="haiku",
    description="Find tests",
    prompt="""Find all tests for: $ARGUMENTS

Search in:
- tests/unit/
- tests/integration/
- tests/regression/

Return: Test files, test cases, fixtures, coverage."""
)

# Agent 5: Documentation search
Task(
    subagent_type="Explore",
    model="haiku",
    description="Find documentation",
    prompt="""Find documentation for: $ARGUMENTS

Search in:
- docs/
- README files
- CLAUDE.md
- Docstrings

Return: Documentation, examples, usage guides."""
)

# Agent 6: Dependencies search
Task(
    subagent_type="Explore",
    model="haiku",
    description="Find dependencies",
    prompt="""Find what depends on and is depended by: $ARGUMENTS

Look for:
- Import statements
- Function calls
- Class inheritance
- Configuration references

Return: Dependency map (what uses it, what it uses)."""
)

# Agent 7: Error handling search
Task(
    subagent_type="Explore",
    model="haiku",
    description="Find error handling",
    prompt="""Find error handling for: $ARGUMENTS

Look for:
- Try/except blocks
- Error classes
- Logging statements
- Validation code

Return: Error handling patterns and gaps."""
)

# Agent 8: Usage examples search
Task(
    subagent_type="Explore",
    model="haiku",
    description="Find usage examples",
    prompt="""Find usage examples of: $ARGUMENTS

Look for:
- How it's called in production code
- Test usage patterns
- Documentation examples
- Integration points

Return: Common usage patterns."""
)
```

### Result Aggregation

After all agents complete, create a comprehensive map:

```markdown
## Swarm Exploration: $ARGUMENTS

### Coverage Summary
- Agents deployed: 8
- Areas searched: core, utils, config, tests, docs, deps, errors, examples

### Core Implementation
{Agent 1 findings}

### Supporting Code
{Agent 2 findings}

### Configuration
{Agent 3 findings}

### Test Coverage
{Agent 4 findings}

### Documentation
{Agent 5 findings}

### Dependencies
{Agent 6 findings}

### Error Handling
{Agent 7 findings}

### Usage Patterns
{Agent 8 findings}

### Quick Reference Map
| Aspect | Location | Key Files |
|--------|----------|-----------|
| Core | {path} | {files} |
| Config | {path} | {files} |
| Tests | {path} | {files} |
| Docs | {path} | {files} |

### Discovered Gaps
- {Missing tests}
- {Missing docs}
- {Error handling gaps}
```
