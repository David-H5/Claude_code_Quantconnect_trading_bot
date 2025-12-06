# Multi-Agent Parallel Search

Search the codebase, documentation, and tests simultaneously using 3 parallel agents.

## Arguments
- `$ARGUMENTS`: Search query (what you're looking for)

## Instructions

Execute these 3 agents IN PARALLEL (all in a single response) using the Task tool:

**Agent 1: Codebase Search (haiku - fast)**
```
Task(
  subagent_type="Explore",
  model="haiku",
  description="Search codebase",
  prompt="Search the codebase for: $ARGUMENTS

Look in:
- algorithms/ - trading algorithms
- execution/ - order execution code
- scanners/ - market scanners
- models/ - risk and data models
- llm/ - LLM integration
- indicators/ - technical indicators
- utils/ - utility functions

Return:
- File paths where found
- Key code snippets (max 20 lines each)
- Function/class names
- Usage patterns"
)
```

**Agent 2: Documentation Search (haiku - fast)**
```
Task(
  subagent_type="Explore",
  model="haiku",
  description="Search documentation",
  prompt="Search project documentation for: $ARGUMENTS

Check:
- docs/ directory (all subdirectories)
- README.md files throughout project
- CLAUDE.md for instructions
- Code docstrings
- Inline comments

Return:
- Relevant documentation excerpts
- File locations
- Related sections"
)
```

**Agent 3: Test Search (haiku - fast)**
```
Task(
  subagent_type="Explore",
  model="haiku",
  description="Search test files",
  prompt="Search test files for: $ARGUMENTS

Look in:
- tests/unit/ - unit tests
- tests/integration/ - integration tests
- tests/regression/ - regression tests
- conftest.py files - fixtures

Return:
- Related test cases
- Test patterns used
- Fixtures and mocks
- Test file locations"
)
```

## After All Agents Complete

Synthesize findings:

```markdown
## Search Results: $ARGUMENTS

### Code Locations
[Agent 1 results - key file:line references]

### Documentation
[Agent 2 results - relevant docs]

### Test Coverage
[Agent 3 results - related tests]

### Quick Reference
- Main implementation: [file path]
- Documentation: [doc path]
- Tests: [test path]
```
