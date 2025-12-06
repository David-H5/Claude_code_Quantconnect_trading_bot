# Parallel Code Review

Run a comprehensive 4-agent parallel code review on the specified files.

## Arguments
- `$ARGUMENTS`: File paths or glob patterns to review (e.g., "algorithms/*.py" or "execution/")

## Instructions

Execute these 4 agents IN PARALLEL (all in a single response) using the Task tool:

**Agent 1: Security Scan (haiku - fast)**
```
Task(
  subagent_type="Explore",
  model="haiku",
  description="Security vulnerability scan",
  prompt="Scan for security vulnerabilities in: $ARGUMENTS

Check for:
- SQL injection, command injection
- Hardcoded credentials or secrets
- Path traversal vulnerabilities
- OWASP Top 10 issues
- Unsafe deserialization

Report findings with file:line references and severity (CRITICAL/HIGH/MEDIUM/LOW)."
)
```

**Agent 2: Type Safety Check (haiku - fast)**
```
Task(
  subagent_type="Explore",
  model="haiku",
  description="Type hint validation",
  prompt="Check type hint completeness in: $ARGUMENTS

Verify:
- All functions have type hints
- Return types are specified
- Optional vs None handled correctly
- Generic types used properly

Report missing or incorrect type hints with file:line references."
)
```

**Agent 3: Test Coverage Analysis (haiku - fast)**
```
Task(
  subagent_type="Explore",
  model="haiku",
  description="Test coverage analysis",
  prompt="Analyze test coverage for: $ARGUMENTS

Find:
- Functions without corresponding tests
- Missing edge case tests
- Integration test gaps
- Suggest specific tests to add

Reference the tests/ directory structure."
)
```

**Agent 4: Architecture Review (sonnet - thorough)**
```
Task(
  subagent_type="Plan",
  model="sonnet",
  description="Architecture and design review",
  prompt="Review architecture and design patterns in: $ARGUMENTS

Evaluate:
- SOLID principles adherence
- Design pattern appropriateness
- Coupling and cohesion
- Dependency management
- Code organization

Provide specific, actionable recommendations."
)
```

## After All Agents Complete

Aggregate findings into a structured report:

```markdown
## Parallel Code Review Results

### Security Findings
[Agent 1 results]

### Type Safety Issues
[Agent 2 results]

### Test Coverage Gaps
[Agent 3 results]

### Architecture Recommendations
[Agent 4 results]

### Summary
- Critical issues: X
- High priority: Y
- Medium priority: Z
- Total findings: N
```
