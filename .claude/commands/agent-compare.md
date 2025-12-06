# Agent Compare - Multi-Agent Comparison Analysis

Have multiple agents analyze and compare different options or approaches.

## Arguments
- `$ARGUMENTS`: Options to compare (separated by "vs" or comma)

## Examples
```
/agent-compare Redis vs in-memory caching
/agent-compare sync vs async execution
/agent-compare WebSocket vs polling for real-time data
/agent-compare SQLite, PostgreSQL, MongoDB for trade logs
```

## Instructions

Parse `$ARGUMENTS` to extract options, then spawn one agent per option plus a synthesizer.

### For 2-3 Options

Spawn one analyst per option:

```python
# Option A Analyst
Task(
    subagent_type="general-purpose",
    model="sonnet",
    description="Analyze option: {option_a}",
    prompt="""Analyze {option_a} for this use case: {context}

Evaluate:
1. **Pros**: Key advantages
2. **Cons**: Key disadvantages
3. **Complexity**: Implementation effort (Low/Medium/High)
4. **Performance**: Expected performance characteristics
5. **Maintenance**: Long-term maintenance burden
6. **Fit**: How well it fits this project's patterns

Provide specific, concrete analysis based on the codebase."""
)

# Option B Analyst
Task(
    subagent_type="general-purpose",
    model="sonnet",
    description="Analyze option: {option_b}",
    prompt="""Analyze {option_b} for this use case: {context}

Evaluate:
1. **Pros**: Key advantages
2. **Cons**: Key disadvantages
3. **Complexity**: Implementation effort (Low/Medium/High)
4. **Performance**: Expected performance characteristics
5. **Maintenance**: Long-term maintenance burden
6. **Fit**: How well it fits this project's patterns

Provide specific, concrete analysis based on the codebase."""
)

# Synthesizer
Task(
    subagent_type="Plan",
    model="sonnet",
    description="Synthesize comparison",
    prompt="""Compare these options for the project:
- {option_a}
- {option_b}

Create a comparison matrix and recommendation.
Consider: complexity, performance, maintenance, project fit.

End with a clear recommendation and reasoning."""
)
```

### Result Format

```markdown
## Comparison: {options}

### Option Analysis

#### {Option A}
| Aspect | Rating | Notes |
|--------|--------|-------|
| Pros | | {key advantages} |
| Cons | | {key disadvantages} |
| Complexity | {L/M/H} | {implementation effort} |
| Performance | {L/M/H} | {expected perf} |
| Maintenance | {L/M/H} | {long-term burden} |
| Project Fit | {L/M/H} | {how well it fits} |

#### {Option B}
| Aspect | Rating | Notes |
|--------|--------|-------|
| Pros | | {key advantages} |
| Cons | | {key disadvantages} |
| Complexity | {L/M/H} | {implementation effort} |
| Performance | {L/M/H} | {expected perf} |
| Maintenance | {L/M/H} | {long-term burden} |
| Project Fit | {L/M/H} | {how well it fits} |

### Comparison Matrix

| Criterion | {Option A} | {Option B} | Winner |
|-----------|------------|------------|--------|
| Complexity | {rating} | {rating} | {winner} |
| Performance | {rating} | {rating} | {winner} |
| Maintenance | {rating} | {rating} | {winner} |
| Project Fit | {rating} | {rating} | {winner} |

### Recommendation

**Winner**: {recommended option}

**Reasoning**: {why this option is best for this project}

**Implementation Notes**: {specific guidance for implementing the winner}
```
