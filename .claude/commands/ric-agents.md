# RIC Agents - Phase-Specific Agent Recommendations

Suggest optimal agents for the current RIC phase.

## Arguments
- `$ARGUMENTS`: Optional phase to get recommendations for (P0/P1/P2/P3/P4 or RESEARCH/PLAN/BUILD/VERIFY/REFLECT)

## Usage
```
/ric-agents           # Auto-detect current phase
/ric-agents P0        # Get agents for RESEARCH phase
/ric-agents VERIFY    # Get agents for VERIFY phase
```

## Instructions

### Auto-Detection
If no phase specified, detect from `claude-progress.txt`:
```bash
python3 .claude/hooks/agent_orchestrator.py ric-phase
```

### Phase Recommendations

#### P0 RESEARCH
Best for exploring and gathering information.

**Recommended Agents:**
- `web_researcher` (haiku) - Search web, format for docs
- `doc_finder` (haiku) - Find existing documentation
- `code_finder` (haiku) - Search codebase for patterns

**Recommended Workflow:**
```
/agents run ric_research topic="{topic}"
```

**Or use swarm for comprehensive exploration:**
```
/agent-swarm {topic}
```

#### P1 PLAN
Best for designing implementation approach.

**Recommended Agents:**
- `architect` (sonnet) - Design implementation plan

**Example:**
```python
Task(
    subagent_type="Plan",
    model="sonnet",
    description="Design implementation plan",
    prompt="Design implementation plan for: $ARGUMENTS"
)
```

#### P2 BUILD
Best for implementing features and fixes.

**Recommended Agents:**
- `implementer` (sonnet) - Write code
- `refactorer` (sonnet) - Improve code quality

**Example:**
```python
Task(
    subagent_type="general-purpose",
    model="sonnet",
    description="Implement feature",
    prompt="Implement: $ARGUMENTS"
)
```

#### P3 VERIFY
Best for testing and validation.

**Recommended Agents:**
- `test_analyzer` (haiku) - Check test coverage
- `type_checker` (haiku) - Validate types
- `security_scanner` (haiku) - Security vulnerabilities

**Recommended Workflow:**
```
/agents run ric_verify target="{files}"
```

**Or use parallel review:**
```
/parallel-review {files}
```

#### P4 REFLECT
Best for quality assessment and decision-making.

**Recommended Agents:**
- `deep_architect` (opus) - Only if complex decisions needed

**For consensus decisions:**
```
/agent-consensus "{question}"
```

## Phase → Agent Quick Reference

| Phase | Primary Agents | Model | Command |
|-------|---------------|-------|---------|
| P0 RESEARCH | web_researcher, doc_finder, code_finder | haiku | `/agent-swarm` |
| P1 PLAN | architect | sonnet | Task tool |
| P2 BUILD | implementer, refactorer | sonnet | Task tool |
| P3 VERIFY | test_analyzer, type_checker, security_scanner | haiku | `/parallel-review` |
| P4 REFLECT | deep_architect | opus | `/agent-consensus` |

## Cost Optimization

- P0, P3: Use haiku agents (cheap, fast)
- P1, P2: Use sonnet agents (balanced)
- P4: Use opus only for critical architectural decisions

**Rule**: Max 1 opus call per RIC iteration
