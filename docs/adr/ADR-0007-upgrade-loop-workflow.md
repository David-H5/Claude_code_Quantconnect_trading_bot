# ADR-0007: Enhanced RIC Loop Workflow (8-Phase)

**Status**: Accepted (Superseded by Meta-RIC v3.0)
**Date**: 2025-12-01
**Updated**: 2025-12-04
**Decision Makers**: Claude Code Agent, Project Owner

> **⚠️ SUPERSEDED - DO NOT EDIT FOR PROMPT CHANGES**
>
> This ADR documents historical decisions. Current workflow is **Meta-RIC v3.0** (7 phases).
>
> **To edit RIC prompts/rules, use ONLY:**
>
> - `.claude/hooks/ric_prompts.py` - Prompt text, warnings, templates
> - `.claude/RIC_CONTEXT.md` - Workflow rules, phase definitions
>
> **Current workflow**: [ENHANCED_RIC_WORKFLOW.md](../development/ENHANCED_RIC_WORKFLOW.md)

## Context

Complex development tasks require structured approach to:

- Prevent incomplete implementations
- Ensure thorough testing
- Document architectural decisions
- Identify improvements iteratively
- Avoid scope creep while catching missing pieces

Ad-hoc development leads to gaps, missed edge cases, and incomplete documentation.

## Decision (Original - v1.0)

Adopt a 6-phase iterative upgrade loop for all complex tasks:

1. **Upgrade Path**: Define target state, scope, success criteria
2. **Checklist**: Break into 30min-4hr tasks with dependencies
3. **Coding**: Execute tasks, write tests alongside code
4. **Double-Check**: Verify functional completeness, tests, docs
5. **Introspection**: Identify improvements and lessons learned
6. **Updated Path**: Incorporate insights, check convergence

Loop continues until no new P0 (critical) items identified.

---

## ADR-0008: Enhanced RIC Loop

**Status**: Accepted
**Date**: 2025-12-02
**Research**: [AUTONOMOUS_WORKFLOW_RESEARCH.md](../research/AUTONOMOUS_WORKFLOW_RESEARCH.md)

### Decision (Enhanced - v2.0)

Based on 2025 research on autonomous AI agent workflows, enhance to 8-phase RIC Loop:

0. **Research**: Online research before planning (keyword expansion, timestamped sources)
1. **Upgrade Path**: Define target state, reference research findings
2. **Checklist**: Break into 30min-4hr tasks with dependencies
3. **Coding**: Execute with **single-component changes** and checkpointing
4. **Double-Check**: Verify functional completeness, tests, docs
5. **Introspection**: Check missing files, bugs, expansion ideas
6. **Metacognition**: Self-reflection on decisions, uncertainties, assumptions
7. **Debate** (optional): Multi-agent consensus for major decisions (max 3 rounds)
8. **Convergence**: Score-based exit (>= 0.80) with plateau detection

### Key Enhancements (Research-Backed)

| Enhancement | Source | Rationale |
|-------------|--------|-----------|
| Single-component changes | IMPROVE Framework (Feb 2025) | Enables change attribution |
| Max 3 debate rounds | Multi-Agent Debate (Feb 2025) | More rounds reduce performance |
| Score-based convergence | OpenAI Cookbook (2024-2025) | Quantitative exit criteria |
| Metacognitive reflection | Anthropic (2025) | Catches blind spots |
| Checkpointing | Claude Code SDK (2025) | Safe experimentation |

### Convergence Formula

```text
Score = (0.4 × success_criteria) + (0.3 × P0_complete) + (0.2 × test_coverage) + (0.1 × P1_complete)
Exit when: Score >= 0.80 | Plateau (<2% for 2 iterations) | Max 5 iterations
```

## Consequences

### Positive

- Systematic approach prevents gaps
- Forced double-checking catches issues
- Introspection drives continuous improvement
- Clear convergence criteria
- Documentation built into process
- Audit trail of decisions

### Negative

- Overhead for simple tasks (don't use for trivial changes)
- Requires discipline to follow all phases
- May feel bureaucratic initially

### Neutral

- Learning curve for new process
- Templates needed for consistency

## Alternatives Considered

### Alternative 1: Simple Todo Lists

**Description**: Just use todo lists without phases

**Pros**:

- Simpler
- Less overhead

**Cons**:

- No verification phase
- No introspection
- Easy to miss items
- No convergence check

**Why Rejected**: Too easy to skip verification and improvement steps.

### Alternative 2: Agile Sprints Only

**Description**: Use sprint-based development without upgrade loop

**Pros**:

- Industry standard
- Good for team coordination

**Cons**:

- No per-task verification loop
- Retrospectives only at sprint end
- Gaps can accumulate

**Why Rejected**: Sprints are good for planning but don't provide per-task rigor.

## References

- [docs/development/ENHANCED_RIC_WORKFLOW.md](../development/ENHANCED_RIC_WORKFLOW.md) - **Current** Meta-RIC Loop v3.0 (7 phases)
- [docs/development/UPGRADE_LOOP_WORKFLOW.md](../development/UPGRADE_LOOP_WORKFLOW.md) - Legacy 6-phase (deprecated)
- [docs/research/UPGRADE-012.3-META-RIC-LOOP.md](../research/UPGRADE-012.3-META-RIC-LOOP.md) - v3.0 research and design
- [docs/research/UPGRADE-008-ENHANCED-RIC-LOOP.md](../research/UPGRADE-008-ENHANCED-RIC-LOOP.md) - Legacy v1.0 (superseded)
- [docs/research/AUTONOMOUS_WORKFLOW_RESEARCH.md](../research/AUTONOMOUS_WORKFLOW_RESEARCH.md) - Research sources
- [docs/IMPLEMENTATION_TRACKER.md](../IMPLEMENTATION_TRACKER.md) - Loop integration
- [CLAUDE.md](../../CLAUDE.md) - Quick reference

## Slash Commands

| Command | Purpose |
|---------|---------|
| `/ric-start` | Start new RIC loop session with templates |
| `/ric-research` | Begin Phase 0 research with keyword expansion |
| `/ric-introspect` | Run Phase 5-6 introspection and metacognition |
| `/ric-converge` | Calculate convergence score and decide next step |

## Notes

**When to Use Enhanced RIC Loop**:

- New features (multi-file)
- Refactoring (multiple components)
- Architecture changes
- Complex bug fixes
- Research-required implementations (uncertain approach)
- Major decisions (trigger Phase 7 Debate)

**When to Use Basic Loop (Phases 1-5)**:

- Implementation with known approach
- Refactoring with clear pattern

**When NOT to Use**:

- Simple bug fixes (single file)
- Documentation-only changes
- Configuration updates

**Convergence Limits**:

- Soft limit: 3 iterations (review if exceeding)
- Hard limit: 5 iterations (mandatory exit with report)
- Score threshold: >= 0.80 for success exit
- Plateau threshold: <2% improvement for 2 iterations
