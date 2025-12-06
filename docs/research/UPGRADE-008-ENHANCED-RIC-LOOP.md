# UPGRADE-008: Enhanced Research-Introspection-Convergence (RIC) Loop

> ⚠️ **SUPERSEDED**: This document describes the original 8-phase RIC Loop v1.0.
> The workflow has been updated to **Meta-RIC Loop v3.0** (7 phases) in [UPGRADE-012.3](UPGRADE-012.3-META-RIC-LOOP.md).
> Key changes: consolidated to 7 phases, insight-driven iteration, tiered integration checks.
>
> **To edit RIC prompts/rules, use ONLY:**
>
> - `.claude/hooks/ric_prompts.py` - Prompt text, warnings, templates
> - `.claude/RIC_CONTEXT.md` - Workflow rules, phase definitions
>
> **Use**: [ENHANCED_RIC_WORKFLOW.md](../development/ENHANCED_RIC_WORKFLOW.md) for the current workflow.

**Version**: 1.0 (Superseded by v2.2)
**Created**: December 2, 2025
**Status**: ~~Ready for Implementation~~ **SUPERSEDED**
**Research Source**: [AUTONOMOUS_WORKFLOW_RESEARCH.md](AUTONOMOUS_WORKFLOW_RESEARCH.md)
**Full Workflow**: [ENHANCED_RIC_WORKFLOW.md](../development/ENHANCED_RIC_WORKFLOW.md) (v2.2)

---

## Executive Summary

This upgrade enhances the existing 6-phase upgrade loop with research-backed improvements from 2025 autonomous AI agent best practices. The enhanced workflow adds:

1. **Research Phase** - Online research before planning
2. **Metacognitive Introspection** - Agent self-reflection
3. **Enhanced Convergence** - Score thresholds and plateau detection
4. **Single-Component Focus** - One change at a time
5. **Checkpointing** - Git-based state preservation
6. **Debate Mechanism** - Multi-agent consensus for major decisions

---

## Checklist for User

### Pre-Implementation

- [ ] Read [AUTONOMOUS_WORKFLOW_RESEARCH.md](AUTONOMOUS_WORKFLOW_RESEARCH.md) for research context
- [ ] Review existing [UPGRADE_LOOP_WORKFLOW.md](../development/UPGRADE_LOOP_WORKFLOW.md) for baseline
- [ ] Understand the 8-phase enhanced loop in [ENHANCED_RIC_WORKFLOW.md](../development/ENHANCED_RIC_WORKFLOW.md)

### Implementation Checklist

#### Phase 0: Research Mode

- [ ] Conduct initial online research with timestamped queries
- [ ] Extract keywords from initial findings
- [ ] Conduct 2-3 expanded searches with new keywords
- [ ] Document all sources with publication dates
- [ ] Synthesize findings into actionable ideas
- [ ] Save research document to `docs/research/[TOPIC]_RESEARCH.md`

#### Phase 1: Upgrade Path (Existing + Enhanced)

- [ ] Define target state document
- [ ] Define scope (included/excluded)
- [ ] Set measurable success criteria
- [ ] Identify dependencies
- [ ] Complete risk assessment
- [ ] **NEW**: Reference research findings in rationale

#### Phase 2: Checklist (Existing)

- [ ] Create task breakdown (30min-4hr each)
- [ ] Map dependency graph
- [ ] Identify critical path
- [ ] Define test requirements
- [ ] List documentation updates

#### Phase 3: Coding (Enhanced)

- [ ] **NEW**: Modify only ONE component per change
- [ ] **NEW**: Create git checkpoint after each change
- [ ] Write tests alongside code
- [ ] Follow coding standards
- [ ] **NEW**: Attribute each change to specific requirement

#### Phase 4: Double-Check (Existing)

- [ ] Verify functional completeness
- [ ] Check test coverage >70%
- [ ] Update documentation
- [ ] Verify integration
- [ ] Confirm security/safety

#### Phase 5: Introspection (Enhanced)

- [ ] Check for missing files
- [ ] Identify bugs and broken code
- [ ] Catalogue incomplete features
- [ ] **NEW**: Generate expansion ideas using prompts:
  - Adjacent features?
  - Edge cases?
  - Integration points?
  - Performance improvements?
  - Usability enhancements?

#### Phase 6: Metacognitive Self-Introspection (NEW)

- [ ] Answer 5 reflection questions:
  1. What decisions did I make?
  2. What am I uncertain about?
  3. What assumptions am I making?
  4. What might I be missing?
  5. What would I do differently?
- [ ] Complete reasoning quality check
- [ ] Assign confidence scores (1-5) for:
  - Architecture correctness
  - Implementation quality
  - Test coverage adequacy
  - Documentation completeness
  - Overall success

#### Phase 7: Debate Mechanism (Optional)

**Trigger if ANY apply**:
- [ ] Decision affects >3 files
- [ ] Trade-offs with no clear winner
- [ ] Irreversible or expensive decision
- [ ] Initial confidence <70%
- [ ] Conflicting signals

**If triggered**:
- [ ] Articulate Position A (Pro)
- [ ] Articulate Position B (Con)
- [ ] Complete 1-2 rounds of rebuttals (max 3)
- [ ] Document decision with reasoning
- [ ] Preserve dissenting view

#### Phase 8: Updated Path & Convergence (Enhanced)

- [ ] Calculate composite score:
  - `(0.4 × success_criteria) + (0.3 × P0_complete) + (0.2 × test_coverage) + (0.1 × P1_complete)`
- [ ] Check convergence conditions:
  - **SUCCESS EXIT**: Score >= 0.80
  - **PLATEAU EXIT**: <2% improvement for 2 iterations
  - **FORCED EXIT**: Max 5 iterations reached
- [ ] If continuing: Check if new research needed
- [ ] Document decision and next iteration focus

---

## Key Differences from Original Workflow

| Aspect | Original (v1.0) | Enhanced (v2.0) |
|--------|-----------------|-----------------|
| Research | Assumed done externally | Explicit Phase 0 with templated documentation |
| Coding Changes | Multiple per iteration | Single-component per change (attribution) |
| State Management | Manual | Git checkpointing at each phase |
| Self-Reflection | Phase 5 introspection | Phase 5 + NEW Phase 6 metacognition |
| Major Decisions | Implicit | Explicit debate mechanism (Phase 7) |
| Convergence | P0 items + coverage | Composite score + plateau detection |
| Exit Conditions | 3/5 iteration limits | Score thresholds + plateau + forced |
| Documentation | After completion | Timestamped throughout |

---

## Research-Backed Rationale

Each enhancement is backed by 2025 research:

| Enhancement | Source | Key Finding |
|-------------|--------|-------------|
| Single-component changes | IMPROVE Framework (Feb 2025) | Attribute changes, stable improvement |
| Max 3 debate rounds | Multi-Agent Debate (Feb 2025) | More rounds = reduced performance |
| Score-based convergence | OpenAI Cookbook (2024-2025) | Threshold + max retries pattern |
| Metacognitive reflection | Anthropic (2025) | Debugging through introspection conversation |
| Checkpointing | Claude Code SDK (2025) | State preservation and reversion |
| Timestamped research | Knowledge Management (2025) | Provenance tracking, source verification |

---

## Quick Start

### For Your Next Complex Task

1. **Start with Research** (Phase 0):
   ```
   Search: "[your topic] best practices 2025"
   Document: docs/research/[TOPIC]_RESEARCH.md
   ```

2. **Plan with Research Context** (Phase 1):
   - Reference findings in rationale
   - Set measurable success criteria

3. **Code One Thing at a Time** (Phase 3):
   - Single-component changes
   - Checkpoint after each change
   - Clear attribution

4. **Reflect Deeply** (Phases 5-6):
   - Check for missing pieces
   - Ask yourself the 5 metacognitive questions
   - Document uncertainties

5. **Measure Convergence** (Phase 8):
   - Calculate composite score
   - Check for plateau
   - Exit or continue with reason

---

## Integration Points

### CLAUDE.md Update Required

Add to CLAUDE.md under "CRITICAL: Upgrade Loop Workflow":

```markdown
### Enhanced RIC Loop (UPGRADE-008)

For complex research-required tasks, use the **Enhanced Research-Introspection-Convergence Loop**:

- **When**: Complex tasks, uncertain approaches, major decisions, research needed
- **Phases**: 0 (Research) → 1-4 (Core) → 5 (Introspection) → 6 (Metacognition) → 7 (Debate, optional) → 8 (Convergence)
- **Key Rules**:
  - Single-component changes only (Phase 3)
  - Checkpoint after each change
  - Max 3 debate rounds
  - Exit when score >= 0.80 or plateau detected

**See**: [ENHANCED_RIC_WORKFLOW.md](docs/development/ENHANCED_RIC_WORKFLOW.md)
```

### Research Index Update

Update `docs/research/README.md` to include:
- AUTONOMOUS_WORKFLOW_RESEARCH.md
- UPGRADE-008-ENHANCED-RIC-LOOP.md

---

## Success Metrics

After implementing this upgrade, track:

| Metric | Baseline | Target | How to Measure |
|--------|----------|--------|----------------|
| Iterations per upgrade | ~3 | <3 | Track in IMPLEMENTATION_TRACKER |
| Research attribution | 0% | >80% | Count decisions with cited sources |
| Change attribution | Partial | 100% | Every change linked to requirement |
| Convergence accuracy | N/A | >90% | Compare predicted vs actual completion |

---

## Change Log

| Date | Version | Change |
|------|---------|--------|
| 2025-12-02 | 1.0 | Initial upgrade guide created |

---

**Created By**: Claude Code Agent
**Research Date**: December 2, 2025
**Review Required**: Before first use on production task
