# Upgrade Loop Workflow (DEPRECATED)

> ## ⚠️ DEPRECATED - Use Meta-RIC Loop v3.0 Instead
>
> **This document has been superseded by the Meta-RIC Loop v3.0 (7 phases).**
>
> **To edit RIC prompts/rules, use ONLY these 2 files:**
>
> - `.claude/hooks/ric_prompts.py` - Prompt text, warnings, templates
> - `.claude/RIC_CONTEXT.md` - Workflow rules, phase definitions
>
> The current Meta-RIC Loop v3.0 features:
>
> - **Phase 0: Research Mode** - Online research before planning
> - **Phase 5: Introspection** - Gap and bug detection
> - **Phase 6: Metacognition** - Classified insights (P0/P1/P2)
> - **Phase 7: Integration** - Loop decision (ALL P0-P2 must be resolved)
> - **Single-Component Rule** - Modify ONE file at a time
> - **Minimum 3 Iterations** - Ensures thoroughness
>
> **See**: [ENHANCED_RIC_WORKFLOW.md](ENHANCED_RIC_WORKFLOW.md) for the current 7-phase workflow
>
> **Slash Commands**:
>
> - `/ric-start` - Start new RIC loop session
> - `/ric-research` - Begin Phase 0 research
> - `/ric-introspect` - Run Phases 5-6
> - `/ric-converge` - Phase 7 integration decision

---

**Version**: 1.0 (DEPRECATED - See v3.0 in ENHANCED_RIC_WORKFLOW.md)
**Created**: December 1, 2025
**Deprecated**: December 2, 2025
**Replacement**: [ENHANCED_RIC_WORKFLOW.md](ENHANCED_RIC_WORKFLOW.md)
**Purpose**: Legacy reference - DO NOT USE for new tasks

---

## Overview (LEGACY)

> **WARNING**: This 6-phase loop is deprecated. Use the [Meta-RIC Loop v3.0 (7 phases)](ENHANCED_RIC_WORKFLOW.md) instead.

This document defines a **6-phase iterative loop** for implementing upgrades, features, and improvements. Each phase must be completed and verified before proceeding to the next. The loop continues until convergence (no new improvements identified) or explicit completion criteria met.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        UPGRADE LOOP WORKFLOW                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                │
│   │ 1. UPGRADE  │────▶│ 2. CHECKLIST│────▶│ 3. CODING   │                │
│   │    PATH     │     │             │     │             │                │
│   └─────────────┘     └─────────────┘     └──────┬──────┘                │
│         ▲                                         │                       │
│         │                                         ▼                       │
│   ┌─────┴───────┐     ┌─────────────┐     ┌─────────────┐                │
│   │ 6. UPDATED  │◀────│ 5. INTRO-   │◀────│ 4. DOUBLE   │                │
│   │    PATH     │     │   SPECTION  │     │    CHECK    │                │
│   └─────────────┘     └─────────────┘     └─────────────┘                │
│         │                                                                 │
│         │  Convergence Check                                             │
│         │  ├── New improvements found? → Continue loop                   │
│         │  └── No new improvements? → EXIT (Complete)                    │
│         │                                                                 │
│         └─────────────────────────────────────────────────────────────── │
│                                  LOOP                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Upgrade Path

### Purpose
Define the target state, scope, and objectives of the upgrade.

### Required Outputs
- [ ] **Target State Document**: Clear description of what the system should do after the upgrade
- [ ] **Scope Definition**: What is included and explicitly excluded
- [ ] **Success Criteria**: Measurable criteria for completion
- [ ] **Dependencies**: What must exist before starting
- [ ] **Risk Assessment**: Potential issues and mitigations

### Template

```markdown
## Upgrade Path: [Name]

### Target State
[Describe the desired end state in detail]

### Scope
**Included:**
- [Item 1]
- [Item 2]

**Excluded:**
- [Item 1]
- [Item 2]

### Success Criteria
| Criterion | Metric | Target |
|-----------|--------|--------|
| [Criterion 1] | [How to measure] | [Target value] |

### Dependencies
- [ ] [Dependency 1]
- [ ] [Dependency 2]

### Risks
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| [Risk 1] | [L/M/H] | [L/M/H] | [Strategy] |

### Estimated Effort
[Time estimate and confidence level]
```

### Verification Checklist
- [ ] Target state is specific and measurable
- [ ] Scope boundaries are clear
- [ ] Success criteria are testable
- [ ] All dependencies identified
- [ ] Risk assessment complete
- [ ] Effort estimate provided

### Gate: Proceed when all verification items checked

---

## Phase 2: Checklist

### Purpose
Convert the upgrade path into actionable, sequential checklist items.

### Required Outputs
- [ ] **Task Breakdown**: Granular tasks (30 min - 4 hours each)
- [ ] **Dependency Graph**: Task dependencies mapped
- [ ] **Priority Order**: Critical path identified
- [ ] **Test Requirements**: Tests needed for each task
- [ ] **Documentation Updates**: Docs to create/update

### Template

```markdown
## Checklist: [Upgrade Name]

### Task Breakdown

| ID | Task | Est. Time | Depends On | Priority | Tests Required |
|----|------|-----------|------------|----------|----------------|
| T1 | [Task description] | [Time] | - | P0 | [Test IDs] |
| T2 | [Task description] | [Time] | T1 | P0 | [Test IDs] |

### Critical Path
T1 → T2 → T3 → ... → Tn

### Test Matrix

| Task ID | Unit Tests | Integration Tests | E2E Tests |
|---------|------------|-------------------|-----------|
| T1 | [ ] test_xxx | [ ] test_xxx_integration | - |

### Documentation Updates

| Document | Update Type | Priority |
|----------|-------------|----------|
| [Doc path] | [Create/Update/Delete] | [P0/P1/P2] |

### Blockers Identified
- [ ] [Blocker 1]: [Resolution strategy]
```

### Verification Checklist
- [ ] All tasks are actionable (start with verb)
- [ ] Each task is 30 min - 4 hours
- [ ] Dependencies are accurate
- [ ] Tests identified for each task
- [ ] Documentation needs listed
- [ ] No circular dependencies
- [ ] Critical path is achievable

### Gate: Proceed when all verification items checked

---

## Phase 3: Coding

### Purpose
Execute the checklist items in priority order.

### Process

1. **Before Starting Each Task:**
   - [ ] Mark task as `in_progress` in TodoWrite
   - [ ] Read all related files first
   - [ ] Review existing patterns in codebase
   - [ ] Identify test file locations

2. **During Implementation:**
   - [ ] Follow coding standards ([CODING_STANDARDS.md](CODING_STANDARDS.md))
   - [ ] Write tests alongside code
   - [ ] Add type hints to all functions
   - [ ] Document public APIs
   - [ ] Keep commits atomic and descriptive

3. **After Each Task:**
   - [ ] Run tests: `pytest tests/test_[module].py -v`
   - [ ] Run linting: `flake8 [module]`
   - [ ] Run type checking: `mypy [module]`
   - [ ] Mark task as `completed` in TodoWrite
   - [ ] Update documentation if needed

### Template: Task Completion Record

```markdown
## Task Completion: [Task ID]

### Implementation Summary
- Files Created: [list]
- Files Modified: [list]
- Lines of Code: [count]

### Tests Added
- [test_file.py::test_name] - [description]

### Verification
- [ ] All tests pass
- [ ] No linting errors
- [ ] Type hints complete
- [ ] Documentation updated

### Notes
[Any issues encountered or decisions made]
```

### Verification Checklist
- [ ] All tasks marked complete
- [ ] All tests pass (`pytest --tb=short`)
- [ ] No linting errors (`flake8`)
- [ ] No type errors (`mypy`)
- [ ] Code reviewed (self or peer)
- [ ] Commits pushed to branch

### Gate: Proceed when all verification items checked

---

## Phase 4: Double-Check for Missing Parts

### Purpose
Systematic verification that all requirements were met and nothing was overlooked.

### Checklist Categories

#### 4.1 Functional Completeness
- [ ] All success criteria from Phase 1 met?
- [ ] All scope items implemented?
- [ ] All edge cases handled?
- [ ] Error handling complete?
- [ ] Logging added where needed?

#### 4.2 Test Coverage
- [ ] Unit test coverage > 70%?
- [ ] Integration tests exist?
- [ ] Edge case tests written?
- [ ] Error path tests exist?
- [ ] Performance tests (if applicable)?

#### 4.3 Documentation
- [ ] API documentation complete?
- [ ] README updated (if public API)?
- [ ] CLAUDE.md updated (if workflow changed)?
- [ ] Code comments adequate?
- [ ] Examples provided?

#### 4.4 Integration
- [ ] Works with existing components?
- [ ] No regressions introduced?
- [ ] Configuration documented?
- [ ] Migration path clear (if breaking)?

#### 4.5 Security & Safety
- [ ] No sensitive data exposed?
- [ ] Input validation present?
- [ ] Circuit breaker integration (if trading)?
- [ ] Fail-safe defaults used?

### Template: Double-Check Report

```markdown
## Double-Check Report: [Upgrade Name]

### Date: [YYYY-MM-DD]
### Checked By: [Name/Agent]

### Functional Completeness: [PASS/FAIL]
| Criterion | Status | Notes |
|-----------|--------|-------|
| [Criterion 1] | [✅/❌] | [Notes] |

### Test Coverage: [XX%]
- Unit: [X/Y passing]
- Integration: [X/Y passing]
- E2E: [X/Y passing]

### Documentation: [COMPLETE/INCOMPLETE]
| Document | Status | Notes |
|----------|--------|-------|
| [Doc 1] | [✅/❌] | [Notes] |

### Missing Parts Identified
1. [Missing item 1]
2. [Missing item 2]

### Action Items
- [ ] [Action 1]
- [ ] [Action 2]
```

### Verification Checklist
- [ ] All 5 categories reviewed
- [ ] Missing parts documented
- [ ] Action items created for gaps
- [ ] Critical gaps addressed immediately

### Gate: Proceed when critical gaps fixed, non-critical documented

---

## Phase 5: Introspection on Additional Features

### Purpose
Reflect on the implementation to identify improvements, extensions, and lessons learned.

### Guiding Questions

#### 5.1 Code Quality Improvements
- Could any code be simplified?
- Are there repeated patterns that could be abstracted?
- Would additional utilities help?
- Is the architecture scalable?

#### 5.2 Feature Extensions
- What adjacent features would users want?
- What optional enhancements would add value?
- What integrations are now possible?
- What optimizations could improve performance?

#### 5.3 Developer Experience
- Is the API intuitive?
- Are error messages helpful?
- Is debugging easy?
- Would better tooling help?

#### 5.4 Lessons Learned
- What took longer than expected?
- What would you do differently?
- What assumptions were wrong?
- What worked well?

### Template: Introspection Report

```markdown
## Introspection Report: [Upgrade Name]

### Date: [YYYY-MM-DD]

### Code Quality Improvements
| Improvement | Priority | Effort | Impact |
|-------------|----------|--------|--------|
| [Improvement 1] | [P0/P1/P2] | [S/M/L] | [Description] |

### Feature Extensions
| Feature | Priority | Effort | Value |
|---------|----------|--------|-------|
| [Feature 1] | [P0/P1/P2] | [S/M/L] | [Description] |

### Developer Experience
| Enhancement | Priority | Effort |
|-------------|----------|--------|
| [Enhancement 1] | [P0/P1/P2] | [S/M/L] |

### Lessons Learned
1. **What worked:** [Description]
2. **What didn't:** [Description]
3. **Key insight:** [Description]

### Recommended Next Steps
1. [Step 1]
2. [Step 2]
3. [Step 3]
```

### Verification Checklist
- [ ] All 4 categories explored
- [ ] Improvements prioritized
- [ ] Effort estimates provided
- [ ] Lessons documented
- [ ] Next steps identified

### Gate: Proceed when introspection complete

---

## Phase 6: Updated Upgrade Path

### Purpose
Incorporate insights from introspection into the upgrade path for the next iteration.

### Process

1. **Review Introspection Output:**
   - High-priority improvements
   - Valuable feature extensions
   - Lessons to apply

2. **Update Planning Documents:**
   - Add new items to roadmap
   - Create new implementation tasks
   - Update success criteria if needed

3. **Check Convergence:**
   - Are there new P0 items? → Continue loop
   - Only P1/P2 items? → Consider completion
   - No new items? → Declare convergence

### Template: Updated Path Summary

```markdown
## Updated Upgrade Path: [Upgrade Name] - Iteration [N]

### Previous Iteration Summary
- Tasks Completed: [X/Y]
- Test Coverage: [XX%]
- Missing Parts Fixed: [X/Y]

### New Items Identified
| Item | Type | Priority | Add to Iteration |
|------|------|----------|------------------|
| [Item 1] | [Improvement/Feature/Fix] | [P0/P1/P2] | [Yes/No/Backlog] |

### Convergence Status
- [ ] No new P0 items identified
- [ ] All success criteria met
- [ ] No critical gaps remaining

### Decision
- [ ] **CONTINUE LOOP** - New P0 items to address
- [ ] **EXIT LOOP** - Convergence achieved
- [ ] **PAUSE** - Waiting for external dependency

### Next Iteration Focus
[If continuing, what is the focus?]
```

### Convergence Criteria

The loop exits when ALL of the following are true:

1. **No new P0 items**: All critical improvements addressed
2. **Success criteria met**: All Phase 1 criteria achieved
3. **Test coverage adequate**: >70% coverage maintained
4. **Documentation complete**: All docs updated
5. **No critical gaps**: Phase 4 double-check passes

### Maximum Iterations

To prevent infinite loops:
- **Soft limit**: 3 iterations (review if exceeding)
- **Hard limit**: 5 iterations (mandatory exit with report)

---

## Quick Reference: Phase Gates

| Phase | Gate Criteria | Blocker Action |
|-------|---------------|----------------|
| 1. Upgrade Path | All verification items checked | Cannot proceed until complete |
| 2. Checklist | All verification items checked | Cannot proceed until complete |
| 3. Coding | All tests pass, no lint errors | Fix failures before proceeding |
| 4. Double-Check | Critical gaps addressed | Fix critical items, document others |
| 5. Introspection | Report complete | Cannot skip, even if brief |
| 6. Updated Path | Convergence decision made | Must decide: continue/exit/pause |

---

## Integration with Project Workflow

### When to Use This Workflow

Use the Upgrade Loop Workflow for:
- **New features** requiring multiple files/modules
- **Refactoring** touching multiple components
- **Architecture changes** with broad impact
- **Complex bug fixes** with multiple root causes
- **Research-based implementations** from new findings

Do NOT use for:
- Simple bug fixes (single file)
- Documentation-only changes
- Configuration updates
- One-off scripts

### Linking to Implementation Tracker

When starting an upgrade loop:
1. Create entry in [IMPLEMENTATION_TRACKER.md](../IMPLEMENTATION_TRACKER.md)
2. Reference the upgrade loop iteration number
3. Update tracker after each iteration
4. Mark complete when loop exits

### Linking to CLAUDE.md

The main [CLAUDE.md](../../CLAUDE.md) file references this workflow. When Claude Code encounters a complex task:
1. Check if upgrade loop applies
2. If yes, follow this workflow
3. Record progress in implementation tracker
4. Update CLAUDE.md if workflow improvements identified

---

## Example: Full Workflow Execution

### Example: Implementing Circuit Breaker Agent Integration

#### Iteration 1

**Phase 1: Upgrade Path**
```
Target: Integrate circuit breaker into all LLM agents
Scope: SafeAgentWrapper, factory functions, tests
Success: All agents check circuit breaker before trading
```

**Phase 2: Checklist**
```
T1: Create SafeAgentWrapper class (2h) - P0
T2: Add risk tier classification (1h) - P0, depends T1
T3: Create factory functions (2h) - P0, depends T1
T4: Write unit tests (2h) - P0, depends T1-T3
T5: Write integration tests (1h) - P1, depends T4
```

**Phase 3: Coding**
- Implemented SafeAgentWrapper
- Created 5 factory functions
- Added 60+ test cases
- All tests passing

**Phase 4: Double-Check**
- Functional: PASS (all agents wrapped)
- Tests: 85% coverage
- Docs: API docs added
- Missing: No integration with supervisor

**Phase 5: Introspection**
- Improvement: Add caching to circuit breaker checks
- Feature: Add circuit breaker dashboard widget
- Lesson: Start with supervisor integration next time

**Phase 6: Updated Path**
- New P0: Supervisor integration (was missing)
- Decision: CONTINUE LOOP

#### Iteration 2

**Phase 1: Updated Upgrade Path**
```
Target: Integrate circuit breaker with supervisor agent
Scope: Supervisor modification, integration tests
Success: Supervisor checks circuit breaker for all decisions
```

[Continue workflow...]

**Phase 6: Updated Path**
- No new P0 items
- All success criteria met
- Decision: EXIT LOOP - Convergence achieved

---

## Change Log

| Date | Version | Change |
|------|---------|--------|
| 2025-12-01 | 1.0 | Initial workflow document created |

---

**Maintained By**: Claude Code Agent + Human Review
**Review Schedule**: Monthly or after major upgrades
