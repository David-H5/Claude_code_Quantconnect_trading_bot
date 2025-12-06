# UPGRADE-014 - Comprehensive Implementation Guide

**Generated**: 2025-12-03T23:53:02.990942
**Purpose**: Pre-loaded context for overnight autonomous sessions
**Survives**: Context compaction, session restarts, crashes

---

## Quick Start (READ FIRST)

1. **Check current state**: `tail -50 claude-progress.txt`
2. **Find next task**: Look for first unchecked `- [ ]` item
3. **Start working**: Implement task, run tests, commit
4. **Mark complete**: Change `- [ ]` to `- [x]` in progress file

### Key Files
| File | Purpose |
|------|---------|
| `claude-progress.txt` | Task list and completion status |
| `claude-session-notes.md` | Context for relay-race pattern |
| `claude-recovery-context.md` | Post-compaction recovery |
| This file | Domain knowledge and guides |

---

## RIC Loop Quick Reference

For complex tasks, use the Meta-RIC Loop (7 phases, min 3 iterations):

| Phase | Purpose | Gate Criteria |
|-------|---------|---------------|
| **0. Research** | Online research | Sources timestamped |
| **1. Upgrade Path** | Define scope, success criteria | Clear target state |
| **2. Checklist** | Break into tasks (30min-4hr each) | Tasks prioritized P0/P1/P2 |
| **3. Coding** | Execute ONE component at a time | Tests pass, committed |
| **4. Double-Check** | Verify completeness | >70% coverage |
| **5. Introspection** | Find gaps, bugs, expansion ideas | All gaps documented |
| **6. Metacognition** | Self-reflection, classify insights | P0/P1/P2 classified |
| **7. Integration** | Loop decision: continue or exit | All P0-P2 resolved |

### Exit Rules
- Minimum 3 iterations required
- ALL P0, P1, AND P2 must be resolved
- Exit only when iteration >= 3 AND no P0/P1/P2 remain

### Slash Commands
- `/ric-start` - Start new RIC loop
- `/ric-introspect` - Run introspection phase
- `/ric-converge` - Check convergence and decide

---

## Overview

This guide contains all information needed to complete UPGRADE-014.
After context compaction, Claude should READ THIS FILE to restore context.

## Priority Order

Complete categories in this order:
1. **P0 (Critical)**: Must complete first - BLOCKING
2. **P1 (Important)**: Required for full implementation - REQUIRED
3. **P2 (Nice-to-have)**: Also REQUIRED for overnight sessions

**IMPORTANT**: The stop hook enforces 100% completion by default.
You cannot stop until ALL P0, P1, AND P2 tasks are complete.

---

## P0 Categories

### Category 1: Architecture Enhancements [✓ COMPLETE]

### Category 2: Observability & Debugging [✓ COMPLETE]

### Category 3: Fault Tolerance [✓ COMPLETE]

### Category 5: Safety Guardrails [✓ COMPLETE]

### Category 11: Overnight Session Enhancements [✓ COMPLETE]

### Category 12: Claude Code Specific [✓ COMPLETE]

## P1 Categories

### Category 4: Memory Management [✓ COMPLETE]

### Category 6: Cost Optimization [✓ COMPLETE]

### Category 7: State Persistence [✓ COMPLETE]

### Category 8: Testing & Simulation [✓ COMPLETE]

## P2 Categories

### Category 9: Self-Improvement [○ PENDING]

## Implementation Guide: Category 9: Self-Improvement

**Research Doc**: `docs/research/UPGRADE-014-CAT9-SELF-IMPROVEMENT-RESEARCH.md`

### Tasks to Complete
- [○] 9.1 Feedback loop (capture outcomes, learn)
- [○] 9.2 Automatic prompt optimization (APO)
- [○] 9.3 Evaluator-Optimizer pattern
- [○] 9.4 Performance trend analysis
- [○] Tests for self-improvement

### Design Patterns
- Feedback loop with outcome capture
- Automatic prompt optimization (APO)
- Evaluator-Optimizer pattern
- Performance trend analysis

### Key Concepts
- Feedback loop = observe outcome, adjust behavior
- APO = iteratively refine prompts based on performance
- Evaluator = scores agent outputs objectively
- Optimizer = modifies prompts to improve scores

### Implementation Hints
- Log all decisions with confidence and outcomes
- Calculate calibration error (confidence vs accuracy)
- Use A/B testing for prompt variations
- Implement prompt mutation operators

### Test Cases to Write
- Test feedback capture and storage
- Test prompt optimization converges
- Test evaluator scoring consistency
- Test trend detection accuracy

### Completion Criteria
- All tasks marked with [x] in progress file
- Tests pass with >70% coverage
- No linting errors
- Documentation updated

### Category 10: Workspace Management [○ PENDING]

## Implementation Guide: Category 10: Workspace Management

**Research Doc**: `docs/research/UPGRADE-014-CAT10-WORKSPACE-MANAGEMENT-RESEARCH.md`

### Tasks to Complete
- [○] 10.1 AGENTS.md standard (per-directory instructions)
- [○] 10.2 Real-time codebase indexing
- [○] 10.3 Event-based triggers
- [○] 10.4 Multi-agent coordination
- [○] Tests for workspace management

### Design Patterns
- AGENTS.md standard for directory instructions
- Real-time codebase indexing
- Event-based triggers for agent actions
- Multi-agent coordination protocols

### Key Concepts
- AGENTS.md = per-directory agent instructions
- Codebase index = fast symbol/file lookup
- Event triggers = file changes, git commits, etc.
- Coordination = agent communication and task handoff

### Implementation Hints
- Parse AGENTS.md files recursively
- Use AST parsing for code indexing
- Implement file watcher for events
- Use message passing for agent coordination

### Test Cases to Write
- Test AGENTS.md parsing and merging
- Test index accuracy and performance
- Test event detection and handling
- Test multi-agent task handoff

### Completion Criteria
- All tasks marked with [x] in progress file
- Tests pass with >70% coverage
- No linting errors
- Documentation updated

## Recent Upgrade Documents

- **UPGRADE-014-CAT9-SELF-IMPROVEMENT-RESEARCH**: UPGRADE-014 Category 9: Self-Improvement Research... [Unknown]
- **UPGRADE-014-CAT8-TESTING-SIMULATION-RESEARCH**: UPGRADE-014-CAT8-TESTING-SIMULATION-RESEARCH... [Unknown]
- **UPGRADE-014-CAT7-STATE-PERSISTENCE-RESEARCH**: UPGRADE-014-CAT7-STATE-PERSISTENCE-RESEARCH... [Unknown]
- **UPGRADE-014-CAT6-COST-OPTIMIZATION-RESEARCH**: UPGRADE-014-CAT6-COST-OPTIMIZATION-RESEARCH... [Unknown]
- **UPGRADE-014-CAT5-SAFETY-GUARDRAILS-RESEARCH**: UPGRADE-014-CAT5-SAFETY-GUARDRAILS-RESEARCH... [Unknown]

---

## Robustness Guidelines

### After Context Compaction
1. Read this file (claude-upgrade-guide.md)
2. Read claude-progress.txt for current status
3. Read claude-recovery-context.md for session state
4. Continue from where you left off

### Error Recovery
- If a task fails, document the error in BLOCKERS section
- Try alternative approaches before giving up
- Create git checkpoint after each successful task

### Verification
- Run tests after each implementation
- Check for linting errors
- Verify imports work correctly

---

## Quick Reference

### Files to Update
- `claude-progress.txt` - Mark tasks [x] when complete
- `claude-session-notes.md` - Document key decisions
- Test files in `tests/` - Create tests for new code

### Commands
- Run tests: `.venv/bin/pytest tests/ -v`
- Lint: `ruff check .`
- Checkpoint: `git add -A && git commit -m "checkpoint: [description]"`
- QA Check: `python scripts/qa_validator.py`

### Slash Commands
- `/overnight` - Setup overnight session
- `/overnight-status` - Check session status
- `/ric-start` - Start RIC loop for complex tasks
- `/qa-debug` - Run debug-focused QA checks
