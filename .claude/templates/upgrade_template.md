# UPGRADE-{NUMBER}: {TITLE}

## Overview

**Created**: {YYYY-MM-DD} at {HH:MM} {TZ}
**Agent**: Claude Code
**Status**: Planning | In Progress | Complete | Blocked
**Priority**: P0 (Critical) | P1 (High) | P2 (Medium)

### Summary

{2-3 sentences explaining what this upgrade accomplishes and why}

---

## 0. Prerequisites: Codebase Consolidation

> **MANDATORY**: Before implementing ANY upgrade, complete this section.
> This prevents creating duplicate systems and amplifying technical debt.

### 0.1 Conflict Analysis

**Run Before Starting:**

```bash
python -m utils.codebase_analyzer --check-conflicts "{proposed feature description}"
```

**Results:**

| Field | Value |
|-------|-------|
| Risk Level | HIGH / MEDIUM / LOW |
| Existing Implementations | {count} |
| Related Categories | {list} |
| Date Analyzed | {YYYY-MM-DD} at {HH:MM} |

### 0.2 Existing Code Audit

> **REQUIRED**: List ALL existing code related to this upgrade.
> For each file, specify action: EXTEND, DELETE, or MERGE.

```text
+-----------------------------------------------------------------------------+
|                    EXISTING CODE AUDIT CHECKLIST                              |
+-----------------------------------------------------------------------------+
|                                                                               |
|  [ ] Ran conflict analysis (utils/codebase_analyzer.py)                      |
|  [ ] Searched for similar class/function names                                |
|  [ ] Checked KNOWN_DUPLICATIONS in codebase_analyzer.py                       |
|  [ ] Reviewed module overlaps                                                 |
|                                                                               |
+-----------------------------------------------------------------------------+
```

| File | Class/Function | Action | Justification |
|------|----------------|--------|---------------|
| `path/to/existing.py` | ExistingClass | EXTEND | Adding new method fits here |
| `path/to/old.py` | OldFunction | DELETE | Replaced by this upgrade |
| `a.py` + `b.py` | DuplicateCode | MERGE | Consolidating into unified solution |

### 0.3 Consolidation Plan

#### Deletions (Required BEFORE new code)

| # | File to Delete | Reason | Blocked By |
|---|----------------|--------|------------|
| 1 | `path/deprecated.py` | Functionality moved to X | None |
| 2 | `path/legacy.py` | No longer needed | Migration script |

#### Merges

| # | Source Files | Target File | Keep | Remove |
|---|--------------|-------------|------|--------|
| 1 | `a.py` + `b.py` | `unified.py` | func_a(), class_b() | duplicate_func() |

#### Extensions (Preferred over new files)

| # | Existing File | New Addition | Why Extension |
|---|---------------|--------------|---------------|
| 1 | `existing/module.py` | new_method() | Fits existing class responsibility |

### 0.4 Canonical Location

**New Code Path**: `{module}/{submodule}/{file}.py`

**Layer**: {0-4} (See ARCHITECTURE.md)

**Justification**:

- {Why this location fits the layer architecture}
- {What existing modules it relates to}
- {Single import path for all users}

### 0.5 Pre-Implementation Verification

```bash
# Verify conflict analysis was run
python -m utils.codebase_analyzer --check-conflicts "{feature}" --json > /tmp/conflicts.json

# Check result
cat /tmp/conflicts.json | jq '.conflict_risk'  # Should acknowledge risk level
```

---

## 1. Scope

### Goals

| # | Goal | Success Metric | Status |
|---|------|----------------|--------|
| 1 | {Goal} | {Measurable outcome} | [ ] |
| 2 | {Goal} | {Measurable outcome} | [ ] |

### Non-Goals

- {What this upgrade will NOT do}
- {Explicitly out of scope}

### Anti-Patterns to Avoid

| Anti-Pattern | Why Bad | What to Do Instead |
|--------------|---------|-------------------|
| Creating `{thing}_v2.py` | Creates parallel implementation | Extend or replace original |
| New file for one function | Unnecessary fragmentation | Add to existing module |
| Deprecation wrapper | Keeps old code alive | DELETE after migration |
| "We'll clean up later" | Never happens | Clean up NOW |

---

## 2. Research

### Phase 1: {Topic}

**Search Timestamp**: {YYYY-MM-DD} at {HH:MM} {TZ}

**Queries**: "{query1}", "{query2}"

**Sources**:

| # | Source | Published | Key Finding |
|---|--------|-----------|-------------|
| 1 | [Title](URL) | YYYY-MM | {Finding} |

> **TIMESTAMPING RULES**:
> - Search timestamp: Exact time when search was performed
> - Published date: Use `YYYY-MM`, `~YYYY` for estimates, `Unknown` if undetermined

**Applied**: {What was implemented from this research}

---

## 3. Implementation Checklist

### Status Legend

| Symbol | Meaning |
|--------|---------|
| [ ] | Not started |
| [~] | In progress |
| [x] | Complete |
| [!] | Blocked |

---

### Phase 1: Consolidation (BLOCKING)

**Goal**: Clean up existing code before adding new functionality

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 1.1 | Delete deprecated modules listed in 0.3 | `{files}` | [ ] |
| 1.2 | Merge duplicate implementations | `{files}` | [ ] |
| 1.3 | Update imports in dependent files | `{files}` | [ ] |
| 1.4 | Run tests to verify no regressions | `tests/` | [ ] |

**Gate**: Phase 1 MUST complete before Phase 2.

---

### Phase 2: Core Implementation

**Goal**: {What this phase accomplishes}
**Depends on**: Phase 1 complete

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 2.1 | {Task description} | `path/file.py` | [ ] |
| 2.2 | {Task description} | `path/file.py` | [ ] |
| 2.3 | {Task description} | `path/file.py` | [ ] |

---

### Phase 3: Integration

**Goal**: Wire new code into existing systems
**Depends on**: Phase 2 complete

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 3.1 | Update exports in `__init__.py` | `{module}/__init__.py` | [ ] |
| 3.2 | Add imports to dependent modules | `{files}` | [ ] |
| 3.3 | Update configuration if needed | `config/` | [ ] |

---

### Phase 4: Testing

**Goal**: Validate implementation

| # | Task | Target | Status |
|---|------|--------|--------|
| 4.1 | Unit tests for new code | >80% coverage | [ ] |
| 4.2 | Integration tests | All pass | [ ] |
| 4.3 | Run linter (`ruff check .`) | No errors | [ ] |
| 4.4 | Run type checker (`mypy`) | No new errors | [ ] |
| 4.5 | Verify imports work | `python -c "from X import Y"` | [ ] |

---

### Phase 5: Documentation

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 5.1 | Update CLAUDE.md if needed | `CLAUDE.md` | [ ] |
| 5.2 | Add docstrings to new code | New files | [ ] |
| 5.3 | Remove TODOs/debug code | All files | [ ] |
| 5.4 | Update KNOWN_DUPLICATIONS if applicable | `utils/codebase_analyzer.py` | [ ] |

---

## 4. Files

### Files to Delete (from Section 0.3)

| File | Reason | Replaced By |
|------|--------|-------------|
| `path/to/deprecated.py` | {Reason} | `path/to/new.py` |

### New Files

| File | Purpose | Layer |
|------|---------|-------|
| `path/to/file.py` | {Purpose} | {0-4} |

### Modified Files

| File | Changes |
|------|---------|
| `path/to/file.py` | {What changes} |

---

## 5. Migration Path

### For Existing Callers

| Current Import | New Import | Files Affected |
|----------------|------------|----------------|
| `from old import X` | `from new import X` | {count} files |

### Migration Script (if needed)

```bash
#!/bin/bash
# scripts/migrate_{upgrade_name}.sh

# Update imports
find . -name "*.py" -exec sed -i \
  's/from old.module import/from new.module import/g' {} \;

echo "Migration complete. Run tests to verify."
```

### Deprecation Timeline

| Phase | Action | Deadline |
|-------|--------|----------|
| 1 | Add deprecation warnings | Immediate |
| 2 | Update all internal callers | +1 sprint |
| 3 | DELETE deprecated code | +2 sprints max |

---

## 6. Progress

| Phase | Tasks | Done | Status |
|-------|-------|------|--------|
| 0: Prerequisites | 5 | 0 | [ ] |
| 1: Consolidation | X | 0 | [ ] |
| 2: Implementation | X | 0 | [ ] |
| 3: Integration | X | 0 | [ ] |
| 4: Testing | X | 0 | [ ] |
| 5: Documentation | X | 0 | [ ] |
| **Total** | **XX** | **0** | **0%** |

---

## 7. Definition of Done

### Per Task

- [ ] Code compiles without errors
- [ ] Unit tests added and passing
- [ ] No linting errors

### Per Phase

- [ ] All tasks in phase complete
- [ ] All tests passing
- [ ] Phase gate requirements met

### Phase 0 Gate (BLOCKING)

- [ ] Conflict analysis run and documented
- [ ] Existing code audit complete
- [ ] Consolidation plan approved
- [ ] Canonical location justified

### Phase 1 Gate (BLOCKING)

- [ ] All deletions complete
- [ ] All merges complete
- [ ] No broken imports
- [ ] Tests still pass

### Overall

- [ ] All phases complete
- [ ] Coverage >= 70%
- [ ] CLAUDE.md updated (if needed)
- [ ] No TODO comments in code
- [ ] No parallel implementations created
- [ ] Old code DELETED (not just deprecated)

---

## 8. Rollback

**Trigger**: If critical functionality breaks

```bash
git revert HEAD --no-edit
```

**Rollback Checklist**:

- [ ] Identify breaking change
- [ ] Revert commit(s)
- [ ] Verify system stability
- [ ] Document what went wrong
- [ ] Plan fix before re-attempting

---

## 9. Verification Commands

```bash
# 1. Verify no duplicate implementations created
python -m utils.codebase_analyzer --check-conflicts "{feature}" | grep "conflict_risk"

# 2. Verify deleted files are gone
ls -la path/to/deleted/files  # Should fail

# 3. Verify new imports work
python -c "from {new.module} import {NewClass}; print('OK')"

# 4. Run tests
pytest tests/ -v --tb=short

# 5. Check for remaining deprecated imports
grep -r "from {old.module} import" --include="*.py" | wc -l  # Should be 0
```

---

## 10. Change Log

| Date | Change |
|------|--------|
| {YYYY-MM-DD} | Initial creation |

---

## 11. Tags

`upgrade-{number}` `{category}` `consolidation-first`

---

## Guidelines for Using This Template

### Before Starting ANY Upgrade

1. **Run conflict analysis** - NEVER skip this step
2. **Complete Section 0** - This is BLOCKING
3. **Get consolidation plan approved** - Before writing new code
4. **Delete old code FIRST** - Not after, not "later"

### During Implementation

1. **Follow phase gates** - Don't skip ahead
2. **Update progress table** - Keep it current
3. **Run verification commands** - At each phase gate
4. **Mark tasks complete immediately** - Don't batch

### Anti-Patterns That Will Be REJECTED

- Upgrade guide without Section 0: **REJECTED**
- Plans creating parallel implementations: **REJECTED**
- Proposals without deletion timeline: **REJECTED**
- "We'll consolidate later": **REJECTED**

### Example: Good vs Bad Upgrade Plans

**BAD:**

```text
Goal: Add new sentiment analyzer
New file: llm/sentiment_v2.py
```

**GOOD:**

```text
Goal: Enhance sentiment analysis
Action: EXTEND llm/sentiment.py with new methods
Deletions: MERGE llm/emotion_detector.py into llm/sentiment.py
```

---

## References

- [Consolidation-First Policy](../../../docs/CONSOLIDATION_FIRST_POLICY.md)
- [PARALLEL_UPGRADE_COORDINATION.md](../../../docs/PARALLEL_UPGRADE_COORDINATION.md)
- [Codebase Analyzer](../../../utils/codebase_analyzer.py)
- [ARCHITECTURE.md](../../../docs/ARCHITECTURE.md)
