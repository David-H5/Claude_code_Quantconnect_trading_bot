# UPGRADE-{NUMBER}: {TITLE}

## 📋 Overview

**Created**: {YYYY-MM-DD} at {HH:MM} {TZ}
**Agent**: Claude Code
**Status**: Planning | In Progress | Complete | Blocked
**Priority**: P0 (Critical) | P1 (High) | P2 (Medium)

### Summary

{2-3 sentences explaining what this upgrade accomplishes and why}

---

## 🎯 Scope

### Goals

| # | Goal | Success Metric | Status |
|---|------|----------------|--------|
| 1 | {Goal} | {Measurable outcome} | ⬜ |
| 2 | {Goal} | {Measurable outcome} | ⬜ |

### Non-Goals

- {What this upgrade will NOT do}
- {Explicitly out of scope}

---

## 📊 Research

### Phase 1: {Topic}

**Search Timestamp**: {YYYY-MM-DD} at {HH:MM} {TZ}  ← **REQUIRED**

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

## ✅ Implementation Checklist

### Status Legend

| Symbol | Meaning |
|--------|---------|
| ⬜ | Not started |
| 🔄 | In progress |
| ✅ | Complete |
| ⏸️ | Blocked |

---

### Phase 1: {Name}

**Goal**: {What this phase accomplishes}

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 1.1 | {Task description} | `path/file.py` | ⬜ |
| 1.2 | {Task description} | `path/file.py` | ⬜ |
| 1.3 | {Task description} | `path/file.py` | ⬜ |

---

### Phase 2: {Name}

**Goal**: {What this phase accomplishes}
**Depends on**: Phase 1 complete

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 2.1 | {Task description} | `path/file.py` | ⬜ |
| 2.2 | {Task description} | `path/file.py` | ⬜ |

---

### Phase 3: Testing

**Goal**: Validate implementation

| # | Task | Target | Status |
|---|------|--------|--------|
| 3.1 | Unit tests | >80% coverage | ⬜ |
| 3.2 | Integration tests | All pass | ⬜ |
| 3.3 | Run linter | No errors | ⬜ |

---

### Phase 4: Documentation

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 4.1 | Update CLAUDE.md | `CLAUDE.md` | ⬜ |
| 4.2 | Add docstrings | New files | ⬜ |
| 4.3 | Remove TODOs/debug code | All files | ⬜ |

---

## 📁 Files

### New Files

| File | Purpose |
|------|---------|
| `path/to/file.py` | {Purpose} |

### Modified Files

| File | Changes |
|------|---------|
| `path/to/file.py` | {What changes} |

---

## 📊 Progress

| Phase | Tasks | Done | Status |
|-------|-------|------|--------|
| 1: {Name} | X | 0 | ⬜ |
| 2: {Name} | X | 0 | ⬜ |
| 3: Testing | X | 0 | ⬜ |
| 4: Docs | X | 0 | ⬜ |
| **Total** | **XX** | **0** | **0%** |

---

## ✔️ Definition of Done

### Per Task

- [ ] Code compiles without errors
- [ ] Unit tests added and passing
- [ ] No linting errors

### Per Phase

- [ ] All tasks in phase complete
- [ ] All tests passing

### Overall

- [ ] All phases complete
- [ ] Coverage ≥ 70%
- [ ] CLAUDE.md updated
- [ ] No TODO comments in code

---

## 🔙 Rollback

**Trigger**: If critical functionality breaks

```bash
git revert HEAD --no-edit
```

---

## 📝 Change Log

| Date | Change |
|------|--------|
| {YYYY-MM-DD} | Initial creation |

---

## 📊 Tags

`upgrade-{number}` `{category}`
