# Consolidation-First Policy

**Priority**: P0 - Required for all planning and upgrade work

## Core Principle

> **NEVER create new systems when existing ones can be extended.**
> **ALWAYS consolidate duplicates before adding features.**
> **DELETE old code, don't just deprecate it.**

---

## Pre-Planning Checklist (MANDATORY)

Before proposing ANY upgrade, feature, or new system:

### Step 1: Run Conflict Analysis
```bash
python -m utils.codebase_analyzer --check-conflicts "your proposed feature"
python -m utils.codebase_analyzer --find-similar "ClassName"
```

### Step 2: Check Known Duplications

| Category | Current Files | Required Action |
|----------|---------------|-----------------|
| **Anomaly Detection** | `observability/anomaly_detector.py`, `models/anomaly_detector.py` | MUST consolidate before adding new anomaly code |
| **Sentiment Analysis** | `llm/sentiment.py`, `emotion_detector.py`, `reddit_sentiment.py`, `sentiment_filter.py` | EXTEND existing, DO NOT create new analyzers |
| **Spread Analysis** | `execution/spread_analysis.py`, `spread_anomaly.py` | MERGE before touching |
| **Logging** | `observability/logging/` (5+ modules) | Use existing loggers, don't create new |
| **Monitoring** | `observability/monitoring/` | Add to existing monitors |
| **Validation** | Multiple validators exist | Check before creating |

### Step 3: Use AgentLogger Conflict Check
```python
from observability.logging import create_agent_logger

logger = create_agent_logger("my-agent")
result = logger.check_conflicts_before_change("Add new sentiment analyzer")

if result["conflict_risk"] == "high":
    # STOP - Review existing implementations first
    print(result["recommendations"])
```

---

## Upgrade Guide Requirements

Every upgrade guide MUST contain:

### Required Section 1: Existing Code Audit

```markdown
## Existing Related Code

| File | Component | Action | Justification |
|------|-----------|--------|---------------|
| `path/file.py` | ClassName | EXTEND | Adding new method fits here |
| `old/dup.py` | OldClass | DELETE | Replaced by unified solution |
| `a.py` + `b.py` | Mixed | MERGE → `unified.py` | Consolidating duplicates |
```

### Required Section 2: Consolidation Plan

```markdown
## Consolidation Actions

### Deletions (Required before new code)
1. `old/deprecated_analyzer.py` - Functionality moved to X
2. `utils/legacy_helper.py` - No longer needed

### Merges
1. `module_a.py` + `module_b.py` → `unified_module.py`
   - Keep: function_from_a(), class_from_b()
   - Remove: duplicate_function()

### Extensions (Preferred over new files)
1. `existing/analyzer.py` - Add new_method() to existing class
```

### Required Section 3: Single Canonical Location

```markdown
## Canonical Location

New code location: `observability/monitoring/trading/new_monitor.py`

Justification:
- Fits Layer 1 (Infrastructure) architecture
- Groups with related monitors
- Single import path for all users
```

---

## Anti-Patterns (PROHIBITED)

| ❌ Don't Do This | ✅ Do This Instead |
|------------------|-------------------|
| Create `sentiment_v2.py` alongside `sentiment.py` | Extend or replace `sentiment.py` |
| Add deprecation wrapper, keep old code | DELETE old code after migration |
| Create parallel implementation "temporarily" | Consolidate immediately |
| New file for 1 function | Add to existing module |
| `utils/new_helper.py` for every task | Extend existing utils |
| "We'll clean this up later" | Clean up NOW |

---

## Enforcement

### Automatic Checks
- `utils/codebase_analyzer.py` - Run before any planning
- `AgentLogger.check_conflicts_before_change()` - Logs conflict analysis
- Pre-commit hooks validate no new duplications

### Review Requirements
- Upgrade guides without Existing Code Audit section: **REJECTED**
- Plans creating parallel implementations: **REJECTED**
- Proposals without deletion timeline: **REJECTED**

---

## Quick Reference Commands

```bash
# Check if proposed feature conflicts with existing code
python -m utils.codebase_analyzer --check-conflicts "add caching system"

# Find similar implementations
python -m utils.codebase_analyzer --find-similar "Logger"

# Get full planning context
python -m utils.codebase_analyzer --planning-context

# Full duplication report
python -m utils.codebase_analyzer --report --json
```

---

## Template for Upgrade Proposals

```markdown
# Upgrade Proposal: [Feature Name]

## 1. Conflict Analysis Results
- Risk Level: [HIGH/MEDIUM/LOW]
- Existing Implementations: [count]
- Related Categories: [list]

## 2. Existing Code Audit
| File | Component | Action | Justification |
|------|-----------|--------|---------------|
| ... | ... | EXTEND/DELETE/MERGE | ... |

## 3. Consolidation Actions
### Deletions
- [ ] `file_to_delete.py` - Reason

### Merges
- [ ] `file_a.py` + `file_b.py` → `merged.py`

### Extensions
- [ ] `existing.py` - Add new_method()

## 4. New Code Location
- Path: `module/submodule/file.py`
- Layer: [0-4]
- Justification: [why this location]

## 5. Migration Plan
- Step 1: ...
- Step 2: ...
- Deletion deadline: [date/sprint]

## 6. Verification
- [ ] Ran `--check-conflicts` (result attached)
- [ ] No parallel implementations created
- [ ] Old code has deletion timeline
```
