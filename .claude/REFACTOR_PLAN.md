# Master Refactoring Plan

**Created:** 2025-12-05
**Updated:** 2025-12-05
**Status:** All Phases Complete ✅

---

## Progress Summary

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | P0 Critical Fixes | ✅ COMPLETE (commit 09324fb) |
| Phase 2 | Hook Organization | ✅ COMPLETE |
| Phase 3 | State File Organization | ✅ COMPLETE (commit 610622e) |
| Phase 4 | Agent Architecture Cleanup | ✅ COMPLETE (already done) |
| Phase 5 | Documentation Cleanup | ✅ COMPLETE (already done) |
| Phase 6 | Final Cleanup | ✅ COMPLETE |

---

## Quick Start

```bash
# To execute the next phase, tell Claude:
"Read .claude/REFACTOR_PLAN.md and execute Phase 6"
```

---

## Overview

This document contains the complete refactoring plan for cleaning up the project. Execute phases in order, committing after each phase.

### Problems Being Fixed
1. **P0:** 3 duplicate session state implementations
2. **P0:** 2 duplicate audit logger implementations
3. **P1:** 2 incompatible agent architectures
4. **P1:** 27 scattered hooks needing organization
5. **P2:** Mixed trading infrastructure and dev tools
6. **P2:** Versioned filenames in documentation

---

## Phase 1: P0 Critical Fixes ✅ COMPLETE

> **Completed:** 2025-12-05 (commit 09324fb)
> - Session state management consolidated to `hook_utils.py` with separate state file
> - Duplicate audit logger removed (`models/audit_logger.py` deleted)
> - All imports updated

### 1.1 Consolidate Session State Management

**Problem:** Three files manage session state differently:
- `scripts/session_state_manager.py` → uses `logs/session_state.json`
- `.claude/hooks/session_stop.py` → uses `logs/session_state.json`
- `.claude/hooks/hook_utils.py` → uses `session_state.json` (different path!)

**Action:**
```bash
# Step 1: Make session_state_manager.py the single source of truth
# Step 2: Update hook_utils.py to import from session_state_manager
# Step 3: Update session_stop.py to import from session_state_manager
# Step 4: Ensure all use logs/session_state.json
```

**Files to modify:**
1. `.claude/hooks/hook_utils.py` - Remove duplicate state management, import from scripts/
2. `.claude/hooks/session_stop.py` - Remove duplicate state management, import from scripts/

**Verification:**
```bash
grep -r "session_state.json" .claude/hooks/ scripts/
# Should show only ONE implementation in scripts/session_state_manager.py
```

---

### 1.2 Remove Duplicate Audit Logger

**Problem:** Two identical implementations:
- `compliance/audit_logger.py` (KEEP)
- `models/audit_logger.py` (DELETE)

**Action:**
```bash
# Step 1: Verify compliance/audit_logger.py is complete
# Step 2: Find all imports of models/audit_logger.py
grep -r "from models.audit_logger\|from models import audit_logger" --include="*.py"

# Step 3: Update imports to use compliance/audit_logger.py
# Step 4: Delete models/audit_logger.py
rm models/audit_logger.py

# Step 5: Update models/__init__.py if needed
```

**Files to modify:**
1. Any file importing from `models.audit_logger` → change to `compliance.audit_logger`
2. Delete `models/audit_logger.py`

**Verification:**
```bash
python3 -c "from compliance.audit_logger import AuditLogger; print('OK')"
```

---

### 1.3 Commit Phase 1

```bash
git add -A
git commit -m "refactor: P0 fixes - consolidate session state and audit logger

- Consolidated session state management to scripts/session_state_manager.py
- Removed duplicate audit logger from models/
- Updated all imports to use single implementations

🤖 Generated with Claude Code"
```

---

## Phase 2: Hook Organization (Option A) ✅ COMPLETE

> **Completed:** 2025-12-05
> - Created subdirectories: `core/`, `validation/`, `research/`, `trading/`, `formatting/`, `agents/`
> - Moved all 27 hooks to appropriate categories
> - Updated `settings.json` with new paths
> - Updated `registry.json` with new structure
> - Internal imports updated

### 2.1 Create Subdirectory Structure

```bash
cd .claude/hooks

# Create subdirectories
mkdir -p core validation research trading formatting agents

# Keep deprecated and __pycache__ where they are
```

### 2.2 Move Hooks to Subdirectories

**Core hooks (critical functionality):**
```bash
mv ric.py core/
mv session_stop.py core/
mv pre_compact.py core/
mv protect_files.py core/
mv hook_utils.py core/
```

**Validation hooks:**
```bash
mv validate_algorithm.py validation/
mv validate_research.py validation/
mv validate_category_docs.py validation/
mv qa_auto_check.py validation/
mv algo_change_guard.py validation/
```

**Research hooks:**
```bash
mv document_research.py research/
mv research_saver.py research/
mv thorough_research.py research/
mv auto_research_trigger.py research/
```

**Trading hooks:**
```bash
mv risk_validator.py trading/
mv log_trade.py trading/
mv parse_backtest.py trading/
mv load_trading_context.py trading/
```

**Formatting hooks:**
```bash
mv auto_format.py formatting/
mv update_cross_refs.py formatting/
mv template_detector.py formatting/
```

**Agent hooks:**
```bash
mv agent_orchestrator.py agents/
mv multi_agent.py agents/
```

### 2.3 Update settings.json Paths

**Find and replace all hook paths:**

| Old Path | New Path |
|----------|----------|
| `hooks/ric.py` | `hooks/core/ric.py` |
| `hooks/session_stop.py` | `hooks/core/session_stop.py` |
| `hooks/pre_compact.py` | `hooks/core/pre_compact.py` |
| `hooks/protect_files.py` | `hooks/core/protect_files.py` |
| `hooks/validate_algorithm.py` | `hooks/validation/validate_algorithm.py` |
| `hooks/validate_research.py` | `hooks/validation/validate_research.py` |
| `hooks/validate_category_docs.py` | `hooks/validation/validate_category_docs.py` |
| `hooks/qa_auto_check.py` | `hooks/validation/qa_auto_check.py` |
| `hooks/algo_change_guard.py` | `hooks/validation/algo_change_guard.py` |
| `hooks/document_research.py` | `hooks/research/document_research.py` |
| `hooks/auto_format.py` | `hooks/formatting/auto_format.py` |
| `hooks/risk_validator.py` | `hooks/trading/risk_validator.py` |
| `hooks/log_trade.py` | `hooks/trading/log_trade.py` |
| `hooks/parse_backtest.py` | `hooks/trading/parse_backtest.py` |
| `hooks/load_trading_context.py` | `hooks/trading/load_trading_context.py` |

### 2.4 Update registry.json Paths

Same path updates as settings.json, in the `hooks` section.

### 2.5 Update Internal Imports

Any hook that imports from another hook needs path updates:
```python
# Old
from hook_utils import some_function

# New
from .core.hook_utils import some_function
# OR
import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from core.hook_utils import some_function
```

### 2.6 Verify Phase 2

```bash
# Test each hook can be imported
python3 -c "import sys; sys.path.insert(0, '.claude/hooks'); from core.ric import main"
python3 -c "import sys; sys.path.insert(0, '.claude/hooks'); from core.session_stop import main"
# ... etc

# Validate JSON files
python3 -c "import json; json.load(open('.claude/settings.json'))"
python3 -c "import json; json.load(open('.claude/registry.json'))"
```

### 2.7 Commit Phase 2

```bash
git add -A
git commit -m "refactor: Organize hooks into subdirectories

- Created core/, validation/, research/, trading/, formatting/, agents/ subdirectories
- Moved 27 hooks to appropriate categories
- Updated settings.json and registry.json with new paths
- Updated internal imports

🤖 Generated with Claude Code"
```

---

## Phase 3: State File Organization ✅ COMPLETE

> **Completed:** 2025-12-05 (commit 610622e)
> - State directory `.claude/state/` contains: ric.json, doc_updates.json, circuit_breaker.json, autonomous_issues.json
> - Config directory `.claude/config/` contains: agents.json
> - Updated scripts/watchdog.py to use new paths
> - Removed stale .claude/ric_state.json duplicate

### 3.1 Create State Directory

```bash
mkdir -p .claude/state
mkdir -p .claude/config
```

### 3.2 Move State Files

```bash
# Move state files
mv .claude/ric_state.json .claude/state/ric.json
mv .claude/research_state.json .claude/state/research.json 2>/dev/null || true
mv .claude/ric_doc_updates.json .claude/state/doc_updates.json

# Move config files
mv .claude/agent_config.json .claude/config/agents.json
# settings.json stays in .claude/ (Claude Code expects it there)
```

### 3.3 Update File References

Search and update all references:
```bash
grep -r "ric_state.json" --include="*.py" --include="*.md"
grep -r "agent_config.json" --include="*.py" --include="*.md"
```

Update paths in:
- `.claude/hooks/core/ric.py`
- Any scripts referencing these files

### 3.4 Commit Phase 3

```bash
git add -A
git commit -m "refactor: Organize state and config files

- Created .claude/state/ for runtime state files
- Created .claude/config/ for configuration files
- Updated all file references

🤖 Generated with Claude Code"
```

---

## Phase 4: Agent Architecture Cleanup ✅ COMPLETE

> **Status:** Already completed in a prior session
> - `agents/` directory was renamed to `agents_deprecated/`
> - `llm/agents/` is the primary agent implementation
> - No imports from `agents.` remain in the codebase

### 4.1 Deprecate `agents/` Directory

The `agents/` directory uses an older, simpler agent pattern. The `llm/agents/` directory has the more sophisticated implementation.

**Action:**
```bash
# Move agents/ to agents_deprecated/
mv agents agents_deprecated

# Update any imports (there may be few)
grep -r "from agents\." --include="*.py" | grep -v agents_deprecated
```

### 4.2 Update Imports

Any file importing from `agents.` should be updated to use `llm.agents.` or have functionality moved.

### 4.3 Commit Phase 4

```bash
git add -A
git commit -m "refactor: Deprecate old agents/ directory

- Moved agents/ to agents_deprecated/
- llm/agents/ is now the primary agent implementation
- Updated imports as needed

🤖 Generated with Claude Code"
```

---

## Phase 5: Documentation Cleanup ✅ COMPLETE

> **Status:** Already completed in a prior session
> - `docs/prompts/versions/` contains v4, v5, v6 frameworks
> - `docs/prompts/FRAMEWORK.md` is the current version (v6)
> - `docs/prompts/README.md` has comprehensive version index

### 5.1 Organize Versioned Documentation

```bash
mkdir -p docs/prompts/versions

# Move versioned prompt frameworks
mv docs/PROMPT_V4_FRAMEWORK.md docs/prompts/versions/v4_framework.md
mv docs/PROMPT_V5_FRAMEWORK.md docs/prompts/versions/v5_framework.md
mv docs/PROMPT_V6_FRAMEWORK.md docs/prompts/versions/v6_framework.md

# Create current version pointer
cp docs/prompts/versions/v6_framework.md docs/prompts/FRAMEWORK.md
```

### 5.2 Create Version Index

Create `docs/prompts/README.md`:
```markdown
# Prompt Framework

Current version: v6

## Current Framework
See [FRAMEWORK.md](FRAMEWORK.md)

## Version History
- [v6](versions/v6_framework.md) - Current
- [v5](versions/v5_framework.md) - Previous
- [v4](versions/v4_framework.md) - Legacy
```

### 5.3 Commit Phase 5

```bash
git add -A
git commit -m "refactor: Organize versioned documentation

- Moved PROMPT_V4/V5/V6_FRAMEWORK.md to docs/prompts/versions/
- Created docs/prompts/FRAMEWORK.md as current version
- Added version index

🤖 Generated with Claude Code"
```

---

## Phase 6: Final Cleanup ✅ COMPLETE

> **Completed:** 2025-12-05
> - No unnecessary empty directories found
> - Updated CLAUDE.md with correct hook paths after reorganization
> - All Python files compile successfully
> - All JSON files valid
> - RIC hook works correctly
> - All 40 tests pass

### 6.1 Remove Empty Directories

```bash
rmdir agents_deprecated 2>/dev/null || true
find . -type d -empty -delete
```

### 6.2 Update CLAUDE.md

Add section about new structure:
```markdown
## Project Structure

### Claude Code Tools (.claude/)
- `hooks/core/` - Critical hooks (RIC, session, protection)
- `hooks/validation/` - Code and doc validators
- `hooks/research/` - Research tracking hooks
- `hooks/trading/` - Trading-specific hooks
- `hooks/formatting/` - Code formatting hooks
- `hooks/agents/` - Agent orchestration hooks
- `state/` - Runtime state files
- `config/` - Configuration files
- `commands/` - Slash commands
- `templates/` - Document templates

### Trading Infrastructure
- `algorithms/` - Trading algorithms
- `llm/agents/` - LLM trading agents
- `execution/` - Order execution
- `mcp/` - MCP servers
- `models/` - Trading models
```

### 6.3 Final Verification

```bash
# Validate all Python files
find .claude/hooks -name "*.py" -exec python3 -m py_compile {} \;

# Validate JSON files
python3 -c "import json; json.load(open('.claude/settings.json')); json.load(open('.claude/registry.json')); print('✅ All JSON valid')"

# Test RIC hook
python3 .claude/hooks/core/ric.py status

# Run tests
pytest tests/ -x -q
```

### 6.4 Final Commit

```bash
git add -A
git commit -m "refactor: Complete project reorganization

Completed refactoring phases:
- P0: Consolidated session state and audit logger
- P1: Organized 27 hooks into 6 categories
- P1: Deprecated old agents/ directory
- P2: Organized state and config files
- P2: Cleaned up versioned documentation

See .claude/REFACTOR_PLAN.md for details.

🤖 Generated with Claude Code"
```

---

## Post-Refactor Checklist

- [ ] All hooks work correctly
- [ ] settings.json paths are correct
- [ ] registry.json paths are correct
- [ ] RIC loop functions (`python3 .claude/hooks/core/ric.py status`)
- [ ] Session stop works
- [ ] Tests pass
- [ ] No import errors in any Python file

---

## Rollback Plan

If something breaks:

```bash
# Revert to before refactoring
git log --oneline -10  # Find commit before refactoring
git revert HEAD~N..HEAD  # Revert N commits

# Or hard reset (destructive)
git reset --hard <commit-before-refactoring>
```

---

## File Structure After Refactoring

```
.claude/
├── hooks/
│   ├── core/
│   │   ├── ric.py
│   │   ├── session_stop.py
│   │   ├── pre_compact.py
│   │   ├── protect_files.py
│   │   └── hook_utils.py
│   ├── validation/
│   │   ├── validate_algorithm.py
│   │   ├── validate_research.py
│   │   ├── validate_category_docs.py
│   │   ├── qa_auto_check.py
│   │   └── algo_change_guard.py
│   ├── research/
│   │   ├── document_research.py
│   │   ├── research_saver.py
│   │   ├── thorough_research.py
│   │   └── auto_research_trigger.py
│   ├── trading/
│   │   ├── risk_validator.py
│   │   ├── log_trade.py
│   │   ├── parse_backtest.py
│   │   └── load_trading_context.py
│   ├── formatting/
│   │   ├── auto_format.py
│   │   ├── update_cross_refs.py
│   │   └── template_detector.py
│   ├── agents/
│   │   ├── agent_orchestrator.py
│   │   └── multi_agent.py
│   └── deprecated/
│       └── ... (existing deprecated files)
├── state/
│   ├── ric.json
│   ├── session.json
│   └── research.json
├── config/
│   └── agents.json
├── commands/
│   └── ... (unchanged)
├── templates/
│   └── ... (unchanged)
├── settings.json
├── registry.json
├── AUDIT_REPORT.md
└── REFACTOR_PLAN.md
```

---

## Notes

1. **Execute phases in order** - Each phase depends on the previous
2. **Commit after each phase** - Allows easy rollback
3. **Test after each phase** - Catch issues early
4. **Don't skip verification steps** - They catch path issues

---

## Quick Commands

```bash
# Check current phase status
git log --oneline -5

# Validate hooks
find .claude/hooks -name "*.py" -exec python3 -m py_compile {} \;

# Test RIC
python3 .claude/hooks/core/ric.py status  # After Phase 2
python3 .claude/hooks/ric.py status       # Before Phase 2

# Validate JSON
python3 -c "import json; json.load(open('.claude/settings.json'))"
```
