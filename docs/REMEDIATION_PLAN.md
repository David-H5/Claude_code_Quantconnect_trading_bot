# REMEDIATION PLAN: Post-Overnight System Audit
# Created: 2025-12-06
# Source: Comprehensive audit of overnight system execution

================================================================================
## OVERVIEW
================================================================================

This plan addresses ALL issues discovered during the audit of the overnight
system's execution of REFACTOR-001 and OVERNIGHT-002. Items are organized by
priority and include specific file paths, code changes, and verification steps.

**Total Issues**: 47 items across 6 categories
**Estimated Effort**: 2-3 focused sessions

================================================================================
## PHASE 0: CRITICAL BLOCKERS (Must Fix First)
================================================================================
Priority: IMMEDIATE - These block other work

### P0-1: Fix Module Import Chain (BLOCKING)
**Issue**: `utils/__init__.py` import fails due to missing `psutil`
**Impact**: Entire `utils/` module is broken, cannot test new utilities
**File**: `requirements.txt`

```bash
# Verification that it's broken:
python3 -c "from utils.overnight_state import OvernightStateManager"
# Expected: ModuleNotFoundError: No module named 'psutil'
```

**Fix**:
```bash
pip install psutil
# Or add to requirements.txt if not present
echo "psutil>=5.9.0" >> requirements.txt
```

**Verify**:
```bash
python3 -c "from utils.overnight_state import OvernightStateManager; print('OK')"
```

---

### P0-2: Create Missing Logging Directory
**Issue**: `.claude/logs/` directory doesn't exist
**Impact**: All hook logging silently fails (HOOK_LOG_FILE, ERROR_LOG_FILE, RECOVERY_LOG_FILE)
**File**: Directory structure

**Fix**:
```bash
mkdir -p /home/dshooter/projects/Claude_code_Quantconnect_trading_bot/.claude/logs
```

**Verify**:
```bash
ls -la .claude/logs/
# Should exist and be writable
```

---

### P0-3: Initialize Hook Log Files
**Issue**: Log files referenced in hook_utils.py don't exist
**Impact**: `log_hook_activity()`, `log_error()`, `create_recovery_point()` fail silently
**Files to create**:
- `.claude/logs/hook_activity.json`
- `.claude/logs/hook_errors.json`
- `.claude/logs/recovery_points.json`
- `logs/hook_activity_state.json`

**Fix**:
```bash
echo "[]" > .claude/logs/hook_activity.json
echo "[]" > .claude/logs/hook_errors.json
echo "[]" > .claude/logs/recovery_points.json
echo "{}" > logs/hook_activity_state.json
```

**Verify**:
```bash
python3 -c "
from pathlib import Path
import json
for f in ['.claude/logs/hook_activity.json', '.claude/logs/hook_errors.json',
          '.claude/logs/recovery_points.json', 'logs/hook_activity_state.json']:
    p = Path(f)
    assert p.exists(), f'{f} missing'
    json.loads(p.read_text())  # Valid JSON
print('All log files OK')
"
```

================================================================================
## PHASE 1: INTEGRATE NEW UTILITIES (P1 - High Priority)
================================================================================
Priority: HIGH - Files exist but aren't wired in

### P1-1: Integrate OvernightStateManager into session_stop.py
**Issue**: `utils/overnight_state.py` created but not used
**File**: `.claude/hooks/core/session_stop.py`

**Changes Required**:
1. Add import at top of file
2. Replace fragmented state access with unified manager
3. Update checkpoint creation to use new system

**Code to add** (near imports):
```python
from utils.overnight_state import OvernightStateManager, OvernightState
```

**Replace** state loading patterns:
```python
# OLD (fragmented):
# session_state = json.loads(Path('logs/session_state.json').read_text())
# hook_state = json.loads(Path('logs/hook_activity_state.json').read_text())

# NEW (unified):
state_manager = OvernightStateManager()
state = state_manager.load()
```

**Verify**:
```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from utils.overnight_state import OvernightStateManager
mgr = OvernightStateManager()
state = mgr.load()
print(f'State loaded: session_id={state.session_id}')
"
```

---

### P1-2: Integrate OvernightConfig into run_overnight.sh
**Issue**: `utils/overnight_config.py` and `config/overnight.yaml` created but not used
**Files**:
- `scripts/run_overnight.sh`
- `scripts/watchdog.py`

**Changes Required**:
1. Create Python wrapper script to load config
2. Source config values in bash script
3. Replace magic numbers with config values

**Create** `scripts/load_overnight_config.py`:
```python
#!/usr/bin/env python3
"""Load overnight config and export as environment variables."""
import sys
sys.path.insert(0, '.')
from utils.overnight_config import OvernightConfig

config = OvernightConfig.load()

# Export key values for bash
print(f"MAX_RUNTIME_HOURS={config.session.max_runtime_hours}")
print(f"MAX_IDLE_MINUTES={config.session.max_idle_minutes}")
print(f"MAX_COST_USD={config.budget.max_cost_usd}")
print(f"MAX_RESTARTS={config.recovery.max_restarts}")
```

**Add to run_overnight.sh** (near top):
```bash
# Load configuration from overnight.yaml
eval $(python3 scripts/load_overnight_config.py)
```

**Verify**:
```bash
python3 scripts/load_overnight_config.py
# Should output: MAX_RUNTIME_HOURS=10, etc.
```

---

### P1-3: Integrate ProgressParser into hook_utils.py
**Issue**: `utils/progress_parser.py` created but hook_utils.py still has duplicate parsing
**File**: `.claude/hooks/core/hook_utils.py`

**Changes Required**:
1. Add import
2. Replace `get_current_task()`, `get_pending_tasks()`, `get_completed_tasks()`
   to use the new parser

**Code to add** (near imports):
```python
from utils.progress_parser import ProgressParser
```

**Replace** existing functions (lines 283-336):
```python
def get_current_task() -> str | None:
    """Get the current task from progress file."""
    parser = ProgressParser()
    data = parser.parse_file(PROGRESS_FILE)
    pending = [t for t in data.all_tasks if t.status == "pending"]
    return pending[0].content if pending else None

def get_pending_tasks() -> list[str]:
    """Get all pending tasks from progress file."""
    parser = ProgressParser()
    data = parser.parse_file(PROGRESS_FILE)
    return [t.content for t in data.all_tasks if t.status == "pending"]

def get_completed_tasks() -> list[str]:
    """Get all completed tasks from progress file."""
    parser = ProgressParser()
    data = parser.parse_file(PROGRESS_FILE)
    return [t.content for t in data.all_tasks if t.status == "completed"]

def get_progress_stats() -> dict[str, int]:
    """Get progress statistics."""
    parser = ProgressParser()
    data = parser.parse_file(PROGRESS_FILE)
    return {
        "total": data.total_tasks,
        "completed": data.completed_tasks,
        "pending": data.pending_tasks,
        "completion_pct": int(data.completion_percentage),
    }
```

**Verify**:
```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from utils.progress_parser import ProgressParser
parser = ProgressParser()
data = parser.parse_file('claude-progress.txt')
print(f'Tasks: {data.total_tasks}, Complete: {data.completed_tasks}')
"
```

---

### P1-4: Integrate ProgressParser into auto-resume.sh
**Issue**: `scripts/auto-resume.sh` has duplicate progress parsing in bash
**File**: `scripts/auto-resume.sh`

**Changes Required**:
1. Create Python helper for progress stats
2. Replace bash parsing with Python call

**Create** `scripts/get_progress_stats.py`:
```python
#!/usr/bin/env python3
"""Get progress stats for bash scripts."""
import sys
sys.path.insert(0, '.')
from utils.progress_parser import ProgressParser

parser = ProgressParser()
data = parser.parse_file('claude-progress.txt')

print(f"TOTAL_TASKS={data.total_tasks}")
print(f"COMPLETED_TASKS={data.completed_tasks}")
print(f"PENDING_TASKS={data.pending_tasks}")
print(f"COMPLETION_PCT={int(data.completion_percentage)}")
```

**Replace** bash parsing in auto-resume.sh:
```bash
# OLD: grep/awk parsing of progress file
# NEW:
eval $(python3 scripts/get_progress_stats.py)
```

**Verify**:
```bash
python3 scripts/get_progress_stats.py
# Should output: TOTAL_TASKS=X, COMPLETED_TASKS=Y, etc.
```

================================================================================
## PHASE 2: CREATE MISSING FILES (P1 - High Priority)
================================================================================

### P2-1: Create docs/INDEX.md
**Issue**: Task 6.4 claimed complete but file doesn't exist
**File**: `docs/INDEX.md`

**Content to create**:
```markdown
# Documentation Index

## Architecture & Design
- [COMPREHENSIVE_REFACTOR_PLAN.md](COMPREHENSIVE_REFACTOR_PLAN.md) - Master refactoring plan
- [REFACTORING_INSTRUCTIONS.md](REFACTORING_INSTRUCTIONS.md) - Implementation guide
- [OVERNIGHT_SYSTEM_ANALYSIS.md](OVERNIGHT_SYSTEM_ANALYSIS.md) - Overnight system docs

## Configuration
- [../config/overnight.yaml](../config/overnight.yaml) - Overnight session config
- [../config/settings.json](../config/settings.json) - General settings

## API Reference
- [api/](../api/) - REST and WebSocket APIs
- [mcp/](../mcp/) - Model Context Protocol servers

## Models & Data Structures
- [../models/base_classes.py](../models/base_classes.py) - Base component classes
- [../models/risk_limits.py](../models/risk_limits.py) - Risk limit definitions

## Utilities
- [../utils/overnight_state.py](../utils/overnight_state.py) - Unified state manager
- [../utils/overnight_config.py](../utils/overnight_config.py) - Config loader
- [../utils/progress_parser.py](../utils/progress_parser.py) - Progress file parser

## Testing
- [../tests/test_p0_security_fixes.py](../tests/test_p0_security_fixes.py) - P0 security tests

## Overnight System
- [../scripts/run_overnight.sh](../scripts/run_overnight.sh) - Main launcher
- [../scripts/watchdog.py](../scripts/watchdog.py) - Safety monitor
- [../.claude/hooks/core/session_stop.py](../.claude/hooks/core/session_stop.py) - Stop hook
```

**Verify**:
```bash
test -f docs/INDEX.md && echo "INDEX.md exists"
```

---

### P2-2: Create scripts/health_check.py
**Issue**: Health Check API task incomplete
**File**: `scripts/health_check.py`

**Content to create**:
```python
#!/usr/bin/env python3
"""
Health Check HTTP Server for Overnight Sessions

Provides /health and /status endpoints for monitoring.
Part of OVERNIGHT-002 refactoring.
"""

import json
import sys
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.overnight_state import OvernightStateManager
from utils.overnight_config import OvernightConfig
from utils.progress_parser import ProgressParser


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoints."""

    def do_GET(self):
        if self.path == "/health":
            self.send_health()
        elif self.path == "/status":
            self.send_status()
        else:
            self.send_error(404)

    def send_health(self):
        """Simple health check - returns 200 if server is running."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }).encode())

    def send_status(self):
        """Detailed status with session info."""
        try:
            state_mgr = OvernightStateManager()
            state = state_mgr.load()

            parser = ProgressParser()
            progress = parser.parse_file(Path("claude-progress.txt"))

            status = {
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "session": {
                    "id": state.session_id,
                    "goal": state.goal,
                    "started_at": state.started_at,
                    "restart_count": state.restart_count,
                },
                "progress": {
                    "total_tasks": progress.total_tasks,
                    "completed": progress.completed_tasks,
                    "pending": progress.pending_tasks,
                    "completion_pct": progress.completion_percentage,
                },
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status, indent=2).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "error",
                "error": str(e)
            }).encode())

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"Health check server running on port {port}")
    print(f"  /health - Simple health check")
    print(f"  /status - Detailed session status")
    server.serve_forever()


if __name__ == "__main__":
    main()
```

**Verify**:
```bash
python3 scripts/health_check.py &
curl http://localhost:8765/health
# Should return: {"status": "healthy", ...}
kill %1
```

================================================================================
## PHASE 3: FIX SESSION TRACKING (P1 - High Priority)
================================================================================

### P3-1: Fix session-history.jsonl to Capture Test Results
**Issue**: All 37 sessions show `"tests_passed": null`
**File**: `.claude/hooks/core/session_stop.py`

**Changes Required**:
Find where session history is written and add test execution:

```python
def get_test_results() -> dict:
    """Run tests and capture results."""
    import subprocess
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "tests/", "-q", "--tb=no"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=Path(__file__).parent.parent.parent.parent
        )
        # Parse pytest output for pass/fail counts
        output = result.stdout
        # Example: "45 passed, 2 failed in 12.34s"
        import re
        match = re.search(r'(\d+) passed', output)
        passed = int(match.group(1)) if match else 0
        match = re.search(r'(\d+) failed', output)
        failed = int(match.group(1)) if match else 0
        return {
            "passed": passed,
            "failed": failed,
            "total": passed + failed,
            "exit_code": result.returncode
        }
    except Exception as e:
        return {"error": str(e)}
```

**Update session history entry**:
```python
# Where session_history is written, add:
entry["tests_passed"] = get_test_results()
entry["runtime_hours"] = calculate_runtime_hours()
```

**Verify**:
```bash
tail -1 logs/session-history.jsonl | python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
assert data.get('tests_passed') is not None, 'tests_passed still null'
print('Test tracking: OK')
"
```

---

### P3-2: Fix compaction-history.jsonl Checkpoint Creation
**Issue**: All 84 compactions show `"checkpoint_created": false`
**File**: `.claude/hooks/core/session_stop.py` (or wherever compaction is handled)

**Changes Required**:
Find the compaction logging and ensure checkpoint creation:

```python
def log_compaction(reason: str):
    """Log compaction with checkpoint creation."""
    checkpoint_created = False
    backup_path = None

    # Actually create checkpoint
    try:
        import subprocess
        result = subprocess.run(
            ["git", "stash", "push", "-m", f"compaction-checkpoint-{datetime.now().isoformat()}"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            checkpoint_created = True
            backup_path = f"stash@{{0}}"  # Git stash reference
    except Exception:
        pass

    entry = {
        "timestamp": datetime.now().isoformat(),
        "estimated_tokens": estimate_tokens(),  # Implement token estimation
        "compaction_reason": reason,
        "backup_path": backup_path,
        "checkpoint_created": checkpoint_created,
    }

    # Append to history
    history_file = Path("logs/compaction-history.jsonl")
    with history_file.open("a") as f:
        f.write(json.dumps(entry) + "\n")
```

**Verify**:
```bash
# After next compaction:
tail -1 logs/compaction-history.jsonl | python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
assert data.get('checkpoint_created') == True, 'checkpoint not created'
print('Checkpoint creation: OK')
"
```

---

### P3-3: Fix continuation_history.jsonl Updates
**Issue**: Last entry is Dec 3, no entries for Dec 6 work
**File**: `.claude/hooks/core/session_stop.py`

**Changes Required**:
Ensure continuation decisions are logged:

```python
def log_continuation_decision(decision: str, reason: str, stats: dict):
    """Log session continuation decision."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "decision": decision,  # "continue" or "stop"
        "reason": reason,
        "completion_pct": stats.get("completion_pct", 0),
        "p0": stats.get("p0", "unknown"),
        "p1": stats.get("p1", "unknown"),
        "p2": stats.get("p2", "unknown"),
        "pending_categories": stats.get("pending_categories", 0),
    }

    history_file = Path("logs/continuation_history.jsonl")
    with history_file.open("a") as f:
        f.write(json.dumps(entry) + "\n")
```

**Verify**:
```bash
wc -l logs/continuation_history.jsonl
# Should have more than 2 entries after remediation
```

================================================================================
## PHASE 4: RUN TESTS AND VERIFY (P1 - High Priority)
================================================================================

### P4-1: Run P0 Security Tests
**Issue**: Tests exist but were never executed
**File**: `tests/test_p0_security_fixes.py`

**Commands**:
```bash
cd /home/dshooter/projects/Claude_code_Quantconnect_trading_bot
python3 -m pytest tests/test_p0_security_fixes.py -v --tb=short
```

**Expected**: All tests pass
**If failures**: Document and fix each failing test

---

### P4-2: Run Full Test Suite
**Issue**: Session history shows tests never verified

**Commands**:
```bash
cd /home/dshooter/projects/Claude_code_Quantconnect_trading_bot
python3 -m pytest tests/ -v --tb=short -x  # Stop on first failure
```

**Document results** in `logs/test-results.json`:
```json
{
  "timestamp": "2025-12-06T...",
  "total": 3393,
  "passed": 3265,
  "failed": 116,
  "skipped": 12,
  "failures": [
    {"test": "test_name", "error": "error message"}
  ]
}
```

---

### P4-3: Verify Import Chain
**Issue**: Module imports may be broken

**Commands**:
```bash
python3 -c "
import sys
sys.path.insert(0, '.')

# Test all new modules
from utils.overnight_state import OvernightStateManager
from utils.overnight_config import OvernightConfig
from utils.progress_parser import ProgressParser
from models.base_classes import ExecutorBase, ScannerBase

print('All imports OK')
"
```

================================================================================
## PHASE 5: REMAINING OVERNIGHT-002 TASKS (P2 - Medium Priority)
================================================================================

### P5-1: Add Claude CLI Availability Check
**Issue**: auto-resume.sh:288 has no Claude CLI check
**File**: `scripts/auto-resume.sh`

**Add** near the top of auto-resume.sh:
```bash
# Check Claude CLI availability
check_claude_cli() {
    if ! command -v claude &> /dev/null; then
        echo "ERROR: Claude CLI not found in PATH"
        echo "Install with: npm install -g @anthropic-ai/claude-code"
        exit 1
    fi

    # Verify it works
    if ! claude --version &> /dev/null; then
        echo "ERROR: Claude CLI installed but not responding"
        exit 1
    fi

    echo "Claude CLI: $(claude --version)"
}

check_claude_cli
```

---

### P5-2: Add Tests for State Consolidation
**File**: `tests/test_overnight_state.py` (NEW)

**Content**:
```python
"""Tests for unified overnight state management."""
import pytest
from pathlib import Path
from utils.overnight_state import OvernightStateManager, OvernightState


class TestOvernightStateManager:
    def test_load_creates_default_state(self, tmp_path):
        mgr = OvernightStateManager(state_file=tmp_path / "state.json")
        state = mgr.load()
        assert isinstance(state, OvernightState)
        assert state.session_id == ""

    def test_save_and_load_roundtrip(self, tmp_path):
        mgr = OvernightStateManager(state_file=tmp_path / "state.json")
        state = OvernightState(session_id="test-123", goal="Test goal")
        mgr.save(state)
        loaded = mgr.load()
        assert loaded.session_id == "test-123"
        assert loaded.goal == "Test goal"

    def test_file_locking(self, tmp_path):
        """Verify concurrent access is safe."""
        import threading
        mgr = OvernightStateManager(state_file=tmp_path / "state.json")
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    state = mgr.load()
                    state.restart_count = n * 100 + i
                    mgr.save(state)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent access errors: {errors}"
```

---

### P5-3: Add Tests for Configuration Loading
**File**: `tests/test_overnight_config.py` (NEW)

**Content**:
```python
"""Tests for overnight configuration system."""
import pytest
import os
from pathlib import Path
from utils.overnight_config import OvernightConfig


class TestOvernightConfig:
    def test_load_default_config(self):
        config = OvernightConfig.load()
        assert config.session.max_runtime_hours == 10
        assert config.budget.max_cost_usd == 50.0

    def test_environment_variable_expansion(self, monkeypatch):
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/webhook")
        config = OvernightConfig.load()
        assert config.notifications.discord_webhook == "https://discord.test/webhook"

    def test_missing_env_var_returns_empty(self, monkeypatch):
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        config = OvernightConfig.load()
        assert config.notifications.discord_webhook == ""
```

---

### P5-4: Add Tests for Progress Parser
**File**: `tests/test_progress_parser.py` (NEW)

**Content**:
```python
"""Tests for progress file parser."""
import pytest
from pathlib import Path
from utils.progress_parser import ProgressParser, ProgressData, Task


class TestProgressParser:
    def test_parse_completed_task(self, tmp_path):
        progress_file = tmp_path / "progress.txt"
        progress_file.write_text("- [x] **1.1** Complete task one\n")

        parser = ProgressParser()
        data = parser.parse_file(progress_file)

        assert data.completed_tasks == 1
        assert data.pending_tasks == 0

    def test_parse_pending_task(self, tmp_path):
        progress_file = tmp_path / "progress.txt"
        progress_file.write_text("- [ ] **1.2** Pending task\n")

        parser = ProgressParser()
        data = parser.parse_file(progress_file)

        assert data.completed_tasks == 0
        assert data.pending_tasks == 1

    def test_completion_percentage(self, tmp_path):
        progress_file = tmp_path / "progress.txt"
        progress_file.write_text(
            "- [x] Task 1\n"
            "- [x] Task 2\n"
            "- [ ] Task 3\n"
            "- [ ] Task 4\n"
        )

        parser = ProgressParser()
        data = parser.parse_file(progress_file)

        assert data.completion_percentage == 50.0
```

================================================================================
## PHASE 6: DOCUMENTATION UPDATES (P2 - Medium Priority)
================================================================================

### P6-1: Update claude-progress.txt with Accurate Status
**Issue**: Progress file claims false completions
**File**: `claude-progress.txt`

**Changes Required**:
Mark incomplete items as incomplete:
```
- [ ] Migrate existing code to use unified state manager
- [ ] Add tests for state consolidation
- [ ] Replace magic numbers across overnight scripts
- [ ] Add tests for configuration loading
- [ ] Replace duplicate parsing code in session_stop.py
- [ ] Replace duplicate parsing code in hook_utils.py
- [ ] Replace duplicate parsing code in auto-resume.sh
- [ ] Add tests for progress parser
- [ ] Add Claude CLI availability check in auto-resume.sh
- [ ] Create scripts/health_check.py
- [ ] Create docs/INDEX.md
```

---

### P6-2: Update claude-session-notes.md
**Issue**: Notes don't reflect actual state
**File**: `claude-session-notes.md`

**Add section**:
```markdown
## REMEDIATION-001: Post-Audit Fixes
### Issues Discovered
- Module import chain broken (psutil missing)
- Logging directory missing (.claude/logs/)
- New utilities not integrated
- Tests never executed
- Session tracking incomplete

### Fixes Applied
- [List fixes as they are applied]
```

================================================================================
## VERIFICATION CHECKLIST
================================================================================

After completing all phases, verify:

```bash
#!/bin/bash
# remediation-verify.sh

echo "=== VERIFICATION CHECKLIST ==="

# P0 Blockers
echo -n "P0-1 psutil installed: "
python3 -c "import psutil; print('OK')" 2>/dev/null || echo "FAIL"

echo -n "P0-2 .claude/logs/ exists: "
test -d .claude/logs && echo "OK" || echo "FAIL"

echo -n "P0-3 Log files exist: "
test -f .claude/logs/hook_activity.json && echo "OK" || echo "FAIL"

# P1 Integration
echo -n "P1-1 overnight_state imports: "
python3 -c "from utils.overnight_state import OvernightStateManager; print('OK')" 2>/dev/null || echo "FAIL"

echo -n "P1-2 overnight_config imports: "
python3 -c "from utils.overnight_config import OvernightConfig; print('OK')" 2>/dev/null || echo "FAIL"

echo -n "P1-3 progress_parser imports: "
python3 -c "from utils.progress_parser import ProgressParser; print('OK')" 2>/dev/null || echo "FAIL"

# P2 Missing Files
echo -n "P2-1 docs/INDEX.md exists: "
test -f docs/INDEX.md && echo "OK" || echo "FAIL"

echo -n "P2-2 health_check.py exists: "
test -f scripts/health_check.py && echo "OK" || echo "FAIL"

# P4 Tests
echo -n "P4-1 P0 tests pass: "
python3 -m pytest tests/test_p0_security_fixes.py -q 2>/dev/null && echo "OK" || echo "FAIL"

echo "=== END VERIFICATION ==="
```

================================================================================
## EXECUTION ORDER
================================================================================

1. **PHASE 0** (5 min): Fix blockers first
   - P0-1: Install psutil
   - P0-2: Create .claude/logs/
   - P0-3: Initialize log files

2. **PHASE 2** (10 min): Create missing files
   - P2-1: docs/INDEX.md
   - P2-2: scripts/health_check.py

3. **PHASE 4** (15 min): Run tests
   - P4-1: P0 security tests
   - P4-3: Verify imports

4. **PHASE 1** (30 min): Integrate utilities
   - P1-1: session_stop.py integration
   - P1-2: run_overnight.sh integration
   - P1-3: hook_utils.py integration
   - P1-4: auto-resume.sh integration

5. **PHASE 3** (20 min): Fix tracking
   - P3-1: Test result capture
   - P3-2: Checkpoint creation
   - P3-3: Continuation logging

6. **PHASE 5** (30 min): Add tests
   - P5-1: Claude CLI check
   - P5-2: State tests
   - P5-3: Config tests
   - P5-4: Parser tests

7. **PHASE 6** (10 min): Documentation
   - P6-1: Fix progress file
   - P6-2: Update notes

================================================================================
## SUCCESS CRITERIA
================================================================================

All items pass verification:
- [ ] All imports work without errors
- [ ] All log files exist and are valid JSON
- [ ] All tests pass (or known failures documented)
- [ ] Session tracking captures test results
- [ ] Compaction creates checkpoints
- [ ] New utilities are actively used by overnight scripts

================================================================================
## PHASE 7: OVERNIGHT AGENT LOG FINDINGS (P1 - High Priority)
================================================================================
# Added from secondary analysis of overnight agent chat logs

### P7-1: Fix datetime.utcnow() Deprecation (71 files)
**Issue**: Python 3.12+ deprecated `datetime.utcnow()` - generates 54+ warnings
**Scope**: 71 files across the codebase
**Impact**: Test output cluttered, future Python versions will break

**Files affected** (sample):
- `llm/decision_logger.py`
- `llm/agents/supervisor.py`
- `evaluation/feedback_loop.py`
- `observability/logging/audit.py`
- `compliance/*.py`
- `scripts/notify.py`
- `.claude/hooks/trading/*.py`

**Fix pattern**:
```python
# OLD (deprecated):
from datetime import datetime
timestamp = datetime.utcnow()

# NEW (correct):
from datetime import datetime, timezone
timestamp = datetime.now(timezone.utc)
```

**Bulk fix command**:
```bash
# Find and replace pattern
find . -name "*.py" -not -path "./.venv/*" -exec \
  sed -i 's/datetime\.utcnow()/datetime.now(timezone.utc)/g' {} \;

# Add timezone import where missing
# (Manual review required for each file)
```

**Verify**:
```bash
grep -r "datetime.utcnow()" --include="*.py" . | grep -v ".venv" | wc -l
# Should be 0
```

---

### P7-2: Migrate Pydantic V1 Validators to V2 (mcp/schemas.py)
**Issue**: 8 uses of deprecated `@validator` in `mcp/schemas.py`
**Impact**: PydanticDeprecatedSince20 warnings, will break in Pydantic V3

**Locations** (line numbers):
- Line 139: `@validator("symbol")`
- Line 196: `@validator("underlying")`
- Line 235: `@validator("symbol")`
- Line 239: `@validator("end_date")`
- Line 283: `@validator("symbol")`
- Line 287: `@validator("limit_price", always=True)`
- Line 294: `@validator("stop_price", always=True)`
- Line 462: `@validator("end_date")`

**Fix pattern**:
```python
# OLD (Pydantic V1):
from pydantic import validator

@validator("symbol")
def validate_symbol(cls, v):
    ...

# NEW (Pydantic V2):
from pydantic import field_validator

@field_validator("symbol")
@classmethod
def validate_symbol(cls, v):
    ...
```

**File**: `mcp/schemas.py`

**Verify**:
```bash
grep -n "@validator" mcp/schemas.py
# Should return no results
```

---

### P7-3: Add TRADING_API_TOKEN to Test Fixtures
**Issue**: 30+ tests fail with `OSError: TRADING_API_TOKEN environment variable is required`
**File**: `tests/conftest.py`

**Tests affected**:
- `tests/test_integration.py` (8 tests)
- `tests/test_order_queue_api.py` (15+ tests)

**Fix - Add to conftest.py**:
```python
import os
import secrets

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up required environment variables for tests."""
    # Generate a valid test token if not set
    if "TRADING_API_TOKEN" not in os.environ:
        os.environ["TRADING_API_TOKEN"] = secrets.token_urlsafe(32)

    yield

    # Cleanup is optional since tests may need the token throughout

# Alternative: session-scoped for efficiency
@pytest.fixture(scope="session", autouse=True)
def setup_trading_token():
    """Set TRADING_API_TOKEN for entire test session."""
    original = os.environ.get("TRADING_API_TOKEN")
    os.environ["TRADING_API_TOKEN"] = secrets.token_urlsafe(32)
    yield
    if original:
        os.environ["TRADING_API_TOKEN"] = original
    else:
        os.environ.pop("TRADING_API_TOKEN", None)
```

**Verify**:
```bash
python3 -m pytest tests/test_order_queue_api.py -v --tb=short 2>&1 | grep -c "TRADING_API_TOKEN"
# Should be 0 (no errors about missing token)
```

---

### P7-4: Fix Remaining Test API Signature Mismatches
**Issue**: Overnight agent fixed some tests but several still fail
**Files with failures**:

| File | Issue | Fix Required |
|------|-------|--------------|
| `tests/test_rest_api.py` | `AttributeError: 'NoneType' object has no attribute 'get_or_none'` | Mock algorithm properly |
| `tests/test_retraining.py` | `DataMissingError: Missing required field: job_id` | Update test assertions |
| `tests/test_rl_rebalancer.py` | `AttributeError: 'ValueNetwork' object has no attribute 'get_parameters'` | Update mock or API |
| `tests/test_simulation.py` | Reproducibility assertion failure | Fix seed handling |
| `tests/test_tgarch.py` | Model fitting errors | Update test data requirements |

**Approach for each**:
1. Read the test file
2. Read the implementation it's testing
3. Update test mocks/assertions to match current API

---

### P7-5: Verify Hook Directory Reorganization
**Issue**: Hooks moved from `.claude/hooks/` to subdirectories
**New structure**:
```
.claude/hooks/
├── core/           # Core hooks (session_stop.py, hook_utils.py)
├── trading/        # Trading hooks (risk_validator.py, log_trade.py, etc.)
└── validation/     # Validation hooks (algo_change_guard.py, etc.)
```

**Files moved**:
- `.claude/hooks/trading/risk_validator.py`
- `.claude/hooks/trading/log_trade.py`
- `.claude/hooks/trading/load_trading_context.py`
- `.claude/hooks/trading/parse_backtest.py`
- `.claude/hooks/validation/algo_change_guard.py`

**Tests updated by overnight agent**:
- `tests/hooks/test_log_trade.py` ✓
- `tests/hooks/test_risk_validator.py` ✓
- `integration_tests/test_safety_hooks.py` ✓

**Verify all references updated**:
```bash
# Check for old paths still referenced
grep -r "\.claude/hooks/risk_validator" --include="*.py" . | grep -v ".venv"
grep -r "\.claude/hooks/log_trade" --include="*.py" . | grep -v ".venv"
# Should return no results
```

---

### P7-6: Fix Type Annotation Error (callable vs Callable)
**Issue**: `evaluation/classic_evaluation.py:613` used `callable | None` instead of `Callable | None`
**Status**: FIXED by overnight agent (verified - grep returns no matches)

**Verification**:
```bash
grep -r "callable |" --include="*.py" . | grep -v ".venv" | grep -v "Callable"
# Should return no results
```

---

### P7-7: Linting Issues Partially Fixed
**Issue**: ruff found 5828 errors, overnight agent auto-fixed 5158
**Remaining**: 828 linting issues

**Check current state**:
```bash
cd /home/dshooter/projects/Claude_code_Quantconnect_trading_bot
source .venv/bin/activate
ruff check . --statistics
```

**Categories likely remaining**:
- Import sorting (isort)
- Deprecated dict type hints
- Unsorted `__all__`
- Line length violations

**Fix**:
```bash
ruff check . --fix --unsafe-fixes
```

================================================================================
## UPDATED EXECUTION ORDER
================================================================================

1. **PHASE 0** (5 min): Fix blockers
   - P0-1: Install psutil
   - P0-2: Create .claude/logs/
   - P0-3: Initialize log files

2. **PHASE 7.3** (5 min): Add TRADING_API_TOKEN to conftest.py
   - Unblocks 30+ integration tests

3. **PHASE 2** (10 min): Create missing files
   - P2-1: docs/INDEX.md
   - P2-2: scripts/health_check.py

4. **PHASE 7.1** (30 min): Fix datetime.utcnow() deprecation
   - 71 files to update
   - Can be partially automated

5. **PHASE 7.2** (15 min): Migrate Pydantic validators
   - 8 validators in mcp/schemas.py

6. **PHASE 4** (15 min): Run tests
   - P4-1: P0 security tests
   - P4-3: Verify imports

7. **PHASE 7.4** (30 min): Fix remaining test failures
   - 5 test files with API mismatches

8. **PHASE 1** (30 min): Integrate utilities
   - P1-1 through P1-4

9. **PHASE 3** (20 min): Fix tracking
   - P3-1 through P3-3

10. **PHASE 7.7** (10 min): Final linting pass
    - Clean up remaining 828 issues

11. **PHASE 5** (30 min): Add tests
    - P5-1 through P5-4

12. **PHASE 6** (10 min): Documentation
    - P6-1, P6-2

================================================================================
## UPDATED SUMMARY
================================================================================

**Original Issues**: 47 items
**New Issues from Overnight Agent Logs**: 7 items
**Total Issues**: 54 items

| Phase | Priority | Items | Description |
|-------|----------|-------|-------------|
| P0 | CRITICAL | 3 | Blockers (psutil, dirs, logs) |
| P1 | HIGH | 4 | Integrate new utilities |
| P2 | HIGH | 2 | Create missing files |
| P3 | HIGH | 3 | Fix session tracking |
| P4 | HIGH | 3 | Run and verify tests |
| P5 | MEDIUM | 4 | Add tests for new utilities |
| P6 | MEDIUM | 2 | Documentation updates |
| **P7** | **HIGH** | **7** | **Overnight agent log findings** |

**Estimated Total Effort**: 3-4 focused sessions
