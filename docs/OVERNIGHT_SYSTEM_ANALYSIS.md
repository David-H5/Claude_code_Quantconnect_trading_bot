# Overnight System: Refactoring & Function Analysis

*Version: 1.0.0*
*Generated: 2025-12-06*
*Scope: Autonomous overnight development infrastructure*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Component Function Analysis](#3-component-function-analysis)
4. [Code Duplications](#4-code-duplications)
5. [Identified Flaws](#5-identified-flaws)
6. [Refactoring Recommendations](#6-refactoring-recommendations)
7. [Best Practices](#7-best-practices)

---

## 1. Executive Summary

The overnight system is an **autonomous development infrastructure** enabling Claude Code to run unattended for 8-10+ hours. It combines session management, safety monitoring, crash recovery, and context persistence.

### System Stats

| Metric | Value |
|--------|-------|
| **Core Scripts** | 6 (Bash + Python) |
| **Total Lines** | ~3,500 |
| **Functions** | 85+ |
| **State Files** | 5 |
| **Hooks** | 3 |

### Current Autonomy Score: **8/10**

**Strengths:**
- Robust crash recovery with exponential backoff
- Compaction-proof state persistence
- Multi-channel notifications (Discord/Slack)
- RIC Loop integration for methodical work
- Priority-aware task completion enforcement

**Weaknesses:**
- State file fragmentation (5 separate files)
- Duplicate code between Bash and Python
- Missing centralized configuration
- No health check API endpoint
- Race conditions in state updates

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    OVERNIGHT SYSTEM                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ run_overnight.sh│───▶│   Claude Code   │                     │
│  │   (Launcher)    │    │    Session      │                     │
│  └────────┬────────┘    └────────┬────────┘                     │
│           │                      │                               │
│           ▼                      ▼                               │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │   watchdog.py   │    │ session_stop.py │                     │
│  │   (Monitor)     │    │   (Stop Hook)   │                     │
│  └────────┬────────┘    └────────┬────────┘                     │
│           │                      │                               │
│           ▼                      ▼                               │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ auto-resume.sh  │    │session_state_   │                     │
│  │ (Crash Recovery)│    │manager.py       │                     │
│  └────────┬────────┘    └─────────────────┘                     │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │   notify.py     │    │  hook_utils.py  │                     │
│  │ (Notifications) │    │  (Utilities)    │                     │
│  └─────────────────┘    └─────────────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

STATE FILES:
├── logs/session_state.json         (Session state manager)
├── logs/auto-resume-state.json     (Crash recovery state)
├── logs/hook_activity_state.json   (Hook state)
├── .claude/state/ric.json          (RIC Loop state)
├── logs/continuation_history.jsonl (Stop hook log)
└── claude-progress.txt             (Human-readable tasks)
```

---

## 3. Component Function Analysis

### 3.1 run_overnight.sh (884 lines)

**Purpose:** Main entry point for overnight autonomous sessions

| Function | Lines | Purpose | Dependencies |
|----------|-------|---------|--------------|
| `preflight_checks()` | 200-265 | Validate environment (tmux, claude, python) | External commands |
| `init_progress_file()` | 268-304 | Create task tracking file | None |
| `init_session_notes()` | 307-351 | Initialize relay-race context | Template file |
| `preload_upgrade_guide()` | 354-365 | Generate domain knowledge doc | `preload_upgrade_guide.py` |
| `run_issue_detection()` | 368-394 | Proactive issue scanning | `issue_detector.py` |
| `init_watchdog_config()` | 397-419 | Create watchdog JSON config | None |
| `start_watchdog()` | 422-449 | Launch watchdog process | `watchdog.py` |
| `start_auto_resume()` | 452-477 | Launch crash recovery monitor | `auto-resume.sh` |
| `send_start_notification()` | 480-498 | Send Discord/Slack notification | `notify.py` |
| `start_claude_session()` | 501-707 | Main Claude invocation | Claude CLI |
| `cleanup()` | 710-828 | Stop monitors, log outcome | Multiple |
| `main()` | 836-883 | Orchestrate startup | All above |

**Environment Variables Set:**
- `CONTINUOUS_MODE=1` - Enables stop hook enforcement
- `RIC_MODE=SUGGESTED` - RIC Loop enforcement level
- `RIC_AUTONOMOUS_MODE=1` - Enables non-blocking mode
- `RIC_ENFORCEMENT_LEVEL=WARN` - Warn but don't block

**Flaws Identified:**
1. **Line 706:** Uses `claude --print` which may not exist in all versions
2. **Line 741-784:** Complex inline Python for session outcome logging
3. **Lines 517-534:** Dynamic routing fallback logic duplicated

---

### 3.2 watchdog.py (640 lines)

**Purpose:** External safety monitor for autonomous sessions

| Class/Function | Lines | Purpose | Return Type |
|----------------|-------|---------|-------------|
| `WatchdogConfig` | 124-148 | Configuration dataclass | N/A |
| `WatchdogConfig.from_file()` | 140-147 | Load from JSON | `WatchdogConfig` |
| `Watchdog.__init__()` | 157-165 | Initialize monitor | N/A |
| `Watchdog.find_claude_process()` | 192-211 | Find Claude PID | `Process | None` |
| `Watchdog.check_runtime()` | 213-223 | Check max runtime | `tuple[bool, str]` |
| `Watchdog.check_idle_time()` | 225-241 | Check idle duration | `tuple[bool, str]` |
| `Watchdog.check_cost()` | 243-262 | Check API budget | `tuple[bool, str]` |
| `Watchdog.check_checkpoints()` | 264-294 | Check git commits | `tuple[bool, str]` |
| `Watchdog.check_process_health()` | 296-317 | Check process status | `tuple[bool, str]` |
| `Watchdog.trigger_checkpoint()` | 319-356 | Trigger git checkpoint | `tuple[bool, str]` |
| `Watchdog.terminate_session()` | 358-397 | Kill Claude process | None |
| `Watchdog.check_ric_status()` | 435-480 | Monitor RIC Loop | `tuple[bool, str]` |
| `Watchdog.run_checks()` | 482-509 | Run all checks | `tuple[bool, list[str]]` |
| `Watchdog.start()` | 511-576 | Main monitoring loop | None |
| `Watchdog.test()` | 578-590 | Single check cycle | `bool` |

**Flaws Identified:**
1. **Lines 41-45:** Hard dependency on psutil (no graceful fallback)
2. **Lines 88-121:** Duplicate stub functions for missing notifications
3. **Line 309:** CPU usage check `interval=1` blocks for 1 second
4. **Lines 549-561:** Cost warning logic duplicates budget file reading

---

### 3.3 session_stop.py (699 lines)

**Purpose:** Stop hook enforcing task completion before exit

| Function | Lines | Purpose | Return Type |
|----------|-------|---------|-------------|
| `parse_progress_file()` | 52-133 | Parse tasks with priorities | `dict[str, Any]` |
| `get_next_pending_task()` | 136-146 | Find next P0/P1/P2 task | `tuple[str, str]` |
| `get_pending_categories()` | 149-171 | List incomplete categories | `list[dict]` |
| `calculate_completion_status()` | 174-242 | Calculate completion % | `dict[str, Any]` |
| `load_state()` | 245-272 | Load persisted state | `dict[str, Any]` |
| `check_ric_compliance()` | 275-379 | Check RIC Loop exit rules | `dict[str, Any]` |
| `generate_ric_continuation_prompt()` | 382-433 | Build RIC resume prompt | `str` |
| `save_state()` | 436-455 | Persist state to disk | None |
| `log_continuation()` | 458-474 | Log continuation decision | None |
| `generate_continuation_prompt()` | 477-513 | Build task resume prompt | `str` |
| `should_continue()` | 516-595 | Main continuation logic | `tuple[bool, str, str]` |
| `get_session_stats()` | 598-630 | Calculate git stats | `dict` |
| `create_final_checkpoint()` | 633-652 | Create git commit | `bool` |
| `main()` | 655-694 | Entry point | `int` |

**Flaws Identified:**
1. **Line 48:** `MAX_CONTINUATION_ATTEMPTS = 20` - Magic number should be configurable
2. **Lines 245-272:** State loading duplicates logic from session_state_manager.py
3. **Lines 607-616:** Git log parsing doesn't handle git errors
4. **Line 645:** Uses `--no-verify` which bypasses pre-commit hooks

---

### 3.4 session_state_manager.py (201 lines)

**Purpose:** Compaction-proof state persistence

| Class/Method | Lines | Purpose | Return Type |
|--------------|-------|---------|-------------|
| `SessionStateManager.__init__()` | 38-40 | Initialize manager | N/A |
| `SessionStateManager._load_state()` | 42-64 | Load or create state | `dict[str, Any]` |
| `SessionStateManager._save_state()` | 66-70 | Persist to disk | None |
| `SessionStateManager.initialize_session()` | 72-76 | Start new session | None |
| `SessionStateManager.record_compaction()` | 78-81 | Track compaction event | None |
| `SessionStateManager.mark_task_complete()` | 83-87 | Mark task done | None |
| `SessionStateManager.set_current_work()` | 89-93 | Set focus | None |
| `SessionStateManager.add_progress_note()` | 95-101 | Add progress note | None |
| `SessionStateManager.add_blocker()` | 103-107 | Record blocker | None |
| `SessionStateManager.add_key_decision()` | 109-113 | Record decision | None |
| `SessionStateManager.set_next_steps()` | 115-118 | Set next steps | None |
| `SessionStateManager.get_recovery_summary()` | 120-171 | Generate recovery prompt | `str` |
| `SessionStateManager.to_dict()` | 173-175 | Export state | `dict[str, Any]` |
| `create_recovery_file()` | 178-187 | Write recovery markdown | `str` |

**Flaws Identified:**
1. **Lines 99-100:** Hard limit of 20 progress notes with no archive
2. **Line 38:** Single state file path not configurable
3. **No locking mechanism for concurrent access**

---

### 3.5 auto-resume.sh (392 lines)

**Purpose:** Automatic crash recovery with exponential backoff

| Function | Lines | Purpose | Dependencies |
|----------|-------|---------|--------------|
| `log()` | 78-83 | Timestamped logging | None |
| `init_state()` | 91-96 | Initialize state file | jq |
| `get_state()` | 99-106 | Read state value | jq |
| `set_state()` | 109-116 | Update state value | jq |
| `is_claude_running()` | 119-121 | Check process | pgrep |
| `get_claude_pid()` | 124-126 | Get PID | pgrep |
| `calculate_backoff()` | 130-151 | Exponential + jitter | Bash math |
| `get_current_task()` | 154-161 | Parse progress file | grep, sed |
| `send_notification()` | 164-182 | Send via notify.py | Python |
| `update_progress()` | 185-193 | Update progress file | None |
| `get_ric_state()` | 196-235 | Parse RIC Loop state | Python |
| `get_session_state()` | 238-244 | Get session recovery | session_state_manager.py |
| `build_resume_prompt()` | 247-278 | Build resume context | Multiple |
| `start_claude()` | 281-300 | Start Claude session | Claude CLI |
| `monitor_loop()` | 303-351 | Main monitoring loop | All above |
| `cleanup()` | 354-356 | Cleanup handler | None |
| `main()` | 361-389 | Entry point | All above |

**Flaws Identified:**
1. **Line 101-105:** jq dependency not validated before use
2. **Lines 196-235:** Complex inline Python for RIC state parsing
3. **Line 288:** `claude --print` used without checking availability
4. **No PID file cleanup on abnormal exit**

---

### 3.6 hook_utils.py (532 lines)

**Purpose:** Shared utilities for all hooks

| Function | Lines | Purpose | Return Type |
|----------|-------|---------|-------------|
| `ensure_dirs()` | 48-51 | Create required directories | None |
| `log_hook_activity()` | 59-92 | Log hook events to JSON | None |
| `log_error()` | 95-129 | Log errors with recovery info | None |
| `create_recovery_point()` | 137-174 | Save recovery snapshot | `bool` |
| `get_latest_recovery_point()` | 177-186 | Get recent snapshot | `dict | None` |
| `recover_from_point()` | 189-208 | Restore from snapshot | `dict | None` |
| `load_hook_state()` | 216-236 | Load hook activity state | `dict[str, Any]` |
| `save_hook_state()` | 239-247 | Persist hook state | `bool` |
| `update_hook_stat()` | 250-259 | Update statistics | None |
| `get_current_task()` | 283-293 | Get pending task | `str | None` |
| `get_pending_tasks()` | 296-307 | List pending tasks | `list[str]` |
| `get_completed_tasks()` | 310-322 | List completed tasks | `list[str]` |
| `get_progress_stats()` | 325-336 | Calculate progress % | `dict[str, int]` |
| `is_autonomous_mode()` | 344-346 | Check env var | `bool` |
| `is_continuous_mode()` | 349-351 | Check env var | `bool` |
| `get_enforcement_level()` | 354-365 | Get warn/log level | `str` |
| `HookContext` class | 404-467 | Context manager for hooks | N/A |
| `get_session_summary()` | 475-502 | Get session summary | `dict[str, Any]` |

**Flaws Identified:**
1. **Lines 89-92:** Silent exception swallowing on logging errors
2. **Lines 263-275:** Deprecated functions still present
3. **Line 440:** Suppresses all exceptions in context manager

---

### 3.7 notify.py (469 lines)

**Purpose:** Discord/Slack notifications for session events

| Function | Lines | Purpose | Return Type |
|----------|-------|---------|-------------|
| `_post_json()` | 48-72 | HTTP POST helper | `bool` |
| `send_discord()` | 75-115 | Send Discord embed | `bool` |
| `send_slack()` | 118-158 | Send Slack attachment | `bool` |
| `notify()` | 161-213 | Send to all channels | `bool` |
| `notify_session_start()` | 221-240 | Session started | None |
| `notify_checkpoint()` | 243-249 | Checkpoint created | None |
| `notify_context_warning()` | 252-281 | Context usage warning | None |
| `notify_warning()` | 284-290 | Generic warning | None |
| `notify_error()` | 293-299 | Error notification | None |
| `notify_completion()` | 302-316 | Session complete | None |
| `notify_cost_alert()` | 319-334 | Budget warning | None |
| `notify_backtest_result()` | 337-363 | Backtest results | None |
| `notify_rate_limit()` | 366-377 | Rate limit hit | None |
| `notify_idle_warning()` | 380-387 | Idle session warning | None |
| `notify_stuck_detected()` | 390-396 | Stuck loop detection | None |
| `notify_recovery_attempt()` | 399-405 | Recovery attempt | None |
| `notify_recovery_result()` | 408-420 | Recovery result | None |
| `test_notifications()` | 428-455 | Test webhooks | None |

**Flaws Identified:**
1. **Lines 30-38:** Fallback to urllib without proper error handling
2. **Line 101:** Uses deprecated `datetime.utcnow()` instead of `datetime.now(timezone.utc)`
3. **No rate limiting on notification frequency**

---

## 4. Code Duplications

### 4.1 State Loading/Saving

**Duplicated across 3 files:**

| File | Function | Lines |
|------|----------|-------|
| `session_state_manager.py` | `_load_state()` | 42-64 |
| `session_stop.py` | `load_state()` | 245-272 |
| `hook_utils.py` | `load_hook_state()` | 216-236 |

**Impact:** Different default values, potential desynchronization

**Solution:**
```python
# Unified state manager in utils/state_manager.py
class UnifiedStateManager:
    """Single source of truth for all session state."""

    def __init__(self, state_file: Path = Path("logs/session_state.json")):
        self.state_file = state_file
        self._lock = threading.Lock()

    def load(self) -> dict[str, Any]:
        with self._lock:
            # ... unified loading logic

    def save(self, state: dict[str, Any]) -> None:
        with self._lock:
            # ... unified saving logic
```

---

### 4.2 Progress File Parsing

**Duplicated across 4 locations:**

| File | Function | Lines |
|------|----------|-------|
| `session_stop.py` | `parse_progress_file()` | 52-133 |
| `hook_utils.py` | `get_pending_tasks()` | 296-307 |
| `hook_utils.py` | `get_completed_tasks()` | 310-322 |
| `auto-resume.sh` | `get_current_task()` | 154-161 |

**Impact:** Inconsistent parsing, maintenance burden

**Solution:**
```python
# Unified progress parser in utils/progress_parser.py
class ProgressParser:
    """Single implementation for progress file parsing."""

    def __init__(self, progress_file: Path = Path("claude-progress.txt")):
        self.progress_file = progress_file

    def parse(self) -> ProgressData:
        """Parse progress file returning structured data."""
        # ... unified parsing logic

    def get_next_task(self) -> Task | None:
        # ...

    def get_completion_stats(self) -> CompletionStats:
        # ...
```

---

### 4.3 RIC State Reading

**Duplicated across 3 locations:**

| File | Function | Lines |
|------|----------|-------|
| `session_stop.py` | `check_ric_compliance()` | 275-379 |
| `watchdog.py` | `check_ric_status()` | 435-480 |
| `auto-resume.sh` | `get_ric_state()` | 196-235 (inline Python) |

**Impact:** Inconsistent RIC state interpretation

**Solution:**
```python
# Unified RIC client in utils/ric_client.py
class RICClient:
    """Unified RIC Loop state access."""

    RIC_STATE_FILE = Path(".claude/state/ric.json")

    def get_status(self) -> RICStatus:
        """Get current RIC status."""
        # ...

    def can_exit(self) -> tuple[bool, str]:
        """Check if RIC criteria allow exit."""
        # ...
```

---

### 4.4 Notification Stub Functions

**Duplicated in watchdog.py lines 88-121:**

```python
if not HAS_NOTIFICATIONS:
    def notify_session_start(*args, **kwargs): pass
    def notify_checkpoint(*args, **kwargs): pass
    def notify_warning(*args, **kwargs): pass
    # ... 10 more stubs
```

**Solution:** Use a null object pattern:

```python
class NullNotifier:
    """Null notifier when notifications unavailable."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

notifier = Notifier() if HAS_NOTIFICATIONS else NullNotifier()
```

---

## 5. Identified Flaws

### 5.1 Critical Flaws

| ID | Location | Issue | Impact |
|----|----------|-------|--------|
| **C1** | `hook_utils.py:440` | Suppresses ALL exceptions | Masks bugs |
| **C2** | `session_stop.py:645` | `git commit --no-verify` | Bypasses hooks |
| **C3** | `auto-resume.sh:288` | No Claude CLI availability check | Crash on start |
| **C4** | All state files | No file locking | Race conditions |

### 5.2 High Severity Flaws

| ID | Location | Issue | Impact |
|----|----------|-------|--------|
| **H1** | `watchdog.py:309` | `cpu_percent(interval=1)` blocks | Monitoring delay |
| **H2** | `run_overnight.sh:706` | Assumes `claude --print` exists | Version incompatibility |
| **H3** | `notify.py:101` | `datetime.utcnow()` deprecated | Python 3.12+ warning |
| **H4** | `session_state_manager.py:99` | Hard limit 20 notes, no archive | Data loss |
| **H5** | 5 state files | Fragmented state management | Desync risk |

### 5.3 Medium Severity Flaws

| ID | Location | Issue | Impact |
|----|----------|-------|--------|
| **M1** | `watchdog.py:41-45` | Hard psutil dependency | Import failure |
| **M2** | `auto-resume.sh:101` | Assumes jq available | Silent failures |
| **M3** | `hook_utils.py:89-92` | Silent logging failures | Lost diagnostics |
| **M4** | `notify.py:30-38` | urllib fallback not tested | Potential failures |
| **M5** | No centralized config | Magic numbers everywhere | Hard to tune |

### 5.4 Low Severity / Polish

| ID | Location | Issue | Impact |
|----|----------|-------|--------|
| **L1** | `hook_utils.py:263-275` | Deprecated function aliases | Technical debt |
| **L2** | Inconsistent log formats | JSON vs text logging | Hard to parse |
| **L3** | No health check endpoint | Can't query system status | Limited observability |
| **L4** | No metrics collection | No Prometheus/StatsD | No dashboards |

---

## 6. Refactoring Recommendations

### 6.1 Phase 1: State Consolidation (Priority: High)

**Goal:** Single source of truth for session state

**Current State Files:**
```
logs/session_state.json          # SessionStateManager
logs/auto-resume-state.json      # auto-resume.sh
logs/hook_activity_state.json    # hook_utils.py
.claude/state/ric.json           # RIC Loop
logs/continuation_history.jsonl  # session_stop.py
```

**Target State:**
```
logs/overnight_state.json        # Unified state
  ├── session: {...}             # Session info
  ├── recovery: {...}            # Crash recovery
  ├── hooks: {...}               # Hook activity
  ├── ric: {...}                 # RIC Loop state
  └── continuations: [...]       # Continuation history
```

**Implementation:**
```python
# utils/overnight_state.py
"""Unified overnight state management with file locking."""
import fcntl
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

@dataclass
class OvernightState:
    """Complete overnight session state."""

    # Session info
    session_id: str = ""
    goal: str = ""
    started_at: str = ""
    last_updated: str = ""

    # Progress
    completed_tasks: list[dict] = field(default_factory=list)
    current_task: str | None = None
    completion_pct: float = 0.0

    # Recovery
    restart_count: int = 0
    last_restart: str | None = None
    continuation_count: int = 0

    # RIC Loop
    ric_active: bool = False
    ric_iteration: int = 0
    ric_phase: int = 0

    # Hooks
    hook_events: int = 0
    hook_errors: int = 0

class OvernightStateManager:
    """Thread-safe state manager with file locking."""

    STATE_FILE = Path("logs/overnight_state.json")

    def __init__(self):
        self.STATE_FILE.parent.mkdir(exist_ok=True)

    def load(self) -> OvernightState:
        """Load state with file locking."""
        if not self.STATE_FILE.exists():
            return OvernightState()

        with open(self.STATE_FILE, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
                return OvernightState(**data)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def save(self, state: OvernightState) -> None:
        """Save state with file locking."""
        state.last_updated = datetime.now().isoformat()

        with open(self.STATE_FILE, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(asdict(state), f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def update(self, **kwargs) -> OvernightState:
        """Atomic read-modify-write."""
        state = self.load()
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)
        self.save(state)
        return state
```

---

### 6.2 Phase 2: Centralized Configuration (Priority: High)

**Goal:** Single configuration file with environment overrides

**Create:** `config/overnight.yaml`

```yaml
# Overnight Session Configuration
session:
  max_runtime_hours: 10
  max_idle_minutes: 30
  checkpoint_interval_minutes: 15

budget:
  max_cost_usd: 50.0
  warning_threshold_pct: 80

recovery:
  max_restarts: 5
  backoff_base_seconds: 30
  backoff_max_seconds: 600

enforcement:
  continuous_mode: true
  ric_mode: SUGGESTED  # ENFORCED | SUGGESTED | DISABLED
  min_completion_pct: 100
  require_p1: true
  require_p2: true

notifications:
  discord_webhook: ${DISCORD_WEBHOOK_URL}
  slack_webhook: ${SLACK_WEBHOOK_URL}
  idle_warning_pct: 70
  cost_warning_pct: 80

logging:
  level: INFO
  format: json
  max_entries: 500
```

**Implementation:**
```python
# utils/overnight_config.py
"""Centralized configuration with environment override support."""
import os
import yaml
from dataclasses import dataclass
from pathlib import Path

@dataclass
class OvernightConfig:
    """Validated overnight configuration."""

    # Session
    max_runtime_hours: float = 10.0
    max_idle_minutes: float = 30.0
    checkpoint_interval_minutes: float = 15.0

    # Budget
    max_cost_usd: float = 50.0
    cost_warning_threshold_pct: float = 80.0

    # Recovery
    max_restarts: int = 5
    backoff_base_seconds: int = 30
    backoff_max_seconds: int = 600

    # Enforcement
    continuous_mode: bool = True
    ric_mode: str = "SUGGESTED"
    min_completion_pct: int = 100

    # Notifications
    discord_webhook: str | None = None
    slack_webhook: str | None = None

    @classmethod
    def load(cls, config_path: Path = Path("config/overnight.yaml")) -> "OvernightConfig":
        """Load config with environment variable expansion."""
        if config_path.exists():
            with open(config_path) as f:
                raw = yaml.safe_load(f)
            # Expand ${VAR} patterns
            data = cls._expand_env_vars(raw)
            return cls._from_nested_dict(data)
        return cls()

    @staticmethod
    def _expand_env_vars(data: dict) -> dict:
        """Recursively expand ${VAR} patterns."""
        # ... implementation
```

---

### 6.3 Phase 3: Progress Parser Unification (Priority: Medium)

**Goal:** Single parser for progress file

**Create:** `utils/progress_parser.py`

```python
"""Unified progress file parser."""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

@dataclass
class Task:
    """A single task from progress file."""
    description: str
    complete: bool
    category: str
    priority: str  # P0, P1, P2

@dataclass
class Category:
    """A category from progress file."""
    name: str
    priority: str
    tasks: list[Task] = field(default_factory=list)

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self.tasks if t.complete)

    @property
    def total_count(self) -> int:
        return len(self.tasks)

    @property
    def is_complete(self) -> bool:
        return self.completed_count == self.total_count and self.total_count > 0

@dataclass
class ProgressData:
    """Parsed progress file data."""
    session_id: str | None = None
    goal: str | None = None
    categories: dict[str, Category] = field(default_factory=dict)

    @property
    def all_tasks(self) -> list[Task]:
        return [t for c in self.categories.values() for t in c.tasks]

    @property
    def completion_pct(self) -> float:
        total = len(self.all_tasks)
        if total == 0:
            return 100.0
        completed = sum(1 for t in self.all_tasks if t.complete)
        return (completed / total) * 100

    def get_next_task(self) -> Task | None:
        """Get next pending task by priority (P0 > P1 > P2)."""
        for priority in ["P0", "P1", "P2"]:
            for task in self.all_tasks:
                if task.priority == priority and not task.complete:
                    return task
        return None

class ProgressParser:
    """Parser for claude-progress.txt files."""

    CATEGORY_PATTERN = re.compile(
        r"^##\s*CATEGORY\s+(\d+):\s*(.+?)\s*\((P[0-2])\)"
    )
    TASK_PATTERN = re.compile(r"^-\s*\[([ xX])\]\s*(.+)$")
    SESSION_PATTERN = re.compile(r"^#\s*Session:\s*(.+)$")
    GOAL_PATTERN = re.compile(r"^#\s*Goal:\s*(.+)$")

    def __init__(self, progress_file: Path = Path("claude-progress.txt")):
        self.progress_file = progress_file

    def parse(self) -> ProgressData:
        """Parse the progress file."""
        data = ProgressData()

        if not self.progress_file.exists():
            return data

        content = self.progress_file.read_text()
        current_category: Category | None = None

        for line in content.split("\n"):
            # Session ID
            if match := self.SESSION_PATTERN.match(line):
                data.session_id = match.group(1).strip()
                continue

            # Goal
            if match := self.GOAL_PATTERN.match(line):
                data.goal = match.group(1).strip()
                continue

            # Category header
            if match := self.CATEGORY_PATTERN.match(line):
                cat_num = match.group(1)
                cat_name = match.group(2)
                priority = match.group(3)
                name = f"Category {cat_num}: {cat_name}"
                current_category = Category(name=name, priority=priority)
                data.categories[name] = current_category
                continue

            # Task item
            if current_category and (match := self.TASK_PATTERN.match(line.strip())):
                is_complete = match.group(1).lower() == "x"
                description = match.group(2).strip()
                task = Task(
                    description=description,
                    complete=is_complete,
                    category=current_category.name,
                    priority=current_category.priority,
                )
                current_category.tasks.append(task)

        return data
```

---

### 6.4 Phase 4: Fix Critical Issues (Priority: Critical)

#### 6.4.1 Fix Exception Suppression

**File:** `hook_utils.py:440`

```python
# BEFORE (bad)
def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type:
        log_error(...)
        return True  # Suppresses ALL exceptions

# AFTER (good)
def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type:
        log_error(...)
        # Only suppress expected, recoverable exceptions
        if isinstance(exc_val, (OSError, json.JSONDecodeError)):
            return True
    return False  # Let unexpected exceptions propagate
```

#### 6.4.2 Fix Git Commit Bypass

**File:** `session_stop.py:645`

```python
# BEFORE (bad)
subprocess.run(
    ["git", "commit", "-m", f"checkpoint: Session end at {timestamp}", "--no-verify"],
    timeout=30,
)

# AFTER (good)
subprocess.run(
    ["git", "commit", "-m", f"checkpoint: Session end at {timestamp}"],
    timeout=30,
)
# If hooks fail, that's a signal to investigate, not bypass
```

#### 6.4.3 Add Claude CLI Availability Check

**File:** `auto-resume.sh`

```bash
# Add after line 117
check_claude_cli() {
    if ! command -v claude &> /dev/null; then
        log_error "Claude CLI not found in PATH"
        exit 1
    fi

    # Check for --print support
    if ! claude --help 2>&1 | grep -q "\-\-print"; then
        log_warn "Claude CLI may not support --print flag"
        # Use alternative invocation
        CLAUDE_CMD="claude"
    else
        CLAUDE_CMD="claude --print"
    fi
}
```

#### 6.4.4 Add File Locking to State Operations

See Phase 1 implementation above - `fcntl.flock()` for file locking.

---

### 6.5 Phase 5: Add Health Check API (Priority: Medium)

**Create:** `scripts/health_check.py`

```python
"""Health check endpoint for overnight system monitoring."""
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from datetime import datetime

from utils.overnight_state import OvernightStateManager
from utils.progress_parser import ProgressParser

class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health checks."""

    def do_GET(self):
        if self.path == "/health":
            self._send_health()
        elif self.path == "/status":
            self._send_status()
        else:
            self.send_error(404)

    def _send_health(self):
        """Basic health check - is system running?"""
        # Check if Claude process is running
        import psutil
        claude_running = any(
            "claude" in (p.name() or "").lower()
            for p in psutil.process_iter(["name"])
        )

        health = {
            "status": "healthy" if claude_running else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "claude_running": claude_running,
        }

        self._send_json(health, 200 if claude_running else 503)

    def _send_status(self):
        """Detailed status check."""
        state_manager = OvernightStateManager()
        progress_parser = ProgressParser()

        state = state_manager.load()
        progress = progress_parser.parse()

        status = {
            "timestamp": datetime.now().isoformat(),
            "session": {
                "id": state.session_id,
                "goal": state.goal,
                "started_at": state.started_at,
                "runtime_hours": self._calculate_runtime(state.started_at),
            },
            "progress": {
                "completion_pct": progress.completion_pct,
                "tasks_completed": sum(1 for t in progress.all_tasks if t.complete),
                "tasks_total": len(progress.all_tasks),
                "next_task": progress.get_next_task().description if progress.get_next_task() else None,
            },
            "recovery": {
                "restart_count": state.restart_count,
                "continuation_count": state.continuation_count,
            },
            "ric": {
                "active": state.ric_active,
                "iteration": state.ric_iteration,
                "phase": state.ric_phase,
            },
        }

        self._send_json(status, 200)

    def _send_json(self, data: dict, status: int):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    @staticmethod
    def _calculate_runtime(started_at: str) -> float:
        if not started_at:
            return 0.0
        start = datetime.fromisoformat(started_at)
        return (datetime.now() - start).total_seconds() / 3600

def run_health_server(port: int = 8080):
    """Run health check server."""
    server = HTTPServer(("", port), HealthHandler)
    print(f"Health server running on port {port}")
    server.serve_forever()

if __name__ == "__main__":
    run_health_server()
```

---

## 7. Best Practices

### 7.1 State Management

```python
# GOOD: Use unified state manager with locking
from utils.overnight_state import OvernightStateManager

state_manager = OvernightStateManager()
state = state_manager.update(
    restart_count=state.restart_count + 1,
    last_restart=datetime.now().isoformat()
)

# BAD: Direct file manipulation without locking
with open("logs/state.json", "w") as f:
    json.dump(state, f)  # Race condition!
```

### 7.2 Configuration Access

```python
# GOOD: Use centralized config
from utils.overnight_config import OvernightConfig

config = OvernightConfig.load()
max_runtime = config.max_runtime_hours  # Single source of truth

# BAD: Hardcoded magic numbers
MAX_RUNTIME = 10  # Where did this come from?
```

### 7.3 Progress Parsing

```python
# GOOD: Use unified parser
from utils.progress_parser import ProgressParser

parser = ProgressParser()
progress = parser.parse()
next_task = progress.get_next_task()

# BAD: Duplicate regex parsing
for line in content.split("\n"):
    if "- [ ]" in line:  # Different from other parsers!
        task = line.replace("- [ ]", "")
```

### 7.4 Error Handling

```python
# GOOD: Specific exception handling
try:
    state = load_state()
except FileNotFoundError:
    state = default_state()
except json.JSONDecodeError as e:
    log_error("state_parse", str(e), recoverable=True)
    state = default_state()

# BAD: Blanket exception swallowing
try:
    state = load_state()
except Exception:
    pass  # What went wrong? We'll never know!
```

### 7.5 Logging

```python
# GOOD: Structured JSON logging
log_hook_activity(
    hook_name="session_stop",
    event_type="continuation_required",
    details={
        "completion_pct": 75.5,
        "pending_tasks": 12,
        "continuation_count": 3
    },
    level="INFO"
)

# BAD: Unstructured text logging
print(f"Continuing session: 75.5% complete, 12 tasks pending")
```

---

## Summary

The overnight system is a sophisticated autonomous development infrastructure with strong fundamentals but fragmented state management. Key refactoring priorities:

1. **Consolidate state files** (5 → 1) with proper locking
2. **Centralize configuration** with environment overrides
3. **Unify progress parsing** into single implementation
4. **Fix critical flaws** (exception suppression, git bypass)
5. **Add health check API** for monitoring

Estimated effort: 3-5 days for full implementation.
