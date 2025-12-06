#!/usr/bin/env python3
"""
Hook Utilities (v4.3) - Unified Logging, Recovery, and Progress Tracking

Provides shared utilities for all hooks:
- Centralized logging
- Recovery point tracking
- Progress persistence
- Error recovery
- Session context

All hooks should import and use these utilities for consistent behavior.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# Import unified progress parser
# Part of P1-3 integration from REMEDIATION_PLAN.md
try:
    _project_root = str(Path(__file__).parent.parent.parent.parent)
except NameError:
    _project_root = str(Path.cwd())
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from utils.progress_parser import ProgressParser


# =============================================================================
# PATHS
# =============================================================================

CLAUDE_DIR = Path(".claude")
LOG_DIR = Path("logs")  # Unified with scripts/session_state_manager.py
STATE_DIR = CLAUDE_DIR / "state"

# Log files
HOOK_LOG_FILE = CLAUDE_DIR / "logs" / "hook_activity.json"
ERROR_LOG_FILE = CLAUDE_DIR / "logs" / "hook_errors.json"
RECOVERY_LOG_FILE = CLAUDE_DIR / "logs" / "recovery_points.json"
PROGRESS_FILE = Path("claude-progress.txt")

# State files
# Hook activity state is separate from session state (different data structure)
# Session state is managed by scripts/session_state_manager.py
HOOK_STATE_FILE = LOG_DIR / "hook_activity_state.json"
RECOVERY_STATE_FILE = STATE_DIR / "recovery_state.json"


# =============================================================================
# INITIALIZATION
# =============================================================================


def ensure_dirs() -> None:
    """Ensure all required directories exist."""
    for dir_path in [CLAUDE_DIR, LOG_DIR, STATE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOGGING
# =============================================================================


def log_hook_activity(hook_name: str, event_type: str, details: dict[str, Any], level: str = "INFO") -> None:
    """
    Log hook activity for monitoring and debugging.

    Args:
        hook_name: Name of the hook (e.g., "ric", "risk_validator")
        event_type: Type of event (e.g., "check", "warn", "auto_fix")
        details: Additional event details
        level: Log level (INFO, WARN, ERROR)
    """
    try:
        ensure_dirs()
        log = []
        if HOOK_LOG_FILE.exists():
            try:
                log = json.loads(HOOK_LOG_FILE.read_text())
            except json.JSONDecodeError:
                log = []

        log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "hook": hook_name,
                "event": event_type,
                "level": level,
                "details": details,
            }
        )

        # Keep last 500 entries
        log = log[-500:]
        HOOK_LOG_FILE.write_text(json.dumps(log, indent=2))
    except OSError:
        pass  # Don't fail on logging errors


def log_error(hook_name: str, error: str, context: dict[str, Any], recoverable: bool = True) -> None:
    """
    Log an error with recovery information.

    Args:
        hook_name: Name of the hook that encountered the error
        error: Error message
        context: Context when error occurred
        recoverable: Whether the error was recovered from
    """
    try:
        ensure_dirs()
        log = []
        if ERROR_LOG_FILE.exists():
            try:
                log = json.loads(ERROR_LOG_FILE.read_text())
            except json.JSONDecodeError:
                log = []

        log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "hook": hook_name,
                "error": error,
                "context": context,
                "recoverable": recoverable,
                "recovered": recoverable,  # Assume recovered if recoverable
            }
        )

        # Keep last 200 errors
        log = log[-200:]
        ERROR_LOG_FILE.write_text(json.dumps(log, indent=2))
    except OSError:
        pass


# =============================================================================
# RECOVERY POINTS
# =============================================================================


def create_recovery_point(name: str, state: dict[str, Any], description: str = "") -> bool:
    """
    Create a recovery point for session state.

    Args:
        name: Unique name for the recovery point
        state: State to save
        description: Optional description

    Returns:
        True if recovery point created successfully
    """
    try:
        ensure_dirs()
        points = []
        if RECOVERY_LOG_FILE.exists():
            try:
                points = json.loads(RECOVERY_LOG_FILE.read_text())
            except json.JSONDecodeError:
                points = []

        points.append(
            {
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "description": description,
                "state": state,
            }
        )

        # Keep last 50 recovery points
        points = points[-50:]
        RECOVERY_LOG_FILE.write_text(json.dumps(points, indent=2))

        log_hook_activity("recovery", "point_created", {"name": name})
        return True
    except OSError:
        return False


def get_latest_recovery_point() -> dict[str, Any] | None:
    """Get the most recent recovery point."""
    try:
        if RECOVERY_LOG_FILE.exists():
            points = json.loads(RECOVERY_LOG_FILE.read_text())
            if points:
                return points[-1]
    except (OSError, json.JSONDecodeError):
        pass
    return None


def recover_from_point(name: str) -> dict[str, Any] | None:
    """
    Recover state from a named recovery point.

    Args:
        name: Name of the recovery point

    Returns:
        State dict if found, None otherwise
    """
    try:
        if RECOVERY_LOG_FILE.exists():
            points = json.loads(RECOVERY_LOG_FILE.read_text())
            for point in reversed(points):
                if point.get("name") == name:
                    log_hook_activity("recovery", "point_restored", {"name": name})
                    return point.get("state")
    except (OSError, json.JSONDecodeError):
        pass
    return None


# =============================================================================
# HOOK STATE (separate from session state)
# =============================================================================


def load_hook_state() -> dict[str, Any]:
    """Load the current hook activity state.

    Note: This is separate from session state (managed by session_state_manager.py).
    Hook state tracks: hook_counts, warnings, errors, auto_fixes.
    """
    try:
        ensure_dirs()
        if HOOK_STATE_FILE.exists():
            return json.loads(HOOK_STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        pass

    return {
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "started_at": datetime.now().isoformat(),
        "hook_counts": {},
        "warnings": [],
        "errors": [],
        "auto_fixes": [],
    }


def save_hook_state(state: dict[str, Any]) -> bool:
    """Save the current hook activity state."""
    try:
        ensure_dirs()
        state["last_updated"] = datetime.now().isoformat()
        HOOK_STATE_FILE.write_text(json.dumps(state, indent=2))
        return True
    except OSError:
        return False


def update_hook_stat(stat_name: str, increment: int = 1) -> None:
    """Update a hook statistic."""
    try:
        state = load_hook_state()
        if "stats" not in state:
            state["stats"] = {}
        state["stats"][stat_name] = state["stats"].get(stat_name, 0) + increment
        save_hook_state(state)
    except Exception:
        pass


# Backwards compatibility aliases (deprecated - use hook_state functions)
def load_session_state() -> dict[str, Any]:
    """Deprecated: Use load_hook_state() instead."""
    return load_hook_state()


def save_session_state(state: dict[str, Any]) -> bool:
    """Deprecated: Use save_hook_state() instead."""
    return save_hook_state(state)


def update_session_stat(stat_name: str, increment: int = 1) -> None:
    """Deprecated: Use update_hook_stat() instead."""
    update_hook_stat(stat_name, increment)


# =============================================================================
# PROGRESS TRACKING (Using unified ProgressParser from utils/)
# Part of P1-3 integration from REMEDIATION_PLAN.md
# =============================================================================


def get_current_task() -> str | None:
    """Get the current (next pending) task from progress file.

    Uses unified ProgressParser for consistent parsing.
    Prioritizes P0 > P1 > P2 tasks.
    """
    try:
        parser = ProgressParser(PROGRESS_FILE)
        data = parser.parse()
        next_task = data.get_next_task()
        return next_task.description if next_task else None
    except Exception:
        return None


def get_pending_tasks() -> list[str]:
    """Get all pending tasks from progress file.

    Uses unified ProgressParser for consistent parsing.
    """
    try:
        parser = ProgressParser(PROGRESS_FILE)
        data = parser.parse()
        return [task.description for task in data.pending_tasks]
    except Exception:
        return []


def get_completed_tasks() -> list[str]:
    """Get all completed tasks from progress file.

    Uses unified ProgressParser for consistent parsing.
    """
    try:
        parser = ProgressParser(PROGRESS_FILE)
        data = parser.parse()
        return [task.description for task in data.completed_tasks]
    except Exception:
        return []


def get_progress_stats() -> dict[str, int]:
    """Get progress statistics.

    Uses unified ProgressParser for consistent parsing.
    """
    try:
        parser = ProgressParser(PROGRESS_FILE)
        data = parser.parse()
        return {
            "total": data.total_count,
            "completed": data.completed_count,
            "pending": len(data.pending_tasks),
            "completion_pct": int(data.completion_pct),
        }
    except Exception:
        return {
            "total": 0,
            "completed": 0,
            "pending": 0,
            "completion_pct": 0,
        }


# =============================================================================
# AUTONOMOUS MODE
# =============================================================================


def is_autonomous_mode() -> bool:
    """Check if running in autonomous/overnight mode."""
    return os.environ.get("RIC_AUTONOMOUS_MODE", "0") == "1"


def is_continuous_mode() -> bool:
    """Check if running in continuous mode."""
    return os.environ.get("CONTINUOUS_MODE", "0") == "1"


def get_enforcement_level() -> str:
    """
    Get enforcement level based on mode.

    Returns:
        'WARN' - Warn but never block (default in v4.3)
        'LOG' - Just log, no warnings (silent mode)
    """
    # v4.3: Never block, only warn or log
    if is_autonomous_mode():
        return "LOG"  # Silent in autonomous mode
    return "WARN"  # Warn in interactive mode


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def safe_json_loads(content: str, default: Any = None) -> Any:
    """Safely load JSON with default fallback."""
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else {}


def safe_read_file(path: Path) -> str | None:
    """Safely read a file, returning None on error."""
    try:
        return path.read_text()
    except OSError:
        return None


def safe_write_file(path: Path, content: str) -> bool:
    """Safely write a file, returning success status."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return True
    except OSError:
        return False


# =============================================================================
# CONTEXT MANAGER FOR HOOKS
# =============================================================================


class HookContext:
    """
    Context manager for hook execution with automatic logging and error handling.

    Usage:
        with HookContext("my_hook") as ctx:
            ctx.log("doing something")
            result = do_something()
            if not result:
                ctx.warn("something went wrong")
    """

    def __init__(self, hook_name: str):
        self.hook_name = hook_name
        self.start_time = datetime.now()
        self.events: list[dict] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def __enter__(self):
        log_hook_activity(self.hook_name, "start", {"time": self.start_time.isoformat()})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000

        if exc_type:
            log_error(
                self.hook_name,
                str(exc_val),
                {
                    "type": exc_type.__name__,
                    "duration_ms": duration_ms,
                },
                recoverable=True,
            )
            # Only suppress expected, recoverable exceptions
            # Let unexpected exceptions propagate for debugging
            if isinstance(exc_val, (OSError, json.JSONDecodeError, FileNotFoundError)):
                return True  # Suppress recoverable I/O errors
            return False  # Let unexpected exceptions propagate

        log_hook_activity(
            self.hook_name,
            "complete",
            {
                "duration_ms": duration_ms,
                "events": len(self.events),
                "warnings": len(self.warnings),
                "errors": len(self.errors),
            },
        )
        return False

    def log(self, message: str, details: dict | None = None) -> None:
        """Log an event."""
        self.events.append({"message": message, "details": details or {}})
        log_hook_activity(self.hook_name, "event", {"message": message, **(details or {})})

    def warn(self, message: str) -> None:
        """Log a warning."""
        self.warnings.append(message)
        log_hook_activity(self.hook_name, "warning", {"message": message}, level="WARN")

    def error(self, message: str, recoverable: bool = True) -> None:
        """Log an error."""
        self.errors.append(message)
        log_error(self.hook_name, message, {}, recoverable=recoverable)


# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================


def get_session_summary() -> dict[str, Any]:
    """Get a summary of the current session hook activity."""
    state = load_hook_state()

    # Count recent activity
    try:
        hook_log = []
        if HOOK_LOG_FILE.exists():
            hook_log = json.loads(HOOK_LOG_FILE.read_text())

        error_log = []
        if ERROR_LOG_FILE.exists():
            error_log = json.loads(ERROR_LOG_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        hook_log = []
        error_log = []

    return {
        "session_id": state.get("session_id"),
        "started_at": state.get("started_at"),
        "last_updated": state.get("last_updated"),
        "total_hook_events": len(hook_log),
        "total_errors": len(error_log),
        "recoverable_errors": sum(1 for e in error_log if e.get("recoverable")),
        "progress": get_progress_stats(),
        "autonomous_mode": is_autonomous_mode(),
        "continuous_mode": is_continuous_mode(),
    }


def print_session_summary() -> None:
    """Print a formatted session summary."""
    summary = get_session_summary()
    print(f"""
Session Summary
===============
Session ID: {summary['session_id']}
Started: {summary['started_at']}
Last Activity: {summary['last_updated']}

Activity:
  Hook Events: {summary['total_hook_events']}
  Errors: {summary['total_errors']} ({summary['recoverable_errors']} recovered)

Progress:
  Completed: {summary['progress']['completed']}/{summary['progress']['total']} ({summary['progress']['completion_pct']}%)
  Pending: {summary['progress']['pending']}

Mode:
  Autonomous: {summary['autonomous_mode']}
  Continuous: {summary['continuous_mode']}
""")


if __name__ == "__main__":
    # If run directly, print session summary
    print_session_summary()
