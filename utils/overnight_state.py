"""
Unified Overnight State Management

Consolidates 5 separate state files into a single source of truth
with proper file locking for concurrent access safety.

Part of OVERNIGHT-002 refactoring based on docs/OVERNIGHT_SYSTEM_ANALYSIS.md

Previous state files consolidated:
- logs/session_state.json (SessionStateManager)
- logs/auto-resume-state.json (auto-resume.sh)
- logs/hook_activity_state.json (hook_utils.py)
- .claude/state/ric.json (RIC Loop - read-only reference)
- logs/continuation_history.jsonl (session_stop.py)
"""

from __future__ import annotations

import fcntl
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class OvernightState:
    """Complete overnight session state."""

    # Session info
    session_id: str = ""
    goal: str = ""
    started_at: str = ""
    last_updated: str = ""

    # Progress tracking
    completed_tasks: list[dict[str, Any]] = field(default_factory=list)
    current_task: str | None = None
    completion_pct: float = 0.0
    total_tasks: int = 0
    completed_count: int = 0

    # Recovery tracking
    restart_count: int = 0
    last_restart: str | None = None
    continuation_count: int = 0
    max_continuations: int = 20

    # RIC Loop state (read from .claude/state/ric.json)
    ric_active: bool = False
    ric_iteration: int = 0
    ric_phase: int = 0
    ric_can_exit: bool = True

    # Hook activity
    hook_events: int = 0
    hook_errors: int = 0
    last_hook_activity: str | None = None

    # Progress notes (limited to avoid memory issues)
    progress_notes: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    key_decisions: list[dict[str, str]] = field(default_factory=list)

    # Continuation history
    continuations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OvernightState:
        """Create from dictionary."""
        # Handle unknown fields gracefully
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


class OvernightStateManager:
    """
    Thread-safe state manager with file locking.

    Usage:
        manager = OvernightStateManager()
        state = manager.load()
        state.restart_count += 1
        manager.save(state)

        # Or use atomic update:
        manager.update(restart_count=5, last_restart=datetime.now().isoformat())
    """

    STATE_FILE = Path("logs/overnight_state.json")
    RIC_STATE_FILE = Path(".claude/state/ric.json")
    MAX_PROGRESS_NOTES = 50
    MAX_CONTINUATIONS = 100

    def __init__(self, state_file: Path | None = None):
        """Initialize state manager."""
        self.state_file = state_file or self.STATE_FILE
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> OvernightState:
        """Load state with shared file locking."""
        if not self.state_file.exists():
            return OvernightState()

        try:
            with open(self.state_file) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                    state = OvernightState.from_dict(data)
                    # Merge RIC state if available
                    self._merge_ric_state(state)
                    return state
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load state: {e}, using default")
            return OvernightState()

    def save(self, state: OvernightState) -> None:
        """Save state with exclusive file locking."""
        state.last_updated = datetime.now(timezone.utc).isoformat()

        # Enforce limits
        if len(state.progress_notes) > self.MAX_PROGRESS_NOTES:
            state.progress_notes = state.progress_notes[-self.MAX_PROGRESS_NOTES :]
        if len(state.continuations) > self.MAX_CONTINUATIONS:
            state.continuations = state.continuations[-self.MAX_CONTINUATIONS :]

        try:
            with open(self.state_file, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(state.to_dict(), f, indent=2)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError as e:
            logger.error(f"Failed to save state: {e}")
            raise

    def update(self, **kwargs: Any) -> OvernightState:
        """
        Atomic read-modify-write operation.

        Args:
            **kwargs: Fields to update

        Returns:
            Updated state
        """
        state = self.load()
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)
            else:
                logger.warning(f"Unknown state field: {key}")
        self.save(state)
        return state

    def _merge_ric_state(self, state: OvernightState) -> None:
        """Merge RIC Loop state (read-only)."""
        if not self.RIC_STATE_FILE.exists():
            return

        try:
            with open(self.RIC_STATE_FILE) as f:
                ric_data = json.load(f)
                state.ric_active = ric_data.get("active", False)
                state.ric_iteration = ric_data.get("iteration", 0)
                state.ric_phase = ric_data.get("phase", 0)
                state.ric_can_exit = ric_data.get("can_exit", True)
        except (json.JSONDecodeError, OSError):
            pass  # RIC state is optional

    # Convenience methods for common operations

    def initialize_session(
        self,
        session_id: str,
        goal: str,
        total_tasks: int = 0,
    ) -> OvernightState:
        """Initialize a new session."""
        state = OvernightState(
            session_id=session_id,
            goal=goal,
            started_at=datetime.now(timezone.utc).isoformat(),
            total_tasks=total_tasks,
        )
        self.save(state)
        return state

    def record_restart(self) -> OvernightState:
        """Record a session restart."""
        state = self.load()
        state.restart_count += 1
        state.last_restart = datetime.now(timezone.utc).isoformat()
        self.save(state)
        return state

    def record_continuation(
        self,
        reason: str,
        completion_pct: float,
        pending_tasks: int,
    ) -> OvernightState:
        """Record a continuation decision."""
        state = self.load()
        state.continuation_count += 1
        state.continuations.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "completion_pct": completion_pct,
                "pending_tasks": pending_tasks,
                "count": state.continuation_count,
            }
        )
        self.save(state)
        return state

    def mark_task_complete(
        self,
        task_description: str,
        category: str = "",
    ) -> OvernightState:
        """Mark a task as complete."""
        state = self.load()
        state.completed_tasks.append(
            {
                "description": task_description,
                "category": category,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        state.completed_count = len(state.completed_tasks)
        if state.total_tasks > 0:
            state.completion_pct = (state.completed_count / state.total_tasks) * 100
        self.save(state)
        return state

    def set_current_work(self, task: str | None) -> OvernightState:
        """Set current work focus."""
        return self.update(current_task=task)

    def add_progress_note(self, note: str) -> OvernightState:
        """Add a progress note."""
        state = self.load()
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        state.progress_notes.append(f"[{timestamp}] {note}")
        self.save(state)
        return state

    def add_blocker(self, blocker: str) -> OvernightState:
        """Record a blocker."""
        state = self.load()
        state.blockers.append(blocker)
        self.save(state)
        return state

    def add_key_decision(self, decision: str, rationale: str) -> OvernightState:
        """Record a key decision."""
        state = self.load()
        state.key_decisions.append(
            {
                "decision": decision,
                "rationale": rationale,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.save(state)
        return state

    def record_hook_activity(self, hook_name: str, success: bool = True) -> OvernightState:
        """Record hook activity."""
        state = self.load()
        state.hook_events += 1
        if not success:
            state.hook_errors += 1
        state.last_hook_activity = f"{hook_name}@{datetime.now(timezone.utc).isoformat()}"
        self.save(state)
        return state

    def get_recovery_summary(self) -> str:
        """Generate a recovery summary for session resume."""
        state = self.load()

        lines = [
            "## Session Recovery Context",
            "",
            f"**Session**: {state.session_id}",
            f"**Goal**: {state.goal}",
            f"**Started**: {state.started_at}",
            f"**Restarts**: {state.restart_count}",
            f"**Continuations**: {state.continuation_count}",
            "",
            "### Progress",
            f"- Completion: {state.completion_pct:.1f}%",
            f"- Completed: {state.completed_count}/{state.total_tasks}",
            f"- Current: {state.current_task or 'None'}",
        ]

        if state.blockers:
            lines.extend(["", "### Blockers"])
            for blocker in state.blockers[-5:]:
                lines.append(f"- {blocker}")

        if state.key_decisions:
            lines.extend(["", "### Key Decisions"])
            for decision in state.key_decisions[-5:]:
                lines.append(f"- {decision['decision']}: {decision['rationale']}")

        if state.progress_notes:
            lines.extend(["", "### Recent Progress"])
            for note in state.progress_notes[-10:]:
                lines.append(f"- {note}")

        return "\n".join(lines)

    def can_continue(self) -> tuple[bool, str]:
        """Check if session can continue."""
        state = self.load()

        if state.continuation_count >= state.max_continuations:
            return False, f"Max continuations ({state.max_continuations}) reached"

        if state.completion_pct >= 100:
            return False, "All tasks complete"

        if not state.ric_can_exit and state.ric_active:
            return True, "RIC Loop requires completion"

        if state.completion_pct < 100:
            return True, f"Tasks pending ({100 - state.completion_pct:.1f}% remaining)"

        return False, "No pending work"


# Global instance for convenience
_default_manager: OvernightStateManager | None = None


def get_overnight_state_manager() -> OvernightStateManager:
    """Get the default overnight state manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = OvernightStateManager()
    return _default_manager


def create_recovery_file(output_path: Path | None = None) -> str:
    """Create a recovery markdown file."""
    manager = get_overnight_state_manager()
    summary = manager.get_recovery_summary()

    if output_path is None:
        output_path = Path("logs/recovery_context.md")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary)

    return str(output_path)
