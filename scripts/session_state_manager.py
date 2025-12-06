#!/usr/bin/env python3
"""
Session State Manager for Compaction-Proof Overnight Sessions

This module provides persistent state management that survives:
1. Context compaction (70% token threshold)
2. Session restarts / crashes
3. Auto-resume recovery

Key features:
- JSON state file persisted to disk
- Automatic state recovery on session start
- Progress tracking across compaction events
- Clear continuation directives for Claude

Usage:
    # At session start
    manager = SessionStateManager()
    manager.initialize_session("Complete UPGRADE-014")

    # During work
    manager.mark_task_complete("4.1 Hierarchical memory")
    manager.record_progress("Finished checkpointing implementation")

    # After compaction (call this to regenerate context)
    summary = manager.get_recovery_summary()
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class SessionStateManager:
    """Manages persistent session state across compaction events."""

    def __init__(self, state_file: str = "logs/session_state.json"):
        self.state_file = Path(state_file)
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        """Load state from disk or create default."""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except (OSError, json.JSONDecodeError):
                pass

        return {
            "session_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "goal": "",
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "compaction_count": 0,
            "continuation_count": 0,
            "completed_tasks": [],
            "current_category": None,
            "current_task": None,
            "progress_notes": [],
            "blockers": [],
            "key_decisions": [],
            "next_steps": [],
        }

    def _save_state(self):
        """Persist state to disk."""
        self.state["last_updated"] = datetime.now().isoformat()
        self.state_file.parent.mkdir(exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def initialize_session(self, goal: str):
        """Initialize a new session with a goal."""
        self.state["goal"] = goal
        self.state["started_at"] = datetime.now().isoformat()
        self._save_state()

    def record_compaction(self):
        """Record that a context compaction occurred."""
        self.state["compaction_count"] += 1
        self._save_state()

    def mark_task_complete(self, task: str, category: str | None = None):
        """Mark a task as complete."""
        entry = {"task": task, "category": category, "completed_at": datetime.now().isoformat()}
        self.state["completed_tasks"].append(entry)
        self._save_state()

    def set_current_work(self, category: str, task: str):
        """Set the current work focus."""
        self.state["current_category"] = category
        self.state["current_task"] = task
        self._save_state()

    def add_progress_note(self, note: str):
        """Add a progress note."""
        entry = {"note": note, "timestamp": datetime.now().isoformat()}
        self.state["progress_notes"].append(entry)
        # Keep only last 20 notes
        self.state["progress_notes"] = self.state["progress_notes"][-20:]
        self._save_state()

    def add_blocker(self, blocker: str, workaround: str | None = None):
        """Record a blocker."""
        entry = {"blocker": blocker, "workaround": workaround, "timestamp": datetime.now().isoformat()}
        self.state["blockers"].append(entry)
        self._save_state()

    def add_key_decision(self, decision: str, rationale: str):
        """Record a key decision for future reference."""
        entry = {"decision": decision, "rationale": rationale, "timestamp": datetime.now().isoformat()}
        self.state["key_decisions"].append(entry)
        self._save_state()

    def set_next_steps(self, steps: list[str]):
        """Set explicit next steps."""
        self.state["next_steps"] = steps
        self._save_state()

    def get_recovery_summary(self) -> str:
        """Generate a recovery summary after compaction.

        This summary should be read by Claude after context compaction
        to restore working context.
        """
        completed = len(self.state["completed_tasks"])
        recent_notes = self.state["progress_notes"][-5:]

        summary = f"""
## SESSION RECOVERY CONTEXT (Post-Compaction #{self.state['compaction_count']})

**Session ID**: {self.state['session_id']}
**Goal**: {self.state['goal']}
**Started**: {self.state['started_at']}
**Tasks Completed This Session**: {completed}

### Current Focus
- **Category**: {self.state['current_category'] or 'Not set'}
- **Task**: {self.state['current_task'] or 'Not set'}

### Recent Progress
"""
        for note in recent_notes:
            summary += f"- {note['note']}\n"

        if self.state["key_decisions"]:
            summary += "\n### Key Decisions Made\n"
            for dec in self.state["key_decisions"][-5:]:
                summary += f"- {dec['decision']}: {dec['rationale']}\n"

        if self.state["blockers"]:
            summary += "\n### Known Blockers\n"
            for b in self.state["blockers"]:
                summary += f"- {b['blocker']}"
                if b.get("workaround"):
                    summary += f" (Workaround: {b['workaround']})"
                summary += "\n"

        if self.state["next_steps"]:
            summary += "\n### Next Steps (in order)\n"
            for i, step in enumerate(self.state["next_steps"], 1):
                summary += f"{i}. {step}\n"

        summary += """
### ACTION REQUIRED
1. Read claude-progress.txt for full task list
2. Continue from current focus above
3. Mark tasks with [x] as completed
4. DO NOT STOP until all P0/P1/P2 tasks are complete
"""
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Export state as dictionary."""
        return self.state.copy()


def create_recovery_file():
    """Create a recovery file for Claude to read after compaction."""
    manager = SessionStateManager()
    summary = manager.get_recovery_summary()

    recovery_file = Path("claude-recovery-context.md")
    recovery_file.write_text(summary)

    print(f"Recovery context written to {recovery_file}")
    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "recovery":
            print(create_recovery_file())
        elif sys.argv[1] == "status":
            manager = SessionStateManager()
            print(json.dumps(manager.to_dict(), indent=2))
    else:
        print("Usage: python session_state_manager.py [recovery|status]")
