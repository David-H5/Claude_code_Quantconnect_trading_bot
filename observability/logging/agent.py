"""Agent Activity Logger.

Provides structured logging for AI agent activities, including:
- Session tracking (start, end, handoff)
- File operations (created, modified, deleted)
- Task progress (started, completed, failed)
- Git operations (commit, push)
- Cross-agent communication (handoffs, conflicts)

Integrates with the unified logging infrastructure and supports
logging from external sessions (outside the project folder).

Usage:
    from observability.logging.agent import AgentLogger, create_agent_logger

    # Create logger (auto-detects project root)
    logger = create_agent_logger("my-agent-id")

    # Log session start
    logger.log_session_start(task="Implement feature X", cwd="/some/path")

    # Log file operations
    logger.log_file_created("docs/NEW_FILE.md", description="New documentation")
    logger.log_file_modified("src/module.py", changes="Added new function")

    # Log task progress
    logger.log_task_started("Fix bug #123")
    logger.log_task_completed("Fix bug #123", duration_seconds=120)

    # Log git operations
    logger.log_git_commit("abc123", message="fix: resolve bug", files=["src/module.py"])
    logger.log_git_push(branch="main", remote="origin")

    # Log session end
    logger.log_session_end(status="completed", summary="Fixed 3 bugs")

Part of the Parallel Agent Coordination System.
"""

from __future__ import annotations

import fcntl
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from observability.logging.base import (
    AbstractLogger,
    LogCategory,
    LogEntry,
    LogLevel,
)


class AgentEventType(Enum):
    """Types of agent events."""

    # Session events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_HANDOFF = "session_handoff"
    SESSION_CHECKPOINT = "session_checkpoint"

    # File events
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    FILE_READ = "file_read"
    FILE_LOCKED = "file_locked"
    FILE_UNLOCKED = "file_unlocked"

    # Task events
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_BLOCKED = "task_blocked"

    # Git events
    GIT_COMMIT = "git_commit"
    GIT_PUSH = "git_push"
    GIT_PULL = "git_pull"
    GIT_MERGE = "git_merge"
    GIT_CONFLICT = "git_conflict"

    # Coordination events
    LOCK_ACQUIRED = "lock_acquired"
    LOCK_RELEASED = "lock_released"
    LOCK_BLOCKED = "lock_blocked"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"

    # Analysis events
    CODEBASE_ANALYZED = "codebase_analyzed"
    RESEARCH_COMPLETED = "research_completed"
    PLAN_CREATED = "plan_created"
    VERIFICATION_COMPLETED = "verification_completed"


@dataclass
class AgentLogEntry(LogEntry):
    """Extended log entry for agent activities."""

    agent_id: str = ""
    stream_id: str | None = None
    event_type: str = ""
    files_affected: list[str] = field(default_factory=list)
    git_info: dict[str, Any] = field(default_factory=dict)
    task_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary including agent-specific fields."""
        result = super().to_dict()
        result.update(
            {
                "agent_id": self.agent_id,
                "stream_id": self.stream_id,
                "files_affected": self.files_affected,
                "git_info": self.git_info,
                "task_info": self.task_info,
            }
        )
        return result


class AgentLogger(AbstractLogger):
    """Logger for AI agent activities.

    Writes to multiple destinations:
    - .claude/state/agent_activity.jsonl (append-only activity log)
    - .claude/state/handoff.json (current handoff context)
    - claude-progress.txt (human-readable progress)
    - logs/agent_decisions.jsonl (decision audit trail)
    """

    def __init__(
        self,
        agent_id: str,
        stream_id: str | None = None,
        project_root: Path | None = None,
    ) -> None:
        """Initialize agent logger.

        Args:
            agent_id: Unique identifier for this agent
            stream_id: Optional work stream identifier (A-H)
            project_root: Project root directory (auto-detected if not provided)
        """
        self.agent_id = agent_id
        self.stream_id = stream_id
        self.project_root = project_root or self._find_project_root()
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.session_start = datetime.now(timezone.utc)
        self.tasks_completed: list[str] = []
        self.files_modified: list[str] = []
        self.git_commits: list[str] = []

        # Ensure state directories exist
        self._ensure_directories()

    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        # Try common locations
        candidates = [
            Path("/home/dshooter/projects/Claude_code_Quantconnect_trading_bot"),
            Path.cwd(),
            Path.cwd().parent,
        ]

        for candidate in candidates:
            if (candidate / "CLAUDE.md").exists() or (candidate / ".claude").exists():
                return candidate

        # Default to first candidate
        return candidates[0]

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        state_dir = self.project_root / ".claude" / "state"
        logs_dir = state_dir / "logs"
        state_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

    def _get_activity_log_path(self) -> Path:
        """Get path to activity log file."""
        return self.project_root / ".claude" / "state" / "agent_activity.jsonl"

    def _get_handoff_path(self) -> Path:
        """Get path to handoff file."""
        return self.project_root / ".claude" / "state" / "handoff.json"

    def _get_progress_path(self) -> Path:
        """Get path to progress file."""
        return self.project_root / "claude-progress.txt"

    def _get_decisions_log_path(self) -> Path:
        """Get path to decisions log file."""
        return self.project_root / "logs" / "agent_decisions.jsonl"

    def _append_to_jsonl(self, path: Path, entry: dict[str, Any]) -> None:
        """Append entry to JSONL file with file locking."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(entry) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        """Write JSON file with file locking."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _append_to_progress(self, text: str) -> None:
        """Append to progress file."""
        path = self._get_progress_path()
        with open(path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(text + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def log(
        self,
        level: LogLevel,
        category: LogCategory,
        event_type: str,
        message: str,
        data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LogEntry:
        """Log an event.

        Args:
            level: Log severity level
            category: Log category
            event_type: Type of event
            message: Human-readable message
            data: Additional data to log
            **kwargs: Additional fields

        Returns:
            The created LogEntry.
        """
        entry = AgentLogEntry(
            level=level,
            category=category,
            event_type=event_type,
            message=message,
            data=data or {},
            agent_id=self.agent_id,
            stream_id=self.stream_id,
            session_id=self.session_id,
            source=f"agent:{self.agent_id}",
            **kwargs,
        )

        # Write to activity log
        self._append_to_jsonl(self._get_activity_log_path(), entry.to_dict())

        return entry

    # =========================================================================
    # SESSION METHODS
    # =========================================================================

    def log_session_start(self, task: str, cwd: str | None = None, context: dict[str, Any] | None = None) -> LogEntry:
        """Log session start."""
        entry = self.log(
            level=LogLevel.INFO,
            category=LogCategory.AGENT,
            event_type=AgentEventType.SESSION_START.value,
            message=f"Agent {self.agent_id} started session",
            data={
                "task": task,
                "cwd": cwd or str(Path.cwd()),
                "project_root": str(self.project_root),
                "context": context or {},
            },
        )

        # Update progress file
        self._append_to_progress(
            f"\n## Agent Session: {self.agent_id}\n"
            f"- Time: {self.session_start.isoformat()}\n"
            f"- Task: {task}\n"
            f"- CWD: {cwd or Path.cwd()}\n"
        )

        return entry

    def log_session_end(
        self,
        status: str = "completed",
        summary: str = "",
        errors: list[str] | None = None,
    ) -> LogEntry:
        """Log session end."""
        duration = (datetime.now(timezone.utc) - self.session_start).total_seconds()

        entry = self.log(
            level=LogLevel.INFO,
            category=LogCategory.AGENT,
            event_type=AgentEventType.SESSION_END.value,
            message=f"Agent {self.agent_id} ended session: {status}",
            data={
                "status": status,
                "summary": summary,
                "duration_seconds": duration,
                "tasks_completed": self.tasks_completed,
                "files_modified": self.files_modified,
                "git_commits": self.git_commits,
                "errors": errors or [],
            },
            duration_ms=duration * 1000,
            outcome=status,
        )

        # Update progress file
        self._append_to_progress(
            f"\n**Session End**: {status}\n"
            f"- Duration: {duration:.1f}s\n"
            f"- Tasks: {len(self.tasks_completed)} completed\n"
            f"- Files: {len(self.files_modified)} modified\n"
            f"- Commits: {len(self.git_commits)}\n"
            f"- Summary: {summary}\n"
        )

        return entry

    def log_session_handoff(
        self,
        to_agent: str,
        completed: list[str],
        pending: list[str],
        warnings: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> LogEntry:
        """Log handoff to another agent."""
        entry = self.log(
            level=LogLevel.INFO,
            category=LogCategory.AGENT,
            event_type=AgentEventType.SESSION_HANDOFF.value,
            message=f"Handoff from {self.agent_id} to {to_agent}",
            data={
                "to_agent": to_agent,
                "completed": completed,
                "pending": pending,
                "warnings": warnings or [],
                "context": context or {},
            },
        )

        # Update handoff file
        handoff_data = {
            "from_agent": self.agent_id,
            "to_agent": to_agent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "context": {
                "completed": completed,
                "pending": pending,
                "warnings": warnings or [],
                **(context or {}),
            },
            "files_to_review": self.files_modified,
        }
        self._write_json(self._get_handoff_path(), handoff_data)

        return entry

    # =========================================================================
    # FILE OPERATION METHODS
    # =========================================================================

    def log_file_created(self, file_path: str, description: str = "") -> LogEntry:
        """Log file creation."""
        self.files_modified.append(file_path)
        return self.log(
            level=LogLevel.INFO,
            category=LogCategory.AGENT,
            event_type=AgentEventType.FILE_CREATED.value,
            message=f"Created: {file_path}",
            data={"file_path": file_path, "description": description},
            files_affected=[file_path],
            resource=file_path,
        )

    def log_file_modified(self, file_path: str, changes: str = "") -> LogEntry:
        """Log file modification."""
        if file_path not in self.files_modified:
            self.files_modified.append(file_path)
        return self.log(
            level=LogLevel.INFO,
            category=LogCategory.AGENT,
            event_type=AgentEventType.FILE_MODIFIED.value,
            message=f"Modified: {file_path}",
            data={"file_path": file_path, "changes": changes},
            files_affected=[file_path],
            resource=file_path,
        )

    def log_file_deleted(self, file_path: str, reason: str = "") -> LogEntry:
        """Log file deletion."""
        return self.log(
            level=LogLevel.WARNING,
            category=LogCategory.AGENT,
            event_type=AgentEventType.FILE_DELETED.value,
            message=f"Deleted: {file_path}",
            data={"file_path": file_path, "reason": reason},
            files_affected=[file_path],
            resource=file_path,
        )

    # =========================================================================
    # TASK METHODS
    # =========================================================================

    def log_task_started(self, task: str, details: dict[str, Any] | None = None) -> LogEntry:
        """Log task start."""
        return self.log(
            level=LogLevel.INFO,
            category=LogCategory.AGENT,
            event_type=AgentEventType.TASK_STARTED.value,
            message=f"Started: {task}",
            data={"task": task, "details": details or {}},
            task_info={"name": task, "status": "in_progress"},
        )

    def log_task_completed(self, task: str, duration_seconds: float | None = None, result: str = "") -> LogEntry:
        """Log task completion."""
        self.tasks_completed.append(task)
        return self.log(
            level=LogLevel.INFO,
            category=LogCategory.AGENT,
            event_type=AgentEventType.TASK_COMPLETED.value,
            message=f"Completed: {task}",
            data={"task": task, "result": result},
            duration_ms=(duration_seconds * 1000) if duration_seconds else None,
            task_info={"name": task, "status": "completed"},
            outcome="success",
        )

    def log_task_failed(self, task: str, error: str, recoverable: bool = True) -> LogEntry:
        """Log task failure."""
        return self.log(
            level=LogLevel.ERROR,
            category=LogCategory.AGENT,
            event_type=AgentEventType.TASK_FAILED.value,
            message=f"Failed: {task} - {error}",
            data={"task": task, "error": error, "recoverable": recoverable},
            task_info={"name": task, "status": "failed"},
            outcome="failure",
        )

    # =========================================================================
    # GIT METHODS
    # =========================================================================

    def log_git_commit(self, commit_hash: str, message: str, files: list[str] | None = None) -> LogEntry:
        """Log git commit."""
        self.git_commits.append(commit_hash)
        return self.log(
            level=LogLevel.INFO,
            category=LogCategory.AGENT,
            event_type=AgentEventType.GIT_COMMIT.value,
            message=f"Committed: {commit_hash[:8]} - {message[:50]}",
            data={"commit_hash": commit_hash, "message": message, "files": files or []},
            git_info={"commit": commit_hash, "message": message},
            files_affected=files or [],
        )

    def log_git_push(self, branch: str, remote: str = "origin", success: bool = True, error: str = "") -> LogEntry:
        """Log git push."""
        level = LogLevel.INFO if success else LogLevel.ERROR
        return self.log(
            level=level,
            category=LogCategory.AGENT,
            event_type=AgentEventType.GIT_PUSH.value,
            message=f"Pushed to {remote}/{branch}" + ("" if success else f" (failed: {error})"),
            data={"branch": branch, "remote": remote, "success": success, "error": error},
            git_info={"branch": branch, "remote": remote, "pushed": success},
            outcome="success" if success else "failure",
        )

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def log_analysis(
        self, analysis_type: str, findings: dict[str, Any], files_analyzed: list[str] | None = None
    ) -> LogEntry:
        """Log analysis results."""
        return self.log(
            level=LogLevel.INFO,
            category=LogCategory.AGENT,
            event_type=AgentEventType.CODEBASE_ANALYZED.value,
            message=f"Analysis completed: {analysis_type}",
            data={"analysis_type": analysis_type, "findings": findings},
            files_affected=files_analyzed or [],
        )

    def log_research(self, topic: str, sources: list[str], summary: str) -> LogEntry:
        """Log research completion."""
        return self.log(
            level=LogLevel.INFO,
            category=LogCategory.AGENT,
            event_type=AgentEventType.RESEARCH_COMPLETED.value,
            message=f"Research completed: {topic}",
            data={"topic": topic, "sources": sources, "summary": summary},
        )

    def log_plan_created(self, plan_name: str, steps: list[str], file_path: str | None = None) -> LogEntry:
        """Log plan creation."""
        if file_path:
            self.files_modified.append(file_path)
        return self.log(
            level=LogLevel.INFO,
            category=LogCategory.AGENT,
            event_type=AgentEventType.PLAN_CREATED.value,
            message=f"Plan created: {plan_name}",
            data={"plan_name": plan_name, "steps": steps, "file_path": file_path},
            files_affected=[file_path] if file_path else [],
        )

    # =========================================================================
    # AUDIT METHOD (Required by AbstractLogger)
    # =========================================================================

    def audit(
        self,
        action: str,
        resource: str,
        outcome: str,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> LogEntry:
        """Log an audit trail entry for compliance.

        Args:
            action: What action was performed
            resource: What resource was affected
            outcome: Result (SUCCESS, FAILED, etc.)
            actor: Who/what performed the action (defaults to agent_id)
            details: Additional details

        Returns:
            The created LogEntry.
        """
        entry = self.log(
            level=LogLevel.AUDIT,
            category=LogCategory.AUDIT,
            event_type=action,
            message=f"{action} on {resource}: {outcome}",
            data=details or {},
            actor=actor if actor != "system" else self.agent_id,
            resource=resource,
            outcome=outcome,
        )

        # Also write to decisions log for audit trail
        self._append_to_jsonl(
            self._get_decisions_log_path(),
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_id": self.agent_id,
                "session_id": self.session_id,
                "action": action,
                "resource": resource,
                "outcome": outcome,
                "actor": actor if actor != "system" else self.agent_id,
                "details": details or {},
            },
        )

        return entry


def create_agent_logger(
    agent_id: str,
    stream_id: str | None = None,
    project_root: Path | str | None = None,
) -> AgentLogger:
    """Create an agent logger instance.

    Args:
        agent_id: Unique identifier for the agent
        stream_id: Optional work stream identifier (A-H)
        project_root: Optional project root path

    Returns:
        Configured AgentLogger instance.
    """
    root = Path(project_root) if project_root else None
    return AgentLogger(agent_id=agent_id, stream_id=stream_id, project_root=root)


def get_current_cwd() -> str:
    """Get current working directory."""
    return str(Path.cwd())


def is_external_session() -> bool:
    """Check if running from outside the project folder."""
    project_root = Path("/home/dshooter/projects/Claude_code_Quantconnect_trading_bot")
    cwd = Path.cwd()
    try:
        cwd.relative_to(project_root)
        return False
    except ValueError:
        return True
