"""
Agent Checkpointing System

LangGraph-style checkpointing for fault tolerance in agent workflows.
Enables state persistence, recovery, and resumption from failures.

UPGRADE-014 Category 3: Fault Tolerance

Features:
- Save state at critical decision points
- Resume from checkpoints after failures
- Thread-based conversation separation
- Multiple storage backends (memory, file, future: Redis/Postgres)

QuantConnect Compatible: Yes
- Non-blocking checkpoint operations
- Configurable storage backends
- Memory-efficient serialization
"""

import json
import logging
import pickle
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""

    checkpoint_id: str
    thread_id: str
    step: int
    created_at: datetime
    agent_name: str
    node_name: str
    parent_checkpoint_id: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "thread_id": self.thread_id,
            "step": self.step,
            "created_at": self.created_at.isoformat(),
            "agent_name": self.agent_name,
            "node_name": self.node_name,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            thread_id=data["thread_id"],
            step=data["step"],
            created_at=datetime.fromisoformat(data["created_at"]),
            agent_name=data["agent_name"],
            node_name=data["node_name"],
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
            tags=data.get("tags", {}),
        )


@dataclass
class Checkpoint:
    """Complete checkpoint with state and metadata."""

    metadata: CheckpointMetadata
    state: dict[str, Any]
    pending_writes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "state": self.state,
            "pending_writes": self.pending_writes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """Create from dictionary."""
        return cls(
            metadata=CheckpointMetadata.from_dict(data["metadata"]),
            state=data["state"],
            pending_writes=data.get("pending_writes", []),
        )


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint operations."""

    thread_id: str
    checkpoint_ns: str = ""  # Namespace for multi-tenant
    parent_checkpoint_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "thread_id": self.thread_id,
            "checkpoint_ns": self.checkpoint_ns,
            "parent_checkpoint_id": self.parent_checkpoint_id,
        }


@dataclass
class CheckpointTuple:
    """Tuple of checkpoint with config for retrieval."""

    config: CheckpointConfig
    checkpoint: Checkpoint
    parent_config: CheckpointConfig | None = None


# ============================================================================
# Abstract Base Classes
# ============================================================================


class BaseCheckpointSaver(ABC):
    """
    Abstract base class for checkpoint storage backends.

    Implements the LangGraph checkpointer interface:
    - .put(): Store checkpoint with config and metadata
    - .put_writes(): Store intermediate/pending writes
    - .get_tuple(): Fetch checkpoint for given thread_id
    - .list(): List checkpoints matching filter criteria
    """

    @abstractmethod
    def put(
        self,
        config: CheckpointConfig,
        checkpoint: Checkpoint,
    ) -> CheckpointConfig:
        """Store a checkpoint."""
        pass

    @abstractmethod
    def put_writes(
        self,
        config: CheckpointConfig,
        writes: list[dict[str, Any]],
    ) -> None:
        """Store pending writes for a checkpoint."""
        pass

    @abstractmethod
    def get_tuple(
        self,
        config: CheckpointConfig,
    ) -> CheckpointTuple | None:
        """Retrieve the latest checkpoint for a thread."""
        pass

    @abstractmethod
    def list(
        self,
        thread_id: str | None = None,
        agent_name: str | None = None,
        limit: int = 100,
        before: datetime | None = None,
    ) -> list[CheckpointTuple]:
        """List checkpoints matching filter criteria."""
        pass

    def get_latest(self, thread_id: str) -> Checkpoint | None:
        """Convenience method to get latest checkpoint for thread."""
        config = CheckpointConfig(thread_id=thread_id)
        result = self.get_tuple(config)
        return result.checkpoint if result else None


# ============================================================================
# In-Memory Checkpointer
# ============================================================================


class MemoryCheckpointSaver(BaseCheckpointSaver):
    """
    In-memory checkpoint storage for development and testing.

    Not persistent across restarts - use for testing or short-lived agents.
    """

    def __init__(self, max_checkpoints: int = 1000):
        """Initialize memory saver."""
        self.max_checkpoints = max_checkpoints
        self._checkpoints: dict[str, list[Checkpoint]] = {}  # thread_id -> checkpoints
        self._lock = Lock()

    def put(
        self,
        config: CheckpointConfig,
        checkpoint: Checkpoint,
    ) -> CheckpointConfig:
        """Store a checkpoint in memory."""
        with self._lock:
            thread_id = config.thread_id

            if thread_id not in self._checkpoints:
                self._checkpoints[thread_id] = []

            self._checkpoints[thread_id].append(checkpoint)

            # Trim to max size
            if len(self._checkpoints[thread_id]) > self.max_checkpoints:
                self._checkpoints[thread_id] = self._checkpoints[thread_id][-self.max_checkpoints :]

            logger.debug(f"Stored checkpoint {checkpoint.metadata.checkpoint_id} " f"for thread {thread_id}")

            return CheckpointConfig(
                thread_id=thread_id,
                checkpoint_ns=config.checkpoint_ns,
                parent_checkpoint_id=checkpoint.metadata.checkpoint_id,
            )

    def put_writes(
        self,
        config: CheckpointConfig,
        writes: list[dict[str, Any]],
    ) -> None:
        """Store pending writes for latest checkpoint."""
        with self._lock:
            thread_id = config.thread_id
            if self._checkpoints.get(thread_id):
                self._checkpoints[thread_id][-1].pending_writes.extend(writes)

    def get_tuple(
        self,
        config: CheckpointConfig,
    ) -> CheckpointTuple | None:
        """Get latest checkpoint for thread."""
        with self._lock:
            thread_id = config.thread_id

            if thread_id not in self._checkpoints or not self._checkpoints[thread_id]:
                return None

            checkpoint = self._checkpoints[thread_id][-1]

            parent_config = None
            if len(self._checkpoints[thread_id]) > 1:
                parent_checkpoint = self._checkpoints[thread_id][-2]
                parent_config = CheckpointConfig(
                    thread_id=thread_id,
                    parent_checkpoint_id=parent_checkpoint.metadata.checkpoint_id,
                )

            return CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                parent_config=parent_config,
            )

    def list(
        self,
        thread_id: str | None = None,
        agent_name: str | None = None,
        limit: int = 100,
        before: datetime | None = None,
    ) -> list[CheckpointTuple]:
        """List checkpoints matching criteria."""
        results = []

        with self._lock:
            threads = [thread_id] if thread_id else list(self._checkpoints.keys())

            for tid in threads:
                if tid not in self._checkpoints:
                    continue

                for checkpoint in reversed(self._checkpoints[tid]):
                    # Apply filters
                    if agent_name and checkpoint.metadata.agent_name != agent_name:
                        continue
                    if before and checkpoint.metadata.created_at >= before:
                        continue

                    results.append(
                        CheckpointTuple(
                            config=CheckpointConfig(thread_id=tid),
                            checkpoint=checkpoint,
                        )
                    )

                    if len(results) >= limit:
                        break

                if len(results) >= limit:
                    break

        return results

    def clear(self, thread_id: str | None = None) -> int:
        """Clear checkpoints. Returns number cleared."""
        with self._lock:
            if thread_id:
                count = len(self._checkpoints.get(thread_id, []))
                self._checkpoints.pop(thread_id, None)
            else:
                count = sum(len(v) for v in self._checkpoints.values())
                self._checkpoints.clear()
            return count


# ============================================================================
# SQLite Checkpointer (File-based persistence)
# ============================================================================


class SQLiteCheckpointSaver(BaseCheckpointSaver):
    """
    SQLite-based checkpoint storage for local persistence.

    Good for local development and single-instance deployments.
    """

    def __init__(self, db_path: str = "checkpoints.db"):
        """Initialize SQLite saver."""
        self.db_path = Path(db_path)
        self._lock = Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        checkpoint_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        step INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        agent_name TEXT NOT NULL,
                        node_name TEXT NOT NULL,
                        parent_checkpoint_id TEXT,
                        tags TEXT,
                        state BLOB NOT NULL,
                        pending_writes BLOB
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_thread_step
                    ON checkpoints(thread_id, step DESC)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_agent_name
                    ON checkpoints(agent_name)
                """)
                conn.commit()
            finally:
                conn.close()

    def put(
        self,
        config: CheckpointConfig,
        checkpoint: Checkpoint,
    ) -> CheckpointConfig:
        """Store a checkpoint in SQLite."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO checkpoints
                    (checkpoint_id, thread_id, step, created_at, agent_name,
                     node_name, parent_checkpoint_id, tags, state, pending_writes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        checkpoint.metadata.checkpoint_id,
                        checkpoint.metadata.thread_id,
                        checkpoint.metadata.step,
                        checkpoint.metadata.created_at.isoformat(),
                        checkpoint.metadata.agent_name,
                        checkpoint.metadata.node_name,
                        checkpoint.metadata.parent_checkpoint_id,
                        json.dumps(checkpoint.metadata.tags),
                        pickle.dumps(checkpoint.state),
                        pickle.dumps(checkpoint.pending_writes),
                    ),
                )
                conn.commit()

                logger.debug(f"Stored checkpoint {checkpoint.metadata.checkpoint_id} " f"to SQLite")

                return CheckpointConfig(
                    thread_id=config.thread_id,
                    checkpoint_ns=config.checkpoint_ns,
                    parent_checkpoint_id=checkpoint.metadata.checkpoint_id,
                )
            finally:
                conn.close()

    def put_writes(
        self,
        config: CheckpointConfig,
        writes: list[dict[str, Any]],
    ) -> None:
        """Store pending writes for latest checkpoint."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()

                # Get latest checkpoint for thread
                cursor.execute(
                    """
                    SELECT checkpoint_id, pending_writes FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY step DESC LIMIT 1
                """,
                    (config.thread_id,),
                )
                row = cursor.fetchone()
                if row:
                    checkpoint_id, existing_writes_blob = row
                    existing_writes = pickle.loads(existing_writes_blob) if existing_writes_blob else []
                    existing_writes.extend(writes)

                    cursor.execute(
                        """
                        UPDATE checkpoints SET pending_writes = ?
                        WHERE checkpoint_id = ?
                    """,
                        (pickle.dumps(existing_writes), checkpoint_id),
                    )
                    conn.commit()
            finally:
                conn.close()

    def get_tuple(
        self,
        config: CheckpointConfig,
    ) -> CheckpointTuple | None:
        """Get latest checkpoint for thread."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT checkpoint_id, thread_id, step, created_at, agent_name,
                           node_name, parent_checkpoint_id, tags, state, pending_writes
                    FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY step DESC LIMIT 1
                """,
                    (config.thread_id,),
                )
                row = cursor.fetchone()

                if not row:
                    return None

                metadata = CheckpointMetadata(
                    checkpoint_id=row[0],
                    thread_id=row[1],
                    step=row[2],
                    created_at=datetime.fromisoformat(row[3]),
                    agent_name=row[4],
                    node_name=row[5],
                    parent_checkpoint_id=row[6],
                    tags=json.loads(row[7]) if row[7] else {},
                )

                checkpoint = Checkpoint(
                    metadata=metadata,
                    state=pickle.loads(row[8]),
                    pending_writes=pickle.loads(row[9]) if row[9] else [],
                )

                return CheckpointTuple(
                    config=config,
                    checkpoint=checkpoint,
                )
            finally:
                conn.close()

    def list(
        self,
        thread_id: str | None = None,
        agent_name: str | None = None,
        limit: int = 100,
        before: datetime | None = None,
    ) -> list[CheckpointTuple]:
        """List checkpoints matching criteria."""
        results = []

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()

                query = """
                    SELECT checkpoint_id, thread_id, step, created_at, agent_name,
                           node_name, parent_checkpoint_id, tags, state, pending_writes
                    FROM checkpoints WHERE 1=1
                """
                params: list[Any] = []

                if thread_id:
                    query += " AND thread_id = ?"
                    params.append(thread_id)
                if agent_name:
                    query += " AND agent_name = ?"
                    params.append(agent_name)
                if before:
                    query += " AND created_at < ?"
                    params.append(before.isoformat())

                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)

                for row in cursor.fetchall():
                    metadata = CheckpointMetadata(
                        checkpoint_id=row[0],
                        thread_id=row[1],
                        step=row[2],
                        created_at=datetime.fromisoformat(row[3]),
                        agent_name=row[4],
                        node_name=row[5],
                        parent_checkpoint_id=row[6],
                        tags=json.loads(row[7]) if row[7] else {},
                    )

                    checkpoint = Checkpoint(
                        metadata=metadata,
                        state=pickle.loads(row[8]),
                        pending_writes=pickle.loads(row[9]) if row[9] else [],
                    )

                    results.append(
                        CheckpointTuple(
                            config=CheckpointConfig(thread_id=row[1]),
                            checkpoint=checkpoint,
                        )
                    )

                return results
            finally:
                conn.close()


# ============================================================================
# Agent Checkpointer Mixin
# ============================================================================


class CheckpointableMixin:
    """
    Mixin for agents to support checkpointing.

    Usage:
        class MyAgent(TradingAgent, CheckpointableMixin):
            def __init__(self, ...):
                super().__init__(...)
                self.init_checkpointing(saver)

            def process(self, message):
                self.save_checkpoint("before_process", {"message": message})
                result = self._do_process(message)
                self.save_checkpoint("after_process", {"result": result})
                return result
    """

    _checkpoint_saver: BaseCheckpointSaver | None = None
    _checkpoint_thread_id: str | None = None
    _checkpoint_step: int = 0

    def init_checkpointing(
        self,
        saver: BaseCheckpointSaver,
        thread_id: str | None = None,
    ) -> None:
        """Initialize checkpointing for this agent."""
        self._checkpoint_saver = saver
        self._checkpoint_thread_id = thread_id or str(uuid4())
        self._checkpoint_step = 0

    def save_checkpoint(
        self,
        node_name: str,
        state: dict[str, Any],
        tags: dict[str, str] | None = None,
    ) -> str:
        """Save a checkpoint and return checkpoint ID."""
        if not self._checkpoint_saver:
            logger.warning("Checkpointing not initialized")
            return ""

        self._checkpoint_step += 1

        # Get agent name (assumes self has a 'name' attribute)
        agent_name = getattr(self, "name", self.__class__.__name__)

        checkpoint = Checkpoint(
            metadata=CheckpointMetadata(
                checkpoint_id=str(uuid4()),
                thread_id=self._checkpoint_thread_id or "",
                step=self._checkpoint_step,
                created_at=datetime.now(timezone.utc),
                agent_name=agent_name,
                node_name=node_name,
                tags=tags or {},
            ),
            state=state,
        )

        config = CheckpointConfig(thread_id=self._checkpoint_thread_id or "")
        self._checkpoint_saver.put(config, checkpoint)

        return checkpoint.metadata.checkpoint_id

    def restore_checkpoint(
        self,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Restore from checkpoint. Returns state dict or None."""
        if not self._checkpoint_saver:
            logger.warning("Checkpointing not initialized")
            return None

        config = CheckpointConfig(thread_id=self._checkpoint_thread_id or "")
        result = self._checkpoint_saver.get_tuple(config)

        if result:
            checkpoint = result.checkpoint
            self._checkpoint_step = checkpoint.metadata.step
            return checkpoint.state

        return None

    def get_checkpoint_history(self, limit: int = 10) -> list[CheckpointMetadata]:
        """Get recent checkpoint history for this thread."""
        if not self._checkpoint_saver:
            return []

        results = self._checkpoint_saver.list(thread_id=self._checkpoint_thread_id, limit=limit)
        return [r.checkpoint.metadata for r in results]


# ============================================================================
# Factory Functions
# ============================================================================


def create_memory_checkpointer(max_checkpoints: int = 1000) -> MemoryCheckpointSaver:
    """Create an in-memory checkpoint saver."""
    return MemoryCheckpointSaver(max_checkpoints=max_checkpoints)


def create_sqlite_checkpointer(db_path: str = "checkpoints.db") -> SQLiteCheckpointSaver:
    """Create a SQLite checkpoint saver."""
    return SQLiteCheckpointSaver(db_path=db_path)


# Global checkpointer instance
_global_checkpointer: BaseCheckpointSaver | None = None
_global_lock = Lock()


def get_global_checkpointer() -> BaseCheckpointSaver:
    """Get or create the global checkpointer."""
    global _global_checkpointer

    if _global_checkpointer is None:
        with _global_lock:
            if _global_checkpointer is None:
                _global_checkpointer = create_memory_checkpointer()

    return _global_checkpointer


def set_global_checkpointer(checkpointer: BaseCheckpointSaver) -> None:
    """Set the global checkpointer instance."""
    global _global_checkpointer
    with _global_lock:
        _global_checkpointer = checkpointer


# ============================================================================
# Convenience Functions
# ============================================================================


def checkpoint(
    thread_id: str,
    agent_name: str,
    node_name: str,
    state: dict[str, Any],
    tags: dict[str, str] | None = None,
) -> str:
    """Convenience function to save a checkpoint."""
    saver = get_global_checkpointer()

    # Get next step number
    config = CheckpointConfig(thread_id=thread_id)
    existing = saver.get_tuple(config)
    step = existing.checkpoint.metadata.step + 1 if existing else 1

    cp = Checkpoint(
        metadata=CheckpointMetadata(
            checkpoint_id=str(uuid4()),
            thread_id=thread_id,
            step=step,
            created_at=datetime.now(timezone.utc),
            agent_name=agent_name,
            node_name=node_name,
            tags=tags or {},
        ),
        state=state,
    )

    saver.put(config, cp)
    return cp.metadata.checkpoint_id


def restore(thread_id: str) -> dict[str, Any] | None:
    """Convenience function to restore from latest checkpoint."""
    saver = get_global_checkpointer()
    config = CheckpointConfig(thread_id=thread_id)
    result = saver.get_tuple(config)
    return result.checkpoint.state if result else None
