#!/usr/bin/env python3
"""
Hierarchical Memory System for Autonomous Sessions

Implements three-tier memory architecture:
- Short-term: Current task context, immediate state (volatile)
- Medium-term: Session decisions, subgoal tracking (session-persistent)
- Long-term: Domain knowledge, learned patterns (cross-session persistent)

Based on research showing agents need differentiated memory for:
- Reducing repeated errors
- Maintaining consistency across tasks
- Preserving learned insights

Usage:
    from hierarchical_memory import HierarchicalMemory

    memory = HierarchicalMemory()

    # Short-term (auto-expires)
    memory.store_short_term("current_file", "models/checkpointing.py")

    # Medium-term (session-persistent)
    memory.store_medium_term("decision", "Use SQLite for checkpoints", rationale="Simpler than PostgreSQL")

    # Long-term (cross-session)
    memory.store_long_term("pattern", "numpy_import_fix", "Use try/except for optional numpy imports")
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class MemoryEntry:
    """A single memory entry with metadata."""

    key: str
    value: Any
    category: str  # short, medium, long
    created_at: float
    expires_at: float | None = None
    access_count: int = 0
    last_accessed: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def access(self):
        self.access_count += 1
        self.last_accessed = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "category": self.category,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(
            key=data["key"],
            value=data["value"],
            category=data["category"],
            created_at=data["created_at"],
            expires_at=data.get("expires_at"),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed"),
            metadata=data.get("metadata", {}),
        )


class HierarchicalMemory:
    """
    Three-tier hierarchical memory system.

    Short-term memory:
    - TTL: 30 minutes
    - Capacity: 50 entries (LRU eviction)
    - Use: Current task context, immediate state
    - Volatility: Lost on session end

    Medium-term memory:
    - TTL: Session duration (typically 8-10 hours)
    - Capacity: 200 entries
    - Use: Decisions made, subgoals completed, context for current upgrade
    - Volatility: Preserved within session, lost on session end

    Long-term memory:
    - TTL: Permanent (until explicit removal)
    - Capacity: 1000 entries
    - Use: Domain patterns, learned fixes, cross-session insights
    - Volatility: Persistent across sessions
    """

    def __init__(
        self,
        memory_dir: str = "logs/memory",
        short_term_ttl: int = 1800,  # 30 minutes
        short_term_capacity: int = 50,
        medium_term_capacity: int = 200,
        long_term_capacity: int = 1000,
    ):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # TTL settings
        self.short_term_ttl = short_term_ttl
        self.short_term_capacity = short_term_capacity
        self.medium_term_capacity = medium_term_capacity
        self.long_term_capacity = long_term_capacity

        # In-memory stores
        self._short_term: dict[str, MemoryEntry] = {}
        self._medium_term: dict[str, MemoryEntry] = {}
        self._long_term: dict[str, MemoryEntry] = {}

        # Load persistent memories
        self._load_medium_term()
        self._load_long_term()

    # ========================================================================
    # Short-term Memory (volatile, auto-expiring)
    # ========================================================================

    def store_short_term(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Store in short-term memory with automatic expiration."""
        self._cleanup_expired()

        # Evict if at capacity
        if len(self._short_term) >= self.short_term_capacity:
            self._evict_lru(self._short_term)

        entry = MemoryEntry(
            key=key,
            value=value,
            category="short",
            created_at=time.time(),
            expires_at=time.time() + (ttl or self.short_term_ttl),
            metadata=metadata or {},
        )
        self._short_term[key] = entry

    def get_short_term(self, key: str) -> Any | None:
        """Retrieve from short-term memory."""
        self._cleanup_expired()
        entry = self._short_term.get(key)
        if entry and not entry.is_expired():
            entry.access()
            return entry.value
        return None

    # ========================================================================
    # Medium-term Memory (session-persistent)
    # ========================================================================

    def store_medium_term(
        self,
        key: str,
        value: Any,
        rationale: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Store in medium-term memory (persists within session)."""
        if len(self._medium_term) >= self.medium_term_capacity:
            self._evict_lru(self._medium_term)

        meta = metadata or {}
        if rationale:
            meta["rationale"] = rationale

        entry = MemoryEntry(
            key=key,
            value=value,
            category="medium",
            created_at=time.time(),
            metadata=meta,
        )
        self._medium_term[key] = entry
        self._save_medium_term()

    def get_medium_term(self, key: str) -> Any | None:
        """Retrieve from medium-term memory."""
        entry = self._medium_term.get(key)
        if entry:
            entry.access()
            return entry.value
        return None

    def list_medium_term(self) -> list[dict[str, Any]]:
        """List all medium-term memories."""
        return [e.to_dict() for e in self._medium_term.values()]

    # ========================================================================
    # Long-term Memory (cross-session persistent)
    # ========================================================================

    def store_long_term(
        self,
        category: str,
        key: str,
        value: Any,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Store in long-term memory (persists across sessions)."""
        if len(self._long_term) >= self.long_term_capacity:
            self._evict_lru(self._long_term)

        full_key = f"{category}:{key}"
        meta = metadata or {}
        if tags:
            meta["tags"] = tags

        entry = MemoryEntry(
            key=full_key,
            value=value,
            category="long",
            created_at=time.time(),
            metadata=meta,
        )
        self._long_term[full_key] = entry
        self._save_long_term()

    def get_long_term(self, category: str, key: str) -> Any | None:
        """Retrieve from long-term memory."""
        full_key = f"{category}:{key}"
        entry = self._long_term.get(full_key)
        if entry:
            entry.access()
            return entry.value
        return None

    def search_long_term(self, query: str) -> list[dict[str, Any]]:
        """Search long-term memory by key or value content."""
        results = []
        query_lower = query.lower()
        for key, entry in self._long_term.items():
            if query_lower in key.lower() or query_lower in str(entry.value).lower():
                results.append(entry.to_dict())
        return results

    def list_long_term_by_category(self, category: str) -> list[dict[str, Any]]:
        """List long-term memories by category."""
        results = []
        prefix = f"{category}:"
        for key, entry in self._long_term.items():
            if key.startswith(prefix):
                results.append(entry.to_dict())
        return results

    # ========================================================================
    # Persistence
    # ========================================================================

    def _save_medium_term(self):
        """Persist medium-term memory to disk."""
        path = self.memory_dir / "medium_term.json"
        data = {k: v.to_dict() for k, v in self._medium_term.items()}
        path.write_text(json.dumps(data, indent=2))

    def _load_medium_term(self):
        """Load medium-term memory from disk."""
        path = self.memory_dir / "medium_term.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self._medium_term = {k: MemoryEntry.from_dict(v) for k, v in data.items()}
            except (json.JSONDecodeError, KeyError):
                self._medium_term = {}

    def _save_long_term(self):
        """Persist long-term memory to disk."""
        path = self.memory_dir / "long_term.json"
        data = {k: v.to_dict() for k, v in self._long_term.items()}
        path.write_text(json.dumps(data, indent=2))

    def _load_long_term(self):
        """Load long-term memory from disk."""
        path = self.memory_dir / "long_term.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self._long_term = {k: MemoryEntry.from_dict(v) for k, v in data.items()}
            except (json.JSONDecodeError, KeyError):
                self._long_term = {}

    # ========================================================================
    # Maintenance
    # ========================================================================

    def _cleanup_expired(self):
        """Remove expired entries from short-term memory."""
        expired_keys = [k for k, v in self._short_term.items() if v.is_expired()]
        for key in expired_keys:
            del self._short_term[key]

    def _evict_lru(self, store: dict[str, MemoryEntry]):
        """Evict least recently used entry."""
        if not store:
            return

        # Find entry with oldest access time (or creation time if never accessed)
        oldest_key = min(
            store.keys(),
            key=lambda k: store[k].last_accessed or store[k].created_at,
        )
        del store[oldest_key]

    def clear_session(self):
        """Clear session-specific memories (short and medium term)."""
        self._short_term = {}
        self._medium_term = {}
        # Clear medium-term file
        path = self.memory_dir / "medium_term.json"
        if path.exists():
            path.unlink()

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "short_term": {
                "count": len(self._short_term),
                "capacity": self.short_term_capacity,
                "utilization": len(self._short_term) / self.short_term_capacity,
            },
            "medium_term": {
                "count": len(self._medium_term),
                "capacity": self.medium_term_capacity,
                "utilization": len(self._medium_term) / self.medium_term_capacity,
            },
            "long_term": {
                "count": len(self._long_term),
                "capacity": self.long_term_capacity,
                "utilization": len(self._long_term) / self.long_term_capacity,
            },
        }

    def generate_recovery_context(self) -> str:
        """Generate context for session recovery after compaction."""
        context = f"""## Hierarchical Memory Recovery Context

**Generated**: {datetime.now().isoformat()}

### Medium-term Memory ({len(self._medium_term)} entries)
Key decisions and context from this session:

"""
        for key, entry in list(self._medium_term.items())[-10:]:  # Last 10
            context += f"- **{key}**: {entry.value}\n"
            if entry.metadata.get("rationale"):
                context += f"  - Rationale: {entry.metadata['rationale']}\n"

        context += f"""
### Long-term Memory ({len(self._long_term)} entries)
Cross-session patterns and knowledge:

"""
        # Group by category
        categories: dict[str, list] = {}
        for key, entry in self._long_term.items():
            cat = key.split(":")[0] if ":" in key else "general"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(entry)

        for cat, entries in list(categories.items())[:5]:
            context += f"#### {cat.title()}\n"
            for entry in entries[:5]:
                context += f"- {entry.key.split(':', 1)[-1]}: {entry.value}\n"

        return context

    def get_ordered_context(
        self,
        max_entries: int = 50,
        include_short: bool = True,
        include_medium: bool = True,
        include_long: bool = True,
    ) -> str:
        """
        Generate context with strategic ordering to avoid lost-in-middle effect.

        LLMs have U-shaped attention - they focus on beginning and end more than
        the middle. This method orders memories strategically:
        - BEGINNING: Most critical (P0) - recent short-term, high-access long-term
        - MIDDLE: Less critical (P1/P2) - older entries, lower access counts
        - END: Important context - key decisions, patterns

        Args:
            max_entries: Maximum entries to include
            include_short: Include short-term memories
            include_medium: Include medium-term memories
            include_long: Include long-term memories

        Returns:
            Formatted context string with strategic ordering
        """
        # Collect all entries with priority scores
        scored_entries: list[tuple[float, str, MemoryEntry]] = []

        if include_short:
            self._cleanup_expired()
            for _key, entry in self._short_term.items():
                # Short-term gets high base priority (recent = important)
                priority = 100 - (time.time() - entry.created_at) / 60  # Decay per minute
                scored_entries.append((priority, "short", entry))

        if include_medium:
            for _key, entry in self._medium_term.items():
                # Medium-term: prioritize by access count and recency
                access_score = min(entry.access_count * 5, 30)
                recency_score = 50 - (time.time() - entry.created_at) / 3600  # Decay per hour
                priority = access_score + recency_score
                scored_entries.append((priority, "medium", entry))

        if include_long:
            for _key, entry in self._long_term.items():
                # Long-term: prioritize by access count (frequently used = important)
                priority = min(entry.access_count * 10, 80)
                scored_entries.append((priority, "long", entry))

        # Sort by priority (highest first)
        scored_entries.sort(key=lambda x: x[0], reverse=True)

        # Limit entries
        scored_entries = scored_entries[:max_entries]

        if not scored_entries:
            return "## Context\n\nNo memories available.\n"

        # Strategic reordering for U-shaped attention:
        # - Top 25% → Beginning (highest priority)
        # - Bottom 25% → End (also visible)
        # - Middle 50% → Middle (less attention)
        n = len(scored_entries)
        top_quarter = n // 4 or 1
        bottom_quarter = n // 4 or 1

        beginning = scored_entries[:top_quarter]
        end = scored_entries[-(bottom_quarter):]
        middle = scored_entries[top_quarter : n - bottom_quarter]

        # Reorder: beginning + middle + end
        ordered = beginning + middle + end

        # Build context string
        context = f"""## Strategic Context (Lost-in-Middle Optimized)

**Generated**: {datetime.now().isoformat()}
**Entries**: {len(ordered)} (ordered by importance)

### Critical Context (Beginning)
"""
        for _priority, tier, entry in ordered[: len(beginning)]:
            context += f"- [{tier}] **{entry.key}**: {entry.value}\n"

        context += "\n### Standard Context (Middle)\n"
        for _priority, tier, entry in ordered[len(beginning) : len(beginning) + len(middle)]:
            context += f"- [{tier}] {entry.key}: {entry.value}\n"

        context += "\n### Important Context (End)\n"
        for _priority, tier, entry in ordered[len(beginning) + len(middle) :]:
            context += f"- [{tier}] **{entry.key}**: {entry.value}\n"

        return context

    def get_priority_memories(self, category: str | None = None, limit: int = 10) -> list[dict]:
        """
        Get memories sorted by priority/importance.

        Priority is calculated based on:
        - Access frequency (more access = more important)
        - Recency (newer = higher priority for short-term)
        - Tier (long-term patterns have inherent importance)

        Args:
            category: Filter to specific category (e.g., "fix", "pattern")
            limit: Maximum entries to return

        Returns:
            List of memory dicts sorted by priority
        """
        entries: list[tuple[float, MemoryEntry]] = []

        # Score all long-term entries (most likely to contain patterns)
        for key, entry in self._long_term.items():
            if category and not key.startswith(f"{category}:"):
                continue
            score = entry.access_count * 10 + 50  # Base score for long-term
            entries.append((score, entry))

        # Add medium-term (skip when filtering by category - medium-term has no category)
        if not category:
            for _key, entry in self._medium_term.items():
                score = entry.access_count * 5 + 30  # Lower base for medium-term
                entries.append((score, entry))

        # Sort by score
        entries.sort(key=lambda x: x[0], reverse=True)

        return [e.to_dict() for _, e in entries[:limit]]


# ============================================================================
# Convenience Functions
# ============================================================================

_global_memory: HierarchicalMemory | None = None


def get_memory() -> HierarchicalMemory:
    """Get or create global hierarchical memory instance."""
    global _global_memory
    if _global_memory is None:
        _global_memory = HierarchicalMemory()
    return _global_memory


def remember_decision(key: str, value: str, rationale: str | None = None):
    """Convenience function to store a decision in medium-term memory."""
    get_memory().store_medium_term(key, value, rationale)


def remember_pattern(category: str, key: str, value: str, tags: list[str] | None = None):
    """Convenience function to store a pattern in long-term memory."""
    get_memory().store_long_term(category, key, value, tags)


def recall_pattern(category: str, key: str) -> str | None:
    """Convenience function to recall a pattern from long-term memory."""
    return get_memory().get_long_term(category, key)


if __name__ == "__main__":
    # Demo usage
    memory = HierarchicalMemory()

    # Store some test data
    memory.store_short_term("current_task", "Implementing hierarchical memory")
    memory.store_medium_term("architecture_decision", "Use three-tier memory", "Matches research recommendations")
    memory.store_long_term("fix", "numpy_import", "Use try/except for optional numpy imports", ["python", "imports"])

    # Get stats
    print("Memory Statistics:")
    print(json.dumps(memory.get_stats(), indent=2))

    # Generate recovery context
    print("\nRecovery Context:")
    print(memory.generate_recovery_context())
