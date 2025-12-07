"""Tests for hierarchical memory system."""

import tempfile
import time

import pytest

from scripts.hierarchical_memory import (
    HierarchicalMemory,
    MemoryEntry,
    get_memory,
    recall_pattern,
    remember_decision,
    remember_pattern,
)


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_entry_creation(self):
        """Test basic entry creation."""
        entry = MemoryEntry(
            key="test_key",
            value="test_value",
            category="short",
            created_at=time.time(),
        )
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.category == "short"
        assert entry.access_count == 0

    def test_entry_expiration_none(self):
        """Test entry with no expiration."""
        entry = MemoryEntry(
            key="test",
            value="val",
            category="long",
            created_at=time.time(),
            expires_at=None,
        )
        assert not entry.is_expired()

    def test_entry_expiration_future(self):
        """Test entry that hasn't expired yet."""
        entry = MemoryEntry(
            key="test",
            value="val",
            category="short",
            created_at=time.time(),
            expires_at=time.time() + 3600,  # 1 hour from now
        )
        assert not entry.is_expired()

    def test_entry_expiration_past(self):
        """Test entry that has expired."""
        entry = MemoryEntry(
            key="test",
            value="val",
            category="short",
            created_at=time.time() - 3600,
            expires_at=time.time() - 1,  # 1 second ago
        )
        assert entry.is_expired()

    def test_entry_access_updates(self):
        """Test that access() updates counters."""
        entry = MemoryEntry(
            key="test",
            value="val",
            category="medium",
            created_at=time.time(),
        )
        assert entry.access_count == 0
        assert entry.last_accessed is None

        entry.access()

        assert entry.access_count == 1
        assert entry.last_accessed is not None

    def test_entry_to_dict(self):
        """Test serialization to dict."""
        now = time.time()
        entry = MemoryEntry(
            key="test",
            value={"nested": "value"},
            category="medium",
            created_at=now,
            metadata={"tag": "important"},
        )
        data = entry.to_dict()

        assert data["key"] == "test"
        assert data["value"] == {"nested": "value"}
        assert data["category"] == "medium"
        assert data["created_at"] == now
        assert data["metadata"]["tag"] == "important"

    def test_entry_from_dict(self):
        """Test deserialization from dict."""
        now = time.time()
        data = {
            "key": "restored",
            "value": "test_val",
            "category": "long",
            "created_at": now,
            "expires_at": None,
            "access_count": 5,
            "last_accessed": now - 100,
            "metadata": {"source": "test"},
        }
        entry = MemoryEntry.from_dict(data)

        assert entry.key == "restored"
        assert entry.value == "test_val"
        assert entry.access_count == 5
        assert entry.metadata["source"] == "test"


class TestHierarchicalMemory:
    """Tests for HierarchicalMemory class."""

    @pytest.fixture
    def temp_memory_dir(self):
        """Create temporary directory for memory files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def memory(self, temp_memory_dir):
        """Create HierarchicalMemory instance."""
        return HierarchicalMemory(
            memory_dir=temp_memory_dir,
            short_term_ttl=5,  # 5 seconds for testing
            short_term_capacity=3,
            medium_term_capacity=5,
            long_term_capacity=10,
        )

    # Short-term memory tests
    def test_short_term_store_and_retrieve(self, memory):
        """Test basic short-term storage."""
        memory.store_short_term("key1", "value1")
        assert memory.get_short_term("key1") == "value1"

    def test_short_term_expiration(self, memory):
        """Test short-term memory expires."""
        memory.store_short_term("expires", "soon", ttl=1)
        assert memory.get_short_term("expires") == "soon"

        time.sleep(1.5)  # Wait for expiration

        assert memory.get_short_term("expires") is None

    def test_short_term_capacity_eviction(self, memory):
        """Test LRU eviction when capacity reached."""
        # Fill to capacity
        memory.store_short_term("key1", "val1")
        memory.store_short_term("key2", "val2")
        memory.store_short_term("key3", "val3")

        # Access key1 to make it recently used
        memory.get_short_term("key1")

        # Add one more - should evict key2 (least recently used)
        memory.store_short_term("key4", "val4")

        assert memory.get_short_term("key1") == "val1"  # Still there
        assert memory.get_short_term("key3") == "val3"  # Still there
        assert memory.get_short_term("key4") == "val4"  # New entry
        assert memory.get_short_term("key2") is None  # Evicted

    def test_short_term_missing_key(self, memory):
        """Test retrieving non-existent key."""
        assert memory.get_short_term("nonexistent") is None

    # Medium-term memory tests
    def test_medium_term_store_and_retrieve(self, memory):
        """Test basic medium-term storage."""
        memory.store_medium_term("decision", "use_sqlite", rationale="simpler")
        assert memory.get_medium_term("decision") == "use_sqlite"

    def test_medium_term_persistence(self, temp_memory_dir):
        """Test medium-term memory persists to disk."""
        # Create and store
        mem1 = HierarchicalMemory(memory_dir=temp_memory_dir)
        mem1.store_medium_term("persistent", "value")

        # Create new instance - should load from disk
        mem2 = HierarchicalMemory(memory_dir=temp_memory_dir)
        assert mem2.get_medium_term("persistent") == "value"

    def test_medium_term_list(self, memory):
        """Test listing medium-term memories."""
        memory.store_medium_term("key1", "val1")
        memory.store_medium_term("key2", "val2")

        items = memory.list_medium_term()
        assert len(items) == 2
        keys = {item["key"] for item in items}
        assert keys == {"key1", "key2"}

    def test_medium_term_with_rationale(self, memory):
        """Test rationale is stored in metadata."""
        memory.store_medium_term("choice", "option_a", rationale="best performance")
        items = memory.list_medium_term()
        item = next(i for i in items if i["key"] == "choice")
        assert item["metadata"]["rationale"] == "best performance"

    # Long-term memory tests
    def test_long_term_store_and_retrieve(self, memory):
        """Test basic long-term storage."""
        memory.store_long_term("pattern", "numpy_fix", "use try/except")
        assert memory.get_long_term("pattern", "numpy_fix") == "use try/except"

    def test_long_term_persistence(self, temp_memory_dir):
        """Test long-term memory persists across instances."""
        mem1 = HierarchicalMemory(memory_dir=temp_memory_dir)
        mem1.store_long_term("learned", "best_practice", "always test first")

        mem2 = HierarchicalMemory(memory_dir=temp_memory_dir)
        assert mem2.get_long_term("learned", "best_practice") == "always test first"

    def test_long_term_search(self, memory):
        """Test searching long-term memory."""
        memory.store_long_term("fix", "import_error", "check imports")
        memory.store_long_term("fix", "type_error", "add type hints")
        memory.store_long_term("pattern", "testing", "use pytest")

        # Search by category
        results = memory.search_long_term("fix")
        assert len(results) == 2

        # Search by content
        results = memory.search_long_term("pytest")
        assert len(results) == 1

    def test_long_term_list_by_category(self, memory):
        """Test listing by category."""
        memory.store_long_term("fix", "error1", "solution1")
        memory.store_long_term("fix", "error2", "solution2")
        memory.store_long_term("pattern", "design1", "approach1")

        fixes = memory.list_long_term_by_category("fix")
        assert len(fixes) == 2

        patterns = memory.list_long_term_by_category("pattern")
        assert len(patterns) == 1

    def test_long_term_with_tags(self, memory):
        """Test tags are stored in metadata."""
        memory.store_long_term("fix", "numpy", "solution", tags=["python", "imports"])
        items = memory.list_long_term_by_category("fix")
        assert items[0]["metadata"]["tags"] == ["python", "imports"]

    # Maintenance tests
    def test_clear_session(self, memory):
        """Test clearing session memories."""
        memory.store_short_term("short", "val")
        memory.store_medium_term("medium", "val")
        memory.store_long_term("category", "long", "val")

        memory.clear_session()

        assert memory.get_short_term("short") is None
        assert memory.get_medium_term("medium") is None
        assert memory.get_long_term("category", "long") == "val"  # Preserved

    def test_get_stats(self, memory):
        """Test statistics generation."""
        memory.store_short_term("s1", "v1")
        memory.store_medium_term("m1", "v1")
        memory.store_medium_term("m2", "v2")
        memory.store_long_term("c", "l1", "v1")

        stats = memory.get_stats()

        assert stats["short_term"]["count"] == 1
        assert stats["short_term"]["capacity"] == 3
        assert stats["medium_term"]["count"] == 2
        assert stats["long_term"]["count"] == 1

    def test_recovery_context_generation(self, memory):
        """Test recovery context string generation."""
        memory.store_medium_term("decision1", "value1", rationale="reason1")
        memory.store_long_term("pattern", "fix1", "solution1")

        context = memory.generate_recovery_context()

        assert "decision1" in context
        assert "value1" in context
        assert "reason1" in context
        assert "fix1" in context
        assert "solution1" in context


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_global_memory(self):
        """Reset global memory before each test."""
        import scripts.hierarchical_memory as hm

        hm._global_memory = None
        yield
        hm._global_memory = None

    def test_get_memory_singleton(self, tmp_path, monkeypatch):
        """Test that get_memory returns singleton."""
        import scripts.hierarchical_memory as hm

        # Store original __init__ before patching to avoid recursion
        original_init = hm.HierarchicalMemory.__init__

        def patched_init(self, memory_dir="logs/memory", **kwargs):
            return original_init(self, memory_dir=str(tmp_path), **kwargs)

        monkeypatch.setattr(hm.HierarchicalMemory, "__init__", patched_init)

        mem1 = get_memory()
        mem2 = get_memory()
        assert mem1 is mem2

    def test_remember_decision(self, tmp_path, monkeypatch):
        """Test remember_decision convenience function."""
        import scripts.hierarchical_memory as hm

        # Create memory with temp dir
        hm._global_memory = HierarchicalMemory(memory_dir=str(tmp_path))

        remember_decision("arch_choice", "microservices", "scalability")
        mem = get_memory()
        assert mem.get_medium_term("arch_choice") == "microservices"

    def test_remember_and_recall_pattern(self, tmp_path):
        """Test remember_pattern and recall_pattern."""
        import scripts.hierarchical_memory as hm

        hm._global_memory = HierarchicalMemory(memory_dir=str(tmp_path))

        remember_pattern("fix", "timeout", "increase retry count", ["networking"])

        result = recall_pattern("fix", "timeout")
        assert result == "increase retry count"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def memory(self, tmp_path):
        """Create memory instance."""
        return HierarchicalMemory(memory_dir=str(tmp_path))

    def test_complex_values(self, memory):
        """Test storing complex values."""
        complex_val = {
            "nested": {"deep": [1, 2, 3]},
            "list": ["a", "b"],
            "number": 42,
        }
        memory.store_medium_term("complex", complex_val)
        retrieved = memory.get_medium_term("complex")
        assert retrieved == complex_val

    def test_unicode_values(self, memory):
        """Test storing unicode values."""
        memory.store_long_term("lang", "greeting", "こんにちは 🎉")
        assert memory.get_long_term("lang", "greeting") == "こんにちは 🎉"

    def test_empty_search(self, memory):
        """Test searching empty memory."""
        results = memory.search_long_term("nonexistent")
        assert results == []

    def test_overwrite_key(self, memory):
        """Test overwriting existing key."""
        memory.store_short_term("key", "original")
        memory.store_short_term("key", "updated")
        assert memory.get_short_term("key") == "updated"

    def test_corrupted_json_recovery(self, tmp_path):
        """Test recovery from corrupted JSON file."""
        # Write corrupted JSON
        (tmp_path / "long_term.json").write_text("{ invalid json }")

        # Should recover gracefully
        memory = HierarchicalMemory(memory_dir=str(tmp_path))
        assert memory.get_long_term("any", "key") is None

        # Should work after recovery
        memory.store_long_term("test", "key", "value")
        assert memory.get_long_term("test", "key") == "value"

    def test_none_values(self, memory):
        """Test storing None values."""
        memory.store_short_term("nullable", None)
        # None is valid, but get returns None for missing keys too
        # We need to check it's actually stored
        assert "nullable" in memory._short_term

    def test_empty_string_key(self, memory):
        """Test empty string as key."""
        memory.store_medium_term("", "empty_key_value")
        assert memory.get_medium_term("") == "empty_key_value"


class TestStrategicContextOrdering:
    """Tests for lost-in-middle optimization."""

    @pytest.fixture
    def populated_memory(self, tmp_path):
        """Create memory with entries for testing ordering."""
        memory = HierarchicalMemory(
            memory_dir=str(tmp_path),
            short_term_capacity=20,
            medium_term_capacity=50,
            long_term_capacity=100,
        )

        # Add entries with varying access patterns
        for i in range(5):
            memory.store_short_term(f"short_{i}", f"short_value_{i}")

        for i in range(10):
            memory.store_medium_term(f"medium_{i}", f"medium_value_{i}")
            # Simulate access patterns
            for _ in range(i):
                memory.get_medium_term(f"medium_{i}")

        for i in range(8):
            memory.store_long_term("pattern", f"long_{i}", f"long_value_{i}")
            # Simulate access patterns
            for _ in range(i * 2):
                memory.get_long_term("pattern", f"long_{i}")

        return memory

    def test_ordered_context_structure(self, populated_memory):
        """Test that ordered context has expected structure."""
        context = populated_memory.get_ordered_context(max_entries=20)

        assert "Strategic Context" in context
        assert "Critical Context (Beginning)" in context
        assert "Standard Context (Middle)" in context
        assert "Important Context (End)" in context

    def test_ordered_context_respects_limit(self, populated_memory):
        """Test that max_entries limit is respected."""
        context = populated_memory.get_ordered_context(max_entries=5)
        # Should have limited entries
        lines = [line for line in context.split("\n") if line.startswith("- [")]
        assert len(lines) <= 5

    def test_ordered_context_tier_filtering(self, populated_memory):
        """Test filtering by memory tier."""
        # Only long-term
        context = populated_memory.get_ordered_context(include_short=False, include_medium=False, include_long=True)
        assert "[long]" in context
        assert "[short]" not in context
        assert "[medium]" not in context

    def test_ordered_context_empty(self, tmp_path):
        """Test ordered context with empty memory."""
        memory = HierarchicalMemory(memory_dir=str(tmp_path))
        context = memory.get_ordered_context()
        assert "No memories available" in context

    def test_priority_memories(self, populated_memory):
        """Test getting priority-sorted memories."""
        priorities = populated_memory.get_priority_memories(limit=5)
        assert len(priorities) == 5

        # Higher access count should come first
        scores = []
        for p in priorities:
            scores.append(p["access_count"])
        # Verify non-increasing order (higher access first)
        assert scores == sorted(scores, reverse=True)

    def test_priority_memories_category_filter(self, populated_memory):
        """Test filtering priority memories by category."""
        # Only pattern category
        priorities = populated_memory.get_priority_memories(category="pattern", limit=10)
        for p in priorities:
            assert p["key"].startswith("pattern:")


class TestConcurrentAccess:
    """Tests for thread-safe access patterns."""

    def test_rapid_writes(self, tmp_path):
        """Test rapid sequential writes don't corrupt state."""
        memory = HierarchicalMemory(memory_dir=str(tmp_path))

        for i in range(100):
            memory.store_short_term(f"key_{i}", f"value_{i}")

        # Verify last few are accessible
        for i in range(97, 100):
            # May have been evicted due to capacity, but shouldn't error
            memory.get_short_term(f"key_{i}")

    def test_mixed_operations(self, tmp_path):
        """Test interleaved read/write operations."""
        memory = HierarchicalMemory(memory_dir=str(tmp_path))

        for i in range(50):
            memory.store_medium_term(f"m_{i}", i)
            memory.store_long_term("cat", f"l_{i}", i * 2)

            if i > 0:
                memory.get_medium_term(f"m_{i-1}")
                memory.get_long_term("cat", f"l_{i-1}")

        # Verify state is consistent
        stats = memory.get_stats()
        assert stats["medium_term"]["count"] <= 50
        assert stats["long_term"]["count"] <= 50
