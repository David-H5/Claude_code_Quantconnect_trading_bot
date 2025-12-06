"""
Tests for Fault Tolerance Components

UPGRADE-014 Category 3: Fault Tolerance

Tests checkpointing, graceful degradation, and recovery mechanisms.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from models.checkpointing import (
    Checkpoint,
    CheckpointableMixin,
    CheckpointConfig,
    CheckpointMetadata,
    checkpoint,
    create_memory_checkpointer,
    create_sqlite_checkpointer,
    get_global_checkpointer,
    restore,
)
from models.graceful_degradation import (
    FallbackReason,
    HealthStatus,
    ServiceLevel,
    create_degradation_manager,
    create_fallback_chain,
    get_global_degradation_manager,
    get_global_fallback_chain,
)


# ============================================================================
# Checkpointing Tests
# ============================================================================


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata dataclass."""

    def test_to_dict(self):
        """Test serialization to dict."""
        meta = CheckpointMetadata(
            checkpoint_id="cp-1",
            thread_id="thread-1",
            step=1,
            created_at=datetime.now(timezone.utc),
            agent_name="test_agent",
            node_name="process",
            tags={"key": "value"},
        )

        d = meta.to_dict()

        assert d["checkpoint_id"] == "cp-1"
        assert d["thread_id"] == "thread-1"
        assert d["step"] == 1
        assert d["agent_name"] == "test_agent"
        assert d["tags"]["key"] == "value"

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "checkpoint_id": "cp-1",
            "thread_id": "thread-1",
            "step": 1,
            "created_at": "2025-12-03T12:00:00+00:00",
            "agent_name": "test",
            "node_name": "process",
        }

        meta = CheckpointMetadata.from_dict(d)

        assert meta.checkpoint_id == "cp-1"
        assert meta.thread_id == "thread-1"


class TestMemoryCheckpointSaver:
    """Tests for in-memory checkpoint saver."""

    def test_put_and_get(self):
        """Test storing and retrieving a checkpoint."""
        saver = create_memory_checkpointer()

        checkpoint = Checkpoint(
            metadata=CheckpointMetadata(
                checkpoint_id="cp-1",
                thread_id="thread-1",
                step=1,
                created_at=datetime.now(timezone.utc),
                agent_name="test",
                node_name="process",
            ),
            state={"key": "value"},
        )

        config = CheckpointConfig(thread_id="thread-1")
        saver.put(config, checkpoint)

        result = saver.get_tuple(config)

        assert result is not None
        assert result.checkpoint.metadata.checkpoint_id == "cp-1"
        assert result.checkpoint.state["key"] == "value"

    def test_get_latest(self):
        """Test getting latest checkpoint."""
        saver = create_memory_checkpointer()

        # Store multiple checkpoints
        for i in range(3):
            checkpoint = Checkpoint(
                metadata=CheckpointMetadata(
                    checkpoint_id=f"cp-{i}",
                    thread_id="thread-1",
                    step=i,
                    created_at=datetime.now(timezone.utc),
                    agent_name="test",
                    node_name="process",
                ),
                state={"step": i},
            )
            config = CheckpointConfig(thread_id="thread-1")
            saver.put(config, checkpoint)

        latest = saver.get_latest("thread-1")

        assert latest is not None
        assert latest.state["step"] == 2

    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        saver = create_memory_checkpointer()

        for i in range(5):
            checkpoint = Checkpoint(
                metadata=CheckpointMetadata(
                    checkpoint_id=f"cp-{i}",
                    thread_id="thread-1",
                    step=i,
                    created_at=datetime.now(timezone.utc),
                    agent_name="test",
                    node_name="process",
                ),
                state={},
            )
            saver.put(CheckpointConfig(thread_id="thread-1"), checkpoint)

        results = saver.list(thread_id="thread-1", limit=3)

        assert len(results) == 3

    def test_clear(self):
        """Test clearing checkpoints."""
        saver = create_memory_checkpointer()

        checkpoint = Checkpoint(
            metadata=CheckpointMetadata(
                checkpoint_id="cp-1",
                thread_id="thread-1",
                step=1,
                created_at=datetime.now(timezone.utc),
                agent_name="test",
                node_name="process",
            ),
            state={},
        )
        saver.put(CheckpointConfig(thread_id="thread-1"), checkpoint)

        count = saver.clear()

        assert count == 1
        assert saver.get_latest("thread-1") is None

    def test_pending_writes(self):
        """Test storing pending writes."""
        saver = create_memory_checkpointer()

        checkpoint = Checkpoint(
            metadata=CheckpointMetadata(
                checkpoint_id="cp-1",
                thread_id="thread-1",
                step=1,
                created_at=datetime.now(timezone.utc),
                agent_name="test",
                node_name="process",
            ),
            state={},
        )
        config = CheckpointConfig(thread_id="thread-1")
        saver.put(config, checkpoint)

        # Add pending writes
        saver.put_writes(config, [{"action": "test"}])

        result = saver.get_tuple(config)
        assert len(result.checkpoint.pending_writes) == 1


class TestSQLiteCheckpointSaver:
    """Tests for SQLite checkpoint saver."""

    def test_put_and_get(self):
        """Test storing and retrieving a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            saver = create_sqlite_checkpointer(str(db_path))

            checkpoint = Checkpoint(
                metadata=CheckpointMetadata(
                    checkpoint_id="cp-1",
                    thread_id="thread-1",
                    step=1,
                    created_at=datetime.now(timezone.utc),
                    agent_name="test",
                    node_name="process",
                ),
                state={"key": "value"},
            )

            config = CheckpointConfig(thread_id="thread-1")
            saver.put(config, checkpoint)

            result = saver.get_tuple(config)

            assert result is not None
            assert result.checkpoint.state["key"] == "value"

    def test_persistence(self):
        """Test that checkpoints persist across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # First instance - write
            saver1 = create_sqlite_checkpointer(str(db_path))
            checkpoint = Checkpoint(
                metadata=CheckpointMetadata(
                    checkpoint_id="cp-1",
                    thread_id="thread-1",
                    step=1,
                    created_at=datetime.now(timezone.utc),
                    agent_name="test",
                    node_name="process",
                ),
                state={"persistent": True},
            )
            saver1.put(CheckpointConfig(thread_id="thread-1"), checkpoint)

            # Second instance - read
            saver2 = create_sqlite_checkpointer(str(db_path))
            result = saver2.get_tuple(CheckpointConfig(thread_id="thread-1"))

            assert result is not None
            assert result.checkpoint.state["persistent"] is True


class TestCheckpointableMixin:
    """Tests for CheckpointableMixin."""

    def test_save_and_restore(self):
        """Test saving and restoring checkpoints via mixin."""

        class TestAgent(CheckpointableMixin):
            def __init__(self, name: str):
                self.name = name

        agent = TestAgent("test_agent")
        saver = create_memory_checkpointer()
        agent.init_checkpointing(saver, thread_id="test-thread")

        # Save checkpoint
        cp_id = agent.save_checkpoint("before_action", {"counter": 42})
        assert cp_id != ""

        # Restore checkpoint
        state = agent.restore_checkpoint()
        assert state is not None
        assert state["counter"] == 42

    def test_checkpoint_history(self):
        """Test getting checkpoint history."""

        class TestAgent(CheckpointableMixin):
            def __init__(self, name: str):
                self.name = name

        agent = TestAgent("test_agent")
        saver = create_memory_checkpointer()
        agent.init_checkpointing(saver, thread_id="test-thread")

        # Save multiple checkpoints
        for i in range(5):
            agent.save_checkpoint(f"step_{i}", {"step": i})

        history = agent.get_checkpoint_history(limit=3)
        assert len(history) == 3


class TestConvenienceFunctions:
    """Tests for checkpoint convenience functions."""

    def test_checkpoint_and_restore(self):
        """Test global checkpoint and restore functions."""
        # Save checkpoint
        cp_id = checkpoint(
            thread_id="test-thread",
            agent_name="test",
            node_name="process",
            state={"test": True},
        )
        assert cp_id != ""

        # Restore
        state = restore("test-thread")
        assert state is not None
        assert state["test"] is True


# ============================================================================
# Graceful Degradation Tests
# ============================================================================


class TestModelFallbackChain:
    """Tests for model fallback chain."""

    def test_initial_model(self):
        """Test initial model is first in chain."""
        chain = create_fallback_chain()
        assert chain.current_model.tier == 1  # Premium model

    def test_record_success(self):
        """Test recording successful call."""
        chain = create_fallback_chain()

        chain.record_success("claude-3-opus", latency_ms=1500.0)

        health = chain.get_health_summary()
        assert health["claude-3-opus"]["status"] == "healthy"
        assert health["claude-3-opus"]["latency_ms"] == 1500.0

    def test_fallback_on_failure(self):
        """Test automatic fallback on failures."""
        chain = create_fallback_chain()
        initial_model = chain.current_model.name

        # Record multiple failures
        for _ in range(3):
            chain.record_failure(
                initial_model,
                reason=FallbackReason.ERROR,
                error="API error",
            )

        # Should have fallen back
        assert chain.current_model.name != initial_model

    def test_recovery_upgrade(self):
        """Test recovery back to better model."""
        chain = create_fallback_chain()

        # Force fallback
        for _ in range(3):
            chain.record_failure(
                chain.current_model.name,
                reason=FallbackReason.ERROR,
            )

        # Record success on original model to trigger recovery
        chain.record_success("claude-3-opus", latency_ms=1000.0)

        # Should upgrade back
        assert chain.current_model.name == "claude-3-opus"

    def test_health_summary(self):
        """Test health summary generation."""
        chain = create_fallback_chain()

        chain.record_success("claude-3-opus", latency_ms=1500.0)
        chain.record_failure("gpt-4o", FallbackReason.TIMEOUT)

        summary = chain.get_health_summary()

        assert "claude-3-opus" in summary
        assert "gpt-4o" in summary
        assert summary["gpt-4o"]["failures"] == 1

    def test_fallback_history(self):
        """Test fallback history tracking."""
        chain = create_fallback_chain()

        # Trigger fallbacks
        for _ in range(3):
            chain.record_failure(
                chain.current_model.name,
                reason=FallbackReason.RATE_LIMIT,
            )

        history = chain.get_fallback_history(limit=10)
        assert len(history) >= 1
        assert history[0]["reason"] == "rate_limit"

    def test_reset(self):
        """Test resetting chain to initial state."""
        chain = create_fallback_chain()

        # Trigger fallback
        for _ in range(3):
            chain.record_failure(
                chain.current_model.name,
                reason=FallbackReason.ERROR,
            )

        chain.reset()

        assert chain._current_index == 0
        health = chain.get_health_summary()
        for model_health in health.values():
            assert model_health["status"] == "healthy"


class TestServiceDegradationManager:
    """Tests for service degradation manager."""

    def test_register_service(self):
        """Test registering services."""
        manager = create_degradation_manager()

        manager.register_service("llm")
        manager.register_service("database")

        assert manager.service_level == ServiceLevel.FULL

    def test_degradation_on_unhealthy(self):
        """Test service level degrades with unhealthy services."""
        manager = create_degradation_manager()

        for i in range(10):
            manager.register_service(f"service_{i}")

        # Mark 60% as unhealthy
        for i in range(6):
            manager.update_service_health(
                f"service_{i}",
                HealthStatus.UNHEALTHY,
            )

        assert manager.service_level == ServiceLevel.DEGRADED

    def test_minimal_service_level(self):
        """Test minimal service level with most services down."""
        manager = create_degradation_manager()

        for i in range(10):
            manager.register_service(f"service_{i}")

        # Mark 90% as unhealthy
        for i in range(9):
            manager.update_service_health(
                f"service_{i}",
                HealthStatus.UNHEALTHY,
            )

        assert manager.service_level == ServiceLevel.MINIMAL

    def test_feature_availability(self):
        """Test feature availability at different levels."""
        manager = create_degradation_manager()

        manager.register_service("llm")

        # Full service
        features = manager.get_feature_availability()
        assert all(features.values())

        # Degrade
        manager.update_service_health("llm", HealthStatus.UNHEALTHY)

        features = manager.get_feature_availability()
        # Some features should be unavailable
        assert not features["multi_agent_consensus"]

    def test_callback_notification(self):
        """Test degradation callback notification."""
        manager = create_degradation_manager()
        notifications = []

        def on_degradation(level: ServiceLevel):
            notifications.append(level)

        manager.register_callback(on_degradation)

        manager.register_service("service_1")
        manager.register_service("service_2")

        # Trigger degradation
        manager.update_service_health("service_1", HealthStatus.UNHEALTHY)
        manager.update_service_health("service_2", HealthStatus.UNHEALTHY)

        assert len(notifications) > 0


class TestGlobalInstances:
    """Tests for global singleton instances."""

    def test_global_fallback_chain(self):
        """Test global fallback chain is singleton."""
        chain1 = get_global_fallback_chain()
        chain2 = get_global_fallback_chain()
        assert chain1 is chain2

    def test_global_degradation_manager(self):
        """Test global degradation manager is singleton."""
        manager1 = get_global_degradation_manager()
        manager2 = get_global_degradation_manager()
        assert manager1 is manager2

    def test_global_checkpointer(self):
        """Test global checkpointer is singleton."""
        cp1 = get_global_checkpointer()
        cp2 = get_global_checkpointer()
        assert cp1 is cp2
