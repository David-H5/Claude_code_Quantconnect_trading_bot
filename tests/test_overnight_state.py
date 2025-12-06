"""Tests for unified overnight state management."""

import json
import threading
from pathlib import Path

import pytest

from utils.overnight_state import OvernightState, OvernightStateManager


class TestOvernightStateManager:
    """Tests for OvernightStateManager."""

    def test_load_creates_default_state(self, tmp_path: Path) -> None:
        """Test that loading from non-existent file creates default state."""
        state_file = tmp_path / "state.json"
        mgr = OvernightStateManager(state_file=state_file)
        state = mgr.load()

        assert isinstance(state, OvernightState)
        assert state.session_id == ""
        assert state.continuation_count == 0

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Test that save and load preserve state."""
        state_file = tmp_path / "state.json"
        mgr = OvernightStateManager(state_file=state_file)

        # Create and save state
        state = OvernightState(session_id="test-123", goal="Test goal")
        mgr.save(state)

        # Load and verify
        loaded = mgr.load()
        assert loaded.session_id == "test-123"
        assert loaded.goal == "Test goal"

    def test_state_file_created(self, tmp_path: Path) -> None:
        """Test that state file is created on save."""
        state_file = tmp_path / "state.json"
        mgr = OvernightStateManager(state_file=state_file)

        state = OvernightState(session_id="test")
        mgr.save(state)

        assert state_file.exists()
        content = json.loads(state_file.read_text())
        assert content["session_id"] == "test"

    def test_file_locking_concurrent_access(self, tmp_path: Path) -> None:
        """Verify concurrent access is safe with file locking."""
        state_file = tmp_path / "state.json"
        mgr = OvernightStateManager(state_file=state_file)
        errors: list[Exception] = []

        def writer(n: int) -> None:
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

    def test_record_continuation(self, tmp_path: Path) -> None:
        """Test recording continuation events."""
        state_file = tmp_path / "state.json"
        mgr = OvernightStateManager(state_file=state_file)

        # Record a continuation
        mgr.record_continuation(
            reason="Test continuation",
            completion_pct=50.0,
            pending_tasks=10,
        )

        state = mgr.load()
        assert state.continuation_count == 1
        assert len(state.continuations) == 1
        assert state.continuations[0]["reason"] == "Test continuation"


class TestOvernightState:
    """Tests for OvernightState dataclass."""

    def test_default_values(self) -> None:
        """Test default state values."""
        state = OvernightState()
        assert state.session_id == ""
        assert state.goal == ""
        assert state.restart_count == 0
        assert state.continuation_count == 0
        assert state.completed_tasks == []
        assert state.blockers == []

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        state = OvernightState(session_id="test", goal="Test goal")
        d = state.to_dict()

        assert isinstance(d, dict)
        assert d["session_id"] == "test"
        assert d["goal"] == "Test goal"

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "session_id": "test-456",
            "goal": "Another goal",
            "restart_count": 2,
        }
        state = OvernightState.from_dict(data)

        assert state.session_id == "test-456"
        assert state.goal == "Another goal"
        assert state.restart_count == 2

    def test_from_dict_ignores_unknown_fields(self) -> None:
        """Test that unknown fields are ignored."""
        data = {
            "session_id": "test",
            "unknown_field": "value",
            "another_unknown": 123,
        }
        state = OvernightState.from_dict(data)
        assert state.session_id == "test"
