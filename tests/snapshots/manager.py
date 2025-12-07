"""
Snapshot Testing Manager

Compare complex outputs against stored snapshots to detect regressions.

UPGRADE-015: Advanced Test Framework - Snapshot Testing
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest


@dataclass
class SnapshotInfo:
    """Metadata about a snapshot."""
    name: str
    created_at: str
    content_hash: str
    size_bytes: int


class SnapshotManager:
    """
    Manage snapshot testing for deterministic outputs.

    Usage:
        manager = SnapshotManager()

        # In test
        result = generate_complex_report()
        manager.assert_matches("report_basic", result)

        # To update snapshots, run with --snapshot-update
    """

    def __init__(
        self,
        snapshot_dir: Path | str = "tests/snapshots/data",
        update_mode: bool = False,
    ):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.update_mode = update_mode
        self._compared: list[str] = []
        self._created: list[str] = []
        self._updated: list[str] = []

    def _snapshot_path(self, name: str) -> Path:
        """Get path for a snapshot file."""
        return self.snapshot_dir / f"{name}.json"

    def _normalize(self, value: Any) -> Any:
        """
        Normalize value for deterministic comparison.

        - Sorts dict keys
        - Rounds floats to 6 decimal places
        - Converts datetime to ISO format
        - Handles dataclasses and objects with to_dict()
        """
        if isinstance(value, dict):
            return {k: self._normalize(v) for k, v in sorted(value.items())}
        elif isinstance(value, (list, tuple)):
            return [self._normalize(v) for v in value]
        elif isinstance(value, float):
            return round(value, 6)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif hasattr(value, "to_dict"):
            return self._normalize(value.to_dict())
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            return self._normalize(vars(value))
        return value

    def _serialize(self, value: Any) -> str:
        """Serialize value to JSON string."""
        normalized = self._normalize(value)
        return json.dumps(normalized, indent=2, sort_keys=True)

    def _compute_hash(self, content: str) -> str:
        """Compute content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def save_snapshot(self, name: str, value: Any) -> SnapshotInfo:
        """Save a new snapshot."""
        content = self._serialize(value)
        path = self._snapshot_path(name)
        path.write_text(content)

        info = SnapshotInfo(
            name=name,
            created_at=datetime.now().isoformat(),
            content_hash=self._compute_hash(content),
            size_bytes=len(content),
        )

        # Save metadata
        meta_path = self.snapshot_dir / f"{name}.meta.json"
        meta_path.write_text(json.dumps({
            "name": info.name,
            "created_at": info.created_at,
            "content_hash": info.content_hash,
            "size_bytes": info.size_bytes,
        }, indent=2))

        return info

    def load_snapshot(self, name: str) -> Any:
        """Load an existing snapshot."""
        path = self._snapshot_path(name)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def assert_matches(
        self,
        name: str,
        value: Any,
        message: str = "",
    ):
        """
        Assert value matches stored snapshot.

        Creates snapshot if it doesn't exist.
        Updates snapshot if update_mode is True.
        """
        self._compared.append(name)
        actual_content = self._serialize(value)
        path = self._snapshot_path(name)

        # Create new snapshot
        if not path.exists():
            self.save_snapshot(name, value)
            self._created.append(name)
            return

        # Update mode - overwrite snapshot
        if self.update_mode:
            self.save_snapshot(name, value)
            self._updated.append(name)
            return

        # Compare with existing
        expected_content = path.read_text()

        if actual_content != expected_content:
            # Generate diff
            diff = self._generate_diff(expected_content, actual_content)

            fail_msg = f"Snapshot mismatch for '{name}'"
            if message:
                fail_msg += f": {message}"
            fail_msg += f"\n\nDiff:\n{diff}"
            fail_msg += "\n\nRun with --snapshot-update to update snapshots."

            pytest.fail(fail_msg)

    def _generate_diff(self, expected: str, actual: str) -> str:
        """Generate a readable diff between two JSON strings."""
        expected_lines = expected.splitlines()
        actual_lines = actual.splitlines()

        diff_lines = []
        max_lines = max(len(expected_lines), len(actual_lines))

        for i in range(min(max_lines, 20)):  # Limit diff output
            exp = expected_lines[i] if i < len(expected_lines) else ""
            act = actual_lines[i] if i < len(actual_lines) else ""

            if exp != act:
                diff_lines.append(f"- {exp}")
                diff_lines.append(f"+ {act}")
            else:
                diff_lines.append(f"  {exp}")

        if max_lines > 20:
            diff_lines.append(f"... ({max_lines - 20} more lines)")

        return "\n".join(diff_lines)

    def get_summary(self) -> dict:
        """Get summary of snapshot operations."""
        return {
            "compared": len(self._compared),
            "created": self._created,
            "updated": self._updated,
            "failed": [],  # Would be populated by pytest
        }

    def list_snapshots(self) -> list[SnapshotInfo]:
        """List all available snapshots."""
        snapshots = []
        for path in self.snapshot_dir.glob("*.json"):
            if path.name.endswith(".meta.json"):
                continue

            meta_path = path.with_suffix(".meta.json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                snapshots.append(SnapshotInfo(**meta))
            else:
                content = path.read_text()
                snapshots.append(SnapshotInfo(
                    name=path.stem,
                    created_at="unknown",
                    content_hash=self._compute_hash(content),
                    size_bytes=len(content),
                ))

        return snapshots


# Global manager instance
_global_manager: SnapshotManager | None = None


def get_snapshot_manager() -> SnapshotManager:
    """Get global snapshot manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = SnapshotManager()
    return _global_manager


@pytest.fixture
def snapshot_manager(request):
    """
    Pytest fixture for snapshot testing.

    Supports --snapshot-update flag to update snapshots.
    """
    update_mode = request.config.getoption("--snapshot-update", default=False)
    return SnapshotManager(update_mode=update_mode)


def pytest_addoption(parser):
    """Add --snapshot-update option to pytest."""
    parser.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        help="Update snapshot files instead of comparing",
    )


def snapshot_test(name: str):
    """
    Decorator for snapshot tests.

    Usage:
        @snapshot_test("portfolio_report")
        def test_portfolio_report_format():
            return generate_report()  # Return value is compared to snapshot
    """
    def decorator(func):
        @pytest.mark.snapshot
        def wrapper(snapshot_manager, *args, **kwargs):
            result = func(*args, **kwargs)
            snapshot_manager.assert_matches(name, result)
            return result
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


__all__ = [
    "SnapshotInfo",
    "SnapshotManager",
    "get_snapshot_manager",
    "snapshot_manager",
    "snapshot_test",
]
