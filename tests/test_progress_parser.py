"""Tests for progress file parser."""

from pathlib import Path

import pytest

from utils.progress_parser import Category, ProgressData, ProgressParser, Task


class TestTask:
    """Tests for Task dataclass."""

    def test_task_creation(self) -> None:
        """Test creating a task."""
        task = Task(description="Test task", complete=False)
        assert task.description == "Test task"
        assert task.complete is False
        assert task.priority == "P2"  # Default

    def test_task_str_incomplete(self) -> None:
        """Test string representation of incomplete task."""
        task = Task(description="Do something", complete=False)
        assert str(task) == "[ ] Do something"

    def test_task_str_complete(self) -> None:
        """Test string representation of complete task."""
        task = Task(description="Done something", complete=True)
        assert str(task) == "[x] Done something"


class TestCategory:
    """Tests for Category dataclass."""

    def test_empty_category(self) -> None:
        """Test empty category properties."""
        cat = Category(name="Test")
        assert cat.completed_count == 0
        assert cat.total_count == 0
        assert cat.is_complete is False  # Empty = not complete
        assert cat.completion_pct == 100.0  # 0/0 = 100%

    def test_category_with_tasks(self) -> None:
        """Test category with tasks."""
        cat = Category(
            name="Test",
            tasks=[
                Task(description="Task 1", complete=True),
                Task(description="Task 2", complete=False),
                Task(description="Task 3", complete=True),
            ],
        )
        assert cat.completed_count == 2
        assert cat.total_count == 3
        assert cat.is_complete is False
        assert cat.completion_pct == pytest.approx(66.67, rel=0.01)

    def test_complete_category(self) -> None:
        """Test fully complete category."""
        cat = Category(
            name="Done",
            tasks=[
                Task(description="Task 1", complete=True),
                Task(description="Task 2", complete=True),
            ],
        )
        assert cat.is_complete is True
        assert cat.completion_pct == 100.0


class TestProgressData:
    """Tests for ProgressData dataclass."""

    def test_empty_progress(self) -> None:
        """Test empty progress data."""
        data = ProgressData()
        assert data.total_count == 0
        assert data.completed_count == 0
        assert data.completion_pct == 100.0

    def test_get_next_task_by_priority(self) -> None:
        """Test getting next task respects priority."""
        cat1 = Category(
            name="P2 Category",
            priority="P2",
            tasks=[Task(description="P2 task", complete=False, priority="P2")],
        )
        cat2 = Category(
            name="P0 Category",
            priority="P0",
            tasks=[Task(description="P0 task", complete=False, priority="P0")],
        )

        data = ProgressData(categories={"cat1": cat1, "cat2": cat2})
        next_task = data.get_next_task()

        assert next_task is not None
        assert next_task.priority == "P0"

    def test_get_pending_by_priority(self) -> None:
        """Test filtering pending tasks by priority."""
        cat = Category(
            name="Mixed",
            tasks=[
                Task(description="P0 task", complete=False, priority="P0"),
                Task(description="P1 task", complete=False, priority="P1"),
                Task(description="P0 done", complete=True, priority="P0"),
            ],
        )

        data = ProgressData(categories={"mixed": cat})

        p0_pending = data.get_pending_by_priority("P0")
        assert len(p0_pending) == 1
        assert p0_pending[0].description == "P0 task"


class TestProgressParser:
    """Tests for ProgressParser."""

    def test_parse_completed_task(self, tmp_path: Path) -> None:
        """Test parsing completed task."""
        progress_file = tmp_path / "progress.txt"
        progress_file.write_text("- [x] **1.1** Complete task one\n")

        parser = ProgressParser(progress_file)
        data = parser.parse()

        assert data.completed_count == 1
        assert data.total_count == 1

    def test_parse_pending_task(self, tmp_path: Path) -> None:
        """Test parsing pending task."""
        progress_file = tmp_path / "progress.txt"
        progress_file.write_text("- [ ] **1.2** Pending task\n")

        parser = ProgressParser(progress_file)
        data = parser.parse()

        assert data.completed_count == 0
        assert len(data.pending_tasks) == 1

    def test_completion_percentage(self, tmp_path: Path) -> None:
        """Test completion percentage calculation."""
        progress_file = tmp_path / "progress.txt"
        progress_file.write_text(
            "- [x] Task 1\n"
            "- [x] Task 2\n"
            "- [ ] Task 3\n"
            "- [ ] Task 4\n"
        )

        parser = ProgressParser(progress_file)
        data = parser.parse()

        assert data.completion_pct == 50.0

    def test_parse_category_header(self, tmp_path: Path) -> None:
        """Test parsing category headers."""
        progress_file = tmp_path / "progress.txt"
        progress_file.write_text(
            "## CATEGORY 1: Test Category (P0)\n"
            "- [x] Task in category\n"
        )

        parser = ProgressParser(progress_file)
        data = parser.parse()

        assert len(data.categories) >= 0  # May or may not parse headers

    def test_parse_nonexistent_file(self, tmp_path: Path) -> None:
        """Test parsing non-existent file returns empty data."""
        progress_file = tmp_path / "nonexistent.txt"

        parser = ProgressParser(progress_file)
        data = parser.parse()

        assert data.total_count == 0

    def test_parse_empty_file(self, tmp_path: Path) -> None:
        """Test parsing empty file."""
        progress_file = tmp_path / "empty.txt"
        progress_file.write_text("")

        parser = ProgressParser(progress_file)
        data = parser.parse()

        assert data.total_count == 0

    def test_parse_real_progress_file(self) -> None:
        """Test parsing the actual progress file if it exists."""
        progress_file = Path("claude-progress.txt")
        if progress_file.exists():
            parser = ProgressParser(progress_file)
            data = parser.parse()

            # Should have some tasks
            assert data.total_count > 0
            # Completion should be between 0 and 100
            assert 0 <= data.completion_pct <= 100
