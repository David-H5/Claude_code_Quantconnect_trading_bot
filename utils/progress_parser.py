"""
Unified Progress File Parser

Single implementation for parsing claude-progress.txt files.
Part of OVERNIGHT-002 refactoring based on docs/OVERNIGHT_SYSTEM_ANALYSIS.md

Replaces duplicate parsing code in:
- session_stop.py
- hook_utils.py
- auto-resume.sh (inline Python)

Usage:
    from utils.progress_parser import ProgressParser, get_progress_parser

    parser = get_progress_parser()
    progress = parser.parse()

    print(f"Completion: {progress.completion_pct:.1f}%")
    next_task = progress.get_next_task()
    if next_task:
        print(f"Next: {next_task.description}")
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Task:
    """A single task from progress file."""

    description: str
    complete: bool
    category: str = ""
    priority: str = "P2"  # Default priority
    line_number: int = 0

    def __str__(self) -> str:
        status = "[x]" if self.complete else "[ ]"
        return f"{status} {self.description}"


@dataclass
class Category:
    """A category from progress file."""

    name: str
    number: int = 0
    priority: str = "P2"
    tasks: list[Task] = field(default_factory=list)
    line_number: int = 0

    @property
    def completed_count(self) -> int:
        """Count of completed tasks."""
        return sum(1 for t in self.tasks if t.complete)

    @property
    def total_count(self) -> int:
        """Total task count."""
        return len(self.tasks)

    @property
    def is_complete(self) -> bool:
        """Check if category is complete."""
        return self.completed_count == self.total_count and self.total_count > 0

    @property
    def completion_pct(self) -> float:
        """Completion percentage for this category."""
        if self.total_count == 0:
            return 100.0
        return (self.completed_count / self.total_count) * 100


@dataclass
class ProgressData:
    """Parsed progress file data."""

    session_id: str | None = None
    goal: str | None = None
    categories: dict[str, Category] = field(default_factory=dict)
    raw_content: str = ""

    @property
    def all_tasks(self) -> list[Task]:
        """Get all tasks across categories."""
        return [t for c in self.categories.values() for t in c.tasks]

    @property
    def completed_tasks(self) -> list[Task]:
        """Get all completed tasks."""
        return [t for t in self.all_tasks if t.complete]

    @property
    def pending_tasks(self) -> list[Task]:
        """Get all pending tasks."""
        return [t for t in self.all_tasks if not t.complete]

    @property
    def total_count(self) -> int:
        """Total task count."""
        return len(self.all_tasks)

    @property
    def completed_count(self) -> int:
        """Completed task count."""
        return len(self.completed_tasks)

    @property
    def completion_pct(self) -> float:
        """Overall completion percentage."""
        if self.total_count == 0:
            return 100.0
        return (self.completed_count / self.total_count) * 100

    def get_next_task(self) -> Task | None:
        """Get next pending task by priority (P0 > P1 > P2)."""
        for priority in ["P0", "P1", "P2"]:
            for task in self.all_tasks:
                if task.priority == priority and not task.complete:
                    return task
        return None

    def get_pending_by_priority(self, priority: str) -> list[Task]:
        """Get pending tasks for a specific priority."""
        return [t for t in self.pending_tasks if t.priority == priority]

    def get_incomplete_categories(self) -> list[Category]:
        """Get categories that have pending tasks."""
        return [c for c in self.categories.values() if not c.is_complete]

    def get_categories_by_priority(self, priority: str) -> list[Category]:
        """Get categories with a specific priority."""
        return [c for c in self.categories.values() if c.priority == priority]

    def has_pending_p0(self) -> bool:
        """Check if there are pending P0 tasks."""
        return len(self.get_pending_by_priority("P0")) > 0

    def has_pending_p1(self) -> bool:
        """Check if there are pending P1 tasks."""
        return len(self.get_pending_by_priority("P1")) > 0

    def to_summary(self) -> str:
        """Generate a text summary."""
        lines = [
            f"Progress: {self.completion_pct:.1f}% ({self.completed_count}/{self.total_count})",
        ]

        if self.session_id:
            lines.insert(0, f"Session: {self.session_id}")
        if self.goal:
            lines.insert(1, f"Goal: {self.goal}")

        # Add category summaries
        for cat in self.categories.values():
            status = "COMPLETE" if cat.is_complete else f"{cat.completed_count}/{cat.total_count}"
            lines.append(f"  {cat.name} ({cat.priority}): {status}")

        # Add next task
        next_task = self.get_next_task()
        if next_task:
            lines.append(f"Next: [{next_task.priority}] {next_task.description}")

        return "\n".join(lines)


class ProgressParser:
    """
    Parser for claude-progress.txt files.

    Supports multiple formats:
    - Category headers with priorities: ## CATEGORY N: Name (P0)
    - Task items: - [ ] Task description or - [x] Completed task
    - Session markers: # Session: ID
    - Goal markers: # Goal: Description
    """

    # Pattern for category headers like "## CATEGORY 1: Name (P0)"
    CATEGORY_PATTERN = re.compile(
        r"^##\s*(?:CATEGORY\s+)?(\d+)[:.]?\s*(.+?)\s*\((P[0-2])\)\s*$",
        re.IGNORECASE,
    )

    # Alternative pattern for "## Phase N: Name (P0 - Priority)"
    PHASE_PATTERN = re.compile(
        r"^##\s*(?:Phase|PHASE)\s+(\d+)[:.]?\s*(.+?)\s*\((P[0-2])",
        re.IGNORECASE,
    )

    # Pattern for task items: - [ ] Task or - [x] Task
    TASK_PATTERN = re.compile(r"^\s*-\s*\[([ xX])\]\s*(.+)$")

    # Pattern for session ID: # Session: ID
    SESSION_PATTERN = re.compile(r"^#\s*Session:\s*(.+)$", re.IGNORECASE)

    # Pattern for goal: # Goal: Description or # GOAL
    GOAL_PATTERN = re.compile(r"^#\s*GOAL[:.]?\s*(.*)$", re.IGNORECASE)

    # Pattern for generic headers that might contain tasks
    GENERIC_HEADER_PATTERN = re.compile(r"^##\s*(.+?)(?:\s*\((P[0-2])\))?\s*$")

    def __init__(self, progress_file: Path | str | None = None):
        """
        Initialize parser.

        Args:
            progress_file: Path to progress file, defaults to claude-progress.txt
        """
        if progress_file is None:
            self.progress_file = Path("claude-progress.txt")
        else:
            self.progress_file = Path(progress_file)

    def parse(self) -> ProgressData:
        """
        Parse the progress file.

        Returns:
            Parsed progress data
        """
        data = ProgressData()

        if not self.progress_file.exists():
            return data

        try:
            content = self.progress_file.read_text()
            data.raw_content = content
        except OSError:
            return data

        current_category: Category | None = None
        current_priority = "P2"  # Default priority

        for line_num, line in enumerate(content.split("\n"), 1):
            # Session ID
            if match := self.SESSION_PATTERN.match(line):
                data.session_id = match.group(1).strip()
                continue

            # Goal (might be on same line or next lines)
            if match := self.GOAL_PATTERN.match(line):
                goal_text = match.group(1).strip()
                if goal_text:
                    data.goal = goal_text
                continue

            # Category header (specific format)
            if match := self.CATEGORY_PATTERN.match(line):
                cat_num = int(match.group(1))
                cat_name = match.group(2).strip()
                priority = match.group(3).upper()
                name = f"Category {cat_num}: {cat_name}"
                current_category = Category(
                    name=name,
                    number=cat_num,
                    priority=priority,
                    line_number=line_num,
                )
                current_priority = priority
                data.categories[name] = current_category
                continue

            # Phase header (alternative format)
            if match := self.PHASE_PATTERN.match(line):
                phase_num = int(match.group(1))
                phase_name = match.group(2).strip()
                priority = match.group(3).upper()
                name = f"Phase {phase_num}: {phase_name}"
                current_category = Category(
                    name=name,
                    number=phase_num,
                    priority=priority,
                    line_number=line_num,
                )
                current_priority = priority
                data.categories[name] = current_category
                continue

            # Generic header (creates category if has priority)
            if match := self.GENERIC_HEADER_PATTERN.match(line):
                header_name = match.group(1).strip()
                priority = match.group(2) if match.group(2) else "P2"

                # Skip certain non-category headers
                skip_headers = ["GOAL", "SESSION", "BLOCKERS", "CHECKPOINTS", "SUMMARY"]
                if any(skip in header_name.upper() for skip in skip_headers):
                    current_category = None
                    continue

                # Check if it looks like a category
                if "COMPLETE" in header_name.upper():
                    current_category = None
                    continue

                name = header_name
                current_category = Category(
                    name=name,
                    priority=priority.upper(),
                    line_number=line_num,
                )
                current_priority = priority.upper()
                data.categories[name] = current_category
                continue

            # Task item
            if match := self.TASK_PATTERN.match(line):
                is_complete = match.group(1).lower() == "x"
                description = match.group(2).strip()

                # Create task
                task = Task(
                    description=description,
                    complete=is_complete,
                    category=current_category.name if current_category else "",
                    priority=current_priority,
                    line_number=line_num,
                )

                # Add to current category or create default
                if current_category:
                    current_category.tasks.append(task)
                else:
                    # Create default category for orphan tasks
                    if "Uncategorized" not in data.categories:
                        data.categories["Uncategorized"] = Category(
                            name="Uncategorized",
                            priority="P2",
                        )
                    data.categories["Uncategorized"].tasks.append(task)

        return data

    def get_completion_status(self) -> dict[str, float | int]:
        """
        Get completion status summary.

        Returns:
            Dict with completion stats
        """
        progress = self.parse()
        return {
            "total": progress.total_count,
            "completed": progress.completed_count,
            "pending": progress.total_count - progress.completed_count,
            "completion_pct": progress.completion_pct,
            "p0_pending": len(progress.get_pending_by_priority("P0")),
            "p1_pending": len(progress.get_pending_by_priority("P1")),
            "p2_pending": len(progress.get_pending_by_priority("P2")),
        }

    def iter_pending_tasks(self) -> Iterator[Task]:
        """Iterate over pending tasks by priority."""
        progress = self.parse()
        for priority in ["P0", "P1", "P2"]:
            for task in progress.get_pending_by_priority(priority):
                yield task


# Global instance cache
_parser_cache: ProgressParser | None = None


def get_progress_parser(reload: bool = False) -> ProgressParser:
    """
    Get the progress parser (cached).

    Args:
        reload: Force new parser instance

    Returns:
        Progress parser
    """
    global _parser_cache
    if _parser_cache is None or reload:
        _parser_cache = ProgressParser()
    return _parser_cache


def parse_progress_file(progress_file: Path | str | None = None) -> ProgressData:
    """
    Parse a progress file (convenience function).

    Args:
        progress_file: Path to progress file

    Returns:
        Parsed progress data
    """
    parser = ProgressParser(progress_file)
    return parser.parse()
