#!/usr/bin/env python3
"""Scan codebase for TODO/FIXME items and generate tracking report.

Scans Python files for TODO, FIXME, XXX, HACK, and BUG comments,
classifies them by priority, and generates a markdown report.

Categories:
- P0: Critical (blocks functionality, security issues)
- P1: Important (should fix soon, bugs)
- P2: Polish (nice to have, refactoring)

Usage:
    python scripts/scan_todos.py [--json] [--output FILE] [--quiet]

Examples:
    python scripts/scan_todos.py                    # Generate report
    python scripts/scan_todos.py --json             # JSON output
    python scripts/scan_todos.py --output report.md # Custom output file
"""

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# Pattern to match TODO-style comments
TODO_PATTERN = re.compile(
    r"#\s*(TODO|FIXME|XXX|HACK|BUG)[\s:]*(.+)",
    re.IGNORECASE,
)

# Keywords for priority classification
PRIORITY_KEYWORDS = {
    "P0": [
        "critical",
        "blocking",
        "urgent",
        "security",
        "crash",
        "data loss",
        "vulnerability",
        "p0",
        "severity:critical",
    ],
    "P1": [
        "important",
        "should",
        "need",
        "bug",
        "fix",
        "broken",
        "error",
        "p1",
        "severity:high",
    ],
    "P2": [
        "polish",
        "refactor",
        "cleanup",
        "nice",
        "maybe",
        "consider",
        "improve",
        "optimization",
        "p2",
    ],
}

# Directories to skip
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".hypothesis",
    "node_modules",
    ".backups",
    "research",
    ".claude",
}


@dataclass
class TodoItem:
    """A single TODO item found in the codebase."""

    file: str
    line: int
    todo_type: str  # TODO, FIXME, XXX, HACK, BUG
    text: str
    priority: str  # P0, P1, P2
    context: str = ""  # Surrounding code context

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file": self.file,
            "line": self.line,
            "type": self.todo_type,
            "text": self.text,
            "priority": self.priority,
            "context": self.context,
        }


@dataclass
class ScanResult:
    """Result of scanning the codebase."""

    items: dict[str, list[TodoItem]] = field(default_factory=lambda: {"P0": [], "P1": [], "P2": []})
    files_scanned: int = 0
    scan_time: datetime = field(default_factory=datetime.now)

    def total_count(self) -> int:
        """Get total TODO count."""
        return sum(len(items) for items in self.items.values())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scan_time": self.scan_time.isoformat(),
            "files_scanned": self.files_scanned,
            "total_count": self.total_count(),
            "by_priority": {p: len(items) for p, items in self.items.items()},
            "items": {p: [item.to_dict() for item in items] for p, items in self.items.items()},
        }


def classify_priority(text: str, todo_type: str) -> str:
    """Classify TODO priority based on keywords and type.

    Args:
        text: The TODO text content.
        todo_type: The type (TODO, FIXME, BUG, etc.).

    Returns:
        Priority string (P0, P1, or P2).
    """
    text_lower = text.lower()
    type_lower = todo_type.lower()

    # BUG and FIXME default to higher priority
    if type_lower in ("bug", "fixme"):
        default_priority = "P1"
    else:
        default_priority = "P2"

    # Check for priority keywords
    for priority, keywords in PRIORITY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return priority

    return default_priority


def scan_file(filepath: Path) -> list[TodoItem]:
    """Scan a single file for TODOs.

    Args:
        filepath: Path to the file to scan.

    Returns:
        List of TodoItem objects found.
    """
    todos = []

    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    lines = content.splitlines()

    for i, line in enumerate(lines, 1):
        match = TODO_PATTERN.search(line)
        if match:
            todo_type = match.group(1).upper()
            text = match.group(2).strip()

            # Get context (line before and after)
            context_lines = []
            if i > 1:
                context_lines.append(lines[i - 2].strip())
            context_lines.append(line.strip())
            if i < len(lines):
                context_lines.append(lines[i].strip())

            todos.append(
                TodoItem(
                    file=str(filepath),
                    line=i,
                    todo_type=todo_type,
                    text=text,
                    priority=classify_priority(text, todo_type),
                    context="\n".join(context_lines),
                )
            )

    return todos


def scan_codebase(root: Path | None = None) -> ScanResult:
    """Scan entire codebase for TODOs.

    Args:
        root: Root directory to scan. Defaults to current directory.

    Returns:
        ScanResult with all found items.
    """
    root = root or Path(".")
    result = ScanResult()

    for py_file in root.rglob("*.py"):
        # Skip excluded directories
        if any(skip_dir in py_file.parts for skip_dir in SKIP_DIRS):
            continue

        result.files_scanned += 1
        file_todos = scan_file(py_file)

        for todo in file_todos:
            result.items[todo.priority].append(todo)

    return result


def generate_markdown_report(result: ScanResult) -> str:
    """Generate markdown report from scan results.

    Args:
        result: ScanResult to format.

    Returns:
        Markdown formatted report string.
    """
    lines = [
        "# Technical Debt Report",
        "",
        f"*Generated: {result.scan_time.strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Summary",
        "",
        "| Priority | Count | Description |",
        "|----------|------:|-------------|",
        f"| **P0** | {len(result.items['P0'])} | Critical - blocks functionality |",
        f"| **P1** | {len(result.items['P1'])} | Important - should fix soon |",
        f"| **P2** | {len(result.items['P2'])} | Polish - nice to have |",
        f"| **Total** | **{result.total_count()}** | |",
        "",
    ]

    # Add items by priority
    for priority in ["P0", "P1", "P2"]:
        items = result.items[priority]
        if not items:
            continue

        lines.extend(
            [
                f"## {priority} Items ({len(items)})",
                "",
            ]
        )

        # Group by file
        by_file: dict[str, list[TodoItem]] = {}
        for item in items:
            if item.file not in by_file:
                by_file[item.file] = []
            by_file[item.file].append(item)

        for filepath, file_items in sorted(by_file.items()):
            lines.append(f"### `{filepath}`")
            lines.append("")

            for item in sorted(file_items, key=lambda x: x.line):
                lines.append(f"- [ ] **Line {item.line}** [{item.todo_type}]: {item.text}")

            lines.append("")

    # Add instructions
    lines.extend(
        [
            "## Addressing Technical Debt",
            "",
            "### Priority Guidelines",
            "",
            "- **P0**: Address immediately - these block functionality or pose security risks",
            "- **P1**: Plan for current or next sprint - bugs and important improvements",
            "- **P2**: Add to backlog - address during refactoring sessions",
            "",
            "### Workflow",
            "",
            "1. Pick an item from P0 (if any), otherwise P1",
            "2. Create a branch: `git checkout -b fix/todo-description`",
            "3. Fix the issue and remove the TODO comment",
            "4. Run tests: `pytest tests/ -v`",
            "5. Commit with message: `fix: resolve TODO - description`",
            "",
            "### Regenerate Report",
            "",
            "```bash",
            "python scripts/scan_todos.py",
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """Main entry point for TODO scanner.

    Returns:
        Exit code (0 for success).
    """
    # Parse arguments
    json_output = "--json" in sys.argv
    quiet = "--quiet" in sys.argv

    output_file = Path("docs/TECHNICAL_DEBT.md")
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_file = Path(sys.argv[idx + 1])

    # Run scan
    if not quiet:
        print("Scanning codebase for TODOs...")

    result = scan_codebase()

    # Save state
    state_file = Path(".claude/state/todos.json")
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(result.to_dict(), indent=2))

    if json_output:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    # Generate and save report
    report = generate_markdown_report(result)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report)

    if not quiet:
        # Print summary
        print(f"\nScanned {result.files_scanned} files")
        print(f"Found {result.total_count()} TODOs:")
        for priority in ["P0", "P1", "P2"]:
            count = len(result.items[priority])
            if count > 0:
                icon = "🔴" if priority == "P0" else "🟡" if priority == "P1" else "🟢"
                print(f"  {icon} {priority}: {count}")

        print(f"\nReport saved to {output_file}")

        # Warning for critical items
        if result.items["P0"]:
            print("\n⚠️  ATTENTION: Critical P0 items found that need immediate attention!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
