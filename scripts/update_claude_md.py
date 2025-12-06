#!/usr/bin/env python3
"""
Auto-update CLAUDE.md with current project metrics.

Updates an auto-generated section of CLAUDE.md with:
- Module statistics (file count, LOC)
- Test coverage
- Agent registry
- Command summary
- Recent changes

Usage:
    python scripts/update_claude_md.py [--dry-run]

The script looks for markers:
    <!-- AUTO-GENERATED-METRICS-START -->
    <!-- AUTO-GENERATED-METRICS-END -->

If not found, it appends the section.
"""

import contextlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def get_module_stats() -> dict[str, dict[str, Any]]:
    """
    Get statistics for each module.

    Returns:
        Dict mapping module name to stats (files, loc).
    """
    modules = [
        "algorithms",
        "api",
        "config",
        "evaluation",
        "execution",
        "indicators",
        "llm",
        "models",
        "observability",
        "scanners",
        "utils",
        "ui",
    ]

    stats = {}
    for mod in modules:
        mod_path = Path(mod)
        if not mod_path.exists():
            continue

        py_files = list(mod_path.rglob("*.py"))
        py_files = [f for f in py_files if "__pycache__" not in str(f)]

        total_loc = 0
        for f in py_files:
            with contextlib.suppress(OSError, UnicodeDecodeError):
                total_loc += len(f.read_text().splitlines())

        stats[mod] = {
            "files": len(py_files),
            "loc": total_loc,
        }

    return stats


def format_module_table(stats: dict[str, dict[str, Any]]) -> str:
    """
    Format module stats as markdown table.

    Args:
        stats: Module statistics dict.

    Returns:
        Markdown table string.
    """
    lines = [
        "| Module | Files | LOC |",
        "|--------|------:|----:|",
    ]

    total_files = 0
    total_loc = 0

    for mod, data in sorted(stats.items()):
        lines.append(f"| `{mod}/` | {data['files']} | {data['loc']:,} |")
        total_files += data["files"]
        total_loc += data["loc"]

    lines.append(f"| **Total** | **{total_files}** | **{total_loc:,}** |")

    return "\n".join(lines)


def get_test_coverage() -> str:
    """
    Get test coverage from pytest-cov if available.

    Returns:
        Coverage string or placeholder.
    """
    cov_json = Path("coverage.json")
    if cov_json.exists():
        try:
            data = json.loads(cov_json.read_text())
            total = data.get("totals", {}).get("percent_covered", 0)
            return f"{total:.1f}%"
        except (json.JSONDecodeError, KeyError):
            pass

    # Try to read from .coverage
    try:
        result = subprocess.run(
            ["python", "-m", "coverage", "report", "--format=total"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return f"{result.stdout.strip()}%"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return "Run `pytest --cov` to calculate"


def get_agent_count() -> int:
    """
    Count registered trading agents.

    Returns:
        Number of agent classes.
    """
    agents_dir = Path("llm/agents")
    if not agents_dir.exists():
        return 0

    count = 0
    for py_file in agents_dir.glob("*.py"):
        if py_file.name.startswith("__"):
            continue

        try:
            content = py_file.read_text()
            # Count class definitions that likely inherit from Agent
            count += content.count("class ") - content.count("class _")
        except (OSError, UnicodeDecodeError):
            pass

    return count


def get_command_count() -> int:
    """
    Count available Claude Code commands.

    Returns:
        Number of command files.
    """
    commands_dir = Path(".claude/commands")
    if not commands_dir.exists():
        return 0

    return len(list(commands_dir.glob("*.md")))


def get_hook_count() -> int:
    """
    Count Claude Code hooks.

    Returns:
        Number of hook files.
    """
    hooks_dir = Path(".claude/hooks")
    if not hooks_dir.exists():
        return 0

    count = 0
    for category_dir in hooks_dir.iterdir():
        if category_dir.is_dir() and not category_dir.name.startswith("_"):
            count += len(list(category_dir.glob("*.py")))

    return count


def get_recent_commits(n: int = 5) -> list[str]:
    """
    Get recent git commit summaries.

    Args:
        n: Number of commits to retrieve.

    Returns:
        List of commit summaries.
    """
    try:
        result = subprocess.run(
            ["git", "log", f"-{n}", "--oneline", "--no-decorate"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [line for line in result.stdout.strip().split("\n") if line]
    except subprocess.CalledProcessError:
        return []


def get_docstring_coverage() -> str:
    """
    Get docstring coverage from state file.

    Returns:
        Coverage percentage string.
    """
    state_file = Path(".claude/state/docstring_coverage.json")
    if state_file.exists():
        try:
            history = json.loads(state_file.read_text())
            if history:
                latest = history[-1]
                results = latest.get("results", {})
                total_pct = sum(r.get("coverage_pct", 0) for r in results.values())
                avg_pct = total_pct / len(results) if results else 0
                return f"{avg_pct:.1f}%"
        except (json.JSONDecodeError, KeyError):
            pass

    return "Run `python scripts/docstring_coverage.py`"


def generate_metrics_section() -> str:
    """
    Generate the auto-generated metrics section.

    Returns:
        Markdown content for metrics section.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    module_stats = get_module_stats()

    lines = [
        "",
        "<!-- AUTO-GENERATED-METRICS-START -->",
        "",
        "## Auto-Generated Project Metrics",
        "",
        f"*Last updated: {timestamp}*",
        "",
        "### Module Statistics",
        "",
        format_module_table(module_stats),
        "",
        "### Quality Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Test Coverage | {get_test_coverage()} |",
        f"| Docstring Coverage | {get_docstring_coverage()} |",
        f"| Agent Count | {get_agent_count()} agents |",
        f"| Claude Commands | {get_command_count()} commands |",
        f"| Claude Hooks | {get_hook_count()} hooks |",
        "",
        "### Recent Changes",
        "",
    ]

    commits = get_recent_commits(5)
    if commits:
        for commit in commits:
            lines.append(f"- `{commit}`")
    else:
        lines.append("- No recent commits found")

    lines.extend(
        [
            "",
            "### Regenerate Metrics",
            "",
            "```bash",
            "python scripts/update_claude_md.py",
            "```",
            "",
            "<!-- AUTO-GENERATED-METRICS-END -->",
            "",
        ]
    )

    return "\n".join(lines)


def update_claude_md(dry_run: bool = False) -> bool:
    """
    Update CLAUDE.md with current metrics.

    Args:
        dry_run: If True, print changes without writing.

    Returns:
        True if successful.
    """
    claude_md = Path("CLAUDE.md")
    if not claude_md.exists():
        print("Error: CLAUDE.md not found")
        return False

    content = claude_md.read_text()
    new_section = generate_metrics_section()

    marker_start = "<!-- AUTO-GENERATED-METRICS-START -->"
    marker_end = "<!-- AUTO-GENERATED-METRICS-END -->"

    if marker_start in content:
        # Replace existing section
        pattern = f"{re.escape(marker_start)}.*?{re.escape(marker_end)}"
        new_content = re.sub(
            pattern,
            new_section.strip(),
            content,
            flags=re.DOTALL,
        )
    else:
        # Append to end
        new_content = content.rstrip() + "\n" + new_section

    if dry_run:
        print("--- DRY RUN ---")
        print("Would update CLAUDE.md with:")
        print(new_section)
        return True

    claude_md.write_text(new_content)
    print("Updated CLAUDE.md with current metrics")
    return True


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 for success).
    """
    dry_run = "--dry-run" in sys.argv

    if update_claude_md(dry_run):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
