#!/usr/bin/env python3
"""
QA Auto-Check Hook for Claude Code.

This hook automatically runs targeted QA checks after certain tool operations.
It integrates with the QA validator to provide real-time feedback during development.

Usage in .claude/settings.json:
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": "python3 .claude/hooks/qa_auto_check.py"
      }]
    }]
  }
}

Author: Claude Code
Created: 2025-12-03
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Find project root by looking for CLAUDE.md."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "CLAUDE.md").exists():
            return current
        current = current.parent
    return Path.cwd()


def get_tool_input() -> dict:
    """Get the tool input from environment variable."""
    tool_input = os.environ.get("CLAUDE_TOOL_INPUT", "{}")
    try:
        return json.loads(tool_input)
    except json.JSONDecodeError:
        return {}


def run_quick_check(file_path: str, project_root: Path) -> dict:
    """Run quick QA checks on the modified file."""
    results = {
        "issues": [],
        "passed": True,
    }

    if not file_path:
        return results

    full_path = project_root / file_path
    if not full_path.exists():
        return results

    # Only check Python files
    if not file_path.endswith(".py"):
        return results

    try:
        content = full_path.read_text()

        # Quick debug checks
        debug_issues = []

        # Check for breakpoints
        if "breakpoint()" in content:
            debug_issues.append("breakpoint() found - remove before commit")
            results["passed"] = False

        # Check for pdb imports
        if "import pdb" in content or "import ipdb" in content:
            debug_issues.append("Debug import (pdb/ipdb) found")

        # Check for print statements in non-script files
        if "script" not in file_path.lower() and "test" not in file_path.lower():
            import re

            prints = re.findall(r"^\s*print\s*\(", content, re.MULTILINE)
            if len(prints) > 5:
                debug_issues.append(f"Many print statements ({len(prints)}) - consider logging")

        # Check for syntax errors
        try:
            result = subprocess.run(
                ["python3", "-m", "py_compile", str(full_path)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                debug_issues.append(f"Syntax error: {result.stderr.strip()[:100]}")
                results["passed"] = False
        except Exception:
            pass

        # Check for incomplete code markers
        incomplete_markers = ["# TODO:", "# FIXME:", "# BUG:", "raise NotImplementedError"]
        for marker in incomplete_markers:
            if marker in content:
                debug_issues.append(f"Found {marker.split(':')[0].strip('#').strip()}")

        results["issues"] = debug_issues

    except Exception as e:
        results["issues"].append(f"Check error: {e}")

    return results


def main():
    """Main hook entry point."""
    project_root = get_project_root()
    tool_input = get_tool_input()

    # Get the file path from tool input
    file_path = tool_input.get("file_path", "")
    if file_path:
        # Make relative if absolute
        try:
            file_path = str(Path(file_path).relative_to(project_root))
        except ValueError:
            pass

    # Run quick check
    results = run_quick_check(file_path, project_root)

    # Output results if there are issues
    if results["issues"]:
        print(f"\n--- QA Quick Check: {file_path} ---")
        for issue in results["issues"]:
            prefix = "❌" if not results["passed"] else "⚠️"
            print(f"  {prefix} {issue}")
        print("---")

        if not results["passed"]:
            print("\n💡 Run `python scripts/qa_validator.py --check debug` for full debug check")

    # Exit 0 to allow the operation to continue
    # Exit 1 would block the operation
    sys.exit(0)


if __name__ == "__main__":
    main()
