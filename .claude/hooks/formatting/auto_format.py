#!/usr/bin/env python3
"""
Auto-format hook for Claude Code.

This hook runs after Edit/Write operations on Python files and:
1. Runs ruff format on the modified file
2. Runs ruff check --fix for auto-fixable issues
3. Reports any remaining issues

Referenced by .claude/settings.json PostToolUse hook configuration.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def get_tool_input() -> dict | None:
    """Get the tool input from CLAUDE_TOOL_INPUT env var."""
    tool_input = os.environ.get("CLAUDE_TOOL_INPUT", "")
    if not tool_input:
        return None
    try:
        return json.loads(tool_input)
    except json.JSONDecodeError:
        return None


def is_python_file(file_path: str) -> bool:
    """Check if the file is a Python file."""
    return file_path.endswith(".py")


def run_ruff_format(file_path: str) -> bool:
    """Run ruff format on the file."""
    try:
        result = subprocess.run(
            ["ruff", "format", file_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            if result.stdout.strip():
                print(f"Formatted: {file_path}")
            return True
        else:
            print(f"Format warning: {result.stderr}", file=sys.stderr)
            return True  # Don't block on format warnings
    except FileNotFoundError:
        # ruff not installed, skip silently
        return True
    except subprocess.TimeoutExpired:
        print("Format timed out", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Format error: {e}", file=sys.stderr)
        return True


def run_ruff_fix(file_path: str) -> bool:
    """Run ruff check --fix for auto-fixable issues."""
    try:
        result = subprocess.run(
            ["ruff", "check", "--fix", "--quiet", file_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # ruff check returns non-zero if there are unfixable issues
        # We don't want to block on those, just report
        if result.returncode != 0 and result.stdout.strip():
            # There are remaining issues (unfixable)
            lines = result.stdout.strip().split("\n")
            if len(lines) <= 3:
                for line in lines:
                    print(f"Lint: {line}")
            else:
                print(f"Lint: {len(lines)} issues remaining in {file_path}")
        return True
    except FileNotFoundError:
        # ruff not installed, skip silently
        return True
    except subprocess.TimeoutExpired:
        print("Lint timed out", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Lint error: {e}", file=sys.stderr)
        return True


def main():
    """Main auto-format hook execution."""
    tool_input = get_tool_input()
    if not tool_input:
        return 0

    # Get file path from tool input
    file_path = tool_input.get("file_path", "")
    if not file_path:
        return 0

    # Only format Python files
    if not is_python_file(file_path):
        return 0

    # Check file exists
    if not Path(file_path).exists():
        return 0

    # Run formatting
    run_ruff_format(file_path)
    run_ruff_fix(file_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
