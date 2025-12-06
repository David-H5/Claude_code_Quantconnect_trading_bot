#!/usr/bin/env python3
"""
PostToolUse hook to validate algorithm changes.

This hook is called after Edit or Write operations on algorithm files.
It provides feedback about potential issues without blocking.

Exit code 0 for informational messages (doesn't block).
"""

import json
import os
import re
import sys
from pathlib import Path


# Patterns that indicate potential issues in trading algorithms
DANGEROUS_PATTERNS = [
    (r"SetLiveMode\s*\(\s*True\s*\)", "Live mode explicitly enabled - requires review"),
    (r"SetBrokerageModel.*Live", "Live brokerage model detected - requires review"),
    (r"self\.SetEnvironment.*Live", "Live environment setting detected"),
]

# Patterns that indicate look-ahead bias
LOOKAHEAD_PATTERNS = [
    (r"\.shift\s*\(\s*-\d+\s*\)", "Potential look-ahead bias: negative shift"),
    (r"data\[.*\+.*\]", "Potential look-ahead bias: future data access"),
]

# Required patterns for safe algorithms
REQUIRED_PATTERNS = [
    (r"SetWarmUp|self\.warmup", "No warmup period detected - indicators may not be ready"),
    (r"if.*IsWarmingUp", "Missing IsWarmingUp check"),
    (r"ContainsKey|data\.get", "Missing data validation"),
]

# Risk management patterns
RISK_PATTERNS = [
    (r"SetHoldings\s*\([^)]*,\s*[1-9]\d*\.?\d*\s*\)", "Position size > 100% detected"),
    (r"Liquidate|MarketOrder|LimitOrder", None),  # Just tracking, not a warning
]


def check_algorithm(content: str, file_path: str) -> list[tuple[str, str]]:
    """
    Check algorithm content for potential issues.

    Returns:
        List of (severity, message) tuples
    """
    issues: list[tuple[str, str]] = []

    # Check dangerous patterns
    for pattern, message in DANGEROUS_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            issues.append(("WARNING", message))

    # Check look-ahead bias
    for pattern, message in LOOKAHEAD_PATTERNS:
        if re.search(pattern, content):
            issues.append(("WARNING", message))

    # Check for missing required patterns (only for main algorithm files)
    if "algorithms/" in file_path and file_path.endswith(".py"):
        for pattern, message in REQUIRED_PATTERNS:
            if not re.search(pattern, content, re.IGNORECASE):
                issues.append(("INFO", message))

    # Check position sizing
    for pattern, message in RISK_PATTERNS:
        if message and re.search(pattern, content):
            issues.append(("WARNING", message))

    return issues


def main() -> None:
    """Main hook function."""
    # Get environment variables
    tool_name = os.environ.get("TOOL_NAME", "")
    tool_input_str = os.environ.get("TOOL_INPUT", "{}")

    # Only check Edit and Write operations
    if tool_name not in ("Edit", "Write"):
        sys.exit(0)

    # Parse tool input
    try:
        tool_input = json.loads(tool_input_str)
    except json.JSONDecodeError:
        sys.exit(0)

    file_path = tool_input.get("file_path", "")

    # Only check algorithm files
    if "algorithms/" not in file_path:
        sys.exit(0)

    # Try to read the file content
    try:
        with open(file_path) as f:
            content = f.read()
    except (FileNotFoundError, PermissionError):
        sys.exit(0)

    # Check for issues
    issues = check_algorithm(content, file_path)

    if issues:
        print(f"\n--- Algorithm Validation: {Path(file_path).name} ---", file=sys.stderr)
        for severity, message in issues:
            print(f"  [{severity}] {message}", file=sys.stderr)
        print("", file=sys.stderr)

    # Always exit 0 - this is informational only
    sys.exit(0)


if __name__ == "__main__":
    main()
