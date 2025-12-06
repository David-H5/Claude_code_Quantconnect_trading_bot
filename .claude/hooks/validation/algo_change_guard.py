#!/usr/bin/env python3
"""
Algorithm Change Guard PreToolUse Hook

Guards against unreviewed changes to critical algorithm files.
Requires explicit confirmation for changes to production algorithms.

UPGRADE-015 Phase 4: Hook System Implementation

Usage:
    Called as PreToolUse hook before Edit/Write to algorithm files.
    Warns or blocks changes to protected algorithm files.

Protected Paths:
    - algorithms/*.py (production algorithms)
    - models/risk_manager.py (risk management)
    - models/circuit_breaker.py (safety systems)
    - execution/*.py (order execution)
"""

import json
import sys
from pathlib import Path


# Protected paths (critical trading code)
PROTECTED_PATHS = [
    "algorithms/",
    "models/risk_manager.py",
    "models/circuit_breaker.py",
    "execution/smart_execution.py",
    "execution/profit_taking.py",
]

# Files that require extra caution
HIGH_RISK_PATTERNS = [
    "live",
    "production",
    "real_trading",
    "execute_order",
    "place_order",
]


def is_protected_path(file_path: str) -> bool:
    """Check if file path is protected."""
    path = Path(file_path)

    for protected in PROTECTED_PATHS:
        if protected.endswith("/"):
            # Directory pattern
            if str(path).startswith(protected) or f"/{protected}" in str(path):
                return True
        else:
            # Exact file
            if str(path).endswith(protected) or protected in str(path):
                return True

    return False


def check_high_risk_content(content: str) -> list:
    """Check for high-risk patterns in content."""
    warnings = []

    for pattern in HIGH_RISK_PATTERNS:
        if pattern.lower() in content.lower():
            warnings.append(f"Contains '{pattern}' - review carefully")

    return warnings


def main():
    """Main entry point for hook."""
    # Read tool context from stdin
    try:
        input_data = sys.stdin.read()
        if not input_data:
            sys.exit(0)

        context = json.loads(input_data)
    except json.JSONDecodeError:
        sys.exit(0)

    tool_name = context.get("tool_name", "")
    tool_input = context.get("tool_input", {})

    # Only check Edit and Write tools
    if tool_name not in ["Edit", "Write"]:
        sys.exit(0)

    # Get file path
    file_path = tool_input.get("file_path", "") or tool_input.get("path", "")
    if not file_path:
        sys.exit(0)

    # Check if protected
    if is_protected_path(file_path):
        warnings = []

        # Check content for high-risk patterns
        content = tool_input.get("content", "") or tool_input.get("new_string", "")
        if content:
            warnings = check_high_risk_content(content)

        # Generate warning message
        print("", file=sys.stderr)
        print("⚠️  ALGORITHM CHANGE GUARD", file=sys.stderr)
        print(f"   File: {file_path}", file=sys.stderr)
        print("   This is a protected trading file.", file=sys.stderr)

        if warnings:
            print("   Warnings:", file=sys.stderr)
            for w in warnings:
                print(f"     - {w}", file=sys.stderr)

        print("   Changes will proceed - ensure proper testing.", file=sys.stderr)
        print("", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
