#!/usr/bin/env python3
"""
Auto-Docstring Generator Hook

Runs PostToolUse on Edit|Write operations.
Uses Claude to generate Google-style docstrings for functions without them.

Usage:
    Automatically invoked by Claude Code hooks system.
    Outputs JSON message to Claude when undocumented functions found.
"""

import ast
import contextlib
import json
import sys
from pathlib import Path
from typing import Any


def find_undocumented_functions(filepath: str) -> list[dict[str, Any]]:
    """
    Find functions without docstrings in a Python file.

    Args:
        filepath: Path to the Python file to analyze.

    Returns:
        List of dicts containing function info:
        - name: Function name
        - line: Line number
        - args: List of argument names
        - returns: Return type annotation if present
        - is_method: Whether it's a class method
        - class_name: Parent class name if method
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError):
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    undocumented = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # Skip private/dunder methods (often self-explanatory)
            if node.name.startswith("__") and node.name.endswith("__"):
                continue
            if node.name.startswith("_") and len(node.name) > 1:
                # Single underscore private - still should be documented
                pass

            # Check if has docstring
            if ast.get_docstring(node):
                continue

            # Build function info
            args = []
            for arg in node.args.args:
                arg_info = {"name": arg.arg}
                if arg.annotation:
                    with contextlib.suppress(Exception):
                        arg_info["type"] = ast.unparse(arg.annotation)
                args.append(arg_info)

            return_type = None
            if node.returns:
                with contextlib.suppress(Exception):
                    return_type = ast.unparse(node.returns)

            # Check if it's a method in a class
            is_method = False
            class_name = None
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    for item in parent.body:
                        if item is node:
                            is_method = True
                            class_name = parent.name
                            break

            undocumented.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "args": args,
                    "returns": return_type,
                    "is_method": is_method,
                    "class_name": class_name,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                }
            )

    return undocumented


def find_undocumented_classes(filepath: str) -> list[dict[str, Any]]:
    """
    Find classes without docstrings in a Python file.

    Args:
        filepath: Path to the Python file to analyze.

    Returns:
        List of dicts containing class info:
        - name: Class name
        - line: Line number
        - bases: List of base class names
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError):
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    undocumented = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if ast.get_docstring(node):
                continue

            bases = []
            for base in node.bases:
                try:
                    bases.append(ast.unparse(base))
                except Exception:
                    bases.append("?")

            undocumented.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "bases": bases,
                }
            )

    return undocumented


def main() -> None:
    """Main entry point for the hook."""
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    tool_input = input_data.get("tool_input", {})
    filepath = tool_input.get("file_path", "")

    # Only process Python files
    if not filepath.endswith(".py"):
        sys.exit(0)

    # Skip test files (often have minimal docs)
    if "/tests/" in filepath or filepath.startswith("tests/"):
        sys.exit(0)

    # Skip files that are unlikely to need docs
    if "__pycache__" in filepath or ".venv" in filepath:
        sys.exit(0)

    # Check if file exists
    if not Path(filepath).exists():
        sys.exit(0)

    # Find undocumented items
    undoc_funcs = find_undocumented_functions(filepath)
    undoc_classes = find_undocumented_classes(filepath)

    if not undoc_funcs and not undoc_classes:
        sys.exit(0)

    # Build message
    filename = Path(filepath).name
    parts = []

    if undoc_funcs:
        func_list = ", ".join(f"`{f['name']}`" for f in undoc_funcs[:5])
        if len(undoc_funcs) > 5:
            func_list += f" (+{len(undoc_funcs) - 5} more)"
        parts.append(f"{len(undoc_funcs)} functions need docstrings: {func_list}")

    if undoc_classes:
        class_list = ", ".join(f"`{c['name']}`" for c in undoc_classes[:3])
        parts.append(f"{len(undoc_classes)} classes need docstrings: {class_list}")

    message = f"[Auto-Doc] In `{filename}`: {'; '.join(parts)}"

    # Output for Claude
    output = {
        "result": "continue",
        "message": message,
        "data": {
            "undocumented_functions": undoc_funcs[:10],  # Limit to 10
            "undocumented_classes": undoc_classes[:5],  # Limit to 5
            "file": filepath,
        },
    }

    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
