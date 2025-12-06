#!/usr/bin/env python3
"""
Docstring Coverage Tracking Script

Calculates and tracks docstring coverage across modules over time.
Generates reports for monitoring documentation quality improvements.

Usage:
    python scripts/docstring_coverage.py [--json] [--verbose]

Outputs:
    - Console summary
    - State file at .claude/state/docstring_coverage.json (history)
    - Optional JSON output for CI integration
"""

import ast
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def calculate_file_coverage(filepath: Path) -> dict[str, Any]:
    """
    Calculate docstring coverage for a single Python file.

    Args:
        filepath: Path to the Python file to analyze.

    Returns:
        Dict with coverage stats:
        - functions_total: Total function count
        - functions_documented: Functions with docstrings
        - classes_total: Total class count
        - classes_documented: Classes with docstrings
        - undocumented_items: List of items missing docs
    """
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {
            "functions_total": 0,
            "functions_documented": 0,
            "classes_total": 0,
            "classes_documented": 0,
            "undocumented_items": [],
        }

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {
            "functions_total": 0,
            "functions_documented": 0,
            "classes_total": 0,
            "classes_documented": 0,
            "undocumented_items": [],
        }

    functions_total = 0
    functions_documented = 0
    classes_total = 0
    classes_documented = 0
    undocumented = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # Skip dunder methods
            if node.name.startswith("__") and node.name.endswith("__"):
                continue

            functions_total += 1
            if ast.get_docstring(node):
                functions_documented += 1
            else:
                undocumented.append(
                    {
                        "type": "function",
                        "name": node.name,
                        "line": node.lineno,
                    }
                )

        elif isinstance(node, ast.ClassDef):
            classes_total += 1
            if ast.get_docstring(node):
                classes_documented += 1
            else:
                undocumented.append(
                    {
                        "type": "class",
                        "name": node.name,
                        "line": node.lineno,
                    }
                )

    return {
        "functions_total": functions_total,
        "functions_documented": functions_documented,
        "classes_total": classes_total,
        "classes_documented": classes_documented,
        "undocumented_items": undocumented,
    }


def calculate_module_coverage(module_path: str) -> dict[str, Any]:
    """
    Calculate docstring coverage for an entire module directory.

    Args:
        module_path: Path to the module directory.

    Returns:
        Dict with aggregated coverage stats and per-file breakdown.
    """
    mod_dir = Path(module_path)
    if not mod_dir.exists():
        return {
            "exists": False,
            "total_functions": 0,
            "documented_functions": 0,
            "total_classes": 0,
            "documented_classes": 0,
            "coverage_pct": 0.0,
            "files": {},
        }

    total_functions = 0
    documented_functions = 0
    total_classes = 0
    documented_classes = 0
    files_data = {}

    for py_file in mod_dir.rglob("*.py"):
        # Skip test files and cache
        if "test" in str(py_file).lower():
            continue
        if "__pycache__" in str(py_file):
            continue

        file_stats = calculate_file_coverage(py_file)
        relative_path = str(py_file.relative_to(mod_dir))

        total_functions += file_stats["functions_total"]
        documented_functions += file_stats["functions_documented"]
        total_classes += file_stats["classes_total"]
        documented_classes += file_stats["classes_documented"]

        if file_stats["functions_total"] > 0 or file_stats["classes_total"] > 0:
            file_total = file_stats["functions_total"] + file_stats["classes_total"]
            file_documented = file_stats["functions_documented"] + file_stats["classes_documented"]
            file_coverage = (file_documented / file_total * 100) if file_total > 0 else 100.0

            files_data[relative_path] = {
                "functions": f"{file_stats['functions_documented']}/{file_stats['functions_total']}",
                "classes": f"{file_stats['classes_documented']}/{file_stats['classes_total']}",
                "coverage_pct": round(file_coverage, 1),
                "undocumented": len(file_stats["undocumented_items"]),
            }

    total = total_functions + total_classes
    documented = documented_functions + documented_classes
    coverage_pct = (documented / total * 100) if total > 0 else 100.0

    return {
        "exists": True,
        "total_functions": total_functions,
        "documented_functions": documented_functions,
        "total_classes": total_classes,
        "documented_classes": documented_classes,
        "coverage_pct": round(coverage_pct, 2),
        "files": files_data,
    }


def save_history(results: dict[str, Any]) -> None:
    """
    Save coverage results to state history file.

    Args:
        results: Coverage results to save.
    """
    state_dir = Path(".claude/state")
    state_dir.mkdir(parents=True, exist_ok=True)

    state_file = state_dir / "docstring_coverage.json"

    history = []
    if state_file.exists():
        try:
            history = json.loads(state_file.read_text())
        except json.JSONDecodeError:
            history = []

    # Add new entry
    entry = {
        "date": datetime.now().isoformat(),
        "results": {
            module: {
                "coverage_pct": data["coverage_pct"],
                "functions": f"{data['documented_functions']}/{data['total_functions']}",
                "classes": f"{data['documented_classes']}/{data['total_classes']}",
            }
            for module, data in results.items()
            if data.get("exists", True)
        },
    }

    history.append(entry)
    history = history[-30:]  # Keep 30 days of history

    state_file.write_text(json.dumps(history, indent=2))


def print_summary(results: dict[str, Any], verbose: bool = False) -> None:
    """
    Print coverage summary to console.

    Args:
        results: Coverage results to display.
        verbose: Whether to show per-file breakdown.
    """
    print("\n" + "=" * 60)
    print("DOCSTRING COVERAGE REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60 + "\n")

    # Calculate overall stats
    overall_total = 0
    overall_documented = 0

    print(f"{'Module':<20} {'Coverage':>10} {'Functions':>15} {'Classes':>12}")
    print("-" * 60)

    for module, data in sorted(results.items()):
        if not data.get("exists", True):
            print(f"{module:<20} {'N/A':>10} {'(not found)':>15}")
            continue

        coverage = data["coverage_pct"]
        funcs = f"{data['documented_functions']}/{data['total_functions']}"
        classes = f"{data['documented_classes']}/{data['total_classes']}"

        # Color indicator
        indicator = "  " if coverage >= 90 else "! " if coverage >= 70 else "!!"
        print(f"{indicator}{module:<18} {coverage:>8.1f}% {funcs:>15} {classes:>12}")

        overall_total += data["total_functions"] + data["total_classes"]
        overall_documented += data["documented_functions"] + data["documented_classes"]

        if verbose and data.get("files"):
            for file_path, file_data in sorted(data["files"].items()):
                if file_data["undocumented"] > 0:
                    print(f"   - {file_path}: {file_data['coverage_pct']:.0f}% ({file_data['undocumented']} missing)")

    # Overall
    overall_pct = (overall_documented / overall_total * 100) if overall_total > 0 else 0
    print("-" * 60)
    print(f"{'OVERALL':<20} {overall_pct:>8.1f}% {overall_documented:>9}/{overall_total:<5}")

    # Target check
    print("\n" + "-" * 60)
    target = 95.0
    if overall_pct >= target:
        print(f"TARGET MET: {overall_pct:.1f}% >= {target:.0f}%")
    else:
        needed = int((target / 100 * overall_total) - overall_documented)
        print(f"TARGET GAP: {overall_pct:.1f}% < {target:.0f}% (need ~{needed} more docstrings)")
    print()


def main() -> int:
    """
    Main entry point for docstring coverage tracking.

    Returns:
        Exit code (0 for success).
    """
    # Parse args
    json_output = "--json" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # Modules to analyze
    modules = [
        "llm",
        "execution",
        "models",
        "observability",
        "evaluation",
        "algorithms",
        "scanners",
        "indicators",
        "config",
        "utils",
    ]

    # Calculate coverage
    results = {}
    for module in modules:
        results[module] = calculate_module_coverage(module)

    # Save history
    save_history(results)

    # Output
    if json_output:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results, verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
