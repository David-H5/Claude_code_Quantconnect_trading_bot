#!/usr/bin/env python3
"""
Layer Violation Checker for Module Architecture

Enforces the 5-layer architecture defined in NEXT_REFACTOR_PLAN.md:

Layer 4: Applications (algorithms/, api/, ui/)
    ↓
Layer 3: Domain Logic (execution/, llm/, evaluation/, scanners/, indicators/)
    ↓
Layer 2: Core Models (models/, compliance/)
    ↓
Layer 1: Infrastructure (observability/, infrastructure/, config/)
    ↓
Layer 0: Utilities (utils/)

Rules:
- Each layer can only import from layers below it
- No circular dependencies within a layer (cross-module)
- Cross-cutting concerns use dependency injection

Usage:
    python scripts/check_layer_violations.py [--fix] [--verbose]

Author: QuantConnect Trading Bot
Date: 2025-12-05
"""

import argparse
import ast
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# Layer definitions (higher number = higher layer)
LAYER_DEFINITIONS = {
    # Layer 0: Utilities
    "utils": 0,
    # Layer 1: Infrastructure
    "observability": 1,
    "infrastructure": 1,
    "config": 1,
    # Layer 2: Core Models
    "models": 2,
    "compliance": 2,
    # Layer 3: Domain Logic
    "execution": 3,
    "llm": 3,
    "evaluation": 3,
    "scanners": 3,
    "indicators": 3,
    # Layer 4: Applications
    "algorithms": 4,
    "api": 4,
    "ui": 4,
    # Special: Tests and scripts are allowed to import anything
    "tests": 99,
    "scripts": 99,
    ".claude": 99,
    "mcp": 99,
    "docs": 99,
}

# Known acceptable exceptions (legacy code, will be fixed later)
ALLOWED_EXCEPTIONS = {
    # (source_module, target_module): "reason"
    ("observability", "evaluation"): "Legacy: anomaly bridge imports evaluation metrics",
    ("utils", "observability"): "Legacy: re-exports for backwards compatibility",
    ("llm", "evaluation"): "Legacy: agent metrics import",
    ("observability", "models"): "Infrastructure: exception logger imports TradingError for structured logging",
}


@dataclass
class Violation:
    """Represents a layer violation."""

    source_file: Path
    source_module: str
    source_layer: int
    target_module: str
    target_layer: int
    import_line: int
    import_statement: str

    def __str__(self) -> str:
        return (
            f"{self.source_file}:{self.import_line}: "
            f"Layer violation: {self.source_module} (L{self.source_layer}) "
            f"imports {self.target_module} (L{self.target_layer})"
        )


def get_module_layer(module_name: str) -> int | None:
    """Get the layer number for a module."""
    # Handle submodules (e.g., observability.metrics -> observability)
    top_level = module_name.split(".")[0]
    return LAYER_DEFINITIONS.get(top_level)


def extract_imports(file_path: Path) -> list[tuple[str, int, str]]:
    """Extract imports from a Python file.

    Returns:
        List of (module_name, line_number, import_statement) tuples
    """
    imports = []
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, node.lineno, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom):
                # Skip relative imports (level > 0 means relative import)
                # e.g., "from .execution import X" has level=1
                if node.module and node.level == 0:
                    stmt = f"from {node.module} import ..."
                    imports.append((node.module, node.lineno, stmt))
    except (SyntaxError, UnicodeDecodeError):
        pass

    return imports


def check_file(file_path: Path, verbose: bool = False) -> list[Violation]:
    """Check a single file for layer violations."""
    violations = []

    # Determine source module
    parts = file_path.parts
    if not parts:
        return violations

    # Find the top-level project module
    source_module = None
    for part in parts:
        if part in LAYER_DEFINITIONS:
            source_module = part
            break

    if not source_module:
        return violations

    source_layer = LAYER_DEFINITIONS.get(source_module)
    if source_layer is None or source_layer >= 99:
        return violations  # Skip tests, scripts, etc.

    # Check imports
    imports = extract_imports(file_path)
    for module_name, line_no, import_stmt in imports:
        target_layer = get_module_layer(module_name)
        if target_layer is None:
            continue  # External module

        # Check for violation (importing from higher or same layer cross-module)
        target_module = module_name.split(".")[0]

        if target_layer > source_layer:
            # Definite violation: importing from higher layer
            violation = Violation(
                source_file=file_path,
                source_module=source_module,
                source_layer=source_layer,
                target_module=target_module,
                target_layer=target_layer,
                import_line=line_no,
                import_statement=import_stmt,
            )
            violations.append(violation)
            if verbose:
                print(f"  VIOLATION: {violation}")

        elif target_layer == source_layer and target_module != source_module:
            # Same-layer cross-module import - may be acceptable in some cases
            # but we flag it for review
            if verbose:
                print(f"  WARNING: Same-layer import: {source_module} -> {target_module}")

    return violations


def check_project(project_root: Path, verbose: bool = False) -> dict[str, list[Violation]]:
    """Check all Python files in the project."""
    all_violations: dict[str, list[Violation]] = defaultdict(list)

    for py_file in project_root.rglob("*.py"):
        # Skip virtual environments, cache, etc.
        if any(
            part in py_file.parts
            for part in [
                "venv",
                ".venv",
                "__pycache__",
                ".git",
                "node_modules",
                ".hypothesis",
            ]
        ):
            continue

        violations = check_file(py_file, verbose)
        for v in violations:
            # Check if this is an allowed exception
            key = (v.source_module, v.target_module)
            if key not in ALLOWED_EXCEPTIONS:
                all_violations[v.source_module].append(v)

    return all_violations


def print_report(violations: dict[str, list[Violation]]) -> None:
    """Print a summary report of violations."""
    print("=" * 70)
    print("LAYER VIOLATION REPORT")
    print("=" * 70)
    print()

    if not violations:
        print("No layer violations found!")
        print()
        return

    total = sum(len(v) for v in violations.values())
    print(f"Total violations: {total}")
    print()

    # Group by source module
    for module in sorted(violations.keys()):
        module_violations = violations[module]
        print(f"\n{module}/ ({len(module_violations)} violations):")
        print("-" * 50)

        for v in module_violations[:10]:  # Show first 10
            print(f"  {v.source_file.name}:{v.import_line}")
            print(f"    → imports {v.target_module} (L{v.target_layer})")

        if len(module_violations) > 10:
            print(f"  ... and {len(module_violations) - 10} more")


def print_layer_diagram() -> None:
    """Print the layer architecture diagram."""
    print()
    print("LAYER ARCHITECTURE")
    print("=" * 70)
    print(
        """
Layer 4: Applications
  ├── algorithms/
  ├── api/
  └── ui/
        ↓
Layer 3: Domain Logic
  ├── execution/
  ├── llm/
  ├── evaluation/
  ├── scanners/
  └── indicators/
        ↓
Layer 2: Core Models
  ├── models/
  └── compliance/
        ↓
Layer 1: Infrastructure
  ├── observability/
  ├── infrastructure/
  └── config/
        ↓
Layer 0: Utilities
  └── utils/

Rules:
  • Each layer can only import from layers BELOW it
  • No upward dependencies allowed
  • Same-layer cross-module imports should be minimized
"""
    )


def main():
    parser = argparse.ArgumentParser(description="Check for layer architecture violations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--diagram", "-d", action="store_true", help="Show layer diagram")
    parser.add_argument(
        "--strict",
        "-s",
        action="store_true",
        help="Exit with error on any violation",
    )
    args = parser.parse_args()

    if args.diagram:
        print_layer_diagram()
        return

    project_root = Path(__file__).parent.parent

    print("Checking layer violations...")
    print()

    violations = check_project(project_root, args.verbose)
    print_report(violations)

    # Print allowed exceptions
    if ALLOWED_EXCEPTIONS:
        print("\nALLOWED EXCEPTIONS (legacy code):")
        print("-" * 50)
        for (src, tgt), reason in ALLOWED_EXCEPTIONS.items():
            print(f"  {src} → {tgt}: {reason}")

    print()
    print_layer_diagram()

    if violations and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
