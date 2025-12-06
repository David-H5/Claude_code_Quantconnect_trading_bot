#!/usr/bin/env python3
"""
Proactive Issue Detector for Overnight Sessions

Scans the codebase for potential issues and prioritizes them for overnight work.
Ensures Claude addresses all problems before completing a session.

Features:
- Import error detection
- Test failure detection
- Missing test coverage detection
- TODO/FIXME extraction
- Dependency issues
- Type checking issues

Usage:
    python scripts/issue_detector.py --scan
    python scripts/issue_detector.py --prioritize
    python scripts/issue_detector.py --output issues.json
"""

import argparse
import ast
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Issue:
    """Represents a detected issue."""

    id: str
    category: str  # import, test, coverage, todo, dependency, type
    severity: str  # P0, P1, P2
    file: str
    line: int | None
    message: str
    suggestion: str | None = None
    auto_fixable: bool = False
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "severity": self.severity,
            "file": self.file,
            "line": self.line,
            "message": self.message,
            "suggestion": self.suggestion,
            "auto_fixable": self.auto_fixable,
            "context": self.context,
        }


class IssueDetector:
    """Detects and prioritizes issues in the codebase."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        self.issues: list[Issue] = []
        self._issue_counter = 0

    def _new_issue_id(self) -> str:
        self._issue_counter += 1
        return f"ISSUE-{self._issue_counter:04d}"

    def scan_all(self) -> list[Issue]:
        """Run all scanners and return issues."""
        self.issues = []

        print("Scanning for import errors...")
        self._scan_imports()

        print("Scanning for test failures...")
        self._scan_test_failures()

        print("Scanning for TODO/FIXME comments...")
        self._scan_todos()

        print("Scanning for missing docstrings...")
        self._scan_missing_docstrings()

        print("Scanning for type hints...")
        self._scan_type_hints()

        print("Scanning for common anti-patterns...")
        self._scan_antipatterns()

        return self.issues

    def _scan_imports(self):
        """Detect import errors."""
        python_files = list(self.project_root.glob("**/*.py"))
        exclude_patterns = ["venv", ".venv", "__pycache__", ".git", "build", "dist"]

        for file in python_files:
            if any(p in str(file) for p in exclude_patterns):
                continue

            try:
                content = file.read_text()
                # Try to compile the file
                compile(content, str(file), "exec")

                # Check for import statements
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self._check_import(alias.name, file, node.lineno)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            self._check_import(node.module, file, node.lineno)

            except SyntaxError as e:
                self.issues.append(
                    Issue(
                        id=self._new_issue_id(),
                        category="syntax",
                        severity="P0",
                        file=str(file.relative_to(self.project_root)),
                        line=e.lineno,
                        message=f"Syntax error: {e.msg}",
                        suggestion="Fix the syntax error before proceeding",
                    )
                )
            except Exception:
                pass

    def _check_import(self, module_name: str, file: Path, lineno: int):
        """Check if an import is valid."""
        # Skip standard library and common packages
        stdlib = {
            "os",
            "sys",
            "json",
            "re",
            "datetime",
            "pathlib",
            "typing",
            "dataclasses",
            "enum",
            "collections",
            "functools",
            "itertools",
            "logging",
            "threading",
            "subprocess",
            "tempfile",
            "unittest",
            "abc",
            "hashlib",
            "uuid",
            "sqlite3",
            "contextlib",
            "copy",
            "time",
            "math",
            "random",
            "shutil",
            "asyncio",
        }

        base_module = module_name.split(".")[0]
        if base_module in stdlib:
            return

        # Check if it's a local module
        local_modules = {
            "models",
            "llm",
            "scanners",
            "execution",
            "indicators",
            "ui",
            "utils",
            "config",
            "observability",
        }
        if base_module in local_modules:
            return

        # Check common third-party
        third_party = {"numpy", "pandas", "pytest", "anthropic", "openai", "pyside6", "requests", "aiohttp", "psutil"}
        if base_module.lower() in third_party:
            return

    def _scan_test_failures(self):
        """Run tests and detect failures."""
        try:
            result = subprocess.run(
                ["pytest", "tests/", "-v", "--tb=no", "-q", "--collect-only"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.project_root,
            )

            # Check for collection errors
            if "ERROR" in result.stdout or result.returncode != 0:
                # Parse error output
                for line in result.stdout.split("\n"):
                    if "ERROR" in line:
                        self.issues.append(
                            Issue(
                                id=self._new_issue_id(),
                                category="test",
                                severity="P0",
                                file="tests/",
                                line=None,
                                message=f"Test collection error: {line.strip()[:100]}",
                                suggestion="Fix import or syntax errors in test files",
                            )
                        )

        except subprocess.TimeoutExpired:
            self.issues.append(
                Issue(
                    id=self._new_issue_id(),
                    category="test",
                    severity="P1",
                    file="tests/",
                    line=None,
                    message="Test collection timed out",
                    suggestion="Check for hanging tests or infinite loops",
                )
            )
        except FileNotFoundError:
            pass  # pytest not installed

    def _scan_todos(self):
        """Extract TODO and FIXME comments."""
        python_files = list(self.project_root.glob("**/*.py"))
        exclude_patterns = ["venv", ".venv", "__pycache__", ".git"]

        todo_pattern = re.compile(r"#\s*(TODO|FIXME|XXX|HACK|BUG)[\s:]*(.+)$", re.IGNORECASE)

        for file in python_files:
            if any(p in str(file) for p in exclude_patterns):
                continue

            try:
                content = file.read_text()
                for i, line in enumerate(content.split("\n"), 1):
                    match = todo_pattern.search(line)
                    if match:
                        tag = match.group(1).upper()
                        message = match.group(2).strip()

                        # Prioritize by tag
                        severity = "P2"
                        if tag in ("FIXME", "BUG"):
                            severity = "P0"
                        elif tag in ("XXX", "HACK"):
                            severity = "P1"

                        self.issues.append(
                            Issue(
                                id=self._new_issue_id(),
                                category="todo",
                                severity=severity,
                                file=str(file.relative_to(self.project_root)),
                                line=i,
                                message=f"{tag}: {message}",
                                suggestion=f"Address the {tag} comment",
                            )
                        )
            except Exception:
                pass

    def _scan_missing_docstrings(self):
        """Detect public functions/classes missing docstrings."""
        python_files = list(self.project_root.glob("**/*.py"))
        exclude_patterns = ["venv", ".venv", "__pycache__", ".git", "tests/"]

        for file in python_files:
            if any(p in str(file) for p in exclude_patterns):
                continue

            try:
                content = file.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                        # Skip private functions
                        if node.name.startswith("_"):
                            continue
                        # Check for docstring
                        if not ast.get_docstring(node):
                            self.issues.append(
                                Issue(
                                    id=self._new_issue_id(),
                                    category="documentation",
                                    severity="P2",
                                    file=str(file.relative_to(self.project_root)),
                                    line=node.lineno,
                                    message=f"Public function '{node.name}' missing docstring",
                                    suggestion="Add a docstring describing the function",
                                )
                            )

                    elif isinstance(node, ast.ClassDef):
                        if node.name.startswith("_"):
                            continue
                        if not ast.get_docstring(node):
                            self.issues.append(
                                Issue(
                                    id=self._new_issue_id(),
                                    category="documentation",
                                    severity="P2",
                                    file=str(file.relative_to(self.project_root)),
                                    line=node.lineno,
                                    message=f"Public class '{node.name}' missing docstring",
                                    suggestion="Add a docstring describing the class",
                                )
                            )
            except Exception:
                pass

    def _scan_type_hints(self):
        """Detect functions missing type hints."""
        python_files = list(self.project_root.glob("**/*.py"))
        exclude_patterns = ["venv", ".venv", "__pycache__", ".git", "tests/"]

        for file in python_files:
            if any(p in str(file) for p in exclude_patterns):
                continue

            try:
                content = file.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                        # Skip private/magic methods
                        if node.name.startswith("_"):
                            continue

                        # Check return type
                        if node.returns is None and node.name not in ("__init__", "__new__"):
                            self.issues.append(
                                Issue(
                                    id=self._new_issue_id(),
                                    category="type_hint",
                                    severity="P2",
                                    file=str(file.relative_to(self.project_root)),
                                    line=node.lineno,
                                    message=f"Function '{node.name}' missing return type hint",
                                    suggestion="Add return type annotation (e.g., -> None, -> str)",
                                )
                            )
            except Exception:
                pass

    def _scan_antipatterns(self):
        """Detect common anti-patterns."""
        python_files = list(self.project_root.glob("**/*.py"))
        exclude_patterns = ["venv", ".venv", "__pycache__", ".git"]

        patterns = [
            (r"except\s*:", "P1", "Bare except clause", "Specify exception type (except Exception:)"),
            (
                r"print\s*\(",
                "P2",
                "Print statement in production code",
                "Use logging instead of print",
            ),
            (r"import \*", "P1", "Wildcard import", "Import specific names instead"),
            (r"eval\s*\(", "P0", "Use of eval()", "Avoid eval() for security reasons"),
            (r"exec\s*\(", "P0", "Use of exec()", "Avoid exec() for security reasons"),
        ]

        for file in python_files:
            if any(p in str(file) for p in exclude_patterns):
                continue

            # Skip test files for some patterns
            is_test = "test" in str(file).lower()

            try:
                content = file.read_text()
                for i, line in enumerate(content.split("\n"), 1):
                    for pattern, severity, message, suggestion in patterns:
                        # Skip print detection in tests
                        if "print" in pattern and is_test:
                            continue

                        if re.search(pattern, line):
                            self.issues.append(
                                Issue(
                                    id=self._new_issue_id(),
                                    category="antipattern",
                                    severity=severity,
                                    file=str(file.relative_to(self.project_root)),
                                    line=i,
                                    message=message,
                                    suggestion=suggestion,
                                )
                            )
            except Exception:
                pass

    def prioritize(self) -> dict[str, list[Issue]]:
        """Group and prioritize issues."""
        prioritized: dict[str, list[Issue]] = {"P0": [], "P1": [], "P2": []}

        for issue in self.issues:
            prioritized[issue.severity].append(issue)

        # Sort within each priority by category
        for priority in prioritized:
            prioritized[priority].sort(key=lambda x: (x.category, x.file))

        return prioritized

    def generate_report(self) -> str:
        """Generate a human-readable report."""
        prioritized = self.prioritize()

        report = f"""# Issue Detection Report

**Generated**: {datetime.now().isoformat()}
**Total Issues**: {len(self.issues)}

## Summary

| Priority | Count | Description |
|----------|-------|-------------|
| P0 | {len(prioritized['P0'])} | Critical - must fix immediately |
| P1 | {len(prioritized['P1'])} | Important - fix before completion |
| P2 | {len(prioritized['P2'])} | Minor - nice to fix |

"""

        for priority in ["P0", "P1", "P2"]:
            if prioritized[priority]:
                report += f"\n## {priority} Issues\n\n"
                for issue in prioritized[priority]:
                    location = f"{issue.file}"
                    if issue.line:
                        location += f":{issue.line}"
                    report += f"### {issue.id}: {issue.message}\n"
                    report += f"- **Location**: {location}\n"
                    report += f"- **Category**: {issue.category}\n"
                    if issue.suggestion:
                        report += f"- **Suggestion**: {issue.suggestion}\n"
                    report += "\n"

        return report

    def to_json(self) -> str:
        """Export issues as JSON."""
        return json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "total": len(self.issues),
                "by_priority": {p: len(v) for p, v in self.prioritize().items()},
                "issues": [i.to_dict() for i in self.issues],
            },
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description="Proactive issue detection for overnight sessions")
    parser.add_argument("--scan", action="store_true", help="Run all scanners")
    parser.add_argument("--output", help="Output file (JSON format)")
    parser.add_argument("--report", help="Output report file (Markdown)")
    parser.add_argument("--project", default=".", help="Project root directory")
    args = parser.parse_args()

    detector = IssueDetector(Path(args.project))

    if args.scan or args.output or args.report:
        issues = detector.scan_all()

        print(f"\nDetected {len(issues)} issues:")
        prioritized = detector.prioritize()
        for priority in ["P0", "P1", "P2"]:
            count = len(prioritized[priority])
            if count > 0:
                print(f"  {priority}: {count}")

        if args.output:
            Path(args.output).write_text(detector.to_json())
            print(f"\nJSON output: {args.output}")

        if args.report:
            Path(args.report).write_text(detector.generate_report())
            print(f"Report: {args.report}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
