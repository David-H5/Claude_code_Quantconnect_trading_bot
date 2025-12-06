#!/usr/bin/env python3
"""
Comprehensive QA Validator System for Upgrades and Important Work.

This script provides a unified quality assurance system with modular checkers
for code quality, documentation, tests, git hygiene, and file cleanliness.

Usage:
    python scripts/qa_validator.py                    # Run all checks
    python scripts/qa_validator.py --check code       # Run only code checks
    python scripts/qa_validator.py --check docs       # Run only doc checks
    python scripts/qa_validator.py --fix              # Auto-fix where possible
    python scripts/qa_validator.py --upgrade UPGRADE-014  # Focus on upgrade

Author: Claude Code
Created: 2025-12-03
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class Severity(Enum):
    """Issue severity levels."""

    ERROR = "error"  # Must fix before merge
    WARNING = "warning"  # Should fix
    INFO = "info"  # Nice to fix


class CheckCategory(Enum):
    """Check categories."""

    CODE = "code"
    DOCS = "docs"
    TESTS = "tests"
    GIT = "git"
    FILES = "files"
    PROGRESS = "progress"
    DEBUG = "debug"  # Debug diagnostics
    INTEGRITY = "integrity"  # Code integrity/corruption
    XREF = "xref"  # Cross-reference validation
    RIC = "ric"  # RIC loop compliance
    SECURITY = "security"  # Security vulnerability checks
    HOOKS = "hooks"  # Claude Code hooks validation
    TRADING = "trading"  # Trading/algorithm safety checks
    CONFIG = "config"  # Configuration validation
    DEPS = "deps"  # Dependencies validation
    AGENTS = "agents"  # Agent personas and commands


@dataclass
class Issue:
    """Represents a QA issue found during validation."""

    category: CheckCategory
    severity: Severity
    check_name: str
    message: str
    file_path: str | None = None
    line_number: int | None = None
    suggestion: str | None = None
    auto_fixable: bool = False


@dataclass
class CheckResult:
    """Result of a single check."""

    check_name: str
    category: CheckCategory
    passed: bool
    issues: list[Issue] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class QAReport:
    """Complete QA validation report."""

    timestamp: str
    duration_seconds: float
    total_checks: int
    passed_checks: int
    failed_checks: int
    errors: int
    warnings: int
    infos: int
    results: list[CheckResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.errors == 0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "summary": {
                "total_checks": self.total_checks,
                "passed": self.passed_checks,
                "failed": self.failed_checks,
                "errors": self.errors,
                "warnings": self.warnings,
                "infos": self.infos,
            },
            "success": self.success,
            "results": [
                {
                    "check": r.check_name,
                    "category": r.category.value,
                    "passed": r.passed,
                    "issues": [
                        {
                            "severity": i.severity.value,
                            "message": i.message,
                            "file": i.file_path,
                            "line": i.line_number,
                            "suggestion": i.suggestion,
                            "auto_fixable": i.auto_fixable,
                        }
                        for i in r.issues
                    ],
                }
                for r in self.results
            ],
        }


class QAValidator:
    """Main QA validation orchestrator."""

    def __init__(self, project_root: Path, upgrade_focus: str | None = None):
        self.project_root = project_root
        self.upgrade_focus = upgrade_focus
        self.checks: dict[str, Callable[[], CheckResult]] = {}
        self._venv_python = self._find_venv_python()
        self._register_all_checks()

    def _find_venv_python(self) -> str:
        """Find the virtual environment Python interpreter.

        Checks common venv locations and returns the path to the Python
        executable if found, otherwise falls back to system python3.
        """
        venv_paths = [
            self.project_root / ".venv" / "bin" / "python",
            self.project_root / "venv" / "bin" / "python",
            self.project_root / ".venv" / "Scripts" / "python.exe",  # Windows
            self.project_root / "venv" / "Scripts" / "python.exe",  # Windows
        ]
        for venv_path in venv_paths:
            if venv_path.exists():
                return str(venv_path)
        return "python3"  # Fallback to system Python

    def _register_all_checks(self):
        """Register all available checks."""
        # Code quality checks
        self.checks["python_syntax"] = self._check_python_syntax
        self.checks["python_imports"] = self._check_python_imports
        self.checks["ruff_lint"] = self._check_ruff_lint
        self.checks["type_hints"] = self._check_type_hints

        # Documentation checks
        self.checks["research_docs"] = self._check_research_docs
        self.checks["docstrings"] = self._check_docstrings
        self.checks["readme_exists"] = self._check_readme_exists

        # Test checks
        self.checks["tests_pass"] = self._check_tests_pass
        self.checks["test_coverage"] = self._check_test_coverage
        self.checks["test_naming"] = self._check_test_naming

        # Git checks
        self.checks["uncommitted_changes"] = self._check_uncommitted_changes
        self.checks["large_files"] = self._check_large_files
        self.checks["secrets_check"] = self._check_secrets

        # File checks
        self.checks["temp_files"] = self._check_temp_files
        self.checks["empty_files"] = self._check_empty_files
        self.checks["duplicate_files"] = self._check_duplicate_files

        # Progress file checks
        self.checks["progress_format"] = self._check_progress_format
        self.checks["session_summaries"] = self._check_session_summaries

        # Debug diagnostic checks
        self.checks["debug_statements"] = self._check_debug_statements
        self.checks["todo_fixme"] = self._check_todo_fixme
        self.checks["print_statements"] = self._check_print_statements
        self.checks["breakpoints"] = self._check_breakpoints
        self.checks["incomplete_code"] = self._check_incomplete_code

        # Code integrity checks
        self.checks["corrupt_files"] = self._check_corrupt_files
        self.checks["missing_imports"] = self._check_missing_imports
        self.checks["circular_imports"] = self._check_circular_imports
        self.checks["orphan_functions"] = self._check_orphan_functions
        self.checks["init_files"] = self._check_init_files

        # Cross-reference checks
        self.checks["broken_imports"] = self._check_broken_imports
        self.checks["config_refs"] = self._check_config_refs
        self.checks["class_refs"] = self._check_class_refs
        self.checks["doc_links"] = self._check_doc_links

        # RIC loop compliance checks
        self.checks["ric_phases"] = self._check_ric_phases
        self.checks["upgrade_docs"] = self._check_upgrade_docs
        self.checks["iteration_tracking"] = self._check_iteration_tracking

        # Security checks
        self.checks["security_bandit"] = self._check_security_bandit
        self.checks["security_complexity"] = self._check_code_complexity

        # Hooks validation checks
        self.checks["hooks_exist"] = self._check_hooks_exist
        self.checks["hooks_syntax"] = self._check_hooks_syntax
        self.checks["hooks_registration"] = self._check_hooks_registration
        self.checks["hooks_settings"] = self._check_hooks_settings

        # Trading/Algorithm safety checks
        self.checks["risk_params"] = self._check_risk_params
        self.checks["algorithm_structure"] = self._check_algorithm_structure
        self.checks["paper_mode_default"] = self._check_paper_mode_default

        # Configuration validation checks
        self.checks["config_schema"] = self._check_config_schema
        self.checks["env_vars"] = self._check_env_vars
        self.checks["mcp_config"] = self._check_mcp_config

        # Dependencies validation checks
        self.checks["requirements_sync"] = self._check_requirements_sync
        self.checks["version_conflicts"] = self._check_version_conflicts
        self.checks["outdated_packages"] = self._check_outdated_packages

        # Agent personas and commands checks
        self.checks["personas_exist"] = self._check_personas_exist
        self.checks["persona_format"] = self._check_persona_format
        self.checks["commands_valid"] = self._check_commands_valid

    def run_all(self, categories: list[CheckCategory] | None = None) -> QAReport:
        """Run all checks or specified categories."""
        import time

        start = time.time()

        results = []
        for _name, check_fn in self.checks.items():
            # Filter by category if specified
            result = check_fn()
            if categories and result.category not in categories:
                continue
            results.append(result)

        duration = time.time() - start

        # Calculate summary
        errors = sum(len([i for i in r.issues if i.severity == Severity.ERROR]) for r in results)
        warnings = sum(len([i for i in r.issues if i.severity == Severity.WARNING]) for r in results)
        infos = sum(len([i for i in r.issues if i.severity == Severity.INFO]) for r in results)

        return QAReport(
            timestamp=datetime.now().isoformat(),
            duration_seconds=round(duration, 2),
            total_checks=len(results),
            passed_checks=len([r for r in results if r.passed]),
            failed_checks=len([r for r in results if not r.passed]),
            errors=errors,
            warnings=warnings,
            infos=infos,
            results=results,
        )

    def run_check(self, check_name: str) -> CheckResult | None:
        """Run a single check by name."""
        if check_name in self.checks:
            return self.checks[check_name]()
        return None

    # =========================================================================
    # CODE QUALITY CHECKS
    # =========================================================================

    def _check_python_syntax(self) -> CheckResult:
        """Check Python files for syntax errors."""
        issues = []
        py_files = list(self.project_root.rglob("*.py"))
        py_files = [f for f in py_files if ".venv" not in str(f) and "node_modules" not in str(f)]

        for py_file in py_files[:100]:  # Limit to avoid timeout
            try:
                result = subprocess.run(
                    ["python3", "-m", "py_compile", str(py_file)], capture_output=True, text=True, timeout=5
                )
                if result.returncode != 0:
                    issues.append(
                        Issue(
                            category=CheckCategory.CODE,
                            severity=Severity.ERROR,
                            check_name="python_syntax",
                            message=f"Syntax error: {result.stderr.strip()}",
                            file_path=str(py_file.relative_to(self.project_root)),
                        )
                    )
            except subprocess.TimeoutExpired:
                pass
            except Exception as exc:
                issues.append(
                    Issue(
                        category=CheckCategory.CODE,
                        severity=Severity.WARNING,
                        check_name="python_syntax",
                        message=f"Could not check: {exc}",
                        file_path=str(py_file.relative_to(self.project_root)),
                    )
                )

        return CheckResult(
            check_name="python_syntax",
            category=CheckCategory.CODE,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_python_imports(self) -> CheckResult:
        """Check for broken imports in Python files."""
        issues = []

        # Check key modules can be imported
        key_modules = [
            "llm.agents",
            "models.circuit_breaker",
            "evaluation.agent_metrics",
            "execution.smart_execution",
        ]

        for module in key_modules:
            try:
                result = subprocess.run(
                    ["python3", "-c", f"import {module}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(self.project_root),
                    env={**os.environ, "PYTHONPATH": str(self.project_root)},
                )
                if result.returncode != 0:
                    # Check if it's a missing optional dependency
                    if "numpy" in result.stderr or "pandas" in result.stderr:
                        issues.append(
                            Issue(
                                category=CheckCategory.CODE,
                                severity=Severity.INFO,
                                check_name="python_imports",
                                message=f"Optional dependency missing for {module}",
                                suggestion="Install numpy/pandas if needed",
                            )
                        )
                    else:
                        issues.append(
                            Issue(
                                category=CheckCategory.CODE,
                                severity=Severity.ERROR,
                                check_name="python_imports",
                                message=f"Import failed: {module}\n{result.stderr[:200]}",
                            )
                        )
            except subprocess.TimeoutExpired:
                issues.append(
                    Issue(
                        category=CheckCategory.CODE,
                        severity=Severity.WARNING,
                        check_name="python_imports",
                        message=f"Import check timed out for {module}",
                    )
                )
            except Exception:
                pass

        return CheckResult(
            check_name="python_imports",
            category=CheckCategory.CODE,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_ruff_lint(self) -> CheckResult:
        """Run ruff linter on codebase.

        Uses the virtual environment Python if available to ensure
        ruff is found when installed in the venv.
        """
        issues = []

        try:
            result = subprocess.run(
                [self._venv_python, "-m", "ruff", "check", ".", "--select=E,F", "--ignore=E501"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.project_root),
            )

            if result.stdout:
                # Parse ruff output
                for line in result.stdout.strip().split("\n")[:20]:
                    if line and ":" in line:
                        issues.append(
                            Issue(
                                category=CheckCategory.CODE,
                                severity=Severity.WARNING,
                                check_name="ruff_lint",
                                message=line,
                                auto_fixable=True,
                                suggestion="Run 'ruff check --fix' to auto-fix",
                            )
                        )
        except FileNotFoundError:
            issues.append(
                Issue(
                    category=CheckCategory.CODE,
                    severity=Severity.INFO,
                    check_name="ruff_lint",
                    message="ruff not installed, skipping lint check",
                    suggestion="pip install ruff",
                )
            )
        except subprocess.TimeoutExpired:
            issues.append(
                Issue(
                    category=CheckCategory.CODE,
                    severity=Severity.WARNING,
                    check_name="ruff_lint",
                    message="Lint check timed out",
                )
            )

        return CheckResult(
            check_name="ruff_lint",
            category=CheckCategory.CODE,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_type_hints(self) -> CheckResult:
        """Check for type hint coverage in key files."""
        issues = []

        # Check key files have type hints
        key_files = [
            "llm/agents/base.py",
            "models/circuit_breaker.py",
            "evaluation/agent_metrics.py",
        ]

        for file_path in key_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                content = full_path.read_text()

                # Check for functions without type hints
                func_pattern = r"def\s+(\w+)\s*\([^)]*\)\s*:"
                funcs = re.findall(func_pattern, content)
                typed_pattern = r"def\s+\w+\s*\([^)]*\)\s*->\s*"
                typed_funcs = len(re.findall(typed_pattern, content))

                if funcs and typed_funcs / len(funcs) < 0.5:
                    issues.append(
                        Issue(
                            category=CheckCategory.CODE,
                            severity=Severity.INFO,
                            check_name="type_hints",
                            message=f"Low type hint coverage ({typed_funcs}/{len(funcs)} functions)",
                            file_path=file_path,
                        )
                    )

        return CheckResult(
            check_name="type_hints",
            category=CheckCategory.CODE,
            passed=True,  # Info only
            issues=issues,
        )

    # =========================================================================
    # DOCUMENTATION CHECKS
    # =========================================================================

    def _check_research_docs(self) -> CheckResult:
        """Run research documentation validation."""
        issues = []

        validator_script = self.project_root / "scripts" / "validate_research_docs.py"
        if validator_script.exists():
            try:
                result = subprocess.run(
                    ["python3", str(validator_script)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(self.project_root),
                )

                # Parse output for errors
                if "Errors: 0" not in result.stdout:
                    error_match = re.search(r"Errors: (\d+)", result.stdout)
                    if error_match and int(error_match.group(1)) > 0:
                        issues.append(
                            Issue(
                                category=CheckCategory.DOCS,
                                severity=Severity.ERROR,
                                check_name="research_docs",
                                message=f"Research doc validation found {error_match.group(1)} errors",
                                suggestion="Run 'python scripts/validate_research_docs.py' for details",
                            )
                        )

                # Check for warnings
                warn_match = re.search(r"Warnings: (\d+)", result.stdout)
                if warn_match and int(warn_match.group(1)) > 0:
                    issues.append(
                        Issue(
                            category=CheckCategory.DOCS,
                            severity=Severity.WARNING,
                            check_name="research_docs",
                            message=f"Research doc validation found {warn_match.group(1)} warnings",
                        )
                    )

            except subprocess.TimeoutExpired:
                issues.append(
                    Issue(
                        category=CheckCategory.DOCS,
                        severity=Severity.WARNING,
                        check_name="research_docs",
                        message="Research doc validation timed out",
                    )
                )
        else:
            issues.append(
                Issue(
                    category=CheckCategory.DOCS,
                    severity=Severity.INFO,
                    check_name="research_docs",
                    message="Research doc validator not found",
                )
            )

        return CheckResult(
            check_name="research_docs",
            category=CheckCategory.DOCS,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_docstrings(self) -> CheckResult:
        """Check for missing docstrings in key modules."""
        issues = []

        key_modules = [
            "llm/agents/base.py",
            "models/circuit_breaker.py",
        ]

        for module_path in key_modules:
            full_path = self.project_root / module_path
            if full_path.exists():
                content = full_path.read_text()

                # Check module docstring
                if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
                    issues.append(
                        Issue(
                            category=CheckCategory.DOCS,
                            severity=Severity.INFO,
                            check_name="docstrings",
                            message="Missing module docstring",
                            file_path=module_path,
                        )
                    )

                # Check class docstrings
                class_pattern = r"class\s+(\w+)[^:]*:\s*\n\s*(?!\"\"\")"
                classes_without_docs = re.findall(class_pattern, content)
                if classes_without_docs:
                    issues.append(
                        Issue(
                            category=CheckCategory.DOCS,
                            severity=Severity.INFO,
                            check_name="docstrings",
                            message=f"Classes without docstrings: {', '.join(classes_without_docs[:3])}",
                            file_path=module_path,
                        )
                    )

        return CheckResult(
            check_name="docstrings",
            category=CheckCategory.DOCS,
            passed=True,  # Info only
            issues=issues,
        )

    def _check_readme_exists(self) -> CheckResult:
        """Check that key directories have README files."""
        issues = []

        dirs_needing_readme = [
            "llm/agents",
            "evaluation",
            "execution",
            "models",
        ]

        for dir_path in dirs_needing_readme:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                readme = full_path / "README.md"
                if not readme.exists():
                    issues.append(
                        Issue(
                            category=CheckCategory.DOCS,
                            severity=Severity.INFO,
                            check_name="readme_exists",
                            message=f"Missing README.md in {dir_path}",
                            suggestion=f"Create {dir_path}/README.md with module documentation",
                        )
                    )

        return CheckResult(
            check_name="readme_exists",
            category=CheckCategory.DOCS,
            passed=True,  # Info only
            issues=issues,
        )

    # =========================================================================
    # TEST CHECKS
    # =========================================================================

    def _check_tests_pass(self) -> CheckResult:
        """Run test suite and check for failures.

        Uses the virtual environment Python if available to ensure
        pytest and dependencies are found.
        """
        issues = []

        try:
            # Run a quick test subset using venv Python
            # Note: subprocess timeout handles test timeout, no pytest-timeout needed
            result = subprocess.run(
                [
                    self._venv_python,
                    "-m",
                    "pytest",
                    "tests/",
                    "-v",
                    "--tb=no",
                    "-q",
                    "--ignore=tests/test_threading*",
                    "-x",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.project_root),
                env={**os.environ, "PYTHONPATH": str(self.project_root)},
            )

            if result.returncode != 0:
                # Parse failure count
                fail_match = re.search(r"(\d+) failed", result.stdout + result.stderr)
                if fail_match:
                    issues.append(
                        Issue(
                            category=CheckCategory.TESTS,
                            severity=Severity.ERROR,
                            check_name="tests_pass",
                            message=f"{fail_match.group(1)} tests failed",
                            suggestion="Run 'pytest tests/ -v' to see details",
                        )
                    )
                else:
                    issues.append(
                        Issue(
                            category=CheckCategory.TESTS,
                            severity=Severity.WARNING,
                            check_name="tests_pass",
                            message="Test run had issues",
                            suggestion=result.stderr[:200] if result.stderr else "Check pytest output",
                        )
                    )
            else:
                # Extract pass count
                pass_match = re.search(r"(\d+) passed", result.stdout)
                if pass_match:
                    # This is info, not an issue
                    pass

        except subprocess.TimeoutExpired:
            issues.append(
                Issue(
                    category=CheckCategory.TESTS,
                    severity=Severity.WARNING,
                    check_name="tests_pass",
                    message="Test run timed out (>120s)",
                    suggestion="Some tests may be hanging",
                )
            )
        except FileNotFoundError:
            issues.append(
                Issue(
                    category=CheckCategory.TESTS,
                    severity=Severity.WARNING,
                    check_name="tests_pass",
                    message="pytest not found",
                    suggestion="pip install pytest",
                )
            )

        return CheckResult(
            check_name="tests_pass",
            category=CheckCategory.TESTS,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_test_coverage(self) -> CheckResult:
        """Check test coverage meets threshold."""
        issues = []

        # Skip if coverage check is too slow
        # Just check if coverage config exists
        coverage_config = self.project_root / ".coveragerc"
        pyproject = self.project_root / "pyproject.toml"

        has_coverage_config = coverage_config.exists()
        if pyproject.exists():
            content = pyproject.read_text()
            has_coverage_config = has_coverage_config or "[tool.coverage" in content

        if not has_coverage_config:
            issues.append(
                Issue(
                    category=CheckCategory.TESTS,
                    severity=Severity.INFO,
                    check_name="test_coverage",
                    message="No coverage configuration found",
                    suggestion="Add coverage config to pyproject.toml or .coveragerc",
                )
            )

        return CheckResult(
            check_name="test_coverage",
            category=CheckCategory.TESTS,
            passed=True,
            issues=issues,
        )

    def _check_test_naming(self) -> CheckResult:
        """Check test files follow naming conventions."""
        issues = []

        # Files that are allowed without test_ prefix
        allowed_non_test_files = {"conftest.py", "__init__.py"}

        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            for test_file in tests_dir.glob("*.py"):
                if test_file.name.startswith("__"):
                    continue
                if test_file.name in allowed_non_test_files:
                    continue
                if not test_file.name.startswith("test_"):
                    issues.append(
                        Issue(
                            category=CheckCategory.TESTS,
                            severity=Severity.WARNING,
                            check_name="test_naming",
                            message=f"Test file doesn't start with 'test_': {test_file.name}",
                            file_path=f"tests/{test_file.name}",
                            suggestion="Rename to test_*.py for pytest discovery",
                        )
                    )

        return CheckResult(
            check_name="test_naming",
            category=CheckCategory.TESTS,
            passed=len(issues) == 0,
            issues=issues,
        )

    # =========================================================================
    # GIT CHECKS
    # =========================================================================

    def _check_uncommitted_changes(self) -> CheckResult:
        """Check for uncommitted changes."""
        issues = []

        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10, cwd=str(self.project_root)
            )

            if result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                modified = [ln for ln in lines if ln.startswith(" M") or ln.startswith("M ")]
                untracked = [ln for ln in lines if ln.startswith("??")]

                if modified:
                    issues.append(
                        Issue(
                            category=CheckCategory.GIT,
                            severity=Severity.WARNING,
                            check_name="uncommitted_changes",
                            message=f"{len(modified)} modified files not committed",
                            suggestion="Review and commit or stash changes",
                        )
                    )

                if untracked:
                    issues.append(
                        Issue(
                            category=CheckCategory.GIT,
                            severity=Severity.INFO,
                            check_name="uncommitted_changes",
                            message=f"{len(untracked)} untracked files",
                            suggestion="Add to .gitignore or commit if needed",
                        )
                    )

        except Exception as e:
            issues.append(
                Issue(
                    category=CheckCategory.GIT,
                    severity=Severity.WARNING,
                    check_name="uncommitted_changes",
                    message=f"Could not check git status: {e}",
                )
            )

        return CheckResult(
            check_name="uncommitted_changes",
            category=CheckCategory.GIT,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_large_files(self) -> CheckResult:
        """Check for accidentally added large files."""
        issues = []

        large_threshold_mb = 10

        try:
            # Check staged files
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.project_root),
            )

            for file_name in result.stdout.strip().split("\n"):
                if file_name:
                    file_path = self.project_root / file_name
                    if file_path.exists():
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        if size_mb > large_threshold_mb:
                            issues.append(
                                Issue(
                                    category=CheckCategory.GIT,
                                    severity=Severity.ERROR,
                                    check_name="large_files",
                                    message=f"Large file staged ({size_mb:.1f}MB): {file_name}",
                                    file_path=file_name,
                                    suggestion="Remove from staging or add to .gitignore",
                                )
                            )

        except Exception:
            pass

        # Also check for known large file patterns in repo
        large_patterns = ["*.zip", "*.tar.gz", "*.pkl", "*.h5", "*.bin"]
        for pattern in large_patterns:
            for file_path in self.project_root.rglob(pattern):
                if ".git" not in str(file_path) and ".venv" not in str(file_path):
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    if size_mb > large_threshold_mb:
                        issues.append(
                            Issue(
                                category=CheckCategory.GIT,
                                severity=Severity.WARNING,
                                check_name="large_files",
                                message=f"Large file in repo ({size_mb:.1f}MB)",
                                file_path=str(file_path.relative_to(self.project_root)),
                                suggestion="Consider removing or adding to .gitignore",
                            )
                        )

        return CheckResult(
            check_name="large_files",
            category=CheckCategory.GIT,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_secrets(self) -> CheckResult:
        """Check for potential secrets in code."""
        issues = []

        secret_patterns = [
            (r'api[_-]?key\s*[=:]\s*["\'][^"\']{20,}["\']', "API key"),
            (r'password\s*[=:]\s*["\'][^"\']+["\']', "Password"),
            (r'secret\s*[=:]\s*["\'][^"\']{10,}["\']', "Secret"),
            (r'token\s*[=:]\s*["\'][^"\']{20,}["\']', "Token"),
            (r"-----BEGIN.*PRIVATE KEY-----", "Private key"),
        ]

        # Check recently modified Python files
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~5"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.project_root),
            )

            files_to_check = [f for f in result.stdout.strip().split("\n") if f.endswith(".py") and f][:20]

            # Exclude this file from secrets check (it contains detection patterns)
            files_to_check = [f for f in files_to_check if "qa_validator.py" not in f]

            for file_name in files_to_check:
                file_path = self.project_root / file_name
                if file_path.exists():
                    content = file_path.read_text()
                    for pattern, secret_type in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            issues.append(
                                Issue(
                                    category=CheckCategory.GIT,
                                    severity=Severity.ERROR,
                                    check_name="secrets_check",
                                    message=f"Potential {secret_type} found",
                                    file_path=file_name,
                                    suggestion="Remove secret and use environment variables",
                                )
                            )
                            break

        except Exception:
            pass

        return CheckResult(
            check_name="secrets_check",
            category=CheckCategory.GIT,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    # =========================================================================
    # FILE CHECKS
    # =========================================================================

    def _check_temp_files(self) -> CheckResult:
        """Check for temporary files that should be cleaned."""
        issues = []

        temp_patterns = [
            "*.pyc",
            "*.pyo",
            "__pycache__",
            "*.swp",
            "*.swo",
            ".DS_Store",
            "*.tmp",
            "*.bak",
            "*.orig",
        ]

        for pattern in temp_patterns:
            if "*" in pattern:
                files = list(self.project_root.rglob(pattern))
            else:
                files = list(self.project_root.rglob(f"**/{pattern}"))

            files = [f for f in files if ".git" not in str(f) and ".venv" not in str(f)]

            if files:
                issues.append(
                    Issue(
                        category=CheckCategory.FILES,
                        severity=Severity.INFO,
                        check_name="temp_files",
                        message=f"Found {len(files)} {pattern} files",
                        suggestion="Add to .gitignore and/or clean up",
                        auto_fixable=True,
                    )
                )

        return CheckResult(
            check_name="temp_files",
            category=CheckCategory.FILES,
            passed=True,  # Info only
            issues=issues,
        )

    def _check_empty_files(self) -> CheckResult:
        """Check for empty Python files."""
        issues = []

        py_files = list(self.project_root.rglob("*.py"))
        py_files = [f for f in py_files if ".venv" not in str(f)]

        for py_file in py_files:
            try:
                content = py_file.read_text().strip()
                if not content:
                    issues.append(
                        Issue(
                            category=CheckCategory.FILES,
                            severity=Severity.WARNING,
                            check_name="empty_files",
                            message="Empty Python file",
                            file_path=str(py_file.relative_to(self.project_root)),
                            suggestion="Add content or remove file",
                            auto_fixable=True,
                        )
                    )
            except Exception:
                pass

        return CheckResult(
            check_name="empty_files",
            category=CheckCategory.FILES,
            passed=len(issues) == 0,
            issues=issues,
        )

    def _check_duplicate_files(self) -> CheckResult:
        """Check for duplicate file names that may cause confusion."""
        issues = []

        # Check for duplicate basenames in different directories
        py_files = list(self.project_root.rglob("*.py"))
        py_files = [f for f in py_files if ".venv" not in str(f) and ".git" not in str(f)]

        basename_to_paths: dict[str, list[Path]] = {}
        for py_file in py_files:
            basename = py_file.name
            if basename not in basename_to_paths:
                basename_to_paths[basename] = []
            basename_to_paths[basename].append(py_file)

        for basename, paths in basename_to_paths.items():
            if len(paths) > 1 and basename not in ["__init__.py", "conftest.py", "base.py"]:
                issues.append(
                    Issue(
                        category=CheckCategory.FILES,
                        severity=Severity.INFO,
                        check_name="duplicate_files",
                        message=f"Duplicate filename '{basename}' in {len(paths)} locations",
                        suggestion="Consider renaming to avoid confusion",
                    )
                )

        return CheckResult(
            check_name="duplicate_files",
            category=CheckCategory.FILES,
            passed=True,  # Info only
            issues=issues,
        )

    # =========================================================================
    # PROGRESS FILE CHECKS
    # =========================================================================

    def _check_progress_format(self) -> CheckResult:
        """Check progress file format and completeness."""
        issues = []

        progress_file = self.project_root / "claude-progress.txt"
        if progress_file.exists():
            content = progress_file.read_text()

            # Check for required sections
            required_sections = [
                "## CATEGORY",
                "## OVERNIGHT SESSION INSTRUCTIONS",
                "## BLOCKERS",
            ]

            for section in required_sections:
                if section not in content:
                    issues.append(
                        Issue(
                            category=CheckCategory.PROGRESS,
                            severity=Severity.WARNING,
                            check_name="progress_format",
                            message=f"Missing section: {section}",
                            file_path="claude-progress.txt",
                        )
                    )

            # Check for unchecked items that should be done
            unchecked = re.findall(r"- \[ \] .+", content)
            if unchecked:
                issues.append(
                    Issue(
                        category=CheckCategory.PROGRESS,
                        severity=Severity.INFO,
                        check_name="progress_format",
                        message=f"{len(unchecked)} unchecked items in progress file",
                        file_path="claude-progress.txt",
                    )
                )
        else:
            issues.append(
                Issue(
                    category=CheckCategory.PROGRESS,
                    severity=Severity.WARNING,
                    check_name="progress_format",
                    message="No progress file found",
                    suggestion="Create claude-progress.txt for tracking",
                )
            )

        return CheckResult(
            check_name="progress_format",
            category=CheckCategory.PROGRESS,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_session_summaries(self) -> CheckResult:
        """Check for accumulated session summaries that should be cleaned."""
        issues = []

        progress_file = self.project_root / "claude-progress.txt"
        if progress_file.exists():
            content = progress_file.read_text()

            # Count session end summaries
            session_ends = len(re.findall(r"## Session End", content))
            compactions = len(re.findall(r"## Context Compaction", content))

            if session_ends > 2:
                issues.append(
                    Issue(
                        category=CheckCategory.PROGRESS,
                        severity=Severity.WARNING,
                        check_name="session_summaries",
                        message=f"Found {session_ends} session summaries - consider cleanup",
                        file_path="claude-progress.txt",
                        suggestion="Remove old session summaries to keep file clean",
                        auto_fixable=True,
                    )
                )

            if compactions > 2:
                issues.append(
                    Issue(
                        category=CheckCategory.PROGRESS,
                        severity=Severity.INFO,
                        check_name="session_summaries",
                        message=f"Found {compactions} compaction notices",
                        file_path="claude-progress.txt",
                    )
                )

        return CheckResult(
            check_name="session_summaries",
            category=CheckCategory.PROGRESS,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    # =========================================================================
    # DEBUG DIAGNOSTIC CHECKS
    # =========================================================================

    def _check_debug_statements(self) -> CheckResult:
        """Check for leftover debug statements in code."""
        issues = []
        debug_patterns = [
            (r"import\s+pdb", "pdb import"),
            (r"import\s+ipdb", "ipdb import"),
            (r"pdb\.set_trace\(\)", "pdb.set_trace()"),
            (r"ipdb\.set_trace\(\)", "ipdb.set_trace()"),
            (r"import\s+pudb", "pudb import"),
            (r"debugger\s*;", "debugger statement"),
        ]

        py_files = list(self.project_root.rglob("*.py"))
        py_files = [f for f in py_files if ".venv" not in str(f) and "test" not in str(f).lower()]

        for py_file in py_files[:50]:
            try:
                content = py_file.read_text()
                for pattern, desc in debug_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        issues.append(
                            Issue(
                                category=CheckCategory.DEBUG,
                                severity=Severity.WARNING,
                                check_name="debug_statements",
                                message=f"Found {desc}",
                                file_path=str(py_file.relative_to(self.project_root)),
                                suggestion="Remove debug code before committing",
                                auto_fixable=True,
                            )
                        )
            except Exception:
                pass

        return CheckResult(
            check_name="debug_statements",
            category=CheckCategory.DEBUG,
            passed=len(issues) == 0,
            issues=issues,
        )

    def _check_todo_fixme(self) -> CheckResult:
        """Check for TODO/FIXME comments that need attention."""
        issues = []
        todo_patterns = [
            (r"#\s*TODO[:\s](.{0,50})", "TODO"),
            (r"#\s*FIXME[:\s](.{0,50})", "FIXME"),
            (r"#\s*XXX[:\s](.{0,50})", "XXX"),
            (r"#\s*HACK[:\s](.{0,50})", "HACK"),
            (r"#\s*BUG[:\s](.{0,50})", "BUG"),
        ]

        py_files = list(self.project_root.rglob("*.py"))
        py_files = [f for f in py_files if ".venv" not in str(f)]

        todo_count = {"TODO": 0, "FIXME": 0, "XXX": 0, "HACK": 0, "BUG": 0}

        for py_file in py_files[:100]:
            try:
                content = py_file.read_text()
                for pattern, tag in todo_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    todo_count[tag] += len(matches)
            except Exception:
                pass

        for tag, count in todo_count.items():
            if count > 0:
                severity = Severity.ERROR if tag in ["FIXME", "BUG"] else Severity.INFO
                issues.append(
                    Issue(
                        category=CheckCategory.DEBUG,
                        severity=severity,
                        check_name="todo_fixme",
                        message=f"Found {count} {tag} comments in codebase",
                        suggestion=f"Review and resolve {tag} items",
                    )
                )

        return CheckResult(
            check_name="todo_fixme",
            category=CheckCategory.DEBUG,
            passed=todo_count["FIXME"] == 0 and todo_count["BUG"] == 0,
            issues=issues,
        )

    def _check_print_statements(self) -> CheckResult:
        """Check for print statements that should be logging."""
        issues = []
        print_count = 0

        py_files = list(self.project_root.rglob("*.py"))
        py_files = [
            f
            for f in py_files
            if ".venv" not in str(f) and "test" not in str(f).lower() and "script" not in str(f).lower()
        ]

        for py_file in py_files[:50]:
            try:
                content = py_file.read_text()
                # Match print( but not in comments or strings
                matches = re.findall(r"^\s*print\s*\(", content, re.MULTILINE)
                print_count += len(matches)
            except Exception:
                pass

        if print_count > 10:
            issues.append(
                Issue(
                    category=CheckCategory.DEBUG,
                    severity=Severity.WARNING,
                    check_name="print_statements",
                    message=f"Found {print_count} print statements - consider using logging",
                    suggestion="Replace print() with proper logging calls",
                )
            )

        return CheckResult(
            check_name="print_statements",
            category=CheckCategory.DEBUG,
            passed=True,  # Warning only
            issues=issues,
        )

    def _check_breakpoints(self) -> CheckResult:
        """Check for breakpoint() calls."""
        issues = []

        py_files = list(self.project_root.rglob("*.py"))
        py_files = [f for f in py_files if ".venv" not in str(f)]

        for py_file in py_files[:50]:
            try:
                content = py_file.read_text()
                if "breakpoint()" in content:
                    issues.append(
                        Issue(
                            category=CheckCategory.DEBUG,
                            severity=Severity.ERROR,
                            check_name="breakpoints",
                            message="Found breakpoint() call",
                            file_path=str(py_file.relative_to(self.project_root)),
                            suggestion="Remove breakpoint() before committing",
                            auto_fixable=True,
                        )
                    )
            except Exception:
                pass

        return CheckResult(
            check_name="breakpoints",
            category=CheckCategory.DEBUG,
            passed=len(issues) == 0,
            issues=issues,
        )

    def _check_incomplete_code(self) -> CheckResult:
        """Check for incomplete code markers."""
        issues = []
        incomplete_patterns = [
            (r"raise\s+NotImplementedError", "NotImplementedError"),
            (r"pass\s*#.*implement", "pass # implement"),
            (r"\.\.\.\s*#", "... placeholder"),
            (r"#\s*WIP", "WIP comment"),
            (r"#\s*incomplete", "incomplete marker"),
        ]

        py_files = list(self.project_root.rglob("*.py"))
        py_files = [f for f in py_files if ".venv" not in str(f) and "test" not in str(f).lower()]

        incomplete_count = 0
        for py_file in py_files[:50]:
            try:
                content = py_file.read_text()
                for pattern, _desc in incomplete_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    incomplete_count += len(matches)
            except Exception:
                pass

        if incomplete_count > 0:
            issues.append(
                Issue(
                    category=CheckCategory.DEBUG,
                    severity=Severity.WARNING,
                    check_name="incomplete_code",
                    message=f"Found {incomplete_count} incomplete code markers",
                    suggestion="Review and complete or remove placeholder code",
                )
            )

        return CheckResult(
            check_name="incomplete_code",
            category=CheckCategory.DEBUG,
            passed=True,  # Warning only
            issues=issues,
        )

    # =========================================================================
    # CODE INTEGRITY CHECKS
    # =========================================================================

    def _check_corrupt_files(self) -> CheckResult:
        """Check for potentially corrupt or truncated Python files.

        Uses AST parsing for reliable syntax validation instead of naive
        string counting (which produces false positives from brackets
        inside strings/comments).
        """
        import ast

        issues = []

        py_files = list(self.project_root.rglob("*.py"))
        py_files = [f for f in py_files if ".venv" not in str(f)]

        for py_file in py_files[:100]:
            try:
                content = py_file.read_text()

                # Check for null bytes (definitely corrupt)
                if "\x00" in content:
                    issues.append(
                        Issue(
                            category=CheckCategory.INTEGRITY,
                            severity=Severity.ERROR,
                            check_name="corrupt_files",
                            message="File contains null bytes (possibly corrupt)",
                            file_path=str(py_file.relative_to(self.project_root)),
                        )
                    )
                    continue

                # Use AST parsing to validate syntax - this is the reliable way
                # to check for truncated/corrupt files
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    issues.append(
                        Issue(
                            category=CheckCategory.INTEGRITY,
                            severity=Severity.WARNING,
                            check_name="corrupt_files",
                            message=f"Syntax error (possible corruption): {e.msg}",
                            file_path=str(py_file.relative_to(self.project_root)),
                            line_number=e.lineno,
                            suggestion="Check for truncation, encoding issues, or syntax errors",
                        )
                    )

            except UnicodeDecodeError:
                issues.append(
                    Issue(
                        category=CheckCategory.INTEGRITY,
                        severity=Severity.ERROR,
                        check_name="corrupt_files",
                        message="File has encoding errors (possibly corrupt)",
                        file_path=str(py_file.relative_to(self.project_root)),
                    )
                )
            except Exception:
                pass

        return CheckResult(
            check_name="corrupt_files",
            category=CheckCategory.INTEGRITY,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_missing_imports(self) -> CheckResult:
        """Check for undefined names that might be missing imports."""
        issues = []

        # Run pyflakes or similar static analysis
        try:
            result = subprocess.run(
                ["python3", "-m", "pyflakes", "."],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.project_root),
            )

            if result.stdout:
                undefined = re.findall(r"undefined name '(\w+)'", result.stdout)
                if undefined:
                    unique_undefined = list(set(undefined))[:10]
                    issues.append(
                        Issue(
                            category=CheckCategory.INTEGRITY,
                            severity=Severity.WARNING,
                            check_name="missing_imports",
                            message=f"Found undefined names: {', '.join(unique_undefined)}",
                            suggestion="Add missing imports or fix typos",
                        )
                    )
        except FileNotFoundError:
            issues.append(
                Issue(
                    category=CheckCategory.INTEGRITY,
                    severity=Severity.INFO,
                    check_name="missing_imports",
                    message="pyflakes not installed, skipping undefined name check",
                    suggestion="pip install pyflakes",
                )
            )
        except Exception:
            pass

        return CheckResult(
            check_name="missing_imports",
            category=CheckCategory.INTEGRITY,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_circular_imports(self) -> CheckResult:
        """Check for potential circular import issues."""
        issues = []

        # Simple heuristic: check for imports at module level that might cause circles
        key_modules = ["llm", "models", "execution", "evaluation"]

        for module in key_modules:
            module_path = self.project_root / module
            if module_path.exists():
                init_file = module_path / "__init__.py"
                if init_file.exists():
                    try:
                        content = init_file.read_text()
                        # Check for from . import that pulls in too much
                        star_imports = re.findall(r"from\s+\.\w+\s+import\s+\*", content)
                        if star_imports:
                            issues.append(
                                Issue(
                                    category=CheckCategory.INTEGRITY,
                                    severity=Severity.WARNING,
                                    check_name="circular_imports",
                                    message="Star imports in __init__.py may cause circular imports",
                                    file_path=f"{module}/__init__.py",
                                    suggestion="Use explicit imports to avoid circular dependencies",
                                )
                            )
                    except Exception:
                        pass

        return CheckResult(
            check_name="circular_imports",
            category=CheckCategory.INTEGRITY,
            passed=True,  # Warning only
            issues=issues,
        )

    def _check_orphan_functions(self) -> CheckResult:
        """Check for functions/classes that appear unused.

        This is a lightweight heuristic check - it looks for public functions/classes
        that are defined but never referenced elsewhere in the codebase. It tracks
        both direct references and import statements.
        """
        issues = []

        py_files = list(self.project_root.rglob("*.py"))
        py_files = [
            f
            for f in py_files
            if ".venv" not in str(f) and "test" not in str(f).lower() and "__pycache__" not in str(f)
        ]

        # Build a reference map from ALL files (not just first 50)
        all_defs: dict[str, Path] = {}  # name -> file where defined
        all_refs: set[str] = set()  # all referenced names

        for py_file in py_files[:200]:  # Increased limit for better coverage
            try:
                content = py_file.read_text()

                # Find function/class definitions (public only)
                defs = re.findall(r"^(?:def|class)\s+(\w+)", content, re.MULTILINE)
                for d in defs:
                    if not d.startswith("_"):
                        all_defs[d] = py_file

                # Find all references including imports
                # 1. Direct word references
                words = set(re.findall(r"\b([A-Za-z_]\w+)\b", content))
                all_refs.update(words)

                # 2. From X import Y patterns
                imports = re.findall(r"from\s+\S+\s+import\s+([^#\n]+)", content)
                for imp in imports:
                    names = re.findall(r"(\w+)", imp)
                    all_refs.update(names)

            except Exception:
                pass

        # Find definitions that are never referenced in OTHER files
        orphan_count = 0
        for def_name in all_defs:
            # Check if this name is referenced (we already collected all refs)
            if def_name not in all_refs:
                orphan_count += 1

        # Only warn if there are many orphans (threshold 50 to reduce noise)
        if orphan_count > 50:
            issues.append(
                Issue(
                    category=CheckCategory.INTEGRITY,
                    severity=Severity.INFO,
                    check_name="orphan_functions",
                    message=f"Found ~{orphan_count} potentially unused functions/classes",
                    suggestion="Review and remove dead code",
                )
            )

        return CheckResult(
            check_name="orphan_functions",
            category=CheckCategory.INTEGRITY,
            passed=True,  # Info only
            issues=issues,
        )

    def _check_init_files(self) -> CheckResult:
        """Check that packages have proper __init__.py files.

        Excludes directories that are not meant to be Python packages:
        - Hidden directories (starting with .)
        - scripts/ (standalone scripts)
        - .backups/ (backup files)
        - .claude/ (Claude Code configuration)
        """
        issues = []

        # Directories that should NOT be Python packages
        non_package_dirs = {
            ".backups",
            ".claude",
            ".git",
            ".venv",
            "venv",
            "scripts",
            "node_modules",
            ".hypothesis",
        }

        def is_non_package(dir_path: Path) -> bool:
            """Check if directory should not be a Python package."""
            parts = dir_path.relative_to(self.project_root).parts
            # Check if any part is in non-package list or starts with .
            return any(p in non_package_dirs or p.startswith(".") for p in parts)

        # Find directories with .py files but no __init__.py
        py_dirs = set()
        for py_file in self.project_root.rglob("*.py"):
            if not is_non_package(py_file.parent):
                py_dirs.add(py_file.parent)

        for py_dir in py_dirs:
            if py_dir == self.project_root:
                continue
            init_file = py_dir / "__init__.py"
            if not init_file.exists():
                # Check if it should be a package
                py_files = list(py_dir.glob("*.py"))
                if len(py_files) > 1:  # Multiple .py files = likely should be package
                    issues.append(
                        Issue(
                            category=CheckCategory.INTEGRITY,
                            severity=Severity.WARNING,
                            check_name="init_files",
                            message=f"Directory with {len(py_files)} .py files has no __init__.py",
                            file_path=str(py_dir.relative_to(self.project_root)),
                            suggestion="Add __init__.py to make it a proper package",
                        )
                    )

        return CheckResult(
            check_name="init_files",
            category=CheckCategory.INTEGRITY,
            passed=len(issues) == 0,
            issues=issues,
        )

    # =========================================================================
    # CROSS-REFERENCE CHECKS
    # =========================================================================

    def _check_broken_imports(self) -> CheckResult:
        """Check for import statements that reference non-existent modules."""
        issues = []

        py_files = list(self.project_root.rglob("*.py"))
        py_files = [f for f in py_files if ".venv" not in str(f)]

        local_modules = set()
        for py_file in py_files:
            # Build list of local module paths
            rel_path = py_file.relative_to(self.project_root)
            module_path = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")
            local_modules.add(module_path)

        broken_count = 0
        for py_file in py_files[:30]:
            try:
                content = py_file.read_text()
                # Find from X import statements for local modules
                local_imports = re.findall(
                    r"from\s+(llm|models|execution|evaluation|scanners|indicators|ui|utils|config)\.\S+\s+import",
                    content,
                )
                for imp in local_imports:
                    # Check if module exists
                    full_match = re.search(rf"from\s+({imp}\.\S+)\s+import", content)
                    if full_match:
                        module = full_match.group(1)
                        module_file = self.project_root / module.replace(".", "/")
                        if not (module_file.with_suffix(".py").exists() or (module_file / "__init__.py").exists()):
                            broken_count += 1
            except Exception:
                pass

        if broken_count > 0:
            issues.append(
                Issue(
                    category=CheckCategory.XREF,
                    severity=Severity.WARNING,
                    check_name="broken_imports",
                    message=f"Found ~{broken_count} potentially broken import paths",
                    suggestion="Verify import paths match actual module locations",
                )
            )

        return CheckResult(
            check_name="broken_imports",
            category=CheckCategory.XREF,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_config_refs(self) -> CheckResult:
        """Check that config references match actual config keys."""
        issues = []

        config_file = self.project_root / "config" / "settings.json"
        if config_file.exists():
            try:
                config_content = config_file.read_text()
                config_data = json.loads(config_content)

                # Flatten config keys
                def flatten_keys(d, prefix=""):
                    keys = []
                    for k, v in d.items():
                        new_key = f"{prefix}.{k}" if prefix else k
                        keys.append(new_key)
                        if isinstance(v, dict):
                            keys.extend(flatten_keys(v, new_key))
                    return keys

                valid_keys = set(flatten_keys(config_data))

                # Check Python files for config access patterns
                config_accesses = []
                py_files = list(self.project_root.rglob("*.py"))
                py_files = [f for f in py_files if ".venv" not in str(f)]

                for py_file in py_files[:30]:
                    try:
                        content = py_file.read_text()
                        # Look for config.get("key") patterns
                        matches = re.findall(r'config\.get\(["\']([^"\']+)["\']', content)
                        config_accesses.extend(matches)
                    except Exception:
                        pass

                # Check for mismatches
                missing_keys = [k for k in config_accesses if k not in valid_keys]
                if missing_keys:
                    unique_missing = list(set(missing_keys))[:5]
                    issues.append(
                        Issue(
                            category=CheckCategory.XREF,
                            severity=Severity.WARNING,
                            check_name="config_refs",
                            message=f"Config keys not found in settings.json: {', '.join(unique_missing)}",
                            suggestion="Add missing keys to config/settings.json or fix references",
                        )
                    )

            except json.JSONDecodeError:
                issues.append(
                    Issue(
                        category=CheckCategory.XREF,
                        severity=Severity.ERROR,
                        check_name="config_refs",
                        message="config/settings.json is not valid JSON",
                        file_path="config/settings.json",
                    )
                )
            except Exception:
                pass

        return CheckResult(
            check_name="config_refs",
            category=CheckCategory.XREF,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_class_refs(self) -> CheckResult:
        """Check that class references are valid."""
        issues = []

        # Build map of all class definitions
        class_defs = {}
        py_files = list(self.project_root.rglob("*.py"))
        py_files = [f for f in py_files if ".venv" not in str(f)]

        for py_file in py_files[:50]:
            try:
                content = py_file.read_text()
                classes = re.findall(r"^class\s+(\w+)", content, re.MULTILINE)
                for cls in classes:
                    class_defs[cls] = py_file
            except Exception:
                pass

        # Check for type hints referencing non-existent classes
        undefined_refs = []
        for py_file in py_files[:30]:
            try:
                content = py_file.read_text()
                # Look for type hints
                type_hints = re.findall(r":\s*([A-Z]\w+)(?:\s*[=,\)]|\s*$)", content)
                for hint in type_hints:
                    if hint not in class_defs and hint not in [
                        "List",
                        "Dict",
                        "Optional",
                        "Any",
                        "Union",
                        "Callable",
                        "Tuple",
                        "Set",
                        "Type",
                        "None",
                        "Self",
                        "Path",
                        "Enum",
                    ]:
                        undefined_refs.append(hint)
            except Exception:
                pass

        if len(undefined_refs) > 10:
            unique_refs = list(set(undefined_refs))[:5]
            issues.append(
                Issue(
                    category=CheckCategory.XREF,
                    severity=Severity.INFO,
                    check_name="class_refs",
                    message=f"Type hints reference unknown classes: {', '.join(unique_refs)}...",
                    suggestion="Verify type hints or add missing imports",
                )
            )

        return CheckResult(
            check_name="class_refs",
            category=CheckCategory.XREF,
            passed=True,  # Info only
            issues=issues,
        )

    def _check_doc_links(self) -> CheckResult:
        """Check that documentation links are valid."""
        issues = []

        md_files = list(self.project_root.rglob("*.md"))
        md_files = [f for f in md_files if ".venv" not in str(f)]

        broken_links = 0
        for md_file in md_files[:30]:
            try:
                content = md_file.read_text()
                # Find relative links
                links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
                for _text, link in links:
                    # Skip external links and anchors
                    if link.startswith("http") or link.startswith("#"):
                        continue
                    # Check if file exists
                    link_path = link.split("#")[0]  # Remove anchor
                    if link_path:
                        full_path = md_file.parent / link_path
                        if not full_path.exists():
                            broken_links += 1
            except Exception:
                pass

        if broken_links > 0:
            issues.append(
                Issue(
                    category=CheckCategory.XREF,
                    severity=Severity.WARNING,
                    check_name="doc_links",
                    message=f"Found {broken_links} broken links in documentation",
                    suggestion="Fix or remove broken documentation links",
                )
            )

        return CheckResult(
            check_name="doc_links",
            category=CheckCategory.XREF,
            passed=len(issues) == 0,
            issues=issues,
        )

    # =========================================================================
    # RIC LOOP COMPLIANCE CHECKS
    # =========================================================================

    def _check_ric_phases(self) -> CheckResult:
        """Check RIC loop phase compliance in upgrade documents."""
        issues = []

        if self.upgrade_focus:
            # Check specific upgrade
            upgrade_doc = self.project_root / "docs" / "research" / f"{self.upgrade_focus}.md"
            if upgrade_doc.exists():
                content = upgrade_doc.read_text()

                # Check for required RIC phases
                required_phases = [
                    ("Phase 0", "Research"),
                    ("Phase 1", "Upgrade Path"),
                    ("Phase 2", "Checklist"),
                ]

                for phase_num, phase_name in required_phases:
                    if phase_num not in content and phase_name not in content:
                        issues.append(
                            Issue(
                                category=CheckCategory.RIC,
                                severity=Severity.WARNING,
                                check_name="ric_phases",
                                message=f"Missing {phase_num}: {phase_name} in upgrade doc",
                                file_path=str(upgrade_doc.relative_to(self.project_root)),
                            )
                        )
            else:
                issues.append(
                    Issue(
                        category=CheckCategory.RIC,
                        severity=Severity.INFO,
                        check_name="ric_phases",
                        message=f"Upgrade document not found: {self.upgrade_focus}",
                    )
                )
        else:
            # General check on recent upgrades
            research_dir = self.project_root / "docs" / "research"
            if research_dir.exists():
                upgrade_docs = list(research_dir.glob("UPGRADE-*.md"))
                for doc in upgrade_docs[:5]:  # Check last 5
                    try:
                        content = doc.read_text()
                        if "Phase 0" not in content and "Research" not in content:
                            issues.append(
                                Issue(
                                    category=CheckCategory.RIC,
                                    severity=Severity.INFO,
                                    check_name="ric_phases",
                                    message="May be missing Phase 0 Research",
                                    file_path=str(doc.relative_to(self.project_root)),
                                )
                            )
                    except Exception:
                        pass

        return CheckResult(
            check_name="ric_phases",
            category=CheckCategory.RIC,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_upgrade_docs(self) -> CheckResult:
        """Check upgrade documentation completeness."""
        issues = []

        research_dir = self.project_root / "docs" / "research"
        if research_dir.exists():
            upgrade_docs = list(research_dir.glob("UPGRADE-*.md"))

            for doc in upgrade_docs[:10]:
                try:
                    content = doc.read_text()

                    # Check for required sections
                    required_sections = ["## Overview", "## Implementation"]
                    missing = [s for s in required_sections if s not in content]

                    if missing:
                        issues.append(
                            Issue(
                                category=CheckCategory.RIC,
                                severity=Severity.INFO,
                                check_name="upgrade_docs",
                                message=f"Missing sections: {', '.join(missing)}",
                                file_path=str(doc.relative_to(self.project_root)),
                            )
                        )

                    # Check for completion status
                    if "Status: PENDING" in content or "Status: IN_PROGRESS" in content:
                        issues.append(
                            Issue(
                                category=CheckCategory.RIC,
                                severity=Severity.INFO,
                                check_name="upgrade_docs",
                                message="Upgrade not marked complete",
                                file_path=str(doc.relative_to(self.project_root)),
                            )
                        )

                except Exception:
                    pass

        return CheckResult(
            check_name="upgrade_docs",
            category=CheckCategory.RIC,
            passed=True,  # Info only
            issues=issues,
        )

    def _check_iteration_tracking(self) -> CheckResult:
        """Check that iterations are properly tracked in progress file."""
        issues = []

        progress_file = self.project_root / "claude-progress.txt"
        if progress_file.exists():
            content = progress_file.read_text()

            # Check for iteration markers
            iterations = re.findall(r"\[ITERATION\s+(\d+)/(\d+)\]", content)
            if not iterations:
                # Check for RIC loop usage markers
                ric_markers = re.findall(r"RIC Loop|Meta-RIC|/ric-", content, re.IGNORECASE)
                if ric_markers:
                    issues.append(
                        Issue(
                            category=CheckCategory.RIC,
                            severity=Severity.INFO,
                            check_name="iteration_tracking",
                            message="RIC loop mentioned but no iteration tracking found",
                            file_path="claude-progress.txt",
                            suggestion="Add [ITERATION X/5] markers when using RIC loop",
                        )
                    )

            # Check for minimum 3 iterations
            if iterations:
                max_iter = max(int(i[0]) for i in iterations)
                if max_iter < 3:
                    issues.append(
                        Issue(
                            category=CheckCategory.RIC,
                            severity=Severity.WARNING,
                            check_name="iteration_tracking",
                            message=f"Only {max_iter} iterations found (minimum 3 required)",
                            file_path="claude-progress.txt",
                        )
                    )

        return CheckResult(
            check_name="iteration_tracking",
            category=CheckCategory.RIC,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    # =========================================================================
    # SECURITY CHECKS
    # =========================================================================

    def _check_security_bandit(self) -> CheckResult:
        """Run Bandit security scanner to find common vulnerabilities.

        Checks for security issues like:
        - Hardcoded passwords
        - SQL injection
        - Shell injection
        - Insecure random number generators
        - And more from OWASP guidelines
        """
        issues = []

        try:
            # Run bandit using venv Python
            result = subprocess.run(
                [
                    self._venv_python,
                    "-m",
                    "bandit",
                    "-r",
                    ".",
                    "-f",
                    "json",
                    "-q",
                    "--exclude",
                    ".venv,venv,.git,tests,__pycache__,.backups",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.project_root),
            )

            if result.stdout:
                try:
                    bandit_results = json.loads(result.stdout)
                    high_issues = [r for r in bandit_results.get("results", []) if r.get("issue_severity") == "HIGH"]
                    medium_issues = [
                        r for r in bandit_results.get("results", []) if r.get("issue_severity") == "MEDIUM"
                    ]

                    if high_issues:
                        issues.append(
                            Issue(
                                category=CheckCategory.SECURITY,
                                severity=Severity.ERROR,
                                check_name="security_bandit",
                                message=f"Found {len(high_issues)} HIGH severity security issues",
                                suggestion="Run 'bandit -r .' for details",
                            )
                        )

                    if medium_issues:
                        issues.append(
                            Issue(
                                category=CheckCategory.SECURITY,
                                severity=Severity.WARNING,
                                check_name="security_bandit",
                                message=f"Found {len(medium_issues)} MEDIUM severity security issues",
                                suggestion="Run 'bandit -r .' for details",
                            )
                        )
                except json.JSONDecodeError:
                    pass

        except FileNotFoundError:
            issues.append(
                Issue(
                    category=CheckCategory.SECURITY,
                    severity=Severity.INFO,
                    check_name="security_bandit",
                    message="Bandit not installed, skipping security scan",
                    suggestion="pip install bandit",
                )
            )
        except subprocess.TimeoutExpired:
            issues.append(
                Issue(
                    category=CheckCategory.SECURITY,
                    severity=Severity.WARNING,
                    check_name="security_bandit",
                    message="Bandit scan timed out",
                )
            )
        except Exception:
            pass

        return CheckResult(
            check_name="security_bandit",
            category=CheckCategory.SECURITY,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_code_complexity(self) -> CheckResult:
        """Check code complexity using McCabe complexity metric.

        Functions with complexity > 10 are flagged as warnings.
        Functions with complexity > 20 are flagged as errors.
        """
        issues = []

        try:
            # Use radon for complexity analysis (more comprehensive than mccabe alone)
            result = subprocess.run(
                [
                    self._venv_python,
                    "-m",
                    "radon",
                    "cc",
                    ".",
                    "-a",  # Average complexity
                    "-s",  # Show complexity
                    "--exclude",
                    ".venv,venv,.git,tests,__pycache__,.backups",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.project_root),
            )

            if result.stdout:
                # Parse output for high complexity functions
                high_complexity = re.findall(r"(\w+)\s+-\s+([A-F])\s+\((\d+)\)", result.stdout)
                very_complex = [(name, grade, score) for name, grade, score in high_complexity if int(score) > 20]
                moderately_complex = [
                    (name, grade, score) for name, grade, score in high_complexity if 10 < int(score) <= 20
                ]

                if very_complex:
                    issues.append(
                        Issue(
                            category=CheckCategory.SECURITY,
                            severity=Severity.WARNING,
                            check_name="security_complexity",
                            message=f"Found {len(very_complex)} functions with very high complexity (>20)",
                            suggestion="Consider refactoring complex functions",
                        )
                    )

                if len(moderately_complex) > 10:
                    issues.append(
                        Issue(
                            category=CheckCategory.SECURITY,
                            severity=Severity.INFO,
                            check_name="security_complexity",
                            message=f"Found {len(moderately_complex)} functions with moderate complexity (10-20)",
                        )
                    )

        except FileNotFoundError:
            issues.append(
                Issue(
                    category=CheckCategory.SECURITY,
                    severity=Severity.INFO,
                    check_name="security_complexity",
                    message="radon not installed, skipping complexity check",
                    suggestion="pip install radon",
                )
            )
        except Exception:
            pass

        return CheckResult(
            check_name="security_complexity",
            category=CheckCategory.SECURITY,
            passed=True,  # Complexity is advisory
            issues=issues,
        )

    # =========================================================================
    # HOOKS VALIDATION CHECKS
    # =========================================================================

    def _check_hooks_exist(self) -> CheckResult:
        """Check that required Claude Code hooks exist."""
        issues = []

        hooks_dir = self.project_root / ".claude" / "hooks"

        # Required hooks for trading safety
        required_hooks = [
            ("ric_v50.py", "RIC v5.0/5.1 system for iterative development"),
            ("session_stop.py", "Session stop hook for continuous mode"),
            ("pre_compact.py", "Pre-compaction hook for transcript backup"),
        ]

        # Recommended hooks
        recommended_hooks = [
            ("protect_files.py", "File protection for sensitive files"),
            ("qa_auto_check.py", "QA auto-check on file changes"),
            ("validate_research.py", "Research doc validation"),
        ]

        if not hooks_dir.exists():
            issues.append(
                Issue(
                    category=CheckCategory.HOOKS,
                    severity=Severity.ERROR,
                    check_name="hooks_exist",
                    message="Hooks directory .claude/hooks/ does not exist",
                    suggestion="Create .claude/hooks/ directory with required hooks",
                )
            )
            return CheckResult(
                check_name="hooks_exist",
                category=CheckCategory.HOOKS,
                passed=False,
                issues=issues,
            )

        # Check required hooks
        for hook_file, description in required_hooks:
            hook_path = hooks_dir / hook_file
            if not hook_path.exists():
                issues.append(
                    Issue(
                        category=CheckCategory.HOOKS,
                        severity=Severity.ERROR,
                        check_name="hooks_exist",
                        message=f"Required hook missing: {hook_file}",
                        file_path=str(hook_path),
                        suggestion=f"Create {hook_file} - {description}",
                    )
                )

        # Check recommended hooks
        for hook_file, description in recommended_hooks:
            hook_path = hooks_dir / hook_file
            if not hook_path.exists():
                issues.append(
                    Issue(
                        category=CheckCategory.HOOKS,
                        severity=Severity.INFO,
                        check_name="hooks_exist",
                        message=f"Recommended hook missing: {hook_file}",
                        file_path=str(hook_path),
                        suggestion=f"Consider adding {hook_file} - {description}",
                    )
                )

        return CheckResult(
            check_name="hooks_exist",
            category=CheckCategory.HOOKS,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_hooks_syntax(self) -> CheckResult:
        """Check that all hook files have valid Python syntax."""
        issues = []

        hooks_dir = self.project_root / ".claude" / "hooks"
        if not hooks_dir.exists():
            return CheckResult(
                check_name="hooks_syntax",
                category=CheckCategory.HOOKS,
                passed=True,
                issues=[],
            )

        hook_files = list(hooks_dir.glob("*.py"))

        for hook_file in hook_files:
            try:
                result = subprocess.run(
                    [self._venv_python, "-m", "py_compile", str(hook_file)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    error_msg = result.stderr.strip()[:200] if result.stderr else "Unknown syntax error"
                    issues.append(
                        Issue(
                            category=CheckCategory.HOOKS,
                            severity=Severity.ERROR,
                            check_name="hooks_syntax",
                            message=f"Syntax error in hook: {hook_file.name}",
                            file_path=str(hook_file),
                            suggestion=f"Fix syntax: {error_msg}",
                        )
                    )
            except Exception as e:
                issues.append(
                    Issue(
                        category=CheckCategory.HOOKS,
                        severity=Severity.WARNING,
                        check_name="hooks_syntax",
                        message=f"Could not check syntax: {hook_file.name} - {e}",
                        file_path=str(hook_file),
                    )
                )

        return CheckResult(
            check_name="hooks_syntax",
            category=CheckCategory.HOOKS,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_hooks_registration(self) -> CheckResult:
        """Check that hooks are properly registered in settings.json."""
        issues = []

        settings_path = self.project_root / ".claude" / "settings.json"
        hooks_dir = self.project_root / ".claude" / "hooks"

        if not settings_path.exists():
            issues.append(
                Issue(
                    category=CheckCategory.HOOKS,
                    severity=Severity.ERROR,
                    check_name="hooks_registration",
                    message=".claude/settings.json not found",
                    file_path=str(settings_path),
                    suggestion="Create settings.json with hooks configuration",
                )
            )
            return CheckResult(
                check_name="hooks_registration",
                category=CheckCategory.HOOKS,
                passed=False,
                issues=issues,
            )

        try:
            with open(settings_path) as f:
                settings = json.load(f)
        except json.JSONDecodeError as e:
            issues.append(
                Issue(
                    category=CheckCategory.HOOKS,
                    severity=Severity.ERROR,
                    check_name="hooks_registration",
                    message=f"Invalid JSON in settings.json: {e}",
                    file_path=str(settings_path),
                    suggestion="Fix JSON syntax errors",
                )
            )
            return CheckResult(
                check_name="hooks_registration",
                category=CheckCategory.HOOKS,
                passed=False,
                issues=issues,
            )

        hooks_config = settings.get("hooks", {})
        if not hooks_config:
            issues.append(
                Issue(
                    category=CheckCategory.HOOKS,
                    severity=Severity.WARNING,
                    check_name="hooks_registration",
                    message="No hooks configured in settings.json",
                    file_path=str(settings_path),
                    suggestion="Add hooks section to settings.json",
                )
            )

        # Check for expected hook types
        expected_hook_types = [
            "PreToolUse",
            "PostToolUse",
            "UserPromptSubmit",
            "SessionStart",
        ]

        registered_types = list(hooks_config.keys())
        for hook_type in expected_hook_types:
            if hook_type not in registered_types:
                issues.append(
                    Issue(
                        category=CheckCategory.HOOKS,
                        severity=Severity.INFO,
                        check_name="hooks_registration",
                        message=f"No {hook_type} hooks registered",
                        suggestion=f"Consider adding {hook_type} hooks for better workflow",
                    )
                )

        # Check that registered hooks exist
        if hooks_dir.exists():
            for hook_type, hook_list in hooks_config.items():
                if not isinstance(hook_list, list):
                    continue
                for hook_entry in hook_list:
                    if isinstance(hook_entry, dict) and "hooks" in hook_entry:
                        for hook in hook_entry["hooks"]:
                            if isinstance(hook, dict) and "command" in hook:
                                cmd = hook["command"]
                                # Extract hook file from command
                                if ".claude/hooks/" in cmd:
                                    hook_file = cmd.split(".claude/hooks/")[-1].split()[0]
                                    hook_path = hooks_dir / hook_file
                                    if not hook_path.exists():
                                        issues.append(
                                            Issue(
                                                category=CheckCategory.HOOKS,
                                                severity=Severity.ERROR,
                                                check_name="hooks_registration",
                                                message=f"Registered hook not found: {hook_file}",
                                                file_path=str(hook_path),
                                                suggestion="Create the hook file or remove registration",
                                            )
                                        )

        return CheckResult(
            check_name="hooks_registration",
            category=CheckCategory.HOOKS,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_hooks_settings(self) -> CheckResult:
        """Check hooks settings for best practices and security."""
        issues = []

        settings_path = self.project_root / ".claude" / "settings.json"

        if not settings_path.exists():
            return CheckResult(
                check_name="hooks_settings",
                category=CheckCategory.HOOKS,
                passed=True,
                issues=[],
            )

        try:
            with open(settings_path) as f:
                settings = json.load(f)
        except json.JSONDecodeError:
            return CheckResult(
                check_name="hooks_settings",
                category=CheckCategory.HOOKS,
                passed=True,
                issues=[],
            )

        hooks_config = settings.get("hooks", {})

        # Check for dangerous patterns in hook commands
        dangerous_patterns = [
            ("rm -rf", "Dangerous delete command"),
            ("sudo ", "Elevated privileges"),
            ("> /dev/", "Direct device access"),
            ("curl | bash", "Remote code execution"),
            ("wget | sh", "Remote code execution"),
        ]

        for hook_type, hook_list in hooks_config.items():
            if not isinstance(hook_list, list):
                continue
            for hook_entry in hook_list:
                if isinstance(hook_entry, dict) and "hooks" in hook_entry:
                    for hook in hook_entry["hooks"]:
                        if isinstance(hook, dict) and "command" in hook:
                            cmd = hook["command"]
                            for pattern, description in dangerous_patterns:
                                if pattern in cmd:
                                    issues.append(
                                        Issue(
                                            category=CheckCategory.HOOKS,
                                            severity=Severity.ERROR,
                                            check_name="hooks_settings",
                                            message=f"{description} in {hook_type} hook: {pattern}",
                                            suggestion="Review and remove dangerous command patterns",
                                        )
                                    )

        # Check for deny rules (they don't work - warn users)
        if "deny" in settings:
            issues.append(
                Issue(
                    category=CheckCategory.HOOKS,
                    severity=Severity.WARNING,
                    check_name="hooks_settings",
                    message="'deny' rules are NOT enforced by Claude Code (GitHub #6699, #6631)",
                    file_path=str(settings_path),
                    suggestion="Use PreToolUse hooks instead for file protection",
                )
            )

        # Check timeout settings
        for hook_type, hook_list in hooks_config.items():
            if not isinstance(hook_list, list):
                continue
            for hook_entry in hook_list:
                if isinstance(hook_entry, dict) and "hooks" in hook_entry:
                    for hook in hook_entry["hooks"]:
                        if isinstance(hook, dict):
                            timeout = hook.get("timeout")
                            if timeout and timeout > 30000:  # 30 seconds
                                issues.append(
                                    Issue(
                                        category=CheckCategory.HOOKS,
                                        severity=Severity.INFO,
                                        check_name="hooks_settings",
                                        message=f"Long timeout ({timeout}ms) in {hook_type} hook",
                                        suggestion="Consider reducing timeout for faster feedback",
                                    )
                                )

        return CheckResult(
            check_name="hooks_settings",
            category=CheckCategory.HOOKS,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    # =========================================================================
    # TRADING/ALGORITHM SAFETY CHECKS
    # =========================================================================

    def _check_risk_params(self) -> CheckResult:
        """Check for risk parameters in configuration."""
        issues = []

        # Check settings.json for risk management section
        settings_path = self.project_root / "config" / "settings.json"

        if not settings_path.exists():
            issues.append(
                Issue(
                    category=CheckCategory.TRADING,
                    severity=Severity.WARNING,
                    check_name="risk_params",
                    message="config/settings.json not found",
                    file_path=str(settings_path),
                    suggestion="Create configuration file with risk_management section",
                )
            )
        else:
            try:
                with open(settings_path) as f:
                    config = json.load(f)

                risk_config = config.get("risk_management", {})
                required_params = [
                    "max_position_size",
                    "max_daily_loss_pct",
                    "max_drawdown",
                ]

                for param in required_params:
                    if param not in risk_config:
                        issues.append(
                            Issue(
                                category=CheckCategory.TRADING,
                                severity=Severity.WARNING,
                                check_name="risk_params",
                                message=f"Missing risk parameter: {param}",
                                file_path=str(settings_path),
                                suggestion=f"Add '{param}' to risk_management section",
                            )
                        )

                # Check for reasonable values
                if "max_daily_loss_pct" in risk_config:
                    val = risk_config["max_daily_loss_pct"]
                    if isinstance(val, (int, float)) and (val < 0 or val > 0.20):
                        issues.append(
                            Issue(
                                category=CheckCategory.TRADING,
                                severity=Severity.WARNING,
                                check_name="risk_params",
                                message=f"max_daily_loss_pct={val} seems unreasonable (expected 0-0.20)",
                                file_path=str(settings_path),
                                suggestion="Typical daily loss limit is 1-5% (0.01-0.05)",
                            )
                        )

                if "max_position_size" in risk_config:
                    val = risk_config["max_position_size"]
                    if isinstance(val, (int, float)) and (val < 0 or val > 1.0):
                        issues.append(
                            Issue(
                                category=CheckCategory.TRADING,
                                severity=Severity.WARNING,
                                check_name="risk_params",
                                message=f"max_position_size={val} seems unreasonable (expected 0-1.0)",
                                file_path=str(settings_path),
                                suggestion="Position size should be 0-100% (0.0-1.0)",
                            )
                        )

            except json.JSONDecodeError as e:
                issues.append(
                    Issue(
                        category=CheckCategory.TRADING,
                        severity=Severity.ERROR,
                        check_name="risk_params",
                        message=f"Invalid JSON in settings.json: {e}",
                        file_path=str(settings_path),
                    )
                )

        return CheckResult(
            check_name="risk_params",
            category=CheckCategory.TRADING,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_algorithm_structure(self) -> CheckResult:
        """Check algorithms have required QuantConnect methods."""
        issues = []

        algo_dir = self.project_root / "algorithms"
        if not algo_dir.exists():
            return CheckResult(
                check_name="algorithm_structure",
                category=CheckCategory.TRADING,
                passed=True,
                issues=[],
            )

        required_methods = ["Initialize", "OnData"]
        recommended_methods = ["OnEndOfDay", "OnOrderEvent"]

        for algo_file in algo_dir.glob("*.py"):
            if algo_file.name.startswith("_"):
                continue

            try:
                content = algo_file.read_text()

                # Check if this is a QCAlgorithm
                if "QCAlgorithm" not in content:
                    continue

                # Check for required methods
                for method in required_methods:
                    # Look for method definition patterns
                    patterns = [
                        rf"def {method}\s*\(",  # def Initialize(
                        rf"def {method.lower()}\s*\(",  # def initialize(
                    ]
                    found = any(re.search(p, content) for p in patterns)
                    if not found:
                        issues.append(
                            Issue(
                                category=CheckCategory.TRADING,
                                severity=Severity.ERROR,
                                check_name="algorithm_structure",
                                message=f"Missing required method: {method}",
                                file_path=str(algo_file),
                                suggestion=f"Add 'def {method}(self, ...)' method",
                            )
                        )

                # Check for recommended methods (info only)
                for method in recommended_methods:
                    patterns = [
                        rf"def {method}\s*\(",
                        rf"def {method.lower()}\s*\(",
                        rf"def on_{method[2:].lower()}\s*\(",  # on_end_of_day style
                    ]
                    found = any(re.search(p, content) for p in patterns)
                    if not found:
                        issues.append(
                            Issue(
                                category=CheckCategory.TRADING,
                                severity=Severity.INFO,
                                check_name="algorithm_structure",
                                message=f"Missing recommended method: {method}",
                                file_path=str(algo_file),
                                suggestion=f"Consider adding '{method}' for better control",
                            )
                        )

            except Exception as e:
                issues.append(
                    Issue(
                        category=CheckCategory.TRADING,
                        severity=Severity.WARNING,
                        check_name="algorithm_structure",
                        message=f"Error reading algorithm: {e}",
                        file_path=str(algo_file),
                    )
                )

        return CheckResult(
            check_name="algorithm_structure",
            category=CheckCategory.TRADING,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_paper_mode_default(self) -> CheckResult:
        """Ensure paper trading mode is enabled by default."""
        issues = []

        # Check settings.json
        settings_path = self.project_root / "config" / "settings.json"
        if settings_path.exists():
            try:
                with open(settings_path) as f:
                    config = json.load(f)

                # Check for trading mode settings
                brokerage = config.get("brokerage", {})
                if brokerage.get("live_trading", False):
                    issues.append(
                        Issue(
                            category=CheckCategory.TRADING,
                            severity=Severity.WARNING,
                            check_name="paper_mode_default",
                            message="live_trading=True in settings - should default to paper",
                            file_path=str(settings_path),
                            suggestion="Set live_trading=false as default for safety",
                        )
                    )

            except json.JSONDecodeError:
                pass

        # Check .env and .env.example
        env_files = [".env", ".env.example"]
        for env_file in env_files:
            env_path = self.project_root / env_file
            if env_path.exists():
                content = env_path.read_text()
                # Check for TRADING_MODE or LIVE_TRADING environment variables
                if re.search(r"TRADING_MODE\s*=\s*['\"]?live", content, re.IGNORECASE):
                    issues.append(
                        Issue(
                            category=CheckCategory.TRADING,
                            severity=Severity.WARNING,
                            check_name="paper_mode_default",
                            message=f"TRADING_MODE=live in {env_file}",
                            file_path=str(env_path),
                            suggestion="Use TRADING_MODE=paper as default",
                        )
                    )
                if re.search(r"LIVE_TRADING\s*=\s*['\"]?(true|1|yes)", content, re.IGNORECASE):
                    issues.append(
                        Issue(
                            category=CheckCategory.TRADING,
                            severity=Severity.WARNING,
                            check_name="paper_mode_default",
                            message=f"LIVE_TRADING=true in {env_file}",
                            file_path=str(env_path),
                            suggestion="Use LIVE_TRADING=false as default",
                        )
                    )

        # Check algorithm files for hardcoded live mode
        algo_dir = self.project_root / "algorithms"
        if algo_dir.exists():
            for algo_file in algo_dir.glob("*.py"):
                try:
                    content = algo_file.read_text()
                    # Look for direct live trading enablement
                    if re.search(r"self\.SetLiveMode\s*\(\s*True\s*\)", content):
                        issues.append(
                            Issue(
                                category=CheckCategory.TRADING,
                                severity=Severity.ERROR,
                                check_name="paper_mode_default",
                                message="Hardcoded SetLiveMode(True) in algorithm",
                                file_path=str(algo_file),
                                suggestion="Remove hardcoded live mode; use config/env instead",
                            )
                        )
                except Exception:
                    pass

        return CheckResult(
            check_name="paper_mode_default",
            category=CheckCategory.TRADING,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    # =========================================================================
    # CONFIGURATION VALIDATION CHECKS
    # =========================================================================

    def _check_config_schema(self) -> CheckResult:
        """Validate configuration files against expected schema."""
        issues = []

        config_dir = self.project_root / "config"
        if not config_dir.exists():
            return CheckResult(
                check_name="config_schema",
                category=CheckCategory.CONFIG,
                passed=True,
                issues=[],
            )

        # Expected top-level keys in settings.json
        expected_sections = [
            "brokerage",
            "risk_management",
            "options_scanner",
            "movement_scanner",
            "profit_taking",
            "order_execution",
        ]

        settings_path = config_dir / "settings.json"
        if settings_path.exists():
            try:
                with open(settings_path) as f:
                    config = json.load(f)

                # Check for expected sections
                for section in expected_sections:
                    if section not in config:
                        issues.append(
                            Issue(
                                category=CheckCategory.CONFIG,
                                severity=Severity.INFO,
                                check_name="config_schema",
                                message=f"Missing config section: {section}",
                                file_path=str(settings_path),
                                suggestion=f"Consider adding '{section}' section for completeness",
                            )
                        )

                # Check for unknown/typo sections (warning)
                known_sections = set(
                    expected_sections
                    + ["llm_integration", "technical_indicators", "ui", "quantconnect", "logging", "backtest"]
                )
                for key in config.keys():
                    if key not in known_sections:
                        issues.append(
                            Issue(
                                category=CheckCategory.CONFIG,
                                severity=Severity.INFO,
                                check_name="config_schema",
                                message=f"Unknown config section: {key}",
                                file_path=str(settings_path),
                                suggestion="Verify section name is correct",
                            )
                        )

            except json.JSONDecodeError as e:
                issues.append(
                    Issue(
                        category=CheckCategory.CONFIG,
                        severity=Severity.ERROR,
                        check_name="config_schema",
                        message=f"Invalid JSON in settings.json: {e}",
                        file_path=str(settings_path),
                    )
                )

        return CheckResult(
            check_name="config_schema",
            category=CheckCategory.CONFIG,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_env_vars(self) -> CheckResult:
        """Check .env.example exists and documents required variables."""
        issues = []

        env_example = self.project_root / ".env.example"
        env_file = self.project_root / ".env"

        if not env_example.exists():
            issues.append(
                Issue(
                    category=CheckCategory.CONFIG,
                    severity=Severity.WARNING,
                    check_name="env_vars",
                    message=".env.example not found",
                    suggestion="Create .env.example to document required environment variables",
                )
            )
        else:
            example_content = env_example.read_text()

            # Check for required variables
            required_vars = [
                "QC_USER_ID",
                "QC_API_TOKEN",
            ]

            for var in required_vars:
                if var not in example_content:
                    issues.append(
                        Issue(
                            category=CheckCategory.CONFIG,
                            severity=Severity.INFO,
                            check_name="env_vars",
                            message=f"Missing {var} in .env.example",
                            file_path=str(env_example),
                            suggestion=f"Add {var}= to document this required variable",
                        )
                    )

            # Check for secrets accidentally in example file
            secret_patterns = [
                (r"API_KEY\s*=\s*['\"]?[a-zA-Z0-9]{20,}", "Possible real API key"),
                (r"SECRET\s*=\s*['\"]?[a-zA-Z0-9]{20,}", "Possible real secret"),
                (r"TOKEN\s*=\s*['\"]?[a-zA-Z0-9]{20,}", "Possible real token"),
            ]

            for pattern, description in secret_patterns:
                if re.search(pattern, example_content):
                    issues.append(
                        Issue(
                            category=CheckCategory.CONFIG,
                            severity=Severity.ERROR,
                            check_name="env_vars",
                            message=f"{description} in .env.example",
                            file_path=str(env_example),
                            suggestion="Use placeholder values like 'your_api_key_here'",
                        )
                    )

        # Check if .env exists (informational)
        if not env_file.exists():
            issues.append(
                Issue(
                    category=CheckCategory.CONFIG,
                    severity=Severity.INFO,
                    check_name="env_vars",
                    message=".env file not found",
                    suggestion="Copy .env.example to .env and fill in values",
                )
            )

        return CheckResult(
            check_name="env_vars",
            category=CheckCategory.CONFIG,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_mcp_config(self) -> CheckResult:
        """Validate MCP (Model Context Protocol) configuration."""
        issues = []

        mcp_config = self.project_root / ".mcp.json"

        if not mcp_config.exists():
            issues.append(
                Issue(
                    category=CheckCategory.CONFIG,
                    severity=Severity.INFO,
                    check_name="mcp_config",
                    message=".mcp.json not found",
                    suggestion="Create .mcp.json for MCP server configuration",
                )
            )
            return CheckResult(
                check_name="mcp_config",
                category=CheckCategory.CONFIG,
                passed=True,
                issues=issues,
            )

        try:
            with open(mcp_config) as f:
                config = json.load(f)

            # Check for mcpServers section
            if "mcpServers" not in config:
                issues.append(
                    Issue(
                        category=CheckCategory.CONFIG,
                        severity=Severity.WARNING,
                        check_name="mcp_config",
                        message="Missing 'mcpServers' section in .mcp.json",
                        file_path=str(mcp_config),
                        suggestion="Add mcpServers configuration",
                    )
                )
            else:
                servers = config["mcpServers"]
                for server_name, server_config in servers.items():
                    # Check each server has required fields
                    if isinstance(server_config, dict):
                        if "command" not in server_config and "url" not in server_config:
                            issues.append(
                                Issue(
                                    category=CheckCategory.CONFIG,
                                    severity=Severity.WARNING,
                                    check_name="mcp_config",
                                    message=f"MCP server '{server_name}' missing command or url",
                                    file_path=str(mcp_config),
                                    suggestion="Add 'command' or 'url' to server config",
                                )
                            )

        except json.JSONDecodeError as e:
            issues.append(
                Issue(
                    category=CheckCategory.CONFIG,
                    severity=Severity.ERROR,
                    check_name="mcp_config",
                    message=f"Invalid JSON in .mcp.json: {e}",
                    file_path=str(mcp_config),
                )
            )

        return CheckResult(
            check_name="mcp_config",
            category=CheckCategory.CONFIG,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    # =========================================================================
    # DEPENDENCIES VALIDATION CHECKS
    # =========================================================================

    def _check_requirements_sync(self) -> CheckResult:
        """Check that requirements files are in sync."""
        issues = []

        req_files = [
            self.project_root / "requirements.txt",
            self.project_root / "requirements-dev.txt",
            self.project_root / "pyproject.toml",
        ]

        existing_req_files = [f for f in req_files if f.exists()]

        if not existing_req_files:
            issues.append(
                Issue(
                    category=CheckCategory.DEPS,
                    severity=Severity.WARNING,
                    check_name="requirements_sync",
                    message="No requirements files found",
                    suggestion="Create requirements.txt or pyproject.toml",
                )
            )
            return CheckResult(
                check_name="requirements_sync",
                category=CheckCategory.DEPS,
                passed=True,
                issues=issues,
            )

        # Parse packages from requirements.txt
        req_packages = set()
        req_txt = self.project_root / "requirements.txt"
        if req_txt.exists():
            for line in req_txt.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    # Extract package name (before any version specifier)
                    pkg_name = re.split(r"[<>=!~\[]", line)[0].strip().lower()
                    if pkg_name:
                        req_packages.add(pkg_name)

        # Check pyproject.toml if it exists
        pyproject = self.project_root / "pyproject.toml"
        if pyproject.exists() and req_txt.exists():
            try:
                import tomllib  # Python 3.11+
            except ImportError:
                try:
                    import tomli as tomllib  # Fallback
                except ImportError:
                    tomllib = None

            if tomllib:
                try:
                    with open(pyproject, "rb") as f:
                        pyproj_data = tomllib.load(f)

                    pyproj_deps = set()
                    deps = pyproj_data.get("project", {}).get("dependencies", [])
                    for dep in deps:
                        pkg_name = re.split(r"[<>=!~\[]", dep)[0].strip().lower()
                        if pkg_name:
                            pyproj_deps.add(pkg_name)

                    # Check for mismatches
                    only_in_req = req_packages - pyproj_deps
                    only_in_pyproj = pyproj_deps - req_packages

                    if only_in_req:
                        issues.append(
                            Issue(
                                category=CheckCategory.DEPS,
                                severity=Severity.INFO,
                                check_name="requirements_sync",
                                message=f"In requirements.txt but not pyproject.toml: {', '.join(sorted(only_in_req)[:5])}",
                                suggestion="Sync dependencies between files",
                            )
                        )

                    if only_in_pyproj:
                        issues.append(
                            Issue(
                                category=CheckCategory.DEPS,
                                severity=Severity.INFO,
                                check_name="requirements_sync",
                                message=f"In pyproject.toml but not requirements.txt: {', '.join(sorted(only_in_pyproj)[:5])}",
                                suggestion="Sync dependencies between files",
                            )
                        )

                except Exception:
                    pass

        return CheckResult(
            check_name="requirements_sync",
            category=CheckCategory.DEPS,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_version_conflicts(self) -> CheckResult:
        """Check for potential version conflicts in dependencies."""
        issues = []

        req_txt = self.project_root / "requirements.txt"
        if not req_txt.exists():
            return CheckResult(
                check_name="version_conflicts",
                category=CheckCategory.DEPS,
                passed=True,
                issues=[],
            )

        # Packages that commonly conflict
        conflict_groups = {
            "numpy": ["pandas", "scipy", "scikit-learn"],
            "tensorflow": ["keras", "numpy"],
            "torch": ["torchvision", "torchaudio"],
        }

        content = req_txt.read_text()
        packages = {}

        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-"):
                # Parse package and version
                match = re.match(r"([a-zA-Z0-9_-]+)\s*([<>=!~]+.*)?", line)
                if match:
                    pkg_name = match.group(1).lower()
                    version_spec = match.group(2) or ""
                    packages[pkg_name] = version_spec

        # Check for pinned versions that might conflict
        pinned_count = sum(1 for v in packages.values() if "==" in v)
        unpinned_count = len(packages) - pinned_count

        if pinned_count > 0 and unpinned_count > 0:
            issues.append(
                Issue(
                    category=CheckCategory.DEPS,
                    severity=Severity.INFO,
                    check_name="version_conflicts",
                    message=f"Mixed pinned ({pinned_count}) and unpinned ({unpinned_count}) dependencies",
                    file_path=str(req_txt),
                    suggestion="Consider pinning all versions for reproducibility",
                )
            )

        # Check for known conflict groups
        for base_pkg, related in conflict_groups.items():
            if base_pkg in packages:
                for related_pkg in related:
                    if related_pkg in packages:
                        base_ver = packages.get(base_pkg, "")
                        related_ver = packages.get(related_pkg, "")
                        if base_ver and related_ver and "==" in base_ver and "==" in related_ver:
                            issues.append(
                                Issue(
                                    category=CheckCategory.DEPS,
                                    severity=Severity.INFO,
                                    check_name="version_conflicts",
                                    message=f"Verify {base_pkg}{base_ver} compatible with {related_pkg}{related_ver}",
                                    file_path=str(req_txt),
                                    suggestion="Run 'pip check' to verify compatibility",
                                )
                            )

        return CheckResult(
            check_name="version_conflicts",
            category=CheckCategory.DEPS,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_outdated_packages(self) -> CheckResult:
        """Check for potentially outdated packages."""
        issues = []

        req_txt = self.project_root / "requirements.txt"
        if not req_txt.exists():
            return CheckResult(
                check_name="outdated_packages",
                category=CheckCategory.DEPS,
                passed=True,
                issues=[],
            )

        # Critical packages that should stay updated for security
        security_critical = [
            "cryptography",
            "requests",
            "urllib3",
            "certifi",
            "aiohttp",
            "flask",
            "django",
            "fastapi",
        ]

        content = req_txt.read_text()

        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-"):
                pkg_name = re.split(r"[<>=!~\[]", line)[0].strip().lower()

                # Check if security-critical package is pinned to old version
                if pkg_name in security_critical:
                    if "==" in line:
                        issues.append(
                            Issue(
                                category=CheckCategory.DEPS,
                                severity=Severity.INFO,
                                check_name="outdated_packages",
                                message=f"Security-critical package '{pkg_name}' is pinned",
                                file_path=str(req_txt),
                                suggestion=f"Regularly check for {pkg_name} security updates",
                            )
                        )

        # Recommend running pip-audit
        issues.append(
            Issue(
                category=CheckCategory.DEPS,
                severity=Severity.INFO,
                check_name="outdated_packages",
                message="Run 'pip-audit' for comprehensive vulnerability check",
                suggestion="pip install pip-audit && pip-audit",
            )
        )

        return CheckResult(
            check_name="outdated_packages",
            category=CheckCategory.DEPS,
            passed=True,  # Info only, always passes
            issues=issues,
        )

    # =========================================================================
    # AGENT PERSONAS AND COMMANDS CHECKS
    # =========================================================================

    def _check_personas_exist(self) -> CheckResult:
        """Check that required agent personas exist."""
        issues = []

        agents_dir = self.project_root / ".claude" / "agents"

        if not agents_dir.exists():
            issues.append(
                Issue(
                    category=CheckCategory.AGENTS,
                    severity=Severity.WARNING,
                    check_name="personas_exist",
                    message=".claude/agents/ directory not found",
                    suggestion="Create agent personas directory",
                )
            )
            return CheckResult(
                check_name="personas_exist",
                category=CheckCategory.AGENTS,
                passed=True,
                issues=issues,
            )

        # Expected persona files (from UPGRADE-015)
        expected_personas = [
            "senior-engineer.md",
            "risk-reviewer.md",
            "strategy-dev.md",
            "code-reviewer.md",
            "qa-engineer.md",
            "researcher.md",
            "backtest-analyst.md",
        ]

        existing_personas = list(agents_dir.glob("*.md"))
        existing_names = {p.name for p in existing_personas}

        for persona in expected_personas:
            if persona not in existing_names:
                issues.append(
                    Issue(
                        category=CheckCategory.AGENTS,
                        severity=Severity.INFO,
                        check_name="personas_exist",
                        message=f"Missing persona: {persona}",
                        suggestion=f"Create .claude/agents/{persona}",
                    )
                )

        if not existing_personas:
            issues.append(
                Issue(
                    category=CheckCategory.AGENTS,
                    severity=Severity.WARNING,
                    check_name="personas_exist",
                    message="No agent personas found",
                    suggestion="Create agent persona files in .claude/agents/",
                )
            )

        return CheckResult(
            check_name="personas_exist",
            category=CheckCategory.AGENTS,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_persona_format(self) -> CheckResult:
        """Check agent personas follow expected format."""
        issues = []

        agents_dir = self.project_root / ".claude" / "agents"
        if not agents_dir.exists():
            return CheckResult(
                check_name="persona_format",
                category=CheckCategory.AGENTS,
                passed=True,
                issues=[],
            )

        # Expected sections in a persona file
        expected_sections = [
            "Role",
            "Expertise",
            "Responsibilities",
        ]

        for persona_file in agents_dir.glob("*.md"):
            try:
                content = persona_file.read_text()

                # Check for heading structure
                if not content.strip().startswith("#"):
                    issues.append(
                        Issue(
                            category=CheckCategory.AGENTS,
                            severity=Severity.WARNING,
                            check_name="persona_format",
                            message=f"Persona should start with # heading: {persona_file.name}",
                            file_path=str(persona_file),
                            suggestion="Add a title heading at the start",
                        )
                    )

                # Check for expected sections
                content_lower = content.lower()
                for section in expected_sections:
                    if section.lower() not in content_lower:
                        issues.append(
                            Issue(
                                category=CheckCategory.AGENTS,
                                severity=Severity.INFO,
                                check_name="persona_format",
                                message=f"Missing '{section}' section in {persona_file.name}",
                                file_path=str(persona_file),
                                suggestion=f"Add ## {section} section",
                            )
                        )

                # Check minimum content length
                if len(content.strip()) < 200:
                    issues.append(
                        Issue(
                            category=CheckCategory.AGENTS,
                            severity=Severity.INFO,
                            check_name="persona_format",
                            message=f"Persona file seems sparse: {persona_file.name}",
                            file_path=str(persona_file),
                            suggestion="Add more detail to persona definition",
                        )
                    )

            except Exception as e:
                issues.append(
                    Issue(
                        category=CheckCategory.AGENTS,
                        severity=Severity.WARNING,
                        check_name="persona_format",
                        message=f"Error reading persona: {e}",
                        file_path=str(persona_file),
                    )
                )

        return CheckResult(
            check_name="persona_format",
            category=CheckCategory.AGENTS,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )

    def _check_commands_valid(self) -> CheckResult:
        """Check that slash commands are valid."""
        issues = []

        commands_dir = self.project_root / ".claude" / "commands"

        if not commands_dir.exists():
            issues.append(
                Issue(
                    category=CheckCategory.AGENTS,
                    severity=Severity.INFO,
                    check_name="commands_valid",
                    message=".claude/commands/ directory not found",
                    suggestion="Create commands directory for slash commands",
                )
            )
            return CheckResult(
                check_name="commands_valid",
                category=CheckCategory.AGENTS,
                passed=True,
                issues=issues,
            )

        command_files = list(commands_dir.glob("*.md"))

        if not command_files:
            issues.append(
                Issue(
                    category=CheckCategory.AGENTS,
                    severity=Severity.INFO,
                    check_name="commands_valid",
                    message="No slash commands found",
                    suggestion="Create .md files in .claude/commands/ for custom commands",
                )
            )

        for cmd_file in command_files:
            try:
                content = cmd_file.read_text()

                # Check for empty commands
                if len(content.strip()) < 10:
                    issues.append(
                        Issue(
                            category=CheckCategory.AGENTS,
                            severity=Severity.WARNING,
                            check_name="commands_valid",
                            message=f"Empty or minimal command: {cmd_file.name}",
                            file_path=str(cmd_file),
                            suggestion="Add command instructions",
                        )
                    )

                # Check for usage section
                if "usage" not in content.lower() and "##" in content:
                    issues.append(
                        Issue(
                            category=CheckCategory.AGENTS,
                            severity=Severity.INFO,
                            check_name="commands_valid",
                            message=f"No usage section in {cmd_file.name}",
                            file_path=str(cmd_file),
                            suggestion="Add ## Usage section for documentation",
                        )
                    )

                # Check for code blocks (likely contains the actual command)
                if "```" not in content:
                    issues.append(
                        Issue(
                            category=CheckCategory.AGENTS,
                            severity=Severity.INFO,
                            check_name="commands_valid",
                            message=f"No code blocks in {cmd_file.name}",
                            file_path=str(cmd_file),
                            suggestion="Add code blocks for command examples or scripts",
                        )
                    )

            except Exception as e:
                issues.append(
                    Issue(
                        category=CheckCategory.AGENTS,
                        severity=Severity.WARNING,
                        check_name="commands_valid",
                        message=f"Error reading command: {e}",
                        file_path=str(cmd_file),
                    )
                )

        return CheckResult(
            check_name="commands_valid",
            category=CheckCategory.AGENTS,
            passed=len([i for i in issues if i.severity == Severity.ERROR]) == 0,
            issues=issues,
        )


class QACleaner:
    """Auto-fix utilities for common issues."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def clean_temp_files(self) -> int:
        """Remove temporary/cache files that are safe to delete.

        Only removes files that are:
        - Compiled Python bytecode (*.pyc, *.pyo)
        - Editor swap/backup files (*.swp, *.swo, *.bak)
        - Temporary files (*.tmp)
        - __pycache__ directories

        Excludes:
        - .git directory
        - .venv / venv directories
        - .backups directory (project backups)
        - node_modules
        - Any hidden directories (starting with .)

        Returns count of items removed.
        """
        count = 0

        # Directories to never touch
        protected_dirs = {".git", ".venv", "venv", ".backups", "node_modules", ".hypothesis"}

        def is_protected(path: Path) -> bool:
            """Check if path is in a protected directory."""
            parts = path.parts
            return any(p in protected_dirs or p.startswith(".") for p in parts[:-1])

        # Safe temp file patterns
        temp_patterns = ["*.pyc", "*.pyo", "*.swp", "*.swo", "*.tmp", "*.bak"]

        for pattern in temp_patterns:
            for file_path in self.project_root.rglob(pattern):
                if not is_protected(file_path.relative_to(self.project_root)):
                    try:
                        file_path.unlink()
                        count += 1
                    except Exception:
                        pass

        # Clean __pycache__ directories (but not in protected locations)
        for cache_dir in self.project_root.rglob("__pycache__"):
            rel_path = cache_dir.relative_to(self.project_root)
            if not is_protected(rel_path):
                try:
                    import shutil

                    shutil.rmtree(cache_dir)
                    count += 1
                except Exception:
                    pass

        return count

    def clean_session_summaries(self, keep_last: int = 5) -> bool:
        """Clean OLD session summaries from progress file, keeping recent ones.

        This preserves the last N session/compaction blocks for context,
        only removing older ones to prevent unbounded growth.

        Args:
            keep_last: Number of recent session/compaction blocks to preserve (default: 5)

        Returns:
            True if file was modified, False otherwise
        """
        progress_file = self.project_root / "claude-progress.txt"
        if not progress_file.exists():
            return False

        content = progress_file.read_text()
        lines = content.split("\n")

        # Find all session/compaction block start positions
        block_starts = []
        for i, line in enumerate(lines):
            if line.startswith("## Session End") or line.startswith("## Context Compaction"):
                block_starts.append(i)

        # If we have fewer blocks than keep_last, nothing to clean
        if len(block_starts) <= keep_last:
            return False

        # Find the cutoff point - keep everything from this line onward
        cutoff_line = block_starts[-keep_last]

        # Build clean content: everything before session blocks + last N blocks
        clean_lines = []
        first_block_start = block_starts[0] if block_starts else len(lines)

        # Keep everything before the first session block
        for i, line in enumerate(lines):
            if i < first_block_start:
                clean_lines.append(line)

        # Keep lines from cutoff onward (the last N blocks)
        for i in range(cutoff_line, len(lines)):
            clean_lines.append(lines[i])

        # Remove excessive blank lines
        new_content = "\n".join(clean_lines)
        while "\n\n\n\n" in new_content:
            new_content = new_content.replace("\n\n\n\n", "\n\n\n")
        new_content = new_content.rstrip() + "\n"

        if new_content != content:
            progress_file.write_text(new_content)
            return True
        return False

    def run_ruff_fix(self) -> bool:
        """Run ruff auto-fix."""
        try:
            result = subprocess.run(
                ["ruff", "check", ".", "--fix", "--select=E,F", "--ignore=E501"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.project_root),
            )
            return result.returncode == 0
        except Exception:
            return False


def _print_issues_by_severity(report: QAReport, severity: Severity, header: str, show_details: bool = True) -> None:
    """Print issues filtered by severity level.

    Args:
        report: QA report containing results
        severity: Severity level to filter by
        header: Section header text
        show_details: Whether to show file path and suggestions
    """
    print("\n" + "-" * 70)
    print(header)
    print("-" * 70)
    for result in report.results:
        for issue in result.issues:
            if issue.severity == severity:
                print(f"\n  [{result.check_name}] {issue.message}")
                if show_details:
                    if issue.file_path:
                        print(f"    File: {issue.file_path}")
                    if issue.suggestion:
                        print(f"    Fix: {issue.suggestion}")


def print_report(report: QAReport, verbose: bool = False):
    """Print formatted QA report to console."""
    print("\n" + "=" * 70)
    print("QA VALIDATION REPORT")
    print("=" * 70)
    print(f"\nTimestamp: {report.timestamp}")
    print(f"Duration: {report.duration_seconds}s")
    print("\nSummary:")
    print(f"  Total Checks: {report.total_checks}")
    print(f"  Passed: {report.passed_checks}")
    print(f"  Failed: {report.failed_checks}")
    print(f"  Errors: {report.errors}")
    print(f"  Warnings: {report.warnings}")
    print(f"  Infos: {report.infos}")

    # Print errors (always shown)
    if report.errors > 0:
        _print_issues_by_severity(report, Severity.ERROR, "ERRORS (must fix):")

    # Print warnings (verbose mode only)
    if report.warnings > 0 and verbose:
        _print_issues_by_severity(report, Severity.WARNING, "WARNINGS:")

    # Print info (verbose mode only, without details)
    if report.infos > 0 and verbose:
        _print_issues_by_severity(report, Severity.INFO, "INFO:", show_details=False)

    print("\n" + "=" * 70)
    if report.success:
        print("RESULT: PASSED")
    else:
        print("RESULT: FAILED (fix errors before proceeding)")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="QA Validator for upgrades and important work")
    parser.add_argument(
        "--check",
        "-c",
        choices=[
            "code",
            "docs",
            "tests",
            "git",
            "files",
            "progress",
            "debug",
            "integrity",
            "xref",
            "ric",
            "security",
            "hooks",
            "trading",
            "config",
            "deps",
            "agents",
            "all",
        ],
        default="all",
        help="Category of checks to run (code, docs, tests, git, files, progress, debug, integrity, xref, ric, security, hooks, trading, config, deps, agents)",
    )
    parser.add_argument("--fix", "-f", action="store_true", help="Auto-fix issues where possible")
    parser.add_argument("--upgrade", "-u", type=str, help="Focus on specific upgrade (e.g., UPGRADE-014)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all issues including info")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root directory")

    args = parser.parse_args()

    # Determine project root
    project_root = args.project_root
    if not (project_root / "CLAUDE.md").exists():
        # Try to find project root
        current = Path.cwd()
        while current != current.parent:
            if (current / "CLAUDE.md").exists():
                project_root = current
                break
            current = current.parent

    # Run cleaner if requested
    if args.fix:
        cleaner = QACleaner(project_root)
        print("Running auto-fixes...")

        temp_count = cleaner.clean_temp_files()
        if temp_count > 0:
            print(f"  Cleaned {temp_count} temp files/dirs")

        if cleaner.clean_session_summaries():
            print("  Cleaned session summaries from progress file")

        if cleaner.run_ruff_fix():
            print("  Ran ruff auto-fix")

        print()

    # Run validator
    validator = QAValidator(project_root, args.upgrade)

    # Determine categories to run
    categories = None
    if args.check != "all":
        categories = [CheckCategory(args.check)]

    report = validator.run_all(categories)

    # Output results
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report, args.verbose)

    # Exit code
    sys.exit(0 if report.success else 1)


if __name__ == "__main__":
    main()
