"""
Unit tests for QA Validator System.

Tests the core functionality of scripts/qa_validator.py including:
- Check registration and execution
- Issue detection and severity classification
- Report generation
- Cleaner utilities
"""

import sys
from pathlib import Path


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from scripts.qa_validator import (
    CheckCategory,
    CheckResult,
    Issue,
    QACleaner,
    QAReport,
    QAValidator,
    Severity,
    print_report,
)


class TestQAValidator:
    """Tests for QAValidator class."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project structure for testing."""
        # Create basic structure
        (tmp_path / "CLAUDE.md").write_text("# Test Project")
        (tmp_path / "algorithms").mkdir()
        (tmp_path / "algorithms" / "__init__.py").write_text("")
        (tmp_path / "algorithms" / "test_algo.py").write_text("def hello(): pass")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "__init__.py").write_text("")
        return tmp_path

    def test_validator_initialization(self, temp_project):
        """Test validator initializes correctly."""
        validator = QAValidator(temp_project)
        assert validator.project_root == temp_project
        assert len(validator.checks) > 0
        assert "python_syntax" in validator.checks

    def test_venv_python_detection(self, temp_project):
        """Test venv Python detection falls back to python3."""
        validator = QAValidator(temp_project)
        # Without a venv, should fall back to python3
        assert validator._venv_python == "python3"

    def test_venv_python_finds_venv(self, temp_project):
        """Test venv Python detection finds existing venv."""
        # Create fake venv
        venv_dir = temp_project / ".venv" / "bin"
        venv_dir.mkdir(parents=True)
        (venv_dir / "python").write_text("#!/bin/bash\necho test")

        validator = QAValidator(temp_project)
        assert ".venv" in validator._venv_python

    def test_run_all_returns_report(self, temp_project):
        """Test run_all returns a QAReport."""
        validator = QAValidator(temp_project)
        report = validator.run_all()

        assert isinstance(report, QAReport)
        assert report.total_checks >= 0
        assert report.timestamp is not None

    def test_run_check_by_name(self, temp_project):
        """Test running a single check by name."""
        validator = QAValidator(temp_project)
        result = validator.run_check("python_syntax")

        assert result is None or isinstance(result, CheckResult)

    def test_run_check_unknown_name(self, temp_project):
        """Test running unknown check returns None."""
        validator = QAValidator(temp_project)
        result = validator.run_check("nonexistent_check")

        assert result is None


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_check_result_creation(self):
        """Test CheckResult creation."""
        result = CheckResult(
            check_name="test_check",
            category=CheckCategory.CODE,
            passed=True,
            issues=[],
        )
        assert result.check_name == "test_check"
        assert result.passed is True
        assert len(result.issues) == 0

    def test_check_result_with_issues(self):
        """Test CheckResult with issues."""
        issues = [
            Issue(
                category=CheckCategory.CODE,
                severity=Severity.WARNING,
                check_name="test",
                message="Test warning",
            )
        ]
        result = CheckResult(
            check_name="test_check",
            category=CheckCategory.CODE,
            passed=False,
            issues=issues,
        )
        assert len(result.issues) == 1
        assert result.issues[0].severity == Severity.WARNING


class TestIssue:
    """Tests for Issue dataclass."""

    def test_issue_creation(self):
        """Test Issue creation with required fields."""
        issue = Issue(
            category=CheckCategory.CODE,
            severity=Severity.ERROR,
            check_name="syntax_check",
            message="Syntax error found",
        )
        assert issue.category == CheckCategory.CODE
        assert issue.severity == Severity.ERROR
        assert issue.message == "Syntax error found"
        assert issue.file_path is None

    def test_issue_with_optional_fields(self):
        """Test Issue with all fields."""
        issue = Issue(
            category=CheckCategory.TESTS,
            severity=Severity.INFO,
            check_name="coverage",
            message="Low coverage",
            file_path="tests/test_foo.py",
            line_number=42,
            suggestion="Add more tests",
            auto_fixable=True,
        )
        assert issue.file_path == "tests/test_foo.py"
        assert issue.line_number == 42
        assert issue.suggestion == "Add more tests"
        assert issue.auto_fixable is True


class TestQAReport:
    """Tests for QAReport dataclass."""

    def test_report_success_with_no_errors(self):
        """Test report success when no errors."""
        report = QAReport(
            timestamp="2025-01-01T00:00:00",
            duration_seconds=1.0,
            total_checks=5,
            passed_checks=5,
            failed_checks=0,
            errors=0,
            warnings=2,
            infos=1,
        )
        assert report.success is True

    def test_report_failure_with_errors(self):
        """Test report failure when errors exist."""
        report = QAReport(
            timestamp="2025-01-01T00:00:00",
            duration_seconds=1.0,
            total_checks=5,
            passed_checks=4,
            failed_checks=1,
            errors=1,
            warnings=0,
            infos=0,
        )
        assert report.success is False

    def test_report_to_dict(self):
        """Test report serialization."""
        report = QAReport(
            timestamp="2025-01-01T00:00:00",
            duration_seconds=1.5,
            total_checks=3,
            passed_checks=2,
            failed_checks=1,
            errors=1,
            warnings=0,
            infos=0,
        )
        d = report.to_dict()
        assert d["timestamp"] == "2025-01-01T00:00:00"
        assert d["summary"]["total_checks"] == 3
        assert d["summary"]["errors"] == 1


class TestQACleaner:
    """Tests for QACleaner class."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create temporary project with cleanable files."""
        (tmp_path / "CLAUDE.md").write_text("# Test")
        (tmp_path / "test.pyc").write_text("bytecode")
        (tmp_path / "test.tmp").write_text("temp")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "module.pyc").write_text("cache")
        return tmp_path

    def test_clean_temp_files(self, temp_project):
        """Test cleaning temp files."""
        cleaner = QACleaner(temp_project)
        count = cleaner.clean_temp_files()

        # Should have cleaned at least the .pyc and .tmp files
        assert count >= 2
        assert not (temp_project / "test.pyc").exists()
        assert not (temp_project / "test.tmp").exists()

    def test_clean_respects_protected_dirs(self, temp_project):
        """Test cleaner respects protected directories."""
        # Create file in .git (should be protected)
        git_dir = temp_project / ".git"
        git_dir.mkdir()
        (git_dir / "test.tmp").write_text("protected")

        cleaner = QACleaner(temp_project)
        cleaner.clean_temp_files()

        # File in .git should NOT be deleted
        assert (git_dir / "test.tmp").exists()

    def test_clean_session_summaries_no_file(self, temp_project):
        """Test clean_session_summaries with no progress file."""
        cleaner = QACleaner(temp_project)
        result = cleaner.clean_session_summaries()
        assert result is False

    def test_clean_session_summaries_preserves_recent(self, temp_project):
        """Test clean_session_summaries keeps recent blocks."""
        progress_file = temp_project / "claude-progress.txt"
        content = """# Progress

## Tasks
- [x] Task 1

## Session End 1
Summary 1

## Session End 2
Summary 2

## Session End 3
Summary 3
"""
        progress_file.write_text(content)

        cleaner = QACleaner(temp_project)
        result = cleaner.clean_session_summaries(keep_last=5)

        # Should not modify since we have < 5 blocks
        assert result is False


class TestPrintReport:
    """Tests for print_report function."""

    def test_print_report_success(self, capsys):
        """Test printing successful report."""
        report = QAReport(
            timestamp="2025-01-01T00:00:00",
            duration_seconds=1.0,
            total_checks=5,
            passed_checks=5,
            failed_checks=0,
            errors=0,
            warnings=0,
            infos=0,
        )
        print_report(report)

        captured = capsys.readouterr()
        assert "RESULT: PASSED" in captured.out
        assert "Total Checks: 5" in captured.out

    def test_print_report_failure(self, capsys):
        """Test printing failed report."""
        report = QAReport(
            timestamp="2025-01-01T00:00:00",
            duration_seconds=1.0,
            total_checks=5,
            passed_checks=4,
            failed_checks=1,
            errors=1,
            warnings=0,
            infos=0,
            results=[
                CheckResult(
                    check_name="test_check",
                    category=CheckCategory.CODE,
                    passed=False,
                    issues=[
                        Issue(
                            category=CheckCategory.CODE,
                            severity=Severity.ERROR,
                            check_name="test_check",
                            message="Test error",
                        )
                    ],
                )
            ],
        )
        print_report(report)

        captured = capsys.readouterr()
        assert "RESULT: FAILED" in captured.out
        assert "ERRORS" in captured.out


class TestSeverityEnum:
    """Tests for Severity enum."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert Severity.ERROR.value == "error"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"


class TestCheckCategoryEnum:
    """Tests for CheckCategory enum."""

    def test_category_values(self):
        """Test check category values."""
        assert CheckCategory.CODE.value == "code"
        assert CheckCategory.DOCS.value == "docs"
        assert CheckCategory.TESTS.value == "tests"
        assert CheckCategory.SECURITY.value == "security"
