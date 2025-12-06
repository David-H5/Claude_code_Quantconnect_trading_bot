#!/usr/bin/env python3
"""
Algorithm Validator for QuantConnect Trading Bot

This module provides validation for trading algorithms before they are
deployed to backtesting or live trading. It checks for:
- Syntax errors
- Required methods
- Risk management implementation
- Code quality standards
- Common QuantConnect pitfalls

Author: QuantConnect Trading Bot
Date: 2025-11-25
"""

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"  # Must be fixed before deployment
    WARNING = "warning"  # Should be reviewed
    INFO = "info"  # Informational only


@dataclass
class ValidationIssue:
    """Represents a validation issue found in an algorithm."""

    severity: ValidationSeverity
    message: str
    line_number: int | None = None
    suggestion: str | None = None


@dataclass
class ValidationResult:
    """Result of algorithm validation."""

    is_valid: bool
    algorithm_name: str
    file_path: str
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def add_issue(
        self,
        severity: ValidationSeverity,
        message: str,
        line_number: int | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Add a validation issue."""
        self.issues.append(ValidationIssue(severity, message, line_number, suggestion))

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get all error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get all warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def summary(self) -> str:
        """Generate a summary of validation results."""
        lines = [
            f"Validation Result for {self.algorithm_name}",
            f"File: {self.file_path}",
            f"Valid: {self.is_valid}",
            f"Errors: {len(self.errors)}, Warnings: {len(self.warnings)}",
            "",
        ]

        if self.errors:
            lines.append("ERRORS:")
            for issue in self.errors:
                line_info = f" (line {issue.line_number})" if issue.line_number else ""
                lines.append(f"  - {issue.message}{line_info}")
                if issue.suggestion:
                    lines.append(f"    Suggestion: {issue.suggestion}")

        if self.warnings:
            lines.append("\nWARNINGS:")
            for issue in self.warnings:
                line_info = f" (line {issue.line_number})" if issue.line_number else ""
                lines.append(f"  - {issue.message}{line_info}")

        return "\n".join(lines)


class AlgorithmValidator:
    """
    Validates QuantConnect trading algorithms.

    Performs static analysis and pattern checking to identify
    potential issues before deployment.
    """

    REQUIRED_METHODS = ["Initialize", "OnData"]
    RECOMMENDED_METHODS = ["OnEndOfAlgorithm", "OnOrderEvent"]

    RISK_PATTERNS = [
        (r"SetHoldings\s*\(\s*[^,]+\s*,\s*1\.0\s*\)", "Full position size (100%) detected"),
        (r"Liquidate\s*\(\s*\)", "Liquidate all positions - ensure this is intentional"),
    ]

    DANGEROUS_PATTERNS = [
        (r"while\s+True:", "Infinite loop detected - this will crash the algorithm"),
        (r"time\.sleep", "sleep() calls not allowed in QuantConnect"),
        (r"input\s*\(", "input() calls not allowed in QuantConnect"),
        (r"print\s*\(", "Use self.Debug() or self.Log() instead of print()"),
    ]

    def __init__(self):
        """Initialize the validator."""
        pass

    def validate_file(self, filepath: Path) -> ValidationResult:
        """
        Validate an algorithm file.

        Args:
            filepath: Path to the algorithm file

        Returns:
            ValidationResult with all findings
        """
        filepath = Path(filepath)
        result = ValidationResult(
            is_valid=True,
            algorithm_name=filepath.stem,
            file_path=str(filepath),
        )

        if not filepath.exists():
            result.is_valid = False
            result.add_issue(ValidationSeverity.ERROR, "File does not exist")
            return result

        try:
            with open(filepath) as f:
                content = f.read()
                lines = content.split("\n")
        except Exception as e:
            result.is_valid = False
            result.add_issue(ValidationSeverity.ERROR, f"Cannot read file: {e}")
            return result

        # Run all validation checks
        self._check_syntax(content, result)
        self._check_required_methods(content, result)
        self._check_class_structure(content, result)
        self._check_risk_management(content, lines, result)
        self._check_dangerous_patterns(content, lines, result)
        self._check_data_validation(content, lines, result)
        self._check_indicator_warmup(content, lines, result)
        self._check_code_quality(content, lines, result)

        # Calculate metrics
        result.metrics = self._calculate_metrics(content, lines)

        # Set final validity
        result.is_valid = len(result.errors) == 0

        return result

    def _check_syntax(self, content: str, result: ValidationResult) -> None:
        """Check for Python syntax errors."""
        try:
            ast.parse(content)
        except SyntaxError as e:
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Syntax error: {e.msg}",
                line_number=e.lineno,
            )

    def _check_required_methods(self, content: str, result: ValidationResult) -> None:
        """Check for required QuantConnect methods."""
        for method in self.REQUIRED_METHODS:
            pattern = rf"def\s+{method}\s*\("
            if not re.search(pattern, content):
                result.add_issue(
                    ValidationSeverity.ERROR,
                    f"Required method '{method}' not found",
                    suggestion=f"Add 'def {method}(self, ...)' method to your algorithm",
                )

        for method in self.RECOMMENDED_METHODS:
            pattern = rf"def\s+{method}\s*\("
            if not re.search(pattern, content):
                result.add_issue(
                    ValidationSeverity.INFO,
                    f"Recommended method '{method}' not implemented",
                )

    def _check_class_structure(self, content: str, result: ValidationResult) -> None:
        """Check algorithm class structure."""
        # Check for QCAlgorithm inheritance
        if not re.search(r"class\s+\w+\s*\(\s*QCAlgorithm\s*\)", content):
            result.add_issue(
                ValidationSeverity.ERROR,
                "Algorithm class must inherit from QCAlgorithm",
                suggestion="class MyAlgorithm(QCAlgorithm):",
            )

        # Check for AlgorithmImports
        if "from AlgorithmImports import" not in content:
            result.add_issue(
                ValidationSeverity.WARNING,
                "Missing 'from AlgorithmImports import *'",
                suggestion="Add 'from AlgorithmImports import *' at the top",
            )

    def _check_risk_management(self, content: str, lines: list[str], result: ValidationResult) -> None:
        """Check for risk management implementation."""
        # Check for position sizing
        for pattern, message in self.RISK_PATTERNS:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    result.add_issue(
                        ValidationSeverity.WARNING,
                        message,
                        line_number=i,
                    )

        # Check for stop loss implementation
        has_stop_loss = any(
            keyword in content.lower() for keyword in ["stoploss", "stop_loss", "stoplimitorder", "stopmarketorder"]
        )
        if not has_stop_loss:
            result.add_issue(
                ValidationSeverity.WARNING,
                "No stop-loss implementation detected",
                suggestion="Consider adding stop-loss orders for risk management",
            )

    def _check_dangerous_patterns(self, content: str, lines: list[str], result: ValidationResult) -> None:
        """Check for dangerous code patterns."""
        for pattern, message in self.DANGEROUS_PATTERNS:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    result.add_issue(
                        ValidationSeverity.ERROR,
                        message,
                        line_number=i,
                    )

    def _check_data_validation(self, content: str, lines: list[str], result: ValidationResult) -> None:
        """Check for proper data validation."""
        # Check for data access without ContainsKey
        has_data_access = re.search(r"data\[", content)
        has_contains_key = "ContainsKey" in content

        if has_data_access and not has_contains_key:
            result.add_issue(
                ValidationSeverity.WARNING,
                "Data accessed without ContainsKey check",
                suggestion="Use 'if data.ContainsKey(symbol):' before accessing data",
            )

    def _check_indicator_warmup(self, content: str, lines: list[str], result: ValidationResult) -> None:
        """Check for indicator warmup handling."""
        # Check if indicators are created
        indicator_patterns = [
            r"self\.RSI\s*\(",
            r"self\.SMA\s*\(",
            r"self\.EMA\s*\(",
            r"self\.MACD\s*\(",
            r"self\.BB\s*\(",
        ]

        has_indicators = any(re.search(p, content) for p in indicator_patterns)

        if has_indicators:
            # Check for warmup
            if "SetWarmUp" not in content:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "Indicators used without SetWarmUp",
                    suggestion="Add self.SetWarmUp(period) in Initialize()",
                )

            # Check for IsWarmingUp check
            if "IsWarmingUp" not in content:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "No IsWarmingUp check in OnData",
                    suggestion="Add 'if self.IsWarmingUp: return' at start of OnData",
                )

            # Check for IsReady check
            if "IsReady" not in content:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "Indicator IsReady not checked before use",
                    suggestion="Check indicator.IsReady before using indicator values",
                )

    def _check_code_quality(self, content: str, lines: list[str], result: ValidationResult) -> None:
        """Check code quality standards."""
        # Check for docstrings
        if '"""' not in content and "'''" not in content:
            result.add_issue(
                ValidationSeverity.INFO,
                "No docstrings found",
                suggestion="Add docstrings to classes and methods",
            )

        # Check for type hints
        if " -> " not in content:
            result.add_issue(
                ValidationSeverity.INFO,
                "No return type hints found",
                suggestion="Add type hints to method signatures",
            )

        # Check line length
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                result.add_issue(
                    ValidationSeverity.INFO,
                    f"Line exceeds 120 characters ({len(line)} chars)",
                    line_number=i,
                )

    def _calculate_metrics(self, content: str, lines: list[str]) -> dict:
        """Calculate code metrics."""
        return {
            "total_lines": len(lines),
            "code_lines": len([l for l in lines if l.strip() and not l.strip().startswith("#")]),
            "comment_lines": len([l for l in lines if l.strip().startswith("#")]),
            "blank_lines": len([l for l in lines if not l.strip()]),
            "num_methods": len(re.findall(r"def\s+\w+\s*\(", content)),
            "num_classes": len(re.findall(r"class\s+\w+", content)),
        }


def validate_algorithm(filepath: str) -> tuple[bool, str]:
    """
    Convenience function to validate an algorithm.

    Args:
        filepath: Path to algorithm file

    Returns:
        Tuple of (is_valid, summary_message)
    """
    validator = AlgorithmValidator()
    result = validator.validate_file(Path(filepath))
    return result.is_valid, result.summary()


def validate_all_algorithms(algorithms_dir: str = "algorithms") -> dict[str, ValidationResult]:
    """
    Validate all algorithms in a directory.

    Args:
        algorithms_dir: Path to algorithms directory

    Returns:
        Dictionary mapping filenames to validation results
    """
    validator = AlgorithmValidator()
    results = {}

    algorithms_path = Path(algorithms_dir)
    for algo_file in algorithms_path.glob("*.py"):
        if algo_file.name != "__init__.py":
            results[algo_file.name] = validator.validate_file(algo_file)

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "algorithms/simple_momentum.py"

    is_valid, summary = validate_algorithm(filepath)
    print(summary)
    sys.exit(0 if is_valid else 1)
