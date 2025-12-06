#!/usr/bin/env python3
"""
Results Analyzer for QuantConnect Trading Bot

This module provides structured JSON analysis of test and backtest results
for Claude Code autonomous iteration. It generates actionable recommendations
that Claude can parse and act upon.

Author: QuantConnect Trading Bot
Date: 2025-11-25
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TestFailure:
    """Represents a test failure with actionable information."""

    test_name: str
    expected: str
    actual: str
    relevant_lines: list[int] = field(default_factory=list)
    suggested_investigation: str = ""
    file_path: str | None = None


@dataclass
class AnalysisResult:
    """Complete analysis result for Claude Code consumption."""

    timestamp: str
    pass_rate: float
    total_tests: int
    passed: int
    failed: int
    skipped: int
    coverage_percent: float | None = None
    failures: list[TestFailure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    actionable_recommendations: list[str] = field(default_factory=list)
    backtest_metrics: dict[str, Any] = field(default_factory=dict)
    convergence_status: str = "unknown"  # success, failed, stuck, in_progress

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), indent=2)


class ResultsAnalyzer:
    """
    Analyzes test and backtest results for autonomous iteration.

    Generates structured JSON output that Claude Code can parse
    to determine next actions.
    """

    # Performance thresholds
    SHARPE_TARGET = 1.0
    MAX_DRAWDOWN_TARGET = 0.20
    PROFIT_FACTOR_TARGET = 1.5
    PASS_RATE_THRESHOLD = 0.95

    def __init__(self, results_dir: Path):
        """
        Initialize analyzer.

        Args:
            results_dir: Directory containing test results
        """
        self.results_dir = Path(results_dir)

    def analyze_pytest_results(self, xml_file: Path) -> tuple[dict, list[TestFailure]]:
        """
        Analyze pytest JUnit XML results.

        Args:
            xml_file: Path to JUnit XML file

        Returns:
            Tuple of (summary_dict, failures_list)
        """
        if not xml_file.exists():
            return {"error": "Results file not found"}, []

        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get test suite stats
        testsuite = root.find("testsuite") or root
        tests = int(testsuite.get("tests", 0))
        failures = int(testsuite.get("failures", 0))
        errors = int(testsuite.get("errors", 0))
        skipped = int(testsuite.get("skipped", 0))

        passed = tests - failures - errors - skipped

        summary = {
            "total": tests,
            "passed": passed,
            "failed": failures + errors,
            "skipped": skipped,
            "pass_rate": passed / tests if tests > 0 else 0,
        }

        # Extract failure details
        failure_list = []
        for testcase in root.iter("testcase"):
            failure = testcase.find("failure")
            error = testcase.find("error")

            if failure is not None or error is not None:
                fail_elem = failure if failure is not None else error
                test_name = testcase.get("name", "unknown")
                classname = testcase.get("classname", "")

                # Parse failure message
                message = fail_elem.get("message", "")
                text = fail_elem.text or ""

                # Extract expected vs actual
                expected, actual = self._parse_assertion(message, text)

                # Try to find relevant line numbers
                lines = self._extract_line_numbers(text)

                # Generate investigation suggestion
                suggestion = self._generate_suggestion(test_name, message, text)

                failure_list.append(
                    TestFailure(
                        test_name=f"{classname}::{test_name}",
                        expected=expected,
                        actual=actual,
                        relevant_lines=lines,
                        suggested_investigation=suggestion,
                        file_path=self._extract_file_path(classname),
                    )
                )

        return summary, failure_list

    def _parse_assertion(self, message: str, text: str) -> tuple[str, str]:
        """Parse assertion message to extract expected vs actual."""
        expected = "unknown"
        actual = "unknown"

        # Try common assertion patterns
        patterns = [
            r"assert\s+(.+?)\s*==\s*(.+)",
            r"expected\s+(.+?)\s+but\s+got\s+(.+)",
            r"AssertionError:\s*(.+?)\s*!=\s*(.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, message + " " + text, re.IGNORECASE)
            if match:
                actual = match.group(1).strip()[:100]
                expected = match.group(2).strip()[:100]
                break

        return expected, actual

    def _extract_line_numbers(self, text: str) -> list[int]:
        """Extract line numbers from traceback."""
        lines = []
        pattern = r":(\d+):"
        matches = re.findall(pattern, text)
        lines = [int(m) for m in matches[:5]]  # Limit to 5 lines
        return lines

    def _extract_file_path(self, classname: str) -> str | None:
        """Extract file path from classname."""
        if "." in classname:
            parts = classname.replace(".", "/")
            return f"{parts}.py"
        return None

    def _generate_suggestion(self, test_name: str, message: str, text: str) -> str:
        """Generate investigation suggestion based on failure."""
        suggestions = []

        # Common patterns
        if "AttributeError" in message:
            suggestions.append("Check if the attribute exists and is properly initialized")
        if "TypeError" in message:
            suggestions.append("Verify function arguments and return types")
        if "KeyError" in message:
            suggestions.append("Check if the key exists in the dictionary")
        if "IndexError" in message:
            suggestions.append("Verify array/list bounds before access")
        if "AssertionError" in message:
            suggestions.append("Compare expected vs actual values in test")

        # Test-specific suggestions
        if "risk" in test_name.lower():
            suggestions.append("Review risk limit calculations and thresholds")
        if "indicator" in test_name.lower():
            suggestions.append("Check indicator warmup period and data validation")
        if "position" in test_name.lower():
            suggestions.append("Verify position sizing and portfolio calculations")

        return "; ".join(suggestions) if suggestions else "Review the test assertion and related code"

    def analyze_coverage(self, coverage_file: Path) -> float | None:
        """
        Analyze coverage results.

        Args:
            coverage_file: Path to coverage JSON file

        Returns:
            Coverage percentage or None
        """
        if not coverage_file.exists():
            return None

        try:
            with open(coverage_file) as f:
                data = json.load(f)

            # Handle different coverage report formats
            if "totals" in data:
                return data["totals"].get("percent_covered", 0)
            elif "meta" in data and "coverage" in data:
                total_lines = 0
                covered_lines = 0
                for file_data in data.get("files", {}).values():
                    summary = file_data.get("summary", {})
                    total_lines += summary.get("num_statements", 0)
                    covered_lines += summary.get("covered_lines", 0)
                return (covered_lines / total_lines * 100) if total_lines > 0 else 0

        except Exception as e:
            print(f"Error parsing coverage: {e}", file=sys.stderr)
            return None

        return None

    def analyze_backtest(self, backtest_file: Path) -> dict[str, Any]:
        """
        Analyze backtest results.

        Args:
            backtest_file: Path to backtest results JSON

        Returns:
            Backtest metrics dictionary
        """
        if not backtest_file.exists():
            return {}

        try:
            with open(backtest_file) as f:
                data = json.load(f)

            metrics = {
                "sharpe_ratio": data.get("SharpeRatio", 0),
                "total_return": data.get("TotalReturn", 0),
                "max_drawdown": data.get("MaxDrawdown", 0),
                "win_rate": data.get("WinRate", 0),
                "profit_factor": data.get("ProfitFactor", 0),
                "total_trades": data.get("TotalTrades", 0),
            }

            # Determine if backtest meets targets
            metrics["meets_sharpe_target"] = metrics["sharpe_ratio"] >= self.SHARPE_TARGET
            metrics["meets_drawdown_target"] = abs(metrics["max_drawdown"]) <= self.MAX_DRAWDOWN_TARGET
            metrics["meets_targets"] = metrics["meets_sharpe_target"] and metrics["meets_drawdown_target"]

            return metrics

        except Exception as e:
            print(f"Error parsing backtest: {e}", file=sys.stderr)
            return {}

    def generate_recommendations(
        self,
        summary: dict,
        failures: list[TestFailure],
        coverage: float | None,
        backtest: dict,
    ) -> list[str]:
        """
        Generate actionable recommendations.

        Args:
            summary: Test summary dict
            failures: List of test failures
            coverage: Coverage percentage
            backtest: Backtest metrics

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        # Test failure recommendations
        for failure in failures[:5]:  # Limit to top 5
            rec = f"Fix {failure.test_name}: {failure.suggested_investigation}"
            if failure.relevant_lines:
                rec += f" (lines: {failure.relevant_lines})"
            recommendations.append(rec)

        # Coverage recommendations
        if coverage is not None and coverage < 70:
            recommendations.append(f"Increase test coverage from {coverage:.1f}% to at least 70%")

        # Backtest recommendations
        if backtest:
            if not backtest.get("meets_sharpe_target", True):
                recommendations.append(
                    f"Improve Sharpe ratio from {backtest.get('sharpe_ratio', 0):.2f} to > {self.SHARPE_TARGET}"
                )
            if not backtest.get("meets_drawdown_target", True):
                recommendations.append(
                    f"Reduce max drawdown from {backtest.get('max_drawdown', 0):.2%} to < {self.MAX_DRAWDOWN_TARGET:.0%}"
                )

        return recommendations

    def determine_convergence(
        self,
        pass_rate: float,
        failures: list[TestFailure],
        backtest: dict,
        iteration: int = 0,
        max_iterations: int = 10,
    ) -> str:
        """
        Determine convergence status for autonomous iteration.

        Args:
            pass_rate: Test pass rate
            failures: Test failures
            backtest: Backtest metrics
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations

        Returns:
            Convergence status string
        """
        # All tests pass
        if pass_rate >= 1.0:
            if backtest.get("meets_targets", True):
                return "success"
            else:
                return "in_progress"  # Tests pass but backtest needs work

        # Acceptable threshold (95%)
        if pass_rate >= self.PASS_RATE_THRESHOLD:
            return "acceptable"

        # Max iterations reached
        if iteration >= max_iterations:
            return "max_iterations"

        # Check for stuck state (would need iteration history)
        # For now, return in_progress
        return "in_progress"

    def analyze(
        self,
        timestamp: str | None = None,
        iteration: int = 0,
    ) -> AnalysisResult:
        """
        Run complete analysis.

        Args:
            timestamp: Timestamp to look for in result files
            iteration: Current iteration number

        Returns:
            AnalysisResult with all findings
        """
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Find result files
        xml_files = list(self.results_dir.glob(f"*{timestamp}*.xml"))
        coverage_files = list(self.results_dir.glob(f"*{timestamp}*coverage*.json"))
        backtest_files = list(self.results_dir.glob("*backtest*.json"))

        # Analyze each component
        if xml_files:
            summary, failures = self.analyze_pytest_results(xml_files[0])
        else:
            # Try latest file
            all_xml = sorted(self.results_dir.glob("*.xml"))
            if all_xml:
                summary, failures = self.analyze_pytest_results(all_xml[-1])
            else:
                summary = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "pass_rate": 0}
                failures = []

        coverage = None
        if coverage_files:
            coverage = self.analyze_coverage(coverage_files[0])
        else:
            all_coverage = sorted(self.results_dir.glob("*coverage*.json"))
            if all_coverage:
                coverage = self.analyze_coverage(all_coverage[-1])

        backtest = {}
        if backtest_files:
            backtest = self.analyze_backtest(backtest_files[-1])

        # Generate recommendations
        recommendations = self.generate_recommendations(summary, failures, coverage, backtest)

        # Determine convergence
        pass_rate = summary.get("pass_rate", 0)
        convergence = self.determine_convergence(pass_rate, failures, backtest, iteration)

        # Build warnings
        warnings = []
        if coverage is not None and coverage < 70:
            warnings.append(f"Low test coverage: {coverage:.1f}%")
        if summary.get("skipped", 0) > 0:
            warnings.append(f"{summary.get('skipped')} tests were skipped")

        return AnalysisResult(
            timestamp=timestamp,
            pass_rate=pass_rate,
            total_tests=summary.get("total", 0),
            passed=summary.get("passed", 0),
            failed=summary.get("failed", 0),
            skipped=summary.get("skipped", 0),
            coverage_percent=coverage,
            failures=failures,
            warnings=warnings,
            actionable_recommendations=recommendations,
            backtest_metrics=backtest,
            convergence_status=convergence,
        )


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(description="Analyze test and backtest results for Claude Code iteration")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing result files",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Timestamp to filter result files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Current iteration number for convergence tracking",
    )

    args = parser.parse_args()

    # Create results dir if it doesn't exist
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    analyzer = ResultsAnalyzer(args.results_dir)
    result = analyzer.analyze(timestamp=args.timestamp, iteration=args.iteration)

    # Output
    json_output = result.to_json()

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_output)
        print(f"Analysis written to: {args.output}")
    else:
        print(json_output)

    # Exit with appropriate code
    if result.convergence_status == "success" or result.convergence_status == "acceptable":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
