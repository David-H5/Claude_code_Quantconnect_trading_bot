"""
Test Data Quality Validation for Evaluation Framework.

Validates test case quality, detects data issues, and ensures comprehensive
edge case coverage before running evaluations.

Features:
- Test case schema validation
- Market data quality checks (gaps, outliers, data integrity)
- Edge case coverage analysis
- Duplicate detection
- Data freshness validation (ensure 2024-2025 data)
- Synthetic data generation for missing edge cases

References:
- https://blog.poespas.me/posts/2025/02/15/automated-testing-for-edge-cases-python-behave/
- https://testrigor.com/blog/error-handling-strategies-in-automated-tests/
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ValidationIssue:
    """Issue found during test data validation."""

    severity: str  # error | warning | info
    category: str  # schema | data_quality | coverage | duplicates
    test_case_id: str
    message: str
    suggestion: str


@dataclass
class CoverageReport:
    """Edge case coverage analysis report."""

    total_test_cases: int
    success_cases: int
    edge_cases: int
    failure_cases: int
    coverage_score: float  # 0-100
    missing_scenarios: list[str]
    redundant_scenarios: list[str]


@dataclass
class ValidationResult:
    """Complete validation result."""

    passed: bool
    total_issues: int
    errors: int
    warnings: int
    issues: list[ValidationIssue]
    coverage_report: CoverageReport
    data_quality_score: float  # 0-100


class TestDataValidator:
    """
    Validates test case quality and detects data issues.

    Ensures test cases are well-formed, comprehensive, and use high-quality
    market data before running evaluations.
    """

    def __init__(self):
        """Initialize test data validator."""
        self.required_fields = {
            "case_id": str,
            "category": str,
            "agent_type": str,
            "scenario": str,
            "input_data": dict,
            "expected_output": dict,
            "success_criteria": dict,
        }

        self.valid_categories = {"success", "edge", "failure"}

    def validate_test_cases(self, test_cases: list[dict[str, Any]]) -> ValidationResult:
        """
        Validate test cases for quality and coverage.

        Args:
            test_cases: List of test case dictionaries

        Returns:
            ValidationResult with issues and coverage report
        """
        issues = []

        # Schema validation
        schema_issues = self._validate_schema(test_cases)
        issues.extend(schema_issues)

        # Duplicate detection
        duplicate_issues = self._detect_duplicates(test_cases)
        issues.extend(duplicate_issues)

        # Data quality checks
        data_quality_issues = self._check_data_quality(test_cases)
        issues.extend(data_quality_issues)

        # Coverage analysis
        coverage_report = self._analyze_coverage(test_cases)

        # Missing scenarios
        for scenario in coverage_report.missing_scenarios:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="coverage",
                    test_case_id="N/A",
                    message=f"Missing edge case scenario: {scenario}",
                    suggestion=f"Add test case for {scenario} to improve coverage",
                )
            )

        # Calculate scores
        errors = sum(1 for i in issues if i.severity == "error")
        warnings = sum(1 for i in issues if i.severity == "warning")

        # Data quality score (100 - penalties)
        data_quality_score = 100.0
        data_quality_score -= errors * 10  # -10 per error
        data_quality_score -= warnings * 2  # -2 per warning
        data_quality_score = max(0, data_quality_score)

        return ValidationResult(
            passed=errors == 0,
            total_issues=len(issues),
            errors=errors,
            warnings=warnings,
            issues=issues,
            coverage_report=coverage_report,
            data_quality_score=data_quality_score,
        )

    def _validate_schema(self, test_cases: list[dict[str, Any]]) -> list[ValidationIssue]:
        """Validate test case schema."""
        issues = []

        for test_case in test_cases:
            case_id = test_case.get("case_id", "UNKNOWN")

            # Check required fields
            for field, field_type in self.required_fields.items():
                if field not in test_case:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            category="schema",
                            test_case_id=case_id,
                            message=f"Missing required field: {field}",
                            suggestion=f"Add {field} field to test case",
                        )
                    )
                elif not isinstance(test_case[field], field_type):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            category="schema",
                            test_case_id=case_id,
                            message=f"Invalid type for {field}: expected {field_type.__name__}",
                            suggestion=f"Convert {field} to {field_type.__name__}",
                        )
                    )

            # Validate category
            category = test_case.get("category")
            if category and category not in self.valid_categories:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        category="schema",
                        test_case_id=case_id,
                        message=f"Invalid category: {category}",
                        suggestion=f"Use one of: {', '.join(self.valid_categories)}",
                    )
                )

        return issues

    def _detect_duplicates(self, test_cases: list[dict[str, Any]]) -> list[ValidationIssue]:
        """Detect duplicate test cases."""
        issues = []
        seen_ids: set[str] = set()
        seen_scenarios: dict[str, str] = {}

        for test_case in test_cases:
            case_id = test_case.get("case_id", "UNKNOWN")
            scenario = test_case.get("scenario", "")

            # Duplicate IDs
            if case_id in seen_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        category="duplicates",
                        test_case_id=case_id,
                        message=f"Duplicate case_id: {case_id}",
                        suggestion="Use unique case_id for each test",
                    )
                )
            seen_ids.add(case_id)

            # Duplicate scenarios
            scenario_key = f"{test_case.get('agent_type')}:{scenario}"
            if scenario_key in seen_scenarios:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        category="duplicates",
                        test_case_id=case_id,
                        message=f"Duplicate scenario: {scenario}",
                        suggestion=f"Combine with {seen_scenarios[scenario_key]} or differentiate scenarios",
                    )
                )
            seen_scenarios[scenario_key] = case_id

        return issues

    def _check_data_quality(self, test_cases: list[dict[str, Any]]) -> list[ValidationIssue]:
        """Check data quality in test cases."""
        issues = []

        for test_case in test_cases:
            case_id = test_case.get("case_id", "UNKNOWN")
            input_data = test_case.get("input_data", {})

            # Check for data freshness (2024-2025 data)
            date_field = input_data.get("date")
            if date_field:
                try:
                    date_obj = datetime.fromisoformat(date_field.replace("Z", "+00:00"))
                    if date_obj.year < 2024:
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                category="data_quality",
                                test_case_id=case_id,
                                message=f"Test uses old data ({date_obj.year})",
                                suggestion="Use 2024-2025 data for contamination-free testing",
                            )
                        )
                except Exception:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            category="data_quality",
                            test_case_id=case_id,
                            message=f"Invalid date format: {date_field}",
                            suggestion="Use ISO format (YYYY-MM-DD)",
                        )
                    )

            # Check for unrealistic values
            if "sharpe_ratio" in input_data:
                sharpe = input_data["sharpe_ratio"]
                if sharpe > 5.0:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            category="data_quality",
                            test_case_id=case_id,
                            message=f"Unrealistic Sharpe ratio: {sharpe}",
                            suggestion="Use realistic values (typically <3.0)",
                        )
                    )

            if "win_rate" in input_data:
                win_rate = input_data["win_rate"]
                if not (0 <= win_rate <= 1):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            category="data_quality",
                            test_case_id=case_id,
                            message=f"Invalid win_rate: {win_rate}",
                            suggestion="Use values between 0 and 1",
                        )
                    )

        return issues

    def _analyze_coverage(self, test_cases: list[dict[str, Any]]) -> CoverageReport:
        """Analyze edge case coverage."""
        total_cases = len(test_cases)
        success_cases = sum(1 for tc in test_cases if tc.get("category") == "success")
        edge_cases = sum(1 for tc in test_cases if tc.get("category") == "edge")
        failure_cases = sum(1 for tc in test_cases if tc.get("category") == "failure")

        # Define required edge case scenarios
        required_edge_scenarios = {
            "high_vix": "High VIX >35 (market volatility)",
            "low_liquidity": "Low liquidity (wide spreads)",
            "earnings_gap": "Overnight gap from earnings",
            "flash_crash": "Flash crash scenario",
            "circuit_breaker": "Market circuit breaker",
            "near_limit": "Near position/risk limits",
            "conflicting_signals": "Mixed timeframe signals",
            "data_gap": "Missing data points",
            "extreme_move": "Extreme price movement (>5%)",
        }

        # Check which scenarios are covered
        covered_scenarios = set()
        for test_case in test_cases:
            scenario = test_case.get("scenario", "").lower()
            for key in required_edge_scenarios:
                if key.replace("_", " ") in scenario:
                    covered_scenarios.add(key)

        missing_scenarios = [
            required_edge_scenarios[key] for key in required_edge_scenarios if key not in covered_scenarios
        ]

        # Calculate coverage score
        scenario_coverage = len(covered_scenarios) / len(required_edge_scenarios)
        distribution_score = (
            1.0
            - abs(success_cases / total_cases - 0.40)
            - abs(edge_cases / total_cases - 0.40)
            - abs(failure_cases / total_cases - 0.20)
        )
        coverage_score = (scenario_coverage * 0.6 + distribution_score * 0.4) * 100

        return CoverageReport(
            total_test_cases=total_cases,
            success_cases=success_cases,
            edge_cases=edge_cases,
            failure_cases=failure_cases,
            coverage_score=coverage_score,
            missing_scenarios=missing_scenarios,
            redundant_scenarios=[],  # TODO: Implement redundancy detection
        )

    def generate_synthetic_test_cases(
        self,
        agent_type: str,
        missing_scenarios: list[str],
    ) -> list[dict[str, Any]]:
        """
        Generate synthetic test cases for missing edge case scenarios.

        Args:
            agent_type: Type of agent (TechnicalAnalyst, etc.)
            missing_scenarios: List of missing scenario descriptions

        Returns:
            List of generated test case dictionaries
        """
        synthetic_cases = []

        for idx, scenario in enumerate(missing_scenarios):
            case_id = f"SYNTHETIC_{agent_type}_{idx:03d}"

            # Generate test case based on scenario
            if "high_vix" in scenario.lower():
                synthetic_case = {
                    "case_id": case_id,
                    "category": "edge",
                    "agent_type": agent_type,
                    "scenario": "High VIX >35 distorts patterns (synthetic)",
                    "input_data": {
                        "vix_level": 38.5,
                        "pattern_type": "bull_flag",
                        "in_sample_win_rate": 0.68,
                        "out_of_sample_win_rate": 0.48,
                        "volume_confirmation": True,
                        "date": "2024-02-05",
                    },
                    "expected_output": {
                        "signal": "neutral",
                        "confidence": 0.48,
                        "rejection_reason": "High VIX distorts patterns",
                    },
                    "success_criteria": {
                        "signal_correct": True,
                        "out_of_sample_check": True,
                    },
                }
                synthetic_cases.append(synthetic_case)

        return synthetic_cases


def generate_validation_report(result: ValidationResult) -> str:
    """
    Generate test data validation report.

    Args:
        result: ValidationResult object

    Returns:
        Formatted markdown report
    """
    report = []
    report.append("# Test Data Validation Report\n")
    report.append(f"**Status**: {'✅ PASSED' if result.passed else '❌ FAILED'}")
    report.append(f"**Data Quality Score**: {result.data_quality_score:.1f}/100\n")

    # Issue summary
    report.append("## Issue Summary\n")
    report.append(f"- Total Issues: {result.total_issues}")
    report.append(f"- Errors: {result.errors}")
    report.append(f"- Warnings: {result.warnings}\n")

    # Coverage report
    report.append("## Coverage Analysis\n")
    report.append(f"- Total Test Cases: {result.coverage_report.total_test_cases}")
    report.append(
        f"- Success Cases: {result.coverage_report.success_cases} ({result.coverage_report.success_cases/result.coverage_report.total_test_cases:.1%})"
    )
    report.append(
        f"- Edge Cases: {result.coverage_report.edge_cases} ({result.coverage_report.edge_cases/result.coverage_report.total_test_cases:.1%})"
    )
    report.append(
        f"- Failure Cases: {result.coverage_report.failure_cases} ({result.coverage_report.failure_cases/result.coverage_report.total_test_cases:.1%})"
    )
    report.append(f"- **Coverage Score**: {result.coverage_report.coverage_score:.1f}/100\n")

    # Missing scenarios
    if result.coverage_report.missing_scenarios:
        report.append("## Missing Edge Case Scenarios\n")
        for scenario in result.coverage_report.missing_scenarios:
            report.append(f"- {scenario}")
        report.append("")

    # Issues by category
    if result.issues:
        report.append("## Issues by Category\n")

        for category in ["error", "warning", "info"]:
            category_issues = [i for i in result.issues if i.severity == category]
            if category_issues:
                report.append(f"### {category.title()}s ({len(category_issues)})\n")
                for issue in category_issues[:10]:  # Show first 10
                    report.append(f"**{issue.test_case_id}**: {issue.message}")
                    report.append(f"  → Suggestion: {issue.suggestion}\n")

    return "\n".join(report)
