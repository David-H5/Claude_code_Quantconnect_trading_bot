"""
Core evaluation engine for autonomous AI agent testing.

Implements STOCKBENCH-inspired methodology with contamination-free testing
using 2024-2025 market data.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TestCase:
    """Single test case for agent evaluation."""

    case_id: str
    category: str  # success | edge | failure
    agent_type: str
    scenario: str
    input_data: dict[str, Any]
    expected_output: dict[str, Any]
    success_criteria: dict[str, Any]


@dataclass
class TestResult:
    """Result of evaluating a single test case."""

    case_id: str
    category: str
    scenario: str
    passed: bool
    actual_output: dict[str, Any]
    expected_output: dict[str, Any]
    errors: list[str]
    warnings: list[str]
    execution_time_ms: float


@dataclass
class EvaluationResult:
    """Complete evaluation results for an agent."""

    agent_type: str
    version: str
    timestamp: datetime
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    category_breakdown: dict[str, dict[str, int]]
    test_results: list[TestResult]
    metrics: dict[str, Any]
    duration_seconds: float
    production_ready: bool


class AgentEvaluator:
    """
    Evaluates trading agents against test case datasets.

    Implements STOCKBENCH methodology with 30+ test cases per agent type
    covering success scenarios, edge cases, and failure modes.
    """

    def __init__(
        self,
        agent_type: str,
        version: str,
        test_cases: list[TestCase],
        agent_callable: Any | None = None,
    ):
        """
        Initialize agent evaluator.

        Args:
            agent_type: Type of agent (TechnicalAnalyst, ConservativeTrader, etc.)
            version: Agent version (v6.0, v6.1, etc.)
            test_cases: List of test cases to evaluate
            agent_callable: Callable that executes the agent (optional)
        """
        self.agent_type = agent_type
        self.version = version
        self.test_cases = test_cases
        self.agent_callable = agent_callable
        self.results: list[TestResult] = []

    def run(self) -> EvaluationResult:
        """
        Run evaluation against all test cases.

        Returns:
            EvaluationResult with complete evaluation metrics
        """
        from time import time

        start_time = time()
        self.results = []

        for test_case in self.test_cases:
            result = self._evaluate_case(test_case)
            self.results.append(result)

        duration = time() - start_time

        return self._compile_results(duration)

    def _evaluate_case(self, test_case: TestCase) -> TestResult:
        """
        Evaluate a single test case.

        Args:
            test_case: Test case to evaluate

        Returns:
            TestResult for this case
        """
        from time import time

        start_time = time()
        errors = []
        warnings = []
        actual_output = {}

        try:
            if self.agent_callable:
                # Execute agent with input data
                actual_output = self.agent_callable(test_case.input_data)
            else:
                # Placeholder for manual testing
                actual_output = test_case.expected_output
                warnings.append("No agent callable provided - using expected output")

            # Validate against success criteria
            passed = self._check_success_criteria(
                actual_output,
                test_case.expected_output,
                test_case.success_criteria,
                errors,
            )

        except Exception as e:
            errors.append(f"Execution error: {e!s}")
            passed = False

        execution_time = (time() - start_time) * 1000  # Convert to ms

        return TestResult(
            case_id=test_case.case_id,
            category=test_case.category,
            scenario=test_case.scenario,
            passed=passed,
            actual_output=actual_output,
            expected_output=test_case.expected_output,
            errors=errors,
            warnings=warnings,
            execution_time_ms=execution_time,
        )

    def _check_success_criteria(
        self,
        actual: dict[str, Any],
        expected: dict[str, Any],
        criteria: dict[str, Any],
        errors: list[str],
    ) -> bool:
        """
        Check if actual output meets success criteria.

        Args:
            actual: Actual agent output
            expected: Expected output
            criteria: Success criteria to check
            errors: List to append error messages

        Returns:
            True if all criteria pass
        """
        all_passed = True

        for criterion_name, criterion_value in criteria.items():
            try:
                if criterion_name == "signal_correct":
                    if actual.get("signal") != expected.get("signal"):
                        errors.append(
                            f"Signal mismatch: expected {expected.get('signal')}, " f"got {actual.get('signal')}"
                        )
                        all_passed = False

                elif criterion_name == "confidence_within_range":
                    min_conf, max_conf = criterion_value
                    actual_conf = actual.get("confidence", 0)
                    if not (min_conf <= actual_conf <= max_conf):
                        errors.append(
                            f"Confidence {actual_conf:.2f} outside range " f"[{min_conf:.2f}, {max_conf:.2f}]"
                        )
                        all_passed = False

                elif criterion_name == "out_of_sample_check":
                    if criterion_value and not actual.get("out_of_sample_validated"):
                        errors.append("Out-of-sample validation not performed")
                        all_passed = False

                elif criterion_name == "degradation_under_15":
                    degradation = actual.get("degradation_pct", 0)
                    if degradation > 15:
                        errors.append(f"Degradation {degradation:.1f}% exceeds 15% threshold")
                        all_passed = False

                elif criterion_name == "veto_triggered":
                    if actual.get("veto_triggered") != criterion_value:
                        errors.append(
                            f"Veto status mismatch: expected {criterion_value}, " f"got {actual.get('veto_triggered')}"
                        )
                        all_passed = False

                elif criterion_name == "position_size_within_range":
                    min_size, max_size = criterion_value
                    actual_size = actual.get("position_size_pct", 0)
                    if not (min_size <= actual_size <= max_size):
                        errors.append(
                            f"Position size {actual_size:.2%} outside range " f"[{min_size:.2%}, {max_size:.2%}]"
                        )
                        all_passed = False

            except Exception as e:
                errors.append(f"Criterion check error ({criterion_name}): {e!s}")
                all_passed = False

        return all_passed

    def _compile_results(self, duration: float) -> EvaluationResult:
        """
        Compile evaluation results and calculate metrics.

        Args:
            duration: Total evaluation duration in seconds

        Returns:
            EvaluationResult with complete metrics
        """
        from evaluation.metrics import calculate_agent_metrics

        total_cases = len(self.results)
        passed_cases = sum(1 for r in self.results if r.passed)
        failed_cases = total_cases - passed_cases
        pass_rate = passed_cases / total_cases if total_cases > 0 else 0

        # Category breakdown
        category_breakdown = {}
        for category in ["success", "edge", "failure"]:
            category_results = [r for r in self.results if r.category == category]
            category_passed = sum(1 for r in category_results if r.passed)
            category_total = len(category_results)
            category_breakdown[category] = {
                "total": category_total,
                "passed": category_passed,
                "pass_rate": category_passed / category_total if category_total > 0 else 0,
            }

        # Calculate agent-specific metrics
        metrics = calculate_agent_metrics(self.agent_type, self.results)

        # Determine production readiness
        production_ready = self._check_production_ready(pass_rate, category_breakdown, metrics)

        return EvaluationResult(
            agent_type=self.agent_type,
            version=self.version,
            timestamp=datetime.now(),
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            pass_rate=pass_rate,
            category_breakdown=category_breakdown,
            test_results=self.results,
            metrics=metrics,
            duration_seconds=duration,
            production_ready=production_ready,
        )

    def _check_production_ready(
        self,
        pass_rate: float,
        category_breakdown: dict[str, dict[str, int]],
        metrics: dict[str, Any],
    ) -> bool:
        """
        Check if agent meets production deployment criteria.

        Args:
            pass_rate: Overall pass rate
            category_breakdown: Pass rates by category
            metrics: Agent-specific metrics

        Returns:
            True if production-ready
        """
        # Phase 1: Evaluation Dataset criteria
        if pass_rate < 0.90:
            return False

        if category_breakdown.get("success", {}).get("pass_rate", 0) < 0.90:
            return False

        if category_breakdown.get("edge", {}).get("pass_rate", 0) < 0.80:
            return False

        if category_breakdown.get("failure", {}).get("pass_rate", 0) < 0.95:
            return False

        # Agent-specific criteria
        if self.agent_type in ["TechnicalAnalyst", "SentimentAnalyst"]:
            if metrics.get("accuracy", 0) < 0.60:
                return False
            if metrics.get("false_positive_rate", 1) > 0.30:
                return False

        elif self.agent_type in ["ConservativeTrader", "ModerateTrader", "AggressiveTrader"]:
            min_win_rate = {"ConservativeTrader": 0.65, "ModerateTrader": 0.60, "AggressiveTrader": 0.55}
            if metrics.get("win_rate", 0) < min_win_rate.get(self.agent_type, 0.60):
                return False

        elif self.agent_type in ["PositionRiskManager", "PortfolioRiskManager", "CircuitBreakerManager"]:
            if metrics.get("zero_violations_rate", 0) < 0.95:
                return False

        return True

    def save_results(self, output_dir: Path) -> Path:
        """
        Save evaluation results to JSON file.

        Args:
            output_dir: Directory to save results

        Returns:
            Path to saved results file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{self.agent_type}_{self.version}_results.json"
        filepath = output_dir / filename

        # Convert results to dict
        result = self.run()
        result_dict = {
            "agent_type": result.agent_type,
            "version": result.version,
            "timestamp": result.timestamp.isoformat(),
            "total_cases": result.total_cases,
            "passed_cases": result.passed_cases,
            "failed_cases": result.failed_cases,
            "pass_rate": result.pass_rate,
            "category_breakdown": result.category_breakdown,
            "metrics": result.metrics,
            "duration_seconds": result.duration_seconds,
            "production_ready": result.production_ready,
            "test_results": [
                {
                    "case_id": r.case_id,
                    "category": r.category,
                    "scenario": r.scenario,
                    "passed": r.passed,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "execution_time_ms": r.execution_time_ms,
                }
                for r in result.test_results
            ],
        }

        with open(filepath, "w") as f:
            json.dump(result_dict, f, indent=2)

        return filepath

    def generate_report(self) -> str:
        """
        Generate markdown evaluation report.

        Returns:
            Formatted markdown report
        """
        result = self.run()

        report = []
        report.append("=" * 60)
        report.append(f"EVALUATION REPORT: {result.agent_type} {result.version}")
        report.append("=" * 60)
        report.append(f"Date: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Cases: {result.total_cases}")
        report.append(f"Duration: {result.duration_seconds:.1f} seconds")
        report.append("")

        report.append("CATEGORY BREAKDOWN:")
        for category in ["success", "edge", "failure"]:
            cat_data = result.category_breakdown.get(category, {})
            passed = cat_data.get("passed", 0)
            total = cat_data.get("total", 0)
            pass_rate = cat_data.get("pass_rate", 0)
            status = "✅" if pass_rate >= 0.80 else "⚠️"
            report.append(f"  {category.title()} Cases ({passed}/{total}): {pass_rate:.1%} {status}")

        report.append("")
        overall_status = "✅" if result.pass_rate >= 0.90 else "⚠️"
        report.append(f"OVERALL ACCURACY: {result.pass_rate:.1%} {overall_status} (Target: >90%)")

        report.append("")
        report.append("METRICS:")
        for key, value in result.metrics.items():
            if isinstance(value, float):
                report.append(f"  {key.replace('_', ' ').title()}: {value:.1%}")
            else:
                report.append(f"  {key.replace('_', ' ').title()}: {value}")

        report.append("")
        report.append("DETAILED RESULTS:")
        for test_result in result.test_results:
            status_icon = "✅" if test_result.passed else "❌"
            report.append(f"  {status_icon} {test_result.case_id}: {test_result.scenario}")
            if test_result.errors:
                for error in test_result.errors:
                    report.append(f"      ERROR: {error}")

        report.append("")
        prod_status = "✅ PRODUCTION-READY" if result.production_ready else "⚠️ NOT PRODUCTION-READY"
        report.append(f"PASS STATUS: {prod_status}")

        return "\n".join(report)
