"""
Component-Level Evaluation Framework for Multi-Agent Trading System.

Implements component-level testing alongside end-to-end evaluation. This approach
tests individual subsystems (analysts, traders, risk managers) in isolation before
testing the full integrated system.

Based on DeepEval approach and best practices from 2025 AI agent evaluation research.

References:
- https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide
- https://www.getmaxim.ai/articles/ai-agent-evaluation-metrics-strategies-and-best-practices/
- https://wandb.ai/onlineinference/genai-research/reports/AI-agent-evaluation-Metrics-strategies-and-best-practices--VmlldzoxMjM0NjQzMQ
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ComponentType(Enum):
    """Types of components in the trading system."""

    SIGNAL_GENERATION = "signal_generation"  # Analysts
    POSITION_SIZING = "position_sizing"  # Traders
    RISK_MANAGEMENT = "risk_management"  # Risk Managers
    DECISION_MAKING = "decision_making"  # Supervisor
    TOOL_SELECTION = "tool_selection"  # Multi-tool agents
    CONTEXT_RETENTION = "context_retention"  # Memory systems
    ERROR_RECOVERY = "error_recovery"  # Failure handling


@dataclass
class ComponentTestCase:
    """Test case for individual component."""

    test_id: str
    component_type: ComponentType
    component_name: str
    input_data: dict[str, Any]
    expected_output: dict[str, Any]
    success_criteria: dict[str, Any]
    category: str  # success | edge | failure


@dataclass
class ComponentTestResult:
    """Result of component-level test."""

    test_id: str
    component_name: str
    passed: bool
    execution_time_ms: float
    errors: list[str]
    warnings: list[str]
    actual_output: dict[str, Any]


@dataclass
class ComponentEvaluationResult:
    """Evaluation results for a single component."""

    component_name: str
    component_type: ComponentType
    total_tests: int
    passed_tests: int
    pass_rate: float
    avg_execution_time_ms: float
    error_rate: float
    test_results: list[ComponentTestResult]
    component_ready: bool  # Ready for integration testing


class ComponentEvaluator:
    """
    Evaluates individual components of the trading system.

    This allows testing each subsystem in isolation before integration testing,
    helping identify which specific component is causing issues.
    """

    def __init__(
        self,
        component_name: str,
        component_type: ComponentType,
        component_callable: Callable[[dict[str, Any]], dict[str, Any]],
        test_cases: list[ComponentTestCase],
    ):
        """
        Initialize component evaluator.

        Args:
            component_name: Name of the component (e.g., "TechnicalAnalyst.pattern_recognition")
            component_type: Type of component
            component_callable: Function that executes the component
            test_cases: List of test cases for this component
        """
        self.component_name = component_name
        self.component_type = component_type
        self.component_callable = component_callable
        self.test_cases = test_cases

    def run(self) -> ComponentEvaluationResult:
        """
        Run component evaluation.

        Returns:
            ComponentEvaluationResult with test outcomes
        """
        from time import time

        test_results = []

        for test_case in self.test_cases:
            start_time = time()
            errors = []
            warnings = []
            actual_output = {}

            try:
                # Execute component
                actual_output = self.component_callable(test_case.input_data)

                # Validate against expected output
                passed = self._validate_output(
                    actual_output,
                    test_case.expected_output,
                    test_case.success_criteria,
                    errors,
                )

            except Exception as e:
                errors.append(f"Execution error: {e!s}")
                passed = False

            execution_time = (time() - start_time) * 1000

            test_results.append(
                ComponentTestResult(
                    test_id=test_case.test_id,
                    component_name=self.component_name,
                    passed=passed,
                    execution_time_ms=execution_time,
                    errors=errors,
                    warnings=warnings,
                    actual_output=actual_output,
                )
            )

        # Aggregate results
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.passed)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0

        avg_execution_time = sum(r.execution_time_ms for r in test_results) / total_tests if total_tests > 0 else 0

        error_count = sum(1 for r in test_results if len(r.errors) > 0)
        error_rate = error_count / total_tests if total_tests > 0 else 0

        # Component ready if pass rate > 90% and error rate < 5%
        component_ready = pass_rate > 0.90 and error_rate < 0.05

        return ComponentEvaluationResult(
            component_name=self.component_name,
            component_type=self.component_type,
            total_tests=total_tests,
            passed_tests=passed_tests,
            pass_rate=pass_rate,
            avg_execution_time_ms=avg_execution_time,
            error_rate=error_rate,
            test_results=test_results,
            component_ready=component_ready,
        )

    def _validate_output(
        self,
        actual: dict[str, Any],
        expected: dict[str, Any],
        criteria: dict[str, Any],
        errors: list[str],
    ) -> bool:
        """Validate component output against criteria."""
        all_passed = True

        for key, expected_value in criteria.items():
            if key == "output_present":
                # Check if required keys are present
                for required_key in expected_value:
                    if required_key not in actual:
                        errors.append(f"Missing required output key: {required_key}")
                        all_passed = False

            elif key == "value_match":
                # Check if values match
                for field, expected_val in expected_value.items():
                    actual_val = actual.get(field)
                    if actual_val != expected_val:
                        errors.append(f"Value mismatch for {field}: expected {expected_val}, got {actual_val}")
                        all_passed = False

            elif key == "value_range":
                # Check if values are in acceptable range
                for field, (min_val, max_val) in expected_value.items():
                    actual_val = actual.get(field)
                    if actual_val is None or not (min_val <= actual_val <= max_val):
                        errors.append(f"{field} = {actual_val} outside range [{min_val}, {max_val}]")
                        all_passed = False

        return all_passed


def run_component_suite(
    components: list[ComponentEvaluator],
) -> dict[str, ComponentEvaluationResult]:
    """
    Run evaluation suite for multiple components.

    Args:
        components: List of ComponentEvaluator instances

    Returns:
        Dict mapping component names to evaluation results
    """
    results = {}

    for component in components:
        print(f"Evaluating {component.component_name}...")
        result = component.run()
        results[component.component_name] = result
        print(f"  Pass rate: {result.pass_rate:.1%}, Component ready: {result.component_ready}")

    return results


def generate_component_report(result: ComponentEvaluationResult) -> str:
    """
    Generate component evaluation report.

    Args:
        result: ComponentEvaluationResult object

    Returns:
        Formatted markdown report
    """
    report = []
    report.append(f"# Component Evaluation: {result.component_name}\n")
    report.append(f"**Component Type**: {result.component_type.value}")
    report.append(f"**Component Ready**: {'✅ YES' if result.component_ready else '⚠️ NO'}\n")

    report.append("## Test Results\n")
    report.append(f"- Total Tests: {result.total_tests}")
    report.append(f"- Passed: {result.passed_tests}/{result.total_tests} ({result.pass_rate:.1%})")
    report.append(f"- Error Rate: {result.error_rate:.1%}")
    report.append(f"- Avg Execution Time: {result.avg_execution_time_ms:.1f}ms\n")

    # Failed tests
    failed_tests = [r for r in result.test_results if not r.passed]
    if failed_tests:
        report.append("## Failed Tests\n")
        for test in failed_tests:
            report.append(f"### {test.test_id}")
            for error in test.errors:
                report.append(f"  - ❌ {error}")

    return "\n".join(report)


def create_signal_generation_test_cases() -> list[ComponentTestCase]:
    """
    Create test cases for signal generation component (analysts).

    Returns:
        List of ComponentTestCase objects
    """
    return [
        ComponentTestCase(
            test_id="SIG_001",
            component_type=ComponentType.SIGNAL_GENERATION,
            component_name="TechnicalAnalyst.pattern_recognition",
            input_data={
                "pattern_type": "bull_flag",
                "volume_confirmation": True,
                "in_sample_win_rate": 0.68,
            },
            expected_output={
                "signal": "bullish",
                "confidence": 0.62,
            },
            success_criteria={
                "output_present": ["signal", "confidence"],
                "value_match": {"signal": "bullish"},
                "value_range": {"confidence": (0.58, 0.70)},
            },
            category="success",
        ),
        # Add more test cases...
    ]


def create_position_sizing_test_cases() -> list[ComponentTestCase]:
    """
    Create test cases for position sizing component (traders).

    Returns:
        List of ComponentTestCase objects
    """
    return [
        ComponentTestCase(
            test_id="POS_001",
            component_type=ComponentType.POSITION_SIZING,
            component_name="ConservativeTrader.kelly_calculator",
            input_data={
                "win_rate": 0.68,
                "avg_win": 0.10,
                "avg_loss": 0.12,
                "fractional_kelly": 0.20,
            },
            expected_output={
                "position_size_pct": 0.027,
            },
            success_criteria={
                "output_present": ["position_size_pct"],
                "value_range": {"position_size_pct": (0.024, 0.030)},
            },
            category="success",
        ),
        # Add more test cases...
    ]


def create_risk_management_test_cases() -> list[ComponentTestCase]:
    """
    Create test cases for risk management component.

    Returns:
        List of ComponentTestCase objects
    """
    return [
        ComponentTestCase(
            test_id="RISK_001",
            component_type=ComponentType.RISK_MANAGEMENT,
            component_name="PositionRiskManager.validate_greeks",
            input_data={
                "delta": 45,
                "gamma": 2.1,
                "theta_per_day": -125,
            },
            expected_output={
                "greeks_valid": True,
            },
            success_criteria={
                "output_present": ["greeks_valid"],
                "value_match": {"greeks_valid": True},
            },
            category="success",
        ),
        # Add more test cases...
    ]


def create_tool_selection_test_cases() -> list[ComponentTestCase]:
    """
    Create test cases for tool selection component.

    Returns:
        List of ComponentTestCase objects
    """
    return [
        ComponentTestCase(
            test_id="TOOL_001",
            component_type=ComponentType.TOOL_SELECTION,
            component_name="Supervisor.select_analyst",
            input_data={
                "task": "analyze_earnings",
                "available_analysts": ["TechnicalAnalyst", "SentimentAnalyst"],
            },
            expected_output={
                "selected_analyst": "SentimentAnalyst",
            },
            success_criteria={
                "output_present": ["selected_analyst"],
                "value_match": {"selected_analyst": "SentimentAnalyst"},
            },
            category="success",
        ),
        # Add more test cases...
    ]
