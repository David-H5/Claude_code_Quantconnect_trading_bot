#!/usr/bin/env python3
"""
Example usage of the evaluation framework.

Demonstrates how to:
1. Load test cases for an agent type
2. Create an evaluator with a custom agent callable
3. Run evaluation
4. Generate reports
"""

from typing import Any

from evaluation.datasets import get_technical_analyst_cases
from evaluation.evaluation_framework import AgentEvaluator


def example_technical_analyst(input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Example Technical Analyst agent implementation.

    This is a placeholder that demonstrates the expected interface.
    Replace with your actual LLM-powered agent.

    Args:
        input_data: Test case input data with keys like:
            - symbol: str
            - pattern_type: str
            - in_sample_win_rate: float
            - out_of_sample_win_rate: float
            - volume_confirmation: bool

    Returns:
        Dict with keys:
            - signal: "bullish" | "bearish" | "neutral"
            - confidence: float (0-1)
            - pattern_valid: bool
            - out_of_sample_validated: bool
            - degradation_pct: float (optional)
            - rejection_reason: str (optional)
    """
    # Simple example logic (replace with actual agent)
    in_sample = input_data.get("in_sample_win_rate", 0.5)
    out_sample = input_data.get("out_of_sample_win_rate", 0.5)
    volume_conf = input_data.get("volume_confirmation", False)

    # Calculate degradation
    degradation = ((in_sample - out_sample) / in_sample * 100) if in_sample > 0 else 0

    # Decision logic
    if degradation > 15:
        return {
            "signal": "neutral",
            "confidence": out_sample,
            "pattern_valid": False,
            "out_of_sample_validated": True,
            "degradation_pct": degradation,
            "rejection_reason": "Excessive degradation >15%",
        }

    if not volume_conf:
        return {
            "signal": "neutral",
            "confidence": out_sample * 0.8,
            "pattern_valid": False,
            "out_of_sample_validated": True,
            "rejection_reason": "No volume confirmation",
        }

    # Determine signal from pattern type
    pattern_type = input_data.get("pattern_type", "")
    if "bull" in pattern_type.lower() or "ascending" in pattern_type.lower():
        signal = "bullish"
    elif "bear" in pattern_type.lower() or "head" in pattern_type.lower():
        signal = "bearish"
    else:
        signal = "neutral"

    return {
        "signal": signal,
        "confidence": out_sample,
        "confidence_adjustment": out_sample / in_sample if in_sample > 0 else 1.0,
        "pattern_valid": True,
        "out_of_sample_validated": True,
        "degradation_pct": degradation,
    }


def main():
    """Run example evaluation."""
    print("=" * 60)
    print("Evaluation Framework - Example Usage")
    print("=" * 60)

    # 1. Load test cases
    print("\n1. Loading test cases...")
    test_cases = get_technical_analyst_cases()
    print(f"   Loaded {len(test_cases)} test cases for TechnicalAnalyst")

    # 2. Create evaluator with custom agent callable
    print("\n2. Creating evaluator...")
    evaluator = AgentEvaluator(
        agent_type="TechnicalAnalyst",
        version="v6.1",
        test_cases=test_cases,
        agent_callable=example_technical_analyst,  # Your agent here
    )

    # 3. Run evaluation
    print("\n3. Running evaluation...")
    result = evaluator.run()

    # 4. Print results
    print("\n4. Results:")
    print(f"   Total cases: {result.total_cases}")
    print(f"   Passed: {result.passed_cases}")
    print(f"   Failed: {result.failed_cases}")
    print(f"   Pass rate: {result.pass_rate:.1%}")
    print(f"   Duration: {result.duration_seconds:.1f}s")
    print(f"   Production ready: {'✅ YES' if result.production_ready else '⚠️ NO'}")

    # 5. Category breakdown
    print("\n5. Category breakdown:")
    for category, stats in result.category_breakdown.items():
        print(f"   {category.title()}: {stats['passed']}/{stats['total']} " f"({stats['pass_rate']:.1%})")

    # 6. Metrics
    print("\n6. Metrics:")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.1%}")
        else:
            print(f"   {key}: {value}")

    # 7. Generate markdown report
    print("\n7. Generating report...")
    report = evaluator.generate_report()
    print(report)

    # 8. Save results (optional)
    from pathlib import Path

    output_dir = Path(__file__).parent / "results"
    result_file = evaluator.save_results(output_dir)
    print(f"\n8. Results saved to: {result_file}")


if __name__ == "__main__":
    main()
