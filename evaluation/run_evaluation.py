#!/usr/bin/env python3
"""
Main evaluation runner for autonomous AI agent trading system.

Runs STOCKBENCH-inspired evaluation against all 9 agents with contamination-free
2024-2025 market data.

Usage:
    # Run evaluation for all agents
    python evaluation/run_evaluation.py --all

    # Run evaluation for specific agent type
    python evaluation/run_evaluation.py --agent TechnicalAnalyst

    # Run evaluation with specific version
    python evaluation/run_evaluation.py --agent ConservativeTrader --version v6.1

    # Generate only the report (skip execution)
    python evaluation/run_evaluation.py --report-only --results results/latest_results.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.datasets import (
    get_aggressive_trader_cases,
    get_circuit_breaker_manager_cases,
    get_conservative_trader_cases,
    get_moderate_trader_cases,
    get_portfolio_risk_manager_cases,
    get_position_risk_manager_cases,
    get_sentiment_analyst_cases,
    get_technical_analyst_cases,
)
from evaluation.evaluation_framework import AgentEvaluator, EvaluationResult


AGENT_TYPES = {
    "Supervisor": None,  # No test cases yet
    "TechnicalAnalyst": get_technical_analyst_cases,
    "SentimentAnalyst": get_sentiment_analyst_cases,
    "ConservativeTrader": get_conservative_trader_cases,
    "ModerateTrader": get_moderate_trader_cases,
    "AggressiveTrader": get_aggressive_trader_cases,
    "PositionRiskManager": get_position_risk_manager_cases,
    "PortfolioRiskManager": get_portfolio_risk_manager_cases,
    "CircuitBreakerManager": get_circuit_breaker_manager_cases,
}


def run_agent_evaluation(
    agent_type: str,
    version: str = "v6.1",
    agent_callable: Any | None = None,
) -> EvaluationResult:
    """
    Run evaluation for a single agent type.

    Args:
        agent_type: Type of agent to evaluate
        version: Agent version (v6.0, v6.1, etc.)
        agent_callable: Optional callable that executes the agent

    Returns:
        EvaluationResult with complete metrics
    """
    # Get test cases for this agent type
    test_case_loader = AGENT_TYPES.get(agent_type)
    if test_case_loader is None:
        raise ValueError(f"No test cases available for agent type: {agent_type}")

    test_cases = test_case_loader()

    print(f"\n{'='*60}")
    print(f"Evaluating {agent_type} {version}")
    print(f"{'='*60}")
    print(f"Total test cases: {len(test_cases)}")

    # Create evaluator
    evaluator = AgentEvaluator(
        agent_type=agent_type,
        version=version,
        test_cases=test_cases,
        agent_callable=agent_callable,
    )

    # Run evaluation
    result = evaluator.run()

    # Print summary
    print("\nResults:")
    print(f"  Pass rate: {result.pass_rate:.1%}")
    print(f"  Passed: {result.passed_cases}/{result.total_cases}")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    print(f"  Production ready: {'✅ YES' if result.production_ready else '⚠️ NO'}")

    return result


def run_full_evaluation(
    version: str = "v6.1",
    threshold: float = 0.90,
    output_dir: Path | None = None,
) -> bool:
    """
    Run evaluation for all agents.

    Args:
        version: Agent version to evaluate
        threshold: Minimum pass rate threshold (default 0.90)
        output_dir: Optional directory to save results

    Returns:
        True if all agents pass threshold, False otherwise
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("FULL EVALUATION: All 9 Trading Agents")
    print("=" * 80)
    print(f"Version: {version}")
    print(f"Threshold: {threshold:.1%}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}
    all_passed = True

    # Run evaluation for each agent type (except Supervisor - no test cases yet)
    for agent_type, test_case_loader in AGENT_TYPES.items():
        if test_case_loader is None:
            print(f"\nSkipping {agent_type} (no test cases available)")
            continue

        try:
            result = run_agent_evaluation(agent_type, version)
            all_results[agent_type] = result

            # Check if passed threshold
            if result.pass_rate < threshold:
                all_passed = False
                print(f"  ❌ FAILED: Pass rate {result.pass_rate:.1%} < {threshold:.1%}")
            else:
                print(f"  ✅ PASSED: Pass rate {result.pass_rate:.1%} >= {threshold:.1%}")

            # Save individual results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = output_dir / f"{timestamp}_{agent_type}_{version}_results.json"

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
            }

            with open(result_file, "w") as f:
                json.dump(result_dict, f, indent=2)

            print(f"  Results saved: {result_file}")

        except Exception as e:
            print(f"  ❌ ERROR: {e!s}")
            all_passed = False

    # Generate summary report
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    total_agents = len([a for a in AGENT_TYPES.values() if a is not None])
    passed_agents = sum(1 for r in all_results.values() if r.pass_rate >= threshold)

    print(f"Agents evaluated: {total_agents}")
    print(f"Agents passed: {passed_agents}/{total_agents}")
    print(f"Overall pass rate: {passed_agents/total_agents:.1%}")

    if all_passed:
        print("\n✅ ALL AGENTS PASSED - READY FOR PAPER TRADING")
        return True
    else:
        print("\n⚠️ SOME AGENTS FAILED - FIX BEFORE DEPLOYMENT")
        return False


def generate_markdown_report(results_file: Path) -> str:
    """
    Generate markdown report from saved results.

    Args:
        results_file: Path to saved results JSON file

    Returns:
        Markdown formatted report
    """
    with open(results_file) as f:
        data = json.load(f)

    report = []
    report.append("# Evaluation Report\n")
    report.append(f"**Agent**: {data['agent_type']} {data['version']}\n")
    report.append(f"**Date**: {data['timestamp']}\n")
    report.append(f"**Duration**: {data['duration_seconds']:.1f} seconds\n")
    report.append("\n## Summary\n")
    report.append(f"- Total cases: {data['total_cases']}")
    report.append(f"- Passed: {data['passed_cases']}")
    report.append(f"- Failed: {data['failed_cases']}")
    report.append(f"- Pass rate: **{data['pass_rate']:.1%}**")
    report.append(f"- Production ready: **{'✅ YES' if data['production_ready'] else '⚠️ NO'}**\n")

    report.append("\n## Category Breakdown\n")
    for category, stats in data["category_breakdown"].items():
        report.append(f"- **{category.title()}**: {stats['passed']}/{stats['total']} " f"({stats['pass_rate']:.1%})")

    report.append("\n## Metrics\n")
    for key, value in data["metrics"].items():
        if isinstance(value, float):
            report.append(f"- {key.replace('_', ' ').title()}: {value:.1%}")
        else:
            report.append(f"- {key.replace('_', ' ').title()}: {value}")

    return "\n".join(report)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run evaluation for autonomous AI trading agents")

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run evaluation for all agents",
    )

    parser.add_argument(
        "--agent",
        type=str,
        choices=list(AGENT_TYPES.keys()),
        help="Agent type to evaluate",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v6.1",
        help="Agent version to evaluate (default: v6.1)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Minimum pass rate threshold (default: 0.90)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results (default: evaluation/results)",
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report from existing results without running evaluation",
    )

    parser.add_argument(
        "--results",
        type=Path,
        help="Path to results file for --report-only mode",
    )

    args = parser.parse_args()

    # Report-only mode
    if args.report_only:
        if not args.results:
            print("ERROR: --results required for --report-only mode")
            sys.exit(1)

        report = generate_markdown_report(args.results)
        print(report)
        sys.exit(0)

    # Run evaluation
    if args.all:
        success = run_full_evaluation(
            version=args.version,
            threshold=args.threshold,
            output_dir=args.output_dir,
        )
        sys.exit(0 if success else 1)

    elif args.agent:
        result = run_agent_evaluation(args.agent, args.version)

        # Save results
        output_dir = args.output_dir or (Path(__file__).parent / "results")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"{timestamp}_{args.agent}_{args.version}_results.json"

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
        }

        with open(result_file, "w") as f:
            json.dump(result_dict, f, indent=2)

        print(f"\nResults saved: {result_file}")

        # Generate and print report
        report = generate_markdown_report(result_file)
        print("\n" + report)

        sys.exit(0 if result.production_ready else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
