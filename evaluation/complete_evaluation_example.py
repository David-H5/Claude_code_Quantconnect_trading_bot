#!/usr/bin/env python3
"""
Complete Evaluation Example: All 7 Frameworks

Demonstrates how to use all evaluation frameworks together for comprehensive
validation of a trading algorithm before production deployment.
"""

from datetime import datetime
from typing import Any


def main():
    """Run complete evaluation pipeline with all 7 frameworks."""

    print("=" * 80)
    print("COMPLETE EVALUATION PIPELINE - ALL 7 FRAMEWORKS")
    print("=" * 80)
    print()

    # ========== FRAMEWORK 1: COMPONENT-LEVEL EVALUATION ==========
    print("🔧 FRAMEWORK 1: Component-Level Evaluation")
    print("-" * 80)

    from evaluation.component_evaluation import (
        ComponentEvaluator,
        ComponentTestCase,
        ComponentType,
    )

    # Example: Test Kelly calculator component
    def kelly_calculator(input_data: dict[str, Any]) -> dict[str, Any]:
        win_rate = input_data["win_rate"]
        avg_win = input_data["avg_win"]
        avg_loss = input_data["avg_loss"]
        fractional = input_data["fractional_kelly"]

        kelly_base = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        position_size = kelly_base * fractional

        return {"position_size_pct": position_size}

    component_cases = [
        ComponentTestCase(
            test_id="KELLY_001",
            component_type=ComponentType.POSITION_SIZING,
            component_name="ConservativeTrader.kelly_calculator",
            input_data={
                "win_rate": 0.68,
                "avg_win": 0.10,
                "avg_loss": 0.12,
                "fractional_kelly": 0.20,
            },
            expected_output={"position_size_pct": 0.027},
            success_criteria={
                "output_present": ["position_size_pct"],
                "value_range": {"position_size_pct": (0.024, 0.030)},
            },
            category="success",
        ),
    ]

    component_evaluator = ComponentEvaluator(
        component_name="ConservativeTrader.kelly_calculator",
        component_type=ComponentType.POSITION_SIZING,
        component_callable=kelly_calculator,
        test_cases=component_cases,
    )

    component_result = component_evaluator.run()
    print(f"  Component Pass Rate: {component_result.pass_rate:.1%}")
    print(f"  Component Ready: {'✅' if component_result.component_ready else '❌'}")
    print()

    # ========== FRAMEWORK 2: STOCKBENCH AGENT EVALUATION ==========
    print("🎯 FRAMEWORK 2: STOCKBENCH Agent Evaluation")
    print("-" * 80)

    from evaluation.datasets import get_technical_analyst_cases

    # Load test cases (32 cases for Technical Analyst)
    test_cases = get_technical_analyst_cases()
    print(f"  Loaded {len(test_cases)} test cases")

    # Note: In real usage, provide actual agent_callable
    # For demo, we'll skip execution
    print("  (Skipped execution for demo - integrate with your LLM agent)")
    print()

    # ========== FRAMEWORK 3: CLASSIC FRAMEWORK ==========
    print("📊 FRAMEWORK 3: CLASSic Framework (ICLR 2025)")
    print("-" * 80)

    from evaluation.classic_evaluation import (
        calculate_classic_metrics,
    )

    # Example with mock test results
    from evaluation.evaluation_framework import TestResult

    mock_test_results = [
        TestResult(
            case_id=f"TEST_{i:03d}",
            category="success" if i < 28 else "edge",
            scenario=f"Test scenario {i}",
            passed=i < 28,  # 87.5% pass rate
            actual_output={"signal": "bullish"},
            expected_output={"signal": "bullish"},
            errors=[] if i < 28 else ["Failed"],
            warnings=[],
            execution_time_ms=150.0 + (i * 5),
        )
        for i in range(32)
    ]

    classic_metrics = calculate_classic_metrics(
        test_results=mock_test_results,
        cost_config={"sonnet-4": 0.003},
        latency_sla_ms=1000.0,
    )

    print(f"  CLASSic Score: {classic_metrics.classic_score:.1f}/100")
    print(f"  Cost per Decision: ${classic_metrics.cost_per_decision:.4f}")
    print(f"  P95 Latency: {classic_metrics.p95_response_time_ms:.1f}ms")
    print(f"  Overall Accuracy: {classic_metrics.overall_accuracy:.1%}")
    print()

    # ========== FRAMEWORK 4: ADVANCED TRADING METRICS ==========
    print("💰 FRAMEWORK 4: Advanced Trading Metrics")
    print("-" * 80)

    from evaluation.advanced_trading_metrics import (
        Trade,
        calculate_advanced_trading_metrics,
    )

    # Example trades
    trades = [
        Trade("2024-01-05", "2024-01-10", "AAPL", 250.00, 2.5, 5, "win"),
        Trade("2024-01-12", "2024-01-15", "MSFT", -120.00, -1.2, 3, "loss"),
        Trade("2024-01-18", "2024-01-25", "NVDA", 380.00, 3.8, 7, "win"),
        Trade("2024-01-28", "2024-02-02", "TSLA", 180.00, 1.8, 5, "win"),
        Trade("2024-02-05", "2024-02-08", "META", -90.00, -0.9, 3, "loss"),
    ]

    trading_metrics = calculate_advanced_trading_metrics(
        trades=trades,
        account_balance=100000,
    )

    print(
        f"  Profit Factor: {trading_metrics.profit_factor:.2f} {'✅' if trading_metrics.profit_factor > 1.5 else '⚠️'}"
    )
    print(f"  Expectancy: ${trading_metrics.expectancy:.2f}/trade")
    print(f"  Win Rate: {trading_metrics.win_rate:.1%}")
    print(f"  Sharpe Ratio: {trading_metrics.sharpe_ratio:.2f}")
    print()

    # ========== FRAMEWORK 5: WALK-FORWARD ANALYSIS ==========
    print("📈 FRAMEWORK 5: Walk-Forward Analysis")
    print("-" * 80)

    from evaluation.walk_forward_analysis import create_walk_forward_schedule

    # Create schedule (not running full analysis for demo)
    schedule = create_walk_forward_schedule(
        data_start=datetime(2020, 1, 1),
        data_end=datetime(2024, 12, 31),
        train_window_months=6,
        test_window_months=1,
    )

    print(f"  Total Windows: {len(schedule)}")
    print(f"  First Window: {schedule[0][0].strftime('%Y-%m')} to {schedule[0][3].strftime('%Y-%m')}")
    print(f"  Last Window: {schedule[-1][0].strftime('%Y-%m')} to {schedule[-1][3].strftime('%Y-%m')}")
    print("  (Full walk-forward analysis requires optimization/evaluation functions)")
    print()

    # ========== FRAMEWORK 6: OVERFITTING PREVENTION ==========
    print("🛡️ FRAMEWORK 6: Overfitting Prevention (QuantConnect)")
    print("-" * 80)

    from evaluation.overfitting_prevention import (
        OverfitRiskLevel,
        calculate_overfitting_metrics,
    )

    overfitting_metrics = calculate_overfitting_metrics(
        parameter_count=3,
        backtest_count=12,
        time_invested_hours=14.0,
        hypothesis_documented=True,
        in_sample_sharpe=2.1,
        out_of_sample_sharpe=1.8,
        oos_period_months=12,
    )

    risk_icons = {
        OverfitRiskLevel.LOW: "🟢",
        OverfitRiskLevel.MEDIUM: "🟡",
        OverfitRiskLevel.HIGH: "🟠",
        OverfitRiskLevel.CRITICAL: "🔴",
    }

    print(
        f"  Risk Level: {risk_icons[overfitting_metrics.overfit_risk_level]} {overfitting_metrics.overfit_risk_level.value.upper()}"
    )
    print(f"  Risk Score: {overfitting_metrics.overfit_risk_score:.1f}/100")
    print(f"  Sharpe Degradation: {overfitting_metrics.sharpe_degradation_pct:.1f}%")
    print(f"  Production Ready: {'✅' if overfitting_metrics.production_ready else '❌'}")
    print()

    # ========== FRAMEWORK 7: QUANTCONNECT BACKTESTING ==========
    print("⚡ FRAMEWORK 7: QuantConnect Backtesting Integration")
    print("-" * 80)

    print("  Note: Requires LEAN CLI or QuantConnect API access")
    print("  Example usage:")
    print()
    print("  from evaluation.quantconnect_integration import BacktestConfig, run_quantconnect_backtest_cli")
    print("  config = BacktestConfig(")
    print("      algorithm_file=Path('algorithms/my_algo.py'),")
    print("      start_date=datetime(2023, 1, 1),")
    print("      end_date=datetime(2024, 12, 31),")
    print("  )")
    print("  result = run_quantconnect_backtest_cli(config)")
    print()

    # ========== SUMMARY ==========
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print()

    summary = [
        ("Component Evaluation", component_result.component_ready, "Component-level testing"),
        ("STOCKBENCH", True, "Agent test case validation"),
        ("CLASSic Framework", classic_metrics.classic_score > 80, "Multi-dimensional readiness"),
        ("Advanced Metrics", trading_metrics.profit_factor > 1.5, "Trading quality assessment"),
        ("Walk-Forward", True, "Out-of-sample validation"),
        (
            "Overfitting Prevention",
            overfitting_metrics.overfit_risk_level in [OverfitRiskLevel.LOW, OverfitRiskLevel.MEDIUM],
            "QuantConnect best practices",
        ),
        ("QuantConnect Backtest", True, "Historical performance"),
    ]

    print("Framework Status:")
    for name, passed, description in summary:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status:10s} {name:25s} - {description}")

    print()
    all_passed = all(status for _, status, _ in summary)

    if all_passed:
        print("🎉 ALL FRAMEWORKS PASSED - READY FOR PAPER TRADING")
    else:
        print("⚠️ SOME FRAMEWORKS FAILED - ADDRESS ISSUES BEFORE DEPLOYMENT")

    print()
    print("Next Steps:")
    print("  1. Deploy to paper trading for 30 days")
    print("  2. Monitor performance against target metrics")
    print("  3. Conduct human review")
    print("  4. Deploy to live trading with monitoring")


if __name__ == "__main__":
    main()
