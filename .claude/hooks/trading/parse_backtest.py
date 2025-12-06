#!/usr/bin/env python3
"""
Backtest Parser PostToolUse Hook

Parses backtest results and extracts key metrics.
Provides formatted summary and validates against targets.

UPGRADE-015 Phase 4: Hook System Implementation

Usage:
    Called as PostToolUse hook after backtest tool calls.
    Parses results and validates against performance targets.

Metrics Validated:
    - Sharpe ratio (target: > 1.0)
    - Max drawdown (target: < 20%)
    - Win rate (target: > 45%)
    - Profit factor (target: > 1.2)
"""

import json
import sys


# Performance targets
TARGETS = {
    "sharpe_ratio": {"min": 1.0, "description": "Sharpe Ratio"},
    "max_drawdown": {"max": -0.20, "description": "Max Drawdown"},
    "win_rate": {"min": 0.45, "description": "Win Rate"},
    "profit_factor": {"min": 1.2, "description": "Profit Factor"},
    "total_return": {"min": 0.0, "description": "Total Return"},
}


def validate_metrics(metrics: dict) -> list:
    """Validate metrics against targets."""
    issues = []
    successes = []

    for metric_name, target in TARGETS.items():
        if metric_name not in metrics:
            continue

        value = metrics[metric_name]

        if "min" in target:
            if value < target["min"]:
                issues.append(f"❌ {target['description']}: {value:.2f} (target: ≥{target['min']})")
            else:
                successes.append(f"✓ {target['description']}: {value:.2f}")

        if "max" in target:
            if value > target["max"]:
                issues.append(f"❌ {target['description']}: {value:.2%} (target: ≤{target['max']:.0%})")
            else:
                successes.append(f"✓ {target['description']}: {value:.2%}")

    return issues, successes


def format_backtest_summary(results: dict) -> str:
    """Format backtest results summary."""
    lines = [
        "=" * 50,
        "BACKTEST RESULTS SUMMARY",
        "=" * 50,
    ]

    # Basic info
    if "backtest_id" in results:
        lines.append(f"Backtest ID: {results['backtest_id']}")
    if "algorithm_id" in results:
        lines.append(f"Algorithm: {results['algorithm_id']}")
    if "start_date" in results and "end_date" in results:
        lines.append(f"Period: {results['start_date']} to {results['end_date']}")

    # Financial summary
    if "initial_cash" in results and "final_value" in results:
        initial = results["initial_cash"]
        final = results["final_value"]
        pnl = final - initial
        pnl_pct = (final / initial - 1) * 100
        lines.append("")
        lines.append(f"Initial: ${initial:,.0f}")
        lines.append(f"Final:   ${final:,.0f}")
        lines.append(f"P&L:     ${pnl:+,.0f} ({pnl_pct:+.1f}%)")

    # Metrics
    if "metrics" in results:
        metrics = results["metrics"]
        lines.append("")
        lines.append("KEY METRICS:")

        metric_display = [
            ("sharpe_ratio", "Sharpe Ratio", "{:.2f}"),
            ("sortino_ratio", "Sortino Ratio", "{:.2f}"),
            ("max_drawdown", "Max Drawdown", "{:.1%}"),
            ("win_rate", "Win Rate", "{:.1%}"),
            ("profit_factor", "Profit Factor", "{:.2f}"),
            ("total_trades", "Total Trades", "{:,}"),
        ]

        for key, label, fmt in metric_display:
            if key in metrics:
                value = metrics[key]
                formatted = fmt.format(value)
                lines.append(f"  {label}: {formatted}")

        # Validate against targets
        issues, successes = validate_metrics(metrics)

        if successes:
            lines.append("")
            lines.append("TARGETS MET:")
            for s in successes:
                lines.append(f"  {s}")

        if issues:
            lines.append("")
            lines.append("⚠️ TARGETS MISSED:")
            for issue in issues:
                lines.append(f"  {issue}")

    lines.append("=" * 50)
    return "\n".join(lines)


def main():
    """Main entry point for hook."""
    # Read tool context from stdin
    try:
        input_data = sys.stdin.read()
        if not input_data:
            sys.exit(0)

        context = json.loads(input_data)
    except json.JSONDecodeError:
        sys.exit(0)

    tool_name = context.get("tool_name", "")
    tool_output = context.get("tool_output")

    # Only process backtest tools
    backtest_tools = ["run_backtest", "get_backtest_results"]

    if tool_name not in backtest_tools:
        sys.exit(0)

    if not tool_output:
        sys.exit(0)

    # Handle nested data structures
    if isinstance(tool_output, dict):
        # Check if this is a successful result with metrics
        if "metrics" in tool_output or "final_value" in tool_output:
            try:
                summary = format_backtest_summary(tool_output)
                print(summary, file=sys.stderr)
            except Exception as e:
                print(f"Backtest parser error: {e}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
