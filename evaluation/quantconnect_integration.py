"""
QuantConnect Backtesting Integration for Evaluation Framework.

Integrates with QuantConnect's LEAN engine to run backtests and extract
performance metrics for evaluation. Supports both local LEAN CLI and cloud API.

References:
- https://www.quantconnect.com/docs/v2/cloud-platform/backtesting
- https://www.quantconnect.com/docs/v2/cloud-platform/backtesting/results
- https://algotrading101.com/learn/quantconnect-guide/
"""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class BacktestConfig:
    """Configuration for QuantConnect backtest."""

    algorithm_file: Path
    start_date: datetime
    end_date: datetime
    initial_cash: float = 100000
    data_resolution: str = "Minute"  # Tick | Second | Minute | Hour | Daily
    brokerage: str = "CharlesSchwab"
    account_type: str = "Margin"


@dataclass
class BacktestResult:
    """Results from QuantConnect backtest."""

    # Basic info
    backtest_id: str
    algorithm_name: str
    start_date: datetime
    end_date: datetime
    duration_days: int

    # Performance metrics
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    cagr: float

    # Trading activity
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Additional metrics
    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float

    # Execution quality
    avg_fill_time_ms: float
    slippage_total: float
    commission_total: float

    # Raw JSON result
    raw_result: dict[str, Any]


def run_quantconnect_backtest_cli(
    config: BacktestConfig,
    lean_cli_path: str = "lean",
) -> BacktestResult:
    """
    Run backtest using local LEAN CLI.

    Args:
        config: BacktestConfig with algorithm and parameters
        lean_cli_path: Path to LEAN CLI executable (default: "lean" in PATH)

    Returns:
        BacktestResult with performance metrics
    """
    # Build LEAN CLI command
    cmd = [
        lean_cli_path,
        "backtest",
        str(config.algorithm_file),
        "--start",
        config.start_date.strftime("%Y%m%d"),
        "--end",
        config.end_date.strftime("%Y%m%d"),
        "--cash",
        str(config.initial_cash),
        "--data-resolution",
        config.data_resolution,
    ]

    # Run backtest
    print(f"Running backtest: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"Backtest failed: {result.stderr}")

    # Parse results (LEAN CLI outputs JSON)
    try:
        backtest_data = json.loads(result.stdout)
    except json.JSONDecodeError:
        # Try to extract JSON from output
        output_lines = result.stdout.split("\n")
        for line in output_lines:
            if line.strip().startswith("{"):
                backtest_data = json.loads(line)
                break
        else:
            raise RuntimeError("Could not parse backtest results")

    return _parse_backtest_result(backtest_data, config)


def run_quantconnect_backtest_api(
    config: BacktestConfig,
    api_token: str,
    project_id: int,
) -> BacktestResult:
    """
    Run backtest using QuantConnect Cloud API.

    Args:
        config: BacktestConfig with algorithm and parameters
        api_token: QuantConnect API token
        project_id: Project ID in QuantConnect

    Returns:
        BacktestResult with performance metrics
    """
    import requests

    # Upload algorithm if needed
    # (Assumes algorithm already exists in project)

    # Create backtest
    url = "https://www.quantconnect.com/api/v2/backtests/create"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "projectId": project_id,
        "compileId": "latest",  # Use latest compilation
        "backtestName": f"Eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    backtest_data = response.json()
    backtest_id = backtest_data.get("backtestId")

    # Poll for completion
    while True:
        import time

        time.sleep(5)

        status_url = (
            f"https://www.quantconnect.com/api/v2/backtests/read?projectId={project_id}&backtestId={backtest_id}"
        )
        status_response = requests.get(status_url, headers=headers)
        status_response.raise_for_status()

        status_data = status_response.json()
        if status_data.get("completed"):
            break

    # Get results
    return _parse_backtest_result(status_data, config)


def _parse_backtest_result(data: dict[str, Any], config: BacktestConfig) -> BacktestResult:
    """
    Parse QuantConnect backtest result JSON.

    Args:
        data: Raw JSON data from LEAN
        config: BacktestConfig used

    Returns:
        BacktestResult with extracted metrics
    """
    # Extract statistics (QuantConnect returns different structures for CLI vs API)
    if "Statistics" in data:
        stats = data["Statistics"]
    elif "statistics" in data:
        stats = data["statistics"]
    else:
        stats = data

    # Helper function to safely get float value
    def get_stat(key: str, default: float = 0.0) -> float:
        value = stats.get(key, default)
        if isinstance(value, str):
            # Remove % sign and convert
            value = value.replace("%", "")
            try:
                return float(value) / 100 if "%" in stats.get(key, "") else float(value)
            except ValueError:
                return default
        return float(value)

    # Basic info
    backtest_id = data.get("backtestId", data.get("BacktestId", "unknown"))
    algorithm_name = config.algorithm_file.stem
    duration_days = (config.end_date - config.start_date).days

    # Performance metrics
    total_return = get_stat("Total Net Profit", get_stat("Compounding Annual Return"))
    sharpe_ratio = get_stat("Sharpe Ratio")
    sortino_ratio = get_stat("Sortino Ratio", sharpe_ratio)  # Fallback to Sharpe if missing
    calmar_ratio = get_stat("Calmar Ratio", 0.0)
    max_drawdown = abs(get_stat("Drawdown"))
    cagr = get_stat("Compounding Annual Return")

    # Trading activity
    total_trades = int(get_stat("Total Orders", 0))
    win_rate = get_stat("Win Rate", 0)
    profit_factor = get_stat("Profit-Loss Ratio", 1.0)
    avg_win = get_stat("Average Win", 0)
    avg_loss = get_stat("Average Loss", 0)

    # Additional metrics
    beta = get_stat("Beta", 0)
    alpha = get_stat("Alpha", 0)
    tracking_error = get_stat("Tracking Error", 0)
    information_ratio = get_stat("Information Ratio", 0)

    # Execution quality (if available)
    avg_fill_time = get_stat("Average Fill Time", 0)  # ms
    slippage_total = get_stat("Total Slippage", 0)
    commission_total = get_stat("Total Fees", 0)

    return BacktestResult(
        backtest_id=backtest_id,
        algorithm_name=algorithm_name,
        start_date=config.start_date,
        end_date=config.end_date,
        duration_days=duration_days,
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_drawdown,
        cagr=cagr,
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        beta=beta,
        alpha=alpha,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
        avg_fill_time_ms=avg_fill_time,
        slippage_total=slippage_total,
        commission_total=commission_total,
        raw_result=data,
    )


def generate_backtest_report(result: BacktestResult) -> str:
    """
    Generate backtest evaluation report.

    Args:
        result: BacktestResult object

    Returns:
        Formatted markdown report
    """
    report = []
    report.append(f"# Backtest Report: {result.algorithm_name}\n")
    report.append(f"**Backtest ID**: {result.backtest_id}")
    report.append(
        f"**Period**: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')} ({result.duration_days} days)\n"
    )

    # Performance summary
    report.append("## 📊 Performance Summary\n")
    report.append(f"- **Total Return**: {result.total_return:.1%} {'✅' if result.total_return > 0.10 else '⚠️'}")
    report.append(f"- **CAGR**: {result.cagr:.1%}")
    report.append(f"- **Max Drawdown**: {result.max_drawdown:.1%} {'✅' if result.max_drawdown < 0.20 else '⚠️'}\n")

    # Risk-adjusted metrics
    report.append("## ⚖️ Risk-Adjusted Metrics\n")
    report.append(f"- **Sharpe Ratio**: {result.sharpe_ratio:.2f} {'✅' if result.sharpe_ratio > 1.0 else '⚠️'}")
    report.append(f"- **Sortino Ratio**: {result.sortino_ratio:.2f} {'✅' if result.sortino_ratio > 1.5 else '⚠️'}")
    report.append(f"- **Calmar Ratio**: {result.calmar_ratio:.2f}\n")

    # Trading activity
    report.append("## 🔄 Trading Activity\n")
    report.append(f"- **Total Trades**: {result.total_trades}")
    report.append(f"- **Win Rate**: {result.win_rate:.1%} {'✅' if result.win_rate > 0.55 else '⚠️'}")
    report.append(f"- **Profit Factor**: {result.profit_factor:.2f} {'✅' if result.profit_factor > 1.5 else '⚠️'}")
    report.append(f"- **Average Win**: ${result.avg_win:,.2f}")
    report.append(f"- **Average Loss**: ${result.avg_loss:,.2f}\n")

    # Market comparison
    report.append("## 📈 Market Comparison\n")
    report.append(f"- **Beta**: {result.beta:.2f}")
    report.append(f"- **Alpha**: {result.alpha:.1%}")
    report.append(f"- **Information Ratio**: {result.information_ratio:.2f}\n")

    # Execution quality
    report.append("## ⚡ Execution Quality\n")
    report.append(f"- **Average Fill Time**: {result.avg_fill_time_ms:.1f}ms")
    report.append(f"- **Total Slippage**: ${result.slippage_total:,.2f}")
    report.append(f"- **Total Commission**: ${result.commission_total:,.2f}\n")

    # Overall assessment
    report.append("## ✅ Assessment\n")

    criteria_met = 0
    total_criteria = 5

    if result.sharpe_ratio > 1.0:
        report.append("✅ Sharpe Ratio > 1.0 (good risk-adjusted returns)")
        criteria_met += 1
    else:
        report.append("⚠️ Sharpe Ratio < 1.0 (weak risk-adjusted returns)")

    if result.max_drawdown < 0.20:
        report.append("✅ Max Drawdown < 20% (controlled risk)")
        criteria_met += 1
    else:
        report.append("⚠️ Max Drawdown > 20% (high risk)")

    if result.win_rate > 0.55:
        report.append("✅ Win Rate > 55% (consistent)")
        criteria_met += 1
    else:
        report.append("⚠️ Win Rate < 55% (inconsistent)")

    if result.profit_factor > 1.5:
        report.append("✅ Profit Factor > 1.5 (strong edge)")
        criteria_met += 1
    else:
        report.append("⚠️ Profit Factor < 1.5 (weak edge)")

    if result.total_return > 0.10:
        report.append("✅ Total Return > 10% (profitable)")
        criteria_met += 1
    else:
        report.append("⚠️ Total Return < 10% (low profitability)")

    report.append(f"\n**Criteria Met**: {criteria_met}/{total_criteria}")

    if criteria_met >= 4:
        report.append("\n**VERDICT**: ✅ Strategy passed backtest - ready for out-of-sample validation")
    elif criteria_met >= 3:
        report.append("\n**VERDICT**: ⚠️ Strategy shows promise but needs improvements")
    else:
        report.append("\n**VERDICT**: ❌ Strategy failed backtest - requires significant changes")

    return "\n".join(report)


def compare_backtests(
    results: list[BacktestResult],
) -> str:
    """
    Compare multiple backtest results.

    Args:
        results: List of BacktestResult objects

    Returns:
        Formatted comparison report
    """
    if not results:
        return "No backtest results to compare"

    report = []
    report.append("# Backtest Comparison Report\n")
    report.append(f"**Number of Backtests**: {len(results)}\n")

    # Comparison table
    report.append("## Performance Comparison\n")
    report.append("| Algorithm | Period | Sharpe | Sortino | Max DD | Win Rate | P.Factor |")
    report.append("|-----------|--------|--------|---------|--------|----------|----------|")

    for result in results:
        period = f"{result.start_date.strftime('%Y-%m')} to {result.end_date.strftime('%Y-%m')}"
        report.append(
            f"| {result.algorithm_name} | {period} | {result.sharpe_ratio:.2f} | "
            f"{result.sortino_ratio:.2f} | {result.max_drawdown:.1%} | "
            f"{result.win_rate:.1%} | {result.profit_factor:.2f} |"
        )

    # Best performer
    best_sharpe = max(results, key=lambda r: r.sharpe_ratio)
    report.append(f"\n**Best Sharpe Ratio**: {best_sharpe.algorithm_name} ({best_sharpe.sharpe_ratio:.2f})")

    return "\n".join(report)
