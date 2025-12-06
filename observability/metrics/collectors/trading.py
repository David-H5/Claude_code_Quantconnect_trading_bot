"""
Advanced Trading Metrics for Algorithm Evaluation.

Implements professional-grade trading metrics beyond basic Sharpe/Sortino:
- Expectancy (average return per trade)
- Profit Factor (gross profit / gross loss)
- Omega Ratio (probability-weighted gains vs losses)
- Win/Loss Ratio and Streaks
- Recovery Factor (net profit / max drawdown)
- Ulcer Index (downside volatility)
- Keller Ratio (modified Sharpe for trading)

Refactored: Phase 3 - Consolidated Metrics Infrastructure

Location: observability/metrics/collectors/trading.py
Old location: evaluation/advanced_trading_metrics.py (re-exports for compatibility)

References:
- https://yourrobotrader.com/en/evaluating-trading-bot-performance/
- https://www.utradealgos.com/blog/5-key-metrics-to-evaluate-the-performance-of-your-trading-algorithms
- https://sd-korp.com/algorithmic-trading-metrics-a-deep-dive-into-sharpe-sortino-and-more/
"""

import math
import statistics
from dataclasses import dataclass
from typing import Any


@dataclass
class Trade:
    """Single trade record."""

    entry_date: str
    exit_date: str
    symbol: str
    pnl: float  # Profit/loss in dollars
    pnl_pct: float  # Profit/loss in percentage
    holding_period_days: int
    result: str  # "win" or "loss"


@dataclass
class AdvancedTradingMetrics:
    """Comprehensive trading performance metrics."""

    # Basic metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Profit metrics
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float
    expectancy: float  # Average profit per trade
    expectancy_pct: float  # Average profit per trade (%)

    # Ratio metrics
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    largest_win: float
    largest_loss: float

    # Streak metrics
    max_consecutive_wins: int
    max_consecutive_losses: int
    current_streak: int
    current_streak_type: str  # "win" or "loss"

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    ulcer_index: float
    recovery_factor: float

    # Drawdown metrics
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration_days: int

    # Additional metrics
    trades_per_day: float
    avg_holding_period_days: float
    profit_per_day: float


def calculate_advanced_trading_metrics(
    trades: list[Trade],
    account_balance: float,
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
) -> AdvancedTradingMetrics:
    """
    Calculate comprehensive trading performance metrics.

    Args:
        trades: List of Trade objects
        account_balance: Starting account balance
        risk_free_rate: Annual risk-free rate (default: 5%)
        trading_days: Number of trading days (default: 252)

    Returns:
        AdvancedTradingMetrics with all calculated metrics
    """
    if not trades:
        raise ValueError("trades list cannot be empty")

    # ========== BASIC METRICS ==========

    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.result == "win")
    losing_trades = sum(1 for t in trades if t.result == "loss")
    win_rate = winning_trades / total_trades

    # ========== PROFIT METRICS ==========

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    net_profit = gross_profit - gross_loss

    # Profit Factor: gross_profit / gross_loss
    # >1.5 is considered strong
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Expectancy: average profit per trade
    expectancy = sum(t.pnl for t in trades) / total_trades
    expectancy_pct = sum(t.pnl_pct for t in trades) / total_trades

    # ========== RATIO METRICS ==========

    winning_pnls = [t.pnl for t in trades if t.result == "win"]
    losing_pnls = [abs(t.pnl) for t in trades if t.result == "loss"]

    avg_win = statistics.mean(winning_pnls) if winning_pnls else 0
    avg_loss = statistics.mean(losing_pnls) if losing_pnls else 0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

    largest_win = max(winning_pnls) if winning_pnls else 0
    largest_loss = max(losing_pnls) if losing_pnls else 0

    # ========== STREAK METRICS ==========

    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_streak = 0
    current_streak_type = ""

    consecutive_wins = 0
    consecutive_losses = 0

    for trade in trades:
        if trade.result == "win":
            consecutive_wins += 1
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            consecutive_wins = 0

        max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

    # Current streak (last trade)
    if trades:
        last_result = trades[-1].result
        current_streak_type = last_result
        current_streak = consecutive_wins if last_result == "win" else consecutive_losses

    # ========== RISK-ADJUSTED METRICS ==========

    returns = [t.pnl_pct / 100 for t in trades]

    # Sharpe Ratio
    if len(returns) > 1:
        avg_return = statistics.mean(returns)
        std_dev = statistics.stdev(returns)
        sharpe_ratio = (
            (avg_return - risk_free_rate / trading_days) / std_dev * math.sqrt(trading_days) if std_dev > 0 else 0
        )
    else:
        sharpe_ratio = 0

    # Sortino Ratio (downside deviation only)
    downside_returns = [r for r in returns if r < 0]
    if len(downside_returns) > 1:
        downside_std = math.sqrt(sum(r**2 for r in downside_returns) / len(downside_returns))
        avg_return = statistics.mean(returns)
        sortino_ratio = (
            (avg_return - risk_free_rate / trading_days) / downside_std * math.sqrt(trading_days)
            if downside_std > 0
            else 0
        )
    else:
        sortino_ratio = sharpe_ratio  # If no downside, use Sharpe

    # Calculate drawdown
    cumulative_pnl = 0
    peak = 0
    max_dd = 0
    drawdowns = []

    for trade in trades:
        cumulative_pnl += trade.pnl
        if cumulative_pnl > peak:
            peak = cumulative_pnl
        dd = peak - cumulative_pnl
        if dd > 0:
            drawdowns.append(dd)
        max_dd = max(max_dd, dd)

    max_drawdown = max_dd / account_balance if account_balance > 0 else 0
    avg_drawdown = statistics.mean(drawdowns) / account_balance if drawdowns and account_balance > 0 else 0

    # Calmar Ratio: return / max_drawdown
    annual_return = net_profit / account_balance
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # Omega Ratio (probability-weighted gains vs losses)
    threshold = 0  # Threshold return (typically 0 or risk-free rate)
    gains = sum(max(0, r - threshold) for r in returns)
    losses = sum(max(0, threshold - r) for r in returns)
    omega_ratio = gains / losses if losses > 0 else float("inf")

    # Ulcer Index (downside volatility)
    if drawdowns:
        ulcer_index = math.sqrt(sum(dd**2 for dd in drawdowns) / len(drawdowns)) / account_balance
    else:
        ulcer_index = 0

    # Recovery Factor: net_profit / max_drawdown
    recovery_factor = net_profit / max_dd if max_dd > 0 else float("inf")

    # Drawdown duration (estimate based on number of losing trades)
    drawdown_duration_days = sum(t.holding_period_days for t in trades if t.result == "loss")

    # ========== ADDITIONAL METRICS ==========

    total_days = sum(t.holding_period_days for t in trades)
    trades_per_day = total_trades / total_days if total_days > 0 else 0
    avg_holding_period_days = total_days / total_trades if total_trades > 0 else 0
    profit_per_day = net_profit / total_days if total_days > 0 else 0

    return AdvancedTradingMetrics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,
        profit_factor=profit_factor,
        expectancy=expectancy,
        expectancy_pct=expectancy_pct,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio,
        largest_win=largest_win,
        largest_loss=largest_loss,
        max_consecutive_wins=max_consecutive_wins,
        max_consecutive_losses=max_consecutive_losses,
        current_streak=current_streak,
        current_streak_type=current_streak_type,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        omega_ratio=omega_ratio,
        ulcer_index=ulcer_index,
        recovery_factor=recovery_factor,
        max_drawdown=max_drawdown,
        avg_drawdown=avg_drawdown,
        drawdown_duration_days=drawdown_duration_days,
        trades_per_day=trades_per_day,
        avg_holding_period_days=avg_holding_period_days,
        profit_per_day=profit_per_day,
    )


def generate_trading_metrics_report(metrics: AdvancedTradingMetrics) -> str:
    """
    Generate comprehensive trading metrics report.

    Args:
        metrics: AdvancedTradingMetrics object

    Returns:
        Formatted markdown report
    """
    report = []
    report.append("# Advanced Trading Metrics Report\n")

    # Basic performance
    report.append("## 📊 Basic Performance")
    report.append(f"- Total Trades: {metrics.total_trades}")
    report.append(f"- Win Rate: {metrics.win_rate:.1%} ({metrics.winning_trades}W / {metrics.losing_trades}L)")
    report.append(f"- Net Profit: ${metrics.net_profit:,.2f}\n")

    # Profit metrics
    report.append("## 💰 Profit Metrics")
    report.append(f"- Gross Profit: ${metrics.gross_profit:,.2f}")
    report.append(f"- Gross Loss: ${metrics.gross_loss:,.2f}")
    report.append(f"- **Profit Factor**: {metrics.profit_factor:.2f} {'✅' if metrics.profit_factor > 1.5 else '⚠️'}")
    report.append(f"- **Expectancy**: ${metrics.expectancy:.2f} per trade ({metrics.expectancy_pct:.2f}%)")
    report.append(f"- Profit per Day: ${metrics.profit_per_day:.2f}\n")

    # Win/Loss analysis
    report.append("## 📈 Win/Loss Analysis")
    report.append(f"- Average Win: ${metrics.avg_win:,.2f}")
    report.append(f"- Average Loss: ${metrics.avg_loss:,.2f}")
    report.append(f"- **Win/Loss Ratio**: {metrics.win_loss_ratio:.2f}")
    report.append(f"- Largest Win: ${metrics.largest_win:,.2f}")
    report.append(f"- Largest Loss: ${metrics.largest_loss:,.2f}\n")

    # Streak analysis
    report.append("## 🔥 Streak Analysis")
    report.append(f"- Max Consecutive Wins: {metrics.max_consecutive_wins}")
    report.append(f"- Max Consecutive Losses: {metrics.max_consecutive_losses}")
    report.append(f"- Current Streak: {metrics.current_streak} {metrics.current_streak_type}s\n")

    # Risk-adjusted metrics
    report.append("## ⚖️ Risk-Adjusted Metrics")
    report.append(f"- **Sharpe Ratio**: {metrics.sharpe_ratio:.2f} {'✅' if metrics.sharpe_ratio > 1.0 else '⚠️'}")
    report.append(f"- **Sortino Ratio**: {metrics.sortino_ratio:.2f} {'✅' if metrics.sortino_ratio > 1.5 else '⚠️'}")
    report.append(f"- **Calmar Ratio**: {metrics.calmar_ratio:.2f}")
    report.append(f"- **Omega Ratio**: {metrics.omega_ratio:.2f} {'✅' if metrics.omega_ratio > 1.5 else '⚠️'}")
    report.append(f"- **Recovery Factor**: {metrics.recovery_factor:.2f}\n")

    # Drawdown analysis
    report.append("## 📉 Drawdown Analysis")
    report.append(f"- **Max Drawdown**: {metrics.max_drawdown:.1%} {'✅' if metrics.max_drawdown < 0.20 else '⚠️'}")
    report.append(f"- Average Drawdown: {metrics.avg_drawdown:.1%}")
    report.append(f"- Ulcer Index: {metrics.ulcer_index:.3f}")
    report.append(f"- Drawdown Duration: {metrics.drawdown_duration_days} days\n")

    # Trading activity
    report.append("## 🔄 Trading Activity")
    report.append(f"- Trades per Day: {metrics.trades_per_day:.2f}")
    report.append(f"- Average Holding Period: {metrics.avg_holding_period_days:.1f} days\n")

    # Overall assessment
    report.append("## ✅ Overall Assessment\n")

    criteria = []
    if metrics.profit_factor > 1.5:
        criteria.append("✅ Profit Factor > 1.5 (strong)")
    else:
        criteria.append("⚠️ Profit Factor < 1.5 (needs improvement)")

    if metrics.sharpe_ratio > 1.0:
        criteria.append("✅ Sharpe Ratio > 1.0 (good risk-adjusted returns)")
    else:
        criteria.append("⚠️ Sharpe Ratio < 1.0 (weak risk-adjusted returns)")

    if metrics.win_rate > 0.60:
        criteria.append("✅ Win Rate > 60% (consistent)")
    else:
        criteria.append("⚠️ Win Rate < 60% (inconsistent)")

    if metrics.max_drawdown < 0.20:
        criteria.append("✅ Max Drawdown < 20% (controlled risk)")
    else:
        criteria.append("⚠️ Max Drawdown > 20% (high risk)")

    for criterion in criteria:
        report.append(criterion)

    passed = sum(1 for c in criteria if c.startswith("✅"))
    if passed >= 3:
        report.append("\n**VERDICT**: ✅ Strategy meets professional standards")
    elif passed >= 2:
        report.append("\n**VERDICT**: ⚠️ Strategy shows promise but needs improvement")
    else:
        report.append("\n**VERDICT**: ❌ Strategy requires significant improvements")

    return "\n".join(report)


def get_trading_metrics_thresholds() -> dict[str, dict[str, float]]:
    """
    Get professional trading metrics thresholds (2025 Updated).

    Updated based on December 2025 research findings:
    - Leading AI bots achieve Sharpe ratios 2.5-3.2
    - Successful bots maintain 85-90% positive months
    - Profit Factor >4.0 suggests overfitting
    - Omega Ratio benchmark ~1.15 for "very good"

    Returns:
        Dict with thresholds for each metric
    """
    return get_trading_metrics_thresholds_2025()


def get_trading_metrics_thresholds_2025() -> dict[str, dict[str, Any]]:
    """
    2025 professional trading metrics thresholds.

    Based on research from December 2025:
    - https://redhub.ai/ai-trading-bots-2025/
    - https://www.luxalgo.com/blog/top-5-metrics-for-evaluating-trading-strategies/

    Returns:
        Dict with updated thresholds including 2025 benchmarks
    """
    return {
        "profit_factor": {
            "excellent": 2.0,
            "good": 1.75,  # Updated: dependable returns threshold
            "acceptable": 1.5,
            "overfitting_warning": 4.0,  # NEW: >4.0 suggests overfitting
        },
        "sharpe_ratio": {
            "excellent": 2.5,  # Updated from 2.0
            "good": 1.5,  # Updated from 1.0
            "acceptable": 1.0,  # Updated from 0.5
            "leading_ai_bots_min": 2.5,  # NEW: 2025 benchmark
            "leading_ai_bots_max": 3.2,  # NEW: 2025 benchmark
        },
        "sortino_ratio": {
            "excellent": 3.0,  # Updated from 2.5
            "good": 2.0,  # Updated from 1.5
            "acceptable": 1.5,  # Updated from 1.0
        },
        "win_rate": {
            "excellent": 0.70,
            "good": 0.60,
            "acceptable": 0.55,
        },
        "max_drawdown": {
            "excellent": 0.03,  # NEW: Best portfolios ~3%
            "good": 0.10,  # Updated from 0.15
            "acceptable": 0.15,  # Updated from 0.20
            "unacceptable": 0.20,  # NEW: 20% max
        },
        "omega_ratio": {
            "excellent": 1.5,
            "good": 1.15,  # Updated: "very good" benchmark
            "acceptable": 1.0,
        },
        "expectancy": {
            "excellent": 100.0,  # >$100 per trade
            "good": 50.0,  # >$50 per trade
            "acceptable": 20.0,  # Updated from 10.0
        },
        "monthly_positive_pct": {  # NEW metric
            "excellent": 0.90,  # 90% positive months
            "good": 0.85,  # 85% positive months
            "acceptable": 0.75,
        },
        "recovery_factor": {
            "excellent": 5.0,
            "good": 3.0,
            "acceptable": 2.0,
        },
    }


def get_trading_metrics_thresholds_legacy() -> dict[str, dict[str, float]]:
    """
    Legacy 2024 thresholds for backward compatibility.

    Returns:
        Dict with original (pre-2025) thresholds
    """
    return {
        "profit_factor": {
            "excellent": 2.0,
            "good": 1.5,
            "acceptable": 1.2,
        },
        "sharpe_ratio": {
            "excellent": 2.0,
            "good": 1.0,
            "acceptable": 0.5,
        },
        "sortino_ratio": {
            "excellent": 2.5,
            "good": 1.5,
            "acceptable": 1.0,
        },
        "win_rate": {
            "excellent": 0.70,
            "good": 0.60,
            "acceptable": 0.55,
        },
        "max_drawdown": {
            "excellent": 0.10,
            "good": 0.15,
            "acceptable": 0.20,
        },
        "omega_ratio": {
            "excellent": 2.0,
            "good": 1.5,
            "acceptable": 1.2,
        },
        "expectancy": {
            "excellent": 100.0,
            "good": 50.0,
            "acceptable": 10.0,
        },
    }


def check_overfitting_signals(metrics: AdvancedTradingMetrics) -> dict[str, Any]:
    """
    Check for overfitting signals in trading metrics.

    Based on 2025 research, high profit factors (>4.0) often indicate
    overfitting to historical data that won't persist in live trading.

    Args:
        metrics: AdvancedTradingMetrics object

    Returns:
        Dict with overfitting warning flags and explanations
    """
    warnings = []
    risk_level = "low"

    # Profit Factor >4.0 warning (from research)
    if metrics.profit_factor > 4.0:
        warnings.append(
            {
                "metric": "profit_factor",
                "value": metrics.profit_factor,
                "threshold": 4.0,
                "message": f"Profit Factor {metrics.profit_factor:.2f} > 4.0 suggests potential overfitting. "
                "Verify with out-of-sample testing.",
            }
        )
        risk_level = "high"

    # Extremely high Sharpe (>3.5) might indicate overfitting
    if metrics.sharpe_ratio > 3.5:
        warnings.append(
            {
                "metric": "sharpe_ratio",
                "value": metrics.sharpe_ratio,
                "threshold": 3.5,
                "message": f"Sharpe Ratio {metrics.sharpe_ratio:.2f} exceeds top AI bot benchmarks (2.5-3.2). "
                "Verify this is not curve-fitting.",
            }
        )
        risk_level = "medium" if risk_level == "low" else risk_level

    # Perfect or near-perfect win rate is suspicious
    if metrics.win_rate > 0.85:
        warnings.append(
            {
                "metric": "win_rate",
                "value": metrics.win_rate,
                "threshold": 0.85,
                "message": f"Win Rate {metrics.win_rate:.1%} is unusually high. "
                "May indicate look-ahead bias or overfitting.",
            }
        )
        risk_level = "medium" if risk_level == "low" else risk_level

    # Very low max drawdown with high returns might be unrealistic
    if metrics.max_drawdown < 0.02 and metrics.sharpe_ratio > 2.0:
        warnings.append(
            {
                "metric": "max_drawdown",
                "value": metrics.max_drawdown,
                "threshold": 0.02,
                "message": f"Max Drawdown {metrics.max_drawdown:.1%} with Sharpe {metrics.sharpe_ratio:.2f} "
                "seems unrealistic. Check for survivorship bias.",
            }
        )
        risk_level = "medium" if risk_level == "low" else risk_level

    return {
        "overfitting_risk": risk_level,
        "warning_count": len(warnings),
        "warnings": warnings,
        "recommendation": _get_overfitting_recommendation(risk_level, len(warnings)),
    }


def _get_overfitting_recommendation(risk_level: str, warning_count: int) -> str:
    """Get recommendation based on overfitting risk level."""
    if risk_level == "high":
        return (
            "HIGH RISK: Multiple overfitting signals detected. "
            "Run walk-forward analysis with at least 12 months out-of-sample. "
            "Limit backtests to ≤20 and parameters to ≤5."
        )
    elif risk_level == "medium":
        return (
            "MODERATE RISK: Some metrics suggest possible overfitting. "
            "Verify with out-of-sample testing before live deployment."
        )
    else:
        return "LOW RISK: No obvious overfitting signals detected."


def compare_to_2025_benchmarks(metrics: AdvancedTradingMetrics) -> dict[str, Any]:
    """
    Compare strategy metrics to 2025 AI trading bot benchmarks.

    Based on research: Leading AI bots achieve Sharpe 2.5-3.2,
    85-90% positive months, max drawdown ~3%.

    Args:
        metrics: AdvancedTradingMetrics object

    Returns:
        Dict with comparison to 2025 benchmarks
    """
    benchmarks = {
        "sharpe_ratio": {"min": 2.5, "max": 3.2, "label": "Leading AI Bots"},
        "max_drawdown": {"target": 0.03, "label": "Best Portfolios"},
    }

    comparisons = []

    # Sharpe comparison
    sharpe_status = "below"
    if 2.5 <= metrics.sharpe_ratio <= 3.2:
        sharpe_status = "meets"
    elif metrics.sharpe_ratio > 3.2:
        sharpe_status = "exceeds"
    comparisons.append(
        {
            "metric": "sharpe_ratio",
            "value": metrics.sharpe_ratio,
            "benchmark": "2.5-3.2",
            "status": sharpe_status,
        }
    )

    # Max drawdown comparison
    dd_status = "meets" if metrics.max_drawdown <= 0.03 else "exceeds"
    comparisons.append(
        {
            "metric": "max_drawdown",
            "value": metrics.max_drawdown,
            "benchmark": "~3%",
            "status": dd_status,
        }
    )

    # Overall assessment
    meets_benchmarks = sum(1 for c in comparisons if c["status"] in ["meets", "exceeds"])

    return {
        "comparisons": comparisons,
        "benchmarks_met": meets_benchmarks,
        "total_benchmarks": len(comparisons),
        "performance_tier": _get_performance_tier(metrics),
    }


def _get_performance_tier(metrics: AdvancedTradingMetrics) -> str:
    """Determine performance tier based on 2025 benchmarks."""
    if metrics.sharpe_ratio >= 2.5 and metrics.max_drawdown <= 0.05:
        return "elite"
    elif metrics.sharpe_ratio >= 1.5 and metrics.max_drawdown <= 0.10:
        return "professional"
    elif metrics.sharpe_ratio >= 1.0 and metrics.max_drawdown <= 0.15:
        return "acceptable"
    else:
        return "needs_improvement"
