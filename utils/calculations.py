"""
Common calculation utilities for trading algorithms.

This module provides reusable calculation functions for:
- Position sizing
- Risk metrics
- Performance calculations
- Statistical helpers

Author: QuantConnect Trading Bot
Date: 2025-11-25
"""

import math


def calculate_position_size(
    portfolio_value: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
    max_position_pct: float = 1.0,
) -> float:
    """
    Calculate position size based on risk per trade.

    Uses the formula: Position Size = (Portfolio * Risk%) / (Entry - Stop)

    Args:
        portfolio_value: Total portfolio value
        risk_per_trade: Percentage of portfolio to risk (e.g., 0.02 for 2%)
        entry_price: Expected entry price
        stop_loss_price: Stop loss price
        max_position_pct: Maximum position as percentage of portfolio (default 1.0)

    Returns:
        Number of shares to buy (floored to whole number)

    Raises:
        ValueError: If stop loss is not valid for the position direction
    """
    if portfolio_value <= 0:
        raise ValueError("Portfolio value must be positive")

    if risk_per_trade <= 0 or risk_per_trade > 1:
        raise ValueError("Risk per trade must be between 0 and 1")

    if entry_price <= 0 or stop_loss_price <= 0:
        raise ValueError("Prices must be positive")

    # Calculate risk per share
    risk_per_share = abs(entry_price - stop_loss_price)

    if risk_per_share == 0:
        raise ValueError("Entry price and stop loss cannot be the same")

    # Calculate dollar risk
    dollar_risk = portfolio_value * risk_per_trade

    # Calculate shares based on risk
    shares = dollar_risk / risk_per_share

    # Apply maximum position constraint
    max_shares = (portfolio_value * max_position_pct) / entry_price
    shares = min(shares, max_shares)

    return math.floor(shares)


def calculate_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float | None:
    """
    Calculate the Sharpe ratio of a returns series.

    Args:
        returns: List of periodic returns (as decimals, e.g., 0.01 for 1%)
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year (default 252 for daily)

    Returns:
        Annualized Sharpe ratio, or None if calculation not possible
    """
    if len(returns) < 2:
        return None

    # Convert annual risk-free rate to periodic
    periodic_rf = risk_free_rate / periods_per_year

    # Calculate excess returns
    excess_returns = [r - periodic_rf for r in returns]

    # Calculate mean and standard deviation
    mean_return = sum(excess_returns) / len(excess_returns)
    variance = sum((r - mean_return) ** 2 for r in excess_returns) / (len(excess_returns) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0

    if std_dev == 0:
        return None

    # Annualize
    sharpe = (mean_return / std_dev) * math.sqrt(periods_per_year)

    return sharpe


def calculate_max_drawdown(equity_curve: list[float]) -> tuple[float, int, int]:
    """
    Calculate maximum drawdown from an equity curve.

    Args:
        equity_curve: List of portfolio values over time

    Returns:
        Tuple of (max_drawdown_pct, peak_index, trough_index)
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0

    max_drawdown = 0.0
    peak_value = equity_curve[0]
    peak_idx = 0
    dd_peak_idx = 0
    dd_trough_idx = 0

    for i, value in enumerate(equity_curve):
        if value > peak_value:
            peak_value = value
            peak_idx = i

        if peak_value > 0:
            drawdown = (peak_value - value) / peak_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                dd_peak_idx = peak_idx
                dd_trough_idx = i

    return max_drawdown, dd_peak_idx, dd_trough_idx


def calculate_sortino_ratio(
    returns: list[float],
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float | None:
    """
    Calculate the Sortino ratio (uses downside deviation instead of std dev).

    Args:
        returns: List of periodic returns
        target_return: Minimum acceptable return (default 0)
        periods_per_year: Trading periods per year

    Returns:
        Annualized Sortino ratio, or None if calculation not possible
    """
    if len(returns) < 2:
        return None

    # Calculate mean return
    mean_return = sum(returns) / len(returns)

    # Calculate downside deviation (only negative returns vs target)
    downside_returns = [min(0, r - target_return) for r in returns]
    downside_variance = sum(r**2 for r in downside_returns) / len(returns)
    downside_dev = math.sqrt(downside_variance) if downside_variance > 0 else 0

    if downside_dev == 0:
        return None

    # Annualize
    sortino = ((mean_return - target_return) / downside_dev) * math.sqrt(periods_per_year)

    return sortino


def calculate_win_rate(trades: list[float]) -> float | None:
    """
    Calculate win rate from a list of trade P&Ls.

    Args:
        trades: List of trade profits/losses

    Returns:
        Win rate as decimal (0-1), or None if no trades
    """
    if not trades:
        return None

    winning_trades = sum(1 for t in trades if t > 0)
    return winning_trades / len(trades)


def calculate_profit_factor(trades: list[float]) -> float | None:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        trades: List of trade profits/losses

    Returns:
        Profit factor, or None if no losing trades
    """
    if not trades:
        return None

    gross_profit = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t < 0))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else None

    return gross_profit / gross_loss


def calculate_cagr(
    initial_value: float,
    final_value: float,
    years: float,
) -> float | None:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        initial_value: Starting portfolio value
        final_value: Ending portfolio value
        years: Number of years

    Returns:
        CAGR as decimal, or None if calculation not possible
    """
    if initial_value <= 0 or final_value <= 0 or years <= 0:
        return None

    return (final_value / initial_value) ** (1 / years) - 1


def calculate_volatility(
    returns: list[float],
    periods_per_year: int = 252,
) -> float | None:
    """
    Calculate annualized volatility from returns.

    Args:
        returns: List of periodic returns
        periods_per_year: Trading periods per year

    Returns:
        Annualized volatility as decimal
    """
    if len(returns) < 2:
        return None

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance)

    return std_dev * math.sqrt(periods_per_year)


def calculate_kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """
    Calculate Kelly Criterion fraction for position sizing.

    Args:
        win_rate: Probability of winning (0-1)
        win_loss_ratio: Average win / average loss

    Returns:
        Optimal fraction of portfolio to bet
    """
    if win_rate <= 0 or win_rate >= 1 or win_loss_ratio <= 0:
        return 0.0

    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

    # Return 0 if negative (don't bet)
    return max(0.0, kelly)
