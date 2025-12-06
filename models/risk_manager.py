"""
Risk Management Models

This module provides risk management classes for controlling trading risk,
including position limits, drawdown controls, and exposure management.

Author: QuantConnect Trading Bot
Date: 2025-11-25
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class RiskAction(Enum):
    """Actions that risk manager can recommend."""

    ALLOW = "allow"
    REDUCE = "reduce"
    CLOSE = "close"
    BLOCK = "block"


@dataclass
class RiskLimits:
    """Configuration for risk limits."""

    # Position limits
    max_position_size: float = 0.25  # Max 25% of portfolio in single position
    max_total_exposure: float = 1.0  # Max 100% total exposure

    # Loss limits
    max_daily_loss: float = 0.03  # Max 3% daily loss
    max_drawdown: float = 0.10  # Max 10% drawdown

    # Trade limits
    max_trades_per_day: int = 10
    min_time_between_trades: int = 60  # Seconds

    # Risk per trade
    max_risk_per_trade: float = 0.02  # Max 2% risk per trade


@dataclass
class PositionInfo:
    """Information about a current position."""

    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        """Calculate current market value of position."""
        return abs(self.quantity) * self.current_price

    @property
    def pnl_percent(self) -> float:
        """Calculate P&L as percentage of entry value."""
        entry_value = abs(self.quantity) * self.entry_price
        if entry_value == 0:
            return 0.0
        return self.unrealized_pnl / entry_value


class RiskManager:
    """
    Risk management system for controlling trading risk.

    This class monitors portfolio risk and provides recommendations
    for position sizing and risk controls.

    Attributes:
        limits: Risk limit configuration
        starting_equity: Initial portfolio value
        current_equity: Current portfolio value
        daily_starting_equity: Equity at start of trading day
        trades_today: Number of trades executed today
        peak_equity: Highest equity reached
    """

    def __init__(
        self,
        starting_equity: float,
        limits: RiskLimits | None = None,
    ):
        """
        Initialize RiskManager.

        Args:
            starting_equity: Initial portfolio value
            limits: Risk limit configuration (uses defaults if None)
        """
        self.limits = limits or RiskLimits()
        self.starting_equity = starting_equity
        self.current_equity = starting_equity
        self.daily_starting_equity = starting_equity
        self.peak_equity = starting_equity
        self.trades_today = 0
        self._positions: dict[str, PositionInfo] = {}
        self._last_trade_time: float | None = None
        self._daily_pnl = 0.0

    def update_equity(self, new_equity: float) -> None:
        """
        Update current equity value.

        Args:
            new_equity: New portfolio value
        """
        old_equity = self.current_equity
        self.current_equity = new_equity

        # Update peak
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity

        # Update daily P&L
        self._daily_pnl = (new_equity - self.daily_starting_equity) / self.daily_starting_equity

    def reset_daily_stats(self) -> None:
        """Reset daily statistics at start of new trading day."""
        self.daily_starting_equity = self.current_equity
        self.trades_today = 0
        self._daily_pnl = 0.0

    def update_position(self, position: PositionInfo) -> None:
        """
        Update or add a position.

        Args:
            position: Position information to update
        """
        if position.quantity == 0:
            self._positions.pop(position.symbol, None)
        else:
            self._positions[position.symbol] = position

    def remove_position(self, symbol: str) -> None:
        """Remove a position from tracking."""
        self._positions.pop(symbol, None)

    @property
    def current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    @property
    def daily_pnl(self) -> float:
        """Get current daily P&L percentage."""
        return self._daily_pnl

    @property
    def total_exposure(self) -> float:
        """Calculate total market exposure as percentage of equity."""
        if self.current_equity <= 0:
            return 0.0
        total_value = sum(pos.market_value for pos in self._positions.values())
        return total_value / self.current_equity

    def get_position_weight(self, symbol: str) -> float:
        """Get position weight as percentage of portfolio."""
        if symbol not in self._positions or self.current_equity <= 0:
            return 0.0
        return self._positions[symbol].market_value / self.current_equity

    def check_can_trade(self, current_time: float | None = None) -> tuple[bool, str]:
        """
        Check if trading is allowed based on risk limits.

        Args:
            current_time: Current timestamp for time-based checks

        Returns:
            Tuple of (can_trade, reason)
        """
        # Check daily loss limit
        if self._daily_pnl < -self.limits.max_daily_loss:
            return False, f"Daily loss limit exceeded: {self._daily_pnl:.2%}"

        # Check drawdown limit
        if self.current_drawdown > self.limits.max_drawdown:
            return False, f"Max drawdown exceeded: {self.current_drawdown:.2%}"

        # Check trade count
        if self.trades_today >= self.limits.max_trades_per_day:
            return False, f"Daily trade limit reached: {self.trades_today}"

        # Check time between trades
        if current_time and self._last_trade_time:
            time_since_last = current_time - self._last_trade_time
            if time_since_last < self.limits.min_time_between_trades:
                return False, f"Min time between trades not met: {time_since_last:.0f}s"

        return True, "OK"

    def check_position_size(
        self,
        symbol: str,
        proposed_size: float,
        price: float,
    ) -> tuple[RiskAction, float, str]:
        """
        Check if a proposed position size is within risk limits.

        Args:
            symbol: Symbol to check
            proposed_size: Proposed position size (fraction of portfolio)
            price: Current price

        Returns:
            Tuple of (action, adjusted_size, reason)
        """
        # Check position limit
        if proposed_size > self.limits.max_position_size:
            return (
                RiskAction.REDUCE,
                self.limits.max_position_size,
                f"Position reduced to max {self.limits.max_position_size:.0%}",
            )

        # Check total exposure with new position
        current_weight = self.get_position_weight(symbol)
        new_exposure = self.total_exposure - current_weight + proposed_size

        if new_exposure > self.limits.max_total_exposure:
            max_allowed = self.limits.max_total_exposure - (self.total_exposure - current_weight)
            if max_allowed <= 0:
                return RiskAction.BLOCK, 0.0, "Total exposure limit reached"
            return (
                RiskAction.REDUCE,
                max_allowed,
                f"Position reduced for exposure limit: {max_allowed:.2%}",
            )

        return RiskAction.ALLOW, proposed_size, "OK"

    def calculate_stop_loss(
        self,
        entry_price: float,
        position_size: float,
        is_long: bool = True,
    ) -> float:
        """
        Calculate stop loss price based on risk per trade.

        Args:
            entry_price: Entry price
            position_size: Position size as fraction of portfolio
            is_long: True for long position, False for short

        Returns:
            Stop loss price
        """
        # Risk amount = position_value * stop_distance_pct
        # max_risk_per_trade = risk_amount / equity
        # stop_distance_pct = max_risk_per_trade / position_size

        if position_size <= 0:
            return entry_price

        stop_distance_pct = self.limits.max_risk_per_trade / position_size

        if is_long:
            return entry_price * (1 - stop_distance_pct)
        else:
            return entry_price * (1 + stop_distance_pct)

    def record_trade(self, current_time: float | None = None) -> None:
        """
        Record that a trade was executed.

        Args:
            current_time: Timestamp of trade
        """
        self.trades_today += 1
        self._last_trade_time = current_time

    def get_risk_status(self) -> dict[str, Any]:
        """
        Get current risk status summary.

        Returns:
            Dictionary with risk metrics
        """
        return {
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "drawdown": self.current_drawdown,
            "drawdown_limit": self.limits.max_drawdown,
            "daily_pnl": self._daily_pnl,
            "daily_loss_limit": self.limits.max_daily_loss,
            "total_exposure": self.total_exposure,
            "exposure_limit": self.limits.max_total_exposure,
            "trades_today": self.trades_today,
            "trade_limit": self.limits.max_trades_per_day,
            "positions": len(self._positions),
        }

    def should_reduce_risk(self) -> tuple[bool, str]:
        """
        Check if risk should be reduced based on current conditions.

        Returns:
            Tuple of (should_reduce, reason)
        """
        # Check if approaching drawdown limit (75% of max)
        if self.current_drawdown > self.limits.max_drawdown * 0.75:
            return True, f"Approaching drawdown limit: {self.current_drawdown:.2%}"

        # Check if approaching daily loss limit
        if self._daily_pnl < -self.limits.max_daily_loss * 0.75:
            return True, f"Approaching daily loss limit: {self._daily_pnl:.2%}"

        return False, "OK"

    def sync_from_algorithm(self, algorithm) -> None:
        """
        Sync risk manager state from QuantConnect algorithm.

        INTEGRATION: Call this from algorithm.OnData() to update positions and equity

        Example:
            def OnData(self, slice):
                if self.IsWarmingUp:
                    return

                self.risk_manager.sync_from_algorithm(self)

                # Check if can trade
                can_trade, reason = self.risk_manager.check_can_trade(self.Time.timestamp())
                if not can_trade:
                    self.Debug(f"Trading blocked: {reason}")
                    return

        Args:
            algorithm: QCAlgorithm instance
        """
        # Update equity from Portfolio
        self.update_equity(algorithm.Portfolio.TotalPortfolioValue)

        # Clear and update positions
        self._positions.clear()

        # Iterate through Portfolio.Values (official QuantConnect pattern)
        for holding in algorithm.Portfolio.Values:
            if not holding.Invested:
                continue

            position = PositionInfo(
                symbol=str(holding.Symbol),
                quantity=holding.Quantity,
                entry_price=holding.AveragePrice,
                current_price=holding.Price,
                unrealized_pnl=holding.UnrealizedProfit,
            )
            self._positions[str(holding.Symbol)] = position

    def check_can_open_position_qc(
        self,
        algorithm,
        symbol: str,
        target_quantity: int,
    ) -> tuple[RiskAction, int, str]:
        """
        Check if a new position can be opened in QuantConnect.

        INTEGRATION: Call this before submitting orders

        Example:
            def OnData(self, slice):
                target_qty = 100
                action, adjusted_qty, reason = self.risk_manager.check_can_open_position_qc(
                    self, "SPY", target_qty
                )

                if action == RiskAction.BLOCK:
                    self.Debug(f"Order blocked: {reason}")
                    return
                elif action == RiskAction.REDUCE:
                    self.Debug(f"Order reduced: {reason}")
                    target_qty = adjusted_qty

                self.MarketOrder("SPY", target_qty)

        Args:
            algorithm: QCAlgorithm instance
            symbol: Symbol to trade
            target_quantity: Desired quantity

        Returns:
            Tuple of (action, adjusted_quantity, reason)
        """
        # First check if can trade at all
        can_trade, reason = self.check_can_trade(algorithm.Time.timestamp())
        if not can_trade:
            return RiskAction.BLOCK, 0, reason

        # Get price
        if not algorithm.Securities.ContainsKey(symbol):
            return RiskAction.BLOCK, 0, f"Symbol {symbol} not found in Securities"

        price = algorithm.Securities[symbol].Price
        if price <= 0:
            return RiskAction.BLOCK, 0, f"Invalid price for {symbol}"

        # Calculate position size as fraction of portfolio
        position_value = abs(target_quantity) * price
        position_size = position_value / self.current_equity if self.current_equity > 0 else 0

        # Check position size limits
        action, adjusted_size, msg = self.check_position_size(symbol, position_size, price)

        if action == RiskAction.BLOCK:
            return action, 0, msg
        elif action == RiskAction.REDUCE:
            # Convert adjusted size back to quantity
            adjusted_qty = int((adjusted_size * self.current_equity) / price)
            return action, adjusted_qty, msg
        else:
            return RiskAction.ALLOW, target_quantity, msg

    def record_trade_qc(self, algorithm) -> None:
        """
        Record a trade execution in QuantConnect.

        INTEGRATION: Call this from OnOrderEvent() when filled

        Example:
            def OnOrderEvent(self, order_event):
                from AlgorithmImports import OrderStatus

                if order_event.Status == OrderStatus.Filled:
                    self.risk_manager.record_trade_qc(self)

        Args:
            algorithm: QCAlgorithm instance
        """
        self.record_trade(algorithm.Time.timestamp())

    def reset_daily_stats_qc(self, algorithm) -> None:
        """
        Reset daily statistics at start of new trading day.

        INTEGRATION: Call this at start of each trading day

        Example:
            def Initialize(self):
                # Schedule daily reset at market open
                self.Schedule.On(
                    self.DateRules.EveryDay("SPY"),
                    self.TimeRules.AfterMarketOpen("SPY", 1),
                    lambda: self.risk_manager.reset_daily_stats_qc(self)
                )

        Args:
            algorithm: QCAlgorithm instance
        """
        self.reset_daily_stats()
        algorithm.Debug(f"Risk manager daily reset - Starting equity: ${self.daily_starting_equity:,.2f}")
