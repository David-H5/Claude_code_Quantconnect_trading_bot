"""
Unified Performance Tracker for Trading Bot

Provides real-time performance tracking with:
- Live P&L and drawdown monitoring
- Session-based metrics (daily, weekly, monthly)
- Strategy-level performance breakdown
- Trade recording and analysis
- Integration with structured logging
- Object Store persistence

UPGRADE-010: Performance Tracker (December 2025)
"""

from __future__ import annotations

import json
import statistics
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class TradeRecord:
    """Individual trade record with full details."""

    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    strategy: str = "unknown"
    direction: str = "long"  # "long" or "short"
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exit_time: datetime | None = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage_bps: float = 0.0
    is_closed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def close(
        self,
        exit_price: float,
        exit_time: datetime | None = None,
        commission: float = 0.0,
    ) -> None:
        """Close the trade and calculate P&L.

        Args:
            exit_price: Exit price
            exit_time: Exit timestamp
            commission: Commission for exit
        """
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now(timezone.utc)
        self.commission += commission
        self.is_closed = True

        # Calculate P&L
        if self.direction == "long":
            self.pnl = (exit_price - self.entry_price) * self.quantity - self.commission
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity - self.commission

        if self.entry_price > 0:
            self.pnl_pct = self.pnl / (self.entry_price * self.quantity)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "direction": self.direction,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "commission": self.commission,
            "slippage_bps": self.slippage_bps,
            "is_closed": self.is_closed,
        }


@dataclass
class SessionMetrics:
    """Metrics for a trading session (day/week/month)."""

    session_date: date
    session_type: str = "daily"  # "daily", "weekly", "monthly"
    trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    commissions: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_equity: float = 0.0
    ending_equity: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    trade_pnls: list[float] = field(default_factory=list)

    def update_metrics(self) -> None:
        """Recalculate derived metrics."""
        if self.trades > 0:
            self.win_rate = self.winning_trades / self.trades

        # Calculate avg win/loss
        wins = [p for p in self.trade_pnls if p > 0]
        losses = [p for p in self.trade_pnls if p < 0]

        if wins:
            self.avg_win = statistics.mean(wins)
        if losses:
            self.avg_loss = abs(statistics.mean(losses))

        # Profit factor
        gross_profit = sum(p for p in self.trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in self.trade_pnls if p < 0))
        if gross_loss > 0:
            self.profit_factor = gross_profit / gross_loss

        # Sharpe ratio (simplified daily)
        if len(self.trade_pnls) > 1:
            try:
                mean_return = statistics.mean(self.trade_pnls)
                std_return = statistics.stdev(self.trade_pnls)
                if std_return > 0:
                    self.sharpe_ratio = (mean_return / std_return) * (252**0.5)
            except Exception:
                pass

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_date": self.session_date.isoformat(),
            "session_type": self.session_type,
            "trades": self.trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "net_pnl": self.net_pnl,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
        }


@dataclass
class StrategyMetrics:
    """Metrics for a specific strategy."""

    strategy_name: str
    trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_hold_time_hours: float = 0.0
    trade_pnls: list[float] = field(default_factory=list)

    def update_metrics(self) -> None:
        """Recalculate derived metrics."""
        if self.trades > 0:
            self.win_rate = self.winning_trades / self.trades
            self.avg_pnl = self.net_pnl / self.trades

        # Profit factor
        gross_profit = sum(p for p in self.trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in self.trade_pnls if p < 0))
        if gross_loss > 0:
            self.profit_factor = gross_profit / gross_loss

        # Sharpe ratio
        if len(self.trade_pnls) > 1:
            try:
                mean_return = statistics.mean(self.trade_pnls)
                std_return = statistics.stdev(self.trade_pnls)
                if std_return > 0:
                    self.sharpe_ratio = (mean_return / std_return) * (252**0.5)
            except Exception:
                pass

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "trades": self.trades,
            "winning_trades": self.winning_trades,
            "net_pnl": self.net_pnl,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "avg_hold_time_hours": self.avg_hold_time_hours,
        }


@dataclass
class PerformanceSummary:
    """Overall performance summary."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    current_equity: float = 0.0
    peak_equity: float = 0.0
    starting_equity: float = 0.0
    return_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "current_drawdown_pct": self.current_drawdown_pct,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "return_pct": self.return_pct,
        }


# ============================================================================
# Performance Tracker
# ============================================================================


class PerformanceTracker:
    """
    Unified performance tracker for trading operations.

    Features:
    - Real-time P&L tracking
    - Session management (daily/weekly/monthly)
    - Strategy-level breakdown
    - Drawdown monitoring
    - Trade recording
    - Logging integration
    - Object Store persistence

    Example:
        >>> tracker = PerformanceTracker(starting_equity=100000)
        >>> trade = tracker.open_trade("SPY", "iron_condor", "long", 450.0, 10)
        >>> tracker.close_trade(trade.trade_id, 455.0)
        >>> summary = tracker.get_summary()
    """

    def __init__(
        self,
        starting_equity: float = 100000.0,
        logger: Any | None = None,
        object_store: Any | None = None,
    ):
        """Initialize performance tracker.

        Args:
            starting_equity: Initial portfolio equity
            logger: Optional StructuredLogger for event logging
            object_store: Optional ObjectStoreManager for persistence
        """
        self.starting_equity = starting_equity
        self.current_equity = starting_equity
        self.peak_equity = starting_equity
        self.logger = logger
        self.object_store = object_store

        # Trade tracking
        self._open_trades: dict[str, TradeRecord] = {}
        self._closed_trades: list[TradeRecord] = []

        # Session tracking
        self._daily_sessions: dict[date, SessionMetrics] = {}
        self._weekly_sessions: dict[date, SessionMetrics] = {}
        self._monthly_sessions: dict[date, SessionMetrics] = {}

        # Strategy tracking
        self._strategies: dict[str, StrategyMetrics] = {}

        # Real-time metrics
        self._unrealized_pnl: float = 0.0
        self._realized_pnl: float = 0.0
        self._max_drawdown: float = 0.0
        self._max_drawdown_pct: float = 0.0

        # Event listeners
        self._listeners: list[Callable[[str, dict[str, Any]], None]] = []

        # Thread safety
        self._lock = threading.Lock()

    # =========================================================================
    # Trade Management
    # =========================================================================

    def open_trade(
        self,
        symbol: str,
        strategy: str,
        direction: str,
        entry_price: float,
        quantity: int,
        commission: float = 0.0,
        **metadata,
    ) -> TradeRecord:
        """Open a new trade.

        Args:
            symbol: Trading symbol
            strategy: Strategy name
            direction: "long" or "short"
            entry_price: Entry price
            quantity: Position size
            commission: Entry commission
            **metadata: Additional trade metadata

        Returns:
            TradeRecord for the new trade
        """
        trade = TradeRecord(
            symbol=symbol,
            strategy=strategy,
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            commission=commission,
            metadata=metadata,
        )

        with self._lock:
            self._open_trades[trade.trade_id] = trade

            # Ensure strategy exists
            if strategy not in self._strategies:
                self._strategies[strategy] = StrategyMetrics(strategy_name=strategy)

        # Log event
        if self.logger:
            self.logger.log_position_opened(
                position_id=trade.trade_id,
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                strategy=strategy,
            )

        # Notify listeners
        self._notify("trade_opened", trade.to_dict())

        return trade

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        commission: float = 0.0,
        slippage_bps: float = 0.0,
    ) -> TradeRecord | None:
        """Close an open trade.

        Args:
            trade_id: ID of trade to close
            exit_price: Exit price
            commission: Exit commission
            slippage_bps: Slippage in basis points

        Returns:
            Closed TradeRecord or None if not found
        """
        with self._lock:
            if trade_id not in self._open_trades:
                return None

            trade = self._open_trades.pop(trade_id)
            trade.close(exit_price, commission=commission)
            trade.slippage_bps = slippage_bps
            self._closed_trades.append(trade)

            # Update realized P&L
            self._realized_pnl += trade.pnl
            self.current_equity += trade.pnl

            # Update peak and drawdown
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity

            drawdown = self.peak_equity - self.current_equity
            if drawdown > self._max_drawdown:
                self._max_drawdown = drawdown
                if self.peak_equity > 0:
                    self._max_drawdown_pct = drawdown / self.peak_equity

            # Update session metrics
            self._update_session_metrics(trade)

            # Update strategy metrics
            self._update_strategy_metrics(trade)

        # Log event
        if self.logger:
            hold_time = 0.0
            if trade.exit_time and trade.entry_time:
                hold_time = (trade.exit_time - trade.entry_time).total_seconds() / 3600

            self.logger.log_position_closed(
                position_id=trade.trade_id,
                symbol=trade.symbol,
                quantity=trade.quantity,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                pnl=trade.pnl,
                pnl_pct=trade.pnl_pct,
                hold_time_hours=hold_time,
                exit_reason="manual_close",
            )

        # Notify listeners
        self._notify("trade_closed", trade.to_dict())

        return trade

    def update_unrealized_pnl(self, pnl: float) -> None:
        """Update unrealized P&L from open positions.

        Args:
            pnl: Current unrealized P&L
        """
        with self._lock:
            self._unrealized_pnl = pnl
            total_equity = self.starting_equity + self._realized_pnl + pnl

            if total_equity > self.peak_equity:
                self.peak_equity = total_equity

            # Check for new max drawdown
            drawdown = self.peak_equity - total_equity
            if drawdown > self._max_drawdown:
                self._max_drawdown = drawdown
                if self.peak_equity > 0:
                    self._max_drawdown_pct = drawdown / self.peak_equity

    def _update_session_metrics(self, trade: TradeRecord) -> None:
        """Update session metrics with closed trade."""
        today = trade.exit_time.date() if trade.exit_time else date.today()

        # Daily session
        if today not in self._daily_sessions:
            self._daily_sessions[today] = SessionMetrics(
                session_date=today,
                session_type="daily",
            )

        session = self._daily_sessions[today]
        session.trades += 1
        session.net_pnl += trade.pnl
        session.commissions += trade.commission
        session.trade_pnls.append(trade.pnl)

        if trade.pnl > 0:
            session.winning_trades += 1
            session.gross_pnl += trade.pnl
        else:
            session.losing_trades += 1

        session.update_metrics()

    def _update_strategy_metrics(self, trade: TradeRecord) -> None:
        """Update strategy metrics with closed trade."""
        strategy_name = trade.strategy

        if strategy_name not in self._strategies:
            self._strategies[strategy_name] = StrategyMetrics(strategy_name=strategy_name)

        strategy = self._strategies[strategy_name]
        strategy.trades += 1
        strategy.net_pnl += trade.pnl
        strategy.trade_pnls.append(trade.pnl)

        if trade.pnl > 0:
            strategy.winning_trades += 1
            strategy.gross_pnl += trade.pnl
        else:
            strategy.losing_trades += 1

        # Calculate hold time
        if trade.exit_time and trade.entry_time:
            hold_hours = (trade.exit_time - trade.entry_time).total_seconds() / 3600
            total_hours = strategy.avg_hold_time_hours * (strategy.trades - 1) + hold_hours
            strategy.avg_hold_time_hours = total_hours / strategy.trades

        strategy.update_metrics()

    # =========================================================================
    # Metrics Retrieval
    # =========================================================================

    def get_summary(self) -> PerformanceSummary:
        """Get overall performance summary.

        Returns:
            PerformanceSummary with all metrics
        """
        with self._lock:
            total_trades = len(self._closed_trades)
            winning = sum(1 for t in self._closed_trades if t.pnl > 0)
            losing = sum(1 for t in self._closed_trades if t.pnl < 0)

            # Calculate profit factor
            gross_profit = sum(t.pnl for t in self._closed_trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in self._closed_trades if t.pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

            # Calculate Sharpe ratio
            sharpe = 0.0
            if total_trades > 1:
                pnls = [t.pnl for t in self._closed_trades]
                try:
                    mean_pnl = statistics.mean(pnls)
                    std_pnl = statistics.stdev(pnls)
                    if std_pnl > 0:
                        sharpe = (mean_pnl / std_pnl) * (252**0.5)
                except Exception:
                    pass

            # Calculate Sortino ratio
            sortino = 0.0
            if total_trades > 1:
                pnls = [t.pnl for t in self._closed_trades]
                downside_pnls = [p for p in pnls if p < 0]
                try:
                    mean_pnl = statistics.mean(pnls)
                    if downside_pnls:
                        downside_std = statistics.stdev(downside_pnls)
                        if downside_std > 0:
                            sortino = (mean_pnl / downside_std) * (252**0.5)
                except Exception:
                    pass

            total_equity = self.starting_equity + self._realized_pnl + self._unrealized_pnl
            current_dd = self.peak_equity - total_equity
            current_dd_pct = current_dd / self.peak_equity if self.peak_equity > 0 else 0.0

            return PerformanceSummary(
                total_trades=total_trades,
                winning_trades=winning,
                losing_trades=losing,
                win_rate=winning / total_trades if total_trades > 0 else 0.0,
                total_pnl=self._realized_pnl + self._unrealized_pnl,
                realized_pnl=self._realized_pnl,
                unrealized_pnl=self._unrealized_pnl,
                max_drawdown=self._max_drawdown,
                max_drawdown_pct=self._max_drawdown_pct,
                current_drawdown=current_dd,
                current_drawdown_pct=current_dd_pct,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                current_equity=total_equity,
                peak_equity=self.peak_equity,
                starting_equity=self.starting_equity,
                return_pct=(total_equity - self.starting_equity) / self.starting_equity
                if self.starting_equity > 0
                else 0.0,
            )

    def get_session_metrics(
        self,
        session_type: str = "daily",
        limit: int = 30,
    ) -> list[SessionMetrics]:
        """Get session metrics.

        Args:
            session_type: "daily", "weekly", or "monthly"
            limit: Maximum number of sessions to return

        Returns:
            List of SessionMetrics, most recent first
        """
        with self._lock:
            if session_type == "daily":
                sessions = list(self._daily_sessions.values())
            elif session_type == "weekly":
                sessions = list(self._weekly_sessions.values())
            else:
                sessions = list(self._monthly_sessions.values())

            return sorted(sessions, key=lambda s: s.session_date, reverse=True)[:limit]

    def get_strategy_metrics(self) -> dict[str, StrategyMetrics]:
        """Get metrics for all strategies.

        Returns:
            Dict mapping strategy name to StrategyMetrics
        """
        with self._lock:
            return dict(self._strategies)

    def get_open_trades(self) -> list[TradeRecord]:
        """Get all open trades.

        Returns:
            List of open TradeRecord objects
        """
        with self._lock:
            return list(self._open_trades.values())

    def get_closed_trades(self, limit: int = 100) -> list[TradeRecord]:
        """Get closed trades.

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of closed TradeRecord objects, most recent first
        """
        with self._lock:
            return sorted(
                self._closed_trades,
                key=lambda t: t.exit_time or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )[:limit]

    # =========================================================================
    # Event Listeners
    # =========================================================================

    def add_listener(
        self,
        listener: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """Add event listener.

        Args:
            listener: Callback function (event_type, data)
        """
        self._listeners.append(listener)

    def remove_listener(
        self,
        listener: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """Remove event listener.

        Args:
            listener: Callback function to remove
        """
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify(self, event_type: str, data: dict[str, Any]) -> None:
        """Notify all listeners of an event."""
        for listener in self._listeners:
            try:
                listener(event_type, data)
            except Exception:
                pass  # Don't let listener errors break tracking

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_to_object_store(self, key_prefix: str = "performance") -> bool:
        """Save current state to Object Store.

        Args:
            key_prefix: Prefix for storage keys

        Returns:
            True if successful
        """
        if not self.object_store:
            return False

        try:
            data = {
                "summary": self.get_summary().to_dict(),
                "strategies": {name: s.to_dict() for name, s in self._strategies.items()},
                "recent_sessions": [s.to_dict() for s in self.get_session_metrics("daily", 30)],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self.object_store.save(
                key=f"{key_prefix}/current_state.json",
                data=json.dumps(data),
                category="trading_state",
            )
            return True

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "save_performance_state")
            return False

    def load_from_object_store(self, key_prefix: str = "performance") -> bool:
        """Load state from Object Store.

        Args:
            key_prefix: Prefix for storage keys

        Returns:
            True if successful
        """
        if not self.object_store:
            return False

        try:
            data = self.object_store.load(f"{key_prefix}/current_state.json")
            if data:
                # Restore basic state from saved data
                summary = data.get("summary", {})
                self._realized_pnl = summary.get("realized_pnl", 0.0)
                self._max_drawdown = summary.get("max_drawdown", 0.0)
                self._max_drawdown_pct = summary.get("max_drawdown_pct", 0.0)
                self.peak_equity = summary.get("peak_equity", self.starting_equity)
                return True

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "load_performance_state")

        return False

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self.current_equity = self.starting_equity
            self.peak_equity = self.starting_equity
            self._open_trades.clear()
            self._closed_trades.clear()
            self._daily_sessions.clear()
            self._weekly_sessions.clear()
            self._monthly_sessions.clear()
            self._strategies.clear()
            self._unrealized_pnl = 0.0
            self._realized_pnl = 0.0
            self._max_drawdown = 0.0
            self._max_drawdown_pct = 0.0


# ============================================================================
# Factory Functions
# ============================================================================


def create_performance_tracker(
    starting_equity: float = 100000.0,
    logger: Any | None = None,
    object_store: Any | None = None,
) -> PerformanceTracker:
    """Create a performance tracker.

    Args:
        starting_equity: Initial portfolio equity
        logger: Optional StructuredLogger
        object_store: Optional ObjectStoreManager

    Returns:
        Configured PerformanceTracker
    """
    return PerformanceTracker(
        starting_equity=starting_equity,
        logger=logger,
        object_store=object_store,
    )
