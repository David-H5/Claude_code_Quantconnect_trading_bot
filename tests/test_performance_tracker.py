"""
Tests for Performance Tracker

Tests:
- Trade management (open, close)
- P&L calculations
- Session metrics
- Strategy metrics
- Drawdown tracking
- Event listeners
- Persistence

UPGRADE-010: Performance Tracker (December 2025)
"""

from datetime import date
from unittest.mock import MagicMock

import pytest

from models.performance_tracker import (
    PerformanceSummary,
    PerformanceTracker,
    SessionMetrics,
    StrategyMetrics,
    TradeRecord,
    create_performance_tracker,
)


# ============================================================================
# TRADE RECORD TESTS
# ============================================================================


class TestTradeRecord:
    """Tests for TradeRecord dataclass."""

    def test_trade_record_creation(self):
        """Test basic trade record creation."""
        trade = TradeRecord(
            symbol="SPY",
            strategy="iron_condor",
            direction="long",
            entry_price=450.0,
            quantity=10,
        )

        assert trade.symbol == "SPY"
        assert trade.strategy == "iron_condor"
        assert trade.direction == "long"
        assert trade.entry_price == 450.0
        assert trade.quantity == 10
        assert not trade.is_closed
        assert trade.trade_id is not None

    def test_trade_record_close_long_profit(self):
        """Test closing a long trade with profit."""
        trade = TradeRecord(
            symbol="SPY",
            direction="long",
            entry_price=450.0,
            quantity=10,
        )

        trade.close(exit_price=455.0, commission=5.0)

        assert trade.is_closed
        assert trade.exit_price == 455.0
        assert trade.pnl == (455.0 - 450.0) * 10 - 5.0  # 45.0
        assert trade.pnl > 0

    def test_trade_record_close_long_loss(self):
        """Test closing a long trade with loss."""
        trade = TradeRecord(
            symbol="SPY",
            direction="long",
            entry_price=450.0,
            quantity=10,
        )

        trade.close(exit_price=445.0, commission=5.0)

        assert trade.is_closed
        assert trade.pnl == (445.0 - 450.0) * 10 - 5.0  # -55.0
        assert trade.pnl < 0

    def test_trade_record_close_short(self):
        """Test closing a short trade."""
        trade = TradeRecord(
            symbol="SPY",
            direction="short",
            entry_price=450.0,
            quantity=10,
        )

        trade.close(exit_price=445.0)

        assert trade.is_closed
        # Short: (entry - exit) * quantity
        assert trade.pnl == (450.0 - 445.0) * 10  # 50.0
        assert trade.pnl > 0

    def test_trade_record_to_dict(self):
        """Test trade record to_dict conversion."""
        trade = TradeRecord(
            symbol="AAPL",
            strategy="covered_call",
            direction="long",
            entry_price=180.0,
            quantity=100,
        )

        d = trade.to_dict()

        assert d["symbol"] == "AAPL"
        assert d["strategy"] == "covered_call"
        assert d["entry_price"] == 180.0


# ============================================================================
# SESSION METRICS TESTS
# ============================================================================


class TestSessionMetrics:
    """Tests for SessionMetrics dataclass."""

    def test_session_metrics_creation(self):
        """Test session metrics creation."""
        session = SessionMetrics(
            session_date=date.today(),
            session_type="daily",
        )

        assert session.session_date == date.today()
        assert session.trades == 0
        assert session.net_pnl == 0.0

    def test_session_metrics_update(self):
        """Test session metrics update."""
        session = SessionMetrics(
            session_date=date.today(),
            trades=10,
            winning_trades=6,
            losing_trades=4,
        )
        session.trade_pnls = [100, 50, -30, 200, -50, 75, -20, 150, -40, 80]

        session.update_metrics()

        assert session.win_rate == 0.6
        assert session.profit_factor > 0

    def test_session_metrics_to_dict(self):
        """Test session metrics to_dict conversion."""
        session = SessionMetrics(
            session_date=date(2025, 12, 1),
            trades=5,
            net_pnl=250.0,
        )

        d = session.to_dict()

        assert d["session_date"] == "2025-12-01"
        assert d["trades"] == 5
        assert d["net_pnl"] == 250.0


# ============================================================================
# STRATEGY METRICS TESTS
# ============================================================================


class TestStrategyMetrics:
    """Tests for StrategyMetrics dataclass."""

    def test_strategy_metrics_creation(self):
        """Test strategy metrics creation."""
        strategy = StrategyMetrics(strategy_name="iron_condor")

        assert strategy.strategy_name == "iron_condor"
        assert strategy.trades == 0
        assert strategy.win_rate == 0.0

    def test_strategy_metrics_update(self):
        """Test strategy metrics update."""
        strategy = StrategyMetrics(
            strategy_name="bull_put_spread",
            trades=8,
            winning_trades=5,
            net_pnl=400.0,
        )
        strategy.trade_pnls = [100, 75, -50, 125, -30, 80, 50, 50]

        strategy.update_metrics()

        assert strategy.win_rate == 5 / 8
        assert strategy.avg_pnl == 50.0
        assert strategy.profit_factor > 0

    def test_strategy_metrics_to_dict(self):
        """Test strategy metrics to_dict conversion."""
        strategy = StrategyMetrics(
            strategy_name="covered_call",
            trades=20,
            win_rate=0.75,
        )

        d = strategy.to_dict()

        assert d["strategy_name"] == "covered_call"
        assert d["trades"] == 20


# ============================================================================
# PERFORMANCE TRACKER TESTS
# ============================================================================


class TestPerformanceTracker:
    """Tests for PerformanceTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a test tracker."""
        return PerformanceTracker(starting_equity=100000.0)

    def test_tracker_creation(self, tracker):
        """Test tracker creation."""
        assert tracker.starting_equity == 100000.0
        assert tracker.current_equity == 100000.0
        assert tracker.peak_equity == 100000.0

    def test_open_trade(self, tracker):
        """Test opening a trade."""
        trade = tracker.open_trade(
            symbol="SPY",
            strategy="iron_condor",
            direction="long",
            entry_price=450.0,
            quantity=10,
        )

        assert trade.symbol == "SPY"
        assert trade.strategy == "iron_condor"
        assert trade in tracker.get_open_trades()

    def test_close_trade(self, tracker):
        """Test closing a trade."""
        trade = tracker.open_trade(
            symbol="SPY",
            strategy="iron_condor",
            direction="long",
            entry_price=450.0,
            quantity=10,
        )

        closed = tracker.close_trade(trade.trade_id, exit_price=455.0)

        assert closed is not None
        assert closed.is_closed
        assert closed not in tracker.get_open_trades()
        assert closed in tracker.get_closed_trades()

    def test_close_nonexistent_trade(self, tracker):
        """Test closing a trade that doesn't exist."""
        result = tracker.close_trade("nonexistent", exit_price=100.0)

        assert result is None

    def test_pnl_tracking(self, tracker):
        """Test P&L tracking after trades."""
        trade = tracker.open_trade(
            symbol="SPY",
            strategy="test",
            direction="long",
            entry_price=100.0,
            quantity=100,
        )

        tracker.close_trade(trade.trade_id, exit_price=110.0)

        summary = tracker.get_summary()
        assert summary.realized_pnl == 1000.0  # (110-100) * 100
        assert summary.total_pnl == 1000.0

    def test_drawdown_tracking(self, tracker):
        """Test drawdown tracking."""
        # Make a winning trade to increase equity
        trade1 = tracker.open_trade(
            symbol="SPY",
            strategy="test",
            direction="long",
            entry_price=100.0,
            quantity=100,
        )
        tracker.close_trade(trade1.trade_id, exit_price=110.0)

        # Make a losing trade
        trade2 = tracker.open_trade(
            symbol="SPY",
            strategy="test",
            direction="long",
            entry_price=110.0,
            quantity=100,
        )
        tracker.close_trade(trade2.trade_id, exit_price=100.0)

        summary = tracker.get_summary()
        assert summary.max_drawdown > 0
        assert summary.max_drawdown_pct > 0

    def test_session_metrics_tracking(self, tracker):
        """Test session metrics are updated."""
        trade = tracker.open_trade(
            symbol="SPY",
            strategy="test",
            direction="long",
            entry_price=100.0,
            quantity=10,
        )
        tracker.close_trade(trade.trade_id, exit_price=105.0)

        sessions = tracker.get_session_metrics("daily")

        assert len(sessions) > 0
        assert sessions[0].trades == 1
        assert sessions[0].winning_trades == 1

    def test_strategy_metrics_tracking(self, tracker):
        """Test strategy metrics are updated."""
        trade = tracker.open_trade(
            symbol="SPY",
            strategy="iron_condor",
            direction="long",
            entry_price=100.0,
            quantity=10,
        )
        tracker.close_trade(trade.trade_id, exit_price=105.0)

        strategies = tracker.get_strategy_metrics()

        assert "iron_condor" in strategies
        assert strategies["iron_condor"].trades == 1
        assert strategies["iron_condor"].winning_trades == 1

    def test_multiple_strategies(self, tracker):
        """Test tracking multiple strategies."""
        # Iron condor trade
        t1 = tracker.open_trade(
            symbol="SPY",
            strategy="iron_condor",
            direction="long",
            entry_price=100.0,
            quantity=10,
        )
        tracker.close_trade(t1.trade_id, exit_price=105.0)

        # Bull put spread trade
        t2 = tracker.open_trade(
            symbol="SPY",
            strategy="bull_put_spread",
            direction="long",
            entry_price=100.0,
            quantity=10,
        )
        tracker.close_trade(t2.trade_id, exit_price=102.0)

        strategies = tracker.get_strategy_metrics()

        assert len(strategies) == 2
        assert "iron_condor" in strategies
        assert "bull_put_spread" in strategies

    def test_win_rate_calculation(self, tracker):
        """Test win rate calculation."""
        # 3 wins
        for _ in range(3):
            trade = tracker.open_trade(
                symbol="SPY",
                strategy="test",
                direction="long",
                entry_price=100.0,
                quantity=10,
            )
            tracker.close_trade(trade.trade_id, exit_price=105.0)

        # 2 losses
        for _ in range(2):
            trade = tracker.open_trade(
                symbol="SPY",
                strategy="test",
                direction="long",
                entry_price=100.0,
                quantity=10,
            )
            tracker.close_trade(trade.trade_id, exit_price=95.0)

        summary = tracker.get_summary()

        assert summary.total_trades == 5
        assert summary.winning_trades == 3
        assert summary.losing_trades == 2
        assert summary.win_rate == 0.6

    def test_unrealized_pnl_update(self, tracker):
        """Test unrealized P&L update."""
        tracker.update_unrealized_pnl(500.0)

        summary = tracker.get_summary()

        assert summary.unrealized_pnl == 500.0
        assert summary.total_pnl == 500.0

    def test_event_listener(self, tracker):
        """Test event listener."""
        events = []

        def listener(event_type, data):
            events.append((event_type, data))

        tracker.add_listener(listener)

        trade = tracker.open_trade(
            symbol="SPY",
            strategy="test",
            direction="long",
            entry_price=100.0,
            quantity=10,
        )
        tracker.close_trade(trade.trade_id, exit_price=105.0)

        assert len(events) == 2
        assert events[0][0] == "trade_opened"
        assert events[1][0] == "trade_closed"

    def test_remove_listener(self, tracker):
        """Test removing event listener."""
        events = []

        def listener(event_type, data):
            events.append((event_type, data))

        tracker.add_listener(listener)
        tracker.remove_listener(listener)

        tracker.open_trade(
            symbol="SPY",
            strategy="test",
            direction="long",
            entry_price=100.0,
            quantity=10,
        )

        assert len(events) == 0

    def test_reset(self, tracker):
        """Test tracker reset."""
        trade = tracker.open_trade(
            symbol="SPY",
            strategy="test",
            direction="long",
            entry_price=100.0,
            quantity=10,
        )
        tracker.close_trade(trade.trade_id, exit_price=105.0)

        tracker.reset()

        assert tracker.current_equity == tracker.starting_equity
        assert len(tracker.get_open_trades()) == 0
        assert len(tracker.get_closed_trades()) == 0
        assert len(tracker.get_strategy_metrics()) == 0


# ============================================================================
# LOGGER INTEGRATION TESTS
# ============================================================================


class TestLoggerIntegration:
    """Tests for StructuredLogger integration."""

    def test_logs_trade_opened(self):
        """Test that opening trade is logged."""
        mock_logger = MagicMock()
        tracker = PerformanceTracker(logger=mock_logger)

        tracker.open_trade(
            symbol="SPY",
            strategy="test",
            direction="long",
            entry_price=100.0,
            quantity=10,
        )

        mock_logger.log_position_opened.assert_called_once()

    def test_logs_trade_closed(self):
        """Test that closing trade is logged."""
        mock_logger = MagicMock()
        tracker = PerformanceTracker(logger=mock_logger)

        trade = tracker.open_trade(
            symbol="SPY",
            strategy="test",
            direction="long",
            entry_price=100.0,
            quantity=10,
        )
        tracker.close_trade(trade.trade_id, exit_price=105.0)

        mock_logger.log_position_closed.assert_called_once()


# ============================================================================
# PERSISTENCE TESTS
# ============================================================================


class TestPersistence:
    """Tests for Object Store persistence."""

    def test_save_to_object_store(self):
        """Test saving to Object Store."""
        mock_store = MagicMock()
        tracker = PerformanceTracker(object_store=mock_store)

        trade = tracker.open_trade(
            symbol="SPY",
            strategy="test",
            direction="long",
            entry_price=100.0,
            quantity=10,
        )
        tracker.close_trade(trade.trade_id, exit_price=105.0)

        result = tracker.save_to_object_store()

        assert result is True
        mock_store.save.assert_called_once()

    def test_save_without_object_store(self):
        """Test save without Object Store returns False."""
        tracker = PerformanceTracker()

        result = tracker.save_to_object_store()

        assert result is False

    def test_load_from_object_store(self):
        """Test loading from Object Store."""
        mock_store = MagicMock()
        mock_store.load.return_value = {
            "summary": {
                "realized_pnl": 5000.0,
                "max_drawdown": 1000.0,
                "max_drawdown_pct": 0.01,
                "peak_equity": 105000.0,
            }
        }
        tracker = PerformanceTracker(object_store=mock_store)

        result = tracker.load_from_object_store()

        assert result is True
        assert tracker._realized_pnl == 5000.0
        assert tracker.peak_equity == 105000.0


# ============================================================================
# FACTORY FUNCTION TESTS
# ============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_performance_tracker(self):
        """Test create_performance_tracker factory."""
        tracker = create_performance_tracker(
            starting_equity=50000.0,
        )

        assert tracker.starting_equity == 50000.0
        assert tracker.current_equity == 50000.0

    def test_create_with_logger(self):
        """Test creating tracker with logger."""
        mock_logger = MagicMock()
        tracker = create_performance_tracker(logger=mock_logger)

        assert tracker.logger == mock_logger


# ============================================================================
# SUMMARY TESTS
# ============================================================================


class TestPerformanceSummary:
    """Tests for PerformanceSummary."""

    def test_summary_to_dict(self):
        """Test summary to_dict conversion."""
        summary = PerformanceSummary(
            total_trades=100,
            winning_trades=60,
            win_rate=0.60,
            total_pnl=5000.0,
            sharpe_ratio=1.5,
        )

        d = summary.to_dict()

        assert d["total_trades"] == 100
        assert d["win_rate"] == 0.60
        assert d["sharpe_ratio"] == 1.5

    def test_summary_from_tracker(self):
        """Test getting summary from tracker."""
        tracker = PerformanceTracker(starting_equity=100000.0)

        # Make some trades
        for i in range(5):
            trade = tracker.open_trade(
                symbol="SPY",
                strategy="test",
                direction="long",
                entry_price=100.0,
                quantity=10,
            )
            exit_price = 105.0 if i % 2 == 0 else 95.0
            tracker.close_trade(trade.trade_id, exit_price=exit_price)

        summary = tracker.get_summary()

        assert summary.total_trades == 5
        assert summary.winning_trades == 3
        assert summary.losing_trades == 2
        assert summary.starting_equity == 100000.0
