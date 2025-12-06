"""
Tests for Position Tracker UI

Tests for unified position tracking widget with real-time updates.
"""

from datetime import datetime

import pytest


# Check if PySide6 is available
try:
    import sys

    from PySide6.QtWidgets import QApplication

    # Create QApplication instance for testing (required for Qt widgets)
    if not QApplication.instance():
        app = QApplication(sys.argv)

    from ui.position_tracker import (
        PositionData,
        PositionGreeks,
        PositionSource,
        PositionTrackerWidget,
        create_position_tracker,
    )

    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False
    pytestmark = pytest.mark.skip("PySide6 not available")


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestPositionTrackerInitialization:
    """Tests for PositionTrackerWidget initialization."""

    @pytest.mark.unit
    def test_widget_creation(self):
        """Test widget instance creation."""
        tracker = create_position_tracker()

        assert tracker is not None
        assert len(tracker.positions) == 0
        assert tracker.stats["total_positions"] == 0

    @pytest.mark.unit
    def test_initial_statistics(self):
        """Test initial statistics."""
        tracker = PositionTrackerWidget()

        stats = tracker.get_statistics()

        assert stats["total_positions"] == 0
        assert stats["total_pnl"] == 0.0
        assert stats["portfolio_delta"] == 0.0


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestPositionData:
    """Tests for PositionData class."""

    @pytest.mark.unit
    def test_pnl_calculation(self):
        """Test P&L calculation."""
        position = PositionData(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            strategy_type="iron_condor",
            entry_price=-500.0,
            current_value=-250.0,
            entry_time=datetime.now(),
            quantity=1,
        )

        # 50% gain
        assert abs(position.pnl - 0.50) < 0.01

    @pytest.mark.unit
    def test_pnl_dollars_calculation(self):
        """Test P&L in dollars."""
        position = PositionData(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            strategy_type="iron_condor",
            entry_price=-500.0,
            current_value=-250.0,
            entry_time=datetime.now(),
            quantity=2,
        )

        # ($250 gain) * 2 contracts = $500
        assert abs(position.pnl_dollars - 500.0) < 0.01


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestAddPosition:
    """Tests for adding positions."""

    @pytest.mark.unit
    def test_add_single_position(self):
        """Test adding a single position."""
        tracker = PositionTrackerWidget()

        position = PositionData(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            strategy_type="iron_condor",
            entry_price=-500.0,
            current_value=-250.0,
            entry_time=datetime.now(),
            quantity=1,
        )

        tracker.add_position(position)

        assert len(tracker.positions) == 1
        assert tracker.stats["total_positions"] == 1
        assert "pos_1" in tracker.positions

    @pytest.mark.unit
    def test_add_multiple_positions(self):
        """Test adding multiple positions."""
        tracker = PositionTrackerWidget()

        for i in range(5):
            position = PositionData(
                position_id=f"pos_{i}",
                symbol="SPY",
                source=PositionSource.AUTONOMOUS,
                strategy_type="iron_condor",
                entry_price=-500.0,
                current_value=-250.0,
                entry_time=datetime.now(),
                quantity=1,
            )
            tracker.add_position(position)

        assert len(tracker.positions) == 5
        assert tracker.stats["total_positions"] == 5

    @pytest.mark.unit
    def test_add_positions_from_different_sources(self):
        """Test adding positions from different sources."""
        tracker = PositionTrackerWidget()

        # Autonomous
        tracker.add_position(
            PositionData(
                position_id="auto_1",
                symbol="SPY",
                source=PositionSource.AUTONOMOUS,
                strategy_type="iron_condor",
                entry_price=-500.0,
                current_value=-250.0,
                entry_time=datetime.now(),
                quantity=1,
            )
        )

        # Manual UI
        tracker.add_position(
            PositionData(
                position_id="manual_1",
                symbol="AAPL",
                source=PositionSource.MANUAL_UI,
                strategy_type="butterfly",
                entry_price=-300.0,
                current_value=-150.0,
                entry_time=datetime.now(),
                quantity=1,
            )
        )

        # Recurring
        tracker.add_position(
            PositionData(
                position_id="recurring_1",
                symbol="QQQ",
                source=PositionSource.RECURRING,
                strategy_type="iron_condor",
                entry_price=-400.0,
                current_value=-200.0,
                entry_time=datetime.now(),
                quantity=1,
            )
        )

        assert len(tracker.positions) == 3
        assert len(tracker.get_positions_by_source(PositionSource.AUTONOMOUS)) == 1
        assert len(tracker.get_positions_by_source(PositionSource.MANUAL_UI)) == 1
        assert len(tracker.get_positions_by_source(PositionSource.RECURRING)) == 1


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestUpdatePosition:
    """Tests for updating positions."""

    @pytest.mark.unit
    def test_update_current_value(self):
        """Test updating position current value."""
        tracker = PositionTrackerWidget()

        position = PositionData(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            strategy_type="iron_condor",
            entry_price=-500.0,
            current_value=-250.0,
            entry_time=datetime.now(),
            quantity=1,
        )
        tracker.add_position(position)

        # Update value
        tracker.update_position("pos_1", current_value=-200.0)

        updated = tracker.get_position("pos_1")
        assert updated.current_value == -200.0

    @pytest.mark.unit
    def test_update_greeks(self):
        """Test updating position Greeks."""
        tracker = PositionTrackerWidget()

        position = PositionData(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            strategy_type="iron_condor",
            entry_price=-500.0,
            current_value=-250.0,
            entry_time=datetime.now(),
            quantity=1,
        )
        tracker.add_position(position)

        # Update Greeks
        new_greeks = PositionGreeks(delta=0.25, theta=-5.0)
        tracker.update_position("pos_1", greeks=new_greeks)

        updated = tracker.get_position("pos_1")
        assert updated.greeks.delta == 0.25
        assert updated.greeks.theta == -5.0

    @pytest.mark.unit
    def test_update_quantity(self):
        """Test updating position quantity."""
        tracker = PositionTrackerWidget()

        position = PositionData(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            strategy_type="iron_condor",
            entry_price=-500.0,
            current_value=-250.0,
            entry_time=datetime.now(),
            quantity=10,
        )
        tracker.add_position(position)

        # Update quantity (partial close)
        tracker.update_position("pos_1", quantity=7)

        updated = tracker.get_position("pos_1")
        assert updated.quantity == 7


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestRemovePosition:
    """Tests for removing positions."""

    @pytest.mark.unit
    def test_remove_position(self):
        """Test removing a position."""
        tracker = PositionTrackerWidget()

        position = PositionData(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            strategy_type="iron_condor",
            entry_price=-500.0,
            current_value=-250.0,
            entry_time=datetime.now(),
            quantity=1,
        )
        tracker.add_position(position)

        assert len(tracker.positions) == 1

        tracker.remove_position("pos_1")

        assert len(tracker.positions) == 0
        assert tracker.stats["total_positions"] == 0

    @pytest.mark.unit
    def test_remove_nonexistent_position(self):
        """Test removing a position that doesn't exist."""
        tracker = PositionTrackerWidget()

        # Should not raise error
        tracker.remove_position("nonexistent")

        assert len(tracker.positions) == 0


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestPositionRetrieval:
    """Tests for position retrieval methods."""

    @pytest.fixture
    def tracker_with_positions(self):
        """Create tracker with sample positions."""
        tracker = PositionTrackerWidget()

        # Add various positions
        tracker.add_position(
            PositionData(
                position_id="spy_auto_1",
                symbol="SPY",
                source=PositionSource.AUTONOMOUS,
                strategy_type="iron_condor",
                entry_price=-500.0,
                current_value=-250.0,
                entry_time=datetime.now(),
                quantity=1,
            )
        )

        tracker.add_position(
            PositionData(
                position_id="spy_manual_1",
                symbol="SPY",
                source=PositionSource.MANUAL_UI,
                strategy_type="butterfly",
                entry_price=-300.0,
                current_value=-150.0,
                entry_time=datetime.now(),
                quantity=1,
            )
        )

        tracker.add_position(
            PositionData(
                position_id="aapl_auto_1",
                symbol="AAPL",
                source=PositionSource.AUTONOMOUS,
                strategy_type="iron_condor",
                entry_price=-400.0,
                current_value=-200.0,
                entry_time=datetime.now(),
                quantity=1,
            )
        )

        return tracker

    @pytest.mark.unit
    def test_get_all_positions(self, tracker_with_positions):
        """Test getting all positions."""
        positions = tracker_with_positions.get_all_positions()

        assert len(positions) == 3

    @pytest.mark.unit
    def test_get_positions_by_source(self, tracker_with_positions):
        """Test filtering by source."""
        auto_positions = tracker_with_positions.get_positions_by_source(PositionSource.AUTONOMOUS)
        manual_positions = tracker_with_positions.get_positions_by_source(PositionSource.MANUAL_UI)

        assert len(auto_positions) == 2
        assert len(manual_positions) == 1

    @pytest.mark.unit
    def test_get_positions_by_symbol(self, tracker_with_positions):
        """Test filtering by symbol."""
        spy_positions = tracker_with_positions.get_positions_by_symbol("SPY")
        aapl_positions = tracker_with_positions.get_positions_by_symbol("AAPL")

        assert len(spy_positions) == 2
        assert len(aapl_positions) == 1


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestPortfolioStatistics:
    """Tests for portfolio statistics calculation."""

    @pytest.mark.unit
    def test_portfolio_pnl_aggregation(self):
        """Test portfolio P&L aggregation."""
        tracker = PositionTrackerWidget()

        # Position 1: +50% ($250)
        tracker.add_position(
            PositionData(
                position_id="pos_1",
                symbol="SPY",
                source=PositionSource.AUTONOMOUS,
                strategy_type="iron_condor",
                entry_price=-500.0,
                current_value=-250.0,
                entry_time=datetime.now(),
                quantity=1,
            )
        )

        # Position 2: +100% ($300)
        tracker.add_position(
            PositionData(
                position_id="pos_2",
                symbol="AAPL",
                source=PositionSource.MANUAL_UI,
                strategy_type="butterfly",
                entry_price=-300.0,
                current_value=0.0,
                entry_time=datetime.now(),
                quantity=1,
            )
        )

        # Force update
        tracker._update_summary()

        stats = tracker.get_statistics()

        # Total P&L in dollars: $250 + $300 = $550
        assert abs(stats["total_pnl_dollars"] - 550.0) < 0.01

    @pytest.mark.unit
    def test_portfolio_greeks_aggregation(self):
        """Test portfolio Greeks aggregation."""
        tracker = PositionTrackerWidget()

        # Position 1
        tracker.add_position(
            PositionData(
                position_id="pos_1",
                symbol="SPY",
                source=PositionSource.AUTONOMOUS,
                strategy_type="iron_condor",
                entry_price=-500.0,
                current_value=-250.0,
                entry_time=datetime.now(),
                quantity=1,
                greeks=PositionGreeks(delta=0.10, theta=-2.0),
            )
        )

        # Position 2
        tracker.add_position(
            PositionData(
                position_id="pos_2",
                symbol="AAPL",
                source=PositionSource.MANUAL_UI,
                strategy_type="butterfly",
                entry_price=-300.0,
                current_value=-150.0,
                entry_time=datetime.now(),
                quantity=2,  # 2 contracts
                greeks=PositionGreeks(delta=0.15, theta=-3.0),
            )
        )

        # Force update
        tracker._update_summary()

        stats = tracker.get_statistics()

        # Portfolio delta: (0.10 * 1) + (0.15 * 2) = 0.40
        assert abs(stats["portfolio_delta"] - 0.40) < 0.01

        # Portfolio theta: (-2.0 * 1) + (-3.0 * 2) = -8.0
        assert abs(stats["portfolio_theta"] - (-8.0)) < 0.01


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestCallbacks:
    """Tests for callback functionality."""

    @pytest.mark.unit
    def test_close_position_callback(self):
        """Test close position callback."""
        tracker = PositionTrackerWidget()

        callback_called = False
        callback_position_id = None

        def on_close(position_id):
            nonlocal callback_called, callback_position_id
            callback_called = True
            callback_position_id = position_id

        tracker.on_close_position = on_close

        position = PositionData(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            strategy_type="iron_condor",
            entry_price=-500.0,
            current_value=-250.0,
            entry_time=datetime.now(),
            quantity=1,
        )
        tracker.add_position(position)

        # Simulate callback (would be triggered by UI in real usage)
        if tracker.on_close_position:
            tracker.on_close_position("pos_1")

        assert callback_called is True
        assert callback_position_id == "pos_1"


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestTableDisplay:
    """Tests for table display functionality."""

    @pytest.mark.unit
    def test_table_row_count(self):
        """Test table has correct number of rows."""
        tracker = PositionTrackerWidget()

        # Add 3 positions
        for i in range(3):
            tracker.add_position(
                PositionData(
                    position_id=f"pos_{i}",
                    symbol="SPY",
                    source=PositionSource.AUTONOMOUS,
                    strategy_type="iron_condor",
                    entry_price=-500.0,
                    current_value=-250.0,
                    entry_time=datetime.now(),
                    quantity=1,
                )
            )

        # Force table update
        tracker._update_table()

        assert tracker.table.rowCount() == 3

    @pytest.mark.unit
    def test_table_column_count(self):
        """Test table has correct number of columns."""
        tracker = PositionTrackerWidget()

        # Should have 12 columns
        assert tracker.table.columnCount() == 12
