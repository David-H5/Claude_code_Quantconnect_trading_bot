"""
Tests for Bot-Managed Positions

Tests for automatic profit-taking, stop-loss, and position management.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from execution.bot_managed_positions import (
    BotManagedPosition,
    ManagementAction,
    PositionSource,
    ProfitThreshold,
    create_bot_position_manager,
)


class TestBotPositionManagerInitialization:
    """Tests for BotPositionManager initialization."""

    @pytest.mark.unit
    def test_manager_creation(self):
        """Test manager instance creation."""
        algorithm = Mock()
        manager = create_bot_position_manager(algorithm, enable_logging=False)

        assert manager.algorithm == algorithm
        assert manager.stop_loss_threshold == -2.00
        assert manager.min_dte_for_roll == 7
        assert len(manager.default_profit_thresholds) == 3

    @pytest.mark.unit
    def test_custom_thresholds(self):
        """Test custom profit thresholds."""
        algorithm = Mock()
        manager = create_bot_position_manager(
            algorithm,
            profit_thresholds=[(0.25, 0.50), (0.75, 0.50)],
            enable_logging=False,
        )

        assert len(manager.default_profit_thresholds) == 2
        assert manager.default_profit_thresholds[0].gain_pct == 0.25
        assert manager.default_profit_thresholds[0].take_pct == 0.50


class TestAddPosition:
    """Tests for adding positions to management."""

    @pytest.fixture
    def manager(self):
        """Create manager instance for testing."""
        algorithm = Mock()
        algorithm.Debug = Mock()
        return create_bot_position_manager(algorithm, enable_logging=False)

    @pytest.mark.unit
    def test_add_autonomous_position(self, manager):
        """Test adding an autonomous position."""
        position = manager.add_position(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            entry_price=-500.0,
            quantity=1,
            strategy_type="iron_condor",
            legs=[],
        )

        assert position.position_id == "pos_1"
        assert position.source == PositionSource.AUTONOMOUS
        assert position.management_enabled is True
        assert len(manager.positions) == 1
        assert manager.stats["total_positions"] == 1

    @pytest.mark.unit
    def test_add_manual_ui_position(self, manager):
        """Test adding a manual UI position."""
        position = manager.add_position(
            position_id="pos_2",
            symbol="AAPL",
            source=PositionSource.MANUAL_UI,
            entry_price=-300.0,
            quantity=2,
            strategy_type="butterfly_call",
            legs=[],
        )

        assert position.source == PositionSource.MANUAL_UI
        assert position.current_quantity == 2

    @pytest.mark.unit
    def test_add_with_custom_thresholds(self, manager):
        """Test adding position with custom thresholds."""
        position = manager.add_position(
            position_id="pos_3",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            entry_price=-500.0,
            quantity=1,
            strategy_type="iron_condor",
            legs=[],
            custom_thresholds=[(0.30, 1.00)],  # 100% at +30%
        )

        assert len(position.profit_thresholds) == 1
        assert position.profit_thresholds[0].gain_pct == 0.30
        assert position.profit_thresholds[0].take_pct == 1.00

    @pytest.mark.unit
    def test_position_tracking_by_symbol(self, manager):
        """Test that positions are tracked by symbol."""
        manager.add_position(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            entry_price=-500.0,
            quantity=1,
            strategy_type="iron_condor",
            legs=[],
        )

        manager.add_position(
            position_id="pos_2",
            symbol="SPY",
            source=PositionSource.MANUAL_UI,
            entry_price=-300.0,
            quantity=1,
            strategy_type="butterfly",
            legs=[],
        )

        spy_positions = manager.get_positions_by_symbol("SPY")
        assert len(spy_positions) == 2


class TestPnLCalculation:
    """Tests for P&L calculation."""

    @pytest.mark.unit
    def test_profit_calculation(self):
        """Test P&L calculation for profitable position."""
        position = BotManagedPosition(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            entry_price=-500.0,  # Paid $500 debit
            entry_time=datetime.now(),
            current_quantity=1,
            strategy_type="iron_condor",
            legs=[],
            profit_thresholds=[],
            stop_loss_threshold=-2.00,
            min_dte_for_roll=7,
        )

        # Position now costs $250 to close (50% gain)
        # Entry: paid $500, Current: costs $250
        # Gain: $500 - $250 = $250 (50% of $500)
        pnl_pct = position.calculate_pnl_pct(-250.0)
        assert abs(pnl_pct - 0.50) < 0.01  # +50%

    @pytest.mark.unit
    def test_loss_calculation(self):
        """Test P&L calculation for losing position."""
        position = BotManagedPosition(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            entry_price=-500.0,  # Paid $500 debit
            entry_time=datetime.now(),
            current_quantity=1,
            strategy_type="iron_condor",
            legs=[],
            profit_thresholds=[],
            stop_loss_threshold=-2.00,
            min_dte_for_roll=7,
        )

        # Position now costs $1500 to close (200% loss)
        # Entry: paid $500, Current: costs $1500
        # Loss: $500 - $1500 = -$1000 (200% loss of $500)
        pnl_pct = position.calculate_pnl_pct(-1500.0)
        assert abs(pnl_pct - (-2.00)) < 0.01  # -200%


class TestProfitTaking:
    """Tests for profit-taking logic."""

    @pytest.fixture
    def manager(self):
        """Create manager instance for testing."""
        algorithm = Mock()
        algorithm.Debug = Mock()
        return create_bot_position_manager(
            algorithm,
            profit_thresholds=[
                (0.50, 0.30),  # 30% at +50%
                (1.00, 0.50),  # 50% at +100%
            ],
            enable_logging=False,
        )

    @pytest.mark.unit
    def test_profit_threshold_triggering(self, manager):
        """Test that profit thresholds trigger correctly."""
        position = manager.add_position(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            entry_price=-500.0,
            quantity=10,
            strategy_type="iron_condor",
            legs=[],
        )

        # Simulate +50% gain
        pnl_pct = 0.50
        action = manager._check_profit_thresholds(position, pnl_pct)

        # First threshold should trigger
        assert action == ManagementAction.TAKE_PROFIT
        assert position.profit_thresholds[0].triggered is True
        # Should close 30% (3 contracts)
        assert position.current_quantity == 7

    @pytest.mark.unit
    def test_multiple_thresholds(self, manager):
        """Test multiple profit thresholds."""
        position = manager.add_position(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            entry_price=-500.0,
            quantity=10,
            strategy_type="iron_condor",
            legs=[],
        )

        # First threshold at +50%
        manager._check_profit_thresholds(position, 0.50)
        assert position.current_quantity == 7

        # Second threshold at +100%
        manager._check_profit_thresholds(position, 1.00)
        # Should close 50% of remaining (3-4 contracts)
        assert position.current_quantity <= 4

    @pytest.mark.unit
    def test_already_triggered_threshold(self, manager):
        """Test that already triggered thresholds don't trigger again."""
        position = manager.add_position(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            entry_price=-500.0,
            quantity=10,
            strategy_type="iron_condor",
            legs=[],
        )

        # Trigger first threshold
        manager._check_profit_thresholds(position, 0.50)
        quantity_after_first = position.current_quantity

        # Try to trigger again
        manager._check_profit_thresholds(position, 0.50)

        # Quantity should not change
        assert position.current_quantity == quantity_after_first


class TestStopLoss:
    """Tests for stop-loss logic."""

    @pytest.fixture
    def manager(self):
        """Create manager instance for testing."""
        algorithm = Mock()
        algorithm.Debug = Mock()
        return create_bot_position_manager(
            algorithm,
            stop_loss_threshold=-2.00,  # -200%
            enable_logging=False,
        )

    @pytest.mark.unit
    def test_stop_loss_execution(self, manager):
        """Test stop loss closes entire position."""
        position = manager.add_position(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            entry_price=-500.0,
            quantity=5,
            strategy_type="iron_condor",
            legs=[],
        )

        # Execute stop loss
        action = manager._execute_stop_loss(position)

        assert action == ManagementAction.STOP_LOSS
        assert "pos_1" not in manager.positions
        assert manager.stats["stop_losses"] == 1

    @pytest.mark.unit
    def test_stop_loss_callback(self, manager):
        """Test stop loss callback is called."""
        callback_called = False

        def on_stop_loss(position):
            nonlocal callback_called
            callback_called = True

        manager.on_stop_loss = on_stop_loss

        position = manager.add_position(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            entry_price=-500.0,
            quantity=5,
            strategy_type="iron_condor",
            legs=[],
        )

        manager._execute_stop_loss(position)

        assert callback_called is True


class TestRolling:
    """Tests for position rolling."""

    @pytest.fixture
    def manager(self):
        """Create manager instance for testing."""
        algorithm = Mock()
        algorithm.Debug = Mock()
        return create_bot_position_manager(
            algorithm,
            min_dte_for_roll=7,
            enable_logging=False,
        )

    @pytest.mark.unit
    def test_roll_execution(self, manager):
        """Test position rolling."""
        position = manager.add_position(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            entry_price=-500.0,
            quantity=5,
            strategy_type="iron_condor",
            legs=[],
        )

        # Execute roll
        action = manager._execute_roll(position)

        assert action == ManagementAction.ROLL
        assert manager.stats["rolls"] == 1
        assert len(position.management_history) == 1


class TestPositionManagement:
    """Tests for overall position management."""

    @pytest.fixture
    def manager(self):
        """Create manager instance for testing."""
        algorithm = Mock()
        algorithm.Debug = Mock()
        return create_bot_position_manager(algorithm, enable_logging=False)

    @pytest.mark.unit
    def test_disable_management(self, manager):
        """Test disabling bot management for a position."""
        position = manager.add_position(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.MANUAL_UI,
            entry_price=-500.0,
            quantity=1,
            strategy_type="iron_condor",
            legs=[],
        )

        # Disable management
        success = manager.disable_management("pos_1")

        assert success is True
        assert position.management_enabled is False

    @pytest.mark.unit
    def test_enable_management(self, manager):
        """Test enabling bot management for a position."""
        position = manager.add_position(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.MANUAL_UI,
            entry_price=-500.0,
            quantity=1,
            strategy_type="iron_condor",
            legs=[],
            management_enabled=False,
        )

        # Enable management
        success = manager.enable_management("pos_1")

        assert success is True
        assert position.management_enabled is True

    @pytest.mark.unit
    def test_get_all_positions(self, manager):
        """Test getting all positions."""
        for i in range(5):
            manager.add_position(
                position_id=f"pos_{i}",
                symbol="SPY",
                source=PositionSource.AUTONOMOUS,
                entry_price=-500.0,
                quantity=1,
                strategy_type="iron_condor",
                legs=[],
            )

        positions = manager.get_all_positions()
        assert len(positions) == 5


class TestStatistics:
    """Tests for management statistics."""

    @pytest.fixture
    def manager(self):
        """Create manager instance for testing."""
        algorithm = Mock()
        algorithm.Debug = Mock()
        return create_bot_position_manager(algorithm, enable_logging=False)

    @pytest.mark.unit
    def test_statistics_tracking(self, manager):
        """Test that statistics are tracked correctly."""
        # Add positions
        for i in range(3):
            manager.add_position(
                position_id=f"pos_{i}",
                symbol="SPY",
                source=PositionSource.AUTONOMOUS,
                entry_price=-500.0,
                quantity=10,
                strategy_type="iron_condor",
                legs=[],
            )

        # Trigger profit-taking on one
        position = manager.get_position("pos_0")
        manager._check_profit_thresholds(position, 0.50)

        # Execute stop loss on another
        position = manager.get_position("pos_1")
        manager._execute_stop_loss(position)

        stats = manager.get_statistics()

        assert stats["total_positions"] == 3
        assert stats["active_positions"] == 2  # One closed by stop loss
        assert stats["profit_takes"] == 1
        assert stats["stop_losses"] == 1


class TestManagementHistory:
    """Tests for management history tracking."""

    @pytest.mark.unit
    def test_action_recording(self):
        """Test that actions are recorded in history."""
        position = BotManagedPosition(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            entry_price=-500.0,
            entry_time=datetime.now(),
            current_quantity=10,
            strategy_type="iron_condor",
            legs=[],
            profit_thresholds=[],
            stop_loss_threshold=-2.00,
            min_dte_for_roll=7,
        )

        # Record some actions
        position.record_action(
            ManagementAction.TAKE_PROFIT,
            {"quantity_closed": 3, "reason": "Test"},
        )

        position.record_action(
            ManagementAction.ROLL,
            {"reason": "Low DTE"},
        )

        assert len(position.management_history) == 2
        assert position.management_history[0]["action"] == "take_profit"
        assert position.management_history[1]["action"] == "roll"


class TestPositionSerialization:
    """Tests for position serialization."""

    @pytest.mark.unit
    def test_position_to_dict(self):
        """Test converting position to dictionary."""
        position = BotManagedPosition(
            position_id="pos_1",
            symbol="SPY",
            source=PositionSource.AUTONOMOUS,
            entry_price=-500.0,
            entry_time=datetime.now(),
            current_quantity=10,
            strategy_type="iron_condor",
            legs=[],
            profit_thresholds=[
                ProfitThreshold(0.50, 0.30),
                ProfitThreshold(1.00, 0.50),
            ],
            stop_loss_threshold=-2.00,
            min_dte_for_roll=7,
        )

        position_dict = position.to_dict()

        assert position_dict["position_id"] == "pos_1"
        assert position_dict["symbol"] == "SPY"
        assert position_dict["source"] == "autonomous"
        assert len(position_dict["profit_thresholds"]) == 2
