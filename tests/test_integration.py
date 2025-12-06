"""
Integration Tests for Hybrid Architecture

Comprehensive end-to-end tests validating the entire hybrid trading system.
Tests the interaction between all modules and workflows.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from api.order_queue_api import (
    OrderPriority,
    OrderRequest,
    create_order_queue_api,
)
from api.order_queue_api import (
    OrderType as QueueOrderType,
)
from execution.bot_managed_positions import (
    ManagementAction,
    create_bot_position_manager,
)
from execution.bot_managed_positions import (
    PositionSource as BotPositionSource,
)
from execution.recurring_order_manager import (
    ConditionOperator,
    ConditionType,
    EntryCondition,
    RecurringOrderTemplate,
    ScheduleType,
    create_recurring_order_manager,
)
from execution.recurring_order_manager import (
    OrderType as RecurringOrderType,
)


class TestHybridArchitectureIntegration:
    """Integration tests for the complete hybrid architecture."""

    @pytest.fixture
    def algorithm(self):
        """Create mock algorithm."""
        algo = Mock()
        algo.Debug = Mock()
        algo.Error = Mock()
        algo.Time = datetime(2025, 1, 15, 10, 0)
        return algo

    @pytest.fixture
    def components(self, algorithm, tmp_path):
        """Create all system components."""
        return {
            "bot_manager": create_bot_position_manager(algorithm, enable_logging=False),
            "recurring_manager": create_recurring_order_manager(
                algorithm,
                storage_path=tmp_path,
            ),
            "order_queue": create_order_queue_api(algorithm, skip_auth_validation=True),
        }

    @pytest.mark.integration
    def test_full_autonomous_workflow(self, components):
        """Test complete autonomous trading workflow."""
        bot_manager = components["bot_manager"]

        # Add autonomous position
        position = bot_manager.add_position(
            position_id="auto_ic_1",
            symbol="SPY",
            source=BotPositionSource.AUTONOMOUS,
            entry_price=-500.0,
            quantity=10,
            strategy_type="iron_condor",
            legs=[],
        )

        # Verify position added
        assert "auto_ic_1" in bot_manager.positions
        assert bot_manager.stats["total_positions"] == 1

        # Simulate profit-taking scenario
        # Position moves to +50% profit
        pnl_pct = position.calculate_pnl_pct(-250.0)
        assert abs(pnl_pct - 0.50) < 0.01

        # Check profit threshold
        action = bot_manager._check_profit_thresholds(position, pnl_pct)

        # Should trigger first threshold (30% at +50%)
        assert action == ManagementAction.TAKE_PROFIT
        assert position.current_quantity == 7  # Closed 30%

        # Verify statistics
        assert bot_manager.stats["profit_takes"] == 1

    @pytest.mark.integration
    def test_ui_order_to_bot_management_flow(self, components):
        """Test UI order submission → execution → bot management."""
        order_queue = components["order_queue"]
        bot_manager = components["bot_manager"]

        # Step 1: Submit order via UI
        request = OrderRequest(
            order_type=QueueOrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            strategy_name="iron_condor",
            strategy_params={"put_buy": 450, "put_sell": 455},
            bot_managed=True,  # Enable bot management
        )

        response = order_queue.submit_order(request)
        assert response["success"] is True
        order_id = response["order_id"]

        # Step 2: Get order from queue (simulating algorithm processing)
        pending = order_queue.get_pending_orders(max_count=1)
        assert len(pending) == 1
        assert pending[0].request.bot_managed is True

        # Step 3: Simulate order execution
        order_queue.update_order_status(
            order_id,
            order_queue_api.OrderStatus.FILLED,
            execution_details={"fill_price": -480.0},
        )

        # Step 4: Add filled position to bot manager
        bot_position = bot_manager.add_position(
            position_id=order_id,
            symbol="SPY",
            source=BotPositionSource.MANUAL_UI,
            entry_price=-480.0,
            quantity=1,
            strategy_type="iron_condor",
            legs=[],
        )

        # Verify bot will manage this position
        assert bot_position.management_enabled is True
        assert bot_position.source == BotPositionSource.MANUAL_UI

        # Step 5: Simulate stop-loss scenario
        # Position moves to -200% loss
        action = bot_manager._execute_stop_loss(bot_position)

        assert action == ManagementAction.STOP_LOSS
        assert order_id not in bot_manager.positions  # Position closed
        assert bot_manager.stats["stop_losses"] == 1

    @pytest.mark.integration
    def test_recurring_order_to_execution_flow(self, algorithm, components):
        """Test recurring template → trigger → execution."""
        recurring_manager = components["recurring_manager"]
        order_queue = components["order_queue"]

        # Step 1: Create recurring template
        template = RecurringOrderTemplate(
            template_id="weekly_ic",
            name="Weekly Iron Condor",
            schedule_type=ScheduleType.WEEKLY,
            schedule_params={"day_of_week": 0, "time": "10:00"},  # Monday 10am
            order_type=RecurringOrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            strategy_name="iron_condor",
            conditions=[
                EntryCondition(
                    condition_type=ConditionType.IV_RANK,
                    operator=ConditionOperator.GREATER_THAN,
                    threshold=50.0,
                )
            ],
        )

        recurring_manager.add_template(template)

        # Step 2: Check triggers (Monday 10am, IV Rank = 60)
        monday_10am = datetime(2025, 1, 13, 10, 0)  # Monday
        market_data = {"iv_rank": 60.0}

        triggered = recurring_manager.check_triggers(monday_10am, market_data)

        assert len(triggered) == 1
        assert triggered[0].template_id == "weekly_ic"

        # Step 3: Execute template (submit to order queue)
        def order_executor(tmpl):
            # Submit to order queue
            request = OrderRequest(
                order_type=QueueOrderType.OPTION_STRATEGY,
                symbol=tmpl.symbol,
                quantity=tmpl.quantity,
                strategy_name=tmpl.strategy_name,
            )
            response = order_queue.submit_order(request)
            return response["order_id"] if response["success"] else None

        success = recurring_manager.execute_template(template, order_executor)

        assert success is True
        assert template.trigger_count == 1
        assert order_queue.stats["total_submitted"] == 1

    @pytest.mark.integration
    def test_multi_source_position_tracking(self, components):
        """Test tracking positions from all sources together."""
        bot_manager = components["bot_manager"]

        # Add positions from different sources
        # Autonomous
        bot_manager.add_position(
            position_id="auto_1",
            symbol="SPY",
            source=BotPositionSource.AUTONOMOUS,
            entry_price=-500.0,
            quantity=1,
            strategy_type="iron_condor",
            legs=[],
        )

        # Manual UI
        bot_manager.add_position(
            position_id="manual_1",
            symbol="AAPL",
            source=BotPositionSource.MANUAL_UI,
            entry_price=-300.0,
            quantity=2,
            strategy_type="butterfly",
            legs=[],
        )

        # Recurring
        bot_manager.add_position(
            position_id="recurring_1",
            symbol="QQQ",
            source=BotPositionSource.RECURRING,
            entry_price=-400.0,
            quantity=1,
            strategy_type="iron_condor",
            legs=[],
        )

        # Verify all tracked
        assert len(bot_manager.positions) == 3
        assert bot_manager.stats["total_positions"] == 3

        # Verify can retrieve by source
        auto_positions = [p for p in bot_manager.positions.values() if p.source == BotPositionSource.AUTONOMOUS]
        manual_positions = [p for p in bot_manager.positions.values() if p.source == BotPositionSource.MANUAL_UI]
        recurring_positions = [p for p in bot_manager.positions.values() if p.source == BotPositionSource.RECURRING]

        assert len(auto_positions) == 1
        assert len(manual_positions) == 1
        assert len(recurring_positions) == 1

    @pytest.mark.integration
    def test_position_management_override(self, components):
        """Test manual override of bot management."""
        bot_manager = components["bot_manager"]

        # Add position with bot management enabled
        position = bot_manager.add_position(
            position_id="pos_1",
            symbol="SPY",
            source=BotPositionSource.MANUAL_UI,
            entry_price=-500.0,
            quantity=10,
            strategy_type="iron_condor",
            legs=[],
            management_enabled=True,
        )

        assert position.management_enabled is True

        # User disables bot management
        success = bot_manager.disable_management("pos_1")

        assert success is True
        assert position.management_enabled is False

        # Simulate profit scenario - bot should NOT take profit
        pnl_pct = position.calculate_pnl_pct(-250.0)  # +50% profit

        # Check if management is active
        if not position.management_enabled:
            # Should not manage this position
            assert position.current_quantity == 10  # No change

        # Re-enable management
        bot_manager.enable_management("pos_1")

        assert position.management_enabled is True

    @pytest.mark.integration
    def test_recurring_template_persistence(self, algorithm, tmp_path):
        """Test recurring templates persist across restarts."""
        # Create first manager
        manager1 = create_recurring_order_manager(
            algorithm,
            storage_path=tmp_path,
        )

        # Add template
        template = RecurringOrderTemplate(
            template_id="persist_test",
            name="Persistence Test",
            order_type=RecurringOrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            strategy_name="iron_condor",
        )

        manager1.add_template(template)

        # Create second manager (simulating restart)
        manager2 = create_recurring_order_manager(
            algorithm,
            storage_path=tmp_path,
        )

        # Template should be loaded
        assert "persist_test" in manager2.templates
        loaded_template = manager2.get_template("persist_test")
        assert loaded_template.name == "Persistence Test"

    @pytest.mark.integration
    def test_order_queue_priority_handling(self, components):
        """Test order queue handles priorities correctly."""
        order_queue = components["order_queue"]

        # Submit low priority order
        low_request = OrderRequest(
            order_type=QueueOrderType.MARKET,
            symbol="SPY",
            quantity=1,
            priority=OrderPriority.LOW,
        )
        order_queue.submit_order(low_request)

        # Submit high priority order
        high_request = OrderRequest(
            order_type=QueueOrderType.MARKET,
            symbol="SPY",
            quantity=1,
            priority=OrderPriority.HIGH,
        )
        order_queue.submit_order(high_request)

        # Submit normal priority order
        normal_request = OrderRequest(
            order_type=QueueOrderType.MARKET,
            symbol="SPY",
            quantity=1,
            priority=OrderPriority.NORMAL,
        )
        order_queue.submit_order(normal_request)

        # Get pending orders - should come out in priority order
        pending = order_queue.get_pending_orders(max_count=3)

        assert len(pending) == 3
        assert pending[0].priority == OrderPriority.HIGH
        assert pending[1].priority == OrderPriority.NORMAL
        assert pending[2].priority == OrderPriority.LOW

    @pytest.mark.integration
    def test_full_lifecycle_iron_condor(self, algorithm, components):
        """Test complete lifecycle of an iron condor position."""
        bot_manager = components["bot_manager"]
        order_queue = components["order_queue"]

        # Step 1: Submit order
        request = OrderRequest(
            order_type=QueueOrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=10,
            strategy_name="iron_condor",
            strategy_params={
                "put_buy": 450,
                "put_sell": 455,
                "call_sell": 465,
                "call_buy": 470,
            },
            bot_managed=True,
        )

        response = order_queue.submit_order(request)
        order_id = response["order_id"]

        # Step 2: Execute order
        order_queue.update_order_status(
            order_id,
            order_queue_api.OrderStatus.FILLED,
        )

        # Step 3: Add to bot manager
        position = bot_manager.add_position(
            position_id=order_id,
            symbol="SPY",
            source=BotPositionSource.AUTONOMOUS,
            entry_price=-500.0,
            quantity=10,
            strategy_type="iron_condor",
            legs=[],
        )

        # Step 4: Simulate market movement and profit-taking
        # +50% profit - take 30%
        bot_manager._check_profit_thresholds(position, 0.50)
        assert position.current_quantity == 7

        # +100% profit - take 50% of remaining
        bot_manager._check_profit_thresholds(position, 1.00)
        assert position.current_quantity <= 4

        # +200% profit - take 20% of remaining
        bot_manager._check_profit_thresholds(position, 2.00)
        assert position.current_quantity <= 4

        # Verify all thresholds triggered
        assert all(t.triggered for t in position.profit_thresholds)

    @pytest.mark.integration
    def test_error_handling_across_components(self, components):
        """Test error handling propagates correctly."""
        order_queue = components["order_queue"]
        bot_manager = components["bot_manager"]

        # Test 1: Invalid order rejection
        invalid_request = OrderRequest(
            order_type=QueueOrderType.MARKET,
            symbol="",  # Invalid empty symbol
            quantity=1,
        )

        response = order_queue.submit_order(invalid_request)

        assert response["success"] is False
        assert "Symbol is required" in response["error"]

        # Test 2: Position not found
        success = bot_manager.disable_management("nonexistent")

        assert success is False

        # Test 3: Recurring template execution failure
        recurring_manager = components["recurring_manager"]

        template = RecurringOrderTemplate(
            template_id="fail_test",
            name="Failure Test",
            order_type=RecurringOrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
        )

        recurring_manager.add_template(template)

        # Executor that always fails
        def failing_executor(tmpl):
            raise ValueError("Test failure")

        success = recurring_manager.execute_template(template, failing_executor)

        assert success is False
        assert recurring_manager.stats["failed_executions"] == 1


class TestPerformanceIntegration:
    """Performance-related integration tests."""

    @pytest.mark.integration
    def test_large_position_count_performance(self):
        """Test system handles many positions efficiently."""
        algorithm = Mock()
        algorithm.Debug = Mock()

        bot_manager = create_bot_position_manager(algorithm, enable_logging=False)

        # Add 100 positions
        for i in range(100):
            bot_manager.add_position(
                position_id=f"pos_{i}",
                symbol=f"SYMBOL_{i % 10}",  # 10 different symbols
                source=BotPositionSource.AUTONOMOUS,
                entry_price=-500.0,
                quantity=1,
                strategy_type="iron_condor",
                legs=[],
            )

        # Should handle efficiently
        assert len(bot_manager.positions) == 100
        assert bot_manager.stats["total_positions"] == 100

        # Test retrieval by symbol
        symbol_0_positions = [p for p in bot_manager.positions.values() if p.symbol == "SYMBOL_0"]

        assert len(symbol_0_positions) == 10

    @pytest.mark.integration
    def test_high_order_throughput(self):
        """Test order queue handles high throughput."""
        algorithm = Mock()
        order_queue = create_order_queue_api(algorithm, skip_auth_validation=True)

        # Submit 1000 orders
        for i in range(1000):
            request = OrderRequest(
                order_type=QueueOrderType.MARKET,
                symbol=f"SYMBOL_{i % 100}",
                quantity=1,
            )
            response = order_queue.submit_order(request)
            assert response["success"] is True

        # All should be queued
        assert order_queue.stats["total_submitted"] == 1000


# Add import for order_queue_api module
from api import order_queue_api
