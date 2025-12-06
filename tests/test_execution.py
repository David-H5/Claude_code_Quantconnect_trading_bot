"""
Execution and Order Management Tests

Tests for order execution including:
- Slippage simulation and estimation
- Cancel/replace logic
- Partial fills handling
- Order timing and latency effects
- Fill rate tracking

Based on best practices from:
- QuantConnect execution documentation
- Option Alpha safeguards
- 3Commas risk management guide
"""

from datetime import datetime, timedelta

import pytest

from config import OrderExecutionConfig, ProfitTakingConfig, ProfitThreshold
from execution.profit_taking import (
    ProfitTakingRiskModel,
)
from execution.smart_execution import (
    SmartExecutionModel,
)
from execution.smart_execution import (
    SmartOrderStatus as OrderStatus,
)
from execution.smart_execution import (
    SmartOrderType as OrderType,
)


class TestSlippageSimulation:
    """Tests for slippage estimation and handling."""

    @pytest.fixture
    def execution_config(self):
        """Create execution config for testing."""
        return OrderExecutionConfig(
            cancel_replace_enabled=True,
            cancel_after_seconds=2.5,
            max_cancel_replace_attempts=3,
            max_bid_increase_pct=0.05,
            bid_increment_pct=0.01,
            use_mid_price=True,
        )

    @pytest.mark.unit
    def test_slippage_calculation_buy_order(self, execution_config):
        """Test slippage calculation for buy orders."""
        executor = SmartExecutionModel(execution_config)

        # Submit buy order at mid price
        order = executor.submit_order(
            symbol="SPY",
            side="buy",
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=100.00,
            bid_price=99.95,
            ask_price=100.05,
        )

        # Simulate fill at slightly worse price
        executor.update_order_status(
            order.order_id,
            OrderStatus.FILLED,
            filled_quantity=100,
            fill_price=100.03,  # 3 cents slippage
        )

        stats = executor.get_statistics()
        # average_slippage = total_slippage / successful_fills = 0.03 / 1 = 0.03
        assert stats["average_slippage"] == pytest.approx(0.03, abs=0.001)

    @pytest.mark.unit
    def test_slippage_accumulates_across_orders(self, execution_config):
        """Test that slippage accumulates correctly."""
        executor = SmartExecutionModel(execution_config)

        # Multiple orders with varying slippage
        for i in range(3):
            order = executor.submit_order(
                symbol="SPY",
                side="buy",
                quantity=100,
                limit_price=100.00,
            )
            executor.update_order_status(
                order.order_id,
                OrderStatus.FILLED,
                filled_quantity=100,
                fill_price=100.00 + (i + 1) * 0.01,  # 1, 2, 3 cents slippage
            )

        stats = executor.get_statistics()
        # Total slippage: 0.01 + 0.02 + 0.03 = 0.06, average = 0.06 / 3 = 0.02
        assert stats["average_slippage"] == pytest.approx(0.02, abs=0.001)

    @pytest.mark.unit
    def test_negative_slippage_price_improvement(self, execution_config):
        """Test handling of price improvement (negative slippage)."""
        executor = SmartExecutionModel(execution_config)

        order = executor.submit_order(
            symbol="SPY",
            side="buy",
            quantity=100,
            limit_price=100.00,
        )

        # Fill at better price than expected
        executor.update_order_status(
            order.order_id,
            OrderStatus.FILLED,
            filled_quantity=100,
            fill_price=99.97,  # 3 cents improvement
        )

        # Price improvement should still be tracked (as negative slippage)
        assert order.average_fill_price == 99.97


class TestCancelReplaceLogic:
    """Tests for cancel/replace order management."""

    @pytest.fixture
    def executor(self):
        """Create executor with cancel/replace enabled."""
        config = OrderExecutionConfig(
            cancel_replace_enabled=True,
            cancel_after_seconds=2.5,
            max_cancel_replace_attempts=3,
            max_bid_increase_pct=0.05,
            bid_increment_pct=0.01,
        )
        return SmartExecutionModel(config)

    @pytest.mark.unit
    def test_cancel_replace_increases_price(self, executor):
        """Test that cancel/replace increases bid price."""
        order = executor.submit_order(
            symbol="SPY",
            side="buy",
            quantity=100,
            limit_price=100.00,
        )

        # Fast forward time (simulate order aging)
        order.created_at = datetime.now() - timedelta(seconds=3)

        # Trigger cancel/replace
        replaced = executor.check_and_replace(order.order_id, 99.95, 100.05)

        assert replaced is not None
        assert replaced.limit_price > 100.00
        assert replaced.cancel_replace_count == 1

    @pytest.mark.unit
    def test_max_cancel_replace_attempts(self, executor):
        """Test that cancel/replace respects max attempts."""
        order = executor.submit_order(
            symbol="SPY",
            side="buy",
            quantity=100,
            limit_price=100.00,
        )

        # Exhaust cancel/replace attempts - use high ask to allow bid increases
        for _ in range(3):
            order.created_at = datetime.now() - timedelta(seconds=3)
            # Use high ask (110.00) to allow bid to increase without hitting cap
            executor.check_and_replace(order.order_id, 99.95, 110.00)

        # Fourth attempt should fail due to max attempts
        order.created_at = datetime.now() - timedelta(seconds=3)
        result = executor.check_and_replace(order.order_id, 99.95, 110.00)

        assert result is None
        assert order.cancel_replace_count == 3

    @pytest.mark.unit
    def test_max_bid_increase_limit(self, executor):
        """Test that bid increase respects maximum limit."""
        order = executor.submit_order(
            symbol="SPY",
            side="buy",
            quantity=100,
            limit_price=100.00,
        )

        # Set price at exactly max increase (5%)
        order.limit_price = 105.00  # At 5% increase limit
        order.created_at = datetime.now() - timedelta(seconds=3)

        # Any increment would exceed 5% max, so should not replace
        # Use high ask to ensure price isn't capped by ask
        result = executor.check_and_replace(order.order_id, 104.85, 110.00)

        assert result is None

    @pytest.mark.unit
    def test_cancel_replace_disabled(self):
        """Test behavior when cancel/replace is disabled."""
        config = OrderExecutionConfig(cancel_replace_enabled=False)
        executor = SmartExecutionModel(config)

        order = executor.submit_order(
            symbol="SPY",
            side="buy",
            quantity=100,
            limit_price=100.00,
        )
        order.created_at = datetime.now() - timedelta(seconds=10)

        result = executor.check_and_replace(order.order_id, 99.95, 100.05)

        assert result is None


class TestPartialFills:
    """Tests for partial fill handling."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        config = OrderExecutionConfig()
        return SmartExecutionModel(config)

    @pytest.mark.unit
    def test_partial_fill_updates_quantity(self, executor):
        """Test that partial fills correctly update quantities."""
        order = executor.submit_order(
            symbol="SPY",
            side="buy",
            quantity=1000,
            limit_price=100.00,
        )

        # Partial fill
        executor.update_order_status(
            order.order_id,
            OrderStatus.PARTIALLY_FILLED,
            filled_quantity=300,
            fill_price=100.00,
        )

        assert order.filled_quantity == 300
        assert order.unfilled_quantity == 700
        assert not order.is_complete

    @pytest.mark.unit
    def test_multiple_partial_fills(self, executor):
        """Test handling of multiple partial fills."""
        order = executor.submit_order(
            symbol="SPY",
            side="buy",
            quantity=1000,
            limit_price=100.00,
        )

        # First partial fill at 100.00
        executor.update_order_status(
            order.order_id,
            OrderStatus.PARTIALLY_FILLED,
            filled_quantity=300,
            fill_price=100.00,
        )

        # Second partial fill at 100.02
        executor.update_order_status(
            order.order_id,
            OrderStatus.PARTIALLY_FILLED,
            filled_quantity=500,
            fill_price=100.02,
        )

        # Final fill at 100.01
        executor.update_order_status(
            order.order_id,
            OrderStatus.FILLED,
            filled_quantity=1000,
            fill_price=100.01,
        )

        assert order.is_complete
        assert order.filled_quantity == 1000

    @pytest.mark.unit
    def test_partial_fill_does_not_complete_order(self, executor):
        """Test that partial fill doesn't mark order as complete."""
        order = executor.submit_order(
            symbol="SPY",
            side="buy",
            quantity=1000,
            limit_price=100.00,
        )

        executor.update_order_status(
            order.order_id,
            OrderStatus.PARTIALLY_FILLED,
            filled_quantity=999,  # Almost complete
            fill_price=100.00,
        )

        assert not order.is_complete
        assert order.status == OrderStatus.PARTIALLY_FILLED


class TestProfitTaking:
    """Tests for graduated profit taking."""

    @pytest.fixture
    def profit_config(self):
        """Create profit taking configuration."""
        return ProfitTakingConfig(
            enabled=True,
            thresholds=[
                ProfitThreshold(gain_pct=1.00, sell_pct=0.25),  # +100%: sell 25%
                ProfitThreshold(gain_pct=2.00, sell_pct=0.25),  # +200%: sell 25%
                ProfitThreshold(gain_pct=4.00, sell_pct=0.25),  # +400%: sell 25%
                ProfitThreshold(gain_pct=10.00, sell_pct=0.25),  # +1000%: sell 25%
            ],
        )

    @pytest.mark.unit
    def test_first_profit_threshold(self, profit_config):
        """Test profit taking at first threshold."""
        taker = ProfitTakingRiskModel(profit_config)

        # Add position at $100
        taker.register_position("SPY_CALL", entry_price=100.0, quantity=100)

        # Price doubles - should trigger first threshold
        orders = taker.update_position("SPY_CALL", current_price=200.0, current_quantity=100)

        assert len(orders) == 1
        assert orders[0].sell_pct == 0.25
        assert orders[0].quantity == 25  # 25% of 100

    @pytest.mark.unit
    def test_multiple_thresholds_triggered(self, profit_config):
        """Test when price jumps past multiple thresholds."""
        taker = ProfitTakingRiskModel(profit_config)
        taker.register_position("SPY_CALL", entry_price=100.0, quantity=100)

        # Price goes to 5x - should trigger multiple thresholds
        orders = taker.update_position("SPY_CALL", current_price=500.0, current_quantity=100)

        # Should generate orders for +100%, +200%, +400% thresholds
        assert len(orders) >= 3

    @pytest.mark.unit
    def test_no_orders_below_threshold(self, profit_config):
        """Test no orders generated below first threshold."""
        taker = ProfitTakingRiskModel(profit_config)
        taker.register_position("SPY_CALL", entry_price=100.0, quantity=100)

        # Price up 50% - below first threshold
        orders = taker.update_position("SPY_CALL", current_price=150.0, current_quantity=100)

        assert len(orders) == 0

    @pytest.mark.unit
    def test_disabled_profit_taking(self):
        """Test behavior when profit taking is disabled."""
        config = ProfitTakingConfig(enabled=False)
        taker = ProfitTakingRiskModel(config)
        taker.register_position("SPY_CALL", entry_price=100.0, quantity=100)

        orders = taker.update_position("SPY_CALL", current_price=500.0, current_quantity=100)

        assert len(orders) == 0


class TestOrderTiming:
    """Tests for order timing and latency effects."""

    @pytest.mark.unit
    def test_order_age_calculation(self):
        """Test accurate calculation of order age."""
        config = OrderExecutionConfig()
        executor = SmartExecutionModel(config)

        order = executor.submit_order(
            symbol="SPY",
            side="buy",
            quantity=100,
            limit_price=100.00,
        )

        # Set order creation time to 5 seconds ago
        order.created_at = datetime.now() - timedelta(seconds=5)

        assert order.age_seconds >= 5.0
        assert order.age_seconds < 6.0

    @pytest.mark.unit
    def test_quick_cancel_timing(self):
        """Test quick cancel logic timing (2.5 second threshold)."""
        config = OrderExecutionConfig(
            cancel_replace_enabled=True,
            cancel_after_seconds=2.5,
        )
        executor = SmartExecutionModel(config)

        order = executor.submit_order(
            symbol="SPY",
            side="buy",
            quantity=100,
            limit_price=100.00,
        )

        # Order not old enough for cancel/replace
        order.created_at = datetime.now() - timedelta(seconds=2.0)
        result = executor.check_and_replace(order.order_id, 99.95, 100.05)
        assert result is None

        # Order old enough for cancel/replace
        order.created_at = datetime.now() - timedelta(seconds=3.0)
        result = executor.check_and_replace(order.order_id, 99.95, 100.05)
        assert result is not None


class TestFillRateTracking:
    """Tests for fill rate tracking and statistics."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        config = OrderExecutionConfig()
        return SmartExecutionModel(config)

    @pytest.mark.unit
    def test_fill_rate_calculation(self, executor):
        """Test accurate fill rate calculation."""
        # Submit 10 orders
        for i in range(10):
            order = executor.submit_order(
                symbol="SPY",
                side="buy",
                quantity=100,
                limit_price=100.00,
            )

            # Fill 7 orders, cancel 3
            if i < 7:
                executor.update_order_status(
                    order.order_id,
                    OrderStatus.FILLED,
                    filled_quantity=100,
                    fill_price=100.00,
                )
            else:
                executor.update_order_status(
                    order.order_id,
                    OrderStatus.CANCELLED,
                )

        stats = executor.get_statistics()
        assert stats["fill_rate"] == pytest.approx(0.7, abs=0.01)

    @pytest.mark.unit
    def test_statistics_after_no_orders(self, executor):
        """Test statistics when no orders have been submitted."""
        stats = executor.get_statistics()

        assert stats["total_orders"] == 0
        assert stats["fill_rate"] == 0
        assert stats["average_slippage"] == 0

    @pytest.mark.unit
    def test_cancel_replace_count_tracking(self, executor):
        """Test tracking of cancel/replace count."""
        config = OrderExecutionConfig(
            cancel_replace_enabled=True,
            cancel_after_seconds=0,  # Immediate
            max_cancel_replace_attempts=5,
            max_bid_increase_pct=0.50,  # 50% max to allow multiple replacements
            bid_increment_pct=0.01,  # 1% increment
        )
        executor = SmartExecutionModel(config)

        order = executor.submit_order(
            symbol="SPY",
            side="buy",
            quantity=100,
            limit_price=100.00,
        )

        # Perform 3 cancel/replaces - use high ask to allow bid increases
        for _ in range(3):
            order.created_at = datetime.now() - timedelta(seconds=1)
            executor.check_and_replace(order.order_id, 99.95, 150.00)

        stats = executor.get_statistics()
        assert stats["cancel_replaces"] == 3
