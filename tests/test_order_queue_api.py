"""
Tests for Order Queue API

Tests for UI order submission, queue management, and status tracking.
"""

from unittest.mock import Mock

import pytest

from api.order_queue_api import (
    OrderPriority,
    OrderQueueAPI,
    OrderRequest,
    OrderStatus,
    OrderType,
    create_order_queue_api,
)


class TestOrderQueueInitialization:
    """Tests for OrderQueueAPI initialization."""

    @pytest.mark.unit
    def test_api_creation(self):
        """Test API instance creation."""
        algorithm = Mock()
        # Use skip_auth_validation for tests that don't need to test auth
        api = create_order_queue_api(algorithm, skip_auth_validation=True)

        assert api.algorithm == algorithm
        assert api.queue.qsize() == 0
        assert len(api.orders) == 0

    @pytest.mark.unit
    def test_api_with_custom_settings(self):
        """Test API with custom settings."""
        algorithm = Mock()
        api = OrderQueueAPI(
            algorithm=algorithm,
            skip_auth_validation=True,
            max_queue_size=500,
            enable_logging=False,
        )

        assert api.max_queue_size == 500
        assert api.enable_logging is False


class TestOrderSubmission:
    """Tests for order submission."""

    @pytest.fixture
    def api(self):
        """Create API instance for testing."""
        algorithm = Mock()
        algorithm.Debug = Mock()
        algorithm.Error = Mock()
        return create_order_queue_api(algorithm, skip_auth_validation=True)

    @pytest.mark.unit
    def test_submit_option_strategy_order(self, api):
        """Test submitting an option strategy order."""
        request = OrderRequest(
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            strategy_name="iron_condor",
            strategy_params={
                "put_buy": 450,
                "put_sell": 455,
                "call_sell": 465,
                "call_buy": 470,
            },
        )

        response = api.submit_order(request)

        assert response["success"] is True
        assert response["order_id"] is not None
        assert response["status"] == OrderStatus.QUEUED.value
        assert api.queue.qsize() == 1
        assert api.stats["total_submitted"] == 1

    @pytest.mark.unit
    def test_submit_manual_legs_order(self, api):
        """Test submitting a manual legs order."""
        request = OrderRequest(
            order_type=OrderType.MANUAL_LEGS,
            symbol="SPY",
            quantity=1,
            legs=[
                {"symbol": "SPY_CALL_450", "quantity": 1, "side": "buy"},
                {"symbol": "SPY_CALL_455", "quantity": -2, "side": "sell"},
                {"symbol": "SPY_CALL_460", "quantity": 1, "side": "buy"},
            ],
            two_part_config={
                "debit_fill_target": 0.35,
                "credit_fill_target": 0.65,
            },
        )

        response = api.submit_order(request)

        assert response["success"] is True
        assert api.queue.qsize() == 1

    @pytest.mark.unit
    def test_submit_limit_order(self, api):
        """Test submitting a limit order."""
        request = OrderRequest(
            order_type=OrderType.LIMIT,
            symbol="SPY",
            quantity=10,
            limit_price=450.50,
        )

        response = api.submit_order(request)

        assert response["success"] is True

    @pytest.mark.unit
    def test_submit_with_priority(self, api):
        """Test submitting orders with different priorities."""
        # Submit low priority order
        low_request = OrderRequest(
            order_type=OrderType.MARKET,
            symbol="SPY",
            quantity=1,
            priority=OrderPriority.LOW,
        )
        api.submit_order(low_request)

        # Submit high priority order
        high_request = OrderRequest(
            order_type=OrderType.MARKET,
            symbol="SPY",
            quantity=1,
            priority=OrderPriority.HIGH,
        )
        api.submit_order(high_request)

        # High priority should come out first
        orders = api.get_pending_orders(max_count=2)
        assert orders[0].priority == OrderPriority.HIGH
        assert orders[1].priority == OrderPriority.LOW


class TestOrderValidation:
    """Tests for order validation."""

    @pytest.fixture
    def api(self):
        """Create API instance for testing."""
        algorithm = Mock()
        return create_order_queue_api(algorithm, skip_auth_validation=True)

    @pytest.mark.unit
    def test_reject_missing_symbol(self, api):
        """Test rejection of order with missing symbol."""
        request = OrderRequest(
            order_type=OrderType.MARKET,
            symbol="",
            quantity=1,
        )

        response = api.submit_order(request)

        assert response["success"] is False
        assert "Symbol is required" in response["error"]

    @pytest.mark.unit
    def test_reject_invalid_quantity(self, api):
        """Test rejection of order with invalid quantity."""
        request = OrderRequest(
            order_type=OrderType.MARKET,
            symbol="SPY",
            quantity=0,
        )

        response = api.submit_order(request)

        assert response["success"] is False
        assert "Quantity must be positive" in response["error"]

    @pytest.mark.unit
    def test_reject_option_strategy_without_name(self, api):
        """Test rejection of option strategy without strategy name."""
        request = OrderRequest(
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            strategy_name=None,
        )

        response = api.submit_order(request)

        assert response["success"] is False
        assert "Strategy name required" in response["error"]

    @pytest.mark.unit
    def test_reject_manual_legs_without_legs(self, api):
        """Test rejection of manual legs order without legs."""
        request = OrderRequest(
            order_type=OrderType.MANUAL_LEGS,
            symbol="SPY",
            quantity=1,
            legs=None,
        )

        response = api.submit_order(request)

        assert response["success"] is False
        assert "Legs required" in response["error"]

    @pytest.mark.unit
    def test_reject_limit_order_without_price(self, api):
        """Test rejection of limit order without price."""
        request = OrderRequest(
            order_type=OrderType.LIMIT,
            symbol="SPY",
            quantity=1,
            limit_price=None,
        )

        response = api.submit_order(request)

        assert response["success"] is False
        assert "Limit price required" in response["error"]

    @pytest.mark.unit
    def test_reject_excessive_quantity(self, api):
        """Test rejection of order with excessive quantity."""
        request = OrderRequest(
            order_type=OrderType.MARKET,
            symbol="SPY",
            quantity=150,  # Exceeds max of 100
        )

        response = api.submit_order(request)

        assert response["success"] is False
        assert "exceeds maximum" in response["error"]


class TestOrderProcessing:
    """Tests for order processing."""

    @pytest.fixture
    def api(self):
        """Create API instance for testing."""
        algorithm = Mock()
        return create_order_queue_api(algorithm, skip_auth_validation=True)

    @pytest.mark.unit
    def test_get_pending_orders(self, api):
        """Test getting pending orders from queue."""
        # Submit 3 orders
        for i in range(3):
            request = OrderRequest(
                order_type=OrderType.MARKET,
                symbol="SPY",
                quantity=1,
            )
            api.submit_order(request)

        # Get pending orders
        orders = api.get_pending_orders(max_count=2)

        assert len(orders) == 2
        assert all(order.status == OrderStatus.EXECUTING for order in orders)
        assert api.queue.qsize() == 1  # One left in queue

    @pytest.mark.unit
    def test_update_order_status(self, api):
        """Test updating order status."""
        # Submit order
        request = OrderRequest(
            order_type=OrderType.MARKET,
            symbol="SPY",
            quantity=1,
        )
        response = api.submit_order(request)
        order_id = response["order_id"]

        # Update to filled
        success = api.update_order_status(
            order_id,
            OrderStatus.FILLED,
            execution_details={"fill_price": 450.50},
        )

        assert success is True
        assert api.orders[order_id].status == OrderStatus.FILLED
        assert api.stats["total_filled"] == 1

    @pytest.mark.unit
    def test_cancel_order(self, api):
        """Test cancelling an order."""
        # Submit order
        request = OrderRequest(
            order_type=OrderType.MARKET,
            symbol="SPY",
            quantity=1,
        )
        response = api.submit_order(request)
        order_id = response["order_id"]

        # Cancel it
        cancel_response = api.cancel_order(order_id)

        assert cancel_response["success"] is True
        assert api.orders[order_id].status == OrderStatus.CANCELLED
        assert api.stats["total_cancelled"] == 1

    @pytest.mark.unit
    def test_cannot_cancel_executing_order(self, api):
        """Test that executing orders cannot be cancelled."""
        # Submit order
        request = OrderRequest(
            order_type=OrderType.MARKET,
            symbol="SPY",
            quantity=1,
        )
        response = api.submit_order(request)
        order_id = response["order_id"]

        # Get it (changes status to EXECUTING)
        api.get_pending_orders(max_count=1)

        # Try to cancel
        cancel_response = api.cancel_order(order_id)

        assert cancel_response["success"] is False
        assert "Cannot cancel" in cancel_response["error"]


class TestOrderRetrieval:
    """Tests for order retrieval."""

    @pytest.fixture
    def api(self):
        """Create API instance for testing."""
        algorithm = Mock()
        return create_order_queue_api(algorithm, skip_auth_validation=True)

    @pytest.mark.unit
    def test_get_order_status(self, api):
        """Test getting order status."""
        # Submit order
        request = OrderRequest(
            order_type=OrderType.MARKET,
            symbol="SPY",
            quantity=1,
        )
        response = api.submit_order(request)
        order_id = response["order_id"]

        # Get status
        status = api.get_order_status(order_id)

        assert status is not None
        assert status["id"] == order_id
        assert status["status"] == OrderStatus.QUEUED.value

    @pytest.mark.unit
    def test_get_all_orders(self, api):
        """Test getting all orders."""
        # Submit 5 orders
        for i in range(5):
            request = OrderRequest(
                order_type=OrderType.MARKET,
                symbol="SPY",
                quantity=1,
            )
            api.submit_order(request)

        # Get all orders
        orders = api.get_all_orders()

        assert len(orders) == 5

    @pytest.mark.unit
    def test_get_orders_by_status(self, api):
        """Test getting orders filtered by status."""
        # Submit and process orders
        for i in range(3):
            request = OrderRequest(
                order_type=OrderType.MARKET,
                symbol="SPY",
                quantity=1,
            )
            response = api.submit_order(request)

            if i == 0:
                # Fill first order
                api.update_order_status(response["order_id"], OrderStatus.FILLED)

        # Get filled orders
        filled = api.get_all_orders(status_filter=OrderStatus.FILLED)
        queued = api.get_all_orders(status_filter=OrderStatus.QUEUED)

        assert len(filled) == 1
        assert len(queued) == 2


class TestStatistics:
    """Tests for API statistics."""

    @pytest.fixture
    def api(self):
        """Create API instance for testing."""
        algorithm = Mock()
        return create_order_queue_api(algorithm, skip_auth_validation=True)

    @pytest.mark.unit
    def test_statistics_tracking(self, api):
        """Test that statistics are tracked correctly."""
        # Submit orders
        order_ids = []
        for i in range(5):
            request = OrderRequest(
                order_type=OrderType.MARKET,
                symbol="SPY",
                quantity=1,
            )
            response = api.submit_order(request)
            order_ids.append(response["order_id"])

        # Get orders from queue (simulating algorithm processing)
        api.get_pending_orders(max_count=3)

        # Fill some, reject some
        api.update_order_status(order_ids[0], OrderStatus.FILLED)
        api.update_order_status(order_ids[1], OrderStatus.FILLED)
        api.update_order_status(order_ids[2], OrderStatus.REJECTED)

        stats = api.get_statistics()

        assert stats["total_submitted"] == 5
        assert stats["total_filled"] == 2
        assert stats["total_rejected"] == 1
        assert stats["queue_size"] == 2  # 2 orders still in queue
        assert stats["total_orders"] == 5


class TestBotManagedOrders:
    """Tests for bot-managed order configuration."""

    @pytest.fixture
    def api(self):
        """Create API instance for testing."""
        algorithm = Mock()
        return create_order_queue_api(algorithm, skip_auth_validation=True)

    @pytest.mark.unit
    def test_submit_bot_managed_order(self, api):
        """Test submitting order with bot management enabled."""
        request = OrderRequest(
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            strategy_name="iron_condor",
            bot_managed=True,  # Bot should manage profit-taking/stop-loss
        )

        response = api.submit_order(request)
        order_id = response["order_id"]

        order = api.orders[order_id]
        assert order.request.bot_managed is True

    @pytest.mark.unit
    def test_submit_manual_management_order(self, api):
        """Test submitting order with manual management."""
        request = OrderRequest(
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            strategy_name="butterfly_call",
            bot_managed=False,  # User will manage manually
        )

        response = api.submit_order(request)
        order_id = response["order_id"]

        order = api.orders[order_id]
        assert order.request.bot_managed is False


class TestRecurringOrders:
    """Tests for recurring order configuration."""

    @pytest.fixture
    def api(self):
        """Create API instance for testing."""
        algorithm = Mock()
        return create_order_queue_api(algorithm, skip_auth_validation=True)

    @pytest.mark.unit
    def test_submit_recurring_order(self, api):
        """Test submitting a recurring order."""
        request = OrderRequest(
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            strategy_name="iron_condor",
            recurring_config={
                "frequency": "weekly",  # Every Monday
                "condition": "iv_rank > 50",
                "enabled": True,
            },
        )

        response = api.submit_order(request)
        order_id = response["order_id"]

        order = api.orders[order_id]
        assert order.request.recurring_config is not None
        assert order.request.recurring_config["frequency"] == "weekly"
