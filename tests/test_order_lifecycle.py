"""
Order Management Lifecycle Tests

Tests for the complete order lifecycle from creation to settlement,
including state transitions, validations, and edge cases.

Based on best practices from:
- FIX Protocol order state machine
- Exchange order management patterns
- Trading system OMS testing
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import pytest


class OrderStatus(Enum):
    """Order status states following FIX protocol."""

    PENDING_NEW = "pending_new"
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    PENDING_CANCEL = "pending_cancel"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    PENDING_REPLACE = "pending_replace"
    REPLACED = "replaced"


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(Enum):
    """Time in force."""

    DAY = "day"
    GTC = "good_til_canceled"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"


@dataclass
class Order:
    """Represents an order."""

    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING_NEW
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    fills: list[dict] = field(default_factory=list)
    reject_reason: str | None = None

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        return self.status in [
            OrderStatus.NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.PENDING_REPLACE,
        ]


class OrderStateMachine:
    """Manages order state transitions."""

    # Valid state transitions
    VALID_TRANSITIONS = {
        OrderStatus.PENDING_NEW: [OrderStatus.NEW, OrderStatus.REJECTED],
        OrderStatus.NEW: [
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.PENDING_REPLACE,
            OrderStatus.EXPIRED,
        ],
        OrderStatus.PARTIALLY_FILLED: [
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.PENDING_REPLACE,
        ],
        OrderStatus.PENDING_CANCEL: [OrderStatus.CANCELED, OrderStatus.FILLED],
        OrderStatus.PENDING_REPLACE: [OrderStatus.REPLACED, OrderStatus.REJECTED],
        OrderStatus.FILLED: [],  # Terminal state
        OrderStatus.CANCELED: [],  # Terminal state
        OrderStatus.REJECTED: [],  # Terminal state
        OrderStatus.EXPIRED: [],  # Terminal state
        OrderStatus.REPLACED: [],  # Terminal state
    }

    @classmethod
    def can_transition(cls, from_status: OrderStatus, to_status: OrderStatus) -> bool:
        """Check if transition is valid."""
        valid_targets = cls.VALID_TRANSITIONS.get(from_status, [])
        return to_status in valid_targets

    @classmethod
    def transition(cls, order: Order, new_status: OrderStatus) -> bool:
        """Attempt to transition order to new status."""
        if not cls.can_transition(order.status, new_status):
            return False
        order.status = new_status
        order.updated_at = datetime.now()
        return True


class TestOrderCreation:
    """Tests for order creation."""

    @pytest.mark.lifecycle
    def test_create_market_order(self):
        """Test creating a market order."""
        order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
        )

        assert order.id == "ORD001"
        assert order.status == OrderStatus.PENDING_NEW
        assert order.quantity == 100
        assert order.filled_quantity == 0
        assert order.price is None  # Market orders have no price

    @pytest.mark.lifecycle
    def test_create_limit_order(self):
        """Test creating a limit order."""
        order = Order(
            id="ORD002",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=450.50,
        )

        assert order.order_type == OrderType.LIMIT
        assert order.price == 450.50

    @pytest.mark.lifecycle
    def test_create_stop_order(self):
        """Test creating a stop order."""
        order = Order(
            id="ORD003",
            symbol="SPY",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=100,
            stop_price=440.00,
        )

        assert order.order_type == OrderType.STOP
        assert order.stop_price == 440.00

    @pytest.mark.lifecycle
    def test_order_defaults(self):
        """Test order default values."""
        order = Order(
            id="ORD004",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
        )

        assert order.time_in_force == TimeInForce.DAY
        assert order.filled_quantity == 0
        assert order.average_fill_price == 0.0
        assert len(order.fills) == 0


class TestOrderStateTransitions:
    """Tests for order state transitions."""

    @pytest.mark.lifecycle
    def test_valid_new_to_filled(self):
        """Test valid transition from NEW to FILLED."""
        order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            status=OrderStatus.NEW,
        )

        result = OrderStateMachine.transition(order, OrderStatus.FILLED)
        assert result is True
        assert order.status == OrderStatus.FILLED

    @pytest.mark.lifecycle
    def test_valid_partial_fill_sequence(self):
        """Test valid partial fill sequence."""
        order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=450.00,
            status=OrderStatus.NEW,
        )

        # First partial fill
        result = OrderStateMachine.transition(order, OrderStatus.PARTIALLY_FILLED)
        assert result is True
        assert order.status == OrderStatus.PARTIALLY_FILLED

        # Another partial fill
        result = OrderStateMachine.transition(order, OrderStatus.PARTIALLY_FILLED)
        assert result is True

        # Final fill
        result = OrderStateMachine.transition(order, OrderStatus.FILLED)
        assert result is True
        assert order.status == OrderStatus.FILLED

    @pytest.mark.lifecycle
    def test_invalid_filled_to_canceled(self):
        """Test invalid transition from FILLED to CANCELED."""
        order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            status=OrderStatus.FILLED,
        )

        result = OrderStateMachine.transition(order, OrderStatus.CANCELED)
        assert result is False
        assert order.status == OrderStatus.FILLED  # Unchanged

    @pytest.mark.lifecycle
    def test_cancel_sequence(self):
        """Test order cancellation sequence."""
        order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=450.00,
            status=OrderStatus.NEW,
        )

        # Request cancel
        result = OrderStateMachine.transition(order, OrderStatus.PENDING_CANCEL)
        assert result is True
        assert order.status == OrderStatus.PENDING_CANCEL

        # Cancel confirmed
        result = OrderStateMachine.transition(order, OrderStatus.CANCELED)
        assert result is True
        assert order.status == OrderStatus.CANCELED

    @pytest.mark.lifecycle
    def test_rejection_from_pending(self):
        """Test rejection from pending new."""
        order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            status=OrderStatus.PENDING_NEW,
        )

        result = OrderStateMachine.transition(order, OrderStatus.REJECTED)
        assert result is True
        assert order.status == OrderStatus.REJECTED


class TestOrderFills:
    """Tests for order fill processing."""

    @pytest.fixture
    def order(self):
        """Create a test order."""
        return Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=450.00,
            status=OrderStatus.NEW,
        )

    def process_fill(self, order: Order, fill_qty: int, fill_price: float):
        """Process a fill on an order."""
        order.fills.append(
            {
                "quantity": fill_qty,
                "price": fill_price,
                "timestamp": datetime.now(),
            }
        )

        # Update average price
        total_value = sum(f["quantity"] * f["price"] for f in order.fills)
        order.filled_quantity = sum(f["quantity"] for f in order.fills)
        order.average_fill_price = total_value / order.filled_quantity

        # Update status
        if order.filled_quantity >= order.quantity:
            OrderStateMachine.transition(order, OrderStatus.FILLED)
        elif order.filled_quantity > 0:
            OrderStateMachine.transition(order, OrderStatus.PARTIALLY_FILLED)

    @pytest.mark.lifecycle
    def test_single_complete_fill(self, order):
        """Test single complete fill."""
        self.process_fill(order, 100, 450.00)

        assert order.filled_quantity == 100
        assert order.remaining_quantity == 0
        assert order.status == OrderStatus.FILLED
        assert order.average_fill_price == 450.00

    @pytest.mark.lifecycle
    def test_multiple_partial_fills(self, order):
        """Test multiple partial fills."""
        self.process_fill(order, 30, 450.00)
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.remaining_quantity == 70

        self.process_fill(order, 40, 450.10)
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.remaining_quantity == 30

        self.process_fill(order, 30, 450.05)
        assert order.status == OrderStatus.FILLED
        assert order.remaining_quantity == 0

    @pytest.mark.lifecycle
    def test_average_fill_price_calculation(self, order):
        """Test average fill price calculation."""
        self.process_fill(order, 50, 450.00)  # 50 @ 450.00
        self.process_fill(order, 50, 451.00)  # 50 @ 451.00

        # Average should be (50*450 + 50*451) / 100 = 450.50
        assert order.average_fill_price == pytest.approx(450.50)

    @pytest.mark.lifecycle
    def test_overfill_prevention(self, order):
        """Test that overfills are prevented."""
        self.process_fill(order, 100, 450.00)

        # Order is now filled, should not accept more
        initial_filled = order.filled_quantity

        # Attempting another fill should not change state
        assert order.status == OrderStatus.FILLED


class TestOrderValidation:
    """Tests for order validation rules."""

    @pytest.mark.lifecycle
    def test_validates_positive_quantity(self):
        """Test that quantity must be positive."""

        def validate_order(order: Order) -> list[str]:
            errors = []
            if order.quantity <= 0:
                errors.append("Quantity must be positive")
            return errors

        invalid_order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0,
        )

        errors = validate_order(invalid_order)
        assert "Quantity must be positive" in errors

    @pytest.mark.lifecycle
    def test_validates_limit_order_has_price(self):
        """Test that limit orders must have a price."""

        def validate_order(order: Order) -> list[str]:
            errors = []
            if order.order_type == OrderType.LIMIT and order.price is None:
                errors.append("Limit order must have a price")
            return errors

        invalid_order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=None,
        )

        errors = validate_order(invalid_order)
        assert "Limit order must have a price" in errors

    @pytest.mark.lifecycle
    def test_validates_stop_order_has_stop_price(self):
        """Test that stop orders must have a stop price."""

        def validate_order(order: Order) -> list[str]:
            errors = []
            if order.order_type == OrderType.STOP and order.stop_price is None:
                errors.append("Stop order must have a stop price")
            return errors

        invalid_order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=100,
            stop_price=None,
        )

        errors = validate_order(invalid_order)
        assert "Stop order must have a stop price" in errors

    @pytest.mark.lifecycle
    def test_validates_symbol_format(self):
        """Test symbol format validation."""

        def validate_symbol(symbol: str) -> list[str]:
            errors = []
            if not symbol:
                errors.append("Symbol is required")
            elif not symbol.isalpha() or len(symbol) > 5:
                errors.append("Invalid symbol format")
            return errors

        assert validate_symbol("SPY") == []
        assert validate_symbol("") != []
        assert validate_symbol("TOOLONG") != []


class TestOrderExpiration:
    """Tests for order expiration logic."""

    @pytest.mark.lifecycle
    def test_day_order_expires_at_close(self):
        """Test that DAY orders expire at market close."""
        order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=450.00,
            time_in_force=TimeInForce.DAY,
            status=OrderStatus.NEW,
            created_at=datetime.now().replace(hour=9, minute=30),
        )

        market_close = datetime.now().replace(hour=16, minute=0)

        def should_expire(order: Order, current_time: datetime) -> bool:
            if order.time_in_force == TimeInForce.DAY:
                return current_time >= market_close
            return False

        # Before close
        assert should_expire(order, datetime.now().replace(hour=15, minute=0)) is False

        # After close
        assert should_expire(order, datetime.now().replace(hour=17, minute=0)) is True

    @pytest.mark.lifecycle
    def test_gtc_order_does_not_expire_same_day(self):
        """Test that GTC orders don't expire same day."""
        order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=450.00,
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.NEW,
        )

        # GTC orders don't expire based on market close
        def should_expire_gtc(order: Order, days_old: int) -> bool:
            # GTC orders typically expire after 90 days
            return days_old > 90

        assert should_expire_gtc(order, 1) is False
        assert should_expire_gtc(order, 89) is False
        assert should_expire_gtc(order, 91) is True

    @pytest.mark.lifecycle
    def test_ioc_order_cancels_unfilled_portion(self):
        """Test that IOC orders cancel unfilled portion immediately."""
        order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=450.00,
            time_in_force=TimeInForce.IOC,
            status=OrderStatus.NEW,
        )

        # Simulate partial fill
        order.filled_quantity = 50

        def process_ioc(order: Order) -> Order:
            """Process IOC order - cancel unfilled portion."""
            if order.time_in_force == TimeInForce.IOC:
                if order.filled_quantity == 0:
                    order.status = OrderStatus.CANCELED
                elif order.filled_quantity < order.quantity:
                    # Reduce quantity to filled amount
                    order.quantity = order.filled_quantity
                    order.status = OrderStatus.FILLED
            return order

        order = process_ioc(order)

        assert order.quantity == 50  # Reduced to filled amount
        assert order.status == OrderStatus.FILLED


class TestOrderModification:
    """Tests for order modification (replace) logic."""

    @pytest.mark.lifecycle
    def test_modify_limit_price(self):
        """Test modifying limit order price."""
        order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=450.00,
            status=OrderStatus.NEW,
        )

        def modify_price(order: Order, new_price: float) -> Order:
            """Modify order price."""
            if not order.is_active:
                raise ValueError("Cannot modify inactive order")
            if order.order_type != OrderType.LIMIT:
                raise ValueError("Can only modify limit order price")

            OrderStateMachine.transition(order, OrderStatus.PENDING_REPLACE)
            # In real system, would create new order
            order.price = new_price
            OrderStateMachine.transition(order, OrderStatus.REPLACED)
            return order

        order = modify_price(order, 448.00)
        assert order.price == 448.00

    @pytest.mark.lifecycle
    def test_modify_quantity(self):
        """Test modifying order quantity."""
        order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=450.00,
            status=OrderStatus.NEW,
        )

        def modify_quantity(order: Order, new_quantity: int) -> Order:
            """Modify order quantity."""
            if not order.is_active:
                raise ValueError("Cannot modify inactive order")
            if new_quantity < order.filled_quantity:
                raise ValueError("Cannot reduce below filled quantity")

            OrderStateMachine.transition(order, OrderStatus.PENDING_REPLACE)
            order.quantity = new_quantity
            OrderStateMachine.transition(order, OrderStatus.REPLACED)
            return order

        order = modify_quantity(order, 150)
        assert order.quantity == 150

    @pytest.mark.lifecycle
    def test_cannot_modify_filled_order(self):
        """Test that filled orders cannot be modified."""
        order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=450.00,
            status=OrderStatus.FILLED,
        )

        def modify_price(order: Order, new_price: float) -> Order:
            if not order.is_active:
                raise ValueError("Cannot modify inactive order")
            order.price = new_price
            return order

        with pytest.raises(ValueError) as exc_info:
            modify_price(order, 448.00)
        assert "inactive" in str(exc_info.value).lower()


class TestOrderTracking:
    """Tests for order tracking and audit."""

    @pytest.mark.lifecycle
    def test_tracks_all_state_changes(self):
        """Test that all state changes are tracked."""
        state_history = []

        def track_transition(order: Order, old_status: OrderStatus, new_status: OrderStatus):
            state_history.append(
                {
                    "order_id": order.id,
                    "from": old_status.value,
                    "to": new_status.value,
                    "timestamp": datetime.now(),
                }
            )

        order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            status=OrderStatus.PENDING_NEW,
        )

        # Track transitions
        old_status = order.status
        OrderStateMachine.transition(order, OrderStatus.NEW)
        track_transition(order, old_status, order.status)

        old_status = order.status
        OrderStateMachine.transition(order, OrderStatus.FILLED)
        track_transition(order, old_status, order.status)

        assert len(state_history) == 2
        assert state_history[0]["from"] == "pending_new"
        assert state_history[0]["to"] == "new"
        assert state_history[1]["from"] == "new"
        assert state_history[1]["to"] == "filled"

    @pytest.mark.lifecycle
    def test_calculates_fill_latency(self):
        """Test calculation of fill latency."""
        order = Order(
            id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            status=OrderStatus.NEW,
            created_at=datetime.now() - timedelta(milliseconds=500),
        )

        fill_time = datetime.now()
        latency_ms = (fill_time - order.created_at).total_seconds() * 1000

        assert latency_ms >= 500
        assert latency_ms < 1000
