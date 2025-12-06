"""
Order Submission and Management Endpoints

Provides REST endpoints for:
- Submitting new orders
- Querying order status
- Cancelling orders
- Order history

UPGRADE-008: REST API Server (December 2025)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class OrderSide(str, Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class ExecutionType(str, Enum):
    """Order execution type."""

    OPTION_STRATEGY = "option_strategy"
    MANUAL_LEGS = "manual_legs"
    EQUITY = "equity"
    TWO_PART_SPREAD = "two_part_spread"


class OrderPriority(str, Enum):
    """Order priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class LegDefinition(BaseModel):
    """Option leg definition for manual orders."""

    symbol: str = Field(..., description="Option contract symbol")
    side: OrderSide = Field(..., description="Buy or sell")
    quantity: int = Field(1, ge=1, description="Number of contracts")
    limit_price: float | None = Field(None, description="Limit price per contract")


class OrderSubmission(BaseModel):
    """Order submission request body."""

    symbol: str = Field(..., description="Underlying symbol (e.g., SPY)")
    execution_type: ExecutionType = Field(..., description="How to execute the order")
    strategy_name: str | None = Field(None, description="Strategy name for option_strategy type")
    legs: list[LegDefinition] | None = Field(None, description="Legs for manual_legs type")
    quantity: int = Field(1, ge=1, description="Order quantity")
    limit_price: float | None = Field(None, description="Limit price (net for spreads)")
    priority: OrderPriority = Field(OrderPriority.NORMAL, description="Order priority")
    notes: str | None = Field(None, max_length=500, description="Order notes")
    expiration_minutes: int | None = Field(None, ge=1, le=1440, description="Order expiration in minutes")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "SPY",
                "execution_type": "option_strategy",
                "strategy_name": "iron_condor",
                "quantity": 1,
                "limit_price": 2.50,
                "priority": "normal",
                "notes": "Weekly iron condor on SPY",
            }
        }
    )


class OrderResponse(BaseModel):
    """Order submission response."""

    order_id: str
    status: str
    created_at: datetime
    message: str


class OrderDetails(BaseModel):
    """Full order details."""

    order_id: str
    symbol: str
    execution_type: str
    strategy_name: str | None
    quantity: int
    limit_price: float | None
    status: str
    priority: str
    created_at: datetime
    updated_at: datetime | None
    filled_at: datetime | None
    fill_price: float | None
    notes: str | None
    error_message: str | None


class OrderListResponse(BaseModel):
    """Order list response."""

    orders: list[OrderDetails]
    count: int
    total: int


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/orders", response_model=OrderResponse, status_code=201)
async def submit_order(order: OrderSubmission):
    """Submit a new order to the queue.

    The order will be added to the pending queue and processed by the
    algorithm during the next OnData cycle.

    Returns:
        OrderResponse with order ID and status
    """
    from ..rest_server import get_order_queue, get_ws_manager
    from ..websocket_handler import EventType

    order_queue = get_order_queue()
    if order_queue is None:
        raise HTTPException(
            status_code=503,
            detail="Order queue not initialized. Server may be starting up.",
        )

    try:
        # Convert legs to dict format if present
        legs_data = None
        if order.legs:
            legs_data = [
                {
                    "symbol": leg.symbol,
                    "side": leg.side.value,
                    "quantity": leg.quantity,
                    "limit_price": leg.limit_price,
                }
                for leg in order.legs
            ]

        # Submit to queue
        order_request = order_queue.submit_order(
            symbol=order.symbol,
            execution_type=order.execution_type.value,
            strategy_name=order.strategy_name,
            legs=legs_data,
            quantity=order.quantity,
            limit_price=order.limit_price,
            priority=order.priority.value,
            notes=order.notes,
        )

        logger.info(f"Order submitted: {order_request.order_id} ({order.symbol})")

        # Broadcast to WebSocket clients
        ws_manager = get_ws_manager()
        if ws_manager:
            await ws_manager.broadcast_order_event(
                EventType.ORDER_SUBMITTED,
                order_request.order_id,
                {
                    "symbol": order.symbol,
                    "execution_type": order.execution_type.value,
                    "quantity": order.quantity,
                },
            )

        return OrderResponse(
            order_id=order_request.order_id,
            status="pending",
            created_at=datetime.now(timezone.utc),
            message="Order submitted successfully",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting order: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/orders", response_model=OrderListResponse)
async def list_orders(
    status: str | None = Query(None, description="Filter by status"),
    symbol: str | None = Query(None, description="Filter by symbol"),
    execution_type: str | None = Query(None, description="Filter by execution type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """List orders with optional filters.

    Returns:
        OrderListResponse with filtered orders
    """
    from ..rest_server import get_order_queue

    order_queue = get_order_queue()
    if order_queue is None:
        raise HTTPException(status_code=503, detail="Order queue not initialized")

    try:
        # Get all orders (queue returns pending orders)
        all_orders = order_queue.get_all_orders()

        # Apply filters
        filtered = all_orders
        if status:
            filtered = [o for o in filtered if o.status.value == status]
        if symbol:
            filtered = [o for o in filtered if o.request.symbol == symbol]
        if execution_type:
            filtered = [o for o in filtered if o.request.execution_type == execution_type]

        total = len(filtered)

        # Apply pagination
        paginated = filtered[offset : offset + limit]

        # Convert to response format
        orders = [
            OrderDetails(
                order_id=o.order_id,
                symbol=o.request.symbol,
                execution_type=o.request.execution_type,
                strategy_name=o.request.strategy_name,
                quantity=o.request.quantity,
                limit_price=o.request.limit_price,
                status=o.status.value,
                priority=o.priority.value,
                created_at=o.created_at,
                updated_at=o.updated_at,
                filled_at=o.filled_at,
                fill_price=o.fill_price,
                notes=o.request.notes,
                error_message=o.error_message,
            )
            for o in paginated
        ]

        return OrderListResponse(
            orders=orders,
            count=len(orders),
            total=total,
        )

    except Exception as e:
        logger.error(f"Error listing orders: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/orders/{order_id}", response_model=OrderDetails)
async def get_order(order_id: str):
    """Get order details by ID.

    Args:
        order_id: Unique order identifier

    Returns:
        OrderDetails for the specified order
    """
    from ..rest_server import get_order_queue

    order_queue = get_order_queue()
    if order_queue is None:
        raise HTTPException(status_code=503, detail="Order queue not initialized")

    order = order_queue.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail=f"Order not found: {order_id}")

    return OrderDetails(
        order_id=order.order_id,
        symbol=order.request.symbol,
        execution_type=order.request.execution_type,
        strategy_name=order.request.strategy_name,
        quantity=order.request.quantity,
        limit_price=order.request.limit_price,
        status=order.status.value,
        priority=order.priority.value,
        created_at=order.created_at,
        updated_at=order.updated_at,
        filled_at=order.filled_at,
        fill_price=order.fill_price,
        notes=order.request.notes,
        error_message=order.error_message,
    )


@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel a pending order.

    Args:
        order_id: Unique order identifier

    Returns:
        Confirmation message
    """
    from ..rest_server import get_order_queue, get_ws_manager
    from ..websocket_handler import EventType

    order_queue = get_order_queue()
    if order_queue is None:
        raise HTTPException(status_code=503, detail="Order queue not initialized")

    # Check if order exists
    order = order_queue.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail=f"Order not found: {order_id}")

    # Try to cancel
    success = order_queue.cancel_order(order_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel order in status: {order.status.value}",
        )

    logger.info(f"Order cancelled: {order_id}")

    # Broadcast to WebSocket clients
    ws_manager = get_ws_manager()
    if ws_manager:
        await ws_manager.broadcast_order_event(
            EventType.ORDER_CANCELLED,
            order_id,
        )

    return {
        "message": "Order cancelled successfully",
        "order_id": order_id,
    }


@router.get("/orders/pending/count")
async def get_pending_count():
    """Get count of pending orders.

    Returns:
        Count of orders in pending status
    """
    from ..rest_server import get_order_queue

    order_queue = get_order_queue()
    if order_queue is None:
        raise HTTPException(status_code=503, detail="Order queue not initialized")

    pending = order_queue.get_pending_orders()
    return {"pending_count": len(pending)}
