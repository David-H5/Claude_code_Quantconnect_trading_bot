"""
API Module

Provides REST API and WebSocket interface for UI order submission:
- Order queue management for UI-submitted orders
- Real-time position and order status updates
- Authentication and authorization
- Order validation and sanitization
- Support for both OptionStrategies and manual leg orders

UPGRADE-008: REST API Server (December 2025)

Layer: 4 (Applications)
May import from: Layers 0-3 (all lower layers)
This is the top layer - nothing should import from api.
"""

from .order_queue_api import (
    OrderPriority,
    OrderQueueAPI,
    OrderRequest,
    OrderStatus,
    OrderType,
    QueuedOrder,
    create_order_queue_api,
)

# REST API Server (UPGRADE-008)
from .rest_server import (
    app,
    create_app,
    get_algorithm,
    get_order_queue,
    get_ws_manager,
    run_server,
    set_algorithm,
)
from .websocket_handler import (
    EventType,
    WebSocketClient,
    WebSocketManager,
)


__all__ = [
    # Order Queue API
    "OrderRequest",
    "OrderType",
    "OrderStatus",
    "OrderPriority",
    "QueuedOrder",
    "OrderQueueAPI",
    "create_order_queue_api",
    # REST API Server (UPGRADE-008)
    "create_app",
    "run_server",
    "get_order_queue",
    "get_ws_manager",
    "get_algorithm",
    "set_algorithm",
    "app",
    # WebSocket Handler
    "WebSocketManager",
    "WebSocketClient",
    "EventType",
]
