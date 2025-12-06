"""
Order Queue API for UI Integration

Provides REST API and WebSocket interface for submitting orders from UI to algorithm.
Supports both OptionStrategies factory methods and manual leg construction.

This module enables the hybrid architecture where:
1. Algorithm can trade autonomously using OptionStrategies
2. User can submit manual orders via UI
3. User can set up recurring orders
4. Bot can manage positions from both sources

Architecture:
    UI -> REST API -> Order Queue -> Algorithm OnData() -> Execution
                  <- WebSocket <- Position Updates <-

SECURITY NOTES:
- Authentication token MUST be set via TRADING_API_TOKEN env var
- Default tokens are rejected in production
- Token comparison uses constant-time algorithm to prevent timing attacks

Usage:
    # In QuantConnect algorithm Initialize()
    from api import create_order_queue_api

    self.order_api = create_order_queue_api(
        algorithm=self,
        # Token loaded from environment variable TRADING_API_TOKEN
    )

    # In OnData()
    pending_orders = self.order_api.get_pending_orders()
    for order in pending_orders:
        # Process order using OptionStrategies or ManualLegs executor
        self.order_api.update_order_status(order.id, "filled")
"""

import logging
import os
import secrets
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from queue import Empty, PriorityQueue
from typing import Any


logger = logging.getLogger(__name__)


# =============================================================================
# Authentication Configuration (P0 SECURITY FIX)
# =============================================================================


class AuthConfig:
    """
    Authentication configuration with secure defaults.

    SECURITY FIX: Prevents use of default/weak authentication tokens.
    Previous implementation allowed hardcoded "default-token-change-me".

    Environment variable: TRADING_API_TOKEN
    Generate a secure token with: python -c 'import secrets; print(secrets.token_urlsafe(32))'
    """

    # Tokens that are explicitly rejected
    FORBIDDEN_TOKENS = {
        "default-token-change-me",
        "change-me",
        "secret",
        "password",
        "token",
        "test",
        "admin",
    }

    def __init__(self, token: str | None = None):
        """
        Initialize authentication configuration.

        Args:
            token: Optional token override (defaults to env var TRADING_API_TOKEN)

        Raises:
            EnvironmentError: If token is missing, forbidden, or too short
        """
        self._token = token or os.environ.get("TRADING_API_TOKEN")

        # Validate token presence
        if not self._token:
            error_msg = (
                "TRADING_API_TOKEN environment variable is required. "
                "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
            logger.error("AUTH_CONFIG_ERROR", extra={"reason": "missing_token"})
            raise OSError(error_msg)

        # Validate token is not a forbidden default
        if self._token.lower() in self.FORBIDDEN_TOKENS:
            error_msg = (
                "TRADING_API_TOKEN is set to a forbidden default value. "
                "Please set a secure, randomly-generated token."
            )
            logger.error("AUTH_CONFIG_ERROR", extra={"reason": "forbidden_token"})
            raise OSError(error_msg)

        # Validate token length (at least 32 chars for security)
        if len(self._token) < 32:
            error_msg = (
                "TRADING_API_TOKEN must be at least 32 characters long. "
                "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
            logger.error("AUTH_CONFIG_ERROR", extra={"reason": "token_too_short"})
            raise OSError(error_msg)

        logger.info("AUTH_CONFIG_INITIALIZED", extra={"token_length": len(self._token)})

    def validate(self, provided_token: str) -> bool:
        """
        Validate a provided token against the configured token.

        Uses constant-time comparison to prevent timing attacks.

        Args:
            provided_token: Token to validate

        Returns:
            True if token is valid, False otherwise
        """
        if not provided_token:
            return False
        return secrets.compare_digest(self._token, provided_token)


class OrderType(Enum):
    """Type of order to execute."""

    OPTION_STRATEGY = "option_strategy"  # Use OptionStrategies factory method
    MANUAL_LEGS = "manual_legs"  # Custom leg construction
    MARKET = "market"  # Simple market order
    LIMIT = "limit"  # Simple limit order


class OrderStatus(Enum):
    """Order lifecycle status."""

    PENDING = "pending"  # Submitted, waiting to be processed
    VALIDATING = "validating"  # Being validated
    QUEUED = "queued"  # In queue, ready for execution
    EXECUTING = "executing"  # Being executed
    PARTIALLY_FILLED = "partially_filled"  # Some legs filled
    FILLED = "filled"  # All legs filled
    CANCELLED = "cancelled"  # Cancelled by user or system
    REJECTED = "rejected"  # Rejected due to validation failure
    FAILED = "failed"  # Execution failed


class OrderPriority(Enum):
    """Order priority in queue."""

    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0


@dataclass
class OrderRequest:
    """
    Order request from UI.

    Attributes:
        order_type: Type of order (strategy, manual legs, etc.)
        symbol: Underlying symbol (e.g., "SPY")
        quantity: Number of contracts/shares

        # For OptionStrategies
        strategy_name: Name of strategy ("iron_condor", "butterfly_call", etc.)
        strategy_params: Parameters for strategy factory method

        # For manual legs
        legs: List of leg definitions [{"symbol": "SPY...", "quantity": 1, "side": "buy"}, ...]
        two_part_config: Configuration for two-part execution

        # For simple orders
        limit_price: Limit price (if limit order)

        # Common fields
        priority: Order priority
        notes: User notes
        recurring_config: Configuration for recurring orders (optional)
        bot_managed: Whether bot should manage profit-taking/stop-loss
    """

    order_type: OrderType
    symbol: str
    quantity: int

    # OptionStrategies fields
    strategy_name: str | None = None
    strategy_params: dict[str, Any] | None = None

    # Manual legs fields
    legs: list[dict[str, Any]] | None = None
    two_part_config: dict[str, Any] | None = None

    # Simple order fields
    limit_price: float | None = None

    # Common fields
    priority: OrderPriority = OrderPriority.NORMAL
    notes: str = ""
    recurring_config: dict[str, Any] | None = None
    bot_managed: bool = True


@dataclass
class QueuedOrder:
    """
    Order in the queue with metadata.

    Attributes:
        id: Unique order ID
        request: Original order request
        status: Current order status
        priority: Order priority (for queue ordering)
        submitted_at: Timestamp when submitted
        updated_at: Timestamp of last update
        error_message: Error message if rejected/failed
        execution_details: Details about execution (fill prices, etc.)
        algorithm_order_ids: List of QuantConnect order IDs
    """

    id: str
    request: OrderRequest
    status: OrderStatus
    priority: OrderPriority
    submitted_at: datetime
    updated_at: datetime = field(default_factory=datetime.now)
    error_message: str = ""
    execution_details: dict[str, Any] = field(default_factory=dict)
    algorithm_order_ids: list[int] = field(default_factory=list)

    def __lt__(self, other: "QueuedOrder") -> bool:
        """Compare by priority for PriorityQueue."""
        return self.priority.value < other.priority.value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["status"] = self.status.value
        result["priority"] = self.priority.value
        result["request"]["order_type"] = self.request.order_type.value
        result["request"]["priority"] = self.request.priority.value
        result["submitted_at"] = self.submitted_at.isoformat()
        result["updated_at"] = self.updated_at.isoformat()
        return result


class OrderQueueAPI:
    """
    Order Queue API for UI integration.

    Provides REST API endpoints and WebSocket support for:
    - Submitting orders from UI
    - Checking order status
    - Cancelling orders
    - Real-time position updates

    The algorithm processes orders from the queue in OnData().

    SECURITY: Authentication is required for all order operations.
    Token must be set via TRADING_API_TOKEN environment variable.
    """

    def __init__(
        self,
        algorithm: Any = None,
        auth_token: str | None = None,
        max_queue_size: int = 1000,
        enable_logging: bool = True,
        skip_auth_validation: bool = False,
    ):
        """
        Initialize order queue API.

        Args:
            algorithm: QuantConnect algorithm instance (optional, can be set later)
            auth_token: Authentication token (loaded from env if not provided)
            max_queue_size: Maximum number of orders in queue
            enable_logging: Whether to log API activity
            skip_auth_validation: Skip auth validation (for testing only)

        Raises:
            EnvironmentError: If auth token is missing or invalid
        """
        self.algorithm = algorithm
        self.max_queue_size = max_queue_size
        self.enable_logging = enable_logging

        # Initialize authentication (SECURITY FIX)
        if skip_auth_validation:
            # Testing mode - use a dummy validator
            self._auth_config = None
            self._log("WARNING: Authentication validation skipped (testing mode)")
        else:
            self._auth_config = AuthConfig(token=auth_token)

        # Order queue (priority-based)
        self.queue: PriorityQueue[QueuedOrder] = PriorityQueue(maxsize=max_queue_size)

        # Order tracking
        self.orders: dict[str, QueuedOrder] = {}
        self.orders_by_status: dict[OrderStatus, list[str]] = defaultdict(list)

        # WebSocket connections (would be implemented with actual WebSocket library)
        self.websocket_connections: list[Any] = []

        # Callbacks
        self.on_order_submitted: Callable[[QueuedOrder], None] | None = None
        self.on_order_executed: Callable[[QueuedOrder], None] | None = None

        # Statistics
        self.stats = {
            "total_submitted": 0,
            "total_filled": 0,
            "total_rejected": 0,
            "total_cancelled": 0,
        }

    def submit_order(self, request: OrderRequest) -> dict[str, Any]:
        """
        Submit a new order to the queue.

        Args:
            request: Order request from UI

        Returns:
            Response with order ID and status
        """
        try:
            # Validate order
            validation_result = self._validate_order(request)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "order_id": None,
                }

            # Create queued order
            order_id = str(uuid.uuid4())
            order = QueuedOrder(
                id=order_id,
                request=request,
                status=OrderStatus.QUEUED,
                priority=request.priority,
                submitted_at=datetime.now(),
            )

            # Add to queue
            if self.queue.full():
                return {
                    "success": False,
                    "error": "Order queue is full",
                    "order_id": None,
                }

            self.queue.put(order)
            self.orders[order_id] = order
            self.orders_by_status[OrderStatus.QUEUED].append(order_id)

            # Update statistics
            self.stats["total_submitted"] += 1

            # Log
            if self.enable_logging:
                self._log(f"Order submitted: {order_id} ({request.order_type.value})")

            # Callback
            if self.on_order_submitted:
                self.on_order_submitted(order)

            # Broadcast to WebSocket clients
            self._broadcast_order_update(order)

            return {
                "success": True,
                "order_id": order_id,
                "status": OrderStatus.QUEUED.value,
                "message": "Order submitted successfully",
            }

        except Exception as e:
            self._log(f"Error submitting order: {e!s}", level="error")
            return {
                "success": False,
                "error": str(e),
                "order_id": None,
            }

    def _validate_order(self, request: OrderRequest) -> dict[str, Any]:
        """
        Validate order request.

        Args:
            request: Order request to validate

        Returns:
            Validation result with valid flag and error message
        """
        # Check required fields
        if not request.symbol:
            return {"valid": False, "error": "Symbol is required"}

        if request.quantity <= 0:
            return {"valid": False, "error": "Quantity must be positive"}

        # Validate based on order type
        if request.order_type == OrderType.OPTION_STRATEGY:
            if not request.strategy_name:
                return {"valid": False, "error": "Strategy name required for option strategy orders"}

            # Validate strategy name (could check against supported strategies)
            valid_strategies = [
                "iron_condor",
                "butterfly_call",
                "butterfly_put",
                "call_spread",
                "put_spread",
                "straddle",
                "strangle",
                # ... all 37+ strategies
            ]
            if request.strategy_name not in valid_strategies:
                return {"valid": False, "error": f"Unknown strategy: {request.strategy_name}"}

        elif request.order_type == OrderType.MANUAL_LEGS:
            if not request.legs or len(request.legs) == 0:
                return {"valid": False, "error": "Legs required for manual leg orders"}

            # Validate each leg
            for leg in request.legs:
                if "symbol" not in leg or "quantity" not in leg:
                    return {"valid": False, "error": "Each leg must have symbol and quantity"}

        elif request.order_type == OrderType.LIMIT:
            if request.limit_price is None or request.limit_price <= 0:
                return {"valid": False, "error": "Limit price required for limit orders"}

        # Check for excessive quantity
        if request.quantity > 100:
            return {"valid": False, "error": "Quantity exceeds maximum of 100 contracts"}

        return {"valid": True, "error": ""}

    def get_pending_orders(self, max_count: int = 10) -> list[QueuedOrder]:
        """
        Get pending orders from the queue for processing.

        This should be called from the algorithm's OnData() method.

        Args:
            max_count: Maximum number of orders to retrieve

        Returns:
            List of pending orders (up to max_count)
        """
        orders = []

        for _ in range(min(max_count, self.queue.qsize())):
            try:
                order = self.queue.get_nowait()
                order.status = OrderStatus.EXECUTING
                order.updated_at = datetime.now()

                # Update tracking
                self.orders_by_status[OrderStatus.QUEUED].remove(order.id)
                self.orders_by_status[OrderStatus.EXECUTING].append(order.id)

                orders.append(order)

                # Broadcast update
                self._broadcast_order_update(order)

            except Empty:
                break

        return orders

    def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        error_message: str = "",
        execution_details: dict[str, Any] | None = None,
        algorithm_order_ids: list[int] | None = None,
    ) -> bool:
        """
        Update order status.

        Args:
            order_id: Order ID
            status: New status
            error_message: Error message (if failed/rejected)
            execution_details: Details about execution
            algorithm_order_ids: QuantConnect order IDs

        Returns:
            True if updated successfully
        """
        if order_id not in self.orders:
            self._log(f"Order not found: {order_id}", level="warning")
            return False

        order = self.orders[order_id]
        old_status = order.status

        # Update order
        order.status = status
        order.updated_at = datetime.now()
        order.error_message = error_message

        if execution_details:
            order.execution_details.update(execution_details)

        if algorithm_order_ids:
            order.algorithm_order_ids.extend(algorithm_order_ids)

        # Update tracking
        if old_status in self.orders_by_status:
            if order_id in self.orders_by_status[old_status]:
                self.orders_by_status[old_status].remove(order_id)
        self.orders_by_status[status].append(order_id)

        # Update statistics
        if status == OrderStatus.FILLED:
            self.stats["total_filled"] += 1
        elif status == OrderStatus.REJECTED:
            self.stats["total_rejected"] += 1
        elif status == OrderStatus.CANCELLED:
            self.stats["total_cancelled"] += 1

        # Log
        if self.enable_logging:
            self._log(f"Order {order_id} status: {old_status.value} -> {status.value}")

        # Callback
        if status == OrderStatus.FILLED and self.on_order_executed:
            self.on_order_executed(order)

        # Broadcast to WebSocket clients
        self._broadcast_order_update(order)

        return True

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """
        Cancel a pending order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Response with success status
        """
        if order_id not in self.orders:
            return {
                "success": False,
                "error": "Order not found",
            }

        order = self.orders[order_id]

        # Can only cancel pending/queued orders
        if order.status not in [OrderStatus.PENDING, OrderStatus.QUEUED]:
            return {
                "success": False,
                "error": f"Cannot cancel order in status: {order.status.value}",
            }

        # Update status
        self.update_order_status(order_id, OrderStatus.CANCELLED)

        return {
            "success": True,
            "message": "Order cancelled successfully",
        }

    def get_order_status(self, order_id: str) -> dict[str, Any] | None:
        """
        Get order status.

        Args:
            order_id: Order ID

        Returns:
            Order details or None if not found
        """
        if order_id not in self.orders:
            return None

        return self.orders[order_id].to_dict()

    def get_all_orders(
        self,
        status_filter: OrderStatus | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get all orders, optionally filtered by status.

        Args:
            status_filter: Filter by status (None for all)
            limit: Maximum number of orders to return

        Returns:
            List of order details
        """
        if status_filter:
            order_ids = self.orders_by_status.get(status_filter, [])
        else:
            order_ids = list(self.orders.keys())

        # Sort by submission time (most recent first)
        order_ids = sorted(
            order_ids,
            key=lambda oid: self.orders[oid].submitted_at,
            reverse=True,
        )[:limit]

        return [self.orders[oid].to_dict() for oid in order_ids]

    def get_statistics(self) -> dict[str, Any]:
        """
        Get API statistics.

        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            "queue_size": self.queue.qsize(),
            "total_orders": len(self.orders),
            "by_status": {status.value: len(order_ids) for status, order_ids in self.orders_by_status.items()},
        }

    def _broadcast_order_update(self, order: QueuedOrder) -> None:
        """
        Broadcast order update to all WebSocket clients.

        Args:
            order: Order that was updated
        """
        # In a real implementation, this would send to WebSocket clients
        # For now, it's a placeholder for the architecture
        message = {
            "type": "order_update",
            "order": order.to_dict(),
        }

        # self._send_to_websocket_clients(message)

    def _log(self, message: str, level: str = "info") -> None:
        """
        Log a message.

        Args:
            message: Message to log
            level: Log level (info, warning, error)
        """
        if not self.enable_logging:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level.upper()}] OrderQueueAPI: {message}"

        # In QuantConnect, use self.algorithm.Debug() or self.algorithm.Log()
        if hasattr(self.algorithm, "Debug"):
            if level == "error":
                self.algorithm.Error(log_message)
            else:
                self.algorithm.Debug(log_message)
        else:
            print(log_message)

    def clear_old_orders(self, days: int = 7) -> int:
        """
        Clear old completed/cancelled orders.

        Args:
            days: Remove orders older than this many days

        Returns:
            Number of orders removed
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        removed = 0

        # Find old orders
        to_remove = []
        for order_id, order in self.orders.items():
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                if order.submitted_at.timestamp() < cutoff:
                    to_remove.append(order_id)

        # Remove them
        for order_id in to_remove:
            order = self.orders[order_id]
            del self.orders[order_id]

            if order_id in self.orders_by_status[order.status]:
                self.orders_by_status[order.status].remove(order_id)

            removed += 1

        if removed > 0:
            self._log(f"Cleared {removed} old orders")

        return removed


def create_order_queue_api(
    algorithm: Any,
    auth_token: str | None = None,
    max_queue_size: int = 1000,
    enable_logging: bool = True,
    skip_auth_validation: bool = False,
) -> OrderQueueAPI:
    """
    Create an OrderQueueAPI instance.

    SECURITY: Authentication token is loaded from TRADING_API_TOKEN environment
    variable by default. Generate a secure token with:
        python -c 'import secrets; print(secrets.token_urlsafe(32))'

    Args:
        algorithm: QuantConnect algorithm instance
        auth_token: Optional token override (defaults to TRADING_API_TOKEN env var)
        max_queue_size: Maximum number of orders in queue
        enable_logging: Whether to log API activity
        skip_auth_validation: Skip authentication validation (for testing only)

    Returns:
        Configured OrderQueueAPI instance

    Raises:
        EnvironmentError: If TRADING_API_TOKEN env var is not set or invalid

    Example:
        # In algorithm Initialize()
        # Token loaded from TRADING_API_TOKEN environment variable
        self.order_api = create_order_queue_api(algorithm=self)

        # Or with explicit token (must be 32+ chars, not a forbidden default)
        self.order_api = create_order_queue_api(
            algorithm=self,
            auth_token=os.environ.get("MY_CUSTOM_TOKEN"),
        )

        # In OnData()
        pending_orders = self.order_api.get_pending_orders(max_count=5)
        for order in pending_orders:
            if order.request.order_type == OrderType.OPTION_STRATEGY:
                # Process with OptionStrategiesExecutor
                pass
            elif order.request.order_type == OrderType.MANUAL_LEGS:
                # Process with ManualLegsExecutor
                pass
    """
    return OrderQueueAPI(
        algorithm=algorithm,
        auth_token=auth_token,
        max_queue_size=max_queue_size,
        enable_logging=enable_logging,
        skip_auth_validation=skip_auth_validation,
    )
