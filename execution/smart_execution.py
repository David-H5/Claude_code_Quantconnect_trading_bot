"""
Smart Execution Model

Implements intelligent order execution with automatic cancel/replace
for unfilled orders, aggressive price improvement, and execution analytics.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from config import OrderExecutionConfig


logger = logging.getLogger(__name__)


class SmartOrderStatus(Enum):
    """Order status enumeration for smart execution tracking."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SmartOrderType(Enum):
    """
    Order type enumeration for smart execution.

    Note: Renamed from 'OrderType' to avoid collision with QuantConnect's
    built-in OrderType class (from AlgorithmImports).
    """

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class ExecutionOrder:
    """Represents an order being managed by smart execution."""

    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    order_type: SmartOrderType
    limit_price: float | None = None
    stop_price: float | None = None
    status: SmartOrderStatus = SmartOrderStatus.PENDING
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    original_price: float = 0.0
    current_price: float = 0.0
    cancel_replace_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def unfilled_quantity(self) -> int:
        """Remaining unfilled quantity."""
        return self.quantity - self.filled_quantity

    @property
    def is_complete(self) -> bool:
        """Check if order is complete (filled or cancelled)."""
        return self.status in (SmartOrderStatus.FILLED, SmartOrderStatus.CANCELLED, SmartOrderStatus.REJECTED)

    @property
    def age_seconds(self) -> float:
        """Time since order creation in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "status": self.status.value,
            "cancel_replace_count": self.cancel_replace_count,
            "age_seconds": self.age_seconds,
            "average_fill_price": self.average_fill_price,
        }


@dataclass
class ExecutionResult:
    """Result of an execution attempt."""

    success: bool
    order: ExecutionOrder
    message: str
    slippage: float = 0.0
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "order": self.order.to_dict(),
            "message": self.message,
            "slippage": self.slippage,
            "execution_time_ms": self.execution_time_ms,
        }


class SmartExecutionModel:
    """
    Smart order execution with automatic cancel/replace.

    Features:
    - Automatic price improvement through cancel/replace
    - Configurable maximum bid increase
    - Incremental price adjustments
    - Execution analytics and tracking
    """

    def __init__(
        self,
        config: OrderExecutionConfig,
        order_callback: Callable[[ExecutionOrder, str], None] | None = None,
    ):
        """
        Initialize smart execution model.

        Args:
            config: Execution configuration
            order_callback: Callback when order action taken (order, action)
        """
        self.config = config
        self.order_callback = order_callback

        # Active orders being managed
        self._orders: dict[str, ExecutionOrder] = {}

        # Execution statistics
        self._stats = {
            "total_orders": 0,
            "successful_fills": 0,
            "cancel_replaces": 0,
            "total_slippage": 0.0,
        }

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: SmartOrderType = SmartOrderType.LIMIT,
        limit_price: float | None = None,
        bid_price: float | None = None,
        ask_price: float | None = None,
    ) -> ExecutionOrder:
        """
        Submit an order for smart execution.

        Args:
            symbol: Symbol to trade
            side: "buy" or "sell"
            quantity: Order quantity
            order_type: Type of order
            limit_price: Limit price (required for limit orders)
            bid_price: Current bid price
            ask_price: Current ask price

        Returns:
            ExecutionOrder object
        """
        # Generate order ID
        order_id = f"{symbol}_{int(time.time() * 1000)}"

        # Calculate initial price if using mid-price
        if limit_price is None and self.config.use_mid_price:
            if bid_price and ask_price:
                limit_price = (bid_price + ask_price) / 2

        order = ExecutionOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            original_price=limit_price or 0.0,
            current_price=limit_price or 0.0,
            status=SmartOrderStatus.SUBMITTED,
        )

        self._orders[order_id] = order
        self._stats["total_orders"] += 1

        if self.order_callback:
            self.order_callback(order, "submitted")

        return order

    def update_order_status(
        self,
        order_id: str,
        status: SmartOrderStatus,
        filled_quantity: int = 0,
        fill_price: float = 0.0,
    ) -> ExecutionOrder | None:
        """
        Update order status from broker.

        Args:
            order_id: Order ID
            status: New status
            filled_quantity: Total quantity filled (cumulative)
            fill_price: Price of latest fill

        Returns:
            Updated order or None
        """
        if order_id not in self._orders:
            return None

        order = self._orders[order_id]
        order.status = status
        order.last_updated = datetime.now()

        if fill_price > 0 and filled_quantity > 0:
            # Calculate proper weighted average fill price
            # SAFETY FIX: Previous implementation was mathematically incorrect
            old_filled_qty = order.filled_quantity
            new_fill_qty = filled_quantity - old_filled_qty

            if new_fill_qty > 0:
                if old_filled_qty == 0:
                    # First fill
                    order.average_fill_price = fill_price
                else:
                    # Weighted average: (old_cost + new_cost) / total_qty
                    old_cost = order.average_fill_price * old_filled_qty
                    new_cost = fill_price * new_fill_qty
                    total_qty = old_filled_qty + new_fill_qty
                    order.average_fill_price = (old_cost + new_cost) / total_qty

                logger.debug(
                    "FILL_PRICE_UPDATED",
                    extra={
                        "order_id": order_id,
                        "old_qty": old_filled_qty,
                        "new_qty": new_fill_qty,
                        "fill_price": fill_price,
                        "avg_price": order.average_fill_price,
                    },
                )

        # Update filled quantity after calculating average
        order.filled_quantity = filled_quantity

        if status == SmartOrderStatus.FILLED:
            self._stats["successful_fills"] += 1
            # Calculate slippage
            if order.original_price > 0:
                slippage = abs(order.average_fill_price - order.original_price)
                self._stats["total_slippage"] += slippage

        return order

    def check_and_replace(
        self,
        order_id: str,
        current_bid: float,
        current_ask: float,
    ) -> ExecutionOrder | None:
        """
        Check if order should be cancelled and replaced.

        Args:
            order_id: Order ID
            current_bid: Current bid price
            current_ask: Current ask price

        Returns:
            Replaced order or None if no action needed
        """
        if not self.config.cancel_replace_enabled:
            return None

        if order_id not in self._orders:
            return None

        order = self._orders[order_id]

        # Skip if complete
        if order.is_complete:
            return None

        # Skip if not enough time has passed
        if order.age_seconds < self.config.cancel_after_seconds:
            return None

        # Skip if max attempts reached
        if order.cancel_replace_count >= self.config.max_cancel_replace_attempts:
            return None

        # Calculate new price
        new_price = self._calculate_replacement_price(order, current_bid, current_ask)

        if new_price is None:
            return None

        # Check if price increase exceeds maximum
        price_increase_pct = (new_price - order.original_price) / order.original_price
        if price_increase_pct > self.config.max_bid_increase_pct:
            # Hit max increase, don't replace
            return None

        # Perform cancel/replace
        order.limit_price = new_price
        order.current_price = new_price
        order.cancel_replace_count += 1
        order.last_updated = datetime.now()
        order.status = SmartOrderStatus.SUBMITTED

        self._stats["cancel_replaces"] += 1

        if self.order_callback:
            self.order_callback(order, "replaced")

        return order

    def _calculate_replacement_price(
        self,
        order: ExecutionOrder,
        current_bid: float,
        current_ask: float,
    ) -> float | None:
        """Calculate new price for replacement order."""
        if order.limit_price is None:
            return None

        increment_pct = self.config.bid_increment_pct
        current_price = order.limit_price

        if order.side == "buy":
            # For buys, increase bid
            new_price = current_price * (1 + increment_pct)
            # Don't exceed ask
            new_price = min(new_price, current_ask)
        else:
            # For sells, decrease ask
            new_price = current_price * (1 - increment_pct)
            # Don't go below bid
            new_price = max(new_price, current_bid)

        # Only replace if price actually changed
        if abs(new_price - current_price) < 0.01:
            return None

        return new_price

    def process_unfilled_orders(
        self,
        quote_data: dict[str, tuple[float, float]],
    ) -> list[ExecutionOrder]:
        """
        Process all unfilled orders and perform cancel/replace as needed.

        Args:
            quote_data: Dict of symbol -> (bid, ask) tuples

        Returns:
            List of orders that were replaced
        """
        replaced = []

        for order_id, order in list(self._orders.items()):
            if order.is_complete:
                continue

            if order.symbol not in quote_data:
                continue

            bid, ask = quote_data[order.symbol]
            result = self.check_and_replace(order_id, bid, ask)

            if result:
                replaced.append(result)

        return replaced

    def cancel_order(self, order_id: str) -> ExecutionOrder | None:
        """
        Cancel an order.

        Args:
            order_id: Order ID

        Returns:
            Cancelled order or None
        """
        if order_id not in self._orders:
            return None

        order = self._orders[order_id]
        order.status = SmartOrderStatus.CANCELLED
        order.last_updated = datetime.now()

        if self.order_callback:
            self.order_callback(order, "cancelled")

        return order

    def get_order(self, order_id: str) -> ExecutionOrder | None:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_active_orders(self) -> list[ExecutionOrder]:
        """Get all active (non-complete) orders."""
        return [o for o in self._orders.values() if not o.is_complete]

    def get_orders_for_symbol(self, symbol: str) -> list[ExecutionOrder]:
        """Get all orders for a symbol."""
        return [o for o in self._orders.values() if o.symbol == symbol]

    def cleanup_old_orders(self, max_age_hours: int = 24) -> int:
        """
        Remove completed orders older than max_age.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of orders removed
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []

        for order_id, order in self._orders.items():
            if order.is_complete and order.last_updated < cutoff:
                to_remove.append(order_id)

        for order_id in to_remove:
            del self._orders[order_id]

        return len(to_remove)

    def get_statistics(self) -> dict[str, Any]:
        """Get execution statistics."""
        active_orders = len(self.get_active_orders())
        total_orders = self._stats["total_orders"]
        successful = self._stats["successful_fills"]

        return {
            "total_orders": total_orders,
            "successful_fills": successful,
            "fill_rate": successful / total_orders if total_orders > 0 else 0,
            "cancel_replaces": self._stats["cancel_replaces"],
            "average_slippage": (self._stats["total_slippage"] / successful if successful > 0 else 0),
            "active_orders": active_orders,
        }


class SmartExecutionExecutionModel:
    """
    QuantConnect-compatible Execution Model.

    Integrates with Algorithm Framework's execution pipeline.

    INTEGRATION REQUIRED:
    To enable smart cancel/replace functionality, connect OnOrderEvent() in your algorithm:

    ```python
    def Initialize(self):
        self.execution_model = SmartExecutionExecutionModel(config)
        self.SetExecution(self.execution_model)

    def OnOrderEvent(self, order_event):
        # Update internal tracking for cancel/replace logic
        if order_event.Status == OrderStatus.Filled:
            self.execution_model._executor.update_order_status(
                order_event.OrderId,
                "filled",
                order_event.FillQuantity,
                order_event.FillPrice
            )
        elif order_event.Status == OrderStatus.Canceled:
            self.execution_model._executor.update_order_status(
                order_event.OrderId,
                "cancelled",
                0,
                0
            )
    ```
    """

    def __init__(self, config: OrderExecutionConfig):
        """Initialize the execution model."""
        self._executor = SmartExecutionModel(config)

    def Execute(
        self,
        algorithm: Any,
        targets: list[Any],
    ) -> list[Any]:
        """
        Execute portfolio targets with smart order management.

        Args:
            algorithm: QCAlgorithm instance
            targets: List of PortfolioTarget objects

        Returns:
            List of Order tickets
        """
        orders = []

        for target in targets:
            symbol = target.Symbol
            quantity = target.Quantity
            security = algorithm.Securities[symbol]

            if quantity == 0:
                continue

            # Get current quotes
            bid = security.BidPrice
            ask = security.AskPrice
            mid = (bid + ask) / 2

            # Determine side
            side = "buy" if quantity > 0 else "sell"

            # Create smart order for internal tracking
            order = self._executor.submit_order(
                symbol=str(symbol),
                side=side,
                quantity=abs(quantity),
                order_type=SmartOrderType.LIMIT,
                limit_price=mid,
                bid_price=bid,
                ask_price=ask,
            )

            # Submit to broker via QuantConnect algorithm
            # Use LimitOrder for better price control with smart execution
            try:
                ticket = algorithm.LimitOrder(
                    symbol,
                    quantity,  # Signed quantity (positive = buy, negative = sell)
                    mid,  # Limit price at mid-point
                )

                # Store ticket for tracking (link internal order to QC order ticket)
                if hasattr(order, "qc_order_id"):
                    order.qc_order_id = ticket.OrderId

                orders.append(ticket)

            except Exception as e:
                algorithm.Debug(f"Order submission failed for {symbol}: {e}")
                # Mark internal order as failed
                if hasattr(self._executor, "update_order_status"):
                    self._executor.update_order_status(order.order_id, "failed", 0, 0)

        return orders


def create_smart_execution_model(
    config: OrderExecutionConfig | None = None,
    order_callback: Callable[[ExecutionOrder, str], None] | None = None,
) -> SmartExecutionModel:
    """
    Create smart execution model from configuration.

    Args:
        config: Execution configuration
        order_callback: Optional callback for order events

    Returns:
        Configured SmartExecutionModel instance
    """
    if config is None:
        config = OrderExecutionConfig()

    return SmartExecutionModel(config, order_callback=order_callback)


# =============================================================================
# Slippage Budget Enforcement (P1 FIX)
# =============================================================================


@dataclass
class SlippageBudgetConfig:
    """
    Configuration for slippage budget enforcement.

    P1 FIX: Adds ability to halt trading when slippage exceeds budget.

    Attributes:
        max_slippage_per_trade_bps: Max slippage per individual trade (basis points)
        max_slippage_per_session_bps: Max cumulative slippage per session
        max_slippage_per_session_dollars: Max dollar slippage per session
        halt_on_breach: Whether to halt trading on budget breach
        alert_at_pct: Alert when this percentage of budget is consumed (0.0-1.0)
    """

    max_slippage_per_trade_bps: float = 25.0  # 25 bps per trade
    max_slippage_per_session_bps: float = 100.0  # 100 bps per session cumulative
    max_slippage_per_session_dollars: float = 1000.0  # $1000 per session
    halt_on_breach: bool = True
    alert_at_pct: float = 0.80  # Alert at 80% of budget


class SlippageBudget:
    """
    Slippage budget tracker and enforcer.

    P1 FIX: Enforces slippage limits to prevent runaway costs.

    Usage:
        budget = SlippageBudget(SlippageBudgetConfig(
            max_slippage_per_trade_bps=25.0,
            max_slippage_per_session_dollars=1000.0,
        ))

        # Check before trading
        if not budget.can_trade():
            logger.warning("Trading halted - slippage budget exceeded")
            return

        # Record slippage after each fill
        budget.record_slippage(
            slippage_bps=5.2,
            slippage_dollars=15.60,
            order_id="ORD001",
        )

        # Get budget status
        status = budget.get_status()
        print(f"Session slippage: ${status['session_dollars']:.2f}")
    """

    def __init__(self, config: SlippageBudgetConfig | None = None):
        """
        Initialize slippage budget tracker.

        Args:
            config: Budget configuration (defaults provided if None)
        """
        self.config = config or SlippageBudgetConfig()

        # Tracking
        self._session_slippage_bps: float = 0.0
        self._session_slippage_dollars: float = 0.0
        self._trade_count: int = 0
        self._breach_count: int = 0
        self._trading_halted: bool = False
        self._halt_reason: str = ""
        self._history: list[dict[str, Any]] = []

        # Callbacks
        self.on_alert: Callable[[str, float], None] | None = None
        self.on_halt: Callable[[str], None] | None = None

    def record_slippage(
        self,
        slippage_bps: float,
        slippage_dollars: float,
        order_id: str,
        symbol: str = "",
    ) -> bool:
        """
        Record slippage from an order fill.

        Args:
            slippage_bps: Slippage in basis points
            slippage_dollars: Slippage in dollars (adverse only)
            order_id: Order identifier
            symbol: Security symbol

        Returns:
            True if within budget, False if budget breached
        """
        self._trade_count += 1

        # Check per-trade limit
        if abs(slippage_bps) > self.config.max_slippage_per_trade_bps:
            self._breach_count += 1
            logger.warning(
                "SLIPPAGE_TRADE_BREACH",
                extra={
                    "order_id": order_id,
                    "slippage_bps": slippage_bps,
                    "limit_bps": self.config.max_slippage_per_trade_bps,
                },
            )

        # Update session totals (only track adverse slippage)
        if slippage_dollars > 0:
            self._session_slippage_dollars += slippage_dollars
            self._session_slippage_bps += abs(slippage_bps)

        # Record in history
        self._history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "order_id": order_id,
                "symbol": symbol,
                "slippage_bps": slippage_bps,
                "slippage_dollars": slippage_dollars,
                "session_total_bps": self._session_slippage_bps,
                "session_total_dollars": self._session_slippage_dollars,
            }
        )

        # Check budget alerts
        self._check_budget_alerts()

        return not self._trading_halted

    def _check_budget_alerts(self) -> None:
        """Check and trigger budget alerts if needed."""
        # Check session dollar limit
        dollar_pct = self._session_slippage_dollars / self.config.max_slippage_per_session_dollars

        if dollar_pct >= 1.0:
            self._halt_trading(
                f"Session slippage ${self._session_slippage_dollars:.2f} exceeds limit ${self.config.max_slippage_per_session_dollars:.2f}"
            )
        elif dollar_pct >= self.config.alert_at_pct:
            if self.on_alert:
                self.on_alert(f"Slippage at {dollar_pct:.0%} of budget", self._session_slippage_dollars)

        # Check session bps limit
        bps_pct = self._session_slippage_bps / self.config.max_slippage_per_session_bps

        if bps_pct >= 1.0:
            self._halt_trading(
                f"Session slippage {self._session_slippage_bps:.1f}bps exceeds limit {self.config.max_slippage_per_session_bps:.1f}bps"
            )

    def _halt_trading(self, reason: str) -> None:
        """Halt trading due to budget breach."""
        if not self.config.halt_on_breach:
            logger.warning("SLIPPAGE_BUDGET_EXCEEDED", extra={"reason": reason, "halt": False})
            return

        self._trading_halted = True
        self._halt_reason = reason
        logger.error("TRADING_HALTED_SLIPPAGE", extra={"reason": reason})

        if self.on_halt:
            self.on_halt(reason)

    def can_trade(self) -> bool:
        """Check if trading is allowed within budget."""
        return not self._trading_halted

    def reset_session(self) -> None:
        """Reset session tracking (call at start of trading day)."""
        self._session_slippage_bps = 0.0
        self._session_slippage_dollars = 0.0
        self._trade_count = 0
        self._breach_count = 0
        self._trading_halted = False
        self._halt_reason = ""
        self._history.clear()
        logger.info("SLIPPAGE_BUDGET_RESET")

    def resume_trading(self, authorized_by: str) -> None:
        """Resume trading after halt (requires authorization)."""
        if self._trading_halted:
            logger.warning(
                "TRADING_RESUMED",
                extra={
                    "authorized_by": authorized_by,
                    "previous_reason": self._halt_reason,
                },
            )
            self._trading_halted = False
            self._halt_reason = ""

    def get_status(self) -> dict[str, Any]:
        """Get current budget status."""
        return {
            "session_bps": self._session_slippage_bps,
            "session_dollars": self._session_slippage_dollars,
            "trade_count": self._trade_count,
            "breach_count": self._breach_count,
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "budget_used_pct": self._session_slippage_dollars / self.config.max_slippage_per_session_dollars,
            "config": {
                "max_per_trade_bps": self.config.max_slippage_per_trade_bps,
                "max_session_bps": self.config.max_slippage_per_session_bps,
                "max_session_dollars": self.config.max_slippage_per_session_dollars,
            },
        }


def create_slippage_budget(
    max_per_trade_bps: float = 25.0,
    max_session_dollars: float = 1000.0,
    halt_on_breach: bool = True,
) -> SlippageBudget:
    """
    Factory function to create a slippage budget tracker.

    Args:
        max_per_trade_bps: Max slippage per trade in basis points
        max_session_dollars: Max session slippage in dollars
        halt_on_breach: Whether to halt trading on breach

    Returns:
        Configured SlippageBudget instance
    """
    config = SlippageBudgetConfig(
        max_slippage_per_trade_bps=max_per_trade_bps,
        max_slippage_per_session_dollars=max_session_dollars,
        halt_on_breach=halt_on_breach,
    )
    return SlippageBudget(config)


__all__ = [
    "SmartOrderStatus",
    "SmartOrderType",
    "ExecutionOrder",
    "ExecutionResult",
    "SmartExecutionModel",
    "SmartExecutionExecutionModel",
    "create_smart_execution_model",
    # Slippage Budget (P1 FIX)
    "SlippageBudgetConfig",
    "SlippageBudget",
    "create_slippage_budget",
]
