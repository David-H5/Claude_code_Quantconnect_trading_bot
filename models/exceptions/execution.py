"""Order execution exceptions."""

from .base import TradingError


class ExecutionError(TradingError):
    """Base class for execution errors."""

    pass


class OrderSubmissionError(ExecutionError):
    """Failed to submit order to broker."""

    def __init__(
        self,
        order_id: str,
        reason: str,
        broker_code: str | None = None,
    ):
        super().__init__(
            f"Order {order_id} submission failed: {reason}",
            recoverable=True,
        )
        self.order_id = order_id
        self.reason = reason
        self.broker_code = broker_code
        self.with_context(order_id=order_id, broker_code=broker_code)


class OrderFillError(ExecutionError):
    """Order fill failed or was partial."""

    def __init__(
        self,
        order_id: str,
        requested_qty: int,
        filled_qty: int,
        reason: str,
    ):
        super().__init__(
            f"Order {order_id} fill failed: requested {requested_qty}, " f"filled {filled_qty}. Reason: {reason}",
            recoverable=True,
        )
        self.order_id = order_id
        self.requested_qty = requested_qty
        self.filled_qty = filled_qty
        self.reason = reason
        self.with_context(order_id=order_id)


class SlippageExceededError(ExecutionError):
    """Execution slippage exceeded threshold."""

    def __init__(
        self,
        order_id: str,
        expected_price: float,
        actual_price: float,
        threshold_pct: float,
    ):
        slippage_pct = abs(actual_price - expected_price) / expected_price * 100
        super().__init__(
            f"Order {order_id} slippage {slippage_pct:.2f}% exceeds " f"threshold {threshold_pct:.2f}%",
            recoverable=False,
        )
        self.order_id = order_id
        self.expected_price = expected_price
        self.actual_price = actual_price
        self.slippage_pct = slippage_pct
        self.threshold_pct = threshold_pct
        self.with_context(order_id=order_id)


class OrderTimeoutError(ExecutionError):
    """Order execution timed out."""

    def __init__(self, order_id: str, timeout_seconds: float):
        super().__init__(
            f"Order {order_id} timed out after {timeout_seconds}s",
            recoverable=True,
        )
        self.order_id = order_id
        self.timeout_seconds = timeout_seconds
        self.with_context(order_id=order_id)


class MarketClosedError(ExecutionError):
    """Market is closed for trading."""

    def __init__(self, market: str, next_open: str | None = None):
        msg = f"Market {market} is closed"
        if next_open:
            msg += f", opens at {next_open}"
        super().__init__(msg, recoverable=True)
        self.market = market
        self.next_open = next_open


class OrderRejectedError(ExecutionError):
    """Order was rejected by the broker."""

    def __init__(
        self,
        order_id: str,
        reason: str,
        broker_code: str | None = None,
    ):
        super().__init__(
            f"Order {order_id} rejected: {reason}",
            recoverable=False,
        )
        self.order_id = order_id
        self.reason = reason
        self.broker_code = broker_code
        self.with_context(order_id=order_id, broker_code=broker_code)


class InsufficientFundsError(ExecutionError):
    """Insufficient funds for the trade."""

    def __init__(self, required: float, available: float):
        super().__init__(
            f"Insufficient funds: required ${required:,.2f}, " f"available ${available:,.2f}",
            recoverable=False,
        )
        self.required = required
        self.available = available


class InvalidPositionSizeError(ExecutionError):
    """Position size is invalid."""

    def __init__(self, symbol: str, requested: int, max_allowed: int):
        super().__init__(
            f"Invalid position size for {symbol}: {requested} exceeds max {max_allowed}",
            recoverable=False,
        )
        self.symbol = symbol
        self.requested = requested
        self.max_allowed = max_allowed
        self.with_context(symbol=symbol)


__all__ = [
    "ExecutionError",
    "InsufficientFundsError",
    "InvalidPositionSizeError",
    "MarketClosedError",
    "OrderFillError",
    "OrderRejectedError",
    "OrderSubmissionError",
    "OrderTimeoutError",
    "SlippageExceededError",
]
