"""Infrastructure exceptions (connections, timeouts, etc.)."""

from .base import TradingError


class InfrastructureError(TradingError):
    """Base class for infrastructure errors."""

    pass


class ConnectionFailedError(InfrastructureError):
    """Failed to connect to a service."""

    def __init__(
        self,
        service: str,
        host: str,
        port: int | None = None,
        reason: str | None = None,
    ):
        location = f"{host}:{port}" if port else host
        msg = f"Failed to connect to {service} at {location}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, recoverable=True)
        self.service = service
        self.host = host
        self.port = port
        self.reason = reason
        self.with_context(component=service)


class ServiceTimeoutError(InfrastructureError):
    """Service call timed out."""

    def __init__(self, service: str, operation: str, timeout_seconds: float):
        super().__init__(
            f"{service}.{operation}() timed out after {timeout_seconds}s",
            recoverable=True,
        )
        self.service = service
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.with_context(component=service, operation=operation)


class RedisError(InfrastructureError):
    """Redis operation failed."""

    def __init__(
        self,
        operation: str,
        key: str | None = None,
        reason: str = "",
    ):
        msg = f"Redis {operation} failed"
        if key:
            msg += f" for key '{key}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, recoverable=True)
        self.operation = operation
        self.key = key
        self.reason = reason
        self.with_context(component="redis", operation=operation)


class DataFeedError(InfrastructureError):
    """Data feed error."""

    def __init__(
        self,
        feed: str,
        symbol: str | None = None,
        reason: str = "",
    ):
        msg = f"Data feed '{feed}' error"
        if symbol:
            msg += f" for {symbol}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, recoverable=True)
        self.feed = feed
        self.symbol = symbol
        self.reason = reason
        self.with_context(component=feed, symbol=symbol)


class WebSocketError(InfrastructureError):
    """WebSocket connection error."""

    def __init__(
        self,
        endpoint: str,
        reason: str,
        reconnect_possible: bool = True,
    ):
        super().__init__(
            f"WebSocket error for {endpoint}: {reason}",
            recoverable=reconnect_possible,
        )
        self.endpoint = endpoint
        self.reason = reason
        self.reconnect_possible = reconnect_possible


class MessageQueueError(InfrastructureError):
    """Message queue (pub/sub) error."""

    def __init__(self, queue: str, operation: str, reason: str):
        super().__init__(
            f"Message queue '{queue}' {operation} failed: {reason}",
            recoverable=True,
        )
        self.queue = queue
        self.operation = operation
        self.reason = reason
        self.with_context(component="pubsub", operation=operation)


class TimeSeriesError(InfrastructureError):
    """Time series database error."""

    def __init__(self, operation: str, metric: str, reason: str):
        super().__init__(
            f"TimeSeries {operation} for '{metric}' failed: {reason}",
            recoverable=True,
        )
        self.operation = operation
        self.metric = metric
        self.reason = reason
        self.with_context(component="timeseries", operation=operation)


class CacheError(InfrastructureError):
    """Cache operation error."""

    def __init__(self, cache_name: str, operation: str, reason: str):
        super().__init__(
            f"Cache '{cache_name}' {operation} failed: {reason}",
            recoverable=True,
        )
        self.cache_name = cache_name
        self.operation = operation
        self.reason = reason


__all__ = [
    "CacheError",
    "ConnectionFailedError",
    "DataFeedError",
    "InfrastructureError",
    "MessageQueueError",
    "RedisError",
    "ServiceTimeoutError",
    "TimeSeriesError",
    "WebSocketError",
]
