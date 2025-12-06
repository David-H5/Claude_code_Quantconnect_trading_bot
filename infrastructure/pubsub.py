"""
Pub/Sub Manager for Trading Infrastructure

UPGRADE-015 Phase 7: Redis Infrastructure

Provides publish/subscribe messaging using Redis pub/sub.
Supports trading events, alerts, and system notifications.

Features:
- Pattern-based subscriptions
- Message handlers
- Automatic reconnection
- Trading-specific channels
"""

import contextlib
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


# Optional import - gracefully degrade if not installed
try:
    import redis
    from redis import Redis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore
    Redis = None  # type: ignore
    REDIS_AVAILABLE = False


class MessageType(Enum):
    """Types of pub/sub messages."""

    TRADE_SIGNAL = "trade_signal"
    ORDER_UPDATE = "order_update"
    POSITION_UPDATE = "position_update"
    PRICE_ALERT = "price_alert"
    RISK_ALERT = "risk_alert"
    SYSTEM_EVENT = "system_event"
    HEARTBEAT = "heartbeat"
    CUSTOM = "custom"


@dataclass
class PubSubConfig:
    """Configuration for pub/sub manager."""

    channel_prefix: str = "trading"
    reconnect_delay: float = 1.0
    max_reconnect_attempts: int = 5
    heartbeat_interval: int = 30


@dataclass
class Message:
    """A pub/sub message."""

    type: MessageType
    channel: str
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "system"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for transmission."""
        return {
            "type": self.type.value,
            "channel": self.channel,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            type=MessageType(data.get("type", "custom")),
            channel=data.get("channel", ""),
            data=data.get("data", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            source=data.get("source", "unknown"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# Type alias for message handlers
MessageHandler = Callable[[Message], None]


class PubSubManager:
    """Pub/Sub manager using Redis."""

    # Standard trading channels
    CHANNEL_SIGNALS = "signals"
    CHANNEL_ORDERS = "orders"
    CHANNEL_POSITIONS = "positions"
    CHANNEL_ALERTS = "alerts"
    CHANNEL_SYSTEM = "system"

    def __init__(
        self,
        redis_client: Any,
        config: PubSubConfig | None = None,
    ):
        """
        Initialize pub/sub manager.

        Args:
            redis_client: Redis client instance
            config: Pub/sub configuration
        """
        self.config = config or PubSubConfig()
        self._client = redis_client
        self._pubsub: Any = None
        self._handlers: dict[str, list[MessageHandler]] = {}
        self._pattern_handlers: dict[str, list[MessageHandler]] = {}
        self._running = False
        self._thread: threading.Thread | None = None

        if self._client:
            self._pubsub = self._client.pubsub()

    def _make_channel(self, channel: str) -> str:
        """Create full channel name with prefix."""
        if channel.startswith(self.config.channel_prefix):
            return channel
        return f"{self.config.channel_prefix}:{channel}"

    # ==========================================================================
    # Publishing
    # ==========================================================================

    def publish(
        self,
        channel: str,
        message: Message | dict[str, Any],
    ) -> int:
        """
        Publish a message to a channel.

        Args:
            channel: Channel name
            message: Message to publish

        Returns:
            Number of subscribers that received the message
        """
        if not self._client:
            return 0

        full_channel = self._make_channel(channel)

        if isinstance(message, Message):
            data = json.dumps(message.to_dict(), default=str)
        else:
            data = json.dumps(message, default=str)

        try:
            return self._client.publish(full_channel, data)
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to publish to channel '{channel}': {e}")
            return 0

    def publish_trade_signal(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Publish a trade signal.

        Args:
            symbol: Stock symbol
            signal: Signal type (BUY, SELL, HOLD)
            confidence: Confidence level (0-1)
            metadata: Additional metadata

        Returns:
            Number of subscribers
        """
        message = Message(
            type=MessageType.TRADE_SIGNAL,
            channel=self.CHANNEL_SIGNALS,
            data={
                "symbol": symbol.upper(),
                "signal": signal.upper(),
                "confidence": confidence,
                **(metadata or {}),
            },
        )
        return self.publish(self.CHANNEL_SIGNALS, message)

    def publish_order_update(
        self,
        order_id: str,
        status: str,
        symbol: str,
        details: dict[str, Any] | None = None,
    ) -> int:
        """
        Publish an order update.

        Args:
            order_id: Order identifier
            status: Order status
            symbol: Stock symbol
            details: Order details

        Returns:
            Number of subscribers
        """
        message = Message(
            type=MessageType.ORDER_UPDATE,
            channel=self.CHANNEL_ORDERS,
            data={
                "order_id": order_id,
                "status": status,
                "symbol": symbol.upper(),
                **(details or {}),
            },
        )
        return self.publish(self.CHANNEL_ORDERS, message)

    def publish_position_update(
        self,
        symbol: str,
        quantity: int,
        avg_price: float,
        current_price: float | None = None,
    ) -> int:
        """
        Publish a position update.

        Args:
            symbol: Stock symbol
            quantity: Position quantity
            avg_price: Average entry price
            current_price: Current market price

        Returns:
            Number of subscribers
        """
        data = {
            "symbol": symbol.upper(),
            "quantity": quantity,
            "avg_price": avg_price,
        }
        if current_price:
            data["current_price"] = current_price
            data["pnl"] = (current_price - avg_price) * quantity

        message = Message(
            type=MessageType.POSITION_UPDATE,
            channel=self.CHANNEL_POSITIONS,
            data=data,
        )
        return self.publish(self.CHANNEL_POSITIONS, message)

    def publish_alert(
        self,
        alert_type: str,
        message_text: str,
        severity: str = "info",
        data: dict[str, Any] | None = None,
    ) -> int:
        """
        Publish an alert.

        Args:
            alert_type: Type of alert (price, risk, system)
            message_text: Alert message
            severity: Severity level (info, warning, critical)
            data: Additional data

        Returns:
            Number of subscribers
        """
        msg_type = (
            MessageType.RISK_ALERT
            if alert_type == "risk"
            else (MessageType.PRICE_ALERT if alert_type == "price" else MessageType.SYSTEM_EVENT)
        )

        message = Message(
            type=msg_type,
            channel=self.CHANNEL_ALERTS,
            data={
                "alert_type": alert_type,
                "message": message_text,
                "severity": severity,
                **(data or {}),
            },
        )
        return self.publish(self.CHANNEL_ALERTS, message)

    def publish_heartbeat(self, source: str = "system") -> int:
        """
        Publish a heartbeat message.

        Args:
            source: Source identifier

        Returns:
            Number of subscribers
        """
        message = Message(
            type=MessageType.HEARTBEAT,
            channel=self.CHANNEL_SYSTEM,
            data={"status": "alive"},
            source=source,
        )
        return self.publish(self.CHANNEL_SYSTEM, message)

    # ==========================================================================
    # Subscribing
    # ==========================================================================

    def subscribe(
        self,
        channel: str,
        handler: MessageHandler,
    ) -> bool:
        """
        Subscribe to a channel.

        Args:
            channel: Channel name
            handler: Message handler function

        Returns:
            True if subscribed
        """
        if not self._pubsub:
            return False

        full_channel = self._make_channel(channel)

        if full_channel not in self._handlers:
            self._handlers[full_channel] = []
            try:
                self._pubsub.subscribe(full_channel)
            except (redis.RedisError, OSError) as e:
                logger.debug(f"Failed to subscribe to channel '{channel}': {e}")
                return False

        self._handlers[full_channel].append(handler)
        return True

    def subscribe_pattern(
        self,
        pattern: str,
        handler: MessageHandler,
    ) -> bool:
        """
        Subscribe to channels matching a pattern.

        Args:
            pattern: Glob pattern (e.g., "trading:*")
            handler: Message handler function

        Returns:
            True if subscribed
        """
        if not self._pubsub:
            return False

        full_pattern = self._make_channel(pattern)

        if full_pattern not in self._pattern_handlers:
            self._pattern_handlers[full_pattern] = []
            try:
                self._pubsub.psubscribe(full_pattern)
            except (redis.RedisError, OSError) as e:
                logger.debug(f"Failed to subscribe to pattern '{pattern}': {e}")
                return False

        self._pattern_handlers[full_pattern].append(handler)
        return True

    def unsubscribe(self, channel: str) -> bool:
        """
        Unsubscribe from a channel.

        Args:
            channel: Channel name

        Returns:
            True if unsubscribed
        """
        if not self._pubsub:
            return False

        full_channel = self._make_channel(channel)

        if full_channel in self._handlers:
            try:
                self._pubsub.unsubscribe(full_channel)
                del self._handlers[full_channel]
                return True
            except (redis.RedisError, OSError) as e:
                logger.debug(f"Failed to unsubscribe from channel '{channel}': {e}")
                return False

        return False

    def unsubscribe_pattern(self, pattern: str) -> bool:
        """
        Unsubscribe from a pattern.

        Args:
            pattern: Glob pattern

        Returns:
            True if unsubscribed
        """
        if not self._pubsub:
            return False

        full_pattern = self._make_channel(pattern)

        if full_pattern in self._pattern_handlers:
            try:
                self._pubsub.punsubscribe(full_pattern)
                del self._pattern_handlers[full_pattern]
                return True
            except (redis.RedisError, OSError) as e:
                logger.debug(f"Failed to unsubscribe from pattern '{pattern}': {e}")
                return False

        return False

    # ==========================================================================
    # Message Processing
    # ==========================================================================

    def _process_message(self, raw_message: dict[str, Any]) -> None:
        """Process a received message."""
        msg_type = raw_message.get("type")

        if msg_type == "message":
            channel = raw_message.get("channel", "")
            data = raw_message.get("data", "")

            # Decode bytes if necessary
            if isinstance(channel, bytes):
                channel = channel.decode("utf-8")
            if isinstance(data, bytes):
                data = data.decode("utf-8")

            try:
                message = Message.from_json(data)
            except (json.JSONDecodeError, KeyError):
                message = Message(
                    type=MessageType.CUSTOM,
                    channel=channel,
                    data={"raw": data},
                )

            # Call registered handlers
            handlers = self._handlers.get(channel, [])
            for handler in handlers:
                with contextlib.suppress(Exception):
                    handler(message)

        elif msg_type == "pmessage":
            pattern = raw_message.get("pattern", "")
            channel = raw_message.get("channel", "")
            data = raw_message.get("data", "")

            # Decode bytes if necessary
            if isinstance(pattern, bytes):
                pattern = pattern.decode("utf-8")
            if isinstance(channel, bytes):
                channel = channel.decode("utf-8")
            if isinstance(data, bytes):
                data = data.decode("utf-8")

            try:
                message = Message.from_json(data)
            except (json.JSONDecodeError, KeyError):
                message = Message(
                    type=MessageType.CUSTOM,
                    channel=channel,
                    data={"raw": data},
                )

            # Call pattern handlers
            handlers = self._pattern_handlers.get(pattern, [])
            for handler in handlers:
                with contextlib.suppress(Exception):
                    handler(message)

    def listen_once(self, timeout: float = 1.0) -> Message | None:
        """
        Listen for a single message.

        Args:
            timeout: Timeout in seconds

        Returns:
            Message if received, None otherwise
        """
        if not self._pubsub:
            return None

        try:
            raw_message = self._pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=timeout,
            )

            if raw_message:
                self._process_message(raw_message)

                # Return the parsed message
                data = raw_message.get("data", "")
                if isinstance(data, bytes):
                    data = data.decode("utf-8")

                try:
                    return Message.from_json(data)
                except (json.JSONDecodeError, KeyError):
                    return None

            return None
        except (redis.RedisError, OSError) as e:
            logger.debug(f"Failed to listen for message: {e}")
            return None

    def start_listening(self) -> None:
        """Start listening for messages in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop_listening(self) -> None:
        """Stop the listening thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _listen_loop(self) -> None:
        """Background listening loop."""
        while self._running:
            try:
                raw_message = self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )

                if raw_message:
                    self._process_message(raw_message)
            except (redis.RedisError, OSError) as e:
                # Attempt reconnection after error
                logger.debug(f"Error in listen loop, reconnecting: {e}")
                time.sleep(self.config.reconnect_delay)

    # ==========================================================================
    # Convenience Methods
    # ==========================================================================

    def on_trade_signal(self, handler: MessageHandler) -> bool:
        """Subscribe to trade signals."""
        return self.subscribe(self.CHANNEL_SIGNALS, handler)

    def on_order_update(self, handler: MessageHandler) -> bool:
        """Subscribe to order updates."""
        return self.subscribe(self.CHANNEL_ORDERS, handler)

    def on_position_update(self, handler: MessageHandler) -> bool:
        """Subscribe to position updates."""
        return self.subscribe(self.CHANNEL_POSITIONS, handler)

    def on_alert(self, handler: MessageHandler) -> bool:
        """Subscribe to alerts."""
        return self.subscribe(self.CHANNEL_ALERTS, handler)

    def on_system_event(self, handler: MessageHandler) -> bool:
        """Subscribe to system events."""
        return self.subscribe(self.CHANNEL_SYSTEM, handler)

    # ==========================================================================
    # Management
    # ==========================================================================

    def get_channels(self) -> list[str]:
        """Get list of subscribed channels."""
        return list(self._handlers.keys())

    def get_patterns(self) -> list[str]:
        """Get list of subscribed patterns."""
        return list(self._pattern_handlers.keys())

    def close(self) -> None:
        """Close the pub/sub connection."""
        self.stop_listening()

        if self._pubsub:
            try:
                self._pubsub.unsubscribe()
                self._pubsub.punsubscribe()
                self._pubsub.close()
            except Exception:
                pass

        self._handlers.clear()
        self._pattern_handlers.clear()


def create_pubsub_manager(
    redis_client: Any = None,
    channel_prefix: str = "trading",
) -> PubSubManager:
    """
    Factory function to create a pub/sub manager.

    Args:
        redis_client: Redis client (required)
        channel_prefix: Prefix for channel names

    Returns:
        Configured PubSubManager
    """
    config = PubSubConfig(channel_prefix=channel_prefix)
    return PubSubManager(redis_client, config)
