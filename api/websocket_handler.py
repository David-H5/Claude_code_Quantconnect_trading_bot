"""
WebSocket Handler for Real-Time Updates

Manages WebSocket connections and broadcasts trading events:
- Order status updates
- Position changes
- P&L updates
- System alerts

UPGRADE-008: REST API Server (December 2025)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import WebSocket


logger = logging.getLogger(__name__)


class EventType(Enum):
    """WebSocket event types."""

    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_PARTIAL = "order_partial"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    PNL_UPDATE = "pnl_update"
    ALERT = "alert"
    CIRCUIT_BREAKER = "circuit_breaker"
    PING = "ping"
    PONG = "pong"


@dataclass
class WebSocketClient:
    """Represents a connected WebSocket client."""

    websocket: WebSocket
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    subscriptions: set[str] = field(default_factory=set)
    client_id: str | None = None

    def is_subscribed(self, channel: str) -> bool:
        """Check if client is subscribed to a channel."""
        return channel in self.subscriptions or "*" in self.subscriptions


class WebSocketManager:
    """Manages WebSocket connections for real-time updates.

    Features:
    - Connection management (connect, disconnect, reconnect)
    - Channel-based subscriptions
    - Broadcast to all clients
    - Send to specific clients
    - Thread-safe operations

    Example:
        >>> manager = WebSocketManager()
        >>> await manager.connect(websocket)
        >>> await manager.broadcast({"type": "order_filled", "order_id": "123"})
    """

    def __init__(self, ping_interval: float = 30.0):
        """Initialize WebSocket manager.

        Args:
            ping_interval: Interval for ping messages in seconds
        """
        self._clients: list[WebSocketClient] = []
        self._lock = asyncio.Lock()
        self._ping_interval = ping_interval

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self._clients)

    async def connect(self, websocket: WebSocket, client_id: str | None = None):
        """Accept a new WebSocket connection.

        Args:
            websocket: WebSocket connection to accept
            client_id: Optional client identifier
        """
        await websocket.accept()
        client = WebSocketClient(
            websocket=websocket,
            client_id=client_id,
            subscriptions={"*"},  # Subscribe to all by default
        )
        async with self._lock:
            self._clients.append(client)

        logger.info(f"WebSocket connected (id={client_id}). " f"Total connections: {len(self._clients)}")

        # Send welcome message
        await self.send_to_client(
            websocket,
            {
                "type": "connected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "client_id": client_id,
            },
        )

    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection to remove
        """
        async with self._lock:
            self._clients = [c for c in self._clients if c.websocket != websocket]

        logger.info(f"WebSocket disconnected. Total connections: {len(self._clients)}")

    async def disconnect_all(self):
        """Disconnect all WebSocket connections."""
        async with self._lock:
            for client in self._clients:
                try:
                    await client.websocket.close(code=1000, reason="Server shutdown")
                except Exception:
                    pass
            self._clients.clear()

        logger.info("All WebSocket connections closed")

    async def subscribe(self, websocket: WebSocket, channel: str):
        """Subscribe a client to a channel.

        Args:
            websocket: Client WebSocket
            channel: Channel to subscribe to
        """
        async with self._lock:
            for client in self._clients:
                if client.websocket == websocket:
                    client.subscriptions.add(channel)
                    logger.debug(f"Client subscribed to: {channel}")
                    break

    async def unsubscribe(self, websocket: WebSocket, channel: str):
        """Unsubscribe a client from a channel.

        Args:
            websocket: Client WebSocket
            channel: Channel to unsubscribe from
        """
        async with self._lock:
            for client in self._clients:
                if client.websocket == websocket:
                    client.subscriptions.discard(channel)
                    logger.debug(f"Client unsubscribed from: {channel}")
                    break

    async def _cleanup_disconnected(self, disconnected: list[WebSocketClient]) -> int:
        """
        Remove disconnected clients atomically.

        SAFETY FIX: Prevents TOCTOU race condition.
        Previous implementation checked 'if client in self._clients' then removed,
        which could fail if another coroutine modified the list between check and remove.

        Args:
            disconnected: List of clients to remove

        Returns:
            Number of clients actually removed
        """
        if not disconnected:
            return 0

        removed_count = 0

        async with self._lock:
            # Build new list excluding disconnected (atomic operation)
            disconnected_ids = {id(c) for c in disconnected}
            original_count = len(self._clients)

            self._clients = [c for c in self._clients if id(c) not in disconnected_ids]

            removed_count = original_count - len(self._clients)

            if removed_count > 0:
                logger.info(
                    "WEBSOCKET_CLIENTS_REMOVED",
                    extra={"removed_count": removed_count, "remaining_count": len(self._clients)},
                )

        # Close connections outside the lock to avoid holding lock during I/O
        for client in disconnected:
            try:
                await client.websocket.close()
            except Exception as e:
                logger.debug(f"Error closing websocket: {e}")

        return removed_count

    async def broadcast(
        self,
        message: dict[str, Any],
        channel: str | None = None,
    ):
        """Broadcast message to all connected clients.

        Args:
            message: Message to broadcast
            channel: Optional channel to broadcast to (None = all)
        """
        if not self._clients:
            return

        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now(timezone.utc).isoformat()

        disconnected = []

        async with self._lock:
            for client in self._clients:
                # Check subscription if channel specified
                if channel and not client.is_subscribed(channel):
                    continue

                try:
                    await client.websocket.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to broadcast to client: {e}")
                    disconnected.append(client)

        # Clean up disconnected clients (SAFETY FIX: atomic removal)
        if disconnected:
            await self._cleanup_disconnected(disconnected)

    async def send_to_client(
        self,
        websocket: WebSocket,
        message: dict[str, Any],
    ):
        """Send message to a specific client.

        Args:
            websocket: Target client WebSocket
            message: Message to send
        """
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now(timezone.utc).isoformat()

        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send to client: {e}")
            await self.disconnect(websocket)

    async def broadcast_order_event(
        self,
        event_type: EventType,
        order_id: str,
        details: dict[str, Any] | None = None,
    ):
        """Broadcast an order-related event.

        Args:
            event_type: Type of order event
            order_id: Order ID
            details: Additional event details
        """
        message = {
            "type": event_type.value,
            "order_id": order_id,
            **(details or {}),
        }
        await self.broadcast(message, channel="orders")

    async def broadcast_position_event(
        self,
        event_type: EventType,
        symbol: str,
        details: dict[str, Any] | None = None,
    ):
        """Broadcast a position-related event.

        Args:
            event_type: Type of position event
            symbol: Position symbol
            details: Additional event details
        """
        message = {
            "type": event_type.value,
            "symbol": symbol,
            **(details or {}),
        }
        await self.broadcast(message, channel="positions")

    async def broadcast_pnl_update(
        self,
        daily_pnl: float,
        total_pnl: float,
        unrealized_pnl: float,
    ):
        """Broadcast P&L update.

        Args:
            daily_pnl: Daily P&L
            total_pnl: Total P&L
            unrealized_pnl: Unrealized P&L
        """
        message = {
            "type": EventType.PNL_UPDATE.value,
            "daily_pnl": daily_pnl,
            "total_pnl": total_pnl,
            "unrealized_pnl": unrealized_pnl,
        }
        await self.broadcast(message, channel="pnl")

    async def broadcast_alert(
        self,
        level: str,
        message_text: str,
        source: str | None = None,
    ):
        """Broadcast an alert message.

        Args:
            level: Alert level (info, warning, error, critical)
            message_text: Alert message
            source: Alert source
        """
        message = {
            "type": EventType.ALERT.value,
            "level": level,
            "message": message_text,
            "source": source,
        }
        await self.broadcast(message, channel="alerts")

    async def broadcast_circuit_breaker(
        self,
        is_halted: bool,
        reason: str | None = None,
    ):
        """Broadcast circuit breaker status.

        Args:
            is_halted: Whether trading is halted
            reason: Reason for halt
        """
        message = {
            "type": EventType.CIRCUIT_BREAKER.value,
            "is_halted": is_halted,
            "reason": reason,
        }
        await self.broadcast(message, channel="alerts")

    def get_client_info(self) -> list[dict[str, Any]]:
        """Get information about connected clients.

        Returns:
            List of client information dictionaries
        """
        return [
            {
                "client_id": client.client_id,
                "connected_at": client.connected_at.isoformat(),
                "subscriptions": list(client.subscriptions),
            }
            for client in self._clients
        ]
