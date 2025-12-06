"""
FastAPI REST Server for Trading Bot UI Communication

Provides:
- Order submission and management
- Position and P&L queries
- WebSocket for real-time updates
- Health checks and configuration

UPGRADE-008: REST API Server (December 2025)

SECURITY NOTES:
- CORS is restricted by default (configure CORS_ALLOWED_ORIGINS env var)
- Global state uses thread-safe references
- Authentication required for trading endpoints
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Generic, TypeVar

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False
    logger_init = logging.getLogger(__name__)
    logger_init.warning("slowapi not installed - rate limiting disabled")

if TYPE_CHECKING:
    from .order_queue_api import OrderQueueAPI
    from .websocket_handler import WebSocketManager

logger = logging.getLogger(__name__)


# =============================================================================
# Thread-Safe Reference Wrapper (P0 FIX: Race Condition Prevention)
# =============================================================================

T = TypeVar("T")


class ThreadSafeRef(Generic[T]):
    """
    Thread-safe reference wrapper with initialization guarantee.

    SAFETY FIX: Prevents race conditions in global state access.
    Previous implementation used bare Optional types without synchronization.

    Usage:
        ref = ThreadSafeRef("MyComponent")
        ref.set(my_instance)  # Thread-safe initialization
        instance = ref.get()  # Thread-safe access with timeout
    """

    def __init__(self, name: str):
        self._value: T | None = None
        self._lock = threading.RLock()
        self._name = name
        self._initialized = threading.Event()

    def set(self, value: T) -> None:
        """Set the reference value (can only be called once)."""
        with self._lock:
            if self._value is not None:
                raise RuntimeError(f"{self._name} already initialized")
            self._value = value
            self._initialized.set()
            logger.info(f"{self._name}_INITIALIZED")

    def get(self, timeout: float = 5.0) -> T:
        """Get the reference value, waiting for initialization if needed."""
        if not self._initialized.wait(timeout):
            raise RuntimeError(f"{self._name} not initialized within {timeout}s")
        with self._lock:
            if self._value is None:
                raise RuntimeError(f"{self._name} was cleared")
            return self._value

    def get_or_none(self) -> T | None:
        """Get the reference value without waiting (returns None if not set)."""
        with self._lock:
            return self._value

    def clear(self) -> None:
        """Clear the reference (for shutdown)."""
        with self._lock:
            self._value = None
            self._initialized.clear()

    @contextmanager
    def access(self):
        """Context manager for extended access with lock held."""
        with self._lock:
            yield self._value


# Thread-safe global state (P0 FIX: Replaces bare Optional types)
_order_queue: ThreadSafeRef[OrderQueueAPI] = ThreadSafeRef("OrderQueue")
_ws_manager: ThreadSafeRef[WebSocketManager] = ThreadSafeRef("WebSocketManager")
_algorithm: ThreadSafeRef = ThreadSafeRef("Algorithm")

# Rate limiter instance (P1 FIX: API rate limiting)
_limiter: Limiter | None = None
if RATE_LIMITING_AVAILABLE:
    _limiter = Limiter(key_func=get_remote_address)


def get_limiter() -> Limiter | None:
    """Get the rate limiter instance."""
    return _limiter


def get_order_queue() -> OrderQueueAPI | None:
    """Get the global order queue instance (thread-safe)."""
    return _order_queue.get_or_none()


def get_ws_manager() -> WebSocketManager | None:
    """Get the global WebSocket manager instance (thread-safe)."""
    return _ws_manager.get_or_none()


def get_algorithm():
    """Get the running algorithm instance (thread-safe)."""
    return _algorithm.get_or_none()


def set_algorithm(algo):
    """Set the running algorithm reference (thread-safe, one-time only)."""
    _algorithm.set(algo)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Initializes and cleans up global resources using thread-safe references.
    """
    from .order_queue_api import OrderQueueAPI
    from .websocket_handler import WebSocketManager

    logger.info("Starting Trading Bot REST API Server...")

    # Initialize order queue (thread-safe)
    _order_queue.set(OrderQueueAPI())
    logger.info("Order queue initialized")

    # Initialize WebSocket manager (thread-safe)
    _ws_manager.set(WebSocketManager())
    logger.info("WebSocket manager initialized")

    yield

    # Cleanup
    logger.info("Shutting down REST API Server...")
    ws_manager = _ws_manager.get_or_none()
    if ws_manager:
        await ws_manager.disconnect_all()

    # Clear references
    _order_queue.clear()
    _ws_manager.clear()
    _algorithm.clear()
    logger.info("REST API Server shutdown complete")


def _get_cors_origins() -> list:
    """
    Get CORS allowed origins from environment.

    SECURITY FIX: Prevents wildcard CORS in production.

    Environment variable: CORS_ALLOWED_ORIGINS (comma-separated)
    Example: CORS_ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com
    """
    origins_str = os.environ.get("CORS_ALLOWED_ORIGINS", "")
    origins = [origin.strip() for origin in origins_str.split(",") if origin.strip()]

    if not origins:
        # Check if running in production
        environment = os.environ.get("ENVIRONMENT", "development")
        if environment == "production":
            logger.error(
                "CORS_SECURITY_ERROR: CORS_ALLOWED_ORIGINS must be set in production. "
                "Example: CORS_ALLOWED_ORIGINS=https://yourdomain.com"
            )
            raise OSError(
                "CORS_ALLOWED_ORIGINS environment variable is required in production. "
                "Set it to your allowed domains (comma-separated)."
            )
        # Development defaults
        origins = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080"]
        logger.warning("CORS_DEV_MODE", extra={"origins": origins, "warning": "Using development CORS settings"})

    return origins


def create_app(
    title: str = "Trading Bot API",
    version: str = "1.0.0",
    debug: bool = False,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        title: API title for documentation
        version: API version
        debug: Enable debug mode

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        description="REST API for Semi-Autonomous Options Trading Bot",
        version=version,
        lifespan=lifespan,
        debug=debug,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware - SECURITY FIX: No more wildcard origins
    cors_origins = _get_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )
    logger.info("CORS_CONFIGURED", extra={"origins": cors_origins})

    # Rate limiting middleware (P1 FIX: API rate limiting)
    if RATE_LIMITING_AVAILABLE and _limiter is not None:
        app.state.limiter = _limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        logger.info("RATE_LIMITING_ENABLED", extra={"status": "active"})
    else:
        logger.warning("RATE_LIMITING_DISABLED", extra={"reason": "slowapi not installed"})

    # Import and include routers
    from .routes import health, orders, positions, templates

    app.include_router(orders.router, prefix="/api/v1", tags=["Orders"])
    app.include_router(positions.router, prefix="/api/v1", tags=["Positions"])
    app.include_router(templates.router, prefix="/api/v1", tags=["Templates"])
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        manager = get_ws_manager()
        if manager is None:
            await websocket.close(code=1011, reason="Server not ready")
            return

        await manager.connect(websocket)
        try:
            while True:
                # Keep connection alive and handle client messages
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=30.0,  # Ping every 30 seconds
                    )
                    # Handle client messages (subscriptions, etc.)
                    await _handle_ws_message(websocket, data)
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await websocket.send_json({"type": "ping"})
        except WebSocketDisconnect:
            await manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await manager.disconnect(websocket)

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API info."""
        return {
            "name": title,
            "version": version,
            "status": "running",
            "docs": "/docs",
            "websocket": "/ws",
        }

    return app


async def _handle_ws_message(websocket: WebSocket, data: dict):
    """Handle incoming WebSocket messages from clients.

    Args:
        websocket: Client WebSocket connection
        data: Message data from client
    """
    msg_type = data.get("type", "")

    if msg_type == "ping":
        await websocket.send_json({"type": "pong"})
    elif msg_type == "subscribe":
        # Handle subscription requests
        channel = data.get("channel", "")
        logger.debug(f"Client subscribed to: {channel}")
        await websocket.send_json(
            {
                "type": "subscribed",
                "channel": channel,
            }
        )
    elif msg_type == "unsubscribe":
        channel = data.get("channel", "")
        logger.debug(f"Client unsubscribed from: {channel}")
        await websocket.send_json(
            {
                "type": "unsubscribed",
                "channel": channel,
            }
        )
    else:
        logger.warning(f"Unknown WebSocket message type: {msg_type}")


# Create default app instance
app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info",
):
    """Run the API server with uvicorn.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
        log_level: Logging level
    """
    import uvicorn

    uvicorn.run(
        "api.rest_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    run_server(reload=True)
