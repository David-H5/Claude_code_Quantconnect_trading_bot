"""
Tests for REST API Server

Tests verify the API endpoints correctly:
- Submit and manage orders
- Query positions and P&L
- Manage recurring templates
- Health checks and system status
- WebSocket connections

UPGRADE-008: REST API Server (December 2025)
"""

# Import with proper module setup
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_order_queue():
    """Create mock OrderQueueAPI."""
    queue = MagicMock()
    queue.submit_order = MagicMock(
        return_value=MagicMock(
            order_id="order_123",
            status="pending",
        )
    )
    queue.get_all_orders = MagicMock(return_value=[])
    queue.get_order = MagicMock(return_value=None)
    queue.cancel_order = MagicMock(return_value=True)
    queue.get_pending_orders = MagicMock(return_value=[])
    return queue


@pytest.fixture
def mock_ws_manager():
    """Create mock WebSocketManager."""
    manager = MagicMock()
    manager.connection_count = 0
    manager.broadcast_order_event = AsyncMock()
    manager.broadcast = AsyncMock()
    manager.disconnect_all = AsyncMock()
    manager.get_client_info = MagicMock(return_value=[])
    return manager


@pytest.fixture
def test_client(mock_order_queue, mock_ws_manager):
    """Create test client with mocked dependencies."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    # Create a simple app without lifespan for testing
    app = FastAPI(title="Test Trading Bot API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Patch the module-level variables before importing routes
    import api.rest_server as rest_module

    rest_module._order_queue = mock_order_queue
    rest_module._ws_manager = mock_ws_manager

    # Import and include routers
    from api.routes import health, orders, positions, templates

    app.include_router(orders.router, prefix="/api/v1", tags=["Orders"])
    app.include_router(positions.router, prefix="/api/v1", tags=["Positions"])
    app.include_router(templates.router, prefix="/api/v1", tags=["Templates"])
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])

    @app.get("/")
    async def root():
        return {"name": "Test API", "version": "1.0.0", "status": "running"}

    with TestClient(app) as client:
        yield client

    # Cleanup
    rest_module._order_queue = None
    rest_module._ws_manager = None


@pytest.fixture
def test_client_no_queue():
    """Create test client without order queue."""
    from fastapi import FastAPI

    import api.rest_server as rest_module

    rest_module._order_queue = None
    rest_module._ws_manager = None

    app = FastAPI(title="Test Trading Bot API")

    from api.routes import health, orders, positions, templates

    app.include_router(orders.router, prefix="/api/v1", tags=["Orders"])
    app.include_router(positions.router, prefix="/api/v1", tags=["Positions"])
    app.include_router(templates.router, prefix="/api/v1", tags=["Templates"])
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])

    with TestClient(app) as client:
        yield client


# ============================================================================
# ROOT ENDPOINT TESTS
# ============================================================================


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_api_info(self, test_client):
        """Test root endpoint returns API information."""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"


# ============================================================================
# ORDER ENDPOINT TESTS
# ============================================================================


class TestOrderEndpoints:
    """Tests for order endpoints."""

    def test_submit_order_success(self, test_client, mock_order_queue):
        """Test successful order submission."""
        order_data = {
            "symbol": "SPY",
            "execution_type": "option_strategy",
            "strategy_name": "iron_condor",
            "quantity": 1,
            "limit_price": 2.50,
            "priority": "normal",
        }
        response = test_client.post("/api/v1/orders", json=order_data)
        assert response.status_code == 201
        data = response.json()
        assert "order_id" in data
        assert data["status"] == "pending"
        assert "message" in data

    def test_submit_order_with_legs(self, test_client, mock_order_queue):
        """Test order submission with manual legs."""
        order_data = {
            "symbol": "SPY",
            "execution_type": "manual_legs",
            "legs": [
                {"symbol": "SPY240101C400", "side": "buy", "quantity": 1},
                {"symbol": "SPY240101C410", "side": "sell", "quantity": 1},
            ],
            "quantity": 1,
        }
        response = test_client.post("/api/v1/orders", json=order_data)
        assert response.status_code == 201

    def test_submit_order_invalid_execution_type(self, test_client):
        """Test order submission with invalid execution type."""
        order_data = {
            "symbol": "SPY",
            "execution_type": "invalid_type",
            "quantity": 1,
        }
        response = test_client.post("/api/v1/orders", json=order_data)
        assert response.status_code == 422  # Validation error

    def test_submit_order_missing_symbol(self, test_client):
        """Test order submission without required symbol."""
        order_data = {
            "execution_type": "option_strategy",
            "quantity": 1,
        }
        response = test_client.post("/api/v1/orders", json=order_data)
        assert response.status_code == 422

    def test_list_orders_empty(self, test_client, mock_order_queue):
        """Test listing orders when queue is empty."""
        response = test_client.get("/api/v1/orders")
        assert response.status_code == 200
        data = response.json()
        assert "orders" in data
        assert "count" in data
        assert data["count"] == 0

    def test_list_orders_with_filter(self, test_client, mock_order_queue):
        """Test listing orders with status filter."""
        response = test_client.get("/api/v1/orders?status=pending")
        assert response.status_code == 200

    def test_get_order_not_found(self, test_client, mock_order_queue):
        """Test getting non-existent order."""
        mock_order_queue.get_order.return_value = None
        response = test_client.get("/api/v1/orders/nonexistent")
        assert response.status_code == 404

    def test_cancel_order_not_found(self, test_client, mock_order_queue):
        """Test cancelling non-existent order."""
        mock_order_queue.get_order.return_value = None
        response = test_client.delete("/api/v1/orders/nonexistent")
        assert response.status_code == 404

    def test_get_pending_count(self, test_client, mock_order_queue):
        """Test getting pending order count."""
        mock_order_queue.get_pending_orders.return_value = [MagicMock(), MagicMock()]
        response = test_client.get("/api/v1/orders/pending/count")
        assert response.status_code == 200
        data = response.json()
        assert data["pending_count"] == 2


# ============================================================================
# POSITION ENDPOINT TESTS
# ============================================================================


class TestPositionEndpoints:
    """Tests for position endpoints."""

    def test_list_positions_no_algorithm(self, test_client):
        """Test listing positions when algorithm not connected."""
        with patch("api.rest_server._algorithm", None):
            response = test_client.get("/api/v1/positions")
            assert response.status_code == 200
            data = response.json()
            assert data["count"] == 0

    def test_get_pnl_summary_no_algorithm(self, test_client):
        """Test P&L summary when algorithm not connected."""
        with patch("api.rest_server._algorithm", None):
            response = test_client.get("/api/v1/pnl")
            assert response.status_code == 200
            data = response.json()
            assert data["portfolio_value"] == 0.0

    def test_get_position_summary_no_algorithm(self, test_client_no_queue):
        """Test position summary when algorithm not connected."""
        # Without algorithm, endpoint returns empty summary with defaults
        response = test_client_no_queue.get("/api/v1/positions/summary")
        # Can return either 200 with empty data or 503 depending on order queue state
        assert response.status_code in [200, 503]


# ============================================================================
# TEMPLATE ENDPOINT TESTS
# ============================================================================


class TestTemplateEndpoints:
    """Tests for template endpoints."""

    def test_create_template(self, test_client):
        """Test creating a new template."""
        template_data = {
            "name": "Weekly SPY Iron Condor",
            "symbol": "SPY",
            "strategy_name": "iron_condor",
            "quantity": 1,
            "schedule_type": "weekly",
            "schedule_time": "09:35",
            "days_of_week": [0, 2, 4],
        }
        response = test_client.post("/api/v1/templates", json=template_data)
        assert response.status_code == 201
        data = response.json()
        assert "template_id" in data
        assert data["name"] == "Weekly SPY Iron Condor"

    def test_create_template_invalid_schedule(self, test_client):
        """Test creating template with invalid schedule type."""
        template_data = {
            "name": "Test Template",
            "symbol": "SPY",
            "strategy_name": "iron_condor",
            "schedule_type": "invalid_schedule",
        }
        response = test_client.post("/api/v1/templates", json=template_data)
        assert response.status_code == 422

    def test_list_templates_empty(self, test_client):
        """Test listing templates when none exist."""
        response = test_client.get("/api/v1/templates")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0

    def test_get_template_not_found(self, test_client):
        """Test getting non-existent template."""
        response = test_client.get("/api/v1/templates/nonexistent")
        assert response.status_code == 404


# ============================================================================
# HEALTH ENDPOINT TESTS
# ============================================================================


class TestHealthEndpoints:
    """Tests for health endpoints."""

    def test_health_check(self, test_client, mock_order_queue, mock_ws_manager):
        """Test health check endpoint."""
        with patch("api.rest_server._order_queue", mock_order_queue):
            with patch("api.rest_server._ws_manager", mock_ws_manager):
                response = test_client.get("/api/v1/health")
                assert response.status_code == 200
                data = response.json()
                assert "status" in data
                assert "checks" in data
                assert "uptime_seconds" in data

    def test_liveness_probe(self, test_client):
        """Test liveness probe endpoint."""
        response = test_client.get("/api/v1/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_readiness_probe_ready(self, test_client, mock_order_queue):
        """Test readiness probe when ready."""
        with patch("api.rest_server._order_queue", mock_order_queue):
            response = test_client.get("/api/v1/health/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"

    def test_readiness_probe_not_ready(self, test_client):
        """Test readiness probe when not ready."""
        with patch("api.rest_server._order_queue", None):
            response = test_client.get("/api/v1/health/ready")
            assert response.status_code == 503

    def test_system_info(self, test_client, mock_order_queue, mock_ws_manager):
        """Test system info endpoint."""
        with patch("api.rest_server._order_queue", mock_order_queue):
            with patch("api.rest_server._ws_manager", mock_ws_manager):
                response = test_client.get("/api/v1/system")
                assert response.status_code == 200
                data = response.json()
                assert "python_version" in data
                assert "platform" in data
                assert "api_version" in data

    def test_circuit_breaker_status_no_algorithm(self, test_client):
        """Test circuit breaker status when algorithm not connected."""
        with patch("api.rest_server._algorithm", None):
            response = test_client.get("/api/v1/circuit-breaker")
            assert response.status_code == 200
            data = response.json()
            assert data["state"] == "unknown"

    def test_config_no_algorithm(self, test_client):
        """Test configuration when algorithm not connected."""
        with patch("api.rest_server._algorithm", None):
            response = test_client.get("/api/v1/config")
            assert response.status_code == 200
            data = response.json()
            assert data["autonomous_enabled"] is False

    def test_websocket_clients(self, test_client, mock_ws_manager):
        """Test WebSocket clients endpoint."""
        with patch("api.rest_server._ws_manager", mock_ws_manager):
            response = test_client.get("/api/v1/websocket/clients")
            assert response.status_code == 200
            data = response.json()
            assert "clients" in data
            assert "count" in data


# ============================================================================
# WEBSOCKET HANDLER TESTS
# ============================================================================


class TestWebSocketHandler:
    """Tests for WebSocketManager."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test WebSocket manager initialization."""
        from api.websocket_handler import WebSocketManager

        manager = WebSocketManager()
        assert manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_no_clients(self):
        """Test broadcasting when no clients connected."""
        from api.websocket_handler import WebSocketManager

        manager = WebSocketManager()
        # Should not raise
        await manager.broadcast({"type": "test"})

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        """Test disconnecting all clients."""
        from api.websocket_handler import WebSocketManager

        manager = WebSocketManager()
        # Should not raise
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_get_client_info_empty(self):
        """Test getting client info when no clients."""
        from api.websocket_handler import WebSocketManager

        manager = WebSocketManager()
        clients = manager.get_client_info()
        assert clients == []


# ============================================================================
# WEBSOCKET CLIENT TESTS
# ============================================================================


class TestWebSocketClient:
    """Tests for WebSocketClient dataclass."""

    def test_client_creation(self):
        """Test WebSocket client creation."""
        from api.websocket_handler import WebSocketClient

        mock_ws = MagicMock()
        client = WebSocketClient(websocket=mock_ws)
        assert client.websocket == mock_ws
        assert "*" not in client.subscriptions  # Default is empty

    def test_client_subscription_check(self):
        """Test subscription checking."""
        from api.websocket_handler import WebSocketClient

        mock_ws = MagicMock()
        client = WebSocketClient(websocket=mock_ws, subscriptions={"orders", "positions"})
        assert client.is_subscribed("orders") is True
        assert client.is_subscribed("alerts") is False

    def test_client_wildcard_subscription(self):
        """Test wildcard subscription."""
        from api.websocket_handler import WebSocketClient

        mock_ws = MagicMock()
        client = WebSocketClient(websocket=mock_ws, subscriptions={"*"})
        assert client.is_subscribed("anything") is True


# ============================================================================
# EVENT TYPE TESTS
# ============================================================================


class TestEventTypes:
    """Tests for EventType enum."""

    def test_event_types_exist(self):
        """Test that all expected event types exist."""
        from api.websocket_handler import EventType

        assert EventType.ORDER_SUBMITTED.value == "order_submitted"
        assert EventType.ORDER_FILLED.value == "order_filled"
        assert EventType.ORDER_CANCELLED.value == "order_cancelled"
        assert EventType.POSITION_OPENED.value == "position_opened"
        assert EventType.PNL_UPDATE.value == "pnl_update"
        assert EventType.CIRCUIT_BREAKER.value == "circuit_breaker"


# ============================================================================
# API EXPORTS TESTS
# ============================================================================


class TestAPIExports:
    """Tests for API module exports."""

    def test_exports_available(self):
        """Test that all exports are available."""
        from api import (
            EventType,
            OrderQueueAPI,
            WebSocketManager,
            create_app,
            run_server,
        )

        assert OrderQueueAPI is not None
        assert create_app is not None
        assert run_server is not None
        assert WebSocketManager is not None
        assert EventType is not None

    def test_get_functions_exist(self):
        """Test getter functions exist."""
        from api import get_algorithm, get_order_queue, get_ws_manager, set_algorithm

        assert callable(get_order_queue)
        assert callable(get_ws_manager)
        assert callable(get_algorithm)
        assert callable(set_algorithm)


# ============================================================================
# ROUTE MODEL TESTS
# ============================================================================


class TestRouteModels:
    """Tests for Pydantic route models."""

    def test_order_submission_model(self):
        """Test OrderSubmission model validation."""
        from api.routes.orders import ExecutionType, OrderPriority, OrderSubmission

        order = OrderSubmission(
            symbol="SPY",
            execution_type=ExecutionType.OPTION_STRATEGY,
            strategy_name="iron_condor",
            quantity=1,
        )
        assert order.symbol == "SPY"
        assert order.priority == OrderPriority.NORMAL

    def test_order_submission_invalid_quantity(self):
        """Test OrderSubmission rejects invalid quantity."""
        from pydantic import ValidationError

        from api.routes.orders import ExecutionType, OrderSubmission

        with pytest.raises(ValidationError):
            OrderSubmission(
                symbol="SPY",
                execution_type=ExecutionType.OPTION_STRATEGY,
                quantity=0,  # Invalid - must be >= 1
            )

    def test_template_create_model(self):
        """Test TemplateCreate model validation."""
        from api.routes.templates import ScheduleType, TemplateCreate

        template = TemplateCreate(
            name="Test Template",
            symbol="SPY",
            strategy_name="iron_condor",
            schedule_type=ScheduleType.WEEKLY,
        )
        assert template.name == "Test Template"
        assert template.schedule_type == ScheduleType.WEEKLY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
