"""
Tests for Broker MCP Server

UPGRADE-015 Phase 2: Broker MCP Server

Tests cover:
- Server initialization and lifecycle
- Position management
- Order placement and validation
- Order cancellation
- Fill history
- Account information
- Error handling and safety checks
"""

import sys

import pytest


sys.path.insert(0, "/home/dshooter/projects/Claude_code_Quantconnect_trading_bot")

from mcp.base_server import (
    ServerConfig,
    ServerState,
    ToolCategory,
)
from mcp.broker_server import (
    BrokerServer,
    create_broker_server,
)
from mcp.schemas import TradingMode


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def server_config():
    """Create test server configuration."""
    return ServerConfig(
        name="test-broker",
        version="1.0.0",
        description="Test broker server",
        mock_mode=True,
        timeout_seconds=5.0,
    )


@pytest.fixture
def broker_server(server_config):
    """Create broker server instance."""
    return BrokerServer(config=server_config, mock_mode=True)


@pytest.fixture
async def running_server(broker_server):
    """Create and start a server."""
    await broker_server.start()
    yield broker_server
    await broker_server.stop()


# =============================================================================
# Server Lifecycle Tests
# =============================================================================


class TestServerLifecycle:
    """Test server lifecycle management."""

    def test_server_creation(self, server_config):
        """Test server can be created."""
        server = BrokerServer(config=server_config)
        assert server is not None
        assert server.name == "test-broker"
        assert server.version == "1.0.0"
        assert server.state == ServerState.STOPPED

    def test_create_broker_server_helper(self):
        """Test create_broker_server helper function."""
        server = create_broker_server(mock_mode=True)
        assert server is not None
        assert server.name == "broker"
        assert server.mock_mode is True

    def test_create_broker_server_trading_mode(self):
        """Test create_broker_server with trading mode."""
        server = create_broker_server(mock_mode=True, trading_mode=TradingMode.PAPER)
        assert server.trading_mode == TradingMode.PAPER

    @pytest.mark.asyncio
    async def test_server_start_stop(self, broker_server):
        """Test server can start and stop."""
        assert broker_server.state == ServerState.STOPPED

        await broker_server.start()
        assert broker_server.state == ServerState.RUNNING
        assert broker_server.is_running is True

        await broker_server.stop()
        assert broker_server.state == ServerState.STOPPED
        assert broker_server.is_running is False

    @pytest.mark.asyncio
    async def test_server_health_check(self, running_server):
        """Test server health check."""
        health = running_server.health_check()

        assert health["server"] == "test-broker"
        assert health["is_healthy"] is True
        assert health["state"] == "running"
        assert health["tools_registered"] > 0
        assert "uptime_seconds" in health


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestToolRegistration:
    """Test tool registration and listing."""

    def test_tools_registered(self, broker_server):
        """Test that tools are registered on creation."""
        schemas = broker_server.get_tool_schemas()
        assert len(schemas) >= 6

        tool_names = [s.name for s in schemas]
        assert "get_positions" in tool_names
        assert "get_orders" in tool_names
        assert "place_order" in tool_names
        assert "cancel_order" in tool_names
        assert "get_fills" in tool_names
        assert "get_account_info" in tool_names

    def test_list_tools_mcp_format(self, broker_server):
        """Test list_tools returns MCP-compatible format."""
        tools = broker_server.list_tools()
        assert len(tools) >= 6

        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert "type" in tool["inputSchema"]
            assert tool["inputSchema"]["type"] == "object"

    def test_dangerous_tools_marked(self, broker_server):
        """Test dangerous tools are properly marked."""
        schemas = broker_server.get_tool_schemas()

        place_order = next(s for s in schemas if s.name == "place_order")
        cancel_order = next(s for s in schemas if s.name == "cancel_order")

        assert place_order.is_dangerous is True
        assert cancel_order.is_dangerous is True

    def test_get_tools_by_category(self, broker_server):
        """Test filtering tools by category."""
        broker_tools = broker_server.get_tools_by_category(ToolCategory.BROKER)
        assert len(broker_tools) >= 2  # get_orders, get_fills

        execution_tools = broker_server.get_tools_by_category(ToolCategory.EXECUTION)
        assert len(execution_tools) >= 2  # place_order, cancel_order


# =============================================================================
# Position Tests
# =============================================================================


class TestGetPositions:
    """Test get_positions tool."""

    @pytest.mark.asyncio
    async def test_get_positions_basic(self, running_server):
        """Test basic position retrieval."""
        result = await running_server.call_tool("get_positions", {})

        assert result.success is True
        assert result.data is not None
        assert "positions" in result.data
        assert "total_positions" in result.data
        assert "total_market_value" in result.data
        assert "timestamp" in result.data

    @pytest.mark.asyncio
    async def test_get_positions_has_mock_data(self, running_server):
        """Test mock data is populated."""
        result = await running_server.call_tool("get_positions", {})

        assert len(result.data["positions"]) > 0

    @pytest.mark.asyncio
    async def test_get_positions_filter_by_symbol(self, running_server):
        """Test filtering positions by symbol."""
        result = await running_server.call_tool(
            "get_positions",
            {"symbol": "SPY"},
        )

        assert result.success is True
        positions = result.data["positions"]
        for pos in positions:
            assert pos["symbol"] == "SPY"

    @pytest.mark.asyncio
    async def test_position_structure(self, running_server):
        """Test position data structure."""
        result = await running_server.call_tool("get_positions", {})

        if result.data["positions"]:
            pos = result.data["positions"][0]
            assert "symbol" in pos
            assert "quantity" in pos
            assert "average_cost" in pos
            assert "current_price" in pos
            assert "market_value" in pos
            assert "unrealized_pnl" in pos
            assert "unrealized_pnl_pct" in pos


# =============================================================================
# Order Tests
# =============================================================================


class TestGetOrders:
    """Test get_orders tool."""

    @pytest.mark.asyncio
    async def test_get_orders_basic(self, running_server):
        """Test basic order retrieval."""
        result = await running_server.call_tool("get_orders", {})

        assert result.success is True
        assert "orders" in result.data
        assert "total_orders" in result.data
        assert "timestamp" in result.data

    @pytest.mark.asyncio
    async def test_get_orders_filter_by_status(self, running_server):
        """Test filtering orders by status."""
        result = await running_server.call_tool(
            "get_orders",
            {"status": "open"},
        )

        assert result.success is True
        for order in result.data["orders"]:
            assert order["status"] == "open"

    @pytest.mark.asyncio
    async def test_get_orders_all_status(self, running_server):
        """Test getting all orders."""
        result = await running_server.call_tool(
            "get_orders",
            {"status": "all"},
        )

        assert result.success is True


class TestPlaceOrder:
    """Test place_order tool."""

    @pytest.mark.asyncio
    async def test_place_market_order(self, running_server):
        """Test placing a market order."""
        result = await running_server.call_tool(
            "place_order",
            {
                "symbol": "GOOGL",
                "quantity": 10,
                "side": "buy",
                "order_type": "market",
            },
        )

        assert result.success is True
        assert result.data["success"] is True
        assert "order" in result.data
        order = result.data["order"]
        assert order["symbol"] == "GOOGL"
        assert order["quantity"] == 10
        assert order["side"] == "buy"
        assert order["status"] == "filled"  # Market orders fill immediately in mock

    @pytest.mark.asyncio
    async def test_place_limit_order(self, running_server):
        """Test placing a limit order."""
        result = await running_server.call_tool(
            "place_order",
            {
                "symbol": "AMZN",
                "quantity": 5,
                "side": "buy",
                "order_type": "limit",
                "limit_price": 180.00,
            },
        )

        assert result.success is True
        assert result.data["success"] is True
        order = result.data["order"]
        assert order["symbol"] == "AMZN"
        assert order["order_type"] == "limit"
        assert order["limit_price"] == 180.00
        assert order["status"] == "open"

    @pytest.mark.asyncio
    async def test_place_order_missing_limit_price(self, running_server):
        """Test limit order without limit price fails."""
        result = await running_server.call_tool(
            "place_order",
            {
                "symbol": "TSLA",
                "quantity": 10,
                "side": "buy",
                "order_type": "limit",
            },
        )

        assert result.success is True  # Tool executed
        assert result.data["success"] is False  # But order rejected
        assert result.data["error_code"] == "MISSING_LIMIT_PRICE"

    @pytest.mark.asyncio
    async def test_place_order_missing_stop_price(self, running_server):
        """Test stop order without stop price fails."""
        result = await running_server.call_tool(
            "place_order",
            {
                "symbol": "TSLA",
                "quantity": 10,
                "side": "buy",
                "order_type": "stop",
            },
        )

        assert result.success is True
        assert result.data["success"] is False
        assert result.data["error_code"] == "MISSING_STOP_PRICE"

    @pytest.mark.asyncio
    async def test_place_order_invalid_quantity(self, running_server):
        """Test order with invalid quantity fails."""
        result = await running_server.call_tool(
            "place_order",
            {
                "symbol": "NVDA",
                "quantity": 0,
                "side": "buy",
                "order_type": "market",
            },
        )

        assert result.success is True
        assert result.data["success"] is False
        assert result.data["error_code"] == "INVALID_QUANTITY"

    @pytest.mark.asyncio
    async def test_place_order_live_mode_blocked(self):
        """Test live trading is blocked."""
        server = create_broker_server(mock_mode=True, trading_mode=TradingMode.LIVE)
        await server.start()

        try:
            result = await server.call_tool(
                "place_order",
                {
                    "symbol": "SPY",
                    "quantity": 100,
                    "side": "buy",
                    "order_type": "market",
                },
            )

            assert result.success is True
            assert result.data["success"] is False
            assert result.data["error_code"] == "LIVE_TRADING_BLOCKED"
        finally:
            await server.stop()


class TestCancelOrder:
    """Test cancel_order tool."""

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, running_server):
        """Test cancelling an open order."""
        # First place an order
        place_result = await running_server.call_tool(
            "place_order",
            {
                "symbol": "META",
                "quantity": 20,
                "side": "buy",
                "order_type": "limit",
                "limit_price": 500.00,
            },
        )
        order_id = place_result.data["order"]["order_id"]

        # Cancel it
        result = await running_server.call_tool(
            "cancel_order",
            {"order_id": order_id},
        )

        assert result.success is True
        assert result.data["success"] is True
        assert result.data["order_id"] == order_id

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, running_server):
        """Test cancelling non-existent order."""
        result = await running_server.call_tool(
            "cancel_order",
            {"order_id": "FAKE-ORDER-123"},
        )

        assert result.success is True
        assert result.data["success"] is False
        assert result.data["error_code"] == "ORDER_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_cancel_filled_order_fails(self, running_server):
        """Test cannot cancel filled order."""
        # Place and fill a market order
        place_result = await running_server.call_tool(
            "place_order",
            {
                "symbol": "NFLX",
                "quantity": 5,
                "side": "buy",
                "order_type": "market",
            },
        )
        order_id = place_result.data["order"]["order_id"]

        # Try to cancel
        result = await running_server.call_tool(
            "cancel_order",
            {"order_id": order_id},
        )

        assert result.success is True
        assert result.data["success"] is False
        assert result.data["error_code"] == "INVALID_ORDER_STATUS"


# =============================================================================
# Fill Tests
# =============================================================================


class TestGetFills:
    """Test get_fills tool."""

    @pytest.mark.asyncio
    async def test_get_fills_basic(self, running_server):
        """Test basic fill retrieval."""
        result = await running_server.call_tool("get_fills", {})

        assert result.success is True
        assert "fills" in result.data
        assert "total_fills" in result.data
        assert "total_volume" in result.data
        assert "timestamp" in result.data

    @pytest.mark.asyncio
    async def test_get_fills_has_mock_data(self, running_server):
        """Test mock fills exist."""
        result = await running_server.call_tool("get_fills", {})

        assert len(result.data["fills"]) > 0

    @pytest.mark.asyncio
    async def test_get_fills_filter_by_symbol(self, running_server):
        """Test filtering fills by symbol."""
        result = await running_server.call_tool(
            "get_fills",
            {"symbol": "SPY"},
        )

        assert result.success is True
        for fill in result.data["fills"]:
            assert fill["symbol"] == "SPY"

    @pytest.mark.asyncio
    async def test_fill_structure(self, running_server):
        """Test fill data structure."""
        result = await running_server.call_tool("get_fills", {})

        if result.data["fills"]:
            fill = result.data["fills"][0]
            assert "fill_id" in fill
            assert "order_id" in fill
            assert "symbol" in fill
            assert "quantity" in fill
            assert "price" in fill
            assert "side" in fill
            assert "commission" in fill
            assert "fill_time" in fill


# =============================================================================
# Account Tests
# =============================================================================


class TestGetAccountInfo:
    """Test get_account_info tool."""

    @pytest.mark.asyncio
    async def test_get_account_info(self, running_server):
        """Test account info retrieval."""
        result = await running_server.call_tool("get_account_info", {})

        assert result.success is True
        assert "cash" in result.data
        assert "buying_power" in result.data
        assert "total_value" in result.data
        assert "trading_mode" in result.data
        assert "timestamp" in result.data

    @pytest.mark.asyncio
    async def test_account_values_positive(self, running_server):
        """Test account values are positive."""
        result = await running_server.call_tool("get_account_info", {})

        assert result.data["cash"] > 0
        assert result.data["buying_power"] > 0
        assert result.data["total_value"] > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestOrderFlow:
    """Test complete order flow."""

    @pytest.mark.asyncio
    async def test_buy_updates_position(self, running_server):
        """Test buying creates/updates position."""
        # Place market buy
        result = await running_server.call_tool(
            "place_order",
            {
                "symbol": "AMD",
                "quantity": 50,
                "side": "buy",
                "order_type": "market",
            },
        )
        assert result.data["success"] is True

        # Check positions updated
        after = await running_server.call_tool("get_positions", {"symbol": "AMD"})
        assert len(after.data["positions"]) > 0
        assert after.data["positions"][0]["quantity"] == 50

    @pytest.mark.asyncio
    async def test_buy_creates_fill(self, running_server):
        """Test buying creates fill record."""
        # Place market buy
        result = await running_server.call_tool(
            "place_order",
            {
                "symbol": "CRM",
                "quantity": 25,
                "side": "buy",
                "order_type": "market",
            },
        )
        order_id = result.data["order"]["order_id"]

        # Check fill created
        fills = await running_server.call_tool(
            "get_fills",
            {"order_id": order_id},
        )
        assert len(fills.data["fills"]) == 1
        assert fills.data["fills"][0]["symbol"] == "CRM"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_unknown_tool(self, running_server):
        """Test calling unknown tool returns error."""
        result = await running_server.call_tool("unknown_tool", {})

        assert result.success is False
        assert result.error_code == "TOOL_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_call_on_stopped_server(self, broker_server):
        """Test calling tool on stopped server returns error."""
        result = await broker_server.call_tool("get_positions", {})

        assert result.success is False
        assert result.error_code == "SERVER_NOT_RUNNING"


# =============================================================================
# Sync Wrapper Tests
# =============================================================================


class TestSyncWrapper:
    """Test synchronous wrapper method."""

    def test_call_tool_sync(self, broker_server):
        """Test synchronous tool call wrapper."""
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(broker_server.start())

        try:
            result = broker_server.call_tool_sync("get_account_info", {})
            assert result.success is True
            assert "cash" in result.data
        finally:
            loop.run_until_complete(broker_server.stop())
            loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
