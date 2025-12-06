"""
Tests for Market Data MCP Server

UPGRADE-015 Phase 1: MCP Server Foundation

Tests cover:
- Server initialization and lifecycle
- Tool registration and listing
- Quote retrieval
- Option chain retrieval
- Historical data retrieval
- IV surface data
- Error handling
"""

import asyncio

# Import the server and related classes
import sys
from datetime import date

import pytest


sys.path.insert(0, "/home/dshooter/projects/Claude_code_Quantconnect_trading_bot")

from mcp.base_server import (
    ServerConfig,
    ServerState,
    ToolCategory,
)
from mcp.market_data_server import (
    MarketDataServer,
    create_market_data_server,
)
from mcp.schemas import (
    HistoricalRequest,
    OptionChainRequest,
    QuoteRequest,
    Resolution,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def server_config():
    """Create test server configuration."""
    return ServerConfig(
        name="test-market-data",
        version="1.0.0",
        description="Test market data server",
        mock_mode=True,
        timeout_seconds=5.0,
    )


@pytest.fixture
def market_server(server_config):
    """Create market data server instance."""
    return MarketDataServer(config=server_config, mock_mode=True)


@pytest.fixture
async def running_server(market_server):
    """Create and start a server."""
    await market_server.start()
    yield market_server
    await market_server.stop()


# =============================================================================
# Server Lifecycle Tests
# =============================================================================


class TestServerLifecycle:
    """Test server lifecycle management."""

    def test_server_creation(self, server_config):
        """Test server can be created."""
        server = MarketDataServer(config=server_config)
        assert server is not None
        assert server.name == "test-market-data"
        assert server.version == "1.0.0"
        assert server.state == ServerState.STOPPED

    def test_create_market_data_server_helper(self):
        """Test create_market_data_server helper function."""
        server = create_market_data_server(mock_mode=True)
        assert server is not None
        assert server.name == "market-data"
        assert server.mock_mode is True

    @pytest.mark.asyncio
    async def test_server_start_stop(self, market_server):
        """Test server can start and stop."""
        assert market_server.state == ServerState.STOPPED

        await market_server.start()
        assert market_server.state == ServerState.RUNNING
        assert market_server.is_running is True

        await market_server.stop()
        assert market_server.state == ServerState.STOPPED
        assert market_server.is_running is False

    @pytest.mark.asyncio
    async def test_server_health_check(self, running_server):
        """Test server health check."""
        health = running_server.health_check()

        assert health["server"] == "test-market-data"
        assert health["is_healthy"] is True
        assert health["state"] == "running"
        assert health["tools_registered"] > 0
        assert "uptime_seconds" in health


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestToolRegistration:
    """Test tool registration and listing."""

    def test_tools_registered(self, market_server):
        """Test that tools are registered on creation."""
        schemas = market_server.get_tool_schemas()
        assert len(schemas) > 0

        tool_names = [s.name for s in schemas]
        assert "get_quote" in tool_names
        assert "get_option_chain" in tool_names
        assert "get_greeks" in tool_names
        assert "get_historical" in tool_names
        assert "get_iv_surface" in tool_names
        assert "get_market_status" in tool_names

    def test_list_tools_mcp_format(self, market_server):
        """Test list_tools returns MCP-compatible format."""
        tools = market_server.list_tools()
        assert len(tools) > 0

        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert "type" in tool["inputSchema"]
            assert tool["inputSchema"]["type"] == "object"

    def test_get_tools_by_category(self, market_server):
        """Test filtering tools by category."""
        market_tools = market_server.get_tools_by_category(ToolCategory.MARKET_DATA)
        assert len(market_tools) >= 5  # At least our main tools

        for tool in market_tools:
            assert tool.category == ToolCategory.MARKET_DATA


# =============================================================================
# Quote Tool Tests
# =============================================================================


class TestGetQuoteTool:
    """Test get_quote tool."""

    @pytest.mark.asyncio
    async def test_get_quote_basic(self, running_server):
        """Test basic quote retrieval."""
        result = await running_server.call_tool("get_quote", {"symbol": "SPY"})

        assert result.success is True
        assert result.data is not None
        assert result.data["symbol"] == "SPY"
        assert "price" in result.data
        assert "timestamp" in result.data

    @pytest.mark.asyncio
    async def test_get_quote_with_volume(self, running_server):
        """Test quote with volume data."""
        result = await running_server.call_tool(
            "get_quote",
            {
                "symbol": "AAPL",
                "include_volume": True,
            },
        )

        assert result.success is True
        assert "volume" in result.data
        assert "current" in result.data["volume"]
        assert "average_30d" in result.data["volume"]

    @pytest.mark.asyncio
    async def test_get_quote_with_greeks(self, running_server):
        """Test quote with Greeks."""
        result = await running_server.call_tool(
            "get_quote",
            {
                "symbol": "MSFT",
                "include_greeks": True,
            },
        )

        assert result.success is True
        assert "greeks" in result.data
        greeks = result.data["greeks"]
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "theta" in greeks
        assert "vega" in greeks
        assert "implied_volatility" in greeks

    @pytest.mark.asyncio
    async def test_get_quote_symbol_normalization(self, running_server):
        """Test symbol normalization (lowercase to uppercase)."""
        result = await running_server.call_tool("get_quote", {"symbol": "aapl"})

        assert result.success is True
        assert result.data["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_quote_price_structure(self, running_server):
        """Test quote price data structure."""
        result = await running_server.call_tool("get_quote", {"symbol": "SPY"})

        price = result.data["price"]
        assert "last" in price
        assert "bid" in price
        assert "ask" in price
        assert "open" in price
        assert "high" in price
        assert "low" in price
        assert "close_previous" in price
        assert "change" in price
        assert "change_pct" in price


# =============================================================================
# Option Chain Tool Tests
# =============================================================================


class TestGetOptionChainTool:
    """Test get_option_chain tool."""

    @pytest.mark.asyncio
    async def test_get_option_chain_basic(self, running_server):
        """Test basic option chain retrieval."""
        result = await running_server.call_tool(
            "get_option_chain",
            {
                "underlying": "SPY",
            },
        )

        assert result.success is True
        assert result.data["underlying"] == "SPY"
        assert "contracts" in result.data
        assert "underlying_price" in result.data
        assert len(result.data["contracts"]) > 0

    @pytest.mark.asyncio
    async def test_get_option_chain_with_filters(self, running_server):
        """Test option chain with expiry and strike filters."""
        result = await running_server.call_tool(
            "get_option_chain",
            {
                "underlying": "AAPL",
                "expiry_min_days": 7,
                "expiry_max_days": 60,
                "strike_range_pct": 0.05,
            },
        )

        assert result.success is True
        assert len(result.data["contracts"]) > 0

    @pytest.mark.asyncio
    async def test_get_option_chain_calls_only(self, running_server):
        """Test option chain filtered to calls only."""
        result = await running_server.call_tool(
            "get_option_chain",
            {
                "underlying": "SPY",
                "option_type": "call",
            },
        )

        assert result.success is True
        for contract in result.data["contracts"]:
            assert contract["option_type"] == "call"

    @pytest.mark.asyncio
    async def test_get_option_chain_puts_only(self, running_server):
        """Test option chain filtered to puts only."""
        result = await running_server.call_tool(
            "get_option_chain",
            {
                "underlying": "SPY",
                "option_type": "put",
            },
        )

        assert result.success is True
        for contract in result.data["contracts"]:
            assert contract["option_type"] == "put"

    @pytest.mark.asyncio
    async def test_option_contract_structure(self, running_server):
        """Test option contract data structure."""
        result = await running_server.call_tool(
            "get_option_chain",
            {
                "underlying": "SPY",
            },
        )

        contract = result.data["contracts"][0]
        assert "symbol" in contract
        assert "underlying" in contract
        assert "strike" in contract
        assert "expiry" in contract
        assert "option_type" in contract
        assert "bid" in contract
        assert "ask" in contract
        assert "volume" in contract
        assert "open_interest" in contract
        assert "greeks" in contract


# =============================================================================
# Greeks Tool Tests
# =============================================================================


class TestGetGreeksTool:
    """Test get_greeks tool."""

    @pytest.mark.asyncio
    async def test_get_greeks(self, running_server):
        """Test Greeks retrieval for a contract."""
        result = await running_server.call_tool(
            "get_greeks",
            {
                "contract_symbol": "SPY241220C00450000",
            },
        )

        assert result.success is True
        assert "greeks" in result.data
        greeks = result.data["greeks"]
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "theta" in greeks
        assert "vega" in greeks
        assert "rho" in greeks
        assert "implied_volatility" in greeks

    @pytest.mark.asyncio
    async def test_greeks_value_ranges(self, running_server):
        """Test Greeks are within valid ranges."""
        result = await running_server.call_tool(
            "get_greeks",
            {
                "contract_symbol": "AAPL241220C00175000",
            },
        )

        greeks = result.data["greeks"]
        assert -1 <= greeks["delta"] <= 1
        assert greeks["gamma"] >= 0
        assert greeks["vega"] >= 0
        assert greeks["implied_volatility"] >= 0


# =============================================================================
# Historical Data Tool Tests
# =============================================================================


class TestGetHistoricalTool:
    """Test get_historical tool."""

    @pytest.mark.asyncio
    async def test_get_historical_basic(self, running_server):
        """Test basic historical data retrieval."""
        result = await running_server.call_tool(
            "get_historical",
            {
                "symbol": "SPY",
            },
        )

        assert result.success is True
        assert result.data["symbol"] == "SPY"
        assert "bars" in result.data
        assert len(result.data["bars"]) > 0

    @pytest.mark.asyncio
    async def test_get_historical_with_days(self, running_server):
        """Test historical data with custom days."""
        result = await running_server.call_tool(
            "get_historical",
            {
                "symbol": "AAPL",
                "days": 10,
            },
        )

        assert result.success is True
        # May be fewer due to weekends
        assert len(result.data["bars"]) <= 10

    @pytest.mark.asyncio
    async def test_get_historical_resolution(self, running_server):
        """Test historical data with resolution."""
        result = await running_server.call_tool(
            "get_historical",
            {
                "symbol": "MSFT",
                "days": 5,
                "resolution": "daily",
            },
        )

        assert result.success is True
        assert result.data["resolution"] == "daily"

    @pytest.mark.asyncio
    async def test_ohlcv_bar_structure(self, running_server):
        """Test OHLCV bar data structure."""
        result = await running_server.call_tool(
            "get_historical",
            {
                "symbol": "SPY",
                "days": 5,
            },
        )

        bar = result.data["bars"][0]
        assert "timestamp" in bar
        assert "open" in bar
        assert "high" in bar
        assert "low" in bar
        assert "close" in bar
        assert "volume" in bar

        # OHLC relationship check
        assert bar["high"] >= bar["open"]
        assert bar["high"] >= bar["close"]
        assert bar["low"] <= bar["open"]
        assert bar["low"] <= bar["close"]


# =============================================================================
# IV Surface Tool Tests
# =============================================================================


class TestGetIVSurfaceTool:
    """Test get_iv_surface tool."""

    @pytest.mark.asyncio
    async def test_get_iv_surface(self, running_server):
        """Test IV surface retrieval."""
        result = await running_server.call_tool(
            "get_iv_surface",
            {
                "underlying": "SPY",
            },
        )

        assert result.success is True
        assert result.data["underlying"] == "SPY"
        assert "surface" in result.data
        assert "atm_iv" in result.data
        assert len(result.data["surface"]) > 0

    @pytest.mark.asyncio
    async def test_iv_surface_point_structure(self, running_server):
        """Test IV surface point data structure."""
        result = await running_server.call_tool(
            "get_iv_surface",
            {
                "underlying": "AAPL",
            },
        )

        point = result.data["surface"][0]
        assert "strike" in point
        assert "dte" in point
        assert "moneyness" in point
        assert "iv" in point
        assert point["iv"] > 0


# =============================================================================
# Market Status Tool Tests
# =============================================================================


class TestGetMarketStatusTool:
    """Test get_market_status tool."""

    @pytest.mark.asyncio
    async def test_get_market_status(self, running_server):
        """Test market status retrieval."""
        result = await running_server.call_tool("get_market_status", {})

        assert result.success is True
        assert "status" in result.data
        assert result.data["status"] in ["open", "closed", "pre_market", "after_hours"]
        assert "timestamp" in result.data
        assert "is_trading_day" in result.data


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
    async def test_call_on_stopped_server(self, market_server):
        """Test calling tool on stopped server returns error."""
        # Server not started
        result = await market_server.call_tool("get_quote", {"symbol": "SPY"})

        assert result.success is False
        assert result.error_code == "SERVER_NOT_RUNNING"

    @pytest.mark.asyncio
    async def test_execution_time_tracking(self, running_server):
        """Test execution time is tracked."""
        result = await running_server.call_tool("get_quote", {"symbol": "SPY"})

        assert result.success is True
        assert result.execution_time_ms > 0


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestSchemaValidation:
    """Test Pydantic schema validation."""

    def test_quote_request_validation(self):
        """Test QuoteRequest validation."""
        request = QuoteRequest(symbol="aapl")
        assert request.symbol == "AAPL"  # Normalized

        request = QuoteRequest(symbol="SPY", include_greeks=True)
        assert request.include_greeks is True

    def test_option_chain_request_validation(self):
        """Test OptionChainRequest validation."""
        request = OptionChainRequest(
            underlying="spy",
            expiry_min_days=0,
            expiry_max_days=90,
        )
        assert request.underlying == "SPY"

    def test_historical_request_validation(self):
        """Test HistoricalRequest validation."""
        request = HistoricalRequest(
            symbol="AAPL",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 1),
            resolution=Resolution.DAILY,
        )
        assert request.symbol == "AAPL"

    def test_historical_request_date_validation(self):
        """Test HistoricalRequest date validation."""
        with pytest.raises(ValueError):
            HistoricalRequest(
                symbol="AAPL",
                start_date=date(2025, 12, 1),
                end_date=date(2025, 1, 1),  # Before start
            )


# =============================================================================
# Sync Wrapper Tests
# =============================================================================


class TestSyncWrapper:
    """Test synchronous wrapper method."""

    def test_call_tool_sync(self, market_server):
        """Test synchronous tool call wrapper."""
        # Start server synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(market_server.start())

        try:
            result = market_server.call_tool_sync("get_quote", {"symbol": "SPY"})
            assert result.success is True
            assert result.data["symbol"] == "SPY"
        finally:
            loop.run_until_complete(market_server.stop())
            loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
