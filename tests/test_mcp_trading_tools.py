"""
Tests for MCP Trading Tools Server

UPGRADE-014 Category 1: Architecture Enhancements
Tests for MCP server exposing trading-related tools.
"""

import pytest

from mcp.trading_tools_server import (
    ToolCategory,
    ToolResult,
    ToolSchema,
    TradingToolsServer,
    analyze_technicals,
    check_risk_limits,
    create_trading_tools_server,
    execute_order,
    get_market_data,
    get_news_sentiment,
    get_portfolio_status,
)


class TestToolFunctions:
    """Tests for individual tool functions."""

    def test_get_market_data_basic(self):
        """Test basic market data retrieval."""
        result = get_market_data("AAPL")

        assert result["symbol"] == "AAPL"
        assert "timestamp" in result
        assert "price" in result
        assert "volume" in result

    def test_get_market_data_with_greeks(self):
        """Test market data with Greeks."""
        result = get_market_data("AAPL", include_Greeks=True)

        assert "greeks" in result
        assert "delta" in result["greeks"]
        assert "gamma" in result["greeks"]
        assert "theta" in result["greeks"]
        assert "vega" in result["greeks"]

    def test_get_market_data_exclude_options(self):
        """Test market data excluding certain data."""
        result = get_market_data("SPY", include_price=False, include_volume=False)

        assert "price" not in result
        assert "volume" not in result

    def test_get_portfolio_status_full(self):
        """Test full portfolio status."""
        result = get_portfolio_status()

        assert "cash" in result
        assert "total_value" in result
        assert "positions" in result
        assert "pnl" in result
        assert "exposure" in result

    def test_get_portfolio_status_positions_only(self):
        """Test portfolio with positions only."""
        result = get_portfolio_status(
            include_positions=True,
            include_pnl=False,
            include_exposure=False,
        )

        assert "positions" in result
        assert "pnl" not in result
        assert "exposure" not in result

    def test_check_risk_limits_within(self):
        """Test risk check when within limits."""
        result = check_risk_limits(
            symbol="AAPL",
            position_size_pct=10.0,
            order_value=5000.0,
        )

        assert result["within_limits"] is True
        assert len(result["checks"]) >= 2

    def test_check_risk_limits_exceeded(self):
        """Test risk check when limits exceeded."""
        result = check_risk_limits(
            position_size_pct=50.0,  # Exceeds 25% limit
        )

        assert result["within_limits"] is False
        # Find the position_size check
        position_check = next(
            (c for c in result["checks"] if c["name"] == "position_size"),
            None,
        )
        assert position_check is not None
        assert position_check["passed"] is False

    def test_analyze_technicals_default(self):
        """Test technical analysis with default indicators."""
        result = analyze_technicals("AAPL")

        assert result["symbol"] == "AAPL"
        assert "indicators" in result
        assert "rsi" in result["indicators"]
        assert "macd" in result["indicators"]
        assert "overall_signal" in result
        assert "confidence" in result

    def test_analyze_technicals_specific_indicators(self):
        """Test technical analysis with specific indicators."""
        result = analyze_technicals("SPY", indicators=["rsi", "vwap"])

        assert "rsi" in result["indicators"]
        assert "vwap" in result["indicators"]
        # Should not have other indicators
        assert "bbands" not in result["indicators"]

    def test_analyze_technicals_timeframe(self):
        """Test technical analysis with different timeframe."""
        result = analyze_technicals("AAPL", timeframe="4H")

        assert result["timeframe"] == "4H"

    def test_get_news_sentiment(self):
        """Test news sentiment retrieval."""
        result = get_news_sentiment("AAPL")

        assert result["symbol"] == "AAPL"
        assert "articles" in result
        assert "aggregate_sentiment" in result
        assert "sentiment_label" in result
        assert "confidence" in result
        assert len(result["articles"]) > 0

    def test_get_news_sentiment_lookback(self):
        """Test news sentiment with custom lookback."""
        result = get_news_sentiment("SPY", lookback_hours=48)

        assert result["lookback_hours"] == 48

    def test_execute_order_dry_run(self):
        """Test order execution in dry run mode."""
        result = execute_order(
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            dry_run=True,
        )

        assert result["symbol"] == "AAPL"
        assert result["side"] == "buy"
        assert result["quantity"] == 100
        assert result["dry_run"] is True
        assert result["status"] == "simulated"

    def test_execute_order_limit(self):
        """Test limit order execution."""
        result = execute_order(
            symbol="SPY",
            side="sell",
            quantity=50,
            order_type="limit",
            limit_price=450.00,
            dry_run=True,
        )

        assert result["order_type"] == "limit"
        assert result["limit_price"] == 450.00
        assert result["fill_price"] == 450.00

    def test_execute_order_live_rejected(self):
        """Test that live orders are rejected."""
        result = execute_order(
            symbol="AAPL",
            side="buy",
            quantity=100,
            dry_run=False,
        )

        assert result["status"] == "rejected"
        assert "not implemented" in result["message"].lower()


class TestTradingToolsServer:
    """Tests for TradingToolsServer class."""

    @pytest.fixture
    def server(self):
        """Create a server instance."""
        return create_trading_tools_server()

    def test_create_server(self, server):
        """Test server creation."""
        assert server is not None
        assert isinstance(server, TradingToolsServer)

    def test_list_tools(self, server):
        """Test listing available tools."""
        tools = server.list_tools()

        assert len(tools) >= 6
        tool_names = [t.name for t in tools]
        assert "get_market_data" in tool_names
        assert "get_portfolio_status" in tool_names
        assert "check_risk_limits" in tool_names
        assert "analyze_technicals" in tool_names
        assert "get_news_sentiment" in tool_names
        assert "execute_order" in tool_names

    def test_get_tool_schema(self, server):
        """Test getting tool schema."""
        schema = server.get_tool_schema("get_market_data")

        assert schema is not None
        assert schema.name == "get_market_data"
        assert schema.category == ToolCategory.MARKET_DATA
        assert "properties" in schema.parameters
        assert "symbol" in schema.parameters["properties"]

    def test_get_unknown_tool_schema(self, server):
        """Test getting unknown tool schema."""
        schema = server.get_tool_schema("nonexistent_tool")
        assert schema is None

    def test_call_tool_success(self, server):
        """Test successful tool call."""
        result = server.call_tool("get_market_data", {"symbol": "AAPL"})

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.data is not None
        assert result.data["symbol"] == "AAPL"
        assert result.execution_time_ms > 0

    def test_call_tool_unknown(self, server):
        """Test calling unknown tool."""
        result = server.call_tool("nonexistent_tool", {})

        assert result.success is False
        assert "Unknown tool" in result.error

    def test_call_tool_error(self, server):
        """Test tool call with error."""
        # Missing required parameter
        result = server.call_tool("get_market_data", {})

        assert result.success is False
        assert result.error is not None

    def test_call_portfolio_tool(self, server):
        """Test portfolio status tool."""
        result = server.call_tool("get_portfolio_status", {})

        assert result.success is True
        assert "cash" in result.data
        assert "positions" in result.data

    def test_call_risk_tool(self, server):
        """Test risk limits tool."""
        result = server.call_tool(
            "check_risk_limits",
            {"symbol": "AAPL", "position_size_pct": 10.0},
        )

        assert result.success is True
        assert "within_limits" in result.data
        assert "checks" in result.data

    def test_call_technicals_tool(self, server):
        """Test technicals analysis tool."""
        result = server.call_tool(
            "analyze_technicals",
            {"symbol": "SPY", "indicators": ["rsi", "macd"]},
        )

        assert result.success is True
        assert "indicators" in result.data
        assert "overall_signal" in result.data

    def test_call_sentiment_tool(self, server):
        """Test news sentiment tool."""
        result = server.call_tool(
            "get_news_sentiment",
            {"symbol": "AAPL", "lookback_hours": 12},
        )

        assert result.success is True
        assert "articles" in result.data
        assert "aggregate_sentiment" in result.data

    def test_call_order_tool(self, server):
        """Test order execution tool."""
        result = server.call_tool(
            "execute_order",
            {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "dry_run": True,
            },
        )

        assert result.success is True
        assert result.data["status"] == "simulated"


class TestToolSchema:
    """Tests for ToolSchema dataclass."""

    def test_schema_creation(self):
        """Test creating a tool schema."""
        schema = ToolSchema(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.ANALYSIS,
            parameters={"type": "object", "properties": {}},
            returns="Test result",
        )

        assert schema.name == "test_tool"
        assert schema.category == ToolCategory.ANALYSIS


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = ToolResult(
            success=True,
            data={"key": "value"},
            execution_time_ms=50.0,
        )

        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_error_result(self):
        """Test error result."""
        result = ToolResult(
            success=False,
            error="Something went wrong",
            execution_time_ms=10.0,
        )

        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.data is None

    def test_to_dict(self):
        """Test serialization."""
        result = ToolResult(
            success=True,
            data={"symbol": "AAPL"},
            execution_time_ms=25.0,
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["data"]["symbol"] == "AAPL"
        assert d["execution_time_ms"] == 25.0


class TestMCPFormat:
    """Tests for MCP format export."""

    def test_to_mcp_format(self):
        """Test exporting to MCP format."""
        server = create_trading_tools_server()
        mcp_format = server.to_mcp_format()

        assert "tools" in mcp_format
        assert len(mcp_format["tools"]) >= 6

        # Check tool format
        first_tool = mcp_format["tools"][0]
        assert "name" in first_tool
        assert "description" in first_tool
        assert "inputSchema" in first_tool


class TestServerWithProviders:
    """Tests for server with custom providers."""

    def test_server_with_mock_provider(self):
        """Test server with mock data provider."""

        class MockDataProvider:
            def get_price(self, symbol):
                return 200.00

        server = create_trading_tools_server(
            data_provider=MockDataProvider(),
        )

        # Should still work with mock provider
        result = server.call_tool("get_market_data", {"symbol": "TEST"})
        assert result.success is True
