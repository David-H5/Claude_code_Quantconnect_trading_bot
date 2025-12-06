"""
MCP Trading Tools Server

Exposes trading-related tools via the Model Context Protocol for use by
Claude and other LLM agents. Provides market data, portfolio status,
risk checks, technical analysis, and sentiment tools.

UPGRADE-014 Category 1: Architecture Enhancements

Usage with Claude:
    Configure in .mcp.json or use directly as a tool server.

Usage in code:
    from mcp.trading_tools_server import create_trading_tools_server

    server = create_trading_tools_server()
    result = server.call_tool("get_market_data", {"symbol": "AAPL"})

QuantConnect Compatible: Yes
- Defensive error handling
- Configurable data sources
- Mock mode for testing
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of trading tools."""

    MARKET_DATA = "market_data"
    PORTFOLIO = "portfolio"
    RISK = "risk"
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    SENTIMENT = "sentiment"


@dataclass
class ToolSchema:
    """
    Schema for an MCP tool.

    Attributes:
        name: Tool name
        description: Human-readable description
        category: Tool category
        parameters: JSON Schema for parameters
        returns: Description of return value
    """

    name: str
    description: str
    category: ToolCategory
    parameters: dict[str, Any]
    returns: str


@dataclass
class ToolResult:
    """
    Result from a tool call.

    Attributes:
        success: Whether the call succeeded
        data: Result data if successful
        error: Error message if failed
        execution_time_ms: Execution time in milliseconds
    """

    success: bool
    data: Any | None = None
    error: str | None = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
        }


# Tool Implementations
# These are standalone functions that can be used directly or via the server


def get_market_data(
    symbol: str,
    include_price: bool = True,
    include_volume: bool = True,
    include_Greeks: bool = False,
    data_provider: Any | None = None,
) -> dict[str, Any]:
    """
    Get current market data for a symbol.

    Args:
        symbol: Ticker symbol (e.g., "AAPL", "SPY")
        include_price: Include price data
        include_volume: Include volume data
        include_Greeks: Include options Greeks (if applicable)
        data_provider: Optional data provider for live data

    Returns:
        Dictionary with market data
    """
    # Mock implementation - in production, would use QuantConnect data
    timestamp = datetime.utcnow().isoformat()

    result = {
        "symbol": symbol.upper(),
        "timestamp": timestamp,
    }

    if include_price:
        result["price"] = {
            "last": 150.00,  # Mock price
            "bid": 149.95,
            "ask": 150.05,
            "open": 148.50,
            "high": 151.00,
            "low": 148.00,
            "close_previous": 149.00,
            "change_pct": 0.67,
        }

    if include_volume:
        result["volume"] = {
            "current": 5_000_000,
            "average_30d": 8_000_000,
            "relative": 0.625,
        }

    if include_Greeks:
        result["greeks"] = {
            "delta": 0.55,
            "gamma": 0.03,
            "theta": -0.05,
            "vega": 0.15,
            "implied_volatility": 0.25,
        }

    return result


def get_portfolio_status(
    include_positions: bool = True,
    include_pnl: bool = True,
    include_exposure: bool = True,
    portfolio_provider: Any | None = None,
) -> dict[str, Any]:
    """
    Get current portfolio status.

    Args:
        include_positions: Include position details
        include_pnl: Include P&L data
        include_exposure: Include exposure metrics
        portfolio_provider: Optional portfolio provider for live data

    Returns:
        Dictionary with portfolio status
    """
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "cash": 50_000.00,
        "total_value": 100_000.00,
        "buying_power": 150_000.00,
    }

    if include_positions:
        result["positions"] = [
            {
                "symbol": "AAPL",
                "quantity": 100,
                "avg_cost": 145.00,
                "current_price": 150.00,
                "unrealized_pnl": 500.00,
                "unrealized_pnl_pct": 3.45,
            },
            {
                "symbol": "SPY",
                "quantity": 50,
                "avg_cost": 420.00,
                "current_price": 425.00,
                "unrealized_pnl": 250.00,
                "unrealized_pnl_pct": 1.19,
            },
        ]

    if include_pnl:
        result["pnl"] = {
            "daily": 750.00,
            "daily_pct": 0.75,
            "weekly": 2_500.00,
            "weekly_pct": 2.56,
            "monthly": 5_000.00,
            "monthly_pct": 5.26,
            "ytd": 15_000.00,
            "ytd_pct": 17.65,
        }

    if include_exposure:
        result["exposure"] = {
            "long": 65_000.00,
            "short": 0.00,
            "net": 65_000.00,
            "gross": 65_000.00,
            "cash_pct": 50.0,
            "equity_pct": 50.0,
        }

    return result


def check_risk_limits(
    symbol: str | None = None,
    order_value: float | None = None,
    position_size_pct: float | None = None,
    risk_manager: Any | None = None,
) -> dict[str, Any]:
    """
    Check if proposed action is within risk limits.

    Args:
        symbol: Symbol for position-specific checks
        order_value: Value of proposed order
        position_size_pct: Proposed position size as % of portfolio
        risk_manager: Optional risk manager for limit checking

    Returns:
        Dictionary with risk check results
    """
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "within_limits": True,
        "checks": [],
    }

    # Check position size limit (25% max)
    if position_size_pct is not None:
        max_position = 25.0
        passed = position_size_pct <= max_position
        result["checks"].append(
            {
                "name": "position_size",
                "passed": passed,
                "limit": max_position,
                "actual": position_size_pct,
                "message": f"Position size {'OK' if passed else 'exceeds limit'}",
            }
        )
        if not passed:
            result["within_limits"] = False

    # Check order value limit
    if order_value is not None:
        max_order = 25_000.00
        passed = order_value <= max_order
        result["checks"].append(
            {
                "name": "order_value",
                "passed": passed,
                "limit": max_order,
                "actual": order_value,
                "message": f"Order value {'OK' if passed else 'exceeds limit'}",
            }
        )
        if not passed:
            result["within_limits"] = False

    # Check daily loss limit (3%)
    daily_loss_pct = 1.5  # Mock current daily loss
    max_daily_loss = 3.0
    passed = daily_loss_pct <= max_daily_loss
    result["checks"].append(
        {
            "name": "daily_loss",
            "passed": passed,
            "limit": max_daily_loss,
            "actual": daily_loss_pct,
            "message": f"Daily loss {'OK' if passed else 'exceeds limit'}",
        }
    )
    if not passed:
        result["within_limits"] = False

    # Check concentration limit
    if symbol:
        concentration_pct = 15.0  # Mock concentration
        max_concentration = 30.0
        passed = concentration_pct <= max_concentration
        result["checks"].append(
            {
                "name": "concentration",
                "passed": passed,
                "limit": max_concentration,
                "actual": concentration_pct,
                "symbol": symbol,
                "message": f"Concentration in {symbol} {'OK' if passed else 'exceeds limit'}",
            }
        )
        if not passed:
            result["within_limits"] = False

    return result


def analyze_technicals(
    symbol: str,
    indicators: list[str] | None = None,
    timeframe: str = "1D",
    technical_analyzer: Any | None = None,
) -> dict[str, Any]:
    """
    Analyze technical indicators for a symbol.

    Args:
        symbol: Ticker symbol
        indicators: List of indicators to analyze (default: all)
        timeframe: Timeframe for analysis ("1D", "4H", "1H", etc.)
        technical_analyzer: Optional analyzer for live data

    Returns:
        Dictionary with technical analysis
    """
    if indicators is None:
        indicators = ["rsi", "macd", "sma", "ema", "bbands", "vwap"]

    result = {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "timestamp": datetime.utcnow().isoformat(),
        "indicators": {},
        "signals": [],
        "overall_signal": "neutral",
    }

    bullish_count = 0
    bearish_count = 0

    for indicator in indicators:
        if indicator.lower() == "rsi":
            value = 55.0  # Mock RSI
            result["indicators"]["rsi"] = {
                "value": value,
                "signal": "neutral" if 30 < value < 70 else ("oversold" if value <= 30 else "overbought"),
                "period": 14,
            }
            if value <= 30:
                bullish_count += 1
            elif value >= 70:
                bearish_count += 1

        elif indicator.lower() == "macd":
            result["indicators"]["macd"] = {
                "macd_line": 1.5,
                "signal_line": 1.2,
                "histogram": 0.3,
                "signal": "bullish",
            }
            bullish_count += 1

        elif indicator.lower() in ["sma", "ema"]:
            result["indicators"][indicator.lower()] = {
                "sma_20": 148.00,
                "sma_50": 145.00,
                "sma_200": 140.00,
                "price_vs_sma20": "above",
                "price_vs_sma50": "above",
                "price_vs_sma200": "above",
                "signal": "bullish",
            }
            bullish_count += 1

        elif indicator.lower() == "bbands":
            result["indicators"]["bbands"] = {
                "upper": 155.00,
                "middle": 150.00,
                "lower": 145.00,
                "width": 0.067,
                "position": "middle",
                "signal": "neutral",
            }

        elif indicator.lower() == "vwap":
            result["indicators"]["vwap"] = {
                "value": 149.50,
                "price_vs_vwap": "above",
                "signal": "bullish",
            }
            bullish_count += 1

    # Determine overall signal
    if bullish_count > bearish_count + 1:
        result["overall_signal"] = "bullish"
        result["signals"].append("Multiple indicators showing bullish bias")
    elif bearish_count > bullish_count + 1:
        result["overall_signal"] = "bearish"
        result["signals"].append("Multiple indicators showing bearish bias")
    else:
        result["overall_signal"] = "neutral"
        result["signals"].append("Mixed signals - no clear direction")

    result["confidence"] = min(0.9, 0.5 + abs(bullish_count - bearish_count) * 0.1)

    return result


def get_news_sentiment(
    symbol: str,
    lookback_hours: int = 24,
    min_relevance: float = 0.5,
    sentiment_analyzer: Any | None = None,
) -> dict[str, Any]:
    """
    Get news sentiment for a symbol.

    Args:
        symbol: Ticker symbol
        lookback_hours: Hours to look back for news
        min_relevance: Minimum relevance score to include
        sentiment_analyzer: Optional analyzer for live sentiment

    Returns:
        Dictionary with news sentiment analysis
    """
    result = {
        "symbol": symbol.upper(),
        "timestamp": datetime.utcnow().isoformat(),
        "lookback_hours": lookback_hours,
        "news_count": 5,
        "articles": [
            {
                "headline": f"{symbol} reports strong quarterly earnings",
                "source": "Reuters",
                "sentiment": 0.75,
                "relevance": 0.95,
                "published": "2 hours ago",
            },
            {
                "headline": f"Analysts upgrade {symbol} to buy",
                "source": "Bloomberg",
                "sentiment": 0.65,
                "relevance": 0.85,
                "published": "5 hours ago",
            },
            {
                "headline": f"Market volatility impacts {symbol} trading",
                "source": "CNBC",
                "sentiment": -0.15,
                "relevance": 0.70,
                "published": "12 hours ago",
            },
        ],
        "aggregate_sentiment": 0.42,
        "sentiment_label": "positive",
        "confidence": 0.75,
        "key_themes": [
            "earnings",
            "analyst_upgrade",
            "volatility",
        ],
    }

    return result


def execute_order(
    symbol: str,
    side: str,
    quantity: int,
    order_type: str = "market",
    limit_price: float | None = None,
    stop_price: float | None = None,
    time_in_force: str = "day",
    dry_run: bool = True,
    order_executor: Any | None = None,
) -> dict[str, Any]:
    """
    Execute (or simulate) an order.

    Args:
        symbol: Ticker symbol
        side: "buy" or "sell"
        quantity: Number of shares
        order_type: "market", "limit", "stop", "stop_limit"
        limit_price: Limit price for limit orders
        stop_price: Stop price for stop orders
        time_in_force: "day", "gtc", "ioc", "fok"
        dry_run: If True, simulate only (default True for safety)
        order_executor: Optional executor for live orders

    Returns:
        Dictionary with order result
    """
    order_id = f"ORD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    result = {
        "order_id": order_id,
        "symbol": symbol.upper(),
        "side": side.lower(),
        "quantity": quantity,
        "order_type": order_type,
        "limit_price": limit_price,
        "stop_price": stop_price,
        "time_in_force": time_in_force,
        "timestamp": datetime.utcnow().isoformat(),
        "dry_run": dry_run,
    }

    if dry_run:
        result["status"] = "simulated"
        result["fill_price"] = limit_price if limit_price else 150.00  # Mock fill
        result["fill_quantity"] = quantity
        result["message"] = "Order simulated (dry_run=True)"
    else:
        # In production, would submit to broker
        result["status"] = "rejected"
        result["message"] = "Live orders not implemented - use QuantConnect"

    return result


class TradingToolsServer:
    """
    MCP Server exposing trading tools.

    Provides a consistent interface for tool discovery and execution.

    Usage:
        server = TradingToolsServer()

        # List available tools
        tools = server.list_tools()

        # Call a tool
        result = server.call_tool("get_market_data", {"symbol": "AAPL"})
    """

    def __init__(
        self,
        data_provider: Any | None = None,
        portfolio_provider: Any | None = None,
        risk_manager: Any | None = None,
        sentiment_analyzer: Any | None = None,
    ):
        """
        Initialize the trading tools server.

        Args:
            data_provider: Provider for market data
            portfolio_provider: Provider for portfolio data
            risk_manager: Risk manager for limit checks
            sentiment_analyzer: Analyzer for sentiment data
        """
        self.data_provider = data_provider
        self.portfolio_provider = portfolio_provider
        self.risk_manager = risk_manager
        self.sentiment_analyzer = sentiment_analyzer

        # Register tools
        self._tools: dict[str, Callable] = {
            "get_market_data": self._wrap_get_market_data,
            "get_portfolio_status": self._wrap_get_portfolio_status,
            "check_risk_limits": self._wrap_check_risk_limits,
            "analyze_technicals": self._wrap_analyze_technicals,
            "get_news_sentiment": self._wrap_get_news_sentiment,
            "execute_order": self._wrap_execute_order,
        }

        # Tool schemas for discovery
        self._schemas: dict[str, ToolSchema] = {
            "get_market_data": ToolSchema(
                name="get_market_data",
                description="Get current market data for a symbol",
                category=ToolCategory.MARKET_DATA,
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Ticker symbol"},
                        "include_price": {"type": "boolean", "default": True},
                        "include_volume": {"type": "boolean", "default": True},
                        "include_Greeks": {"type": "boolean", "default": False},
                    },
                    "required": ["symbol"],
                },
                returns="Market data including price, volume, and optionally Greeks",
            ),
            "get_portfolio_status": ToolSchema(
                name="get_portfolio_status",
                description="Get current portfolio status including positions and P&L",
                category=ToolCategory.PORTFOLIO,
                parameters={
                    "type": "object",
                    "properties": {
                        "include_positions": {"type": "boolean", "default": True},
                        "include_pnl": {"type": "boolean", "default": True},
                        "include_exposure": {"type": "boolean", "default": True},
                    },
                },
                returns="Portfolio status with positions, P&L, and exposure metrics",
            ),
            "check_risk_limits": ToolSchema(
                name="check_risk_limits",
                description="Check if proposed action is within risk limits",
                category=ToolCategory.RISK,
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "order_value": {"type": "number"},
                        "position_size_pct": {"type": "number"},
                    },
                },
                returns="Risk check results with pass/fail for each limit",
            ),
            "analyze_technicals": ToolSchema(
                name="analyze_technicals",
                description="Analyze technical indicators for a symbol",
                category=ToolCategory.ANALYSIS,
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Ticker symbol"},
                        "indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of indicators to analyze",
                        },
                        "timeframe": {"type": "string", "default": "1D"},
                    },
                    "required": ["symbol"],
                },
                returns="Technical analysis with indicator values and signals",
            ),
            "get_news_sentiment": ToolSchema(
                name="get_news_sentiment",
                description="Get news sentiment analysis for a symbol",
                category=ToolCategory.SENTIMENT,
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Ticker symbol"},
                        "lookback_hours": {"type": "integer", "default": 24},
                        "min_relevance": {"type": "number", "default": 0.5},
                    },
                    "required": ["symbol"],
                },
                returns="News articles with sentiment scores and aggregate sentiment",
            ),
            "execute_order": ToolSchema(
                name="execute_order",
                description="Execute or simulate a trading order (dry_run=True by default)",
                category=ToolCategory.EXECUTION,
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "side": {"type": "string", "enum": ["buy", "sell"]},
                        "quantity": {"type": "integer"},
                        "order_type": {
                            "type": "string",
                            "enum": ["market", "limit", "stop", "stop_limit"],
                            "default": "market",
                        },
                        "limit_price": {"type": "number"},
                        "stop_price": {"type": "number"},
                        "time_in_force": {
                            "type": "string",
                            "enum": ["day", "gtc", "ioc", "fok"],
                            "default": "day",
                        },
                        "dry_run": {"type": "boolean", "default": True},
                    },
                    "required": ["symbol", "side", "quantity"],
                },
                returns="Order result with status and fill details",
            ),
        }

    def _wrap_get_market_data(self, **kwargs) -> dict[str, Any]:
        """Wrapper for get_market_data with provider injection."""
        return get_market_data(data_provider=self.data_provider, **kwargs)

    def _wrap_get_portfolio_status(self, **kwargs) -> dict[str, Any]:
        """Wrapper for get_portfolio_status with provider injection."""
        return get_portfolio_status(portfolio_provider=self.portfolio_provider, **kwargs)

    def _wrap_check_risk_limits(self, **kwargs) -> dict[str, Any]:
        """Wrapper for check_risk_limits with manager injection."""
        return check_risk_limits(risk_manager=self.risk_manager, **kwargs)

    def _wrap_analyze_technicals(self, **kwargs) -> dict[str, Any]:
        """Wrapper for analyze_technicals."""
        return analyze_technicals(**kwargs)

    def _wrap_get_news_sentiment(self, **kwargs) -> dict[str, Any]:
        """Wrapper for get_news_sentiment with analyzer injection."""
        return get_news_sentiment(sentiment_analyzer=self.sentiment_analyzer, **kwargs)

    def _wrap_execute_order(self, **kwargs) -> dict[str, Any]:
        """Wrapper for execute_order."""
        return execute_order(**kwargs)

    def list_tools(self) -> list[ToolSchema]:
        """
        List all available tools.

        Returns:
            List of tool schemas
        """
        return list(self._schemas.values())

    def get_tool_schema(self, tool_name: str) -> ToolSchema | None:
        """
        Get schema for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            ToolSchema or None if not found
        """
        return self._schemas.get(tool_name)

    def call_tool(self, tool_name: str, parameters: dict[str, Any]) -> ToolResult:
        """
        Call a tool with parameters.

        Args:
            tool_name: Name of the tool to call
            parameters: Tool parameters

        Returns:
            ToolResult with success/error and data
        """
        import time

        start_time = time.time()

        if tool_name not in self._tools:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_name}",
            )

        try:
            tool_func = self._tools[tool_name]
            data = tool_func(**parameters)

            execution_time_ms = (time.time() - start_time) * 1000

            return ToolResult(
                success=True,
                data=data,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Tool '{tool_name}' failed: {e}")

            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time_ms,
            )

    def to_mcp_format(self) -> dict[str, Any]:
        """
        Export tools in MCP-compatible format.

        Returns:
            Dictionary suitable for MCP server registration
        """
        return {
            "tools": [
                {
                    "name": schema.name,
                    "description": schema.description,
                    "inputSchema": schema.parameters,
                }
                for schema in self._schemas.values()
            ]
        }


def create_trading_tools_server(
    data_provider: Any | None = None,
    portfolio_provider: Any | None = None,
    risk_manager: Any | None = None,
    sentiment_analyzer: Any | None = None,
) -> TradingToolsServer:
    """
    Factory function to create a trading tools server.

    Args:
        data_provider: Provider for market data
        portfolio_provider: Provider for portfolio data
        risk_manager: Risk manager for limit checks
        sentiment_analyzer: Analyzer for sentiment data

    Returns:
        Configured TradingToolsServer instance
    """
    return TradingToolsServer(
        data_provider=data_provider,
        portfolio_provider=portfolio_provider,
        risk_manager=risk_manager,
        sentiment_analyzer=sentiment_analyzer,
    )


# =========================================================================
# MCP Server CLI (UPGRADE-014 Category 1)
# =========================================================================


def run_stdio_server() -> None:
    """
    Run the trading tools server in stdio mode for MCP.

    This enables integration with Claude Code and other MCP clients.
    Usage: python -m mcp.trading_tools_server --stdio
    """
    import sys

    server = create_trading_tools_server()

    def handle_request(request: dict[str, Any]) -> dict[str, Any]:
        """Handle a single MCP request."""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        if method == "tools/list":
            # Return available tools
            tools = server.list_tools()
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.parameters,
                        }
                        for t in tools
                    ]
                },
            }

        elif method == "tools/call":
            # Execute a tool
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = server.call_tool(tool_name, arguments)

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result.to_dict(), indent=2),
                        }
                    ],
                    "isError": not result.success,
                },
            }

        elif method == "initialize":
            # Initialize response
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "trading-tools",
                        "version": "1.0.0",
                    },
                    "capabilities": {
                        "tools": {},
                    },
                },
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }

    # Main stdio loop
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = handle_request(request)
            print(json.dumps(response), flush=True)
        except json.JSONDecodeError:
            continue
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": str(e),
                },
            }
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    import sys

    if "--stdio" in sys.argv:
        run_stdio_server()
    elif "--list-tools" in sys.argv:
        # Quick tool listing for debugging
        server = create_trading_tools_server()
        for tool in server.list_tools():
            print(f"  - {tool.name}: {tool.description}")
    else:
        print("Trading Tools MCP Server")
        print("Usage:")
        print("  python -m mcp.trading_tools_server --stdio   # Run in MCP mode")
        print("  python -m mcp.trading_tools_server --list-tools  # List available tools")
