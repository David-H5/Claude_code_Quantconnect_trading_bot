"""
MCP (Model Context Protocol) Integration

Provides trading-specific MCP servers and tools for integration with
Claude and other LLM agents.

UPGRADE-014 Category 1: Architecture Enhancements
UPGRADE-015 Phase 1-3: Extended MCP Server Ecosystem

Available Servers:
- trading_tools_server: Market data, portfolio, risk, and analysis tools (UPGRADE-014)
- market_data_server: Quotes, option chains, Greeks, historical data (UPGRADE-015)
- broker_server: Order management, positions, fills (UPGRADE-015)
- portfolio_server: Portfolio status, exposure, risk metrics (UPGRADE-015)
- backtest_server: Backtest execution and result analysis (UPGRADE-015)
"""

# Base server classes (UPGRADE-015)
# Backtest Server (UPGRADE-015)
from mcp.backtest_server import (
    BacktestServer,
    create_backtest_server,
)
from mcp.base_server import (
    BaseMCPServer,
    ServerConfig,
    ServerState,
    ToolCategory,
    create_base_server,
)
from mcp.base_server import (
    ToolResult as BaseToolResult,
)
from mcp.base_server import (
    ToolSchema as BaseToolSchema,
)

# Broker Server (UPGRADE-015)
from mcp.broker_server import (
    BrokerServer,
    create_broker_server,
)

# Market Data Server (UPGRADE-015)
from mcp.market_data_server import (
    MarketDataServer,
    create_market_data_server,
)

# Portfolio Server (UPGRADE-015)
from mcp.portfolio_server import (
    PortfolioServer,
    create_portfolio_server,
)

# Pydantic schemas (UPGRADE-015)
from mcp.schemas import (
    BacktestMetrics,
    # Backtest
    BacktestRequest,
    BacktestResponse,
    CancelOrderRequest,
    CancelOrderResponse,
    ErrorResponse,
    ExposureData,
    FillRecord,
    GreeksData,
    # System
    HealthCheckResponse,
    HistoricalRequest,
    HistoricalResponse,
    OHLCVBar,
    OptionChainRequest,
    OptionChainResponse,
    OptionContract,
    OptionType,
    # Orders
    OrderRequest,
    OrderResponse,
    # Enums
    OrderSide,
    OrderStatus,
    OrderType,
    PnLData,
    # Portfolio
    PortfolioRequest,
    PortfolioResponse,
    Position,
    PriceData,
    # Market Data
    QuoteRequest,
    QuoteResponse,
    Resolution,
    # Risk
    RiskCheckRequest,
    RiskCheckResponse,
    RiskMetrics,
    TimeInForce,
    TradingMode,
    VolumeData,
)

# Trading Tools Server (UPGRADE-014)
from mcp.trading_tools_server import (
    ToolResult,
    ToolSchema,
    TradingToolsServer,
    analyze_technicals,
    check_risk_limits,
    create_trading_tools_server,
    execute_order,
    # Tool functions for direct use
    get_market_data,
    get_news_sentiment,
    get_portfolio_status,
)


__all__ = [
    # Base classes
    "BaseMCPServer",
    "ServerConfig",
    "ServerState",
    "BaseToolSchema",
    "BaseToolResult",
    "ToolCategory",
    "create_base_server",
    # Schemas - Enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "OptionType",
    "Resolution",
    "TradingMode",
    # Schemas - Market Data
    "QuoteRequest",
    "QuoteResponse",
    "PriceData",
    "VolumeData",
    "GreeksData",
    "OptionChainRequest",
    "OptionChainResponse",
    "OptionContract",
    "HistoricalRequest",
    "HistoricalResponse",
    "OHLCVBar",
    # Schemas - Orders
    "OrderRequest",
    "OrderResponse",
    "CancelOrderRequest",
    "CancelOrderResponse",
    "FillRecord",
    # Schemas - Portfolio
    "PortfolioRequest",
    "PortfolioResponse",
    "Position",
    "PnLData",
    "ExposureData",
    # Schemas - Risk
    "RiskCheckRequest",
    "RiskCheckResponse",
    "RiskMetrics",
    # Schemas - Backtest
    "BacktestRequest",
    "BacktestResponse",
    "BacktestMetrics",
    # Schemas - System
    "HealthCheckResponse",
    "ErrorResponse",
    # Market Data Server
    "MarketDataServer",
    "create_market_data_server",
    # Broker Server
    "BrokerServer",
    "create_broker_server",
    # Portfolio Server
    "PortfolioServer",
    "create_portfolio_server",
    # Backtest Server
    "BacktestServer",
    "create_backtest_server",
    # Trading Tools Server (legacy)
    "TradingToolsServer",
    "ToolSchema",
    "ToolResult",
    "create_trading_tools_server",
    "get_market_data",
    "get_portfolio_status",
    "check_risk_limits",
    "analyze_technicals",
    "get_news_sentiment",
    "execute_order",
]
