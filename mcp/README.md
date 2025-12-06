# MCP (Model Context Protocol) Servers

This directory contains specialized MCP servers for the autonomous AI trading system. These servers provide trading-specific tools that can be invoked by Claude and other LLM agents.

## Overview

UPGRADE-015 implements a comprehensive MCP server ecosystem with four specialized servers:

| Server | Tools | Purpose |
|--------|-------|---------|
| **market-data** | 6 | Quotes, option chains, Greeks, historical data, IV surface |
| **broker** | 6 | Orders, positions, fills, account info |
| **portfolio** | 5 | Portfolio status, exposure, risk metrics, P&L |
| **backtest** | 5 | Backtest execution, status, results, analysis |

## Architecture

```
mcp/
├── __init__.py              # Package exports
├── base_server.py           # BaseMCPServer abstract class
├── schemas.py               # Pydantic models for all requests/responses
├── market_data_server.py    # Market data tools
├── broker_server.py         # Order management tools
├── portfolio_server.py      # Portfolio analysis tools
├── backtest_server.py       # Backtesting tools
├── trading_tools_server.py  # Legacy combined server (UPGRADE-014)
└── README.md                # This file
```

## Quick Start

### Using in Claude Code

The servers are configured in `.mcp.json` and available automatically:

```python
# In Claude Code, tools are available as:
# mcp__market-data__get_quote
# mcp__broker__place_order
# mcp__portfolio__get_risk_metrics
# mcp__backtest__run_backtest
```

### Direct Python Usage

```python
from mcp import (
    create_market_data_server,
    create_broker_server,
    create_portfolio_server,
    create_backtest_server,
)

# Create server with mock data for testing
server = create_market_data_server(mock_mode=True)

# Start server
await server.start()

# Call tools
result = await server.call_tool("get_quote", {"symbol": "SPY"})
print(result.data)

# Stop server
await server.stop()
```

## Server Details

### Market Data Server

**Tools:**
- `get_quote` - Get real-time quote with optional Greeks and volume
- `get_option_chain` - Get option contracts with filters
- `get_greeks` - Get Greeks for specific contract
- `get_historical` - Get OHLCV historical data
- `get_iv_surface` - Get implied volatility surface
- `get_market_status` - Get market open/closed status

**Example:**
```python
result = await server.call_tool("get_quote", {
    "symbol": "AAPL",
    "include_greeks": True,
    "include_volume": True,
})
```

### Broker Server

**Tools:**
- `get_positions` - Get current portfolio positions
- `get_orders` - Get open/recent orders
- `place_order` - Submit order (PAPER MODE ONLY)
- `cancel_order` - Cancel pending order
- `get_fills` - Get execution history
- `get_account_info` - Get account balance and buying power

**Safety:** Live trading is blocked by default. Only paper trading is allowed via MCP tools.

**Example:**
```python
result = await server.call_tool("place_order", {
    "symbol": "SPY",
    "quantity": 100,
    "side": "buy",
    "order_type": "limit",
    "limit_price": 450.00,
})
```

### Portfolio Server

**Tools:**
- `get_portfolio` - Complete portfolio status
- `get_exposure` - Exposure breakdown by sector/asset/symbol
- `get_risk_metrics` - VaR, Sharpe, Sortino, max drawdown
- `get_pnl` - P&L by period (daily/weekly/monthly/ytd)
- `get_holdings_summary` - Holdings with key metrics

**Example:**
```python
result = await server.call_tool("get_risk_metrics", {
    "lookback_days": 30,
    "confidence_level": 0.95,
})
```

### Backtest Server

**Tools:**
- `run_backtest` - Execute backtest with algorithm
- `get_backtest_status` - Check running/completed status
- `get_backtest_results` - Get detailed metrics and trades
- `parse_backtest_report` - Parse report files
- `list_backtests` - List recent backtests

**Example:**
```python
result = await server.call_tool("run_backtest", {
    "algorithm_id": "hybrid_options_bot",
    "start_date": "2024-01-01",
    "end_date": "2024-12-01",
    "initial_cash": 100000,
})
```

## Schema Validation

All requests and responses use Pydantic models defined in `schemas.py`:

```python
from mcp.schemas import (
    QuoteRequest,
    OrderRequest,
    PortfolioRequest,
    BacktestRequest,
)

# Automatic validation and normalization
request = QuoteRequest(symbol="aapl")  # Normalizes to "AAPL"
```

## Error Handling

All tools return `ToolResult` with success/failure status:

```python
result = await server.call_tool("get_quote", {"symbol": "SPY"})

if result.success:
    print(result.data)
else:
    print(f"Error: {result.error} (code: {result.error_code})")
```

**Error Codes:**
- `TOOL_NOT_FOUND` - Unknown tool name
- `SERVER_NOT_RUNNING` - Server not started
- `TIMEOUT` - Tool execution timeout
- `EXECUTION_ERROR` - General execution error
- `LIVE_TRADING_BLOCKED` - Live trading attempted
- `BACKTEST_NOT_FOUND` - Backtest ID not found

## Health Checks

All servers support health checks:

```python
health = server.health_check()
# Returns: {
#     "server": "market-data",
#     "version": "1.0.0",
#     "state": "running",
#     "is_healthy": True,
#     "tools_registered": 6,
#     "total_calls": 42,
#     "error_rate": 0.02,
# }
```

## Configuration

Server configuration via `ServerConfig`:

```python
from mcp.base_server import ServerConfig

config = ServerConfig(
    name="custom-server",
    version="1.0.0",
    description="My custom server",
    max_concurrent_calls=10,
    timeout_seconds=30.0,
    mock_mode=True,
)
```

## Testing

Run tests for all MCP servers:

```bash
# All MCP tests
pytest tests/mcp/ -v

# Specific server
pytest tests/mcp/test_market_data_server.py -v
pytest tests/mcp/test_broker_server.py -v
pytest tests/mcp/test_portfolio_backtest_server.py -v
```

## Related Documentation

- [UPGRADE-015 Research](../docs/research/UPGRADE-015-AUTONOMOUS-TRADING-EXTENDED-RESEARCH.md)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [Claude Code MCP](https://docs.anthropic.com/claude-code)
