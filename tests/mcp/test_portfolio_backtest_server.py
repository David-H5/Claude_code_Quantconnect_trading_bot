"""
Tests for Portfolio and Backtest MCP Servers

UPGRADE-015 Phase 3: Portfolio & Backtest MCP

Tests cover:
- Portfolio server initialization and tools
- Backtest server initialization and tools
- Portfolio status, exposure, risk metrics
- Backtest execution and results
- Error handling
"""

import sys

import pytest


sys.path.insert(0, "/home/dshooter/projects/Claude_code_Quantconnect_trading_bot")

from mcp.backtest_server import (
    BacktestServer,
    create_backtest_server,
)
from mcp.base_server import (
    ServerConfig,
    ServerState,
)
from mcp.portfolio_server import (
    PortfolioServer,
    create_portfolio_server,
)


# =============================================================================
# Portfolio Server Fixtures
# =============================================================================


@pytest.fixture
def portfolio_config():
    """Create test portfolio server configuration."""
    return ServerConfig(
        name="test-portfolio",
        version="1.0.0",
        description="Test portfolio server",
        mock_mode=True,
        timeout_seconds=5.0,
    )


@pytest.fixture
def portfolio_server(portfolio_config):
    """Create portfolio server instance."""
    return PortfolioServer(config=portfolio_config, mock_mode=True)


@pytest.fixture
async def running_portfolio_server(portfolio_server):
    """Create and start a portfolio server."""
    await portfolio_server.start()
    yield portfolio_server
    await portfolio_server.stop()


# =============================================================================
# Backtest Server Fixtures
# =============================================================================


@pytest.fixture
def backtest_config():
    """Create test backtest server configuration."""
    return ServerConfig(
        name="test-backtest",
        version="1.0.0",
        description="Test backtest server",
        mock_mode=True,
        timeout_seconds=5.0,
    )


@pytest.fixture
def backtest_server(backtest_config):
    """Create backtest server instance."""
    return BacktestServer(config=backtest_config, mock_mode=True)


@pytest.fixture
async def running_backtest_server(backtest_server):
    """Create and start a backtest server."""
    await backtest_server.start()
    yield backtest_server
    await backtest_server.stop()


# =============================================================================
# Portfolio Server Lifecycle Tests
# =============================================================================


class TestPortfolioServerLifecycle:
    """Test portfolio server lifecycle management."""

    def test_server_creation(self, portfolio_config):
        """Test server can be created."""
        server = PortfolioServer(config=portfolio_config)
        assert server is not None
        assert server.name == "test-portfolio"
        assert server.state == ServerState.STOPPED

    def test_create_portfolio_server_helper(self):
        """Test create_portfolio_server helper function."""
        server = create_portfolio_server(mock_mode=True)
        assert server is not None
        assert server.name == "portfolio"

    @pytest.mark.asyncio
    async def test_server_start_stop(self, portfolio_server):
        """Test server can start and stop."""
        await portfolio_server.start()
        assert portfolio_server.state == ServerState.RUNNING

        await portfolio_server.stop()
        assert portfolio_server.state == ServerState.STOPPED

    @pytest.mark.asyncio
    async def test_tools_registered(self, running_portfolio_server):
        """Test tools are registered."""
        schemas = running_portfolio_server.get_tool_schemas()
        tool_names = [s.name for s in schemas]

        assert "get_portfolio" in tool_names
        assert "get_exposure" in tool_names
        assert "get_risk_metrics" in tool_names
        assert "get_pnl" in tool_names
        assert "get_holdings_summary" in tool_names


# =============================================================================
# Portfolio Tools Tests
# =============================================================================


class TestGetPortfolio:
    """Test get_portfolio tool."""

    @pytest.mark.asyncio
    async def test_get_portfolio_basic(self, running_portfolio_server):
        """Test basic portfolio retrieval."""
        result = await running_portfolio_server.call_tool("get_portfolio", {})

        assert result.success is True
        assert "cash" in result.data
        assert "total_value" in result.data
        assert "buying_power" in result.data
        assert "timestamp" in result.data

    @pytest.mark.asyncio
    async def test_get_portfolio_with_positions(self, running_portfolio_server):
        """Test portfolio with positions."""
        result = await running_portfolio_server.call_tool(
            "get_portfolio",
            {"include_positions": True},
        )

        assert result.success is True
        assert "positions" in result.data
        assert len(result.data["positions"]) > 0

    @pytest.mark.asyncio
    async def test_get_portfolio_with_pnl(self, running_portfolio_server):
        """Test portfolio with P&L."""
        result = await running_portfolio_server.call_tool(
            "get_portfolio",
            {"include_pnl": True},
        )

        assert result.success is True
        assert "pnl" in result.data


class TestGetExposure:
    """Test get_exposure tool."""

    @pytest.mark.asyncio
    async def test_get_exposure_by_sector(self, running_portfolio_server):
        """Test exposure by sector."""
        result = await running_portfolio_server.call_tool(
            "get_exposure",
            {"group_by": "sector"},
        )

        assert result.success is True
        assert "exposure" in result.data
        assert "gross_exposure" in result.data
        assert "net_exposure" in result.data

    @pytest.mark.asyncio
    async def test_get_exposure_by_symbol(self, running_portfolio_server):
        """Test exposure by symbol."""
        result = await running_portfolio_server.call_tool(
            "get_exposure",
            {"group_by": "symbol"},
        )

        assert result.success is True
        assert result.data["grouped_by"] == "symbol"


class TestGetRiskMetrics:
    """Test get_risk_metrics tool."""

    @pytest.mark.asyncio
    async def test_get_risk_metrics(self, running_portfolio_server):
        """Test risk metrics retrieval."""
        result = await running_portfolio_server.call_tool("get_risk_metrics", {})

        assert result.success is True
        assert "var" in result.data
        assert "sharpe_ratio" in result.data
        assert "max_drawdown" in result.data
        assert "beta" in result.data

    @pytest.mark.asyncio
    async def test_risk_metrics_confidence_level(self, running_portfolio_server):
        """Test risk metrics with confidence level."""
        result = await running_portfolio_server.call_tool(
            "get_risk_metrics",
            {"confidence_level": 0.99},
        )

        assert result.success is True
        assert result.data["confidence_level"] == 0.99


class TestGetPnL:
    """Test get_pnl tool."""

    @pytest.mark.asyncio
    async def test_get_pnl_daily(self, running_portfolio_server):
        """Test daily P&L."""
        result = await running_portfolio_server.call_tool(
            "get_pnl",
            {"period": "daily"},
        )

        assert result.success is True
        assert result.data["period"] == "daily"
        assert "pnl" in result.data
        assert "pnl_pct" in result.data

    @pytest.mark.asyncio
    async def test_get_pnl_by_symbol(self, running_portfolio_server):
        """Test P&L by symbol."""
        result = await running_portfolio_server.call_tool(
            "get_pnl",
            {"period": "daily", "by_symbol": True},
        )

        assert result.success is True
        assert "by_symbol" in result.data


class TestGetHoldingsSummary:
    """Test get_holdings_summary tool."""

    @pytest.mark.asyncio
    async def test_get_holdings_summary(self, running_portfolio_server):
        """Test holdings summary."""
        result = await running_portfolio_server.call_tool("get_holdings_summary", {})

        assert result.success is True
        assert "holdings" in result.data
        assert "total_holdings" in result.data

    @pytest.mark.asyncio
    async def test_holdings_sorted_by_pnl(self, running_portfolio_server):
        """Test holdings sorted by P&L."""
        result = await running_portfolio_server.call_tool(
            "get_holdings_summary",
            {"sort_by": "pnl"},
        )

        assert result.success is True
        assert result.data["sorted_by"] == "pnl"


# =============================================================================
# Backtest Server Lifecycle Tests
# =============================================================================


class TestBacktestServerLifecycle:
    """Test backtest server lifecycle management."""

    def test_server_creation(self, backtest_config):
        """Test server can be created."""
        server = BacktestServer(config=backtest_config)
        assert server is not None
        assert server.name == "test-backtest"
        assert server.state == ServerState.STOPPED

    def test_create_backtest_server_helper(self):
        """Test create_backtest_server helper function."""
        server = create_backtest_server(mock_mode=True)
        assert server is not None
        assert server.name == "backtest"

    @pytest.mark.asyncio
    async def test_server_start_stop(self, backtest_server):
        """Test server can start and stop."""
        await backtest_server.start()
        assert backtest_server.state == ServerState.RUNNING

        await backtest_server.stop()
        assert backtest_server.state == ServerState.STOPPED

    @pytest.mark.asyncio
    async def test_tools_registered(self, running_backtest_server):
        """Test tools are registered."""
        schemas = running_backtest_server.get_tool_schemas()
        tool_names = [s.name for s in schemas]

        assert "run_backtest" in tool_names
        assert "get_backtest_status" in tool_names
        assert "get_backtest_results" in tool_names
        assert "parse_backtest_report" in tool_names
        assert "list_backtests" in tool_names


# =============================================================================
# Backtest Tools Tests
# =============================================================================


class TestRunBacktest:
    """Test run_backtest tool."""

    @pytest.mark.asyncio
    async def test_run_backtest_basic(self, running_backtest_server):
        """Test basic backtest execution."""
        result = await running_backtest_server.call_tool(
            "run_backtest",
            {"algorithm_id": "test_algo"},
        )

        assert result.success is True
        assert "backtest_id" in result.data
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_run_backtest_with_dates(self, running_backtest_server):
        """Test backtest with custom dates."""
        result = await running_backtest_server.call_tool(
            "run_backtest",
            {
                "algorithm_id": "test_algo",
                "start_date": "2024-01-01",
                "end_date": "2024-06-30",
            },
        )

        assert result.success is True
        assert result.data["start_date"] == "2024-01-01"
        assert result.data["end_date"] == "2024-06-30"

    @pytest.mark.asyncio
    async def test_run_backtest_with_cash(self, running_backtest_server):
        """Test backtest with custom initial cash."""
        result = await running_backtest_server.call_tool(
            "run_backtest",
            {
                "algorithm_id": "test_algo",
                "initial_cash": 50000,
            },
        )

        assert result.success is True
        assert result.data["initial_cash"] == 50000


class TestGetBacktestStatus:
    """Test get_backtest_status tool."""

    @pytest.mark.asyncio
    async def test_get_backtest_status(self, running_backtest_server):
        """Test getting backtest status."""
        # First run a backtest
        run_result = await running_backtest_server.call_tool(
            "run_backtest",
            {"algorithm_id": "test_algo"},
        )
        backtest_id = run_result.data["backtest_id"]

        # Get status
        result = await running_backtest_server.call_tool(
            "get_backtest_status",
            {"backtest_id": backtest_id},
        )

        assert result.success is True
        assert result.data["backtest_id"] == backtest_id
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_backtest_status_not_found(self, running_backtest_server):
        """Test status for non-existent backtest."""
        result = await running_backtest_server.call_tool(
            "get_backtest_status",
            {"backtest_id": "FAKE-ID"},
        )

        assert result.success is True
        assert result.data.get("success") is False
        assert result.data["error_code"] == "BACKTEST_NOT_FOUND"


class TestGetBacktestResults:
    """Test get_backtest_results tool."""

    @pytest.mark.asyncio
    async def test_get_backtest_results(self, running_backtest_server):
        """Test getting backtest results."""
        # First run a backtest
        run_result = await running_backtest_server.call_tool(
            "run_backtest",
            {"algorithm_id": "test_algo"},
        )
        backtest_id = run_result.data["backtest_id"]

        # Get results
        result = await running_backtest_server.call_tool(
            "get_backtest_results",
            {"backtest_id": backtest_id},
        )

        assert result.success is True
        assert "metrics" in result.data
        assert "total_return" in result.data["metrics"]
        assert "sharpe_ratio" in result.data["metrics"]

    @pytest.mark.asyncio
    async def test_get_backtest_results_with_trades(self, running_backtest_server):
        """Test results with trades."""
        run_result = await running_backtest_server.call_tool(
            "run_backtest",
            {"algorithm_id": "test_algo"},
        )
        backtest_id = run_result.data["backtest_id"]

        result = await running_backtest_server.call_tool(
            "get_backtest_results",
            {"backtest_id": backtest_id, "include_trades": True},
        )

        assert result.success is True
        assert "trades" in result.data

    @pytest.mark.asyncio
    async def test_get_backtest_results_with_equity_curve(self, running_backtest_server):
        """Test results with equity curve."""
        run_result = await running_backtest_server.call_tool(
            "run_backtest",
            {"algorithm_id": "test_algo"},
        )
        backtest_id = run_result.data["backtest_id"]

        result = await running_backtest_server.call_tool(
            "get_backtest_results",
            {"backtest_id": backtest_id, "include_equity_curve": True},
        )

        assert result.success is True
        assert "equity_curve" in result.data


class TestParseBacktestReport:
    """Test parse_backtest_report tool."""

    @pytest.mark.asyncio
    async def test_parse_backtest_report(self, running_backtest_server):
        """Test parsing backtest report."""
        result = await running_backtest_server.call_tool(
            "parse_backtest_report",
            {"report_path": "/path/to/report.json"},
        )

        assert result.success is True
        assert result.data["parsed"] is True
        assert "metrics" in result.data


class TestListBacktests:
    """Test list_backtests tool."""

    @pytest.mark.asyncio
    async def test_list_backtests(self, running_backtest_server):
        """Test listing backtests."""
        result = await running_backtest_server.call_tool("list_backtests", {})

        assert result.success is True
        assert "backtests" in result.data
        assert "total" in result.data

    @pytest.mark.asyncio
    async def test_list_backtests_with_limit(self, running_backtest_server):
        """Test listing backtests with limit."""
        # Run multiple backtests
        for _ in range(3):
            await running_backtest_server.call_tool(
                "run_backtest",
                {"algorithm_id": "test_algo"},
            )

        result = await running_backtest_server.call_tool(
            "list_backtests",
            {"limit": 2},
        )

        assert result.success is True
        assert len(result.data["backtests"]) <= 2

    @pytest.mark.asyncio
    async def test_list_backtests_by_status(self, running_backtest_server):
        """Test listing backtests by status."""
        result = await running_backtest_server.call_tool(
            "list_backtests",
            {"status": "completed"},
        )

        assert result.success is True
        assert result.data["filter_status"] == "completed"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling for both servers."""

    @pytest.mark.asyncio
    async def test_portfolio_unknown_tool(self, running_portfolio_server):
        """Test calling unknown tool on portfolio server."""
        result = await running_portfolio_server.call_tool("unknown_tool", {})

        assert result.success is False
        assert result.error_code == "TOOL_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_backtest_unknown_tool(self, running_backtest_server):
        """Test calling unknown tool on backtest server."""
        result = await running_backtest_server.call_tool("unknown_tool", {})

        assert result.success is False
        assert result.error_code == "TOOL_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_call_on_stopped_portfolio_server(self, portfolio_server):
        """Test calling tool on stopped portfolio server."""
        result = await portfolio_server.call_tool("get_portfolio", {})

        assert result.success is False
        assert result.error_code == "SERVER_NOT_RUNNING"

    @pytest.mark.asyncio
    async def test_call_on_stopped_backtest_server(self, backtest_server):
        """Test calling tool on stopped backtest server."""
        result = await backtest_server.call_tool("run_backtest", {"algorithm_id": "test"})

        assert result.success is False
        assert result.error_code == "SERVER_NOT_RUNNING"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
