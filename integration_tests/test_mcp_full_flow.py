"""
MCP Full Flow Integration Tests

UPGRADE-015 Phase 12: Integration & Final Validation

Tests complete MCP server workflow:
- Market data server tools
- Broker server tools
- Portfolio server tools
- Backtest server tools
"""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMCPMarketDataServer:
    """Integration tests for Market Data MCP Server."""

    def test_market_data_server_import(self):
        """Test that market data server can be imported."""
        from mcp.market_data_server import MarketDataServer

        server = MarketDataServer()
        assert server is not None

    def test_market_data_server_tools(self):
        """Test market data server has required tools."""
        from mcp.market_data_server import MarketDataServer

        server = MarketDataServer()
        tools = server.get_tools()

        # Verify expected tools exist
        tool_names = [t["name"] for t in tools]
        expected_tools = [
            "get_quote",
            "get_option_chain",
            "get_historical_data",
            "get_iv_rank",
            "get_market_status",
            "get_earnings_calendar",
        ]

        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"


class TestMCPBrokerServer:
    """Integration tests for Broker MCP Server."""

    def test_broker_server_import(self):
        """Test that broker server can be imported."""
        from mcp.broker_server import BrokerServer

        server = BrokerServer()
        assert server is not None

    def test_broker_server_tools(self):
        """Test broker server has required tools."""
        from mcp.broker_server import BrokerServer

        server = BrokerServer()
        tools = server.get_tools()

        tool_names = [t["name"] for t in tools]
        expected_tools = [
            "get_positions",
            "get_orders",
            "place_order",
            "cancel_order",
            "get_fills",
            "get_account_info",
        ]

        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"

    def test_broker_paper_mode_only(self):
        """Test that broker defaults to paper trading mode."""
        from mcp.broker_server import BrokerServer

        server = BrokerServer()
        assert server.is_paper_mode() is True


class TestMCPPortfolioServer:
    """Integration tests for Portfolio MCP Server."""

    def test_portfolio_server_import(self):
        """Test that portfolio server can be imported."""
        from mcp.portfolio_server import PortfolioServer

        server = PortfolioServer()
        assert server is not None

    def test_portfolio_server_tools(self):
        """Test portfolio server has required tools."""
        from mcp.portfolio_server import PortfolioServer

        server = PortfolioServer()
        tools = server.get_tools()

        tool_names = [t["name"] for t in tools]
        expected_tools = [
            "get_portfolio_summary",
            "get_portfolio_positions",
            "get_portfolio_performance",
            "get_risk_metrics",
            "get_allocation",
        ]

        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"


class TestMCPBacktestServer:
    """Integration tests for Backtest MCP Server."""

    def test_backtest_server_import(self):
        """Test that backtest server can be imported."""
        from mcp.backtest_server import BacktestServer

        server = BacktestServer()
        assert server is not None

    def test_backtest_server_tools(self):
        """Test backtest server has required tools."""
        from mcp.backtest_server import BacktestServer

        server = BacktestServer()
        tools = server.get_tools()

        tool_names = [t["name"] for t in tools]
        expected_tools = [
            "run_backtest",
            "get_backtest_results",
            "list_backtests",
            "compare_backtests",
            "get_backtest_trades",
        ]

        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"


class TestMCPServerCoordination:
    """Integration tests for MCP server coordination."""

    def test_all_servers_coexist(self):
        """Test that all servers can be instantiated together."""
        from mcp.backtest_server import BacktestServer
        from mcp.broker_server import BrokerServer
        from mcp.market_data_server import MarketDataServer
        from mcp.portfolio_server import PortfolioServer

        market_server = MarketDataServer()
        broker_server = BrokerServer()
        portfolio_server = PortfolioServer()
        backtest_server = BacktestServer()

        assert market_server is not None
        assert broker_server is not None
        assert portfolio_server is not None
        assert backtest_server is not None

    def test_total_tools_count(self):
        """Test total number of MCP tools available."""
        from mcp.backtest_server import BacktestServer
        from mcp.broker_server import BrokerServer
        from mcp.market_data_server import MarketDataServer
        from mcp.portfolio_server import PortfolioServer

        servers = [
            MarketDataServer(),
            BrokerServer(),
            PortfolioServer(),
            BacktestServer(),
        ]

        total_tools = sum(len(s.get_tools()) for s in servers)

        # Should have at least 22 tools (6+6+5+5)
        assert total_tools >= 22, f"Expected 22+ tools, got {total_tools}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
