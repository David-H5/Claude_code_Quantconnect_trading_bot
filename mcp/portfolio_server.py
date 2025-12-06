"""
Portfolio MCP Server Implementation

Provides portfolio management and analysis tools for the MCP ecosystem.
Includes portfolio status, exposure analysis, risk metrics, and P&L tracking.

UPGRADE-015 Phase 3: Portfolio & Backtest MCP

Tools:
    - get_portfolio: Get complete portfolio status
    - get_exposure: Get portfolio exposure breakdown
    - get_risk_metrics: Get portfolio risk metrics (VaR, Sharpe, etc.)
    - get_pnl: Get profit/loss breakdown
    - get_holdings_summary: Get summary of current holdings

Usage:
    server = create_portfolio_server(mock_mode=True)
    await server.start()
    result = await server.call_tool("get_portfolio", {})
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from mcp.base_server import (
    BaseMCPServer,
    ServerConfig,
    ToolCategory,
    ToolSchema,
)


logger = logging.getLogger(__name__)


class PortfolioServer(BaseMCPServer):
    """
    MCP server for portfolio management operations.

    Provides tools for portfolio analysis, exposure tracking, risk metrics,
    and P&L reporting.
    """

    def __init__(
        self,
        config: ServerConfig | None = None,
        mock_mode: bool = True,
    ):
        """
        Initialize the portfolio server.

        Args:
            config: Server configuration
            mock_mode: Whether to use mock data (for testing)
        """
        if config is None:
            config = ServerConfig(
                name="portfolio",
                version="1.0.0",
                description="Portfolio management: status, exposure, risk, P&L",
            )

        super().__init__(config)
        self.mock_mode = mock_mode

        # Mock portfolio data
        self._portfolio_data: dict[str, Any] = {}

        self._register_tools()

    def _register_tools(self) -> None:
        """Register all portfolio tools."""

        # Get Portfolio Tool
        self.register_tool(
            self.get_portfolio,
            ToolSchema(
                name="get_portfolio",
                description="Get complete portfolio status including positions, cash, and total value",
                category=ToolCategory.PORTFOLIO,
                parameters={
                    "properties": {
                        "include_positions": {
                            "type": "boolean",
                            "description": "Include position details",
                            "default": True,
                        },
                        "include_pnl": {
                            "type": "boolean",
                            "description": "Include P&L breakdown",
                            "default": True,
                        },
                    },
                    "required": [],
                },
                returns="Complete portfolio status",
            ),
        )

        # Get Exposure Tool
        self.register_tool(
            self.get_exposure,
            ToolSchema(
                name="get_exposure",
                description="Get portfolio exposure breakdown by sector, asset class, and position",
                category=ToolCategory.RISK,
                parameters={
                    "properties": {
                        "group_by": {
                            "type": "string",
                            "enum": ["sector", "asset_class", "symbol"],
                            "description": "How to group exposure",
                            "default": "sector",
                        },
                    },
                    "required": [],
                },
                returns="Portfolio exposure breakdown",
            ),
        )

        # Get Risk Metrics Tool
        self.register_tool(
            self.get_risk_metrics,
            ToolSchema(
                name="get_risk_metrics",
                description="Get portfolio risk metrics including VaR, Sharpe ratio, max drawdown",
                category=ToolCategory.RISK,
                parameters={
                    "properties": {
                        "lookback_days": {
                            "type": "integer",
                            "description": "Days of history for calculations",
                            "default": 30,
                        },
                        "confidence_level": {
                            "type": "number",
                            "description": "VaR confidence level (0.95 or 0.99)",
                            "default": 0.95,
                        },
                    },
                    "required": [],
                },
                returns="Risk metrics including VaR, Sharpe, Sortino, max drawdown",
            ),
        )

        # Get P&L Tool
        self.register_tool(
            self.get_pnl,
            ToolSchema(
                name="get_pnl",
                description="Get profit and loss breakdown by period",
                category=ToolCategory.PORTFOLIO,
                parameters={
                    "properties": {
                        "period": {
                            "type": "string",
                            "enum": ["daily", "weekly", "monthly", "ytd", "all_time"],
                            "description": "P&L period",
                            "default": "daily",
                        },
                        "by_symbol": {
                            "type": "boolean",
                            "description": "Break down by symbol",
                            "default": False,
                        },
                    },
                    "required": [],
                },
                returns="P&L breakdown for specified period",
            ),
        )

        # Get Holdings Summary Tool
        self.register_tool(
            self.get_holdings_summary,
            ToolSchema(
                name="get_holdings_summary",
                description="Get summary of current holdings with key metrics",
                category=ToolCategory.PORTFOLIO,
                parameters={
                    "properties": {
                        "sort_by": {
                            "type": "string",
                            "enum": ["value", "pnl", "pnl_pct", "symbol"],
                            "description": "Sort holdings by",
                            "default": "value",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max holdings to return",
                            "default": 20,
                        },
                    },
                    "required": [],
                },
                returns="Holdings summary with key metrics",
            ),
        )

    async def initialize(self) -> None:
        """Initialize the portfolio server."""
        logger.info("Initializing portfolio server")

        if self.mock_mode:
            self._initialize_mock_data()

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up portfolio server")

    def _initialize_mock_data(self) -> None:
        """Initialize mock portfolio data."""
        self._portfolio_data = {
            "cash": 45000.00,
            "total_value": 125000.00,
            "buying_power": 90000.00,
            "positions": [
                {
                    "symbol": "SPY",
                    "quantity": 100,
                    "average_cost": 445.50,
                    "current_price": 450.25,
                    "market_value": 45025.00,
                    "unrealized_pnl": 475.00,
                    "unrealized_pnl_pct": 0.0107,
                    "weight": 0.36,
                    "sector": "ETF",
                    "asset_class": "equity",
                },
                {
                    "symbol": "AAPL",
                    "quantity": 50,
                    "average_cost": 172.30,
                    "current_price": 175.80,
                    "market_value": 8790.00,
                    "unrealized_pnl": 175.00,
                    "unrealized_pnl_pct": 0.0203,
                    "weight": 0.07,
                    "sector": "Technology",
                    "asset_class": "equity",
                },
                {
                    "symbol": "MSFT",
                    "quantity": 30,
                    "average_cost": 378.00,
                    "current_price": 385.50,
                    "market_value": 11565.00,
                    "unrealized_pnl": 225.00,
                    "unrealized_pnl_pct": 0.0198,
                    "weight": 0.09,
                    "sector": "Technology",
                    "asset_class": "equity",
                },
                {
                    "symbol": "GOOGL",
                    "quantity": 40,
                    "average_cost": 140.25,
                    "current_price": 145.60,
                    "market_value": 5824.00,
                    "unrealized_pnl": 214.00,
                    "unrealized_pnl_pct": 0.0381,
                    "weight": 0.047,
                    "sector": "Technology",
                    "asset_class": "equity",
                },
                {
                    "symbol": "JPM",
                    "quantity": 25,
                    "average_cost": 195.00,
                    "current_price": 198.50,
                    "market_value": 4962.50,
                    "unrealized_pnl": 87.50,
                    "unrealized_pnl_pct": 0.0179,
                    "weight": 0.04,
                    "sector": "Financial",
                    "asset_class": "equity",
                },
            ],
            "pnl": {
                "daily": 325.50,
                "daily_pct": 0.0026,
                "weekly": 1250.00,
                "weekly_pct": 0.0101,
                "monthly": 4500.00,
                "monthly_pct": 0.0373,
                "ytd": 12500.00,
                "ytd_pct": 0.1111,
            },
            "risk_metrics": {
                "var_95": -2500.00,
                "var_99": -4100.00,
                "max_drawdown": -0.085,
                "sharpe_ratio": 1.45,
                "sortino_ratio": 1.82,
                "beta": 1.05,
                "correlation_spy": 0.92,
                "volatility_30d": 0.145,
            },
        }

    async def get_portfolio(
        self,
        include_positions: bool = True,
        include_pnl: bool = True,
    ) -> dict[str, Any]:
        """
        Get complete portfolio status.

        Args:
            include_positions: Include position details
            include_pnl: Include P&L breakdown

        Returns:
            Complete portfolio status
        """
        if self.mock_mode:
            result = {
                "cash": self._portfolio_data["cash"],
                "total_value": self._portfolio_data["total_value"],
                "buying_power": self._portfolio_data["buying_power"],
                "timestamp": datetime.utcnow().isoformat(),
            }

            if include_positions:
                result["positions"] = self._portfolio_data["positions"]
                result["total_positions"] = len(self._portfolio_data["positions"])

            if include_pnl:
                result["pnl"] = self._portfolio_data["pnl"]

            return result

        raise NotImplementedError("Live portfolio not yet implemented")

    async def get_exposure(
        self,
        group_by: str = "sector",
    ) -> dict[str, Any]:
        """
        Get portfolio exposure breakdown.

        Args:
            group_by: How to group exposure (sector, asset_class, symbol)

        Returns:
            Portfolio exposure breakdown
        """
        if self.mock_mode:
            positions = self._portfolio_data["positions"]
            total_value = self._portfolio_data["total_value"]

            exposure: dict[str, float] = {}

            for pos in positions:
                key = pos.get(group_by, "Unknown")
                if key not in exposure:
                    exposure[key] = 0.0
                exposure[key] += pos["market_value"]

            # Convert to percentages
            exposure_pct = {k: {"value": v, "pct": v / total_value} for k, v in exposure.items()}

            # Calculate gross and net exposure
            long_value = sum(p["market_value"] for p in positions if p["quantity"] > 0)
            short_value = sum(abs(p["market_value"]) for p in positions if p["quantity"] < 0)

            return {
                "grouped_by": group_by,
                "exposure": exposure_pct,
                "gross_exposure": long_value + short_value,
                "net_exposure": long_value - short_value,
                "long_exposure": long_value,
                "short_exposure": short_value,
                "cash": self._portfolio_data["cash"],
                "cash_pct": self._portfolio_data["cash"] / total_value,
                "timestamp": datetime.utcnow().isoformat(),
            }

        raise NotImplementedError("Live exposure not yet implemented")

    async def get_risk_metrics(
        self,
        lookback_days: int = 30,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """
        Get portfolio risk metrics.

        Args:
            lookback_days: Days of history for calculations
            confidence_level: VaR confidence level

        Returns:
            Risk metrics including VaR, Sharpe, Sortino, max drawdown
        """
        if self.mock_mode:
            metrics = self._portfolio_data["risk_metrics"]

            # Adjust VaR based on confidence level
            var_key = "var_95" if confidence_level <= 0.95 else "var_99"

            return {
                "var": metrics[var_key],
                "var_pct": metrics[var_key] / self._portfolio_data["total_value"],
                "confidence_level": confidence_level,
                "max_drawdown": metrics["max_drawdown"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "sortino_ratio": metrics["sortino_ratio"],
                "beta": metrics["beta"],
                "correlation_spy": metrics["correlation_spy"],
                "volatility_30d": metrics["volatility_30d"],
                "lookback_days": lookback_days,
                "timestamp": datetime.utcnow().isoformat(),
            }

        raise NotImplementedError("Live risk metrics not yet implemented")

    async def get_pnl(
        self,
        period: str = "daily",
        by_symbol: bool = False,
    ) -> dict[str, Any]:
        """
        Get profit and loss breakdown.

        Args:
            period: P&L period (daily, weekly, monthly, ytd, all_time)
            by_symbol: Break down by symbol

        Returns:
            P&L breakdown for specified period
        """
        if self.mock_mode:
            pnl = self._portfolio_data["pnl"]

            result = {
                "period": period,
                "timestamp": datetime.utcnow().isoformat(),
            }

            if period == "daily":
                result["pnl"] = pnl["daily"]
                result["pnl_pct"] = pnl["daily_pct"]
            elif period == "weekly":
                result["pnl"] = pnl["weekly"]
                result["pnl_pct"] = pnl["weekly_pct"]
            elif period == "monthly":
                result["pnl"] = pnl["monthly"]
                result["pnl_pct"] = pnl["monthly_pct"]
            elif period == "ytd":
                result["pnl"] = pnl["ytd"]
                result["pnl_pct"] = pnl["ytd_pct"]
            else:  # all_time
                result["pnl"] = pnl["ytd"]  # Mock: same as ytd
                result["pnl_pct"] = pnl["ytd_pct"]

            if by_symbol:
                # Generate mock per-symbol P&L
                positions = self._portfolio_data["positions"]
                result["by_symbol"] = {
                    p["symbol"]: {
                        "pnl": p["unrealized_pnl"],
                        "pnl_pct": p["unrealized_pnl_pct"],
                    }
                    for p in positions
                }

            return result

        raise NotImplementedError("Live P&L not yet implemented")

    async def get_holdings_summary(
        self,
        sort_by: str = "value",
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Get summary of current holdings.

        Args:
            sort_by: Sort holdings by (value, pnl, pnl_pct, symbol)
            limit: Max holdings to return

        Returns:
            Holdings summary with key metrics
        """
        if self.mock_mode:
            positions = list(self._portfolio_data["positions"])

            # Sort positions
            if sort_by == "value":
                positions.sort(key=lambda x: x["market_value"], reverse=True)
            elif sort_by == "pnl":
                positions.sort(key=lambda x: x["unrealized_pnl"], reverse=True)
            elif sort_by == "pnl_pct":
                positions.sort(key=lambda x: x["unrealized_pnl_pct"], reverse=True)
            elif sort_by == "symbol":
                positions.sort(key=lambda x: x["symbol"])

            positions = positions[:limit]

            # Calculate summary stats
            total_value = sum(p["market_value"] for p in positions)
            total_pnl = sum(p["unrealized_pnl"] for p in positions)

            return {
                "holdings": positions,
                "total_holdings": len(positions),
                "total_market_value": total_value,
                "total_unrealized_pnl": total_pnl,
                "sorted_by": sort_by,
                "timestamp": datetime.utcnow().isoformat(),
            }

        raise NotImplementedError("Live holdings not yet implemented")


def create_portfolio_server(mock_mode: bool = True) -> PortfolioServer:
    """
    Create a portfolio server instance.

    Args:
        mock_mode: Whether to use mock data

    Returns:
        Configured PortfolioServer instance
    """
    config = ServerConfig(
        name="portfolio",
        version="1.0.0",
        description="Portfolio management: status, exposure, risk, P&L",
        mock_mode=mock_mode,
    )
    return PortfolioServer(config=config, mock_mode=mock_mode)


# Standalone entry point for MCP stdio mode
if __name__ == "__main__":
    import sys

    mock = "--mock" in sys.argv or "--stdio" in sys.argv
    server = create_portfolio_server(mock_mode=mock)

    if "--stdio" in sys.argv:
        asyncio.run(server.run_stdio())
    else:
        # Run server for testing
        async def main():
            await server.start()
            print(f"Portfolio server running with {len(server.list_tools())} tools")
            print("Tools:", [t["name"] for t in server.list_tools()])

            # Test get_portfolio
            result = await server.call_tool("get_portfolio", {})
            print(f"Portfolio value: ${result.data['total_value']:,.2f}")

            await server.stop()

        asyncio.run(main())
