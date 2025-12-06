"""
Backtest MCP Server Implementation

Provides backtesting tools for the MCP ecosystem.
Includes backtest execution, result parsing, and performance analysis.

UPGRADE-015 Phase 3: Portfolio & Backtest MCP

Tools:
    - run_backtest: Execute a backtest with specified parameters
    - get_backtest_status: Get status of running/completed backtest
    - get_backtest_results: Get detailed backtest results
    - parse_backtest_report: Parse a backtest report file
    - list_backtests: List recent backtests

Usage:
    server = create_backtest_server(mock_mode=True)
    await server.start()
    result = await server.call_tool("run_backtest", {"algorithm_id": "my_algo"})
"""

import asyncio
import logging
import random
import uuid
from datetime import date, datetime, timedelta
from typing import Any

from mcp.base_server import (
    BaseMCPServer,
    ServerConfig,
    ToolCategory,
    ToolSchema,
)


logger = logging.getLogger(__name__)


class BacktestServer(BaseMCPServer):
    """
    MCP server for backtesting operations.

    Provides tools for running backtests, parsing results,
    and analyzing performance metrics.
    """

    def __init__(
        self,
        config: ServerConfig | None = None,
        mock_mode: bool = True,
    ):
        """
        Initialize the backtest server.

        Args:
            config: Server configuration
            mock_mode: Whether to use mock data (for testing)
        """
        if config is None:
            config = ServerConfig(
                name="backtest",
                version="1.0.0",
                description="Backtesting: run, status, results, analysis",
            )

        super().__init__(config)
        self.mock_mode = mock_mode

        # In-memory backtest storage
        self._backtests: dict[str, dict[str, Any]] = {}

        self._register_tools()

    def _register_tools(self) -> None:
        """Register all backtest tools."""

        # Run Backtest Tool
        self.register_tool(
            self.run_backtest,
            ToolSchema(
                name="run_backtest",
                description="Execute a backtest with specified algorithm and parameters",
                category=ToolCategory.BACKTEST,
                parameters={
                    "properties": {
                        "algorithm_id": {
                            "type": "string",
                            "description": "Algorithm identifier or file path",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD)",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD)",
                        },
                        "initial_cash": {
                            "type": "number",
                            "description": "Initial cash amount",
                            "default": 100000,
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Algorithm parameters",
                            "default": {},
                        },
                    },
                    "required": ["algorithm_id"],
                },
                returns="Backtest ID and initial status",
                is_dangerous=True,
            ),
        )

        # Get Backtest Status Tool
        self.register_tool(
            self.get_backtest_status,
            ToolSchema(
                name="get_backtest_status",
                description="Get status of a running or completed backtest",
                category=ToolCategory.BACKTEST,
                parameters={
                    "properties": {
                        "backtest_id": {
                            "type": "string",
                            "description": "Backtest ID",
                        },
                    },
                    "required": ["backtest_id"],
                },
                returns="Backtest status and progress",
            ),
        )

        # Get Backtest Results Tool
        self.register_tool(
            self.get_backtest_results,
            ToolSchema(
                name="get_backtest_results",
                description="Get detailed results from a completed backtest",
                category=ToolCategory.BACKTEST,
                parameters={
                    "properties": {
                        "backtest_id": {
                            "type": "string",
                            "description": "Backtest ID",
                        },
                        "include_trades": {
                            "type": "boolean",
                            "description": "Include trade list",
                            "default": False,
                        },
                        "include_equity_curve": {
                            "type": "boolean",
                            "description": "Include equity curve data",
                            "default": False,
                        },
                    },
                    "required": ["backtest_id"],
                },
                returns="Detailed backtest results with metrics",
            ),
        )

        # Parse Backtest Report Tool
        self.register_tool(
            self.parse_backtest_report,
            ToolSchema(
                name="parse_backtest_report",
                description="Parse a backtest report file and extract key metrics",
                category=ToolCategory.BACKTEST,
                parameters={
                    "properties": {
                        "report_path": {
                            "type": "string",
                            "description": "Path to backtest report file",
                        },
                    },
                    "required": ["report_path"],
                },
                returns="Parsed metrics from backtest report",
            ),
        )

        # List Backtests Tool
        self.register_tool(
            self.list_backtests,
            ToolSchema(
                name="list_backtests",
                description="List recent backtests with summary",
                category=ToolCategory.BACKTEST,
                parameters={
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Max backtests to return",
                            "default": 10,
                        },
                        "status": {
                            "type": "string",
                            "enum": ["all", "running", "completed", "failed"],
                            "description": "Filter by status",
                            "default": "all",
                        },
                    },
                    "required": [],
                },
                returns="List of backtests with summary",
            ),
        )

    async def initialize(self) -> None:
        """Initialize the backtest server."""
        logger.info("Initializing backtest server")

        if self.mock_mode:
            self._initialize_mock_data()

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up backtest server")

    def _initialize_mock_data(self) -> None:
        """Initialize mock backtest data."""
        # Create a completed mock backtest
        backtest_id = "BT-MOCK-001"
        self._backtests[backtest_id] = self._generate_mock_backtest(
            backtest_id=backtest_id,
            algorithm_id="hybrid_options_bot",
            start_date="2024-01-01",
            end_date="2024-12-01",
            status="completed",
        )

    def _generate_mock_backtest(
        self,
        backtest_id: str,
        algorithm_id: str,
        start_date: str,
        end_date: str,
        initial_cash: float = 100000,
        status: str = "running",
    ) -> dict[str, Any]:
        """Generate mock backtest data."""
        # Generate random but realistic metrics
        total_return = random.uniform(-0.1, 0.5)
        sharpe = random.uniform(0.5, 2.5)
        max_dd = random.uniform(-0.25, -0.05)
        win_rate = random.uniform(0.4, 0.7)
        total_trades = random.randint(50, 500)

        return {
            "backtest_id": backtest_id,
            "algorithm_id": algorithm_id,
            "start_date": start_date,
            "end_date": end_date,
            "initial_cash": initial_cash,
            "final_value": initial_cash * (1 + total_return),
            "status": status,
            "progress": 100 if status == "completed" else random.randint(10, 90),
            "started_at": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
            "completed_at": datetime.utcnow().isoformat() if status == "completed" else None,
            "metrics": {
                "total_return": total_return,
                "annual_return": total_return * 1.2,  # Simplified annualization
                "sharpe_ratio": sharpe,
                "sortino_ratio": sharpe * 1.2,
                "max_drawdown": max_dd,
                "win_rate": win_rate,
                "profit_factor": 1 + (win_rate - 0.5) * 2,
                "total_trades": total_trades,
                "winning_trades": int(total_trades * win_rate),
                "losing_trades": total_trades - int(total_trades * win_rate),
                "average_win": random.uniform(100, 500),
                "average_loss": random.uniform(-300, -50),
                "largest_win": random.uniform(500, 2000),
                "largest_loss": random.uniform(-1500, -200),
                "avg_trade_duration": f"{random.randint(1, 10)} days",
            },
            "equity_curve": self._generate_equity_curve(initial_cash, total_return, 100),
            "trades": self._generate_mock_trades(total_trades, start_date, end_date),
        }

    def _generate_equity_curve(
        self,
        initial_cash: float,
        total_return: float,
        points: int,
    ) -> list[dict[str, Any]]:
        """Generate mock equity curve."""
        curve = []
        value = initial_cash
        step_return = total_return / points

        for i in range(points):
            # Add some noise
            noise = random.uniform(-0.01, 0.01)
            value *= 1 + step_return + noise
            curve.append(
                {
                    "index": i,
                    "value": round(value, 2),
                    "return_pct": round((value / initial_cash - 1) * 100, 2),
                }
            )

        return curve

    def _generate_mock_trades(
        self,
        count: int,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        """Generate mock trade list."""
        symbols = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
        trades = []

        for i in range(min(count, 20)):  # Limit to 20 for mock
            is_win = random.random() > 0.4
            pnl = random.uniform(50, 500) if is_win else random.uniform(-400, -50)

            trades.append(
                {
                    "trade_id": f"TRD-{i+1:04d}",
                    "symbol": random.choice(symbols),
                    "side": random.choice(["buy", "sell"]),
                    "quantity": random.randint(10, 100),
                    "entry_price": random.uniform(100, 500),
                    "exit_price": random.uniform(100, 500),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl / 1000 * 100, 2),
                    "entry_time": start_date,
                    "exit_time": end_date,
                }
            )

        return trades

    async def run_backtest(
        self,
        algorithm_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        initial_cash: float = 100000,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a backtest.

        Args:
            algorithm_id: Algorithm identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_cash: Initial cash amount
            parameters: Algorithm parameters

        Returns:
            Backtest ID and initial status
        """
        # Default dates
        if start_date is None:
            start_date = (date.today() - timedelta(days=365)).isoformat()
        if end_date is None:
            end_date = date.today().isoformat()

        backtest_id = f"BT-{uuid.uuid4().hex[:8].upper()}"

        if self.mock_mode:
            # Create mock backtest
            self._backtests[backtest_id] = self._generate_mock_backtest(
                backtest_id=backtest_id,
                algorithm_id=algorithm_id,
                start_date=start_date,
                end_date=end_date,
                initial_cash=initial_cash,
                status="completed",  # Instant completion in mock
            )

            return {
                "backtest_id": backtest_id,
                "algorithm_id": algorithm_id,
                "status": "completed",
                "message": f"Backtest {backtest_id} completed (mock mode)",
                "start_date": start_date,
                "end_date": end_date,
                "initial_cash": initial_cash,
                "timestamp": datetime.utcnow().isoformat(),
            }

        raise NotImplementedError("Live backtesting not yet implemented")

    async def get_backtest_status(
        self,
        backtest_id: str,
    ) -> dict[str, Any]:
        """
        Get status of a backtest.

        Args:
            backtest_id: Backtest ID

        Returns:
            Backtest status and progress
        """
        if self.mock_mode:
            if backtest_id not in self._backtests:
                return {
                    "success": False,
                    "error": f"Backtest {backtest_id} not found",
                    "error_code": "BACKTEST_NOT_FOUND",
                }

            bt = self._backtests[backtest_id]
            return {
                "backtest_id": backtest_id,
                "status": bt["status"],
                "progress": bt["progress"],
                "algorithm_id": bt["algorithm_id"],
                "started_at": bt["started_at"],
                "completed_at": bt["completed_at"],
                "timestamp": datetime.utcnow().isoformat(),
            }

        raise NotImplementedError("Live backtest status not yet implemented")

    async def get_backtest_results(
        self,
        backtest_id: str,
        include_trades: bool = False,
        include_equity_curve: bool = False,
    ) -> dict[str, Any]:
        """
        Get detailed backtest results.

        Args:
            backtest_id: Backtest ID
            include_trades: Include trade list
            include_equity_curve: Include equity curve data

        Returns:
            Detailed backtest results with metrics
        """
        if self.mock_mode:
            if backtest_id not in self._backtests:
                return {
                    "success": False,
                    "error": f"Backtest {backtest_id} not found",
                    "error_code": "BACKTEST_NOT_FOUND",
                }

            bt = self._backtests[backtest_id]

            if bt["status"] != "completed":
                return {
                    "success": False,
                    "error": f"Backtest {backtest_id} not completed (status: {bt['status']})",
                    "error_code": "BACKTEST_NOT_COMPLETE",
                }

            result = {
                "backtest_id": backtest_id,
                "algorithm_id": bt["algorithm_id"],
                "start_date": bt["start_date"],
                "end_date": bt["end_date"],
                "initial_cash": bt["initial_cash"],
                "final_value": bt["final_value"],
                "metrics": bt["metrics"],
                "timestamp": datetime.utcnow().isoformat(),
            }

            if include_trades:
                result["trades"] = bt["trades"]
                result["total_trades"] = len(bt["trades"])

            if include_equity_curve:
                result["equity_curve"] = bt["equity_curve"]

            return result

        raise NotImplementedError("Live backtest results not yet implemented")

    async def parse_backtest_report(
        self,
        report_path: str,
    ) -> dict[str, Any]:
        """
        Parse a backtest report file.

        Args:
            report_path: Path to backtest report file

        Returns:
            Parsed metrics from backtest report
        """
        if self.mock_mode:
            # Return mock parsed data
            return {
                "report_path": report_path,
                "parsed": True,
                "metrics": {
                    "total_return": 0.25,
                    "sharpe_ratio": 1.8,
                    "max_drawdown": -0.12,
                    "win_rate": 0.58,
                    "total_trades": 150,
                },
                "warnings": [],
                "timestamp": datetime.utcnow().isoformat(),
            }

        # In real implementation, would parse actual file
        raise NotImplementedError("Live report parsing not yet implemented")

    async def list_backtests(
        self,
        limit: int = 10,
        status: str = "all",
    ) -> dict[str, Any]:
        """
        List recent backtests.

        Args:
            limit: Max backtests to return
            status: Filter by status

        Returns:
            List of backtests with summary
        """
        if self.mock_mode:
            backtests = list(self._backtests.values())

            if status != "all":
                backtests = [b for b in backtests if b["status"] == status]

            # Sort by started_at (most recent first)
            backtests.sort(key=lambda x: x["started_at"], reverse=True)
            backtests = backtests[:limit]

            # Create summaries
            summaries = []
            for bt in backtests:
                summaries.append(
                    {
                        "backtest_id": bt["backtest_id"],
                        "algorithm_id": bt["algorithm_id"],
                        "status": bt["status"],
                        "start_date": bt["start_date"],
                        "end_date": bt["end_date"],
                        "total_return": bt["metrics"]["total_return"] if bt["status"] == "completed" else None,
                        "sharpe_ratio": bt["metrics"]["sharpe_ratio"] if bt["status"] == "completed" else None,
                        "started_at": bt["started_at"],
                    }
                )

            return {
                "backtests": summaries,
                "total": len(summaries),
                "filter_status": status,
                "timestamp": datetime.utcnow().isoformat(),
            }

        raise NotImplementedError("Live backtest listing not yet implemented")


def create_backtest_server(mock_mode: bool = True) -> BacktestServer:
    """
    Create a backtest server instance.

    Args:
        mock_mode: Whether to use mock data

    Returns:
        Configured BacktestServer instance
    """
    config = ServerConfig(
        name="backtest",
        version="1.0.0",
        description="Backtesting: run, status, results, analysis",
        mock_mode=mock_mode,
    )
    return BacktestServer(config=config, mock_mode=mock_mode)


# Standalone entry point for MCP stdio mode
if __name__ == "__main__":
    import sys

    mock = "--mock" in sys.argv or "--stdio" in sys.argv
    server = create_backtest_server(mock_mode=mock)

    if "--stdio" in sys.argv:
        asyncio.run(server.run_stdio())
    else:
        # Run server for testing
        async def main():
            await server.start()
            print(f"Backtest server running with {len(server.list_tools())} tools")
            print("Tools:", [t["name"] for t in server.list_tools()])

            # Test run_backtest
            result = await server.call_tool(
                "run_backtest",
                {"algorithm_id": "test_algo"},
            )
            print(f"Backtest started: {result.data['backtest_id']}")

            await server.stop()

        asyncio.run(main())
