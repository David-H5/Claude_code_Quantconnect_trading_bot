"""
Broker MCP Server Implementation

Provides order management and position tracking tools for the MCP ecosystem.
Supports paper trading mode for safe testing.

UPGRADE-015 Phase 2: Broker MCP Server

Tools:
    - get_positions: Get current portfolio positions
    - get_orders: Get open/recent orders
    - place_order: Submit new order (paper mode only)
    - cancel_order: Cancel pending order
    - get_fills: Get execution history
    - get_account_info: Get account balance and buying power

Usage:
    server = create_broker_server(mock_mode=True)
    await server.start()
    result = await server.call_tool("get_positions", {})
"""

import asyncio
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Any

from mcp.base_server import (
    BaseMCPServer,
    ServerConfig,
    ToolCategory,
    ToolSchema,
)
from mcp.schemas import (
    TradingMode,
)


logger = logging.getLogger(__name__)


class BrokerServer(BaseMCPServer):
    """
    MCP server for broker operations.

    Provides tools for order management, position tracking, and fill history.
    Supports paper trading mode for safe testing without real money.
    """

    def __init__(
        self,
        config: ServerConfig | None = None,
        mock_mode: bool = True,
        trading_mode: TradingMode = TradingMode.PAPER,
    ):
        """
        Initialize the broker server.

        Args:
            config: Server configuration
            mock_mode: Whether to use mock data (for testing)
            trading_mode: Trading mode (paper/live/backtest)
        """
        if config is None:
            config = ServerConfig(
                name="broker",
                version="1.0.0",
                description="Broker operations: orders, positions, fills",
            )

        super().__init__(config)
        self.mock_mode = mock_mode
        self.trading_mode = trading_mode

        # In-memory state for mock mode
        self._positions: dict[str, dict[str, Any]] = {}
        self._orders: dict[str, dict[str, Any]] = {}
        self._fills: list[dict[str, Any]] = []
        self._account: dict[str, Any] = {
            "cash": 100000.0,
            "buying_power": 200000.0,
            "total_value": 100000.0,
            "day_trade_count": 0,
        }

        self._register_tools()

    def _register_tools(self) -> None:
        """Register all broker tools."""

        # Get Positions Tool
        self.register_tool(
            self.get_positions,
            ToolSchema(
                name="get_positions",
                description="Get current portfolio positions with market values and P&L",
                category=ToolCategory.PORTFOLIO,
                parameters={
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Filter by symbol (optional)",
                        },
                        "include_closed": {
                            "type": "boolean",
                            "description": "Include closed positions",
                            "default": False,
                        },
                    },
                    "required": [],
                },
                returns="List of positions with quantities, costs, and P&L",
            ),
        )

        # Get Orders Tool
        self.register_tool(
            self.get_orders,
            ToolSchema(
                name="get_orders",
                description="Get open and recent orders",
                category=ToolCategory.BROKER,
                parameters={
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["open", "filled", "cancelled", "all"],
                            "description": "Filter by order status",
                            "default": "open",
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Filter by symbol (optional)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max orders to return",
                            "default": 50,
                        },
                    },
                    "required": [],
                },
                returns="List of orders with status and fill information",
            ),
        )

        # Place Order Tool
        self.register_tool(
            self.place_order,
            ToolSchema(
                name="place_order",
                description="Submit a new order (paper trading mode only)",
                category=ToolCategory.EXECUTION,
                parameters={
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Symbol to trade",
                        },
                        "quantity": {
                            "type": "integer",
                            "description": "Number of shares/contracts",
                        },
                        "side": {
                            "type": "string",
                            "enum": ["buy", "sell"],
                            "description": "Order side",
                        },
                        "order_type": {
                            "type": "string",
                            "enum": ["market", "limit", "stop", "stop_limit"],
                            "description": "Order type",
                            "default": "limit",
                        },
                        "limit_price": {
                            "type": "number",
                            "description": "Limit price (required for limit orders)",
                        },
                        "stop_price": {
                            "type": "number",
                            "description": "Stop price (required for stop orders)",
                        },
                        "time_in_force": {
                            "type": "string",
                            "enum": ["day", "gtc", "ioc", "fok"],
                            "description": "Time in force",
                            "default": "day",
                        },
                    },
                    "required": ["symbol", "quantity", "side"],
                },
                returns="Order confirmation with order ID",
                is_dangerous=True,
                requires_auth=True,
            ),
        )

        # Cancel Order Tool
        self.register_tool(
            self.cancel_order,
            ToolSchema(
                name="cancel_order",
                description="Cancel a pending order",
                category=ToolCategory.EXECUTION,
                parameters={
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "Order ID to cancel",
                        },
                    },
                    "required": ["order_id"],
                },
                returns="Cancellation confirmation",
                is_dangerous=True,
            ),
        )

        # Get Fills Tool
        self.register_tool(
            self.get_fills,
            ToolSchema(
                name="get_fills",
                description="Get execution history (fills)",
                category=ToolCategory.BROKER,
                parameters={
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "Filter by order ID (optional)",
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Filter by symbol (optional)",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD)",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max fills to return",
                            "default": 100,
                        },
                    },
                    "required": [],
                },
                returns="List of fills with prices, quantities, and timestamps",
            ),
        )

        # Get Account Info Tool
        self.register_tool(
            self.get_account_info,
            ToolSchema(
                name="get_account_info",
                description="Get account balance and buying power",
                category=ToolCategory.PORTFOLIO,
                parameters={
                    "properties": {},
                    "required": [],
                },
                returns="Account information including cash, buying power, and total value",
            ),
        )

    async def initialize(self) -> None:
        """Initialize the broker server."""
        logger.info(f"Initializing broker server in {self.trading_mode.value} mode")

        if self.mock_mode:
            self._initialize_mock_data()

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up broker server")
        # In a real implementation, close broker connections here

    def _initialize_mock_data(self) -> None:
        """Initialize mock positions and orders for testing."""
        # Mock positions
        self._positions = {
            "SPY": {
                "symbol": "SPY",
                "quantity": 100,
                "average_cost": 445.50,
                "current_price": 450.25,
                "market_value": 45025.00,
                "unrealized_pnl": 475.00,
                "unrealized_pnl_pct": 0.0107,
                "realized_pnl": 0.0,
                "side": "long",
                "opened_at": (datetime.utcnow() - timedelta(days=5)).isoformat(),
            },
            "AAPL": {
                "symbol": "AAPL",
                "quantity": 50,
                "average_cost": 172.30,
                "current_price": 175.80,
                "market_value": 8790.00,
                "unrealized_pnl": 175.00,
                "unrealized_pnl_pct": 0.0203,
                "realized_pnl": 125.50,
                "side": "long",
                "opened_at": (datetime.utcnow() - timedelta(days=10)).isoformat(),
            },
        }

        # Mock open orders
        self._orders = {
            "ORD-001": {
                "order_id": "ORD-001",
                "symbol": "MSFT",
                "quantity": 25,
                "side": "buy",
                "order_type": "limit",
                "limit_price": 375.00,
                "stop_price": None,
                "status": "open",
                "filled_quantity": 0,
                "average_fill_price": None,
                "time_in_force": "day",
                "submitted_at": datetime.utcnow().isoformat(),
                "filled_at": None,
            },
        }

        # Mock fills
        self._fills = [
            {
                "fill_id": "FILL-001",
                "order_id": "ORD-HIST-001",
                "symbol": "SPY",
                "quantity": 100,
                "price": 445.50,
                "side": "buy",
                "commission": 0.0,
                "fill_time": (datetime.utcnow() - timedelta(days=5)).isoformat(),
            },
            {
                "fill_id": "FILL-002",
                "order_id": "ORD-HIST-002",
                "symbol": "AAPL",
                "quantity": 50,
                "price": 172.30,
                "side": "buy",
                "commission": 0.0,
                "fill_time": (datetime.utcnow() - timedelta(days=10)).isoformat(),
            },
        ]

        # Update account with positions
        total_market_value = sum(p["market_value"] for p in self._positions.values())
        self._account["total_value"] = self._account["cash"] + total_market_value

    async def get_positions(
        self,
        symbol: str | None = None,
        include_closed: bool = False,
    ) -> dict[str, Any]:
        """
        Get current portfolio positions.

        Args:
            symbol: Filter by symbol (optional)
            include_closed: Include closed positions

        Returns:
            Dictionary with positions list and summary
        """
        if self.mock_mode:
            positions = list(self._positions.values())

            if symbol:
                symbol = symbol.upper()
                positions = [p for p in positions if p["symbol"] == symbol]

            if not include_closed:
                positions = [p for p in positions if p["quantity"] != 0]

            total_value = sum(p["market_value"] for p in positions)
            total_pnl = sum(p["unrealized_pnl"] for p in positions)

            return {
                "positions": positions,
                "total_positions": len(positions),
                "total_market_value": total_value,
                "total_unrealized_pnl": total_pnl,
                "timestamp": datetime.utcnow().isoformat(),
            }

        raise NotImplementedError("Live broker not yet implemented")

    async def get_orders(
        self,
        status: str = "open",
        symbol: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Get open and recent orders.

        Args:
            status: Filter by status (open, filled, cancelled, all)
            symbol: Filter by symbol (optional)
            limit: Max orders to return

        Returns:
            Dictionary with orders list
        """
        if self.mock_mode:
            orders = list(self._orders.values())

            if status != "all":
                orders = [o for o in orders if o["status"] == status]

            if symbol:
                symbol = symbol.upper()
                orders = [o for o in orders if o["symbol"] == symbol]

            orders = orders[:limit]

            return {
                "orders": orders,
                "total_orders": len(orders),
                "filter_status": status,
                "timestamp": datetime.utcnow().isoformat(),
            }

        raise NotImplementedError("Live broker not yet implemented")

    async def place_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str = "limit",
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        """
        Submit a new order (paper trading mode only).

        Args:
            symbol: Symbol to trade
            quantity: Number of shares/contracts
            side: Order side (buy/sell)
            order_type: Order type (market/limit/stop/stop_limit)
            limit_price: Limit price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            time_in_force: Time in force (day/gtc/ioc/fok)

        Returns:
            Order confirmation with order ID
        """
        # Safety check - only allow paper trading
        if self.trading_mode == TradingMode.LIVE:
            return {
                "success": False,
                "error": "Live trading not permitted via MCP tools",
                "error_code": "LIVE_TRADING_BLOCKED",
            }

        # Validate inputs
        symbol = symbol.upper()
        side = side.lower()
        order_type = order_type.lower()
        time_in_force = time_in_force.lower()

        # Validate order type requirements
        if order_type in ("limit", "stop_limit") and limit_price is None:
            return {
                "success": False,
                "error": "Limit price required for limit/stop_limit orders",
                "error_code": "MISSING_LIMIT_PRICE",
            }

        if order_type in ("stop", "stop_limit") and stop_price is None:
            return {
                "success": False,
                "error": "Stop price required for stop/stop_limit orders",
                "error_code": "MISSING_STOP_PRICE",
            }

        # Validate quantity
        if quantity <= 0:
            return {
                "success": False,
                "error": "Quantity must be positive",
                "error_code": "INVALID_QUANTITY",
            }

        # Check buying power (simplified)
        if side == "buy":
            price = limit_price or stop_price or 100.0  # Default for market orders
            required_value = price * quantity
            if required_value > self._account["buying_power"]:
                return {
                    "success": False,
                    "error": f"Insufficient buying power. Required: ${required_value:.2f}, Available: ${self._account['buying_power']:.2f}",
                    "error_code": "INSUFFICIENT_BUYING_POWER",
                }

        # Create order
        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "order_type": order_type,
            "limit_price": limit_price,
            "stop_price": stop_price,
            "status": "submitted",
            "filled_quantity": 0,
            "average_fill_price": None,
            "time_in_force": time_in_force,
            "submitted_at": datetime.utcnow().isoformat(),
            "filled_at": None,
            "trading_mode": self.trading_mode.value,
        }

        if self.mock_mode:
            # Simulate immediate fill for market orders
            if order_type == "market":
                await self._simulate_fill(order)
            else:
                order["status"] = "open"
                self._orders[order_id] = order

        logger.info(f"Order submitted: {order_id} {side} {quantity} {symbol}")

        return {
            "success": True,
            "order": order,
            "message": f"Order {order_id} submitted successfully",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def _simulate_fill(self, order: dict[str, Any]) -> None:
        """Simulate order fill for mock mode."""
        # Generate mock fill price
        base_price = order.get("limit_price") or order.get("stop_price") or 100.0
        slippage = random.uniform(-0.001, 0.002)  # -0.1% to +0.2% slippage
        fill_price = base_price * (1 + slippage)

        order["status"] = "filled"
        order["filled_quantity"] = order["quantity"]
        order["average_fill_price"] = round(fill_price, 2)
        order["filled_at"] = datetime.utcnow().isoformat()

        # Create fill record
        fill = {
            "fill_id": f"FILL-{uuid.uuid4().hex[:8].upper()}",
            "order_id": order["order_id"],
            "symbol": order["symbol"],
            "quantity": order["quantity"],
            "price": order["average_fill_price"],
            "side": order["side"],
            "commission": 0.0,
            "fill_time": order["filled_at"],
        }
        self._fills.append(fill)

        # Update positions
        await self._update_position(order)

        self._orders[order["order_id"]] = order

    async def _update_position(self, order: dict[str, Any]) -> None:
        """Update position after fill."""
        symbol = order["symbol"]
        quantity = order["filled_quantity"]
        price = order["average_fill_price"]
        side = order["side"]

        if symbol in self._positions:
            pos = self._positions[symbol]
            if side == "buy":
                # Add to position
                total_cost = pos["average_cost"] * pos["quantity"] + price * quantity
                pos["quantity"] += quantity
                pos["average_cost"] = total_cost / pos["quantity"] if pos["quantity"] > 0 else 0
            else:
                # Reduce position
                pos["quantity"] -= quantity
                if pos["quantity"] <= 0:
                    del self._positions[symbol]
                    return

            pos["current_price"] = price
            pos["market_value"] = pos["quantity"] * price
            pos["unrealized_pnl"] = (price - pos["average_cost"]) * pos["quantity"]
            pos["unrealized_pnl_pct"] = (
                pos["unrealized_pnl"] / (pos["average_cost"] * pos["quantity"]) if pos["quantity"] > 0 else 0
            )
        else:
            # New position
            self._positions[symbol] = {
                "symbol": symbol,
                "quantity": quantity if side == "buy" else -quantity,
                "average_cost": price,
                "current_price": price,
                "market_value": quantity * price,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
                "realized_pnl": 0.0,
                "side": "long" if side == "buy" else "short",
                "opened_at": datetime.utcnow().isoformat(),
            }

        # Update account
        if side == "buy":
            self._account["cash"] -= price * quantity
        else:
            self._account["cash"] += price * quantity

        total_market_value = sum(p["market_value"] for p in self._positions.values())
        self._account["total_value"] = self._account["cash"] + total_market_value

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """
        Cancel a pending order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancellation confirmation
        """
        if self.mock_mode:
            if order_id not in self._orders:
                return {
                    "success": False,
                    "error": f"Order {order_id} not found",
                    "error_code": "ORDER_NOT_FOUND",
                }

            order = self._orders[order_id]
            if order["status"] not in ("open", "submitted"):
                return {
                    "success": False,
                    "error": f"Cannot cancel order in {order['status']} status",
                    "error_code": "INVALID_ORDER_STATUS",
                }

            order["status"] = "cancelled"
            order["cancelled_at"] = datetime.utcnow().isoformat()

            logger.info(f"Order cancelled: {order_id}")

            return {
                "success": True,
                "order_id": order_id,
                "message": f"Order {order_id} cancelled successfully",
                "timestamp": datetime.utcnow().isoformat(),
            }

        raise NotImplementedError("Live broker not yet implemented")

    async def get_fills(
        self,
        order_id: str | None = None,
        symbol: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Get execution history (fills).

        Args:
            order_id: Filter by order ID (optional)
            symbol: Filter by symbol (optional)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Max fills to return

        Returns:
            Dictionary with fills list
        """
        if self.mock_mode:
            fills = list(self._fills)

            if order_id:
                fills = [f for f in fills if f["order_id"] == order_id]

            if symbol:
                symbol = symbol.upper()
                fills = [f for f in fills if f["symbol"] == symbol]

            # Date filtering (simplified)
            if start_date:
                fills = [f for f in fills if f["fill_time"] >= start_date]
            if end_date:
                fills = [f for f in fills if f["fill_time"] <= end_date]

            fills = fills[:limit]

            total_volume = sum(f["quantity"] * f["price"] for f in fills)
            total_commission = sum(f["commission"] for f in fills)

            return {
                "fills": fills,
                "total_fills": len(fills),
                "total_volume": total_volume,
                "total_commission": total_commission,
                "timestamp": datetime.utcnow().isoformat(),
            }

        raise NotImplementedError("Live broker not yet implemented")

    async def get_account_info(self) -> dict[str, Any]:
        """
        Get account balance and buying power.

        Returns:
            Account information including cash, buying power, and total value
        """
        if self.mock_mode:
            return {
                "cash": self._account["cash"],
                "buying_power": self._account["buying_power"],
                "total_value": self._account["total_value"],
                "day_trade_count": self._account["day_trade_count"],
                "trading_mode": self.trading_mode.value,
                "margin_used": 0.0,
                "margin_available": self._account["buying_power"],
                "timestamp": datetime.utcnow().isoformat(),
            }

        raise NotImplementedError("Live broker not yet implemented")


def create_broker_server(
    mock_mode: bool = True,
    trading_mode: TradingMode = TradingMode.PAPER,
) -> BrokerServer:
    """
    Create a broker server instance.

    Args:
        mock_mode: Whether to use mock data
        trading_mode: Trading mode (paper/live/backtest)

    Returns:
        Configured BrokerServer instance
    """
    config = ServerConfig(
        name="broker",
        version="1.0.0",
        description="Broker operations: orders, positions, fills",
        mock_mode=mock_mode,
    )
    return BrokerServer(config=config, mock_mode=mock_mode, trading_mode=trading_mode)


# Standalone entry point for MCP stdio mode
if __name__ == "__main__":
    import sys

    mock = "--mock" in sys.argv or "--stdio" in sys.argv
    server = create_broker_server(mock_mode=mock)

    if "--stdio" in sys.argv:
        asyncio.run(server.run_stdio())
    else:
        # Run server for testing
        async def main():
            await server.start()
            print(f"Broker server running with {len(server.list_tools())} tools")
            print("Tools:", [t["name"] for t in server.list_tools()])

            # Test get_positions
            result = await server.call_tool("get_positions", {})
            print(f"Positions: {result.data}")

            await server.stop()

        asyncio.run(main())
