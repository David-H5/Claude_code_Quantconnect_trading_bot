"""
Market Data MCP Server

Specialized MCP server for market data operations including quotes,
option chains, Greeks, and historical data.

UPGRADE-015 Phase 1: MCP Server Foundation

Usage:
    # Start server in stdio mode
    python -m mcp.market_data_server --stdio

    # Use programmatically
    from mcp.market_data_server import create_market_data_server

    server = create_market_data_server()
    await server.start()
    result = await server.call_tool("get_quote", {"symbol": "AAPL"})
"""

import argparse
import asyncio
import json
import logging
import math
import random
from datetime import date, datetime, timedelta
from typing import Any

from mcp.base_server import (
    BaseMCPServer,
    ServerConfig,
    ToolCategory,
    ToolSchema,
)


logger = logging.getLogger(__name__)


class MarketDataServer(BaseMCPServer):
    """
    MCP Server for market data operations.

    Provides tools for:
    - Real-time quotes
    - Option chains with Greeks
    - Historical OHLCV data
    - Implied volatility data
    """

    def __init__(
        self,
        config: ServerConfig | None = None,
        mock_mode: bool = True,
        data_provider: Any | None = None,
    ):
        """
        Initialize market data server.

        Args:
            config: Server configuration
            mock_mode: Use mock data (True) or real provider
            data_provider: Optional real data provider
        """
        if config is None:
            config = ServerConfig(
                name="market-data",
                version="1.0.0",
                description="Market data MCP server for quotes, options, and historical data",
                mock_mode=mock_mode,
            )

        super().__init__(config)

        self.mock_mode = mock_mode or config.mock_mode
        self.data_provider = data_provider

        # Register all tools
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all market data tools."""

        # get_quote
        self.register_tool(
            self.get_quote,
            ToolSchema(
                name="get_quote",
                description="Get current quote for a stock or ETF including price, volume, and optionally Greeks",
                category=ToolCategory.MARKET_DATA,
                parameters={
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Ticker symbol (e.g., AAPL, SPY)",
                        },
                        "include_greeks": {
                            "type": "boolean",
                            "description": "Include options Greeks if available",
                            "default": False,
                        },
                        "include_volume": {
                            "type": "boolean",
                            "description": "Include volume data",
                            "default": True,
                        },
                    },
                    "required": ["symbol"],
                },
                returns="Quote data with price, volume, and optional Greeks",
            ),
        )

        # get_option_chain
        self.register_tool(
            self.get_option_chain,
            ToolSchema(
                name="get_option_chain",
                description="Get option chain for an underlying with Greeks",
                category=ToolCategory.MARKET_DATA,
                parameters={
                    "properties": {
                        "underlying": {
                            "type": "string",
                            "description": "Underlying ticker symbol",
                        },
                        "expiry_min_days": {
                            "type": "integer",
                            "description": "Minimum days to expiration",
                            "default": 0,
                        },
                        "expiry_max_days": {
                            "type": "integer",
                            "description": "Maximum days to expiration",
                            "default": 90,
                        },
                        "strike_range_pct": {
                            "type": "number",
                            "description": "Strike range as percentage from ATM (0.10 = 10%)",
                            "default": 0.10,
                        },
                        "option_type": {
                            "type": "string",
                            "enum": ["call", "put"],
                            "description": "Filter by option type",
                        },
                    },
                    "required": ["underlying"],
                },
                returns="Option chain with contracts and Greeks",
            ),
        )

        # get_greeks
        self.register_tool(
            self.get_greeks,
            ToolSchema(
                name="get_greeks",
                description="Get Greeks for a specific option contract",
                category=ToolCategory.MARKET_DATA,
                parameters={
                    "properties": {
                        "contract_symbol": {
                            "type": "string",
                            "description": "Option contract symbol",
                        },
                    },
                    "required": ["contract_symbol"],
                },
                returns="Greeks (delta, gamma, theta, vega, rho) and IV",
            ),
        )

        # get_historical
        self.register_tool(
            self.get_historical,
            ToolSchema(
                name="get_historical",
                description="Get historical OHLCV data for a symbol",
                category=ToolCategory.MARKET_DATA,
                parameters={
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Ticker symbol",
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days of history",
                            "default": 30,
                        },
                        "resolution": {
                            "type": "string",
                            "enum": ["minute", "hour", "daily"],
                            "description": "Data resolution",
                            "default": "daily",
                        },
                    },
                    "required": ["symbol"],
                },
                returns="Historical OHLCV bars",
            ),
        )

        # get_iv_surface
        self.register_tool(
            self.get_iv_surface,
            ToolSchema(
                name="get_iv_surface",
                description="Get implied volatility surface for an underlying",
                category=ToolCategory.MARKET_DATA,
                parameters={
                    "properties": {
                        "underlying": {
                            "type": "string",
                            "description": "Underlying ticker symbol",
                        },
                    },
                    "required": ["underlying"],
                },
                returns="IV surface data (strike x expiry x IV)",
            ),
        )

        # get_market_status
        self.register_tool(
            self.get_market_status,
            ToolSchema(
                name="get_market_status",
                description="Get current market status (open/closed/pre/post)",
                category=ToolCategory.MARKET_DATA,
                parameters={
                    "properties": {},
                    "required": [],
                },
                returns="Market status information",
            ),
        )

    async def initialize(self) -> None:
        """Initialize server resources."""
        logger.info(f"Initializing market data server (mock_mode={self.mock_mode})")

        if not self.mock_mode and self.data_provider:
            # Initialize real data provider connection
            logger.info("Connecting to data provider...")
            # await self.data_provider.connect()

        logger.info("Market data server initialized")

    async def cleanup(self) -> None:
        """Clean up server resources."""
        logger.info("Cleaning up market data server")

        if not self.mock_mode and self.data_provider:
            # Close data provider connection
            # await self.data_provider.disconnect()
            pass

    # =========================================================================
    # Tool Implementations
    # =========================================================================

    async def get_quote(
        self,
        symbol: str,
        include_greeks: bool = False,
        include_volume: bool = True,
    ) -> dict[str, Any]:
        """Get current quote for a symbol."""
        symbol = symbol.upper().strip()

        if self.mock_mode:
            return self._mock_quote(symbol, include_greeks, include_volume)

        # Real implementation would call data provider
        raise NotImplementedError("Real data provider not implemented")

    async def get_option_chain(
        self,
        underlying: str,
        expiry_min_days: int = 0,
        expiry_max_days: int = 90,
        strike_range_pct: float = 0.10,
        option_type: str | None = None,
    ) -> dict[str, Any]:
        """Get option chain with Greeks."""
        underlying = underlying.upper().strip()

        if self.mock_mode:
            return self._mock_option_chain(
                underlying,
                expiry_min_days,
                expiry_max_days,
                strike_range_pct,
                option_type,
            )

        raise NotImplementedError("Real data provider not implemented")

    async def get_greeks(self, contract_symbol: str) -> dict[str, Any]:
        """Get Greeks for a specific option contract."""
        if self.mock_mode:
            return self._mock_greeks(contract_symbol)

        raise NotImplementedError("Real data provider not implemented")

    async def get_historical(
        self,
        symbol: str,
        days: int = 30,
        resolution: str = "daily",
    ) -> dict[str, Any]:
        """Get historical OHLCV data."""
        symbol = symbol.upper().strip()

        if self.mock_mode:
            return self._mock_historical(symbol, days, resolution)

        raise NotImplementedError("Real data provider not implemented")

    async def get_iv_surface(self, underlying: str) -> dict[str, Any]:
        """Get implied volatility surface."""
        underlying = underlying.upper().strip()

        if self.mock_mode:
            return self._mock_iv_surface(underlying)

        raise NotImplementedError("Real data provider not implemented")

    async def get_market_status(self) -> dict[str, Any]:
        """Get current market status."""
        now = datetime.utcnow()
        hour = now.hour

        # Simple mock logic for market hours (9:30 AM - 4:00 PM ET)
        # This is simplified and doesn't account for holidays
        if 14 <= hour < 21:  # Roughly 9:30 AM - 4:00 PM ET in UTC
            status = "open"
        elif 13 <= hour < 14:  # Pre-market
            status = "pre_market"
        elif 21 <= hour < 22:  # After hours
            status = "after_hours"
        else:
            status = "closed"

        return {
            "status": status,
            "timestamp": now.isoformat(),
            "next_open": "2025-12-05T14:30:00Z",  # Mock
            "next_close": "2025-12-05T21:00:00Z",  # Mock
            "is_trading_day": now.weekday() < 5,
        }

    # =========================================================================
    # Mock Data Generators
    # =========================================================================

    def _mock_quote(
        self,
        symbol: str,
        include_greeks: bool,
        include_volume: bool,
    ) -> dict[str, Any]:
        """Generate mock quote data."""
        # Base prices for common symbols
        base_prices = {
            "SPY": 450.0,
            "AAPL": 175.0,
            "MSFT": 380.0,
            "GOOGL": 140.0,
            "AMZN": 180.0,
            "NVDA": 480.0,
            "TSLA": 250.0,
        }

        base = base_prices.get(symbol, 100.0)
        noise = random.uniform(-0.02, 0.02)
        last = round(base * (1 + noise), 2)

        result = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "price": {
                "last": last,
                "bid": round(last - 0.05, 2),
                "ask": round(last + 0.05, 2),
                "open": round(base * 0.998, 2),
                "high": round(last * 1.01, 2),
                "low": round(last * 0.99, 2),
                "close_previous": round(base, 2),
                "change": round(last - base, 2),
                "change_pct": round((last - base) / base * 100, 2),
            },
        }

        if include_volume:
            result["volume"] = {
                "current": random.randint(5_000_000, 20_000_000),
                "average_30d": 10_000_000,
                "relative": round(random.uniform(0.5, 1.5), 2),
            }

        if include_greeks:
            result["greeks"] = {
                "delta": round(random.uniform(0.3, 0.7), 3),
                "gamma": round(random.uniform(0.01, 0.05), 4),
                "theta": round(random.uniform(-0.10, -0.01), 4),
                "vega": round(random.uniform(0.05, 0.20), 4),
                "implied_volatility": round(random.uniform(0.15, 0.40), 4),
            }

        return result

    def _mock_option_chain(
        self,
        underlying: str,
        expiry_min_days: int,
        expiry_max_days: int,
        strike_range_pct: float,
        option_type: str | None,
    ) -> dict[str, Any]:
        """Generate mock option chain."""
        base_prices = {"SPY": 450.0, "AAPL": 175.0, "MSFT": 380.0}
        underlying_price = base_prices.get(underlying, 100.0)

        contracts = []
        today = date.today()

        # Generate expiries
        expiries = []
        for days in range(expiry_min_days, expiry_max_days, 7):
            expiry = today + timedelta(days=days)
            if expiry.weekday() == 4:  # Friday
                expiries.append(expiry)

        if not expiries:
            expiries = [today + timedelta(days=30)]

        # Generate strikes
        min_strike = underlying_price * (1 - strike_range_pct)
        max_strike = underlying_price * (1 + strike_range_pct)
        strike_step = 5 if underlying_price > 100 else 2.5
        strikes = []
        strike = math.floor(min_strike / strike_step) * strike_step
        while strike <= max_strike:
            strikes.append(strike)
            strike += strike_step

        # Generate contracts
        for expiry in expiries[:3]:  # Limit expiries
            dte = (expiry - today).days
            for strike in strikes:
                types_to_generate = ["call", "put"]
                if option_type:
                    types_to_generate = [option_type]

                for opt_type in types_to_generate:
                    moneyness = (underlying_price - strike) / underlying_price
                    if opt_type == "put":
                        moneyness = -moneyness

                    # Simple Black-Scholes-like pricing
                    iv = 0.25 + abs(moneyness) * 0.5  # IV smile
                    time_value = iv * math.sqrt(dte / 365) * underlying_price * 0.4
                    intrinsic = (
                        max(0, underlying_price - strike) if opt_type == "call" else max(0, strike - underlying_price)
                    )
                    price = intrinsic + time_value

                    # Delta approximation
                    if opt_type == "call":
                        delta = 0.5 + moneyness * 2  # Simplified
                        delta = max(0.01, min(0.99, delta))
                    else:
                        delta = -0.5 + moneyness * 2
                        delta = max(-0.99, min(-0.01, delta))

                    contract = {
                        "symbol": f"{underlying}{expiry.strftime('%y%m%d')}{opt_type[0].upper()}{int(strike*1000):08d}",
                        "underlying": underlying,
                        "strike": strike,
                        "expiry": expiry.isoformat(),
                        "option_type": opt_type,
                        "bid": round(max(0.01, price - 0.05), 2),
                        "ask": round(price + 0.05, 2),
                        "last": round(price, 2),
                        "volume": random.randint(100, 5000),
                        "open_interest": random.randint(1000, 50000),
                        "greeks": {
                            "delta": round(delta, 3),
                            "gamma": round(0.02 / (1 + abs(moneyness) * 10), 4),
                            "theta": round(-price * 0.01 / max(1, dte), 4),
                            "vega": round(underlying_price * 0.01 * math.sqrt(dte / 365), 4),
                            "implied_volatility": round(iv, 4),
                        },
                    }
                    contracts.append(contract)

        return {
            "underlying": underlying,
            "underlying_price": underlying_price,
            "timestamp": datetime.utcnow().isoformat(),
            "contracts": contracts,
            "total_contracts": len(contracts),
        }

    def _mock_greeks(self, contract_symbol: str) -> dict[str, Any]:
        """Generate mock Greeks for a contract."""
        return {
            "contract_symbol": contract_symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "greeks": {
                "delta": round(random.uniform(-0.9, 0.9), 3),
                "gamma": round(random.uniform(0.01, 0.05), 4),
                "theta": round(random.uniform(-0.15, -0.01), 4),
                "vega": round(random.uniform(0.05, 0.25), 4),
                "rho": round(random.uniform(-0.05, 0.05), 4),
                "implied_volatility": round(random.uniform(0.15, 0.50), 4),
            },
        }

    def _mock_historical(
        self,
        symbol: str,
        days: int,
        resolution: str,
    ) -> dict[str, Any]:
        """Generate mock historical data."""
        base_prices = {"SPY": 450.0, "AAPL": 175.0, "MSFT": 380.0}
        base = base_prices.get(symbol, 100.0)

        bars = []
        current_price = base
        today = datetime.utcnow().replace(hour=16, minute=0, second=0, microsecond=0)

        for i in range(days, 0, -1):
            bar_date = today - timedelta(days=i)

            # Skip weekends for daily
            if resolution == "daily" and bar_date.weekday() >= 5:
                continue

            # Random walk
            change = random.uniform(-0.02, 0.02)
            open_price = current_price
            close_price = current_price * (1 + change)
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))

            bars.append(
                {
                    "timestamp": bar_date.isoformat(),
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": random.randint(5_000_000, 20_000_000),
                }
            )

            current_price = close_price

        return {
            "symbol": symbol,
            "resolution": resolution,
            "bars": bars,
            "total_bars": len(bars),
        }

    def _mock_iv_surface(self, underlying: str) -> dict[str, Any]:
        """Generate mock IV surface data."""
        base_prices = {"SPY": 450.0, "AAPL": 175.0, "MSFT": 380.0}
        spot = base_prices.get(underlying, 100.0)

        # Generate surface points
        expiries = [7, 14, 30, 60, 90, 180]  # Days to expiry
        moneyness_range = [-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10]

        surface = []
        for dte in expiries:
            for m in moneyness_range:
                strike = spot * (1 + m)
                # IV smile: higher IV at wings
                base_iv = 0.20
                smile_adjustment = abs(m) * 2  # Higher IV away from ATM
                term_adjustment = -0.001 * dte  # Slight term structure
                iv = base_iv + smile_adjustment + term_adjustment + random.uniform(-0.02, 0.02)

                surface.append(
                    {
                        "strike": round(strike, 2),
                        "dte": dte,
                        "moneyness": round(m, 4),
                        "iv": round(max(0.05, iv), 4),
                    }
                )

        return {
            "underlying": underlying,
            "spot_price": spot,
            "timestamp": datetime.utcnow().isoformat(),
            "surface": surface,
            "atm_iv": round(0.20 + random.uniform(-0.02, 0.02), 4),
            "skew_25d": round(random.uniform(0.02, 0.06), 4),
        }


def create_market_data_server(
    mock_mode: bool = True,
    data_provider: Any | None = None,
    **config_kwargs,
) -> MarketDataServer:
    """
    Create a market data server instance.

    Args:
        mock_mode: Use mock data
        data_provider: Optional real data provider
        **config_kwargs: Additional config options

    Returns:
        MarketDataServer instance
    """
    config = ServerConfig(
        name="market-data",
        version="1.0.0",
        description="Market data MCP server",
        mock_mode=mock_mode,
        **config_kwargs,
    )
    return MarketDataServer(config=config, mock_mode=mock_mode, data_provider=data_provider)


async def main():
    """Run server in stdio mode."""
    parser = argparse.ArgumentParser(description="Market Data MCP Server")
    parser.add_argument("--stdio", action="store_true", help="Run in stdio mode")
    parser.add_argument("--mock", action="store_true", default=True, help="Use mock data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    server = create_market_data_server(mock_mode=args.mock)

    if args.stdio:
        await server.run_stdio()
    else:
        # Start server for testing
        await server.start()
        print("Server started. Tools available:")
        for schema in server.get_tool_schemas():
            print(f"  - {schema.name}: {schema.description}")

        # Example call
        result = await server.call_tool("get_quote", {"symbol": "SPY"})
        print(f"\nExample call result: {json.dumps(result.to_dict(), indent=2)}")

        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
