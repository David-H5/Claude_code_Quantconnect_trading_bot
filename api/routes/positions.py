"""
Position and P&L Query Endpoints

Provides REST endpoints for:
- Querying current positions
- Getting P&L summaries
- Position history

UPGRADE-008: REST API Server (December 2025)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel


logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class PositionSource(str, Enum):
    """Position source types."""

    BOT = "bot"
    MANUAL = "manual"
    AUTONOMOUS = "autonomous"
    ALL = "all"


class GreeksData(BaseModel):
    """Option Greeks data."""

    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None
    iv: float | None = None


class PositionData(BaseModel):
    """Position details."""

    symbol: str
    underlying: str
    quantity: int
    avg_cost: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float
    source: str
    opened_at: datetime
    last_updated: datetime
    greeks: GreeksData | None = None
    days_held: int = 0
    notes: str | None = None


class PositionListResponse(BaseModel):
    """Position list response."""

    positions: list[PositionData]
    count: int
    total_value: float
    total_unrealized_pnl: float


class PnLSummary(BaseModel):
    """P&L summary data."""

    daily_pnl: float
    daily_pnl_pct: float
    weekly_pnl: float
    weekly_pnl_pct: float
    monthly_pnl: float
    monthly_pnl_pct: float
    total_pnl: float
    total_pnl_pct: float
    unrealized_pnl: float
    realized_pnl: float
    portfolio_value: float
    cash: float
    buying_power: float
    timestamp: datetime


class PositionSummary(BaseModel):
    """Position summary statistics."""

    total_positions: int
    winning_positions: int
    losing_positions: int
    bot_positions: int
    manual_positions: int
    autonomous_positions: int
    avg_holding_days: float
    best_performer: str | None = None
    worst_performer: str | None = None


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/positions", response_model=PositionListResponse)
async def list_positions(
    source: PositionSource | None = Query(None, description="Filter by source"),
    symbol: str | None = Query(None, description="Filter by symbol/underlying"),
    min_pnl: float | None = Query(None, description="Minimum unrealized P&L"),
    max_pnl: float | None = Query(None, description="Maximum unrealized P&L"),
):
    """List all current positions with optional filters.

    Returns:
        PositionListResponse with filtered positions
    """
    from ..rest_server import get_algorithm

    algo = get_algorithm()

    # If no algorithm connected, return empty but valid response
    if algo is None:
        return PositionListResponse(
            positions=[],
            count=0,
            total_value=0.0,
            total_unrealized_pnl=0.0,
        )

    try:
        positions = []
        total_value = 0.0
        total_unrealized = 0.0

        # Get positions from algorithm's Portfolio
        if hasattr(algo, "Portfolio"):
            for holding in algo.Portfolio.Values:
                if not holding.Invested:
                    continue

                # Get basic position data
                symbol_str = str(holding.Symbol)
                underlying = _get_underlying(symbol_str)
                quantity = int(holding.Quantity)
                avg_cost = float(holding.AveragePrice)
                current_price = float(holding.Price)
                unrealized = float(holding.UnrealizedProfit)
                unrealized_pct = float(holding.UnrealizedProfitPercent) * 100

                # Apply filters
                if source and source != PositionSource.ALL:
                    # Would need to track source in bot_manager
                    pass
                if symbol and symbol.upper() not in symbol_str.upper():
                    continue
                if min_pnl is not None and unrealized < min_pnl:
                    continue
                if max_pnl is not None and unrealized > max_pnl:
                    continue

                position = PositionData(
                    symbol=symbol_str,
                    underlying=underlying,
                    quantity=quantity,
                    avg_cost=avg_cost,
                    current_price=current_price,
                    unrealized_pnl=unrealized,
                    unrealized_pnl_pct=unrealized_pct,
                    realized_pnl=0.0,  # Would need to track
                    source="algorithm",
                    opened_at=datetime.now(timezone.utc),  # Would need to track
                    last_updated=datetime.now(timezone.utc),
                    days_held=0,
                )
                positions.append(position)
                total_value += quantity * current_price
                total_unrealized += unrealized

        return PositionListResponse(
            positions=positions,
            count=len(positions),
            total_value=total_value,
            total_unrealized_pnl=total_unrealized,
        )

    except Exception as e:
        logger.error(f"Error listing positions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/positions/{symbol}")
async def get_position(symbol: str):
    """Get position details for a specific symbol.

    Args:
        symbol: Position symbol

    Returns:
        Position details or 404 if not found
    """
    from ..rest_server import get_algorithm

    algo = get_algorithm()
    if algo is None:
        raise HTTPException(status_code=503, detail="Algorithm not connected")

    try:
        if hasattr(algo, "Portfolio"):
            for holding in algo.Portfolio.Values:
                if symbol.upper() in str(holding.Symbol).upper() and holding.Invested:
                    return PositionData(
                        symbol=str(holding.Symbol),
                        underlying=_get_underlying(str(holding.Symbol)),
                        quantity=int(holding.Quantity),
                        avg_cost=float(holding.AveragePrice),
                        current_price=float(holding.Price),
                        unrealized_pnl=float(holding.UnrealizedProfit),
                        unrealized_pnl_pct=float(holding.UnrealizedProfitPercent) * 100,
                        realized_pnl=0.0,
                        source="algorithm",
                        opened_at=datetime.now(timezone.utc),
                        last_updated=datetime.now(timezone.utc),
                    )

        raise HTTPException(status_code=404, detail=f"Position not found: {symbol}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting position: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/pnl", response_model=PnLSummary)
async def get_pnl_summary():
    """Get P&L summary for the portfolio.

    Returns:
        PnLSummary with current P&L metrics
    """
    from ..rest_server import get_algorithm

    algo = get_algorithm()

    # Return zeros if no algorithm connected
    if algo is None:
        return PnLSummary(
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            weekly_pnl=0.0,
            weekly_pnl_pct=0.0,
            monthly_pnl=0.0,
            monthly_pnl_pct=0.0,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            portfolio_value=0.0,
            cash=0.0,
            buying_power=0.0,
            timestamp=datetime.now(timezone.utc),
        )

    try:
        portfolio_value = float(algo.Portfolio.TotalPortfolioValue)
        cash = float(algo.Portfolio.Cash) if hasattr(algo.Portfolio, "Cash") else 0.0

        # Calculate unrealized P&L
        unrealized = 0.0
        if hasattr(algo, "Portfolio"):
            for holding in algo.Portfolio.Values:
                if holding.Invested:
                    unrealized += float(holding.UnrealizedProfit)

        # Get starting equity if available
        starting_equity = getattr(algo, "_starting_equity", portfolio_value)
        total_pnl = portfolio_value - starting_equity
        total_pnl_pct = (total_pnl / starting_equity * 100) if starting_equity > 0 else 0.0

        return PnLSummary(
            daily_pnl=getattr(algo, "_daily_pnl", 0.0),
            daily_pnl_pct=getattr(algo, "_daily_pnl_pct", 0.0),
            weekly_pnl=0.0,  # Would need to track
            weekly_pnl_pct=0.0,
            monthly_pnl=0.0,  # Would need to track
            monthly_pnl_pct=0.0,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            unrealized_pnl=unrealized,
            realized_pnl=total_pnl - unrealized,
            portfolio_value=portfolio_value,
            cash=cash,
            buying_power=getattr(algo.Portfolio, "MarginRemaining", cash),
            timestamp=datetime.now(timezone.utc),
        )

    except Exception as e:
        logger.error(f"Error getting P&L summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/positions/summary", response_model=PositionSummary)
async def get_position_summary():
    """Get position summary statistics.

    Returns:
        PositionSummary with aggregated statistics
    """
    from ..rest_server import get_algorithm

    algo = get_algorithm()

    if algo is None:
        return PositionSummary(
            total_positions=0,
            winning_positions=0,
            losing_positions=0,
            bot_positions=0,
            manual_positions=0,
            autonomous_positions=0,
            avg_holding_days=0.0,
        )

    try:
        total = 0
        winning = 0
        losing = 0

        if hasattr(algo, "Portfolio"):
            for holding in algo.Portfolio.Values:
                if holding.Invested:
                    total += 1
                    if float(holding.UnrealizedProfit) > 0:
                        winning += 1
                    elif float(holding.UnrealizedProfit) < 0:
                        losing += 1

        return PositionSummary(
            total_positions=total,
            winning_positions=winning,
            losing_positions=losing,
            bot_positions=0,  # Would need to track
            manual_positions=0,
            autonomous_positions=0,
            avg_holding_days=0.0,
        )

    except Exception as e:
        logger.error(f"Error getting position summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def _get_underlying(symbol: str) -> str:
    """Extract underlying symbol from option symbol.

    Args:
        symbol: Full option symbol

    Returns:
        Underlying symbol
    """
    # Simple extraction - first part before space or numbers
    parts = symbol.split()
    if parts:
        underlying = parts[0]
        # Remove trailing option identifiers
        for suffix in ["C", "P", "CALL", "PUT"]:
            if underlying.endswith(suffix):
                underlying = underlying[: -len(suffix)]
        return underlying
    return symbol
