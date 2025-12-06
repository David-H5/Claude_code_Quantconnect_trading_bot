"""
MCP Schemas - Data Models for Request/Response Validation

Provides type-safe request and response models for all MCP server tools.
Used for input validation, documentation, and serialization.

UPGRADE-015 Phase 1: MCP Server Foundation

Usage:
    from mcp.schemas import QuoteRequest, QuoteResponse

    request = QuoteRequest(symbol="AAPL", include_greeks=True)
    # Validates input automatically

Note: Uses pydantic if available, falls back to dataclasses otherwise.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any


# Try to use Pydantic if available, otherwise use dataclasses
try:
    from pydantic import BaseModel, Field, field_validator

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

    # Create compatibility layer using dataclasses
    class BaseModel:
        """Fallback BaseModel using dataclasses pattern."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                if hasattr(self, key):
                    # Apply validators if defined
                    validator_name = f"validate_{key}"
                    if hasattr(self, validator_name):
                        value = getattr(self, validator_name)(value)
                setattr(self, key, value)

        def dict(self) -> dict[str, Any]:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def Field(default=None, description="", **kwargs):  # noqa: N802
        """Compatibility Field function."""
        return default

    def field_validator(field_name, **kwargs):
        """Compatibility field_validator decorator."""

        def decorator(func):
            return func

        return decorator


# =============================================================================
# Enums
# =============================================================================


class OrderSide(str, Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    """Time in force."""

    DAY = "day"
    GTC = "gtc"  # Good til cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


class OptionType(str, Enum):
    """Option type."""

    CALL = "call"
    PUT = "put"


class Resolution(str, Enum):
    """Data resolution."""

    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAILY = "daily"


class TradingMode(str, Enum):
    """Trading mode."""

    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


# =============================================================================
# Market Data Schemas
# =============================================================================


class QuoteRequest(BaseModel):
    """Request for stock/option quote."""

    symbol: str = Field(..., description="Ticker symbol (e.g., AAPL, SPY)")
    include_greeks: bool = Field(False, description="Include options Greeks")
    include_volume: bool = Field(True, description="Include volume data")

    @field_validator("symbol", mode="before")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.upper().strip()


class PriceData(BaseModel):
    """Price data for a security."""

    last: float = Field(..., description="Last trade price")
    bid: float = Field(..., description="Best bid price")
    ask: float = Field(..., description="Best ask price")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close_previous: float = Field(..., description="Previous close")
    change: float = Field(..., description="Price change")
    change_pct: float = Field(..., description="Price change percentage")


class VolumeData(BaseModel):
    """Volume data for a security."""

    current: int = Field(..., description="Current day volume")
    average_30d: int = Field(..., description="30-day average volume")
    relative: float = Field(..., description="Relative volume (current/avg)")


class GreeksData(BaseModel):
    """Options Greeks."""

    delta: float = Field(..., ge=-1, le=1, description="Delta")
    gamma: float = Field(..., ge=0, description="Gamma")
    theta: float = Field(..., description="Theta (daily decay)")
    vega: float = Field(..., ge=0, description="Vega")
    rho: float = Field(0.0, description="Rho")
    implied_volatility: float = Field(..., ge=0, description="Implied volatility")


class QuoteResponse(BaseModel):
    """Response containing quote data."""

    symbol: str
    timestamp: datetime
    price: PriceData
    volume: VolumeData | None = None
    greeks: GreeksData | None = None


class OptionChainRequest(BaseModel):
    """Request for option chain."""

    underlying: str = Field(..., description="Underlying symbol")
    expiry_min_days: int = Field(0, ge=0, description="Minimum days to expiry")
    expiry_max_days: int = Field(90, ge=1, description="Maximum days to expiry")
    strike_range_pct: float = Field(0.10, ge=0.01, le=0.50, description="Strike range as % from ATM")
    option_type: OptionType | None = Field(None, description="Filter by call/put")

    @field_validator("underlying", mode="before")
    @classmethod
    def uppercase_underlying(cls, v: str) -> str:
        return v.upper().strip()


class OptionContract(BaseModel):
    """Single option contract."""

    symbol: str = Field(..., description="Option symbol")
    underlying: str
    strike: float
    expiry: date
    option_type: OptionType
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    greeks: GreeksData | None = None


class OptionChainResponse(BaseModel):
    """Response containing option chain."""

    underlying: str
    underlying_price: float
    timestamp: datetime
    contracts: list[OptionContract]
    total_contracts: int


class HistoricalRequest(BaseModel):
    """Request for historical data."""

    symbol: str
    start_date: date
    end_date: date
    resolution: Resolution = Resolution.DAILY

    @field_validator("symbol", mode="before")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("end_date", mode="after")
    @classmethod
    def validate_dates(cls, v: date, info) -> date:
        if hasattr(info, "data") and "start_date" in info.data and v < info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


class OHLCVBar(BaseModel):
    """OHLCV bar data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class HistoricalResponse(BaseModel):
    """Response containing historical data."""

    symbol: str
    resolution: Resolution
    bars: list[OHLCVBar]
    total_bars: int


# =============================================================================
# Order Schemas
# =============================================================================


class OrderRequest(BaseModel):
    """Request to place an order."""

    symbol: str = Field(..., description="Symbol to trade")
    quantity: int = Field(..., gt=0, description="Number of shares/contracts")
    side: OrderSide = Field(..., description="Buy or sell")
    order_type: OrderType = Field(OrderType.LIMIT, description="Order type")
    limit_price: float | None = Field(None, gt=0, description="Limit price")
    stop_price: float | None = Field(None, gt=0, description="Stop price")
    time_in_force: TimeInForce = Field(TimeInForce.DAY, description="Time in force")
    trading_mode: TradingMode = Field(TradingMode.PAPER, description="Trading mode")

    @field_validator("symbol", mode="before")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("limit_price", mode="after")
    @classmethod
    def validate_limit_price(cls, v: float | None, info) -> float | None:
        order_type = info.data.get("order_type") if hasattr(info, "data") else None
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and v is None:
            raise ValueError("limit_price required for limit/stop_limit orders")
        return v

    @field_validator("stop_price", mode="after")
    @classmethod
    def validate_stop_price(cls, v: float | None, info) -> float | None:
        order_type = info.data.get("order_type") if hasattr(info, "data") else None
        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and v is None:
            raise ValueError("stop_price required for stop/stop_limit orders")
        return v


class OrderResponse(BaseModel):
    """Response after placing an order."""

    order_id: str
    symbol: str
    quantity: int
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    limit_price: float | None = None
    stop_price: float | None = None
    filled_quantity: int = 0
    average_fill_price: float | None = None
    submitted_at: datetime
    filled_at: datetime | None = None
    message: str | None = None


class CancelOrderRequest(BaseModel):
    """Request to cancel an order."""

    order_id: str = Field(..., description="Order ID to cancel")


class CancelOrderResponse(BaseModel):
    """Response after cancelling an order."""

    order_id: str
    success: bool
    message: str
    cancelled_at: datetime | None = None


class FillRecord(BaseModel):
    """Record of an order fill."""

    fill_id: str
    order_id: str
    symbol: str
    quantity: int
    price: float
    side: OrderSide
    commission: float = 0.0
    fill_time: datetime


# =============================================================================
# Portfolio Schemas
# =============================================================================


class PortfolioRequest(BaseModel):
    """Request for portfolio data."""

    include_positions: bool = Field(True, description="Include position details")
    include_pnl: bool = Field(True, description="Include P&L data")
    include_exposure: bool = Field(True, description="Include exposure metrics")


class Position(BaseModel):
    """Single portfolio position."""

    symbol: str
    quantity: int
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float = 0.0
    weight_pct: float = 0.0


class PnLData(BaseModel):
    """Profit and Loss data."""

    daily: float = Field(..., description="Daily P&L")
    daily_pct: float = Field(..., description="Daily P&L percentage")
    weekly: float = Field(0.0, description="Weekly P&L")
    weekly_pct: float = Field(0.0, description="Weekly P&L percentage")
    monthly: float = Field(0.0, description="Monthly P&L")
    monthly_pct: float = Field(0.0, description="Monthly P&L percentage")
    ytd: float = Field(0.0, description="Year-to-date P&L")
    ytd_pct: float = Field(0.0, description="Year-to-date P&L percentage")


class ExposureData(BaseModel):
    """Portfolio exposure metrics."""

    gross_exposure: float = Field(..., description="Total long + short value")
    net_exposure: float = Field(..., description="Long - short value")
    long_exposure: float = Field(..., description="Total long value")
    short_exposure: float = Field(..., description="Total short value")
    sector_exposure: dict[str, float] = Field(default_factory=dict)


class PortfolioResponse(BaseModel):
    """Response containing portfolio data."""

    timestamp: datetime
    cash: float
    total_value: float
    buying_power: float
    positions: list[Position] = Field(default_factory=list)
    pnl: PnLData | None = None
    exposure: ExposureData | None = None


# =============================================================================
# Risk Schemas
# =============================================================================


class RiskCheckRequest(BaseModel):
    """Request for risk check."""

    symbol: str
    quantity: int
    side: OrderSide
    price: float


class RiskCheckResponse(BaseModel):
    """Response from risk check."""

    approved: bool
    symbol: str
    quantity: int
    side: OrderSide
    checks: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)
    rejection_reason: str | None = None


class RiskMetrics(BaseModel):
    """Portfolio risk metrics."""

    var_95: float = Field(..., description="95% Value at Risk")
    var_99: float = Field(..., description="99% Value at Risk")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(0.0, description="Sortino ratio")
    beta: float = Field(1.0, description="Portfolio beta")
    correlation_spy: float = Field(0.0, description="Correlation to SPY")


# =============================================================================
# Backtest Schemas
# =============================================================================


class BacktestRequest(BaseModel):
    """Request to run a backtest."""

    algorithm_id: str = Field(..., description="Algorithm identifier")
    start_date: date
    end_date: date
    initial_cash: float = Field(100000, gt=0, description="Initial cash")
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("end_date", mode="after")
    @classmethod
    def validate_dates(cls, v: date, info) -> date:
        if hasattr(info, "data") and "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


class BacktestMetrics(BaseModel):
    """Backtest performance metrics."""

    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float


class BacktestResponse(BaseModel):
    """Response containing backtest results."""

    backtest_id: str
    algorithm_id: str
    start_date: date
    end_date: date
    initial_cash: float
    final_value: float
    metrics: BacktestMetrics
    equity_curve: list[dict[str, Any]] = Field(default_factory=list)
    trades: list[dict[str, Any]] = Field(default_factory=list)
    completed_at: datetime


# =============================================================================
# System Schemas
# =============================================================================


class HealthCheckResponse(BaseModel):
    """Server health check response."""

    server: str
    version: str
    state: str
    is_healthy: bool
    uptime_seconds: float | None = None
    tools_registered: int
    total_calls: int
    error_count: int
    error_rate: float
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Standard error response."""

    success: bool = False
    error: str
    error_code: str
    details: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Utility Functions
# =============================================================================


def validate_symbol(symbol: str) -> str:
    """Validate and normalize a ticker symbol."""
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    symbol = symbol.upper().strip()
    if len(symbol) > 20:
        raise ValueError("Symbol too long")
    return symbol


def create_error_response(error: str, code: str, details: dict | None = None) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        error=error,
        error_code=code,
        details=details,
    )
