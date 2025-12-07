"""
Test Data Builders

Fluent builder pattern implementations for creating test data.
Reduces boilerplate and ensures consistent test object creation.

UPGRADE-015: Test Framework Enhancement - Data Builders
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any
import random


# ============================================================================
# ENUMS FOR BUILDERS
# ============================================================================


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# ============================================================================
# DATA CLASSES FOR BUILDER OUTPUT
# ============================================================================


@dataclass
class TestOrder:
    """Test order data structure."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float | None
    stop_price: float | None
    status: OrderStatus
    submitted_at: datetime
    filled_at: datetime | None
    filled_price: float | None
    filled_quantity: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_price": self.filled_price,
            "filled_quantity": self.filled_quantity,
            "metadata": self.metadata,
        }


@dataclass
class TestPosition:
    """Test position data structure."""
    symbol: str
    quantity: int
    average_price: float
    market_price: float
    unrealized_pnl: float
    realized_pnl: float
    opened_at: datetime

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.average_price

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_price": self.average_price,
            "market_price": self.market_price,
            "market_value": self.market_value,
            "cost_basis": self.cost_basis,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "opened_at": self.opened_at.isoformat(),
        }


@dataclass
class TestPortfolio:
    """Test portfolio data structure."""
    portfolio_id: str
    cash: float
    positions: list[TestPosition]
    starting_equity: float
    created_at: datetime

    @property
    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions)

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions)

    @property
    def total_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions)

    @property
    def drawdown_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return max(0, (self.starting_equity - self.equity) / self.starting_equity)

    def to_dict(self) -> dict:
        return {
            "portfolio_id": self.portfolio_id,
            "cash": self.cash,
            "equity": self.equity,
            "positions": [p.to_dict() for p in self.positions],
            "starting_equity": self.starting_equity,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_realized_pnl": self.total_realized_pnl,
            "drawdown_pct": self.drawdown_pct,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TestPriceBar:
    """Test price bar (OHLCV) data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


# ============================================================================
# ORDER BUILDER
# ============================================================================


class OrderBuilder:
    """
    Fluent builder for creating test orders.

    Usage:
        order = (OrderBuilder()
            .with_symbol("SPY")
            .buy()
            .limit(450.00)
            .quantity(100)
            .filled()
            .build())
    """

    def __init__(self):
        self._order_id = f"ORD-{random.randint(100000, 999999)}"
        self._symbol = "SPY"
        self._side = OrderSide.BUY
        self._order_type = OrderType.MARKET
        self._quantity = 100
        self._price: float | None = None
        self._stop_price: float | None = None
        self._status = OrderStatus.PENDING
        self._submitted_at = datetime.now()
        self._filled_at: datetime | None = None
        self._filled_price: float | None = None
        self._filled_quantity = 0
        self._metadata: dict = {}

    def with_id(self, order_id: str) -> "OrderBuilder":
        """Set order ID."""
        self._order_id = order_id
        return self

    def with_symbol(self, symbol: str) -> "OrderBuilder":
        """Set symbol."""
        self._symbol = symbol
        return self

    def buy(self) -> "OrderBuilder":
        """Set as buy order."""
        self._side = OrderSide.BUY
        return self

    def sell(self) -> "OrderBuilder":
        """Set as sell order."""
        self._side = OrderSide.SELL
        return self

    def market(self) -> "OrderBuilder":
        """Set as market order."""
        self._order_type = OrderType.MARKET
        self._price = None
        return self

    def limit(self, price: float) -> "OrderBuilder":
        """Set as limit order with price."""
        self._order_type = OrderType.LIMIT
        self._price = price
        return self

    def stop(self, stop_price: float) -> "OrderBuilder":
        """Set as stop order."""
        self._order_type = OrderType.STOP
        self._stop_price = stop_price
        return self

    def stop_limit(self, stop_price: float, limit_price: float) -> "OrderBuilder":
        """Set as stop-limit order."""
        self._order_type = OrderType.STOP_LIMIT
        self._stop_price = stop_price
        self._price = limit_price
        return self

    def quantity(self, qty: int) -> "OrderBuilder":
        """Set quantity."""
        self._quantity = qty
        return self

    def pending(self) -> "OrderBuilder":
        """Set status to pending."""
        self._status = OrderStatus.PENDING
        self._filled_at = None
        self._filled_price = None
        self._filled_quantity = 0
        return self

    def submitted(self) -> "OrderBuilder":
        """Set status to submitted."""
        self._status = OrderStatus.SUBMITTED
        return self

    def filled(self, price: float | None = None) -> "OrderBuilder":
        """Set status to filled."""
        self._status = OrderStatus.FILLED
        self._filled_at = datetime.now()
        self._filled_price = price or self._price or 100.0
        self._filled_quantity = self._quantity
        return self

    def partially_filled(self, filled_qty: int, price: float | None = None) -> "OrderBuilder":
        """Set status to partially filled."""
        self._status = OrderStatus.PARTIALLY_FILLED
        self._filled_at = datetime.now()
        self._filled_price = price or self._price or 100.0
        self._filled_quantity = filled_qty
        return self

    def cancelled(self) -> "OrderBuilder":
        """Set status to cancelled."""
        self._status = OrderStatus.CANCELLED
        return self

    def rejected(self) -> "OrderBuilder":
        """Set status to rejected."""
        self._status = OrderStatus.REJECTED
        return self

    def with_metadata(self, **kwargs) -> "OrderBuilder":
        """Add metadata."""
        self._metadata.update(kwargs)
        return self

    def at_time(self, dt: datetime) -> "OrderBuilder":
        """Set submission time."""
        self._submitted_at = dt
        return self

    def build(self) -> TestOrder:
        """Build the order."""
        return TestOrder(
            order_id=self._order_id,
            symbol=self._symbol,
            side=self._side,
            order_type=self._order_type,
            quantity=self._quantity,
            price=self._price,
            stop_price=self._stop_price,
            status=self._status,
            submitted_at=self._submitted_at,
            filled_at=self._filled_at,
            filled_price=self._filled_price,
            filled_quantity=self._filled_quantity,
            metadata=self._metadata,
        )


# ============================================================================
# POSITION BUILDER
# ============================================================================


class PositionBuilder:
    """
    Fluent builder for creating test positions.

    Usage:
        position = (PositionBuilder()
            .symbol("AAPL")
            .long(100)
            .at_price(150.00)
            .with_gain(0.05)
            .build())
    """

    def __init__(self):
        self._symbol = "SPY"
        self._quantity = 100
        self._average_price = 100.0
        self._market_price = 100.0
        self._realized_pnl = 0.0
        self._opened_at = datetime.now() - timedelta(days=1)

    def symbol(self, sym: str) -> "PositionBuilder":
        """Set symbol."""
        self._symbol = sym
        return self

    def long(self, quantity: int) -> "PositionBuilder":
        """Set long position."""
        self._quantity = abs(quantity)
        return self

    def short(self, quantity: int) -> "PositionBuilder":
        """Set short position."""
        self._quantity = -abs(quantity)
        return self

    def at_price(self, price: float) -> "PositionBuilder":
        """Set average entry price."""
        self._average_price = price
        self._market_price = price  # Default market price to entry
        return self

    def current_price(self, price: float) -> "PositionBuilder":
        """Set current market price."""
        self._market_price = price
        return self

    def with_gain(self, pct: float) -> "PositionBuilder":
        """Set position with percentage gain."""
        self._market_price = self._average_price * (1 + pct)
        return self

    def with_loss(self, pct: float) -> "PositionBuilder":
        """Set position with percentage loss."""
        self._market_price = self._average_price * (1 - pct)
        return self

    def realized_pnl(self, amount: float) -> "PositionBuilder":
        """Set realized P&L."""
        self._realized_pnl = amount
        return self

    def opened_at(self, dt: datetime) -> "PositionBuilder":
        """Set open time."""
        self._opened_at = dt
        return self

    def build(self) -> TestPosition:
        """Build the position."""
        unrealized = (self._market_price - self._average_price) * self._quantity
        return TestPosition(
            symbol=self._symbol,
            quantity=self._quantity,
            average_price=self._average_price,
            market_price=self._market_price,
            unrealized_pnl=unrealized,
            realized_pnl=self._realized_pnl,
            opened_at=self._opened_at,
        )


# ============================================================================
# PORTFOLIO BUILDER
# ============================================================================


class PortfolioBuilder:
    """
    Fluent builder for creating test portfolios.

    Usage:
        portfolio = (PortfolioBuilder()
            .with_cash(50000)
            .starting_equity(100000)
            .add_position(PositionBuilder().symbol("SPY").long(100).at_price(450).build())
            .add_position(PositionBuilder().symbol("AAPL").long(50).at_price(180).build())
            .build())
    """

    def __init__(self):
        self._portfolio_id = f"PTF-{random.randint(100000, 999999)}"
        self._cash = 100000.0
        self._positions: list[TestPosition] = []
        self._starting_equity = 100000.0
        self._created_at = datetime.now() - timedelta(days=30)

    def with_id(self, portfolio_id: str) -> "PortfolioBuilder":
        """Set portfolio ID."""
        self._portfolio_id = portfolio_id
        return self

    def with_cash(self, cash: float) -> "PortfolioBuilder":
        """Set cash balance."""
        self._cash = cash
        return self

    def starting_equity(self, equity: float) -> "PortfolioBuilder":
        """Set starting equity."""
        self._starting_equity = equity
        return self

    def add_position(self, position: TestPosition) -> "PortfolioBuilder":
        """Add a position."""
        self._positions.append(position)
        return self

    def with_positions(self, positions: list[TestPosition]) -> "PortfolioBuilder":
        """Set all positions."""
        self._positions = positions
        return self

    def created_at(self, dt: datetime) -> "PortfolioBuilder":
        """Set creation time."""
        self._created_at = dt
        return self

    def in_profit(self, pct: float = 0.10) -> "PortfolioBuilder":
        """Configure portfolio to be in profit."""
        # Adjust positions to show profit
        for pos in self._positions:
            pos.unrealized_pnl = abs(pos.cost_basis) * pct / len(self._positions)
        return self

    def in_drawdown(self, pct: float = 0.10) -> "PortfolioBuilder":
        """Configure portfolio to be in drawdown."""
        total_loss = self._starting_equity * pct
        self._cash -= total_loss
        return self

    def build(self) -> TestPortfolio:
        """Build the portfolio."""
        return TestPortfolio(
            portfolio_id=self._portfolio_id,
            cash=self._cash,
            positions=self._positions,
            starting_equity=self._starting_equity,
            created_at=self._created_at,
        )


# ============================================================================
# PRICE HISTORY BUILDER
# ============================================================================


class PriceHistoryBuilder:
    """
    Fluent builder for creating price history data.

    Usage:
        bars = (PriceHistoryBuilder()
            .symbol("SPY")
            .starting_price(450.0)
            .trending_up(0.02)
            .with_volatility(0.01)
            .num_bars(100)
            .build())
    """

    def __init__(self):
        self._symbol = "SPY"
        self._starting_price = 100.0
        self._trend = 0.0  # Daily trend (e.g., 0.001 = 0.1% daily)
        self._volatility = 0.01  # Daily volatility
        self._num_bars = 30
        self._interval = timedelta(days=1)
        self._start_time = datetime.now() - timedelta(days=30)
        self._volume_base = 1000000
        self._seed: int | None = None
        self._crash_at: int | None = None
        self._crash_pct = 0.0
        self._gap_at: int | None = None
        self._gap_pct = 0.0

    def symbol(self, sym: str) -> "PriceHistoryBuilder":
        """Set symbol."""
        self._symbol = sym
        return self

    def starting_price(self, price: float) -> "PriceHistoryBuilder":
        """Set starting price."""
        self._starting_price = price
        return self

    def trending_up(self, daily_pct: float) -> "PriceHistoryBuilder":
        """Set upward trend."""
        self._trend = abs(daily_pct)
        return self

    def trending_down(self, daily_pct: float) -> "PriceHistoryBuilder":
        """Set downward trend."""
        self._trend = -abs(daily_pct)
        return self

    def flat(self) -> "PriceHistoryBuilder":
        """Set flat (no trend)."""
        self._trend = 0.0
        return self

    def with_volatility(self, vol: float) -> "PriceHistoryBuilder":
        """Set volatility."""
        self._volatility = vol
        return self

    def num_bars(self, n: int) -> "PriceHistoryBuilder":
        """Set number of bars."""
        self._num_bars = n
        return self

    def daily(self) -> "PriceHistoryBuilder":
        """Set daily interval."""
        self._interval = timedelta(days=1)
        return self

    def hourly(self) -> "PriceHistoryBuilder":
        """Set hourly interval."""
        self._interval = timedelta(hours=1)
        return self

    def minute(self) -> "PriceHistoryBuilder":
        """Set minute interval."""
        self._interval = timedelta(minutes=1)
        return self

    def start_time(self, dt: datetime) -> "PriceHistoryBuilder":
        """Set start time."""
        self._start_time = dt
        return self

    def with_seed(self, seed: int) -> "PriceHistoryBuilder":
        """Set random seed for reproducibility."""
        self._seed = seed
        return self

    def with_crash(self, at_bar: int, crash_pct: float) -> "PriceHistoryBuilder":
        """Add a crash event at specific bar."""
        self._crash_at = at_bar
        self._crash_pct = crash_pct
        return self

    def with_gap(self, at_bar: int, gap_pct: float) -> "PriceHistoryBuilder":
        """Add a gap event at specific bar."""
        self._gap_at = at_bar
        self._gap_pct = gap_pct
        return self

    def build(self) -> list[TestPriceBar]:
        """Build the price history."""
        rng = random.Random(self._seed)
        bars = []
        price = self._starting_price
        current_time = self._start_time

        for i in range(self._num_bars):
            # Apply gap if specified
            if self._gap_at is not None and i == self._gap_at:
                price = price * (1 + self._gap_pct)

            # Generate random component
            random_return = rng.gauss(0, self._volatility)

            # Apply crash if specified
            if self._crash_at is not None and i == self._crash_at:
                daily_return = -self._crash_pct
            else:
                daily_return = self._trend + random_return

            # Calculate OHLC
            open_price = price
            close_price = price * (1 + daily_return)

            # High and low with some randomness
            intraday_range = abs(daily_return) + self._volatility * rng.random()
            if close_price >= open_price:
                high = max(open_price, close_price) * (1 + intraday_range * rng.random())
                low = min(open_price, close_price) * (1 - intraday_range * rng.random())
            else:
                high = max(open_price, close_price) * (1 + intraday_range * rng.random())
                low = min(open_price, close_price) * (1 - intraday_range * rng.random())

            # Ensure high >= open, close and low <= open, close
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)

            # Volume with some randomness
            volume = int(self._volume_base * (0.5 + rng.random()))

            bars.append(TestPriceBar(
                symbol=self._symbol,
                timestamp=current_time,
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close_price, 2),
                volume=volume,
            ))

            price = close_price
            current_time += self._interval

        return bars


# ============================================================================
# SCENARIO BUILDER
# ============================================================================


class ScenarioBuilder:
    """
    High-level builder for complete test scenarios.

    Usage:
        scenario = (ScenarioBuilder()
            .name("market_crash_test")
            .with_portfolio(PortfolioBuilder().with_cash(100000).build())
            .with_price_history(PriceHistoryBuilder().with_crash(50, 0.20).build())
            .with_orders([OrderBuilder().buy().quantity(100).build()])
            .build())
    """

    def __init__(self):
        self._name = "test_scenario"
        self._description = ""
        self._portfolio: TestPortfolio | None = None
        self._price_history: list[TestPriceBar] = []
        self._orders: list[TestOrder] = []
        self._metadata: dict = {}

    def name(self, name: str) -> "ScenarioBuilder":
        """Set scenario name."""
        self._name = name
        return self

    def description(self, desc: str) -> "ScenarioBuilder":
        """Set scenario description."""
        self._description = desc
        return self

    def with_portfolio(self, portfolio: TestPortfolio) -> "ScenarioBuilder":
        """Set portfolio."""
        self._portfolio = portfolio
        return self

    def with_price_history(self, history: list[TestPriceBar]) -> "ScenarioBuilder":
        """Set price history."""
        self._price_history = history
        return self

    def with_orders(self, orders: list[TestOrder]) -> "ScenarioBuilder":
        """Set orders."""
        self._orders = orders
        return self

    def add_order(self, order: TestOrder) -> "ScenarioBuilder":
        """Add an order."""
        self._orders.append(order)
        return self

    def with_metadata(self, **kwargs) -> "ScenarioBuilder":
        """Add metadata."""
        self._metadata.update(kwargs)
        return self

    def build(self) -> dict[str, Any]:
        """Build the scenario."""
        return {
            "name": self._name,
            "description": self._description,
            "portfolio": self._portfolio.to_dict() if self._portfolio else None,
            "price_history": [bar.to_dict() for bar in self._price_history],
            "orders": [order.to_dict() for order in self._orders],
            "metadata": self._metadata,
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_simple_order(
    symbol: str = "SPY",
    side: str = "buy",
    quantity: int = 100,
    price: float | None = None,
) -> TestOrder:
    """Create a simple test order quickly."""
    builder = OrderBuilder().with_symbol(symbol).quantity(quantity)
    if side.lower() == "buy":
        builder.buy()
    else:
        builder.sell()
    if price:
        builder.limit(price)
    else:
        builder.market()
    return builder.build()


def create_simple_portfolio(
    cash: float = 100000,
    positions: list[tuple[str, int, float]] | None = None,
) -> TestPortfolio:
    """
    Create a simple test portfolio quickly.

    Args:
        cash: Cash balance
        positions: List of (symbol, quantity, price) tuples
    """
    builder = PortfolioBuilder().with_cash(cash).starting_equity(cash)
    if positions:
        for symbol, qty, price in positions:
            pos = PositionBuilder().symbol(symbol).long(qty).at_price(price).build()
            builder.add_position(pos)
    return builder.build()


def create_trending_history(
    symbol: str = "SPY",
    starting_price: float = 100.0,
    trend: str = "up",
    num_bars: int = 30,
    seed: int | None = None,
) -> list[TestPriceBar]:
    """Create trending price history quickly."""
    builder = (
        PriceHistoryBuilder()
        .symbol(symbol)
        .starting_price(starting_price)
        .num_bars(num_bars)
    )
    if seed is not None:
        builder.with_seed(seed)
    if trend == "up":
        builder.trending_up(0.002)
    elif trend == "down":
        builder.trending_down(0.002)
    else:
        builder.flat()
    return builder.build()


def create_crash_scenario(
    starting_price: float = 100.0,
    crash_pct: float = 0.20,
    crash_at_bar: int = 50,
    num_bars: int = 100,
    seed: int | None = None,
) -> list[TestPriceBar]:
    """Create price history with a crash event."""
    return (
        PriceHistoryBuilder()
        .starting_price(starting_price)
        .with_crash(crash_at_bar, crash_pct)
        .num_bars(num_bars)
        .with_seed(seed)
        .build()
    )


__all__ = [
    # Enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    # Data classes
    "TestOrder",
    "TestPosition",
    "TestPortfolio",
    "TestPriceBar",
    # Builders
    "OrderBuilder",
    "PositionBuilder",
    "PortfolioBuilder",
    "PriceHistoryBuilder",
    "ScenarioBuilder",
    # Convenience functions
    "create_simple_order",
    "create_simple_portfolio",
    "create_trending_history",
    "create_crash_scenario",
]
