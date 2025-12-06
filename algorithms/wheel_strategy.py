"""
The Wheel Strategy Implementation

A premium collection strategy that cycles through:
1. Sell cash-secured puts (collect premium, wait for assignment or expiry)
2. If assigned, sell covered calls on the shares
3. If called away, start again with cash-secured puts

Based on ThetaGang and research from options trading communities.

Key parameters:
- Target delta: 0.25-0.35 for puts, 0.20-0.30 for calls
- DTE: 30-45 days
- Strike selection: Below support for puts, above resistance for calls
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# QuantConnect imports (available at runtime)
try:
    from AlgorithmImports import *
except ImportError:
    # Stubs for development
    class QCAlgorithm:
        pass

    class Slice:
        pass


class WheelPhase(Enum):
    """Current phase of the wheel strategy."""

    CASH = "cash"  # Have cash, sell puts
    SHORT_PUT = "short_put"  # Waiting for put expiry/assignment
    ASSIGNED = "assigned"  # Got assigned shares, sell calls
    SHORT_CALL = "short_call"  # Waiting for call expiry/assignment


@dataclass
class WheelPosition:
    """Tracks a single wheel position."""

    symbol: str
    phase: WheelPhase
    shares: int = 0
    cost_basis: float = 0.0  # Per share cost if holding
    option_symbol: str | None = None
    option_strike: float = 0.0
    option_expiry: datetime | None = None
    option_premium: float = 0.0
    total_premium_collected: float = 0.0
    cycles_completed: int = 0
    entry_date: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "phase": self.phase.value,
            "shares": self.shares,
            "cost_basis": self.cost_basis,
            "option_symbol": self.option_symbol,
            "option_strike": self.option_strike,
            "option_expiry": self.option_expiry.isoformat() if self.option_expiry else None,
            "option_premium": self.option_premium,
            "total_premium_collected": self.total_premium_collected,
            "cycles_completed": self.cycles_completed,
        }


@dataclass
class WheelConfig:
    """Configuration for wheel strategy."""

    # Put parameters
    put_delta_target: float = 0.30  # Target delta for puts
    put_dte_min: int = 30  # Minimum days to expiry
    put_dte_max: int = 45  # Maximum days to expiry

    # Call parameters
    call_delta_target: float = 0.25  # Target delta for covered calls
    call_dte_min: int = 30
    call_dte_max: int = 45

    # Position sizing
    max_positions: int = 5  # Maximum wheel positions
    allocation_per_position: float = 0.20  # 20% per position

    # Exit parameters
    profit_target_pct: float = 0.50  # Close at 50% profit
    loss_limit_pct: float = 2.00  # Close at 200% loss (2x premium)
    days_before_expiry_roll: int = 7  # Roll if < 7 DTE

    # Assignment handling
    accept_early_assignment: bool = True
    sell_call_after_assignment_days: int = 1  # Wait before selling call

    # Restrictions
    min_option_volume: int = 100
    min_open_interest: int = 500
    max_spread_pct: float = 0.05  # Max 5% bid-ask spread


class WheelStrategy:
    """
    The Wheel strategy manager.

    Manages multiple wheel positions across different underlyings.
    """

    def __init__(
        self,
        config: WheelConfig,
        order_callback: Callable[[dict], None] | None = None,
    ):
        """
        Initialize wheel strategy.

        Args:
            config: Strategy configuration
            order_callback: Callback for order generation
        """
        self.config = config
        self.order_callback = order_callback
        self.positions: dict[str, WheelPosition] = {}

    def add_symbol(self, symbol: str) -> WheelPosition:
        """
        Add a symbol to wheel.

        Args:
            symbol: Underlying symbol

        Returns:
            New WheelPosition
        """
        position = WheelPosition(symbol=symbol, phase=WheelPhase.CASH)
        self.positions[symbol] = position
        return position

    def get_position(self, symbol: str) -> WheelPosition | None:
        """Get position for symbol."""
        return self.positions.get(symbol)

    def update_phase(self, symbol: str, phase: WheelPhase) -> None:
        """Update position phase."""
        if symbol in self.positions:
            self.positions[symbol].phase = phase

    def select_put_strike(
        self,
        symbol: str,
        spot_price: float,
        option_chain: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Select optimal put strike for selling.

        Args:
            symbol: Underlying symbol
            spot_price: Current price
            option_chain: Available options

        Returns:
            Selected option contract or None
        """
        # Filter to puts within DTE range
        puts = [
            c
            for c in option_chain
            if c.get("type") == "put"
            and self.config.put_dte_min <= c.get("dte", 0) <= self.config.put_dte_max
            and c.get("volume", 0) >= self.config.min_option_volume
            and c.get("open_interest", 0) >= self.config.min_open_interest
        ]

        if not puts:
            return None

        # Check bid-ask spread
        puts = [
            c
            for c in puts
            if c.get("ask", 0) > 0
            and (c.get("ask", 0) - c.get("bid", 0)) / c.get("ask", 0) <= self.config.max_spread_pct
        ]

        if not puts:
            return None

        # Find strike closest to target delta
        target_delta = -abs(self.config.put_delta_target)  # Puts have negative delta
        selected = min(puts, key=lambda c: abs(c.get("delta", 0) - target_delta))

        return selected

    def select_call_strike(
        self,
        symbol: str,
        cost_basis: float,
        option_chain: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Select optimal call strike for selling.

        Ensures strike is above cost basis if possible.

        Args:
            symbol: Underlying symbol
            cost_basis: Cost basis per share
            option_chain: Available options

        Returns:
            Selected option contract or None
        """
        # Filter to calls within DTE range and above cost basis
        calls = [
            c
            for c in option_chain
            if c.get("type") == "call"
            and self.config.call_dte_min <= c.get("dte", 0) <= self.config.call_dte_max
            and c.get("strike", 0) >= cost_basis  # Don't sell below cost basis
            and c.get("volume", 0) >= self.config.min_option_volume
            and c.get("open_interest", 0) >= self.config.min_open_interest
        ]

        if not calls:
            # If no strikes above cost basis, consider any call
            calls = [
                c
                for c in option_chain
                if c.get("type") == "call" and self.config.call_dte_min <= c.get("dte", 0) <= self.config.call_dte_max
            ]

        if not calls:
            return None

        # Check bid-ask spread
        calls = [
            c
            for c in calls
            if c.get("ask", 0) > 0
            and (c.get("ask", 0) - c.get("bid", 0)) / c.get("ask", 0) <= self.config.max_spread_pct
        ]

        if not calls:
            return None

        # Find strike closest to target delta
        target_delta = abs(self.config.call_delta_target)
        selected = min(calls, key=lambda c: abs(c.get("delta", 0) - target_delta))

        return selected

    def check_position_management(
        self,
        symbol: str,
        current_option_price: float,
    ) -> str | None:
        """
        Check if position needs management (early close, roll, etc.).

        Args:
            symbol: Position symbol
            current_option_price: Current option price

        Returns:
            Action to take ("close", "roll", None)
        """
        position = self.positions.get(symbol)
        if not position or position.phase == WheelPhase.CASH:
            return None

        if position.option_premium == 0:
            return None

        # Check profit target
        profit_pct = (position.option_premium - current_option_price) / position.option_premium
        if profit_pct >= self.config.profit_target_pct:
            return "close"

        # Check loss limit
        loss_pct = (current_option_price - position.option_premium) / position.option_premium
        if loss_pct >= self.config.loss_limit_pct:
            return "close"

        # Check days to expiry for roll
        if position.option_expiry:
            dte = (position.option_expiry - datetime.now()).days
            if dte <= self.config.days_before_expiry_roll:
                return "roll"

        return None

    def process_assignment(
        self,
        symbol: str,
        shares: int,
        strike: float,
    ) -> None:
        """
        Process put assignment.

        Args:
            symbol: Symbol that was assigned
            shares: Number of shares assigned
            strike: Strike price (cost basis)
        """
        position = self.positions.get(symbol)
        if not position:
            return

        # Calculate cost basis including premium received
        premium_per_share = position.option_premium / (shares / 100)
        effective_cost = strike - premium_per_share

        position.phase = WheelPhase.ASSIGNED
        position.shares = shares
        position.cost_basis = effective_cost
        position.option_symbol = None
        position.option_strike = 0
        position.option_expiry = None

    def process_call_assignment(
        self,
        symbol: str,
        strike: float,
    ) -> float:
        """
        Process call assignment (shares called away).

        Args:
            symbol: Symbol that was assigned
            strike: Strike price (sale price)

        Returns:
            Profit/loss from the round trip
        """
        position = self.positions.get(symbol)
        if not position:
            return 0.0

        # Calculate P/L
        shares = position.shares
        pnl = (strike - position.cost_basis) * shares
        pnl += position.total_premium_collected

        # Reset to cash phase
        position.phase = WheelPhase.CASH
        position.shares = 0
        position.cost_basis = 0
        position.cycles_completed += 1
        position.option_symbol = None

        return pnl

    def sell_put(
        self,
        symbol: str,
        contract: dict[str, Any],
        quantity: int = 1,
    ) -> dict[str, Any]:
        """
        Record selling a put.

        Args:
            symbol: Underlying symbol
            contract: Option contract details
            quantity: Number of contracts

        Returns:
            Order details
        """
        position = self.positions.get(symbol)
        if not position:
            position = self.add_symbol(symbol)

        premium = contract.get("bid", 0) * 100 * quantity  # Per contract

        position.phase = WheelPhase.SHORT_PUT
        position.option_symbol = contract.get("symbol", "")
        position.option_strike = contract.get("strike", 0)
        position.option_expiry = contract.get("expiry")
        position.option_premium = premium
        position.total_premium_collected += premium

        order = {
            "action": "sell_to_open",
            "symbol": contract.get("symbol"),
            "underlying": symbol,
            "type": "put",
            "strike": contract.get("strike"),
            "expiry": contract.get("expiry"),
            "quantity": quantity,
            "price": contract.get("bid"),
            "premium": premium,
        }

        if self.order_callback:
            self.order_callback(order)

        return order

    def sell_call(
        self,
        symbol: str,
        contract: dict[str, Any],
        quantity: int = 1,
    ) -> dict[str, Any]:
        """
        Record selling a covered call.

        Args:
            symbol: Underlying symbol
            contract: Option contract details
            quantity: Number of contracts

        Returns:
            Order details
        """
        position = self.positions.get(symbol)
        if not position:
            return {}

        premium = contract.get("bid", 0) * 100 * quantity

        position.phase = WheelPhase.SHORT_CALL
        position.option_symbol = contract.get("symbol", "")
        position.option_strike = contract.get("strike", 0)
        position.option_expiry = contract.get("expiry")
        position.option_premium = premium
        position.total_premium_collected += premium

        order = {
            "action": "sell_to_open",
            "symbol": contract.get("symbol"),
            "underlying": symbol,
            "type": "call",
            "strike": contract.get("strike"),
            "expiry": contract.get("expiry"),
            "quantity": quantity,
            "price": contract.get("bid"),
            "premium": premium,
        }

        if self.order_callback:
            self.order_callback(order)

        return order

    def close_option(
        self,
        symbol: str,
        close_price: float,
        quantity: int = 1,
    ) -> dict[str, Any]:
        """
        Close an option position.

        Args:
            symbol: Underlying symbol
            close_price: Price to close at
            quantity: Number of contracts

        Returns:
            Order details
        """
        position = self.positions.get(symbol)
        if not position or not position.option_symbol:
            return {}

        cost = close_price * 100 * quantity
        profit = position.option_premium - cost

        order = {
            "action": "buy_to_close",
            "symbol": position.option_symbol,
            "underlying": symbol,
            "quantity": quantity,
            "price": close_price,
            "cost": cost,
            "profit": profit,
        }

        # Update position
        if position.phase == WheelPhase.SHORT_PUT:
            position.phase = WheelPhase.CASH
        elif position.phase == WheelPhase.SHORT_CALL:
            position.phase = WheelPhase.ASSIGNED

        position.option_symbol = None
        position.option_strike = 0
        position.option_expiry = None
        position.option_premium = 0

        if self.order_callback:
            self.order_callback(order)

        return order

    def get_status(self) -> dict[str, Any]:
        """Get status of all wheel positions."""
        return {
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "total_positions": len(self.positions),
            "phases": {
                phase.value: sum(1 for p in self.positions.values() if p.phase == phase) for phase in WheelPhase
            },
            "total_premium": sum(p.total_premium_collected for p in self.positions.values()),
            "cycles_completed": sum(p.cycles_completed for p in self.positions.values()),
        }


class WheelAlgorithm(QCAlgorithm):
    """
    QuantConnect algorithm implementing The Wheel strategy.

    Can be used standalone or integrated with main options bot.
    """

    def Initialize(self) -> None:
        """Initialize the wheel algorithm."""
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        # Wheel configuration
        self.wheel_config = WheelConfig(
            put_delta_target=0.30,
            call_delta_target=0.25,
            put_dte_min=30,
            put_dte_max=45,
            max_positions=3,
            allocation_per_position=0.30,
            profit_target_pct=0.50,
        )

        self.wheel = WheelStrategy(
            config=self.wheel_config,
            order_callback=self._execute_order,
        )

        # Add symbols for the wheel
        self.symbols = []
        watchlist = ["AAPL", "MSFT", "SPY"]

        for ticker in watchlist:
            equity = self.AddEquity(ticker, Resolution.Minute)
            option = self.AddOption(ticker, Resolution.Minute)
            option.SetFilter(self._option_filter)
            self.symbols.append(equity.Symbol)
            self.wheel.add_symbol(ticker)

        # Schedule daily checks
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self._daily_wheel_check,
        )

    def _option_filter(self, universe) -> Any:
        """Filter option contracts."""
        return (
            universe.IncludeWeeklys()
            .Strikes(-10, 10)
            .Expiration(
                self.wheel_config.put_dte_min,
                self.wheel_config.put_dte_max + 7,
            )
        )

    def OnData(self, data: Slice) -> None:
        """
        Store option chains for use in scheduled functions.

        Scheduled functions don't have access to the data parameter,
        so we store option chains here for later use.
        """
        if not hasattr(self, "_option_chains"):
            self._option_chains = {}

        # Store current option chains
        for symbol in data.option_chains.keys():
            self._option_chains[symbol] = data.option_chains[symbol]

    def _daily_wheel_check(self) -> None:
        """Daily check and management of wheel positions."""
        for symbol in self.symbols:
            position = self.wheel.get_position(str(symbol))
            if not position:
                continue

            # Get current option chain from stored chains
            # Note: Option chains are stored in OnData() to self._option_chains
            # because scheduled functions don't have access to data parameter
            if not hasattr(self, "_option_chains"):
                continue

            chain = self._option_chains.get(symbol)
            if not chain:
                continue

            # Convert to list of dicts
            option_chain = self._chain_to_list(chain, symbol)

            # Handle based on phase
            if position.phase == WheelPhase.CASH:
                self._handle_cash_phase(str(symbol), option_chain)
            elif position.phase == WheelPhase.SHORT_PUT:
                self._handle_short_put_phase(str(symbol), option_chain)
            elif position.phase == WheelPhase.ASSIGNED:
                self._handle_assigned_phase(str(symbol), option_chain)
            elif position.phase == WheelPhase.SHORT_CALL:
                self._handle_short_call_phase(str(symbol), option_chain)

    def _chain_to_list(self, chain: Any, underlying: Any) -> list[dict]:
        """
        Convert QuantConnect option chain to list of dicts.

        Note: As of LEAN PR #6720, Greeks use implied volatility and require NO warmup.
        Greeks are available immediately upon option data arrival.
        """
        result = []
        for contract in chain:
            result.append(
                {
                    "symbol": str(contract.Symbol),
                    "type": "call" if contract.Right == OptionRight.Call else "put",
                    "strike": contract.Strike,
                    "expiry": contract.Expiry,
                    "bid": contract.BidPrice,
                    "ask": contract.AskPrice,
                    # Greeks are IV-based (LEAN PR #6720), always available immediately
                    # No None check needed - Greeks object always exists
                    "delta": contract.Greeks.Delta,
                    "dte": (contract.Expiry - self.Time).days,
                    "volume": contract.Volume,
                    "open_interest": contract.OpenInterest,
                }
            )
        return result

    def _handle_cash_phase(self, symbol: str, chain: list[dict]) -> None:
        """Handle cash phase - sell puts."""
        # Check if we have cash for this position
        required_cash = self.Portfolio.TotalPortfolioValue * self.wheel_config.allocation_per_position
        if self.Portfolio.Cash < required_cash:
            return

        # Select put to sell
        contract = self.wheel.select_put_strike(symbol, self.Securities[symbol].Price, chain)
        if contract:
            self.wheel.sell_put(symbol, contract)

    def _handle_short_put_phase(self, symbol: str, chain: list[dict]) -> None:
        """Handle short put phase - monitor for close/roll."""
        position = self.wheel.get_position(symbol)
        if not position or not position.option_symbol:
            return

        # Get current option price
        if position.option_symbol in self.Securities:
            current_price = self.Securities[position.option_symbol].Price
            action = self.wheel.check_position_management(symbol, current_price)

            if action == "close":
                self.wheel.close_option(symbol, current_price)

    def _handle_assigned_phase(self, symbol: str, chain: list[dict]) -> None:
        """Handle assigned phase - sell covered calls."""
        position = self.wheel.get_position(symbol)
        if not position:
            return

        # Select call to sell
        contract = self.wheel.select_call_strike(symbol, position.cost_basis, chain)
        if contract:
            contracts = position.shares // 100
            self.wheel.sell_call(symbol, contract, contracts)

    def _handle_short_call_phase(self, symbol: str, chain: list[dict]) -> None:
        """Handle short call phase - monitor for close/roll."""
        position = self.wheel.get_position(symbol)
        if not position or not position.option_symbol:
            return

        # Get current option price
        if position.option_symbol in self.Securities:
            current_price = self.Securities[position.option_symbol].Price
            action = self.wheel.check_position_management(symbol, current_price)

            if action == "close":
                self.wheel.close_option(symbol, current_price)

    def _execute_order(self, order: dict) -> None:
        """Execute order from wheel strategy."""
        action = order.get("action", "")
        symbol = order.get("symbol", "")
        quantity = order.get("quantity", 1)

        if action == "sell_to_open":
            self.MarketOrder(symbol, -quantity)
        elif action == "buy_to_close":
            self.MarketOrder(symbol, quantity)

    def OnOrderEvent(self, orderEvent) -> None:
        """Handle order events including assignments."""
        if orderEvent.Status != OrderStatus.Filled:
            return

        # Check for option assignment
        if "assignment" in str(orderEvent.Message).lower():
            symbol = str(orderEvent.Symbol.Underlying)
            if orderEvent.Direction == OrderDirection.Buy:
                # Put assigned - bought shares
                self.wheel.process_assignment(symbol, orderEvent.FillQuantity, orderEvent.FillPrice)
            else:
                # Call assigned - shares called away
                self.wheel.process_call_assignment(symbol, orderEvent.FillPrice)


def create_wheel_strategy(config: WheelConfig | None = None) -> WheelStrategy:
    """Create wheel strategy with optional config."""
    if config is None:
        config = WheelConfig()
    return WheelStrategy(config)


__all__ = [
    "WheelAlgorithm",
    "WheelConfig",
    "WheelPhase",
    "WheelPosition",
    "WheelStrategy",
    "create_wheel_strategy",
]
