"""
Multi-Leg Options Strategy Module

Supports complex options strategies including:
- Vertical Spreads (Bull/Bear Call/Put)
- Iron Condors
- Strangles and Straddles
- Butterflies
- Calendar Spreads
- Covered Calls/Puts

Based on patterns from OptionSuite, Quantsbin, and ThetaGang.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Enumeration of supported strategy types."""

    # Single leg
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_CALL = "short_call"
    SHORT_PUT = "short_put"

    # Vertical spreads
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"

    # Neutral strategies
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"

    # Multi-leg
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    LONG_BUTTERFLY = "long_butterfly"
    SHORT_BUTTERFLY = "short_butterfly"

    # Income strategies
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    THE_WHEEL = "the_wheel"

    # Calendar
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"


@dataclass
class OptionLeg:
    """Represents a single leg of an options strategy."""

    symbol: str
    underlying: str
    option_type: str  # "call" or "put"
    strike: float
    expiry: datetime
    quantity: int  # Positive for long, negative for short
    entry_price: float
    current_price: float = 0.0

    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    implied_volatility: float = 0.0

    def payoff_at_expiry(self, underlying_price: float) -> float:
        """Calculate payoff at expiration."""
        if self.option_type == "call":
            intrinsic = max(underlying_price - self.strike, 0)
        else:
            intrinsic = max(self.strike - underlying_price, 0)

        return intrinsic * self.quantity * 100  # Per contract

    def current_pnl(self) -> float:
        """Calculate current P&L."""
        return (self.current_price - self.entry_price) * self.quantity * 100

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def days_to_expiry(self) -> int:
        """Days until expiration."""
        return (self.expiry - datetime.now()).days

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "underlying": self.underlying,
            "type": self.option_type,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "pnl": self.current_pnl(),
        }


@dataclass
class PortfolioGreeks:
    """Aggregated Greeks for a portfolio or strategy."""

    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
        }


@dataclass
class MultiLegStrategy:
    """
    Represents a multi-leg options strategy.

    Aggregates Greeks, calculates P&L, and provides risk metrics.
    """

    name: str
    strategy_type: StrategyType
    legs: list[OptionLeg] = field(default_factory=list)
    underlying_symbol: str = ""
    underlying_price: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)

    # Risk parameters
    max_loss: float | None = None
    max_profit: float | None = None
    breakeven_prices: list[float] = field(default_factory=list)

    def add_leg(self, leg: OptionLeg) -> None:
        """Add a leg to the strategy."""
        self.legs.append(leg)
        if not self.underlying_symbol:
            self.underlying_symbol = leg.underlying

    def get_portfolio_greeks(self) -> PortfolioGreeks:
        """Calculate aggregate Greeks for the strategy."""
        return PortfolioGreeks(
            delta=sum(leg.delta * leg.quantity for leg in self.legs),
            gamma=sum(leg.gamma * leg.quantity for leg in self.legs),
            theta=sum(leg.theta * leg.quantity for leg in self.legs),
            vega=sum(leg.vega * leg.quantity for leg in self.legs),
            rho=sum(leg.rho * leg.quantity for leg in self.legs),
        )

    def total_pnl(self) -> float:
        """Calculate total P&L across all legs."""
        return sum(leg.current_pnl() for leg in self.legs)

    def net_premium(self) -> float:
        """Calculate net premium paid/received."""
        return sum(leg.entry_price * leg.quantity * 100 for leg in self.legs)

    def payoff_at_price(self, underlying_price: float) -> float:
        """Calculate strategy payoff at a given underlying price."""
        total_payoff = sum(leg.payoff_at_expiry(underlying_price) for leg in self.legs)
        return total_payoff - self.net_premium()

    def calculate_breakevens(self, price_range: tuple[float, float] = None) -> list[float]:
        """Find breakeven prices for the strategy."""
        if price_range is None:
            # Default to ±50% from current underlying
            low = self.underlying_price * 0.5
            high = self.underlying_price * 1.5
        else:
            low, high = price_range

        breakevens = []
        step = (high - low) / 1000

        prev_payoff = self.payoff_at_price(low)
        price = low + step

        while price <= high:
            current_payoff = self.payoff_at_price(price)

            # Check for sign change (crossing zero)
            if prev_payoff * current_payoff < 0:
                # Linear interpolation to find exact breakeven
                breakeven = price - step * current_payoff / (current_payoff - prev_payoff)
                breakevens.append(round(breakeven, 2))

            prev_payoff = current_payoff
            price += step

        self.breakeven_prices = breakevens
        return breakevens

    def calculate_risk_reward(self) -> dict[str, float]:
        """Calculate max profit, max loss, and risk/reward ratio."""
        # Sample payoffs across price range
        low = self.underlying_price * 0.5
        high = self.underlying_price * 1.5
        step = (high - low) / 100

        payoffs = []
        price = low
        while price <= high:
            payoffs.append(self.payoff_at_price(price))
            price += step

        self.max_profit = max(payoffs)
        self.max_loss = min(payoffs)

        # Risk/reward ratio
        if self.max_loss < 0 and self.max_profit > 0:
            risk_reward = abs(self.max_profit / self.max_loss)
        else:
            risk_reward = 0.0

        return {
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "risk_reward_ratio": risk_reward,
            "breakevens": self.breakeven_prices,
        }

    def get_margin_requirement(self) -> float:
        """
        Estimate margin requirement for the strategy.

        This is a simplified calculation - actual margin depends on broker.
        """
        margin = 0.0

        for leg in self.legs:
            if leg.is_short:
                if leg.option_type == "put":
                    # Cash-secured put: strike * 100
                    margin += leg.strike * 100 * abs(leg.quantity)
                else:
                    # Short call: typically 20% of underlying + premium
                    margin += self.underlying_price * 100 * 0.20 * abs(leg.quantity)

        # Reduce margin for defined-risk strategies
        if self.strategy_type in (
            StrategyType.IRON_CONDOR,
            StrategyType.BULL_CALL_SPREAD,
            StrategyType.BEAR_CALL_SPREAD,
            StrategyType.BULL_PUT_SPREAD,
            StrategyType.BEAR_PUT_SPREAD,
        ):
            # Max loss is the margin requirement for spreads
            if self.max_loss is not None:
                margin = abs(self.max_loss)

        return margin

    def to_dict(self) -> dict[str, Any]:
        """Convert strategy to dictionary."""
        greeks = self.get_portfolio_greeks()
        risk = self.calculate_risk_reward()

        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "underlying": self.underlying_symbol,
            "underlying_price": self.underlying_price,
            "legs": [leg.to_dict() for leg in self.legs],
            "greeks": greeks.to_dict(),
            "total_pnl": self.total_pnl(),
            "net_premium": self.net_premium(),
            "risk_reward": risk,
            "margin_requirement": self.get_margin_requirement(),
        }


class StrategyBuilder:
    """
    Factory class for building common options strategies.

    Creates properly configured MultiLegStrategy objects.
    """

    @staticmethod
    def create_vertical_spread(
        underlying: str,
        underlying_price: float,
        option_type: str,  # "call" or "put"
        long_strike: float,
        short_strike: float,
        expiry: datetime,
        long_price: float,
        short_price: float,
        quantity: int = 1,
    ) -> MultiLegStrategy:
        """Create a vertical spread (bull/bear call/put spread)."""
        # Determine strategy type
        if option_type == "call":
            if long_strike < short_strike:
                strategy_type = StrategyType.BULL_CALL_SPREAD
                name = f"Bull Call Spread {underlying}"
            else:
                strategy_type = StrategyType.BEAR_CALL_SPREAD
                name = f"Bear Call Spread {underlying}"
        else:
            if long_strike > short_strike:
                strategy_type = StrategyType.BULL_PUT_SPREAD
                name = f"Bull Put Spread {underlying}"
            else:
                strategy_type = StrategyType.BEAR_PUT_SPREAD
                name = f"Bear Put Spread {underlying}"

        strategy = MultiLegStrategy(
            name=name,
            strategy_type=strategy_type,
            underlying_symbol=underlying,
            underlying_price=underlying_price,
        )

        # Long leg
        strategy.add_leg(
            OptionLeg(
                symbol=f"{underlying}_{expiry.strftime('%y%m%d')}_{option_type[0].upper()}{long_strike}",
                underlying=underlying,
                option_type=option_type,
                strike=long_strike,
                expiry=expiry,
                quantity=quantity,
                entry_price=long_price,
                current_price=long_price,
            )
        )

        # Short leg
        strategy.add_leg(
            OptionLeg(
                symbol=f"{underlying}_{expiry.strftime('%y%m%d')}_{option_type[0].upper()}{short_strike}",
                underlying=underlying,
                option_type=option_type,
                strike=short_strike,
                expiry=expiry,
                quantity=-quantity,
                entry_price=short_price,
                current_price=short_price,
            )
        )

        strategy.calculate_breakevens()
        strategy.calculate_risk_reward()

        return strategy

    @staticmethod
    def create_iron_condor(
        underlying: str,
        underlying_price: float,
        expiry: datetime,
        put_long_strike: float,
        put_short_strike: float,
        call_short_strike: float,
        call_long_strike: float,
        put_long_price: float,
        put_short_price: float,
        call_short_price: float,
        call_long_price: float,
        quantity: int = 1,
    ) -> MultiLegStrategy:
        """Create an iron condor strategy."""
        strategy = MultiLegStrategy(
            name=f"Iron Condor {underlying}",
            strategy_type=StrategyType.IRON_CONDOR,
            underlying_symbol=underlying,
            underlying_price=underlying_price,
        )

        # Long put (protection)
        strategy.add_leg(
            OptionLeg(
                symbol=f"{underlying}_{expiry.strftime('%y%m%d')}_P{put_long_strike}",
                underlying=underlying,
                option_type="put",
                strike=put_long_strike,
                expiry=expiry,
                quantity=quantity,
                entry_price=put_long_price,
                current_price=put_long_price,
            )
        )

        # Short put
        strategy.add_leg(
            OptionLeg(
                symbol=f"{underlying}_{expiry.strftime('%y%m%d')}_P{put_short_strike}",
                underlying=underlying,
                option_type="put",
                strike=put_short_strike,
                expiry=expiry,
                quantity=-quantity,
                entry_price=put_short_price,
                current_price=put_short_price,
            )
        )

        # Short call
        strategy.add_leg(
            OptionLeg(
                symbol=f"{underlying}_{expiry.strftime('%y%m%d')}_C{call_short_strike}",
                underlying=underlying,
                option_type="call",
                strike=call_short_strike,
                expiry=expiry,
                quantity=-quantity,
                entry_price=call_short_price,
                current_price=call_short_price,
            )
        )

        # Long call (protection)
        strategy.add_leg(
            OptionLeg(
                symbol=f"{underlying}_{expiry.strftime('%y%m%d')}_C{call_long_strike}",
                underlying=underlying,
                option_type="call",
                strike=call_long_strike,
                expiry=expiry,
                quantity=quantity,
                entry_price=call_long_price,
                current_price=call_long_price,
            )
        )

        strategy.calculate_breakevens()
        strategy.calculate_risk_reward()

        return strategy

    @staticmethod
    def create_short_strangle(
        underlying: str,
        underlying_price: float,
        expiry: datetime,
        put_strike: float,
        call_strike: float,
        put_price: float,
        call_price: float,
        quantity: int = 1,
        target_delta: float = 0.16,
    ) -> MultiLegStrategy:
        """
        Create a short strangle (sell OTM put and call).

        Default target delta of 0.16 based on PyOptionTrader research.
        """
        strategy = MultiLegStrategy(
            name=f"Short Strangle {underlying}",
            strategy_type=StrategyType.SHORT_STRANGLE,
            underlying_symbol=underlying,
            underlying_price=underlying_price,
        )

        # Short put
        strategy.add_leg(
            OptionLeg(
                symbol=f"{underlying}_{expiry.strftime('%y%m%d')}_P{put_strike}",
                underlying=underlying,
                option_type="put",
                strike=put_strike,
                expiry=expiry,
                quantity=-quantity,
                entry_price=put_price,
                current_price=put_price,
                delta=-target_delta,  # Short put has positive delta exposure
            )
        )

        # Short call
        strategy.add_leg(
            OptionLeg(
                symbol=f"{underlying}_{expiry.strftime('%y%m%d')}_C{call_strike}",
                underlying=underlying,
                option_type="call",
                strike=call_strike,
                expiry=expiry,
                quantity=-quantity,
                entry_price=call_price,
                current_price=call_price,
                delta=target_delta,  # Short call has negative delta exposure
            )
        )

        strategy.calculate_breakevens()
        strategy.calculate_risk_reward()

        return strategy

    @staticmethod
    def create_straddle(
        underlying: str,
        underlying_price: float,
        expiry: datetime,
        strike: float,
        put_price: float,
        call_price: float,
        is_long: bool = True,
        quantity: int = 1,
    ) -> MultiLegStrategy:
        """Create a straddle (buy/sell ATM put and call)."""
        direction = 1 if is_long else -1
        strategy_type = StrategyType.LONG_STRADDLE if is_long else StrategyType.SHORT_STRADDLE

        strategy = MultiLegStrategy(
            name=f"{'Long' if is_long else 'Short'} Straddle {underlying}",
            strategy_type=strategy_type,
            underlying_symbol=underlying,
            underlying_price=underlying_price,
        )

        # Put leg
        strategy.add_leg(
            OptionLeg(
                symbol=f"{underlying}_{expiry.strftime('%y%m%d')}_P{strike}",
                underlying=underlying,
                option_type="put",
                strike=strike,
                expiry=expiry,
                quantity=quantity * direction,
                entry_price=put_price,
                current_price=put_price,
            )
        )

        # Call leg
        strategy.add_leg(
            OptionLeg(
                symbol=f"{underlying}_{expiry.strftime('%y%m%d')}_C{strike}",
                underlying=underlying,
                option_type="call",
                strike=strike,
                expiry=expiry,
                quantity=quantity * direction,
                entry_price=call_price,
                current_price=call_price,
            )
        )

        strategy.calculate_breakevens()
        strategy.calculate_risk_reward()

        return strategy

    @staticmethod
    def create_covered_call(
        underlying: str,
        underlying_price: float,
        shares: int,
        call_strike: float,
        call_expiry: datetime,
        call_price: float,
        share_entry_price: float,
    ) -> MultiLegStrategy:
        """Create a covered call strategy."""
        strategy = MultiLegStrategy(
            name=f"Covered Call {underlying}",
            strategy_type=StrategyType.COVERED_CALL,
            underlying_symbol=underlying,
            underlying_price=underlying_price,
        )

        contracts = shares // 100

        # Short call
        strategy.add_leg(
            OptionLeg(
                symbol=f"{underlying}_{call_expiry.strftime('%y%m%d')}_C{call_strike}",
                underlying=underlying,
                option_type="call",
                strike=call_strike,
                expiry=call_expiry,
                quantity=-contracts,
                entry_price=call_price,
                current_price=call_price,
            )
        )

        # Note: Stock position is tracked separately

        return strategy


def find_delta_strikes(
    option_chain: list[dict[str, Any]],
    target_delta: float,
    option_type: str,
) -> dict[str, Any] | None:
    """
    Find option contract closest to target delta.

    Based on PyOptionTrader's 16-delta selection.

    Args:
        option_chain: List of option contract dicts with 'delta' field
        target_delta: Target delta (e.g., 0.16 for 16-delta)
        option_type: "call" or "put"

    Returns:
        Contract dict closest to target delta
    """
    filtered = [c for c in option_chain if c.get("type") == option_type]

    if not filtered:
        return None

    # For puts, delta is negative; for calls, positive
    if option_type == "put":
        target = -abs(target_delta)
    else:
        target = abs(target_delta)

    closest = min(filtered, key=lambda c: abs(c.get("delta", 0) - target))

    return closest


__all__ = [
    "MultiLegStrategy",
    "OptionLeg",
    "PortfolioGreeks",
    "StrategyBuilder",
    "StrategyType",
    "find_delta_strikes",
]
