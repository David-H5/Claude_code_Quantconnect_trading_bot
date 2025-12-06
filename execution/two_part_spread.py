"""
Two-Part Spread Strategy Module

Specialized for legging into butterflies and iron condors in two parts:
1. Find underpriced DEBIT spread (wide spread = opportunity for good fill)
2. Match with CREDIT spread further OTM that covers the debit cost

Key insight: Wide bid-ask spreads are OPPORTUNITIES, not risks, when you can
get filled below mid on debit spreads.

Tracks ACTUAL fill rates during the trading session, not hypothetical estimates.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class SpreadType(Enum):
    """Type of spread."""

    CALL_DEBIT = "call_debit"  # Bull call spread (buy lower, sell higher)
    CALL_CREDIT = "call_credit"  # Bear call spread (sell lower, buy higher)
    PUT_DEBIT = "put_debit"  # Bear put spread (buy higher, sell lower)
    PUT_CREDIT = "put_credit"  # Bull put spread (sell higher, buy lower)


class PositionStatus(Enum):
    """Status of two-part position."""

    SCANNING = "scanning"  # Looking for debit spread
    DEBIT_PENDING = "debit_pending"  # Debit order submitted
    DEBIT_FILLED = "debit_filled"  # Debit filled, looking for credit
    CREDIT_PENDING = "credit_pending"  # Credit order submitted
    COMPLETE = "complete"  # Both legs filled
    PARTIAL = "partial"  # Only debit filled, no matching credit
    CANCELLED = "cancelled"


class FillLocation(Enum):
    """Where the fill occurred relative to quotes."""

    AT_BID = "at_bid"
    BELOW_MID = "below_mid"  # Better than mid for debit
    AT_MID = "at_mid"
    ABOVE_MID = "above_mid"  # Better than mid for credit
    AT_ASK = "at_ask"


@dataclass
class SpreadQuote:
    """Current quote for a spread."""

    symbol: str
    spread_type: SpreadType
    long_strike: float
    short_strike: float
    expiry: datetime

    # Prices
    bid: float  # Can sell spread at this price
    ask: float  # Can buy spread at this price
    mid: float = 0.0

    # Theoretical value (if available)
    theoretical_value: float = 0.0

    # Leg details
    long_leg_bid: float = 0.0
    long_leg_ask: float = 0.0
    short_leg_bid: float = 0.0
    short_leg_ask: float = 0.0

    # Metrics
    spread_width_bps: float = 0.0  # Bid-ask spread in bps
    discount_to_theo: float = 0.0  # % below theoretical (positive = cheap)

    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.mid == 0 and self.bid > 0 and self.ask > 0:
            self.mid = (self.bid + self.ask) / 2
        if self.mid > 0:
            self.spread_width_bps = (self.ask - self.bid) / self.mid * 10000

    @property
    def is_debit(self) -> bool:
        """True if this is a debit spread."""
        return self.spread_type in (SpreadType.CALL_DEBIT, SpreadType.PUT_DEBIT)

    @property
    def is_credit(self) -> bool:
        """True if this is a credit spread."""
        return self.spread_type in (SpreadType.CALL_CREDIT, SpreadType.PUT_CREDIT)

    @property
    def width(self) -> float:
        """Strike width of the spread."""
        return abs(self.long_strike - self.short_strike)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "type": self.spread_type.value,
            "long_strike": self.long_strike,
            "short_strike": self.short_strike,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
            "spread_width_bps": self.spread_width_bps,
            "theoretical": self.theoretical_value,
            "discount_pct": self.discount_to_theo,
        }


@dataclass
class SpreadFill:
    """Record of an actual spread fill."""

    timestamp: datetime
    symbol: str
    spread_type: SpreadType
    long_strike: float
    short_strike: float
    quantity: int
    fill_price: float

    # Quote at time of fill
    bid_at_fill: float
    ask_at_fill: float
    mid_at_fill: float

    # Fill quality
    fill_location: FillLocation
    improvement_vs_mid: float = 0.0  # Positive = got better price than mid
    time_to_fill_seconds: float = 0.0

    def __post_init__(self):
        # Calculate fill location
        if self.mid_at_fill > 0:
            self.improvement_vs_mid = (self.mid_at_fill - self.fill_price) / self.mid_at_fill

            # Determine location
            spread_width = self.ask_at_fill - self.bid_at_fill
            if spread_width > 0:
                position_in_spread = (self.fill_price - self.bid_at_fill) / spread_width
                if position_in_spread <= 0.1:
                    self.fill_location = FillLocation.AT_BID
                elif position_in_spread < 0.45:
                    self.fill_location = FillLocation.BELOW_MID
                elif position_in_spread <= 0.55:
                    self.fill_location = FillLocation.AT_MID
                elif position_in_spread < 0.9:
                    self.fill_location = FillLocation.ABOVE_MID
                else:
                    self.fill_location = FillLocation.AT_ASK


@dataclass
class SessionFillStats:
    """Actual fill statistics for the current trading session."""

    session_date: date
    total_orders: int = 0
    filled_orders: int = 0
    partial_fills: int = 0
    unfilled_orders: int = 0

    # By spread type
    debit_fills: int = 0
    debit_attempts: int = 0
    credit_fills: int = 0
    credit_attempts: int = 0

    # Fill quality
    fills_at_bid: int = 0
    fills_below_mid: int = 0
    fills_at_mid: int = 0
    fills_above_mid: int = 0
    fills_at_ask: int = 0

    total_improvement_bps: float = 0.0
    avg_time_to_fill_seconds: float = 0.0

    @property
    def fill_rate(self) -> float:
        """Overall fill rate for session."""
        if self.total_orders == 0:
            return 0.0
        return self.filled_orders / self.total_orders

    @property
    def debit_fill_rate(self) -> float:
        """Fill rate for debit spreads."""
        if self.debit_attempts == 0:
            return 0.0
        return self.debit_fills / self.debit_attempts

    @property
    def credit_fill_rate(self) -> float:
        """Fill rate for credit spreads."""
        if self.credit_attempts == 0:
            return 0.0
        return self.credit_fills / self.credit_attempts

    @property
    def avg_improvement_bps(self) -> float:
        """Average price improvement vs mid in bps."""
        if self.filled_orders == 0:
            return 0.0
        return self.total_improvement_bps / self.filled_orders

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.session_date.isoformat(),
            "total_orders": self.total_orders,
            "filled": self.filled_orders,
            "fill_rate": f"{self.fill_rate:.1%}",
            "debit_fill_rate": f"{self.debit_fill_rate:.1%}",
            "credit_fill_rate": f"{self.credit_fill_rate:.1%}",
            "avg_improvement_bps": self.avg_improvement_bps,
            "avg_time_to_fill": self.avg_time_to_fill_seconds,
            "fills_below_mid": self.fills_below_mid,
            "fills_at_mid": self.fills_at_mid,
        }


@dataclass
class DebitOpportunity:
    """An underpriced debit spread opportunity."""

    quote: SpreadQuote
    opportunity_score: float  # 0-100, higher = better opportunity
    reasons: list[str]
    suggested_limit_price: float
    max_price_to_pay: float  # Maximum acceptable price

    # Matching credit requirements
    min_credit_needed: float
    target_credit: float  # Ideal credit to collect

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quote": self.quote.to_dict(),
            "score": self.opportunity_score,
            "reasons": self.reasons,
            "limit_price": self.suggested_limit_price,
            "max_price": self.max_price_to_pay,
            "min_credit_needed": self.min_credit_needed,
            "target_credit": self.target_credit,
        }


@dataclass
class CreditMatch:
    """A credit spread that matches a debit opportunity."""

    quote: SpreadQuote
    match_quality: float  # 0-100, how well it matches
    net_credit: float  # Credit received minus debit paid
    max_risk: float  # Maximum loss on combined position
    reward_risk_ratio: float
    forms_butterfly: bool
    forms_condor: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quote": self.quote.to_dict(),
            "match_quality": self.match_quality,
            "net_credit": self.net_credit,
            "max_risk": self.max_risk,
            "reward_risk": self.reward_risk_ratio,
            "is_butterfly": self.forms_butterfly,
            "is_condor": self.forms_condor,
        }


@dataclass
class TwoPartPosition:
    """A two-part butterfly/condor position being built."""

    id: str
    symbol: str
    status: PositionStatus
    created_at: datetime

    # Debit leg
    debit_spread: SpreadQuote | None = None
    debit_order_price: float = 0.0
    debit_fill: SpreadFill | None = None
    debit_quantity: int = 0

    # Credit leg
    credit_spread: SpreadQuote | None = None
    credit_order_price: float = 0.0
    credit_fill: SpreadFill | None = None
    credit_quantity: int = 0

    # Combined position
    net_cost: float = 0.0  # Negative = net credit
    max_profit: float = 0.0
    max_loss: float = 0.0

    # Timing
    debit_filled_at: datetime | None = None
    credit_filled_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def is_complete(self) -> bool:
        """True if both legs are filled."""
        return self.status == PositionStatus.COMPLETE

    @property
    def time_between_legs_seconds(self) -> float | None:
        """Time between debit and credit fills."""
        if self.debit_filled_at and self.credit_filled_at:
            return (self.credit_filled_at - self.debit_filled_at).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "status": self.status.value,
            "debit": self.debit_spread.to_dict() if self.debit_spread else None,
            "debit_fill_price": self.debit_fill.fill_price if self.debit_fill else None,
            "credit": self.credit_spread.to_dict() if self.credit_spread else None,
            "credit_fill_price": self.credit_fill.fill_price if self.credit_fill else None,
            "net_cost": self.net_cost,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "time_between_legs": self.time_between_legs_seconds,
        }


class TwoPartSpreadStrategy:
    """
    Manages two-part spread strategy for legging into butterflies/condors.

    Strategy:
    1. Scan for underpriced debit spreads (wide spread = opportunity)
    2. Place debit order below mid price
    3. Once filled, find matching credit spread further OTM
    4. Place credit order to achieve net credit or zero cost

    Tracks ACTUAL fill rates during the trading session.
    """

    def __init__(
        self,
        min_discount_pct: float = 0.05,  # Min 5% below theoretical
        max_spread_width_bps: float = 500,  # Max 5% spread width
        min_credit_ratio: float = 1.0,  # Credit must be >= debit
        target_credit_ratio: float = 1.1,  # Aim for 10% more credit
        fill_callback: Callable[[SpreadFill], None] | None = None,
    ):
        """
        Initialize strategy.

        Args:
            min_discount_pct: Minimum discount to theoretical for debit opportunity
            max_spread_width_bps: Maximum bid-ask spread to consider
            min_credit_ratio: Minimum credit/debit ratio
            target_credit_ratio: Target credit/debit ratio
            fill_callback: Callback when fill occurs
        """
        self.min_discount_pct = min_discount_pct
        self.max_spread_width_bps = max_spread_width_bps
        self.min_credit_ratio = min_credit_ratio
        self.target_credit_ratio = target_credit_ratio
        self.fill_callback = fill_callback

        # Session tracking
        self.session_stats = SessionFillStats(session_date=date.today())
        self.fill_history: list[SpreadFill] = []

        # Active positions
        self.positions: dict[str, TwoPartPosition] = {}
        self.pending_debits: list[TwoPartPosition] = []
        self.completed_positions: list[TwoPartPosition] = []

        # Order tracking
        self.pending_orders: dict[str, dict] = {}  # order_id -> order details
        self._order_counter = 0

    def reset_session(self) -> SessionFillStats:
        """
        Reset session statistics for new trading day.

        Returns previous session stats.
        """
        old_stats = self.session_stats
        self.session_stats = SessionFillStats(session_date=date.today())
        self.fill_history = []
        return old_stats

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"TPS_{date.today().strftime('%Y%m%d')}_{self._order_counter:04d}"

    def scan_debit_opportunities(
        self,
        quotes: list[SpreadQuote],
        symbol: str | None = None,
    ) -> list[DebitOpportunity]:
        """
        Scan for underpriced debit spread opportunities.

        Wide spreads are treated as OPPORTUNITIES (can get filled below mid)
        not as risks to avoid.

        Args:
            quotes: List of current spread quotes
            symbol: Filter to specific symbol

        Returns:
            List of opportunities sorted by score
        """
        opportunities = []

        for quote in quotes:
            # Filter
            if symbol and quote.symbol != symbol:
                continue
            if not quote.is_debit:
                continue
            if quote.ask <= 0:
                continue

            reasons = []
            score = 50.0  # Base score

            # Check spread width - WIDER is actually better for getting good fills
            if quote.spread_width_bps > self.max_spread_width_bps:
                # Too wide even for our strategy
                continue

            # Moderate spread width (50-200 bps) is ideal
            if 50 <= quote.spread_width_bps <= 200:
                score += 15
                reasons.append(f"Good spread width ({quote.spread_width_bps:.0f} bps) - room for price improvement")
            elif quote.spread_width_bps > 200:
                score += 10
                reasons.append(f"Wide spread ({quote.spread_width_bps:.0f} bps) - high improvement potential")

            # Check discount to theoretical
            if quote.theoretical_value > 0:
                discount = (quote.theoretical_value - quote.ask) / quote.theoretical_value
                if discount >= self.min_discount_pct:
                    score += 20 * (discount / 0.10)  # Scale by discount
                    reasons.append(f"Trading {discount:.1%} below theoretical")
                    quote.discount_to_theo = discount

            # Check if ask is close to bid (can get filled near bid)
            bid_ask_ratio = quote.bid / quote.ask if quote.ask > 0 else 0
            if bid_ask_ratio > 0.85:
                score += 10
                reasons.append("Tight market - likely quick fill")
            elif bid_ask_ratio > 0.70:
                score += 5
                reasons.append("Reasonable market depth")

            # Calculate suggested prices
            # For debits, we want to pay LESS than mid
            suggested_limit = quote.bid + (quote.mid - quote.bid) * 0.3  # 30% of way from bid to mid
            max_price = quote.mid  # Never pay more than mid

            # Calculate credit requirements
            min_credit = quote.ask * self.min_credit_ratio
            target_credit = quote.ask * self.target_credit_ratio

            if score >= 50 and reasons:
                opportunities.append(
                    DebitOpportunity(
                        quote=quote,
                        opportunity_score=min(100, score),
                        reasons=reasons,
                        suggested_limit_price=suggested_limit,
                        max_price_to_pay=max_price,
                        min_credit_needed=min_credit,
                        target_credit=target_credit,
                    )
                )

        # Sort by score descending
        opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)
        return opportunities

    def find_credit_matches(
        self,
        debit: DebitOpportunity,
        credit_quotes: list[SpreadQuote],
    ) -> list[CreditMatch]:
        """
        Find credit spreads that match a debit opportunity.

        Credit spread should be further OTM and provide enough credit
        to cover the debit cost.

        Args:
            debit: The debit opportunity to match
            credit_quotes: Available credit spread quotes

        Returns:
            List of matches sorted by quality
        """
        matches = []
        debit_quote = debit.quote

        for credit in credit_quotes:
            if credit.symbol != debit_quote.symbol:
                continue
            if not credit.is_credit:
                continue
            if credit.bid <= 0:
                continue

            # Check expiry matches
            if credit.expiry != debit_quote.expiry:
                continue

            # Credit should be further OTM
            is_valid_condor = False
            is_valid_butterfly = False

            if debit_quote.spread_type == SpreadType.CALL_DEBIT:
                # Call debit = bullish side
                # Need bear call credit with higher strikes
                if credit.spread_type == SpreadType.CALL_CREDIT:
                    if credit.short_strike > debit_quote.short_strike:
                        is_valid_condor = True
                    if credit.short_strike == debit_quote.short_strike:
                        is_valid_butterfly = True

            elif debit_quote.spread_type == SpreadType.PUT_DEBIT:
                # Put debit = bearish side
                # Need bull put credit with lower strikes
                if credit.spread_type == SpreadType.PUT_CREDIT:
                    if credit.short_strike < debit_quote.short_strike:
                        is_valid_condor = True
                    if credit.short_strike == debit_quote.short_strike:
                        is_valid_butterfly = True

            if not (is_valid_condor or is_valid_butterfly):
                continue

            # Calculate net credit/debit
            # Using realistic fill prices: debit at suggested, credit at mid
            debit_cost = debit.suggested_limit_price
            credit_received = credit.mid  # Conservative estimate
            net_credit = credit_received - debit_cost

            # Check if meets minimum credit requirement
            if credit.bid < debit.min_credit_needed:
                continue

            # Calculate max risk
            debit_width = debit_quote.width
            credit_width = credit.width

            if is_valid_butterfly:
                max_risk = max(debit_width, credit_width) - abs(net_credit)
            else:  # Condor
                max_risk = max(debit_width, credit_width) - abs(net_credit)

            max_profit = abs(net_credit) if net_credit > 0 else min(debit_width, credit_width)

            # Calculate match quality
            quality = 50.0

            # Higher credit is better
            credit_ratio = credit.bid / debit_cost if debit_cost > 0 else 0
            if credit_ratio >= self.target_credit_ratio:
                quality += 25
            elif credit_ratio >= self.min_credit_ratio:
                quality += 15

            # Net credit is ideal
            if net_credit > 0:
                quality += 20
            elif net_credit >= -0.05 * debit_cost:  # Within 5% of break-even
                quality += 10

            # Reward/risk ratio
            rr_ratio = max_profit / max_risk if max_risk > 0 else 0
            if rr_ratio > 1.0:
                quality += 10

            matches.append(
                CreditMatch(
                    quote=credit,
                    match_quality=min(100, quality),
                    net_credit=net_credit,
                    max_risk=max_risk,
                    reward_risk_ratio=rr_ratio,
                    forms_butterfly=is_valid_butterfly,
                    forms_condor=is_valid_condor,
                )
            )

        # Sort by match quality
        matches.sort(key=lambda x: x.match_quality, reverse=True)
        return matches

    def create_position(
        self,
        debit: DebitOpportunity,
        credit: CreditMatch | None = None,
        quantity: int = 1,
    ) -> TwoPartPosition:
        """
        Create a new two-part position.

        Args:
            debit: Debit opportunity to enter
            credit: Optional pre-selected credit match
            quantity: Number of spreads

        Returns:
            New TwoPartPosition
        """
        position_id = self._generate_order_id()

        position = TwoPartPosition(
            id=position_id,
            symbol=debit.quote.symbol,
            status=PositionStatus.SCANNING,
            created_at=datetime.now(),
            debit_spread=debit.quote,
            debit_order_price=debit.suggested_limit_price,
            debit_quantity=quantity,
        )

        if credit:
            position.credit_spread = credit.quote
            position.credit_order_price = credit.quote.mid
            position.credit_quantity = quantity
            position.net_cost = position.debit_order_price - credit.quote.mid
            position.max_profit = credit.net_credit if credit.net_credit > 0 else debit.quote.width
            position.max_loss = credit.max_risk

        self.positions[position_id] = position
        return position

    def submit_debit_order(
        self,
        position_id: str,
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Submit debit spread order.

        Args:
            position_id: Position ID
            limit_price: Override limit price

        Returns:
            Order details
        """
        if position_id not in self.positions:
            return {"error": "Position not found"}

        position = self.positions[position_id]

        if position.status != PositionStatus.SCANNING:
            return {"error": f"Invalid status: {position.status.value}"}

        price = limit_price or position.debit_order_price

        order = {
            "order_id": f"{position_id}_DEBIT",
            "position_id": position_id,
            "type": "debit",
            "symbol": position.symbol,
            "spread": position.debit_spread.to_dict() if position.debit_spread else None,
            "quantity": position.debit_quantity,
            "limit_price": price,
            "submitted_at": datetime.now(),
            "bid_at_submit": position.debit_spread.bid if position.debit_spread else 0,
            "ask_at_submit": position.debit_spread.ask if position.debit_spread else 0,
        }

        position.status = PositionStatus.DEBIT_PENDING
        position.debit_order_price = price

        self.pending_orders[order["order_id"]] = order
        self.session_stats.total_orders += 1
        self.session_stats.debit_attempts += 1

        return order

    def submit_credit_order(
        self,
        position_id: str,
        credit_spread: SpreadQuote,
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Submit credit spread order.

        Args:
            position_id: Position ID
            credit_spread: Credit spread to sell
            limit_price: Override limit price

        Returns:
            Order details
        """
        if position_id not in self.positions:
            return {"error": "Position not found"}

        position = self.positions[position_id]

        if position.status != PositionStatus.DEBIT_FILLED:
            return {"error": f"Invalid status: {position.status.value}"}

        # Default to mid price for credit
        price = limit_price or credit_spread.mid

        order = {
            "order_id": f"{position_id}_CREDIT",
            "position_id": position_id,
            "type": "credit",
            "symbol": position.symbol,
            "spread": credit_spread.to_dict(),
            "quantity": position.credit_quantity,
            "limit_price": price,
            "submitted_at": datetime.now(),
            "bid_at_submit": credit_spread.bid,
            "ask_at_submit": credit_spread.ask,
        }

        position.status = PositionStatus.CREDIT_PENDING
        position.credit_spread = credit_spread
        position.credit_order_price = price

        self.pending_orders[order["order_id"]] = order
        self.session_stats.total_orders += 1
        self.session_stats.credit_attempts += 1

        return order

    def record_fill(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: int,
        current_bid: float,
        current_ask: float,
    ) -> SpreadFill | None:
        """
        Record an actual fill from the broker.

        This updates ACTUAL fill statistics, not hypothetical.

        Args:
            order_id: Order ID that was filled
            fill_price: Actual fill price
            fill_quantity: Quantity filled
            current_bid: Bid at time of fill
            current_ask: Ask at time of fill

        Returns:
            SpreadFill record
        """
        if order_id not in self.pending_orders:
            return None

        order = self.pending_orders[order_id]
        position_id = order["position_id"]

        if position_id not in self.positions:
            return None

        position = self.positions[position_id]

        current_mid = (current_bid + current_ask) / 2
        time_to_fill = (datetime.now() - order["submitted_at"]).total_seconds()

        # Determine spread type from position
        is_debit = order["type"] == "debit"
        spread_info = position.debit_spread if is_debit else position.credit_spread

        fill = SpreadFill(
            timestamp=datetime.now(),
            symbol=position.symbol,
            spread_type=spread_info.spread_type if spread_info else SpreadType.CALL_DEBIT,
            long_strike=spread_info.long_strike if spread_info else 0,
            short_strike=spread_info.short_strike if spread_info else 0,
            quantity=fill_quantity,
            fill_price=fill_price,
            bid_at_fill=current_bid,
            ask_at_fill=current_ask,
            mid_at_fill=current_mid,
            time_to_fill_seconds=time_to_fill,
            fill_location=FillLocation.AT_MID,  # Will be recalculated in __post_init__
        )

        # Update session stats
        self.session_stats.filled_orders += 1
        self.session_stats.total_improvement_bps += fill.improvement_vs_mid * 10000

        # Update cumulative average time to fill
        n = self.session_stats.filled_orders
        old_avg = self.session_stats.avg_time_to_fill_seconds
        self.session_stats.avg_time_to_fill_seconds = old_avg + (time_to_fill - old_avg) / n

        # Track fill location
        if fill.fill_location == FillLocation.AT_BID:
            self.session_stats.fills_at_bid += 1
        elif fill.fill_location == FillLocation.BELOW_MID:
            self.session_stats.fills_below_mid += 1
        elif fill.fill_location == FillLocation.AT_MID:
            self.session_stats.fills_at_mid += 1
        elif fill.fill_location == FillLocation.ABOVE_MID:
            self.session_stats.fills_above_mid += 1
        else:
            self.session_stats.fills_at_ask += 1

        # Update position
        if is_debit:
            position.debit_fill = fill
            position.debit_filled_at = datetime.now()
            position.status = PositionStatus.DEBIT_FILLED
            self.session_stats.debit_fills += 1
            self.pending_debits.append(position)
        else:
            position.credit_fill = fill
            position.credit_filled_at = datetime.now()
            position.status = PositionStatus.COMPLETE
            position.completed_at = datetime.now()
            self.session_stats.credit_fills += 1

            # Calculate final position metrics
            if position.debit_fill:
                position.net_cost = position.debit_fill.fill_price - fill.fill_price

            self.completed_positions.append(position)
            if position in self.pending_debits:
                self.pending_debits.remove(position)

        # Store fill and remove pending order
        self.fill_history.append(fill)
        del self.pending_orders[order_id]

        # Callback
        if self.fill_callback:
            self.fill_callback(fill)

        return fill

    def record_unfilled(self, order_id: str) -> None:
        """Record that an order was not filled (cancelled or expired)."""
        if order_id not in self.pending_orders:
            return

        order = self.pending_orders[order_id]
        self.session_stats.unfilled_orders += 1

        del self.pending_orders[order_id]

    def get_session_fill_rate(self) -> float:
        """Get actual fill rate for current session."""
        return self.session_stats.fill_rate

    def should_attempt_trade(
        self,
        min_fill_rate: float = 0.25,
    ) -> tuple[bool, str]:
        """
        Check if we should attempt a trade based on ACTUAL session fill rate.

        Args:
            min_fill_rate: Minimum acceptable fill rate

        Returns:
            (should_trade, reason)
        """
        stats = self.session_stats

        # Need some data to make a decision
        if stats.total_orders < 3:
            return True, "Insufficient data - proceeding with trade"

        if stats.fill_rate >= min_fill_rate:
            return True, f"Session fill rate {stats.fill_rate:.1%} meets threshold"
        else:
            return False, f"Session fill rate {stats.fill_rate:.1%} below {min_fill_rate:.1%} threshold"

    def submit_debit_spread_order_qc(
        self,
        algorithm,
        opportunity: DebitOpportunity,
        long_contract_symbol,
        short_contract_symbol,
        quantity: int = 1,
    ) -> str | None:
        """
        Submit debit spread order using QuantConnect ComboOrder.

        INTEGRATION: Use QuantConnect's ComboLimitOrder for atomic execution
        of multi-leg spreads. This ensures both legs fill together.

        CRITICAL: Must pass actual Symbol objects from option chain, not strings!

        Note: From QuantConnect GitHub, LEAN has 24 files dedicated to
        multi-leg strategy matching for ComboOrders.

        Example:
            def OnData(self, slice):
                # Get option chain (Python API uses snake_case)
                chain = slice.option_chains.get(self.option_symbol)
                if chain:
                    # Find specific contracts by strike
                    long_contract = None
                    short_contract = None
                    for contract in chain:
                        if contract.Strike == desired_long_strike:
                            long_contract = contract.Symbol  # Symbol object
                        if contract.Strike == desired_short_strike:
                            short_contract = contract.Symbol  # Symbol object

                    if long_contract and short_contract:
                        self.strategy.submit_debit_spread_order_qc(
                            self, opportunity, long_contract, short_contract, quantity=1
                        )

        Args:
            algorithm: QCAlgorithm instance
            opportunity: Debit opportunity to execute
            long_contract_symbol: Symbol object for the long leg (from option chain)
            short_contract_symbol: Symbol object for the short leg (from option chain)
            quantity: Number of spreads (contracts)

        Returns:
            Order ID or None if submission failed
        """
        try:
            from AlgorithmImports import Leg

            # CRITICAL: Leg.Create() requires Symbol objects, not strings
            # The symbols must come from the actual option chain contracts
            legs = []
            legs.append(Leg.Create(long_contract_symbol, 1))  # Buy
            legs.append(Leg.Create(short_contract_symbol, -1))  # Sell

            # Submit ComboLimitOrder for atomic execution
            # Note: ComboLimitOrder signature from LEAN uses positional params
            ticket = algorithm.ComboLimitOrder(legs, quantity, opportunity.suggested_limit_price)

            order_id = str(ticket.OrderId)

            # Track order
            self.pending_orders[order_id] = {
                "type": "debit",
                "opportunity": opportunity,
                "quantity": quantity,
                "submitted_at": datetime.now(),
            }

            self.session_stats.total_orders += 1
            self.session_stats.debit_attempts += 1

            algorithm.Debug(
                f"Submitted debit spread order {order_id}: " f"debit spread @ ${opportunity.suggested_limit_price:.2f}"
            )

            return order_id

        except ImportError:
            algorithm.Debug("ComboOrder not available - need AlgorithmImports")
            return None
        except Exception as e:
            algorithm.Debug(f"Error submitting debit spread order: {e}")
            return None

    def submit_credit_spread_order_qc(
        self,
        algorithm,
        credit_match: CreditMatch,
        short_contract_symbol,
        long_contract_symbol,
        debit_position: TwoPartPosition,
        quantity: int = 1,
    ) -> str | None:
        """
        Submit credit spread order using QuantConnect ComboOrder.

        INTEGRATION: Use ComboLimitOrder for atomic execution.

        CRITICAL: Must pass actual Symbol objects from option chain, not strings!

        Example:
            # Find contracts in option chain
            for contract in chain:
                if contract.Strike == short_strike:
                    short_symbol = contract.Symbol
                if contract.Strike == long_strike:
                    long_symbol = contract.Symbol

            # Submit credit spread
            self.strategy.submit_credit_spread_order_qc(
                self, credit_match, short_symbol, long_symbol, position, quantity=1
            )

        Args:
            algorithm: QCAlgorithm instance
            credit_match: Credit spread to execute
            short_contract_symbol: Symbol object for the short leg (from option chain)
            long_contract_symbol: Symbol object for the long leg (from option chain)
            debit_position: The debit position being matched
            quantity: Number of spreads

        Returns:
            Order ID or None if submission failed
        """
        try:
            from AlgorithmImports import Leg

            quote = credit_match.quote

            # Create legs for credit spread using Symbol objects
            legs = []
            legs.append(Leg.Create(short_contract_symbol, -1))  # Sell
            legs.append(Leg.Create(long_contract_symbol, 1))  # Buy

            # For credit spreads, we receive premium
            # So limit price is the MINIMUM credit we want to receive
            # Note: ComboLimitOrder signature uses positional params
            ticket = algorithm.ComboLimitOrder(
                legs,
                quantity,
                quote.mid,  # Conservative: aim for mid
            )

            order_id = str(ticket.OrderId)

            # Track order
            self.pending_orders[order_id] = {
                "type": "credit",
                "credit_match": credit_match,
                "debit_position_id": debit_position.id,
                "quantity": quantity,
                "submitted_at": datetime.now(),
            }

            self.session_stats.total_orders += 1
            self.session_stats.credit_attempts += 1

            algorithm.Debug(
                f"Submitted credit spread order {order_id}: " f"{quote.spread_type.value} @ ${quote.mid:.2f}"
            )

            return order_id

        except ImportError:
            algorithm.Debug("ComboOrder not available - need AlgorithmImports")
            return None
        except Exception as e:
            algorithm.Debug(f"Error submitting credit spread order: {e}")
            return None

    def handle_order_event_qc(self, algorithm, order_event) -> None:
        """
        Handle order fill events from QuantConnect.

        INTEGRATION: Call this from algorithm.OnOrderEvent()

        Example:
            def OnOrderEvent(self, order_event):
                from AlgorithmImports import OrderStatus

                if order_event.Status == OrderStatus.Filled:
                    self.strategy.handle_order_event_qc(self, order_event)

        Args:
            algorithm: QCAlgorithm instance
            order_event: OrderEvent from OnOrderEvent
        """
        try:
            from AlgorithmImports import OrderStatus

            if order_event.Status != OrderStatus.Filled:
                return

            order_id = str(order_event.OrderId)

            if order_id not in self.pending_orders:
                return

            order_details = self.pending_orders[order_id]
            order_type = order_details["type"]

            # Create fill record
            fill = SpreadFill(
                timestamp=datetime.now(),
                symbol=order_event.Symbol.Value if hasattr(order_event.Symbol, "Value") else str(order_event.Symbol),
                spread_type=order_details.get("opportunity", order_details.get("credit_match")).quote.spread_type,
                long_strike=0.0,  # Extracted from legs
                short_strike=0.0,
                quantity=order_event.FillQuantity,
                fill_price=order_event.FillPrice,
                bid_at_fill=0.0,  # Would need to track from slice
                ask_at_fill=0.0,
                mid_at_fill=order_event.FillPrice,  # Approximation
            )

            # Update session stats
            self.session_stats.filled_orders += 1

            if order_type == "debit":
                self.session_stats.debit_fills += 1
                algorithm.Debug(f"Debit spread filled: {order_id} @ ${order_event.FillPrice:.2f}")

                # Trigger callback
                if self.fill_callback:
                    self.fill_callback(fill)

            elif order_type == "credit":
                self.session_stats.credit_fills += 1
                algorithm.Debug(f"Credit spread filled: {order_id} @ ${order_event.FillPrice:.2f}")

                # Trigger callback
                if self.fill_callback:
                    self.fill_callback(fill)

            # Add to history
            self.fill_history.append(fill)

            # Remove from pending
            del self.pending_orders[order_id]

        except ImportError:
            algorithm.Debug("OrderStatus not available - need AlgorithmImports")
        except Exception as e:
            algorithm.Debug(f"Error handling order event: {e}")

    def get_pending_debit_positions(self) -> list[TwoPartPosition]:
        """Get positions with filled debit awaiting credit."""
        return [p for p in self.pending_debits if p.status == PositionStatus.DEBIT_FILLED]

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive strategy summary."""
        return {
            "session": self.session_stats.to_dict(),
            "active_positions": len(self.positions),
            "pending_debits": len(self.pending_debits),
            "completed_today": len(self.completed_positions),
            "pending_orders": len(self.pending_orders),
            "recent_fills": [f.fill_price for f in self.fill_history[-5:]],
        }

    def get_llm_summary(self) -> str:
        """Generate LLM-ready summary."""
        stats = self.session_stats

        text = f"""
TWO-PART SPREAD STRATEGY - SESSION SUMMARY
==========================================
Date: {stats.session_date}

ACTUAL FILL STATISTICS
----------------------
Total Orders: {stats.total_orders}
Filled: {stats.filled_orders} ({stats.fill_rate:.1%})
Debit Fill Rate: {stats.debit_fill_rate:.1%} ({stats.debit_fills}/{stats.debit_attempts})
Credit Fill Rate: {stats.credit_fill_rate:.1%} ({stats.credit_fills}/{stats.credit_attempts})

FILL QUALITY
------------
Average Improvement vs Mid: {stats.avg_improvement_bps:.1f} bps
Average Time to Fill: {stats.avg_time_to_fill_seconds:.1f} seconds

Fill Locations:
  At Bid: {stats.fills_at_bid}
  Below Mid: {stats.fills_below_mid}
  At Mid: {stats.fills_at_mid}
  Above Mid: {stats.fills_above_mid}
  At Ask: {stats.fills_at_ask}

POSITIONS
---------
Active: {len(self.positions)}
Pending Debits (awaiting credit): {len(self.pending_debits)}
Completed Today: {len(self.completed_positions)}
"""
        return text


def create_two_part_strategy(
    min_discount_pct: float = 0.05,
    max_spread_width_bps: float = 500,
    min_credit_ratio: float = 1.0,
    target_credit_ratio: float = 1.1,
    fill_callback: Callable[[SpreadFill], None] | None = None,
) -> TwoPartSpreadStrategy:
    """Create two-part spread strategy instance."""
    return TwoPartSpreadStrategy(
        min_discount_pct=min_discount_pct,
        max_spread_width_bps=max_spread_width_bps,
        min_credit_ratio=min_credit_ratio,
        target_credit_ratio=target_credit_ratio,
        fill_callback=fill_callback,
    )


__all__ = [
    "CreditMatch",
    "DebitOpportunity",
    "FillLocation",
    "PositionStatus",
    "SessionFillStats",
    "SpreadFill",
    "SpreadQuote",
    "SpreadType",
    "TwoPartPosition",
    "TwoPartSpreadStrategy",
    "create_two_part_strategy",
]
