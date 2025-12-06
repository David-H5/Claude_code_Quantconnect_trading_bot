"""
Arbitrage Executor Module

Implements intelligent arbitrage execution for two-part spread strategies:
- Maximizes credit received on credit leg within strike ranges
- Repeats profitable trades with adjustable delays to avoid MM detection
- Balances debit/credit positions per option chain
- Concurrent trading across multiple expirations
- Adaptive order sizing with autonomous learning
- Quick cancel logic (2-3 seconds for unfilled orders)
- Autonomous optimization of timing and sizing parameters
"""

import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class ExpirationCategory(Enum):
    """Expiration time categories."""

    WEEKLY = "weekly"  # < 14 days
    MONTHLY = "monthly"  # 14-45 days
    QUARTERLY = "quarterly"  # 45-120 days
    LEAPS = "leaps"  # > 120 days


class OrderResult(Enum):
    """Result of an order attempt."""

    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


class TradingPhase(Enum):
    """Market trading phases."""

    PRE_MARKET = "pre_market"
    OPEN_AUCTION = "open_auction"  # First 15 min
    MORNING = "morning"  # 9:45-11:30
    MIDDAY = "midday"  # 11:30-2:00
    AFTERNOON = "afternoon"  # 2:00-3:45
    CLOSE_AUCTION = "close_auction"  # Last 15 min
    AFTER_HOURS = "after_hours"


@dataclass
class StrikeRange:
    """Range of strikes for opportunity scanning."""

    lower_strike: float
    upper_strike: float
    underlying_price: float

    @property
    def width(self) -> float:
        return self.upper_strike - self.lower_strike

    @property
    def lower_moneyness(self) -> float:
        return self.lower_strike / self.underlying_price

    @property
    def upper_moneyness(self) -> float:
        return self.upper_strike / self.underlying_price


@dataclass
class SpreadLeg:
    """Individual leg of a spread."""

    strike: float
    expiration: datetime
    option_type: str  # 'call' or 'put'
    side: str  # 'buy' or 'sell'
    bid: float
    ask: float
    volume: int
    open_interest: int
    iv: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_width(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        if self.mid > 0:
            return self.spread_width / self.mid
        return float("inf")


@dataclass
class ArbitrageOpportunity:
    """Identified arbitrage opportunity in option chain."""

    symbol: str
    expiration: datetime
    debit_long_leg: SpreadLeg
    debit_short_leg: SpreadLeg
    credit_long_leg: SpreadLeg
    credit_short_leg: SpreadLeg

    # Calculated values
    debit_cost: float  # What we pay for debit spread
    credit_received: float  # What we receive for credit spread
    net_premium: float  # credit - debit (positive = net credit)
    max_profit: float
    max_loss: float

    # Quality metrics
    fill_probability: float
    liquidity_score: float
    iv_edge: float  # IV discount on debit, premium on credit

    timestamp: datetime = field(default_factory=datetime.now)
    opportunity_id: str = ""

    def __post_init__(self):
        if not self.opportunity_id:
            self.opportunity_id = f"{self.symbol}_{self.expiration.strftime('%Y%m%d')}_{int(time.time()*1000)}"

    @property
    def is_net_credit(self) -> bool:
        return self.net_premium >= 0

    @property
    def days_to_expiry(self) -> int:
        return (self.expiration - datetime.now()).days

    @property
    def risk_reward_ratio(self) -> float:
        if self.max_loss > 0:
            return self.max_profit / self.max_loss
        return float("inf")


@dataclass
class OrderAttempt:
    """Record of an order attempt."""

    opportunity_id: str
    order_type: str  # 'debit' or 'credit'
    contracts: int
    limit_price: float
    submitted_time: datetime
    result: OrderResult
    fill_time: datetime | None = None
    fill_price: float | None = None
    fill_quantity: int = 0
    cancel_time: datetime | None = None

    @property
    def time_to_fill_ms(self) -> int | None:
        if self.fill_time:
            return int((self.fill_time - self.submitted_time).total_seconds() * 1000)
        return None

    @property
    def was_quick_fill(self) -> bool:
        """Fill within 2 seconds is considered quick."""
        ttf = self.time_to_fill_ms
        return ttf is not None and ttf < 2000


@dataclass
class PositionBalance:
    """Tracks debit/credit balance for an option chain."""

    symbol: str
    expiration: datetime
    debit_contracts: int = 0
    credit_contracts: int = 0
    total_debit_paid: float = 0.0
    total_credit_received: float = 0.0

    @property
    def net_contracts(self) -> int:
        """Positive = more debits, negative = more credits."""
        return self.debit_contracts - self.credit_contracts

    @property
    def is_balanced(self) -> bool:
        return self.debit_contracts == self.credit_contracts

    @property
    def net_premium(self) -> float:
        return self.total_credit_received - self.total_debit_paid

    def add_debit(self, contracts: int, cost: float) -> None:
        self.debit_contracts += contracts
        self.total_debit_paid += cost

    def add_credit(self, contracts: int, credit: float) -> None:
        self.credit_contracts += contracts
        self.total_credit_received += credit


@dataclass
class TimingParameters:
    """Order timing parameters."""

    cancel_after_seconds: float = 2.5  # Cancel if not filled
    min_delay_between_attempts: float = 3.0  # Minimum delay
    max_delay_between_attempts: float = 15.0  # Maximum delay
    random_delay_factor: float = 0.3  # Randomization factor

    # Trading phase adjustments
    open_auction_delay_multiplier: float = 2.0  # More caution at open
    close_auction_delay_multiplier: float = 1.5  # Slightly more at close

    def get_delay(self, phase: TradingPhase) -> float:
        """Get delay with randomization and phase adjustment."""
        base_delay = random.uniform(self.min_delay_between_attempts, self.max_delay_between_attempts)

        # Add random jitter
        jitter = base_delay * self.random_delay_factor * (random.random() - 0.5)
        delay = base_delay + jitter

        # Apply phase multipliers
        if phase == TradingPhase.OPEN_AUCTION:
            delay *= self.open_auction_delay_multiplier
        elif phase == TradingPhase.CLOSE_AUCTION:
            delay *= self.close_auction_delay_multiplier

        return max(delay, self.min_delay_between_attempts)


@dataclass
class SizingParameters:
    """Order sizing parameters."""

    min_contracts: int = 1
    max_contracts: int = 10
    current_contracts: int = 1

    # Learning parameters
    target_fill_rate: float = 0.50  # Target 50% fill rate
    size_increase_threshold: float = 0.70  # Increase if > 70% fills
    size_decrease_threshold: float = 0.30  # Decrease if < 30% fills

    # Adjustments per result
    fill_weight: float = 1.0
    partial_weight: float = 0.5
    timeout_weight: float = 0.0

    def adjust_size(self, recent_fill_rate: float) -> int:
        """Adjust contract size based on recent fill rate."""
        old_size = self.current_contracts

        if recent_fill_rate >= self.size_increase_threshold:
            # Good fill rate, try more contracts
            self.current_contracts = min(self.current_contracts + 1, self.max_contracts)
        elif recent_fill_rate <= self.size_decrease_threshold:
            # Poor fill rate, reduce size
            self.current_contracts = max(self.current_contracts - 1, self.min_contracts)

        return self.current_contracts


@dataclass
class OptimizationMetrics:
    """Metrics for autonomous optimization."""

    # Fill rate by contract size
    fills_by_size: dict[int, list[bool]] = field(default_factory=dict)

    # Fill rate by time of day
    fills_by_hour: dict[int, list[bool]] = field(default_factory=dict)

    # Fill rate by delay
    fills_by_delay: dict[str, list[bool]] = field(default_factory=dict)  # 'short', 'medium', 'long'

    # Time to fill distribution
    fill_times_ms: list[int] = field(default_factory=list)

    # Profit by opportunity type
    profits_by_net_premium: dict[str, list[float]] = field(default_factory=dict)

    def record_attempt(
        self,
        contracts: int,
        hour: int,
        delay_category: str,
        filled: bool,
        fill_time_ms: int | None = None,
        profit: float | None = None,
    ) -> None:
        """Record an order attempt for learning."""
        # By size
        if contracts not in self.fills_by_size:
            self.fills_by_size[contracts] = []
        self.fills_by_size[contracts].append(filled)

        # By hour
        if hour not in self.fills_by_hour:
            self.fills_by_hour[hour] = []
        self.fills_by_hour[hour].append(filled)

        # By delay
        if delay_category not in self.fills_by_delay:
            self.fills_by_delay[delay_category] = []
        self.fills_by_delay[delay_category].append(filled)

        # Fill time
        if fill_time_ms is not None:
            self.fill_times_ms.append(fill_time_ms)

    def get_optimal_size(self) -> int:
        """Get optimal contract size based on fill rate."""
        best_size = 1
        best_score = 0.0

        for size, fills in self.fills_by_size.items():
            if len(fills) >= 10:  # Minimum sample size
                fill_rate = sum(fills) / len(fills)
                # Score = fill_rate * size (maximize filled volume)
                score = fill_rate * size
                if score > best_score:
                    best_score = score
                    best_size = size

        return best_size

    def get_optimal_hours(self) -> list[int]:
        """Get hours with best fill rates."""
        hour_rates = []
        for hour, fills in self.fills_by_hour.items():
            if len(fills) >= 5:
                rate = sum(fills) / len(fills)
                hour_rates.append((hour, rate))

        hour_rates.sort(key=lambda x: x[1], reverse=True)
        return [h for h, r in hour_rates if r >= 0.25]

    def get_optimal_cancel_time_ms(self) -> int:
        """Get optimal cancel time based on fill time distribution."""
        if not self.fill_times_ms:
            return 2500  # Default 2.5 seconds

        # Use 95th percentile of successful fills
        sorted_times = sorted(self.fill_times_ms)
        idx = int(len(sorted_times) * 0.95)
        p95 = sorted_times[min(idx, len(sorted_times) - 1)]

        # Add small buffer, cap at 5 seconds
        return min(p95 + 500, 5000)


class CreditMaximizer:
    """Finds maximum credit opportunities within strike ranges."""

    def __init__(
        self,
        min_credit_threshold: float = 0.0,  # Minimum net credit required
        max_spread_width: int = 10,  # Maximum strike width
        prefer_atm: bool = True,  # Prefer at-the-money for liquidity
    ):
        self.min_credit_threshold = min_credit_threshold
        self.max_spread_width = max_spread_width
        self.prefer_atm = prefer_atm

    def find_best_credit_spread(
        self,
        debit_opportunity: tuple[SpreadLeg, SpreadLeg],
        available_options: list[SpreadLeg],
        underlying_price: float,
    ) -> tuple[SpreadLeg, SpreadLeg, float] | None:
        """
        Find the credit spread that maximizes credit received.

        Returns: (long_leg, short_leg, credit_received) or None
        """
        debit_long, debit_short = debit_opportunity
        debit_cost = self._calculate_debit_cost(debit_long, debit_short)

        # Filter options for valid credit spread candidates
        # Credit spread should be further OTM than debit spread
        candidates = self._filter_credit_candidates(
            debit_short.strike, debit_short.option_type, available_options, underlying_price
        )

        best_spread = None
        best_credit = 0.0

        for i, short_leg in enumerate(candidates):
            for long_leg in candidates[i + 1 :]:
                # Long leg should be further OTM than short leg
                if not self._is_valid_credit_spread(short_leg, long_leg):
                    continue

                credit = self._calculate_credit(short_leg, long_leg)
                net_premium = credit - debit_cost

                # Check if meets minimum credit threshold
                if net_premium >= self.min_credit_threshold:
                    # Score by credit amount and liquidity
                    score = self._score_credit_spread(short_leg, long_leg, credit, underlying_price)

                    if score > best_credit:
                        best_credit = score
                        best_spread = (long_leg, short_leg, credit)

        return best_spread

    def _calculate_debit_cost(self, long_leg: SpreadLeg, short_leg: SpreadLeg) -> float:
        """Calculate realistic debit cost (try to get filled below mid)."""
        # Long leg: try to buy below mid (0.4 of spread)
        long_price = long_leg.bid + 0.4 * (long_leg.ask - long_leg.bid)
        # Short leg: try to sell above mid (0.6 of spread)
        short_price = short_leg.bid + 0.6 * (short_leg.ask - short_leg.bid)

        return long_price - short_price

    def _calculate_credit(self, short_leg: SpreadLeg, long_leg: SpreadLeg) -> float:
        """Calculate realistic credit received."""
        # Short leg: try to sell above mid
        short_price = short_leg.bid + 0.6 * (short_leg.ask - short_leg.bid)
        # Long leg: try to buy below mid
        long_price = long_leg.bid + 0.4 * (long_leg.ask - long_leg.bid)

        return short_price - long_price

    def _filter_credit_candidates(
        self, debit_strike: float, option_type: str, options: list[SpreadLeg], underlying_price: float
    ) -> list[SpreadLeg]:
        """Filter options valid for credit spread leg."""
        candidates = []

        for opt in options:
            if opt.option_type != option_type:
                continue

            # Credit spread should be further OTM
            if option_type == "call":
                if opt.strike <= debit_strike:
                    continue
            else:  # put
                if opt.strike >= debit_strike:
                    continue

            # Check liquidity
            if opt.volume < 10 or opt.open_interest < 50:
                continue

            # Check spread width
            if opt.spread_pct > 0.20:  # Max 20% spread
                continue

            candidates.append(opt)

        # Sort by distance from money
        candidates.sort(key=lambda x: abs(x.strike - underlying_price), reverse=(not self.prefer_atm))

        return candidates

    def _is_valid_credit_spread(self, short_leg: SpreadLeg, long_leg: SpreadLeg) -> bool:
        """Check if credit spread is valid."""
        # Width check
        width = abs(long_leg.strike - short_leg.strike)
        if width > self.max_spread_width:
            return False

        # Both should have reasonable liquidity
        if min(short_leg.volume, long_leg.volume) < 5:
            return False

        return True

    def _score_credit_spread(
        self, short_leg: SpreadLeg, long_leg: SpreadLeg, credit: float, underlying_price: float
    ) -> float:
        """Score a credit spread for selection."""
        # Base score is credit amount
        score = credit * 100  # Scale up

        # Liquidity bonus
        avg_volume = (short_leg.volume + long_leg.volume) / 2
        score += min(avg_volume / 100, 5)  # Up to 5 points for volume

        # Tight spread bonus
        avg_spread_pct = (short_leg.spread_pct + long_leg.spread_pct) / 2
        if avg_spread_pct < 0.10:
            score += 3
        elif avg_spread_pct < 0.15:
            score += 1

        return score


class PositionBalancer:
    """Manages position balance per option chain."""

    def __init__(
        self,
        max_imbalance: int = 5,  # Max contracts imbalance
        force_balance_at_close: bool = True,
    ):
        self.max_imbalance = max_imbalance
        self.force_balance_at_close = force_balance_at_close
        self.balances: dict[str, PositionBalance] = {}

    def get_balance_key(self, symbol: str, expiration: datetime) -> str:
        return f"{symbol}_{expiration.strftime('%Y%m%d')}"

    def get_balance(self, symbol: str, expiration: datetime) -> PositionBalance:
        key = self.get_balance_key(symbol, expiration)
        if key not in self.balances:
            self.balances[key] = PositionBalance(symbol=symbol, expiration=expiration)
        return self.balances[key]

    def can_add_debit(self, symbol: str, expiration: datetime, contracts: int) -> bool:
        """Check if we can add more debit positions."""
        balance = self.get_balance(symbol, expiration)
        potential_imbalance = balance.net_contracts + contracts
        return potential_imbalance <= self.max_imbalance

    def can_add_credit(self, symbol: str, expiration: datetime, contracts: int) -> bool:
        """Check if we can add more credit positions."""
        balance = self.get_balance(symbol, expiration)
        potential_imbalance = balance.net_contracts - contracts
        return potential_imbalance >= -self.max_imbalance

    def record_debit_fill(self, symbol: str, expiration: datetime, contracts: int, cost: float) -> None:
        balance = self.get_balance(symbol, expiration)
        balance.add_debit(contracts, cost)

    def record_credit_fill(self, symbol: str, expiration: datetime, contracts: int, credit: float) -> None:
        balance = self.get_balance(symbol, expiration)
        balance.add_credit(contracts, credit)

    def get_required_credits(self, symbol: str, expiration: datetime) -> int:
        """Get number of credit contracts needed to balance."""
        balance = self.get_balance(symbol, expiration)
        return max(0, balance.net_contracts)

    def get_required_debits(self, symbol: str, expiration: datetime) -> int:
        """Get number of debit contracts needed to balance."""
        balance = self.get_balance(symbol, expiration)
        return max(0, -balance.net_contracts)

    def get_all_imbalanced(self) -> list[PositionBalance]:
        """Get all chains with position imbalance."""
        return [b for b in self.balances.values() if not b.is_balanced]

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all positions."""
        total_debit = sum(b.total_debit_paid for b in self.balances.values())
        total_credit = sum(b.total_credit_received for b in self.balances.values())
        total_imbalance = sum(abs(b.net_contracts) for b in self.balances.values())

        return {
            "chains": len(self.balances),
            "total_debit_paid": total_debit,
            "total_credit_received": total_credit,
            "net_premium": total_credit - total_debit,
            "total_imbalance": total_imbalance,
            "balanced_chains": sum(1 for b in self.balances.values() if b.is_balanced),
        }


class ExpirationTrader:
    """Handles trading for a single expiration date."""

    def __init__(
        self,
        symbol: str,
        expiration: datetime,
        timing: TimingParameters,
        sizing: SizingParameters,
        balancer: PositionBalancer,
        credit_maximizer: CreditMaximizer,
        order_callback: Callable,  # Function to submit orders
        quote_callback: Callable,  # Function to get current quotes
    ):
        self.symbol = symbol
        self.expiration = expiration
        self.timing = timing
        self.sizing = sizing
        self.balancer = balancer
        self.credit_maximizer = credit_maximizer
        self.order_callback = order_callback
        self.quote_callback = quote_callback

        self.active = False
        self.current_opportunity: ArbitrageOpportunity | None = None
        self.attempts: list[OrderAttempt] = []
        self.session_fills = 0
        self.session_attempts = 0
        self._lock = threading.Lock()

    @property
    def days_to_expiry(self) -> int:
        return (self.expiration - datetime.now()).days

    @property
    def expiration_category(self) -> ExpirationCategory:
        days = self.days_to_expiry
        if days < 14:
            return ExpirationCategory.WEEKLY
        elif days < 45:
            return ExpirationCategory.MONTHLY
        elif days < 120:
            return ExpirationCategory.QUARTERLY
        else:
            return ExpirationCategory.LEAPS

    @property
    def is_optimal_expiration(self) -> bool:
        """30-180 days is optimal range."""
        return 30 <= self.days_to_expiry <= 180

    def get_fill_rate(self) -> float:
        if self.session_attempts == 0:
            return 0.0
        return self.session_fills / self.session_attempts

    def scan_opportunities(self, options: list[SpreadLeg], underlying_price: float) -> list[ArbitrageOpportunity]:
        """Scan for arbitrage opportunities in this expiration."""
        opportunities = []

        # Group options by type
        calls = [o for o in options if o.option_type == "call"]
        puts = [o for o in options if o.option_type == "put"]

        # Scan call spreads
        opportunities.extend(self._scan_spread_opportunities(calls, underlying_price, "call"))

        # Scan put spreads
        opportunities.extend(self._scan_spread_opportunities(puts, underlying_price, "put"))

        # Sort by net premium (best opportunities first)
        opportunities.sort(key=lambda x: x.net_premium, reverse=True)

        return opportunities

    def _scan_spread_opportunities(
        self, options: list[SpreadLeg], underlying_price: float, option_type: str
    ) -> list[ArbitrageOpportunity]:
        """Scan for debit/credit spread opportunities."""
        opportunities = []

        # Sort by strike
        options.sort(key=lambda x: x.strike)

        # Look for debit spread opportunities (wide spreads = opportunity)
        for i, long_leg in enumerate(options):
            for short_leg in options[i + 1 :]:
                # Check if valid debit spread
                if not self._is_valid_debit_spread(long_leg, short_leg, option_type):
                    continue

                # Calculate debit cost
                debit_cost = self._estimate_debit_fill_price(long_leg, short_leg)

                # Find best credit spread to match
                credit_result = self.credit_maximizer.find_best_credit_spread(
                    (long_leg, short_leg), options, underlying_price
                )

                if credit_result:
                    credit_long, credit_short, credit = credit_result
                    net_premium = credit - debit_cost

                    if net_premium >= 0:  # Net credit or break even
                        opp = ArbitrageOpportunity(
                            symbol=self.symbol,
                            expiration=self.expiration,
                            debit_long_leg=long_leg,
                            debit_short_leg=short_leg,
                            credit_long_leg=credit_long,
                            credit_short_leg=credit_short,
                            debit_cost=debit_cost,
                            credit_received=credit,
                            net_premium=net_premium,
                            max_profit=self._calc_max_profit(
                                long_leg, short_leg, credit_long, credit_short, net_premium
                            ),
                            max_loss=self._calc_max_loss(long_leg, short_leg, credit_long, credit_short, net_premium),
                            fill_probability=self._estimate_fill_probability(
                                long_leg, short_leg, credit_long, credit_short
                            ),
                            liquidity_score=self._calc_liquidity_score(long_leg, short_leg, credit_long, credit_short),
                            iv_edge=0.0,  # TODO: Calculate IV edge
                        )
                        opportunities.append(opp)

        return opportunities

    def _is_valid_debit_spread(self, long_leg: SpreadLeg, short_leg: SpreadLeg, option_type: str) -> bool:
        """Check if debit spread is valid for our strategy."""
        # Width check (want reasonable width for profit potential)
        width = abs(short_leg.strike - long_leg.strike)
        if width < 1 or width > 20:
            return False

        # Need some liquidity
        if min(long_leg.volume, short_leg.volume) < 5:
            return False
        if min(long_leg.open_interest, short_leg.open_interest) < 20:
            return False

        # Wide spread is an OPPORTUNITY (can get filled below mid)
        # So we don't filter based on spread width

        return True

    def _estimate_debit_fill_price(self, long_leg: SpreadLeg, short_leg: SpreadLeg) -> float:
        """Estimate realistic fill price for debit spread."""
        # Try to buy long leg below mid
        long_price = long_leg.bid + 0.35 * (long_leg.ask - long_leg.bid)
        # Try to sell short leg above mid
        short_price = short_leg.bid + 0.65 * (short_leg.ask - short_leg.bid)

        return long_price - short_price

    def _calc_max_profit(
        self,
        debit_long: SpreadLeg,
        debit_short: SpreadLeg,
        credit_long: SpreadLeg,
        credit_short: SpreadLeg,
        net_premium: float,
    ) -> float:
        """Calculate maximum profit for the butterfly/condor."""
        # For iron condor: max profit is net premium
        # For butterfly: max profit at center strike
        # Simplified calculation
        return net_premium * 100  # Per contract

    def _calc_max_loss(
        self,
        debit_long: SpreadLeg,
        debit_short: SpreadLeg,
        credit_long: SpreadLeg,
        credit_short: SpreadLeg,
        net_premium: float,
    ) -> float:
        """Calculate maximum loss for the butterfly/condor."""
        # Max loss is width of spreads minus net premium
        debit_width = abs(debit_short.strike - debit_long.strike)
        credit_width = abs(credit_short.strike - credit_long.strike)
        max_width = max(debit_width, credit_width)

        return (max_width - net_premium) * 100  # Per contract

    def _estimate_fill_probability(
        self, debit_long: SpreadLeg, debit_short: SpreadLeg, credit_long: SpreadLeg, credit_short: SpreadLeg
    ) -> float:
        """Estimate fill probability based on current session data."""
        # Use actual session fill rate
        if self.session_attempts > 0:
            return self.get_fill_rate()

        # Initial estimate based on liquidity
        avg_volume = (debit_long.volume + debit_short.volume + credit_long.volume + credit_short.volume) / 4

        if avg_volume >= 100:
            return 0.50
        elif avg_volume >= 50:
            return 0.35
        elif avg_volume >= 20:
            return 0.25
        else:
            return 0.15

    def _calc_liquidity_score(
        self, debit_long: SpreadLeg, debit_short: SpreadLeg, credit_long: SpreadLeg, credit_short: SpreadLeg
    ) -> float:
        """Calculate liquidity score 0-100."""
        legs = [debit_long, debit_short, credit_long, credit_short]

        # Volume score
        avg_volume = sum(l.volume for l in legs) / 4
        volume_score = min(avg_volume / 10, 30)  # Max 30 points

        # OI score
        avg_oi = sum(l.open_interest for l in legs) / 4
        oi_score = min(avg_oi / 50, 30)  # Max 30 points

        # Spread tightness score
        avg_spread = sum(l.spread_pct for l in legs) / 4
        if avg_spread < 0.05:
            spread_score = 40
        elif avg_spread < 0.10:
            spread_score = 30
        elif avg_spread < 0.15:
            spread_score = 20
        elif avg_spread < 0.20:
            spread_score = 10
        else:
            spread_score = 0

        return volume_score + oi_score + spread_score

    def execute_trade_cycle(self) -> dict[str, Any] | None:
        """Execute one trade cycle: find opportunity and attempt to fill."""
        with self._lock:
            if not self.active:
                return None

            # Get current quotes
            options = self.quote_callback(self.symbol, self.expiration)
            if not options:
                return {"status": "no_quotes"}

            underlying_price = self._get_underlying_price()

            # Scan for opportunities
            opportunities = self.scan_opportunities(options, underlying_price)

            if not opportunities:
                return {"status": "no_opportunities"}

            # Check position balance
            best_opp = None
            for opp in opportunities:
                if self.balancer.can_add_debit(self.symbol, self.expiration, self.sizing.current_contracts):
                    best_opp = opp
                    break

            if not best_opp:
                return {"status": "balance_limit_reached"}

            self.current_opportunity = best_opp

            # Execute debit spread first
            debit_result = self._execute_debit_spread(best_opp)

            if debit_result["filled"]:
                # Immediately try credit spread
                credit_result = self._execute_credit_spread(best_opp)

                return {"status": "completed", "opportunity": best_opp, "debit": debit_result, "credit": credit_result}
            else:
                return {"status": "debit_not_filled", "opportunity": best_opp, "debit": debit_result}

    def _execute_debit_spread(self, opp: ArbitrageOpportunity) -> dict[str, Any]:
        """Execute debit spread with quick cancel."""
        self.session_attempts += 1

        # Calculate limit price (below mid)
        limit_price = opp.debit_cost

        attempt = OrderAttempt(
            opportunity_id=opp.opportunity_id,
            order_type="debit",
            contracts=self.sizing.current_contracts,
            limit_price=limit_price,
            submitted_time=datetime.now(),
            result=OrderResult.PENDING,
        )

        # Submit order
        order_id = self.order_callback(
            symbol=self.symbol,
            legs=[
                ("buy", opp.debit_long_leg.strike, opp.debit_long_leg.option_type),
                ("sell", opp.debit_short_leg.strike, opp.debit_short_leg.option_type),
            ],
            quantity=self.sizing.current_contracts,
            limit_price=limit_price,
            expiration=self.expiration,
        )

        # Wait for fill or timeout
        start_time = time.time()
        while time.time() - start_time < self.timing.cancel_after_seconds:
            status = self._check_order_status(order_id)
            if status == "filled":
                attempt.result = OrderResult.FILLED
                attempt.fill_time = datetime.now()
                attempt.fill_quantity = self.sizing.current_contracts
                self.session_fills += 1

                # Record with balancer
                self.balancer.record_debit_fill(
                    self.symbol, self.expiration, self.sizing.current_contracts, limit_price
                )

                self.attempts.append(attempt)
                return {"filled": True, "attempt": attempt}

            time.sleep(0.1)  # Poll every 100ms

        # Timeout - cancel order
        self._cancel_order(order_id)
        attempt.result = OrderResult.TIMEOUT
        attempt.cancel_time = datetime.now()
        self.attempts.append(attempt)

        return {"filled": False, "attempt": attempt}

    def _execute_credit_spread(self, opp: ArbitrageOpportunity) -> dict[str, Any]:
        """Execute credit spread with quick cancel."""
        self.session_attempts += 1

        # Calculate limit price (above mid for credit)
        limit_price = opp.credit_received

        attempt = OrderAttempt(
            opportunity_id=opp.opportunity_id,
            order_type="credit",
            contracts=self.sizing.current_contracts,
            limit_price=limit_price,
            submitted_time=datetime.now(),
            result=OrderResult.PENDING,
        )

        # Submit order
        order_id = self.order_callback(
            symbol=self.symbol,
            legs=[
                ("sell", opp.credit_short_leg.strike, opp.credit_short_leg.option_type),
                ("buy", opp.credit_long_leg.strike, opp.credit_long_leg.option_type),
            ],
            quantity=self.sizing.current_contracts,
            limit_price=limit_price,
            expiration=self.expiration,
        )

        # Wait for fill or timeout
        start_time = time.time()
        while time.time() - start_time < self.timing.cancel_after_seconds:
            status = self._check_order_status(order_id)
            if status == "filled":
                attempt.result = OrderResult.FILLED
                attempt.fill_time = datetime.now()
                attempt.fill_quantity = self.sizing.current_contracts
                self.session_fills += 1

                # Record with balancer
                self.balancer.record_credit_fill(
                    self.symbol, self.expiration, self.sizing.current_contracts, limit_price
                )

                self.attempts.append(attempt)
                return {"filled": True, "attempt": attempt}

            time.sleep(0.1)

        # Timeout - cancel order
        self._cancel_order(order_id)
        attempt.result = OrderResult.TIMEOUT
        attempt.cancel_time = datetime.now()
        self.attempts.append(attempt)

        return {"filled": False, "attempt": attempt}

    def _get_underlying_price(self) -> float:
        """Get current underlying price."""
        # Would be implemented with actual broker API
        return 100.0  # Placeholder

    def _check_order_status(self, order_id: str) -> str:
        """Check order status with broker."""
        # Would be implemented with actual broker API
        return "pending"  # Placeholder

    def _cancel_order(self, order_id: str) -> bool:
        """Cancel order with broker."""
        # Would be implemented with actual broker API
        return True  # Placeholder


class ArbitrageExecutor:
    """
    Main orchestrator for arbitrage execution across multiple expirations.

    Features:
    - Concurrent trading across expirations
    - Adaptive sizing and timing
    - Position balancing
    - Autonomous optimization
    """

    def __init__(
        self,
        order_callback: Callable,
        quote_callback: Callable,
        underlying_callback: Callable,
        timing: TimingParameters | None = None,
        sizing: SizingParameters | None = None,
        min_fill_rate: float = 0.25,
        optimal_expiry_range: tuple[int, int] = (30, 180),
    ):
        self.order_callback = order_callback
        self.quote_callback = quote_callback
        self.underlying_callback = underlying_callback

        self.timing = timing or TimingParameters()
        self.sizing = sizing or SizingParameters()
        self.min_fill_rate = min_fill_rate
        self.optimal_expiry_range = optimal_expiry_range

        self.balancer = PositionBalancer()
        self.credit_maximizer = CreditMaximizer()
        self.metrics = OptimizationMetrics()

        self.traders: dict[str, ExpirationTrader] = {}
        self.active = False
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

    def add_symbol(self, symbol: str, expirations: list[datetime]) -> None:
        """Add symbol with expirations to trade."""
        for exp in expirations:
            days = (exp - datetime.now()).days

            # Check if in acceptable range
            if days < 7:  # Too close to expiry
                continue

            key = f"{symbol}_{exp.strftime('%Y%m%d')}"

            if key not in self.traders:
                trader = ExpirationTrader(
                    symbol=symbol,
                    expiration=exp,
                    timing=self.timing,
                    sizing=SizingParameters(**vars(self.sizing)),  # Copy
                    balancer=self.balancer,
                    credit_maximizer=self.credit_maximizer,
                    order_callback=self.order_callback,
                    quote_callback=self.quote_callback,
                )
                self.traders[key] = trader

    def start(self) -> None:
        """Start all expiration traders concurrently."""
        with self._lock:
            if self.active:
                return

            self.active = True

            for key, trader in self.traders.items():
                trader.active = True
                thread = threading.Thread(target=self._run_trader_loop, args=(key, trader), daemon=True)
                thread.start()
                self._threads.append(thread)

    def stop(self) -> None:
        """Stop all traders."""
        with self._lock:
            self.active = False
            for trader in self.traders.values():
                trader.active = False

    def _run_trader_loop(self, key: str, trader: ExpirationTrader) -> None:
        """Main trading loop for a single expiration."""
        phase = self._get_trading_phase()

        while self.active and trader.active:
            try:
                # Check fill rate threshold
                if trader.session_attempts >= 10:
                    if trader.get_fill_rate() < self.min_fill_rate:
                        # Pause trading for this expiration
                        time.sleep(60)  # Wait 1 minute before retrying
                        continue

                # Execute trade cycle
                result = trader.execute_trade_cycle()

                if result:
                    self._record_result(trader, result)

                    # Adjust sizing based on results
                    if trader.session_attempts % 20 == 0:
                        self._adjust_sizing(trader)

                # Get delay with phase awareness
                phase = self._get_trading_phase()
                delay = self.timing.get_delay(phase)
                time.sleep(delay)

            except Exception:
                # Log error and continue
                time.sleep(5)

    def _get_trading_phase(self) -> TradingPhase:
        """Determine current trading phase."""
        now = datetime.now()
        hour = now.hour
        minute = now.minute

        # Market hours: 9:30 AM - 4:00 PM ET
        if hour < 9 or (hour == 9 and minute < 30):
            return TradingPhase.PRE_MARKET
        elif hour == 9 and minute < 45:
            return TradingPhase.OPEN_AUCTION
        elif hour < 11 or (hour == 11 and minute < 30):
            return TradingPhase.MORNING
        elif hour < 14:
            return TradingPhase.MIDDAY
        elif hour < 15 or (hour == 15 and minute < 45):
            return TradingPhase.AFTERNOON
        elif hour < 16:
            return TradingPhase.CLOSE_AUCTION
        else:
            return TradingPhase.AFTER_HOURS

    def _record_result(self, trader: ExpirationTrader, result: dict[str, Any]) -> None:
        """Record trade result for optimization."""
        if result.get("status") == "completed":
            debit = result.get("debit", {})
            credit = result.get("credit", {})

            # Record fills
            if debit.get("filled"):
                attempt = debit.get("attempt")
                if attempt:
                    self.metrics.record_attempt(
                        contracts=attempt.contracts,
                        hour=datetime.now().hour,
                        delay_category=self._categorize_delay(),
                        filled=True,
                        fill_time_ms=attempt.time_to_fill_ms,
                    )

            if credit.get("filled"):
                attempt = credit.get("attempt")
                if attempt:
                    self.metrics.record_attempt(
                        contracts=attempt.contracts,
                        hour=datetime.now().hour,
                        delay_category=self._categorize_delay(),
                        filled=True,
                        fill_time_ms=attempt.time_to_fill_ms,
                    )

    def _categorize_delay(self) -> str:
        """Categorize current delay setting."""
        avg_delay = (self.timing.min_delay_between_attempts + self.timing.max_delay_between_attempts) / 2
        if avg_delay < 5:
            return "short"
        elif avg_delay < 10:
            return "medium"
        else:
            return "long"

    def _adjust_sizing(self, trader: ExpirationTrader) -> None:
        """Adjust sizing based on fill rate."""
        fill_rate = trader.get_fill_rate()
        trader.sizing.adjust_size(fill_rate)

    def optimize_parameters(self) -> dict[str, Any]:
        """Run autonomous optimization based on collected data."""
        recommendations = {}

        # Optimal contract size
        optimal_size = self.metrics.get_optimal_size()
        recommendations["optimal_contracts"] = optimal_size

        # Optimal trading hours
        optimal_hours = self.metrics.get_optimal_hours()
        recommendations["optimal_hours"] = optimal_hours

        # Optimal cancel time
        optimal_cancel = self.metrics.get_optimal_cancel_time_ms()
        recommendations["optimal_cancel_ms"] = optimal_cancel

        return recommendations

    def apply_optimization(self, recommendations: dict[str, Any]) -> None:
        """Apply optimization recommendations."""
        if "optimal_contracts" in recommendations:
            self.sizing.current_contracts = recommendations["optimal_contracts"]

        if "optimal_cancel_ms" in recommendations:
            self.timing.cancel_after_seconds = recommendations["optimal_cancel_ms"] / 1000

    def get_status(self) -> dict[str, Any]:
        """Get current status of all traders."""
        status = {
            "active": self.active,
            "traders": {},
            "position_summary": self.balancer.get_summary(),
            "optimization_metrics": {
                "fills_by_size": {k: sum(v) / len(v) if v else 0 for k, v in self.metrics.fills_by_size.items()},
                "fills_by_hour": {k: sum(v) / len(v) if v else 0 for k, v in self.metrics.fills_by_hour.items()},
            },
        }

        for key, trader in self.traders.items():
            status["traders"][key] = {
                "symbol": trader.symbol,
                "expiration": trader.expiration.isoformat(),
                "days_to_expiry": trader.days_to_expiry,
                "category": trader.expiration_category.value,
                "is_optimal": trader.is_optimal_expiration,
                "session_attempts": trader.session_attempts,
                "session_fills": trader.session_fills,
                "fill_rate": trader.get_fill_rate(),
                "current_contracts": trader.sizing.current_contracts,
            }

        return status

    def get_llm_summary(self) -> str:
        """Get summary for LLM analysis."""
        status = self.get_status()
        pos = status["position_summary"]

        summary = f"""Arbitrage Executor Status:

Active Traders: {len(status['traders'])}
Total Net Premium: ${pos['net_premium']:.2f}
Position Imbalance: {pos['total_imbalance']} contracts
Balanced Chains: {pos['balanced_chains']}/{pos['chains']}

Performance by Contract Size:
"""
        for size, rate in status["optimization_metrics"]["fills_by_size"].items():
            summary += f"  {size} contracts: {rate:.1%} fill rate\n"

        summary += "\nPerformance by Hour:\n"
        for hour, rate in sorted(status["optimization_metrics"]["fills_by_hour"].items()):
            summary += f"  {hour}:00: {rate:.1%} fill rate\n"

        best_traders = sorted(status["traders"].values(), key=lambda x: x["fill_rate"], reverse=True)[:5]

        summary += "\nTop Performing Expirations:\n"
        for t in best_traders:
            summary += f"  {t['symbol']} {t['expiration'][:10]}: {t['fill_rate']:.1%} ({t['session_fills']}/{t['session_attempts']})\n"

        return summary


def create_arbitrage_executor(
    order_callback: Callable,
    quote_callback: Callable,
    underlying_callback: Callable,
    min_fill_rate: float = 0.25,
    cancel_after_seconds: float = 2.5,
    min_delay: float = 3.0,
    max_delay: float = 15.0,
    max_contracts: int = 10,
) -> ArbitrageExecutor:
    """Factory function to create ArbitrageExecutor with custom settings."""
    timing = TimingParameters(
        cancel_after_seconds=cancel_after_seconds,
        min_delay_between_attempts=min_delay,
        max_delay_between_attempts=max_delay,
    )

    sizing = SizingParameters(
        min_contracts=1,
        max_contracts=max_contracts,
        current_contracts=1,  # Start conservative
    )

    return ArbitrageExecutor(
        order_callback=order_callback,
        quote_callback=quote_callback,
        underlying_callback=underlying_callback,
        timing=timing,
        sizing=sizing,
        min_fill_rate=min_fill_rate,
    )


__all__ = [
    "ArbitrageExecutor",
    "ArbitrageOpportunity",
    "CreditMaximizer",
    "ExpirationCategory",
    "ExpirationTrader",
    "OptimizationMetrics",
    "OrderAttempt",
    "OrderResult",
    "PositionBalance",
    "PositionBalancer",
    "SizingParameters",
    "SpreadLeg",
    "StrikeRange",
    "TimingParameters",
    "TradingPhase",
    "create_arbitrage_executor",
]
