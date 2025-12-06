"""
Liquidity Scorer (UPGRADE-010 Sprint 4 Expansion)

Scores option liquidity for execution quality assessment.
Multi-factor scoring based on spread, volume, and open interest.

Author: Claude Code
Date: December 2025
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class LiquidityRating(Enum):
    """Liquidity quality rating."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    AVOID = "avoid"


@dataclass
class LiquidityConfig:
    """Configuration for liquidity scoring."""

    # Bid-ask spread thresholds (as % of mid price)
    excellent_spread_pct: float = 0.02  # 2%
    good_spread_pct: float = 0.05  # 5%
    acceptable_spread_pct: float = 0.10  # 10%
    max_spread_pct: float = 0.20  # 20%

    # Volume thresholds
    excellent_volume: int = 1000
    good_volume: int = 500
    acceptable_volume: int = 100
    min_volume: int = 10

    # Open interest thresholds
    excellent_oi: int = 5000
    good_oi: int = 2000
    acceptable_oi: int = 500
    min_oi: int = 50

    # Scoring weights
    spread_weight: float = 0.50
    volume_weight: float = 0.30
    oi_weight: float = 0.20

    # Minimum score thresholds
    min_tradeable_score: float = 40.0
    target_score: float = 70.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "excellent_spread_pct": self.excellent_spread_pct,
            "good_spread_pct": self.good_spread_pct,
            "acceptable_spread_pct": self.acceptable_spread_pct,
            "max_spread_pct": self.max_spread_pct,
            "excellent_volume": self.excellent_volume,
            "good_volume": self.good_volume,
            "acceptable_volume": self.acceptable_volume,
            "min_volume": self.min_volume,
            "excellent_oi": self.excellent_oi,
            "good_oi": self.good_oi,
            "acceptable_oi": self.acceptable_oi,
            "min_oi": self.min_oi,
            "spread_weight": self.spread_weight,
            "volume_weight": self.volume_weight,
            "oi_weight": self.oi_weight,
            "min_tradeable_score": self.min_tradeable_score,
            "target_score": self.target_score,
        }


@dataclass
class OptionLiquidityData:
    """Option contract liquidity data."""

    symbol: str
    bid: float
    ask: float
    volume: int
    open_interest: int
    underlying_price: float = 0.0
    strike: float = 0.0
    days_to_expiry: int = 0
    is_call: bool = True

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2 if self.ask > 0 else self.bid

    @property
    def spread(self) -> float:
        """Calculate absolute spread."""
        return self.ask - self.bid if self.ask > 0 else 0.0

    @property
    def spread_pct(self) -> float:
        """Calculate spread as percentage of mid price."""
        if self.mid_price > 0:
            return self.spread / self.mid_price
        return 1.0  # Max if no mid price

    @property
    def moneyness(self) -> float:
        """Calculate moneyness (for weighting)."""
        if self.underlying_price > 0 and self.strike > 0:
            return self.strike / self.underlying_price
        return 1.0


@dataclass
class LiquidityScore:
    """
    Liquidity assessment for an option contract.

    Attributes:
        score: Overall score 0-100
        rating: Quality rating (EXCELLENT, GOOD, etc.)
        bid_ask_score: Score component from bid-ask spread
        volume_score: Score component from trading volume
        oi_score: Score component from open interest
        is_tradeable: Whether contract meets minimum liquidity
        estimated_slippage_bps: Estimated slippage in basis points
        recommendation: Trading recommendation
    """

    score: float
    rating: LiquidityRating
    bid_ask_score: float
    volume_score: float
    oi_score: float
    is_tradeable: bool
    estimated_slippage_bps: float
    recommendation: str
    symbol: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "rating": self.rating.value,
            "bid_ask_score": self.bid_ask_score,
            "volume_score": self.volume_score,
            "oi_score": self.oi_score,
            "is_tradeable": self.is_tradeable,
            "estimated_slippage_bps": self.estimated_slippage_bps,
            "recommendation": self.recommendation,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ChainLiquiditySummary:
    """Summary of liquidity across an option chain."""

    underlying: str
    total_contracts: int
    tradeable_contracts: int
    excellent_count: int
    good_count: int
    acceptable_count: int
    poor_count: int
    avoid_count: int
    avg_score: float
    best_call_strike: float | None = None
    best_put_strike: float | None = None
    avg_spread_pct: float = 0.0
    avg_volume: float = 0.0
    avg_oi: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "underlying": self.underlying,
            "total_contracts": self.total_contracts,
            "tradeable_contracts": self.tradeable_contracts,
            "excellent_count": self.excellent_count,
            "good_count": self.good_count,
            "acceptable_count": self.acceptable_count,
            "poor_count": self.poor_count,
            "avoid_count": self.avoid_count,
            "avg_score": self.avg_score,
            "best_call_strike": self.best_call_strike,
            "best_put_strike": self.best_put_strike,
            "avg_spread_pct": self.avg_spread_pct,
            "avg_volume": self.avg_volume,
            "avg_oi": self.avg_oi,
            "timestamp": self.timestamp.isoformat(),
        }


class LiquidityScorer:
    """
    Score option liquidity for execution quality.

    Multi-factor scoring based on:
    - Bid-ask spread (50% weight default)
    - Trading volume (30% weight default)
    - Open interest (20% weight default)

    Example:
        scorer = LiquidityScorer()

        # Score a single contract
        contract_data = OptionLiquidityData(
            symbol="SPY_241220C450",
            bid=5.00,
            ask=5.10,
            volume=500,
            open_interest=2000,
        )
        score = scorer.score_contract(contract_data)
        print(f"Score: {score.score:.0f}/100 ({score.rating.value})")

        # Score an entire chain
        chain_scores = scorer.score_chain(contracts)
        liquid_contracts = scorer.filter_liquid_contracts(contracts, min_score=60)
    """

    def __init__(self, config: LiquidityConfig | None = None):
        """
        Initialize Liquidity Scorer.

        Args:
            config: Scoring configuration
        """
        self.config = config or LiquidityConfig()

    def score_contract(self, data: OptionLiquidityData) -> LiquidityScore:
        """
        Score a single option contract's liquidity.

        Args:
            data: Option liquidity data

        Returns:
            LiquidityScore with assessment
        """
        # Calculate spread score (0-100)
        spread_score = self._score_spread(data.spread_pct)

        # Calculate volume score (0-100)
        volume_score = self._score_volume(data.volume)

        # Calculate OI score (0-100)
        oi_score = self._score_open_interest(data.open_interest)

        # Weighted average
        total_score = (
            spread_score * self.config.spread_weight
            + volume_score * self.config.volume_weight
            + oi_score * self.config.oi_weight
        )

        # Determine rating
        rating = self._get_rating(total_score)

        # Check if tradeable (must have minimum score AND non-zero volume)
        is_tradeable = total_score >= self.config.min_tradeable_score and data.volume > 0

        # Estimate slippage (rough approximation)
        estimated_slippage_bps = data.spread_pct * 100 * 0.5  # Half spread

        # Generate recommendation
        recommendation = self._get_recommendation(rating, data)

        return LiquidityScore(
            score=total_score,
            rating=rating,
            bid_ask_score=spread_score,
            volume_score=volume_score,
            oi_score=oi_score,
            is_tradeable=is_tradeable,
            estimated_slippage_bps=estimated_slippage_bps,
            recommendation=recommendation,
            symbol=data.symbol,
        )

    def _score_spread(self, spread_pct: float) -> float:
        """Score based on bid-ask spread percentage."""
        if spread_pct <= self.config.excellent_spread_pct:
            return 100.0
        elif spread_pct <= self.config.good_spread_pct:
            # Linear interpolation 100 -> 80
            progress = (spread_pct - self.config.excellent_spread_pct) / (
                self.config.good_spread_pct - self.config.excellent_spread_pct
            )
            return 100.0 - progress * 20.0
        elif spread_pct <= self.config.acceptable_spread_pct:
            # Linear interpolation 80 -> 50
            progress = (spread_pct - self.config.good_spread_pct) / (
                self.config.acceptable_spread_pct - self.config.good_spread_pct
            )
            return 80.0 - progress * 30.0
        elif spread_pct <= self.config.max_spread_pct:
            # Linear interpolation 50 -> 20
            progress = (spread_pct - self.config.acceptable_spread_pct) / (
                self.config.max_spread_pct - self.config.acceptable_spread_pct
            )
            return 50.0 - progress * 30.0
        else:
            # Above max spread
            return max(0.0, 20.0 - (spread_pct - self.config.max_spread_pct) * 100)

    def _score_volume(self, volume: int) -> float:
        """Score based on trading volume."""
        if volume >= self.config.excellent_volume:
            return 100.0
        elif volume >= self.config.good_volume:
            progress = (volume - self.config.good_volume) / (self.config.excellent_volume - self.config.good_volume)
            return 80.0 + progress * 20.0
        elif volume >= self.config.acceptable_volume:
            progress = (volume - self.config.acceptable_volume) / (
                self.config.good_volume - self.config.acceptable_volume
            )
            return 50.0 + progress * 30.0
        elif volume >= self.config.min_volume:
            progress = (volume - self.config.min_volume) / (self.config.acceptable_volume - self.config.min_volume)
            return 20.0 + progress * 30.0
        else:
            return max(0.0, volume / self.config.min_volume * 20.0)

    def _score_open_interest(self, oi: int) -> float:
        """Score based on open interest."""
        if oi >= self.config.excellent_oi:
            return 100.0
        elif oi >= self.config.good_oi:
            progress = (oi - self.config.good_oi) / (self.config.excellent_oi - self.config.good_oi)
            return 80.0 + progress * 20.0
        elif oi >= self.config.acceptable_oi:
            progress = (oi - self.config.acceptable_oi) / (self.config.good_oi - self.config.acceptable_oi)
            return 50.0 + progress * 30.0
        elif oi >= self.config.min_oi:
            progress = (oi - self.config.min_oi) / (self.config.acceptable_oi - self.config.min_oi)
            return 20.0 + progress * 30.0
        else:
            return max(0.0, oi / self.config.min_oi * 20.0) if self.config.min_oi > 0 else 0.0

    def _get_rating(self, score: float) -> LiquidityRating:
        """Get rating from score."""
        if score >= 85:
            return LiquidityRating.EXCELLENT
        elif score >= 70:
            return LiquidityRating.GOOD
        elif score >= 50:
            return LiquidityRating.ACCEPTABLE
        elif score >= 30:
            return LiquidityRating.POOR
        else:
            return LiquidityRating.AVOID

    def _get_recommendation(self, rating: LiquidityRating, data: OptionLiquidityData) -> str:
        """Generate trading recommendation."""
        if rating == LiquidityRating.EXCELLENT:
            return "Excellent liquidity - trade at will"
        elif rating == LiquidityRating.GOOD:
            return "Good liquidity - standard execution"
        elif rating == LiquidityRating.ACCEPTABLE:
            if data.spread_pct > self.config.good_spread_pct:
                return "Wide spread - use limit orders, consider mid-price"
            else:
                return "Low volume - may have slippage on larger orders"
        elif rating == LiquidityRating.POOR:
            return "Poor liquidity - use caution, small size only"
        else:
            return "Avoid - insufficient liquidity for trading"

    def score_chain(
        self,
        contracts: list[OptionLiquidityData],
        underlying: str = "",
    ) -> dict[str, LiquidityScore]:
        """
        Score all contracts in an option chain.

        Args:
            contracts: List of option contract data
            underlying: Underlying symbol

        Returns:
            Dictionary mapping symbol to LiquidityScore
        """
        return {c.symbol: self.score_contract(c) for c in contracts}

    def filter_liquid_contracts(
        self,
        contracts: list[OptionLiquidityData],
        min_score: float | None = None,
        min_rating: LiquidityRating | None = None,
    ) -> list[OptionLiquidityData]:
        """
        Filter contracts to only include liquid ones.

        Args:
            contracts: List of option contract data
            min_score: Minimum score threshold (default: config.min_tradeable_score)
            min_rating: Minimum rating threshold

        Returns:
            List of contracts meeting liquidity criteria
        """
        if min_score is None:
            min_score = self.config.min_tradeable_score

        rating_order = [
            LiquidityRating.AVOID,
            LiquidityRating.POOR,
            LiquidityRating.ACCEPTABLE,
            LiquidityRating.GOOD,
            LiquidityRating.EXCELLENT,
        ]

        liquid_contracts = []
        for contract in contracts:
            score = self.score_contract(contract)

            # Check score threshold
            if score.score < min_score:
                continue

            # Check rating threshold if specified
            if min_rating is not None:
                if rating_order.index(score.rating) < rating_order.index(min_rating):
                    continue

            liquid_contracts.append(contract)

        return liquid_contracts

    def get_chain_summary(
        self,
        contracts: list[OptionLiquidityData],
        underlying: str = "",
    ) -> ChainLiquiditySummary:
        """
        Get summary of liquidity across option chain.

        Args:
            contracts: List of option contract data
            underlying: Underlying symbol

        Returns:
            ChainLiquiditySummary with statistics
        """
        if not contracts:
            return ChainLiquiditySummary(
                underlying=underlying,
                total_contracts=0,
                tradeable_contracts=0,
                excellent_count=0,
                good_count=0,
                acceptable_count=0,
                poor_count=0,
                avoid_count=0,
                avg_score=0.0,
            )

        scores = [self.score_contract(c) for c in contracts]

        # Count by rating
        excellent_count = sum(1 for s in scores if s.rating == LiquidityRating.EXCELLENT)
        good_count = sum(1 for s in scores if s.rating == LiquidityRating.GOOD)
        acceptable_count = sum(1 for s in scores if s.rating == LiquidityRating.ACCEPTABLE)
        poor_count = sum(1 for s in scores if s.rating == LiquidityRating.POOR)
        avoid_count = sum(1 for s in scores if s.rating == LiquidityRating.AVOID)

        # Calculate averages
        avg_score = sum(s.score for s in scores) / len(scores)
        avg_spread_pct = sum(c.spread_pct for c in contracts) / len(contracts)
        avg_volume = sum(c.volume for c in contracts) / len(contracts)
        avg_oi = sum(c.open_interest for c in contracts) / len(contracts)

        # Find best call and put strikes
        best_call_score = 0.0
        best_call_strike = None
        best_put_score = 0.0
        best_put_strike = None

        for contract, score in zip(contracts, scores):
            if contract.is_call:
                if score.score > best_call_score:
                    best_call_score = score.score
                    best_call_strike = contract.strike
            else:
                if score.score > best_put_score:
                    best_put_score = score.score
                    best_put_strike = contract.strike

        return ChainLiquiditySummary(
            underlying=underlying,
            total_contracts=len(contracts),
            tradeable_contracts=sum(1 for s in scores if s.is_tradeable),
            excellent_count=excellent_count,
            good_count=good_count,
            acceptable_count=acceptable_count,
            poor_count=poor_count,
            avoid_count=avoid_count,
            avg_score=avg_score,
            best_call_strike=best_call_strike,
            best_put_strike=best_put_strike,
            avg_spread_pct=avg_spread_pct,
            avg_volume=avg_volume,
            avg_oi=avg_oi,
        )

    def get_best_contracts(
        self,
        contracts: list[OptionLiquidityData],
        n: int = 5,
        calls_only: bool = False,
        puts_only: bool = False,
    ) -> list[tuple[OptionLiquidityData, LiquidityScore]]:
        """
        Get the most liquid contracts.

        Args:
            contracts: List of option contract data
            n: Number of contracts to return
            calls_only: Only consider calls
            puts_only: Only consider puts

        Returns:
            List of (contract, score) tuples sorted by score
        """
        # Filter by type if needed
        filtered = contracts
        if calls_only:
            filtered = [c for c in contracts if c.is_call]
        elif puts_only:
            filtered = [c for c in contracts if not c.is_call]

        # Score and sort
        scored = [(c, self.score_contract(c)) for c in filtered]
        scored.sort(key=lambda x: x[1].score, reverse=True)

        return scored[:n]


def create_liquidity_scorer(
    spread_weight: float = 0.50,
    volume_weight: float = 0.30,
    min_score: float = 40.0,
) -> LiquidityScorer:
    """
    Factory function to create a LiquidityScorer.

    Args:
        spread_weight: Weight for bid-ask spread (0-1)
        volume_weight: Weight for volume (0-1)
        min_score: Minimum score for tradeable

    Returns:
        Configured LiquidityScorer instance
    """
    oi_weight = 1.0 - spread_weight - volume_weight
    config = LiquidityConfig(
        spread_weight=spread_weight,
        volume_weight=volume_weight,
        oi_weight=max(0.0, oi_weight),
        min_tradeable_score=min_score,
    )
    return LiquidityScorer(config=config)
