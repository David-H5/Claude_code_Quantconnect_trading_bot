"""
Bid-Ask Spread Analysis Module

Provides execution cost analysis and optimization:
- Spread quality metrics
- Execution cost estimation
- Optimal order timing signals
- Liquidity scoring
- Market microstructure analysis

Based on patterns from quantitative execution research.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class SpreadQuality(Enum):
    """Spread quality classification."""

    EXCELLENT = "excellent"  # < 0.05% spread
    GOOD = "good"  # 0.05% - 0.1%
    FAIR = "fair"  # 0.1% - 0.3%
    POOR = "poor"  # 0.3% - 0.5%
    VERY_POOR = "very_poor"  # > 0.5%


class ExecutionUrgency(Enum):
    """Execution urgency level."""

    PASSIVE = "passive"  # Can wait for best price
    NORMAL = "normal"  # Standard execution
    AGGRESSIVE = "aggressive"  # Need to execute soon
    IMMEDIATE = "immediate"  # Market order required


@dataclass
class SpreadSnapshot:
    """Snapshot of bid-ask spread data."""

    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: int = 0
    ask_size: int = 0
    last_price: float = 0.0
    volume: int = 0

    @property
    def mid_price(self) -> float:
        """Calculate mid-price."""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> float:
        """Absolute spread."""
        return self.ask_price - self.bid_price

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid-price."""
        if self.mid_price == 0:
            return 0.0
        return self.spread / self.mid_price

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return self.spread_pct * 10000

    @property
    def imbalance(self) -> float:
        """
        Order book imbalance.

        Positive = more bid pressure (bullish)
        Negative = more ask pressure (bearish)
        """
        total_size = self.bid_size + self.ask_size
        if total_size == 0:
            return 0.0
        return (self.bid_size - self.ask_size) / total_size

    def get_quality(self) -> SpreadQuality:
        """Classify spread quality."""
        pct = self.spread_pct
        if pct < 0.0005:
            return SpreadQuality.EXCELLENT
        elif pct < 0.001:
            return SpreadQuality.GOOD
        elif pct < 0.003:
            return SpreadQuality.FAIR
        elif pct < 0.005:
            return SpreadQuality.POOR
        else:
            return SpreadQuality.VERY_POOR


@dataclass
class ExecutionCostEstimate:
    """Estimated execution costs."""

    half_spread_cost: float  # Half spread (crossing spread cost)
    market_impact: float  # Estimated market impact
    timing_cost: float  # Cost of delayed execution
    total_cost: float  # Total estimated cost
    cost_bps: float  # Total cost in basis points
    recommended_order_type: str  # "limit" or "market"
    confidence: float = 0.0  # Confidence in estimate (0-1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "half_spread_cost": self.half_spread_cost,
            "market_impact": self.market_impact,
            "timing_cost": self.timing_cost,
            "total_cost": self.total_cost,
            "cost_bps": self.cost_bps,
            "recommended_order_type": self.recommended_order_type,
            "confidence": self.confidence,
        }


@dataclass
class SpreadMetrics:
    """Aggregated spread metrics over a period."""

    avg_spread_bps: float = 0.0
    min_spread_bps: float = 0.0
    max_spread_bps: float = 0.0
    std_spread_bps: float = 0.0
    avg_imbalance: float = 0.0
    avg_bid_size: float = 0.0
    avg_ask_size: float = 0.0
    samples: int = 0
    quality_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "avg_spread_bps": self.avg_spread_bps,
            "min_spread_bps": self.min_spread_bps,
            "max_spread_bps": self.max_spread_bps,
            "std_spread_bps": self.std_spread_bps,
            "avg_imbalance": self.avg_imbalance,
            "avg_bid_size": self.avg_bid_size,
            "avg_ask_size": self.avg_ask_size,
            "samples": self.samples,
            "quality_distribution": self.quality_distribution,
        }


class SpreadAnalyzer:
    """
    Analyzes bid-ask spreads for execution optimization.

    Provides metrics, cost estimates, and execution recommendations.
    """

    def __init__(
        self,
        max_history: int = 1000,
        impact_coefficient: float = 0.1,
    ):
        """
        Initialize spread analyzer.

        Args:
            max_history: Maximum snapshots to retain
            impact_coefficient: Market impact coefficient (Kyle's lambda)
        """
        self.max_history = max_history
        self.impact_coefficient = impact_coefficient
        self.history: dict[str, list[SpreadSnapshot]] = {}
        self.daily_volume: dict[str, float] = {}

    def update(
        self,
        symbol: str,
        bid: float,
        ask: float,
        bid_size: int = 0,
        ask_size: int = 0,
        last_price: float = 0.0,
        volume: int = 0,
        timestamp: datetime | None = None,
    ) -> SpreadSnapshot:
        """
        Update with new quote data.

        Args:
            symbol: Symbol
            bid: Bid price
            ask: Ask price
            bid_size: Bid size
            ask_size: Ask size
            last_price: Last trade price
            volume: Volume
            timestamp: Timestamp (default now)

        Returns:
            SpreadSnapshot
        """
        if timestamp is None:
            timestamp = datetime.now()

        snapshot = SpreadSnapshot(
            timestamp=timestamp,
            bid_price=bid,
            ask_price=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            last_price=last_price,
            volume=volume,
        )

        if symbol not in self.history:
            self.history[symbol] = []

        self.history[symbol].append(snapshot)

        # Trim history
        if len(self.history[symbol]) > self.max_history:
            self.history[symbol] = self.history[symbol][-self.max_history :]

        return snapshot

    def get_current_spread(self, symbol: str) -> SpreadSnapshot | None:
        """Get most recent spread snapshot."""
        if symbol not in self.history or not self.history[symbol]:
            return None
        return self.history[symbol][-1]

    def get_spread_metrics(
        self,
        symbol: str,
        lookback_minutes: int = 60,
    ) -> SpreadMetrics:
        """
        Calculate spread metrics over a lookback period.

        Args:
            symbol: Symbol
            lookback_minutes: Minutes of history to analyze

        Returns:
            SpreadMetrics
        """
        if symbol not in self.history or not self.history[symbol]:
            return SpreadMetrics()

        cutoff = datetime.now() - timedelta(minutes=lookback_minutes)
        samples = [s for s in self.history[symbol] if s.timestamp >= cutoff]

        if not samples:
            return SpreadMetrics()

        spreads_bps = [s.spread_bps for s in samples]
        imbalances = [s.imbalance for s in samples]
        bid_sizes = [s.bid_size for s in samples]
        ask_sizes = [s.ask_size for s in samples]

        # Calculate statistics
        avg_spread = sum(spreads_bps) / len(spreads_bps)
        min_spread = min(spreads_bps)
        max_spread = max(spreads_bps)

        # Standard deviation
        variance = sum((s - avg_spread) ** 2 for s in spreads_bps) / len(spreads_bps)
        std_spread = math.sqrt(variance)

        # Quality distribution
        quality_dist = {}
        for s in samples:
            q = s.get_quality().value
            quality_dist[q] = quality_dist.get(q, 0) + 1

        return SpreadMetrics(
            avg_spread_bps=avg_spread,
            min_spread_bps=min_spread,
            max_spread_bps=max_spread,
            std_spread_bps=std_spread,
            avg_imbalance=sum(imbalances) / len(imbalances),
            avg_bid_size=sum(bid_sizes) / len(bid_sizes) if bid_sizes else 0,
            avg_ask_size=sum(ask_sizes) / len(ask_sizes) if ask_sizes else 0,
            samples=len(samples),
            quality_distribution=quality_dist,
        )

    def estimate_execution_cost(
        self,
        symbol: str,
        quantity: int,
        side: str,  # "buy" or "sell"
        urgency: ExecutionUrgency = ExecutionUrgency.NORMAL,
        average_daily_volume: float | None = None,
    ) -> ExecutionCostEstimate:
        """
        Estimate total execution cost.

        Args:
            symbol: Symbol
            quantity: Order quantity
            side: "buy" or "sell"
            urgency: Execution urgency
            average_daily_volume: ADV for impact calculation

        Returns:
            ExecutionCostEstimate
        """
        current = self.get_current_spread(symbol)
        if current is None:
            return ExecutionCostEstimate(
                half_spread_cost=0,
                market_impact=0,
                timing_cost=0,
                total_cost=0,
                cost_bps=0,
                recommended_order_type="limit",
                confidence=0,
            )

        # 1. Half-spread cost (crossing the spread)
        half_spread = current.spread / 2
        half_spread_cost = half_spread * quantity

        # 2. Market impact (simplified square-root model)
        # Impact ≈ lambda * sqrt(Q/ADV) * sigma * P
        if average_daily_volume and average_daily_volume > 0:
            participation = quantity / average_daily_volume
            # Simplified: assume daily volatility of 2%
            daily_vol = 0.02
            impact_pct = self.impact_coefficient * math.sqrt(participation) * daily_vol
            market_impact = current.mid_price * impact_pct * quantity
        else:
            # Without ADV, estimate based on order book
            if current.bid_size + current.ask_size > 0:
                relevant_size = current.ask_size if side == "buy" else current.bid_size
                if relevant_size > 0 and quantity > relevant_size:
                    # Order exceeds visible liquidity
                    layers = quantity / relevant_size
                    impact_pct = 0.001 * layers  # 1 bp per layer
                    market_impact = current.mid_price * impact_pct * quantity
                else:
                    market_impact = 0
            else:
                market_impact = 0

        # 3. Timing cost (opportunity cost of waiting)
        metrics = self.get_spread_metrics(symbol, lookback_minutes=30)
        if urgency == ExecutionUrgency.PASSIVE:
            timing_cost = 0  # Willing to wait
        elif urgency == ExecutionUrgency.IMMEDIATE:
            timing_cost = half_spread_cost * 0.5  # Pay premium for speed
        else:
            # Normal/Aggressive: some timing cost
            timing_cost = half_spread_cost * 0.2

        total_cost = half_spread_cost + market_impact + timing_cost
        cost_bps = (total_cost / (current.mid_price * quantity)) * 10000 if quantity > 0 else 0

        # Determine recommended order type
        spread_quality = current.get_quality()
        if urgency == ExecutionUrgency.IMMEDIATE:
            recommended = "market"
        elif spread_quality in (SpreadQuality.EXCELLENT, SpreadQuality.GOOD):
            recommended = "limit"  # Tight spread, try to capture
        elif urgency == ExecutionUrgency.AGGRESSIVE:
            recommended = "market"  # Need execution
        else:
            # Fair/Poor spread with normal urgency
            # Use limit at mid or slightly aggressive
            recommended = "limit"

        # Confidence based on data quality
        confidence = min(1.0, metrics.samples / 30)  # Need ~30 samples

        return ExecutionCostEstimate(
            half_spread_cost=half_spread_cost,
            market_impact=market_impact,
            timing_cost=timing_cost,
            total_cost=total_cost,
            cost_bps=cost_bps,
            recommended_order_type=recommended,
            confidence=confidence,
        )

    def get_optimal_limit_price(
        self,
        symbol: str,
        side: str,
        urgency: ExecutionUrgency = ExecutionUrgency.NORMAL,
    ) -> float | None:
        """
        Calculate optimal limit price based on spread analysis.

        Args:
            symbol: Symbol
            side: "buy" or "sell"
            urgency: Execution urgency

        Returns:
            Optimal limit price
        """
        current = self.get_current_spread(symbol)
        if current is None:
            return None

        metrics = self.get_spread_metrics(symbol, lookback_minutes=30)

        if side == "buy":
            if urgency == ExecutionUrgency.IMMEDIATE:
                # Take the ask
                return current.ask_price
            elif urgency == ExecutionUrgency.PASSIVE:
                # Post at bid
                return current.bid_price
            elif urgency == ExecutionUrgency.AGGRESSIVE:
                # Above mid, close to ask
                return current.mid_price + current.spread * 0.3
            else:
                # Normal: at or slightly above mid
                return current.mid_price + current.spread * 0.1
        else:  # sell
            if urgency == ExecutionUrgency.IMMEDIATE:
                # Take the bid
                return current.bid_price
            elif urgency == ExecutionUrgency.PASSIVE:
                # Post at ask
                return current.ask_price
            elif urgency == ExecutionUrgency.AGGRESSIVE:
                # Below mid, close to bid
                return current.mid_price - current.spread * 0.3
            else:
                # Normal: at or slightly below mid
                return current.mid_price - current.spread * 0.1

    def is_execution_favorable(
        self,
        symbol: str,
        side: str,
        threshold_bps: float = 10.0,
    ) -> tuple[bool, str]:
        """
        Check if current conditions are favorable for execution.

        Args:
            symbol: Symbol
            side: "buy" or "sell"
            threshold_bps: Maximum acceptable spread in bps

        Returns:
            (is_favorable, reason)
        """
        current = self.get_current_spread(symbol)
        if current is None:
            return False, "No spread data available"

        metrics = self.get_spread_metrics(symbol, lookback_minutes=60)

        # Check spread quality
        if current.spread_bps > threshold_bps:
            return False, f"Spread too wide: {current.spread_bps:.1f} bps > {threshold_bps} bps"

        # Check if spread is better than recent average
        if metrics.samples > 10 and current.spread_bps > metrics.avg_spread_bps * 1.5:
            return False, f"Spread wider than average: {current.spread_bps:.1f} vs {metrics.avg_spread_bps:.1f} bps"

        # Check order book imbalance for directional trades
        if side == "buy" and current.imbalance < -0.5:
            return False, f"Heavy sell pressure (imbalance: {current.imbalance:.2f})"
        if side == "sell" and current.imbalance > 0.5:
            return False, f"Heavy buy pressure (imbalance: {current.imbalance:.2f})"

        # Conditions look favorable
        quality = current.get_quality()
        return True, f"Favorable conditions ({quality.value} spread: {current.spread_bps:.1f} bps)"

    def get_liquidity_score(self, symbol: str) -> float:
        """
        Calculate liquidity score (0-100).

        Higher score = better liquidity.
        """
        current = self.get_current_spread(symbol)
        if current is None:
            return 0.0

        metrics = self.get_spread_metrics(symbol, lookback_minutes=60)

        # Spread component (tighter = better)
        spread_score = max(0, 50 - current.spread_bps * 5)  # 50 at 0 bps, 0 at 10 bps

        # Depth component (more size = better)
        total_size = current.bid_size + current.ask_size
        depth_score = min(50, total_size / 100)  # 50 for 5000+ shares each side

        return min(100, spread_score + depth_score)

    def get_summary(self, symbol: str) -> dict[str, Any]:
        """Get complete spread analysis summary."""
        current = self.get_current_spread(symbol)
        metrics = self.get_spread_metrics(symbol)

        if current is None:
            return {"symbol": symbol, "error": "No data"}

        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current": {
                "bid": current.bid_price,
                "ask": current.ask_price,
                "mid": current.mid_price,
                "spread_bps": current.spread_bps,
                "quality": current.get_quality().value,
                "imbalance": current.imbalance,
                "bid_size": current.bid_size,
                "ask_size": current.ask_size,
            },
            "metrics": metrics.to_dict(),
            "liquidity_score": self.get_liquidity_score(symbol),
        }


def create_spread_analyzer(
    max_history: int = 1000,
    impact_coefficient: float = 0.1,
) -> SpreadAnalyzer:
    """
    Create spread analyzer instance.

    Args:
        max_history: Maximum snapshots per symbol
        impact_coefficient: Market impact coefficient

    Returns:
        SpreadAnalyzer instance
    """
    return SpreadAnalyzer(
        max_history=max_history,
        impact_coefficient=impact_coefficient,
    )


__all__ = [
    "ExecutionCostEstimate",
    "ExecutionUrgency",
    "SpreadAnalyzer",
    "SpreadMetrics",
    "SpreadQuality",
    "SpreadSnapshot",
    "create_spread_analyzer",
]
