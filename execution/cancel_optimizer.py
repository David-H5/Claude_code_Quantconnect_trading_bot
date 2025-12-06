"""
Adaptive Cancel Timing Optimizer

Dynamically determines optimal cancel timing for unfilled orders
based on market conditions and historical fill patterns.

Key insight: Orders that don't fill in 2-3 seconds typically won't fill at all.
This module learns optimal timing from data.

Part of UPGRADE-010 Sprint 4: Risk & Execution.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from execution.fill_predictor import FillOutcome


logger = logging.getLogger(__name__)


class CancelReason(Enum):
    """Reasons for cancel recommendation."""

    TIMEOUT_EXCEEDED = "timeout_exceeded"
    LOW_FILL_PROBABILITY = "low_fill_probability"
    MARKET_MOVED = "market_moved"
    VOLATILITY_SPIKE = "volatility_spike"
    SPREAD_WIDENED = "spread_widened"
    BETTER_OPPORTUNITY = "better_opportunity"
    MANUAL = "manual"


@dataclass
class CancelTimingFeatures:
    """Features for cancel timing prediction."""

    spread_bps: float  # Current bid-ask spread
    fill_probability: float  # Predicted fill probability
    time_since_submit: float  # Seconds since order submitted
    partial_fill_pct: float  # Percentage already filled (0-1)
    volatility_regime: str  # low/normal/high/extreme
    order_age_percentile: float  # How old vs typical fills (0-1)
    price_movement_since_submit: float  # % move since submit
    spread_change_since_submit: float  # Spread change in bps
    volume_since_submit: int  # Contracts traded since submit

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spread_bps": self.spread_bps,
            "fill_probability": self.fill_probability,
            "time_since_submit": self.time_since_submit,
            "partial_fill_pct": self.partial_fill_pct,
            "volatility_regime": self.volatility_regime,
            "order_age_percentile": self.order_age_percentile,
            "price_movement": self.price_movement_since_submit,
            "spread_change": self.spread_change_since_submit,
            "volume_since_submit": self.volume_since_submit,
        }


@dataclass
class CancelDecision:
    """Cancel timing recommendation."""

    should_cancel: bool
    optimal_wait_seconds: float  # How long to wait before canceling
    confidence: float  # Confidence in recommendation (0-1)
    reason: CancelReason
    reasoning: str  # Human-readable explanation
    suggested_price_adjustment: float | None = None  # Price adjustment if repricing

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "should_cancel": self.should_cancel,
            "optimal_wait_seconds": self.optimal_wait_seconds,
            "confidence": self.confidence,
            "reason": self.reason.value,
            "reasoning": self.reasoning,
            "suggested_price_adjustment": self.suggested_price_adjustment,
        }


@dataclass
class TimingRecord:
    """Historical record for timing analysis."""

    order_id: str
    symbol: str
    submit_time: datetime
    outcome: FillOutcome
    time_to_outcome: float  # Seconds
    features_at_submit: CancelTimingFeatures
    final_fill_pct: float = 0.0  # Final fill percentage
    was_repriced: bool = False


@dataclass
class TimingStatistics:
    """Statistics for cancel timing optimization."""

    total_orders: int = 0
    filled_count: int = 0
    avg_fill_time: float = 0.0
    median_fill_time: float = 0.0
    percentile_90_fill_time: float = 0.0
    optimal_cancel_time: float = 2.5  # Default 2.5 seconds
    fill_rate_by_wait_time: dict[float, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_orders": self.total_orders,
            "filled_count": self.filled_count,
            "fill_rate": self.filled_count / self.total_orders if self.total_orders > 0 else 0,
            "avg_fill_time": self.avg_fill_time,
            "median_fill_time": self.median_fill_time,
            "percentile_90_fill_time": self.percentile_90_fill_time,
            "optimal_cancel_time": self.optimal_cancel_time,
        }


class CancelOptimizer:
    """
    Adaptive cancel timing based on market conditions.

    Learns from historical fill patterns to determine optimal
    cancel timing for unfilled orders.
    """

    # Base timeout (user observed: orders not filling in 2-3s won't fill)
    DEFAULT_TIMEOUT = 2.5

    # Minimum and maximum timeouts
    MIN_TIMEOUT = 1.0
    MAX_TIMEOUT = 15.0

    def __init__(
        self,
        base_timeout: float = 2.5,
        learning_rate: float = 0.1,
        min_samples_for_learning: int = 20,
    ):
        """
        Initialize cancel optimizer.

        Args:
            base_timeout: Default cancel timeout in seconds
            learning_rate: Learning rate for updating estimates
            min_samples_for_learning: Minimum samples before using learned timing
        """
        self.base_timeout = base_timeout
        self.learning_rate = learning_rate
        self.min_samples = min_samples_for_learning

        # Historical records
        self.history: dict[str, list[TimingRecord]] = defaultdict(list)

        # Per-symbol statistics
        self.statistics: dict[str, TimingStatistics] = {}

        # Global fill time distribution
        self.fill_times: list[float] = []

        # Timeout adjustments by condition
        self.timeout_adjustments = {
            "vol_low": 1.2,  # 20% longer in low vol
            "vol_normal": 1.0,
            "vol_high": 0.8,  # 20% shorter in high vol
            "vol_extreme": 0.6,  # 40% shorter in extreme vol
            "spread_tight": 1.3,  # Longer for tight spreads (more patient)
            "spread_wide": 0.7,  # Shorter for wide spreads
        }

    def get_optimal_timeout(
        self,
        features: CancelTimingFeatures,
        symbol: str | None = None,
    ) -> CancelDecision:
        """
        Calculate optimal cancel timeout dynamically.

        Args:
            features: Current order features
            symbol: Optional symbol for symbol-specific learning

        Returns:
            CancelDecision with recommendation
        """
        # Start with base timeout
        timeout = self.base_timeout

        # Adjust for volatility regime
        vol_key = f"vol_{features.volatility_regime}"
        timeout *= self.timeout_adjustments.get(vol_key, 1.0)

        # Adjust for spread
        if features.spread_bps < 10:
            timeout *= self.timeout_adjustments["spread_tight"]
        elif features.spread_bps > 50:
            timeout *= self.timeout_adjustments["spread_wide"]

        # Adjust for fill probability
        if features.fill_probability < 0.25:
            # Low probability - cancel quickly
            timeout *= 0.6
        elif features.fill_probability > 0.7:
            # High probability - be more patient
            timeout *= 1.3

        # Adjust for partial fills
        if features.partial_fill_pct > 0:
            # Already partially filled - be more patient
            timeout *= 1.0 + features.partial_fill_pct

        # Use symbol-specific statistics if available
        if symbol and symbol in self.statistics:
            stats = self.statistics[symbol]
            if stats.total_orders >= self.min_samples:
                # Use learned optimal time
                timeout = self.learning_rate * stats.optimal_cancel_time + (1 - self.learning_rate) * timeout

        # Clamp to reasonable range
        timeout = max(self.MIN_TIMEOUT, min(self.MAX_TIMEOUT, timeout))

        # Determine if should cancel now
        should_cancel = features.time_since_submit >= timeout

        # Determine reason
        if should_cancel:
            if features.fill_probability < 0.25:
                reason = CancelReason.LOW_FILL_PROBABILITY
                reasoning = f"Fill probability {features.fill_probability:.1%} below threshold"
            elif abs(features.price_movement_since_submit) > 0.5:
                reason = CancelReason.MARKET_MOVED
                reasoning = f"Market moved {features.price_movement_since_submit:.2%} since submit"
            elif features.spread_change_since_submit > 10:
                reason = CancelReason.SPREAD_WIDENED
                reasoning = f"Spread widened {features.spread_change_since_submit:.1f} bps"
            else:
                reason = CancelReason.TIMEOUT_EXCEEDED
                reasoning = f"Exceeded optimal timeout of {timeout:.1f}s"
        else:
            reason = CancelReason.TIMEOUT_EXCEEDED
            remaining = timeout - features.time_since_submit
            reasoning = f"Wait {remaining:.1f}s more (optimal timeout: {timeout:.1f}s)"

        # Calculate suggested price adjustment if repricing
        price_adjustment = None
        if should_cancel and features.fill_probability < 0.5:
            # Suggest more aggressive pricing
            # Move towards ask for buys, towards bid for sells
            price_adjustment = features.spread_bps * 0.1 / 100  # 10% of spread

        # Confidence based on data availability
        if symbol and symbol in self.statistics:
            stats = self.statistics[symbol]
            confidence = min(0.9, 0.5 + stats.total_orders / 200)
        else:
            confidence = 0.5

        return CancelDecision(
            should_cancel=should_cancel,
            optimal_wait_seconds=timeout,
            confidence=confidence,
            reason=reason,
            reasoning=reasoning,
            suggested_price_adjustment=price_adjustment,
        )

    def record_outcome(
        self,
        order_id: str,
        symbol: str,
        outcome: FillOutcome,
        time_to_outcome: float,
        features: CancelTimingFeatures,
        fill_pct: float = 0.0,
        was_repriced: bool = False,
    ) -> None:
        """
        Record outcome for learning.

        Args:
            order_id: Order identifier
            symbol: Symbol
            outcome: Fill outcome
            time_to_outcome: Time in seconds to outcome
            features: Features at submit time
            fill_pct: Final fill percentage
            was_repriced: Whether order was repriced
        """
        record = TimingRecord(
            order_id=order_id,
            symbol=symbol,
            submit_time=datetime.now() - timedelta(seconds=time_to_outcome),
            outcome=outcome,
            time_to_outcome=time_to_outcome,
            features_at_submit=features,
            final_fill_pct=fill_pct,
            was_repriced=was_repriced,
        )

        self.history[symbol].append(record)

        # Track fill times
        if outcome in (FillOutcome.FILLED, FillOutcome.PARTIAL):
            self.fill_times.append(time_to_outcome)

        # Update statistics
        self._update_statistics(symbol)

        # Keep history manageable
        if len(self.history[symbol]) > 500:
            self.history[symbol] = self.history[symbol][-500:]

    def _update_statistics(self, symbol: str) -> None:
        """Update statistics for symbol."""
        records = self.history[symbol]
        if not records:
            return

        stats = TimingStatistics()
        stats.total_orders = len(records)

        fill_times = []
        for record in records:
            if record.outcome in (FillOutcome.FILLED, FillOutcome.PARTIAL):
                stats.filled_count += 1
                fill_times.append(record.time_to_outcome)

        if fill_times:
            fill_times_arr = np.array(fill_times)
            stats.avg_fill_time = float(np.mean(fill_times_arr))
            stats.median_fill_time = float(np.median(fill_times_arr))
            stats.percentile_90_fill_time = float(np.percentile(fill_times_arr, 90))

            # Optimal cancel time is 90th percentile of fill times
            # Orders not filled by then are unlikely to fill
            stats.optimal_cancel_time = stats.percentile_90_fill_time

        # Calculate fill rate by wait time buckets
        wait_buckets = [1, 2, 3, 5, 10, 15]
        for bucket in wait_buckets:
            orders_waited = [r for r in records if r.time_to_outcome >= bucket]
            if orders_waited:
                filled = sum(1 for r in orders_waited if r.outcome in (FillOutcome.FILLED, FillOutcome.PARTIAL))
                stats.fill_rate_by_wait_time[bucket] = filled / len(orders_waited)

        self.statistics[symbol] = stats

    def get_statistics(self, symbol: str) -> TimingStatistics | None:
        """Get timing statistics for symbol."""
        return self.statistics.get(symbol)

    def get_global_statistics(self) -> TimingStatistics:
        """Get global timing statistics across all symbols."""
        stats = TimingStatistics()

        for symbol_stats in self.statistics.values():
            stats.total_orders += symbol_stats.total_orders
            stats.filled_count += symbol_stats.filled_count

        if self.fill_times:
            fill_times_arr = np.array(self.fill_times)
            stats.avg_fill_time = float(np.mean(fill_times_arr))
            stats.median_fill_time = float(np.median(fill_times_arr))
            stats.percentile_90_fill_time = float(np.percentile(fill_times_arr, 90))
            stats.optimal_cancel_time = stats.percentile_90_fill_time

        return stats

    def analyze_timing_patterns(self) -> dict[str, Any]:
        """Analyze timing patterns across all data."""
        if not self.fill_times:
            return {"error": "No fill data available"}

        fill_times = np.array(self.fill_times)

        return {
            "total_fills": len(fill_times),
            "mean_fill_time": float(np.mean(fill_times)),
            "median_fill_time": float(np.median(fill_times)),
            "std_fill_time": float(np.std(fill_times)),
            "percentiles": {
                "p10": float(np.percentile(fill_times, 10)),
                "p25": float(np.percentile(fill_times, 25)),
                "p50": float(np.percentile(fill_times, 50)),
                "p75": float(np.percentile(fill_times, 75)),
                "p90": float(np.percentile(fill_times, 90)),
                "p95": float(np.percentile(fill_times, 95)),
                "p99": float(np.percentile(fill_times, 99)),
            },
            "recommended_timeout": float(np.percentile(fill_times, 90)),
            "symbols_tracked": len(self.statistics),
        }

    def should_reprice(
        self,
        features: CancelTimingFeatures,
        current_price: float,
        best_bid: float,
        best_ask: float,
    ) -> tuple[bool, float | None]:
        """
        Determine if order should be repriced and at what price.

        Args:
            features: Current features
            current_price: Current limit price
            best_bid: Current best bid
            best_ask: Current best ask

        Returns:
            Tuple of (should_reprice, new_price)
        """
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        # If already at or better than mid, don't reprice
        if current_price >= mid:
            return False, None

        # If fill probability is low and we've waited
        if features.fill_probability < 0.3 and features.time_since_submit > 2.0:
            # Move to mid price
            return True, mid

        # If spread widened significantly
        if features.spread_change_since_submit > 20:
            # Move to inside market
            new_price = best_bid + spread * 0.4
            return True, new_price

        return False, None

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of optimizer state."""
        global_stats = self.get_global_statistics()

        return {
            "base_timeout": self.base_timeout,
            "symbols_tracked": len(self.statistics),
            "total_records": sum(len(h) for h in self.history.values()),
            "global_statistics": global_stats.to_dict(),
            "timeout_adjustments": self.timeout_adjustments,
        }


def create_cancel_optimizer(base_timeout: float = 2.5) -> CancelOptimizer:
    """Factory function to create cancel optimizer."""
    return CancelOptimizer(base_timeout=base_timeout)


__all__ = [
    "CancelDecision",
    "CancelOptimizer",
    "CancelReason",
    "CancelTimingFeatures",
    "TimingRecord",
    "TimingStatistics",
    "create_cancel_optimizer",
]
