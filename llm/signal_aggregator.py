"""
Multi-Source Signal Aggregator Module

Combines signals from multiple data sources (Reddit, News, Earnings, etc.)
with intelligent weighting and conflict resolution.
Part of UPGRADE-010 Sprint 3 Expansion - Intelligence & Data Sources.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class SignalSource(Enum):
    """Sources of trading signals."""

    REDDIT = "reddit"
    NEWS = "news"
    EARNINGS = "earnings"
    TECHNICAL = "technical"
    FUNDAMENTALS = "fundamentals"
    OPTIONS_FLOW = "options_flow"
    INSIDER_TRADES = "insider_trades"
    ANALYST = "analyst"
    CUSTOM = "custom"


class SignalDirection(Enum):
    """Direction of a trading signal."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class AggregatedAction(Enum):
    """Recommended action from aggregated signals."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    CONFLICTING = "conflicting"


@dataclass
class SourceSignal:
    """A signal from a single source."""

    source: SignalSource
    ticker: str
    direction: SignalDirection
    confidence: float  # 0-1
    strength: float  # 0-1, how strong the signal is
    timestamp: datetime
    expiry: datetime | None = None  # When the signal expires
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source.value,
            "ticker": self.ticker,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "strength": self.strength,
            "timestamp": self.timestamp.isoformat(),
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "metadata": self.metadata,
        }

    @property
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        if self.expiry is None:
            return False
        return datetime.now() > self.expiry

    @property
    def weighted_score(self) -> float:
        """Calculate weighted score for ranking."""
        direction_mult = {
            SignalDirection.BULLISH: 1.0,
            SignalDirection.BEARISH: -1.0,
            SignalDirection.NEUTRAL: 0.0,
        }
        return direction_mult[self.direction] * self.confidence * self.strength


@dataclass
class AggregatedSignal:
    """An aggregated signal from multiple sources."""

    ticker: str
    action: AggregatedAction
    direction: SignalDirection

    # Composite scores
    composite_score: float  # -1 (bearish) to +1 (bullish)
    confidence: float  # 0-1
    actionability: float  # 0-1, how actionable is the signal

    # Source agreement
    agreement_score: float  # 0-1, how much sources agree
    bullish_count: int
    bearish_count: int
    neutral_count: int

    # Individual signals
    signals: list[SourceSignal]
    conflicting_signals: list[SourceSignal]

    # Analysis
    primary_driver: SignalSource
    analysis_summary: str

    # Metadata
    aggregation_time_ms: float
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "action": self.action.value,
            "direction": self.direction.value,
            "composite_score": self.composite_score,
            "confidence": self.confidence,
            "actionability": self.actionability,
            "agreement_score": self.agreement_score,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "signals": [s.to_dict() for s in self.signals],
            "conflicting_signals": [s.to_dict() for s in self.conflicting_signals],
            "primary_driver": self.primary_driver.value,
            "analysis_summary": self.analysis_summary,
            "aggregation_time_ms": self.aggregation_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SignalAggregatorConfig:
    """Configuration for signal aggregator."""

    # Source weights (0-1)
    source_weights: dict[SignalSource, float] = field(
        default_factory=lambda: {
            SignalSource.NEWS: 1.0,
            SignalSource.REDDIT: 0.7,
            SignalSource.EARNINGS: 1.2,
            SignalSource.TECHNICAL: 0.9,
            SignalSource.FUNDAMENTALS: 1.0,
            SignalSource.OPTIONS_FLOW: 1.1,
            SignalSource.INSIDER_TRADES: 1.0,
            SignalSource.ANALYST: 0.8,
            SignalSource.CUSTOM: 0.5,
        }
    )

    # Thresholds
    strong_signal_threshold: float = 0.7
    actionable_threshold: float = 0.5
    conflict_threshold: float = 0.3  # Min diff to consider conflicting

    # Time weighting
    recency_decay_hours: float = 24.0  # Signals decay over this period
    max_signal_age_hours: float = 72.0  # Ignore signals older than this

    # Aggregation
    min_signals_for_action: int = 2
    require_majority: bool = True


class SignalAggregator:
    """
    Aggregates trading signals from multiple sources.

    Provides:
    - Source weighting based on historical reliability
    - Conflict resolution when sources disagree
    - Agreement scoring for confidence assessment
    - Actionability rating for trading decisions
    """

    def __init__(
        self,
        config: SignalAggregatorConfig | None = None,
        alert_callback: Callable[[AggregatedSignal], None] | None = None,
    ):
        """
        Initialize signal aggregator.

        Args:
            config: Aggregator configuration
            alert_callback: Callback for high-confidence signals
        """
        self.config = config or SignalAggregatorConfig()
        self.alert_callback = alert_callback

        # Signal storage by ticker
        self._signals: dict[str, list[SourceSignal]] = {}

        # Stats
        self._aggregation_count = 0
        self._total_signals_processed = 0

    def add_signal(self, signal: SourceSignal) -> None:
        """
        Add a new signal.

        Args:
            signal: Signal to add
        """
        ticker = signal.ticker.upper()

        if ticker not in self._signals:
            self._signals[ticker] = []

        self._signals[ticker].append(signal)
        self._total_signals_processed += 1

        # Prune old signals
        self._prune_old_signals(ticker)

    def add_signals(self, signals: list[SourceSignal]) -> None:
        """Add multiple signals."""
        for signal in signals:
            self.add_signal(signal)

    def aggregate(self, ticker: str) -> AggregatedSignal | None:
        """
        Aggregate all signals for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            AggregatedSignal or None if insufficient signals
        """
        start_time = time.time()
        ticker = ticker.upper()

        if ticker not in self._signals:
            return None

        # Get valid (non-expired) signals
        signals = [s for s in self._signals[ticker] if not s.is_expired]

        if len(signals) < self.config.min_signals_for_action:
            return None

        # Apply recency weighting
        weighted_signals = self._apply_recency_weighting(signals)

        # Count directions
        bullish = [s for s in weighted_signals if s.direction == SignalDirection.BULLISH]
        bearish = [s for s in weighted_signals if s.direction == SignalDirection.BEARISH]
        neutral = [s for s in weighted_signals if s.direction == SignalDirection.NEUTRAL]

        # Calculate composite score
        composite_score = self._calculate_composite_score(weighted_signals)

        # Calculate agreement
        agreement_score = self._calculate_agreement(bullish, bearish, neutral)

        # Identify conflicts
        conflicting = self._identify_conflicts(weighted_signals)

        # Determine direction and action
        direction = self._determine_direction(composite_score)
        action = self._determine_action(composite_score, agreement_score, len(bullish), len(bearish), len(conflicting))

        # Calculate confidence and actionability
        confidence = self._calculate_confidence(weighted_signals, agreement_score)
        actionability = self._calculate_actionability(confidence, agreement_score, len(weighted_signals))

        # Find primary driver
        primary_driver = self._find_primary_driver(weighted_signals)

        # Generate summary
        summary = self._generate_summary(action, direction, bullish, bearish, primary_driver)

        aggregation_time_ms = (time.time() - start_time) * 1000
        self._aggregation_count += 1

        result = AggregatedSignal(
            ticker=ticker,
            action=action,
            direction=direction,
            composite_score=composite_score,
            confidence=confidence,
            actionability=actionability,
            agreement_score=agreement_score,
            bullish_count=len(bullish),
            bearish_count=len(bearish),
            neutral_count=len(neutral),
            signals=weighted_signals,
            conflicting_signals=conflicting,
            primary_driver=primary_driver,
            analysis_summary=summary,
            aggregation_time_ms=aggregation_time_ms,
            timestamp=datetime.now(),
        )

        # Trigger callback if actionable
        if self.alert_callback and actionability >= self.config.actionable_threshold:
            self.alert_callback(result)

        return result

    def aggregate_all(self) -> dict[str, AggregatedSignal]:
        """
        Aggregate signals for all tickers.

        Returns:
            Dict mapping ticker to AggregatedSignal
        """
        results = {}

        for ticker in self._signals.keys():
            result = self.aggregate(ticker)
            if result:
                results[ticker] = result

        return results

    def get_top_signals(
        self,
        direction: SignalDirection | None = None,
        min_actionability: float = 0.5,
        limit: int = 10,
    ) -> list[AggregatedSignal]:
        """
        Get top actionable signals.

        Args:
            direction: Filter by direction (or None for all)
            min_actionability: Minimum actionability score
            limit: Maximum results

        Returns:
            List of AggregatedSignals sorted by actionability
        """
        all_signals = self.aggregate_all()

        # Filter
        filtered = [s for s in all_signals.values() if s.actionability >= min_actionability]

        if direction:
            filtered = [s for s in filtered if s.direction == direction]

        # Sort by actionability * abs(composite_score)
        filtered.sort(key=lambda s: s.actionability * abs(s.composite_score), reverse=True)

        return filtered[:limit]

    def _apply_recency_weighting(
        self,
        signals: list[SourceSignal],
    ) -> list[SourceSignal]:
        """Apply time-based decay to signal confidence."""
        now = datetime.now()
        decay_hours = self.config.recency_decay_hours

        for signal in signals:
            age_hours = (now - signal.timestamp).total_seconds() / 3600

            if age_hours > self.config.max_signal_age_hours:
                signal.confidence *= 0.1  # Heavily discount old signals
            else:
                # Exponential decay
                decay_factor = 0.5 ** (age_hours / decay_hours)
                signal.confidence *= decay_factor

        return signals

    def _calculate_composite_score(
        self,
        signals: list[SourceSignal],
    ) -> float:
        """Calculate weighted composite score."""
        if not signals:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for signal in signals:
            source_weight = self.config.source_weights.get(signal.source, 0.5)
            signal_weight = source_weight * signal.confidence * signal.strength

            direction_value = {
                SignalDirection.BULLISH: 1.0,
                SignalDirection.BEARISH: -1.0,
                SignalDirection.NEUTRAL: 0.0,
            }[signal.direction]

            weighted_sum += direction_value * signal_weight
            total_weight += signal_weight

        if total_weight == 0:
            return 0.0

        # Normalize to -1 to +1
        return max(-1.0, min(1.0, weighted_sum / total_weight))

    def _calculate_agreement(
        self,
        bullish: list[SourceSignal],
        bearish: list[SourceSignal],
        neutral: list[SourceSignal],
    ) -> float:
        """Calculate source agreement score."""
        total = len(bullish) + len(bearish) + len(neutral)
        if total == 0:
            return 0.0

        # Find majority direction
        counts = [len(bullish), len(bearish), len(neutral)]
        max_count = max(counts)

        # Agreement = majority / total
        return max_count / total

    def _identify_conflicts(
        self,
        signals: list[SourceSignal],
    ) -> list[SourceSignal]:
        """Identify conflicting high-confidence signals."""
        conflicts = []

        bullish = [s for s in signals if s.direction == SignalDirection.BULLISH and s.confidence > 0.5]
        bearish = [s for s in signals if s.direction == SignalDirection.BEARISH and s.confidence > 0.5]

        # If we have both bullish and bearish high-confidence signals
        if bullish and bearish:
            conflicts.extend(bullish)
            conflicts.extend(bearish)

        return conflicts

    def _determine_direction(self, composite_score: float) -> SignalDirection:
        """Determine signal direction from composite score."""
        if composite_score > 0.1:
            return SignalDirection.BULLISH
        elif composite_score < -0.1:
            return SignalDirection.BEARISH
        return SignalDirection.NEUTRAL

    def _determine_action(
        self,
        composite_score: float,
        agreement_score: float,
        bullish_count: int,
        bearish_count: int,
        conflict_count: int,
    ) -> AggregatedAction:
        """Determine recommended action."""
        # High conflict = conflicting recommendation
        if conflict_count > 2 and agreement_score < 0.5:
            return AggregatedAction.CONFLICTING

        # Strong signals with high agreement
        if composite_score >= self.config.strong_signal_threshold and agreement_score >= 0.7:
            return AggregatedAction.STRONG_BUY
        elif composite_score <= -self.config.strong_signal_threshold and agreement_score >= 0.7:
            return AggregatedAction.STRONG_SELL

        # Moderate signals
        if composite_score >= 0.3:
            return AggregatedAction.BUY
        elif composite_score <= -0.3:
            return AggregatedAction.SELL

        return AggregatedAction.HOLD

    def _calculate_confidence(
        self,
        signals: list[SourceSignal],
        agreement_score: float,
    ) -> float:
        """Calculate overall confidence."""
        if not signals:
            return 0.0

        # Average weighted confidence
        total_weight = 0.0
        weighted_confidence = 0.0

        for signal in signals:
            source_weight = self.config.source_weights.get(signal.source, 0.5)
            weighted_confidence += signal.confidence * source_weight
            total_weight += source_weight

        avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0

        # Adjust by agreement
        confidence = avg_confidence * (0.5 + 0.5 * agreement_score)

        return min(1.0, confidence)

    def _calculate_actionability(
        self,
        confidence: float,
        agreement_score: float,
        signal_count: int,
    ) -> float:
        """Calculate how actionable the aggregated signal is."""
        # Base actionability from confidence
        actionability = confidence

        # Boost for agreement
        actionability *= 0.7 + 0.3 * agreement_score

        # Boost for multiple signals
        signal_factor = min(1.0, signal_count / 5)
        actionability *= 0.6 + 0.4 * signal_factor

        return min(1.0, actionability)

    def _find_primary_driver(
        self,
        signals: list[SourceSignal],
    ) -> SignalSource:
        """Find the primary source driving the signal."""
        if not signals:
            return SignalSource.CUSTOM

        # Find signal with highest weighted score
        best_signal = max(signals, key=lambda s: abs(s.weighted_score) * self.config.source_weights.get(s.source, 0.5))

        return best_signal.source

    def _generate_summary(
        self,
        action: AggregatedAction,
        direction: SignalDirection,
        bullish: list[SourceSignal],
        bearish: list[SourceSignal],
        primary_driver: SignalSource,
    ) -> str:
        """Generate human-readable summary."""
        if action == AggregatedAction.CONFLICTING:
            return f"Conflicting signals: {len(bullish)} bullish vs {len(bearish)} bearish"

        action_text = {
            AggregatedAction.STRONG_BUY: "Strong bullish consensus",
            AggregatedAction.BUY: "Moderately bullish",
            AggregatedAction.HOLD: "No clear direction",
            AggregatedAction.SELL: "Moderately bearish",
            AggregatedAction.STRONG_SELL: "Strong bearish consensus",
        }[action]

        return (
            f"{action_text}. Primary driver: {primary_driver.value}. "
            f"Signals: {len(bullish)} bullish, {len(bearish)} bearish."
        )

    def _prune_old_signals(self, ticker: str) -> None:
        """Remove expired and old signals."""
        if ticker not in self._signals:
            return

        cutoff = datetime.now() - timedelta(hours=self.config.max_signal_age_hours)

        self._signals[ticker] = [s for s in self._signals[ticker] if s.timestamp > cutoff and not s.is_expired]

    def clear_signals(self, ticker: str | None = None) -> None:
        """
        Clear stored signals.

        Args:
            ticker: Clear for specific ticker, or all if None
        """
        if ticker:
            self._signals.pop(ticker.upper(), None)
        else:
            self._signals.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get aggregator statistics."""
        return {
            "total_signals": self._total_signals_processed,
            "active_tickers": len(self._signals),
            "aggregations_performed": self._aggregation_count,
            "signals_by_ticker": {ticker: len(signals) for ticker, signals in self._signals.items()},
        }


def create_signal_aggregator(
    config: SignalAggregatorConfig | None = None,
    alert_callback: Callable[[AggregatedSignal], None] | None = None,
) -> SignalAggregator:
    """
    Factory function to create a signal aggregator.

    Args:
        config: Optional configuration
        alert_callback: Optional callback for high-confidence signals

    Returns:
        Configured SignalAggregator instance
    """
    return SignalAggregator(config, alert_callback)
