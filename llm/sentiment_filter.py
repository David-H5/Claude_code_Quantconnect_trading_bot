"""
Sentiment-based entry filter for trading decisions.

Uses ensemble sentiment analysis to determine if market conditions
are favorable for trade entry. Blocks entries when sentiment is
unfavorable to reduce false positives.

UPGRADE-014: LLM Sentiment Integration (December 2025)
EXPANSION: December 2025 - Regime-Adaptive Weighting, Position Sizing, Soft Voting

Research Sources:
- TradingAgents Multi-Agent Framework (arXiv Dec 2024)
- Sentiment Trading with LLMs (ScienceDirect 2024)
- QuantInsti Sentiment Analysis Trading (2024)
- FinDPO: Financial Sentiment Analysis (arXiv 2025)
- S&P 500 Volatility with Regime-Switching (arXiv Oct 2025)
- LLM Uncertainty in Sentiment Analysis (Frontiers AI 2025)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from .base import SentimentResult


logger = logging.getLogger(__name__)


# ==============================================================================
# Market Regime Detection (UPGRADE-014 Expansion)
# ==============================================================================


class MarketRegime(Enum):
    """Market regime classification based on volatility and trend."""

    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MEAN_REVERTING = "mean_reverting"


@dataclass
class RegimeConfig:
    """
    Configuration for sentiment weighting based on market regime.

    Based on research from arXiv Oct 2025 (Regime-Switching Models)
    and SAGE Journal 2025 (Sentiment Indicators in Trading).
    """

    sentiment_weight: float  # Weight given to sentiment signals
    technical_weight: float  # Weight given to technical signals
    min_confidence: float  # Minimum confidence threshold
    position_mult: float  # Position size multiplier
    description: str = ""


# Default regime configurations based on research
DEFAULT_REGIME_CONFIGS: dict[MarketRegime, RegimeConfig] = {
    MarketRegime.BULL_TRENDING: RegimeConfig(
        sentiment_weight=0.7,
        technical_weight=0.3,
        min_confidence=0.5,
        position_mult=1.2,
        description="Sentiment drives momentum in trends",
    ),
    MarketRegime.BEAR_TRENDING: RegimeConfig(
        sentiment_weight=0.6,
        technical_weight=0.4,
        min_confidence=0.6,
        position_mult=0.8,
        description="Be more cautious, use technicals for timing",
    ),
    MarketRegime.HIGH_VOLATILITY: RegimeConfig(
        sentiment_weight=0.4,
        technical_weight=0.6,
        min_confidence=0.75,
        position_mult=0.5,
        description="Sentiment less reliable in chaos",
    ),
    MarketRegime.LOW_VOLATILITY: RegimeConfig(
        sentiment_weight=0.5,
        technical_weight=0.5,
        min_confidence=0.55,
        position_mult=1.0,
        description="Standard operation in calm markets",
    ),
    MarketRegime.MEAN_REVERTING: RegimeConfig(
        sentiment_weight=0.3,
        technical_weight=0.7,
        min_confidence=0.65,
        position_mult=0.9,
        description="Technical levels more important",
    ),
}


@dataclass
class RegimeState:
    """Current market regime state with metadata."""

    regime: MarketRegime
    config: RegimeConfig
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    volatility_percentile: float = 50.0  # 0-100
    trend_strength: float = 0.0  # -1 to 1
    confidence: float = 0.5  # 0-1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime": self.regime.value,
            "sentiment_weight": self.config.sentiment_weight,
            "technical_weight": self.config.technical_weight,
            "min_confidence": self.config.min_confidence,
            "position_mult": self.config.position_mult,
            "volatility_percentile": self.volatility_percentile,
            "trend_strength": self.trend_strength,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
        }


class RegimeDetector:
    """
    Detect market regime from volatility and trend data.

    Based on research:
    - arXiv Oct 2025: Dual-memory HAR model for regime detection
    - arXiv Sep 2025: Adaptive alpha weighting based on regime
    """

    def __init__(
        self,
        volatility_window: int = 20,
        trend_window: int = 50,
        high_vol_threshold: float = 75.0,  # percentile
        low_vol_threshold: float = 25.0,  # percentile
        trend_threshold: float = 0.3,  # abs value for trending
        regime_configs: dict[MarketRegime, RegimeConfig] | None = None,
    ):
        """
        Initialize regime detector.

        Args:
            volatility_window: Days for volatility calculation
            trend_window: Days for trend calculation
            high_vol_threshold: Percentile for high volatility
            low_vol_threshold: Percentile for low volatility
            trend_threshold: Threshold for trending market detection
            regime_configs: Custom regime configurations
        """
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.trend_threshold = trend_threshold
        self.regime_configs = regime_configs or DEFAULT_REGIME_CONFIGS

        self._current_regime: RegimeState | None = None
        self._volatility_history: list[float] = []
        self._returns_history: list[float] = []

    def update(
        self,
        volatility: float,
        daily_return: float,
    ) -> RegimeState:
        """
        Update regime detection with new data.

        Args:
            volatility: Current volatility (e.g., 20-day realized vol)
            daily_return: Daily return for trend calculation

        Returns:
            Current regime state
        """
        # Update histories
        self._volatility_history.append(volatility)
        self._returns_history.append(daily_return)

        # Keep limited history
        if len(self._volatility_history) > 252:  # 1 year of trading days
            self._volatility_history = self._volatility_history[-252:]
        if len(self._returns_history) > 252:
            self._returns_history = self._returns_history[-252:]

        # Calculate volatility percentile
        vol_percentile = self._calculate_percentile(volatility, self._volatility_history)

        # Calculate trend strength
        trend_strength = self._calculate_trend_strength()

        # Detect regime
        regime = self._classify_regime(vol_percentile, trend_strength)
        config = self.regime_configs[regime]

        # Calculate detection confidence
        confidence = self._calculate_confidence(vol_percentile, trend_strength, regime)

        self._current_regime = RegimeState(
            regime=regime,
            config=config,
            volatility_percentile=vol_percentile,
            trend_strength=trend_strength,
            confidence=confidence,
        )

        return self._current_regime

    def _calculate_percentile(self, value: float, history: list[float]) -> float:
        """Calculate percentile of value in history."""
        if not history:
            return 50.0

        sorted_history = sorted(history)
        count_below = sum(1 for v in sorted_history if v < value)
        return (count_below / len(sorted_history)) * 100

    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength from returns (-1 to 1)."""
        if len(self._returns_history) < self.trend_window:
            return 0.0

        recent = self._returns_history[-self.trend_window :]
        cumulative = sum(recent)

        # Normalize to -1 to 1 range
        # Assuming ±50% cumulative return is extreme
        return max(-1.0, min(1.0, cumulative / 0.5))

    def _classify_regime(self, vol_percentile: float, trend_strength: float) -> MarketRegime:
        """Classify market regime based on volatility and trend."""
        # High volatility takes precedence
        if vol_percentile >= self.high_vol_threshold:
            return MarketRegime.HIGH_VOLATILITY

        # Low volatility
        if vol_percentile <= self.low_vol_threshold:
            return MarketRegime.LOW_VOLATILITY

        # Check for trending
        if trend_strength >= self.trend_threshold:
            return MarketRegime.BULL_TRENDING
        elif trend_strength <= -self.trend_threshold:
            return MarketRegime.BEAR_TRENDING

        # Default to mean reverting
        return MarketRegime.MEAN_REVERTING

    def _calculate_confidence(self, vol_percentile: float, trend_strength: float, regime: MarketRegime) -> float:
        """Calculate confidence in regime classification."""
        # Base confidence from extremity of signals
        vol_extremity = abs(vol_percentile - 50) / 50  # 0-1
        trend_extremity = abs(trend_strength)  # 0-1

        if regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.LOW_VOLATILITY]:
            # Volatility regimes: higher confidence when vol is more extreme
            return 0.5 + (vol_extremity * 0.5)
        elif regime in [MarketRegime.BULL_TRENDING, MarketRegime.BEAR_TRENDING]:
            # Trending regimes: higher confidence when trend is stronger
            return 0.5 + (trend_extremity * 0.5)
        else:
            # Mean reverting: moderate confidence
            return 0.5 + ((1 - vol_extremity) * (1 - trend_extremity) * 0.3)

    def get_current_regime(self) -> RegimeState | None:
        """Get current regime state."""
        return self._current_regime

    def get_sentiment_weight(self) -> float:
        """Get current sentiment weight based on regime."""
        if self._current_regime:
            return self._current_regime.config.sentiment_weight
        return 0.5  # Default balanced weight

    def get_position_multiplier(self) -> float:
        """Get position size multiplier based on regime."""
        if self._current_regime:
            return self._current_regime.config.position_mult
        return 1.0  # Default no adjustment


# ==============================================================================
# Confidence-Based Position Sizing (UPGRADE-014 Expansion)
# ==============================================================================


@dataclass
class PositionSizeResult:
    """Result of confidence-based position sizing calculation."""

    base_size: float  # Original position size
    adjusted_size: float  # After confidence adjustment
    confidence_mult: float  # Multiplier from confidence
    agreement_mult: float  # Multiplier from ensemble agreement
    regime_mult: float  # Multiplier from market regime
    final_mult: float  # Combined multiplier
    capped: bool  # Whether size was capped

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_size": self.base_size,
            "adjusted_size": self.adjusted_size,
            "confidence_mult": self.confidence_mult,
            "agreement_mult": self.agreement_mult,
            "regime_mult": self.regime_mult,
            "final_mult": self.final_mult,
            "capped": self.capped,
        }


def calculate_confidence_position_size(
    base_size: float,
    sentiment_confidence: float,
    ensemble_agreement: float = 1.0,
    regime_mult: float = 1.0,
    min_mult: float = 0.5,
    max_mult: float = 2.0,
) -> PositionSizeResult:
    """
    Calculate position size adjusted for sentiment confidence.

    Based on FinDPO research (arXiv 2025) - logit-to-score conversion
    for continuous ranking and position sizing.

    Args:
        base_size: Base position size (e.g., 0.02 = 2% of portfolio)
        sentiment_confidence: Model confidence [0, 1]
        ensemble_agreement: Agreement between models [0, 1]
        regime_mult: Multiplier from market regime
        min_mult: Minimum total multiplier
        max_mult: Maximum total multiplier

    Returns:
        PositionSizeResult with adjusted size and metadata
    """
    # Confidence multiplier: maps [0, 1] -> [0.5, 1.5]
    confidence_mult = 0.5 + sentiment_confidence

    # Agreement multiplier: maps [0, 1] -> [0.5, 1.5]
    agreement_mult = 0.5 + ensemble_agreement

    # Combined multiplier
    raw_mult = confidence_mult * agreement_mult * regime_mult

    # Cap the multiplier
    final_mult = max(min_mult, min(max_mult, raw_mult))
    capped = final_mult != raw_mult

    # Calculate adjusted size
    adjusted_size = base_size * final_mult

    return PositionSizeResult(
        base_size=base_size,
        adjusted_size=adjusted_size,
        confidence_mult=confidence_mult,
        agreement_mult=agreement_mult,
        regime_mult=regime_mult,
        final_mult=final_mult,
        capped=capped,
    )


def logit_to_score(logit: float) -> float:
    """
    Convert logit to continuous score [0, 1].

    Based on FinDPO (arXiv 2025) - enables continuous ranking
    instead of discrete sentiment labels.

    Args:
        logit: Raw logit from model

    Returns:
        Score in [0, 1] range
    """
    return 1.0 / (1.0 + math.exp(-logit))


# ==============================================================================
# Soft Voting Ensemble (UPGRADE-014 Expansion)
# ==============================================================================


@dataclass
class VotingResult:
    """Result of ensemble voting."""

    final_score: float  # Final sentiment score
    final_confidence: float  # Combined confidence
    voting_method: str  # Method used
    model_scores: dict[str, float]  # Individual model scores
    model_weights: dict[str, float]  # Weights used

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "final_score": self.final_score,
            "final_confidence": self.final_confidence,
            "voting_method": self.voting_method,
            "model_scores": self.model_scores,
            "model_weights": self.model_weights,
        }


def soft_vote_ensemble(
    predictions: list[tuple[str, float, float]],  # (model_name, score, confidence)
    weights: dict[str, float] | None = None,
    use_confidence_weighting: bool = True,
) -> VotingResult:
    """
    Soft voting ensemble that averages probabilities.

    Based on research from Frontiers AI 2025 - soft voting preserves
    confidence information better than hard voting.

    Args:
        predictions: List of (model_name, sentiment_score, confidence) tuples
        weights: Optional model weights (default: equal)
        use_confidence_weighting: Whether to weight by confidence

    Returns:
        VotingResult with combined prediction
    """
    if not predictions:
        return VotingResult(
            final_score=0.0,
            final_confidence=0.0,
            voting_method="soft_vote",
            model_scores={},
            model_weights={},
        )

    model_scores = {name: score for name, score, _ in predictions}
    model_confidences = {name: conf for name, _, conf in predictions}

    # Determine weights
    if weights is None:
        weights = {name: 1.0 for name, _, _ in predictions}

    # Normalize weights
    total_weight = sum(weights.values())
    norm_weights = {k: v / total_weight for k, v in weights.items()}

    # Calculate weighted scores
    if use_confidence_weighting:
        # Weight by both model weight and confidence
        weighted_sum = 0.0
        weight_total = 0.0
        for name, score, conf in predictions:
            w = norm_weights.get(name, 0.0) * conf
            weighted_sum += score * w
            weight_total += w

        final_score = weighted_sum / weight_total if weight_total > 0 else 0.0
    else:
        # Weight by model weight only
        final_score = sum(score * norm_weights.get(name, 0.0) for name, score, _ in predictions)

    # Calculate combined confidence (average)
    final_confidence = sum(conf for _, _, conf in predictions) / len(predictions)

    return VotingResult(
        final_score=final_score,
        final_confidence=final_confidence,
        voting_method="soft_vote_confidence" if use_confidence_weighting else "soft_vote",
        model_scores=model_scores,
        model_weights=norm_weights,
    )


def hard_vote_ensemble(
    predictions: list[tuple[str, float, float]],  # (model_name, score, confidence)
) -> VotingResult:
    """
    Hard voting ensemble that uses majority label.

    Simpler but loses confidence information.

    Args:
        predictions: List of (model_name, sentiment_score, confidence) tuples

    Returns:
        VotingResult with majority prediction
    """
    if not predictions:
        return VotingResult(
            final_score=0.0,
            final_confidence=0.0,
            voting_method="hard_vote",
            model_scores={},
            model_weights={},
        )

    model_scores = {name: score for name, score, _ in predictions}

    # Count votes for positive/negative/neutral
    bullish = sum(1 for _, score, _ in predictions if score > 0.1)
    bearish = sum(1 for _, score, _ in predictions if score < -0.1)
    neutral = len(predictions) - bullish - bearish

    # Majority vote
    if bullish > bearish and bullish > neutral:
        final_score = sum(s for _, s, _ in predictions if s > 0.1) / bullish
    elif bearish > bullish and bearish > neutral:
        final_score = sum(s for _, s, _ in predictions if s < -0.1) / bearish
    else:
        final_score = 0.0

    # Confidence is proportion of agreement
    max_votes = max(bullish, bearish, neutral)
    final_confidence = max_votes / len(predictions)

    return VotingResult(
        final_score=final_score,
        final_confidence=final_confidence,
        voting_method="hard_vote",
        model_scores=model_scores,
        model_weights={name: 1.0 / len(predictions) for name, _, _ in predictions},
    )


def weighted_soft_vote_ensemble(
    predictions: list[tuple[str, float, float]],  # (model_name, score, confidence)
    performance_weights: dict[str, float],  # Historical performance weights
) -> VotingResult:
    """
    Weighted soft voting based on historical model performance.

    Models with better historical accuracy get higher weights.

    Args:
        predictions: List of (model_name, sentiment_score, confidence) tuples
        performance_weights: Weights based on historical performance

    Returns:
        VotingResult with performance-weighted prediction
    """
    return soft_vote_ensemble(
        predictions,
        weights=performance_weights,
        use_confidence_weighting=True,
    )


# ==============================================================================
# Sentiment Momentum & Mean Reversion (UPGRADE-014 Expansion - Feature 7)
# ==============================================================================


class SentimentExtreme(Enum):
    """Classification of sentiment extreme conditions."""

    EXTREME_BULLISH = "extreme_bullish"
    EXTREME_BEARISH = "extreme_bearish"
    NORMAL = "normal"


@dataclass
class MomentumSignal:
    """Sentiment momentum analysis result."""

    symbol: str
    current_score: float  # Current sentiment score
    momentum: float  # Rate of change (-1 to 1)
    acceleration: float  # Second derivative of sentiment
    extreme: SentimentExtreme  # Current extreme classification
    mean_reversion_signal: bool  # True if mean reversion expected
    divergence_from_price: float | None  # Sentiment vs price divergence
    lookback_periods: int  # Periods used for calculation
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "current_score": self.current_score,
            "momentum": self.momentum,
            "acceleration": self.acceleration,
            "extreme": self.extreme.value,
            "mean_reversion_signal": self.mean_reversion_signal,
            "divergence_from_price": self.divergence_from_price,
            "lookback_periods": self.lookback_periods,
            "timestamp": self.timestamp.isoformat(),
        }


class SentimentMomentumTracker:
    """
    Track sentiment momentum and detect mean reversion opportunities.

    UPGRADE-014 Expansion - Feature 7 (December 2025)

    Based on research:
    - SAGE Journal 2025: Sentiment indicators in trading
    - QuantConnect Forum: Sentiment mean strategy
    - QuantInsti 2024: Sentiment momentum patterns

    Features:
    1. Momentum (rate of change) tracking
    2. Extreme sentiment detection for mean reversion
    3. Sentiment-price divergence signals
    4. Acceleration tracking (momentum of momentum)
    """

    def __init__(
        self,
        lookback_periods: int = 20,
        extreme_threshold: float = 0.7,
        momentum_smoothing: float = 0.3,  # EMA alpha for smoothing
        mean_reversion_threshold: float = 0.8,
    ):
        """
        Initialize momentum tracker.

        Args:
            lookback_periods: Number of periods for momentum calculation
            extreme_threshold: Threshold for extreme sentiment (0-1)
            momentum_smoothing: EMA smoothing factor for momentum
            mean_reversion_threshold: Threshold for mean reversion signal
        """
        self.lookback_periods = lookback_periods
        self.extreme_threshold = extreme_threshold
        self.momentum_smoothing = momentum_smoothing
        self.mean_reversion_threshold = mean_reversion_threshold

        # History per symbol
        self._sentiment_history: dict[str, list[tuple[datetime, float]]] = {}
        self._momentum_history: dict[str, list[float]] = {}
        self._price_history: dict[str, list[tuple[datetime, float]]] = {}

    def update_sentiment(
        self,
        symbol: str,
        sentiment_score: float,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Update sentiment history for a symbol.

        Args:
            symbol: Ticker symbol
            sentiment_score: Current sentiment score (-1 to 1)
            timestamp: Observation timestamp
        """
        ts = timestamp or datetime.now(timezone.utc)

        if symbol not in self._sentiment_history:
            self._sentiment_history[symbol] = []

        self._sentiment_history[symbol].append((ts, sentiment_score))

        # Keep limited history (2x lookback for calculations)
        max_history = self.lookback_periods * 2
        if len(self._sentiment_history[symbol]) > max_history:
            self._sentiment_history[symbol] = self._sentiment_history[symbol][-max_history:]

    def update_price(
        self,
        symbol: str,
        price: float,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Update price history for divergence calculations.

        Args:
            symbol: Ticker symbol
            price: Current price
            timestamp: Observation timestamp
        """
        ts = timestamp or datetime.now(timezone.utc)

        if symbol not in self._price_history:
            self._price_history[symbol] = []

        self._price_history[symbol].append((ts, price))

        # Keep limited history
        max_history = self.lookback_periods * 2
        if len(self._price_history[symbol]) > max_history:
            self._price_history[symbol] = self._price_history[symbol][-max_history:]

    def get_momentum_signal(self, symbol: str) -> MomentumSignal | None:
        """
        Calculate momentum signal for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            MomentumSignal if enough data, None otherwise
        """
        history = self._sentiment_history.get(symbol, [])

        if len(history) < 3:
            return None  # Need at least 3 points for momentum

        # Get scores only
        scores = [s for _, s in history]
        current_score = scores[-1]

        # Calculate momentum (rate of change)
        momentum = self._calculate_momentum(scores)

        # Calculate acceleration (momentum of momentum)
        acceleration = self._calculate_acceleration(symbol, momentum)

        # Detect extreme condition
        extreme = self._classify_extreme(current_score)

        # Check for mean reversion signal
        mean_reversion = self._check_mean_reversion(current_score, momentum, extreme)

        # Calculate price divergence if price data available
        divergence = self._calculate_divergence(symbol, scores)

        return MomentumSignal(
            symbol=symbol,
            current_score=current_score,
            momentum=momentum,
            acceleration=acceleration,
            extreme=extreme,
            mean_reversion_signal=mean_reversion,
            divergence_from_price=divergence,
            lookback_periods=min(len(scores), self.lookback_periods),
        )

    def _calculate_momentum(self, scores: list[float]) -> float:
        """Calculate sentiment momentum using EMA-smoothed rate of change."""
        if len(scores) < 2:
            return 0.0

        # Calculate simple rate of change over lookback
        lookback = min(len(scores), self.lookback_periods)
        if lookback < 2:
            return 0.0

        recent = scores[-lookback:]

        # Calculate period-over-period changes
        changes = [recent[i] - recent[i - 1] for i in range(1, len(recent))]

        # EMA-smooth the momentum
        if not changes:
            return 0.0

        ema = changes[0]
        for change in changes[1:]:
            ema = self.momentum_smoothing * change + (1 - self.momentum_smoothing) * ema

        # Normalize to -1 to 1 range (assuming max change of 0.5 per period)
        return max(-1.0, min(1.0, ema / 0.5))

    def _calculate_acceleration(self, symbol: str, current_momentum: float) -> float:
        """Calculate momentum acceleration (rate of change of momentum)."""
        if symbol not in self._momentum_history:
            self._momentum_history[symbol] = []

        self._momentum_history[symbol].append(current_momentum)

        # Keep limited history
        if len(self._momentum_history[symbol]) > self.lookback_periods:
            self._momentum_history[symbol] = self._momentum_history[symbol][-self.lookback_periods :]

        if len(self._momentum_history[symbol]) < 2:
            return 0.0

        # Simple acceleration as difference in momentum
        recent_momentum = self._momentum_history[symbol][-2:]
        return recent_momentum[1] - recent_momentum[0]

    def _classify_extreme(self, score: float) -> SentimentExtreme:
        """Classify if sentiment is at an extreme level."""
        if score >= self.extreme_threshold:
            return SentimentExtreme.EXTREME_BULLISH
        elif score <= -self.extreme_threshold:
            return SentimentExtreme.EXTREME_BEARISH
        else:
            return SentimentExtreme.NORMAL

    def _check_mean_reversion(
        self,
        score: float,
        momentum: float,
        extreme: SentimentExtreme,
    ) -> bool:
        """
        Check if mean reversion is likely.

        Mean reversion signal when:
        1. Sentiment is at extreme
        2. Momentum is slowing or reversing
        """
        if extreme == SentimentExtreme.NORMAL:
            return False

        # Extreme bullish with slowing/reversing momentum
        if extreme == SentimentExtreme.EXTREME_BULLISH:
            return momentum < 0 or abs(score) >= self.mean_reversion_threshold

        # Extreme bearish with slowing/reversing momentum
        if extreme == SentimentExtreme.EXTREME_BEARISH:
            return momentum > 0 or abs(score) >= self.mean_reversion_threshold

        return False

    def _calculate_divergence(
        self,
        symbol: str,
        sentiment_scores: list[float],
    ) -> float | None:
        """
        Calculate divergence between sentiment and price movements.

        Positive divergence: Sentiment rising while price falling (bullish signal)
        Negative divergence: Sentiment falling while price rising (bearish signal)
        """
        price_history = self._price_history.get(symbol, [])

        if len(price_history) < 2 or len(sentiment_scores) < 2:
            return None

        # Calculate price change (normalized)
        prices = [p for _, p in price_history[-len(sentiment_scores) :]]
        if len(prices) < 2 or prices[0] == 0:
            return None

        price_change = (prices[-1] - prices[0]) / prices[0]  # Percentage change

        # Calculate sentiment change
        sentiment_change = sentiment_scores[-1] - sentiment_scores[0]

        # Divergence: sentiment direction vs price direction
        # Positive = sentiment bullish while price bearish (potential reversal up)
        # Negative = sentiment bearish while price bullish (potential reversal down)

        # Normalize price change to sentiment scale (-1 to 1)
        # Assuming 10% price move is significant
        normalized_price = max(-1.0, min(1.0, price_change / 0.10))

        divergence = sentiment_change - normalized_price

        return divergence

    def get_all_momentum_signals(self) -> dict[str, MomentumSignal]:
        """Get momentum signals for all tracked symbols."""
        signals = {}
        for symbol in self._sentiment_history.keys():
            signal = self.get_momentum_signal(symbol)
            if signal:
                signals[symbol] = signal
        return signals

    def get_extreme_symbols(self) -> dict[str, SentimentExtreme]:
        """Get all symbols currently at sentiment extremes."""
        extremes = {}
        for symbol in self._sentiment_history.keys():
            signal = self.get_momentum_signal(symbol)
            if signal and signal.extreme != SentimentExtreme.NORMAL:
                extremes[symbol] = signal.extreme
        return extremes

    def get_mean_reversion_candidates(self) -> list[str]:
        """Get symbols with active mean reversion signals."""
        candidates = []
        for symbol in self._sentiment_history.keys():
            signal = self.get_momentum_signal(symbol)
            if signal and signal.mean_reversion_signal:
                candidates.append(symbol)
        return candidates

    def get_divergence_signals(
        self,
        min_divergence: float = 0.3,
    ) -> dict[str, float]:
        """
        Get symbols with significant sentiment-price divergence.

        Args:
            min_divergence: Minimum absolute divergence to report

        Returns:
            Dict of symbol -> divergence score
        """
        divergences = {}
        for symbol in self._sentiment_history.keys():
            signal = self.get_momentum_signal(symbol)
            if signal and signal.divergence_from_price is not None:
                if abs(signal.divergence_from_price) >= min_divergence:
                    divergences[symbol] = signal.divergence_from_price
        return divergences

    def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics."""
        return {
            "tracked_symbols": len(self._sentiment_history),
            "total_observations": sum(len(h) for h in self._sentiment_history.values()),
            "extreme_count": len(self.get_extreme_symbols()),
            "mean_reversion_count": len(self.get_mean_reversion_candidates()),
            "lookback_periods": self.lookback_periods,
            "extreme_threshold": self.extreme_threshold,
        }


def create_momentum_tracker(
    lookback_periods: int = 20,
    extreme_threshold: float = 0.7,
    momentum_smoothing: float = 0.3,
    mean_reversion_threshold: float = 0.8,
) -> SentimentMomentumTracker:
    """
    Factory function to create a sentiment momentum tracker.

    Args:
        lookback_periods: Periods for momentum calculation
        extreme_threshold: Threshold for extreme sentiment detection
        momentum_smoothing: EMA alpha for momentum smoothing
        mean_reversion_threshold: Threshold for mean reversion signal

    Returns:
        Configured SentimentMomentumTracker
    """
    return SentimentMomentumTracker(
        lookback_periods=lookback_periods,
        extreme_threshold=extreme_threshold,
        momentum_smoothing=momentum_smoothing,
        mean_reversion_threshold=mean_reversion_threshold,
    )


# ==============================================================================
# Data Types
# ==============================================================================


class FilterDecision(Enum):
    """Decision from sentiment filter."""

    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_REVIEW = "require_review"


class FilterReason(Enum):
    """Reason for filter decision."""

    SENTIMENT_FAVORABLE = "sentiment_favorable"
    SENTIMENT_UNFAVORABLE = "sentiment_unfavorable"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    TREND_MISMATCH = "trend_mismatch"
    NO_DATA = "no_data"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class SentimentSignal:
    """Sentiment analysis result for filtering."""

    symbol: str
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    source: str  # "finbert", "ensemble", etc.
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    articles_analyzed: int = 0
    raw_results: list[SentimentResult] | None = None

    @property
    def is_bullish(self) -> bool:
        """Check if sentiment is bullish."""
        return self.sentiment_score > 0.1

    @property
    def is_bearish(self) -> bool:
        """Check if sentiment is bearish."""
        return self.sentiment_score < -0.1

    @property
    def is_neutral(self) -> bool:
        """Check if sentiment is neutral."""
        return -0.1 <= self.sentiment_score <= 0.1

    @property
    def strength(self) -> str:
        """Get sentiment strength label."""
        abs_score = abs(self.sentiment_score)
        if abs_score >= 0.7:
            return "strong"
        elif abs_score >= 0.4:
            return "moderate"
        elif abs_score >= 0.1:
            return "weak"
        else:
            return "neutral"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "articles_analyzed": self.articles_analyzed,
            "is_bullish": self.is_bullish,
            "is_bearish": self.is_bearish,
            "strength": self.strength,
        }


@dataclass
class FilterResult:
    """Result of sentiment filter check."""

    decision: FilterDecision
    reason: FilterReason
    message: str
    signal: SentimentSignal | None = None
    trend_data: list[float] | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_allowed(self) -> bool:
        """Check if trade is allowed."""
        return self.decision == FilterDecision.ALLOW

    @property
    def is_blocked(self) -> bool:
        """Check if trade is blocked."""
        return self.decision == FilterDecision.BLOCK

    @property
    def needs_review(self) -> bool:
        """Check if trade needs review."""
        return self.decision == FilterDecision.REQUIRE_REVIEW

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision": self.decision.value,
            "reason": self.reason.value,
            "message": self.message,
            "signal": self.signal.to_dict() if self.signal else None,
            "timestamp": self.timestamp.isoformat(),
        }


# ==============================================================================
# Sentiment Filter
# ==============================================================================


class SentimentFilter:
    """
    Filter trade entries based on sentiment analysis.

    Uses configurable thresholds to allow, block, or flag
    trades based on current sentiment conditions.

    Features:
    - Configurable thresholds for long/short entries
    - Minimum confidence requirements
    - Trend analysis for confirmation
    - Signal history tracking
    - Override capability for manual trades

    Example:
        >>> filter = SentimentFilter(
        ...     min_sentiment_for_long=0.1,
        ...     max_sentiment_for_short=-0.1,
        ...     min_confidence=0.6,
        ... )
        >>> signal = SentimentSignal("AAPL", 0.35, 0.8, "finbert")
        >>> result = filter.check_entry("AAPL", "long", signal)
        >>> if result.is_allowed:
        ...     execute_trade()
    """

    def __init__(
        self,
        min_sentiment_for_long: float = 0.0,
        max_sentiment_for_short: float = 0.0,
        min_confidence: float = 0.5,
        lookback_hours: int = 24,
        require_positive_trend: bool = False,
        trend_window_size: int = 3,
        max_history_per_symbol: int = 100,
    ):
        """
        Initialize sentiment filter.

        Args:
            min_sentiment_for_long: Minimum sentiment score for long entries
            max_sentiment_for_short: Maximum sentiment score for short entries
            min_confidence: Minimum confidence level required
            lookback_hours: Hours of history to consider for trends
            require_positive_trend: Whether to require trend confirmation
            trend_window_size: Number of signals to use for trend analysis
            max_history_per_symbol: Maximum signals to keep per symbol
        """
        self.min_sentiment_for_long = min_sentiment_for_long
        self.max_sentiment_for_short = max_sentiment_for_short
        self.min_confidence = min_confidence
        self.lookback_hours = lookback_hours
        self.require_positive_trend = require_positive_trend
        self.trend_window_size = trend_window_size
        self.max_history_per_symbol = max_history_per_symbol

        self._signal_history: dict[str, list[SentimentSignal]] = {}
        self._override_symbols: dict[str, datetime] = {}
        self._filter_stats: dict[str, int] = {
            "total_checks": 0,
            "allowed": 0,
            "blocked": 0,
            "review_required": 0,
        }
        self._listeners: list[Callable[[FilterResult], None]] = []

    def check_entry(
        self,
        symbol: str,
        direction: str,
        current_signal: SentimentSignal | None = None,
    ) -> FilterResult:
        """
        Check if trade entry should be allowed.

        Args:
            symbol: Trading symbol
            direction: "long" or "short"
            current_signal: Current sentiment analysis (optional, uses latest from history)

        Returns:
            FilterResult indicating allow/block/review
        """
        self._filter_stats["total_checks"] += 1

        # If no current signal provided, use latest from history
        if current_signal is None:
            history = self._signal_history.get(symbol, [])
            if history:
                current_signal = history[-1]
            else:
                # No data for this symbol
                return FilterResult(
                    decision=FilterDecision.REQUIRE_REVIEW,
                    reason=FilterReason.NO_DATA,
                    message=f"No sentiment data available for {symbol}",
                    signal=None,
                )

        # Check for manual override
        if self._is_override_active(symbol):
            result = FilterResult(
                decision=FilterDecision.ALLOW,
                reason=FilterReason.MANUAL_OVERRIDE,
                message=f"Manual override active for {symbol}",
                signal=current_signal,
            )
            self._record_and_notify(result)
            return result

        # Store signal in history (only if new signal provided)
        if current_signal and current_signal not in self._signal_history.get(symbol, []):
            self._add_signal(symbol, current_signal)

        # Check confidence threshold
        if current_signal.confidence < self.min_confidence:
            result = FilterResult(
                decision=FilterDecision.REQUIRE_REVIEW,
                reason=FilterReason.CONFIDENCE_TOO_LOW,
                message=(f"Confidence {current_signal.confidence:.2f} below " f"minimum {self.min_confidence:.2f}"),
                signal=current_signal,
            )
            self._filter_stats["review_required"] += 1
            self._record_and_notify(result)
            return result

        # Check sentiment alignment with direction
        if direction == "long":
            if current_signal.sentiment_score < self.min_sentiment_for_long:
                result = FilterResult(
                    decision=FilterDecision.BLOCK,
                    reason=FilterReason.SENTIMENT_UNFAVORABLE,
                    message=(
                        f"Long blocked: sentiment {current_signal.sentiment_score:.2f} "
                        f"below threshold {self.min_sentiment_for_long:.2f}"
                    ),
                    signal=current_signal,
                )
                self._filter_stats["blocked"] += 1
                self._record_and_notify(result)
                return result

        elif direction == "short":
            if current_signal.sentiment_score > self.max_sentiment_for_short:
                result = FilterResult(
                    decision=FilterDecision.BLOCK,
                    reason=FilterReason.SENTIMENT_UNFAVORABLE,
                    message=(
                        f"Short blocked: sentiment {current_signal.sentiment_score:.2f} "
                        f"above threshold {self.max_sentiment_for_short:.2f}"
                    ),
                    signal=current_signal,
                )
                self._filter_stats["blocked"] += 1
                self._record_and_notify(result)
                return result

        # Check trend if required
        if self.require_positive_trend:
            trend_result = self._check_trend(symbol, direction)
            if not trend_result["supports_direction"]:
                result = FilterResult(
                    decision=FilterDecision.REQUIRE_REVIEW,
                    reason=FilterReason.TREND_MISMATCH,
                    message=f"Trend does not support {direction}: {trend_result['reason']}",
                    signal=current_signal,
                    trend_data=trend_result.get("scores"),
                )
                self._filter_stats["review_required"] += 1
                self._record_and_notify(result)
                return result

        # All checks passed
        result = FilterResult(
            decision=FilterDecision.ALLOW,
            reason=FilterReason.SENTIMENT_FAVORABLE,
            message=f"{direction.capitalize()} allowed: sentiment {current_signal.sentiment_score:.2f}",
            signal=current_signal,
        )
        self._filter_stats["allowed"] += 1
        self._record_and_notify(result)
        return result

    def _check_trend(self, symbol: str, direction: str) -> dict[str, Any]:
        """Check if sentiment trend supports direction."""
        history = self._signal_history.get(symbol, [])

        if len(history) < 2:
            return {
                "supports_direction": True,
                "reason": "Insufficient history",
                "scores": [],
            }

        recent = history[-self.trend_window_size :]
        scores = [s.sentiment_score for s in recent]

        if direction == "long":
            # For long, want positive or improving sentiment
            all_positive = all(s > 0 for s in scores)
            improving = len(scores) >= 2 and scores[-1] > scores[0]

            return {
                "supports_direction": all_positive or improving,
                "reason": "Positive trend" if all_positive or improving else "Negative trend",
                "scores": scores,
            }
        else:
            # For short, want negative or declining sentiment
            all_negative = all(s < 0 for s in scores)
            declining = len(scores) >= 2 and scores[-1] < scores[0]

            return {
                "supports_direction": all_negative or declining,
                "reason": "Negative trend" if all_negative or declining else "Positive trend",
                "scores": scores,
            }

    def add_signal(self, signal: SentimentSignal) -> None:
        """
        Add a sentiment signal to history.

        Public method for adding signals from external sources like
        the LLM ensemble in hybrid_options_bot.py.

        Args:
            signal: SentimentSignal to add
        """
        self._add_signal(signal.symbol, signal)

    def _add_signal(self, symbol: str, signal: SentimentSignal) -> None:
        """Add signal to history (internal)."""
        if symbol not in self._signal_history:
            self._signal_history[symbol] = []

        self._signal_history[symbol].append(signal)

        # Trim history if needed
        if len(self._signal_history[symbol]) > self.max_history_per_symbol:
            self._signal_history[symbol] = self._signal_history[symbol][-self.max_history_per_symbol :]

    def _is_override_active(self, symbol: str) -> bool:
        """Check if override is active for symbol."""
        if symbol not in self._override_symbols:
            return False

        expiry = self._override_symbols[symbol]
        if datetime.now(timezone.utc) > expiry:
            del self._override_symbols[symbol]
            return False

        return True

    def _record_and_notify(self, result: FilterResult) -> None:
        """Record result and notify listeners."""
        for listener in self._listeners:
            try:
                listener(result)
            except Exception as e:
                logger.error(f"Error notifying filter listener: {e}")

    # ==========================================================================
    # Override Management
    # ==========================================================================

    def set_override(self, symbol: str, duration_minutes: int = 60) -> None:
        """
        Set manual override for a symbol.

        Args:
            symbol: Symbol to override
            duration_minutes: Override duration in minutes
        """
        expiry = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        self._override_symbols[symbol] = expiry
        logger.info(f"Override set for {symbol} until {expiry}")

    def clear_override(self, symbol: str) -> None:
        """Clear manual override for a symbol."""
        if symbol in self._override_symbols:
            del self._override_symbols[symbol]
            logger.info(f"Override cleared for {symbol}")

    def clear_all_overrides(self) -> None:
        """Clear all manual overrides."""
        self._override_symbols.clear()
        logger.info("All overrides cleared")

    # ==========================================================================
    # History and Stats
    # ==========================================================================

    def get_signal_history(self, symbol: str, limit: int = 10) -> list[SentimentSignal]:
        """Get recent signal history for a symbol."""
        history = self._signal_history.get(symbol, [])
        return history[-limit:]

    def get_weighted_sentiment(
        self,
        symbol: str,
        decay_rate: float = 0.9,
        max_age_hours: int = 24,
    ) -> dict[str, Any] | None:
        """
        Get time-weighted sentiment with exponential decay.

        More recent signals have higher weight than older ones.

        Args:
            symbol: Trading symbol
            decay_rate: Weight decay factor (0.9 = 10% decay per hour)
            max_age_hours: Maximum age of signals to consider

        Returns:
            Dictionary with weighted sentiment metrics, or None if no data
        """
        history = self._signal_history.get(symbol, [])
        if not history:
            return None

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=max_age_hours)

        # Filter and weight signals
        weighted_scores = []
        weighted_confidences = []
        total_weight = 0.0

        for signal in history:
            # Skip signals older than cutoff
            if signal.timestamp < cutoff:
                continue

            # Calculate age in hours
            age_hours = (now - signal.timestamp).total_seconds() / 3600.0

            # Exponential decay: weight = decay_rate ^ age_hours
            weight = decay_rate**age_hours

            weighted_scores.append(signal.sentiment_score * weight)
            weighted_confidences.append(signal.confidence * weight)
            total_weight += weight

        if total_weight == 0:
            return None

        # Calculate weighted averages
        weighted_avg_score = sum(weighted_scores) / total_weight
        weighted_avg_confidence = sum(weighted_confidences) / total_weight

        # Get trend (recent vs older sentiment)
        recent_signals = [s for s in history if s.timestamp >= now - timedelta(hours=1)]
        older_signals = [
            s for s in history if s.timestamp >= now - timedelta(hours=6) and s.timestamp < now - timedelta(hours=1)
        ]

        if recent_signals and older_signals:
            recent_avg = sum(s.sentiment_score for s in recent_signals) / len(recent_signals)
            older_avg = sum(s.sentiment_score for s in older_signals) / len(older_signals)
            trend = recent_avg - older_avg
        else:
            trend = 0.0

        return {
            "symbol": symbol,
            "weighted_score": weighted_avg_score,
            "weighted_confidence": weighted_avg_confidence,
            "trend": trend,
            "signals_used": len(weighted_scores),
            "total_weight": total_weight,
            "is_bullish": weighted_avg_score > 0.1,
            "is_bearish": weighted_avg_score < -0.1,
        }

    def get_bulk_sentiment(
        self,
        symbols: list[str],
        decay_rate: float = 0.9,
    ) -> dict[str, dict[str, Any]]:
        """
        Get time-weighted sentiment for multiple symbols.

        Args:
            symbols: List of symbols to analyze
            decay_rate: Weight decay factor

        Returns:
            Dictionary mapping symbol -> weighted sentiment data
        """
        results = {}
        for symbol in symbols:
            weighted = self.get_weighted_sentiment(symbol, decay_rate)
            if weighted:
                results[symbol] = weighted
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get filter statistics."""
        total = self._filter_stats["total_checks"]
        return {
            **self._filter_stats,
            "allow_rate": (self._filter_stats["allowed"] / total if total > 0 else 0.0),
            "block_rate": (self._filter_stats["blocked"] / total if total > 0 else 0.0),
            "review_rate": (self._filter_stats["review_required"] / total if total > 0 else 0.0),
            "symbols_tracked": len(self._signal_history),
            "active_overrides": len(self._override_symbols),
        }

    def reset_stats(self) -> None:
        """Reset filter statistics."""
        self._filter_stats = {
            "total_checks": 0,
            "allowed": 0,
            "blocked": 0,
            "review_required": 0,
        }

    def clear_history(self, symbol: str | None = None) -> None:
        """Clear signal history."""
        if symbol:
            self._signal_history.pop(symbol, None)
        else:
            self._signal_history.clear()

    # ==========================================================================
    # Listener Management
    # ==========================================================================

    def add_listener(self, callback: Callable[[FilterResult], None]) -> None:
        """Add filter result listener."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[FilterResult], None]) -> None:
        """Remove filter result listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    # ==========================================================================
    # Configuration
    # ==========================================================================

    def update_thresholds(
        self,
        min_sentiment_for_long: float | None = None,
        max_sentiment_for_short: float | None = None,
        min_confidence: float | None = None,
    ) -> None:
        """Update filter thresholds."""
        if min_sentiment_for_long is not None:
            self.min_sentiment_for_long = min_sentiment_for_long
        if max_sentiment_for_short is not None:
            self.max_sentiment_for_short = max_sentiment_for_short
        if min_confidence is not None:
            self.min_confidence = min_confidence

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        return {
            "min_sentiment_for_long": self.min_sentiment_for_long,
            "max_sentiment_for_short": self.max_sentiment_for_short,
            "min_confidence": self.min_confidence,
            "lookback_hours": self.lookback_hours,
            "require_positive_trend": self.require_positive_trend,
            "trend_window_size": self.trend_window_size,
            "max_history_per_symbol": self.max_history_per_symbol,
        }


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_sentiment_filter(
    config: dict[str, Any] | None = None,
) -> SentimentFilter:
    """
    Factory function to create sentiment filter.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured SentimentFilter instance
    """
    config = config or {}

    return SentimentFilter(
        min_sentiment_for_long=config.get("min_sentiment_for_long", 0.0),
        max_sentiment_for_short=config.get("max_sentiment_for_short", 0.0),
        min_confidence=config.get("min_confidence", 0.5),
        lookback_hours=config.get("lookback_hours", 24),
        require_positive_trend=config.get("require_positive_trend", False),
        trend_window_size=config.get("trend_window_size", 3),
        max_history_per_symbol=config.get("max_history_per_symbol", 100),
    )


def create_signal_from_ensemble(
    symbol: str,
    ensemble_result: Any,  # EnsembleResult
) -> SentimentSignal:
    """
    Create SentimentSignal from ensemble result.

    Args:
        symbol: Trading symbol
        ensemble_result: Result from LLMEnsemble

    Returns:
        SentimentSignal for use with filter
    """
    return SentimentSignal(
        symbol=symbol,
        sentiment_score=ensemble_result.sentiment.score,
        confidence=ensemble_result.sentiment.confidence,
        source="ensemble",
        articles_analyzed=len(ensemble_result.individual_results),
        raw_results=ensemble_result.individual_results,
    )
