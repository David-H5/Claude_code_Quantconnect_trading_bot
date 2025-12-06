"""
Market Regime Detection Module

UPGRADE-015 Phase 9: Backtesting Robustness

Provides market regime detection for strategy validation:
- Volatility regime classification
- Trend regime detection
- Correlation regime analysis
- Regime-conditional backtesting

Features:
- Hidden Markov Model-inspired states
- Rolling statistics for detection
- Regime transition analysis
- Strategy performance by regime
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class MarketRegime(Enum):
    """Market regime classifications."""

    # Volatility regimes
    LOW_VOL = "low_volatility"
    NORMAL_VOL = "normal_volatility"
    HIGH_VOL = "high_volatility"
    CRISIS = "crisis"

    # Trend regimes
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

    # Combined
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"


@dataclass
class RegimeState:
    """Current regime state."""

    volatility_regime: MarketRegime
    trend_regime: MarketRegime
    combined_regime: MarketRegime
    confidence: float
    timestamp: datetime

    # Metrics used for classification
    volatility_percentile: float = 0.0
    trend_strength: float = 0.0
    correlation_avg: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "volatility_regime": self.volatility_regime.value,
            "trend_regime": self.trend_regime.value,
            "combined_regime": self.combined_regime.value,
            "confidence": self.confidence,
            "volatility_percentile": self.volatility_percentile,
            "trend_strength": self.trend_strength,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RegimeTransition:
    """Regime transition event."""

    from_regime: MarketRegime
    to_regime: MarketRegime
    timestamp: datetime
    duration_days: int


@dataclass
class RegimeAnalysis:
    """Complete regime analysis results."""

    # Current state
    current_regime: RegimeState

    # Historical analysis
    regime_history: list[RegimeState]
    transitions: list[RegimeTransition]

    # Statistics
    regime_durations: dict[str, float]  # Average days in each regime
    regime_frequencies: dict[str, float]  # Percentage time in each regime
    transition_matrix: dict[str, dict[str, float]]  # Transition probabilities

    # Performance by regime (if provided)
    performance_by_regime: dict[str, dict[str, float]] = field(default_factory=dict)

    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_regime": self.current_regime.to_dict(),
            "regime_frequencies": self.regime_frequencies,
            "regime_durations": self.regime_durations,
            "performance_by_regime": self.performance_by_regime,
        }


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""

    # Volatility thresholds (percentiles)
    low_vol_threshold: float = 25.0
    high_vol_threshold: float = 75.0
    crisis_vol_threshold: float = 95.0

    # Trend thresholds
    strong_trend_threshold: float = 0.03  # 3% monthly
    weak_trend_threshold: float = 0.01  # 1% monthly

    # Rolling windows
    volatility_window: int = 20  # Days for realized vol
    trend_window: int = 60  # Days for trend calculation
    lookback_window: int = 252  # Days for percentile calculation

    # Minimum regime duration
    min_regime_duration: int = 5  # Days


class RegimeDetector:
    """Market regime detection and analysis."""

    def __init__(
        self,
        config: RegimeConfig | None = None,
    ):
        """
        Initialize regime detector.

        Args:
            config: Detection configuration
        """
        self.config = config or RegimeConfig()
        self._history: list[RegimeState] = []
        self._returns_buffer: list[float] = []
        self._vol_history: list[float] = []

    # ==========================================================================
    # Volatility Regime Detection
    # ==========================================================================

    def detect_volatility_regime(
        self,
        returns: list[float],
    ) -> tuple[MarketRegime, float, float]:
        """
        Detect current volatility regime.

        Args:
            returns: Historical returns

        Returns:
            (regime, confidence, percentile)
        """
        if len(returns) < self.config.volatility_window:
            return MarketRegime.NORMAL_VOL, 0.5, 50.0

        returns_array = np.array(returns)

        # Calculate current realized volatility
        current_vol = np.std(returns_array[-self.config.volatility_window :]) * np.sqrt(252)

        # Calculate historical volatility for percentiles
        lookback = min(len(returns), self.config.lookback_window)
        rolling_vols = []
        for i in range(self.config.volatility_window, lookback):
            window_returns = returns_array[i - self.config.volatility_window : i]
            vol = np.std(window_returns) * np.sqrt(252)
            rolling_vols.append(vol)

        if not rolling_vols:
            return MarketRegime.NORMAL_VOL, 0.5, 50.0

        # Calculate percentile
        percentile = np.sum(np.array(rolling_vols) < current_vol) / len(rolling_vols) * 100

        # Classify regime
        if percentile >= self.config.crisis_vol_threshold:
            regime = MarketRegime.CRISIS
            confidence = min((percentile - self.config.crisis_vol_threshold) / 5, 1.0)
        elif percentile >= self.config.high_vol_threshold:
            regime = MarketRegime.HIGH_VOL
            confidence = (percentile - self.config.high_vol_threshold) / (
                self.config.crisis_vol_threshold - self.config.high_vol_threshold
            )
        elif percentile <= self.config.low_vol_threshold:
            regime = MarketRegime.LOW_VOL
            confidence = 1.0 - percentile / self.config.low_vol_threshold
        else:
            regime = MarketRegime.NORMAL_VOL
            # Confidence based on distance from middle
            mid = (self.config.low_vol_threshold + self.config.high_vol_threshold) / 2
            confidence = 1.0 - abs(percentile - mid) / (self.config.high_vol_threshold - mid)

        return regime, confidence, percentile

    # ==========================================================================
    # Trend Regime Detection
    # ==========================================================================

    def detect_trend_regime(
        self,
        prices: list[float],
    ) -> tuple[MarketRegime, float, float]:
        """
        Detect current trend regime.

        Args:
            prices: Historical prices

        Returns:
            (regime, confidence, trend_strength)
        """
        if len(prices) < self.config.trend_window:
            return MarketRegime.SIDEWAYS, 0.5, 0.0

        prices_array = np.array(prices)

        # Calculate trend using linear regression
        x = np.arange(self.config.trend_window)
        y = prices_array[-self.config.trend_window :]

        # Normalized trend (monthly return equivalent)
        slope = np.polyfit(x, y, 1)[0]
        monthly_trend = (slope * 21) / y[0]  # 21 trading days per month

        # Calculate R-squared for confidence
        y_pred = np.polyval(np.polyfit(x, y, 1), x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Classify regime
        if monthly_trend > self.config.strong_trend_threshold:
            regime = MarketRegime.STRONG_UPTREND
        elif monthly_trend > self.config.weak_trend_threshold:
            regime = MarketRegime.WEAK_UPTREND
        elif monthly_trend < -self.config.strong_trend_threshold:
            regime = MarketRegime.STRONG_DOWNTREND
        elif monthly_trend < -self.config.weak_trend_threshold:
            regime = MarketRegime.WEAK_DOWNTREND
        else:
            regime = MarketRegime.SIDEWAYS

        return regime, r_squared, monthly_trend

    # ==========================================================================
    # Combined Regime Detection
    # ==========================================================================

    def detect_combined_regime(
        self,
        vol_regime: MarketRegime,
        trend_regime: MarketRegime,
    ) -> MarketRegime:
        """
        Determine combined regime from volatility and trend.

        Args:
            vol_regime: Volatility regime
            trend_regime: Trend regime

        Returns:
            Combined regime
        """
        is_bull = trend_regime in [
            MarketRegime.STRONG_UPTREND,
            MarketRegime.WEAK_UPTREND,
        ]
        is_bear = trend_regime in [
            MarketRegime.STRONG_DOWNTREND,
            MarketRegime.WEAK_DOWNTREND,
        ]
        is_high_vol = vol_regime in [MarketRegime.HIGH_VOL, MarketRegime.CRISIS]
        is_low_vol = vol_regime == MarketRegime.LOW_VOL

        if is_bull and is_low_vol:
            return MarketRegime.BULL_LOW_VOL
        elif is_bull and is_high_vol:
            return MarketRegime.BULL_HIGH_VOL
        elif is_bear and is_low_vol:
            return MarketRegime.BEAR_LOW_VOL
        elif is_bear and is_high_vol:
            return MarketRegime.BEAR_HIGH_VOL
        else:
            return vol_regime  # Default to volatility regime

    # ==========================================================================
    # Full Analysis
    # ==========================================================================

    def analyze(
        self,
        returns: list[float],
        prices: list[float],
        timestamps: list[datetime] | None = None,
        strategy_returns: list[float] | None = None,
    ) -> RegimeAnalysis:
        """
        Perform full regime analysis.

        Args:
            returns: Historical returns
            prices: Historical prices
            timestamps: Optional timestamps
            strategy_returns: Optional strategy returns for performance analysis

        Returns:
            RegimeAnalysis
        """
        if not returns or not prices:
            dummy_state = RegimeState(
                volatility_regime=MarketRegime.NORMAL_VOL,
                trend_regime=MarketRegime.SIDEWAYS,
                combined_regime=MarketRegime.NORMAL_VOL,
                confidence=0.5,
                timestamp=datetime.utcnow(),
            )
            return RegimeAnalysis(
                current_regime=dummy_state,
                regime_history=[],
                transitions=[],
                regime_durations={},
                regime_frequencies={},
                transition_matrix={},
            )

        # Generate timestamps if not provided
        if timestamps is None:
            from datetime import timedelta

            base = datetime.utcnow() - timedelta(days=len(returns))
            timestamps = [base + timedelta(days=i) for i in range(len(returns))]

        # Detect regimes over time
        regime_history = []
        min_window = max(self.config.volatility_window, self.config.trend_window)

        for i in range(min_window, len(returns)):
            window_returns = returns[: i + 1]
            window_prices = prices[: i + 1]

            vol_regime, vol_conf, vol_pct = self.detect_volatility_regime(window_returns)
            trend_regime, trend_conf, trend_str = self.detect_trend_regime(window_prices)
            combined = self.detect_combined_regime(vol_regime, trend_regime)

            state = RegimeState(
                volatility_regime=vol_regime,
                trend_regime=trend_regime,
                combined_regime=combined,
                confidence=(vol_conf + trend_conf) / 2,
                timestamp=timestamps[i],
                volatility_percentile=vol_pct,
                trend_strength=trend_str,
            )
            regime_history.append(state)

        # Get current state
        current_state = regime_history[-1] if regime_history else None

        # Detect transitions
        transitions = self._detect_transitions(regime_history)

        # Calculate statistics
        regime_durations = self._calculate_durations(regime_history)
        regime_frequencies = self._calculate_frequencies(regime_history)
        transition_matrix = self._calculate_transition_matrix(transitions)

        # Performance by regime
        performance_by_regime = {}
        if strategy_returns and len(strategy_returns) >= len(regime_history):
            performance_by_regime = self._calculate_performance_by_regime(
                regime_history, strategy_returns[-len(regime_history) :]
            )

        return RegimeAnalysis(
            current_regime=current_state
            or RegimeState(
                volatility_regime=MarketRegime.NORMAL_VOL,
                trend_regime=MarketRegime.SIDEWAYS,
                combined_regime=MarketRegime.NORMAL_VOL,
                confidence=0.5,
                timestamp=datetime.utcnow(),
            ),
            regime_history=regime_history,
            transitions=transitions,
            regime_durations=regime_durations,
            regime_frequencies=regime_frequencies,
            transition_matrix=transition_matrix,
            performance_by_regime=performance_by_regime,
        )

    def _detect_transitions(
        self,
        regime_history: list[RegimeState],
    ) -> list[RegimeTransition]:
        """Detect regime transitions."""
        transitions = []
        if len(regime_history) < 2:
            return transitions

        current_regime = regime_history[0].combined_regime
        regime_start_idx = 0

        for i in range(1, len(regime_history)):
            if regime_history[i].combined_regime != current_regime:
                duration = i - regime_start_idx
                if duration >= self.config.min_regime_duration:
                    transitions.append(
                        RegimeTransition(
                            from_regime=current_regime,
                            to_regime=regime_history[i].combined_regime,
                            timestamp=regime_history[i].timestamp,
                            duration_days=duration,
                        )
                    )
                current_regime = regime_history[i].combined_regime
                regime_start_idx = i

        return transitions

    def _calculate_durations(
        self,
        regime_history: list[RegimeState],
    ) -> dict[str, float]:
        """Calculate average duration of each regime."""
        durations: dict[str, list[int]] = {}

        if not regime_history:
            return {}

        current_regime = regime_history[0].combined_regime.value
        current_duration = 1

        for state in regime_history[1:]:
            if state.combined_regime.value == current_regime:
                current_duration += 1
            else:
                if current_regime not in durations:
                    durations[current_regime] = []
                durations[current_regime].append(current_duration)
                current_regime = state.combined_regime.value
                current_duration = 1

        # Add final regime
        if current_regime not in durations:
            durations[current_regime] = []
        durations[current_regime].append(current_duration)

        return {k: np.mean(v) for k, v in durations.items()}

    def _calculate_frequencies(
        self,
        regime_history: list[RegimeState],
    ) -> dict[str, float]:
        """Calculate percentage time in each regime."""
        if not regime_history:
            return {}

        counts: dict[str, int] = {}
        for state in regime_history:
            regime = state.combined_regime.value
            counts[regime] = counts.get(regime, 0) + 1

        total = len(regime_history)
        return {k: v / total for k, v in counts.items()}

    def _calculate_transition_matrix(
        self,
        transitions: list[RegimeTransition],
    ) -> dict[str, dict[str, float]]:
        """Calculate regime transition probabilities."""
        matrix: dict[str, dict[str, int]] = {}

        for t in transitions:
            from_r = t.from_regime.value
            to_r = t.to_regime.value

            if from_r not in matrix:
                matrix[from_r] = {}
            matrix[from_r][to_r] = matrix[from_r].get(to_r, 0) + 1

        # Normalize to probabilities
        result: dict[str, dict[str, float]] = {}
        for from_r, to_dict in matrix.items():
            total = sum(to_dict.values())
            result[from_r] = {k: v / total for k, v in to_dict.items()}

        return result

    def _calculate_performance_by_regime(
        self,
        regime_history: list[RegimeState],
        strategy_returns: list[float],
    ) -> dict[str, dict[str, float]]:
        """Calculate strategy performance in each regime."""
        returns_by_regime: dict[str, list[float]] = {}

        for state, ret in zip(regime_history, strategy_returns):
            regime = state.combined_regime.value
            if regime not in returns_by_regime:
                returns_by_regime[regime] = []
            returns_by_regime[regime].append(ret)

        performance = {}
        for regime, rets in returns_by_regime.items():
            rets_array = np.array(rets)
            performance[regime] = {
                "mean_return": float(np.mean(rets_array)),
                "total_return": float(np.prod(1 + rets_array) - 1),
                "volatility": float(np.std(rets_array) * np.sqrt(252)),
                "sharpe": float(
                    np.mean(rets_array) / np.std(rets_array) * np.sqrt(252) if np.std(rets_array) > 0 else 0
                ),
                "num_days": len(rets),
            }

        return performance


def create_regime_detector(
    volatility_window: int = 20,
    trend_window: int = 60,
    low_vol_threshold: float = 25.0,
    high_vol_threshold: float = 75.0,
) -> RegimeDetector:
    """
    Factory function to create a regime detector.

    Args:
        volatility_window: Window for volatility calculation
        trend_window: Window for trend calculation
        low_vol_threshold: Percentile threshold for low vol
        high_vol_threshold: Percentile threshold for high vol

    Returns:
        Configured RegimeDetector
    """
    config = RegimeConfig(
        volatility_window=volatility_window,
        trend_window=trend_window,
        low_vol_threshold=low_vol_threshold,
        high_vol_threshold=high_vol_threshold,
    )
    return RegimeDetector(config)
