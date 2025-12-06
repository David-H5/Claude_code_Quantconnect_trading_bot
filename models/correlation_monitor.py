"""
Correlation Monitor (UPGRADE-010 Sprint 4 Expansion)

Monitors position correlations and concentration risk.
Tracks rolling correlations and alerts on high concentration.

Author: Claude Code
Date: December 2025
"""

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class ConcentrationLevel(Enum):
    """Concentration risk levels."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CorrelationConfig:
    """Configuration for correlation monitoring."""

    # Lookback period for correlation calculation
    lookback_days: int = 60

    # Correlation thresholds
    high_correlation_threshold: float = 0.70
    critical_correlation_threshold: float = 0.85

    # Concentration thresholds
    max_correlated_weight: float = 0.40  # Max combined weight of highly correlated positions
    max_single_position_weight: float = 0.25

    # Diversification targets
    min_positions: int = 3
    target_diversification_score: float = 0.70

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lookback_days": self.lookback_days,
            "high_correlation_threshold": self.high_correlation_threshold,
            "critical_correlation_threshold": self.critical_correlation_threshold,
            "max_correlated_weight": self.max_correlated_weight,
            "max_single_position_weight": self.max_single_position_weight,
            "min_positions": self.min_positions,
            "target_diversification_score": self.target_diversification_score,
        }


@dataclass
class CorrelationPair:
    """Correlation between two positions."""

    symbol1: str
    symbol2: str
    correlation: float
    weight1: float = 0.0
    weight2: float = 0.0
    combined_weight: float = 0.0
    observations: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol1": self.symbol1,
            "symbol2": self.symbol2,
            "correlation": self.correlation,
            "weight1": self.weight1,
            "weight2": self.weight2,
            "combined_weight": self.combined_weight,
            "observations": self.observations,
        }


@dataclass
class CorrelationAlert:
    """Alert for correlation/concentration issues."""

    level: ConcentrationLevel
    pair: tuple[str, str] | None = None
    correlation: float = 0.0
    combined_weight: float = 0.0
    message: str = ""
    recommendation: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "pair": list(self.pair) if self.pair else None,
            "correlation": self.correlation,
            "combined_weight": self.combined_weight,
            "message": self.message,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DiversificationScore:
    """Portfolio diversification assessment."""

    score: float  # 0-1 (higher is more diversified)
    effective_positions: float  # Herfindahl-Hirschman based
    concentration_ratio: float  # Top 3 position weight
    avg_correlation: float
    max_correlation: float
    level: str  # "POOR", "FAIR", "GOOD", "EXCELLENT"
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "effective_positions": self.effective_positions,
            "concentration_ratio": self.concentration_ratio,
            "avg_correlation": self.avg_correlation,
            "max_correlation": self.max_correlation,
            "level": self.level,
            "recommendations": self.recommendations,
        }


class CorrelationMonitor:
    """
    Monitor position correlations and concentration risk.

    Tracks rolling correlations between positions and alerts
    when concentration risk is elevated.

    Features:
    - Rolling correlation calculation
    - Concentration risk monitoring
    - Diversification scoring
    - Alert generation for high correlations
    - Position weight tracking

    Example:
        monitor = CorrelationMonitor(config=CorrelationConfig())

        # Update returns daily
        monitor.update_returns("SPY", daily_return_spy)
        monitor.update_returns("QQQ", daily_return_qqq)

        # Update position weights
        monitor.update_weight("SPY", 0.30)
        monitor.update_weight("QQQ", 0.25)

        # Check concentration
        alerts = monitor.check_concentration()
        for alert in alerts:
            print(f"ALERT: {alert.message}")

        # Get diversification score
        score = monitor.get_diversification_score()
        print(f"Diversification: {score.level} ({score.score:.2f})")
    """

    def __init__(self, config: CorrelationConfig | None = None):
        """
        Initialize Correlation Monitor.

        Args:
            config: Correlation monitoring configuration
        """
        self.config = config or CorrelationConfig()
        self._returns: dict[str, deque[float]] = {}
        self._weights: dict[str, float] = {}
        self._alert_callbacks: list[Callable[[CorrelationAlert], None]] = []
        self._correlation_cache: dict[tuple[str, str], float] | None = None
        self._cache_valid = False

    def update_returns(self, symbol: str, daily_return: float) -> None:
        """
        Update returns history for a symbol.

        Args:
            symbol: Position symbol
            daily_return: Daily return (e.g., 0.01 for 1%)
        """
        if symbol not in self._returns:
            self._returns[symbol] = deque(maxlen=self.config.lookback_days)

        self._returns[symbol].append(daily_return)
        self._cache_valid = False

    def update_weight(self, symbol: str, weight: float) -> None:
        """
        Update position weight in portfolio.

        Args:
            symbol: Position symbol
            weight: Weight as fraction of portfolio (0-1)
        """
        if weight <= 0:
            self._weights.pop(symbol, None)
        else:
            self._weights[symbol] = weight

    def remove_position(self, symbol: str) -> None:
        """Remove a position from monitoring."""
        self._returns.pop(symbol, None)
        self._weights.pop(symbol, None)
        self._cache_valid = False

    def clear_all(self) -> None:
        """Clear all positions and returns."""
        self._returns.clear()
        self._weights.clear()
        self._cache_valid = False
        self._correlation_cache = None

    def calculate_correlation(self, symbol1: str, symbol2: str) -> float | None:
        """
        Calculate correlation between two positions.

        Args:
            symbol1: First symbol
            symbol2: Second symbol

        Returns:
            Correlation coefficient or None if insufficient data
        """
        if symbol1 not in self._returns or symbol2 not in self._returns:
            return None

        returns1 = list(self._returns[symbol1])
        returns2 = list(self._returns[symbol2])

        # Need at least 20 observations
        min_obs = min(len(returns1), len(returns2))
        if min_obs < 20:
            return None

        # Align returns
        returns1 = returns1[-min_obs:]
        returns2 = returns2[-min_obs:]

        # Calculate correlation
        try:
            corr = np.corrcoef(returns1, returns2)[0, 1]
            if np.isnan(corr):
                return None
            return float(corr)
        except Exception:
            return None

    def calculate_correlation_matrix(self) -> tuple[np.ndarray, list[str]]:
        """
        Calculate full correlation matrix for all positions.

        Returns:
            Tuple of (correlation_matrix, list_of_symbols)
        """
        symbols = list(self._returns.keys())
        n = len(symbols)

        if n < 2:
            return np.array([[1.0]]) if n == 1 else np.array([]), symbols

        matrix = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                corr = self.calculate_correlation(symbols[i], symbols[j])
                if corr is not None:
                    matrix[i, j] = corr
                    matrix[j, i] = corr

        return matrix, symbols

    def get_high_correlations(self) -> list[CorrelationPair]:
        """
        Get pairs with correlation above threshold.

        Returns:
            List of CorrelationPair objects
        """
        high_corr_pairs = []
        symbols = list(self._returns.keys())

        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i >= j:
                    continue

                corr = self.calculate_correlation(sym1, sym2)
                if corr is not None and abs(corr) >= self.config.high_correlation_threshold:
                    weight1 = self._weights.get(sym1, 0.0)
                    weight2 = self._weights.get(sym2, 0.0)

                    high_corr_pairs.append(
                        CorrelationPair(
                            symbol1=sym1,
                            symbol2=sym2,
                            correlation=corr,
                            weight1=weight1,
                            weight2=weight2,
                            combined_weight=weight1 + weight2,
                            observations=min(
                                len(self._returns.get(sym1, [])),
                                len(self._returns.get(sym2, [])),
                            ),
                        )
                    )

        # Sort by correlation descending
        high_corr_pairs.sort(key=lambda x: abs(x.correlation), reverse=True)
        return high_corr_pairs

    def check_concentration(self) -> list[CorrelationAlert]:
        """
        Check for concentration risk issues.

        Returns:
            List of CorrelationAlert objects
        """
        alerts = []

        # Check single position concentration
        for symbol, weight in self._weights.items():
            if weight > self.config.max_single_position_weight:
                level = (
                    ConcentrationLevel.CRITICAL
                    if weight > self.config.max_single_position_weight * 1.5
                    else ConcentrationLevel.HIGH
                )
                alerts.append(
                    CorrelationAlert(
                        level=level,
                        combined_weight=weight,
                        message=f"Position {symbol} is {weight:.1%} of portfolio",
                        recommendation=f"Consider reducing {symbol} to under {self.config.max_single_position_weight:.0%}",
                    )
                )

        # Check correlated position concentration
        high_corr_pairs = self.get_high_correlations()
        for pair in high_corr_pairs:
            if pair.combined_weight > self.config.max_correlated_weight:
                level = (
                    ConcentrationLevel.CRITICAL
                    if pair.correlation >= self.config.critical_correlation_threshold
                    else ConcentrationLevel.HIGH
                )
                alerts.append(
                    CorrelationAlert(
                        level=level,
                        pair=(pair.symbol1, pair.symbol2),
                        correlation=pair.correlation,
                        combined_weight=pair.combined_weight,
                        message=f"{pair.symbol1} and {pair.symbol2} are {pair.correlation:.0%} correlated "
                        f"with combined weight of {pair.combined_weight:.1%}",
                        recommendation="Consider reducing exposure to one of these positions",
                    )
                )

        # Check minimum positions
        if len(self._weights) < self.config.min_positions and len(self._weights) > 0:
            alerts.append(
                CorrelationAlert(
                    level=ConcentrationLevel.MODERATE,
                    message=f"Only {len(self._weights)} positions (target: {self.config.min_positions}+)",
                    recommendation="Consider adding uncorrelated positions for diversification",
                )
            )

        # Trigger callbacks
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Correlation alert callback error: {e}")

        return alerts

    def get_diversification_score(self) -> DiversificationScore:
        """
        Calculate portfolio diversification score.

        Returns:
            DiversificationScore with assessment
        """
        weights = list(self._weights.values())
        n_positions = len(weights)

        if n_positions == 0:
            return DiversificationScore(
                score=0.0,
                effective_positions=0.0,
                concentration_ratio=0.0,
                avg_correlation=0.0,
                max_correlation=0.0,
                level="POOR",
                recommendations=["Portfolio is empty"],
            )

        # Calculate Herfindahl-Hirschman Index (HHI)
        hhi = sum(w**2 for w in weights)
        effective_positions = 1.0 / hhi if hhi > 0 else n_positions

        # Concentration ratio (top 3)
        sorted_weights = sorted(weights, reverse=True)
        concentration_ratio = sum(sorted_weights[:3]) if len(sorted_weights) >= 3 else sum(sorted_weights)

        # Calculate average and max correlation
        high_corr_pairs = self.get_high_correlations()
        if high_corr_pairs:
            correlations = [abs(p.correlation) for p in high_corr_pairs]
            avg_correlation = sum(correlations) / len(correlations)
            max_correlation = max(correlations)
        else:
            # Calculate average correlation across all pairs
            matrix, symbols = self.calculate_correlation_matrix()
            if len(symbols) >= 2:
                # Get upper triangle values (excluding diagonal)
                upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
                valid_corrs = upper_tri[~np.isnan(upper_tri)]
                avg_correlation = float(np.mean(np.abs(valid_corrs))) if len(valid_corrs) > 0 else 0.0
                max_correlation = float(np.max(np.abs(valid_corrs))) if len(valid_corrs) > 0 else 0.0
            else:
                avg_correlation = 0.0
                max_correlation = 0.0

        # Calculate diversification score (0-1)
        # Factors: effective positions, concentration, correlation
        position_score = min(1.0, effective_positions / max(self.config.min_positions, 1))
        concentration_score = max(0.0, 1.0 - concentration_ratio / 0.8)  # Penalize >80% concentration

        # Only credit correlation if we have multiple positions to correlate
        # Single position = 0 correlation score (no diversification benefit)
        if n_positions >= 2:
            correlation_score = 1.0 - avg_correlation
        else:
            correlation_score = 0.0  # Single position gets no correlation benefit

        # Weighted average
        score = 0.4 * position_score + 0.3 * concentration_score + 0.3 * correlation_score
        score = max(0.0, min(1.0, score))

        # Determine level
        if score >= 0.8:
            level = "EXCELLENT"
        elif score >= 0.6:
            level = "GOOD"
        elif score >= 0.4:
            level = "FAIR"
        else:
            level = "POOR"

        # Generate recommendations
        recommendations = []
        if effective_positions < self.config.min_positions:
            recommendations.append(
                f"Add more positions (current: {effective_positions:.1f}, target: {self.config.min_positions})"
            )
        if concentration_ratio > 0.6:
            recommendations.append("Reduce top position concentration")
        if avg_correlation > 0.5:
            recommendations.append("Add uncorrelated assets")
        if max_correlation > self.config.critical_correlation_threshold:
            recommendations.append("Reduce exposure to highly correlated pairs")

        return DiversificationScore(
            score=score,
            effective_positions=effective_positions,
            concentration_ratio=concentration_ratio,
            avg_correlation=avg_correlation,
            max_correlation=max_correlation,
            level=level,
            recommendations=recommendations,
        )

    def register_alert_callback(self, callback: Callable[[CorrelationAlert], None]) -> None:
        """Register a callback for concentration alerts."""
        self._alert_callbacks.append(callback)

    def get_summary(self) -> dict[str, Any]:
        """Get monitoring summary."""
        score = self.get_diversification_score()
        high_corr = self.get_high_correlations()
        alerts = self.check_concentration()

        return {
            "position_count": len(self._weights),
            "diversification_score": score.to_dict(),
            "high_correlation_pairs": [p.to_dict() for p in high_corr[:5]],
            "alerts": [a.to_dict() for a in alerts],
            "config": self.config.to_dict(),
        }


def create_correlation_monitor(
    lookback_days: int = 60,
    high_correlation_threshold: float = 0.70,
    max_correlated_weight: float = 0.40,
) -> CorrelationMonitor:
    """
    Factory function to create a CorrelationMonitor.

    Args:
        lookback_days: Days of returns history
        high_correlation_threshold: Threshold for high correlation alerts
        max_correlated_weight: Max combined weight for correlated positions

    Returns:
        Configured CorrelationMonitor instance
    """
    config = CorrelationConfig(
        lookback_days=lookback_days,
        high_correlation_threshold=high_correlation_threshold,
        max_correlated_weight=max_correlated_weight,
    )
    return CorrelationMonitor(config=config)
