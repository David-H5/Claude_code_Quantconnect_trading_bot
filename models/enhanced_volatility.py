"""
Enhanced Volatility Analysis Module

Advanced volatility analysis including:
- IV Rank and IV Percentile calculations
- Realized vs Implied volatility comparison
- Volatility regime detection (GARCH-inspired)
- Volatility clustering analysis
- Mean reversion signals
- LLM-ready volatility summaries

Based on research from Barchart, TastyTrade, SpotGamma, and academic papers.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class VolatilityRegime(Enum):
    """Market volatility regime classification."""

    VERY_LOW = "very_low"  # IV Percentile < 10
    LOW = "low"  # IV Percentile 10-30
    NORMAL = "normal"  # IV Percentile 30-70
    ELEVATED = "elevated"  # IV Percentile 70-90
    EXTREME = "extreme"  # IV Percentile > 90


class VolatilityTrend(Enum):
    """Volatility trend direction."""

    EXPANDING = "expanding"
    CONTRACTING = "contracting"
    STABLE = "stable"


@dataclass
class VolatilitySnapshot:
    """Point-in-time volatility data."""

    timestamp: datetime
    implied_volatility: float
    underlying_price: float
    realized_vol_5d: float = 0.0
    realized_vol_20d: float = 0.0
    realized_vol_60d: float = 0.0
    vix_level: float = 0.0  # Market-wide vol reference


@dataclass
class IVMetrics:
    """Implied volatility ranking metrics."""

    current_iv: float
    iv_rank: float  # 0-100, position in 52-week range
    iv_percentile: float  # 0-100, % of days below current
    iv_52w_high: float
    iv_52w_low: float
    iv_30d_avg: float
    iv_zscore: float  # Standard deviations from mean

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "current_iv": self.current_iv,
            "iv_rank": self.iv_rank,
            "iv_percentile": self.iv_percentile,
            "52w_high": self.iv_52w_high,
            "52w_low": self.iv_52w_low,
            "30d_avg": self.iv_30d_avg,
            "zscore": self.iv_zscore,
        }


@dataclass
class RealizedVolMetrics:
    """Realized volatility metrics."""

    rv_5d: float  # 5-day realized vol
    rv_20d: float  # 20-day (1-month)
    rv_60d: float  # 60-day (3-month)
    rv_252d: float  # 252-day (1-year)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "5d": self.rv_5d,
            "20d": self.rv_20d,
            "60d": self.rv_60d,
            "252d": self.rv_252d,
        }


@dataclass
class VolatilityPremium:
    """IV vs RV premium analysis."""

    iv_rv_spread: float  # IV - RV
    iv_rv_ratio: float  # IV / RV
    premium_percentile: float  # Historical percentile of spread
    is_iv_overpriced: bool
    suggested_strategy: str  # "sell_premium" or "buy_premium"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spread": self.iv_rv_spread,
            "ratio": self.iv_rv_ratio,
            "percentile": self.premium_percentile,
            "iv_overpriced": self.is_iv_overpriced,
            "strategy": self.suggested_strategy,
        }


@dataclass
class VolatilityRegimeAnalysis:
    """Comprehensive regime analysis."""

    current_regime: VolatilityRegime
    trend: VolatilityTrend
    days_in_regime: int
    regime_probability: float  # Confidence in regime classification
    mean_reversion_signal: float  # -1 to +1, positive = expect vol to rise
    clustering_strength: float  # 0-1, how strong is vol persistence

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime": self.current_regime.value,
            "trend": self.trend.value,
            "days_in_regime": self.days_in_regime,
            "probability": self.regime_probability,
            "mean_reversion_signal": self.mean_reversion_signal,
            "clustering_strength": self.clustering_strength,
        }


class EnhancedVolatilityAnalyzer:
    """
    Advanced volatility analysis for options trading decisions.

    Provides IV Rank, IV Percentile, regime detection, and
    LLM-ready summaries for trading decisions.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        regime_window: int = 20,
    ):
        """
        Initialize analyzer.

        Args:
            lookback_days: Days of history for percentile calculations
            regime_window: Window for regime detection
        """
        self.lookback_days = lookback_days
        self.regime_window = regime_window
        self.history: dict[str, list[VolatilitySnapshot]] = {}
        self.iv_history: dict[str, list[float]] = {}
        self.price_history: dict[str, list[float]] = {}
        self.iv_rv_spread_history: dict[str, list[float]] = {}

    def update(
        self,
        symbol: str,
        implied_volatility: float,
        underlying_price: float,
        timestamp: datetime | None = None,
        vix_level: float = 0.0,
    ) -> None:
        """
        Update with new volatility data.

        Args:
            symbol: Underlying symbol
            implied_volatility: Current ATM IV
            underlying_price: Current underlying price
            timestamp: Data timestamp
            vix_level: VIX level for context
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Initialize if needed
        if symbol not in self.history:
            self.history[symbol] = []
            self.iv_history[symbol] = []
            self.price_history[symbol] = []
            self.iv_rv_spread_history[symbol] = []

        # Calculate realized volatility
        rv_5d = self._calculate_realized_vol(symbol, 5)
        rv_20d = self._calculate_realized_vol(symbol, 20)
        rv_60d = self._calculate_realized_vol(symbol, 60)

        snapshot = VolatilitySnapshot(
            timestamp=timestamp,
            implied_volatility=implied_volatility,
            underlying_price=underlying_price,
            realized_vol_5d=rv_5d,
            realized_vol_20d=rv_20d,
            realized_vol_60d=rv_60d,
            vix_level=vix_level,
        )

        self.history[symbol].append(snapshot)
        self.iv_history[symbol].append(implied_volatility)
        self.price_history[symbol].append(underlying_price)

        # Track IV-RV spread
        if rv_20d > 0:
            spread = implied_volatility - rv_20d
            self.iv_rv_spread_history[symbol].append(spread)

        # Trim history
        max_history = self.lookback_days + 60
        if len(self.history[symbol]) > max_history:
            self.history[symbol] = self.history[symbol][-max_history:]
            self.iv_history[symbol] = self.iv_history[symbol][-max_history:]
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
            self.iv_rv_spread_history[symbol] = self.iv_rv_spread_history[symbol][-max_history:]

    def _calculate_realized_vol(
        self,
        symbol: str,
        window: int,
        annualization: float = 252,
    ) -> float:
        """Calculate realized volatility from price history."""
        if symbol not in self.price_history:
            return 0.0

        prices = self.price_history[symbol]
        if len(prices) < window + 1:
            return 0.0

        # Calculate log returns
        recent_prices = prices[-(window + 1) :]
        returns = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i - 1] > 0:
                ret = math.log(recent_prices[i] / recent_prices[i - 1])
                returns.append(ret)

        if len(returns) < 2:
            return 0.0

        # Standard deviation of returns
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        daily_vol = math.sqrt(variance)

        # Annualize
        return daily_vol * math.sqrt(annualization)

    def get_iv_metrics(self, symbol: str) -> IVMetrics | None:
        """
        Calculate IV Rank and IV Percentile.

        IV Rank = (Current IV - 52w Low) / (52w High - 52w Low) * 100
        IV Percentile = % of days where IV was below current level
        """
        if symbol not in self.iv_history:
            return None

        iv_data = self.iv_history[symbol]
        if len(iv_data) < 20:
            return None

        current_iv = iv_data[-1]

        # Use available history up to lookback_days
        lookback = min(len(iv_data), self.lookback_days)
        historical_iv = iv_data[-lookback:]

        # IV Rank calculation
        iv_52w_high = max(historical_iv)
        iv_52w_low = min(historical_iv)

        if iv_52w_high == iv_52w_low:
            iv_rank = 50.0
        else:
            iv_rank = ((current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low)) * 100
            iv_rank = max(0, min(100, iv_rank))

        # IV Percentile calculation
        days_below = sum(1 for iv in historical_iv if iv < current_iv)
        iv_percentile = (days_below / len(historical_iv)) * 100

        # 30-day average
        recent_30 = iv_data[-30:] if len(iv_data) >= 30 else iv_data
        iv_30d_avg = sum(recent_30) / len(recent_30)

        # Z-score
        mean_iv = sum(historical_iv) / len(historical_iv)
        if len(historical_iv) > 1:
            variance = sum((iv - mean_iv) ** 2 for iv in historical_iv) / (len(historical_iv) - 1)
            std_iv = math.sqrt(variance)
            iv_zscore = (current_iv - mean_iv) / std_iv if std_iv > 0 else 0
        else:
            iv_zscore = 0

        return IVMetrics(
            current_iv=current_iv,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            iv_52w_high=iv_52w_high,
            iv_52w_low=iv_52w_low,
            iv_30d_avg=iv_30d_avg,
            iv_zscore=iv_zscore,
        )

    def get_realized_vol_metrics(self, symbol: str) -> RealizedVolMetrics | None:
        """Get realized volatility across multiple timeframes."""
        rv_5d = self._calculate_realized_vol(symbol, 5)
        rv_20d = self._calculate_realized_vol(symbol, 20)
        rv_60d = self._calculate_realized_vol(symbol, 60)
        rv_252d = self._calculate_realized_vol(symbol, 252)

        return RealizedVolMetrics(
            rv_5d=rv_5d,
            rv_20d=rv_20d,
            rv_60d=rv_60d,
            rv_252d=rv_252d,
        )

    def get_volatility_premium(self, symbol: str) -> VolatilityPremium | None:
        """
        Analyze IV vs RV premium.

        Key insight: IV typically overstates realized moves.
        When IV >> RV, selling premium is favorable.
        When IV << RV, buying options may be underpriced.
        """
        iv_metrics = self.get_iv_metrics(symbol)
        rv_metrics = self.get_realized_vol_metrics(symbol)

        if iv_metrics is None or rv_metrics is None:
            return None

        if rv_metrics.rv_20d == 0:
            return None

        current_iv = iv_metrics.current_iv
        rv_20d = rv_metrics.rv_20d

        # IV - RV spread
        iv_rv_spread = current_iv - rv_20d
        iv_rv_ratio = current_iv / rv_20d

        # Historical percentile of spread
        if symbol in self.iv_rv_spread_history and len(self.iv_rv_spread_history[symbol]) >= 20:
            spreads = self.iv_rv_spread_history[symbol]
            days_below = sum(1 for s in spreads if s < iv_rv_spread)
            premium_percentile = (days_below / len(spreads)) * 100
        else:
            premium_percentile = 50.0

        # Trading recommendation
        # IV is typically 15-20% higher than realized
        # When ratio > 1.3, IV is significantly overpriced
        is_iv_overpriced = iv_rv_ratio > 1.15

        if iv_rv_ratio > 1.3 and iv_metrics.iv_percentile > 50:
            suggested_strategy = "sell_premium"
        elif iv_rv_ratio < 0.9 and iv_metrics.iv_percentile < 30:
            suggested_strategy = "buy_premium"
        else:
            suggested_strategy = "neutral"

        return VolatilityPremium(
            iv_rv_spread=iv_rv_spread,
            iv_rv_ratio=iv_rv_ratio,
            premium_percentile=premium_percentile,
            is_iv_overpriced=is_iv_overpriced,
            suggested_strategy=suggested_strategy,
        )

    def get_regime_analysis(self, symbol: str) -> VolatilityRegimeAnalysis | None:
        """
        Detect volatility regime and provide mean reversion signals.

        Uses GARCH-inspired volatility clustering analysis.
        """
        iv_metrics = self.get_iv_metrics(symbol)
        if iv_metrics is None:
            return None

        if symbol not in self.iv_history:
            return None

        iv_data = self.iv_history[symbol]

        # Classify regime based on IV percentile
        pct = iv_metrics.iv_percentile
        if pct < 10:
            regime = VolatilityRegime.VERY_LOW
        elif pct < 30:
            regime = VolatilityRegime.LOW
        elif pct < 70:
            regime = VolatilityRegime.NORMAL
        elif pct < 90:
            regime = VolatilityRegime.ELEVATED
        else:
            regime = VolatilityRegime.EXTREME

        # Determine trend
        if len(iv_data) >= 10:
            recent_5 = sum(iv_data[-5:]) / 5
            prior_5 = sum(iv_data[-10:-5]) / 5
            change_pct = (recent_5 - prior_5) / prior_5 if prior_5 > 0 else 0

            if change_pct > 0.05:
                trend = VolatilityTrend.EXPANDING
            elif change_pct < -0.05:
                trend = VolatilityTrend.CONTRACTING
            else:
                trend = VolatilityTrend.STABLE
        else:
            trend = VolatilityTrend.STABLE

        # Days in current regime (simplified)
        days_in_regime = 1
        current_regime_range = self._get_regime_range(regime)
        for i in range(len(iv_data) - 2, -1, -1):
            iv = iv_data[i]
            # Calculate rough percentile for historical point
            if current_regime_range[0] <= iv <= current_regime_range[1]:
                days_in_regime += 1
            else:
                break

        # Mean reversion signal
        # Volatility tends to mean-revert
        # High IV percentile -> expect contraction (-1)
        # Low IV percentile -> expect expansion (+1)
        mean_reversion_signal = (50 - pct) / 50  # Normalized to -1 to +1

        # Volatility clustering strength (simplified GARCH alpha)
        # Measures persistence of volatility shocks
        if len(iv_data) >= 20:
            # Calculate autocorrelation of squared returns
            changes = [iv_data[i] - iv_data[i - 1] for i in range(1, len(iv_data))]
            sq_changes = [c**2 for c in changes[-20:]]
            if len(sq_changes) >= 2:
                mean_sq = sum(sq_changes) / len(sq_changes)
                # Lag-1 autocorrelation
                cov = sum(
                    (sq_changes[i] - mean_sq) * (sq_changes[i - 1] - mean_sq) for i in range(1, len(sq_changes))
                ) / (len(sq_changes) - 1)
                var = sum((s - mean_sq) ** 2 for s in sq_changes) / (len(sq_changes) - 1)
                clustering_strength = abs(cov / var) if var > 0 else 0
                clustering_strength = min(1.0, clustering_strength)
            else:
                clustering_strength = 0.5
        else:
            clustering_strength = 0.5

        # Regime probability (confidence)
        # Higher confidence when further from regime boundaries
        if pct < 15 or pct > 85:
            regime_probability = 0.9
        elif pct < 25 or pct > 75:
            regime_probability = 0.75
        else:
            regime_probability = 0.6

        return VolatilityRegimeAnalysis(
            current_regime=regime,
            trend=trend,
            days_in_regime=days_in_regime,
            regime_probability=regime_probability,
            mean_reversion_signal=mean_reversion_signal,
            clustering_strength=clustering_strength,
        )

    def _get_regime_range(self, regime: VolatilityRegime) -> tuple[float, float]:
        """Get approximate IV range for a regime (for classification)."""
        # These are approximate ranges based on typical IV levels
        ranges = {
            VolatilityRegime.VERY_LOW: (0, 0.15),
            VolatilityRegime.LOW: (0.10, 0.20),
            VolatilityRegime.NORMAL: (0.15, 0.35),
            VolatilityRegime.ELEVATED: (0.30, 0.50),
            VolatilityRegime.EXTREME: (0.45, 1.0),
        }
        return ranges.get(regime, (0, 1.0))

    def get_trading_signals(self, symbol: str) -> dict[str, Any]:
        """
        Get actionable trading signals based on volatility analysis.

        Returns signals suitable for both automated trading and LLM analysis.
        """
        iv_metrics = self.get_iv_metrics(symbol)
        rv_metrics = self.get_realized_vol_metrics(symbol)
        premium = self.get_volatility_premium(symbol)
        regime = self.get_regime_analysis(symbol)

        if iv_metrics is None:
            return {"symbol": symbol, "error": "Insufficient data"}

        signals = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "iv_metrics": iv_metrics.to_dict(),
            "rv_metrics": rv_metrics.to_dict() if rv_metrics else None,
            "premium_analysis": premium.to_dict() if premium else None,
            "regime": regime.to_dict() if regime else None,
            "recommendations": [],
        }

        # Generate recommendations
        recommendations = []

        # IV Rank based signals
        if iv_metrics.iv_rank > 70:
            recommendations.append(
                {
                    "signal": "HIGH_IV_RANK",
                    "action": "Consider selling premium (covered calls, credit spreads)",
                    "confidence": min(0.9, iv_metrics.iv_rank / 100),
                }
            )
        elif iv_metrics.iv_rank < 30:
            recommendations.append(
                {
                    "signal": "LOW_IV_RANK",
                    "action": "Consider buying options or debit spreads",
                    "confidence": min(0.9, (100 - iv_metrics.iv_rank) / 100),
                }
            )

        # Premium analysis signals
        if premium and premium.suggested_strategy == "sell_premium":
            recommendations.append(
                {
                    "signal": "IV_OVERPRICED",
                    "action": f"IV/RV ratio {premium.iv_rv_ratio:.2f} - sell premium",
                    "confidence": 0.75,
                }
            )
        elif premium and premium.suggested_strategy == "buy_premium":
            recommendations.append(
                {
                    "signal": "IV_UNDERPRICED",
                    "action": f"IV/RV ratio {premium.iv_rv_ratio:.2f} - buy options",
                    "confidence": 0.7,
                }
            )

        # Regime signals
        if regime:
            if regime.current_regime == VolatilityRegime.EXTREME:
                recommendations.append(
                    {
                        "signal": "EXTREME_VOLATILITY",
                        "action": "Caution - reduce position sizes, expect mean reversion",
                        "confidence": regime.regime_probability,
                    }
                )
            elif regime.current_regime == VolatilityRegime.VERY_LOW:
                recommendations.append(
                    {
                        "signal": "VOLATILITY_FLOOR",
                        "action": "Options cheap - consider long straddles/strangles",
                        "confidence": regime.regime_probability,
                    }
                )

            if regime.mean_reversion_signal > 0.5:
                recommendations.append(
                    {
                        "signal": "EXPECT_VOL_EXPANSION",
                        "action": "Volatility likely to increase - favor long vega",
                        "confidence": abs(regime.mean_reversion_signal),
                    }
                )
            elif regime.mean_reversion_signal < -0.5:
                recommendations.append(
                    {
                        "signal": "EXPECT_VOL_CONTRACTION",
                        "action": "Volatility likely to decrease - favor short vega",
                        "confidence": abs(regime.mean_reversion_signal),
                    }
                )

        signals["recommendations"] = recommendations

        return signals

    def get_llm_summary(self, symbol: str) -> str:
        """
        Generate human-readable summary for LLM analysis.

        Returns a formatted string suitable for LLM context.
        """
        signals = self.get_trading_signals(symbol)

        if "error" in signals:
            return f"Insufficient volatility data for {symbol}"

        iv = signals["iv_metrics"]
        regime = signals.get("regime", {})
        premium = signals.get("premium_analysis", {})

        summary = f"""
VOLATILITY ANALYSIS FOR {symbol}
================================
Current IV: {iv['current_iv']:.1%}
IV Rank: {iv['iv_rank']:.1f}% (position in 52-week range)
IV Percentile: {iv['iv_percentile']:.1f}% (% of days below current)
52-Week Range: {iv['52w_low']:.1%} - {iv['52w_high']:.1%}
30-Day Avg IV: {iv['30d_avg']:.1%}
Z-Score: {iv['zscore']:.2f} standard deviations from mean
"""

        if regime:
            summary += f"""
VOLATILITY REGIME
-----------------
Current Regime: {regime.get('regime', 'N/A').upper()}
Trend: {regime.get('trend', 'N/A')}
Days in Regime: {regime.get('days_in_regime', 0)}
Mean Reversion Signal: {regime.get('mean_reversion_signal', 0):.2f} (-1=expect drop, +1=expect rise)
"""

        if premium:
            summary += f"""
IV vs REALIZED VOL
------------------
IV/RV Ratio: {premium.get('ratio', 0):.2f}
IV-RV Spread: {premium.get('spread', 0):.1%}
IV Overpriced: {premium.get('iv_overpriced', False)}
Suggested Strategy: {premium.get('strategy', 'neutral').upper()}
"""

        if signals.get("recommendations"):
            summary += "\nRECOMMENDATIONS\n---------------\n"
            for rec in signals["recommendations"]:
                summary += f"- [{rec['signal']}] {rec['action']} (confidence: {rec['confidence']:.0%})\n"

        return summary

    def update_from_qc_chain(
        self,
        algorithm,
        slice,
        underlying_symbol: str,
    ) -> None:
        """
        Update volatility data from QuantConnect option chain.

        INTEGRATION: Call this from algorithm.OnData()

        Example:
            def OnData(self, slice):
                if self.IsWarmingUp:
                    return

                self.vol_analyzer.update_from_qc_chain(self, slice, "SPY")
                metrics = self.vol_analyzer.get_iv_metrics("SPY")
                self.Debug(f"IV Rank: {metrics.iv_rank:.1f}%")

        Args:
            algorithm: QCAlgorithm instance
            slice: Slice object from OnData
            underlying_symbol: Underlying symbol (e.g., "SPY")
        """
        # Get underlying price
        if not algorithm.Securities.ContainsKey(underlying_symbol):
            return

        underlying_price = algorithm.Securities[underlying_symbol].Price

        # Find option chain for this underlying
        # Official QuantConnect Python API: use snake_case option_chains
        chain = None
        for chain_data in slice.option_chains.values():
            # Each chain has an Underlying property
            if str(chain_data.Underlying.Symbol) == underlying_symbol:
                chain = chain_data
                break

        if chain is None:
            return

        # Calculate ATM IV from option chain
        # Find ATM options and average their IV
        atm_contracts = []
        for contract in chain:
            # Look for near-ATM contracts
            if abs(contract.Strike - underlying_price) <= underlying_price * 0.02:  # Within 2% of ATM
                if hasattr(contract, "ImpliedVolatility") and contract.ImpliedVolatility > 0:
                    atm_contracts.append(contract)

        if not atm_contracts:
            # Fallback: use any contracts with IV
            for contract in chain:
                if hasattr(contract, "ImpliedVolatility") and contract.ImpliedVolatility > 0:
                    atm_contracts.append(contract)

        if atm_contracts:
            # Average IV of ATM contracts
            avg_iv = sum(c.ImpliedVolatility for c in atm_contracts) / len(atm_contracts)

            # Get VIX if available (optional)
            vix_level = 0.0
            if algorithm.Securities.ContainsKey("VIX"):
                vix_level = algorithm.Securities["VIX"].Price

            # Update the analyzer
            self.update(
                symbol=underlying_symbol,
                implied_volatility=avg_iv,
                underlying_price=underlying_price,
                timestamp=algorithm.Time,
                vix_level=vix_level,
            )


def create_enhanced_volatility_analyzer(
    lookback_days: int = 252,
    regime_window: int = 20,
) -> EnhancedVolatilityAnalyzer:
    """Create enhanced volatility analyzer instance."""
    return EnhancedVolatilityAnalyzer(
        lookback_days=lookback_days,
        regime_window=regime_window,
    )


__all__ = [
    "EnhancedVolatilityAnalyzer",
    "IVMetrics",
    "RealizedVolMetrics",
    "VolatilityPremium",
    "VolatilityRegime",
    "VolatilityRegimeAnalysis",
    "VolatilitySnapshot",
    "VolatilityTrend",
    "create_enhanced_volatility_analyzer",
]
