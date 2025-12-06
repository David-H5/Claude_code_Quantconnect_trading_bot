"""
Volatility Skew Module

UPGRADE-015 Phase 8: Options Analytics Engine

Provides volatility skew analysis:
- Put/Call skew measurement
- Risk reversal calculation
- Butterfly spread analysis
- Skew slope metrics

Features:
- Multiple skew metrics
- Delta-based analysis
- Strike-based analysis
- Historical comparison
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SkewType(Enum):
    """Types of volatility skew."""

    NORMAL = "normal"  # Puts > Calls (typical equity skew)
    REVERSE = "reverse"  # Calls > Puts (commodity skew)
    SMILE = "smile"  # Both OTM expensive
    FLAT = "flat"


@dataclass
class SkewPoint:
    """A single point for skew analysis."""

    strike: float
    delta: float  # Option delta (-1 to 1)
    iv: float
    moneyness: float
    option_type: str = "call"  # "call" or "put"


@dataclass
class SkewMetrics:
    """Comprehensive skew metrics."""

    # Risk Reversal: 25D Call IV - 25D Put IV
    risk_reversal_25d: float = 0.0
    risk_reversal_10d: float = 0.0

    # Butterfly: (25D Call + 25D Put) / 2 - ATM
    butterfly_25d: float = 0.0
    butterfly_10d: float = 0.0

    # Slope metrics
    put_skew_slope: float = 0.0  # OTM put IV slope
    call_skew_slope: float = 0.0  # OTM call IV slope

    # Key IV levels
    atm_iv: float = 0.0
    iv_25d_put: float = 0.0
    iv_25d_call: float = 0.0
    iv_10d_put: float = 0.0
    iv_10d_call: float = 0.0

    # Classification
    skew_type: SkewType = SkewType.FLAT
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SkewConfig:
    """Configuration for skew analysis."""

    atm_delta_range: tuple[float, float] = (0.45, 0.55)
    otm_put_delta: float = -0.25
    otm_call_delta: float = 0.25
    deep_otm_put_delta: float = -0.10
    deep_otm_call_delta: float = 0.10
    flat_threshold: float = 0.01  # 1% for flat classification


class VolatilitySkew:
    """Volatility Skew analyzer for options."""

    def __init__(
        self,
        underlying_price: float,
        expiry_days: int = 30,
        config: SkewConfig | None = None,
    ):
        """
        Initialize skew analyzer.

        Args:
            underlying_price: Current underlying price
            expiry_days: Days to expiration for this skew
            config: Configuration
        """
        self.underlying_price = underlying_price
        self.expiry_days = expiry_days
        self.config = config or SkewConfig()
        self._points: list[SkewPoint] = []
        self._last_updated = datetime.utcnow()

    # ==========================================================================
    # Data Management
    # ==========================================================================

    def add_point(
        self,
        strike: float,
        iv: float,
        delta: float,
        option_type: str = "call",
    ) -> None:
        """
        Add a skew data point.

        Args:
            strike: Strike price
            iv: Implied volatility
            delta: Option delta
            option_type: "call" or "put"
        """
        moneyness = strike / self.underlying_price

        # Normalize delta for puts to negative
        if option_type == "put" and delta > 0:
            delta = -delta

        point = SkewPoint(
            strike=strike,
            delta=delta,
            iv=iv,
            moneyness=moneyness,
            option_type=option_type,
        )

        self._points.append(point)
        self._last_updated = datetime.utcnow()

    def add_option_chain(
        self,
        options: list[dict[str, Any]],
    ) -> None:
        """
        Add data from option chain.

        Args:
            options: List of option dicts with strike, iv, delta, type
        """
        for opt in options:
            self.add_point(
                strike=opt["strike"],
                iv=opt["iv"],
                delta=opt.get("delta", 0.5),
                option_type=opt.get("type", "call"),
            )

    def clear(self) -> None:
        """Clear all data points."""
        self._points.clear()

    # ==========================================================================
    # IV Retrieval
    # ==========================================================================

    def get_iv_at_delta(self, target_delta: float) -> float | None:
        """
        Get IV at a specific delta level.

        Args:
            target_delta: Target delta (positive for calls, negative for puts)

        Returns:
            Interpolated IV
        """
        if not self._points:
            return None

        # Find closest points by delta
        sorted_points = sorted(
            self._points,
            key=lambda p: abs(p.delta - target_delta),
        )

        if not sorted_points:
            return None

        # If very close, return directly
        if abs(sorted_points[0].delta - target_delta) < 0.01:
            return sorted_points[0].iv

        # Interpolate between two closest
        if len(sorted_points) >= 2:
            p1, p2 = sorted_points[0], sorted_points[1]
            if p1.delta != p2.delta:
                weight = (target_delta - p1.delta) / (p2.delta - p1.delta)
                weight = max(0, min(1, weight))
                return p1.iv + weight * (p2.iv - p1.iv)

        return sorted_points[0].iv

    def get_iv_at_moneyness(self, moneyness: float) -> float | None:
        """
        Get IV at a specific moneyness level.

        Args:
            moneyness: Moneyness ratio (1.0 = ATM)

        Returns:
            Interpolated IV
        """
        if not self._points:
            return None

        sorted_points = sorted(
            self._points,
            key=lambda p: abs(p.moneyness - moneyness),
        )

        if not sorted_points:
            return None

        if abs(sorted_points[0].moneyness - moneyness) < 0.005:
            return sorted_points[0].iv

        if len(sorted_points) >= 2:
            p1, p2 = sorted_points[0], sorted_points[1]
            if p1.moneyness != p2.moneyness:
                weight = (moneyness - p1.moneyness) / (p2.moneyness - p1.moneyness)
                weight = max(0, min(1, weight))
                return p1.iv + weight * (p2.iv - p1.iv)

        return sorted_points[0].iv

    def get_atm_iv(self) -> float | None:
        """Get ATM implied volatility."""
        return self.get_iv_at_moneyness(1.0)

    # ==========================================================================
    # Skew Metrics
    # ==========================================================================

    def calculate_metrics(self) -> SkewMetrics:
        """
        Calculate comprehensive skew metrics.

        Returns:
            SkewMetrics dataclass
        """
        metrics = SkewMetrics(timestamp=self._last_updated)

        # ATM IV
        atm = self.get_atm_iv()
        if atm is not None:
            metrics.atm_iv = atm

        # 25 Delta IVs
        iv_25d_put = self.get_iv_at_delta(self.config.otm_put_delta)
        iv_25d_call = self.get_iv_at_delta(self.config.otm_call_delta)

        if iv_25d_put is not None:
            metrics.iv_25d_put = iv_25d_put
        if iv_25d_call is not None:
            metrics.iv_25d_call = iv_25d_call

        # 10 Delta IVs
        iv_10d_put = self.get_iv_at_delta(self.config.deep_otm_put_delta)
        iv_10d_call = self.get_iv_at_delta(self.config.deep_otm_call_delta)

        if iv_10d_put is not None:
            metrics.iv_10d_put = iv_10d_put
        if iv_10d_call is not None:
            metrics.iv_10d_call = iv_10d_call

        # Risk Reversals
        if iv_25d_put and iv_25d_call:
            metrics.risk_reversal_25d = iv_25d_call - iv_25d_put
        if iv_10d_put and iv_10d_call:
            metrics.risk_reversal_10d = iv_10d_call - iv_10d_put

        # Butterflies
        if atm and iv_25d_put and iv_25d_call:
            metrics.butterfly_25d = (iv_25d_put + iv_25d_call) / 2 - atm
        if atm and iv_10d_put and iv_10d_call:
            metrics.butterfly_10d = (iv_10d_put + iv_10d_call) / 2 - atm

        # Slopes
        if atm and iv_25d_put:
            metrics.put_skew_slope = (iv_25d_put - atm) / 0.25  # Per 25 delta
        if atm and iv_25d_call:
            metrics.call_skew_slope = (iv_25d_call - atm) / 0.25

        # Classification
        metrics.skew_type = self._classify_skew(metrics)

        return metrics

    def _classify_skew(self, metrics: SkewMetrics) -> SkewType:
        """Classify the skew shape."""
        rr = metrics.risk_reversal_25d
        bf = metrics.butterfly_25d

        if abs(rr) < self.config.flat_threshold:
            if bf > self.config.flat_threshold:
                return SkewType.SMILE
            return SkewType.FLAT

        if rr < 0:  # Puts more expensive
            return SkewType.NORMAL
        return SkewType.REVERSE

    # ==========================================================================
    # Analysis
    # ==========================================================================

    def get_put_call_ratio(self) -> float | None:
        """
        Get put/call IV ratio at 25 delta.

        Returns:
            Put IV / Call IV ratio
        """
        iv_put = self.get_iv_at_delta(self.config.otm_put_delta)
        iv_call = self.get_iv_at_delta(self.config.otm_call_delta)

        if iv_put and iv_call and iv_call > 0:
            return iv_put / iv_call
        return None

    def get_smile(
        self,
        num_points: int = 11,
    ) -> list[tuple[float, float]]:
        """
        Get the volatility smile as a curve.

        Args:
            num_points: Number of points in the smile

        Returns:
            List of (moneyness, IV) tuples
        """
        result = []
        for i in range(num_points):
            m = 0.85 + i * 0.03  # 0.85 to 1.15 moneyness
            iv = self.get_iv_at_moneyness(m)
            if iv is not None:
                result.append((m, iv))
        return result

    def get_delta_smile(
        self,
        deltas: list[float] | None = None,
    ) -> list[tuple[float, float]]:
        """
        Get volatility smile in delta space.

        Args:
            deltas: List of deltas to sample

        Returns:
            List of (delta, IV) tuples
        """
        if deltas is None:
            deltas = [-0.10, -0.25, -0.40, -0.50, 0.50, 0.40, 0.25, 0.10]

        result = []
        for d in deltas:
            iv = self.get_iv_at_delta(d)
            if iv is not None:
                result.append((d, iv))
        return result

    def estimate_tail_risk(self) -> dict[str, Any]:
        """
        Estimate tail risk from skew.

        Returns:
            Tail risk metrics
        """
        metrics = self.calculate_metrics()

        # Higher put skew suggests more downside fear
        put_tail_premium = 0.0
        if metrics.iv_10d_put and metrics.atm_iv:
            put_tail_premium = (metrics.iv_10d_put - metrics.atm_iv) / metrics.atm_iv

        call_tail_premium = 0.0
        if metrics.iv_10d_call and metrics.atm_iv:
            call_tail_premium = (metrics.iv_10d_call - metrics.atm_iv) / metrics.atm_iv

        return {
            "put_tail_premium_pct": put_tail_premium * 100,
            "call_tail_premium_pct": call_tail_premium * 100,
            "downside_fear": "high" if put_tail_premium > 0.3 else "normal",
            "upside_fear": "high" if call_tail_premium > 0.2 else "normal",
            "risk_reversal": metrics.risk_reversal_25d,
        }

    # ==========================================================================
    # Output
    # ==========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get skew statistics."""
        metrics = self.calculate_metrics()

        return {
            "underlying_price": self.underlying_price,
            "expiry_days": self.expiry_days,
            "num_points": len(self._points),
            "atm_iv": metrics.atm_iv,
            "risk_reversal_25d": metrics.risk_reversal_25d,
            "butterfly_25d": metrics.butterfly_25d,
            "skew_type": metrics.skew_type.value,
            "put_call_ratio": self.get_put_call_ratio(),
            "last_updated": self._last_updated.isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        metrics = self.calculate_metrics()

        return {
            "underlying_price": self.underlying_price,
            "expiry_days": self.expiry_days,
            "points": [
                {
                    "strike": p.strike,
                    "delta": p.delta,
                    "iv": p.iv,
                    "moneyness": p.moneyness,
                    "type": p.option_type,
                }
                for p in self._points
            ],
            "metrics": {
                "atm_iv": metrics.atm_iv,
                "iv_25d_put": metrics.iv_25d_put,
                "iv_25d_call": metrics.iv_25d_call,
                "risk_reversal_25d": metrics.risk_reversal_25d,
                "butterfly_25d": metrics.butterfly_25d,
                "skew_type": metrics.skew_type.value,
            },
            "last_updated": self._last_updated.isoformat(),
        }


def create_volatility_skew(
    underlying_price: float,
    expiry_days: int = 30,
) -> VolatilitySkew:
    """
    Factory function to create a volatility skew analyzer.

    Args:
        underlying_price: Current underlying price
        expiry_days: Days to expiration

    Returns:
        Configured VolatilitySkew
    """
    return VolatilitySkew(underlying_price, expiry_days)
