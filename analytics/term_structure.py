"""
Term Structure Module

UPGRADE-015 Phase 8: Options Analytics Engine

Provides term structure analysis for implied volatility:
- ATM IV term structure
- Forward volatility calculation
- Term structure shape classification
- Contango/backwardation detection

Features:
- Multiple maturity support
- Forward IV computation
- Shape metrics (slope, curvature)
- Historical comparison
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TermStructureShape(Enum):
    """Term structure shape classification."""

    CONTANGO = "contango"  # Upward sloping (normal)
    BACKWARDATION = "backwardation"  # Downward sloping (inverted)
    FLAT = "flat"
    HUMPED = "humped"  # Peak in the middle
    UNCERTAIN = "uncertain"


@dataclass
class TermPoint:
    """A single point on the term structure."""

    expiry_days: int
    iv: float
    forward_iv: float | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def annualized_dte(self) -> float:
        """Get DTE as fraction of year."""
        return self.expiry_days / 365.0


@dataclass
class TermStructureConfig:
    """Configuration for term structure analysis."""

    standard_tenors: list[int] = field(default_factory=lambda: [7, 14, 21, 30, 45, 60, 90, 120, 180, 365])
    flat_threshold: float = 0.02  # 2% difference for flat classification
    min_points: int = 3


class TermStructure:
    """ATM Implied Volatility Term Structure analyzer."""

    def __init__(
        self,
        underlying: str = "",
        config: TermStructureConfig | None = None,
    ):
        """
        Initialize term structure analyzer.

        Args:
            underlying: Underlying symbol
            config: Configuration
        """
        self.underlying = underlying
        self.config = config or TermStructureConfig()
        self._points: list[TermPoint] = []
        self._last_updated = datetime.utcnow()

    # ==========================================================================
    # Data Management
    # ==========================================================================

    def add_point(
        self,
        expiry_days: int,
        iv: float,
    ) -> None:
        """
        Add a term structure point.

        Args:
            expiry_days: Days to expiration
            iv: ATM implied volatility
        """
        point = TermPoint(expiry_days=expiry_days, iv=iv)
        self._points.append(point)
        self._points.sort(key=lambda p: p.expiry_days)
        self._calculate_forwards()
        self._last_updated = datetime.utcnow()

    def add_points(
        self,
        points: list[tuple[int, float]],
    ) -> None:
        """
        Add multiple points.

        Args:
            points: List of (expiry_days, iv) tuples
        """
        for dte, iv in points:
            self._points.append(TermPoint(expiry_days=dte, iv=iv))

        self._points.sort(key=lambda p: p.expiry_days)
        self._calculate_forwards()
        self._last_updated = datetime.utcnow()

    def clear(self) -> None:
        """Clear all points."""
        self._points.clear()

    # ==========================================================================
    # Forward Volatility
    # ==========================================================================

    def _calculate_forwards(self) -> None:
        """Calculate forward volatilities between adjacent points."""
        if len(self._points) < 2:
            return

        for i in range(1, len(self._points)):
            prev = self._points[i - 1]
            curr = self._points[i]

            # Forward variance: var_fwd = (var2*T2 - var1*T1) / (T2 - T1)
            t1 = prev.annualized_dte
            t2 = curr.annualized_dte

            var1 = prev.iv**2 * t1
            var2 = curr.iv**2 * t2

            if t2 > t1:
                forward_var = (var2 - var1) / (t2 - t1)
                if forward_var > 0:
                    curr.forward_iv = math.sqrt(forward_var)
                else:
                    # Negative forward variance indicates calendar arbitrage
                    curr.forward_iv = None

    def get_forward_iv(
        self,
        start_dte: int,
        end_dte: int,
    ) -> float | None:
        """
        Calculate forward IV between two dates.

        Args:
            start_dte: Start DTE
            end_dte: End DTE

        Returns:
            Forward IV or None if cannot calculate
        """
        start_iv = self.get_iv(start_dte)
        end_iv = self.get_iv(end_dte)

        if start_iv is None or end_iv is None:
            return None

        t1 = start_dte / 365.0
        t2 = end_dte / 365.0

        if t2 <= t1:
            return None

        var1 = start_iv**2 * t1
        var2 = end_iv**2 * t2

        forward_var = (var2 - var1) / (t2 - t1)

        if forward_var > 0:
            return math.sqrt(forward_var)
        return None

    # ==========================================================================
    # Interpolation
    # ==========================================================================

    def get_iv(self, expiry_days: int) -> float | None:
        """
        Get IV for a given expiry (with interpolation).

        Args:
            expiry_days: Days to expiration

        Returns:
            Interpolated IV
        """
        if not self._points:
            return None

        # Exact match
        for p in self._points:
            if p.expiry_days == expiry_days:
                return p.iv

        # Interpolate
        return self._interpolate(expiry_days)

    def _interpolate(self, target_dte: int) -> float | None:
        """Linear interpolation in variance space."""
        if len(self._points) < 2:
            return self._points[0].iv if self._points else None

        # Find surrounding points
        lower = None
        upper = None

        for p in self._points:
            if p.expiry_days <= target_dte:
                lower = p
            elif upper is None:
                upper = p
                break

        # Extrapolation
        if lower is None:
            return self._points[0].iv
        if upper is None:
            return self._points[-1].iv

        # Linear interpolation in variance space
        t1 = lower.annualized_dte
        t2 = upper.annualized_dte
        t = target_dte / 365.0

        var1 = lower.iv**2 * t1
        var2 = upper.iv**2 * t2

        # Interpolate total variance
        total_var = var1 + (var2 - var1) * (t - t1) / (t2 - t1)

        if total_var > 0 and t > 0:
            return math.sqrt(total_var / t)
        return None

    # ==========================================================================
    # Shape Analysis
    # ==========================================================================

    def get_shape(self) -> TermStructureShape:
        """
        Classify the term structure shape.

        Returns:
            Shape classification
        """
        if len(self._points) < self.config.min_points:
            return TermStructureShape.UNCERTAIN

        ivs = [p.iv for p in self._points]

        # Calculate slope
        slope = (ivs[-1] - ivs[0]) / ivs[0] if ivs[0] > 0 else 0

        # Check for flat
        if abs(slope) < self.config.flat_threshold:
            return TermStructureShape.FLAT

        # Check for hump
        mid_idx = len(ivs) // 2
        if ivs[mid_idx] > max(ivs[0], ivs[-1]):
            return TermStructureShape.HUMPED

        # Contango or backwardation
        if slope > 0:
            return TermStructureShape.CONTANGO
        return TermStructureShape.BACKWARDATION

    def get_slope(self) -> float | None:
        """
        Get term structure slope (annualized).

        Returns:
            Slope in IV points per year
        """
        if len(self._points) < 2:
            return None

        first = self._points[0]
        last = self._points[-1]

        dte_diff = (last.expiry_days - first.expiry_days) / 365.0

        if dte_diff <= 0:
            return None

        return (last.iv - first.iv) / dte_diff

    def get_curvature(self) -> float | None:
        """
        Get term structure curvature.

        Returns:
            Curvature metric (positive = convex, negative = concave)
        """
        if len(self._points) < 3:
            return None

        # Use first, mid, and last points
        first = self._points[0]
        mid = self._points[len(self._points) // 2]
        last = self._points[-1]

        # Expected mid IV if linear
        t1 = first.annualized_dte
        t2 = mid.annualized_dte
        t3 = last.annualized_dte

        if t3 <= t1:
            return None

        expected_mid = first.iv + (last.iv - first.iv) * (t2 - t1) / (t3 - t1)

        # Curvature as deviation from linear
        return mid.iv - expected_mid

    # ==========================================================================
    # Analytics
    # ==========================================================================

    def get_vix_equivalent(self) -> float | None:
        """
        Get VIX-equivalent 30-day IV (interpolated).

        Returns:
            30-day ATM IV
        """
        return self.get_iv(30)

    def get_volatility_risk_premium(
        self,
        realized_vol: float,
    ) -> dict[str, float]:
        """
        Calculate volatility risk premium across term structure.

        Args:
            realized_vol: Historical realized volatility

        Returns:
            VRP for different tenors
        """
        result = {}
        for p in self._points:
            vrp = p.iv - realized_vol
            result[f"{p.expiry_days}d"] = vrp
        return result

    def detect_calendar_arbitrage(self) -> list[dict[str, Any]]:
        """
        Detect calendar spread arbitrage opportunities.

        Returns:
            List of arbitrage signals
        """
        arbitrage = []

        for i in range(1, len(self._points)):
            prev = self._points[i - 1]
            curr = self._points[i]

            # Forward IV should be positive
            if curr.forward_iv is None:
                arbitrage.append(
                    {
                        "type": "negative_forward_variance",
                        "short_dte": prev.expiry_days,
                        "long_dte": curr.expiry_days,
                        "short_iv": prev.iv,
                        "long_iv": curr.iv,
                    }
                )

        return arbitrage

    # ==========================================================================
    # Output
    # ==========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get term structure statistics."""
        if not self._points:
            return {"num_points": 0}

        ivs = [p.iv for p in self._points]

        return {
            "underlying": self.underlying,
            "num_points": len(self._points),
            "min_iv": min(ivs),
            "max_iv": max(ivs),
            "avg_iv": sum(ivs) / len(ivs),
            "shape": self.get_shape().value,
            "slope": self.get_slope(),
            "curvature": self.get_curvature(),
            "vix_30d": self.get_vix_equivalent(),
            "min_dte": self._points[0].expiry_days,
            "max_dte": self._points[-1].expiry_days,
            "last_updated": self._last_updated.isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "underlying": self.underlying,
            "points": [
                {
                    "expiry_days": p.expiry_days,
                    "iv": p.iv,
                    "forward_iv": p.forward_iv,
                }
                for p in self._points
            ],
            "shape": self.get_shape().value,
            "last_updated": self._last_updated.isoformat(),
        }

    def to_curve(self) -> list[tuple[int, float]]:
        """Get as list of (DTE, IV) tuples."""
        return [(p.expiry_days, p.iv) for p in self._points]


def create_term_structure(
    underlying: str = "",
    points: list[tuple[int, float]] | None = None,
) -> TermStructure:
    """
    Factory function to create a term structure.

    Args:
        underlying: Underlying symbol
        points: Optional initial points as (dte, iv) tuples

    Returns:
        Configured TermStructure
    """
    ts = TermStructure(underlying)
    if points:
        ts.add_points(points)
    return ts
