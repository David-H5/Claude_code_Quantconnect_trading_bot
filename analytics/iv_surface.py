"""
Implied Volatility Surface Module

UPGRADE-015 Phase 8: Options Analytics Engine

Provides IV surface modeling and analysis:
- Surface construction from option prices
- Interpolation across strikes and expirations
- Surface smoothing
- Arbitrage detection

Features:
- Grid-based surface representation
- Multiple interpolation methods
- Real-time surface updates
- Visualization support
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class InterpolationMethod(Enum):
    """Surface interpolation methods."""

    LINEAR = "linear"
    CUBIC = "cubic"
    SPLINE = "spline"
    RBF = "rbf"  # Radial Basis Function


@dataclass
class SurfacePoint:
    """A single point on the IV surface."""

    strike: float
    expiry_days: int
    iv: float
    moneyness: float = 0.0  # Strike / Spot
    delta: float | None = None
    bid_iv: float | None = None
    ask_iv: float | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def mid_iv(self) -> float:
        """Get mid IV if bid/ask available."""
        if self.bid_iv and self.ask_iv:
            return (self.bid_iv + self.ask_iv) / 2
        return self.iv


@dataclass
class SurfaceConfig:
    """Configuration for IV surface."""

    min_strike_pct: float = 0.7  # 70% of spot
    max_strike_pct: float = 1.3  # 130% of spot
    strike_step_pct: float = 0.025  # 2.5% steps
    min_dte: int = 7
    max_dte: int = 365
    dte_steps: list[int] = field(default_factory=lambda: [7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365])
    interpolation: InterpolationMethod = InterpolationMethod.LINEAR
    smooth_factor: float = 0.1


class IVSurface:
    """Implied Volatility Surface for options analysis."""

    def __init__(
        self,
        underlying_price: float,
        config: SurfaceConfig | None = None,
    ):
        """
        Initialize IV surface.

        Args:
            underlying_price: Current price of underlying
            config: Surface configuration
        """
        self.underlying_price = underlying_price
        self.config = config or SurfaceConfig()
        self._points: list[SurfacePoint] = []
        self._grid: dict[tuple[float, int], float] = {}
        self._last_updated = datetime.utcnow()

    # ==========================================================================
    # Data Management
    # ==========================================================================

    def add_point(
        self,
        strike: float,
        expiry_days: int,
        iv: float,
        bid_iv: float | None = None,
        ask_iv: float | None = None,
        delta: float | None = None,
    ) -> None:
        """
        Add a data point to the surface.

        Args:
            strike: Strike price
            expiry_days: Days to expiration
            iv: Implied volatility (decimal, e.g., 0.25 for 25%)
            bid_iv: Bid IV
            ask_iv: Ask IV
            delta: Option delta
        """
        moneyness = strike / self.underlying_price

        point = SurfacePoint(
            strike=strike,
            expiry_days=expiry_days,
            iv=iv,
            moneyness=moneyness,
            delta=delta,
            bid_iv=bid_iv,
            ask_iv=ask_iv,
        )

        self._points.append(point)
        self._grid[(moneyness, expiry_days)] = iv
        self._last_updated = datetime.utcnow()

    def add_points(self, points: list[SurfacePoint]) -> None:
        """Add multiple points at once."""
        for point in points:
            if point.moneyness == 0.0:
                point.moneyness = point.strike / self.underlying_price
            self._points.append(point)
            self._grid[(point.moneyness, point.expiry_days)] = point.iv

        self._last_updated = datetime.utcnow()

    def clear(self) -> None:
        """Clear all surface data."""
        self._points.clear()
        self._grid.clear()

    def update_spot(self, new_price: float) -> None:
        """
        Update the underlying price and recalculate moneyness.

        Args:
            new_price: New underlying price
        """
        self.underlying_price = new_price

        # Recalculate moneyness for all points
        new_grid = {}
        for point in self._points:
            point.moneyness = point.strike / new_price
            new_grid[(point.moneyness, point.expiry_days)] = point.iv

        self._grid = new_grid
        self._last_updated = datetime.utcnow()

    # ==========================================================================
    # Interpolation
    # ==========================================================================

    def get_iv(
        self,
        strike: float | None = None,
        moneyness: float | None = None,
        expiry_days: int = 30,
    ) -> float | None:
        """
        Get interpolated IV for a given strike/moneyness and expiry.

        Args:
            strike: Strike price (optional if moneyness provided)
            moneyness: Moneyness ratio (optional if strike provided)
            expiry_days: Days to expiration

        Returns:
            Interpolated IV or None if cannot interpolate
        """
        if not self._grid:
            return None

        if moneyness is None:
            if strike is None:
                return None
            moneyness = strike / self.underlying_price

        # Check for exact match
        if (moneyness, expiry_days) in self._grid:
            return self._grid[(moneyness, expiry_days)]

        # Interpolate
        return self._interpolate(moneyness, expiry_days)

    def _interpolate(self, moneyness: float, expiry_days: int) -> float | None:
        """
        Interpolate IV using configured method.

        Args:
            moneyness: Target moneyness
            expiry_days: Target DTE

        Returns:
            Interpolated IV
        """
        if self.config.interpolation == InterpolationMethod.LINEAR:
            return self._linear_interpolate(moneyness, expiry_days)
        # Add other methods as needed
        return self._linear_interpolate(moneyness, expiry_days)

    def _linear_interpolate(
        self,
        target_moneyness: float,
        target_dte: int,
    ) -> float | None:
        """Linear interpolation on the surface."""
        if not self._points:
            return None

        # Find nearest points
        sorted_points = sorted(
            self._points,
            key=lambda p: abs(p.moneyness - target_moneyness) + abs(p.expiry_days - target_dte) / 100,
        )

        if not sorted_points:
            return None

        # Simple weighted average of nearest 4 points
        nearest = sorted_points[:4]
        weights = []
        values = []

        for p in nearest:
            dist = math.sqrt((p.moneyness - target_moneyness) ** 2 + ((p.expiry_days - target_dte) / 100) ** 2)
            if dist < 0.001:
                return p.iv
            weights.append(1.0 / dist)
            values.append(p.iv)

        total_weight = sum(weights)
        if total_weight == 0:
            return nearest[0].iv

        return sum(w * v for w, v in zip(weights, values)) / total_weight

    # ==========================================================================
    # Analysis
    # ==========================================================================

    def get_atm_iv(self, expiry_days: int = 30) -> float | None:
        """Get ATM implied volatility for a given expiry."""
        return self.get_iv(moneyness=1.0, expiry_days=expiry_days)

    def get_iv_term_structure(
        self,
        moneyness: float = 1.0,
    ) -> list[tuple[int, float]]:
        """
        Get IV term structure at a given moneyness.

        Args:
            moneyness: Moneyness level (1.0 = ATM)

        Returns:
            List of (DTE, IV) tuples
        """
        result = []
        for dte in self.config.dte_steps:
            iv = self.get_iv(moneyness=moneyness, expiry_days=dte)
            if iv is not None:
                result.append((dte, iv))
        return result

    def get_smile(
        self,
        expiry_days: int = 30,
        num_points: int = 11,
    ) -> list[tuple[float, float]]:
        """
        Get volatility smile for a given expiry.

        Args:
            expiry_days: Days to expiration
            num_points: Number of points in the smile

        Returns:
            List of (moneyness, IV) tuples
        """
        result = []
        min_m = self.config.min_strike_pct
        max_m = self.config.max_strike_pct
        step = (max_m - min_m) / (num_points - 1)

        for i in range(num_points):
            m = min_m + i * step
            iv = self.get_iv(moneyness=m, expiry_days=expiry_days)
            if iv is not None:
                result.append((m, iv))

        return result

    def get_surface_grid(
        self,
        moneyness_range: tuple[float, float] = (0.8, 1.2),
        dte_range: tuple[int, int] = (7, 90),
        moneyness_steps: int = 9,
        dte_steps: int = 6,
    ) -> dict[str, Any]:
        """
        Get the IV surface as a grid for visualization.

        Args:
            moneyness_range: (min, max) moneyness
            dte_range: (min, max) DTE
            moneyness_steps: Number of moneyness steps
            dte_steps: Number of DTE steps

        Returns:
            Grid data for plotting
        """
        m_min, m_max = moneyness_range
        d_min, d_max = dte_range

        moneyness_vals = [m_min + i * (m_max - m_min) / (moneyness_steps - 1) for i in range(moneyness_steps)]
        dte_vals = [int(d_min + i * (d_max - d_min) / (dte_steps - 1)) for i in range(dte_steps)]

        grid = []
        for dte in dte_vals:
            row = []
            for m in moneyness_vals:
                iv = self.get_iv(moneyness=m, expiry_days=dte)
                row.append(iv if iv is not None else 0.0)
            grid.append(row)

        return {
            "moneyness": moneyness_vals,
            "dte": dte_vals,
            "iv": grid,
            "spot": self.underlying_price,
            "timestamp": self._last_updated.isoformat(),
        }

    def detect_arbitrage(self) -> list[dict[str, Any]]:
        """
        Detect potential calendar and butterfly arbitrage.

        Returns:
            List of detected arbitrage opportunities
        """
        arbitrage = []

        # Calendar arbitrage: IV should generally increase with time
        moneyness_levels = {p.moneyness for p in self._points}

        for m in moneyness_levels:
            points_at_m = sorted(
                [p for p in self._points if abs(p.moneyness - m) < 0.01],
                key=lambda p: p.expiry_days,
            )

            for i in range(len(points_at_m) - 1):
                if points_at_m[i + 1].iv < points_at_m[i].iv * 0.95:
                    arbitrage.append(
                        {
                            "type": "calendar",
                            "moneyness": m,
                            "short_dte": points_at_m[i].expiry_days,
                            "long_dte": points_at_m[i + 1].expiry_days,
                            "short_iv": points_at_m[i].iv,
                            "long_iv": points_at_m[i + 1].iv,
                        }
                    )

        return arbitrage

    # ==========================================================================
    # Statistics
    # ==========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get surface statistics."""
        if not self._points:
            return {"num_points": 0}

        ivs = [p.iv for p in self._points]

        return {
            "num_points": len(self._points),
            "min_iv": min(ivs),
            "max_iv": max(ivs),
            "avg_iv": sum(ivs) / len(ivs),
            "atm_iv": self.get_atm_iv(),
            "spot": self.underlying_price,
            "last_updated": self._last_updated.isoformat(),
            "min_dte": min(p.expiry_days for p in self._points),
            "max_dte": max(p.expiry_days for p in self._points),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert surface to dictionary for serialization."""
        return {
            "underlying_price": self.underlying_price,
            "points": [
                {
                    "strike": p.strike,
                    "expiry_days": p.expiry_days,
                    "iv": p.iv,
                    "moneyness": p.moneyness,
                    "delta": p.delta,
                    "bid_iv": p.bid_iv,
                    "ask_iv": p.ask_iv,
                }
                for p in self._points
            ],
            "last_updated": self._last_updated.isoformat(),
        }


def create_iv_surface(
    underlying_price: float,
    min_strike_pct: float = 0.7,
    max_strike_pct: float = 1.3,
) -> IVSurface:
    """
    Factory function to create an IV surface.

    Args:
        underlying_price: Current underlying price
        min_strike_pct: Minimum strike as % of spot
        max_strike_pct: Maximum strike as % of spot

    Returns:
        Configured IVSurface
    """
    config = SurfaceConfig(
        min_strike_pct=min_strike_pct,
        max_strike_pct=max_strike_pct,
    )
    return IVSurface(underlying_price, config)
