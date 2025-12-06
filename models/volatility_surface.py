"""
Volatility Surface Analysis Module

Analyzes implied volatility across strikes and expirations to:
- Detect volatility smile/skew
- Identify mispriced options
- Track volatility term structure
- Calculate volatility percentiles

Based on patterns from py_vollib, QuantLib, and academic research.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class VolatilityPoint:
    """Single point on the volatility surface."""

    strike: float
    expiry: datetime
    implied_volatility: float
    moneyness: float  # strike / spot
    days_to_expiry: int
    option_type: str  # "call" or "put"
    volume: int = 0
    open_interest: int = 0

    @property
    def log_moneyness(self) -> float:
        """Log moneyness (ln(K/S))."""
        if self.moneyness > 0:
            return math.log(self.moneyness)
        return 0.0


@dataclass
class VolatilitySlice:
    """Volatility across strikes for a single expiration."""

    expiry: datetime
    days_to_expiry: int
    points: list[VolatilityPoint] = field(default_factory=list)
    atm_volatility: float = 0.0
    skew: float = 0.0  # 25-delta put IV - 25-delta call IV
    smile_curvature: float = 0.0

    def calculate_metrics(self, spot_price: float) -> None:
        """Calculate slice metrics."""
        if not self.points:
            return

        # Find ATM volatility (closest to spot)
        atm_point = min(self.points, key=lambda p: abs(p.strike - spot_price))
        self.atm_volatility = atm_point.implied_volatility

        # Calculate skew (requires delta information or approximation)
        # Simplified: use 5% OTM puts vs 5% OTM calls
        otm_puts = [p for p in self.points if p.option_type == "put" and p.moneyness < 0.95]
        otm_calls = [p for p in self.points if p.option_type == "call" and p.moneyness > 1.05]

        if otm_puts and otm_calls:
            avg_put_iv = sum(p.implied_volatility for p in otm_puts) / len(otm_puts)
            avg_call_iv = sum(p.implied_volatility for p in otm_calls) / len(otm_calls)
            self.skew = avg_put_iv - avg_call_iv

        # Calculate smile curvature (wings vs ATM)
        if len(self.points) >= 3:
            ivs = [p.implied_volatility for p in sorted(self.points, key=lambda x: x.strike)]
            if len(ivs) >= 3:
                # Average of wings minus center
                wings_avg = (ivs[0] + ivs[-1]) / 2
                center_avg = sum(ivs[len(ivs) // 3 : 2 * len(ivs) // 3]) / (len(ivs) // 3 + 1)
                self.smile_curvature = wings_avg - center_avg


@dataclass
class TermStructure:
    """Volatility term structure across expirations."""

    spot_price: float
    timestamp: datetime
    slices: list[VolatilitySlice] = field(default_factory=list)

    def get_atm_term_structure(self) -> list[tuple[int, float]]:
        """Get ATM volatility by days to expiry."""
        return [(s.days_to_expiry, s.atm_volatility) for s in self.slices if s.atm_volatility > 0]

    def is_contango(self) -> bool:
        """Check if term structure is in contango (longer-dated higher IV)."""
        term = self.get_atm_term_structure()
        if len(term) < 2:
            return False

        sorted_term = sorted(term, key=lambda x: x[0])
        return sorted_term[-1][1] > sorted_term[0][1]

    def is_backwardation(self) -> bool:
        """Check if term structure is in backwardation (shorter-dated higher IV)."""
        term = self.get_atm_term_structure()
        if len(term) < 2:
            return False

        sorted_term = sorted(term, key=lambda x: x[0])
        return sorted_term[-1][1] < sorted_term[0][1]


class VolatilitySurface:
    """
    Volatility surface analyzer.

    Constructs and analyzes the implied volatility surface
    from option chain data.
    """

    def __init__(self, spot_price: float, risk_free_rate: float = 0.05):
        """
        Initialize volatility surface.

        Args:
            spot_price: Current underlying price
            risk_free_rate: Risk-free interest rate
        """
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.points: list[VolatilityPoint] = []
        self.slices: dict[datetime, VolatilitySlice] = {}
        self.iv_history: list[float] = []  # For percentile calculation

    def add_option(
        self,
        strike: float,
        expiry: datetime,
        implied_volatility: float,
        option_type: str,
        volume: int = 0,
        open_interest: int = 0,
    ) -> None:
        """Add an option to the surface."""
        moneyness = strike / self.spot_price
        days_to_expiry = (expiry - datetime.now()).days

        point = VolatilityPoint(
            strike=strike,
            expiry=expiry,
            implied_volatility=implied_volatility,
            moneyness=moneyness,
            days_to_expiry=days_to_expiry,
            option_type=option_type,
            volume=volume,
            open_interest=open_interest,
        )
        self.points.append(point)

        # Add to slice
        if expiry not in self.slices:
            self.slices[expiry] = VolatilitySlice(expiry=expiry, days_to_expiry=days_to_expiry)
        self.slices[expiry].points.append(point)

    def build_from_chain(self, option_chain: list[dict[str, Any]]) -> None:
        """
        Build surface from option chain data.

        Expected chain format:
        [{'strike': 100, 'expiry': datetime, 'iv': 0.25, 'type': 'call', ...}, ...]
        """
        for contract in option_chain:
            self.add_option(
                strike=contract.get("strike", 0),
                expiry=contract.get("expiry", datetime.now()),
                implied_volatility=contract.get("iv", contract.get("implied_volatility", 0)),
                option_type=contract.get("type", contract.get("option_type", "call")),
                volume=contract.get("volume", 0),
                open_interest=contract.get("open_interest", 0),
            )

        # Calculate metrics for each slice
        for slice_ in self.slices.values():
            slice_.calculate_metrics(self.spot_price)

    def get_atm_volatility(self, expiry: datetime | None = None) -> float:
        """
        Get ATM implied volatility.

        Args:
            expiry: Specific expiry (None for nearest)

        Returns:
            ATM implied volatility
        """
        if expiry and expiry in self.slices:
            return self.slices[expiry].atm_volatility

        # Find nearest expiry
        if not self.slices:
            return 0.0

        nearest = min(self.slices.keys(), key=lambda e: abs((e - datetime.now()).days))
        return self.slices[nearest].atm_volatility

    def get_skew(self, expiry: datetime | None = None) -> float:
        """Get volatility skew for expiry."""
        if expiry and expiry in self.slices:
            return self.slices[expiry].skew

        if not self.slices:
            return 0.0

        nearest = min(self.slices.keys(), key=lambda e: abs((e - datetime.now()).days))
        return self.slices[nearest].skew

    def get_smile_curvature(self, expiry: datetime | None = None) -> float:
        """Get smile curvature for expiry."""
        if expiry and expiry in self.slices:
            return self.slices[expiry].smile_curvature

        if not self.slices:
            return 0.0

        nearest = min(self.slices.keys(), key=lambda e: abs((e - datetime.now()).days))
        return self.slices[nearest].smile_curvature

    def get_term_structure(self) -> TermStructure:
        """Get volatility term structure."""
        return TermStructure(
            spot_price=self.spot_price,
            timestamp=datetime.now(),
            slices=list(self.slices.values()),
        )

    def find_mispriced_options(
        self,
        threshold: float = 0.10,
    ) -> list[dict[str, Any]]:
        """
        Find potentially mispriced options based on volatility surface.

        Options with IV significantly different from interpolated surface
        value may be mispriced.

        Args:
            threshold: Minimum deviation from surface (as fraction)

        Returns:
            List of potentially mispriced options
        """
        mispriced = []

        for expiry, slice_ in self.slices.items():
            if len(slice_.points) < 3:
                continue

            # Simple interpolation: linear between points
            sorted_points = sorted(slice_.points, key=lambda p: p.strike)
            strikes = [p.strike for p in sorted_points]
            ivs = [p.implied_volatility for p in sorted_points]

            for i, point in enumerate(sorted_points):
                # Estimate expected IV from neighbors
                if i == 0:
                    expected_iv = ivs[1] if len(ivs) > 1 else ivs[0]
                elif i == len(sorted_points) - 1:
                    expected_iv = ivs[-2] if len(ivs) > 1 else ivs[-1]
                else:
                    # Linear interpolation
                    expected_iv = (ivs[i - 1] + ivs[i + 1]) / 2

                deviation = (point.implied_volatility - expected_iv) / expected_iv

                if abs(deviation) > threshold:
                    mispriced.append(
                        {
                            "strike": point.strike,
                            "expiry": point.expiry,
                            "type": point.option_type,
                            "actual_iv": point.implied_volatility,
                            "expected_iv": expected_iv,
                            "deviation": deviation,
                            "underpriced": deviation < 0,  # Lower IV = potentially cheap
                            "volume": point.volume,
                            "open_interest": point.open_interest,
                        }
                    )

        return sorted(mispriced, key=lambda x: abs(x["deviation"]), reverse=True)

    def calculate_iv_percentile(
        self,
        current_iv: float,
        lookback_days: int = 252,
    ) -> float:
        """
        Calculate IV percentile based on historical data.

        Args:
            current_iv: Current implied volatility
            lookback_days: Days of history to consider

        Returns:
            IV percentile (0-100)
        """
        if len(self.iv_history) < 20:
            return 50.0  # Default if insufficient data

        history = self.iv_history[-lookback_days:]
        count_below = sum(1 for iv in history if iv < current_iv)

        return (count_below / len(history)) * 100

    def update_iv_history(self, atm_iv: float) -> None:
        """Update historical IV for percentile calculation."""
        self.iv_history.append(atm_iv)

        # Keep last 252 trading days
        if len(self.iv_history) > 252:
            self.iv_history = self.iv_history[-252:]

    def get_surface_data(self) -> dict[str, Any]:
        """Get surface data for visualization or export."""
        data = {
            "spot_price": self.spot_price,
            "timestamp": datetime.now().isoformat(),
            "atm_iv": self.get_atm_volatility(),
            "skew": self.get_skew(),
            "smile_curvature": self.get_smile_curvature(),
            "slices": [],
        }

        for expiry, slice_ in sorted(self.slices.items()):
            slice_data = {
                "expiry": expiry.isoformat(),
                "days_to_expiry": slice_.days_to_expiry,
                "atm_iv": slice_.atm_volatility,
                "skew": slice_.skew,
                "points": [
                    {
                        "strike": p.strike,
                        "iv": p.implied_volatility,
                        "moneyness": p.moneyness,
                        "type": p.option_type,
                    }
                    for p in slice_.points
                ],
            }
            data["slices"].append(slice_data)

        return data


class VolatilityAnalyzer:
    """
    High-level volatility analysis utilities.

    Provides signals based on volatility patterns.
    """

    def __init__(self, surface: VolatilitySurface):
        """Initialize analyzer with surface."""
        self.surface = surface

    def is_volatility_elevated(self, percentile_threshold: float = 70) -> bool:
        """Check if current volatility is elevated."""
        atm_iv = self.surface.get_atm_volatility()
        percentile = self.surface.calculate_iv_percentile(atm_iv)
        return percentile > percentile_threshold

    def is_volatility_compressed(self, percentile_threshold: float = 30) -> bool:
        """Check if current volatility is compressed (low)."""
        atm_iv = self.surface.get_atm_volatility()
        percentile = self.surface.calculate_iv_percentile(atm_iv)
        return percentile < percentile_threshold

    def get_skew_signal(self) -> str:
        """
        Get trading signal based on skew.

        Positive skew (puts expensive) often indicates fear/hedging demand.
        """
        skew = self.surface.get_skew()

        if skew > 0.05:
            return "elevated_put_demand"  # Market fear, consider protective puts
        elif skew < -0.03:
            return "elevated_call_demand"  # Bullish speculation
        else:
            return "neutral"

    def get_term_structure_signal(self) -> str:
        """Get signal based on term structure."""
        term = self.surface.get_term_structure()

        if term.is_backwardation():
            return "backwardation"  # Short-term stress, consider selling near-term vol
        elif term.is_contango():
            return "contango"  # Normal, consider calendar spreads
        else:
            return "flat"

    def get_smile_signal(self) -> str:
        """Get signal based on smile shape."""
        curvature = self.surface.get_smile_curvature()

        if curvature > 0.02:
            return "pronounced_smile"  # Wings expensive, consider selling strangles
        elif curvature < -0.01:
            return "inverted_smile"  # Unusual, investigate
        else:
            return "normal"

    def get_composite_signal(self) -> dict[str, Any]:
        """Get composite volatility signal."""
        atm_iv = self.surface.get_atm_volatility()
        percentile = self.surface.calculate_iv_percentile(atm_iv)

        return {
            "atm_iv": atm_iv,
            "iv_percentile": percentile,
            "volatility_level": "high" if percentile > 70 else "low" if percentile < 30 else "normal",
            "skew_signal": self.get_skew_signal(),
            "term_structure": self.get_term_structure_signal(),
            "smile_signal": self.get_smile_signal(),
            "mispriced_count": len(self.surface.find_mispriced_options()),
        }


def create_volatility_surface(
    spot_price: float,
    option_chain: list[dict[str, Any]],
    risk_free_rate: float = 0.05,
) -> VolatilitySurface:
    """
    Create and populate a volatility surface from option chain.

    Args:
        spot_price: Current underlying price
        option_chain: List of option contract dicts
        risk_free_rate: Risk-free rate

    Returns:
        Populated VolatilitySurface
    """
    surface = VolatilitySurface(spot_price, risk_free_rate)
    surface.build_from_chain(option_chain)
    return surface


__all__ = [
    "TermStructure",
    "VolatilityAnalyzer",
    "VolatilityPoint",
    "VolatilitySlice",
    "VolatilitySurface",
    "create_volatility_surface",
]
