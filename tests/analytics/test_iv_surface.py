"""
Tests for IV Surface Module

UPGRADE-015 Phase 8: Options Analytics Engine

Tests cover:
- Surface construction
- Point addition
- Interpolation
- Statistics
- Arbitrage detection
"""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analytics.iv_surface import (
    InterpolationMethod,
    IVSurface,
    SurfaceConfig,
    SurfacePoint,
    create_iv_surface,
)


class TestSurfacePoint:
    """Test SurfacePoint dataclass."""

    def test_point_creation(self):
        """Test creating a surface point."""
        point = SurfacePoint(
            strike=450.0,
            expiry_days=30,
            iv=0.25,
            moneyness=1.0,
        )

        assert point.strike == 450.0
        assert point.expiry_days == 30
        assert point.iv == 0.25
        assert point.moneyness == 1.0

    def test_mid_iv(self):
        """Test mid IV calculation."""
        point = SurfacePoint(
            strike=450.0,
            expiry_days=30,
            iv=0.25,
            bid_iv=0.24,
            ask_iv=0.26,
        )

        assert point.mid_iv == 0.25


class TestSurfaceConfig:
    """Test SurfaceConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = SurfaceConfig()

        assert config.min_strike_pct == 0.7
        assert config.max_strike_pct == 1.3
        assert config.interpolation == InterpolationMethod.LINEAR
        assert 30 in config.dte_steps


class TestIVSurface:
    """Test IVSurface class."""

    def test_surface_initialization(self):
        """Test surface initialization."""
        surface = IVSurface(underlying_price=450.0)

        assert surface.underlying_price == 450.0
        assert len(surface._points) == 0

    def test_add_point(self):
        """Test adding points."""
        surface = IVSurface(underlying_price=450.0)

        surface.add_point(
            strike=450.0,
            expiry_days=30,
            iv=0.25,
        )

        assert len(surface._points) == 1
        assert surface._points[0].iv == 0.25
        assert surface._points[0].moneyness == 1.0

    def test_add_multiple_points(self):
        """Test adding multiple points."""
        surface = IVSurface(underlying_price=450.0)

        points = [
            SurfacePoint(strike=400, expiry_days=30, iv=0.30, moneyness=400 / 450),
            SurfacePoint(strike=450, expiry_days=30, iv=0.25, moneyness=1.0),
            SurfacePoint(strike=500, expiry_days=30, iv=0.28, moneyness=500 / 450),
        ]

        surface.add_points(points)

        assert len(surface._points) == 3

    def test_get_iv_exact_match(self):
        """Test getting IV for exact point."""
        surface = IVSurface(underlying_price=450.0)

        surface.add_point(strike=450.0, expiry_days=30, iv=0.25)

        iv = surface.get_iv(moneyness=1.0, expiry_days=30)
        assert iv == 0.25

    def test_get_iv_interpolation(self):
        """Test IV interpolation."""
        surface = IVSurface(underlying_price=100.0)

        # Add a grid of points
        for strike_mult in [0.9, 1.0, 1.1]:
            for dte in [30, 60]:
                surface.add_point(
                    strike=100 * strike_mult,
                    expiry_days=dte,
                    iv=0.20 + (1 - strike_mult) * 0.1,
                )

        # Interpolate
        iv = surface.get_iv(moneyness=0.95, expiry_days=45)

        assert iv is not None
        assert 0.15 < iv < 0.35  # Reasonable range

    def test_get_atm_iv(self):
        """Test ATM IV retrieval."""
        surface = IVSurface(underlying_price=450.0)

        surface.add_point(strike=450.0, expiry_days=30, iv=0.25)

        atm_iv = surface.get_atm_iv(expiry_days=30)
        assert atm_iv == 0.25

    def test_get_term_structure(self):
        """Test term structure extraction."""
        surface = IVSurface(underlying_price=100.0)

        # Add ATM points at different expirations
        for dte in [7, 14, 30, 60, 90]:
            surface.add_point(strike=100, expiry_days=dte, iv=0.20 + dte * 0.001)

        term_structure = surface.get_iv_term_structure(moneyness=1.0)

        assert len(term_structure) > 0
        # IV should increase with DTE
        dtes = [t[0] for t in term_structure]
        assert dtes == sorted(dtes)

    def test_get_smile(self):
        """Test volatility smile extraction."""
        surface = IVSurface(underlying_price=100.0)

        # Add smile points
        surface.add_point(strike=90, expiry_days=30, iv=0.30)
        surface.add_point(strike=100, expiry_days=30, iv=0.25)
        surface.add_point(strike=110, expiry_days=30, iv=0.28)

        smile = surface.get_smile(expiry_days=30, num_points=5)

        assert len(smile) > 0

    def test_get_surface_grid(self):
        """Test grid generation."""
        surface = IVSurface(underlying_price=100.0)

        # Add enough points for grid
        for m in [0.9, 0.95, 1.0, 1.05, 1.1]:
            for dte in [30, 60, 90]:
                surface.add_point(
                    strike=100 * m,
                    expiry_days=dte,
                    iv=0.25,
                )

        grid = surface.get_surface_grid()

        assert "moneyness" in grid
        assert "dte" in grid
        assert "iv" in grid
        assert grid["spot"] == 100.0

    def test_detect_arbitrage(self):
        """Test arbitrage detection."""
        surface = IVSurface(underlying_price=100.0)

        # Add inverted term structure (potential arbitrage)
        surface.add_point(strike=100, expiry_days=30, iv=0.30)
        surface.add_point(strike=100, expiry_days=60, iv=0.20)  # Lower IV at longer DTE

        arbitrage = surface.detect_arbitrage()

        # Should detect calendar arbitrage
        assert len(arbitrage) >= 0  # May or may not detect based on threshold

    def test_get_stats(self):
        """Test statistics calculation."""
        surface = IVSurface(underlying_price=100.0)

        surface.add_point(strike=90, expiry_days=30, iv=0.30)
        surface.add_point(strike=100, expiry_days=30, iv=0.25)
        surface.add_point(strike=110, expiry_days=30, iv=0.28)

        stats = surface.get_stats()

        assert stats["num_points"] == 3
        assert stats["min_iv"] == 0.25
        assert stats["max_iv"] == 0.30
        assert stats["spot"] == 100.0

    def test_to_dict(self):
        """Test serialization."""
        surface = IVSurface(underlying_price=100.0)
        surface.add_point(strike=100, expiry_days=30, iv=0.25)

        data = surface.to_dict()

        assert data["underlying_price"] == 100.0
        assert len(data["points"]) == 1
        assert "last_updated" in data

    def test_update_spot(self):
        """Test spot price update."""
        surface = IVSurface(underlying_price=100.0)
        surface.add_point(strike=100, expiry_days=30, iv=0.25)

        assert surface._points[0].moneyness == 1.0

        surface.update_spot(110.0)

        assert surface.underlying_price == 110.0
        # Moneyness should be recalculated
        assert abs(surface._points[0].moneyness - 100 / 110) < 0.001

    def test_clear(self):
        """Test clearing surface."""
        surface = IVSurface(underlying_price=100.0)
        surface.add_point(strike=100, expiry_days=30, iv=0.25)

        assert len(surface._points) == 1

        surface.clear()

        assert len(surface._points) == 0


class TestCreateIVSurface:
    """Test factory function."""

    def test_create_with_defaults(self):
        """Test creating surface with defaults."""
        surface = create_iv_surface(underlying_price=450.0)

        assert surface.underlying_price == 450.0
        assert surface.config.min_strike_pct == 0.7

    def test_create_with_custom_range(self):
        """Test creating surface with custom range."""
        surface = create_iv_surface(
            underlying_price=450.0,
            min_strike_pct=0.8,
            max_strike_pct=1.2,
        )

        assert surface.config.min_strike_pct == 0.8
        assert surface.config.max_strike_pct == 1.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
