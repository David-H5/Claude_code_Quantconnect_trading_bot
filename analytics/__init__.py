"""
Analytics Module

UPGRADE-015 Phase 8: Options Analytics Engine

Provides options analytics capabilities including:
- Implied Volatility Surface modeling
- Term Structure analysis
- Volatility Skew calculation
- Greeks calculation
- Options pricing models

Usage:
    from analytics import (
        IVSurface,
        TermStructure,
        VolatilitySkew,
        GreeksCalculator,
        BlackScholes,
    )
"""

from analytics.greeks_calculator import (
    GreeksCalculator,
    GreeksResult,
    create_greeks_calculator,
)
from analytics.iv_surface import (
    IVSurface,
    SurfacePoint,
    create_iv_surface,
)
from analytics.pricing_models import (
    BinomialTree,
    BlackScholes,
    PricingResult,
    create_pricer,
)
from analytics.term_structure import (
    TermPoint,
    TermStructure,
    create_term_structure,
)
from analytics.volatility_skew import (
    SkewMetrics,
    VolatilitySkew,
    create_volatility_skew,
)


__all__ = [
    # IV Surface
    "IVSurface",
    "SurfacePoint",
    "create_iv_surface",
    # Term Structure
    "TermStructure",
    "TermPoint",
    "create_term_structure",
    # Volatility Skew
    "VolatilitySkew",
    "SkewMetrics",
    "create_volatility_skew",
    # Greeks
    "GreeksCalculator",
    "GreeksResult",
    "create_greeks_calculator",
    # Pricing
    "BlackScholes",
    "BinomialTree",
    "PricingResult",
    "create_pricer",
]
