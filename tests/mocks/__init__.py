"""
Consolidated Mock Registry

Central location for all mock classes used in tests.
Import from here instead of redefining mocks in test files.

UPGRADE-015: Mock Consolidation - Reduce Duplication
"""

from tests.conftest import (
    MockIndicator,
    MockIndicatorDataPoint,
    MockPortfolio,
    MockPortfolioHolding,
    MockSecurities,
    MockSecurityHolding,
    MockSlice,
    MockSymbol,
    MockTradeBar,
)

__all__ = [
    # Core mocks from conftest
    "MockSymbol",
    "MockTradeBar",
    "MockSlice",
    "MockIndicator",
    "MockIndicatorDataPoint",
    "MockPortfolio",
    "MockPortfolioHolding",
    "MockSecurities",
    "MockSecurityHolding",
]
