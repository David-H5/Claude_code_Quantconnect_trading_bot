"""
Greeks Risk Monitor

DEPRECATED: This module has been moved to observability.monitoring.trading.greeks.
This file provides backwards-compatible re-exports.

Use the new import path:
    from observability.monitoring.trading import GreeksMonitor, create_greeks_monitor

Part of CONSOLIDATE-001 Phase 4: Monitoring Consolidation
"""

# Re-export everything from new location for backwards compatibility
from observability.monitoring.trading.greeks import (
    GreeksAlert,
    GreeksAlertLevel,
    GreeksLimits,
    GreeksMonitor,
    GreeksType,
    HedgeRecommendation,
    PortfolioGreeksExposure,
    PositionGreeksSnapshot,
    RiskProfile,
    create_greeks_monitor,
)

# Backwards compatibility alias
GreeksSnapshot = PositionGreeksSnapshot

__all__ = [
    "GreeksAlert",
    "GreeksAlertLevel",
    "GreeksLimits",
    "GreeksMonitor",
    "GreeksSnapshot",
    "GreeksType",
    "HedgeRecommendation",
    "PortfolioGreeksExposure",
    "PositionGreeksSnapshot",
    "RiskProfile",
    "create_greeks_monitor",
]
