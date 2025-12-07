"""
VaR Monitor

DEPRECATED: This module has been moved to observability.monitoring.trading.var.
This file provides backwards-compatible re-exports.

Use the new import path:
    from observability.monitoring.trading import VaRMonitor, create_var_monitor

Part of CONSOLIDATE-001 Phase 4: Monitoring Consolidation
"""

# Re-export everything from new location for backwards compatibility
from observability.monitoring.trading.var import (
    PositionRisk,
    RiskLevel,
    VaRAlert,
    VaRLimits,
    VaRMethod,
    VaRMonitor,
    VaRResult,
    create_var_monitor,
)


__all__ = [
    "PositionRisk",
    "RiskLevel",
    "VaRAlert",
    "VaRLimits",
    "VaRMethod",
    "VaRMonitor",
    "VaRResult",
    "create_var_monitor",
]
