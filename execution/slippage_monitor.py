"""
Slippage Monitor

DEPRECATED: This module has been moved to observability.monitoring.trading.slippage.
This file provides backwards-compatible re-exports.

Use the new import path:
    from observability.monitoring.trading import SlippageMonitor, create_slippage_monitor

Part of CONSOLIDATE-001 Phase 4: Monitoring Consolidation
"""

# Re-export everything from new location for backwards compatibility
from observability.monitoring.trading.slippage import (
    AlertLevel,
    FillRecord,
    SlippageAlert,
    SlippageDirection,
    SlippageMetrics,
    SlippageMonitor,
    create_slippage_monitor,
)


__all__ = [
    "AlertLevel",
    "FillRecord",
    "SlippageAlert",
    "SlippageDirection",
    "SlippageMetrics",
    "SlippageMonitor",
    "create_slippage_monitor",
]
