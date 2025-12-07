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
    ExecutionQualityMetrics,
    FillRecord,
    SlippageAlert,
    SlippageDirection,
    SlippageMonitor,
    SymbolSlippageStats,
    create_slippage_monitor,
    generate_slippage_report,
)


__all__ = [
    "AlertLevel",
    "ExecutionQualityMetrics",
    "FillRecord",
    "SlippageAlert",
    "SlippageDirection",
    "SlippageMonitor",
    "SymbolSlippageStats",
    "create_slippage_monitor",
    "generate_slippage_report",
]
