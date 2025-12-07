"""
Correlation Monitor

DEPRECATED: This module has been moved to observability.monitoring.trading.correlation.
This file provides backwards-compatible re-exports.

Use the new import path:
    from observability.monitoring.trading import CorrelationMonitor, create_correlation_monitor

Part of CONSOLIDATE-001 Phase 4: Monitoring Consolidation
"""

# Re-export everything from new location for backwards compatibility
from observability.monitoring.trading.correlation import (
    ConcentrationLevel,
    CorrelationAlert,
    CorrelationConfig,
    CorrelationMonitor,
    CorrelationPair,
    DiversificationScore,
    create_correlation_monitor,
)


__all__ = [
    "ConcentrationLevel",
    "CorrelationAlert",
    "CorrelationConfig",
    "CorrelationMonitor",
    "CorrelationPair",
    "DiversificationScore",
    "create_correlation_monitor",
]
