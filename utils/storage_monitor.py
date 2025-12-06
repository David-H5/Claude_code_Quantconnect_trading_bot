"""
Storage Monitor for QuantConnect Object Store

DEPRECATED: This module has been moved to observability.monitoring.system.storage.
This file provides backwards-compatible re-exports.

Use the new import path:
    from observability.monitoring.system import StorageMonitor, StorageAlert

Original: Object Store usage monitoring
Refactored: Phase 4 - Unified Monitoring & Alerting Infrastructure
"""

# Re-export everything from new location for backwards compatibility
from observability.monitoring.system.storage import (
    StorageAlert,
    StorageMonitor,
    create_storage_monitor,
)


__all__ = [
    "StorageAlert",
    "StorageMonitor",
    "create_storage_monitor",
]
