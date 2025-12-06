"""
Resource Monitor for QuantConnect Trading Bot

DEPRECATED: This module has been moved to observability.monitoring.system.resource.
This file provides backwards-compatible re-exports.

Use the new import path:
    from observability.monitoring.system import ResourceMonitor, ResourceMetrics

Original: Resource monitoring for QuantConnect compute nodes
Refactored: Phase 4 - Unified Monitoring & Alerting Infrastructure
"""

# Re-export everything from new location for backwards compatibility
from observability.monitoring.system.resource import (
    ResourceAlert,
    ResourceMetrics,
    ResourceMonitor,
    create_resource_monitor,
)


__all__ = [
    "ResourceAlert",
    "ResourceMetrics",
    "ResourceMonitor",
    "create_resource_monitor",
]
