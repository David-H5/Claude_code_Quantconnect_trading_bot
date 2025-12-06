"""
System Monitor for Trading Bot

DEPRECATED: This module has been moved to observability.monitoring.system.health.
This file provides backwards-compatible re-exports.

Use the new import path:
    from observability.monitoring.system import SystemMonitor, HealthStatus

Original: System health monitoring
Refactored: Phase 4 - Unified Monitoring & Alerting Infrastructure
"""

# Re-export everything from new location for backwards compatibility
from observability.monitoring.system.health import (
    HealthStatus,
    ServiceCheck,
    ServiceState,
    SystemMetrics,
    SystemMonitor,
    create_disk_check,
    create_http_check,
    create_memory_check,
    create_system_monitor,
)


__all__ = [
    "HealthStatus",
    "ServiceCheck",
    "ServiceState",
    "SystemMetrics",
    "SystemMonitor",
    "create_disk_check",
    "create_http_check",
    "create_memory_check",
    "create_system_monitor",
]
