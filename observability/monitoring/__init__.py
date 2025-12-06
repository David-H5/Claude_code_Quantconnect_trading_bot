"""
Unified Monitoring Infrastructure

Provides consolidated monitoring for the trading bot:
- Base monitoring interfaces (AbstractMonitor, CompositeMonitor)
- System monitors (resource, storage, health)
- Domain monitors (trading, anomaly detection)

Usage:
    # Base monitoring
    from observability.monitoring import (
        AbstractMonitor,
        CompositeMonitor,
        HealthStatus,
        MonitorSeverity,
        MonitorCategory,
        MonitorState,
        MonitorAlert,
    )

    # System monitoring
    from observability.monitoring.system import (
        ResourceMonitor,
        StorageMonitor,
        SystemMonitor,
    )

    # Domain monitoring
    from observability.monitoring.domain import (
        ContinuousMonitor,
        AnomalyAlertingBridge,
    )

Refactored: Phase 4 - Unified Monitoring & Alerting Infrastructure
"""

# Base types
from observability.monitoring.base import (
    AbstractAlertHandler,
    # Abstract base classes
    AbstractMonitor,
    # Composite
    CompositeMonitor,
    # Enums
    HealthStatus,
    MonitorAlert,
    MonitorCategory,
    # Data classes
    MonitorCheck,
    MonitorMetrics,
    MonitorSeverity,
    MonitorState,
    create_alert,
    create_composite_monitor,
    # Factory functions
    create_monitor_state,
)


__all__ = [
    # Enums
    "HealthStatus",
    "MonitorSeverity",
    "MonitorCategory",
    # Data classes
    "MonitorCheck",
    "MonitorState",
    "MonitorAlert",
    "MonitorMetrics",
    # Abstract base classes
    "AbstractMonitor",
    "AbstractAlertHandler",
    # Composite
    "CompositeMonitor",
    # Factory functions
    "create_monitor_state",
    "create_alert",
    "create_composite_monitor",
]
