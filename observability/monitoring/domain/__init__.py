"""
Domain-Specific Monitoring (Placeholder)

Domain-specific monitors (trading, evaluation) live in their respective
Layer 3 modules to maintain proper layer architecture:

    Layer 3: Domain Logic
    - evaluation/continuous_monitoring.py  → ContinuousMonitor
    - evaluation/anomaly_alerting_bridge.py → AnomalyAlertingBridge
    - execution/ → ExecutionMonitor (future)

This placeholder module documents the architecture but does NOT re-export
Layer 3 modules to avoid layer violations.

Usage (import from original locations):
    from evaluation.continuous_monitoring import ContinuousMonitor
    from evaluation.anomaly_alerting_bridge import AnomalyAlertingBridge

Refactored: Phase 5 - Module Layer Boundaries
"""

# Layer: 1 (Infrastructure)
# This module intentionally does NOT import from higher layers

__all__: list[str] = []


def get_domain_monitor_locations() -> dict[str, str]:
    """
    Get the import paths for domain-specific monitors.

    Returns:
        Dictionary mapping monitor name to its import path
    """
    return {
        "ContinuousMonitor": "evaluation.continuous_monitoring",
        "AnomalyAlertingBridge": "evaluation.anomaly_alerting_bridge",
        "PerformanceSnapshot": "evaluation.continuous_monitoring",
        "DriftAlert": "evaluation.continuous_monitoring",
        "create_continuous_monitor": "evaluation.continuous_monitoring",
        "create_alerting_pipeline": "evaluation.anomaly_alerting_bridge",
    }
