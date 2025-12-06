"""
Token Usage Metrics Tracker

DEPRECATED: This module has been moved to observability.metrics.collectors.token.
This file provides backwards-compatible re-exports.

Use the new import path:
    from observability.metrics.collectors.token import TokenUsageTracker
    from observability.metrics.collectors.token import record_usage

Original: UPGRADE-014 Category 2: Observability & Debugging
Refactored: Phase 3 - Consolidated Metrics Infrastructure
"""

# Re-export everything from new location for backwards compatibility
from observability.metrics.collectors.token import (
    # Constants
    MODEL_COSTS,
    # Data classes
    TokenUsageRecord,
    TokenUsageSummary,
    # Main class
    TokenUsageTracker,
    create_tracker,
    # Factory functions
    get_global_tracker,
    get_model_cost,
    # Convenience functions
    record_usage,
)


__all__ = [
    # Constants
    "MODEL_COSTS",
    "get_model_cost",
    # Data classes
    "TokenUsageRecord",
    "TokenUsageSummary",
    # Main class
    "TokenUsageTracker",
    # Factory functions
    "get_global_tracker",
    "create_tracker",
    # Convenience functions
    "record_usage",
]
