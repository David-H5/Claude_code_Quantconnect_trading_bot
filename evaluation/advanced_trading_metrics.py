"""
Advanced Trading Metrics for Algorithm Evaluation.

DEPRECATED: This module has been moved to observability.metrics.collectors.trading.
This file provides backwards-compatible re-exports.

Use the new import path:
    from observability.metrics.collectors.trading import calculate_advanced_trading_metrics
    from observability.metrics.collectors.trading import AdvancedTradingMetrics

Original: Professional trading metrics implementation
Refactored: Phase 3 - Consolidated Metrics Infrastructure
"""

# Re-export everything from new location for backwards compatibility
from observability.metrics.collectors.trading import (
    AdvancedTradingMetrics,
    # Data classes
    Trade,
    # Main functions
    calculate_advanced_trading_metrics,
    # Analysis functions
    check_overfitting_signals,
    compare_to_2025_benchmarks,
    generate_trading_metrics_report,
    # Threshold functions
    get_trading_metrics_thresholds,
    get_trading_metrics_thresholds_2025,
    get_trading_metrics_thresholds_legacy,
)


__all__ = [
    # Data classes
    "Trade",
    "AdvancedTradingMetrics",
    # Main functions
    "calculate_advanced_trading_metrics",
    "generate_trading_metrics_report",
    # Threshold functions
    "get_trading_metrics_thresholds",
    "get_trading_metrics_thresholds_2025",
    "get_trading_metrics_thresholds_legacy",
    # Analysis functions
    "check_overfitting_signals",
    "compare_to_2025_benchmarks",
]
