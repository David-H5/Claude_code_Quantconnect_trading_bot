"""
Trading Monitoring Infrastructure

Consolidated trading-specific monitors for execution quality,
risk metrics, and market conditions.

Canonical locations (use these imports):
    from observability.monitoring.trading import SlippageMonitor
    from observability.monitoring.trading import GreeksMonitor
    from observability.monitoring.trading import CorrelationMonitor
    from observability.monitoring.trading import VaRMonitor

Part of CONSOLIDATE-001 Phase 4: Monitoring Consolidation
"""

from observability.monitoring.trading.slippage import (
    SlippageAlert,
    SlippageMetrics,
    SlippageMonitor,
    create_slippage_monitor,
)
from observability.monitoring.trading.greeks import (
    GreeksAlert,
    GreeksMetrics,
    GreeksMonitor,
    create_greeks_monitor,
)
from observability.monitoring.trading.correlation import (
    CorrelationAlert,
    CorrelationMetrics,
    CorrelationMonitor,
    create_correlation_monitor,
)
from observability.monitoring.trading.var import (
    VaRAlert,
    VaRMetrics,
    VaRMonitor,
    create_var_monitor,
)


__all__ = [
    # Slippage monitoring
    "SlippageAlert",
    "SlippageMetrics",
    "SlippageMonitor",
    "create_slippage_monitor",
    # Greeks monitoring
    "GreeksAlert",
    "GreeksMetrics",
    "GreeksMonitor",
    "create_greeks_monitor",
    # Correlation monitoring
    "CorrelationAlert",
    "CorrelationMetrics",
    "CorrelationMonitor",
    "create_correlation_monitor",
    # VaR monitoring
    "VaRAlert",
    "VaRMetrics",
    "VaRMonitor",
    "create_var_monitor",
]
