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
    AlertLevel,
    FillRecord,
    SlippageAlert,
    SlippageDirection,
    SlippageMonitor,
    create_slippage_monitor,
)
from observability.monitoring.trading.greeks import (
    GreeksAlert,
    GreeksAlertLevel,
    GreeksLimits,
    GreeksMonitor,
    PositionGreeksSnapshot,
    create_greeks_monitor,
)
from observability.monitoring.trading.correlation import (
    ConcentrationLevel,
    CorrelationAlert,
    CorrelationConfig,
    CorrelationMonitor,
    CorrelationPair,
    DiversificationScore,
    create_correlation_monitor,
)
from observability.monitoring.trading.var import (
    PositionRisk,
    RiskLevel,
    VaRAlert,
    VaRLimits,
    VaRMethod,
    VaRMonitor,
    VaRResult,
    create_var_monitor,
)


__all__ = [
    # Slippage monitoring
    "AlertLevel",
    "FillRecord",
    "SlippageAlert",
    "SlippageDirection",
    "SlippageMonitor",
    "create_slippage_monitor",
    # Greeks monitoring
    "GreeksAlert",
    "GreeksAlertLevel",
    "GreeksLimits",
    "GreeksMonitor",
    "PositionGreeksSnapshot",
    "create_greeks_monitor",
    # Correlation monitoring
    "ConcentrationLevel",
    "CorrelationAlert",
    "CorrelationConfig",
    "CorrelationMonitor",
    "CorrelationPair",
    "DiversificationScore",
    "create_correlation_monitor",
    # VaR monitoring
    "PositionRisk",
    "RiskLevel",
    "VaRAlert",
    "VaRLimits",
    "VaRMethod",
    "VaRMonitor",
    "VaRResult",
    "create_var_monitor",
]
