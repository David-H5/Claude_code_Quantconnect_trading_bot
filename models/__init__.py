"""
Risk and Portfolio Models

Layer: 2 (Core Models)
May import from: Layers 0-1 (utils, observability, infrastructure, config)
May be imported by: Layers 3-4

This package contains risk management and portfolio construction models:
- Risk management (position limits, drawdown controls)
- Circuit breaker (safety mechanism)
- Multi-leg options strategies (spreads, condors, strangles)
- Volatility surface analysis (smile, skew, term structure)
- Enhanced volatility analysis (IV Rank, IV Percentile, regimes)
- P&L attribution by Greeks
"""

# Anomaly Detector (UPGRADE-010 Sprint 1 - December 2025)
# Note: Requires numpy - lazy import to avoid breaking imports when numpy unavailable
try:
    from .anomaly_detector import (
        AnomalyDetector,
        AnomalyDetectorConfig,
        AnomalyResult,
        AnomalySeverity,
        AnomalyType,
        MarketDataPoint,
        create_anomaly_detector,
    )

    _ANOMALY_DETECTOR_AVAILABLE = True
except ImportError:
    _ANOMALY_DETECTOR_AVAILABLE = False

# Multi-Head Attention Layer (UPGRADE-010 Sprint 2 - December 2025)
from .attention_layer import (
    AssetEncoder,
    AttentionBlock,
    AttentionConfig,
    AttentionOutput,
    AttentionPPOConfig,
    AttentionType,
    AttentionWeights,
    LinearLayer,
    MultiHeadAttention,
    PositionalEncoding,
    create_asset_encoder,
    create_attention_layer,
    scaled_dot_product_attention,
)
from .circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    TradingCircuitBreaker,
    TripReason,
    create_circuit_breaker,
)
from .enhanced_volatility import (
    EnhancedVolatilityAnalyzer,
    IVMetrics,
    RealizedVolMetrics,
    VolatilityPremium,
    VolatilityRegime,
    VolatilityRegimeAnalysis,
    VolatilityTrend,
    create_enhanced_volatility_analyzer,
)
from .error_handler import (
    AlertTrigger,
    ErrorAggregation,
    ErrorCategory,
    ErrorHandler,
    ErrorSeverity,
    ServiceHealth,
    ServiceStatus,
    create_error_handler,
)
from .error_handler import (
    TradingError as StructuredTradingError,
)
from .exceptions import (
    CircuitBreakerTripped,
    ConfigurationError,
    DataValidationError,
    ExecutionError,
    InsufficientFunds,
    InvalidPositionSize,
    OptionPricingError,
    OrderRejected,
    RiskLimitExceeded,
    StrategyError,
    TradingError,
)
from .multi_leg_strategy import (
    MultiLegStrategy,
    OptionLeg,
    PortfolioGreeks,
    StrategyBuilder,
    StrategyType,
    find_delta_strikes,
)
from .performance_tracker import (
    PerformanceSummary,
    PerformanceTracker,
    SessionMetrics,
    StrategyMetrics,
    TradeRecord,
    create_performance_tracker,
)
from .pnl_attribution import (
    GreeksSnapshot,
    PnLAttributor,
    PnLBreakdown,
    PortfolioPnLAttributor,
    RealizedVolatilityCalculator,
    create_attributor_from_trades,
)
from .portfolio_hedging import (
    HedgeRecommendation,
    HedgeTargets,
    HedgeType,
    PortfolioHedger,
    Position,
    create_hedger_from_positions,
)
from .retry_handler import (
    RetryConfig,
    RetryHandler,
    async_retry_with_backoff,
    calculate_delay,
    create_retry_handler,
    is_retryable_exception,
    retry_with_backoff,
)
from .risk_manager import PositionInfo, RiskAction, RiskLimits, RiskManager

# RL Rebalancer (UPGRADE-010 Sprint 3 - December 2025)
from .rl_rebalancer import (
    AssetState,
    PolicyNetwork,
    PortfolioState,
    RebalanceAction,
    RebalanceDecision,
    RebalanceExperience,
    RebalanceFrequency,
    RLRebalancer,
    RLRebalancerConfig,
    ValueNetwork,
    create_rl_rebalancer,
)
from .volatility_surface import (
    TermStructure,
    VolatilityAnalyzer,
    VolatilityPoint,
    VolatilitySlice,
    VolatilitySurface,
    create_volatility_surface,
)


# VaR Monitor (UPGRADE-010 Sprint 4 - December 2025)
# Note: Requires numpy - lazy import to avoid breaking imports when numpy unavailable
try:
    from .var_monitor import (
        PositionRisk,
        RiskLevel,
        VaRAlert,
        VaRLimits,
        VaRMethod,
        VaRMonitor,
        VaRResult,
        create_var_monitor,
    )

    _VAR_MONITOR_AVAILABLE = True
except ImportError:
    _VAR_MONITOR_AVAILABLE = False

# TGARCH Volatility Model (UPGRADE-010 Sprint 4 - December 2025)
# Note: Requires numpy - lazy import to avoid breaking imports when numpy unavailable
try:
    from .tgarch import (
        TGARCHFitResult,
        TGARCHModel,
        TGARCHParams,
        create_tgarch_model,
    )

    _TGARCH_AVAILABLE = True
except ImportError:
    _TGARCH_AVAILABLE = False

# Monte Carlo Stress Tester (UPGRADE-010 Sprint 4 - December 2025)
# Note: Requires numpy - lazy import to avoid breaking imports when numpy unavailable
try:
    from .monte_carlo import (
        STRESS_SCENARIOS,
        DrawdownAnalysis,
        MonteCarloStressTester,
        ScenarioType,
        SimulationConfig,
        SimulationResult,
        StressScenario,
        create_monte_carlo_tester,
    )

    _MONTE_CARLO_AVAILABLE = True
except ImportError:
    _MONTE_CARLO_AVAILABLE = False

# Greeks Risk Monitor (UPGRADE-010 Sprint 4 Expansion - December 2025)
# Note: Requires numpy - lazy import to avoid breaking imports when numpy unavailable
try:
    from .greeks_monitor import (
        GreeksAlert,
        GreeksAlertLevel,
        GreeksLimits,
        GreeksMonitor,
        GreeksType,
        HedgeRecommendation,
        PortfolioGreeksExposure,
        PositionGreeksSnapshot,
        RiskProfile,
        create_greeks_monitor,
    )

    _GREEKS_MONITOR_AVAILABLE = True
except ImportError:
    _GREEKS_MONITOR_AVAILABLE = False

# Correlation Monitor (UPGRADE-010 Sprint 4 Expansion - December 2025)
# Note: Requires numpy - lazy import to avoid breaking imports when numpy unavailable
try:
    from .correlation_monitor import (
        ConcentrationLevel,
        CorrelationAlert,
        CorrelationConfig,
        CorrelationMonitor,
        CorrelationPair,
        DiversificationScore,
        create_correlation_monitor,
    )

    _CORRELATION_MONITOR_AVAILABLE = True
except ImportError:
    _CORRELATION_MONITOR_AVAILABLE = False


__all__ = [
    # Risk Manager
    "RiskManager",
    "RiskLimits",
    "RiskAction",
    "PositionInfo",
    # Circuit Breaker
    "TradingCircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "TripReason",
    "create_circuit_breaker",
    # Multi-Leg Strategies
    "StrategyType",
    "OptionLeg",
    "PortfolioGreeks",
    "MultiLegStrategy",
    "StrategyBuilder",
    "find_delta_strikes",
    # Volatility Surface
    "VolatilityPoint",
    "VolatilitySlice",
    "TermStructure",
    "VolatilitySurface",
    "VolatilityAnalyzer",
    "create_volatility_surface",
    # P&L Attribution
    "GreeksSnapshot",
    "PnLBreakdown",
    "PnLAttributor",
    "PortfolioPnLAttributor",
    "RealizedVolatilityCalculator",
    "create_attributor_from_trades",
    # Portfolio Hedging
    "HedgeType",
    "Position",
    "HedgeRecommendation",
    "HedgeTargets",
    "PortfolioHedger",
    "create_hedger_from_positions",
    # Enhanced Volatility Analysis
    "VolatilityRegime",
    "VolatilityTrend",
    "IVMetrics",
    "RealizedVolMetrics",
    "VolatilityPremium",
    "VolatilityRegimeAnalysis",
    "EnhancedVolatilityAnalyzer",
    "create_enhanced_volatility_analyzer",
    # Exceptions
    "TradingError",
    "RiskLimitExceeded",
    "InsufficientFunds",
    "OrderRejected",
    "CircuitBreakerTripped",
    "InvalidPositionSize",
    "DataValidationError",
    "OptionPricingError",
    "ExecutionError",
    "ConfigurationError",
    "StrategyError",
    # Performance Tracker (UPGRADE-010)
    "TradeRecord",
    "SessionMetrics",
    "StrategyMetrics",
    "PerformanceSummary",
    "PerformanceTracker",
    "create_performance_tracker",
    # Error Handling (UPGRADE-012)
    "ErrorSeverity",
    "ErrorCategory",
    "ServiceStatus",
    "StructuredTradingError",
    "ServiceHealth",
    "ErrorAggregation",
    "AlertTrigger",
    "ErrorHandler",
    "create_error_handler",
    # Retry Handler (UPGRADE-012)
    "RetryConfig",
    "RetryHandler",
    "retry_with_backoff",
    "async_retry_with_backoff",
    "calculate_delay",
    "is_retryable_exception",
    "create_retry_handler",
    # Anomaly Detector (UPGRADE-010 Sprint 1)
    "AnomalyDetector",
    "AnomalyDetectorConfig",
    "AnomalyResult",
    "AnomalyType",
    "AnomalySeverity",
    "MarketDataPoint",
    "create_anomaly_detector",
    # Multi-Head Attention Layer (UPGRADE-010 Sprint 2)
    "AttentionType",
    "AttentionConfig",
    "AttentionWeights",
    "AttentionOutput",
    "LinearLayer",
    "MultiHeadAttention",
    "PositionalEncoding",
    "AssetEncoder",
    "AttentionPPOConfig",
    "AttentionBlock",
    "create_attention_layer",
    "create_asset_encoder",
    "scaled_dot_product_attention",
    # RL Rebalancer (UPGRADE-010 Sprint 3)
    "RebalanceAction",
    "RebalanceFrequency",
    "AssetState",
    "PortfolioState",
    "RebalanceDecision",
    "RebalanceExperience",
    "RLRebalancerConfig",
    "PolicyNetwork",
    "ValueNetwork",
    "RLRebalancer",
    "create_rl_rebalancer",
    # VaR Monitor (UPGRADE-010 Sprint 4)
    "PositionRisk",
    "RiskLevel",
    "VaRAlert",
    "VaRLimits",
    "VaRMethod",
    "VaRMonitor",
    "VaRResult",
    "create_var_monitor",
    # TGARCH Volatility Model (UPGRADE-010 Sprint 4)
    "TGARCHFitResult",
    "TGARCHModel",
    "TGARCHParams",
    "create_tgarch_model",
    # Monte Carlo Stress Tester (UPGRADE-010 Sprint 4)
    "DrawdownAnalysis",
    "MonteCarloStressTester",
    "ScenarioType",
    "SimulationConfig",
    "SimulationResult",
    "StressScenario",
    "STRESS_SCENARIOS",
    "create_monte_carlo_tester",
    # Greeks Risk Monitor (UPGRADE-010 Sprint 4 Expansion)
    "GreeksAlert",
    "GreeksAlertLevel",
    "GreeksLimits",
    "GreeksMonitor",
    "GreeksType",
    "HedgeRecommendation",
    "PortfolioGreeksExposure",
    "PositionGreeksSnapshot",
    "RiskProfile",
    "create_greeks_monitor",
    # Correlation Monitor (UPGRADE-010 Sprint 4 Expansion)
    "ConcentrationLevel",
    "CorrelationAlert",
    "CorrelationConfig",
    "CorrelationMonitor",
    "CorrelationPair",
    "DiversificationScore",
    "create_correlation_monitor",
]
