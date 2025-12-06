"""
Execution Module

Layer: 3 (Domain Logic)
May import from: Layers 0-2 (utils, observability, config, models, compliance)
May be imported by: Layer 4 (algorithms, api, ui)

Provides smart order execution and profit-taking risk management:
- Graduated profit-taking at configurable thresholds
- Smart cancel/replace for unfilled orders
- Bid-ask spread analysis and execution cost estimation
- Spread anomaly detection (quote stuffing, manipulation)
- Fill rate prediction with 25% minimum threshold
- Two-part spread strategy (leg into butterflies/condors)
- Arbitrage executor with concurrent expiration trading
- Autonomous optimization of sizing and timing
- Bot-managed positions with automatic profit-taking and stop-loss
- Recurring order templates with scheduling and conditions
- Execution analytics
- ML-based fill prediction (UPGRADE-010 Sprint 4)
- Intelligent cancel timing optimization (UPGRADE-010 Sprint 4)
- Option chain liquidity scoring (UPGRADE-010 Sprint 4)
- Slippage monitoring with alerts (UPGRADE-010 Sprint 5)
- Execution quality metrics dashboard (UPGRADE-010 Sprint 5)
"""

from .arbitrage_executor import (
    ArbitrageExecutor,
    ArbitrageOpportunity,
    CreditMaximizer,
    ExpirationCategory,
    ExpirationTrader,
    OptimizationMetrics,
    OrderAttempt,
    OrderResult,
    PositionBalance,
    PositionBalancer,
    SizingParameters,
    SpreadLeg,
    StrikeRange,
    TimingParameters,
    TradingPhase,
    create_arbitrage_executor,
)
from .bot_managed_positions import (
    BotManagedPosition,
    BotPositionManager,
    ManagementAction,
    PositionSource,
    ProfitThreshold,
    create_bot_position_manager,
)

# Cancel Optimizer (UPGRADE-010 Sprint 4 - December 2025)
from .cancel_optimizer import (
    CancelDecision,
    CancelOptimizer,
    CancelReason,
    CancelTimingFeatures,
    TimingRecord,
    TimingStatistics,
    create_cancel_optimizer,
)

# Execution Quality Metrics (UPGRADE-010 Sprint 5 - December 2025)
from .execution_quality_metrics import (
    ExecutionDashboard,
    ExecutionQualityTracker,
    OrderRecord,
    QualityThresholds,
    create_execution_tracker,
    generate_execution_report,
)
from .execution_quality_metrics import (
    OrderStatus as ExecutionOrderStatus,  # Aliased to avoid conflict with smart_execution
)

# ML Fill Prediction (UPGRADE-010 Sprint 4 - December 2025)
from .fill_ml_model import (
    FillFeatures,
    FillMLModel,
    MarketRegime,
    MLFillPrediction,
    ModelType,
    TrainingRecord,
    TrainingResult,
    create_fill_ml_model,
)
from .fill_predictor import (
    FillOutcome,
    FillPrediction,
    FillRatePredictor,
    FillRecord,
    FillStatistics,
    OrderPlacement,
    create_fill_predictor,
)

# Liquidity Scorer (UPGRADE-010 Sprint 4 Expansion - December 2025)
from .liquidity_scorer import (
    ChainLiquiditySummary,
    LiquidityConfig,
    LiquidityRating,
    LiquidityScore,
    LiquidityScorer,
    OptionLiquidityData,
    create_liquidity_scorer,
)
from .manual_legs_executor import (
    ComboOrder,
    ExecutionPhase,
    LegType,
    ManualLeg,
    ManualLegsExecutor,
    create_manual_legs_executor,
)
from .manual_legs_executor import (
    TwoPartPosition as ManualTwoPartPosition,
)
from .option_strategies_executor import (
    FactoryPosition,
    OptionStrategiesExecutor,
    StrategyCondition,
    StrategyConfig,
    create_option_strategies_executor,
)
from .profit_taking import (
    OrderSide,
    PositionState,
    ProfitTakeOrder,
    ProfitTakingRiskManagementModel,
    ProfitTakingRiskModel,
    create_profit_taking_model,
)
from .recurring_order_manager import (
    ConditionOperator,
    ConditionType,
    EntryCondition,
    RecurringOrderManager,
    RecurringOrderTemplate,
    ScheduleType,
    StrikeSelection,
    StrikeSelectionMode,
    create_recurring_order_manager,
)

# Slippage Monitor (UPGRADE-010 Sprint 5 - December 2025)
from .slippage_monitor import (
    AlertLevel,
    SlippageAlert,
    SlippageDirection,
    SlippageMonitor,
    SymbolSlippageStats,
    create_slippage_monitor,
)
from .slippage_monitor import (
    ExecutionQualityMetrics as SlippageMetrics,  # Aliased for clarity
)
from .slippage_monitor import (
    FillRecord as SlippageFillRecord,  # Aliased to avoid conflict with fill_predictor
)
from .smart_execution import (
    ExecutionOrder,
    ExecutionResult,
    SmartExecutionExecutionModel,
    SmartExecutionModel,
    create_smart_execution_model,
)
from .smart_execution import (
    SmartOrderStatus as OrderStatus,
)
from .smart_execution import (
    SmartOrderType as OrderType,
)
from .spread_analysis import (
    ExecutionCostEstimate,
    ExecutionUrgency,
    SpreadAnalyzer,
    SpreadMetrics,
    SpreadQuality,
    SpreadSnapshot,
    create_spread_analyzer,
)
from .spread_anomaly import (
    AnomalySeverity,
    QuoteUpdate,
    SpreadAnomaly,
    SpreadAnomalyDetector,
    SpreadAnomalyType,
    SpreadBaseline,
    create_spread_anomaly_detector,
)
from .two_part_spread import (
    CreditMatch,
    DebitOpportunity,
    FillLocation,
    PositionStatus,
    SessionFillStats,
    SpreadFill,
    SpreadQuote,
    SpreadType,
    TwoPartPosition,
    TwoPartSpreadStrategy,
    create_two_part_strategy,
)


__all__ = [
    # Profit taking
    "OrderSide",
    "ProfitTakeOrder",
    "PositionState",
    "ProfitTakingRiskModel",
    "ProfitTakingRiskManagementModel",
    "create_profit_taking_model",
    # Smart execution
    "OrderStatus",
    "OrderType",
    "ExecutionOrder",
    "ExecutionResult",
    "SmartExecutionModel",
    "SmartExecutionExecutionModel",
    "create_smart_execution_model",
    # Spread analysis
    "SpreadQuality",
    "ExecutionUrgency",
    "SpreadSnapshot",
    "ExecutionCostEstimate",
    "SpreadMetrics",
    "SpreadAnalyzer",
    "create_spread_analyzer",
    # Spread anomaly detection
    "SpreadAnomalyType",
    "AnomalySeverity",
    "QuoteUpdate",
    "SpreadAnomaly",
    "SpreadBaseline",
    "SpreadAnomalyDetector",
    "create_spread_anomaly_detector",
    # Fill rate prediction
    "FillOutcome",
    "OrderPlacement",
    "FillRecord",
    "FillPrediction",
    "FillStatistics",
    "FillRatePredictor",
    "create_fill_predictor",
    # Two-part spread strategy
    "SpreadType",
    "PositionStatus",
    "FillLocation",
    "SpreadQuote",
    "SpreadFill",
    "SessionFillStats",
    "DebitOpportunity",
    "CreditMatch",
    "TwoPartPosition",
    "TwoPartSpreadStrategy",
    "create_two_part_strategy",
    # Arbitrage executor
    "ExpirationCategory",
    "OrderResult",
    "TradingPhase",
    "StrikeRange",
    "SpreadLeg",
    "ArbitrageOpportunity",
    "OrderAttempt",
    "PositionBalance",
    "TimingParameters",
    "SizingParameters",
    "OptimizationMetrics",
    "CreditMaximizer",
    "PositionBalancer",
    "ExpirationTrader",
    "ArbitrageExecutor",
    "create_arbitrage_executor",
    # Bot-managed positions
    "ManagementAction",
    "PositionSource",
    "ProfitThreshold",
    "BotManagedPosition",
    "BotPositionManager",
    "create_bot_position_manager",
    # Recurring order templates
    "ScheduleType",
    "ConditionType",
    "ConditionOperator",
    "StrikeSelectionMode",
    "EntryCondition",
    "StrikeSelection",
    "RecurringOrderTemplate",
    "RecurringOrderManager",
    "create_recurring_order_manager",
    # Option strategies executor
    "StrategyCondition",
    "StrategyConfig",
    "FactoryPosition",
    "OptionStrategiesExecutor",
    "create_option_strategies_executor",
    # Manual legs executor
    "LegType",
    "ExecutionPhase",
    "ManualLeg",
    "ComboOrder",
    "ManualTwoPartPosition",
    "ManualLegsExecutor",
    "create_manual_legs_executor",
    # ML Fill Prediction (UPGRADE-010 Sprint 4)
    "FillFeatures",
    "FillMLModel",
    "MarketRegime",
    "MLFillPrediction",
    "ModelType",
    "TrainingRecord",
    "TrainingResult",
    "create_fill_ml_model",
    # Cancel Optimizer (UPGRADE-010 Sprint 4)
    "CancelDecision",
    "CancelOptimizer",
    "CancelReason",
    "CancelTimingFeatures",
    "TimingRecord",
    "TimingStatistics",
    "create_cancel_optimizer",
    # Liquidity Scorer (UPGRADE-010 Sprint 4 Expansion)
    "ChainLiquiditySummary",
    "LiquidityConfig",
    "LiquidityRating",
    "LiquidityScore",
    "LiquidityScorer",
    "OptionLiquidityData",
    "create_liquidity_scorer",
    # Slippage Monitor (UPGRADE-010 Sprint 5)
    "AlertLevel",
    "SlippageFillRecord",
    "SlippageMetrics",
    "SlippageAlert",
    "SlippageDirection",
    "SlippageMonitor",
    "SymbolSlippageStats",
    "create_slippage_monitor",
    # Execution Quality Metrics (UPGRADE-010 Sprint 5)
    "ExecutionDashboard",
    "ExecutionQualityTracker",
    "OrderRecord",
    "ExecutionOrderStatus",
    "QualityThresholds",
    "create_execution_tracker",
    "generate_execution_report",
]
