"""
Trading Dashboard UI Module

Provides PySide6-based trading dashboard with:
- Real-time positions display
- Options and movement scanners
- News and alerts panel
- Order management
- Position tracker for all sources
- Configurable layout

Layer: 4 (Applications)
May import from: Layers 0-3 (all lower layers)
This is the top layer - nothing should import from ui.
"""

# LLM Dashboard Widgets (UPGRADE-006 - December 2025)
from .agent_metrics_widget import (
    AgentMetricsWidget,
    MetricDisplayConfig,
    MetricLabel,
    MetricsChartWidget,
    create_agent_metrics_widget,
)

# Chart Components (UPGRADE-007 - December 2025)
from .charts import (
    MATPLOTLIB_AVAILABLE,
    BaseChartWidget,
    CalibrationBin,
    CalibrationChartWidget,
    DecisionChartWidget,
    DecisionOutcome,
    DecisionStats,
    EvolutionChartWidget,
    EvolutionCycle,
    MetricsDataPoint,
    create_base_chart,
    create_calibration_chart,
    create_decision_chart,
    create_evolution_chart,
    create_metrics_chart,
)
from .charts import (
    MetricsChartWidget as MetricsTrendChart,
)
from .custom_leg_builder import (
    CustomLegBuilderWidget,
    OptionLeg,
    OptionType,
    PLDiagramWidget,
    Side,
    SpreadDefinition,
)
from .dashboard import (
    NewsPanel,
    OrdersPanel,
    PositionsPanel,
    ScannerPanel,
    TradingDashboard,
    create_dashboard,
    create_llm_dashboard,
    run_dashboard,
)
from .debate_viewer import (
    ArgumentPanel,
    DebateViewer,
    ModeratorPanel,
    create_debate_viewer,
)
from .decision_log_viewer import (
    DecisionDetailPanel,
    DecisionLogViewer,
    create_decision_log_viewer,
)
from .evolution_monitor import (
    CycleHistoryWidget,
    EvolutionMonitor,
    EvolutionProgressWidget,
    PromptVersionWidget,
    create_evolution_monitor,
)
from .position_tracker import (
    PositionData,
    PositionGreeks,
    PositionSource,
    PositionTrackerWidget,
    create_position_tracker,
)
from .reasoning_viewer import (
    ChainDetailPanel,
    ReasoningStepWidget,
    ReasoningViewerWidget,
    create_reasoning_viewer,
)
from .strategy_selector import (
    STRATEGY_DEFINITIONS,
    ExecutionType,
    OrderSubmission,
    StrategyDefinition,
    StrategySelectorWidget,
    StrikeSelectionMode,
)
from .widgets import (
    PYSIDE_AVAILABLE,
    AlertData,
    AlertPopup,
    DataTable,
    PriceDisplay,
    SettingsPanel,
    StatusIndicator,
    StyledButton,
)


__all__ = [
    # Check for PySide6
    "PYSIDE_AVAILABLE",
    # Widgets
    "AlertData",
    "StyledButton",
    "DataTable",
    "StatusIndicator",
    "PriceDisplay",
    "AlertPopup",
    "SettingsPanel",
    # Dashboard panels
    "PositionsPanel",
    "ScannerPanel",
    "NewsPanel",
    "OrdersPanel",
    # Main dashboard
    "TradingDashboard",
    "create_dashboard",
    "run_dashboard",
    # Position tracker
    "PositionTrackerWidget",
    "PositionData",
    "PositionSource",
    "PositionGreeks",
    "create_position_tracker",
    # Strategy selector
    "StrategySelectorWidget",
    "StrategyDefinition",
    "STRATEGY_DEFINITIONS",
    "ExecutionType",
    "StrikeSelectionMode",
    "OrderSubmission",
    # Custom leg builder
    "CustomLegBuilderWidget",
    "OptionLeg",
    "OptionType",
    "Side",
    "SpreadDefinition",
    "PLDiagramWidget",
    # LLM Dashboard Widgets (UPGRADE-006 - December 2025)
    "AgentMetricsWidget",
    "MetricLabel",
    "MetricDisplayConfig",
    "MetricsChartWidget",
    "create_agent_metrics_widget",
    "DebateViewer",
    "ArgumentPanel",
    "ModeratorPanel",
    "create_debate_viewer",
    "EvolutionMonitor",
    "EvolutionProgressWidget",
    "CycleHistoryWidget",
    "PromptVersionWidget",
    "create_evolution_monitor",
    "DecisionLogViewer",
    "DecisionDetailPanel",
    "create_decision_log_viewer",
    "create_llm_dashboard",
    # Chart Components (UPGRADE-007 - December 2025)
    "MATPLOTLIB_AVAILABLE",
    "BaseChartWidget",
    "create_base_chart",
    "MetricsDataPoint",
    "MetricsTrendChart",
    "create_metrics_chart",
    "CalibrationBin",
    "CalibrationChartWidget",
    "create_calibration_chart",
    "EvolutionCycle",
    "EvolutionChartWidget",
    "create_evolution_chart",
    "DecisionOutcome",
    "DecisionStats",
    "DecisionChartWidget",
    "create_decision_chart",
    # Reasoning Viewer (UPGRADE-010 Sprint 3)
    "ReasoningViewerWidget",
    "ReasoningStepWidget",
    "ChainDetailPanel",
    "create_reasoning_viewer",
]
