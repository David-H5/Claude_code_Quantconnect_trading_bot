"""
Chart Components for Trading Dashboard

Provides matplotlib-based visualization widgets with PySide6 integration.
Falls back gracefully when matplotlib is not available.

Example Usage:
    >>> from ui.charts import (
    ...     create_metrics_chart,
    ...     create_calibration_chart,
    ...     create_evolution_chart,
    ...     create_decision_chart,
    ...     MATPLOTLIB_AVAILABLE,
    ... )
    >>>
    >>> # Check if matplotlib is available
    >>> if MATPLOTLIB_AVAILABLE:
    ...     chart = create_metrics_chart()
    ...     chart.set_data([...])
    ... else:
    ...     print("Charts unavailable - install matplotlib")

Available Chart Types:
    - MetricsChartWidget: Agent metrics trend visualization
    - CalibrationChartWidget: Confidence calibration reliability diagrams
    - EvolutionChartWidget: Evolution progress across cycles
    - DecisionChartWidget: Decision distribution (bar or pie)

Data Classes:
    - MetricsDataPoint: Single metrics measurement
    - CalibrationBin: Calibration bin data
    - EvolutionCycle: Evolution cycle data
    - DecisionStats: Decision statistics by type
    - DecisionOutcome: Enum of possible outcomes
"""

from .attention_chart import (
    AttentionHeatmapChart,
    create_attention_heatmap_chart,
)
from .base_chart import (
    MATPLOTLIB_AVAILABLE,
    BaseChartWidget,
    create_base_chart,
)
from .calibration_chart import (
    CalibrationBin,
    CalibrationChartWidget,
    create_calibration_chart,
)
from .decision_chart import (
    DecisionChartWidget,
    DecisionOutcome,
    DecisionStats,
    create_decision_chart,
)
from .elo_history_chart import (
    ELOHistoryChart,
    create_elo_history_chart,
)
from .evolution_chart import (
    EvolutionChartWidget,
    EvolutionCycle,
    create_evolution_chart,
)
from .metrics_chart import (
    MetricsChartWidget,
    MetricsDataPoint,
    create_metrics_chart,
)


__all__ = [
    # Base
    "MATPLOTLIB_AVAILABLE",
    "BaseChartWidget",
    "create_base_chart",
    # Metrics
    "MetricsDataPoint",
    "MetricsChartWidget",
    "create_metrics_chart",
    # Calibration
    "CalibrationBin",
    "CalibrationChartWidget",
    "create_calibration_chart",
    # Evolution
    "EvolutionCycle",
    "EvolutionChartWidget",
    "create_evolution_chart",
    # Decision
    "DecisionOutcome",
    "DecisionStats",
    "DecisionChartWidget",
    "create_decision_chart",
    # ELO History (UPGRADE-010 Sprint 2 P1)
    "ELOHistoryChart",
    "create_elo_history_chart",
    # Attention Heatmap (UPGRADE-010 Sprint 2 P1)
    "AttentionHeatmapChart",
    "create_attention_heatmap_chart",
]
