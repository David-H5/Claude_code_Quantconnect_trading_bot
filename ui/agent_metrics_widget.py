"""
Agent Metrics Widget for Trading Dashboard

Real-time display of AI agent performance metrics.
Shows accuracy, confidence, calibration, and efficiency stats.

Author: QuantConnect Trading Bot
Date: 2025-12-01
UPGRADE-006: LLM Dashboard Integration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QColor, QFont
    from PySide6.QtWidgets import (
        QComboBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QProgressBar,
        QPushButton,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False

    class QWidget:
        pass

    class Signal:
        def __init__(self, *args):
            pass


@dataclass
class MetricDisplayConfig:
    """Configuration for metric display."""

    label: str
    format_str: str = "{:.1%}"
    color_thresholds: dict[str, float] | None = None
    invert_colors: bool = False  # True for metrics where lower is better


class MetricLabel(QWidget if PYSIDE_AVAILABLE else object):
    """Widget for displaying a single metric with color coding."""

    def __init__(
        self,
        name: str,
        config: MetricDisplayConfig | None = None,
        parent: QWidget | None = None,
    ):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self.config = config or MetricDisplayConfig(label=name)
        self._setup_ui(name)

    def _setup_ui(self, name: str) -> None:
        """Set up the UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        self._name_label = QLabel(f"{self.config.label}:")
        self._name_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._name_label)

        layout.addStretch()

        self._value_label = QLabel("--")
        self._value_label.setStyleSheet("color: white; font-weight: bold;")
        layout.addWidget(self._value_label)

    def set_value(self, value: float) -> None:
        """Update the displayed value with color coding."""
        if not PYSIDE_AVAILABLE:
            return

        formatted = self.config.format_str.format(value)
        self._value_label.setText(formatted)

        # Determine color based on thresholds
        if self.config.color_thresholds:
            thresholds = self.config.color_thresholds
            good = thresholds.get("good", 0.8)
            warn = thresholds.get("warn", 0.6)

            if self.config.invert_colors:
                # Lower is better (e.g., calibration error)
                if value <= thresholds.get("good", 0.05):
                    color = "#28a745"  # Green
                elif value <= thresholds.get("warn", 0.1):
                    color = "#ffc107"  # Yellow
                else:
                    color = "#dc3545"  # Red
            else:
                # Higher is better (e.g., accuracy)
                if value >= good:
                    color = "#28a745"
                elif value >= warn:
                    color = "#ffc107"
                else:
                    color = "#dc3545"

            self._value_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px;")


class MetricsChartWidget(QWidget if PYSIDE_AVAILABLE else object):
    """Simple bar chart for metrics visualization."""

    def __init__(self, parent: QWidget | None = None):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the chart UI."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Performance Overview")
        title.setStyleSheet("color: white; font-weight: bold;")
        layout.addWidget(title)

        # Bars container
        self._bars_layout = QVBoxLayout()
        layout.addLayout(self._bars_layout)

        # Initialize metric bars
        self._bars: dict[str, QProgressBar] = {}

        metrics = [
            ("accuracy", "Accuracy", "#28a745"),
            ("confidence", "Avg Confidence", "#17a2b8"),
            ("calibration", "Calibration", "#6c757d"),
        ]

        for metric_id, label, color in metrics:
            bar_layout = QHBoxLayout()

            label_widget = QLabel(label)
            label_widget.setStyleSheet("color: #888; min-width: 100px;")
            bar_layout.addWidget(label_widget)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            bar.setFormat("%v%")
            bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: #2d2d2d;
                    border: 1px solid #3d3d3d;
                    border-radius: 4px;
                    text-align: center;
                    color: white;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 3px;
                }}
            """)
            bar_layout.addWidget(bar)

            self._bars[metric_id] = bar
            self._bars_layout.addLayout(bar_layout)

    def update_metrics(
        self,
        accuracy: float,
        confidence: float,
        calibration: float,
    ) -> None:
        """Update the chart with new metrics."""
        if not PYSIDE_AVAILABLE:
            return

        self._bars["accuracy"].setValue(int(accuracy * 100))
        self._bars["confidence"].setValue(int(confidence * 100))
        # Calibration: lower is better, so show 1 - error
        self._bars["calibration"].setValue(int((1 - calibration) * 100))


class AgentMetricsWidget(QWidget if PYSIDE_AVAILABLE else object):
    """
    Widget displaying AI agent performance metrics.

    Shows:
    - Agent selector
    - Accuracy rate and calibration error
    - Confidence metrics (average, std dev)
    - Over/under confidence rates
    - Execution time statistics
    - Visual chart of key metrics

    Example usage:
        from evaluation.agent_metrics import AgentMetricsTracker
        tracker = AgentMetricsTracker()
        widget = AgentMetricsWidget(metrics_tracker=tracker)
        dashboard.add_widget(widget)
    """

    # Signals
    agent_selected = Signal(str) if PYSIDE_AVAILABLE else None
    refresh_requested = Signal() if PYSIDE_AVAILABLE else None

    def __init__(
        self,
        metrics_tracker: Any | None = None,
        parent: QWidget | None = None,
    ):
        """
        Initialize agent metrics widget.

        Args:
            metrics_tracker: AgentMetricsTracker instance
            parent: Parent widget
        """
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self.tracker = metrics_tracker
        self._current_agent: str | None = None

        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header with agent selector
        header_layout = QHBoxLayout()

        title = QLabel("Agent Metrics")
        title.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Agent selector
        header_layout.addWidget(QLabel("Agent:"))
        self.agent_combo = QComboBox()
        self.agent_combo.setMinimumWidth(150)
        self.agent_combo.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3d3d3d;
                padding: 5px;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: white;
                selection-background-color: #0066cc;
            }
        """)
        self.agent_combo.currentTextChanged.connect(self._on_agent_changed)
        header_layout.addWidget(self.agent_combo)

        # Refresh button
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedSize(30, 30)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)
        refresh_btn.clicked.connect(self.refresh)
        header_layout.addWidget(refresh_btn)

        layout.addLayout(header_layout)

        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left: Metrics display
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_layout.setContentsMargins(0, 0, 0, 0)

        # Accuracy section
        accuracy_group = QGroupBox("Accuracy")
        accuracy_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
            }
        """)
        accuracy_layout = QVBoxLayout(accuracy_group)

        self._accuracy_label = MetricLabel(
            "accuracy_rate",
            MetricDisplayConfig(
                label="Accuracy Rate",
                format_str="{:.1%}",
                color_thresholds={"good": 0.75, "warn": 0.50},
            ),
        )
        accuracy_layout.addWidget(self._accuracy_label)

        self._calibration_label = MetricLabel(
            "calibration_error",
            MetricDisplayConfig(
                label="Calibration Error",
                format_str="{:.3f}",
                color_thresholds={"good": 0.05, "warn": 0.10},
                invert_colors=True,
            ),
        )
        accuracy_layout.addWidget(self._calibration_label)

        self._overconfidence_label = MetricLabel(
            "overconfidence",
            MetricDisplayConfig(
                label="Overconfidence Rate",
                format_str="{:.1%}",
                color_thresholds={"good": 0.10, "warn": 0.20},
                invert_colors=True,
            ),
        )
        accuracy_layout.addWidget(self._overconfidence_label)

        self._underconfidence_label = MetricLabel(
            "underconfidence",
            MetricDisplayConfig(
                label="Underconfidence Rate",
                format_str="{:.1%}",
                color_thresholds={"good": 0.10, "warn": 0.20},
                invert_colors=True,
            ),
        )
        accuracy_layout.addWidget(self._underconfidence_label)

        metrics_layout.addWidget(accuracy_group)

        # Confidence section
        confidence_group = QGroupBox("Confidence")
        confidence_group.setStyleSheet(accuracy_group.styleSheet())
        confidence_layout = QVBoxLayout(confidence_group)

        self._avg_confidence_label = MetricLabel(
            "avg_confidence",
            MetricDisplayConfig(
                label="Average Confidence",
                format_str="{:.1%}",
            ),
        )
        confidence_layout.addWidget(self._avg_confidence_label)

        self._std_confidence_label = MetricLabel(
            "std_confidence",
            MetricDisplayConfig(
                label="Std Deviation",
                format_str="{:.3f}",
            ),
        )
        confidence_layout.addWidget(self._std_confidence_label)

        metrics_layout.addWidget(confidence_group)

        # Efficiency section
        efficiency_group = QGroupBox("Efficiency")
        efficiency_group.setStyleSheet(accuracy_group.styleSheet())
        efficiency_layout = QVBoxLayout(efficiency_group)

        self._total_decisions_label = MetricLabel(
            "total_decisions",
            MetricDisplayConfig(
                label="Total Decisions",
                format_str="{:.0f}",
            ),
        )
        efficiency_layout.addWidget(self._total_decisions_label)

        self._evaluated_label = MetricLabel(
            "evaluated",
            MetricDisplayConfig(
                label="Evaluated",
                format_str="{:.0f}",
            ),
        )
        efficiency_layout.addWidget(self._evaluated_label)

        self._avg_exec_time_label = MetricLabel(
            "avg_exec_time",
            MetricDisplayConfig(
                label="Avg Exec Time",
                format_str="{:.0f}ms",
            ),
        )
        efficiency_layout.addWidget(self._avg_exec_time_label)

        self._p95_exec_time_label = MetricLabel(
            "p95_exec_time",
            MetricDisplayConfig(
                label="P95 Exec Time",
                format_str="{:.0f}ms",
            ),
        )
        efficiency_layout.addWidget(self._p95_exec_time_label)

        metrics_layout.addWidget(efficiency_group)
        metrics_layout.addStretch()

        splitter.addWidget(metrics_widget)

        # Right: Chart
        self._chart = MetricsChartWidget()
        splitter.addWidget(self._chart)

        # Set splitter sizes
        splitter.setSizes([250, 350])

        layout.addWidget(splitter)

        # Decision distribution table
        dist_group = QGroupBox("Decision Distribution")
        dist_group.setStyleSheet(accuracy_group.styleSheet())
        dist_layout = QVBoxLayout(dist_group)

        self._distribution_table = QTableWidget()
        self._distribution_table.setColumnCount(2)
        self._distribution_table.setHorizontalHeaderLabels(["Type", "Count"])
        self._distribution_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                gridline-color: #3d3d3d;
                border: none;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #e0e0e0;
                padding: 6px;
                border: 1px solid #3d3d3d;
            }
        """)
        self._distribution_table.setMaximumHeight(120)
        self._distribution_table.horizontalHeader().setStretchLastSection(True)
        dist_layout.addWidget(self._distribution_table)

        dist_group.setMaximumHeight(180)
        layout.addWidget(dist_group)

    def _setup_timer(self) -> None:
        """Set up auto-refresh timer."""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(5000)  # Refresh every 5 seconds

    def _on_agent_changed(self, agent_name: str) -> None:
        """Handle agent selection change."""
        self._current_agent = agent_name
        if self.agent_selected:
            self.agent_selected.emit(agent_name)
        self._update_display()

    def refresh(self) -> None:
        """Refresh the metrics display."""
        if not PYSIDE_AVAILABLE or not self.tracker:
            return

        # Update agent list
        self._update_agent_list()

        # Update display
        self._update_display()

        if self.refresh_requested:
            self.refresh_requested.emit()

    def _update_agent_list(self) -> None:
        """Update the agent combo box with available agents."""
        if not self.tracker:
            return

        current = self.agent_combo.currentText()
        self.agent_combo.blockSignals(True)
        self.agent_combo.clear()

        # Get all agents
        all_metrics = self.tracker.get_all_metrics()
        agents = sorted(all_metrics.keys())

        if agents:
            self.agent_combo.addItems(agents)
            # Restore selection if possible
            if current in agents:
                self.agent_combo.setCurrentText(current)
            else:
                self._current_agent = agents[0]
        else:
            self.agent_combo.addItem("(no agents)")

        self.agent_combo.blockSignals(False)

    def _update_display(self) -> None:
        """Update metrics display for current agent."""
        if not self.tracker or not self._current_agent:
            return

        metrics = self.tracker.get_metrics(self._current_agent)

        # Update labels
        self._accuracy_label.set_value(metrics.accuracy_rate)
        self._calibration_label.set_value(metrics.calibration_error)
        self._overconfidence_label.set_value(metrics.overconfidence_rate)
        self._underconfidence_label.set_value(metrics.underconfidence_rate)

        self._avg_confidence_label.set_value(metrics.average_confidence)
        self._std_confidence_label.set_value(metrics.confidence_std)

        self._total_decisions_label.set_value(float(metrics.total_decisions))
        self._evaluated_label.set_value(float(metrics.decisions_evaluated))
        self._avg_exec_time_label.set_value(metrics.average_execution_time_ms)
        self._p95_exec_time_label.set_value(metrics.p95_execution_time_ms)

        # Update chart
        self._chart.update_metrics(
            accuracy=metrics.accuracy_rate,
            confidence=metrics.average_confidence,
            calibration=metrics.calibration_error,
        )

        # Update distribution table
        self._distribution_table.setRowCount(0)
        for dtype, count in sorted(
            metrics.decision_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            row = self._distribution_table.rowCount()
            self._distribution_table.insertRow(row)
            self._distribution_table.setItem(row, 0, QTableWidgetItem(dtype))
            self._distribution_table.setItem(row, 1, QTableWidgetItem(str(count)))

    def set_metrics_tracker(self, tracker: Any) -> None:
        """
        Set or update the metrics tracker.

        Args:
            tracker: AgentMetricsTracker instance
        """
        self.tracker = tracker
        self.refresh()


def create_agent_metrics_widget(
    metrics_tracker: Any | None = None,
) -> AgentMetricsWidget:
    """
    Factory function to create an agent metrics widget.

    Args:
        metrics_tracker: AgentMetricsTracker instance

    Returns:
        Configured AgentMetricsWidget
    """
    return AgentMetricsWidget(metrics_tracker=metrics_tracker)


# Stub implementation when PySide6 not available
if not PYSIDE_AVAILABLE:

    class MetricLabel:
        def __init__(self, *args, **kwargs):
            pass

        def set_value(self, value):
            pass

    class MetricsChartWidget:
        def __init__(self, *args, **kwargs):
            pass

        def update_metrics(self, *args, **kwargs):
            pass

    class AgentMetricsWidget:
        def __init__(self, *args, **kwargs):
            pass

        def refresh(self):
            pass

        def set_metrics_tracker(self, tracker):
            pass


__all__ = [
    "AgentMetricsWidget",
    "MetricDisplayConfig",
    "MetricLabel",
    "MetricsChartWidget",
    "create_agent_metrics_widget",
]
