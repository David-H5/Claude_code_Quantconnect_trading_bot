"""
Decision Log Viewer Widget for Trading Dashboard

Browse and analyze agent decision history:
- Filterable decision table
- Decision detail panel
- Risk assessment display
- Reasoning chain visualization

Author: QuantConnect Trading Bot
Date: 2025-12-01
UPGRADE-006: LLM Dashboard Integration
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


try:
    from PySide6.QtCore import QDate, Qt, QTimer, Signal
    from PySide6.QtGui import QColor, QFont
    from PySide6.QtWidgets import (
        QComboBox,
        QDateEdit,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
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


class DecisionDetailPanel(QWidget if PYSIDE_AVAILABLE else object):
    """Panel showing detailed view of a single decision."""

    def __init__(self, parent: QWidget | None = None):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the detail panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QLabel("Decision Details")
        header.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        layout.addWidget(header)

        # Basic info grid
        info_grid = QGridLayout()

        info_grid.addWidget(QLabel("Agent:"), 0, 0)
        self._agent_label = QLabel("--")
        self._agent_label.setStyleSheet("color: #17a2b8; font-weight: bold;")
        info_grid.addWidget(self._agent_label, 0, 1)

        info_grid.addWidget(QLabel("Type:"), 0, 2)
        self._type_label = QLabel("--")
        info_grid.addWidget(self._type_label, 0, 3)

        info_grid.addWidget(QLabel("Decision:"), 1, 0)
        self._decision_label = QLabel("--")
        self._decision_label.setStyleSheet("font-weight: bold;")
        info_grid.addWidget(self._decision_label, 1, 1)

        info_grid.addWidget(QLabel("Confidence:"), 1, 2)
        self._confidence_label = QLabel("--")
        info_grid.addWidget(self._confidence_label, 1, 3)

        info_grid.addWidget(QLabel("Outcome:"), 2, 0)
        self._outcome_label = QLabel("--")
        info_grid.addWidget(self._outcome_label, 2, 1)

        info_grid.addWidget(QLabel("Exec Time:"), 2, 2)
        self._exec_time_label = QLabel("--")
        info_grid.addWidget(self._exec_time_label, 2, 3)

        layout.addLayout(info_grid)

        # Query
        query_group = QGroupBox("Query")
        query_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
        """)
        query_layout = QVBoxLayout(query_group)
        self._query_text = QTextEdit()
        self._query_text.setReadOnly(True)
        self._query_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: none;
            }
        """)
        self._query_text.setMaximumHeight(60)
        query_layout.addWidget(self._query_text)
        layout.addWidget(query_group)

        # Reasoning chain
        reasoning_group = QGroupBox("Reasoning Chain")
        reasoning_group.setStyleSheet(query_group.styleSheet())
        reasoning_layout = QVBoxLayout(reasoning_group)
        self._reasoning_list = QListWidget()
        self._reasoning_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: none;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #2d2d2d;
            }
        """)
        self._reasoning_list.setMaximumHeight(100)
        reasoning_layout.addWidget(self._reasoning_list)
        layout.addWidget(reasoning_group)

        # Risk assessment
        risk_group = QGroupBox("Risk Assessment")
        risk_group.setStyleSheet(query_group.styleSheet())
        risk_layout = QGridLayout(risk_group)

        risk_layout.addWidget(QLabel("Level:"), 0, 0)
        self._risk_level = QLabel("--")
        self._risk_level.setStyleSheet("font-weight: bold;")
        risk_layout.addWidget(self._risk_level, 0, 1)

        risk_layout.addWidget(QLabel("Probability:"), 0, 2)
        self._risk_prob = QLabel("--")
        risk_layout.addWidget(self._risk_prob, 0, 3)

        risk_layout.addWidget(QLabel("Worst Case:"), 1, 0)
        self._worst_case = QLabel("--")
        self._worst_case.setWordWrap(True)
        risk_layout.addWidget(self._worst_case, 1, 1, 1, 3)

        layout.addWidget(risk_group)

        # Alternatives
        alt_group = QGroupBox("Alternatives Considered")
        alt_group.setStyleSheet(query_group.styleSheet())
        alt_layout = QVBoxLayout(alt_group)
        self._alternatives_list = QListWidget()
        self._alternatives_list.setStyleSheet(self._reasoning_list.styleSheet())
        self._alternatives_list.setMaximumHeight(80)
        alt_layout.addWidget(self._alternatives_list)
        layout.addWidget(alt_group)

        layout.addStretch()

    def set_decision(self, decision: dict[str, Any]) -> None:
        """Update panel with decision data."""
        if not PYSIDE_AVAILABLE:
            return

        self._agent_label.setText(decision.get("agent_name", "--"))
        self._type_label.setText(decision.get("decision_type", "--"))
        self._decision_label.setText(decision.get("decision", "--"))

        confidence = decision.get("confidence", 0)
        self._confidence_label.setText(f"{confidence:.1%}")
        if confidence >= 0.7:
            self._confidence_label.setStyleSheet("color: #28a745;")
        elif confidence >= 0.5:
            self._confidence_label.setStyleSheet("color: #ffc107;")
        else:
            self._confidence_label.setStyleSheet("color: #dc3545;")

        outcome = decision.get("outcome", "pending")
        outcome_colors = {
            "executed": "#28a745",
            "rejected": "#dc3545",
            "pending": "#ffc107",
            "cancelled": "#888",
            "timed_out": "#dc3545",
        }
        color = outcome_colors.get(outcome, "#888")
        self._outcome_label.setText(outcome.replace("_", " ").title())
        self._outcome_label.setStyleSheet(f"color: {color};")

        exec_time = decision.get("execution_time_ms", 0)
        self._exec_time_label.setText(f"{exec_time:.0f}ms")

        self._query_text.setPlainText(decision.get("query", ""))

        # Reasoning chain
        self._reasoning_list.clear()
        for step in decision.get("reasoning_chain", []):
            if isinstance(step, dict):
                text = f"{step.get('step_number', 0)}. {step.get('thought', '')}"
                conf = step.get("confidence", 0.5)
                item = QListWidgetItem(f"{text} ({conf:.0%})")
            else:
                item = QListWidgetItem(str(step))
            self._reasoning_list.addItem(item)

        # Risk assessment
        risk = decision.get("risk_assessment", {})
        level = risk.get("overall_level", "unknown")
        level_colors = {
            "low": "#28a745",
            "medium": "#ffc107",
            "high": "#dc3545",
            "critical": "#ff0000",
        }
        self._risk_level.setText(level.upper())
        self._risk_level.setStyleSheet(f"color: {level_colors.get(level, '#888')}; font-weight: bold;")

        prob = risk.get("probability_of_loss", 0)
        self._risk_prob.setText(f"{prob:.1%}")

        self._worst_case.setText(risk.get("worst_case_scenario", "--"))

        # Alternatives
        self._alternatives_list.clear()
        for alt in decision.get("alternatives_considered", []):
            if isinstance(alt, dict):
                text = f"• {alt.get('description', '')} (Rejected: {alt.get('reason_rejected', '')})"
            else:
                text = f"• {alt}"
            self._alternatives_list.addItem(text)

    def clear(self) -> None:
        """Clear the panel."""
        if not PYSIDE_AVAILABLE:
            return

        self._agent_label.setText("--")
        self._type_label.setText("--")
        self._decision_label.setText("--")
        self._confidence_label.setText("--")
        self._outcome_label.setText("--")
        self._exec_time_label.setText("--")
        self._query_text.clear()
        self._reasoning_list.clear()
        self._risk_level.setText("--")
        self._risk_prob.setText("--")
        self._worst_case.setText("--")
        self._alternatives_list.clear()


class DecisionLogViewer(QWidget if PYSIDE_AVAILABLE else object):
    """
    Widget for browsing agent decision history.

    Shows:
    - Filterable decision table (agent, type, outcome, date)
    - Decision detail panel
    - Reasoning chain visualization
    - Risk assessment display

    Example usage:
        from llm.decision_logger import DecisionLogger
        logger = DecisionLogger()
        viewer = DecisionLogViewer(decision_logger=logger)
        dashboard.add_widget(viewer)
    """

    # Signals
    decision_selected = Signal(str) if PYSIDE_AVAILABLE else None

    def __init__(
        self,
        decision_logger: Any | None = None,
        parent: QWidget | None = None,
    ):
        """
        Initialize decision log viewer.

        Args:
            decision_logger: DecisionLogger instance
            parent: Parent widget
        """
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self.logger = decision_logger
        self._decisions: list[dict[str, Any]] = []
        self._filtered_decisions: list[dict[str, Any]] = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header
        header_layout = QHBoxLayout()
        title = QLabel("Decision Log")
        title.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        # Refresh button
        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #4d4d4d; }
        """)
        refresh_btn.clicked.connect(self.refresh)
        header_layout.addWidget(refresh_btn)

        layout.addLayout(header_layout)

        # Filters
        filter_frame = QFrame()
        filter_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        filter_layout = QHBoxLayout(filter_frame)

        # Agent filter
        filter_layout.addWidget(QLabel("Agent:"))
        self._agent_filter = QComboBox()
        self._agent_filter.addItem("All Agents")
        self._agent_filter.setMinimumWidth(120)
        self._agent_filter.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #3d3d3d;
                padding: 4px;
                border-radius: 4px;
            }
        """)
        self._agent_filter.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self._agent_filter)

        # Outcome filter
        filter_layout.addWidget(QLabel("Outcome:"))
        self._outcome_filter = QComboBox()
        self._outcome_filter.addItems(["All", "Executed", "Pending", "Rejected", "Cancelled"])
        self._outcome_filter.setStyleSheet(self._agent_filter.styleSheet())
        self._outcome_filter.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self._outcome_filter)

        # Type filter
        filter_layout.addWidget(QLabel("Type:"))
        self._type_filter = QComboBox()
        self._type_filter.addItems(
            ["All", "Trade", "Analysis", "Risk Assessment", "Strategy Selection", "Position Sizing", "Exit Signal"]
        )
        self._type_filter.setStyleSheet(self._agent_filter.styleSheet())
        self._type_filter.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self._type_filter)

        # Date filter
        filter_layout.addWidget(QLabel("From:"))
        self._date_filter = QDateEdit()
        self._date_filter.setDate(QDate.currentDate().addDays(-7))
        self._date_filter.setCalendarPopup(True)
        self._date_filter.setStyleSheet("""
            QDateEdit {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #3d3d3d;
                padding: 4px;
                border-radius: 4px;
            }
        """)
        self._date_filter.dateChanged.connect(self._apply_filters)
        filter_layout.addWidget(self._date_filter)

        # Search
        filter_layout.addWidget(QLabel("Search:"))
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search decisions...")
        self._search_input.setStyleSheet("""
            QLineEdit {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #3d3d3d;
                padding: 4px;
                border-radius: 4px;
            }
        """)
        self._search_input.textChanged.connect(self._apply_filters)
        filter_layout.addWidget(self._search_input)

        filter_layout.addStretch()
        layout.addWidget(filter_frame)

        # Main content splitter
        splitter = QSplitter(Qt.Vertical)

        # Decision table
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(["Timestamp", "Agent", "Type", "Decision", "Confidence", "Outcome"])
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                gridline-color: #3d3d3d;
                border: none;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #0066cc;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #e0e0e0;
                padding: 6px;
                border: 1px solid #3d3d3d;
            }
        """)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)

        table_layout.addWidget(self._table)

        # Status bar
        self._status_label = QLabel("0 decisions")
        self._status_label.setStyleSheet("color: #888;")
        table_layout.addWidget(self._status_label)

        splitter.addWidget(table_widget)

        # Detail panel
        self._detail_panel = DecisionDetailPanel()
        splitter.addWidget(self._detail_panel)

        splitter.setSizes([300, 400])
        layout.addWidget(splitter)

    def _on_selection_changed(self) -> None:
        """Handle table selection change."""
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            self._detail_panel.clear()
            return

        row_idx = rows[0].row()
        if row_idx < len(self._filtered_decisions):
            decision = self._filtered_decisions[row_idx]
            self._detail_panel.set_decision(decision)

            if self.decision_selected:
                self.decision_selected.emit(decision.get("log_id", ""))

    def _apply_filters(self) -> None:
        """Apply filters to decision list."""
        agent = self._agent_filter.currentText()
        outcome = self._outcome_filter.currentText().lower()
        dtype = self._type_filter.currentText().lower().replace(" ", "_")
        date_from = self._date_filter.date().toPython()
        search = self._search_input.text().lower()

        self._filtered_decisions = []

        for dec in self._decisions:
            # Agent filter
            if agent != "All Agents" and dec.get("agent_name") != agent:
                continue

            # Outcome filter
            if outcome != "all" and dec.get("outcome", "").lower() != outcome:
                continue

            # Type filter
            if dtype != "all" and dec.get("decision_type", "").lower() != dtype:
                continue

            # Date filter
            timestamp = dec.get("timestamp", "")
            if isinstance(timestamp, str) and len(timestamp) >= 10:
                try:
                    dec_date = datetime.fromisoformat(timestamp[:10]).date()
                    if dec_date < date_from:
                        continue
                except ValueError:
                    pass

            # Search filter
            if search:
                searchable = " ".join(
                    [
                        str(dec.get("decision", "")),
                        str(dec.get("query", "")),
                        str(dec.get("agent_name", "")),
                    ]
                ).lower()
                if search not in searchable:
                    continue

            self._filtered_decisions.append(dec)

        self._update_table()

    def _update_table(self) -> None:
        """Update table with filtered decisions."""
        self._table.setRowCount(0)

        for dec in self._filtered_decisions:
            row = self._table.rowCount()
            self._table.insertRow(row)

            # Timestamp
            timestamp = dec.get("timestamp", "")
            if isinstance(timestamp, str) and len(timestamp) >= 19:
                timestamp = timestamp[:19].replace("T", " ")
            self._table.setItem(row, 0, QTableWidgetItem(timestamp))

            # Agent
            self._table.setItem(row, 1, QTableWidgetItem(dec.get("agent_name", "")))

            # Type
            dtype = dec.get("decision_type", "").replace("_", " ").title()
            self._table.setItem(row, 2, QTableWidgetItem(dtype))

            # Decision
            decision = dec.get("decision", "")[:50]
            self._table.setItem(row, 3, QTableWidgetItem(decision))

            # Confidence
            conf = dec.get("confidence", 0)
            conf_item = QTableWidgetItem(f"{conf:.0%}")
            if conf >= 0.7:
                conf_item.setForeground(QColor("#28a745"))
            elif conf >= 0.5:
                conf_item.setForeground(QColor("#ffc107"))
            else:
                conf_item.setForeground(QColor("#dc3545"))
            self._table.setItem(row, 4, conf_item)

            # Outcome
            outcome = dec.get("outcome", "pending").replace("_", " ").title()
            outcome_item = QTableWidgetItem(outcome)
            outcome_colors = {
                "Executed": "#28a745",
                "Pending": "#ffc107",
                "Rejected": "#dc3545",
                "Cancelled": "#888",
            }
            color = outcome_colors.get(outcome, "#888")
            outcome_item.setForeground(QColor(color))
            self._table.setItem(row, 5, outcome_item)

        self._status_label.setText(f"{len(self._filtered_decisions)} of {len(self._decisions)} decisions")

    def _update_agent_filter(self) -> None:
        """Update agent filter options."""
        current = self._agent_filter.currentText()
        self._agent_filter.blockSignals(True)
        self._agent_filter.clear()
        self._agent_filter.addItem("All Agents")

        agents = set()
        for dec in self._decisions:
            agent = dec.get("agent_name")
            if agent:
                agents.add(agent)

        for agent in sorted(agents):
            self._agent_filter.addItem(agent)

        idx = self._agent_filter.findText(current)
        if idx >= 0:
            self._agent_filter.setCurrentIndex(idx)

        self._agent_filter.blockSignals(False)

    def refresh(self) -> None:
        """Refresh from decision logger."""
        if not PYSIDE_AVAILABLE:
            return

        if self.logger:
            self._decisions = [log.to_dict() for log in self.logger.logs]
        self._update_agent_filter()
        self._apply_filters()

    def load_decisions(self, decisions: list[dict[str, Any]]) -> None:
        """
        Load decisions directly.

        Args:
            decisions: List of AgentDecisionLog.to_dict() outputs
        """
        if not PYSIDE_AVAILABLE:
            return

        self._decisions = decisions
        self._update_agent_filter()
        self._apply_filters()

    def add_decision(self, decision: dict[str, Any]) -> None:
        """
        Add a single decision.

        Args:
            decision: AgentDecisionLog.to_dict() output
        """
        if not PYSIDE_AVAILABLE:
            return

        self._decisions.insert(0, decision)
        self._update_agent_filter()
        self._apply_filters()

    def set_decision_logger(self, logger: Any) -> None:
        """
        Set or update the decision logger.

        Args:
            logger: DecisionLogger instance
        """
        self.logger = logger
        self.refresh()


def create_decision_log_viewer(
    decision_logger: Any | None = None,
) -> DecisionLogViewer:
    """
    Factory function to create a decision log viewer widget.

    Args:
        decision_logger: DecisionLogger instance

    Returns:
        Configured DecisionLogViewer
    """
    return DecisionLogViewer(decision_logger=decision_logger)


# Stub implementations when PySide6 not available
if not PYSIDE_AVAILABLE:

    class DecisionDetailPanel:
        def __init__(self, *args, **kwargs):
            pass

        def set_decision(self, decision):
            pass

        def clear(self):
            pass

    class DecisionLogViewer:
        def __init__(self, *args, **kwargs):
            pass

        def refresh(self):
            pass

        def load_decisions(self, decisions):
            pass

        def add_decision(self, decision):
            pass

        def set_decision_logger(self, logger):
            pass


__all__ = [
    "DecisionDetailPanel",
    "DecisionLogViewer",
    "create_decision_log_viewer",
]
