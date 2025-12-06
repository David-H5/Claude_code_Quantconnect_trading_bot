"""
Reasoning Chain Viewer Widget for Trading Dashboard (UPGRADE-010 Sprint 3)

Visualizes AI agent reasoning chains with:
- Chain list view with filtering
- Step-by-step visualization with confidence bars
- Search by agent/task/content
- Export to compliance format

QuantConnect Compatible: N/A (UI component)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QColor, QFont
    from PySide6.QtWidgets import (
        QComboBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QProgressBar,
        QPushButton,
        QScrollArea,
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


@dataclass
class ReasoningStepDisplay:
    """Display data for a reasoning step."""

    step_number: int
    thought: str
    evidence: str | None
    confidence: float
    metadata: dict[str, Any]


@dataclass
class ReasoningChainDisplay:
    """Display data for a reasoning chain."""

    chain_id: str
    agent_name: str
    task: str
    started_at: datetime
    completed_at: datetime | None
    status: str
    steps: list[ReasoningStepDisplay]
    final_decision: str | None
    final_confidence: float
    duration_ms: float
    average_confidence: float


class ReasoningStepWidget(QWidget if PYSIDE_AVAILABLE else object):
    """Widget for displaying a single reasoning step."""

    def __init__(
        self,
        step: ReasoningStepDisplay,
        parent: QWidget | None = None,
    ):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self._step = step
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the step widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # Header with step number and confidence bar
        header_layout = QHBoxLayout()

        step_label = QLabel(f"Step {self._step.step_number}")
        step_label.setStyleSheet("color: #17a2b8; font-weight: bold; font-size: 12px;")
        header_layout.addWidget(step_label)

        header_layout.addStretch()

        # Confidence bar
        conf_label = QLabel(f"{self._step.confidence:.0%}")
        conf_label.setStyleSheet("color: #888; font-size: 11px;")
        header_layout.addWidget(conf_label)

        progress = QProgressBar()
        progress.setMaximum(100)
        progress.setValue(int(self._step.confidence * 100))
        progress.setFixedWidth(80)
        progress.setFixedHeight(12)
        progress.setTextVisible(False)

        # Color based on confidence
        if self._step.confidence >= 0.8:
            color = "#28a745"  # Green
        elif self._step.confidence >= 0.5:
            color = "#ffc107"  # Yellow
        else:
            color = "#dc3545"  # Red

        progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: #2d2d2d;
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 4px;
            }}
        """)
        header_layout.addWidget(progress)

        layout.addLayout(header_layout)

        # Thought content
        thought_label = QLabel(self._step.thought)
        thought_label.setWordWrap(True)
        thought_label.setStyleSheet("color: white; padding-left: 16px;")
        layout.addWidget(thought_label)

        # Evidence if present
        if self._step.evidence:
            evidence_label = QLabel(f"Evidence: {self._step.evidence}")
            evidence_label.setWordWrap(True)
            evidence_label.setStyleSheet("color: #6c757d; font-style: italic; padding-left: 16px;")
            layout.addWidget(evidence_label)

        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #3d3d3d;")
        layout.addWidget(line)


class ChainDetailPanel(QWidget if PYSIDE_AVAILABLE else object):
    """Panel showing detailed view of a reasoning chain."""

    def __init__(self, parent: QWidget | None = None):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self._setup_ui()
        self._current_chain: ReasoningChainDisplay | None = None

    def _setup_ui(self) -> None:
        """Set up the detail panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QLabel("Reasoning Chain Details")
        header.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        layout.addWidget(header)

        # Info grid
        info_grid = QGridLayout()

        info_grid.addWidget(QLabel("Agent:"), 0, 0)
        self._agent_label = QLabel("--")
        self._agent_label.setStyleSheet("color: #17a2b8; font-weight: bold;")
        info_grid.addWidget(self._agent_label, 0, 1)

        info_grid.addWidget(QLabel("Status:"), 0, 2)
        self._status_label = QLabel("--")
        info_grid.addWidget(self._status_label, 0, 3)

        info_grid.addWidget(QLabel("Task:"), 1, 0)
        self._task_label = QLabel("--")
        self._task_label.setWordWrap(True)
        info_grid.addWidget(self._task_label, 1, 1, 1, 3)

        info_grid.addWidget(QLabel("Duration:"), 2, 0)
        self._duration_label = QLabel("--")
        info_grid.addWidget(self._duration_label, 2, 1)

        info_grid.addWidget(QLabel("Avg Confidence:"), 2, 2)
        self._avg_conf_label = QLabel("--")
        info_grid.addWidget(self._avg_conf_label, 2, 3)

        layout.addLayout(info_grid)

        # Final decision group
        decision_group = QGroupBox("Final Decision")
        decision_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
        """)
        decision_layout = QHBoxLayout(decision_group)

        self._decision_label = QLabel("--")
        self._decision_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        decision_layout.addWidget(self._decision_label)

        decision_layout.addStretch()

        self._final_conf_bar = QProgressBar()
        self._final_conf_bar.setMaximum(100)
        self._final_conf_bar.setFixedWidth(120)
        self._final_conf_bar.setFixedHeight(16)
        decision_layout.addWidget(self._final_conf_bar)

        layout.addWidget(decision_group)

        # Steps group with scroll area
        steps_group = QGroupBox("Reasoning Steps")
        steps_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
        """)
        steps_layout = QVBoxLayout(steps_group)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: #1e1e1e; border: none;")

        self._steps_container = QWidget()
        self._steps_layout = QVBoxLayout(self._steps_container)
        self._steps_layout.setContentsMargins(0, 0, 0, 0)
        self._steps_layout.setSpacing(0)

        scroll_area.setWidget(self._steps_container)
        steps_layout.addWidget(scroll_area)

        layout.addWidget(steps_group, stretch=1)

    def set_chain(self, chain: ReasoningChainDisplay) -> None:
        """Display a reasoning chain."""
        if not PYSIDE_AVAILABLE:
            return

        self._current_chain = chain

        # Update info
        self._agent_label.setText(chain.agent_name)
        self._task_label.setText(chain.task)
        self._duration_label.setText(f"{chain.duration_ms:.0f} ms")
        self._avg_conf_label.setText(f"{chain.average_confidence:.0%}")

        # Status with color
        status_colors = {
            "completed": "#28a745",
            "in_progress": "#17a2b8",
            "failed": "#dc3545",
            "abandoned": "#6c757d",
        }
        color = status_colors.get(chain.status, "#888")
        self._status_label.setText(chain.status.upper())
        self._status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        # Final decision
        if chain.final_decision:
            self._decision_label.setText(chain.final_decision)
            self._final_conf_bar.setValue(int(chain.final_confidence * 100))

            if chain.final_confidence >= 0.8:
                bar_color = "#28a745"
            elif chain.final_confidence >= 0.5:
                bar_color = "#ffc107"
            else:
                bar_color = "#dc3545"

            self._final_conf_bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: #2d2d2d;
                    border: none;
                    border-radius: 4px;
                }}
                QProgressBar::chunk {{
                    background-color: {bar_color};
                    border-radius: 4px;
                }}
            """)
        else:
            self._decision_label.setText("Pending...")
            self._final_conf_bar.setValue(0)

        # Clear and rebuild steps
        while self._steps_layout.count():
            item = self._steps_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for step in chain.steps:
            step_widget = ReasoningStepWidget(step)
            self._steps_layout.addWidget(step_widget)

        self._steps_layout.addStretch()


class ReasoningViewerWidget(QWidget if PYSIDE_AVAILABLE else object):
    """
    Widget for viewing and analyzing AI reasoning chains.

    Features:
    - Chain list with filtering
    - Detailed step-by-step view
    - Search by agent/task/content
    - Export to compliance format
    """

    chain_selected = Signal(str) if PYSIDE_AVAILABLE else None

    def __init__(self, parent: QWidget | None = None):
        if not PYSIDE_AVAILABLE:
            self._chains: list[ReasoningChainDisplay] = []
            self._filtered_chains: list[ReasoningChainDisplay] = []
            return

        super().__init__(parent)
        self._chains: list[ReasoningChainDisplay] = []
        self._filtered_chains: list[ReasoningChainDisplay] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the main UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Filter bar
        filter_layout = QHBoxLayout()

        filter_layout.addWidget(QLabel("Agent:"))
        self._agent_filter = QComboBox()
        self._agent_filter.addItem("All Agents")
        self._agent_filter.currentTextChanged.connect(self._apply_filters)
        self._agent_filter.setMinimumWidth(150)
        filter_layout.addWidget(self._agent_filter)

        filter_layout.addWidget(QLabel("Status:"))
        self._status_filter = QComboBox()
        self._status_filter.addItems(["All", "Completed", "In Progress", "Failed", "Abandoned"])
        self._status_filter.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self._status_filter)

        filter_layout.addWidget(QLabel("Search:"))
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search task or content...")
        self._search_input.textChanged.connect(self._apply_filters)
        filter_layout.addWidget(self._search_input, stretch=1)

        self._export_btn = QPushButton("Export")
        self._export_btn.clicked.connect(self._export_audit_trail)
        filter_layout.addWidget(self._export_btn)

        layout.addLayout(filter_layout)

        # Splitter for list and detail
        splitter = QSplitter(Qt.Horizontal)

        # Chain list
        list_container = QWidget()
        list_layout = QVBoxLayout(list_container)
        list_layout.setContentsMargins(0, 0, 0, 0)

        list_header = QLabel("Reasoning Chains")
        list_header.setStyleSheet("color: white; font-weight: bold; padding: 4px;")
        list_layout.addWidget(list_header)

        self._chain_table = QTableWidget()
        self._chain_table.setColumnCount(5)
        self._chain_table.setHorizontalHeaderLabels(["Agent", "Task", "Steps", "Confidence", "Status"])
        self._chain_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._chain_table.setSelectionMode(QTableWidget.SingleSelection)
        self._chain_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._chain_table.cellClicked.connect(self._on_chain_selected)
        self._chain_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: white;
                gridline-color: #3d3d3d;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: white;
                padding: 4px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #17a2b8;
            }
        """)
        list_layout.addWidget(self._chain_table)

        splitter.addWidget(list_container)

        # Detail panel
        self._detail_panel = ChainDetailPanel()
        splitter.addWidget(self._detail_panel)

        splitter.setSizes([400, 600])
        layout.addWidget(splitter)

    def add_chain(self, chain_data: dict[str, Any]) -> None:
        """Add a reasoning chain from dictionary data."""
        chain = self._parse_chain(chain_data)
        self._chains.append(chain)
        self._update_agent_filter()
        self._apply_filters()

    def set_chains(self, chains_data: list[dict[str, Any]]) -> None:
        """Set all chains from list of dictionaries."""
        self._chains = [self._parse_chain(c) for c in chains_data]
        self._update_agent_filter()
        self._apply_filters()

    def _parse_chain(self, data: dict[str, Any]) -> ReasoningChainDisplay:
        """Parse chain dictionary into display object."""
        steps = [
            ReasoningStepDisplay(
                step_number=s.get("step_number", i + 1),
                thought=s.get("thought", ""),
                evidence=s.get("evidence"),
                confidence=s.get("confidence", 0.5),
                metadata=s.get("metadata", {}),
            )
            for i, s in enumerate(data.get("steps", []))
        ]

        started_at = data.get("started_at")
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)
        elif started_at is None:
            started_at = datetime.utcnow()

        completed_at = data.get("completed_at")
        if isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at)

        return ReasoningChainDisplay(
            chain_id=data.get("chain_id", ""),
            agent_name=data.get("agent_name", "unknown"),
            task=data.get("task", ""),
            started_at=started_at,
            completed_at=completed_at,
            status=data.get("status", "in_progress"),
            steps=steps,
            final_decision=data.get("final_decision"),
            final_confidence=data.get("final_confidence", 0.0),
            duration_ms=data.get("duration_ms", 0.0),
            average_confidence=data.get("average_confidence", 0.0),
        )

    def _update_agent_filter(self) -> None:
        """Update agent filter dropdown."""
        if not PYSIDE_AVAILABLE:
            return

        agents = sorted(set(c.agent_name for c in self._chains))
        current = self._agent_filter.currentText()

        self._agent_filter.clear()
        self._agent_filter.addItem("All Agents")
        self._agent_filter.addItems(agents)

        if current in agents:
            self._agent_filter.setCurrentText(current)

    def _apply_filters(self) -> None:
        """Apply filters to chain list."""
        if not PYSIDE_AVAILABLE:
            return

        agent_filter = self._agent_filter.currentText()
        status_filter = self._status_filter.currentText().lower().replace(" ", "_")
        search_text = self._search_input.text().lower()

        self._filtered_chains = []
        for chain in self._chains:
            # Agent filter
            if agent_filter != "All Agents" and chain.agent_name != agent_filter:
                continue

            # Status filter
            if status_filter != "all" and chain.status != status_filter:
                continue

            # Search filter
            if search_text:
                searchable = f"{chain.task} {chain.final_decision or ''}"
                for step in chain.steps:
                    searchable += f" {step.thought} {step.evidence or ''}"
                if search_text not in searchable.lower():
                    continue

            self._filtered_chains.append(chain)

        self._refresh_table()

    def _refresh_table(self) -> None:
        """Refresh the chain table."""
        if not PYSIDE_AVAILABLE:
            return

        self._chain_table.setRowCount(len(self._filtered_chains))

        for row, chain in enumerate(self._filtered_chains):
            # Agent
            agent_item = QTableWidgetItem(chain.agent_name)
            agent_item.setForeground(QColor("#17a2b8"))
            self._chain_table.setItem(row, 0, agent_item)

            # Task
            task_item = QTableWidgetItem(chain.task[:50] + "..." if len(chain.task) > 50 else chain.task)
            self._chain_table.setItem(row, 1, task_item)

            # Steps
            steps_item = QTableWidgetItem(str(len(chain.steps)))
            steps_item.setTextAlignment(Qt.AlignCenter)
            self._chain_table.setItem(row, 2, steps_item)

            # Confidence
            conf_item = QTableWidgetItem(f"{chain.average_confidence:.0%}")
            conf_item.setTextAlignment(Qt.AlignCenter)
            if chain.average_confidence >= 0.8:
                conf_item.setForeground(QColor("#28a745"))
            elif chain.average_confidence >= 0.5:
                conf_item.setForeground(QColor("#ffc107"))
            else:
                conf_item.setForeground(QColor("#dc3545"))
            self._chain_table.setItem(row, 3, conf_item)

            # Status
            status_item = QTableWidgetItem(chain.status.upper())
            status_colors = {
                "completed": "#28a745",
                "in_progress": "#17a2b8",
                "failed": "#dc3545",
                "abandoned": "#6c757d",
            }
            status_item.setForeground(QColor(status_colors.get(chain.status, "#888")))
            self._chain_table.setItem(row, 4, status_item)

    def _on_chain_selected(self, row: int, column: int) -> None:
        """Handle chain selection."""
        if not PYSIDE_AVAILABLE or row >= len(self._filtered_chains):
            return

        chain = self._filtered_chains[row]
        self._detail_panel.set_chain(chain)

        if self.chain_selected:
            self.chain_selected.emit(chain.chain_id)

    def _export_audit_trail(self) -> None:
        """Export chains to audit trail format."""
        from datetime import datetime

        audit_data = []
        for chain in self._filtered_chains:
            audit_data.append(
                {
                    "timestamp": chain.started_at.isoformat(),
                    "agent_name": chain.agent_name,
                    "task": chain.task,
                    "chain_id": chain.chain_id,
                    "step_count": len(chain.steps),
                    "final_decision": chain.final_decision,
                    "confidence": chain.final_confidence,
                    "duration_ms": chain.duration_ms,
                    "status": chain.status,
                }
            )

        # In production, would open file dialog
        filename = f"reasoning_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        print(f"Would export {len(audit_data)} chains to {filename}")

    def get_chain_count(self) -> int:
        """Get total number of chains."""
        return len(self._chains)

    def get_filtered_count(self) -> int:
        """Get number of filtered chains."""
        return len(self._filtered_chains)


def create_reasoning_viewer(
    parent: Any | None = None,
) -> ReasoningViewerWidget:
    """
    Factory function to create a reasoning viewer widget.

    Args:
        parent: Parent widget

    Returns:
        ReasoningViewerWidget instance
    """
    return ReasoningViewerWidget(parent=parent)
