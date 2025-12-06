"""
Evolution Monitor Widget for Trading Dashboard

Tracks and visualizes self-evolving agent progress:
- Evolution cycle history
- Prompt version comparison
- Score improvement trends
- Convergence status

Author: QuantConnect Trading Bot
Date: 2025-12-01
UPGRADE-006: LLM Dashboard Integration
"""

from __future__ import annotations

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
        QTabWidget,
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


class EvolutionProgressWidget(QWidget if PYSIDE_AVAILABLE else object):
    """Widget showing evolution progress with visual indicators."""

    def __init__(self, parent: QWidget | None = None):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the progress UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Progress bar
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Evolution Progress:"))

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("Cycle %v of %m")
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #0066cc;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self._progress_bar)

        layout.addLayout(progress_layout)

        # Stats grid
        stats_grid = QGridLayout()

        stats_grid.addWidget(QLabel("Initial Score:"), 0, 0)
        self._initial_score = QLabel("--")
        self._initial_score.setStyleSheet("color: #888;")
        stats_grid.addWidget(self._initial_score, 0, 1)

        stats_grid.addWidget(QLabel("Current Score:"), 0, 2)
        self._current_score = QLabel("--")
        self._current_score.setStyleSheet("color: #28a745; font-weight: bold;")
        stats_grid.addWidget(self._current_score, 0, 3)

        stats_grid.addWidget(QLabel("Target Score:"), 0, 4)
        self._target_score = QLabel("--")
        self._target_score.setStyleSheet("color: #17a2b8;")
        stats_grid.addWidget(self._target_score, 0, 5)

        stats_grid.addWidget(QLabel("Improvement:"), 1, 0)
        self._improvement = QLabel("--")
        stats_grid.addWidget(self._improvement, 1, 1)

        stats_grid.addWidget(QLabel("Status:"), 1, 2)
        self._status = QLabel("--")
        self._status.setStyleSheet("font-weight: bold;")
        stats_grid.addWidget(self._status, 1, 3, 1, 3)

        layout.addLayout(stats_grid)

    def update_progress(
        self,
        cycle: int,
        max_cycles: int,
        initial_score: float,
        current_score: float,
        target_score: float,
        status: str,
    ) -> None:
        """Update progress display."""
        if not PYSIDE_AVAILABLE:
            return

        self._progress_bar.setMaximum(max_cycles)
        self._progress_bar.setValue(cycle)

        self._initial_score.setText(f"{initial_score:.1%}")
        self._current_score.setText(f"{current_score:.1%}")
        self._target_score.setText(f"{target_score:.1%}")

        improvement = current_score - initial_score
        if improvement >= 0:
            self._improvement.setText(f"+{improvement:.1%}")
            self._improvement.setStyleSheet("color: #28a745;")
        else:
            self._improvement.setText(f"{improvement:.1%}")
            self._improvement.setStyleSheet("color: #dc3545;")

        # Status coloring
        status_colors = {
            "evolving": "#ffc107",
            "target_reached": "#28a745",
            "no_improvement": "#dc3545",
            "max_cycles_reached": "#ffc107",
            "regression_detected": "#dc3545",
            "complete": "#28a745",
        }
        color = status_colors.get(status.lower(), "#888")
        self._status.setText(status.replace("_", " ").title())
        self._status.setStyleSheet(f"color: {color}; font-weight: bold;")


class CycleHistoryWidget(QWidget if PYSIDE_AVAILABLE else object):
    """Widget showing evolution cycle history."""

    cycle_selected = Signal(int) if PYSIDE_AVAILABLE else None

    def __init__(self, parent: QWidget | None = None):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self._cycles: list[dict[str, Any]] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the cycle history UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["Cycle", "Pre-Score", "Post-Score", "Improvement", "Refinements"])
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
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)

        layout.addWidget(self._table)

    def _on_selection_changed(self) -> None:
        """Handle selection change."""
        rows = self._table.selectionModel().selectedRows()
        if rows and self.cycle_selected:
            self.cycle_selected.emit(rows[0].row())

    def set_cycles(self, cycles: list[dict[str, Any]]) -> None:
        """Set cycle history data."""
        if not PYSIDE_AVAILABLE:
            return

        self._cycles = cycles
        self._table.setRowCount(0)

        for cycle in cycles:
            row = self._table.rowCount()
            self._table.insertRow(row)

            # Cycle number
            self._table.setItem(row, 0, QTableWidgetItem(str(cycle.get("cycle_number", 0))))

            # Pre-score
            pre = cycle.get("pre_score", 0)
            self._table.setItem(row, 1, QTableWidgetItem(f"{pre:.1%}"))

            # Post-score
            post = cycle.get("post_score", 0)
            post_item = QTableWidgetItem(f"{post:.1%}")
            self._table.setItem(row, 2, post_item)

            # Improvement
            imp = cycle.get("improvement", 0)
            imp_item = QTableWidgetItem(f"{imp:+.1%}")
            if imp > 0:
                imp_item.setForeground(QColor("#28a745"))
            elif imp < 0:
                imp_item.setForeground(QColor("#dc3545"))
            self._table.setItem(row, 3, imp_item)

            # Refinements count
            refinements = cycle.get("refinements_applied", [])
            self._table.setItem(row, 4, QTableWidgetItem(str(len(refinements))))


class PromptVersionWidget(QWidget if PYSIDE_AVAILABLE else object):
    """Widget for viewing and comparing prompt versions."""

    def __init__(self, parent: QWidget | None = None):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self._versions: list[dict[str, Any]] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the prompt version UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Version selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Version:"))

        self._version_combo = QComboBox()
        self._version_combo.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3d3d3d;
                padding: 5px;
                border-radius: 4px;
            }
        """)
        self._version_combo.currentIndexChanged.connect(self._on_version_changed)
        selector_layout.addWidget(self._version_combo)

        self._score_label = QLabel("Score: --")
        self._score_label.setStyleSheet("color: #888; margin-left: 10px;")
        selector_layout.addWidget(self._score_label)

        selector_layout.addStretch()
        layout.addLayout(selector_layout)

        # Prompt text
        self._prompt_text = QTextEdit()
        self._prompt_text.setReadOnly(True)
        self._prompt_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
            }
        """)
        layout.addWidget(self._prompt_text)

    def _on_version_changed(self, index: int) -> None:
        """Handle version selection change."""
        if index < 0 or index >= len(self._versions):
            return

        version = self._versions[index]
        self._prompt_text.setPlainText(version.get("prompt", ""))
        score = version.get("score", 0)
        self._score_label.setText(f"Score: {score:.1%}")

    def set_versions(self, versions: list[dict[str, Any]]) -> None:
        """Set prompt versions data."""
        if not PYSIDE_AVAILABLE:
            return

        self._versions = versions

        self._version_combo.blockSignals(True)
        self._version_combo.clear()

        for v in versions:
            ver = v.get("version", 0)
            score = v.get("score", 0)
            self._version_combo.addItem(f"v{ver} ({score:.1%})")

        self._version_combo.blockSignals(False)

        if versions:
            self._version_combo.setCurrentIndex(len(versions) - 1)
            self._on_version_changed(len(versions) - 1)


class EvolutionMonitor(QWidget if PYSIDE_AVAILABLE else object):
    """
    Widget for monitoring agent evolution progress.

    Shows:
    - Evolution status and progress
    - Cycle-by-cycle improvement history
    - Prompt version comparison
    - Refinement details

    Example usage:
        from llm.self_evolving_agent import SelfEvolvingAgent
        evolving_agents = [agent1, agent2]
        monitor = EvolutionMonitor(evolving_agents=evolving_agents)
        dashboard.add_widget(monitor)
    """

    # Signals
    agent_selected = Signal(str) if PYSIDE_AVAILABLE else None

    def __init__(
        self,
        evolving_agents: list[Any] | None = None,
        parent: QWidget | None = None,
    ):
        """
        Initialize evolution monitor.

        Args:
            evolving_agents: List of SelfEvolvingAgent instances
            parent: Parent widget
        """
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self.agents = evolving_agents or []
        self._current_agent: Any | None = None
        self._evolution_results: dict[str, dict[str, Any]] = {}

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header with agent selector
        header_layout = QHBoxLayout()

        title = QLabel("Evolution Monitor")
        title.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        header_layout.addWidget(QLabel("Agent:"))
        self._agent_combo = QComboBox()
        self._agent_combo.setMinimumWidth(150)
        self._agent_combo.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3d3d3d;
                padding: 5px;
                border-radius: 4px;
            }
        """)
        self._agent_combo.currentTextChanged.connect(self._on_agent_changed)
        header_layout.addWidget(self._agent_combo)

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
            QPushButton:hover { background-color: #4d4d4d; }
        """)
        refresh_btn.clicked.connect(self.refresh)
        header_layout.addWidget(refresh_btn)

        layout.addLayout(header_layout)

        # Progress section
        self._progress_widget = EvolutionProgressWidget()
        layout.addWidget(self._progress_widget)

        # Tabs for cycle history and prompt versions
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #888;
                padding: 8px 16px;
                border: 1px solid #3d3d3d;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #1e1e1e;
                color: white;
            }
        """)

        # Cycle history tab
        self._cycle_history = CycleHistoryWidget()
        self._cycle_history.cycle_selected.connect(self._on_cycle_selected)
        tabs.addTab(self._cycle_history, "Cycle History")

        # Prompt versions tab
        self._prompt_versions = PromptVersionWidget()
        tabs.addTab(self._prompt_versions, "Prompt Versions")

        # Refinements tab
        refinements_widget = QWidget()
        refinements_layout = QVBoxLayout(refinements_widget)

        self._refinements_text = QTextEdit()
        self._refinements_text.setReadOnly(True)
        self._refinements_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: none;
                padding: 8px;
            }
        """)
        refinements_layout.addWidget(self._refinements_text)
        tabs.addTab(refinements_widget, "Refinements")

        layout.addWidget(tabs)

        # Initialize agent combo
        self._update_agent_list()

    def _on_agent_changed(self, agent_name: str) -> None:
        """Handle agent selection change."""
        self._current_agent = agent_name
        if self.agent_selected:
            self.agent_selected.emit(agent_name)
        self._update_display()

    def _on_cycle_selected(self, cycle_index: int) -> None:
        """Handle cycle selection."""
        if not self._current_agent:
            return

        result = self._evolution_results.get(self._current_agent)
        if not result:
            return

        cycles = result.get("cycles", [])
        if cycle_index < len(cycles):
            cycle = cycles[cycle_index]
            refinements = cycle.get("refinements_applied", [])
            weaknesses = cycle.get("weaknesses_identified", [])

            text = f"Cycle {cycle.get('cycle_number', 0)} Details\n"
            text += "=" * 40 + "\n\n"
            text += f"Refinements Applied ({len(refinements)}):\n"
            for r in refinements:
                text += f"  • {r}\n"
            text += f"\nWeaknesses Identified ({len(weaknesses)}):\n"
            for w in weaknesses:
                text += f"  • {w}\n"

            self._refinements_text.setPlainText(text)

    def _update_agent_list(self) -> None:
        """Update agent combo box."""
        self._agent_combo.blockSignals(True)
        self._agent_combo.clear()

        if self.agents:
            for agent in self.agents:
                name = getattr(agent, "agent_name", str(agent))
                if hasattr(agent, "agent") and hasattr(agent.agent, "name"):
                    name = agent.agent.name
                self._agent_combo.addItem(name)

        if self._evolution_results:
            for name in self._evolution_results.keys():
                if self._agent_combo.findText(name) == -1:
                    self._agent_combo.addItem(name)

        if self._agent_combo.count() == 0:
            self._agent_combo.addItem("(no agents)")

        self._agent_combo.blockSignals(False)

        if self._agent_combo.count() > 0:
            self._current_agent = self._agent_combo.currentText()

    def _update_display(self) -> None:
        """Update display for current agent."""
        if not self._current_agent:
            return

        result = self._evolution_results.get(self._current_agent)
        if not result:
            self._progress_widget.update_progress(0, 5, 0, 0, 0.85, "No Data")
            return

        # Update progress
        cycles = result.get("cycles", [])
        self._progress_widget.update_progress(
            cycle=len(cycles),
            max_cycles=result.get("max_cycles", 5),
            initial_score=result.get("initial_score", 0),
            current_score=result.get("final_score", 0),
            target_score=result.get("target_score", 0.85),
            status=result.get("convergence_reason", "unknown"),
        )

        # Update cycle history
        self._cycle_history.set_cycles(cycles)

        # Update prompt versions
        versions = result.get("prompt_versions", [])
        self._prompt_versions.set_versions(versions)

    def add_evolution_result(
        self,
        agent_name: str,
        result: dict[str, Any],
    ) -> None:
        """
        Add an evolution result.

        Args:
            agent_name: Name of the agent
            result: EvolutionResult.to_dict() output
        """
        if not PYSIDE_AVAILABLE:
            return

        self._evolution_results[agent_name] = result

        # Add to combo if not present
        if self._agent_combo.findText(agent_name) == -1:
            self._agent_combo.addItem(agent_name)

        # Update display if this is current agent
        if self._current_agent == agent_name:
            self._update_display()

    def refresh(self) -> None:
        """Refresh the display."""
        if not PYSIDE_AVAILABLE:
            return

        self._update_agent_list()
        self._update_display()

    def set_evolving_agents(self, agents: list[Any]) -> None:
        """
        Set or update the evolving agents.

        Args:
            agents: List of SelfEvolvingAgent instances
        """
        self.agents = agents
        self._update_agent_list()


def create_evolution_monitor(
    evolving_agents: list[Any] | None = None,
) -> EvolutionMonitor:
    """
    Factory function to create an evolution monitor widget.

    Args:
        evolving_agents: List of SelfEvolvingAgent instances

    Returns:
        Configured EvolutionMonitor
    """
    return EvolutionMonitor(evolving_agents=evolving_agents)


# Stub implementations when PySide6 not available
if not PYSIDE_AVAILABLE:

    class EvolutionProgressWidget:
        def __init__(self, *args, **kwargs):
            pass

        def update_progress(self, *args, **kwargs):
            pass

    class CycleHistoryWidget:
        def __init__(self, *args, **kwargs):
            pass

        def set_cycles(self, cycles):
            pass

    class PromptVersionWidget:
        def __init__(self, *args, **kwargs):
            pass

        def set_versions(self, versions):
            pass

    class EvolutionMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def add_evolution_result(self, agent_name, result):
            pass

        def refresh(self):
            pass

        def set_evolving_agents(self, agents):
            pass


__all__ = [
    "CycleHistoryWidget",
    "EvolutionMonitor",
    "EvolutionProgressWidget",
    "PromptVersionWidget",
    "create_evolution_monitor",
]
