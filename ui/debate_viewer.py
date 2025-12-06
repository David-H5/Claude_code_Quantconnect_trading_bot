"""
Debate Viewer Widget for Trading Dashboard

Visualizes Bull/Bear debate sessions with:
- Side-by-side argument display
- Round-by-round navigation
- Outcome and consensus visualization
- Debate history browsing

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
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QScrollArea,
        QSplitter,
        QStackedWidget,
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


class ArgumentPanel(QWidget if PYSIDE_AVAILABLE else object):
    """Panel displaying one side's argument (Bull or Bear)."""

    def __init__(
        self,
        side: str,
        color: str,
        parent: QWidget | None = None,
    ):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self.side = side
        self.color = color
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the argument panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QLabel(f"🐂 {self.side}" if self.side == "Bull" else f"🐻 {self.side}")
        header.setStyleSheet(f"""
            color: {self.color};
            font-size: 16px;
            font-weight: bold;
        """)
        layout.addWidget(header)

        # Confidence bar
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence:"))
        self._confidence_label = QLabel("--")
        self._confidence_label.setStyleSheet(f"color: {self.color}; font-weight: bold;")
        confidence_layout.addWidget(self._confidence_label)
        confidence_layout.addStretch()
        layout.addLayout(confidence_layout)

        # Main argument text
        self._argument_text = QTextEdit()
        self._argument_text.setReadOnly(True)
        self._argument_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid {self.color};
                border-radius: 4px;
                padding: 8px;
            }}
        """)
        self._argument_text.setMinimumHeight(100)
        layout.addWidget(self._argument_text)

        # Key points
        points_group = QGroupBox("Key Points")
        points_group.setStyleSheet(f"""
            QGroupBox {{
                color: {self.color};
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }}
        """)
        points_layout = QVBoxLayout(points_group)
        self._points_list = QListWidget()
        self._points_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: none;
            }
            QListWidget::item {
                padding: 4px;
            }
        """)
        self._points_list.setMaximumHeight(80)
        points_layout.addWidget(self._points_list)
        layout.addWidget(points_group)

        # Evidence
        evidence_group = QGroupBox("Evidence")
        evidence_group.setStyleSheet(points_group.styleSheet())
        evidence_layout = QVBoxLayout(evidence_group)
        self._evidence_list = QListWidget()
        self._evidence_list.setStyleSheet(self._points_list.styleSheet())
        self._evidence_list.setMaximumHeight(60)
        evidence_layout.addWidget(self._evidence_list)
        layout.addWidget(evidence_group)

        # Risks identified
        risks_group = QGroupBox("Risks Identified")
        risks_group.setStyleSheet(points_group.styleSheet())
        risks_layout = QVBoxLayout(risks_group)
        self._risks_list = QListWidget()
        self._risks_list.setStyleSheet(self._points_list.styleSheet())
        self._risks_list.setMaximumHeight(60)
        risks_layout.addWidget(self._risks_list)
        layout.addWidget(risks_group)

    def set_argument(self, argument: dict[str, Any]) -> None:
        """Update the panel with argument data."""
        if not PYSIDE_AVAILABLE:
            return

        self._confidence_label.setText(f"{argument.get('confidence', 0):.1%}")
        self._argument_text.setPlainText(argument.get("content", ""))

        # Key points
        self._points_list.clear()
        for point in argument.get("key_points", []):
            self._points_list.addItem(f"• {point}")

        # Evidence
        self._evidence_list.clear()
        for evidence in argument.get("evidence", []):
            self._evidence_list.addItem(f"📎 {evidence}")

        # Risks
        self._risks_list.clear()
        for risk in argument.get("risks_identified", []):
            item = QListWidgetItem(f"⚠ {risk}")
            item.setForeground(QColor("#ffc107"))
            self._risks_list.addItem(item)

    def clear(self) -> None:
        """Clear the panel."""
        if not PYSIDE_AVAILABLE:
            return

        self._confidence_label.setText("--")
        self._argument_text.clear()
        self._points_list.clear()
        self._evidence_list.clear()
        self._risks_list.clear()


class ModeratorPanel(QWidget if PYSIDE_AVAILABLE else object):
    """Panel displaying moderator assessment."""

    def __init__(self, parent: QWidget | None = None):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the moderator panel UI."""
        layout = QVBoxLayout(self)

        header = QLabel("⚖️ Moderator Assessment")
        header.setStyleSheet("""
            color: #17a2b8;
            font-size: 14px;
            font-weight: bold;
        """)
        layout.addWidget(header)

        # Summary
        self._summary_text = QTextEdit()
        self._summary_text.setReadOnly(True)
        self._summary_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #17a2b8;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        self._summary_text.setMaximumHeight(80)
        layout.addWidget(self._summary_text)

        # Grid for status
        status_grid = QGridLayout()

        status_grid.addWidget(QLabel("Stronger:"), 0, 0)
        self._stronger_label = QLabel("--")
        self._stronger_label.setStyleSheet("font-weight: bold;")
        status_grid.addWidget(self._stronger_label, 0, 1)

        status_grid.addWidget(QLabel("Confidence:"), 0, 2)
        self._confidence_label = QLabel("--")
        self._confidence_label.setStyleSheet("font-weight: bold;")
        status_grid.addWidget(self._confidence_label, 0, 3)

        status_grid.addWidget(QLabel("Action:"), 1, 0)
        self._action_label = QLabel("--")
        self._action_label.setStyleSheet("font-weight: bold; color: #28a745;")
        status_grid.addWidget(self._action_label, 1, 1, 1, 3)

        layout.addLayout(status_grid)

        # Disagreements and agreements
        details_layout = QHBoxLayout()

        disagree_group = QGroupBox("Disagreements")
        disagree_group.setStyleSheet("""
            QGroupBox {
                color: #dc3545;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        disagree_layout = QVBoxLayout(disagree_group)
        self._disagreements_list = QListWidget()
        self._disagreements_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: none;
            }
        """)
        self._disagreements_list.setMaximumHeight(60)
        disagree_layout.addWidget(self._disagreements_list)
        details_layout.addWidget(disagree_group)

        agree_group = QGroupBox("Agreements")
        agree_group.setStyleSheet("""
            QGroupBox {
                color: #28a745;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        agree_layout = QVBoxLayout(agree_group)
        self._agreements_list = QListWidget()
        self._agreements_list.setStyleSheet(self._disagreements_list.styleSheet())
        self._agreements_list.setMaximumHeight(60)
        agree_layout.addWidget(self._agreements_list)
        details_layout.addWidget(agree_group)

        layout.addLayout(details_layout)

    def set_assessment(self, assessment: dict[str, Any]) -> None:
        """Update with moderator assessment data."""
        if not PYSIDE_AVAILABLE:
            return

        self._summary_text.setPlainText(assessment.get("summary", ""))

        stronger = assessment.get("stronger_argument", "")
        if stronger == "bull":
            self._stronger_label.setText("🐂 Bull")
            self._stronger_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        elif stronger == "bear":
            self._stronger_label.setText("🐻 Bear")
            self._stronger_label.setStyleSheet("font-weight: bold; color: #F44336;")
        else:
            self._stronger_label.setText("Tie")
            self._stronger_label.setStyleSheet("font-weight: bold; color: #888;")

        self._confidence_label.setText(f"{assessment.get('confidence', 0):.1%}")
        self._action_label.setText(assessment.get("recommended_action", "--"))

        # Lists
        self._disagreements_list.clear()
        for item in assessment.get("key_disagreements", []):
            self._disagreements_list.addItem(f"• {item}")

        self._agreements_list.clear()
        for item in assessment.get("areas_of_agreement", []):
            self._agreements_list.addItem(f"• {item}")

    def clear(self) -> None:
        """Clear the panel."""
        if not PYSIDE_AVAILABLE:
            return

        self._summary_text.clear()
        self._stronger_label.setText("--")
        self._confidence_label.setText("--")
        self._action_label.setText("--")
        self._disagreements_list.clear()
        self._agreements_list.clear()


class DebateViewer(QWidget if PYSIDE_AVAILABLE else object):
    """
    Widget for viewing Bull/Bear debate sessions.

    Shows:
    - Debate history with selector
    - Side-by-side Bull/Bear arguments
    - Round navigation
    - Moderator assessment
    - Final outcome visualization

    Example usage:
        from llm.agents.debate_mechanism import BullBearDebate
        debate = BullBearDebate(...)
        viewer = DebateViewer(debate_mechanism=debate)
        dashboard.add_widget(viewer)
    """

    # Signals
    debate_selected = Signal(str) if PYSIDE_AVAILABLE else None

    def __init__(
        self,
        debate_mechanism: Any | None = None,
        parent: QWidget | None = None,
    ):
        """
        Initialize debate viewer.

        Args:
            debate_mechanism: BullBearDebate instance
            parent: Parent widget
        """
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self.debate = debate_mechanism
        self._current_debate: dict[str, Any] | None = None
        self._current_round: int = 0
        self._debate_history: list[dict[str, Any]] = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header with debate selector
        header_layout = QHBoxLayout()

        title = QLabel("Debate Viewer")
        title.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        header_layout.addWidget(QLabel("Debate:"))
        self.debate_combo = QComboBox()
        self.debate_combo.setMinimumWidth(200)
        self.debate_combo.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3d3d3d;
                padding: 5px;
                border-radius: 4px;
            }
        """)
        self.debate_combo.currentIndexChanged.connect(self._on_debate_changed)
        header_layout.addWidget(self.debate_combo)

        layout.addLayout(header_layout)

        # Outcome display
        outcome_frame = QFrame()
        outcome_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        outcome_layout = QHBoxLayout(outcome_frame)

        self._outcome_label = QLabel("Outcome: --")
        self._outcome_label.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        outcome_layout.addWidget(self._outcome_label)

        self._consensus_label = QLabel("Consensus: --")
        self._consensus_label.setStyleSheet("color: #888;")
        outcome_layout.addWidget(self._consensus_label)

        self._trigger_label = QLabel("Trigger: --")
        self._trigger_label.setStyleSheet("color: #888;")
        outcome_layout.addWidget(self._trigger_label)

        self._duration_label = QLabel("Duration: --")
        self._duration_label.setStyleSheet("color: #888;")
        outcome_layout.addWidget(self._duration_label)

        outcome_layout.addStretch()

        layout.addWidget(outcome_frame)

        # Round navigation
        round_nav = QHBoxLayout()
        round_nav.addWidget(QLabel("Round:"))

        self._prev_btn = QPushButton("◀")
        self._prev_btn.setFixedSize(30, 30)
        self._prev_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #4d4d4d; }
            QPushButton:disabled { color: #666; }
        """)
        self._prev_btn.clicked.connect(self._prev_round)
        round_nav.addWidget(self._prev_btn)

        self._round_label = QLabel("0 / 0")
        self._round_label.setStyleSheet("color: white; font-weight: bold; min-width: 60px;")
        self._round_label.setAlignment(Qt.AlignCenter)
        round_nav.addWidget(self._round_label)

        self._next_btn = QPushButton("▶")
        self._next_btn.setFixedSize(30, 30)
        self._next_btn.setStyleSheet(self._prev_btn.styleSheet())
        self._next_btn.clicked.connect(self._next_round)
        round_nav.addWidget(self._next_btn)

        round_nav.addStretch()
        layout.addLayout(round_nav)

        # Main content: Bull vs Bear arguments
        splitter = QSplitter(Qt.Horizontal)

        self._bull_panel = ArgumentPanel("Bull", "#4CAF50")
        splitter.addWidget(self._bull_panel)

        self._bear_panel = ArgumentPanel("Bear", "#F44336")
        splitter.addWidget(self._bear_panel)

        splitter.setSizes([300, 300])
        layout.addWidget(splitter)

        # Moderator assessment
        self._moderator_panel = ModeratorPanel()
        layout.addWidget(self._moderator_panel)

    def _on_debate_changed(self, index: int) -> None:
        """Handle debate selection change."""
        if index < 0 or index >= len(self._debate_history):
            return

        self._current_debate = self._debate_history[index]
        self._current_round = 0
        self._update_display()

        if self.debate_selected and self._current_debate:
            self.debate_selected.emit(self._current_debate.get("debate_id", ""))

    def _prev_round(self) -> None:
        """Go to previous round."""
        if self._current_round > 0:
            self._current_round -= 1
            self._update_round_display()

    def _next_round(self) -> None:
        """Go to next round."""
        if self._current_debate:
            rounds = self._current_debate.get("rounds", [])
            if self._current_round < len(rounds) - 1:
                self._current_round += 1
                self._update_round_display()

    def _update_display(self) -> None:
        """Update the full display."""
        if not self._current_debate:
            self._clear_display()
            return

        # Update outcome
        outcome = self._current_debate.get("final_outcome", "")
        outcome_colors = {
            "buy": "#4CAF50",
            "sell": "#F44336",
            "hold": "#ffc107",
            "avoid": "#dc3545",
            "inconclusive": "#888",
        }
        color = outcome_colors.get(outcome, "#888")
        self._outcome_label.setText(f"Outcome: {outcome.upper()}")
        self._outcome_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")

        consensus = self._current_debate.get("consensus_confidence", 0)
        self._consensus_label.setText(f"Consensus: {consensus:.1%}")

        trigger = self._current_debate.get("trigger_reason", "")
        self._trigger_label.setText(f"Trigger: {trigger}")

        duration = self._current_debate.get("total_duration_ms", 0)
        self._duration_label.setText(f"Duration: {duration:.0f}ms")

        self._update_round_display()

    def _update_round_display(self) -> None:
        """Update the round-specific display."""
        if not self._current_debate:
            return

        rounds = self._current_debate.get("rounds", [])
        total_rounds = len(rounds)

        self._round_label.setText(f"{self._current_round + 1} / {total_rounds}")
        self._prev_btn.setEnabled(self._current_round > 0)
        self._next_btn.setEnabled(self._current_round < total_rounds - 1)

        if self._current_round < total_rounds:
            round_data = rounds[self._current_round]
            self._bull_panel.set_argument(round_data.get("bull_argument", {}))
            self._bear_panel.set_argument(round_data.get("bear_argument", {}))
            self._moderator_panel.set_assessment(round_data.get("moderator_assessment", {}))

    def _clear_display(self) -> None:
        """Clear all display elements."""
        self._outcome_label.setText("Outcome: --")
        self._consensus_label.setText("Consensus: --")
        self._trigger_label.setText("Trigger: --")
        self._duration_label.setText("Duration: --")
        self._round_label.setText("0 / 0")
        self._bull_panel.clear()
        self._bear_panel.clear()
        self._moderator_panel.clear()

    def add_debate(self, debate_result: dict[str, Any]) -> None:
        """
        Add a debate result to the viewer.

        Args:
            debate_result: DebateResult.to_dict() output
        """
        if not PYSIDE_AVAILABLE:
            return

        self._debate_history.insert(0, debate_result)

        # Update combo
        debate_id = debate_result.get("debate_id", "unknown")
        timestamp = debate_result.get("timestamp", "")
        outcome = debate_result.get("final_outcome", "")
        label = f"{debate_id[:8]}... ({outcome}) - {timestamp[:10]}"

        self.debate_combo.insertItem(0, label)
        self.debate_combo.setCurrentIndex(0)

    def load_debate_history(self, history: list[dict[str, Any]]) -> None:
        """
        Load multiple debates into history.

        Args:
            history: List of DebateResult.to_dict() outputs
        """
        if not PYSIDE_AVAILABLE:
            return

        self._debate_history = history

        self.debate_combo.clear()
        for debate in history:
            debate_id = debate.get("debate_id", "unknown")
            timestamp = debate.get("timestamp", "")
            outcome = debate.get("final_outcome", "")
            label = f"{debate_id[:8]}... ({outcome}) - {timestamp[:10]}"
            self.debate_combo.addItem(label)

        if history:
            self._current_debate = history[0]
            self._update_display()

    def set_debate_mechanism(self, debate: Any) -> None:
        """
        Set or update the debate mechanism.

        Args:
            debate: BullBearDebate instance
        """
        self.debate = debate


def create_debate_viewer(
    debate_mechanism: Any | None = None,
) -> DebateViewer:
    """
    Factory function to create a debate viewer widget.

    Args:
        debate_mechanism: BullBearDebate instance

    Returns:
        Configured DebateViewer
    """
    return DebateViewer(debate_mechanism=debate_mechanism)


# Stub implementations when PySide6 not available
if not PYSIDE_AVAILABLE:

    class ArgumentPanel:
        def __init__(self, *args, **kwargs):
            pass

        def set_argument(self, argument):
            pass

        def clear(self):
            pass

    class ModeratorPanel:
        def __init__(self, *args, **kwargs):
            pass

        def set_assessment(self, assessment):
            pass

        def clear(self):
            pass

    class DebateViewer:
        def __init__(self, *args, **kwargs):
            pass

        def add_debate(self, debate_result):
            pass

        def load_debate_history(self, history):
            pass

        def set_debate_mechanism(self, debate):
            pass


__all__ = [
    "ArgumentPanel",
    "DebateViewer",
    "ModeratorPanel",
    "create_debate_viewer",
]
