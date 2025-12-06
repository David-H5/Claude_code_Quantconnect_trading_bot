"""
Custom Widgets for Trading Dashboard

Base widgets and reusable components for the trading UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


try:
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QColor, QFont
    from PySide6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False

    # Stub classes for when PySide6 is not available
    class QWidget:
        pass

    class Signal:
        def __init__(self, *args):
            pass


@dataclass
class AlertData:
    """Data for an alert popup."""

    title: str
    message: str
    symbol: str
    urgency: str  # "high", "medium", "low"
    timestamp: datetime
    actions: list[dict[str, Any]]


class StyledButton(QPushButton if PYSIDE_AVAILABLE else object):
    """Styled button with consistent theming."""

    def __init__(
        self,
        text: str,
        button_type: str = "default",
        parent: QWidget | None = None,
    ):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(text, parent)

        styles = {
            "default": "background-color: #3d3d3d; color: white; border: 1px solid #555;",
            "primary": "background-color: #0066cc; color: white; border: none;",
            "success": "background-color: #28a745; color: white; border: none;",
            "danger": "background-color: #dc3545; color: white; border: none;",
            "warning": "background-color: #ffc107; color: black; border: none;",
        }

        base_style = """
            QPushButton {
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                %s
            }
            QPushButton:hover {
                opacity: 0.8;
            }
            QPushButton:pressed {
                opacity: 0.6;
            }
        """ % styles.get(button_type, styles["default"])

        self.setStyleSheet(base_style)


class DataTable(QTableWidget if PYSIDE_AVAILABLE else object):
    """Styled data table for displaying trading data."""

    def __init__(
        self,
        columns: list[str],
        parent: QWidget | None = None,
    ):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)

        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)

        # Styling
        self.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                gridline-color: #3d3d3d;
                border: 1px solid #3d3d3d;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #0066cc;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #e0e0e0;
                padding: 8px;
                border: 1px solid #3d3d3d;
                font-weight: bold;
            }
        """)

        self.setAlternatingRowColors(True)
        self.horizontalHeader().setStretchLastSection(True)

    def add_row(self, data: list[Any], colors: dict[int, str] | None = None) -> None:
        """Add a row of data to the table."""
        row = self.rowCount()
        self.insertRow(row)

        for col, value in enumerate(data):
            item = QTableWidgetItem(str(value))

            if colors and col in colors:
                color = colors[col]
                if color == "green":
                    item.setForeground(QColor("#28a745"))
                elif color == "red":
                    item.setForeground(QColor("#dc3545"))
                elif color == "yellow":
                    item.setForeground(QColor("#ffc107"))

            self.setItem(row, col, item)

    def clear_data(self) -> None:
        """Clear all data from table."""
        self.setRowCount(0)


class StatusIndicator(QWidget if PYSIDE_AVAILABLE else object):
    """Status indicator with colored dot and label."""

    def __init__(
        self,
        label: str,
        status: str = "neutral",
        parent: QWidget | None = None,
    ):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Status dot
        self._dot = QLabel("●")
        self._dot.setFont(QFont("", 12))
        layout.addWidget(self._dot)

        # Label
        self._label = QLabel(label)
        layout.addWidget(self._label)

        layout.addStretch()

        self.set_status(status)

    def set_status(self, status: str) -> None:
        """Update status color."""
        colors = {
            "active": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545",
            "neutral": "#6c757d",
        }
        color = colors.get(status, colors["neutral"])
        self._dot.setStyleSheet(f"color: {color};")


class PriceDisplay(QWidget if PYSIDE_AVAILABLE else object):
    """Widget for displaying price with change indicator."""

    def __init__(
        self,
        symbol: str,
        parent: QWidget | None = None,
    ):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)

        layout = QVBoxLayout(self)

        # Symbol
        self._symbol = QLabel(symbol)
        self._symbol.setFont(QFont("", 14, QFont.Bold))
        layout.addWidget(self._symbol)

        # Price
        self._price = QLabel("--")
        self._price.setFont(QFont("", 24, QFont.Bold))
        layout.addWidget(self._price)

        # Change
        self._change = QLabel("--")
        layout.addWidget(self._change)

    def update_price(
        self,
        price: float,
        change: float,
        change_pct: float,
    ) -> None:
        """Update displayed price."""
        self._price.setText(f"${price:,.2f}")

        if change >= 0:
            color = "#28a745"
            sign = "+"
        else:
            color = "#dc3545"
            sign = ""

        self._change.setText(f"{sign}{change:,.2f} ({sign}{change_pct:.2f}%)")
        self._change.setStyleSheet(f"color: {color};")


class AlertPopup(QWidget if PYSIDE_AVAILABLE else object):
    """Popup widget for trading alerts."""

    closed = Signal() if PYSIDE_AVAILABLE else None

    def __init__(
        self,
        alert: AlertData,
        parent: QWidget | None = None,
    ):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)

        self.alert = alert
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the popup UI."""
        # Urgency-based colors
        urgency_colors = {
            "high": "#dc3545",
            "medium": "#ffc107",
            "low": "#17a2b8",
        }
        border_color = urgency_colors.get(self.alert.urgency, "#17a2b8")

        self.setStyleSheet(f"""
            AlertPopup {{
                background-color: #2d2d2d;
                border: 2px solid {border_color};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header
        header = QHBoxLayout()
        title = QLabel(f"<b>{self.alert.title}</b>")
        title.setStyleSheet("color: white; font-size: 14px;")
        header.addWidget(title)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: white;
                border: none;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #dc3545;
                border-radius: 12px;
            }
        """)
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)

        layout.addLayout(header)

        # Symbol
        symbol = QLabel(f"<b>{self.alert.symbol}</b>")
        symbol.setStyleSheet("color: #0066cc; font-size: 18px;")
        layout.addWidget(symbol)

        # Message
        message = QLabel(self.alert.message)
        message.setStyleSheet("color: #e0e0e0;")
        message.setWordWrap(True)
        layout.addWidget(message)

        # Action buttons
        if self.alert.actions:
            actions_layout = QHBoxLayout()
            for action in self.alert.actions:
                btn = StyledButton(
                    action.get("label", "Action"),
                    action.get("type", "default"),
                )
                if "callback" in action:
                    btn.clicked.connect(action["callback"])
                actions_layout.addWidget(btn)
            layout.addLayout(actions_layout)

        # Auto-close timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.close)
        self._timer.start(10000)  # 10 seconds

    def close(self) -> None:
        """Close the popup."""
        self._timer.stop()
        if self.closed:
            self.closed.emit()
        super().close()


class SettingsPanel(QWidget if PYSIDE_AVAILABLE else object):
    """Settings panel for configuring trading parameters."""

    settings_changed = Signal(dict) if PYSIDE_AVAILABLE else None

    def __init__(
        self,
        settings: dict[str, Any],
        parent: QWidget | None = None,
    ):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)

        self.settings = settings.copy()
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up settings UI."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("<b>Settings</b>")
        title.setStyleSheet("color: white; font-size: 16px;")
        layout.addWidget(title)

        # Settings will be dynamically generated based on config
        # Placeholder for now
        placeholder = QLabel("Settings controls will appear here")
        placeholder.setStyleSheet("color: #6c757d;")
        layout.addWidget(placeholder)

        layout.addStretch()

        # Save button
        save_btn = StyledButton("Save Settings", "primary")
        save_btn.clicked.connect(self._save_settings)
        layout.addWidget(save_btn)

    def _save_settings(self) -> None:
        """Save settings and emit signal."""
        if self.settings_changed:
            self.settings_changed.emit(self.settings)


if not PYSIDE_AVAILABLE:
    # Provide stub implementations when PySide6 is not available
    class StyledButton:
        def __init__(self, *args, **kwargs):
            pass

    class DataTable:
        def __init__(self, *args, **kwargs):
            pass

    class StatusIndicator:
        def __init__(self, *args, **kwargs):
            pass

    class PriceDisplay:
        def __init__(self, *args, **kwargs):
            pass

    class AlertPopup:
        def __init__(self, *args, **kwargs):
            pass

    class SettingsPanel:
        def __init__(self, *args, **kwargs):
            pass


__all__ = [
    "PYSIDE_AVAILABLE",
    "AlertData",
    "AlertPopup",
    "DataTable",
    "PriceDisplay",
    "SettingsPanel",
    "StatusIndicator",
    "StyledButton",
]
