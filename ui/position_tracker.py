"""
Position Tracker UI

Unified position tracking widget for all positions regardless of source.
Displays autonomous, manual, and recurring positions with real-time updates.

Features:
- Single view for all positions (autonomous, manual_ui, recurring)
- Real-time P&L and Greeks updates
- Aggregated Greeks by position and portfolio
- Position management controls (close, adjust, roll)
- Color-coded by strategy type and source
- Export positions to CSV/JSON
- Historical P&L chart

Example:
    from ui.position_tracker import PositionTrackerWidget

    tracker = PositionTrackerWidget()
    tracker.add_position(position_data)
    tracker.update_position("pos_1", current_value=-250.0)
    tracker.show()
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


try:
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QComboBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
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

    class QWidget:
        pass

    class Signal:
        def __init__(self, *args):
            pass


class PositionSource(Enum):
    """Source of the position."""

    AUTONOMOUS = "autonomous"
    MANUAL_UI = "manual_ui"
    RECURRING = "recurring"


@dataclass
class PositionGreeks:
    """Greeks for a position."""

    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


@dataclass
class PositionData:
    """Data for a tracked position."""

    position_id: str
    symbol: str
    source: PositionSource
    strategy_type: str
    entry_price: float
    current_value: float
    entry_time: datetime
    quantity: int
    greeks: PositionGreeks = field(default_factory=PositionGreeks)
    realized_pnl: float = 0.0
    management_enabled: bool = True

    @property
    def pnl(self) -> float:
        """Calculate current P&L."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_value - self.entry_price) / abs(self.entry_price)

    @property
    def pnl_dollars(self) -> float:
        """Calculate P&L in dollars."""
        return (self.current_value - self.entry_price) * self.quantity


class PositionTrackerWidget(QWidget if PYSIDE_AVAILABLE else object):
    """
    Widget for tracking all positions across all sources.

    Provides unified view with real-time updates, Greeks aggregation,
    and position management controls.
    """

    # Signals
    position_closed = Signal(str)  # position_id
    position_adjusted = Signal(str)  # position_id
    position_rolled = Signal(str)  # position_id

    def __init__(self, parent: QWidget | None = None):
        """
        Initialize position tracker widget.

        Args:
            parent: Parent widget
        """
        if not PYSIDE_AVAILABLE:
            raise ImportError("PySide6 is required for PositionTrackerWidget. " "Install with: pip install PySide6")

        super().__init__(parent)

        # Data
        self.positions: dict[str, PositionData] = {}

        # Callbacks
        self.on_close_position: Callable[[str], None] | None = None
        self.on_adjust_position: Callable[[str], None] | None = None
        self.on_roll_position: Callable[[str], None] | None = None

        # Statistics
        self.stats = {
            "total_positions": 0,
            "total_pnl": 0.0,
            "total_pnl_dollars": 0.0,
            "portfolio_delta": 0.0,
            "portfolio_gamma": 0.0,
            "portfolio_theta": 0.0,
            "portfolio_vega": 0.0,
        }

        # Setup UI
        self._init_ui()

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(1000)  # Update every second

    def _init_ui(self) -> None:
        """Initialize UI components."""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Position Tracker")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # Portfolio summary
        summary_group = self._create_summary_group()
        layout.addWidget(summary_group)

        # Filter controls
        filter_layout = self._create_filter_controls()
        layout.addLayout(filter_layout)

        # Positions table
        self.table = self._create_positions_table()
        layout.addWidget(self.table)

        # Action buttons
        actions_layout = self._create_action_buttons()
        layout.addLayout(actions_layout)

        self.setLayout(layout)

    def _create_summary_group(self) -> QGroupBox:
        """Create portfolio summary group."""
        group = QGroupBox("Portfolio Summary")
        layout = QHBoxLayout()

        # P&L
        self.pnl_label = QLabel("P&L: $0.00 (0.00%)")
        self.pnl_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.pnl_label)

        # Greeks
        self.delta_label = QLabel("Δ: 0.00")
        layout.addWidget(self.delta_label)

        self.gamma_label = QLabel("Γ: 0.00")
        layout.addWidget(self.gamma_label)

        self.theta_label = QLabel("Θ: 0.00")
        layout.addWidget(self.theta_label)

        self.vega_label = QLabel("V: 0.00")
        layout.addWidget(self.vega_label)

        # Position count
        self.count_label = QLabel("Positions: 0")
        layout.addWidget(self.count_label)

        layout.addStretch()
        group.setLayout(layout)

        return group

    def _create_filter_controls(self) -> QHBoxLayout:
        """Create filter controls."""
        layout = QHBoxLayout()

        # Source filter
        layout.addWidget(QLabel("Source:"))
        self.source_filter = QComboBox()
        self.source_filter.addItems(["All", "Autonomous", "Manual", "Recurring"])
        self.source_filter.currentTextChanged.connect(self._apply_filters)
        layout.addWidget(self.source_filter)

        # Symbol filter
        layout.addWidget(QLabel("Symbol:"))
        self.symbol_filter = QLineEdit()
        self.symbol_filter.setPlaceholderText("Filter by symbol...")
        self.symbol_filter.textChanged.connect(self._apply_filters)
        layout.addWidget(self.symbol_filter)

        layout.addStretch()

        return layout

    def _create_positions_table(self) -> QTableWidget:
        """Create positions table."""
        table = QTableWidget()
        table.setColumnCount(12)

        headers = [
            "ID",
            "Symbol",
            "Source",
            "Strategy",
            "Qty",
            "Entry",
            "Current",
            "P&L %",
            "P&L $",
            "Delta",
            "Theta",
            "Managed",
        ]
        table.setHorizontalHeaderLabels(headers)

        # Configure table
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setAlternatingRowColors(True)

        # Resize columns
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)

        return table

    def _create_action_buttons(self) -> QHBoxLayout:
        """Create action buttons."""
        layout = QHBoxLayout()

        # Close position
        close_btn = QPushButton("Close Position")
        close_btn.clicked.connect(self._close_selected_position)
        layout.addWidget(close_btn)

        # Adjust position
        adjust_btn = QPushButton("Adjust Position")
        adjust_btn.clicked.connect(self._adjust_selected_position)
        layout.addWidget(adjust_btn)

        # Roll position
        roll_btn = QPushButton("Roll Position")
        roll_btn.clicked.connect(self._roll_selected_position)
        layout.addWidget(roll_btn)

        layout.addStretch()

        # Close all
        close_all_btn = QPushButton("Close All Positions")
        close_all_btn.clicked.connect(self._close_all_positions)
        close_all_btn.setStyleSheet("background-color: #ff6b6b;")
        layout.addWidget(close_all_btn)

        # Export
        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self._export_csv)
        layout.addWidget(export_btn)

        return layout

    def add_position(self, position: PositionData) -> None:
        """
        Add a new position to tracking.

        Args:
            position: Position data
        """
        self.positions[position.position_id] = position
        self.stats["total_positions"] = len(self.positions)
        self._update_display()

    def update_position(
        self,
        position_id: str,
        current_value: float | None = None,
        greeks: PositionGreeks | None = None,
        quantity: int | None = None,
    ) -> None:
        """
        Update position data.

        Args:
            position_id: Position ID
            current_value: New current value
            greeks: Updated Greeks
            quantity: Updated quantity
        """
        if position_id not in self.positions:
            return

        position = self.positions[position_id]

        if current_value is not None:
            position.current_value = current_value

        if greeks is not None:
            position.greeks = greeks

        if quantity is not None:
            position.quantity = quantity

        self._update_display()

    def remove_position(self, position_id: str) -> None:
        """
        Remove a position from tracking.

        Args:
            position_id: Position ID
        """
        if position_id in self.positions:
            del self.positions[position_id]
            self.stats["total_positions"] = len(self.positions)
            self._update_display()

    def get_position(self, position_id: str) -> PositionData | None:
        """
        Get position data.

        Args:
            position_id: Position ID

        Returns:
            Position data if found
        """
        return self.positions.get(position_id)

    def get_all_positions(self) -> list[PositionData]:
        """Get all positions."""
        return list(self.positions.values())

    def get_positions_by_source(
        self,
        source: PositionSource,
    ) -> list[PositionData]:
        """
        Get positions filtered by source.

        Args:
            source: Position source

        Returns:
            List of positions
        """
        return [p for p in self.positions.values() if p.source == source]

    def get_positions_by_symbol(self, symbol: str) -> list[PositionData]:
        """
        Get positions filtered by symbol.

        Args:
            symbol: Symbol

        Returns:
            List of positions
        """
        return [p for p in self.positions.values() if p.symbol == symbol]

    def _update_display(self) -> None:
        """Update all display elements."""
        self._update_table()
        self._update_summary()

    def _update_table(self) -> None:
        """Update positions table."""
        # Get filtered positions
        filtered = self._get_filtered_positions()

        # Clear table
        self.table.setRowCount(0)

        # Populate table
        for position in filtered:
            row = self.table.rowCount()
            self.table.insertRow(row)

            # ID
            self.table.setItem(row, 0, QTableWidgetItem(position.position_id))

            # Symbol
            self.table.setItem(row, 1, QTableWidgetItem(position.symbol))

            # Source
            source_item = QTableWidgetItem(position.source.value)
            source_item.setBackground(self._get_source_color(position.source))
            self.table.setItem(row, 2, source_item)

            # Strategy
            self.table.setItem(row, 3, QTableWidgetItem(position.strategy_type))

            # Quantity
            self.table.setItem(row, 4, QTableWidgetItem(str(position.quantity)))

            # Entry
            self.table.setItem(row, 5, QTableWidgetItem(f"${position.entry_price:.2f}"))

            # Current
            self.table.setItem(row, 6, QTableWidgetItem(f"${position.current_value:.2f}"))

            # P&L %
            pnl_pct = position.pnl * 100
            pnl_item = QTableWidgetItem(f"{pnl_pct:+.2f}%")
            pnl_item.setForeground(QColor("green") if pnl_pct >= 0 else QColor("red"))
            self.table.setItem(row, 7, pnl_item)

            # P&L $
            pnl_dollars = position.pnl_dollars
            pnl_dollar_item = QTableWidgetItem(f"${pnl_dollars:+.2f}")
            pnl_dollar_item.setForeground(QColor("green") if pnl_dollars >= 0 else QColor("red"))
            self.table.setItem(row, 8, pnl_dollar_item)

            # Delta
            self.table.setItem(row, 9, QTableWidgetItem(f"{position.greeks.delta:.2f}"))

            # Theta
            self.table.setItem(row, 10, QTableWidgetItem(f"{position.greeks.theta:.2f}"))

            # Managed
            managed = "Yes" if position.management_enabled else "No"
            self.table.setItem(row, 11, QTableWidgetItem(managed))

    def _update_summary(self) -> None:
        """Update portfolio summary."""
        # Calculate aggregated stats
        total_pnl_pct = 0.0
        total_pnl_dollars = 0.0
        portfolio_delta = 0.0
        portfolio_gamma = 0.0
        portfolio_theta = 0.0
        portfolio_vega = 0.0

        for position in self.positions.values():
            total_pnl_pct += position.pnl
            total_pnl_dollars += position.pnl_dollars
            portfolio_delta += position.greeks.delta * position.quantity
            portfolio_gamma += position.greeks.gamma * position.quantity
            portfolio_theta += position.greeks.theta * position.quantity
            portfolio_vega += position.greeks.vega * position.quantity

        # Average P&L %
        if self.positions:
            avg_pnl_pct = total_pnl_pct / len(self.positions) * 100
        else:
            avg_pnl_pct = 0.0

        # Update stats
        self.stats.update(
            {
                "total_pnl": avg_pnl_pct,
                "total_pnl_dollars": total_pnl_dollars,
                "portfolio_delta": portfolio_delta,
                "portfolio_gamma": portfolio_gamma,
                "portfolio_theta": portfolio_theta,
                "portfolio_vega": portfolio_vega,
            }
        )

        # Update labels
        pnl_color = "green" if total_pnl_dollars >= 0 else "red"
        self.pnl_label.setText(f"P&L: ${total_pnl_dollars:+.2f} ({avg_pnl_pct:+.2f}%)")
        self.pnl_label.setStyleSheet(f"font-size: 14px; color: {pnl_color};")

        self.delta_label.setText(f"Δ: {portfolio_delta:.2f}")
        self.gamma_label.setText(f"Γ: {portfolio_gamma:.4f}")
        self.theta_label.setText(f"Θ: {portfolio_theta:.2f}")
        self.vega_label.setText(f"V: {portfolio_vega:.2f}")

        self.count_label.setText(f"Positions: {len(self.positions)}")

    def _get_filtered_positions(self) -> list[PositionData]:
        """Get positions based on current filters."""
        positions = list(self.positions.values())

        # Source filter
        source_filter = self.source_filter.currentText()
        if source_filter != "All":
            source_map = {
                "Autonomous": PositionSource.AUTONOMOUS,
                "Manual": PositionSource.MANUAL_UI,
                "Recurring": PositionSource.RECURRING,
            }
            if source_filter in source_map:
                positions = [p for p in positions if p.source == source_map[source_filter]]

        # Symbol filter
        symbol_filter = self.symbol_filter.text().strip().upper()
        if symbol_filter:
            positions = [p for p in positions if symbol_filter in p.symbol.upper()]

        return positions

    def _get_source_color(self, source: PositionSource) -> QColor:
        """Get color for position source."""
        colors = {
            PositionSource.AUTONOMOUS: QColor(173, 216, 230),  # Light blue
            PositionSource.MANUAL_UI: QColor(255, 218, 185),  # Peach
            PositionSource.RECURRING: QColor(221, 160, 221),  # Plum
        }
        return colors.get(source, QColor(255, 255, 255))

    def _close_selected_position(self) -> None:
        """Close selected position."""
        selected = self.table.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "Warning", "Please select a position")
            return

        position_id = self.table.item(selected, 0).text()

        reply = QMessageBox.question(
            self,
            "Confirm Close",
            f"Close position {position_id}?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            if self.on_close_position:
                self.on_close_position(position_id)

            self.position_closed.emit(position_id)

    def _adjust_selected_position(self) -> None:
        """Adjust selected position."""
        selected = self.table.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "Warning", "Please select a position")
            return

        position_id = self.table.item(selected, 0).text()

        if self.on_adjust_position:
            self.on_adjust_position(position_id)

        self.position_adjusted.emit(position_id)

    def _roll_selected_position(self) -> None:
        """Roll selected position."""
        selected = self.table.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "Warning", "Please select a position")
            return

        position_id = self.table.item(selected, 0).text()

        reply = QMessageBox.question(
            self,
            "Confirm Roll",
            f"Roll position {position_id}?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            if self.on_roll_position:
                self.on_roll_position(position_id)

            self.position_rolled.emit(position_id)

    def _close_all_positions(self) -> None:
        """Close all positions."""
        if not self.positions:
            QMessageBox.information(self, "Info", "No positions to close")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Close All",
            f"Close all {len(self.positions)} positions?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            for position_id in list(self.positions.keys()):
                if self.on_close_position:
                    self.on_close_position(position_id)

                self.position_closed.emit(position_id)

    def _apply_filters(self) -> None:
        """Apply current filters."""
        self._update_display()

    def _export_csv(self) -> None:
        """Export positions to CSV."""
        if not self.positions:
            QMessageBox.information(self, "Info", "No positions to export")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Positions",
            "",
            "CSV Files (*.csv)",
        )

        if not filename:
            return

        try:
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)

                # Header
                writer.writerow(
                    [
                        "Position ID",
                        "Symbol",
                        "Source",
                        "Strategy",
                        "Quantity",
                        "Entry Price",
                        "Current Value",
                        "P&L %",
                        "P&L $",
                        "Delta",
                        "Gamma",
                        "Theta",
                        "Vega",
                        "Managed",
                    ]
                )

                # Data
                for position in self.positions.values():
                    writer.writerow(
                        [
                            position.position_id,
                            position.symbol,
                            position.source.value,
                            position.strategy_type,
                            position.quantity,
                            position.entry_price,
                            position.current_value,
                            position.pnl * 100,
                            position.pnl_dollars,
                            position.greeks.delta,
                            position.greeks.gamma,
                            position.greeks.theta,
                            position.greeks.vega,
                            "Yes" if position.management_enabled else "No",
                        ]
                    )

            QMessageBox.information(self, "Success", f"Exported {len(self.positions)} positions")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e!s}")

    def get_statistics(self) -> dict[str, Any]:
        """Get tracker statistics."""
        return self.stats.copy()


def create_position_tracker() -> PositionTrackerWidget:
    """
    Create a PositionTrackerWidget instance.

    Returns:
        PositionTrackerWidget instance

    Example:
        >>> tracker = create_position_tracker()
        >>>
        >>> # Add position
        >>> position = PositionData(
        ...     position_id="pos_1",
        ...     symbol="SPY",
        ...     source=PositionSource.AUTONOMOUS,
        ...     strategy_type="iron_condor",
        ...     entry_price=-500.0,
        ...     current_value=-250.0,
        ...     entry_time=datetime.now(),
        ...     quantity=1,
        ...     greeks=PositionGreeks(delta=0.05, theta=-2.5),
        ... )
        >>> tracker.add_position(position)
        >>>
        >>> # Update position
        >>> tracker.update_position("pos_1", current_value=-200.0)
        >>>
        >>> # Show widget
        >>> tracker.show()
    """
    return PositionTrackerWidget()
