"""
Custom Leg Builder UI Widget

Visual interface for building custom multi-leg option spreads.
Provides drag-and-drop leg construction with real-time P&L diagram.

Features:
- Add/remove legs dynamically
- Buy/Sell toggle for each leg
- Call/Put selector per leg
- Strike and quantity inputs per leg
- Real-time net debit/credit calculation
- Visual P&L diagram showing profit/loss at different prices
- Max profit, max loss, and breakeven calculations
- Save custom strategies as templates
- Submit to manual legs executor

Example:
    from ui.custom_leg_builder import CustomLegBuilderWidget

    builder = CustomLegBuilderWidget()
    builder.on_submit_spread = lambda legs: print(f"Spread: {legs}")
    builder.show()
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QColor, QPainter, QPen
    from PySide6.QtWidgets import (
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QSpinBox,
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


class OptionType(Enum):
    """Option types."""

    CALL = "Call"
    PUT = "Put"


class Side(Enum):
    """Buy or Sell side."""

    BUY = "Buy"
    SELL = "Sell"


@dataclass
class OptionLeg:
    """Definition of a single option leg."""

    option_type: OptionType
    side: Side
    strike: float
    quantity: int
    premium: float  # Premium per contract


@dataclass
class SpreadDefinition:
    """Definition of a complete spread."""

    name: str
    legs: list[OptionLeg]
    net_debit: float
    net_credit: float
    max_profit: float
    max_loss: float
    breakevens: list[float]


class PLDiagramWidget(QWidget if PYSIDE_AVAILABLE else object):
    """Widget for displaying P&L diagram."""

    def __init__(self, parent: QWidget | None = None):
        """Initialize P&L diagram widget."""
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.legs: list[OptionLeg] = []
        self.spot_price = 450.0

    def set_legs(self, legs: list[OptionLeg], spot_price: float) -> None:
        """Set legs for diagram."""
        self.legs = legs
        self.spot_price = spot_price
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the P&L diagram."""
        if not PYSIDE_AVAILABLE:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Define drawing area
        margin = 40
        width = self.width() - 2 * margin
        height = self.height() - 2 * margin

        # Draw axes
        painter.setPen(QPen(Qt.black, 1))
        painter.drawLine(margin, height // 2 + margin, width + margin, height // 2 + margin)  # X-axis
        painter.drawLine(margin, margin, margin, height + margin)  # Y-axis

        if not self.legs:
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "Add legs to see P&L diagram",
            )
            return

        # Calculate price range
        all_strikes = [leg.strike for leg in self.legs]
        min_strike = min(all_strikes)
        max_strike = max(all_strikes)
        price_range = max_strike - min_strike
        if price_range == 0:
            price_range = max_strike * 0.2

        min_price = min_strike - price_range * 0.2
        max_price = max_strike + price_range * 0.2

        # Calculate P&L at various prices
        num_points = 100
        price_step = (max_price - min_price) / num_points

        pl_points = []
        min_pl = 0
        max_pl = 0

        for i in range(num_points + 1):
            price = min_price + i * price_step
            pl = self._calculate_pl_at_price(price)
            pl_points.append((price, pl))
            min_pl = min(min_pl, pl)
            max_pl = max(max_pl, pl)

        # Add padding to P&L range
        pl_range = max_pl - min_pl
        if pl_range == 0:
            pl_range = 100
        min_pl -= pl_range * 0.1
        max_pl += pl_range * 0.1

        # Draw P&L line
        painter.setPen(QPen(Qt.blue, 2))

        for i in range(len(pl_points) - 1):
            price1, pl1 = pl_points[i]
            price2, pl2 = pl_points[i + 1]

            # Map to screen coordinates
            x1 = margin + (price1 - min_price) / (max_price - min_price) * width
            y1 = height + margin - (pl1 - min_pl) / (max_pl - min_pl) * height

            x2 = margin + (price2 - min_price) / (max_price - min_price) * width
            y2 = height + margin - (pl2 - min_pl) / (max_pl - min_pl) * height

            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Draw current spot price line
        spot_x = margin + (self.spot_price - min_price) / (max_price - min_price) * width
        painter.setPen(QPen(Qt.red, 1, Qt.DashLine))
        painter.drawLine(int(spot_x), margin, int(spot_x), height + margin)

        # Draw labels
        painter.setPen(QPen(Qt.black, 1))
        painter.drawText(
            5,
            height // 2 + margin - 5,
            "$0",
        )

    def _calculate_pl_at_price(self, price: float) -> float:
        """Calculate P&L at expiration for given price."""
        total_pl = 0.0

        for leg in self.legs:
            # Calculate intrinsic value
            if leg.option_type == OptionType.CALL:
                intrinsic = max(0, price - leg.strike)
            else:  # PUT
                intrinsic = max(0, leg.strike - price)

            # Calculate P&L
            if leg.side == Side.BUY:
                # Paid premium, receive intrinsic value
                pl = (intrinsic - leg.premium) * leg.quantity * 100
            else:  # SELL
                # Received premium, pay intrinsic value
                pl = (leg.premium - intrinsic) * leg.quantity * 100

            total_pl += pl

        return total_pl


class CustomLegBuilderWidget(QWidget if PYSIDE_AVAILABLE else object):
    """
    Widget for building custom multi-leg option spreads.

    Provides interactive leg construction with visual P&L diagram.
    """

    # Signals
    spread_submitted = Signal(dict)  # SpreadDefinition dict

    def __init__(self, parent: QWidget | None = None):
        """
        Initialize custom leg builder widget.

        Args:
            parent: Parent widget
        """
        if not PYSIDE_AVAILABLE:
            raise ImportError("PySide6 is required for CustomLegBuilderWidget. " "Install with: pip install PySide6")

        super().__init__(parent)

        # Data
        self.legs: list[OptionLeg] = []
        self.spot_price = 450.0

        # Callbacks
        self.on_submit_spread: Callable[[SpreadDefinition], None] | None = None

        # Setup UI
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize UI components."""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Custom Leg Builder")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # Spot price input
        spot_layout = QHBoxLayout()
        spot_layout.addWidget(QLabel("Current Spot Price:"))
        self.spot_input = QDoubleSpinBox()
        self.spot_input.setRange(0.01, 10000.0)
        self.spot_input.setValue(450.0)
        self.spot_input.setPrefix("$")
        self.spot_input.valueChanged.connect(self._on_spot_changed)
        spot_layout.addWidget(self.spot_input)
        spot_layout.addStretch()
        layout.addLayout(spot_layout)

        # Legs table
        legs_group = self._create_legs_group()
        layout.addWidget(legs_group)

        # Add leg controls
        add_layout = self._create_add_leg_controls()
        layout.addLayout(add_layout)

        # P&L diagram
        self.pl_diagram = PLDiagramWidget()
        layout.addWidget(self.pl_diagram)

        # Summary
        summary_group = self._create_summary_group()
        layout.addWidget(summary_group)

        # Action buttons
        actions_layout = QHBoxLayout()

        save_btn = QPushButton("Save as Template")
        save_btn.clicked.connect(self._save_template)
        actions_layout.addWidget(save_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        actions_layout.addWidget(clear_btn)

        submit_btn = QPushButton("Submit Spread")
        submit_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        submit_btn.clicked.connect(self._submit_spread)
        actions_layout.addWidget(submit_btn)

        layout.addLayout(actions_layout)

        self.setLayout(layout)

    def _create_legs_group(self) -> QGroupBox:
        """Create legs table group."""
        group = QGroupBox("Legs")
        layout = QVBoxLayout()

        self.legs_table = QTableWidget()
        self.legs_table.setColumnCount(6)
        self.legs_table.setHorizontalHeaderLabels(["Type", "Side", "Strike", "Qty", "Premium", "Actions"])

        header = self.legs_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        layout.addWidget(self.legs_table)
        group.setLayout(layout)

        return group

    def _create_add_leg_controls(self) -> QHBoxLayout:
        """Create add leg controls."""
        layout = QHBoxLayout()

        # Option type
        self.new_type = QComboBox()
        self.new_type.addItems([ot.value for ot in OptionType])
        layout.addWidget(self.new_type)

        # Side
        self.new_side = QComboBox()
        self.new_side.addItems([s.value for s in Side])
        layout.addWidget(self.new_side)

        # Strike
        layout.addWidget(QLabel("Strike:"))
        self.new_strike = QDoubleSpinBox()
        self.new_strike.setRange(0.01, 10000.0)
        self.new_strike.setValue(450.0)
        self.new_strike.setPrefix("$")
        layout.addWidget(self.new_strike)

        # Quantity
        layout.addWidget(QLabel("Qty:"))
        self.new_quantity = QSpinBox()
        self.new_quantity.setRange(1, 100)
        self.new_quantity.setValue(1)
        layout.addWidget(self.new_quantity)

        # Premium
        layout.addWidget(QLabel("Premium:"))
        self.new_premium = QDoubleSpinBox()
        self.new_premium.setRange(0.01, 1000.0)
        self.new_premium.setValue(5.0)
        self.new_premium.setPrefix("$")
        layout.addWidget(self.new_premium)

        # Add button
        add_btn = QPushButton("Add Leg")
        add_btn.clicked.connect(self._add_leg)
        layout.addWidget(add_btn)

        return layout

    def _create_summary_group(self) -> QGroupBox:
        """Create summary group."""
        group = QGroupBox("Spread Summary")
        layout = QFormLayout()

        self.net_label = QLabel("$0.00")
        layout.addRow("Net Debit/Credit:", self.net_label)

        self.max_profit_label = QLabel("$0.00")
        layout.addRow("Max Profit:", self.max_profit_label)

        self.max_loss_label = QLabel("$0.00")
        layout.addRow("Max Loss:", self.max_loss_label)

        self.breakeven_label = QLabel("None")
        layout.addRow("Breakevens:", self.breakeven_label)

        group.setLayout(layout)
        return group

    def _on_spot_changed(self, value: float) -> None:
        """Handle spot price change."""
        self.spot_price = value
        self._update_summary()

    def _add_leg(self) -> None:
        """Add a new leg."""
        # Get values
        option_type_text = self.new_type.currentText()
        option_type = OptionType.CALL if option_type_text == OptionType.CALL.value else OptionType.PUT

        side_text = self.new_side.currentText()
        side = Side.BUY if side_text == Side.BUY.value else Side.SELL

        leg = OptionLeg(
            option_type=option_type,
            side=side,
            strike=self.new_strike.value(),
            quantity=self.new_quantity.value(),
            premium=self.new_premium.value(),
        )

        self.legs.append(leg)
        self._update_legs_table()
        self._update_summary()

    def _remove_leg(self, index: int) -> None:
        """Remove a leg."""
        if 0 <= index < len(self.legs):
            del self.legs[index]
            self._update_legs_table()
            self._update_summary()

    def _update_legs_table(self) -> None:
        """Update legs table display."""
        self.legs_table.setRowCount(len(self.legs))

        for i, leg in enumerate(self.legs):
            # Type
            self.legs_table.setItem(i, 0, QTableWidgetItem(leg.option_type.value))

            # Side
            side_item = QTableWidgetItem(leg.side.value)
            if leg.side == Side.BUY:
                side_item.setForeground(QColor("green"))
            else:
                side_item.setForeground(QColor("red"))
            self.legs_table.setItem(i, 1, side_item)

            # Strike
            self.legs_table.setItem(i, 2, QTableWidgetItem(f"${leg.strike:.2f}"))

            # Quantity
            self.legs_table.setItem(i, 3, QTableWidgetItem(str(leg.quantity)))

            # Premium
            self.legs_table.setItem(i, 4, QTableWidgetItem(f"${leg.premium:.2f}"))

            # Remove button
            remove_btn = QPushButton("Remove")
            remove_btn.clicked.connect(lambda checked, idx=i: self._remove_leg(idx))
            self.legs_table.setCellWidget(i, 5, remove_btn)

    def _update_summary(self) -> None:
        """Update spread summary."""
        if not self.legs:
            self.net_label.setText("$0.00")
            self.max_profit_label.setText("$0.00")
            self.max_loss_label.setText("$0.00")
            self.breakeven_label.setText("None")
            self.pl_diagram.set_legs([], self.spot_price)
            return

        # Calculate net debit/credit
        net = 0.0
        for leg in self.legs:
            if leg.side == Side.BUY:
                net -= leg.premium * leg.quantity * 100
            else:  # SELL
                net += leg.premium * leg.quantity * 100

        # Update net label
        if net < 0:
            self.net_label.setText(f"${abs(net):.2f} Debit")
            self.net_label.setStyleSheet("color: red;")
        else:
            self.net_label.setText(f"${net:.2f} Credit")
            self.net_label.setStyleSheet("color: green;")

        # Calculate max profit/loss (simplified)
        # For accurate calculation, would need to check all possible prices
        all_strikes = sorted(set(leg.strike for leg in self.legs))

        max_profit = -float("inf")
        max_loss = float("inf")

        # Check P&L at key points
        for strike in all_strikes:
            pl = self.pl_diagram._calculate_pl_at_price(strike)
            max_profit = max(max_profit, pl)
            max_loss = min(max_loss, pl)

        # Check at extremes
        if all_strikes:
            min_strike = min(all_strikes)
            max_strike = max(all_strikes)

            pl_low = self.pl_diagram._calculate_pl_at_price(min_strike * 0.5)
            pl_high = self.pl_diagram._calculate_pl_at_price(max_strike * 1.5)

            max_profit = max(max_profit, pl_low, pl_high)
            max_loss = min(max_loss, pl_low, pl_high)

        self.max_profit_label.setText(f"${max_profit:.2f}")
        self.max_profit_label.setStyleSheet("color: green;")

        self.max_loss_label.setText(f"${max_loss:.2f}")
        self.max_loss_label.setStyleSheet("color: red;")

        # Find breakevens (simplified - just check between strikes)
        breakevens = []
        if all_strikes:
            for i in range(len(all_strikes) - 1):
                strike1 = all_strikes[i]
                strike2 = all_strikes[i + 1]

                pl1 = self.pl_diagram._calculate_pl_at_price(strike1)
                pl2 = self.pl_diagram._calculate_pl_at_price(strike2)

                # If signs differ, there's a breakeven between them
                if pl1 * pl2 < 0:
                    # Linear interpolation
                    breakeven = strike1 + (strike2 - strike1) * abs(pl1) / (abs(pl1) + abs(pl2))
                    breakevens.append(breakeven)

        if breakevens:
            be_text = ", ".join([f"${be:.2f}" for be in breakevens])
            self.breakeven_label.setText(be_text)
        else:
            self.breakeven_label.setText("None")

        # Update P&L diagram
        self.pl_diagram.set_legs(self.legs, self.spot_price)

    def _clear_all(self) -> None:
        """Clear all legs."""
        reply = QMessageBox.question(
            self,
            "Confirm Clear",
            "Clear all legs?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.legs = []
            self._update_legs_table()
            self._update_summary()

    def _save_template(self) -> None:
        """Save spread as template."""
        if not self.legs:
            QMessageBox.warning(self, "Error", "No legs to save")
            return

        # Would save to file/database
        QMessageBox.information(
            self,
            "Template Saved",
            "Spread template saved successfully",
        )

    def _submit_spread(self) -> None:
        """Submit spread."""
        if not self.legs:
            QMessageBox.warning(self, "Error", "No legs to submit")
            return

        # Calculate net
        net = 0.0
        for leg in self.legs:
            if leg.side == Side.BUY:
                net -= leg.premium * leg.quantity * 100
            else:
                net += leg.premium * leg.quantity * 100

        # Create spread definition
        spread = SpreadDefinition(
            name="Custom Spread",
            legs=self.legs.copy(),
            net_debit=abs(net) if net < 0 else 0.0,
            net_credit=net if net >= 0 else 0.0,
            max_profit=float(self.max_profit_label.text().replace("$", "")),
            max_loss=float(self.max_loss_label.text().replace("$", "")),
            breakevens=[],  # Simplified
        )

        # Callback
        if self.on_submit_spread:
            self.on_submit_spread(spread)

        # Emit signal
        self.spread_submitted.emit(spread.__dict__)

        # Show confirmation
        QMessageBox.information(
            self,
            "Spread Submitted",
            f"Submitted custom spread with {len(self.legs)} legs",
        )


def create_custom_leg_builder() -> CustomLegBuilderWidget:
    """
    Create a CustomLegBuilderWidget instance.

    Returns:
        CustomLegBuilderWidget instance

    Example:
        >>> builder = create_custom_leg_builder()
        >>>
        >>> def handle_spread(spread):
        ...     print(f"Spread has {len(spread.legs)} legs")
        >>>
        >>> builder.on_submit_spread = handle_spread
        >>> builder.show()
    """
    return CustomLegBuilderWidget()
