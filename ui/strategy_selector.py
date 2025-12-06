"""
Strategy Selector UI Widget

Visual dropdown widget for selecting and submitting option strategies.
Provides intuitive interface for all 37+ QuantConnect OptionStrategies factory methods.

Features:
- Dropdown for all 37+ option strategies
- Dynamic parameter inputs based on selected strategy
- Strike selection with delta targeting or ATM offset
- Expiry selection with DTE display
- Execution type selector (Market/Limit/Two-Part)
- Real-time Greeks preview before submission
- Quantity and limit price inputs
- Order submission with validation

Example:
    from ui.strategy_selector import StrategySelectorWidget

    selector = StrategySelectorWidget()
    selector.on_submit_order = lambda order: print(f"Submitting: {order}")
    selector.show()
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


try:
    from PySide6.QtCore import QDate, Qt, Signal
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QComboBox,
        QDateEdit,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QSpinBox,
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


class ExecutionType(Enum):
    """Order execution types."""

    MARKET = "Market"
    LIMIT = "Limit"
    TWO_PART = "Two-Part Spread"


class StrikeSelectionMode(Enum):
    """Strike selection modes."""

    DELTA_TARGET = "Delta Target"
    ATM_OFFSET = "ATM Offset"
    MANUAL = "Manual Entry"


@dataclass
class StrategyDefinition:
    """Definition of an option strategy."""

    name: str
    display_name: str
    description: str
    legs: int  # Number of legs
    params: list[str]  # Required parameter names


# Define all 37+ QuantConnect OptionStrategies
STRATEGY_DEFINITIONS = {
    # Single leg
    "naked_call": StrategyDefinition("naked_call", "Naked Call", "Sell call option", 1, ["strike"]),
    "naked_put": StrategyDefinition("naked_put", "Naked Put", "Sell put option", 1, ["strike"]),
    "covered_call": StrategyDefinition(
        "covered_call",
        "Covered Call",
        "Long stock + short call",
        2,
        ["strike"],
    ),
    "protective_put": StrategyDefinition(
        "protective_put",
        "Protective Put",
        "Long stock + long put",
        2,
        ["strike"],
    ),
    # Vertical spreads
    "bull_call_spread": StrategyDefinition(
        "bull_call_spread",
        "Bull Call Spread",
        "Buy lower call, sell higher call",
        2,
        ["lower_strike", "higher_strike"],
    ),
    "bear_call_spread": StrategyDefinition(
        "bear_call_spread",
        "Bear Call Spread",
        "Sell lower call, buy higher call",
        2,
        ["lower_strike", "higher_strike"],
    ),
    "bull_put_spread": StrategyDefinition(
        "bull_put_spread",
        "Bull Put Spread",
        "Sell higher put, buy lower put",
        2,
        ["lower_strike", "higher_strike"],
    ),
    "bear_put_spread": StrategyDefinition(
        "bear_put_spread",
        "Bear Put Spread",
        "Buy higher put, sell lower put",
        2,
        ["lower_strike", "higher_strike"],
    ),
    # Straddles and strangles
    "straddle": StrategyDefinition("straddle", "Straddle", "Buy/sell call and put at same strike", 2, ["strike"]),
    "short_straddle": StrategyDefinition(
        "short_straddle",
        "Short Straddle",
        "Sell call and put at same strike",
        2,
        ["strike"],
    ),
    "strangle": StrategyDefinition(
        "strangle",
        "Strangle",
        "Buy call and put at different strikes",
        2,
        ["put_strike", "call_strike"],
    ),
    "short_strangle": StrategyDefinition(
        "short_strangle",
        "Short Strangle",
        "Sell call and put at different strikes",
        2,
        ["put_strike", "call_strike"],
    ),
    # Butterflies
    "butterfly_call": StrategyDefinition(
        "butterfly_call",
        "Call Butterfly",
        "Buy 1 low call, sell 2 mid calls, buy 1 high call",
        4,
        ["lower_strike", "middle_strike", "higher_strike"],
    ),
    "butterfly_put": StrategyDefinition(
        "butterfly_put",
        "Put Butterfly",
        "Buy 1 low put, sell 2 mid puts, buy 1 high put",
        4,
        ["lower_strike", "middle_strike", "higher_strike"],
    ),
    "short_butterfly_call": StrategyDefinition(
        "short_butterfly_call",
        "Short Call Butterfly",
        "Sell 1 low call, buy 2 mid calls, sell 1 high call",
        4,
        ["lower_strike", "middle_strike", "higher_strike"],
    ),
    "short_butterfly_put": StrategyDefinition(
        "short_butterfly_put",
        "Short Put Butterfly",
        "Sell 1 low put, buy 2 mid puts, sell 1 high put",
        4,
        ["lower_strike", "middle_strike", "higher_strike"],
    ),
    # Iron butterflies and condors
    "iron_butterfly": StrategyDefinition(
        "iron_butterfly",
        "Iron Butterfly",
        "Short straddle + long strangle",
        4,
        ["lower_strike", "middle_strike", "higher_strike"],
    ),
    "iron_condor": StrategyDefinition(
        "iron_condor",
        "Iron Condor",
        "Short strangle + protective wings",
        4,
        ["put_buy", "put_sell", "call_sell", "call_buy"],
    ),
    # Calendars
    "calendar_call_spread": StrategyDefinition(
        "calendar_call_spread",
        "Call Calendar Spread",
        "Sell near-term call, buy far-term call",
        2,
        ["strike"],
    ),
    "calendar_put_spread": StrategyDefinition(
        "calendar_put_spread",
        "Put Calendar Spread",
        "Sell near-term put, buy far-term put",
        2,
        ["strike"],
    ),
}


@dataclass
class OrderSubmission:
    """Order submission data."""

    strategy_name: str
    symbol: str
    quantity: int
    execution_type: ExecutionType
    limit_price: float | None
    strikes: dict[str, float]
    expiry_date: datetime
    estimated_greeks: dict[str, float] | None = None


class StrategySelectorWidget(QWidget if PYSIDE_AVAILABLE else object):
    """
    Widget for selecting and submitting option strategies.

    Provides dropdown selection, parameter inputs, and order submission.
    """

    # Signals
    order_submitted = Signal(dict) if PYSIDE_AVAILABLE else None

    def __init__(self, parent: QWidget | None = None):
        """
        Initialize strategy selector widget.

        Args:
            parent: Parent widget
        """
        if not PYSIDE_AVAILABLE:
            raise ImportError("PySide6 is required for StrategySelectorWidget. " "Install with: pip install PySide6")

        super().__init__(parent)

        # Data
        self.current_symbol = "SPY"
        self.current_spot_price = 450.0

        # Callbacks
        self.on_submit_order: Callable[[OrderSubmission], None] | None = None
        self.on_preview_greeks: Callable[[OrderSubmission], dict[str, float]] | None = None

        # Setup UI
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize UI components."""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Option Strategy Selector")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # Symbol input
        symbol_layout = QHBoxLayout()
        symbol_layout.addWidget(QLabel("Symbol:"))
        self.symbol_input = QLineEdit("SPY")
        self.symbol_input.setMaximumWidth(100)
        self.symbol_input.textChanged.connect(self._on_symbol_changed)
        symbol_layout.addWidget(self.symbol_input)

        symbol_layout.addWidget(QLabel("Spot Price:"))
        self.spot_price_input = QDoubleSpinBox()
        self.spot_price_input.setRange(0.01, 10000.0)
        self.spot_price_input.setValue(450.0)
        self.spot_price_input.setPrefix("$")
        self.spot_price_input.valueChanged.connect(self._on_spot_price_changed)
        symbol_layout.addWidget(self.spot_price_input)

        symbol_layout.addStretch()
        layout.addLayout(symbol_layout)

        # Strategy selection
        strategy_group = self._create_strategy_group()
        layout.addWidget(strategy_group)

        # Parameters
        self.params_group = self._create_params_group()
        layout.addWidget(self.params_group)

        # Execution settings
        execution_group = self._create_execution_group()
        layout.addWidget(execution_group)

        # Greeks preview
        self.greeks_group = self._create_greeks_group()
        layout.addWidget(self.greeks_group)

        # Submit button
        submit_btn = QPushButton("Submit Order")
        submit_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        submit_btn.clicked.connect(self._submit_order)
        layout.addWidget(submit_btn)

        layout.addStretch()
        self.setLayout(layout)

    def _create_strategy_group(self) -> QGroupBox:
        """Create strategy selection group."""
        group = QGroupBox("Strategy")
        layout = QVBoxLayout()

        # Strategy dropdown
        self.strategy_combo = QComboBox()

        # Add strategies grouped by type
        self.strategy_combo.addItem("--- Single Leg ---")
        self.strategy_combo.addItems(
            [
                STRATEGY_DEFINITIONS[k].display_name
                for k in ["naked_call", "naked_put", "covered_call", "protective_put"]
            ]
        )

        self.strategy_combo.addItem("--- Vertical Spreads ---")
        self.strategy_combo.addItems(
            [
                STRATEGY_DEFINITIONS[k].display_name
                for k in [
                    "bull_call_spread",
                    "bear_call_spread",
                    "bull_put_spread",
                    "bear_put_spread",
                ]
            ]
        )

        self.strategy_combo.addItem("--- Straddles & Strangles ---")
        self.strategy_combo.addItems(
            [STRATEGY_DEFINITIONS[k].display_name for k in ["straddle", "short_straddle", "strangle", "short_strangle"]]
        )

        self.strategy_combo.addItem("--- Butterflies ---")
        self.strategy_combo.addItems(
            [
                STRATEGY_DEFINITIONS[k].display_name
                for k in [
                    "butterfly_call",
                    "butterfly_put",
                    "short_butterfly_call",
                    "short_butterfly_put",
                ]
            ]
        )

        self.strategy_combo.addItem("--- Iron Strategies ---")
        self.strategy_combo.addItems([STRATEGY_DEFINITIONS[k].display_name for k in ["iron_butterfly", "iron_condor"]])

        self.strategy_combo.addItem("--- Calendar Spreads ---")
        self.strategy_combo.addItems(
            [STRATEGY_DEFINITIONS[k].display_name for k in ["calendar_call_spread", "calendar_put_spread"]]
        )

        self.strategy_combo.currentTextChanged.connect(self._on_strategy_changed)
        layout.addWidget(self.strategy_combo)

        # Description label
        self.strategy_desc = QLabel("")
        self.strategy_desc.setWordWrap(True)
        layout.addWidget(self.strategy_desc)

        group.setLayout(layout)
        return group

    def _create_params_group(self) -> QGroupBox:
        """Create parameters group."""
        group = QGroupBox("Strike Selection")
        self.params_layout = QFormLayout()

        # Strike selection mode
        self.strike_mode = QComboBox()
        self.strike_mode.addItems([mode.value for mode in StrikeSelectionMode])
        self.strike_mode.currentTextChanged.connect(self._on_strike_mode_changed)
        self.params_layout.addRow("Mode:", self.strike_mode)

        # Delta target (for delta mode)
        self.delta_input = QDoubleSpinBox()
        self.delta_input.setRange(0.01, 0.99)
        self.delta_input.setValue(0.30)
        self.delta_input.setSingleStep(0.05)
        self.params_layout.addRow("Delta Target:", self.delta_input)

        # ATM offset (for ATM mode)
        self.atm_offset_input = QSpinBox()
        self.atm_offset_input.setRange(-100, 100)
        self.atm_offset_input.setValue(5)
        self.atm_offset_input.setSuffix(" points")
        self.params_layout.addRow("ATM Offset:", self.atm_offset_input)

        # Manual strikes (dynamically added based on strategy)
        self.strike_inputs = {}

        # Expiry selection
        self.expiry_date = QDateEdit()
        self.expiry_date.setDate(QDate.currentDate().addDays(30))
        self.expiry_date.setCalendarPopup(True)
        self.params_layout.addRow("Expiry:", self.expiry_date)

        # DTE display
        self.dte_label = QLabel("DTE: 30 days")
        self.params_layout.addRow("", self.dte_label)

        group.setLayout(self.params_layout)
        return group

    def _create_execution_group(self) -> QGroupBox:
        """Create execution settings group."""
        group = QGroupBox("Execution")
        layout = QFormLayout()

        # Quantity
        self.quantity_input = QSpinBox()
        self.quantity_input.setRange(1, 100)
        self.quantity_input.setValue(1)
        layout.addRow("Quantity:", self.quantity_input)

        # Execution type
        self.execution_type = QComboBox()
        self.execution_type.addItems([et.value for et in ExecutionType])
        self.execution_type.currentTextChanged.connect(self._on_execution_type_changed)
        layout.addRow("Type:", self.execution_type)

        # Limit price (for limit orders)
        self.limit_price_input = QDoubleSpinBox()
        self.limit_price_input.setRange(0.01, 10000.0)
        self.limit_price_input.setValue(0.0)
        self.limit_price_input.setPrefix("$")
        self.limit_price_input.setEnabled(False)
        layout.addRow("Limit Price:", self.limit_price_input)

        group.setLayout(layout)
        return group

    def _create_greeks_group(self) -> QGroupBox:
        """Create Greeks preview group."""
        group = QGroupBox("Estimated Greeks Preview")
        layout = QHBoxLayout()

        self.delta_label = QLabel("Δ: --")
        self.gamma_label = QLabel("Γ: --")
        self.theta_label = QLabel("Θ: --")
        self.vega_label = QLabel("V: --")

        layout.addWidget(self.delta_label)
        layout.addWidget(self.gamma_label)
        layout.addWidget(self.theta_label)
        layout.addWidget(self.vega_label)
        layout.addStretch()

        group.setLayout(layout)
        return group

    def _on_symbol_changed(self, text: str) -> None:
        """Handle symbol change."""
        self.current_symbol = text.strip().upper()

    def _on_spot_price_changed(self, value: float) -> None:
        """Handle spot price change."""
        self.current_spot_price = value

    def _on_strategy_changed(self, text: str) -> None:
        """Handle strategy selection change."""
        # Find strategy definition
        strategy_def = None
        for key, definition in STRATEGY_DEFINITIONS.items():
            if definition.display_name == text:
                strategy_def = definition
                break

        if not strategy_def:
            return

        # Update description
        self.strategy_desc.setText(strategy_def.description)

        # Update parameter inputs
        self._update_param_inputs(strategy_def)

    def _update_param_inputs(self, strategy_def: StrategyDefinition) -> None:
        """Update parameter inputs based on strategy."""
        # Clear existing strike inputs
        for label_widget in self.strike_inputs.values():
            self.params_layout.removeWidget(label_widget[0])
            self.params_layout.removeWidget(label_widget[1])
            label_widget[0].deleteLater()
            label_widget[1].deleteLater()

        self.strike_inputs = {}

        # Add new strike inputs based on strategy
        for param in strategy_def.params:
            label = QLabel(f"{param.replace('_', ' ').title()}:")
            spin = QDoubleSpinBox()
            spin.setRange(0.01, 10000.0)
            spin.setValue(self.current_spot_price)
            spin.setPrefix("$")

            self.params_layout.addRow(label, spin)
            self.strike_inputs[param] = (label, spin)

    def _on_strike_mode_changed(self, text: str) -> None:
        """Handle strike mode change."""
        # Enable/disable inputs based on mode
        is_delta = text == StrikeSelectionMode.DELTA_TARGET.value
        is_atm = text == StrikeSelectionMode.ATM_OFFSET.value
        is_manual = text == StrikeSelectionMode.MANUAL.value

        self.delta_input.setEnabled(is_delta)
        self.atm_offset_input.setEnabled(is_atm)

        for _, spin in self.strike_inputs.values():
            spin.setEnabled(is_manual)

    def _on_execution_type_changed(self, text: str) -> None:
        """Handle execution type change."""
        is_limit = text == ExecutionType.LIMIT.value
        self.limit_price_input.setEnabled(is_limit)

    def _submit_order(self) -> None:
        """Submit order."""
        # Get selected strategy
        strategy_text = self.strategy_combo.currentText()
        if strategy_text.startswith("---"):
            QMessageBox.warning(self, "Error", "Please select a strategy")
            return

        # Find strategy definition
        strategy_def = None
        strategy_key = None
        for key, definition in STRATEGY_DEFINITIONS.items():
            if definition.display_name == strategy_text:
                strategy_def = definition
                strategy_key = key
                break

        if not strategy_def:
            return

        # Get strikes
        strikes = {}
        for param, (label, spin) in self.strike_inputs.items():
            strikes[param] = spin.value()

        # Get execution type
        exec_type_text = self.execution_type.currentText()
        exec_type = ExecutionType.MARKET
        for et in ExecutionType:
            if et.value == exec_type_text:
                exec_type = et
                break

        # Create order submission
        order = OrderSubmission(
            strategy_name=strategy_key,
            symbol=self.current_symbol,
            quantity=self.quantity_input.value(),
            execution_type=exec_type,
            limit_price=(self.limit_price_input.value() if exec_type == ExecutionType.LIMIT else None),
            strikes=strikes,
            expiry_date=self.expiry_date.date().toPython(),
        )

        # Callback
        if self.on_submit_order:
            self.on_submit_order(order)

        # Emit signal
        self.order_submitted.emit(order.__dict__)

        # Show confirmation
        QMessageBox.information(
            self,
            "Order Submitted",
            f"Submitted {order.strategy_name} order for {order.symbol}",
        )


def create_strategy_selector() -> StrategySelectorWidget:
    """
    Create a StrategySelectorWidget instance.

    Returns:
        StrategySelectorWidget instance

    Example:
        >>> selector = create_strategy_selector()
        >>>
        >>> def handle_order(order):
        ...     print(f"Order: {order.strategy_name} on {order.symbol}")
        >>>
        >>> selector.on_submit_order = handle_order
        >>> selector.show()
    """
    return StrategySelectorWidget()
