"""
Tests for Strategy Selector UI Widget

Basic tests for dropdown strategy selection widget.
"""

from datetime import datetime

import pytest


# Check if PySide6 is available
try:
    import sys

    from PySide6.QtWidgets import QApplication

    # Create QApplication instance for testing (required for Qt widgets)
    if not QApplication.instance():
        app = QApplication(sys.argv)

    from ui.strategy_selector import (
        STRATEGY_DEFINITIONS,
        ExecutionType,
        OrderSubmission,
        StrategyDefinition,
        StrategySelectorWidget,
        StrikeSelectionMode,
    )

    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False
    pytestmark = pytest.mark.skip("PySide6 not available")


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestStrategyDefinitions:
    """Tests for strategy definitions."""

    @pytest.mark.unit
    def test_strategy_definitions_exist(self):
        """Test that strategy definitions are loaded."""
        assert len(STRATEGY_DEFINITIONS) > 0

    @pytest.mark.unit
    def test_strategy_definition_structure(self):
        """Test strategy definition dataclass structure."""
        # Pick any strategy
        strategy = STRATEGY_DEFINITIONS["iron_condor"]

        assert hasattr(strategy, "name")
        assert hasattr(strategy, "display_name")
        assert hasattr(strategy, "legs")

    @pytest.mark.unit
    def test_execution_types(self):
        """Test ExecutionType enum."""
        assert ExecutionType.MARKET.value == "Market"
        assert ExecutionType.LIMIT.value == "Limit"
        assert ExecutionType.TWO_PART.value == "Two-Part Spread"

    @pytest.mark.unit
    def test_strike_selection_modes(self):
        """Test StrikeSelectionMode enum."""
        assert StrikeSelectionMode.DELTA_TARGET.value == "Delta Target"
        assert StrikeSelectionMode.ATM_OFFSET.value == "ATM Offset"
        assert StrikeSelectionMode.MANUAL.value == "Manual Entry"


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestStrategySelectorWidget:
    """Tests for StrategySelectorWidget."""

    @pytest.mark.unit
    def test_widget_creation(self):
        """Test widget instance creation."""
        widget = StrategySelectorWidget()

        assert widget is not None
        assert widget.on_submit_order is None

    @pytest.mark.unit
    def test_widget_has_required_attributes(self):
        """Test widget has required attributes."""
        widget = StrategySelectorWidget()

        # Should have symbol input
        assert hasattr(widget, "symbol_input")
        assert widget.symbol_input is not None

        # Should have spot price input
        assert hasattr(widget, "spot_price_input")
        assert widget.spot_price_input is not None

        # Should have strategy combo
        assert hasattr(widget, "strategy_combo")
        assert widget.strategy_combo is not None

    @pytest.mark.unit
    def test_default_symbol(self):
        """Test default symbol is SPY."""
        widget = StrategySelectorWidget()

        assert widget.current_symbol == "SPY"

    @pytest.mark.unit
    def test_default_spot_price(self):
        """Test default spot price."""
        widget = StrategySelectorWidget()

        assert widget.current_spot_price == 450.0


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestOrderSubmission:
    """Tests for OrderSubmission dataclass."""

    @pytest.mark.unit
    def test_order_submission_creation(self):
        """Test creating an order submission."""
        submission = OrderSubmission(
            strategy_name="iron_condor",
            symbol="SPY",
            quantity=1,
            execution_type=ExecutionType.LIMIT,
            limit_price=1.50,
            strikes={"put_buy": 440.0, "put_sell": 445.0},
            expiry_date=datetime(2025, 12, 19),
        )

        assert submission.strategy_name == "iron_condor"
        assert submission.symbol == "SPY"
        assert submission.quantity == 1
        assert submission.limit_price == 1.50

    @pytest.mark.unit
    def test_order_submission_with_greeks(self):
        """Test order submission with estimated Greeks."""
        submission = OrderSubmission(
            strategy_name="bull_call_spread",
            symbol="AAPL",
            quantity=2,
            execution_type=ExecutionType.MARKET,
            limit_price=None,
            strikes={"buy_lower": 170.0, "sell_higher": 180.0},
            expiry_date=datetime(2025, 12, 19),
            estimated_greeks={"delta": 0.30, "theta": -5.0},
        )

        assert submission.estimated_greeks is not None
        assert submission.estimated_greeks["delta"] == 0.30
