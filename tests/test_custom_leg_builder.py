"""
Tests for Custom Leg Builder UI Widget

Basic tests for custom multi-leg option spread construction widget.
"""

import pytest


# Check if PySide6 is available
try:
    import sys

    from PySide6.QtWidgets import QApplication

    # Create QApplication instance for testing (required for Qt widgets)
    if not QApplication.instance():
        app = QApplication(sys.argv)

    from ui.custom_leg_builder import (
        CustomLegBuilderWidget,
        OptionLeg,
        OptionType,
        PLDiagramWidget,
        Side,
        SpreadDefinition,
    )

    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False
    pytestmark = pytest.mark.skip("PySide6 not available")


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestEnums:
    """Tests for enums."""

    @pytest.mark.unit
    def test_option_type_enum(self):
        """Test OptionType enum."""
        assert OptionType.CALL.value == "Call"
        assert OptionType.PUT.value == "Put"

    @pytest.mark.unit
    def test_side_enum(self):
        """Test Side enum."""
        assert Side.BUY.value == "Buy"
        assert Side.SELL.value == "Sell"


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestOptionLeg:
    """Tests for OptionLeg dataclass."""

    @pytest.mark.unit
    def test_option_leg_creation(self):
        """Test creating an option leg."""
        leg = OptionLeg(
            option_type=OptionType.CALL,
            side=Side.BUY,
            strike=450.0,
            quantity=1,
            premium=5.50,
        )

        assert leg.option_type == OptionType.CALL
        assert leg.side == Side.BUY
        assert leg.strike == 450.0
        assert leg.quantity == 1
        assert leg.premium == 5.50

    @pytest.mark.unit
    def test_option_leg_put(self):
        """Test creating a put option leg."""
        leg = OptionLeg(
            option_type=OptionType.PUT,
            side=Side.SELL,
            strike=440.0,
            quantity=2,
            premium=3.25,
        )

        assert leg.option_type == OptionType.PUT
        assert leg.side == Side.SELL
        assert leg.quantity == 2


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestSpreadDefinition:
    """Tests for SpreadDefinition dataclass."""

    @pytest.mark.unit
    def test_spread_definition_creation(self):
        """Test creating a spread definition."""
        legs = [
            OptionLeg(OptionType.CALL, Side.BUY, 450.0, 1, 5.50),
            OptionLeg(OptionType.CALL, Side.SELL, 460.0, 1, 2.25),
        ]

        spread = SpreadDefinition(
            name="Bull Call Spread",
            legs=legs,
            net_debit=3.25,
            net_credit=0.0,
            max_profit=6.75,
            max_loss=3.25,
            breakevens=[453.25],
        )

        assert spread.name == "Bull Call Spread"
        assert len(spread.legs) == 2
        assert spread.net_debit == 3.25
        assert spread.max_profit == 6.75


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestPLDiagramWidget:
    """Tests for PLDiagramWidget."""

    @pytest.mark.unit
    def test_diagram_widget_creation(self):
        """Test P&L diagram widget creation."""
        diagram = PLDiagramWidget()

        assert diagram is not None
        assert diagram.spot_price == 450.0
        assert len(diagram.legs) == 0

    @pytest.mark.unit
    def test_set_legs(self):
        """Test setting legs for diagram."""
        diagram = PLDiagramWidget()

        legs = [
            OptionLeg(OptionType.CALL, Side.BUY, 450.0, 1, 5.50),
            OptionLeg(OptionType.CALL, Side.SELL, 460.0, 1, 2.25),
        ]

        diagram.set_legs(legs, spot_price=455.0)

        assert len(diagram.legs) == 2
        assert diagram.spot_price == 455.0


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestCustomLegBuilderWidget:
    """Tests for CustomLegBuilderWidget."""

    @pytest.mark.unit
    def test_widget_creation(self):
        """Test widget instance creation."""
        widget = CustomLegBuilderWidget()

        assert widget is not None

    @pytest.mark.unit
    def test_widget_has_required_attributes(self):
        """Test widget has required attributes."""
        widget = CustomLegBuilderWidget()

        # Should have spot price input
        assert hasattr(widget, "spot_input")
        assert widget.spot_input is not None

        # Should have legs table
        assert hasattr(widget, "legs_table")
        assert widget.legs_table is not None

        # Should have P&L diagram
        assert hasattr(widget, "pl_diagram")
        assert widget.pl_diagram is not None

    @pytest.mark.unit
    def test_default_spot_price(self):
        """Test default spot price."""
        widget = CustomLegBuilderWidget()

        # Default spot price should be 450.0
        assert widget.spot_input.value() == 450.0

    @pytest.mark.unit
    def test_initial_empty_legs(self):
        """Test widget starts with no legs."""
        widget = CustomLegBuilderWidget()

        # Should start with 0 legs
        assert widget.legs_table.rowCount() == 0
