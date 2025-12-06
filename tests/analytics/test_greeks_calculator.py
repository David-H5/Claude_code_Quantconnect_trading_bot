"""
Tests for Greeks Calculator Module

UPGRADE-015 Phase 8: Options Analytics Engine

Tests cover:
- First-order Greeks (Delta, Gamma, Theta, Vega, Rho)
- Second-order Greeks
- Position aggregation
- Scenario analysis
"""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analytics.greeks_calculator import (
    GreeksCalculator,
    GreeksResult,
    OptionType,
    create_greeks_calculator,
)


class TestGreeksResult:
    """Test GreeksResult dataclass."""

    def test_result_creation(self):
        """Test creating Greeks result."""
        result = GreeksResult(
            delta=0.5,
            gamma=0.05,
            theta=-0.02,
            vega=0.15,
            option_type=OptionType.CALL,
        )

        assert result.delta == 0.5
        assert result.gamma == 0.05
        assert result.theta == -0.02
        assert result.vega == 0.15

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = GreeksResult(
            delta=0.5,
            option_type=OptionType.CALL,
            spot=100,
            strike=100,
        )

        data = result.to_dict()

        assert data["delta"] == 0.5
        assert data["option_type"] == "call"


class TestGreeksCalculator:
    """Test GreeksCalculator class."""

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calc = GreeksCalculator(risk_free_rate=0.05, dividend_yield=0.02)

        assert calc.risk_free_rate == 0.05
        assert calc.dividend_yield == 0.02

    def test_atm_call_delta(self):
        """Test ATM call delta is approximately 0.5."""
        calc = GreeksCalculator()

        result = calc.calculate(
            spot=100,
            strike=100,
            time_to_expiry=30 / 365,
            iv=0.25,
            option_type=OptionType.CALL,
        )

        # ATM call delta should be close to 0.5
        assert 0.45 < result.delta < 0.55

    def test_atm_put_delta(self):
        """Test ATM put delta is approximately -0.5."""
        calc = GreeksCalculator()

        result = calc.calculate(
            spot=100,
            strike=100,
            time_to_expiry=30 / 365,
            iv=0.25,
            option_type=OptionType.PUT,
        )

        # ATM put delta should be close to -0.5
        assert -0.55 < result.delta < -0.45

    def test_itm_call_delta(self):
        """Test ITM call delta is high."""
        calc = GreeksCalculator()

        result = calc.calculate(
            spot=110,
            strike=100,
            time_to_expiry=30 / 365,
            iv=0.25,
            option_type=OptionType.CALL,
        )

        # ITM call delta should be high
        assert result.delta > 0.7

    def test_otm_call_delta(self):
        """Test OTM call delta is low."""
        calc = GreeksCalculator()

        result = calc.calculate(
            spot=90,
            strike=100,
            time_to_expiry=30 / 365,
            iv=0.25,
            option_type=OptionType.CALL,
        )

        # OTM call delta should be low
        assert result.delta < 0.3

    def test_gamma_positive(self):
        """Test gamma is positive."""
        calc = GreeksCalculator()

        result = calc.calculate(
            spot=100,
            strike=100,
            time_to_expiry=30 / 365,
            iv=0.25,
            option_type=OptionType.CALL,
        )

        assert result.gamma > 0

    def test_gamma_same_for_call_put(self):
        """Test gamma is the same for calls and puts at same strike."""
        calc = GreeksCalculator()

        call_result = calc.calculate(
            spot=100,
            strike=100,
            time_to_expiry=30 / 365,
            iv=0.25,
            option_type=OptionType.CALL,
        )

        put_result = calc.calculate(
            spot=100,
            strike=100,
            time_to_expiry=30 / 365,
            iv=0.25,
            option_type=OptionType.PUT,
        )

        assert abs(call_result.gamma - put_result.gamma) < 0.001

    def test_theta_negative_for_long(self):
        """Test theta is negative for long options."""
        calc = GreeksCalculator()

        result = calc.calculate(
            spot=100,
            strike=100,
            time_to_expiry=30 / 365,
            iv=0.25,
            option_type=OptionType.CALL,
            quantity=1,
        )

        assert result.theta < 0

    def test_vega_positive(self):
        """Test vega is positive."""
        calc = GreeksCalculator()

        result = calc.calculate(
            spot=100,
            strike=100,
            time_to_expiry=30 / 365,
            iv=0.25,
            option_type=OptionType.CALL,
        )

        assert result.vega > 0

    def test_vega_same_for_call_put(self):
        """Test vega is the same for calls and puts."""
        calc = GreeksCalculator()

        call_result = calc.calculate(
            spot=100,
            strike=100,
            time_to_expiry=30 / 365,
            iv=0.25,
            option_type=OptionType.CALL,
        )

        put_result = calc.calculate(
            spot=100,
            strike=100,
            time_to_expiry=30 / 365,
            iv=0.25,
            option_type=OptionType.PUT,
        )

        assert abs(call_result.vega - put_result.vega) < 0.001

    def test_rho_call_positive(self):
        """Test rho is positive for calls."""
        calc = GreeksCalculator()

        result = calc.calculate(
            spot=100,
            strike=100,
            time_to_expiry=30 / 365,
            iv=0.25,
            option_type=OptionType.CALL,
        )

        assert result.rho > 0

    def test_rho_put_negative(self):
        """Test rho is negative for puts."""
        calc = GreeksCalculator()

        result = calc.calculate(
            spot=100,
            strike=100,
            time_to_expiry=30 / 365,
            iv=0.25,
            option_type=OptionType.PUT,
        )

        assert result.rho < 0

    def test_dollar_greeks(self):
        """Test dollar Greeks calculation."""
        calc = GreeksCalculator()

        result = calc.calculate(
            spot=100,
            strike=100,
            time_to_expiry=30 / 365,
            iv=0.25,
            option_type=OptionType.CALL,
        )

        # Dollar delta should be delta * spot
        assert abs(result.dollar_delta - result.delta * 100) < 0.01

    def test_edge_case_zero_time(self):
        """Test edge case with zero time to expiry."""
        calc = GreeksCalculator()

        result = calc.calculate(
            spot=100,
            strike=100,
            time_to_expiry=0,
            iv=0.25,
            option_type=OptionType.CALL,
        )

        # At expiration, all greeks should be 0 (except intrinsic)
        assert result.delta == 0

    def test_edge_case_zero_iv(self):
        """Test edge case with zero IV."""
        calc = GreeksCalculator()

        result = calc.calculate(
            spot=100,
            strike=100,
            time_to_expiry=30 / 365,
            iv=0,
            option_type=OptionType.CALL,
        )

        # Should return zeros
        assert result.delta == 0


class TestPositionAggregation:
    """Test position aggregation."""

    def test_aggregate_single_position(self):
        """Test aggregating a single position."""
        calc = GreeksCalculator()

        options = [
            {
                "spot": 100,
                "strike": 100,
                "dte": 30,
                "iv": 0.25,
                "type": "call",
                "quantity": 1,
            }
        ]

        result = calc.aggregate_position(options)

        assert result.num_positions == 1
        assert result.net_delta > 0

    def test_aggregate_multiple_positions(self):
        """Test aggregating multiple positions."""
        calc = GreeksCalculator()

        options = [
            {"spot": 100, "strike": 100, "dte": 30, "iv": 0.25, "type": "call", "quantity": 1},
            {"spot": 100, "strike": 100, "dte": 30, "iv": 0.25, "type": "put", "quantity": 1},
        ]

        result = calc.aggregate_position(options)

        assert result.num_positions == 2
        # Straddle should have near-zero delta
        assert abs(result.net_delta) < 0.1

    def test_aggregate_delta_neutral(self):
        """Test delta-neutral position."""
        calc = GreeksCalculator()

        # Long call, short call at same strike
        options = [
            {"spot": 100, "strike": 100, "dte": 30, "iv": 0.25, "type": "call", "quantity": 1},
            {"spot": 100, "strike": 100, "dte": 30, "iv": 0.25, "type": "call", "quantity": -1},
        ]

        result = calc.aggregate_position(options)

        assert abs(result.net_delta) < 0.001


class TestScenarioAnalysis:
    """Test scenario analysis."""

    def test_scenario_analysis(self):
        """Test basic scenario analysis."""
        calc = GreeksCalculator()

        option = {
            "spot": 100,
            "strike": 100,
            "dte": 30,
            "iv": 0.25,
            "type": "call",
        }

        result = calc.scenario_analysis(option)

        assert "base_greeks" in result
        assert "spot_scenarios" in result
        assert "iv_scenarios" in result

    def test_spot_scenarios(self):
        """Test spot price scenarios."""
        calc = GreeksCalculator()

        option = {
            "spot": 100,
            "strike": 100,
            "dte": 30,
            "iv": 0.25,
            "type": "call",
        }

        result = calc.scenario_analysis(option, spot_changes=[-0.1, 0, 0.1])

        spot_scenarios = result["spot_scenarios"]
        assert len(spot_scenarios) == 3

        # Delta should increase with spot for calls
        deltas = [s["delta"] for s in spot_scenarios]
        assert deltas[-1] > deltas[0]


class TestCreateGreeksCalculator:
    """Test factory function."""

    def test_create_with_defaults(self):
        """Test creating calculator with defaults."""
        calc = create_greeks_calculator()

        assert calc.risk_free_rate == 0.05
        assert calc.dividend_yield == 0.0

    def test_create_with_custom(self):
        """Test creating calculator with custom values."""
        calc = create_greeks_calculator(risk_free_rate=0.03, dividend_yield=0.01)

        assert calc.risk_free_rate == 0.03
        assert calc.dividend_yield == 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
