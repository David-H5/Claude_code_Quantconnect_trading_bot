"""
Tests for Portfolio Greeks Hedging Module (Sprint 5 Expansion)

Tests portfolio-level Greeks aggregation and hedging recommendations.
Part of UPGRADE-010 Sprint 5 - Quality & Test Coverage.
"""

from datetime import datetime, timedelta

import pytest

from models.portfolio_hedging import (
    HedgeRecommendation,
    HedgeTargets,
    HedgeType,
    PortfolioHedger,
    Position,
    create_hedger_from_positions,
)


class TestHedgeType:
    """Tests for HedgeType enum."""

    def test_values(self):
        """Test enum values."""
        assert HedgeType.DELTA.value == "delta"
        assert HedgeType.GAMMA.value == "gamma"
        assert HedgeType.VEGA.value == "vega"
        assert HedgeType.DELTA_GAMMA.value == "delta_gamma"
        assert HedgeType.FULL.value == "full"


class TestPosition:
    """Tests for Position dataclass."""

    @pytest.fixture
    def stock_position(self):
        """Create stock position."""
        return Position(
            symbol="SPY",
            asset_type="stock",
            quantity=100,
            entry_price=450.0,
            current_price=455.0,
            underlying_symbol="SPY",
        )

    @pytest.fixture
    def option_position(self):
        """Create option position."""
        return Position(
            symbol="SPY_C460",
            asset_type="option",
            quantity=2,
            entry_price=5.00,
            current_price=6.50,
            underlying_symbol="SPY",
            delta=0.45,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
            strike=460.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type="call",
        )

    def test_stock_creation(self, stock_position):
        """Test stock position creation."""
        assert stock_position.symbol == "SPY"
        assert stock_position.asset_type == "stock"
        assert stock_position.quantity == 100
        assert stock_position.delta == 1.0  # Default for stock

    def test_option_creation(self, option_position):
        """Test option position creation."""
        assert option_position.symbol == "SPY_C460"
        assert option_position.asset_type == "option"
        assert option_position.delta == 0.45
        assert option_position.gamma == 0.02

    def test_position_delta_stock(self, stock_position):
        """Test position delta for stock."""
        # Stock: quantity * 1 (delta)
        assert stock_position.position_delta == 100

    def test_position_delta_option(self, option_position):
        """Test position delta for option."""
        # Option: delta * quantity * 100 = 0.45 * 2 * 100 = 90
        assert option_position.position_delta == pytest.approx(90.0, rel=0.01)

    def test_position_gamma_stock(self, stock_position):
        """Test position gamma for stock (always 0)."""
        assert stock_position.position_gamma == 0.0

    def test_position_gamma_option(self, option_position):
        """Test position gamma for option."""
        # gamma * quantity * 100 = 0.02 * 2 * 100 = 4
        assert option_position.position_gamma == pytest.approx(4.0, rel=0.01)

    def test_position_theta_stock(self, stock_position):
        """Test position theta for stock (always 0)."""
        assert stock_position.position_theta == 0.0

    def test_position_theta_option(self, option_position):
        """Test position theta for option."""
        # theta * quantity * 100 = -0.05 * 2 * 100 = -10
        assert option_position.position_theta == pytest.approx(-10.0, rel=0.01)

    def test_position_vega_stock(self, stock_position):
        """Test position vega for stock (always 0)."""
        assert stock_position.position_vega == 0.0

    def test_position_vega_option(self, option_position):
        """Test position vega for option."""
        # vega * quantity * 100 = 0.15 * 2 * 100 = 30
        assert option_position.position_vega == pytest.approx(30.0, rel=0.01)

    def test_market_value_stock(self, stock_position):
        """Test market value for stock."""
        # quantity * price = 100 * 455 = 45500
        assert stock_position.market_value == pytest.approx(45500.0, rel=0.01)

    def test_market_value_option(self, option_position):
        """Test market value for option."""
        # quantity * price * 100 = 2 * 6.50 * 100 = 1300
        assert option_position.market_value == pytest.approx(1300.0, rel=0.01)

    def test_to_dict(self, option_position):
        """Test conversion to dictionary."""
        d = option_position.to_dict()

        assert d["symbol"] == "SPY_C460"
        assert d["type"] == "option"
        assert d["quantity"] == 2
        assert d["delta"] == pytest.approx(90.0, rel=0.01)
        assert "market_value" in d


class TestHedgeRecommendation:
    """Tests for HedgeRecommendation dataclass."""

    def test_creation(self):
        """Test recommendation creation."""
        rec = HedgeRecommendation(
            hedge_type=HedgeType.DELTA,
            action="buy",
            symbol="SPY",
            asset_type="stock",
            quantity=50,
            rationale="Delta hedge",
            expected_delta_change=-100,
            estimated_cost=22500.0,
            priority=1,
        )

        assert rec.hedge_type == HedgeType.DELTA
        assert rec.action == "buy"
        assert rec.quantity == 50
        assert rec.priority == 1

    def test_defaults(self):
        """Test default values."""
        rec = HedgeRecommendation(
            hedge_type=HedgeType.GAMMA,
            action="sell",
            symbol="SPY_STRADDLE",
            asset_type="option",
            quantity=5,
            rationale="Gamma hedge",
        )

        assert rec.expected_delta_change == 0.0
        assert rec.expected_gamma_change == 0.0
        assert rec.expected_vega_change == 0.0
        assert rec.estimated_cost == 0.0
        assert rec.priority == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rec = HedgeRecommendation(
            hedge_type=HedgeType.VEGA,
            action="buy",
            symbol="SPY_60DTE",
            asset_type="option",
            quantity=10,
            rationale="Vega hedge",
            expected_vega_change=100.0,
        )

        d = rec.to_dict()

        assert d["hedge_type"] == "vega"
        assert d["action"] == "buy"
        assert d["symbol"] == "SPY_60DTE"
        assert d["vega_change"] == 100.0


class TestHedgeTargets:
    """Tests for HedgeTargets dataclass."""

    def test_defaults(self):
        """Test default values (delta neutral)."""
        targets = HedgeTargets()

        assert targets.target_delta == 0.0
        assert targets.target_gamma == 0.0
        assert targets.target_vega == 0.0
        assert targets.delta_tolerance == 50.0
        assert targets.gamma_tolerance == 10.0
        assert targets.vega_tolerance == 100.0

    def test_custom_values(self):
        """Test custom target values."""
        targets = HedgeTargets(
            target_delta=100.0,  # Slightly long
            target_gamma=5.0,
            target_vega=50.0,
            delta_tolerance=25.0,
        )

        assert targets.target_delta == 100.0
        assert targets.delta_tolerance == 25.0


class TestPortfolioHedger:
    """Tests for PortfolioHedger class."""

    @pytest.fixture
    def hedger(self):
        """Create default hedger."""
        return PortfolioHedger()

    @pytest.fixture
    def stock_position(self):
        """Create stock position."""
        return Position(
            symbol="SPY",
            asset_type="stock",
            quantity=100,
            entry_price=450.0,
            current_price=455.0,
            underlying_symbol="SPY",
        )

    @pytest.fixture
    def call_position(self):
        """Create long call position."""
        return Position(
            symbol="SPY_C460",
            asset_type="option",
            quantity=2,
            entry_price=5.00,
            current_price=6.50,
            underlying_symbol="SPY",
            delta=0.45,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
            strike=460.0,
            option_type="call",
        )

    @pytest.fixture
    def put_position(self):
        """Create short put position."""
        return Position(
            symbol="SPY_P440",
            asset_type="option",
            quantity=-1,  # Short
            entry_price=4.00,
            current_price=3.50,
            underlying_symbol="SPY",
            delta=-0.35,  # Put delta
            gamma=0.018,
            theta=-0.04,
            vega=0.12,
            strike=440.0,
            option_type="put",
        )

    def test_initialization(self, hedger):
        """Test hedger initialization."""
        assert len(hedger.positions) == 0
        assert hedger.targets.target_delta == 0.0

    def test_custom_targets(self):
        """Test initialization with custom targets."""
        targets = HedgeTargets(target_delta=50.0)
        hedger = PortfolioHedger(targets=targets)

        assert hedger.targets.target_delta == 50.0

    def test_add_position(self, hedger, stock_position):
        """Test adding position."""
        hedger.add_position(stock_position)

        assert "SPY" in hedger.positions
        assert hedger.positions["SPY"].quantity == 100

    def test_add_position_updates_underlying_prices(self, hedger, call_position):
        """Test that adding position updates underlying prices."""
        hedger.add_position(call_position)

        assert "SPY" in hedger.underlying_prices
        assert hedger.underlying_prices["SPY"] == 6.50

    def test_remove_position(self, hedger, stock_position):
        """Test removing position."""
        hedger.add_position(stock_position)
        hedger.remove_position("SPY")

        assert "SPY" not in hedger.positions

    def test_remove_nonexistent_position(self, hedger):
        """Test removing nonexistent position (no error)."""
        hedger.remove_position("UNKNOWN")  # Should not raise

    def test_update_position_greeks(self, hedger, call_position):
        """Test updating position Greeks."""
        hedger.add_position(call_position)
        hedger.update_position_greeks(
            symbol="SPY_C460",
            delta=0.50,
            gamma=0.025,
            theta=-0.06,
            vega=0.16,
            current_price=7.00,
        )

        pos = hedger.positions["SPY_C460"]
        assert pos.delta == 0.50
        assert pos.gamma == 0.025
        assert pos.current_price == 7.00

    def test_update_position_greeks_unknown(self, hedger):
        """Test updating Greeks for unknown position (no error)."""
        hedger.update_position_greeks(
            symbol="UNKNOWN",
            delta=0.5,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
            current_price=5.00,
        )
        # Should not raise

    def test_get_portfolio_greeks_empty(self, hedger):
        """Test portfolio Greeks with no positions."""
        greeks = hedger.get_portfolio_greeks()

        assert greeks.delta == 0.0
        assert greeks.gamma == 0.0
        assert greeks.theta == 0.0
        assert greeks.vega == 0.0

    def test_get_portfolio_greeks_stock_only(self, hedger, stock_position):
        """Test portfolio Greeks with stock only."""
        hedger.add_position(stock_position)
        greeks = hedger.get_portfolio_greeks()

        assert greeks.delta == 100  # 100 shares
        assert greeks.gamma == 0.0  # Stock has no gamma
        assert greeks.theta == 0.0
        assert greeks.vega == 0.0

    def test_get_portfolio_greeks_mixed(self, hedger, stock_position, call_position):
        """Test portfolio Greeks with mixed positions."""
        hedger.add_position(stock_position)
        hedger.add_position(call_position)
        greeks = hedger.get_portfolio_greeks()

        # Stock: 100 delta + Option: 90 delta = 190
        assert greeks.delta == pytest.approx(190.0, rel=0.01)
        # Option only: 4 gamma
        assert greeks.gamma == pytest.approx(4.0, rel=0.01)

    def test_get_portfolio_greeks_filtered(self, hedger, stock_position, call_position):
        """Test portfolio Greeks filtered by underlying."""
        hedger.add_position(stock_position)
        hedger.add_position(call_position)

        # Add another underlying
        qqq_position = Position(
            symbol="QQQ",
            asset_type="stock",
            quantity=50,
            entry_price=380.0,
            current_price=385.0,
            underlying_symbol="QQQ",
        )
        hedger.add_position(qqq_position)

        # Filter by SPY
        spy_greeks = hedger.get_portfolio_greeks("SPY")
        assert spy_greeks.delta == pytest.approx(190.0, rel=0.01)

        # Filter by QQQ
        qqq_greeks = hedger.get_portfolio_greeks("QQQ")
        assert qqq_greeks.delta == 50

    def test_get_greeks_by_underlying(self, hedger, stock_position, call_position):
        """Test Greeks breakdown by underlying."""
        hedger.add_position(stock_position)
        hedger.add_position(call_position)

        by_underlying = hedger.get_greeks_by_underlying()

        assert "SPY" in by_underlying
        assert by_underlying["SPY"].delta == pytest.approx(190.0, rel=0.01)

    def test_calculate_delta_hedge_within_tolerance(self, hedger, stock_position):
        """Test delta hedge when within tolerance."""
        # Set targets with high tolerance
        hedger.targets = HedgeTargets(target_delta=100.0, delta_tolerance=50.0)
        hedger.add_position(stock_position)

        # 100 delta vs 100 target = 0 exposure (within 50 tolerance)
        rec = hedger.calculate_delta_hedge("SPY")

        assert rec is None

    def test_calculate_delta_hedge_stock(self, hedger, stock_position, call_position):
        """Test delta hedge recommendation with stock."""
        hedger.add_position(stock_position)
        hedger.add_position(call_position)
        hedger.underlying_prices["SPY"] = 455.0

        # Delta: 190, Target: 0, Exposure: 190
        rec = hedger.calculate_delta_hedge("SPY", use_options=False)

        assert rec is not None
        assert rec.hedge_type == HedgeType.DELTA
        assert rec.action == "sell"  # Need to sell to reduce delta
        assert rec.asset_type == "stock"
        assert rec.quantity > 0

    def test_calculate_delta_hedge_options(self, hedger, stock_position, call_position):
        """Test delta hedge recommendation with options."""
        hedger.add_position(stock_position)
        hedger.add_position(call_position)

        rec = hedger.calculate_delta_hedge("SPY", use_options=True)

        assert rec is not None
        assert rec.hedge_type == HedgeType.DELTA
        assert rec.asset_type == "option"
        assert "ATM" in rec.symbol

    def test_calculate_gamma_hedge(self, hedger):
        """Test gamma hedge recommendation."""
        # Create short put with high gamma exposure
        # Need abs(gamma_exposure) >= 5.0 for contracts >= 1
        # position_gamma = quantity * gamma * 100
        high_gamma_position = Position(
            symbol="SPY_P440",
            asset_type="option",
            quantity=-5,  # 5 contracts short
            entry_price=4.00,
            current_price=3.50,
            underlying_symbol="SPY",
            delta=-0.35,
            gamma=0.02,  # Higher gamma per contract
            theta=-0.04,
            vega=0.12,
            strike=440.0,
            option_type="put",
        )
        # position_gamma = -5 * 0.02 * 100 = -10.0
        hedger.add_position(high_gamma_position)
        hedger.targets = HedgeTargets(gamma_tolerance=0.5)

        rec = hedger.calculate_gamma_hedge("SPY")

        assert rec is not None
        assert rec.hedge_type == HedgeType.GAMMA
        # Short options = need to buy to increase gamma
        assert rec.action == "buy"

    def test_calculate_gamma_hedge_within_tolerance(self, hedger, call_position):
        """Test gamma hedge when within tolerance."""
        hedger.add_position(call_position)
        hedger.targets = HedgeTargets(gamma_tolerance=10.0)

        rec = hedger.calculate_gamma_hedge("SPY")

        assert rec is None

    def test_calculate_vega_hedge(self, hedger, put_position):
        """Test vega hedge recommendation."""
        # Short put = negative vega
        hedger.add_position(put_position)
        hedger.targets = HedgeTargets(vega_tolerance=5.0)

        rec = hedger.calculate_vega_hedge("SPY")

        assert rec is not None
        assert rec.hedge_type == HedgeType.VEGA
        # Short options = need to buy to increase vega
        assert rec.action == "buy"

    def test_calculate_vega_hedge_within_tolerance(self, hedger, call_position):
        """Test vega hedge when within tolerance."""
        hedger.add_position(call_position)
        hedger.targets = HedgeTargets(vega_tolerance=100.0)

        rec = hedger.calculate_vega_hedge("SPY")

        assert rec is None

    def test_get_all_hedge_recommendations(self, hedger, stock_position, put_position):
        """Test getting all hedge recommendations."""
        hedger.add_position(stock_position)
        hedger.add_position(put_position)
        hedger.targets = HedgeTargets(
            delta_tolerance=10.0,
            gamma_tolerance=1.0,
            vega_tolerance=5.0,
        )

        recommendations = hedger.get_all_hedge_recommendations()

        # Should have recommendations sorted by priority
        assert len(recommendations) > 0
        # First should be highest priority
        assert recommendations[0].priority <= recommendations[-1].priority

    def test_get_all_hedge_recommendations_specific_underlying(self, hedger, stock_position):
        """Test getting hedge recommendations for specific underlying."""
        hedger.add_position(stock_position)
        hedger.targets = HedgeTargets(delta_tolerance=10.0)

        recommendations = hedger.get_all_hedge_recommendations(underlying="SPY")

        assert len(recommendations) > 0
        for rec in recommendations:
            assert "SPY" in rec.symbol or rec.symbol == "SPY"

    def test_is_delta_neutral_true(self, hedger):
        """Test is_delta_neutral when neutral."""
        hedger.targets = HedgeTargets(delta_tolerance=50.0)

        # No positions = 0 delta = neutral
        assert hedger.is_delta_neutral() is True

    def test_is_delta_neutral_false(self, hedger, stock_position):
        """Test is_delta_neutral when not neutral."""
        hedger.add_position(stock_position)
        hedger.targets = HedgeTargets(delta_tolerance=50.0)

        # 100 delta > 50 tolerance
        assert hedger.is_delta_neutral() is False

    def test_is_gamma_neutral_true(self, hedger):
        """Test is_gamma_neutral when neutral."""
        hedger.targets = HedgeTargets(gamma_tolerance=10.0)

        assert hedger.is_gamma_neutral() is True

    def test_is_gamma_neutral_false(self, hedger, call_position):
        """Test is_gamma_neutral when not neutral."""
        hedger.add_position(call_position)
        hedger.targets = HedgeTargets(gamma_tolerance=1.0)

        # 4 gamma > 1 tolerance
        assert hedger.is_gamma_neutral() is False

    def test_integrate_with_algorithm(self, hedger):
        """Test algorithm integration."""

        class MockAlgorithm:
            def Debug(self, msg):
                self.last_debug = msg

        algo = MockAlgorithm()
        hedger.integrate_with_algorithm(algo)

        assert hedger.algorithm is algo
        assert "integrated" in algo.last_debug.lower()

    def test_get_portfolio_summary(self, hedger, stock_position, call_position):
        """Test portfolio summary generation."""
        hedger.add_position(stock_position)
        hedger.add_position(call_position)

        summary = hedger.get_portfolio_summary()

        assert "timestamp" in summary
        assert "aggregate_greeks" in summary
        assert "by_underlying" in summary
        assert "targets" in summary
        assert "risk_metrics" in summary
        assert "hedge_recommendations" in summary
        assert "positions" in summary


class TestCreateHedgerFromPositions:
    """Tests for factory function."""

    def test_empty_positions(self):
        """Test creation from empty list."""
        hedger = create_hedger_from_positions([])

        assert len(hedger.positions) == 0

    def test_stock_positions(self):
        """Test creation from stock positions."""
        positions = [
            {
                "symbol": "SPY",
                "type": "stock",
                "quantity": 100,
                "entry_price": 450.0,
                "current_price": 455.0,
                "underlying": "SPY",
            },
            {
                "symbol": "QQQ",
                "asset_type": "stock",  # Alternative key name
                "quantity": 50,
                "entry_price": 380.0,
                "price": 385.0,  # Alternative key name
            },
        ]

        hedger = create_hedger_from_positions(positions)

        assert "SPY" in hedger.positions
        assert "QQQ" in hedger.positions
        assert hedger.positions["SPY"].quantity == 100
        assert hedger.positions["QQQ"].current_price == 385.0

    def test_option_positions(self):
        """Test creation from option positions."""
        positions = [
            {
                "symbol": "SPY_C460",
                "type": "option",
                "quantity": 2,
                "entry_price": 5.00,
                "current_price": 6.50,
                "underlying": "SPY",
                "delta": 0.45,
                "gamma": 0.02,
                "theta": -0.05,
                "vega": 0.15,
                "strike": 460.0,
                "option_type": "call",
            }
        ]

        hedger = create_hedger_from_positions(positions)

        assert "SPY_C460" in hedger.positions
        pos = hedger.positions["SPY_C460"]
        assert pos.delta == 0.45
        assert pos.strike == 460.0
        assert pos.option_type == "call"

    def test_with_expiry_datetime(self):
        """Test creation with datetime expiry."""
        expiry = datetime.now() + timedelta(days=30)
        positions = [
            {
                "symbol": "SPY_C460",
                "type": "option",
                "quantity": 1,
                "entry_price": 5.00,
                "current_price": 6.50,
                "expiry": expiry,
            }
        ]

        hedger = create_hedger_from_positions(positions)

        assert hedger.positions["SPY_C460"].expiry == expiry

    def test_with_expiry_string(self):
        """Test creation with ISO string expiry."""
        expiry_str = "2024-02-15T00:00:00"
        positions = [
            {
                "symbol": "SPY_C460",
                "type": "option",
                "quantity": 1,
                "entry_price": 5.00,
                "current_price": 6.50,
                "expiry": expiry_str,
            }
        ]

        hedger = create_hedger_from_positions(positions)

        assert hedger.positions["SPY_C460"].expiry is not None

    def test_with_custom_targets(self):
        """Test creation with custom targets."""
        positions = [
            {
                "symbol": "SPY",
                "type": "stock",
                "quantity": 100,
                "entry_price": 450.0,
                "current_price": 455.0,
            }
        ]
        targets = HedgeTargets(target_delta=50.0)

        hedger = create_hedger_from_positions(positions, targets=targets)

        assert hedger.targets.target_delta == 50.0


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_covered_call_portfolio(self):
        """Test hedger with covered call portfolio."""
        hedger = PortfolioHedger()

        # Long 100 shares SPY
        hedger.add_position(
            Position(
                symbol="SPY",
                asset_type="stock",
                quantity=100,
                entry_price=450.0,
                current_price=455.0,
                underlying_symbol="SPY",
            )
        )

        # Short 1 call (covered)
        hedger.add_position(
            Position(
                symbol="SPY_C460",
                asset_type="option",
                quantity=-1,  # Short
                entry_price=5.00,
                current_price=4.00,
                underlying_symbol="SPY",
                delta=0.45,
                gamma=0.02,
                theta=-0.05,
                vega=0.15,
                strike=460.0,
                option_type="call",
            )
        )

        greeks = hedger.get_portfolio_greeks()

        # Net delta = 100 - 45 = 55
        assert greeks.delta == pytest.approx(55.0, rel=0.01)
        # Short option = negative gamma
        assert greeks.gamma < 0

    def test_iron_condor_portfolio(self):
        """Test hedger with iron condor portfolio."""
        hedger = PortfolioHedger()

        # Iron condor legs
        legs = [
            # Short put spread
            ("SPY_P445", -1, -0.25, 0.015),  # Short put
            ("SPY_P440", 1, -0.15, 0.012),  # Long put
            # Short call spread
            ("SPY_C465", -1, 0.25, 0.015),  # Short call
            ("SPY_C470", 1, 0.15, 0.012),  # Long call
        ]

        for symbol, qty, delta, gamma in legs:
            hedger.add_position(
                Position(
                    symbol=symbol,
                    asset_type="option",
                    quantity=qty,
                    entry_price=2.00,
                    current_price=1.50,
                    underlying_symbol="SPY",
                    delta=delta,
                    gamma=gamma,
                    theta=-0.03,
                    vega=0.10,
                )
            )

        greeks = hedger.get_portfolio_greeks()

        # Iron condor should be roughly delta neutral
        assert abs(greeks.delta) < 20

    def test_hedge_workflow(self):
        """Test complete hedging workflow."""
        # Create hedger with strict tolerances
        targets = HedgeTargets(
            delta_tolerance=20.0,
            gamma_tolerance=2.0,
            vega_tolerance=20.0,
        )
        hedger = PortfolioHedger(targets=targets)

        # Add unhedged long position
        hedger.add_position(
            Position(
                symbol="SPY_C460",
                asset_type="option",
                quantity=5,
                entry_price=5.00,
                current_price=6.50,
                underlying_symbol="SPY",
                delta=0.50,
                gamma=0.02,
                theta=-0.05,
                vega=0.15,
            )
        )
        hedger.underlying_prices["SPY"] = 455.0

        # Check initial state
        assert not hedger.is_delta_neutral()

        # Get recommendations
        recommendations = hedger.get_all_hedge_recommendations()
        assert len(recommendations) > 0

        # Should recommend selling to reduce delta
        delta_recs = [r for r in recommendations if r.hedge_type == HedgeType.DELTA]
        assert len(delta_recs) > 0
        assert delta_recs[0].action == "sell"  # Reduce long delta

    def test_multi_underlying_portfolio(self):
        """Test portfolio with multiple underlyings."""
        hedger = PortfolioHedger()

        # SPY position
        hedger.add_position(
            Position(
                symbol="SPY",
                asset_type="stock",
                quantity=100,
                entry_price=450.0,
                current_price=455.0,
                underlying_symbol="SPY",
            )
        )

        # QQQ position
        hedger.add_position(
            Position(
                symbol="QQQ",
                asset_type="stock",
                quantity=50,
                entry_price=380.0,
                current_price=385.0,
                underlying_symbol="QQQ",
            )
        )

        # Get breakdown
        by_underlying = hedger.get_greeks_by_underlying()

        assert "SPY" in by_underlying
        assert "QQQ" in by_underlying
        assert by_underlying["SPY"].delta == 100
        assert by_underlying["QQQ"].delta == 50

        # Hedge specific underlying
        hedger.targets = HedgeTargets(delta_tolerance=25.0)
        recommendations = hedger.get_all_hedge_recommendations(underlying="SPY")

        # Should only have SPY recommendations
        for rec in recommendations:
            assert "SPY" in rec.symbol or rec.symbol == "SPY"
