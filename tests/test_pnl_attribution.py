"""
Tests for P&L Attribution Module (Sprint 5 Expansion)

Tests Greek-based P&L decomposition (delta, gamma, theta, vega).
Part of UPGRADE-010 Sprint 5 - Quality & Test Coverage.
"""

from datetime import datetime, timedelta, timezone

import pytest

from models.pnl_attribution import (
    GreeksSnapshot,
    PnLAttributor,
    PnLBreakdown,
    PortfolioPnLAttributor,
    RealizedVolatilityCalculator,
    create_attributor_from_trades,
)


class TestGreeksSnapshot:
    """Tests for GreeksSnapshot dataclass."""

    def test_creation(self):
        """Test snapshot creation."""
        now = datetime.now(timezone.utc)
        snapshot = GreeksSnapshot(
            timestamp=now,
            underlying_price=450.0,
            option_price=5.50,
            implied_volatility=0.25,
            delta=0.45,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
        )

        assert snapshot.underlying_price == 450.0
        assert snapshot.delta == 0.45
        assert snapshot.rho == 0.0  # Default value

    def test_with_rho(self):
        """Test snapshot with rho value."""
        now = datetime.now(timezone.utc)
        snapshot = GreeksSnapshot(
            timestamp=now,
            underlying_price=450.0,
            option_price=5.50,
            implied_volatility=0.25,
            delta=0.45,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
            rho=0.08,
        )

        assert snapshot.rho == 0.08


class TestPnLBreakdown:
    """Tests for PnLBreakdown dataclass."""

    def test_default_values(self):
        """Test default breakdown values."""
        breakdown = PnLBreakdown()

        assert breakdown.delta_pnl == 0.0
        assert breakdown.gamma_pnl == 0.0
        assert breakdown.theta_pnl == 0.0
        assert breakdown.vega_pnl == 0.0
        assert breakdown.rho_pnl == 0.0
        assert breakdown.unexplained == 0.0
        assert breakdown.total_pnl == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        breakdown = PnLBreakdown(
            delta_pnl=100.0,
            gamma_pnl=20.0,
            theta_pnl=-15.0,
            vega_pnl=30.0,
            rho_pnl=5.0,
            unexplained=10.0,
            total_pnl=150.0,
        )

        d = breakdown.to_dict()

        assert d["delta_pnl"] == 100.0
        assert d["gamma_pnl"] == 20.0
        assert d["theta_pnl"] == -15.0
        assert d["total_pnl"] == 150.0

    def test_as_percentages(self):
        """Test percentage breakdown calculation."""
        breakdown = PnLBreakdown(
            delta_pnl=80.0,
            gamma_pnl=20.0,
            theta_pnl=-10.0,
            vega_pnl=10.0,
            rho_pnl=0.0,
            unexplained=0.0,
            total_pnl=100.0,
        )

        pcts = breakdown.as_percentages()

        assert pcts["delta_pct"] == 80.0
        assert pcts["gamma_pct"] == 20.0
        assert pcts["theta_pct"] == -10.0
        assert pcts["vega_pct"] == 10.0

    def test_as_percentages_zero_total(self):
        """Test percentage breakdown with zero total."""
        breakdown = PnLBreakdown(total_pnl=0.0)

        pcts = breakdown.as_percentages()

        # When total is 0, all percentages are 0
        # Keys match to_dict() keys
        assert pcts["delta_pnl"] == 0.0
        assert pcts["gamma_pnl"] == 0.0

    def test_as_percentages_negative_total(self):
        """Test percentage breakdown with negative total."""
        breakdown = PnLBreakdown(
            delta_pnl=-60.0,
            gamma_pnl=10.0,
            theta_pnl=-20.0,
            vega_pnl=-5.0,
            total_pnl=-75.0,
        )

        pcts = breakdown.as_percentages()

        # Percentages based on absolute total
        assert pcts["delta_pct"] == pytest.approx(-80.0, rel=0.01)


class TestPnLAttributor:
    """Tests for PnLAttributor class."""

    @pytest.fixture
    def attributor(self):
        """Create default attributor."""
        return PnLAttributor()

    @pytest.fixture
    def start_snapshot(self):
        """Create starting snapshot."""
        return GreeksSnapshot(
            timestamp=datetime.now(timezone.utc),
            underlying_price=450.0,
            option_price=5.00,
            implied_volatility=0.25,
            delta=0.50,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
        )

    @pytest.fixture
    def end_snapshot(self, start_snapshot):
        """Create ending snapshot with price increase."""
        return GreeksSnapshot(
            timestamp=start_snapshot.timestamp + timedelta(days=1),
            underlying_price=455.0,  # +5
            option_price=7.50,  # +2.50
            implied_volatility=0.26,  # +0.01
            delta=0.55,
            gamma=0.018,
            theta=-0.06,
            vega=0.14,
        )

    def test_initialization(self, attributor):
        """Test attributor initialization."""
        assert attributor.contract_multiplier == 100
        assert len(attributor.history) == 0

    def test_custom_multiplier(self):
        """Test with custom contract multiplier."""
        attributor = PnLAttributor(contract_multiplier=10)
        assert attributor.contract_multiplier == 10

    def test_add_snapshot(self, attributor, start_snapshot):
        """Test adding snapshots to history."""
        attributor.add_snapshot(start_snapshot)

        assert len(attributor.history) == 1
        assert attributor.history[0] == start_snapshot

    def test_calculate_attribution_long_call(self, attributor, start_snapshot, end_snapshot):
        """Test attribution for long call with underlying increase."""
        breakdown = attributor.calculate_attribution(
            start=start_snapshot,
            end=end_snapshot,
            quantity=1,
        )

        # Delta P&L: 0.50 * 5 * 100 = 250
        assert breakdown.delta_pnl == pytest.approx(250.0, rel=0.01)

        # Gamma P&L: 0.5 * 0.02 * 25 * 100 = 25
        assert breakdown.gamma_pnl == pytest.approx(25.0, rel=0.01)

        # Total P&L: (7.50 - 5.00) * 100 = 250
        assert breakdown.total_pnl == pytest.approx(250.0, rel=0.01)

        # Vega P&L: 0.15 * 1 * 100 = 15 (1% IV increase)
        assert breakdown.vega_pnl == pytest.approx(15.0, rel=0.01)

    def test_calculate_attribution_short_put(self, attributor, start_snapshot, end_snapshot):
        """Test attribution for short position (negative quantity)."""
        breakdown = attributor.calculate_attribution(
            start=start_snapshot,
            end=end_snapshot,
            quantity=-1,
        )

        # Short position: all P&L is inverted
        assert breakdown.delta_pnl == pytest.approx(-250.0, rel=0.01)
        assert breakdown.total_pnl == pytest.approx(-250.0, rel=0.01)

    def test_calculate_attribution_multiple_contracts(self, attributor, start_snapshot, end_snapshot):
        """Test attribution for multiple contracts."""
        breakdown = attributor.calculate_attribution(
            start=start_snapshot,
            end=end_snapshot,
            quantity=5,
        )

        # 5x the single contract P&L
        assert breakdown.delta_pnl == pytest.approx(1250.0, rel=0.01)
        assert breakdown.total_pnl == pytest.approx(1250.0, rel=0.01)

    def test_calculate_attribution_price_decrease(self, attributor, start_snapshot):
        """Test attribution when underlying decreases."""
        end_snapshot = GreeksSnapshot(
            timestamp=start_snapshot.timestamp + timedelta(days=1),
            underlying_price=445.0,  # -5
            option_price=3.00,  # -2.00
            implied_volatility=0.25,
            delta=0.45,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
        )

        breakdown = attributor.calculate_attribution(
            start=start_snapshot,
            end=end_snapshot,
            quantity=1,
        )

        # Delta P&L: 0.50 * (-5) * 100 = -250
        assert breakdown.delta_pnl == pytest.approx(-250.0, rel=0.01)

        # Total P&L: (3.00 - 5.00) * 100 = -200
        assert breakdown.total_pnl == pytest.approx(-200.0, rel=0.01)

    def test_calculate_period_attribution(self, attributor, start_snapshot, end_snapshot):
        """Test period-based attribution using history."""
        attributor.add_snapshot(start_snapshot)
        attributor.add_snapshot(end_snapshot)

        # Use times AFTER the snapshots so they qualify (method finds last snapshot <= time)
        breakdown = attributor.calculate_period_attribution(
            start_time=start_snapshot.timestamp + timedelta(hours=1),  # After first snapshot
            end_time=end_snapshot.timestamp + timedelta(hours=1),  # After last snapshot
            quantity=1,
        )

        assert breakdown is not None
        assert breakdown.total_pnl == pytest.approx(250.0, rel=0.01)

    def test_calculate_period_attribution_no_data(self, attributor):
        """Test period attribution with no matching data."""
        now = datetime.now(timezone.utc)

        breakdown = attributor.calculate_period_attribution(
            start_time=now - timedelta(hours=2),
            end_time=now,
            quantity=1,
        )

        assert breakdown is None

    def test_get_cumulative_attribution_empty(self, attributor):
        """Test cumulative attribution with no history."""
        breakdown = attributor.get_cumulative_attribution()

        assert breakdown.total_pnl == 0.0
        assert breakdown.delta_pnl == 0.0

    def test_get_cumulative_attribution_single(self, attributor, start_snapshot):
        """Test cumulative attribution with single snapshot."""
        attributor.add_snapshot(start_snapshot)

        breakdown = attributor.get_cumulative_attribution()

        # Need at least 2 snapshots
        assert breakdown.total_pnl == 0.0

    def test_get_cumulative_attribution_multiple(self, attributor, start_snapshot, end_snapshot):
        """Test cumulative attribution with multiple snapshots."""
        attributor.add_snapshot(start_snapshot)
        attributor.add_snapshot(end_snapshot)

        breakdown = attributor.get_cumulative_attribution(quantity=1)

        assert breakdown.total_pnl == pytest.approx(250.0, rel=0.01)


class TestPortfolioPnLAttributor:
    """Tests for PortfolioPnLAttributor class."""

    @pytest.fixture
    def portfolio(self):
        """Create portfolio attributor."""
        return PortfolioPnLAttributor()

    @pytest.fixture
    def spy_snapshots(self):
        """Create SPY option snapshots."""
        now = datetime.now(timezone.utc)
        return [
            GreeksSnapshot(
                timestamp=now,
                underlying_price=450.0,
                option_price=5.00,
                implied_volatility=0.25,
                delta=0.50,
                gamma=0.02,
                theta=-0.05,
                vega=0.15,
            ),
            GreeksSnapshot(
                timestamp=now + timedelta(days=1),
                underlying_price=455.0,
                option_price=7.50,
                implied_volatility=0.26,
                delta=0.55,
                gamma=0.018,
                theta=-0.06,
                vega=0.14,
            ),
        ]

    @pytest.fixture
    def qqq_snapshots(self):
        """Create QQQ option snapshots."""
        now = datetime.now(timezone.utc)
        return [
            GreeksSnapshot(
                timestamp=now,
                underlying_price=380.0,
                option_price=4.00,
                implied_volatility=0.28,
                delta=0.45,
                gamma=0.025,
                theta=-0.04,
                vega=0.12,
            ),
            GreeksSnapshot(
                timestamp=now + timedelta(days=1),
                underlying_price=385.0,
                option_price=6.00,
                implied_volatility=0.29,
                delta=0.50,
                gamma=0.023,
                theta=-0.045,
                vega=0.11,
            ),
        ]

    def test_initialization(self, portfolio):
        """Test portfolio initialization."""
        assert len(portfolio.positions) == 0
        assert len(portfolio.quantities) == 0

    def test_add_position(self, portfolio):
        """Test adding position to portfolio."""
        attributor = portfolio.add_position("SPY_C450", quantity=2)

        assert "SPY_C450" in portfolio.positions
        assert portfolio.quantities["SPY_C450"] == 2
        assert isinstance(attributor, PnLAttributor)

    def test_update_position(self, portfolio, spy_snapshots):
        """Test updating position with snapshots."""
        portfolio.add_position("SPY_C450", quantity=1)

        for snapshot in spy_snapshots:
            portfolio.update_position("SPY_C450", snapshot)

        assert len(portfolio.positions["SPY_C450"].history) == 2

    def test_update_position_unknown(self, portfolio, spy_snapshots):
        """Test updating unknown position (should not crash)."""
        portfolio.update_position("UNKNOWN", spy_snapshots[0])

        assert "UNKNOWN" not in portfolio.positions

    def test_get_position_attribution(self, portfolio, spy_snapshots):
        """Test getting attribution for single position."""
        portfolio.add_position("SPY_C450", quantity=1)
        for snapshot in spy_snapshots:
            portfolio.update_position("SPY_C450", snapshot)

        breakdown = portfolio.get_position_attribution("SPY_C450")

        assert breakdown is not None
        assert breakdown.total_pnl == pytest.approx(250.0, rel=0.01)

    def test_get_position_attribution_unknown(self, portfolio):
        """Test getting attribution for unknown position."""
        breakdown = portfolio.get_position_attribution("UNKNOWN")

        assert breakdown is None

    def test_get_portfolio_attribution(self, portfolio, spy_snapshots, qqq_snapshots):
        """Test aggregated portfolio attribution."""
        # Add SPY position
        portfolio.add_position("SPY_C450", quantity=1)
        for snapshot in spy_snapshots:
            portfolio.update_position("SPY_C450", snapshot)

        # Add QQQ position
        portfolio.add_position("QQQ_C380", quantity=2)
        for snapshot in qqq_snapshots:
            portfolio.update_position("QQQ_C380", snapshot)

        breakdown = portfolio.get_portfolio_attribution()

        # SPY: 250 + QQQ: (6-4)*100*2 = 400 = 650 total
        assert breakdown.total_pnl == pytest.approx(650.0, rel=0.01)

    def test_get_report(self, portfolio, spy_snapshots):
        """Test report generation."""
        portfolio.add_position("SPY_C450", quantity=1)
        for snapshot in spy_snapshots:
            portfolio.update_position("SPY_C450", snapshot)

        report = portfolio.get_report()

        assert "timestamp" in report
        assert "portfolio" in report
        assert "positions" in report
        assert "SPY_C450" in report["positions"]
        assert "breakdown" in report["portfolio"]
        assert "percentages" in report["portfolio"]


class TestRealizedVolatilityCalculator:
    """Tests for RealizedVolatilityCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create calculator with default lookback."""
        return RealizedVolatilityCalculator(lookback_periods=20)

    def test_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator.lookback == 20
        assert len(calculator.price_history) == 0
        assert len(calculator.return_history) == 0

    def test_update_single(self, calculator):
        """Test single price update."""
        calculator.update(100.0)

        assert len(calculator.price_history) == 1
        assert len(calculator.return_history) == 0  # Need 2 prices for return

    def test_update_multiple(self, calculator):
        """Test multiple price updates."""
        calculator.update(100.0)
        calculator.update(101.0)
        calculator.update(102.0)

        assert len(calculator.price_history) == 3
        assert len(calculator.return_history) == 2

    def test_update_trims_history(self):
        """Test that history is trimmed to lookback period."""
        calculator = RealizedVolatilityCalculator(lookback_periods=5)

        for i in range(10):
            calculator.update(100.0 + i)

        assert len(calculator.price_history) == 6  # lookback + 1
        assert len(calculator.return_history) == 5  # lookback

    def test_calculate_realized_volatility_insufficient(self, calculator):
        """Test volatility calculation with insufficient data."""
        calculator.update(100.0)

        vol = calculator.calculate_realized_volatility()

        assert vol == 0.0

    def test_calculate_realized_volatility(self, calculator):
        """Test volatility calculation with sufficient data."""
        # Add stable prices with some variation
        prices = [100.0, 101.0, 99.5, 100.5, 101.5, 100.0, 99.0, 100.0, 101.0, 100.5]
        for price in prices:
            calculator.update(price)

        vol = calculator.calculate_realized_volatility()

        # Should be positive and reasonable (10-50% annualized)
        assert vol > 0
        assert vol < 1.0  # Less than 100% annualized

    def test_calculate_realized_volatility_constant_prices(self, calculator):
        """Test volatility calculation with constant prices."""
        for _ in range(10):
            calculator.update(100.0)

        vol = calculator.calculate_realized_volatility()

        assert vol == 0.0

    def test_get_vol_ratio(self, calculator):
        """Test volatility ratio calculation."""
        # Add prices with known volatility
        prices = [100, 102, 98, 101, 99, 103, 97, 100, 102, 99]
        for price in prices:
            calculator.update(price)

        ratio = calculator.get_vol_ratio(implied_volatility=0.25)

        # Ratio should be positive
        assert ratio > 0

    def test_get_vol_ratio_zero_iv(self, calculator):
        """Test volatility ratio with zero implied vol."""
        calculator.update(100.0)
        calculator.update(101.0)

        ratio = calculator.get_vol_ratio(implied_volatility=0.0)

        assert ratio == 0.0


class TestCreateAttributorFromTrades:
    """Tests for factory function."""

    def test_create_from_empty(self):
        """Test creation from empty trade list."""
        attributor = create_attributor_from_trades([])

        assert len(attributor.positions) == 0

    def test_create_from_single_trade(self):
        """Test creation from single trade."""
        trades = [
            {
                "symbol": "SPY_C450",
                "quantity": 2,
                "timestamp": datetime.now(timezone.utc),
                "underlying_price": 450.0,
                "option_price": 5.00,
                "iv": 0.25,
                "delta": 0.50,
                "gamma": 0.02,
                "theta": -0.05,
                "vega": 0.15,
            }
        ]

        attributor = create_attributor_from_trades(trades)

        assert "SPY_C450" in attributor.positions
        assert attributor.quantities["SPY_C450"] == 2
        assert len(attributor.positions["SPY_C450"].history) == 1

    def test_create_from_multiple_trades_same_symbol(self):
        """Test creation from multiple trades of same symbol."""
        now = datetime.now(timezone.utc)
        trades = [
            {
                "symbol": "SPY_C450",
                "quantity": 2,
                "timestamp": now,
                "underlying_price": 450.0,
                "option_price": 5.00,
                "iv": 0.25,
                "delta": 0.50,
                "gamma": 0.02,
                "theta": -0.05,
                "vega": 0.15,
            },
            {
                "symbol": "SPY_C450",
                "quantity": 2,
                "timestamp": now + timedelta(days=1),
                "underlying_price": 455.0,
                "option_price": 7.50,
                "iv": 0.26,
                "delta": 0.55,
                "gamma": 0.018,
                "theta": -0.06,
                "vega": 0.14,
            },
        ]

        attributor = create_attributor_from_trades(trades)

        assert len(attributor.positions["SPY_C450"].history) == 2

    def test_create_from_multiple_symbols(self):
        """Test creation from trades with different symbols."""
        now = datetime.now(timezone.utc)
        trades = [
            {
                "symbol": "SPY_C450",
                "quantity": 1,
                "timestamp": now,
                "underlying_price": 450.0,
                "option_price": 5.00,
                "iv": 0.25,
                "delta": 0.50,
                "gamma": 0.02,
                "theta": -0.05,
                "vega": 0.15,
            },
            {
                "symbol": "QQQ_C380",
                "quantity": 2,
                "timestamp": now,
                "underlying_price": 380.0,
                "option_price": 4.00,
                "iv": 0.28,
                "delta": 0.45,
                "gamma": 0.025,
                "theta": -0.04,
                "vega": 0.12,
            },
        ]

        attributor = create_attributor_from_trades(trades)

        assert "SPY_C450" in attributor.positions
        assert "QQQ_C380" in attributor.positions
        assert attributor.quantities["SPY_C450"] == 1
        assert attributor.quantities["QQQ_C380"] == 2


class TestIntegrationScenarios:
    """Integration tests for realistic trading scenarios."""

    def test_long_call_profitable(self):
        """Test attribution for profitable long call."""
        now = datetime.now(timezone.utc)
        attributor = PnLAttributor()

        # Entry: Buy call when underlying at 450
        entry = GreeksSnapshot(
            timestamp=now,
            underlying_price=450.0,
            option_price=5.00,
            implied_volatility=0.20,
            delta=0.50,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
        )

        # Exit: Sell after rally
        exit_snapshot = GreeksSnapshot(
            timestamp=now + timedelta(days=5),
            underlying_price=460.0,  # +10 rally
            option_price=10.00,  # +5 profit
            implied_volatility=0.22,  # IV increased slightly
            delta=0.65,
            gamma=0.015,
            theta=-0.07,
            vega=0.12,
        )

        breakdown = attributor.calculate_attribution(entry, exit_snapshot, quantity=1)

        # Should be profitable
        assert breakdown.total_pnl > 0
        # Delta should be largest contributor
        assert breakdown.delta_pnl > breakdown.gamma_pnl

    def test_short_put_theta_decay(self):
        """Test attribution for short put with theta decay."""
        now = datetime.now(timezone.utc)
        attributor = PnLAttributor()

        # Entry: Sell put
        entry = GreeksSnapshot(
            timestamp=now,
            underlying_price=450.0,
            option_price=3.00,
            implied_volatility=0.25,
            delta=-0.30,  # Put delta
            gamma=0.02,
            theta=-0.08,  # Significant theta
            vega=0.15,
        )

        # Exit: Price stable, time decay
        exit_snapshot = GreeksSnapshot(
            timestamp=now + timedelta(days=10),
            underlying_price=451.0,  # Nearly unchanged
            option_price=1.50,  # Decayed
            implied_volatility=0.24,
            delta=-0.25,
            gamma=0.015,
            theta=-0.05,
            vega=0.10,
        )

        # Short position (negative quantity)
        breakdown = attributor.calculate_attribution(entry, exit_snapshot, quantity=-1)

        # Should be profitable for short seller
        assert breakdown.total_pnl > 0  # (3.00 - 1.50) * 100 * (-1) = -150 → +150 for short

    def test_gamma_scalping_scenario(self):
        """Test gamma P&L attribution during volatile movement."""
        now = datetime.now(timezone.utc)
        attributor = PnLAttributor()

        # High gamma position
        entry = GreeksSnapshot(
            timestamp=now,
            underlying_price=450.0,
            option_price=5.00,
            implied_volatility=0.30,
            delta=0.50,
            gamma=0.05,  # High gamma
            theta=-0.10,
            vega=0.20,
        )

        # Large move
        exit_snapshot = GreeksSnapshot(
            timestamp=now + timedelta(hours=4),
            underlying_price=465.0,  # +15 big move
            option_price=14.00,
            implied_volatility=0.35,  # IV spike
            delta=0.70,
            gamma=0.03,
            theta=-0.08,
            vega=0.15,
        )

        breakdown = attributor.calculate_attribution(entry, exit_snapshot, quantity=1)

        # Gamma P&L should be significant with large move
        # Gamma P&L = 0.5 * 0.05 * 15^2 * 100 = 562.5
        assert breakdown.gamma_pnl > 400
