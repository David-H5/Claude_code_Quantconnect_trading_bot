"""
Tests for multi_leg_strategy module.

Tests multi-leg options strategies including:
- Vertical spreads
- Iron condors
- Strangles and straddles
- Covered calls
- Greeks calculations
- Risk/reward analysis
"""

from datetime import datetime, timedelta

import pytest

from models.multi_leg_strategy import (
    MultiLegStrategy,
    OptionLeg,
    PortfolioGreeks,
    StrategyBuilder,
    StrategyType,
    find_delta_strikes,
)


class TestStrategyType:
    """Tests for StrategyType enum."""

    def test_single_leg_strategies(self):
        """Test single leg strategy types exist."""
        assert StrategyType.LONG_CALL.value == "long_call"
        assert StrategyType.LONG_PUT.value == "long_put"
        assert StrategyType.SHORT_CALL.value == "short_call"
        assert StrategyType.SHORT_PUT.value == "short_put"

    def test_vertical_spread_strategies(self):
        """Test vertical spread types exist."""
        assert StrategyType.BULL_CALL_SPREAD.value == "bull_call_spread"
        assert StrategyType.BEAR_CALL_SPREAD.value == "bear_call_spread"
        assert StrategyType.BULL_PUT_SPREAD.value == "bull_put_spread"
        assert StrategyType.BEAR_PUT_SPREAD.value == "bear_put_spread"

    def test_neutral_strategies(self):
        """Test neutral strategy types exist."""
        assert StrategyType.LONG_STRADDLE.value == "long_straddle"
        assert StrategyType.SHORT_STRADDLE.value == "short_straddle"
        assert StrategyType.LONG_STRANGLE.value == "long_strangle"
        assert StrategyType.SHORT_STRANGLE.value == "short_strangle"

    def test_multi_leg_strategies(self):
        """Test multi-leg strategy types exist."""
        assert StrategyType.IRON_CONDOR.value == "iron_condor"
        assert StrategyType.IRON_BUTTERFLY.value == "iron_butterfly"
        assert StrategyType.LONG_BUTTERFLY.value == "long_butterfly"
        assert StrategyType.SHORT_BUTTERFLY.value == "short_butterfly"

    def test_income_strategies(self):
        """Test income strategy types exist."""
        assert StrategyType.COVERED_CALL.value == "covered_call"
        assert StrategyType.CASH_SECURED_PUT.value == "cash_secured_put"
        assert StrategyType.THE_WHEEL.value == "the_wheel"

    def test_calendar_strategies(self):
        """Test calendar strategy types exist."""
        assert StrategyType.CALENDAR_SPREAD.value == "calendar_spread"
        assert StrategyType.DIAGONAL_SPREAD.value == "diagonal_spread"

    def test_enum_count(self):
        """Test total number of strategy types."""
        assert len(StrategyType) == 21


class TestOptionLeg:
    """Tests for OptionLeg dataclass."""

    @pytest.fixture
    def expiry(self):
        """Create expiry date 30 days out."""
        return datetime.now() + timedelta(days=30)

    @pytest.fixture
    def call_leg(self, expiry):
        """Create sample long call leg."""
        return OptionLeg(
            symbol="SPY_250115_C450",
            underlying="SPY",
            option_type="call",
            strike=450.0,
            expiry=expiry,
            quantity=1,
            entry_price=5.00,
            current_price=6.00,
            delta=0.50,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
            rho=0.03,
            implied_volatility=0.25,
        )

    @pytest.fixture
    def put_leg(self, expiry):
        """Create sample short put leg."""
        return OptionLeg(
            symbol="SPY_250115_P440",
            underlying="SPY",
            option_type="put",
            strike=440.0,
            expiry=expiry,
            quantity=-1,
            entry_price=3.00,
            current_price=2.50,
            delta=-0.30,
            gamma=0.01,
            theta=-0.04,
            vega=0.12,
        )

    def test_call_payoff_itm(self, call_leg):
        """Test call payoff when ITM at expiry."""
        # Strike 450, underlying at 460 = 10 intrinsic
        payoff = call_leg.payoff_at_expiry(460.0)
        assert payoff == 1000.0  # 10 * 1 * 100

    def test_call_payoff_otm(self, call_leg):
        """Test call payoff when OTM at expiry."""
        # Strike 450, underlying at 440 = 0 intrinsic
        payoff = call_leg.payoff_at_expiry(440.0)
        assert payoff == 0.0

    def test_call_payoff_atm(self, call_leg):
        """Test call payoff when ATM at expiry."""
        payoff = call_leg.payoff_at_expiry(450.0)
        assert payoff == 0.0

    def test_put_payoff_itm(self, put_leg):
        """Test put payoff when ITM at expiry."""
        # Strike 440, underlying at 430 = 10 intrinsic
        # Short put (quantity -1) means -10 * -1 * 100 = -1000 (loss)
        payoff = put_leg.payoff_at_expiry(430.0)
        assert payoff == -1000.0

    def test_put_payoff_otm(self, put_leg):
        """Test put payoff when OTM at expiry."""
        # Strike 440, underlying at 450 = 0 intrinsic
        payoff = put_leg.payoff_at_expiry(450.0)
        assert payoff == 0.0

    def test_current_pnl_profit(self, call_leg):
        """Test current P&L calculation for profit."""
        # Entry 5.00, current 6.00, quantity 1
        pnl = call_leg.current_pnl()
        assert pnl == 100.0  # (6 - 5) * 1 * 100

    def test_current_pnl_short_profit(self, put_leg):
        """Test current P&L for short position profit."""
        # Entry 3.00, current 2.50, quantity -1 (short)
        pnl = put_leg.current_pnl()
        assert pnl == 50.0  # (2.50 - 3.00) * -1 * 100 = 50

    def test_is_long(self, call_leg):
        """Test is_long property."""
        assert call_leg.is_long is True

    def test_is_short(self, put_leg):
        """Test is_short property."""
        assert put_leg.is_short is True
        assert put_leg.is_long is False

    def test_days_to_expiry(self, call_leg, expiry):
        """Test days_to_expiry property."""
        # Should be approximately 30 days
        assert 28 <= call_leg.days_to_expiry <= 31

    def test_to_dict(self, call_leg):
        """Test to_dict conversion."""
        d = call_leg.to_dict()

        assert d["symbol"] == "SPY_250115_C450"
        assert d["underlying"] == "SPY"
        assert d["type"] == "call"
        assert d["strike"] == 450.0
        assert d["quantity"] == 1
        assert d["entry_price"] == 5.00
        assert d["current_price"] == 6.00
        assert d["delta"] == 0.50
        assert d["gamma"] == 0.02
        assert d["theta"] == -0.05
        assert d["vega"] == 0.15
        assert d["pnl"] == 100.0


class TestPortfolioGreeks:
    """Tests for PortfolioGreeks dataclass."""

    def test_initialization(self):
        """Test default initialization."""
        greeks = PortfolioGreeks()

        assert greeks.delta == 0.0
        assert greeks.gamma == 0.0
        assert greeks.theta == 0.0
        assert greeks.vega == 0.0
        assert greeks.rho == 0.0

    def test_custom_values(self):
        """Test with custom values."""
        greeks = PortfolioGreeks(
            delta=0.25,
            gamma=0.05,
            theta=-0.10,
            vega=0.30,
            rho=0.02,
        )

        assert greeks.delta == 0.25
        assert greeks.theta == -0.10

    def test_to_dict(self):
        """Test to_dict conversion."""
        greeks = PortfolioGreeks(delta=0.5, gamma=0.1, theta=-0.05, vega=0.2, rho=0.01)
        d = greeks.to_dict()

        assert d["delta"] == 0.5
        assert d["gamma"] == 0.1
        assert d["theta"] == -0.05
        assert d["vega"] == 0.2
        assert d["rho"] == 0.01


class TestMultiLegStrategy:
    """Tests for MultiLegStrategy dataclass."""

    @pytest.fixture
    def expiry(self):
        """Create expiry date."""
        return datetime.now() + timedelta(days=30)

    @pytest.fixture
    def bull_call_spread(self, expiry):
        """Create a bull call spread."""
        strategy = MultiLegStrategy(
            name="Bull Call Spread SPY",
            strategy_type=StrategyType.BULL_CALL_SPREAD,
            underlying_symbol="SPY",
            underlying_price=450.0,
        )

        # Long 445 call @ 8.00
        strategy.add_leg(
            OptionLeg(
                symbol="SPY_C445",
                underlying="SPY",
                option_type="call",
                strike=445.0,
                expiry=expiry,
                quantity=1,
                entry_price=8.00,
                current_price=9.00,
                delta=0.60,
                gamma=0.02,
                theta=-0.06,
                vega=0.18,
            )
        )

        # Short 455 call @ 3.00
        strategy.add_leg(
            OptionLeg(
                symbol="SPY_C455",
                underlying="SPY",
                option_type="call",
                strike=455.0,
                expiry=expiry,
                quantity=-1,
                entry_price=3.00,
                current_price=3.50,
                delta=0.40,
                gamma=0.02,
                theta=-0.04,
                vega=0.15,
            )
        )

        return strategy

    def test_add_leg_sets_underlying(self, expiry):
        """Test add_leg sets underlying symbol."""
        strategy = MultiLegStrategy(
            name="Test",
            strategy_type=StrategyType.LONG_CALL,
        )

        leg = OptionLeg(
            symbol="AAPL_C180",
            underlying="AAPL",
            option_type="call",
            strike=180.0,
            expiry=expiry,
            quantity=1,
            entry_price=5.00,
        )

        strategy.add_leg(leg)

        assert strategy.underlying_symbol == "AAPL"
        assert len(strategy.legs) == 1

    def test_get_portfolio_greeks(self, bull_call_spread):
        """Test aggregate Greeks calculation."""
        greeks = bull_call_spread.get_portfolio_greeks()

        # Long 1 @ delta 0.60, Short 1 @ delta 0.40
        # Net delta = 0.60 * 1 + 0.40 * (-1) = 0.20
        assert greeks.delta == pytest.approx(0.20, rel=1e-6)

        # Net gamma = 0.02 * 1 + 0.02 * (-1) = 0
        assert greeks.gamma == pytest.approx(0.0, rel=1e-6)

        # Net theta = -0.06 * 1 + (-0.04) * (-1) = -0.02
        assert greeks.theta == pytest.approx(-0.02, rel=1e-6)

        # Net vega = 0.18 * 1 + 0.15 * (-1) = 0.03
        assert greeks.vega == pytest.approx(0.03, rel=1e-6)

    def test_total_pnl(self, bull_call_spread):
        """Test total P&L calculation."""
        pnl = bull_call_spread.total_pnl()

        # Long call: (9.00 - 8.00) * 1 * 100 = 100
        # Short call: (3.50 - 3.00) * (-1) * 100 = -50
        assert pnl == pytest.approx(50.0, rel=1e-6)

    def test_net_premium(self, bull_call_spread):
        """Test net premium calculation."""
        premium = bull_call_spread.net_premium()

        # Long: 8.00 * 1 * 100 = 800 (paid)
        # Short: 3.00 * (-1) * 100 = -300 (received)
        assert premium == pytest.approx(500.0, rel=1e-6)

    def test_payoff_at_price_max_profit(self, bull_call_spread):
        """Test payoff at max profit price."""
        # At 460 (above both strikes)
        # Long 445 call: (460-445) * 1 * 100 = 1500
        # Short 455 call: (460-455) * (-1) * 100 = -500
        # Net premium: -500 (cost)
        # Total payoff: 1500 - 500 - 500 = 500
        payoff = bull_call_spread.payoff_at_price(460.0)
        assert payoff == pytest.approx(500.0, rel=1e-6)

    def test_payoff_at_price_max_loss(self, bull_call_spread):
        """Test payoff at max loss price."""
        # At 440 (below both strikes)
        # Both calls expire worthless
        # Net premium: -500 (cost)
        payoff = bull_call_spread.payoff_at_price(440.0)
        assert payoff == pytest.approx(-500.0, rel=1e-6)

    def test_calculate_breakevens(self, bull_call_spread):
        """Test breakeven calculation."""
        breakevens = bull_call_spread.calculate_breakevens()

        # For bull call spread: breakeven = long strike + net debit
        # Net debit = 8.00 - 3.00 = 5.00
        # Breakeven = 445 + 5 = 450
        assert len(breakevens) >= 1
        assert any(449 <= be <= 451 for be in breakevens)

    def test_calculate_risk_reward(self, bull_call_spread):
        """Test risk/reward calculation."""
        risk = bull_call_spread.calculate_risk_reward()

        # Max profit: spread width - net debit = (455-445)*100 - 500 = 500
        # Max loss: net debit = 500
        assert risk["max_profit"] == pytest.approx(500.0, rel=0.1)
        assert risk["max_loss"] == pytest.approx(-500.0, rel=0.1)
        assert risk["risk_reward_ratio"] == pytest.approx(1.0, rel=0.1)

    def test_get_margin_requirement_spread(self, bull_call_spread):
        """Test margin for defined-risk spread."""
        bull_call_spread.calculate_risk_reward()
        margin = bull_call_spread.get_margin_requirement()

        # For spreads, margin = max loss = 500
        assert margin == pytest.approx(500.0, rel=0.1)

    def test_get_margin_requirement_short_put(self, expiry):
        """Test margin for undefined-risk short put."""
        strategy = MultiLegStrategy(
            name="Short Put",
            strategy_type=StrategyType.SHORT_PUT,
            underlying_symbol="SPY",
            underlying_price=450.0,
        )

        strategy.add_leg(
            OptionLeg(
                symbol="SPY_P440",
                underlying="SPY",
                option_type="put",
                strike=440.0,
                expiry=expiry,
                quantity=-1,
                entry_price=3.00,
            )
        )

        margin = strategy.get_margin_requirement()

        # Cash-secured put: strike * 100 = 440 * 100 = 44000
        assert margin == pytest.approx(44000.0, rel=1e-6)

    def test_get_margin_requirement_short_call(self, expiry):
        """Test margin for short call."""
        strategy = MultiLegStrategy(
            name="Short Call",
            strategy_type=StrategyType.SHORT_CALL,
            underlying_symbol="SPY",
            underlying_price=450.0,
        )

        strategy.add_leg(
            OptionLeg(
                symbol="SPY_C460",
                underlying="SPY",
                option_type="call",
                strike=460.0,
                expiry=expiry,
                quantity=-1,
                entry_price=2.00,
            )
        )

        margin = strategy.get_margin_requirement()

        # Short call: 20% of underlying * 100 = 450 * 100 * 0.20 = 9000
        assert margin == pytest.approx(9000.0, rel=1e-6)

    def test_to_dict(self, bull_call_spread):
        """Test to_dict conversion."""
        d = bull_call_spread.to_dict()

        assert d["name"] == "Bull Call Spread SPY"
        assert d["type"] == "bull_call_spread"
        assert d["underlying"] == "SPY"
        assert d["underlying_price"] == 450.0
        assert len(d["legs"]) == 2
        assert "greeks" in d
        assert "total_pnl" in d
        assert "net_premium" in d
        assert "risk_reward" in d
        assert "margin_requirement" in d


class TestStrategyBuilder:
    """Tests for StrategyBuilder factory class."""

    @pytest.fixture
    def expiry(self):
        """Create expiry date."""
        return datetime.now() + timedelta(days=30)

    def test_create_bull_call_spread(self, expiry):
        """Test creating bull call spread."""
        strategy = StrategyBuilder.create_vertical_spread(
            underlying="SPY",
            underlying_price=450.0,
            option_type="call",
            long_strike=445.0,
            short_strike=455.0,
            expiry=expiry,
            long_price=8.00,
            short_price=3.00,
        )

        assert strategy.strategy_type == StrategyType.BULL_CALL_SPREAD
        assert strategy.name == "Bull Call Spread SPY"
        assert len(strategy.legs) == 2
        assert strategy.underlying_symbol == "SPY"

    def test_create_bear_call_spread(self, expiry):
        """Test creating bear call spread."""
        strategy = StrategyBuilder.create_vertical_spread(
            underlying="SPY",
            underlying_price=450.0,
            option_type="call",
            long_strike=455.0,  # Higher strike long
            short_strike=445.0,  # Lower strike short
            expiry=expiry,
            long_price=3.00,
            short_price=8.00,
        )

        assert strategy.strategy_type == StrategyType.BEAR_CALL_SPREAD
        assert strategy.name == "Bear Call Spread SPY"

    def test_create_bull_put_spread(self, expiry):
        """Test creating bull put spread."""
        strategy = StrategyBuilder.create_vertical_spread(
            underlying="SPY",
            underlying_price=450.0,
            option_type="put",
            long_strike=455.0,  # Higher strike long
            short_strike=445.0,  # Lower strike short
            expiry=expiry,
            long_price=8.00,
            short_price=3.00,
        )

        assert strategy.strategy_type == StrategyType.BULL_PUT_SPREAD
        assert strategy.name == "Bull Put Spread SPY"

    def test_create_bear_put_spread(self, expiry):
        """Test creating bear put spread."""
        strategy = StrategyBuilder.create_vertical_spread(
            underlying="SPY",
            underlying_price=450.0,
            option_type="put",
            long_strike=445.0,  # Lower strike long
            short_strike=455.0,  # Higher strike short
            expiry=expiry,
            long_price=3.00,
            short_price=8.00,
        )

        assert strategy.strategy_type == StrategyType.BEAR_PUT_SPREAD
        assert strategy.name == "Bear Put Spread SPY"

    def test_create_vertical_spread_quantity(self, expiry):
        """Test creating spread with custom quantity."""
        strategy = StrategyBuilder.create_vertical_spread(
            underlying="SPY",
            underlying_price=450.0,
            option_type="call",
            long_strike=445.0,
            short_strike=455.0,
            expiry=expiry,
            long_price=8.00,
            short_price=3.00,
            quantity=5,
        )

        assert strategy.legs[0].quantity == 5
        assert strategy.legs[1].quantity == -5

    def test_create_iron_condor(self, expiry):
        """Test creating iron condor."""
        strategy = StrategyBuilder.create_iron_condor(
            underlying="SPY",
            underlying_price=450.0,
            expiry=expiry,
            put_long_strike=430.0,
            put_short_strike=440.0,
            call_short_strike=460.0,
            call_long_strike=470.0,
            put_long_price=1.00,
            put_short_price=2.50,
            call_short_price=2.50,
            call_long_price=1.00,
        )

        assert strategy.strategy_type == StrategyType.IRON_CONDOR
        assert strategy.name == "Iron Condor SPY"
        assert len(strategy.legs) == 4
        assert strategy.underlying_symbol == "SPY"

        # Verify leg types
        put_legs = [leg for leg in strategy.legs if leg.option_type == "put"]
        call_legs = [leg for leg in strategy.legs if leg.option_type == "call"]
        assert len(put_legs) == 2
        assert len(call_legs) == 2

    def test_create_iron_condor_credit(self, expiry):
        """Test iron condor receives net credit."""
        strategy = StrategyBuilder.create_iron_condor(
            underlying="SPY",
            underlying_price=450.0,
            expiry=expiry,
            put_long_strike=430.0,
            put_short_strike=440.0,
            call_short_strike=460.0,
            call_long_strike=470.0,
            put_long_price=1.00,
            put_short_price=2.50,
            call_short_price=2.50,
            call_long_price=1.00,
        )

        premium = strategy.net_premium()
        # Should receive net credit
        # Long: (1.00 + 1.00) * 1 * 100 = 200 paid
        # Short: (2.50 + 2.50) * (-1) * 100 = -500 received
        assert premium < 0  # Net credit

    def test_create_short_strangle(self, expiry):
        """Test creating short strangle."""
        strategy = StrategyBuilder.create_short_strangle(
            underlying="SPY",
            underlying_price=450.0,
            expiry=expiry,
            put_strike=435.0,
            call_strike=465.0,
            put_price=2.00,
            call_price=2.00,
        )

        assert strategy.strategy_type == StrategyType.SHORT_STRANGLE
        assert strategy.name == "Short Strangle SPY"
        assert len(strategy.legs) == 2

        # Both should be short
        assert strategy.legs[0].quantity == -1
        assert strategy.legs[1].quantity == -1

    def test_create_short_strangle_with_delta(self, expiry):
        """Test short strangle with custom target delta."""
        strategy = StrategyBuilder.create_short_strangle(
            underlying="SPY",
            underlying_price=450.0,
            expiry=expiry,
            put_strike=440.0,
            call_strike=460.0,
            put_price=2.50,
            call_price=2.50,
            target_delta=0.20,
        )

        # Check deltas were set
        put_leg = [leg for leg in strategy.legs if leg.option_type == "put"][0]
        call_leg = [leg for leg in strategy.legs if leg.option_type == "call"][0]
        assert put_leg.delta == -0.20
        assert call_leg.delta == 0.20

    def test_create_long_straddle(self, expiry):
        """Test creating long straddle."""
        strategy = StrategyBuilder.create_straddle(
            underlying="SPY",
            underlying_price=450.0,
            expiry=expiry,
            strike=450.0,
            put_price=5.00,
            call_price=5.00,
            is_long=True,
        )

        assert strategy.strategy_type == StrategyType.LONG_STRADDLE
        assert strategy.name == "Long Straddle SPY"
        assert len(strategy.legs) == 2

        # Both should be long
        assert strategy.legs[0].quantity == 1
        assert strategy.legs[1].quantity == 1

    def test_create_short_straddle(self, expiry):
        """Test creating short straddle."""
        strategy = StrategyBuilder.create_straddle(
            underlying="SPY",
            underlying_price=450.0,
            expiry=expiry,
            strike=450.0,
            put_price=5.00,
            call_price=5.00,
            is_long=False,
        )

        assert strategy.strategy_type == StrategyType.SHORT_STRADDLE
        assert strategy.name == "Short Straddle SPY"

        # Both should be short
        assert strategy.legs[0].quantity == -1
        assert strategy.legs[1].quantity == -1

    def test_create_covered_call(self, expiry):
        """Test creating covered call."""
        strategy = StrategyBuilder.create_covered_call(
            underlying="SPY",
            underlying_price=450.0,
            shares=100,
            call_strike=460.0,
            call_expiry=expiry,
            call_price=2.50,
            share_entry_price=448.0,
        )

        assert strategy.strategy_type == StrategyType.COVERED_CALL
        assert strategy.name == "Covered Call SPY"
        assert len(strategy.legs) == 1

        # Should be short call
        assert strategy.legs[0].quantity == -1
        assert strategy.legs[0].option_type == "call"

    def test_create_covered_call_multiple_contracts(self, expiry):
        """Test covered call with multiple contracts."""
        strategy = StrategyBuilder.create_covered_call(
            underlying="SPY",
            underlying_price=450.0,
            shares=500,  # 5 contracts worth
            call_strike=460.0,
            call_expiry=expiry,
            call_price=2.50,
            share_entry_price=448.0,
        )

        assert strategy.legs[0].quantity == -5


class TestFindDeltaStrikes:
    """Tests for find_delta_strikes function."""

    @pytest.fixture
    def call_chain(self):
        """Create sample call option chain."""
        return [
            {"type": "call", "strike": 440, "delta": 0.70, "price": 12.00},
            {"type": "call", "strike": 445, "delta": 0.60, "price": 9.00},
            {"type": "call", "strike": 450, "delta": 0.50, "price": 6.50},
            {"type": "call", "strike": 455, "delta": 0.40, "price": 4.50},
            {"type": "call", "strike": 460, "delta": 0.30, "price": 3.00},
            {"type": "call", "strike": 465, "delta": 0.20, "price": 1.80},
            {"type": "call", "strike": 470, "delta": 0.12, "price": 1.00},
        ]

    @pytest.fixture
    def put_chain(self):
        """Create sample put option chain."""
        return [
            {"type": "put", "strike": 430, "delta": -0.12, "price": 1.00},
            {"type": "put", "strike": 435, "delta": -0.20, "price": 1.80},
            {"type": "put", "strike": 440, "delta": -0.30, "price": 3.00},
            {"type": "put", "strike": 445, "delta": -0.40, "price": 4.50},
            {"type": "put", "strike": 450, "delta": -0.50, "price": 6.50},
        ]

    def test_find_call_16_delta(self, call_chain):
        """Test finding 16-delta call."""
        contract = find_delta_strikes(call_chain, 0.16, "call")

        assert contract is not None
        # 0.12 and 0.20 are equidistant from 0.16; min() picks first (0.20 at 465)
        assert contract["strike"] == 465

    def test_find_call_30_delta(self, call_chain):
        """Test finding 30-delta call."""
        contract = find_delta_strikes(call_chain, 0.30, "call")

        assert contract is not None
        assert contract["strike"] == 460  # Exact 0.30 delta

    def test_find_put_16_delta(self, put_chain):
        """Test finding 16-delta put."""
        contract = find_delta_strikes(put_chain, 0.16, "put")

        assert contract is not None
        # Should find -0.12 or -0.20, closest to -0.16
        assert contract["strike"] in [430, 435]

    def test_find_put_30_delta(self, put_chain):
        """Test finding 30-delta put."""
        contract = find_delta_strikes(put_chain, 0.30, "put")

        assert contract is not None
        assert contract["strike"] == 440  # Exact -0.30 delta

    def test_empty_chain(self):
        """Test with empty chain."""
        result = find_delta_strikes([], 0.16, "call")
        assert result is None

    def test_no_matching_type(self, call_chain):
        """Test with no contracts of requested type."""
        result = find_delta_strikes(call_chain, 0.16, "put")
        assert result is None

    def test_mixed_chain(self, call_chain, put_chain):
        """Test with mixed chain."""
        mixed = call_chain + put_chain

        call = find_delta_strikes(mixed, 0.30, "call")
        put = find_delta_strikes(mixed, 0.30, "put")

        assert call is not None
        assert put is not None
        assert call["type"] == "call"
        assert put["type"] == "put"


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    @pytest.fixture
    def expiry(self):
        """Create expiry date 45 days out."""
        return datetime.now() + timedelta(days=45)

    def test_iron_condor_profit_zone(self, expiry):
        """Test iron condor profit within range."""
        strategy = StrategyBuilder.create_iron_condor(
            underlying="SPY",
            underlying_price=450.0,
            expiry=expiry,
            put_long_strike=425.0,
            put_short_strike=435.0,
            call_short_strike=465.0,
            call_long_strike=475.0,
            put_long_price=0.80,
            put_short_price=2.00,
            call_short_price=2.00,
            call_long_price=0.80,
        )

        # At middle of range (450), all options expire worthless
        payoff = strategy.payoff_at_price(450.0)
        # Net premium received (credit)
        assert payoff > 0

        # At extremes, should lose
        payoff_low = strategy.payoff_at_price(400.0)
        payoff_high = strategy.payoff_at_price(500.0)
        assert payoff_low < 0
        assert payoff_high < 0

    def test_bull_call_spread_risk_profile(self, expiry):
        """Test bull call spread has defined risk."""
        strategy = StrategyBuilder.create_vertical_spread(
            underlying="SPY",
            underlying_price=450.0,
            option_type="call",
            long_strike=445.0,
            short_strike=455.0,
            expiry=expiry,
            long_price=10.00,
            short_price=5.00,
        )

        risk = strategy.calculate_risk_reward()

        # Max loss is net debit paid
        assert risk["max_loss"] == pytest.approx(-500.0, rel=0.1)

        # Max profit is spread width minus debit
        # (455-445)*100 - 500 = 500
        assert risk["max_profit"] == pytest.approx(500.0, rel=0.1)

    def test_straddle_unlimited_profit_potential(self, expiry):
        """Test straddle has unlimited profit potential on upside."""
        strategy = StrategyBuilder.create_straddle(
            underlying="SPY",
            underlying_price=450.0,
            expiry=expiry,
            strike=450.0,
            put_price=8.00,
            call_price=8.00,
            is_long=True,
        )

        # At extreme price, profit should be very high
        payoff_high = strategy.payoff_at_price(550.0)
        # Call intrinsic: (550-450)*100 = 10000
        # Put intrinsic: 0
        # Premium paid: (8+8)*100 = 1600
        # Net: 10000 - 1600 = 8400
        assert payoff_high == pytest.approx(8400.0, rel=1e-6)

    def test_portfolio_greeks_delta_neutral(self, expiry):
        """Test creating delta-neutral position."""
        strategy = MultiLegStrategy(
            name="Delta Neutral",
            strategy_type=StrategyType.LONG_STRADDLE,
            underlying_symbol="SPY",
            underlying_price=450.0,
        )

        # ATM call and put should have offsetting deltas
        strategy.add_leg(
            OptionLeg(
                symbol="SPY_C450",
                underlying="SPY",
                option_type="call",
                strike=450.0,
                expiry=expiry,
                quantity=1,
                entry_price=8.00,
                delta=0.50,
            )
        )

        strategy.add_leg(
            OptionLeg(
                symbol="SPY_P450",
                underlying="SPY",
                option_type="put",
                strike=450.0,
                expiry=expiry,
                quantity=1,
                entry_price=8.00,
                delta=-0.50,
            )
        )

        greeks = strategy.get_portfolio_greeks()

        # Should be approximately delta neutral
        assert greeks.delta == pytest.approx(0.0, abs=0.01)
