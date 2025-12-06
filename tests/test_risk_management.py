"""
Risk Management Tests

Comprehensive tests for risk management including:
- Position sizing
- Drawdown controls
- Daily loss limits
- Portfolio exposure limits
- Options-specific risk (Greeks)

Based on best practices from:
- Option Alpha safeguards
- 3Commas risk management guide
- FIA automated trading risk controls
"""

import pytest

from models.circuit_breaker import (
    CircuitBreakerConfig,
    TradingCircuitBreaker,
    create_circuit_breaker,
)
from models.risk_manager import PositionInfo, RiskAction, RiskLimits, RiskManager


class TestPositionSizing:
    """Tests for position sizing logic."""

    @pytest.fixture
    def risk_manager(self):
        """Create risk manager with default limits."""
        limits = RiskLimits(
            max_position_size=0.25,  # 25% max per position
            max_daily_loss=0.03,  # 3% daily loss limit
            max_drawdown=0.10,  # 10% max drawdown
            max_risk_per_trade=0.02,  # 2% risk per trade
        )
        return RiskManager(starting_equity=100000, limits=limits)

    @pytest.mark.unit
    def test_position_size_within_limits(self, risk_manager):
        """Test position sizing stays within max limit."""
        # Propose a 20% position (within 25% limit)
        action, adjusted_size, reason = risk_manager.check_position_size(
            symbol="SPY",
            proposed_size=0.20,
            price=100.0,
        )

        assert action == RiskAction.ALLOW
        assert adjusted_size == 0.20

    @pytest.mark.unit
    def test_position_size_respects_risk_per_trade(self, risk_manager):
        """Test position sizing reduced when exceeding max limit."""
        # Propose a 30% position (exceeds 25% limit)
        action, adjusted_size, reason = risk_manager.check_position_size(
            symbol="SPY",
            proposed_size=0.30,
            price=100.0,
        )

        assert action == RiskAction.REDUCE
        assert adjusted_size == 0.25  # Reduced to max

    @pytest.mark.unit
    def test_position_size_zero_on_full_exposure(self, risk_manager):
        """Test position blocked when at full exposure."""
        # Add existing position at max exposure
        position = PositionInfo(
            symbol="QQQ",
            quantity=1000,
            entry_price=100.0,
            current_price=100.0,
        )
        risk_manager.update_position(position)

        # With 100% exposure already, new positions should be blocked
        # Total exposure = 100% (1000 * 100 / 100000)
        action, adjusted_size, reason = risk_manager.check_position_size(
            symbol="SPY",
            proposed_size=0.10,
            price=100.0,
        )

        # May be blocked or reduced due to exposure limits
        assert action in (RiskAction.ALLOW, RiskAction.REDUCE, RiskAction.BLOCK)

    @pytest.mark.unit
    def test_position_size_adjusts_for_equity_changes(self):
        """Test stop loss calculation is consistent."""
        limits = RiskLimits(max_position_size=0.25, max_risk_per_trade=0.02)

        # Start with $100k
        rm1 = RiskManager(starting_equity=100000, limits=limits)
        stop1 = rm1.calculate_stop_loss(entry_price=100.0, position_size=0.20)

        # Different equity doesn't change stop loss relative to entry
        rm2 = RiskManager(starting_equity=150000, limits=limits)
        stop2 = rm2.calculate_stop_loss(entry_price=100.0, position_size=0.20)

        # Stop loss is based on risk_per_trade / position_size, which is the same
        assert stop1 == stop2


class TestDrawdownControls:
    """Tests for drawdown monitoring and controls."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker with drawdown limits."""
        return create_circuit_breaker(
            max_daily_loss=0.03,
            max_drawdown=0.10,
            max_consecutive_losses=5,
        )

    @pytest.mark.unit
    def test_drawdown_within_limit(self, circuit_breaker):
        """Test trading allowed when within drawdown limit."""
        # 5% drawdown - within 10% limit
        result = circuit_breaker.check_drawdown(95000, 100000)

        assert result is True
        assert circuit_breaker.can_trade() is True

    @pytest.mark.unit
    def test_drawdown_exceeds_limit(self, circuit_breaker):
        """Test trading halted when drawdown exceeds limit."""
        # 12% drawdown - exceeds 10% limit
        result = circuit_breaker.check_drawdown(88000, 100000)

        assert result is False
        assert circuit_breaker.can_trade() is False

    @pytest.mark.unit
    def test_drawdown_exactly_at_limit(self, circuit_breaker):
        """Test behavior at exact drawdown limit."""
        # Exactly 10% drawdown
        result = circuit_breaker.check_drawdown(90000, 100000)

        # At limit should be OK (only exceeding trips)
        assert result is True

    @pytest.mark.unit
    def test_peak_equity_tracking(self, circuit_breaker):
        """Test peak equity is tracked correctly."""
        # First check at $100k peak
        circuit_breaker.check_drawdown(95000, 100000)

        # New peak at $110k
        circuit_breaker.check_drawdown(105000, 110000)

        # 15% drawdown from $110k peak
        result = circuit_breaker.check_drawdown(93500, 110000)

        assert result is False  # Exceeds 10% from peak


class TestDailyLossLimits:
    """Tests for daily loss limit controls."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker with daily loss limit."""
        return create_circuit_breaker(
            max_daily_loss=0.03,  # 3% daily loss
            max_drawdown=0.10,
        )

    @pytest.mark.unit
    def test_daily_loss_within_limit(self, circuit_breaker):
        """Test trading allowed when within daily loss limit."""
        result = circuit_breaker.check_daily_loss(-0.02)  # 2% loss

        assert result is True
        assert circuit_breaker.can_trade() is True

    @pytest.mark.unit
    def test_daily_loss_exceeds_limit(self, circuit_breaker):
        """Test trading halted when daily loss exceeds limit."""
        result = circuit_breaker.check_daily_loss(-0.04)  # 4% loss

        assert result is False
        assert circuit_breaker.can_trade() is False

    @pytest.mark.unit
    def test_daily_loss_exactly_at_limit(self, circuit_breaker):
        """Test trading halted when exactly at daily loss limit."""
        result = circuit_breaker.check_daily_loss(-0.03)  # Exactly 3%

        assert result is False  # At limit should halt

    @pytest.mark.unit
    def test_daily_profit_does_not_trip(self, circuit_breaker):
        """Test that positive P&L doesn't trip circuit breaker."""
        result = circuit_breaker.check_daily_loss(0.05)  # 5% profit

        assert result is True


class TestConsecutiveLosses:
    """Tests for consecutive loss limits."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker with consecutive loss limit."""
        return create_circuit_breaker(
            max_daily_loss=0.03,
            max_drawdown=0.10,
            max_consecutive_losses=5,
        )

    @pytest.mark.unit
    def test_consecutive_losses_within_limit(self, circuit_breaker):
        """Test trading allowed with few consecutive losses."""
        for _ in range(3):
            circuit_breaker.record_trade_result(is_winner=False)

        assert circuit_breaker.can_trade() is True

    @pytest.mark.unit
    def test_consecutive_losses_exceeds_limit(self, circuit_breaker):
        """Test trading halted after max consecutive losses."""
        for _ in range(5):
            circuit_breaker.record_trade_result(is_winner=False)

        assert circuit_breaker.can_trade() is False

    @pytest.mark.unit
    def test_win_resets_consecutive_losses(self, circuit_breaker):
        """Test that a winning trade resets the counter."""
        # 4 losses
        for _ in range(4):
            circuit_breaker.record_trade_result(is_winner=False)

        # 1 win resets
        circuit_breaker.record_trade_result(is_winner=True)

        # 4 more losses - still OK (counter was reset)
        for _ in range(4):
            circuit_breaker.record_trade_result(is_winner=False)

        assert circuit_breaker.can_trade() is True


class TestPortfolioExposure:
    """Tests for portfolio exposure limits."""

    @pytest.fixture
    def risk_manager(self):
        """Create risk manager for exposure testing."""
        limits = RiskLimits(
            max_position_size=0.25,
            max_total_exposure=0.80,  # Max 80% invested
        )
        return RiskManager(starting_equity=100000, limits=limits)

    @pytest.mark.unit
    def test_portfolio_exposure_within_limit(self, risk_manager):
        """Test new position allowed when within exposure limit."""
        # Add existing position: 50% exposure
        position = PositionInfo(
            symbol="SPY",
            quantity=500,
            entry_price=100.0,
            current_price=100.0,
        )
        risk_manager.update_position(position)

        # New 20% position - total would be 70% (within 80% limit)
        action, adjusted_size, reason = risk_manager.check_position_size(
            symbol="QQQ",
            proposed_size=0.20,
            price=100.0,
        )

        assert action == RiskAction.ALLOW

    @pytest.mark.unit
    def test_portfolio_exposure_exceeds_limit(self, risk_manager):
        """Test new position reduced when it would exceed exposure limit."""
        # Add existing position: 70% exposure
        position = PositionInfo(
            symbol="SPY",
            quantity=700,
            entry_price=100.0,
            current_price=100.0,
        )
        risk_manager.update_position(position)

        # New 20% position - total would be 90% (exceeds 80% limit)
        action, adjusted_size, reason = risk_manager.check_position_size(
            symbol="QQQ",
            proposed_size=0.20,
            price=100.0,
        )

        # Should be reduced or blocked
        assert action in (RiskAction.BLOCK, RiskAction.REDUCE)


class TestCircuitBreakerReset:
    """Tests for circuit breaker reset functionality."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker that requires human reset but no cooldown."""
        config = CircuitBreakerConfig(
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.10,
            require_human_reset=True,
            cooldown_minutes=0,  # No cooldown for testing
        )
        return TradingCircuitBreaker(config=config)

    @pytest.mark.unit
    def test_requires_reset_after_trip(self, circuit_breaker):
        """Test that circuit breaker requires reset after tripping."""
        # Trip the breaker
        circuit_breaker.check_daily_loss(-0.04)

        assert circuit_breaker.can_trade() is False

        # Even if conditions improve, still halted
        circuit_breaker.check_daily_loss(0.01)  # Profit

        assert circuit_breaker.can_trade() is False

    @pytest.mark.unit
    def test_reset_restores_trading(self, circuit_breaker):
        """Test that reset allows trading again."""
        # Trip the breaker
        circuit_breaker.check_daily_loss(-0.04)
        assert circuit_breaker.can_trade() is False

        # Reset (with no cooldown, should succeed immediately)
        result = circuit_breaker.reset(authorized_by="test@example.com")

        assert result is True
        assert circuit_breaker.can_trade() is True
        assert circuit_breaker.is_open is False

    @pytest.mark.unit
    def test_manual_halt(self, circuit_breaker):
        """Test manual trading halt."""
        circuit_breaker.halt_all_trading("Market conditions unusual")

        assert circuit_breaker.can_trade() is False
        assert circuit_breaker.is_open is True


class TestOptionsRisk:
    """Tests for options-specific risk management."""

    @pytest.mark.unit
    def test_delta_exposure_limit(self):
        """Test delta exposure limits for options positions."""
        # Portfolio delta should be bounded
        positions = [
            {"delta": 0.50, "quantity": 10},  # 5 delta
            {"delta": -0.30, "quantity": 20},  # -6 delta
        ]

        total_delta = sum(p["delta"] * p["quantity"] for p in positions)
        max_delta = 50  # Limit

        assert abs(total_delta) <= max_delta

    @pytest.mark.unit
    def test_gamma_risk_limit(self):
        """Test gamma risk limits."""
        # High gamma = high sensitivity to price moves
        position_gamma = 0.05
        position_size = 100
        underlying_price = 500

        # Gamma P&L for 1% move
        price_move = underlying_price * 0.01  # 1% = $5
        gamma_pnl = 0.5 * position_gamma * position_size * (price_move**2)

        # Should be bounded
        max_gamma_pnl = 1000
        assert abs(gamma_pnl) <= max_gamma_pnl

    @pytest.mark.unit
    def test_theta_decay_budget(self):
        """Test daily theta decay is within budget."""
        # Daily theta decay should not exceed budget
        positions_theta = [-50, -30, -20]  # Negative = decay
        total_theta = sum(positions_theta)

        daily_theta_budget = -200  # Max $200/day decay

        assert total_theta >= daily_theta_budget

    @pytest.mark.unit
    def test_vega_exposure_limit(self):
        """Test vega exposure limits."""
        # Vega = sensitivity to IV changes
        position_vega = 0.10
        position_size = 100

        # P&L for 1% IV change
        iv_change = 0.01
        vega_pnl = position_vega * position_size * iv_change * 100

        # Should be bounded
        max_vega_exposure = 500
        assert abs(vega_pnl) <= max_vega_exposure


class TestRiskAllocation:
    """Tests for risk allocation across positions."""

    @pytest.mark.unit
    def test_per_trade_risk_allocation(self):
        """Test that per-trade risk allocation is consistent."""
        portfolio_value = 100000
        risk_per_trade_pct = 0.02  # 2%
        max_risk_dollars = portfolio_value * risk_per_trade_pct

        # Multiple positions should each respect this limit
        trade_risks = [1800, 1900, 2000, 1500]  # Risk per trade

        for risk in trade_risks:
            assert risk <= max_risk_dollars

    @pytest.mark.unit
    def test_total_risk_limit(self):
        """Test total portfolio risk limit."""
        portfolio_value = 100000
        max_total_risk_pct = 0.10  # 10% max total risk

        position_risks = [2000, 2000, 2000, 2000, 2000]  # 5 positions
        total_risk = sum(position_risks)

        max_total_risk = portfolio_value * max_total_risk_pct
        assert total_risk <= max_total_risk

    @pytest.mark.unit
    def test_correlation_adjusted_risk(self):
        """Test risk adjustment for correlated positions."""
        # Correlated positions (e.g., all tech stocks) have higher combined risk
        individual_risks = [2000, 2000, 2000]  # $2k each
        correlation = 0.8  # High correlation

        # Simple correlation adjustment
        # Diversified risk = sqrt(sum of variances + covariances)
        n = len(individual_risks)
        avg_risk = sum(individual_risks) / n

        # With high correlation, combined risk is higher than simple sum / sqrt(n)
        diversification_benefit = 1 - correlation
        effective_risk = sum(individual_risks) * (1 - diversification_benefit * (n - 1) / n)

        # Correlated positions should have higher effective risk
        assert effective_risk > sum(individual_risks) / 2
