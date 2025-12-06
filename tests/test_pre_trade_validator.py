"""
Tests for Pre-Trade Validator

These tests verify the pre-trade validation system correctly
enforces position limits, risk constraints, and data quality checks.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from execution.pre_trade_validator import (
    Order,
    PreTradeValidator,
    ValidationConfig,
    ValidationStatus,
    create_validator,
)


@pytest.fixture
def config():
    """Create a test configuration."""
    return ValidationConfig(
        max_position_pct=0.25,
        max_total_exposure_pct=1.0,
        max_daily_loss_pct=0.03,
        max_order_value=50000.0,
        min_order_value=100.0,
        max_data_age_seconds=5,
        dedup_window_seconds=1,
        enforce_circuit_breaker=True,
    )


@pytest.fixture
def portfolio():
    """Create a test portfolio."""
    return {
        "total_value": 100000.0,
        "daily_pnl_pct": -0.01,  # -1% daily loss
        "positions": {
            "SPY": {"market_value": 20000.0},
            "AAPL": {"market_value": 15000.0},
        },
        "market_data": {
            "SPY": {"volume": 1000000},
            "AAPL": {"volume": 500000},
        },
    }


@pytest.fixture
def circuit_breaker():
    """Create a mock circuit breaker."""
    breaker = MagicMock()
    breaker.can_trade.return_value = True
    breaker.get_status.return_value = {"state": "closed"}
    return breaker


@pytest.fixture
def validator(config, portfolio, circuit_breaker):
    """Create a configured validator."""
    v = PreTradeValidator(
        config=config,
        portfolio=portfolio,
        circuit_breaker=circuit_breaker,
    )
    # Add price data
    v.update_price("SPY", 450.0)
    v.update_price("AAPL", 175.0)
    return v


class TestValidatorCreation:
    """Tests for validator creation."""

    def test_create_validator_with_defaults(self):
        """Validator can be created with default settings."""
        validator = create_validator()
        assert validator is not None
        assert validator.config.max_position_pct == 0.25

    def test_create_validator_with_custom_settings(self):
        """Validator can be created with custom settings."""
        validator = create_validator(
            max_position_pct=0.30,
            max_daily_loss_pct=0.05,
        )
        assert validator.config.max_position_pct == 0.30
        assert validator.config.max_daily_loss_pct == 0.05


class TestCircuitBreakerCheck:
    """Tests for circuit breaker validation."""

    def test_passes_when_trading_allowed(self, validator):
        """Validation passes when circuit breaker allows trading."""
        order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order)

        cb_check = next(c for c in result.checks if c.name == "circuit_breaker")
        assert cb_check.passed

    def test_fails_when_trading_halted(self, validator, circuit_breaker):
        """Validation fails when circuit breaker halts trading."""
        circuit_breaker.can_trade.return_value = False
        circuit_breaker.get_status.return_value = {
            "state": "open",
            "trip_reason": "daily_loss_exceeded",
        }

        order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order)

        assert not result.approved
        cb_check = next(c for c in result.checks if c.name == "circuit_breaker")
        assert not cb_check.passed
        assert "daily_loss" in cb_check.message

    def test_skipped_when_no_circuit_breaker(self, config, portfolio):
        """Check is skipped when no circuit breaker configured."""
        validator = PreTradeValidator(config=config, portfolio=portfolio)
        validator.update_price("SPY", 450.0)

        order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order)

        cb_check = next(c for c in result.checks if c.name == "circuit_breaker")
        assert cb_check.status == ValidationStatus.SKIPPED


class TestPositionLimitCheck:
    """Tests for position limit validation."""

    def test_passes_within_limit(self, validator):
        """Validation passes when position is within limit."""
        order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order)

        pos_check = next(c for c in result.checks if c.name == "position_limit")
        assert pos_check.passed

    def test_fails_exceeds_limit(self, validator):
        """Validation fails when position exceeds limit."""
        # Current SPY: $20,000 (20%)
        # Order: 50 shares * $450 = $22,500
        # New position: $42,500 (42.5%) > 25% limit
        order = Order(symbol="SPY", quantity=50, side="buy", order_type="limit")
        result = validator.validate(order)

        pos_check = next(c for c in result.checks if c.name == "position_limit")
        assert not pos_check.passed
        assert "exceed limit" in pos_check.message

    def test_sell_reduces_position(self, validator):
        """Selling reduces position, should always pass limit check."""
        order = Order(symbol="SPY", quantity=20, side="sell", order_type="limit")
        result = validator.validate(order)

        pos_check = next(c for c in result.checks if c.name == "position_limit")
        assert pos_check.passed


class TestDailyLossLimitCheck:
    """Tests for daily loss limit validation."""

    def test_passes_within_limit(self, validator):
        """Validation passes when within daily loss limit."""
        order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order)

        loss_check = next(c for c in result.checks if c.name == "daily_loss_limit")
        assert loss_check.passed

    def test_fails_exceeds_limit(self, validator, portfolio):
        """Validation fails when daily loss limit exceeded."""
        portfolio["daily_pnl_pct"] = -0.04  # 4% loss > 3% limit
        validator.update_portfolio(portfolio)

        order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order)

        loss_check = next(c for c in result.checks if c.name == "daily_loss_limit")
        assert not loss_check.passed


class TestOrderValueCheck:
    """Tests for order value validation."""

    def test_passes_within_range(self, validator):
        """Validation passes when order value is within range."""
        order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order)

        value_check = next(c for c in result.checks if c.name == "order_value")
        assert value_check.passed

    def test_fails_exceeds_max(self, validator):
        """Validation fails when order value exceeds maximum."""
        # 200 shares * $450 = $90,000 > $50,000 max
        order = Order(symbol="SPY", quantity=200, side="buy", order_type="limit")
        result = validator.validate(order)

        value_check = next(c for c in result.checks if c.name == "order_value")
        assert not value_check.passed
        assert "exceeds max" in value_check.message

    def test_warning_below_min(self, validator):
        """Validation warns when order value is below minimum."""
        validator.update_price("AAPL", 50.0)  # Low price
        order = Order(symbol="AAPL", quantity=1, side="buy", order_type="limit")
        result = validator.validate(order)

        value_check = next(c for c in result.checks if c.name == "order_value")
        assert value_check.status == ValidationStatus.WARNING


class TestDataFreshnessCheck:
    """Tests for data freshness validation."""

    def test_passes_fresh_data(self, validator):
        """Validation passes when price data is fresh."""
        validator.update_price("SPY", 450.0)  # Fresh update

        order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order)

        fresh_check = next(c for c in result.checks if c.name == "data_freshness")
        assert fresh_check.passed

    def test_fails_stale_data(self, validator):
        """Validation fails when price data is stale."""
        # Manually make price stale
        validator._price_cache["SPY"] = (450.0, datetime.utcnow() - timedelta(seconds=10))

        order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order)

        fresh_check = next(c for c in result.checks if c.name == "data_freshness")
        assert not fresh_check.passed
        assert "stale" in fresh_check.message.lower() or "old" in fresh_check.message.lower()


class TestDuplicateOrderCheck:
    """Tests for duplicate order detection."""

    def test_passes_first_order(self, validator):
        """First order always passes duplicate check."""
        order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order)

        dup_check = next(c for c in result.checks if c.name == "duplicate_order")
        assert dup_check.passed

    def test_fails_duplicate_order(self, validator):
        """Duplicate order within window fails."""
        order1 = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        validator.validate(order1)  # First order

        # Same order immediately after
        order2 = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order2)

        dup_check = next(c for c in result.checks if c.name == "duplicate_order")
        assert not dup_check.passed
        assert "duplicate" in dup_check.message.lower()

    def test_different_order_passes(self, validator):
        """Different order passes duplicate check."""
        order1 = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        validator.validate(order1)

        # Different quantity
        order2 = Order(symbol="SPY", quantity=20, side="buy", order_type="limit")
        result = validator.validate(order2)

        dup_check = next(c for c in result.checks if c.name == "duplicate_order")
        assert dup_check.passed


class TestLiquidityCheck:
    """Tests for liquidity validation."""

    def test_passes_sufficient_liquidity(self, validator):
        """Validation passes when liquidity is sufficient."""
        order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order)

        liq_check = next(c for c in result.checks if c.name == "liquidity")
        assert liq_check.passed

    def test_warning_low_liquidity(self, validator, portfolio):
        """Validation warns when liquidity is low."""
        portfolio["market_data"]["SPY"]["volume"] = 50  # Low volume
        validator.update_portfolio(portfolio)

        order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order)

        liq_check = next(c for c in result.checks if c.name == "liquidity")
        assert liq_check.status == ValidationStatus.WARNING


class TestComboOrderBalance:
    """Tests for multi-leg order balance validation."""

    def test_balanced_combo_passes(self, validator):
        """Balanced combo order passes."""
        # For balanced spread: buy 1 lower, sell 1 higher (quantities positive, side determines direction)
        leg1 = Order(symbol="SPY_CALL_450", quantity=1, side="buy", order_type="limit")
        leg2 = Order(symbol="SPY_CALL_460", quantity=1, side="sell", order_type="limit")

        combo = Order(
            symbol="SPY_COMBO",
            quantity=1,
            side="buy",
            order_type="limit",
            is_combo=True,
            legs=[leg1, leg2],
        )

        result = validator.validate(combo)

        balance_check = next(c for c in result.checks if c.name == "combo_balance")
        assert balance_check.passed

    def test_unbalanced_combo_warns(self, validator):
        """Unbalanced combo order warns."""
        leg1 = Order(symbol="SPY_CALL_450", quantity=2, side="buy", order_type="limit")
        leg2 = Order(symbol="SPY_CALL_460", quantity=-1, side="sell", order_type="limit")

        combo = Order(
            symbol="SPY_COMBO",
            quantity=1,
            side="buy",
            order_type="limit",
            is_combo=True,
            legs=[leg1, leg2],
        )

        result = validator.validate(combo)

        balance_check = next(c for c in result.checks if c.name == "combo_balance")
        assert balance_check.status == ValidationStatus.WARNING


class TestValidationResult:
    """Tests for validation result handling."""

    def test_approved_when_all_pass(self, validator):
        """Order approved when all checks pass."""
        order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order)

        assert result.approved
        assert len(result.failed_checks) == 0

    def test_rejected_when_any_fails(self, validator, circuit_breaker):
        """Order rejected when any check fails."""
        circuit_breaker.can_trade.return_value = False
        circuit_breaker.get_status.return_value = {"state": "open", "trip_reason": "test"}

        order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order)

        assert not result.approved
        assert len(result.failed_checks) > 0

    def test_result_to_dict(self, validator):
        """Result can be converted to dictionary."""
        order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order)

        result_dict = result.to_dict()

        assert "approved" in result_dict
        assert "timestamp" in result_dict
        assert "checks" in result_dict
        assert "failed_count" in result_dict


class TestPortfolioUpdate:
    """Tests for portfolio update functionality."""

    def test_update_portfolio(self, validator):
        """Portfolio can be updated."""
        new_portfolio = {
            "total_value": 150000.0,
            "daily_pnl_pct": 0.02,
            "positions": {},
        }

        validator.update_portfolio(new_portfolio)

        assert validator.portfolio["total_value"] == 150000.0

    def test_update_price(self, validator):
        """Price can be updated."""
        validator.update_price("MSFT", 350.0)

        assert validator._price_cache["MSFT"][0] == 350.0


@pytest.mark.unit
class TestIntegrationWithCircuitBreaker:
    """Integration tests with actual circuit breaker."""

    def test_validator_with_real_circuit_breaker(self, config, portfolio):
        """Validator works with real circuit breaker."""
        from models.circuit_breaker import create_circuit_breaker

        breaker = create_circuit_breaker(
            max_daily_loss=0.03,
            require_human_reset=False,
        )

        validator = PreTradeValidator(
            config=config,
            portfolio=portfolio,
            circuit_breaker=breaker,
        )
        validator.update_price("SPY", 450.0)

        order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit")
        result = validator.validate(order)

        assert result.approved

        # Trip the breaker
        breaker.halt_all_trading("Test")

        result = validator.validate(order)
        assert not result.approved
