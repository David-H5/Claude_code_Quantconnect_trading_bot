"""
Tests for Risk Validator PreToolUse Hook

UPGRADE-015 Phase 4: Hook System Implementation

Tests cover:
- Order validation against risk limits
- Live trading blocking
- Daily order limits
- Volume limits
- Blocked symbols
- State management
"""

import sys
from pathlib import Path

import pytest


# Add hooks/trading directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".claude" / "hooks" / "trading"))

from risk_validator import (
    RISK_LIMITS,
    create_new_state,
    load_daily_state,
    save_daily_state,
    validate_trading_tool,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def clean_state():
    """Provide clean state for each test."""
    return create_new_state()


@pytest.fixture
def mock_state_file(tmp_path):
    """Mock state file location."""
    import risk_validator

    original = risk_validator.DAILY_STATE_FILE
    risk_validator.DAILY_STATE_FILE = tmp_path / "test_state.json"
    yield tmp_path / "test_state.json"
    risk_validator.DAILY_STATE_FILE = original


# =============================================================================
# State Management Tests
# =============================================================================


class TestStateManagement:
    """Test daily state management functions."""

    def test_create_new_state(self):
        """Test creating new daily state."""
        state = create_new_state()

        assert "date" in state
        assert state["order_count"] == 0
        assert state["total_volume"] == 0.0
        assert state["symbols_traded"] == []

    def test_load_daily_state_fresh(self, mock_state_file):
        """Test loading state when no file exists."""
        state = load_daily_state()

        assert state["order_count"] == 0
        assert state["total_volume"] == 0.0

    def test_save_and_load_state(self, mock_state_file):
        """Test saving and loading state."""
        state = create_new_state()
        state["order_count"] = 5
        state["total_volume"] = 10000.0
        state["symbols_traded"] = ["AAPL", "MSFT"]

        save_daily_state(state)
        loaded = load_daily_state()

        assert loaded["order_count"] == 5
        assert loaded["total_volume"] == 10000.0
        assert "AAPL" in loaded["symbols_traded"]

    def test_state_resets_on_new_day(self, mock_state_file):
        """Test that state resets on new day."""
        state = create_new_state()
        state["date"] = "1999-01-01"  # Old date
        state["order_count"] = 50

        save_daily_state(state)
        loaded = load_daily_state()

        # Should reset to new state
        assert loaded["order_count"] == 0


# =============================================================================
# Order Validation Tests
# =============================================================================


class TestOrderValidation:
    """Test order validation against risk limits."""

    def test_valid_order(self, mock_state_file):
        """Test valid order passes validation."""
        tool_input = {
            "symbol": "AAPL",
            "quantity": 10,
            "limit_price": 150.0,
            "trading_mode": "paper",
        }

        is_valid, message = validate_trading_tool("place_order", tool_input)

        assert is_valid is True
        assert "VALIDATED" in message

    def test_order_exceeds_max_value(self, mock_state_file):
        """Test order exceeding max value is blocked."""
        tool_input = {
            "symbol": "AAPL",
            "quantity": 1000,
            "limit_price": 150.0,  # $150,000 > $50,000 limit
            "trading_mode": "paper",
        }

        is_valid, message = validate_trading_tool("place_order", tool_input)

        assert is_valid is False
        assert "exceeds max" in message

    def test_order_below_minimum(self, mock_state_file):
        """Test order below minimum triggers warning."""
        tool_input = {
            "symbol": "AAPL",
            "quantity": 1,
            "limit_price": 10.0,  # $10 < $100 minimum
            "trading_mode": "paper",
        }

        is_valid, message = validate_trading_tool("place_order", tool_input)

        assert is_valid is False
        assert "below minimum" in message


# =============================================================================
# Live Trading Block Tests
# =============================================================================


class TestLiveTradingBlock:
    """Test live trading is blocked."""

    def test_live_trading_blocked(self, mock_state_file):
        """Test live trading mode is blocked."""
        tool_input = {
            "symbol": "AAPL",
            "quantity": 10,
            "limit_price": 150.0,
            "trading_mode": "live",
        }

        is_valid, message = validate_trading_tool("place_order", tool_input)

        assert is_valid is False
        assert "Live trading is disabled" in message

    def test_paper_trading_allowed(self, mock_state_file):
        """Test paper trading mode is allowed."""
        tool_input = {
            "symbol": "AAPL",
            "quantity": 10,
            "limit_price": 150.0,
            "trading_mode": "paper",
        }

        is_valid, message = validate_trading_tool("place_order", tool_input)

        assert is_valid is True


# =============================================================================
# Daily Limit Tests
# =============================================================================


class TestDailyLimits:
    """Test daily trading limits."""

    def test_daily_order_count_limit(self, mock_state_file):
        """Test daily order count limit enforcement."""

        # Set state near limit
        state = create_new_state()
        state["order_count"] = RISK_LIMITS["max_daily_orders"]
        save_daily_state(state)

        tool_input = {
            "symbol": "AAPL",
            "quantity": 1,
            "limit_price": 150.0,
            "trading_mode": "paper",
        }

        is_valid, message = validate_trading_tool("place_order", tool_input)

        assert is_valid is False
        assert "Daily order limit" in message

    def test_daily_volume_limit(self, mock_state_file):
        """Test daily volume limit enforcement."""
        # Set state near volume limit
        state = create_new_state()
        state["total_volume"] = RISK_LIMITS["max_daily_volume"] - 1000
        save_daily_state(state)

        tool_input = {
            "symbol": "AAPL",
            "quantity": 100,
            "limit_price": 150.0,  # $15,000 would exceed limit
            "trading_mode": "paper",
        }

        is_valid, message = validate_trading_tool("place_order", tool_input)

        assert is_valid is False
        assert "exceed daily volume limit" in message


# =============================================================================
# Cancel Order Tests
# =============================================================================


class TestCancelOrder:
    """Test cancel order validation."""

    def test_valid_cancel(self, mock_state_file):
        """Test valid cancel order."""
        tool_input = {
            "order_id": "ORD-12345",
        }

        is_valid, message = validate_trading_tool("cancel_order", tool_input)

        assert is_valid is True
        assert "Cancel request" in message

    def test_cancel_without_order_id(self, mock_state_file):
        """Test cancel without order_id is blocked."""
        tool_input = {}

        is_valid, message = validate_trading_tool("cancel_order", tool_input)

        assert is_valid is False
        assert "requires order_id" in message


# =============================================================================
# Non-Trading Tool Tests
# =============================================================================


class TestNonTradingTools:
    """Test non-trading tools pass through."""

    def test_non_trading_tool_allowed(self, mock_state_file):
        """Test non-trading tools are allowed."""
        tool_input = {"query": "some query"}

        is_valid, message = validate_trading_tool("get_market_data", tool_input)

        assert is_valid is True


# =============================================================================
# State Tracking Tests
# =============================================================================


class TestStateTracking:
    """Test that state is properly tracked across orders."""

    def test_order_count_increments(self, mock_state_file):
        """Test order count increments after valid order."""
        tool_input = {
            "symbol": "AAPL",
            "quantity": 10,
            "limit_price": 150.0,
            "trading_mode": "paper",
        }

        # First order
        validate_trading_tool("place_order", tool_input)
        state1 = load_daily_state()

        # Second order
        validate_trading_tool("place_order", tool_input)
        state2 = load_daily_state()

        assert state2["order_count"] == state1["order_count"] + 1

    def test_volume_accumulates(self, mock_state_file):
        """Test volume accumulates across orders."""
        tool_input = {
            "symbol": "AAPL",
            "quantity": 10,
            "limit_price": 150.0,  # $1,500
            "trading_mode": "paper",
        }

        validate_trading_tool("place_order", tool_input)
        state = load_daily_state()

        assert state["total_volume"] == 1500.0

    def test_symbols_tracked(self, mock_state_file):
        """Test traded symbols are tracked."""
        # Trade AAPL
        tool_input1 = {
            "symbol": "AAPL",
            "quantity": 10,
            "limit_price": 150.0,
            "trading_mode": "paper",
        }
        validate_trading_tool("place_order", tool_input1)

        # Trade MSFT
        tool_input2 = {
            "symbol": "MSFT",
            "quantity": 10,
            "limit_price": 400.0,
            "trading_mode": "paper",
        }
        validate_trading_tool("place_order", tool_input2)

        state = load_daily_state()

        assert "AAPL" in state["symbols_traded"]
        assert "MSFT" in state["symbols_traded"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
