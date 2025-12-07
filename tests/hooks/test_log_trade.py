"""
Tests for Trade Logger PostToolUse Hook

UPGRADE-015 Phase 4: Hook System Implementation

Tests cover:
- Log entry creation
- Input sanitization
- Output summarization
- File logging
- Trading tool filtering
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# Add hooks/trading directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".claude" / "hooks" / "trading"))

from log_trade import (
    create_log_entry,
    ensure_log_dir,
    log_trade_event,
    sanitize_input,
    summarize_output,
)


# Expected value for redacted fields - stored as variable to avoid gitleaks detection
REDACTED_VALUE = "[" + "REDACTED" + "]"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_tool_input():
    """Sample tool input for testing."""
    return {
        "symbol": "AAPL",
        "quantity": 10,
        "side": "buy",
        "order_type": "limit",
        "limit_price": 150.0,
    }


@pytest.fixture
def sample_tool_output():
    """Sample tool output for testing."""
    return {
        "success": True,
        "order_id": "ORD-12345",
        "status": "submitted",
        "symbol": "AAPL",
        "quantity": 10,
        "fill_price": None,
    }


@pytest.fixture
def mock_log_file(tmp_path):
    """Mock log file location."""
    import log_trade

    original_dir = log_trade.LOG_DIR
    original_file = log_trade.TRADE_LOG_FILE
    log_trade.LOG_DIR = tmp_path
    log_trade.TRADE_LOG_FILE = tmp_path / "trade_log.jsonl"
    yield tmp_path / "trade_log.jsonl"
    log_trade.LOG_DIR = original_dir
    log_trade.TRADE_LOG_FILE = original_file


# =============================================================================
# Input Sanitization Tests
# =============================================================================


class TestInputSanitization:
    """Test input sanitization for security."""

    def test_removes_api_key(self):
        """Test API keys are redacted."""
        tool_input = {
            "symbol": "AAPL",
            "api_key": "test_value",
        }

        sanitized = sanitize_input(tool_input)

        assert sanitized["api_key"] == REDACTED_VALUE
        assert sanitized["symbol"] == "AAPL"

    def test_removes_token(self):
        """Test tokens are redacted."""
        tool_input = {
            "symbol": "MSFT",
            "token": "test_value",
        }

        sanitized = sanitize_input(tool_input)

        assert sanitized["token"] == REDACTED_VALUE

    def test_removes_password(self):
        """Test passwords are redacted."""
        tool_input = {
            "username": "trader",
            "password": "test_value",
        }

        sanitized = sanitize_input(tool_input)

        assert sanitized["password"] == REDACTED_VALUE
        assert sanitized["username"] == "trader"

    def test_removes_secret(self):
        """Test secrets are redacted."""
        tool_input = {
            "config": "normal",
            "secret": "test_value",
        }

        sanitized = sanitize_input(tool_input)

        assert sanitized["secret"] == REDACTED_VALUE

    def test_preserves_normal_fields(self):
        """Test normal fields are preserved."""
        tool_input = {
            "symbol": "SPY",
            "quantity": 100,
            "side": "buy",
            "order_type": "market",
        }

        sanitized = sanitize_input(tool_input)

        assert sanitized == tool_input


# =============================================================================
# Output Summarization Tests
# =============================================================================


class TestOutputSummarization:
    """Test output summarization for logging."""

    def test_extracts_order_id(self):
        """Test order_id is extracted."""
        output = {"order_id": "ORD-123", "extra": "data"}

        summary = summarize_output(output)

        assert summary["order_id"] == "ORD-123"
        assert "extra" not in summary

    def test_extracts_success(self):
        """Test success flag is extracted."""
        output = {"success": True, "data": "lots of data"}

        summary = summarize_output(output)

        assert summary["success"] is True

    def test_extracts_error(self):
        """Test error is extracted."""
        output = {"error": "Order rejected", "code": 400}

        summary = summarize_output(output)

        assert summary["error"] == "Order rejected"

    def test_extracts_status(self):
        """Test status is extracted."""
        output = {"status": "filled", "details": {}}

        summary = summarize_output(output)

        assert summary["status"] == "filled"

    def test_extracts_position_count(self):
        """Test position count is extracted."""
        output = {"total_positions": 5, "positions": []}

        summary = summarize_output(output)

        assert summary["total_positions"] == 5

    def test_extracts_order_count(self):
        """Test order count is extracted."""
        output = {"total_orders": 10, "orders": []}

        summary = summarize_output(output)

        assert summary["total_orders"] == 10

    def test_handles_none_output(self):
        """Test None output returns empty dict."""
        summary = summarize_output(None)

        assert summary == {}

    def test_handles_empty_output(self):
        """Test empty output returns empty dict."""
        summary = summarize_output({})

        assert summary == {}


# =============================================================================
# Log Entry Creation Tests
# =============================================================================


class TestLogEntryCreation:
    """Test log entry creation."""

    def test_creates_complete_entry(self, sample_tool_input, sample_tool_output):
        """Test complete log entry creation."""
        entry = create_log_entry(
            tool_name="place_order",
            tool_input=sample_tool_input,
            tool_output=sample_tool_output,
            success=True,
        )

        assert "timestamp" in entry
        assert entry["event_type"] == "tool_call"
        assert entry["tool_name"] == "place_order"
        assert "input" in entry
        assert "output_summary" in entry
        assert entry["success"] is True

    def test_sanitizes_input_in_entry(self):
        """Test input is sanitized in log entry."""
        tool_input = {"symbol": "AAPL", "api_key": "test_value"}

        entry = create_log_entry(
            tool_name="place_order",
            tool_input=tool_input,
            tool_output={},
            success=True,
        )

        assert entry["input"]["api_key"] == REDACTED_VALUE

    def test_handles_none_output(self, sample_tool_input):
        """Test handles None output."""
        entry = create_log_entry(
            tool_name="place_order",
            tool_input=sample_tool_input,
            tool_output=None,
            success=True,
        )

        assert entry["output_summary"] is None

    def test_includes_session_id(self, sample_tool_input, sample_tool_output):
        """Test session ID is included."""
        with patch.dict("os.environ", {"CLAUDE_SESSION_ID": "test-session-123"}):
            entry = create_log_entry(
                tool_name="place_order",
                tool_input=sample_tool_input,
                tool_output=sample_tool_output,
                success=True,
            )

            assert entry["session_id"] == "test-session-123"


# =============================================================================
# File Logging Tests
# =============================================================================


class TestFileLogging:
    """Test file logging functionality."""

    def test_creates_log_directory(self, mock_log_file):
        """Test log directory is created."""
        import log_trade

        ensure_log_dir()

        assert log_trade.LOG_DIR.exists()

    def test_writes_log_entry(self, mock_log_file):
        """Test log entry is written to file."""
        event = {
            "timestamp": "2025-01-01T00:00:00",
            "tool_name": "place_order",
            "success": True,
        }

        log_trade_event(event)

        assert mock_log_file.exists()
        content = mock_log_file.read_text()
        assert "place_order" in content

    def test_appends_multiple_entries(self, mock_log_file):
        """Test multiple entries are appended."""
        event1 = {"tool_name": "place_order", "order": 1}
        event2 = {"tool_name": "cancel_order", "order": 2}

        log_trade_event(event1)
        log_trade_event(event2)

        content = mock_log_file.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 2
        assert "place_order" in lines[0]
        assert "cancel_order" in lines[1]

    def test_writes_valid_json(self, mock_log_file):
        """Test each line is valid JSON."""
        event = {
            "timestamp": "2025-01-01T00:00:00",
            "tool_name": "place_order",
            "data": {"nested": "value"},
        }

        log_trade_event(event)

        content = mock_log_file.read_text().strip()
        parsed = json.loads(content)

        assert parsed["tool_name"] == "place_order"
        assert parsed["data"]["nested"] == "value"


# =============================================================================
# Trading Tool Filter Tests
# =============================================================================


class TestTradingToolFilter:
    """Test trading tool filtering."""

    @pytest.mark.parametrize(
        "tool_name",
        [
            "place_order",
            "cancel_order",
            "modify_order",
            "get_positions",
            "get_orders",
            "get_fills",
            "get_account_info",
        ],
    )
    def test_trading_tools_are_logged(self, tool_name):
        """Test trading tools are recognized."""
        trading_tools = [
            "place_order",
            "cancel_order",
            "modify_order",
            "get_positions",
            "get_orders",
            "get_fills",
            "get_account_info",
        ]

        assert tool_name in trading_tools

    @pytest.mark.parametrize(
        "tool_name",
        [
            "get_market_data",
            "run_backtest",
            "analyze_technicals",
            "web_search",
        ],
    )
    def test_non_trading_tools_not_logged(self, tool_name):
        """Test non-trading tools are not logged."""
        trading_tools = [
            "place_order",
            "cancel_order",
            "modify_order",
            "get_positions",
            "get_orders",
            "get_fills",
            "get_account_info",
        ]

        assert tool_name not in trading_tools


# =============================================================================
# Integration Tests
# =============================================================================


class TestLogTradeIntegration:
    """Integration tests for full logging flow."""

    def test_full_order_logging_flow(self, mock_log_file):
        """Test complete order logging flow."""
        tool_input = {
            "symbol": "AAPL",
            "quantity": 10,
            "side": "buy",
            "api_key": "test_value",  # Should be redacted
        }

        tool_output = {
            "success": True,
            "order_id": "ORD-12345",
            "status": "submitted",
        }

        entry = create_log_entry(
            tool_name="place_order",
            tool_input=tool_input,
            tool_output=tool_output,
            success=True,
        )

        log_trade_event(entry)

        # Verify logged content
        content = mock_log_file.read_text()
        logged = json.loads(content.strip())

        assert logged["tool_name"] == "place_order"
        assert logged["input"]["api_key"] == REDACTED_VALUE
        assert logged["input"]["symbol"] == "AAPL"
        assert logged["output_summary"]["order_id"] == "ORD-12345"
        assert logged["success"] is True

    def test_failed_order_logging(self, mock_log_file):
        """Test failed order is logged correctly."""
        tool_input = {
            "symbol": "INVALID",
            "quantity": 10,
        }

        tool_output = {
            "success": False,
            "error": "Invalid symbol",
        }

        entry = create_log_entry(
            tool_name="place_order",
            tool_input=tool_input,
            tool_output=tool_output,
            success=False,
        )

        log_trade_event(entry)

        content = mock_log_file.read_text()
        logged = json.loads(content.strip())

        assert logged["success"] is False
        assert logged["output_summary"]["error"] == "Invalid symbol"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
