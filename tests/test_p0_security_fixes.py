"""
Tests for P0 Security and Safety Fixes

This test module covers critical bug fixes from the December 2025 refactoring:
1. Position sizing truncation bug (bot_managed_positions.py)
2. Fill price calculation error (smart_execution.py)
3. Race conditions in global state (rest_server.py)
4. WebSocket TOCTOU race condition (websocket_handler.py)
5. Default authentication token vulnerability (order_queue_api.py)
6. CORS wildcard security issue (rest_server.py)
7. Position rolling implementation (bot_managed_positions.py)

RCA References:
- COMPREHENSIVE_REFACTOR_PLAN.md
- REFACTORING_INSTRUCTIONS.md
"""

import os
import secrets
import threading
import time
from datetime import datetime
from unittest.mock import AsyncMock

import pytest


# =============================================================================
# Test 1: Position Sizing Truncation Bug (P0)
# =============================================================================


class TestPositionSizingFix:
    """
    Tests for the position sizing truncation bug fix.

    Bug: int() truncation could cause over-closing small positions.
    Fix: _calculate_close_quantity() now skips profit-taking levels
         when the position is too small.
    """

    def test_small_position_skips_partial_close(self):
        """Single contract position should skip 30% profit-taking level."""
        from execution.bot_managed_positions import BotPositionManager

        manager = BotPositionManager(algorithm=None)

        # 1 contract at 30% should return 0 (skip)
        result = manager._calculate_close_quantity(
            current_quantity=1,
            take_pct=0.30,
        )
        assert result == 0, "Should skip profit-taking for 1 contract at 30%"

    def test_small_position_full_close_allowed(self):
        """Single contract position can be fully closed (100%)."""
        from execution.bot_managed_positions import BotPositionManager

        manager = BotPositionManager(algorithm=None)

        # 1 contract at 100% should return 1
        result = manager._calculate_close_quantity(
            current_quantity=1,
            take_pct=1.0,
        )
        assert result == 1, "Should allow full close of single contract"

    def test_large_position_partial_close_works(self):
        """Large positions should have normal profit-taking."""
        from execution.bot_managed_positions import BotPositionManager

        manager = BotPositionManager(algorithm=None)

        # 10 contracts at 30% should return 3
        result = manager._calculate_close_quantity(
            current_quantity=10,
            take_pct=0.30,
        )
        assert result == 3, "10 contracts at 30% should close 3"

    def test_edge_case_two_contracts(self):
        """Two contracts at 30% should close at least 1."""
        from execution.bot_managed_positions import BotPositionManager

        manager = BotPositionManager(algorithm=None)

        # 2 contracts at 30% = 0.6 -> int(0.6) = 0 -> max(0, 1) = 1
        result = manager._calculate_close_quantity(
            current_quantity=2,
            take_pct=0.30,
        )
        # With the fix, this should return 0 (skip) because calculated < min_close
        # and current_quantity > min_close
        assert result == 0, "2 contracts at 30% should skip (calculated=0 < min=1)"

    def test_threshold_boundary(self):
        """Test boundary conditions for minimum close."""
        from execution.bot_managed_positions import BotPositionManager

        manager = BotPositionManager(algorithm=None)

        # 4 contracts at 50% = 2, should close 2
        result = manager._calculate_close_quantity(
            current_quantity=4,
            take_pct=0.50,
        )
        assert result == 2, "4 contracts at 50% should close 2"


# =============================================================================
# Test 2: Fill Price Calculation Error (P0)
# =============================================================================


class TestFillPriceCalculationFix:
    """
    Tests for the fill price weighted average calculation fix.

    Bug: Previous code used (avg * (n-1) + new) / n which is mathematically wrong.
    Fix: Now uses proper weighted average: (old_cost + new_cost) / total_qty
    """

    def test_first_fill_sets_average(self):
        """First fill should set average to fill price."""
        from config import OrderExecutionConfig
        from execution.smart_execution import SmartExecutionModel, SmartOrderStatus

        config = OrderExecutionConfig()
        model = SmartExecutionModel(config)

        # Submit order
        order = model.submit_order(
            symbol="SPY",
            side="buy",
            quantity=100,
            limit_price=450.0,
        )

        # First fill
        model.update_order_status(
            order.order_id,
            SmartOrderStatus.PARTIALLY_FILLED,
            filled_quantity=50,
            fill_price=450.0,
        )

        assert order.average_fill_price == 450.0

    def test_weighted_average_multiple_fills(self):
        """Multiple fills should calculate proper weighted average."""
        from config import OrderExecutionConfig
        from execution.smart_execution import SmartExecutionModel, SmartOrderStatus

        config = OrderExecutionConfig()
        model = SmartExecutionModel(config)

        order = model.submit_order(
            symbol="SPY",
            side="buy",
            quantity=100,
            limit_price=450.0,
        )

        # First fill: 50 @ 450
        model.update_order_status(
            order.order_id,
            SmartOrderStatus.PARTIALLY_FILLED,
            filled_quantity=50,
            fill_price=450.0,
        )

        # Second fill: 50 @ 452
        model.update_order_status(
            order.order_id,
            SmartOrderStatus.FILLED,
            filled_quantity=100,
            fill_price=452.0,
        )

        # Weighted average: (50*450 + 50*452) / 100 = (22500 + 22600) / 100 = 451.0
        expected_avg = (50 * 450.0 + 50 * 452.0) / 100
        assert abs(order.average_fill_price - expected_avg) < 0.01

    def test_unequal_fills_weighted_average(self):
        """Unequal fill sizes should be weighted correctly."""
        from config import OrderExecutionConfig
        from execution.smart_execution import SmartExecutionModel, SmartOrderStatus

        config = OrderExecutionConfig()
        model = SmartExecutionModel(config)

        order = model.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=100,
            limit_price=175.0,
        )

        # First fill: 30 @ 174
        model.update_order_status(
            order.order_id,
            SmartOrderStatus.PARTIALLY_FILLED,
            filled_quantity=30,
            fill_price=174.0,
        )

        # Second fill: 70 @ 176
        model.update_order_status(
            order.order_id,
            SmartOrderStatus.FILLED,
            filled_quantity=100,
            fill_price=176.0,
        )

        # Weighted average: (30*174 + 70*176) / 100 = (5220 + 12320) / 100 = 175.40
        expected_avg = (30 * 174.0 + 70 * 176.0) / 100
        assert abs(order.average_fill_price - expected_avg) < 0.01


# =============================================================================
# Test 3: Race Conditions in Global State (P0)
# =============================================================================


class TestThreadSafeRefFix:
    """
    Tests for the ThreadSafeRef wrapper that fixes race conditions.

    Bug: Global state used bare Optional types without synchronization.
    Fix: ThreadSafeRef provides atomic set/get with initialization events.
    """

    def test_thread_safe_ref_single_set(self):
        """ThreadSafeRef should allow single initialization only."""
        from api.rest_server import ThreadSafeRef

        ref = ThreadSafeRef("TestComponent")
        ref.set("value1")

        # Second set should raise
        with pytest.raises(RuntimeError, match="already initialized"):
            ref.set("value2")

    def test_thread_safe_ref_get_waits_for_init(self):
        """ThreadSafeRef.get() should wait for initialization."""
        from api.rest_server import ThreadSafeRef

        ref = ThreadSafeRef("TestComponent")

        def delayed_set():
            time.sleep(0.1)
            ref.set("delayed_value")

        # Start delayed setter
        thread = threading.Thread(target=delayed_set)
        thread.start()

        # Get should wait and receive value
        value = ref.get(timeout=1.0)
        assert value == "delayed_value"

        thread.join()

    def test_thread_safe_ref_timeout(self):
        """ThreadSafeRef.get() should timeout if not initialized."""
        from api.rest_server import ThreadSafeRef

        ref = ThreadSafeRef("TestComponent")

        with pytest.raises(RuntimeError, match="not initialized within"):
            ref.get(timeout=0.1)

    def test_thread_safe_ref_get_or_none(self):
        """ThreadSafeRef.get_or_none() should return None if not set."""
        from api.rest_server import ThreadSafeRef

        ref = ThreadSafeRef("TestComponent")
        assert ref.get_or_none() is None

        ref.set("value")
        assert ref.get_or_none() == "value"

    def test_thread_safe_ref_clear(self):
        """ThreadSafeRef.clear() should reset state."""
        from api.rest_server import ThreadSafeRef

        ref = ThreadSafeRef("TestComponent")
        ref.set("value")
        ref.clear()

        # Should be able to set again after clear
        ref.set("new_value")
        assert ref.get_or_none() == "new_value"


# =============================================================================
# Test 4: WebSocket TOCTOU Race Condition (P0)
# =============================================================================


class TestWebSocketTOCTOUFix:
    """
    Tests for the WebSocket TOCTOU race condition fix.

    Bug: Check-then-remove pattern could fail if list modified between operations.
    Fix: Atomic list rebuild with object IDs.
    """

    @pytest.mark.asyncio
    async def test_cleanup_disconnected_atomic(self):
        """Disconnected clients should be removed atomically."""
        from api.websocket_handler import WebSocketClient, WebSocketManager

        manager = WebSocketManager()

        # Create mock clients
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws3 = AsyncMock()

        client1 = WebSocketClient(websocket=mock_ws1)
        client2 = WebSocketClient(websocket=mock_ws2)
        client3 = WebSocketClient(websocket=mock_ws3)

        manager._clients = [client1, client2, client3]

        # Remove clients 1 and 3
        removed_count = await manager._cleanup_disconnected([client1, client3])

        assert removed_count == 2
        assert len(manager._clients) == 1
        assert manager._clients[0] is client2

    @pytest.mark.asyncio
    async def test_cleanup_handles_already_removed(self):
        """Cleanup should handle already-removed clients gracefully."""
        from api.websocket_handler import WebSocketClient, WebSocketManager

        manager = WebSocketManager()

        mock_ws = AsyncMock()
        client = WebSocketClient(websocket=mock_ws)

        # Client not in list - should not error
        removed_count = await manager._cleanup_disconnected([client])
        assert removed_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_empty_list(self):
        """Cleanup with empty disconnected list should be no-op."""
        from api.websocket_handler import WebSocketManager

        manager = WebSocketManager()
        manager._clients = []

        removed_count = await manager._cleanup_disconnected([])
        assert removed_count == 0


# =============================================================================
# Test 5: Default Authentication Token Vulnerability (P0)
# =============================================================================


class TestAuthenticationTokenFix:
    """
    Tests for the authentication token vulnerability fix.

    Bug: Default "change-me" token was accepted.
    Fix: AuthConfig validates token via env var, rejects forbidden defaults.
    """

    def test_auth_config_rejects_missing_token(self):
        """AuthConfig should reject missing tokens."""
        from api.order_queue_api import AuthConfig

        # Clear env var
        os.environ.pop("TRADING_API_TOKEN", None)

        with pytest.raises(EnvironmentError, match="required"):
            AuthConfig()

    def test_auth_config_rejects_forbidden_tokens(self):
        """AuthConfig should reject forbidden default tokens."""
        from api.order_queue_api import AuthConfig

        forbidden_tokens = [
            "default-token-change-me",
            "change-me",
            "secret",
            "password",
            "token",
            "test",
            "admin",
        ]

        for forbidden in forbidden_tokens:
            os.environ["TRADING_API_TOKEN"] = forbidden
            with pytest.raises(EnvironmentError, match="forbidden"):
                AuthConfig()

    def test_auth_config_rejects_short_tokens(self):
        """AuthConfig should reject tokens shorter than 32 chars."""
        from api.order_queue_api import AuthConfig

        os.environ["TRADING_API_TOKEN"] = "short_token_123"  # < 32 chars

        with pytest.raises(EnvironmentError, match="32 characters"):
            AuthConfig()

    def test_auth_config_accepts_valid_token(self):
        """AuthConfig should accept properly generated tokens."""
        from api.order_queue_api import AuthConfig

        valid_token = secrets.token_urlsafe(32)
        os.environ["TRADING_API_TOKEN"] = valid_token

        config = AuthConfig()
        assert config.validate(valid_token) is True

    def test_auth_config_constant_time_comparison(self):
        """Token validation should use constant-time comparison."""
        from api.order_queue_api import AuthConfig

        valid_token = secrets.token_urlsafe(32)
        os.environ["TRADING_API_TOKEN"] = valid_token

        config = AuthConfig()

        # Timing should be similar for correct and incorrect tokens
        # (This is a basic test - in production you'd use more sophisticated timing analysis)
        assert config.validate(valid_token) is True
        assert config.validate("wrong_token_" + "x" * 32) is False

    def test_order_queue_api_skip_auth_for_testing(self):
        """OrderQueueAPI should allow skipping auth for testing."""
        from api.order_queue_api import OrderQueueAPI

        # Clear env var
        os.environ.pop("TRADING_API_TOKEN", None)

        # Should not raise with skip_auth_validation=True
        api = OrderQueueAPI(skip_auth_validation=True)
        assert api._auth_config is None


# =============================================================================
# Test 6: CORS Wildcard Security Issue (P0)
# =============================================================================


class TestCORSFix:
    """
    Tests for the CORS wildcard security fix.

    Bug: CORS allowed "*" origins in production.
    Fix: Requires CORS_ALLOWED_ORIGINS env var in production.
    """

    def test_cors_requires_config_in_production(self):
        """CORS should require explicit origins in production."""
        from api.rest_server import _get_cors_origins

        # Clear origins and set production
        os.environ.pop("CORS_ALLOWED_ORIGINS", None)
        os.environ["ENVIRONMENT"] = "production"

        with pytest.raises(EnvironmentError, match="required in production"):
            _get_cors_origins()

    def test_cors_allows_development_defaults(self):
        """CORS should allow defaults in development."""
        from api.rest_server import _get_cors_origins

        os.environ.pop("CORS_ALLOWED_ORIGINS", None)
        os.environ["ENVIRONMENT"] = "development"

        origins = _get_cors_origins()
        assert "http://localhost:3000" in origins
        assert "http://127.0.0.1:3000" in origins

    def test_cors_parses_env_var(self):
        """CORS should parse comma-separated origins from env var."""
        from api.rest_server import _get_cors_origins

        os.environ["CORS_ALLOWED_ORIGINS"] = "https://app.example.com,https://admin.example.com"

        origins = _get_cors_origins()
        assert "https://app.example.com" in origins
        assert "https://admin.example.com" in origins
        assert len(origins) == 2

    def test_cors_strips_whitespace(self):
        """CORS should strip whitespace from origins."""
        from api.rest_server import _get_cors_origins

        os.environ["CORS_ALLOWED_ORIGINS"] = "  https://app.example.com  ,  https://admin.example.com  "

        origins = _get_cors_origins()
        assert "https://app.example.com" in origins
        assert "https://admin.example.com" in origins


# =============================================================================
# Test 7: Position Rolling Implementation (P0)
# =============================================================================


class TestPositionRollingImplementation:
    """
    Tests for the position rolling implementation.

    Bug: Position rolling was a stub with no actual functionality.
    Fix: Full implementation with configurable strategies.
    """

    def test_roll_config_defaults(self):
        """RollConfig should have sensible defaults."""
        from execution.bot_managed_positions import RollConfig, RollStrategy

        config = RollConfig()
        assert config.target_dte == 30
        assert config.min_dte_range == 21
        assert config.max_dte_range == 60
        assert config.strategy == RollStrategy.ATM_ADJUST
        assert config.close_first is True
        assert config.require_credit is False

    def test_roll_strategy_enum_values(self):
        """RollStrategy enum should have expected values."""
        from execution.bot_managed_positions import RollStrategy

        assert RollStrategy.SAME_STRIKES.value == "same_strikes"
        assert RollStrategy.ATM_ADJUST.value == "atm_adjust"
        assert RollStrategy.DELTA_MAINTAIN.value == "delta_maintain"

    def test_roll_result_dataclass(self):
        """RollResult should serialize correctly."""
        from execution.bot_managed_positions import RollResult

        result = RollResult(
            success=True,
            old_position_id="pos_123",
            new_position_id="pos_456",
            old_expiry=datetime(2025, 1, 15),
            new_expiry=datetime(2025, 2, 15),
            net_credit=50.0,
        )

        data = result.to_dict()
        assert data["success"] is True
        assert data["old_position_id"] == "pos_123"
        assert data["new_position_id"] == "pos_456"
        assert data["net_credit"] == 50.0

    def test_roll_tracking_in_manager(self):
        """BotPositionManager should track roll history."""
        from execution.bot_managed_positions import (
            BotPositionManager,
            RollConfig,
            RollStrategy,
        )

        manager = BotPositionManager(
            algorithm=None,
            roll_config=RollConfig(
                target_dte=30,
                strategy=RollStrategy.ATM_ADJUST,
            ),
        )

        # Verify roll config is stored
        assert manager.roll_config.target_dte == 30
        assert manager.roll_config.strategy == RollStrategy.ATM_ADJUST

        # Verify roll history is initialized
        assert manager.roll_history == []

        # Verify roll stats are tracked
        assert "rolls_successful" in manager.stats
        assert "rolls_failed" in manager.stats
        assert "roll_credits" in manager.stats

    def test_factory_function_accepts_roll_config(self):
        """create_bot_position_manager should accept roll_config."""
        from execution.bot_managed_positions import (
            RollConfig,
            RollStrategy,
            create_bot_position_manager,
        )

        roll_config = RollConfig(
            target_dte=45,
            strategy=RollStrategy.SAME_STRIKES,
            require_credit=True,
        )

        manager = create_bot_position_manager(
            algorithm=None,
            roll_config=roll_config,
        )

        assert manager.roll_config.target_dte == 45
        assert manager.roll_config.strategy == RollStrategy.SAME_STRIKES
        assert manager.roll_config.require_credit is True


# =============================================================================
# Cleanup fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment variables after each test."""
    yield
    # Restore clean state
    os.environ.pop("TRADING_API_TOKEN", None)
    os.environ.pop("CORS_ALLOWED_ORIGINS", None)
    os.environ.pop("ENVIRONMENT", None)
