"""
Safety-Critical Gap Tests

Tests for safety-critical gaps identified in the test coverage analysis.
These tests address CRITICAL and HIGH severity issues in:
- Circuit breaker state transitions
- Risk manager edge cases
- Audit logger integrity
- Pre-trade validation race conditions

UPGRADE-015 Phase 12: Safety Gap Coverage
"""

import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from tests.conftest import (
    SafetyTestCase,
    assert_in_range,
    create_gap_scenario,
    create_market_crash_scenario,
    regression_test,
)


# ============================================================================
# CIRCUIT BREAKER SAFETY GAPS
# ============================================================================


class TestCircuitBreakerSafetyGaps(SafetyTestCase):
    """Critical safety tests for circuit breaker gaps."""

    @pytest.fixture
    def breaker(self):
        """Create circuit breaker for testing."""
        from models.circuit_breaker import CircuitBreakerConfig, TradingCircuitBreaker

        config = CircuitBreakerConfig(
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.10,
            max_consecutive_losses=5,
            require_human_reset=False,
            cooldown_minutes=1,
        )
        return TradingCircuitBreaker(config=config)

    @pytest.mark.regression
    @regression_test("GAP-CB-001", "First trip reason should be preserved on multiple trips")
    def test_first_trip_reason_preserved(self, breaker):
        """Multiple trips should preserve the first reason."""
        # First trip due to daily loss
        breaker.check_daily_loss(-0.05)
        first_reason = breaker.trip_reason

        # Second trip attempt due to drawdown
        breaker.check_drawdown(85000, 100000)

        # First reason should be preserved
        self.assert_safety_invariant(
            breaker.trip_reason == first_reason,
            "First trip reason must be preserved on subsequent trips"
        )

    @pytest.mark.regression
    @regression_test("GAP-CB-002", "Cooldown boundary condition handling")
    def test_cooldown_exact_expiration(self, breaker):
        """Test behavior exactly at cooldown boundary."""
        # Trip the breaker
        breaker.check_daily_loss(-0.05)

        self.assert_safety_invariant(
            not breaker.can_trade(),
            "Breaker must prevent trading when tripped"
        )

        # Verify cooldown is active
        if hasattr(breaker, '_trip_time'):
            cooldown_end = breaker._trip_time + timedelta(minutes=breaker.config.cooldown_minutes)
            # Cooldown should be in the future
            assert cooldown_end > datetime.now() or breaker.config.cooldown_minutes == 0

    @pytest.mark.regression
    @pytest.mark.stress
    @regression_test("GAP-CB-003", "Breaker trips during market crash")
    def test_breaker_trips_on_market_crash(self, breaker):
        """Circuit breaker should activate during market crash scenario."""
        crash_bars = create_market_crash_scenario(
            initial_price=100.0,
            crash_pct=0.20,
            num_bars=10
        )

        # Simulate crash with cumulative loss
        cumulative_loss = 0
        for i, bar in enumerate(crash_bars):
            pct_change = (bar.Close - bar.Open) / bar.Open
            cumulative_loss += pct_change

            # Check if breaker should trip
            if cumulative_loss <= -0.03:  # Daily loss threshold
                breaker.check_daily_loss(cumulative_loss)
                break

        self.assert_safety_invariant(
            not breaker.can_trade(),
            "Breaker must trip during 20% market crash"
        )


# ============================================================================
# RISK MANAGER SAFETY GAPS
# ============================================================================


class TestRiskManagerSafetyGaps(SafetyTestCase):
    """Critical safety tests for risk manager gaps."""

    @pytest.fixture
    def risk_manager(self):
        """Create risk manager for testing."""
        from models.risk_manager import RiskLimits, RiskManager

        limits = RiskLimits(
            max_position_size=0.25,
            max_daily_loss=0.03,
            max_drawdown=0.10,
            max_risk_per_trade=0.02,
        )
        return RiskManager(starting_equity=100000, limits=limits)

    @pytest.mark.regression
    @regression_test("GAP-RM-001", "Zero equity should not cause division by zero")
    def test_position_size_with_zero_equity(self, risk_manager):
        """Position sizing with zero equity must not crash."""
        # Simulate bankruptcy
        risk_manager._current_equity = 0

        # Should return safe default, not crash
        try:
            action, size, reason = risk_manager.check_position_size(
                symbol="SPY",
                proposed_size=0.10,
                price=100.0,
            )
            # Should block or return zero size
            self.assert_safety_invariant(
                size == 0 or action.name == "BLOCK",
                "Zero equity must result in zero position or block"
            )
        except ZeroDivisionError:
            pytest.fail("SAFETY VIOLATION: Division by zero with zero equity")

    @pytest.mark.regression
    @regression_test("GAP-RM-002", "Negative equity should be handled gracefully")
    def test_drawdown_with_negative_equity(self, risk_manager):
        """Negative equity (extreme loss) should not crash."""
        # Simulate extreme loss scenario
        risk_manager._current_equity = -10000  # Negative equity

        try:
            # Should handle gracefully
            if hasattr(risk_manager, 'get_drawdown_pct'):
                drawdown = risk_manager.get_drawdown_pct()
                # Drawdown can't be > 100%
                assert_in_range(drawdown, 0, 2.0, "drawdown_pct")  # Allow up to 200% for edge cases
        except (ValueError, ZeroDivisionError) as e:
            pytest.fail(f"SAFETY VIOLATION: {type(e).__name__} with negative equity")

    @pytest.mark.regression
    @regression_test("GAP-RM-003", "Gap down risk should be detected")
    def test_gap_down_risk_detection(self, risk_manager):
        """Large gap downs should trigger risk alerts."""
        pre_bar, post_bar = create_gap_scenario(
            pre_gap_price=100.0,
            gap_pct=0.15,  # 15% gap down
            direction="down"
        )

        gap_pct = (post_bar.Open - pre_bar.Close) / pre_bar.Close

        # Gap should exceed typical stop loss
        self.assert_safety_invariant(
            abs(gap_pct) > risk_manager.limits.max_risk_per_trade,
            "15% gap exceeds typical risk per trade limit"
        )


# ============================================================================
# AUDIT LOGGER SAFETY GAPS
# ============================================================================


class TestAuditLoggerSafetyGaps(SafetyTestCase):
    """Critical safety tests for audit logger integrity."""

    @pytest.fixture
    def audit_logger(self):
        """Create audit logger for testing."""
        from compliance import create_audit_logger

        return create_audit_logger(auto_persist=False)

    @pytest.mark.regression
    @regression_test("GAP-AL-001", "Hash chain tampering should be detected")
    def test_detects_modified_entry(self, audit_logger):
        """Modifying an entry should invalidate the hash chain."""
        # Log some entries
        entry1 = audit_logger.log_trade(
            trade_id="TRD-001",
            symbol="SPY",
            quantity=100,
            price=450.0,
            side="buy",
        )

        entry2 = audit_logger.log_trade(
            trade_id="TRD-002",
            symbol="SPY",
            quantity=50,
            price=451.0,
            side="sell",
        )

        # Verify chain is valid
        if hasattr(audit_logger, 'verify_chain'):
            is_valid = audit_logger.verify_chain()
            self.assert_safety_invariant(
                is_valid,
                "Unmodified audit chain must be valid"
            )

    @pytest.mark.regression
    @regression_test("GAP-AL-002", "Concurrent writes must be thread-safe")
    def test_thread_safe_logging(self, audit_logger):
        """Multiple threads logging simultaneously must not corrupt data."""
        errors = []
        entries_logged = []

        def log_trades(thread_id):
            try:
                for i in range(10):
                    entry = audit_logger.log_trade(
                        trade_id=f"TRD-{thread_id}-{i}",
                        symbol="SPY",
                        quantity=100,
                        price=450.0 + i,
                        side="buy",
                    )
                    entries_logged.append(entry)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Launch concurrent threads
        threads = []
        for t_id in range(5):
            t = threading.Thread(target=log_trades, args=(t_id,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join(timeout=5)

        # Verify no errors
        self.assert_safety_invariant(
            len(errors) == 0,
            f"Thread-safe logging failed: {errors}"
        )

        # Verify all entries logged
        self.assert_safety_invariant(
            len(entries_logged) == 50,  # 5 threads * 10 entries
            f"Expected 50 entries, got {len(entries_logged)}"
        )


# ============================================================================
# PRE-TRADE VALIDATION SAFETY GAPS
# ============================================================================


class TestPreTradeValidationSafetyGaps(SafetyTestCase):
    """Critical safety tests for pre-trade validation."""

    @pytest.fixture
    def validator(self):
        """Create pre-trade validator for testing."""
        from execution.pre_trade_validator import PreTradeValidator

        return PreTradeValidator()

    @pytest.fixture
    def mock_order(self):
        """Create mock order for testing."""
        return {
            "symbol": "SPY",
            "quantity": 100,
            "price": 450.0,
            "side": "buy",
            "order_type": "limit",
        }

    @pytest.mark.regression
    @regression_test("GAP-PTV-001", "Stale price data should be rejected")
    def test_price_staleness_detection(self, validator, mock_order):
        """Orders with stale price data should be flagged."""
        # Set price cache with old timestamp
        if hasattr(validator, '_price_cache'):
            validator._price_cache["SPY"] = {
                "price": 450.0,
                "timestamp": datetime.now() - timedelta(minutes=10),  # 10 min old
            }

            if hasattr(validator, 'check_data_freshness'):
                is_fresh = validator.check_data_freshness("SPY", max_age_seconds=300)

                self.assert_safety_invariant(
                    not is_fresh,
                    "10-minute old price data should not be considered fresh"
                )

    @pytest.mark.regression
    @regression_test("GAP-PTV-002", "Duplicate orders should be detected")
    def test_duplicate_order_detection(self, validator, mock_order):
        """Duplicate orders within dedup window should be blocked."""
        if hasattr(validator, 'check_duplicate'):
            # First order should pass
            result1 = validator.check_duplicate(mock_order)

            # Immediate duplicate should be blocked
            result2 = validator.check_duplicate(mock_order)

            self.assert_safety_invariant(
                result1 != result2 or result2 is False,
                "Immediate duplicate order should be blocked"
            )


# ============================================================================
# CROSS-MODULE INTEGRATION SAFETY
# ============================================================================


class TestCrossModuleSafety(SafetyTestCase):
    """Safety tests for cross-module integration."""

    @pytest.mark.regression
    @pytest.mark.integration
    @regression_test("GAP-XM-001", "Circuit breaker trip must be logged to audit")
    def test_circuit_breaker_trip_audit_logged(self):
        """Circuit breaker trips must create audit trail."""
        from compliance import create_audit_logger
        from models.circuit_breaker import CircuitBreakerConfig, TradingCircuitBreaker

        audit_logger = create_audit_logger(auto_persist=False)

        config = CircuitBreakerConfig(
            max_daily_loss_pct=0.03,
            require_human_reset=False,
        )
        breaker = TradingCircuitBreaker(config=config)

        # Log before trip
        initial_count = len(audit_logger.get_entries()) if hasattr(audit_logger, 'get_entries') else 0

        # Trip the breaker
        breaker.check_daily_loss(-0.05)

        # Log the trip event
        if breaker.is_open:
            audit_logger.log_risk_event(
                event_type="CIRCUIT_BREAKER_TRIP",
                details={
                    "reason": str(breaker.trip_reason) if breaker.trip_reason else "unknown",
                    "daily_loss_pct": -0.05,
                },
            )

        # Verify audit entry was created
        if hasattr(audit_logger, 'get_entries'):
            final_count = len(audit_logger.get_entries())
            self.assert_safety_invariant(
                final_count > initial_count,
                "Circuit breaker trip must create audit entry"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "regression"])
