"""
Tests for TradingCircuitBreaker

These tests verify the circuit breaker safety mechanism works correctly
to halt trading when risk thresholds are breached.
"""

import tempfile
from pathlib import Path

import pytest

from models.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    TradingCircuitBreaker,
    TripReason,
    create_circuit_breaker,
)


@pytest.fixture
def breaker():
    """Create a circuit breaker with default config."""
    config = CircuitBreakerConfig(
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.10,
        max_consecutive_losses=5,
        require_human_reset=False,  # Disable for testing
    )
    return TradingCircuitBreaker(config=config)


@pytest.fixture
def breaker_human_reset():
    """Create a circuit breaker requiring human reset."""
    config = CircuitBreakerConfig(
        require_human_reset=True,
        cooldown_minutes=0,  # No cooldown for testing
    )
    return TradingCircuitBreaker(config=config)


class TestCircuitBreakerInitialization:
    """Tests for circuit breaker initialization."""

    def test_initial_state_is_closed(self, breaker):
        """Circuit breaker should start in closed (trading allowed) state."""
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.can_trade() is True
        assert breaker.is_open is False

    def test_create_circuit_breaker_convenience(self):
        """Test the convenience function creates breaker correctly."""
        breaker = create_circuit_breaker(
            max_daily_loss=0.05,
            max_drawdown=0.15,
            max_consecutive_losses=3,
        )
        assert breaker.config.max_daily_loss_pct == 0.05
        assert breaker.config.max_drawdown_pct == 0.15
        assert breaker.config.max_consecutive_losses == 3


class TestDailyLossCheck:
    """Tests for daily loss limit."""

    def test_within_daily_loss_limit(self, breaker):
        """Trading allowed when within daily loss limit."""
        assert breaker.check_daily_loss(-0.02) is True
        assert breaker.can_trade() is True

    def test_exceeds_daily_loss_limit(self, breaker):
        """Trading halted when daily loss limit exceeded."""
        assert breaker.check_daily_loss(-0.04) is False
        assert breaker.is_open is True
        assert breaker.can_trade() is False

    def test_exactly_at_daily_loss_limit(self, breaker):
        """Trading halted when exactly at limit."""
        assert breaker.check_daily_loss(-0.03) is False
        assert breaker.is_open is True


class TestDrawdownCheck:
    """Tests for max drawdown limit."""

    def test_within_drawdown_limit(self, breaker):
        """Trading allowed when within drawdown limit."""
        assert breaker.check_drawdown(95000, 100000) is True  # 5% drawdown
        assert breaker.can_trade() is True

    def test_exceeds_drawdown_limit(self, breaker):
        """Trading halted when drawdown limit exceeded."""
        assert breaker.check_drawdown(85000, 100000) is False  # 15% drawdown
        assert breaker.is_open is True
        assert breaker.can_trade() is False

    def test_zero_peak_equity(self, breaker):
        """Handle zero peak equity gracefully."""
        assert breaker.check_drawdown(0, 0) is True


class TestConsecutiveLosses:
    """Tests for consecutive losing trades."""

    def test_winning_trade_resets_counter(self, breaker):
        """Winning trade resets consecutive loss counter."""
        breaker.record_trade_result(is_winner=False)
        breaker.record_trade_result(is_winner=False)
        breaker.record_trade_result(is_winner=True)
        assert breaker._consecutive_losses == 0

    def test_consecutive_losses_trips_breaker(self, breaker):
        """Circuit breaker trips after max consecutive losses."""
        for _ in range(5):
            breaker.record_trade_result(is_winner=False)

        assert breaker.is_open is True
        assert breaker.can_trade() is False

    def test_mixed_results_no_trip(self, breaker):
        """Mixed win/loss doesn't trip breaker."""
        for i in range(10):
            breaker.record_trade_result(is_winner=i % 2 == 0)

        assert breaker.can_trade() is True


class TestComprehensiveCheck:
    """Tests for the comprehensive check method."""

    def test_check_passes_good_portfolio(self, breaker):
        """Check passes for healthy portfolio."""
        portfolio = {
            "daily_pnl_pct": 0.01,
            "current_equity": 105000,
            "peak_equity": 100000,
        }
        can_trade, reason = breaker.check(portfolio)
        assert can_trade is True
        assert reason is None

    def test_check_fails_on_daily_loss(self, breaker):
        """Check fails when daily loss exceeded."""
        portfolio = {
            "daily_pnl_pct": -0.05,
            "current_equity": 95000,
            "peak_equity": 100000,
        }
        can_trade, reason = breaker.check(portfolio)
        assert can_trade is False
        assert "Daily loss" in reason

    def test_check_fails_on_drawdown(self, breaker):
        """Check fails when drawdown exceeded."""
        portfolio = {
            "daily_pnl_pct": -0.01,
            "current_equity": 85000,
            "peak_equity": 100000,
        }
        can_trade, reason = breaker.check(portfolio)
        assert can_trade is False
        assert "drawdown" in reason.lower()


class TestManualHalt:
    """Tests for manual halt functionality."""

    def test_manual_halt_stops_trading(self, breaker):
        """Manual halt stops all trading."""
        breaker.halt_all_trading("Testing manual halt")
        assert breaker.is_open is True
        assert breaker.can_trade() is False

    def test_manual_halt_sets_reason(self, breaker):
        """Manual halt sets correct trip reason."""
        breaker.halt_all_trading("Emergency stop")
        assert breaker._trip_reason == TripReason.MANUAL_HALT


class TestReset:
    """Tests for circuit breaker reset."""

    def test_reset_restores_trading(self, breaker):
        """Reset allows trading to resume."""
        breaker.halt_all_trading("Test halt")
        assert breaker.can_trade() is False

        success = breaker.reset(authorized_by="test@example.com")
        assert success is True
        assert breaker.can_trade() is True

    def test_reset_clears_consecutive_losses(self, breaker):
        """Reset clears consecutive loss counter."""
        for _ in range(3):
            breaker.record_trade_result(is_winner=False)

        breaker.reset(authorized_by="test@example.com")
        assert breaker._consecutive_losses == 0

    def test_human_reset_required(self, breaker_human_reset):
        """Human reset requirement is enforced."""
        breaker_human_reset.halt_all_trading("Test")
        # Should succeed since cooldown is 0
        success = breaker_human_reset.reset(authorized_by="admin@example.com")
        assert success is True


class TestStatus:
    """Tests for status reporting."""

    def test_get_status_closed(self, breaker):
        """Status correctly reports closed state."""
        status = breaker.get_status()
        assert status["state"] == "closed"
        assert status["can_trade"] is True
        assert status["trip_reason"] is None

    def test_get_status_open(self, breaker):
        """Status correctly reports open (tripped) state."""
        breaker.halt_all_trading("Test")
        status = breaker.get_status()
        assert status["state"] == "open"
        assert status["can_trade"] is False
        assert status["trip_reason"] == "manual_halt"


class TestAlertCallback:
    """Tests for alert callback functionality."""

    def test_alert_callback_called_on_trip(self):
        """Alert callback is called when breaker trips."""
        alerts = []

        def callback(message, details):
            alerts.append((message, details))

        breaker = TradingCircuitBreaker(alert_callback=callback)
        breaker.halt_all_trading("Test alert")

        assert len(alerts) == 1
        assert "Trading halted" in alerts[0][0]

    def test_alert_human_calls_callback(self):
        """alert_human method calls the callback."""
        alerts = []

        def callback(message, details):
            alerts.append((message, details))

        breaker = TradingCircuitBreaker(alert_callback=callback)
        breaker.alert_human("Human intervention needed!")

        assert len(alerts) == 1
        assert "Human intervention" in alerts[0][0]


class TestLogging:
    """Tests for audit logging."""

    def test_log_file_created(self):
        """Log file is created when breaker trips."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test_log.json"
            breaker = TradingCircuitBreaker(log_file=log_file)
            breaker.halt_all_trading("Test logging")

            assert log_file.exists()

    def test_events_recorded(self, breaker):
        """Events are recorded in event history."""
        breaker.halt_all_trading("Event 1")
        breaker.reset(authorized_by="test")
        breaker.halt_all_trading("Event 2")

        assert len(breaker._events) == 3  # trip, reset, trip


@pytest.mark.unit
class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with risk manager."""

    def test_circuit_breaker_with_risk_manager(self):
        """Circuit breaker works with RiskManager data."""
        from models.risk_manager import RiskLimits, RiskManager

        # Set up risk manager
        limits = RiskLimits(max_daily_loss=0.03, max_drawdown=0.10)
        risk_mgr = RiskManager(starting_equity=100000, limits=limits)

        # Set up circuit breaker
        breaker = create_circuit_breaker(
            max_daily_loss=limits.max_daily_loss,
            max_drawdown=limits.max_drawdown,
        )

        # Simulate drawdown
        risk_mgr.update_equity(88000)  # 12% drawdown

        # Check circuit breaker with risk manager data
        can_trade = breaker.check_drawdown(risk_mgr.current_equity, risk_mgr.peak_equity)

        assert can_trade is False
        assert breaker.is_open is True


@pytest.mark.regression
class TestKillSwitchScenarios:
    """
    Kill switch scenario tests for circuit breaker.

    These tests verify the circuit breaker correctly handles
    critical trading safety scenarios.
    """

    def test_rapid_loss_triggers_immediate_halt(self):
        """
        Scenario: 3% loss in 1 minute triggers immediate halt.

        Expected: Circuit breaker trips immediately, no delay.
        """
        config = CircuitBreakerConfig(
            max_daily_loss_pct=0.03,
            require_human_reset=True,
        )
        breaker = TradingCircuitBreaker(config=config)

        # Simulate rapid loss (3.5% in one check)
        assert breaker.can_trade() is True

        result = breaker.check_daily_loss(-0.035)

        assert result is False
        assert breaker.is_open is True
        assert breaker.can_trade() is False
        assert breaker._trip_reason == TripReason.DAILY_LOSS

    def test_flash_crash_protection(self):
        """
        Scenario: 10% market drop triggers defensive mode.

        Expected: Drawdown limit triggers circuit breaker.
        """
        config = CircuitBreakerConfig(
            max_drawdown_pct=0.10,
            require_human_reset=True,
        )
        breaker = TradingCircuitBreaker(config=config)

        # Simulate flash crash - 15% drop from peak
        peak_equity = 100000
        crash_equity = 85000  # 15% drawdown

        result = breaker.check_drawdown(crash_equity, peak_equity)

        assert result is False
        assert breaker.is_open is True
        assert breaker._trip_reason == TripReason.MAX_DRAWDOWN

    def test_api_failure_triggers_halt(self):
        """
        Scenario: Broker API failure should trigger system error halt.

        Expected: Manual halt can be triggered for system errors.
        """
        breaker = TradingCircuitBreaker()

        # Simulate API failure detection
        breaker.halt_all_trading("Broker API connection lost")

        assert breaker.is_open is True
        assert breaker.can_trade() is False
        assert breaker._trip_reason == TripReason.MANUAL_HALT

    def test_manual_override_always_works(self):
        """
        Scenario: Human can halt trading at any time.

        Expected: Manual halt succeeds regardless of other states.
        """
        breaker = TradingCircuitBreaker()

        # Trading is normal
        assert breaker.can_trade() is True

        # Human triggers emergency halt
        breaker.halt_all_trading("Emergency: Unusual market conditions")

        assert breaker.is_open is True
        assert breaker.can_trade() is False

    def test_cascading_failures_handled(self):
        """
        Scenario: Multiple failure conditions occur simultaneously.

        Expected: First trigger halts trading, subsequent checks
        don't cause errors.
        """
        config = CircuitBreakerConfig(
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.10,
            max_consecutive_losses=5,
        )
        breaker = TradingCircuitBreaker(config=config)

        # Trigger multiple conditions
        breaker.check_daily_loss(-0.05)  # First trigger
        breaker.check_drawdown(85000, 100000)  # Already halted
        for _ in range(6):
            breaker.record_trade_result(is_winner=False)  # Already halted

        # Should still be halted, first reason preserved
        assert breaker.is_open is True
        assert breaker._trip_reason == TripReason.DAILY_LOSS

    def test_reset_requires_cooldown(self):
        """
        Scenario: Prevent premature reset during volatile conditions.

        Expected: Reset blocked during cooldown period.
        """
        config = CircuitBreakerConfig(
            max_daily_loss_pct=0.03,
            cooldown_minutes=30,
            require_human_reset=True,
        )
        breaker = TradingCircuitBreaker(config=config)

        # Trip the breaker
        breaker.check_daily_loss(-0.05)
        assert breaker.is_open is True

        # Attempt immediate reset (should fail due to cooldown)
        result = breaker.reset(authorized_by="trader@example.com")
        assert result is False
        assert breaker.is_open is True

    def test_consecutive_loss_streak_detection(self):
        """
        Scenario: Detect losing streak before catastrophic loss.

        Expected: 5 consecutive losses triggers halt.
        """
        config = CircuitBreakerConfig(
            max_consecutive_losses=5,
            require_human_reset=False,
        )
        breaker = TradingCircuitBreaker(config=config)

        # Record 4 losses - should still trade
        for _ in range(4):
            breaker.record_trade_result(is_winner=False)
        assert breaker.can_trade() is True
        assert breaker._consecutive_losses == 4

        # 5th loss triggers halt
        breaker.record_trade_result(is_winner=False)
        assert breaker.is_open is True
        assert breaker.can_trade() is False
        assert breaker._trip_reason == TripReason.CONSECUTIVE_LOSSES

    def test_winning_trade_breaks_loss_streak(self):
        """
        Scenario: Single win resets consecutive loss counter.

        Expected: Counter resets to 0 after winning trade.
        """
        config = CircuitBreakerConfig(
            max_consecutive_losses=5,
        )
        breaker = TradingCircuitBreaker(config=config)

        # Record 4 losses
        for _ in range(4):
            breaker.record_trade_result(is_winner=False)
        assert breaker._consecutive_losses == 4

        # One win resets counter
        breaker.record_trade_result(is_winner=True)
        assert breaker._consecutive_losses == 0
        assert breaker.can_trade() is True

    def test_alert_callback_fires_on_trip(self):
        """
        Scenario: External systems notified when circuit breaker trips.

        Expected: Alert callback called with trip details.
        """
        alerts_received = []

        def alert_handler(message, details):
            alerts_received.append({"message": message, "details": details})

        breaker = TradingCircuitBreaker(alert_callback=alert_handler)

        breaker.check_daily_loss(-0.05)

        assert len(alerts_received) == 1
        assert "Trading halted" in alerts_received[0]["message"]
        assert "daily_loss" in alerts_received[0]["details"]["reason"]

    def test_status_provides_full_context(self):
        """
        Scenario: Status endpoint provides all information for monitoring.

        Expected: Status includes state, reason, config, and timing.
        """
        config = CircuitBreakerConfig(
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.10,
            max_consecutive_losses=5,
            require_human_reset=True,
        )
        breaker = TradingCircuitBreaker(config=config)

        # Trip the breaker
        breaker.check_daily_loss(-0.05)

        status = breaker.get_status()

        # Verify all status fields
        assert status["state"] == "open"
        assert status["can_trade"] is False
        assert status["trip_reason"] == "daily_loss_exceeded"
        assert status["trip_time"] is not None
        assert status["config"]["max_daily_loss_pct"] == 0.03
        assert status["config"]["max_drawdown_pct"] == 0.10
        assert status["config"]["max_consecutive_losses"] == 5
        assert status["config"]["require_human_reset"] is True

    def test_event_history_preserved(self):
        """
        Scenario: All circuit breaker events logged for audit.

        Expected: Event history contains all trips and resets.
        """
        config = CircuitBreakerConfig(
            require_human_reset=False,
            cooldown_minutes=0,
        )
        breaker = TradingCircuitBreaker(config=config)

        # Trip and reset multiple times
        breaker.halt_all_trading("Test halt 1")
        breaker.reset(authorized_by="admin1")
        breaker.halt_all_trading("Test halt 2")
        breaker.reset(authorized_by="admin2")

        # Check event history
        assert len(breaker._events) == 4
        assert breaker._events[0].reason == TripReason.MANUAL_HALT
        assert breaker._events[1].resolved is True
        assert breaker._events[2].reason == TripReason.MANUAL_HALT
        assert breaker._events[3].resolved_by == "admin2"

    def test_partial_portfolio_data_handled(self):
        """
        Scenario: Portfolio data may be incomplete during market open.

        Expected: Handle missing or partial data gracefully.
        """
        breaker = TradingCircuitBreaker()

        # Partial portfolio data
        portfolio_partial = {
            "daily_pnl_pct": 0.01,
            # Missing current_equity and peak_equity
        }

        can_trade, reason = breaker.check(portfolio_partial)

        # Should not crash, should allow trading with partial data
        assert can_trade is True
        assert reason is None

    def test_zero_equity_edge_case(self):
        """
        Scenario: Edge case where peak equity is zero.

        Expected: Avoid division by zero, allow trading.
        """
        breaker = TradingCircuitBreaker()

        # Zero peak equity (startup condition)
        result = breaker.check_drawdown(current_equity=0, peak_equity=0)

        assert result is True
        assert breaker.can_trade() is True

    def test_negative_pnl_percentage_format(self):
        """
        Scenario: Ensure negative percentage handled correctly.

        Expected: Negative values trigger correctly.
        """
        config = CircuitBreakerConfig(max_daily_loss_pct=0.03)
        breaker = TradingCircuitBreaker(config=config)

        # Test exactly at threshold (should trigger)
        result = breaker.check_daily_loss(-0.03)
        assert result is False

        # Reset for next test
        breaker._state = CircuitBreakerState.CLOSED

        # Test just under threshold (should pass)
        result = breaker.check_daily_loss(-0.029)
        assert result is True
