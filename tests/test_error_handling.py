"""
Tests for Error Handling Infrastructure (UPGRADE-012)

Tests error handler, retry handler, and service health tracking.
"""

from unittest.mock import MagicMock

import pytest

from models.error_handler import (
    AlertTrigger,
    ErrorAggregation,
    ErrorCategory,
    ErrorHandler,
    ErrorSeverity,
    ServiceStatus,
    TradingError,
    create_error_handler,
)
from models.retry_handler import (
    RetryHandler,
    calculate_delay,
    create_retry_handler,
    is_retryable_exception,
    retry_with_backoff,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def error_handler():
    """Create an error handler for testing."""
    return ErrorHandler()


@pytest.fixture
def retry_handler():
    """Create a retry handler for testing."""
    return RetryHandler(max_retries=3, base_delay=0.01, max_delay=0.1)


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    logger = MagicMock()
    logger.log = MagicMock()
    return logger


# ==============================================================================
# Test TradingError
# ==============================================================================


class TestTradingError:
    """Test TradingError dataclass."""

    def test_create_trading_error(self):
        """Test creating a trading error."""
        error = TradingError(
            category=ErrorCategory.TRANSIENT,
            severity=ErrorSeverity.ERROR,
            message="Connection timeout",
        )
        assert error.category == ErrorCategory.TRANSIENT
        assert error.severity == ErrorSeverity.ERROR
        assert error.message == "Connection timeout"
        assert error.recoverable is True  # TRANSIENT is recoverable

    def test_from_exception(self):
        """Test creating from exception."""
        try:
            raise ValueError("Invalid value")
        except Exception as e:
            error = TradingError.from_exception(
                e,
                category=ErrorCategory.DATA,
                context={"field": "price"},
            )

        assert error.exception_type == "ValueError"
        assert error.exception_message == "Invalid value"
        assert error.context == {"field": "price"}
        assert "ValueError" in error.stack_trace

    def test_to_dict(self):
        """Test converting to dictionary."""
        error = TradingError(
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.WARNING,
            message="Order rejected",
            component="execution",
            operation="submit_order",
        )
        data = error.to_dict()

        assert data["category"] == "execution"
        assert data["severity"] == "warning"
        assert data["message"] == "Order rejected"
        assert data["component"] == "execution"


# ==============================================================================
# Test ErrorHandler
# ==============================================================================


class TestErrorHandler:
    """Test ErrorHandler class."""

    def test_handler_creation(self, error_handler):
        """Test error handler creation."""
        assert error_handler is not None

    def test_handle_error(self, error_handler):
        """Test handling an exception."""
        try:
            raise ConnectionError("Network timeout")
        except Exception as e:
            error = error_handler.handle_error(
                e,
                category=ErrorCategory.TRANSIENT,
                context={"endpoint": "/api/orders"},
            )

        assert error.category == ErrorCategory.TRANSIENT
        assert "Network timeout" in error.message

    def test_auto_classify_transient(self, error_handler):
        """Test auto-classification of transient errors."""
        try:
            raise TimeoutError("Connection timed out")
        except Exception as e:
            error = error_handler.handle_error(e)

        assert error.category == ErrorCategory.TRANSIENT

    def test_auto_classify_data(self, error_handler):
        """Test auto-classification of data errors."""
        try:
            raise ValueError("Invalid format")
        except Exception as e:
            error = error_handler.handle_error(e)

        # ValueError with "invalid" or "format" in message goes to DATA
        assert error.category == ErrorCategory.DATA

    def test_error_history(self, error_handler):
        """Test error history tracking."""
        for i in range(5):
            try:
                raise Exception(f"Error {i}")
            except Exception as e:
                error_handler.handle_error(e)

        errors = error_handler.get_recent_errors(limit=3)
        assert len(errors) == 3
        assert "Error 4" in errors[-1].message

    def test_error_counts(self, error_handler):
        """Test error counting by category."""
        error_handler.handle_error(Exception("Test"), category=ErrorCategory.TRANSIENT)
        error_handler.handle_error(Exception("Test"), category=ErrorCategory.TRANSIENT)
        error_handler.handle_error(Exception("Test"), category=ErrorCategory.PERMANENT)

        counts = error_handler.get_error_counts()
        assert counts["transient"] == 2
        assert counts["permanent"] == 1

    def test_error_listener(self, error_handler):
        """Test error event listener."""
        received_errors = []

        def listener(error):
            received_errors.append(error)

        error_handler.add_listener(listener)
        error_handler.handle_error(Exception("Test error"))

        assert len(received_errors) == 1
        assert "Test error" in received_errors[0].message

    def test_remove_listener(self, error_handler):
        """Test removing listener."""
        received = []

        def listener(e):
            received.append(e)

        error_handler.add_listener(listener)
        error_handler.remove_listener(listener)
        error_handler.handle_error(Exception("Test"))

        assert len(received) == 0


# ==============================================================================
# Test Service Health
# ==============================================================================


class TestServiceHealth:
    """Test service health tracking."""

    def test_register_service(self, error_handler):
        """Test registering a service."""
        error_handler.register_service("broker_api")
        health = error_handler.get_service_health("broker_api")

        assert health is not None
        assert health.name == "broker_api"
        assert health.status == ServiceStatus.HEALTHY

    def test_mark_degraded(self, error_handler):
        """Test marking service as degraded."""
        error_handler.register_service("data_feed")
        error_handler.mark_service_degraded("data_feed")

        health = error_handler.get_service_health("data_feed")
        assert health.status == ServiceStatus.DEGRADED
        assert health.is_available() is True  # Degraded is still available

    def test_mark_failed(self, error_handler):
        """Test marking service as failed."""
        error_handler.register_service("llm_service")
        error_handler.mark_service_failed("llm_service")

        health = error_handler.get_service_health("llm_service")
        assert health.status == ServiceStatus.FAILED
        assert health.is_available() is False

    def test_mark_healthy(self, error_handler):
        """Test recovering a service."""
        error_handler.register_service("api")
        error_handler.mark_service_failed("api")
        error_handler.mark_service_healthy("api")

        health = error_handler.get_service_health("api")
        assert health.status == ServiceStatus.HEALTHY

    def test_get_degraded_services(self, error_handler):
        """Test getting list of degraded services."""
        error_handler.register_service("service1")
        error_handler.register_service("service2")
        error_handler.register_service("service3")

        error_handler.mark_service_degraded("service1")
        error_handler.mark_service_failed("service2")

        degraded = error_handler.get_degraded_services()
        assert "service1" in degraded
        assert "service2" in degraded
        assert "service3" not in degraded


# ==============================================================================
# Test RetryHandler
# ==============================================================================


class TestRetryHandler:
    """Test RetryHandler class."""

    def test_successful_execution(self, retry_handler):
        """Test successful execution without retry."""
        result = retry_handler.execute(lambda: 42)
        assert result == 42

    def test_retry_on_failure(self, retry_handler):
        """Test retry on transient failure."""
        counter = [0]

        def flaky_operation():
            counter[0] += 1
            if counter[0] < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = retry_handler.execute(
            flaky_operation,
            exceptions=(ConnectionError,),
        )
        assert result == "success"
        assert counter[0] == 3

    def test_max_retries_exceeded(self, retry_handler):
        """Test exception when max retries exceeded."""

        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            retry_handler.execute(
                always_fails,
                exceptions=(ValueError,),
            )

    def test_retry_stats(self, retry_handler):
        """Test retry statistics."""
        counter = [0]

        def flaky():
            counter[0] += 1
            if counter[0] < 2:
                raise Exception("Fail")
            return True

        retry_handler.execute(flaky)
        stats = retry_handler.get_stats()

        assert stats["total_retries"] >= 1
        assert stats["successful_retries"] >= 1


# ==============================================================================
# Test retry_with_backoff Decorator
# ==============================================================================


class TestRetryDecorator:
    """Test retry_with_backoff decorator."""

    def test_decorator_success(self):
        """Test decorator with successful function."""

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def simple_function():
            return "success"

        assert simple_function() == "success"

    def test_decorator_retry(self):
        """Test decorator with retry."""
        counter = [0]

        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            exceptions=(ValueError,),
        )
        def flaky_function():
            counter[0] += 1
            if counter[0] < 2:
                raise ValueError("Temporary")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert counter[0] == 2

    def test_decorator_callback(self):
        """Test decorator with retry callback."""
        retries = []

        @retry_with_backoff(
            max_retries=2,
            base_delay=0.01,
            on_retry=lambda e, n: retries.append(n),
        )
        def always_fails():
            raise Exception("Fail")

        with pytest.raises(Exception):
            always_fails()

        assert len(retries) == 2
        assert retries == [1, 2]


# ==============================================================================
# Test Utility Functions
# ==============================================================================


class TestUtilityFunctions:
    """Test utility functions."""

    def test_calculate_delay(self):
        """Test delay calculation."""
        # First attempt
        delay0 = calculate_delay(0, base_delay=1.0, jitter=False)
        assert delay0 == 1.0

        # Second attempt (exponential)
        delay1 = calculate_delay(1, base_delay=1.0, jitter=False)
        assert delay1 == 2.0

        # Third attempt
        delay2 = calculate_delay(2, base_delay=1.0, jitter=False)
        assert delay2 == 4.0

    def test_calculate_delay_max(self):
        """Test max delay cap."""
        delay = calculate_delay(10, base_delay=1.0, max_delay=10.0, jitter=False)
        assert delay == 10.0

    def test_calculate_delay_jitter(self):
        """Test jitter adds variation."""
        delays = [calculate_delay(0, base_delay=1.0, jitter=True) for _ in range(10)]
        # With jitter, delays should vary
        assert len(set(delays)) > 1

    def test_is_retryable_exception(self):
        """Test exception retryability check."""
        # Timeout errors are retryable
        assert is_retryable_exception(TimeoutError("timed out")) is True

        # Connection errors are retryable
        assert is_retryable_exception(ConnectionError("connection")) is True

        # Generic errors checked by message
        assert is_retryable_exception(Exception("rate limit exceeded")) is True


# ==============================================================================
# Test Factory Functions
# ==============================================================================


class TestErrorHandlingFactoryFunctions:
    """Test factory functions."""

    def test_create_error_handler(self):
        """Test error handler factory."""
        handler = create_error_handler(max_history=500)
        assert isinstance(handler, ErrorHandler)
        assert handler.max_history == 500

    def test_create_retry_handler(self):
        """Test retry handler factory."""
        handler = create_retry_handler(max_retries=5, base_delay=2.0)
        assert isinstance(handler, RetryHandler)
        assert handler.max_retries == 5
        assert handler.base_delay == 2.0


# ==============================================================================
# Test Integration
# ==============================================================================


class TestErrorHandlingIntegration:
    """Test error handler integration with logger."""

    def test_logger_integration(self, mock_logger):
        """Test error handler logs to structured logger."""
        handler = ErrorHandler(logger=mock_logger)

        try:
            raise ValueError("Test error")
        except Exception as e:
            handler.handle_error(e)

        # Logger should have been called
        assert mock_logger.log.called

    def test_error_rate_calculation(self, error_handler):
        """Test error rate calculation."""
        # Add some errors
        for _ in range(5):
            error_handler.handle_error(Exception("Test"))

        rate = error_handler.get_error_rate(window_minutes=5)
        assert rate == 1.0  # 5 errors / 5 minutes


# ==============================================================================
# Test Error Classification
# ==============================================================================


class TestErrorClassification:
    """Test error classification enums."""

    def test_error_severity_values(self):
        """Test error severity enum values."""
        assert ErrorSeverity.DEBUG.value == "debug"
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_error_category_values(self):
        """Test error category enum values."""
        assert ErrorCategory.TRANSIENT.value == "transient"
        assert ErrorCategory.PERMANENT.value == "permanent"
        assert ErrorCategory.RECOVERABLE.value == "recoverable"

    def test_service_status_values(self):
        """Test service status enum values."""
        assert ServiceStatus.HEALTHY.value == "healthy"
        assert ServiceStatus.DEGRADED.value == "degraded"
        assert ServiceStatus.FAILED.value == "failed"


# ==============================================================================
# Test ErrorAggregation (UPGRADE-012 expansion)
# ==============================================================================


class TestErrorAggregation:
    """Test ErrorAggregation dataclass."""

    def test_create_aggregation(self):
        """Test creating an error aggregation."""
        agg = ErrorAggregation(
            key="transient:broker:order",
            category=ErrorCategory.TRANSIENT,
            severity=ErrorSeverity.WARNING,
            component="broker",
            operation="order",
        )
        assert agg.key == "transient:broker:order"
        assert agg.count == 0
        assert agg.alerted is False

    def test_add_error(self):
        """Test adding errors to aggregation."""
        agg = ErrorAggregation(
            key="test",
            category=ErrorCategory.TRANSIENT,
            severity=ErrorSeverity.WARNING,
        )

        error = TradingError(
            category=ErrorCategory.TRANSIENT,
            severity=ErrorSeverity.WARNING,
            message="Test error",
        )

        agg.add_error(error)
        assert agg.count == 1
        assert len(agg.sample_errors) == 1

        # Add more errors
        for _ in range(10):
            agg.add_error(error)

        assert agg.count == 11
        assert len(agg.sample_errors) == 5  # max_samples is 5

    def test_to_dict(self):
        """Test converting aggregation to dict."""
        agg = ErrorAggregation(
            key="test",
            category=ErrorCategory.TRANSIENT,
            severity=ErrorSeverity.WARNING,
            component="broker",
            operation="order",
        )
        data = agg.to_dict()

        assert data["key"] == "test"
        assert data["category"] == "transient"
        assert data["severity"] == "warning"
        assert "duration_seconds" in data


# ==============================================================================
# Test AlertTrigger (UPGRADE-012 expansion)
# ==============================================================================


class TestAlertTrigger:
    """Test AlertTrigger dataclass."""

    def test_create_trigger(self):
        """Test creating an alert trigger."""
        trigger = AlertTrigger(
            name="test_alert",
            condition="spike",
            threshold=10.0,
        )
        assert trigger.name == "test_alert"
        assert trigger.enabled is True
        assert trigger.trigger_count == 0

    def test_can_trigger(self):
        """Test trigger cooldown check."""
        trigger = AlertTrigger(
            name="test",
            condition="spike",
            cooldown_minutes=5,
        )

        # Should be able to trigger initially
        assert trigger.can_trigger() is True

        # After triggering, should not be able to trigger again
        trigger.trigger()
        assert trigger.can_trigger() is False
        assert trigger.trigger_count == 1

    def test_disabled_trigger(self):
        """Test disabled trigger cannot fire."""
        trigger = AlertTrigger(
            name="test",
            condition="spike",
            enabled=False,
        )
        assert trigger.can_trigger() is False


# ==============================================================================
# Test Error Aggregation Methods (UPGRADE-012 expansion)
# ==============================================================================


class TestErrorAggregationMethods:
    """Test ErrorHandler aggregation methods."""

    def test_aggregate_error(self, error_handler):
        """Test aggregating an error."""
        try:
            raise ConnectionError("Test connection error")
        except Exception as e:
            error = error_handler.handle_error(
                e,
                component="broker",
                operation="connect",
            )

        agg = error_handler.aggregate_error(error)
        assert agg is not None
        assert agg.count == 1

    def test_multiple_aggregations(self, error_handler):
        """Test multiple error aggregations."""
        # Add errors with different keys
        for i in range(5):
            try:
                raise ValueError(f"Error {i}")
            except Exception as e:
                error = error_handler.handle_error(
                    e,
                    component="scanner",
                    operation="scan",
                )
                error_handler.aggregate_error(error)

        aggregations = error_handler.get_aggregations()
        assert len(aggregations) >= 1

    def test_get_spike_candidates(self, error_handler):
        """Test getting spike candidates."""
        # Add many errors to trigger spike
        for i in range(15):
            try:
                raise TimeoutError(f"Timeout {i}")
            except Exception as e:
                error = error_handler.handle_error(
                    e,
                    component="api",
                    operation="call",
                )
                error_handler.aggregate_error(error)

        candidates = error_handler.get_spike_candidates(threshold=5)
        assert len(candidates) >= 1

    def test_check_error_spike(self, error_handler):
        """Test checking for error spikes."""
        # Initially no spike
        assert error_handler.check_error_spike(threshold=100) is False

        # Add many errors
        for i in range(20):
            try:
                raise Exception(f"Error {i}")
            except Exception as e:
                error_handler.handle_error(e)

        # Should detect spike now
        assert error_handler.check_error_spike(threshold=5) is True


# ==============================================================================
# Test Alert Management (UPGRADE-012 expansion)
# ==============================================================================


class TestAlertManagement:
    """Test ErrorHandler alert management."""

    def test_add_alert_listener(self, error_handler):
        """Test adding alert listener."""
        alerts_received = []

        def listener(name, data):
            alerts_received.append((name, data))

        error_handler.add_alert_listener(listener)
        error_handler.trigger_manual_alert("test", "Test message")

        assert len(alerts_received) == 1
        assert alerts_received[0][0] == "test"

    def test_remove_alert_listener(self, error_handler):
        """Test removing alert listener."""
        alerts_received = []

        def listener(name, data):
            alerts_received.append((name, data))

        error_handler.add_alert_listener(listener)
        error_handler.remove_alert_listener(listener)
        error_handler.trigger_manual_alert("test", "Test message")

        assert len(alerts_received) == 0

    def test_configure_alert(self, error_handler):
        """Test configuring alerts."""
        error_handler.configure_alert(
            "custom_alert",
            enabled=True,
            threshold=5.0,
            window_minutes=10,
        )

        status = error_handler.get_alert_status()
        assert "custom_alert" in status
        assert status["custom_alert"]["threshold"] == 5.0

    def test_get_alert_status(self, error_handler):
        """Test getting alert status."""
        status = error_handler.get_alert_status()

        # Should have default alerts
        assert "error_spike" in status
        assert "critical_error" in status
        assert "service_degraded" in status

    def test_trigger_manual_alert(self, error_handler):
        """Test triggering manual alert."""
        result = error_handler.trigger_manual_alert(
            "manual_test",
            "Manual test message",
            data={"extra": "data"},
        )
        assert result is True

        # Second trigger should fail (cooldown)
        result = error_handler.trigger_manual_alert(
            "manual_test",
            "Another message",
        )
        assert result is False

    def test_alert_cooldown(self, error_handler):
        """Test alert cooldown behavior."""
        # Configure alert with 0 cooldown for testing
        error_handler.configure_alert(
            "no_cooldown",
            cooldown_minutes=0,
        )

        # Should be able to trigger multiple times
        result1 = error_handler.trigger_manual_alert("no_cooldown", "First")
        assert result1 is True

        # With 0 cooldown, should still work
        result2 = error_handler.trigger_manual_alert("no_cooldown", "Second")
        assert result2 is True
