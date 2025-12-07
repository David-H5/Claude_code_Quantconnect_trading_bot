"""
Tests for Structured Logging Infrastructure

Tests:
- LogEvent creation and serialization
- StructuredLogger functionality
- Log handlers (rotating, async, callback)
- Context propagation
- Correlation ID tracking

UPGRADE-009: Structured Logging (December 2025)
"""

import json
import logging
import threading
import time
from unittest.mock import MagicMock

import pytest

from utils.log_handlers import (
    AsyncHandler,
    CallbackHandler,
    CompressedRotatingFileHandler,
    create_rotating_file_handler,
)
from observability.logging.structured import (
    ExecutionEventType,
    LogCategory,
    LogEvent,
    LogLevel,
    RiskEventType,
    StructuredLogger,
    create_structured_logger,
    get_logger,
    set_logger,
)


# ============================================================================
# LOG EVENT TESTS
# ============================================================================


class TestLogEvent:
    """Tests for LogEvent dataclass."""

    def test_log_event_creation(self):
        """Test basic LogEvent creation."""
        event = LogEvent(
            category=LogCategory.EXECUTION,
            level=LogLevel.INFO,
            event_type="order_submitted",
            message="Order submitted for SPY",
        )

        assert event.category == LogCategory.EXECUTION
        assert event.level == LogLevel.INFO
        assert event.event_type == "order_submitted"
        assert event.message == "Order submitted for SPY"
        assert event.event_id is not None
        assert event.timestamp is not None

    def test_log_event_to_dict(self):
        """Test LogEvent to_dict conversion."""
        event = LogEvent(
            category=LogCategory.RISK,
            level=LogLevel.WARNING,
            event_type="risk_warning",
            message="Drawdown warning",
            data={"drawdown": 0.15},
            context={"algorithm": "test"},
        )

        d = event.to_dict()

        assert d["category"] == "risk"
        assert d["level"] == "warning"
        assert d["event_type"] == "risk_warning"
        assert d["data"]["drawdown"] == 0.15
        assert d["context"]["algorithm"] == "test"

    def test_log_event_to_json(self):
        """Test LogEvent to_json conversion."""
        event = LogEvent(
            category=LogCategory.STRATEGY,
            level=LogLevel.INFO,
            event_type="signal",
            message="Buy signal for AAPL",
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["category"] == "strategy"
        assert parsed["event_type"] == "signal"
        assert "timestamp" in parsed

    def test_log_event_with_correlation_id(self):
        """Test LogEvent with correlation ID."""
        event = LogEvent(
            category=LogCategory.EXECUTION,
            level=LogLevel.INFO,
            event_type="order_filled",
            message="Order filled",
            correlation_id="REQ-12345",
        )

        d = event.to_dict()
        assert d["correlation_id"] == "REQ-12345"

    def test_log_event_with_duration(self):
        """Test LogEvent with duration."""
        event = LogEvent(
            category=LogCategory.SYSTEM,
            level=LogLevel.INFO,
            event_type="api_request",
            message="API request completed",
            duration_ms=150.5,
        )

        d = event.to_dict()
        assert d["duration_ms"] == 150.5


# ============================================================================
# STRUCTURED LOGGER TESTS
# ============================================================================


class TestStructuredLogger:
    """Tests for StructuredLogger class."""

    @pytest.fixture
    def logger(self):
        """Create a test logger."""
        return StructuredLogger(name="test_logger", min_level=LogLevel.DEBUG)

    def test_logger_creation(self, logger):
        """Test logger creation."""
        assert logger.name == "test_logger"
        assert logger.min_level == LogLevel.DEBUG

    def test_log_basic_event(self, logger):
        """Test logging a basic event."""
        event = logger.log(
            LogCategory.SYSTEM,
            LogLevel.INFO,
            "test_event",
            "Test message",
        )

        assert event.category == LogCategory.SYSTEM
        assert event.level == LogLevel.INFO
        assert event.message == "Test message"

    def test_log_with_data(self, logger):
        """Test logging with data."""
        event = logger.log(
            LogCategory.EXECUTION,
            LogLevel.INFO,
            "order_submitted",
            "Order submitted",
            data={"order_id": "ORD-001", "symbol": "SPY"},
        )

        assert event.data["order_id"] == "ORD-001"
        assert event.data["symbol"] == "SPY"

    def test_context_propagation(self, logger):
        """Test context propagation."""
        logger.set_context(algorithm="hybrid_bot", session="abc123")

        event = logger.log(
            LogCategory.SYSTEM,
            LogLevel.INFO,
            "test",
            "Test with context",
        )

        assert event.context["algorithm"] == "hybrid_bot"
        assert event.context["session"] == "abc123"

    def test_clear_context(self, logger):
        """Test context clearing."""
        logger.set_context(key="value")
        logger.clear_context()

        event = logger.log(
            LogCategory.SYSTEM,
            LogLevel.INFO,
            "test",
            "Test",
        )

        assert "key" not in event.context

    def test_correlation_id(self, logger):
        """Test correlation ID setting."""
        logger.set_correlation_id("CORR-001")

        event = logger.log(
            LogCategory.EXECUTION,
            LogLevel.INFO,
            "test",
            "Test",
        )

        assert event.correlation_id == "CORR-001"

    def test_correlation_scope(self, logger):
        """Test correlation scope context manager."""
        with logger.correlation_scope("SCOPE-001") as cid:
            event1 = logger.log(
                LogCategory.EXECUTION,
                LogLevel.INFO,
                "test1",
                "Test 1",
            )
            assert event1.correlation_id == "SCOPE-001"
            assert cid == "SCOPE-001"

        # After scope, correlation ID should be cleared
        event2 = logger.log(
            LogCategory.EXECUTION,
            LogLevel.INFO,
            "test2",
            "Test 2",
        )
        assert event2.correlation_id is None

    def test_event_listener(self, logger):
        """Test event listener."""
        events = []

        def listener(event):
            events.append(event)

        logger.add_listener(listener)

        logger.log(
            LogCategory.SYSTEM,
            LogLevel.INFO,
            "test",
            "Test",
        )

        assert len(events) == 1
        assert events[0].event_type == "test"

    def test_remove_listener(self, logger):
        """Test removing event listener."""
        events = []

        def listener(event):
            events.append(event)

        logger.add_listener(listener)
        logger.remove_listener(listener)

        logger.log(
            LogCategory.SYSTEM,
            LogLevel.INFO,
            "test",
            "Test",
        )

        assert len(events) == 0


# ============================================================================
# EXECUTION LOG METHODS TESTS
# ============================================================================


class TestExecutionLogMethods:
    """Tests for execution log convenience methods."""

    @pytest.fixture
    def logger(self):
        """Create a test logger."""
        return StructuredLogger(name="execution_test")

    def test_log_order_submitted(self, logger):
        """Test order submitted logging."""
        event = logger.log_order_submitted(
            order_id="ORD-001",
            symbol="SPY",
            side="buy",
            quantity=10,
            limit_price=450.00,
            strategy="iron_condor",
        )

        assert event.category == LogCategory.EXECUTION
        assert event.event_type == ExecutionEventType.ORDER_SUBMITTED.value
        assert event.data["order_id"] == "ORD-001"
        assert event.data["symbol"] == "SPY"
        assert event.data["quantity"] == 10

    def test_log_order_filled(self, logger):
        """Test order filled logging."""
        event = logger.log_order_filled(
            order_id="ORD-001",
            symbol="SPY",
            fill_price=450.50,
            quantity=10,
            slippage_bps=5.0,
        )

        assert event.event_type == ExecutionEventType.ORDER_FILLED.value
        assert event.data["fill_price"] == 450.50
        assert event.data["slippage_bps"] == 5.0

    def test_log_order_cancelled(self, logger):
        """Test order cancelled logging."""
        event = logger.log_order_cancelled(
            order_id="ORD-001",
            symbol="SPY",
            reason="Timeout after 2.5 seconds",
        )

        assert event.level == LogLevel.WARNING
        assert event.event_type == ExecutionEventType.ORDER_CANCELLED.value
        assert "Timeout" in event.data["reason"]

    def test_log_position_opened(self, logger):
        """Test position opened logging."""
        event = logger.log_position_opened(
            position_id="POS-001",
            symbol="SPY 450C",
            quantity=5,
            entry_price=12.50,
            strategy="bull_call_spread",
        )

        assert event.event_type == ExecutionEventType.POSITION_OPENED.value
        assert event.data["position_id"] == "POS-001"
        assert event.data["entry_price"] == 12.50

    def test_log_position_closed_profit(self, logger):
        """Test position closed with profit."""
        event = logger.log_position_closed(
            position_id="POS-001",
            symbol="SPY 450C",
            quantity=5,
            entry_price=12.50,
            exit_price=18.75,
            pnl=31.25,
            pnl_pct=0.50,
            exit_reason="profit_target",
        )

        assert event.level == LogLevel.INFO  # Profit
        assert event.data["pnl"] == 31.25
        assert event.data["pnl_pct"] == 0.50

    def test_log_position_closed_loss(self, logger):
        """Test position closed with loss."""
        event = logger.log_position_closed(
            position_id="POS-001",
            symbol="SPY 450C",
            quantity=5,
            entry_price=12.50,
            exit_price=8.00,
            pnl=-22.50,
            pnl_pct=-0.36,
            exit_reason="stop_loss",
        )

        assert event.level == LogLevel.WARNING  # Loss


# ============================================================================
# RISK LOG METHODS TESTS
# ============================================================================


class TestRiskLogMethods:
    """Tests for risk log convenience methods."""

    @pytest.fixture
    def logger(self):
        """Create a test logger."""
        return StructuredLogger(name="risk_test")

    def test_log_circuit_breaker_halt(self, logger):
        """Test circuit breaker halt logging."""
        event = logger.log_circuit_breaker(
            is_halted=True,
            reason="Daily loss exceeded 3%",
            metrics={"daily_loss_pct": 0.035},
        )

        assert event.level == LogLevel.CRITICAL
        assert event.event_type == RiskEventType.CIRCUIT_BREAKER_TRIGGERED.value
        assert event.data["is_halted"] is True

    def test_log_circuit_breaker_reset(self, logger):
        """Test circuit breaker reset logging."""
        event = logger.log_circuit_breaker(
            is_halted=False,
            reason="Manual reset by admin",
        )

        assert event.level == LogLevel.INFO
        assert event.event_type == RiskEventType.CIRCUIT_BREAKER_RESET.value

    def test_log_risk_warning(self, logger):
        """Test risk warning logging."""
        event = logger.log_risk_warning(
            warning_type="daily_loss_warning",
            current_value=0.025,
            threshold=0.03,
        )

        assert event.level == LogLevel.WARNING
        assert event.data["current_value"] == 0.025
        assert event.data["threshold"] == 0.03

    def test_log_risk_breach(self, logger):
        """Test risk breach logging."""
        event = logger.log_risk_breach(
            breach_type="position_limit_exceeded",
            current_value=0.30,
            threshold=0.25,
            action_taken="Order rejected",
        )

        assert event.level == LogLevel.ERROR
        assert "RISK BREACH" in event.message


# ============================================================================
# HANDLER TESTS
# ============================================================================


class TestCompressedRotatingFileHandler:
    """Tests for CompressedRotatingFileHandler."""

    def test_handler_creation(self, tmp_path):
        """Test handler creation."""
        log_file = tmp_path / "logs" / "test.log"
        handler = CompressedRotatingFileHandler(
            str(log_file),
            maxBytes=1024,
            backupCount=3,
        )

        assert handler.maxBytes == 1024
        assert handler.backupCount == 3
        handler.close()

    def test_handler_creates_directory(self, tmp_path):
        """Test handler creates parent directory."""
        log_file = tmp_path / "new_dir" / "nested" / "test.log"
        handler = CompressedRotatingFileHandler(str(log_file))

        assert log_file.parent.exists()
        handler.close()


class TestAsyncHandler:
    """Tests for AsyncHandler."""

    def test_async_handler_creation(self):
        """Test async handler creation."""
        mock_handler = MagicMock(spec=logging.Handler)
        async_handler = AsyncHandler(mock_handler, queue_size=100)

        assert async_handler.handler == mock_handler
        async_handler.close()

    def test_async_handler_emits(self):
        """Test async handler emits records."""
        mock_handler = MagicMock(spec=logging.Handler)
        async_handler = AsyncHandler(mock_handler, queue_size=100)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        async_handler.emit(record)

        # Give async thread time to process
        time.sleep(0.1)

        mock_handler.emit.assert_called()
        async_handler.close()


class TestCallbackHandler:
    """Tests for CallbackHandler."""

    def test_callback_handler_calls_callback(self):
        """Test callback handler calls callback."""
        events = []

        def callback(event):
            events.append(event)

        handler = CallbackHandler(callback)
        handler.setFormatter(logging.Formatter("%(message)s"))

        # Create a JSON log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='{"level": "info", "message": "test"}',
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        assert len(events) == 1
        assert events[0]["level"] == "info"

    def test_callback_handler_level_filter(self):
        """Test callback handler level filtering."""
        events = []

        def callback(event):
            events.append(event)

        handler = CallbackHandler(callback, include_levels=["error", "critical"])

        # Info level should be filtered
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        assert len(events) == 0


# ============================================================================
# FACTORY FUNCTION TESTS
# ============================================================================


class TestStructuredLoggingFactoryFunctions:
    """Tests for factory functions."""

    def test_create_structured_logger(self):
        """Test create_structured_logger factory."""
        logger = create_structured_logger(
            name="factory_test",
            console=True,
            min_level=LogLevel.DEBUG,
        )

        assert logger.name == "factory_test"
        assert logger.min_level == LogLevel.DEBUG

    def test_create_rotating_file_handler(self, tmp_path):
        """Test create_rotating_file_handler factory."""
        log_file = tmp_path / "test.log"
        handler = create_rotating_file_handler(
            str(log_file),
            max_bytes=1024,
            backup_count=5,
        )

        assert isinstance(handler, CompressedRotatingFileHandler)
        handler.close()

    def test_get_set_logger(self):
        """Test get_logger and set_logger."""
        original = get_logger()

        new_logger = create_structured_logger("new_logger")
        set_logger(new_logger)

        assert get_logger() == new_logger

        # Restore
        set_logger(original)


# ============================================================================
# THREAD SAFETY TESTS
# ============================================================================


class TestStructuredLoggingThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_logging(self):
        """Test logging from multiple threads."""
        logger = StructuredLogger(name="thread_test")
        events = []

        def listener(event):
            events.append(event)

        logger.add_listener(listener)

        def log_from_thread(thread_id):
            for i in range(10):
                logger.log(
                    LogCategory.SYSTEM,
                    LogLevel.INFO,
                    "thread_log",
                    f"Thread {thread_id} message {i}",
                    thread_id=thread_id,
                )

        threads = [threading.Thread(target=log_from_thread, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(events) == 50  # 5 threads * 10 messages

    def test_context_thread_isolation(self):
        """Test context is properly locked."""
        logger = StructuredLogger(name="context_test")

        def set_context_in_thread(value):
            logger.set_context(thread_value=value)
            time.sleep(0.01)
            logger.clear_context()

        threads = [threading.Thread(target=set_context_in_thread, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # After all threads complete, context should be empty
        event = logger.log(LogCategory.SYSTEM, LogLevel.INFO, "test", "Test")
        assert "thread_value" not in event.context
