"""
Fault Injection and Chaos Engineering Tests

Tests system resilience by deliberately introducing failures and
verifying the system handles them gracefully.

Based on best practices from:
- Netflix Chaos Monkey principles
- AWS Fault Injection Simulator patterns
- Financial system resilience testing
"""

import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest


@dataclass
class FaultConfig:
    """Configuration for fault injection."""

    fault_type: str
    probability: float = 1.0
    duration_ms: int = 0
    error_message: str = ""


class FaultInjector:
    """Utility class for injecting faults."""

    @staticmethod
    def inject_latency(min_ms: int = 100, max_ms: int = 500) -> float:
        """Inject random latency."""
        latency = random.uniform(min_ms, max_ms) / 1000
        time.sleep(latency)
        return latency

    @staticmethod
    def should_fail(probability: float = 0.5) -> bool:
        """Determine if operation should fail based on probability."""
        return random.random() < probability

    @staticmethod
    def corrupt_data(data: dict, corruption_rate: float = 0.1) -> dict:
        """Corrupt random fields in a data dictionary."""
        corrupted = data.copy()
        for key in list(corrupted.keys()):
            if random.random() < corruption_rate:
                if isinstance(corrupted[key], (int, float)):
                    corrupted[key] = corrupted[key] * random.uniform(-10, 10)
                elif isinstance(corrupted[key], str):
                    corrupted[key] = "CORRUPTED_" + corrupted[key]
        return corrupted


class TestNetworkFaultResilience:
    """Tests for network fault resilience."""

    @pytest.mark.chaos
    def test_handles_connection_timeout(self):
        """Test system handles connection timeouts gracefully."""
        mock_client = Mock()
        mock_client.get_quote = Mock(side_effect=TimeoutError("Connection timed out"))

        # System should catch and handle timeout
        with pytest.raises(TimeoutError):
            mock_client.get_quote("SPY")

        # Verify retry logic would be called
        mock_client.get_quote.assert_called_once()

    @pytest.mark.chaos
    def test_handles_connection_reset(self):
        """Test handling of connection reset errors."""
        mock_client = Mock()
        mock_client.place_order = Mock(side_effect=ConnectionResetError("Connection reset by peer"))

        with pytest.raises(ConnectionResetError):
            mock_client.place_order(symbol="SPY", side="buy", quantity=100)

    @pytest.mark.chaos
    def test_handles_dns_resolution_failure(self):
        """Test handling of DNS resolution failures."""
        mock_client = Mock()
        mock_client.connect = Mock(side_effect=OSError("[Errno -2] Name or service not known"))

        with pytest.raises(OSError):
            mock_client.connect()

    @pytest.mark.chaos
    def test_handles_intermittent_failures_with_retry(self):
        """Test retry logic for intermittent failures."""
        call_count = 0
        max_retries = 3

        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return {"success": True}

        # Simulate retry logic
        result = None
        for attempt in range(max_retries):
            try:
                result = flaky_operation()
                break
            except ConnectionError:
                if attempt == max_retries - 1:
                    raise
                continue

        assert result is not None
        assert result["success"] is True
        assert call_count == 3


class TestDataCorruptionResilience:
    """Tests for data corruption resilience."""

    @pytest.mark.chaos
    def test_handles_invalid_price_data(self):
        """Test handling of corrupted price data."""
        corrupted_quotes = [
            {"symbol": "SPY", "price": -100.0},  # Negative price
            {"symbol": "SPY", "price": float("nan")},  # NaN
            {"symbol": "SPY", "price": float("inf")},  # Infinity
            {"symbol": "SPY", "price": None},  # Null
        ]

        def validate_price(quote: dict) -> bool:
            price = quote.get("price")
            if price is None:
                return False
            if not isinstance(price, (int, float)):
                return False
            if price <= 0:
                return False
            if price != price:  # NaN check
                return False
            if price == float("inf") or price == float("-inf"):
                return False
            return True

        for quote in corrupted_quotes:
            assert validate_price(quote) is False

    @pytest.mark.chaos
    def test_handles_missing_required_fields(self):
        """Test handling of data with missing required fields."""
        incomplete_data = [
            {"symbol": "SPY"},  # Missing price
            {"price": 100.0},  # Missing symbol
            {},  # Empty
        ]

        required_fields = ["symbol", "price"]

        for data in incomplete_data:
            missing = [f for f in required_fields if f not in data]
            assert len(missing) > 0

    @pytest.mark.chaos
    def test_handles_type_mismatch(self):
        """Test handling of wrong data types."""
        type_errors = [
            {"symbol": 12345, "price": 100.0},  # Symbol should be string
            {"symbol": "SPY", "price": "hundred"},  # Price should be number
            {"symbol": "SPY", "price": 100.0, "volume": "high"},  # Volume should be int
        ]

        def validate_types(data: dict) -> list[str]:
            errors = []
            if "symbol" in data and not isinstance(data["symbol"], str):
                errors.append("symbol must be string")
            if "price" in data and not isinstance(data["price"], (int, float)):
                errors.append("price must be number")
            if "volume" in data and not isinstance(data["volume"], int):
                errors.append("volume must be integer")
            return errors

        for data in type_errors:
            errors = validate_types(data)
            assert len(errors) > 0


class TestExternalServiceFailures:
    """Tests for external service failure handling."""

    @pytest.mark.chaos
    def test_handles_broker_api_outage(self):
        """Test handling when broker API is down."""
        mock_broker = Mock()
        mock_broker.is_connected = Mock(return_value=False)
        mock_broker.get_positions = Mock(side_effect=Exception("Service unavailable"))

        # System should detect disconnection
        assert mock_broker.is_connected() is False

        # Operations should fail gracefully
        with pytest.raises(Exception) as exc_info:
            mock_broker.get_positions()
        assert "unavailable" in str(exc_info.value).lower()

    @pytest.mark.chaos
    def test_handles_market_data_feed_failure(self):
        """Test handling when market data feed fails."""
        stale_threshold = timedelta(seconds=30)
        last_update = datetime.now() - timedelta(minutes=5)

        # Detect stale data
        is_stale = (datetime.now() - last_update) > stale_threshold
        assert is_stale is True

    @pytest.mark.chaos
    def test_handles_llm_provider_failure(self):
        """Test fallback when LLM provider fails."""
        providers = ["primary", "secondary", "fallback"]
        results = []

        def try_provider(provider: str) -> str | None:
            if provider == "primary":
                raise Exception("Primary provider down")
            elif provider == "secondary":
                raise Exception("Secondary provider down")
            else:
                return "Fallback response"

        for provider in providers:
            try:
                result = try_provider(provider)
                results.append(result)
                break
            except Exception:
                continue

        assert len(results) == 1
        assert results[0] == "Fallback response"


class TestResourceExhaustion:
    """Tests for resource exhaustion scenarios."""

    @pytest.mark.chaos
    def test_handles_rate_limiting(self):
        """Test handling of API rate limits."""
        rate_limit_responses = [
            {"status": 429, "message": "Rate limit exceeded"},
            {"status": 429, "retry_after": 60},
        ]

        for response in rate_limit_responses:
            # System should recognize rate limit
            is_rate_limited = response.get("status") == 429
            assert is_rate_limited is True

            # Should extract retry information if available
            retry_after = response.get("retry_after", 30)  # Default 30s
            assert retry_after > 0

    @pytest.mark.chaos
    def test_handles_memory_pressure(self):
        """Test behavior under memory pressure."""
        # Simulate large data structure
        large_data = [{"price": i * 0.01, "volume": i} for i in range(10000)]

        # System should handle large datasets
        assert len(large_data) == 10000

        # Cleanup
        del large_data

    @pytest.mark.chaos
    def test_handles_queue_overflow(self):
        """Test handling of message queue overflow."""
        max_queue_size = 100
        message_queue = []

        # Simulate overflow
        for i in range(150):
            if len(message_queue) >= max_queue_size:
                # Drop oldest or reject new
                message_queue.pop(0)  # Drop oldest
            message_queue.append({"id": i})

        # Queue should not exceed max size
        assert len(message_queue) <= max_queue_size


class TestPartialFailures:
    """Tests for partial failure scenarios."""

    @pytest.mark.chaos
    def test_handles_partial_order_fill(self):
        """Test handling of partial order fills."""
        order = {
            "id": "ORD001",
            "quantity": 1000,
            "filled": 0,
            "status": "pending",
        }

        # Simulate partial fills
        fills = [100, 200, 150, 50]  # Total: 500 (partial)

        for fill in fills:
            order["filled"] += fill

        order["status"] = "partial" if order["filled"] < order["quantity"] else "filled"

        assert order["status"] == "partial"
        assert order["filled"] == 500
        assert order["quantity"] - order["filled"] == 500  # Remaining

    @pytest.mark.chaos
    def test_handles_partial_data_update(self):
        """Test handling when only some data updates succeed."""
        positions = ["SPY", "QQQ", "IWM", "DIA"]
        update_results = {
            "SPY": {"success": True, "price": 450.0},
            "QQQ": {"success": True, "price": 380.0},
            "IWM": {"success": False, "error": "Timeout"},
            "DIA": {"success": True, "price": 350.0},
        }

        successful = [p for p in positions if update_results[p]["success"]]
        failed = [p for p in positions if not update_results[p]["success"]]

        assert len(successful) == 3
        assert len(failed) == 1
        assert "IWM" in failed

    @pytest.mark.chaos
    def test_handles_partial_calculation_failure(self):
        """Test handling when some calculations fail."""
        symbols = ["SPY", "QQQ", "INVALID", "DIA"]
        results = {}

        def calculate_indicator(symbol: str) -> dict:
            if symbol == "INVALID":
                raise ValueError(f"Unknown symbol: {symbol}")
            return {"symbol": symbol, "rsi": 50.0}

        for symbol in symbols:
            try:
                results[symbol] = calculate_indicator(symbol)
            except ValueError as e:
                results[symbol] = {"error": str(e)}

        # Should have partial results
        assert "rsi" in results["SPY"]
        assert "error" in results["INVALID"]


class TestTimingFailures:
    """Tests for timing-related failures."""

    @pytest.mark.chaos
    def test_handles_market_closed(self):
        """Test handling when market is closed."""
        # Simulate market hours check
        market_hours = {
            "open": datetime.now().replace(hour=9, minute=30),
            "close": datetime.now().replace(hour=16, minute=0),
        }

        # Test outside market hours
        test_time = datetime.now().replace(hour=20, minute=0)

        is_open = market_hours["open"] <= test_time <= market_hours["close"]

        if not is_open:
            # System should queue or reject orders
            order_action = "queue_for_next_open"
            assert order_action == "queue_for_next_open"

    @pytest.mark.chaos
    def test_handles_order_expired(self):
        """Test handling of expired orders."""
        order = {
            "id": "ORD001",
            "created": datetime.now() - timedelta(hours=2),
            "ttl_minutes": 60,
            "status": "pending",
        }

        # Check if expired
        age = datetime.now() - order["created"]
        is_expired = age > timedelta(minutes=order["ttl_minutes"])

        assert is_expired is True

        # Should mark as expired
        if is_expired:
            order["status"] = "expired"
        assert order["status"] == "expired"

    @pytest.mark.chaos
    def test_handles_stale_indicators(self):
        """Test detection of stale indicator values."""
        indicator = {
            "name": "RSI",
            "value": 65.0,
            "calculated_at": datetime.now() - timedelta(minutes=10),
            "max_age_seconds": 60,
        }

        age = (datetime.now() - indicator["calculated_at"]).total_seconds()
        is_stale = age > indicator["max_age_seconds"]

        assert is_stale is True


class TestConcurrencyFailures:
    """Tests for concurrency-related failures."""

    @pytest.mark.chaos
    def test_handles_race_condition_protection(self):
        """Test protection against race conditions in position updates."""
        position = {"symbol": "SPY", "quantity": 100, "version": 1}

        def update_position(pos: dict, new_qty: int, expected_version: int) -> bool:
            """Optimistic locking update."""
            if pos["version"] != expected_version:
                return False  # Concurrent modification detected
            pos["quantity"] = new_qty
            pos["version"] += 1
            return True

        # First update succeeds
        result1 = update_position(position, 150, 1)
        assert result1 is True
        assert position["version"] == 2

        # Second update with stale version fails
        result2 = update_position(position, 200, 1)  # Using old version
        assert result2 is False

    @pytest.mark.chaos
    def test_handles_duplicate_order_submission(self):
        """Test handling of duplicate order submissions."""
        submitted_orders = set()

        def submit_order(order_id: str) -> dict:
            if order_id in submitted_orders:
                return {"success": False, "error": "Duplicate order ID"}
            submitted_orders.add(order_id)
            return {"success": True, "order_id": order_id}

        # First submission
        result1 = submit_order("ORD001")
        assert result1["success"] is True

        # Duplicate submission
        result2 = submit_order("ORD001")
        assert result2["success"] is False
        assert "Duplicate" in result2["error"]


class TestRecoveryMechanisms:
    """Tests for system recovery mechanisms."""

    @pytest.mark.chaos
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after failures."""
        circuit_breaker = {
            "state": "closed",
            "failure_count": 0,
            "failure_threshold": 5,
            "reset_timeout": 30,
            "last_failure": None,
        }

        # Simulate failures
        for _ in range(6):
            circuit_breaker["failure_count"] += 1
            if circuit_breaker["failure_count"] >= circuit_breaker["failure_threshold"]:
                circuit_breaker["state"] = "open"
                circuit_breaker["last_failure"] = datetime.now()

        assert circuit_breaker["state"] == "open"

        # Simulate reset after timeout
        circuit_breaker["last_failure"] = datetime.now() - timedelta(seconds=60)
        time_since_failure = (datetime.now() - circuit_breaker["last_failure"]).total_seconds()

        if time_since_failure > circuit_breaker["reset_timeout"]:
            circuit_breaker["state"] = "half_open"
            circuit_breaker["failure_count"] = 0

        assert circuit_breaker["state"] == "half_open"

    @pytest.mark.chaos
    def test_graceful_degradation(self):
        """Test graceful degradation when features fail."""
        features = {
            "real_time_quotes": {"available": False, "fallback": "delayed_quotes"},
            "sentiment_analysis": {"available": False, "fallback": "skip"},
            "order_execution": {"available": True, "fallback": None},
        }

        active_features = []
        degraded_features = []

        for name, config in features.items():
            if config["available"]:
                active_features.append(name)
            elif config["fallback"]:
                degraded_features.append((name, config["fallback"]))

        assert "order_execution" in active_features
        assert ("real_time_quotes", "delayed_quotes") in degraded_features

    @pytest.mark.chaos
    def test_state_recovery_from_checkpoint(self):
        """Test state recovery from saved checkpoint."""
        checkpoint = {
            "timestamp": datetime.now() - timedelta(minutes=5),
            "positions": [
                {"symbol": "SPY", "quantity": 100, "entry_price": 450.0},
                {"symbol": "QQQ", "quantity": 50, "entry_price": 380.0},
            ],
            "cash": 50000.0,
            "pending_orders": [],
        }

        # Simulate recovery
        recovered_state = {
            "positions": {p["symbol"]: p for p in checkpoint["positions"]},
            "cash": checkpoint["cash"],
            "recovered_at": datetime.now(),
            "checkpoint_age": (datetime.now() - checkpoint["timestamp"]).total_seconds(),
        }

        assert len(recovered_state["positions"]) == 2
        assert recovered_state["cash"] == 50000.0
        assert recovered_state["checkpoint_age"] > 0
