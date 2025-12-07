"""
Pytest configuration and fixtures for QuantConnect trading bot tests.

This module provides common fixtures and utilities for testing algorithms,
indicators, and other components without requiring the actual LEAN engine.

TIMEOUT CONFIGURATION:
- Default timeout: 30 seconds (configured in pytest.ini)
- Override per-test: @pytest.mark.timeout(60)
- Override per-class: @pytest.mark.timeout(120)
- Skip timeout: @pytest.mark.timeout(0)

FIXTURE SCOPES:
- function (default): New instance per test - use for mutable state
- class: Shared within test class - use for immutable/read-only data
- session: Shared across all tests - use sparingly, only for setup

Example usage:
    @pytest.mark.timeout(10)  # 10 second timeout
    def test_fast_operation():
        ...

    @pytest.mark.timeout(60)  # 60 second timeout for slow tests
    def test_slow_operation():
        ...
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

if TYPE_CHECKING:
    from typing import Generator


# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Mock QuantConnect imports for testing
class MockSymbol:
    """Mock QuantConnect Symbol."""

    def __init__(self, ticker: str):
        self.Value = ticker
        self._ticker = ticker

    def __str__(self):
        return self._ticker

    def __hash__(self):
        return hash(self._ticker)

    def __eq__(self, other):
        if isinstance(other, MockSymbol):
            return self._ticker == other._ticker
        return self._ticker == other


class MockTradeBar:
    """Mock QuantConnect TradeBar."""

    def __init__(
        self,
        open_price: float = 100.0,
        high: float = 105.0,
        low: float = 95.0,
        close: float = 102.0,
        volume: int = 1000000,
    ):
        self.Open = open_price
        self.High = high
        self.Low = low
        self.Close = close
        self.Volume = volume
        self.Time = datetime.now()
        self.EndTime = datetime.now()


class MockSlice:
    """Mock QuantConnect Slice."""

    def __init__(self, data: dict = None):
        self._data = data or {}
        self.Time = datetime.now()

    def ContainsKey(self, symbol) -> bool:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        return key in self._data or symbol in self._data

    def __getitem__(self, symbol):
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        return self._data.get(key) or self._data.get(symbol)

    def get(self, symbol, default=None):
        try:
            return self[symbol]
        except KeyError:
            return default


class MockIndicatorDataPoint:
    """Mock indicator data point."""

    def __init__(self, value: float):
        self.Value = value


class MockIndicator:
    """Mock technical indicator."""

    def __init__(self, value: float = 50.0, is_ready: bool = True):
        self._value = value
        self.IsReady = is_ready
        self.Current = MockIndicatorDataPoint(value)

    def Update(self, time, value):
        self._value = value
        self.Current = MockIndicatorDataPoint(value)

    def set_value(self, value: float):
        self._value = value
        self.Current = MockIndicatorDataPoint(value)


class MockPortfolioHolding:
    """Mock portfolio holding."""

    def __init__(
        self,
        invested: bool = False,
        quantity: float = 0,
        average_price: float = 0,
        unrealized_pnl: float = 0,
    ):
        self.Invested = invested
        self.Quantity = quantity
        self.AveragePrice = average_price
        self.UnrealizedProfit = unrealized_pnl


class MockPortfolio:
    """Mock portfolio."""

    def __init__(self, cash: float = 100000):
        self._holdings = {}
        self.Cash = cash
        self.TotalPortfolioValue = cash

    def __getitem__(self, symbol):
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        if key not in self._holdings:
            self._holdings[key] = MockPortfolioHolding()
        return self._holdings[key]

    def set_holding(
        self,
        symbol,
        invested: bool = True,
        quantity: float = 100,
        avg_price: float = 100.0,
    ):
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        self._holdings[key] = MockPortfolioHolding(invested, quantity, avg_price)


class MockSecurityHolding:
    """Mock security info."""

    def __init__(self, symbol: str, price: float = 100.0):
        self.Symbol = MockSymbol(symbol)
        self.Price = price


class MockSecurities:
    """Mock securities dictionary."""

    def __init__(self):
        self._securities = {}

    def __getitem__(self, symbol):
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        if key not in self._securities:
            self._securities[key] = MockSecurityHolding(key)
        return self._securities[key]

    def __contains__(self, symbol):
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        return key in self._securities


# ============================================================================
# FIXTURES - Core Mock Objects
# ============================================================================
# Scope Guide:
#   - function: Mutable state, needs isolation (default for safety)
#   - class: Immutable/read-only, shared within test class for performance
#   - session: Setup-only, shared across all tests


@pytest.fixture(scope="class")
def mock_symbol() -> MockSymbol:
    """Create a mock symbol (class-scoped, immutable)."""
    return MockSymbol("SPY")


@pytest.fixture(scope="class")
def mock_trade_bar() -> MockTradeBar:
    """Create a mock trade bar with default values (class-scoped, read-only)."""
    return MockTradeBar()


@pytest.fixture
def mock_slice(mock_symbol: MockSymbol, mock_trade_bar: MockTradeBar) -> MockSlice:
    """Create a mock slice with SPY data (function-scoped for isolation)."""
    return MockSlice({mock_symbol.Value: mock_trade_bar})


@pytest.fixture(scope="class")
def empty_slice() -> MockSlice:
    """Create an empty mock slice (class-scoped, immutable)."""
    return MockSlice({})


@pytest.fixture
def mock_portfolio() -> MockPortfolio:
    """Create a mock portfolio (function-scoped, mutable state)."""
    return MockPortfolio()


@pytest.fixture
def mock_securities() -> MockSecurities:
    """Create mock securities (function-scoped, mutable state)."""
    return MockSecurities()


@pytest.fixture
def mock_rsi() -> MockIndicator:
    """Create a mock RSI indicator (function-scoped, tests may modify)."""
    return MockIndicator(value=50.0, is_ready=True)


@pytest.fixture
def mock_algorithm(
    mock_symbol: MockSymbol,
    mock_portfolio: MockPortfolio,
    mock_securities: MockSecurities,
) -> Mock:
    """Create a fully mocked algorithm instance (function-scoped, complex mutable state)."""
    algo = Mock()
    algo.symbol = mock_symbol
    algo.Portfolio = mock_portfolio
    algo.Securities = mock_securities
    algo.IsWarmingUp = False
    algo.Time = datetime.now()

    # Mock methods
    algo.SetHoldings = Mock()
    algo.Liquidate = Mock()
    algo.MarketOrder = Mock()
    algo.LimitOrder = Mock()
    algo.Debug = Mock()
    algo.Log = Mock()
    algo.Error = Mock()

    return algo


# Helper functions


def create_price_series(
    start_price: float = 100.0,
    num_bars: int = 50,
    volatility: float = 0.02,
) -> list[MockTradeBar]:
    """Create a series of price bars for testing."""
    import random

    bars = []
    price = start_price

    for _ in range(num_bars):
        change = price * volatility * (random.random() * 2 - 1)
        open_price = price
        close_price = price + change
        high = max(open_price, close_price) * (1 + random.random() * volatility)
        low = min(open_price, close_price) * (1 - random.random() * volatility)

        bars.append(MockTradeBar(open_price=open_price, high=high, low=low, close=close_price, volume=1000000))
        price = close_price

    return bars


def create_rsi_values(oversold_count: int = 0, overbought_count: int = 0, neutral_count: int = 50):
    """Create a list of RSI values for testing."""
    values = []
    values.extend([25.0] * oversold_count)
    values.extend([75.0] * overbought_count)
    values.extend([50.0] * neutral_count)
    return values


# ============================================================================
# TIMEOUT CONFIGURATION
# ============================================================================

# Timeout thresholds by test type (seconds)
TIMEOUT_THRESHOLDS = {
    "unit": 10,  # Fast unit tests
    "integration": 30,  # Integration tests
    "slow": 120,  # Tests marked as slow
    "stress": 300,  # Stress tests
    "e2e": 180,  # End-to-end tests
    "backtest": 300,  # Backtest tests
    "montecarlo": 120,  # Monte Carlo simulations
}


def pytest_configure(config):
    """Configure pytest with timeout settings."""
    # Log timeout configuration
    timeout = config.getini("timeout")
    if timeout:
        print(f"\n[pytest-timeout] Default timeout: {timeout}s")


def pytest_collection_modifyitems(config, items):
    """Apply appropriate timeouts based on test markers."""
    for item in items:
        # Skip if test already has explicit timeout marker
        if any(marker.name == "timeout" for marker in item.iter_markers()):
            continue

        # Apply timeout based on test type markers
        for marker_name, timeout in TIMEOUT_THRESHOLDS.items():
            if any(marker.name == marker_name for marker in item.iter_markers()):
                item.add_marker(pytest.mark.timeout(timeout))
                break


@pytest.fixture(scope="session", autouse=True)
def timeout_warning() -> Generator[None, None, None]:
    """Session-scoped fixture to warn about timeout configuration."""
    # Check if pytest-timeout is installed
    try:
        import pytest_timeout  # noqa: F401
    except ImportError:
        print("\n[WARNING] pytest-timeout not installed. Tests may hang indefinitely.")
        print("Install with: pip install pytest-timeout")
    yield


# Convenience fixtures for tests that need longer timeouts
@pytest.fixture
def long_timeout() -> None:
    """Fixture for tests that need longer execution time (120s)."""
    # This is a marker fixture - use @pytest.mark.timeout(120) instead
    pass


@pytest.fixture
def stress_timeout() -> None:
    """Fixture for stress tests that need extended time (300s)."""
    # This is a marker fixture - use @pytest.mark.timeout(300) instead
    pass


# ============================================================================
# TESTING FRAMEWORK UTILITIES
# ============================================================================
# Reusable utilities for reducing test duplication and improving consistency


class FaultInjector:
    """Utility class for chaos engineering and fault injection tests.

    Usage:
        injector = FaultInjector(failure_rate=0.3)
        if injector.should_fail():
            raise NetworkError("Simulated failure")
    """

    def __init__(self, failure_rate: float = 0.1, seed: int | None = None):
        import random
        self._rng = random.Random(seed)
        self.failure_rate = failure_rate

    def should_fail(self) -> bool:
        """Returns True with probability equal to failure_rate."""
        return self._rng.random() < self.failure_rate

    def inject_latency(self, min_ms: int = 10, max_ms: int = 100) -> float:
        """Returns a random latency value in milliseconds."""
        return self._rng.uniform(min_ms, max_ms)

    def corrupt_data(self, data: dict, corruption_rate: float = 0.1) -> dict:
        """Randomly corrupt dictionary values for robustness testing."""
        corrupted = data.copy()
        for key in corrupted:
            if self._rng.random() < corruption_rate:
                if isinstance(corrupted[key], (int, float)):
                    corrupted[key] = float('nan')
                elif isinstance(corrupted[key], str):
                    corrupted[key] = ""
                else:
                    corrupted[key] = None
        return corrupted


@pytest.fixture
def fault_injector() -> FaultInjector:
    """Fixture for fault injection testing."""
    return FaultInjector(failure_rate=0.3, seed=42)


# ============================================================================
# PARAMETRIZED TEST HELPERS
# ============================================================================


def assert_dataclass_to_dict(obj, expected_fields: dict | None = None) -> dict:
    """Helper to test dataclass.to_dict() methods consistently.

    Args:
        obj: Dataclass instance with to_dict() method
        expected_fields: Optional dict of field_name -> expected_value

    Returns:
        The resulting dictionary for additional assertions
    """
    assert hasattr(obj, "to_dict"), f"{type(obj).__name__} missing to_dict() method"
    result = obj.to_dict()
    assert isinstance(result, dict), "to_dict() should return a dict"

    if expected_fields:
        for field, value in expected_fields.items():
            assert field in result, f"Missing field: {field}"
            assert result[field] == value, f"Field {field}: expected {value}, got {result[field]}"

    return result


def assert_config_defaults(config_class, expected_defaults: dict):
    """Helper to test configuration class default values.

    Args:
        config_class: Configuration class to test
        expected_defaults: Dict of attribute_name -> expected_default_value
    """
    config = config_class()
    for attr, expected in expected_defaults.items():
        actual = getattr(config, attr)
        assert actual == expected, f"{attr}: expected {expected}, got {actual}"


def assert_factory_creates_valid(factory_func, *args, **kwargs):
    """Helper to test factory functions create valid objects.

    Args:
        factory_func: Factory function to test
        *args, **kwargs: Arguments to pass to factory

    Returns:
        Created object for additional assertions
    """
    obj = factory_func(*args, **kwargs)
    assert obj is not None, f"{factory_func.__name__} returned None"
    return obj


def assert_in_range(value, min_val, max_val, name: str = "value"):
    """Helper to assert value is within expected range.

    Args:
        value: Value to check
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name for error messages
    """
    assert min_val <= value <= max_val, \
        f"{name} = {value} not in range [{min_val}, {max_val}]"


# ============================================================================
# SAFETY-CRITICAL TEST UTILITIES
# ============================================================================


class SafetyTestCase:
    """Base class for safety-critical tests with required assertions.

    Subclass this for tests involving:
    - Risk limits
    - Circuit breakers
    - Order validation
    - Compliance checks

    Usage:
        class TestCircuitBreakerSafety(SafetyTestCase):
            def test_halt_prevents_trading(self, circuit_breaker):
                self.assert_safety_invariant(
                    not circuit_breaker.can_trade(),
                    "Halted breaker must prevent trading"
                )
    """

    def assert_safety_invariant(self, condition: bool, message: str):
        """Assert a safety-critical invariant with descriptive message."""
        assert condition, f"SAFETY VIOLATION: {message}"

    def assert_risk_limit(self, value, limit, limit_name: str):
        """Assert a value respects a risk limit."""
        assert value <= limit, \
            f"RISK LIMIT BREACH: {limit_name} = {value} exceeds limit {limit}"

    def assert_position_valid(self, quantity, max_position, symbol: str = ""):
        """Assert position size is valid."""
        self.assert_risk_limit(
            abs(quantity),
            max_position,
            f"Position size for {symbol}"
        )

    def assert_loss_acceptable(self, loss_pct, max_loss_pct, context: str = ""):
        """Assert loss is within acceptable limits."""
        self.assert_risk_limit(
            loss_pct,
            max_loss_pct,
            f"Loss percentage {context}"
        )


@pytest.fixture
def safety_assertions() -> SafetyTestCase:
    """Fixture providing safety assertion helpers."""
    return SafetyTestCase()


# ============================================================================
# MARKET SCENARIO GENERATORS
# ============================================================================


def create_market_crash_scenario(
    initial_price: float = 100.0,
    crash_pct: float = 0.20,
    num_bars: int = 10,
) -> list[MockTradeBar]:
    """Create a market crash price series for stress testing.

    Args:
        initial_price: Starting price
        crash_pct: Total crash percentage (0.20 = 20% drop)
        num_bars: Number of bars for the crash
    """
    bars = []
    price = initial_price
    drop_per_bar = crash_pct / num_bars

    for i in range(num_bars):
        open_price = price
        close_price = price * (1 - drop_per_bar)
        low = close_price * 0.99  # Slight overshoot
        high = open_price * 1.01
        bars.append(MockTradeBar(
            open_price=open_price,
            high=high,
            low=low,
            close=close_price,
            volume=5000000  # High volume during crash
        ))
        price = close_price

    return bars


def create_volatile_scenario(
    base_price: float = 100.0,
    volatility: float = 0.05,
    num_bars: int = 50,
    seed: int | None = None,
) -> list[MockTradeBar]:
    """Create a high-volatility price series.

    Args:
        base_price: Base price around which to oscillate
        volatility: Daily volatility (0.05 = 5%)
        num_bars: Number of bars to generate
        seed: Random seed for reproducibility
    """
    import random
    rng = random.Random(seed)

    bars = []
    price = base_price

    for _ in range(num_bars):
        change = price * volatility * (rng.random() * 2 - 1)
        open_price = price
        close_price = price + change

        # High volatility = wider high/low range
        high = max(open_price, close_price) * (1 + rng.random() * volatility)
        low = min(open_price, close_price) * (1 - rng.random() * volatility)

        bars.append(MockTradeBar(
            open_price=open_price,
            high=high,
            low=low,
            close=close_price,
            volume=int(1000000 * (1 + abs(change / price)))  # Volume spike on moves
        ))
        price = close_price

    return bars


def create_gap_scenario(
    pre_gap_price: float = 100.0,
    gap_pct: float = 0.10,
    direction: str = "down",
) -> tuple[MockTradeBar, MockTradeBar]:
    """Create a gap up/down scenario (2 bars).

    Args:
        pre_gap_price: Price before the gap
        gap_pct: Gap percentage (0.10 = 10% gap)
        direction: "up" or "down"

    Returns:
        Tuple of (pre_gap_bar, post_gap_bar)
    """
    multiplier = (1 + gap_pct) if direction == "up" else (1 - gap_pct)
    post_gap_price = pre_gap_price * multiplier

    pre_bar = MockTradeBar(
        open_price=pre_gap_price * 0.99,
        high=pre_gap_price * 1.01,
        low=pre_gap_price * 0.98,
        close=pre_gap_price,
        volume=1000000
    )

    post_bar = MockTradeBar(
        open_price=post_gap_price,  # Gap open
        high=post_gap_price * 1.02,
        low=post_gap_price * 0.98,
        close=post_gap_price * 1.01,
        volume=3000000  # High volume on gap
    )

    return pre_bar, post_bar


# ============================================================================
# REGRESSION TEST HELPERS
# ============================================================================


def regression_test(bug_id: str, description: str):
    """Decorator to mark regression tests with bug tracking info.

    Usage:
        @regression_test("BUG-123", "Division by zero in position sizing")
        def test_zero_equity_handling():
            ...
    """
    def decorator(func):
        func._bug_id = bug_id
        func._bug_description = description
        # Add pytest marker
        return pytest.mark.regression(func)
    return decorator


# ============================================================================
# EXPORTED MOCK CLASSES
# ============================================================================
# These can be imported in test files for custom mock construction:
#   from tests.conftest import MockSymbol, MockPortfolio, MockTradeBar

__all__ = [
    # Mock classes
    "MockSymbol",
    "MockTradeBar",
    "MockSlice",
    "MockIndicator",
    "MockIndicatorDataPoint",
    "MockPortfolio",
    "MockPortfolioHolding",
    "MockSecurities",
    "MockSecurityHolding",
    # Helper functions
    "create_price_series",
    "create_rsi_values",
    # Testing framework utilities
    "FaultInjector",
    "SafetyTestCase",
    "assert_dataclass_to_dict",
    "assert_config_defaults",
    "assert_factory_creates_valid",
    "assert_in_range",
    # Market scenario generators
    "create_market_crash_scenario",
    "create_volatile_scenario",
    "create_gap_scenario",
    # Regression helpers
    "regression_test",
]
