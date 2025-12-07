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
]
