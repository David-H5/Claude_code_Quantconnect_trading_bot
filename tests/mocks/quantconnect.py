"""
QuantConnect Mock Classes

Consolidated mocks for QuantConnect algorithm testing.
Use these instead of redefining in individual test files.

UPGRADE-015: Mock Consolidation
"""

from datetime import datetime
from typing import Any, Callable
from unittest.mock import Mock


class MockQCAlgorithm:
    """
    Mock QuantConnect algorithm base class.

    Provides common algorithm properties and methods for testing
    without requiring the full QuantConnect framework.
    """

    def __init__(self):
        self.IsWarmingUp = False
        self.Time = datetime(2024, 11, 1, 10, 0, 0)
        self.Portfolio = MockQCPortfolio()
        self.Transactions = MockTransactions()
        self.Schedule = MockSchedule()
        self.DateRules = MockDateRules()
        self.TimeRules = MockTimeRules()
        self.Securities = {}
        self._debug_messages = []
        self._log_messages = []

    def SetStartDate(self, year: int, month: int, day: int):
        pass

    def SetEndDate(self, year: int, month: int, day: int):
        pass

    def SetCash(self, amount: float):
        pass

    def SetBrokerageModel(self, brokerage, account_type):
        pass

    def Debug(self, message: str):
        self._debug_messages.append(message)

    def Log(self, message: str):
        self._log_messages.append(message)

    def Error(self, message: str):
        self._log_messages.append(f"ERROR: {message}")

    def add_equity(self, symbol: str, resolution=None):
        mock_security = Mock()
        mock_security.Symbol = symbol
        self.Securities[symbol] = mock_security
        return mock_security

    def add_option(self, symbol: str, resolution=None):
        mock_option = Mock()
        mock_option.Symbol = f"{symbol}_OPTION"
        mock_option.set_filter = Mock()
        return mock_option


class MockQCPortfolio:
    """Mock QuantConnect portfolio."""

    def __init__(self, total_value: float = 100000.0):
        self.TotalPortfolioValue = total_value
        self.Cash = total_value
        self._holdings: dict[str, MockQCHolding] = {}

    @property
    def Values(self):
        """Returns list of holdings."""
        return list(self._holdings.values())

    def ContainsKey(self, symbol: str) -> bool:
        return symbol in self._holdings

    def __getitem__(self, symbol: str):
        if symbol in self._holdings:
            return self._holdings[symbol]
        # Return empty holding
        return MockQCHolding(symbol, invested=False)

    def add_position(self, symbol: str, quantity: int, avg_price: float):
        """Helper to add a position."""
        self._holdings[symbol] = MockQCHolding(
            symbol=symbol,
            invested=True,
            quantity=quantity,
            average_price=avg_price
        )


class MockQCHolding:
    """Mock portfolio holding."""

    def __init__(
        self,
        symbol: str,
        invested: bool = True,
        quantity: int = 0,
        average_price: float = 0.0,
    ):
        self.Symbol = symbol
        self.Invested = invested
        self.Quantity = quantity
        self.AveragePrice = average_price
        self.HoldingsValue = quantity * average_price

    @property
    def UnrealizedProfit(self) -> float:
        return 0.0  # Override in tests as needed


class MockTransactions:
    """Mock transactions manager."""

    def __init__(self):
        self._orders: dict[int, Any] = {}

    def GetOrderById(self, order_id: int):
        if order_id in self._orders:
            return self._orders[order_id]
        mock_order = Mock()
        mock_order.Symbol = "SPY"
        mock_order.OrderId = order_id
        return mock_order

    def add_order(self, order_id: int, order: Any):
        self._orders[order_id] = order


class MockDateRules:
    """Mock date rules for scheduling."""

    def EveryDay(self, symbol=None):
        return "EveryDay"

    def MonthStart(self, symbol=None):
        return "MonthStart"

    def WeekStart(self, symbol=None):
        return "WeekStart"


class MockTimeRules:
    """Mock time rules for scheduling."""

    def Every(self, interval):
        return f"Every_{interval}"

    def BeforeMarketClose(self, symbol: str, minutes: int):
        return f"BeforeMarketClose_{symbol}_{minutes}"

    def AfterMarketOpen(self, symbol: str, minutes: int):
        return f"AfterMarketOpen_{symbol}_{minutes}"

    def At(self, hour: int, minute: int):
        return f"At_{hour}_{minute}"


class MockSchedule:
    """Mock schedule manager."""

    def __init__(self):
        self._scheduled: list[tuple] = []

    def On(self, date_rule, time_rule, func: Callable):
        self._scheduled.append((date_rule, time_rule, func))


class MockQCSlice:
    """Mock market data slice."""

    def __init__(self, time: datetime | None = None):
        self.OptionChains = {}
        self.Bars = {}
        self.Time = time or datetime(2024, 11, 1, 10, 0, 0)

    def add_bar(self, symbol: str, open_: float, high: float, low: float, close: float, volume: int = 1000000):
        """Helper to add a trade bar."""
        from tests.conftest import MockTradeBar
        self.Bars[symbol] = MockTradeBar(
            open_price=open_,
            high=high,
            low=low,
            close=close,
            volume=volume
        )


class MockOrderRequest:
    """Mock order request."""

    def __init__(
        self,
        order_id: str = "TEST001",
        execution_type: str = "option_strategy",
        net_debit: float = 2.50,
        quantity: int = 1,
    ):
        self.order_id = order_id
        self.execution_type = execution_type
        self.net_debit = net_debit
        self.quantity = quantity


class MockLLMClient:
    """Mock LLM client for agent testing."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or ["Mock response"]
        self._call_count = 0

    async def complete(self, prompt: str, **kwargs) -> str:
        response = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return response

    def set_responses(self, responses: list[str]):
        self._responses = responses
        self._call_count = 0


class MockLLMResponse:
    """Mock LLM response object."""

    def __init__(self, content: str = "Mock response", model: str = "mock-model"):
        self.content = content
        self.model = model
        self.usage = {"prompt_tokens": 10, "completion_tokens": 20}


__all__ = [
    "MockQCAlgorithm",
    "MockQCPortfolio",
    "MockQCHolding",
    "MockTransactions",
    "MockDateRules",
    "MockTimeRules",
    "MockSchedule",
    "MockQCSlice",
    "MockOrderRequest",
    "MockLLMClient",
    "MockLLMResponse",
]
