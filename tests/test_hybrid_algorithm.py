"""
Tests for HybridOptionsBot algorithm.

Tests algorithm initialization, module integration, and order processing.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Mock QuantConnect classes
class MockQCAlgorithm:
    """Mock QuantConnect algorithm base class."""

    def __init__(self):
        self.IsWarmingUp = False
        self.Time = datetime(2024, 11, 1, 10, 0, 0)
        self.Portfolio = MockPortfolio()
        self.Transactions = MockTransactions()
        self.Schedule = MockSchedule()
        self.DateRules = MockDateRules()
        self.TimeRules = MockTimeRules()

    def SetStartDate(self, year, month, day):
        pass

    def SetEndDate(self, year, month, day):
        pass

    def SetCash(self, amount):
        pass

    def SetBrokerageModel(self, brokerage, account_type):
        pass

    def Debug(self, message):
        print(f"DEBUG: {message}")

    def add_equity(self, symbol, resolution):
        mock_security = Mock()
        mock_security.Symbol = symbol
        return mock_security

    def add_option(self, symbol, resolution):
        mock_option = Mock()
        mock_option.Symbol = f"{symbol}_OPTION"
        mock_option.set_filter = Mock()
        return mock_option


class MockHolding:
    """Mock portfolio holding."""

    def __init__(self, symbol, invested=True):
        self.Symbol = symbol
        self.Invested = invested


class MockPortfolio:
    """Mock portfolio."""

    def __init__(self):
        self.TotalPortfolioValue = 100000.0
        self._holdings = {}

    @property
    def Values(self):
        """Returns list of holdings (like QC Portfolio.Values property)."""
        return list(self._holdings.values())

    def ContainsKey(self, symbol):
        return symbol in self._holdings


class MockTransactions:
    """Mock transactions."""

    def GetOrderById(self, order_id):
        mock_order = Mock()
        mock_order.Symbol = "SPY"
        mock_order.OrderId = order_id
        return mock_order


class MockDateRules:
    """Mock date rules for scheduling."""

    def EveryDay(self, symbol=None):
        return "EveryDay"


class MockTimeRules:
    """Mock time rules for scheduling."""

    def Every(self, interval):
        return f"Every_{interval}"

    def BeforeMarketClose(self, symbol, minutes):
        return f"BeforeMarketClose_{symbol}_{minutes}"

    def AfterMarketOpen(self, symbol, minutes):
        return f"AfterMarketOpen_{symbol}_{minutes}"


class MockSchedule:
    """Mock schedule."""

    def On(self, date_rule, time_rule, func):
        pass


class MockSlice:
    """Mock market data slice."""

    def __init__(self):
        self.OptionChains = {}
        self.Time = datetime(2024, 11, 1, 10, 0, 0)


class MockOrderRequest:
    """Mock order request."""

    def __init__(self, order_id="TEST001", execution_type="option_strategy"):
        self.order_id = order_id
        self.execution_type = execution_type
        self.net_debit = 2.50
        self.quantity = 1


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_qc_imports():
    """Mock QuantConnect imports with proper module structure."""
    from types import ModuleType

    # Create a proper mock module with all required classes
    mock_algorithm_imports = ModuleType("AlgorithmImports")
    mock_algorithm_imports.QCAlgorithm = type("QCAlgorithm", (), {})
    mock_algorithm_imports.Resolution = type("Resolution", (), {"Daily": "Daily", "Hour": "Hour", "Minute": "Minute"})
    mock_algorithm_imports.Slice = type("Slice", (), {})
    mock_algorithm_imports.BrokerageName = type("BrokerageName", (), {"CharlesSchwab": "CharlesSchwab"})
    mock_algorithm_imports.AccountType = type("AccountType", (), {"Margin": "Margin"})
    mock_algorithm_imports.OrderEvent = type("OrderEvent", (), {})
    mock_algorithm_imports.OptionStrategies = type("OptionStrategies", (), {})
    mock_algorithm_imports.OptionRight = type("OptionRight", (), {"Call": "Call", "Put": "Put"})
    mock_algorithm_imports.OrderStatus = type("OrderStatus", (), {"Filled": "Filled", "Canceled": "Canceled"})
    mock_algorithm_imports.Leg = type("Leg", (), {"Create": lambda *args: None})
    mock_algorithm_imports.__all__ = [
        "QCAlgorithm",
        "Resolution",
        "Slice",
        "BrokerageName",
        "AccountType",
        "OrderEvent",
        "OptionStrategies",
        "OptionRight",
        "OrderStatus",
        "Leg",
    ]

    # Clear cached modules that depend on AlgorithmImports
    mods_to_clear = [
        mod for mod in list(sys.modules.keys()) if any(x in mod for x in ["algorithms.", "execution.", "api."])
    ]
    for mod in mods_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]

    with patch.dict("sys.modules", {"AlgorithmImports": mock_algorithm_imports}):
        yield


@pytest.fixture
def mock_config():
    """Mock configuration."""
    config = Mock()
    config.get = Mock(
        side_effect=lambda key, default: {
            "risk_management": {
                "max_position_size_pct": 0.25,
                "max_daily_loss_pct": 0.03,
                "max_drawdown_pct": 0.10,
                "max_risk_per_trade_pct": 0.02,
                "max_consecutive_losses": 5,
                "require_human_reset": True,
                "max_open_positions": 5,
            },
            "option_strategies_executor": {
                "enabled": True,
                "primary_symbols": ["SPY", "QQQ", "IWM"],
            },
            "manual_legs_executor": {
                "enabled": True,
            },
            "bot_managed_positions": {
                "enabled": True,
            },
            "recurring_order_manager": {
                "enabled": True,
            },
            "quantconnect": {
                "resource_limits": {},
                "object_store": {"enabled": False},
                "compute_nodes": {
                    "backtesting": {
                        "model": "B8-16",
                        "ram_gb": 16,
                        "cores": 8,
                    }
                },
            },
        }.get(key, default)
    )
    return config


@pytest.fixture
def hybrid_algorithm(mock_qc_imports, mock_config):
    """Create HybridOptionsBot instance with mocks."""
    with patch("algorithms.hybrid_options_bot.get_config", return_value=mock_config):
        with patch("algorithms.hybrid_options_bot.create_resource_monitor"):
            with patch("algorithms.hybrid_options_bot.create_option_strategies_executor"):
                with patch("algorithms.hybrid_options_bot.create_manual_legs_executor"):
                    with patch("algorithms.hybrid_options_bot.create_bot_position_manager"):
                        with patch("algorithms.hybrid_options_bot.create_recurring_order_manager"):
                            with patch("algorithms.hybrid_options_bot.OrderQueueAPI"):
                                with patch("algorithms.hybrid_options_bot.RiskManager"):
                                    with patch("algorithms.hybrid_options_bot.TradingCircuitBreaker"):
                                        # Import after mocking
                                        from algorithms.hybrid_options_bot import HybridOptionsBot

                                        # Replace base class with mock
                                        HybridOptionsBot.__bases__ = (MockQCAlgorithm,)

                                        algo = HybridOptionsBot()
                                        algo.Initialize()
                                        return algo


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


@pytest.mark.unit
def test_algorithm_initialization(mock_qc_imports, mock_config):
    """Test that algorithm initializes all components."""
    with patch("algorithms.hybrid_options_bot.get_config", return_value=mock_config):
        with patch("algorithms.hybrid_options_bot.create_resource_monitor") as mock_resource:
            with patch("algorithms.hybrid_options_bot.create_option_strategies_executor") as mock_options:
                with patch("algorithms.hybrid_options_bot.create_manual_legs_executor") as mock_manual:
                    with patch("algorithms.hybrid_options_bot.create_bot_position_manager") as mock_bot:
                        with patch("algorithms.hybrid_options_bot.create_recurring_order_manager") as mock_recurring:
                            with patch("algorithms.hybrid_options_bot.OrderQueueAPI") as mock_queue:
                                with patch("algorithms.hybrid_options_bot.RiskManager") as mock_risk:
                                    with patch("algorithms.hybrid_options_bot.TradingCircuitBreaker") as mock_breaker:
                                        from algorithms.hybrid_options_bot import HybridOptionsBot

                                        # Replace base class
                                        HybridOptionsBot.__bases__ = (MockQCAlgorithm,)

                                        algo = HybridOptionsBot()
                                        algo.Initialize()

                                        # Verify all components created
                                        assert mock_resource.called
                                        assert mock_options.called
                                        assert mock_manual.called
                                        assert mock_bot.called
                                        assert mock_recurring.called
                                        assert mock_queue.called
                                        assert mock_risk.called
                                        assert mock_breaker.called


@pytest.mark.unit
def test_algorithm_initialization_without_config(mock_qc_imports):
    """Test algorithm handles missing config gracefully."""
    with patch("algorithms.hybrid_options_bot.get_config", side_effect=FileNotFoundError()):
        with patch("algorithms.hybrid_options_bot.create_resource_monitor"):
            with patch("algorithms.hybrid_options_bot.create_option_strategies_executor"):
                with patch("algorithms.hybrid_options_bot.create_manual_legs_executor"):
                    with patch("algorithms.hybrid_options_bot.create_bot_position_manager"):
                        with patch("algorithms.hybrid_options_bot.create_recurring_order_manager"):
                            with patch("algorithms.hybrid_options_bot.OrderQueueAPI"):
                                with patch("algorithms.hybrid_options_bot.RiskManager"):
                                    with patch("algorithms.hybrid_options_bot.TradingCircuitBreaker"):
                                        from algorithms.hybrid_options_bot import HybridOptionsBot

                                        HybridOptionsBot.__bases__ = (MockQCAlgorithm,)

                                        algo = HybridOptionsBot()
                                        algo.Initialize()

                                        # Should initialize with defaults
                                        assert algo.config is None


@pytest.mark.unit
def test_data_subscriptions(hybrid_algorithm):
    """Test that algorithm subscribes to options data."""
    # Verify option_symbols dictionary exists
    assert hasattr(hybrid_algorithm, "option_symbols")
    assert isinstance(hybrid_algorithm.option_symbols, dict)


@pytest.mark.unit
def test_scheduled_tasks_setup(hybrid_algorithm):
    """Test that scheduled tasks are configured."""
    # Schedule.On should have been called during Initialize
    # (Mocked, so just verify method exists)
    assert hasattr(hybrid_algorithm, "_scheduled_strategy_check")
    assert hasattr(hybrid_algorithm, "_scheduled_recurring_check")
    assert hasattr(hybrid_algorithm, "_daily_risk_review")


# ============================================================================
# ONDATA TESTS
# ============================================================================


@pytest.mark.unit
def test_ondata_skips_when_warming_up(hybrid_algorithm):
    """Test that OnData skips processing during warmup."""
    hybrid_algorithm.IsWarmingUp = True
    hybrid_algorithm.circuit_breaker = Mock()
    hybrid_algorithm.circuit_breaker.can_trade = Mock(return_value=True)

    slice_data = MockSlice()
    hybrid_algorithm.OnData(slice_data)

    # Should not call circuit breaker when warming up
    hybrid_algorithm.circuit_breaker.can_trade.assert_not_called()


@pytest.mark.unit
def test_ondata_skips_when_circuit_breaker_halted(hybrid_algorithm):
    """Test that OnData skips trading when circuit breaker is halted."""
    hybrid_algorithm.IsWarmingUp = False
    hybrid_algorithm.circuit_breaker = Mock()
    hybrid_algorithm.circuit_breaker.can_trade = Mock(return_value=False)
    hybrid_algorithm.order_queue = Mock()

    slice_data = MockSlice()
    hybrid_algorithm.OnData(slice_data)

    # Should not process orders when halted
    hybrid_algorithm.order_queue.get_pending_orders.assert_not_called()


@pytest.mark.unit
def test_ondata_processes_order_queue(hybrid_algorithm):
    """Test that OnData processes queued orders."""
    hybrid_algorithm.IsWarmingUp = False
    hybrid_algorithm.circuit_breaker = Mock()
    hybrid_algorithm.circuit_breaker.can_trade = Mock(return_value=True)
    hybrid_algorithm.order_queue = Mock()
    hybrid_algorithm.order_queue.get_pending_orders = Mock(return_value=[])

    slice_data = MockSlice()
    hybrid_algorithm.OnData(slice_data)

    # Should check for pending orders
    hybrid_algorithm.order_queue.get_pending_orders.assert_called_once()


@pytest.mark.unit
def test_ondata_runs_autonomous_strategies(hybrid_algorithm):
    """Test that OnData runs autonomous strategies."""
    hybrid_algorithm.IsWarmingUp = False
    hybrid_algorithm.circuit_breaker = Mock()
    hybrid_algorithm.circuit_breaker.can_trade = Mock(return_value=True)
    hybrid_algorithm.order_queue = Mock()
    hybrid_algorithm.order_queue.get_pending_orders = Mock(return_value=[])
    hybrid_algorithm.options_executor = Mock()
    hybrid_algorithm.options_executor.on_data = Mock()
    hybrid_algorithm._last_check_time = datetime(2024, 11, 1, 9, 0, 0)  # 1 hour ago

    slice_data = MockSlice()
    hybrid_algorithm.OnData(slice_data)

    # Should call options executor
    hybrid_algorithm.options_executor.on_data.assert_called_once_with(slice_data)


@pytest.mark.unit
def test_ondata_updates_bot_positions(hybrid_algorithm):
    """Test that OnData updates bot-managed positions."""
    hybrid_algorithm.IsWarmingUp = False
    hybrid_algorithm.circuit_breaker = Mock()
    hybrid_algorithm.circuit_breaker.can_trade = Mock(return_value=True)
    hybrid_algorithm.order_queue = Mock()
    hybrid_algorithm.order_queue.get_pending_orders = Mock(return_value=[])
    hybrid_algorithm.bot_manager = Mock()
    hybrid_algorithm.bot_manager.on_data = Mock()

    slice_data = MockSlice()
    hybrid_algorithm.OnData(slice_data)

    # Should update bot manager
    hybrid_algorithm.bot_manager.on_data.assert_called_once_with(slice_data)


# ============================================================================
# ORDER PROCESSING TESTS
# ============================================================================


@pytest.mark.unit
def test_process_order_queue_with_option_strategy(hybrid_algorithm):
    """Test processing option strategy order from queue."""
    hybrid_algorithm.order_queue = Mock()
    order = MockOrderRequest(order_id="OPT001", execution_type="option_strategy")
    hybrid_algorithm.order_queue.get_pending_orders = Mock(return_value=[order])
    hybrid_algorithm.options_executor = Mock()
    hybrid_algorithm.options_executor.execute_strategy_order = Mock()

    slice_data = MockSlice()
    hybrid_algorithm._process_order_queue(slice_data)

    # Should execute via options executor
    hybrid_algorithm.options_executor.execute_strategy_order.assert_called_once()
    hybrid_algorithm.order_queue.mark_order_processing.assert_called_once_with("OPT001")


@pytest.mark.unit
def test_process_order_queue_with_manual_legs(hybrid_algorithm):
    """Test processing manual legs order from queue."""
    hybrid_algorithm.order_queue = Mock()
    order = MockOrderRequest(order_id="MAN001", execution_type="manual_legs")
    hybrid_algorithm.order_queue.get_pending_orders = Mock(return_value=[order])
    hybrid_algorithm.manual_executor = Mock()
    hybrid_algorithm.manual_executor.execute_manual_order = Mock()

    slice_data = MockSlice()
    hybrid_algorithm._process_order_queue(slice_data)

    # Should execute via manual executor
    hybrid_algorithm.manual_executor.execute_manual_order.assert_called_once()
    hybrid_algorithm.order_queue.mark_order_processing.assert_called_once_with("MAN001")


@pytest.mark.unit
def test_process_order_queue_rejects_over_risk_limit(hybrid_algorithm):
    """Test that orders exceeding risk limits are rejected."""
    hybrid_algorithm.order_queue = Mock()
    order = MockOrderRequest(order_id="RISK001", execution_type="option_strategy")
    order.net_debit = 100.0  # Very large order
    order.quantity = 100
    hybrid_algorithm.order_queue.get_pending_orders = Mock(return_value=[order])
    hybrid_algorithm.Portfolio.TotalPortfolioValue = 100000.0

    slice_data = MockSlice()
    hybrid_algorithm._process_order_queue(slice_data)

    # Should reject order
    hybrid_algorithm.order_queue.mark_order_rejected.assert_called_once()


# ============================================================================
# RISK MANAGEMENT TESTS
# ============================================================================


@pytest.mark.unit
def test_check_risk_limits_passes_small_order(hybrid_algorithm):
    """Test risk check passes for small order."""
    order = MockOrderRequest()
    order.net_debit = 2.50
    order.quantity = 1
    hybrid_algorithm.Portfolio.TotalPortfolioValue = 100000.0

    result = hybrid_algorithm._check_risk_limits(order)
    assert result is True


@pytest.mark.unit
def test_check_risk_limits_rejects_large_order(hybrid_algorithm):
    """Test risk check rejects order exceeding position size limit."""
    order = MockOrderRequest()
    order.net_debit = 300.0  # $30,000 order
    order.quantity = 100
    hybrid_algorithm.Portfolio.TotalPortfolioValue = 100000.0
    hybrid_algorithm.risk_limits.max_position_size = 0.25  # 25% max

    result = hybrid_algorithm._check_risk_limits(order)
    assert result is False


@pytest.mark.unit
def test_daily_risk_review_updates_metrics(hybrid_algorithm):
    """Test daily risk review updates P&L and checks circuit breaker."""
    hybrid_algorithm.Portfolio.TotalPortfolioValue = 105000.0
    hybrid_algorithm.risk_manager = Mock()
    hybrid_algorithm.risk_manager.starting_equity = 100000.0
    hybrid_algorithm.circuit_breaker = Mock()

    hybrid_algorithm._daily_risk_review()

    # Should calculate P&L
    assert hybrid_algorithm._daily_pnl == pytest.approx(0.05, rel=1e-3)

    # Should check circuit breaker
    hybrid_algorithm.circuit_breaker.check_daily_loss.assert_called_once()
    hybrid_algorithm.circuit_breaker.check_drawdown.assert_called_once()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
def test_full_initialization_and_data_cycle(hybrid_algorithm):
    """Test full initialization and single data cycle."""
    # Setup
    hybrid_algorithm.IsWarmingUp = False
    hybrid_algorithm.circuit_breaker.can_trade = Mock(return_value=True)
    hybrid_algorithm.order_queue.get_pending_orders = Mock(return_value=[])
    hybrid_algorithm.options_executor.on_data = Mock()
    hybrid_algorithm.bot_manager.on_data = Mock()
    hybrid_algorithm._last_check_time = datetime(2024, 11, 1, 9, 0, 0)

    # Execute
    slice_data = MockSlice()
    hybrid_algorithm.OnData(slice_data)

    # Verify all components called
    assert hybrid_algorithm.circuit_breaker.can_trade.called
    assert hybrid_algorithm.order_queue.get_pending_orders.called
    assert hybrid_algorithm.options_executor.on_data.called
    assert hybrid_algorithm.bot_manager.on_data.called


@pytest.mark.integration
def test_end_of_algorithm_reporting(hybrid_algorithm):
    """Test OnEndOfAlgorithm produces final report."""
    hybrid_algorithm.Portfolio.TotalPortfolioValue = 110000.0
    hybrid_algorithm.risk_manager.starting_equity = 100000.0
    hybrid_algorithm._peak_equity = 115000.0
    hybrid_algorithm.circuit_breaker.is_halted = False

    # Should not raise exception
    hybrid_algorithm.OnEndOfAlgorithm()


# ============================================================================
# HELPER METHOD TESTS
# ============================================================================


@pytest.mark.unit
def test_get_config_with_config_loaded(hybrid_algorithm, mock_config):
    """Test _get_config returns config value when loaded."""
    result = hybrid_algorithm._get_config("risk_management", {})
    assert result is not None


@pytest.mark.unit
def test_get_config_with_no_config(hybrid_algorithm):
    """Test _get_config returns default when config is None."""
    hybrid_algorithm.config = None
    result = hybrid_algorithm._get_config("missing_section", {"default": "value"})
    assert result == {"default": "value"}


@pytest.mark.unit
def test_get_node_info(hybrid_algorithm):
    """Test _get_node_info returns node information."""
    info = hybrid_algorithm._get_node_info()
    assert "B8-16" in info
    assert "8 cores" in info
    assert "16GB RAM" in info


@pytest.mark.unit
def test_should_check_strategies_returns_true_after_delay(hybrid_algorithm):
    """Test _should_check_strategies returns True after 5 minutes."""
    hybrid_algorithm._last_check_time = datetime(2024, 11, 1, 9, 0, 0)
    hybrid_algorithm.Time = datetime(2024, 11, 1, 9, 6, 0)  # 6 minutes later

    assert hybrid_algorithm._should_check_strategies() is True


@pytest.mark.unit
def test_should_check_strategies_returns_false_before_delay(hybrid_algorithm):
    """Test _should_check_strategies returns False before 5 minutes."""
    hybrid_algorithm._last_check_time = datetime(2024, 11, 1, 9, 0, 0)
    hybrid_algorithm.Time = datetime(2024, 11, 1, 9, 3, 0)  # 3 minutes later

    assert hybrid_algorithm._should_check_strategies() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
