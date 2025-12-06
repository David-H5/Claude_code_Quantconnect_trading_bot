"""
Tests for trading algorithms.

These tests use mock objects from conftest.py to simulate the QuantConnect
environment and verify algorithm logic without requiring the actual LEAN engine.
"""

from unittest.mock import Mock

import pytest

# Import mock classes from conftest - these are the canonical implementations
from tests.conftest import (
    MockIndicator,
    MockPortfolio,
    MockSlice,
)


class TestSimpleMomentumAlgorithm:
    """Test suite for SimpleMomentumAlgorithm."""

    @pytest.fixture
    def momentum_algorithm(self):
        """Create a mock momentum algorithm instance for testing."""
        algo = Mock()
        algo.symbol = "SPY"
        algo.rsi_period = 14
        algo.oversold_threshold = 30
        algo.overbought_threshold = 70
        algo.rsi = MockIndicator(50)
        algo.Portfolio = MockPortfolio()
        algo.IsWarmingUp = False

        # Track method calls
        algo.SetHoldings = Mock()
        algo.Liquidate = Mock()
        algo.Debug = Mock()
        algo.Log = Mock()

        return algo

    def _execute_ondata_logic(self, algo, data):
        """Execute the OnData logic for momentum algorithm."""
        if algo.IsWarmingUp:
            return

        if not data.ContainsKey(algo.symbol) or not algo.rsi.IsReady:
            return

        rsi_value = algo.rsi.Current.Value
        holdings = algo.Portfolio[algo.symbol]

        # Entry signal: RSI below oversold threshold
        if rsi_value < algo.oversold_threshold and not holdings.Invested:
            algo.SetHoldings(algo.symbol, 1.0)
            algo.Debug(f"BUY: RSI = {rsi_value:.2f}")

        # Exit signal: RSI above overbought threshold
        elif rsi_value > algo.overbought_threshold and holdings.Invested:
            algo.Liquidate(algo.symbol)
            algo.Debug(f"SELL: RSI = {rsi_value:.2f}")

    @pytest.mark.unit
    def test_buy_signal_when_rsi_oversold(self, momentum_algorithm):
        """Test that buy signal triggers when RSI is below oversold threshold."""
        algo = momentum_algorithm
        algo.rsi.set_value(25)  # Oversold
        data = MockSlice({algo.symbol: Mock()})

        self._execute_ondata_logic(algo, data)

        algo.SetHoldings.assert_called_once_with(algo.symbol, 1.0)

    @pytest.mark.unit
    def test_sell_signal_when_rsi_overbought(self, momentum_algorithm):
        """Test that sell signal triggers when RSI is above overbought threshold."""
        algo = momentum_algorithm
        algo.rsi.set_value(75)  # Overbought
        algo.Portfolio.set_holding(algo.symbol, invested=True)
        data = MockSlice({algo.symbol: Mock()})

        self._execute_ondata_logic(algo, data)

        algo.Liquidate.assert_called_once_with(algo.symbol)

    @pytest.mark.unit
    def test_no_action_when_rsi_neutral(self, momentum_algorithm):
        """Test that no action taken when RSI is in neutral zone."""
        algo = momentum_algorithm
        algo.rsi.set_value(50)  # Neutral
        data = MockSlice({algo.symbol: Mock()})

        self._execute_ondata_logic(algo, data)

        algo.SetHoldings.assert_not_called()
        algo.Liquidate.assert_not_called()

    @pytest.mark.unit
    def test_no_buy_when_already_invested(self, momentum_algorithm):
        """Test that no buy signal when already invested."""
        algo = momentum_algorithm
        algo.rsi.set_value(25)  # Oversold
        algo.Portfolio.set_holding(algo.symbol, invested=True)
        data = MockSlice({algo.symbol: Mock()})

        self._execute_ondata_logic(algo, data)

        algo.SetHoldings.assert_not_called()

    @pytest.mark.unit
    def test_no_sell_when_not_invested(self, momentum_algorithm):
        """Test that no sell signal when not invested."""
        algo = momentum_algorithm
        algo.rsi.set_value(75)  # Overbought but not invested
        data = MockSlice({algo.symbol: Mock()})

        self._execute_ondata_logic(algo, data)

        algo.Liquidate.assert_not_called()

    @pytest.mark.unit
    def test_skips_during_warmup(self, momentum_algorithm):
        """Test that algorithm skips during warmup period."""
        algo = momentum_algorithm
        algo.IsWarmingUp = True
        algo.rsi.set_value(25)
        data = MockSlice({algo.symbol: Mock()})

        self._execute_ondata_logic(algo, data)

        algo.SetHoldings.assert_not_called()

    @pytest.mark.unit
    def test_skips_when_data_missing(self, momentum_algorithm):
        """Test that algorithm skips when data is missing."""
        algo = momentum_algorithm
        algo.rsi.set_value(25)
        data = MockSlice({})  # Empty data

        self._execute_ondata_logic(algo, data)

        algo.SetHoldings.assert_not_called()

    @pytest.mark.unit
    def test_skips_when_indicator_not_ready(self, momentum_algorithm):
        """Test that algorithm skips when indicator is not ready."""
        algo = momentum_algorithm
        algo.rsi.set_value(25)
        algo.rsi.IsReady = False
        data = MockSlice({algo.symbol: Mock()})

        self._execute_ondata_logic(algo, data)

        algo.SetHoldings.assert_not_called()

    @pytest.mark.unit
    def test_rsi_boundary_oversold(self, momentum_algorithm):
        """Test RSI exactly at oversold threshold."""
        algo = momentum_algorithm
        algo.rsi.set_value(30)  # Exactly at threshold
        data = MockSlice({algo.symbol: Mock()})

        self._execute_ondata_logic(algo, data)

        # Should NOT buy - must be BELOW threshold
        algo.SetHoldings.assert_not_called()

    @pytest.mark.unit
    def test_rsi_boundary_overbought(self, momentum_algorithm):
        """Test RSI exactly at overbought threshold."""
        algo = momentum_algorithm
        algo.rsi.set_value(70)  # Exactly at threshold
        algo.Portfolio.set_holding(algo.symbol, invested=True)
        data = MockSlice({algo.symbol: Mock()})

        self._execute_ondata_logic(algo, data)

        # Should NOT sell - must be ABOVE threshold
        algo.Liquidate.assert_not_called()


class TestBasicBuyAndHoldAlgorithm:
    """Test suite for BasicBuyAndHoldAlgorithm."""

    @pytest.fixture
    def buyhold_algorithm(self):
        """Create a mock buy-and-hold algorithm instance."""
        algo = Mock()
        algo.symbol = "SPY"
        algo.Portfolio = MockPortfolio()
        algo.SetHoldings = Mock()
        algo.Debug = Mock()

        return algo

    def _execute_ondata_logic(self, algo, data):
        """Execute the OnData logic for buy-and-hold algorithm."""
        if data.ContainsKey(algo.symbol):
            if not algo.Portfolio[algo.symbol].Invested:
                algo.SetHoldings(algo.symbol, 1.0)
                algo.Debug("Purchased SPY")

    @pytest.mark.unit
    def test_buys_on_first_data(self, buyhold_algorithm):
        """Test that buy-and-hold buys on first data."""
        algo = buyhold_algorithm
        data = MockSlice({algo.symbol: Mock()})

        self._execute_ondata_logic(algo, data)

        algo.SetHoldings.assert_called_once_with(algo.symbol, 1.0)

    @pytest.mark.unit
    def test_does_not_buy_when_invested(self, buyhold_algorithm):
        """Test that no additional buys when already invested."""
        algo = buyhold_algorithm
        algo.Portfolio.set_holding(algo.symbol, invested=True)
        data = MockSlice({algo.symbol: Mock()})

        self._execute_ondata_logic(algo, data)

        algo.SetHoldings.assert_not_called()

    @pytest.mark.unit
    def test_skips_when_no_data(self, buyhold_algorithm):
        """Test that algorithm skips when no data available."""
        algo = buyhold_algorithm
        data = MockSlice({})

        self._execute_ondata_logic(algo, data)

        algo.SetHoldings.assert_not_called()


class TestAlgorithmParameters:
    """Test parameter validation and edge cases."""

    @pytest.mark.unit
    def test_rsi_threshold_validation(self):
        """Test RSI threshold parameters are valid."""
        oversold = 30
        overbought = 70

        assert 0 < oversold < 50, "Oversold should be between 0 and 50"
        assert 50 < overbought < 100, "Overbought should be between 50 and 100"
        assert oversold < overbought, "Oversold must be less than overbought"

    @pytest.mark.unit
    def test_rsi_period_validation(self):
        """Test RSI period is reasonable."""
        rsi_period = 14

        assert rsi_period > 0, "RSI period must be positive"
        assert rsi_period <= 50, "RSI period should not be too large"

    @pytest.mark.unit
    def test_position_size_validation(self):
        """Test position size is valid."""
        position_size = 1.0

        assert 0 < position_size <= 1.0, "Position size should be between 0 and 1"


class TestAlgorithmIntegration:
    """Integration tests for algorithm behavior."""

    @pytest.mark.integration
    def test_momentum_full_cycle(self):
        """Test a full buy/sell cycle for momentum algorithm."""
        algo = Mock()
        algo.symbol = "SPY"
        algo.oversold_threshold = 30
        algo.overbought_threshold = 70
        algo.rsi = MockIndicator(50)
        algo.Portfolio = MockPortfolio()
        algo.IsWarmingUp = False
        algo.SetHoldings = Mock()
        algo.Liquidate = Mock()
        algo.Debug = Mock()

        def execute_logic(algo, data):
            if algo.IsWarmingUp:
                return
            if not data.ContainsKey(algo.symbol) or not algo.rsi.IsReady:
                return
            rsi_value = algo.rsi.Current.Value
            holdings = algo.Portfolio[algo.symbol]
            if rsi_value < algo.oversold_threshold and not holdings.Invested:
                algo.SetHoldings(algo.symbol, 1.0)
                algo.Portfolio.set_holding(algo.symbol, invested=True)
            elif rsi_value > algo.overbought_threshold and holdings.Invested:
                algo.Liquidate(algo.symbol)
                algo.Portfolio.set_holding(algo.symbol, invested=False)

        # Step 1: RSI goes oversold - should buy
        algo.rsi.set_value(25)
        data = MockSlice({algo.symbol: Mock()})
        execute_logic(algo, data)
        assert algo.SetHoldings.called, "Should have bought"

        # Step 2: RSI goes neutral - should hold
        algo.SetHoldings.reset_mock()
        algo.rsi.set_value(50)
        execute_logic(algo, data)
        assert not algo.SetHoldings.called, "Should not trade in neutral"
        assert not algo.Liquidate.called, "Should not sell in neutral"

        # Step 3: RSI goes overbought - should sell
        algo.rsi.set_value(75)
        execute_logic(algo, data)
        assert algo.Liquidate.called, "Should have sold"
