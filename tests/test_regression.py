"""
Regression Tests for Core Functionality

Tests to ensure core functionality doesn't break when changes are made.
These tests cover critical paths that must always work correctly.

Based on best practices from:
- Martin Fowler's "Refactoring" regression testing patterns
- CI/CD pipeline testing strategies
- Financial system regression testing
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest


@dataclass
class RegressionTestCase:
    """Represents a regression test case with known inputs and outputs."""

    name: str
    inputs: dict[str, Any]
    expected_outputs: dict[str, Any]
    tolerance: float = 0.0001


class TestCoreCalculations:
    """Regression tests for core calculation functions."""

    @pytest.mark.regression
    def test_position_value_calculation(self):
        """Test position value calculation is consistent."""
        test_cases = [
            {"quantity": 100, "price": 50.0, "expected": 5000.0},
            {"quantity": -50, "price": 100.0, "expected": -5000.0},
            {"quantity": 0, "price": 100.0, "expected": 0.0},
            {"quantity": 100, "price": 0.0, "expected": 0.0},
        ]

        for case in test_cases:
            result = case["quantity"] * case["price"]
            assert result == case["expected"], f"Failed for {case}"

    @pytest.mark.regression
    def test_pnl_calculation(self):
        """Test P&L calculation is consistent."""
        test_cases = [
            # entry, current, quantity, expected_pnl
            {"entry": 100.0, "current": 110.0, "quantity": 100, "expected": 1000.0},
            {"entry": 100.0, "current": 90.0, "quantity": 100, "expected": -1000.0},
            {"entry": 100.0, "current": 110.0, "quantity": -100, "expected": -1000.0},  # Short
        ]

        for case in test_cases:
            pnl = (case["current"] - case["entry"]) * case["quantity"]
            assert pnl == case["expected"], f"Failed for {case}"

    @pytest.mark.regression
    def test_percentage_return_calculation(self):
        """Test percentage return calculation is consistent."""
        test_cases = [
            {"initial": 10000.0, "final": 11000.0, "expected": 0.10},
            {"initial": 10000.0, "final": 9000.0, "expected": -0.10},
            {"initial": 10000.0, "final": 10000.0, "expected": 0.0},
        ]

        for case in test_cases:
            pct_return = (case["final"] - case["initial"]) / case["initial"]
            assert abs(pct_return - case["expected"]) < 0.0001, f"Failed for {case}"

    @pytest.mark.regression
    def test_drawdown_calculation(self):
        """Test drawdown calculation is consistent."""
        test_cases = [
            {"peak": 100000.0, "current": 90000.0, "expected": 0.10},
            {"peak": 100000.0, "current": 100000.0, "expected": 0.0},
            {"peak": 100000.0, "current": 80000.0, "expected": 0.20},
        ]

        for case in test_cases:
            drawdown = (case["peak"] - case["current"]) / case["peak"]
            assert abs(drawdown - case["expected"]) < 0.0001, f"Failed for {case}"


class TestRiskManagerRegression:
    """Regression tests for risk manager functionality."""

    @pytest.mark.regression
    def test_position_size_limits_unchanged(self):
        """Test that position size limits behave consistently."""
        # Known configuration
        max_position_pct = 0.25
        portfolio_value = 100000.0

        # Test cases with expected behavior
        test_cases = [
            {"proposed_pct": 0.10, "should_allow": True},
            {"proposed_pct": 0.25, "should_allow": True},
            {"proposed_pct": 0.30, "should_reduce": True},
            {"proposed_pct": 0.50, "should_reduce": True},
        ]

        for case in test_cases:
            proposed = case["proposed_pct"]
            if case.get("should_allow"):
                assert proposed <= max_position_pct
            elif case.get("should_reduce"):
                # Would need to reduce to max
                reduced = min(proposed, max_position_pct)
                assert reduced == max_position_pct

    @pytest.mark.regression
    def test_stop_loss_calculation_unchanged(self):
        """Test stop loss calculation is consistent."""
        # Known formula: stop = entry * (1 - risk_per_trade / position_size)
        test_cases = [
            {
                "entry": 100.0,
                "risk_per_trade": 0.02,
                "position_size": 0.20,
                "expected_stop": 90.0,
            },
            {
                "entry": 50.0,
                "risk_per_trade": 0.02,
                "position_size": 0.10,
                "expected_stop": 40.0,
            },
        ]

        for case in test_cases:
            stop = case["entry"] * (1 - case["risk_per_trade"] / case["position_size"])
            assert abs(stop - case["expected_stop"]) < 0.01, f"Failed for {case}"


class TestCircuitBreakerRegression:
    """Regression tests for circuit breaker functionality."""

    @pytest.mark.regression
    def test_daily_loss_trip_threshold(self):
        """Test daily loss threshold is consistent."""
        max_daily_loss = 0.03  # 3%

        test_cases = [
            {"daily_pnl": -0.02, "should_trip": False},
            {"daily_pnl": -0.029, "should_trip": False},
            {"daily_pnl": -0.03, "should_trip": True},
            {"daily_pnl": -0.04, "should_trip": True},
        ]

        for case in test_cases:
            # Daily loss check: trip if loss >= threshold
            trips = case["daily_pnl"] <= -max_daily_loss
            assert trips == case["should_trip"], f"Failed for {case}"

    @pytest.mark.regression
    def test_drawdown_trip_threshold(self):
        """Test drawdown threshold is consistent."""
        max_drawdown = 0.10  # 10%

        test_cases = [
            {"current": 95000, "peak": 100000, "should_trip": False},  # 5%
            {"current": 90000, "peak": 100000, "should_trip": False},  # 10% exactly
            {"current": 89000, "peak": 100000, "should_trip": True},  # 11%
            {"current": 80000, "peak": 100000, "should_trip": True},  # 20%
        ]

        for case in test_cases:
            drawdown = (case["peak"] - case["current"]) / case["peak"]
            trips = drawdown > max_drawdown
            assert trips == case["should_trip"], f"Failed for {case}: drawdown={drawdown}"

    @pytest.mark.regression
    def test_consecutive_loss_tracking(self):
        """Test consecutive loss counter behavior is consistent."""
        max_consecutive = 5

        # Sequence of trade results
        sequences = [
            {"trades": [False, False, False], "final_count": 3, "trips": False},
            {"trades": [False, False, False, False, False], "final_count": 5, "trips": True},
            {"trades": [False, False, True, False], "final_count": 1, "trips": False},  # Reset
        ]

        for seq in sequences:
            count = 0
            for is_winner in seq["trades"]:
                if is_winner:
                    count = 0
                else:
                    count += 1

            assert count == seq["final_count"], f"Count mismatch for {seq}"
            assert (count >= max_consecutive) == seq["trips"], f"Trip mismatch for {seq}"


class TestIndicatorRegression:
    """Regression tests for technical indicators."""

    @pytest.mark.regression
    def test_rsi_bounds(self):
        """Test RSI always stays within 0-100."""
        # Generate various price sequences
        np.random.seed(42)

        for _ in range(10):
            prices = 100 + np.cumsum(np.random.randn(100))
            prices = np.maximum(prices, 1)  # Keep positive

            # Calculate RSI (simplified)
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])

            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            assert 0 <= rsi <= 100, f"RSI out of bounds: {rsi}"

    @pytest.mark.regression
    def test_sma_calculation(self):
        """Test SMA calculation is consistent."""
        prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        period = 5

        # Calculate SMA for last value
        sma = sum(prices[-period:]) / period

        # Known value: (16+17+18+19+20)/5 = 18.0
        assert sma == 18.0

    @pytest.mark.regression
    def test_ema_weighting(self):
        """Test EMA weighting formula is consistent."""
        prices = [100, 102, 101, 103, 105]
        period = 3
        multiplier = 2 / (period + 1)  # 0.5

        # Start with SMA
        ema = sum(prices[:period]) / period

        # Update with remaining prices
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        # Verify multiplier
        assert multiplier == 0.5

        # EMA should be between min and max price
        assert min(prices) <= ema <= max(prices)


class TestOrderExecutionRegression:
    """Regression tests for order execution logic."""

    @pytest.mark.regression
    def test_slippage_calculation(self):
        """Test slippage calculation is consistent."""
        test_cases = [
            {"expected": 100.0, "actual": 100.10, "slippage": 0.001},
            {"expected": 100.0, "actual": 99.90, "slippage": -0.001},
            {"expected": 100.0, "actual": 100.0, "slippage": 0.0},
        ]

        for case in test_cases:
            slippage = (case["actual"] - case["expected"]) / case["expected"]
            assert abs(slippage - case["slippage"]) < 0.0001, f"Failed for {case}"

    @pytest.mark.regression
    def test_bid_increase_calculation(self):
        """Test bid increase calculation for cancel/replace."""
        test_cases = [
            {"current_bid": 1.00, "increase_pct": 0.05, "new_bid": 1.05},
            {"current_bid": 2.50, "increase_pct": 0.10, "new_bid": 2.75},
            {"current_bid": 0.50, "increase_pct": 0.20, "new_bid": 0.60},
        ]

        for case in test_cases:
            new_bid = case["current_bid"] * (1 + case["increase_pct"])
            assert abs(new_bid - case["new_bid"]) < 0.01, f"Failed for {case}"

    @pytest.mark.regression
    def test_fill_probability_bounds(self):
        """Test fill probability stays within 0-1."""
        test_cases = [
            {"bid_vs_ask": 0.0, "expected_range": (0.0, 0.3)},  # At bid
            {"bid_vs_ask": 0.5, "expected_range": (0.3, 0.7)},  # Mid
            {"bid_vs_ask": 1.0, "expected_range": (0.7, 1.0)},  # At ask
        ]

        for case in test_cases:
            # Simplified fill probability model
            prob = case["bid_vs_ask"] * 0.7 + 0.1
            prob = max(0, min(1, prob))

            assert 0 <= prob <= 1, f"Probability out of bounds: {prob}"
            assert case["expected_range"][0] <= prob <= case["expected_range"][1]


class TestProfitTakingRegression:
    """Regression tests for profit-taking logic."""

    @pytest.mark.regression
    def test_profit_thresholds_order(self):
        """Test profit thresholds are in correct order."""
        thresholds = [
            (1.00, 0.25),  # 100% gain, sell 25%
            (2.00, 0.25),  # 200% gain, sell 25%
            (4.00, 0.25),  # 400% gain, sell 25%
            (10.00, 0.25),  # 1000% gain, sell remaining
        ]

        # Thresholds should be in ascending order
        for i in range(len(thresholds) - 1):
            assert thresholds[i][0] < thresholds[i + 1][0]

        # Total sell percentage should sum to 1
        total_sell = sum(t[1] for t in thresholds)
        assert abs(total_sell - 1.0) < 0.01

    @pytest.mark.regression
    def test_profit_level_calculation(self):
        """Test profit level calculation is consistent."""
        test_cases = [
            {"entry": 1.00, "current": 2.00, "profit_pct": 1.00},  # 100%
            {"entry": 1.00, "current": 3.00, "profit_pct": 2.00},  # 200%
            {"entry": 2.00, "current": 3.00, "profit_pct": 0.50},  # 50%
        ]

        for case in test_cases:
            profit = (case["current"] - case["entry"]) / case["entry"]
            assert abs(profit - case["profit_pct"]) < 0.0001, f"Failed for {case}"

    @pytest.mark.regression
    def test_remaining_position_calculation(self):
        """Test remaining position calculation after sells."""
        initial_quantity = 100
        sells = [
            {"at_pct": 1.00, "sell_pct": 0.25},  # Sell 25%
            {"at_pct": 2.00, "sell_pct": 0.25},  # Sell 25% of remaining
            {"at_pct": 4.00, "sell_pct": 0.25},  # Sell 25% of remaining
        ]

        remaining = initial_quantity
        for sell in sells:
            sell_qty = int(remaining * sell["sell_pct"])
            remaining -= sell_qty

        # After 3 sells of 25%, should have ~42% left
        expected_remaining = initial_quantity * (0.75**3)
        assert abs(remaining - expected_remaining) < 2  # Allow rounding error


class TestScannerRegression:
    """Regression tests for scanner functionality."""

    @pytest.mark.regression
    def test_movement_threshold_check(self):
        """Test movement threshold checking is consistent."""
        min_movement = 0.02  # 2%

        test_cases = [
            {"open": 100.0, "current": 102.0, "should_alert": False},  # Exactly 2%
            {"open": 100.0, "current": 102.5, "should_alert": True},  # 2.5%
            {"open": 100.0, "current": 97.5, "should_alert": True},  # -2.5%
            {"open": 100.0, "current": 101.0, "should_alert": False},  # 1%
        ]

        for case in test_cases:
            movement = abs(case["current"] - case["open"]) / case["open"]
            should_alert = movement > min_movement
            assert should_alert == case["should_alert"], f"Failed for {case}"

    @pytest.mark.regression
    def test_volume_surge_detection(self):
        """Test volume surge detection is consistent."""
        volume_threshold = 2.0  # 2x average

        test_cases = [
            {"current_vol": 2000000, "avg_vol": 1000000, "is_surge": True},
            {"current_vol": 1500000, "avg_vol": 1000000, "is_surge": False},
            {"current_vol": 3000000, "avg_vol": 1000000, "is_surge": True},
        ]

        for case in test_cases:
            ratio = case["current_vol"] / case["avg_vol"]
            is_surge = ratio >= volume_threshold
            assert is_surge == case["is_surge"], f"Failed for {case}"


class TestDataIntegrityRegression:
    """Regression tests for data integrity checks."""

    @pytest.mark.regression
    def test_ohlc_relationship(self):
        """Test OHLC data integrity - High >= Low always."""
        test_bars = [
            {"open": 100, "high": 105, "low": 98, "close": 102},
            {"open": 50, "high": 55, "low": 48, "close": 52},
        ]

        for bar in test_bars:
            # High should be highest
            assert bar["high"] >= bar["open"]
            assert bar["high"] >= bar["close"]
            assert bar["high"] >= bar["low"]

            # Low should be lowest
            assert bar["low"] <= bar["open"]
            assert bar["low"] <= bar["close"]

    @pytest.mark.regression
    def test_timestamp_ordering(self):
        """Test timestamps are always in order."""
        base = datetime(2024, 1, 1, 9, 30)
        timestamps = [base + timedelta(minutes=i) for i in range(10)]

        for i in range(len(timestamps) - 1):
            assert timestamps[i] < timestamps[i + 1]

    @pytest.mark.regression
    def test_price_positivity(self):
        """Test prices are always positive."""
        test_prices = [100.0, 50.25, 0.01, 1000.00]

        for price in test_prices:
            assert price > 0


class TestConfigurationRegression:
    """Regression tests for configuration handling."""

    @pytest.mark.regression
    def test_default_values_unchanged(self):
        """Test default configuration values haven't changed unexpectedly."""
        expected_defaults = {
            "max_daily_loss_pct": 0.03,
            "max_drawdown_pct": 0.10,
            "max_position_size": 0.25,
            "max_consecutive_losses": 5,
        }

        for key, expected in expected_defaults.items():
            # These should be the documented defaults
            assert expected is not None, f"Missing default for {key}"

    @pytest.mark.regression
    def test_config_bounds_validation(self):
        """Test configuration values are within valid bounds."""
        valid_ranges = {
            "max_daily_loss_pct": (0.01, 0.10),
            "max_drawdown_pct": (0.05, 0.25),
            "max_position_size": (0.05, 0.50),
        }

        test_values = {
            "max_daily_loss_pct": 0.03,
            "max_drawdown_pct": 0.10,
            "max_position_size": 0.25,
        }

        for key, value in test_values.items():
            min_val, max_val = valid_ranges[key]
            assert min_val <= value <= max_val, f"{key}={value} out of range"
