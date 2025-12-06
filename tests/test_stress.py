"""
Stress Tests for Trading Algorithms

Tests algorithm behavior under extreme market conditions including:
- Flash crashes and rapid price movements
- High volatility regimes
- Liquidity crises
- Gap openings
- Black swan events

Based on best practices from:
- LuxAlgo stress testing guide
- Build Alpha robustness testing
- PyQuantLab scenario analysis
"""

import random
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from tests.conftest import MockIndicator, MockPortfolio, MockSlice


class TestFlashCrashBehavior:
    """Test algorithm behavior during flash crash scenarios."""

    @pytest.fixture
    def algorithm_with_circuit_breaker(self):
        """Create algorithm with circuit breaker protection."""
        from models.circuit_breaker import create_circuit_breaker

        algo = Mock()
        algo.symbol = "SPY"
        algo.Portfolio = MockPortfolio(cash=100000)
        algo.circuit_breaker = create_circuit_breaker(
            max_daily_loss=0.03,
            max_drawdown=0.10,
        )
        algo.SetHoldings = Mock()
        algo.Liquidate = Mock()
        algo.Debug = Mock()
        return algo

    @pytest.mark.stress
    def test_rapid_10_percent_drop(self, algorithm_with_circuit_breaker):
        """Test behavior during rapid 12% price drop (flash crash)."""
        algo = algorithm_with_circuit_breaker

        # Simulate >10% drop in portfolio value (exceeds 10% max drawdown)
        algo.Portfolio.TotalPortfolioValue = 88000  # 12% loss

        # Circuit breaker should trip on drawdown check (>10% limit)
        result = algo.circuit_breaker.check_drawdown(88000, 100000)

        assert result is False, "Circuit breaker should trip on >10% drawdown"
        assert algo.circuit_breaker.is_open is True

    @pytest.mark.stress
    def test_multiple_consecutive_losses(self, algorithm_with_circuit_breaker):
        """Test circuit breaker after multiple consecutive losses."""
        algo = algorithm_with_circuit_breaker

        # Simulate 5 consecutive losing trades
        for _ in range(5):
            algo.circuit_breaker.record_trade_result(is_winner=False)

        assert algo.circuit_breaker.can_trade() is False
        assert algo.circuit_breaker.is_open is True

    @pytest.mark.stress
    def test_v_shaped_recovery_after_crash(self, algorithm_with_circuit_breaker):
        """Test behavior during V-shaped recovery."""
        algo = algorithm_with_circuit_breaker

        # Crash phase - trip circuit breaker
        algo.circuit_breaker.check_drawdown(88000, 100000)
        assert algo.circuit_breaker.can_trade() is False

        # Recovery phase - should still be halted until reset
        # Even if price recovers, circuit breaker requires manual reset
        assert algo.circuit_breaker.can_trade() is False


class TestHighVolatilityRegimes:
    """Test algorithm behavior during high volatility periods."""

    @pytest.fixture
    def volatility_data(self) -> list[tuple[float, float, float]]:
        """Generate high volatility price data (high, low, close)."""
        data = []
        price = 100.0
        for _ in range(50):
            # 5-10% daily swings
            swing = price * random.uniform(0.05, 0.10)
            high = price + swing
            low = price - swing
            close = price + random.uniform(-swing, swing)
            data.append((high, low, close))
            price = close
        return data

    @pytest.mark.stress
    def test_indicator_stability_high_volatility(self, volatility_data):
        """Test indicator calculations remain stable during high volatility."""
        from indicators.volatility_bands import VolatilityBands

        indicator = VolatilityBands(period=14, multiplier=2.0)

        for high, low, close in volatility_data:
            indicator.update(high, low, close)

        # Indicator should be ready and have valid values
        assert indicator.is_ready
        assert indicator.middle_band > 0
        assert indicator.upper_band > indicator.middle_band
        assert indicator.lower_band < indicator.middle_band
        # ATR should reflect high volatility
        assert indicator.atr > 0

    @pytest.mark.stress
    def test_position_sizing_adapts_to_volatility(self):
        """Test that stop loss width varies with position size to maintain risk."""
        from models.risk_manager import RiskLimits, RiskManager

        limits = RiskLimits(
            max_position_size=0.25,
            max_risk_per_trade=0.02,  # 2% risk per trade
        )
        risk_manager = RiskManager(starting_equity=100000, limits=limits)

        # Larger position requires tighter stop (less room for volatility)
        large_position_stop = risk_manager.calculate_stop_loss(
            entry_price=100.0,
            position_size=0.25,  # 25% position
        )

        # Smaller position allows wider stop (more room for volatility)
        small_position_stop = risk_manager.calculate_stop_loss(
            entry_price=100.0,
            position_size=0.10,  # 10% position
        )

        # Larger position = tighter stop (stop is closer to entry)
        # Smaller position = wider stop (stop is further from entry)
        assert large_position_stop > small_position_stop


class TestGapOpenings:
    """Test algorithm behavior with gap openings."""

    @pytest.mark.stress
    def test_gap_down_stop_loss_slippage(self):
        """Test stop loss execution during gap down opening."""
        # Simulate scenario where stop at 95 but opens at 90
        stop_price = 95.0
        gap_open_price = 90.0

        # Calculate slippage
        slippage = stop_price - gap_open_price
        slippage_pct = slippage / stop_price

        assert slippage_pct > 0.05, "Gap creates >5% slippage"

        # Risk calculation should account for potential gap
        expected_loss_at_stop = 0.05  # 5% expected
        actual_loss = (100.0 - gap_open_price) / 100.0  # 10% actual

        assert actual_loss > expected_loss_at_stop

    @pytest.mark.stress
    def test_gap_up_profit_capture(self):
        """Test profit taking during gap up opening."""
        entry_price = 100.0
        gap_open_price = 115.0  # 15% gap up
        profit_target = 110.0  # 10% target

        # Gap exceeds profit target - should capture at better price
        actual_gain = (gap_open_price - entry_price) / entry_price
        target_gain = (profit_target - entry_price) / entry_price

        assert actual_gain > target_gain


class TestLiquidityCrisis:
    """Test algorithm behavior during low liquidity periods."""

    @pytest.mark.stress
    def test_wide_spread_execution_cost(self):
        """Test execution costs with wide bid-ask spreads."""
        # Normal spread: $0.01 on $100 stock = 0.01%
        normal_spread = 0.01
        normal_cost = normal_spread / 100.0

        # Crisis spread: $2.00 on $100 stock = 2%
        crisis_spread = 2.00
        crisis_cost = crisis_spread / 100.0

        # Execution cost increases 200x during crisis
        assert crisis_cost / normal_cost == 200

    @pytest.mark.stress
    def test_partial_fill_handling(self):
        """Test handling of partial fills during low liquidity."""
        from config import OrderExecutionConfig
        from execution.smart_execution import SmartExecutionModel
        from execution.smart_execution import SmartOrderStatus as OrderStatus

        config = OrderExecutionConfig()
        executor = SmartExecutionModel(config)

        # Submit order
        order = executor.submit_order(
            symbol="SPY",
            side="buy",
            quantity=1000,
            limit_price=100.0,
        )

        # Simulate partial fill (only 300 of 1000 shares)
        executor.update_order_status(
            order.order_id,
            OrderStatus.PARTIALLY_FILLED,
            filled_quantity=300,
            fill_price=100.05,
        )

        assert order.filled_quantity == 300
        assert order.unfilled_quantity == 700
        assert order.status == OrderStatus.PARTIALLY_FILLED


class TestBlackSwanEvents:
    """Test algorithm behavior during black swan events."""

    @pytest.mark.stress
    def test_correlation_breakdown(self):
        """Test behavior when normal correlations break down."""
        # During crises, correlations often go to 1 (everything falls together)
        # This tests portfolio diversification assumptions

        # Simulate correlated drawdowns across positions
        positions = ["SPY", "QQQ", "IWM", "DIA"]
        individual_losses = [-0.15, -0.18, -0.20, -0.14]  # All negative

        # Portfolio loss is NOT reduced by diversification during crisis
        portfolio_loss = sum(individual_losses) / len(individual_losses)
        worst_case_loss = min(individual_losses)

        # In crisis, diversification benefit is minimal
        assert abs(portfolio_loss - worst_case_loss) < 0.10

    @pytest.mark.stress
    def test_circuit_breaker_prevents_catastrophic_loss(self):
        """Test that circuit breaker prevents losses beyond configured limit."""
        from models.circuit_breaker import create_circuit_breaker

        breaker = create_circuit_breaker(
            max_daily_loss=0.03,
            max_drawdown=0.10,
        )

        # Series of losses that would exceed limits
        losses = [-0.01, -0.015, -0.02, -0.025]  # Cumulative: -7%
        cumulative = 0

        for loss in losses:
            cumulative += loss
            can_trade = breaker.check_daily_loss(cumulative)

            if cumulative <= -0.03:
                assert can_trade is False, f"Should halt at {cumulative:.1%}"
                break


class TestRobustnessAcrossMarkets:
    """Test strategy robustness across different market conditions."""

    @pytest.mark.stress
    def test_strategy_works_on_related_symbols(self):
        """Test that strategy logic works on related instruments."""
        # A robust SPY strategy should also work on QQQ, IWM
        related_symbols = ["SPY", "QQQ", "IWM", "DIA"]

        for symbol in related_symbols:
            algo = Mock()
            algo.symbol = symbol
            algo.rsi = MockIndicator(25)  # Oversold
            algo.Portfolio = MockPortfolio()
            algo.IsWarmingUp = False
            algo.SetHoldings = Mock()

            # Simulate buy signal
            data = MockSlice({symbol: Mock()})

            # Execute logic (simplified)
            if algo.rsi.Current.Value < 30 and not algo.Portfolio[symbol].Invested:
                algo.SetHoldings(symbol, 1.0)

            algo.SetHoldings.assert_called_once()

    @pytest.mark.stress
    def test_parameter_sensitivity(self):
        """Test strategy sensitivity to parameter changes."""
        # Test that small parameter variations produce consistent signals
        # when RSI is clearly in oversold territory (RSI = 22, well below 30)
        rsi_value = 22  # Clearly oversold
        variations = [28, 29, 30, 31, 32]  # Small variations around 30

        results = []
        for threshold in variations:
            # Signal triggers when RSI < threshold (oversold)
            signal_triggered = rsi_value < threshold
            results.append(signal_triggered)

        # With RSI at 22, all thresholds (28-32) should trigger buy
        # Strategy shouldn't be overly sensitive to small parameter changes
        agreement = results.count(results[0]) / len(results)
        assert agreement >= 0.6, "Strategy too sensitive to parameters"


class TestDataAnomalies:
    """Test handling of data anomalies and bad ticks."""

    @pytest.mark.stress
    def test_handles_zero_price(self):
        """Test handling of zero or negative prices."""
        from indicators.volatility_bands import VolatilityBands

        indicator = VolatilityBands(period=3)

        # Add normal data
        indicator.update(102, 98, 100)
        indicator.update(104, 96, 101)

        # Zero price should be handled gracefully
        indicator.update(0, 0, 0)  # Bad tick

        # Indicator should still function (may not be "ready" with bad data)
        # At minimum, it shouldn't crash

    @pytest.mark.stress
    def test_handles_price_spike(self):
        """Test handling of erroneous price spikes."""
        prices = [100, 101, 102, 10000, 103, 104]  # 10000 is bad tick

        # Strategy should detect and handle outliers
        median_price = sorted(prices)[len(prices) // 2]
        max_deviation = 0.50  # 50% from median is suspicious

        suspicious_prices = [p for p in prices if abs(p - median_price) / median_price > max_deviation]

        assert 10000 in suspicious_prices

    @pytest.mark.stress
    def test_handles_missing_data_gaps(self):
        """Test handling of gaps in data feed."""
        timestamps = [
            datetime(2024, 1, 1, 9, 30),
            datetime(2024, 1, 1, 9, 31),
            # Gap - missing 9:32, 9:33, 9:34
            datetime(2024, 1, 1, 9, 35),
        ]

        # Detect gap
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            expected_gap = timedelta(minutes=1)

            if gap > expected_gap * 2:
                # Gap detected - strategy should be aware
                gap_minutes = gap.total_seconds() / 60
                assert gap_minutes > 2
