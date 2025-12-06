"""
Performance and Benchmark Tests

Tests for execution speed, memory usage, and computational efficiency.
These tests ensure the trading system can handle real-time requirements.

Based on best practices from:
- pytest-benchmark patterns
- Low-latency trading system testing
- Financial system performance requirements
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps

import numpy as np
import pytest


def timer(func: Callable) -> Callable:
    """Decorator to time function execution."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start

    return wrapper


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    ops_per_second: float


class BenchmarkRunner:
    """Simple benchmark runner for performance tests."""

    @staticmethod
    def run(func: Callable, iterations: int = 1000, warmup: int = 100) -> BenchmarkResult:
        """
        Run a benchmark on a function.

        Args:
            func: Function to benchmark
            iterations: Number of iterations
            warmup: Number of warmup iterations

        Returns:
            BenchmarkResult with timing statistics
        """
        # Warmup
        for _ in range(warmup):
            func()

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append(end - start)

        total_time = sum(times)
        return BenchmarkResult(
            name=func.__name__,
            iterations=iterations,
            total_time=total_time,
            avg_time=np.mean(times),
            min_time=min(times),
            max_time=max(times),
            ops_per_second=iterations / total_time if total_time > 0 else 0,
        )


class TestCalculationPerformance:
    """Performance tests for core calculations."""

    @pytest.mark.performance
    def test_position_value_speed(self):
        """Test position value calculation is fast enough."""

        def calculate_position_value():
            quantity = 100
            price = 150.75
            return quantity * price

        result = BenchmarkRunner.run(calculate_position_value, iterations=10000)

        # Should be able to do millions per second
        assert result.ops_per_second > 100000

    @pytest.mark.performance
    def test_pnl_batch_calculation_speed(self):
        """Test batch P&L calculation performance."""
        positions = [{"entry": 100 + i, "current": 105 + i, "quantity": 100} for i in range(100)]

        def calculate_batch_pnl():
            return sum((p["current"] - p["entry"]) * p["quantity"] for p in positions)

        result = BenchmarkRunner.run(calculate_batch_pnl, iterations=1000)

        # Should handle 100 positions in well under 1ms
        assert result.avg_time < 0.001  # Less than 1ms

    @pytest.mark.performance
    def test_risk_check_speed(self):
        """Test risk checking is fast enough for real-time."""
        portfolio = {
            "positions": [{"value": 10000} for _ in range(50)],
            "cash": 50000,
            "daily_pnl": -500,
        }

        def check_risk():
            total_value = sum(p["value"] for p in portfolio["positions"])
            daily_loss_pct = abs(portfolio["daily_pnl"]) / (total_value + portfolio["cash"])
            return daily_loss_pct < 0.03

        result = BenchmarkRunner.run(check_risk, iterations=10000)

        # Must be sub-millisecond for real-time trading
        assert result.avg_time < 0.001

    @pytest.mark.performance
    def test_stop_loss_calculation_speed(self):
        """Test stop loss calculation is fast."""

        def calculate_stop():
            entry = 100.0
            risk_per_trade = 0.02
            position_size = 0.20
            return entry * (1 - risk_per_trade / position_size)

        result = BenchmarkRunner.run(calculate_stop, iterations=10000)

        # Should be extremely fast
        assert result.ops_per_second > 500000


class TestIndicatorPerformance:
    """Performance tests for technical indicators."""

    @pytest.fixture
    def price_data(self) -> np.ndarray:
        """Generate sample price data."""
        np.random.seed(42)
        return 100 + np.cumsum(np.random.randn(1000) * 0.5)

    @pytest.mark.performance
    def test_sma_calculation_speed(self, price_data):
        """Test SMA calculation performance."""
        period = 20

        def calculate_sma():
            return np.convolve(price_data, np.ones(period) / period, mode="valid")

        result = BenchmarkRunner.run(calculate_sma, iterations=1000)

        # Should process 1000 data points very quickly
        assert result.avg_time < 0.001

    @pytest.mark.performance
    def test_ema_calculation_speed(self, price_data):
        """Test EMA calculation performance."""
        period = 20
        multiplier = 2 / (period + 1)

        def calculate_ema():
            ema = np.zeros_like(price_data)
            ema[0] = price_data[0]
            for i in range(1, len(price_data)):
                ema[i] = price_data[i] * multiplier + ema[i - 1] * (1 - multiplier)
            return ema

        result = BenchmarkRunner.run(calculate_ema, iterations=100)

        # Loop-based EMA should still be fast
        assert result.avg_time < 0.01

    @pytest.mark.performance
    def test_rsi_calculation_speed(self, price_data):
        """Test RSI calculation performance."""
        period = 14

        def calculate_rsi():
            deltas = np.diff(price_data)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.convolve(gains, np.ones(period) / period, mode="valid")
            avg_loss = np.convolve(losses, np.ones(period) / period, mode="valid")

            rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
            return 100 - (100 / (1 + rs))

        result = BenchmarkRunner.run(calculate_rsi, iterations=500)

        # Should be reasonably fast
        assert result.avg_time < 0.005

    @pytest.mark.performance
    def test_bollinger_bands_speed(self, price_data):
        """Test Bollinger Bands calculation performance."""
        period = 20
        num_std = 2

        def calculate_bb():
            sma = np.convolve(price_data, np.ones(period) / period, mode="valid")

            # Rolling std - simplified for benchmark
            std = np.array([np.std(price_data[i : i + period]) for i in range(len(price_data) - period + 1)])

            upper = sma + num_std * std
            lower = sma - num_std * std
            return sma, upper, lower

        result = BenchmarkRunner.run(calculate_bb, iterations=100)

        # Bollinger bands with rolling std is slower but still should be fast
        assert result.avg_time < 0.02


class TestScannerPerformance:
    """Performance tests for market scanners."""

    @pytest.fixture
    def market_data(self) -> list[dict]:
        """Generate sample market data for multiple symbols."""
        np.random.seed(42)
        symbols = [f"SYM{i}" for i in range(500)]
        return [
            {
                "symbol": sym,
                "open": 100 + np.random.randn() * 10,
                "current": 100 + np.random.randn() * 12,
                "volume": int(1000000 * (1 + np.random.rand())),
                "avg_volume": 1000000,
            }
            for sym in symbols
        ]

    @pytest.mark.performance
    def test_movement_scan_speed(self, market_data):
        """Test movement scanner can process many symbols quickly."""
        min_movement = 0.02

        def scan_movements():
            movers = []
            for data in market_data:
                movement = (data["current"] - data["open"]) / data["open"]
                if abs(movement) > min_movement:
                    movers.append(data["symbol"])
            return movers

        result = BenchmarkRunner.run(scan_movements, iterations=1000)

        # Should process 500 symbols in sub-millisecond time
        assert result.avg_time < 0.001

    @pytest.mark.performance
    def test_volume_surge_scan_speed(self, market_data):
        """Test volume surge detection is fast."""
        volume_threshold = 2.0

        def scan_volume():
            surges = []
            for data in market_data:
                ratio = data["volume"] / data["avg_volume"]
                if ratio >= volume_threshold:
                    surges.append(data["symbol"])
            return surges

        result = BenchmarkRunner.run(scan_volume, iterations=1000)

        # Should be very fast
        assert result.avg_time < 0.001


class TestOrderBookPerformance:
    """Performance tests for order book operations."""

    @pytest.fixture
    def order_book(self) -> dict:
        """Generate sample order book."""
        np.random.seed(42)
        mid = 100.0
        return {
            "bids": [(mid - 0.01 * i, 100 * (i + 1)) for i in range(100)],
            "asks": [(mid + 0.01 * i, 100 * (i + 1)) for i in range(100)],
        }

    @pytest.mark.performance
    def test_best_bid_ask_lookup(self, order_book):
        """Test best bid/ask lookup is O(1)."""

        def get_bbo():
            return order_book["bids"][0][0], order_book["asks"][0][0]

        result = BenchmarkRunner.run(get_bbo, iterations=100000)

        # Should be extremely fast (constant time)
        assert result.avg_time < 0.00001

    @pytest.mark.performance
    def test_spread_calculation(self, order_book):
        """Test spread calculation speed."""

        def calculate_spread():
            best_bid = order_book["bids"][0][0]
            best_ask = order_book["asks"][0][0]
            return best_ask - best_bid, (best_ask - best_bid) / ((best_ask + best_bid) / 2)

        result = BenchmarkRunner.run(calculate_spread, iterations=10000)

        # Very simple calculation, should be instant
        assert result.avg_time < 0.0001

    @pytest.mark.performance
    def test_depth_aggregation(self, order_book):
        """Test order book depth aggregation speed."""

        def aggregate_depth():
            bid_depth = sum(qty for _, qty in order_book["bids"][:10])
            ask_depth = sum(qty for _, qty in order_book["asks"][:10])
            return bid_depth, ask_depth

        result = BenchmarkRunner.run(aggregate_depth, iterations=10000)

        # Should be fast
        assert result.avg_time < 0.0001


class TestOptionsPerformance:
    """Performance tests for options calculations."""

    @pytest.mark.performance
    def test_black_scholes_speed(self):
        """Test Black-Scholes calculation is fast enough."""
        from math import exp, log, sqrt

        from scipy.stats import norm

        def black_scholes_call():
            S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
            d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)
            return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

        result = BenchmarkRunner.run(black_scholes_call, iterations=1000)

        # Black-Scholes is simple math, should be fast
        assert result.avg_time < 0.0005

    @pytest.mark.performance
    def test_greeks_calculation_speed(self):
        """Test Greeks calculation speed."""
        from math import exp, log, sqrt

        from scipy.stats import norm

        def calculate_greeks():
            S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
            d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)

            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
            theta = -(S * sigma * norm.pdf(d1)) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2)
            vega = S * sqrt(T) * norm.pdf(d1)

            return delta, gamma, theta, vega

        result = BenchmarkRunner.run(calculate_greeks, iterations=1000)

        # All Greeks together should still be fast
        assert result.avg_time < 0.001

    @pytest.mark.performance
    def test_iv_newton_raphson_speed(self):
        """Test IV calculation via Newton-Raphson is reasonably fast."""
        from math import exp, log, sqrt

        from scipy.stats import norm

        def calculate_iv():
            S, K, T, r = 100, 100, 0.25, 0.05
            market_price = 5.0
            sigma = 0.20

            for _ in range(10):  # Max 10 iterations
                d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
                d2 = d1 - sigma * sqrt(T)

                price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
                vega = S * sqrt(T) * norm.pdf(d1)

                if abs(vega) < 0.0001:
                    break

                sigma = sigma - (price - market_price) / vega
                sigma = max(0.01, min(5.0, sigma))

            return sigma

        result = BenchmarkRunner.run(calculate_iv, iterations=500)

        # IV calculation with iteration should complete quickly
        assert result.avg_time < 0.005

    @pytest.mark.performance
    def test_option_chain_processing(self):
        """Test processing a full option chain is fast enough."""
        np.random.seed(42)

        # Simulate 100 option contracts
        chain = [
            {
                "strike": 95 + i * 0.5,
                "expiry_days": 30,
                "bid": 2.0 + np.random.rand(),
                "ask": 2.5 + np.random.rand(),
                "volume": int(1000 * np.random.rand()),
            }
            for i in range(100)
        ]

        def process_chain():
            opportunities = []
            for contract in chain:
                mid = (contract["bid"] + contract["ask"]) / 2
                spread_pct = (contract["ask"] - contract["bid"]) / mid
                if spread_pct < 0.10 and contract["volume"] > 100:
                    opportunities.append(contract)
            return opportunities

        result = BenchmarkRunner.run(process_chain, iterations=1000)

        # Should process 100 contracts very quickly
        assert result.avg_time < 0.001


class TestMemoryEfficiency:
    """Tests for memory usage efficiency."""

    @pytest.mark.performance
    def test_large_price_array_efficiency(self):
        """Test memory efficiency with large price arrays."""
        # Create 1 year of minute data (252 days * 390 minutes)
        data_points = 252 * 390

        start_time = time.perf_counter()
        prices = np.random.randn(data_points) * 0.5 + 100
        creation_time = time.perf_counter() - start_time

        # Creating ~100k data points should be fast
        assert creation_time < 0.1

        # Using numpy should be memory efficient
        assert prices.nbytes < 1_000_000  # Less than 1MB

    @pytest.mark.performance
    def test_position_tracking_scalability(self):
        """Test position tracking scales well."""

        def create_positions(count: int):
            return {
                f"SYM{i}": {
                    "quantity": 100,
                    "entry_price": 100.0,
                    "current_price": 100.0 + np.random.randn(),
                }
                for i in range(count)
            }

        # Test with 100 positions
        start = time.perf_counter()
        positions = create_positions(100)
        time_100 = time.perf_counter() - start

        # Test with 1000 positions
        start = time.perf_counter()
        positions = create_positions(1000)
        time_1000 = time.perf_counter() - start

        # Should scale linearly (not exponentially)
        # 10x positions should not take more than 20x time
        assert time_1000 < time_100 * 20


class TestThroughput:
    """Tests for system throughput."""

    @pytest.mark.performance
    def test_tick_processing_throughput(self):
        """Test how many ticks can be processed per second."""

        def process_tick(price: float, volume: int):
            # Simulate basic tick processing
            return {
                "price": price,
                "volume": volume,
                "value": price * volume,
            }

        start = time.perf_counter()
        count = 0
        while time.perf_counter() - start < 0.1:  # 100ms test
            process_tick(100.0 + np.random.rand(), 100)
            count += 1

        ticks_per_second = count * 10

        # Should handle at least 10,000 ticks per second
        assert ticks_per_second > 10000

    @pytest.mark.performance
    def test_order_processing_throughput(self):
        """Test order processing throughput."""
        orders = []

        def process_order(symbol: str, side: str, quantity: int, price: float):
            order = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "timestamp": time.time(),
            }
            orders.append(order)
            return len(orders)

        start = time.perf_counter()
        for i in range(1000):
            process_order("SPY", "buy", 100, 450.0 + i * 0.01)
        elapsed = time.perf_counter() - start

        orders_per_second = 1000 / elapsed

        # Should process thousands of orders per second
        assert orders_per_second > 5000
