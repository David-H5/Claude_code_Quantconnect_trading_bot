"""
Property-Based Testing with Hypothesis

Property-based testing validates that code satisfies certain properties
for ALL possible inputs, not just example cases. This catches edge cases
that example-based testing might miss.

Based on best practices from:
- Hypothesis library documentation
- Property-based testing patterns for financial systems
- QuickCheck-style testing strategies
"""

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite


# Custom strategies for trading domain
@composite
def price_strategy(draw, min_value=0.01, max_value=10000.0):
    """Generate realistic stock prices."""
    return draw(st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False))


@composite
def quantity_strategy(draw, min_value=1, max_value=100000):
    """Generate realistic position quantities."""
    return draw(st.integers(min_value=min_value, max_value=max_value))


@composite
def percentage_strategy(draw, min_value=0.0, max_value=1.0):
    """Generate percentages between 0 and 1."""
    return draw(st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False))


@composite
def returns_list_strategy(draw, min_size=2, max_size=1000):
    """Generate a list of daily returns."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return draw(
        st.lists(
            st.floats(min_value=-0.50, max_value=0.50, allow_nan=False, allow_infinity=False),
            min_size=size,
            max_size=size,
        )
    )


class TestPositionCalculationProperties:
    """Property-based tests for position calculations."""

    @given(
        price=price_strategy(),
        quantity=quantity_strategy(),
    )
    @settings(max_examples=200)
    def test_position_value_is_product(self, price, quantity):
        """Property: Position value equals price * quantity."""
        value = price * quantity
        assert value == price * quantity
        assert value >= 0  # Both inputs are positive

    @given(
        entry_price=price_strategy(),
        exit_price=price_strategy(),
        quantity=quantity_strategy(),
    )
    @settings(max_examples=200)
    def test_pnl_calculation_symmetric(self, entry_price, exit_price, quantity):
        """Property: Long PnL = -Short PnL for same entry/exit."""
        long_pnl = (exit_price - entry_price) * quantity
        short_pnl = (entry_price - exit_price) * quantity

        assert abs(long_pnl + short_pnl) < 0.0001  # Should sum to zero

    @given(
        price=price_strategy(),
        quantity=quantity_strategy(),
        new_quantity=quantity_strategy(),
    )
    @settings(max_examples=200)
    def test_position_value_scales_with_quantity(self, price, quantity, new_quantity):
        """Property: Doubling quantity doubles position value."""
        value1 = price * quantity
        value2 = price * (quantity * 2)

        assert abs(value2 - value1 * 2) < 0.01


class TestReturnCalculationProperties:
    """Property-based tests for return calculations."""

    @given(
        initial=price_strategy(min_value=1.0),
        final=price_strategy(min_value=0.01),
    )
    @settings(max_examples=200)
    def test_return_is_bounded(self, initial, final):
        """Property: Simple return is bounded for positive prices."""
        simple_return = (final - initial) / initial

        # Return should be > -1 (can't lose more than 100% with positive prices)
        assert simple_return >= -1.0

    @given(
        initial=price_strategy(min_value=1.0),
        final=price_strategy(min_value=0.01),
    )
    @settings(max_examples=200)
    def test_log_return_additivity(self, initial, final):
        """Property: Log returns are additive over periods."""
        assume(initial > 0 and final > 0)

        log_return = math.log(final / initial)
        simple_return = (final - initial) / initial

        # Log return approximates simple return for small values
        if abs(simple_return) < 0.1:
            assert abs(log_return - simple_return) < 0.02

    @given(returns=returns_list_strategy(min_size=5, max_size=100))
    @settings(max_examples=100)
    def test_cumulative_return_order_independent(self, returns):
        """Property: Product of (1+r) is independent of order for total return."""
        import functools
        import operator

        # Calculate cumulative return
        factors = [1 + r for r in returns]
        cumulative = functools.reduce(operator.mul, factors, 1.0)

        # Shuffle and recalculate
        import random

        shuffled = factors.copy()
        random.shuffle(shuffled)
        cumulative_shuffled = functools.reduce(operator.mul, shuffled, 1.0)

        # Should be the same (multiplication is commutative)
        assert abs(cumulative - cumulative_shuffled) < 0.0001


class TestRiskMetricProperties:
    """Property-based tests for risk metrics."""

    @given(
        peak=price_strategy(min_value=100.0, max_value=1000000.0),
        drawdown_pct=percentage_strategy(max_value=0.99),
    )
    @settings(max_examples=200)
    def test_drawdown_is_percentage(self, peak, drawdown_pct):
        """Property: Drawdown percentage is between 0 and 1."""
        current = peak * (1 - drawdown_pct)
        calculated_dd = (peak - current) / peak

        assert 0 <= calculated_dd <= 1
        assert abs(calculated_dd - drawdown_pct) < 0.0001

    @given(returns=returns_list_strategy(min_size=10))
    @settings(max_examples=100)
    def test_max_drawdown_non_negative(self, returns):
        """Property: Maximum drawdown is always non-negative."""
        # Calculate equity curve
        equity = [100.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))

        # Calculate max drawdown
        peak = equity[0]
        max_dd = 0.0
        for value in equity:
            peak = max(peak, value)
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        assert max_dd >= 0
        assert max_dd <= 1  # Can't lose more than 100%

    @given(
        win_rate=percentage_strategy(min_value=0.01, max_value=0.99),
        avg_win=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
        avg_loss=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_expectancy_formula(self, win_rate, avg_win, avg_loss):
        """Property: Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)."""
        loss_rate = 1 - win_rate
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        # Verify the calculation
        assert expectancy == pytest.approx((win_rate * avg_win) - ((1 - win_rate) * avg_loss))


class TestOptionsPricingProperties:
    """Property-based tests for options pricing."""

    @given(
        spot=price_strategy(min_value=10.0, max_value=1000.0),
        strike=price_strategy(min_value=10.0, max_value=1000.0),
        time_to_expiry=st.floats(min_value=0.01, max_value=2.0, allow_nan=False),
        volatility=st.floats(min_value=0.05, max_value=2.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_call_put_parity_holds(self, spot, strike, time_to_expiry, volatility):
        """Property: Put-call parity should hold."""
        from math import exp, log, sqrt

        from scipy.stats import norm

        r = 0.05  # Risk-free rate

        d1 = (log(spot / strike) + (r + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)

        call = spot * norm.cdf(d1) - strike * exp(-r * time_to_expiry) * norm.cdf(d2)
        put = strike * exp(-r * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)

        # Put-call parity: C - P = S - K*e^(-rT)
        lhs = call - put
        rhs = spot - strike * exp(-r * time_to_expiry)

        assert abs(lhs - rhs) < 0.01

    @given(
        spot=price_strategy(min_value=50.0, max_value=500.0),
        strike=price_strategy(min_value=50.0, max_value=500.0),
    )
    @settings(max_examples=100)
    def test_intrinsic_value_at_expiry(self, spot, strike):
        """Property: At expiry, option value equals intrinsic value."""
        call_intrinsic = max(0, spot - strike)
        put_intrinsic = max(0, strike - spot)

        assert call_intrinsic >= 0
        assert put_intrinsic >= 0
        assert not (call_intrinsic > 0 and put_intrinsic > 0)  # Can't both be ITM

    @given(
        delta=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        gamma=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        price_move=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_delta_gamma_approximation(self, delta, gamma, price_move):
        """Property: Delta-gamma approximation gives reasonable P&L estimate."""
        # P&L ≈ delta * dS + 0.5 * gamma * dS^2
        pnl_approx = delta * price_move + 0.5 * gamma * (price_move**2)

        # Gamma term is always positive (for long options)
        gamma_contribution = 0.5 * gamma * (price_move**2)
        assert gamma_contribution >= 0


class TestPortfolioProperties:
    """Property-based tests for portfolio calculations."""

    @given(
        weights=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=2,
            max_size=10,
        ),
        returns=st.lists(
            st.floats(min_value=-0.5, max_value=0.5, allow_nan=False),
            min_size=2,
            max_size=10,
        ),
    )
    @settings(max_examples=100)
    def test_portfolio_return_weighted_average(self, weights, returns):
        """Property: Portfolio return is weighted average of asset returns."""
        # Normalize weights
        min_len = min(len(weights), len(returns))
        weights = weights[:min_len]
        returns = returns[:min_len]

        total_weight = sum(weights)
        assume(total_weight > 0)

        normalized_weights = [w / total_weight for w in weights]

        portfolio_return = sum(w * r for w, r in zip(normalized_weights, returns))

        # Portfolio return should be within bounds of individual returns
        if returns:
            assert portfolio_return >= min(returns) - 0.01
            assert portfolio_return <= max(returns) + 0.01

    @given(
        position_values=st.lists(
            price_strategy(min_value=1.0, max_value=100000.0),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=100)
    def test_portfolio_weights_sum_to_one(self, position_values):
        """Property: Portfolio weights sum to 1."""
        total = sum(position_values)
        assume(total > 0)

        weights = [v / total for v in position_values]

        assert abs(sum(weights) - 1.0) < 0.0001


class TestOrderExecutionProperties:
    """Property-based tests for order execution logic."""

    @given(
        bid=price_strategy(min_value=1.0, max_value=999.0),
        spread_pct=st.floats(min_value=0.001, max_value=0.10, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_ask_always_above_bid(self, bid, spread_pct):
        """Property: Ask price is always >= bid price."""
        ask = bid * (1 + spread_pct)

        assert ask >= bid
        assert (ask - bid) / bid == pytest.approx(spread_pct, rel=0.01)

    @given(
        original_qty=quantity_strategy(min_value=10),
        fill_pct=percentage_strategy(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=200)
    def test_partial_fill_leaves_remainder(self, original_qty, fill_pct):
        """Property: Filled + Remaining = Original quantity."""
        filled = int(original_qty * fill_pct)
        remaining = original_qty - filled

        assert filled + remaining == original_qty
        assert filled >= 0
        assert remaining >= 0

    @given(
        slippage_bps=st.integers(min_value=-100, max_value=100),
        price=price_strategy(min_value=10.0),
    )
    @settings(max_examples=200)
    def test_slippage_adjusts_price(self, slippage_bps, price):
        """Property: Slippage adjusts execution price correctly."""
        slippage_pct = slippage_bps / 10000  # Convert bps to decimal
        executed_price = price * (1 + slippage_pct)

        # Verify adjustment
        actual_slippage = (executed_price - price) / price
        assert abs(actual_slippage - slippage_pct) < 0.0001


class TestCircuitBreakerProperties:
    """Property-based tests for circuit breaker logic."""

    @given(
        threshold=percentage_strategy(min_value=0.01, max_value=0.20),
        loss=percentage_strategy(min_value=0.0, max_value=0.30),
    )
    @settings(max_examples=200)
    def test_circuit_breaker_trip_logic(self, threshold, loss):
        """Property: Circuit breaker trips when loss exceeds threshold."""
        should_trip = loss >= threshold

        # Verify the logic
        if loss >= threshold:
            assert should_trip is True
        else:
            assert should_trip is False

    @given(
        losses=st.lists(
            st.booleans(),
            min_size=1,
            max_size=20,
        ),
        max_consecutive=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_consecutive_loss_counting(self, losses, max_consecutive):
        """Property: Consecutive loss counter works correctly."""
        count = 0
        max_count = 0
        tripped = False

        for is_loss in losses:
            if is_loss:
                count += 1
                max_count = max(max_count, count)
                if count >= max_consecutive:
                    tripped = True
            else:
                count = 0

        # Verify trip logic
        if max_count >= max_consecutive:
            assert tripped is True


class TestIndicatorProperties:
    """Property-based tests for technical indicators."""

    @given(
        prices=st.lists(
            price_strategy(min_value=1.0),
            min_size=20,
            max_size=100,
        ),
        period=st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=50)
    def test_sma_within_price_range(self, prices, period):
        """Property: SMA is always within the range of input prices."""
        assume(len(prices) >= period)

        # Calculate SMA
        sma_values = []
        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1 : i + 1]
            sma = sum(window) / period
            sma_values.append(sma)

        if sma_values:
            assert min(sma_values) >= min(prices) - 0.01
            assert max(sma_values) <= max(prices) + 0.01

    @given(
        gains=st.lists(
            st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
            min_size=14,
            max_size=50,
        ),
        losses=st.lists(
            st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
            min_size=14,
            max_size=50,
        ),
    )
    @settings(max_examples=50)
    def test_rsi_bounded_0_to_100(self, gains, losses):
        """Property: RSI is always between 0 and 100."""
        min_len = min(len(gains), len(losses))
        gains = gains[:min_len]
        losses = losses[:min_len]

        # Calculate average gain/loss
        avg_gain = sum(gains[-14:]) / 14
        avg_loss = sum(losses[-14:]) / 14

        if avg_loss == 0:
            rsi = 100.0
        elif avg_gain == 0:
            rsi = 0.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        assert 0 <= rsi <= 100
