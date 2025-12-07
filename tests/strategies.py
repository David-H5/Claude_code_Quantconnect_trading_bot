"""
Property-Based Test Strategies

Hypothesis strategies for property-based testing of the trading bot.
These strategies generate valid test data for fuzzing and invariant testing.

UPGRADE-015: Test Framework Enhancement - Property-Based Testing

Usage:
    from tests.strategies import valid_price, valid_quantity, valid_order
    from hypothesis import given

    @given(price=valid_price())
    def test_price_handling(price):
        # Test with any valid price
        assert process_price(price) >= 0
"""

from datetime import datetime, timedelta
from typing import Any

try:
    from hypothesis import strategies as st
    from hypothesis.strategies import SearchStrategy
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Provide stub for when hypothesis is not installed
    SearchStrategy = Any

    class st:
        """Stub strategies when hypothesis not available."""
        @staticmethod
        def floats(*args, **kwargs):
            raise ImportError("hypothesis not installed")

        @staticmethod
        def integers(*args, **kwargs):
            raise ImportError("hypothesis not installed")

        @staticmethod
        def text(*args, **kwargs):
            raise ImportError("hypothesis not installed")

        @staticmethod
        def sampled_from(*args, **kwargs):
            raise ImportError("hypothesis not installed")

        @staticmethod
        def fixed_dictionaries(*args, **kwargs):
            raise ImportError("hypothesis not installed")

        @staticmethod
        def builds(*args, **kwargs):
            raise ImportError("hypothesis not installed")

        @staticmethod
        def composite(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

        @staticmethod
        def lists(*args, **kwargs):
            raise ImportError("hypothesis not installed")

        @staticmethod
        def datetimes(*args, **kwargs):
            raise ImportError("hypothesis not installed")


# ============================================================================
# BASIC TRADING VALUE STRATEGIES
# ============================================================================


def valid_price(
    min_price: float = 0.01,
    max_price: float = 10000.0,
) -> SearchStrategy:
    """
    Strategy for generating valid stock prices.

    Excludes NaN, infinity, and negative values.
    """
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed. Run: pip install hypothesis")

    return st.floats(
        min_value=min_price,
        max_value=max_price,
        allow_nan=False,
        allow_infinity=False,
    )


def valid_quantity(
    min_qty: int = 1,
    max_qty: int = 100000,
) -> SearchStrategy:
    """Strategy for generating valid order quantities."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    return st.integers(min_value=min_qty, max_value=max_qty)


def valid_percentage(
    min_pct: float = 0.0,
    max_pct: float = 1.0,
) -> SearchStrategy:
    """Strategy for generating valid percentages (0.0 to 1.0)."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    return st.floats(
        min_value=min_pct,
        max_value=max_pct,
        allow_nan=False,
        allow_infinity=False,
    )


def valid_symbol() -> SearchStrategy:
    """Strategy for generating valid stock symbols."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    # Common symbols for realistic testing
    common_symbols = [
        "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
        "TSLA", "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS",
        "BAC", "XOM", "VZ", "ADBE", "CRM", "NFLX", "INTC", "AMD", "PYPL",
    ]
    return st.sampled_from(common_symbols)


def valid_order_side() -> SearchStrategy:
    """Strategy for generating order sides."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    return st.sampled_from(["buy", "sell"])


def valid_order_type() -> SearchStrategy:
    """Strategy for generating order types."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    return st.sampled_from(["market", "limit", "stop", "stop_limit"])


# ============================================================================
# COMPOSITE STRATEGIES
# ============================================================================


def valid_order(
    include_stop: bool = True,
) -> SearchStrategy:
    """
    Strategy for generating complete valid order dictionaries.

    Usage:
        @given(order=valid_order())
        def test_order_validation(order):
            result = validator.validate(order)
            assert result.is_valid
    """
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    order_types = ["market", "limit"]
    if include_stop:
        order_types.extend(["stop", "stop_limit"])

    return st.fixed_dictionaries({
        "symbol": valid_symbol(),
        "side": valid_order_side(),
        "order_type": st.sampled_from(order_types),
        "quantity": valid_quantity(),
        "price": st.one_of(st.none(), valid_price()),
        "stop_price": st.one_of(st.none(), valid_price()),
    })


def valid_position() -> SearchStrategy:
    """Strategy for generating valid position dictionaries."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    return st.fixed_dictionaries({
        "symbol": valid_symbol(),
        "quantity": st.integers(min_value=-10000, max_value=10000).filter(lambda x: x != 0),
        "average_price": valid_price(),
        "market_price": valid_price(),
    })


def valid_trade_bar() -> SearchStrategy:
    """Strategy for generating valid OHLCV bars."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    @st.composite
    def _trade_bar(draw):
        # Generate base price
        base = draw(valid_price(min_price=10.0, max_price=1000.0))

        # Generate OHLC ensuring high >= low, and open/close within range
        low = base * draw(st.floats(min_value=0.95, max_value=1.0))
        high = base * draw(st.floats(min_value=1.0, max_value=1.05))
        open_price = draw(st.floats(min_value=low, max_value=high))
        close_price = draw(st.floats(min_value=low, max_value=high))

        return {
            "symbol": draw(valid_symbol()),
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close_price, 2),
            "volume": draw(st.integers(min_value=1000, max_value=100000000)),
        }

    return _trade_bar()


# ============================================================================
# RISK MANAGEMENT STRATEGIES
# ============================================================================


def valid_risk_limits() -> SearchStrategy:
    """Strategy for generating valid risk limit configurations."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    return st.fixed_dictionaries({
        "max_position_size": valid_percentage(0.05, 0.50),
        "max_daily_loss": valid_percentage(0.01, 0.10),
        "max_drawdown": valid_percentage(0.05, 0.30),
        "max_risk_per_trade": valid_percentage(0.005, 0.05),
    })


def valid_circuit_breaker_config() -> SearchStrategy:
    """Strategy for generating valid circuit breaker configurations."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    return st.fixed_dictionaries({
        "max_daily_loss_pct": valid_percentage(0.01, 0.10),
        "max_drawdown_pct": valid_percentage(0.05, 0.30),
        "max_consecutive_losses": st.integers(min_value=2, max_value=10),
        "cooldown_minutes": st.integers(min_value=0, max_value=60),
        "require_human_reset": st.booleans(),
    })


# ============================================================================
# EDGE CASE STRATEGIES
# ============================================================================


def edge_case_prices() -> SearchStrategy:
    """Strategy for edge case prices (boundaries, near-zero, etc.)."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    return st.sampled_from([
        0.01,       # Minimum price
        0.001,      # Sub-penny
        0.0001,     # Micro-price
        1.0,        # Round number
        100.0,      # Common price
        999.99,     # Near round number
        1000.0,     # Large round number
        9999.99,    # Near max typical
        10000.0,    # Max typical
        0.0,        # Zero (invalid but test handling)
    ])


def edge_case_quantities() -> SearchStrategy:
    """Strategy for edge case quantities."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    return st.sampled_from([
        1,          # Minimum
        10,         # Small
        100,        # Standard lot
        1000,       # Large lot
        10000,      # Very large
        100000,     # Massive
        0,          # Zero (invalid but test handling)
        -1,         # Negative (invalid)
        -100,       # Negative lot
    ])


def edge_case_percentages() -> SearchStrategy:
    """Strategy for edge case percentages."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    return st.sampled_from([
        0.0,        # Zero
        0.0001,     # Near-zero
        0.01,       # 1%
        0.05,       # 5%
        0.10,       # 10%
        0.50,       # 50%
        0.99,       # Near 100%
        1.0,        # 100%
        1.01,       # Over 100% (invalid in some contexts)
        -0.01,      # Negative (invalid)
    ])


# ============================================================================
# TIME-BASED STRATEGIES
# ============================================================================


def valid_timestamp(
    min_days_ago: int = 365,
    max_days_ago: int = 0,
) -> SearchStrategy:
    """Strategy for generating valid timestamps within a range."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    now = datetime.now()
    min_date = now - timedelta(days=min_days_ago)
    max_date = now - timedelta(days=max_days_ago)

    return st.datetimes(min_value=min_date, max_value=max_date)


def market_hours_timestamp() -> SearchStrategy:
    """Strategy for generating timestamps during market hours (9:30 AM - 4:00 PM ET)."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    @st.composite
    def _market_hours(draw):
        # Generate a date
        base_date = draw(valid_timestamp(min_days_ago=30, max_days_ago=1))

        # Set to market hours (9:30 AM - 4:00 PM)
        hour = draw(st.integers(min_value=9, max_value=15))
        if hour == 9:
            minute = draw(st.integers(min_value=30, max_value=59))
        elif hour == 15:
            minute = draw(st.integers(min_value=0, max_value=59))
        else:
            minute = draw(st.integers(min_value=0, max_value=59))

        return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

    return _market_hours()


# ============================================================================
# SCENARIO STRATEGIES
# ============================================================================


def crash_scenario() -> SearchStrategy:
    """Strategy for generating market crash scenarios."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    return st.fixed_dictionaries({
        "initial_price": valid_price(min_price=50.0, max_price=500.0),
        "crash_pct": valid_percentage(0.10, 0.50),  # 10-50% crash
        "recovery_pct": valid_percentage(0.0, 0.30),  # 0-30% recovery
        "num_bars": st.integers(min_value=5, max_value=50),
    })


def volatility_scenario() -> SearchStrategy:
    """Strategy for generating high volatility scenarios."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    return st.fixed_dictionaries({
        "base_price": valid_price(min_price=50.0, max_price=500.0),
        "volatility": valid_percentage(0.02, 0.10),  # 2-10% daily volatility
        "trend": st.floats(min_value=-0.01, max_value=0.01),  # Daily trend
        "num_bars": st.integers(min_value=10, max_value=100),
    })


def gap_scenario() -> SearchStrategy:
    """Strategy for generating gap up/down scenarios."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    return st.fixed_dictionaries({
        "pre_gap_price": valid_price(min_price=50.0, max_price=500.0),
        "gap_pct": st.floats(min_value=-0.20, max_value=0.20).filter(
            lambda x: abs(x) >= 0.03  # At least 3% gap
        ),
        "direction": st.sampled_from(["up", "down"]),
    })


# ============================================================================
# AUDIT & COMPLIANCE STRATEGIES
# ============================================================================


def valid_trade_record() -> SearchStrategy:
    """Strategy for generating valid trade audit records."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    return st.fixed_dictionaries({
        "trade_id": st.text(
            alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            min_size=10,
            max_size=20,
        ),
        "symbol": valid_symbol(),
        "side": valid_order_side(),
        "quantity": valid_quantity(),
        "price": valid_price(),
        "commission": st.floats(min_value=0, max_value=100),
    })


def valid_risk_event() -> SearchStrategy:
    """Strategy for generating valid risk event records."""
    if not HYPOTHESIS_AVAILABLE:
        raise ImportError("hypothesis not installed")

    return st.fixed_dictionaries({
        "event_type": st.sampled_from([
            "CIRCUIT_BREAKER_TRIP",
            "RISK_LIMIT_BREACH",
            "POSITION_SIZE_EXCEEDED",
            "DAILY_LOSS_EXCEEDED",
            "DRAWDOWN_EXCEEDED",
        ]),
        "severity": st.sampled_from(["low", "medium", "high", "critical"]),
        "details": st.fixed_dictionaries({
            "value": valid_percentage(),
            "limit": valid_percentage(),
        }),
    })


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def assert_invariant(condition: bool, message: str):
    """Assert an invariant with a descriptive message."""
    assert condition, f"INVARIANT VIOLATION: {message}"


def check_price_invariants(bar: dict) -> bool:
    """Check OHLC price invariants."""
    return (
        bar["high"] >= bar["low"]
        and bar["high"] >= bar["open"]
        and bar["high"] >= bar["close"]
        and bar["low"] <= bar["open"]
        and bar["low"] <= bar["close"]
    )


def check_position_invariants(position: dict) -> bool:
    """Check position invariants."""
    return (
        position["quantity"] != 0
        and position["average_price"] > 0
        and position["market_price"] > 0
    )


def check_order_invariants(order: dict) -> bool:
    """Check order invariants."""
    if order["order_type"] == "limit":
        return order["price"] is not None and order["price"] > 0
    if order["order_type"] in ("stop", "stop_limit"):
        return order["stop_price"] is not None and order["stop_price"] > 0
    return True


__all__ = [
    # Status
    "HYPOTHESIS_AVAILABLE",
    # Basic strategies
    "valid_price",
    "valid_quantity",
    "valid_percentage",
    "valid_symbol",
    "valid_order_side",
    "valid_order_type",
    # Composite strategies
    "valid_order",
    "valid_position",
    "valid_trade_bar",
    # Risk management
    "valid_risk_limits",
    "valid_circuit_breaker_config",
    # Edge cases
    "edge_case_prices",
    "edge_case_quantities",
    "edge_case_percentages",
    # Time-based
    "valid_timestamp",
    "market_hours_timestamp",
    # Scenarios
    "crash_scenario",
    "volatility_scenario",
    "gap_scenario",
    # Audit
    "valid_trade_record",
    "valid_risk_event",
    # Helpers
    "assert_invariant",
    "check_price_invariants",
    "check_position_invariants",
    "check_order_invariants",
]
