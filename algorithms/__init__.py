"""
QuantConnect Trading Algorithms

This package contains trading algorithm implementations for the QuantConnect platform.

Available algorithms:
- BasicBuyAndHoldAlgorithm: Simple buy-and-hold strategy for SPY
- SimpleMomentumAlgorithm: RSI-based momentum trading strategy
- HybridOptionsBot: Semi-autonomous hybrid options trading system
- OptionsTradingBot: Comprehensive options trading with LLM analysis
- WheelStrategyAlgorithm: Wheel strategy (sell puts, covered calls)

Note: These algorithms require QuantConnect's AlgorithmImports and are designed
to run within the LEAN engine. For testing, use mocks via unittest.mock.patch.

Layer: 4 (Applications)
May import from: Layers 0-3 (all lower layers)
This is the top layer - nothing should import from algorithms.
"""

import importlib
from types import ModuleType


def __getattr__(name: str) -> ModuleType:
    """Lazy load algorithm submodules.

    This allows `algorithms.hybrid_options_bot` to work for patching
    while deferring the actual import until access time.
    """
    submodules = [
        "basic_buy_hold",
        "simple_momentum",
        "hybrid_options_bot",
        "options_trading_bot",
        "wheel_strategy",
    ]
    if name in submodules:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "basic_buy_hold",
    "hybrid_options_bot",
    "options_trading_bot",
    "simple_momentum",
    "wheel_strategy",
]
