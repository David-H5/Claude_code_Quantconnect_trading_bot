"""Strategy-related exceptions."""

from .base import TradingError


class StrategyError(TradingError):
    """Base class for strategy errors."""

    pass


class StrategyInitializationError(StrategyError):
    """Strategy failed to initialize."""

    def __init__(self, strategy_name: str, reason: str):
        super().__init__(
            f"Strategy '{strategy_name}' failed to initialize: {reason}",
            recoverable=False,
        )
        self.strategy_name = strategy_name
        self.reason = reason


class StrategyExecutionError(StrategyError):
    """Strategy execution failed."""

    def __init__(
        self,
        strategy_name: str,
        phase: str,
        reason: str,
    ):
        super().__init__(
            f"Strategy '{strategy_name}' failed during {phase}: {reason}",
            recoverable=True,
        )
        self.strategy_name = strategy_name
        self.phase = phase
        self.reason = reason


class SignalGenerationError(StrategyError):
    """Failed to generate trading signal."""

    def __init__(self, strategy_name: str, symbol: str, reason: str):
        super().__init__(
            f"Strategy '{strategy_name}' failed to generate signal " f"for {symbol}: {reason}",
            recoverable=True,
        )
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.reason = reason
        self.with_context(symbol=symbol)


class BacktestError(StrategyError):
    """Backtest execution failed."""

    def __init__(
        self,
        strategy_name: str,
        reason: str,
        backtest_id: str | None = None,
    ):
        msg = f"Backtest for '{strategy_name}' failed: {reason}"
        if backtest_id:
            msg = f"Backtest {backtest_id} for '{strategy_name}' failed: {reason}"
        super().__init__(msg, recoverable=False)
        self.strategy_name = strategy_name
        self.reason = reason
        self.backtest_id = backtest_id


class IndicatorError(StrategyError):
    """Technical indicator calculation error."""

    def __init__(self, indicator_name: str, reason: str):
        super().__init__(
            f"Indicator '{indicator_name}' error: {reason}",
            recoverable=True,
        )
        self.indicator_name = indicator_name
        self.reason = reason


class SpreadConstructionError(StrategyError):
    """Failed to construct option spread."""

    def __init__(
        self,
        spread_type: str,
        symbol: str,
        reason: str,
    ):
        super().__init__(
            f"Failed to construct {spread_type} for {symbol}: {reason}",
            recoverable=True,
        )
        self.spread_type = spread_type
        self.symbol = symbol
        self.reason = reason
        self.with_context(symbol=symbol)


class LegBalanceError(StrategyError):
    """Spread legs are imbalanced."""

    def __init__(
        self,
        symbol: str,
        long_count: int,
        short_count: int,
    ):
        super().__init__(
            f"Imbalanced spread for {symbol}: {long_count} longs, " f"{short_count} shorts",
            recoverable=True,
        )
        self.symbol = symbol
        self.long_count = long_count
        self.short_count = short_count
        self.with_context(symbol=symbol)


__all__ = [
    "BacktestError",
    "IndicatorError",
    "LegBalanceError",
    "SignalGenerationError",
    "SpreadConstructionError",
    "StrategyError",
    "StrategyExecutionError",
    "StrategyInitializationError",
]
