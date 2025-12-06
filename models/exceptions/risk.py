"""Risk management exceptions."""

from .base import ErrorContext, TradingError


class RiskError(TradingError):
    """Base class for risk errors. Non-recoverable by default."""

    def __init__(
        self,
        message: str,
        context: ErrorContext | None = None,
        recoverable: bool = False,
    ):
        super().__init__(message, context=context, recoverable=recoverable)


class RiskLimitExceededError(RiskError):
    """A risk limit was exceeded."""

    def __init__(
        self,
        limit_name: str,
        current_value: float,
        limit_value: float,
        unit: str = "%",
    ):
        if unit == "%":
            message = f"{limit_name}: {current_value:.2%} exceeds limit {limit_value:.2%}"
        else:
            message = f"{limit_name}: {current_value:.2f}{unit} exceeds " f"limit {limit_value:.2f}{unit}"
        super().__init__(message)
        self.limit_name = limit_name
        self.current_value = current_value
        self.limit_value = limit_value
        self.unit = unit


class CircuitBreakerTrippedError(RiskError):
    """Circuit breaker has tripped."""

    def __init__(
        self,
        reason: str,
        breaker_state: str = "OPEN",
        requires_manual_reset: bool = True,
    ):
        msg = f"Circuit breaker tripped ({breaker_state}): {reason}"
        if requires_manual_reset:
            msg += " [REQUIRES MANUAL RESET]"
        super().__init__(msg)
        self.reason = reason
        self.breaker_state = breaker_state
        self.requires_manual_reset = requires_manual_reset


class MaxDrawdownExceededError(RiskError):
    """Maximum drawdown threshold exceeded."""

    def __init__(self, current_drawdown: float, max_allowed: float):
        super().__init__(f"Drawdown {current_drawdown:.2%} exceeds max {max_allowed:.2%}")
        self.current_drawdown = current_drawdown
        self.max_allowed = max_allowed


class ConcentrationRiskError(RiskError):
    """Position concentration too high."""

    def __init__(self, symbol: str, concentration_pct: float, max_allowed: float):
        super().__init__(f"{symbol} concentration {concentration_pct:.2%} " f"exceeds max {max_allowed:.2%}")
        self.symbol = symbol
        self.concentration_pct = concentration_pct
        self.max_allowed = max_allowed
        self.with_context(symbol=symbol)


class DailyLossLimitError(RiskError):
    """Daily loss limit exceeded."""

    def __init__(self, current_loss: float, max_allowed: float):
        super().__init__(f"Daily loss {current_loss:.2%} exceeds limit {max_allowed:.2%}")
        self.current_loss = current_loss
        self.max_allowed = max_allowed


class ConsecutiveLossError(RiskError):
    """Too many consecutive losses."""

    def __init__(self, loss_count: int, max_allowed: int):
        super().__init__(f"Consecutive losses ({loss_count}) exceeds limit ({max_allowed})")
        self.loss_count = loss_count
        self.max_allowed = max_allowed


class ExposureLimitError(RiskError):
    """Total exposure limit exceeded."""

    def __init__(self, current_exposure: float, max_allowed: float, exposure_type: str):
        super().__init__(f"{exposure_type} exposure {current_exposure:.2%} " f"exceeds limit {max_allowed:.2%}")
        self.current_exposure = current_exposure
        self.max_allowed = max_allowed
        self.exposure_type = exposure_type


__all__ = [
    "CircuitBreakerTrippedError",
    "ConcentrationRiskError",
    "ConsecutiveLossError",
    "DailyLossLimitError",
    "ExposureLimitError",
    "MaxDrawdownExceededError",
    "RiskError",
    "RiskLimitExceededError",
]
