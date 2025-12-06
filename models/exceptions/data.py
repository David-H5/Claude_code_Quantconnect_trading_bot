"""Data validation and processing exceptions."""

from typing import Any

from .base import TradingError


class DataError(TradingError):
    """Base class for data-related errors."""

    pass


class DataValidationError(DataError):
    """Data validation failed."""

    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            f"Invalid {field}={value!r}: {reason}",
            recoverable=False,
        )
        self.field = field
        self.value = value
        self.reason = reason


class DataMissingError(DataError):
    """Required data is missing."""

    def __init__(self, field: str, source: str | None = None):
        msg = f"Missing required field: {field}"
        if source:
            msg += f" from {source}"
        super().__init__(msg, recoverable=False)
        self.field = field
        self.source = source


class DataStaleError(DataError):
    """Data is too old/stale to use."""

    def __init__(
        self,
        data_type: str,
        age_seconds: float,
        max_age_seconds: float,
    ):
        super().__init__(
            f"{data_type} is stale: {age_seconds:.1f}s old " f"(max: {max_age_seconds:.1f}s)",
            recoverable=True,
        )
        self.data_type = data_type
        self.age_seconds = age_seconds
        self.max_age_seconds = max_age_seconds


class DataParseError(DataError):
    """Failed to parse data."""

    def __init__(self, data_type: str, reason: str, raw_value: str | None = None):
        msg = f"Failed to parse {data_type}: {reason}"
        if raw_value:
            truncated = raw_value[:100] + "..." if len(raw_value) > 100 else raw_value
            msg += f" (value: {truncated!r})"
        super().__init__(msg, recoverable=False)
        self.data_type = data_type
        self.reason = reason
        self.raw_value = raw_value


class OptionPricingError(DataError):
    """Option pricing calculation failed."""

    def __init__(self, contract: str, reason: str):
        super().__init__(
            f"Option pricing error for {contract}: {reason}",
            recoverable=False,
        )
        self.contract = contract
        self.reason = reason
        self.with_context(symbol=contract)


class GreeksCalculationError(DataError):
    """Failed to calculate Greeks."""

    def __init__(self, contract: str, greek: str, reason: str):
        super().__init__(
            f"Failed to calculate {greek} for {contract}: {reason}",
            recoverable=False,
        )
        self.contract = contract
        self.greek = greek
        self.reason = reason
        self.with_context(symbol=contract)


class MarketDataError(DataError):
    """Market data is invalid or unavailable."""

    def __init__(self, symbol: str, reason: str):
        super().__init__(
            f"Market data error for {symbol}: {reason}",
            recoverable=True,
        )
        self.symbol = symbol
        self.reason = reason
        self.with_context(symbol=symbol)


class ConfigurationError(DataError):
    """Configuration is invalid."""

    def __init__(self, key: str, reason: str):
        super().__init__(
            f"Configuration error for '{key}': {reason}",
            recoverable=False,
        )
        self.key = key
        self.reason = reason


__all__ = [
    "ConfigurationError",
    "DataError",
    "DataMissingError",
    "DataParseError",
    "DataStaleError",
    "DataValidationError",
    "GreeksCalculationError",
    "MarketDataError",
    "OptionPricingError",
]
