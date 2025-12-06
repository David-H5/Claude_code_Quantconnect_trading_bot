"""Trading Exception Hierarchy.

This module provides a comprehensive exception hierarchy for the trading system.
All exceptions inherit from TradingError and provide rich context for debugging.

Usage:
    from models.exceptions import (
        TradingError,
        OrderSubmissionError,
        CircuitBreakerTrippedError,
        AgentTimeoutError,
    )

    try:
        submit_order(order)
    except OrderSubmissionError as e:
        logger.error(f"Order failed: {e}", extra=e.to_dict())
        if e.recoverable:
            retry_order(order)
    except TradingError as e:
        # Catch-all for trading errors
        logger.error(f"Trading error: {e}")

Exception Hierarchy:
    TradingError (base)
    ├── ExecutionError
    │   ├── OrderSubmissionError
    │   ├── OrderFillError
    │   ├── SlippageExceededError
    │   ├── OrderTimeoutError
    │   ├── MarketClosedError
    │   ├── OrderRejectedError
    │   ├── InsufficientFundsError
    │   └── InvalidPositionSizeError
    ├── RiskError
    │   ├── RiskLimitExceededError
    │   ├── CircuitBreakerTrippedError
    │   ├── MaxDrawdownExceededError
    │   ├── ConcentrationRiskError
    │   ├── DailyLossLimitError
    │   ├── ConsecutiveLossError
    │   └── ExposureLimitError
    ├── AgentError
    │   ├── AgentTimeoutError
    │   ├── AgentRateLimitError
    │   ├── ConsensusFailedError
    │   ├── AgentHallucinationError
    │   ├── PromptVersionError
    │   ├── AgentConfigurationError
    │   ├── AgentCommunicationError
    │   └── DebateResolutionError
    ├── InfrastructureError
    │   ├── ConnectionFailedError
    │   ├── ServiceTimeoutError
    │   ├── RedisError
    │   ├── DataFeedError
    │   ├── WebSocketError
    │   ├── MessageQueueError
    │   ├── TimeSeriesError
    │   └── CacheError
    ├── DataError
    │   ├── DataValidationError
    │   ├── DataMissingError
    │   ├── DataStaleError
    │   ├── DataParseError
    │   ├── OptionPricingError
    │   ├── GreeksCalculationError
    │   ├── MarketDataError
    │   └── ConfigurationError
    └── StrategyError
        ├── StrategyInitializationError
        ├── StrategyExecutionError
        ├── SignalGenerationError
        ├── BacktestError
        ├── IndicatorError
        ├── SpreadConstructionError
        └── LegBalanceError
"""

# Agent errors
from .agent import (
    AgentCommunicationError,
    AgentConfigurationError,
    AgentError,
    AgentHallucinationError,
    AgentRateLimitError,
    AgentTimeoutError,
    ConsensusFailedError,
    DebateResolutionError,
    PromptVersionError,
)
from .base import ErrorContext, TradingError

# Data errors
from .data import (
    ConfigurationError,
    DataError,
    DataMissingError,
    DataParseError,
    DataStaleError,
    DataValidationError,
    GreeksCalculationError,
    MarketDataError,
    OptionPricingError,
)

# Execution errors
from .execution import (
    ExecutionError,
    InsufficientFundsError,
    InvalidPositionSizeError,
    MarketClosedError,
    OrderFillError,
    OrderRejectedError,
    OrderSubmissionError,
    OrderTimeoutError,
    SlippageExceededError,
)

# Infrastructure errors
from .infrastructure import (
    CacheError,
    ConnectionFailedError,
    DataFeedError,
    InfrastructureError,
    MessageQueueError,
    RedisError,
    ServiceTimeoutError,
    TimeSeriesError,
    WebSocketError,
)

# Risk errors
from .risk import (
    CircuitBreakerTrippedError,
    ConcentrationRiskError,
    ConsecutiveLossError,
    DailyLossLimitError,
    ExposureLimitError,
    MaxDrawdownExceededError,
    RiskError,
    RiskLimitExceededError,
)

# Strategy errors
from .strategy import (
    BacktestError,
    IndicatorError,
    LegBalanceError,
    SignalGenerationError,
    SpreadConstructionError,
    StrategyError,
    StrategyExecutionError,
    StrategyInitializationError,
)


# Backwards compatibility aliases for old exception names
RiskLimitExceeded = RiskLimitExceededError
InsufficientFunds = InsufficientFundsError
OrderRejected = OrderRejectedError
CircuitBreakerTripped = CircuitBreakerTrippedError
InvalidPositionSize = InvalidPositionSizeError

__all__ = [
    # Base
    "ErrorContext",
    "TradingError",
    # Execution
    "ExecutionError",
    "OrderSubmissionError",
    "OrderFillError",
    "SlippageExceededError",
    "OrderTimeoutError",
    "MarketClosedError",
    "OrderRejectedError",
    "InsufficientFundsError",
    "InvalidPositionSizeError",
    # Risk
    "RiskError",
    "RiskLimitExceededError",
    "CircuitBreakerTrippedError",
    "MaxDrawdownExceededError",
    "ConcentrationRiskError",
    "DailyLossLimitError",
    "ConsecutiveLossError",
    "ExposureLimitError",
    # Agent
    "AgentError",
    "AgentTimeoutError",
    "AgentRateLimitError",
    "ConsensusFailedError",
    "AgentHallucinationError",
    "PromptVersionError",
    "AgentConfigurationError",
    "AgentCommunicationError",
    "DebateResolutionError",
    # Infrastructure
    "InfrastructureError",
    "ConnectionFailedError",
    "ServiceTimeoutError",
    "RedisError",
    "DataFeedError",
    "WebSocketError",
    "MessageQueueError",
    "TimeSeriesError",
    "CacheError",
    # Data
    "DataError",
    "DataValidationError",
    "DataMissingError",
    "DataStaleError",
    "DataParseError",
    "OptionPricingError",
    "GreeksCalculationError",
    "MarketDataError",
    "ConfigurationError",
    # Strategy
    "StrategyError",
    "StrategyInitializationError",
    "StrategyExecutionError",
    "SignalGenerationError",
    "BacktestError",
    "IndicatorError",
    "SpreadConstructionError",
    "LegBalanceError",
    # Backwards compatibility aliases
    "RiskLimitExceeded",
    "InsufficientFunds",
    "OrderRejected",
    "CircuitBreakerTripped",
    "InvalidPositionSize",
]
