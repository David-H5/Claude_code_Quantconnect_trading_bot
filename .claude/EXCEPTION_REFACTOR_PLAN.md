# Exception Handling & Error Architecture Refactor Plan

**Created:** 2025-12-05
**Status:** Ready for Execution
**Prerequisite:** REFACTOR_PLAN.md Complete, NEXT_REFACTOR_PLAN.md Complete
**Estimated Effort:** 2-3 weeks

---

## Executive Summary

### Current State

| Metric | Current | Target |
|--------|---------|--------|
| Bare `except:` clauses | 1 (in project code) | 0 |
| `except Exception` clauses | 80+ files | <10 (only at boundaries) |
| Custom exceptions defined | 11 | 30+ |
| Exception chaining used | Minimal | All re-raises |
| Error context preserved | Inconsistent | Always |
| Generic `raise ValueError` | 67+ files | 0 |

### The Problem

1. **Silent Failures**: Bare `except:` swallows `KeyboardInterrupt`, `SystemExit`
2. **Poor Debugging**: Generic exceptions (`ValueError`, `RuntimeError`) lack context
3. **No Error Hierarchy**: 11 exceptions insufficient for 238K+ LOC codebase
4. **Missing Error Chaining**: Original cause lost in re-raises
5. **Inconsistent Patterns**: Each module handles errors differently

### The Solution

Build a comprehensive exception hierarchy with:
- Domain-specific exception types
- Rich error context
- Proper exception chaining
- Consistent error handling patterns
- Decorators for retry/fallback logic

---

## Research Sources

- [Python Exception Handling Best Practices](https://docs.python.org/3/tutorial/errors.html)
- [Exception Chaining PEP 3134](https://peps.python.org/pep-3134/)
- [Real Python: Exception Handling](https://realpython.com/python-exceptions/)
- [Trading System Error Patterns](https://www.amazon.com/Python-Algorithmic-Trading-Cookbook/dp/1838989358)

---

## Phase 1: Exception Hierarchy Expansion (Days 1-3)

**Goal:** Expand from 11 to 30+ domain-specific exceptions

### 1.1 Current Exception Structure

```
models/exceptions.py (11 exceptions)
├── TradingError (base)
├── RiskLimitExceeded
├── InsufficientFunds
├── OrderRejected
├── CircuitBreakerTripped
├── InvalidPositionSize
├── DataValidationError
├── OptionPricingError
├── ExecutionError
├── ConfigurationError
└── StrategyError
```

### 1.2 Proposed Exception Structure

```
models/exceptions/
├── __init__.py           # Public exports
├── base.py               # Base classes with context
├── execution.py          # Order execution errors
├── risk.py               # Risk management errors
├── agent.py              # LLM agent errors
├── data.py               # Data feed/validation errors
├── infrastructure.py     # Connection/timeout errors
└── api.py                # API/broker errors
```

### 1.3 New Exception Classes

**File:** `models/exceptions/base.py`

```python
"""Base exception classes with rich context."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import traceback


@dataclass
class ErrorContext:
    """Rich context for debugging exceptions."""
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = ""
    operation: str = ""
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    agent_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "operation": self.operation,
            "symbol": self.symbol,
            "order_id": self.order_id,
            "agent_name": self.agent_name,
            **self.extra
        }


class TradingError(Exception):
    """
    Base exception for all trading-related errors.

    Provides:
    - Rich error context
    - Automatic timestamp
    - Chaining support
    - Structured logging
    """

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.context = context or ErrorContext()
        self.recoverable = recoverable
        self.timestamp = datetime.now()

    def with_context(self, **kwargs) -> "TradingError":
        """Add context and return self for chaining."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.extra[key] = value
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to structured dict for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "recoverable": self.recoverable,
            "context": self.context.to_dict(),
            "traceback": traceback.format_exc()
        }
```

**File:** `models/exceptions/execution.py`

```python
"""Order execution exceptions."""
from .base import TradingError, ErrorContext
from typing import Optional


class ExecutionError(TradingError):
    """Base class for execution errors."""
    pass


class OrderSubmissionError(ExecutionError):
    """Failed to submit order to broker."""

    def __init__(self, order_id: str, reason: str, broker_code: Optional[str] = None):
        super().__init__(
            f"Order {order_id} submission failed: {reason}",
            recoverable=True
        )
        self.order_id = order_id
        self.broker_code = broker_code
        self.with_context(order_id=order_id, broker_code=broker_code)


class OrderFillError(ExecutionError):
    """Order fill failed or was partial."""

    def __init__(
        self,
        order_id: str,
        requested_qty: int,
        filled_qty: int,
        reason: str
    ):
        super().__init__(
            f"Order {order_id} fill failed: requested {requested_qty}, "
            f"filled {filled_qty}. Reason: {reason}",
            recoverable=True
        )
        self.order_id = order_id
        self.requested_qty = requested_qty
        self.filled_qty = filled_qty


class SlippageExceededError(ExecutionError):
    """Execution slippage exceeded threshold."""

    def __init__(
        self,
        order_id: str,
        expected_price: float,
        actual_price: float,
        threshold_pct: float
    ):
        slippage_pct = abs(actual_price - expected_price) / expected_price * 100
        super().__init__(
            f"Order {order_id} slippage {slippage_pct:.2f}% exceeds "
            f"threshold {threshold_pct:.2f}%",
            recoverable=False
        )
        self.order_id = order_id
        self.expected_price = expected_price
        self.actual_price = actual_price
        self.slippage_pct = slippage_pct


class OrderTimeoutError(ExecutionError):
    """Order execution timed out."""

    def __init__(self, order_id: str, timeout_seconds: float):
        super().__init__(
            f"Order {order_id} timed out after {timeout_seconds}s",
            recoverable=True
        )
        self.order_id = order_id
        self.timeout_seconds = timeout_seconds


class MarketClosedError(ExecutionError):
    """Market is closed for trading."""

    def __init__(self, market: str, next_open: Optional[str] = None):
        msg = f"Market {market} is closed"
        if next_open:
            msg += f", opens at {next_open}"
        super().__init__(msg, recoverable=True)
        self.market = market
        self.next_open = next_open
```

**File:** `models/exceptions/agent.py`

```python
"""LLM Agent exceptions."""
from .base import TradingError
from typing import Optional, List


class AgentError(TradingError):
    """Base class for agent errors."""
    pass


class AgentTimeoutError(AgentError):
    """Agent execution timed out."""

    def __init__(self, agent_name: str, timeout_ms: int):
        super().__init__(
            f"Agent '{agent_name}' timed out after {timeout_ms}ms",
            recoverable=True
        )
        self.agent_name = agent_name
        self.timeout_ms = timeout_ms
        self.with_context(agent_name=agent_name)


class AgentRateLimitError(AgentError):
    """LLM API rate limit hit."""

    def __init__(self, agent_name: str, retry_after_seconds: Optional[int] = None):
        msg = f"Agent '{agent_name}' hit rate limit"
        if retry_after_seconds:
            msg += f", retry after {retry_after_seconds}s"
        super().__init__(msg, recoverable=True)
        self.agent_name = agent_name
        self.retry_after_seconds = retry_after_seconds


class ConsensusFailedError(AgentError):
    """Multi-agent consensus failed."""

    def __init__(
        self,
        agents: List[str],
        reason: str,
        votes: Optional[dict] = None
    ):
        super().__init__(
            f"Consensus failed among {len(agents)} agents: {reason}",
            recoverable=True
        )
        self.agents = agents
        self.votes = votes or {}


class AgentHallucinationError(AgentError):
    """Agent produced invalid/hallucinated output."""

    def __init__(self, agent_name: str, field: str, invalid_value: str):
        super().__init__(
            f"Agent '{agent_name}' produced invalid {field}: {invalid_value}",
            recoverable=True
        )
        self.agent_name = agent_name
        self.field = field
        self.invalid_value = invalid_value


class PromptVersionError(AgentError):
    """Prompt version not found or invalid."""

    def __init__(self, agent_role: str, version: str):
        super().__init__(
            f"Prompt version '{version}' not found for {agent_role}",
            recoverable=False
        )
        self.agent_role = agent_role
        self.version = version
```

**File:** `models/exceptions/risk.py`

```python
"""Risk management exceptions."""
from .base import TradingError
from typing import Optional


class RiskError(TradingError):
    """Base class for risk errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, recoverable=False, **kwargs)


class RiskLimitExceededError(RiskError):
    """A risk limit was exceeded."""

    def __init__(
        self,
        limit_name: str,
        current_value: float,
        limit_value: float,
        unit: str = "%"
    ):
        if unit == "%":
            super().__init__(
                f"{limit_name}: {current_value:.2%} exceeds limit {limit_value:.2%}"
            )
        else:
            super().__init__(
                f"{limit_name}: {current_value:.2f}{unit} exceeds limit {limit_value:.2f}{unit}"
            )
        self.limit_name = limit_name
        self.current_value = current_value
        self.limit_value = limit_value


class CircuitBreakerTrippedError(RiskError):
    """Circuit breaker has tripped."""

    def __init__(
        self,
        reason: str,
        breaker_state: str,
        requires_manual_reset: bool = True
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
        super().__init__(
            f"Drawdown {current_drawdown:.2%} exceeds max {max_allowed:.2%}"
        )
        self.current_drawdown = current_drawdown
        self.max_allowed = max_allowed


class ConcentrationRiskError(RiskError):
    """Position concentration too high."""

    def __init__(self, symbol: str, concentration_pct: float, max_allowed: float):
        super().__init__(
            f"{symbol} concentration {concentration_pct:.2%} exceeds max {max_allowed:.2%}"
        )
        self.symbol = symbol
        self.concentration_pct = concentration_pct
        self.with_context(symbol=symbol)
```

**File:** `models/exceptions/infrastructure.py`

```python
"""Infrastructure exceptions (connections, timeouts, etc.)."""
from .base import TradingError
from typing import Optional


class InfrastructureError(TradingError):
    """Base class for infrastructure errors."""
    pass


class ConnectionError(InfrastructureError):
    """Failed to connect to a service."""

    def __init__(self, service: str, host: str, port: Optional[int] = None):
        location = f"{host}:{port}" if port else host
        super().__init__(
            f"Failed to connect to {service} at {location}",
            recoverable=True
        )
        self.service = service
        self.host = host
        self.port = port


class ServiceTimeoutError(InfrastructureError):
    """Service call timed out."""

    def __init__(self, service: str, operation: str, timeout_seconds: float):
        super().__init__(
            f"{service}.{operation}() timed out after {timeout_seconds}s",
            recoverable=True
        )
        self.service = service
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.with_context(component=service, operation=operation)


class RedisError(InfrastructureError):
    """Redis operation failed."""

    def __init__(self, operation: str, key: Optional[str] = None, reason: str = ""):
        msg = f"Redis {operation} failed"
        if key:
            msg += f" for key '{key}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, recoverable=True)
        self.operation = operation
        self.key = key


class DataFeedError(InfrastructureError):
    """Data feed error."""

    def __init__(self, feed: str, symbol: Optional[str] = None, reason: str = ""):
        msg = f"Data feed '{feed}' error"
        if symbol:
            msg += f" for {symbol}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, recoverable=True)
        self.feed = feed
        self.with_context(symbol=symbol)
```

**File:** `models/exceptions/__init__.py`

```python
"""
Trading Exception Hierarchy

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
"""

from .base import TradingError, ErrorContext

# Execution errors
from .execution import (
    ExecutionError,
    OrderSubmissionError,
    OrderFillError,
    SlippageExceededError,
    OrderTimeoutError,
    MarketClosedError,
)

# Risk errors
from .risk import (
    RiskError,
    RiskLimitExceededError,
    CircuitBreakerTrippedError,
    MaxDrawdownExceededError,
    ConcentrationRiskError,
)

# Agent errors
from .agent import (
    AgentError,
    AgentTimeoutError,
    AgentRateLimitError,
    ConsensusFailedError,
    AgentHallucinationError,
    PromptVersionError,
)

# Infrastructure errors
from .infrastructure import (
    InfrastructureError,
    ConnectionError,
    ServiceTimeoutError,
    RedisError,
    DataFeedError,
)

__all__ = [
    # Base
    "TradingError",
    "ErrorContext",

    # Execution
    "ExecutionError",
    "OrderSubmissionError",
    "OrderFillError",
    "SlippageExceededError",
    "OrderTimeoutError",
    "MarketClosedError",

    # Risk
    "RiskError",
    "RiskLimitExceededError",
    "CircuitBreakerTrippedError",
    "MaxDrawdownExceededError",
    "ConcentrationRiskError",

    # Agent
    "AgentError",
    "AgentTimeoutError",
    "AgentRateLimitError",
    "ConsensusFailedError",
    "AgentHallucinationError",
    "PromptVersionError",

    # Infrastructure
    "InfrastructureError",
    "ConnectionError",
    "ServiceTimeoutError",
    "RedisError",
    "DataFeedError",
]
```

### 1.4 Migration of Existing Exceptions

**Action:** Move existing 11 exceptions to new structure with backwards-compatible re-exports.

```python
# models/exceptions.py (legacy - keep for backwards compat)
"""DEPRECATED: Import from models.exceptions instead."""
from models.exceptions import *

import warnings
warnings.warn(
    "models.exceptions is deprecated. "
    "Import from models.exceptions (subpackage) instead.",
    DeprecationWarning,
    stacklevel=2
)
```

### 1.5 Commit Phase 1

```bash
git add -A
git commit -m "feat(exceptions): Expand exception hierarchy to 30+ types

- Created models/exceptions/ subpackage with categorized exceptions
- Added ErrorContext dataclass for rich debugging info
- Added execution, risk, agent, infrastructure exception modules
- Maintained backwards compatibility with legacy imports
- All exceptions include .to_dict() for structured logging

Implements Phase 1 of EXCEPTION_REFACTOR_PLAN.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Phase 2: Fix Critical Error Handling (Days 4-6)

**Goal:** Fix bare `except:` and critical `except Exception` patterns

### 2.1 Fix Bare Except in llm/agents/base.py

**Current (line 631):**
```python
except:
    pass
```

**Fixed:**
```python
except Exception as e:
    logger.warning(f"Failed to cleanup agent resources: {e}")
```

### 2.2 Fix High-Priority except Exception Patterns

**Priority Files (based on importance to trading):**

| File | Count | Action |
|------|-------|--------|
| infrastructure/redis_client.py | 12 | Convert to RedisError |
| infrastructure/market_stream.py | 9 | Convert to DataFeedError |
| evaluation/orchestration_pipeline.py | 12 | Convert to specific types |
| infrastructure/timeseries.py | 12 | Convert to RedisError |
| infrastructure/pubsub.py | 8 | Convert to InfrastructureError |

**Pattern to fix:**

```python
# BEFORE (bad)
try:
    result = redis_client.get(key)
except Exception as e:
    logger.error(f"Redis error: {e}")
    return None

# AFTER (good)
try:
    result = redis_client.get(key)
except redis.RedisError as e:
    raise RedisError("get", key=key, reason=str(e)) from e
except ConnectionRefusedError as e:
    raise ConnectionError("redis", host, port) from e
```

### 2.3 Exception Chaining Rule

**ALWAYS use `from e` when re-raising:**

```python
try:
    do_something()
except ValueError as e:
    # Preserves original traceback
    raise DataValidationError(field, value, str(e)) from e
```

### 2.4 Commit Phase 2

```bash
git add -A
git commit -m "fix(exceptions): Fix bare except and add exception chaining

- Fixed bare except: in llm/agents/base.py
- Added exception chaining (from e) to all re-raises
- Converted generic exceptions in infrastructure/ to specific types
- All errors now preserve original traceback

Implements Phase 2 of EXCEPTION_REFACTOR_PLAN.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Phase 3: Replace Generic Exceptions (Days 7-10)

**Goal:** Replace `raise ValueError/RuntimeError` with domain exceptions

### 3.1 Files to Update

| Module | Generic Raises | Action |
|--------|---------------|--------|
| execution/*.py | 15+ | Use ExecutionError subclasses |
| llm/agents/*.py | 10+ | Use AgentError subclasses |
| evaluation/*.py | 20+ | Use appropriate types |
| models/*.py | 5+ | Use TradingError subclasses |

### 3.2 Conversion Patterns

**Pattern 1: ValueError → DataValidationError**

```python
# BEFORE
if not isinstance(quantity, int):
    raise ValueError(f"Quantity must be int, got {type(quantity)}")

# AFTER
if not isinstance(quantity, int):
    raise DataValidationError(
        field="quantity",
        value=quantity,
        reason=f"must be int, got {type(quantity).__name__}"
    )
```

**Pattern 2: RuntimeError → Specific Type**

```python
# BEFORE
if backtest_result.returncode != 0:
    raise RuntimeError(f"Backtest failed: {backtest_result.stderr}")

# AFTER
if backtest_result.returncode != 0:
    raise ExecutionError(
        order_id="backtest",
        reason=backtest_result.stderr
    ).with_context(component="quantconnect", operation="backtest")
```

**Pattern 3: Generic Exception → Specific**

```python
# BEFORE
try:
    stream.read()
except Exception as e:
    logger.error(f"Stream error: {e}")

# AFTER
try:
    stream.read()
except redis.ConnectionError as e:
    raise DataFeedError("redis_stream", symbol=symbol, reason=str(e)) from e
except TimeoutError as e:
    raise ServiceTimeoutError("redis", "read", timeout_seconds=5) from e
```

### 3.3 Commit Phase 3

```bash
git add -A
git commit -m "refactor(exceptions): Replace generic exceptions with domain types

- Replaced ValueError with DataValidationError (15 files)
- Replaced RuntimeError with specific execution errors (10 files)
- Updated evaluation/ to use appropriate exception types
- All exceptions now carry domain context

Implements Phase 3 of EXCEPTION_REFACTOR_PLAN.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Phase 4: Error Recovery Patterns (Days 11-13)

**Goal:** Add retry decorators and fallback patterns

### 4.1 Retry Decorator

**File:** `utils/error_handling.py`

```python
"""Error handling utilities."""
import functools
import logging
import time
from typing import Callable, Tuple, Type, Optional

logger = logging.getLogger(__name__)


def retry(
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Retry decorator with exponential backoff.

    Args:
        exceptions: Tuple of exception types to catch
        max_attempts: Maximum retry attempts
        delay_seconds: Initial delay between retries
        backoff_factor: Multiply delay by this factor each attempt
        max_delay: Maximum delay cap
        on_retry: Callback function(exception, attempt_number)

    Usage:
        @retry(exceptions=(ConnectionError, TimeoutError), max_attempts=3)
        def fetch_data():
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = delay_seconds
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise

                    if on_retry:
                        on_retry(e, attempt)

                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s"
                    )

                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

            raise last_exception
        return wrapper
    return decorator


def fallback(default_value=None, exceptions=(Exception,), log_level="warning"):
    """
    Return default value on exception instead of raising.

    Usage:
        @fallback(default_value=[], exceptions=(DataFeedError,))
        def get_prices():
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                log_func = getattr(logger, log_level)
                log_func(f"{func.__name__} failed, using fallback: {e}")
                return default_value
        return wrapper
    return decorator


class ErrorAccumulator:
    """
    Accumulate errors without stopping execution.

    Usage:
        with ErrorAccumulator() as errors:
            errors.try_run(validate_order, order)
            errors.try_run(check_risk, order)
            errors.try_run(verify_funds, order)

        if errors.has_errors:
            raise MultipleErrors(errors.errors)
    """

    def __init__(self):
        self.errors: list = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def try_run(self, func: Callable, *args, **kwargs):
        """Run function and capture any exception."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.errors.append(e)
            return None

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def raise_if_errors(self):
        """Raise first error if any occurred."""
        if self.errors:
            raise self.errors[0]
```

### 4.2 Usage Examples

```python
from utils.error_handling import retry, fallback
from models.exceptions import DataFeedError, ServiceTimeoutError

# Retry on transient errors
@retry(
    exceptions=(ServiceTimeoutError, ConnectionError),
    max_attempts=3,
    delay_seconds=2.0
)
def fetch_market_data(symbol: str):
    return market_feed.get_quote(symbol)


# Fallback for non-critical data
@fallback(default_value=0.5, exceptions=(DataFeedError,))
def get_volatility(symbol: str) -> float:
    return analytics.calculate_iv(symbol)
```

### 4.3 Commit Phase 4

```bash
git add -A
git commit -m "feat(utils): Add error recovery decorators

- Added @retry with exponential backoff
- Added @fallback for graceful degradation
- Added ErrorAccumulator for batch validation
- Integrated with observability logging

Implements Phase 4 of EXCEPTION_REFACTOR_PLAN.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Phase 5: Logging Integration (Days 14-15)

**Goal:** Connect exceptions to structured logging and observability

### 5.1 Exception Logging Hook

**File:** `observability/exception_logger.py`

```python
"""Structured exception logging."""
import logging
import json
from typing import Optional
from models.exceptions import TradingError

logger = logging.getLogger(__name__)


def log_exception(
    e: Exception,
    context: Optional[dict] = None,
    level: str = "error"
) -> None:
    """
    Log exception with structured context.

    Args:
        e: The exception
        context: Additional context to include
        level: Log level (error, warning, critical)
    """
    log_func = getattr(logger, level)

    if isinstance(e, TradingError):
        # Use built-in structured logging
        log_data = e.to_dict()
        if context:
            log_data["additional_context"] = context
        log_func(str(e), extra={"structured": log_data})
    else:
        # Wrap generic exceptions
        log_data = {
            "error_type": type(e).__name__,
            "message": str(e),
            "context": context or {}
        }
        log_func(str(e), extra={"structured": log_data})


def exception_handler(func):
    """
    Decorator to log all exceptions from a function.

    Usage:
        @exception_handler
        def process_order(order):
            ...
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_exception(e, context={
                "function": func.__name__,
                "args": str(args)[:200],
                "kwargs": str(kwargs)[:200]
            })
            raise
    return wrapper
```

### 5.2 Commit Phase 5

```bash
git add -A
git commit -m "feat(observability): Add structured exception logging

- Created exception_logger.py for structured logging
- Added @exception_handler decorator
- Integrated TradingError.to_dict() with logging
- All exceptions now emit structured JSON logs

Implements Phase 5 of EXCEPTION_REFACTOR_PLAN.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Verification Checklist

### Automated Checks

```bash
# 1. No bare except clauses
grep -r "except:" --include="*.py" | grep -v ".venv" | grep -v "except:.*#"
# Should return 0 results

# 2. All exceptions have chaining
grep -rn "raise.*Error" --include="*.py" | grep -v "from e" | head -20
# Review and fix any without "from e"

# 3. Test exception hierarchy
python3 -c "from models.exceptions import *; print('OK')"

# 4. Validate all Python files
find . -name "*.py" -not -path "./.venv/*" -exec python3 -m py_compile {} \;

# 5. Run tests
pytest tests/ -x -q --tb=short
```

### Manual Review

- [ ] All TradingError subclasses have docstrings
- [ ] ErrorContext is used for debugging info
- [ ] Retry decorator tested with transient failures
- [ ] Logging output includes structured context
- [ ] Backwards compatibility maintained

---

## Summary: Files Created/Modified

### New Files

```
models/exceptions/
├── __init__.py
├── base.py
├── execution.py
├── risk.py
├── agent.py
└── infrastructure.py

utils/error_handling.py
observability/exception_logger.py
```

### Modified Files

```
models/exceptions.py           → Deprecated wrapper
llm/agents/base.py             → Fix bare except
infrastructure/redis_client.py → Use RedisError
infrastructure/market_stream.py → Use DataFeedError
infrastructure/timeseries.py   → Use RedisError
infrastructure/pubsub.py       → Use InfrastructureError
evaluation/*.py                → Use appropriate types
execution/*.py                 → Use ExecutionError subclasses
```

---

## Rollback Plan

If issues arise:

```bash
# Revert to before refactoring
git log --oneline -10
git revert HEAD~N..HEAD

# Or hard reset (destructive)
git reset --hard <commit-before-refactoring>
```

---

## Quick Start

To execute this plan:

```bash
# Execute phase by phase:
"Read .claude/EXCEPTION_REFACTOR_PLAN.md and execute Phase 1"
```

Or all phases:
```bash
"Execute the full exception refactor from .claude/EXCEPTION_REFACTOR_PLAN.md"
```

---

## Notes

1. **Execute phases in order** - Each phase builds on previous
2. **Commit after each phase** - Easy rollback
3. **Run tests after each phase** - Catch regressions early
4. **Maintain backwards compatibility** - Don't break existing imports
5. **Use exception chaining** - Always `raise ... from e`
