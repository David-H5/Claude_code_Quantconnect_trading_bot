"""
Error Handling Utilities

Provides decorators and utilities for robust error handling:
- @retry: Retry with exponential backoff for transient failures
- @fallback: Graceful degradation with default values
- ErrorAccumulator: Batch validation without stopping execution

Part of Exception Refactoring (Phase 4).
"""

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import Any


logger = logging.getLogger(__name__)


def retry(
    exceptions: tuple[type[Exception], ...] = (Exception,),
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable:
    """
    Retry decorator with exponential backoff.

    Automatically retries a function when specified exceptions occur,
    with configurable exponential backoff between attempts.

    Args:
        exceptions: Tuple of exception types to catch and retry on.
        max_attempts: Maximum number of retry attempts (default: 3).
        delay_seconds: Initial delay between retries in seconds (default: 1.0).
        backoff_factor: Multiply delay by this factor each attempt (default: 2.0).
        max_delay: Maximum delay cap in seconds (default: 60.0).
        jitter: Add random jitter to prevent thundering herd (default: True).
        on_retry: Optional callback function(exception, attempt_number) called
            before each retry.

    Returns:
        Decorated function that retries on specified exceptions.

    Usage:
        @retry(exceptions=(ConnectionError, TimeoutError), max_attempts=3)
        def fetch_data():
            return api.get_prices()

        @retry(
            exceptions=(ServiceTimeoutError,),
            max_attempts=5,
            delay_seconds=2.0,
            on_retry=lambda e, n: logger.warning(f"Retry {n}: {e}")
        )
        def connect_to_broker():
            return broker.connect()
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = delay_seconds
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise

                    if on_retry:
                        on_retry(e, attempt)

                    # Calculate delay with optional jitter
                    actual_delay = delay
                    if jitter:
                        actual_delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {actual_delay:.1f}s"
                    )

                    time.sleep(actual_delay)
                    delay = min(delay * backoff_factor, max_delay)

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError(f"{func.__name__} failed with no exception captured")

        return wrapper

    return decorator


def fallback(
    default_value: Any = None,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    log_level: str = "warning",
) -> Callable:
    """
    Return default value on exception instead of raising.

    Provides graceful degradation by returning a fallback value when
    the decorated function fails with specified exceptions.

    Args:
        default_value: Value to return when function fails.
        exceptions: Tuple of exception types to catch.
        log_level: Logging level for the failure message
            (error, warning, info, debug).

    Returns:
        Decorated function that returns default_value on failure.

    Usage:
        @fallback(default_value=[], exceptions=(DataFeedError,))
        def get_prices() -> list:
            return feed.get_latest()

        @fallback(default_value=0.5, log_level="debug")
        def get_volatility(symbol: str) -> float:
            return analytics.calculate_iv(symbol)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
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

    Allows running multiple validation or processing steps, collecting
    all errors instead of failing on the first one.

    Usage:
        with ErrorAccumulator() as errors:
            errors.try_run(validate_order, order)
            errors.try_run(check_risk, order)
            errors.try_run(verify_funds, order)

        if errors.has_errors:
            for err in errors.errors:
                logger.error(f"Validation failed: {err}")
            raise errors.errors[0]  # Or handle all errors

        # Alternative: raise first error if any
        errors.raise_if_errors()

    Attributes:
        errors: List of captured exceptions.
        results: List of successful return values (None for failed calls).
    """

    def __init__(self) -> None:
        """Initialize error accumulator."""
        self.errors: list[Exception] = []
        self.results: list[Any] = []

    def __enter__(self) -> "ErrorAccumulator":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Exit context manager without suppressing exceptions."""
        return False

    def try_run(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any | None:
        """
        Run function and capture any exception.

        Args:
            func: Function to call.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Function return value on success, None on failure.
        """
        try:
            result = func(*args, **kwargs)
            self.results.append(result)
            return result
        except Exception as e:
            self.errors.append(e)
            self.results.append(None)
            return None

    @property
    def has_errors(self) -> bool:
        """Check if any errors were captured."""
        return len(self.errors) > 0

    @property
    def error_count(self) -> int:
        """Get the number of captured errors."""
        return len(self.errors)

    @property
    def success_count(self) -> int:
        """Get the number of successful runs."""
        return len(self.results) - len(self.errors)

    def raise_if_errors(self) -> None:
        """
        Raise first error if any occurred.

        Raises:
            The first captured exception if any errors were accumulated.
        """
        if self.errors:
            raise self.errors[0]

    def raise_all_errors(self) -> None:
        """
        Raise all errors as a combined exception.

        Raises:
            ExceptionGroup containing all captured exceptions (Python 3.11+),
            or the first error with a message listing all errors.
        """
        if not self.errors:
            return

        if len(self.errors) == 1:
            raise self.errors[0]

        # For Python 3.11+, use ExceptionGroup
        try:
            raise ExceptionGroup("Multiple errors occurred", self.errors)
        except NameError:
            # Python < 3.11 fallback
            error_messages = [str(e) for e in self.errors]
            combined_msg = f"Multiple errors ({len(self.errors)}): {'; '.join(error_messages)}"
            raise RuntimeError(combined_msg) from self.errors[0]

    def get_error_summary(self) -> str:
        """
        Get a human-readable summary of all errors.

        Returns:
            Summary string listing all error types and messages.
        """
        if not self.errors:
            return "No errors"

        lines = [f"Accumulated {len(self.errors)} error(s):"]
        for i, err in enumerate(self.errors, 1):
            lines.append(f"  {i}. {type(err).__name__}: {err}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all accumulated errors and results."""
        self.errors.clear()
        self.results.clear()


# Convenience function for simple retry cases
def with_retry(
    func: Callable,
    *args: Any,
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """
    Execute a function with retry logic (non-decorator version).

    Useful when you can't use the decorator syntax.

    Args:
        func: Function to call.
        *args: Positional arguments for the function.
        max_attempts: Maximum number of attempts.
        delay: Initial delay between retries.
        exceptions: Exception types to catch.
        **kwargs: Keyword arguments for the function.

    Returns:
        Function return value on success.

    Raises:
        Last exception if all attempts fail.

    Usage:
        result = with_retry(api.fetch, symbol, max_attempts=3, delay=2.0)
    """

    @retry(exceptions=exceptions, max_attempts=max_attempts, delay_seconds=delay)
    def _wrapped() -> Any:
        return func(*args, **kwargs)

    return _wrapped()


__all__ = [
    "ErrorAccumulator",
    "fallback",
    "retry",
    "with_retry",
]
