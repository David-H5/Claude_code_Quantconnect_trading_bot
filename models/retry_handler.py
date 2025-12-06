"""
Retry Handler with Exponential Backoff for Trading Bot

DEPRECATED: This module is deprecated. Use utils.error_handling instead.

New import paths:
    from utils.error_handling import retry, fallback, ErrorAccumulator

This module provides backwards-compatible re-exports from utils.error_handling.

Provides:
- Decorator-based retry with exponential backoff
- Configurable jitter for retry distribution
- Exception filtering for selective retries
- Integration with error handler
- Async support

UPGRADE-012: Error Handling (December 2025)
Phase 3 Refactoring: Deprecated in favor of utils.error_handling
"""

from __future__ import annotations

import asyncio
import random
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    TypeVar,
)


# Deprecation warning
warnings.warn(
    "models.retry_handler is deprecated. Use utils.error_handling instead. "
    "Import: from utils.error_handling import retry, fallback",
    DeprecationWarning,
    stacklevel=2,
)

T = TypeVar("T")


# ==============================================================================
# Configuration
# ==============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: tuple[float, float] = (0.5, 1.5)
    exceptions: tuple[type[Exception], ...] = (Exception,)
    on_retry: Callable[[Exception, int], None] | None = None
    on_failure: Callable[[Exception, int], None] | None = None


# ==============================================================================
# Retry Decorator
# ==============================================================================


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
    on_failure: Callable[[Exception, int], None] | None = None,
) -> Callable:
    """
    Decorator for retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff calculation
        jitter: Add randomization to delay to prevent thundering herd
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Callback called on each retry (exception, attempt)
        on_failure: Callback called on final failure (exception, attempts)

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_with_backoff(max_retries=3, base_delay=1.0)
        ... def fetch_data(url: str) -> dict:
        ...     return requests.get(url).json()

        >>> @retry_with_backoff(
        ...     max_retries=5,
        ...     exceptions=(ConnectionError, TimeoutError),
        ...     on_retry=lambda e, n: print(f"Retry {n}: {e}")
        ... )
        ... def connect_to_broker():
        ...     pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # Check if we have more retries
                    if attempt < max_retries:
                        # Calculate delay
                        delay = calculate_delay(
                            attempt=attempt,
                            base_delay=base_delay,
                            max_delay=max_delay,
                            exponential_base=exponential_base,
                            jitter=jitter,
                        )

                        # Notify retry callback
                        if on_retry:
                            try:
                                on_retry(e, attempt + 1)
                            except Exception:
                                pass

                        # Wait before retry
                        time.sleep(delay)

            # All retries exhausted
            if on_failure and last_exception:
                try:
                    on_failure(last_exception, max_retries)
                except Exception:
                    pass

            raise last_exception  # type: ignore

        return wrapper

    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
    on_failure: Callable[[Exception, int], None] | None = None,
) -> Callable:
    """
    Async decorator for retry with exponential backoff.

    Same as retry_with_backoff but for async functions.

    Example:
        >>> @async_retry_with_backoff(max_retries=3)
        ... async def fetch_data_async(url: str) -> dict:
        ...     async with aiohttp.ClientSession() as session:
        ...         async with session.get(url) as response:
        ...             return await response.json()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        delay = calculate_delay(
                            attempt=attempt,
                            base_delay=base_delay,
                            max_delay=max_delay,
                            exponential_base=exponential_base,
                            jitter=jitter,
                        )

                        if on_retry:
                            try:
                                on_retry(e, attempt + 1)
                            except Exception:
                                pass

                        await asyncio.sleep(delay)

            if on_failure and last_exception:
                try:
                    on_failure(last_exception, max_retries)
                except Exception:
                    pass

            raise last_exception  # type: ignore

        return wrapper

    return decorator


# ==============================================================================
# Retry Handler Class
# ==============================================================================


class RetryHandler:
    """
    Retry handler with configurable behavior.

    Provides programmatic control over retry logic for more complex scenarios.

    Example:
        >>> handler = RetryHandler(max_retries=3, base_delay=1.0)
        >>> result = handler.execute(
        ...     lambda: api_call(),
        ...     exceptions=(TimeoutError,)
        ... )
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        error_handler: Any | None = None,
    ):
        """
        Initialize retry handler.

        Args:
            max_retries: Maximum retry attempts
            base_delay: Initial delay (seconds)
            max_delay: Maximum delay (seconds)
            exponential_base: Exponential backoff base
            jitter: Add randomization to delays
            error_handler: ErrorHandler for logging retries
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.error_handler = error_handler

        self._total_retries: int = 0
        self._successful_retries: int = 0
        self._failed_operations: int = 0

    def execute(
        self,
        operation: Callable[..., T],
        *args: Any,
        exceptions: tuple[type[Exception], ...] = (Exception,),
        **kwargs: Any,
    ) -> T:
        """
        Execute operation with retry logic.

        Args:
            operation: Callable to execute
            *args: Arguments to pass to operation
            exceptions: Exceptions to catch and retry
            **kwargs: Keyword arguments to pass to operation

        Returns:
            Result of operation

        Raises:
            Exception: Last exception if all retries fail
        """
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                result = operation(*args, **kwargs)
                if attempt > 0:
                    self._successful_retries += 1
                return result
            except exceptions as e:
                last_exception = e
                self._total_retries += 1

                if attempt < self.max_retries:
                    delay = calculate_delay(
                        attempt=attempt,
                        base_delay=self.base_delay,
                        max_delay=self.max_delay,
                        exponential_base=self.exponential_base,
                        jitter=self.jitter,
                    )

                    # Log retry if error handler available
                    if self.error_handler:
                        from .error_handler import ErrorCategory

                        self.error_handler.handle_error(
                            e,
                            category=ErrorCategory.TRANSIENT,
                            context={
                                "attempt": attempt + 1,
                                "max_retries": self.max_retries,
                                "delay": delay,
                            },
                        )

                    time.sleep(delay)

        self._failed_operations += 1
        raise last_exception  # type: ignore

    async def execute_async(
        self,
        operation: Callable[..., T],
        *args: Any,
        exceptions: tuple[type[Exception], ...] = (Exception,),
        **kwargs: Any,
    ) -> T:
        """
        Execute async operation with retry logic.

        Args:
            operation: Async callable to execute
            *args: Arguments to pass to operation
            exceptions: Exceptions to catch and retry
            **kwargs: Keyword arguments to pass to operation

        Returns:
            Result of operation
        """
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await operation(*args, **kwargs)
                if attempt > 0:
                    self._successful_retries += 1
                return result
            except exceptions as e:
                last_exception = e
                self._total_retries += 1

                if attempt < self.max_retries:
                    delay = calculate_delay(
                        attempt=attempt,
                        base_delay=self.base_delay,
                        max_delay=self.max_delay,
                        exponential_base=self.exponential_base,
                        jitter=self.jitter,
                    )

                    await asyncio.sleep(delay)

        self._failed_operations += 1
        raise last_exception  # type: ignore

    def get_stats(self) -> dict:
        """Get retry statistics."""
        return {
            "total_retries": self._total_retries,
            "successful_retries": self._successful_retries,
            "failed_operations": self._failed_operations,
            "success_rate": (self._successful_retries / self._total_retries if self._total_retries > 0 else 1.0),
        }

    def reset_stats(self) -> None:
        """Reset retry statistics."""
        self._total_retries = 0
        self._successful_retries = 0
        self._failed_operations = 0


# ==============================================================================
# Utility Functions
# ==============================================================================


def calculate_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    jitter_range: tuple[float, float] = (0.5, 1.5),
) -> float:
    """
    Calculate delay for retry attempt with exponential backoff.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter
        jitter_range: Range for jitter multiplier

    Returns:
        Delay in seconds
    """
    # Calculate exponential delay
    delay = min(
        base_delay * (exponential_base**attempt),
        max_delay,
    )

    # Add jitter to prevent thundering herd
    if jitter:
        jitter_multiplier = random.uniform(*jitter_range)
        delay *= jitter_multiplier

    return delay


def is_retryable_exception(
    exception: Exception,
    retryable_types: tuple[type[Exception], ...] = (),
) -> bool:
    """
    Check if an exception should be retried.

    Args:
        exception: The exception to check
        retryable_types: Tuple of retryable exception types

    Returns:
        True if exception is retryable
    """
    if retryable_types:
        return isinstance(exception, retryable_types)

    # Default retryable exceptions
    exception_name = type(exception).__name__.lower()
    exception_msg = str(exception).lower()

    retryable_keywords = [
        "timeout",
        "connection",
        "temporary",
        "transient",
        "retry",
        "unavailable",
        "too many requests",
        "rate limit",
    ]

    return any(keyword in exception_name or keyword in exception_msg for keyword in retryable_keywords)


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_retry_handler(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    error_handler: Any | None = None,
) -> RetryHandler:
    """
    Factory function to create a retry handler.

    Args:
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        error_handler: ErrorHandler for logging

    Returns:
        Configured RetryHandler instance
    """
    return RetryHandler(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        error_handler=error_handler,
    )
