# Coding Standards

This document defines the coding standards for all Python code in this project.

---

## Table of Contents

1. [Style Guide](#1-style-guide)
2. [Type Hints](#2-type-hints)
3. [Documentation](#3-documentation)
4. [Error Handling](#4-error-handling)
5. [Logging](#5-logging)
6. [Testing](#6-testing)
7. [Security](#7-security)

---

## 1. Style Guide

### Formatting

We use **Black** for code formatting with line length of 100.

```bash
black . --line-length 100
```

### Import Order

We use **isort** with Black profile:

```python
# Standard library
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np
import pandas as pd

# Local
from models import RiskManager
from utils import calculate_position_size
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `TradingCircuitBreaker` |
| Functions | snake_case | `calculate_position_size` |
| Variables | snake_case | `daily_loss_pct` |
| Constants | UPPER_SNAKE | `MAX_POSITION_SIZE` |
| Private | _prefix | `_internal_state` |
| Module | snake_case | `circuit_breaker.py` |

### Line Length

- Maximum: **100 characters** (enforced by Black)
- Soft limit: **80 characters** for readability
- Ruler at: **100, 120** (visible in VS Code)

---

## 2. Type Hints

### Required on All Public Functions

```python
def calculate_position_size(
    portfolio_value: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
) -> float:
    """Calculate position size based on risk."""
    ...
```

### Use `typing` Module for Complex Types

```python
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Correct for Python 3.8+ compatibility
def get_signals(data: Dict[str, Any]) -> List[Tuple[str, float]]:
    ...

# Use Optional for nullable
def get_config(key: str, default: Optional[str] = None) -> Optional[str]:
    ...
```

### Type Aliases for Clarity

```python
from typing import Dict, List, Tuple

# Define type aliases for complex types
Symbol = str
Price = float
Quantity = int
Signal = Tuple[Symbol, Price, Quantity]
SignalList = List[Signal]

def generate_signals(data: Dict[Symbol, Price]) -> SignalList:
    ...
```

---

## 3. Documentation

### Docstrings (Google Style)

```python
def calculate_position_size(
    portfolio_value: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
    max_position_pct: float = 0.25,
) -> float:
    """Calculate position size based on risk management rules.

    Uses fixed fractional position sizing with a maximum position cap.
    Position size is calculated to risk exactly risk_per_trade amount
    if the stop loss is hit.

    Args:
        portfolio_value: Current portfolio value in dollars.
        risk_per_trade: Maximum amount to risk on this trade in dollars.
        entry_price: Planned entry price per share.
        stop_loss_price: Stop loss price per share.
        max_position_pct: Maximum position size as fraction of portfolio.
            Defaults to 0.25 (25%).

    Returns:
        Position size in number of shares, floored to integer.

    Raises:
        ValueError: If portfolio_value <= 0 or prices are invalid.

    Example:
        >>> size = calculate_position_size(100000, 2000, 50, 45)
        >>> size
        400
    """
```

### Module Docstrings

```python
"""
Circuit Breaker Module

Provides automatic trading halt functionality when risk thresholds
are breached. This is a critical safety mechanism for autonomous
trading systems.

Classes:
    TradingCircuitBreaker: Main circuit breaker implementation.
    CircuitBreakerConfig: Configuration dataclass.

Functions:
    create_circuit_breaker: Factory function for common configurations.

Example:
    >>> from models.circuit_breaker import create_circuit_breaker
    >>> breaker = create_circuit_breaker(max_daily_loss=0.03)
    >>> breaker.can_trade()
    True
"""
```

### Class Docstrings

```python
class TradingCircuitBreaker:
    """
    Circuit breaker that halts trading when risk limits are breached.

    The circuit breaker monitors daily loss, drawdown, and consecutive
    losses. When any threshold is exceeded, trading is halted until
    manually reset (if require_human_reset is True).

    Attributes:
        config: CircuitBreakerConfig with threshold settings.
        state: Current state (CLOSED, OPEN, HALF_OPEN).
        consecutive_losses: Count of consecutive losing trades.

    Example:
        >>> breaker = TradingCircuitBreaker(config)
        >>> if breaker.can_trade():
        ...     execute_trade()
        >>> breaker.record_trade_result(is_winner=True)
    """
```

---

## 4. Error Handling

### Use Specific Exception Types

```python
# WRONG - Catches everything silently
try:
    result = provider.analyze(text)
except Exception:
    pass

# CORRECT - Specific exceptions with logging
try:
    result = provider.analyze(text)
except ConnectionError as e:
    logger.warning("Provider connection failed: %s", e)
    return default_result
except TimeoutError as e:
    logger.warning("Provider timeout: %s", e)
    return default_result
except Exception as e:
    logger.error("Unexpected error: %s", e, exc_info=True)
    raise
```

### Exception Hierarchy

```python
# Define custom exceptions for your domain
class TradingError(Exception):
    """Base exception for trading errors."""
    pass

class RiskLimitExceeded(TradingError):
    """Raised when a risk limit is exceeded."""
    pass

class InsufficientFunds(TradingError):
    """Raised when insufficient funds for trade."""
    pass

class OrderRejected(TradingError):
    """Raised when order is rejected by broker."""
    pass
```

### Fail-Fast for Critical Errors

```python
def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration, fail fast on errors."""
    required_keys = ["max_position_size", "max_daily_loss"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    if config["max_position_size"] > 1.0:
        raise ValueError("max_position_size cannot exceed 1.0 (100%)")
```

---

## 5. Logging

### Use Module-Level Logger

```python
import logging

logger = logging.getLogger(__name__)

class MyClass:
    def my_method(self):
        logger.info("Processing started")
        logger.debug("Details: %s", details)
        logger.warning("Potential issue: %s", issue)
        logger.error("Error occurred: %s", error)
```

### Never Use `print()` for Logging

```python
# WRONG
print(f"Error: {error}")

# CORRECT
logger.error("Error: %s", error)
```

### Never Configure Logging at Module Level

```python
# WRONG - Interferes with application logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORRECT - Let application configure
logger = logging.getLogger(__name__)
```

### Log Format for Trading

```python
# Order events
logger.info("ORDER_SUBMITTED: %s %s %d @ %.2f", symbol, side, qty, price)
logger.info("ORDER_FILLED: %s @ %.2f qty %d", order_id, fill_price, fill_qty)
logger.info("ORDER_CANCELLED: %s - %s", order_id, reason)

# Risk events
logger.info("RISK_CHECK: daily_loss=%.2f%% limit=%.2f%%", loss, limit)
logger.warning("RISK_BREACH: %s - %s", check_name, details)

# Circuit breaker
logger.warning("CIRCUIT_BREAKER_TRIP: %s", reason)
logger.info("CIRCUIT_BREAKER_RESET: by=%s", authorized_by)
```

---

## 6. Testing

### Test File Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_algorithms.py       # Algorithm tests
├── test_circuit_breaker.py  # Risk management tests
├── test_indicators.py       # Technical indicators
├── test_utils.py            # Utility functions
└── fixtures/
    └── sample_data.py       # Test data generators
```

### Test Naming

```python
class TestCircuitBreaker:
    """Tests for TradingCircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Circuit breaker should start in CLOSED state."""
        ...

    def test_trips_on_daily_loss_exceeded(self):
        """Should trip when daily loss exceeds limit."""
        ...

    def test_requires_human_reset_when_configured(self):
        """Should require human authorization to reset."""
        ...
```

### Use Markers

```python
import pytest

@pytest.mark.unit
def test_fast_operation():
    """Quick unit test."""
    ...

@pytest.mark.integration
def test_full_workflow():
    """Integration test requiring multiple components."""
    ...

@pytest.mark.slow
def test_backtest():
    """Slow test, skip in CI unless needed."""
    ...
```

### Coverage Requirements

- **Minimum**: 70% overall coverage
- **Critical modules**: 90%+ coverage (risk_manager, circuit_breaker)
- **Run with**: `pytest --cov=algorithms --cov=models --cov=utils`

---

## 7. Security

### Never Hardcode Secrets

```python
# WRONG
api_key = "sk-12345abcdef"

# CORRECT
api_key = os.environ.get("API_KEY")
if not api_key:
    raise EnvironmentError("API_KEY environment variable required")
```

### Use Environment Variable Substitution

```json
{
  "api_key": "${API_KEY}",
  "secret": "${API_SECRET}"
}
```

### Validate External Input

```python
def process_webhook(data: Dict[str, Any]) -> None:
    """Process incoming webhook data."""
    # Validate required fields
    if "symbol" not in data:
        raise ValueError("Missing required field: symbol")

    # Validate types
    if not isinstance(data["quantity"], (int, float)):
        raise TypeError("quantity must be numeric")

    # Validate ranges
    if data["quantity"] <= 0:
        raise ValueError("quantity must be positive")

    # Sanitize strings
    symbol = data["symbol"].upper().strip()[:10]
```

### Protected Files

The following files should NEVER be committed or accessed by automated tools:

- `.env`, `.env.local`, `.env.production`
- `config/credentials.json`, `config/api_keys.json`
- `secrets.json`
- Any file containing `credential`, `secret`, `password`, `token`, `private`
- `.pem`, `.key` files

---

## Tools Configuration

### Pre-commit Hooks

All code must pass pre-commit hooks:

```bash
pre-commit install
pre-commit run --all-files
```

### VS Code Settings

Project settings are in `.vscode/settings.json`:

- Auto-format on save (Black)
- Auto-sort imports (isort)
- Type checking (Pylance)
- Linting (Flake8)

### Flake8 Configuration

See `.flake8` for settings:

- Max line length: 100
- Max complexity: 15
- Ignore rules conflicting with Black

### MyPy Configuration

See `mypy.ini` for settings:

- Python 3.10 target
- Strict optional checking
- Ignore missing imports for third-party

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-29 | Initial release |

---

*All code in this project must follow these standards. Use `pre-commit run --all-files` before committing.*
