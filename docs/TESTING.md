# Testing Best Practices

## Overview

This document outlines testing best practices for the QuantConnect Trading Bot project.
Current test suite: **3,604+ tests** across **127 test files**.

## Test Organization

### Directory Structure

```
tests/
├── conftest.py           # Shared fixtures and mock classes
├── analytics/            # Analytics module tests
├── backtesting/          # Walk-forward, Monte Carlo tests
├── compliance/           # Audit logging, anti-manipulation
├── hooks/                # Claude Code hook tests
├── infrastructure/       # Redis, streaming tests
├── mcp/                  # MCP server tests
├── observability/        # Metrics, token tracking
├── regression/           # Regression test suite
└── test_*.py             # Module-specific tests
```

### Naming Conventions

**Test Files:**
- `test_{module_name}.py` - Standard module tests
- `test_{module}_simple.py` - Structural/syntax tests only
- `test_{feature}_integration.py` - Integration tests

**Test Classes:**
- Use specific, descriptive names: `TestMetricsAggregatorThreadSafety` NOT `TestThreadSafety`
- Include module context in class names to avoid duplicate class names across files
- Pattern: `Test{Module}{Functionality}` (e.g., `TestCircuitBreakerHalt`)

**Test Functions:**
- `test_{what}_{scenario}` pattern
- Examples:
  - `test_calculate_greeks_valid_inputs`
  - `test_calculate_greeks_zero_time_to_expiry`
  - `test_order_submission_missing_symbol`

## Fixtures

### Scope Guidelines

| Scope | Use Case | Example |
|-------|----------|---------|
| `function` (default) | Mutable state, tests may modify | `mock_portfolio`, `mock_algorithm` |
| `class` | Immutable/read-only data shared in class | `mock_symbol`, `empty_slice` |
| `session` | Setup-only, expensive initialization | `timeout_warning` |

```python
# GOOD - Explicit scope with type hint
@pytest.fixture(scope="class")
def mock_symbol() -> MockSymbol:
    """Create a mock symbol (class-scoped, immutable)."""
    return MockSymbol("SPY")

# GOOD - Function scope for mutable state
@pytest.fixture
def mock_portfolio() -> MockPortfolio:
    """Create a mock portfolio (function-scoped, mutable state)."""
    return MockPortfolio()
```

### Using Shared Mocks

Import from `conftest.py` for custom mock construction:

```python
from tests.conftest import MockSymbol, MockPortfolio, MockTradeBar
```

**Available Mock Classes:**
- `MockSymbol` - Stock ticker
- `MockTradeBar` - OHLCV price data
- `MockSlice` - Market data container
- `MockIndicator` - Technical indicators
- `MockPortfolio` - Portfolio state
- `MockSecurities` - Security registry

## Timeouts

### Configuration

Default timeout: **30 seconds** (configured in `pytest.ini`)

### Per-Test Timeout Override

```python
@pytest.mark.timeout(60)  # 60 seconds for slow tests
def test_slow_operation():
    ...

@pytest.mark.timeout(0)  # Disable timeout
def test_manual_verification():
    ...
```

### Marker-Based Auto-Timeout

Tests with these markers get automatic timeouts:

| Marker | Timeout |
|--------|---------|
| `@pytest.mark.unit` | 10s |
| `@pytest.mark.integration` | 30s |
| `@pytest.mark.slow` | 120s |
| `@pytest.mark.stress` | 300s |
| `@pytest.mark.e2e` | 180s |
| `@pytest.mark.backtest` | 300s |
| `@pytest.mark.montecarlo` | 120s |

## Test Markers

Use markers to categorize tests:

```python
@pytest.mark.unit
def test_calculate_delta():
    """Fast, isolated unit test."""
    ...

@pytest.mark.integration
def test_full_pipeline():
    """Tests multiple components together."""
    ...

@pytest.mark.regression
def test_historical_bug_fix():
    """Ensures bug doesn't recur."""
    ...
```

**Available Markers:**
- `unit` - Fast, isolated tests
- `integration` - Multi-component tests
- `regression` - Core stability tests
- `lifecycle` - Order state machines
- `chaos` - Fault injection
- `montecarlo` - Monte Carlo simulations
- `performance` - Benchmarks
- `guardrail` - LLM safety tests
- `stress` - Extreme conditions

## Safety-Critical Testing

### Trading System Tests MUST Include:

1. **Boundary Testing**
   ```python
   def test_position_size_at_max_limit():
       """Test behavior exactly at position limit."""
       ...

   def test_position_size_exceeds_limit():
       """Test rejection when limit exceeded."""
       ...
   ```

2. **Error Recovery**
   ```python
   def test_recovers_from_network_failure():
       """Verify system continues after transient failure."""
       ...
   ```

3. **State Consistency**
   ```python
   def test_portfolio_state_after_partial_fill():
       """Verify portfolio is consistent after partial execution."""
       ...
   ```

### What NOT to Skip

Never skip without documenting why:
- Risk limit enforcement tests
- Order validation tests
- Authentication/authorization tests
- Data integrity tests

## Best Tests Examples

### Well-Structured Unit Test

```python
@pytest.mark.unit
def test_greeks_delta_at_the_money():
    """
    Test delta calculation for at-the-money call option.

    Delta should be approximately 0.5 for ATM options.
    """
    calc = create_greeks_calculator(risk_free_rate=0.05)

    greeks = calc.calculate(
        spot=100.0,
        strike=100.0,
        time_to_expiry=30 / 365,
        iv=0.20,
        option_type="call",
    )

    assert 0.45 <= greeks.delta <= 0.55, f"ATM delta should be ~0.5, got {greeks.delta}"
```

### Well-Structured Integration Test

```python
@pytest.mark.integration
class TestOrderExecutionPipeline:
    """Tests for complete order execution flow."""

    def test_order_validation_to_fill(self, mock_algorithm):
        """Test order flows from validation through execution to fill."""
        # Setup
        order = create_test_order(symbol="SPY", quantity=100)

        # Execute
        result = execute_order(mock_algorithm, order)

        # Verify each stage
        assert result.validation_passed
        assert result.order_submitted
        assert result.fill_received
        assert result.portfolio_updated
```

### Edge Case Testing

```python
@pytest.mark.unit
class TestCircuitBreakerEdgeCases:
    """Edge case tests for circuit breaker."""

    def test_halt_with_empty_reason(self):
        """Circuit breaker should accept empty reason but log warning."""
        breaker = TradingCircuitBreaker()
        breaker.halt_all_trading(reason="")
        assert not breaker.can_trade()

    def test_multiple_halt_calls(self):
        """Multiple halt calls should be idempotent."""
        breaker = TradingCircuitBreaker()
        breaker.halt_all_trading(reason="First")
        breaker.halt_all_trading(reason="Second")
        assert not breaker.can_trade()
```

## Anti-Patterns to Avoid

### 1. Generic Test Class Names

```python
# BAD - Duplicate class names across files
class TestThreadSafety:
    ...

# GOOD - Include module context
class TestMetricsAggregatorThreadSafety:
    ...
```

### 2. Missing Type Hints on Fixtures

```python
# BAD - No return type
@pytest.fixture
def mock_portfolio():
    return MockPortfolio()

# GOOD - With return type
@pytest.fixture
def mock_portfolio() -> MockPortfolio:
    return MockPortfolio()
```

### 3. Duplicating Mocks in Test Files

```python
# BAD - Redefining MockPortfolio in test file
class MockPortfolio:
    def __init__(self):
        self.TotalPortfolioValue = 100000

# GOOD - Import from conftest
from tests.conftest import MockPortfolio
```

### 4. Quarantining Instead of Fixing

```python
# BAD - Permanent quarantine
@pytest.mark.skip(reason="QUARANTINED: Deadlock in global lock")
def test_concurrent_access():
    ...

# GOOD - Fix the issue or document plan
@pytest.mark.skip(reason="TODO #123: Fix RLock deadlock, ETA: Sprint 5")
def test_concurrent_access():
    ...
```

## Running Tests

### Common Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific marker
pytest -m unit
pytest -m "integration and not slow"

# Run specific file
pytest tests/test_circuit_breaker.py

# Run with verbose output
pytest -v

# Stop on first failure
pytest -x

# Run parallel (if pytest-xdist installed)
pytest -n auto
```

### CI Configuration

The test suite is configured for CI with:
- **Timeout protection**: 30s default prevents hangs
- **Strict markers**: Typos in markers cause errors
- **Coverage tracking**: HTML + terminal reports

## Known Issues & Technical Debt

### ~~Duplicate Class Names~~ (FIXED)

Previously duplicate class names have been renamed to include module context:
- `TestThreadSafety` → `Test{Module}ThreadSafety` (e.g., `TestStructuredLoggingThreadSafety`)
- `TestFactoryFunctions` → `Test{Module}FactoryFunctions` (e.g., `TestErrorHandlingFactoryFunctions`)
- `TestIntegration` → `Test{Module}Integration` (e.g., `TestAlertingServiceIntegration`)

### Similar Test Files

These file pairs test related functionality:
- `test_walk_forward.py` vs `backtesting/test_walk_forward.py`
- `test_monte_carlo.py` vs `backtesting/test_monte_carlo.py`

**Impact:** None - tests are complementary, not duplicated.

## Contributing Tests

1. Use descriptive test names
2. Add docstrings explaining test purpose
3. Use appropriate markers
4. Follow fixture scope guidelines
5. Avoid duplicating mock classes
6. Test edge cases and error conditions
7. Include cleanup in fixtures
