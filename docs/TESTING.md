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

## Safety-Critical vs Bloat Analysis

### Safety-Critical Tests (NEVER SKIP)

These 36 test files contain safety-critical tests for trading systems:

| Category | Files | Key Tests |
|----------|-------|-----------|
| **Circuit Breaker** | `test_circuit_breaker.py` | Halt functionality, level triggers (7%/13%/20%) |
| **Risk Management** | `test_risk_management.py`, `hooks/test_risk_validator.py` | Position limits, max loss, daily limits |
| **Compliance** | `compliance/test_audit_logger.py`, `compliance/test_anti_manipulation.py` | Trade logging, wash trade detection |
| **Pre-Trade Validation** | `test_pre_trade_validator.py` | Order validation, limit checks |
| **Regression** | `regression/test_historical_bugs.py`, `regression/test_edge_cases.py` | Known bug prevention |

**Safety Test Requirements:**
- All assertions must be explicit (not just mock assertions)
- Must test both positive and negative cases
- Must include boundary conditions
- Must be marked with `@pytest.mark.regression` if fixing known issues

### Potential Bloat (Review Periodically)

| Category | Count | Action |
|----------|-------|--------|
| Tests without `assert` statements | 77 | Review - may use mock assertions |
| Very short tests (1-2 lines) | 53 | Verify meaningful coverage |
| Optional dependency skips (PySide6) | 17 | Consider CI matrix |
| Threading tests quarantined | 2 classes | Investigate root cause |

### Test Quality Metrics

```
Total Test Functions:     3,558
Safety-Critical Tests:    ~400 (11%)
Tests with Assertions:    3,481 (98%)
Skipped Tests:            28 (0.8%)
Coverage:                 67%
```

## Quarantined Tests

### Threading Tests (CI Hangs)

Two threading test classes are quarantined due to CI timeouts:

1. `tests/test_token_metrics.py::TestTokenMetricsThreadSafety`
2. `tests/test_metrics_aggregator.py::TestMetricsAggregatorThreadSafety`

**Root Cause:** Not deadlocks - the underlying code uses proper locking (including RLock for nested locks). Issue is likely CI resource contention or timeout settings.

**Workaround:** Tests pass locally but hang in CI environment.

### Optional Dependency Skips

Tests skipped due to missing optional dependencies:
- **PySide6** (17 tests): GUI components
- **SHAP/LIME** (4 tests): ML explainability
- **PRAW** (1 test): Reddit API

## Testing Framework Utilities

### Available Utilities in conftest.py

Import these utilities for consistent, reusable testing patterns:

```python
from tests.conftest import (
    # Assertion helpers
    assert_dataclass_to_dict,
    assert_config_defaults,
    assert_factory_creates_valid,
    assert_in_range,
    # Safety testing
    SafetyTestCase,
    # Fault injection
    FaultInjector,
    # Market scenarios
    create_market_crash_scenario,
    create_volatile_scenario,
    create_gap_scenario,
    # Regression
    regression_test,
)
```

### SafetyTestCase Base Class

Use for safety-critical tests:

```python
class TestCircuitBreakerSafety(SafetyTestCase):
    """Safety tests for circuit breaker."""

    def test_halt_prevents_all_trading(self, circuit_breaker):
        circuit_breaker.halt_all_trading(reason="Test")
        self.assert_safety_invariant(
            not circuit_breaker.can_trade(),
            "Halted breaker must prevent trading"
        )

    def test_position_within_limits(self, risk_manager, position):
        self.assert_position_valid(
            position.quantity,
            risk_manager.max_position,
            symbol=position.symbol
        )
```

### FaultInjector for Chaos Testing

```python
@pytest.mark.chaos
def test_handles_network_failures(fault_injector):
    """Test system resilience to network failures."""
    for _ in range(10):
        if fault_injector.should_fail():
            with pytest.raises(NetworkError):
                submit_order(order)
        else:
            result = submit_order(order)
            assert result.success
```

### Market Scenario Generators

```python
@pytest.mark.stress
def test_circuit_breaker_triggers_on_crash():
    """Test circuit breaker activates during market crash."""
    crash_bars = create_market_crash_scenario(
        initial_price=100.0,
        crash_pct=0.20,  # 20% crash
        num_bars=10
    )

    for bar in crash_bars:
        breaker.check(bar)

    assert not breaker.can_trade()

@pytest.mark.stress
def test_handles_gap_down():
    """Test gap risk handling."""
    pre_bar, post_bar = create_gap_scenario(
        pre_gap_price=100.0,
        gap_pct=0.10,
        direction="down"
    )

    assert post_bar.Open < pre_bar.Close * 0.95
```

### Test Data Builders

Use fluent builders for creating test data (from `tests/builders.py`):

```python
from tests.builders import (
    OrderBuilder,
    PositionBuilder,
    PortfolioBuilder,
    PriceHistoryBuilder,
    ScenarioBuilder,
    create_simple_order,
    create_crash_scenario,
)

# Fluent builder pattern
order = (OrderBuilder()
    .with_symbol("SPY")
    .buy()
    .limit(450.00)
    .quantity(100)
    .filled()
    .build())

# Position with gain
position = (PositionBuilder()
    .symbol("AAPL")
    .long(100)
    .at_price(150.00)
    .with_gain(0.05)  # 5% profit
    .build())

# Portfolio in drawdown
portfolio = (PortfolioBuilder()
    .with_cash(50000)
    .starting_equity(100000)
    .add_position(position)
    .in_drawdown(0.10)  # 10% drawdown
    .build())

# Price history with crash
crash_bars = (PriceHistoryBuilder()
    .symbol("SPY")
    .starting_price(450.0)
    .trending_up(0.002)
    .with_crash(50, 0.20)  # 20% crash at bar 50
    .num_bars(100)
    .with_seed(42)  # Reproducible
    .build())

# Convenience functions
order = create_simple_order("SPY", "buy", 100, price=450.0)
bars = create_crash_scenario(starting_price=100, crash_pct=0.20)
```

### Property-Based Testing

Use Hypothesis strategies for fuzzing (from `tests/strategies.py`):

```python
from hypothesis import given, settings
from tests.strategies import (
    valid_price,
    valid_quantity,
    valid_order,
    valid_position,
    edge_case_prices,
    crash_scenario,
    check_price_invariants,
)

@given(price=valid_price())
def test_price_always_positive(price):
    """Property: processed prices should never be negative."""
    result = process_price(price)
    assert result >= 0

@given(order=valid_order())
def test_order_validation_accepts_valid_orders(order):
    """Property: valid orders should pass validation."""
    result = validator.validate(order)
    assert result.is_valid or result.has_expected_rejection

@given(scenario=crash_scenario())
@settings(max_examples=50)
def test_circuit_breaker_trips_on_large_crashes(scenario):
    """Property: circuit breaker should trip on crashes > 10%."""
    if scenario["crash_pct"] > 0.10:
        breaker = create_breaker(max_daily_loss_pct=0.10)
        simulate_crash(breaker, scenario)
        assert not breaker.can_trade()

# Edge case testing
@given(price=edge_case_prices())
def test_handles_boundary_prices(price):
    """Test handling of boundary/edge case prices."""
    # Should not crash
    result = process_price_safely(price)
    assert result is not None
```

**Available Strategies:**
- `valid_price()` - Realistic stock prices
- `valid_quantity()` - Order quantities
- `valid_percentage()` - 0.0 to 1.0 values
- `valid_symbol()` - Common stock tickers
- `valid_order()` - Complete order dicts
- `valid_position()` - Position dicts
- `valid_trade_bar()` - OHLCV bars with invariants
- `valid_risk_limits()` - Risk config dicts
- `edge_case_prices()` - Boundary values
- `crash_scenario()` - Market crash configs
- `gap_scenario()` - Gap up/down configs

### Parametrized Test Patterns

Replace duplicate tests with parametrization:

```python
# BEFORE: 18 duplicate test_default_config functions
def test_default_config(self):
    config = WalkForwardConfig()
    assert config.method == "rolling"

# AFTER: Single parametrized test
@pytest.mark.parametrize("config_class,defaults", [
    (WalkForwardConfig, {"method": "rolling", "num_windows": 5}),
    (MonteCarloConfig, {"num_simulations": 10000}),
    (LiquidityConfig, {"excellent_spread_pct": 0.02}),
])
def test_config_defaults(config_class, defaults):
    assert_config_defaults(config_class, defaults)
```

### Regression Test Decorator

```python
from tests.conftest import regression_test

@regression_test("BUG-123", "Division by zero with zero equity")
def test_zero_equity_handling():
    """Regression test for BUG-123."""
    manager = RiskManager()
    manager.portfolio_value = 0
    # Should not raise, should return safe default
    result = manager.calculate_position_size(100)
    assert result == 0
```

## Test Consolidation Opportunities

### Duplicate Files to Review

| Root File | Module File | Status |
|-----------|-------------|--------|
| `test_walk_forward.py` (445 lines) | `backtesting/test_walk_forward.py` (352 lines) | Complementary |
| `test_monte_carlo.py` (840 lines) | `backtesting/test_monte_carlo.py` (364 lines) | Complementary |

**Recommendation:** These files test different aspects:
- Root files: Comprehensive standalone tests with manual dataclasses
- Module files: Factory-based tests with proper imports

### Parametrization Opportunities

| Pattern | Current Count | Recommended |
|---------|--------------|-------------|
| `test_to_dict` | 61 instances | 1 parametrized test |
| `test_default_config` | 18 instances | 1 parametrized test |
| `test_creation` | 28 instances | 1 parametrized test |
| `test_create_*` variations | 40+ instances | 3-5 parametrized tests |

### Consolidation Priorities

1. **High Priority (Immediate)**
   - Use `assert_dataclass_to_dict()` helper instead of duplicating
   - Use `assert_config_defaults()` for all config tests
   - Use `assert_factory_creates_valid()` for factory tests
   - Import mocks from `tests/mocks/` instead of redefining

2. **Medium Priority (Next Sprint)**
   - Create base test classes for monitors (8 files share patterns)
   - Create base test classes for optimizers (7 files share patterns)
   - Run duplicate analysis with `analyze_test_duplicates()`

3. **Low Priority (Technical Debt)**
   - Reorganize root-level tests into module directories
   - Add more parametrized test cases

### New Framework Components (Latest)

| Component | Location | Purpose |
|-----------|----------|---------|
| Mock Registry | `tests/mocks/__init__.py` | Central import for all mocks |
| QC Mocks | `tests/mocks/quantconnect.py` | `MockQCAlgorithm`, `MockQCPortfolio`, etc. |
| Duplicate Finder | `tests/analysis/duplicate_finder.py` | `DuplicateFinder`, `analyze_test_duplicates()` |

**Mock Consolidation Example:**
```python
# BEFORE (duplicated in multiple files)
class MockPortfolio:
    def __init__(self):
        self.TotalPortfolioValue = 100000.0

# AFTER (import from centralized location)
from tests.mocks.quantconnect import MockQCPortfolio
```

**Duplicate Analysis Example:**
```python
from tests.analysis import analyze_test_duplicates
print(analyze_test_duplicates("tests/"))
```

## Safety-Critical Test Gaps (Action Items)

### Circuit Breaker Gaps

| Gap | Severity | Recommended Test |
|-----|----------|------------------|
| Half-open state transitions | CRITICAL | `test_half_open_state_allows_limited_trading` |
| Multiple simultaneous trips | HIGH | `test_first_trip_reason_preserved` |
| Cooldown boundary conditions | MEDIUM | `test_cooldown_exact_expiration` |

### Risk Management Gaps

| Gap | Severity | Recommended Test |
|-----|----------|------------------|
| Zero equity handling | CRITICAL | `test_position_size_with_zero_equity` |
| Negative portfolio values | HIGH | `test_drawdown_with_negative_equity` |
| Concurrent position updates | HIGH | `test_thread_safe_position_updates` |

### Pre-Trade Validation Gaps

| Gap | Severity | Recommended Test |
|-----|----------|------------------|
| Price change during validation | HIGH | `test_price_staleness_detection` |
| Concurrent validation race | HIGH | `test_concurrent_validation_isolation` |
| Combo order ratio validation | MEDIUM | `test_butterfly_unequal_quantities_rejected` |

### Audit Logger Gaps

| Gap | Severity | Recommended Test |
|-----|----------|------------------|
| Hash chain tampering | CRITICAL | `test_detects_modified_entry` |
| Concurrent writes | HIGH | `test_thread_safe_logging` |
| Retention policy enforcement | MEDIUM | `test_old_entries_purged` |

## Contributing Tests

1. Use descriptive test names
2. Add docstrings explaining test purpose
3. Use appropriate markers
4. Follow fixture scope guidelines
5. Avoid duplicating mock classes
6. Test edge cases and error conditions
7. Include cleanup in fixtures
8. **Never skip safety-critical tests without issue tracking**
9. **Prefer explicit assertions over mock assertions**
10. **Use framework utilities** (`assert_dataclass_to_dict`, `SafetyTestCase`, etc.)
11. **Prefer parametrization** over duplicate test functions
12. **Document regression tests** with bug IDs using `@regression_test` decorator

---

## Compiled Best Practices Summary

### The 10 Best Tests Pattern

Based on analysis of the 3,600+ test codebase, the most effective tests share these characteristics:

| Rank | Pattern | Example | Why It's Effective |
|------|---------|---------|-------------------|
| 1 | **Safety invariant tests** | `test_circuit_breaker_prevents_trading_when_tripped` | Catches critical failures |
| 2 | **Boundary condition tests** | `test_position_at_exactly_max_limit` | Finds off-by-one errors |
| 3 | **Error recovery tests** | `test_recovers_from_network_timeout` | Validates resilience |
| 4 | **State machine tests** | `test_order_lifecycle_pending_to_filled` | Validates transitions |
| 5 | **Concurrent access tests** | `test_thread_safe_portfolio_updates` | Catches race conditions |
| 6 | **Integration flow tests** | `test_order_validation_through_execution` | Validates real paths |
| 7 | **Property-based tests** | `@given(valid_order()) def test_validation()` | Discovers edge cases |
| 8 | **Regression tests** | `@regression_test("BUG-123") def test_zero_equity()` | Prevents recurrence |
| 9 | **Chaos/fault tests** | `test_handles_corrupted_data` | Validates robustness |
| 10 | **Cross-module tests** | `test_circuit_breaker_trip_audit_logged` | Validates integration |

### Safety vs Bloat Decision Framework

```
Is this test safety-critical?
├── YES → NEVER REMOVE
│   ├── Circuit breaker functionality
│   ├── Risk limit enforcement
│   ├── Order validation
│   ├── Compliance/audit logging
│   └── Data integrity checks
│
└── NO → Evaluate bloat criteria:
    ├── Does it duplicate another test? → CONSOLIDATE
    ├── Is it testing trivial behavior? → CONSIDER REMOVING
    ├── Does it have no assertions? → FIX or REMOVE
    ├── Is it perpetually skipped? → FIX or REMOVE
    └── Does it test implementation details? → REFACTOR
```

### Test Consolidation Rules

| Current State | Action | Tool |
|--------------|--------|------|
| 61+ `test_to_dict` functions | Use parametrized test | `assert_dataclass_to_dict()` |
| 18+ `test_default_config` functions | Use parametrized test | `assert_config_defaults()` |
| 40+ `test_create_*` functions | Use parametrized test | `assert_factory_creates_valid()` |
| Duplicate mock classes | Import from conftest | `from tests.conftest import Mock*` |
| Repetitive setup code | Use fixtures | `@pytest.fixture` |
| Similar test scenarios | Use builders | `OrderBuilder().buy().build()` |

### Framework Files Reference

| File | Purpose | Key Exports |
|------|---------|-------------|
| `tests/conftest.py` | Shared fixtures & utilities | `SafetyTestCase`, `FaultInjector`, `assert_*` helpers |
| `tests/builders.py` | Test data builders | `OrderBuilder`, `PortfolioBuilder`, `PriceHistoryBuilder` |
| `tests/strategies.py` | Property-based strategies | `valid_price()`, `valid_order()`, `crash_scenario()` |
| `tests/state_machines/test_order_lifecycle.py` | State machine tests | `OrderStateMachine`, lifecycle transition tests |
| `tests/performance/tracker.py` | Performance regression | `PerformanceTracker`, `@benchmark` decorator |
| `tests/snapshots/manager.py` | Snapshot testing | `SnapshotManager`, `@snapshot_test` decorator |
| `tests/regression/test_safety_critical_gaps.py` | Safety gap tests | Gap tests for CB, RM, AL, PTV |
| `tests/test_parametrized_examples.py` | Consolidation templates | Parametrized test examples |

### State Machine Testing

Test complex state transitions with `OrderStateMachine`:

```python
from tests.state_machines.test_order_lifecycle import OrderStateMachine, OrderState

def test_order_lifecycle():
    """Test valid order state transitions."""
    sm = OrderStateMachine()
    assert sm.state == OrderState.PENDING

    # Submit order
    assert sm.submit()
    assert sm.state == OrderState.SUBMITTED

    # Partial fills
    sm._total_quantity = 100
    assert sm.partial_fill(50)
    assert sm.state == OrderState.PARTIALLY_FILLED

    # Complete fill
    assert sm.partial_fill(50)
    assert sm.state == OrderState.FILLED
    assert sm.is_terminal

def test_invalid_transitions_blocked():
    """Test invalid transitions are rejected."""
    sm = OrderStateMachine()
    assert not sm.fill()  # Can't fill without submitting
    assert sm.state == OrderState.PENDING
```

### Performance Regression Testing

Track performance metrics with `PerformanceTracker`:

```python
from tests.performance.tracker import PerformanceTracker, benchmark

# Using context manager
tracker = PerformanceTracker()

with tracker.measure("order_placement", threshold_ms=50):
    place_order(order)

# Using decorator
@benchmark("risk_calculation", threshold_ms=100, iterations=100)
def test_risk_calculation_performance():
    calculate_risk(portfolio)

# Track and save metrics
tracker.save()
```

### Snapshot Testing

Compare complex outputs against stored snapshots:

```python
from tests.snapshots.manager import SnapshotManager

def test_portfolio_report_format(snapshot_manager):
    """Test portfolio report matches expected format."""
    report = generate_portfolio_report(portfolio)
    snapshot_manager.assert_matches("portfolio_report_standard", report)

# Run with --snapshot-update to update snapshots:
# pytest tests/ --snapshot-update
```

### Test Quality Checklist

Before committing new tests:

- [ ] Test has descriptive name following `test_{what}_{scenario}` pattern
- [ ] Test has docstring explaining purpose
- [ ] Test uses appropriate marker (`@pytest.mark.unit`, etc.)
- [ ] Test has explicit assertions (not just mock calls)
- [ ] Safety tests inherit from `SafetyTestCase`
- [ ] Test uses builders for complex data setup
- [ ] Test doesn't duplicate existing tests (check for similar tests first)
- [ ] Regression tests use `@regression_test()` decorator with bug ID
- [ ] Test handles both success and failure cases
- [ ] Boundary conditions are tested separately

### Final Metrics

```
Test Suite Statistics (Final):
├── Total Tests:              3,604+
├── Test Files:               135+ (added state machines, performance, snapshots, mocks, analysis)
├── Shared Fixtures:          342 across 92 files
├── Safety-Critical Tests:    ~85 core safety + ~62 regression markers
├── Tests with Assertions:    98%
├── Skip Rate:                0.8%
├── Coverage:                 67%
├── Framework Components:
│   ├── Test Data Builders:   5 (Order, Position, Portfolio, PriceHistory, Scenario)
│   ├── Hypothesis Strategies: 15+ property-based generators
│   ├── State Machines:       2 (Order lifecycle, Circuit breaker)
│   ├── Assertion Helpers:    6 (safety, range, dataclass, config, factory)
│   ├── Market Generators:    3 (crash, volatile, gap scenarios)
│   ├── Mock Registry:        1 centralized + 10+ QC mocks
│   └── Duplicate Analyzer:   1 (DuplicateFinder tool)
└── Recommended Actions:
    ├── Consolidate:          ~85 files with duplicate patterns → parametrized tests
    ├── Use Mocks:            Import from tests/mocks/ instead of redefining
    ├── Use Builders:         Replace manual setup with fluent API
    ├── Property-Based:       Extend to all validators
    └── Run Analysis:         Use DuplicateFinder to identify opportunities
```

### Test Coverage by Category

| Category | Test Count | Safety Coverage | Status |
|----------|------------|-----------------|--------|
| Circuit Breaker | 92+ | CRITICAL | Good |
| Risk Management | 69+ | CRITICAL | Good |
| Pre-Trade Validation | 29+ | HIGH | Good |
| Order Lifecycle | 25+ | HIGH | Good |
| Audit/Compliance | 40+ | HIGH | Good |
| Options Pricing | 50+ | MEDIUM | Good |
| API Integration | 45+ | MEDIUM | Good |
| State Machines | 20+ | HIGH | New |
| Performance | 10+ | MEDIUM | New |

### Duplicate Test Consolidation Progress

| Pattern | Before | After | Reduction |
|---------|--------|-------|-----------|
| `test_to_dict` | 61 | Use `assert_dataclass_to_dict()` | -61 |
| `test_default_config` | 18 | Use `assert_config_defaults()` | -18 |
| `test_creation` | 28 | Use `assert_factory_creates_valid()` | -28 |
| Manual mock setup | 100+ | Use builders | Ongoing |
| **Total** | **140+** | **Helpers available** | **Framework ready** |

---

## Safety vs Bloat Analysis (Compiled Best Practices)

### Classification Framework

Tests are classified into three tiers based on their criticality:

| Tier | Description | Action | Examples |
|------|-------------|--------|----------|
| **TIER 1: Safety-Critical** | Prevents financial loss, regulatory violations | NEVER REMOVE | Circuit breakers, risk limits, compliance |
| **TIER 2: Correctness** | Ensures correct behavior | Consolidate if duplicate | Order lifecycle, calculations, state machines |
| **TIER 3: Convenience** | Nice-to-have, may duplicate | Remove/consolidate aggressively | test_to_dict, test_creation patterns |

### Current Safety Coverage (TIER 1)

```
Safety-Critical Test Distribution:
├── Circuit Breaker Tests:      107 tests (test_circuit_breaker.py)
├── Risk Management Tests:      102 tests (test_risk_management.py)
├── Pre-Trade Validation:       28 tests (test_pre_trade_validator.py)
├── Compliance/Audit:           40+ tests (compliance/)
├── Safety Gap Regression:      22 tests (regression/test_safety_critical_gaps.py)
├── Historical Bug Regression:  23 tests (regression/test_historical_bugs.py)
└── Total Safety-Critical:      ~320 tests (NEVER REMOVE)
```

### Bloat Candidates (TIER 3)

```
Consolidation Opportunities:
├── test_to_dict patterns:      61 instances → 1 parametrized test
├── test_default_config:        18 instances → 1 parametrized test
├── test_creation patterns:     28 instances → 1 parametrized test
├── Duplicate mock classes:     9 classes → import from centralized
├── Duplicate fixtures:         2 fixtures → move to conftest.py
└── Total Bloat Reduction:      ~118 individual tests → ~5 parametrized
```

### Best Practices Hierarchy

#### 1. Safety-Critical Tests (MUST DO)

```python
# ALWAYS include for trading systems:
- test_circuit_breaker_halts_on_loss_threshold()
- test_position_size_cannot_exceed_limit()
- test_max_daily_loss_enforced()
- test_unauthorized_trades_rejected()
- test_audit_log_immutable()
```

#### 2. Correctness Tests (SHOULD DO)

```python
# Use SafetyTestCase base class:
class TestRiskLimits(SafetyTestCase):
    def test_position_within_limits(self, risk_manager, position):
        self.assert_position_valid(
            position.quantity,
            risk_manager.max_position,
            symbol=position.symbol
        )
```

#### 3. Convenience Tests (CONSOLIDATE)

```python
# BEFORE (61 separate tests):
def test_to_dict_order_config():
    config = OrderConfig()
    assert "symbol" in config.to_dict()

# AFTER (1 parametrized test):
@pytest.mark.parametrize("class_type,expected_keys", [
    (OrderConfig, ["symbol", "quantity"]),
    (RiskConfig, ["max_position", "max_loss"]),
    # ... all 61 classes
])
def test_dataclass_to_dict(class_type, expected_keys):
    assert_dataclass_to_dict(class_type(), expected_keys)
```

### Decision Matrix

Use this matrix to decide what to do with each test:

```
                    ┌─────────────────────────────────────────┐
                    │        Is it safety-critical?           │
                    │         (prevents financial loss,       │
                    │          regulatory violations)         │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
               YES: TIER 1                        NO
               NEVER REMOVE                          │
                                                     ▼
                              ┌─────────────────────────────────┐
                              │    Is it testing correctness?   │
                              │    (validates business logic)   │
                              └─────────────────┬───────────────┘
                                                │
                              ┌─────────────────┴───────────────┐
                              ▼                                 ▼
                         YES: TIER 2                         NO: TIER 3
                         CONSOLIDATE                         REMOVE/CONSOLIDATE
                         if duplicate                        aggressively
```

### Top 10 Best Tests Pattern

Based on analysis of 4,535+ tests, these patterns are most effective:

| Rank | Pattern | Description | Implementation |
|------|---------|-------------|----------------|
| 1 | **Safety invariant** | Tests critical safety properties | `SafetyTestCase.assert_safety_invariant()` |
| 2 | **Boundary condition** | Tests at exact limits | `test_position_at_exactly_max_limit()` |
| 3 | **Error recovery** | Tests system resilience | `FaultInjector` + recovery assertion |
| 4 | **State machine** | Tests valid transitions | `OrderStateMachine.can_transition_to()` |
| 5 | **Concurrent access** | Tests thread safety | `ThreadPoolExecutor` + lock assertions |
| 6 | **Integration flow** | Tests end-to-end | Multi-component scenario |
| 7 | **Property-based** | Discovers edge cases | `@given(valid_order())` |
| 8 | **Regression** | Prevents bug recurrence | `@regression_test("BUG-123")` |
| 9 | **Chaos/fault** | Tests robustness | `FaultInjector.should_fail()` |
| 10 | **Cross-module** | Tests integration points | `test_circuit_breaker_trip_logged()` |

### Recommended Test Infrastructure

```
tests/
├── conftest.py           # Core fixtures, SafetyTestCase, FaultInjector
├── builders.py           # Fluent test data builders
├── strategies.py         # Hypothesis property-based strategies
├── mocks/
│   ├── __init__.py       # Mock registry
│   └── quantconnect.py   # Consolidated QC mocks
├── state_machines/       # State transition tests
├── performance/          # Performance regression tracking
├── snapshots/            # Deterministic output comparison
├── regression/           # Bug regression tests
├── compliance/           # Regulatory compliance tests
└── analysis/             # Duplicate detection tools
```

### Final Recommendations

#### Immediate Actions
1. **Run duplicate finder** before adding new tests
2. **Use builders** instead of manual test data setup
3. **Import mocks** from centralized locations
4. **Mark safety tests** with `SafetyTestCase` or `@pytest.mark.regression`

#### Weekly Maintenance
1. Check for new duplicate patterns
2. Review skipped tests - fix or remove
3. Update performance baselines
4. Run mutation testing on critical paths

#### Monthly Review
1. Analyze test coverage by category
2. Identify tests that never fail (may be ineffective)
3. Review compliance test completeness
4. Update safety gap analysis

### Metrics to Track

```
Weekly Metrics:
├── Test count (should stabilize, not grow unbounded)
├── Skip rate (target: <1%)
├── Safety test count (should never decrease)
├── Duplicate pattern count (should decrease)
└── Coverage (target: >80% on critical paths)

Monthly Metrics:
├── Mutation score (target: >80% on TIER 1)
├── Time to run full suite (target: <10 min)
├── Flaky test count (target: 0)
└── New regression tests added
```
