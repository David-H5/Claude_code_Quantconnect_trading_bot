# Evaluation Domain Context

You are working on **testing, evaluation, or metrics** code.

## Test Framework

- **Framework**: pytest
- **Minimum Coverage**: 70%
- **Test Types**: Unit, Integration, E2E

## Test Pyramid Targets

| Type | Target | Purpose |
|------|--------|---------|
| Unit | 50% | Fast, isolated component tests |
| Integration | 30% | Module interaction tests |
| E2E | 20% | Full system validation |

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest --cov=. --cov-fail-under=70

# Specific markers
pytest -m unit        # Unit tests only
pytest -m integration # Integration tests only
pytest -m regression  # Regression tests
```

## Test Patterns

**Test behavior, not implementation**:

```python
# Good: Test behavior
def test_circuit_breaker_halts_on_daily_loss():
    breaker = CircuitBreaker(max_daily_loss=0.03)
    breaker.record_loss(0.035)
    assert not breaker.can_trade()

# Bad: Test implementation
def test_circuit_breaker_internal_state():
    breaker = CircuitBreaker()
    assert breaker._daily_loss_counter == 0  # Don't test internals
```

## Evaluation Frameworks

The project includes 7 evaluation methodologies:

1. **Classic Evaluation**: Traditional ML metrics
2. **Walk-Forward Analysis**: Time-series validation
3. **Agent-as-Judge**: LLM-based evaluation
4. **Monte Carlo**: Stress testing
5. **PSI Drift Detection**: Distribution shift monitoring
6. **TCA Evaluation**: Transaction cost analysis
7. **Long-Horizon**: Extended performance tracking

## Key Files

- `evaluation/` - All evaluation frameworks
- `tests/` - Test suite
- `tests/regression/` - Regression tests for fixed bugs

## Before Committing

- [ ] Tests pass locally
- [ ] Coverage >= 70%
- [ ] No production data in tests
- [ ] Test isolation verified
