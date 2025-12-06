# Contributing Guide

Thank you for contributing to this QuantConnect trading bot project!

**IMPORTANT**: This is a trading system that handles real money. All contributions must follow our safety-first development practices.

## Required Reading

Before contributing, read these documents:

- [Development Best Practices](docs/development/BEST_PRACTICES.md) - Safety, risk management, backtesting
- [Coding Standards](docs/development/CODING_STANDARDS.md) - Style guide, type hints, documentation
- [CLAUDE.md](CLAUDE.md) - Project overview and Claude Code instructions

---

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/my-new-feature
# or
git checkout -b bugfix/fix-issue-123
```

### 2. Make Changes

- Follow coding standards (Black, isort, type hints)
- Add tests for new functionality
- Update documentation if needed

### 3. Run Pre-Commit Checks

```bash
pre-commit run --all-files
```

### 4. Run Tests

```bash
pytest tests/ -v --cov=algorithms --cov=models
```

### 5. Validate Algorithm (if changed)

```bash
python scripts/algorithm_validator.py algorithms/your_algorithm.py
```

### 6. Commit and Push

```bash
git add .
git commit -m "Add feature: description"
git push origin feature/my-new-feature
```

### 7. Create Pull Request

- Use the PR template
- Link related issues
- Wait for CI to pass

---

## Safety-First Checklist

### Before ANY Algorithm PR

**Risk Management:**

- [ ] Circuit breaker implemented and tested
- [ ] Risk manager integrated with position limits
- [ ] Stop losses on all positions
- [ ] Daily loss limit configured
- [ ] Maximum drawdown limit configured

**Code Safety:**

- [ ] No hardcoded credentials or API keys
- [ ] All external input validated
- [ ] Proper exception handling with logging
- [ ] No look-ahead bias in backtesting

**Testing:**

- [ ] All tests pass (70%+ coverage)
- [ ] Algorithm validator passes (no errors)
- [ ] Backtest results reviewed

**Documentation:**

- [ ] Docstrings on all public functions
- [ ] Type hints on all functions
- [ ] Strategy documented in code comments

### Before Live Deployment

- [ ] Paper trading for 2+ weeks with no errors
- [ ] Fill rates within 10% of backtest
- [ ] Risk limits trigger correctly
- [ ] Monitoring and alerts configured
- [ ] Human review and approval

---

## Coding Standards

### Python Style

- **Formatter**: Black with line length 100
- **Imports**: isort with Black profile
- **Type hints**: Required on all functions
- **Docstrings**: Google style
- **Linting**: Flake8 with max complexity 15

### Example Function

```python
def calculate_position_size(
    portfolio_value: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
) -> float:
    """Calculate position size based on risk management rules.

    Args:
        portfolio_value: Current portfolio value in dollars.
        risk_per_trade: Maximum risk amount in dollars.
        entry_price: Planned entry price per share.
        stop_loss_price: Stop loss price per share.

    Returns:
        Position size in number of shares.

    Raises:
        ValueError: If inputs are invalid.
    """
    if portfolio_value <= 0:
        raise ValueError("Portfolio value must be positive")
    # ... implementation
```

### Algorithm Structure

```python
from AlgorithmImports import *

class MyAlgorithm(QCAlgorithm):
    """
    Brief description of algorithm strategy.

    Risk Management:
        - Max position: 25%
        - Daily loss limit: 3%
        - Stop loss: Based on ATR

    Attributes:
        symbol: The trading symbol
        risk_manager: RiskManager instance
        circuit_breaker: TradingCircuitBreaker instance
    """

    def Initialize(self) -> None:
        """Initialize algorithm parameters."""
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)

        # ALWAYS initialize risk management
        self.risk_manager = RiskManager(...)
        self.circuit_breaker = TradingCircuitBreaker(...)

        # Set warmup period for indicators
        self.SetWarmUp(20)

    def OnData(self, data: Slice) -> None:
        """Process incoming data."""
        # ALWAYS check warmup
        if self.IsWarmingUp:
            return

        # ALWAYS check data exists
        if not data.ContainsKey(self.symbol):
            return

        # ALWAYS check risk limits
        if not self.circuit_breaker.can_trade():
            return

        # Trading logic here
```

---

## Testing Requirements

### Minimum Coverage

- **Overall**: 70%
- **Risk modules**: 90% (circuit_breaker, risk_manager)
- **Critical paths**: 100%

### Test Types

```python
import pytest

@pytest.mark.unit
def test_position_sizing():
    """Unit test for fast, isolated functionality."""
    pass

@pytest.mark.integration
def test_full_trade_cycle():
    """Integration test for multi-component flows."""
    pass

@pytest.mark.slow
def test_backtest_validation():
    """Slow test, can skip in CI with -m 'not slow'."""
    pass
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=algorithms --cov=models --cov-report=html

# Skip slow tests
pytest tests/ -v -m "not slow"

# Specific file
pytest tests/test_circuit_breaker.py -v
```

---

## Commit Message Format

Use clear, imperative messages:

```
Add momentum-based entry signals

- Implement RSI crossover detection
- Add position sizing based on volatility
- Include circuit breaker integration
- Add unit tests for signal generation
```

For safety-related changes:

```
SAFETY: Add daily loss circuit breaker

- Implement 3% daily loss limit
- Add automatic position liquidation on breach
- Require human reset after trip
- Add comprehensive logging
```

---

## Pull Request Process

### PR Title Format

- `feat: Add new feature`
- `fix: Fix bug in X`
- `safety: Add risk management for Y`
- `docs: Update documentation`
- `test: Add tests for Z`

### PR Description

Use this template:

```markdown
## Summary
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Safety improvement
- [ ] Documentation

## Safety Checklist
- [ ] Risk management implemented
- [ ] Circuit breaker integrated
- [ ] Tests pass (70%+ coverage)
- [ ] Algorithm validator passes
- [ ] No hardcoded secrets

## Testing
Describe how you tested these changes.

## Screenshots/Backtest Results
If applicable, add screenshots or backtest metrics.
```

---

## Code Review Guidelines

Reviewers should check:

1. **Safety First**
   - Risk management present?
   - Circuit breaker integrated?
   - Stop losses implemented?

2. **Code Quality**
   - Type hints present?
   - Docstrings complete?
   - Proper error handling?

3. **Testing**
   - Tests cover new code?
   - Edge cases tested?
   - Coverage acceptable?

4. **Security**
   - No hardcoded secrets?
   - External input validated?
   - Sensitive files protected?

---

## Getting Help

- Check [CLAUDE.md](CLAUDE.md) for project overview
- Read [Best Practices](docs/development/BEST_PRACTICES.md)
- Search existing issues
- Open a discussion for questions
- Tag maintainers in PRs if needed

---

## Questions?

Feel free to open an issue for any questions or concerns!
