# Algorithm Domain Context

You are working on **trading algorithm code** in a QuantConnect/LEAN environment.

## Critical Safety Rules

1. **NEVER deploy untested code to live trading**
2. **Always validate through**: Local Tests → Algorithm Validator → Backtest → Paper Trading → Live
3. **Circuit breaker must be integrated** for any trading logic

## Platform: QuantConnect with Charles Schwab

**Key Constraint**: Charles Schwab allows ONLY ONE algorithm per account.

- All strategies must be combined into a single algorithm
- Deploying a second algorithm stops the first
- OAuth re-authentication required weekly

## Code Patterns

**Python API uses snake_case** (not C# PascalCase):

```python
self.set_start_date(2020, 1, 1)  # NOT SetStartDate
self.add_equity("SPY")           # NOT AddEquity
```

**Framework properties stay PascalCase**:

```python
if data.ContainsKey(self.symbol):   # ContainsKey stays PascalCase
    if self.indicator.IsReady:       # IsReady stays PascalCase
```

## Greeks (LEAN PR #6720)

- Greeks now use implied volatility - **NO warmup required**
- Values match Interactive Brokers and major brokerages
- Access immediately: `contract.Greeks.Delta`, `contract.Greeks.Gamma`

## Multi-Leg Options

Use `ComboLimitOrder()` with net debit/credit pricing:

```python
legs = [
    Leg.Create(call_low, 1),
    Leg.Create(call_mid, -2),
    Leg.Create(call_high, 1),
]
self.ComboLimitOrder(legs, quantity=1, limit_price=net_debit)
```

## Validation Commands

```bash
python scripts/algorithm_validator.py algorithms/your_algo.py
pytest tests/ -v -m unit
```

## Before Committing

- [ ] Algorithm validator passes
- [ ] Circuit breaker integration verified
- [ ] No hardcoded credentials
- [ ] Risk limits enforced
