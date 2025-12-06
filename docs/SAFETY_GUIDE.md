# Trading Safety Guide

This guide covers safety-first development practices for the trading bot.

## Safety-First Development

**NEVER deploy untested code to live trading.** This project enforces a strict testing pipeline:

1. **Local Tests** - Unit tests must pass
2. **Validation** - Algorithm validator must approve
3. **Backtest** - Cloud backtest must complete successfully
4. **Paper Trading** - Paper trading validation required
5. **Live** - Only after all stages pass

## Pre-Deployment Checklist

Before deploying to paper trading or live:

### Code Quality

- [ ] All unit tests pass (`pytest tests/ -v`)
- [ ] Code coverage > 70% (`pytest --cov=. --cov-fail-under=70`)
- [ ] No linting errors (`ruff check .`)
- [ ] Type checking clean (`mypy --config-file mypy.ini`)
- [ ] No security vulnerabilities (GitLeaks clean)

### Documentation

- [ ] CLAUDE.md updated (if workflow changed)
- [ ] ADR created (if architecture decision made)
- [ ] Code comments adequate for complex logic

### Trading Safety

- [ ] Circuit breaker integration verified
- [ ] Kill switch tested and functional
- [ ] Position limits enforced
- [ ] Risk parameters within bounds

### Validation

- [ ] Algorithm validator passes
- [ ] Backtest metrics meet targets (Sharpe > 1.0, Drawdown < 20%)
- [ ] Paper trading behavior matches backtest

### Approval

- [ ] Human review completed
- [ ] Explicit deployment approval obtained

## Circuit Breaker Safety

Use the TradingCircuitBreaker for autonomous trading safety:

```python
from models.circuit_breaker import TradingCircuitBreaker, create_circuit_breaker

# Create with default limits
breaker = create_circuit_breaker(
    max_daily_loss=0.03,      # 3% daily loss limit
    max_drawdown=0.10,        # 10% max drawdown
    max_consecutive_losses=5,  # Halt after 5 losses
    require_human_reset=True   # Require human to resume
)

# In trading loop
if not breaker.can_trade():
    return  # Trading halted

# Check conditions
breaker.check_daily_loss(portfolio.daily_pnl_pct)
breaker.check_drawdown(current_equity, peak_equity)

# Record trade results
breaker.record_trade_result(is_winner=True)

# Manual halt
breaker.halt_all_trading("Market conditions unusual")

# Reset requires authorization
breaker.reset(authorized_by="trader@example.com")
```

## Pre-Trade Validation

All order submissions must pass pre-trade validation:

```python
from execution.pre_trade_validator import PreTradeValidator, Order, create_validator

# Create validator with circuit breaker integration
validator = create_validator(
    circuit_breaker=breaker,
    max_position_pct=0.25,
    max_daily_loss_pct=0.03,
)

# Validate before submitting order
order = Order(symbol="SPY", quantity=100, side="buy", order_type="limit")
result = validator.validate(order)

if not result.approved:
    for check in result.failed_checks:
        print(f"Failed: {check.name} - {check.message}")
    return  # Don't submit order

# Safe to submit
submit_order(order)
```

### Validation Checks

| Check | Description |
|-------|-------------|
| Circuit Breaker | Trading not halted |
| Position Limit | Within 25% max per position |
| Daily Loss | Within 3% daily loss limit |
| Concentration | Total exposure within limit |
| Order Value | Within min/max range |
| Data Freshness | Price data not stale |
| Duplicate Order | Not a duplicate within 1s |
| Liquidity | Sufficient volume |

## Common Mistakes to Avoid

1. **Look-ahead bias**: Don't use future data in decisions
2. **Missing warmup**: Always warm up indicators
3. **Unchecked data access**: Verify data exists before using
4. **Over-optimization**: Avoid curve-fitting to historical data
5. **Ignoring costs**: Account for commissions and slippage
6. **No risk management**: Always include position limits and stops
7. **Skipping tests**: Never deploy without passing all test stages

## Backup System

Backups are automatically created:

- Before any code change (via pre-commit)
- Before paper trading deployment
- Can be restored if issues occur

```python
from scripts.backup_manager import BackupManager

manager = BackupManager()
manager.list_backups()  # View all backups
manager.restore_file(Path("algorithms/my_algo.py"), backup_index=0)  # Restore latest
```

### Before Making Changes

Always create a backup before modifying algorithm files:

```python
from scripts.backup_manager import create_pre_change_backup
create_pre_change_backup("algorithms/my_algo.py", "adding new indicator")
```

## Disaster Recovery

**Key Metrics**:

| Metric | Definition | Target |
|--------|------------|--------|
| RTO (Recovery Time Objective) | Max downtime allowed | <4 hours |
| RPO (Recovery Point Objective) | Max data loss allowed | <1 hour |

**Checklist**:

- [ ] Daily backups of configuration and state
- [ ] Weekly backup restoration test
- [ ] Documented recovery procedures
- [ ] Annual DR simulation drill
- [ ] Off-site backup storage

## Audit Logging (Trading Compliance)

Trading systems require comprehensive audit logging for regulatory compliance:

**Required Log Fields** (for every trading action):

| Field | Description | Example |
|-------|-------------|---------|
| `timestamp` | ISO 8601 with timezone | `2025-12-02T14:30:00Z` |
| `actor` | User or system identifier | `algorithm:hybrid_bot` |
| `action` | What happened | `ORDER_SUBMITTED` |
| `resource` | Affected entity | `SPY_241220C450` |
| `details` | Action-specific data | `{"qty": 10, "price": 4.50}` |
| `outcome` | Result | `SUCCESS` or `FAILED` |

**Compliance Requirements**:

| Regulation | Retention | Notes |
|------------|-----------|-------|
| SOX | 7 years | Financial reporting controls |
| PCI DSS 4.0 | 12 months (3 months accessible) | If handling card data |
| General Best Practice | 12+ months | Minimum for trading systems |

## Root Cause Analysis (RCA)

When production bugs or critical issues occur, follow the RCA process:

### When RCA is Required

- Production incidents affecting trading
- Financial loss due to bugs
- P0/P1 bugs found in testing
- Security vulnerabilities

### 5 Whys Method

Ask "Why?" iteratively to find the root cause:

```text
Problem: Order executed at wrong price
Why #1: Stale price data was used
Why #2: Price update handler wasn't called
Why #3: WebSocket reconnection failed silently
Why #4: No error logging for reconnection failures
Why #5: Reconnection logic added without tests
Root Cause: Missing test coverage for error paths
Fix: Add reconnection tests and error logging
```

### Post-RCA Actions

1. **Create regression test** in `tests/regression/`
2. **Update documentation** (CLAUDE.md, ADRs)
3. **Add to incident log** (`docs/incidents/README.md`)
4. **Review in retrospective**
