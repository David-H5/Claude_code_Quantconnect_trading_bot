# INT-003: Initial Backtest Validation Plan

**Task ID**: INT-003
**Status**: Ready for Execution
**Created**: December 2, 2025
**Priority**: P1 - High

---

## Phase 0: Research Summary

### Algorithm Status

| Component | Status | Tests |
|-----------|--------|-------|
| `algorithms/hybrid_options_bot.py` | Ready | 19/19 pass |
| Risk Manager | Configured | Integrated |
| Circuit Breaker | Configured | Integrated |
| LLM Sentiment | Configured | 227 tests pass |
| OptionStrategiesExecutor | Configured | Integrated |
| ManualLegsExecutor | Configured | Integrated |
| BotManagedPositions | Configured | Integrated |
| OrderQueueAPI | Configured | 38 tests pass |

### Current Configuration

```python
# From algorithms/hybrid_options_bot.py
SetStartDate(2024, 11, 1)   # November 2024 backtest
SetEndDate(2024, 11, 30)    # 1 month duration
SetCash(100000)             # $100K starting capital

# Brokerage
SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)

# Risk Limits (from config/settings.json)
max_position_size = 25%
max_daily_loss = 3%
max_drawdown = 10%
max_risk_per_trade = 2%

# Universe
primary_symbols = ["SPY", "QQQ", "IWM"]
option_filter = "±20 strikes, 0-180 DTE"
resolution = Resolution.Minute
```

---

## Phase 1: Scope and Success Criteria

### Scope

**Included**:
- Algorithm initialization (all modules)
- Data subscription (SPY, QQQ, IWM options)
- Circuit breaker functionality
- OnData loop processing
- Daily risk review

**Excluded**:
- REST API server (not needed for backtest)
- WebSocket updates (UI feature)
- LLM API calls (no external API in backtest)
- Object Store persistence (optional)

### Success Criteria

| Criteria | Target | Pass Condition |
|----------|--------|----------------|
| Initialization | No errors | All modules instantiate |
| Data Processing | No crashes | OnData runs for 22 trading days |
| Circuit Breaker | Functions | Can halt/allow trading |
| Risk Limits | Enforced | Position limits respected |
| Options Chains | Available | At least 1 chain loaded |
| No Exceptions | 0 unhandled | No error messages |

### Metrics to Collect

| Metric | Target | Acceptable Range |
|--------|--------|------------------|
| Sharpe Ratio | > 0.0 | > -1.0 (conservative) |
| Max Drawdown | < 20% | < 30% |
| Win Rate | > 40% | > 30% |
| Total Trades | > 0 | Any (confirms execution) |
| Final Equity | > $90K | > $80K |

---

## Phase 2: Backtest Checklist

### Pre-Backtest

- [x] Algorithm file exists (`algorithms/hybrid_options_bot.py`)
- [x] Configuration file exists (`config/settings.json`)
- [x] All imports resolve (no ImportError)
- [x] Unit tests pass (560+ tests)
- [x] Structural tests pass (19 tests)

### Backtest Stages

#### Stage 1: Initialization Test (1 day)

```
Start: 2024-11-01
End: 2024-11-01
Duration: 1 trading day
Purpose: Verify all modules initialize
```

**Checklist**:
- [ ] Algorithm starts without crash
- [ ] "HYBRID OPTIONS BOT INITIALIZED SUCCESSFULLY" logged
- [ ] SPY, QQQ, IWM options subscribed
- [ ] Risk manager initialized
- [ ] Circuit breaker initialized

#### Stage 2: Data Processing Test (1 week)

```
Start: 2024-11-01
End: 2024-11-08
Duration: 5 trading days
Purpose: Verify OnData processes data
```

**Checklist**:
- [ ] OnData called multiple times
- [ ] Circuit breaker checks occur
- [ ] Daily risk review runs
- [ ] No unhandled exceptions
- [ ] Resource monitoring works

#### Stage 3: Full Month Test

```
Start: 2024-11-01
End: 2024-11-30
Duration: 22 trading days
Purpose: Verify complete trading cycle
```

**Checklist**:
- [ ] Backtest completes
- [ ] Final equity > $80K
- [ ] Max drawdown < 30%
- [ ] At least 1 position logged
- [ ] Algorithm summary generated

### Post-Backtest

- [ ] Review debug logs for errors
- [ ] Document any issues found
- [ ] Collect performance metrics
- [ ] Update Implementation Tracker

---

## Phase 3: Backtest Configuration

### QuantConnect Cloud Backtest Settings

```json
{
  "algorithm": "algorithms/hybrid_options_bot.py",
  "node": "B8-16",
  "start_date": "2024-11-01",
  "end_date": "2024-11-30",
  "cash": 100000,
  "brokerage": "CharlesSchwab",
  "account_type": "Margin",
  "resolution": "Minute"
}
```

### Deploy Command

```bash
# Analyze algorithm requirements
python scripts/deploy_with_nodes.py algorithms/hybrid_options_bot.py --analyze-only

# Deploy to backtest node
python scripts/deploy_with_nodes.py algorithms/hybrid_options_bot.py --type backtest
```

### Expected Resource Usage

| Resource | Estimate | Limit |
|----------|----------|-------|
| RAM | ~8GB | 16GB (B8-16) |
| CPU | 6-8 cores | 8 cores |
| Option Chains | 3 | 5 max |
| Contracts/Chain | ~100 | 100 max |

---

## Phase 4: Algorithm Readiness Verification

### Static Analysis

```bash
# Run type checking
mypy --config-file mypy.ini algorithms/hybrid_options_bot.py

# Run linting
ruff check algorithms/hybrid_options_bot.py

# Run all tests
pytest tests/ -v --tb=short
```

### Structural Verification

| Check | Expected | Command |
|-------|----------|---------|
| Line count | ~625 lines | `wc -l algorithms/hybrid_options_bot.py` |
| Initialize method | Present | `grep "def Initialize" ...` |
| OnData method | Present | `grep "def OnData" ...` |
| OnOrderEvent method | Present | `grep "def OnOrderEvent" ...` |
| No syntax errors | 0 errors | `python -m py_compile ...` |

### Risk Pattern Check

| Pattern | Expected |
|---------|----------|
| Circuit breaker integration | ✅ Yes |
| Position limit check | ✅ Yes |
| Daily loss limit | ✅ Yes |
| Max drawdown tracking | ✅ Yes |
| Error handling | ✅ Try/except blocks |

---

## Phase 5: Potential Issues

### Known Limitations

1. **No LLM API in Backtest**: LLM providers (OpenAI, Anthropic) not available
   - Mitigation: Sentiment components have mock/fallback mode

2. **No Object Store in Backtest**: Persistence not available locally
   - Mitigation: Already handled with `if object_store_config.get("enabled", False)`

3. **No REST API in Backtest**: UI integration not tested
   - Mitigation: API tested separately (38 tests)

### Expected Warnings

```
⚠️  Config file not found, using defaults
⚠️  Object Store disabled (templates/positions won't persist)
```

These are expected in local backtest mode and do not indicate problems.

---

## Execution Instructions

### For QuantConnect Cloud

1. Login to [QuantConnect.com](https://www.quantconnect.com)
2. Create new project or open existing
3. Upload `algorithms/hybrid_options_bot.py`
4. Upload dependencies:
   - `config/settings.json`
   - `config/__init__.py`
   - `models/` directory
   - `execution/` directory
   - `llm/` directory
   - `api/` directory
   - `utils/` directory
5. Set backtest parameters:
   - Start: November 1, 2024
   - End: November 30, 2024
   - Cash: $100,000
6. Run backtest
7. Review results

### For Local LEAN

```bash
# Install LEAN
pip install lean

# Initialize project
lean init

# Run backtest
lean backtest algorithms/hybrid_options_bot.py
```

---

## Results Documentation Template

```markdown
## Backtest Results - [DATE]

### Configuration
- Start Date: 2024-11-01
- End Date: 2024-11-30
- Starting Cash: $100,000
- Node: B8-16

### Performance Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Final Equity | $X | > $90K | ✅/❌ |
| Total Return | X% | > 0% | ✅/❌ |
| Sharpe Ratio | X | > 0 | ✅/❌ |
| Max Drawdown | X% | < 20% | ✅/❌ |
| Total Trades | X | > 0 | ✅/❌ |

### Issues Found
1. [Issue description]

### Next Steps
1. [Action item]
```

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-02 | Initial validation plan created |

---

**Created By**: Claude Code Agent
**Associated Task**: INT-003
**Review Required**: Before executing on QuantConnect Cloud
