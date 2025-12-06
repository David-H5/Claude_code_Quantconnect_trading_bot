# Trading Safety Review

Run a specialized 3-agent parallel review focused on trading safety and correctness.

## Arguments
- `$ARGUMENTS`: File paths or algorithm name to review

## Instructions

Execute these 3 agents IN PARALLEL (all in a single response) using the Task tool:

**Agent 1: Risk Management Review (sonnet - thorough)**
```
Task(
  subagent_type="Explore",
  model="sonnet",
  description="Trading risk analysis",
  prompt="Review trading code for risk management: $ARGUMENTS

CRITICAL CHECKS:
1. Circuit breaker integration
   - Is TradingCircuitBreaker imported and used?
   - Are halt conditions checked before orders?

2. Position limits
   - Max position size enforced? (should be <= 25%)
   - Max daily loss checked? (should be <= 3%)
   - Max drawdown protected? (should be <= 10%)

3. Risk per trade
   - Is risk per trade limited? (should be <= 2%)
   - Stop losses implemented?

4. Order validation
   - PreTradeValidator used before orders?
   - Order sizes validated?

FLAG ANY CODE THAT:
- Bypasses safety checks
- Has hardcoded large position sizes
- Missing circuit breaker checks
- No stop loss implementation

Reference models/circuit_breaker.py and execution/pre_trade_validator.py"
)
```

**Agent 2: Order Execution Review (haiku - fast)**
```
Task(
  subagent_type="Explore",
  model="haiku",
  description="Order execution review",
  prompt="Review order execution logic: $ARGUMENTS

CHECK:
1. Order validation before submission
   - Symbol validation
   - Quantity checks
   - Price reasonableness

2. Fill handling
   - Partial fill handling
   - Fill price recording
   - Position update after fill

3. Cancel/replace logic
   - Proper order state tracking
   - Cancel confirmation handling
   - Replace order sequencing

4. Duplicate prevention
   - Order deduplication
   - Rate limiting

Reference execution/smart_execution.py patterns."
)
```

**Agent 3: Look-Ahead Bias Check (haiku - fast)**
```
Task(
  subagent_type="Explore",
  model="haiku",
  description="Backtesting integrity check",
  prompt="Check for backtesting issues: $ARGUMENTS

LOOK-AHEAD BIAS:
- Using future data in decisions?
- Checking data.ContainsKey() before access?
- Proper IsWarmingUp checks?

SURVIVORSHIP BIAS:
- Only using currently listed stocks?
- Handling delistings?

UNREALISTIC ASSUMPTIONS:
- Assuming 100% fills?
- Ignoring slippage?
- Missing transaction costs?

DATA SNOOPING:
- Parameters fit to historical data?
- Walk-forward validation used?

Reference algorithms/options_trading_bot.py for correct patterns."
)
```

## After All Agents Complete

Compile a safety assessment:

```markdown
## Trading Safety Review: $ARGUMENTS

### Risk Status: [PASS/FAIL/NEEDS REVIEW]

### Risk Management
[Agent 1 results]

### Execution Safety
[Agent 2 results]

### Backtest Integrity
[Agent 3 results]

### Action Items
- [ ] [Critical items first]
- [ ] [High priority items]
- [ ] [Medium priority items]

### Safety Checklist
- [ ] Circuit breaker integrated
- [ ] Position limits enforced
- [ ] Risk per trade limited
- [ ] Order validation present
- [ ] No look-ahead bias
- [ ] Transaction costs included
```
