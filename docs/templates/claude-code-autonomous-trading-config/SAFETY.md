# Trading Safety & Risk Controls

## MANDATORY LIMITS

These limits are enforced by hooks and cannot be bypassed:

| Limit | Value | Enforcement |
|-------|-------|-------------|
| Max position size | 2% of portfolio | PreToolUse hook |
| Max single loss | $500 | Kill switch trigger |
| Max daily loss | $2,000 | Kill switch trigger |
| Max drawdown | 5% | Kill switch trigger |
| Max open positions | 10 | Execution service |
| Max sector exposure | 20% | Risk validator |
| Max options per underlying | 3 contracts | Risk validator |

## Kill Switch Triggers

The kill switch activates automatically and halts ALL trading when:

1. Daily loss exceeds $2,000
2. Portfolio drawdown exceeds 5%
3. Any single position loses > $500
4. API errors exceed 5 in 1 minute
5. Market data staleness > 30 seconds
6. Manual activation via `/kill` command

**Recovery requires**: Manual review, root cause analysis, explicit restart command.

## Paper Trading Requirements

Before ANY strategy goes live:

- [ ] Minimum 100 backtested trades
- [ ] Sharpe ratio > 1.0
- [ ] Maximum drawdown < 10%
- [ ] Win rate documented
- [ ] 2-4 weeks paper trading
- [ ] Consistent positive expectancy
- [ ] Emergency procedures tested
- [ ] Code review completed

## Order Validation Checklist

Every order MUST pass:

1. **Position size**: Within 2% portfolio limit
2. **Sector exposure**: Doesn't exceed 20% in any sector
3. **Liquidity**: Bid-ask spread < 10% for options
4. **Greeks**: Delta exposure within tolerance
5. **Correlation**: Not adding to correlated positions
6. **Time**: Within market hours (or valid for extended)

## Circuit Breaker States

```
CLOSED ──(failure threshold)──► OPEN
   ▲                              │
   │                              │ (timeout)
   │                              ▼
   └────(success)──────── HALF_OPEN
```

- **CLOSED**: Normal operation
- **OPEN**: All requests blocked, alert sent
- **HALF_OPEN**: Limited requests allowed for testing

## Audit Requirements

All trading decisions logged with:
- Timestamp (UTC)
- Signal source and confidence
- Position sizing calculation
- Risk check results
- Order details and fills
- P&L attribution

Logs retained minimum 7 years per regulatory requirements.

## Emergency Contacts

- Kill switch command: `/kill` in Claude Code
- Manual broker halt: Schwab 1-800-435-4000
- System admin escalation: [defined in .env]
