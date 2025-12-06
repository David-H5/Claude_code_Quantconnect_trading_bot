# ADR-0005: Circuit Breaker Safety Pattern

**Status**: Accepted
**Date**: 2025-12-01 (Retroactive)
**Decision Makers**: Project Owner

## Context

Autonomous trading systems can fail catastrophically if not properly constrained. Need safety mechanisms that:

- Halt trading during adverse conditions
- Prevent cascading losses
- Require human intervention to resume after major events
- Log all safety triggers for analysis

LEAN engine does not provide a built-in circuit breaker; must implement custom solution.

## Decision

Implement a multi-trigger circuit breaker pattern with the following halt conditions:

1. **Daily Loss Limit**: 3% daily loss triggers halt
2. **Max Drawdown**: 10% drawdown from peak triggers halt
3. **Consecutive Losses**: 5 consecutive losing trades triggers halt
4. **Market Anomaly**: Unusual market conditions (detected via volatility)
5. **API Failure**: Broker API issues trigger defensive mode
6. **Manual Trigger**: Human can halt at any time

All triggers require human authorization to reset.

## Consequences

### Positive

- Limits maximum daily loss
- Prevents emotional/algorithmic spiral
- Creates natural pause points for analysis
- Audit trail of all triggers
- Human in the loop for resumption

### Negative

- May halt during recoverable situations
- Requires monitoring for resets
- Could miss rebound opportunities
- Human dependency for reset

### Neutral

- State must persist across restarts
- Triggers logged to Object Store

## Alternatives Considered

### Alternative 1: Simple Stop-Loss Only

**Description**: Only use position-level stop-losses

**Pros**:

- Simple implementation
- Per-position risk control

**Cons**:

- No portfolio-level protection
- No pattern detection (consecutive losses)
- No market condition awareness

**Why Rejected**: Insufficient for portfolio-level risk management.

### Alternative 2: External Risk Service

**Description**: Use third-party risk management service

**Pros**:

- Professional risk monitoring
- Real-time alerts

**Cons**:

- Additional cost
- Latency
- External dependency

**Why Rejected**: Adds complexity and latency; custom implementation provides better integration.

## References

- [models/circuit_breaker.py](../../models/circuit_breaker.py) - Implementation
- [tests/test_circuit_breaker.py](../../tests/test_circuit_breaker.py) - Test cases
- [config/settings.json](../../config/settings.json) - Threshold configuration

## Notes

**Default Thresholds**:

```json
{
  "circuit_breaker": {
    "max_daily_loss_pct": 0.03,
    "max_drawdown_pct": 0.10,
    "max_consecutive_losses": 5,
    "require_human_reset": true
  }
}
```

**State Persistence**: Circuit breaker state is saved to QuantConnect Object Store to survive algorithm restarts.

**Kill Switch**: `breaker.halt_all_trading(reason)` can be called from UI or algorithm code for immediate stop.
