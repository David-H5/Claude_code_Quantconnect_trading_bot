# ADR-0004: Hybrid Manual + Autonomous Architecture

**Status**: Accepted
**Date**: 2025-12-01 (Retroactive)
**Decision Makers**: Project Owner

## Context

The trading bot needs to balance:

- Autonomous execution for speed and consistency
- Human oversight for safety and control
- Manual intervention capability for unusual situations
- Graduated autonomy based on confidence levels

Fully autonomous trading is risky; fully manual defeats automation benefits.

## Decision

Implement a hybrid architecture with three operating modes:

1. **Autonomous Mode**: Bot executes without human approval (high-confidence signals)
2. **Supervised Mode**: Bot recommends, human approves (medium-confidence signals)
3. **Manual Mode**: Human initiates, bot assists (low-confidence or unusual situations)

Mode selection based on confidence thresholds and circuit breaker state.

## Consequences

### Positive

- Safety through human oversight when needed
- Speed through automation when appropriate
- Graduated trust model
- Clear audit trail of decisions
- Manual override always available

### Negative

- More complex state management
- UI required for supervised mode
- Latency in supervised mode decisions
- Training needed for operators

### Neutral

- Mode transitions need clear logging
- Both autonomous and manual paths must be tested

## Alternatives Considered

### Alternative 1: Fully Autonomous

**Description**: Bot makes all decisions without human intervention

**Pros**:

- Fastest execution
- No human bottleneck
- Consistent behavior

**Cons**:

- High risk during anomalies
- No safety net
- Regulatory concerns

**Why Rejected**: Too risky for options trading; one bad decision can cause significant losses.

### Alternative 2: Fully Manual with Bot Suggestions

**Description**: All decisions require human approval

**Pros**:

- Maximum safety
- Human judgment for every trade

**Cons**:

- Slow execution
- Missed opportunities
- Human fatigue
- Defeats purpose of automation

**Why Rejected**: Too slow; misses time-sensitive opportunities; operator fatigue.

## References

- [docs/architecture/HYBRID_ARCHITECTURE.md](../architecture/HYBRID_ARCHITECTURE.md)
- [ui/dashboard.py](../../ui/dashboard.py) - UI for supervised mode
- [models/circuit_breaker.py](../../models/circuit_breaker.py) - Safety triggers

## Notes

**Mode Thresholds** (configurable):

- Autonomous: Confidence > 85%, Circuit breaker OK
- Supervised: Confidence 60-85%
- Manual: Confidence < 60% or circuit breaker triggered

**Circuit Breaker Integration**: Any circuit breaker trigger forces supervised or manual mode regardless of confidence.
