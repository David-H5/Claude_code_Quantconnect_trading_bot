# ADR-0006: Two-Part Spread Entry Strategy

**Status**: Accepted
**Date**: 2025-12-01 (Retroactive)
**Decision Makers**: Project Owner

## Context

Options spreads (butterflies, iron condors) typically require paying net debit or receiving reduced credit due to bid-ask spreads. The goal is to:

- Enter complex positions at net-credit or minimal cost
- Exploit wide bid-ask spreads as opportunities
- Maintain balanced positions (no excess longs)
- Achieve reasonable fill rates

Traditional single-order spread entry often results in poor fills.

## Decision

Implement a two-part spread entry strategy:

**Part 1: Debit Spread Entry**

1. Find underpriced debit spread (wide bid-ask = opportunity)
2. Place limit order at 35% from bid
3. Cancel unfilled orders after 2.5 seconds
4. Random delay (3-15 seconds) before retry

**Part 2: Credit Spread Completion**

1. Find credit spread further OTM
2. Ensure credit >= Part 1 debit cost
3. Place at 65% from bid
4. Complete creates net-credit butterfly/condor

## Consequences

### Positive

- Net-credit or minimal-cost entries achievable
- Wide spreads become opportunities, not obstacles
- Position balance maintained (no excess long exposure)
- Market maker pattern detection avoided (random delays)
- Works with Schwab ComboLimitOrder

### Negative

- Lower fill rate (estimated 25%)
- More complex execution logic
- Requires monitoring and adjustment
- May miss some opportunities

### Neutral

- Two-phase execution takes longer than single order
- Strategy effectiveness varies by market conditions

## Alternatives Considered

### Alternative 1: Single ComboOrder at Mid

**Description**: Enter entire spread at mid-price in one order

**Pros**:

- Simpler execution
- Single order to manage

**Cons**:

- Often poor fills
- Pays full bid-ask spread
- Usually results in net debit

**Why Rejected**: Fill quality is poor; consistently pays more than two-part approach.

### Alternative 2: Leg-by-Leg Entry

**Description**: Enter each leg separately

**Pros**:

- Maximum control per leg

**Cons**:

- Execution risk (partial fills)
- Can leave unbalanced position
- Higher transaction costs

**Why Rejected**: Too risky for multi-leg positions; can leave unhedged exposure.

## References

- [docs/strategies/TWO_PART_SPREAD_STRATEGY.md](../strategies/TWO_PART_SPREAD_STRATEGY.md)
- [execution/two_part_spread.py](../../execution/two_part_spread.py)
- [execution/arbitrage_executor.py](../../execution/arbitrage_executor.py)

## Notes

**Key Parameters** (from live trading observations):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Quick cancel timeout | 2.5 seconds | Orders not filling in 2-3s won't fill |
| Min delay between attempts | 3 seconds | Avoid pattern detection |
| Max delay between attempts | 15 seconds | Balance speed vs detection |
| Starting contract size | 1 contract | Highest fill probability |
| Target fill rate | 25% | Realistic expectation |

**User Observation**: "Orders that don't fill in 2-3 seconds won't fill at all" - this informed the 2.5s quick cancel timeout.
