# ADR-0002: Charles Schwab as Primary Brokerage

**Status**: Accepted
**Date**: 2025-12-01 (Retroactive)
**Decision Makers**: Project Owner

## Context

The project needs a live trading brokerage that:

- Supports options trading
- Integrates with QuantConnect LEAN
- Has reasonable commission structure
- Supports multi-leg options orders (ComboOrders)
- Is accessible for US-based trading

## Decision

Use Charles Schwab as the primary brokerage for live trading.

## Consequences

### Positive

- Full QuantConnect integration
- ComboOrders supported (ComboMarketOrder, ComboLimitOrder)
- Competitive commission structure
- Established, regulated broker
- Good options liquidity

### Negative

- **CRITICAL**: Only ONE algorithm per account (deploying second stops first)
- OAuth re-authentication required approximately weekly
- ComboLegLimitOrder (individual leg limits) NOT supported
- Must combine all strategies into single algorithm

### Neutral

- Standard settlement times (T+1 for options)
- Regular API maintenance windows

## Alternatives Considered

### Alternative 1: Interactive Brokers

**Description**: Use IBKR as primary brokerage

**Pros**:

- More API features
- Better international support
- Lower commissions for high volume

**Cons**:

- Different API integration required
- More complex account structure
- QuantConnect integration less mature than Schwab

**Why Rejected**: Schwab integration with QuantConnect is more mature and simpler.

### Alternative 2: TD Ameritrade

**Description**: Use TDA (now part of Schwab)

**Pros**:

- Good options support
- Thinkorswim platform

**Cons**:

- Being migrated to Schwab
- API deprecation concerns

**Why Rejected**: TD Ameritrade is being absorbed into Schwab; using Schwab directly is future-proof.

## References

- [Charles Schwab Developer Portal](https://developer.schwab.com/)
- [QuantConnect Schwab Documentation](https://www.quantconnect.com/docs/v2/cloud-platform/live-trading/brokerages/charles-schwab)
- [docs/research/PHASE_2_INTEGRATION_RESEARCH.md](../research/PHASE_2_INTEGRATION_RESEARCH.md)

## Notes

**Critical Architecture Implication**: The one-algorithm-per-account limitation means all trading strategies must be combined into `algorithms/hybrid_options_bot.py`. This is a hard constraint that shapes the entire system architecture.

**ComboOrder Support Confirmed**: As of November 2025 research, ComboLimitOrder is fully supported on Schwab, enabling the two-part spread strategy.
