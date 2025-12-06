# ADR-0001: Use QuantConnect LEAN Platform

**Status**: Accepted
**Date**: 2025-12-01 (Retroactive)
**Decision Makers**: Project Owner

## Context

The project needs a reliable algorithmic trading platform that supports:

- Options trading with full Greeks access
- Python development
- Cloud backtesting
- Live trading deployment
- Integration with US brokerages (specifically Charles Schwab)

## Decision

Use QuantConnect's LEAN engine as the primary trading platform.

## Consequences

### Positive

- Full Python API support with snake_case conventions
- Comprehensive options data including Greeks (IV-based since PR #6720)
- Cloud backtesting with various compute node options
- Built-in risk management primitives
- Active development and community support
- Paper trading environment for validation
- 37+ OptionStrategies factory methods for multi-leg positions

### Negative

- Vendor lock-in to QuantConnect ecosystem
- Monthly cost for compute nodes (~$92/month for recommended setup)
- Learning curve for LEAN-specific patterns
- Limited to supported brokerages

### Neutral

- Code must follow QuantConnect conventions (PascalCase for some properties)
- Research notebooks run in QuantConnect environment

## Alternatives Considered

### Alternative 1: Zipline + Custom Infrastructure

**Description**: Use open-source Zipline with custom deployment

**Pros**:

- Fully open source
- No monthly platform fees

**Cons**:

- No built-in live trading
- Limited options support
- No cloud backtesting
- Significant infrastructure work required

**Why Rejected**: Too much custom infrastructure needed; options support inadequate.

### Alternative 2: Interactive Brokers API Direct

**Description**: Build directly on IBKR API

**Pros**:

- Direct broker access
- Full control

**Cons**:

- No backtesting engine
- Complex API
- All infrastructure must be built

**Why Rejected**: Would require building entire backtesting infrastructure from scratch.

## References

- [QuantConnect Documentation](https://www.quantconnect.com/docs)
- [LEAN Engine GitHub](https://github.com/QuantConnect/Lean)
- [LEAN PR #6720 - IV-based Greeks](https://github.com/QuantConnect/Lean/pull/6720)

## Notes

This decision was made at project inception. The platform has proven reliable and the Greeks improvements in 2024 (PR #6720) validated the choice.
