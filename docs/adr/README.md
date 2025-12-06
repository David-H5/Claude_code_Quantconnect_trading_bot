# Architectural Decision Records (ADRs)

This directory contains Architectural Decision Records for the QuantConnect Trading Bot project.

## What is an ADR?

An Architectural Decision Record (ADR) captures a single architectural decision and its rationale, including trade-offs and consequences. ADRs help future team members understand why certain decisions were made.

## When to Create an ADR

Create an ADR when:

- Adopting a new technology or framework
- Making architecture pattern changes
- Selecting major libraries or tools
- Making breaking changes to APIs
- Adding new integration points
- Changing deployment strategies

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-0001](ADR-0001-use-quantconnect-lean.md) | Use QuantConnect LEAN Platform | Accepted | 2025-12-01 |
| [ADR-0002](ADR-0002-charles-schwab-brokerage.md) | Charles Schwab as Primary Brokerage | Accepted | 2025-12-01 |
| [ADR-0003](ADR-0003-llm-ensemble-approach.md) | LLM Ensemble for Sentiment Analysis | Accepted | 2025-12-01 |
| [ADR-0004](ADR-0004-hybrid-architecture.md) | Hybrid Manual + Autonomous Architecture | Accepted | 2025-12-01 |
| [ADR-0005](ADR-0005-circuit-breaker-pattern.md) | Circuit Breaker Safety Pattern | Accepted | 2025-12-01 |
| [ADR-0006](ADR-0006-two-part-spread-strategy.md) | Two-Part Spread Entry Strategy | Accepted | 2025-12-01 |
| [ADR-0007](ADR-0007-upgrade-loop-workflow.md) | 6-Phase Upgrade Loop Workflow | Accepted | 2025-12-01 |

## ADR Statuses

- **Proposed**: Under discussion
- **Accepted**: Decision approved and in effect
- **Deprecated**: No longer valid but kept for historical reference
- **Superseded**: Replaced by a newer ADR

## Creating a New ADR

1. Copy `template.md` to `ADR-NNNN-short-title.md`
2. Fill in all sections
3. Submit for review via PR
4. Update this README with the new entry

## Resources

- [ADR GitHub Organization](https://adr.github.io/)
- [Michael Nygard's Original Article](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [MADR Template](https://adr.github.io/madr/)
