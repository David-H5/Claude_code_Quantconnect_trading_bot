# ADR-0003: LLM Ensemble for Sentiment Analysis

**Status**: Accepted
**Date**: 2025-12-01 (Retroactive)
**Decision Makers**: Project Owner

## Context

The project requires sentiment analysis for:

- News headline interpretation
- Market sentiment scoring
- Trading signal generation
- Risk assessment

Single LLM providers have limitations (hallucinations, biases, inconsistency). Need a robust approach that mitigates these risks.

## Decision

Use a weighted ensemble of multiple LLM providers:

1. **FinBERT**: Domain-specific financial sentiment (local)
2. **GPT-4o**: General reasoning and context understanding (OpenAI)
3. **Claude**: Alternative perspective and validation (Anthropic)

Combine outputs using weighted voting with confidence thresholds.

## Consequences

### Positive

- Reduces single-provider failure risk
- Mitigates hallucination through consensus
- Leverages domain expertise (FinBERT) + general reasoning (GPT/Claude)
- Confidence scores improve decision quality
- Graceful degradation if one provider fails

### Negative

- Higher API costs (multiple providers)
- Increased latency (sequential or parallel calls)
- Complexity in weight tuning
- Multiple API key management

### Neutral

- Need to maintain multiple provider integrations
- Weights may need periodic recalibration

## Alternatives Considered

### Alternative 1: Single Provider (GPT-4o Only)

**Description**: Use only OpenAI for all sentiment analysis

**Pros**:

- Simpler implementation
- Lower latency
- Single API to manage

**Cons**:

- Single point of failure
- No cross-validation
- Provider-specific biases unchecked

**Why Rejected**: Too risky for trading decisions; no validation of outputs.

### Alternative 2: FinBERT Only

**Description**: Use only FinBERT for sentiment

**Pros**:

- Free (local model)
- Fast
- Domain-specific

**Cons**:

- Limited to sentiment classification
- No contextual reasoning
- Can't handle complex news scenarios

**Why Rejected**: Insufficient for complex market analysis and news interpretation.

### Alternative 3: Custom Fine-tuned Model

**Description**: Fine-tune a custom model on financial data

**Pros**:

- Optimized for our use case
- Full control

**Cons**:

- Significant development effort
- Ongoing maintenance
- Training data requirements

**Why Rejected**: Too much upfront investment; ensemble approach is faster to deploy.

## References

- [llm/ensemble.py](../../llm/ensemble.py) - Ensemble implementation
- [llm/sentiment.py](../../llm/sentiment.py) - Sentiment analyzers
- [llm/providers.py](../../llm/providers.py) - LLM provider integrations

## Notes

**Ensemble Weights** (configurable in `config/settings.json`):

- FinBERT: 0.3 (domain expertise)
- GPT-4o: 0.4 (reasoning capability)
- Claude: 0.3 (alternative perspective)

**Confidence Threshold**: Only act on signals with >70% agreement score.
