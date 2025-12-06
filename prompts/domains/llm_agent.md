# LLM Agent Domain Context

You are working on **LLM integration, agents, or sentiment analysis** code.

## Architecture Overview

The project uses a multi-agent ensemble architecture:

- **FinBERT**: Financial sentiment analysis
- **GPT-4o**: OpenAI provider for general analysis
- **Claude**: Anthropic provider for reasoning tasks
- **Ensemble**: Weighted combination of multiple providers

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Base classes | `llm/base.py` | Sentiment, NewsItem |
| Sentiment | `llm/sentiment.py` | FinBERT + Simple analyzers |
| Providers | `llm/providers.py` | OpenAI + Anthropic clients |
| Ensemble | `llm/ensemble.py` | Weighted predictions |
| Agents | `llm/agents/` | Specialized trading agents |
| Guardrails | `llm/agents/llm_guardrails.py` | Safety constraints |

## Safety Guardrails

**All LLM outputs must pass guardrails before trading actions**:

```python
from llm.agents.llm_guardrails import LLMGuardrails

guardrails = LLMGuardrails()
if not guardrails.validate(decision):
    # Reject decision
```

## Multi-Agent Patterns

**Bull/Bear Debate** (for major decisions):

```python
from llm.agents.debate_mechanism import BullBearDebate

debate = BullBearDebate(bull_agent, bear_agent)
result = debate.run_debate(opportunity, analysis)
```

**Max 3 debate rounds** - research shows more rounds reduce performance.

## API Rate Limits

- OpenAI: Respect TPM/RPM limits
- Anthropic: Rate limit handling built-in
- Always implement exponential backoff

## Decision Logging

All agent decisions must be logged:

```python
from llm.decision_logger import DecisionLogger

logger = DecisionLogger(auto_persist=True)
logger.log_decision(agent_name, decision, confidence, context)
```

## Before Committing

- [ ] Guardrails tested
- [ ] Rate limit handling verified
- [ ] Decision logging enabled
- [ ] No API keys in code
