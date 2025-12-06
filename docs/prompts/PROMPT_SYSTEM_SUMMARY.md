# Prompt Template System - Implementation Summary

## Overview

This document summarizes the implementation of the comprehensive prompt template system for the multi-agent trading system. This system provides version control, performance tracking, A/B testing, and easy iteration on prompt templates.

**Date**: 2025-12-01
**Status**: ✅ Complete
**Related**: [ENHANCEMENT_PLAN_REFINED.md](architecture/ENHANCEMENT_PLAN_REFINED.md)

---

## What Was Implemented

### 1. Core Registry System
**File**: [llm/prompts/prompt_registry.py](../llm/prompts/prompt_registry.py)

A comprehensive version control system for managing prompt templates:

- **PromptVersion**: Dataclass storing prompt template, model config, metadata, and metrics
- **PromptMetrics**: Performance tracking (usage, success rate, response time, confidence, accuracy)
- **PromptRegistry**: Central registry managing all prompts with version control
- **Persistence**: Automatic saving/loading to JSON file
- **A/B Testing**: Compare two versions side-by-side with statistical metrics
- **Best Version Selection**: Automatically identify best-performing prompts

**Key Features**:
- Multiple versions per agent role
- Active version tracking (which version is currently in use)
- Performance metrics tracking
- Version comparison
- Easy rollback to previous versions
- Global registry pattern for easy access

### 2. Agent Prompt Templates

Created comprehensive prompt templates for **9 agent roles**:

#### Supervisor Agents
**File**: [llm/prompts/supervisor_prompts.py](../llm/prompts/supervisor_prompts.py)
- **SupervisorAgent** (Claude Opus 4)
  - v1.0: Initial orchestrator
  - v1.1: Enhanced with consensus scoring and market regime awareness

#### Analyst Agents
**File**: [llm/prompts/analyst_prompts.py](../llm/prompts/analyst_prompts.py)
- **TechnicalAnalyst** (Claude Sonnet 4)
  - Analyzes VWAP, RSI, MACD, CCI, Bollinger, OBV, Ichimoku
  - Provides trend direction and support/resistance levels
- **SentimentAnalyst** (Claude Sonnet 4 + FinBERT)
  - Integrates FinBERT sentiment scores
  - Analyzes news, social media, analyst ratings, options flow
  - Identifies contrarian opportunities

#### Trader Agents
**File**: [llm/prompts/trader_prompts.py](../llm/prompts/trader_prompts.py)
- **ConservativeTrader** (Claude Opus 4, temp 0.3)
  - Focus: Capital preservation, high win rate (>65%)
  - Strategies: Iron condors, credit spreads, butterflies
  - Position sizing: Max 15%, max 5% risk
- **ModerateTrader** (Claude Opus 4, temp 0.5)
  - Focus: Balance risk/reward
  - Strategies: Debit spreads, credit spreads, diagonals, calendars
  - Position sizing: Max 25%, max 10% risk
- **AggressiveTrader** (Claude Opus 4, temp 0.7)
  - Focus: High-reward opportunities, asymmetric risk/reward
  - Strategies: Debit spreads, butterflies, ratio spreads
  - Position sizing: Max 30%, max 15% risk

#### Risk Manager Agents
**File**: [llm/prompts/risk_prompts.py](../llm/prompts/risk_prompts.py)
- **PositionRiskManager** (Claude Haiku, temp 0.1)
  - Approves/rejects individual trades
  - Enforces: Position size (25%), risk (5%), position count (10), win probability (40%)
  - Cannot be overruled by Supervisor
- **PortfolioRiskManager** (Claude Haiku, temp 0.1)
  - Monitors portfolio-level risk
  - Enforces: Daily loss (3%), drawdown (10%), delta exposure (±30%), sector concentration (40%)
- **CircuitBreakerManager** (Claude Haiku, temp 0.05)
  - Emergency trading halt system
  - Triggers: Daily loss (3%), drawdown (10%), consecutive losses (5), VIX >60, flash crash, system errors

### 3. Package Structure
**File**: [llm/prompts/__init__.py](../llm/prompts/__init__.py)

Convenient package interface with helper functions:
- `get_prompt(role, version)`: Retrieve a prompt template
- `get_registry()`: Access the global registry
- `list_all_prompts()`: List all registered prompts
- `get_active_prompts()`: Get all currently active prompts
- `print_prompt_summary()`: Display summary of all prompts

### 4. Demonstration Script
**File**: [scripts/demo_prompt_system.py](../scripts/demo_prompt_system.py)

Comprehensive demo showing:
1. Basic prompt retrieval
2. Listing all versions
3. Recording usage metrics
4. Comparing versions (A/B testing)
5. Finding best performing version
6. Switching active versions
7. Viewing all registered roles

---

## Prompt Template Design

Each prompt template follows a consistent structure:

```
ROLE: Clear description of agent's purpose

YOUR PHILOSOPHY: (for trader agents)
- Core beliefs
- Risk tolerance
- Strategy preferences

YOUR PROCESS:
1. Step-by-step analysis/decision process
2. ...

OUTPUT FORMAT (JSON):
{
    "field": "value",
    ...
}

DECISION CRITERIA:
- When to recommend BUY/SELL/HOLD
- Confidence thresholds

CONSTRAINTS:
- Hard limits
- What agent must/must not do

EXAMPLES:
Example 1 - ...
Example 2 - ...
```

**Benefits of this structure**:
- Clear role definition
- Explicit decision-making process
- Structured JSON output (easy to parse)
- Concrete examples for few-shot learning
- Hard constraints to prevent errors

---

## Usage Examples

### Basic Usage

```python
from llm.prompts import get_prompt, AgentRole

# Get active supervisor prompt
prompt = get_prompt(AgentRole.SUPERVISOR)

print(f"Model: {prompt.model}")
print(f"Temperature: {prompt.temperature}")
print(f"Max Tokens: {prompt.max_tokens}")
print(f"Template: {prompt.template}")
```

### Recording Metrics

```python
from llm.prompts import get_registry

registry = get_registry()

# After using a prompt
registry.record_usage(
    role=AgentRole.TECHNICAL_ANALYST,
    version="v1.0",
    success=True,
    response_time_ms=1250.5,
    confidence=0.85
)
```

### A/B Testing

```python
# Compare two versions
comparison = registry.compare_versions(
    role=AgentRole.SUPERVISOR,
    version1="v1.0",
    version2="v1.1"
)

print(f"v1.0 confidence: {comparison['metrics_comparison']['avg_confidence']['v1']}")
print(f"v1.1 confidence: {comparison['metrics_comparison']['avg_confidence']['v2']}")
print(f"Winner: {comparison['metrics_comparison']['avg_confidence']['winner']}")
```

### Finding Best Version

```python
# Get best performing version
best = registry.get_best_version(
    AgentRole.SUPERVISOR,
    metric="avg_confidence"
)

print(f"Best version: {best.version}")
print(f"Avg confidence: {best.metrics.avg_confidence:.3f}")
```

### Switching Versions

```python
# Activate a different version
registry.set_active(AgentRole.SUPERVISOR, version="v1.1")

# Now get_prompt will return v1.1
prompt = get_prompt(AgentRole.SUPERVISOR)  # Returns v1.1
```

---

## Prompt Versions

### Supervisor Prompts

| Version | Description | Changes |
|---------|-------------|---------|
| v1.0 | Initial supervisor prompt | Base orchestrator with 70% confidence threshold |
| v1.1 | Enhanced supervisor | Added consensus_score, market regime awareness, 75% confidence threshold, max position 0.3 |

### All Other Agents

Currently at **v1.0** (initial versions):
- TechnicalAnalyst v1.0
- SentimentAnalyst v1.0
- ConservativeTrader v1.0
- ModerateTrader v1.0
- AggressiveTrader v1.0
- PositionRiskManager v1.0
- PortfolioRiskManager v1.0
- CircuitBreakerManager v1.0

---

## Performance Metrics Tracked

For each prompt version, the system tracks:

| Metric | Description |
|--------|-------------|
| `total_uses` | Number of times this prompt was used |
| `successful_responses` | Number of successful completions |
| `failed_responses` | Number of failures |
| `avg_response_time_ms` | Average LLM response time |
| `avg_confidence` | Average confidence score from agent |
| `accuracy` | Accuracy vs ground truth (when available) |
| `user_feedback_score` | Manual rating (1-5) if provided |

---

## File Structure

```
llm/prompts/
├── __init__.py                 # Package interface
├── prompt_registry.py          # Core registry system
├── supervisor_prompts.py       # Supervisor agent templates
├── analyst_prompts.py          # Analyst agent templates
├── trader_prompts.py           # Trader agent templates
├── risk_prompts.py             # Risk manager templates
└── registry.json               # Persisted registry data (auto-generated)

scripts/
└── demo_prompt_system.py       # Demonstration and testing
```

---

## Testing

The demonstration script was successfully run with the following results:

```bash
$ PYTHONPATH=/home/dshooter/projects/Claude_code_Quantconnect_trading_bot python3 scripts/demo_prompt_system.py
```

**Results**:
- ✅ All 9 agent roles registered successfully
- ✅ 11 prompt versions created (Supervisor has 2 versions, others have 1)
- ✅ Metrics tracking working (simulated 10 uses each for Supervisor v1.0 and v1.1)
- ✅ Version comparison showing v1.1 has better confidence (0.868 vs 0.840)
- ✅ Active version switching working
- ✅ Registry persisted to JSON file successfully

---

## Integration with Agents

The prompt templates are designed to integrate with the agent base class ([llm/agents/base.py](../llm/agents/base.py)):

```python
from llm.prompts import get_prompt, AgentRole
from llm.agents.base import TradingAgent

class SupervisorAgent(TradingAgent):
    def __init__(self, llm_client):
        # Get active prompt for this role
        prompt_version = get_prompt(AgentRole.SUPERVISOR)

        super().__init__(
            name="Supervisor",
            role=AgentRole.SUPERVISOR,
            system_prompt=prompt_version.template,
            llm_client=llm_client,
        )

        self.prompt_version = prompt_version
        self.registry = get_registry()

    def analyze(self, query: str, context: dict) -> AgentResponse:
        start_time = time.time()

        # Use parent class analyze method
        response = super().analyze(query, context)

        # Record usage metrics
        response_time = (time.time() - start_time) * 1000
        self.registry.record_usage(
            role=AgentRole.SUPERVISOR,
            version=self.prompt_version.version,
            success=response.success,
            response_time_ms=response_time,
            confidence=response.confidence,
        )

        return response
```

---

## Next Steps

To complete the proof of concept, the following work remains:

### 1. Implement Concrete Agent Classes
- [ ] SupervisorAgent (inherits from TradingAgent)
- [ ] TechnicalAnalyst (inherits from TradingAgent)
- [ ] SentimentAnalyst (inherits from TradingAgent, uses FinBERT)
- [ ] ConservativeTrader (inherits from TradingAgent)
- [ ] PositionRiskManager (inherits from TradingAgent)

### 2. Integrate Anthropic API
- [ ] Install anthropic SDK: `pip install anthropic`
- [ ] Create Claude client wrapper
- [ ] Implement LLM calling in base agent
- [ ] Add retry logic and error handling
- [ ] Add rate limiting

### 3. Integrate FinBERT
- [ ] Install transformers: `pip install transformers torch`
- [ ] Load FinBERT model (ProsusAI/finbert)
- [ ] Create sentiment analysis tool
- [ ] Add to SentimentAnalyst

### 4. Create Agent Orchestration
- [ ] Install LangGraph: `pip install langgraph`
- [ ] Define TradingFirmState
- [ ] Create StateGraph with 5 agents
- [ ] Test orchestration flow

### 5. Write Tests
- [ ] Unit tests for each agent
- [ ] Integration tests for orchestration
- [ ] Backtest with agents in QuantConnect

---

## Cost Estimates

Based on the refined plan, estimated monthly costs:

| Agent Role | Model | Est. Usage | Monthly Cost |
|-----------|-------|------------|--------------|
| Supervisor | Claude Opus 4 | ~100K tokens/day | $50-100 |
| Analysts (3) | Claude Sonnet 4 | ~300K tokens/day | $100-200 |
| Traders (1 POC) | Claude Opus 4 | ~50K tokens/day | $25-50 |
| Risk Managers (1 POC) | Claude Haiku | ~200K tokens/day | $20-50 |
| FinBERT | Open-source | Unlimited | $0 |
| **Total** | | | **$195-400/month** |

Note: These are rough estimates. Actual costs will depend on:
- Number of trading signals per day
- Prompt length and response length
- Number of active agents
- Market volatility (more signals = higher costs)

---

## References

- [Enhancement Plan (Refined)](architecture/ENHANCEMENT_PLAN_REFINED.md) - Overall enhancement strategy
- [Agent Base Classes](../llm/agents/base.py) - Foundation for all agents
- [Prompt Registry](../llm/prompts/prompt_registry.py) - Version control system
- [Demo Script](../scripts/demo_prompt_system.py) - Usage examples

---

## Changelog

### 2025-12-01
- ✅ Implemented core registry system (prompt_registry.py)
- ✅ Created 11 prompt versions across 9 agent roles
- ✅ Implemented package structure with helper functions
- ✅ Created demonstration script
- ✅ Tested entire system successfully
- ✅ Documented usage and integration patterns
