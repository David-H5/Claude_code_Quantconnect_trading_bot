---
title: "Prompt Enhancements Research"
topic: prompts
related_upgrades: []
related_docs: []
tags: [prompts]
created: 2025-12-01
updated: 2025-12-02
---

# Prompt Template Enhancements - Research-Driven Iteration

## Overview

This document summarizes the v2.0 prompt enhancements based on research into real-world financial AI agents, trading bots, and LLM trading frameworks from 2024-2025.

**Date**: 2025-12-01
**Status**: ✅ Complete
**Research Sources**: 10+ academic papers, trading frameworks, and industry examples

---

## Research Summary

### Key Findings

**1. TradingAgents Framework** ([arxiv.org/abs/2412.20138](https://arxiv.org/abs/2412.20138))
- Multi-agent system with specialized roles (fundamental, sentiment, technical analysts)
- **Bull and Bear researchers** evaluating market conditions through debate
- **Risk management team** overseeing exposure and implementing mitigation strategies
- All agents use **ReAct prompting framework** (already in our base.py)
- **Multi-agent debate** produces better decisions than single-agent approaches

**2. Market Condition Assessment** ([TradingAgents](https://tradingagents-ai.github.io/))
- Risk managers assess **market volatility and liquidity**
- Dynamic limit adjustment based on VIX levels
- Correlation breakdown detection (systemic risk)
- Put/Call ratio as fear gauge

**3. Performance Metrics** ([StockBench](https://arxiv.org/html/2510.02209v1))
- Evaluate using **Sharpe ratio, Sortino ratio, max drawdown**
- **Win rate** and profit factor tracking
- Risk-adjusted returns over absolute returns

**4. Multi-Modal Integration** ([AutoStrategy](https://arxiv.org/abs/2409.06289))
- Combine news + social + fundamentals + technical analysis
- News-based risk assessment scores
- Multi-timeframe technical analysis

**5. Pattern Recognition** ([Industry Examples](https://www.godofprompt.ai/blog/10-chatgpt-prompts-to-enhance-your-trading-with-ai))
- "Act as an experienced day trader" with specific analysis tasks
- Advanced charting tools and pattern recognition
- Specific entry/exit levels (not vague opinions)

---

## V2.0 Prompt Enhancements

### 1. Supervisor v2.0 - Multi-Agent Debate Pattern

**File**: [llm/prompts/supervisor_prompts.py](../llm/prompts/supervisor_prompts.py)

**Key Enhancements**:

**Multi-Agent Debate Framework**:
- GATHER → DEBATE → WEIGH → SYNTHESIZE → RISK FILTER → DECIDE
- Explicit bull/bear/neutral case analysis
- Disagreement level tracking (low/medium/high)
- Historical performance tracking and reflection

**Multi-Modal Signal Integration**:
```json
"multi_modal_signals": {
    "technical_signal": "bullish|bearish|neutral",
    "sentiment_signal": "bullish|bearish|neutral",
    "fundamental_signal": "bullish|bearish|neutral",
    "macro_signal": "bullish|bearish|neutral",
    "overall_alignment": 0.0-1.0
}
```

**Expected Performance Metrics**:
```json
"expected_metrics": {
    "expected_return": 0.0,
    "max_drawdown": 0.0,
    "win_probability": 0.0-1.0,
    "risk_reward_ratio": 0.0-10.0,
    "sortino_ratio": 0.0
}
```

**Historical Performance Tracking**:
- Weight agent opinions by historical Sharpe ratio
- Identify patterns in successful vs failed trades
- Adjust confidence based on similar past setups
- Reflection mechanism: "What did we learn from similar past trades?"

**Market Regime Specific Rules**:
- Trending Bull: Favor debit spreads, increase size to 30%
- Trending Bear: Favor debit spreads, reduce size to 15%
- Mean-Reverting: Favor iron condors/butterflies, 20% size
- High-Volatility (VIX >30): Reduce positions 50%, confidence threshold 0.85
- Low-Volatility (VIX <15): Standard positions, confidence threshold 0.75

**Example Improvements**:
- v1.0: Simple consensus voting
- v2.0: Multi-agent debate with bull/bear cases, multi-modal alignment score, historical performance comparison

**Temperature**: Reduced from 0.7 to 0.6 for more consistent decision-making
**Max Tokens**: Increased from 1500 to 2000 for detailed reasoning

---

### 2. Technical Analyst v2.0 - Multi-Timeframe Pattern Recognition

**File**: [llm/prompts/analyst_prompts.py](../llm/prompts/analyst_prompts.py)

**Key Enhancements**:

**Multi-Timeframe Analysis**:
```json
"timeframe_analysis": {
    "weekly_trend": "strong_uptrend|uptrend|sideways|downtrend|strong_downtrend",
    "daily_trend": "strong_uptrend|uptrend|sideways|downtrend|strong_downtrend",
    "intraday_trend": "strong_uptrend|uptrend|sideways|downtrend|strong_downtrend",
    "alignment": "aligned|partially_aligned|conflicting"
}
```

**Chart Pattern Recognition**:
- **Continuation**: Ascending/Descending Triangles, Flags, Pennants, Cup and Handle
- **Reversal**: Head and Shoulders, Double Top/Bottom, Wedges
- **Candlestick**: Hammer, Shooting Star, Engulfing patterns, Doji

**Divergence Detection** (Critical Signals):
- Bullish divergence: Price lower low + RSI higher low = reversal signal
- Bearish divergence: Price higher high + RSI lower high = reversal signal
- Volume divergence: Price rises on declining volume = weak rally

**Specific Trade Setups**:
```json
"trade_setup": {
    "entry_price": 177.00,
    "stop_loss": 173.50,
    "profit_target_1": 182.00,
    "profit_target_2": 185.00,
    "risk_reward_ratio": 2.86,
    "position_sizing_note": "Use 50% position at PT1, trail stop for remaining"
}
```

**Additional Indicators**:
- ATR (Average True Range) for volatility measurement
- Volume Profile for institutional support/resistance levels
- Bollinger Bandwidth for squeeze/expansion detection

**Example Improvements**:
- v1.0: Single-timeframe analysis, basic indicators
- v2.0: Multi-timeframe alignment, pattern recognition, divergences, specific entry/stop/targets

**Temperature**: Reduced from 0.5 to 0.4 for more consistent technical analysis
**Max Tokens**: Increased from 1000 to 1500 for detailed pattern descriptions

---

### 3. Portfolio Risk Manager v2.0 - Dynamic Limit Adjustment

**File**: [llm/prompts/risk_prompts.py](../llm/prompts/risk_prompts.py)

**Key Enhancements**:

**Market Condition Assessment**:
```json
"market_conditions": {
    "vix_level": 32.0,
    "vix_status": "elevated",
    "market_liquidity": "stressed",
    "avg_bid_ask_spread_pct": 7.0,
    "put_call_ratio": 1.15,
    "correlation_breakdown": false
}
```

**VIX-Based Dynamic Limits**:
```
- VIX <15: position_size_multiplier = 1.2 (can increase 20%)
- VIX 15-25: position_size_multiplier = 1.0 (standard)
- VIX 25-35: position_size_multiplier = 0.8 (reduce 20%)
- VIX 35-50: position_size_multiplier = 0.5 (reduce 50%)
- VIX >50: position_size_multiplier = 0.0 (halt new trades)
```

**Liquidity Assessment**:
- Normal: Avg spread <5%, allow full trading
- Stressed: Avg spread 5-10%, reduce position sizes 30%
- Illiquid: Avg spread >10%, halt new trades

**Risk-Adjusted Metrics**:
```json
"risk_adjusted_metrics": {
    "sharpe_ratio": 1.8,
    "sortino_ratio": 2.2,
    "max_drawdown_pct": 3.0,
    "win_rate": 0.62,
    "profit_factor": 2.5
}
```

**Additional Exposures Tracked**:
- Net vega (volatility risk): ±10% limit
- Net gamma (second-order delta risk)
- Correlation breakdown detection (systemic risk)
- Put/Call ratio (fear gauge)

**Example Improvements**:
- v1.0: Static limits, basic Greeks exposure
- v2.0: Dynamic VIX-based limits, liquidity assessment, Sharpe/Sortino tracking, correlation breakdown detection

**Temperature**: Reduced from 0.1 to 0.05 for maximum consistency in risk decisions
**Max Tokens**: Increased from 1000 to 1200 for detailed market condition analysis

---

## Comparison: v1.0 vs v2.0

| Feature | v1.0 | v2.0 |
|---------|------|------|
| **Supervisor** | | |
| Decision Making | Simple consensus | Multi-agent debate (bull/bear cases) |
| Data Integration | Basic | Multi-modal (technical + sentiment + fundamental + macro) |
| Historical Tracking | None | Historical performance, reflection mechanism |
| Expected Metrics | Basic | Sortino ratio, max drawdown, win probability |
| Market Regime | Simple adjustments | Detailed regime-specific rules |
| **Technical Analyst** | | |
| Timeframes | Single | Multi-timeframe (weekly/daily/intraday alignment) |
| Patterns | None | 15+ patterns (continuation, reversal, candlestick) |
| Divergences | Not detected | Bullish/bearish/volume divergences |
| Trade Setups | Vague | Specific entry/stop/targets, risk/reward ratio |
| Additional Indicators | 7 indicators | +ATR, Volume Profile, Bollinger Bandwidth |
| **Portfolio Risk Manager** | | |
| Position Limits | Static | Dynamic (VIX-based multiplier) |
| Market Conditions | Not monitored | VIX status, liquidity assessment, correlation |
| Risk Metrics | Basic | Sharpe, Sortino, profit factor, win rate |
| Greeks Exposure | Delta, Theta | +Gamma, Vega |
| Systemic Risk | Not detected | Correlation breakdown, put/call ratio |

---

## Implementation Status

### Completed ✅
- [x] Supervisor v2.0 with multi-agent debate
- [x] Technical Analyst v2.0 with multi-timeframe analysis
- [x] Portfolio Risk Manager v2.0 with dynamic limits
- [x] All prompts registered in version control system
- [x] Testing confirmed all prompts load correctly

### Total Prompts Created

| Agent Role | v1.0 | v1.1 | v2.0 | Total |
|------------|------|------|------|-------|
| Supervisor | ✅ | ✅ | ✅ | 3 |
| Technical Analyst | ✅ | - | ✅ | 2 |
| Sentiment Analyst | ✅ | - | - | 1 |
| Conservative Trader | ✅ | - | - | 1 |
| Moderate Trader | ✅ | - | - | 1 |
| Aggressive Trader | ✅ | - | - | 1 |
| Position Risk Manager | ✅ | - | - | 1 |
| Portfolio Risk Manager | ✅ | - | ✅ | 2 |
| Circuit Breaker Manager | ✅ | - | - | 1 |
| **Total** | **9** | **1** | **3** | **13 versions** |

---

## Key Patterns Incorporated

### 1. **Act as an Expert** Pattern
All v2.0 prompts start with "Act as an experienced [role]" to invoke domain expertise.

**Source**: [God of Prompt](https://www.godofprompt.ai/blog/10-chatgpt-prompts-to-enhance-your-trading-with-ai)

**Example**: "Act as an experienced fund manager coordinating a team..." (Supervisor v2.0)

### 2. **Multi-Agent Debate** Pattern
Explicit bull/bear case analysis before final decision.

**Source**: [TradingAgents Framework](https://tradingagents-ai.github.io/)

**Example**: Supervisor v2.0 DEBATE step with bull_case/bear_case/disagreement_level

### 3. **ReAct Prompting** (Already in Base)
Think → Act → Observe cycle for transparent reasoning.

**Source**: [TradingAgents](https://arxiv.org/abs/2412.20138)

**Implementation**: All agents inherit from TradingAgent (base.py)

### 4. **Specific Trade Setups** Pattern
Always provide entry, stop loss, profit targets - not vague opinions.

**Source**: [Day Trading AI Prompts](https://optimusfutures.com/blog/ai-prompts-for-day-trading/)

**Example**: Technical Analyst v2.0 trade_setup with specific price levels

### 5. **Risk-Adjusted Metrics** Pattern
Evaluate strategies using Sharpe, Sortino, max drawdown, not just returns.

**Source**: [StockBench](https://arxiv.org/html/2510.02209v1)

**Example**: Portfolio Risk Manager v2.0 risk_adjusted_metrics

### 6. **Dynamic Adaptation** Pattern
Adjust parameters based on market conditions (VIX, liquidity).

**Source**: [TradingAgents Risk Management](https://tradingagents-ai.github.io/)

**Example**: Portfolio Risk Manager v2.0 VIX-based position_size_multiplier

---

## Research Sources

### Academic Papers

1. **TradingAgents: Multi-Agents LLM Financial Trading Framework**
   - URL: [https://arxiv.org/abs/2412.20138](https://arxiv.org/abs/2412.20138)
   - Key Contribution: Multi-agent architecture, ReAct framework, risk team oversight

2. **StockBench: Can LLM Agents Trade Stocks Profitably?**
   - URL: [https://arxiv.org/html/2510.02209v1](https://arxiv.org/html/2510.02209v1)
   - Key Contribution: Risk-adjusted performance metrics (Sharpe, Sortino, max drawdown)

3. **Automate Strategy Finding with LLM in Quant Investment**
   - URL: [https://arxiv.org/abs/2409.06289](https://arxiv.org/abs/2409.06289)
   - Key Contribution: Multi-modal agent evaluation, market status awareness

### Industry Resources

4. **God of Prompt: 10 Best ChatGPT Prompts For Stock Trading**
   - URL: [https://www.godofprompt.ai/blog/10-chatgpt-prompts-to-enhance-your-trading-with-ai](https://www.godofprompt.ai/blog/10-chatgpt-prompts-to-enhance-your-trading-with-ai)
   - Key Contribution: "Act as experienced trader" pattern, specific analysis tasks

5. **Optimus Futures: AI Prompts for Day Trading**
   - URL: [https://optimusfutures.com/blog/ai-prompts-for-day-trading/](https://optimusfutures.com/blog/ai-prompts-for-day-trading/)
   - Key Contribution: Advanced charting, pattern recognition, specific trade setups

6. **FlowHunt: Autonomous AI Stock Trading Bot**
   - URL: [https://www.flowhunt.io/ai-flow-templates/ai-trading-bot/](https://www.flowhunt.io/ai-flow-templates/ai-trading-bot/)
   - Key Contribution: Autonomous decision-making (buy/sell/hold/close/short)

7. **Alpaca: How Traders Use AI Agents**
   - URL: [https://alpaca.markets/learn/how-traders-are-using-ai-agents-to-create-trading-bots-with-alpaca](https://alpaca.markets/learn/how-traders-are-using-ai-agents-to-create-trading-bots-with-alpaca)
   - Key Contribution: Natural language trading, multi-sub-agent coordination

### Framework Documentation

8. **TradingAgents Framework Homepage**
   - URL: [https://tradingagents-ai.github.io/](https://tradingagents-ai.github.io/)
   - Key Contribution: Complete multi-agent architecture, risk management team

9. **Medium: From Trading Bot to Trading Agent**
   - URL: [https://medium.com/@gwrx2005/from-trading-bot-to-trading-agent-how-to-build-an-ai-based-investment-system-313d4c370c60](https://medium.com/@gwrx2005/from-trading-bot-to-trading-agent-how-to-build-an-ai-based-investment-system-313d4c370c60)
   - Key Contribution: Evolution from static bots to adaptive agents

10. **Medium: AI Agent Created Trading Strategy Autonomously**
    - URL: [https://medium.com/@austin-starks/an-ai-agent-created-this-trading-strategy-autonomously-with-1-prompt-8c6737ec49d3](https://medium.com/@austin-starks/an-ai-agent-created-this-trading-strategy-autonomously-with-1-prompt-8c6737ec49d3)
    - Key Contribution: Single-prompt strategy creation, natural language trading

---

## Cost Implications

The v2.0 prompts use slightly higher token counts:

| Agent | v1.0 Max Tokens | v2.0 Max Tokens | Increase |
|-------|----------------|-----------------|----------|
| Supervisor | 1500 | 2000 | +33% |
| Technical Analyst | 1000 | 1500 | +50% |
| Portfolio Risk Manager | 1000 | 1200 | +20% |

**Estimated Cost Impact**: +20-30% per agent call

**Justification**:
- More detailed reasoning improves decision quality
- Specific trade setups reduce ambiguity
- Historical tracking prevents repeating mistakes
- Multi-modal integration increases win rate

**Expected ROI**:
- Better decisions → Higher Sharpe ratio
- Specific setups → Better risk/reward
- Historical learning → Fewer repeated mistakes
- Net positive despite higher token costs

---

## Next Steps

### Immediate (Continue Implementation)

1. **Implement Concrete Agent Classes** ⏭️ NEXT
   - Create SupervisorAgent, TechnicalAnalyst, etc.
   - Integrate with prompt templates
   - Add tool calling capabilities

2. **Integrate Anthropic API**
   - Install anthropic SDK
   - Create Claude client wrapper
   - Implement retry logic and rate limiting

3. **Integrate FinBERT**
   - Install transformers + torch
   - Load FinBERT model
   - Create sentiment analysis tool

4. **Create Agent Orchestration**
   - Install LangGraph
   - Define TradingFirmState
   - Create StateGraph with 5 POC agents

### Future Enhancements

5. **Create v2.0 for Remaining Agents**
   - Sentiment Analyst v2.0 with news-based risk scores
   - Trader agents v2.0 with strategy execution patterns
   - Circuit Breaker v2.0 with flash crash detection

6. **Add Historical Performance Tracking Database**
   - Store all agent decisions and outcomes
   - Calculate per-agent Sharpe ratios
   - Enable reflection mechanism

7. **A/B Test v1.0 vs v2.0**
   - Run both versions in parallel
   - Compare Sharpe ratio, win rate, max drawdown
   - Activate best-performing versions

---

## Changelog

### 2025-12-01
- ✅ Researched 10+ sources on trading AI agents
- ✅ Created Supervisor v2.0 with multi-agent debate
- ✅ Created Technical Analyst v2.0 with multi-timeframe analysis
- ✅ Created Portfolio Risk Manager v2.0 with dynamic limits
- ✅ Documented all enhancements and research sources
- ✅ Total: 13 prompt versions across 9 agent roles
