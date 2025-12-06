---
title: "Specialized Prompts Research"
topic: prompts
related_upgrades: []
related_docs: []
tags: [prompts]
created: 2025-12-01
updated: 2025-12-02
---

# Specialized AI Prompt Research for Trading Agents

**Date**: 2024-12-01
**Purpose**: Research specialized prompts for each agent type to enhance prompt templates

---

## Research Summary

This document contains findings from specialized web searches on AI prompts for each agent role in our multi-agent trading system. These findings will be incorporated into prompt enhancements.

---

## 1. Supervisor / Orchestration Agent Research

### Key Findings

**Orchestration Patterns:**
- **Centralized Orchestration**: Single orchestrator oversees all agents (good for tight control, risk of bottleneck)
- **Hierarchical Orchestration**: Layers of control with top-level delegating to intermediate orchestrators (better scalability)
- **Group Chat Pattern**: Multiple agents collaborate through shared conversation thread with chat manager coordination

**Recent Developments (2024):**
- Amazon Bedrock multi-agent collaboration: 70% improvement in goal success rates vs single-agent
- Payload referencing improves code-intensive tasks by 23%
- Major cloud providers (AWS, Azure, IBM) actively developing orchestration frameworks

**Design Considerations:**
- Clear leadership hierarchy
- Dynamic team construction
- Effective information sharing
- Planning mechanisms like chain-of-thought prompting
- Memory systems for contextual learning
- Strategic orchestration of specialized models

**Popular Frameworks:**
- LangGraph: Dynamic graphs with skill-based/role-based agents
- AutoGen: Coordinator + Worker designs with reflective agents
- CrewAI: Role-based team members
- Botpress: Flexible multi-agent conversations

**Prompt Enhancements to Incorporate:**
- Chain-of-thought planning steps
- Memory/context tracking across decisions
- Dynamic team member weighting based on historical accuracy
- Reflection on past decisions
- Conflict resolution protocols

### Sources:
- [AWS Multi-Agent Orchestration Guidance](https://aws.amazon.com/solutions/guidance/multi-agent-orchestration-on-aws/)
- [Azure AI Agent Design Patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)
- [GenAI Multi-Agent Collaboration Research (Dec 2024)](https://arxiv.org/html/2412.05449v1)
- [Databricks Multi-Agent Supervisor](https://docs.databricks.com/aws/en/generative-ai/agent-bricks/multi-agent-supervisor)
- [Botpress AI Agent Orchestration](https://botpress.com/blog/ai-agent-orchestration)
- [Medium: Building Multi-Agent Architectures](https://medium.com/@akankshasinha247/building-multi-agent-architectures-orchestrating-intelligent-agent-systems-46700e50250b)
- [Superbo AI Agent Orchestration](https://superbo.ai/ai-agent-orchestration-enabling-intelligent-multi-agent-systems/)
- [IBM AI Agent Orchestration](https://www.ibm.com/think/topics/ai-agent-orchestration)
- [Medium: Technical Guide to Multi-Agent Orchestration](https://dominguezdaniel.medium.com/a-technical-guide-to-multi-agent-orchestration-5f979c831c0d)
- [AWS Multi-Agent Orchestration with Reasoning](https://aws.amazon.com/blogs/machine-learning/design-multi-agent-orchestration-with-reasoning-using-amazon-bedrock-and-open-source-frameworks/)

---

## 2. Technical Analyst Research

### Key Findings

**AI-Enhanced Technical Analysis:**
- Automated pattern recognition eliminates human bias and emotional decision-making
- Simultaneous analysis of thousands of stocks
- Real-time market scanning across multiple timeframes
- Objective signal generation with no fatigue

**Leading AI Tools:**
- **TrendSpider**: Automated pattern recognition, AI Strategy Lab, multi-timeframe analysis
- **Trade Ideas**: Real-time trade signals and scanning
- **Tickeron**: 40+ distinct chart patterns (stocks, ETFs, forex, crypto)
- **ChartPatterns.ai**: 16 essential chart patterns with visual detection

**Common Patterns Detected:**
- **Reversal Patterns**: Head and Shoulders, Cup and Handle, Double Top/Bottom
- **Continuation Patterns**: Triangles, Wedges, Flags, Pennants
- **Candlestick Patterns**: Hammers, Engulfing patterns, Morning/Evening Stars
- Pattern reliability ranking systems

**Important Considerations:**
- Don't treat AI signals as gospel
- Markets can behave unpredictably during unusual events
- Use AI recommendations as ONE input, not the only factor
- Combine with fundamental analysis and risk management

**Prompt Enhancements to Incorporate:**
- Comprehensive pattern library (40+ patterns)
- Pattern reliability scoring (high/medium/low confidence)
- Multi-timeframe confirmation requirements
- Objective bias-free analysis structure
- Specific entry/exit/stop recommendations
- Risk/reward ratio calculations
- Pattern invalidation conditions

### Sources:
- [Stock Charts AI](https://stockchartsai.com/)
- [Intellectia AI Stock Chart Patterns](https://intellectia.ai/features/stock-chart-patterns)
- [Pocket Option Chart Pattern Recognition](https://pocketoption.com/blog/en/interesting/trading-platforms/chart-patterns/)
- [Wall Street Zen: 6 Best AI Technical Analysis Tools 2025](https://www.wallstreetzen.com/blog/best-ai-technical-analysis-tools/)
- [Kavout AI Trading Chart Patterns](https://www.kavout.com/trading-signals/)
- [ChartPatterns.ai](https://chartpatterns.ai/)
- [Tickeron Pattern Search Engine](https://tickeron.com/stock-pattern-screener/)
- [Stock Chart Pattern Recognition with Deep Learning (arXiv)](https://arxiv.org/pdf/1808.00418)
- [AI Chart Pattern Scanner (Google Play)](https://play.google.com/store/apps/details?id=com.snapresearch.boto&hl=en_US)
- [Pragmatic Coders: Top AI Tools for Traders 2026](https://www.pragmaticcoders.com/blog/top-ai-tools-for-traders)

---

## 3. Sentiment Analyst Research

### Key Findings

**Core Techniques:**
- **NLP (Natural Language Processing)**: Scans news, financial reports, earnings calls, SEC filings
- **Advanced Models**: FinBERT, BERT, GPT-3 enhance context understanding and detect subtle shifts
- **Emotional Tone Detection**: Optimistic, pessimistic, neutral with nuanced understanding
- **Behavioral Finance**: Investor psychology and sentiment influence decision-making

**Trading Psychology & Market Sentiment:**
- Sentiment influences market decisions from psychological perspective
- Gauge market sentiment: bullish (positive), bearish (negative), or neutral
- Real-time sentiment analysis now possible with NLP and ML advances
- Process vast amounts of textual data almost instantaneously

**Predictive Accuracy:**
- Incorporating user sentiment increases price forecast accuracy by 20%
- Translates into better portfolio performance through informed asset allocation
- Enhanced predictive models for stock market behavior

**Challenges:**
- Data noise: Social media mixes relevant with irrelevant information
- Emotional biases complicate data accuracy
- Need to filter signal from noise

**Prompt Enhancements to Incorporate:**
- Behavioral finance perspective in analysis
- Emotional tone classification (optimistic/pessimistic/neutral)
- Noise filtering instructions
- Real-time vs historical sentiment comparison
- Contrarian signal detection (extreme sentiment as reversal indicator)
- Sentiment-price divergence analysis
- Multi-source aggregation (news + social + analyst ratings)

### Sources:
- [AIMultiple: Sentiment Analysis Stock Market](https://research.aimultiple.com/sentiment-analysis-stock-market/)
- [QuantifiedStrategies: AI Sentiment Analysis for Trading](https://www.quantifiedstrategies.com/ai-sentiment-analysis-for-trading/)
- [Moody's: Power of News Sentiment](https://www.moodys.com/web/en/us/insights/digital-transformation/the-power-of-news-sentiment-in-modern-financial-analysis.html)
- [Markets.com: How AI Analyzes Market Sentiment](https://www.markets.com/education-centre/how-ai-analyzes-market-sentiment/)
- [ACM: Financial Sentiment Analysis Techniques](https://dl.acm.org/doi/10.1145/3649451)
- [Insight7: Financial Sentiment Analysis Overview](https://insight7.io/financial-sentiment-analysis-quick-overview/)
- [StockGeist.ai Market Sentiment Platform](https://www.stockgeist.ai/)
- [ResearchGate: Sentiment Analysis in Financial News](https://www.researchgate.net/publication/390056578_Sentiment_Analysis_in_Financial_News_Enhancing_Predictive_Models_for_Stock_Market_Behavior)
- [Accio Analytics: Ultimate Guide to Sentiment Analysis](https://accioanalytics.io/insights/ultimate-guide-to-sentiment-analysis-in-finance/)
- [StockGeist.ai Features](https://www.stockgeist.ai/stock-sentiment-features/)

---

## 4. Trading Strategy (Conservative Trader) Research

### Key Findings

**Conservative Trading Persona:**
- "You are a conservative institutional trader with 15 years experience managing pension fund assets"
- "Your primary mandate is capital preservation with steady returns"
- Risk-averse mindset prioritizes protecting capital over aggressive growth

**Risk Management Parameters:**
- **Max risk per trade**: 0.5% (very conservative) to 1% (moderately conservative)
- **Daily drawdown limits**: 2%
- **Position sizing**: Based on stop loss distance to risk exactly the specified percentage
- **Account exposure**: Typically 1-3% per position

**Options Trading Strategies (Conservative):**
- **Low volatility conditions**: Sell covered calls or cash-secured puts for steady income
- **High IV conditions**: Avoid selling premium, consider buying spreads
- **Preference**: Defined-risk strategies (spreads vs naked options)
- **Expiration**: Favor 30-60 day expirations for theta decay

**Prompt Structure Best Practices:**
- Specify asset class, trading style, risk tolerance explicitly
- Provide context: market view, risk tolerance, timeframe
- Include specific constraints (max loss, target return, holding period)
- Guide exploration: strategy development, risk management, market analysis
- More context = more on-target responses

**Position Sizing Formulas:**
- Fixed fractional: Risk % of account on each trade
- Volatility-scaled: Adjust size based on ATR or recent volatility
- Kelly Criterion: Optimize bet size based on win rate and risk/reward
- Always calculate based on stop loss distance

**Prompt Enhancements to Incorporate:**
- Conservative institutional trader persona
- Explicit risk parameters (0.5-1% per trade, 2% daily limit)
- Capital preservation mandate
- Strategy preference guidance (covered calls, CSPs, spreads)
- IV regime considerations
- Position sizing calculations
- Context-rich prompts requiring market view, timeframe, risk tolerance

### Sources:
- [MQL5: Perfect System Prompt for Trading Style](https://www.mql5.com/en/blogs/post/764373)
- [God of Prompt: 10 Best ChatGPT Prompts for Stock Trading](https://www.godofprompt.ai/blog/10-chatgpt-prompts-to-enhance-your-trading-with-ai)
- [AI Signals: Best AI Options Trading Strategies](https://ai-signals.com/best-ai-options-trading-strategies-for-high-probability-trades/)
- [LearnPrompt: 61 Best ChatGPT Prompts for Stock Trading](https://www.learnprompt.org/chatgpt-prompts-for-stock-trading/)
- [ClickUp: Best AI Prompts for Trading](https://clickup.com/p/ai/prompts/trading)
- [Pepperstone: ChatGPT Prompts for Trading](https://pepperstone.com/en/learn-to-trade/trading-guides/how-to-use-chatgpt-in-trading/)
- [OptionsTrading.org: Can ChatGPT Improve Options Trading?](https://www.optionstrading.org/blog/can-chatgpt-improve-options-trading/)
- [DocsBot: Options Trading Prompts](https://docsbot.ai/prompts/business/options-trading-prompts)
- [ITI Rupati: ChatGPT Prompts for Trading Strategy](https://itirupati.com/chatgpt-prompt-for-trading-strategy/)
- [ChatGPT AI Hub: 99+ Successful Prompts for Stock Trading](https://chatgptaihub.com/chatgpt-prompts-for-stock-trading/)

---

## 5. Risk Management Research

### Key Findings

**Circuit Breakers:**
- **Level 1** (7% decline): 15-minute trading pause
- **Level 2** (13% decline): 15-minute trading pause
- **Level 3** (20% decline): Trading halted for rest of day
- Essential emergency measures to prevent market manipulation
- Regulators enforce stricter compliance

**Stop-Loss Strategies:**
- **Fixed stops**: Set percentage or dollar amount
- **Trailing stops**: Move with profitable price action
- **Volatility-based stops**: ATR (Average True Range) ensures stops aren't too tight or loose
- AI adjusts stop-loss levels based on real-time market movements
- ML models predict optimal exit timing

**Portfolio Risk Controls:**
- Diversification across strategies and assets
- Correlation analysis to avoid concentrated risk
- Stress testing under extreme market conditions
- Drawdown limits prevent spiraling losses
- Max loss per strategy or asset class

**Trading Limits & Position Sizing:**
- **Smart position sizing methods**:
  - Fixed fractional: 1-3% of portfolio per trade
  - Volatility-scaled: Adjust for recent volatility
  - Notional target sizing: Dollar amount per position
- Drawdown limits stop trading after losing specified percentage
- Max concurrent positions to limit overexposure

**AI Risk Management Prompts:**
- Develop comprehensive risk management strategies
- Calculate optimal position sizing using AI algorithms
- Implement stop loss optimization techniques
- Create risk tolerance assessment frameworks
- Monitor real-time risk exposure
- Dynamic limit adjustment based on market conditions

**Prompt Enhancements to Incorporate:**
- Circuit breaker thresholds (7%/13%/20%)
- Volatility-based stop logic (ATR)
- Drawdown limit enforcement
- Position sizing formulas (1-3% per trade)
- Diversification requirements
- Stress testing scenarios
- Real-time risk monitoring
- Dynamic limit adjustment protocols
- Emergency halt conditions

### Sources:
- [LuxAlgo: Risk Management for Algo Trading](https://www.luxalgo.com/blog/risk-management-strategies-for-algo-trading/)
- [3Commas: AI Trading Bot Risk Management 2025 Guide](https://3commas.io/blog/ai-trading-bot-risk-management-guide-2025)
- [3Commas: Risk Management Settings Configuration](https://3commas.io/blog/ai-trading-bot-risk-management-guide)
- [ResearchGate: AI in Financial Markets Risk Management](https://www.researchgate.net/publication/390321542_Artificial_Intelligence_in_Financial_Markets_Optimizing_Risk_Management_Portfolio_Allocation_and_Algorithmic_Trading)
- [Nurp: 7 Risk Management Strategies for Algorithmic Trading](https://nurp.com/wisdom/7-risk-management-strategies-for-algorithmic-trading/)
- [Medium: Implement AI-Driven Risk Management](https://medium.com/@deepml1818/how-to-implement-ai-driven-risk-management-in-trading-909539c6f95c)
- [IG UK: Automated Risk Management](https://www.ig.com/uk/research/future-of-trading/risk-management-is-automatic)
- [Funded Nest: 10 Essential AI Trading Prompts](https://fundednest.com/10-essential-ai-trading-prompts-for-stock-market-analysis-and-better-investment-decisions/)
- [Tradetron: Reducing Drawdown Techniques](https://tradetron.tech/blog/reducing-drawdown-7-risk-management-techniques-for-algo-traders)
- [Wall Street Prep: AI in Risk Management](https://www.wallstreetprep.com/knowledge/ai-in-risk-management/)

---

## Key Patterns to Incorporate Across All Prompts

### 1. **Structured Decision Frameworks**
- Step-by-step reasoning processes
- Chain-of-thought prompting
- Clear decision criteria
- Explicit output formats (JSON)

### 2. **Context-Rich Prompts**
- Require comprehensive input data
- Specify all necessary parameters
- Include market context (VIX, regime, liquidity)
- Historical performance reference

### 3. **Risk-First Approach**
- Always consider downside before upside
- Explicit risk limits and constraints
- Position sizing calculations
- Stop loss and exit criteria

### 4. **Multi-Modal Integration**
- Combine technical + sentiment + fundamental signals
- Cross-validate signals across sources
- Alignment scoring
- Conflicting signal resolution

### 5. **Continuous Learning**
- Reference historical decisions
- Learn from past mistakes
- Performance tracking metrics
- Adaptive limit adjustment

### 6. **Bias Mitigation**
- Objective, data-driven analysis
- Avoid emotional language
- Contrarian signal detection
- Devil's advocate reasoning

### 7. **Specific Actionability**
- Concrete entry/exit prices
- Stop loss levels
- Profit targets
- Position sizing recommendations
- Risk/reward ratios

---

## Next Steps

1. **Iterate on Existing Prompts**: Enhance v1.0, v1.1, v2.0 with these findings
2. **Create v3.0 Prompts**: Incorporate all research for comprehensive versions
3. **Create Missing v2.0 Prompts**: Ensure all agent roles have latest versions
4. **Document Enhancements**: Track what was added from which source
5. **Test Enhanced Prompts**: Validate improvements with sample scenarios

---

**Total Sources**: 40+ research sources across 5 agent types
**Key Themes**: Structured frameworks, context-rich prompts, risk-first approach, multi-modal integration, continuous learning, bias mitigation, specific actionability
