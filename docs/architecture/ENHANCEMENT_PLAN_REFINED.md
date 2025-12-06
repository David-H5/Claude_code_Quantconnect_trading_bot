# Hybrid Algorithm Enhancement Plan - REFINED

**Version**: 2.1 (Refined)
**Date**: November 30, 2025
**Focus**: Anthropic Claude-Based Multi-Agent System + FinBERT
**Status**: Proof of Concept Phase

---

## Executive Summary

**REFINED APPROACH**: Focus on Anthropic Claude ecosystem + open-source FinBERT

**Key Changes from v2.0**:
1. ✅ **All agents use Anthropic Claude** (not mixed LLM providers)
   - Claude Sonnet 4: Quick analysis, data processing
   - Claude Opus 4: Deep reasoning, strategy decisions
   - Claude Haiku: Fast tool calling, simple tasks
2. ✅ **FinBERT for financial sentiment** (proven, open-source)
3. ✅ **Prompt template system** for easy iteration
4. ✅ **Proof of concept first** before full 8-week implementation
5. ✅ **QuantConnect-native integration** from day one

**Why This Approach?**:
- **Consistency**: Single provider (Anthropic) = easier maintenance
- **Quality**: Claude excels at financial reasoning and structured output
- **Cost-effective**: Claude pricing competitive, FinBERT free
- **Proven**: FinBERT is state-of-the-art for financial sentiment
- **Integration**: Already using Claude Code for development

---

## Refined Architecture

### Multi-Agent System (Claude-Based)

```
TradingFirm
├─ Supervisor Agent (Claude Opus 4)
│   └─ Orchestrates all agents, makes final decisions
│
├─ Analysis Team (Claude Sonnet 4)
│   ├─ FundamentalsAnalyst - Earnings, financials, valuation
│   ├─ TechnicalAnalyst - Charts, indicators, patterns
│   ├─ SentimentAnalyst - FinBERT + news aggregation
│   ├─ NewsAnalyst - Breaking news, macro events
│   └─ VolatilityAnalyst - IV analysis, volatility predictions
│
├─ Research Team (Claude Sonnet 4)
│   ├─ BullResearcher - Bullish thesis development
│   ├─ BearResearcher - Bearish thesis development
│   └─ MarketRegimeAnalyst - Trend/range/volatile detection
│
├─ Trading Team (Claude Opus 4)
│   ├─ ConservativeTrader - Low risk, high probability
│   ├─ ModerateTrader - Balanced approach
│   └─ AggressiveTrader - High risk, high reward
│
└─ Risk Management Team (Claude Haiku)
    ├─ PositionRiskManager - Per-trade validation
    ├─ PortfolioRiskManager - Portfolio-level checks
    └─ CircuitBreakerManager - Safety monitoring
```

### Model Selection Strategy

**Claude Opus 4** (Deep Reasoning):
- Supervisor agent (orchestration)
- Trading team (final decisions)
- Complex multi-step analysis
- Cost: Higher, but worth it for critical decisions

**Claude Sonnet 4** (Balanced):
- Analysis team (most agents)
- Research team
- General reasoning tasks
- Cost: Moderate, best value for performance

**Claude Haiku** (Fast):
- Risk management (quick checks)
- Tool calling
- Simple validation tasks
- Cost: Lowest, fastest responses

**FinBERT** (Specialized):
- Financial sentiment analysis
- Open-source, free
- Fine-tuned on financial data
- Runs locally or via HuggingFace

---

## Proof of Concept Plan

### Phase 1: Foundation (Week 1)

**Goal**: Working multi-agent system with 5 agents minimum

**Agents to Implement**:
1. ✅ Supervisor Agent (Opus 4)
2. ✅ TechnicalAnalyst (Sonnet 4)
3. ✅ SentimentAnalyst (Sonnet 4 + FinBERT)
4. ✅ ConservativeTrader (Opus 4)
5. ✅ PositionRiskManager (Haiku)

**Deliverables**:
- Working agent framework with prompt templates
- Anthropic API integration
- FinBERT integration
- Prompt management system
- Basic orchestration flow
- Unit tests for each agent

### Phase 2: Integration (Week 2)

**Goal**: Integrate agents into HybridOptionsBot

**Tasks**:
- Add agent system to Initialize()
- Route decisions through agents in OnData()
- Add agent logging and tracking
- Create simple dashboard
- Run backtest with agents

**Success Criteria**:
- Algorithm initializes with agents
- Agents provide recommendations
- Human can approve/reject
- All logging working

### Phase 3: Iteration (Week 3+)

**Goal**: Refine prompts based on real performance

**Tasks**:
- Analyze agent decisions
- Measure accuracy vs expert analysis
- A/B test different prompts
- Optimize for performance
- Add remaining agents as needed

---

## Prompt Template System

### Design Principles

1. **Modular**: Each role has separate prompt template
2. **Versioned**: Track prompt changes over time
3. **Testable**: Easy to A/B test variations
4. **Dynamic**: Variables injected at runtime
5. **Documented**: Clear explanation of each section

### Template Structure

```python
# Each prompt template has:
1. Role Definition - Who is the agent
2. Responsibilities - What they do
3. Tools Available - What they can call
4. Output Format - Structured response required
5. Examples - Few-shot learning
6. Constraints - What NOT to do
```

### Template Location

```
llm/prompts/
├── __init__.py
├── base_prompts.py           # Shared components
├── supervisor_prompts.py     # Supervisor variations
├── analyst_prompts.py        # Analysis team prompts
├── trader_prompts.py         # Trading team prompts
├── risk_prompts.py           # Risk management prompts
└── prompt_registry.py        # Version management
```

### Prompt Versioning System

```python
class PromptVersion:
    version: str              # e.g., "v1.0", "v1.1"
    template: str             # Prompt text with {variables}
    created_date: datetime
    performance_metrics: Dict # Win rate, accuracy, etc.
    notes: str               # What changed and why

class PromptRegistry:
    """Manages prompt versions and A/B testing."""

    def get_prompt(self, role: str, version: str = "latest") -> str
    def register_prompt(self, role: str, template: str, notes: str) -> str
    def compare_versions(self, role: str, v1: str, v2: str) -> Dict
    def get_best_performing(self, role: str) -> PromptVersion
```

---

## Implementation Checklist

### Week 1: Proof of Concept

**Day 1-2: Foundation**
- [ ] Set up Anthropic API client
- [ ] Create prompt template system
- [ ] Implement base agent with Claude integration
- [ ] Write 5 core prompt templates

**Day 3-4: Agents**
- [ ] Implement Supervisor (Opus)
- [ ] Implement TechnicalAnalyst (Sonnet)
- [ ] Implement SentimentAnalyst (Sonnet + FinBERT)
- [ ] Implement ConservativeTrader (Opus)
- [ ] Implement PositionRiskManager (Haiku)

**Day 5-7: Integration & Testing**
- [ ] Create agent orchestration flow
- [ ] Write unit tests
- [ ] Create simple UI for agent decisions
- [ ] Test end-to-end workflow
- [ ] Document usage

### Week 2: HybridOptionsBot Integration

**Tasks**:
- [ ] Add agent initialization in Algorithm.Initialize()
- [ ] Route strategy decisions through agents
- [ ] Add agent approval workflow
- [ ] Create agent performance dashboard
- [ ] Run backtest with agent recommendations

### Week 3+: Refinement

**Tasks**:
- [ ] Analyze agent decision quality
- [ ] A/B test prompt variations
- [ ] Measure performance impact
- [ ] Add remaining agents as needed
- [ ] Optimize for best results

---

## Anthropic Claude Integration

### API Setup

```python
# Install Anthropic SDK
pip install anthropic

# In algorithm
from anthropic import Anthropic

class HybridOptionsBot(QCAlgorithm):
    def Initialize(self):
        # Initialize Anthropic client
        api_key = self.GetParameter("anthropic_api_key")
        self.anthropic_client = Anthropic(api_key=api_key)

        # Create agents with client
        self.supervisor = SupervisorAgent(
            llm_client=self.anthropic_client,
            model="claude-opus-4-20250514"
        )
```

### Model Selection

```python
CLAUDE_MODELS = {
    "deep_reasoning": "claude-opus-4-20250514",      # Supervisor, Traders
    "balanced": "claude-sonnet-4-20250514",          # Analysts, Researchers
    "fast": "claude-3-5-haiku-20241022",             # Risk managers
}
```

### Rate Limiting

```python
# Respect Anthropic rate limits
- Tier 1: 50 requests/min
- Tier 2: 1,000 requests/min
- Tier 3: 2,000 requests/min
- Tier 4: 4,000 requests/min

# Implement exponential backoff
# Cache responses where possible
# Batch requests when appropriate
```

---

## FinBERT Integration

### Setup

```python
# Install required packages
pip install transformers torch

# Load FinBERT model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
```

### Usage in SentimentAnalyst

```python
class SentimentAnalyst(TradingAgent):
    def __init__(self, ...):
        super().__init__(...)
        # Load FinBERT locally
        self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert"
        )

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using FinBERT.

        Returns:
            {
                "positive": 0.0-1.0,
                "negative": 0.0-1.0,
                "neutral": 0.0-1.0,
            }
        """
        inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.finbert_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return {
            "positive": scores[0][0].item(),
            "negative": scores[0][1].item(),
            "neutral": scores[0][2].item(),
        }
```

---

## Prompt Templates (Examples)

### Supervisor Agent (Opus 4)

```
You are the Supervisor of a quantitative trading firm specializing in options trading.

ROLE:
You coordinate a team of specialized analysts, researchers, traders, and risk managers.
Your job is to synthesize their analyses into a final trading decision.

TEAM MEMBERS:
- Technical Analyst: Provides chart analysis and indicator signals
- Sentiment Analyst: Analyzes market sentiment and news
- Bull Researcher: Develops bullish thesis
- Bear Researcher: Develops bearish thesis
- Conservative Trader: Recommends low-risk trades
- Risk Manager: Validates all trades meet risk limits

DECISION PROCESS:
1. Review all team member analyses
2. Identify areas of agreement and disagreement
3. Weigh conflicting opinions based on confidence scores
4. Make final decision: BUY, SELL, HOLD, or NO_ACTION
5. Provide clear reasoning for your decision

OUTPUT FORMAT (JSON):
{
    "decision": "BUY|SELL|HOLD|NO_ACTION",
    "confidence": 0.0-1.0,
    "reasoning": "2-3 sentence explanation",
    "supporting_evidence": ["point 1", "point 2", ...],
    "risks": ["risk 1", "risk 2", ...],
    "recommended_strategy": "iron_condor|butterfly|...",
    "position_size": 0.0-1.0  // fraction of available capital
}

CONSTRAINTS:
- Never recommend trades that violate risk limits
- Always require >70% confidence for action
- Explain any disagreements between team members
- Be conservative in uncertain market conditions
```

### Technical Analyst (Sonnet 4)

```
You are a Technical Analyst specializing in options trading on equity indices.

ROLE:
Analyze price charts, technical indicators, and patterns to provide trading signals.

INDICATORS YOU ANALYZE:
- VWAP (Volume Weighted Average Price)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Support/Resistance levels
- Volume analysis

ANALYSIS PROCESS:
1. Review current price action and trend
2. Check indicator signals (bullish, bearish, neutral)
3. Identify key support/resistance levels
4. Assess volume confirmation
5. Determine overall technical outlook

OUTPUT FORMAT (JSON):
{
    "signal": "BULLISH|BEARISH|NEUTRAL",
    "confidence": 0.0-1.0,
    "trend": "UPTREND|DOWNTREND|RANGE",
    "key_levels": {
        "support": [price1, price2, ...],
        "resistance": [price1, price2, ...]
    },
    "indicators": {
        "rsi": {"value": 0-100, "signal": "overbought|neutral|oversold"},
        "macd": {"signal": "bullish|bearish|neutral"},
        "vwap": {"signal": "above|below"}
    },
    "reasoning": "2-3 sentences explaining your analysis"
}

CONSTRAINTS:
- Base analysis only on provided data (no look-ahead bias)
- Provide confidence score based on signal strength
- Acknowledge uncertainty when indicators conflict
```

### Sentiment Analyst (Sonnet 4 + FinBERT)

```
You are a Sentiment Analyst combining AI sentiment analysis with market news.

ROLE:
Analyze market sentiment from multiple sources and provide overall sentiment score.

DATA SOURCES:
- FinBERT sentiment scores (provided by tool)
- Recent news headlines
- Social media trends
- Market mood indicators

ANALYSIS PROCESS:
1. Use get_finbert_sentiment tool for each news item
2. Aggregate FinBERT scores across all news
3. Look for sentiment divergence from price action
4. Assess sentiment strength and conviction
5. Identify potential sentiment-driven catalysts

TOOLS AVAILABLE:
- get_finbert_sentiment(text: str) -> {"positive": float, "negative": float, "neutral": float}
- get_recent_news(symbol: str) -> List[NewsItem]
- get_social_sentiment(symbol: str) -> {"score": float, "volume": int}

OUTPUT FORMAT (JSON):
{
    "sentiment_score": -1.0 to +1.0,  // negative = bearish, positive = bullish
    "confidence": 0.0-1.0,
    "dominant_emotion": "fear|greed|uncertainty|optimism|pessimism",
    "news_summary": "1-2 sentence summary of key news",
    "divergence": null or "positive|negative",  // sentiment vs price action
    "catalyst_potential": "high|medium|low",
    "reasoning": "2-3 sentences explaining sentiment"
}

CONSTRAINTS:
- Use FinBERT for objective sentiment measurement
- Combine quantitative scores with qualitative analysis
- Flag unusual sentiment patterns
- Acknowledge when sentiment is mixed or unclear
```

### Conservative Trader (Opus 4)

```
You are a Conservative Trader with a low risk tolerance.

ROLE:
Recommend options strategies with high win rate and limited downside risk.

PREFERRED STRATEGIES:
1. Iron Condors (high probability, range-bound)
2. Credit Spreads (defined risk, high win rate)
3. Covered Calls (income generation)
4. Cash-Secured Puts (value entry)

RISK TOLERANCE:
- Maximum risk per trade: 2% of portfolio
- Minimum win probability: 65%
- Preferred delta: 0.15-0.25 (far OTM)
- Preferred DTE: 30-45 days

DECISION PROCESS:
1. Review all analyst reports
2. Assess if market conditions favor conservative strategies
3. Check if IV Rank supports premium selling
4. Verify risk/reward meets your criteria
5. Recommend specific strategy with parameters

OUTPUT FORMAT (JSON):
{
    "recommendation": "TRADE|NO_TRADE",
    "strategy": "iron_condor|credit_spread|covered_call|csp",
    "confidence": 0.0-1.0,
    "parameters": {
        "underlying": "SPY",
        "dte": 30-45,
        "short_delta": 0.15-0.25,
        "strikes": {
            "call_short": price,
            "call_long": price,
            "put_short": price,
            "put_long": price
        },
        "max_profit": dollars,
        "max_loss": dollars,
        "win_probability": 0.0-1.0
    },
    "reasoning": "Why this trade fits conservative criteria",
    "risks": ["risk 1", "risk 2"]
}

CONSTRAINTS:
- NEVER recommend trades with >2% risk
- NEVER recommend naked options
- NEVER trade during high volatility events (earnings, FOMC)
- ALWAYS require defined risk
- PREFER high probability over high reward
```

### Position Risk Manager (Haiku)

```
You are a Position Risk Manager. Your job: validate trades meet risk limits.

ROLE:
Quick, binary decision: APPROVE or REJECT trades based on risk rules.

RISK LIMITS:
- Max position size: 25% of portfolio
- Max risk per trade: 2% of portfolio
- Max open positions: 5
- Max portfolio delta: ±0.30
- Max portfolio theta: -0.50 per day

VALIDATION PROCESS:
1. Check position size vs limit
2. Check risk amount vs limit
3. Check current open positions count
4. Calculate new portfolio delta
5. Calculate new portfolio theta
6. Return APPROVE or REJECT

OUTPUT FORMAT (JSON):
{
    "decision": "APPROVE|REJECT",
    "reason": "1 sentence explanation if REJECT",
    "risk_metrics": {
        "position_size_pct": 0.0-1.0,
        "risk_pct": 0.0-1.0,
        "portfolio_delta": -1.0 to +1.0,
        "portfolio_theta": float
    }
}

CONSTRAINTS:
- Be fast (< 1 second response)
- Be strict (when in doubt, REJECT)
- No complex reasoning needed
- Simple yes/no decision
```

---

## Prompt Iteration System

### A/B Testing Framework

```python
class PromptABTest:
    """A/B test different prompt versions."""

    def __init__(self, role: str, prompt_a: str, prompt_b: str):
        self.role = role
        self.prompt_a = prompt_a
        self.prompt_b = prompt_b
        self.results_a = []
        self.results_b = []

    def run_test(self, test_cases: List[Dict], num_runs: int = 10):
        """Run both prompts on same test cases."""
        for case in test_cases:
            # Alternate between prompts
            for i in range(num_runs):
                if i % 2 == 0:
                    result = self.test_prompt(self.prompt_a, case)
                    self.results_a.append(result)
                else:
                    result = self.test_prompt(self.prompt_b, case)
                    self.results_b.append(result)

    def analyze_results(self) -> Dict:
        """Compare performance of both prompts."""
        return {
            "prompt_a_accuracy": np.mean([r["correct"] for r in self.results_a]),
            "prompt_b_accuracy": np.mean([r["correct"] for r in self.results_b]),
            "winner": "A" if mean_a > mean_b else "B",
            "confidence": scipy.stats.ttest_ind(...),
        }
```

### Prompt Performance Tracking

```python
class PromptPerformanceTracker:
    """Track prompt performance over time."""

    def record_decision(
        self,
        role: str,
        prompt_version: str,
        decision: Dict,
        actual_outcome: Dict,
    ):
        """Record agent decision and actual outcome."""
        self.db.insert({
            "role": role,
            "prompt_version": prompt_version,
            "decision": decision,
            "outcome": actual_outcome,
            "correct": self.is_correct(decision, actual_outcome),
            "timestamp": datetime.now(),
        })

    def get_best_prompt(self, role: str) -> str:
        """Get best-performing prompt version for role."""
        results = self.db.query(
            f"SELECT prompt_version, AVG(correct) as accuracy "
            f"FROM decisions WHERE role = '{role}' "
            f"GROUP BY prompt_version ORDER BY accuracy DESC LIMIT 1"
        )
        return results[0]["prompt_version"]
```

---

## Success Metrics

### Proof of Concept (Week 1-2)

**Technical Metrics**:
- [ ] All 5 agents implemented
- [ ] Claude API integration working
- [ ] FinBERT integration working
- [ ] Prompt system functional
- [ ] Unit tests passing

**Performance Metrics**:
- [ ] Agent decisions align with expert analysis >70%
- [ ] Response time <3 seconds per agent
- [ ] No API errors or timeouts
- [ ] Structured output parsing 100% success

### Integration (Week 2)

**Integration Metrics**:
- [ ] HybridOptionsBot initializes with agents
- [ ] Agents provide recommendations in OnData()
- [ ] Logging captures all agent decisions
- [ ] Dashboard displays agent reasoning
- [ ] Backtest completes without errors

### Refinement (Week 3+)

**Optimization Metrics**:
- [ ] Prompt improvements measured via A/B tests
- [ ] Agent accuracy improving over time
- [ ] Performance impact positive (Sharpe, win rate)
- [ ] User satisfaction with agent recommendations

---

## Refined Timeline

### Week 1: Proof of Concept
**Focus**: Get 5 agents working with Claude + FinBERT

**Deliverables**:
- Working agent framework
- 5 prompt templates
- Anthropic + FinBERT integration
- Unit tests
- Basic orchestration

### Week 2: Integration
**Focus**: Add agents to HybridOptionsBot

**Deliverables**:
- Algorithm integration
- Agent decision workflow
- Simple dashboard
- Backtest with agents
- Performance logging

### Week 3-4: Iteration
**Focus**: Refine prompts, measure performance

**Deliverables**:
- A/B tested prompts
- Performance tracking
- Best prompts identified
- Documentation updated
- Metrics dashboard

### Week 5+: Expansion (Optional)
**Focus**: Add remaining agents, advanced features

**Deliverables**:
- 8 additional agents (if needed)
- Advanced orchestration
- RL integration (if valuable)
- Production deployment

---

## Cost Analysis

### Anthropic Pricing (as of 2025)

**Claude Opus 4**:
- Input: $15/MTok
- Output: $75/MTok
- Use case: Supervisor (1 call per opportunity), Traders (3 calls per day)
- Estimated: ~$50-100/month

**Claude Sonnet 4**:
- Input: $3/MTok
- Output: $15/MTok
- Use case: Analysts (5 agents, 10 calls per day each)
- Estimated: ~$100-200/month

**Claude Haiku**:
- Input: $0.80/MTok
- Output: $4/MTok
- Use case: Risk managers (fast checks, many calls)
- Estimated: ~$20-50/month

**FinBERT**:
- Free (open-source)
- Can run locally or on HuggingFace (free tier)

**Total Estimated Cost**: ~$170-350/month

**ROI**: If agents improve Sharpe from 1.0 to 1.5+ on a $100K account, easily worth it.

---

## Conclusion

**Refined approach focuses on**:
1. ✅ Anthropic Claude ecosystem (Opus/Sonnet/Haiku)
2. ✅ FinBERT for financial sentiment
3. ✅ Proof of concept with 5 core agents
4. ✅ Prompt template system for iteration
5. ✅ Realistic 2-4 week timeline

**Next Steps**:
1. Implement proof of concept (5 agents)
2. Create prompt templates
3. Integrate Anthropic API
4. Add FinBERT sentiment
5. Test and iterate

**Status**: Ready to implement proof of concept

---

**Document Version**: 2.1 (Refined)
**Last Updated**: November 30, 2025
**Focus**: Claude-based POC with prompt iteration system
