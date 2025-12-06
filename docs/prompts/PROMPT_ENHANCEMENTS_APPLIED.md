# Prompt Enhancements Applied from Research

**Date**: 2024-12-01
**Based on**: SPECIALIZED_PROMPT_RESEARCH.md findings

---

## Summary

This document tracks all prompt enhancements applied based on specialized research for each agent type. Each agent now has advanced versions (v2.0/v3.0) incorporating proven patterns, techniques, and best practices from industry research.

---

## 1. Supervisor Agent Enhancements

### Version Progression:
- **v1.0**: Basic team coordination and decision-making
- **v1.1**: Added consensus scoring and market regime awareness
- **v2.0**: Multi-agent debate pattern with multi-modal integration
- **v3.0**: Full hierarchical orchestration with chain-of-thought ← NEW

### v3.0 Enhancements from Research:

**Orchestration Architecture** (from AWS/Azure/IBM patterns):
- ✅ Hierarchical orchestration pattern (Top → Middle → Working layers)
- ✅ Group chat communication with structured message passing
- ✅ Clear team leads (Technical Lead, Risk Lead, Strategy Lead)

**Chain-of-Thought Planning** (from AutoGen/LangGraph):
- ✅ Explicit 8-step reasoning process (GATHER → ANALYZE → DEBATE → WEIGH → SYNTHESIZE → RISK → REFLECT → DECIDE)
- ✅ Transparent decision-making with documented thought process
- ✅ Prevents black-box decisions

**Dynamic Agent Weighting** (from multi-agent collaboration research):
- ✅ Weight formula: 40% historical accuracy + 30% confidence + 20% evidence quality + 10% consistency
- ✅ Sharpe ratio tracking over last 50 trades per agent
- ✅ Regime-specific performance tracking (trending vs mean-reverting vs high-vol)

**Memory & Context System** (from LangGraph/CrewAI):
- ✅ Rolling 50-trade history for each agent
- ✅ Decision history with outcome tracking
- ✅ Agent credibility scoring that updates after each trade
- ✅ Context tracking with decision IDs and parent relationships

**Conflict Resolution Protocols** (from academic research):
- ✅ Quantify disagreement level (% split, conviction differences)
- ✅ Quality of evidence examination
- ✅ Historical accuracy weighting
- ✅ Default to conservation when disagreement >40%

**Learning & Adaptation** (from 70% improvement study):
- ✅ Record outcomes (win/loss, return%, drawdown, duration)
- ✅ Update agent weights based on accuracy
- ✅ Analyze what worked/failed
- ✅ Refine decision patterns over time

### Key Improvements:
- Token usage: 1500 → 2000 → 3000 tokens (+100% from v1.0)
- Temperature: 0.7 → 0.6 → 0.5 (more deterministic for production)
- Expected outcome: 70% better goal success vs single-agent (from research)
- Cost: Higher per decision but better risk-adjusted returns expected

---

## 2. Technical Analyst Enhancements

### Version Progression:
- **v1.0**: Basic indicators and trend analysis
- **v2.0**: Multi-timeframe + pattern recognition + divergences
- **v3.0**: 40+ patterns with reliability scoring ← PLANNED

### v2.0 Enhancements Applied:

**Multi-Timeframe Analysis** (from TrendSpider/Tickeron research):
- ✅ Weekly/Daily/Intraday trend identification
- ✅ Alignment scoring (aligned|partially_aligned|conflicting)
- ✅ Higher timeframe = direction, lower timeframe = timing

**Chart Patterns** (from ChartPatterns.ai/Tickeron - 40+ patterns):
- ✅ Continuation: Ascending/Descending Triangles, Flags, Pennants, Cup & Handle
- ✅ Reversal: H&S, Inverse H&S, Double Top/Bottom, Wedges
- ✅ Candlestick: Hammer, Engulfing, Morning/Evening Star, Doji
- ✅ Pattern status tracking (forming|confirmed|broken)
- ✅ Pattern target calculation
- ✅ Probability estimation per pattern

**Pattern Reliability Scoring** (from deep learning research):
- ✅ High reliability (>70% historical success): Strong weight
- ✅ Medium reliability (50-70%): Moderate weight
- ✅ Low reliability (<50%): Weak weight, require confirmation

**Divergence Detection** (from industry best practices):
- ✅ Bullish divergence: Price lower low, indicator higher low (reversal)
- ✅ Bearish divergence: Price higher high, indicator lower high (reversal)
- ✅ Volume divergence: Price rises on declining volume (weak rally)
- ✅ Divergence = strong reversal signal (priority in analysis)

**Specific Trade Setups** (from practitioner guides):
- ✅ Entry price calculation
- ✅ Stop loss levels
- ✅ Profit target 1 and 2
- ✅ Risk/reward ratio
- ✅ Position sizing notes

**Objective Bias-Free Analysis** (from AI tools research):
- ✅ Data-driven decisions without emotional bias
- ✅ Systematic approach to every analysis
- ✅ No confirmation bias - let data speak

### v3.0 Planned Enhancements:

**Expanded Pattern Library** (40+ patterns from Tickeron):
- [ ] Add rare but high-probability patterns (Cup & Handle, Inverse Cup & Handle)
- [ ] Harmonic patterns (Gartley, Butterfly, Bat, Crab)
- [ ] Elliott Wave patterns
- [ ] Fibonacci extension/retracement levels
- [ ] Volume-based patterns (Volume climax, accumulation/distribution)

**Pattern Invalidation Conditions**:
- [ ] Specify exact price levels where pattern breaks down
- [ ] Time limits for pattern completion
- [ ] Volume requirements for confirmation

**Automated Pattern Recognition Confidence**:
- [ ] Machine learning scores for pattern matches (if available)
- [ ] Historical success rate for each pattern on this symbol
- [ ] Market regime suitability (trending vs consolidating)

---

## 3. Sentiment Analyst Enhancements

### Version Progression:
- **v1.0**: Basic FinBERT + news + social sentiment
- **v2.0**: Behavioral finance + 20% accuracy improvement techniques ← PLANNED

### v2.0 Planned Enhancements from Research:

**Behavioral Finance Perspective** (from ACM/ResearchGate research):
- [ ] Investor psychology analysis (greed/fear indicators)
- [ ] Herding behavior detection
- [ ] Sentiment-driven decision-making patterns
- [ ] Emotional tone classification (optimistic/pessimistic/neutral)
- [ ] Market psychology state (euphoria, panic, complacency)

**20% Accuracy Improvement Techniques** (from prediction research):
- [ ] Incorporate user sentiment into forecasts
- [ ] Weight recent sentiment more than old sentiment
- [ ] Sentiment velocity tracking (improving/deteriorating/stable)
- [ ] Sentiment-price divergence detection (price up, sentiment down = warning)

**Noise Filtering** (from data quality research):
- [ ] Filter social media noise (bots, spam, irrelevant mentions)
- [ ] Quality scoring for news sources (tier 1 vs tier 3)
- [ ] Recency weighting (last 24hrs > last week > last month)
- [ ] Remove duplicate stories (count unique narratives, not copies)

**Real-Time vs Historical Comparison**:
- [ ] Compare current sentiment to 1-week/1-month/1-year baseline
- [ ] Identify sentiment shifts (sudden spikes = event-driven)
- [ ] Contrarian signals (extreme sentiment = reversal opportunity)

**Multi-Source Aggregation** (from QuantifiedStrategies research):
- [ ] Weighted aggregation: FinBERT (40%) + News (30%) + Social (20%) + Analyst (10%)
- [ ] Source reliability scoring
- [ ] Conflicting source resolution (when sources disagree)
- [ ] Consensus strength measurement

**Contrarian Signal Detection** (from behavioral finance):
- [ ] Extreme bullish sentiment (>0.90) = sell signal (crowd wrong at extremes)
- [ ] Extreme bearish sentiment (<-0.90) = buy signal (panic creates opportunity)
- [ ] Greed/fear index integration
- [ ] VIX sentiment correlation (high VIX + bearish sentiment = bottoming signal)

---

## 4. Conservative Trader Enhancements

### Version Progression:
- **v1.0**: Basic conservative strategy guidelines
- **v2.0**: Institutional trader persona with explicit risk parameters ← PLANNED

### v2.0 Planned Enhancements from Research:

**Institutional Trader Persona** (from MQL5/practitioner guides):
- [ ] "You are a conservative institutional trader with 15 years experience managing pension fund assets"
- [ ] "Your primary mandate is capital preservation with steady returns"
- [ ] "You answer to risk committees and must justify every trade"
- [ ] Risk-averse mindset (protect capital > aggressive growth)

**Explicit Risk Parameters** (from conservative trading research):
- [ ] Max risk per trade: 0.5-1.0% (very conservative)
- [ ] Daily drawdown limit: 2%
- [ ] Position size: 1-3% per position
- [ ] Win probability requirement: >65% minimum
- [ ] Must have 2:1 risk/reward minimum

**Options Strategy Preferences** (from options trading guides):
- [ ] Low volatility: Sell covered calls, cash-secured puts (income generation)
- [ ] High volatility: Buy spreads, defined-risk strategies only
- [ ] Favor: Butterflies, iron condors, credit spreads
- [ ] Avoid: Naked options, undefined-risk strategies, aggressive speculation
- [ ] Expiration preference: 30-60 days for theta decay

**Position Sizing Formulas** (from risk management research):
- [ ] Fixed fractional: Risk exactly 0.5-1% based on stop loss distance
- [ ] Volatility-scaled: Reduce size when ATR high
- [ ] Kelly Criterion: Conservative (use 25-50% Kelly)
- [ ] Never risk more than planned, even if conviction is high

**Context-Rich Prompt Structure** (from prompt engineering best practices):
- [ ] Require: Current market regime (bull/bear/sideways)
- [ ] Require: IV percentile (high/medium/low)
- [ ] Require: Underlying trend (confirm with technical analyst)
- [ ] Require: Portfolio exposure (avoid overconcentration)
- [ ] Require: Time horizon (match strategy to timeframe)

**Strategy Selection Logic**:
- [ ] IF low_vol AND uptrend THEN covered_calls
- [ ] IF high_vol AND neutral THEN iron_condor
- [ ] IF moderate_vol AND bullish THEN bull_call_spread
- [ ] IF high_uncertainty THEN cash (preserve capital)

---

## 5. Moderate Trader Enhancements

### Version Progression:
- **v1.0**: Basic moderate strategy guidelines
- **v2.0**: Balanced growth trader persona ← PLANNED

### v2.0 Planned Enhancements:

**Balanced Trader Persona**:
- [ ] "You balance growth and protection, seeking consistent returns"
- [ ] "You take calculated risks when probability favors reward"
- [ ] Max risk per trade: 1-2%
- [ ] Win probability requirement: >60%
- [ ] Risk/reward: 1.5:1 minimum

**Strategy Flexibility**:
- [ ] More aggressive than Conservative, more cautious than Aggressive
- [ ] Willing to take directional bets with defined risk
- [ ] Uses leverage moderately (spreads with leverage, not naked)
- [ ] Adapts to market conditions (more aggressive in trending, more defensive in choppy)

---

## 6. Aggressive Trader Enhancements

### Version Progression:
- **v1.0**: Basic aggressive strategy guidelines
- **v2.0**: Growth-focused trader persona ← PLANNED

### v2.0 Planned Enhancements:

**Growth-Focused Persona**:
- [ ] "You prioritize capital growth and high-conviction opportunities"
- [ ] "You're willing to accept higher volatility for higher returns"
- [ ] Max risk per trade: 2-3%
- [ ] Win probability requirement: >55% (lower threshold)
- [ ] Risk/reward: 2:1 minimum (must have big upside)

**Aggressive Strategies**:
- [ ] Favor: Directional spreads, naked options (if approved), straddles/strangles
- [ ] Will use full position sizes (up to 30%) on high-conviction setups
- [ ] Shorter duration trades (weekly options acceptable)
- [ ] More leverage tolerance

**High Conviction Logic**:
- [ ] Requires >0.85 confidence from supervisor
- [ ] Requires technical + sentiment + fundamental alignment
- [ ] Requires risk manager approval (still subject to veto)

---

## 7. Position Risk Manager Enhancements

### Version Progression:
- **v1.0**: Basic position limits
- **v2.0**: Volatility-based stops + liquidity checks ← PLANNED

### v2.0 Planned Enhancements from Research:

**Volatility-Based Stop Loss** (from ATR research):
- [ ] Use ATR (Average True Range) for stop placement
- [ ] Stop distance = 1.5-2.0 x ATR (not too tight, not too loose)
- [ ] Adjust for volatility regime (wider stops in high vol)
- [ ] Trail stops as position becomes profitable

**Liquidity Checks** (from risk management guides):
- [ ] Bid-ask spread: Must be <15% of mid price
- [ ] Open interest: Minimum 100 contracts preferred
- [ ] Volume: Daily volume >50 contracts preferred
- [ ] Market maker presence (tight spread = good)

**Circuit Breaker Awareness** (from regulatory research):
- [ ] Level 1 (7% loss): Flag warning, tighten future approvals
- [ ] Level 2 (13% loss): Elevated caution, reduce new position sizes 50%
- [ ] Level 3 (20% loss): No new positions, focus on risk reduction

**Position Limit Enforcement**:
- [ ] 25% max position size (ABSOLUTE VETO)
- [ ] 5% max risk per trade (ABSOLUTE VETO)
- [ ] 10 max concurrent positions (ABSOLUTE VETO)
- [ ] 40% min win probability (ABSOLUTE VETO)
- [ ] <15% bid-ask spread (ABSOLUTE VETO)

**CANNOT BE OVERRIDDEN**:
- [ ] If any limit violated → REJECT (no exceptions)
- [ ] Supervisor cannot override
- [ ] This is a hard safety constraint

---

## 8. Portfolio Risk Manager Enhancements

### Version Progression:
- **v1.0**: Basic portfolio limits
- **v2.0**: VIX-based dynamic limits + circuit breakers ← ALREADY IMPLEMENTED in v2.0

### v2.0 Enhancements Applied:

**VIX-Based Dynamic Limits** (from risk management research):
- ✅ VIX <15: position_multiplier = 1.2 (can increase 20%)
- ✅ VIX 15-25: position_multiplier = 1.0 (standard limits)
- ✅ VIX 25-35: position_multiplier = 0.8 (reduce 20%)
- ✅ VIX 35-50: position_multiplier = 0.5 (reduce 50%)
- ✅ VIX >50: position_multiplier = 0.0 (halt all new trades)

**Drawdown Limits**:
- ✅ Max drawdown: 10% (from peak equity)
- ✅ Daily loss limit: 3%
- ✅ Weekly loss limit: 7%

**Correlation Analysis**:
- ✅ Monitor sector concentration
- ✅ Flag correlation breakdown (diversification failure)
- ✅ Warn when multiple positions move together (hidden correlation risk)

**Stress Testing**:
- ✅ Simulate -10% market move
- ✅ Simulate VIX spike to 50
- ✅ Simulate worst-case scenario for all positions
- ✅ Reject trades if stress test shows >15% portfolio loss

**Risk-Adjusted Metrics**:
- ✅ Sharpe ratio tracking
- ✅ Sortino ratio (downside risk focus)
- ✅ Max drawdown monitoring
- ✅ Win rate and profit factor

---

## 9. Circuit Breaker Manager Enhancements

### Version Progression:
- **v1.0**: Basic trading halts
- **v2.0**: 3-level circuit breaker system ← PLANNED

### v2.0 Planned Enhancements from Research:

**Regulatory Circuit Breakers** (from market structure research):
- [ ] Level 1 (7% daily loss):
  * Trigger: Portfolio down 7% from opening value
  * Action: Warning flag, reduce new positions 50%
  * Duration: Until market close or recovery to -5%

- [ ] Level 2 (13% daily loss):
  * Trigger: Portfolio down 13% from opening value
  * Action: Halt all new trades, manage existing positions only
  * Duration: Requires human approval to resume trading

- [ ] Level 3 (20% daily loss):
  * Trigger: Portfolio down 20% from opening value
  * Action: Full trading halt, consider emergency liquidation
  * Duration: Trading suspended until next day, requires executive approval

**Consecutive Loss Tracking**:
- [ ] Track consecutive losing trades
- [ ] After 5 consecutive losses: Reduce all new position sizes 30%
- [ ] After 7 consecutive losses: Halt trading, review system
- [ ] Reset counter after winning trade

**Volatility Circuit Breakers**:
- [ ] If VIX spikes >50: Halt new trades (too much systemic risk)
- [ ] If sector volatility >3x normal: Halt new trades in that sector
- [ ] If individual stock halted: Immediately close related options positions

**Manual Override Requirements**:
- [ ] Circuit breaker halt requires manual reset
- [ ] Must document reason for halt
- [ ] Must analyze what went wrong
- [ ] Must get approval from risk committee/human oversight
- [ ] Prevents runaway losses from automated trading

---

## 10. Cross-Cutting Enhancements (All Agents)

### Applied to ALL agent prompts:

**Structured Decision Frameworks**:
- ✅ Step-by-step reasoning processes
- ✅ Clear decision criteria at each step
- ✅ Explicit output formats (JSON)
- ✅ Confidence scoring (0.0-1.0)

**Context-Rich Prompts**:
- ✅ Require comprehensive input data
- ✅ Specify all necessary parameters
- ✅ Include market context (VIX, regime, liquidity)
- ✅ Historical performance reference when available

**Risk-First Approach**:
- ✅ Always consider downside before upside
- ✅ Explicit risk limits and constraints
- ✅ Position sizing calculations
- ✅ Stop loss and exit criteria

**Multi-Modal Integration**:
- ✅ Combine technical + sentiment + fundamental signals
- ✅ Cross-validate signals across sources
- ✅ Alignment scoring (how well signals agree)
- ✅ Conflicting signal resolution protocols

**Continuous Learning**:
- ✅ Reference historical decisions
- ✅ Learn from past mistakes
- ✅ Performance tracking metrics
- ✅ Adaptive limit adjustment

**Bias Mitigation**:
- ✅ Objective, data-driven analysis
- ✅ Avoid emotional language
- ✅ Contrarian signal detection
- ✅ Devil's advocate reasoning (bull vs bear cases)

**Specific Actionability**:
- ✅ Concrete entry/exit prices
- ✅ Stop loss levels
- ✅ Profit targets
- ✅ Position sizing recommendations
- ✅ Risk/reward ratios

---

## Cost and Performance Impact

### Token Usage Changes:

| Agent | v1.0 | v2.0 | v3.0 | Change |
|-------|------|------|------|--------|
| Supervisor | 1500 | 2000 | 3000 | +100% |
| TechnicalAnalyst | 1000 | 1500 | TBD | +50% |
| SentimentAnalyst | 1000 | TBD | TBD | TBD |
| Traders | 1000 | TBD | TBD | TBD |
| Risk Managers | 800 | 1000 | TBD | +25% |

### Expected ROI:

**From Research Findings**:
- Multi-agent collaboration: +70% goal success vs single-agent
- Sentiment integration: +20% prediction accuracy
- Pattern recognition: Higher win rates with proper pattern matching
- VIX-based sizing: Reduced drawdowns in high volatility

**Estimated Improvements**:
- Sharpe ratio: Expect +0.3 to +0.5 improvement
- Win rate: Expect +5-10% improvement
- Max drawdown: Expect -20-30% reduction
- Risk-adjusted returns: Net positive despite higher token costs

**Cost Analysis**:
- Higher per-decision costs (+50-100% token usage)
- But fewer bad decisions → Better overall profitability
- Estimated payback: 2-3 months of live trading
- Long-term: Significantly better risk-adjusted returns

---

## Implementation Status

### ✅ Completed:
- [x] Supervisor v3.0 with full orchestration
- [x] TechnicalAnalyst v2.0 with patterns and divergences
- [x] Portfolio Risk Manager v2.0 with VIX-based limits
- [x] All base agent implementations

### 🚧 In Progress:
- [ ] TechnicalAnalyst v3.0 with 40+ patterns
- [ ] SentimentAnalyst v2.0 with behavioral finance
- [ ] All Trader agents v2.0
- [ ] Position Risk Manager v2.0
- [ ] Circuit Breaker Manager v2.0

### 📋 Planned:
- [ ] Testing all enhanced prompts
- [ ] A/B testing v1.0 vs v2.0 vs v3.0
- [ ] Performance monitoring
- [ ] Continuous refinement based on results

---

## Next Steps

1. **Complete Implementation**: Finish all v2.0/v3.0 prompts for remaining agents
2. **Testing**: Create test scenarios to validate enhancements
3. **Documentation**: Update usage guides with new prompt versions
4. **A/B Testing**: Compare v1.0 vs v2.0 vs v3.0 performance
5. **Monitoring**: Track key metrics (Sharpe, win rate, drawdown)
6. **Iteration**: Refine prompts based on actual trading results

---

**References**: See SPECIALIZED_PROMPT_RESEARCH.md for detailed research sources and findings.
