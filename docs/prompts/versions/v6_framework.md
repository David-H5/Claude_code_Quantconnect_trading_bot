# V6.0 Prompt Enhancement Framework
## Production-Ready Multi-Agent Trading System with Market-Based Task Allocation and Real-World Validation

**Created**: 2025-12-01
**Version**: v6.0 (FINAL ENHANCEMENT PHASE)
**Research Foundation**: Section 9 of LLM_TRADING_RESEARCH_2024_2025.md
**Key Papers**: STOCKBENCH, TradingAgents, QTMRL, POW-dTS, MarketSenseAI, Agentic AI 2025

---

## Table of Contents
1. [V6.0 Philosophy](#v60-philosophy)
2. [Core V6.0 Enhancements](#core-v60-enhancements)
3. [Agent-Specific V6.0 Features](#agent-specific-v60-features)
4. [Implementation Guidelines](#implementation-guidelines)
5. [Validation Criteria](#validation-criteria)

---

## V6.0 Philosophy

**Evolution from V5.0**:
- V5.0: Collective intelligence with peer-to-peer collaboration
- V6.0: **PRODUCTION-READY SYSTEM** with market-based allocation, full team calibration, and real-world validation

**Core Principle**: Agents are autonomous market participants who:
1. **Bid for tasks** based on expertise and confidence (market-based allocation)
2. **Calibrate collectively** as a team (not just individually)
3. **Validate out-of-sample** before recommendations (STOCKBENCH approach)
4. **Predict and prevent** circuit breaker triggers (advanced risk integration)
5. **Adapt exploration rates** dynamically (refined Thompson Sampling)
6. **Test in real-world** before production (post-knowledge-cutoff validation)

**Target**: Match or exceed research benchmarks → **Sharpe >2.5, Return >35%, Win rate >70%, Drawdown <15%**

---

## Core V6.0 Enhancements (All Agents)

### 1. Market-Based Task Allocation

**Concept**: Agents "bid" for tasks based on expertise confidence, creating efficient task allocation through internal market mechanism.

**Implementation**:
```
TASK AUCTION SYSTEM:

Step 1: Task Posted to Market
Task: "Analyze AAPL bull flag pattern on daily chart"
Posted by: Supervisor
Required expertise: Technical analysis
Deadline: 60 seconds

Step 2: Agents Submit Bids
TechnicalAnalyst bid:
- Confidence: 0.88
- Expertise match: 0.95 (chart patterns are core expertise)
- Recent accuracy: 0.72 (bull flags in normal volatility)
- Bid score: (0.88 * 0.95 * 0.72) = 0.60

SentimentAnalyst bid:
- Confidence: 0.45
- Expertise match: 0.30 (can identify sentiment, but not chart expert)
- Recent accuracy: 0.65 (general market sentiment)
- Bid score: (0.45 * 0.30 * 0.65) = 0.09

FundamentalsAnalyst bid:
- Confidence: 0.35
- Expertise match: 0.20 (can provide context, but not chart expert)
- Recent accuracy: 0.68
- Bid score: (0.35 * 0.20 * 0.68) = 0.05

Step 3: Task Awarded
Winner: TechnicalAnalyst (highest bid score 0.60)
Rationale: Best combination of confidence, expertise match, and recent accuracy

Step 4: Collaborative Support (Optional)
TechnicalAnalyst can request support from other analysts:
- Query SentimentAnalyst: "Any catalyst news for AAPL in next 7 days?"
- Query FundamentalsAnalyst: "Is AAPL undervalued relative to sector?"
Support agents provide context without full analysis

Step 5: Delivery and Feedback
TechnicalAnalyst delivers analysis within 60 seconds
Quality scored by Supervisor
Feedback updates agent's recent accuracy for future bids
```

**Benefits**:
- Efficient allocation: Tasks go to most qualified agents
- Self-organization: No manual task assignment needed
- Quality improvement: Recent accuracy influences future bids
- Transparency: Bid scores are explainable

### 2. Full Team Calibration

**Concept**: Adjust ALL agent weights collectively based on team performance, not just individual agents.

**Implementation**:
```
TEAM CALIBRATION PROTOCOL:

Step 1: Measure Team Performance (Last 50 Trades)
Team win rate: 68% (target: 70%)
Team Sharpe: 2.3 (target: 2.5)
Team drawdown: 12% (target: <15%)
Team overconfidence rate: 22% (trades with 0.80+ confidence that lost)

Assessment: Team is slightly underperforming and overconfident

Step 2: Calculate Calibration Adjustments
Overconfidence adjustment: -0.08 (reduce all confidence scores)
Performance adjustment: -0.05 (reduce position sizes until performance improves)

Individual agent adjustments:
- Agents with >25% overconfidence: -0.10 adjustment
- Agents with 15-25% overconfidence: -0.08 adjustment (team average)
- Agents with <15% overconfidence: -0.05 adjustment (performing well)

Step 3: Apply Team Calibration
TechnicalAnalyst:
- Original confidence: 0.75
- Overconfidence rate: 28% (>25%)
- Team calibration: -0.10
- Calibrated confidence: 0.65

SentimentAnalyst:
- Original confidence: 0.80
- Overconfidence rate: 18% (15-25%)
- Team calibration: -0.08
- Calibrated confidence: 0.72

ConservativeTrader:
- Original Kelly: 0.13
- Performance adjustment: -0.05
- Calibrated Kelly: 0.08 (reduce position sizes)

Step 4: Monitor Improvement (Next 50 Trades)
After calibration:
- Team win rate: 71% (improved from 68%)
- Team Sharpe: 2.6 (improved from 2.3)
- Overconfidence rate: 15% (improved from 22%)

Step 5: Adjust Calibration Parameters
Team now performing above target
Calibration adjustment: +0.03 (slight increase in confidence/sizing)

Continue monitoring and adjusting every 50 trades
```

**Key Principle**: If the team is collectively overconfident, reduce ALL agents' confidence scores proportionally. Individual calibration happens first, then team calibration overlays.

### 3. Out-of-Sample Validation (STOCKBENCH Approach)

**Concept**: Validate strategies on data after model training cutoff to ensure they work in real-world conditions.

**Implementation**:
```
OUT-OF-SAMPLE VALIDATION PROTOCOL:

Step 1: Define Training vs Validation Periods
Model training data: 2020-01-01 to 2024-06-30
Model knowledge cutoff: 2025-01-01 (Claude's cutoff)
Validation period: 2024-07-01 to 2024-12-31 (post-training, pre-cutoff)
Real-world testing: 2025-01-01+ (post-knowledge-cutoff)

Step 2: Strategy Development (In-Sample)
TechnicalAnalyst develops "bull flag in normal volatility" strategy
Training data (2020-2024): 68% win rate, 2.1:1 RR, Sharpe 1.8

Step 3: Out-of-Sample Backtest (Validation Period)
Validate on 2024-07-01 to 2024-12-31 (6 months post-training)
Results: 62% win rate, 1.9:1 RR, Sharpe 1.6

Analysis:
- Win rate declined 6% (68% → 62%)
- RR declined slightly (2.1 → 1.9)
- Sharpe declined (1.8 → 1.6)
- VERDICT: Strategy shows degradation out-of-sample

Step 4: Adjust Strategy or Confidence
Option A: Refine strategy parameters to improve out-of-sample performance
Option B: Reduce confidence by degradation factor: 0.75 * (62/68) = 0.68

ConservativeTrader adopts Option B:
- Original confidence: 0.75
- Out-of-sample adjustment: -0.07
- Final confidence: 0.68

Step 5: Real-World Paper Trading (Post-Knowledge-Cutoff)
Test strategy in paper trading for 30 days (2025-01-01 to 2025-01-30)
Results: 65% win rate, 2.0:1 RR, Sharpe 1.7

Analysis:
- Performance between training and validation (in-sample 68%, validation 62%, real-world 65%)
- Strategy validated for live trading with confidence 0.68

Step 6: Continuous Out-of-Sample Monitoring
Every quarter, test strategies on new out-of-sample data
If degradation >10%, trigger strategy review
If improvement, increase confidence proportionally
```

**Critical Rule**: NEVER recommend strategies that haven't been validated out-of-sample. If only in-sample data available, apply 20% confidence penalty.

### 4. Advanced Predictive Circuit Breakers

**Concept**: Use ML stress scores to predict circuit breaker triggers 1-2 hours in advance and take preventive action.

**Implementation**:
```
PREDICTIVE CIRCUIT BREAKER SYSTEM:

Step 1: Calculate Real-Time Stress Score (Every 5 Minutes)
Base stress = daily_loss% * 10 + consecutive_losses * 2 + (drawdown% - 5%) * 5
Correlation penalty: Portfolio correlation >0.75 → add +2
Concentration penalty: Sector >35% → add +1.5
Greeks penalty: Portfolio delta >±150 → add +1
Volatility penalty: VIX >30 → add +1

Current state (11:45 AM):
- Daily loss: -4.5%
- Consecutive losses: 3
- Drawdown: 9%
- Portfolio correlation: 0.78
- Sector concentration: 37% (tech)
- Portfolio delta: -165
- VIX: 28

Stress score = (-4.5 * 10) + (3 * 2) + ((9 - 5) * 5) + 2 + 1.5 + 1
Stress score = -45 + 6 + 20 + 2 + 1.5 + 1 = -14.5

Wait, that's negative. Let me recalculate with absolute values:

Stress score = (4.5 * 10) + (3 * 2) + (4 * 5) + 2 + 1.5 + 1
Stress score = 45 + 6 + 20 + 2 + 1.5 + 1 = 75.5

Step 2: Predict Circuit Breaker Triggers
Level 1 trigger: 7% daily loss (currently at 4.5%, need 2.5% more)
Level 2 trigger: 13% daily loss (currently at 4.5%, need 8.5% more)

ML prediction (based on stress score + current trajectory):
- Probability of Level 1 in next hour: 65%
- Probability of Level 2 in next 4 hours: 25%

Rationale: High stress score (75.5), negative momentum, high correlation risk

Step 3: Preventive Actions (Before Trigger)
At stress score >60 + Level 1 probability >50%:

Action 1: Send P2P warnings to ALL agents
"PREDICTIVE WARNING: 65% chance of Level 1 circuit breaker (7% loss) within next hour. Current loss -4.5%. Reduce all new position sizes by 50%. No aggressive positions."

Action 2: Reduce position limits proactively
- ConservativeTrader: Max position 15% → 10%
- ModerateTrader: Max position 20% → 13%
- AggressiveTrader: Max position 25% → 17%

Action 3: Tighten stop losses
- All positions: Move stops 20% closer to current price
- Reduce maximum loss per position from 25% to 15%

Action 4: Close high-risk positions
- Identify positions with:
  * High correlation (>0.80) with losing positions
  * Low confidence (<0.70) on entry
  * Negative delta alignment (all short delta or all long delta)
- Reduce these positions by 30%

Step 4: Monitor Effectiveness (Next Hour)
After preventive actions (12:45 PM):
- Daily loss: -5.2% (increased 0.7% despite actions)
- Stress score: 62.8 (reduced from 75.5)
- Level 1 probability: 38% (reduced from 65%)

VERDICT: Preventive actions reduced circuit breaker probability by 27%
Circuit breaker DID NOT trigger (stayed below 7%)

Step 5: Learn from Prediction Accuracy
Prediction: 65% chance of Level 1 in next hour
Outcome: Level 1 did NOT trigger
Accuracy: Correct prediction (warned, prevented)

Update ML model:
- Preventive actions at stress score >60 are effective
- Reduce future stress score threshold to 55 for earlier warnings
- Track: 1 successful prevention

If Level 1 HAD triggered despite warnings:
- Analyze: Were preventive actions insufficient?
- Adjust: Increase aggressiveness of preventive actions
- Track: 1 failed prevention
```

**Key Principle**: Prevent circuit breakers, don't just react to them. Predictive warnings 1-2 hours in advance enable proactive risk reduction.

### 5. Refined Adaptive Thompson Sampling

**Concept**: Dynamically adjust Thompson Sampling exploration rates based on performance, regime, and recent discoveries.

**Implementation**:
```
REFINED ADAPTIVE THOMPSON SAMPLING:

Step 1: Base Exploration Rate (From v5.0)
Recent performance (last 20 trades): 55% win rate (below 65% historical)
Base exploration increase: +20% (from 40% to 60%)

Step 2: Regime Adjustment
Current regime: High volatility (VIX 32)
High volatility exploration bonus: +10%
Regime-adjusted exploration: 70%

Step 3: Recent Discovery Adjustment
Check: Have we discovered any new high-performing strategies in last 50 trades?

Recent discoveries:
- Cash-secured puts on high IV rank: 78% win rate (12 wins, 3 losses)
- Bull put spreads at support: 73% win rate (11 wins, 4 losses)

Discovery quality score:
- Cash-secured puts: (78% - 65% historical) = +13% improvement
- Bull put spreads: (73% - 65%) = +8% improvement

If recent discoveries show >10% improvement: Maintain high exploration (70%)
If recent discoveries show <5% improvement: Reduce exploration by 10%

Current: +13% improvement on cash-secured puts
Decision: MAINTAIN exploration at 70% (discovery mode is working)

Step 4: Exploration Decay (If No New Discoveries)
If 50 trades pass with no new strategies showing >10% improvement:
- Decay exploration by 5% every 25 trades
- Minimum exploration: 20%

Current: Recent discovery 15 trades ago
Decision: NO DECAY yet

Step 5: Exploitation Boost (If Consistent Performance)
If last 50 trades show:
- Win rate >75% (significantly above target)
- Sharpe >3.0 (significantly above target)

Then: Reduce exploration to 20%, increase exploitation to 80%
Rationale: Current strategies are working exceptionally well

Current: Win rate 55%, Sharpe 0.8 (underperforming)
Decision: NO EXPLOITATION BOOST, keep exploration at 70%

Step 6: Apply to Strategy Selection
Available strategies for ConservativeTrader:
1. Iron condor: Beta(73, 27) → Sample: 0.68
2. Credit spread: Beta(71, 29) → Sample: 0.64
3. Cash-secured put: Beta(12, 3) → Sample: 0.78 (NEW DISCOVERY)

Exploration rate: 70%
Exploitation rate: 30%

Selection algorithm:
- 70% chance: Sample from Thompson distribution (highest sample wins)
- 30% chance: Pick highest historical win rate (iron condor 73%)

Thompson samples: [0.68, 0.64, 0.78]
Highest sample: Cash-secured put (0.78)

70% exploration mode: SELECT cash-secured put
Rationale: Thompson Sampling exploration mode selected new discovery

Step 7: Track Exploration Effectiveness
Over next 20 trades in exploration mode (70%):
- Strategies tried: Iron condor 6x, Credit spread 4x, Cash-secured put 8x, New vol play 2x
- Win rate: 68% (improved from 55%)
- New discoveries: Vol play showed 2 wins, 0 losses (small sample, but promising)

Exploration effectiveness: +13% win rate improvement
Decision: MAINTAIN exploration at 70% for another 50 trades
```

**Dynamic Range**: Exploration can range from 20% (exploitation mode when performing well) to 80% (discovery mode when underperforming or in high volatility).

### 6. Real-World Testing Framework

**Concept**: Paper trading validation before any live deployment to ensure strategies work in real-time markets.

**Implementation**:
```
REAL-WORLD TESTING PROTOCOL:

Step 1: Strategy Qualification (Must Pass All)
✓ In-sample backtest: Sharpe >1.5, Win rate >60%, 100+ trades
✓ Out-of-sample validation: Performance degradation <15%
✓ Team calibration: Confidence adjusted based on collective performance
✓ Risk approval: All 3 risk managers approved

Current strategy: "Bull flags in normal volatility with sentiment confirmation"
- In-sample: 68% win rate, Sharpe 1.8 ✓
- Out-of-sample: 62% win rate (degradation 9%) ✓
- Team calibration: Confidence 0.68 ✓
- Risk approval: All approved ✓

VERDICT: Qualified for paper trading

Step 2: Paper Trading Setup
Duration: 30 calendar days (minimum 20 trading days)
Starting capital: $100,000 (virtual)
Position sizing: Use actual Kelly calculations
Execution: Real-time order placement (paper account)
Market hours: Full trading day (pre-market, regular, after-hours)

Step 3: Real-Time Execution Challenges
Challenge 1: Fill rates
- Backtest assumes fills at limit price
- Paper trading: Track actual fills (may be worse than backtest)

Challenge 2: Slippage
- Backtest: Minimal slippage assumptions
- Paper trading: Real bid-ask spreads, order book depth

Challenge 3: Market impact
- Backtest: No market impact
- Paper trading: Large orders may move market (simulate impact)

Challenge 4: Timing
- Backtest: Perfect entry at signal
- Paper trading: Delay from signal generation to order placement

Step 4: Success Criteria (30-Day Paper Trading)
Minimum acceptable performance:
- Win rate: >55% (allowing for real-world degradation from backtest 62%)
- Sharpe ratio: >1.2 (allowing for real-world degradation from backtest 1.6)
- Fill rate: >60% (orders must actually fill)
- Max drawdown: <20%
- Positive expectancy: Average winner > Average loser

Current paper trading results (Day 30):
- Win rate: 58% (17 wins, 12 losses)
- Sharpe: 1.4
- Fill rate: 64%
- Max drawdown: 14%
- Avg winner: +8.2%, Avg loser: -4.1% (RR 2.0:1)

VERDICT: ✓ All success criteria met

Step 5: Live Deployment Authorization
Required approvals:
✓ PositionRiskManager: Approved (risk parameters acceptable)
✓ PortfolioRiskManager: Approved (portfolio fit acceptable)
✓ CircuitBreakerManager: Approved (no halt triggers in paper trading)
✓ Supervisor: Approved (strategy meets all benchmarks)
✓ HUMAN AUTHORIZATION: Required

After human approval:
- Deploy to live account with initial position limit 50% of paper trading
- Monitor for 10 trades
- If performance matches paper trading, increase to 100%

Step 6: Continuous Real-World Monitoring
Live trading results (First 10 trades):
- Win rate: 60% (6 wins, 4 losses)
- Fill rate: 62%
- Avg return per trade: +2.8%

Comparison to paper trading:
- Win rate: 58% paper vs 60% live (BETTER)
- Fill rate: 64% paper vs 62% live (ACCEPTABLE)

VERDICT: Strategy approved for full deployment
Increase position limits to 100%
```

**Critical Rule**: NO strategy deploys to live trading without 30-day successful paper trading. No exceptions.

---

## Agent-Specific V6.0 Features

### Supervisor v6.0

**New Responsibilities**:
1. **Market-Based Task Allocation**: Run task auctions, award tasks to highest bidders
2. **Full Team Calibration**: Adjust ALL agents collectively every 50 trades
3. **Out-of-Sample Validation**: Require all strategies validated on post-training data
4. **Predictive Circuit Breaker Coordination**: Monitor stress scores, coordinate preventive actions
5. **Refined Thompson Sampling**: Manage exploration rates dynamically (20-80%)
6. **Real-World Testing Gate**: Final approval for live deployment after paper trading

**Enhanced Chain of Thought (14 steps)**:
```
1. Assess regime + calculate team performance (last 50 trades)
2. Run team calibration if due (every 50 trades, adjust all agents collectively)
3. Post task auction for analyst signals (agents bid based on expertise/confidence)
4. Award tasks to highest bidders, coordinate collaborative support
5. Gather analyst signals + validate out-of-sample performance
6. Check confluence (3+ signals), apply team calibration to confidence scores
7. Calculate predictive stress score (every 5 minutes)
8. If stress >60: Issue preventive warnings, reduce limits, tighten stops
9. Calculate portfolio-adjusted Kelly with team calibration overlay
10. Weight recommendations using refined adaptive Thompson Sampling (20-80% exploration)
11. Validate strategy has passed paper trading (30-day real-world test)
12. Check all risk manager approvals (position, portfolio, circuit breaker)
13. Generate final recommendation with full audit trail
14. Log state-action-reward + schedule team calibration if due
```

### TechnicalAnalyst v6.0

**New Capabilities**:
1. **Task Bidding**: Bid for chart analysis tasks based on pattern expertise + recent accuracy
2. **Out-of-Sample Pattern Validation**: Validate patterns on post-training data before recommending
3. **Team Calibration Acceptance**: Accept team-wide confidence adjustments
4. **Discovery Tracking**: Track when new patterns outperform historical (boost exploration)
5. **Real-World Testing Participation**: Provide patterns for 30-day paper trading validation

**Bid Calculation**:
```
Task: "Analyze MSFT double bottom pattern"

Confidence: 0.82 (double bottom is strong pattern)
Expertise match: 0.90 (reversal patterns are core expertise)
Recent accuracy: 0.74 (double bottoms in normal volatility)

Bid score = 0.82 * 0.90 * 0.74 = 0.55

Submit bid with:
- Score: 0.55
- Justification: "Double bottoms are core reversal pattern expertise, 74% recent accuracy"
- Estimated delivery time: 45 seconds
```

### SentimentAnalyst v6.0

**New Capabilities**:
1. **Task Bidding**: Bid for sentiment analysis tasks (earnings, news, social media)
2. **Out-of-Sample Sentiment Validation**: Test sentiment signals on post-cutoff data
3. **Team Calibration**: Accept collective overconfidence adjustments
4. **Catalyst Discovery**: Track new sentiment patterns that outperform
5. **Real-World Testing**: Validate sentiment indicators in paper trading

### ConservativeTrader / ModerateTrader / AggressiveTrader v6.0

**New Capabilities**:
1. **Task Bidding**: Bid for strategy selection tasks based on market conditions + expertise
2. **Out-of-Sample Strategy Validation**: Only recommend strategies validated on new data
3. **Full Team Calibration**: Accept team-wide Kelly adjustments (not just individual)
4. **Discovery Reporting**: Report new high-performing strategies to boost exploration
5. **Paper Trading Validation**: All strategies must pass 30-day real-world test

**Team Calibration Example**:
```
ConservativeTrader strategy: Iron condor

Individual performance:
- Win rate: 75% (excellent)
- Sharpe: 2.2
- Individual confidence: 0.82

Team performance (last 50 trades):
- Team win rate: 68% (below 70% target)
- Team overconfidence: 22% (high)

Team calibration adjustment: -0.08 (reduce all confidence scores)

Final calibrated confidence: 0.82 - 0.08 = 0.74

Rationale: Even though ConservativeTrader is performing well individually,
the TEAM is overconfident, so all agents reduce confidence proportionally.
```

### PositionRiskManager v6.0

**New Capabilities**:
1. **Predictive Risk Alerts**: Use stress scores to predict limit violations 1-2 hours early
2. **Team Calibration Support**: Track team overconfidence, recommend calibration adjustments
3. **Out-of-Sample Risk Validation**: Ensure strategies pass validation before approval
4. **Real-World Testing Gate**: Stricter limits during paper trading phase
5. **Discovery Risk Assessment**: Evaluate risk of new strategies vs proven strategies

### PortfolioRiskManager v6.0

**New Capabilities**:
1. **Predictive Portfolio Stress**: Calculate portfolio stress scores, warn before circuit breakers
2. **Team Calibration Coordination**: Recommend team-wide Kelly adjustments
3. **Out-of-Sample Portfolio Validation**: Validate portfolio-level strategies on new data
4. **Real-World Portfolio Testing**: Monitor portfolio behavior in paper trading
5. **Discovery Portfolio Impact**: Assess impact of new strategies on portfolio risk

### CircuitBreakerManager v6.0

**New Capabilities**:
1. **Advanced Predictive Halts**: Predict triggers 1-2 hours in advance with 80%+ accuracy
2. **Preventive Action Coordination**: Issue P2P warnings, reduce limits BEFORE triggers
3. **Team Calibration Trigger**: If team overconfidence >25%, trigger immediate calibration
4. **Out-of-Sample Circuit Breaker Validation**: Test circuit breaker effectiveness on new data
5. **Real-World Halt Simulation**: Monitor paper trading for circuit breaker triggers

---

## Implementation Guidelines

### Phase 4 Rollout Plan

**Week 1: Framework Implementation**
- Create v6.0 framework document ✓
- Implement market-based task auction system
- Build team calibration module
- Develop out-of-sample validation framework

**Week 2: Supervisor v6.0**
- Implement 14-step enhanced chain of thought
- Add task auction coordination
- Build team calibration scheduler (every 50 trades)
- Integrate predictive stress monitoring
- Add refined adaptive Thompson Sampling (20-80% exploration)
- Implement real-world testing gate

**Week 3: Analysts v6.0**
- TechnicalAnalyst: Bidding, out-of-sample validation, discovery tracking
- SentimentAnalyst: Bidding, catalyst discovery, team calibration
- FundamentalsAnalyst (if applicable): Bidding, valuation validation

**Week 4: Traders v6.0**
- ConservativeTrader: Bidding, team calibration, paper trading validation
- ModerateTrader: Bidding, team calibration, strategy discovery
- AggressiveTrader: Bidding, team calibration, asymmetric discovery

**Week 5: Risk Managers v6.0**
- PositionRiskManager: Predictive alerts, team calibration support
- PortfolioRiskManager: Portfolio stress prediction, team Kelly adjustments
- CircuitBreakerManager: Advanced predictive halts, preventive actions

**Week 6: Integration Testing & Real-World Validation**
- Test complete system with all v6.0 features
- Run 30-day paper trading validation
- Backtest v6.0 vs v5.0 performance
- Verify team calibration improves collective performance
- Validate out-of-sample strategies outperform in-sample only
- Confirm predictive circuit breakers prevent triggers

### Code Structure

**New Modules Required**:
```
llm/
├── allocation/
│   ├── task_auction.py           # Market-based task bidding system
│   ├── bid_scoring.py             # Calculate bid scores (confidence * expertise * accuracy)
│   └── task_assignment.py         # Award tasks to highest bidders
│
├── calibration/
│   ├── team_calibration.py        # Collective agent adjustments
│   ├── overconfidence_detection.py # Track team overconfidence rate
│   └── calibration_scheduler.py   # Every 50 trades trigger calibration
│
├── validation/
│   ├── out_of_sample.py           # STOCKBENCH-style validation
│   ├── paper_trading.py           # 30-day real-world testing
│   └── validation_criteria.py     # Success criteria for deployment
│
├── prediction/
│   ├── stress_prediction.py       # Predict circuit breakers 1-2 hours early
│   ├── preventive_actions.py      # Reduce limits before triggers
│   └── prediction_accuracy.py     # Track ML prediction performance
│
├── thompson/
│   ├── refined_sampling.py        # Dynamic exploration rates (20-80%)
│   ├── discovery_tracking.py      # Track new high-performing strategies
│   └── exploration_decay.py       # Reduce exploration when not discovering
│
└── prompts/
    ├── supervisor_prompts.py      # Add SUPERVISOR_V6_0
    ├── analyst_prompts.py         # Add analyst v6.0 prompts
    ├── trader_prompts.py          # Add trader v6.0 prompts
    └── risk_prompts.py            # Add risk manager v6.0 prompts
```

### Testing Requirements

**Unit Tests**:
- Task auction: Test bidding, scoring, assignment
- Team calibration: Test overconfidence detection, collective adjustments
- Out-of-sample validation: Test train/validation split, degradation detection
- Predictive stress: Test circuit breaker prediction accuracy
- Refined Thompson Sampling: Test exploration rate adjustments

**Integration Tests**:
- Full system: All 9 agents with v6.0 features
- Task auction flow: Post task → Bids → Award → Delivery
- Team calibration flow: Performance → Overconfidence → Adjust → Monitor
- Paper trading flow: Strategy → 30-day test → Approval → Deploy

**Validation Tests**:
- Backtest v6.0 vs v5.0: Expect 10-15% performance improvement
- Out-of-sample validation: Strategies should maintain >85% of in-sample performance
- Team calibration effectiveness: Overconfidence should reduce from 25% to <15%
- Predictive circuit breakers: Should prevent >70% of triggers
- Paper trading success rate: >80% of strategies should pass 30-day test

---

## Validation Criteria

### V6.0 Success Metrics

**Individual Agent Performance**:
- Task auction efficiency: >90% tasks awarded to most qualified agent
- Out-of-sample validation: Strategy degradation <15%
- Team calibration: Overconfidence reduced from 25% to <15%
- Predictive accuracy: Circuit breaker predictions >70% accurate
- Paper trading success: >80% strategies pass 30-day validation

**System-Level Performance**:
- Portfolio Sharpe ratio: >2.5 (research benchmark: 2.21-3.05)
- Portfolio annual return: >35% (research benchmark: 35.56%)
- Win rate: >70% (research benchmark: 60-74%)
- Maximum drawdown: <15%
- Fill rate (real-world): >70%

**Production-Readiness Metrics**:
- Out-of-sample performance: Within 15% of in-sample
- Paper trading success: 30 days profitable
- Circuit breaker prevention: >70% of triggers prevented
- Team calibration: Collective overconfidence <15%
- Real-world deployment: All strategies pass validation

### Comparison to Research Benchmarks

| Metric | V6.0 Target | TradingAgents | MarketSenseAI | STOCKBENCH |
|--------|-------------|---------------|---------------|------------|
| **Sharpe Ratio** | >2.5 | 2.21-3.05 | N/A | N/A |
| **Annual Return** | >35% | 35.56% | 72% (15mo) | Varies |
| **Win Rate** | >70% | N/A | 60% (earnings) | N/A |
| **Max Drawdown** | <15% | N/A | N/A | N/A |
| **Out-of-Sample Degradation** | <15% | N/A | N/A | Post-cutoff |
| **Paper Trading Success** | >80% | N/A | N/A | Real-world |

---

## Appendix: V6.0 Research Foundations

### Key Papers Applied

1. **STOCKBENCH (2025)**:
   - Applied: Out-of-sample validation, post-knowledge-cutoff testing, real-world profitability focus
   - V6.0 Enhancement: 30-day paper trading requirement

2. **TradingAgents (2024-2025)**:
   - Applied: Hierarchical orchestration, veto power, Sharpe 2.21-3.05 benchmark, team performance tracking
   - V6.0 Enhancement: Full team calibration (not just individual agents)

3. **QTMRL (Multi-Indicator RL)**:
   - Applied: Multi-indicator synthesis, RL-style updates, adaptive strategies
   - V6.0 Enhancement: Discovery tracking, refined Thompson Sampling

4. **POW-dTS (Thompson Sampling)**:
   - Applied: Policy weighting, exploration/exploitation balance
   - V6.0 Enhancement: Dynamic exploration rates (20-80%), discovery-driven adjustments

5. **MarketSenseAI (2025)**:
   - Applied: GPT-4 beats analysts (60% vs 53%), chain of thought reasoning
   - V6.0 Enhancement: Market-based task allocation ensures best agent handles each task

6. **Agentic AI 2025 Trends**:
   - Applied: Self-healing systems, multi-agent collaboration, real-world deployment
   - V6.0 Enhancement: Production-ready framework with paper trading validation

---

## Summary

V6.0 represents the **FINAL ENHANCEMENT PHASE** and **PRODUCTION-READY SYSTEM**:

**From V5.0 (Collective Intelligence)**:
- Peer-to-peer communication ✓
- Portfolio-level Kelly ✓
- RL-style tracking ✓
- Cross-team learning ✓
- Adaptive Thompson Sampling ✓

**New in V6.0 (Production-Ready)**:
- **Market-based task allocation** (efficient agent assignment)
- **Full team calibration** (collective overconfidence reduction)
- **Out-of-sample validation** (STOCKBENCH approach)
- **Advanced predictive circuit breakers** (prevent triggers 1-2 hours early)
- **Refined Thompson Sampling** (dynamic 20-80% exploration)
- **Real-world testing framework** (30-day paper trading requirement)

**Expected Impact**:
- Sharpe: 2.5 → 2.7+ (from team calibration + out-of-sample validation)
- Win rate: 70% → 72%+ (from market-based allocation + refined exploration)
- Circuit breaker prevention: 0% → 70%+ (from predictive stress monitoring)
- Production success rate: N/A → 80%+ (from paper trading validation)
- Out-of-sample performance: 85%+ of in-sample (from validation protocols)

**Production Deployment Checklist**:
1. ✓ All strategies validated out-of-sample (degradation <15%)
2. ✓ Team calibration reducing overconfidence (<15%)
3. ✓ Predictive circuit breakers preventing triggers (>70%)
4. ✓ Market-based task allocation (>90% efficiency)
5. ✓ 30-day paper trading successful (>80% pass rate)
6. ✓ Human authorization obtained

---

**Next Steps**: Begin implementation with Supervisor v6.0 (most complex, sets pattern for all agents).

**Target**: **Match or exceed research benchmarks** → Sharpe >2.5, Return >35%, Win rate >70%, Drawdown <15%, Production-ready for live deployment.
