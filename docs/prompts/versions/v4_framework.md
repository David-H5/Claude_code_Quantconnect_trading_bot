# V4.0 Prompt Enhancement Framework
## 2025 Research-Backed Multi-Agent Trading System

**Document Status**: Framework complete for implementation across all 9 agents
**Research Basis**: STOCKBENCH, MarketSenseAI, POW-dTS, Agentic AI 2025, TradingAgents
**Version**: v4.0 (2025-12-01)

---

## Executive Summary

V4.0 represents a comprehensive upgrade to the multi-agent trading system based on 2025 academic research. The framework introduces ML validation, self-healing capabilities, Thompson Sampling for exploration/exploitation, and hierarchical multi-agent coordination.

**Completion Status**:
- ✅ **Supervisor v4.0**: Complete (~910 lines)
- ✅ **TechnicalAnalyst v4.0**: Complete (~887 lines)
- ✅ **SentimentAnalyst v4.0**: Complete (~480 lines)
- 🔄 **Traders (3)**: ConservativeTrader, ModerateTrader, AggressiveTrader
- 🔄 **Risk Managers (3)**: PositionRiskManager, PortfolioRiskManager, CircuitBreakerManager

---

## Core V4.0 Enhancements (All Agents)

### 1. ML Signal/Pattern Validation (STOCKBENCH/MarketSenseAI)

**Purpose**: Backtest signals/patterns before execution with regime-specific statistics

**Implementation**:
```
For each signal/pattern:
1. Identify signal characteristics
2. Query historical database (regime-specific)
3. Calculate win rate, Sharpe, risk/reward
4. Validate statistical edge (>55% win rate, >1.5:1 RR, >1.0 Sharpe)
5. Estimate conditional probability given current conditions
6. Boost confidence +15-25% if edge confirmed
```

**Research Basis**: STOCKBENCH (2025) emphasizes profitability validation, not just prediction accuracy

### 2. Self-Healing Error Recovery (Agentic AI 2025)

**Purpose**: Auto-recovery from data failures, API timeouts, calculation errors

**Error Types**:
- Data feed failures → Retry + backup source + cached data
- API timeouts → Exponential backoff + alternative providers
- Calculation errors → Fallback methods + simplified analysis
- Multi-source failures → Emergency mode (basic analysis, max confidence 0.40)

**Confidence Impact**: Reduce 10-25% per error, log all failures

**Research Basis**: Self-healing systems = top 2025 Agentic AI trend

### 3. Thompson Sampling (POW-dTS Algorithm)

**Purpose**: Exploration/exploitation balance for pattern/signal/strategy selection

**Algorithm**:
```python
For each option (pattern/signal/strategy):
    α = wins + 1
    β = losses + 1
    thompson_score = sample_beta(α, β)

Select option with highest thompson_score
Combine 60% traditional + 40% Thompson Sampling
```

**Benefits**: Adapts to recent performance, discovers new edges, balances proven vs exploratory

**Research Basis**: POW-dTS (Policy Weighting with Discounted Thompson Sampling) for market making

### 4. Enhanced Confidence Calibration

**Purpose**: Regime-specific accuracy tracking with dynamic adjustments

**Calibration Factors** (8-10 per agent):
- ML validation boost (+0.15 to +0.25)
- Regime-specific accuracy (±0.05 to ±0.15)
- Source/pattern agreement (±0.10 to ±0.20)
- Recent performance (±0.10 to ±0.15)
- Validation factor count (±0.10 to ±0.15)
- Error recovery penalty (-0.10 to -0.25)
- Signal momentum/strength (±0.05 to ±0.10)

**Regime Tracking**: Low/Normal/High/Extreme volatility → separate accuracy stats

### 5. Blackboard Integration

**Purpose**: Share findings via centralized multi-agent state

**Write Format**:
```json
{
    "blackboard_state": {
        "agent_name_analysis": {
            "analyst": "AgentName",
            "timestamp": "ISO8601",
            "symbol": "TICKER",
            "findings": {...},
            "risk_factors": [...],
            "conviction_level": "low|medium|high"
        }
    }
}
```

**Read Usage**: Check other agents' findings for conflicts/synergies

**Research Basis**: Async collaboration pattern from multi-agent framework comparison

### 6. Team Lead Reporting

**Purpose**: Hierarchical coordination reduces information overload

**Hierarchy**:
```
SUPERVISOR (Chief Trading Officer)
│
├── TECHNICAL LEAD → TechnicalAnalyst, SentimentAnalyst, Fundamentals
├── STRATEGY LEAD → BullResearcher, BearResearcher, Traders
└── RISK LEAD → PositionRisk, PortfolioRisk, CircuitBreaker
```

**Reporting Format**:
```
TO: [Lead Name]
FROM: [Agent Name]
RE: [Symbol] [Analysis Type]

SUMMARY: [1-2 sentence key finding]

KEY FINDINGS:
- [Bullet points]

RISK FACTORS:
- [Bullet points]

CONFIDENCE: [0.30-0.90] ([low|medium|high])
CONVICTION: [Low|Medium|High] - [reasoning]

COORDINATION NOTES:
- [Cross-agent recommendations]
```

**Research Basis**: TradingAgents framework hierarchical team structure

### 7. Enhanced Chain of Thought

**Purpose**: Structured reasoning for transparency and accuracy

**Process** (varies by agent type):

**Analysts** (5-6 steps):
1. GATHER: Multi-source data collection
2. VALIDATE: ML historical performance check
3. ANALYZE: Pattern/signal recognition
4. CROSS-CHECK: Multi-source validation
5. SYNTHESIZE: Confidence calibration
6. REPORT: Blackboard write + Team Lead

**Traders** (6-7 steps):
1. GATHER: Read Blackboard + analyst inputs
2. SYNTHESIZE: Combine technical + sentiment + fundamentals
3. STRATEGY SELECTION: ML-validate strategy performance
4. POSITION SIZING: Kelly Criterion with regime adjustment
5. RISK ASSESSMENT: Check portfolio heat + circuit breakers
6. DECISION: Trade recommendation
7. REPORT: Write to Blackboard + Strategy Lead

**Risk Managers** (5-6 steps):
1. MONITOR: Portfolio state + market regime
2. CALCULATE: Risk metrics (heat, VaR, Sharpe, drawdown)
3. VALIDATE: Check against limits
4. PREDICT: Stress scenarios with ML
5. DECIDE: Approve/reject/reduce
6. REPORT: Write to Blackboard + Risk Lead

**Research Basis**: MarketSenseAI (GPT-4 72% return with Chain of Thought)

---

## Agent-Specific V4.0 Enhancements

### Supervisor v4.0 ✅ COMPLETE

**Unique Features**:
- Hierarchical team delegation (Technical/Strategy/Risk Leads)
- Multi-agent debate protocol with weighted voting
- ML policy validation (backtest team composition performance)
- Thompson Sampling for agent weighting (not static Sharpe-based)
- Emergency mode for system degradation

**Model**: opus-4 (requires deepest reasoning)
**Temperature**: 0.4
**Max Tokens**: 4000

### TechnicalAnalyst v4.0 ✅ COMPLETE

**Unique Features**:
- ML pattern validation (127+ historical instances per pattern)
- Thompson Sampling for pattern selection (40+ patterns)
- Self-healing data feeds (price, volume, indicators)
- Regime-specific pattern performance (4 volatility regimes)
- 6 error recovery protocols

**Model**: opus-4 (pattern recognition requires reasoning)
**Temperature**: 0.3
**Max Tokens**: 3000

### SentimentAnalyst v4.0 ✅ COMPLETE

**Unique Features**:
- ML sentiment signal validation (contrarian/consensus)
- Thompson Sampling for source weighting (FinBERT/News/Social/Analyst)
- Self-healing API failures (6 source types)
- Regime-specific sentiment accuracy (4 regimes)
- Contrarian validation stricter (3+ factors required)

**Model**: opus-4 (behavioral psychology nuanced)
**Temperature**: 0.5
**Max Tokens**: 3000

### ConservativeTrader v4.0 🔄 TODO

**Unique Features** (v4.0 additions):
- ML strategy validation (iron condor, credit spreads, butterflies)
- Thompson Sampling for strategy selection (high-prob strategies)
- Kelly Criterion with regime-adjusted half-Kelly (0.10-0.25)
- Position sizing limits: Max 15% per trade, 5% risk
- Self-healing quote data failures

**Risk Profile**: 0.5-1.0% per trade, 65-80% win rate target
**Model**: sonnet-4 (structured decision-making)
**Temperature**: 0.3
**Max Tokens**: 2500

### ModerateTrader v4.0 🔄 TODO

**Unique Features** (v4.0 additions):
- ML strategy validation (debit spreads, calendars, diagonals)
- Thompson Sampling for strategy selection (balanced strategies)
- Kelly Criterion with regime-adjusted fractional (0.25-0.50)
- Position sizing limits: Max 20% per trade, 8% risk
- Self-healing Greeks calculation errors

**Risk Profile**: 1.0-2.0% per trade, 55-70% win rate target
**Model**: sonnet-4
**Temperature**: 0.4
**Max Tokens**: 2500

### AggressiveTrader v4.0 🔄 TODO

**Unique Features** (v4.0 additions):
- ML strategy validation (long options, volatility plays, earnings)
- Thompson Sampling for strategy selection (high-reward strategies)
- Kelly Criterion with regime-adjusted full-Kelly (0.50-1.00)
- Position sizing limits: Max 25% per trade, 10% risk
- Self-healing IV surface construction

**Risk Profile**: 2.0-3.0% per trade, 45-60% win rate target
**Model**: opus-4 (aggressive requires nuanced risk assessment)
**Temperature**: 0.5
**Max Tokens**: 2500

### PositionRiskManager v4.0 🔄 TODO

**Unique Features** (v4.0 additions):
- ML position stress testing (regime-specific scenarios)
- Thompson Sampling for limit adjustment (adaptive thresholds)
- Self-healing Greeks feed failures
- Real-time position heat monitoring
- ABSOLUTE VETO POWER (overrides all)

**Limits**: Position size, Delta exposure, Theta decay, Vega exposure
**Model**: opus-4 (risk requires deep reasoning)
**Temperature**: 0.2 (conservative)
**Max Tokens**: 2500

### PortfolioRiskManager v4.0 🔄 TODO

**Unique Features** (v4.0 additions):
- ML portfolio stress testing (VaR, CVaR, tail risk)
- Thompson Sampling for dynamic limit adjustment (VIX-based)
- Self-healing portfolio data aggregation
- Correlation matrix monitoring
- Strategy allocation optimization

**Limits**: Portfolio heat, sector concentration, correlation risk
**Model**: opus-4
**Temperature**: 0.2
**Max Tokens**: 2500

### CircuitBreakerManager v4.0 🔄 TODO

**Unique Features** (v4.0 additions):
- ML drawdown prediction (early warning system)
- Thompson Sampling for threshold adaptation
- Self-healing stress score calculation
- 3-level halt system (7%/13%/20% daily loss)
- Human override required for reset

**Triggers**: Daily loss, drawdown, consecutive losses, market stress
**Model**: opus-4 (critical safety requires best reasoning)
**Temperature**: 0.1 (extremely conservative)
**Max Tokens**: 2500

---

## Research Foundations (2025)

### STOCKBENCH (March-June 2025)
- **Key Finding**: Real-world profitability validation, not just prediction accuracy
- **Application**: ML validation must demonstrate trading edge (win rate, Sharpe, RR)
- **Metrics**: 82 trading days, $100k starting capital, commission-inclusive

### MarketSenseAI (GPT-4 Framework)
- **Key Finding**: GPT-4 beats analysts 60% vs 53% accuracy, 72% cumulative return
- **Application**: Chain of Thought reasoning process, multi-factor synthesis
- **Approach**: Sentiment + Technicals + Fundamentals + Macroeconomics

### POW-dTS (Policy Weighting with Discounted Thompson Sampling)
- **Key Finding**: Thompson Sampling balances exploration/exploitation for market making
- **Application**: Adaptive weighting for patterns/signals/strategies/sources
- **Algorithm**: Beta distributions, recent performance weighted higher (0.9^days_ago)

### Agentic AI Market 2025 ($8.31B → $154.84B by 2033)
- **Key Finding**: Self-healing systems = #1 trend, 44.21% CAGR
- **Application**: Auto-recovery from errors, resilience, autonomous operation
- **Use Cases**: Multi-agent collaboration, real-time monitoring, specialized teams

### TradingAgents Framework (Sharpe 2.21-3.05, 35.56% returns)
- **Key Finding**: Self-reflection reduces overconfidence 30-40%
- **Application**: Hierarchical team structure, Technical/Strategy/Risk Leads
- **Approach**: Debate, critique, learning from errors

### FinBERT (20% Accuracy Improvement)
- **Key Finding**: Financial-specific NLP significantly outperforms generic models
- **Application**: Primary sentiment source (40% weight), multi-source validation
- **Usage**: Recent news most predictive (1-5 days), compare to historical baselines

---

## Implementation Guidelines

### Creating V4.0 Prompts (Remaining 6 Agents)

**Structure** (~600-900 lines per agent):

```
1. VERSION 4.0 ENHANCEMENTS (150-200 lines)
   - New capabilities list
   - Research basis citations
   - Enhanced competencies

2. ML VALIDATION (150-200 lines)
   - 6-step validation process
   - Statistical edge criteria
   - Conditional probability estimation
   - Output format with ML fields

3. SELF-HEALING ERROR RECOVERY (150-200 lines)
   - 5-6 error types specific to agent
   - Recovery protocols with fallbacks
   - Confidence impact calculation
   - Emergency mode definition

4. THOMPSON SAMPLING (100-150 lines)
   - Algorithm explanation for agent context
   - Practical examples (3-4)
   - Weighting formula (traditional + Thompson)
   - Exploration bonus for discovery

5. ENHANCED CONFIDENCE CALIBRATION (100-150 lines)
   - Regime performance tracking
   - Dynamic adjustment formula (8-10 factors)
   - Thresholds and caps

6. BLACKBOARD INTEGRATION (50-75 lines)
   - Write format
   - Read usage
   - Coordination patterns

7. TEAM LEAD REPORTING (50-75 lines)
   - Hierarchy diagram
   - Report format
   - Coordination notes

8. ENHANCED CHAIN OF THOUGHT (50-75 lines)
   - Step-by-step process (5-7 steps)
   - Decision gates
   - Output format

9. AGENT-SPECIFIC CONTENT (100-200 lines)
   - Unique capabilities (v3.0 base + v4.0 enhancements)
   - Strategy/pattern/signal library
   - Decision criteria

10. OUTPUT FORMAT (50-75 lines)
    - JSON schema with v4.0 fields
    - ML validation results
    - Thompson sampling scores
    - Self-healing logs
    - Confidence calibration breakdown

11. CONSTRAINTS & EXAMPLES (75-100 lines)
    - Updated constraints (v4.0 specific)
    - Complete workflow example
    - Error recovery example
```

### Registration Template

```python
register_prompt(
    role=AgentRole.AGENT_NAME,
    template=AGENT_NAME_V4_0,
    version="v4.0",
    model="opus-4|sonnet-4",  # opus for deep reasoning, sonnet for structured
    temperature=0.1-0.5,  # lower for risk, higher for creative
    max_tokens=2500-4000,  # supervisor highest, others 2500-3000
    description="[One-line v4.0 summary with key enhancements]",
    changelog="v4.0 2025 RESEARCH ENHANCEMENTS: Added [ML validation details], [Self-healing details], [Thompson Sampling details], [Confidence calibration details], [Blackboard integration], [Team Lead reporting], [Enhanced Chain of Thought], Research foundations: [List 3-4 key papers with findings]",
    created_by="claude_code_agent",
)
```

### Model Selection Criteria

| Agent | Model | Rationale |
|-------|-------|-----------|
| Supervisor | opus-4 | Deepest multi-agent orchestration reasoning |
| TechnicalAnalyst | opus-4 | Pattern recognition nuanced |
| SentimentAnalyst | opus-4 | Behavioral psychology complex |
| ConservativeTrader | sonnet-4 | Structured low-risk decision-making |
| ModerateTrader | sonnet-4 | Balanced systematic approach |
| AggressiveTrader | opus-4 | High-risk nuanced assessment |
| PositionRiskManager | opus-4 | Critical safety reasoning |
| PortfolioRiskManager | opus-4 | Complex correlation analysis |
| CircuitBreakerManager | opus-4 | Critical halt decisions |

**Rationale**: Opus for nuanced reasoning (4 agents), Sonnet for structured (2 agents)

### Temperature Guidelines

| Risk Level | Temperature | Agents |
|-----------|-------------|--------|
| Critical Safety | 0.1-0.2 | CircuitBreaker, Risk Managers |
| Conservative | 0.3 | TechnicalAnalyst, ConservativeTrader |
| Moderate | 0.4 | Supervisor, ModerateTrader |
| Exploratory | 0.5 | SentimentAnalyst, AggressiveTrader |

---

## Testing & Validation

### Unit Testing V4.0 Enhancements

For each agent, test:
1. ML validation returns correct statistical edge assessment
2. Self-healing triggers appropriate fallbacks for each error type
3. Thompson Sampling correctly weights based on recent performance
4. Confidence calibration applies all factors correctly
5. Blackboard writes/reads function properly
6. Team Lead reports contain all required sections

### Integration Testing

1. **Full Pipeline**: Supervisor → Analysts → Traders → Risk Managers
2. **Blackboard Sync**: All agents read/write without conflicts
3. **Hierarchical Communication**: Leads correctly aggregate specialist inputs
4. **Error Cascade**: Self-healing prevents chain failures
5. **Thompson Adaptation**: Weights update based on outcomes

### Performance Benchmarks (Target Metrics)

Based on research papers:
- **Sharpe Ratio**: >2.0 (TradingAgents: 2.21-3.05)
- **Annualized Return**: >25% (TradingAgents: 35.56%, MarketSenseAI: 72%/15mo = 57%/yr)
- **Win Rate**: >60% (MarketSenseAI: 60% earnings prediction)
- **Max Drawdown**: <20%
- **Prediction Accuracy**: >70% in normal volatility (FinBERT: 20% improvement)

---

## Implementation Schedule

**Completed** (2025-12-01):
- [x] 2025 Research Document Update (~450 lines, Section 9)
- [x] Supervisor v4.0 (~910 lines)
- [x] TechnicalAnalyst v4.0 (~887 lines)
- [x] SentimentAnalyst v4.0 (~480 lines)

**Remaining** (Estimated ~4500-5500 lines total):
- [ ] ConservativeTrader v4.0 (~600-700 lines)
- [ ] ModerateTrader v4.0 (~600-700 lines)
- [ ] AggressiveTrader v4.0 (~700-800 lines, most complex trader)
- [ ] PositionRiskManager v4.0 (~700-800 lines)
- [ ] PortfolioRiskManager v4.0 (~700-800 lines)
- [ ] CircuitBreakerManager v4.0 (~600-700 lines)

**Estimated Time**: 3-4 hours for remaining 6 agents at current pace

---

## Changelog

### v4.0 (2025-12-01)
- Initial framework document created
- 3 of 9 agents complete (Supervisor, TechnicalAnalyst, SentimentAnalyst)
- Research basis: STOCKBENCH, MarketSenseAI, POW-dTS, Agentic AI 2025, TradingAgents, FinBERT
- 7 core enhancements defined for all agents
- Agent-specific enhancement specifications for remaining 6 agents

---

## References

1. **STOCKBENCH** (2025): "Real-World Profitability Validation for LLM Trading Systems"
2. **MarketSenseAI** (2025): "GPT-4 Framework for Financial Analysis: 60% vs 53% Accuracy, 72% Returns"
3. **POW-dTS** (2025): "Policy Weighting with Discounted Thompson Sampling for Market Making"
4. **Agentic AI Market Report** (2025): "$8.31B → $154.84B by 2033, 44.21% CAGR"
5. **TradingAgents Framework** (2024): "Multi-Agent Trading with Self-Reflection: Sharpe 2.21-3.05, 35.56% Returns"
6. **FinBERT Study** (2024): "20% Accuracy Improvement in Financial Sentiment Analysis"
7. **QTMRL** (2025): "Quantitative Trading with Multi-Indicator Reinforcement Learning"

---

**Document Maintained By**: Claude Code Agent
**Last Updated**: 2025-12-01
**Status**: Framework Complete, Implementation In Progress (3/9 agents)
