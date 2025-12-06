# Autonomous AI Agent Trading System - Evaluation Framework

## Overview

This evaluation system implements the **STOCKBENCH methodology** ([arXiv:2510.02209](https://arxiv.org/abs/2510.02209)) combined with 2025 best practices for autonomous AI agent testing. The system validates all 9 trading agents against 30+ test cases per agent type covering success scenarios, edge cases, and failure modes.

## Framework Architecture

```
evaluation/
├── README.md                    # This file
├── evaluation_framework.py      # Core evaluation engine
├── metrics.py                   # Performance metrics tracking
├── run_evaluation.py            # Main evaluation runner
├── datasets/                    # Test case datasets
│   ├── analyst_cases.py         # Technical & Sentiment analyst test cases
│   ├── trader_cases.py          # Conservative/Moderate/Aggressive trader cases
│   ├── risk_manager_cases.py   # Position/Portfolio/Circuit Breaker cases
│   └── integration_cases.py    # Multi-agent integration scenarios
└── results/                     # Evaluation results (auto-generated)
    ├── {timestamp}_results.json
    └── {timestamp}_report.md
```

## Evaluation Methodology (STOCKBENCH-Inspired)

### 1. **Contamination-Free Testing**
- All test cases use market data from **2024-2025** (post-training cutoff)
- No overlap with LLM training corpora
- Real market scenarios with verifiable outcomes

### 2. **Multi-Domain Test Cases**
Each agent type has 30+ test cases across three categories:

**Success Cases (40%)**: High-confidence scenarios that should succeed
- Technical: Head & shoulders reversal, double bottom breakout
- Sentiment: Earnings beat with positive guidance
- Traders: Iron condors in low VIX (Conservative), asymmetric long calls (Aggressive)
- Risk: Proper position sizing, Greeks within bounds

**Edge Cases (40%)**: Challenging scenarios requiring careful handling
- High VIX >35 (distorts patterns)
- Low liquidity (slippage risk)
- Mixed sentiment (beat EPS, miss revenue)
- Near-limit positions (24% vs 25% threshold)
- Borderline stress scores (59 vs 60 threshold)

**Failure Scenarios (20%)**: Cases that should trigger rejection/warnings
- False breakouts, failed patterns
- Sentiment reversals post-earnings
- Position limit violations
- Missing stop losses
- Greeks exceeding bounds

### 3. **Performance Metrics**

**Agent-Specific Metrics**:
- **Analysts**: Accuracy >60% (Technical), >55% (Sentiment), False positive rate <30%
- **Traders**: Win rate >65% (Conservative), >60% (Moderate), >55% (Aggressive), Kelly calculation accuracy
- **Risk Managers**: Zero violations >95%, Stop loss effectiveness >75%, Greeks within bounds >95%

**STOCKBENCH Standard Metrics**:
- Cumulative Return
- Maximum Drawdown
- Sortino Ratio
- Calmar Ratio (Return / Max Drawdown)
- Win Rate
- Profit Factor

**v6.1 Enhanced Metrics**:
- Out-of-sample validation accuracy
- Team calibration effectiveness
- Predictive circuit breaker prevention rate >70%
- False positive rate <20%
- Token budget adherence <50K/day

### 4. **Evaluation Workflow**

```python
# 1. Load test case dataset for agent type
test_cases = load_agent_test_cases(agent_type="TechnicalAnalyst")

# 2. Initialize agent with v6.1 prompt
agent = create_agent(role="TechnicalAnalyst", version="v6.1")

# 3. Run agent against all test cases
results = []
for case in test_cases:
    response = agent.evaluate(case.input_data)
    result = compare_with_expected(response, case.expected_output)
    results.append(result)

# 4. Calculate metrics
metrics = calculate_metrics(results)

# 5. Generate report
generate_evaluation_report(metrics, results)
```

## Test Case Structure

Each test case follows this structure:

```python
{
    "case_id": "TECH_SUCCESS_001",
    "category": "success",  # success | edge | failure
    "agent_type": "TechnicalAnalyst",
    "scenario": "Bull flag pattern on AAPL daily chart",
    "input_data": {
        "symbol": "AAPL",
        "pattern_type": "bull_flag",
        "timeframe": "daily",
        "in_sample_win_rate": 0.68,
        "out_of_sample_win_rate": 0.62,  # 8.8% degradation (acceptable)
        "volume_confirmation": True,
        "support_resistance": [175.50, 178.20, 180.00]
    },
    "expected_output": {
        "signal": "bullish",
        "confidence": 0.62,  # Out-of-sample adjusted
        "confidence_adjustment": 0.91,  # (62/68)
        "pattern_valid": True,
        "out_of_sample_validated": True,
        "degradation_pct": 8.8
    },
    "success_criteria": {
        "signal_correct": True,
        "confidence_within_range": [0.58, 0.66],
        "out_of_sample_check": True,
        "degradation_under_15": True
    }
}
```

## Running Evaluations

### Quick Start

```bash
# Run evaluation for all agents
python evaluation/run_evaluation.py --all

# Run evaluation for specific agent type
python evaluation/run_evaluation.py --agent TechnicalAnalyst

# Run evaluation with specific version
python evaluation/run_evaluation.py --agent ConservativeTrader --version v6.1

# Generate only the report (skip execution)
python evaluation/run_evaluation.py --report-only --results results/latest_results.json
```

### Continuous Evaluation (CI/CD Integration)

```bash
# Run evaluation suite before deployment
python evaluation/run_evaluation.py --all --threshold 0.90

# Exit code 0 if all agents pass threshold
# Exit code 1 if any agent fails
```

### Paper Trading Integration

```bash
# Validate 30-day paper trading results
python evaluation/validate_paper_trading.py \
    --start-date 2025-01-01 \
    --end-date 2025-01-30 \
    --agents all
```

## Success Criteria for Production Deployment

Before deploying to paper trading or live trading, ALL agents must pass:

### Phase 1: Evaluation Dataset (30+ Cases)
- ✅ Success case accuracy: >90%
- ✅ Edge case accuracy: >80%
- ✅ Failure case detection: >95%
- ✅ Overall pass rate: >90%

### Phase 2: Paper Trading (30 Days)
- ✅ Win rate targets achieved (Conservative >65%, Moderate >60%, Aggressive >55%)
- ✅ Sharpe ratio targets (Conservative >1.5, Moderate >1.3, Aggressive >1.0)
- ✅ Maximum drawdown under limits (Conservative <10%, Moderate <15%, Aggressive <20%)
- ✅ Fill rate targets (Conservative >70%, Moderate >65%, Aggressive >60%)
- ✅ Zero critical violations (position limits, Greeks bounds, stop losses)

### Phase 3: Out-of-Sample Validation
- ✅ Strategy degradation <15% on 2024-2025 data
- ✅ Pattern/sentiment accuracy maintained
- ✅ Risk model effectiveness >75%

### Phase 4: Team Performance
- ✅ Team Sharpe >2.5
- ✅ Team win rate >70%
- ✅ Circuit breaker prevention >70%
- ✅ Team overconfidence <20%
- ✅ Token budget <50K/day

## Example Evaluation Report

```
========================================
EVALUATION REPORT: TechnicalAnalyst v6.1
========================================
Date: 2025-12-01 14:30:00
Total Cases: 32
Duration: 45.2 seconds

CATEGORY BREAKDOWN:
  Success Cases (13/13): 100.0% ✅
  Edge Cases (12/13):     92.3% ✅
  Failure Cases (6/6):   100.0% ✅

OVERALL ACCURACY: 96.9% ✅ (Target: >90%)

METRICS:
  Pattern Recognition Accuracy: 94.2%
  Out-of-Sample Validation: 91.7%
  False Positive Rate: 8.3% ✅ (Target: <30%)
  Confidence Calibration: 0.03 RMSE
  ReAct Framework Usage: 100%

DETAILED RESULTS:
  ✅ TECH_SUCCESS_001: Bull flag pattern (AAPL)
  ✅ TECH_SUCCESS_002: Head & shoulders reversal (MSFT)
  ...
  ✅ TECH_EDGE_001: High VIX pattern distortion
  ⚠️ TECH_EDGE_009: Low volume false breakout [FAILED]
  ...
  ✅ TECH_FAIL_001: Failed triangle breakout detection

PASS STATUS: ✅ PRODUCTION-READY
```

## Integration with Existing System

### 1. Import Evaluation Framework

```python
from evaluation.evaluation_framework import AgentEvaluator
from evaluation.datasets.analyst_cases import get_technical_analyst_cases

# Create evaluator
evaluator = AgentEvaluator(
    agent_type="TechnicalAnalyst",
    version="v6.1",
    test_cases=get_technical_analyst_cases()
)

# Run evaluation
results = evaluator.run()
print(f"Pass rate: {results.pass_rate:.1%}")
```

### 2. Pre-Deployment Gate

```python
from evaluation.run_evaluation import run_full_evaluation

# Run before deploying to paper trading
if run_full_evaluation(threshold=0.90):
    print("✅ All agents passed evaluation. Ready for paper trading.")
    deploy_to_paper_trading()
else:
    print("❌ Evaluation failed. Fix failing agents before deployment.")
```

## References

**Evaluation Methodology**:
- [STOCKBENCH: Can LLM Agents Trade Stocks Profitably](https://arxiv.org/abs/2510.02209)
- [Agent Trading Arena: Competitive Multi-Agent Trading](https://arxiv.org/html/2502.17967v2)
- [TradingGroup: Multi-Agent Trading with Self-Reflection](https://arxiv.org/html/2508.17565)
- [How to Test AI Agents Effectively](https://galileo.ai/learn/test-ai-agents)
- [The Future of AI Agent Testing: Trends 2025](https://qawerk.com/blog/ai-agent-testing-trends/)

**Best Practices**:
- [Evaluation-Driven Development of LLM Agents](https://arxiv.org/html/2411.13768v2)
- [AI Evaluations: A Moat Bigger than the Model](https://medium.com/@zhengfke/ai-evaluations-a-moat-bigger-than-the-model-0577aa32c4d7)
- [UiPath AI Agent Testing Best Practices](https://www.uipath.com/blog/ai/agent-builder-best-practices)

## License

This evaluation framework is part of the QuantConnect Trading Bot project and follows the same license terms.
