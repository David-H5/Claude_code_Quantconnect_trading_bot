# Evaluation Framework - Quick Start Guide

## Overview

This evaluation framework implements the **STOCKBENCH methodology** with 30+ contamination-free test cases per agent type using 2024-2025 market data.

## What Was Built

### Core Framework (`evaluation/`)
- ✅ `evaluation_framework.py` - Core evaluation engine (AgentEvaluator, TestCase, EvaluationResult classes)
- ✅ `metrics.py` - Performance metrics calculation (agent-specific, STOCKBENCH, v6.1 enhanced)
- ✅ `run_evaluation.py` - Main CLI runner for executing evaluations
- ✅ `example_usage.py` - Complete example showing how to integrate with your agents

### Test Case Datasets (`evaluation/datasets/`)
- ✅ `analyst_cases.py` - 32 test cases for TechnicalAnalyst, 32 for SentimentAnalyst
- ✅ `trader_cases.py` - 32 test cases each for Conservative/Moderate/Aggressive traders
- ✅ `risk_manager_cases.py` - 32 test cases each for Position/Portfolio/CircuitBreaker managers

**Total**: 288 test cases across 8 agent types (Supervisor pending)

### Test Case Categories

Each agent has 30+ test cases distributed as:
- **Success cases (40%)**: High-confidence scenarios that should succeed
- **Edge cases (40%)**: Challenging scenarios requiring careful handling
- **Failure cases (20%)**: Cases that should trigger rejection/warnings

## Quick Usage

### 1. Run Evaluation for All Agents

```bash
PYTHONPATH=/home/dshooter/projects/Claude_code_Quantconnect_trading_bot \
python3 evaluation/run_evaluation.py --all
```

### 2. Run Evaluation for Specific Agent

```bash
PYTHONPATH=/home/dshooter/projects/Claude_code_Quantconnect_trading_bot \
python3 evaluation/run_evaluation.py --agent TechnicalAnalyst --version v6.1
```

### 3. Run with Custom Threshold

```bash
PYTHONPATH=/home/dshooter/projects/Claude_code_Quantconnect_trading_bot \
python3 evaluation/run_evaluation.py --all --threshold 0.85
```

### 4. Generate Report from Saved Results

```bash
PYTHONPATH=/home/dshooter/projects/Claude_code_Quantconnect_trading_bot \
python3 evaluation/run_evaluation.py --report-only \
    --results evaluation/results/20251201_143000_TechnicalAnalyst_v6.1_results.json
```

## Integration with Your Agents

### Step 1: Implement Agent Callable

Your agent must accept `Dict[str, Any]` input and return `Dict[str, Any]` output:

```python
def my_technical_analyst(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Your LLM-powered agent implementation.

    Args:
        input_data: Test case input (symbol, pattern_type, win_rates, etc.)

    Returns:
        Dict with keys: signal, confidence, pattern_valid, out_of_sample_validated
    """
    # Your agent logic here (LLM call, pattern analysis, etc.)
    return {
        "signal": "bullish",  # or "bearish" or "neutral"
        "confidence": 0.68,
        "pattern_valid": True,
        "out_of_sample_validated": True,
    }
```

### Step 2: Load Test Cases

```python
from evaluation.datasets import get_technical_analyst_cases

test_cases = get_technical_analyst_cases()  # 32 test cases
```

### Step 3: Create Evaluator

```python
from evaluation.evaluation_framework import AgentEvaluator

evaluator = AgentEvaluator(
    agent_type="TechnicalAnalyst",
    version="v6.1",
    test_cases=test_cases,
    agent_callable=my_technical_analyst,  # Your agent here
)
```

### Step 4: Run Evaluation

```python
result = evaluator.run()

print(f"Pass rate: {result.pass_rate:.1%}")
print(f"Production ready: {result.production_ready}")
```

### Step 5: Save Results

```python
from pathlib import Path

output_dir = Path("evaluation/results")
result_file = evaluator.save_results(output_dir)
print(f"Saved to: {result_file}")
```

## Expected Output Format by Agent Type

### Analysts (Technical/Sentiment)

```python
{
    "signal": "bullish" | "bearish" | "neutral",
    "confidence": float,  # 0-1
    "pattern_valid": bool,
    "out_of_sample_validated": bool,
    "degradation_pct": float,  # optional
    "rejection_reason": str,  # optional
}
```

### Traders (Conservative/Moderate/Aggressive)

```python
{
    "position_size_pct": float,  # 0-1
    "approved": bool,
    "out_of_sample_adjusted": bool,
    "kelly_base": float,  # optional
    "fractional_kelly": float,  # optional
    "rejection_reason": str,  # optional
}
```

### Risk Managers (Position/Portfolio/CircuitBreaker)

```python
{
    "approved": bool,
    "veto_triggered": bool,
    "violations": List[str],
    "greeks_exceeded": bool,  # PositionRiskManager
    "correlation_exceeded": bool,  # PortfolioRiskManager
    "circuit_breaker_triggered": bool,  # CircuitBreakerManager
    "preventive_action_taken": bool,  # CircuitBreakerManager
    "rejection_reason": str,  # optional
}
```

## Production Deployment Criteria

Before deploying to paper trading, ALL agents must pass:

### Phase 1: Evaluation Dataset (30+ Cases)
- ✅ Success case accuracy: >90%
- ✅ Edge case accuracy: >80%
- ✅ Failure case detection: >95%
- ✅ Overall pass rate: >90%

### Phase 2: Paper Trading (30 Days)
- ✅ Win rate targets (Conservative >65%, Moderate >60%, Aggressive >55%)
- ✅ Sharpe ratio targets (Conservative >1.5, Moderate >1.3, Aggressive >1.0)
- ✅ Maximum drawdown under limits
- ✅ Zero critical violations

### Phase 3: Out-of-Sample Validation
- ✅ Strategy degradation <15% on 2024-2025 data
- ✅ Pattern/sentiment accuracy maintained
- ✅ Risk model effectiveness >75%

### Phase 4: Team Performance
- ✅ Team Sharpe >2.5
- ✅ Team win rate >70%
- ✅ Circuit breaker prevention >70%
- ✅ Token budget <50K/day

## Example: Running Complete Evaluation

See `evaluation/example_usage.py` for a complete working example:

```bash
PYTHONPATH=/home/dshooter/projects/Claude_code_Quantconnect_trading_bot \
python3 evaluation/example_usage.py
```

This demonstrates:
1. Loading test cases
2. Creating evaluator with custom agent
3. Running evaluation
4. Generating reports
5. Saving results

## Test Results Interpretation

### Example Output

```
============================================================
EVALUATION REPORT: TechnicalAnalyst v6.1
============================================================
Total Cases: 32
Duration: 0.5 seconds

CATEGORY BREAKDOWN:
  Success Cases (11/13): 84.6% ✅
  Edge Cases (11/13): 84.6% ✅
  Failure Cases (6/6): 100.0% ✅

OVERALL ACCURACY: 87.5% ⚠️ (Target: >90%)

METRICS:
  Accuracy: 87.5%
  False Positive Rate: 3.1%
  Out Of Sample Usage Rate: 100.0%

PASS STATUS: ⚠️ NOT PRODUCTION-READY
```

**Interpretation**:
- Success cases: 84.6% < 90% target → **Needs improvement**
- Edge cases: 84.6% > 80% target → **Acceptable**
- Failure cases: 100% > 95% target → **Excellent**
- Overall: 87.5% < 90% threshold → **Not production ready**

**Action**: Fix the 2 failing success cases before deployment.

## File Structure

```
evaluation/
├── README.md                    # Complete documentation
├── QUICK_START.md               # This file
├── __init__.py                  # Module exports
├── evaluation_framework.py      # Core evaluation engine
├── metrics.py                   # Performance metrics
├── run_evaluation.py            # CLI runner
├── example_usage.py             # Integration example
├── datasets/                    # Test case datasets
│   ├── __init__.py
│   ├── analyst_cases.py         # Technical & Sentiment (64 cases)
│   ├── trader_cases.py          # 3 trader types (96 cases)
│   └── risk_manager_cases.py    # 3 risk managers (96 cases)
└── results/                     # Auto-generated results
    └── {timestamp}_*.json
```

## Next Steps

1. **Implement agent callables** for your v6.1 LLM-powered agents
2. **Run evaluations** using the CLI or integration code
3. **Iterate on failures** until all agents pass >90% threshold
4. **Deploy to paper trading** once evaluation passes
5. **Monitor live performance** against paper trading metrics

## Support

- Full documentation: [evaluation/README.md](./README.md)
- Example code: [evaluation/example_usage.py](./example_usage.py)
- Test case datasets: [evaluation/datasets/](./datasets/)

## References

- [STOCKBENCH: Can LLM Agents Trade Stocks Profitably](https://arxiv.org/abs/2510.02209)
- [Agent Trading Arena](https://arxiv.org/html/2502.17967v2)
- [How to Test AI Agents Effectively](https://galileo.ai/learn/test-ai-agents)
