# Advanced Evaluation Frameworks (2025)

This document describes the advanced evaluation frameworks integrated into our trading system based on cutting-edge research from 2025.

## Table of Contents

1. [CLASSic Framework (ICLR 2025)](#classic-framework)
2. [Walk-Forward Analysis](#walk-forward-analysis)
3. [Advanced Trading Metrics](#advanced-trading-metrics)
4. [Component-Level Evaluation](#component-level-evaluation)
5. [Overfitting Prevention (QuantConnect Best Practices)](#overfitting-prevention)
6. [QuantConnect Backtesting Integration](#quantconnect-integration)

---

## 1. CLASSic Framework (ICLR 2025) {#classic-framework}

### Overview

The CLASSic framework evaluates AI agents across five critical dimensions:
- **C**ost: Token usage, API calls, cost per decision
- **L**atency: Response times, SLA compliance
- **A**ccuracy: Precision, recall, F1 score
- **S**tability: Error rates, uptime, consistency
- **S**ecurity: Data leakage, unauthorized access

**Reference**: [CLASSic Framework - ICLR 2025 Workshop](https://aisera.com/ai-agents-evaluation/)

### Usage Example

```python
from evaluation.classic_evaluation import calculate_classic_metrics

# Calculate CLASSic metrics from test results
classic_metrics = calculate_classic_metrics(
    test_results=agent_test_results,
    cost_config={"opus-4": 0.015, "sonnet-4": 0.003},
    latency_sla_ms=1000.0,
    security_incidents={"data_leakage_incidents": 0},
)

# Generate report
from evaluation.classic_evaluation import generate_classic_report
report = generate_classic_report(classic_metrics)
print(report)
```

### Metrics Explained

| Dimension | Key Metrics | Target |
|-----------|-------------|--------|
| **Cost** | Cost per decision, Token usage | <$0.01 per decision |
| **Latency** | P95 response time, SLA compliance | <1000ms P95 |
| **Accuracy** | Precision, Recall, F1 Score | >90% accuracy |
| **Stability** | Error rate, MTBF | <5% error rate |
| **Security** | Incident count | Zero incidents |

### Overall CLASSic Score

Weighted average of all dimensions (0-100 scale):
- Cost: 15% weight
- Latency: 20% weight
- Accuracy: 35% weight
- Stability: 20% weight
- Security: 10% weight

**Thresholds**:
- ≥90: Excellent (Production ready)
- ≥80: Good (Production ready with minor improvements)
- ≥70: Acceptable (Requires improvements)
- <70: Needs work (Significant improvements required)

---

## 2. Walk-Forward Analysis {#walk-forward-analysis}

### Overview

Walk-forward analysis prevents overfitting by optimizing parameters on one data segment, then testing on the next out-of-sample segment. This simulates real-world conditions where future data is unknown.

**References**:
- [2025 Guide to Backtesting AI Trading](https://3commas.io/blog/comprehensive-2025-guide-to-backtesting-ai-trading)
- [Advanced Backtesting Techniques](https://www.golocalmag.com/advanced-trading-bot-backtesting-techniques-for-2025/)

### How It Works

```
Data Timeline:
|--- Train (6 mo) ---|--- Test (1 mo) ---|--- Train (6 mo) ---|--- Test (1 mo) ---|
     Optimize             Validate            Optimize             Validate
     params on             on OOS             new params            on OOS
     historical                               historical
```

### Usage Example

```python
from evaluation.walk_forward_analysis import run_walk_forward_analysis
from datetime import datetime

def optimize_params(train_start, train_end, parameter_space):
    # Your optimization logic
    return {"rsi_period": 14, "threshold": 30}

def evaluate_performance(test_start, test_end, params):
    # Your evaluation logic
    return {"sharpe_ratio": 1.5, "win_rate": 0.65}

# Run walk-forward analysis
result = run_walk_forward_analysis(
    data_start=datetime(2020, 1, 1),
    data_end=datetime(2024, 12, 31),
    train_window_months=6,
    test_window_months=1,
    optimization_func=optimize_params,
    evaluation_func=evaluate_performance,
    parameter_space={"rsi_period": [10, 14, 20], "threshold": [20, 30, 40]},
)

# Generate report
from evaluation.walk_forward_analysis import generate_walk_forward_report
print(generate_walk_forward_report(result))
```

### Production Readiness Criteria

✅ **Ready for Production**:
- Average degradation < 15%
- Robustness score > 0.80 (test/train Sharpe)
- Test Sharpe > 0.8
- Parameter consistency > 0.60

---

## 3. Advanced Trading Metrics {#advanced-trading-metrics}

### Overview

Professional-grade trading metrics beyond basic Sharpe/Sortino ratios.

**References**:
- [Evaluating Trading Bot Performance](https://yourrobotrader.com/en/evaluating-trading-bot-performance/)
- [Key Metrics 2025](https://www.utradealgos.com/blog/5-key-metrics-to-evaluate-the-performance-of-your-trading-algorithms)
- [Algorithmic Trading Metrics Deep Dive](https://sd-korp.com/algorithmic-trading-metrics-a-deep-dive-into-sharpe-sortino-and-more/)

### Key Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Expectancy** | Avg profit per trade | >$50 |
| **Profit Factor** | Gross profit / Gross loss | >1.5 |
| **Omega Ratio** | Probability-weighted gains/losses | >1.5 |
| **Win/Loss Ratio** | Avg win / Avg loss | >2.0 |
| **Recovery Factor** | Net profit / Max drawdown | >3.0 |
| **Ulcer Index** | Downside volatility measure | <10% |

### Usage Example

```python
from evaluation.advanced_trading_metrics import (
    calculate_advanced_trading_metrics,
    Trade,
)

# Create trade list
trades = [
    Trade(
        entry_date="2024-01-05",
        exit_date="2024-01-10",
        symbol="AAPL",
        pnl=250.00,
        pnl_pct=2.5,
        holding_period_days=5,
        result="win",
    ),
    # ... more trades
]

# Calculate metrics
metrics = calculate_advanced_trading_metrics(
    trades=trades,
    account_balance=100000,
    risk_free_rate=0.05,
)

# Generate report
from evaluation.advanced_trading_metrics import generate_trading_metrics_report
print(generate_trading_metrics_report(metrics))
```

### Professional Standards

**Criteria for Professional-Grade Strategy**:
- ✅ Profit Factor > 1.5
- ✅ Sharpe Ratio > 1.0
- ✅ Win Rate > 60%
- ✅ Max Drawdown < 20%
- ✅ Expectancy > $50/trade

**Pass**: Meet ≥3 of 5 criteria

---

## 4. Component-Level Evaluation {#component-level-evaluation}

### Overview

Tests individual subsystems (analysts, traders, risk managers) in isolation before integration testing. Based on DeepEval approach from 2025 research.

**References**:
- [LLM Agent Evaluation Complete Guide](https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide)
- [AI Agent Evaluation Metrics](https://www.getmaxim.ai/articles/ai-agent-evaluation-metrics-strategies-and-best-practices/)

### Component Types

```python
from evaluation.component_evaluation import ComponentType

# Available component types:
ComponentType.SIGNAL_GENERATION    # Analysts
ComponentType.POSITION_SIZING       # Traders
ComponentType.RISK_MANAGEMENT       # Risk Managers
ComponentType.DECISION_MAKING       # Supervisor
ComponentType.TOOL_SELECTION        # Multi-tool agents
ComponentType.CONTEXT_RETENTION     # Memory systems
ComponentType.ERROR_RECOVERY        # Failure handling
```

### Usage Example

```python
from evaluation.component_evaluation import (
    ComponentEvaluator,
    ComponentTestCase,
    ComponentType,
)

# Define component callable
def kelly_calculator(input_data):
    win_rate = input_data["win_rate"]
    avg_win = input_data["avg_win"]
    avg_loss = input_data["avg_loss"]
    fractional = input_data["fractional_kelly"]

    kelly_base = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    position_size = kelly_base * fractional

    return {"position_size_pct": position_size}

# Create test cases
test_cases = [
    ComponentTestCase(
        test_id="POS_001",
        component_type=ComponentType.POSITION_SIZING,
        component_name="ConservativeTrader.kelly_calculator",
        input_data={
            "win_rate": 0.68,
            "avg_win": 0.10,
            "avg_loss": 0.12,
            "fractional_kelly": 0.20,
        },
        expected_output={"position_size_pct": 0.027},
        success_criteria={
            "output_present": ["position_size_pct"],
            "value_range": {"position_size_pct": (0.024, 0.030)},
        },
        category="success",
    ),
]

# Evaluate component
evaluator = ComponentEvaluator(
    component_name="ConservativeTrader.kelly_calculator",
    component_type=ComponentType.POSITION_SIZING,
    component_callable=kelly_calculator,
    test_cases=test_cases,
)

result = evaluator.run()
print(f"Component ready: {result.component_ready}")
```

### Component Readiness Criteria

✅ **Component Ready for Integration**:
- Pass rate > 90%
- Error rate < 5%

---

## 5. Overfitting Prevention (QuantConnect Best Practices) {#overfitting-prevention}

### Overview

Implements QuantConnect's Research Guide best practices to detect and prevent overfitting.

**References**:
- [QuantConnect Research Guide](https://www.quantconnect.com/docs/v2/cloud-platform/backtesting/research-guide)
- [7 Tips for Fixing Your Strategy Backtesting](https://www.quantconnect.com/blog/7-tips-for-fixing-your-strategy-backtesting-a-qa-with-top-quants/)

### QuantConnect Best Practices

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| **Time Investment** | ≤16 hours | Excessive time = over-optimization |
| **Backtest Count** | ≤20 backtests | Too many = finding patterns by chance |
| **Parameter Count** | ≤5 parameters | More parameters = curve-fitting |
| **Out-of-Sample Period** | ≥12 months | Validate on recent unseen data |
| **Hypothesis Documented** | Required | Prevents data-driven fishing |

### Usage Example

```python
from evaluation.overfitting_prevention import calculate_overfitting_metrics

metrics = calculate_overfitting_metrics(
    parameter_count=3,
    backtest_count=8,
    time_invested_hours=12.5,
    hypothesis_documented=True,
    in_sample_sharpe=2.1,
    out_of_sample_sharpe=1.8,
    oos_period_months=12,
)

# Generate report
from evaluation.overfitting_prevention import generate_overfitting_report
print(generate_overfitting_report(metrics))
```

### Risk Levels

| Risk Level | Score | Action |
|------------|-------|--------|
| 🟢 **LOW** | <25 | Safe to proceed |
| 🟡 **MEDIUM** | 25-50 | Proceed with caution |
| 🟠 **HIGH** | 50-75 | Significant risk |
| 🔴 **CRITICAL** | >75 | Do NOT deploy |

---

## 6. QuantConnect Backtesting Integration {#quantconnect-integration}

### Overview

Integrates with QuantConnect's LEAN engine to run backtests and extract performance metrics.

**References**:
- [QuantConnect Backtesting](https://www.quantconnect.com/docs/v2/cloud-platform/backtesting)
- [QuantConnect Results](https://www.quantconnect.com/docs/v2/cloud-platform/backtesting/results)

### Usage Example

```python
from evaluation.quantconnect_integration import (
    BacktestConfig,
    run_quantconnect_backtest_cli,
)
from datetime import datetime
from pathlib import Path

# Configure backtest
config = BacktestConfig(
    algorithm_file=Path("algorithms/options_trading_bot.py"),
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 31),
    initial_cash=100000,
    data_resolution="Minute",
    brokerage="CharlesSchwab",
)

# Run backtest
result = run_quantconnect_backtest_cli(config)

# Generate report
from evaluation.quantconnect_integration import generate_backtest_report
print(generate_backtest_report(result))
```

### Backtest Success Criteria

✅ **Passed Backtest**:
- Sharpe Ratio > 1.0
- Max Drawdown < 20%
- Win Rate > 55%
- Profit Factor > 1.5
- Total Return > 10%

**Pass**: Meet ≥4 of 5 criteria

---

## Integration Example: Complete Evaluation Pipeline

```python
from datetime import datetime
from pathlib import Path

# 1. Component-Level Evaluation
from evaluation.component_evaluation import ComponentEvaluator, ComponentType
component_evaluator = ComponentEvaluator(...)
component_result = component_evaluator.run()

if not component_result.component_ready:
    print("❌ Component failed - fix before integration")
    exit(1)

# 2. Agent-Level Evaluation (STOCKBENCH)
from evaluation import AgentEvaluator
from evaluation.datasets import get_technical_analyst_cases
agent_evaluator = AgentEvaluator(
    agent_type="TechnicalAnalyst",
    version="v6.1",
    test_cases=get_technical_analyst_cases(),
    agent_callable=my_agent_callable,
)
agent_result = agent_evaluator.run()

if agent_result.pass_rate < 0.90:
    print("❌ Agent failed STOCKBENCH evaluation")
    exit(1)

# 3. CLASSic Framework Evaluation
from evaluation.classic_evaluation import calculate_classic_metrics
classic_metrics = calculate_classic_metrics(
    test_results=agent_result.test_results,
)

if classic_metrics.classic_score < 80:
    print("⚠️ CLASSic score below threshold")

# 4. Walk-Forward Analysis
from evaluation.walk_forward_analysis import run_walk_forward_analysis
wf_result = run_walk_forward_analysis(
    data_start=datetime(2020, 1, 1),
    data_end=datetime(2024, 12, 31),
    optimization_func=optimize_params,
    evaluation_func=evaluate_performance,
)

if not wf_result.production_ready:
    print("❌ Walk-forward analysis failed")
    exit(1)

# 5. Overfitting Prevention Check
from evaluation.overfitting_prevention import calculate_overfitting_metrics
overfitting_metrics = calculate_overfitting_metrics(
    parameter_count=3,
    backtest_count=12,
    time_invested_hours=14.0,
    hypothesis_documented=True,
    in_sample_sharpe=wf_result.avg_train_performance["sharpe_ratio"],
    out_of_sample_sharpe=wf_result.avg_test_performance["sharpe_ratio"],
    oos_period_months=12,
)

if overfitting_metrics.overfit_risk_level in ["HIGH", "CRITICAL"]:
    print("❌ High overfitting risk detected")
    exit(1)

# 6. QuantConnect Backtest
from evaluation.quantconnect_integration import BacktestConfig, run_quantconnect_backtest_cli
backtest_result = run_quantconnect_backtest_cli(
    config=BacktestConfig(
        algorithm_file=Path("algorithms/my_algorithm.py"),
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 12, 31),
    )
)

if backtest_result.sharpe_ratio < 1.0:
    print("❌ Backtest Sharpe ratio too low")
    exit(1)

print("✅ ALL EVALUATIONS PASSED - READY FOR PAPER TRADING")
```

---

## Summary of Evaluation Frameworks

| Framework | Purpose | Key Metric | Target |
|-----------|---------|------------|--------|
| **STOCKBENCH** | Agent test cases | Pass rate | >90% |
| **CLASSic** | Production readiness | CLASSic score | >80/100 |
| **Walk-Forward** | Out-of-sample validation | Degradation | <15% |
| **Advanced Metrics** | Trading quality | Profit Factor | >1.5 |
| **Component** | Subsystem testing | Pass rate | >90% |
| **Overfitting** | Risk detection | Risk score | <50/100 |
| **Backtest** | Historical performance | Sharpe Ratio | >1.0 |

---

## Sources

- [Agent Evaluation in 2025](https://orq.ai/blog/agent-evaluation)
- [AI Agent Evaluation Metrics](https://www.getmaxim.ai/articles/ai-agent-evaluation-metrics-strategies-and-best-practices/)
- [CLASSic Framework - ICLR 2025](https://aisera.com/ai-agents-evaluation/)
- [2025 Backtesting Guide](https://3commas.io/blog/comprehensive-2025-guide-to-backtesting-ai-trading)
- [Trading Metrics Deep Dive](https://sd-korp.com/algorithmic-trading-metrics-a-deep-dive-into-sharpe-sortino-and-more/)
- [QuantConnect Research Guide](https://www.quantconnect.com/docs/v2/cloud-platform/backtesting/research-guide)
- [LLM Agent Evaluation Survey 2025](https://arxiv.org/abs/2503.16416)
