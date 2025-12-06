# Evaluation Framework Upgrade Guide

## Gap Analysis: Research vs Implementation (December 2025)

This document analyzes gaps between the research findings in `docs/research/EVALUATION_FRAMEWORK_RESEARCH.md` and the current implementation, providing a prioritized upgrade roadmap.

---

## Executive Summary

| Category | Current State | Gap Severity | Priority |
|----------|---------------|--------------|----------|
| Test Datasets | 2024 data, mock scenarios | 🔴 Critical | P0 |
| STOCKBENCH Methodology | Basic structure, placeholders | 🔴 Critical | P0 |
| Drift Detection (PSI) | Basic thresholds only | 🟠 High | P1 |
| 2025 Benchmarks | Outdated thresholds | 🟠 High | P1 |
| Walk-Forward Analysis | Missing Monte Carlo | 🟡 Medium | P2 |
| CLASSic Security | Incident counting only | 🟡 Medium | P2 |
| Orchestration Pipeline | Placeholder integrations | 🟡 Medium | P2 |

---

## Critical Gaps (P0) - Immediate Action Required

### 1. Test Dataset Contamination Risk

**Research Finding (STOCKBENCH, October 2025)**:
- LLMs trained on pre-2024 data cannot be fairly tested on historical data
- STOCKBENCH uses **March-June 2025** market data (82 trading days)
- Top 20 DJIA stocks with $100,000 starting capital
- Up to 5 time-relevant news articles per stock (within 48 hours)

**Current Implementation**:
- Uses 2024 data with mock scenarios
- No actual market data integration
- No news data integration
- Static test cases without real market dynamics

**Gap**:
```
Research: March-June 2025 real market data
Current:  2024 Q3-Q4 mock scenarios
Risk:     LLM contamination, unreliable evaluation results
```

**Upgrade Required**:
```python
# New: evaluation/datasets/stockbench_2025.py
class StockBench2025Dataset:
    """
    STOCKBENCH-compliant dataset with 2025 market data.

    Attributes:
        market_period: "downturn" (Jan-Apr 2025) or "upturn" (May-Aug 2025)
        symbols: Top 20 DJIA stocks
        starting_capital: $100,000
        news_window_hours: 48
    """

    def __init__(self, market_period: str = "upturn"):
        self.market_period = market_period
        self.symbols = self._get_djia_top_20()
        self.trading_days = 82  # March-June 2025

    def get_test_cases(self) -> List[TestCase]:
        """Generate contamination-free test cases."""
        pass

    def get_news_context(self, symbol: str, date: datetime) -> List[NewsArticle]:
        """Get time-relevant news (within 48 hours)."""
        pass
```

---

### 2. STOCKBENCH Agent Workflow Missing

**Research Finding**:
- Agent workflow: Portfolio overview → Stock analysis → Decision generation → Order execution
- Model rankings shift between downturn and upturn periods
- LLM agents struggle to outperform buy-and-hold during downturns

**Current Implementation**:
- Basic TestCase structure without workflow stages
- No market period differentiation
- No buy-and-hold benchmark comparison

**Gap**:
```
Research: 4-stage agent workflow with market period awareness
Current:  Single-shot test case evaluation
Risk:     Missing realistic agent evaluation flow
```

**Upgrade Required**:
```python
# New: evaluation/stockbench_workflow.py
@dataclass
class StockBenchWorkflow:
    """STOCKBENCH 4-stage agent workflow."""

    portfolio_overview: PortfolioState
    stock_analysis: Dict[str, StockAnalysis]
    decisions: List[TradingDecision]
    executions: List[OrderExecution]

    # Market period awareness
    market_period: str  # "downturn" or "upturn"
    buy_and_hold_benchmark: float  # For comparison
```

---

## High Priority Gaps (P1) - This Sprint

### 3. Population Stability Index (PSI) Missing

**Research Finding (December 2025)**:
- PSI thresholds: <0.1 no drift, 0.1-0.25 growing drift, >0.25 significant shift
- AI strategy half-life: 11 months (2025) vs 18 months (2020)
- Models unchanged for 6+ months saw 35% error rate increase

**Current Implementation**:
- Basic drift detection with percentage thresholds
- No PSI calculation
- No strategy half-life tracking

**Gap**:
```
Research: PSI-based drift detection with 11-month half-life
Current:  Simple percentage-based thresholds
Risk:     Missing early drift warnings, strategy decay undetected
```

**Upgrade Required**:
```python
# Add to: evaluation/continuous_monitoring.py
def calculate_psi(expected: List[float], actual: List[float], bins: int = 10) -> float:
    """
    Calculate Population Stability Index.

    Thresholds (from 2025 research):
    - PSI < 0.10: No significant population shift
    - 0.10 <= PSI < 0.25: Moderate shift, investigate
    - PSI >= 0.25: Significant shift, action required
    """
    # Bin the distributions
    expected_pcts = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_pcts = np.histogram(actual, bins=bins)[0] / len(actual)

    # Calculate PSI
    psi = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
    return psi

@dataclass
class DriftMetrics:
    psi_score: float
    drift_level: str  # "none", "moderate", "significant"
    strategy_age_months: int
    estimated_decay_pct: float  # Based on 11-month half-life
```

---

### 4. 2025 Performance Benchmarks Update

**Research Finding**:
- Leading AI bots achieve Sharpe ratios 2.5-3.2 (not just 1.0-2.0)
- Successful bots: 85-90% positive monthly returns
- Best portfolios: Sharpe >2.5, max drawdown ~3%, near-zero market correlation
- Omega Ratio benchmark: ~1.15 meets "very good" threshold
- Profit Factor: >1.75 for dependable returns, beware >4.0 (overfitting signal)

**Current Implementation**:
- Sharpe thresholds: excellent=2.0, good=1.0, acceptable=0.5
- Profit factor: excellent=2.0, good=1.5, acceptable=1.2
- No overfitting warning for Profit Factor >4.0

**Gap**:
```
Research: Sharpe 2.5-3.2 for leading bots, PF >4.0 overfitting warning
Current:  Lower thresholds, no overfitting detection
Risk:     Underestimating requirements, missing overfitting signals
```

**Upgrade Required**:
```python
# Update: evaluation/advanced_trading_metrics.py
def get_trading_metrics_thresholds_2025() -> Dict[str, Dict[str, float]]:
    """Updated 2025 professional trading metrics thresholds."""
    return {
        "sharpe_ratio": {
            "excellent": 2.5,  # Updated from 2.0
            "good": 1.5,       # Updated from 1.0
            "acceptable": 1.0, # Updated from 0.5
            "leading_ai_bots": (2.5, 3.2),  # NEW: 2025 benchmark range
        },
        "profit_factor": {
            "excellent": 2.0,
            "good": 1.75,      # Updated per research
            "acceptable": 1.5,
            "overfitting_warning": 4.0,  # NEW: >4.0 suggests overfitting
        },
        "omega_ratio": {
            "excellent": 1.5,
            "good": 1.15,      # Updated per research benchmark
            "acceptable": 1.0,
        },
        "monthly_positive_pct": {
            "excellent": 0.90,  # NEW: 90% positive months
            "good": 0.85,       # NEW: 85% positive months
            "acceptable": 0.75,
        },
    }
```

---

## Medium Priority Gaps (P2) - Next Sprint

### 5. Walk-Forward Monte Carlo Integration

**Research Finding**:
- Combine walk-forward with Monte Carlo simulations
- Stress testing and sensitivity analysis recommended
- 70/30 train/validation split standard
- Parameter consistency threshold: >0.60

**Current Implementation**:
- Walk-forward only, no Monte Carlo
- Variable train/test splits
- No sensitivity analysis

**Upgrade Required**:
```python
# Add to: evaluation/walk_forward_analysis.py
def run_monte_carlo_walk_forward(
    data_start: datetime,
    data_end: datetime,
    n_simulations: int = 1000,
    confidence_level: float = 0.95,
) -> MonteCarloWalkForwardResult:
    """
    Walk-forward with Monte Carlo uncertainty quantification.

    Runs multiple walk-forward simulations with randomized:
    - Parameter perturbations
    - Market condition variations
    - Noise injection
    """
    pass
```

---

### 6. CLASSic Security Dimension Enhancement

**Research Finding (ICLR March 2025)**:
- Gemini 1.5 Pro refuses 78.5% of jailbreak prompts
- Claude 3.5 Sonnet refuses 99.8% of jailbreak prompts
- Security testing should include prompt injection, data leakage, sensitive exposure

**Current Implementation**:
- Only counts security incidents
- No actual jailbreak testing
- No prompt injection resistance testing

**Upgrade Required**:
```python
# Add to: evaluation/classic_evaluation.py
def run_security_evaluation(agent_callable: Callable) -> SecurityMetrics:
    """
    Run CLASSic security dimension tests.

    Tests:
    1. Prompt injection attempts (10 standard attacks)
    2. Jailbreak resistance (20 jailbreak prompts)
    3. Data leakage probing (5 extraction attempts)
    4. Sensitive data exposure (API keys, credentials)
    """
    pass
```

---

### 7. Orchestration Pipeline Real Integration

**Research Finding**:
- Retry with exponential backoff + jitter: 2s → 4s → 8s + random jitter
- Maximum 3 retries per framework
- Graceful degradation on non-critical failures
- Checkpoint/resume for long-running evaluations

**Current Implementation**:
- Placeholder functions for all frameworks
- No actual integration with evaluation modules
- Missing jitter in retry logic

**Upgrade Required**:
```python
# Update: evaluation/orchestration_pipeline.py

def _execute_with_retry_jitter(self, ...):
    """Execute with exponential backoff AND jitter."""
    delay = self.retry_delay_seconds

    while retry_count <= self.max_retries:
        try:
            # Execute framework
            return executor(dependency)
        except Exception as e:
            retry_count += 1
            # Add jitter: ±25% of delay
            jitter = delay * 0.25 * (random.random() * 2 - 1)
            sleep_time = delay + jitter
            time.sleep(sleep_time)
            delay *= 2  # Exponential backoff
```

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)

| Task | Files | Effort |
|------|-------|--------|
| Create STOCKBENCH 2025 dataset | `datasets/stockbench_2025.py` | 4h |
| Implement STOCKBENCH workflow | `stockbench_workflow.py` | 3h |
| Add PSI calculation | `continuous_monitoring.py` | 2h |
| Update 2025 thresholds | `advanced_trading_metrics.py` | 1h |

### Phase 2: High Priority (Week 2)

| Task | Files | Effort |
|------|-------|--------|
| Add overfitting detection | `advanced_trading_metrics.py` | 2h |
| Implement strategy half-life tracking | `continuous_monitoring.py` | 2h |
| Market period differentiation | `datasets/stockbench_2025.py` | 3h |

### Phase 3: Medium Priority (Week 3)

| Task | Files | Effort |
|------|-------|--------|
| Monte Carlo walk-forward | `walk_forward_analysis.py` | 4h |
| CLASSic security tests | `classic_evaluation.py` | 4h |
| Orchestration real integration | `orchestration_pipeline.py` | 4h |
| Add retry jitter | `orchestration_pipeline.py` | 1h |

---

## New Files to Create

```
evaluation/
├── datasets/
│   ├── stockbench_2025.py          # NEW: 2025 contamination-free data
│   └── market_periods.py           # NEW: Downturn/upturn period handling
├── stockbench_workflow.py          # NEW: 4-stage agent workflow
├── psi_drift_detection.py          # NEW: Population Stability Index
├── monte_carlo_analysis.py         # NEW: Monte Carlo simulations
└── security_evaluation.py          # NEW: CLASSic security tests
```

---

## Migration Notes

### Breaking Changes

1. **Threshold Updates**: `get_trading_metrics_thresholds()` will return updated values
   - Mitigation: Add `get_trading_metrics_thresholds_2024()` for backward compatibility

2. **Dataset Format**: STOCKBENCH 2025 datasets have different structure
   - Mitigation: Keep old datasets, add new loader for STOCKBENCH format

3. **Drift Detection**: PSI replaces percentage-based thresholds
   - Mitigation: Support both methods, PSI as default

### Backward Compatibility

```python
# Support both old and new threshold systems
def get_thresholds(version: str = "2025") -> Dict:
    if version == "2024":
        return get_trading_metrics_thresholds_legacy()
    return get_trading_metrics_thresholds_2025()
```

---

## Validation Checklist

After implementing upgrades, verify:

- [ ] STOCKBENCH 2025 dataset generates 82+ trading days of test cases
- [ ] Agent workflow includes all 4 stages
- [ ] PSI calculation matches research thresholds (<0.1, 0.1-0.25, >0.25)
- [ ] 2025 thresholds reflect leading AI bot benchmarks (Sharpe 2.5-3.2)
- [ ] Profit Factor >4.0 triggers overfitting warning
- [ ] Retry logic includes jitter
- [ ] Walk-forward supports Monte Carlo mode
- [ ] Security evaluation tests prompt injection resistance

---

## References

- [STOCKBENCH Paper (October 2025)](https://arxiv.org/abs/2510.02209)
- [CLASSic Framework (ICLR March 2025)](https://iclr.cc/virtual/2025/33362)
- [AI Strategy Half-Life Research (2025)](docs/research/EVALUATION_FRAMEWORK_RESEARCH.md)
- [PSI Drift Detection (2025)](https://labelyourdata.com/articles/machine-learning/data-drift)

---

**Document Version**: 1.0
**Created**: December 1, 2025
**Status**: Ready for Implementation
