# Evaluation Framework Upgrade Guide - December 2025

Comprehensive upgrade roadmap based on Phase 5 research findings. This guide identifies gaps in the current evaluation framework and provides implementation recommendations.

## 📋 Executive Summary

**Research Date**: December 1, 2025
**Current Framework Version**: 2.0 (7 integrated methodologies)
**Target Framework Version**: 3.0 (13 integrated methodologies)

### Key Findings

| Finding | Impact | Priority |
|---------|--------|----------|
| o3 achieves only 46.8% accuracy | Sets realistic expectations | HIGH |
| FINSABER 20-year degradation | Short-term tests misleading | HIGH |
| TCA missing from evaluation | Execution quality blind spot | CRITICAL |
| Agent-as-a-Judge paradigm | Better trajectory evaluation | MEDIUM |
| Multi-asset testing needed | Single-asset bias | MEDIUM |
| Real-time evaluation missing | Live performance gaps | HIGH |

---

## 🔴 Critical Gaps (Must Fix)

### Gap 1: Transaction Cost Analysis (TCA)

**Current State**: No execution quality evaluation

**Problem**: Algorithm can be "profitable" in backtests but lose money due to execution costs

**Research Source**: [TCA Best Execution Guide - Refinitiv (2024-2025)](https://www.lseg.com/en/data-analytics/pre-trade-post-trade-analytics)

**Implementation**:

```python
# evaluation/tca_evaluation.py

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from enum import Enum


class ExecutionQuality(Enum):
    EXCELLENT = "excellent"  # < 2 bps
    GOOD = "good"           # 2-5 bps
    FAIR = "fair"           # 5-10 bps
    POOR = "poor"           # > 10 bps


@dataclass
class ExecutionRecord:
    """Single execution for TCA analysis."""
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    decision_price: float      # Price when decision made
    arrival_price: float       # Price when order received
    execution_price: float     # Actual fill price
    vwap: float               # Market VWAP during execution
    execution_time: datetime
    market_volume: float       # Total market volume during execution
    order_volume: float        # Our order volume


@dataclass
class TCAMetrics:
    """Transaction Cost Analysis metrics."""
    # Primary benchmarks
    vwap_deviation_bps: float          # vs Volume-Weighted Average Price
    implementation_shortfall_bps: float # Decision price - Execution price
    arrival_cost_bps: float            # Arrival price - Execution price

    # Market impact
    market_impact_bps: float           # Price movement caused by our order
    realized_spread_bps: float         # Actual spread paid

    # Timing analysis
    timing_cost_bps: float             # Cost of waiting to execute
    opportunity_cost_bps: float        # Cost of not executing immediately

    # Aggregate
    total_cost_bps: float              # All-in execution cost
    execution_quality: ExecutionQuality

    # Compliance
    mifid_compliant: bool              # Meets best execution requirements
    audit_trail_complete: bool         # Full documentation available


def calculate_vwap_deviation(
    execution_price: float,
    vwap: float,
    side: str
) -> float:
    """Calculate deviation from VWAP in basis points."""
    if side == "buy":
        return ((execution_price - vwap) / vwap) * 10000
    else:
        return ((vwap - execution_price) / vwap) * 10000


def calculate_implementation_shortfall(
    decision_price: float,
    execution_price: float,
    side: str
) -> float:
    """Calculate implementation shortfall in basis points."""
    if side == "buy":
        return ((execution_price - decision_price) / decision_price) * 10000
    else:
        return ((decision_price - execution_price) / decision_price) * 10000


def calculate_market_impact(
    order_volume: float,
    market_volume: float,
    price_change_pct: float
) -> float:
    """Estimate market impact in basis points."""
    participation_rate = order_volume / market_volume if market_volume > 0 else 0
    # Square-root market impact model (standard in TCA)
    impact = price_change_pct * (participation_rate ** 0.5) * 10000
    return impact


def calculate_tca_metrics(
    executions: List[ExecutionRecord]
) -> TCAMetrics:
    """Calculate comprehensive TCA metrics."""
    if not executions:
        return TCAMetrics(
            vwap_deviation_bps=0.0,
            implementation_shortfall_bps=0.0,
            arrival_cost_bps=0.0,
            market_impact_bps=0.0,
            realized_spread_bps=0.0,
            timing_cost_bps=0.0,
            opportunity_cost_bps=0.0,
            total_cost_bps=0.0,
            execution_quality=ExecutionQuality.EXCELLENT,
            mifid_compliant=True,
            audit_trail_complete=True
        )

    # Volume-weighted averages
    total_volume = sum(e.quantity for e in executions)

    vwap_devs = []
    impl_shortfalls = []
    arrival_costs = []
    market_impacts = []

    for e in executions:
        weight = e.quantity / total_volume

        vwap_dev = calculate_vwap_deviation(e.execution_price, e.vwap, e.side)
        vwap_devs.append(vwap_dev * weight)

        impl_sf = calculate_implementation_shortfall(
            e.decision_price, e.execution_price, e.side
        )
        impl_shortfalls.append(impl_sf * weight)

        arr_cost = calculate_implementation_shortfall(
            e.arrival_price, e.execution_price, e.side
        )
        arrival_costs.append(arr_cost * weight)

        price_change = abs(e.execution_price - e.arrival_price) / e.arrival_price
        impact = calculate_market_impact(
            e.order_volume, e.market_volume, price_change
        )
        market_impacts.append(impact * weight)

    vwap_deviation = sum(vwap_devs)
    impl_shortfall = sum(impl_shortfalls)
    arrival_cost = sum(arrival_costs)
    market_impact = sum(market_impacts)

    # Derived metrics
    timing_cost = impl_shortfall - arrival_cost
    realized_spread = arrival_cost - market_impact
    total_cost = impl_shortfall

    # Quality classification
    if abs(total_cost) < 2:
        quality = ExecutionQuality.EXCELLENT
    elif abs(total_cost) < 5:
        quality = ExecutionQuality.GOOD
    elif abs(total_cost) < 10:
        quality = ExecutionQuality.FAIR
    else:
        quality = ExecutionQuality.POOR

    return TCAMetrics(
        vwap_deviation_bps=vwap_deviation,
        implementation_shortfall_bps=impl_shortfall,
        arrival_cost_bps=arrival_cost,
        market_impact_bps=market_impact,
        realized_spread_bps=realized_spread,
        timing_cost_bps=timing_cost,
        opportunity_cost_bps=0.0,  # Requires counterfactual analysis
        total_cost_bps=total_cost,
        execution_quality=quality,
        mifid_compliant=abs(total_cost) < 10,  # Simplified check
        audit_trail_complete=True
    )


# TCA Thresholds (Professional Standards)
TCA_THRESHOLDS = {
    "vwap_deviation_bps": 5.0,           # < 5 bps for liquid assets
    "implementation_shortfall_bps": 10.0, # < 10 bps optimal
    "market_impact_bps": 3.0,             # < 3 bps for small orders
    "market_impact_large_bps": 15.0,      # < 15 bps for large orders
    "total_cost_bps": 10.0,               # All-in threshold
}
```

**Thresholds**:

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| VWAP Deviation | < 5 bps | 5-10 bps | > 10 bps |
| Implementation Shortfall | < 10 bps | 10-20 bps | > 20 bps |
| Market Impact | < 3 bps | 3-10 bps | > 10 bps |
| Total Cost | < 10 bps | 10-25 bps | > 25 bps |

**Integration Point**: Add to orchestration pipeline after backtest evaluation

---

### Gap 2: Long-Horizon Backtesting (FINSABER)

**Current State**: Walk-forward uses 6-month windows

**Problem**: LLM strategies that look good in 2-year tests fail in 20-year tests

**Research Source**: [FINSABER (Published: 2025)](https://arxiv.org/abs/2502.17979)

**Implementation**:

```python
# evaluation/long_horizon_evaluation.py

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta


@dataclass
class HorizonResult:
    """Results for a specific time horizon."""
    horizon_years: int
    sharpe_ratio: float
    max_drawdown: float
    cagr: float
    win_rate: float
    profit_factor: float
    degradation_vs_short: float  # % degradation vs 2-year results


@dataclass
class LongHorizonMetrics:
    """Long-horizon backtesting metrics."""
    horizon_results: List[HorizonResult]
    degradation_rate: float          # Annual degradation %
    stable_horizon_years: int        # Years before significant degradation
    long_term_viability: bool        # Viable for > 10 years?
    regime_consistency: float        # Performance consistency across regimes


def run_long_horizon_backtest(
    algorithm_path: str,
    horizons: List[int] = [2, 5, 10, 15, 20]
) -> LongHorizonMetrics:
    """
    Run backtests across multiple time horizons.

    Per FINSABER research, LLM strategies often show:
    - Good 2-year performance
    - Degraded 5-year performance
    - Significantly worse 10+ year performance
    """
    results = []
    base_sharpe = None

    for years in horizons:
        # Run backtest for this horizon
        # (Implementation depends on QuantConnect integration)
        result = _run_single_horizon(algorithm_path, years)

        if base_sharpe is None:
            base_sharpe = result.sharpe_ratio

        degradation = 0.0
        if base_sharpe > 0:
            degradation = (base_sharpe - result.sharpe_ratio) / base_sharpe

        results.append(HorizonResult(
            horizon_years=years,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
            cagr=result.cagr,
            win_rate=result.win_rate,
            profit_factor=result.profit_factor,
            degradation_vs_short=degradation
        ))

    # Calculate degradation rate (linear fit)
    if len(results) >= 2:
        x = [r.horizon_years for r in results]
        y = [r.sharpe_ratio for r in results]
        # Simple linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_xx = sum(x[i] ** 2 for i in range(n))

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
        degradation_rate = abs(slope / results[0].sharpe_ratio) if results[0].sharpe_ratio > 0 else 0
    else:
        degradation_rate = 0.0

    # Find stable horizon (where Sharpe stays > 0.5)
    stable_horizon = 0
    for r in results:
        if r.sharpe_ratio >= 0.5:
            stable_horizon = r.horizon_years
        else:
            break

    # Long-term viability check
    long_term_viable = any(
        r.sharpe_ratio >= 0.5 and r.horizon_years >= 10
        for r in results
    )

    return LongHorizonMetrics(
        horizon_results=results,
        degradation_rate=degradation_rate,
        stable_horizon_years=stable_horizon,
        long_term_viability=long_term_viable,
        regime_consistency=_calculate_regime_consistency(results)
    )


# FINSABER Thresholds
LONG_HORIZON_THRESHOLDS = {
    "max_degradation_rate": 0.05,      # < 5% annual degradation
    "min_stable_horizon": 10,          # At least 10 years stable
    "min_long_sharpe": 0.5,            # Sharpe > 0.5 at 10+ years
    "min_regime_consistency": 0.70,    # 70% consistency across regimes
}
```

**Thresholds**:

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Degradation Rate | < 5%/year | 5-10%/year | > 10%/year |
| Stable Horizon | > 10 years | 5-10 years | < 5 years |
| Long-Term Sharpe | > 0.5 | 0.3-0.5 | < 0.3 |
| Regime Consistency | > 70% | 50-70% | < 50% |

---

### Gap 3: Real-Time Execution Quality Monitoring

**Current State**: Continuous monitoring tracks Sharpe/win rate drift only

**Problem**: Execution quality degrades in live trading but isn't detected

**Research Source**: [Agent Market Arena (2025)](https://arxiv.org/abs/2502.15574)

**Implementation**:

```python
# Add to evaluation/continuous_monitoring.py

@dataclass
class RealTimeExecutionMetrics:
    """Real-time execution quality tracking."""
    rolling_vwap_deviation_bps: float
    rolling_slippage_bps: float
    rolling_fill_rate: float
    rolling_latency_ms: float
    execution_quality_score: float  # 0-100

    # Alerts
    slippage_alert: bool
    latency_alert: bool
    fill_rate_alert: bool


class RealTimeExecutionMonitor:
    """Monitor execution quality in real-time."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.executions: List[ExecutionRecord] = []

        # Thresholds
        self.slippage_threshold_bps = 10.0
        self.latency_threshold_ms = 100.0
        self.fill_rate_threshold = 0.95

    def record_execution(self, execution: ExecutionRecord) -> None:
        """Record new execution and update metrics."""
        self.executions.append(execution)
        if len(self.executions) > self.window_size:
            self.executions.pop(0)

    def get_metrics(self) -> RealTimeExecutionMetrics:
        """Calculate current execution metrics."""
        if not self.executions:
            return self._empty_metrics()

        # Calculate rolling averages
        vwap_devs = [
            calculate_vwap_deviation(e.execution_price, e.vwap, e.side)
            for e in self.executions
        ]

        slippages = [
            calculate_implementation_shortfall(
                e.arrival_price, e.execution_price, e.side
            )
            for e in self.executions
        ]

        avg_vwap_dev = sum(vwap_devs) / len(vwap_devs)
        avg_slippage = sum(slippages) / len(slippages)

        # Quality score (0-100)
        quality_score = max(0, 100 - abs(avg_slippage) * 5)

        return RealTimeExecutionMetrics(
            rolling_vwap_deviation_bps=avg_vwap_dev,
            rolling_slippage_bps=avg_slippage,
            rolling_fill_rate=0.95,  # Would need fill tracking
            rolling_latency_ms=50.0,  # Would need latency tracking
            execution_quality_score=quality_score,
            slippage_alert=abs(avg_slippage) > self.slippage_threshold_bps,
            latency_alert=False,
            fill_rate_alert=False
        )
```

---

## 🟡 Important Gaps (Should Fix)

### Gap 4: Agent-as-a-Judge Evaluation

**Current State**: Rule-based evaluation only

**Problem**: Missing trajectory-level assessment that experts provide

**Research Source**: [Agent-as-a-Judge (2025)](https://arxiv.org/abs/2410.10934)

**Implementation**:

```python
# evaluation/agent_as_judge.py

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class DecisionQuality(Enum):
    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    CRITICAL_ERROR = 1


@dataclass
class TrajectoryStep:
    """Single step in agent decision trajectory."""
    timestamp: str
    observation: str
    reasoning: str
    action: str
    outcome: str
    tool_calls: List[str]


@dataclass
class TrajectoryEvaluation:
    """Evaluation of a complete decision trajectory."""
    trajectory_id: str
    steps: List[TrajectoryStep]

    # Step-level scores
    step_scores: List[DecisionQuality]

    # Trajectory-level metrics
    reasoning_coherence: float       # 0-1, logical flow
    information_utilization: float   # 0-1, used available info
    risk_awareness: float           # 0-1, considered risks
    execution_efficiency: float     # 0-1, minimal unnecessary steps
    outcome_alignment: float        # 0-1, achieved intended goal

    # Overall score
    trajectory_score: float         # 0-100

    # Judge reasoning
    judge_reasoning: str
    improvement_suggestions: List[str]


TRAJECTORY_EVALUATION_PROMPT = '''
You are an expert trading system evaluator. Analyze the following agent decision trajectory and provide a detailed evaluation.

## Trajectory to Evaluate

{trajectory_json}

## Evaluation Criteria

1. **Reasoning Coherence** (0-1): Is the logical flow sound? Does each step follow from the previous?

2. **Information Utilization** (0-1): Did the agent effectively use available market data, news, and indicators?

3. **Risk Awareness** (0-1): Did the agent properly consider position sizing, stop losses, and portfolio risk?

4. **Execution Efficiency** (0-1): Were unnecessary steps avoided? Was the decision timely?

5. **Outcome Alignment** (0-1): Did the agent achieve its stated objective?

## Required Output (JSON)

```json
{
    "step_scores": [1-5 for each step],
    "reasoning_coherence": 0.0-1.0,
    "information_utilization": 0.0-1.0,
    "risk_awareness": 0.0-1.0,
    "execution_efficiency": 0.0-1.0,
    "outcome_alignment": 0.0-1.0,
    "trajectory_score": 0-100,
    "judge_reasoning": "Detailed explanation...",
    "improvement_suggestions": ["Suggestion 1", "Suggestion 2"]
}
```

Provide your evaluation:
'''


async def evaluate_trajectory_with_judge(
    trajectory: List[TrajectoryStep],
    judge_model: str = "claude-3-5-sonnet-20241022"
) -> TrajectoryEvaluation:
    """
    Use LLM as judge to evaluate agent trajectory.

    Benefits over rule-based evaluation:
    - Captures nuanced reasoning quality
    - Assesses decision context appropriateness
    - Identifies subtle errors humans would catch
    - Aligns better with expert human evaluation
    """
    import json
    from anthropic import Anthropic

    client = Anthropic()

    trajectory_json = json.dumps([
        {
            "timestamp": s.timestamp,
            "observation": s.observation,
            "reasoning": s.reasoning,
            "action": s.action,
            "outcome": s.outcome,
            "tool_calls": s.tool_calls
        }
        for s in trajectory
    ], indent=2)

    prompt = TRAJECTORY_EVALUATION_PROMPT.format(trajectory_json=trajectory_json)

    response = client.messages.create(
        model=judge_model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse response
    result_json = json.loads(response.content[0].text)

    return TrajectoryEvaluation(
        trajectory_id=f"traj_{hash(trajectory_json) % 10000}",
        steps=trajectory,
        step_scores=[DecisionQuality(s) for s in result_json["step_scores"]],
        reasoning_coherence=result_json["reasoning_coherence"],
        information_utilization=result_json["information_utilization"],
        risk_awareness=result_json["risk_awareness"],
        execution_efficiency=result_json["execution_efficiency"],
        outcome_alignment=result_json["outcome_alignment"],
        trajectory_score=result_json["trajectory_score"],
        judge_reasoning=result_json["judge_reasoning"],
        improvement_suggestions=result_json["improvement_suggestions"]
    )
```

**Thresholds**:

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Trajectory Score | > 80 | 60-80 | < 60 |
| Reasoning Coherence | > 0.8 | 0.6-0.8 | < 0.6 |
| Risk Awareness | > 0.9 | 0.7-0.9 | < 0.7 |

---

### Gap 5: Multi-Asset Evaluation

**Current State**: Single-asset (equities/options) evaluation

**Problem**: Strategies may not generalize across asset classes

**Research Source**: [InvestorBench (ACL 2025)](https://arxiv.org/abs/2501.00174)

**Implementation**:

```python
# evaluation/multi_asset_evaluation.py

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum


class AssetClass(Enum):
    EQUITIES = "equities"
    OPTIONS = "options"
    CRYPTO = "crypto"
    ETF = "etf"
    FOREX = "forex"
    FUTURES = "futures"


@dataclass
class AssetClassResult:
    """Evaluation results for single asset class."""
    asset_class: AssetClass
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    pass_rate: float


@dataclass
class MultiAssetMetrics:
    """Multi-asset evaluation metrics."""
    asset_results: Dict[AssetClass, AssetClassResult]

    # Cross-asset metrics
    average_sharpe: float
    consistency_score: float    # How consistent across assets (0-1)
    weakest_asset: AssetClass
    strongest_asset: AssetClass

    # Diversification benefit
    correlation_matrix: Dict[str, Dict[str, float]]
    diversification_ratio: float


def run_multi_asset_evaluation(
    test_cases_by_asset: Dict[AssetClass, List],
    agent
) -> MultiAssetMetrics:
    """
    Evaluate agent across multiple asset classes.

    Per InvestorBench:
    - Stocks: Traditional equity analysis
    - Crypto: High volatility, 24/7 markets
    - ETFs: Portfolio construction, sector rotation
    """
    results = {}

    for asset_class, cases in test_cases_by_asset.items():
        # Run evaluation for this asset class
        result = _evaluate_asset_class(agent, cases, asset_class)
        results[asset_class] = result

    # Calculate cross-asset metrics
    sharpes = [r.sharpe_ratio for r in results.values()]
    avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0

    # Consistency: std dev of Sharpes (lower = more consistent)
    if len(sharpes) > 1:
        mean = avg_sharpe
        variance = sum((s - mean) ** 2 for s in sharpes) / len(sharpes)
        std_dev = variance ** 0.5
        consistency = max(0, 1 - std_dev / mean) if mean > 0 else 0
    else:
        consistency = 1.0

    # Find weakest/strongest
    sorted_results = sorted(results.items(), key=lambda x: x[1].sharpe_ratio)
    weakest = sorted_results[0][0]
    strongest = sorted_results[-1][0]

    return MultiAssetMetrics(
        asset_results=results,
        average_sharpe=avg_sharpe,
        consistency_score=consistency,
        weakest_asset=weakest,
        strongest_asset=strongest,
        correlation_matrix={},  # Would need return series
        diversification_ratio=1.0  # Placeholder
    )


# InvestorBench Thresholds
MULTI_ASSET_THRESHOLDS = {
    "min_consistency": 0.60,      # At least 60% consistent
    "min_avg_sharpe": 0.8,        # Average Sharpe across assets
    "max_asset_gap": 1.0,         # Max Sharpe difference between assets
}
```

---

### Gap 6: Realistic Expectations Calibration

**Current State**: Thresholds assume high success rates

**Problem**: Finance Agent Benchmark shows o3 achieves only 46.8% accuracy

**Research Source**: [Finance Agent Benchmark (November 2025)](https://paperswithcode.com/sota/financial-agent-benchmark)

**Implementation**:

Update thresholds to reflect realistic LLM capabilities:

```python
# evaluation/realistic_thresholds.py

"""
Realistic thresholds based on 2025 benchmark findings.

Key insight: Even the best LLMs (o3) achieve only 46.8% accuracy
on complex financial decisions at $3.79/query cost.

Implications:
1. Don't expect >60% accuracy on complex decisions
2. Focus on risk management over prediction accuracy
3. Ensemble approaches may help but won't solve fundamental limits
4. Cost-adjusted performance is critical
"""

# Updated 2025 Realistic Thresholds
REALISTIC_THRESHOLDS_2025 = {
    # Decision Accuracy (lowered from unrealistic expectations)
    "complex_decision_accuracy": {
        "excellent": 0.55,      # > 55% is exceptional for LLMs
        "good": 0.48,           # o3 level
        "acceptable": 0.40,     # Still usable with risk management
        "poor": 0.35,           # Below random with bias
    },

    # Simple Decision Accuracy (less complex decisions)
    "simple_decision_accuracy": {
        "excellent": 0.75,
        "good": 0.65,
        "acceptable": 0.55,
        "poor": 0.45,
    },

    # Cost Efficiency ($ per correct decision)
    "cost_per_correct_decision": {
        "excellent": 2.00,      # < $2 per correct decision
        "good": 4.00,           # $2-4
        "acceptable": 8.00,     # $4-8
        "poor": 15.00,          # > $8 (o3 at $3.79/query ÷ 0.468 = $8.10)
    },

    # Risk-Adjusted Returns (more important than accuracy)
    "risk_adjusted_strategy": {
        "sharpe_target": 1.5,   # Lower than 2.5-3.2 "leading" bots
        "max_drawdown": 0.15,   # 15% max (conservative)
        "win_rate": 0.50,       # 50% is acceptable with good R:R
        "profit_factor": 1.3,   # 1.3 is realistic for LLM strategies
    }
}


def calibrate_expectations(current_accuracy: float) -> Dict[str, Any]:
    """
    Calibrate expectations based on realistic benchmarks.

    Returns assessment and recommendations.
    """
    assessment = {
        "accuracy": current_accuracy,
        "percentile_vs_benchmarks": _calculate_percentile(current_accuracy),
        "is_realistic": current_accuracy <= 0.60,
        "recommendations": []
    }

    if current_accuracy > 0.60:
        assessment["recommendations"].append(
            "WARNING: Accuracy > 60% may indicate overfitting or test contamination. "
            "Finance Agent Benchmark shows even o3 achieves only 46.8%."
        )

    if current_accuracy > 0.55:
        assessment["recommendations"].append(
            "Accuracy exceeds o3 benchmark. Verify test cases are truly out-of-sample."
        )

    if current_accuracy < 0.40:
        assessment["recommendations"].append(
            "Consider ensemble approach or simpler decision framework."
        )

    return assessment
```

---

## 🟢 Nice-to-Have Enhancements

### Enhancement 1: Multi-Agent Debate Evaluation

Based on TradingAgents research showing debate improves decision quality.

```python
# evaluation/multi_agent_debate.py

@dataclass
class DebateRound:
    """Single round of bull vs bear debate."""
    round_number: int
    bull_argument: str
    bear_argument: str
    evidence_cited: List[str]


@dataclass
class DebateEvaluation:
    """Evaluate multi-agent debate quality."""
    rounds: List[DebateRound]

    # Quality metrics
    argument_diversity: float      # Different perspectives covered
    evidence_quality: float        # Factual basis of arguments
    synthesis_quality: float       # Final decision quality

    # Decision metrics
    final_decision: str
    confidence_before_debate: float
    confidence_after_debate: float
    decision_changed: bool
```

### Enhancement 2: Market Regime Classification

Automatic detection of market conditions for regime-aware evaluation.

```python
# evaluation/regime_detection.py

class MarketRegime(Enum):
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RANGE_BOUND = "range_bound"
    CRISIS = "crisis"


def detect_market_regime(
    returns: List[float],
    volatility: float,
    vix: float
) -> MarketRegime:
    """Classify current market regime."""
    # Implementation based on regime detection literature
    pass


def evaluate_by_regime(
    results: Dict[str, Any],
    regime: MarketRegime
) -> Dict[str, float]:
    """
    Adjust evaluation based on market regime.

    Key insight from STOCKBENCH: Model rankings shift between
    upturn and downturn periods.
    """
    pass
```

---

## 📊 Implementation Roadmap

### Phase 1: Critical Gaps (Week 1-2)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| TCA Module | CRITICAL | 3 days | Execution data |
| Long-Horizon Backtest | HIGH | 2 days | QuantConnect integration |
| Real-Time Execution Monitor | HIGH | 2 days | TCA Module |

### Phase 2: Important Gaps (Week 3-4)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Agent-as-a-Judge | MEDIUM | 3 days | LLM API access |
| Multi-Asset Evaluation | MEDIUM | 2 days | Test case datasets |
| Realistic Thresholds | MEDIUM | 1 day | None |

### Phase 3: Enhancements (Week 5+)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Multi-Agent Debate | LOW | 3 days | Multi-agent setup |
| Regime Detection | LOW | 2 days | Market data |
| Integration Testing | MEDIUM | 2 days | All above |

---

## 📋 Integration Checklist

### Pre-Integration

- [ ] Review current orchestration pipeline
- [ ] Identify data sources for TCA
- [ ] Set up long-horizon backtest environment
- [ ] Configure LLM API for Agent-as-a-Judge

### Integration Steps

1. [ ] Add TCA module to `evaluation/`
2. [ ] Update `orchestration_pipeline.py` to include TCA
3. [ ] Add long-horizon tests to CI/CD
4. [ ] Implement real-time execution monitoring
5. [ ] Add Agent-as-a-Judge to evaluation flow
6. [ ] Update thresholds with realistic values
7. [ ] Create multi-asset test datasets
8. [ ] Update documentation

### Post-Integration

- [ ] Run full evaluation pipeline
- [ ] Validate TCA metrics against manual calculations
- [ ] Compare Agent-as-a-Judge scores with human evaluation
- [ ] Document lessons learned

---

## 📝 Change Log

| Date | Change | Impact |
|------|--------|--------|
| 2025-12-01 | Guide created | Identified 6 critical/important gaps |
| 2025-12-01 | TCA implementation designed | Execution quality evaluation |
| 2025-12-01 | Long-horizon testing spec | FINSABER-inspired 20-year tests |
| 2025-12-01 | Agent-as-a-Judge spec | Trajectory-level evaluation |
| 2025-12-01 | Realistic thresholds defined | Based on o3 46.8% accuracy |

---

**Guide Version**: 1.0
**Created**: December 1, 2025
**Target Framework Version**: 3.0

**Sources**: All recommendations based on December 2025 research documented in [EVALUATION_FRAMEWORK_RESEARCH.md](EVALUATION_FRAMEWORK_RESEARCH.md)
