"""
STOCKBENCH 2025 Contamination-Free Test Dataset.

Implements the STOCKBENCH methodology for contamination-free LLM agent testing
using March-June 2025 market data that is guaranteed to be after LLM training cutoffs.

Key Features:
- 82 trading days (March-June 2025)
- Top 20 DJIA stocks with $100,000 starting capital
- Up to 5 time-relevant news articles per stock (within 48 hours)
- Market period awareness (downturn vs upturn)
- 4-stage agent workflow evaluation

Reference: https://arxiv.org/abs/2510.02209 (Published: October 2, 2025)
Reference: https://stockbench.github.io/

Note: This module provides the structure and templates for STOCKBENCH-compliant
test cases. Actual market data should be loaded from QuantConnect or other sources.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

from evaluation.evaluation_framework import TestCase


class MarketPeriod(Enum):
    """Market period classification based on STOCKBENCH research."""

    DOWNTURN = "downturn"  # Jan-Apr 2025: Model rankings shift
    UPTURN = "upturn"  # May-Aug 2025: Different performance patterns
    NEUTRAL = "neutral"  # Sideways market


class WorkflowStage(Enum):
    """STOCKBENCH 4-stage agent workflow."""

    PORTFOLIO_OVERVIEW = "portfolio_overview"
    STOCK_ANALYSIS = "stock_analysis"
    DECISION_GENERATION = "decision_generation"
    ORDER_EXECUTION = "order_execution"


@dataclass
class NewsArticle:
    """Time-relevant news article for stock analysis."""

    headline: str
    source: str
    published_at: datetime
    sentiment_score: float  # -1.0 to 1.0
    relevance_score: float  # 0.0 to 1.0
    summary: str


@dataclass
class StockAnalysis:
    """Stock analysis data for STOCKBENCH evaluation."""

    symbol: str
    date: date
    open_price: float
    close_price: float
    high: float
    low: float
    volume: int
    daily_return_pct: float
    volatility_5d: float
    volatility_20d: float
    rsi_14: float
    macd_signal: str  # "bullish", "bearish", "neutral"
    news_articles: list[NewsArticle] = field(default_factory=list)


@dataclass
class PortfolioState:
    """Portfolio state for workflow evaluation."""

    cash: float
    positions: dict[str, int]  # symbol -> shares
    position_values: dict[str, float]  # symbol -> market value
    total_value: float
    daily_pnl: float
    total_pnl: float
    drawdown_pct: float


@dataclass
class TradingDecision:
    """Trading decision from agent workflow."""

    symbol: str
    action: str  # "buy", "sell", "hold"
    quantity: int
    reason: str
    confidence: float
    risk_assessment: str
    workflow_stage: WorkflowStage = WorkflowStage.DECISION_GENERATION


@dataclass
class OrderExecution:
    """Order execution result."""

    symbol: str
    action: str
    quantity: int
    fill_price: float
    commission: float
    slippage_pct: float
    execution_time_ms: float
    success: bool


@dataclass
class StockBenchWorkflow:
    """
    Complete STOCKBENCH 4-stage agent workflow.

    Evaluates agent through:
    1. Portfolio Overview - Understanding current state
    2. Stock Analysis - Analyzing market data and news
    3. Decision Generation - Making trading decisions
    4. Order Execution - Executing trades
    """

    portfolio_overview: PortfolioState
    stock_analyses: dict[str, StockAnalysis]
    decisions: list[TradingDecision]
    executions: list[OrderExecution]

    # Market context
    market_period: MarketPeriod
    trading_date: date
    buy_and_hold_benchmark: float  # For comparison

    # Workflow metrics
    workflow_completion_rate: float  # % of stages completed
    decision_quality_score: float  # 0-1
    execution_efficiency: float  # 0-1


# DJIA Top 20 Stocks (as of 2025)
DJIA_TOP_20 = [
    "AAPL",
    "MSFT",
    "UNH",
    "GS",
    "HD",
    "CAT",
    "AMGN",
    "V",
    "MCD",
    "CRM",
    "TRV",
    "AXP",
    "HON",
    "BA",
    "IBM",
    "JPM",
    "JNJ",
    "MMM",
    "WMT",
    "DIS",
]

# STOCKBENCH Test Period Constants
STOCKBENCH_START_DATE = date(2025, 3, 1)
STOCKBENCH_END_DATE = date(2025, 6, 30)
STOCKBENCH_TRADING_DAYS = 82
STOCKBENCH_STARTING_CAPITAL = 100000.0
NEWS_WINDOW_HOURS = 48


def get_market_period(trading_date: date) -> MarketPeriod:
    """
    Determine market period based on STOCKBENCH research.

    Research finding: Model rankings shift between downturn (Jan-Apr 2025)
    and upturn (May-Aug 2025) periods.

    Args:
        trading_date: Date to classify

    Returns:
        MarketPeriod enum value
    """
    if trading_date.month >= 1 and trading_date.month <= 4:
        return MarketPeriod.DOWNTURN
    elif trading_date.month >= 5 and trading_date.month <= 8:
        return MarketPeriod.UPTURN
    else:
        return MarketPeriod.NEUTRAL


def create_stockbench_test_case(
    case_id: str,
    category: str,
    agent_type: str,
    trading_date: date,
    scenario_description: str,
    portfolio_state: PortfolioState,
    stock_analyses: dict[str, StockAnalysis],
    expected_decisions: list[TradingDecision],
    success_criteria: dict[str, Any],
) -> TestCase:
    """
    Create a STOCKBENCH-compliant test case.

    Args:
        case_id: Unique test case identifier
        category: "success", "edge", or "failure"
        agent_type: Type of agent being tested
        trading_date: Date for the test scenario
        scenario_description: Human-readable scenario description
        portfolio_state: Current portfolio state
        stock_analyses: Dict of symbol -> StockAnalysis
        expected_decisions: Expected trading decisions
        success_criteria: Criteria for passing the test

    Returns:
        TestCase compatible with evaluation framework
    """
    market_period = get_market_period(trading_date)

    # Build input data
    input_data = {
        "trading_date": trading_date.isoformat(),
        "market_period": market_period.value,
        "portfolio": {
            "cash": portfolio_state.cash,
            "positions": portfolio_state.positions,
            "total_value": portfolio_state.total_value,
            "drawdown_pct": portfolio_state.drawdown_pct,
        },
        "stock_analyses": {
            symbol: {
                "close_price": analysis.close_price,
                "daily_return_pct": analysis.daily_return_pct,
                "volatility_20d": analysis.volatility_20d,
                "rsi_14": analysis.rsi_14,
                "macd_signal": analysis.macd_signal,
                "news_count": len(analysis.news_articles),
                "news_sentiment_avg": (
                    sum(n.sentiment_score for n in analysis.news_articles) / len(analysis.news_articles)
                    if analysis.news_articles
                    else 0.0
                ),
            }
            for symbol, analysis in stock_analyses.items()
        },
    }

    # Build expected output
    expected_output = {
        "decisions": [
            {
                "symbol": d.symbol,
                "action": d.action,
                "quantity": d.quantity,
                "confidence": d.confidence,
            }
            for d in expected_decisions
        ],
        "workflow_complete": True,
        "market_period_aware": True,
    }

    return TestCase(
        case_id=case_id,
        category=category,
        agent_type=agent_type,
        scenario=f"[{market_period.value.upper()}] {scenario_description}",
        input_data=input_data,
        expected_output=expected_output,
        success_criteria=success_criteria,
    )


def get_stockbench_2025_cases(
    agent_type: str = "TradingAgent",
    market_period: MarketPeriod | None = None,
) -> list[TestCase]:
    """
    Get STOCKBENCH-compliant test cases for 2025.

    Generates test cases across:
    - 40% Success cases (clear opportunities)
    - 40% Edge cases (challenging scenarios)
    - 20% Failure cases (should avoid action)

    Args:
        agent_type: Type of agent being tested
        market_period: Filter by market period (None = all)

    Returns:
        List of TestCase objects
    """
    cases = []

    # ========== SUCCESS CASES (40%) ==========

    # Success: Clear bullish opportunity in upturn market
    cases.append(
        create_stockbench_test_case(
            case_id="SB2025_SUCCESS_001",
            category="success",
            agent_type=agent_type,
            trading_date=date(2025, 5, 15),
            scenario_description="AAPL earnings beat with positive guidance, RSI=45, clear entry",
            portfolio_state=PortfolioState(
                cash=50000.0,
                positions={"MSFT": 100},
                position_values={"MSFT": 40000.0},
                total_value=90000.0,
                daily_pnl=500.0,
                total_pnl=-10000.0,
                drawdown_pct=0.10,
            ),
            stock_analyses={
                "AAPL": StockAnalysis(
                    symbol="AAPL",
                    date=date(2025, 5, 15),
                    open_price=178.50,
                    close_price=185.20,
                    high=186.00,
                    low=177.80,
                    volume=95000000,
                    daily_return_pct=3.75,
                    volatility_5d=0.018,
                    volatility_20d=0.022,
                    rsi_14=45.0,
                    macd_signal="bullish",
                    news_articles=[
                        NewsArticle(
                            headline="Apple Reports Record Q2 Earnings, Raises Guidance",
                            source="Reuters",
                            published_at=datetime(2025, 5, 14, 16, 30),
                            sentiment_score=0.85,
                            relevance_score=0.95,
                            summary="Apple beat earnings estimates by 12%.",
                        ),
                    ],
                ),
            },
            expected_decisions=[
                TradingDecision(
                    symbol="AAPL",
                    action="buy",
                    quantity=50,
                    reason="Earnings beat with positive guidance, technical setup favorable",
                    confidence=0.75,
                    risk_assessment="moderate",
                ),
            ],
            success_criteria={
                "signal_correct": True,
                "confidence_within_range": [0.65, 0.85],
                "position_size_appropriate": True,
            },
        )
    )

    # Success: Bearish setup in downturn market
    cases.append(
        create_stockbench_test_case(
            case_id="SB2025_SUCCESS_002",
            category="success",
            agent_type=agent_type,
            trading_date=date(2025, 3, 10),
            scenario_description="Market downturn, BA with negative news, RSI=72 overbought",
            portfolio_state=PortfolioState(
                cash=30000.0,
                positions={"BA": 100, "MSFT": 50},
                position_values={"BA": 25000.0, "MSFT": 20000.0},
                total_value=75000.0,
                daily_pnl=-1500.0,
                total_pnl=-25000.0,
                drawdown_pct=0.25,
            ),
            stock_analyses={
                "BA": StockAnalysis(
                    symbol="BA",
                    date=date(2025, 3, 10),
                    open_price=255.00,
                    close_price=248.50,
                    high=256.20,
                    low=247.00,
                    volume=12000000,
                    daily_return_pct=-2.55,
                    volatility_5d=0.035,
                    volatility_20d=0.042,
                    rsi_14=72.0,
                    macd_signal="bearish",
                    news_articles=[
                        NewsArticle(
                            headline="Boeing Faces New Safety Investigation",
                            source="WSJ",
                            published_at=datetime(2025, 3, 9, 14, 0),
                            sentiment_score=-0.75,
                            relevance_score=0.90,
                            summary="FAA launches investigation into manufacturing.",
                        ),
                    ],
                ),
            },
            expected_decisions=[
                TradingDecision(
                    symbol="BA",
                    action="sell",
                    quantity=50,
                    reason="Negative news, overbought RSI, downturn market - reduce exposure",
                    confidence=0.70,
                    risk_assessment="high",
                ),
            ],
            success_criteria={
                "signal_correct": True,
                "risk_awareness": True,
                "market_period_considered": True,
            },
        )
    )

    # ========== EDGE CASES (40%) ==========

    # Edge: Mixed signals in upturn market
    cases.append(
        create_stockbench_test_case(
            case_id="SB2025_EDGE_001",
            category="edge",
            agent_type=agent_type,
            trading_date=date(2025, 6, 5),
            scenario_description="NVDA positive news but overbought RSI=78, high volatility",
            portfolio_state=PortfolioState(
                cash=40000.0,
                positions={"NVDA": 20},
                position_values={"NVDA": 30000.0},
                total_value=70000.0,
                daily_pnl=2000.0,
                total_pnl=-30000.0,
                drawdown_pct=0.30,
            ),
            stock_analyses={
                "NVDA": StockAnalysis(
                    symbol="NVDA",
                    date=date(2025, 6, 5),
                    open_price=145.00,
                    close_price=152.00,
                    high=154.50,
                    low=144.20,
                    volume=85000000,
                    daily_return_pct=4.83,
                    volatility_5d=0.045,
                    volatility_20d=0.055,
                    rsi_14=78.0,
                    macd_signal="bullish",
                    news_articles=[
                        NewsArticle(
                            headline="NVIDIA AI Chips in High Demand",
                            source="Bloomberg",
                            published_at=datetime(2025, 6, 4, 10, 0),
                            sentiment_score=0.80,
                            relevance_score=0.85,
                            summary="Data center demand continues to surge.",
                        ),
                        NewsArticle(
                            headline="Analysts Warn of NVDA Valuation Concerns",
                            source="MarketWatch",
                            published_at=datetime(2025, 6, 5, 8, 0),
                            sentiment_score=-0.30,
                            relevance_score=0.75,
                            summary="PE ratio at historic highs.",
                        ),
                    ],
                ),
            },
            expected_decisions=[
                TradingDecision(
                    symbol="NVDA",
                    action="hold",
                    quantity=0,
                    reason="Mixed signals: positive news but overbought, wait for pullback",
                    confidence=0.55,
                    risk_assessment="high",
                ),
            ],
            success_criteria={
                "signal_correct": True,
                "mixed_signal_recognition": True,
            },
        )
    )

    # Edge: Downturn market with oversold stock
    cases.append(
        create_stockbench_test_case(
            case_id="SB2025_EDGE_002",
            category="edge",
            agent_type=agent_type,
            trading_date=date(2025, 2, 20),
            scenario_description="Market downturn but JPM oversold RSI=25, value opportunity",
            portfolio_state=PortfolioState(
                cash=60000.0,
                positions={},
                position_values={},
                total_value=60000.0,
                daily_pnl=-500.0,
                total_pnl=-40000.0,
                drawdown_pct=0.40,
            ),
            stock_analyses={
                "JPM": StockAnalysis(
                    symbol="JPM",
                    date=date(2025, 2, 20),
                    open_price=165.00,
                    close_price=158.50,
                    high=166.20,
                    low=157.80,
                    volume=18000000,
                    daily_return_pct=-3.94,
                    volatility_5d=0.040,
                    volatility_20d=0.035,
                    rsi_14=25.0,
                    macd_signal="neutral",
                    news_articles=[
                        NewsArticle(
                            headline="Banking Sector Under Pressure",
                            source="CNBC",
                            published_at=datetime(2025, 2, 19, 15, 0),
                            sentiment_score=-0.40,
                            relevance_score=0.80,
                            summary="Interest rate uncertainty weighs on banks.",
                        ),
                    ],
                ),
            },
            expected_decisions=[
                TradingDecision(
                    symbol="JPM",
                    action="hold",
                    quantity=0,
                    reason="Oversold but market downturn - avoid catching falling knife",
                    confidence=0.50,
                    risk_assessment="high",
                ),
            ],
            success_criteria={
                "market_period_awareness": True,
                "signal_correct": True,
            },
        )
    )

    # ========== FAILURE CASES (20%) ==========

    # Failure: Should avoid action during high VIX
    cases.append(
        create_stockbench_test_case(
            case_id="SB2025_FAIL_001",
            category="failure",
            agent_type=agent_type,
            trading_date=date(2025, 4, 5),
            scenario_description="VIX spike >35, all signals unreliable",
            portfolio_state=PortfolioState(
                cash=80000.0,
                positions={"SPY": 50},
                position_values={"SPY": 20000.0},
                total_value=100000.0,
                daily_pnl=-5000.0,
                total_pnl=0.0,
                drawdown_pct=0.05,
            ),
            stock_analyses={
                "AAPL": StockAnalysis(
                    symbol="AAPL",
                    date=date(2025, 4, 5),
                    open_price=175.00,
                    close_price=168.00,
                    high=176.50,
                    low=165.00,
                    volume=150000000,
                    daily_return_pct=-4.00,
                    volatility_5d=0.065,
                    volatility_20d=0.045,
                    rsi_14=35.0,
                    macd_signal="bearish",
                    news_articles=[],
                ),
            },
            expected_decisions=[
                TradingDecision(
                    symbol="AAPL",
                    action="hold",
                    quantity=0,
                    reason="High volatility environment - all signals unreliable",
                    confidence=0.30,
                    risk_assessment="extreme",
                ),
            ],
            success_criteria={
                "no_new_positions": True,
                "volatility_awareness": True,
            },
        )
    )

    # Filter by market period if specified
    if market_period:
        cases = [c for c in cases if market_period.value in c.scenario.lower()]

    return cases


def get_stockbench_downturn_cases(agent_type: str = "TradingAgent") -> list[TestCase]:
    """
    Get test cases specifically for downturn market periods.

    Research finding: LLM agents struggle to outperform buy-and-hold during downturns.

    Args:
        agent_type: Type of agent being tested

    Returns:
        List of TestCase objects for downturn scenarios
    """
    return get_stockbench_2025_cases(agent_type, MarketPeriod.DOWNTURN)


def get_stockbench_upturn_cases(agent_type: str = "TradingAgent") -> list[TestCase]:
    """
    Get test cases specifically for upturn market periods.

    Research finding: Model rankings shift significantly in upturn markets.

    Args:
        agent_type: Type of agent being tested

    Returns:
        List of TestCase objects for upturn scenarios
    """
    return get_stockbench_2025_cases(agent_type, MarketPeriod.UPTURN)


def validate_stockbench_compliance(
    test_cases: list[TestCase],
) -> dict[str, Any]:
    """
    Validate that test cases meet STOCKBENCH methodology requirements.

    Args:
        test_cases: List of test cases to validate

    Returns:
        Dict with validation results
    """
    issues = []

    # Check date range
    for case in test_cases:
        if "trading_date" in case.input_data:
            case_date = date.fromisoformat(case.input_data["trading_date"])
            if case_date < STOCKBENCH_START_DATE or case_date > STOCKBENCH_END_DATE:
                issues.append(
                    f"{case.case_id}: Date {case_date} outside STOCKBENCH range "
                    f"({STOCKBENCH_START_DATE} to {STOCKBENCH_END_DATE})"
                )

    # Check category distribution
    categories = [c.category for c in test_cases]
    success_pct = categories.count("success") / len(categories) if categories else 0
    edge_pct = categories.count("edge") / len(categories) if categories else 0
    failure_pct = categories.count("failure") / len(categories) if categories else 0

    if not (0.35 <= success_pct <= 0.45):
        issues.append(f"Success cases {success_pct:.1%} outside target range (40%)")
    if not (0.35 <= edge_pct <= 0.45):
        issues.append(f"Edge cases {edge_pct:.1%} outside target range (40%)")
    if not (0.15 <= failure_pct <= 0.25):
        issues.append(f"Failure cases {failure_pct:.1%} outside target range (20%)")

    # Check market period coverage
    market_periods = set()
    for case in test_cases:
        if "market_period" in case.input_data:
            market_periods.add(case.input_data["market_period"])

    if "downturn" not in market_periods:
        issues.append("Missing downturn market period test cases")
    if "upturn" not in market_periods:
        issues.append("Missing upturn market period test cases")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "stats": {
            "total_cases": len(test_cases),
            "success_pct": success_pct,
            "edge_pct": edge_pct,
            "failure_pct": failure_pct,
            "market_periods": list(market_periods),
        },
    }
