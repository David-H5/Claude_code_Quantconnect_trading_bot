"""
Test case datasets for agent evaluation.

Provides contamination-free test cases using 2024-2025 market data.
"""

from evaluation.datasets.analyst_cases import (
    get_sentiment_analyst_cases,
    get_technical_analyst_cases,
)
from evaluation.datasets.risk_manager_cases import (
    get_circuit_breaker_manager_cases,
    get_portfolio_risk_manager_cases,
    get_position_risk_manager_cases,
)
from evaluation.datasets.trader_cases import (
    get_aggressive_trader_cases,
    get_conservative_trader_cases,
    get_moderate_trader_cases,
)


__all__ = [
    "get_aggressive_trader_cases",
    "get_circuit_breaker_manager_cases",
    "get_conservative_trader_cases",
    "get_moderate_trader_cases",
    "get_portfolio_risk_manager_cases",
    "get_position_risk_manager_cases",
    "get_sentiment_analyst_cases",
    "get_technical_analyst_cases",
]
