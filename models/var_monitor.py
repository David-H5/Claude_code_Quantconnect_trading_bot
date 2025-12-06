"""
Real-Time Value at Risk (VaR) Monitor

Implements multiple VaR calculation methods:
- Parametric (variance-covariance)
- Historical simulation
- Monte Carlo simulation

Provides continuous risk monitoring with limit alerts.

Part of UPGRADE-010 Sprint 4: Risk & Execution.
"""

import logging
import math
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class VaRMethod(Enum):
    """VaR calculation methods."""

    PARAMETRIC = "parametric"  # Assumes normal distribution
    HISTORICAL = "historical"  # Uses historical returns distribution
    MONTE_CARLO = "monte_carlo"  # Simulation-based


class RiskLevel(Enum):
    """Risk level classification."""

    LOW = "low"
    MODERATE = "moderate"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class VaRResult:
    """Value at Risk calculation result."""

    var_95: float  # 95% VaR ($ amount at risk)
    var_99: float  # 99% VaR
    cvar_95: float  # Conditional VaR (expected shortfall) at 95%
    cvar_99: float  # Conditional VaR at 99%
    var_95_pct: float  # VaR as % of portfolio
    var_99_pct: float
    method: VaRMethod
    confidence: float  # Confidence in calculation
    calculation_time_ms: float
    positions_included: int
    portfolio_value: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "var_95_pct": self.var_95_pct,
            "var_99_pct": self.var_99_pct,
            "method": self.method.value,
            "confidence": self.confidence,
            "calculation_time_ms": self.calculation_time_ms,
            "positions_included": self.positions_included,
            "portfolio_value": self.portfolio_value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class VaRLimits:
    """VaR-based risk limits."""

    max_var_pct: float = 0.05  # Max 5% daily VaR
    max_cvar_pct: float = 0.08  # Max 8% CVaR
    warning_threshold: float = 0.7  # Warn at 70% of limit
    critical_threshold: float = 0.9  # Critical at 90% of limit

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_var_pct": self.max_var_pct,
            "max_cvar_pct": self.max_cvar_pct,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
        }


@dataclass
class VaRAlert:
    """Alert for VaR limit breach."""

    level: RiskLevel
    message: str
    current_var_pct: float
    limit_var_pct: float
    utilization_pct: float
    positions_contributing: list[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "message": self.message,
            "current_var_pct": self.current_var_pct,
            "limit_var_pct": self.limit_var_pct,
            "utilization_pct": self.utilization_pct,
            "positions_contributing": self.positions_contributing,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PositionRisk:
    """Risk metrics for a single position."""

    symbol: str
    market_value: float
    weight: float  # Portfolio weight
    volatility: float  # Annualized volatility
    var_contribution: float  # Contribution to portfolio VaR
    var_contribution_pct: float  # % of total VaR
    beta: float = 1.0  # Beta to portfolio
    correlation: float = 1.0  # Correlation to portfolio

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "market_value": self.market_value,
            "weight": self.weight,
            "volatility": self.volatility,
            "var_contribution": self.var_contribution,
            "var_contribution_pct": self.var_contribution_pct,
            "beta": self.beta,
            "correlation": self.correlation,
        }


class VaRMonitor:
    """
    Real-time Value at Risk monitoring.

    Calculates portfolio VaR using multiple methods and monitors
    against risk limits.
    """

    # Trading days per year
    TRADING_DAYS = 252

    # Z-scores for confidence levels
    Z_SCORES = {
        0.90: 1.282,
        0.95: 1.645,
        0.99: 2.326,
    }

    def __init__(
        self,
        limits: VaRLimits | None = None,
        lookback_days: int = 252,
        min_history_days: int = 30,
    ):
        """
        Initialize VaR monitor.

        Args:
            limits: VaR limits configuration
            lookback_days: Days of history for calculations
            min_history_days: Minimum days required for calculation
        """
        self.limits = limits or VaRLimits()
        self.lookback_days = lookback_days
        self.min_history_days = min_history_days

        # Historical returns by symbol
        self.returns_history: dict[str, list[float]] = defaultdict(list)

        # Position tracking
        self._positions: dict[str, float] = {}  # symbol -> market_value

        # Alert callbacks
        self._alert_callbacks: list[Callable[[VaRAlert], None]] = []

        # Latest calculation
        self._latest_var: VaRResult | None = None

    def update_returns(self, symbol: str, daily_return: float) -> None:
        """
        Update historical returns for a symbol.

        Args:
            symbol: Symbol
            daily_return: Daily return (e.g., 0.01 for 1%)
        """
        self.returns_history[symbol].append(daily_return)

        # Keep history manageable
        if len(self.returns_history[symbol]) > self.lookback_days:
            self.returns_history[symbol] = self.returns_history[symbol][-self.lookback_days :]

    def update_position(self, symbol: str, market_value: float) -> None:
        """
        Update position market value.

        Args:
            symbol: Symbol
            market_value: Current market value
        """
        if market_value == 0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = market_value

    def calculate_var(
        self,
        method: VaRMethod = VaRMethod.HISTORICAL,
        confidence_level: float = 0.95,
        time_horizon_days: int = 1,
    ) -> VaRResult:
        """
        Calculate portfolio VaR.

        Args:
            method: VaR calculation method
            confidence_level: Confidence level (0.90, 0.95, 0.99)
            time_horizon_days: Time horizon in days

        Returns:
            VaRResult with VaR metrics
        """
        start_time = time.time()

        portfolio_value = sum(self._positions.values())
        if portfolio_value == 0:
            return self._empty_var_result(method)

        # Get portfolio returns
        portfolio_returns = self._calculate_portfolio_returns()

        if len(portfolio_returns) < self.min_history_days:
            return self._estimated_var_result(portfolio_value, method, confidence_level)

        # Calculate based on method
        if method == VaRMethod.PARAMETRIC:
            var_95, var_99, cvar_95, cvar_99 = self._parametric_var(
                portfolio_returns, portfolio_value, time_horizon_days
            )
        elif method == VaRMethod.HISTORICAL:
            var_95, var_99, cvar_95, cvar_99 = self._historical_var(
                portfolio_returns, portfolio_value, time_horizon_days
            )
        else:  # MONTE_CARLO
            var_95, var_99, cvar_95, cvar_99 = self._monte_carlo_var(
                portfolio_returns, portfolio_value, time_horizon_days
            )

        calc_time = (time.time() - start_time) * 1000

        result = VaRResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            var_95_pct=var_95 / portfolio_value if portfolio_value > 0 else 0,
            var_99_pct=var_99 / portfolio_value if portfolio_value > 0 else 0,
            method=method,
            confidence=min(0.95, len(portfolio_returns) / self.lookback_days),
            calculation_time_ms=calc_time,
            positions_included=len(self._positions),
            portfolio_value=portfolio_value,
        )

        self._latest_var = result
        self._check_limits_and_alert(result)

        return result

    def _calculate_portfolio_returns(self) -> np.ndarray:
        """Calculate weighted portfolio returns."""
        if not self._positions:
            return np.array([])

        portfolio_value = sum(self._positions.values())
        if portfolio_value == 0:
            return np.array([])

        # Get minimum history length across all positions
        min_len = min(len(self.returns_history.get(symbol, [])) for symbol in self._positions.keys())

        if min_len == 0:
            return np.array([])

        # Calculate weighted returns
        portfolio_returns = np.zeros(min_len)

        for symbol, value in self._positions.items():
            weight = value / portfolio_value
            returns = self.returns_history.get(symbol, [])

            if returns:
                # Use most recent returns
                symbol_returns = np.array(returns[-min_len:])
                portfolio_returns += weight * symbol_returns

        return portfolio_returns

    def _parametric_var(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        time_horizon: int,
    ) -> tuple[float, float, float, float]:
        """Calculate parametric (variance-covariance) VaR."""
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Scale for time horizon
        std_scaled = std_return * math.sqrt(time_horizon)

        # VaR at different confidence levels
        var_95 = portfolio_value * (mean_return - self.Z_SCORES[0.95] * std_scaled)
        var_99 = portfolio_value * (mean_return - self.Z_SCORES[0.99] * std_scaled)

        # CVaR (expected shortfall) - assumes normal distribution
        # E[X | X < VaR] = mu - sigma * phi(z) / (1 - Phi(z))
        pdf_95 = self._standard_normal_pdf(self.Z_SCORES[0.95])
        pdf_99 = self._standard_normal_pdf(self.Z_SCORES[0.99])

        cvar_95 = portfolio_value * (mean_return - std_scaled * pdf_95 / 0.05)
        cvar_99 = portfolio_value * (mean_return - std_scaled * pdf_99 / 0.01)

        return abs(var_95), abs(var_99), abs(cvar_95), abs(cvar_99)

    def _historical_var(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        time_horizon: int,
    ) -> tuple[float, float, float, float]:
        """Calculate historical simulation VaR."""
        # Scale returns for time horizon (simplified)
        scaled_returns = returns * math.sqrt(time_horizon)

        # Sort returns (worst to best)
        sorted_returns = np.sort(scaled_returns)

        # VaR at percentiles
        var_95_return = np.percentile(sorted_returns, 5)
        var_99_return = np.percentile(sorted_returns, 1)

        var_95 = abs(var_95_return * portfolio_value)
        var_99 = abs(var_99_return * portfolio_value)

        # CVaR - average of returns below VaR
        cvar_95_idx = int(len(sorted_returns) * 0.05)
        cvar_99_idx = int(len(sorted_returns) * 0.01)

        cvar_95_return = np.mean(sorted_returns[: max(1, cvar_95_idx)])
        cvar_99_return = np.mean(sorted_returns[: max(1, cvar_99_idx)])

        cvar_95 = abs(cvar_95_return * portfolio_value)
        cvar_99 = abs(cvar_99_return * portfolio_value)

        return var_95, var_99, cvar_95, cvar_99

    def _monte_carlo_var(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        time_horizon: int,
        num_simulations: int = 10000,
    ) -> tuple[float, float, float, float]:
        """Calculate Monte Carlo VaR."""
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Generate simulated returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(
            mean_return * time_horizon,
            std_return * math.sqrt(time_horizon),
            num_simulations,
        )

        # Calculate simulated P&L
        simulated_pnl = portfolio_value * simulated_returns

        # Sort (worst to best)
        sorted_pnl = np.sort(simulated_pnl)

        # VaR at percentiles
        var_95 = abs(np.percentile(sorted_pnl, 5))
        var_99 = abs(np.percentile(sorted_pnl, 1))

        # CVaR
        cvar_95_idx = int(num_simulations * 0.05)
        cvar_99_idx = int(num_simulations * 0.01)

        cvar_95 = abs(np.mean(sorted_pnl[: max(1, cvar_95_idx)]))
        cvar_99 = abs(np.mean(sorted_pnl[: max(1, cvar_99_idx)]))

        return var_95, var_99, cvar_95, cvar_99

    def _standard_normal_pdf(self, x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def _empty_var_result(self, method: VaRMethod) -> VaRResult:
        """Return empty VaR result."""
        return VaRResult(
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            var_95_pct=0.0,
            var_99_pct=0.0,
            method=method,
            confidence=0.0,
            calculation_time_ms=0.0,
            positions_included=0,
            portfolio_value=0.0,
        )

    def _estimated_var_result(
        self,
        portfolio_value: float,
        method: VaRMethod,
        confidence_level: float,
    ) -> VaRResult:
        """Return estimated VaR when insufficient history."""
        # Use market average volatility estimate (20% annualized)
        avg_daily_vol = 0.20 / math.sqrt(self.TRADING_DAYS)

        var_95 = portfolio_value * avg_daily_vol * self.Z_SCORES[0.95]
        var_99 = portfolio_value * avg_daily_vol * self.Z_SCORES[0.99]

        return VaRResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=var_95 * 1.2,  # Rough estimate
            cvar_99=var_99 * 1.2,
            var_95_pct=var_95 / portfolio_value if portfolio_value > 0 else 0,
            var_99_pct=var_99 / portfolio_value if portfolio_value > 0 else 0,
            method=method,
            confidence=0.3,  # Low confidence for estimate
            calculation_time_ms=0.0,
            positions_included=len(self._positions),
            portfolio_value=portfolio_value,
        )

    def check_limits(self, var_result: VaRResult | None = None) -> tuple[bool, str]:
        """
        Check if VaR exceeds limits.

        Args:
            var_result: VaR result to check (uses latest if None)

        Returns:
            Tuple of (within_limits, message)
        """
        result = var_result or self._latest_var

        if result is None:
            return True, "No VaR calculation available"

        # Check VaR limit
        if result.var_95_pct > self.limits.max_var_pct:
            return False, f"VaR {result.var_95_pct:.2%} exceeds limit {self.limits.max_var_pct:.2%}"

        # Check CVaR limit
        cvar_pct = result.cvar_95 / result.portfolio_value if result.portfolio_value > 0 else 0
        if cvar_pct > self.limits.max_cvar_pct:
            return False, f"CVaR {cvar_pct:.2%} exceeds limit {self.limits.max_cvar_pct:.2%}"

        return True, "Within limits"

    def _check_limits_and_alert(self, result: VaRResult) -> None:
        """Check limits and trigger alerts if needed."""
        utilization = result.var_95_pct / self.limits.max_var_pct

        if utilization >= self.limits.critical_threshold:
            level = RiskLevel.CRITICAL
            message = f"CRITICAL: VaR at {utilization:.0%} of limit"
        elif utilization >= self.limits.warning_threshold:
            level = RiskLevel.HIGH
            message = f"WARNING: VaR at {utilization:.0%} of limit"
        elif utilization >= 0.5:
            level = RiskLevel.ELEVATED
            message = f"ELEVATED: VaR at {utilization:.0%} of limit"
        else:
            return  # No alert needed

        # Find top contributing positions
        top_positions = self._get_top_risk_contributors(3)

        alert = VaRAlert(
            level=level,
            message=message,
            current_var_pct=result.var_95_pct,
            limit_var_pct=self.limits.max_var_pct,
            utilization_pct=utilization,
            positions_contributing=[p.symbol for p in top_positions],
        )

        # Trigger callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def get_var_contribution(self, symbol: str) -> float:
        """
        Get individual position's contribution to portfolio VaR.

        Args:
            symbol: Symbol to check

        Returns:
            Contribution to VaR as dollar amount
        """
        if symbol not in self._positions:
            return 0.0

        position_value = self._positions[symbol]
        returns = self.returns_history.get(symbol, [])

        if len(returns) < self.min_history_days:
            # Estimate using average volatility
            return position_value * 0.20 / math.sqrt(self.TRADING_DAYS) * self.Z_SCORES[0.95]

        std_return = np.std(returns)
        return position_value * std_return * self.Z_SCORES[0.95]

    def _get_top_risk_contributors(self, n: int = 5) -> list[PositionRisk]:
        """Get top risk contributing positions."""
        position_risks = []
        portfolio_value = sum(self._positions.values())
        total_var = sum(self.get_var_contribution(s) for s in self._positions.keys())

        for symbol, value in self._positions.items():
            returns = self.returns_history.get(symbol, [])
            volatility = np.std(returns) * math.sqrt(self.TRADING_DAYS) if returns else 0.20
            var_contrib = self.get_var_contribution(symbol)

            position_risks.append(
                PositionRisk(
                    symbol=symbol,
                    market_value=value,
                    weight=value / portfolio_value if portfolio_value > 0 else 0,
                    volatility=volatility,
                    var_contribution=var_contrib,
                    var_contribution_pct=var_contrib / total_var if total_var > 0 else 0,
                )
            )

        # Sort by VaR contribution
        position_risks.sort(key=lambda x: x.var_contribution, reverse=True)

        return position_risks[:n]

    def register_alert_callback(self, callback: Callable[[VaRAlert], None]) -> None:
        """Register callback for VaR alerts."""
        self._alert_callbacks.append(callback)

    def get_risk_summary(self) -> dict[str, Any]:
        """Get comprehensive risk summary."""
        portfolio_value = sum(self._positions.values())

        return {
            "portfolio_value": portfolio_value,
            "positions": len(self._positions),
            "latest_var": self._latest_var.to_dict() if self._latest_var else None,
            "limits": self.limits.to_dict(),
            "top_contributors": [p.to_dict() for p in self._get_top_risk_contributors(5)],
            "history_days": min(len(returns) for returns in self.returns_history.values())
            if self.returns_history
            else 0,
        }


def create_var_monitor(
    max_var_pct: float = 0.05,
    lookback_days: int = 252,
) -> VaRMonitor:
    """Factory function to create VaR monitor."""
    limits = VaRLimits(max_var_pct=max_var_pct)
    return VaRMonitor(limits=limits, lookback_days=lookback_days)


__all__ = [
    "PositionRisk",
    "RiskLevel",
    "VaRAlert",
    "VaRLimits",
    "VaRMethod",
    "VaRMonitor",
    "VaRResult",
    "create_var_monitor",
]
