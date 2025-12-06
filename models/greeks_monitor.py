"""
Greeks Risk Monitor (UPGRADE-010 Sprint 4 Expansion)

Real-time Greeks exposure monitoring with configurable limits and alerts.
Tracks delta, gamma, vega, and theta exposure across the portfolio.

Author: Claude Code
Date: December 2025
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class GreeksAlertLevel(Enum):
    """Severity level for Greeks alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACH = "breach"


class GreeksType(Enum):
    """Types of Greeks being monitored."""

    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    THETA = "theta"
    RHO = "rho"


class RiskProfile(Enum):
    """Risk profile presets for Greeks limits."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass
class GreeksLimits:
    """
    Greeks exposure limits configuration.

    Limits can be set per risk profile:
    - Conservative: Tight limits for risk-averse trading
    - Moderate: Balanced limits for typical options trading
    - Aggressive: Wider limits for experienced traders
    """

    # Delta limits (equivalent underlying shares)
    max_delta: float = 50.0
    min_delta: float = -50.0

    # Gamma limits (delta change per $1 move)
    max_gamma: float = 10.0

    # Vega limits (P&L per 1% IV change)
    max_vega: float = 1000.0

    # Theta limits (daily time decay)
    max_theta: float = 0.0  # Allow positive theta
    min_theta: float = -500.0  # Max daily decay

    # Rho limits (P&L per 1% rate change)
    max_rho: float = 500.0

    # Alert thresholds
    warning_threshold: float = 0.75  # Warn at 75% of limit
    critical_threshold: float = 0.90  # Critical at 90%

    # Per-position limits
    max_position_delta: float = 25.0
    max_position_gamma: float = 5.0

    @classmethod
    def from_profile(cls, profile: RiskProfile) -> "GreeksLimits":
        """Create limits from a risk profile preset."""
        if profile == RiskProfile.CONSERVATIVE:
            return cls(
                max_delta=30.0,
                min_delta=-30.0,
                max_gamma=5.0,
                max_vega=500.0,
                min_theta=-200.0,
                max_position_delta=15.0,
            )
        elif profile == RiskProfile.MODERATE:
            return cls(
                max_delta=50.0,
                min_delta=-50.0,
                max_gamma=10.0,
                max_vega=1000.0,
                min_theta=-500.0,
                max_position_delta=25.0,
            )
        elif profile == RiskProfile.AGGRESSIVE:
            return cls(
                max_delta=100.0,
                min_delta=-100.0,
                max_gamma=20.0,
                max_vega=2000.0,
                min_theta=-1000.0,
                max_position_delta=50.0,
            )
        else:
            return cls()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_delta": self.max_delta,
            "min_delta": self.min_delta,
            "max_gamma": self.max_gamma,
            "max_vega": self.max_vega,
            "max_theta": self.max_theta,
            "min_theta": self.min_theta,
            "max_rho": self.max_rho,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "max_position_delta": self.max_position_delta,
            "max_position_gamma": self.max_position_gamma,
        }


@dataclass
class PositionGreeksSnapshot:
    """Greeks snapshot for a single position."""

    symbol: str
    underlying: str
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float = 0.0
    quantity: int = 0
    contract_multiplier: int = 100
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def net_delta(self) -> float:
        """Net delta exposure (delta * quantity * multiplier)."""
        return self.delta * self.quantity * self.contract_multiplier

    @property
    def net_gamma(self) -> float:
        """Net gamma exposure."""
        return self.gamma * self.quantity * self.contract_multiplier

    @property
    def net_vega(self) -> float:
        """Net vega exposure."""
        return self.vega * self.quantity * self.contract_multiplier

    @property
    def net_theta(self) -> float:
        """Net theta exposure."""
        return self.theta * self.quantity * self.contract_multiplier

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "underlying": self.underlying,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
            "quantity": self.quantity,
            "net_delta": self.net_delta,
            "net_gamma": self.net_gamma,
            "net_vega": self.net_vega,
            "net_theta": self.net_theta,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class GreeksAlert:
    """Alert for Greeks limit breach or warning."""

    level: GreeksAlertLevel
    greeks_type: GreeksType
    current_value: float
    limit_value: float
    utilization_pct: float
    message: str
    position: str | None = None  # If position-specific
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "greeks_type": self.greeks_type.value,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
            "utilization_pct": self.utilization_pct,
            "message": self.message,
            "position": self.position,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PortfolioGreeksExposure:
    """Aggregated Greeks exposure for the portfolio."""

    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_vega: float = 0.0
    total_theta: float = 0.0
    total_rho: float = 0.0
    position_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_delta": self.total_delta,
            "total_gamma": self.total_gamma,
            "total_vega": self.total_vega,
            "total_theta": self.total_theta,
            "total_rho": self.total_rho,
            "position_count": self.position_count,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HedgeRecommendation:
    """Recommendation for hedging Greeks exposure."""

    needed: bool
    priority: str  # "LOW", "MEDIUM", "HIGH", "URGENT"
    delta_adjustment: float  # Target delta change
    suggested_hedge: str  # Description of suggested hedge
    estimated_cost: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "needed": self.needed,
            "priority": self.priority,
            "delta_adjustment": self.delta_adjustment,
            "suggested_hedge": self.suggested_hedge,
            "estimated_cost": self.estimated_cost,
            "reason": self.reason,
        }


class GreeksMonitor:
    """
    Real-time Greeks exposure monitoring with alerts.

    Tracks portfolio Greeks exposure against configurable limits
    and generates alerts when thresholds are approached or breached.

    Features:
    - Portfolio-level Greeks aggregation
    - Position-level Greeks tracking
    - Configurable limits by risk profile
    - Multi-level alerts (warning, critical, breach)
    - Hedging recommendations
    - Alert callbacks for integration

    Example:
        monitor = GreeksMonitor(
            limits=GreeksLimits.from_profile(RiskProfile.MODERATE)
        )

        # Update positions
        monitor.update_position(PositionGreeksSnapshot(
            symbol="SPY_241220C450",
            underlying="SPY",
            delta=0.45,
            gamma=0.02,
            vega=0.15,
            theta=-0.03,
            quantity=10
        ))

        # Check limits
        within_limits, alerts = monitor.check_limits()
        if not within_limits:
            for alert in alerts:
                print(f"ALERT: {alert.message}")

        # Get hedging recommendation
        hedge = monitor.get_hedging_recommendation()
        if hedge.needed:
            print(f"Hedge: {hedge.suggested_hedge}")
    """

    def __init__(
        self,
        limits: GreeksLimits | None = None,
        risk_profile: RiskProfile = RiskProfile.MODERATE,
    ):
        """
        Initialize Greeks Monitor.

        Args:
            limits: Custom Greeks limits (overrides risk_profile)
            risk_profile: Preset risk profile for limits
        """
        if limits is not None:
            self.limits = limits
        else:
            self.limits = GreeksLimits.from_profile(risk_profile)

        self.risk_profile = risk_profile
        self._positions: dict[str, PositionGreeksSnapshot] = {}
        self._alert_callbacks: list[Callable[[GreeksAlert], None]] = []
        self._alert_history: list[GreeksAlert] = []
        self._max_history = 1000

    def update_position(self, snapshot: PositionGreeksSnapshot) -> None:
        """
        Update or add a position's Greeks.

        Args:
            snapshot: Position Greeks snapshot
        """
        if snapshot.quantity == 0:
            self._positions.pop(snapshot.symbol, None)
        else:
            self._positions[snapshot.symbol] = snapshot
            logger.debug(
                f"Updated position {snapshot.symbol}: "
                f"delta={snapshot.net_delta:.2f}, gamma={snapshot.net_gamma:.4f}"
            )

    def remove_position(self, symbol: str) -> None:
        """Remove a position from monitoring."""
        self._positions.pop(symbol, None)

    def clear_positions(self) -> None:
        """Clear all positions."""
        self._positions.clear()

    def get_portfolio_exposure(self) -> PortfolioGreeksExposure:
        """
        Get aggregated portfolio Greeks exposure.

        Returns:
            PortfolioGreeksExposure with totals
        """
        total_delta = sum(p.net_delta for p in self._positions.values())
        total_gamma = sum(p.net_gamma for p in self._positions.values())
        total_vega = sum(p.net_vega for p in self._positions.values())
        total_theta = sum(p.net_theta for p in self._positions.values())
        total_rho = sum(p.rho * p.quantity * p.contract_multiplier for p in self._positions.values())

        return PortfolioGreeksExposure(
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_vega=total_vega,
            total_theta=total_theta,
            total_rho=total_rho,
            position_count=len(self._positions),
        )

    def check_limits(self) -> tuple[bool, list[GreeksAlert]]:
        """
        Check if current exposure is within limits.

        Returns:
            Tuple of (within_limits, list_of_alerts)
        """
        alerts: list[GreeksAlert] = []
        exposure = self.get_portfolio_exposure()

        # Check delta limits
        alerts.extend(self._check_delta_limits(exposure))

        # Check gamma limits
        alerts.extend(self._check_gamma_limits(exposure))

        # Check vega limits
        alerts.extend(self._check_vega_limits(exposure))

        # Check theta limits
        alerts.extend(self._check_theta_limits(exposure))

        # Check position-level limits
        alerts.extend(self._check_position_limits())

        # Trigger callbacks and store history
        for alert in alerts:
            self._alert_history.append(alert)
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

        # Trim history
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history :]

        # Determine if within limits (no breach-level alerts)
        within_limits = not any(a.level == GreeksAlertLevel.BREACH for a in alerts)

        return within_limits, alerts

    def _check_delta_limits(self, exposure: PortfolioGreeksExposure) -> list[GreeksAlert]:
        """Check delta exposure against limits."""
        alerts = []
        delta = exposure.total_delta

        # Check max delta
        if delta > 0:
            utilization = delta / self.limits.max_delta if self.limits.max_delta > 0 else 0
            alert = self._create_limit_alert(
                GreeksType.DELTA,
                delta,
                self.limits.max_delta,
                utilization,
                "Long delta exposure",
            )
            if alert:
                alerts.append(alert)

        # Check min delta (short exposure)
        if delta < 0:
            utilization = abs(delta) / abs(self.limits.min_delta) if self.limits.min_delta != 0 else 0
            alert = self._create_limit_alert(
                GreeksType.DELTA,
                delta,
                self.limits.min_delta,
                utilization,
                "Short delta exposure",
            )
            if alert:
                alerts.append(alert)

        return alerts

    def _check_gamma_limits(self, exposure: PortfolioGreeksExposure) -> list[GreeksAlert]:
        """Check gamma exposure against limits."""
        alerts = []
        gamma = abs(exposure.total_gamma)

        utilization = gamma / self.limits.max_gamma if self.limits.max_gamma > 0 else 0
        alert = self._create_limit_alert(
            GreeksType.GAMMA,
            exposure.total_gamma,
            self.limits.max_gamma,
            utilization,
            "Gamma exposure",
        )
        if alert:
            alerts.append(alert)

        return alerts

    def _check_vega_limits(self, exposure: PortfolioGreeksExposure) -> list[GreeksAlert]:
        """Check vega exposure against limits."""
        alerts = []
        vega = abs(exposure.total_vega)

        utilization = vega / self.limits.max_vega if self.limits.max_vega > 0 else 0
        alert = self._create_limit_alert(
            GreeksType.VEGA,
            exposure.total_vega,
            self.limits.max_vega,
            utilization,
            "Vega exposure",
        )
        if alert:
            alerts.append(alert)

        return alerts

    def _check_theta_limits(self, exposure: PortfolioGreeksExposure) -> list[GreeksAlert]:
        """Check theta exposure against limits."""
        alerts = []
        theta = exposure.total_theta

        # Only alert on negative theta (decay)
        if theta < self.limits.min_theta:
            utilization = abs(theta) / abs(self.limits.min_theta) if self.limits.min_theta != 0 else 0
            alert = self._create_limit_alert(
                GreeksType.THETA,
                theta,
                self.limits.min_theta,
                utilization,
                "Theta decay exposure",
            )
            if alert:
                alerts.append(alert)

        return alerts

    def _check_position_limits(self) -> list[GreeksAlert]:
        """Check individual position Greeks against limits."""
        alerts = []

        for symbol, pos in self._positions.items():
            # Check position delta
            if abs(pos.net_delta) > self.limits.max_position_delta:
                alerts.append(
                    GreeksAlert(
                        level=GreeksAlertLevel.WARNING,
                        greeks_type=GreeksType.DELTA,
                        current_value=pos.net_delta,
                        limit_value=self.limits.max_position_delta,
                        utilization_pct=abs(pos.net_delta) / self.limits.max_position_delta,
                        message=f"Position {symbol} exceeds delta limit",
                        position=symbol,
                    )
                )

            # Check position gamma
            if abs(pos.net_gamma) > self.limits.max_position_gamma:
                alerts.append(
                    GreeksAlert(
                        level=GreeksAlertLevel.WARNING,
                        greeks_type=GreeksType.GAMMA,
                        current_value=pos.net_gamma,
                        limit_value=self.limits.max_position_gamma,
                        utilization_pct=abs(pos.net_gamma) / self.limits.max_position_gamma,
                        message=f"Position {symbol} exceeds gamma limit",
                        position=symbol,
                    )
                )

        return alerts

    def _create_limit_alert(
        self,
        greeks_type: GreeksType,
        current_value: float,
        limit_value: float,
        utilization: float,
        description: str,
    ) -> GreeksAlert | None:
        """Create alert if threshold exceeded."""
        if utilization >= 1.0:
            level = GreeksAlertLevel.BREACH
            msg = f"BREACH: {description} ({current_value:.2f}) exceeds limit ({limit_value:.2f})"
        elif utilization >= self.limits.critical_threshold:
            level = GreeksAlertLevel.CRITICAL
            msg = f"CRITICAL: {description} at {utilization:.0%} of limit"
        elif utilization >= self.limits.warning_threshold:
            level = GreeksAlertLevel.WARNING
            msg = f"WARNING: {description} at {utilization:.0%} of limit"
        else:
            return None

        return GreeksAlert(
            level=level,
            greeks_type=greeks_type,
            current_value=current_value,
            limit_value=limit_value,
            utilization_pct=utilization,
            message=msg,
        )

    def get_hedging_recommendation(self, underlying_price: float = 100.0) -> HedgeRecommendation:
        """
        Get recommendation for hedging current exposure.

        Args:
            underlying_price: Current price of primary underlying

        Returns:
            HedgeRecommendation with suggested actions
        """
        exposure = self.get_portfolio_exposure()

        # Determine priority based on delta exposure
        delta_utilization = abs(exposure.total_delta) / max(abs(self.limits.max_delta), abs(self.limits.min_delta))

        if delta_utilization < self.limits.warning_threshold:
            return HedgeRecommendation(
                needed=False,
                priority="LOW",
                delta_adjustment=0.0,
                suggested_hedge="No hedge needed",
                reason=f"Delta utilization at {delta_utilization:.0%}",
            )

        # Calculate required delta adjustment
        if exposure.total_delta > self.limits.max_delta * 0.8:
            target_delta = self.limits.max_delta * 0.5  # Target 50% of limit
            delta_adjustment = target_delta - exposure.total_delta
        elif exposure.total_delta < self.limits.min_delta * 0.8:
            target_delta = self.limits.min_delta * 0.5
            delta_adjustment = target_delta - exposure.total_delta
        else:
            delta_adjustment = -exposure.total_delta * 0.3  # Reduce by 30%

        # Determine hedge instrument
        shares_needed = int(delta_adjustment)
        if shares_needed > 0:
            suggested_hedge = f"Buy {shares_needed} shares of underlying"
        elif shares_needed < 0:
            suggested_hedge = f"Sell {abs(shares_needed)} shares of underlying"
        else:
            suggested_hedge = "Consider delta-neutral spreads"

        # Estimate cost
        estimated_cost = abs(shares_needed) * underlying_price * 0.001  # Rough slippage estimate

        # Determine priority
        if delta_utilization >= 1.0:
            priority = "URGENT"
        elif delta_utilization >= self.limits.critical_threshold:
            priority = "HIGH"
        elif delta_utilization >= self.limits.warning_threshold:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        return HedgeRecommendation(
            needed=True,
            priority=priority,
            delta_adjustment=delta_adjustment,
            suggested_hedge=suggested_hedge,
            estimated_cost=estimated_cost,
            reason=f"Delta at {delta_utilization:.0%} of limit ({exposure.total_delta:.2f})",
        )

    def register_alert_callback(self, callback: Callable[[GreeksAlert], None]) -> None:
        """Register a callback for Greeks alerts."""
        self._alert_callbacks.append(callback)

    def get_exposure_by_underlying(self) -> dict[str, PortfolioGreeksExposure]:
        """Get Greeks exposure grouped by underlying symbol."""
        by_underlying: dict[str, list[PositionGreeksSnapshot]] = {}

        for pos in self._positions.values():
            if pos.underlying not in by_underlying:
                by_underlying[pos.underlying] = []
            by_underlying[pos.underlying].append(pos)

        result = {}
        for underlying, positions in by_underlying.items():
            result[underlying] = PortfolioGreeksExposure(
                total_delta=sum(p.net_delta for p in positions),
                total_gamma=sum(p.net_gamma for p in positions),
                total_vega=sum(p.net_vega for p in positions),
                total_theta=sum(p.net_theta for p in positions),
                total_rho=sum(p.rho * p.quantity * p.contract_multiplier for p in positions),
                position_count=len(positions),
            )

        return result

    def get_alert_history(self, limit: int = 100) -> list[GreeksAlert]:
        """Get recent alert history."""
        return self._alert_history[-limit:]

    def get_summary(self) -> dict[str, Any]:
        """Get monitoring summary."""
        exposure = self.get_portfolio_exposure()
        within_limits, alerts = self.check_limits()

        # Calculate utilization percentages
        delta_util = abs(exposure.total_delta) / max(abs(self.limits.max_delta), abs(self.limits.min_delta), 1)
        gamma_util = abs(exposure.total_gamma) / max(self.limits.max_gamma, 1)
        vega_util = abs(exposure.total_vega) / max(self.limits.max_vega, 1)
        theta_util = abs(exposure.total_theta) / max(abs(self.limits.min_theta), 1)

        return {
            "within_limits": within_limits,
            "exposure": exposure.to_dict(),
            "limits": self.limits.to_dict(),
            "utilization": {
                "delta_pct": delta_util,
                "gamma_pct": gamma_util,
                "vega_pct": vega_util,
                "theta_pct": theta_util,
            },
            "alerts": [a.to_dict() for a in alerts],
            "position_count": len(self._positions),
            "risk_profile": self.risk_profile.value,
        }


def create_greeks_monitor(
    risk_profile: str = "moderate",
    max_delta: float | None = None,
    max_gamma: float | None = None,
    max_vega: float | None = None,
) -> GreeksMonitor:
    """
    Factory function to create a GreeksMonitor.

    Args:
        risk_profile: "conservative", "moderate", or "aggressive"
        max_delta: Optional custom delta limit (overrides profile)
        max_gamma: Optional custom gamma limit (overrides profile)
        max_vega: Optional custom vega limit (overrides profile)

    Returns:
        Configured GreeksMonitor instance
    """
    profile_map = {
        "conservative": RiskProfile.CONSERVATIVE,
        "moderate": RiskProfile.MODERATE,
        "aggressive": RiskProfile.AGGRESSIVE,
    }

    profile = profile_map.get(risk_profile.lower(), RiskProfile.MODERATE)
    limits = GreeksLimits.from_profile(profile)

    # Apply custom overrides
    if max_delta is not None:
        limits.max_delta = max_delta
        limits.min_delta = -max_delta
    if max_gamma is not None:
        limits.max_gamma = max_gamma
    if max_vega is not None:
        limits.max_vega = max_vega

    return GreeksMonitor(limits=limits, risk_profile=profile)
