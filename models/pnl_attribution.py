"""
P&L Attribution Module

Decomposes options P&L into components:
- Delta P&L (underlying movement)
- Gamma P&L (convexity)
- Theta P&L (time decay)
- Vega P&L (volatility changes)
- Unexplained (model error, bid-ask, etc.)

Based on Taylor expansion of option value.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class GreeksSnapshot:
    """Snapshot of option Greeks at a point in time."""

    timestamp: datetime
    underlying_price: float
    option_price: float
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float = 0.0


@dataclass
class PnLBreakdown:
    """Breakdown of P&L by Greek component."""

    delta_pnl: float = 0.0  # From underlying price change
    gamma_pnl: float = 0.0  # From delta change (convexity)
    theta_pnl: float = 0.0  # From time decay
    vega_pnl: float = 0.0  # From IV change
    rho_pnl: float = 0.0  # From interest rate change
    unexplained: float = 0.0  # Residual
    total_pnl: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "delta_pnl": self.delta_pnl,
            "gamma_pnl": self.gamma_pnl,
            "theta_pnl": self.theta_pnl,
            "vega_pnl": self.vega_pnl,
            "rho_pnl": self.rho_pnl,
            "unexplained": self.unexplained,
            "total_pnl": self.total_pnl,
        }

    def as_percentages(self) -> dict[str, float]:
        """Get breakdown as percentages of total."""
        if self.total_pnl == 0:
            return dict.fromkeys(self.to_dict(), 0.0)

        total_abs = abs(self.total_pnl)
        return {
            "delta_pct": self.delta_pnl / total_abs * 100,
            "gamma_pct": self.gamma_pnl / total_abs * 100,
            "theta_pct": self.theta_pnl / total_abs * 100,
            "vega_pct": self.vega_pnl / total_abs * 100,
            "rho_pct": self.rho_pnl / total_abs * 100,
            "unexplained_pct": self.unexplained / total_abs * 100,
        }


class PnLAttributor:
    """
    Attributes P&L to Greek components using Taylor expansion.

    Formula:
    dV ≈ δ * dS + 0.5 * γ * dS² + θ * dt + ν * dσ + ρ * dr

    Where:
    - dV = change in option value
    - δ = delta, dS = change in underlying
    - γ = gamma
    - θ = theta, dt = time elapsed
    - ν = vega, dσ = change in IV
    - ρ = rho, dr = change in rates
    """

    def __init__(self, contract_multiplier: int = 100):
        """
        Initialize attributor.

        Args:
            contract_multiplier: Shares per contract (typically 100)
        """
        self.contract_multiplier = contract_multiplier
        self.history: list[GreeksSnapshot] = []

    def add_snapshot(self, snapshot: GreeksSnapshot) -> None:
        """Add a Greeks snapshot."""
        self.history.append(snapshot)

    def calculate_attribution(
        self,
        start: GreeksSnapshot,
        end: GreeksSnapshot,
        quantity: int = 1,
    ) -> PnLBreakdown:
        """
        Calculate P&L attribution between two snapshots.

        Args:
            start: Starting snapshot
            end: Ending snapshot
            quantity: Number of contracts (positive for long, negative for short)

        Returns:
            PnLBreakdown with component P&L
        """
        multiplier = self.contract_multiplier * quantity

        # Changes
        ds = end.underlying_price - start.underlying_price  # Price change
        dt = (end.timestamp - start.timestamp).total_seconds() / (365.25 * 24 * 3600)  # Years
        d_iv = end.implied_volatility - start.implied_volatility

        # Delta P&L
        delta_pnl = start.delta * ds * multiplier

        # Gamma P&L (second-order effect)
        gamma_pnl = 0.5 * start.gamma * (ds**2) * multiplier

        # Theta P&L (note: theta is typically negative and quoted as daily)
        # Convert theta to match time period
        theta_pnl = start.theta * dt * 365 * multiplier  # Theta is often daily

        # Vega P&L (vega is per 1% IV change, convert d_iv to percentage points)
        vega_pnl = start.vega * d_iv * 100 * multiplier

        # Actual P&L
        actual_pnl = (end.option_price - start.option_price) * multiplier

        # Unexplained (residual)
        explained = delta_pnl + gamma_pnl + theta_pnl + vega_pnl
        unexplained = actual_pnl - explained

        return PnLBreakdown(
            delta_pnl=delta_pnl,
            gamma_pnl=gamma_pnl,
            theta_pnl=theta_pnl,
            vega_pnl=vega_pnl,
            unexplained=unexplained,
            total_pnl=actual_pnl,
        )

    def calculate_period_attribution(
        self,
        start_time: datetime,
        end_time: datetime,
        quantity: int = 1,
    ) -> PnLBreakdown | None:
        """
        Calculate attribution for a time period using history.

        Args:
            start_time: Period start
            end_time: Period end
            quantity: Number of contracts

        Returns:
            PnLBreakdown or None if insufficient data
        """
        # Find closest snapshots
        start_snapshot = None
        end_snapshot = None

        for snapshot in self.history:
            if snapshot.timestamp <= start_time:
                start_snapshot = snapshot
            if snapshot.timestamp <= end_time:
                end_snapshot = snapshot

        if start_snapshot is None or end_snapshot is None:
            return None

        return self.calculate_attribution(start_snapshot, end_snapshot, quantity)

    def get_cumulative_attribution(self, quantity: int = 1) -> PnLBreakdown:
        """Calculate cumulative attribution from full history."""
        if len(self.history) < 2:
            return PnLBreakdown()

        total = PnLBreakdown()

        for i in range(1, len(self.history)):
            period = self.calculate_attribution(self.history[i - 1], self.history[i], quantity)
            total.delta_pnl += period.delta_pnl
            total.gamma_pnl += period.gamma_pnl
            total.theta_pnl += period.theta_pnl
            total.vega_pnl += period.vega_pnl
            total.unexplained += period.unexplained
            total.total_pnl += period.total_pnl

        return total


class PortfolioPnLAttributor:
    """
    Attributes P&L for a portfolio of options.

    Aggregates individual position attributions.
    """

    def __init__(self):
        """Initialize portfolio attributor."""
        self.positions: dict[str, PnLAttributor] = {}
        self.quantities: dict[str, int] = {}

    def add_position(self, symbol: str, quantity: int) -> PnLAttributor:
        """
        Add a position to track.

        Args:
            symbol: Position identifier
            quantity: Number of contracts

        Returns:
            PnLAttributor for the position
        """
        self.positions[symbol] = PnLAttributor()
        self.quantities[symbol] = quantity
        return self.positions[symbol]

    def update_position(self, symbol: str, snapshot: GreeksSnapshot) -> None:
        """Update a position with new snapshot."""
        if symbol in self.positions:
            self.positions[symbol].add_snapshot(snapshot)

    def get_position_attribution(self, symbol: str) -> PnLBreakdown | None:
        """Get attribution for a single position."""
        if symbol not in self.positions:
            return None

        return self.positions[symbol].get_cumulative_attribution(self.quantities.get(symbol, 1))

    def get_portfolio_attribution(self) -> PnLBreakdown:
        """Get aggregated attribution for entire portfolio."""
        total = PnLBreakdown()

        for symbol, attributor in self.positions.items():
            quantity = self.quantities.get(symbol, 1)
            pos_attr = attributor.get_cumulative_attribution(quantity)

            total.delta_pnl += pos_attr.delta_pnl
            total.gamma_pnl += pos_attr.gamma_pnl
            total.theta_pnl += pos_attr.theta_pnl
            total.vega_pnl += pos_attr.vega_pnl
            total.unexplained += pos_attr.unexplained
            total.total_pnl += pos_attr.total_pnl

        return total

    def get_report(self) -> dict[str, Any]:
        """Generate P&L attribution report."""
        portfolio_attr = self.get_portfolio_attribution()

        report = {
            "timestamp": datetime.now().isoformat(),
            "portfolio": {
                "breakdown": portfolio_attr.to_dict(),
                "percentages": portfolio_attr.as_percentages(),
            },
            "positions": {},
        }

        for symbol in self.positions:
            pos_attr = self.get_position_attribution(symbol)
            if pos_attr:
                report["positions"][symbol] = {
                    "quantity": self.quantities.get(symbol, 0),
                    "breakdown": pos_attr.to_dict(),
                    "percentages": pos_attr.as_percentages(),
                }

        return report


class RealizedVolatilityCalculator:
    """
    Calculate realized volatility for gamma P&L analysis.

    Compares realized vs implied volatility to identify
    profitable gamma scalping opportunities.
    """

    def __init__(self, lookback_periods: int = 20):
        """
        Initialize calculator.

        Args:
            lookback_periods: Number of periods for rolling calculation
        """
        self.lookback = lookback_periods
        self.price_history: list[float] = []
        self.return_history: list[float] = []

    def update(self, price: float) -> None:
        """Update with new price."""
        if self.price_history:
            ret = (price - self.price_history[-1]) / self.price_history[-1]
            self.return_history.append(ret)

        self.price_history.append(price)

        # Trim history
        if len(self.price_history) > self.lookback + 1:
            self.price_history = self.price_history[-self.lookback - 1 :]
            self.return_history = self.return_history[-self.lookback :]

    def calculate_realized_volatility(self, annualization_factor: float = 252) -> float:
        """
        Calculate realized volatility.

        Args:
            annualization_factor: Trading days per year

        Returns:
            Annualized realized volatility
        """
        if len(self.return_history) < 2:
            return 0.0

        # Standard deviation of returns
        mean_ret = sum(self.return_history) / len(self.return_history)
        variance = sum((r - mean_ret) ** 2 for r in self.return_history) / (len(self.return_history) - 1)
        daily_vol = variance**0.5

        # Annualize
        return daily_vol * (annualization_factor**0.5)

    def get_vol_ratio(self, implied_volatility: float) -> float:
        """
        Get ratio of realized to implied volatility.

        Ratio > 1: Realized vol higher than implied (gamma profitable)
        Ratio < 1: Realized vol lower than implied (theta profitable)
        """
        realized = self.calculate_realized_volatility()
        if implied_volatility == 0:
            return 0.0
        return realized / implied_volatility


def create_attributor_from_trades(
    trades: list[dict[str, Any]],
) -> PortfolioPnLAttributor:
    """
    Create portfolio attributor from trade history.

    Args:
        trades: List of trade dicts with Greeks information

    Returns:
        Configured PortfolioPnLAttributor
    """
    attributor = PortfolioPnLAttributor()

    for trade in trades:
        symbol = trade.get("symbol", "")
        quantity = trade.get("quantity", 0)

        if symbol not in attributor.positions:
            attributor.add_position(symbol, quantity)

        snapshot = GreeksSnapshot(
            timestamp=trade.get("timestamp", datetime.now()),
            underlying_price=trade.get("underlying_price", 0),
            option_price=trade.get("option_price", 0),
            implied_volatility=trade.get("iv", 0),
            delta=trade.get("delta", 0),
            gamma=trade.get("gamma", 0),
            theta=trade.get("theta", 0),
            vega=trade.get("vega", 0),
        )
        attributor.update_position(symbol, snapshot)

    return attributor


__all__ = [
    "GreeksSnapshot",
    "PnLAttributor",
    "PnLBreakdown",
    "PortfolioPnLAttributor",
    "RealizedVolatilityCalculator",
    "create_attributor_from_trades",
]
