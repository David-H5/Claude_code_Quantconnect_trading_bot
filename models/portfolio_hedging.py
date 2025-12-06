"""
Portfolio Greeks Hedging Module

Provides portfolio-level Greeks aggregation and hedging recommendations:
- Aggregate Greeks across all positions
- Delta hedging with underlying
- Gamma/Vega hedging with options
- Target Greeks maintenance
- Hedging trade generation

Based on patterns from ThetaGang, PyOptionTrader, and risk management research.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .multi_leg_strategy import PortfolioGreeks


class HedgeType(Enum):
    """Types of hedging operations."""

    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    DELTA_GAMMA = "delta_gamma"
    FULL = "full"


@dataclass
class Position:
    """Represents a position (stock or option)."""

    symbol: str
    asset_type: str  # "stock" or "option"
    quantity: float
    entry_price: float
    current_price: float
    underlying_symbol: str = ""

    # Greeks (for options)
    delta: float = 1.0  # Stock has delta of 1
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    # Option details
    strike: float = 0.0
    expiry: datetime | None = None
    option_type: str = ""  # "call" or "put"

    @property
    def position_delta(self) -> float:
        """Get position delta (quantity-adjusted)."""
        if self.asset_type == "stock":
            return self.quantity
        return self.delta * self.quantity * 100  # 100 shares per contract

    @property
    def position_gamma(self) -> float:
        """Get position gamma."""
        if self.asset_type == "stock":
            return 0.0
        return self.gamma * self.quantity * 100

    @property
    def position_theta(self) -> float:
        """Get position theta."""
        if self.asset_type == "stock":
            return 0.0
        return self.theta * self.quantity * 100

    @property
    def position_vega(self) -> float:
        """Get position vega."""
        if self.asset_type == "stock":
            return 0.0
        return self.vega * self.quantity * 100

    @property
    def market_value(self) -> float:
        """Get current market value."""
        if self.asset_type == "stock":
            return self.quantity * self.current_price
        return self.quantity * self.current_price * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "type": self.asset_type,
            "quantity": self.quantity,
            "market_value": self.market_value,
            "delta": self.position_delta,
            "gamma": self.position_gamma,
            "theta": self.position_theta,
            "vega": self.position_vega,
        }


@dataclass
class HedgeRecommendation:
    """A recommendation for a hedging trade."""

    hedge_type: HedgeType
    action: str  # "buy" or "sell"
    symbol: str
    asset_type: str
    quantity: float
    rationale: str
    expected_delta_change: float = 0.0
    expected_gamma_change: float = 0.0
    expected_vega_change: float = 0.0
    estimated_cost: float = 0.0
    priority: int = 1  # 1 = highest priority

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hedge_type": self.hedge_type.value,
            "action": self.action,
            "symbol": self.symbol,
            "asset_type": self.asset_type,
            "quantity": self.quantity,
            "rationale": self.rationale,
            "delta_change": self.expected_delta_change,
            "gamma_change": self.expected_gamma_change,
            "vega_change": self.expected_vega_change,
            "cost": self.estimated_cost,
            "priority": self.priority,
        }


@dataclass
class HedgeTargets:
    """Target Greeks for hedging."""

    target_delta: float = 0.0  # Default: delta-neutral
    target_gamma: float = 0.0
    target_vega: float = 0.0
    delta_tolerance: float = 50.0  # Allow ±50 delta
    gamma_tolerance: float = 10.0
    vega_tolerance: float = 100.0


class PortfolioHedger:
    """
    Portfolio-level Greeks management and hedging.

    Aggregates Greeks across positions and generates hedging recommendations.
    """

    def __init__(self, targets: HedgeTargets | None = None):
        """
        Initialize portfolio hedger.

        Args:
            targets: Target Greeks for hedging
        """
        self.positions: dict[str, Position] = {}
        self.targets = targets or HedgeTargets()
        self.underlying_prices: dict[str, float] = {}
        self.algorithm = None  # Set by integrate_with_algorithm()

    def add_position(self, position: Position) -> None:
        """Add or update a position."""
        self.positions[position.symbol] = position
        if position.underlying_symbol:
            self.underlying_prices[position.underlying_symbol] = position.current_price

    def remove_position(self, symbol: str) -> None:
        """Remove a position."""
        self.positions.pop(symbol, None)

    def update_position_greeks(
        self,
        symbol: str,
        delta: float,
        gamma: float,
        theta: float,
        vega: float,
        current_price: float,
    ) -> None:
        """Update Greeks for a position."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.delta = delta
            pos.gamma = gamma
            pos.theta = theta
            pos.vega = vega
            pos.current_price = current_price

    def get_portfolio_greeks(self, underlying: str | None = None) -> PortfolioGreeks:
        """
        Calculate aggregate portfolio Greeks.

        Args:
            underlying: Filter to specific underlying (None for all)

        Returns:
            PortfolioGreeks with aggregated values
        """
        delta = 0.0
        gamma = 0.0
        theta = 0.0
        vega = 0.0
        rho = 0.0

        for pos in self.positions.values():
            # Filter by underlying if specified
            if underlying and pos.underlying_symbol != underlying:
                if pos.symbol != underlying:
                    continue

            delta += pos.position_delta
            gamma += pos.position_gamma
            theta += pos.position_theta
            vega += pos.position_vega
            rho += pos.rho * pos.quantity * 100 if pos.asset_type == "option" else 0

        return PortfolioGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
        )

    def get_greeks_by_underlying(self) -> dict[str, PortfolioGreeks]:
        """Get Greeks breakdown by underlying symbol."""
        underlyings = set()
        for pos in self.positions.values():
            if pos.underlying_symbol:
                underlyings.add(pos.underlying_symbol)
            elif pos.asset_type == "stock":
                underlyings.add(pos.symbol)

        return {ul: self.get_portfolio_greeks(ul) for ul in underlyings}

    def calculate_delta_hedge(
        self,
        underlying: str,
        use_options: bool = False,
    ) -> HedgeRecommendation | None:
        """
        Calculate delta hedge for an underlying.

        Args:
            underlying: Underlying symbol to hedge
            use_options: If True, suggest option hedge; otherwise stock

        Returns:
            HedgeRecommendation or None if already hedged
        """
        greeks = self.get_portfolio_greeks(underlying)
        delta_exposure = greeks.delta - self.targets.target_delta

        # Check if within tolerance
        if abs(delta_exposure) <= self.targets.delta_tolerance:
            return None

        # Calculate hedge quantity
        if use_options:
            # Would need option chain to find appropriate option
            # Simplified: assume ATM option with 0.5 delta
            contracts = int(abs(delta_exposure) / 50)  # 50 delta per contract
            if contracts == 0:
                return None

            action = "buy" if delta_exposure > 0 else "sell"
            option_type = "put" if delta_exposure > 0 else "call"

            return HedgeRecommendation(
                hedge_type=HedgeType.DELTA,
                action=action,
                symbol=f"{underlying}_ATM_{option_type.upper()}",
                asset_type="option",
                quantity=contracts,
                rationale=f"Delta hedge: reduce {delta_exposure:.0f} delta exposure",
                expected_delta_change=-delta_exposure * 0.9,  # Approximate
                priority=1,
            )
        else:
            # Stock hedge
            shares = int(-delta_exposure)  # Negative to offset
            if shares == 0:
                return None

            action = "buy" if shares > 0 else "sell"

            price = self.underlying_prices.get(underlying, 100)

            return HedgeRecommendation(
                hedge_type=HedgeType.DELTA,
                action=action,
                symbol=underlying,
                asset_type="stock",
                quantity=abs(shares),
                rationale=f"Delta hedge: offset {delta_exposure:.0f} delta",
                expected_delta_change=-delta_exposure,
                estimated_cost=abs(shares) * price,
                priority=1,
            )

    def calculate_gamma_hedge(
        self,
        underlying: str,
        available_options: list[dict[str, Any]] | None = None,
    ) -> HedgeRecommendation | None:
        """
        Calculate gamma hedge.

        Gamma can only be hedged with options.
        """
        greeks = self.get_portfolio_greeks(underlying)
        gamma_exposure = greeks.gamma - self.targets.target_gamma

        if abs(gamma_exposure) <= self.targets.gamma_tolerance:
            return None

        # Need to buy/sell options to adjust gamma
        # Positive gamma comes from long options, negative from short
        action = "buy" if gamma_exposure < 0 else "sell"

        # Simplified: estimate contracts needed
        # ATM options have highest gamma, approximately 0.05-0.10 per contract
        estimated_gamma_per_contract = 5.0  # Per 100 shares
        contracts = int(abs(gamma_exposure) / estimated_gamma_per_contract)

        if contracts == 0:
            return None

        return HedgeRecommendation(
            hedge_type=HedgeType.GAMMA,
            action=action,
            symbol=f"{underlying}_ATM_STRADDLE",
            asset_type="option",
            quantity=contracts,
            rationale=f"Gamma hedge: adjust {gamma_exposure:.2f} gamma exposure",
            expected_gamma_change=-gamma_exposure * 0.8,
            priority=2,
        )

    def calculate_vega_hedge(
        self,
        underlying: str,
    ) -> HedgeRecommendation | None:
        """Calculate vega hedge recommendation."""
        greeks = self.get_portfolio_greeks(underlying)
        vega_exposure = greeks.vega - self.targets.target_vega

        if abs(vega_exposure) <= self.targets.vega_tolerance:
            return None

        # Vega is hedged with options - long options = positive vega
        action = "buy" if vega_exposure < 0 else "sell"

        # Longer-dated options have more vega
        estimated_vega_per_contract = 10.0  # Varies with DTE

        contracts = int(abs(vega_exposure) / estimated_vega_per_contract)
        if contracts == 0:
            return None

        return HedgeRecommendation(
            hedge_type=HedgeType.VEGA,
            action=action,
            symbol=f"{underlying}_60DTE_STRADDLE",
            asset_type="option",
            quantity=contracts,
            rationale=f"Vega hedge: adjust {vega_exposure:.0f} vega exposure",
            expected_vega_change=-vega_exposure * 0.7,
            priority=3,
        )

    def get_all_hedge_recommendations(
        self,
        underlying: str | None = None,
    ) -> list[HedgeRecommendation]:
        """
        Get all hedging recommendations.

        Args:
            underlying: Specific underlying to hedge (None for all)

        Returns:
            List of hedge recommendations sorted by priority
        """
        recommendations = []

        if underlying:
            underlyings = [underlying]
        else:
            underlyings = list(self.get_greeks_by_underlying().keys())

        for ul in underlyings:
            # Delta hedge (highest priority)
            delta_hedge = self.calculate_delta_hedge(ul)
            if delta_hedge:
                recommendations.append(delta_hedge)

            # Gamma hedge
            gamma_hedge = self.calculate_gamma_hedge(ul)
            if gamma_hedge:
                recommendations.append(gamma_hedge)

            # Vega hedge
            vega_hedge = self.calculate_vega_hedge(ul)
            if vega_hedge:
                recommendations.append(vega_hedge)

        return sorted(recommendations, key=lambda x: x.priority)

    def is_delta_neutral(self, underlying: str | None = None) -> bool:
        """Check if portfolio is delta neutral."""
        greeks = self.get_portfolio_greeks(underlying)
        return abs(greeks.delta - self.targets.target_delta) <= self.targets.delta_tolerance

    def is_gamma_neutral(self, underlying: str | None = None) -> bool:
        """Check if portfolio is gamma neutral."""
        greeks = self.get_portfolio_greeks(underlying)
        return abs(greeks.gamma - self.targets.target_gamma) <= self.targets.gamma_tolerance

    def integrate_with_algorithm(self, algorithm) -> None:
        """
        Integrate with QuantConnect algorithm for live Greeks tracking.

        INTEGRATION: Call this from algorithm.Initialize()

        Example:
            def Initialize(self):
                self.hedger = PortfolioHedger(targets=HedgeTargets(...))
                self.hedger.integrate_with_algorithm(self)

        Args:
            algorithm: QCAlgorithm instance
        """
        self.algorithm = algorithm
        algorithm.Debug("Portfolio hedger integrated with algorithm")

    def sync_from_algorithm(self, algorithm) -> None:
        """
        Sync positions from QuantConnect algorithm portfolio.

        INTEGRATION: Call this from algorithm.OnData() to update positions

        Note: As of LEAN PR #6720, Greeks use implied volatility and require
        NO warmup period. Greeks are available immediately.

        Example:
            def OnData(self, slice):
                if self.IsWarmingUp:
                    return

                self.hedger.sync_from_algorithm(self)
                greeks = self.hedger.get_portfolio_greeks()
                self.Debug(f"Portfolio Delta: {greeks.delta:.0f}")

        Args:
            algorithm: QCAlgorithm instance
        """
        # Clear existing positions
        self.positions.clear()

        # Iterate through portfolio holdings (canonical QuantConnect pattern)
        for holding in algorithm.Portfolio.Values:
            if not holding.Invested:
                continue

            symbol = holding.Symbol

            # Determine if equity or option
            try:
                from AlgorithmImports import SecurityType

                is_option = holding.Type == SecurityType.Option
                is_equity = holding.Type == SecurityType.Equity
            except ImportError:
                # Fallback: check symbol format
                is_option = "_" in str(symbol) or " " in str(symbol)
                is_equity = not is_option

            if is_equity:
                # Stock position
                position = Position(
                    symbol=str(symbol),
                    asset_type="stock",
                    quantity=holding.Quantity,
                    entry_price=holding.AveragePrice,
                    current_price=holding.Price,
                    underlying_symbol=str(symbol),
                    delta=1.0,  # Stock delta = 1
                )
                self.add_position(position)
                self.underlying_prices[str(symbol)] = holding.Price

            elif is_option:
                # Option position - extract Greeks
                # Greeks are IV-based (LEAN PR #6720), no warmup required
                contract = algorithm.Securities[symbol]

                # Extract underlying symbol
                try:
                    underlying = str(contract.Underlying.Symbol)
                except (AttributeError, KeyError):
                    underlying = ""

                # Get option details
                try:
                    strike = float(contract.StrikePrice)
                    expiry = contract.Expiry
                    from AlgorithmImports import OptionRight

                    option_type = "call" if contract.Right == OptionRight.Call else "put"
                except (AttributeError, ImportError):
                    strike = 0.0
                    expiry = None
                    option_type = ""

                # Access IV-based Greeks (no warmup needed)
                if hasattr(contract, "Greeks") and contract.Greeks:
                    delta = contract.Greeks.Delta
                    gamma = contract.Greeks.Gamma
                    theta = contract.Greeks.Theta
                    vega = contract.Greeks.Vega
                    rho = contract.Greeks.Rho
                else:
                    delta = gamma = theta = vega = rho = 0.0

                position = Position(
                    symbol=str(symbol),
                    asset_type="option",
                    quantity=holding.Quantity,
                    entry_price=holding.AveragePrice,
                    current_price=holding.Price,
                    underlying_symbol=underlying,
                    delta=delta,
                    gamma=gamma,
                    theta=theta,
                    vega=vega,
                    rho=rho,
                    strike=strike,
                    expiry=expiry,
                    option_type=option_type,
                )
                self.add_position(position)

                # Update underlying price
                if underlying and algorithm.Securities.ContainsKey(underlying):
                    self.underlying_prices[underlying] = algorithm.Securities[underlying].Price

    def get_hedge_recommendations_from_chain(
        self,
        algorithm,
        slice,
        underlying: str,
        hedge_type: HedgeType = HedgeType.DELTA,
    ) -> list[HedgeRecommendation]:
        """
        Get specific hedge recommendations using actual option chain data.

        INTEGRATION: Call this from algorithm.OnData() when hedging needed

        Example:
            def OnData(self, slice):
                if not self.hedger.is_delta_neutral("SPY"):
                    recommendations = self.hedger.get_hedge_recommendations_from_chain(
                        self, slice, "SPY", HedgeType.DELTA
                    )
                    for rec in recommendations:
                        self.Debug(f"Hedge: {rec.action} {rec.quantity}x {rec.symbol}")

        Args:
            algorithm: QCAlgorithm instance
            slice: Slice object from OnData
            underlying: Underlying symbol to hedge
            hedge_type: Type of hedge to calculate

        Returns:
            List of specific hedge recommendations with real contract symbols
        """
        recommendations = []

        # First check if hedge is needed
        greeks = self.get_portfolio_greeks(underlying)

        if hedge_type == HedgeType.DELTA:
            delta_exposure = greeks.delta - self.targets.target_delta
            if abs(delta_exposure) <= self.targets.delta_tolerance:
                return []

            # Get option chain for this underlying
            # Note: Requires option subscription in Initialize()
            option_symbol = None
            for symbol in algorithm.Securities.Keys:
                if hasattr(algorithm.Securities[symbol], "Underlying"):
                    if str(algorithm.Securities[symbol].Underlying.Symbol) == underlying:
                        option_symbol = symbol
                        break

            # Python API uses snake_case: option_chains.get()
            chain = slice.option_chains.get(option_symbol)
            if chain is None:
                # Fall back to stock hedge
                shares = int(-delta_exposure)
                if shares != 0:
                    action = "buy" if shares > 0 else "sell"
                    price = (
                        algorithm.Securities[underlying].Price if algorithm.Securities.ContainsKey(underlying) else 100
                    )

                    recommendations.append(
                        HedgeRecommendation(
                            hedge_type=HedgeType.DELTA,
                            action=action,
                            symbol=underlying,
                            asset_type="stock",
                            quantity=abs(shares),
                            rationale=f"Delta hedge: offset {delta_exposure:.0f} delta via stock",
                            expected_delta_change=-delta_exposure,
                            estimated_cost=abs(shares) * price,
                            priority=1,
                        )
                    )
                return recommendations

            # Find appropriate option contract
            spot_price = algorithm.Securities[underlying].Price

            # Filter for ATM options (for delta hedging)
            atm_contracts = []
            for contract in chain:
                # Look for near-ATM contracts
                if abs(contract.Strike - spot_price) <= spot_price * 0.05:  # Within 5% of ATM
                    atm_contracts.append(contract)

            if atm_contracts:
                # Sort by closest to ATM
                atm_contracts.sort(key=lambda c: abs(c.Strike - spot_price))
                best_contract = atm_contracts[0]

                # Calculate contracts needed based on delta
                contract_delta = best_contract.Greeks.Delta if best_contract.Greeks else 0.5
                contracts_needed = int(abs(delta_exposure) / (abs(contract_delta) * 100))

                if contracts_needed > 0:
                    action = "buy" if delta_exposure < 0 else "sell"

                    recommendations.append(
                        HedgeRecommendation(
                            hedge_type=HedgeType.DELTA,
                            action=action,
                            symbol=str(best_contract.Symbol),
                            asset_type="option",
                            quantity=contracts_needed,
                            rationale=f"Delta hedge: offset {delta_exposure:.0f} delta with {contract_delta:.2f} delta contracts",
                            expected_delta_change=-contracts_needed * contract_delta * 100,
                            estimated_cost=contracts_needed * best_contract.AskPrice * 100,
                            priority=1,
                        )
                    )

        return recommendations

    def get_risk_metrics(self) -> dict[str, Any]:
        """Get portfolio risk metrics."""
        greeks = self.get_portfolio_greeks()

        # Calculate dollar risks
        # Dollar delta: P&L for 1% move
        # Simplified: assume average underlying price of 100
        avg_price = 100.0
        if self.underlying_prices:
            avg_price = sum(self.underlying_prices.values()) / len(self.underlying_prices)

        dollar_delta = greeks.delta * avg_price * 0.01  # 1% move
        dollar_gamma = 0.5 * greeks.gamma * (avg_price * 0.01) ** 2
        daily_theta = greeks.theta  # Daily theta (IB-compatible, PR #6720)

        return {
            "portfolio_delta": greeks.delta,
            "portfolio_gamma": greeks.gamma,
            "portfolio_theta": greeks.theta,  # Use daily theta
            "portfolio_vega": greeks.vega,
            "dollar_delta_1pct": dollar_delta,
            "dollar_gamma_1pct": dollar_gamma,
            "daily_theta_decay": daily_theta,
            "is_delta_neutral": self.is_delta_neutral(),
            "is_gamma_neutral": self.is_gamma_neutral(),
            "position_count": len(self.positions),
        }

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get complete portfolio summary."""
        greeks = self.get_portfolio_greeks()
        by_underlying = self.get_greeks_by_underlying()
        recommendations = self.get_all_hedge_recommendations()

        return {
            "timestamp": datetime.now().isoformat(),
            "aggregate_greeks": greeks.to_dict(),
            "by_underlying": {ul: g.to_dict() for ul, g in by_underlying.items()},
            "targets": {
                "delta": self.targets.target_delta,
                "gamma": self.targets.target_gamma,
                "vega": self.targets.target_vega,
            },
            "risk_metrics": self.get_risk_metrics(),
            "hedge_recommendations": [r.to_dict() for r in recommendations],
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
        }


def create_hedger_from_positions(
    positions: list[dict[str, Any]],
    targets: HedgeTargets | None = None,
) -> PortfolioHedger:
    """
    Create portfolio hedger from position list.

    Args:
        positions: List of position dicts
        targets: Target Greeks

    Returns:
        Configured PortfolioHedger
    """
    hedger = PortfolioHedger(targets)

    for pos_data in positions:
        position = Position(
            symbol=pos_data.get("symbol", ""),
            asset_type=pos_data.get("type", pos_data.get("asset_type", "stock")),
            quantity=pos_data.get("quantity", 0),
            entry_price=pos_data.get("entry_price", 0),
            current_price=pos_data.get("current_price", pos_data.get("price", 0)),
            underlying_symbol=pos_data.get("underlying", pos_data.get("underlying_symbol", "")),
            delta=pos_data.get("delta", 1.0 if pos_data.get("type") == "stock" else 0),
            gamma=pos_data.get("gamma", 0),
            theta=pos_data.get("theta", 0),
            vega=pos_data.get("vega", 0),
            strike=pos_data.get("strike", 0),
            option_type=pos_data.get("option_type", ""),
        )

        if pos_data.get("expiry"):
            if isinstance(pos_data["expiry"], datetime):
                position.expiry = pos_data["expiry"]
            elif isinstance(pos_data["expiry"], str):
                position.expiry = datetime.fromisoformat(pos_data["expiry"])

        hedger.add_position(position)

    return hedger


__all__ = [
    "HedgeRecommendation",
    "HedgeTargets",
    "HedgeType",
    "PortfolioHedger",
    "Position",
    "create_hedger_from_positions",
]
