"""
Greeks Calculator Module

UPGRADE-015 Phase 8: Options Analytics Engine

Provides options Greeks calculation:
- First-order Greeks (Delta, Gamma, Theta, Vega, Rho)
- Second-order Greeks (Vanna, Volga, Charm, etc.)
- Position-level aggregation
- Greek sensitivities

Features:
- Black-Scholes based calculations
- Position aggregation
- Scenario analysis
- Risk metrics
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OptionType(Enum):
    """Option type enumeration."""

    CALL = "call"
    PUT = "put"


@dataclass
class GreeksResult:
    """Complete Greeks for an option."""

    # First-order Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0  # Per day
    vega: float = 0.0  # Per 1% IV change
    rho: float = 0.0  # Per 1% rate change

    # Second-order Greeks
    vanna: float = 0.0  # dDelta/dIV
    volga: float = 0.0  # dVega/dIV (also called Vomma)
    charm: float = 0.0  # dDelta/dTime
    veta: float = 0.0  # dVega/dTime
    speed: float = 0.0  # dGamma/dSpot

    # Option info
    option_type: OptionType = OptionType.CALL
    strike: float = 0.0
    spot: float = 0.0
    time_to_expiry: float = 0.0
    iv: float = 0.0
    risk_free_rate: float = 0.0

    # Derived metrics
    dollar_delta: float = 0.0  # Delta * Spot
    dollar_gamma: float = 0.0  # Gamma * Spot * 0.01 (per 1% move)
    dollar_theta: float = 0.0  # Theta in dollar terms

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
            "vanna": self.vanna,
            "volga": self.volga,
            "charm": self.charm,
            "option_type": self.option_type.value,
            "strike": self.strike,
            "spot": self.spot,
            "iv": self.iv,
            "dollar_delta": self.dollar_delta,
            "dollar_gamma": self.dollar_gamma,
            "dollar_theta": self.dollar_theta,
        }


@dataclass
class PositionGreeks:
    """Aggregated Greeks for a position."""

    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    net_rho: float = 0.0

    # Dollar Greeks
    dollar_delta: float = 0.0
    dollar_gamma: float = 0.0
    dollar_theta: float = 0.0
    dollar_vega: float = 0.0

    num_positions: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class GreeksCalculator:
    """Calculator for options Greeks using Black-Scholes model."""

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
    ):
        """
        Initialize Greeks calculator.

        Args:
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Continuous dividend yield (annualized)
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

    # ==========================================================================
    # Core Calculations
    # ==========================================================================

    def calculate(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        iv: float,
        option_type: OptionType | str = OptionType.CALL,
        quantity: int = 1,
    ) -> GreeksResult:
        """
        Calculate all Greeks for an option.

        Args:
            spot: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years (e.g., 30/365 for 30 days)
            iv: Implied volatility (decimal, e.g., 0.25 for 25%)
            option_type: CALL or PUT
            quantity: Position size (negative for short)

        Returns:
            GreeksResult with all Greeks
        """
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())

        # Handle edge cases
        if time_to_expiry <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
            return GreeksResult(
                option_type=option_type,
                strike=strike,
                spot=spot,
                time_to_expiry=time_to_expiry,
                iv=iv,
                risk_free_rate=self.risk_free_rate,
            )

        # Calculate d1 and d2
        d1 = self._d1(spot, strike, time_to_expiry, iv)
        d2 = self._d2(d1, iv, time_to_expiry)

        # Calculate Greeks
        result = GreeksResult(
            option_type=option_type,
            strike=strike,
            spot=spot,
            time_to_expiry=time_to_expiry,
            iv=iv,
            risk_free_rate=self.risk_free_rate,
        )

        # First-order Greeks
        result.delta = self._delta(d1, option_type) * quantity
        result.gamma = self._gamma(spot, d1, time_to_expiry, iv) * abs(quantity)
        result.theta = self._theta(spot, strike, d1, d2, time_to_expiry, iv, option_type) * quantity
        result.vega = self._vega(spot, d1, time_to_expiry) * quantity
        result.rho = self._rho(strike, d2, time_to_expiry, option_type) * quantity

        # Second-order Greeks
        result.vanna = self._vanna(spot, d1, d2, time_to_expiry, iv) * quantity
        result.volga = self._volga(spot, d1, d2, time_to_expiry, iv) * quantity
        result.charm = self._charm(d1, d2, time_to_expiry, iv, option_type) * quantity

        # Dollar Greeks
        result.dollar_delta = result.delta * spot
        result.dollar_gamma = result.gamma * spot * 0.01 * spot  # Per 1% move
        result.dollar_theta = result.theta

        return result

    def calculate_from_dict(
        self,
        option: dict[str, Any],
    ) -> GreeksResult:
        """
        Calculate Greeks from option dictionary.

        Args:
            option: Dict with spot, strike, dte/expiry, iv, type, quantity

        Returns:
            GreeksResult
        """
        spot = option.get("spot", option.get("underlying_price", 100))
        strike = option["strike"]
        dte = option.get("dte", option.get("expiry_days", 30))
        time_to_expiry = dte / 365.0
        iv = option.get("iv", option.get("implied_volatility", 0.25))
        option_type = option.get("type", option.get("option_type", "call"))
        quantity = option.get("quantity", 1)

        return self.calculate(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            iv=iv,
            option_type=option_type,
            quantity=quantity,
        )

    # ==========================================================================
    # Black-Scholes Components
    # ==========================================================================

    def _d1(
        self,
        spot: float,
        strike: float,
        time: float,
        iv: float,
    ) -> float:
        """Calculate d1 in Black-Scholes formula."""
        numerator = math.log(spot / strike) + (self.risk_free_rate - self.dividend_yield + 0.5 * iv**2) * time
        return numerator / (iv * math.sqrt(time))

    def _d2(self, d1: float, iv: float, time: float) -> float:
        """Calculate d2 in Black-Scholes formula."""
        return d1 - iv * math.sqrt(time)

    def _norm_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _norm_pdf(self, x: float) -> float:
        """Standard normal probability density function."""
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

    # ==========================================================================
    # First-Order Greeks
    # ==========================================================================

    def _delta(self, d1: float, option_type: OptionType) -> float:
        """Calculate delta."""
        discount = math.exp(-self.dividend_yield * 1)  # Simplified
        if option_type == OptionType.CALL:
            return discount * self._norm_cdf(d1)
        return discount * (self._norm_cdf(d1) - 1)

    def _gamma(
        self,
        spot: float,
        d1: float,
        time: float,
        iv: float,
    ) -> float:
        """Calculate gamma (same for calls and puts)."""
        discount = math.exp(-self.dividend_yield * time)
        return discount * self._norm_pdf(d1) / (spot * iv * math.sqrt(time))

    def _theta(
        self,
        spot: float,
        strike: float,
        d1: float,
        d2: float,
        time: float,
        iv: float,
        option_type: OptionType,
    ) -> float:
        """Calculate theta (per day)."""
        term1 = -(spot * iv * self._norm_pdf(d1)) / (2 * math.sqrt(time))

        if option_type == OptionType.CALL:
            term2 = -self.risk_free_rate * strike * math.exp(-self.risk_free_rate * time) * self._norm_cdf(d2)
            term3 = self.dividend_yield * spot * self._norm_cdf(d1)
        else:
            term2 = self.risk_free_rate * strike * math.exp(-self.risk_free_rate * time) * self._norm_cdf(-d2)
            term3 = -self.dividend_yield * spot * self._norm_cdf(-d1)

        # Convert to per-day (divide by 365)
        return (term1 + term2 + term3) / 365

    def _vega(
        self,
        spot: float,
        d1: float,
        time: float,
    ) -> float:
        """Calculate vega (per 1% IV change)."""
        discount = math.exp(-self.dividend_yield * time)
        # Per 1% (0.01) change in volatility
        return spot * discount * self._norm_pdf(d1) * math.sqrt(time) / 100

    def _rho(
        self,
        strike: float,
        d2: float,
        time: float,
        option_type: OptionType,
    ) -> float:
        """Calculate rho (per 1% rate change)."""
        discount = math.exp(-self.risk_free_rate * time)
        if option_type == OptionType.CALL:
            return strike * time * discount * self._norm_cdf(d2) / 100
        return -strike * time * discount * self._norm_cdf(-d2) / 100

    # ==========================================================================
    # Second-Order Greeks
    # ==========================================================================

    def _vanna(
        self,
        spot: float,
        d1: float,
        d2: float,
        time: float,
        iv: float,
    ) -> float:
        """Calculate vanna (dDelta/dIV)."""
        return -self._norm_pdf(d1) * d2 / iv / 100

    def _volga(
        self,
        spot: float,
        d1: float,
        d2: float,
        time: float,
        iv: float,
    ) -> float:
        """Calculate volga/vomma (dVega/dIV)."""
        return spot * self._norm_pdf(d1) * math.sqrt(time) * d1 * d2 / iv / 10000

    def _charm(
        self,
        d1: float,
        d2: float,
        time: float,
        iv: float,
        option_type: OptionType,
    ) -> float:
        """Calculate charm (dDelta/dTime)."""
        term1 = self.dividend_yield * math.exp(-self.dividend_yield * time)
        term2_num = 2 * (self.risk_free_rate - self.dividend_yield) * time - d2 * iv * math.sqrt(time)
        term2 = self._norm_pdf(d1) * term2_num / (2 * time * iv * math.sqrt(time))

        if option_type == OptionType.CALL:
            return term1 * self._norm_cdf(d1) - term2
        return -term1 * self._norm_cdf(-d1) - term2

    # ==========================================================================
    # Position Aggregation
    # ==========================================================================

    def aggregate_position(
        self,
        options: list[dict[str, Any]],
        spot: float | None = None,
    ) -> PositionGreeks:
        """
        Calculate aggregated Greeks for a portfolio of options.

        Args:
            options: List of option dictionaries
            spot: Underlying price (uses first option's spot if not provided)

        Returns:
            PositionGreeks with aggregated values
        """
        result = PositionGreeks()

        for opt in options:
            if spot is not None:
                opt["spot"] = spot

            greeks = self.calculate_from_dict(opt)
            quantity = opt.get("quantity", 1)

            result.net_delta += greeks.delta
            result.net_gamma += greeks.gamma
            result.net_theta += greeks.theta
            result.net_vega += greeks.vega
            result.net_rho += greeks.rho

            result.dollar_delta += greeks.dollar_delta
            result.dollar_gamma += greeks.dollar_gamma
            result.dollar_theta += greeks.dollar_theta
            result.dollar_vega += greeks.vega * abs(quantity)

        result.num_positions = len(options)
        return result

    # ==========================================================================
    # Scenario Analysis
    # ==========================================================================

    def scenario_analysis(
        self,
        option: dict[str, Any],
        spot_changes: list[float] | None = None,
        iv_changes: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Run scenario analysis on an option.

        Args:
            option: Option dictionary
            spot_changes: List of spot price changes (e.g., [-0.1, -0.05, 0, 0.05, 0.1])
            iv_changes: List of IV changes (e.g., [-0.05, -0.025, 0, 0.025, 0.05])

        Returns:
            Scenario results
        """
        if spot_changes is None:
            spot_changes = [-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10]
        if iv_changes is None:
            iv_changes = [-0.05, -0.025, 0, 0.025, 0.05]

        base_spot = option.get("spot", 100)
        base_iv = option.get("iv", 0.25)

        results = {
            "base_greeks": self.calculate_from_dict(option).to_dict(),
            "spot_scenarios": [],
            "iv_scenarios": [],
        }

        # Spot scenarios
        for change in spot_changes:
            new_spot = base_spot * (1 + change)
            opt_copy = option.copy()
            opt_copy["spot"] = new_spot
            greeks = self.calculate_from_dict(opt_copy)
            results["spot_scenarios"].append(
                {
                    "spot_change_pct": change * 100,
                    "new_spot": new_spot,
                    "delta": greeks.delta,
                    "gamma": greeks.gamma,
                }
            )

        # IV scenarios
        for change in iv_changes:
            new_iv = base_iv + change
            if new_iv > 0:
                opt_copy = option.copy()
                opt_copy["iv"] = new_iv
                greeks = self.calculate_from_dict(opt_copy)
                results["iv_scenarios"].append(
                    {
                        "iv_change_pct": change * 100,
                        "new_iv": new_iv,
                        "vega": greeks.vega,
                        "delta": greeks.delta,
                    }
                )

        return results


def create_greeks_calculator(
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
) -> GreeksCalculator:
    """
    Factory function to create a Greeks calculator.

    Args:
        risk_free_rate: Risk-free rate
        dividend_yield: Dividend yield

    Returns:
        Configured GreeksCalculator
    """
    return GreeksCalculator(risk_free_rate, dividend_yield)
