"""
Options Pricing Models Module

UPGRADE-015 Phase 8: Options Analytics Engine

Provides options pricing models:
- Black-Scholes (European options)
- Binomial Tree (American options)
- Implied volatility solver
- Price comparison tools

Features:
- Multiple pricing models
- American option support
- IV calculation from prices
- Greek-consistent pricing
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ExerciseStyle(Enum):
    """Option exercise style."""

    EUROPEAN = "european"
    AMERICAN = "american"


class OptionType(Enum):
    """Option type enumeration."""

    CALL = "call"
    PUT = "put"


@dataclass
class PricingResult:
    """Result of option pricing calculation."""

    price: float
    model: str
    option_type: OptionType
    exercise_style: ExerciseStyle

    # Inputs
    spot: float = 0.0
    strike: float = 0.0
    time_to_expiry: float = 0.0
    iv: float = 0.0
    risk_free_rate: float = 0.0
    dividend_yield: float = 0.0

    # Additional info
    intrinsic_value: float = 0.0
    time_value: float = 0.0
    early_exercise_premium: float = 0.0  # For American options
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "price": self.price,
            "model": self.model,
            "option_type": self.option_type.value,
            "exercise_style": self.exercise_style.value,
            "spot": self.spot,
            "strike": self.strike,
            "time_to_expiry": self.time_to_expiry,
            "iv": self.iv,
            "intrinsic_value": self.intrinsic_value,
            "time_value": self.time_value,
            "early_exercise_premium": self.early_exercise_premium,
        }


class BlackScholes:
    """Black-Scholes option pricing model for European options."""

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
    ):
        """
        Initialize Black-Scholes pricer.

        Args:
            risk_free_rate: Risk-free rate (annualized)
            dividend_yield: Continuous dividend yield (annualized)
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

    def price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        iv: float,
        option_type: OptionType | str = OptionType.CALL,
    ) -> PricingResult:
        """
        Calculate option price using Black-Scholes formula.

        Args:
            spot: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            iv: Implied volatility (decimal)
            option_type: CALL or PUT

        Returns:
            PricingResult with price and details
        """
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())

        # Handle edge cases
        if time_to_expiry <= 0:
            intrinsic = self._intrinsic_value(spot, strike, option_type)
            return PricingResult(
                price=intrinsic,
                model="black_scholes",
                option_type=option_type,
                exercise_style=ExerciseStyle.EUROPEAN,
                spot=spot,
                strike=strike,
                time_to_expiry=0,
                iv=iv,
                risk_free_rate=self.risk_free_rate,
                dividend_yield=self.dividend_yield,
                intrinsic_value=intrinsic,
                time_value=0,
            )

        d1 = self._d1(spot, strike, time_to_expiry, iv)
        d2 = self._d2(d1, iv, time_to_expiry)

        if option_type == OptionType.CALL:
            price = self._call_price(spot, strike, time_to_expiry, d1, d2)
        else:
            price = self._put_price(spot, strike, time_to_expiry, d1, d2)

        intrinsic = self._intrinsic_value(spot, strike, option_type)
        time_value = max(0, price - intrinsic)

        return PricingResult(
            price=price,
            model="black_scholes",
            option_type=option_type,
            exercise_style=ExerciseStyle.EUROPEAN,
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            iv=iv,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            intrinsic_value=intrinsic,
            time_value=time_value,
        )

    def _d1(
        self,
        spot: float,
        strike: float,
        time: float,
        iv: float,
    ) -> float:
        """Calculate d1."""
        numerator = math.log(spot / strike) + (self.risk_free_rate - self.dividend_yield + 0.5 * iv**2) * time
        return numerator / (iv * math.sqrt(time))

    def _d2(self, d1: float, iv: float, time: float) -> float:
        """Calculate d2."""
        return d1 - iv * math.sqrt(time)

    def _norm_cdf(self, x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _call_price(
        self,
        spot: float,
        strike: float,
        time: float,
        d1: float,
        d2: float,
    ) -> float:
        """Calculate call price."""
        discount_spot = spot * math.exp(-self.dividend_yield * time)
        discount_strike = strike * math.exp(-self.risk_free_rate * time)
        return discount_spot * self._norm_cdf(d1) - discount_strike * self._norm_cdf(d2)

    def _put_price(
        self,
        spot: float,
        strike: float,
        time: float,
        d1: float,
        d2: float,
    ) -> float:
        """Calculate put price."""
        discount_spot = spot * math.exp(-self.dividend_yield * time)
        discount_strike = strike * math.exp(-self.risk_free_rate * time)
        return discount_strike * self._norm_cdf(-d2) - discount_spot * self._norm_cdf(-d1)

    def _intrinsic_value(
        self,
        spot: float,
        strike: float,
        option_type: OptionType,
    ) -> float:
        """Calculate intrinsic value."""
        if option_type == OptionType.CALL:
            return max(0, spot - strike)
        return max(0, strike - spot)

    def implied_volatility(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        option_type: OptionType | str = OptionType.CALL,
        precision: float = 0.0001,
        max_iterations: int = 100,
    ) -> float | None:
        """
        Calculate implied volatility from market price using Newton-Raphson.

        Args:
            market_price: Observed market price
            spot: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            option_type: CALL or PUT
            precision: Desired precision
            max_iterations: Maximum iterations

        Returns:
            Implied volatility or None if cannot converge
        """
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())

        # Initial guess
        iv = 0.25

        for _ in range(max_iterations):
            result = self.price(spot, strike, time_to_expiry, iv, option_type)
            price_diff = result.price - market_price

            if abs(price_diff) < precision:
                return iv

            # Calculate vega for Newton-Raphson
            vega = self._vega(spot, time_to_expiry, iv)

            if abs(vega) < 0.0001:
                # Vega too small, try bisection
                return self._iv_bisection(market_price, spot, strike, time_to_expiry, option_type)

            iv = iv - price_diff / (vega * 100)  # vega is per 1%
            iv = max(0.001, min(5.0, iv))  # Keep IV reasonable

        return iv

    def _vega(
        self,
        spot: float,
        time: float,
        iv: float,
    ) -> float:
        """Calculate vega for Newton-Raphson."""
        d1 = (math.log(spot / spot) + (self.risk_free_rate + 0.5 * iv**2) * time) / (iv * math.sqrt(time))
        norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
        return spot * norm_pdf * math.sqrt(time) / 100

    def _iv_bisection(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time: float,
        option_type: OptionType,
        low: float = 0.001,
        high: float = 3.0,
        iterations: int = 50,
    ) -> float | None:
        """Bisection method for IV when Newton-Raphson fails."""
        for _ in range(iterations):
            mid = (low + high) / 2
            result = self.price(spot, strike, time, mid, option_type)

            if abs(result.price - market_price) < 0.0001:
                return mid

            if result.price > market_price:
                high = mid
            else:
                low = mid

        return (low + high) / 2


class BinomialTree:
    """Binomial Tree option pricing model for American options."""

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        steps: int = 100,
    ):
        """
        Initialize Binomial Tree pricer.

        Args:
            risk_free_rate: Risk-free rate (annualized)
            dividend_yield: Continuous dividend yield (annualized)
            steps: Number of tree steps
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.steps = steps

    def price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        iv: float,
        option_type: OptionType | str = OptionType.CALL,
        exercise_style: ExerciseStyle = ExerciseStyle.AMERICAN,
    ) -> PricingResult:
        """
        Calculate option price using Binomial Tree.

        Args:
            spot: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            iv: Implied volatility (decimal)
            option_type: CALL or PUT
            exercise_style: AMERICAN or EUROPEAN

        Returns:
            PricingResult with price and details
        """
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())

        # Handle edge cases
        if time_to_expiry <= 0:
            intrinsic = self._intrinsic_value(spot, strike, option_type)
            return PricingResult(
                price=intrinsic,
                model="binomial_tree",
                option_type=option_type,
                exercise_style=exercise_style,
                spot=spot,
                strike=strike,
                time_to_expiry=0,
                iv=iv,
                intrinsic_value=intrinsic,
                time_value=0,
            )

        dt = time_to_expiry / self.steps
        u = math.exp(iv * math.sqrt(dt))
        d = 1 / u
        p = (math.exp((self.risk_free_rate - self.dividend_yield) * dt) - d) / (u - d)
        discount = math.exp(-self.risk_free_rate * dt)

        # Build price tree at expiration
        prices = []
        for i in range(self.steps + 1):
            price_at_node = spot * (u ** (self.steps - i)) * (d**i)
            prices.append(self._intrinsic_value(price_at_node, strike, option_type))

        # Work backward through tree
        for step in range(self.steps - 1, -1, -1):
            new_prices = []
            for i in range(step + 1):
                # Expected value from continuing
                hold_value = discount * (p * prices[i] + (1 - p) * prices[i + 1])

                if exercise_style == ExerciseStyle.AMERICAN:
                    # Check early exercise
                    spot_at_node = spot * (u ** (step - i)) * (d**i)
                    exercise_value = self._intrinsic_value(spot_at_node, strike, option_type)
                    new_prices.append(max(hold_value, exercise_value))
                else:
                    new_prices.append(hold_value)

            prices = new_prices

        price = prices[0]
        intrinsic = self._intrinsic_value(spot, strike, option_type)
        time_value = max(0, price - intrinsic)

        # Calculate early exercise premium
        bs = BlackScholes(self.risk_free_rate, self.dividend_yield)
        european_price = bs.price(spot, strike, time_to_expiry, iv, option_type).price
        early_exercise_premium = max(0, price - european_price)

        return PricingResult(
            price=price,
            model="binomial_tree",
            option_type=option_type,
            exercise_style=exercise_style,
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            iv=iv,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            intrinsic_value=intrinsic,
            time_value=time_value,
            early_exercise_premium=early_exercise_premium,
        )

    def _intrinsic_value(
        self,
        spot: float,
        strike: float,
        option_type: OptionType,
    ) -> float:
        """Calculate intrinsic value."""
        if option_type == OptionType.CALL:
            return max(0, spot - strike)
        return max(0, strike - spot)


def create_pricer(
    model: str = "black_scholes",
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
    **kwargs: Any,
) -> BlackScholes | BinomialTree:
    """
    Factory function to create a pricing model.

    Args:
        model: "black_scholes" or "binomial_tree"
        risk_free_rate: Risk-free rate
        dividend_yield: Dividend yield
        **kwargs: Additional model-specific arguments

    Returns:
        Configured pricer
    """
    if model.lower() == "binomial_tree":
        steps = kwargs.get("steps", 100)
        return BinomialTree(risk_free_rate, dividend_yield, steps)

    return BlackScholes(risk_free_rate, dividend_yield)


def compare_prices(
    spot: float,
    strike: float,
    time_to_expiry: float,
    iv: float,
    option_type: OptionType | str = OptionType.CALL,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
) -> dict[str, Any]:
    """
    Compare prices from different models.

    Args:
        spot: Underlying price
        strike: Strike price
        time_to_expiry: Time in years
        iv: Implied volatility
        option_type: CALL or PUT
        risk_free_rate: Risk-free rate
        dividend_yield: Dividend yield

    Returns:
        Comparison of model prices
    """
    bs = BlackScholes(risk_free_rate, dividend_yield)
    bt_american = BinomialTree(risk_free_rate, dividend_yield, steps=100)
    bt_european = BinomialTree(risk_free_rate, dividend_yield, steps=100)

    bs_result = bs.price(spot, strike, time_to_expiry, iv, option_type)
    bt_am_result = bt_american.price(spot, strike, time_to_expiry, iv, option_type, ExerciseStyle.AMERICAN)
    bt_eu_result = bt_european.price(spot, strike, time_to_expiry, iv, option_type, ExerciseStyle.EUROPEAN)

    return {
        "inputs": {
            "spot": spot,
            "strike": strike,
            "time_to_expiry": time_to_expiry,
            "iv": iv,
            "option_type": option_type if isinstance(option_type, str) else option_type.value,
        },
        "prices": {
            "black_scholes": bs_result.price,
            "binomial_european": bt_eu_result.price,
            "binomial_american": bt_am_result.price,
        },
        "early_exercise_premium": bt_am_result.early_exercise_premium,
        "intrinsic_value": bs_result.intrinsic_value,
        "time_value": bs_result.time_value,
    }
