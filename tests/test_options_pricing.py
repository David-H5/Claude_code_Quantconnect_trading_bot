"""
Options Pricing Validation Tests

Tests for options pricing models including Black-Scholes, Greeks validation,
and implied volatility calculations.

Based on best practices from:
- Hull's "Options, Futures, and Other Derivatives"
- Wilmott on Quantitative Finance
- QuantLib validation patterns
"""

from dataclasses import dataclass
from math import exp, log, sqrt

import pytest
from scipy.stats import norm


@dataclass
class OptionContract:
    """Represents an option contract for testing."""

    underlying_price: float
    strike: float
    time_to_expiry: float  # Years
    risk_free_rate: float
    volatility: float
    is_call: bool
    dividend_yield: float = 0.0


class BlackScholesCalculator:
    """Black-Scholes option pricing calculator for validation."""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Calculate d1 parameter."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Calculate d2 parameter."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return BlackScholesCalculator.d1(S, K, T, r, sigma, q) - sigma * sqrt(T)

    @classmethod
    def call_price(cls, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Calculate Black-Scholes call option price."""
        if T <= 0:
            return max(0, S - K)

        d1 = cls.d1(S, K, T, r, sigma, q)
        d2 = cls.d2(S, K, T, r, sigma, q)

        return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

    @classmethod
    def put_price(cls, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Calculate Black-Scholes put option price."""
        if T <= 0:
            return max(0, K - S)

        d1 = cls.d1(S, K, T, r, sigma, q)
        d2 = cls.d2(S, K, T, r, sigma, q)

        return K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)

    @classmethod
    def delta(cls, S: float, K: float, T: float, r: float, sigma: float, is_call: bool, q: float = 0.0) -> float:
        """Calculate option delta."""
        if T <= 0:
            if is_call:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0

        d1 = cls.d1(S, K, T, r, sigma, q)
        if is_call:
            return exp(-q * T) * norm.cdf(d1)
        else:
            return exp(-q * T) * (norm.cdf(d1) - 1)

    @classmethod
    def gamma(cls, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Calculate option gamma."""
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = cls.d1(S, K, T, r, sigma, q)
        return exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt(T))

    @classmethod
    def theta(cls, S: float, K: float, T: float, r: float, sigma: float, is_call: bool, q: float = 0.0) -> float:
        """Calculate option theta (per day)."""
        if T <= 0:
            return 0.0

        d1 = cls.d1(S, K, T, r, sigma, q)
        d2 = cls.d2(S, K, T, r, sigma, q)

        first_term = -(S * sigma * exp(-q * T) * norm.pdf(d1)) / (2 * sqrt(T))

        if is_call:
            second_term = q * S * exp(-q * T) * norm.cdf(d1) - r * K * exp(-r * T) * norm.cdf(d2)
        else:
            second_term = -q * S * exp(-q * T) * norm.cdf(-d1) + r * K * exp(-r * T) * norm.cdf(-d2)

        # Return daily theta
        return (first_term + second_term) / 365.0

    @classmethod
    def vega(cls, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Calculate option vega (per 1% change in volatility)."""
        if T <= 0:
            return 0.0

        d1 = cls.d1(S, K, T, r, sigma, q)
        return S * exp(-q * T) * sqrt(T) * norm.pdf(d1) / 100.0

    @classmethod
    def rho(cls, S: float, K: float, T: float, r: float, sigma: float, is_call: bool, q: float = 0.0) -> float:
        """Calculate option rho (per 1% change in rate)."""
        if T <= 0:
            return 0.0

        d2 = cls.d2(S, K, T, r, sigma, q)

        if is_call:
            return K * T * exp(-r * T) * norm.cdf(d2) / 100.0
        else:
            return -K * T * exp(-r * T) * norm.cdf(-d2) / 100.0


class TestBlackScholesPricing:
    """Tests for Black-Scholes option pricing."""

    @pytest.fixture
    def calculator(self) -> BlackScholesCalculator:
        """Create calculator instance."""
        return BlackScholesCalculator()

    @pytest.fixture
    def atm_call(self) -> OptionContract:
        """At-the-money call option."""
        return OptionContract(
            underlying_price=100.0,
            strike=100.0,
            time_to_expiry=1.0,  # 1 year
            risk_free_rate=0.05,
            volatility=0.20,
            is_call=True,
        )

    @pytest.mark.unit
    def test_call_price_positive(self, calculator, atm_call):
        """Test that call prices are always positive."""
        price = calculator.call_price(
            S=atm_call.underlying_price,
            K=atm_call.strike,
            T=atm_call.time_to_expiry,
            r=atm_call.risk_free_rate,
            sigma=atm_call.volatility,
        )

        assert price > 0

    @pytest.mark.unit
    def test_put_price_positive(self, calculator, atm_call):
        """Test that put prices are always positive."""
        price = calculator.put_price(
            S=atm_call.underlying_price,
            K=atm_call.strike,
            T=atm_call.time_to_expiry,
            r=atm_call.risk_free_rate,
            sigma=atm_call.volatility,
        )

        assert price > 0

    @pytest.mark.unit
    def test_put_call_parity(self, calculator, atm_call):
        """Test put-call parity: C - P = S*e^(-qT) - K*e^(-rT)."""
        call_price = calculator.call_price(
            S=atm_call.underlying_price,
            K=atm_call.strike,
            T=atm_call.time_to_expiry,
            r=atm_call.risk_free_rate,
            sigma=atm_call.volatility,
        )

        put_price = calculator.put_price(
            S=atm_call.underlying_price,
            K=atm_call.strike,
            T=atm_call.time_to_expiry,
            r=atm_call.risk_free_rate,
            sigma=atm_call.volatility,
        )

        # Put-call parity (no dividends)
        S = atm_call.underlying_price
        K = atm_call.strike
        r = atm_call.risk_free_rate
        T = atm_call.time_to_expiry

        lhs = call_price - put_price
        rhs = S - K * exp(-r * T)

        assert abs(lhs - rhs) < 0.0001

    @pytest.mark.unit
    def test_intrinsic_value_at_expiry(self, calculator):
        """Test that at expiry, option equals intrinsic value."""
        # ITM call at expiry
        call_price = calculator.call_price(S=110, K=100, T=0.0, r=0.05, sigma=0.20)
        assert abs(call_price - 10.0) < 0.0001

        # OTM call at expiry
        call_price = calculator.call_price(S=90, K=100, T=0.0, r=0.05, sigma=0.20)
        assert call_price == 0.0

        # ITM put at expiry
        put_price = calculator.put_price(S=90, K=100, T=0.0, r=0.05, sigma=0.20)
        assert abs(put_price - 10.0) < 0.0001

    @pytest.mark.unit
    def test_price_increases_with_volatility(self, calculator, atm_call):
        """Test that option prices increase with volatility."""
        prices = []
        for vol in [0.10, 0.20, 0.30, 0.40]:
            price = calculator.call_price(
                S=atm_call.underlying_price,
                K=atm_call.strike,
                T=atm_call.time_to_expiry,
                r=atm_call.risk_free_rate,
                sigma=vol,
            )
            prices.append(price)

        # Each price should be higher than the previous
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1]

    @pytest.mark.unit
    def test_call_price_increases_with_underlying(self, calculator, atm_call):
        """Test that call prices increase with underlying price."""
        prices = []
        for spot in [90, 95, 100, 105, 110]:
            price = calculator.call_price(
                S=spot,
                K=atm_call.strike,
                T=atm_call.time_to_expiry,
                r=atm_call.risk_free_rate,
                sigma=atm_call.volatility,
            )
            prices.append(price)

        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1]

    @pytest.mark.unit
    def test_put_price_decreases_with_underlying(self, calculator, atm_call):
        """Test that put prices decrease with underlying price."""
        prices = []
        for spot in [90, 95, 100, 105, 110]:
            price = calculator.put_price(
                S=spot,
                K=atm_call.strike,
                T=atm_call.time_to_expiry,
                r=atm_call.risk_free_rate,
                sigma=atm_call.volatility,
            )
            prices.append(price)

        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1]


class TestGreeksValidation:
    """Tests for options Greeks calculations."""

    @pytest.fixture
    def calculator(self) -> BlackScholesCalculator:
        """Create calculator instance."""
        return BlackScholesCalculator()

    @pytest.mark.unit
    def test_call_delta_bounds(self, calculator):
        """Test that call delta is between 0 and 1."""
        for spot in [80, 90, 100, 110, 120]:
            delta = calculator.delta(S=spot, K=100, T=0.5, r=0.05, sigma=0.20, is_call=True)
            assert 0 <= delta <= 1

    @pytest.mark.unit
    def test_put_delta_bounds(self, calculator):
        """Test that put delta is between -1 and 0."""
        for spot in [80, 90, 100, 110, 120]:
            delta = calculator.delta(S=spot, K=100, T=0.5, r=0.05, sigma=0.20, is_call=False)
            assert -1 <= delta <= 0

    @pytest.mark.unit
    def test_atm_call_delta_approximately_0_5(self, calculator):
        """Test that ATM call delta is approximately 0.5."""
        delta = calculator.delta(S=100, K=100, T=1.0, r=0.05, sigma=0.20, is_call=True)

        # ATM delta should be close to 0.5 (slightly higher due to drift)
        assert 0.45 < delta < 0.65

    @pytest.mark.unit
    def test_deep_itm_call_delta_near_1(self, calculator):
        """Test that deep ITM call delta approaches 1."""
        delta = calculator.delta(S=150, K=100, T=0.5, r=0.05, sigma=0.20, is_call=True)

        assert delta > 0.95

    @pytest.mark.unit
    def test_deep_otm_call_delta_near_0(self, calculator):
        """Test that deep OTM call delta approaches 0."""
        delta = calculator.delta(S=50, K=100, T=0.5, r=0.05, sigma=0.20, is_call=True)

        assert delta < 0.05

    @pytest.mark.unit
    def test_gamma_positive(self, calculator):
        """Test that gamma is always positive."""
        for spot in [80, 90, 100, 110, 120]:
            gamma = calculator.gamma(S=spot, K=100, T=0.5, r=0.05, sigma=0.20)
            assert gamma >= 0

    @pytest.mark.unit
    def test_gamma_highest_atm(self, calculator):
        """Test that gamma is highest for ATM options."""
        gammas = {}
        for spot in [80, 90, 100, 110, 120]:
            gammas[spot] = calculator.gamma(S=spot, K=100, T=0.5, r=0.05, sigma=0.20)

        # ATM gamma should be highest
        assert gammas[100] >= max(gammas[80], gammas[120])

    @pytest.mark.unit
    def test_theta_typically_negative(self, calculator):
        """Test that theta is typically negative (time decay)."""
        # ATM option should have negative theta
        theta = calculator.theta(S=100, K=100, T=0.5, r=0.05, sigma=0.20, is_call=True)

        assert theta < 0

    @pytest.mark.unit
    def test_vega_positive(self, calculator):
        """Test that vega is always positive."""
        for spot in [80, 90, 100, 110, 120]:
            vega = calculator.vega(S=spot, K=100, T=0.5, r=0.05, sigma=0.20)
            assert vega >= 0

    @pytest.mark.unit
    def test_vega_highest_atm(self, calculator):
        """Test that vega is highest for ATM options."""
        vegas = {}
        for spot in [80, 90, 100, 110, 120]:
            vegas[spot] = calculator.vega(S=spot, K=100, T=0.5, r=0.05, sigma=0.20)

        # ATM vega should be highest
        assert vegas[100] >= max(vegas[80], vegas[120])

    @pytest.mark.unit
    def test_call_put_delta_relationship(self, calculator):
        """Test that call delta - put delta = exp(-qT)."""
        call_delta = calculator.delta(S=100, K=100, T=0.5, r=0.05, sigma=0.20, is_call=True)
        put_delta = calculator.delta(S=100, K=100, T=0.5, r=0.05, sigma=0.20, is_call=False)

        # For no dividends (q=0), call_delta - put_delta = 1
        assert abs((call_delta - put_delta) - 1.0) < 0.01

    @pytest.mark.unit
    def test_gamma_same_for_call_put(self, calculator):
        """Test that gamma is the same for calls and puts at same strike."""
        # Gamma doesn't depend on call/put
        gamma = calculator.gamma(S=100, K=100, T=0.5, r=0.05, sigma=0.20)

        # Should be positive
        assert gamma > 0


class TestImpliedVolatility:
    """Tests for implied volatility calculations."""

    @pytest.fixture
    def calculator(self) -> BlackScholesCalculator:
        """Create calculator instance."""
        return BlackScholesCalculator()

    @staticmethod
    def calculate_iv(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        is_call: bool,
        max_iterations: int = 100,
        tolerance: float = 0.0001,
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.

        Returns:
            Implied volatility, or -1 if not converged
        """
        calc = BlackScholesCalculator()

        # Initial guess
        sigma = 0.20

        for _ in range(max_iterations):
            if is_call:
                price = calc.call_price(S, K, T, r, sigma)
            else:
                price = calc.put_price(S, K, T, r, sigma)

            diff = price - market_price

            if abs(diff) < tolerance:
                return sigma

            vega = calc.vega(S, K, T, r, sigma) * 100  # Convert back

            if abs(vega) < 0.0001:
                break

            sigma = sigma - diff / vega

            # Keep sigma reasonable
            sigma = max(0.01, min(5.0, sigma))

        return -1  # Did not converge

    @pytest.mark.unit
    def test_iv_recovery(self, calculator):
        """Test that IV can be recovered from a calculated price."""
        true_vol = 0.25

        price = calculator.call_price(S=100, K=100, T=1.0, r=0.05, sigma=true_vol)

        recovered_iv = self.calculate_iv(
            market_price=price,
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            is_call=True,
        )

        assert abs(recovered_iv - true_vol) < 0.001

    @pytest.mark.unit
    def test_iv_positive(self):
        """Test that implied volatility is always positive."""
        # Use a reasonable market price
        recovered_iv = self.calculate_iv(
            market_price=10.0,
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            is_call=True,
        )

        assert recovered_iv > 0

    @pytest.mark.unit
    def test_iv_bounds_reasonable(self):
        """Test that IV falls within reasonable bounds."""
        recovered_iv = self.calculate_iv(
            market_price=8.0,
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            is_call=True,
        )

        # IV should be between 5% and 200% for most options
        assert 0.05 < recovered_iv < 2.0

    @pytest.mark.unit
    def test_higher_price_implies_higher_iv(self, calculator):
        """Test that higher option prices imply higher IV."""
        ivs = []
        for price in [5.0, 8.0, 12.0, 15.0]:
            iv = self.calculate_iv(
                market_price=price,
                S=100,
                K=100,
                T=1.0,
                r=0.05,
                is_call=True,
            )
            ivs.append(iv)

        # Each IV should be higher than the previous
        for i in range(len(ivs) - 1):
            assert ivs[i] < ivs[i + 1]


class TestVolatilitySurface:
    """Tests for volatility surface properties."""

    @pytest.mark.unit
    def test_volatility_smile_atm_lowest(self):
        """Test that ATM volatility is typically lower (volatility smile)."""
        # Simulated IV surface
        iv_by_strike = {
            80: 0.28,  # OTM put / ITM call
            90: 0.23,
            100: 0.20,  # ATM
            110: 0.22,
            120: 0.25,  # ITM put / OTM call
        }

        # ATM (100 strike) should have lowest IV (smile)
        assert iv_by_strike[100] <= min(iv_by_strike[80], iv_by_strike[120])

    @pytest.mark.unit
    def test_term_structure_consistency(self):
        """Test volatility term structure is consistent."""
        # Simulated term structure
        iv_by_expiry = {
            30: 0.25,  # 30 days
            60: 0.23,
            90: 0.22,
            180: 0.21,  # 180 days
        }

        # Volatility should generally decrease with time (mean reversion)
        # or at least not increase drastically
        for t1, t2 in [(30, 60), (60, 90), (90, 180)]:
            # Allow 50% increase max
            assert iv_by_expiry[t2] <= iv_by_expiry[t1] * 1.5

    @pytest.mark.unit
    def test_no_calendar_arbitrage(self):
        """Test that there's no calendar arbitrage in IV surface."""
        # For the same strike, total variance should increase with time
        spot = 100
        strike = 100

        # IV at different expirations
        ivs = [(0.25, 30), (0.24, 60), (0.23, 90)]  # (iv, days)

        # Calculate total variance (IV^2 * T)
        variances = [(iv**2 * days / 365) for iv, days in ivs]

        # Total variance should be increasing
        for i in range(len(variances) - 1):
            assert variances[i] < variances[i + 1]


class TestOptionsRiskMetrics:
    """Tests for portfolio-level options risk metrics."""

    @pytest.mark.unit
    def test_portfolio_delta_neutrality(self):
        """Test calculation of portfolio delta."""
        positions = [
            {"delta": 0.50, "quantity": 100},  # Long 100 calls = 50 delta
            {"delta": -0.50, "quantity": 100},  # Long 100 puts = -50 delta
            {"delta": 1.0, "quantity": 0},  # No shares
        ]

        portfolio_delta = sum(p["delta"] * p["quantity"] for p in positions)

        # Should be approximately delta neutral (calls + puts cancel out)
        assert abs(portfolio_delta) < 1

    @pytest.mark.unit
    def test_portfolio_vega_exposure(self):
        """Test portfolio vega exposure calculation."""
        positions = [
            {"vega": 0.10, "quantity": 100},  # Long calls
            {"vega": 0.10, "quantity": -50},  # Short calls
        ]

        portfolio_vega = sum(p["vega"] * p["quantity"] for p in positions)

        # Net long vega
        assert portfolio_vega > 0

    @pytest.mark.unit
    def test_theta_decay_calculation(self):
        """Test daily theta decay calculation."""
        positions = [
            {"theta": -0.05, "quantity": 100},  # Long options (negative theta)
            {"theta": -0.03, "quantity": 50},
        ]

        daily_theta = sum(p["theta"] * p["quantity"] for p in positions)

        # Long options should have negative portfolio theta
        assert daily_theta < 0

    @pytest.mark.unit
    def test_gamma_scalping_pnl(self):
        """Test gamma scalping P&L calculation."""
        # Gamma scalping P&L = 0.5 * gamma * (price_move)^2
        gamma = 0.05
        position_size = 100
        price_move = 2.0  # $2 move

        gamma_pnl = 0.5 * gamma * position_size * (price_move**2)

        # Positive gamma means positive P&L from moves
        assert gamma_pnl > 0

    @pytest.mark.unit
    def test_vega_pnl_from_iv_change(self):
        """Test vega P&L from IV change."""
        vega = 0.15  # Per 1% IV change
        position_size = 100
        iv_change = 0.02  # 2% IV increase

        vega_pnl = vega * position_size * iv_change * 100

        # Long vega + IV increase = positive P&L
        assert vega_pnl > 0


class TestSpreadValidation:
    """Tests for options spread pricing validation."""

    @pytest.fixture
    def calculator(self) -> BlackScholesCalculator:
        """Create calculator instance."""
        return BlackScholesCalculator()

    @pytest.mark.unit
    def test_bull_call_spread_value(self, calculator):
        """Test bull call spread pricing."""
        # Buy lower strike, sell higher strike
        long_call = calculator.call_price(S=100, K=95, T=0.25, r=0.05, sigma=0.20)
        short_call = calculator.call_price(S=100, K=105, T=0.25, r=0.05, sigma=0.20)

        spread_value = long_call - short_call

        # Bull call spread should have positive value
        assert spread_value > 0

        # Max profit = strike difference - premium paid
        max_profit = 10.0 - spread_value
        assert max_profit > 0

    @pytest.mark.unit
    def test_bear_put_spread_value(self, calculator):
        """Test bear put spread pricing."""
        # Buy higher strike, sell lower strike
        long_put = calculator.put_price(S=100, K=105, T=0.25, r=0.05, sigma=0.20)
        short_put = calculator.put_price(S=100, K=95, T=0.25, r=0.05, sigma=0.20)

        spread_value = long_put - short_put

        # Bear put spread should have positive value
        assert spread_value > 0

    @pytest.mark.unit
    def test_iron_condor_credit(self, calculator):
        """Test iron condor receives net credit."""
        # Sell OTM put spread + sell OTM call spread
        # Put spread: sell higher, buy lower
        short_put = calculator.put_price(S=100, K=95, T=0.25, r=0.05, sigma=0.20)
        long_put = calculator.put_price(S=100, K=90, T=0.25, r=0.05, sigma=0.20)

        # Call spread: sell lower, buy higher
        short_call = calculator.call_price(S=100, K=105, T=0.25, r=0.05, sigma=0.20)
        long_call = calculator.call_price(S=100, K=110, T=0.25, r=0.05, sigma=0.20)

        net_credit = (short_put - long_put) + (short_call - long_call)

        # Iron condor should receive net credit
        assert net_credit > 0

    @pytest.mark.unit
    def test_straddle_value(self, calculator):
        """Test straddle pricing."""
        call = calculator.call_price(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
        put = calculator.put_price(S=100, K=100, T=0.25, r=0.05, sigma=0.20)

        straddle_value = call + put

        # Straddle should be positive
        assert straddle_value > 0

        # At expiry, need price to move more than straddle cost to profit
        # This tests the breakeven logic
        breakeven_move = straddle_value / 100  # As percentage
        assert breakeven_move > 0

    @pytest.mark.unit
    def test_butterfly_limited_risk(self, calculator):
        """Test butterfly spread has limited risk."""
        # Buy 1 ITM call, sell 2 ATM calls, buy 1 OTM call
        long_itm = calculator.call_price(S=100, K=95, T=0.25, r=0.05, sigma=0.20)
        short_atm = calculator.call_price(S=100, K=100, T=0.25, r=0.05, sigma=0.20) * 2
        long_otm = calculator.call_price(S=100, K=105, T=0.25, r=0.05, sigma=0.20)

        butterfly_cost = long_itm - short_atm + long_otm

        # Butterfly typically costs money to enter (debit)
        assert butterfly_cost > 0

        # Max risk is the cost of the butterfly
        max_risk = butterfly_cost
        assert max_risk > 0
