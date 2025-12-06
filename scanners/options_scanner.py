"""
Options Scanner Module

Scans option chains for underpriced options using Greeks analysis,
implied volatility comparisons, and LLM-assisted evaluation.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from config import OptionsScannerConfig


logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option type enumeration."""

    CALL = "call"
    PUT = "put"


@dataclass
class OptionContract:
    """Represents a single option contract."""

    symbol: str
    underlying: str
    option_type: OptionType
    strike: float
    expiry: datetime
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Calculate spread as percentage of mid price."""
        if self.mid_price > 0:
            return self.spread / self.mid_price
        return 0.0

    @property
    def days_to_expiry(self) -> int:
        """Days until expiration."""
        return (self.expiry - datetime.now()).days

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "underlying": self.underlying,
            "type": self.option_type.value,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "mid": self.mid_price,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "iv": self.implied_volatility,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "days_to_expiry": self.days_to_expiry,
        }


@dataclass
class UnderpricedOption:
    """Represents an underpriced option opportunity."""

    contract: OptionContract
    fair_value: float
    underpriced_pct: float
    confidence: float
    reasoning: str
    iv_percentile: float
    historical_iv: float

    @property
    def potential_profit(self) -> float:
        """Potential profit if priced correctly."""
        return self.fair_value - self.contract.mid_price

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contract": self.contract.to_dict(),
            "fair_value": self.fair_value,
            "current_price": self.contract.mid_price,
            "underpriced_pct": self.underpriced_pct,
            "potential_profit": self.potential_profit,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "iv_percentile": self.iv_percentile,
            "historical_iv": self.historical_iv,
        }


class OptionsScanner:
    """
    Scans for underpriced options using multiple valuation methods.

    Combines Black-Scholes fair value estimation with IV analysis
    and optional LLM-assisted evaluation.
    """

    def __init__(
        self,
        config: OptionsScannerConfig,
        alert_callback: Callable[[UnderpricedOption], None] | None = None,
    ):
        """
        Initialize options scanner.

        Args:
            config: Scanner configuration
            alert_callback: Callback for underpriced options found
        """
        self.config = config
        self.alert_callback = alert_callback
        self._iv_history: dict[str, list[float]] = {}
        self.option_symbol = None  # Set by integrate_with_algorithm()
        self.underlying_symbol = None

    def _calculate_black_scholes(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: OptionType,
    ) -> float:
        """
        Calculate Black-Scholes option price.

        Args:
            spot: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate
            volatility: Implied volatility
            option_type: Call or put

        Returns:
            Theoretical option price
        """
        if time_to_expiry <= 0:
            if option_type == OptionType.CALL:
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)

        try:
            from scipy.stats import norm

            d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (
                volatility * math.sqrt(time_to_expiry)
            )
            d2 = d1 - volatility * math.sqrt(time_to_expiry)

            if option_type == OptionType.CALL:
                price = spot * norm.cdf(d1) - strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
            else:
                price = strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)

            return max(0, price)
        except ImportError:
            # Fallback: simple intrinsic value
            if option_type == OptionType.CALL:
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)

    def _calculate_iv_percentile(self, symbol: str, current_iv: float, lookback_days: int = 252) -> float:
        """
        Calculate IV percentile based on historical data.

        Args:
            symbol: Underlying symbol
            current_iv: Current implied volatility
            lookback_days: Days of history to consider

        Returns:
            IV percentile (0-100)
        """
        history = self._iv_history.get(symbol, [])

        if len(history) < 20:
            return 50.0  # Default to middle if insufficient data

        # Use last N days
        recent = history[-lookback_days:]
        count_below = sum(1 for iv in recent if iv < current_iv)

        return (count_below / len(recent)) * 100

    def _update_iv_history(self, symbol: str, iv: float) -> None:
        """Update IV history for symbol."""
        if symbol not in self._iv_history:
            self._iv_history[symbol] = []

        self._iv_history[symbol].append(iv)

        # Keep last 252 trading days
        if len(self._iv_history[symbol]) > 252:
            self._iv_history[symbol] = self._iv_history[symbol][-252:]

    def _passes_filters(self, contract: OptionContract) -> bool:
        """Check if contract passes configured filters."""
        # Days to expiry
        if contract.days_to_expiry < self.config.min_days_to_expiry:
            return False
        if contract.days_to_expiry > self.config.max_days_to_expiry:
            return False

        # Open interest
        if contract.open_interest < self.config.min_open_interest:
            return False

        # Volume
        if contract.volume < self.config.min_volume:
            return False

        # Delta range
        delta_min, delta_max = self.config.target_delta_range
        if abs(contract.delta) < delta_min or abs(contract.delta) > delta_max:
            return False

        return True

    def scan_chain(
        self,
        underlying: str,
        spot_price: float,
        chain: list[OptionContract],
        risk_free_rate: float = 0.05,
    ) -> list[UnderpricedOption]:
        """
        Scan option chain for underpriced options.

        Args:
            underlying: Underlying symbol
            spot_price: Current stock price
            chain: List of option contracts
            risk_free_rate: Risk-free rate for pricing

        Returns:
            List of underpriced option opportunities
        """
        if not self.config.enabled:
            return []

        underpriced = []

        for contract in chain:
            # Apply filters
            if not self._passes_filters(contract):
                continue

            # Calculate theoretical fair value
            time_to_expiry = contract.days_to_expiry / 365.0
            fair_value = self._calculate_black_scholes(
                spot=spot_price,
                strike=contract.strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=contract.implied_volatility,
                option_type=contract.option_type,
            )

            # Calculate underpriced percentage
            current_price = contract.mid_price
            if current_price > 0:
                underpriced_pct = (fair_value - current_price) / current_price
            else:
                continue

            # Check if meets threshold
            if underpriced_pct < self.config.underpriced_threshold:
                continue

            # Calculate IV percentile
            self._update_iv_history(underlying, contract.implied_volatility)
            iv_percentile = self._calculate_iv_percentile(underlying, contract.implied_volatility)

            # Determine confidence based on IV percentile and spread
            confidence = self._calculate_confidence(contract, iv_percentile, underpriced_pct)

            # Generate reasoning
            reasoning = self._generate_reasoning(contract, fair_value, iv_percentile, underpriced_pct)

            opportunity = UnderpricedOption(
                contract=contract,
                fair_value=fair_value,
                underpriced_pct=underpriced_pct,
                confidence=confidence,
                reasoning=reasoning,
                iv_percentile=iv_percentile,
                historical_iv=sum(self._iv_history.get(underlying, [0]))
                / max(len(self._iv_history.get(underlying, [1])), 1),
            )

            underpriced.append(opportunity)

            # Trigger callback
            if self.alert_callback:
                self.alert_callback(opportunity)

        # Sort by confidence and potential profit
        underpriced.sort(key=lambda x: x.confidence * x.underpriced_pct, reverse=True)

        return underpriced

    def integrate_with_algorithm(self, algorithm, underlying_symbol: str):
        """
        Subscribe to option chain data in QuantConnect.

        INTEGRATION: Call this from algorithm.Initialize()

        Example:
            def Initialize(self):
                scanner = create_options_scanner(config)
                scanner.integrate_with_algorithm(self, "SPY")

        Args:
            algorithm: QCAlgorithm instance
            underlying_symbol: Underlying equity symbol (e.g., "SPY")

        Returns:
            Option symbol for tracking
        """
        # Subscribe to underlying equity
        equity = algorithm.AddEquity(underlying_symbol)

        # Subscribe to options
        option = algorithm.AddOption(underlying_symbol)

        # Set filter based on config
        option.SetFilter(
            lambda u: u.Strikes(
                -10,
                +10,  # ±10 strikes from ATM
            ).Expiration(self.config.min_days_to_expiry, self.config.max_days_to_expiry)
        )

        # Store for later use
        self.option_symbol = option.Symbol
        self.underlying_symbol = underlying_symbol

        algorithm.Debug(
            f"Options scanner subscribed to {underlying_symbol} "
            f"({self.config.min_days_to_expiry}-{self.config.max_days_to_expiry} DTE)"
        )

        return option.Symbol

    def scan_from_slice(
        self,
        algorithm,
        slice,
        underlying_symbol: str,
    ) -> list[UnderpricedOption]:
        """
        Scan option chain from QuantConnect OnData slice.

        INTEGRATION: Call this from algorithm.OnData()

        Example:
            def OnData(self, slice):
                if self.IsWarmingUp:
                    return

                opportunities = self.scanner.scan_from_slice(self, slice, "SPY")
                for opp in opportunities:
                    self.Debug(f"Found opportunity: {opp.contract.symbol}")

        Args:
            algorithm: QCAlgorithm instance
            slice: Slice object from OnData
            underlying_symbol: Underlying symbol

        Returns:
            List of underpriced opportunities
        """
        # Check if we have option chain data
        if not hasattr(self, "option_symbol") or self.option_symbol is None:
            algorithm.Debug(
                f"Options scanner not initialized for {underlying_symbol}. " f"Call integrate_with_algorithm() first."
            )
            return []

        # Python API uses snake_case: option_chains.get()
        chain = slice.option_chains.get(self.option_symbol)
        if chain is None:
            return []

        # Get underlying price
        if underlying_symbol in slice.Bars:
            spot_price = slice.Bars[underlying_symbol].Close
        elif algorithm.Securities.ContainsKey(underlying_symbol):
            spot_price = algorithm.Securities[underlying_symbol].Price
        else:
            algorithm.Debug(f"Cannot determine spot price for {underlying_symbol}")
            return []

        # Convert QuantConnect chain to OptionContract list
        contracts = []
        for qc_contract in chain:
            # Extract contract details from QC format
            # Note: As of LEAN PR #6720, Greeks are IV-based (no warmup needed)
            try:
                contract = self._convert_qc_contract(qc_contract, underlying_symbol)
                contracts.append(contract)
            except Exception as e:
                algorithm.Debug(f"Error converting contract {qc_contract.Symbol}: {e}")
                continue

        # Use existing scan_chain logic
        opportunities = self.scan_chain(
            underlying=underlying_symbol,
            spot_price=spot_price,
            chain=contracts,
        )

        return opportunities

    def _convert_qc_contract(
        self,
        qc_contract,
        underlying_symbol: str,
    ) -> OptionContract:
        """
        Convert QuantConnect option contract to our OptionContract format.

        Note: As of LEAN PR #6720, Greeks use implied volatility and require
        NO warmup period. Greeks are available immediately.

        Args:
            qc_contract: QuantConnect option contract object
            underlying_symbol: Underlying symbol

        Returns:
            OptionContract instance
        """
        # Determine option type (OptionRight is from QuantConnect)
        try:
            from AlgorithmImports import OptionRight

            if qc_contract.Right == OptionRight.Call:
                option_type = OptionType.CALL
            else:
                option_type = OptionType.PUT
        except ImportError:
            # Fallback if not in QuantConnect environment
            option_type = OptionType.CALL

        # Access Greeks (IV-based as of LEAN PR #6720, no warmup required)
        delta = qc_contract.Greeks.Delta if qc_contract.Greeks else 0.0
        gamma = qc_contract.Greeks.Gamma if qc_contract.Greeks else 0.0
        # Use ThetaPerDay for IB compatibility (daily theta decay)
        theta = qc_contract.Greeks.ThetaPerDay if qc_contract.Greeks else 0.0
        vega = qc_contract.Greeks.Vega if qc_contract.Greeks else 0.0
        rho = qc_contract.Greeks.Rho if qc_contract.Greeks else 0.0

        # Get IV from contract
        iv = qc_contract.ImpliedVolatility if hasattr(qc_contract, "ImpliedVolatility") else 0.0

        # Get bid/ask/last prices
        bid = qc_contract.BidPrice if hasattr(qc_contract, "BidPrice") else 0.0
        ask = qc_contract.AskPrice if hasattr(qc_contract, "AskPrice") else 0.0
        last = qc_contract.LastPrice if hasattr(qc_contract, "LastPrice") else 0.0

        # Get volume and open interest
        volume = qc_contract.Volume if hasattr(qc_contract, "Volume") else 0
        open_interest = qc_contract.OpenInterest if hasattr(qc_contract, "OpenInterest") else 0

        return OptionContract(
            symbol=str(qc_contract.Symbol),
            underlying=underlying_symbol,
            option_type=option_type,
            strike=float(qc_contract.Strike),
            expiry=qc_contract.Expiry,
            bid=bid,
            ask=ask,
            last=last,
            volume=volume,
            open_interest=open_interest,
            implied_volatility=iv,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
        )

    def _calculate_confidence(
        self,
        contract: OptionContract,
        iv_percentile: float,
        underpriced_pct: float,
    ) -> float:
        """Calculate confidence score for opportunity."""
        confidence = 0.5  # Base confidence

        # Higher confidence if IV is low (below 30th percentile)
        if iv_percentile < 30:
            confidence += 0.2

        # Higher confidence if spread is tight
        if contract.spread_pct < 0.05:
            confidence += 0.1

        # Higher confidence if high volume
        if contract.volume > 1000:
            confidence += 0.1

        # Higher confidence for larger underpricing
        if underpriced_pct > 0.2:
            confidence += 0.1

        return min(confidence, 1.0)

    def _generate_reasoning(
        self,
        contract: OptionContract,
        fair_value: float,
        iv_percentile: float,
        underpriced_pct: float,
    ) -> str:
        """Generate reasoning for the opportunity."""
        reasons = []

        reasons.append(f"Theoretical value ${fair_value:.2f} vs market ${contract.mid_price:.2f}")

        if iv_percentile < 30:
            reasons.append(f"Low IV percentile ({iv_percentile:.0f}%)")
        elif iv_percentile > 70:
            reasons.append(f"High IV percentile ({iv_percentile:.0f}%)")

        if contract.spread_pct < 0.05:
            reasons.append("Tight bid-ask spread")

        if contract.volume > contract.open_interest:
            reasons.append("Unusual volume activity")

        return "; ".join(reasons)

    def get_watchlist_opportunities(
        self,
        watchlist: list[str],
        chain_fetcher: Callable[[str], tuple[float, list[OptionContract]]],
    ) -> dict[str, list[UnderpricedOption]]:
        """
        Scan multiple symbols for opportunities.

        Args:
            watchlist: List of symbols to scan
            chain_fetcher: Function that returns (spot_price, chain) for symbol

        Returns:
            Dictionary mapping symbols to their opportunities
        """
        results = {}

        for symbol in watchlist:
            try:
                spot_price, chain = chain_fetcher(symbol)
                opportunities = self.scan_chain(symbol, spot_price, chain)
                if opportunities:
                    results[symbol] = opportunities
            except (ValueError, KeyError, TypeError) as e:
                logger.warning("Failed to fetch chain for %s: %s", symbol, e)
                continue
            except Exception as e:
                logger.error("Unexpected error scanning %s: %s", symbol, e, exc_info=True)
                continue

        return results


def create_options_scanner(
    config: OptionsScannerConfig | None = None,
    alert_callback: Callable[[UnderpricedOption], None] | None = None,
) -> OptionsScanner:
    """
    Create options scanner from configuration.

    Args:
        config: Scanner configuration
        alert_callback: Optional callback for alerts

    Returns:
        Configured OptionsScanner instance
    """
    if config is None:
        config = OptionsScannerConfig()

    return OptionsScanner(config, alert_callback=alert_callback)


__all__ = [
    "OptionContract",
    "OptionType",
    "OptionsScanner",
    "UnderpricedOption",
    "create_options_scanner",
]
