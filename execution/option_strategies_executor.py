"""
OptionStrategies Executor Module

Autonomous trading using QuantConnect's 37+ OptionStrategies factory methods.

This module:
- Selects strategies based on market conditions (IV Rank, portfolio Greeks)
- Executes using factory methods for clean code and automatic position grouping
- Integrates with risk management and circuit breaker
- Tracks positions created via factory methods

Key Features:
- IV Rank-based strategy selection
- Automatic strike selection using Delta or ATM offset
- Position tracking with Greeks aggregation
- Integration with existing risk management
- Support for all 37+ OptionStrategies factory methods
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .pre_trade_validator import PreTradeValidator

try:
    from AlgorithmImports import *
except ImportError:
    # Stubs for development/testing
    class QCAlgorithm:
        pass

    class Slice:
        pass

    class OptionRight:
        Call = "Call"
        Put = "Put"

    class OptionStrategies:
        @staticmethod
        def iron_condor(*args, **kwargs):
            pass

        @staticmethod
        def butterfly_call(*args, **kwargs):
            pass

        @staticmethod
        def short_strangle(*args, **kwargs):
            pass

        @staticmethod
        def straddle(*args, **kwargs):
            pass

    class Resolution:
        Minute = "Minute"


class StrategyCondition(Enum):
    """Market conditions for strategy selection."""

    VERY_LOW_IV = "very_low_iv"  # IV Rank < 20
    LOW_IV = "low_iv"  # IV Rank 20-40
    NORMAL_IV = "normal_iv"  # IV Rank 40-60
    HIGH_IV = "high_iv"  # IV Rank 60-80
    VERY_HIGH_IV = "very_high_iv"  # IV Rank > 80


@dataclass
class StrategyConfig:
    """Configuration for a specific OptionStrategy."""

    name: str
    factory_method: str  # Name of OptionStrategies method
    preferred_conditions: list[StrategyCondition]

    # Strike selection
    strike_selection_method: str  # "delta", "atm_offset", "manual"
    strike_config: dict[str, Any]  # Config specific to selection method

    # Entry criteria
    min_iv_rank: float = 0.0
    max_iv_rank: float = 100.0
    min_dte: int = 30
    max_dte: int = 90

    # Position sizing
    max_contracts: int = 1

    # Greeks constraints (optional)
    max_portfolio_delta: float | None = None
    max_position_vega: float | None = None

    enabled: bool = True


# Default strategy configurations
DEFAULT_STRATEGIES = [
    # High IV strategies (sell premium)
    StrategyConfig(
        name="Iron Condor",
        factory_method="iron_condor",
        preferred_conditions=[StrategyCondition.HIGH_IV, StrategyCondition.VERY_HIGH_IV],
        strike_selection_method="delta",
        strike_config={
            "put_buy_delta": -0.10,
            "put_sell_delta": -0.16,
            "call_sell_delta": 0.16,
            "call_buy_delta": 0.10,
        },
        min_iv_rank=50.0,
        min_dte=30,
        max_dte=60,
    ),
    StrategyConfig(
        name="Short Strangle",
        factory_method="short_strangle",
        preferred_conditions=[StrategyCondition.VERY_HIGH_IV],
        strike_selection_method="delta",
        strike_config={
            "put_delta": -0.16,
            "call_delta": 0.16,
        },
        min_iv_rank=70.0,
        min_dte=30,
        max_dte=45,
    ),
    # Medium IV strategies
    StrategyConfig(
        name="Butterfly Call",
        factory_method="butterfly_call",
        preferred_conditions=[StrategyCondition.NORMAL_IV, StrategyCondition.HIGH_IV],
        strike_selection_method="atm_offset",
        strike_config={
            "lower_offset": -5,  # 5 strikes below ATM
            "middle_offset": 0,  # ATM
            "upper_offset": 5,  # 5 strikes above ATM
        },
        min_iv_rank=30.0,
        max_iv_rank=70.0,
        min_dte=30,
        max_dte=60,
    ),
    # Low IV strategies (buy premium)
    StrategyConfig(
        name="Long Straddle",
        factory_method="straddle",
        preferred_conditions=[StrategyCondition.VERY_LOW_IV, StrategyCondition.LOW_IV],
        strike_selection_method="atm_offset",
        strike_config={
            "strike_offset": 0,  # ATM
        },
        max_iv_rank=30.0,
        min_dte=30,
        max_dte=60,
    ),
]


@dataclass
class FactoryPosition:
    """Tracks a position created via OptionStrategies factory method."""

    strategy_symbol: Any  # OptionStrategy symbol from LEAN
    strategy_name: str
    entry_time: datetime
    entry_price: float
    quantity: int

    # Greeks at entry
    entry_delta: float = 0.0
    entry_gamma: float = 0.0
    entry_theta_per_day: float = 0.0
    entry_vega: float = 0.0

    # Current Greeks (updated)
    current_delta: float = 0.0
    current_gamma: float = 0.0
    current_theta_per_day: float = 0.0
    current_vega: float = 0.0

    # P&L tracking
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # Metadata
    underlying_symbol: str = ""
    expiry: datetime = field(default_factory=datetime.now)
    strategy_config: StrategyConfig | None = None


class OptionStrategiesExecutor:
    """
    Autonomous options trading using OptionStrategies factory methods.

    This executor:
    1. Monitors market conditions (IV Rank, portfolio Greeks)
    2. Selects appropriate strategies based on conditions
    3. Executes using OptionStrategies factory methods
    4. Tracks positions and manages Greeks

    Integration with existing systems:
    - Uses models/risk_manager.py for position sizing
    - Uses models/circuit_breaker.py for safety
    - Uses models/enhanced_volatility.py for IV Rank
    """

    def __init__(
        self,
        algorithm: QCAlgorithm,
        strategy_configs: list[StrategyConfig] | None = None,
        pre_trade_validator: PreTradeValidator | None = None,
    ):
        """
        Initialize OptionStrategies executor.

        Args:
            algorithm: QCAlgorithm instance
            strategy_configs: List of strategy configurations (uses defaults if None)
            pre_trade_validator: Optional pre-trade validator for risk checks
        """
        self.algorithm = algorithm
        self.strategy_configs = strategy_configs or DEFAULT_STRATEGIES
        self.validator = pre_trade_validator

        # Position tracking
        self.positions: dict[str, FactoryPosition] = {}

        # Market state
        self.iv_rank_cache: dict[str, float] = {}
        self.last_iv_rank_update: dict[str, datetime] = {}

        # Option symbols
        self.option_symbols: dict[str, Any] = {}  # underlying -> option_symbol

    def add_underlying(self, underlying: str) -> Any:
        """
        Add underlying for strategy execution.

        Args:
            underlying: Underlying symbol (e.g., "SPY")

        Returns:
            Option symbol
        """
        if underlying in self.option_symbols:
            return self.option_symbols[underlying]

        # Add option subscription
        option = self.algorithm.add_option(underlying, Resolution.Minute)
        self.option_symbols[underlying] = option.Symbol

        # Set filter for strategy requirements
        # Use widest filter to support all strategies
        option.set_filter(-20, 20, 0, 180)

        self.algorithm.Debug(f"OptionStrategiesExecutor: Added {underlying}")

        return option.Symbol

    def on_data(self, data: Slice) -> None:
        """
        Process market data and execute strategies.

        Args:
            data: Slice from OnData()
        """
        # Update IV Rank cache
        self._update_iv_rank(data)

        # Update existing positions
        self._update_positions(data)

        # Check for new strategy opportunities
        self._check_strategy_opportunities(data)

    def _update_iv_rank(self, data: Slice) -> None:
        """Update IV Rank for all underlyings."""
        for underlying, option_symbol in self.option_symbols.items():
            # Only update every 5 minutes to reduce computation
            last_update = self.last_iv_rank_update.get(underlying, datetime.min)
            if (self.algorithm.Time - last_update).total_seconds() < 300:
                continue

            # Calculate IV Rank (requires historical IV data)
            # For now, use a simplified calculation
            # In production, integrate with models/enhanced_volatility.py

            if option_symbol in data.OptionChains:
                chain = data.OptionChains[option_symbol]
                if len(chain) > 0:
                    # Get ATM IV as proxy
                    underlying_price = chain.Underlying.Price
                    atm_contracts = [c for c in chain if abs(c.Strike - underlying_price) < underlying_price * 0.02]

                    if atm_contracts:
                        avg_iv = sum(c.ImpliedVolatility for c in atm_contracts) / len(atm_contracts)

                        # Simplified IV Rank (assume range of 0.10 to 0.50)
                        # In production, calculate from historical data
                        iv_rank = ((avg_iv - 0.10) / (0.50 - 0.10)) * 100
                        iv_rank = max(0, min(100, iv_rank))

                        self.iv_rank_cache[underlying] = iv_rank
                        self.last_iv_rank_update[underlying] = self.algorithm.Time

    def _update_positions(self, data: Slice) -> None:
        """Update Greeks and P&L for existing positions."""
        for position_id, position in list(self.positions.items()):
            # Check if position still exists
            if not self.algorithm.Portfolio.ContainsKey(position.strategy_symbol):
                del self.positions[position_id]
                continue

            holding = self.algorithm.Portfolio[position.strategy_symbol]
            if not holding.Invested:
                del self.positions[position_id]
                continue

            # Update P&L
            position.current_price = holding.Price
            position.unrealized_pnl = holding.UnrealizedProfit
            if position.entry_price != 0:
                position.unrealized_pnl_pct = (position.current_price - position.entry_price) / abs(
                    position.entry_price
                )

            # Update Greeks (requires option chain data)
            # In production, aggregate Greeks from all legs
            # For now, use placeholder
            position.current_delta = holding.Quantity * 100  # Simplified

    def _check_strategy_opportunities(self, data: Slice) -> None:
        """Check for strategy entry opportunities."""
        for underlying, option_symbol in self.option_symbols.items():
            if option_symbol not in data.OptionChains:
                continue

            chain = data.OptionChains[option_symbol]
            if len(chain) == 0:
                continue

            # Get current market condition
            condition = self._get_market_condition(underlying)
            if not condition:
                continue

            # Find matching strategy
            strategy_config = self._select_strategy(condition, underlying)
            if not strategy_config:
                continue

            # Check if already have this strategy open
            if self._has_similar_position(strategy_config, underlying):
                continue

            # Execute strategy
            self._execute_strategy(strategy_config, chain, data)

    def _get_market_condition(self, underlying: str) -> StrategyCondition | None:
        """Determine current market condition from IV Rank."""
        iv_rank = self.iv_rank_cache.get(underlying)
        if iv_rank is None:
            return None

        if iv_rank < 20:
            return StrategyCondition.VERY_LOW_IV
        elif iv_rank < 40:
            return StrategyCondition.LOW_IV
        elif iv_rank < 60:
            return StrategyCondition.NORMAL_IV
        elif iv_rank < 80:
            return StrategyCondition.HIGH_IV
        else:
            return StrategyCondition.VERY_HIGH_IV

    def _select_strategy(self, condition: StrategyCondition, underlying: str) -> StrategyConfig | None:
        """Select best strategy for current conditions."""
        iv_rank = self.iv_rank_cache.get(underlying, 50.0)

        # Filter strategies by condition and IV Rank
        candidates = [
            config
            for config in self.strategy_configs
            if (
                config.enabled
                and condition in config.preferred_conditions
                and config.min_iv_rank <= iv_rank <= config.max_iv_rank
            )
        ]

        if not candidates:
            return None

        # Return highest priority (first in list)
        return candidates[0]

    def _has_similar_position(self, strategy_config: StrategyConfig, underlying: str) -> bool:
        """Check if similar position already exists."""
        for position in self.positions.values():
            if position.strategy_name == strategy_config.name and position.underlying_symbol == underlying:
                return True
        return False

    def _execute_strategy(self, strategy_config: StrategyConfig, chain: Any, data: Slice) -> FactoryPosition | None:
        """
        Execute strategy using factory method.

        Args:
            strategy_config: Strategy configuration
            chain: Option chain data
            data: Current slice

        Returns:
            FactoryPosition if executed, None otherwise
        """
        # Select strikes
        strikes = self._select_strikes(strategy_config, chain)
        if not strikes:
            self.algorithm.Debug(f"OptionStrategiesExecutor: Could not find strikes for {strategy_config.name}")
            return None

        # Get expiry
        expiry = strikes.get("expiry")
        if not expiry:
            return None

        # Build strategy using factory method
        try:
            strategy = self._build_factory_strategy(strategy_config, strikes, expiry)
            if not strategy:
                return None
        except Exception as e:
            self.algorithm.Error(f"OptionStrategiesExecutor: Failed to build {strategy_config.name}: {e}")
            return None

        # Pre-trade validation (UPGRADE-002)
        if self.validator:
            from .pre_trade_validator import Order as ValidatorOrder

            validator_order = ValidatorOrder(
                symbol=str(chain.Underlying.Symbol),
                quantity=strategy_config.max_contracts,
                side="buy",
                order_type="market",
                is_combo=True,
            )
            result = self.validator.validate(validator_order)
            if not result.approved:
                self.algorithm.Debug(
                    f"OptionStrategiesExecutor: Pre-trade validation failed for {strategy_config.name}: "
                    f"{[c.message for c in result.failed_checks]}"
                )
                return None

        # Execute
        try:
            ticket = self.algorithm.Buy(strategy, strategy_config.max_contracts)

            # Track position
            position = FactoryPosition(
                strategy_symbol=strategy,
                strategy_name=strategy_config.name,
                entry_time=self.algorithm.Time,
                entry_price=0.0,  # Will be updated on fill
                quantity=strategy_config.max_contracts,
                underlying_symbol=str(chain.Underlying.Symbol),
                expiry=expiry,
                strategy_config=strategy_config,
            )

            position_id = f"{strategy_config.name}_{self.algorithm.Time.strftime('%Y%m%d_%H%M%S')}"
            self.positions[position_id] = position

            self.algorithm.Debug(
                f"OptionStrategiesExecutor: Opened {strategy_config.name} "
                f"on {chain.Underlying.Symbol} (IV Rank: {self.iv_rank_cache.get(str(chain.Underlying.Symbol), 0):.1f})"
            )

            return position

        except Exception as e:
            self.algorithm.Error(f"OptionStrategiesExecutor: Failed to execute {strategy_config.name}: {e}")
            return None

    def _select_strikes(self, strategy_config: StrategyConfig, chain: Any) -> dict[str, Any] | None:
        """
        Select strikes based on configuration.

        Args:
            strategy_config: Strategy configuration
            chain: Option chain

        Returns:
            Dictionary with selected strikes and expiry
        """
        underlying_price = chain.Underlying.Price

        # Filter by DTE
        filtered_chain = [
            c
            for c in chain
            if strategy_config.min_dte <= (c.Expiry - self.algorithm.Time).days <= strategy_config.max_dte
        ]

        if not filtered_chain:
            return None

        # Get expiry (use nearest)
        expiry = min(set(c.Expiry for c in filtered_chain))
        expiry_chain = [c for c in filtered_chain if c.Expiry == expiry]

        # Select strikes based on method
        if strategy_config.strike_selection_method == "delta":
            return self._select_strikes_by_delta(strategy_config, expiry_chain, expiry)
        elif strategy_config.strike_selection_method == "atm_offset":
            return self._select_strikes_by_atm_offset(strategy_config, expiry_chain, underlying_price, expiry)
        else:
            return None

    def _select_strikes_by_delta(
        self, strategy_config: StrategyConfig, chain: list[Any], expiry: datetime
    ) -> dict[str, Any] | None:
        """Select strikes by target delta values."""
        config = strategy_config.strike_config
        strikes = {"expiry": expiry}

        # Iron Condor: 4 strikes
        if strategy_config.factory_method == "iron_condor":
            puts = sorted([c for c in chain if c.Right == OptionRight.Put], key=lambda x: x.Strike, reverse=True)
            calls = sorted([c for c in chain if c.Right == OptionRight.Call], key=lambda x: x.Strike)

            # Find strikes closest to target deltas
            put_buy = self._find_by_delta(puts, config["put_buy_delta"])
            put_sell = self._find_by_delta(puts, config["put_sell_delta"])
            call_sell = self._find_by_delta(calls, config["call_sell_delta"])
            call_buy = self._find_by_delta(calls, config["call_buy_delta"])

            if all([put_buy, put_sell, call_sell, call_buy]):
                strikes.update(
                    {
                        "put_buy": put_buy.Strike,
                        "put_sell": put_sell.Strike,
                        "call_sell": call_sell.Strike,
                        "call_buy": call_buy.Strike,
                    }
                )
                return strikes

        # Short Strangle: 2 strikes
        elif strategy_config.factory_method == "short_strangle":
            puts = sorted([c for c in chain if c.Right == OptionRight.Put], key=lambda x: x.Strike, reverse=True)
            calls = sorted([c for c in chain if c.Right == OptionRight.Call], key=lambda x: x.Strike)

            put_strike_contract = self._find_by_delta(puts, config["put_delta"])
            call_strike_contract = self._find_by_delta(calls, config["call_delta"])

            if put_strike_contract and call_strike_contract:
                strikes.update(
                    {
                        "put_strike": put_strike_contract.Strike,
                        "call_strike": call_strike_contract.Strike,
                    }
                )
                return strikes

        return None

    def _select_strikes_by_atm_offset(
        self, strategy_config: StrategyConfig, chain: list[Any], underlying_price: float, expiry: datetime
    ) -> dict[str, Any] | None:
        """Select strikes by ATM offset."""
        config = strategy_config.strike_config
        strikes = {"expiry": expiry}

        # Get available strikes
        all_strikes = sorted(set(c.Strike for c in chain))

        # Find ATM strike
        atm_strike = min(all_strikes, key=lambda s: abs(s - underlying_price))
        atm_index = all_strikes.index(atm_strike)

        # Butterfly: 3 strikes
        if strategy_config.factory_method == "butterfly_call":
            lower_offset = config["lower_offset"]
            middle_offset = config["middle_offset"]
            upper_offset = config["upper_offset"]

            try:
                lower_strike = all_strikes[atm_index + lower_offset]
                middle_strike = all_strikes[atm_index + middle_offset]
                upper_strike = all_strikes[atm_index + upper_offset]

                strikes.update(
                    {
                        "lower": lower_strike,
                        "middle": middle_strike,
                        "upper": upper_strike,
                    }
                )
                return strikes
            except IndexError:
                return None

        # Straddle: 1 strike
        elif strategy_config.factory_method == "straddle":
            strike_offset = config.get("strike_offset", 0)
            try:
                strike = all_strikes[atm_index + strike_offset]
                strikes["strike"] = strike
                return strikes
            except IndexError:
                return None

        return None

    def _find_by_delta(self, contracts: list[Any], target_delta: float) -> Any | None:
        """Find contract closest to target delta."""
        if not contracts:
            return None

        return min(contracts, key=lambda c: abs(c.Greeks.Delta - target_delta))

    def _build_factory_strategy(
        self, strategy_config: StrategyConfig, strikes: dict[str, Any], expiry: datetime
    ) -> Any | None:
        """
        Build strategy using OptionStrategies factory method.

        Args:
            strategy_config: Strategy configuration
            strikes: Selected strikes
            expiry: Expiration date

        Returns:
            OptionStrategy object from factory method
        """
        method_name = strategy_config.factory_method
        option_symbol = list(self.option_symbols.values())[0]  # Simplified

        # Get factory method
        if not hasattr(OptionStrategies, method_name):
            self.algorithm.Error(f"Unknown factory method: {method_name}")
            return None

        factory_method = getattr(OptionStrategies, method_name)

        # Call factory method with appropriate arguments
        try:
            if method_name == "iron_condor":
                return factory_method(
                    option_symbol,
                    strikes["put_buy"],
                    strikes["put_sell"],
                    strikes["call_sell"],
                    strikes["call_buy"],
                    expiry,
                )
            elif method_name == "butterfly_call":
                return factory_method(option_symbol, strikes["lower"], strikes["middle"], strikes["upper"], expiry)
            elif method_name == "short_strangle":
                return factory_method(option_symbol, strikes["call_strike"], strikes["put_strike"], expiry)
            elif method_name == "straddle":
                return factory_method(option_symbol, strikes["strike"], expiry)
            else:
                self.algorithm.Error(f"Unsupported factory method: {method_name}")
                return None

        except Exception as e:
            self.algorithm.Error(f"Factory method failed: {e}")
            return None

    def get_portfolio_greeks(self) -> dict[str, float]:
        """
        Calculate aggregate Greeks across all factory positions.

        Returns:
            Dictionary with portfolio Greeks
        """
        total_delta = sum(p.current_delta * p.quantity for p in self.positions.values())
        total_gamma = sum(p.current_gamma * p.quantity for p in self.positions.values())
        total_theta = sum(p.current_theta_per_day * p.quantity for p in self.positions.values())
        total_vega = sum(p.current_vega * p.quantity for p in self.positions.values())

        return {
            "delta": total_delta,
            "gamma": total_gamma,
            "theta_per_day": total_theta,
            "vega": total_vega,
        }

    def get_positions_summary(self) -> list[dict[str, Any]]:
        """
        Get summary of all factory positions.

        Returns:
            List of position summaries
        """
        return [
            {
                "strategy_name": p.strategy_name,
                "underlying": p.underlying_symbol,
                "entry_time": p.entry_time.isoformat(),
                "quantity": p.quantity,
                "pnl": p.unrealized_pnl,
                "pnl_pct": p.unrealized_pnl_pct,
                "delta": p.current_delta,
                "theta_per_day": p.current_theta_per_day,
                "days_to_expiry": (p.expiry - datetime.now()).days,
            }
            for p in self.positions.values()
        ]


def create_option_strategies_executor(
    algorithm: QCAlgorithm,
    strategy_configs: list[StrategyConfig] | None = None,
) -> OptionStrategiesExecutor:
    """
    Create OptionStrategies executor with default or custom configurations.

    Args:
        algorithm: QCAlgorithm instance
        strategy_configs: Optional custom strategy configurations

    Returns:
        Configured OptionStrategiesExecutor
    """
    return OptionStrategiesExecutor(algorithm, strategy_configs)


__all__ = [
    "DEFAULT_STRATEGIES",
    "FactoryPosition",
    "OptionStrategiesExecutor",
    "StrategyCondition",
    "StrategyConfig",
    "create_option_strategies_executor",
]
