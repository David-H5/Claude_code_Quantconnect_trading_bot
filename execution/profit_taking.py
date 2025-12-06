"""
Profit-Taking Risk Management Model

Implements graduated profit-taking at configurable thresholds.
Designed for QuantConnect's Algorithm Framework.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from config import ProfitTakingConfig, ProfitThreshold


logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class ProfitTakeOrder:
    """Represents a profit-taking order to be executed."""

    symbol: str
    quantity: int
    side: OrderSide
    threshold_triggered: float
    current_gain_pct: float
    sell_pct: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "side": self.side.value,
            "threshold_triggered": self.threshold_triggered,
            "current_gain_pct": self.current_gain_pct,
            "sell_pct": self.sell_pct,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PositionState:
    """Tracks state of a position for profit-taking."""

    symbol: str
    entry_price: float
    current_quantity: int
    original_quantity: int
    thresholds_hit: list[float] = field(default_factory=list)
    highest_gain_pct: float = 0.0
    trailing_stop_price: float | None = None

    @property
    def remaining_pct(self) -> float:
        """Percentage of original position remaining."""
        if self.original_quantity > 0:
            return self.current_quantity / self.original_quantity
        return 0.0


class ProfitTakingRiskModel:
    """
    Risk management model implementing graduated profit-taking.

    Sells portions of profitable positions at configurable thresholds:
    - 50% at +100%
    - 25% at +200%
    - 15% at +400%
    - 100% at +1000%

    Also implements trailing stops for remaining positions.
    """

    def __init__(
        self,
        config: ProfitTakingConfig,
        order_callback: Callable[[ProfitTakeOrder], None] | None = None,
    ):
        """
        Initialize profit-taking model.

        Args:
            config: Profit-taking configuration
            order_callback: Callback when profit-take order generated
        """
        self.config = config
        self.order_callback = order_callback

        # Track position states
        self._positions: dict[str, PositionState] = {}

        # Sort thresholds by gain percentage
        self._thresholds = sorted(config.thresholds, key=lambda t: t.gain_pct)

    def register_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
    ) -> None:
        """
        Register a new position for profit-taking management.

        Args:
            symbol: Position symbol
            entry_price: Entry price
            quantity: Position quantity
        """
        self._positions[symbol] = PositionState(
            symbol=symbol,
            entry_price=entry_price,
            current_quantity=quantity,
            original_quantity=quantity,
        )

    def update_position(self, symbol: str, current_price: float, current_quantity: int) -> list[ProfitTakeOrder]:
        """
        Update position and check for profit-taking triggers.

        Args:
            symbol: Position symbol
            current_price: Current market price
            current_quantity: Current position quantity

        Returns:
            List of profit-take orders to execute
        """
        if not self.config.enabled:
            return []

        if symbol not in self._positions:
            return []

        state = self._positions[symbol]
        state.current_quantity = current_quantity

        if current_quantity <= 0:
            del self._positions[symbol]
            return []

        # Calculate current gain
        gain_pct = (current_price - state.entry_price) / state.entry_price

        # Update highest gain
        state.highest_gain_pct = max(state.highest_gain_pct, gain_pct)

        orders = []

        # Check threshold triggers
        for threshold in self._thresholds:
            if threshold.gain_pct in state.thresholds_hit:
                continue

            if gain_pct >= threshold.gain_pct:
                # Calculate quantity to sell
                sell_quantity = int(state.current_quantity * threshold.sell_pct)

                if sell_quantity > 0:
                    order = ProfitTakeOrder(
                        symbol=symbol,
                        quantity=sell_quantity,
                        side=OrderSide.SELL,
                        threshold_triggered=threshold.gain_pct,
                        current_gain_pct=gain_pct,
                        sell_pct=threshold.sell_pct,
                        reason=threshold.description or f"Profit-take at {threshold.gain_pct:.0%}",
                    )
                    orders.append(order)

                    # Mark threshold as hit
                    state.thresholds_hit.append(threshold.gain_pct)

                    # Update remaining quantity (actual update after execution)
                    state.current_quantity -= sell_quantity

                    # Trigger callback
                    if self.order_callback:
                        self.order_callback(order)

        # Check trailing stop
        if self.config.trailing_stop_enabled and state.current_quantity > 0:
            trailing_order = self._check_trailing_stop(state, current_price, gain_pct)
            if trailing_order:
                orders.append(trailing_order)

        return orders

    def _check_trailing_stop(
        self,
        state: PositionState,
        current_price: float,
        gain_pct: float,
    ) -> ProfitTakeOrder | None:
        """Check and update trailing stop."""
        # Only activate trailing stop after some profit
        if gain_pct < 0.10:  # 10% minimum gain to activate
            return None

        # Calculate trailing stop price
        trailing_pct = self.config.trailing_stop_pct
        stop_price = current_price * (1 - trailing_pct)

        # Update stop if price has risen
        if state.trailing_stop_price is None:
            state.trailing_stop_price = stop_price
        else:
            state.trailing_stop_price = max(state.trailing_stop_price, stop_price)

        # Check if stop is triggered
        if current_price <= state.trailing_stop_price:
            order = ProfitTakeOrder(
                symbol=state.symbol,
                quantity=state.current_quantity,
                side=OrderSide.SELL,
                threshold_triggered=trailing_pct,
                current_gain_pct=gain_pct,
                sell_pct=1.0,
                reason=f"Trailing stop triggered at ${state.trailing_stop_price:.2f}",
            )

            if self.order_callback:
                self.order_callback(order)

            return order

        return None

    def get_position_state(self, symbol: str) -> PositionState | None:
        """Get current state of a position."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, PositionState]:
        """Get all tracked positions."""
        return self._positions.copy()

    def remove_position(self, symbol: str) -> None:
        """Remove a position from tracking."""
        if symbol in self._positions:
            del self._positions[symbol]

    def get_unrealized_profit(self, symbol: str, current_price: float) -> float:
        """
        Calculate unrealized profit for a position.

        Args:
            symbol: Position symbol
            current_price: Current market price

        Returns:
            Unrealized profit amount
        """
        state = self._positions.get(symbol)
        if state is None:
            return 0.0

        return (current_price - state.entry_price) * state.current_quantity

    def get_status(self) -> dict[str, Any]:
        """Get status of all positions."""
        return {
            "enabled": self.config.enabled,
            "positions_tracked": len(self._positions),
            "thresholds": [{"gain": t.gain_pct, "sell": t.sell_pct} for t in self._thresholds],
            "trailing_stop_enabled": self.config.trailing_stop_enabled,
            "trailing_stop_pct": self.config.trailing_stop_pct,
        }


class ProfitTakingRiskManagementModel:
    """
    QuantConnect-compatible Risk Management Model.

    Integrates with Algorithm Framework's risk management pipeline.
    """

    def __init__(self, config: ProfitTakingConfig):
        """Initialize the risk management model."""
        self._profit_taker = ProfitTakingRiskModel(config)

    def ManageRisk(
        self,
        algorithm: Any,
        targets: list[Any],
    ) -> list[Any]:
        """
        Manage risk by adjusting portfolio targets.

        Args:
            algorithm: QCAlgorithm instance
            targets: List of PortfolioTarget objects

        Returns:
            Modified list of PortfolioTarget objects
        """
        # This method would be called by QuantConnect's risk framework
        # Implementation depends on QuantConnect's API
        adjusted_targets = []

        for target in targets:
            symbol = target.Symbol
            security = algorithm.Securities[symbol]
            current_price = security.Price

            # Check for profit-taking orders
            # Note: Access Portfolio directly by symbol, not via .items()
            if symbol in algorithm.Portfolio and algorithm.Portfolio[symbol].Invested:
                holding = algorithm.Portfolio[symbol]
                quantity = holding.Quantity
                entry_price = holding.AveragePrice

                # Register if not already tracking
                if str(symbol) not in self._profit_taker._positions:
                    self._profit_taker.register_position(str(symbol), entry_price, quantity)

                # Update and get orders
                orders = self._profit_taker.update_position(str(symbol), current_price, quantity)

                # If profit-taking triggered, create exit targets
                if orders:
                    total_reduction = sum(order.quantity for order in orders)

                    # Create PortfolioTarget to reduce position
                    # Import PortfolioTarget from QuantConnect if not already done
                    try:
                        from AlgorithmImports import PortfolioTarget

                        # Calculate new target quantity (current - profit-take reduction)
                        new_quantity = quantity - total_reduction

                        # Create exit target for profit-taking
                        exit_target = PortfolioTarget(symbol, new_quantity)
                        adjusted_targets.append(exit_target)

                        # Log profit-taking action
                        algorithm.Debug(
                            f"Profit-taking: Reducing {symbol} by {total_reduction} shares "
                            f"at {current_price:.2f} (Gain: {orders[0].current_gain_pct:.1%})"
                        )

                        # Skip original target since we're modifying it
                        continue

                    except ImportError:
                        # Fallback if PortfolioTarget not available
                        algorithm.Debug(f"Cannot create PortfolioTarget for profit-taking {symbol}")

            adjusted_targets.append(target)

        return adjusted_targets


def create_profit_taking_model(
    config: ProfitTakingConfig | None = None,
    order_callback: Callable[[ProfitTakeOrder], None] | None = None,
) -> ProfitTakingRiskModel:
    """
    Create profit-taking model from configuration.

    Args:
        config: Profit-taking configuration
        order_callback: Optional callback for orders

    Returns:
        Configured ProfitTakingRiskModel instance
    """
    if config is None:
        # Default configuration
        config = ProfitTakingConfig(
            enabled=True,
            thresholds=[
                ProfitThreshold(1.00, 0.50, "Sell 50% at +100%"),
                ProfitThreshold(2.00, 0.25, "Sell 25% at +200%"),
                ProfitThreshold(4.00, 0.15, "Sell 15% at +400%"),
                ProfitThreshold(10.00, 1.00, "Sell remaining at +1000%"),
            ],
            trailing_stop_enabled=True,
            trailing_stop_pct=0.25,
        )

    return ProfitTakingRiskModel(config, order_callback=order_callback)


__all__ = [
    "OrderSide",
    "PositionState",
    "ProfitTakeOrder",
    "ProfitTakingRiskManagementModel",
    "ProfitTakingRiskModel",
    "create_profit_taking_model",
]
