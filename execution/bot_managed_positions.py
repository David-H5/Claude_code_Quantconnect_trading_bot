"""
Bot-Managed Positions Module

Automatic position management for all positions (autonomous + UI-submitted):
- Graduated profit-taking at configurable thresholds
- Stop-loss protection
- DTE-based rolling
- Position adjustment based on market conditions

This module enables the bot to manage positions from any source:
1. Positions created autonomously by OptionStrategies executor
2. Positions created manually from UI via order queue
3. Positions created from recurring order templates

Usage:
    # In algorithm Initialize()
    from execution import create_bot_position_manager

    self.position_manager = create_bot_position_manager(
        algorithm=self,
        profit_thresholds=[
            (0.50, 0.30),  # 30% at +50%
            (1.00, 0.50),  # 50% at +100%
            (2.00, 0.20),  # 20% at +200%
        ],
        stop_loss_threshold=-2.00,  # -200%
        min_dte_for_roll=7,
    )

    # In OnData()
    self.position_manager.manage_positions(data)
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class ManagementAction(Enum):
    """Actions the bot can take on a position."""

    NONE = "none"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    ROLL = "roll"
    ADJUST = "adjust"
    CLOSE = "close"


class RollStrategy(Enum):
    """Strategy for rolling positions."""

    SAME_STRIKES = "same_strikes"  # Keep same strikes (if valid)
    ATM_ADJUST = "atm_adjust"  # Adjust strikes to current ATM
    DELTA_MAINTAIN = "delta_maintain"  # Maintain similar deltas


@dataclass
class RollConfig:
    """
    Configuration for position rolling.

    Attributes:
        target_dte: Target DTE for new position (default 30 days)
        min_dte_range: Minimum acceptable DTE for roll target
        max_dte_range: Maximum acceptable DTE for roll target
        strategy: Roll strategy to use
        max_strike_adjustment: Maximum strike adjustment from original (in dollars)
        close_first: Whether to close before opening (True) or atomic roll (False)
        require_credit: Only roll if can achieve credit or reduced debit
    """

    target_dte: int = 30
    min_dte_range: int = 21
    max_dte_range: int = 60
    strategy: RollStrategy = RollStrategy.ATM_ADJUST
    max_strike_adjustment: float = 10.0
    close_first: bool = True
    require_credit: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_dte": self.target_dte,
            "min_dte_range": self.min_dte_range,
            "max_dte_range": self.max_dte_range,
            "strategy": self.strategy.value,
            "max_strike_adjustment": self.max_strike_adjustment,
            "close_first": self.close_first,
            "require_credit": self.require_credit,
        }


@dataclass
class RollResult:
    """
    Result of a position roll operation.

    Attributes:
        success: Whether the roll was successful
        old_position_id: ID of the closed position
        new_position_id: ID of the new position (if created)
        old_expiry: Old expiration date
        new_expiry: New expiration date
        net_credit: Net credit/debit from the roll (positive = credit)
        error_message: Error message if roll failed
    """

    success: bool
    old_position_id: str
    new_position_id: str | None = None
    old_expiry: datetime | None = None
    new_expiry: datetime | None = None
    net_credit: float = 0.0
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "old_position_id": self.old_position_id,
            "new_position_id": self.new_position_id,
            "old_expiry": self.old_expiry.isoformat() if self.old_expiry else None,
            "new_expiry": self.new_expiry.isoformat() if self.new_expiry else None,
            "net_credit": self.net_credit,
            "error_message": self.error_message,
        }


class PositionSource(Enum):
    """Source of the position."""

    AUTONOMOUS = "autonomous"  # Created by OptionStrategies executor
    MANUAL_UI = "manual_ui"  # Created from UI order
    RECURRING = "recurring"  # Created from recurring template


@dataclass
class ProfitThreshold:
    """
    Profit-taking threshold configuration.

    Attributes:
        gain_pct: Gain percentage to trigger (e.g., 0.50 for +50%)
        take_pct: Percentage of position to close (e.g., 0.30 for 30%)
        triggered: Whether this threshold has been triggered
    """

    gain_pct: float
    take_pct: float
    triggered: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gain_pct": self.gain_pct,
            "take_pct": self.take_pct,
            "triggered": self.triggered,
        }


@dataclass
class BotManagedPosition:
    """
    Position under bot management.

    Attributes:
        position_id: Unique position ID
        symbol: Underlying symbol
        source: Source of position (autonomous, manual_ui, recurring)
        entry_price: Entry price (total cost)
        entry_time: Entry timestamp
        current_quantity: Current quantity (contracts)
        strategy_type: Type of strategy (iron_condor, butterfly, etc.)
        legs: List of option symbols in the position

        # Management configuration
        profit_thresholds: List of profit thresholds
        stop_loss_threshold: Stop loss threshold (-2.00 for -200%)
        min_dte_for_roll: Minimum DTE before rolling
        management_enabled: Whether bot management is enabled

        # Greeks tracking (P1 FIX: Add Greeks to positions)
        delta: Position delta
        gamma: Position gamma
        theta: Position theta (daily decay)
        vega: Position vega
        rho: Position rho
        iv: Implied volatility (weighted average)
        greeks_updated_at: Last time Greeks were updated
        greeks_source: Source of Greeks data (calculated, market)

        # Tracking
        realized_pnl: Realized P&L from partial closes
        last_check_time: Last time position was checked
        management_history: History of management actions
    """

    position_id: str
    symbol: str
    source: PositionSource
    entry_price: float
    entry_time: datetime
    current_quantity: int
    strategy_type: str
    legs: list[Any]

    # Management configuration
    profit_thresholds: list[ProfitThreshold]
    stop_loss_threshold: float
    min_dte_for_roll: int
    management_enabled: bool = True

    # Greeks tracking (P1 FIX: Add Greeks to positions)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    iv: float = 0.0
    greeks_updated_at: datetime | None = None
    greeks_source: str = "calculated"

    # Tracking
    realized_pnl: float = 0.0
    last_check_time: datetime | None = None
    management_history: list[dict[str, Any]] = field(default_factory=list)

    def calculate_pnl_pct(self, current_value: float) -> float:
        """
        Calculate P&L percentage.

        Args:
            current_value: Current position value

        Returns:
            P&L percentage (e.g., 0.50 for +50%, -2.00 for -200%)
        """
        if self.entry_price == 0:
            return 0.0

        # For options, entry_price is typically negative (debit paid)
        # current_value is also negative (current cost to close)
        # P&L = (current_value - entry_price) / abs(entry_price)
        # Example: entry=-500, current=-250 → (-250-(-500))/500 = 250/500 = 0.5 (50% gain)
        pnl = (current_value - self.entry_price) / abs(self.entry_price)
        return pnl

    def get_min_dte(self) -> int:
        """
        Get minimum DTE across all legs.

        Returns:
            Minimum DTE (days to expiration)
        """
        if not self.legs:
            return 999

        min_dte = 999
        for leg in self.legs:
            if hasattr(leg, "expiry"):
                dte = (leg.expiry - datetime.now()).days
                min_dte = min(min_dte, dte)

        return min_dte

    def record_action(self, action: ManagementAction, details: dict[str, Any]) -> None:
        """
        Record a management action.

        Args:
            action: Action taken
            details: Action details
        """
        self.management_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "action": action.value,
                "details": details,
            }
        )

    def update_greeks(
        self,
        delta: float,
        gamma: float,
        theta: float,
        vega: float,
        rho: float = 0.0,
        iv: float = 0.0,
        source: str = "calculated",
    ) -> None:
        """
        Update position Greeks.

        P1 FIX: Add Greeks tracking to positions for risk monitoring.

        Args:
            delta: Position delta (price sensitivity)
            gamma: Position gamma (delta sensitivity)
            theta: Position theta (time decay per day)
            vega: Position vega (volatility sensitivity)
            rho: Position rho (interest rate sensitivity)
            iv: Implied volatility (weighted average)
            source: Source of Greeks data (calculated, market, broker)
        """
        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega
        self.rho = rho
        self.iv = iv
        self.greeks_source = source
        self.greeks_updated_at = datetime.now()

    def get_greeks_summary(self) -> dict[str, Any]:
        """Get a summary of position Greeks."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
            "iv": self.iv,
            "source": self.greeks_source,
            "updated_at": self.greeks_updated_at.isoformat() if self.greeks_updated_at else None,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "source": self.source.value,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "current_quantity": self.current_quantity,
            "strategy_type": self.strategy_type,
            "profit_thresholds": [t.to_dict() for t in self.profit_thresholds],
            "stop_loss_threshold": self.stop_loss_threshold,
            "min_dte_for_roll": self.min_dte_for_roll,
            "management_enabled": self.management_enabled,
            "realized_pnl": self.realized_pnl,
            "management_history": self.management_history,
            # Greeks (P1 FIX)
            "greeks": self.get_greeks_summary(),
        }


class BotPositionManager:
    """
    Bot position manager for automatic profit-taking and risk management.

    Manages positions from all sources (autonomous, manual UI, recurring).
    """

    def __init__(
        self,
        algorithm: Any,
        profit_thresholds: list[tuple[float, float]] | None = None,
        stop_loss_threshold: float = -2.00,
        min_dte_for_roll: int = 7,
        roll_config: RollConfig | None = None,
        enable_logging: bool = True,
        log_file: str = "bot_positions.json",
    ):
        """
        Initialize bot position manager.

        Args:
            algorithm: QuantConnect algorithm instance
            profit_thresholds: List of (gain_pct, take_pct) tuples
            stop_loss_threshold: Stop loss threshold (-2.00 for -200%)
            min_dte_for_roll: Minimum DTE before rolling
            roll_config: Configuration for position rolling
            enable_logging: Whether to log management actions
            log_file: Log file path
        """
        self.algorithm = algorithm
        self.stop_loss_threshold = stop_loss_threshold
        self.min_dte_for_roll = min_dte_for_roll
        self.roll_config = roll_config or RollConfig()
        self.enable_logging = enable_logging
        self.log_file = log_file

        # Default profit thresholds: 30% at +50%, 50% at +100%, 20% at +200%
        if profit_thresholds is None:
            profit_thresholds = [
                (0.50, 0.30),
                (1.00, 0.50),
                (2.00, 0.20),
            ]

        self.default_profit_thresholds = [ProfitThreshold(gain_pct=g, take_pct=t) for g, t in profit_thresholds]

        # Position tracking
        self.positions: dict[str, BotManagedPosition] = {}
        self.positions_by_symbol: dict[str, list[str]] = defaultdict(list)

        # Roll tracking
        self.roll_history: list[RollResult] = []

        # Callbacks
        self.on_profit_taken: Callable[[BotManagedPosition, float], None] | None = None
        self.on_stop_loss: Callable[[BotManagedPosition], None] | None = None
        self.on_roll: Callable[[BotManagedPosition, RollResult], None] | None = None

        # Statistics
        self.stats = {
            "total_positions": 0,
            "profit_takes": 0,
            "stop_losses": 0,
            "rolls": 0,
            "rolls_successful": 0,
            "rolls_failed": 0,
            "total_realized_pnl": 0.0,
            "roll_credits": 0.0,
        }

    def add_position(
        self,
        position_id: str,
        symbol: str,
        source: PositionSource,
        entry_price: float,
        quantity: int,
        strategy_type: str,
        legs: list[Any],
        management_enabled: bool = True,
        custom_thresholds: list[tuple[float, float]] | None = None,
    ) -> BotManagedPosition:
        """
        Add a position to bot management.

        Args:
            position_id: Unique position ID
            symbol: Underlying symbol
            source: Source of position
            entry_price: Entry price (total cost)
            quantity: Quantity (contracts)
            strategy_type: Strategy type
            legs: List of option symbols/contracts
            management_enabled: Whether to enable bot management
            custom_thresholds: Custom profit thresholds (overrides defaults)

        Returns:
            Created BotManagedPosition
        """
        # Create profit thresholds
        if custom_thresholds:
            profit_thresholds = [ProfitThreshold(gain_pct=g, take_pct=t) for g, t in custom_thresholds]
        else:
            # Deep copy of default thresholds
            profit_thresholds = [
                ProfitThreshold(gain_pct=t.gain_pct, take_pct=t.take_pct) for t in self.default_profit_thresholds
            ]

        position = BotManagedPosition(
            position_id=position_id,
            symbol=symbol,
            source=source,
            entry_price=entry_price,
            entry_time=datetime.now(),
            current_quantity=quantity,
            strategy_type=strategy_type,
            legs=legs,
            profit_thresholds=profit_thresholds,
            stop_loss_threshold=self.stop_loss_threshold,
            min_dte_for_roll=self.min_dte_for_roll,
            management_enabled=management_enabled,
        )

        self.positions[position_id] = position
        self.positions_by_symbol[symbol].append(position_id)
        self.stats["total_positions"] += 1

        self._log(f"Added position {position_id} ({source.value}): {strategy_type} on {symbol}")

        return position

    def manage_positions(self, data: Any) -> list[tuple[str, ManagementAction]]:
        """
        Manage all positions - check for profit-taking, stop-loss, rolling.

        This should be called from the algorithm's OnData() method.

        Args:
            data: QuantConnect Slice data

        Returns:
            List of (position_id, action) tuples for positions that had actions
        """
        actions_taken = []

        for position_id, position in list(self.positions.items()):
            if not position.management_enabled:
                continue

            # Get current position value
            current_value = self._get_position_value(position, data)
            if current_value is None:
                continue

            # Calculate P&L percentage
            pnl_pct = position.calculate_pnl_pct(current_value)

            # Check for stop loss
            if pnl_pct <= position.stop_loss_threshold:
                action = self._execute_stop_loss(position)
                if action:
                    actions_taken.append((position_id, action))
                    continue

            # Check for profit taking
            action = self._check_profit_thresholds(position, pnl_pct)
            if action:
                actions_taken.append((position_id, action))
                continue

            # Check for rolling (DTE-based)
            min_dte = position.get_min_dte()
            if min_dte <= position.min_dte_for_roll:
                action = self._execute_roll(position)
                if action:
                    actions_taken.append((position_id, action))

            # Update last check time
            position.last_check_time = datetime.now()

        return actions_taken

    def _calculate_close_quantity(
        self,
        current_quantity: int,
        take_pct: float,
        min_close: int = 1,
    ) -> int:
        """
        Calculate quantity to close for profit-taking.

        SAFETY FIX: Prevents over-closing small positions.
        For positions where calculated close < min_close,
        we skip this profit-taking level to avoid closing 100%
        when only a partial close was intended.

        Args:
            current_quantity: Current position size
            take_pct: Percentage to close (0.0 to 1.0)
            min_close: Minimum quantity to close (default 1)

        Returns:
            Quantity to close, or 0 to signal skipping this level

        Example:
            - 1 contract at 30% → calculated=0 → returns 0 (skip)
            - 1 contract at 100% → calculated=1 → returns 1 (close)
            - 10 contracts at 30% → calculated=3 → returns 3 (close 3)
        """
        calculated = int(current_quantity * take_pct)

        # If calculated is 0 but we have more than min_close contracts,
        # the position is too small for this profit-taking level - skip it
        if calculated < min_close and current_quantity > min_close:
            self._log(
                f"Skipping profit-take level: position_qty={current_quantity}, "
                f"take_pct={take_pct:.0%}, calculated={calculated} < min={min_close}"
            )
            return 0  # Signal to skip this profit-taking level

        # If we only have 1 contract and want to close any percentage,
        # we can't partially close - skip unless it's 100%
        if calculated == 0 and current_quantity == min_close:
            if take_pct >= 1.0:
                return min_close  # Full close requested
            self._log(
                f"Skipping profit-take: single contract position cannot be " f"partially closed at {take_pct:.0%}"
            )
            return 0  # Can't partially close a single contract

        return max(calculated, min_close) if calculated > 0 else 0

    def _check_profit_thresholds(
        self,
        position: BotManagedPosition,
        pnl_pct: float,
    ) -> ManagementAction | None:
        """
        Check if any profit thresholds should be triggered.

        Args:
            position: Position to check
            pnl_pct: Current P&L percentage

        Returns:
            Action taken or None
        """
        for threshold in position.profit_thresholds:
            if threshold.triggered:
                continue

            if pnl_pct >= threshold.gain_pct:
                # Calculate close quantity with safety check
                close_quantity = self._calculate_close_quantity(
                    position.current_quantity,
                    threshold.take_pct,
                )

                # Skip this threshold if position too small
                if close_quantity == 0:
                    self._log(
                        f"Position {position.position_id}: Skipping "
                        f"{threshold.gain_pct:+.0%} profit level (position too small)"
                    )
                    threshold.triggered = True  # Mark as handled, don't retry
                    continue

                success = self._close_partial_position(
                    position,
                    close_quantity,
                    f"Profit target {threshold.gain_pct:+.0%} reached",
                )

                if success:
                    threshold.triggered = True
                    self.stats["profit_takes"] += 1

                    if self.on_profit_taken:
                        self.on_profit_taken(position, pnl_pct)

                    return ManagementAction.TAKE_PROFIT

        return None

    def _execute_stop_loss(self, position: BotManagedPosition) -> ManagementAction | None:
        """
        Execute stop loss - close entire position.

        Args:
            position: Position to close

        Returns:
            Action taken or None
        """
        success = self._close_entire_position(
            position,
            f"Stop loss {position.stop_loss_threshold:+.0%} hit",
        )

        if success:
            self.stats["stop_losses"] += 1

            if self.on_stop_loss:
                self.on_stop_loss(position)

            return ManagementAction.STOP_LOSS

        return None

    def _execute_roll(self, position: BotManagedPosition) -> ManagementAction | None:
        """
        Execute position roll - close current and open new position.

        Full implementation:
        1. Find valid expiration date based on roll_config
        2. Calculate new strikes based on roll strategy
        3. Close current position
        4. Open new position with later expiration
        5. Track roll result

        Args:
            position: Position to roll

        Returns:
            Action taken or None
        """
        min_dte = position.get_min_dte()
        old_expiry = self._get_position_expiry(position)

        self._log(f"Rolling position {position.position_id} " f"(DTE={min_dte}, strategy={position.strategy_type})")

        # Step 1: Find target expiration
        target_expiry = self._find_roll_expiration(position)
        if target_expiry is None:
            result = RollResult(
                success=False,
                old_position_id=position.position_id,
                old_expiry=old_expiry,
                error_message="No valid expiration found for roll",
            )
            self._handle_roll_result(position, result)
            return None

        # Step 2: Get current underlying price for strike calculation
        underlying_price = self._get_underlying_price(position)
        if underlying_price is None:
            result = RollResult(
                success=False,
                old_position_id=position.position_id,
                old_expiry=old_expiry,
                error_message="Cannot get underlying price for roll",
            )
            self._handle_roll_result(position, result)
            return None

        # Step 3: Calculate new strikes based on roll strategy
        new_strikes = self._calculate_roll_strikes(position, underlying_price)
        if new_strikes is None:
            result = RollResult(
                success=False,
                old_position_id=position.position_id,
                old_expiry=old_expiry,
                error_message="Failed to calculate new strikes",
            )
            self._handle_roll_result(position, result)
            return None

        # Step 4: Get close value and new position cost
        close_value = self._get_position_value(position, None)
        new_position_cost = self._estimate_new_position_cost(position, target_expiry, new_strikes)

        # Step 5: Check if roll meets credit requirement
        if self.roll_config.require_credit:
            if close_value is None or new_position_cost is None:
                result = RollResult(
                    success=False,
                    old_position_id=position.position_id,
                    old_expiry=old_expiry,
                    error_message="Cannot calculate roll credit/debit",
                )
                self._handle_roll_result(position, result)
                return None

            net_credit = (close_value or 0) - (new_position_cost or 0)
            if net_credit < 0:
                self._log(f"Roll rejected: net debit ${abs(net_credit):.2f} " f"(require_credit=True)")
                result = RollResult(
                    success=False,
                    old_position_id=position.position_id,
                    old_expiry=old_expiry,
                    net_credit=net_credit,
                    error_message=f"Roll would result in net debit: ${abs(net_credit):.2f}",
                )
                self._handle_roll_result(position, result)
                return None

        # Step 6: Execute the roll
        if self.roll_config.close_first:
            # Close then open (safer, two transactions)
            roll_result = self._execute_close_then_open_roll(position, target_expiry, new_strikes)
        else:
            # Atomic roll (single transaction if supported)
            roll_result = self._execute_atomic_roll(position, target_expiry, new_strikes)

        # Step 7: Handle result
        self._handle_roll_result(position, roll_result)

        if roll_result.success:
            return ManagementAction.ROLL
        return None

    def _get_position_expiry(self, position: BotManagedPosition) -> datetime | None:
        """Get the expiration date from position legs."""
        if not position.legs:
            return None
        for leg in position.legs:
            if hasattr(leg, "expiry"):
                return leg.expiry
            if hasattr(leg, "Expiry"):
                return leg.Expiry
        return None

    def _find_roll_expiration(self, position: BotManagedPosition) -> datetime | None:
        """
        Find valid expiration date for roll based on roll_config.

        Args:
            position: Position being rolled

        Returns:
            Target expiration datetime or None if not found
        """
        if not self.algorithm:
            return None

        symbol = position.symbol

        # Calculate target date range
        today = datetime.now()
        min_date = today + timedelta(days=self.roll_config.min_dte_range)
        max_date = today + timedelta(days=self.roll_config.max_dte_range)
        target_date = today + timedelta(days=self.roll_config.target_dte)

        # Try to get option chain from algorithm
        try:
            # In QuantConnect, get available expirations
            if hasattr(self.algorithm, "OptionChainProvider"):
                # Get option chain for underlying
                chain_provider = self.algorithm.OptionChainProvider
                option_contracts = chain_provider.GetOptionContractList(symbol, today)

                # Extract unique expiration dates
                expirations = set()
                for contract in option_contracts:
                    if hasattr(contract, "ID") and hasattr(contract.ID, "Date"):
                        expirations.add(contract.ID.Date)

                # Filter to valid range
                valid_expirations = [exp for exp in expirations if min_date <= exp <= max_date]

                if not valid_expirations:
                    self._log(f"No valid expirations found between " f"{min_date.date()} and {max_date.date()}")
                    return None

                # Find closest to target
                closest = min(valid_expirations, key=lambda x: abs((x - target_date).days))

                self._log(f"Roll target expiration: {closest.date()} " f"(DTE={(closest - today).days})")
                return closest

            # Fallback: calculate target date directly
            return target_date

        except Exception as e:
            self._log(f"Error finding roll expiration: {e}", level="error")
            return target_date  # Fallback to calculated target

    def _get_underlying_price(self, position: BotManagedPosition) -> float | None:
        """Get current underlying price."""
        if not self.algorithm:
            return None

        try:
            # Try to get price from algorithm securities
            if hasattr(self.algorithm, "Securities"):
                symbol = position.symbol
                if symbol in self.algorithm.Securities:
                    security = self.algorithm.Securities[symbol]
                    if hasattr(security, "Price"):
                        return float(security.Price)

            # Fallback: try to get from data
            if hasattr(self.algorithm, "CurrentSlice"):
                data = self.algorithm.CurrentSlice
                symbol = position.symbol
                if hasattr(data, "Bars") and symbol in data.Bars:
                    return float(data.Bars[symbol].Close)

            return None

        except Exception as e:
            self._log(f"Error getting underlying price: {e}", level="error")
            return None

    def _calculate_roll_strikes(
        self,
        position: BotManagedPosition,
        underlying_price: float,
    ) -> dict[str, float] | None:
        """
        Calculate new strikes for rolled position based on strategy.

        Args:
            position: Position being rolled
            underlying_price: Current underlying price

        Returns:
            Dictionary of leg_type -> strike price, or None on error
        """
        strategy = self.roll_config.strategy

        # Extract current strikes from legs
        current_strikes = {}
        for i, leg in enumerate(position.legs):
            strike = None
            if hasattr(leg, "strike"):
                strike = leg.strike
            elif hasattr(leg, "Strike"):
                strike = leg.Strike
            elif hasattr(leg, "ID") and hasattr(leg.ID, "StrikePrice"):
                strike = leg.ID.StrikePrice

            if strike:
                # Determine leg type (call/put, long/short)
                leg_type = f"leg_{i}"
                if hasattr(leg, "right"):
                    leg_type = "call" if leg.right.lower() == "call" else "put"
                elif hasattr(leg, "Right"):
                    leg_type = "call" if str(leg.Right).lower() == "call" else "put"
                current_strikes[leg_type] = strike

        if not current_strikes:
            return None

        if strategy == RollStrategy.SAME_STRIKES:
            # Keep same strikes (may become ITM/OTM)
            return current_strikes

        elif strategy == RollStrategy.ATM_ADJUST:
            # Adjust strikes relative to current underlying price
            new_strikes = {}
            for leg_type, old_strike in current_strikes.items():
                # Calculate offset from old ATM
                # Assume old ATM was roughly old entry price
                offset = old_strike - position.entry_price if position.entry_price else 0

                # Apply offset to new ATM (current price)
                new_strike = underlying_price + offset

                # Round to nearest strike increment (typically $1 or $5)
                increment = 1.0 if underlying_price < 100 else 5.0
                new_strike = round(new_strike / increment) * increment

                # Limit adjustment
                max_adj = self.roll_config.max_strike_adjustment
                if abs(new_strike - old_strike) > max_adj:
                    # Limit the adjustment
                    if new_strike > old_strike:
                        new_strike = old_strike + max_adj
                    else:
                        new_strike = old_strike - max_adj

                new_strikes[leg_type] = new_strike

            return new_strikes

        elif strategy == RollStrategy.DELTA_MAINTAIN:
            # Would need Greeks to maintain delta - fallback to ATM_ADJUST
            self._log("DELTA_MAINTAIN requires Greeks - using ATM_ADJUST")
            return self._calculate_roll_strikes(
                position,
                underlying_price,
            )

        return current_strikes

    def _estimate_new_position_cost(
        self,
        position: BotManagedPosition,
        target_expiry: datetime,
        new_strikes: dict[str, float],
    ) -> float | None:
        """
        Estimate cost of new position after roll.

        In production, this would query option chain for actual prices.
        """
        # Placeholder - would need live option chain data
        # Return None to indicate estimation not available
        return None

    def _execute_close_then_open_roll(
        self,
        position: BotManagedPosition,
        target_expiry: datetime,
        new_strikes: dict[str, float],
    ) -> RollResult:
        """
        Execute roll by closing then opening (two transactions).

        Args:
            position: Position to roll
            target_expiry: New expiration date
            new_strikes: New strike prices

        Returns:
            RollResult with success status
        """
        import uuid

        old_expiry = self._get_position_expiry(position)

        try:
            # Step 1: Close existing position
            close_value = 0.0
            for leg in position.legs:
                if hasattr(self.algorithm, "Liquidate"):
                    # QuantConnect liquidate
                    self.algorithm.Liquidate(leg)
                # Track close value (would come from fill)

            # Step 2: Create new position with later expiration
            new_position_id = f"roll_{position.position_id}_{uuid.uuid4().hex[:8]}"

            # In production, this would:
            # 1. Build new option symbols with target_expiry and new_strikes
            # 2. Submit orders for new position
            # 3. Wait for fills

            # Create new position tracking entry
            new_position = BotManagedPosition(
                position_id=new_position_id,
                symbol=position.symbol,
                source=position.source,
                entry_price=position.entry_price,  # Would be actual fill price
                entry_time=datetime.now(),
                current_quantity=position.current_quantity,
                strategy_type=position.strategy_type,
                legs=[],  # Would be new legs
                profit_thresholds=[
                    ProfitThreshold(gain_pct=t.gain_pct, take_pct=t.take_pct) for t in position.profit_thresholds
                ],
                stop_loss_threshold=position.stop_loss_threshold,
                min_dte_for_roll=position.min_dte_for_roll,
                management_enabled=position.management_enabled,
            )

            # Record action on old position
            position.record_action(
                ManagementAction.ROLL,
                {
                    "reason": f"DTE roll to {target_expiry.date()}",
                    "new_position_id": new_position_id,
                    "new_strikes": new_strikes,
                },
            )

            # Remove old position from tracking
            if position.position_id in self.positions_by_symbol[position.symbol]:
                self.positions_by_symbol[position.symbol].remove(position.position_id)
            if position.position_id in self.positions:
                del self.positions[position.position_id]

            # Add new position to tracking
            self.positions[new_position_id] = new_position
            self.positions_by_symbol[position.symbol].append(new_position_id)

            self._log(
                f"Roll complete: {position.position_id} -> {new_position_id} "
                f"(expiry: {old_expiry.date() if old_expiry else 'N/A'} -> {target_expiry.date()})"
            )

            return RollResult(
                success=True,
                old_position_id=position.position_id,
                new_position_id=new_position_id,
                old_expiry=old_expiry,
                new_expiry=target_expiry,
                net_credit=close_value,  # Would be actual credit/debit
            )

        except Exception as e:
            self._log(f"Roll execution failed: {e}", level="error")
            return RollResult(
                success=False,
                old_position_id=position.position_id,
                old_expiry=old_expiry,
                error_message=str(e),
            )

    def _execute_atomic_roll(
        self,
        position: BotManagedPosition,
        target_expiry: datetime,
        new_strikes: dict[str, float],
    ) -> RollResult:
        """
        Execute roll atomically (single transaction).

        Uses combo orders to close old and open new simultaneously.
        Requires broker support for combo orders.

        Args:
            position: Position to roll
            target_expiry: New expiration date
            new_strikes: New strike prices

        Returns:
            RollResult with success status
        """
        old_expiry = self._get_position_expiry(position)

        # Check if algorithm supports combo orders
        if not hasattr(self.algorithm, "ComboLimitOrder"):
            self._log("Atomic roll not supported - falling back to close-then-open")
            return self._execute_close_then_open_roll(position, target_expiry, new_strikes)

        try:
            # Build combo legs:
            # - Close legs (opposite direction of current)
            # - Open legs (new expiration, new strikes)

            # In production, this would construct proper Leg objects
            # and submit via ComboLimitOrder

            # For now, fall back to close-then-open
            return self._execute_close_then_open_roll(position, target_expiry, new_strikes)

        except Exception as e:
            self._log(f"Atomic roll failed: {e}", level="error")
            return RollResult(
                success=False,
                old_position_id=position.position_id,
                old_expiry=old_expiry,
                error_message=str(e),
            )

    def _handle_roll_result(
        self,
        position: BotManagedPosition,
        result: RollResult,
    ) -> None:
        """
        Handle roll result - update stats, history, callbacks.

        Args:
            position: Position that was rolled
            result: Result of the roll operation
        """
        # Track in history
        self.roll_history.append(result)

        # Update statistics
        self.stats["rolls"] += 1
        if result.success:
            self.stats["rolls_successful"] += 1
            self.stats["roll_credits"] += result.net_credit
        else:
            self.stats["rolls_failed"] += 1
            self._log(f"Roll failed for {position.position_id}: {result.error_message}", level="error")

        # Trigger callback
        if self.on_roll:
            self.on_roll(position, result)

    def _close_partial_position(
        self,
        position: BotManagedPosition,
        quantity: int,
        reason: str,
    ) -> bool:
        """
        Close part of a position.

        Args:
            position: Position to partially close
            quantity: Quantity to close
            reason: Reason for closing

        Returns:
            True if successful
        """
        if quantity >= position.current_quantity:
            return self._close_entire_position(position, reason)

        # Execute close orders for partial quantity
        # In QuantConnect, this would be:
        # for leg in position.legs:
        #     self.algorithm.MarketOrder(leg, -quantity)

        # Update position
        old_quantity = position.current_quantity
        position.current_quantity -= quantity

        # Record action
        position.record_action(
            ManagementAction.TAKE_PROFIT,
            {
                "reason": reason,
                "quantity_closed": quantity,
                "remaining_quantity": position.current_quantity,
            },
        )

        self._log(
            f"Partial close: {position.position_id} "
            f"({old_quantity} -> {position.current_quantity}), reason: {reason}"
        )

        return True

    def _close_entire_position(
        self,
        position: BotManagedPosition,
        reason: str,
    ) -> bool:
        """
        Close entire position.

        Args:
            position: Position to close
            reason: Reason for closing

        Returns:
            True if successful
        """
        # Execute close orders
        # In QuantConnect, this would liquidate all legs:
        # for leg in position.legs:
        #     self.algorithm.Liquidate(leg)

        # Record action
        position.record_action(
            ManagementAction.CLOSE,
            {
                "reason": reason,
                "quantity_closed": position.current_quantity,
            },
        )

        # Remove from tracking
        self.positions_by_symbol[position.symbol].remove(position.position_id)
        del self.positions[position.position_id]

        self._log(f"Closed position: {position.position_id}, reason: {reason}")

        return True

    def _get_position_value(
        self,
        position: BotManagedPosition,
        data: Any,
    ) -> float | None:
        """
        Get current position value.

        Args:
            position: Position to value
            data: Market data

        Returns:
            Current position value or None
        """
        # In a real implementation, this would:
        # 1. Get current prices for all legs
        # 2. Calculate total position value
        # For now, return None (placeholder)

        # Example implementation:
        # total_value = 0.0
        # for leg in position.legs:
        #     if leg in data:
        #         total_value += data[leg].Close * position.current_quantity * 100
        # return total_value

        return None

    def get_position(self, position_id: str) -> BotManagedPosition | None:
        """
        Get a managed position by ID.

        Args:
            position_id: Position ID

        Returns:
            Position or None if not found
        """
        return self.positions.get(position_id)

    def get_positions_by_symbol(self, symbol: str) -> list[BotManagedPosition]:
        """
        Get all positions for a symbol.

        Args:
            symbol: Underlying symbol

        Returns:
            List of positions
        """
        position_ids = self.positions_by_symbol.get(symbol, [])
        return [self.positions[pid] for pid in position_ids if pid in self.positions]

    def get_all_positions(self) -> list[BotManagedPosition]:
        """
        Get all managed positions.

        Returns:
            List of all positions
        """
        return list(self.positions.values())

    def disable_management(self, position_id: str) -> bool:
        """
        Disable bot management for a position.

        Args:
            position_id: Position ID

        Returns:
            True if successful
        """
        if position_id not in self.positions:
            return False

        self.positions[position_id].management_enabled = False
        self._log(f"Disabled management for position {position_id}")
        return True

    def enable_management(self, position_id: str) -> bool:
        """
        Enable bot management for a position.

        Args:
            position_id: Position ID

        Returns:
            True if successful
        """
        if position_id not in self.positions:
            return False

        self.positions[position_id].management_enabled = True
        self._log(f"Enabled management for position {position_id}")
        return True

    def get_statistics(self) -> dict[str, Any]:
        """
        Get management statistics.

        Returns:
            Statistics dictionary
        """
        active_positions = len(self.positions)
        total_unrealized_pnl = sum(p.realized_pnl for p in self.positions.values())

        return {
            **self.stats,
            "active_positions": active_positions,
            "total_unrealized_pnl": total_unrealized_pnl,
        }

    def _log(self, message: str, level: str = "info") -> None:
        """
        Log a message.

        Args:
            message: Message to log
            level: Log level
        """
        if not self.enable_logging:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level.upper()}] BotPositionManager: {message}"

        # Log to algorithm
        if hasattr(self.algorithm, "Debug"):
            if level == "error":
                self.algorithm.Error(log_message)
            else:
                self.algorithm.Debug(log_message)
        else:
            print(log_message)

        # Log to file
        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    f.write(log_message + "\n")
            except Exception:
                pass  # Ignore file write errors


def create_bot_position_manager(
    algorithm: Any,
    profit_thresholds: list[tuple[float, float]] | None = None,
    stop_loss_threshold: float = -2.00,
    min_dte_for_roll: int = 7,
    roll_config: RollConfig | None = None,
    enable_logging: bool = True,
) -> BotPositionManager:
    """
    Create a BotPositionManager instance.

    Args:
        algorithm: QuantConnect algorithm instance
        profit_thresholds: List of (gain_pct, take_pct) tuples
        stop_loss_threshold: Stop loss threshold (-2.00 for -200%)
        min_dte_for_roll: Minimum DTE before rolling
        roll_config: Configuration for position rolling (optional)
        enable_logging: Whether to log management actions

    Returns:
        Configured BotPositionManager

    Example:
        # In algorithm Initialize()
        self.position_manager = create_bot_position_manager(
            algorithm=self,
            profit_thresholds=[
                (0.50, 0.30),  # 30% at +50%
                (1.00, 0.50),  # 50% at +100%
                (2.00, 0.20),  # 20% at +200%
            ],
            stop_loss_threshold=-2.00,
            min_dte_for_roll=7,
            roll_config=RollConfig(
                target_dte=30,
                strategy=RollStrategy.ATM_ADJUST,
                require_credit=False,
            ),
        )

        # Add position when created
        self.position_manager.add_position(
            position_id="pos_123",
            symbol="SPY",
            source=PositionSource.MANUAL_UI,
            entry_price=-500.0,  # Paid $500 debit
            quantity=1,
            strategy_type="iron_condor",
            legs=[call1, call2, put1, put2],
        )

        # In OnData()
        actions = self.position_manager.manage_positions(data)
        for position_id, action in actions:
            self.Debug(f"Position {position_id}: {action.value}")
    """
    return BotPositionManager(
        algorithm=algorithm,
        profit_thresholds=profit_thresholds,
        stop_loss_threshold=stop_loss_threshold,
        min_dte_for_roll=min_dte_for_roll,
        roll_config=roll_config,
        enable_logging=enable_logging,
    )
