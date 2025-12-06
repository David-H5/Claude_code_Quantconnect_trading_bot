"""
Pre-Trade Validator for QuantConnect Trading Bot

This module provides pre-trade validation to enforce position limits,
risk constraints, and data quality checks before order submission.

All execution paths MUST call the validator before submitting orders.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class ValidationStatus(Enum):
    """Result of a validation check."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    name: str
    status: ValidationStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASSED


@dataclass
class ValidationResult:
    """Result of all pre-trade validation checks."""

    approved: bool
    checks: list[ValidationCheck] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_utc_now)

    @property
    def failed_checks(self) -> list[ValidationCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def warnings(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.status == ValidationStatus.WARNING]

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "timestamp": self.timestamp.isoformat(),
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
            "failed_count": len(self.failed_checks),
            "warning_count": len(self.warnings),
        }


@dataclass
class Order:
    """Order representation for validation."""

    symbol: str
    quantity: int
    side: str  # "buy" or "sell"
    order_type: str  # "market", "limit", "stop"
    limit_price: float | None = None
    stop_price: float | None = None
    timestamp: datetime = field(default_factory=_utc_now)

    # For multi-leg orders
    legs: list["Order"] = field(default_factory=list)
    is_combo: bool = False


@dataclass
class ValidationConfig:
    """Configuration for pre-trade validation."""

    # Position limits
    max_position_pct: float = 0.25  # 25% max per position
    max_total_exposure_pct: float = 1.0  # 100% max total exposure

    # Risk limits
    max_daily_loss_pct: float = 0.03  # 3% daily loss limit
    max_single_trade_risk_pct: float = 0.02  # 2% risk per trade

    # Data quality
    max_data_age_seconds: int = 5  # Max age for price data
    min_liquidity_volume: int = 100  # Minimum daily volume

    # Order limits
    max_order_value: float = 50000.0  # Max single order value
    min_order_value: float = 100.0  # Min single order value

    # Duplicates
    dedup_window_seconds: int = 1  # Window for duplicate detection

    # Circuit breaker integration
    enforce_circuit_breaker: bool = True


class PreTradeValidator:
    """
    Pre-trade validator for enforcing position limits and risk constraints.

    All execution paths must call validate() before submitting orders.

    Example:
        validator = PreTradeValidator(config, portfolio, circuit_breaker)

        order = Order(symbol="SPY", quantity=100, side="buy", order_type="limit")
        result = validator.validate(order)

        if not result.approved:
            for check in result.failed_checks:
                print(f"Failed: {check.name} - {check.message}")
            return  # Don't submit order

        # Safe to submit order
        submit_order(order)
    """

    def __init__(
        self,
        config: ValidationConfig | None = None,
        portfolio: dict[str, Any] | None = None,
        circuit_breaker: Any | None = None,
    ):
        """
        Initialize the pre-trade validator.

        Args:
            config: Validation configuration
            portfolio: Portfolio state dictionary
            circuit_breaker: TradingCircuitBreaker instance
        """
        self.config = config or ValidationConfig()
        self.portfolio = portfolio or {}
        self.circuit_breaker = circuit_breaker
        self._recent_orders: list[Order] = []
        self._price_cache: dict[str, tuple[float, datetime]] = {}

    def validate(self, order: Order) -> ValidationResult:
        """
        Run all pre-trade validation checks.

        Args:
            order: Order to validate

        Returns:
            ValidationResult with all check results
        """
        checks = []

        # Run all validation checks
        checks.append(self._check_circuit_breaker(order))
        checks.append(self._check_position_limit(order))
        checks.append(self._check_daily_loss_limit(order))
        checks.append(self._check_concentration_limit(order))
        checks.append(self._check_order_value(order))
        checks.append(self._check_data_freshness(order))
        checks.append(self._check_duplicate_order(order))
        checks.append(self._check_liquidity(order))

        # For multi-leg orders, validate balance
        if order.is_combo and order.legs:
            checks.append(self._check_combo_balance(order))

        # Determine overall approval
        approved = all(c.passed or c.status == ValidationStatus.WARNING for c in checks)

        # Record order for duplicate detection
        if approved:
            self._recent_orders.append(order)
            self._cleanup_recent_orders()

        result = ValidationResult(approved=approved, checks=checks)

        # Log validation result
        if not approved:
            logger.warning(f"Order validation FAILED for {order.symbol}: " f"{[c.name for c in result.failed_checks]}")
        else:
            logger.debug(f"Order validation PASSED for {order.symbol}")

        return result

    def _check_circuit_breaker(self, order: Order) -> ValidationCheck:
        """Check if circuit breaker allows trading."""
        if not self.config.enforce_circuit_breaker:
            return ValidationCheck(
                name="circuit_breaker",
                status=ValidationStatus.SKIPPED,
                message="Circuit breaker check disabled",
            )

        if self.circuit_breaker is None:
            return ValidationCheck(
                name="circuit_breaker",
                status=ValidationStatus.SKIPPED,
                message="No circuit breaker configured",
            )

        if not self.circuit_breaker.can_trade():
            status = self.circuit_breaker.get_status()
            return ValidationCheck(
                name="circuit_breaker",
                status=ValidationStatus.FAILED,
                message=f"Trading halted: {status.get('trip_reason', 'unknown')}",
                details=status,
            )

        return ValidationCheck(
            name="circuit_breaker",
            status=ValidationStatus.PASSED,
            message="Circuit breaker allows trading",
        )

    def _check_position_limit(self, order: Order) -> ValidationCheck:
        """Check if order would exceed position limit."""
        portfolio_value = self.portfolio.get("total_value", 0)
        if portfolio_value <= 0:
            return ValidationCheck(
                name="position_limit",
                status=ValidationStatus.WARNING,
                message="Portfolio value not available",
            )

        # Get current position value for symbol
        positions = self.portfolio.get("positions", {})
        current_position = positions.get(order.symbol, {})
        current_value = current_position.get("market_value", 0)

        # Estimate order value
        price = self._get_price(order.symbol)
        if price is None:
            return ValidationCheck(
                name="position_limit",
                status=ValidationStatus.WARNING,
                message="Price not available for position check",
            )

        order_value = abs(order.quantity * price)
        if order.side == "sell":
            new_position_value = current_value - order_value
        else:
            new_position_value = current_value + order_value

        new_position_pct = abs(new_position_value) / portfolio_value

        if new_position_pct > self.config.max_position_pct:
            return ValidationCheck(
                name="position_limit",
                status=ValidationStatus.FAILED,
                message=f"Position would exceed limit: {new_position_pct:.1%} > {self.config.max_position_pct:.1%}",
                details={
                    "current_pct": current_value / portfolio_value,
                    "new_pct": new_position_pct,
                    "limit_pct": self.config.max_position_pct,
                },
            )

        return ValidationCheck(
            name="position_limit",
            status=ValidationStatus.PASSED,
            message=f"Position within limit: {new_position_pct:.1%}",
            details={"new_pct": new_position_pct, "limit_pct": self.config.max_position_pct},
        )

    def _check_daily_loss_limit(self, order: Order) -> ValidationCheck:
        """Check if daily loss limit has been reached."""
        daily_pnl_pct = self.portfolio.get("daily_pnl_pct", 0)

        if daily_pnl_pct <= -self.config.max_daily_loss_pct:
            return ValidationCheck(
                name="daily_loss_limit",
                status=ValidationStatus.FAILED,
                message=f"Daily loss limit reached: {daily_pnl_pct:.1%}",
                details={
                    "daily_pnl_pct": daily_pnl_pct,
                    "limit_pct": self.config.max_daily_loss_pct,
                },
            )

        return ValidationCheck(
            name="daily_loss_limit",
            status=ValidationStatus.PASSED,
            message=f"Within daily loss limit: {daily_pnl_pct:.1%}",
        )

    def _check_concentration_limit(self, order: Order) -> ValidationCheck:
        """Check if order would create excessive concentration."""
        portfolio_value = self.portfolio.get("total_value", 0)
        if portfolio_value <= 0:
            return ValidationCheck(
                name="concentration_limit",
                status=ValidationStatus.WARNING,
                message="Portfolio value not available",
            )

        # Calculate total exposure
        positions = self.portfolio.get("positions", {})
        total_exposure = sum(abs(p.get("market_value", 0)) for p in positions.values())

        # Add this order
        price = self._get_price(order.symbol)
        if price:
            order_value = abs(order.quantity * price)
            if order.side == "buy":
                total_exposure += order_value

        exposure_pct = total_exposure / portfolio_value

        if exposure_pct > self.config.max_total_exposure_pct:
            return ValidationCheck(
                name="concentration_limit",
                status=ValidationStatus.FAILED,
                message=f"Total exposure would exceed limit: {exposure_pct:.1%}",
                details={
                    "current_exposure_pct": exposure_pct,
                    "limit_pct": self.config.max_total_exposure_pct,
                },
            )

        return ValidationCheck(
            name="concentration_limit",
            status=ValidationStatus.PASSED,
            message=f"Exposure within limit: {exposure_pct:.1%}",
        )

    def _check_order_value(self, order: Order) -> ValidationCheck:
        """Check if order value is within acceptable range."""
        price = self._get_price(order.symbol)
        if price is None:
            return ValidationCheck(
                name="order_value",
                status=ValidationStatus.WARNING,
                message="Price not available for value check",
            )

        order_value = abs(order.quantity * price)

        if order_value > self.config.max_order_value:
            return ValidationCheck(
                name="order_value",
                status=ValidationStatus.FAILED,
                message=f"Order value ${order_value:,.2f} exceeds max ${self.config.max_order_value:,.2f}",
                details={
                    "order_value": order_value,
                    "max_value": self.config.max_order_value,
                },
            )

        if order_value < self.config.min_order_value:
            return ValidationCheck(
                name="order_value",
                status=ValidationStatus.WARNING,
                message=f"Order value ${order_value:,.2f} below min ${self.config.min_order_value:,.2f}",
                details={
                    "order_value": order_value,
                    "min_value": self.config.min_order_value,
                },
            )

        return ValidationCheck(
            name="order_value",
            status=ValidationStatus.PASSED,
            message=f"Order value ${order_value:,.2f} within range",
        )

    def _check_data_freshness(self, order: Order) -> ValidationCheck:
        """Check if price data is fresh enough."""
        cached = self._price_cache.get(order.symbol)
        if cached is None:
            return ValidationCheck(
                name="data_freshness",
                status=ValidationStatus.WARNING,
                message="No price data in cache",
            )

        price, timestamp = cached
        age = (_utc_now() - timestamp).total_seconds()

        if age > self.config.max_data_age_seconds:
            return ValidationCheck(
                name="data_freshness",
                status=ValidationStatus.FAILED,
                message=f"Price data is {age:.1f}s old (max {self.config.max_data_age_seconds}s)",
                details={"age_seconds": age, "max_age_seconds": self.config.max_data_age_seconds},
            )

        return ValidationCheck(
            name="data_freshness",
            status=ValidationStatus.PASSED,
            message=f"Price data is {age:.1f}s old",
        )

    def _check_duplicate_order(self, order: Order) -> ValidationCheck:
        """Check for duplicate orders within dedup window."""
        dedup_window = timedelta(seconds=self.config.dedup_window_seconds)
        cutoff = _utc_now() - dedup_window

        for recent in self._recent_orders:
            if recent.timestamp < cutoff:
                continue

            if recent.symbol == order.symbol and recent.quantity == order.quantity and recent.side == order.side:
                return ValidationCheck(
                    name="duplicate_order",
                    status=ValidationStatus.FAILED,
                    message="Potential duplicate order detected",
                    details={
                        "recent_timestamp": recent.timestamp.isoformat(),
                        "window_seconds": self.config.dedup_window_seconds,
                    },
                )

        return ValidationCheck(
            name="duplicate_order",
            status=ValidationStatus.PASSED,
            message="No duplicate detected",
        )

    def _check_liquidity(self, order: Order) -> ValidationCheck:
        """Check if symbol has sufficient liquidity."""
        market_data = self.portfolio.get("market_data", {})
        symbol_data = market_data.get(order.symbol, {})
        volume = symbol_data.get("volume", 0)

        if volume < self.config.min_liquidity_volume:
            return ValidationCheck(
                name="liquidity",
                status=ValidationStatus.WARNING,
                message=f"Low liquidity: volume {volume} < {self.config.min_liquidity_volume}",
                details={"volume": volume, "min_volume": self.config.min_liquidity_volume},
            )

        return ValidationCheck(
            name="liquidity",
            status=ValidationStatus.PASSED,
            message=f"Liquidity sufficient: volume {volume}",
        )

    def _check_combo_balance(self, order: Order) -> ValidationCheck:
        """Check if multi-leg order is balanced."""
        if not order.legs:
            return ValidationCheck(
                name="combo_balance",
                status=ValidationStatus.SKIPPED,
                message="Not a combo order",
            )

        # Calculate net delta (simplified)
        total_quantity = sum(leg.quantity if leg.side == "buy" else -leg.quantity for leg in order.legs)

        # For balanced strategies like butterflies, net should be small
        if abs(total_quantity) > 0:
            return ValidationCheck(
                name="combo_balance",
                status=ValidationStatus.WARNING,
                message=f"Combo has net quantity: {total_quantity}",
                details={"net_quantity": total_quantity, "leg_count": len(order.legs)},
            )

        return ValidationCheck(
            name="combo_balance",
            status=ValidationStatus.PASSED,
            message="Combo is balanced",
        )

    def update_price(self, symbol: str, price: float) -> None:
        """Update price cache for a symbol."""
        self._price_cache[symbol] = (price, _utc_now())

    def update_portfolio(self, portfolio: dict[str, Any]) -> None:
        """Update portfolio state."""
        self.portfolio = portfolio

    def _get_price(self, symbol: str) -> float | None:
        """Get cached price for symbol."""
        cached = self._price_cache.get(symbol)
        if cached:
            return cached[0]
        return None

    def _cleanup_recent_orders(self) -> None:
        """Remove old orders from duplicate detection cache."""
        cutoff = _utc_now() - timedelta(seconds=self.config.dedup_window_seconds * 10)
        self._recent_orders = [o for o in self._recent_orders if o.timestamp > cutoff]


# Convenience function
def create_validator(
    circuit_breaker: Any | None = None,
    max_position_pct: float = 0.25,
    max_daily_loss_pct: float = 0.03,
) -> PreTradeValidator:
    """
    Create a configured pre-trade validator.

    Args:
        circuit_breaker: Optional TradingCircuitBreaker instance
        max_position_pct: Maximum position size as percentage
        max_daily_loss_pct: Maximum daily loss as percentage

    Returns:
        Configured PreTradeValidator instance
    """
    config = ValidationConfig(
        max_position_pct=max_position_pct,
        max_daily_loss_pct=max_daily_loss_pct,
    )
    return PreTradeValidator(config=config, circuit_breaker=circuit_breaker)
