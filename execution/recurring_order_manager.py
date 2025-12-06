"""
Recurring Order Templates with Scheduling

Enables automated recurring trades based on schedules and market conditions.
Users can create templates that execute automatically when conditions are met.

Features:
- Schedule types: Daily, Weekly, Monthly, Conditional
- Entry conditions: IV Rank, Greeks thresholds, price levels
- Strike selection rules: Delta target, ATM offset
- Template persistence across algorithm restarts
- Enable/disable templates without deletion
- Comprehensive logging of scheduled order executions

Example:
    # Create recurring iron condor every Monday if IV Rank > 50
    template = RecurringOrderTemplate(
        name="Monday Iron Condor",
        schedule_type=ScheduleType.WEEKLY,
        schedule_params={"day_of_week": 0},  # Monday
        order_type=OrderType.OPTION_STRATEGY,
        strategy_name="iron_condor",
        conditions=[
            EntryCondition(
                condition_type=ConditionType.IV_RANK,
                operator=ConditionOperator.GREATER_THAN,
                threshold=50.0,
            )
        ],
    )
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class ScheduleType(Enum):
    """Types of recurring schedules."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CONDITIONAL = "conditional"  # Only when conditions met


class ConditionType(Enum):
    """Types of entry conditions."""

    IV_RANK = "iv_rank"
    IV_PERCENTILE = "iv_percentile"
    PORTFOLIO_DELTA = "portfolio_delta"
    PORTFOLIO_GAMMA = "portfolio_gamma"
    PORTFOLIO_THETA = "portfolio_theta"
    PORTFOLIO_VEGA = "portfolio_vega"
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    RSI = "rsi"
    CUSTOM = "custom"  # User-defined lambda


class ConditionOperator(Enum):
    """Comparison operators for conditions."""

    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="


class StrikeSelectionMode(Enum):
    """Strike selection methods."""

    DELTA_TARGET = "delta_target"  # Target specific delta
    ATM_OFFSET = "atm_offset"  # Offset from ATM
    FIXED_STRIKES = "fixed_strikes"  # Specific strike values
    CUSTOM = "custom"  # User-defined lambda


class OrderType(Enum):
    """Order types for recurring templates."""

    OPTION_STRATEGY = "option_strategy"
    MANUAL_LEGS = "manual_legs"
    EQUITY = "equity"


@dataclass
class EntryCondition:
    """Entry condition for recurring order."""

    condition_type: ConditionType
    operator: ConditionOperator
    threshold: float
    custom_lambda: Callable[[Any], bool] | None = None

    def evaluate(self, value: float) -> bool:
        """Evaluate condition against current value."""
        if self.condition_type == ConditionType.CUSTOM:
            if self.custom_lambda is None:
                return False
            return self.custom_lambda(value)

        ops = {
            ConditionOperator.GREATER_THAN: lambda a, b: a > b,
            ConditionOperator.LESS_THAN: lambda a, b: a < b,
            ConditionOperator.GREATER_EQUAL: lambda a, b: a >= b,
            ConditionOperator.LESS_EQUAL: lambda a, b: a <= b,
            ConditionOperator.EQUAL: lambda a, b: a == b,
            ConditionOperator.NOT_EQUAL: lambda a, b: a != b,
        }

        return ops[self.operator](value, self.threshold)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition_type": self.condition_type.value,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "has_custom_lambda": self.custom_lambda is not None,
        }


@dataclass
class StrikeSelection:
    """Strike selection configuration."""

    mode: StrikeSelectionMode
    params: dict[str, Any] = field(default_factory=dict)
    custom_lambda: Callable[[Any], list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode.value,
            "params": self.params,
            "has_custom_lambda": self.custom_lambda is not None,
        }


@dataclass
class RecurringOrderTemplate:
    """Template for recurring automated orders."""

    # Template identification
    template_id: str
    name: str
    description: str = ""

    # Schedule configuration
    schedule_type: ScheduleType = ScheduleType.CONDITIONAL
    schedule_params: dict[str, Any] = field(default_factory=dict)

    # Order configuration
    order_type: OrderType = OrderType.OPTION_STRATEGY
    symbol: str = ""
    quantity: int = 1
    strategy_name: str | None = None  # For OPTION_STRATEGY
    strategy_params: dict[str, Any] = field(default_factory=dict)
    legs: list[dict[str, Any]] = field(default_factory=list)  # For MANUAL_LEGS

    # Entry conditions
    conditions: list[EntryCondition] = field(default_factory=list)

    # Strike selection
    strike_selection: StrikeSelection | None = None

    # Template state
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: datetime | None = None
    trigger_count: int = 0

    # Execution tracking
    execution_history: list[dict[str, Any]] = field(default_factory=list)

    # Bot management
    bot_managed: bool = True  # Enable bot position management

    def should_trigger_now(self, current_time: datetime) -> bool:
        """Check if template should trigger based on schedule."""
        if not self.enabled:
            return False

        if self.schedule_type == ScheduleType.CONDITIONAL:
            # Only conditions matter, no time-based trigger
            return True

        if self.schedule_type == ScheduleType.DAILY:
            # Check if already triggered today
            if self.last_triggered:
                if self.last_triggered.date() == current_time.date():
                    return False

            # Check time of day
            trigger_time = self.schedule_params.get("time", "09:35")
            hour, minute = map(int, trigger_time.split(":"))

            return current_time.hour == hour and current_time.minute == minute

        if self.schedule_type == ScheduleType.WEEKLY:
            # Check day of week
            day_of_week = self.schedule_params.get("day_of_week", 0)  # 0=Monday

            if current_time.weekday() != day_of_week:
                return False

            # Check if already triggered this week
            if self.last_triggered:
                days_since = (current_time - self.last_triggered).days
                if days_since < 7:
                    return False

            # Check time
            trigger_time = self.schedule_params.get("time", "09:35")
            hour, minute = map(int, trigger_time.split(":"))

            return current_time.hour == hour and current_time.minute == minute

        if self.schedule_type == ScheduleType.MONTHLY:
            # Check day of month
            day_of_month = self.schedule_params.get("day_of_month", 1)

            if current_time.day != day_of_month:
                return False

            # Check if already triggered this month
            if self.last_triggered:
                if self.last_triggered.year == current_time.year and self.last_triggered.month == current_time.month:
                    return False

            # Check time
            trigger_time = self.schedule_params.get("time", "09:35")
            hour, minute = map(int, trigger_time.split(":"))

            return current_time.hour == hour and current_time.minute == minute

        return False

    def check_conditions(self, market_data: dict[str, float]) -> bool:
        """Check if all entry conditions are met."""
        if not self.conditions:
            return True  # No conditions = always true

        for condition in self.conditions:
            value = market_data.get(condition.condition_type.value)

            if value is None:
                return False  # Missing data

            if not condition.evaluate(value):
                return False  # Condition not met

        return True  # All conditions met

    def record_execution(
        self,
        success: bool,
        order_id: str | None = None,
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record execution attempt."""
        self.execution_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "order_id": order_id,
                "error": error,
                "details": details or {},
            }
        )

        if success:
            self.last_triggered = datetime.now()
            self.trigger_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert template to dictionary for persistence."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "schedule_type": self.schedule_type.value,
            "schedule_params": self.schedule_params,
            "order_type": self.order_type.value,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "strategy_name": self.strategy_name,
            "strategy_params": self.strategy_params,
            "legs": self.legs,
            "conditions": [c.to_dict() for c in self.conditions],
            "strike_selection": (self.strike_selection.to_dict() if self.strike_selection else None),
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "last_triggered": (self.last_triggered.isoformat() if self.last_triggered else None),
            "trigger_count": self.trigger_count,
            "execution_history": self.execution_history,
            "bot_managed": self.bot_managed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecurringOrderTemplate":
        """Create template from dictionary."""
        # Reconstruct conditions
        conditions = []
        for cond_data in data.get("conditions", []):
            conditions.append(
                EntryCondition(
                    condition_type=ConditionType(cond_data["condition_type"]),
                    operator=ConditionOperator(cond_data["operator"]),
                    threshold=cond_data["threshold"],
                )
            )

        # Reconstruct strike selection
        strike_selection = None
        if data.get("strike_selection"):
            ss_data = data["strike_selection"]
            strike_selection = StrikeSelection(
                mode=StrikeSelectionMode(ss_data["mode"]),
                params=ss_data.get("params", {}),
            )

        return cls(
            template_id=data["template_id"],
            name=data["name"],
            description=data.get("description", ""),
            schedule_type=ScheduleType(data["schedule_type"]),
            schedule_params=data.get("schedule_params", {}),
            order_type=OrderType(data["order_type"]),
            symbol=data.get("symbol", ""),
            quantity=data.get("quantity", 1),
            strategy_name=data.get("strategy_name"),
            strategy_params=data.get("strategy_params", {}),
            legs=data.get("legs", []),
            conditions=conditions,
            strike_selection=strike_selection,
            enabled=data.get("enabled", True),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_triggered=(datetime.fromisoformat(data["last_triggered"]) if data.get("last_triggered") else None),
            trigger_count=data.get("trigger_count", 0),
            execution_history=data.get("execution_history", []),
            bot_managed=data.get("bot_managed", True),
        )


class RecurringOrderManager:
    """Manages recurring order templates and execution."""

    def __init__(
        self,
        algorithm: Any,
        storage_path: Path | None = None,
        enable_logging: bool = True,
    ):
        """
        Initialize recurring order manager.

        Args:
            algorithm: QuantConnect algorithm instance
            storage_path: Path to store templates (default: ./templates/)
            enable_logging: Enable debug logging
        """
        self.algorithm = algorithm
        self.enable_logging = enable_logging

        # Storage
        if storage_path is None:
            storage_path = Path("./templates/")
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Templates
        self.templates: dict[str, RecurringOrderTemplate] = {}

        # Callbacks
        self.on_order_triggered: Callable[[RecurringOrderTemplate], None] | None = None
        self.on_order_executed: Callable[[RecurringOrderTemplate, bool, str | None], None] | None = None

        # Statistics
        self.stats = {
            "total_templates": 0,
            "active_templates": 0,
            "total_triggers": 0,
            "successful_executions": 0,
            "failed_executions": 0,
        }

        # Load existing templates
        self.load_all_templates()

        self._log("RecurringOrderManager initialized")

    def _log(self, message: str) -> None:
        """Log debug message."""
        if self.enable_logging and hasattr(self.algorithm, "Debug"):
            self.algorithm.Debug(f"[RecurringOrderManager] {message}")

    def add_template(self, template: RecurringOrderTemplate) -> bool:
        """Add a new template."""
        if template.template_id in self.templates:
            self._log(f"Template {template.template_id} already exists")
            return False

        self.templates[template.template_id] = template
        self.stats["total_templates"] += 1
        if template.enabled:
            self.stats["active_templates"] += 1

        # Persist to storage
        self.save_template(template)

        self._log(f"Added template: {template.name}")
        return True

    def remove_template(self, template_id: str) -> bool:
        """Remove a template."""
        if template_id not in self.templates:
            return False

        template = self.templates[template_id]
        if template.enabled:
            self.stats["active_templates"] -= 1

        del self.templates[template_id]
        self.stats["total_templates"] -= 1

        # Delete from storage
        template_file = self.storage_path / f"{template_id}.json"
        if template_file.exists():
            template_file.unlink()

        self._log(f"Removed template: {template.name}")
        return True

    def enable_template(self, template_id: str) -> bool:
        """Enable a template."""
        if template_id not in self.templates:
            return False

        template = self.templates[template_id]
        if not template.enabled:
            template.enabled = True
            self.stats["active_templates"] += 1
            self.save_template(template)
            self._log(f"Enabled template: {template.name}")

        return True

    def disable_template(self, template_id: str) -> bool:
        """Disable a template."""
        if template_id not in self.templates:
            return False

        template = self.templates[template_id]
        if template.enabled:
            template.enabled = False
            self.stats["active_templates"] -= 1
            self.save_template(template)
            self._log(f"Disabled template: {template.name}")

        return True

    def get_template(self, template_id: str) -> RecurringOrderTemplate | None:
        """Get template by ID."""
        return self.templates.get(template_id)

    def get_all_templates(
        self,
        enabled_only: bool = False,
    ) -> list[RecurringOrderTemplate]:
        """Get all templates."""
        templates = list(self.templates.values())

        if enabled_only:
            templates = [t for t in templates if t.enabled]

        return templates

    def get_templates_by_symbol(
        self,
        symbol: str,
        enabled_only: bool = False,
    ) -> list[RecurringOrderTemplate]:
        """Get templates for specific symbol."""
        templates = [t for t in self.templates.values() if t.symbol == symbol]

        if enabled_only:
            templates = [t for t in templates if t.enabled]

        return templates

    def check_triggers(
        self,
        current_time: datetime,
        market_data: dict[str, float],
    ) -> list[RecurringOrderTemplate]:
        """
        Check which templates should trigger.

        Args:
            current_time: Current datetime
            market_data: Market data for condition checking
                Keys should match ConditionType values

        Returns:
            List of templates that should execute
        """
        triggered = []

        for template in self.templates.values():
            if not template.enabled:
                continue

            # Check schedule
            if not template.should_trigger_now(current_time):
                continue

            # Check conditions
            if not template.check_conditions(market_data):
                continue

            triggered.append(template)
            self.stats["total_triggers"] += 1

            # Callback
            if self.on_order_triggered:
                self.on_order_triggered(template)

        return triggered

    def execute_template(
        self,
        template: RecurringOrderTemplate,
        order_executor: Callable[[RecurringOrderTemplate], str | None],
    ) -> bool:
        """
        Execute a template using provided order executor.

        Args:
            template: Template to execute
            order_executor: Function that executes the order
                Should return order_id on success, None on failure

        Returns:
            True if execution successful
        """
        try:
            # Execute order
            order_id = order_executor(template)

            if order_id:
                # Success
                template.record_execution(
                    success=True,
                    order_id=order_id,
                )
                self.stats["successful_executions"] += 1
                self.save_template(template)

                self._log(f"Executed template {template.name}: order {order_id}")

                # Callback
                if self.on_order_executed:
                    self.on_order_executed(template, True, order_id)

                return True
            else:
                # Failure
                template.record_execution(
                    success=False,
                    error="Order executor returned None",
                )
                self.stats["failed_executions"] += 1
                self.save_template(template)

                self._log(f"Failed to execute template {template.name}")

                # Callback
                if self.on_order_executed:
                    self.on_order_executed(template, False, None)

                return False

        except Exception as e:
            # Error
            error_msg = str(e)
            template.record_execution(
                success=False,
                error=error_msg,
            )
            self.stats["failed_executions"] += 1
            self.save_template(template)

            self._log(f"Error executing template {template.name}: {error_msg}")

            # Callback
            if self.on_order_executed:
                self.on_order_executed(template, False, None)

            return False

    def save_template(self, template: RecurringOrderTemplate) -> None:
        """Save template to storage."""
        template_file = self.storage_path / f"{template.template_id}.json"

        with open(template_file, "w") as f:
            json.dump(template.to_dict(), f, indent=2)

    def load_template(self, template_id: str) -> RecurringOrderTemplate | None:
        """Load template from storage."""
        template_file = self.storage_path / f"{template_id}.json"

        if not template_file.exists():
            return None

        try:
            with open(template_file) as f:
                data = json.load(f)

            return RecurringOrderTemplate.from_dict(data)

        except Exception as e:
            self._log(f"Error loading template {template_id}: {e}")
            return None

    def load_all_templates(self) -> int:
        """Load all templates from storage."""
        count = 0

        for template_file in self.storage_path.glob("*.json"):
            template_id = template_file.stem

            template = self.load_template(template_id)
            if template:
                self.templates[template_id] = template
                self.stats["total_templates"] += 1
                if template.enabled:
                    self.stats["active_templates"] += 1
                count += 1

        self._log(f"Loaded {count} templates from storage")
        return count

    def get_statistics(self) -> dict[str, Any]:
        """Get manager statistics."""
        return self.stats.copy()

    def get_upcoming_triggers(
        self,
        current_time: datetime,
        days_ahead: int = 7,
    ) -> list[dict[str, Any]]:
        """Get upcoming scheduled triggers."""
        upcoming = []

        for template in self.templates.values():
            if not template.enabled:
                continue

            if template.schedule_type == ScheduleType.CONDITIONAL:
                # Can't predict conditional triggers
                continue

            # Calculate next trigger time
            next_trigger = self._calculate_next_trigger(template, current_time)

            if next_trigger and (next_trigger - current_time).days <= days_ahead:
                upcoming.append(
                    {
                        "template_id": template.template_id,
                        "name": template.name,
                        "next_trigger": next_trigger.isoformat(),
                        "days_until": (next_trigger - current_time).days,
                    }
                )

        # Sort by next trigger time
        upcoming.sort(key=lambda x: x["next_trigger"])

        return upcoming

    def _calculate_next_trigger(
        self,
        template: RecurringOrderTemplate,
        current_time: datetime,
    ) -> datetime | None:
        """Calculate next trigger time for template."""
        if template.schedule_type == ScheduleType.DAILY:
            trigger_time = template.schedule_params.get("time", "09:35")
            hour, minute = map(int, trigger_time.split(":"))

            next_trigger = current_time.replace(
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0,
            )

            if next_trigger <= current_time:
                next_trigger += timedelta(days=1)

            return next_trigger

        if template.schedule_type == ScheduleType.WEEKLY:
            day_of_week = template.schedule_params.get("day_of_week", 0)
            trigger_time = template.schedule_params.get("time", "09:35")
            hour, minute = map(int, trigger_time.split(":"))

            days_ahead = (day_of_week - current_time.weekday()) % 7
            if days_ahead == 0:
                # Today - check if time has passed
                next_trigger = current_time.replace(
                    hour=hour,
                    minute=minute,
                    second=0,
                    microsecond=0,
                )
                if next_trigger <= current_time:
                    days_ahead = 7

            next_trigger = current_time + timedelta(days=days_ahead)
            next_trigger = next_trigger.replace(
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0,
            )

            return next_trigger

        if template.schedule_type == ScheduleType.MONTHLY:
            day_of_month = template.schedule_params.get("day_of_month", 1)
            trigger_time = template.schedule_params.get("time", "09:35")
            hour, minute = map(int, trigger_time.split(":"))

            next_trigger = current_time.replace(
                day=day_of_month,
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0,
            )

            if next_trigger <= current_time:
                # Next month
                if current_time.month == 12:
                    next_trigger = next_trigger.replace(
                        year=current_time.year + 1,
                        month=1,
                    )
                else:
                    next_trigger = next_trigger.replace(
                        month=current_time.month + 1,
                    )

            return next_trigger

        return None


def create_recurring_order_manager(
    algorithm: Any,
    storage_path: Path | None = None,
    enable_logging: bool = True,
) -> RecurringOrderManager:
    """
    Create a RecurringOrderManager instance.

    Args:
        algorithm: QuantConnect algorithm instance
        storage_path: Path to store templates (default: ./templates/)
        enable_logging: Enable debug logging

    Returns:
        RecurringOrderManager instance

    Example:
        >>> manager = create_recurring_order_manager(self)
        >>>
        >>> # Create template
        >>> template = RecurringOrderTemplate(
        ...     template_id="ic_monday",
        ...     name="Monday Iron Condor",
        ...     schedule_type=ScheduleType.WEEKLY,
        ...     schedule_params={"day_of_week": 0, "time": "09:35"},
        ...     order_type=OrderType.OPTION_STRATEGY,
        ...     symbol="SPY",
        ...     quantity=1,
        ...     strategy_name="iron_condor",
        ...     conditions=[
        ...         EntryCondition(
        ...             condition_type=ConditionType.IV_RANK,
        ...             operator=ConditionOperator.GREATER_THAN,
        ...             threshold=50.0,
        ...         )
        ...     ],
        ... )
        >>>
        >>> manager.add_template(template)
        >>>
        >>> # In OnData
        >>> market_data = {"iv_rank": current_iv_rank}
        >>> triggered = manager.check_triggers(self.Time, market_data)
        >>>
        >>> for template in triggered:
        ...     manager.execute_template(template, my_order_executor)
    """
    return RecurringOrderManager(
        algorithm=algorithm,
        storage_path=storage_path,
        enable_logging=enable_logging,
    )
