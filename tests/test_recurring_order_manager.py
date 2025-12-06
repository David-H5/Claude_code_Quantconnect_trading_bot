"""
Tests for Recurring Order Manager

Tests for automated recurring trades based on schedules and market conditions.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from execution.recurring_order_manager import (
    ConditionOperator,
    ConditionType,
    EntryCondition,
    OrderType,
    RecurringOrderManager,
    RecurringOrderTemplate,
    ScheduleType,
    StrikeSelection,
    StrikeSelectionMode,
    create_recurring_order_manager,
)


class TestRecurringOrderManagerInitialization:
    """Tests for RecurringOrderManager initialization."""

    @pytest.mark.unit
    def test_manager_creation(self, tmp_path):
        """Test manager instance creation."""
        algorithm = Mock()
        manager = create_recurring_order_manager(
            algorithm,
            storage_path=tmp_path,
        )

        assert manager.algorithm == algorithm
        assert manager.storage_path == tmp_path
        assert len(manager.templates) == 0
        assert manager.stats["total_templates"] == 0

    @pytest.mark.unit
    def test_manager_with_logging_disabled(self, tmp_path):
        """Test manager with logging disabled."""
        algorithm = Mock()
        manager = RecurringOrderManager(
            algorithm=algorithm,
            storage_path=tmp_path,
            enable_logging=False,
        )

        assert manager.enable_logging is False


class TestEntryConditions:
    """Tests for entry condition evaluation."""

    @pytest.mark.unit
    def test_iv_rank_condition_met(self):
        """Test IV rank condition evaluation when met."""
        condition = EntryCondition(
            condition_type=ConditionType.IV_RANK,
            operator=ConditionOperator.GREATER_THAN,
            threshold=50.0,
        )

        assert condition.evaluate(60.0) is True

    @pytest.mark.unit
    def test_iv_rank_condition_not_met(self):
        """Test IV rank condition evaluation when not met."""
        condition = EntryCondition(
            condition_type=ConditionType.IV_RANK,
            operator=ConditionOperator.GREATER_THAN,
            threshold=50.0,
        )

        assert condition.evaluate(40.0) is False

    @pytest.mark.unit
    def test_portfolio_delta_condition(self):
        """Test portfolio delta condition."""
        condition = EntryCondition(
            condition_type=ConditionType.PORTFOLIO_DELTA,
            operator=ConditionOperator.LESS_THAN,
            threshold=-100.0,
        )

        assert condition.evaluate(-150.0) is True
        assert condition.evaluate(-50.0) is False

    @pytest.mark.unit
    def test_custom_condition(self):
        """Test custom lambda condition."""
        condition = EntryCondition(
            condition_type=ConditionType.CUSTOM,
            operator=ConditionOperator.GREATER_THAN,  # Not used
            threshold=0.0,  # Not used
            custom_lambda=lambda x: x > 100,
        )

        assert condition.evaluate(150.0) is True
        assert condition.evaluate(50.0) is False


class TestTemplateCreation:
    """Tests for template creation and configuration."""

    @pytest.mark.unit
    def test_create_daily_template(self):
        """Test creating a daily recurring template."""
        template = RecurringOrderTemplate(
            template_id="daily_ic",
            name="Daily Iron Condor",
            schedule_type=ScheduleType.DAILY,
            schedule_params={"time": "09:35"},
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            strategy_name="iron_condor",
        )

        assert template.template_id == "daily_ic"
        assert template.schedule_type == ScheduleType.DAILY
        assert template.enabled is True

    @pytest.mark.unit
    def test_create_weekly_template(self):
        """Test creating a weekly recurring template."""
        template = RecurringOrderTemplate(
            template_id="weekly_bf",
            name="Monday Butterfly",
            schedule_type=ScheduleType.WEEKLY,
            schedule_params={"day_of_week": 0, "time": "10:00"},
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            strategy_name="butterfly_call",
        )

        assert template.schedule_params["day_of_week"] == 0  # Monday

    @pytest.mark.unit
    def test_create_conditional_template(self):
        """Test creating a conditional template."""
        template = RecurringOrderTemplate(
            template_id="cond_ic",
            name="Conditional Iron Condor",
            schedule_type=ScheduleType.CONDITIONAL,
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            strategy_name="iron_condor",
            conditions=[
                EntryCondition(
                    condition_type=ConditionType.IV_RANK,
                    operator=ConditionOperator.GREATER_THAN,
                    threshold=50.0,
                )
            ],
        )

        assert len(template.conditions) == 1
        assert template.schedule_type == ScheduleType.CONDITIONAL

    @pytest.mark.unit
    def test_template_with_strike_selection(self):
        """Test template with strike selection rules."""
        template = RecurringOrderTemplate(
            template_id="delta_target",
            name="Delta Target Strategy",
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            strategy_name="vertical_call_spread",
            strike_selection=StrikeSelection(
                mode=StrikeSelectionMode.DELTA_TARGET,
                params={"target_delta": 0.30},
            ),
        )

        assert template.strike_selection is not None
        assert template.strike_selection.mode == StrikeSelectionMode.DELTA_TARGET


class TestScheduleTriggers:
    """Tests for schedule-based triggering."""

    @pytest.mark.unit
    def test_daily_trigger_at_time(self):
        """Test daily template triggers at specified time."""
        template = RecurringOrderTemplate(
            template_id="daily_test",
            name="Daily Test",
            schedule_type=ScheduleType.DAILY,
            schedule_params={"time": "09:35"},
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
        )

        # Should trigger at 9:35
        trigger_time = datetime(2025, 1, 15, 9, 35)
        assert template.should_trigger_now(trigger_time) is True

        # Should not trigger at different time
        other_time = datetime(2025, 1, 15, 10, 0)
        assert template.should_trigger_now(other_time) is False

    @pytest.mark.unit
    def test_daily_template_already_triggered_today(self):
        """Test daily template doesn't trigger twice in same day."""
        template = RecurringOrderTemplate(
            template_id="daily_test",
            name="Daily Test",
            schedule_type=ScheduleType.DAILY,
            schedule_params={"time": "09:35"},
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
        )

        # First trigger at 9:35
        trigger_time = datetime(2025, 1, 15, 9, 35)
        assert template.should_trigger_now(trigger_time) is True

        # Record the trigger
        template.last_triggered = trigger_time

        # Should not trigger again same day
        same_day = datetime(2025, 1, 15, 9, 35)
        assert template.should_trigger_now(same_day) is False

        # Should trigger next day
        next_day = datetime(2025, 1, 16, 9, 35)
        assert template.should_trigger_now(next_day) is True

    @pytest.mark.unit
    def test_weekly_trigger_on_monday(self):
        """Test weekly template triggers on correct day."""
        template = RecurringOrderTemplate(
            template_id="weekly_test",
            name="Weekly Test",
            schedule_type=ScheduleType.WEEKLY,
            schedule_params={"day_of_week": 0, "time": "09:35"},  # Monday
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
        )

        # Monday, January 13, 2025 at 9:35
        monday = datetime(2025, 1, 13, 9, 35)
        assert monday.weekday() == 0  # Verify it's Monday
        assert template.should_trigger_now(monday) is True

        # Tuesday should not trigger
        tuesday = datetime(2025, 1, 14, 9, 35)
        assert template.should_trigger_now(tuesday) is False

    @pytest.mark.unit
    def test_monthly_trigger_on_day(self):
        """Test monthly template triggers on correct day."""
        template = RecurringOrderTemplate(
            template_id="monthly_test",
            name="Monthly Test",
            schedule_type=ScheduleType.MONTHLY,
            schedule_params={"day_of_month": 15, "time": "09:35"},
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
        )

        # 15th at 9:35
        correct_day = datetime(2025, 1, 15, 9, 35)
        assert template.should_trigger_now(correct_day) is True

        # Different day
        wrong_day = datetime(2025, 1, 16, 9, 35)
        assert template.should_trigger_now(wrong_day) is False

    @pytest.mark.unit
    def test_conditional_template_always_checks(self):
        """Test conditional template always returns True for schedule check."""
        template = RecurringOrderTemplate(
            template_id="cond_test",
            name="Conditional Test",
            schedule_type=ScheduleType.CONDITIONAL,
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
        )

        # Conditional templates always pass schedule check
        # (conditions are checked separately)
        any_time = datetime(2025, 1, 15, 10, 30)
        assert template.should_trigger_now(any_time) is True


class TestConditionChecking:
    """Tests for condition checking logic."""

    @pytest.mark.unit
    def test_single_condition_met(self):
        """Test single condition evaluation."""
        template = RecurringOrderTemplate(
            template_id="test",
            name="Test",
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            conditions=[
                EntryCondition(
                    condition_type=ConditionType.IV_RANK,
                    operator=ConditionOperator.GREATER_THAN,
                    threshold=50.0,
                )
            ],
        )

        market_data = {"iv_rank": 60.0}
        assert template.check_conditions(market_data) is True

    @pytest.mark.unit
    def test_single_condition_not_met(self):
        """Test single condition not met."""
        template = RecurringOrderTemplate(
            template_id="test",
            name="Test",
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            conditions=[
                EntryCondition(
                    condition_type=ConditionType.IV_RANK,
                    operator=ConditionOperator.GREATER_THAN,
                    threshold=50.0,
                )
            ],
        )

        market_data = {"iv_rank": 40.0}
        assert template.check_conditions(market_data) is False

    @pytest.mark.unit
    def test_multiple_conditions_all_met(self):
        """Test multiple conditions all met."""
        template = RecurringOrderTemplate(
            template_id="test",
            name="Test",
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            conditions=[
                EntryCondition(
                    condition_type=ConditionType.IV_RANK,
                    operator=ConditionOperator.GREATER_THAN,
                    threshold=50.0,
                ),
                EntryCondition(
                    condition_type=ConditionType.PORTFOLIO_DELTA,
                    operator=ConditionOperator.LESS_THAN,
                    threshold=100.0,
                ),
            ],
        )

        market_data = {
            "iv_rank": 60.0,
            "portfolio_delta": 50.0,
        }
        assert template.check_conditions(market_data) is True

    @pytest.mark.unit
    def test_multiple_conditions_one_not_met(self):
        """Test multiple conditions with one not met."""
        template = RecurringOrderTemplate(
            template_id="test",
            name="Test",
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            conditions=[
                EntryCondition(
                    condition_type=ConditionType.IV_RANK,
                    operator=ConditionOperator.GREATER_THAN,
                    threshold=50.0,
                ),
                EntryCondition(
                    condition_type=ConditionType.PORTFOLIO_DELTA,
                    operator=ConditionOperator.LESS_THAN,
                    threshold=100.0,
                ),
            ],
        )

        market_data = {
            "iv_rank": 60.0,
            "portfolio_delta": 150.0,  # Fails condition
        }
        assert template.check_conditions(market_data) is False

    @pytest.mark.unit
    def test_no_conditions_always_true(self):
        """Test template with no conditions always passes."""
        template = RecurringOrderTemplate(
            template_id="test",
            name="Test",
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            conditions=[],
        )

        market_data = {}
        assert template.check_conditions(market_data) is True


class TestTemplateManagement:
    """Tests for template management operations."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager instance for testing."""
        algorithm = Mock()
        algorithm.Debug = Mock()
        return create_recurring_order_manager(
            algorithm,
            storage_path=tmp_path,
        )

    @pytest.mark.unit
    def test_add_template(self, manager):
        """Test adding a template."""
        template = RecurringOrderTemplate(
            template_id="test_1",
            name="Test Template",
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
        )

        success = manager.add_template(template)

        assert success is True
        assert manager.stats["total_templates"] == 1
        assert manager.stats["active_templates"] == 1

    @pytest.mark.unit
    def test_add_duplicate_template(self, manager):
        """Test adding duplicate template fails."""
        template = RecurringOrderTemplate(
            template_id="test_1",
            name="Test Template",
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
        )

        manager.add_template(template)
        success = manager.add_template(template)

        assert success is False
        assert manager.stats["total_templates"] == 1

    @pytest.mark.unit
    def test_remove_template(self, manager):
        """Test removing a template."""
        template = RecurringOrderTemplate(
            template_id="test_1",
            name="Test Template",
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
        )

        manager.add_template(template)
        success = manager.remove_template("test_1")

        assert success is True
        assert manager.stats["total_templates"] == 0
        assert manager.stats["active_templates"] == 0

    @pytest.mark.unit
    def test_enable_template(self, manager):
        """Test enabling a template."""
        template = RecurringOrderTemplate(
            template_id="test_1",
            name="Test Template",
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            enabled=False,
        )

        manager.add_template(template)
        assert manager.stats["active_templates"] == 0

        success = manager.enable_template("test_1")

        assert success is True
        assert manager.templates["test_1"].enabled is True
        assert manager.stats["active_templates"] == 1

    @pytest.mark.unit
    def test_disable_template(self, manager):
        """Test disabling a template."""
        template = RecurringOrderTemplate(
            template_id="test_1",
            name="Test Template",
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            enabled=True,
        )

        manager.add_template(template)
        assert manager.stats["active_templates"] == 1

        success = manager.disable_template("test_1")

        assert success is True
        assert manager.templates["test_1"].enabled is False
        assert manager.stats["active_templates"] == 0

    @pytest.mark.unit
    def test_get_templates_by_symbol(self, manager):
        """Test getting templates by symbol."""
        manager.add_template(
            RecurringOrderTemplate(
                template_id="spy_1",
                name="SPY Template",
                order_type=OrderType.OPTION_STRATEGY,
                symbol="SPY",
                quantity=1,
            )
        )

        manager.add_template(
            RecurringOrderTemplate(
                template_id="aapl_1",
                name="AAPL Template",
                order_type=OrderType.OPTION_STRATEGY,
                symbol="AAPL",
                quantity=1,
            )
        )

        spy_templates = manager.get_templates_by_symbol("SPY")
        assert len(spy_templates) == 1
        assert spy_templates[0].symbol == "SPY"


class TestTriggerChecking:
    """Tests for trigger checking logic."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager instance for testing."""
        algorithm = Mock()
        algorithm.Debug = Mock()
        return create_recurring_order_manager(
            algorithm,
            storage_path=tmp_path,
        )

    @pytest.mark.unit
    def test_check_triggers_with_conditions_met(self, manager):
        """Test trigger checking with conditions met."""
        template = RecurringOrderTemplate(
            template_id="test_1",
            name="Test Template",
            schedule_type=ScheduleType.CONDITIONAL,
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            conditions=[
                EntryCondition(
                    condition_type=ConditionType.IV_RANK,
                    operator=ConditionOperator.GREATER_THAN,
                    threshold=50.0,
                )
            ],
        )

        manager.add_template(template)

        current_time = datetime.now()
        market_data = {"iv_rank": 60.0}

        triggered = manager.check_triggers(current_time, market_data)

        assert len(triggered) == 1
        assert triggered[0].template_id == "test_1"
        assert manager.stats["total_triggers"] == 1

    @pytest.mark.unit
    def test_check_triggers_with_conditions_not_met(self, manager):
        """Test trigger checking with conditions not met."""
        template = RecurringOrderTemplate(
            template_id="test_1",
            name="Test Template",
            schedule_type=ScheduleType.CONDITIONAL,
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            conditions=[
                EntryCondition(
                    condition_type=ConditionType.IV_RANK,
                    operator=ConditionOperator.GREATER_THAN,
                    threshold=50.0,
                )
            ],
        )

        manager.add_template(template)

        current_time = datetime.now()
        market_data = {"iv_rank": 40.0}

        triggered = manager.check_triggers(current_time, market_data)

        assert len(triggered) == 0

    @pytest.mark.unit
    def test_disabled_template_does_not_trigger(self, manager):
        """Test disabled templates don't trigger."""
        template = RecurringOrderTemplate(
            template_id="test_1",
            name="Test Template",
            schedule_type=ScheduleType.CONDITIONAL,
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            enabled=False,
        )

        manager.add_template(template)

        current_time = datetime.now()
        market_data = {}

        triggered = manager.check_triggers(current_time, market_data)

        assert len(triggered) == 0


class TestTemplateExecution:
    """Tests for template execution."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager instance for testing."""
        algorithm = Mock()
        algorithm.Debug = Mock()
        return create_recurring_order_manager(
            algorithm,
            storage_path=tmp_path,
        )

    @pytest.mark.unit
    def test_execute_template_success(self, manager):
        """Test successful template execution."""
        template = RecurringOrderTemplate(
            template_id="test_1",
            name="Test Template",
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
        )

        manager.add_template(template)

        # Mock order executor
        def order_executor(tmpl):
            return "order_123"

        success = manager.execute_template(template, order_executor)

        assert success is True
        assert template.trigger_count == 1
        assert template.last_triggered is not None
        assert manager.stats["successful_executions"] == 1
        assert len(template.execution_history) == 1

    @pytest.mark.unit
    def test_execute_template_failure(self, manager):
        """Test failed template execution."""
        template = RecurringOrderTemplate(
            template_id="test_1",
            name="Test Template",
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
        )

        manager.add_template(template)

        # Mock order executor that fails
        def order_executor(tmpl):
            return None

        success = manager.execute_template(template, order_executor)

        assert success is False
        assert template.trigger_count == 0
        assert manager.stats["failed_executions"] == 1

    @pytest.mark.unit
    def test_execute_template_exception(self, manager):
        """Test template execution with exception."""
        template = RecurringOrderTemplate(
            template_id="test_1",
            name="Test Template",
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
        )

        manager.add_template(template)

        # Mock order executor that raises exception
        def order_executor(tmpl):
            raise ValueError("Test error")

        success = manager.execute_template(template, order_executor)

        assert success is False
        assert manager.stats["failed_executions"] == 1
        assert "Test error" in template.execution_history[0]["error"]


class TestTemplatePersistence:
    """Tests for template persistence."""

    @pytest.mark.unit
    def test_template_serialization(self):
        """Test converting template to dictionary."""
        template = RecurringOrderTemplate(
            template_id="test_1",
            name="Test Template",
            schedule_type=ScheduleType.WEEKLY,
            schedule_params={"day_of_week": 0, "time": "09:35"},
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
            strategy_name="iron_condor",
            conditions=[
                EntryCondition(
                    condition_type=ConditionType.IV_RANK,
                    operator=ConditionOperator.GREATER_THAN,
                    threshold=50.0,
                )
            ],
        )

        template_dict = template.to_dict()

        assert template_dict["template_id"] == "test_1"
        assert template_dict["schedule_type"] == "weekly"
        assert len(template_dict["conditions"]) == 1

    @pytest.mark.unit
    def test_template_deserialization(self):
        """Test creating template from dictionary."""
        template_dict = {
            "template_id": "test_1",
            "name": "Test Template",
            "description": "",
            "schedule_type": "weekly",
            "schedule_params": {"day_of_week": 0, "time": "09:35"},
            "order_type": "option_strategy",
            "symbol": "SPY",
            "quantity": 1,
            "strategy_name": "iron_condor",
            "strategy_params": {},
            "legs": [],
            "conditions": [
                {
                    "condition_type": "iv_rank",
                    "operator": ">",
                    "threshold": 50.0,
                }
            ],
            "strike_selection": None,
            "enabled": True,
            "created_at": datetime.now().isoformat(),
            "last_triggered": None,
            "trigger_count": 0,
            "execution_history": [],
            "bot_managed": True,
        }

        template = RecurringOrderTemplate.from_dict(template_dict)

        assert template.template_id == "test_1"
        assert template.schedule_type == ScheduleType.WEEKLY
        assert len(template.conditions) == 1

    @pytest.mark.unit
    def test_save_and_load_template(self, tmp_path):
        """Test saving and loading template."""
        algorithm = Mock()
        manager = create_recurring_order_manager(
            algorithm,
            storage_path=tmp_path,
        )

        template = RecurringOrderTemplate(
            template_id="test_1",
            name="Test Template",
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
        )

        manager.add_template(template)

        # Load in new manager
        manager2 = create_recurring_order_manager(
            algorithm,
            storage_path=tmp_path,
        )

        assert len(manager2.templates) == 1
        assert "test_1" in manager2.templates


class TestUpcomingTriggers:
    """Tests for upcoming trigger calculation."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager instance for testing."""
        algorithm = Mock()
        return create_recurring_order_manager(
            algorithm,
            storage_path=tmp_path,
        )

    @pytest.mark.unit
    def test_get_upcoming_daily_triggers(self, manager):
        """Test getting upcoming daily triggers."""
        template = RecurringOrderTemplate(
            template_id="daily_test",
            name="Daily Test",
            schedule_type=ScheduleType.DAILY,
            schedule_params={"time": "09:35"},
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
        )

        manager.add_template(template)

        current_time = datetime(2025, 1, 15, 8, 0)  # Before trigger time
        upcoming = manager.get_upcoming_triggers(current_time, days_ahead=7)

        assert len(upcoming) == 1
        assert upcoming[0]["template_id"] == "daily_test"

    @pytest.mark.unit
    def test_conditional_templates_not_in_upcoming(self, manager):
        """Test conditional templates are not in upcoming triggers."""
        template = RecurringOrderTemplate(
            template_id="cond_test",
            name="Conditional Test",
            schedule_type=ScheduleType.CONDITIONAL,
            order_type=OrderType.OPTION_STRATEGY,
            symbol="SPY",
            quantity=1,
        )

        manager.add_template(template)

        current_time = datetime.now()
        upcoming = manager.get_upcoming_triggers(current_time, days_ahead=7)

        assert len(upcoming) == 0  # Conditional triggers can't be predicted


class TestStatistics:
    """Tests for manager statistics."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager instance for testing."""
        algorithm = Mock()
        algorithm.Debug = Mock()
        return create_recurring_order_manager(
            algorithm,
            storage_path=tmp_path,
        )

    @pytest.mark.unit
    def test_statistics_tracking(self, manager):
        """Test that statistics are tracked correctly."""
        # Add templates
        for i in range(3):
            template = RecurringOrderTemplate(
                template_id=f"test_{i}",
                name=f"Test {i}",
                order_type=OrderType.OPTION_STRATEGY,
                symbol="SPY",
                quantity=1,
            )
            manager.add_template(template)

        # Execute one successfully
        def success_executor(tmpl):
            return "order_123"

        manager.execute_template(
            manager.templates["test_0"],
            success_executor,
        )

        # Execute one with failure
        def fail_executor(tmpl):
            return None

        manager.execute_template(
            manager.templates["test_1"],
            fail_executor,
        )

        stats = manager.get_statistics()

        assert stats["total_templates"] == 3
        assert stats["active_templates"] == 3
        assert stats["successful_executions"] == 1
        assert stats["failed_executions"] == 1
