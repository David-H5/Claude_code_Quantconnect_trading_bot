"""
FINRA Compliance Checklist Module

UPGRADE-015 Phase 11: Compliance and Audit Logging

Validates trading system compliance with FINRA requirements:
- Rule 3110: Supervision
- Rule 4511: Books and Records
- Rule 5310: Best Execution
- Rule 6140: Other Trading Practices

Provides automated compliance checking and reporting.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ChecklistStatus(Enum):
    """Status of a checklist item."""

    NOT_CHECKED = "not_checked"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class ChecklistCategory(Enum):
    """Categories of FINRA compliance checks."""

    SUPERVISION = "supervision"  # Rule 3110
    RECORDS = "records"  # Rule 4511
    BEST_EXECUTION = "best_execution"  # Rule 5310
    TRADING_PRACTICES = "trading_practices"  # Rule 6140
    MARKET_MANIPULATION = "market_manipulation"  # Various
    SUITABILITY = "suitability"  # Rule 2111
    REPORTING = "reporting"  # Various
    RISK_MANAGEMENT = "risk_management"


class RuleReference(Enum):
    """FINRA rule references."""

    RULE_3110 = "FINRA Rule 3110 - Supervision"
    RULE_4511 = "FINRA Rule 4511 - General Requirements"
    RULE_5310 = "FINRA Rule 5310 - Best Execution"
    RULE_6140 = "FINRA Rule 6140 - Other Trading Practices"
    RULE_2111 = "FINRA Rule 2111 - Suitability"
    RULE_2010 = "FINRA Rule 2010 - Standards of Commercial Honor"
    SEC_10B5 = "SEC Rule 10b-5 - Fraud and Manipulation"
    SEC_17A4 = "SEC Rule 17a-4 - Records to be Preserved"


@dataclass
class ChecklistItem:
    """Individual compliance checklist item."""

    item_id: str
    name: str
    description: str
    category: ChecklistCategory
    rule_reference: RuleReference
    required: bool = True

    # Check function
    check_function: Callable[..., tuple[ChecklistStatus, str]] | None = None

    # Result
    status: ChecklistStatus = ChecklistStatus.NOT_CHECKED
    result_message: str = ""
    checked_at: datetime | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "rule_reference": self.rule_reference.value,
            "required": self.required,
            "status": self.status.value,
            "result_message": self.result_message,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
            "evidence": self.evidence,
        }


@dataclass
class ChecklistResult:
    """Result of running compliance checklist."""

    checklist_name: str
    run_at: datetime
    total_items: int
    passed: int
    failed: int
    warnings: int
    not_applicable: int
    not_checked: int

    # Overall status
    overall_status: ChecklistStatus = ChecklistStatus.NOT_CHECKED
    compliance_score: float = 0.0

    # Details
    items: list[ChecklistItem] = field(default_factory=list)
    critical_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checklist_name": self.checklist_name,
            "run_at": self.run_at.isoformat(),
            "total_items": self.total_items,
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "not_applicable": self.not_applicable,
            "not_checked": self.not_checked,
            "overall_status": self.overall_status.value,
            "compliance_score": self.compliance_score,
            "critical_failures": self.critical_failures,
            "items": [i.to_dict() for i in self.items],
        }


class FINRAChecklist:
    """FINRA compliance checklist for trading systems."""

    def __init__(self):
        """Initialize FINRA checklist with standard items."""
        self._items: list[ChecklistItem] = []
        self._last_result: ChecklistResult | None = None

        # Initialize standard checklist items
        self._initialize_checklist()

    # ==========================================================================
    # Initialization
    # ==========================================================================

    def _initialize_checklist(self) -> None:
        """Initialize standard FINRA compliance checklist items."""
        # Supervision (Rule 3110)
        self._items.extend(
            [
                ChecklistItem(
                    item_id="SUP-001",
                    name="Supervisory System",
                    description="Written supervisory procedures are in place and documented",
                    category=ChecklistCategory.SUPERVISION,
                    rule_reference=RuleReference.RULE_3110,
                ),
                ChecklistItem(
                    item_id="SUP-002",
                    name="Designated Supervisor",
                    description="Qualified supervisor designated for algorithmic trading activities",
                    category=ChecklistCategory.SUPERVISION,
                    rule_reference=RuleReference.RULE_3110,
                ),
                ChecklistItem(
                    item_id="SUP-003",
                    name="Review of Trading Activity",
                    description="Regular review of trading activity is conducted and documented",
                    category=ChecklistCategory.SUPERVISION,
                    rule_reference=RuleReference.RULE_3110,
                ),
            ]
        )

        # Records (Rule 4511)
        self._items.extend(
            [
                ChecklistItem(
                    item_id="REC-001",
                    name="Order Records",
                    description="All orders are recorded with required details (time, price, quantity, symbol)",
                    category=ChecklistCategory.RECORDS,
                    rule_reference=RuleReference.RULE_4511,
                ),
                ChecklistItem(
                    item_id="REC-002",
                    name="Trade Records",
                    description="All trades are recorded with execution details",
                    category=ChecklistCategory.RECORDS,
                    rule_reference=RuleReference.RULE_4511,
                ),
                ChecklistItem(
                    item_id="REC-003",
                    name="Retention Period",
                    description="Records are retained for minimum 6 years (3 years accessible)",
                    category=ChecklistCategory.RECORDS,
                    rule_reference=RuleReference.SEC_17A4,
                ),
                ChecklistItem(
                    item_id="REC-004",
                    name="Audit Trail",
                    description="Complete audit trail maintained for all trading decisions",
                    category=ChecklistCategory.RECORDS,
                    rule_reference=RuleReference.RULE_4511,
                ),
            ]
        )

        # Best Execution (Rule 5310)
        self._items.extend(
            [
                ChecklistItem(
                    item_id="BEX-001",
                    name="Best Execution Policy",
                    description="Written best execution policy exists",
                    category=ChecklistCategory.BEST_EXECUTION,
                    rule_reference=RuleReference.RULE_5310,
                ),
                ChecklistItem(
                    item_id="BEX-002",
                    name="Execution Quality Monitoring",
                    description="Regular monitoring of execution quality",
                    category=ChecklistCategory.BEST_EXECUTION,
                    rule_reference=RuleReference.RULE_5310,
                ),
                ChecklistItem(
                    item_id="BEX-003",
                    name="Venue Evaluation",
                    description="Regular evaluation of execution venues",
                    category=ChecklistCategory.BEST_EXECUTION,
                    rule_reference=RuleReference.RULE_5310,
                ),
            ]
        )

        # Trading Practices (Rule 6140)
        self._items.extend(
            [
                ChecklistItem(
                    item_id="TRD-001",
                    name="Pre-Trade Controls",
                    description="Pre-trade risk controls implemented (position limits, price limits)",
                    category=ChecklistCategory.TRADING_PRACTICES,
                    rule_reference=RuleReference.RULE_6140,
                ),
                ChecklistItem(
                    item_id="TRD-002",
                    name="Kill Switch",
                    description="Ability to immediately halt all trading activity",
                    category=ChecklistCategory.TRADING_PRACTICES,
                    rule_reference=RuleReference.RULE_6140,
                ),
                ChecklistItem(
                    item_id="TRD-003",
                    name="Error Prevention",
                    description="Controls to prevent erroneous orders (fat finger checks)",
                    category=ChecklistCategory.TRADING_PRACTICES,
                    rule_reference=RuleReference.RULE_6140,
                ),
            ]
        )

        # Market Manipulation
        self._items.extend(
            [
                ChecklistItem(
                    item_id="MAN-001",
                    name="Anti-Spoofing Controls",
                    description="Controls to prevent spoofing behavior",
                    category=ChecklistCategory.MARKET_MANIPULATION,
                    rule_reference=RuleReference.SEC_10B5,
                ),
                ChecklistItem(
                    item_id="MAN-002",
                    name="Anti-Layering Controls",
                    description="Controls to prevent layering behavior",
                    category=ChecklistCategory.MARKET_MANIPULATION,
                    rule_reference=RuleReference.SEC_10B5,
                ),
                ChecklistItem(
                    item_id="MAN-003",
                    name="Wash Trade Prevention",
                    description="Controls to prevent wash trading",
                    category=ChecklistCategory.MARKET_MANIPULATION,
                    rule_reference=RuleReference.SEC_10B5,
                ),
                ChecklistItem(
                    item_id="MAN-004",
                    name="Manipulation Monitoring",
                    description="Ongoing monitoring for manipulation patterns",
                    category=ChecklistCategory.MARKET_MANIPULATION,
                    rule_reference=RuleReference.SEC_10B5,
                ),
            ]
        )

        # Risk Management
        self._items.extend(
            [
                ChecklistItem(
                    item_id="RSK-001",
                    name="Position Limits",
                    description="Position limits defined and enforced",
                    category=ChecklistCategory.RISK_MANAGEMENT,
                    rule_reference=RuleReference.RULE_3110,
                ),
                ChecklistItem(
                    item_id="RSK-002",
                    name="Loss Limits",
                    description="Daily loss limits defined and monitored",
                    category=ChecklistCategory.RISK_MANAGEMENT,
                    rule_reference=RuleReference.RULE_3110,
                ),
                ChecklistItem(
                    item_id="RSK-003",
                    name="Circuit Breaker",
                    description="Circuit breaker functionality implemented",
                    category=ChecklistCategory.RISK_MANAGEMENT,
                    rule_reference=RuleReference.RULE_6140,
                ),
                ChecklistItem(
                    item_id="RSK-004",
                    name="Risk Alerts",
                    description="Real-time risk alerts configured",
                    category=ChecklistCategory.RISK_MANAGEMENT,
                    rule_reference=RuleReference.RULE_3110,
                ),
            ]
        )

        # Reporting
        self._items.extend(
            [
                ChecklistItem(
                    item_id="RPT-001",
                    name="Trade Reporting",
                    description="Trades reported to appropriate venues/regulators",
                    category=ChecklistCategory.REPORTING,
                    rule_reference=RuleReference.RULE_6140,
                ),
                ChecklistItem(
                    item_id="RPT-002",
                    name="Exception Reporting",
                    description="Exception reports generated for unusual activity",
                    category=ChecklistCategory.REPORTING,
                    rule_reference=RuleReference.RULE_3110,
                ),
            ]
        )

    # ==========================================================================
    # Check Execution
    # ==========================================================================

    def run_checklist(
        self,
        context: dict[str, Any] | None = None,
    ) -> ChecklistResult:
        """
        Run the full compliance checklist.

        Args:
            context: Context data for checks

        Returns:
            ChecklistResult with all check outcomes
        """
        context = context or {}

        passed = 0
        failed = 0
        warnings = 0
        not_applicable = 0
        not_checked = 0
        critical_failures = []

        for item in self._items:
            if item.check_function:
                try:
                    status, message = item.check_function(context)
                    item.status = status
                    item.result_message = message
                except Exception as e:
                    item.status = ChecklistStatus.FAILED
                    item.result_message = f"Check failed with error: {e!s}"
            else:
                # No automated check - mark as not checked
                item.status = ChecklistStatus.NOT_CHECKED
                item.result_message = "Manual verification required"

            item.checked_at = datetime.utcnow()

            # Count results
            if item.status == ChecklistStatus.PASSED:
                passed += 1
            elif item.status == ChecklistStatus.FAILED:
                failed += 1
                if item.required:
                    critical_failures.append(item.item_id)
            elif item.status == ChecklistStatus.WARNING:
                warnings += 1
            elif item.status == ChecklistStatus.NOT_APPLICABLE:
                not_applicable += 1
            else:
                not_checked += 1

        # Calculate compliance score
        checkable = passed + failed + warnings
        compliance_score = passed / checkable if checkable > 0 else 0

        # Determine overall status
        if failed > 0:
            overall_status = ChecklistStatus.FAILED
        elif warnings > 0:
            overall_status = ChecklistStatus.WARNING
        elif not_checked > 0:
            overall_status = ChecklistStatus.NOT_CHECKED
        else:
            overall_status = ChecklistStatus.PASSED

        result = ChecklistResult(
            checklist_name="FINRA Trading Compliance",
            run_at=datetime.utcnow(),
            total_items=len(self._items),
            passed=passed,
            failed=failed,
            warnings=warnings,
            not_applicable=not_applicable,
            not_checked=not_checked,
            overall_status=overall_status,
            compliance_score=compliance_score,
            items=self._items.copy(),
            critical_failures=critical_failures,
        )

        self._last_result = result
        return result

    def check_item(
        self,
        item_id: str,
        status: ChecklistStatus,
        message: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> bool:
        """
        Manually check a specific item.

        Args:
            item_id: ID of item to check
            status: Status to set
            message: Result message
            evidence: Supporting evidence

        Returns:
            True if item was found and updated
        """
        for item in self._items:
            if item.item_id == item_id:
                item.status = status
                item.result_message = message
                item.checked_at = datetime.utcnow()
                item.evidence = evidence or {}
                return True
        return False

    def register_check(
        self,
        item_id: str,
        check_function: Callable[..., tuple[ChecklistStatus, str]],
    ) -> bool:
        """
        Register an automated check function for an item.

        Args:
            item_id: ID of item
            check_function: Function that returns (status, message)

        Returns:
            True if registered successfully
        """
        for item in self._items:
            if item.item_id == item_id:
                item.check_function = check_function
                return True
        return False

    # ==========================================================================
    # Custom Checks
    # ==========================================================================

    def add_custom_item(
        self,
        item_id: str,
        name: str,
        description: str,
        category: ChecklistCategory,
        rule_reference: RuleReference,
        required: bool = True,
        check_function: Callable[..., tuple[ChecklistStatus, str]] | None = None,
    ) -> ChecklistItem:
        """Add a custom checklist item."""
        item = ChecklistItem(
            item_id=item_id,
            name=name,
            description=description,
            category=category,
            rule_reference=rule_reference,
            required=required,
            check_function=check_function,
        )
        self._items.append(item)
        return item

    # ==========================================================================
    # Retrieval
    # ==========================================================================

    def get_items(
        self,
        category: ChecklistCategory | None = None,
        status: ChecklistStatus | None = None,
    ) -> list[ChecklistItem]:
        """Get filtered checklist items."""
        results = []
        for item in self._items:
            if category and item.category != category:
                continue
            if status and item.status != status:
                continue
            results.append(item)
        return results

    def get_failed_items(self) -> list[ChecklistItem]:
        """Get all failed items."""
        return self.get_items(status=ChecklistStatus.FAILED)

    def get_last_result(self) -> ChecklistResult | None:
        """Get the last run result."""
        return self._last_result

    def get_summary(self) -> dict[str, Any]:
        """Get checklist summary."""
        by_category: dict[str, dict[str, int]] = {}
        for item in self._items:
            cat = item.category.value
            if cat not in by_category:
                by_category[cat] = {"total": 0, "passed": 0, "failed": 0}
            by_category[cat]["total"] += 1
            if item.status == ChecklistStatus.PASSED:
                by_category[cat]["passed"] += 1
            elif item.status == ChecklistStatus.FAILED:
                by_category[cat]["failed"] += 1

        return {
            "total_items": len(self._items),
            "by_category": by_category,
            "last_run": self._last_result.run_at.isoformat() if self._last_result else None,
            "compliance_score": self._last_result.compliance_score if self._last_result else None,
        }


# ==========================================================================
# Standard Check Functions
# ==========================================================================


def check_audit_trail_exists(context: dict[str, Any]) -> tuple[ChecklistStatus, str]:
    """Check if audit trail is configured."""
    audit_logger = context.get("audit_logger")
    if audit_logger:
        stats = audit_logger.get_stats() if hasattr(audit_logger, "get_stats") else {}
        if stats.get("total_entries", 0) > 0:
            return ChecklistStatus.PASSED, f"Audit trail active with {stats['total_entries']} entries"
        return ChecklistStatus.WARNING, "Audit trail configured but no entries"
    return ChecklistStatus.FAILED, "No audit logger configured"


def check_circuit_breaker(context: dict[str, Any]) -> tuple[ChecklistStatus, str]:
    """Check if circuit breaker is configured."""
    circuit_breaker = context.get("circuit_breaker")
    if circuit_breaker:
        return ChecklistStatus.PASSED, "Circuit breaker configured"
    return ChecklistStatus.FAILED, "No circuit breaker configured"


def check_position_limits(context: dict[str, Any]) -> tuple[ChecklistStatus, str]:
    """Check if position limits are configured."""
    limits = context.get("position_limits", {})
    if limits.get("max_position_pct"):
        return ChecklistStatus.PASSED, f"Position limit: {limits['max_position_pct']:.0%}"
    return ChecklistStatus.FAILED, "No position limits configured"


def check_manipulation_monitor(context: dict[str, Any]) -> tuple[ChecklistStatus, str]:
    """Check if manipulation monitoring is configured."""
    monitor = context.get("manipulation_monitor")
    if monitor:
        return ChecklistStatus.PASSED, "Anti-manipulation monitoring active"
    return ChecklistStatus.WARNING, "No manipulation monitor configured"


def create_finra_checklist() -> FINRAChecklist:
    """
    Factory function to create a FINRA checklist.

    Returns:
        Configured FINRAChecklist
    """
    checklist = FINRAChecklist()

    # Register standard automated checks
    checklist.register_check("REC-004", check_audit_trail_exists)
    checklist.register_check("RSK-003", check_circuit_breaker)
    checklist.register_check("RSK-001", check_position_limits)
    checklist.register_check("MAN-004", check_manipulation_monitor)

    return checklist
