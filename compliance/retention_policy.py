"""
Retention Policy Module

UPGRADE-015 Phase 11: Compliance and Audit Logging

Manages data retention policies for regulatory compliance:
- Configurable retention periods by data type
- Automatic archival and deletion
- Compliance tracking
- Legal hold support

Regulatory Requirements:
- SOX: 7 years for financial records
- PCI DSS 4.0: 12 months minimum
- FINRA: 6 years for trading records
- SEC: Various depending on record type
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class DataCategory(Enum):
    """Categories of data for retention."""

    TRADING_RECORDS = "trading_records"  # Orders, trades, fills
    AUDIT_LOGS = "audit_logs"  # System audit trail
    COMMUNICATIONS = "communications"  # Client communications
    ACCOUNT_RECORDS = "account_records"  # Account statements
    RISK_DATA = "risk_data"  # Risk calculations
    MARKET_DATA = "market_data"  # Price/quote data
    CONFIGURATION = "configuration"  # System configs
    TEMPORARY = "temporary"  # Temp/cache data


class RetentionAction(Enum):
    """Actions to take on expired data."""

    DELETE = "delete"  # Permanently delete
    ARCHIVE = "archive"  # Move to archive storage
    ENCRYPT = "encrypt"  # Encrypt in place
    ANONYMIZE = "anonymize"  # Remove PII, keep data


class LegalHoldStatus(Enum):
    """Legal hold status."""

    NONE = "none"
    ACTIVE = "active"
    RELEASED = "released"


@dataclass
class RetentionRule:
    """Retention rule for a data category."""

    category: DataCategory
    retention_days: int  # Days to retain
    action: RetentionAction = RetentionAction.DELETE
    archive_location: str = ""
    description: str = ""
    regulatory_basis: str = ""  # SOX, FINRA, etc.

    def is_expired(self, data_date: datetime) -> bool:
        """Check if data has exceeded retention period."""
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        return data_date < cutoff

    def days_until_expiry(self, data_date: datetime) -> int:
        """Calculate days until data expires."""
        expiry_date = data_date + timedelta(days=self.retention_days)
        remaining = (expiry_date - datetime.utcnow()).days
        return max(0, remaining)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "retention_days": self.retention_days,
            "retention_years": round(self.retention_days / 365, 1),
            "action": self.action.value,
            "archive_location": self.archive_location,
            "description": self.description,
            "regulatory_basis": self.regulatory_basis,
        }


@dataclass
class LegalHold:
    """Legal hold on data."""

    hold_id: str
    description: str
    created_at: datetime
    created_by: str
    categories: list[DataCategory] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)  # Empty = all
    start_date: datetime | None = None  # Data start date
    end_date: datetime | None = None  # Data end date
    status: LegalHoldStatus = LegalHoldStatus.ACTIVE
    released_at: datetime | None = None
    released_by: str = ""

    def covers_data(
        self,
        category: DataCategory,
        data_date: datetime,
        symbol: str = "",
    ) -> bool:
        """Check if this hold covers specific data."""
        if self.status != LegalHoldStatus.ACTIVE:
            return False

        # Check category
        if self.categories and category not in self.categories:
            return False

        # Check symbol
        if self.symbols and symbol and symbol not in self.symbols:
            return False

        # Check date range
        if self.start_date and data_date < self.start_date:
            return False

        return not (self.end_date and data_date > self.end_date)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hold_id": self.hold_id,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "categories": [c.value for c in self.categories],
            "symbols": self.symbols,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "status": self.status.value,
            "released_at": self.released_at.isoformat() if self.released_at else None,
            "released_by": self.released_by,
        }


@dataclass
class RetentionPolicy:
    """Complete retention policy configuration."""

    name: str = "default"
    rules: list[RetentionRule] = field(default_factory=list)
    legal_holds: list[LegalHold] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    updated_by: str = ""

    def get_rule(self, category: DataCategory) -> RetentionRule | None:
        """Get retention rule for category."""
        for rule in self.rules:
            if rule.category == category:
                return rule
        return None

    def is_under_hold(
        self,
        category: DataCategory,
        data_date: datetime,
        symbol: str = "",
    ) -> tuple[bool, list[str]]:
        """Check if data is under legal hold."""
        holding_ids = []
        for hold in self.legal_holds:
            if hold.covers_data(category, data_date, symbol):
                holding_ids.append(hold.hold_id)
        return len(holding_ids) > 0, holding_ids

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "rules": [r.to_dict() for r in self.rules],
            "legal_holds": [h.to_dict() for h in self.legal_holds],
            "last_updated": self.last_updated.isoformat(),
            "updated_by": self.updated_by,
        }


@dataclass
class RetentionCheckResult:
    """Result of checking data against retention policy."""

    category: DataCategory
    data_date: datetime
    is_expired: bool
    days_remaining: int
    recommended_action: RetentionAction
    under_legal_hold: bool
    hold_ids: list[str] = field(default_factory=list)
    can_delete: bool = False
    notes: str = ""


class RetentionPolicyManager:
    """Manage retention policies and enforce compliance."""

    def __init__(
        self,
        policy: RetentionPolicy | None = None,
        policy_file: Path | None = None,
    ):
        """
        Initialize retention policy manager.

        Args:
            policy: Initial policy
            policy_file: File to persist policy
        """
        self.policy = policy or self._create_default_policy()
        self.policy_file = policy_file
        self._hold_counter = 0

        # Load from file if exists
        if policy_file and policy_file.exists():
            self.load_policy(policy_file)

    # ==========================================================================
    # Default Policy
    # ==========================================================================

    def _create_default_policy(self) -> RetentionPolicy:
        """Create default retention policy based on regulations."""
        rules = [
            RetentionRule(
                category=DataCategory.TRADING_RECORDS,
                retention_days=2555,  # 7 years (SOX, FINRA)
                action=RetentionAction.ARCHIVE,
                description="Trade execution records including orders, fills, and cancellations",
                regulatory_basis="SOX Section 802, FINRA Rule 4511",
            ),
            RetentionRule(
                category=DataCategory.AUDIT_LOGS,
                retention_days=2555,  # 7 years (SOX)
                action=RetentionAction.ARCHIVE,
                description="System audit trail and access logs",
                regulatory_basis="SOX Section 802",
            ),
            RetentionRule(
                category=DataCategory.COMMUNICATIONS,
                retention_days=2190,  # 6 years (FINRA)
                action=RetentionAction.ARCHIVE,
                description="Client communications and correspondence",
                regulatory_basis="FINRA Rule 4511, SEC Rule 17a-4",
            ),
            RetentionRule(
                category=DataCategory.ACCOUNT_RECORDS,
                retention_days=2190,  # 6 years
                action=RetentionAction.ARCHIVE,
                description="Account statements and financial records",
                regulatory_basis="FINRA Rule 4511",
            ),
            RetentionRule(
                category=DataCategory.RISK_DATA,
                retention_days=1825,  # 5 years
                action=RetentionAction.ARCHIVE,
                description="Risk calculations and limit monitoring",
                regulatory_basis="Internal policy",
            ),
            RetentionRule(
                category=DataCategory.MARKET_DATA,
                retention_days=365,  # 1 year
                action=RetentionAction.DELETE,
                description="Historical market data (quotes, prices)",
                regulatory_basis="Internal policy - can be re-obtained",
            ),
            RetentionRule(
                category=DataCategory.CONFIGURATION,
                retention_days=2555,  # 7 years
                action=RetentionAction.ARCHIVE,
                description="System configuration history",
                regulatory_basis="SOX audit requirements",
            ),
            RetentionRule(
                category=DataCategory.TEMPORARY,
                retention_days=7,  # 1 week
                action=RetentionAction.DELETE,
                description="Temporary files and cache",
                regulatory_basis="N/A",
            ),
        ]

        return RetentionPolicy(
            name="default_trading_policy",
            rules=rules,
            last_updated=datetime.utcnow(),
            updated_by="system",
        )

    # ==========================================================================
    # Policy Management
    # ==========================================================================

    def add_rule(self, rule: RetentionRule) -> None:
        """Add or update a retention rule."""
        # Remove existing rule for category
        self.policy.rules = [r for r in self.policy.rules if r.category != rule.category]
        self.policy.rules.append(rule)
        self.policy.last_updated = datetime.utcnow()

    def remove_rule(self, category: DataCategory) -> bool:
        """Remove a retention rule."""
        original_count = len(self.policy.rules)
        self.policy.rules = [r for r in self.policy.rules if r.category != category]
        return len(self.policy.rules) < original_count

    def get_rule(self, category: DataCategory) -> RetentionRule | None:
        """Get retention rule for category."""
        return self.policy.get_rule(category)

    # ==========================================================================
    # Legal Hold Management
    # ==========================================================================

    def create_legal_hold(
        self,
        description: str,
        created_by: str,
        categories: list[DataCategory] | None = None,
        symbols: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> LegalHold:
        """
        Create a new legal hold.

        Args:
            description: Hold description
            created_by: Who created the hold
            categories: Data categories to hold
            symbols: Symbols to hold (empty = all)
            start_date: Start date of data to hold
            end_date: End date of data to hold

        Returns:
            Created LegalHold
        """
        self._hold_counter += 1
        hold_id = f"HOLD-{datetime.utcnow().strftime('%Y%m%d')}-{self._hold_counter:04d}"

        hold = LegalHold(
            hold_id=hold_id,
            description=description,
            created_at=datetime.utcnow(),
            created_by=created_by,
            categories=categories or [],
            symbols=symbols or [],
            start_date=start_date,
            end_date=end_date,
            status=LegalHoldStatus.ACTIVE,
        )

        self.policy.legal_holds.append(hold)
        self.policy.last_updated = datetime.utcnow()

        return hold

    def release_legal_hold(
        self,
        hold_id: str,
        released_by: str,
    ) -> bool:
        """Release a legal hold."""
        for hold in self.policy.legal_holds:
            if hold.hold_id == hold_id and hold.status == LegalHoldStatus.ACTIVE:
                hold.status = LegalHoldStatus.RELEASED
                hold.released_at = datetime.utcnow()
                hold.released_by = released_by
                self.policy.last_updated = datetime.utcnow()
                return True
        return False

    def get_active_holds(self) -> list[LegalHold]:
        """Get all active legal holds."""
        return [h for h in self.policy.legal_holds if h.status == LegalHoldStatus.ACTIVE]

    # ==========================================================================
    # Retention Checks
    # ==========================================================================

    def check_retention(
        self,
        category: DataCategory,
        data_date: datetime,
        symbol: str = "",
    ) -> RetentionCheckResult:
        """
        Check if data should be retained or can be deleted.

        Args:
            category: Data category
            data_date: Date of the data
            symbol: Symbol (if applicable)

        Returns:
            RetentionCheckResult
        """
        rule = self.policy.get_rule(category)

        if not rule:
            # No rule = keep indefinitely
            return RetentionCheckResult(
                category=category,
                data_date=data_date,
                is_expired=False,
                days_remaining=-1,  # Indefinite
                recommended_action=RetentionAction.ARCHIVE,
                under_legal_hold=False,
                can_delete=False,
                notes="No retention rule defined - keeping indefinitely",
            )

        is_expired = rule.is_expired(data_date)
        days_remaining = rule.days_until_expiry(data_date)
        under_hold, hold_ids = self.policy.is_under_hold(category, data_date, symbol)

        # Determine if deletion is allowed
        can_delete = is_expired and not under_hold

        notes = ""
        if under_hold:
            notes = f"Under legal hold: {', '.join(hold_ids)}"
        elif is_expired:
            notes = f"Expired - {rule.action.value} recommended"
        else:
            notes = f"{days_remaining} days until expiry"

        return RetentionCheckResult(
            category=category,
            data_date=data_date,
            is_expired=is_expired,
            days_remaining=days_remaining,
            recommended_action=rule.action,
            under_legal_hold=under_hold,
            hold_ids=hold_ids,
            can_delete=can_delete,
            notes=notes,
        )

    def get_expired_data_summary(
        self,
        data_dates: dict[DataCategory, list[datetime]],
    ) -> dict[str, Any]:
        """
        Get summary of expired data across categories.

        Args:
            data_dates: Dict mapping categories to data dates

        Returns:
            Summary of expired data
        """
        summary = {
            "total_checked": 0,
            "expired": 0,
            "under_hold": 0,
            "can_delete": 0,
            "by_category": {},
        }

        for category, dates in data_dates.items():
            cat_summary = {
                "total": len(dates),
                "expired": 0,
                "under_hold": 0,
                "can_delete": 0,
            }

            for data_date in dates:
                result = self.check_retention(category, data_date)
                summary["total_checked"] += 1

                if result.is_expired:
                    summary["expired"] += 1
                    cat_summary["expired"] += 1

                if result.under_legal_hold:
                    summary["under_hold"] += 1
                    cat_summary["under_hold"] += 1

                if result.can_delete:
                    summary["can_delete"] += 1
                    cat_summary["can_delete"] += 1

            summary["by_category"][category.value] = cat_summary

        return summary

    # ==========================================================================
    # Persistence
    # ==========================================================================

    def save_policy(self, filepath: Path | None = None) -> None:
        """Save policy to file."""
        filepath = filepath or self.policy_file
        if not filepath:
            return

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.policy.to_dict(), f, indent=2, default=str)

    def load_policy(self, filepath: Path) -> None:
        """Load policy from file."""
        with open(filepath) as f:
            data = json.load(f)

        rules = []
        for rule_data in data.get("rules", []):
            rules.append(
                RetentionRule(
                    category=DataCategory(rule_data["category"]),
                    retention_days=rule_data["retention_days"],
                    action=RetentionAction(rule_data.get("action", "delete")),
                    archive_location=rule_data.get("archive_location", ""),
                    description=rule_data.get("description", ""),
                    regulatory_basis=rule_data.get("regulatory_basis", ""),
                )
            )

        holds = []
        for hold_data in data.get("legal_holds", []):
            holds.append(
                LegalHold(
                    hold_id=hold_data["hold_id"],
                    description=hold_data["description"],
                    created_at=datetime.fromisoformat(hold_data["created_at"]),
                    created_by=hold_data["created_by"],
                    categories=[DataCategory(c) for c in hold_data.get("categories", [])],
                    symbols=hold_data.get("symbols", []),
                    start_date=datetime.fromisoformat(hold_data["start_date"]) if hold_data.get("start_date") else None,
                    end_date=datetime.fromisoformat(hold_data["end_date"]) if hold_data.get("end_date") else None,
                    status=LegalHoldStatus(hold_data.get("status", "active")),
                    released_at=datetime.fromisoformat(hold_data["released_at"])
                    if hold_data.get("released_at")
                    else None,
                    released_by=hold_data.get("released_by", ""),
                )
            )

        self.policy = RetentionPolicy(
            name=data.get("name", "loaded_policy"),
            rules=rules,
            legal_holds=holds,
            last_updated=datetime.fromisoformat(data.get("last_updated", datetime.utcnow().isoformat())),
            updated_by=data.get("updated_by", ""),
        )

    # ==========================================================================
    # Reporting
    # ==========================================================================

    def get_policy_summary(self) -> dict[str, Any]:
        """Get policy summary."""
        return {
            "policy_name": self.policy.name,
            "last_updated": self.policy.last_updated.isoformat(),
            "rule_count": len(self.policy.rules),
            "active_holds": len(self.get_active_holds()),
            "rules": [
                {
                    "category": r.category.value,
                    "retention_years": round(r.retention_days / 365, 1),
                    "action": r.action.value,
                    "regulatory_basis": r.regulatory_basis,
                }
                for r in self.policy.rules
            ],
        }


def create_retention_manager(
    policy_file: str | Path | None = None,
) -> RetentionPolicyManager:
    """
    Factory function to create a retention policy manager.

    Args:
        policy_file: Path to policy file

    Returns:
        Configured RetentionPolicyManager
    """
    return RetentionPolicyManager(
        policy_file=Path(policy_file) if policy_file else None,
    )
