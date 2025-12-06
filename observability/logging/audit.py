"""
Audit Logger Module

UPGRADE-015 Phase 11: Compliance and Audit Logging
Refactored: Phase 2 - Consolidated Logging Infrastructure

Provides comprehensive audit logging for trading compliance:
- Immutable audit trail
- Structured logging with required fields
- Hash chain for tamper detection
- Retention period enforcement

Compliance Requirements:
- SOX: 7 years retention
- PCI DSS 4.0: 12 months accessible
- FINRA: 6+ years for trading records
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

# Import base types from unified logging infrastructure
from observability.logging.base import (
    AbstractLogger,
)


class AuditLevel(Enum):
    """Audit log severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditCategory(Enum):
    """Audit log categories."""

    ORDER = "order"
    TRADE = "trade"
    POSITION = "position"
    RISK = "risk"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    COMPLIANCE = "compliance"


@dataclass
class AuditEntry:
    """Single audit log entry."""

    # Required fields (per compliance requirements)
    timestamp: datetime
    actor: str  # User or system identifier
    action: str  # What happened
    resource: str  # Affected entity
    outcome: str  # SUCCESS, FAILED, PENDING

    # Optional but recommended
    category: AuditCategory = AuditCategory.SYSTEM
    level: AuditLevel = AuditLevel.INFO
    details: dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    session_id: str = ""

    # Hash chain for integrity
    entry_hash: str = ""
    previous_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "category": self.category.value,
            "level": self.level.value,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "entry_hash": self.entry_hash,
            "previous_hash": self.previous_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEntry:
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            actor=data["actor"],
            action=data["action"],
            resource=data["resource"],
            outcome=data["outcome"],
            category=AuditCategory(data.get("category", "system")),
            level=AuditLevel(data.get("level", "info")),
            details=data.get("details", {}),
            correlation_id=data.get("correlation_id", ""),
            session_id=data.get("session_id", ""),
            entry_hash=data.get("entry_hash", ""),
            previous_hash=data.get("previous_hash", ""),
        )

    def calculate_hash(self) -> str:
        """Calculate hash of entry content."""
        content = json.dumps(
            {
                "timestamp": self.timestamp.isoformat(),
                "actor": self.actor,
                "action": self.action,
                "resource": self.resource,
                "outcome": self.outcome,
                "category": self.category.value,
                "details": self.details,
                "previous_hash": self.previous_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class AuditTrail:
    """Complete audit trail with integrity verification."""

    entries: list[AuditEntry] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    trail_hash: str = ""

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """Verify the integrity of the audit trail."""
        issues = []

        if not self.entries:
            return True, []

        # Check first entry
        if self.entries[0].previous_hash != "":
            issues.append("First entry should have empty previous_hash")

        # Check hash chain
        for i in range(1, len(self.entries)):
            expected_prev = self.entries[i - 1].entry_hash
            actual_prev = self.entries[i].previous_hash

            if expected_prev != actual_prev:
                issues.append(
                    f"Hash chain broken at entry {i}: expected {expected_prev[:8]}..., got {actual_prev[:8]}..."
                )

            # Verify entry hash
            calculated = self.entries[i].calculate_hash()
            if calculated != self.entries[i].entry_hash:
                issues.append(
                    f"Entry {i} hash mismatch: calculated {calculated[:8]}..., stored {self.entries[i].entry_hash[:8]}..."
                )

        return len(issues) == 0, issues

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entries": [e.to_dict() for e in self.entries],
            "start_time": self.start_time.isoformat(),
            "trail_hash": self.trail_hash,
            "entry_count": len(self.entries),
        }


class AuditLogger(AbstractLogger):
    """
    Comprehensive audit logger for trading compliance.

    Implements AbstractLogger interface from observability.logging.base.
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        retention_days: int = 2555,  # ~7 years for SOX
        enable_hash_chain: bool = True,
        auto_persist: bool = True,
    ):
        """
        Initialize audit logger.

        Args:
            log_dir: Directory for log files
            retention_days: Days to retain logs (default 7 years)
            enable_hash_chain: Enable hash chain for integrity
            auto_persist: Automatically persist entries to disk
        """
        self.log_dir = log_dir or Path("audit_logs")
        self.retention_days = retention_days
        self.enable_hash_chain = enable_hash_chain
        self.auto_persist = auto_persist

        self._entries: list[AuditEntry] = []
        self._last_hash = ""
        self._session_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")

        if self.auto_persist:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Core Logging
    # ==========================================================================

    def log(
        self,
        action: str,
        resource: str,
        outcome: str,
        actor: str = "system",
        category: AuditCategory = AuditCategory.SYSTEM,
        level: AuditLevel = AuditLevel.INFO,
        details: dict[str, Any] | None = None,
        correlation_id: str = "",
    ) -> AuditEntry:
        """
        Log an audit entry.

        Args:
            action: What happened
            resource: Affected entity
            outcome: Result (SUCCESS, FAILED, PENDING)
            actor: Who or what performed the action
            category: Log category
            level: Log severity level
            details: Additional details
            correlation_id: ID to correlate related entries

        Returns:
            The created audit entry
        """
        entry = AuditEntry(
            timestamp=datetime.utcnow(),
            actor=actor,
            action=action,
            resource=resource,
            outcome=outcome,
            category=category,
            level=level,
            details=details or {},
            correlation_id=correlation_id,
            session_id=self._session_id,
            previous_hash=self._last_hash if self.enable_hash_chain else "",
        )

        # Calculate hash
        entry.entry_hash = entry.calculate_hash()

        # Update last hash for chain
        if self.enable_hash_chain:
            self._last_hash = entry.entry_hash

        self._entries.append(entry)

        # Auto-persist
        if self.auto_persist:
            self._persist_entry(entry)

        return entry

    def audit(
        self,
        action: str,
        resource: str,
        outcome: str,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """
        Log an audit trail entry for compliance.

        Implements AbstractLogger.audit() interface.
        This is an alias for the log() method with simplified parameters.

        Args:
            action: What action was performed
            resource: What resource was affected
            outcome: Result (SUCCESS, FAILED, etc.)
            actor: Who/what performed the action
            details: Additional details

        Returns:
            The created AuditEntry
        """
        return self.log(
            action=action,
            resource=resource,
            outcome=outcome,
            actor=actor,
            details=details,
        )

    # ==========================================================================
    # Specialized Logging Methods
    # ==========================================================================

    def log_order(
        self,
        order_id: str,
        symbol: str,
        action: str,  # SUBMITTED, MODIFIED, CANCELLED, FILLED
        quantity: int,
        price: float,
        side: str,
        order_type: str,
        outcome: str,
        actor: str = "algorithm",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log an order-related event."""
        order_details = {
            "order_id": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "side": side,
            "order_type": order_type,
            **(details or {}),
        }

        return self.log(
            action=f"ORDER_{action}",
            resource=f"order:{order_id}",
            outcome=outcome,
            actor=actor,
            category=AuditCategory.ORDER,
            level=AuditLevel.INFO,
            details=order_details,
        )

    def log_trade(
        self,
        trade_id: str,
        symbol: str,
        quantity: int,
        price: float,
        side: str,
        commission: float = 0.0,
        actor: str = "algorithm",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log a trade execution."""
        trade_details = {
            "trade_id": trade_id,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "side": side,
            "commission": commission,
            "notional_value": quantity * price,
            **(details or {}),
        }

        return self.log(
            action="TRADE_EXECUTED",
            resource=f"trade:{trade_id}",
            outcome="SUCCESS",
            actor=actor,
            category=AuditCategory.TRADE,
            level=AuditLevel.INFO,
            details=trade_details,
        )

    def log_position_change(
        self,
        symbol: str,
        previous_quantity: int,
        new_quantity: int,
        average_price: float,
        actor: str = "algorithm",
        reason: str = "",
    ) -> AuditEntry:
        """Log a position change."""
        return self.log(
            action="POSITION_CHANGED",
            resource=f"position:{symbol}",
            outcome="SUCCESS",
            actor=actor,
            category=AuditCategory.POSITION,
            level=AuditLevel.INFO,
            details={
                "symbol": symbol,
                "previous_quantity": previous_quantity,
                "new_quantity": new_quantity,
                "change": new_quantity - previous_quantity,
                "average_price": average_price,
                "reason": reason,
            },
        )

    def log_risk_event(
        self,
        event_type: str,  # LIMIT_BREACH, CIRCUIT_BREAKER, etc.
        symbol: str = "",
        severity: str = "WARNING",
        details: dict[str, Any] | None = None,
        actor: str = "risk_manager",
    ) -> AuditEntry:
        """Log a risk management event."""
        level = AuditLevel.WARNING if severity == "WARNING" else AuditLevel.ERROR

        return self.log(
            action=f"RISK_{event_type}",
            resource=f"risk:{symbol}" if symbol else "risk:system",
            outcome="TRIGGERED",
            actor=actor,
            category=AuditCategory.RISK,
            level=level,
            details=details or {},
        )

    def log_authentication(
        self,
        user_id: str,
        action: str,  # LOGIN, LOGOUT, FAILED_LOGIN
        outcome: str,
        ip_address: str = "",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log an authentication event."""
        return self.log(
            action=f"AUTH_{action}",
            resource=f"user:{user_id}",
            outcome=outcome,
            actor=user_id,
            category=AuditCategory.AUTHENTICATION,
            level=AuditLevel.INFO if outcome == "SUCCESS" else AuditLevel.WARNING,
            details={"ip_address": ip_address, **(details or {})},
        )

    def log_configuration_change(
        self,
        setting_name: str,
        old_value: Any,
        new_value: Any,
        actor: str = "admin",
    ) -> AuditEntry:
        """Log a configuration change."""
        return self.log(
            action="CONFIG_CHANGED",
            resource=f"config:{setting_name}",
            outcome="SUCCESS",
            actor=actor,
            category=AuditCategory.CONFIGURATION,
            level=AuditLevel.WARNING,  # Config changes are notable
            details={
                "setting_name": setting_name,
                "old_value": str(old_value),
                "new_value": str(new_value),
            },
        )

    # ==========================================================================
    # Retrieval
    # ==========================================================================

    def get_entries(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        category: AuditCategory | None = None,
        level: AuditLevel | None = None,
        actor: str | None = None,
        action_pattern: str | None = None,
        limit: int = 1000,
    ) -> list[AuditEntry]:
        """
        Get filtered audit entries.

        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            category: Filter by category
            level: Filter by level
            actor: Filter by actor
            action_pattern: Filter by action containing pattern
            limit: Maximum entries to return

        Returns:
            List of matching entries
        """
        results = []

        for entry in self._entries:
            # Time filters
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue

            # Category filter
            if category and entry.category != category:
                continue

            # Level filter
            if level and entry.level != level:
                continue

            # Actor filter
            if actor and entry.actor != actor:
                continue

            # Action pattern filter
            if action_pattern and action_pattern.lower() not in entry.action.lower():
                continue

            results.append(entry)

            if len(results) >= limit:
                break

        return results

    def get_trail(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> AuditTrail:
        """Get audit trail with integrity verification."""
        entries = self.get_entries(start_time=start_time, end_time=end_time, limit=100000)

        trail = AuditTrail(
            entries=entries,
            start_time=start_time or (entries[0].timestamp if entries else datetime.utcnow()),
        )

        # Calculate trail hash
        if entries:
            trail.trail_hash = hashlib.sha256(json.dumps([e.entry_hash for e in entries]).encode()).hexdigest()

        return trail

    def get_by_correlation_id(self, correlation_id: str) -> list[AuditEntry]:
        """Get all entries with a specific correlation ID."""
        return [e for e in self._entries if e.correlation_id == correlation_id]

    # ==========================================================================
    # Persistence
    # ==========================================================================

    def _persist_entry(self, entry: AuditEntry) -> None:
        """Persist a single entry to disk."""
        date_str = entry.timestamp.strftime("%Y-%m-%d")
        log_file = self.log_dir / f"audit_{date_str}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def persist_all(self) -> int:
        """Persist all entries to disk."""
        count = 0
        for entry in self._entries:
            self._persist_entry(entry)
            count += 1
        return count

    def load_entries(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[AuditEntry]:
        """
        Load entries from disk.

        Args:
            start_date: Start date for loading
            end_date: End date for loading

        Returns:
            List of loaded entries
        """
        entries = []
        start_date = start_date or datetime.utcnow() - timedelta(days=30)
        end_date = end_date or datetime.utcnow()

        current = start_date
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            log_file = self.log_dir / f"audit_{date_str}.jsonl"

            if log_file.exists():
                with open(log_file) as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            entries.append(AuditEntry.from_dict(data))
                        except (json.JSONDecodeError, KeyError):
                            continue

            current += timedelta(days=1)

        return entries

    # ==========================================================================
    # Statistics
    # ==========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get audit log statistics."""
        if not self._entries:
            return {
                "total_entries": 0,
                "categories": {},
                "levels": {},
                "outcomes": {},
            }

        categories: dict[str, int] = {}
        levels: dict[str, int] = {}
        outcomes: dict[str, int] = {}

        for entry in self._entries:
            cat = entry.category.value
            categories[cat] = categories.get(cat, 0) + 1

            lvl = entry.level.value
            levels[lvl] = levels.get(lvl, 0) + 1

            outcomes[entry.outcome] = outcomes.get(entry.outcome, 0) + 1

        return {
            "total_entries": len(self._entries),
            "categories": categories,
            "levels": levels,
            "outcomes": outcomes,
            "first_entry": self._entries[0].timestamp.isoformat() if self._entries else None,
            "last_entry": self._entries[-1].timestamp.isoformat() if self._entries else None,
            "hash_chain_enabled": self.enable_hash_chain,
        }

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """Verify integrity of all entries."""
        trail = self.get_trail()
        return trail.verify_integrity()


def create_audit_logger(
    log_dir: str | Path | None = None,
    retention_days: int = 2555,
    enable_hash_chain: bool = True,
    auto_persist: bool = True,
) -> AuditLogger:
    """
    Factory function to create an audit logger.

    Args:
        log_dir: Directory for log files
        retention_days: Days to retain logs
        enable_hash_chain: Enable hash chain
        auto_persist: Auto-persist entries

    Returns:
        Configured AuditLogger
    """
    return AuditLogger(
        log_dir=Path(log_dir) if log_dir else None,
        retention_days=retention_days,
        enable_hash_chain=enable_hash_chain,
        auto_persist=auto_persist,
    )
