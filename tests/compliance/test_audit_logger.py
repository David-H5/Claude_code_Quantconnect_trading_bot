"""
Tests for Audit Logger Module

UPGRADE-015 Phase 11: Compliance and Audit Logging

Tests cover:
- Audit entry creation
- Hash chain integrity
- Specialized logging methods
- Entry retrieval and filtering
- Audit trail verification
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compliance.audit_logger import (
    AuditCategory,
    AuditEntry,
    AuditLevel,
    AuditLogger,
    AuditTrail,
    create_audit_logger,
)


class TestAuditEntry:
    """Test AuditEntry dataclass."""

    def test_entry_creation(self):
        """Test creating an audit entry."""
        entry = AuditEntry(
            timestamp=datetime.utcnow(),
            actor="test_user",
            action="TEST_ACTION",
            resource="test_resource",
            outcome="SUCCESS",
        )

        assert entry.actor == "test_user"
        assert entry.action == "TEST_ACTION"
        assert entry.outcome == "SUCCESS"

    def test_entry_with_details(self):
        """Test entry with additional details."""
        entry = AuditEntry(
            timestamp=datetime.utcnow(),
            actor="algo",
            action="ORDER_SUBMITTED",
            resource="order:123",
            outcome="SUCCESS",
            category=AuditCategory.ORDER,
            level=AuditLevel.INFO,
            details={"symbol": "SPY", "quantity": 100},
        )

        assert entry.category == AuditCategory.ORDER
        assert entry.details["symbol"] == "SPY"

    def test_entry_to_dict(self):
        """Test converting entry to dictionary."""
        entry = AuditEntry(
            timestamp=datetime.utcnow(),
            actor="test",
            action="TEST",
            resource="res",
            outcome="SUCCESS",
            category=AuditCategory.SYSTEM,
        )

        data = entry.to_dict()

        assert "timestamp" in data
        assert data["actor"] == "test"
        assert data["category"] == "system"

    def test_entry_from_dict(self):
        """Test creating entry from dictionary."""
        data = {
            "timestamp": "2025-01-01T12:00:00",
            "actor": "test",
            "action": "TEST",
            "resource": "res",
            "outcome": "SUCCESS",
            "category": "order",
            "level": "warning",
        }

        entry = AuditEntry.from_dict(data)

        assert entry.actor == "test"
        assert entry.category == AuditCategory.ORDER
        assert entry.level == AuditLevel.WARNING

    def test_entry_hash_calculation(self):
        """Test hash calculation."""
        entry = AuditEntry(
            timestamp=datetime.utcnow(),
            actor="test",
            action="TEST",
            resource="res",
            outcome="SUCCESS",
        )

        hash1 = entry.calculate_hash()
        hash2 = entry.calculate_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length


class TestAuditTrail:
    """Test AuditTrail class."""

    def test_empty_trail_verification(self):
        """Test verifying empty trail."""
        trail = AuditTrail()

        is_valid, issues = trail.verify_integrity()

        assert is_valid is True
        assert len(issues) == 0

    def test_trail_to_dict(self):
        """Test converting trail to dictionary."""
        trail = AuditTrail()

        data = trail.to_dict()

        assert "entries" in data
        assert "entry_count" in data


class TestAuditLogger:
    """Test AuditLogger class."""

    def test_initialization(self):
        """Test logger initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(
                log_dir=Path(tmpdir),
                enable_hash_chain=True,
                auto_persist=False,
            )

            assert logger.enable_hash_chain is True
            assert len(logger._entries) == 0

    def test_basic_logging(self):
        """Test basic log entry."""
        logger = AuditLogger(auto_persist=False)

        entry = logger.log(
            action="TEST_ACTION",
            resource="test_resource",
            outcome="SUCCESS",
        )

        assert entry.action == "TEST_ACTION"
        assert entry.outcome == "SUCCESS"
        assert len(logger._entries) == 1

    def test_logging_with_all_fields(self):
        """Test logging with all fields populated."""
        logger = AuditLogger(auto_persist=False)

        entry = logger.log(
            action="FULL_TEST",
            resource="res",
            outcome="SUCCESS",
            actor="test_user",
            category=AuditCategory.TRADE,
            level=AuditLevel.WARNING,
            details={"key": "value"},
            correlation_id="corr-123",
        )

        assert entry.actor == "test_user"
        assert entry.category == AuditCategory.TRADE
        assert entry.level == AuditLevel.WARNING
        assert entry.details["key"] == "value"
        assert entry.correlation_id == "corr-123"

    def test_hash_chain(self):
        """Test hash chain integrity."""
        logger = AuditLogger(enable_hash_chain=True, auto_persist=False)

        # Log multiple entries
        for i in range(5):
            logger.log(action=f"ACTION_{i}", resource="res", outcome="SUCCESS")

        # Verify chain
        for i in range(1, len(logger._entries)):
            assert logger._entries[i].previous_hash == logger._entries[i - 1].entry_hash

    def test_log_order(self):
        """Test order logging."""
        logger = AuditLogger(auto_persist=False)

        entry = logger.log_order(
            order_id="ORD-001",
            symbol="SPY",
            action="SUBMITTED",
            quantity=100,
            price=450.0,
            side="buy",
            order_type="limit",
            outcome="SUCCESS",
        )

        assert entry.action == "ORDER_SUBMITTED"
        assert entry.category == AuditCategory.ORDER
        assert entry.details["symbol"] == "SPY"
        assert entry.details["quantity"] == 100

    def test_log_trade(self):
        """Test trade logging."""
        logger = AuditLogger(auto_persist=False)

        entry = logger.log_trade(
            trade_id="TRD-001",
            symbol="AAPL",
            quantity=50,
            price=175.0,
            side="sell",
            commission=1.50,
        )

        assert entry.action == "TRADE_EXECUTED"
        assert entry.category == AuditCategory.TRADE
        assert entry.details["notional_value"] == 50 * 175.0

    def test_log_position_change(self):
        """Test position change logging."""
        logger = AuditLogger(auto_persist=False)

        entry = logger.log_position_change(
            symbol="SPY",
            previous_quantity=0,
            new_quantity=100,
            average_price=450.0,
            reason="Signal entry",
        )

        assert entry.action == "POSITION_CHANGED"
        assert entry.category == AuditCategory.POSITION
        assert entry.details["change"] == 100

    def test_log_risk_event(self):
        """Test risk event logging."""
        logger = AuditLogger(auto_persist=False)

        entry = logger.log_risk_event(
            event_type="LIMIT_BREACH",
            symbol="SPY",
            severity="WARNING",
            details={"limit": 0.25, "actual": 0.30},
        )

        assert entry.action == "RISK_LIMIT_BREACH"
        assert entry.category == AuditCategory.RISK
        assert entry.level == AuditLevel.WARNING

    def test_log_authentication(self):
        """Test authentication logging."""
        logger = AuditLogger(auto_persist=False)

        entry = logger.log_authentication(
            user_id="trader@example.com",
            action="LOGIN",
            outcome="SUCCESS",
            ip_address="192.168.1.1",
        )

        assert entry.action == "AUTH_LOGIN"
        assert entry.category == AuditCategory.AUTHENTICATION
        assert entry.details["ip_address"] == "192.168.1.1"

    def test_log_configuration_change(self):
        """Test configuration change logging."""
        logger = AuditLogger(auto_persist=False)

        entry = logger.log_configuration_change(
            setting_name="max_position_size",
            old_value=0.20,
            new_value=0.25,
            actor="admin",
        )

        assert entry.action == "CONFIG_CHANGED"
        assert entry.category == AuditCategory.CONFIGURATION
        assert entry.level == AuditLevel.WARNING

    def test_get_entries_no_filter(self):
        """Test getting all entries."""
        logger = AuditLogger(auto_persist=False)

        for i in range(10):
            logger.log(action=f"ACTION_{i}", resource="res", outcome="SUCCESS")

        entries = logger.get_entries()

        assert len(entries) == 10

    def test_get_entries_by_category(self):
        """Test filtering by category."""
        logger = AuditLogger(auto_persist=False)

        logger.log_order("O1", "SPY", "SUBMITTED", 100, 450, "buy", "limit", "SUCCESS")
        logger.log_trade("T1", "SPY", 100, 450, "buy")
        logger.log_order("O2", "AAPL", "CANCELLED", 50, 175, "sell", "limit", "SUCCESS")

        orders = logger.get_entries(category=AuditCategory.ORDER)
        trades = logger.get_entries(category=AuditCategory.TRADE)

        assert len(orders) == 2
        assert len(trades) == 1

    def test_get_entries_by_time_range(self):
        """Test filtering by time range."""
        logger = AuditLogger(auto_persist=False)

        # Add entries
        now = datetime.utcnow()
        logger.log(action="ACTION_1", resource="res", outcome="SUCCESS")

        entries = logger.get_entries(
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )

        assert len(entries) >= 1

    def test_get_entries_by_action_pattern(self):
        """Test filtering by action pattern."""
        logger = AuditLogger(auto_persist=False)

        logger.log(action="ORDER_SUBMITTED", resource="o1", outcome="SUCCESS")
        logger.log(action="ORDER_CANCELLED", resource="o2", outcome="SUCCESS")
        logger.log(action="TRADE_EXECUTED", resource="t1", outcome="SUCCESS")

        order_entries = logger.get_entries(action_pattern="ORDER")

        assert len(order_entries) == 2

    def test_get_by_correlation_id(self):
        """Test getting entries by correlation ID."""
        logger = AuditLogger(auto_persist=False)

        corr_id = "test-correlation-123"
        logger.log(action="A1", resource="r1", outcome="SUCCESS", correlation_id=corr_id)
        logger.log(action="A2", resource="r2", outcome="SUCCESS", correlation_id=corr_id)
        logger.log(action="A3", resource="r3", outcome="SUCCESS", correlation_id="other")

        entries = logger.get_by_correlation_id(corr_id)

        assert len(entries) == 2

    def test_get_trail(self):
        """Test getting audit trail."""
        logger = AuditLogger(enable_hash_chain=True, auto_persist=False)

        for i in range(5):
            logger.log(action=f"ACTION_{i}", resource="res", outcome="SUCCESS")

        trail = logger.get_trail()

        assert len(trail.entries) == 5
        assert trail.trail_hash != ""

    def test_trail_integrity_verification(self):
        """Test audit trail integrity verification."""
        logger = AuditLogger(enable_hash_chain=True, auto_persist=False)

        for i in range(5):
            logger.log(action=f"ACTION_{i}", resource="res", outcome="SUCCESS")

        is_valid, issues = logger.verify_integrity()

        assert is_valid is True
        assert len(issues) == 0

    def test_get_stats(self):
        """Test getting statistics."""
        logger = AuditLogger(auto_persist=False)

        logger.log_order("O1", "SPY", "SUBMITTED", 100, 450, "buy", "limit", "SUCCESS")
        logger.log_trade("T1", "SPY", 100, 450, "buy")
        logger.log_risk_event("ALERT", "SPY", "WARNING")

        stats = logger.get_stats()

        assert stats["total_entries"] == 3
        assert "order" in stats["categories"]
        assert "trade" in stats["categories"]


class TestCreateAuditLogger:
    """Test factory function."""

    def test_create_with_defaults(self):
        """Test creating logger with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = create_audit_logger(log_dir=tmpdir, auto_persist=False)

            assert logger.enable_hash_chain is True
            assert logger.retention_days == 2555

    def test_create_with_custom_config(self):
        """Test creating logger with custom config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = create_audit_logger(
                log_dir=tmpdir,
                retention_days=365,
                enable_hash_chain=False,
                auto_persist=False,
            )

            assert logger.retention_days == 365
            assert logger.enable_hash_chain is False


class TestPersistence:
    """Test persistence functionality."""

    def test_persist_and_load(self):
        """Test persisting and loading entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and log
            logger1 = AuditLogger(log_dir=Path(tmpdir), auto_persist=True)

            logger1.log(action="TEST_ACTION", resource="res", outcome="SUCCESS")
            logger1.log(action="ANOTHER_ACTION", resource="res2", outcome="FAILED")

            # Load in new logger
            logger2 = AuditLogger(log_dir=Path(tmpdir), auto_persist=False)
            loaded = logger2.load_entries()

            assert len(loaded) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
