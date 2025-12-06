"""
Audit Logger Module

DEPRECATED: This module has been moved to observability.logging.audit.
This file provides backwards-compatible re-exports.

Use the new import path:
    from observability.logging import AuditLogger, create_audit_logger

Original: UPGRADE-015 Phase 11: Compliance and Audit Logging
Refactored: Phase 2 - Consolidated Logging Infrastructure
"""

# Re-export everything from new location for backwards compatibility
from observability.logging.audit import (
    AuditCategory,
    # Data classes
    AuditEntry,
    # Enums
    AuditLevel,
    # Main class
    AuditLogger,
    AuditTrail,
    # Factory function
    create_audit_logger,
)


__all__ = [
    "AuditCategory",
    "AuditEntry",
    "AuditLevel",
    "AuditLogger",
    "AuditTrail",
    "create_audit_logger",
]
