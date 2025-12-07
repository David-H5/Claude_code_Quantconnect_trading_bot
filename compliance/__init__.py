"""
Compliance & Audit Logging Module

Layer: 2 (Core Models)
May import from: Layers 0-1 (utils, observability, infrastructure, config)
May be imported by: Layers 3-4

UPGRADE-015 Phase 11: Compliance and Audit Logging

Provides regulatory compliance and audit logging for trading:
- Comprehensive audit trail
- Anti-manipulation detection
- Compliance reporting
- Data retention policies
- FINRA checklist validation

Features:
- Immutable audit logs
- Pattern detection for market manipulation
- Automated compliance reports
- Configurable retention policies
"""

from compliance.anti_manipulation import (
    AntiManipulationMonitor,
    ManipulationAlert,
    ManipulationType,
    create_anti_manipulation_monitor,
)
from observability.logging.audit import (
    AuditEntry,
    AuditLevel,
    AuditLogger,
    AuditTrail,
    create_audit_logger,
)
from compliance.finra_checklist import (
    ChecklistItem,
    ChecklistResult,
    ChecklistStatus,
    FINRAChecklist,
    create_finra_checklist,
)
from compliance.reporting import (
    ComplianceReport,
    ComplianceReporter,
    ReportFormat,
    ReportPeriod,
    create_compliance_reporter,
)
from compliance.retention_policy import (
    RetentionPolicy,
    RetentionPolicyManager,
    RetentionRule,
    create_retention_manager,
)


__all__ = [
    # Audit Logger
    "AuditEntry",
    "AuditLevel",
    "AuditLogger",
    "AuditTrail",
    "create_audit_logger",
    # Anti-Manipulation
    "AntiManipulationMonitor",
    "ManipulationAlert",
    "ManipulationType",
    "create_anti_manipulation_monitor",
    # Reporting
    "ComplianceReport",
    "ComplianceReporter",
    "ReportFormat",
    "ReportPeriod",
    "create_compliance_reporter",
    # Retention Policy
    "RetentionPolicy",
    "RetentionPolicyManager",
    "RetentionRule",
    "create_retention_manager",
    # FINRA Checklist
    "ChecklistItem",
    "ChecklistResult",
    "ChecklistStatus",
    "FINRAChecklist",
    "create_finra_checklist",
]
