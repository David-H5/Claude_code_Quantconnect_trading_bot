# Compliance & Audit Logging Module

UPGRADE-015 Phase 11: Regulatory Compliance for Trading Systems

## Overview

This module provides comprehensive compliance and audit logging capabilities:
- Immutable audit trails with hash chain integrity
- Market manipulation detection
- Compliance reporting
- Data retention policy management
- FINRA checklist validation

## Components

### Audit Logger (`audit_logger.py`)

Comprehensive audit logging with SOX/FINRA compliance:

```python
from compliance import AuditLogger, create_audit_logger

# Create logger
logger = create_audit_logger(
    log_dir="audit_logs",
    retention_days=2555,  # 7 years for SOX
    enable_hash_chain=True,
)

# Log trading activity
logger.log_order(
    order_id="ORD-001",
    symbol="SPY",
    action="SUBMITTED",
    quantity=100,
    price=450.0,
    side="buy",
    order_type="limit",
    outcome="SUCCESS",
)

logger.log_trade(
    trade_id="TRD-001",
    symbol="SPY",
    quantity=100,
    price=450.05,
    side="buy",
    commission=1.50,
)

# Log risk events
logger.log_risk_event(
    event_type="LIMIT_BREACH",
    symbol="SPY",
    severity="WARNING",
    details={"limit": 0.25, "actual": 0.30},
)

# Verify audit trail integrity
is_valid, issues = logger.verify_integrity()
print(f"Integrity: {'VALID' if is_valid else 'INVALID'}")
```

**Features:**
- Hash chain for tamper detection
- Multiple log categories (order, trade, risk, auth)
- Time-range filtering
- Correlation ID tracking
- Automatic persistence

### Anti-Manipulation Monitor (`anti_manipulation.py`)

Detects market manipulation patterns:

```python
from compliance import (
    AntiManipulationMonitor,
    OrderEvent,
    create_anti_manipulation_monitor,
)

# Create monitor
monitor = create_anti_manipulation_monitor(
    spoofing_cancel_threshold=0.90,
    layering_min_levels=3,
)

# Process order events
event = OrderEvent(
    timestamp=datetime.utcnow(),
    order_id="ORD-001",
    symbol="SPY",
    side="buy",
    quantity=100,
    price=450.0,
    event_type="submitted",
)

alerts = monitor.process_order_event(event)

for alert in alerts:
    print(f"ALERT: {alert.manipulation_type.value}")
    print(f"  Severity: {alert.severity.value}")
    print(f"  Confidence: {alert.confidence:.0%}")
    print(f"  Action: {alert.recommended_action}")
```

**Detectable Patterns:**

| Pattern | Description | Indicators |
|---------|-------------|------------|
| Spoofing | Orders intended to cancel | High cancel rate, rapid submission/cancel |
| Layering | Multiple orders at increments | Consistent price steps, many levels |
| Wash Trading | Trading with self | Same account buy/sell, similar prices |
| Momentum Ignition | Trigger then reverse | Volume spike, direction reversal |
| Quote Stuffing | Excessive updates | High message rate, high cancel rate |

### Compliance Reporting (`reporting.py`)

Generate regulatory reports:

```python
from compliance import (
    ComplianceReporter,
    ReportFormat,
    ReportPeriod,
    create_compliance_reporter,
)

# Create reporter
reporter = create_compliance_reporter(
    output_dir="compliance_reports",
    default_format=ReportFormat.HTML,
)

# Generate trading summary
report = reporter.generate_trading_summary(
    trades=trade_list,
    orders=order_list,
    period=ReportPeriod.DAILY,
)

# Generate risk exposure report
risk_report = reporter.generate_risk_exposure(
    positions=positions,
    portfolio_value=100000,
    risk_metrics={"var_95": 5000, "sharpe": 1.5},
)

# Generate audit trail report
audit_report = reporter.generate_audit_trail(
    audit_entries=logger.get_entries(),
    period=ReportPeriod.WEEKLY,
)

# Export reports
reporter.export_report(report, ReportFormat.HTML)
reporter.export_report(report, ReportFormat.JSON)
reporter.export_report(report, ReportFormat.CSV)
```

**Report Types:**
- Trading Summary
- Risk Exposure
- Audit Trail
- Exception Report
- Manipulation Alerts

### Retention Policy (`retention_policy.py`)

Manage data retention for compliance:

```python
from compliance import (
    RetentionPolicyManager,
    DataCategory,
    create_retention_manager,
)

# Create manager (loads default regulatory rules)
manager = create_retention_manager(policy_file="retention_policy.json")

# Check if data can be deleted
result = manager.check_retention(
    category=DataCategory.TRADING_RECORDS,
    data_date=datetime(2020, 1, 1),
    symbol="SPY",
)

print(f"Expired: {result.is_expired}")
print(f"Under Hold: {result.under_legal_hold}")
print(f"Can Delete: {result.can_delete}")
print(f"Days Remaining: {result.days_remaining}")

# Create legal hold
hold = manager.create_legal_hold(
    description="SEC Investigation - Case 12345",
    created_by="legal@company.com",
    categories=[DataCategory.TRADING_RECORDS, DataCategory.COMMUNICATIONS],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
)

# Release hold when resolved
manager.release_legal_hold(hold.hold_id, released_by="legal@company.com")
```

**Default Retention Periods:**

| Category | Retention | Regulatory Basis |
|----------|-----------|------------------|
| Trading Records | 7 years | SOX, FINRA 4511 |
| Audit Logs | 7 years | SOX Section 802 |
| Communications | 6 years | FINRA 4511, SEC 17a-4 |
| Account Records | 6 years | FINRA 4511 |
| Risk Data | 5 years | Internal Policy |
| Market Data | 1 year | Re-obtainable |
| Configuration | 7 years | SOX Audit |
| Temporary | 7 days | N/A |

### FINRA Checklist (`finra_checklist.py`)

Validate compliance with FINRA requirements:

```python
from compliance import FINRAChecklist, create_finra_checklist, ChecklistStatus

# Create checklist with automated checks
checklist = create_finra_checklist()

# Run compliance check
context = {
    "audit_logger": audit_logger,
    "circuit_breaker": circuit_breaker,
    "position_limits": {"max_position_pct": 0.25},
    "manipulation_monitor": manipulation_monitor,
}

result = checklist.run_checklist(context)

print(f"Compliance Score: {result.compliance_score:.0%}")
print(f"Passed: {result.passed}/{result.total_items}")
print(f"Failed: {result.failed}")
print(f"Critical Failures: {result.critical_failures}")

# Get failed items
for item in checklist.get_failed_items():
    print(f"  FAILED: {item.item_id} - {item.name}")
    print(f"    Rule: {item.rule_reference.value}")
    print(f"    Message: {item.result_message}")

# Manual check
checklist.check_item(
    item_id="SUP-001",
    status=ChecklistStatus.PASSED,
    message="WSP document reviewed and approved",
    evidence={"document_id": "WSP-2025-001"},
)
```

**Checklist Categories:**

| Category | Rule Reference | Items |
|----------|---------------|-------|
| Supervision | FINRA 3110 | Supervisory system, designated supervisor, reviews |
| Records | FINRA 4511, SEC 17a-4 | Order records, trade records, retention, audit trail |
| Best Execution | FINRA 5310 | Policy, monitoring, venue evaluation |
| Trading Practices | FINRA 6140 | Pre-trade controls, kill switch, error prevention |
| Manipulation | SEC 10b-5 | Anti-spoofing, anti-layering, wash trade prevention |
| Risk Management | FINRA 3110 | Position limits, loss limits, circuit breaker |
| Reporting | Various | Trade reporting, exception reporting |

## Integration Example

Complete compliance integration:

```python
from compliance import (
    create_audit_logger,
    create_anti_manipulation_monitor,
    create_compliance_reporter,
    create_retention_manager,
    create_finra_checklist,
    DataCategory,
    ReportPeriod,
)

# Initialize all compliance components
audit_logger = create_audit_logger(
    log_dir="audit_logs",
    enable_hash_chain=True,
)

manipulation_monitor = create_anti_manipulation_monitor(
    spoofing_cancel_threshold=0.90,
)

reporter = create_compliance_reporter(
    output_dir="compliance_reports",
)

retention_manager = create_retention_manager(
    policy_file="retention_policy.json",
)

finra_checklist = create_finra_checklist()

# Trading loop with compliance
def on_order_submitted(order):
    # Log to audit trail
    audit_logger.log_order(
        order_id=order.id,
        symbol=order.symbol,
        action="SUBMITTED",
        quantity=order.quantity,
        price=order.price,
        side=order.side,
        order_type=order.type,
        outcome="SUCCESS",
    )

    # Check for manipulation
    alerts = manipulation_monitor.process_order_event(
        OrderEvent(
            timestamp=datetime.utcnow(),
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.price,
            event_type="submitted",
        )
    )

    if alerts:
        for alert in alerts:
            audit_logger.log_risk_event(
                event_type=f"MANIPULATION_{alert.manipulation_type.value.upper()}",
                symbol=alert.symbol,
                severity="WARNING",
                details=alert.evidence,
            )

# Daily compliance routine
def daily_compliance_check():
    # Verify audit integrity
    is_valid, issues = audit_logger.verify_integrity()
    if not is_valid:
        raise ComplianceError(f"Audit integrity failed: {issues}")

    # Generate reports
    reporter.generate_trading_summary(
        trades=get_daily_trades(),
        orders=get_daily_orders(),
        period=ReportPeriod.DAILY,
    )

    reporter.generate_manipulation_report(
        alerts=[a.to_dict() for a in manipulation_monitor.get_alerts()],
        period=ReportPeriod.DAILY,
    )

    # Run FINRA checklist
    result = finra_checklist.run_checklist({
        "audit_logger": audit_logger,
        "manipulation_monitor": manipulation_monitor,
    })

    if result.critical_failures:
        raise ComplianceError(f"FINRA failures: {result.critical_failures}")

    # Check retention
    check_data_retention()
```

## Testing

Run the test suite:

```bash
# All compliance tests
pytest tests/compliance/ -v

# Specific tests
pytest tests/compliance/test_audit_logger.py -v
pytest tests/compliance/test_anti_manipulation.py -v
```

## Regulatory References

| Regulation | Description | Key Requirements |
|------------|-------------|------------------|
| SOX Section 802 | Record retention | 7 years, tampering prohibition |
| FINRA Rule 3110 | Supervision | Written procedures, designated supervisor |
| FINRA Rule 4511 | Books and Records | Complete records, 6 year retention |
| FINRA Rule 5310 | Best Execution | Regular review, reasonable diligence |
| FINRA Rule 6140 | Trading Practices | Pre-trade controls, erroneous trade prevention |
| SEC Rule 10b-5 | Fraud/Manipulation | Prohibition on market manipulation |
| SEC Rule 17a-4 | Records Preservation | Electronic storage requirements |

## Dependencies

- Python 3.10+
- dataclasses (built-in)
- datetime (built-in)
- hashlib (built-in)
- json (built-in)
- pathlib (built-in)

## Related Modules

- `agents/` - Multi-agent architecture
- `observability/` - Metrics and tracing
- `models/circuit_breaker.py` - Trading halt functionality
- `execution/pre_trade_validator.py` - Pre-trade risk checks
