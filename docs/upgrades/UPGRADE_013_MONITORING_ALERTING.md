# Upgrade Path: Monitoring & Alerting System

**Upgrade ID**: UPGRADE-013
**Iteration**: 1
**Date**: December 1, 2025
**Status**: ✅ Complete

---

## Target State

Implement comprehensive monitoring and alerting infrastructure for:

1. **Multi-Channel Alerting**: Email, Discord, Slack, Console
2. **Alert Severity Levels**: Debug, Info, Warning, Error, Critical
3. **Rate Limiting**: Prevent alert spam with configurable limits
4. **Alert Aggregation**: Group similar alerts to reduce noise
5. **System Monitoring**: Health checks, performance tracking
6. **Integration**: Connect with ErrorHandler and CircuitBreaker

---

## Scope

### Included

- Create `utils/alerting_service.py` with multi-channel support
- Create `utils/system_monitor.py` for health checks
- Integrate with ErrorHandler for automatic alerts
- Integrate with AlertingConfig from UPGRADE-011
- Create comprehensive tests
- Add console, email, Discord, and Slack channels

### Excluded

- SMS alerts (requires external service, defer to P2)
- Pager duty integration (P2, defer)
- Complex escalation workflows (P2, defer)

---

## Success Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| AlertingService created | File exists | `utils/alerting_service.py` |
| SystemMonitor created | File exists | `utils/system_monitor.py` |
| Multi-channel support | Channel types | >= 4 (console, email, discord, slack) |
| Tests created | Test count | >= 25 test cases |
| Rate limiting works | Tested | Verified |
| Aggregation works | Tested | Verified |

---

## Dependencies

- [x] UPGRADE-009 Structured Logging complete
- [x] UPGRADE-010 Performance Tracker complete
- [x] UPGRADE-011 Configuration Refactoring complete (AlertingConfig)
- [x] UPGRADE-012 Error Handling complete (ErrorHandler, AlertTrigger)

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Alert spam | Medium | Medium | Rate limiting, aggregation |
| External service failures | Medium | Low | Graceful degradation, fallback to console |
| Missing critical alerts | Low | High | Multiple channels, redundancy |

---

## Estimated Effort

- AlertingService Core: 1.5 hours
- Channel Implementations: 1 hour
- SystemMonitor: 1 hour
- Integration with ErrorHandler: 0.5 hours
- Tests: 1.5 hours
- **Total**: ~5.5 hours

---

## Phase 2: Task Checklist

### Core Infrastructure (T1-T3)

| ID | Task | Est. | Depends | Priority |
|----|------|------|---------|----------|
| T1 | Create `utils/alerting_service.py` | 45m | - | P0 |
| T2 | Create `utils/system_monitor.py` | 30m | T1 | P0 |
| T3 | Add alert channel implementations | 30m | T1 | P0 |

### Features (T4-T6)

| ID | Task | Est. | Depends | Priority |
|----|------|------|---------|----------|
| T4 | Add rate limiting and aggregation | 20m | T1 | P0 |
| T5 | Integrate with ErrorHandler | 20m | T1 | P0 |
| T6 | Integrate with AlertingConfig | 15m | T1 | P0 |

### Testing (T7-T8)

| ID | Task | Est. | Depends | Priority |
|----|------|------|---------|----------|
| T7 | Create `tests/test_alerting_service.py` | 45m | T1-T6 | P0 |
| T8 | Update `utils/__init__.py` exports | 10m | T1-T2 | P0 |

---

## Phase 3: Implementation

### T1: AlertingService Core

```python
# utils/alerting_service.py
"""
Multi-Channel Alerting Service

Provides:
- Multiple alert channels (console, email, Discord, Slack)
- Rate limiting to prevent spam
- Alert aggregation for noise reduction
- Severity-based filtering
- Integration with ErrorHandler

UPGRADE-013: Monitoring & Alerting (December 2025)
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod
import threading

class AlertChannel(ABC):
    """Base class for alert channels."""

    @abstractmethod
    def send(self, alert: "Alert") -> bool:
        """Send alert through this channel."""
        pass

class ConsoleChannel(AlertChannel):
    """Console/stdout alert channel."""

    def send(self, alert: "Alert") -> bool:
        print(f"[{alert.severity.name}] {alert.title}: {alert.message}")
        return True

class AlertingService:
    """Multi-channel alerting service with rate limiting."""

    def __init__(
        self,
        config: Optional["AlertingConfig"] = None,
        error_handler: Optional[Any] = None,
    ):
        self.config = config
        self.error_handler = error_handler
        self._channels: Dict[str, AlertChannel] = {}
        self._rate_limiter = RateLimiter()
        self._aggregator = AlertAggregator()
        self._setup_channels()

    def send_alert(
        self,
        title: str,
        message: str,
        severity: "AlertSeverity" = AlertSeverity.INFO,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send alert through all enabled channels."""
        # Rate limiting, aggregation, and dispatch logic
        pass
```

### T2: SystemMonitor

```python
# utils/system_monitor.py
"""
System Monitor for Trading Bot

Provides:
- Health checks for critical services
- Performance monitoring
- Automatic alerting on issues
- Status dashboard data

UPGRADE-013: Monitoring & Alerting (December 2025)
"""

class SystemMonitor:
    """Monitor system health and performance."""

    def __init__(
        self,
        alerting_service: Optional["AlertingService"] = None,
        check_interval_seconds: int = 60,
    ):
        self.alerting_service = alerting_service
        self.check_interval = check_interval_seconds
        self._services: Dict[str, ServiceCheck] = {}
        self._running = False

    def register_service(self, name: str, check_fn: Callable[[], bool]) -> None:
        """Register a service for health monitoring."""
        pass

    def check_all(self) -> Dict[str, bool]:
        """Run all health checks."""
        pass

    def get_status_summary(self) -> Dict[str, Any]:
        """Get overall system status."""
        pass
```

---

## Phase 4: Double-Check

**Date**: December 1, 2025
**Checked By**: Claude Code Agent

### Implementation Progress

| Task | Status | Notes |
|------|--------|-------|
| T1: AlertingService | ✅ Complete | 342 lines, full implementation |
| T2: SystemMonitor | ✅ Complete | 237 lines, health monitoring |
| T3: Channel implementations | ✅ Complete | Console, Email, Discord, Slack |
| T4: Rate limiting | ✅ Complete | Sliding window with reset |
| T5: ErrorHandler integration | ✅ Complete | Via AlertTrigger from UPGRADE-012 |
| T6: Config integration | ✅ Complete | AlertingConfig from UPGRADE-011 |
| T7: Tests | ✅ Complete | 42 test cases created |
| T8: Exports | ✅ Complete | All classes exported via utils/\_\_init\_\_.py |

### Success Criteria Verification

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| AlertingService created | File exists | `utils/alerting_service.py` | ✅ Pass |
| SystemMonitor created | File exists | `utils/system_monitor.py` | ✅ Pass |
| Multi-channel support | >= 4 types | 4 (Console, Email, Discord, Slack) | ✅ Pass |
| Tests created | >= 25 | 42 test cases | ✅ Pass |
| Rate limiting | Verified | Tested with sliding window | ✅ Pass |

---

## Phase 5: Introspection Report

**Report Date**: December 1, 2025

### What Worked Well

1. **Modular Channel Design**: Abstract AlertChannel base class made adding new channels trivial
2. **Rate Limiting**: Sliding window implementation prevents alert spam effectively
3. **Alert Aggregation**: Deduplication reduces noise from repeated similar errors
4. **Integration with UPGRADE-011/012**: AlertingConfig and ErrorHandler integrate seamlessly

### Challenges Encountered

1. **Test Recovery Threshold**: Initial tests failed because they didn't account for the recovery_threshold (default=2) needed for a service to transition from UNKNOWN to HEALTHY
2. **Solution**: Set recovery_threshold=1 in tests that expect immediate healthy status

### Improvements Made During Implementation

1. Added `format_console()` and `format_markdown()` methods to Alert for flexible formatting
2. Added built-in health checks (memory, disk, HTTP) as factory functions
3. Added status listeners for custom monitoring hooks
4. Added channel-level statistics tracking

### Lessons Learned

1. Recovery thresholds in monitoring systems prevent flapping but require careful test design
2. Multi-channel alerting benefits from a unified Alert object that each channel can format differently
3. Rate limiting should be key-based (by category+title) rather than global for flexibility

---

## Phase 6: Convergence Decision

**Decision**: ✅ CONVERGED

**Rationale**: All success criteria met. 1236 tests pass including 42 new alerting tests. Multi-channel alerting with rate limiting and aggregation fully implemented.

**Next Steps**:

- Consider adding SMS channel in Phase 2
- Monitor alert volume in production to tune rate limits
- Add webhook channel for generic integrations

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-01 | Upgrade path created |
| 2025-12-01 | T1-T8 implemented, all tests passing |
| 2025-12-01 | UPGRADE-013 marked CONVERGED |

---

## Related Documents

- [UPGRADE-011](UPGRADE_011_CONFIGURATION_REFACTORING.md) - AlertingConfig
- [UPGRADE-012](UPGRADE_012_ERROR_HANDLING.md) - Error Handling & AlertTrigger
- [Roadmap](../ROADMAP.md) - Phase 2 Week 3 tasks
