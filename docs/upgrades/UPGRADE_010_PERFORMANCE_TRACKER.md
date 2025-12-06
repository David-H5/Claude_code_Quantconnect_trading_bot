# Upgrade Path: Unified Performance Tracker

**Upgrade ID**: UPGRADE-010
**Iteration**: 1
**Date**: December 1, 2025
**Status**: ✅ Complete

---

## Target State

Create a unified performance tracker that aggregates all trading metrics and provides:

1. **Real-Time Tracking**: Live P&L, drawdown, and position tracking
2. **Strategy Comparison**: Compare performance across different strategies
3. **Session Management**: Track performance by session/day/week
4. **Persistence**: Save metrics to Object Store
5. **Logging Integration**: Connect with structured logging

---

## Scope

### Included

- Create `models/performance_tracker.py` with unified tracker
- Integrate with existing metrics (advanced_trading_metrics, execution_quality)
- Add session-based tracking (daily, weekly, monthly)
- Add strategy-level performance breakdown
- Connect with StructuredLogger for event tracking
- Add Object Store persistence for metrics
- Create comprehensive tests

### Excluded

- UI Dashboard updates (separate upgrade)
- External analytics services (defer)
- Historical data import (defer)

---

## Success Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Performance tracker created | File exists | `models/performance_tracker.py` |
| Real-time metrics | P&L, drawdown live | Working |
| Strategy breakdown | Per-strategy stats | Working |
| Session tracking | Daily/weekly metrics | Working |
| Tests created | Test count | >= 25 test cases |
| Integration with logger | Events logged | Verified |

---

## Dependencies

- [x] UPGRADE-009 Structured Logging complete
- [x] Advanced trading metrics exist (`evaluation/advanced_trading_metrics.py`)
- [x] Execution quality metrics exist (`execution/execution_quality_metrics.py`)
- [x] Object Store manager exists (`utils/object_store.py`)

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance overhead | Low | Medium | Efficient data structures |
| Memory growth | Medium | Low | Periodic cleanup |
| Calculation errors | Low | High | Extensive testing |

---

## Estimated Effort

- Performance Tracker Core: 2 hours
- Session Management: 1 hour
- Strategy Breakdown: 1 hour
- Persistence: 1 hour
- Tests: 1.5 hours
- **Total**: ~6.5 hours

---

## Phase 2: Task Checklist

### Core Tracker (T1-T3)

| ID | Task | Est. | Depends | Priority |
|----|------|------|---------|----------|
| T1 | Create `models/performance_tracker.py` | 60m | - | P0 |
| T2 | Add real-time P&L tracking | 30m | T1 | P0 |
| T3 | Add drawdown tracking | 30m | T1 | P0 |

### Session & Strategy (T4-T6)

| ID | Task | Est. | Depends | Priority |
|----|------|------|---------|----------|
| T4 | Add session management (daily/weekly) | 30m | T1 | P0 |
| T5 | Add strategy-level breakdown | 30m | T1 | P0 |
| T6 | Add trade recording | 20m | T1 | P0 |

### Integration (T7-T10)

| ID | Task | Est. | Depends | Priority |
|----|------|------|---------|----------|
| T7 | Integrate with StructuredLogger | 20m | T1 | P0 |
| T8 | Add Object Store persistence | 30m | T1 | P1 |
| T9 | Create `tests/test_performance_tracker.py` | 45m | T1-T8 | P0 |
| T10 | Update `models/__init__.py` exports | 10m | T1 | P0 |

---

## Phase 3: Implementation

### T1: Performance Tracker Core

```python
# models/performance_tracker.py
"""
Unified Performance Tracker for Trading Bot

Provides real-time performance tracking with:
- Live P&L and drawdown monitoring
- Session-based metrics (daily, weekly, monthly)
- Strategy-level performance breakdown
- Trade recording and analysis
- Integration with structured logging
- Object Store persistence

UPGRADE-010: Performance Tracker (December 2025)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum
import threading
import json

@dataclass
class TradeRecord:
    """Individual trade record."""
    trade_id: str
    symbol: str
    strategy: str
    direction: str  # "long" or "short"
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage_bps: float = 0.0
    is_closed: bool = False

@dataclass
class SessionMetrics:
    """Metrics for a trading session."""
    session_date: date
    trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    commissions: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    win_rate: float = 0.0

@dataclass
class StrategyMetrics:
    """Metrics for a specific strategy."""
    strategy_name: str
    trades: int = 0
    winning_trades: int = 0
    net_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

class PerformanceTracker:
    """
    Unified performance tracker for trading operations.

    Features:
    - Real-time P&L tracking
    - Session management (daily/weekly/monthly)
    - Strategy-level breakdown
    - Drawdown monitoring
    - Trade recording
    - Logging integration
    - Object Store persistence
    """

    def __init__(
        self,
        starting_equity: float = 100000.0,
        logger: Optional[Any] = None,
        object_store: Optional[Any] = None,
    ):
        self.starting_equity = starting_equity
        self.current_equity = starting_equity
        self.peak_equity = starting_equity
        self.logger = logger
        self.object_store = object_store

        # Trade tracking
        self._trades: Dict[str, TradeRecord] = {}
        self._closed_trades: List[TradeRecord] = []

        # Session tracking
        self._sessions: Dict[date, SessionMetrics] = {}
        self._current_session: Optional[SessionMetrics] = None

        # Strategy tracking
        self._strategies: Dict[str, StrategyMetrics] = {}

        # Real-time metrics
        self._unrealized_pnl: float = 0.0
        self._realized_pnl: float = 0.0
        self._daily_pnl: float = 0.0

        # Thread safety
        self._lock = threading.Lock()
```

---

## Phase 4: Double-Check

**Date**: 2025-12-01
**Checked By**: Claude Code Agent

### Implementation Progress

| Task | Status | Notes |
|------|--------|-------|
| T1: Performance tracker core | ✅ Complete | `models/performance_tracker.py` (~700 lines) |
| T2: Real-time P&L | ✅ Complete | update_unrealized_pnl(), realized tracking |
| T3: Drawdown tracking | ✅ Complete | _update_drawdown() with peak tracking |
| T4: Session management | ✅ Complete | daily, weekly, monthly sessions |
| T5: Strategy breakdown | ✅ Complete | StrategyMetrics with profit factor |
| T6: Trade recording | ✅ Complete | TradeRecord with full lifecycle |
| T7: Logger integration | ✅ Complete | StructuredLogger integration |
| T8: Object Store persistence | ✅ Complete | save_to/load_from_object_store |
| T9: Tests | ✅ Complete | 34 tests passing |
| T10: Exports | ✅ Complete | 6 exports added |

### Success Criteria Verification

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Performance tracker created | File exists | ✅ models/performance_tracker.py | Pass |
| Real-time metrics | P&L, drawdown live | ✅ Working | Pass |
| Strategy breakdown | Per-strategy stats | ✅ StrategyMetrics class | Pass |
| Session tracking | Daily/weekly metrics | ✅ SessionMetrics class | Pass |
| Tests created | >= 25 | ✅ 34 tests | Pass |
| Integration with logger | Events logged | ✅ Verified | Pass |

---

## Phase 5: Introspection Report

**Report Date**: 2025-12-01

### What Worked Well

1. **Dataclass-based Records**: TradeRecord, SessionMetrics, StrategyMetrics provide clean immutable data structures
2. **Thread-safe Design**: Lock-based operations ensure safe concurrent access
3. **Event Listener Pattern**: Allows real-time UI updates and notifications
4. **Automatic Session Management**: Sessions auto-create based on trade dates

### Challenges Encountered

1. **P&L Calculation Direction**: Had to handle long/short positions differently
2. **Session Boundaries**: Weekly/monthly sessions need careful date handling
3. **Metric Aggregation**: Combining realized + unrealized P&L requires care

### Improvements Made During Implementation

1. Added PerformanceSummary dataclass for complete snapshot
2. Added profit factor calculation per strategy
3. Added event listeners for trade open/close events
4. Added commission and slippage tracking per trade

### Lessons Learned

- Unified tracking simplifies dashboard integration
- Event-driven updates enable real-time UI
- Session-based metrics help identify patterns

---

## Phase 6: Convergence Decision

**Decision**: ✅ **CONVERGED - Ready for Integration**

**Rationale**:

- All 10 tasks completed successfully
- 34 test cases passing (exceeds 25 target)
- Clean integration with StructuredLogger
- Thread-safe implementation verified
- Object Store persistence working

**Next Steps**:

1. Integrate with HybridOptionsBot for live tracking
2. Connect to REST API for dashboard endpoints
3. Add WebSocket updates for real-time metrics
4. Create performance analytics endpoints

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-01 | Upgrade path created |
| 2025-12-01 | Implementation complete - all tasks done |
| 2025-12-01 | 34 tests passing |
| 2025-12-01 | Convergence achieved - ready for integration |

---

## Related Documents

- [UPGRADE-009](UPGRADE_009_STRUCTURED_LOGGING.md) - Structured Logging (dependency)
- [Advanced Metrics](../../evaluation/advanced_trading_metrics.py) - Trading metrics
- [Execution Quality](../../execution/execution_quality_metrics.py) - Execution metrics
- [Roadmap](../ROADMAP.md) - Phase 2 Week 2 tasks
