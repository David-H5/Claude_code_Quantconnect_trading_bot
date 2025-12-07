# Trading Bot Fix Guide

## Code Quality Issues and Remediation Plan

**Generated:** 2025-12-06
**Scope:** Main algorithm files, configuration module
**Priority Levels:** P0 (Critical), P1 (High), P2 (Medium), P3 (Low)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [P0 - Critical Bugs](#p0---critical-bugs)
3. [P1 - High Priority Refactoring](#p1---high-priority-refactoring)
4. [P2 - Medium Priority Cleanup](#p2---medium-priority-cleanup)
5. [P3 - Low Priority Improvements](#p3---low-priority-improvements)
6. [Testing Requirements](#testing-requirements)
7. [Implementation Order](#implementation-order)

---

## Executive Summary

### Files Analyzed

| File | Lines | Issues Found |
|------|-------|--------------|
| `algorithms/hybrid_options_bot.py` | 1,156 | 6 |
| `algorithms/options_trading_bot.py` | 862 | 5 |
| `config/__init__.py` | 727 | 3 |
| `CLAUDE.md` | 178 | 2 |

### Issue Breakdown

| Priority | Count | Description |
|----------|-------|-------------|
| P0 Critical | 1 | Logic bug causing incorrect behavior |
| P1 High | 3 | Code duplication, maintainability |
| P2 Medium | 5 | Dead code, clarity issues |
| P3 Low | 5 | Style, minor improvements |

---

## P0 - Critical Bugs

### P0-1: Shared Timestamp Bug in Check Methods

**Location:** `algorithms/hybrid_options_bot.py:565-573`

**Problem:** Two methods check different intervals but share the same timestamp variable, causing one to always fail after the other updates.

**Current Code (BROKEN):**
```python
# Line 565-568
def _should_check_strategies(self) -> bool:
    """Determine if it's time to check autonomous strategies."""
    # Check every 5 minutes
    return (self.Time - self._last_check_time).total_seconds() >= 300

# Line 570-573
def _should_check_recurring(self) -> bool:
    """Determine if it's time to check recurring templates."""
    # Check every hour
    return (self.Time - self._last_check_time).total_seconds() >= 3600
```

**Bug Scenario:**
1. At T+0: Both return False (just started)
2. At T+5min: `_should_check_strategies()` returns True
3. Something updates `_last_check_time` to current time
4. At T+6min: `_should_check_recurring()` returns False (only 1 min since update)
5. `_should_check_recurring()` never returns True unless strategies check stops

**Fix:**

```python
# In Initialize() method, around line 280-284, add:
self._last_strategy_check_time = self.Time
self._last_recurring_check_time = self.Time

# Replace methods at lines 565-573:
def _should_check_strategies(self) -> bool:
    """Determine if it's time to check autonomous strategies."""
    if (self.Time - self._last_strategy_check_time).total_seconds() >= 300:
        self._last_strategy_check_time = self.Time
        return True
    return False

def _should_check_recurring(self) -> bool:
    """Determine if it's time to check recurring templates."""
    if (self.Time - self._last_recurring_check_time).total_seconds() >= 3600:
        self._last_recurring_check_time = self.Time
        return True
    return False
```

**Also update:** Line 280 initialization:
```python
# BEFORE (line 280):
self._last_check_time = self.Time

# AFTER:
self._last_strategy_check_time = self.Time
self._last_recurring_check_time = self.Time
self._last_resource_check = self.Time
```

**Testing Required:**
- Unit test that both methods can return True independently
- Integration test running for simulated 2+ hours

---

## P1 - High Priority Refactoring

### P1-1: Massive Code Duplication Between Bot Files

**Location:**
- `algorithms/hybrid_options_bot.py`
- `algorithms/options_trading_bot.py`

**Problem:** ~80% code overlap between files. Changes must be made twice, bugs can diverge.

**Duplicated Sections:**

| Section | hybrid_options_bot.py | options_trading_bot.py |
|---------|----------------------|------------------------|
| Brokerage setup | Lines 123-125 | Lines 91-95 |
| Risk config loading | Lines 140-168 | Lines 105-129 |
| Resource monitor init | Lines 178-183 | Lines 132-137 |
| Object store init | Lines 186-201 | Lines 140-158 |
| Node info method | Lines 1038-1055 | Lines 210-228 |
| Config getter | Lines 914-918 | Lines 204-208 |
| Circuit breaker callback | Lines 904-912 | Lines 774-778 |

**Fix - Create Base Class:**

Create `algorithms/base_options_bot.py`:

```python
"""
Base Options Trading Bot

Shared functionality for all options trading algorithms.
Subclasses implement specific trading strategies.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any

try:
    from AlgorithmImports import *
except ImportError:
    # Stubs for development/testing
    class QCAlgorithm:
        pass
    class Resolution:
        Daily = "Daily"
        Hour = "Hour"
        Minute = "Minute"
    class Slice:
        pass
    class BrokerageName:
        CharlesSchwab = "CharlesSchwab"
    class AccountType:
        Margin = "Margin"
    class OrderEvent:
        pass

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from models import (
    CircuitBreakerConfig,
    RiskLimits,
    RiskManager,
    TradingCircuitBreaker,
)
from observability.monitoring.system.resource import create_resource_monitor
from utils.object_store import create_object_store_manager
from utils.storage_monitor import create_storage_monitor


class BaseOptionsBot(QCAlgorithm):
    """
    Base class for options trading algorithms.

    Provides:
    - Configuration loading
    - Risk management setup
    - Circuit breaker integration
    - Resource monitoring
    - Object store persistence

    Subclasses must implement:
    - _setup_strategy_specific(): Strategy-specific initialization
    - OnData(): Market data processing
    """

    def Initialize(self) -> None:
        """Initialize common algorithm components."""
        self._setup_basic()
        self._setup_config()
        self._setup_risk_management()
        self._setup_monitoring()
        self._setup_strategy_specific()  # Subclass hook
        self._setup_schedules()
        self._log_initialization()

    def _setup_basic(self) -> None:
        """Set up basic algorithm parameters."""
        # Override in subclass for custom dates/cash
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)

        # CRITICAL: Charles Schwab allows ONLY ONE algorithm per account
        # Deploying a second algorithm will automatically stop the first one
        self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)

    def _setup_config(self) -> None:
        """Load configuration."""
        try:
            self.config = get_config()
            self.Debug("Configuration loaded successfully")
        except FileNotFoundError:
            self.config = None
            self.Debug("Config file not found, using defaults")

    def _setup_risk_management(self) -> None:
        """Initialize risk management components."""
        risk_config = self._get_config("risk_management", {})

        # Risk limits
        self.risk_limits = RiskLimits(
            max_position_size=risk_config.get("max_position_size_pct", 0.25),
            max_daily_loss=risk_config.get("max_daily_loss_pct", 0.03),
            max_drawdown=risk_config.get("max_drawdown_pct", 0.10),
            max_risk_per_trade=risk_config.get("max_risk_per_trade_pct", 0.02),
        )

        # Risk manager
        self.risk_manager = RiskManager(
            starting_equity=self.Portfolio.TotalPortfolioValue,
            limits=self.risk_limits,
        )

        # Circuit breaker
        breaker_config = CircuitBreakerConfig(
            max_daily_loss_pct=risk_config.get("max_daily_loss_pct", 0.03),
            max_drawdown_pct=risk_config.get("max_drawdown_pct", 0.10),
            max_consecutive_losses=risk_config.get("max_consecutive_losses", 5),
            require_human_reset=risk_config.get("require_human_reset", True),
        )
        self.circuit_breaker = TradingCircuitBreaker(
            config=breaker_config,
            alert_callback=self._on_circuit_breaker_alert,
        )

    def _setup_monitoring(self) -> None:
        """Initialize resource and storage monitoring."""
        # Resource monitor
        resource_config = self._get_config("quantconnect", {}).get("resource_limits", {})
        self.resource_monitor = create_resource_monitor(
            config=resource_config,
            circuit_breaker=self.circuit_breaker,
        )
        self.Debug(f"Resource monitor initialized: {self._get_node_info()}")

        # Object Store
        object_store_config = self._get_config("quantconnect", {}).get("object_store", {})
        if object_store_config.get("enabled", False):
            self.object_store_manager = create_object_store_manager(
                algorithm=self,
                config=object_store_config,
            )
            self.storage_monitor = create_storage_monitor(
                object_store_manager=self.object_store_manager,
                config=object_store_config,
                circuit_breaker=self.circuit_breaker,
            )
        else:
            self.object_store_manager = None
            self.storage_monitor = None

    def _setup_strategy_specific(self) -> None:
        """
        Hook for subclass-specific initialization.

        Override in subclass to set up:
        - Executors
        - Scanners
        - LLM integration
        - Custom data subscriptions
        """
        raise NotImplementedError("Subclass must implement _setup_strategy_specific()")

    def _setup_schedules(self) -> None:
        """Set up common scheduled events. Override to add more."""
        # Resource monitoring every 30 seconds
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(seconds=30)),
            self._check_resources,
        )

    def _log_initialization(self) -> None:
        """Log initialization summary."""
        self.Debug("=" * 60)
        self.Debug(f"{self.__class__.__name__} INITIALIZED")
        self.Debug("=" * 60)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_config(self, section: str, default: Any) -> Any:
        """Get configuration section safely."""
        if self.config:
            return self.config.get(section, default)
        return default

    def _get_node_info(self) -> str:
        """Get current compute node information."""
        qc_config = self._get_config("quantconnect", {})
        nodes = qc_config.get("compute_nodes", {})

        if hasattr(self, "LiveMode") and self.LiveMode:
            node = nodes.get("live_trading", {})
            node_type = "live"
        else:
            node = nodes.get("backtesting", {})
            node_type = "backtest"

        model = node.get("model", "unknown")
        ram_gb = node.get("ram_gb", 0)
        cores = node.get("cores", 0)

        return f"{model} ({cores}C/{ram_gb}GB) [{node_type}]"

    def _check_resources(self) -> None:
        """Check resource usage."""
        if self.resource_monitor:
            try:
                self.resource_monitor.check_resources()
            except Exception as e:
                self.Debug(f"Resource check error: {e}")

    def _on_circuit_breaker_alert(self, message: str, urgency: str) -> None:
        """Handle circuit breaker alerts."""
        self.Debug(f"CIRCUIT BREAKER [{urgency}]: {message}")

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    def OnEndOfAlgorithm(self) -> None:
        """Common end-of-algorithm reporting."""
        self.Debug("=" * 60)
        self.Debug("ALGORITHM COMPLETED")
        self.Debug(f"Final Equity: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Debug(f"Circuit Breaker: {'HALTED' if self.circuit_breaker.is_halted else 'OK'}")
        self.Debug("=" * 60)
```

**Then refactor hybrid_options_bot.py:**

```python
from algorithms.base_options_bot import BaseOptionsBot

class HybridOptionsBot(BaseOptionsBot):
    """Semi-autonomous hybrid options trading algorithm."""

    def _setup_basic(self) -> None:
        """Override dates for hybrid bot."""
        self.SetStartDate(2024, 11, 1)
        self.SetEndDate(2024, 11, 30)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)

    def _setup_strategy_specific(self) -> None:
        """Initialize hybrid architecture components."""
        self._setup_sentiment_components()
        self._setup_executors()
        self._setup_position_management()
        self._setup_order_sources()
        self._setup_universe()
        # ... rest of hybrid-specific setup
```

**Files to Modify:**
1. Create `algorithms/base_options_bot.py` (new file)
2. Refactor `algorithms/hybrid_options_bot.py` to extend base
3. Refactor `algorithms/options_trading_bot.py` to extend base
4. Update `algorithms/__init__.py` exports

**Estimated Line Reduction:** ~400 lines removed from duplication

---

### P1-2: Bloated Initialize Method

**Location:** `algorithms/hybrid_options_bot.py:113-304`

**Problem:** 190+ line method with 10+ responsibilities.

**Fix - Split into focused methods:**

```python
def Initialize(self) -> None:
    """Initialize algorithm - delegates to focused setup methods."""
    self._setup_basic_parameters()
    self._setup_configuration()
    self._setup_risk_management()
    self._setup_sentiment_components()
    self._setup_monitoring()
    self._setup_executors()
    self._setup_position_management()
    self._setup_order_sources()
    self._setup_data_subscriptions()
    self._setup_scheduled_tasks()
    self._initialize_tracking()
    self._log_initialization_summary()
```

Each method should be 15-30 lines max. This is already partially done with `_setup_sentiment_components()` - extend the pattern to all sections.

---

### P1-3: Remove Dead Scheduled Methods

**Location:** `algorithms/hybrid_options_bot.py:496-504`

**Problem:** Empty methods that do nothing but waste scheduler cycles.

**Current Code:**
```python
def _scheduled_strategy_check(self) -> None:
    """Scheduled check for autonomous strategies."""
    # Strategy checks happen in OnData, this is just a backup trigger
    pass

def _scheduled_recurring_check(self) -> None:
    """Scheduled check for recurring templates."""
    # Recurring checks happen in OnData, this is just a backup trigger
    pass
```

**Fix Options:**

**Option A - Remove entirely:**
```python
# Delete these methods and remove from _setup_schedules():
# Lines 483-491 - remove the Schedule.On calls for these
```

**Option B - Implement properly:**
```python
def _scheduled_strategy_check(self) -> None:
    """Scheduled check for autonomous strategies."""
    if self.options_executor and self.circuit_breaker.can_trade():
        try:
            # Force a strategy check regardless of last check time
            self.options_executor.scheduled_check()
        except Exception as e:
            self.Debug(f"Scheduled strategy check error: {e}")

def _scheduled_recurring_check(self) -> None:
    """Scheduled check for recurring templates."""
    if self.recurring_manager and self.circuit_breaker.can_trade():
        try:
            orders = self.recurring_manager.check_templates_forced()
            for order in orders:
                self.order_queue.submit_order(order)
        except Exception as e:
            self.Debug(f"Scheduled recurring check error: {e}")
```

**Recommendation:** Option A unless there's a specific need for scheduled triggers separate from OnData.

---

## P2 - Medium Priority Cleanup

### P2-1: Remove Large Commented Code Block

**Location:** `algorithms/options_trading_bot.py:637-665`

**Problem:** 30 lines of commented "example" code clutters the file.

**Fix:** Move to documentation or delete.

```python
# DELETE lines 637-665 entirely
# If needed for reference, move to docs/examples/butterfly_strategy.py
```

---

### P2-2: Document or Remove QuantConnectOptionsBot Alias

**Location:** `algorithms/options_trading_bot.py:850-855`

**Problem:** Unexplained alias class.

**Current Code:**
```python
class QuantConnectOptionsBot(OptionsTradingBot):
    """
    Alias for deployment compatibility.
    """
    pass
```

**Fix - Add proper documentation:**
```python
class QuantConnectOptionsBot(OptionsTradingBot):
    """
    Deployment alias for QuantConnect cloud.

    QuantConnect's deployment system expects specific class names.
    This alias allows deploying OptionsTradingBot under the name
    that matches the project configuration.

    Usage:
        # In QuantConnect deployment config:
        "algorithm-type-name": "QuantConnectOptionsBot"

    Note: This is functionally identical to OptionsTradingBot.
    """
    pass
```

**Or if truly unused:** Delete it entirely.

---

### P2-3: Remove Confusing Warmup Comments

**Location:** `algorithms/hybrid_options_bot.py:286-288`

**Current Code:**
```python
# No warmup needed for Greeks (IV-based, PR #6720)
# But warm up for any technical indicators if used
# self.SetWarmUp(timedelta(days=30))
```

**Fix:**
```python
# Greeks use IV and require no warmup (LEAN PR #6720).
# Uncomment below only if adding technical indicators that need history:
# self.SetWarmUp(timedelta(days=30))
```

---

### P2-4: Consolidate Repetitive Config Getters

**Location:** `config/__init__.py:344-515`

**Problem:** 15+ methods following identical pattern.

**Current Pattern (repeated 15 times):**
```python
def get_X_config(self) -> XConfig:
    """Get X configuration."""
    cfg = self._raw_config.get("x", {})
    return XConfig(
        field1=cfg.get("field1", default1),
        field2=cfg.get("field2", default2),
        # ... 5-15 more fields
    )
```

**Fix - Generic loader with dataclass introspection:**

```python
from dataclasses import fields

def _load_config_dataclass(self, section: str, dataclass_type: type) -> Any:
    """
    Generic config loader for dataclasses.

    Uses dataclass field defaults and type hints to load configuration.
    """
    raw = self._raw_config.get(section, {})
    kwargs = {}

    for field in fields(dataclass_type):
        value = raw.get(field.name)
        if value is not None:
            kwargs[field.name] = value
        # else: use dataclass default

    return dataclass_type(**kwargs)

# Then simplify all getters:
def get_risk_config(self) -> RiskConfig:
    """Get risk management configuration."""
    return self._load_config_dataclass("risk_management", RiskConfig)

def get_logging_config(self) -> LoggingConfig:
    """Get structured logging configuration."""
    return self._load_config_dataclass("structured_logging", LoggingConfig)

# ... same pattern for all
```

**Estimated Line Reduction:** ~150 lines

---

### P2-5: Fix Unused Type Imports

**Location:** `config/__init__.py:39`

**Current:**
```python
from typing import Any, Dict, List, Optional, Set
```

**Fix:**
```python
from typing import Any, Optional
```

(Dict, List, Set are unused - modern Python uses `dict`, `list`, `set` directly)

---

## P3 - Low Priority Improvements

### P3-1: Consolidate RIC Loop Documentation in CLAUDE.md

**Location:** `CLAUDE.md:30-37` and `CLAUDE.md:70-84`

**Fix:** Keep one comprehensive section, add cross-reference to the other.

---

### P3-2: Standardize Debug Message Format

**Problem:** Inconsistent emoji usage in debug messages.

**Current (mixed):**
```python
self.Debug("✅ Configuration loaded successfully")
self.Debug("Config file not found, using defaults")  # No emoji
self.Debug("⚠️  Config file not found, using defaults")
```

**Fix - Create constants:**
```python
# At top of file or in utils/logging_utils.py
class LogPrefix:
    SUCCESS = "OK:"
    WARNING = "WARN:"
    ERROR = "ERR:"
    INFO = "INFO:"

# Usage:
self.Debug(f"{LogPrefix.SUCCESS} Configuration loaded")
self.Debug(f"{LogPrefix.WARNING} Config file not found")
```

---

### P3-3: Add Type Hints to Callback Parameters

**Location:** `algorithms/options_trading_bot.py:782-788`

**Current:**
```python
def _on_profit_take_order(self, order: Any) -> None:
def _on_order_event(self, order: Any, action: str) -> None:
```

**Fix:**
```python
from execution import ProfitTakeOrder, SmartOrder

def _on_profit_take_order(self, order: ProfitTakeOrder) -> None:
def _on_order_event(self, order: SmartOrder, action: str) -> None:
```

---

### P3-4: Extract Magic Numbers to Constants

**Locations:** Throughout both bot files

**Examples:**
```python
# Current:
if (self.Time - self._last_check_time).total_seconds() >= 300
if (self.Time - self._last_check_time).total_seconds() >= 3600
if (self.Time - self._last_sentiment_check).total_seconds() < 300

# Fix - Add to class or config:
class CheckIntervals:
    STRATEGY_CHECK_SECONDS = 300      # 5 minutes
    RECURRING_CHECK_SECONDS = 3600    # 1 hour
    SENTIMENT_CHECK_SECONDS = 300     # 5 minutes
    RESOURCE_CHECK_SECONDS = 30       # 30 seconds
```

---

### P3-5: Reduce Initialize Logging Verbosity

**Location:** `algorithms/hybrid_options_bot.py:294-304`

**Current:**
```python
self.Debug("=" * 80)
self.Debug("✅ HYBRID OPTIONS BOT INITIALIZED SUCCESSFULLY")
self.Debug(f"   Autonomous: {self.options_executor is not None}")
self.Debug(f"   Manual: {self.manual_executor is not None}")
self.Debug(f"   Bot Manager: {self.bot_manager is not None}")
self.Debug(f"   Recurring: {self.recurring_manager is not None}")
self.Debug(f"   Object Store: {self.object_store_manager is not None}")
self.Debug(f"   Sentiment Filter: {self.sentiment_filter is not None}")
self.Debug(f"   News Alerts: {self.news_alert_manager is not None}")
self.Debug(f"   LLM Guardrails: {self.llm_guardrails is not None}")
self.Debug("=" * 80)
```

**Fix - Condense:**
```python
components = {
    "Autonomous": self.options_executor,
    "Manual": self.manual_executor,
    "BotMgr": self.bot_manager,
    "Recurring": self.recurring_manager,
    "ObjStore": self.object_store_manager,
    "Sentiment": self.sentiment_filter,
    "NewsAlerts": self.news_alert_manager,
    "Guardrails": self.llm_guardrails,
}
enabled = [k for k, v in components.items() if v is not None]
disabled = [k for k, v in components.items() if v is None]

self.Debug(f"INITIALIZED: {', '.join(enabled)}")
if disabled:
    self.Debug(f"DISABLED: {', '.join(disabled)}")
```

---

## Testing Requirements

### Unit Tests Required

| Fix ID | Test File | Test Cases |
|--------|-----------|------------|
| P0-1 | `tests/test_hybrid_bot_timing.py` | `test_strategy_and_recurring_checks_independent` |
| P1-1 | `tests/test_base_options_bot.py` | `test_base_initialization`, `test_subclass_override` |
| P1-2 | `tests/test_hybrid_bot.py` | `test_initialize_calls_all_setup_methods` |
| P2-4 | `tests/test_config.py` | `test_generic_config_loader` |

### Integration Tests Required

| Fix ID | Test Scenario |
|--------|---------------|
| P0-1 | Run 2-hour simulation, verify both checks trigger correctly |
| P1-1 | Deploy base class with both subclasses, verify no regressions |

---

## Implementation Order

### Phase 1: Critical Fix (Do First)
1. **P0-1**: Fix timestamp bug - prevents incorrect trading behavior

### Phase 2: Architecture (Do Together)
2. **P1-1**: Create base class
3. **P1-2**: Refactor Initialize method
4. **P1-3**: Remove/implement dead methods

### Phase 3: Cleanup (Any Order)
5. **P2-1**: Remove commented code
6. **P2-2**: Document alias class
7. **P2-3**: Fix warmup comments
8. **P2-4**: Consolidate config getters
9. **P2-5**: Fix imports

### Phase 4: Polish (Optional)
10. **P3-1** through **P3-5**: Low priority improvements

---

## Checklist

```
[x] P0-1: Fix shared timestamp bug
[x] P1-1: Create BaseOptionsBot class
[x] P1-2: Split Initialize into focused methods
[x] P1-3: Remove/implement dead scheduled methods
[x] P2-1: Remove commented butterfly code
[x] P2-2: Document QuantConnectOptionsBot alias
[x] P2-3: Clarify warmup comments
[x] P2-4: Add generic config loader
[x] P2-5: Remove unused type imports
[ ] P3-1: Consolidate CLAUDE.md RIC docs
[ ] P3-2: Standardize debug message format
[ ] P3-3: Add callback type hints
[ ] P3-4: Extract magic numbers to constants
[ ] P3-5: Reduce logging verbosity
[ ] Run full test suite
[ ] Update CHANGELOG.md
```

---

## Notes

- All fixes should follow TAP Protocol (see CLAUDE.md)
- Run `pytest tests/ -v --cov=. --cov-fail-under=70` after each phase
- Create feature branch: `fix/code-quality-cleanup`
- Consider splitting into multiple PRs by phase

---

## Related Documents

| Document | Scope | Relationship |
|----------|-------|--------------|
| [MASTER_CONSOLIDATION_PLAN.md](MASTER_CONSOLIDATION_PLAN.md) | Architecture consolidation | **Parallel work** - no conflicts |
| [CONSOLIDATION_PLAN.md](CONSOLIDATION_PLAN.md) | Completed consolidation phases | Reference for done work |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Project structure | Context for changes |

### Execution Order with MASTER_CONSOLIDATION_PLAN

1. **FIX_GUIDE P0-1** (timestamp bug) - Do FIRST, critical
2. **FIX_GUIDE P1-*** (algorithm refactoring) - Can parallel with MASTER_PLAN consolidation
3. **MASTER_PLAN** sentiment/news consolidation - After or parallel with P1
4. **FIX_GUIDE P2-*/P3-*** (cleanup/polish) - After architecture stabilizes
