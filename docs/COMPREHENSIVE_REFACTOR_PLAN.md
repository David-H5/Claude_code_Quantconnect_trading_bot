# Comprehensive Refactoring Plan: Autonomous AI Trading Bot

**Version**: 2.0.0
**Generated**: 2025-12-06
**Status**: Pre-Refactor - Ready for Implementation
**Document Type**: Master Refactoring Guide with Audit Trail

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Pre-Refactoring Protocol](#2-pre-refactoring-protocol)
3. [Codebase Structure Analysis](#3-codebase-structure-analysis)
4. [Critical Fixes (P0) - Financial Safety](#4-critical-fixes-p0---financial-safety)
5. [High Priority Fixes (P1)](#5-high-priority-fixes-p1)
6. [Structural Refactoring (P2)](#6-structural-refactoring-p2)
7. [Code Consolidation](#7-code-consolidation)
8. [Missing Abstractions](#8-missing-abstractions)
9. [Logging Standards](#9-logging-standards)
10. [Audit Trail Implementation](#10-audit-trail-implementation)
11. [Testing Requirements](#11-testing-requirements)
12. [Best Practices](#12-best-practices)
13. [Rollback Procedures](#13-rollback-procedures)
14. [Post-Refactoring Validation](#14-post-refactoring-validation)
15. [Implementation Phases](#15-implementation-phases)
16. [Appendix](#16-appendix)

---

## 1. Executive Summary

### 1.1 Current State Assessment

| Category | Metric | Value | Status |
|----------|--------|-------|--------|
| **Critical** | P0 Bugs (Financial Risk) | 7 | BLOCKING |
| **Critical** | P1 Bugs (High Severity) | 15 | Sprint Priority |
| **Structure** | Code Duplications | 9 categories | Consolidation Needed |
| **Structure** | Missing Base Classes | 4 | Architecture Gap |
| **Testing** | Test Files | 123 | Good Coverage |
| **Testing** | Test Organization | Scattered | Needs Reorganization |
| **Documentation** | Doc Files | 175+ | Needs Curation |
| **Modules** | Largest Module (LLM) | 116MB | Needs Splitting |
| **Production** | Readiness | BLOCKED | Critical Fixes Required |

### 1.2 Top-Level Directory Structure

```
/project-root/
├── .claude/           # Claude Code infrastructure (EXCELLENT)
├── algorithms/        # Trading algorithms (4 strategies)
├── analytics/         # Options pricing & Greeks
├── api/               # REST API & WebSocket servers
├── backtesting/       # Walk-forward, Monte Carlo
├── compliance/        # Audit logging, FINRA, regulatory
├── config/            # Configuration management
├── data/              # Custom data sources
├── docs/              # 175+ documentation files
├── evaluation/        # 25+ evaluation frameworks (SCATTERED)
├── execution/         # Order execution (19 modules - WELL ORGANIZED)
├── indicators/        # Technical analysis
├── infrastructure/    # Redis, pub/sub, streaming
├── integration_tests/ # System-level tests
├── llm/               # LLM integration (116MB - NEEDS SPLITTING)
├── logs/              # Runtime logs
├── mcp/               # MCP servers
├── models/            # Risk management, circuit breaker
├── observability/     # 38 monitoring modules (EXCELLENT)
├── prompts/           # Prompt versioning
├── scanners/          # Market scanners
├── scripts/           # 40+ utility scripts
├── tests/             # 123 test files (NEEDS REORGANIZATION)
├── ui/                # PySide6 dashboard
└── utils/             # Helper utilities
```

### 1.3 Refactoring Goals (Priority Order)

1. **SAFETY FIRST**: Fix all P0 bugs before ANY live trading
2. **RELIABILITY**: Eliminate silent failures, improve error handling
3. **MAINTAINABILITY**: Consolidate duplications, add base classes
4. **OBSERVABILITY**: Standardize logging, implement audit trails
5. **SECURITY**: Fix authentication and CORS vulnerabilities
6. **STRUCTURE**: Split large modules, reorganize tests

### 1.4 Estimated Effort

| Phase | Duration | Priority | Dependencies |
|-------|----------|----------|--------------|
| Phase 1: P0 Critical Fixes | 2-3 days | IMMEDIATE | None |
| Phase 2: P1 High Priority | 1 week | This Sprint | Phase 1 |
| Phase 3: Code Consolidation | 1 week | This Sprint | Phase 1 |
| Phase 4: Structural Refactoring | 2 weeks | Next Sprint | Phase 2-3 |
| Phase 5: Logging/Audit Overhaul | 3-4 days | Next Sprint | Phase 2 |
| Phase 6: Test Reorganization | 3-4 days | Next Sprint | Phase 3 |

---

## 2. Pre-Refactoring Protocol

### 2.1 Environment Setup Checklist

```bash
# Create dedicated refactoring branch
git checkout -b refactor/v2.0-comprehensive

# Tag current state for rollback
git tag pre-refactor-v1.x -m "Pre-refactor snapshot $(date +%Y-%m-%d)"

# Verify all tests pass
pytest tests/ -v --tb=short

# Record test count baseline
pytest tests/ --collect-only | tail -1 > .refactor-baseline.txt

# Backup current Object Store state
python scripts/manage_object_store.py backup

# Document current configuration
cp config/settings.json config/settings.json.pre-refactor
```

### 2.2 Documentation Snapshot

- [ ] Export current PROJECT_STATUS.md metrics
- [ ] Screenshot current dashboard state
- [ ] Document all environment variables in use
- [ ] List all active MCP server configurations
- [ ] Record current test count: `pytest --collect-only`

### 2.3 Safety Measures

- [ ] Set trading bot to **MANUAL** mode (not autonomous)
- [ ] Close all open positions before major changes
- [ ] Disable any scheduled deployments
- [ ] Notify team of refactoring window
- [ ] Create incident response plan

### 2.4 Audit Trail: Refactoring Session Start

```json
{
  "event": "REFACTOR_SESSION_START",
  "timestamp": "2025-12-06T00:00:00Z",
  "version": "2.0.0",
  "baseline": {
    "test_count": 541,
    "git_tag": "pre-refactor-v1.x",
    "git_commit": "<commit-hash>",
    "modules_count": 45,
    "critical_issues": 7,
    "high_issues": 15
  },
  "goals": [
    "Fix P0 financial safety bugs",
    "Consolidate duplicated code",
    "Add missing base classes",
    "Standardize logging",
    "Implement audit trails"
  ]
}
```

---

## 3. Codebase Structure Analysis

### 3.1 Module Quality Assessment

| Module | Size | Files | Quality | Priority Issues |
|--------|------|-------|---------|-----------------|
| **execution/** | 19 modules | 17 .py | GOOD | Missing ExecutorBase |
| **llm/** | 116MB | 25+ | NEEDS WORK | Too large, 56KB file, bloated __init__ |
| **models/** | 18MB | 25 | GOOD | Some orphan modules |
| **evaluation/** | 22MB | 25+ | SCATTERED | 25 competing frameworks |
| **observability/** | 11MB | 38 | EXCELLENT | Minor duplication |
| **tests/** | 123 files | 123 | SCATTERED | 50+ files at root |
| **.claude/** | 10 subdirs | ~50 | EXCELLENT | Clean deprecated/ |

### 3.2 Theme 1: Trading Execution (19 modules) - GOOD

**Location**: `execution/`

**Key Files**:
| File | Size | Purpose |
|------|------|---------|
| `arbitrage_executor.py` | 47KB | 2-part spread execution |
| `two_part_spread.py` | 42KB | Core spread strategy |
| `bot_managed_positions.py` | 23KB | Position lifecycle (**P0 BUG HERE**) |
| `smart_execution.py` | 17KB | Smart order routing (**P0 BUG HERE**) |
| `fill_predictor.py` | 21KB | ML-based fill probability |
| `pre_trade_validator.py` | 19KB | Pre-execution validation |
| `execution_quality_metrics.py` | 1KB | **NEEDS EXPANSION** |

**Issues**:
- P0 Bug: Position sizing truncation in `bot_managed_positions.py:400-402`
- P0 Bug: Fill price calculation in `smart_execution.py:241-242`
- `execution_quality_metrics.py` is severely under-implemented (1KB)
- No ExecutorBase abstract class
- Heavy dict passing instead of dataclasses

**Action Items**:
1. [ ] Fix P0 bugs (immediate)
2. [ ] Create ExecutorBase abstract class
3. [ ] Implement execution_quality_metrics.py properly
4. [ ] Standardize on dataclasses for results

### 3.3 Theme 2: LLM/AI Integration (25+ modules, 116MB) - NEEDS SPLITTING

**Location**: `llm/`

**Critical Issues**:
1. **Bloated `__init__.py`** (11KB) - exports 21+ items
2. **`sentiment_filter.py`** (56KB) - mixing concerns, needs splitting
3. **Unclear separation**:
   - `news_analyzer` vs `news_processor` (redundant?)
   - `decision_logger` vs `reasoning_logger` (overlapping?)
4. **No clear layering**: Agents directly instantiate clients
5. **Inconsistent patterns**: Some agents inherit base, others standalone

**Recommended New Structure**:
```
llm/
├── core/                    # Core interfaces and base classes
│   ├── __init__.py
│   ├── base.py              # AgentBase, ProviderBase
│   ├── providers.py         # LLM provider abstraction
│   └── ensemble.py          # Weighted ensemble
│
├── agents/                  # Agent implementations (exists, keep)
│   ├── __init__.py
│   ├── base.py
│   ├── technical_analyst.py
│   ├── sentiment_analyst.py
│   └── ...
│
├── analysis/                # Analysis modules
│   ├── __init__.py
│   ├── sentiment.py         # Sentiment analysis
│   ├── sentiment_filter.py  # SPLIT THIS
│   ├── news_analyzer.py
│   ├── earnings_analyzer.py
│   └── entity_extractor.py
│
├── tools/                   # Agent tools
│   ├── __init__.py
│   └── ...
│
├── logging/                 # Decision logging
│   ├── __init__.py
│   ├── decision_logger.py
│   └── reasoning_logger.py
│
└── __init__.py              # Slim exports only
```

### 3.4 Theme 3: Evaluation Frameworks (25+ files) - NEEDS CONSOLIDATION

**Location**: `evaluation/`

**Critical Issues**:
- 25+ competing frameworks
- Unclear which is primary
- Naming confusion: `agent_metrics` vs `metrics` vs `advanced_trading_metrics`
- `walk_forward_analysis` should be in `backtesting/`
- Example files checked in (should be tests)

**Recommended Consolidation**:
```
evaluation/
├── core/                    # Primary framework
│   ├── __init__.py
│   ├── framework.py         # Main evaluation framework
│   ├── metrics.py           # Consolidated metrics
│   └── runner.py            # Evaluation runner
│
├── datasets/                # Test datasets (exists, expand)
│   ├── analyst_cases.py
│   ├── trader_cases.py
│   └── benchmarks/
│
├── monitors/                # Monitoring
│   ├── __init__.py
│   ├── continuous.py
│   ├── drift_detection.py
│   └── alerting.py
│
└── adapters/                # Adapters (exists)
```

**Files to Move**:
- `walk_forward_analysis.py` → `backtesting/walk_forward.py`
- `psi_drift_detection.py` → `evaluation/monitors/drift_detection.py`

**Files to Delete** (after extracting useful code):
- `example_usage.py` → convert to tests
- `complete_evaluation_example.py` → convert to tests

### 3.5 Theme 4: Tests (123 files) - NEEDS REORGANIZATION

**Current Structure** (problematic):
```
tests/
├── test_*.py              # 50+ files at root!
├── regression/            # 2 files
├── backtesting/           # 2 files
├── compliance/
├── hooks/
├── infrastructure/
├── mcp/
├── observability/
├── analytics/
└── conftest.py
```

**Recommended Structure** (mirrors source):
```
tests/
├── execution/             # Tests for execution/
│   ├── test_arbitrage_executor.py
│   ├── test_smart_execution.py
│   └── test_bot_managed_positions.py
│
├── llm/                   # Tests for llm/
│   ├── test_ensemble.py
│   ├── test_agents.py
│   └── test_sentiment.py
│
├── models/                # Tests for models/
│   ├── test_risk_manager.py
│   └── test_circuit_breaker.py
│
├── api/                   # Tests for api/
│   ├── test_rest_server.py
│   └── test_websocket.py
│
├── integration/           # Integration tests
├── regression/            # Regression tests
├── performance/           # Performance tests
└── conftest.py            # Shared fixtures
```

### 3.6 Dependency Relationships

```
algorithms/hybrid_options_bot.py
  ├→ llm/ensemble.py
  ├→ llm/agents/*.py
  ├→ execution/arbitrage_executor.py
  │   ├→ execution/two_part_spread.py
  │   ├→ execution/fill_predictor.py
  │   └→ models/risk_manager.py
  ├→ models/risk_manager.py
  ├→ scanners/*.py
  └→ indicators/*.py
```

**Circular Dependency Risks**: Currently low, but monitor:
- `llm/` ↔ `models/` ↔ `execution/`

---

## 4. Critical Fixes (P0) - Financial Safety

> **WARNING**: These issues can cause financial loss. Fix before ANY live trading.

### 4.1 Position Sizing Truncation Bug

**Location**: [execution/bot_managed_positions.py:400-402](execution/bot_managed_positions.py#L400-L402)

**Impact**: For 1-contract position with 30% profit-taking:
- `int(1 * 0.30) = 0` → Override to `1` → **Closes 100% instead of 30%**

**Current Code (BROKEN)**:
```python
close_quantity = int(position.current_quantity * threshold.take_pct)
if close_quantity == 0:
    close_quantity = 1  # BUG: Closes 100% instead of intended %
```

**Fixed Code**:
```python
def calculate_close_quantity(
    current_quantity: int,
    take_pct: float,
    min_close: int = 1
) -> int:
    """
    Calculate quantity to close for profit-taking.

    Args:
        current_quantity: Current position size
        take_pct: Percentage to close (0.0 to 1.0)
        min_close: Minimum quantity to close (default 1)

    Returns:
        Quantity to close, or 0 to skip this profit-taking level

    Note:
        For positions where calculated close < min_close,
        we skip this profit-taking level to avoid over-closing.
    """
    calculated = int(current_quantity * take_pct)

    # If position too small for this profit-taking level, skip it
    if calculated < min_close and current_quantity > min_close:
        logger.info(
            "PROFIT_TAKE_SKIPPED",
            extra={
                "position_qty": current_quantity,
                "take_pct": take_pct,
                "calculated": calculated,
                "reason": "position_too_small_for_level"
            }
        )
        return 0  # Signal to skip this profit-taking level

    return max(calculated, min_close) if calculated > 0 else 0
```

**Test Cases**:
```python
def test_position_sizing_small_positions():
    """Ensure small positions don't over-close."""
    assert calculate_close_quantity(1, 0.30) == 0   # Skip, don't close 100%
    assert calculate_close_quantity(1, 1.00) == 1   # Full close OK
    assert calculate_close_quantity(10, 0.30) == 3  # Normal 30%
    assert calculate_close_quantity(100, 0.25) == 25
```

**Audit Log Entry**:
```python
logger.info(
    "PROFIT_TAKE_CALCULATION",
    extra={
        "position_id": position.id,
        "current_qty": current_quantity,
        "take_pct": take_pct,
        "calculated_close": calculated,
        "action": "skip" if calculated == 0 else "close"
    }
)
```

---

### 4.2 Position Rolling Not Implemented

**Location**: [execution/bot_managed_positions.py:446-475](execution/bot_managed_positions.py#L446-L475)

**Current Code (STUB)**:
```python
def _execute_roll(self, position: BotManagedPosition) -> Optional[ManagementAction]:
    """Execute position roll - close current and open new position."""
    # In a real implementation, this would:
    # 1. Close the current position
    # 2. Open a new position with later expiration
    # For now, just log and mark as rolled  # <-- NOT ACCEPTABLE FOR PRODUCTION
```

**Required Implementation**:
```python
def _execute_roll(
    self,
    position: BotManagedPosition,
    target_dte: int = 30
) -> Optional[ManagementAction]:
    """
    Execute position roll to later expiration.

    Args:
        position: Position to roll
        target_dte: Target days to expiration for new position

    Returns:
        ManagementAction with roll details, or None if roll failed

    Raises:
        RollExecutionError: If roll cannot be completed atomically
    """
    roll_id = str(uuid.uuid4())

    logger.info(
        "ROLL_INITIATED",
        extra={
            "roll_id": roll_id,
            "position_id": position.id,
            "symbol": position.symbol,
            "current_dte": position.days_to_expiration,
            "target_dte": target_dte
        }
    )

    try:
        # Step 1: Find new contract
        new_contract = self._find_roll_target(
            position.underlying,
            position.option_type,
            position.strike,
            target_dte
        )

        if not new_contract:
            logger.warning(
                "ROLL_NO_TARGET",
                extra={"roll_id": roll_id, "reason": "No suitable contract found"}
            )
            return None

        # Step 2: Create atomic roll order (close old + open new)
        roll_order = self._create_roll_combo_order(
            close_contract=position.contract,
            open_contract=new_contract,
            quantity=position.current_quantity
        )

        # Step 3: Execute with pre-trade validation
        validation = self.pre_trade_validator.validate(roll_order)
        if not validation.passed:
            logger.error(
                "ROLL_VALIDATION_FAILED",
                extra={
                    "roll_id": roll_id,
                    "failures": [c.to_dict() for c in validation.failed_checks]
                }
            )
            return None

        # Step 4: Submit order
        order_ticket = self.algorithm.MarketOrder(roll_order)

        # Step 5: Update position tracking
        self._update_position_for_roll(position, new_contract, roll_id)

        logger.info(
            "ROLL_COMPLETED",
            extra={
                "roll_id": roll_id,
                "old_expiry": position.expiry.isoformat(),
                "new_expiry": new_contract.expiry.isoformat(),
                "order_id": str(order_ticket.OrderId)
            }
        )

        return ManagementAction(
            action_type="roll",
            position_id=position.id,
            roll_id=roll_id,
            details={
                "old_contract": str(position.contract),
                "new_contract": str(new_contract),
                "quantity": position.current_quantity
            }
        )

    except Exception as e:
        logger.error(
            "ROLL_FAILED",
            extra={
                "roll_id": roll_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        raise RollExecutionError(f"Roll failed for {position.id}: {e}") from e
```

---

### 4.3 Race Conditions in Global State

**Location**: [api/rest_server.py:29-53](api/rest_server.py#L29-L53)

**Current Code (UNSAFE)**:
```python
_order_queue: Optional["OrderQueueAPI"] = None
_ws_manager: Optional["WebSocketManager"] = None
_algorithm = None  # Reference to running algorithm
```

**Fixed Code**:
```python
import threading
from contextlib import contextmanager
from typing import Optional, TypeVar, Generic

T = TypeVar('T')

class ThreadSafeRef(Generic[T]):
    """Thread-safe reference wrapper with initialization guarantee."""

    def __init__(self, name: str):
        self._value: Optional[T] = None
        self._lock = threading.RLock()
        self._name = name
        self._initialized = threading.Event()

    def set(self, value: T) -> None:
        with self._lock:
            if self._value is not None:
                raise RuntimeError(f"{self._name} already initialized")
            self._value = value
            self._initialized.set()
            logger.info(f"{self._name}_INITIALIZED")

    def get(self, timeout: float = 5.0) -> T:
        if not self._initialized.wait(timeout):
            raise RuntimeError(f"{self._name} not initialized within {timeout}s")
        with self._lock:
            if self._value is None:
                raise RuntimeError(f"{self._name} was cleared")
            return self._value

    def get_or_none(self) -> Optional[T]:
        with self._lock:
            return self._value

    @contextmanager
    def access(self):
        """Context manager for extended access with lock held."""
        with self._lock:
            yield self._value

# Thread-safe global state
_order_queue: ThreadSafeRef["OrderQueueAPI"] = ThreadSafeRef("OrderQueue")
_ws_manager: ThreadSafeRef["WebSocketManager"] = ThreadSafeRef("WebSocketManager")
_algorithm: ThreadSafeRef = ThreadSafeRef("Algorithm")
```

---

### 4.4 WebSocket TOCTOU Race Condition

**Location**: [api/websocket_handler.py:201-205](api/websocket_handler.py#L201-L205)

**Current Code (RACE CONDITION)**:
```python
if disconnected:
    async with self._lock:
        for client in disconnected:
            if client in self._clients:  # Check
                self._clients.remove(client)  # Use - TOCTOU!
```

**Fixed Code**:
```python
async def _cleanup_disconnected(self, disconnected: List[WebSocketClient]) -> int:
    """
    Remove disconnected clients atomically.

    Returns:
        Number of clients actually removed
    """
    removed_count = 0

    async with self._lock:
        # Build new list excluding disconnected (atomic operation)
        disconnected_ids = {id(c) for c in disconnected}
        original_count = len(self._clients)

        self._clients = [
            c for c in self._clients
            if id(c) not in disconnected_ids
        ]

        removed_count = original_count - len(self._clients)

        if removed_count > 0:
            logger.info(
                "WEBSOCKET_CLIENTS_REMOVED",
                extra={
                    "removed_count": removed_count,
                    "remaining_count": len(self._clients)
                }
            )

    # Close connections outside the lock
    for client in disconnected:
        try:
            await client.websocket.close()
        except Exception as e:
            logger.debug(f"Error closing websocket: {e}")

    return removed_count
```

---

### 4.5 Fill Rate Calculation Error

**Location**: [execution/smart_execution.py:241-242](execution/smart_execution.py#L241-L242)

**Current Code (MATHEMATICALLY WRONG)**:
```python
prev_value = order.average_fill_price * (filled_quantity - 1)
order.average_fill_price = (prev_value + fill_price) / filled_quantity
```

**Issue**: Calculates `(avg * (n-1) + new) / n` which is incorrect weighted average.

**Fixed Code**:
```python
def update_average_fill_price(
    order: Order,
    new_fill_price: float,
    new_fill_quantity: int
) -> float:
    """
    Update average fill price with new partial fill.

    Uses proper weighted average: (old_cost + new_cost) / total_qty
    """
    old_quantity = order.filled_quantity
    old_cost = order.average_fill_price * old_quantity
    new_cost = new_fill_price * new_fill_quantity
    total_quantity = old_quantity + new_fill_quantity

    if total_quantity == 0:
        return 0.0

    new_average = (old_cost + new_cost) / total_quantity

    logger.debug(
        "FILL_PRICE_UPDATED",
        extra={
            "order_id": order.id,
            "old_avg": order.average_fill_price,
            "new_avg": new_average,
            "old_qty": old_quantity,
            "new_qty": new_fill_quantity,
            "fill_price": new_fill_price
        }
    )

    order.average_fill_price = new_average
    order.filled_quantity = total_quantity

    return new_average
```

**Test Case**:
```python
def test_average_fill_price_calculation():
    """Verify weighted average is calculated correctly."""
    order = MockOrder(average_fill_price=100.0, filled_quantity=10)

    # Fill 10 more at $110
    new_avg = update_average_fill_price(order, 110.0, 10)

    # Expected: (100*10 + 110*10) / 20 = 2100 / 20 = 105
    assert new_avg == 105.0
    assert order.filled_quantity == 20
```

---

### 4.6 Default Authentication Token

**Location**: [api/order_queue_api.py:612](api/order_queue_api.py#L612)

**Current Code (INSECURE)**:
```python
auth_token: str = "default-token-change-me"
```

**Fixed Code**:
```python
import os
import secrets
from functools import wraps

class AuthConfig:
    """Authentication configuration with secure defaults."""

    def __init__(self):
        self._token = os.environ.get("TRADING_API_TOKEN")

        if not self._token:
            raise EnvironmentError(
                "TRADING_API_TOKEN environment variable is required. "
                "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )

        if self._token == "default-token-change-me":
            raise EnvironmentError(
                "TRADING_API_TOKEN is set to default value. "
                "Please set a secure token."
            )

        if len(self._token) < 32:
            raise EnvironmentError(
                "TRADING_API_TOKEN must be at least 32 characters."
            )

    def validate(self, provided_token: str) -> bool:
        """Constant-time token comparison to prevent timing attacks."""
        return secrets.compare_digest(self._token, provided_token)


def require_auth(func):
    """Decorator to require authentication on API endpoints."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request = kwargs.get('request') or args[0]
        token = request.headers.get("Authorization", "").replace("Bearer ", "")

        if not auth_config.validate(token):
            logger.warning(
                "AUTH_FAILED",
                extra={
                    "endpoint": request.url.path,
                    "client_ip": request.client.host
                }
            )
            raise HTTPException(status_code=401, detail="Invalid or missing token")

        return await func(*args, **kwargs)
    return wrapper

# Initialize at module load - will fail fast if misconfigured
auth_config = AuthConfig()
```

---

### 4.7 CORS Wildcard

**Location**: [api/rest_server.py:113-119](api/rest_server.py#L113-L119)

**Current Code (INSECURE)**:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any domain!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Fixed Code**:
```python
import os

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

if not ALLOWED_ORIGINS:
    if os.environ.get("ENVIRONMENT") == "production":
        raise EnvironmentError(
            "CORS_ALLOWED_ORIGINS must be set in production. "
            "Example: CORS_ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com"
        )
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
    logger.warning("CORS_DEV_MODE", extra={"origins": ALLOWED_ORIGINS})

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

logger.info("CORS_CONFIGURED", extra={"origins": ALLOWED_ORIGINS})
```

---

## 5. High Priority Fixes (P1)

### 5.1 Silent Failure Pattern

**Locations**: Multiple files with `except: pass` or `except Exception: pass`

**Find Pattern**:
```bash
grep -rn "except.*:$" --include="*.py" | grep -v "# Intentional"
grep -rn "pass  *#" --include="*.py" execution/ api/
```

**Replace With**:
```python
# BEFORE (bad)
try:
    self._save_position(position)
except Exception:
    pass

# AFTER (good)
try:
    self._save_position(position)
except Exception as e:
    logger.error(
        "POSITION_SAVE_FAILED",
        extra={
            "position_id": position.id,
            "error": str(e),
            "error_type": type(e).__name__
        }
    )
    if isinstance(e, (IOError, OSError)):
        raise PositionPersistenceError(f"Failed to save {position.id}") from e
```

### 5.2 Add Rate Limiting

**Location**: `api/rest_server.py`

**Implementation**:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/orders")
@limiter.limit("10/minute")  # 10 orders per minute per IP
@require_auth
async def submit_order(request: Request, order: OrderRequest):
    ...
```

### 5.3 Add Greeks Tracking to Positions

**Location**: `execution/bot_managed_positions.py`

```python
@dataclass
class BotManagedPosition:
    """Position managed by the bot with full Greeks tracking."""

    # Existing fields...
    id: str
    symbol: str
    entry_price: float
    current_quantity: int

    # NEW: Greeks fields
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    iv: float = 0.0

    # NEW: Greeks update tracking
    greeks_updated_at: Optional[datetime] = None
    greeks_source: str = "calculated"  # or "market"

    def update_greeks(
        self,
        delta: float,
        gamma: float,
        theta: float,
        vega: float,
        iv: float,
        source: str = "calculated"
    ) -> None:
        """Update position Greeks with audit logging."""
        old_delta = self.delta

        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega
        self.iv = iv
        self.greeks_updated_at = datetime.now(timezone.utc)
        self.greeks_source = source

        logger.debug(
            "POSITION_GREEKS_UPDATED",
            extra={
                "position_id": self.id,
                "old_delta": old_delta,
                "new_delta": delta,
                "iv": iv,
                "source": source
            }
        )

    @property
    def dollar_delta(self) -> float:
        """Position delta in dollar terms."""
        return self.delta * self.current_quantity * 100

    @property
    def daily_theta_decay(self) -> float:
        """Expected daily theta decay in dollars."""
        return self.theta * self.current_quantity * 100
```

### 5.4 Add Slippage Budget Enforcement

**Location**: `execution/smart_execution.py`

```python
@dataclass
class SlippageCheckResult:
    passed: bool
    slippage_pct: float
    slippage_usd: float
    session_total: float = 0.0
    reason: Optional[str] = None

class SlippageBudget:
    """Enforces maximum slippage per trade and per session."""

    def __init__(
        self,
        max_slippage_per_trade_pct: float = 0.5,
        max_slippage_per_session_usd: float = 1000.0,
        alert_threshold_pct: float = 0.3
    ):
        self.max_per_trade_pct = max_slippage_per_trade_pct
        self.max_per_session = max_slippage_per_session_usd
        self.alert_threshold = alert_threshold_pct
        self.session_slippage = 0.0
        self.session_start = datetime.now(timezone.utc)

    def check_slippage(
        self,
        expected_price: float,
        actual_price: float,
        quantity: int,
        order_id: str
    ) -> SlippageCheckResult:
        """Check if slippage is within acceptable bounds."""
        if expected_price <= 0:
            return SlippageCheckResult(passed=True, slippage_pct=0, slippage_usd=0)

        slippage_pct = abs(actual_price - expected_price) / expected_price * 100
        slippage_usd = abs(actual_price - expected_price) * quantity * 100

        self.session_slippage += slippage_usd

        result = SlippageCheckResult(
            passed=True,
            slippage_pct=slippage_pct,
            slippage_usd=slippage_usd,
            session_total=self.session_slippage
        )

        if slippage_pct > self.max_per_trade_pct:
            logger.warning(
                "SLIPPAGE_EXCEEDED_TRADE_LIMIT",
                extra={
                    "order_id": order_id,
                    "slippage_pct": slippage_pct,
                    "limit_pct": self.max_per_trade_pct
                }
            )
            result.passed = False
            result.reason = f"Trade slippage {slippage_pct:.2f}% exceeds {self.max_per_trade_pct}%"

        if self.session_slippage > self.max_per_session:
            logger.error(
                "SLIPPAGE_EXCEEDED_SESSION_LIMIT",
                extra={
                    "session_total": self.session_slippage,
                    "limit": self.max_per_session
                }
            )
            result.passed = False
            result.reason = f"Session slippage ${self.session_slippage:.2f} exceeds ${self.max_per_session}"

        return result
```

---

## 6. Structural Refactoring (P2)

### 6.1 LLM Module Split Plan

**Current**: 116MB in flat structure
**Target**: Organized into 4 subdirectories

**Migration Steps**:

1. **Create new directory structure**:
```bash
mkdir -p llm/{core,analysis,logging}
```

2. **Move files** (with git mv to preserve history):
```bash
# Core
git mv llm/base.py llm/core/
git mv llm/providers.py llm/core/
git mv llm/ensemble.py llm/core/

# Analysis
git mv llm/sentiment.py llm/analysis/
git mv llm/sentiment_filter.py llm/analysis/  # Then split
git mv llm/news_analyzer.py llm/analysis/
git mv llm/earnings_analyzer.py llm/analysis/
git mv llm/entity_extractor.py llm/analysis/

# Logging
git mv llm/decision_logger.py llm/logging/
git mv llm/reasoning_logger.py llm/logging/
```

3. **Update imports** in all files that import from llm/

4. **Split sentiment_filter.py** (56KB):
   - `llm/analysis/sentiment_filter.py` (filtering logic)
   - `llm/analysis/sentiment_rules.py` (rule definitions)
   - `llm/analysis/sentiment_scoring.py` (scoring logic)

5. **Slim down `llm/__init__.py`**:
```python
# llm/__init__.py - Keep exports minimal
from llm.core.base import Sentiment, NewsItem, AgentBase
from llm.core.ensemble import create_ensemble, SentimentEnsemble
from llm.core.providers import create_provider

__all__ = [
    "Sentiment",
    "NewsItem",
    "AgentBase",
    "create_ensemble",
    "SentimentEnsemble",
    "create_provider",
]
```

### 6.2 Evaluation Module Consolidation

**Current**: 25+ competing frameworks
**Target**: 3 focused subdirectories

**Migration Steps**:

1. **Identify primary framework**: `evaluation_framework.py`

2. **Consolidate metrics**:
```python
# evaluation/core/metrics.py - Merge:
# - metrics.py
# - advanced_trading_metrics.py
# - agent_metrics.py (agent-specific only)
```

3. **Move walk-forward to backtesting**:
```bash
git mv evaluation/walk_forward_analysis.py backtesting/walk_forward.py
```

4. **Delete example files**, convert to tests:
```bash
rm evaluation/example_usage.py
rm evaluation/complete_evaluation_example.py
# Extract useful code to tests/evaluation/
```

### 6.3 Test Reorganization Plan

**Current**: 50+ files at root
**Target**: Mirror source structure

**Migration Script**:
```bash
#!/bin/bash
# scripts/reorganize_tests.sh

# Create directories
mkdir -p tests/{execution,llm,models,api,evaluation,observability}

# Move execution tests
git mv tests/test_execution.py tests/execution/
git mv tests/test_smart_execution.py tests/execution/
git mv tests/test_bot_managed_positions.py tests/execution/
git mv tests/test_arbitrage_executor.py tests/execution/

# Move LLM tests
git mv tests/test_llm_*.py tests/llm/

# Move model tests
git mv tests/test_risk_*.py tests/models/
git mv tests/test_circuit_breaker.py tests/models/

# Move API tests
git mv tests/test_api*.py tests/api/
git mv tests/test_websocket*.py tests/api/

# Create __init__.py files
touch tests/execution/__init__.py
touch tests/llm/__init__.py
touch tests/models/__init__.py
touch tests/api/__init__.py
```

---

## 7. Code Consolidation

### 7.1 Risk Validator Consolidation

**Problem**: Two implementations with different limits

| File | MAX_POSITION_SIZE_PCT | MAX_SINGLE_ORDER_VALUE |
|------|----------------------|------------------------|
| `.claude/hooks/trading/risk_validator.py` | 0.25 | 50,000 |
| `docs/templates/.../risk_validator.py` | 0.02 | 5,000 |

**Solution**: Single source of truth

```python
# models/risk_limits.py
"""
Single source of truth for all risk limits.

DO NOT DUPLICATE THESE VALUES ELSEWHERE.
Import from this module only.
"""
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class RiskLimits:
    """Immutable risk limits configuration."""

    # Position sizing
    max_position_size_pct: float = 0.25
    max_single_order_value: float = 50_000.0
    max_daily_orders: int = 100

    # Loss limits
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    max_consecutive_losses: int = 5

    # Blocked symbols
    blocked_symbols: frozenset = frozenset({"GME", "AMC", "BBBY"})

    @classmethod
    def from_env(cls) -> "RiskLimits":
        """Load limits from environment, with defaults."""
        return cls(
            max_position_size_pct=float(
                os.environ.get("RISK_MAX_POSITION_PCT", "0.25")
            ),
            max_single_order_value=float(
                os.environ.get("RISK_MAX_ORDER_VALUE", "50000")
            ),
            max_daily_loss_pct=float(
                os.environ.get("RISK_MAX_DAILY_LOSS_PCT", "0.03")
            ),
        )

# Global instance
RISK_LIMITS = RiskLimits.from_env()
```

### 7.2 Retry Decorator Consolidation

**Problem**: Two implementations in different locations

**Solution**: Keep `utils/error_handling.py`, deprecate `models/retry_handler.py`

```python
# models/retry_handler.py (DEPRECATED)
"""
DEPRECATED: Use utils.error_handling.retry instead.
"""
import warnings
from utils.error_handling import retry as _retry

def retry_with_backoff(*args, **kwargs):
    warnings.warn(
        "retry_with_backoff is deprecated. Use utils.error_handling.retry instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _retry(*args, **kwargs)
```

### 7.3 MCP Server Base Pattern

**Problem**: 4+ MCP servers with duplicate initialization

**Solution**: Extract to proper base class in `mcp/base_server.py`

```python
# mcp/base_server.py
"""Base MCP server with common initialization pattern."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any
from enum import Enum

class ToolCategory(Enum):
    """Canonical tool category definitions."""
    MARKET_DATA = "market_data"
    TRADING = "trading"
    PORTFOLIO = "portfolio"
    RISK = "risk"
    ANALYSIS = "analysis"
    BACKTEST = "backtest"

@dataclass
class ToolSchema:
    """MCP tool schema definition."""
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)

    def to_mcp_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required
            }
        }

class BaseMCPServer(ABC):
    """Base class for all MCP servers."""

    def __init__(self, config=None):
        self.config = config or {}
        self._tools: Dict[str, ToolSchema] = {}
        self._handlers: Dict[str, callable] = {}
        self._register_tools()

    @abstractmethod
    def _register_tools(self) -> None:
        """Register all tools. Override in subclass."""
        pass

    def register_tool(self, schema: ToolSchema, handler: callable) -> None:
        self._tools[schema.name] = schema
        self._handlers[schema.name] = handler
```

---

## 8. Missing Abstractions

### 8.1 ExecutorBase Abstract Class

**Location**: `execution/base.py` (NEW)

```python
# execution/base.py
"""Base classes for execution layer."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

class ExecutionStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"

@dataclass
class ExecutionResult:
    """Standard result for all execution operations."""
    success: bool
    status: ExecutionStatus
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    fill_quantity: Optional[int] = None
    slippage_bps: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ExecutorBase(ABC):
    """Abstract base class for all executors."""

    def __init__(self, algorithm, config: Dict[str, Any] = None):
        self.algorithm = algorithm
        self.config = config or {}
        self._execution_count = 0
        self._success_count = 0

    @abstractmethod
    async def execute(self, **kwargs) -> ExecutionResult:
        """Execute the strategy. Override in subclass."""
        pass

    @abstractmethod
    def validate(self, **kwargs) -> bool:
        """Validate before execution. Override in subclass."""
        pass

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self._execution_count == 0:
            return 0.0
        return self._success_count / self._execution_count

    def _record_execution(self, result: ExecutionResult) -> None:
        """Record execution for metrics."""
        self._execution_count += 1
        if result.success:
            self._success_count += 1
```

### 8.2 AnalyzerBase Abstract Class

**Location**: `llm/core/base.py` (extend existing)

```python
# Add to llm/core/base.py
class AnalyzerBase(ABC):
    """Abstract base class for all analyzers."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._analysis_count = 0

    @abstractmethod
    def analyze(self, data: Any) -> Dict[str, Any]:
        """Perform analysis. Override in subclass."""
        pass

    @abstractmethod
    def get_confidence(self) -> float:
        """Return confidence in analysis (0-1)."""
        pass
```

### 8.3 ScannerBase Abstract Class

**Location**: `scanners/base.py` (NEW)

```python
# scanners/base.py
"""Base classes for market scanners."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class ScanResult:
    """Standard result for scanner operations."""
    symbol: str
    score: float
    signal: str  # "buy", "sell", "hold"
    confidence: float
    metadata: Dict[str, Any]

class ScannerBase(ABC):
    """Abstract base class for all scanners."""

    @abstractmethod
    def scan(self, symbols: List[str]) -> List[ScanResult]:
        """Scan symbols for opportunities."""
        pass

    @abstractmethod
    def filter(self, results: List[ScanResult], min_score: float) -> List[ScanResult]:
        """Filter results by minimum score."""
        pass
```

### 8.4 MonitorBase Abstract Class

**Location**: `observability/monitoring/base.py` (extend existing)

```python
# observability/monitoring/base.py
class MonitorBase(ABC):
    """Abstract base class for all monitors."""

    @abstractmethod
    def check(self) -> Dict[str, Any]:
        """Run monitoring check."""
        pass

    @abstractmethod
    def get_status(self) -> str:
        """Get current status: healthy, degraded, unhealthy."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        pass
```

---

## 9. Logging Standards

### 9.1 Logger Configuration

```python
# utils/logging_config.py
"""Centralized logging configuration."""
import logging
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict

class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter for production."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename",
                "funcName", "levelname", "levelno", "lineno",
                "module", "msecs", "pathname", "process",
                "processName", "relativeCreated", "stack_info",
                "exc_info", "exc_text", "thread", "threadName",
                "message"
            ):
                log_entry[key] = value

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)

def configure_logging(level: str = "INFO", structured: bool = True) -> None:
    """Configure logging for the application."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if structured:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))

    root_logger.addHandler(handler)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
```

### 9.2 Log Event Categories

| Category | Format | Examples |
|----------|--------|----------|
| **Trade Events** | `TRADE_{ACTION}` | `TRADE_OPENED`, `TRADE_CLOSED`, `TRADE_MODIFIED` |
| **Order Events** | `ORDER_{ACTION}` | `ORDER_SUBMITTED`, `ORDER_FILLED`, `ORDER_CANCELLED` |
| **Risk Events** | `RISK_{TYPE}` | `RISK_LIMIT_EXCEEDED`, `RISK_CHECK_PASSED` |
| **Position Events** | `POSITION_{ACTION}` | `POSITION_OPENED`, `POSITION_ROLLED` |
| **System Events** | `SYSTEM_{TYPE}` | `SYSTEM_STARTUP`, `SYSTEM_SHUTDOWN` |
| **Auth Events** | `AUTH_{ACTION}` | `AUTH_SUCCESS`, `AUTH_FAILED` |
| **Error Events** | `{COMPONENT}_FAILED` | `POSITION_SAVE_FAILED`, `ORDER_SUBMIT_FAILED` |

### 9.3 Required Log Fields by Event Type

```python
# Trade Events
logger.info(
    "TRADE_OPENED",
    extra={
        "trade_id": str,          # Required
        "symbol": str,            # Required
        "side": str,              # "buy" or "sell"
        "quantity": int,          # Required
        "price": float,           # Required
        "order_id": str,          # Required
        "strategy": str,          # Which strategy initiated
        "confidence": float,      # AI confidence (0-1)
        "reasoning": str,         # Brief explanation
    }
)

# Order Events
logger.info(
    "ORDER_FILLED",
    extra={
        "order_id": str,          # Required
        "fill_price": float,      # Required
        "fill_quantity": int,     # Required
        "remaining_quantity": int,
        "slippage_pct": float,
        "fill_time_ms": int,
    }
)

# Risk Events
logger.warning(
    "RISK_LIMIT_EXCEEDED",
    extra={
        "limit_type": str,        # Which limit
        "current_value": float,
        "limit_value": float,
        "action_taken": str,
        "position_id": str,
    }
)
```

### 9.4 Log Levels Usage

| Level | Use Case | Example |
|-------|----------|---------|
| **DEBUG** | Detailed diagnostics | Greeks calculations, price updates |
| **INFO** | Normal operations | Trade opened, order filled |
| **WARNING** | Recoverable issues | Retry attempt, approaching limit |
| **ERROR** | Failures requiring attention | Order failed, position save failed |
| **CRITICAL** | System-wide failures | Circuit breaker tripped, auth down |

---

## 10. Audit Trail Implementation

### 10.1 Audit Event Schema

```python
# models/audit.py
"""Audit trail data structures."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import uuid
import json

class AuditAction(Enum):
    """Types of auditable actions."""
    # Trading
    TRADE_OPEN = "trade.open"
    TRADE_CLOSE = "trade.close"
    TRADE_MODIFY = "trade.modify"

    # Orders
    ORDER_SUBMIT = "order.submit"
    ORDER_CANCEL = "order.cancel"
    ORDER_FILL = "order.fill"

    # Risk
    RISK_OVERRIDE = "risk.override"
    RISK_LIMIT_CHANGE = "risk.limit_change"
    CIRCUIT_BREAKER_TRIP = "risk.circuit_breaker_trip"
    CIRCUIT_BREAKER_RESET = "risk.circuit_breaker_reset"

    # Configuration
    CONFIG_CHANGE = "config.change"
    STRATEGY_ENABLE = "config.strategy_enable"
    STRATEGY_DISABLE = "config.strategy_disable"

    # Authentication
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_FAILED = "auth.failed"

    # System
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    MANUAL_INTERVENTION = "system.manual_intervention"

@dataclass
class AuditEntry:
    """Single audit trail entry."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    action: AuditAction
    description: str

    actor: str  # "system", "user:email", "agent:name"
    actor_type: str  # "system", "human", "ai_agent"

    resource_type: str  # "trade", "order", "config"
    resource_id: str

    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None

    reason: Optional[str] = None
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "description": self.description,
            "actor": self.actor,
            "actor_type": self.actor_type,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "reason": self.reason,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
```

### 10.2 Audit Logger

```python
# utils/audit_logger.py
"""Audit trail logging service."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import contextmanager
import threading

from models.audit import AuditEntry, AuditAction

logger = logging.getLogger(__name__)

class AuditLogger:
    """Thread-safe audit trail logger."""

    def __init__(self, audit_dir: str = "logs/audit", retention_days: int = 90):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._correlation_id: Optional[str] = None

    @contextmanager
    def correlation_context(self, correlation_id: str):
        """Set correlation ID for all audit entries in this context."""
        old_id = self._correlation_id
        self._correlation_id = correlation_id
        try:
            yield
        finally:
            self._correlation_id = old_id

    def log(
        self,
        action: AuditAction,
        description: str,
        actor: str,
        actor_type: str,
        resource_type: str,
        resource_id: str,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEntry:
        """Log an auditable action."""
        entry = AuditEntry(
            action=action,
            description=description,
            actor=actor,
            actor_type=actor_type,
            resource_type=resource_type,
            resource_id=resource_id,
            before_state=before_state,
            after_state=after_state,
            reason=reason,
            correlation_id=self._correlation_id,
            metadata=metadata or {}
        )

        self._write_to_file(entry)

        logger.info(f"AUDIT_{action.name}", extra=entry.to_dict())

        return entry

    def _write_to_file(self, entry: AuditEntry) -> None:
        """Write entry to date-partitioned audit file."""
        date_str = entry.timestamp.strftime("%Y-%m-%d")
        file_path = self.audit_dir / f"audit_{date_str}.jsonl"

        with self._lock:
            with open(file_path, "a") as f:
                f.write(entry.to_json() + "\n")

    # Convenience methods
    def log_trade_open(self, trade_id: str, symbol: str, side: str,
                       quantity: int, price: float, actor: str,
                       actor_type: str, **kwargs) -> AuditEntry:
        return self.log(
            action=AuditAction.TRADE_OPEN,
            description=f"Opened {side} {quantity} {symbol} @ {price}",
            actor=actor,
            actor_type=actor_type,
            resource_type="trade",
            resource_id=trade_id,
            after_state={"symbol": symbol, "side": side, "quantity": quantity, "price": price},
            metadata=kwargs
        )

    def log_circuit_breaker_trip(self, trigger: str, trigger_value: float,
                                  threshold: float) -> AuditEntry:
        return self.log(
            action=AuditAction.CIRCUIT_BREAKER_TRIP,
            description=f"Circuit breaker tripped: {trigger}={trigger_value} > {threshold}",
            actor="system",
            actor_type="system",
            resource_type="circuit_breaker",
            resource_id="main",
            after_state={"status": "tripped", "trigger": trigger, "value": trigger_value}
        )

# Global instance
audit_logger = AuditLogger()
```

### 10.3 Audit File Format

**Location**: `logs/audit/audit_YYYY-MM-DD.jsonl`

**Format**: JSON Lines (one JSON object per line)

```json
{"id":"550e8400...","timestamp":"2025-12-06T10:30:00Z","action":"trade.open","description":"Opened buy 10 SPY @ 450.25","actor":"agent:momentum_trader","actor_type":"ai_agent","resource_type":"trade","resource_id":"trade_12345",...}
{"id":"550e8401...","timestamp":"2025-12-06T10:30:05Z","action":"order.fill","description":"Order filled: 10 SPY @ 450.30",...}
```

### 10.4 Audit Retention Policy

- **Default retention**: 90 days
- **Archive before deletion**: Optional S3 backup
- **Compliance**: SOX (7 years for financial), PCI DSS (12 months)

---

## 11. Testing Requirements

### 11.1 Test Coverage Targets

| Component | Current | Target | Priority |
|-----------|---------|--------|----------|
| P0 Critical Fixes | 0% | 100% | Immediate |
| Execution Layer | ~60% | 90% | High |
| Risk Management | ~70% | 95% | High |
| API Endpoints | ~50% | 85% | Medium |
| LLM Integration | ~40% | 75% | Medium |

### 11.2 Required Tests for P0 Fixes

```python
# tests/test_critical_fixes.py
"""Tests for P0 critical fixes."""
import pytest
from execution.bot_managed_positions import calculate_close_quantity
from execution.smart_execution import update_average_fill_price
from api.rest_server import ThreadSafeRef

class TestPositionSizingFix:
    """Test cases for position sizing truncation fix."""

    def test_small_position_skip_partial_close(self):
        """1-contract position should skip 30% profit-take."""
        result = calculate_close_quantity(1, 0.30)
        assert result == 0, "Should skip profit-take for small positions"

    def test_small_position_full_close(self):
        assert calculate_close_quantity(1, 1.00) == 1

    def test_medium_position_partial_close(self):
        assert calculate_close_quantity(10, 0.30) == 3

    @pytest.mark.parametrize("qty,pct,expected", [
        (1, 0.10, 0),
        (1, 0.50, 0),
        (2, 0.50, 1),
        (10, 0.10, 1),
    ])
    def test_edge_cases(self, qty, pct, expected):
        assert calculate_close_quantity(qty, pct) == expected


class TestFillPriceCalculationFix:
    """Test cases for fill price calculation fix."""

    def test_equal_fills(self):
        order = MockOrder(average_fill_price=100.0, filled_quantity=10)
        result = update_average_fill_price(order, 110.0, 10)
        assert result == 105.0
        assert order.filled_quantity == 20


class TestThreadSafeRef:
    """Test cases for thread-safe reference wrapper."""

    def test_set_and_get(self):
        ref = ThreadSafeRef("test")
        ref.set("value")
        assert ref.get() == "value"

    def test_double_set_raises(self):
        ref = ThreadSafeRef("test")
        ref.set("value1")
        with pytest.raises(RuntimeError):
            ref.set("value2")

    def test_concurrent_access(self):
        import threading
        ref = ThreadSafeRef("test")
        ref.set({"counter": 0})
        errors = []

        def increment():
            try:
                with ref.access() as value:
                    value["counter"] += 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert ref.get()["counter"] == 100
```

### 11.3 Integration Test Requirements

```python
# tests/integration/test_order_flow.py
@pytest.mark.integration
class TestOrderFlow:
    """End-to-end order flow tests."""

    async def test_order_submission_with_validation(self):
        """Order goes through validation before submission."""
        order = create_test_order(symbol="SPY", quantity=10)
        result = await submit_order(order)

        assert result.validated is True
        assert result.order_id is not None
        assert audit_logger.last_entry.action == AuditAction.ORDER_SUBMIT

    async def test_order_rejected_by_risk_check(self):
        """Order rejected when exceeding risk limits."""
        order = create_test_order(symbol="SPY", quantity=10000)
        result = await submit_order(order)

        assert result.validated is False
        assert "exceeds limit" in result.rejection_reason
```

---

## 12. Best Practices

### 12.1 Import Guidelines

```python
# GOOD: Import from canonical locations
from models.risk_limits import RISK_LIMITS
from utils.error_handling import retry
from utils.logging_config import get_logger

# BAD: Import from deprecated or duplicate locations
from models.retry_handler import retry_with_backoff  # Deprecated!
from docs.templates.risk_validator import RISK_LIMITS  # Duplicate!
```

### 12.2 Error Handling Guidelines

```python
# GOOD: Specific exceptions with context
class PositionSizingError(TradingError):
    """Raised when position sizing calculation fails."""
    pass

try:
    quantity = calculate_quantity(...)
except PositionSizingError as e:
    logger.error("POSITION_SIZING_FAILED", extra={"error": str(e)})
    raise

# BAD: Bare exceptions
try:
    quantity = calculate_quantity(...)
except:
    pass
```

### 12.3 Type Hints (Python 3.8+ Compatible)

```python
# GOOD: Complete type hints using List, Dict, Optional
from typing import Optional, List, Dict

def calculate_pnl(
    entry_price: float,
    exit_price: float,
    quantity: int,
    fees: float = 0.0
) -> float:
    """Calculate P&L for a trade."""
    return (exit_price - entry_price) * quantity - fees
```

### 12.4 Dataclass Best Practices

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class Trade:
    """Represents a completed trade."""
    id: str
    symbol: str
    entry_price: float
    quantity: int
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None

    # Mutable defaults use field()
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    @property
    def pnl(self) -> Optional[float]:
        if self.is_open:
            return None
        return (self.exit_price - self.entry_price) * self.quantity
```

---

## 13. Rollback Procedures

### 13.1 Pre-Rollback Checklist

- [ ] Confirm all positions are closed or manually managed
- [ ] Save current state to backup location
- [ ] Document what triggered the rollback
- [ ] Notify team of rollback initiation

### 13.2 Rollback Commands

```bash
# Tag current state before rollback
git tag rollback-from-$(date +%Y%m%d-%H%M%S)

# Rollback to pre-refactor state
git checkout pre-refactor-v1.x

# Restore dependencies
pip install -r requirements.txt

# Verify tests pass
pytest tests/ -v

# Restart services
./scripts/restart_services.sh
```

### 13.3 Post-Rollback Documentation

Create incident report:

```markdown
# Rollback Incident Report

**Date**: YYYY-MM-DD HH:MM
**Rollback From**: <commit-hash>
**Rollback To**: <commit-hash>

## Trigger
- What caused the rollback?

## Impact
- What functionality was affected?
- Were any trades impacted?

## Root Cause
- Why did the refactored code fail?

## Prevention
- How do we prevent this in the future?

## Next Steps
- [ ] Fix identified issues
- [ ] Add missing tests
- [ ] Re-attempt refactoring
```

---

## 14. Post-Refactoring Validation

### 14.1 Validation Checklist

- [ ] All 541+ tests passing: `pytest tests/ -v`
- [ ] No new linting errors: `ruff check .`
- [ ] Type checking passes: `mypy .`
- [ ] Critical fixes have test coverage
- [ ] Audit logging functional: `tail -f logs/audit/audit_$(date +%Y-%m-%d).jsonl`
- [ ] API endpoints responsive: `curl http://localhost:8000/health`
- [ ] Paper trading successful for 1 session

### 14.2 Performance Benchmarks

| Metric | Before | After | Acceptable |
|--------|--------|-------|------------|
| Order submission latency | <100ms | <100ms | <200ms |
| Position update latency | <50ms | <50ms | <100ms |
| Memory usage (idle) | <500MB | <500MB | <1GB |
| API response time (p95) | <200ms | <200ms | <500ms |

### 14.3 Sign-Off Requirements

- [ ] Developer sign-off: All changes reviewed
- [ ] Code review: PR approved
- [ ] QA sign-off: All test scenarios passed
- [ ] Risk sign-off: Risk management verified
- [ ] Deployment sign-off: Paper trading successful

---

## 15. Implementation Phases

### Phase 1: Critical Fixes (Days 1-3) - IMMEDIATE

| Task | File | Priority | Tests |
|------|------|----------|-------|
| Fix position sizing bug | `execution/bot_managed_positions.py` | P0 | Required |
| Fix fill price calculation | `execution/smart_execution.py` | P0 | Required |
| Fix global state race condition | `api/rest_server.py` | P0 | Required |
| Fix WebSocket TOCTOU | `api/websocket_handler.py` | P0 | Required |
| Fix authentication | `api/order_queue_api.py` | P0 | Required |
| Fix CORS wildcard | `api/rest_server.py` | P0 | Required |
| Implement position rolling | `execution/bot_managed_positions.py` | P0 | Required |

### Phase 2: High Priority (Week 1)

| Task | Priority |
|------|----------|
| Add rate limiting to API | P1 |
| Add Greeks tracking to positions | P1 |
| Add slippage budget enforcement | P1 |
| Eliminate silent failure patterns | P1 |
| Implement execution_quality_metrics.py | P1 |

### Phase 3: Code Consolidation (Week 1-2)

| Task | Priority |
|------|----------|
| Create models/risk_limits.py | P1 |
| Deprecate models/retry_handler.py | P1 |
| Create mcp/base_server.py | P2 |
| Consolidate duplicate code | P2 |

### Phase 4: Structural Refactoring (Week 2-3)

| Task | Priority |
|------|----------|
| Split llm/ module | P2 |
| Consolidate evaluation/ | P2 |
| Reorganize tests/ | P2 |
| Create base classes | P2 |

### Phase 5: Logging & Audit (Week 3)

| Task | Priority |
|------|----------|
| Implement utils/logging_config.py | P1 |
| Implement utils/audit_logger.py | P1 |
| Implement models/audit.py | P1 |
| Standardize log events | P2 |

### Phase 6: Documentation & Cleanup (Week 4)

| Task | Priority |
|------|----------|
| Update all imports | P2 |
| Document base classes | P2 |
| Clean deprecated code | P3 |
| Create docs/INDEX.md | P3 |

---

## 16. Appendix

### A. File Change Summary

| File | Action | Priority |
|------|--------|----------|
| `execution/bot_managed_positions.py` | Fix position sizing, add Greeks | P0 |
| `execution/smart_execution.py` | Fix fill calculation, add slippage | P0 |
| `api/rest_server.py` | Thread safety, CORS, rate limiting | P0 |
| `api/order_queue_api.py` | Add authentication | P0 |
| `api/websocket_handler.py` | Fix race condition | P0 |
| `models/risk_limits.py` | NEW - Single source of truth | P1 |
| `utils/logging_config.py` | NEW - Centralized logging | P1 |
| `utils/audit_logger.py` | NEW - Audit trail | P1 |
| `models/audit.py` | NEW - Audit data structures | P1 |
| `execution/base.py` | NEW - ExecutorBase | P2 |
| `scanners/base.py` | NEW - ScannerBase | P2 |

### B. Deprecated Files

| File | Replacement | Action |
|------|-------------|--------|
| `models/retry_handler.py` | `utils/error_handling.py` | Add deprecation warning |
| `docs/templates/.../risk_validator.py` | `models/risk_limits.py` | Delete after migration |
| `evaluation/example_usage.py` | Tests | Delete |
| `evaluation/complete_evaluation_example.py` | Tests | Delete |

### C. New Dependencies

```txt
# requirements.txt additions
slowapi>=0.1.8  # Rate limiting
```

### D. Environment Variables

```bash
# Required in production
TRADING_API_TOKEN=<secure-32-char-token>
CORS_ALLOWED_ORIGINS=https://yourdomain.com
ENVIRONMENT=production

# Optional
RISK_MAX_POSITION_PCT=0.25
RISK_MAX_ORDER_VALUE=50000
RISK_MAX_DAILY_LOSS_PCT=0.03
LOG_LEVEL=INFO
```

### E. Audit Trail: Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-06 | Agent Analysis | Initial REFACTORING_INSTRUCTIONS.md |
| 2.0.0 | 2025-12-06 | Merged Analysis | Comprehensive plan with structure analysis |

---

**Document Version**: 2.0.0
**Last Updated**: 2025-12-06
**Status**: Ready for Implementation
**Next Action**: Execute Phase 1 Critical Fixes

---

## Quick Reference: Priority Matrix

```
┌─────────────────────────────────────────────────────────────────┐
│                    REFACTORING PRIORITY MATRIX                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  URGENT + IMPORTANT (P0)          IMPORTANT (P1)                │
│  ┌─────────────────────────┐      ┌─────────────────────────┐   │
│  │ • Position sizing bug   │      │ • Rate limiting         │   │
│  │ • Fill price calc       │      │ • Greeks tracking       │   │
│  │ • Race conditions       │      │ • Slippage budget       │   │
│  │ • Auth/CORS security    │      │ • Silent failures       │   │
│  │ • Position rolling      │      │ • Code consolidation    │   │
│  └─────────────────────────┘      └─────────────────────────┘   │
│         DO FIRST (Days 1-3)             DO NEXT (Week 1)        │
│                                                                  │
│  URGENT (P2)                      NICE TO HAVE (P3)             │
│  ┌─────────────────────────┐      ┌─────────────────────────┐   │
│  │ • Split LLM module      │      │ • Clean deprecated/     │   │
│  │ • Consolidate eval      │      │ • docs/INDEX.md         │   │
│  │ • Reorganize tests      │      │ • Archive old research  │   │
│  │ • Base classes          │      │ • Script organization   │   │
│  │ • Logging standards     │      │                         │   │
│  └─────────────────────────┘      └─────────────────────────┘   │
│         SCHEDULE (Week 2-3)             BACKLOG (Week 4+)       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```
