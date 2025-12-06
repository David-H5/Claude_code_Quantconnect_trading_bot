# Refactoring Instructions: Autonomous AI Agent Trading Bot

*Version: 1.0.0*
*Generated: 2025-12-06*
*Status: Pre-Refactor Analysis Complete*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Pre-Refactoring Checklist](#2-pre-refactoring-checklist)
3. [Critical Fixes (P0)](#3-critical-fixes-p0)
4. [High Priority Fixes (P1)](#4-high-priority-fixes-p1)
5. [Code Consolidation](#5-code-consolidation)
6. [Best Practices](#6-best-practices)
7. [Logging Standards](#7-logging-standards)
8. [Audit Trail Requirements](#8-audit-trail-requirements)
9. [Testing Requirements](#9-testing-requirements)
10. [Rollback Procedures](#10-rollback-procedures)
11. [Post-Refactoring Validation](#11-post-refactoring-validation)
12. [Appendix](#12-appendix)

---

## 1. Executive Summary

### Current State Assessment

| Metric | Value | Status |
|--------|-------|--------|
| **Autonomy Level** | 7.5/10 | Semi-autonomous |
| **Critical Flaws** | 7 | Requires immediate action |
| **High Severity Flaws** | 15 | Plan for current sprint |
| **Code Duplications** | 9 categories | Consolidation needed |
| **Test Coverage** | 541 tests | Maintain during refactor |
| **Production Readiness** | Blocked | Critical fixes required |

### Refactoring Goals

1. **Safety**: Fix position sizing bug and race conditions before any live trading
2. **Reliability**: Eliminate silent failures and improve error handling
3. **Maintainability**: Consolidate duplicated code into single sources of truth
4. **Observability**: Standardize logging and implement comprehensive audit trails
5. **Security**: Address authentication and CORS vulnerabilities

### Estimated Effort

| Phase | Duration | Priority |
|-------|----------|----------|
| Critical Fixes (P0) | 2-3 days | Immediate |
| High Priority (P1) | 1 week | This sprint |
| Code Consolidation | 1 week | This sprint |
| Logging Overhaul | 3-4 days | Next sprint |
| Audit Trail Implementation | 3-4 days | Next sprint |

---

## 2. Pre-Refactoring Checklist

### Environment Setup

- [ ] Create dedicated refactoring branch: `git checkout -b refactor/v2.0-comprehensive`
- [ ] Ensure all 541 tests pass: `pytest tests/ -v`
- [ ] Backup current Object Store state
- [ ] Document current configuration values
- [ ] Disable any scheduled deployments

### Documentation Snapshot

- [ ] Export current PROJECT_STATUS.md metrics
- [ ] Screenshot current dashboard state
- [ ] Document all environment variables in use
- [ ] List all active MCP server configurations

### Communication

- [ ] Notify team of refactoring window
- [ ] Set trading bot to MANUAL mode (not autonomous)
- [ ] Close all open positions before major changes
- [ ] Document rollback point in git: `git tag pre-refactor-v1.x`

---

## 3. Critical Fixes (P0)

> **WARNING**: These issues can cause financial loss. Fix before ANY live trading.

### 3.1 Position Sizing Truncation Bug

**Location**: `execution/bot_managed_positions.py:400-402`

**Current Code (BROKEN)**:
```python
close_quantity = int(position.current_quantity * threshold.take_pct)
if close_quantity == 0:
    close_quantity = 1  # BUG: Closes 100% instead of intended %
```

**Impact**: For 1-contract position with 30% profit-taking target:
- `int(1 * 0.30) = 0` → Override to `1` → Closes 100% instead of 30%

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
        Quantity to close, respecting minimum tradeable unit

    Note:
        For positions where calculated close < min_close,
        we skip this profit-taking level to avoid over-closing.
    """
    calculated = int(current_quantity * take_pct)

    # If position too small for this profit-taking level, skip it
    if calculated < min_close and current_quantity > min_close:
        logger.info(
            f"Skipping profit-take: position={current_quantity}, "
            f"take_pct={take_pct}, calculated={calculated} < min={min_close}"
        )
        return 0  # Signal to skip this profit-taking level

    return max(calculated, min_close) if calculated > 0 else 0
```

**Test Case**:
```python
def test_position_sizing_small_positions():
    """Ensure small positions don't over-close."""
    # 1 contract at 30% should skip, not close 100%
    assert calculate_close_quantity(1, 0.30) == 0
    # 1 contract at 100% should close 1
    assert calculate_close_quantity(1, 1.00) == 1
    # 10 contracts at 30% should close 3
    assert calculate_close_quantity(10, 0.30) == 3
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

### 3.2 Position Rolling Not Implemented

**Location**: `execution/bot_managed_positions.py:446-475`

**Current Code (STUB)**:
```python
def _execute_roll(self, position: BotManagedPosition) -> Optional[ManagementAction]:
    """Execute position roll - close current and open new position."""
    # In a real implementation, this would:
    # 1. Close the current position
    # 2. Open a new position with later expiration
    # For now, just log and mark as rolled  # <-- NOT ACCEPTABLE
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

### 3.3 Race Conditions in Global State

**Location**: `api/rest_server.py:29-53`

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

### 3.4 WebSocket TOCTOU Race Condition

**Location**: `api/websocket_handler.py:201-205`

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

### 3.5 Fill Rate Calculation Error

**Location**: `execution/smart_execution.py:241-242`

**Current Code (MATHEMATICALLY WRONG)**:
```python
prev_value = order.average_fill_price * (filled_quantity - 1)
order.average_fill_price = (prev_value + fill_price) / filled_quantity
```

**Issue**: This calculates `(avg * (n-1) + new) / n` which is incorrect weighted average.

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

    Args:
        order: Order being filled
        new_fill_price: Price of this fill
        new_fill_quantity: Quantity in this fill

    Returns:
        New average fill price
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

### 3.6 Default Authentication Token

**Location**: `api/order_queue_api.py:612`

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

### 3.7 CORS Wildcard

**Location**: `api/rest_server.py:113-119`

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
    # Default to localhost only in development
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

## 4. High Priority Fixes (P1)

### 4.1 Silent Failure Pattern

**Locations**: Multiple files with `except: pass` or `except Exception: pass`

**Pattern to Find**:
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
    # Re-raise if critical, or handle gracefully
    if isinstance(e, (IOError, OSError)):
        raise PositionPersistenceError(f"Failed to save {position.id}") from e
```

---

### 4.2 Add Rate Limiting

**Location**: `api/rest_server.py`

**Add Dependency**:
```bash
pip install slowapi
```

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
    logger.info(
        "ORDER_SUBMITTED",
        extra={
            "client_ip": request.client.host,
            "order_type": order.order_type
        }
    )
    # ... rest of handler
```

---

### 4.3 Add Greeks Tracking to Positions

**Location**: `execution/bot_managed_positions.py`

**Add to Dataclass**:
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
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "iv": iv,
                "source": source
            }
        )

    @property
    def dollar_delta(self) -> float:
        """Position delta in dollar terms."""
        return self.delta * self.current_quantity * 100  # Options multiplier

    @property
    def daily_theta_decay(self) -> float:
        """Expected daily theta decay in dollars."""
        return self.theta * self.current_quantity * 100
```

---

### 4.4 Add Slippage Budget Enforcement

**Location**: `execution/smart_execution.py`

```python
class SlippageBudget:
    """Enforces maximum slippage per trade and per session."""

    def __init__(
        self,
        max_slippage_per_trade_pct: float = 0.5,  # 0.5%
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
        """
        Check if slippage is within acceptable bounds.

        Returns:
            SlippageCheckResult with passed status and details
        """
        if expected_price <= 0:
            return SlippageCheckResult(passed=True, slippage_pct=0, slippage_usd=0)

        slippage_pct = abs(actual_price - expected_price) / expected_price * 100
        slippage_usd = abs(actual_price - expected_price) * quantity * 100

        # Update session total
        self.session_slippage += slippage_usd

        result = SlippageCheckResult(
            passed=True,
            slippage_pct=slippage_pct,
            slippage_usd=slippage_usd,
            session_total=self.session_slippage
        )

        # Check per-trade limit
        if slippage_pct > self.max_per_trade_pct:
            logger.warning(
                "SLIPPAGE_EXCEEDED_TRADE_LIMIT",
                extra={
                    "order_id": order_id,
                    "slippage_pct": slippage_pct,
                    "limit_pct": self.max_per_trade_pct,
                    "expected": expected_price,
                    "actual": actual_price
                }
            )
            result.passed = False
            result.reason = f"Trade slippage {slippage_pct:.2f}% exceeds {self.max_per_trade_pct}%"

        # Check session limit
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

        # Alert on approaching threshold
        elif slippage_pct > self.alert_threshold:
            logger.info(
                "SLIPPAGE_ALERT",
                extra={
                    "order_id": order_id,
                    "slippage_pct": slippage_pct,
                    "threshold_pct": self.alert_threshold
                }
            )

        return result
```

---

## 5. Code Consolidation

### 5.1 Risk Validator Consolidation

**Problem**: Two implementations with different limits

| File | MAX_POSITION_SIZE_PCT | MAX_SINGLE_ORDER_VALUE |
|------|----------------------|------------------------|
| `.claude/hooks/trading/risk_validator.py` | 0.25 | 50,000 |
| `docs/templates/.../risk_validator.py` | 0.02 | 5,000 |

**Solution**: Single source of truth in `models/risk_limits.py`

```python
# models/risk_limits.py
"""
Single source of truth for all risk limits.

DO NOT DUPLICATE THESE VALUES ELSEWHERE.
Import from this module only.
"""
import os
from dataclasses import dataclass
from typing import Set

@dataclass(frozen=True)
class RiskLimits:
    """Immutable risk limits configuration."""

    # Position sizing
    max_position_size_pct: float = 0.25  # 25% of portfolio
    max_single_order_value: float = 50_000.0
    max_daily_orders: int = 100

    # Loss limits
    max_daily_loss_pct: float = 0.03  # 3%
    max_drawdown_pct: float = 0.10  # 10%
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


# Global instance - import this, don't create new instances
RISK_LIMITS = RiskLimits.from_env()
```

**Update Existing Files**:
```python
# In .claude/hooks/trading/risk_validator.py
from models.risk_limits import RISK_LIMITS

def validate_order(order_data: dict) -> dict:
    if order_value > RISK_LIMITS.max_single_order_value:
        return {"approved": False, "reason": "Order exceeds limit"}
```

---

### 5.2 Retry Decorator Consolidation

**Problem**: Two implementations in `utils/error_handling.py` and `models/retry_handler.py`

**Solution**: Keep `utils/error_handling.py`, deprecate `models/retry_handler.py`

```python
# utils/error_handling.py (CANONICAL)
"""
Canonical retry and error handling utilities.

Import from here, not from models/retry_handler.py (deprecated).
"""
import asyncio
import functools
import random
import time
from typing import Callable, Type, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def retry(
    max_attempts: int = 3,
    delay: float = 0.1,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts (including first try)
        delay: Initial delay between retries in seconds
        backoff_factor: Multiply delay by this factor each retry
        max_delay: Maximum delay cap
        exceptions: Exception types to catch and retry
        on_retry: Optional callback(exception, attempt_number)

    Example:
        @retry(max_attempts=3, exceptions=(ConnectionError,))
        def fetch_data():
            return requests.get(url)
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            "RETRY_EXHAUSTED",
                            extra={
                                "function": func.__name__,
                                "attempts": attempt,
                                "error": str(e)
                            }
                        )
                        raise

                    # Add jitter: 50% to 100% of delay
                    jittered_delay = current_delay * (0.5 + random.random() * 0.5)

                    logger.warning(
                        "RETRY_ATTEMPT",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt,
                            "next_delay": jittered_delay,
                            "error": str(e)
                        }
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(jittered_delay)
                    current_delay = min(current_delay * backoff_factor, max_delay)

            raise last_exception

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        raise

                    jittered_delay = current_delay * (0.5 + random.random() * 0.5)

                    logger.warning(
                        "RETRY_ATTEMPT_ASYNC",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt,
                            "next_delay": jittered_delay
                        }
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    await asyncio.sleep(jittered_delay)
                    current_delay = min(current_delay * backoff_factor, max_delay)

            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
```

```python
# models/retry_handler.py (DEPRECATED)
"""
DEPRECATED: Use utils.error_handling.retry instead.

This module is kept for backwards compatibility only.
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

---

### 5.3 MCP Server Base Pattern

**Problem**: 4+ MCP servers with duplicate initialization patterns

**Solution**: Extract to base class properly

```python
# mcp/base_server.py
"""Base MCP server with common initialization pattern."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

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
        """Convert to MCP-compatible schema format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required
            }
        }

@dataclass
class ServerConfig:
    """Common server configuration."""
    name: str
    version: str = "1.0.0"
    max_connections: int = 100
    timeout_seconds: float = 30.0

class BaseMCPServer(ABC):
    """
    Base class for all MCP servers.

    Subclasses should:
    1. Call super().__init__(config)
    2. Implement _register_tools() to add tools
    3. Implement tool handler methods
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig(name=self.__class__.__name__)
        self._tools: Dict[str, ToolSchema] = {}
        self._handlers: Dict[str, callable] = {}

        logger.info(
            "MCP_SERVER_INIT",
            extra={
                "server": self.config.name,
                "version": self.config.version
            }
        )

        self._register_tools()

        logger.info(
            "MCP_SERVER_READY",
            extra={
                "server": self.config.name,
                "tool_count": len(self._tools)
            }
        )

    @abstractmethod
    def _register_tools(self) -> None:
        """Register all tools. Override in subclass."""
        pass

    def register_tool(
        self,
        schema: ToolSchema,
        handler: callable
    ) -> None:
        """Register a tool with its handler."""
        self._tools[schema.name] = schema
        self._handlers[schema.name] = handler

        logger.debug(
            "TOOL_REGISTERED",
            extra={
                "server": self.config.name,
                "tool": schema.name,
                "category": schema.category.value
            }
        )

    async def handle_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """Handle a tool call with logging."""
        if tool_name not in self._handlers:
            raise ValueError(f"Unknown tool: {tool_name}")

        logger.info(
            "TOOL_CALL",
            extra={
                "server": self.config.name,
                "tool": tool_name,
                "args": arguments
            }
        )

        try:
            result = await self._handlers[tool_name](arguments)

            logger.info(
                "TOOL_CALL_SUCCESS",
                extra={
                    "server": self.config.name,
                    "tool": tool_name
                }
            )

            return result

        except Exception as e:
            logger.error(
                "TOOL_CALL_FAILED",
                extra={
                    "server": self.config.name,
                    "tool": tool_name,
                    "error": str(e)
                }
            )
            raise
```

---

## 6. Best Practices

### 6.1 Code Organization

```
project/
├── models/              # Data structures and business logic
│   ├── __init__.py
│   ├── risk_limits.py   # Single source of truth for limits
│   └── positions.py     # Position dataclasses
├── execution/           # Order execution logic
│   ├── __init__.py
│   └── smart_execution.py
├── utils/               # Shared utilities
│   ├── __init__.py
│   ├── error_handling.py  # Canonical retry/error utilities
│   └── logging_config.py  # Logging setup
└── api/                 # External interfaces
    ├── __init__.py
    └── rest_server.py
```

### 6.2 Import Guidelines

```python
# GOOD: Import from canonical locations
from models.risk_limits import RISK_LIMITS
from utils.error_handling import retry
from utils.logging_config import get_logger

# BAD: Import from deprecated or duplicate locations
from models.retry_handler import retry_with_backoff  # Deprecated!
from docs.templates.risk_validator import RISK_LIMITS  # Duplicate!
```

### 6.3 Error Handling Guidelines

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

### 6.4 Type Hints

```python
# GOOD: Complete type hints
from typing import Optional, List, Dict
from datetime import datetime

def calculate_pnl(
    entry_price: float,
    exit_price: float,
    quantity: int,
    fees: float = 0.0
) -> float:
    """Calculate P&L for a trade."""
    return (exit_price - entry_price) * quantity - fees

# BAD: Missing type hints
def calculate_pnl(entry, exit, qty, fees=0):
    return (exit - entry) * qty - fees
```

### 6.5 Dataclass Best Practices

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class Trade:
    """
    Represents a completed trade.

    Attributes:
        id: Unique trade identifier
        symbol: Trading symbol
        entry_price: Entry price per unit
        exit_price: Exit price per unit (None if still open)
        quantity: Number of units
        entry_time: When position was opened
        exit_time: When position was closed (None if still open)
    """
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
        """Check if trade is still open."""
        return self.exit_price is None

    @property
    def pnl(self) -> Optional[float]:
        """Calculate P&L if trade is closed."""
        if self.is_open:
            return None
        return (self.exit_price - self.entry_price) * self.quantity
```

---

## 7. Logging Standards

### 7.1 Logger Configuration

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
        if hasattr(record, "__dict__"):
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

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)

def configure_logging(
    level: str = "INFO",
    structured: bool = True
) -> None:
    """Configure logging for the application."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
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

### 7.2 Log Event Categories

| Category | Format | Example |
|----------|--------|---------|
| **Trade Events** | `TRADE_{ACTION}` | `TRADE_OPENED`, `TRADE_CLOSED` |
| **Order Events** | `ORDER_{ACTION}` | `ORDER_SUBMITTED`, `ORDER_FILLED` |
| **Risk Events** | `RISK_{TYPE}` | `RISK_LIMIT_EXCEEDED`, `RISK_CHECK_PASSED` |
| **System Events** | `SYSTEM_{TYPE}` | `SYSTEM_STARTUP`, `SYSTEM_SHUTDOWN` |
| **Auth Events** | `AUTH_{ACTION}` | `AUTH_SUCCESS`, `AUTH_FAILED` |
| **Error Events** | `{COMPONENT}_FAILED` | `POSITION_SAVE_FAILED` |

### 7.3 Required Log Fields by Event Type

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
        "fill_time_ms": int,      # Time to fill
    }
)

# Risk Events
logger.warning(
    "RISK_LIMIT_EXCEEDED",
    extra={
        "limit_type": str,        # Which limit
        "current_value": float,   # Current value
        "limit_value": float,     # Configured limit
        "action_taken": str,      # What was done
        "position_id": str,       # If applicable
    }
)

# Error Events
logger.error(
    "COMPONENT_FAILED",
    extra={
        "component": str,         # Which component
        "operation": str,         # What operation
        "error": str,             # Error message
        "error_type": str,        # Exception type
        "recoverable": bool,      # Can we retry?
        "context": dict,          # Additional context
    }
)
```

### 7.4 Log Levels Usage

| Level | Use Case | Example |
|-------|----------|---------|
| **DEBUG** | Detailed diagnostic info | Greeks calculations, price updates |
| **INFO** | Normal operations | Trade opened, order filled |
| **WARNING** | Recoverable issues | Retry attempt, approaching limit |
| **ERROR** | Failures requiring attention | Order failed, position save failed |
| **CRITICAL** | System-wide failures | Circuit breaker tripped, auth system down |

---

## 8. Audit Trail Requirements

### 8.1 Audit Event Schema

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
    ORDER_MODIFY = "order.modify"
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

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # What happened
    action: AuditAction
    description: str

    # Who did it
    actor: str  # "system", "user:email", "agent:name"
    actor_type: str  # "system", "human", "ai_agent"

    # What was affected
    resource_type: str  # "trade", "order", "config", etc.
    resource_id: str

    # Change details
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None

    # Context
    reason: Optional[str] = None
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
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
            "ip_address": self.ip_address,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())
```

### 8.2 Audit Logger

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
    """
    Thread-safe audit trail logger.

    Writes to both structured log and append-only audit file.
    """

    def __init__(
        self,
        audit_dir: str = "logs/audit",
        retention_days: int = 90
    ):
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
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEntry:
        """
        Log an auditable action.

        Returns:
            The created AuditEntry
        """
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
            ip_address=ip_address,
            session_id=session_id,
            correlation_id=self._correlation_id,
            metadata=metadata or {}
        )

        # Write to append-only file (thread-safe)
        self._write_to_file(entry)

        # Also log to standard logger for real-time monitoring
        logger.info(
            f"AUDIT_{action.name}",
            extra=entry.to_dict()
        )

        return entry

    def _write_to_file(self, entry: AuditEntry) -> None:
        """Write entry to date-partitioned audit file."""
        date_str = entry.timestamp.strftime("%Y-%m-%d")
        file_path = self.audit_dir / f"audit_{date_str}.jsonl"

        with self._lock:
            with open(file_path, "a") as f:
                f.write(entry.to_json() + "\n")

    # Convenience methods for common actions

    def log_trade_open(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        actor: str,
        actor_type: str,
        strategy: Optional[str] = None,
        confidence: Optional[float] = None,
        reasoning: Optional[str] = None
    ) -> AuditEntry:
        """Log a trade opening."""
        return self.log(
            action=AuditAction.TRADE_OPEN,
            description=f"Opened {side} {quantity} {symbol} @ {price}",
            actor=actor,
            actor_type=actor_type,
            resource_type="trade",
            resource_id=trade_id,
            after_state={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "status": "open"
            },
            metadata={
                "strategy": strategy,
                "confidence": confidence,
                "reasoning": reasoning
            }
        )

    def log_risk_override(
        self,
        override_id: str,
        limit_type: str,
        original_value: Any,
        override_value: Any,
        actor: str,
        reason: str
    ) -> AuditEntry:
        """Log a risk limit override."""
        return self.log(
            action=AuditAction.RISK_OVERRIDE,
            description=f"Overrode {limit_type}: {original_value} -> {override_value}",
            actor=actor,
            actor_type="human",  # Only humans should override
            resource_type="risk_limit",
            resource_id=override_id,
            before_state={"value": original_value},
            after_state={"value": override_value},
            reason=reason
        )

    def log_circuit_breaker_trip(
        self,
        trigger: str,
        trigger_value: float,
        threshold: float
    ) -> AuditEntry:
        """Log circuit breaker activation."""
        return self.log(
            action=AuditAction.CIRCUIT_BREAKER_TRIP,
            description=f"Circuit breaker tripped: {trigger}={trigger_value} > {threshold}",
            actor="system",
            actor_type="system",
            resource_type="circuit_breaker",
            resource_id="main",
            after_state={
                "status": "tripped",
                "trigger": trigger,
                "trigger_value": trigger_value,
                "threshold": threshold
            }
        )


# Global instance
audit_logger = AuditLogger()
```

### 8.3 Audit File Format

**Location**: `logs/audit/audit_YYYY-MM-DD.jsonl`

**Format**: JSON Lines (one JSON object per line)

```json
{"id":"550e8400-e29b-41d4-a716-446655440000","timestamp":"2025-12-06T10:30:00Z","action":"trade.open","description":"Opened buy 10 SPY @ 450.25","actor":"agent:momentum_trader","actor_type":"ai_agent","resource_type":"trade","resource_id":"trade_12345","before_state":null,"after_state":{"symbol":"SPY","side":"buy","quantity":10,"price":450.25,"status":"open"},"reason":null,"ip_address":null,"session_id":"sess_abc123","correlation_id":"corr_xyz789","metadata":{"strategy":"momentum","confidence":0.85,"reasoning":"RSI oversold with bullish divergence"}}
{"id":"550e8400-e29b-41d4-a716-446655440001","timestamp":"2025-12-06T10:30:05Z","action":"order.fill","description":"Order filled: 10 SPY @ 450.30","actor":"system","actor_type":"system","resource_type":"order","resource_id":"order_67890","before_state":{"status":"pending","filled_quantity":0},"after_state":{"status":"filled","filled_quantity":10,"average_price":450.30},"reason":null,"ip_address":null,"session_id":"sess_abc123","correlation_id":"corr_xyz789","metadata":{"slippage_pct":0.011}}
```

### 8.4 Audit Retention Policy

```python
# utils/audit_retention.py
"""Audit log retention management."""
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def cleanup_old_audits(
    audit_dir: str = "logs/audit",
    retention_days: int = 90
) -> int:
    """
    Remove audit files older than retention period.

    Returns:
        Number of files deleted
    """
    audit_path = Path(audit_dir)
    cutoff = datetime.now() - timedelta(days=retention_days)
    deleted = 0

    for file in audit_path.glob("audit_*.jsonl"):
        try:
            # Parse date from filename
            date_str = file.stem.replace("audit_", "")
            file_date = datetime.strptime(date_str, "%Y-%m-%d")

            if file_date < cutoff:
                # Archive to cold storage before deletion (optional)
                # archive_to_s3(file)

                file.unlink()
                deleted += 1

                logger.info(
                    "AUDIT_FILE_DELETED",
                    extra={
                        "file": str(file),
                        "file_date": date_str,
                        "retention_days": retention_days
                    }
                )

        except Exception as e:
            logger.error(
                "AUDIT_CLEANUP_FAILED",
                extra={
                    "file": str(file),
                    "error": str(e)
                }
            )

    return deleted
```

---

## 9. Testing Requirements

### 9.1 Test Coverage Targets

| Component | Current | Target | Priority |
|-----------|---------|--------|----------|
| Critical fixes (P0) | 0% | 100% | Immediate |
| Execution layer | ~60% | 90% | High |
| Risk management | ~70% | 95% | High |
| API endpoints | ~50% | 85% | Medium |
| LLM integration | ~40% | 75% | Medium |

### 9.2 Required Test Cases for P0 Fixes

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
        """1-contract position should allow 100% close."""
        result = calculate_close_quantity(1, 1.00)
        assert result == 1

    def test_medium_position_partial_close(self):
        """10-contract position at 30% should close 3."""
        result = calculate_close_quantity(10, 0.30)
        assert result == 3

    def test_large_position_partial_close(self):
        """100-contract position at 25% should close 25."""
        result = calculate_close_quantity(100, 0.25)
        assert result == 25

    @pytest.mark.parametrize("qty,pct,expected", [
        (1, 0.10, 0),   # Skip
        (1, 0.50, 0),   # Skip
        (2, 0.50, 1),   # Close 1
        (3, 0.33, 1),   # Close 1
        (5, 0.20, 1),   # Close 1
        (10, 0.10, 1),  # Close 1
    ])
    def test_edge_cases(self, qty, pct, expected):
        """Test various edge cases."""
        result = calculate_close_quantity(qty, pct)
        assert result == expected


class TestFillPriceCalculationFix:
    """Test cases for fill price calculation fix."""

    def test_first_fill(self):
        """First fill should set average to fill price."""
        order = MockOrder(average_fill_price=0, filled_quantity=0)
        result = update_average_fill_price(order, 100.0, 10)
        assert result == 100.0
        assert order.filled_quantity == 10

    def test_equal_fills(self):
        """Two equal fills at different prices."""
        order = MockOrder(average_fill_price=100.0, filled_quantity=10)
        result = update_average_fill_price(order, 110.0, 10)
        # (100*10 + 110*10) / 20 = 105
        assert result == 105.0
        assert order.filled_quantity == 20

    def test_unequal_fills(self):
        """Fills with different quantities."""
        order = MockOrder(average_fill_price=100.0, filled_quantity=20)
        result = update_average_fill_price(order, 120.0, 10)
        # (100*20 + 120*10) / 30 = 3200/30 = 106.67
        assert abs(result - 106.67) < 0.01
        assert order.filled_quantity == 30


class TestThreadSafeRef:
    """Test cases for thread-safe reference wrapper."""

    def test_set_and_get(self):
        """Basic set and get."""
        ref = ThreadSafeRef("test")
        ref.set("value")
        assert ref.get() == "value"

    def test_double_set_raises(self):
        """Cannot set twice."""
        ref = ThreadSafeRef("test")
        ref.set("value1")
        with pytest.raises(RuntimeError):
            ref.set("value2")

    def test_get_before_set_timeout(self):
        """Get before set should timeout."""
        ref = ThreadSafeRef("test")
        with pytest.raises(RuntimeError):
            ref.get(timeout=0.1)

    def test_concurrent_access(self):
        """Thread safety under concurrent access."""
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

### 9.3 Integration Test Requirements

```python
# tests/integration/test_order_flow.py
"""Integration tests for order flow."""
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.integration
class TestOrderFlow:
    """End-to-end order flow tests."""

    async def test_order_submission_with_validation(self):
        """Order goes through validation before submission."""
        # Setup
        order = create_test_order(symbol="SPY", quantity=10, price=450.0)

        # Execute
        result = await submit_order(order)

        # Verify
        assert result.validated is True
        assert result.order_id is not None
        assert audit_logger.last_entry.action == AuditAction.ORDER_SUBMIT

    async def test_order_rejected_by_risk_check(self):
        """Order rejected when exceeding risk limits."""
        # Setup - order exceeding max position size
        order = create_test_order(
            symbol="SPY",
            quantity=10000,  # Exceeds limit
            price=450.0
        )

        # Execute
        result = await submit_order(order)

        # Verify
        assert result.validated is False
        assert "exceeds limit" in result.rejection_reason
        assert audit_logger.last_entry.action == AuditAction.ORDER_SUBMIT
        assert "rejected" in audit_logger.last_entry.description
```

---

## 10. Rollback Procedures

### 10.1 Pre-Rollback Checklist

- [ ] Confirm all positions are closed or manually managed
- [ ] Save current state to backup location
- [ ] Document what triggered the rollback
- [ ] Notify team of rollback initiation

### 10.2 Rollback Commands

```bash
# Tag current state before rollback
git tag rollback-from-$(date +%Y%m%d-%H%M%S)

# Rollback to pre-refactor state
git checkout pre-refactor-v1.x

# Or rollback to specific commit
git checkout <commit-hash>

# Restore dependencies
pip install -r requirements.txt

# Verify tests pass
pytest tests/ -v

# Restart services
./scripts/restart_services.sh
```

### 10.3 Rollback Verification

```bash
# Verify critical functionality
pytest tests/test_critical_fixes.py -v

# Check that old behavior is restored
python -c "from execution.bot_managed_positions import *; print('Import OK')"

# Verify API is accessible
curl -s http://localhost:8000/health | jq .
```

### 10.4 Post-Rollback Documentation

Create an incident report:

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

## 11. Post-Refactoring Validation

### 11.1 Validation Checklist

- [ ] All 541+ tests passing
- [ ] No new linting errors: `ruff check .`
- [ ] Type checking passes: `mypy .`
- [ ] Critical fixes have test coverage: `pytest tests/test_critical_fixes.py -v`
- [ ] Audit logging functional: `tail -f logs/audit/audit_$(date +%Y-%m-%d).jsonl`
- [ ] API endpoints responsive: `curl http://localhost:8000/health`
- [ ] Paper trading successful for 1 trading session

### 11.2 Performance Benchmarks

| Metric | Before | After | Acceptable |
|--------|--------|-------|------------|
| Order submission latency | <100ms | <100ms | <200ms |
| Position update latency | <50ms | <50ms | <100ms |
| Memory usage (idle) | <500MB | <500MB | <1GB |
| API response time (p95) | <200ms | <200ms | <500ms |

### 11.3 Sign-Off Requirements

- [ ] Developer sign-off: All changes reviewed
- [ ] Code review: PR approved by second reviewer
- [ ] QA sign-off: All test scenarios passed
- [ ] Risk sign-off: Risk management verified
- [ ] Deployment sign-off: Paper trading successful

---

## 12. Appendix

### A. File Change Summary

| File | Action | Priority |
|------|--------|----------|
| `execution/bot_managed_positions.py` | Fix position sizing, add Greeks | P0 |
| `execution/smart_execution.py` | Fix fill calculation, add slippage | P0 |
| `api/rest_server.py` | Add thread safety, CORS, rate limiting | P0 |
| `api/order_queue_api.py` | Add authentication | P0 |
| `api/websocket_handler.py` | Fix race condition | P0 |
| `models/risk_limits.py` | NEW - Single source of truth | P1 |
| `utils/error_handling.py` | Consolidate retry logic | P1 |
| `utils/logging_config.py` | NEW - Centralized logging | P1 |
| `utils/audit_logger.py` | NEW - Audit trail | P1 |
| `models/audit.py` | NEW - Audit data structures | P1 |

### B. Deprecated Files

| File | Replacement | Action |
|------|-------------|--------|
| `models/retry_handler.py` | `utils/error_handling.py` | Add deprecation warning |
| `docs/templates/.../risk_validator.py` | `models/risk_limits.py` | Delete after migration |

### C. New Dependencies

```
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

---

*Document Version: 1.0.0*
*Last Updated: 2025-12-06*
*Author: Claude Code Analysis*
