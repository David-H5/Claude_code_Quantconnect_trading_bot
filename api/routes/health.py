"""
Health Check and System Status Endpoints

Provides REST endpoints for:
- Server health checks
- System status
- Configuration info
- Circuit breaker status

UPGRADE-008: REST API Server (December 2025)
"""

from __future__ import annotations

import logging
import platform
import sys
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Response Models
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status")
    timestamp: datetime
    version: str
    uptime_seconds: float
    checks: dict[str, bool]


class SystemInfo(BaseModel):
    """System information response."""

    python_version: str
    platform: str
    api_version: str
    algorithm_connected: bool
    order_queue_active: bool
    websocket_connections: int
    timestamp: datetime


class CircuitBreakerStatus(BaseModel):
    """Circuit breaker status response."""

    is_halted: bool
    state: str
    trip_reason: str | None
    last_trip_time: datetime | None
    consecutive_losses: int
    daily_loss_pct: float
    max_daily_loss_pct: float
    can_trade: bool


class ConfigurationInfo(BaseModel):
    """Configuration information response."""

    symbols: list
    autonomous_enabled: bool
    manual_enabled: bool
    bot_manager_enabled: bool
    recurring_enabled: bool
    max_position_size_pct: float
    max_daily_loss_pct: float
    max_drawdown_pct: float


# ============================================================================
# Module-level tracking
# ============================================================================

_start_time = datetime.now(timezone.utc)


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health.

    Returns:
        HealthResponse with component status
    """
    from ..rest_server import get_algorithm, get_order_queue, get_ws_manager

    order_queue = get_order_queue()
    ws_manager = get_ws_manager()
    algo = get_algorithm()

    checks = {
        "order_queue": order_queue is not None,
        "websocket_manager": ws_manager is not None,
        "algorithm_connected": algo is not None,
    }

    # Check if algorithm has required components
    if algo:
        checks["circuit_breaker"] = hasattr(algo, "circuit_breaker")
        checks["risk_manager"] = hasattr(algo, "risk_manager")
        checks["options_executor"] = hasattr(algo, "options_executor")

    # Determine overall status
    critical_checks = ["order_queue", "websocket_manager"]
    all_critical_ok = all(checks.get(c, False) for c in critical_checks)

    status = "healthy" if all_critical_ok else "degraded"
    if not any(checks.values()):
        status = "unhealthy"

    uptime = (datetime.now(timezone.utc) - _start_time).total_seconds()

    return HealthResponse(
        status=status,
        timestamp=datetime.now(timezone.utc),
        version="1.0.0",
        uptime_seconds=uptime,
        checks=checks,
    )


@router.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe.

    Returns 200 if server is running.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_probe():
    """Kubernetes readiness probe.

    Returns 200 if server is ready to accept requests.
    """
    from ..rest_server import get_order_queue

    order_queue = get_order_queue()
    if order_queue is None:
        raise HTTPException(status_code=503, detail="Order queue not ready")

    return {"status": "ready"}


@router.get("/system", response_model=SystemInfo)
async def get_system_info():
    """Get system information.

    Returns:
        SystemInfo with environment details
    """
    from ..rest_server import get_algorithm, get_order_queue, get_ws_manager

    order_queue = get_order_queue()
    ws_manager = get_ws_manager()
    algo = get_algorithm()

    return SystemInfo(
        python_version=sys.version,
        platform=platform.platform(),
        api_version="1.0.0",
        algorithm_connected=algo is not None,
        order_queue_active=order_queue is not None,
        websocket_connections=ws_manager.connection_count if ws_manager else 0,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/circuit-breaker", response_model=CircuitBreakerStatus)
async def get_circuit_breaker_status():
    """Get circuit breaker status.

    Returns:
        CircuitBreakerStatus with trading safety status
    """
    from ..rest_server import get_algorithm

    algo = get_algorithm()

    if algo is None or not hasattr(algo, "circuit_breaker"):
        return CircuitBreakerStatus(
            is_halted=False,
            state="unknown",
            trip_reason=None,
            last_trip_time=None,
            consecutive_losses=0,
            daily_loss_pct=0.0,
            max_daily_loss_pct=0.03,
            can_trade=False,
        )

    breaker = algo.circuit_breaker

    return CircuitBreakerStatus(
        is_halted=breaker.is_halted,
        state=breaker.state.value if hasattr(breaker, "state") else "unknown",
        trip_reason=breaker.trip_reason.value if breaker.trip_reason else None,
        last_trip_time=breaker.last_trip_time,
        consecutive_losses=breaker.consecutive_losses,
        daily_loss_pct=getattr(breaker, "_daily_loss_pct", 0.0),
        max_daily_loss_pct=breaker.max_daily_loss,
        can_trade=breaker.can_trade(),
    )


@router.post("/circuit-breaker/reset")
async def reset_circuit_breaker(authorized_by: str):
    """Reset circuit breaker (requires authorization).

    Args:
        authorized_by: Email/ID of person authorizing reset

    Returns:
        Confirmation message
    """
    from ..rest_server import get_algorithm, get_ws_manager

    algo = get_algorithm()

    if algo is None or not hasattr(algo, "circuit_breaker"):
        raise HTTPException(status_code=503, detail="Circuit breaker not available")

    try:
        algo.circuit_breaker.reset(authorized_by=authorized_by)

        # Broadcast status change
        ws_manager = get_ws_manager()
        if ws_manager:
            await ws_manager.broadcast_circuit_breaker(
                is_halted=False,
                reason=f"Reset by {authorized_by}",
            )

        return {
            "message": "Circuit breaker reset",
            "authorized_by": authorized_by,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error resetting circuit breaker: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/config", response_model=ConfigurationInfo)
async def get_configuration():
    """Get current configuration.

    Returns:
        ConfigurationInfo with active settings
    """
    from ..rest_server import get_algorithm

    algo = get_algorithm()

    if algo is None:
        return ConfigurationInfo(
            symbols=[],
            autonomous_enabled=False,
            manual_enabled=False,
            bot_manager_enabled=False,
            recurring_enabled=False,
            max_position_size_pct=0.25,
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.10,
        )

    # Get symbols from algorithm
    symbols = []
    if hasattr(algo, "option_symbols"):
        symbols = list(algo.option_symbols.keys())

    # Get risk limits
    risk_limits = getattr(algo, "risk_limits", None)
    max_position = risk_limits.max_position_size if risk_limits else 0.25
    max_daily_loss = risk_limits.max_daily_loss if risk_limits else 0.03
    max_drawdown = risk_limits.max_drawdown if risk_limits else 0.10

    return ConfigurationInfo(
        symbols=symbols,
        autonomous_enabled=hasattr(algo, "options_executor") and algo.options_executor is not None,
        manual_enabled=hasattr(algo, "manual_executor") and algo.manual_executor is not None,
        bot_manager_enabled=hasattr(algo, "bot_manager") and algo.bot_manager is not None,
        recurring_enabled=hasattr(algo, "recurring_manager") and algo.recurring_manager is not None,
        max_position_size_pct=max_position,
        max_daily_loss_pct=max_daily_loss,
        max_drawdown_pct=max_drawdown,
    )


@router.get("/websocket/clients")
async def get_websocket_clients():
    """Get information about connected WebSocket clients.

    Returns:
        List of connected clients
    """
    from ..rest_server import get_ws_manager

    ws_manager = get_ws_manager()
    if ws_manager is None:
        return {"clients": [], "count": 0}

    clients = ws_manager.get_client_info()
    return {"clients": clients, "count": len(clients)}
