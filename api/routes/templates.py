"""
Recurring Order Template Endpoints

Provides REST endpoints for:
- Creating recurring order templates
- Managing template schedules
- Template execution history

UPGRADE-008: REST API Server (December 2025)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class ScheduleType(str, Enum):
    """Template schedule types."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"


class TemplateStatus(str, Enum):
    """Template status."""

    ACTIVE = "active"
    PAUSED = "paused"
    EXPIRED = "expired"


class TemplateCreate(BaseModel):
    """Template creation request."""

    name: str = Field(..., min_length=1, max_length=100, description="Template name")
    symbol: str = Field(..., description="Underlying symbol")
    strategy_name: str = Field(..., description="Strategy to execute")
    quantity: int = Field(1, ge=1, description="Order quantity")
    limit_price: float | None = Field(None, description="Limit price")
    schedule_type: ScheduleType = Field(..., description="Schedule type")
    schedule_time: str | None = Field(None, description="Time to execute (HH:MM)")
    days_of_week: list[int] | None = Field(None, description="Days for weekly (0=Mon)")
    day_of_month: int | None = Field(None, ge=1, le=28, description="Day for monthly")
    expires_at: datetime | None = Field(None, description="Expiration date")
    max_executions: int | None = Field(None, ge=1, description="Max executions")
    notes: str | None = Field(None, max_length=500)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Weekly SPY Iron Condor",
                "symbol": "SPY",
                "strategy_name": "iron_condor",
                "quantity": 1,
                "schedule_type": "weekly",
                "schedule_time": "09:35",
                "days_of_week": [0, 2, 4],
                "notes": "M/W/F weekly iron condors",
            }
        }
    )


class TemplateUpdate(BaseModel):
    """Template update request."""

    name: str | None = Field(None, min_length=1, max_length=100)
    quantity: int | None = Field(None, ge=1)
    limit_price: float | None = None
    schedule_time: str | None = None
    status: TemplateStatus | None = None
    notes: str | None = Field(None, max_length=500)


class TemplateData(BaseModel):
    """Template details."""

    template_id: str
    name: str
    symbol: str
    strategy_name: str
    quantity: int
    limit_price: float | None
    schedule_type: str
    schedule_time: str | None
    status: str
    created_at: datetime
    last_executed: datetime | None
    execution_count: int
    max_executions: int | None
    expires_at: datetime | None
    notes: str | None


class TemplateListResponse(BaseModel):
    """Template list response."""

    templates: list[TemplateData]
    count: int


class ExecutionRecord(BaseModel):
    """Template execution record."""

    execution_id: str
    template_id: str
    order_id: str
    executed_at: datetime
    status: str
    fill_price: float | None
    error_message: str | None


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/templates", response_model=TemplateData, status_code=201)
async def create_template(template: TemplateCreate):
    """Create a new recurring order template.

    Returns:
        Created template details
    """
    from ..rest_server import get_algorithm

    algo = get_algorithm()

    # Check if recurring order manager is available
    if algo and hasattr(algo, "recurring_manager"):
        manager = algo.recurring_manager
    else:
        # Return mock response if manager not available
        logger.warning("RecurringOrderManager not available, returning mock response")
        return TemplateData(
            template_id="template_mock",
            name=template.name,
            symbol=template.symbol,
            strategy_name=template.strategy_name,
            quantity=template.quantity,
            limit_price=template.limit_price,
            schedule_type=template.schedule_type.value,
            schedule_time=template.schedule_time,
            status="active",
            created_at=datetime.now(timezone.utc),
            last_executed=None,
            execution_count=0,
            max_executions=template.max_executions,
            expires_at=template.expires_at,
            notes=template.notes,
        )

    try:
        # Create template via manager
        result = manager.create_template(
            name=template.name,
            symbol=template.symbol,
            strategy_name=template.strategy_name,
            quantity=template.quantity,
            limit_price=template.limit_price,
            schedule_type=template.schedule_type.value,
            schedule_time=template.schedule_time,
            days_of_week=template.days_of_week,
            day_of_month=template.day_of_month,
            expires_at=template.expires_at,
            max_executions=template.max_executions,
            notes=template.notes,
        )

        return TemplateData(
            template_id=result.template_id,
            name=result.name,
            symbol=result.symbol,
            strategy_name=result.strategy_name,
            quantity=result.quantity,
            limit_price=result.limit_price,
            schedule_type=result.schedule_type,
            schedule_time=result.schedule_time,
            status=result.status,
            created_at=result.created_at,
            last_executed=result.last_executed,
            execution_count=result.execution_count,
            max_executions=result.max_executions,
            expires_at=result.expires_at,
            notes=result.notes,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating template: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/templates", response_model=TemplateListResponse)
async def list_templates(
    status: TemplateStatus | None = Query(None, description="Filter by status"),
    symbol: str | None = Query(None, description="Filter by symbol"),
):
    """List all recurring order templates.

    Returns:
        TemplateListResponse with filtered templates
    """
    from ..rest_server import get_algorithm

    algo = get_algorithm()

    if algo and hasattr(algo, "recurring_manager"):
        manager = algo.recurring_manager
        templates = manager.get_all_templates()

        # Apply filters
        if status:
            templates = [t for t in templates if t.status == status.value]
        if symbol:
            templates = [t for t in templates if t.symbol.upper() == symbol.upper()]

        template_data = [
            TemplateData(
                template_id=t.template_id,
                name=t.name,
                symbol=t.symbol,
                strategy_name=t.strategy_name,
                quantity=t.quantity,
                limit_price=t.limit_price,
                schedule_type=t.schedule_type,
                schedule_time=t.schedule_time,
                status=t.status,
                created_at=t.created_at,
                last_executed=t.last_executed,
                execution_count=t.execution_count,
                max_executions=t.max_executions,
                expires_at=t.expires_at,
                notes=t.notes,
            )
            for t in templates
        ]

        return TemplateListResponse(
            templates=template_data,
            count=len(template_data),
        )

    # Return empty list if manager not available
    return TemplateListResponse(templates=[], count=0)


@router.get("/templates/{template_id}", response_model=TemplateData)
async def get_template(template_id: str):
    """Get template details by ID.

    Args:
        template_id: Template identifier

    Returns:
        Template details
    """
    from ..rest_server import get_algorithm

    algo = get_algorithm()

    if algo and hasattr(algo, "recurring_manager"):
        manager = algo.recurring_manager
        template = manager.get_template(template_id)

        if template:
            return TemplateData(
                template_id=template.template_id,
                name=template.name,
                symbol=template.symbol,
                strategy_name=template.strategy_name,
                quantity=template.quantity,
                limit_price=template.limit_price,
                schedule_type=template.schedule_type,
                schedule_time=template.schedule_time,
                status=template.status,
                created_at=template.created_at,
                last_executed=template.last_executed,
                execution_count=template.execution_count,
                max_executions=template.max_executions,
                expires_at=template.expires_at,
                notes=template.notes,
            )

    raise HTTPException(status_code=404, detail=f"Template not found: {template_id}")


@router.patch("/templates/{template_id}", response_model=TemplateData)
async def update_template(template_id: str, update: TemplateUpdate):
    """Update a template.

    Args:
        template_id: Template identifier
        update: Fields to update

    Returns:
        Updated template details
    """
    from ..rest_server import get_algorithm

    algo = get_algorithm()

    if algo and hasattr(algo, "recurring_manager"):
        manager = algo.recurring_manager

        try:
            template = manager.update_template(
                template_id=template_id,
                **update.model_dump(exclude_unset=True),
            )

            if template:
                return TemplateData(
                    template_id=template.template_id,
                    name=template.name,
                    symbol=template.symbol,
                    strategy_name=template.strategy_name,
                    quantity=template.quantity,
                    limit_price=template.limit_price,
                    schedule_type=template.schedule_type,
                    schedule_time=template.schedule_time,
                    status=template.status,
                    created_at=template.created_at,
                    last_executed=template.last_executed,
                    execution_count=template.execution_count,
                    max_executions=template.max_executions,
                    expires_at=template.expires_at,
                    notes=template.notes,
                )

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    raise HTTPException(status_code=404, detail=f"Template not found: {template_id}")


@router.delete("/templates/{template_id}")
async def delete_template(template_id: str):
    """Delete a template.

    Args:
        template_id: Template identifier

    Returns:
        Confirmation message
    """
    from ..rest_server import get_algorithm

    algo = get_algorithm()

    if algo and hasattr(algo, "recurring_manager"):
        manager = algo.recurring_manager
        success = manager.delete_template(template_id)

        if success:
            return {"message": "Template deleted", "template_id": template_id}

    raise HTTPException(status_code=404, detail=f"Template not found: {template_id}")


@router.post("/templates/{template_id}/pause")
async def pause_template(template_id: str):
    """Pause a template.

    Args:
        template_id: Template identifier

    Returns:
        Updated template status
    """
    from ..rest_server import get_algorithm

    algo = get_algorithm()

    if algo and hasattr(algo, "recurring_manager"):
        manager = algo.recurring_manager
        success = manager.pause_template(template_id)

        if success:
            return {"message": "Template paused", "template_id": template_id, "status": "paused"}

    raise HTTPException(status_code=404, detail=f"Template not found: {template_id}")


@router.post("/templates/{template_id}/resume")
async def resume_template(template_id: str):
    """Resume a paused template.

    Args:
        template_id: Template identifier

    Returns:
        Updated template status
    """
    from ..rest_server import get_algorithm

    algo = get_algorithm()

    if algo and hasattr(algo, "recurring_manager"):
        manager = algo.recurring_manager
        success = manager.resume_template(template_id)

        if success:
            return {"message": "Template resumed", "template_id": template_id, "status": "active"}

    raise HTTPException(status_code=404, detail=f"Template not found: {template_id}")


@router.get("/templates/{template_id}/executions", response_model=list[ExecutionRecord])
async def get_template_executions(
    template_id: str,
    limit: int = Query(50, ge=1, le=500),
):
    """Get execution history for a template.

    Args:
        template_id: Template identifier
        limit: Maximum records to return

    Returns:
        List of execution records
    """
    from ..rest_server import get_algorithm

    algo = get_algorithm()

    if algo and hasattr(algo, "recurring_manager"):
        manager = algo.recurring_manager
        executions = manager.get_execution_history(template_id, limit=limit)

        return [
            ExecutionRecord(
                execution_id=e.execution_id,
                template_id=e.template_id,
                order_id=e.order_id,
                executed_at=e.executed_at,
                status=e.status,
                fill_price=e.fill_price,
                error_message=e.error_message,
            )
            for e in executions
        ]

    return []
