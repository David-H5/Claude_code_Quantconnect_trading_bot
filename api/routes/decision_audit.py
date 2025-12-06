"""
Decision Audit REST API

Provides endpoints for querying and analyzing agent decision audit trails.
Supports filtering, pagination, and performance analytics.

Endpoints:
    GET /audit/decisions - Query decision logs
    GET /audit/decisions/{id} - Get specific decision
    GET /audit/agents/performance - Agent performance metrics
    GET /audit/explanations/{id} - Get human-readable explanation

Usage:
    from api.routes.decision_audit import router
    app.include_router(router)
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from llm.agents.explanation import create_explanation
from llm.decision_logger import AgentDecisionLog


router = APIRouter(prefix="/audit", tags=["Decision Audit"])

DECISION_LOG_DIR = Path(".claude/state/decisions")


class DecisionQuery(BaseModel):
    """Query parameters for decision search."""

    agent: str | None = None
    symbol: str | None = None
    action: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    outcome: str | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None


class AgentPerformance(BaseModel):
    """Agent performance summary."""

    agent_name: str
    decision_count: int
    actions: dict[str, int]
    avg_confidence: float
    outcomes: dict[str, int]
    accuracy: float | None = None


class PerformanceComparison(BaseModel):
    """Performance comparison across agents."""

    period_days: int
    agents: list[AgentPerformance]
    total_decisions: int
    most_active: str | None = None
    highest_confidence: str | None = None


def _load_decision(log_file: Path) -> dict[str, Any] | None:
    """Load a decision from file."""
    try:
        return json.loads(log_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _matches_filters(
    data: dict[str, Any],
    agent: str | None = None,
    symbol: str | None = None,
    action: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    outcome: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
) -> bool:
    """Check if decision matches filter criteria."""
    if agent and data.get("agent_name") != agent:
        return False

    if symbol:
        ctx_symbol = data.get("context", {}).get("symbol")
        if ctx_symbol != symbol:
            return False

    if action:
        decision = data.get("decision", "").lower()
        if action.lower() not in decision:
            return False

    if since:
        try:
            decision_time = datetime.fromisoformat(data.get("timestamp", ""))
            if decision_time < since:
                return False
        except ValueError:
            return False

    if until:
        try:
            decision_time = datetime.fromisoformat(data.get("timestamp", ""))
            if decision_time > until:
                return False
        except ValueError:
            return False

    if outcome and data.get("outcome") != outcome:
        return False

    confidence = data.get("confidence", 0.5)
    if min_confidence is not None and confidence < min_confidence:
        return False

    return not (max_confidence is not None and confidence > max_confidence)


@router.get("/decisions")
async def get_decisions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=1000),
    agent: str | None = None,
    symbol: str | None = None,
    action: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    outcome: str | None = None,
    min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1),
) -> dict[str, Any]:
    """
    Query decision audit trail.

    Args:
        skip: Number of records to skip (pagination)
        limit: Maximum records to return
        agent: Filter by agent name
        symbol: Filter by trading symbol
        action: Filter by action type (buy, sell, hold)
        since: Only return decisions after this datetime
        until: Only return decisions before this datetime
        outcome: Filter by outcome (pending, executed, rejected)
        min_confidence: Minimum confidence threshold
        max_confidence: Maximum confidence threshold

    Returns:
        Paginated list of decision records
    """
    DECISION_LOG_DIR.mkdir(parents=True, exist_ok=True)

    decisions: list[dict[str, Any]] = []
    total_count = 0

    for log_file in sorted(DECISION_LOG_DIR.glob("*.json"), reverse=True):
        data = _load_decision(log_file)
        if not data:
            continue

        if not _matches_filters(data, agent, symbol, action, since, until, outcome, min_confidence, max_confidence):
            continue

        total_count += 1

        if total_count > skip and len(decisions) < limit:
            decisions.append(data)

    return {
        "total": total_count,
        "skip": skip,
        "limit": limit,
        "count": len(decisions),
        "decisions": decisions,
    }


@router.get("/decisions/{decision_id}")
async def get_decision(decision_id: str) -> dict[str, Any]:
    """
    Get a specific decision by ID.

    Args:
        decision_id: Unique decision identifier

    Returns:
        Decision record with full details
    """
    DECISION_LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = DECISION_LOG_DIR / f"{decision_id}.json"

    if not log_file.exists():
        raise HTTPException(status_code=404, detail="Decision not found")

    data = _load_decision(log_file)
    if not data:
        raise HTTPException(status_code=500, detail="Failed to load decision")

    return data


@router.get("/explanations/{decision_id}")
async def get_explanation(decision_id: str) -> dict[str, Any]:
    """
    Get human-readable explanation for a decision.

    Args:
        decision_id: Unique decision identifier

    Returns:
        Formatted explanation with reasoning chain
    """
    DECISION_LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = DECISION_LOG_DIR / f"{decision_id}.json"

    if not log_file.exists():
        raise HTTPException(status_code=404, detail="Decision not found")

    data = _load_decision(log_file)
    if not data:
        raise HTTPException(status_code=500, detail="Failed to load decision")

    try:
        log = AgentDecisionLog.from_dict(data)
        explanation = create_explanation(log)
        return {
            "explanation": explanation.to_dict(),
            "human_readable": explanation.to_human_readable(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create explanation: {e}") from e


@router.get("/agents/performance")
async def get_agent_performance(days: int = Query(30, ge=1, le=365)) -> PerformanceComparison:
    """
    Get agent performance metrics over time.

    Args:
        days: Number of days to analyze (1-365)

    Returns:
        Performance comparison across all agents
    """
    DECISION_LOG_DIR.mkdir(parents=True, exist_ok=True)

    since = datetime.now() - timedelta(days=days)
    agent_stats: dict[str, dict[str, Any]] = {}

    for log_file in DECISION_LOG_DIR.glob("*.json"):
        data = _load_decision(log_file)
        if not data:
            continue

        try:
            decision_time = datetime.fromisoformat(data.get("timestamp", ""))
            if decision_time < since:
                continue
        except ValueError:
            continue

        agent = data.get("agent_name", "unknown")
        if agent not in agent_stats:
            agent_stats[agent] = {
                "decision_count": 0,
                "actions": {},
                "outcomes": {},
                "confidence_sum": 0.0,
                "correct": 0,
                "total_evaluated": 0,
            }

        stats = agent_stats[agent]
        stats["decision_count"] += 1

        # Track actions (extract from decision text)
        decision = data.get("decision", "").lower()
        for action in ["buy", "sell", "hold", "close"]:
            if action in decision:
                stats["actions"][action] = stats["actions"].get(action, 0) + 1
                break
        else:
            stats["actions"]["other"] = stats["actions"].get("other", 0) + 1

        # Track outcomes
        outcome = data.get("outcome", "pending")
        stats["outcomes"][outcome] = stats["outcomes"].get(outcome, 0) + 1

        # Track confidence
        confidence = data.get("confidence", 0.5)
        stats["confidence_sum"] += confidence

        # Track accuracy (if outcome is known)
        if outcome in ["executed", "rejected"]:
            stats["total_evaluated"] += 1
            if outcome == "executed":
                stats["correct"] += 1

    # Build performance summaries
    agents: list[AgentPerformance] = []
    for agent_name, stats in agent_stats.items():
        avg_conf = stats["confidence_sum"] / max(stats["decision_count"], 1)
        accuracy = None
        if stats["total_evaluated"] > 0:
            accuracy = stats["correct"] / stats["total_evaluated"]

        agents.append(
            AgentPerformance(
                agent_name=agent_name,
                decision_count=stats["decision_count"],
                actions=stats["actions"],
                avg_confidence=round(avg_conf, 3),
                outcomes=stats["outcomes"],
                accuracy=accuracy,
            )
        )

    # Sort by decision count
    agents.sort(key=lambda a: a.decision_count, reverse=True)

    # Find top performers
    most_active = agents[0].agent_name if agents else None
    highest_confidence = None
    if agents:
        highest_confidence = max(agents, key=lambda a: a.avg_confidence).agent_name

    total_decisions = sum(a.decision_count for a in agents)

    return PerformanceComparison(
        period_days=days,
        agents=agents,
        total_decisions=total_decisions,
        most_active=most_active,
        highest_confidence=highest_confidence,
    )


@router.get("/agents/{agent_name}/history")
async def get_agent_history(
    agent_name: str,
    days: int = Query(7, ge=1, le=90),
    limit: int = Query(50, le=500),
) -> dict[str, Any]:
    """
    Get decision history for a specific agent.

    Args:
        agent_name: Name of the agent
        days: Number of days to look back
        limit: Maximum decisions to return

    Returns:
        Agent's decision history with trend analysis
    """
    DECISION_LOG_DIR.mkdir(parents=True, exist_ok=True)

    since = datetime.now() - timedelta(days=days)
    decisions: list[dict[str, Any]] = []
    daily_counts: dict[str, int] = {}
    daily_confidence: dict[str, list[float]] = {}

    for log_file in sorted(DECISION_LOG_DIR.glob("*.json"), reverse=True):
        data = _load_decision(log_file)
        if not data:
            continue

        if data.get("agent_name") != agent_name:
            continue

        try:
            decision_time = datetime.fromisoformat(data.get("timestamp", ""))
            if decision_time < since:
                continue
            day_key = decision_time.strftime("%Y-%m-%d")
        except ValueError:
            continue

        daily_counts[day_key] = daily_counts.get(day_key, 0) + 1

        confidence = data.get("confidence", 0.5)
        if day_key not in daily_confidence:
            daily_confidence[day_key] = []
        daily_confidence[day_key].append(confidence)

        if len(decisions) < limit:
            decisions.append(data)

    # Calculate daily averages
    daily_avg_confidence = {day: sum(confs) / len(confs) for day, confs in daily_confidence.items()}

    return {
        "agent_name": agent_name,
        "period_days": days,
        "total_decisions": len(decisions),
        "daily_counts": daily_counts,
        "daily_avg_confidence": daily_avg_confidence,
        "decisions": decisions,
    }


@router.post("/decisions/{decision_id}/outcome")
async def update_decision_outcome(
    decision_id: str,
    outcome: str = Query(..., regex="^(executed|rejected|cancelled|timed_out)$"),
    notes: str | None = None,
) -> dict[str, Any]:
    """
    Update the outcome of a decision.

    Args:
        decision_id: Unique decision identifier
        outcome: New outcome status
        notes: Optional notes about the outcome

    Returns:
        Updated decision record
    """
    DECISION_LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = DECISION_LOG_DIR / f"{decision_id}.json"

    if not log_file.exists():
        raise HTTPException(status_code=404, detail="Decision not found")

    data = _load_decision(log_file)
    if not data:
        raise HTTPException(status_code=500, detail="Failed to load decision")

    # Update outcome
    data["outcome"] = outcome
    if notes:
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["outcome_notes"] = notes
        data["metadata"]["outcome_updated_at"] = datetime.now().isoformat()

    # Save updated decision
    log_file.write_text(json.dumps(data, indent=2))

    return data


@router.get("/summary")
async def get_audit_summary(days: int = Query(7, ge=1, le=90)) -> dict[str, Any]:
    """
    Get overall audit summary.

    Args:
        days: Number of days to analyze

    Returns:
        Summary statistics for the audit trail
    """
    DECISION_LOG_DIR.mkdir(parents=True, exist_ok=True)

    since = datetime.now() - timedelta(days=days)
    total_decisions = 0
    outcome_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    agent_counts: dict[str, int] = {}
    confidence_values: list[float] = []

    for log_file in DECISION_LOG_DIR.glob("*.json"):
        data = _load_decision(log_file)
        if not data:
            continue

        try:
            decision_time = datetime.fromisoformat(data.get("timestamp", ""))
            if decision_time < since:
                continue
        except ValueError:
            continue

        total_decisions += 1

        # Count outcomes
        outcome = data.get("outcome", "pending")
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        # Count agents
        agent = data.get("agent_name", "unknown")
        agent_counts[agent] = agent_counts.get(agent, 0) + 1

        # Count actions
        decision = data.get("decision", "").lower()
        for action in ["buy", "sell", "hold", "close"]:
            if action in decision:
                action_counts[action] = action_counts.get(action, 0) + 1
                break

        # Collect confidence
        confidence_values.append(data.get("confidence", 0.5))

    # Calculate averages
    avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0

    return {
        "period_days": days,
        "total_decisions": total_decisions,
        "agents_active": len(agent_counts),
        "avg_confidence": round(avg_confidence, 3),
        "outcomes": outcome_counts,
        "actions": action_counts,
        "agent_counts": agent_counts,
    }
