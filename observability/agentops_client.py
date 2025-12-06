"""
AgentOps Client for AI Agent Monitoring

UPGRADE-015 Phase 6: Observability Setup

Provides integration with AgentOps for monitoring AI agent behavior,
tracking sessions, recording events, and analyzing agent performance.

Features:
- Session management
- Event recording
- Cost tracking
- Performance analytics
- Error monitoring
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# Optional import - gracefully degrade if not installed
try:
    import agentops

    AGENTOPS_AVAILABLE = True
except ImportError:
    agentops = None  # type: ignore
    AGENTOPS_AVAILABLE = False


class EventType(Enum):
    """Types of events to track."""

    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    DECISION = "decision"
    TRADE = "trade"
    ERROR = "error"
    SESSION = "session"


@dataclass
class AgentEvent:
    """Represents an agent event for tracking."""

    event_type: EventType
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float | None = None
    success: bool = True
    error: str | None = None


@dataclass
class SessionStats:
    """Statistics for an agent session."""

    session_id: str
    start_time: datetime
    end_time: datetime | None = None
    total_events: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    decisions: int = 0
    trades: int = 0
    errors: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0


class AgentOpsClient:
    """Client for AgentOps integration."""

    def __init__(
        self,
        api_key: str | None = None,
        enabled: bool = True,
        project_name: str = "quantconnect-trading-bot",
        environment: str = "development",
    ):
        """
        Initialize AgentOps client.

        Args:
            api_key: AgentOps API key (or use AGENTOPS_API_KEY env var)
            enabled: Whether tracking is enabled
            project_name: Name of the project for organization
            environment: Environment name (development, staging, production)
        """
        self.api_key = api_key or os.environ.get("AGENTOPS_API_KEY")
        self.enabled = enabled and AGENTOPS_AVAILABLE and self.api_key is not None
        self.project_name = project_name
        self.environment = environment

        self._session = None
        self._session_stats: SessionStats | None = None
        self._events: list[AgentEvent] = []

        if self.enabled:
            self._init_agentops()

    def _init_agentops(self) -> None:
        """Initialize AgentOps library."""
        if not AGENTOPS_AVAILABLE:
            return

        try:
            agentops.init(
                api_key=self.api_key,
                default_tags=[self.project_name, self.environment],
            )
        except Exception as e:
            print(f"Failed to initialize AgentOps: {e}")
            self.enabled = False

    def start_session(
        self,
        session_name: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """
        Start a new agent session.

        Args:
            session_name: Optional name for the session
            tags: Optional tags for the session

        Returns:
            Session ID
        """
        session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        self._session_stats = SessionStats(
            session_id=session_id,
            start_time=datetime.utcnow(),
        )

        if self.enabled and agentops:
            try:
                self._session = agentops.start_session(
                    tags=tags or [self.project_name],
                )
                if self._session:
                    session_id = str(self._session.session_id)
                    self._session_stats.session_id = session_id
            except Exception as e:
                print(f"Failed to start AgentOps session: {e}")

        return session_id

    def end_session(
        self,
        status: str = "success",
        reason: str | None = None,
    ) -> SessionStats | None:
        """
        End the current session.

        Args:
            status: Session end status
            reason: Optional reason for ending

        Returns:
            Session statistics
        """
        if self._session_stats:
            self._session_stats.end_time = datetime.utcnow()

        if self.enabled and agentops and self._session:
            try:
                agentops.end_session(end_state=status, end_state_reason=reason)
            except Exception as e:
                print(f"Failed to end AgentOps session: {e}")

        stats = self._session_stats
        self._session = None
        self._session_stats = None

        return stats

    def record_event(self, event: AgentEvent) -> None:
        """
        Record an agent event.

        Args:
            event: Event to record
        """
        self._events.append(event)

        if self._session_stats:
            self._session_stats.total_events += 1
            if event.event_type == EventType.TOOL_CALL:
                self._session_stats.tool_calls += 1
            elif event.event_type == EventType.LLM_CALL:
                self._session_stats.llm_calls += 1
            elif event.event_type == EventType.DECISION:
                self._session_stats.decisions += 1
            elif event.event_type == EventType.TRADE:
                self._session_stats.trades += 1
            elif event.event_type == EventType.ERROR:
                self._session_stats.errors += 1

        if self.enabled and agentops:
            try:
                agentops.record(
                    agentops.Event(
                        event_type=event.event_type.value,
                        name=event.name,
                        params=event.data,
                    )
                )
            except Exception as e:
                print(f"Failed to record AgentOps event: {e}")

    def record_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Record a tool call event."""
        event = AgentEvent(
            event_type=EventType.TOOL_CALL,
            name=tool_name,
            data={
                "input": tool_input,
                "output": tool_output,
            },
            duration_ms=duration_ms,
            success=success,
            error=error,
        )
        self.record_event(event)

    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Record an LLM API call event."""
        event = AgentEvent(
            event_type=EventType.LLM_CALL,
            name=f"llm_call_{model}",
            data={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            duration_ms=duration_ms,
            success=success,
            error=error,
        )

        if self._session_stats:
            self._session_stats.total_tokens += prompt_tokens + completion_tokens

        self.record_event(event)

    def record_decision(
        self,
        decision_type: str,
        decision: str,
        confidence: float,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record an agent decision event."""
        event = AgentEvent(
            event_type=EventType.DECISION,
            name=f"decision_{decision_type}",
            data={
                "decision_type": decision_type,
                "decision": decision,
                "confidence": confidence,
                "context": context or {},
            },
        )
        self.record_event(event)

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float | None = None,
        order_type: str = "market",
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Record a trading event."""
        event = AgentEvent(
            event_type=EventType.TRADE,
            name=f"trade_{symbol}",
            data={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "order_type": order_type,
            },
            success=success,
            error=error,
        )
        self.record_event(event)

    def record_error(
        self,
        error_type: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record an error event."""
        event = AgentEvent(
            event_type=EventType.ERROR,
            name=f"error_{error_type}",
            data={
                "error_type": error_type,
                "message": message,
                "context": context or {},
            },
            success=False,
            error=message,
        )
        self.record_event(event)

    def get_session_stats(self) -> SessionStats | None:
        """Get current session statistics."""
        return self._session_stats

    def get_events(
        self,
        event_type: EventType | None = None,
        limit: int = 100,
    ) -> list[AgentEvent]:
        """
        Get recorded events.

        Args:
            event_type: Filter by event type
            limit: Maximum events to return

        Returns:
            List of events
        """
        events = self._events
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]


@contextmanager
def agent_session(
    client: AgentOpsClient,
    session_name: str | None = None,
    tags: list[str] | None = None,
):
    """
    Context manager for agent sessions.

    Usage:
        client = AgentOpsClient()
        with agent_session(client, "trading_session") as session_id:
            # Do agent work
            client.record_tool_call("get_quote", {"symbol": "SPY"})
    """
    session_id = client.start_session(session_name, tags)
    try:
        yield session_id
    except Exception as e:
        client.record_error("session_error", str(e))
        client.end_session("failed", str(e))
        raise
    else:
        client.end_session("success")


def create_agentops_client(
    api_key: str | None = None,
    enabled: bool = True,
) -> AgentOpsClient:
    """
    Factory function to create an AgentOps client.

    Args:
        api_key: Optional API key (uses env var if not provided)
        enabled: Whether tracking is enabled

    Returns:
        Configured AgentOpsClient
    """
    return AgentOpsClient(
        api_key=api_key,
        enabled=enabled,
    )
