"""
OpenTelemetry GenAI Tracer

Implements OpenTelemetry semantic conventions for generative AI systems.
Provides tracing infrastructure for LLM calls with standardized attributes.

UPGRADE-014 Category 2: Observability & Debugging

References:
- https://opentelemetry.io/docs/specs/semconv/gen-ai/
- https://github.com/traceloop/openllmetry

QuantConnect Compatible: Yes
- No blocking operations
- Thread-safe implementation
- Defensive error handling
"""

import logging
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any


logger = logging.getLogger(__name__)


# ============================================================================
# OpenTelemetry GenAI Semantic Convention Constants
# ============================================================================


class GenAISystem(Enum):
    """GenAI system identifiers per OTel conventions."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    AWS_BEDROCK = "aws_bedrock"
    GOOGLE_VERTEX = "google_vertex"
    CUSTOM = "custom"


class SpanKind(Enum):
    """Span kinds for tracing."""

    CLIENT = "client"  # LLM API calls
    INTERNAL = "internal"  # Agent reasoning
    SERVER = "server"  # Handling requests


class SpanStatus(Enum):
    """Span status codes."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class TokenUsage:
    """Token usage for a GenAI call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class GenAISpan:
    """
    OpenTelemetry-compatible span for GenAI operations.

    Follows semantic conventions from:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/

    Attributes follow the gen_ai.* namespace.
    """

    # Span identification
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:32])
    parent_span_id: str | None = None

    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    duration_ms: float = 0.0

    # Span metadata
    name: str = ""
    kind: SpanKind = SpanKind.CLIENT
    status: SpanStatus = SpanStatus.UNSET
    status_message: str | None = None

    # GenAI semantic attributes (gen_ai.*)
    gen_ai_system: GenAISystem = GenAISystem.CUSTOM
    gen_ai_request_model: str = ""
    gen_ai_request_max_tokens: int | None = None
    gen_ai_request_temperature: float | None = None
    gen_ai_request_top_p: float | None = None
    gen_ai_request_stop_sequences: list[str] | None = None

    gen_ai_response_id: str | None = None
    gen_ai_response_model: str | None = None
    gen_ai_response_finish_reasons: list[str] | None = None

    gen_ai_usage: TokenUsage = field(default_factory=TokenUsage)

    # Agent-specific attributes
    agent_name: str | None = None
    agent_role: str | None = None
    operation_name: str | None = None

    # Error information
    error_type: str | None = None
    error_message: str | None = None

    # Additional attributes
    attributes: dict[str, Any] = field(default_factory=dict)

    # Events (logs within span)
    events: list[dict[str, Any]] = field(default_factory=list)

    def end(self, status: SpanStatus = SpanStatus.OK, message: str | None = None):
        """End the span and calculate duration."""
        self.end_time = datetime.now(timezone.utc)
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
        if message:
            self.status_message = message

    def add_event(self, name: str, attributes: dict[str, Any] | None = None):
        """Add an event (log) to the span."""
        self.events.append(
            {
                "name": name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attributes": attributes or {},
            }
        )

    def set_error(self, error_type: str, message: str):
        """Mark span as error with details."""
        self.status = SpanStatus.ERROR
        self.error_type = error_type
        self.error_message = message
        self.add_event(
            "exception",
            {
                "exception.type": error_type,
                "exception.message": message,
            },
        )

    def to_otel_attributes(self) -> dict[str, Any]:
        """
        Convert to OpenTelemetry attribute format.

        Returns dict with gen_ai.* namespace attributes.
        """
        attrs = {
            # GenAI request attributes
            "gen_ai.system": self.gen_ai_system.value,
            "gen_ai.request.model": self.gen_ai_request_model,
            # GenAI usage attributes
            "gen_ai.usage.input_tokens": self.gen_ai_usage.input_tokens,
            "gen_ai.usage.output_tokens": self.gen_ai_usage.output_tokens,
            "gen_ai.usage.total_tokens": self.gen_ai_usage.total_tokens,
            # Timing
            "duration_ms": self.duration_ms,
        }

        # Optional request attributes
        if self.gen_ai_request_max_tokens is not None:
            attrs["gen_ai.request.max_tokens"] = self.gen_ai_request_max_tokens
        if self.gen_ai_request_temperature is not None:
            attrs["gen_ai.request.temperature"] = self.gen_ai_request_temperature
        if self.gen_ai_request_top_p is not None:
            attrs["gen_ai.request.top_p"] = self.gen_ai_request_top_p

        # Optional response attributes
        if self.gen_ai_response_id:
            attrs["gen_ai.response.id"] = self.gen_ai_response_id
        if self.gen_ai_response_model:
            attrs["gen_ai.response.model"] = self.gen_ai_response_model
        if self.gen_ai_response_finish_reasons:
            attrs["gen_ai.response.finish_reasons"] = self.gen_ai_response_finish_reasons

        # Agent attributes
        if self.agent_name:
            attrs["agent.name"] = self.agent_name
        if self.agent_role:
            attrs["agent.role"] = self.agent_role
        if self.operation_name:
            attrs["operation.name"] = self.operation_name

        # Error attributes
        if self.error_type:
            attrs["error.type"] = self.error_type
        if self.error_message:
            attrs["error.message"] = self.error_message

        # Merge custom attributes
        attrs.update(self.attributes)

        return attrs

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "status": self.status.value,
            "status_message": self.status_message,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.to_otel_attributes(),
            "events": self.events,
        }


# ============================================================================
# LLM Tracer
# ============================================================================


class LLMTracer:
    """
    Tracer for LLM operations following OpenTelemetry conventions.

    Thread-safe implementation for concurrent agent operations.

    Usage:
        tracer = LLMTracer()

        with tracer.start_span("llm.chat", agent_name="analyst") as span:
            # Make LLM call
            response = llm_client.chat(...)

            # Record usage
            span.gen_ai_usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

        # Get all spans
        spans = tracer.get_spans()
    """

    def __init__(
        self,
        service_name: str = "trading-agents",
        max_spans: int = 10000,
        on_span_end: Callable[[GenAISpan], None] | None = None,
    ):
        """
        Initialize tracer.

        Args:
            service_name: Name of the service for traces
            max_spans: Maximum spans to retain in memory
            on_span_end: Callback when span ends (for exporters)
        """
        self.service_name = service_name
        self.max_spans = max_spans
        self.on_span_end = on_span_end

        self._spans: list[GenAISpan] = []
        self._active_spans: dict[str, GenAISpan] = {}
        self._lock = Lock()
        self._current_trace_id: str | None = None

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.CLIENT,
        agent_name: str | None = None,
        agent_role: str | None = None,
        model: str = "",
        system: GenAISystem = GenAISystem.CUSTOM,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ):
        """
        Context manager for creating spans.

        Args:
            name: Span name (e.g., "llm.chat", "agent.think")
            kind: Span kind
            agent_name: Name of the agent
            agent_role: Role of the agent
            model: LLM model name
            system: GenAI system
            parent_span_id: Parent span ID for nesting
            attributes: Additional attributes

        Yields:
            GenAISpan for the operation
        """
        span = GenAISpan(
            name=name,
            kind=kind,
            agent_name=agent_name,
            agent_role=agent_role,
            gen_ai_request_model=model,
            gen_ai_system=system,
            parent_span_id=parent_span_id,
            trace_id=self._current_trace_id or str(uuid.uuid4())[:32],
            attributes=attributes or {},
        )

        with self._lock:
            self._active_spans[span.span_id] = span

        try:
            yield span
            span.end(SpanStatus.OK)
        except Exception as e:
            span.set_error(type(e).__name__, str(e))
            span.end(SpanStatus.ERROR)
            raise
        finally:
            self._finish_span(span)

    def _finish_span(self, span: GenAISpan):
        """Finish a span and store it."""
        with self._lock:
            # Remove from active
            self._active_spans.pop(span.span_id, None)

            # Add to completed spans
            self._spans.append(span)

            # Trim if over limit
            if len(self._spans) > self.max_spans:
                self._spans = self._spans[-self.max_spans :]

        # Call export callback
        if self.on_span_end:
            try:
                self.on_span_end(span)
            except Exception as e:
                logger.warning(f"Span end callback failed: {e}")

    def start_trace(self) -> str:
        """Start a new trace and return trace ID."""
        self._current_trace_id = str(uuid.uuid4())[:32]
        return self._current_trace_id

    def end_trace(self):
        """End current trace."""
        self._current_trace_id = None

    def get_spans(
        self,
        trace_id: str | None = None,
        agent_name: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[GenAISpan]:
        """
        Get recorded spans with optional filtering.

        Args:
            trace_id: Filter by trace ID
            agent_name: Filter by agent name
            since: Only spans after this time
            limit: Maximum spans to return

        Returns:
            List of matching spans
        """
        with self._lock:
            spans = list(self._spans)

        # Apply filters
        if trace_id:
            spans = [s for s in spans if s.trace_id == trace_id]
        if agent_name:
            spans = [s for s in spans if s.agent_name == agent_name]
        if since:
            spans = [s for s in spans if s.start_time >= since]

        # Sort by start time (newest first) and limit
        spans.sort(key=lambda s: s.start_time, reverse=True)
        return spans[:limit]

    def get_active_spans(self) -> list[GenAISpan]:
        """Get currently active spans."""
        with self._lock:
            return list(self._active_spans.values())

    def update_tokens(
        self,
        span_id: str,
        input_tokens: int,
        output_tokens: int,
    ):
        """
        Update token usage for a span.

        Args:
            span_id: The span ID to update
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        with self._lock:
            # Check active spans first
            if span_id in self._active_spans:
                span = self._active_spans[span_id]
                span.gen_ai_usage = TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                return

            # Check completed spans
            for span in self._spans:
                if span.span_id == span_id:
                    span.gen_ai_usage = TokenUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                    return

    def get_spans_by_agent(self, agent_name: str, limit: int = 100) -> list[GenAISpan]:
        """Get spans for a specific agent."""
        return self.get_spans(agent_name=agent_name, limit=limit)

    def get_spans_by_model(self, model: str, limit: int = 100) -> list[GenAISpan]:
        """Get spans for a specific model."""
        with self._lock:
            spans = [s for s in self._spans if s.gen_ai_request_model == model]
        spans.sort(key=lambda s: s.start_time, reverse=True)
        return spans[:limit]

    def get_error_spans(self, limit: int = 100) -> list[GenAISpan]:
        """Get spans with errors."""
        with self._lock:
            spans = [s for s in self._spans if s.status == SpanStatus.ERROR]
        spans.sort(key=lambda s: s.start_time, reverse=True)
        return spans[:limit]

    def get_recent_spans(self, limit: int = 100) -> list[GenAISpan]:
        """Get most recent spans."""
        with self._lock:
            spans = list(self._spans)
        spans.sort(key=lambda s: s.start_time, reverse=True)
        return spans[:limit]

    def get_metrics(self) -> dict[str, Any]:
        """
        Get aggregated metrics from spans.

        Returns:
            Dictionary with metrics
        """
        with self._lock:
            spans = list(self._spans)

        if not spans:
            return {
                "total_spans": 0,
                "total_tokens": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "avg_duration_ms": 0.0,
                "error_rate": 0.0,
                "error_count": 0,
            }

        total_input_tokens = sum(s.gen_ai_usage.input_tokens for s in spans)
        total_output_tokens = sum(s.gen_ai_usage.output_tokens for s in spans)
        total_tokens = total_input_tokens + total_output_tokens
        total_duration = sum(s.duration_ms for s in spans)
        error_count = sum(1 for s in spans if s.status == SpanStatus.ERROR)

        # Group by agent
        by_agent: dict[str, dict[str, Any]] = {}
        for span in spans:
            agent = span.agent_name or "unknown"
            if agent not in by_agent:
                by_agent[agent] = {
                    "spans": 0,
                    "tokens": 0,
                    "duration_ms": 0.0,
                    "errors": 0,
                }
            by_agent[agent]["spans"] += 1
            by_agent[agent]["tokens"] += span.gen_ai_usage.total_tokens
            by_agent[agent]["duration_ms"] += span.duration_ms
            if span.status == SpanStatus.ERROR:
                by_agent[agent]["errors"] += 1

        return {
            "total_spans": len(spans),
            "total_tokens": total_tokens,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "avg_duration_ms": total_duration / len(spans) if spans else 0.0,
            "error_rate": error_count / len(spans) if spans else 0.0,
            "error_count": error_count,
            "by_agent": by_agent,
        }

    def clear(self):
        """Clear all recorded spans."""
        with self._lock:
            self._spans.clear()
            self._active_spans.clear()


# ============================================================================
# Global Tracer Instance
# ============================================================================

_global_tracer: LLMTracer | None = None
_tracer_lock = Lock()


def get_global_tracer() -> LLMTracer:
    """Get the global tracer singleton."""
    global _global_tracer

    if _global_tracer is None:
        with _tracer_lock:
            if _global_tracer is None:
                _global_tracer = LLMTracer()

    return _global_tracer


def create_tracer(
    service_name: str = "trading-agents",
    max_spans: int = 10000,
    on_span_end: Callable[[GenAISpan], None] | None = None,
) -> LLMTracer:
    """
    Factory function to create a new tracer.

    Args:
        service_name: Name of the service
        max_spans: Maximum spans to retain
        on_span_end: Callback for span export

    Returns:
        New LLMTracer instance
    """
    return LLMTracer(
        service_name=service_name,
        max_spans=max_spans,
        on_span_end=on_span_end,
    )


# ============================================================================
# Convenience Functions
# ============================================================================


def trace_llm_call(
    model: str,
    system: GenAISystem = GenAISystem.CUSTOM,
    agent_name: str | None = None,
):
    """
    Decorator for tracing LLM calls.

    Usage:
        @trace_llm_call(model="claude-3-opus", system=GenAISystem.ANTHROPIC)
        def call_llm(prompt: str) -> str:
            ...
    """

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            tracer = get_global_tracer()
            with tracer.start_span(
                name=f"llm.{func.__name__}",
                model=model,
                system=system,
                agent_name=agent_name,
            ) as span:
                result = func(*args, **kwargs)
                # Try to extract usage if result has it
                if hasattr(result, "usage"):
                    usage = result.usage
                    span.gen_ai_usage = TokenUsage(
                        input_tokens=getattr(usage, "prompt_tokens", 0),
                        output_tokens=getattr(usage, "completion_tokens", 0),
                    )
                return result

        return wrapper

    return decorator
