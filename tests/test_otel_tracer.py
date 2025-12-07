"""
Tests for OpenTelemetry GenAI Tracer

UPGRADE-014 Category 2: Observability & Debugging
"""

import time
from datetime import datetime

import pytest

from observability.otel_tracer import (
    GenAISpan,
    GenAISystem,
    LLMTracer,
    SpanKind,
    SpanStatus,
    TokenUsage,
    create_tracer,
    get_global_tracer,
)


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_default_values(self):
        """Test default token usage values."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_auto_total_calculation(self):
        """Test automatic total token calculation."""
        usage = TokenUsage(input_tokens=100, output_tokens=200)
        assert usage.total_tokens == 300

    def test_to_dict(self):
        """Test serialization."""
        usage = TokenUsage(input_tokens=100, output_tokens=200)
        d = usage.to_dict()
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 200
        assert d["total_tokens"] == 300


class TestGenAISpan:
    """Tests for GenAISpan dataclass."""

    def test_span_creation(self):
        """Test creating a span."""
        span = GenAISpan(
            name="test_span",
            kind=SpanKind.CLIENT,
            gen_ai_system=GenAISystem.ANTHROPIC,
            gen_ai_request_model="claude-3-sonnet",
        )

        assert span.name == "test_span"
        assert span.kind == SpanKind.CLIENT
        assert span.gen_ai_system == GenAISystem.ANTHROPIC
        assert span.gen_ai_request_model == "claude-3-sonnet"
        assert span.span_id is not None
        assert span.trace_id is not None

    def test_span_timestamps(self):
        """Test span has timestamps."""
        span = GenAISpan(
            name="test",
            kind=SpanKind.CLIENT,
            gen_ai_system=GenAISystem.OPENAI,
            gen_ai_request_model="gpt-4",
        )

        assert span.start_time is not None
        assert isinstance(span.start_time, datetime)

    def test_to_otel_attributes(self):
        """Test OpenTelemetry attribute format."""
        span = GenAISpan(
            name="analyze",
            kind=SpanKind.CLIENT,
            gen_ai_system=GenAISystem.ANTHROPIC,
            gen_ai_request_model="claude-3-sonnet",
            agent_name="technical_analyst",
            gen_ai_usage=TokenUsage(input_tokens=500, output_tokens=1000),
        )

        attrs = span.to_otel_attributes()

        assert attrs["gen_ai.system"] == "anthropic"
        assert attrs["gen_ai.request.model"] == "claude-3-sonnet"
        assert attrs["gen_ai.usage.input_tokens"] == 500
        assert attrs["gen_ai.usage.output_tokens"] == 1000
        assert attrs["agent.name"] == "technical_analyst"

    def test_end_span(self):
        """Test ending a span."""
        span = GenAISpan(
            name="test",
            kind=SpanKind.CLIENT,
            gen_ai_system=GenAISystem.OPENAI,
            gen_ai_request_model="gpt-4",
        )

        assert span.end_time is None
        span.end()
        assert span.end_time is not None
        assert span.status == SpanStatus.OK

    def test_end_span_with_error(self):
        """Test ending a span with error."""
        span = GenAISpan(
            name="test",
            kind=SpanKind.CLIENT,
            gen_ai_system=GenAISystem.OPENAI,
            gen_ai_request_model="gpt-4",
        )

        span.end(status=SpanStatus.ERROR, message="Connection timeout")
        assert span.status == SpanStatus.ERROR
        assert span.status_message == "Connection timeout"

    def test_duration_calculation(self):
        """Test duration is calculated correctly."""
        span = GenAISpan(
            name="test",
            kind=SpanKind.CLIENT,
            gen_ai_system=GenAISystem.OPENAI,
            gen_ai_request_model="gpt-4",
        )

        time.sleep(0.05)  # 50ms
        span.end()

        assert span.duration_ms >= 50.0
        assert span.duration_ms < 200.0  # Should not be too long


class TestLLMTracer:
    """Tests for LLMTracer class."""

    def test_tracer_creation(self):
        """Test creating a tracer."""
        tracer = create_tracer(service_name="test_service")
        assert tracer.service_name == "test_service"

    def test_start_span_context_manager(self):
        """Test span creation via context manager."""
        tracer = LLMTracer()

        with tracer.start_span(
            name="analyze",
            kind=SpanKind.CLIENT,
            agent_name="analyst",
            model="claude-3-sonnet",
            system=GenAISystem.ANTHROPIC,
        ) as span:
            assert span.name == "analyze"
            assert span.agent_name == "analyst"
            assert span.gen_ai_request_model == "claude-3-sonnet"
            assert span.end_time is None

        # Span should be ended after context
        assert span.end_time is not None
        assert span.status == SpanStatus.OK

    def test_span_records_error_on_exception(self):
        """Test that exceptions are recorded in span."""
        tracer = LLMTracer()

        with (
            pytest.raises(ValueError),
            tracer.start_span(
                name="failing",
                kind=SpanKind.CLIENT,
                agent_name="test",
                model="test",
            ) as span,
        ):
            raise ValueError("Test error")

        assert span.status == SpanStatus.ERROR
        assert "Test error" in span.error_message

    def test_update_tokens(self):
        """Test updating token counts."""
        tracer = LLMTracer()

        with tracer.start_span(
            name="test",
            kind=SpanKind.CLIENT,
            agent_name="test",
            model="gpt-4",
        ) as span:
            tracer.update_tokens(span.span_id, 100, 200)

        assert span.gen_ai_usage.input_tokens == 100
        assert span.gen_ai_usage.output_tokens == 200

    def test_get_spans_by_agent(self):
        """Test filtering spans by agent."""
        tracer = LLMTracer()

        with tracer.start_span(name="test1", kind=SpanKind.CLIENT, agent_name="agent_a", model="gpt-4"):
            pass

        with tracer.start_span(name="test2", kind=SpanKind.CLIENT, agent_name="agent_b", model="gpt-4"):
            pass

        with tracer.start_span(name="test3", kind=SpanKind.CLIENT, agent_name="agent_a", model="gpt-4"):
            pass

        agent_a_spans = tracer.get_spans_by_agent("agent_a")
        assert len(agent_a_spans) == 2

        agent_b_spans = tracer.get_spans_by_agent("agent_b")
        assert len(agent_b_spans) == 1

    def test_get_spans_by_model(self):
        """Test filtering spans by model."""
        tracer = LLMTracer()

        with tracer.start_span(name="test1", kind=SpanKind.CLIENT, agent_name="test", model="gpt-4"):
            pass

        with tracer.start_span(name="test2", kind=SpanKind.CLIENT, agent_name="test", model="claude-3-sonnet"):
            pass

        gpt4_spans = tracer.get_spans_by_model("gpt-4")
        assert len(gpt4_spans) == 1

    def test_get_error_spans(self):
        """Test getting error spans."""
        tracer = LLMTracer()

        with tracer.start_span(name="success", kind=SpanKind.CLIENT, agent_name="test", model="gpt-4"):
            pass

        try:
            with tracer.start_span(name="failure", kind=SpanKind.CLIENT, agent_name="test", model="gpt-4"):
                raise Exception("Test error")
        except Exception:
            pass

        error_spans = tracer.get_error_spans()
        assert len(error_spans) == 1
        assert error_spans[0].name == "failure"

    def test_get_metrics(self):
        """Test aggregated metrics."""
        tracer = LLMTracer()

        with tracer.start_span(name="test1", kind=SpanKind.CLIENT, agent_name="test", model="gpt-4") as span:
            tracer.update_tokens(span.span_id, 100, 200)

        with tracer.start_span(name="test2", kind=SpanKind.CLIENT, agent_name="test", model="gpt-4") as span:
            tracer.update_tokens(span.span_id, 150, 250)

        metrics = tracer.get_metrics()

        assert metrics["total_spans"] == 2
        assert metrics["total_input_tokens"] == 250
        assert metrics["total_output_tokens"] == 450
        assert metrics["error_count"] == 0
        assert metrics["avg_duration_ms"] > 0

    def test_global_tracer_singleton(self):
        """Test global tracer is singleton."""
        tracer1 = get_global_tracer()
        tracer2 = get_global_tracer()
        assert tracer1 is tracer2

    def test_max_spans_limit(self):
        """Test that old spans are removed when limit reached."""
        tracer = LLMTracer(max_spans=5)

        for i in range(10):
            with tracer.start_span(name=f"test_{i}", kind=SpanKind.CLIENT, agent_name="test", model="gpt-4"):
                pass

        assert len(tracer.get_recent_spans(limit=100)) <= 5

    def test_clear_spans(self):
        """Test clearing all spans."""
        tracer = LLMTracer()

        with tracer.start_span(name="test", kind=SpanKind.CLIENT, agent_name="test", model="gpt-4"):
            pass

        assert len(tracer.get_recent_spans()) == 1

        tracer.clear()

        assert len(tracer.get_recent_spans()) == 0


class TestGenAISystem:
    """Tests for GenAISystem enum."""

    def test_all_systems_exist(self):
        """Test all expected systems exist."""
        systems = [
            GenAISystem.ANTHROPIC,
            GenAISystem.OPENAI,
            GenAISystem.AZURE_OPENAI,
            GenAISystem.AWS_BEDROCK,
            GenAISystem.GOOGLE_VERTEX,
            GenAISystem.CUSTOM,
        ]
        assert len(systems) == 6

    def test_system_values(self):
        """Test system string values."""
        assert GenAISystem.ANTHROPIC.value == "anthropic"
        assert GenAISystem.OPENAI.value == "openai"


class TestSpanKind:
    """Tests for SpanKind enum."""

    def test_all_kinds_exist(self):
        """Test all expected span kinds exist."""
        kinds = [
            SpanKind.CLIENT,
            SpanKind.SERVER,
            SpanKind.INTERNAL,
        ]
        assert len(kinds) == 3
