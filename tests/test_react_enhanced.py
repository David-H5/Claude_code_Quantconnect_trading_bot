"""
Tests for Enhanced ReAct Loop

UPGRADE-014 Category 1: Architecture Enhancements
Tests for structured termination, retries, and metrics tracking.
"""

import time

import pytest

from llm.agents.base import (
    AgentResponse,
    AgentRole,
    AgentThought,
    LoopMetrics,
    TerminationReason,
    ThoughtType,
    TradingAgent,
    create_tool,
)


class MockTradingAgent(TradingAgent):
    """Mock agent for testing ReAct loop."""

    def __init__(
        self,
        name: str = "test_agent",
        tools: list = None,
        max_iterations: int = 5,
        timeout_ms: float = 5000.0,
    ):
        super().__init__(
            name=name,
            role=AgentRole.ANALYST,
            system_prompt="Test agent for ReAct loop",
            tools=tools or [],
            max_iterations=max_iterations,
            timeout_ms=timeout_ms,
        )
        self.think_responses = []
        self.think_index = 0

    def analyze(self, query: str, context: dict) -> AgentResponse:
        """Use react_loop for analysis."""
        response, metrics = self.react_loop(query, context)
        return response

    def set_think_responses(self, responses: list):
        """Set predetermined responses for think()."""
        self.think_responses = responses
        self.think_index = 0

    def think(self, current_state: str, history: list) -> AgentThought:
        """Return predetermined thought or default."""
        if self.think_index < len(self.think_responses):
            thought = self.think_responses[self.think_index]
            self.think_index += 1
            return thought
        return AgentThought(
            thought_type=ThoughtType.FINAL_ANSWER,
            content="Default final answer",
        )


class TestTerminationReason:
    """Tests for TerminationReason enum."""

    def test_all_reasons_exist(self):
        """Test all expected termination reasons exist."""
        reasons = [
            TerminationReason.FINAL_ANSWER_REACHED,
            TerminationReason.MAX_ITERATIONS,
            TerminationReason.TIMEOUT,
            TerminationReason.ERROR,
            TerminationReason.NO_PROGRESS,
            TerminationReason.USER_INTERRUPT,
            TerminationReason.TOOL_FAILURE,
        ]
        assert len(reasons) == 7

    def test_reason_values(self):
        """Test reason value strings."""
        assert TerminationReason.FINAL_ANSWER_REACHED.value == "final_answer"
        assert TerminationReason.MAX_ITERATIONS.value == "max_iterations"
        assert TerminationReason.TIMEOUT.value == "timeout"


class TestLoopMetrics:
    """Tests for LoopMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = LoopMetrics()

        assert metrics.iterations == 0
        assert metrics.tool_calls == 0
        assert metrics.tool_failures == 0
        assert metrics.retries == 0
        assert metrics.execution_time_ms == 0.0
        assert metrics.termination_reason is None
        assert metrics.errors == []

    def test_to_dict(self):
        """Test serialization."""
        metrics = LoopMetrics(
            iterations=3,
            tool_calls=5,
            tool_failures=1,
            retries=2,
            execution_time_ms=150.0,
            termination_reason=TerminationReason.FINAL_ANSWER_REACHED,
            errors=["Minor error"],
        )

        d = metrics.to_dict()
        assert d["iterations"] == 3
        assert d["tool_calls"] == 5
        assert d["tool_failures"] == 1
        assert d["retries"] == 2
        assert d["execution_time_ms"] == 150.0
        assert d["termination_reason"] == "final_answer"
        assert d["errors"] == ["Minor error"]

    def test_to_dict_no_termination(self):
        """Test serialization without termination reason."""
        metrics = LoopMetrics()
        d = metrics.to_dict()
        assert d["termination_reason"] is None


class TestReactLoopBasic:
    """Basic tests for react_loop method."""

    def test_simple_final_answer(self):
        """Test immediate final answer termination."""
        agent = MockTradingAgent()
        agent.set_think_responses(
            [
                AgentThought(
                    thought_type=ThoughtType.FINAL_ANSWER,
                    content="SPY is bullish",
                )
            ]
        )

        response, metrics = agent.react_loop("Analyze SPY", {})

        assert response.success is True
        assert response.final_answer == "SPY is bullish"
        assert metrics.termination_reason == TerminationReason.FINAL_ANSWER_REACHED
        assert metrics.iterations == 1
        assert metrics.tool_calls == 0

    def test_reasoning_then_answer(self):
        """Test reasoning steps before final answer."""
        agent = MockTradingAgent()
        agent.set_think_responses(
            [
                AgentThought(
                    thought_type=ThoughtType.REASONING,
                    content="Looking at RSI levels",
                ),
                AgentThought(
                    thought_type=ThoughtType.REASONING,
                    content="RSI is at 45, neutral",
                ),
                AgentThought(
                    thought_type=ThoughtType.FINAL_ANSWER,
                    content="SPY is neutral",
                ),
            ]
        )

        response, metrics = agent.react_loop("Analyze SPY", {})

        assert response.success is True
        assert response.final_answer == "SPY is neutral"
        assert metrics.iterations == 3
        assert len(response.thoughts) >= 3


class TestReactLoopTermination:
    """Tests for different termination scenarios."""

    def test_max_iterations(self):
        """Test termination by max iterations."""
        agent = MockTradingAgent(max_iterations=3)
        # Only provide reasoning, no final answer
        agent.set_think_responses(
            [
                AgentThought(thought_type=ThoughtType.REASONING, content="Step 1"),
                AgentThought(thought_type=ThoughtType.REASONING, content="Step 2"),
                AgentThought(thought_type=ThoughtType.REASONING, content="Step 3"),
                AgentThought(thought_type=ThoughtType.REASONING, content="Step 4"),
            ]
        )

        response, metrics = agent.react_loop("Analyze", {})

        assert metrics.termination_reason == TerminationReason.MAX_ITERATIONS
        assert metrics.iterations == 3
        assert response.success is False

    def test_timeout(self):
        """Test termination by timeout."""
        agent = MockTradingAgent(timeout_ms=50)  # Very short timeout

        # Set up responses that don't terminate (so timeout can trigger)
        agent.set_think_responses(
            [
                AgentThought(thought_type=ThoughtType.REASONING, content="Still thinking..."),
                AgentThought(thought_type=ThoughtType.REASONING, content="Still thinking..."),
                AgentThought(thought_type=ThoughtType.REASONING, content="Still thinking..."),
            ]
        )

        # Make think() slow enough to exceed timeout
        original_think = agent.think

        def slow_think(state, history):
            time.sleep(0.1)  # 100ms delay, timeout is 50ms
            return original_think(state, history)

        agent.think = slow_think

        response, metrics = agent.react_loop("Analyze", {})

        assert metrics.termination_reason == TerminationReason.TIMEOUT

    def test_no_progress(self):
        """Test termination by no progress."""
        agent = MockTradingAgent()
        # Same content repeated
        agent.set_think_responses(
            [
                AgentThought(thought_type=ThoughtType.REASONING, content="Same thought"),
                AgentThought(thought_type=ThoughtType.REASONING, content="Same thought"),
                AgentThought(thought_type=ThoughtType.REASONING, content="Same thought"),
                AgentThought(thought_type=ThoughtType.REASONING, content="Same thought"),
            ]
        )

        response, metrics = agent.react_loop("Analyze", {}, no_progress_limit=3)

        assert metrics.termination_reason == TerminationReason.NO_PROGRESS
        assert "No progress" in metrics.errors[0]


class TestReactLoopToolCalls:
    """Tests for tool call handling."""

    @pytest.fixture
    def successful_tool(self):
        """Create a successful tool."""
        return create_tool(
            name="get_price",
            description="Get current price",
            function=lambda symbol: {"price": 450.0, "symbol": symbol},
            parameters_schema={"symbol": {"type": "string"}},
        )

    @pytest.fixture
    def failing_tool(self):
        """Create a failing tool."""

        def fail_fn(**kwargs):
            raise Exception("Tool error")

        return create_tool(
            name="bad_tool",
            description="A failing tool",
            function=fail_fn,
            parameters_schema={},
        )

    def test_successful_tool_call(self, successful_tool):
        """Test successful tool execution."""
        agent = MockTradingAgent(tools=[successful_tool])
        agent.set_think_responses(
            [
                AgentThought(
                    thought_type=ThoughtType.ACTION,
                    content="Getting price",
                    metadata={"tool_name": "get_price", "tool_params": {"symbol": "SPY"}},
                ),
                AgentThought(
                    thought_type=ThoughtType.FINAL_ANSWER,
                    content="Price is 450",
                ),
            ]
        )

        response, metrics = agent.react_loop("Get SPY price", {})

        assert metrics.tool_calls == 1
        assert metrics.tool_failures == 0
        assert "get_price" in response.tools_used

    def test_tool_failure_with_retry(self, failing_tool):
        """Test tool failure triggers retries."""
        agent = MockTradingAgent(tools=[failing_tool])
        agent.set_think_responses(
            [
                AgentThought(
                    thought_type=ThoughtType.ACTION,
                    content="Calling bad tool",
                    metadata={"tool_name": "bad_tool", "tool_params": {}},
                ),
            ]
        )

        response, metrics = agent.react_loop("Test", {}, max_retries=2)

        # Initial call + 2 retries = 3 total calls
        assert metrics.tool_calls == 3
        assert metrics.tool_failures == 3
        # Retries counter tracks total retry attempts (3 = initial + 2 retries)
        assert metrics.retries == 3
        assert metrics.termination_reason == TerminationReason.TOOL_FAILURE

    def test_unknown_tool(self):
        """Test handling of unknown tool."""
        agent = MockTradingAgent(tools=[])
        agent.set_think_responses(
            [
                AgentThought(
                    thought_type=ThoughtType.ACTION,
                    content="Calling unknown",
                    metadata={"tool_name": "nonexistent", "tool_params": {}},
                ),
                AgentThought(
                    thought_type=ThoughtType.FINAL_ANSWER,
                    content="Done",
                ),
            ]
        )

        # Use max_retries=0 to disable retries for this test
        response, metrics = agent.react_loop("Test", {}, max_retries=0)

        # Unknown tool counts as call but fails (no retries)
        assert metrics.tool_calls == 1
        assert metrics.tool_failures == 1


class TestReactLoopContext:
    """Tests for context handling."""

    def test_observation_updates_context(self):
        """Test that successful observations update context."""
        tool = create_tool(
            name="get_data",
            description="Get data",
            function=lambda: {"value": 100},
            parameters_schema={},
        )

        agent = MockTradingAgent(tools=[tool])
        agent.set_think_responses(
            [
                AgentThought(
                    thought_type=ThoughtType.ACTION,
                    content="Get data",
                    metadata={"tool_name": "get_data", "tool_params": {}},
                ),
                AgentThought(
                    thought_type=ThoughtType.FINAL_ANSWER,
                    content="Got value 100",
                ),
            ]
        )

        context = {}
        response, metrics = agent.react_loop("Test", context)

        assert "last_observation" in context
        assert context["last_observation"]["value"] == 100


class TestReactLoopErrorHandling:
    """Tests for error handling."""

    def test_exception_during_loop(self):
        """Test exception handling during loop."""
        agent = MockTradingAgent()

        # Make think() raise exception
        def failing_think(state, history):
            raise Exception("Critical error")

        agent.think = failing_think

        response, metrics = agent.react_loop("Test", {})

        assert metrics.termination_reason == TerminationReason.ERROR
        assert "Critical error" in metrics.errors[0]

    def test_error_thought_recorded(self):
        """Test that error thoughts are recorded in metrics."""
        agent = MockTradingAgent()
        agent.set_think_responses(
            [
                AgentThought(
                    thought_type=ThoughtType.REASONING,
                    content="Error in reasoning: LLM unavailable",
                    metadata={"error": True},
                ),
                AgentThought(
                    thought_type=ThoughtType.FINAL_ANSWER,
                    content="Fallback answer",
                ),
            ]
        )

        response, metrics = agent.react_loop("Test", {})

        assert "Error in reasoning: LLM unavailable" in metrics.errors


class TestReactLoopMetrics:
    """Tests for metric accuracy."""

    def test_execution_time_tracked(self):
        """Test execution time is properly tracked."""
        agent = MockTradingAgent()
        agent.set_think_responses(
            [
                AgentThought(
                    thought_type=ThoughtType.FINAL_ANSWER,
                    content="Quick answer",
                ),
            ]
        )

        response, metrics = agent.react_loop("Test", {})

        assert metrics.execution_time_ms > 0
        assert metrics.execution_time_ms == response.execution_time_ms

    def test_metrics_match_response(self):
        """Test that metrics match response data."""
        tool = create_tool(
            name="tool1",
            description="Test tool",
            function=lambda: "result",
            parameters_schema={},
        )

        agent = MockTradingAgent(tools=[tool])
        agent.set_think_responses(
            [
                AgentThought(
                    thought_type=ThoughtType.ACTION,
                    content="Call tool",
                    metadata={"tool_name": "tool1", "tool_params": {}},
                ),
                AgentThought(
                    thought_type=ThoughtType.FINAL_ANSWER,
                    content="Done",
                    metadata={"confidence": 0.85},
                ),
            ]
        )

        response, metrics = agent.react_loop("Test", {})

        assert len(response.tools_used) == 1
        assert metrics.tool_calls == 1


class TestBuildStateSummary:
    """Tests for _build_state_summary method."""

    def test_empty_context(self):
        """Test summary with empty context."""
        agent = MockTradingAgent()

        summary = agent._build_state_summary("Test query", {}, [])

        assert "Query: Test query" in summary

    def test_with_observation(self):
        """Test summary with last observation."""
        agent = MockTradingAgent()
        context = {"last_observation": {"price": 450}}

        summary = agent._build_state_summary("Test", context, [])

        # Summary formats key as "last observation" (with space)
        assert "observation" in summary.lower()

    def test_with_history(self):
        """Test summary with thought history."""
        agent = MockTradingAgent()
        history = [
            AgentThought(thought_type=ThoughtType.REASONING, content="First thought"),
            AgentThought(thought_type=ThoughtType.REASONING, content="Second thought"),
        ]

        summary = agent._build_state_summary("Test", {}, history)

        assert "Recent thoughts" in summary
        assert "reasoning" in summary.lower()


class TestReactLoopIntegration:
    """Integration tests for full ReAct loop scenarios."""

    def test_multi_tool_analysis(self):
        """Test analysis using multiple tools."""
        price_tool = create_tool(
            name="get_price",
            description="Get price",
            function=lambda symbol: {"price": 450.0},
            parameters_schema={"symbol": {"type": "string"}},
        )

        rsi_tool = create_tool(
            name="get_rsi",
            description="Get RSI",
            function=lambda symbol: {"rsi": 55},
            parameters_schema={"symbol": {"type": "string"}},
        )

        agent = MockTradingAgent(tools=[price_tool, rsi_tool])
        agent.set_think_responses(
            [
                AgentThought(
                    thought_type=ThoughtType.ACTION,
                    content="Get price",
                    metadata={"tool_name": "get_price", "tool_params": {"symbol": "SPY"}},
                ),
                AgentThought(
                    thought_type=ThoughtType.ACTION,
                    content="Get RSI",
                    metadata={"tool_name": "get_rsi", "tool_params": {"symbol": "SPY"}},
                ),
                AgentThought(
                    thought_type=ThoughtType.REASONING,
                    content="Price is 450, RSI is 55 - neutral",
                ),
                AgentThought(
                    thought_type=ThoughtType.FINAL_ANSWER,
                    content="SPY is neutral with RSI at 55",
                ),
            ]
        )

        response, metrics = agent.react_loop("Full SPY analysis", {})

        assert response.success is True
        assert metrics.tool_calls == 2
        assert len(response.tools_used) == 2
        assert "get_price" in response.tools_used
        assert "get_rsi" in response.tools_used

    def test_fallback_answer_on_no_final(self):
        """Test fallback answer when no final answer reached."""
        agent = MockTradingAgent(max_iterations=2)
        agent.set_think_responses(
            [
                AgentThought(
                    thought_type=ThoughtType.REASONING,
                    content="Analyzing market conditions",
                ),
                AgentThought(
                    thought_type=ThoughtType.REASONING,
                    content="Further analysis needed",
                ),
            ]
        )

        response, metrics = agent.react_loop("Test", {})

        assert response.success is False
        assert "Analysis incomplete" in response.final_answer
        assert response.confidence == 0.3
