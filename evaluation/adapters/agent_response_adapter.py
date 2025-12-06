"""
Agent Response Adapter.

Converts between LLM agent response formats and evaluation framework formats.

Key Conversions:
- AgentResponse (llm/agents/base.py) -> Dict[str, Any] (evaluation)
- AgentResponse -> AgentDecision (agent_as_judge.py)
- AgentResponse -> TestCase (evaluation_framework.py)

This adapter solves the critical format mismatch gap identified in Phase 6 research:
- Evaluation tests expect Dict[str, Any]
- LLM agents return AgentResponse objects
- Agent-as-Judge expects AgentDecision objects

Version: 1.0 (December 2025)
"""

from datetime import datetime
from typing import Any

from evaluation.agent_as_judge import AgentDecision

# Import evaluation types
from evaluation.evaluation_framework import TestCase

# Import agent types
from llm.agents.base import AgentResponse, ThoughtType


class AgentResponseAdapter:
    """
    Adapter to convert AgentResponse to various evaluation formats.

    Usage:
        adapter = AgentResponseAdapter()

        # Convert to Dict for standard evaluation
        eval_dict = adapter.to_dict(agent_response)

        # Convert to AgentDecision for judge evaluation
        decision = adapter.to_decision(agent_response, symbol="AAPL")

        # Create TestCase for framework evaluation
        test_case = adapter.to_test_case(
            response=agent_response,
            input_data=market_context,
            expected_action="BUY"
        )
    """

    @staticmethod
    def to_dict(response: AgentResponse) -> dict[str, Any]:
        """
        Convert AgentResponse to Dict format for evaluation framework.

        Args:
            response: AgentResponse from LLM agent

        Returns:
            Dict suitable for evaluation test cases
        """
        # Extract action from final answer (parse common formats)
        action = AgentResponseAdapter._extract_action(response.final_answer)

        # Extract reasoning chain
        reasoning_chain = []
        for thought in response.thoughts:
            reasoning_chain.append(
                {
                    "type": thought.thought_type.value,
                    "content": thought.content,
                    "metadata": thought.metadata,
                }
            )

        return {
            "agent_name": response.agent_name,
            "agent_role": response.agent_role.value,
            "action": action,
            "reasoning": response.final_answer,
            "reasoning_chain": reasoning_chain,
            "confidence": response.confidence,
            "tools_used": response.tools_used,
            "execution_time_ms": response.execution_time_ms,
            "success": response.success,
            "error": response.error,
            "query": response.query,
        }

    @staticmethod
    def to_decision(
        response: AgentResponse,
        symbol: str,
        market_context: dict[str, Any] | None = None,
        decision_id: str | None = None,
    ) -> AgentDecision:
        """
        Convert AgentResponse to AgentDecision for judge evaluation.

        Args:
            response: AgentResponse from LLM agent
            symbol: Trading symbol (e.g., "AAPL", "SPY")
            market_context: Market data context for the decision
            decision_id: Optional unique ID (auto-generated if not provided)

        Returns:
            AgentDecision suitable for Agent-as-a-Judge evaluation
        """
        # Extract action/decision type
        action = AgentResponseAdapter._extract_action(response.final_answer)
        decision_type = AgentResponseAdapter._action_to_decision_type(action)

        # Build reasoning from thoughts
        reasoning_parts = []
        risk_assessment = None

        for thought in response.thoughts:
            if thought.thought_type == ThoughtType.REASONING:
                reasoning_parts.append(thought.content)

                # Check if this thought contains risk assessment
                if "risk" in thought.content.lower():
                    risk_assessment = thought.content
            elif thought.thought_type == ThoughtType.FINAL_ANSWER:
                reasoning_parts.append(f"Conclusion: {thought.content}")

        # Add final answer to reasoning
        reasoning_parts.append(f"Final Answer: {response.final_answer}")

        full_reasoning = "\n\n".join(reasoning_parts)

        # Generate decision ID if not provided
        if decision_id is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            decision_id = f"{response.agent_name}_{symbol}_{timestamp_str}"

        return AgentDecision(
            decision_id=decision_id,
            decision_type=decision_type,
            symbol=symbol,
            reasoning=full_reasoning,
            market_context=market_context or {},
            risk_assessment=risk_assessment,
            confidence=response.confidence,
        )

    @staticmethod
    def to_test_case(
        response: AgentResponse,
        input_data: dict[str, Any],
        expected_action: str | None = None,
        expected_confidence_min: float = 0.6,
        case_id: str | None = None,
        category: str = "success",
        scenario: str = "agent_response_test",
    ) -> TestCase:
        """
        Create TestCase from AgentResponse for evaluation framework.

        Args:
            response: AgentResponse from LLM agent
            input_data: Input context that was provided to the agent
            expected_action: Expected action (defaults to actual action)
            expected_confidence_min: Minimum expected confidence threshold
            case_id: Test case ID (auto-generated if not provided)
            category: Test category (success, edge, failure)
            scenario: Test scenario description

        Returns:
            TestCase for evaluation framework
        """
        actual_output = AgentResponseAdapter.to_dict(response)
        action = actual_output.get("action", "HOLD")

        expected_output = {
            "action": expected_action or action,
            "min_confidence": expected_confidence_min,
            "success": True,
        }

        success_criteria = {
            "action_match": True,
            "confidence_above_min": True,
            "no_errors": True,
        }

        # Generate case_id if not provided
        if case_id is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            case_id = f"test_{response.agent_name}_{timestamp_str}"

        return TestCase(
            case_id=case_id,
            category=category,
            agent_type=response.agent_role.value,
            scenario=scenario,
            input_data=input_data,
            expected_output=expected_output,
            success_criteria=success_criteria,
        )

    @staticmethod
    def _extract_action(final_answer: str) -> str:
        """
        Extract trading action from final answer text.

        Parses common formats:
        - "Action: BUY" / "Action: SELL" / "Action: HOLD"
        - "Recommendation: BUY"
        - "I recommend buying..."
        - JSON with "action" field

        Args:
            final_answer: Agent's final answer text

        Returns:
            Extracted action (BUY, SELL, HOLD, or UNKNOWN)
        """
        text = final_answer.upper()

        # Check for explicit action statements
        action_indicators = {
            "BUY": ["BUY", "LONG", "PURCHASE", "ACQUIRE"],
            "SELL": ["SELL", "SHORT", "EXIT", "CLOSE"],
            "HOLD": ["HOLD", "WAIT", "NO ACTION", "MAINTAIN"],
        }

        # Check for "Action: X" format first
        for action, keywords in action_indicators.items():
            for keyword in keywords:
                if f"ACTION: {keyword}" in text or f"RECOMMENDATION: {keyword}" in text:
                    return action

        # Check for keywords in general text
        for action, keywords in action_indicators.items():
            for keyword in keywords:
                if keyword in text:
                    return action

        return "UNKNOWN"

    @staticmethod
    def _action_to_decision_type(action: str) -> str:
        """
        Convert action string to decision type for AgentDecision.

        Args:
            action: Action string (BUY, SELL, HOLD, etc.)

        Returns:
            Decision type string (lowercase)
        """
        action_map = {
            "BUY": "buy",
            "SELL": "sell",
            "HOLD": "hold",
            "LONG": "buy",
            "SHORT": "sell",
            "EXIT": "sell",
            "CLOSE": "sell",
            "UNKNOWN": "hold",
        }
        return action_map.get(action.upper(), "hold")


# Convenience functions for quick conversions
def adapt_response_to_dict(response: AgentResponse) -> dict[str, Any]:
    """Convert AgentResponse to Dict format."""
    return AgentResponseAdapter.to_dict(response)


def adapt_response_to_decision(
    response: AgentResponse,
    symbol: str,
    market_context: dict[str, Any] | None = None,
) -> AgentDecision:
    """Convert AgentResponse to AgentDecision."""
    return AgentResponseAdapter.to_decision(response, symbol, market_context)


def create_test_case_from_response(
    response: AgentResponse,
    input_data: dict[str, Any],
    expected_action: str | None = None,
) -> TestCase:
    """Create TestCase from AgentResponse."""
    return AgentResponseAdapter.to_test_case(response, input_data, expected_action)


def batch_adapt_responses(
    responses: list[AgentResponse],
    symbols: list[str] | None = None,
    market_contexts: list[dict[str, Any]] | None = None,
) -> list[AgentDecision]:
    """
    Batch convert multiple AgentResponses to AgentDecisions.

    Args:
        responses: List of AgentResponse objects
        symbols: List of symbols (one per response, or single symbol for all)
        market_contexts: List of market contexts (one per response, or single for all)

    Returns:
        List of AgentDecision objects
    """
    decisions = []

    for i, response in enumerate(responses):
        # Get symbol for this response
        if symbols is None:
            symbol = "UNKNOWN"
        elif len(symbols) == 1:
            symbol = symbols[0]
        else:
            symbol = symbols[i] if i < len(symbols) else "UNKNOWN"

        # Get context for this response
        if market_contexts is None:
            context = {}
        elif len(market_contexts) == 1:
            context = market_contexts[0]
        else:
            context = market_contexts[i] if i < len(market_contexts) else {}

        decisions.append(AgentResponseAdapter.to_decision(response, symbol, context))

    return decisions
