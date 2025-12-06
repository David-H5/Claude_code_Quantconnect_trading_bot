"""
Agent Response Adapters for Evaluation Framework.

Provides utilities to convert agent responses into formats
suitable for evaluation and testing.

Phase 1 - December 2025
"""

from dataclasses import dataclass
from typing import Any, Protocol

from evaluation.agent_as_judge import AgentDecision


class AgentResponse(Protocol):
    """Protocol for agent responses."""

    content: str
    confidence: float
    reasoning: str | None
    metadata: dict[str, Any]


@dataclass
class AgentResponseAdapter:
    """Adapter to convert various agent response formats to standard format."""

    default_confidence: float = 0.5
    require_reasoning: bool = False
    extract_decision: bool = True

    def adapt(self, response: Any) -> dict[str, Any]:
        """
        Adapt an agent response to a standard dictionary format.

        Args:
            response: The agent response to adapt

        Returns:
            Standardized dictionary with content, confidence, reasoning, metadata
        """
        if isinstance(response, dict):
            return self._adapt_dict(response)
        elif isinstance(response, str):
            return self._adapt_string(response)
        elif hasattr(response, "content"):
            return self._adapt_object(response)
        else:
            return {
                "content": str(response),
                "confidence": self.default_confidence,
                "reasoning": None,
                "metadata": {},
            }

    def _adapt_dict(self, response: dict[str, Any]) -> dict[str, Any]:
        """Adapt a dictionary response."""
        return {
            "content": response.get("content", response.get("text", "")),
            "confidence": response.get("confidence", self.default_confidence),
            "reasoning": response.get("reasoning", response.get("rationale")),
            "metadata": response.get("metadata", {}),
        }

    def _adapt_string(self, response: str) -> dict[str, Any]:
        """Adapt a string response."""
        return {
            "content": response,
            "confidence": self.default_confidence,
            "reasoning": None,
            "metadata": {},
        }

    def _adapt_object(self, response: Any) -> dict[str, Any]:
        """Adapt an object with attributes."""
        return {
            "content": getattr(response, "content", str(response)),
            "confidence": getattr(response, "confidence", self.default_confidence),
            "reasoning": getattr(response, "reasoning", None),
            "metadata": getattr(response, "metadata", {}),
        }


def adapt_response_to_dict(
    response: Any,
    default_confidence: float = 0.5,
) -> dict[str, Any]:
    """
    Convert an agent response to a standard dictionary format.

    Args:
        response: The response to adapt
        default_confidence: Default confidence if not provided

    Returns:
        Standardized dictionary
    """
    adapter = AgentResponseAdapter(default_confidence=default_confidence)
    return adapter.adapt(response)


def adapt_response_to_decision(
    response: Any,
    agent_name: str = "unknown",
    query: str = "",
    context: dict[str, Any] | None = None,
) -> AgentDecision:
    """
    Convert an agent response to an AgentDecision for evaluation.

    Args:
        response: The response to adapt
        agent_name: Name of the agent
        query: Original query
        context: Optional context dictionary

    Returns:
        AgentDecision suitable for judge evaluation
    """
    adapted = adapt_response_to_dict(response)

    return AgentDecision(
        agent_name=agent_name,
        query=query,
        response=adapted["content"],
        confidence=adapted["confidence"],
        reasoning=adapted.get("reasoning"),
        context=context or {},
    )


def create_test_case_from_response(
    response: Any,
    expected_score: float | None = None,
    category: str = "general",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Create a test case from an agent response for evaluation.

    Args:
        response: The agent response
        expected_score: Optional expected score for the response
        category: Category of the test case
        tags: Optional tags for filtering

    Returns:
        Test case dictionary suitable for evaluation framework
    """
    adapted = adapt_response_to_dict(response)

    return {
        "input": adapted.get("metadata", {}).get("query", ""),
        "expected_output": expected_score,
        "actual_output": adapted["content"],
        "confidence": adapted["confidence"],
        "reasoning": adapted["reasoning"],
        "category": category,
        "tags": tags or [],
        "metadata": adapted["metadata"],
    }


def batch_adapt_responses(
    responses: list[Any],
    default_confidence: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Adapt multiple responses to standard format.

    Args:
        responses: List of responses to adapt
        default_confidence: Default confidence if not provided

    Returns:
        List of adapted dictionaries
    """
    adapter = AgentResponseAdapter(default_confidence=default_confidence)
    return [adapter.adapt(response) for response in responses]


__all__ = [
    "AgentResponseAdapter",
    "adapt_response_to_decision",
    "adapt_response_to_dict",
    "batch_adapt_responses",
    "create_test_case_from_response",
]
