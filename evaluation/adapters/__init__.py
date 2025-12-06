"""
Evaluation Adapters Module.

Provides adapter layers to convert between different data formats used by:
- LLM agents (AgentResponse, AgentThought)
- Evaluation framework (TestCase, Dict)
- Agent-as-Judge (AgentDecision)

This module bridges the gap between the LLM multi-agent system and the
evaluation framework, enabling real-time evaluation of agent decisions.

Version: 1.0 (December 2025)
"""

from evaluation.adapters.agent_response_adapter import (
    AgentResponseAdapter,
    adapt_response_to_decision,
    adapt_response_to_dict,
    batch_adapt_responses,
    create_test_case_from_response,
)


__all__ = [
    "AgentResponseAdapter",
    "adapt_response_to_decision",
    "adapt_response_to_dict",
    "batch_adapt_responses",
    "create_test_case_from_response",
]
