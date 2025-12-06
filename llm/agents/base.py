"""
Base Trading Agent

Provides foundation for all LLM-powered trading agents using ReAct framework.

Architecture:
- ReAct (Reasoning + Acting) prompting pattern
- Tool calling interface
- Memory/history tracking
- Error handling and retries
- Decision logging for audit trails

QuantConnect Compatible: Yes
- No blocking operations
- Configurable timeouts
- Defensive error handling
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llm.decision_logger import DecisionLogger
    from llm.reasoning_logger import ReasoningChain, ReasoningLogger


class AgentRole(Enum):
    """Agent role types."""

    # Generic roles
    SUPERVISOR = "supervisor"
    ANALYST = "analyst"
    TRADER = "trader"
    RISK_MANAGER = "risk_manager"
    # Specific analyst roles
    TECHNICAL_ANALYST = "technical_analyst"
    SENTIMENT_ANALYST = "sentiment_analyst"
    NEWS_ANALYST = "news_analyst"  # UPGRADE-014 Feature 6: Multi-Agent Architecture
    # Specific trader roles
    CONSERVATIVE_TRADER = "conservative_trader"
    MODERATE_TRADER = "moderate_trader"
    AGGRESSIVE_TRADER = "aggressive_trader"
    # Specific risk manager roles
    POSITION_RISK_MANAGER = "position_risk_manager"
    PORTFOLIO_RISK_MANAGER = "portfolio_risk_manager"


class ThoughtType(Enum):
    """Types of agent thoughts in ReAct framework."""

    REASONING = "reasoning"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


class TerminationReason(Enum):
    """
    Reasons for ReAct loop termination.

    UPGRADE-014 Category 1: Architecture Enhancements
    """

    FINAL_ANSWER_REACHED = "final_answer"  # Agent produced final answer
    MAX_ITERATIONS = "max_iterations"  # Hit iteration limit
    TIMEOUT = "timeout"  # Exceeded time limit
    ERROR = "error"  # Unrecoverable error
    NO_PROGRESS = "no_progress"  # No progress after retries
    USER_INTERRUPT = "user_interrupt"  # External interruption
    TOOL_FAILURE = "tool_failure"  # All tool retries exhausted


@dataclass
class LoopMetrics:
    """
    Metrics from a ReAct loop execution.

    UPGRADE-014 Category 1: Architecture Enhancements
    """

    iterations: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    retries: int = 0
    execution_time_ms: float = 0.0
    termination_reason: TerminationReason | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "tool_failures": self.tool_failures,
            "retries": self.retries,
            "execution_time_ms": self.execution_time_ms,
            "termination_reason": self.termination_reason.value if self.termination_reason else None,
            "errors": self.errors,
        }


@dataclass
class AgentThought:
    """Single thought in agent's reasoning chain."""

    thought_type: ThoughtType
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Agent's complete response to a query."""

    agent_name: str
    agent_role: AgentRole
    query: str
    thoughts: list[AgentThought]
    final_answer: str
    confidence: float  # 0.0 to 1.0
    tools_used: list[str]
    execution_time_ms: float
    success: bool
    error: str | None = None


@dataclass
class Tool:
    """Tool that agents can call."""

    name: str
    description: str
    function: Callable
    parameters_schema: dict[str, Any]


class TradingAgent(ABC):
    """
    Base class for all trading agents.

    Implements ReAct (Reasoning + Acting) framework:
    1. Think: Reason about the current situation
    2. Act: Call a tool or take an action
    3. Observe: Process the result
    4. Repeat until final answer

    Attributes:
        name: Agent's unique name
        role: Agent's role in the trading system
        system_prompt: Base instructions for the agent
        tools: Available tools the agent can call
        memory: Conversation history
        max_iterations: Maximum reasoning iterations
        timeout_ms: Maximum execution time in milliseconds
    """

    def __init__(
        self,
        name: str,
        role: AgentRole,
        system_prompt: str,
        tools: list[Tool] | None = None,
        max_iterations: int = 5,
        timeout_ms: float = 5000.0,
        llm_client: Any | None = None,
        decision_logger: Optional["DecisionLogger"] = None,
        reasoning_logger: Optional["ReasoningLogger"] = None,
        enable_decision_logging: bool = True,
        enable_reasoning_logging: bool = True,
    ):
        """
        Initialize trading agent.

        Args:
            name: Agent's unique identifier
            role: Agent's role (ANALYST, TRADER, etc.)
            system_prompt: Base instructions defining agent's purpose
            tools: List of tools agent can call
            max_iterations: Max ReAct iterations
            timeout_ms: Max execution time
            llm_client: LLM client (OpenAI, Anthropic, etc.)
            decision_logger: Optional DecisionLogger for audit trails
            reasoning_logger: Optional ReasoningLogger for chain-of-thought logging (Sprint 1)
            enable_decision_logging: Whether to log decisions (default True)
            enable_reasoning_logging: Whether to log reasoning chains (default True)
        """
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.timeout_ms = timeout_ms
        self.llm_client = llm_client
        self.decision_logger = decision_logger
        self.reasoning_logger = reasoning_logger
        self.enable_decision_logging = enable_decision_logging
        self.enable_reasoning_logging = enable_reasoning_logging

        # State
        self.memory: list[dict[str, str]] = []
        self.tool_registry: dict[str, Tool] = {tool.name: tool for tool in self.tools}
        self._active_reasoning_chain: ReasoningChain | None = None

    @abstractmethod
    def analyze(self, query: str, context: dict[str, Any]) -> AgentResponse:
        """
        Main entry point for agent analysis.

        Args:
            query: Question or task for the agent
            context: Additional context (market data, positions, etc.)

        Returns:
            AgentResponse with reasoning chain and final answer
        """
        pass

    def think(self, current_state: str, history: list[AgentThought]) -> AgentThought:
        """
        Reasoning step - agent thinks about current situation.

        Args:
            current_state: Current situation description
            history: Previous thoughts in this session

        Returns:
            Thought containing agent's reasoning
        """
        # Build prompt with history
        prompt = self._build_react_prompt(current_state, history)

        # Get LLM response
        try:
            response = self._call_llm(prompt)
            thought = self._parse_thought(response)
            return thought
        except Exception as e:
            return AgentThought(
                thought_type=ThoughtType.REASONING,
                content=f"Error in reasoning: {e!s}",
                metadata={"error": True},
            )

    def act(self, thought: AgentThought) -> AgentThought:
        """
        Action step - agent calls a tool or takes an action.

        Args:
            thought: Thought containing action to take

        Returns:
            Observation thought with action result
        """
        if thought.thought_type != ThoughtType.ACTION:
            return AgentThought(
                thought_type=ThoughtType.OBSERVATION,
                content="No action specified",
                metadata={"error": True},
            )

        # Extract tool call
        tool_name = thought.metadata.get("tool_name")
        tool_params = thought.metadata.get("tool_params", {})

        if not tool_name or tool_name not in self.tool_registry:
            return AgentThought(
                thought_type=ThoughtType.OBSERVATION,
                content=f"Tool '{tool_name}' not found",
                metadata={"error": True},
            )

        # Call tool
        try:
            tool = self.tool_registry[tool_name]
            result = tool.function(**tool_params)

            return AgentThought(
                thought_type=ThoughtType.OBSERVATION,
                content=f"Tool '{tool_name}' returned: {result}",
                metadata={
                    "tool_name": tool_name,
                    "tool_result": result,
                    "success": True,
                },
            )
        except Exception as e:
            return AgentThought(
                thought_type=ThoughtType.OBSERVATION,
                content=f"Tool '{tool_name}' failed: {e!s}",
                metadata={"error": True, "exception": str(e)},
            )

    def observe(self, observation: AgentThought, context: dict[str, Any]) -> None:
        """
        Observation step - agent processes action result.

        Args:
            observation: Observation thought from action
            context: Additional context to update
        """
        # Update context with observation
        if observation.metadata.get("success"):
            result = observation.metadata.get("tool_result")
            if result:
                context["last_observation"] = result

    def react_loop(
        self,
        query: str,
        context: dict[str, Any],
        max_retries: int = 2,
        no_progress_limit: int = 3,
    ) -> tuple[AgentResponse, LoopMetrics]:
        """
        Execute ReAct loop with structured termination.

        UPGRADE-014 Category 1: Architecture Enhancements

        Implements a robust ReAct loop with:
        - Structured termination reasons
        - Tool failure retries
        - No-progress detection
        - Timeout handling
        - Comprehensive metrics

        Args:
            query: Question or task for the agent
            context: Additional context (market data, positions, etc.)
            max_retries: Maximum retries per tool failure
            no_progress_limit: Iterations without progress before terminating

        Returns:
            Tuple of (AgentResponse, LoopMetrics)
        """
        start_time = time.time()
        metrics = LoopMetrics()
        thoughts: list[AgentThought] = []
        tools_used: list[str] = []
        no_progress_count = 0
        last_thought_content = ""

        # Start reasoning chain if logger available
        self.start_reasoning_chain(query)

        try:
            for iteration in range(self.max_iterations):
                metrics.iterations += 1

                # Check timeout
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms >= self.timeout_ms:
                    metrics.termination_reason = TerminationReason.TIMEOUT
                    break

                # Think step
                current_state = self._build_state_summary(query, context, thoughts)
                thought = self.think(current_state, thoughts)
                thoughts.append(thought)

                # Log reasoning step
                self.log_reasoning_step(
                    thought=thought.content,
                    confidence=0.5,
                )

                # Check for final answer
                if thought.thought_type == ThoughtType.FINAL_ANSWER:
                    metrics.termination_reason = TerminationReason.FINAL_ANSWER_REACHED
                    break

                # Check for no progress
                if thought.content == last_thought_content:
                    no_progress_count += 1
                    if no_progress_count >= no_progress_limit:
                        metrics.termination_reason = TerminationReason.NO_PROGRESS
                        metrics.errors.append(f"No progress after {no_progress_limit} iterations")
                        break
                else:
                    no_progress_count = 0
                    last_thought_content = thought.content

                # Act step (if action thought)
                if thought.thought_type == ThoughtType.ACTION:
                    tool_name = thought.metadata.get("tool_name", "")
                    retry_count = 0

                    while retry_count <= max_retries:
                        metrics.tool_calls += 1
                        observation = self.act(thought)
                        thoughts.append(observation)

                        if observation.metadata.get("error"):
                            metrics.tool_failures += 1
                            retry_count += 1
                            metrics.retries += 1

                            if retry_count > max_retries:
                                metrics.termination_reason = TerminationReason.TOOL_FAILURE
                                metrics.errors.append(f"Tool '{tool_name}' failed after {max_retries} retries")
                                break
                        else:
                            # Success - record tool use and observe
                            if tool_name and tool_name not in tools_used:
                                tools_used.append(tool_name)
                            self.observe(observation, context)
                            break

                    # Exit loop if tool failure terminated it
                    if metrics.termination_reason == TerminationReason.TOOL_FAILURE:
                        break

                # Check for error in reasoning
                if thought.metadata.get("error"):
                    metrics.errors.append(thought.content)

            # Set termination reason if not already set
            if metrics.termination_reason is None:
                metrics.termination_reason = TerminationReason.MAX_ITERATIONS

        except Exception as e:
            metrics.termination_reason = TerminationReason.ERROR
            metrics.errors.append(str(e))
            thoughts.append(
                AgentThought(
                    thought_type=ThoughtType.REASONING,
                    content=f"Loop error: {e!s}",
                    metadata={"error": True, "exception": str(e)},
                )
            )

        # Calculate final execution time
        metrics.execution_time_ms = (time.time() - start_time) * 1000

        # Extract final answer
        final_answer = ""
        confidence = 0.0
        success = False

        for thought in reversed(thoughts):
            if thought.thought_type == ThoughtType.FINAL_ANSWER:
                final_answer = thought.content
                confidence = thought.metadata.get("confidence", 0.7)
                success = True
                break

        # If no final answer, generate one from last reasoning
        if not final_answer:
            for thought in reversed(thoughts):
                if thought.thought_type == ThoughtType.REASONING and not thought.metadata.get("error"):
                    final_answer = f"Analysis incomplete: {thought.content}"
                    confidence = 0.3
                    break

        # Complete reasoning chain
        self.complete_reasoning_chain(final_answer, confidence)

        # Create response
        response = AgentResponse(
            agent_name=self.name,
            agent_role=self.role,
            query=query,
            thoughts=thoughts,
            final_answer=final_answer,
            confidence=confidence,
            tools_used=tools_used,
            execution_time_ms=metrics.execution_time_ms,
            success=success,
            error="; ".join(metrics.errors) if metrics.errors else None,
        )

        return response, metrics

    def _build_state_summary(
        self,
        query: str,
        context: dict[str, Any],
        history: list[AgentThought],
    ) -> str:
        """
        Build current state summary for think step.

        Args:
            query: Original query
            context: Current context
            history: Previous thoughts

        Returns:
            State summary string
        """
        parts = [f"Query: {query}"]

        # Add relevant context
        if context.get("last_observation"):
            parts.append(f"Last observation: {context['last_observation']}")

        # Add recent history summary
        if history:
            recent = history[-3:]  # Last 3 thoughts
            parts.append("Recent thoughts:")
            for t in recent:
                parts.append(f"  - [{t.thought_type.value}] {t.content[:100]}")

        return "\n".join(parts)

    def _build_react_prompt(self, query: str, history: list[AgentThought]) -> str:
        """
        Build ReAct-style prompt with history.

        Format:
        System: [Agent's role and capabilities]
        Query: [User's question]
        [Previous thoughts]
        Next thought:
        """
        prompt_parts = [
            f"System: {self.system_prompt}",
            "",
            f"Query: {query}",
            "",
        ]

        # Add history
        if history:
            prompt_parts.append("Previous thoughts:")
            for thought in history:
                if thought.thought_type == ThoughtType.REASONING:
                    prompt_parts.append(f"Thought: {thought.content}")
                elif thought.thought_type == ThoughtType.ACTION:
                    tool = thought.metadata.get("tool_name", "unknown")
                    prompt_parts.append(f"Action: {tool}")
                    prompt_parts.append(f"Action Input: {thought.metadata.get('tool_params', {})}")
                elif thought.thought_type == ThoughtType.OBSERVATION:
                    prompt_parts.append(f"Observation: {thought.content}")
            prompt_parts.append("")

        # Add tools description
        if self.tools:
            prompt_parts.append("Available tools:")
            for tool in self.tools:
                prompt_parts.append(f"- {tool.name}: {tool.description}")
            prompt_parts.append("")

        prompt_parts.append("What is your next thought? (Thought/Action/Final Answer)")

        return "\n".join(prompt_parts)

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM with prompt.

        Args:
            prompt: Formatted prompt

        Returns:
            LLM response text
        """
        if not self.llm_client:
            # Fallback: Return empty response if no LLM client
            return "Thought: No LLM client configured"

        # Call LLM (implementation depends on client type)
        # This is a placeholder - actual implementation would use
        # OpenAI, Anthropic, or other LLM API
        try:
            # Example for OpenAI-style client:
            # response = self.llm_client.chat.completions.create(
            #     model="gpt-4o-mini",
            #     messages=[{"role": "user", "content": prompt}],
            #     max_tokens=500,
            # )
            # return response.choices[0].message.content

            # Placeholder
            return "Thought: Analysis complete"
        except Exception as e:
            return f"Error calling LLM: {e!s}"

    def _parse_thought(self, llm_response: str) -> AgentThought:
        """
        Parse LLM response into AgentThought.

        Expected formats:
        - "Thought: [reasoning]"
        - "Action: [tool_name]\\nAction Input: {params}"
        - "Final Answer: [answer]"

        Args:
            llm_response: Raw LLM response

        Returns:
            Parsed AgentThought
        """
        lines = llm_response.strip().split("\n")

        if not lines:
            return AgentThought(
                thought_type=ThoughtType.REASONING,
                content="Empty response",
                metadata={"error": True},
            )

        first_line = lines[0].strip()

        # Check for Final Answer
        if first_line.startswith("Final Answer:"):
            content = first_line.replace("Final Answer:", "").strip()
            return AgentThought(
                thought_type=ThoughtType.FINAL_ANSWER,
                content=content,
            )

        # Check for Action
        if first_line.startswith("Action:"):
            tool_name = first_line.replace("Action:", "").strip()
            tool_params = {}

            # Look for Action Input on next line
            if len(lines) > 1 and lines[1].startswith("Action Input:"):
                param_str = lines[1].replace("Action Input:", "").strip()
                try:
                    tool_params = json.loads(param_str)
                except json.JSONDecodeError:
                    # Try eval as fallback (catches various syntax/name errors)
                    try:
                        tool_params = eval(param_str)
                    except Exception as e:
                        logger.debug(f"Failed to parse tool params: {e}")
                        tool_params = {"raw": param_str}

            return AgentThought(
                thought_type=ThoughtType.ACTION,
                content=f"Calling {tool_name}",
                metadata={
                    "tool_name": tool_name,
                    "tool_params": tool_params,
                },
            )

        # Default: Reasoning thought
        content = first_line.replace("Thought:", "").strip()
        return AgentThought(
            thought_type=ThoughtType.REASONING,
            content=content,
        )

    def reset(self) -> None:
        """Reset agent state (clear memory)."""
        self.memory = []

    def get_capabilities(self) -> dict[str, Any]:
        """
        Get agent's capabilities.

        Returns:
            Dictionary describing agent's capabilities
        """
        return {
            "name": self.name,
            "role": self.role.value,
            "tools": [tool.name for tool in self.tools],
            "max_iterations": self.max_iterations,
            "timeout_ms": self.timeout_ms,
            "decision_logging_enabled": self.enable_decision_logging,
            "reasoning_logging_enabled": self.enable_reasoning_logging,
        }

    # =========================================================================
    # Sprint 1: Reasoning Chain Logging Methods
    # =========================================================================

    def start_reasoning_chain(self, task: str) -> Optional["ReasoningChain"]:
        """
        Start a new reasoning chain for logging.

        Sprint 1: Chain-of-Thought Reasoning Logger integration.

        Args:
            task: Description of the task/query being analyzed

        Returns:
            ReasoningChain if logger is configured, None otherwise
        """
        if not self.enable_reasoning_logging or not self.reasoning_logger:
            return None

        try:
            self._active_reasoning_chain = self.reasoning_logger.start_chain(
                agent_name=self.name,
                task=task,
                metadata={"role": self.role.value},
            )
            return self._active_reasoning_chain
        except Exception:
            return None

    def log_reasoning_step(
        self,
        thought: str,
        evidence: str | None = None,
        confidence: float = 0.5,
    ) -> None:
        """
        Log a reasoning step to the active chain.

        Sprint 1: Chain-of-Thought Reasoning Logger integration.

        Args:
            thought: The reasoning thought/observation
            evidence: Supporting evidence (optional)
            confidence: Confidence in this step (0-1)
        """
        if not self._active_reasoning_chain:
            return

        try:
            self._active_reasoning_chain.add_step(
                thought=thought,
                evidence=evidence,
                confidence=confidence,
            )
        except Exception:
            pass

    def complete_reasoning_chain(
        self,
        decision: str,
        confidence: float,
    ) -> Optional["ReasoningChain"]:
        """
        Complete the active reasoning chain with final decision.

        Sprint 1: Chain-of-Thought Reasoning Logger integration.

        Args:
            decision: The final decision made
            confidence: Confidence in the decision (0-1)

        Returns:
            Completed ReasoningChain if successful, None otherwise
        """
        if not self._active_reasoning_chain or not self.reasoning_logger:
            return None

        try:
            chain = self.reasoning_logger.complete_chain(
                chain_id=self._active_reasoning_chain.chain_id,
                decision=decision,
                confidence=confidence,
            )
            self._active_reasoning_chain = None
            return chain
        except Exception:
            self._active_reasoning_chain = None
            return None

    def set_reasoning_logger(self, logger: "ReasoningLogger") -> None:
        """
        Set or update the reasoning logger.

        Sprint 1: Chain-of-Thought Reasoning Logger integration.

        Args:
            logger: ReasoningLogger instance
        """
        self.reasoning_logger = logger

    def log_decision(
        self,
        decision: str,
        confidence: float,
        context: dict[str, Any],
        query: str,
        thoughts: list[AgentThought],
        decision_type: str = "analysis",
        alternatives: list[dict[str, str]] | None = None,
        risk_factors: list[str] | None = None,
        execution_time_ms: float = 0.0,
    ) -> Any | None:
        """
        Log a decision to the decision logger.

        Args:
            decision: The decision made
            confidence: Confidence level (0-1)
            context: Context for the decision
            query: Original query/task
            thoughts: Reasoning chain
            decision_type: Type of decision (trade, analysis, etc.)
            alternatives: Alternatives considered
            risk_factors: Risk factors identified
            execution_time_ms: Time taken to make decision

        Returns:
            AgentDecisionLog if logged, None otherwise
        """
        if not self.enable_decision_logging or not self.decision_logger:
            return None

        try:
            # Import here to avoid circular imports
            from llm.decision_logger import (
                Alternative,
                DecisionType,
                ReasoningStep,
                RiskAssessment,
                RiskLevel,
            )

            # Map decision type string to enum
            type_mapping = {
                "trade": DecisionType.TRADE,
                "analysis": DecisionType.ANALYSIS,
                "risk_assessment": DecisionType.RISK_ASSESSMENT,
                "strategy_selection": DecisionType.STRATEGY_SELECTION,
                "position_sizing": DecisionType.POSITION_SIZING,
                "exit_signal": DecisionType.EXIT_SIGNAL,
                "alert": DecisionType.ALERT,
            }
            dec_type = type_mapping.get(decision_type.lower(), DecisionType.OTHER)

            # Convert thoughts to reasoning steps
            reasoning_chain = [
                ReasoningStep(
                    step_number=i + 1,
                    thought=t.content,
                    confidence=0.5,  # Default if not specified
                )
                for i, t in enumerate(thoughts)
                if t.thought_type == ThoughtType.REASONING
            ]

            # Convert alternatives
            alt_list = []
            if alternatives:
                for alt in alternatives:
                    alt_list.append(
                        Alternative(
                            description=alt.get("description", ""),
                            reason_rejected=alt.get("reason_rejected", ""),
                        )
                    )

            # Create risk assessment
            risk_level = RiskLevel.LOW
            if risk_factors:
                if len(risk_factors) >= 3:
                    risk_level = RiskLevel.HIGH
                elif len(risk_factors) >= 1:
                    risk_level = RiskLevel.MEDIUM

            risk_assessment = RiskAssessment(
                overall_level=risk_level,
                factors=risk_factors or [],
                mitigation_steps=[],
                worst_case_scenario="",
            )

            # Log the decision
            return self.decision_logger.log_decision(
                agent_name=self.name,
                agent_role=self.role.value,
                decision_type=dec_type,
                decision=decision,
                confidence=confidence,
                context=context,
                query=query,
                reasoning_chain=reasoning_chain,
                risk_assessment=risk_assessment,
                alternatives=alt_list,
                execution_time_ms=execution_time_ms,
            )
        except Exception:
            # Don't let logging failures affect agent operation
            return None

    def set_decision_logger(self, logger: "DecisionLogger") -> None:
        """
        Set or update the decision logger.

        Args:
            logger: DecisionLogger instance
        """
        self.decision_logger = logger


def create_tool(
    name: str,
    description: str,
    function: Callable,
    parameters_schema: dict[str, Any] | None = None,
) -> Tool:
    """
    Factory function to create a Tool.

    Args:
        name: Tool name
        description: What the tool does
        function: Function to call
        parameters_schema: JSON schema for parameters

    Returns:
        Tool instance
    """
    return Tool(
        name=name,
        description=description,
        function=function,
        parameters_schema=parameters_schema or {},
    )
