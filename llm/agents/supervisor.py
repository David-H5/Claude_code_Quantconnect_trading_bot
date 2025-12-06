"""
Supervisor Agent Implementation

The orchestrator of the multi-agent trading system. Coordinates team analyses
and makes final trading decisions.

UPGRADE-005 Enhanced: Now includes Bull/Bear debate integration for high-stakes decisions.

QuantConnect Compatible: Yes
"""

import json
import time
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from llm.agents.base import (
    AgentResponse,
    AgentRole,
    AgentThought,
    ThoughtType,
    TradingAgent,
)
from llm.agents.safe_agent_wrapper import (
    RiskTierConfig,
    SafeAgentWrapper,
    wrap_agent_with_safety,
)
from llm.clients import AnthropicClient, ClaudeModel
from llm.prompts import get_prompt, get_registry
from models.circuit_breaker import TradingCircuitBreaker
from models.exceptions import PromptVersionError


if TYPE_CHECKING:
    from llm.agents.debate_mechanism import BullBearDebate, DebateResult


class DebateTriggerReason(Enum):
    """Reasons why a debate was triggered."""

    HIGH_POSITION_SIZE = "high_position_size"
    LOW_CONFIDENCE = "low_confidence"
    CONFLICTING_SIGNALS = "conflicting_signals"
    HIGH_IMPACT_EVENT = "high_impact_event"
    UNUSUAL_MARKET = "unusual_market"
    MANUAL_REQUEST = "manual_request"


class SupervisorAgent(TradingAgent):
    """
    Supervisor agent - orchestrates the multi-agent trading system.

    Responsibilities:
    - Gather analyses from all team members
    - Facilitate multi-agent debate
    - Synthesize conflicting views
    - Make final trading decisions
    - Track historical performance
    - Integrate multi-modal signals

    Uses Claude Opus 4 for deep reasoning on complex decisions.
    """

    def __init__(
        self,
        llm_client: AnthropicClient,
        version: str = "active",
        max_iterations: int = 3,
        timeout_ms: float = 10000.0,
        debate_mechanism: Optional["BullBearDebate"] = None,
        debate_threshold: float = 0.10,
        min_debate_confidence: float = 0.70,
        signal_conflict_threshold: float = 0.30,
    ):
        """
        Initialize supervisor agent.

        Args:
            llm_client: Anthropic API client
            version: Prompt version to use ("active", "v1.0", "v1.1", "v2.0")
            max_iterations: Max ReAct iterations
            timeout_ms: Max execution time
            debate_mechanism: Optional BullBearDebate for high-stakes decisions
            debate_threshold: Position size % that triggers debate (default 10%)
            min_debate_confidence: Confidence below this triggers debate (default 70%)
            signal_conflict_threshold: Signal disagreement threshold for debate (default 30%)
        """
        # Get prompt template
        prompt_version = get_prompt(AgentRole.SUPERVISOR, version=version)
        if not prompt_version:
            raise PromptVersionError(
                agent_role="SUPERVISOR",
                version=version,
            )

        super().__init__(
            name="Supervisor",
            role=AgentRole.SUPERVISOR,
            system_prompt=prompt_version.template,
            tools=[],  # Supervisor doesn't call tools, just synthesizes
            max_iterations=max_iterations,
            timeout_ms=timeout_ms,
            llm_client=llm_client,
        )

        self.prompt_version = prompt_version
        self.registry = get_registry()
        self.model = ClaudeModel.OPUS_4  # Supervisor uses Opus for deep reasoning

        # Debate integration (UPGRADE-005)
        self.debate = debate_mechanism
        self.debate_threshold = debate_threshold
        self.min_debate_confidence = min_debate_confidence
        self.signal_conflict_threshold = signal_conflict_threshold
        self.debate_history: list[dict[str, Any]] = []

    def analyze(
        self,
        query: str,
        context: dict[str, Any],
    ) -> AgentResponse:
        """
        Make final trading decision based on team analyses.

        Args:
            query: Trading question (e.g., "Should we trade AAPL?")
            context: Team analyses and market data
                - analyst_reports: List of analyst responses
                - trader_recommendations: List of trader strategies
                - risk_checks: Risk manager assessments
                - market_data: Current market data
                - historical_trades: Past similar trades (optional)

        Returns:
            AgentResponse with final decision and reasoning
        """
        start_time = time.time()
        thoughts: list[AgentThought] = []

        try:
            # Build the full prompt with team analyses
            full_prompt = self._build_decision_prompt(query, context)

            # Call Claude Opus 4
            messages = [{"role": "user", "content": full_prompt}]

            response = self.llm_client.chat(
                model=self.model,
                messages=messages,
                system=self.system_prompt,
                max_tokens=self.prompt_version.max_tokens,
                temperature=self.prompt_version.temperature,
            )

            # Parse the JSON response
            try:
                decision = json.loads(response.content)
                final_answer = json.dumps(decision, indent=2)
                confidence = decision.get("confidence", 0.5)
                success = True
                error = None

            except json.JSONDecodeError:
                # Fallback if response isn't valid JSON
                final_answer = response.content
                confidence = 0.3
                success = False
                error = "Failed to parse JSON response"

            # Create thought for the decision
            thoughts.append(
                AgentThought(
                    thought_type=ThoughtType.FINAL_ANSWER,
                    content=final_answer,
                    metadata={
                        "raw_response": response.content,
                        "stop_reason": response.stop_reason,
                        "usage": response.usage,
                        "cost": self._estimate_cost(response.usage),
                    },
                )
            )

        except Exception as e:
            final_answer = f"Error: {e!s}"
            confidence = 0.0
            success = False
            error = str(e)

            thoughts.append(
                AgentThought(
                    thought_type=ThoughtType.FINAL_ANSWER,
                    content=final_answer,
                    metadata={"error": True, "exception": str(e)},
                )
            )

        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000

        # Record usage metrics
        self.registry.record_usage(
            role=AgentRole.SUPERVISOR,
            version=self.prompt_version.version,
            success=success,
            response_time_ms=execution_time_ms,
            confidence=confidence,
        )

        return AgentResponse(
            agent_name=self.name,
            agent_role=self.role,
            query=query,
            thoughts=thoughts,
            final_answer=final_answer,
            confidence=confidence,
            tools_used=[],
            execution_time_ms=execution_time_ms,
            success=success,
            error=error,
        )

    def _build_decision_prompt(
        self,
        query: str,
        context: dict[str, Any],
    ) -> str:
        """
        Build the full prompt with team analyses.

        Args:
            query: Trading question
            context: Team analyses and data

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"TRADING DECISION REQUEST: {query}",
            "",
            "=" * 60,
            "TEAM ANALYSES",
            "=" * 60,
        ]

        # Add analyst reports
        if "analyst_reports" in context:
            prompt_parts.append("\n--- ANALYST TEAM REPORTS ---")
            for report in context["analyst_reports"]:
                prompt_parts.append(
                    f"\n{report.get('agent_name', 'Unknown')}" f" (Confidence: {report.get('confidence', 0):.2f}):"
                )
                prompt_parts.append(report.get("analysis", "No analysis provided"))

        # Add trader recommendations
        if "trader_recommendations" in context:
            prompt_parts.append("\n--- TRADER TEAM RECOMMENDATIONS ---")
            for rec in context["trader_recommendations"]:
                prompt_parts.append(
                    f"\n{rec.get('agent_name', 'Unknown')}" f" (Confidence: {rec.get('confidence', 0):.2f}):"
                )
                prompt_parts.append(rec.get("strategy", "No strategy provided"))

        # Add risk checks
        if "risk_checks" in context:
            prompt_parts.append("\n--- RISK TEAM ASSESSMENTS ---")
            for check in context["risk_checks"]:
                prompt_parts.append(
                    f"\n{check.get('agent_name', 'Unknown')}" f" Decision: {check.get('decision', 'UNKNOWN')}:"
                )
                prompt_parts.append(check.get("reasoning", "No reasoning provided"))

        # Add market data
        if "market_data" in context:
            prompt_parts.append("\n--- CURRENT MARKET DATA ---")
            prompt_parts.append(json.dumps(context["market_data"], indent=2))

        # Add historical trades (for reflection)
        if "historical_trades" in context:
            prompt_parts.append("\n--- SIMILAR HISTORICAL TRADES ---")
            for trade in context["historical_trades"][:5]:  # Limit to 5 most recent
                prompt_parts.append(
                    f"\nDate: {trade.get('date')}, "
                    f"Decision: {trade.get('decision')}, "
                    f"Outcome: {trade.get('outcome')}, "
                    f"Return: {trade.get('return_pct', 0):+.2f}%"
                )

        prompt_parts.extend(
            [
                "",
                "=" * 60,
                "YOUR DECISION",
                "=" * 60,
                "",
                "Based on the above team analyses, market data, and historical performance,",
                "make your final trading decision. Provide your response in JSON format",
                "as specified in your role description.",
            ]
        )

        return "\n".join(prompt_parts)

    def _estimate_cost(self, usage: dict[str, int]) -> float:
        """Estimate cost of API call."""
        return self.llm_client.estimate_cost(
            model=self.model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
        )

    def should_debate(
        self,
        opportunity: dict[str, Any],
        context: dict[str, Any],
        initial_confidence: float = 1.0,
    ) -> tuple[bool, DebateTriggerReason | None]:
        """
        Determine if opportunity warrants Bull/Bear debate.

        Triggers debate when ANY of:
        - Position size > debate_threshold (default 10%)
        - Initial confidence < min_debate_confidence (default 70%)
        - Analyst signals conflict by > signal_conflict_threshold (default 30%)
        - High-impact event (earnings, major news)
        - Unusual market conditions (high volatility)
        - Manual request in context

        Args:
            opportunity: Trading opportunity data
            context: Analysis context including analyst signals
            initial_confidence: Initial confidence from preliminary analysis

        Returns:
            Tuple of (should_debate, reason)
        """
        if not self.debate:
            return False, None

        # Check for manual debate request
        if context.get("force_debate", False):
            return True, DebateTriggerReason.MANUAL_REQUEST

        # Check position size threshold
        position_size_pct = opportunity.get("position_size_pct", 0)
        if position_size_pct > self.debate_threshold:
            return True, DebateTriggerReason.HIGH_POSITION_SIZE

        # Check confidence threshold
        if initial_confidence < self.min_debate_confidence:
            return True, DebateTriggerReason.LOW_CONFIDENCE

        # Check for conflicting analyst signals
        analyst_signals = context.get("analyst_signals", {})
        if analyst_signals:
            signal_values = list(analyst_signals.values())
            if len(signal_values) >= 2:
                max_signal = max(signal_values)
                min_signal = min(signal_values)
                if max_signal - min_signal > self.signal_conflict_threshold:
                    return True, DebateTriggerReason.CONFLICTING_SIGNALS

        # Check for high-impact events
        if opportunity.get("has_earnings") or opportunity.get("has_major_news"):
            return True, DebateTriggerReason.HIGH_IMPACT_EVENT

        # Check for unusual market conditions
        if context.get("unusual_volatility") or context.get("market_stress"):
            return True, DebateTriggerReason.UNUSUAL_MARKET

        return False, None

    def analyze_with_debate(
        self,
        query: str,
        context: dict[str, Any],
        force_debate: bool = False,
    ) -> AgentResponse:
        """
        Analyze with optional Bull/Bear debate for major decisions.

        If debate criteria are met, runs a structured debate before
        making the final decision. The debate result is incorporated
        into the analysis.

        Args:
            query: Trading question
            context: Team analyses and market data
            force_debate: Force debate regardless of criteria

        Returns:
            AgentResponse with final decision (may include debate insights)
        """
        # Get initial analysis
        initial = self.analyze(query, context)

        # Extract opportunity from context
        opportunity = context.get("opportunity", {})

        # Determine if debate is needed
        if force_debate:
            context["force_debate"] = True

        should, reason = self.should_debate(
            opportunity,
            context,
            initial_confidence=initial.confidence,
        )

        if not should or not self.debate:
            return initial

        # Run the debate
        debate_result = self.debate.run_debate(
            opportunity,
            {"initial_analysis": initial.final_answer},
        )

        # Record debate in history
        self.debate_history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "trigger_reason": reason.value if reason else None,
                "debate_id": debate_result.debate_id,
                "outcome": debate_result.final_outcome.value,
                "consensus_confidence": debate_result.consensus_confidence,
            }
        )

        # Merge debate result with initial analysis
        return self._merge_debate_result(initial, debate_result, reason)

    def _merge_debate_result(
        self,
        initial: AgentResponse,
        debate_result: "DebateResult",
        trigger_reason: DebateTriggerReason | None,
    ) -> AgentResponse:
        """
        Merge debate result with initial analysis.

        Args:
            initial: Initial analysis response
            debate_result: Result from Bull/Bear debate
            trigger_reason: Why debate was triggered

        Returns:
            Updated AgentResponse incorporating debate insights
        """
        # Add debate thought
        debate_thought = AgentThought(
            thought_type=ThoughtType.REASONING,
            content=f"Conducted Bull/Bear debate ({len(debate_result.rounds)} rounds). "
            f"Outcome: {debate_result.final_outcome.value} "
            f"(consensus: {debate_result.consensus_confidence:.1%})",
            metadata={
                "debate_id": debate_result.debate_id,
                "trigger_reason": trigger_reason.value if trigger_reason else None,
                "rounds": len(debate_result.rounds),
                "key_points_bull": debate_result.key_points_bull[:3],
                "key_points_bear": debate_result.key_points_bear[:3],
                "risk_factors": debate_result.risk_factors[:3],
            },
        )

        # Create enhanced response
        enhanced_thoughts = list(initial.thoughts) + [debate_thought]

        # Adjust confidence based on debate consensus
        adjusted_confidence = (initial.confidence + debate_result.consensus_confidence) / 2

        # Build enhanced answer
        try:
            initial_decision = json.loads(initial.final_answer)
            initial_decision["debate"] = {
                "conducted": True,
                "outcome": debate_result.final_outcome.value,
                "consensus_confidence": debate_result.consensus_confidence,
                "key_bull_points": debate_result.key_points_bull[:3],
                "key_bear_points": debate_result.key_points_bear[:3],
                "risk_factors": debate_result.risk_factors[:3],
            }
            enhanced_answer = json.dumps(initial_decision, indent=2)
        except json.JSONDecodeError:
            enhanced_answer = (
                f"{initial.final_answer}\n\n"
                f"DEBATE RESULT:\n"
                f"Outcome: {debate_result.final_outcome.value}\n"
                f"Consensus: {debate_result.consensus_confidence:.1%}\n"
                f"Bull Points: {', '.join(debate_result.key_points_bull[:3])}\n"
                f"Bear Points: {', '.join(debate_result.key_points_bear[:3])}\n"
                f"Risks: {', '.join(debate_result.risk_factors[:3])}"
            )

        return AgentResponse(
            agent_name=initial.agent_name,
            agent_role=initial.agent_role,
            query=initial.query,
            thoughts=enhanced_thoughts,
            final_answer=enhanced_answer,
            confidence=adjusted_confidence,
            tools_used=initial.tools_used,
            execution_time_ms=initial.execution_time_ms,
            success=initial.success,
            error=initial.error,
        )

    def get_debate_history(self) -> list[dict[str, Any]]:
        """Get history of debates conducted by this supervisor."""
        return self.debate_history

    def clear_debate_history(self) -> None:
        """Clear debate history."""
        self.debate_history = []


def create_supervisor_agent(
    llm_client: AnthropicClient,
    version: str = "active",
    debate_mechanism: Optional["BullBearDebate"] = None,
    debate_threshold: float = 0.10,
    min_debate_confidence: float = 0.70,
) -> SupervisorAgent:
    """
    Factory function to create supervisor agent.

    Args:
        llm_client: Anthropic API client
        version: Prompt version ("active", "v1.0", "v1.1", "v2.0")
        debate_mechanism: Optional BullBearDebate for high-stakes decisions
        debate_threshold: Position size % that triggers debate (default 10%)
        min_debate_confidence: Confidence below this triggers debate (default 70%)

    Returns:
        SupervisorAgent instance
    """
    return SupervisorAgent(
        llm_client=llm_client,
        version=version,
        debate_mechanism=debate_mechanism,
        debate_threshold=debate_threshold,
        min_debate_confidence=min_debate_confidence,
    )


def create_supervisor_with_debate(
    llm_client: AnthropicClient,
    version: str = "active",
    debate_threshold: float = 0.10,
    min_debate_confidence: float = 0.70,
    max_debate_rounds: int = 3,
) -> SupervisorAgent:
    """
    Factory function to create supervisor agent with Bull/Bear debate mechanism.

    Creates a SupervisorAgent with an integrated BullBearDebate for
    high-stakes trading decisions. Automatically triggers debates when
    position size, confidence, or signal conflicts exceed thresholds.

    Args:
        llm_client: Anthropic API client
        version: Prompt version ("active", "v1.0", "v1.1", "v2.0")
        debate_threshold: Position size % that triggers debate (default 10%)
        min_debate_confidence: Confidence below this triggers debate (default 70%)
        max_debate_rounds: Maximum rounds in each debate (default 3)

    Returns:
        SupervisorAgent with debate mechanism

    Usage:
        supervisor = create_supervisor_with_debate(
            llm_client=client,
            debate_threshold=0.15,  # 15% position size triggers debate
            min_debate_confidence=0.65,  # <65% confidence triggers debate
        )

        # Use analyze_with_debate for automatic debate triggering
        response = supervisor.analyze_with_debate(query, context)
    """
    # Lazy import to avoid circular dependencies
    from llm.agents.debate_mechanism import create_debate_mechanism

    debate = create_debate_mechanism(max_rounds=max_debate_rounds)

    return SupervisorAgent(
        llm_client=llm_client,
        version=version,
        debate_mechanism=debate,
        debate_threshold=debate_threshold,
        min_debate_confidence=min_debate_confidence,
    )


def create_safe_supervisor_agent(
    llm_client: AnthropicClient,
    circuit_breaker: TradingCircuitBreaker,
    version: str = "active",
    risk_config: RiskTierConfig | None = None,
) -> SafeAgentWrapper:
    """
    Factory function to create supervisor agent with safety wrapper.

    Wraps the supervisor agent with circuit breaker integration and risk tier
    classification. All decisions pass through safety checks before execution.

    Args:
        llm_client: Anthropic API client
        circuit_breaker: Trading circuit breaker for risk controls
        version: Prompt version ("active", "v1.0", "v1.1", "v2.0")
        risk_config: Optional risk tier configuration

    Returns:
        SafeAgentWrapper wrapping a SupervisorAgent

    Usage:
        from models.circuit_breaker import create_circuit_breaker

        # Create circuit breaker with limits
        breaker = create_circuit_breaker(
            max_daily_loss=0.03,
            max_drawdown=0.10,
            max_consecutive_losses=5,
        )

        # Create safe supervisor
        safe_supervisor = create_safe_supervisor_agent(
            llm_client=client,
            circuit_breaker=breaker,
            version="active",
        )

        # Use the safe supervisor - all decisions pass through safety checks
        response = safe_supervisor.analyze(query, context)

        # Check if decision was blocked by safety
        if "HOLD" in response.final_answer:
            print("Decision blocked by safety controls")
    """
    # Create the base supervisor agent
    supervisor = SupervisorAgent(llm_client=llm_client, version=version)

    # Wrap with safety controls
    return wrap_agent_with_safety(
        agent=supervisor,
        circuit_breaker=circuit_breaker,
        risk_config=risk_config,
    )
