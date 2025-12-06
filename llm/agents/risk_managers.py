"""
Risk Manager Agent Implementations

Position-level and portfolio-level risk managers.

QuantConnect Compatible: Yes
"""

import json
import time
from typing import Any

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


class PositionRiskManager(TradingAgent):
    """
    Position Risk Manager agent - approves or rejects individual trades.

    Responsibilities:
    - Enforce position size limits (25% max)
    - Enforce risk per trade limits (5% max)
    - Enforce position count limits (10 max)
    - Enforce minimum win probability (40%)
    - Check option liquidity (bid-ask spread <15%)
    - CANNOT be overridden by Supervisor

    Uses Claude Haiku for fast, consistent decisions.
    """

    def __init__(
        self,
        llm_client: AnthropicClient,
        version: str = "active",
        max_iterations: int = 1,
        timeout_ms: float = 5000.0,
    ):
        """
        Initialize position risk manager agent.

        Args:
            llm_client: Anthropic API client
            version: Prompt version to use
            max_iterations: Max ReAct iterations
            timeout_ms: Max execution time
        """
        prompt_version = get_prompt(AgentRole.POSITION_RISK_MANAGER, version=version)
        if not prompt_version:
            raise PromptVersionError(
                agent_role="POSITION_RISK_MANAGER",
                version=version,
            )

        super().__init__(
            name="PositionRiskManager",
            role=AgentRole.POSITION_RISK_MANAGER,
            system_prompt=prompt_version.template,
            tools=[],
            max_iterations=max_iterations,
            timeout_ms=timeout_ms,
            llm_client=llm_client,
        )

        self.prompt_version = prompt_version
        self.registry = get_registry()
        self.model = ClaudeModel.HAIKU  # Fast, low-cost risk checks

    def analyze(
        self,
        query: str,
        context: dict[str, Any],
    ) -> AgentResponse:
        """
        Approve or reject a proposed trade.

        Args:
            query: Risk check question
            context: Trade proposal and portfolio state
                - symbol: Ticker symbol
                - strategy: Proposed strategy details
                    - position_size: Requested position size (0.0-1.0)
                    - max_risk: Maximum risk amount
                    - win_probability: Estimated win probability
                - portfolio_state: Current portfolio
                    - current_positions: Number of current positions
                    - portfolio_value: Total portfolio value
                - option_data: Option chain data (for liquidity check)
                    - bid_ask_spread_pct: Bid-ask spread percentage

        Returns:
            AgentResponse with APPROVE or REJECT decision
        """
        start_time = time.time()
        thoughts: list[AgentThought] = []

        try:
            full_prompt = self._build_risk_check_prompt(query, context)

            messages = [{"role": "user", "content": full_prompt}]

            response = self.llm_client.chat(
                model=self.model,
                messages=messages,
                system=self.system_prompt,
                max_tokens=self.prompt_version.max_tokens,
                temperature=self.prompt_version.temperature,
            )

            try:
                decision = json.loads(response.content)
                final_answer = json.dumps(decision, indent=2)

                # Extract decision and confidence
                decision_str = decision.get("decision", "REJECT")
                confidence = decision.get("confidence", 1.0)
                success = True
                error = None

            except json.JSONDecodeError:
                final_answer = response.content
                confidence = 0.3
                success = False
                error = "Failed to parse JSON response"

            thoughts.append(
                AgentThought(
                    thought_type=ThoughtType.FINAL_ANSWER,
                    content=final_answer,
                    metadata={
                        "raw_response": response.content,
                        "usage": response.usage,
                        "cost": self._estimate_cost(response.usage),
                    },
                )
            )

        except Exception as e:
            final_answer = json.dumps(
                {
                    "decision": "REJECT",
                    "reasoning": f"Error during risk check: {e!s}",
                    "confidence": 1.0,
                },
                indent=2,
            )
            confidence = 1.0
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

        self.registry.record_usage(
            role=AgentRole.POSITION_RISK_MANAGER,
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

    def _build_risk_check_prompt(
        self,
        query: str,
        context: dict[str, Any],
    ) -> str:
        """Build risk check prompt with trade proposal."""
        prompt_parts = [
            f"RISK CHECK REQUEST: {query}",
            "",
            "=" * 60,
            "TRADE PROPOSAL",
            "=" * 60,
        ]

        if "symbol" in context:
            prompt_parts.append(f"\nSymbol: {context['symbol']}")

        if "strategy" in context:
            strategy = context["strategy"]
            prompt_parts.append("\n--- PROPOSED STRATEGY ---")
            prompt_parts.append(f"Position Size: {strategy.get('position_size', 0)*100:.1f}%")
            prompt_parts.append(f"Max Risk: ${strategy.get('max_risk', 0):.2f}")
            prompt_parts.append(f"Win Probability: {strategy.get('win_probability', 0)*100:.1f}%")
            if "strategy_type" in strategy:
                prompt_parts.append(f"Strategy Type: {strategy['strategy_type']}")

        if "portfolio_state" in context:
            portfolio = context["portfolio_state"]
            prompt_parts.append("\n--- CURRENT PORTFOLIO STATE ---")
            prompt_parts.append(f"Portfolio Value: ${portfolio.get('portfolio_value', 0):,.2f}")
            prompt_parts.append(f"Current Positions: {portfolio.get('current_positions', 0)}/10")

        if "option_data" in context:
            options = context["option_data"]
            prompt_parts.append("\n--- OPTION LIQUIDITY ---")
            prompt_parts.append(f"Bid-Ask Spread: {options.get('bid_ask_spread_pct', 0):.1f}%")

        prompt_parts.extend(
            [
                "",
                "=" * 60,
                "YOUR DECISION",
                "=" * 60,
                "",
                "Based on the above trade proposal and portfolio state,",
                "decide whether to APPROVE or REJECT this trade in JSON format",
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


def create_position_risk_manager(
    llm_client: AnthropicClient,
    version: str = "active",
) -> PositionRiskManager:
    """Factory function to create position risk manager agent."""
    return PositionRiskManager(llm_client=llm_client, version=version)


def create_safe_position_risk_manager(
    llm_client: AnthropicClient,
    circuit_breaker: TradingCircuitBreaker,
    version: str = "active",
    risk_config: RiskTierConfig | None = None,
) -> SafeAgentWrapper:
    """
    Factory function to create position risk manager agent with safety wrapper.

    Wraps the position risk manager agent with circuit breaker integration and
    risk tier classification. All decisions pass through safety checks before
    execution.

    Note: This creates a double-layer of safety - the PositionRiskManager already
    performs risk checks, and the SafeAgentWrapper adds circuit breaker protection.

    Args:
        llm_client: Anthropic API client
        circuit_breaker: Trading circuit breaker for risk controls
        version: Prompt version ("active", "v1.0")
        risk_config: Optional risk tier configuration

    Returns:
        SafeAgentWrapper wrapping a PositionRiskManager

    Usage:
        from models.circuit_breaker import create_circuit_breaker

        breaker = create_circuit_breaker(
            max_daily_loss=0.03,
            max_drawdown=0.10,
            max_consecutive_losses=5,
        )

        safe_risk_manager = create_safe_position_risk_manager(
            llm_client=client,
            circuit_breaker=breaker,
            version="active",
        )

        response = safe_risk_manager.analyze(query, context)
    """
    risk_manager = PositionRiskManager(llm_client=llm_client, version=version)
    return wrap_agent_with_safety(
        agent=risk_manager,
        circuit_breaker=circuit_breaker,
        risk_config=risk_config,
    )
