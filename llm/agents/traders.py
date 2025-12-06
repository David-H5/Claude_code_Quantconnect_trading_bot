"""
Trader Agent Implementations

Conservative, Moderate, and Aggressive traders with different risk tolerances.

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


class ConservativeTrader(TradingAgent):
    """
    Conservative Trader agent - capital preservation over growth.

    Philosophy:
    - High win rate (>65%) over high profit per trade
    - Defined risk strategies only (iron condors, credit spreads)
    - Max 15% position size, 5% risk per trade
    - Exit at 50% profit

    Uses Claude Opus 4 for complex strategy decisions.
    """

    def __init__(
        self,
        llm_client: AnthropicClient,
        version: str = "active",
        max_iterations: int = 2,
        timeout_ms: float = 8000.0,
    ):
        """
        Initialize conservative trader agent.

        Args:
            llm_client: Anthropic API client
            version: Prompt version to use
            max_iterations: Max ReAct iterations
            timeout_ms: Max execution time
        """
        prompt_version = get_prompt(AgentRole.CONSERVATIVE_TRADER, version=version)
        if not prompt_version:
            raise PromptVersionError(
                agent_role="CONSERVATIVE_TRADER",
                version=version,
            )

        super().__init__(
            name="ConservativeTrader",
            role=AgentRole.CONSERVATIVE_TRADER,
            system_prompt=prompt_version.template,
            tools=[],
            max_iterations=max_iterations,
            timeout_ms=timeout_ms,
            llm_client=llm_client,
        )

        self.prompt_version = prompt_version
        self.registry = get_registry()
        self.model = ClaudeModel.OPUS_4  # Traders use Opus for strategy decisions

    def analyze(
        self,
        query: str,
        context: dict[str, Any],
    ) -> AgentResponse:
        """
        Design conservative options strategy.

        Args:
            query: Strategy question
            context: Market analysis and data
                - symbol: Ticker symbol
                - current_price: Current price
                - technical_analysis: TechnicalAnalyst output
                - sentiment_analysis: SentimentAnalyst output
                - market_outlook: Expected direction (bullish/bearish/neutral)
                - iv_percentile: Implied volatility percentile
                - option_chain: Available options (optional)

        Returns:
            AgentResponse with strategy recommendation
        """
        start_time = time.time()
        thoughts: list[AgentThought] = []

        try:
            full_prompt = self._build_strategy_prompt(query, context)

            messages = [{"role": "user", "content": full_prompt}]

            response = self.llm_client.chat(
                model=self.model,
                messages=messages,
                system=self.system_prompt,
                max_tokens=self.prompt_version.max_tokens,
                temperature=self.prompt_version.temperature,
            )

            try:
                strategy = json.loads(response.content)
                final_answer = json.dumps(strategy, indent=2)
                confidence = strategy.get("confidence", 0.5)
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

        self.registry.record_usage(
            role=AgentRole.CONSERVATIVE_TRADER,
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

    def _build_strategy_prompt(
        self,
        query: str,
        context: dict[str, Any],
    ) -> str:
        """Build strategy prompt with market analysis."""
        prompt_parts = [
            f"STRATEGY DESIGN REQUEST: {query}",
            "",
            "=" * 60,
            "MARKET ANALYSIS & DATA",
            "=" * 60,
        ]

        if "symbol" in context:
            prompt_parts.append(f"\nSymbol: {context['symbol']}")
        if "current_price" in context:
            prompt_parts.append(f"Current Price: ${context['current_price']:.2f}")

        if "technical_analysis" in context:
            prompt_parts.append("\n--- TECHNICAL ANALYSIS ---")
            prompt_parts.append(json.dumps(context["technical_analysis"], indent=2))

        if "sentiment_analysis" in context:
            prompt_parts.append("\n--- SENTIMENT ANALYSIS ---")
            prompt_parts.append(json.dumps(context["sentiment_analysis"], indent=2))

        if "market_outlook" in context:
            prompt_parts.append(f"\nMarket Outlook: {context['market_outlook']}")

        if "iv_percentile" in context:
            prompt_parts.append(f"IV Percentile: {context['iv_percentile']:.0f}%")

        prompt_parts.extend(
            [
                "",
                "=" * 60,
                "YOUR STRATEGY",
                "=" * 60,
                "",
                "Based on the above analysis, design a conservative options strategy",
                "in JSON format as specified in your role description.",
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


def create_conservative_trader(
    llm_client: AnthropicClient,
    version: str = "active",
) -> ConservativeTrader:
    """Factory function to create conservative trader agent."""
    return ConservativeTrader(llm_client=llm_client, version=version)


def create_safe_conservative_trader(
    llm_client: AnthropicClient,
    circuit_breaker: TradingCircuitBreaker,
    version: str = "active",
    risk_config: RiskTierConfig | None = None,
) -> SafeAgentWrapper:
    """
    Factory function to create conservative trader agent with safety wrapper.

    Wraps the conservative trader agent with circuit breaker integration and
    risk tier classification. All decisions pass through safety checks before
    execution.

    Args:
        llm_client: Anthropic API client
        circuit_breaker: Trading circuit breaker for risk controls
        version: Prompt version ("active", "v1.0")
        risk_config: Optional risk tier configuration

    Returns:
        SafeAgentWrapper wrapping a ConservativeTrader

    Usage:
        from models.circuit_breaker import create_circuit_breaker

        breaker = create_circuit_breaker(
            max_daily_loss=0.03,
            max_drawdown=0.10,
            max_consecutive_losses=5,
        )

        safe_trader = create_safe_conservative_trader(
            llm_client=client,
            circuit_breaker=breaker,
            version="active",
        )

        response = safe_trader.analyze(query, context)
    """
    trader = ConservativeTrader(llm_client=llm_client, version=version)
    return wrap_agent_with_safety(
        agent=trader,
        circuit_breaker=circuit_breaker,
        risk_config=risk_config,
    )
