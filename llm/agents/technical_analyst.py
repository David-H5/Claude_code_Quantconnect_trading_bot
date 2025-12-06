"""
Technical Analyst Agent Implementation

Analyzes price action, technical indicators, chart patterns, and volume to identify
high-probability trading setups.

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


class TechnicalAnalyst(TradingAgent):
    """
    Technical Analyst agent - analyzes charts and technical indicators.

    Responsibilities:
    - Multi-timeframe trend analysis
    - Chart pattern recognition
    - Divergence detection
    - Specific trade setup recommendations
    - Support/resistance level identification

    Uses Claude Sonnet 4 for balanced performance and cost.
    """

    def __init__(
        self,
        llm_client: AnthropicClient,
        version: str = "active",
        max_iterations: int = 2,
        timeout_ms: float = 8000.0,
    ):
        """
        Initialize technical analyst agent.

        Args:
            llm_client: Anthropic API client
            version: Prompt version to use ("active", "v1.0", "v2.0")
            max_iterations: Max ReAct iterations
            timeout_ms: Max execution time
        """
        # Get prompt template
        prompt_version = get_prompt(AgentRole.TECHNICAL_ANALYST, version=version)
        if not prompt_version:
            raise PromptVersionError(
                agent_role="TECHNICAL_ANALYST",
                version=version,
            )

        super().__init__(
            name="TechnicalAnalyst",
            role=AgentRole.TECHNICAL_ANALYST,
            system_prompt=prompt_version.template,
            tools=[],  # No tool calling needed for technical analysis
            max_iterations=max_iterations,
            timeout_ms=timeout_ms,
            llm_client=llm_client,
        )

        self.prompt_version = prompt_version
        self.registry = get_registry()
        self.model = ClaudeModel.SONNET_4  # Balanced performance

    def analyze(
        self,
        query: str,
        context: dict[str, Any],
    ) -> AgentResponse:
        """
        Analyze technical indicators and chart patterns.

        Args:
            query: Analysis question (e.g., "Analyze AAPL technical setup")
            context: Market data and indicators
                - symbol: Ticker symbol
                - current_price: Current price
                - indicators: Dict of technical indicators
                    - rsi: RSI value
                    - macd: MACD values {value, signal, histogram}
                    - vwap: VWAP value
                    - bollinger: Bollinger band values
                    - volume: Current volume
                    - avg_volume: Average volume
                - timeframe_data: Multi-timeframe trends (optional)
                    - weekly_trend: Trend on weekly chart
                    - daily_trend: Trend on daily chart
                    - intraday_trend: Trend on intraday chart
                - support_resistance: Support/resistance levels (optional)

        Returns:
            AgentResponse with technical analysis and trade setup
        """
        start_time = time.time()
        thoughts: list[AgentThought] = []

        try:
            # Build the analysis prompt with market data
            full_prompt = self._build_analysis_prompt(query, context)

            # Call Claude Sonnet 4
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
                analysis = json.loads(response.content)
                final_answer = json.dumps(analysis, indent=2)
                confidence = analysis.get("confidence", 0.5)
                success = True
                error = None

            except json.JSONDecodeError:
                # Fallback if response isn't valid JSON
                final_answer = response.content
                confidence = 0.3
                success = False
                error = "Failed to parse JSON response"

            # Create thought for the analysis
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
            role=AgentRole.TECHNICAL_ANALYST,
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

    def _build_analysis_prompt(
        self,
        query: str,
        context: dict[str, Any],
    ) -> str:
        """
        Build the technical analysis prompt with market data.

        Args:
            query: Analysis question
            context: Market data and indicators

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"TECHNICAL ANALYSIS REQUEST: {query}",
            "",
            "=" * 60,
            "MARKET DATA",
            "=" * 60,
        ]

        # Symbol and price
        if "symbol" in context:
            prompt_parts.append(f"\nSymbol: {context['symbol']}")
        if "current_price" in context:
            prompt_parts.append(f"Current Price: ${context['current_price']:.2f}")

        # Technical indicators
        if "indicators" in context:
            prompt_parts.append("\n--- TECHNICAL INDICATORS ---")
            indicators = context["indicators"]

            if "rsi" in indicators:
                prompt_parts.append(f"RSI: {indicators['rsi']:.1f}")

            if "macd" in indicators:
                macd = indicators["macd"]
                prompt_parts.append(
                    f"MACD: Value={macd.get('value', 0):.2f}, Signal={macd.get('signal', 0):.2f}, Histogram={macd.get('histogram', 0):.2f}"
                )

            if "vwap" in indicators:
                prompt_parts.append(f"VWAP: ${indicators['vwap']:.2f}")
                if "current_price" in context:
                    position = "above" if context["current_price"] > indicators["vwap"] else "below"
                    prompt_parts.append(f"Price position: {position} VWAP")

            if "bollinger" in indicators:
                bb = indicators["bollinger"]
                prompt_parts.append(
                    f"Bollinger Bands: Upper=${bb.get('upper', 0):.2f}, Middle=${bb.get('middle', 0):.2f}, Lower=${bb.get('lower', 0):.2f}"
                )

            if "volume" in indicators and "avg_volume" in indicators:
                vol_ratio = indicators["volume"] / indicators["avg_volume"]
                prompt_parts.append(
                    f"Volume: {indicators['volume']:,.0f} (Avg: {indicators['avg_volume']:,.0f}, Ratio: {vol_ratio:.2f}x)"
                )

        # Multi-timeframe data
        if "timeframe_data" in context:
            prompt_parts.append("\n--- MULTI-TIMEFRAME ANALYSIS ---")
            tf = context["timeframe_data"]
            if "weekly_trend" in tf:
                prompt_parts.append(f"Weekly Trend: {tf['weekly_trend']}")
            if "daily_trend" in tf:
                prompt_parts.append(f"Daily Trend: {tf['daily_trend']}")
            if "intraday_trend" in tf:
                prompt_parts.append(f"Intraday Trend: {tf['intraday_trend']}")

        # Support/Resistance levels
        if "support_resistance" in context:
            sr = context["support_resistance"]
            if "support_levels" in sr:
                prompt_parts.append(f"\nSupport Levels: {', '.join([f'${x:.2f}' for x in sr['support_levels']])}")
            if "resistance_levels" in sr:
                prompt_parts.append(f"Resistance Levels: {', '.join([f'${x:.2f}' for x in sr['resistance_levels']])}")

        prompt_parts.extend(
            [
                "",
                "=" * 60,
                "YOUR ANALYSIS",
                "=" * 60,
                "",
                "Based on the above market data and technical indicators,",
                "provide your technical analysis in JSON format as specified",
                "in your role description. Include specific entry, stop loss,",
                "and profit target levels.",
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


def create_technical_analyst(
    llm_client: AnthropicClient,
    version: str = "active",
) -> TechnicalAnalyst:
    """
    Factory function to create technical analyst agent.

    Args:
        llm_client: Anthropic API client
        version: Prompt version ("active", "v1.0", "v2.0")

    Returns:
        TechnicalAnalyst instance
    """
    return TechnicalAnalyst(llm_client=llm_client, version=version)


def create_safe_technical_analyst(
    llm_client: AnthropicClient,
    circuit_breaker: TradingCircuitBreaker,
    version: str = "active",
    risk_config: RiskTierConfig | None = None,
) -> SafeAgentWrapper:
    """
    Factory function to create technical analyst agent with safety wrapper.

    Wraps the technical analyst agent with circuit breaker integration and
    risk tier classification. All decisions pass through safety checks before
    execution.

    Args:
        llm_client: Anthropic API client
        circuit_breaker: Trading circuit breaker for risk controls
        version: Prompt version ("active", "v1.0", "v2.0")
        risk_config: Optional risk tier configuration

    Returns:
        SafeAgentWrapper wrapping a TechnicalAnalyst

    Usage:
        from models.circuit_breaker import create_circuit_breaker

        breaker = create_circuit_breaker(
            max_daily_loss=0.03,
            max_drawdown=0.10,
            max_consecutive_losses=5,
        )

        safe_analyst = create_safe_technical_analyst(
            llm_client=client,
            circuit_breaker=breaker,
            version="active",
        )

        response = safe_analyst.analyze(query, context)
    """
    analyst = TechnicalAnalyst(llm_client=llm_client, version=version)
    return wrap_agent_with_safety(
        agent=analyst,
        circuit_breaker=circuit_breaker,
        risk_config=risk_config,
    )
