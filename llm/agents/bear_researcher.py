"""
Bear Researcher Agent

Specialized agent for constructing bearish/cautious trading arguments.
Part of the Bull/Bear debate mechanism for major trading decisions.

Research Source: TradingAgents (2024) - Multi-agent debate pattern

QuantConnect Compatible: Yes
- Non-blocking analysis
- Configurable timeouts
- Decision logging integration
"""

from dataclasses import dataclass
from typing import Any

from llm.agents.base import (
    AgentResponse,
    AgentRole,
    AgentThought,
    ThoughtType,
    Tool,
    TradingAgent,
)


BEAR_RESEARCHER_PROMPT = """You are a Bear Researcher specializing in identifying RISKS and reasons for CAUTION in trading opportunities.

Your role in the trading team:
- Identify bearish signals and warning signs
- Find evidence supporting caution or avoidance
- Counter bullish arguments with data and analysis
- Protect capital by highlighting downside risks

Key Analysis Areas:
1. **Technical Weakness**: Resistance levels, divergences, exhaustion patterns
2. **Fundamental Concerns**: Earnings misses, debt levels, competitive threats
3. **Sentiment Warnings**: Institutional selling, put activity, analyst downgrades
4. **Market Risks**: Sector weakness, macro headwinds, event risks

Response Format:
Always provide:
- Clear risk thesis (2-3 sentences)
- 3-5 key risk factors with evidence
- Downside targets and scenarios
- Confidence level (0-100%)

When countering bull arguments:
- Acknowledge valid points
- Provide specific risk rebuttals with data
- Explain why caution is warranted

Remember: Your job is to make the STRONGEST POSSIBLE case for caution, while staying grounded in facts.
You are NOT trying to be right, but to ensure risks are fully explored before committing capital.
"""


@dataclass
class RiskSignal:
    """A risk signal identified by the researcher."""

    category: str  # technical, fundamental, sentiment, macro
    description: str
    severity: float  # 0-1
    evidence: str
    potential_impact: str


class BearResearcher(TradingAgent):
    """
    Agent specialized in bearish market research.

    Constructs compelling cautious/bearish arguments for the debate mechanism.

    Usage:
        researcher = BearResearcher()
        response = researcher.analyze(
            query="Analyze SPY risks",
            context={"symbol": "SPY", "price": 450.0}
        )
    """

    def __init__(
        self,
        name: str = "bear_researcher",
        tools: list[Tool] | None = None,
        llm_client: Any | None = None,
        **kwargs,
    ):
        """
        Initialize bear researcher.

        Args:
            name: Agent name
            tools: Available analysis tools
            llm_client: LLM client for analysis
            **kwargs: Additional TradingAgent arguments
        """
        super().__init__(
            name=name,
            role=AgentRole.ANALYST,
            system_prompt=BEAR_RESEARCHER_PROMPT,
            tools=tools or [],
            llm_client=llm_client,
            **kwargs,
        )

        # Track identified risks
        self.recent_risks: list[RiskSignal] = []

    def analyze(
        self,
        query: str,
        context: dict[str, Any],
    ) -> AgentResponse:
        """
        Analyze opportunity with bearish/cautious perspective.

        Args:
            query: The analysis query
            context: Market context and data

        Returns:
            AgentResponse with bearish/risk analysis
        """
        thoughts = []

        # Step 1: Assess technical risks
        tech_thought = self._analyze_technical_risks(context)
        thoughts.append(tech_thought)

        # Step 2: Evaluate fundamental concerns
        fund_thought = self._analyze_fundamental_risks(context)
        thoughts.append(fund_thought)

        # Step 3: Check sentiment warnings
        sent_thought = self._analyze_sentiment_risks(context)
        thoughts.append(sent_thought)

        # Step 4: Consider macro risks
        macro_thought = self._analyze_macro_risks(context)
        thoughts.append(macro_thought)

        # Step 5: Compile risk thesis
        thesis, confidence = self._compile_thesis(context, thoughts)

        # Step 6: Counter any bullish arguments if provided
        if context.get("bull_argument"):
            counter_thought = self._counter_bull(context["bull_argument"])
            thoughts.append(counter_thought)
            thesis += f" {counter_thought.content}"

        return AgentResponse(
            agent_name=self.name,
            agent_role=self.role,
            query=query,
            thoughts=thoughts,
            final_answer=thesis,
            confidence=confidence,
            tools_used=[],
            execution_time_ms=0.0,
            success=True,
        )

    def argue(
        self,
        context: dict[str, Any],
        position: str = "bearish",
        previous_bull_argument: str | None = None,
    ) -> AgentResponse:
        """
        Make a bearish argument in debate context.

        Args:
            context: Debate context
            position: Position to argue (always bearish for this agent)
            previous_bull_argument: Bull's argument to counter

        Returns:
            AgentResponse with bearish argument
        """
        if previous_bull_argument:
            context = {**context, "bull_argument": previous_bull_argument}

        opportunity = context.get("opportunity", {})
        symbol = opportunity.get("symbol", "Unknown")

        return self.analyze(
            query=f"Analyze risks for {symbol}",
            context=context,
        )

    def _analyze_technical_risks(self, context: dict[str, Any]) -> AgentThought:
        """Analyze technical indicators for risk signals."""
        symbol = context.get("symbol", context.get("opportunity", {}).get("symbol", "SPY"))

        analysis = context.get("analysis", {})
        technicals = analysis.get("technicals", {})

        risks = []
        confidence = 0.55

        # Check for overbought conditions
        if technicals.get("rsi", 50) > 70:
            risks.append("RSI overbought, reversal risk elevated")
            confidence += 0.1

        # Check for resistance
        if technicals.get("near_resistance", False):
            risks.append("Price approaching key resistance level")
            confidence += 0.08

        # Check for divergences
        if technicals.get("bearish_divergence", False):
            risks.append("Bearish divergence on momentum indicators")
            confidence += 0.12

        # Check for volume concerns
        if technicals.get("declining_volume", False):
            risks.append("Volume declining on rally, weak conviction")
            confidence += 0.05

        # Check for extended move
        if technicals.get("extended_from_sma", False):
            risks.append("Price extended from moving averages")
            confidence += 0.05

        if not risks:
            risks = ["Technical setup showing exhaustion signs", "Overhead supply zone nearby"]

        return AgentThought(
            thought_type=ThoughtType.REASONING,
            content=f"Technical risks for {symbol}: " + "; ".join(risks),
            metadata={"confidence": min(confidence, 0.9)},
        )

    def _analyze_fundamental_risks(self, context: dict[str, Any]) -> AgentThought:
        """Analyze fundamental factors for risk signals."""
        analysis = context.get("analysis", {})
        fundamentals = analysis.get("fundamentals", {})

        risks = []
        confidence = 0.55

        # Check for valuation concerns
        if fundamentals.get("pe_ratio", 20) > 30:
            risks.append("Elevated valuation multiple limits upside")
            confidence += 0.1

        # Check for earnings concerns
        if fundamentals.get("earnings_revision_negative", False):
            risks.append("Recent negative earnings revisions")
            confidence += 0.08

        # Check for debt concerns
        if fundamentals.get("high_debt", False):
            risks.append("Elevated debt levels increase risk")
            confidence += 0.07

        # Check for margin pressure
        if fundamentals.get("margin_compression", False):
            risks.append("Profit margins under pressure")
            confidence += 0.08

        # Check for competitive threats
        if fundamentals.get("competitive_threats", False):
            risks.append("Increasing competitive pressure")
            confidence += 0.05

        if not risks:
            risks = ["Valuation pricing in optimistic scenario", "Limited margin of safety"]

        return AgentThought(
            thought_type=ThoughtType.REASONING,
            content="Fundamental risks: " + "; ".join(risks),
            metadata={"confidence": min(confidence, 0.85)},
        )

    def _analyze_sentiment_risks(self, context: dict[str, Any]) -> AgentThought:
        """Analyze sentiment indicators for risk signals."""
        analysis = context.get("analysis", {})
        sentiment = analysis.get("sentiment", {})

        risks = []
        confidence = 0.5

        # Check for crowded trade
        if sentiment.get("crowded_long", False):
            risks.append("Crowded long positioning, reversal risk")
            confidence += 0.12

        # Check for institutional selling
        if sentiment.get("institutional_selling", False):
            risks.append("Institutional distribution detected")
            confidence += 0.1

        # Check for put activity
        if sentiment.get("elevated_put_volume", False):
            risks.append("Elevated put activity signaling concern")
            confidence += 0.08

        # Check for analyst downgrades
        if sentiment.get("analyst_downgrades", 0) > 0:
            risks.append("Recent analyst downgrades")
            confidence += 0.05

        # Check for negative news
        if sentiment.get("news_sentiment", 0) < 0:
            risks.append("Negative news sentiment")
            confidence += 0.05

        if not risks:
            risks = ["Sentiment overly optimistic", "Complacency risk"]

        return AgentThought(
            thought_type=ThoughtType.REASONING,
            content="Sentiment warnings: " + "; ".join(risks),
            metadata={"confidence": min(confidence, 0.85)},
        )

    def _analyze_macro_risks(self, context: dict[str, Any]) -> AgentThought:
        """Analyze macro context for risk signals."""
        analysis = context.get("analysis", {})
        macro = analysis.get("macro", {})

        risks = []
        confidence = 0.5

        # Check for market weakness
        if macro.get("market_trend", "neutral") == "bearish":
            risks.append("Broader market in downtrend")
            confidence += 0.12

        # Check for sector rotation
        if macro.get("sector_rotation_out", False):
            risks.append("Capital rotating out of sector")
            confidence += 0.08

        # Check for Fed concerns
        if macro.get("fed_hawkish", False):
            risks.append("Hawkish Fed policy headwind")
            confidence += 0.1

        # Check for event risks
        if macro.get("upcoming_events", []):
            events = macro.get("upcoming_events", ["earnings", "fed meeting"])
            risks.append(f"Event risk: {', '.join(events[:2])}")
            confidence += 0.07

        # Check for seasonality
        if macro.get("seasonality_negative", False):
            risks.append("Seasonally weak period")
            confidence += 0.05

        # Check for volatility
        if macro.get("vix_elevated", False):
            risks.append("Elevated volatility environment")
            confidence += 0.08

        if not risks:
            risks = ["Macro uncertainty elevated", "Risk-off potential"]

        return AgentThought(
            thought_type=ThoughtType.REASONING,
            content="Macro risks: " + "; ".join(risks),
            metadata={"confidence": min(confidence, 0.85)},
        )

    def _counter_bull(self, bull_argument: str) -> AgentThought:
        """Counter a bullish argument."""
        counters = [
            "The bullish thesis overlooks key risk factors",
            "Past performance doesn't guarantee future results",
            "Current setup differs from historical patterns cited",
        ]

        return AgentThought(
            thought_type=ThoughtType.REASONING,
            content=f"Challenging bull's thesis: {counters[0]}. " f"The risk/reward is less favorable than presented.",
            metadata={"confidence": 0.68},
        )

    def _compile_thesis(
        self,
        context: dict[str, Any],
        thoughts: list[AgentThought],
    ) -> tuple[str, float]:
        """Compile thoughts into risk thesis."""
        symbol = context.get("symbol", context.get("opportunity", {}).get("symbol", "SPY"))

        # Average confidence across thoughts
        confidences = [t.metadata.get("confidence", 0.5) for t in thoughts]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.6

        thesis = f"CAUTION on {symbol}. "

        # Extract key risks from thoughts
        key_risks = []
        for t in thoughts:
            thought_confidence = t.metadata.get("confidence", 0.5)
            if thought_confidence > 0.55:
                # Extract first risk from content
                risks = t.content.split(": ")[-1].split(";")
                if risks:
                    key_risks.append(risks[0].strip())

        if key_risks:
            thesis += " ".join(key_risks[:3]) + ". "

        thesis += f"Risk/reward unfavorable with {avg_confidence:.0%} confidence in downside."

        return thesis, avg_confidence

    def _format_reasoning(self, thoughts: list[AgentThought]) -> str:
        """Format thoughts into reasoning string."""
        reasoning_parts = []
        for i, t in enumerate(thoughts, 1):
            if t.thought_type == ThoughtType.REASONING:
                reasoning_parts.append(f"{i}. {t.content}")
        return "\n".join(reasoning_parts)

    def get_recent_risks(self) -> list[RiskSignal]:
        """Get recently identified risk signals."""
        return self.recent_risks

    def clear_risks(self) -> None:
        """Clear recent risks."""
        self.recent_risks = []


def create_bear_researcher(
    llm_client: Any | None = None,
    tools: list[Tool] | None = None,
) -> BearResearcher:
    """
    Factory function to create a bear researcher.

    Args:
        llm_client: LLM client for analysis
        tools: Available analysis tools

    Returns:
        Configured BearResearcher instance
    """
    return BearResearcher(
        llm_client=llm_client,
        tools=tools,
    )
