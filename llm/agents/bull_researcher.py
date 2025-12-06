"""
Bull Researcher Agent

Specialized agent for constructing bullish trading arguments.
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


BULL_RESEARCHER_PROMPT = """You are a Bull Researcher specializing in finding reasons to be OPTIMISTIC about trading opportunities.

Your role in the trading team:
- Identify bullish signals and positive catalysts
- Find supporting evidence for upside potential
- Counter bearish arguments with data and analysis
- Provide conviction scores based on evidence strength

Key Analysis Areas:
1. **Technical Strength**: Price action, momentum, support levels
2. **Fundamental Catalysts**: Earnings growth, revenue trends, market share
3. **Sentiment Indicators**: Institutional buying, options flow, analyst upgrades
4. **Market Context**: Sector strength, macro tailwinds, seasonal patterns

Response Format:
Always provide:
- Clear bullish thesis (2-3 sentences)
- 3-5 key supporting points with evidence
- Upside price targets with timeframes
- Confidence level (0-100%)

When countering bear arguments:
- Acknowledge valid concerns
- Provide specific rebuttals with data
- Explain why positives outweigh negatives

Remember: Your job is to make the STRONGEST POSSIBLE bullish case, while staying grounded in facts.
You are NOT trying to be right, but to ensure bullish perspectives are fully explored.
"""


@dataclass
class BullishSignal:
    """A bullish signal identified by the researcher."""

    category: str  # technical, fundamental, sentiment, macro
    description: str
    strength: float  # 0-1
    evidence: str
    timeframe: str  # short, medium, long


class BullResearcher(TradingAgent):
    """
    Agent specialized in bullish market research.

    Constructs compelling bullish arguments for the debate mechanism.

    Usage:
        researcher = BullResearcher()
        response = researcher.analyze(
            query="Analyze SPY for bullish entry",
            context={"symbol": "SPY", "price": 450.0}
        )
    """

    def __init__(
        self,
        name: str = "bull_researcher",
        tools: list[Tool] | None = None,
        llm_client: Any | None = None,
        **kwargs,
    ):
        """
        Initialize bull researcher.

        Args:
            name: Agent name
            tools: Available analysis tools
            llm_client: LLM client for analysis
            **kwargs: Additional TradingAgent arguments
        """
        super().__init__(
            name=name,
            role=AgentRole.ANALYST,
            system_prompt=BULL_RESEARCHER_PROMPT,
            tools=tools or [],
            llm_client=llm_client,
            **kwargs,
        )

        # Track identified signals
        self.recent_signals: list[BullishSignal] = []

    def analyze(
        self,
        query: str,
        context: dict[str, Any],
    ) -> AgentResponse:
        """
        Analyze opportunity with bullish perspective.

        Args:
            query: The analysis query
            context: Market context and data

        Returns:
            AgentResponse with bullish analysis
        """
        thoughts = []

        # Step 1: Assess technical indicators
        tech_thought = self._analyze_technicals(context)
        thoughts.append(tech_thought)

        # Step 2: Evaluate fundamentals
        fund_thought = self._analyze_fundamentals(context)
        thoughts.append(fund_thought)

        # Step 3: Check sentiment
        sent_thought = self._analyze_sentiment(context)
        thoughts.append(sent_thought)

        # Step 4: Consider macro context
        macro_thought = self._analyze_macro(context)
        thoughts.append(macro_thought)

        # Step 5: Compile bullish thesis
        thesis, confidence = self._compile_thesis(context, thoughts)

        # Step 6: Counter any bearish arguments if provided
        if context.get("bear_argument"):
            counter_thought = self._counter_bear(context["bear_argument"])
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
        position: str = "bullish",
        previous_bear_argument: str | None = None,
    ) -> AgentResponse:
        """
        Make a bullish argument in debate context.

        Args:
            context: Debate context
            position: Position to argue (always bullish for this agent)
            previous_bear_argument: Bear's previous argument to counter

        Returns:
            AgentResponse with bullish argument
        """
        if previous_bear_argument:
            context = {**context, "bear_argument": previous_bear_argument}

        opportunity = context.get("opportunity", {})
        symbol = opportunity.get("symbol", "Unknown")

        return self.analyze(
            query=f"Construct bullish argument for {symbol}",
            context=context,
        )

    def _analyze_technicals(self, context: dict[str, Any]) -> AgentThought:
        """Analyze technical indicators for bullish signals."""
        symbol = context.get("symbol", context.get("opportunity", {}).get("symbol", "SPY"))
        price = context.get("price", context.get("opportunity", {}).get("price", 0))

        # In production, would use actual technical data
        analysis = context.get("analysis", {})
        technicals = analysis.get("technicals", {})

        signals = []
        confidence = 0.6

        # Check common technical indicators
        if technicals.get("rsi", 50) < 70:
            signals.append("RSI not overbought, room to run")
            confidence += 0.05

        if technicals.get("above_sma_20", True):
            signals.append("Trading above 20-day SMA")
            confidence += 0.05

        if technicals.get("macd_bullish", False):
            signals.append("MACD showing bullish crossover")
            confidence += 0.1

        if technicals.get("volume_surge", False):
            signals.append("Volume supporting upward move")
            confidence += 0.05

        if not signals:
            signals = ["Price action shows consolidation pattern", "Setup for potential breakout"]

        return AgentThought(
            thought_type=ThoughtType.REASONING,
            content=f"Technical analysis for {symbol}: " + "; ".join(signals),
            metadata={"confidence": min(confidence, 0.95)},
        )

    def _analyze_fundamentals(self, context: dict[str, Any]) -> AgentThought:
        """Analyze fundamental factors for bullish signals."""
        analysis = context.get("analysis", {})
        fundamentals = analysis.get("fundamentals", {})

        signals = []
        confidence = 0.6

        if fundamentals.get("earnings_growth", 0) > 0:
            signals.append(f"Positive earnings growth: {fundamentals.get('earnings_growth')}%")
            confidence += 0.1

        if fundamentals.get("revenue_growth", 0) > 0:
            signals.append("Revenue trending higher")
            confidence += 0.05

        if fundamentals.get("margin_expanding", False):
            signals.append("Profit margins expanding")
            confidence += 0.05

        if not signals:
            signals = ["Valuations reasonable relative to growth", "Strong market position"]

        return AgentThought(
            thought_type=ThoughtType.REASONING,
            content="Fundamental factors: " + "; ".join(signals),
            metadata={"confidence": min(confidence, 0.9)},
        )

    def _analyze_sentiment(self, context: dict[str, Any]) -> AgentThought:
        """Analyze sentiment indicators for bullish signals."""
        analysis = context.get("analysis", {})
        sentiment = analysis.get("sentiment", {})

        signals = []
        confidence = 0.55

        if sentiment.get("institutional_buying", False):
            signals.append("Institutional accumulation detected")
            confidence += 0.1

        if sentiment.get("analyst_upgrades", 0) > 0:
            signals.append("Recent analyst upgrades")
            confidence += 0.05

        if sentiment.get("options_flow_bullish", False):
            signals.append("Bullish options flow")
            confidence += 0.1

        if sentiment.get("news_sentiment", 0) > 0:
            signals.append("Positive news sentiment")
            confidence += 0.05

        if not signals:
            signals = ["Sentiment neutral, potential for positive surprise"]

        return AgentThought(
            thought_type=ThoughtType.REASONING,
            content="Sentiment indicators: " + "; ".join(signals),
            metadata={"confidence": min(confidence, 0.85)},
        )

    def _analyze_macro(self, context: dict[str, Any]) -> AgentThought:
        """Analyze macro context for bullish signals."""
        analysis = context.get("analysis", {})
        macro = analysis.get("macro", {})

        signals = []
        confidence = 0.55

        if macro.get("market_trend", "neutral") == "bullish":
            signals.append("Broader market in uptrend")
            confidence += 0.1

        if macro.get("sector_strength", False):
            signals.append("Sector showing relative strength")
            confidence += 0.05

        if macro.get("fed_accommodative", False):
            signals.append("Fed policy supportive")
            confidence += 0.1

        if macro.get("seasonality_positive", False):
            signals.append("Seasonal patterns favorable")
            confidence += 0.05

        if not signals:
            signals = ["Macro environment stable", "No major headwinds"]

        return AgentThought(
            thought_type=ThoughtType.REASONING,
            content="Macro context: " + "; ".join(signals),
            metadata={"confidence": min(confidence, 0.8)},
        )

    def _counter_bear(self, bear_argument: str) -> AgentThought:
        """Counter a bearish argument."""
        # Simple counter - in production would use LLM
        counters = [
            "While risks exist, they are priced in",
            "Historical patterns suggest resilience",
            "Risk/reward still favors upside",
        ]

        return AgentThought(
            thought_type=ThoughtType.REASONING,
            content=f"Countering bear concerns: {counters[0]}. " f"The bearish view overlooks key positive catalysts.",
            metadata={"confidence": 0.7},
        )

    def _compile_thesis(
        self,
        context: dict[str, Any],
        thoughts: list[AgentThought],
    ) -> tuple[str, float]:
        """Compile thoughts into bullish thesis."""
        symbol = context.get("symbol", context.get("opportunity", {}).get("symbol", "SPY"))

        # Average confidence across thoughts
        confidences = [t.metadata.get("confidence", 0.5) for t in thoughts]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.6

        thesis = f"BULLISH on {symbol}. "

        # Extract key points from thoughts
        key_points = []
        for t in thoughts:
            thought_confidence = t.metadata.get("confidence", 0.5)
            if thought_confidence > 0.6:
                # Extract first point from content
                points = t.content.split(": ")[-1].split(";")
                if points:
                    key_points.append(points[0].strip())

        if key_points:
            thesis += " ".join(key_points[:3]) + ". "

        thesis += f"Recommend accumulation with confidence {avg_confidence:.0%}."

        return thesis, avg_confidence

    def _format_reasoning(self, thoughts: list[AgentThought]) -> str:
        """Format thoughts into reasoning string."""
        reasoning_parts = []
        for i, t in enumerate(thoughts, 1):
            if t.thought_type == ThoughtType.REASONING:
                reasoning_parts.append(f"{i}. {t.content}")
        return "\n".join(reasoning_parts)

    def get_recent_signals(self) -> list[BullishSignal]:
        """Get recently identified bullish signals."""
        return self.recent_signals

    def clear_signals(self) -> None:
        """Clear recent signals."""
        self.recent_signals = []


def create_bull_researcher(
    llm_client: Any | None = None,
    tools: list[Tool] | None = None,
) -> BullResearcher:
    """
    Factory function to create a bull researcher.

    Args:
        llm_client: LLM client for analysis
        tools: Available analysis tools

    Returns:
        Configured BullResearcher instance
    """
    return BullResearcher(
        llm_client=llm_client,
        tools=tools,
    )
