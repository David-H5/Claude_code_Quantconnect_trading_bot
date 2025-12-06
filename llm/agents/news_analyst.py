"""
News Analyst Agent Implementation

UPGRADE-014 Feature 6: Multi-Agent Architecture

Specialized agent for news event analysis that differs from SentimentAnalyst:
- SentimentAnalyst: Focuses on overall sentiment from multiple sources
- NewsAnalyst: Focuses on news event classification, impact assessment, timing

Based on TradingAgents (2024) multi-agent architecture pattern.

QuantConnect Compatible: Yes
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
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
from llm.prompts import get_prompt, get_registry


try:
    from llm.clients import AnthropicClient, ClaudeModel
except ImportError:
    AnthropicClient = None
    ClaudeModel = None

try:
    from models.circuit_breaker import TradingCircuitBreaker
except ImportError:
    TradingCircuitBreaker = None


class NewsEventType(Enum):
    """Classification of news event types."""

    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    MERGER_ACQUISITION = "merger_acquisition"
    REGULATORY = "regulatory"
    PRODUCT_LAUNCH = "product_launch"
    EXECUTIVE_CHANGE = "executive_change"
    LEGAL = "legal"
    MACROECONOMIC = "macroeconomic"
    ANALYST_ACTION = "analyst_action"
    INSIDER_TRADING = "insider_trading"
    DIVIDEND = "dividend"
    BUYBACK = "buyback"
    GENERAL = "general"


class NewsImpactLevel(Enum):
    """Impact level of news on stock price."""

    CRITICAL = "critical"  # >5% expected move
    HIGH = "high"  # 2-5% expected move
    MEDIUM = "medium"  # 0.5-2% expected move
    LOW = "low"  # <0.5% expected move
    NEUTRAL = "neutral"  # No expected move


class NewsTimeRelevance(Enum):
    """Time relevance of news event."""

    BREAKING = "breaking"  # Just happened, immediate relevance
    TODAY = "today"  # Today's news
    RECENT = "recent"  # Last 1-3 days
    OLD = "old"  # >3 days old
    STALE = "stale"  # >7 days, likely priced in


@dataclass
class NewsAnalysis:
    """Result of news analysis."""

    headline: str
    event_type: NewsEventType
    impact_level: NewsImpactLevel
    time_relevance: NewsTimeRelevance
    sentiment_score: float  # -1.0 to 1.0
    sentiment_direction: str  # "bullish", "bearish", "neutral"
    confidence: float
    key_entities: list[str]
    affected_symbols: list[str]
    source_reliability: float  # 0.0 to 1.0
    trading_implications: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "headline": self.headline,
            "event_type": self.event_type.value,
            "impact_level": self.impact_level.value,
            "time_relevance": self.time_relevance.value,
            "sentiment_score": self.sentiment_score,
            "sentiment_direction": self.sentiment_direction,
            "confidence": self.confidence,
            "key_entities": self.key_entities,
            "affected_symbols": self.affected_symbols,
            "source_reliability": self.source_reliability,
            "trading_implications": self.trading_implications,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class NewsAnalystResult:
    """Complete result from news analyst."""

    symbol: str
    analyses: list[NewsAnalysis]
    overall_sentiment: float  # Weighted average sentiment
    overall_impact: NewsImpactLevel  # Highest impact level
    actionable: bool  # Whether news warrants action
    recommendation: str  # Brief recommendation
    key_risks: list[str]
    key_catalysts: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "analyses": [a.to_dict() for a in self.analyses],
            "overall_sentiment": self.overall_sentiment,
            "overall_impact": self.overall_impact.value,
            "actionable": self.actionable,
            "recommendation": self.recommendation,
            "key_risks": self.key_risks,
            "key_catalysts": self.key_catalysts,
            "timestamp": self.timestamp.isoformat(),
        }


# Source reliability scores (0.0 to 1.0)
SOURCE_RELIABILITY = {
    # Major financial news
    "bloomberg": 0.95,
    "reuters": 0.95,
    "wsj": 0.92,
    "ft": 0.92,
    "cnbc": 0.85,
    "marketwatch": 0.82,
    "benzinga": 0.78,
    "seeking_alpha": 0.72,
    # Company sources
    "sec_filing": 0.99,
    "company_pr": 0.90,
    "earnings_call": 0.95,
    # Aggregators
    "yahoo_finance": 0.75,
    "google_finance": 0.75,
    # Social/blogs
    "twitter": 0.45,
    "reddit": 0.40,
    "blog": 0.35,
    # Default
    "unknown": 0.50,
}


class NewsAnalyst(TradingAgent):
    """
    News Analyst agent - specialized for news event analysis.

    Responsibilities:
    - Classify news event types (earnings, M&A, regulatory, etc.)
    - Assess impact level and trading implications
    - Evaluate source reliability
    - Extract key entities and affected symbols
    - Determine time relevance and whether news is priced in
    - Provide actionable trading recommendations

    Differs from SentimentAnalyst in focusing on:
    - Event classification vs general sentiment
    - Source reliability assessment
    - Trading timing implications
    - Cross-symbol impact analysis

    Based on TradingAgents (2024) multi-agent architecture.
    """

    # Default system prompt for news analyst
    DEFAULT_SYSTEM_PROMPT = """You are a specialized News Analyst for trading decisions.

Your role is to analyze news events and assess their trading implications:

1. EVENT CLASSIFICATION: Identify the type of news (earnings, M&A, regulatory, etc.)
2. IMPACT ASSESSMENT: Estimate the potential price impact (critical, high, medium, low)
3. SOURCE EVALUATION: Assess source reliability and potential bias
4. TIMING ANALYSIS: Determine if news is priced in or still actionable
5. CROSS-SYMBOL IMPACT: Identify other affected securities

Output Format (JSON):
{
    "event_type": "earnings|guidance|merger_acquisition|regulatory|...",
    "impact_level": "critical|high|medium|low|neutral",
    "sentiment_score": -1.0 to 1.0,
    "sentiment_direction": "bullish|bearish|neutral",
    "confidence": 0.0 to 1.0,
    "key_entities": ["list of key people/companies mentioned"],
    "affected_symbols": ["list of affected tickers"],
    "trading_implications": ["list of trading implications"],
    "recommendation": "brief trading recommendation",
    "time_relevance": "breaking|today|recent|old|stale"
}

Focus on ACTIONABLE insights. If news is stale or already priced in, say so.
"""

    def __init__(
        self,
        llm_client: Any | None = None,
        version: str = "active",
        max_iterations: int = 2,
        timeout_ms: float = 8000.0,
        source_reliability: dict[str, float] | None = None,
    ):
        """
        Initialize news analyst agent.

        Args:
            llm_client: Anthropic API client (optional)
            version: Prompt version to use
            max_iterations: Max ReAct iterations
            timeout_ms: Max execution time
            source_reliability: Custom source reliability scores
        """
        # Try to get prompt from registry, fallback to default
        system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self.prompt_version = None

        try:
            prompt_version = get_prompt(AgentRole.NEWS_ANALYST, version=version)
            if prompt_version:
                system_prompt = prompt_version.template
                self.prompt_version = prompt_version
        except Exception:
            pass  # Use default prompt

        super().__init__(
            name="NewsAnalyst",
            role=AgentRole.NEWS_ANALYST,
            system_prompt=system_prompt,
            tools=[],
            max_iterations=max_iterations,
            timeout_ms=timeout_ms,
            llm_client=llm_client,
        )

        self.registry = None
        try:
            self.registry = get_registry()
        except Exception:
            pass

        self.model = ClaudeModel.SONNET_4 if ClaudeModel else None
        self.source_reliability = source_reliability or SOURCE_RELIABILITY.copy()

    def analyze(
        self,
        query: str,
        context: dict[str, Any],
    ) -> AgentResponse:
        """
        Analyze news for trading implications.

        Args:
            query: Analysis question (e.g., "Analyze AAPL news")
            context: News data
                - symbol: Ticker symbol
                - news_articles: List of news articles
                    - headline: Article headline
                    - content: Article content (optional)
                    - source: News source
                    - timestamp: Publication time
                    - url: Article URL (optional)
                - market_context: Current market data (optional)
                    - price: Current price
                    - change_pct: Today's change
                    - volume: Trading volume

        Returns:
            AgentResponse with news analysis
        """
        start_time = time.time()
        thoughts: list[AgentThought] = []
        symbol = context.get("symbol", "UNKNOWN")

        try:
            # Analyze each news article
            analyses = self._analyze_news_articles(context)

            # Build analysis prompt
            full_prompt = self._build_analysis_prompt(query, context, analyses)

            # If LLM client available, get enhanced analysis
            if self.llm_client and self.model:
                messages = [{"role": "user", "content": full_prompt}]

                response = self.llm_client.chat(
                    model=self.model,
                    messages=messages,
                    system=self.system_prompt,
                    max_tokens=1500 if self.prompt_version else 1000,
                    temperature=0.3,
                )

                try:
                    llm_result = json.loads(response.content)
                    # Merge LLM insights with our analysis
                    for analysis in analyses:
                        if "enhanced_implications" in llm_result:
                            analysis.trading_implications.extend(llm_result.get("enhanced_implications", [])[:2])
                except json.JSONDecodeError:
                    pass  # Keep our analysis

            # Create final result
            result = self._compile_result(symbol, analyses)
            final_answer = json.dumps(result.to_dict(), indent=2)
            confidence = result.overall_sentiment  # Use sentiment as confidence proxy

            success = True
            error = None

            thoughts.append(
                AgentThought(
                    thought_type=ThoughtType.FINAL_ANSWER,
                    content=final_answer,
                    metadata={
                        "articles_analyzed": len(analyses),
                        "overall_impact": result.overall_impact.value,
                        "actionable": result.actionable,
                    },
                )
            )

        except Exception as e:
            final_answer = json.dumps(
                {
                    "error": str(e),
                    "symbol": symbol,
                    "recommendation": "Unable to analyze news",
                }
            )
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

        # Record usage metrics if registry available
        if self.registry and self.prompt_version:
            self.registry.record_usage(
                role=AgentRole.NEWS_ANALYST,
                version=self.prompt_version.version,
                success=success,
                response_time_ms=execution_time_ms,
                confidence=abs(confidence),
            )

        return AgentResponse(
            agent_name=self.name,
            agent_role=self.role,
            query=query,
            thoughts=thoughts,
            final_answer=final_answer,
            confidence=abs(confidence),
            tools_used=[],
            execution_time_ms=execution_time_ms,
            success=success,
            error=error,
        )

    def _analyze_news_articles(
        self,
        context: dict[str, Any],
    ) -> list[NewsAnalysis]:
        """Analyze individual news articles."""
        analyses = []
        articles = context.get("news_articles", [])

        for article in articles[:10]:  # Limit to 10 most recent
            headline = article.get("headline", "")
            content = article.get("content", "")
            source = article.get("source", "unknown").lower()
            timestamp_str = article.get("timestamp", "")

            # Classify event type
            event_type = self._classify_event_type(headline, content)

            # Assess impact level
            impact_level = self._assess_impact_level(event_type, headline, content)

            # Determine time relevance
            time_relevance = self._determine_time_relevance(timestamp_str)

            # Calculate sentiment
            sentiment_score, sentiment_direction = self._calculate_sentiment(headline, content)

            # Get source reliability
            source_reliability = self._get_source_reliability(source)

            # Extract entities and symbols
            key_entities = self._extract_entities(headline, content)
            affected_symbols = self._extract_symbols(headline, content, context)

            # Determine trading implications
            trading_implications = self._determine_implications(event_type, impact_level, sentiment_direction)

            analysis = NewsAnalysis(
                headline=headline[:200],  # Truncate
                event_type=event_type,
                impact_level=impact_level,
                time_relevance=time_relevance,
                sentiment_score=sentiment_score,
                sentiment_direction=sentiment_direction,
                confidence=source_reliability * 0.8 + 0.2,  # Base confidence on source
                key_entities=key_entities[:5],
                affected_symbols=affected_symbols[:5],
                source_reliability=source_reliability,
                trading_implications=trading_implications[:3],
            )
            analyses.append(analysis)

        return analyses

    def _classify_event_type(self, headline: str, content: str) -> NewsEventType:
        """Classify the news event type based on content."""
        text = (headline + " " + content).lower()

        # Keywords for each event type
        event_keywords = {
            NewsEventType.EARNINGS: [
                "earnings",
                "eps",
                "revenue",
                "profit",
                "loss",
                "quarter",
                "fiscal",
                "beat",
                "miss",
                "report",
            ],
            NewsEventType.GUIDANCE: ["guidance", "outlook", "forecast", "expects", "raises", "lowers", "target"],
            NewsEventType.MERGER_ACQUISITION: [
                "merger",
                "acquisition",
                "acquire",
                "takeover",
                "buyout",
                "deal",
                "bid",
                "offer",
            ],
            NewsEventType.REGULATORY: [
                "fda",
                "sec",
                "regulatory",
                "approval",
                "investigation",
                "compliance",
                "lawsuit",
                "antitrust",
            ],
            NewsEventType.PRODUCT_LAUNCH: ["launch", "release", "unveil", "announce", "new product", "innovation"],
            NewsEventType.EXECUTIVE_CHANGE: ["ceo", "cfo", "executive", "resign", "appoint", "hire", "step down"],
            NewsEventType.LEGAL: ["lawsuit", "settlement", "court", "judge", "verdict", "litigation"],
            NewsEventType.MACROECONOMIC: ["fed", "interest rate", "inflation", "gdp", "unemployment", "economic"],
            NewsEventType.ANALYST_ACTION: ["upgrade", "downgrade", "price target", "rating", "analyst", "coverage"],
            NewsEventType.DIVIDEND: ["dividend", "payout", "yield", "distribution"],
            NewsEventType.BUYBACK: ["buyback", "repurchase", "share repurchase"],
            NewsEventType.INSIDER_TRADING: ["insider", "form 4", "executive purchase", "executive sell"],
        }

        # Score each event type
        best_type = NewsEventType.GENERAL
        best_score = 0

        for event_type, keywords in event_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_type = event_type

        return best_type

    def _assess_impact_level(
        self,
        event_type: NewsEventType,
        headline: str,
        content: str,
    ) -> NewsImpactLevel:
        """Assess the impact level of the news."""
        text = (headline + " " + content).lower()

        # High-impact event types
        high_impact_events = {
            NewsEventType.EARNINGS,
            NewsEventType.MERGER_ACQUISITION,
            NewsEventType.REGULATORY,
        }

        # Critical keywords
        critical_keywords = [
            "bankruptcy",
            "fraud",
            "fda approval",
            "major acquisition",
            "breakthrough",
            "massive",
            "significant",
            "dramatic",
        ]

        # High impact keywords
        high_keywords = ["beat", "miss", "surprise", "unexpected", "record", "surge", "plunge", "soar"]

        # Check for critical
        if any(kw in text for kw in critical_keywords):
            return NewsImpactLevel.CRITICAL

        # Check for high
        if event_type in high_impact_events:
            return NewsImpactLevel.HIGH

        if any(kw in text for kw in high_keywords):
            return NewsImpactLevel.HIGH

        # Default assessment based on event type
        impact_mapping = {
            NewsEventType.GUIDANCE: NewsImpactLevel.HIGH,
            NewsEventType.EXECUTIVE_CHANGE: NewsImpactLevel.MEDIUM,
            NewsEventType.PRODUCT_LAUNCH: NewsImpactLevel.MEDIUM,
            NewsEventType.ANALYST_ACTION: NewsImpactLevel.MEDIUM,
            NewsEventType.DIVIDEND: NewsImpactLevel.LOW,
            NewsEventType.BUYBACK: NewsImpactLevel.LOW,
            NewsEventType.GENERAL: NewsImpactLevel.LOW,
        }

        return impact_mapping.get(event_type, NewsImpactLevel.MEDIUM)

    def _determine_time_relevance(self, timestamp_str: str) -> NewsTimeRelevance:
        """Determine the time relevance of news."""
        if not timestamp_str:
            return NewsTimeRelevance.RECENT

        try:
            # Try parsing various formats
            for fmt in [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ]:
                try:
                    timestamp = datetime.strptime(timestamp_str[:19], fmt)
                    break
                except ValueError:
                    continue
            else:
                return NewsTimeRelevance.RECENT

            now = datetime.now()
            age_hours = (now - timestamp).total_seconds() / 3600

            if age_hours < 2:
                return NewsTimeRelevance.BREAKING
            elif age_hours < 24:
                return NewsTimeRelevance.TODAY
            elif age_hours < 72:
                return NewsTimeRelevance.RECENT
            elif age_hours < 168:
                return NewsTimeRelevance.OLD
            else:
                return NewsTimeRelevance.STALE

        except Exception:
            return NewsTimeRelevance.RECENT

    def _calculate_sentiment(
        self,
        headline: str,
        content: str,
    ) -> tuple:
        """Calculate sentiment score and direction."""
        text = (headline + " " + content).lower()

        # Sentiment word lists
        bullish_words = [
            "beat",
            "exceed",
            "surge",
            "soar",
            "rally",
            "gain",
            "rise",
            "upgrade",
            "outperform",
            "bullish",
            "strong",
            "growth",
            "record",
            "breakthrough",
            "success",
            "positive",
            "optimistic",
            "approval",
            "profit",
            "increase",
        ]
        bearish_words = [
            "miss",
            "fall",
            "drop",
            "plunge",
            "decline",
            "loss",
            "cut",
            "downgrade",
            "underperform",
            "bearish",
            "weak",
            "slowdown",
            "concern",
            "risk",
            "warning",
            "negative",
            "pessimistic",
            "reject",
            "fail",
            "decrease",
        ]

        bullish_count = sum(1 for w in bullish_words if w in text)
        bearish_count = sum(1 for w in bearish_words if w in text)
        total = bullish_count + bearish_count

        if total == 0:
            return 0.0, "neutral"

        score = (bullish_count - bearish_count) / total
        score = max(-1.0, min(1.0, score))  # Clamp

        if score > 0.2:
            direction = "bullish"
        elif score < -0.2:
            direction = "bearish"
        else:
            direction = "neutral"

        return score, direction

    def _get_source_reliability(self, source: str) -> float:
        """Get reliability score for news source."""
        source_lower = source.lower().replace(" ", "_")

        # Check for partial matches
        for known_source, reliability in self.source_reliability.items():
            if known_source in source_lower or source_lower in known_source:
                return reliability

        return self.source_reliability.get("unknown", 0.50)

    def _extract_entities(self, headline: str, content: str) -> list[str]:
        """Extract key entities from text."""
        # Simple extraction - in production would use NER
        text = headline + " " + content
        entities = []

        # Common entity patterns (capitalized words)
        words = text.split()
        for i, word in enumerate(words):
            if word[0:1].isupper() and len(word) > 2:
                # Skip common words
                if word.lower() not in [
                    "the",
                    "and",
                    "for",
                    "are",
                    "but",
                    "not",
                    "you",
                    "all",
                    "can",
                    "her",
                    "was",
                    "one",
                    "our",
                    "out",
                ]:
                    entities.append(word.strip(".,;:!?\"'()"))

        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique_entities.append(e)

        return unique_entities[:10]

    def _extract_symbols(
        self,
        headline: str,
        content: str,
        context: dict[str, Any],
    ) -> list[str]:
        """Extract ticker symbols from text."""
        symbols = []

        # Add context symbol
        if "symbol" in context:
            symbols.append(context["symbol"])

        # Look for ticker patterns (1-5 uppercase letters)
        text = headline + " " + content
        words = text.split()

        for word in words:
            clean = word.strip(".,;:!?\"'()$")
            if (
                clean.isupper()
                and 1 <= len(clean) <= 5
                and clean.isalpha()
                and clean not in ["A", "I", "THE", "AND", "FOR"]
            ):
                if clean not in symbols:
                    symbols.append(clean)

        return symbols[:10]

    def _determine_implications(
        self,
        event_type: NewsEventType,
        impact_level: NewsImpactLevel,
        sentiment_direction: str,
    ) -> list[str]:
        """Determine trading implications based on analysis."""
        implications = []

        # Impact-based implications
        if impact_level == NewsImpactLevel.CRITICAL:
            implications.append("High volatility expected - consider position sizing")
        elif impact_level == NewsImpactLevel.HIGH:
            implications.append("Significant move possible - watch for confirmation")

        # Event-type specific implications
        event_implications = {
            NewsEventType.EARNINGS: [
                "Post-earnings drift possible if surprise significant",
                "Options IV likely elevated",
            ],
            NewsEventType.MERGER_ACQUISITION: [
                "Acquirer typically sees short-term weakness",
                "Target typically gaps toward deal price",
            ],
            NewsEventType.REGULATORY: ["Binary outcome - high risk/reward", "Sector peers may also be affected"],
            NewsEventType.GUIDANCE: [
                "Forward guidance often more important than results",
                "Revisions drive longer-term moves",
            ],
            NewsEventType.ANALYST_ACTION: [
                "Short-term momentum impact, often fades",
                "Multiple analyst actions more significant",
            ],
        }

        if event_type in event_implications:
            implications.extend(event_implications[event_type][:2])

        # Sentiment-based implications
        if sentiment_direction == "bullish":
            implications.append("Positive bias - look for dips to enter")
        elif sentiment_direction == "bearish":
            implications.append("Negative bias - exercise caution on longs")

        return implications

    def _compile_result(
        self,
        symbol: str,
        analyses: list[NewsAnalysis],
    ) -> NewsAnalystResult:
        """Compile individual analyses into overall result."""
        if not analyses:
            return NewsAnalystResult(
                symbol=symbol,
                analyses=[],
                overall_sentiment=0.0,
                overall_impact=NewsImpactLevel.NEUTRAL,
                actionable=False,
                recommendation="No news to analyze",
                key_risks=[],
                key_catalysts=[],
            )

        # Calculate weighted sentiment (weight by reliability)
        total_weight = sum(a.source_reliability for a in analyses)
        if total_weight > 0:
            overall_sentiment = sum(a.sentiment_score * a.source_reliability for a in analyses) / total_weight
        else:
            overall_sentiment = 0.0

        # Find highest impact
        impact_order = [
            NewsImpactLevel.CRITICAL,
            NewsImpactLevel.HIGH,
            NewsImpactLevel.MEDIUM,
            NewsImpactLevel.LOW,
            NewsImpactLevel.NEUTRAL,
        ]
        overall_impact = NewsImpactLevel.NEUTRAL
        for impact in impact_order:
            if any(a.impact_level == impact for a in analyses):
                overall_impact = impact
                break

        # Determine if actionable
        actionable = overall_impact in [NewsImpactLevel.CRITICAL, NewsImpactLevel.HIGH] and any(
            a.time_relevance in [NewsTimeRelevance.BREAKING, NewsTimeRelevance.TODAY] for a in analyses
        )

        # Extract risks and catalysts
        key_risks = []
        key_catalysts = []
        for a in analyses:
            if a.sentiment_score < -0.2:
                key_risks.extend(a.trading_implications[:1])
            elif a.sentiment_score > 0.2:
                key_catalysts.extend(a.trading_implications[:1])

        # Generate recommendation
        if overall_sentiment > 0.3 and actionable:
            recommendation = f"Bullish news flow - consider long positions on {symbol}"
        elif overall_sentiment < -0.3 and actionable:
            recommendation = f"Bearish news flow - exercise caution on {symbol}"
        elif actionable:
            recommendation = f"Mixed signals - wait for confirmation on {symbol}"
        else:
            recommendation = f"No actionable news - maintain current stance on {symbol}"

        return NewsAnalystResult(
            symbol=symbol,
            analyses=analyses,
            overall_sentiment=overall_sentiment,
            overall_impact=overall_impact,
            actionable=actionable,
            recommendation=recommendation,
            key_risks=list(set(key_risks))[:3],
            key_catalysts=list(set(key_catalysts))[:3],
        )

    def _build_analysis_prompt(
        self,
        query: str,
        context: dict[str, Any],
        analyses: list[NewsAnalysis],
    ) -> str:
        """Build analysis prompt for LLM enhancement."""
        prompt_parts = [
            f"NEWS ANALYSIS REQUEST: {query}",
            "",
            f"Symbol: {context.get('symbol', 'UNKNOWN')}",
            "",
            "=" * 60,
            "PRE-ANALYZED NEWS",
            "=" * 60,
        ]

        for i, a in enumerate(analyses[:5], 1):
            prompt_parts.extend(
                [
                    f"\nArticle {i}: {a.headline[:100]}...",
                    f"  Event Type: {a.event_type.value}",
                    f"  Impact: {a.impact_level.value}",
                    f"  Sentiment: {a.sentiment_direction} ({a.sentiment_score:.2f})",
                    f"  Time Relevance: {a.time_relevance.value}",
                ]
            )

        prompt_parts.extend(
            [
                "",
                "=" * 60,
                "ENHANCE ANALYSIS",
                "=" * 60,
                "",
                "Based on the pre-analyzed news, provide:",
                "1. Any additional trading implications",
                "2. Cross-symbol impacts",
                "3. Risk factors not captured",
                "",
                "Respond in JSON format with 'enhanced_implications' list.",
            ]
        )

        return "\n".join(prompt_parts)


def create_news_analyst(
    llm_client: Any | None = None,
    version: str = "active",
    source_reliability: dict[str, float] | None = None,
) -> NewsAnalyst:
    """
    Factory function to create news analyst agent.

    Args:
        llm_client: Anthropic API client (optional)
        version: Prompt version
        source_reliability: Custom source reliability scores

    Returns:
        NewsAnalyst instance
    """
    return NewsAnalyst(
        llm_client=llm_client,
        version=version,
        source_reliability=source_reliability,
    )


def create_safe_news_analyst(
    llm_client: Any | None,
    circuit_breaker: "TradingCircuitBreaker",
    version: str = "active",
    risk_config: RiskTierConfig | None = None,
) -> SafeAgentWrapper:
    """
    Factory function to create news analyst agent with safety wrapper.

    Args:
        llm_client: Anthropic API client
        circuit_breaker: Trading circuit breaker for risk controls
        version: Prompt version
        risk_config: Optional risk tier configuration

    Returns:
        SafeAgentWrapper wrapping a NewsAnalyst
    """
    analyst = NewsAnalyst(
        llm_client=llm_client,
        version=version,
    )
    return wrap_agent_with_safety(
        agent=analyst,
        circuit_breaker=circuit_breaker,
        risk_config=risk_config,
    )
