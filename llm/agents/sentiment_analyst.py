"""
Sentiment Analyst Agent Implementation

Analyzes market sentiment from news, social media, analyst ratings, and FinBERT scores
to determine crowd psychology and contrarian opportunities.

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
from llm.tools import analyze_financial_text
from models.circuit_breaker import TradingCircuitBreaker
from models.exceptions import PromptVersionError


class SentimentAnalyst(TradingAgent):
    """
    Sentiment Analyst agent - analyzes market sentiment using FinBERT and other sources.

    Responsibilities:
    - Analyze news sentiment with FinBERT
    - Track social media sentiment
    - Monitor analyst ratings
    - Identify contrarian opportunities
    - Assess options flow (put/call ratio)

    Uses Claude Sonnet 4 for balanced performance + FinBERT for sentiment scoring.
    """

    def __init__(
        self,
        llm_client: AnthropicClient,
        version: str = "active",
        use_finbert: bool = True,
        max_iterations: int = 2,
        timeout_ms: float = 8000.0,
    ):
        """
        Initialize sentiment analyst agent.

        Args:
            llm_client: Anthropic API client
            version: Prompt version to use ("active", "v1.0")
            use_finbert: Whether to use FinBERT for sentiment analysis
            max_iterations: Max ReAct iterations
            timeout_ms: Max execution time
        """
        # Get prompt template
        prompt_version = get_prompt(AgentRole.SENTIMENT_ANALYST, version=version)
        if not prompt_version:
            raise PromptVersionError(
                agent_role="SENTIMENT_ANALYST",
                version=version,
            )

        super().__init__(
            name="SentimentAnalyst",
            role=AgentRole.SENTIMENT_ANALYST,
            system_prompt=prompt_version.template,
            tools=[],
            max_iterations=max_iterations,
            timeout_ms=timeout_ms,
            llm_client=llm_client,
        )

        self.prompt_version = prompt_version
        self.registry = get_registry()
        self.model = ClaudeModel.SONNET_4
        self.use_finbert = use_finbert

    def analyze(
        self,
        query: str,
        context: dict[str, Any],
    ) -> AgentResponse:
        """
        Analyze market sentiment from multiple sources.

        Args:
            query: Analysis question (e.g., "Analyze AAPL sentiment")
            context: Sentiment data sources
                - symbol: Ticker symbol
                - news_articles: List of news articles (optional)
                    - headline: Article headline
                    - content: Article content
                    - source: News source
                    - timestamp: Publication time
                - social_mentions: Social media sentiment data (optional)
                    - bullish_mentions: Number of bullish mentions
                    - bearish_mentions: Number of bearish mentions
                    - total_mentions: Total mentions
                - analyst_ratings: Analyst rating data (optional)
                    - buy_ratings: Number of buy ratings
                    - hold_ratings: Number of hold ratings
                    - sell_ratings: Number of sell ratings
                    - recent_changes: Recent upgrades/downgrades
                - options_flow: Options trading data (optional)
                    - put_call_ratio: Put/call ratio
                    - unusual_activity: Description of unusual options activity

        Returns:
            AgentResponse with sentiment analysis
        """
        start_time = time.time()
        thoughts: list[AgentThought] = []

        try:
            # Run FinBERT analysis on news if available
            finbert_scores = []
            if self.use_finbert and "news_articles" in context:
                for article in context["news_articles"][:5]:  # Limit to 5 most recent
                    try:
                        text = f"{article.get('headline', '')} {article.get('content', '')[:200]}"
                        score = analyze_financial_text(text)
                        finbert_scores.append(
                            {
                                "headline": article.get("headline", ""),
                                "score": score.score,
                                "label": score.label,
                                "confidence": score.confidence,
                            }
                        )
                    except Exception:
                        # Skip if FinBERT fails (model not loaded, etc.)
                        pass

            # Build the analysis prompt with sentiment data
            full_prompt = self._build_analysis_prompt(query, context, finbert_scores)

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
                        "finbert_scores": finbert_scores,
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
            role=AgentRole.SENTIMENT_ANALYST,
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
            tools_used=["FinBERT"] if finbert_scores else [],
            execution_time_ms=execution_time_ms,
            success=success,
            error=error,
        )

    def _build_analysis_prompt(
        self,
        query: str,
        context: dict[str, Any],
        finbert_scores: list[dict[str, Any]],
    ) -> str:
        """
        Build the sentiment analysis prompt with all data sources.

        Args:
            query: Analysis question
            context: Sentiment data sources
            finbert_scores: FinBERT analysis results

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"SENTIMENT ANALYSIS REQUEST: {query}",
            "",
            "=" * 60,
            "SENTIMENT DATA",
            "=" * 60,
        ]

        # Symbol
        if "symbol" in context:
            prompt_parts.append(f"\nSymbol: {context['symbol']}")

        # FinBERT scores
        if finbert_scores:
            prompt_parts.append("\n--- FINBERT SENTIMENT SCORES (Financial News) ---")
            avg_score = sum(s["score"] for s in finbert_scores) / len(finbert_scores)
            prompt_parts.append(f"Average FinBERT Score: {avg_score:.3f} (-1.0 to +1.0)")

            for i, score in enumerate(finbert_scores, 1):
                prompt_parts.append(f"\nArticle {i}: {score['headline'][:80]}...")
                prompt_parts.append(
                    f"  Score: {score['score']:.3f}, "
                    f"Label: {score['label']}, "
                    f"Confidence: {score['confidence']:.2f}"
                )

        # News articles (if FinBERT not used)
        elif "news_articles" in context:
            prompt_parts.append("\n--- NEWS ARTICLES ---")
            for i, article in enumerate(context["news_articles"][:5], 1):
                prompt_parts.append(f"\nArticle {i}: {article.get('headline', 'No headline')}")
                prompt_parts.append(f"  Source: {article.get('source', 'Unknown')}")

        # Social media sentiment
        if "social_mentions" in context:
            social = context["social_mentions"]
            prompt_parts.append("\n--- SOCIAL MEDIA SENTIMENT ---")
            total = social.get("total_mentions", 0)
            bullish = social.get("bullish_mentions", 0)
            bearish = social.get("bearish_mentions", 0)

            if total > 0:
                bullish_pct = (bullish / total) * 100
                bearish_pct = (bearish / total) * 100
                prompt_parts.append(f"Total Mentions: {total}")
                prompt_parts.append(f"Bullish: {bullish} ({bullish_pct:.1f}%)")
                prompt_parts.append(f"Bearish: {bearish} ({bearish_pct:.1f}%)")

        # Analyst ratings
        if "analyst_ratings" in context:
            ratings = context["analyst_ratings"]
            prompt_parts.append("\n--- ANALYST RATINGS ---")

            buy = ratings.get("buy_ratings", 0)
            hold = ratings.get("hold_ratings", 0)
            sell = ratings.get("sell_ratings", 0)
            total = buy + hold + sell

            if total > 0:
                prompt_parts.append(f"Buy: {buy} ({(buy/total)*100:.1f}%)")
                prompt_parts.append(f"Hold: {hold} ({(hold/total)*100:.1f}%)")
                prompt_parts.append(f"Sell: {sell} ({(sell/total)*100:.1f}%)")

            if "recent_changes" in ratings:
                prompt_parts.append(f"Recent Changes: {ratings['recent_changes']}")

        # Options flow
        if "options_flow" in context:
            flow = context["options_flow"]
            prompt_parts.append("\n--- OPTIONS FLOW ---")

            if "put_call_ratio" in flow:
                ratio = flow["put_call_ratio"]
                prompt_parts.append(f"Put/Call Ratio: {ratio:.2f}")
                if ratio > 1.2:
                    prompt_parts.append("  (High put buying = bearish)")
                elif ratio < 0.7:
                    prompt_parts.append("  (High call buying = bullish)")
                else:
                    prompt_parts.append("  (Balanced)")

            if "unusual_activity" in flow:
                prompt_parts.append(f"Unusual Activity: {flow['unusual_activity']}")

        prompt_parts.extend(
            [
                "",
                "=" * 60,
                "YOUR ANALYSIS",
                "=" * 60,
                "",
                "Based on the above sentiment data from FinBERT, news, social media,",
                "analyst ratings, and options flow, provide your sentiment analysis",
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


def create_sentiment_analyst(
    llm_client: AnthropicClient,
    version: str = "active",
    use_finbert: bool = True,
) -> SentimentAnalyst:
    """
    Factory function to create sentiment analyst agent.

    Args:
        llm_client: Anthropic API client
        version: Prompt version ("active", "v1.0")
        use_finbert: Whether to use FinBERT for sentiment analysis

    Returns:
        SentimentAnalyst instance
    """
    return SentimentAnalyst(
        llm_client=llm_client,
        version=version,
        use_finbert=use_finbert,
    )


def create_safe_sentiment_analyst(
    llm_client: AnthropicClient,
    circuit_breaker: TradingCircuitBreaker,
    version: str = "active",
    use_finbert: bool = True,
    risk_config: RiskTierConfig | None = None,
) -> SafeAgentWrapper:
    """
    Factory function to create sentiment analyst agent with safety wrapper.

    Wraps the sentiment analyst agent with circuit breaker integration and
    risk tier classification. All decisions pass through safety checks before
    execution.

    Args:
        llm_client: Anthropic API client
        circuit_breaker: Trading circuit breaker for risk controls
        version: Prompt version ("active", "v1.0")
        use_finbert: Whether to use FinBERT for sentiment analysis
        risk_config: Optional risk tier configuration

    Returns:
        SafeAgentWrapper wrapping a SentimentAnalyst

    Usage:
        from models.circuit_breaker import create_circuit_breaker

        breaker = create_circuit_breaker(
            max_daily_loss=0.03,
            max_drawdown=0.10,
            max_consecutive_losses=5,
        )

        safe_analyst = create_safe_sentiment_analyst(
            llm_client=client,
            circuit_breaker=breaker,
            version="active",
            use_finbert=True,
        )

        response = safe_analyst.analyze(query, context)
    """
    analyst = SentimentAnalyst(
        llm_client=llm_client,
        version=version,
        use_finbert=use_finbert,
    )
    return wrap_agent_with_safety(
        agent=analyst,
        circuit_breaker=circuit_breaker,
        risk_config=risk_config,
    )
