"""
LLM Provider Implementations

Provides concrete implementations for OpenAI and Anthropic providers.
"""

import json
import logging
import math
import time
from typing import Any

from .base import (
    AnalysisResult,
    BaseLLMProvider,
    NewsItem,
    Sentiment,
    SentimentResult,
)


logger = logging.getLogger(__name__)


def _validate_confidence(value: Any, default: float = 0.5) -> float:
    """Validate and clamp confidence value to [0, 1] range."""
    try:
        conf = float(value)
        if math.isnan(conf) or math.isinf(conf):
            return default
        return max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        return default


def _validate_score(value: Any, default: float = 0.0) -> float:
    """Validate and clamp score value to [-1, 1] range."""
    try:
        score = float(value)
        if math.isnan(score) or math.isinf(score):
            return default
        return max(-1.0, min(1.0, score))
    except (TypeError, ValueError):
        return default


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider implementation."""

    def __init__(self, config: dict[str, Any]):
        """Initialize OpenAI provider."""
        super().__init__(config)
        self._client = None

    def _ensure_client(self) -> None:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

    @property
    def name(self) -> str:
        """Provider name."""
        return "openai"

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment using GPT."""
        start_time = time.time()
        self._ensure_client()

        prompt = f"""Analyze the sentiment of the following financial text.
Return a JSON object with:
- sentiment: one of "very_bullish", "bullish", "neutral", "bearish", "very_bearish"
- confidence: a number between 0 and 1
- score: a number between -1 (very bearish) and 1 (very bullish)
- reasoning: brief explanation

Text: {text[:1000]}

Respond with only the JSON object, no other text."""

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )

        result_text = response.choices[0].message.content.strip()

        try:
            result = json.loads(result_text)
            sentiment_map = {
                "very_bullish": Sentiment.VERY_BULLISH,
                "bullish": Sentiment.BULLISH,
                "neutral": Sentiment.NEUTRAL,
                "bearish": Sentiment.BEARISH,
                "very_bearish": Sentiment.VERY_BEARISH,
            }
            sentiment = sentiment_map.get(result.get("sentiment", "neutral"), Sentiment.NEUTRAL)
            confidence = _validate_confidence(result.get("confidence", 0.5))
            score = _validate_score(result.get("score", 0))
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse OpenAI sentiment response: %s", e)
            logger.debug("Raw response: %s", result_text[:500])
            sentiment = Sentiment.NEUTRAL
            confidence = 0.5
            score = 0.0
        except (KeyError, TypeError) as e:
            logger.warning("Invalid OpenAI sentiment response structure: %s", e)
            sentiment = Sentiment.NEUTRAL
            confidence = 0.5
            score = 0.0

        processing_time = (time.time() - start_time) * 1000

        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            score=score,
            provider=self.name,
            raw_output=result_text,
            processing_time_ms=processing_time,
        )

    def analyze_news(self, news_items: list[NewsItem]) -> AnalysisResult:
        """Analyze news items for trading insights."""
        start_time = time.time()
        self._ensure_client()

        # Format news for analysis
        news_text = "\n\n".join([f"[{item.source}] {item.title}\n{item.content[:300]}" for item in news_items[:10]])

        prompt = f"""Analyze the following financial news articles for trading insights.

News Articles:
{news_text}

Provide a JSON response with:
- summary: brief overall summary (1-2 sentences)
- key_points: array of 3-5 key points
- trading_signals: array of objects with {{symbol, direction ("buy"/"sell"/"hold"), confidence, reasoning}}
- risk_factors: array of risk factors to consider
- overall_confidence: number 0-1 indicating confidence in analysis

Respond with only the JSON object."""

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        result_text = response.choices[0].message.content.strip()

        try:
            result = json.loads(result_text)
            return AnalysisResult(
                summary=result.get("summary", ""),
                key_points=result.get("key_points", []),
                trading_signals=result.get("trading_signals", []),
                risk_factors=result.get("risk_factors", []),
                confidence=float(result.get("overall_confidence", 0.5)),
                provider=self.name,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return AnalysisResult(
                summary="Failed to parse analysis",
                key_points=[],
                trading_signals=[],
                risk_factors=["Analysis parsing error"],
                confidence=0.0,
                provider=self.name,
            )

    def analyze_option_chain(self, symbol: str, chain_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Analyze option chain for underpriced options."""
        self._ensure_client()

        # Format chain data summary
        chain_summary = json.dumps(chain_data, indent=2)[:3000]

        prompt = f"""Analyze this option chain for {symbol} and identify potentially underpriced options.

Option Chain Data:
{chain_summary}

Look for:
1. Options with IV below historical average
2. Unusual put/call ratios suggesting mispricing
3. Options near support/resistance levels with favorable risk/reward

Return a JSON array of recommendations, each with:
- contract: option contract symbol
- type: "call" or "put"
- strike: strike price
- expiry: expiration date
- current_price: current option price
- fair_value_estimate: your estimated fair value
- underpriced_pct: percentage underpriced
- reasoning: brief explanation
- confidence: 0-1 confidence score

Respond with only the JSON array."""

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        result_text = response.choices[0].message.content.strip()

        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            return []


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self, config: dict[str, Any]):
        """Initialize Anthropic provider."""
        super().__init__(config)
        self._client = None

    def _ensure_client(self) -> None:
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")

    @property
    def name(self) -> str:
        """Provider name."""
        return "anthropic"

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment using Claude."""
        start_time = time.time()
        self._ensure_client()

        prompt = f"""Analyze the sentiment of the following financial text.
Return a JSON object with:
- sentiment: one of "very_bullish", "bullish", "neutral", "bearish", "very_bearish"
- confidence: a number between 0 and 1
- score: a number between -1 (very bearish) and 1 (very bullish)
- reasoning: brief explanation

Text: {text[:1000]}

Respond with only the JSON object, no other text."""

        response = self._client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text.strip()

        try:
            result = json.loads(result_text)
            sentiment_map = {
                "very_bullish": Sentiment.VERY_BULLISH,
                "bullish": Sentiment.BULLISH,
                "neutral": Sentiment.NEUTRAL,
                "bearish": Sentiment.BEARISH,
                "very_bearish": Sentiment.VERY_BEARISH,
            }
            sentiment = sentiment_map.get(result.get("sentiment", "neutral"), Sentiment.NEUTRAL)
            confidence = _validate_confidence(result.get("confidence", 0.5))
            score = _validate_score(result.get("score", 0))
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Anthropic sentiment response: %s", e)
            logger.debug("Raw response: %s", result_text[:500])
            sentiment = Sentiment.NEUTRAL
            confidence = 0.5
            score = 0.0
        except (KeyError, TypeError) as e:
            logger.warning("Invalid Anthropic sentiment response structure: %s", e)
            sentiment = Sentiment.NEUTRAL
            confidence = 0.5
            score = 0.0

        processing_time = (time.time() - start_time) * 1000

        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            score=score,
            provider=self.name,
            raw_output=result_text,
            processing_time_ms=processing_time,
        )

    def analyze_news(self, news_items: list[NewsItem]) -> AnalysisResult:
        """Analyze news items for trading insights."""
        self._ensure_client()

        news_text = "\n\n".join([f"[{item.source}] {item.title}\n{item.content[:300]}" for item in news_items[:10]])

        prompt = f"""Analyze the following financial news articles for trading insights.

News Articles:
{news_text}

Provide a JSON response with:
- summary: brief overall summary (1-2 sentences)
- key_points: array of 3-5 key points
- trading_signals: array of objects with {{symbol, direction ("buy"/"sell"/"hold"), confidence, reasoning}}
- risk_factors: array of risk factors to consider
- overall_confidence: number 0-1 indicating confidence in analysis

Respond with only the JSON object."""

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text.strip()

        try:
            result = json.loads(result_text)
            return AnalysisResult(
                summary=result.get("summary", ""),
                key_points=result.get("key_points", []),
                trading_signals=result.get("trading_signals", []),
                risk_factors=result.get("risk_factors", []),
                confidence=float(result.get("overall_confidence", 0.5)),
                provider=self.name,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return AnalysisResult(
                summary="Failed to parse analysis",
                key_points=[],
                trading_signals=[],
                risk_factors=["Analysis parsing error"],
                confidence=0.0,
                provider=self.name,
            )

    def analyze_option_chain(self, symbol: str, chain_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Analyze option chain for underpriced options."""
        self._ensure_client()

        chain_summary = json.dumps(chain_data, indent=2)[:3000]

        prompt = f"""Analyze this option chain for {symbol} and identify potentially underpriced options.

Option Chain Data:
{chain_summary}

Look for:
1. Options with IV below historical average
2. Unusual put/call ratios suggesting mispricing
3. Options near support/resistance levels with favorable risk/reward

Return a JSON array of recommendations, each with:
- contract: option contract symbol
- type: "call" or "put"
- strike: strike price
- expiry: expiration date
- current_price: current option price
- fair_value_estimate: your estimated fair value
- underpriced_pct: percentage underpriced
- reasoning: brief explanation
- confidence: 0-1 confidence score

Respond with only the JSON array."""

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text.strip()

        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            return []


def create_provider(provider_name: str, config: dict[str, Any]) -> BaseLLMProvider | None:
    """
    Create an LLM provider by name.

    Args:
        provider_name: Name of provider ("openai" or "anthropic")
        config: Provider configuration

    Returns:
        Provider instance or None if invalid
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }

    provider_class = providers.get(provider_name.lower())
    if provider_class:
        return provider_class(config)
    return None


__all__ = [
    "AnthropicProvider",
    "OpenAIProvider",
    "create_provider",
]
