"""
Base classes for LLM integration.

Provides abstract interfaces and common utilities for LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Sentiment(Enum):
    """Sentiment classification."""

    VERY_BULLISH = 2
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1
    VERY_BEARISH = -2


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""

    sentiment: Sentiment
    confidence: float  # 0.0 to 1.0
    score: float  # -1.0 to 1.0
    provider: str
    raw_output: str | None = None
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sentiment": self.sentiment.name,
            "confidence": self.confidence,
            "score": self.score,
            "provider": self.provider,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class NewsItem:
    """Represents a news article or headline."""

    title: str
    content: str
    source: str
    published_at: datetime
    symbols: list[str] = field(default_factory=list)
    url: str | None = None
    sentiment: SentimentResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content[:500] if self.content else "",
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "symbols": self.symbols,
            "url": self.url,
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
        }


@dataclass
class AnalysisResult:
    """Result of LLM analysis."""

    summary: str
    key_points: list[str]
    trading_signals: list[dict[str, Any]]
    risk_factors: list[str]
    confidence: float
    provider: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "key_points": self.key_points,
            "trading_signals": self.trading_signals,
            "risk_factors": self.risk_factors,
            "confidence": self.confidence,
            "provider": self.provider,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize provider with configuration.

        Args:
            config: Provider-specific configuration
        """
        self.config = config
        self.model = config.get("model", "")
        self.api_key = config.get("api_key", "")
        self.max_tokens = config.get("max_tokens", 1000)
        self.temperature = config.get("temperature", 0.3)

    @abstractmethod
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult with classification
        """
        pass

    @abstractmethod
    def analyze_news(self, news_items: list[NewsItem]) -> AnalysisResult:
        """
        Analyze news items for trading insights.

        Args:
            news_items: List of news items to analyze

        Returns:
            AnalysisResult with insights
        """
        pass

    @abstractmethod
    def analyze_option_chain(self, symbol: str, chain_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Analyze option chain for underpriced options.

        Args:
            symbol: Underlying symbol
            chain_data: Option chain data

        Returns:
            List of option recommendations
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @property
    def is_available(self) -> bool:
        """Check if provider is available (API key set, etc.)."""
        return bool(self.api_key)


class BaseSentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers."""

    @abstractmethod
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of text."""
        pass

    @abstractmethod
    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment of multiple texts."""
        pass


__all__ = [
    "AnalysisResult",
    "BaseLLMProvider",
    "BaseSentimentAnalyzer",
    "NewsItem",
    "Sentiment",
    "SentimentResult",
]
