"""
News Analyzer Module

Fetches, parses, and analyzes financial news from multiple sources.
"""

import asyncio
import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .base import NewsItem, SentimentResult
from .ensemble import LLMEnsemble, create_ensemble


logger = logging.getLogger(__name__)


@dataclass
class NewsAlert:
    """Alert generated from news analysis."""

    symbol: str
    headline: str
    sentiment: SentimentResult
    urgency: str  # "high", "medium", "low"
    source: str
    published_at: datetime
    suggested_action: str
    options_suggestions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "headline": self.headline,
            "sentiment": self.sentiment.to_dict(),
            "urgency": self.urgency,
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "suggested_action": self.suggested_action,
            "options_suggestions": self.options_suggestions,
        }


class NewsAnalyzer:
    """
    Analyzes financial news for trading signals.

    Fetches news from configured sources, analyzes sentiment,
    and generates alerts with option suggestions.
    """

    def __init__(
        self,
        config: dict[str, Any],
        ensemble: LLMEnsemble | None = None,
        alert_callback: Callable[[NewsAlert], None] | None = None,
    ):
        """
        Initialize news analyzer.

        Args:
            config: News alerts configuration from settings.json
            ensemble: LLM ensemble for analysis (created if not provided)
            alert_callback: Callback for new alerts
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.sources = config.get("sources", ["reuters", "bloomberg"])
        self.keywords = set(config.get("keywords", []))
        self.watchlist = set(config.get("watchlist", []))
        self.sentiment_threshold = config.get("sentiment_threshold", 0.7)
        self.refresh_interval = config.get("refresh_interval_seconds", 30)

        self.ensemble = ensemble or create_ensemble()
        self.alert_callback = alert_callback

        # Cache to avoid duplicate alerts
        self._seen_news: set[str] = set()
        self._news_cache: list[NewsItem] = []
        self._last_fetch: datetime | None = None

    def _hash_news(self, news: NewsItem) -> str:
        """Generate unique hash for news item."""
        content = f"{news.title}{news.source}{news.published_at.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()

    def _extract_symbols(self, text: str) -> list[str]:
        """Extract stock symbols from text."""
        # Common patterns: $AAPL, (AAPL), AAPL:
        patterns = [
            r"\$([A-Z]{1,5})\b",  # $AAPL
            r"\(([A-Z]{1,5})\)",  # (AAPL)
            r"\b([A-Z]{1,5}):",  # AAPL:
        ]

        symbols = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            symbols.update(matches)

        # Also check watchlist
        text_upper = text.upper()
        for symbol in self.watchlist:
            if symbol in text_upper:
                symbols.add(symbol)

        return list(symbols)

    def _is_relevant(self, news: NewsItem) -> bool:
        """Check if news item is relevant based on watchlist and keywords."""
        # Check symbols
        if any(s in self.watchlist for s in news.symbols):
            return True

        # Check keywords
        text = f"{news.title} {news.content}".lower()
        if any(kw.lower() in text for kw in self.keywords):
            return True

        return False

    def _determine_urgency(self, sentiment: SentimentResult, age_minutes: float) -> str:
        """Determine alert urgency based on sentiment strength and recency."""
        if age_minutes < 5 and abs(sentiment.score) > 0.7:
            return "high"
        elif age_minutes < 30 and abs(sentiment.score) > 0.5:
            return "medium"
        else:
            return "low"

    def _generate_action(self, sentiment: SentimentResult, symbol: str) -> str:
        """Generate suggested action based on sentiment."""
        if sentiment.score > 0.7:
            return f"Consider bullish options on {symbol}"
        elif sentiment.score > 0.3:
            return f"Monitor {symbol} for entry opportunity"
        elif sentiment.score < -0.7:
            return f"Consider bearish options or exit {symbol}"
        elif sentiment.score < -0.3:
            return f"Review positions in {symbol}"
        else:
            return f"No immediate action for {symbol}"

    async def fetch_news_async(self) -> list[NewsItem]:
        """
        Fetch news from configured sources asynchronously.

        Returns:
            List of NewsItem objects
        """
        # This is a placeholder - actual implementation would use
        # news APIs like Polygon, Alpha Vantage, or direct RSS feeds
        news_items = []

        # Example implementation for different sources
        for source in self.sources:
            try:
                if source == "reuters":
                    items = await self._fetch_reuters()
                elif source == "bloomberg":
                    items = await self._fetch_bloomberg()
                elif source == "sec_filings":
                    items = await self._fetch_sec()
                elif source == "twitter":
                    items = await self._fetch_twitter()
                else:
                    items = []

                news_items.extend(items)
            except (ConnectionError, TimeoutError, asyncio.TimeoutError) as e:
                logger.warning("Failed to fetch news from %s: %s", source, e)
                continue
            except Exception as e:
                logger.error("Unexpected error fetching news from %s: %s", source, e, exc_info=True)
                continue

        # Extract symbols and filter duplicates
        for item in news_items:
            item.symbols = self._extract_symbols(f"{item.title} {item.content}")

        # Cache and return
        self._news_cache = news_items
        self._last_fetch = datetime.now()

        return news_items

    async def _fetch_reuters(self) -> list[NewsItem]:
        """Fetch from Reuters (placeholder)."""
        # Implement actual Reuters API integration
        return []

    async def _fetch_bloomberg(self) -> list[NewsItem]:
        """Fetch from Bloomberg (placeholder)."""
        # Implement actual Bloomberg API integration
        return []

    async def _fetch_sec(self) -> list[NewsItem]:
        """Fetch from SEC EDGAR (placeholder)."""
        # Implement actual SEC EDGAR integration
        return []

    async def _fetch_twitter(self) -> list[NewsItem]:
        """Fetch from Twitter/X (placeholder)."""
        # Implement actual Twitter API integration
        return []

    def fetch_news(self) -> list[NewsItem]:
        """Synchronous wrapper for fetch_news_async."""
        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
            raise RuntimeError("Cannot use fetch_news in async context. Use fetch_news_async instead.")
        except RuntimeError:
            # Not in async context - safe to create and run a new loop
            pass

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.fetch_news_async())
        finally:
            loop.close()

    def analyze_news(self, news_items: list[NewsItem] | None = None) -> list[NewsAlert]:
        """
        Analyze news items and generate alerts.

        Args:
            news_items: News to analyze (fetches if not provided)

        Returns:
            List of NewsAlert objects for relevant news
        """
        if not self.enabled:
            return []

        if news_items is None:
            news_items = self.fetch_news()

        alerts = []
        now = datetime.now()

        for news in news_items:
            # Skip if already seen
            news_hash = self._hash_news(news)
            if news_hash in self._seen_news:
                continue

            # Check relevance
            if not self._is_relevant(news):
                continue

            # Analyze sentiment
            text = f"{news.title}. {news.content}"
            result = self.ensemble.analyze_sentiment(text)
            news.sentiment = result.sentiment

            # Check if meets threshold
            if abs(result.sentiment.score) < self.sentiment_threshold:
                self._seen_news.add(news_hash)
                continue

            # Calculate age
            age = (now - news.published_at).total_seconds() / 60

            # Generate alert for each symbol
            for symbol in news.symbols:
                if symbol not in self.watchlist:
                    continue

                urgency = self._determine_urgency(result.sentiment, age)
                action = self._generate_action(result.sentiment, symbol)

                alert = NewsAlert(
                    symbol=symbol,
                    headline=news.title,
                    sentiment=result.sentiment,
                    urgency=urgency,
                    source=news.source,
                    published_at=news.published_at,
                    suggested_action=action,
                )

                alerts.append(alert)

                # Trigger callback
                if self.alert_callback:
                    self.alert_callback(alert)

            self._seen_news.add(news_hash)

        return alerts

    def get_recent_news(self, symbol: str | None = None, hours: int = 24) -> list[NewsItem]:
        """
        Get recent news, optionally filtered by symbol.

        Args:
            symbol: Filter by symbol (optional)
            hours: How many hours back to look

        Returns:
            Filtered list of news items
        """
        cutoff = datetime.now() - timedelta(hours=hours)

        news = [n for n in self._news_cache if n.published_at > cutoff]

        if symbol:
            news = [n for n in news if symbol in n.symbols]

        return sorted(news, key=lambda x: x.published_at, reverse=True)

    def add_to_watchlist(self, symbol: str) -> None:
        """Add symbol to watchlist."""
        self.watchlist.add(symbol.upper())

    def remove_from_watchlist(self, symbol: str) -> None:
        """Remove symbol from watchlist."""
        self.watchlist.discard(symbol.upper())

    def add_keyword(self, keyword: str) -> None:
        """Add keyword to track."""
        self.keywords.add(keyword.lower())

    def remove_keyword(self, keyword: str) -> None:
        """Remove keyword from tracking."""
        self.keywords.discard(keyword.lower())

    def clear_cache(self) -> None:
        """Clear seen news cache."""
        self._seen_news.clear()
        self._news_cache.clear()


def create_news_analyzer(
    config: dict[str, Any] | None = None,
    alert_callback: Callable[[NewsAlert], None] | None = None,
) -> NewsAnalyzer:
    """
    Create news analyzer from configuration.

    Args:
        config: News alerts configuration
        alert_callback: Optional callback for alerts

    Returns:
        Configured NewsAnalyzer instance
    """
    if config is None:
        config = {
            "enabled": True,
            "sources": ["reuters"],
            "keywords": [],
            "watchlist": [],
            "sentiment_threshold": 0.7,
            "refresh_interval_seconds": 30,
        }

    return NewsAnalyzer(config, alert_callback=alert_callback)


__all__ = [
    "NewsAlert",
    "NewsAnalyzer",
    "create_news_analyzer",
]
