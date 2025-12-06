"""
News Alert Manager for Trading Decisions.

Bridges news analysis with the alerting service and circuit breaker.
Monitors news sentiment and generates trading alerts when significant
events are detected.

UPGRADE-014: LLM Sentiment Integration (December 2025)

Research Sources:
- Real-Time News Sentiment Engine patterns (GitHub 2024)
- Moody's News Sentiment in Financial Analysis (2024)
- TradingAgents Multi-Agent Framework (arXiv Dec 2024)
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from .base import NewsItem, SentimentResult


logger = logging.getLogger(__name__)


# ==============================================================================
# Data Types
# ==============================================================================


class NewsImpact(Enum):
    """Impact level of news event."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NewsEventType(Enum):
    """Type of news event."""

    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    REGULATORY = "regulatory"
    ECONOMIC = "economic"
    MARKET_MOVE = "market_move"
    ANALYST_ACTION = "analyst_action"
    MANAGEMENT_CHANGE = "management_change"
    PRODUCT_LAUNCH = "product_launch"
    LEGAL = "legal"
    GENERAL = "general"


@dataclass
class NewsEvent:
    """Processed news event with impact assessment."""

    news_item: NewsItem
    symbols: list[str]
    event_type: NewsEventType
    impact: NewsImpact
    sentiment_score: float
    sentiment_confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    keywords_matched: list[str] = field(default_factory=list)
    suggested_action: str = ""

    @property
    def is_bullish(self) -> bool:
        """Check if event is bullish."""
        return self.sentiment_score > 0.2

    @property
    def is_bearish(self) -> bool:
        """Check if event is bearish."""
        return self.sentiment_score < -0.2

    @property
    def is_high_impact(self) -> bool:
        """Check if event is high impact."""
        return self.impact in (NewsImpact.HIGH, NewsImpact.CRITICAL)

    @property
    def requires_circuit_breaker_check(self) -> bool:
        """Check if event should trigger circuit breaker evaluation."""
        return self.impact == NewsImpact.CRITICAL or (self.is_bearish and self.is_high_impact)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "headline": self.news_item.title,
            "source": self.news_item.source,
            "symbols": self.symbols,
            "event_type": self.event_type.value,
            "impact": self.impact.value,
            "sentiment_score": self.sentiment_score,
            "sentiment_confidence": self.sentiment_confidence,
            "timestamp": self.timestamp.isoformat(),
            "keywords_matched": self.keywords_matched,
            "suggested_action": self.suggested_action,
            "is_bullish": self.is_bullish,
            "is_bearish": self.is_bearish,
            "is_high_impact": self.is_high_impact,
        }


@dataclass
class NewsAlertConfig:
    """Configuration for news alert manager."""

    # Sentiment thresholds
    high_impact_sentiment_threshold: float = 0.7
    circuit_breaker_sentiment_threshold: float = -0.8

    # Impact scoring
    critical_keywords: list[str] = field(
        default_factory=lambda: [
            "bankruptcy",
            "fraud",
            "sec investigation",
            "delisting",
            "halt",
            "suspended",
            "crash",
            "collapse",
            "scandal",
        ]
    )
    high_impact_keywords: list[str] = field(
        default_factory=lambda: [
            "earnings",
            "merger",
            "acquisition",
            "fda",
            "lawsuit",
            "downgrade",
            "upgrade",
            "ceo",
            "buyback",
            "dividend",
        ]
    )

    # Alert configuration
    min_confidence_for_alert: float = 0.5
    alert_cooldown_minutes: int = 5
    max_alerts_per_hour: int = 20

    # Circuit breaker integration
    enable_circuit_breaker_triggers: bool = True
    consecutive_negative_threshold: int = 3


# ==============================================================================
# News Alert Manager
# ==============================================================================


class NewsAlertManager:
    """
    Manages news-driven trading alerts and circuit breaker integration.

    Features:
    - Real-time news monitoring and classification
    - Impact-based alert prioritization
    - Circuit breaker integration for extreme events
    - Cooldown to prevent alert spam
    - Event type classification

    Example:
        >>> manager = NewsAlertManager(
        ...     watchlist=["AAPL", "MSFT", "SPY"],
        ...     config=NewsAlertConfig(),
        ... )
        >>> manager.add_alert_listener(lambda e: print(e.to_dict()))
        >>> events = manager.process_news([news_item])
        >>> for event in events:
        ...     if event.is_high_impact:
        ...         handle_high_impact(event)
    """

    def __init__(
        self,
        watchlist: list[str] | None = None,
        config: NewsAlertConfig | None = None,
        alerting_service: Any | None = None,
        circuit_breaker: Any | None = None,
    ):
        """
        Initialize news alert manager.

        Args:
            watchlist: Symbols to monitor
            config: Alert configuration
            alerting_service: AlertingService for sending alerts
            circuit_breaker: TradingCircuitBreaker for halt triggers
        """
        self.watchlist: set[str] = set(s.upper() for s in (watchlist or []))
        self.config = config or NewsAlertConfig()
        self.alerting_service = alerting_service
        self.circuit_breaker = circuit_breaker

        # Event tracking
        self._event_history: list[NewsEvent] = []
        self._max_history = 500
        self._alert_cooldowns: dict[str, datetime] = {}
        self._alerts_this_hour: int = 0
        self._hour_start: datetime = datetime.now(timezone.utc)

        # Consecutive negative tracking for circuit breaker
        self._consecutive_negative: dict[str, int] = {}

        # Listeners
        self._alert_listeners: list[Callable[[NewsEvent], None]] = []
        self._circuit_breaker_listeners: list[Callable[[NewsEvent], None]] = []

        self._lock = threading.Lock()

    def process_news(
        self,
        news_items: list[NewsItem],
        sentiment_results: dict[str, SentimentResult] | None = None,
    ) -> list[NewsEvent]:
        """
        Process news items and generate events.

        Args:
            news_items: List of NewsItem to process
            sentiment_results: Pre-computed sentiment results keyed by news hash

        Returns:
            List of NewsEvent objects for relevant news
        """
        events = []
        sentiment_results = sentiment_results or {}

        for news in news_items:
            # Filter by watchlist
            relevant_symbols = [s for s in (news.symbols or []) if s.upper() in self.watchlist]

            if not relevant_symbols and self.watchlist:
                # Check headline for watchlist symbols
                for symbol in self.watchlist:
                    if symbol in news.title.upper():
                        relevant_symbols.append(symbol)

            if not relevant_symbols and self.watchlist:
                continue

            # Get sentiment (from pre-computed or from news item)
            news_hash = f"{news.title}:{news.source}"
            sentiment = sentiment_results.get(news_hash)

            if sentiment is None and news.sentiment:
                sentiment = news.sentiment

            if sentiment is None:
                # Skip without sentiment data
                continue

            # Classify event
            event_type = self._classify_event_type(news)
            keywords_matched = self._find_keywords(news)
            impact = self._determine_impact(news, sentiment, keywords_matched)

            # Generate action suggestion
            suggested_action = self._generate_action(relevant_symbols, sentiment, impact)

            event = NewsEvent(
                news_item=news,
                symbols=relevant_symbols or list(self.watchlist)[:5],
                event_type=event_type,
                impact=impact,
                sentiment_score=sentiment.score,
                sentiment_confidence=sentiment.confidence,
                keywords_matched=keywords_matched,
                suggested_action=suggested_action,
            )

            events.append(event)
            self._record_event(event)

            # Send alerts for significant events
            if self._should_alert(event):
                self._send_alert(event)

            # Check circuit breaker conditions
            if event.requires_circuit_breaker_check:
                self._check_circuit_breaker(event)

        return events

    def _classify_event_type(self, news: NewsItem) -> NewsEventType:
        """Classify the type of news event."""
        text = f"{news.title} {news.content}".lower()

        classifications = [
            (NewsEventType.EARNINGS, ["earnings", "revenue", "eps", "quarterly"]),
            (NewsEventType.MERGER_ACQUISITION, ["merger", "acquisition", "takeover", "buyout"]),
            (NewsEventType.REGULATORY, ["fda", "sec", "regulation", "compliance", "approval"]),
            (NewsEventType.ECONOMIC, ["fed", "interest rate", "inflation", "gdp", "employment"]),
            (NewsEventType.ANALYST_ACTION, ["upgrade", "downgrade", "price target", "rating"]),
            (NewsEventType.MANAGEMENT_CHANGE, ["ceo", "cfo", "executive", "resign", "appoint"]),
            (NewsEventType.PRODUCT_LAUNCH, ["launch", "release", "new product", "unveil"]),
            (NewsEventType.LEGAL, ["lawsuit", "settlement", "court", "investigation"]),
            (NewsEventType.MARKET_MOVE, ["surge", "plunge", "rally", "selloff", "volatility"]),
        ]

        for event_type, keywords in classifications:
            if any(kw in text for kw in keywords):
                return event_type

        return NewsEventType.GENERAL

    def _find_keywords(self, news: NewsItem) -> list[str]:
        """Find matched keywords in news."""
        text = f"{news.title} {news.content}".lower()
        matched = []

        # Critical keywords
        for keyword in self.config.critical_keywords:
            if keyword.lower() in text:
                matched.append(keyword)

        # High impact keywords
        for keyword in self.config.high_impact_keywords:
            if keyword.lower() in text:
                matched.append(keyword)

        return matched

    def _determine_impact(
        self,
        news: NewsItem,
        sentiment: SentimentResult,
        keywords: list[str],
    ) -> NewsImpact:
        """Determine the impact level of a news event."""
        # Check for critical keywords
        critical_matches = [k for k in keywords if k.lower() in [kw.lower() for kw in self.config.critical_keywords]]
        if critical_matches:
            return NewsImpact.CRITICAL

        # Check sentiment extremes
        if abs(sentiment.score) >= self.config.high_impact_sentiment_threshold:
            if sentiment.confidence >= 0.7:
                return NewsImpact.HIGH

        # Check high impact keywords
        high_impact_matches = [
            k for k in keywords if k.lower() in [kw.lower() for kw in self.config.high_impact_keywords]
        ]
        if high_impact_matches and abs(sentiment.score) >= 0.5:
            return NewsImpact.HIGH
        elif high_impact_matches:
            return NewsImpact.MEDIUM

        # Moderate sentiment
        if abs(sentiment.score) >= 0.4:
            return NewsImpact.MEDIUM

        return NewsImpact.LOW

    def _generate_action(
        self,
        symbols: list[str],
        sentiment: SentimentResult,
        impact: NewsImpact,
    ) -> str:
        """Generate action suggestion based on event."""
        symbol_str = ", ".join(symbols[:3])

        if impact == NewsImpact.CRITICAL:
            if sentiment.score < 0:
                return f"URGENT: Review all positions in {symbol_str}. Consider immediate exit."
            else:
                return f"URGENT: Major positive event for {symbol_str}. Review for opportunity."

        if impact == NewsImpact.HIGH:
            if sentiment.score > 0.5:
                return f"Consider bullish position in {symbol_str}."
            elif sentiment.score < -0.5:
                return f"Consider reducing exposure to {symbol_str}."
            else:
                return f"Monitor {symbol_str} closely."

        if impact == NewsImpact.MEDIUM:
            if sentiment.score > 0.3:
                return f"Watch {symbol_str} for entry opportunity."
            elif sentiment.score < -0.3:
                return f"Watch {symbol_str} for potential weakness."

        return f"No immediate action for {symbol_str}."

    def _record_event(self, event: NewsEvent) -> None:
        """Record event in history."""
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

    def _should_alert(self, event: NewsEvent) -> bool:
        """Check if event should trigger an alert."""
        # Check confidence threshold
        if event.sentiment_confidence < self.config.min_confidence_for_alert:
            return False

        # Only alert for medium+ impact
        if event.impact == NewsImpact.LOW:
            return False

        # Check hourly limit
        now = datetime.now(timezone.utc)
        if now - self._hour_start > timedelta(hours=1):
            self._alerts_this_hour = 0
            self._hour_start = now

        if self._alerts_this_hour >= self.config.max_alerts_per_hour:
            return False

        # Check cooldown per symbol
        for symbol in event.symbols:
            cooldown_key = f"{symbol}:{event.event_type.value}"
            if cooldown_key in self._alert_cooldowns:
                cooldown_expiry = self._alert_cooldowns[cooldown_key]
                if now < cooldown_expiry:
                    return False

        return True

    def _send_alert(self, event: NewsEvent) -> None:
        """Send alert for event."""
        now = datetime.now(timezone.utc)
        cooldown_delta = timedelta(minutes=self.config.alert_cooldown_minutes)

        # Update cooldowns
        for symbol in event.symbols:
            cooldown_key = f"{symbol}:{event.event_type.value}"
            self._alert_cooldowns[cooldown_key] = now + cooldown_delta

        self._alerts_this_hour += 1

        # Notify listeners
        for listener in self._alert_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Error in alert listener: {e}")

        # Send to alerting service
        if self.alerting_service:
            self._send_to_alerting_service(event)

    def _send_to_alerting_service(self, event: NewsEvent) -> None:
        """Send event to alerting service."""
        try:
            # Map impact to severity
            severity_map = {
                NewsImpact.LOW: "INFO",
                NewsImpact.MEDIUM: "WARNING",
                NewsImpact.HIGH: "ERROR",
                NewsImpact.CRITICAL: "CRITICAL",
            }

            # Build alert data
            data = {
                "symbols": event.symbols,
                "sentiment_score": event.sentiment_score,
                "event_type": event.event_type.value,
                "impact": event.impact.value,
                "source": event.news_item.source,
            }

            # Try different API patterns
            if hasattr(self.alerting_service, "send_trading_alert"):
                from utils.alerting_service import AlertSeverity

                severity = getattr(AlertSeverity, severity_map[event.impact])
                self.alerting_service.send_trading_alert(
                    title=f"News: {event.news_item.title[:50]}",
                    message=event.suggested_action,
                    severity=severity,
                    data=data,
                )
            elif hasattr(self.alerting_service, "send_alert"):
                self.alerting_service.send_alert(
                    title=f"News: {event.news_item.title[:50]}",
                    message=event.suggested_action,
                    severity=severity_map[event.impact],
                    data=data,
                )

        except Exception as e:
            logger.error(f"Error sending to alerting service: {e}")

    def _check_circuit_breaker(self, event: NewsEvent) -> None:
        """Check if event should trigger circuit breaker."""
        if not self.config.enable_circuit_breaker_triggers:
            return

        # Track consecutive negative news
        for symbol in event.symbols:
            if event.is_bearish:
                self._consecutive_negative[symbol] = self._consecutive_negative.get(symbol, 0) + 1
            else:
                self._consecutive_negative[symbol] = 0

        # Check for circuit breaker conditions
        should_trigger = False
        trigger_reason = ""

        # Critical negative sentiment
        if (
            event.sentiment_score <= self.config.circuit_breaker_sentiment_threshold
            and event.impact == NewsImpact.CRITICAL
        ):
            should_trigger = True
            trigger_reason = f"Critical negative news: {event.news_item.title[:50]}"

        # Consecutive negative news
        for symbol in event.symbols:
            if self._consecutive_negative.get(symbol, 0) >= self.config.consecutive_negative_threshold:
                should_trigger = True
                trigger_reason = (
                    f"Consecutive negative news for {symbol} " f"({self._consecutive_negative[symbol]} events)"
                )
                break

        if should_trigger:
            # Notify circuit breaker listeners
            for listener in self._circuit_breaker_listeners:
                try:
                    listener(event)
                except Exception as e:
                    logger.error(f"Error in circuit breaker listener: {e}")

            # Trigger circuit breaker if available
            if self.circuit_breaker:
                self._trigger_circuit_breaker(event, trigger_reason)

    def _trigger_circuit_breaker(
        self,
        event: NewsEvent,
        reason: str,
    ) -> None:
        """Trigger circuit breaker for event."""
        try:
            if hasattr(self.circuit_breaker, "halt_all_trading"):
                self.circuit_breaker.halt_all_trading(reason)
                logger.warning(f"Circuit breaker triggered by news: {reason}")
            elif hasattr(self.circuit_breaker, "trigger"):
                self.circuit_breaker.trigger(reason)
                logger.warning(f"Circuit breaker triggered by news: {reason}")
        except Exception as e:
            logger.error(f"Error triggering circuit breaker: {e}")

    # ==========================================================================
    # Watchlist Management
    # ==========================================================================

    def add_to_watchlist(self, symbol: str) -> None:
        """Add symbol to watchlist."""
        self.watchlist.add(symbol.upper())

    def remove_from_watchlist(self, symbol: str) -> None:
        """Remove symbol from watchlist."""
        self.watchlist.discard(symbol.upper())

    def set_watchlist(self, symbols: list[str]) -> None:
        """Set the entire watchlist."""
        self.watchlist = set(s.upper() for s in symbols)

    # ==========================================================================
    # Listener Management
    # ==========================================================================

    def add_alert_listener(
        self,
        callback: Callable[[NewsEvent], None],
    ) -> None:
        """Add listener for news alerts."""
        self._alert_listeners.append(callback)

    def remove_alert_listener(
        self,
        callback: Callable[[NewsEvent], None],
    ) -> None:
        """Remove alert listener."""
        if callback in self._alert_listeners:
            self._alert_listeners.remove(callback)

    def add_circuit_breaker_listener(
        self,
        callback: Callable[[NewsEvent], None],
    ) -> None:
        """Add listener for circuit breaker events."""
        self._circuit_breaker_listeners.append(callback)

    def remove_circuit_breaker_listener(
        self,
        callback: Callable[[NewsEvent], None],
    ) -> None:
        """Remove circuit breaker listener."""
        if callback in self._circuit_breaker_listeners:
            self._circuit_breaker_listeners.remove(callback)

    # ==========================================================================
    # History and Stats
    # ==========================================================================

    def get_recent_events(
        self,
        limit: int = 20,
        symbol: str | None = None,
        impact: NewsImpact | None = None,
        hours: int = 24,
    ) -> list[NewsEvent]:
        """
        Get recent news events.

        Args:
            limit: Maximum events to return
            symbol: Filter by symbol
            impact: Filter by impact level
            hours: How many hours back to look

        Returns:
            Filtered list of events
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        with self._lock:
            events = [e for e in self._event_history if e.timestamp > cutoff]

        if symbol:
            events = [e for e in events if symbol.upper() in e.symbols]

        if impact:
            events = [e for e in events if e.impact == impact]

        return sorted(
            events,
            key=lambda e: e.timestamp,
            reverse=True,
        )[:limit]

    def get_symbol_sentiment_summary(
        self,
        symbol: str,
        hours: int = 24,
    ) -> dict[str, Any]:
        """
        Get sentiment summary for a symbol.

        Args:
            symbol: Symbol to summarize
            hours: Lookback period

        Returns:
            Dictionary with sentiment summary
        """
        events = self.get_recent_events(
            limit=100,
            symbol=symbol,
            hours=hours,
        )

        if not events:
            return {
                "symbol": symbol,
                "event_count": 0,
                "avg_sentiment": 0.0,
                "latest_sentiment": 0.0,
                "bullish_count": 0,
                "bearish_count": 0,
                "high_impact_count": 0,
            }

        sentiments = [e.sentiment_score for e in events]
        return {
            "symbol": symbol,
            "event_count": len(events),
            "avg_sentiment": sum(sentiments) / len(sentiments),
            "latest_sentiment": events[0].sentiment_score,
            "bullish_count": sum(1 for e in events if e.is_bullish),
            "bearish_count": sum(1 for e in events if e.is_bearish),
            "high_impact_count": sum(1 for e in events if e.is_high_impact),
            "dominant_event_type": max(
                set(e.event_type for e in events),
                key=lambda t: sum(1 for e in events if e.event_type == t),
            ).value,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        with self._lock:
            return {
                "watchlist_size": len(self.watchlist),
                "events_tracked": len(self._event_history),
                "alerts_this_hour": self._alerts_this_hour,
                "active_cooldowns": len(self._alert_cooldowns),
                "alert_listeners": len(self._alert_listeners),
                "circuit_breaker_listeners": len(self._circuit_breaker_listeners),
                "circuit_breaker_connected": self.circuit_breaker is not None,
                "alerting_service_connected": self.alerting_service is not None,
            }

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._event_history.clear()
            self._consecutive_negative.clear()


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_news_alert_manager(
    watchlist: list[str] | None = None,
    config: dict[str, Any] | None = None,
    alerting_service: Any | None = None,
    circuit_breaker: Any | None = None,
) -> NewsAlertManager:
    """
    Factory function to create news alert manager.

    Args:
        watchlist: Symbols to monitor
        config: Configuration dictionary
        alerting_service: AlertingService instance
        circuit_breaker: TradingCircuitBreaker instance

    Returns:
        Configured NewsAlertManager instance
    """
    alert_config = NewsAlertConfig()

    if config:
        # Override defaults with provided config
        if "high_impact_sentiment_threshold" in config:
            alert_config.high_impact_sentiment_threshold = config["high_impact_sentiment_threshold"]
        if "circuit_breaker_sentiment_threshold" in config:
            alert_config.circuit_breaker_sentiment_threshold = config["circuit_breaker_sentiment_threshold"]
        if "critical_keywords" in config:
            alert_config.critical_keywords = config["critical_keywords"]
        if "high_impact_keywords" in config:
            alert_config.high_impact_keywords = config["high_impact_keywords"]
        if "min_confidence_for_alert" in config:
            alert_config.min_confidence_for_alert = config["min_confidence_for_alert"]
        if "alert_cooldown_minutes" in config:
            alert_config.alert_cooldown_minutes = config["alert_cooldown_minutes"]
        if "max_alerts_per_hour" in config:
            alert_config.max_alerts_per_hour = config["max_alerts_per_hour"]
        if "enable_circuit_breaker_triggers" in config:
            alert_config.enable_circuit_breaker_triggers = config["enable_circuit_breaker_triggers"]
        if "consecutive_negative_threshold" in config:
            alert_config.consecutive_negative_threshold = config["consecutive_negative_threshold"]

    return NewsAlertManager(
        watchlist=watchlist,
        config=alert_config,
        alerting_service=alerting_service,
        circuit_breaker=circuit_breaker,
    )


__all__ = [
    "NewsAlertConfig",
    "NewsAlertManager",
    "NewsEvent",
    "NewsEventType",
    "NewsImpact",
    "create_news_alert_manager",
]
