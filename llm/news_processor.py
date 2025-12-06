"""
Real-Time News Processor Module

Low-latency news ingestion with entity extraction and event classification.
Part of UPGRADE-010 Sprint 3 Expansion - Intelligence & Data Sources.
"""

import hashlib
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .base import SentimentResult
from .entity_extractor import EntityExtractor, create_entity_extractor


logger = logging.getLogger(__name__)


class NewsEventType(Enum):
    """Types of news events."""

    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    FDA_APPROVAL = "fda_approval"
    FDA_REJECTION = "fda_rejection"
    FED_DECISION = "fed_decision"
    FED_SPEECH = "fed_speech"
    MERGER_ACQUISITION = "merger_acquisition"
    STOCK_SPLIT = "stock_split"
    DIVIDEND = "dividend"
    BUYBACK = "buyback"
    LAWSUIT = "lawsuit"
    REGULATORY = "regulatory"
    ANALYST_UPGRADE = "analyst_upgrade"
    ANALYST_DOWNGRADE = "analyst_downgrade"
    INSIDER_TRADE = "insider_trade"
    MANAGEMENT_CHANGE = "management_change"
    PRODUCT_LAUNCH = "product_launch"
    CONTRACT_WIN = "contract_win"
    GENERAL = "general"


class NewsUrgency(Enum):
    """Urgency level for news items."""

    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Same-day action
    MEDIUM = "medium"  # Monitor closely
    LOW = "low"  # Informational


@dataclass
class ProcessedNewsEvent:
    """A processed news event with extracted information."""

    event_id: str
    headline: str
    content: str
    source: str
    published_at: datetime
    processed_at: datetime

    # Extracted entities
    tickers: list[str]
    sectors: list[str]
    primary_ticker: str | None

    # Classification
    event_type: NewsEventType
    urgency: NewsUrgency

    # Sentiment
    sentiment: SentimentResult | None = None

    # Processing metrics
    processing_time_ms: float = 0.0

    # Metadata
    url: str | None = None
    author: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "headline": self.headline,
            "content": self.content[:500] if self.content else "",
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "processed_at": self.processed_at.isoformat(),
            "tickers": self.tickers,
            "sectors": self.sectors,
            "primary_ticker": self.primary_ticker,
            "event_type": self.event_type.value,
            "urgency": self.urgency.value,
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "processing_time_ms": self.processing_time_ms,
            "url": self.url,
            "author": self.author,
            "metadata": self.metadata,
        }


@dataclass
class NewsProcessorConfig:
    """Configuration for news processor."""

    # Processing settings
    enable_sentiment: bool = True
    enable_entity_extraction: bool = True
    enable_event_classification: bool = True

    # Performance
    max_content_length: int = 5000
    target_latency_ms: float = 500.0

    # Deduplication
    dedup_window_hours: int = 24
    similarity_threshold: float = 0.85

    # Filtering
    min_relevance_score: float = 0.3
    required_tickers: set[str] | None = None


class NewsProcessor:
    """
    Real-time news processor with low-latency event detection.

    Provides:
    - Entity extraction (tickers, sectors, companies)
    - Event classification (earnings, FDA, Fed, M&A, etc.)
    - Sentiment analysis integration
    - Sub-second processing pipeline
    """

    # Event type detection patterns
    EVENT_PATTERNS: dict[NewsEventType, list[str]] = {
        NewsEventType.EARNINGS: [
            r"earnings",
            r"quarterly results",
            r"q[1-4] (20\d{2})?results",
            r"beat estimates",
            r"missed estimates",
            r"revenue growth",
            r"eps of",
            r"earnings per share",
            r"profit margin",
        ],
        NewsEventType.GUIDANCE: [
            r"guidance",
            r"outlook",
            r"forecast",
            r"full[- ]year",
            r"expects",
            r"projects",
            r"anticipates revenue",
        ],
        NewsEventType.FDA_APPROVAL: [
            r"fda approv",
            r"fda clears",
            r"approved by fda",
            r"receives fda",
            r"regulatory approval",
        ],
        NewsEventType.FDA_REJECTION: [
            r"fda reject",
            r"fda decline",
            r"complete response letter",
            r"crl from fda",
            r"failed to gain approval",
        ],
        NewsEventType.FED_DECISION: [
            r"fed (rate )?(hike|cut|decision)",
            r"fomc",
            r"interest rate",
            r"federal reserve (raise|lower|hold)",
            r"basis points?",
            r"monetary policy",
            r"quantitative",
        ],
        NewsEventType.FED_SPEECH: [
            r"powell (said|says|speech)",
            r"fed chair",
            r"yellen",
            r"fed official",
            r"central bank(er)?",
        ],
        NewsEventType.MERGER_ACQUISITION: [
            r"acqui(re|sition)",
            r"merge(r|s)",
            r"buyout",
            r"takeover",
            r"all[- ]cash (deal|offer)",
            r"per share offer",
            r"hostile bid",
            r"friendly deal",
        ],
        NewsEventType.STOCK_SPLIT: [
            r"stock split",
            r"share split",
            r"\d+-for-\d+ split",
        ],
        NewsEventType.DIVIDEND: [
            r"dividend",
            r"special dividend",
            r"quarterly dividend",
            r"dividend increase",
            r"dividend cut",
        ],
        NewsEventType.BUYBACK: [
            r"buyback",
            r"share repurchase",
            r"stock repurchase",
            r"repurchase program",
        ],
        NewsEventType.LAWSUIT: [
            r"lawsuit",
            r"sued",
            r"legal action",
            r"class action",
            r"settlement",
            r"litigation",
            r"court ruling",
        ],
        NewsEventType.REGULATORY: [
            r"sec (charge|investigat|fine)",
            r"regulatory",
            r"compliance",
            r"antitrust",
            r"doj",
        ],
        NewsEventType.ANALYST_UPGRADE: [
            r"upgrade",
            r"raises? (price )?target",
            r"bullish",
            r"outperform",
            r"overweight",
            r"buy rating",
        ],
        NewsEventType.ANALYST_DOWNGRADE: [
            r"downgrade",
            r"lower(s|ed) (price )?target",
            r"bearish",
            r"underperform",
            r"underweight",
            r"sell rating",
        ],
        NewsEventType.INSIDER_TRADE: [
            r"insider (buy|sell|trade)",
            r"form 4",
            r"ceo (buy|sell|purchase)",
            r"director (buy|sell)",
            r"10b5-1",
        ],
        NewsEventType.MANAGEMENT_CHANGE: [
            r"(ceo|cfo|cto|coo) (step|resign|retire|appoint|hire)",
            r"new (ceo|cfo|cto)",
            r"leadership change",
            r"executive departure",
            r"board shakeup",
        ],
        NewsEventType.PRODUCT_LAUNCH: [
            r"launch(es|ed|ing)",
            r"new product",
            r"unveil",
            r"announces new",
            r"introduces",
            r"rollout",
        ],
        NewsEventType.CONTRACT_WIN: [
            r"(win|award|secure)(s|ed)? contract",
            r"deal worth",
            r"partnership (with|agreement)",
            r"signed deal",
        ],
    }

    # Urgency detection
    CRITICAL_KEYWORDS = {
        "halt",
        "halted",
        "circuit breaker",
        "flash crash",
        "suspended",
        "bankrupt",
        "fraud",
        "investigation",
        "emergency",
        "recall",
    }

    HIGH_URGENCY_KEYWORDS = {
        "breaking",
        "just in",
        "alert",
        "urgent",
        "plunge",
        "surge",
        "soar",
        "crash",
        "spike",
        "acquisition",
        "merger",
        "fda",
    }

    def __init__(
        self,
        config: NewsProcessorConfig | None = None,
        entity_extractor: EntityExtractor | None = None,
        sentiment_analyzer: Any | None = None,
        event_callback: Callable[[ProcessedNewsEvent], None] | None = None,
    ):
        """
        Initialize news processor.

        Args:
            config: Processor configuration
            entity_extractor: Entity extractor instance
            sentiment_analyzer: Sentiment analyzer instance
            event_callback: Callback for processed events
        """
        self.config = config or NewsProcessorConfig()
        self.entity_extractor = entity_extractor or create_entity_extractor()
        self._sentiment_analyzer = sentiment_analyzer
        self.event_callback = event_callback

        # Compile event patterns
        self._compiled_patterns: dict[NewsEventType, list[re.Pattern]] = {}
        for event_type, patterns in self.EVENT_PATTERNS.items():
            self._compiled_patterns[event_type] = [re.compile(p, re.IGNORECASE) for p in patterns]

        # Deduplication cache
        self._seen_hashes: dict[str, datetime] = {}

        # Processing stats
        self._processed_count = 0
        self._total_processing_time_ms = 0.0

    def process(
        self,
        headline: str,
        content: str = "",
        source: str = "unknown",
        published_at: datetime | None = None,
        url: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProcessedNewsEvent | None:
        """
        Process a news item.

        Args:
            headline: News headline
            content: Full article content
            source: News source name
            published_at: Publication timestamp
            url: Article URL
            author: Author name
            metadata: Additional metadata

        Returns:
            ProcessedNewsEvent or None if filtered/duplicate
        """
        start_time = time.time()

        # Generate event ID
        event_id = self._generate_event_id(headline, source)

        # Check for duplicates
        if self._is_duplicate(headline, content):
            logger.debug(f"Duplicate news filtered: {headline[:50]}...")
            return None

        # Truncate content if needed
        if len(content) > self.config.max_content_length:
            content = content[: self.config.max_content_length]

        # Extract entities
        extraction_result = None
        tickers: list[str] = []
        sectors: list[str] = []
        primary_ticker: str | None = None

        if self.config.enable_entity_extraction:
            # Extract from both headline and content
            combined_text = f"{headline} {content}"
            extraction_result = self.entity_extractor.extract(combined_text)
            tickers = extraction_result.tickers
            sectors = extraction_result.sectors

            # Determine primary ticker (first mentioned in headline)
            headline_extraction = self.entity_extractor.extract(headline)
            if headline_extraction.tickers:
                primary_ticker = headline_extraction.tickers[0]
            elif tickers:
                primary_ticker = tickers[0]

        # Check ticker filter
        if self.config.required_tickers:
            if not any(t in self.config.required_tickers for t in tickers):
                return None

        # Classify event type
        event_type = NewsEventType.GENERAL
        if self.config.enable_event_classification:
            event_type = self._classify_event_type(headline, content)

        # Determine urgency
        urgency = self._determine_urgency(headline, content, event_type)

        # Analyze sentiment
        sentiment = None
        if self.config.enable_sentiment and self._sentiment_analyzer:
            try:
                sentiment = self._sentiment_analyzer.analyze(f"{headline} {content[:500]}")
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Create event
        event = ProcessedNewsEvent(
            event_id=event_id,
            headline=headline,
            content=content,
            source=source,
            published_at=published_at or datetime.now(),
            processed_at=datetime.now(),
            tickers=tickers,
            sectors=sectors,
            primary_ticker=primary_ticker,
            event_type=event_type,
            urgency=urgency,
            sentiment=sentiment,
            processing_time_ms=processing_time_ms,
            url=url,
            author=author,
            metadata=metadata or {},
        )

        # Update stats
        self._processed_count += 1
        self._total_processing_time_ms += processing_time_ms

        # Trigger callback
        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception as e:
                logger.error(f"Event callback failed: {e}")

        return event

    def process_batch(
        self,
        news_items: list[dict[str, Any]],
    ) -> list[ProcessedNewsEvent]:
        """
        Process a batch of news items.

        Args:
            news_items: List of news item dicts with headline, content, etc.

        Returns:
            List of processed events (filtered items excluded)
        """
        results = []

        for item in news_items:
            event = self.process(
                headline=item.get("headline", ""),
                content=item.get("content", ""),
                source=item.get("source", "unknown"),
                published_at=item.get("published_at"),
                url=item.get("url"),
                author=item.get("author"),
                metadata=item.get("metadata"),
            )
            if event:
                results.append(event)

        return results

    def _generate_event_id(self, headline: str, source: str) -> str:
        """Generate unique event ID."""
        content = f"{source}:{headline}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _is_duplicate(self, headline: str, content: str) -> bool:
        """Check if news item is a duplicate."""
        # Clean up old entries
        cutoff = datetime.now() - timedelta(hours=self.config.dedup_window_hours)
        self._seen_hashes = {h: t for h, t in self._seen_hashes.items() if t > cutoff}

        # Generate hash
        text = f"{headline.lower().strip()} {content[:200].lower()}"
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash in self._seen_hashes:
            return True

        self._seen_hashes[text_hash] = datetime.now()
        return False

    def _classify_event_type(self, headline: str, content: str) -> NewsEventType:
        """Classify news event type based on content."""
        combined = f"{headline} {content[:1000]}".lower()

        # Check each event type
        type_scores: dict[NewsEventType, int] = {}

        for event_type, patterns in self._compiled_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.search(combined):
                    score += 1
            if score > 0:
                type_scores[event_type] = score

        if not type_scores:
            return NewsEventType.GENERAL

        # Return highest scoring type
        return max(type_scores.keys(), key=lambda t: type_scores[t])

    def _determine_urgency(
        self,
        headline: str,
        content: str,
        event_type: NewsEventType,
    ) -> NewsUrgency:
        """Determine news urgency level."""
        combined_lower = f"{headline} {content[:500]}".lower()

        # Check critical keywords
        if any(kw in combined_lower for kw in self.CRITICAL_KEYWORDS):
            return NewsUrgency.CRITICAL

        # Check high urgency keywords
        if any(kw in combined_lower for kw in self.HIGH_URGENCY_KEYWORDS):
            return NewsUrgency.HIGH

        # Event type based urgency
        high_urgency_types = {
            NewsEventType.FDA_APPROVAL,
            NewsEventType.FDA_REJECTION,
            NewsEventType.MERGER_ACQUISITION,
            NewsEventType.FED_DECISION,
            NewsEventType.MANAGEMENT_CHANGE,
        }

        if event_type in high_urgency_types:
            return NewsUrgency.HIGH

        medium_urgency_types = {
            NewsEventType.EARNINGS,
            NewsEventType.GUIDANCE,
            NewsEventType.ANALYST_UPGRADE,
            NewsEventType.ANALYST_DOWNGRADE,
            NewsEventType.LAWSUIT,
        }

        if event_type in medium_urgency_types:
            return NewsUrgency.MEDIUM

        return NewsUrgency.LOW

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        avg_time = self._total_processing_time_ms / self._processed_count if self._processed_count > 0 else 0

        return {
            "processed_count": self._processed_count,
            "total_processing_time_ms": self._total_processing_time_ms,
            "average_processing_time_ms": avg_time,
            "dedup_cache_size": len(self._seen_hashes),
            "target_latency_ms": self.config.target_latency_ms,
            "latency_within_target": avg_time <= self.config.target_latency_ms,
        }

    def filter_by_ticker(
        self,
        events: list[ProcessedNewsEvent],
        tickers: set[str],
    ) -> list[ProcessedNewsEvent]:
        """Filter events to those mentioning specific tickers."""
        return [e for e in events if any(t in tickers for t in e.tickers)]

    def filter_by_urgency(
        self,
        events: list[ProcessedNewsEvent],
        min_urgency: NewsUrgency,
    ) -> list[ProcessedNewsEvent]:
        """Filter events by minimum urgency level."""
        urgency_order = [
            NewsUrgency.LOW,
            NewsUrgency.MEDIUM,
            NewsUrgency.HIGH,
            NewsUrgency.CRITICAL,
        ]
        min_index = urgency_order.index(min_urgency)

        return [e for e in events if urgency_order.index(e.urgency) >= min_index]

    def filter_by_event_type(
        self,
        events: list[ProcessedNewsEvent],
        event_types: set[NewsEventType],
    ) -> list[ProcessedNewsEvent]:
        """Filter events by event type."""
        return [e for e in events if e.event_type in event_types]


def create_news_processor(
    config: NewsProcessorConfig | None = None,
    sentiment_analyzer: Any | None = None,
    event_callback: Callable[[ProcessedNewsEvent], None] | None = None,
) -> NewsProcessor:
    """
    Factory function to create a news processor.

    Args:
        config: Optional configuration
        sentiment_analyzer: Optional sentiment analyzer
        event_callback: Optional callback for events

    Returns:
        Configured NewsProcessor instance
    """
    return NewsProcessor(
        config=config,
        sentiment_analyzer=sentiment_analyzer,
        event_callback=event_callback,
    )
