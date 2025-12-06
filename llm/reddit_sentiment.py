"""
Reddit Sentiment Analyzer Module

Provides Reddit-specific sentiment analysis for trading signals.
Integrates with FinBERT and handles Reddit-specific language patterns.
"""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .base import Sentiment, SentimentResult
from .sentiment import create_sentiment_analyzer


logger = logging.getLogger(__name__)


class RedditSentimentSignal(Enum):
    """Reddit-derived trading signal."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class RedditSentimentConfig:
    """Configuration for Reddit sentiment analysis."""

    use_finbert: bool = True
    min_mentions_for_signal: int = 5
    min_authors_for_signal: int = 3
    bullish_threshold: float = 0.3
    bearish_threshold: float = -0.3
    strong_threshold: float = 0.6
    weight_by_upvotes: bool = True
    weight_by_author_karma: bool = True
    dd_post_weight: float = 2.0  # DD posts weighted higher
    filter_low_quality: bool = True
    min_content_length: int = 20


@dataclass
class ContentSentiment:
    """Sentiment for a single piece of content."""

    content_id: str
    content_type: str  # "post" or "comment"
    text: str
    sentiment: SentimentResult
    weight: float = 1.0
    is_dd: bool = False
    upvotes: int = 0
    author_karma: int = 0


@dataclass
class TickerSentiment:
    """Aggregated sentiment for a ticker."""

    ticker: str
    signal: RedditSentimentSignal
    avg_score: float
    std_score: float
    confidence: float
    content_count: int
    unique_authors: int
    bullish_pct: float
    bearish_pct: float
    neutral_pct: float
    total_upvotes: int
    weighted_score: float
    content_sentiments: list[ContentSentiment] = field(default_factory=list)
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    subreddit_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "signal": self.signal.value,
            "avg_score": self.avg_score,
            "std_score": self.std_score,
            "confidence": self.confidence,
            "content_count": self.content_count,
            "unique_authors": self.unique_authors,
            "bullish_pct": self.bullish_pct,
            "bearish_pct": self.bearish_pct,
            "neutral_pct": self.neutral_pct,
            "total_upvotes": self.total_upvotes,
            "weighted_score": self.weighted_score,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "subreddit_breakdown": self.subreddit_breakdown,
        }


@dataclass
class RedditSentimentSummary:
    """Summary of Reddit sentiment across multiple tickers."""

    ticker_sentiments: dict[str, TickerSentiment]
    overall_market_sentiment: float
    most_bullish: list[str]
    most_bearish: list[str]
    trending_tickers: list[str]
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    total_posts_analyzed: int = 0
    total_comments_analyzed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker_sentiments": {t: s.to_dict() for t, s in self.ticker_sentiments.items()},
            "overall_market_sentiment": self.overall_market_sentiment,
            "most_bullish": self.most_bullish,
            "most_bearish": self.most_bearish,
            "trending_tickers": self.trending_tickers,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "total_posts_analyzed": self.total_posts_analyzed,
            "total_comments_analyzed": self.total_comments_analyzed,
        }


class RedditSentimentAnalyzer:
    """
    Analyzes sentiment from Reddit content for trading signals.

    Uses FinBERT for financial sentiment analysis with Reddit-specific
    preprocessing and weighting.
    """

    # Reddit-specific bullish/bearish terms not in standard FinBERT
    REDDIT_BULLISH_TERMS = {
        "moon",
        "mooning",
        "rocket",
        "tendies",
        "diamond hands",
        "diamondhands",
        "bullish",
        "calls",
        "long",
        "buy the dip",
        "btd",
        "undervalued",
        "squeeze",
        "gamma squeeze",
        "short squeeze",
        "apes",
        "hold",
        "hodl",
        "to the moon",
        "🚀",
        "💎",
        "🙌",
        "🦍",
        "📈",
    }

    REDDIT_BEARISH_TERMS = {
        "puts",
        "short",
        "bear",
        "bearish",
        "overvalued",
        "crash",
        "dump",
        "sell",
        "paper hands",
        "paperhands",
        "rug pull",
        "dead cat bounce",
        "bagholder",
        "bagholders",
        "📉",
        "🐻",
    }

    # Terms indicating high uncertainty/gambling
    UNCERTAINTY_TERMS = {
        "yolo",
        "gambling",
        "casino",
        "lottery",
        "pray",
        "hope",
        "fingers crossed",
        "who knows",
        "maybe",
        "could go either way",
    }

    def __init__(self, config: RedditSentimentConfig | None = None):
        """
        Initialize Reddit sentiment analyzer.

        Args:
            config: Analyzer configuration
        """
        self.config = config or RedditSentimentConfig()
        self._base_analyzer = None

    def _ensure_initialized(self) -> None:
        """Lazy initialization of base sentiment analyzer."""
        if self._base_analyzer is None:
            self._base_analyzer = create_sentiment_analyzer(use_finbert=self.config.use_finbert)

    def _preprocess_reddit_text(self, text: str) -> str:
        """
        Preprocess Reddit text for sentiment analysis.

        Args:
            text: Raw Reddit text

        Returns:
            Preprocessed text
        """
        # Convert to lowercase for matching
        lower_text = text.lower()

        # Check length threshold
        if len(text) < self.config.min_content_length:
            return ""

        # Remove common Reddit noise
        import re

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove Reddit-specific formatting
        text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # Links
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&lt;", "<", text)

        # Preserve emojis for Reddit-specific sentiment
        # but remove excessive repetition
        text = re.sub(r"(.)\1{4,}", r"\1\1\1", text)

        return text.strip()

    def _calculate_reddit_modifier(self, text: str) -> float:
        """
        Calculate sentiment modifier based on Reddit-specific terms.

        Args:
            text: Text to analyze

        Returns:
            Modifier value (-1 to 1)
        """
        lower_text = text.lower()

        bullish_count = sum(1 for term in self.REDDIT_BULLISH_TERMS if term in lower_text)
        bearish_count = sum(1 for term in self.REDDIT_BEARISH_TERMS if term in lower_text)
        uncertainty_count = sum(1 for term in self.UNCERTAINTY_TERMS if term in lower_text)

        if bullish_count + bearish_count == 0:
            return 0.0

        # Net sentiment
        net = bullish_count - bearish_count
        total = bullish_count + bearish_count

        modifier = net / total

        # Reduce confidence if high uncertainty
        if uncertainty_count > 0:
            modifier *= 0.7

        return modifier * 0.3  # Max ±0.3 modifier

    def _calculate_weight(
        self,
        upvotes: int,
        author_karma: int,
        is_dd: bool,
    ) -> float:
        """
        Calculate weight for content based on quality signals.

        Args:
            upvotes: Content upvotes/score
            author_karma: Author's total karma
            is_dd: Whether this is a Due Diligence post

        Returns:
            Weight multiplier (>= 1.0)
        """
        weight = 1.0

        if self.config.weight_by_upvotes:
            # Logarithmic scaling for upvotes
            if upvotes > 0:
                import math

                weight *= 1 + math.log10(upvotes + 1) * 0.2

        if self.config.weight_by_author_karma:
            # Logarithmic scaling for karma
            if author_karma > 1000:
                import math

                weight *= 1 + math.log10(author_karma / 1000) * 0.1

        if is_dd:
            weight *= self.config.dd_post_weight

        return weight

    def analyze_content(
        self,
        content_id: str,
        content_type: str,
        text: str,
        upvotes: int = 0,
        author_karma: int = 0,
        is_dd: bool = False,
    ) -> ContentSentiment | None:
        """
        Analyze sentiment of a single piece of content.

        Args:
            content_id: Unique identifier for content
            content_type: "post" or "comment"
            text: Content text
            upvotes: Content score
            author_karma: Author karma
            is_dd: Whether this is a DD post

        Returns:
            ContentSentiment if analyzable, None otherwise
        """
        self._ensure_initialized()

        # Preprocess text
        processed_text = self._preprocess_reddit_text(text)
        if not processed_text:
            return None

        # Get base sentiment
        base_sentiment = self._base_analyzer.analyze(processed_text)

        # Apply Reddit modifier
        reddit_modifier = self._calculate_reddit_modifier(text)
        adjusted_score = base_sentiment.score + reddit_modifier
        adjusted_score = max(-1, min(1, adjusted_score))  # Clamp to [-1, 1]

        # Adjust sentiment enum based on new score
        if adjusted_score > 0.6:
            adjusted_sentiment = Sentiment.VERY_BULLISH
        elif adjusted_score > 0.2:
            adjusted_sentiment = Sentiment.BULLISH
        elif adjusted_score < -0.6:
            adjusted_sentiment = Sentiment.VERY_BEARISH
        elif adjusted_score < -0.2:
            adjusted_sentiment = Sentiment.BEARISH
        else:
            adjusted_sentiment = Sentiment.NEUTRAL

        # Create adjusted result
        adjusted_result = SentimentResult(
            sentiment=adjusted_sentiment,
            confidence=base_sentiment.confidence,
            score=adjusted_score,
            provider=f"reddit_{base_sentiment.provider}",
            processing_time_ms=base_sentiment.processing_time_ms,
        )

        # Calculate weight
        weight = self._calculate_weight(upvotes, author_karma, is_dd)

        return ContentSentiment(
            content_id=content_id,
            content_type=content_type,
            text=text[:200],  # Store truncated text
            sentiment=adjusted_result,
            weight=weight,
            is_dd=is_dd,
            upvotes=upvotes,
            author_karma=author_karma,
        )

    def analyze_ticker(
        self,
        ticker: str,
        posts: list[dict[str, Any]],
        comments: list[dict[str, Any]],
    ) -> TickerSentiment:
        """
        Analyze aggregated sentiment for a ticker.

        Args:
            ticker: Stock ticker
            posts: List of post dictionaries with content
            comments: List of comment dictionaries with content

        Returns:
            Aggregated TickerSentiment
        """
        content_sentiments: list[ContentSentiment] = []
        unique_authors: set = set()
        subreddit_scores: dict[str, list[float]] = {}

        # Analyze posts
        for post in posts:
            sentiment = self.analyze_content(
                content_id=post.get("post_id", ""),
                content_type="post",
                text=f"{post.get('title', '')} {post.get('content', '')}",
                upvotes=post.get("upvotes", 0),
                author_karma=post.get("author", {}).get("karma", 0),
                is_dd=post.get("is_dd", False),
            )
            if sentiment:
                content_sentiments.append(sentiment)
                unique_authors.add(post.get("author", {}).get("name", ""))

                # Track by subreddit
                subreddit = post.get("subreddit", "unknown")
                if subreddit not in subreddit_scores:
                    subreddit_scores[subreddit] = []
                subreddit_scores[subreddit].append(sentiment.sentiment.score)

        # Analyze comments
        for comment in comments:
            sentiment = self.analyze_content(
                content_id=comment.get("comment_id", ""),
                content_type="comment",
                text=comment.get("content", ""),
                upvotes=comment.get("score", 0),
                author_karma=comment.get("author", {}).get("karma", 0),
                is_dd=False,
            )
            if sentiment:
                content_sentiments.append(sentiment)
                unique_authors.add(comment.get("author", {}).get("name", ""))

                subreddit = comment.get("subreddit", "unknown")
                if subreddit not in subreddit_scores:
                    subreddit_scores[subreddit] = []
                subreddit_scores[subreddit].append(sentiment.sentiment.score)

        # Calculate aggregates
        if not content_sentiments:
            return TickerSentiment(
                ticker=ticker,
                signal=RedditSentimentSignal.NEUTRAL,
                avg_score=0.0,
                std_score=0.0,
                confidence=0.0,
                content_count=0,
                unique_authors=0,
                bullish_pct=0.0,
                bearish_pct=0.0,
                neutral_pct=1.0,
                total_upvotes=0,
                weighted_score=0.0,
            )

        scores = [cs.sentiment.score for cs in content_sentiments]
        weights = [cs.weight for cs in content_sentiments]
        total_weight = sum(weights)

        # Weighted average score
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        # Simple average
        avg_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0

        # Sentiment distribution
        bullish_count = sum(1 for s in scores if s > self.config.bullish_threshold)
        bearish_count = sum(1 for s in scores if s < self.config.bearish_threshold)
        neutral_count = len(scores) - bullish_count - bearish_count

        bullish_pct = bullish_count / len(scores)
        bearish_pct = bearish_count / len(scores)
        neutral_pct = neutral_count / len(scores)

        # Determine signal
        signal = self._determine_signal(weighted_score, bullish_pct, bearish_pct, len(unique_authors))

        # Calculate confidence
        confidence = self._calculate_confidence(
            content_count=len(content_sentiments),
            unique_authors=len(unique_authors),
            std_score=std_score,
        )

        # Subreddit breakdown
        subreddit_breakdown = {sub: statistics.mean(scores_list) for sub, scores_list in subreddit_scores.items()}

        return TickerSentiment(
            ticker=ticker,
            signal=signal,
            avg_score=avg_score,
            std_score=std_score,
            confidence=confidence,
            content_count=len(content_sentiments),
            unique_authors=len(unique_authors),
            bullish_pct=bullish_pct,
            bearish_pct=bearish_pct,
            neutral_pct=neutral_pct,
            total_upvotes=sum(cs.upvotes for cs in content_sentiments),
            weighted_score=weighted_score,
            content_sentiments=content_sentiments,
            subreddit_breakdown=subreddit_breakdown,
        )

    def _determine_signal(
        self,
        weighted_score: float,
        bullish_pct: float,
        bearish_pct: float,
        unique_authors: int,
    ) -> RedditSentimentSignal:
        """
        Determine trading signal from sentiment data.

        Args:
            weighted_score: Weighted sentiment score
            bullish_pct: Percentage of bullish content
            bearish_pct: Percentage of bearish content
            unique_authors: Number of unique authors

        Returns:
            Trading signal
        """
        # Need minimum participation for strong signals
        min_authors = self.config.min_authors_for_signal

        if unique_authors < min_authors:
            return RedditSentimentSignal.NEUTRAL

        # Strong signals require both high score AND consensus
        if weighted_score > self.config.strong_threshold and bullish_pct > 0.7:
            return RedditSentimentSignal.STRONG_BUY
        elif weighted_score < -self.config.strong_threshold and bearish_pct > 0.7:
            return RedditSentimentSignal.STRONG_SELL
        elif weighted_score > self.config.bullish_threshold:
            return RedditSentimentSignal.BUY
        elif weighted_score < self.config.bearish_threshold:
            return RedditSentimentSignal.SELL
        else:
            return RedditSentimentSignal.NEUTRAL

    def _calculate_confidence(
        self,
        content_count: int,
        unique_authors: int,
        std_score: float,
    ) -> float:
        """
        Calculate confidence in sentiment analysis.

        Args:
            content_count: Number of content pieces analyzed
            unique_authors: Number of unique authors
            std_score: Standard deviation of scores

        Returns:
            Confidence score (0-1)
        """
        # Volume factor
        volume_conf = min(1.0, content_count / 20)

        # Diversity factor
        diversity_conf = min(1.0, unique_authors / 10)

        # Consensus factor (lower std = higher consensus = higher confidence)
        consensus_conf = max(0, 1 - std_score)

        # Combined confidence
        confidence = volume_conf * 0.3 + diversity_conf * 0.4 + consensus_conf * 0.3

        return min(1.0, confidence)

    def analyze_batch(
        self,
        mention_data: dict[str, dict[str, Any]],
    ) -> RedditSentimentSummary:
        """
        Analyze sentiment for multiple tickers.

        Args:
            mention_data: Dictionary mapping tickers to their mention data
                          Each entry should have 'posts' and 'comments' keys

        Returns:
            Summary of sentiment across all tickers
        """
        ticker_sentiments: dict[str, TickerSentiment] = {}
        total_posts = 0
        total_comments = 0

        for ticker, data in mention_data.items():
            posts = data.get("posts", [])
            comments = data.get("comments", [])

            total_posts += len(posts)
            total_comments += len(comments)

            sentiment = self.analyze_ticker(ticker, posts, comments)
            ticker_sentiments[ticker] = sentiment

        # Calculate overall market sentiment
        if ticker_sentiments:
            all_scores = [ts.weighted_score for ts in ticker_sentiments.values()]
            overall_sentiment = statistics.mean(all_scores) if all_scores else 0.0
        else:
            overall_sentiment = 0.0

        # Find most bullish/bearish
        sorted_by_score = sorted(
            ticker_sentiments.items(),
            key=lambda x: x[1].weighted_score,
            reverse=True,
        )

        most_bullish = [t for t, s in sorted_by_score[:5] if s.weighted_score > 0]
        most_bearish = [t for t, s in sorted_by_score[-5:] if s.weighted_score < 0]

        # Find trending (high mention count)
        trending = sorted(
            ticker_sentiments.items(),
            key=lambda x: x[1].content_count,
            reverse=True,
        )[:10]
        trending_tickers = [t for t, _ in trending]

        return RedditSentimentSummary(
            ticker_sentiments=ticker_sentiments,
            overall_market_sentiment=overall_sentiment,
            most_bullish=most_bullish,
            most_bearish=most_bearish,
            trending_tickers=trending_tickers,
            total_posts_analyzed=total_posts,
            total_comments_analyzed=total_comments,
        )


def create_reddit_sentiment_analyzer(
    config: RedditSentimentConfig | None = None,
) -> RedditSentimentAnalyzer:
    """
    Create Reddit sentiment analyzer.

    Args:
        config: Analyzer configuration

    Returns:
        Configured RedditSentimentAnalyzer instance
    """
    return RedditSentimentAnalyzer(config)


__all__ = [
    "ContentSentiment",
    "RedditSentimentAnalyzer",
    "RedditSentimentConfig",
    "RedditSentimentSignal",
    "RedditSentimentSummary",
    "TickerSentiment",
    "create_reddit_sentiment_analyzer",
]
