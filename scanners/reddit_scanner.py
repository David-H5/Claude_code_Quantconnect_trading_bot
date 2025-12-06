"""
Reddit Sentiment Scanner Module

Scans Reddit for stock mentions and sentiment in financial subreddits.
Uses PRAW for Reddit API access and integrates with sentiment analysis.
"""

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)

# Check for PRAW availability
try:
    import praw
    from praw.models import Comment, Submission

    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    praw = None
    Comment = None
    Submission = None


class ContentType(Enum):
    """Type of Reddit content."""

    POST = "post"
    COMMENT = "comment"


class SubredditCategory(Enum):
    """Categories of financial subreddits."""

    WALLSTREETBETS = "wallstreetbets"
    OPTIONS = "options"
    STOCKS = "stocks"
    INVESTING = "investing"
    STOCKMARKET = "stockmarket"
    PENNYSTOCKS = "pennystocks"
    THETAGANG = "thetagang"
    DAYTRADING = "daytrading"


# Default subreddits to monitor
DEFAULT_SUBREDDITS = [
    "wallstreetbets",
    "options",
    "stocks",
    "investing",
    "stockmarket",
    "thetagang",
]


@dataclass
class RedditScannerConfig:
    """Reddit scanner configuration."""

    enabled: bool = True
    client_id: str = ""
    client_secret: str = ""
    user_agent: str = "TradingBot/1.0 (Options Scanner)"
    subreddits: list[str] = field(default_factory=lambda: DEFAULT_SUBREDDITS.copy())
    scan_interval_seconds: int = 60
    max_posts_per_scan: int = 100
    max_comments_per_post: int = 50
    min_upvotes: int = 10
    min_comment_score: int = 5
    lookback_hours: int = 24
    # Filtering
    min_account_age_days: int = 30
    min_karma: int = 100
    filter_bots: bool = True
    # Rate limiting
    requests_per_minute: int = 30


@dataclass
class RedditAuthor:
    """Reddit author information for bot detection."""

    name: str
    account_age_days: int
    karma: int
    is_verified: bool = False
    is_mod: bool = False
    is_suspected_bot: bool = False
    bot_probability: float = 0.0


@dataclass
class RedditPost:
    """Reddit post with metadata."""

    post_id: str
    subreddit: str
    title: str
    content: str
    author: RedditAuthor
    created_utc: datetime
    upvotes: int
    upvote_ratio: float
    num_comments: int
    url: str
    flair: str | None = None
    is_dd: bool = False  # Due Diligence post
    mentioned_tickers: list[str] = field(default_factory=list)
    content_type: ContentType = ContentType.POST

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "post_id": self.post_id,
            "subreddit": self.subreddit,
            "title": self.title,
            "content": self.content[:500],  # Truncate for storage
            "author": {
                "name": self.author.name,
                "account_age_days": self.author.account_age_days,
                "karma": self.author.karma,
                "is_suspected_bot": self.author.is_suspected_bot,
                "bot_probability": self.author.bot_probability,
            },
            "created_utc": self.created_utc.isoformat(),
            "upvotes": self.upvotes,
            "upvote_ratio": self.upvote_ratio,
            "num_comments": self.num_comments,
            "url": self.url,
            "flair": self.flair,
            "is_dd": self.is_dd,
            "mentioned_tickers": self.mentioned_tickers,
            "content_type": self.content_type.value,
        }


@dataclass
class RedditComment:
    """Reddit comment with metadata."""

    comment_id: str
    post_id: str
    subreddit: str
    content: str
    author: RedditAuthor
    created_utc: datetime
    score: int
    parent_id: str
    depth: int
    mentioned_tickers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "comment_id": self.comment_id,
            "post_id": self.post_id,
            "subreddit": self.subreddit,
            "content": self.content[:500],
            "author": {
                "name": self.author.name,
                "account_age_days": self.author.account_age_days,
                "karma": self.author.karma,
                "is_suspected_bot": self.author.is_suspected_bot,
            },
            "created_utc": self.created_utc.isoformat(),
            "score": self.score,
            "mentioned_tickers": self.mentioned_tickers,
        }


@dataclass
class TickerMention:
    """Aggregated ticker mention data."""

    ticker: str
    mention_count: int
    unique_authors: int
    total_upvotes: int
    avg_sentiment: float
    sentiment_std: float
    posts: list[RedditPost] = field(default_factory=list)
    comments: list[RedditComment] = field(default_factory=list)
    subreddit_distribution: dict[str, int] = field(default_factory=dict)
    hourly_distribution: dict[int, int] = field(default_factory=dict)
    first_mention: datetime | None = None
    last_mention: datetime | None = None
    is_trending: bool = False
    trend_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "mention_count": self.mention_count,
            "unique_authors": self.unique_authors,
            "total_upvotes": self.total_upvotes,
            "avg_sentiment": self.avg_sentiment,
            "sentiment_std": self.sentiment_std,
            "post_count": len(self.posts),
            "comment_count": len(self.comments),
            "subreddit_distribution": self.subreddit_distribution,
            "hourly_distribution": self.hourly_distribution,
            "first_mention": self.first_mention.isoformat() if self.first_mention else None,
            "last_mention": self.last_mention.isoformat() if self.last_mention else None,
            "is_trending": self.is_trending,
            "trend_score": self.trend_score,
        }


@dataclass
class RedditSentimentAlert:
    """Alert generated from Reddit sentiment analysis."""

    ticker: str
    alert_type: str  # "surge", "trending", "sentiment_shift", "volume_spike"
    mention_data: TickerMention
    sentiment_score: float
    confidence: float
    subreddit_sources: list[str]
    sample_posts: list[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "alert_type": self.alert_type,
            "mention_count": self.mention_data.mention_count,
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "subreddit_sources": self.subreddit_sources,
            "sample_posts": self.sample_posts[:3],
            "is_trending": self.mention_data.is_trending,
            "trend_score": self.mention_data.trend_score,
            "timestamp": self.timestamp.isoformat(),
        }


class RedditScanner:
    """
    Scans Reddit for stock mentions and sentiment.

    Monitors financial subreddits for ticker mentions, analyzes sentiment,
    and generates alerts for unusual activity.
    """

    # Common stock ticker pattern (1-5 uppercase letters, optionally preceded by $)
    TICKER_PATTERN = re.compile(r"\$?([A-Z]{1,5})\b")

    # Common words that look like tickers but aren't
    TICKER_BLACKLIST = {
        "I",
        "A",
        "AM",
        "PM",
        "US",
        "UK",
        "EU",
        "CEO",
        "CFO",
        "DD",
        "OP",
        "EPS",
        "GDP",
        "IPO",
        "SEC",
        "FBI",
        "CIA",
        "IRS",
        "ATH",
        "ATL",
        "YOLO",
        "FOMO",
        "IMO",
        "TBH",
        "FYI",
        "LOL",
        "LMAO",
        "WTF",
        "OMG",
        "ETF",
        "IRA",
        "USA",
        "NYSE",
        "WSB",
        "RH",
        "TD",
        "API",
        "AI",
        "ML",
        "EOD",
        "EOW",
        "ITM",
        "OTM",
        "ATM",
        "DTE",
        "IV",
        "HV",
        "RSI",
        "MACD",
        "VWAP",
        "SMA",
        "EMA",
        "EDIT",
        "POST",
        "COMMENT",
        "LINK",
        "TEXT",
        "THE",
        "AND",
        "FOR",
        "NOT",
        "ARE",
        "BUT",
        "NEW",
        "OLD",
        "ALL",
        "ANY",
        "CAN",
        "HAD",
        "HER",
        "WAS",
        "ONE",
        "OUR",
        "OUT",
        "YOU",
        "HIS",
        "HAS",
        "ITS",
        "TWO",
        "WAY",
        "WHO",
        "HOW",
        "GET",
        "GOT",
    }

    # Known valid tickers that might be filtered
    TICKER_WHITELIST = {
        "AMD",
        "NVDA",
        "TSLA",
        "AAPL",
        "MSFT",
        "GOOG",
        "GOOGL",
        "AMZN",
        "META",
        "NFLX",
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "GME",
        "AMC",
        "BB",
        "PLTR",
        "SOFI",
        "HOOD",
        "RIVN",
        "LCID",
        "NIO",
        "BABA",
        "JD",
    }

    def __init__(
        self,
        config: RedditScannerConfig,
        alert_callback: Callable[[RedditSentimentAlert], None] | None = None,
    ):
        """
        Initialize Reddit scanner.

        Args:
            config: Scanner configuration with Reddit API credentials
            alert_callback: Optional callback for alerts
        """
        self.config = config
        self.alert_callback = alert_callback

        self._reddit: Any | None = None
        self._initialized = False
        self._last_scan: dict[str, datetime] = {}
        self._mention_history: dict[str, list[TickerMention]] = {}
        self._rate_limiter = RateLimiter(config.requests_per_minute)

    def _ensure_initialized(self) -> bool:
        """
        Lazy initialization of Reddit API client.

        Returns:
            True if initialized successfully, False otherwise
        """
        if self._initialized:
            return True

        if not PRAW_AVAILABLE:
            logger.warning("PRAW not available. Install with: pip install praw")
            return False

        if not self.config.client_id or not self.config.client_secret:
            logger.warning("Reddit API credentials not configured. " "Set client_id and client_secret in config.")
            return False

        try:
            self._reddit = praw.Reddit(
                client_id=self.config.client_id,
                client_secret=self.config.client_secret,
                user_agent=self.config.user_agent,
            )
            # Test the connection
            self._reddit.user.me()
            self._initialized = True
            logger.info("Reddit API initialized successfully")
            return True
        except Exception as e:
            logger.error("Failed to initialize Reddit API: %s", e)
            return False

    def _extract_author_info(self, author: Any) -> RedditAuthor:
        """Extract author information from Reddit author object."""
        if author is None or str(author) == "[deleted]":
            return RedditAuthor(
                name="[deleted]",
                account_age_days=0,
                karma=0,
                is_suspected_bot=True,
                bot_probability=0.9,
            )

        try:
            # Get account creation time
            created = datetime.utcfromtimestamp(author.created_utc)
            age_days = (datetime.utcnow() - created).days

            return RedditAuthor(
                name=str(author.name),
                account_age_days=age_days,
                karma=author.link_karma + author.comment_karma,
                is_verified=getattr(author, "verified", False),
                is_mod=getattr(author, "is_mod", False),
            )
        except Exception as e:
            logger.debug("Could not extract author info: %s", e)
            return RedditAuthor(
                name=str(author) if author else "[unknown]",
                account_age_days=0,
                karma=0,
            )

    def _extract_tickers(self, text: str) -> list[str]:
        """
        Extract stock tickers from text.

        Args:
            text: Text to extract tickers from

        Returns:
            List of unique tickers found
        """
        matches = self.TICKER_PATTERN.findall(text.upper())

        # Filter and validate tickers
        tickers = []
        seen = set()

        for match in matches:
            if match in seen:
                continue
            seen.add(match)

            # Skip blacklisted words
            if match in self.TICKER_BLACKLIST:
                continue

            # Include whitelisted tickers
            if match in self.TICKER_WHITELIST:
                tickers.append(match)
                continue

            # For other matches, apply stricter validation
            if len(match) >= 2:  # At least 2 characters
                tickers.append(match)

        return tickers

    def _is_dd_post(self, submission: Any) -> bool:
        """Check if post is a Due Diligence post."""
        flair = getattr(submission, "link_flair_text", "") or ""
        title = submission.title.lower()

        dd_indicators = ["dd", "due diligence", "analysis", "research"]
        return any(ind in flair.lower() or ind in title for ind in dd_indicators)

    def _process_submission(self, submission: Any) -> RedditPost | None:
        """Process a Reddit submission into a RedditPost object."""
        try:
            self._rate_limiter.wait()

            # Filter by age
            created = datetime.utcfromtimestamp(submission.created_utc)
            age_hours = (datetime.utcnow() - created).total_seconds() / 3600

            if age_hours > self.config.lookback_hours:
                return None

            # Filter by score
            if submission.score < self.config.min_upvotes:
                return None

            # Get content
            content = submission.selftext if hasattr(submission, "selftext") else ""
            full_text = f"{submission.title} {content}"

            # Extract tickers
            tickers = self._extract_tickers(full_text)
            if not tickers:
                return None

            # Get author info
            author = self._extract_author_info(submission.author)

            # Filter by author credibility
            if self.config.filter_bots:
                if author.account_age_days < self.config.min_account_age_days:
                    return None
                if author.karma < self.config.min_karma:
                    return None

            return RedditPost(
                post_id=submission.id,
                subreddit=str(submission.subreddit),
                title=submission.title,
                content=content[:2000],  # Limit content length
                author=author,
                created_utc=created,
                upvotes=submission.score,
                upvote_ratio=submission.upvote_ratio,
                num_comments=submission.num_comments,
                url=f"https://reddit.com{submission.permalink}",
                flair=getattr(submission, "link_flair_text", None),
                is_dd=self._is_dd_post(submission),
                mentioned_tickers=tickers,
            )
        except Exception as e:
            logger.debug("Error processing submission: %s", e)
            return None

    def _process_comment(self, comment: Any, post_id: str, subreddit: str) -> RedditComment | None:
        """Process a Reddit comment into a RedditComment object."""
        try:
            # Filter by score
            if comment.score < self.config.min_comment_score:
                return None

            # Filter by age
            created = datetime.utcfromtimestamp(comment.created_utc)
            age_hours = (datetime.utcnow() - created).total_seconds() / 3600

            if age_hours > self.config.lookback_hours:
                return None

            # Extract tickers
            tickers = self._extract_tickers(comment.body)
            if not tickers:
                return None

            # Get author info
            author = self._extract_author_info(comment.author)

            # Filter by author credibility
            if self.config.filter_bots:
                if author.account_age_days < self.config.min_account_age_days:
                    return None
                if author.karma < self.config.min_karma:
                    return None

            return RedditComment(
                comment_id=comment.id,
                post_id=post_id,
                subreddit=subreddit,
                content=comment.body[:2000],
                author=author,
                created_utc=created,
                score=comment.score,
                parent_id=comment.parent_id,
                depth=comment.depth if hasattr(comment, "depth") else 0,
                mentioned_tickers=tickers,
            )
        except Exception as e:
            logger.debug("Error processing comment: %s", e)
            return None

    def scan_subreddit(
        self,
        subreddit_name: str,
        limit: int | None = None,
        include_comments: bool = True,
    ) -> tuple[list[RedditPost], list[RedditComment]]:
        """
        Scan a subreddit for stock mentions.

        Args:
            subreddit_name: Name of subreddit to scan
            limit: Maximum posts to scan (default: config.max_posts_per_scan)
            include_comments: Whether to include comments

        Returns:
            Tuple of (posts, comments) found
        """
        if not self._ensure_initialized():
            return [], []

        limit = limit or self.config.max_posts_per_scan
        posts: list[RedditPost] = []
        comments: list[RedditComment] = []

        try:
            subreddit = self._reddit.subreddit(subreddit_name)

            # Scan hot and new posts
            for submission in subreddit.hot(limit=limit // 2):
                post = self._process_submission(submission)
                if post:
                    posts.append(post)

                    # Get comments if requested
                    if include_comments:
                        self._rate_limiter.wait()
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments.list()[: self.config.max_comments_per_post]:
                            c = self._process_comment(comment, submission.id, subreddit_name)
                            if c:
                                comments.append(c)

            for submission in subreddit.new(limit=limit // 2):
                post = self._process_submission(submission)
                if post:
                    posts.append(post)

            logger.info(
                "Scanned r/%s: %d posts, %d comments",
                subreddit_name,
                len(posts),
                len(comments),
            )

        except Exception as e:
            logger.error("Error scanning r/%s: %s", subreddit_name, e)

        return posts, comments

    def scan_all_subreddits(self, include_comments: bool = True) -> tuple[list[RedditPost], list[RedditComment]]:
        """
        Scan all configured subreddits.

        Args:
            include_comments: Whether to include comments

        Returns:
            Tuple of (all_posts, all_comments) found
        """
        all_posts: list[RedditPost] = []
        all_comments: list[RedditComment] = []

        for subreddit in self.config.subreddits:
            posts, comments = self.scan_subreddit(subreddit, include_comments=include_comments)
            all_posts.extend(posts)
            all_comments.extend(comments)

        return all_posts, all_comments

    def aggregate_mentions(
        self,
        posts: list[RedditPost],
        comments: list[RedditComment],
    ) -> dict[str, TickerMention]:
        """
        Aggregate ticker mentions from posts and comments.

        Args:
            posts: List of Reddit posts
            comments: List of Reddit comments

        Returns:
            Dictionary mapping tickers to aggregated mention data
        """
        mentions: dict[str, TickerMention] = {}
        authors_by_ticker: dict[str, set[str]] = {}

        # Process posts
        for post in posts:
            for ticker in post.mentioned_tickers:
                if ticker not in mentions:
                    mentions[ticker] = TickerMention(
                        ticker=ticker,
                        mention_count=0,
                        unique_authors=0,
                        total_upvotes=0,
                        avg_sentiment=0.0,
                        sentiment_std=0.0,
                    )
                    authors_by_ticker[ticker] = set()

                m = mentions[ticker]
                m.mention_count += 1
                m.total_upvotes += post.upvotes
                m.posts.append(post)
                authors_by_ticker[ticker].add(post.author.name)

                # Track subreddit distribution
                if post.subreddit not in m.subreddit_distribution:
                    m.subreddit_distribution[post.subreddit] = 0
                m.subreddit_distribution[post.subreddit] += 1

                # Track hourly distribution
                hour = post.created_utc.hour
                if hour not in m.hourly_distribution:
                    m.hourly_distribution[hour] = 0
                m.hourly_distribution[hour] += 1

                # Track first/last mention
                if m.first_mention is None or post.created_utc < m.first_mention:
                    m.first_mention = post.created_utc
                if m.last_mention is None or post.created_utc > m.last_mention:
                    m.last_mention = post.created_utc

        # Process comments
        for comment in comments:
            for ticker in comment.mentioned_tickers:
                if ticker not in mentions:
                    mentions[ticker] = TickerMention(
                        ticker=ticker,
                        mention_count=0,
                        unique_authors=0,
                        total_upvotes=0,
                        avg_sentiment=0.0,
                        sentiment_std=0.0,
                    )
                    authors_by_ticker[ticker] = set()

                m = mentions[ticker]
                m.mention_count += 1
                m.total_upvotes += comment.score
                m.comments.append(comment)
                authors_by_ticker[ticker].add(comment.author.name)

                # Track subreddit distribution
                if comment.subreddit not in m.subreddit_distribution:
                    m.subreddit_distribution[comment.subreddit] = 0
                m.subreddit_distribution[comment.subreddit] += 1

                # Track hourly distribution
                hour = comment.created_utc.hour
                if hour not in m.hourly_distribution:
                    m.hourly_distribution[hour] = 0
                m.hourly_distribution[hour] += 1

        # Update unique author counts
        for ticker, authors in authors_by_ticker.items():
            mentions[ticker].unique_authors = len(authors)

        return mentions

    def calculate_trend_score(self, mention: TickerMention, baseline_mentions: int = 10) -> float:
        """
        Calculate trend score for a ticker.

        Score is based on:
        - Mention count vs baseline
        - Unique author diversity
        - Upvote engagement
        - Time recency

        Args:
            mention: Ticker mention data
            baseline_mentions: Expected baseline mentions

        Returns:
            Trend score (0-100)
        """
        # Mention volume factor (0-25)
        volume_score = min(25, (mention.mention_count / max(baseline_mentions, 1)) * 10)

        # Author diversity factor (0-25)
        diversity_ratio = mention.unique_authors / max(mention.mention_count, 1)
        diversity_score = min(25, diversity_ratio * 30)

        # Engagement factor (0-25)
        avg_upvotes = mention.total_upvotes / max(mention.mention_count, 1)
        engagement_score = min(25, avg_upvotes / 10)

        # Recency factor (0-25)
        if mention.last_mention:
            hours_ago = (datetime.utcnow() - mention.last_mention).total_seconds() / 3600
            recency_score = max(0, 25 - (hours_ago * 2))
        else:
            recency_score = 0

        total_score = volume_score + diversity_score + engagement_score + recency_score
        return min(100, total_score)

    def detect_trending(
        self,
        mentions: dict[str, TickerMention],
        min_mentions: int = 5,
        trend_threshold: float = 50.0,
    ) -> list[TickerMention]:
        """
        Detect trending tickers.

        Args:
            mentions: Aggregated mention data
            min_mentions: Minimum mentions to consider
            trend_threshold: Minimum trend score to be considered trending

        Returns:
            List of trending tickers sorted by trend score
        """
        trending = []

        for ticker, mention in mentions.items():
            if mention.mention_count < min_mentions:
                continue

            score = self.calculate_trend_score(mention)
            mention.trend_score = score
            mention.is_trending = score >= trend_threshold

            if mention.is_trending:
                trending.append(mention)

        # Sort by trend score
        trending.sort(key=lambda x: x.trend_score, reverse=True)
        return trending

    def generate_alerts(
        self,
        mentions: dict[str, TickerMention],
        min_mentions: int = 5,
    ) -> list[RedditSentimentAlert]:
        """
        Generate alerts from mention data.

        Args:
            mentions: Aggregated mention data
            min_mentions: Minimum mentions for alert

        Returns:
            List of alerts
        """
        alerts = []

        for ticker, mention in mentions.items():
            if mention.mention_count < min_mentions:
                continue

            # Calculate trend score
            trend_score = self.calculate_trend_score(mention)
            mention.trend_score = trend_score

            # Determine alert type
            if trend_score >= 70:
                alert_type = "surge"
            elif trend_score >= 50:
                alert_type = "trending"
            else:
                continue

            # Get sample posts
            sample_posts = [p.title for p in mention.posts[:3]]

            alert = RedditSentimentAlert(
                ticker=ticker,
                alert_type=alert_type,
                mention_data=mention,
                sentiment_score=mention.avg_sentiment,
                confidence=min(1.0, mention.unique_authors / 10),
                subreddit_sources=list(mention.subreddit_distribution.keys()),
                sample_posts=sample_posts,
            )

            alerts.append(alert)

            # Trigger callback
            if self.alert_callback:
                self.alert_callback(alert)

        return alerts

    def scan(self, include_comments: bool = True) -> tuple[dict[str, TickerMention], list[RedditSentimentAlert]]:
        """
        Run full scan across all configured subreddits.

        Args:
            include_comments: Whether to include comments

        Returns:
            Tuple of (mentions, alerts)
        """
        if not self.config.enabled:
            return {}, []

        logger.info("Starting Reddit scan across %d subreddits", len(self.config.subreddits))

        # Scan all subreddits
        posts, comments = self.scan_all_subreddits(include_comments)

        # Aggregate mentions
        mentions = self.aggregate_mentions(posts, comments)

        # Generate alerts
        alerts = self.generate_alerts(mentions)

        logger.info(
            "Reddit scan complete: %d tickers mentioned, %d alerts",
            len(mentions),
            len(alerts),
        )

        return mentions, alerts

    def get_ticker_mentions(self, ticker: str, lookback_hours: int = 24) -> TickerMention | None:
        """
        Get mention data for a specific ticker.

        Args:
            ticker: Stock ticker to search
            lookback_hours: Hours to look back

        Returns:
            TickerMention data if found
        """
        if not self._ensure_initialized():
            return None

        posts: list[RedditPost] = []
        comments: list[RedditComment] = []

        try:
            # Search for ticker across all subreddits
            search_query = f"${ticker} OR {ticker}"

            for subreddit_name in self.config.subreddits:
                self._rate_limiter.wait()
                subreddit = self._reddit.subreddit(subreddit_name)

                for submission in subreddit.search(search_query, time_filter="day", limit=50):
                    post = self._process_submission(submission)
                    if post and ticker in post.mentioned_tickers:
                        posts.append(post)

        except Exception as e:
            logger.error("Error searching for %s: %s", ticker, e)
            return None

        if not posts:
            return None

        mentions = self.aggregate_mentions(posts, comments)
        return mentions.get(ticker)


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, requests_per_minute: int):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self._last_request: float | None = None

    def wait(self) -> None:
        """Wait if necessary to respect rate limit."""
        if self._last_request is not None:
            elapsed = time.time() - self._last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

        self._last_request = time.time()


def create_reddit_scanner(
    config: RedditScannerConfig | None = None,
    alert_callback: Callable[[RedditSentimentAlert], None] | None = None,
) -> RedditScanner:
    """
    Create Reddit scanner from configuration.

    Args:
        config: Scanner configuration
        alert_callback: Optional callback for alerts

    Returns:
        Configured RedditScanner instance
    """
    if config is None:
        config = RedditScannerConfig()

    return RedditScanner(config, alert_callback=alert_callback)


__all__ = [
    "PRAW_AVAILABLE",
    "ContentType",
    "RedditAuthor",
    "RedditComment",
    "RedditPost",
    "RedditScanner",
    "RedditScannerConfig",
    "RedditSentimentAlert",
    "SubredditCategory",
    "TickerMention",
    "create_reddit_scanner",
]
