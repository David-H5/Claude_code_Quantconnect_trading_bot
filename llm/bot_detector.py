"""
Bot Detection Filter Module

Detects bot and shill accounts in social media content.
Provides filtering for Reddit sentiment analysis to improve signal quality.

Detection methods:
- Account age and karma analysis
- Posting pattern analysis
- Coordinated campaign detection
- Linguistic pattern analysis
"""

import logging
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class BotIndicator(Enum):
    """Types of bot indicators."""

    # Account-based
    NEW_ACCOUNT = "new_account"
    LOW_KARMA = "low_karma"
    KARMA_FARMING = "karma_farming"
    SUSPENDED_ACCOUNT = "suspended_account"

    # Posting pattern
    HIGH_FREQUENCY = "high_frequency"
    REPETITIVE_CONTENT = "repetitive_content"
    COPY_PASTE = "copy_paste"
    TEMPLATE_TEXT = "template_text"

    # Behavioral
    SINGLE_TOPIC = "single_topic"
    TIME_PATTERN = "time_pattern"  # Posts at unusual intervals
    NO_ENGAGEMENT = "no_engagement"  # Never replies to comments

    # Coordinated
    COORDINATED_TIMING = "coordinated_timing"
    SIMILAR_USERNAMES = "similar_usernames"
    IDENTICAL_CONTENT = "identical_content"

    # Linguistic
    SHILL_LANGUAGE = "shill_language"
    EXCESSIVE_EMOJIS = "excessive_emojis"
    PROMOTIONAL_LINKS = "promotional_links"


class BotConfidence(Enum):
    """Confidence level in bot detection."""

    DEFINITE = "definite"  # >90% confidence
    HIGH = "high"  # 70-90%
    MEDIUM = "medium"  # 50-70%
    LOW = "low"  # 30-50%
    UNLIKELY = "unlikely"  # <30%


@dataclass
class BotDetectorConfig:
    """Configuration for bot detection."""

    enabled: bool = True

    # Account thresholds
    min_account_age_days: int = 30
    min_karma: int = 100
    karma_to_age_ratio_threshold: float = 100.0  # Karma farming detection

    # Posting pattern thresholds
    max_posts_per_hour: int = 5
    max_posts_per_day: int = 50
    repetition_threshold: float = 0.7  # 70% similarity = repetitive

    # Time pattern detection
    min_time_between_posts_seconds: int = 60  # Too fast = bot
    suspicious_time_variance: float = 0.1  # Posts at exact intervals

    # Coordinated campaign thresholds
    coordination_time_window_minutes: int = 10
    min_coordinated_posts: int = 3
    username_similarity_threshold: float = 0.8

    # Linguistic patterns
    max_emoji_ratio: float = 0.15  # More than 15% emojis = suspicious
    min_unique_words_ratio: float = 0.3  # Low variety = template


@dataclass
class BotIndicatorResult:
    """Result of checking a single bot indicator."""

    indicator: BotIndicator
    triggered: bool
    score: float  # 0-1, contribution to bot probability
    details: str
    evidence: dict[str, Any] | None = None


@dataclass
class BotDetectionResult:
    """Complete bot detection result for an account/content."""

    author_name: str
    is_bot: bool
    confidence: BotConfidence
    bot_probability: float
    indicators: list[BotIndicatorResult] = field(default_factory=list)
    account_age_days: int = 0
    karma: int = 0
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "author_name": self.author_name,
            "is_bot": self.is_bot,
            "confidence": self.confidence.value,
            "bot_probability": self.bot_probability,
            "indicators": [
                {
                    "indicator": i.indicator.value,
                    "triggered": i.triggered,
                    "score": i.score,
                    "details": i.details,
                }
                for i in self.indicators
            ],
            "account_age_days": self.account_age_days,
            "karma": self.karma,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "recommendation": self.recommendation,
        }


@dataclass
class CoordinatedCampaign:
    """Detected coordinated campaign."""

    campaign_id: str
    ticker: str
    accounts: list[str]
    post_times: list[datetime]
    similar_content: list[str]
    confidence: float
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "campaign_id": self.campaign_id,
            "ticker": self.ticker,
            "num_accounts": len(self.accounts),
            "accounts": self.accounts[:10],  # Limit for privacy
            "time_window_minutes": (
                (max(self.post_times) - min(self.post_times)).total_seconds() / 60 if self.post_times else 0
            ),
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
        }


class BotDetector:
    """
    Detects bot and shill accounts in social media content.

    Uses multiple signals including account age, posting patterns,
    linguistic analysis, and coordinated campaign detection.
    """

    # Common shill/promotional language patterns
    SHILL_PATTERNS = [
        r"(?i)\bnot financial advice\b",
        r"(?i)\bthis is huge\b",
        r"(?i)\bget in now\b",
        r"(?i)\bbefore it.s too late\b",
        r"(?i)\btrust me\b",
        r"(?i)\beasy money\b",
        r"(?i)\bguaranteed\b",
        r"(?i)\brocket ship\b",
        r"(?i)\bonce in a lifetime\b",
        r"(?i)\bdon.t miss (out|this)\b",
        r"(?i)\bgonna explode\b",
        r"(?i)\bfree money\b",
    ]

    # Template-like phrases (copy-paste indicators)
    TEMPLATE_PATTERNS = [
        r"(?i)^i.m not a financial advisor",
        r"(?i)^do your own research",
        r"(?i)^this is not financial advice",
        r"(?i)position: \d+ shares",
        r"(?i)disclosure: (long|short|none)",
    ]

    def __init__(self, config: BotDetectorConfig | None = None):
        """
        Initialize bot detector.

        Args:
            config: Detection configuration
        """
        self.config = config or BotDetectorConfig()

        # Cache for coordinated campaign detection
        self._post_cache: dict[str, list[tuple[str, datetime, str]]] = defaultdict(list)
        self._detected_campaigns: list[CoordinatedCampaign] = []

        # Compile regex patterns
        self._shill_regexes = [re.compile(p) for p in self.SHILL_PATTERNS]
        self._template_regexes = [re.compile(p) for p in self.TEMPLATE_PATTERNS]

    def check_account_age(self, account_age_days: int) -> BotIndicatorResult:
        """Check if account is suspiciously new."""
        is_new = account_age_days < self.config.min_account_age_days

        if is_new:
            # Score based on how new (newer = more suspicious)
            score = min(1.0, 1 - (account_age_days / self.config.min_account_age_days))
        else:
            score = 0.0

        return BotIndicatorResult(
            indicator=BotIndicator.NEW_ACCOUNT,
            triggered=is_new,
            score=score,
            details=f"Account age: {account_age_days} days (threshold: {self.config.min_account_age_days})",
            evidence={"account_age_days": account_age_days},
        )

    def check_karma(self, karma: int, account_age_days: int) -> list[BotIndicatorResult]:
        """Check karma-based indicators."""
        results = []

        # Low karma check
        is_low_karma = karma < self.config.min_karma
        if is_low_karma:
            score = min(1.0, 1 - (karma / max(self.config.min_karma, 1)))
        else:
            score = 0.0

        results.append(
            BotIndicatorResult(
                indicator=BotIndicator.LOW_KARMA,
                triggered=is_low_karma,
                score=score,
                details=f"Karma: {karma} (threshold: {self.config.min_karma})",
                evidence={"karma": karma},
            )
        )

        # Karma farming check (too much karma too fast)
        if account_age_days > 0:
            karma_per_day = karma / account_age_days
            is_karma_farming = (
                karma_per_day > self.config.karma_to_age_ratio_threshold
                and account_age_days < 90  # Only suspicious for newer accounts
            )

            if is_karma_farming:
                score = min(
                    1.0,
                    karma_per_day / (self.config.karma_to_age_ratio_threshold * 2),
                )
            else:
                score = 0.0

            results.append(
                BotIndicatorResult(
                    indicator=BotIndicator.KARMA_FARMING,
                    triggered=is_karma_farming,
                    score=score,
                    details=f"Karma/day ratio: {karma_per_day:.1f} (threshold: {self.config.karma_to_age_ratio_threshold})",
                    evidence={"karma_per_day": karma_per_day},
                )
            )

        return results

    def check_posting_frequency(self, post_times: list[datetime]) -> list[BotIndicatorResult]:
        """Check posting frequency patterns."""
        results = []

        if len(post_times) < 2:
            return results

        # Sort times
        sorted_times = sorted(post_times)

        # Check posts per hour
        now = datetime.utcnow()
        posts_last_hour = sum(1 for t in sorted_times if (now - t).total_seconds() < 3600)
        is_high_freq = posts_last_hour > self.config.max_posts_per_hour

        if is_high_freq:
            score = min(1.0, posts_last_hour / (self.config.max_posts_per_hour * 2))
        else:
            score = 0.0

        results.append(
            BotIndicatorResult(
                indicator=BotIndicator.HIGH_FREQUENCY,
                triggered=is_high_freq,
                score=score,
                details=f"Posts in last hour: {posts_last_hour} (threshold: {self.config.max_posts_per_hour})",
                evidence={"posts_last_hour": posts_last_hour},
            )
        )

        # Check for suspicious timing patterns (too regular intervals)
        if len(sorted_times) >= 3:
            intervals = [(sorted_times[i + 1] - sorted_times[i]).total_seconds() for i in range(len(sorted_times) - 1)]

            if intervals:
                mean_interval = statistics.mean(intervals)
                if mean_interval > 0:
                    # Low variance = suspicious regularity
                    try:
                        variance = statistics.variance(intervals) if len(intervals) > 1 else 0
                        relative_variance = variance / (mean_interval**2) if mean_interval > 0 else 1
                    except statistics.StatisticsError:
                        relative_variance = 1.0

                    is_too_regular = relative_variance < self.config.suspicious_time_variance and len(intervals) >= 5

                    if is_too_regular:
                        score = min(1.0, 1 - relative_variance)
                    else:
                        score = 0.0

                    results.append(
                        BotIndicatorResult(
                            indicator=BotIndicator.TIME_PATTERN,
                            triggered=is_too_regular,
                            score=score,
                            details=f"Posting interval variance: {relative_variance:.3f} (threshold: {self.config.suspicious_time_variance})",
                            evidence={
                                "mean_interval_seconds": mean_interval,
                                "variance": relative_variance,
                            },
                        )
                    )

        return results

    def check_content_repetition(self, contents: list[str]) -> BotIndicatorResult:
        """Check for repetitive content patterns."""
        if len(contents) < 2:
            return BotIndicatorResult(
                indicator=BotIndicator.REPETITIVE_CONTENT,
                triggered=False,
                score=0.0,
                details="Not enough content to analyze",
            )

        # Calculate pairwise similarity using simple word overlap
        similarities = []
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                sim = self._calculate_similarity(contents[i], contents[j])
                similarities.append(sim)

        if not similarities:
            return BotIndicatorResult(
                indicator=BotIndicator.REPETITIVE_CONTENT,
                triggered=False,
                score=0.0,
                details="Could not calculate similarity",
            )

        avg_similarity = statistics.mean(similarities)
        is_repetitive = avg_similarity > self.config.repetition_threshold

        return BotIndicatorResult(
            indicator=BotIndicator.REPETITIVE_CONTENT,
            triggered=is_repetitive,
            score=avg_similarity if is_repetitive else 0.0,
            details=f"Average content similarity: {avg_similarity:.2f} (threshold: {self.config.repetition_threshold})",
            evidence={
                "avg_similarity": avg_similarity,
                "num_posts_analyzed": len(contents),
            },
        )

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using Jaccard index."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def check_linguistic_patterns(self, text: str) -> list[BotIndicatorResult]:
        """Check for suspicious linguistic patterns."""
        results = []

        # Check for shill language
        shill_matches = sum(1 for regex in self._shill_regexes if regex.search(text))
        is_shill = shill_matches >= 2  # Multiple shill phrases

        results.append(
            BotIndicatorResult(
                indicator=BotIndicator.SHILL_LANGUAGE,
                triggered=is_shill,
                score=min(1.0, shill_matches / 3) if is_shill else 0.0,
                details=f"Shill patterns matched: {shill_matches}",
                evidence={"shill_pattern_count": shill_matches},
            )
        )

        # Check emoji ratio
        emoji_pattern = re.compile(
            "["
            "\U0001f300-\U0001f9ff"  # Symbols & pictographs
            "\U0001fa00-\U0001faff"  # Extended-A
            "\U00002702-\U000027b0"  # Dingbats
            "]+",
            flags=re.UNICODE,
        )
        emojis = emoji_pattern.findall(text)
        emoji_count = sum(len(e) for e in emojis)
        text_length = len(text.replace(" ", ""))

        if text_length > 0:
            emoji_ratio = emoji_count / text_length
            is_excessive = emoji_ratio > self.config.max_emoji_ratio

            results.append(
                BotIndicatorResult(
                    indicator=BotIndicator.EXCESSIVE_EMOJIS,
                    triggered=is_excessive,
                    score=min(1.0, emoji_ratio / (self.config.max_emoji_ratio * 2)),
                    details=f"Emoji ratio: {emoji_ratio:.2%} (threshold: {self.config.max_emoji_ratio:.0%})",
                    evidence={"emoji_ratio": emoji_ratio, "emoji_count": emoji_count},
                )
            )

        # Check for template text
        template_matches = sum(1 for regex in self._template_regexes if regex.search(text))
        is_template = template_matches >= 2

        results.append(
            BotIndicatorResult(
                indicator=BotIndicator.TEMPLATE_TEXT,
                triggered=is_template,
                score=min(1.0, template_matches / 2) if is_template else 0.0,
                details=f"Template patterns matched: {template_matches}",
                evidence={"template_count": template_matches},
            )
        )

        # Check word variety
        words = text.lower().split()
        if len(words) >= 10:
            unique_ratio = len(set(words)) / len(words)
            is_low_variety = unique_ratio < self.config.min_unique_words_ratio

            if is_low_variety:
                score = max(0, (self.config.min_unique_words_ratio - unique_ratio) * 2)
            else:
                score = 0.0

            # Note: This would be COPY_PASTE indicator but we'll use template instead
            # since we already have that

        return results

    def check_single_topic(self, mentioned_tickers: list[list[str]]) -> BotIndicatorResult:
        """Check if account only posts about a single topic/ticker."""
        if not mentioned_tickers:
            return BotIndicatorResult(
                indicator=BotIndicator.SINGLE_TOPIC,
                triggered=False,
                score=0.0,
                details="No tickers to analyze",
            )

        # Flatten and count tickers
        all_tickers = [t for tickers in mentioned_tickers for t in tickers]

        if not all_tickers:
            return BotIndicatorResult(
                indicator=BotIndicator.SINGLE_TOPIC,
                triggered=False,
                score=0.0,
                details="No tickers found",
            )

        # Count ticker frequency
        ticker_counts: dict[str, int] = defaultdict(int)
        for ticker in all_tickers:
            ticker_counts[ticker] += 1

        total_mentions = len(all_tickers)
        most_common_count = max(ticker_counts.values())
        concentration = most_common_count / total_mentions

        is_single_topic = concentration > 0.8 and len(mentioned_tickers) >= 5

        return BotIndicatorResult(
            indicator=BotIndicator.SINGLE_TOPIC,
            triggered=is_single_topic,
            score=concentration if is_single_topic else 0.0,
            details=f"Topic concentration: {concentration:.0%} across {len(mentioned_tickers)} posts",
            evidence={
                "concentration": concentration,
                "num_posts": len(mentioned_tickers),
                "unique_tickers": len(ticker_counts),
            },
        )

    def detect_coordinated_campaign(
        self,
        posts: list[dict[str, Any]],
        ticker: str,
    ) -> CoordinatedCampaign | None:
        """
        Detect coordinated campaigns for a specific ticker.

        Args:
            posts: List of post dictionaries with author, time, content
            ticker: Ticker to analyze

        Returns:
            CoordinatedCampaign if detected, None otherwise
        """
        if len(posts) < self.config.min_coordinated_posts:
            return None

        # Group posts by time window
        time_windows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        window_minutes = self.config.coordination_time_window_minutes

        for post in posts:
            post_time = post.get("created_utc")
            if isinstance(post_time, str):
                post_time = datetime.fromisoformat(post_time.replace("Z", "+00:00"))
            elif not isinstance(post_time, datetime):
                continue

            # Create window key (floor to window size)
            window_key = int(post_time.timestamp() / (window_minutes * 60))
            time_windows[window_key].append(post)

        # Check each window for coordination
        for window_key, window_posts in time_windows.items():
            if len(window_posts) < self.config.min_coordinated_posts:
                continue

            # Get unique authors
            authors = set(p.get("author", {}).get("name", "") for p in window_posts)

            # Check for similar usernames (e.g., bot1, bot2, bot3)
            username_similarities = self._check_username_patterns(list(authors))

            # Check for identical/very similar content
            contents = [f"{p.get('title', '')} {p.get('content', '')}" for p in window_posts]
            content_similarities = []
            for i in range(len(contents)):
                for j in range(i + 1, len(contents)):
                    sim = self._calculate_similarity(contents[i], contents[j])
                    content_similarities.append(sim)

            avg_content_sim = statistics.mean(content_similarities) if content_similarities else 0

            # Determine if coordinated
            is_coordinated = len(authors) >= 3 and (username_similarities > 0.5 or avg_content_sim > 0.6)

            if is_coordinated:
                confidence = (username_similarities + avg_content_sim) / 2
                post_times = [p.get("created_utc") for p in window_posts if isinstance(p.get("created_utc"), datetime)]

                campaign = CoordinatedCampaign(
                    campaign_id=f"{ticker}_{window_key}",
                    ticker=ticker,
                    accounts=list(authors),
                    post_times=[t for t in post_times if t],
                    similar_content=contents[:5],
                    confidence=confidence,
                )

                self._detected_campaigns.append(campaign)
                return campaign

        return None

    def _check_username_patterns(self, usernames: list[str]) -> float:
        """Check for suspicious patterns in usernames."""
        if len(usernames) < 2:
            return 0.0

        # Check for sequential patterns (user1, user2, user3)
        sequential_count = 0
        for name in usernames:
            # Extract trailing numbers
            match = re.search(r"(\d+)$", name)
            if match:
                sequential_count += 1

        # Check for similar base names
        base_names = [re.sub(r"\d+$", "", name.lower()) for name in usernames]
        name_counts: dict[str, int] = defaultdict(int)
        for base in base_names:
            if len(base) >= 3:  # Ignore very short bases
                name_counts[base] += 1

        max_similar = max(name_counts.values()) if name_counts else 0

        # Score based on patterns
        sequential_score = sequential_count / len(usernames)
        similarity_score = max_similar / len(usernames)

        return max(sequential_score, similarity_score)

    def analyze_account(
        self,
        author_name: str,
        account_age_days: int,
        karma: int,
        posts: list[dict[str, Any]] | None = None,
    ) -> BotDetectionResult:
        """
        Perform comprehensive bot detection analysis on an account.

        Args:
            author_name: Username
            account_age_days: Account age in days
            karma: Total karma
            posts: Optional list of posts for pattern analysis

        Returns:
            Complete bot detection result
        """
        indicators: list[BotIndicatorResult] = []

        # Account checks
        indicators.append(self.check_account_age(account_age_days))
        indicators.extend(self.check_karma(karma, account_age_days))

        # Posting pattern checks (if posts provided)
        if posts:
            # Extract times
            post_times = []
            for p in posts:
                t = p.get("created_utc")
                if isinstance(t, datetime):
                    post_times.append(t)
                elif isinstance(t, str):
                    try:
                        post_times.append(datetime.fromisoformat(t.replace("Z", "+00:00")))
                    except ValueError:
                        pass

            indicators.extend(self.check_posting_frequency(post_times))

            # Content checks
            contents = [f"{p.get('title', '')} {p.get('content', '')}" for p in posts]
            indicators.append(self.check_content_repetition(contents))

            # Linguistic checks on combined content
            combined_content = " ".join(contents[:10])  # Limit for performance
            indicators.extend(self.check_linguistic_patterns(combined_content))

            # Single topic check
            mentioned_tickers = [p.get("mentioned_tickers", []) for p in posts]
            indicators.append(self.check_single_topic(mentioned_tickers))

        # Calculate overall probability
        triggered_indicators = [i for i in indicators if i.triggered]
        total_score = sum(i.score for i in triggered_indicators)

        # Normalize to probability
        # Weight account-based indicators more heavily
        account_weight = 2.0
        weighted_score = 0.0
        weight_sum = 0.0

        for i in indicators:
            if i.triggered:
                if i.indicator in {
                    BotIndicator.NEW_ACCOUNT,
                    BotIndicator.LOW_KARMA,
                    BotIndicator.KARMA_FARMING,
                }:
                    weighted_score += i.score * account_weight
                    weight_sum += account_weight
                else:
                    weighted_score += i.score
                    weight_sum += 1.0

        if weight_sum > 0:
            bot_probability = min(1.0, weighted_score / max(weight_sum, 3))
        else:
            bot_probability = 0.0

        # Determine confidence level
        if bot_probability > 0.9:
            confidence = BotConfidence.DEFINITE
        elif bot_probability > 0.7:
            confidence = BotConfidence.HIGH
        elif bot_probability > 0.5:
            confidence = BotConfidence.MEDIUM
        elif bot_probability > 0.3:
            confidence = BotConfidence.LOW
        else:
            confidence = BotConfidence.UNLIKELY

        # Generate recommendation
        if bot_probability > 0.7:
            recommendation = "Exclude from analysis - high bot probability"
        elif bot_probability > 0.5:
            recommendation = "Reduce weight in sentiment analysis"
        elif bot_probability > 0.3:
            recommendation = "Flag for review but include"
        else:
            recommendation = "Include normally"

        return BotDetectionResult(
            author_name=author_name,
            is_bot=bot_probability > 0.5,
            confidence=confidence,
            bot_probability=bot_probability,
            indicators=indicators,
            account_age_days=account_age_days,
            karma=karma,
            recommendation=recommendation,
        )

    def filter_posts(
        self,
        posts: list[dict[str, Any]],
        threshold: float = 0.5,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Filter posts to remove likely bot content.

        Args:
            posts: List of posts to filter
            threshold: Bot probability threshold for removal

        Returns:
            Tuple of (clean_posts, filtered_posts)
        """
        clean_posts = []
        filtered_posts = []

        for post in posts:
            author = post.get("author", {})
            result = self.analyze_account(
                author_name=author.get("name", "unknown"),
                account_age_days=author.get("account_age_days", 0),
                karma=author.get("karma", 0),
                posts=[post],  # Single post analysis
            )

            if result.bot_probability < threshold:
                clean_posts.append(post)
            else:
                filtered_posts.append(post)
                logger.debug(
                    "Filtered bot post: %s (probability: %.2f)",
                    author.get("name"),
                    result.bot_probability,
                )

        logger.info(
            "Filtered %d/%d posts as likely bots",
            len(filtered_posts),
            len(posts),
        )

        return clean_posts, filtered_posts

    def get_detected_campaigns(self) -> list[CoordinatedCampaign]:
        """Get all detected coordinated campaigns."""
        return self._detected_campaigns.copy()

    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._post_cache.clear()
        self._detected_campaigns.clear()


def create_bot_detector(
    config: BotDetectorConfig | None = None,
) -> BotDetector:
    """
    Create bot detector instance.

    Args:
        config: Detection configuration

    Returns:
        Configured BotDetector instance
    """
    return BotDetector(config)


__all__ = [
    "BotConfidence",
    "BotDetectionResult",
    "BotDetector",
    "BotDetectorConfig",
    "BotIndicator",
    "BotIndicatorResult",
    "CoordinatedCampaign",
    "create_bot_detector",
]
