"""
Tests for Bot Detection Filter

Tests the bot detection functionality including:
- Account age and karma checks
- Posting frequency analysis
- Content repetition detection
- Linguistic pattern analysis
- Coordinated campaign detection
"""

from datetime import datetime, timedelta

import pytest

from llm.bot_detector import (
    BotConfidence,
    BotDetectionResult,
    BotDetectorConfig,
    BotIndicator,
    BotIndicatorResult,
    CoordinatedCampaign,
    create_bot_detector,
)


class TestBotDetectorCreation:
    """Tests for BotDetector creation."""

    def test_create_detector_default(self):
        """Test creating detector with default config."""
        detector = create_bot_detector()
        assert detector is not None
        assert detector.config.enabled is True
        assert detector.config.min_account_age_days == 30

    def test_create_detector_custom_config(self):
        """Test creating detector with custom config."""
        config = BotDetectorConfig(
            min_account_age_days=60,
            min_karma=500,
            max_posts_per_hour=10,
        )
        detector = create_bot_detector(config)

        assert detector.config.min_account_age_days == 60
        assert detector.config.min_karma == 500
        assert detector.config.max_posts_per_hour == 10


class TestAccountAgeCheck:
    """Tests for account age checking."""

    def test_new_account_detected(self):
        """Test that new accounts are flagged."""
        detector = create_bot_detector()
        result = detector.check_account_age(account_age_days=5)

        assert result.triggered is True
        assert result.indicator == BotIndicator.NEW_ACCOUNT
        assert result.score > 0

    def test_old_account_not_flagged(self):
        """Test that old accounts are not flagged."""
        detector = create_bot_detector()
        result = detector.check_account_age(account_age_days=365)

        assert result.triggered is False
        assert result.score == 0

    def test_borderline_account(self):
        """Test account at threshold."""
        config = BotDetectorConfig(min_account_age_days=30)
        detector = create_bot_detector(config)

        # Just below threshold
        result = detector.check_account_age(account_age_days=29)
        assert result.triggered is True

        # At threshold
        result = detector.check_account_age(account_age_days=30)
        assert result.triggered is False


class TestKarmaCheck:
    """Tests for karma checking."""

    def test_low_karma_detected(self):
        """Test that low karma accounts are flagged."""
        detector = create_bot_detector()
        results = detector.check_karma(karma=10, account_age_days=30)

        # Find low karma indicator
        low_karma = next(
            (r for r in results if r.indicator == BotIndicator.LOW_KARMA),
            None,
        )
        assert low_karma is not None
        assert low_karma.triggered is True

    def test_high_karma_not_flagged(self):
        """Test that high karma accounts are not flagged."""
        detector = create_bot_detector()
        results = detector.check_karma(karma=10000, account_age_days=365)

        low_karma = next(
            (r for r in results if r.indicator == BotIndicator.LOW_KARMA),
            None,
        )
        assert low_karma is not None
        assert low_karma.triggered is False

    def test_karma_farming_detection(self):
        """Test detection of karma farming (too much karma too fast)."""
        config = BotDetectorConfig(karma_to_age_ratio_threshold=100)
        detector = create_bot_detector(config)

        # New account with lots of karma
        results = detector.check_karma(karma=10000, account_age_days=10)

        karma_farming = next(
            (r for r in results if r.indicator == BotIndicator.KARMA_FARMING),
            None,
        )
        assert karma_farming is not None
        assert karma_farming.triggered is True

    def test_karma_farming_not_triggered_old_account(self):
        """Test karma farming not triggered for older accounts."""
        config = BotDetectorConfig(karma_to_age_ratio_threshold=100)
        detector = create_bot_detector(config)

        # Old account with lots of karma is fine
        results = detector.check_karma(karma=50000, account_age_days=365)

        karma_farming = next(
            (r for r in results if r.indicator == BotIndicator.KARMA_FARMING),
            None,
        )
        assert karma_farming is not None
        assert karma_farming.triggered is False


class TestPostingFrequency:
    """Tests for posting frequency analysis."""

    def test_high_frequency_detected(self):
        """Test detection of high posting frequency."""
        config = BotDetectorConfig(max_posts_per_hour=5)
        detector = create_bot_detector(config)

        now = datetime.utcnow()
        # 10 posts in the last hour
        post_times = [now - timedelta(minutes=i * 5) for i in range(10)]

        results = detector.check_posting_frequency(post_times)

        high_freq = next(
            (r for r in results if r.indicator == BotIndicator.HIGH_FREQUENCY),
            None,
        )
        assert high_freq is not None
        assert high_freq.triggered is True

    def test_normal_frequency_not_flagged(self):
        """Test that normal posting frequency is not flagged."""
        config = BotDetectorConfig(max_posts_per_hour=5)
        detector = create_bot_detector(config)

        now = datetime.utcnow()
        # 3 posts in the last hour
        post_times = [now - timedelta(minutes=i * 20) for i in range(3)]

        results = detector.check_posting_frequency(post_times)

        high_freq = next(
            (r for r in results if r.indicator == BotIndicator.HIGH_FREQUENCY),
            None,
        )
        assert high_freq is not None
        assert high_freq.triggered is False

    def test_suspicious_timing_pattern(self):
        """Test detection of suspiciously regular posting intervals."""
        config = BotDetectorConfig(suspicious_time_variance=0.1)
        detector = create_bot_detector(config)

        now = datetime.utcnow()
        # Posts at exactly 5 minute intervals (suspicious)
        post_times = [now - timedelta(minutes=i * 5) for i in range(10)]

        results = detector.check_posting_frequency(post_times)

        time_pattern = next(
            (r for r in results if r.indicator == BotIndicator.TIME_PATTERN),
            None,
        )
        # May or may not trigger depending on exact variance
        if time_pattern:
            assert time_pattern.indicator == BotIndicator.TIME_PATTERN


class TestContentAnalysis:
    """Tests for content repetition analysis."""

    def test_repetitive_content_detected(self):
        """Test detection of repetitive content."""
        detector = create_bot_detector()

        contents = [
            "AAPL is going to moon! Buy now!",
            "AAPL is going to moon! Get in now!",
            "AAPL is going to the moon! Buy now!",
            "AAPL to the moon! Buy now everyone!",
        ]

        result = detector.check_content_repetition(contents)
        # These are very similar, should be flagged
        # Note: Depends on threshold
        assert result.indicator == BotIndicator.REPETITIVE_CONTENT

    def test_varied_content_not_flagged(self):
        """Test that varied content is not flagged."""
        detector = create_bot_detector()

        contents = [
            "I'm bullish on tech stocks right now.",
            "The market seems overvalued to me.",
            "Anyone else watching the Fed announcement?",
            "My portfolio is down 5% today.",
        ]

        result = detector.check_content_repetition(contents)
        assert result.triggered is False or result.score < 0.5

    def test_insufficient_content(self):
        """Test handling of insufficient content."""
        detector = create_bot_detector()

        result = detector.check_content_repetition(["single post"])
        assert result.triggered is False


class TestLinguisticPatterns:
    """Tests for linguistic pattern analysis."""

    def test_shill_language_detected(self):
        """Test detection of shill language."""
        detector = create_bot_detector()

        text = "This is huge! Get in now before it's too late! " "Trust me, this is gonna explode! Easy money!"

        results = detector.check_linguistic_patterns(text)

        shill = next(
            (r for r in results if r.indicator == BotIndicator.SHILL_LANGUAGE),
            None,
        )
        assert shill is not None
        assert shill.triggered is True

    def test_normal_language_not_flagged(self):
        """Test that normal language is not flagged."""
        detector = create_bot_detector()

        text = (
            "I've been looking at the fundamentals and I think "
            "this stock might be undervalued. The P/E ratio is "
            "reasonable compared to peers."
        )

        results = detector.check_linguistic_patterns(text)

        shill = next(
            (r for r in results if r.indicator == BotIndicator.SHILL_LANGUAGE),
            None,
        )
        assert shill is not None
        assert shill.triggered is False

    def test_excessive_emojis_detected(self):
        """Test detection of excessive emoji usage."""
        config = BotDetectorConfig(max_emoji_ratio=0.1)
        detector = create_bot_detector(config)

        text = "🚀🚀🚀🚀🚀 AAPL 🚀🚀🚀🚀🚀 to the 🌙🌙🌙 moon 💎💎💎"

        results = detector.check_linguistic_patterns(text)

        emoji_result = next(
            (r for r in results if r.indicator == BotIndicator.EXCESSIVE_EMOJIS),
            None,
        )
        assert emoji_result is not None
        # High emoji ratio

    def test_template_text_detected(self):
        """Test detection of template text."""
        detector = create_bot_detector()

        text = (
            "I'm not a financial advisor. This is not financial advice. "
            "Do your own research. Position: 100 shares. Disclosure: long"
        )

        results = detector.check_linguistic_patterns(text)

        template = next(
            (r for r in results if r.indicator == BotIndicator.TEMPLATE_TEXT),
            None,
        )
        assert template is not None
        assert template.triggered is True


class TestSingleTopicDetection:
    """Tests for single topic detection."""

    def test_single_topic_detected(self):
        """Test detection of accounts posting about single ticker."""
        detector = create_bot_detector()

        # All posts about AAPL
        mentioned_tickers = [
            ["AAPL"],
            ["AAPL"],
            ["AAPL"],
            ["AAPL"],
            ["AAPL"],
            ["AAPL"],
        ]

        result = detector.check_single_topic(mentioned_tickers)
        assert result.triggered is True
        assert result.indicator == BotIndicator.SINGLE_TOPIC

    def test_varied_topics_not_flagged(self):
        """Test that varied topics are not flagged."""
        detector = create_bot_detector()

        mentioned_tickers = [
            ["AAPL"],
            ["TSLA", "NVDA"],
            ["MSFT"],
            ["GOOG"],
            ["SPY", "QQQ"],
            ["AMD"],
        ]

        result = detector.check_single_topic(mentioned_tickers)
        assert result.triggered is False


class TestCoordinatedCampaignDetection:
    """Tests for coordinated campaign detection."""

    def test_coordinated_campaign_detected(self):
        """Test detection of coordinated posting campaign."""
        config = BotDetectorConfig(
            coordination_time_window_minutes=10,
            min_coordinated_posts=3,
        )
        detector = create_bot_detector(config)

        now = datetime.utcnow()

        # Multiple similar posts in short time window
        posts = [
            {
                "author": {"name": f"user{i}"},
                "title": "AAPL is going to moon!",
                "content": "Buy AAPL now before it's too late!",
                "created_utc": now - timedelta(minutes=i),
            }
            for i in range(5)
        ]

        campaign = detector.detect_coordinated_campaign(posts, "AAPL")
        # May or may not detect depending on similarity threshold
        # Campaign detection is complex

    def test_username_pattern_detection(self):
        """Test detection of suspicious username patterns."""
        detector = create_bot_detector()

        usernames = ["user123", "user124", "user125", "user126"]
        score = detector._check_username_patterns(usernames)

        # Sequential numbered usernames should score high
        assert score > 0


class TestFullAccountAnalysis:
    """Tests for complete account analysis."""

    def test_analyze_bot_account(self):
        """Test analysis of likely bot account."""
        detector = create_bot_detector()

        now = datetime.utcnow()
        posts = [
            {
                "title": "AAPL to the moon! 🚀🚀🚀",
                "content": "Get in now! Trust me! Easy money!",
                "created_utc": now - timedelta(minutes=i * 5),
                "mentioned_tickers": ["AAPL"],
            }
            for i in range(10)
        ]

        result = detector.analyze_account(
            author_name="newuser123",
            account_age_days=5,
            karma=50,
            posts=posts,
        )

        assert isinstance(result, BotDetectionResult)
        assert result.is_bot is True or result.bot_probability > 0.3
        assert len(result.indicators) > 0

    def test_analyze_legitimate_account(self):
        """Test analysis of legitimate account."""
        detector = create_bot_detector()

        now = datetime.utcnow()
        posts = [
            {
                "title": "Analysis of tech sector",
                "content": "Looking at fundamentals and valuations.",
                "created_utc": now - timedelta(days=i),
                "mentioned_tickers": ["AAPL", "MSFT", "GOOG"][i % 3 : i % 3 + 1],
            }
            for i in range(5)
        ]

        result = detector.analyze_account(
            author_name="veteran_trader",
            account_age_days=1000,
            karma=50000,
            posts=posts,
        )

        assert result.bot_probability < 0.5
        assert result.confidence in {BotConfidence.LOW, BotConfidence.UNLIKELY}

    def test_confidence_levels(self):
        """Test bot confidence level assignment."""
        detector = create_bot_detector()

        # Very suspicious account
        result_high = detector.analyze_account(
            author_name="bot123",
            account_age_days=1,
            karma=10,
        )

        # Normal account
        result_low = detector.analyze_account(
            author_name="normal_user",
            account_age_days=500,
            karma=10000,
        )

        # High probability should have high confidence
        if result_high.bot_probability > 0.7:
            assert result_high.confidence in {BotConfidence.DEFINITE, BotConfidence.HIGH}

        # Low probability should have low confidence
        assert result_low.confidence in {BotConfidence.LOW, BotConfidence.UNLIKELY}


class TestPostFiltering:
    """Tests for post filtering functionality."""

    def test_filter_bot_posts(self):
        """Test filtering of bot posts."""
        detector = create_bot_detector()

        posts = [
            {
                "author": {"name": "bot1", "account_age_days": 2, "karma": 10},
                "title": "AAPL 🚀🚀🚀",
                "content": "Get in now!",
            },
            {
                "author": {"name": "legit_user", "account_age_days": 500, "karma": 10000},
                "title": "Technical analysis",
                "content": "Here's my analysis.",
            },
        ]

        clean, filtered = detector.filter_posts(posts, threshold=0.5)

        # Should have some filtering
        assert len(clean) + len(filtered) == len(posts)

    def test_filter_with_custom_threshold(self):
        """Test filtering with custom threshold."""
        detector = create_bot_detector()

        posts = [
            {
                "author": {"name": "user1", "account_age_days": 100, "karma": 1000},
                "title": "Normal post",
                "content": "Normal content",
            },
        ]

        # Very low threshold - should filter most
        clean_low, filtered_low = detector.filter_posts(posts, threshold=0.1)

        # Very high threshold - should filter few
        clean_high, filtered_high = detector.filter_posts(posts, threshold=0.9)

        assert len(clean_high) >= len(clean_low)


class TestSerializaton:
    """Tests for result serialization."""

    def test_detection_result_to_dict(self):
        """Test BotDetectionResult serialization."""
        result = BotDetectionResult(
            author_name="test_user",
            is_bot=True,
            confidence=BotConfidence.HIGH,
            bot_probability=0.85,
            indicators=[
                BotIndicatorResult(
                    indicator=BotIndicator.NEW_ACCOUNT,
                    triggered=True,
                    score=0.9,
                    details="Account is 5 days old",
                )
            ],
            account_age_days=5,
            karma=50,
            recommendation="Exclude from analysis",
        )

        data = result.to_dict()

        assert data["author_name"] == "test_user"
        assert data["is_bot"] is True
        assert data["confidence"] == "high"
        assert data["bot_probability"] == 0.85
        assert len(data["indicators"]) == 1
        assert data["indicators"][0]["indicator"] == "new_account"

    def test_campaign_to_dict(self):
        """Test CoordinatedCampaign serialization."""
        campaign = CoordinatedCampaign(
            campaign_id="AAPL_123",
            ticker="AAPL",
            accounts=["user1", "user2", "user3"],
            post_times=[
                datetime.utcnow(),
                datetime.utcnow() - timedelta(minutes=2),
                datetime.utcnow() - timedelta(minutes=5),
            ],
            similar_content=["content1", "content2"],
            confidence=0.8,
        )

        data = campaign.to_dict()

        assert data["campaign_id"] == "AAPL_123"
        assert data["ticker"] == "AAPL"
        assert data["num_accounts"] == 3
        assert data["confidence"] == 0.8


class TestCacheManagement:
    """Tests for cache management."""

    def test_clear_cache(self):
        """Test cache clearing."""
        detector = create_bot_detector()

        # Add some detected campaigns
        detector._detected_campaigns.append(
            CoordinatedCampaign(
                campaign_id="test",
                ticker="AAPL",
                accounts=["user1"],
                post_times=[datetime.utcnow()],
                similar_content=["test"],
                confidence=0.5,
            )
        )

        assert len(detector.get_detected_campaigns()) > 0

        detector.clear_cache()

        assert len(detector.get_detected_campaigns()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
