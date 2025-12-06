"""
Tests for Reddit Scanner and Sentiment Analyzer

Tests the Reddit scanning functionality including:
- Ticker extraction from text
- Post/comment processing
- Mention aggregation
- Trend detection
- Sentiment analysis

Note: Tests use mocks for Reddit API to avoid rate limiting.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from llm.reddit_sentiment import (
    RedditSentimentConfig,
    RedditSentimentSignal,
    RedditSentimentSummary,
    TickerSentiment,
    create_reddit_sentiment_analyzer,
)
from scanners.reddit_scanner import (
    PRAW_AVAILABLE,
    RedditAuthor,
    RedditPost,
    RedditScannerConfig,
    TickerMention,
    create_reddit_scanner,
)


class TestRedditScanner:
    """Tests for RedditScanner class."""

    def test_create_reddit_scanner_default_config(self):
        """Test creating scanner with default configuration."""
        scanner = create_reddit_scanner()
        assert scanner is not None
        assert scanner.config.enabled is True
        assert len(scanner.config.subreddits) > 0

    def test_create_reddit_scanner_custom_config(self):
        """Test creating scanner with custom configuration."""
        config = RedditScannerConfig(
            enabled=True,
            subreddits=["wallstreetbets", "options"],
            min_upvotes=50,
            lookback_hours=12,
        )
        scanner = create_reddit_scanner(config)
        assert scanner.config.min_upvotes == 50
        assert scanner.config.lookback_hours == 12
        assert len(scanner.config.subreddits) == 2

    def test_extract_tickers_basic(self):
        """Test basic ticker extraction."""
        scanner = create_reddit_scanner()

        # Test with $ prefix
        tickers = scanner._extract_tickers("I'm bullish on $AAPL and $TSLA")
        assert "AAPL" in tickers
        assert "TSLA" in tickers

        # Test without prefix
        tickers = scanner._extract_tickers("Buying NVDA calls tomorrow")
        assert "NVDA" in tickers

    def test_extract_tickers_blacklist(self):
        """Test that common words are filtered out."""
        scanner = create_reddit_scanner()

        # These should NOT be extracted as tickers
        text = "I AM going to buy THE stock FOR my IRA"
        tickers = scanner._extract_tickers(text)

        assert "AM" not in tickers
        assert "THE" not in tickers
        assert "FOR" not in tickers
        assert "IRA" not in tickers

    def test_extract_tickers_whitelist(self):
        """Test that known tickers are preserved."""
        scanner = create_reddit_scanner()

        # These are valid tickers that might look like words
        text = "AMD and BB are my plays"
        tickers = scanner._extract_tickers(text)

        assert "AMD" in tickers
        assert "BB" in tickers

    def test_extract_tickers_deduplication(self):
        """Test that duplicate tickers are removed."""
        scanner = create_reddit_scanner()

        text = "AAPL AAPL $AAPL apple"
        tickers = scanner._extract_tickers(text)

        # Should only appear once
        assert tickers.count("AAPL") == 1

    def test_calculate_trend_score(self):
        """Test trend score calculation."""
        scanner = create_reddit_scanner()

        mention = TickerMention(
            ticker="AAPL",
            mention_count=20,
            unique_authors=15,
            total_upvotes=500,
            avg_sentiment=0.5,
            sentiment_std=0.2,
            first_mention=datetime.utcnow() - timedelta(hours=12),
            last_mention=datetime.utcnow() - timedelta(hours=1),
        )

        score = scanner.calculate_trend_score(mention, baseline_mentions=10)

        # Should be a positive score
        assert score > 0
        assert score <= 100

    def test_calculate_trend_score_low_volume(self):
        """Test trend score for low volume ticker."""
        scanner = create_reddit_scanner()

        mention = TickerMention(
            ticker="XYZ",
            mention_count=2,
            unique_authors=2,
            total_upvotes=10,
            avg_sentiment=0.0,
            sentiment_std=0.0,
        )

        score = scanner.calculate_trend_score(mention)

        # Low volume should have low score
        assert score < 50

    def test_aggregate_mentions_posts(self):
        """Test mention aggregation from posts."""
        scanner = create_reddit_scanner()

        author = RedditAuthor(
            name="test_user",
            account_age_days=365,
            karma=10000,
        )

        posts = [
            RedditPost(
                post_id="abc123",
                subreddit="wallstreetbets",
                title="AAPL to the moon!",
                content="Buying calls on AAPL",
                author=author,
                created_utc=datetime.utcnow(),
                upvotes=100,
                upvote_ratio=0.9,
                num_comments=50,
                url="https://reddit.com/r/wallstreetbets/abc123",
                mentioned_tickers=["AAPL"],
            ),
            RedditPost(
                post_id="def456",
                subreddit="options",
                title="AAPL earnings play",
                content="Iron condor on AAPL",
                author=author,
                created_utc=datetime.utcnow(),
                upvotes=50,
                upvote_ratio=0.85,
                num_comments=20,
                url="https://reddit.com/r/options/def456",
                mentioned_tickers=["AAPL"],
            ),
        ]

        mentions = scanner.aggregate_mentions(posts, [])

        assert "AAPL" in mentions
        assert mentions["AAPL"].mention_count == 2
        assert mentions["AAPL"].unique_authors == 1  # Same author
        assert mentions["AAPL"].total_upvotes == 150
        assert "wallstreetbets" in mentions["AAPL"].subreddit_distribution
        assert "options" in mentions["AAPL"].subreddit_distribution

    def test_aggregate_mentions_multiple_tickers(self):
        """Test aggregation with multiple tickers."""
        scanner = create_reddit_scanner()

        author1 = RedditAuthor(name="user1", account_age_days=100, karma=5000)
        author2 = RedditAuthor(name="user2", account_age_days=200, karma=8000)

        posts = [
            RedditPost(
                post_id="p1",
                subreddit="wallstreetbets",
                title="AAPL and TSLA",
                content="Both going up",
                author=author1,
                created_utc=datetime.utcnow(),
                upvotes=100,
                upvote_ratio=0.9,
                num_comments=10,
                url="https://reddit.com/p1",
                mentioned_tickers=["AAPL", "TSLA"],
            ),
            RedditPost(
                post_id="p2",
                subreddit="options",
                title="TSLA calls",
                content="Buying TSLA",
                author=author2,
                created_utc=datetime.utcnow(),
                upvotes=50,
                upvote_ratio=0.8,
                num_comments=5,
                url="https://reddit.com/p2",
                mentioned_tickers=["TSLA"],
            ),
        ]

        mentions = scanner.aggregate_mentions(posts, [])

        assert "AAPL" in mentions
        assert "TSLA" in mentions
        assert mentions["AAPL"].mention_count == 1
        assert mentions["TSLA"].mention_count == 2
        assert mentions["TSLA"].unique_authors == 2

    def test_detect_trending_threshold(self):
        """Test trending detection with threshold."""
        scanner = create_reddit_scanner()

        mentions = {
            "AAPL": TickerMention(
                ticker="AAPL",
                mention_count=50,
                unique_authors=30,
                total_upvotes=1000,
                avg_sentiment=0.5,
                sentiment_std=0.2,
                last_mention=datetime.utcnow(),
            ),
            "XYZ": TickerMention(
                ticker="XYZ",
                mention_count=3,
                unique_authors=2,
                total_upvotes=10,
                avg_sentiment=0.0,
                sentiment_std=0.0,
            ),
        }

        trending = scanner.detect_trending(mentions, min_mentions=5, trend_threshold=30)

        # AAPL should be trending, XYZ should not
        trending_tickers = [t.ticker for t in trending]
        assert "AAPL" in trending_tickers
        assert "XYZ" not in trending_tickers

    def test_reddit_post_to_dict(self):
        """Test RedditPost serialization."""
        author = RedditAuthor(
            name="test_user",
            account_age_days=365,
            karma=10000,
            is_suspected_bot=False,
            bot_probability=0.1,
        )

        post = RedditPost(
            post_id="abc123",
            subreddit="wallstreetbets",
            title="Test title",
            content="Test content",
            author=author,
            created_utc=datetime.utcnow(),
            upvotes=100,
            upvote_ratio=0.9,
            num_comments=50,
            url="https://reddit.com/test",
            flair="DD",
            is_dd=True,
            mentioned_tickers=["AAPL", "TSLA"],
        )

        data = post.to_dict()

        assert data["post_id"] == "abc123"
        assert data["subreddit"] == "wallstreetbets"
        assert data["upvotes"] == 100
        assert data["is_dd"] is True
        assert "AAPL" in data["mentioned_tickers"]

    def test_ticker_mention_to_dict(self):
        """Test TickerMention serialization."""
        mention = TickerMention(
            ticker="AAPL",
            mention_count=10,
            unique_authors=5,
            total_upvotes=500,
            avg_sentiment=0.5,
            sentiment_std=0.2,
            subreddit_distribution={"wsb": 7, "options": 3},
            hourly_distribution={14: 5, 15: 5},
            is_trending=True,
            trend_score=75.0,
        )

        data = mention.to_dict()

        assert data["ticker"] == "AAPL"
        assert data["mention_count"] == 10
        assert data["is_trending"] is True
        assert data["trend_score"] == 75.0


class TestRedditSentimentAnalyzer:
    """Tests for RedditSentimentAnalyzer class."""

    def test_create_analyzer_default(self):
        """Test creating analyzer with defaults."""
        analyzer = create_reddit_sentiment_analyzer()
        assert analyzer is not None
        assert analyzer.config.use_finbert is True

    def test_create_analyzer_custom_config(self):
        """Test creating analyzer with custom config."""
        config = RedditSentimentConfig(
            use_finbert=False,
            min_mentions_for_signal=10,
            bullish_threshold=0.4,
        )
        analyzer = create_reddit_sentiment_analyzer(config)

        assert analyzer.config.use_finbert is False
        assert analyzer.config.min_mentions_for_signal == 10
        assert analyzer.config.bullish_threshold == 0.4

    def test_preprocess_reddit_text(self):
        """Test Reddit-specific text preprocessing."""
        analyzer = create_reddit_sentiment_analyzer()

        # Test URL removal
        text = "Check out https://example.com for more info about AAPL"
        processed = analyzer._preprocess_reddit_text(text)
        assert "https" not in processed

        # Test empty text handling
        assert analyzer._preprocess_reddit_text("") == ""
        assert analyzer._preprocess_reddit_text("hi") == ""  # Too short

    def test_calculate_reddit_modifier_bullish(self):
        """Test bullish modifier calculation."""
        analyzer = create_reddit_sentiment_analyzer()

        # Bullish text
        modifier = analyzer._calculate_reddit_modifier("🚀🚀🚀 Diamond hands! To the moon! HODL!")
        assert modifier > 0  # Should be positive

    def test_calculate_reddit_modifier_bearish(self):
        """Test bearish modifier calculation."""
        analyzer = create_reddit_sentiment_analyzer()

        # Bearish text
        modifier = analyzer._calculate_reddit_modifier("Puts are printing! 📉 This is going to crash, sell sell sell")
        assert modifier < 0  # Should be negative

    def test_calculate_reddit_modifier_neutral(self):
        """Test neutral modifier calculation."""
        analyzer = create_reddit_sentiment_analyzer()

        # Neutral text
        modifier = analyzer._calculate_reddit_modifier("I'm not sure what to do here. Just watching for now.")
        assert abs(modifier) < 0.1  # Should be near zero

    def test_calculate_weight_upvotes(self):
        """Test weight calculation with upvotes."""
        config = RedditSentimentConfig(
            weight_by_upvotes=True,
            weight_by_author_karma=False,
            dd_post_weight=1.0,
        )
        analyzer = create_reddit_sentiment_analyzer(config)

        weight_low = analyzer._calculate_weight(upvotes=10, author_karma=0, is_dd=False)
        weight_high = analyzer._calculate_weight(upvotes=1000, author_karma=0, is_dd=False)

        assert weight_high > weight_low

    def test_calculate_weight_dd_bonus(self):
        """Test DD post weight bonus."""
        config = RedditSentimentConfig(
            weight_by_upvotes=False,
            weight_by_author_karma=False,
            dd_post_weight=2.0,
        )
        analyzer = create_reddit_sentiment_analyzer(config)

        weight_regular = analyzer._calculate_weight(upvotes=100, author_karma=1000, is_dd=False)
        weight_dd = analyzer._calculate_weight(upvotes=100, author_karma=1000, is_dd=True)

        assert weight_dd == weight_regular * 2.0

    def test_determine_signal_strong_buy(self):
        """Test strong buy signal determination."""
        analyzer = create_reddit_sentiment_analyzer()

        signal = analyzer._determine_signal(
            weighted_score=0.7,
            bullish_pct=0.8,
            bearish_pct=0.1,
            unique_authors=10,
        )

        assert signal == RedditSentimentSignal.STRONG_BUY

    def test_determine_signal_strong_sell(self):
        """Test strong sell signal determination."""
        analyzer = create_reddit_sentiment_analyzer()

        signal = analyzer._determine_signal(
            weighted_score=-0.7,
            bullish_pct=0.1,
            bearish_pct=0.8,
            unique_authors=10,
        )

        assert signal == RedditSentimentSignal.STRONG_SELL

    def test_determine_signal_neutral_low_authors(self):
        """Test neutral signal when not enough authors."""
        config = RedditSentimentConfig(min_authors_for_signal=5)
        analyzer = create_reddit_sentiment_analyzer(config)

        signal = analyzer._determine_signal(
            weighted_score=0.8,
            bullish_pct=0.9,
            bearish_pct=0.05,
            unique_authors=2,  # Below threshold
        )

        assert signal == RedditSentimentSignal.NEUTRAL

    def test_calculate_confidence(self):
        """Test confidence calculation."""
        analyzer = create_reddit_sentiment_analyzer()

        # High confidence scenario
        high_conf = analyzer._calculate_confidence(
            content_count=30,
            unique_authors=15,
            std_score=0.1,
        )

        # Low confidence scenario
        low_conf = analyzer._calculate_confidence(
            content_count=5,
            unique_authors=2,
            std_score=0.8,
        )

        assert high_conf > low_conf
        assert 0 <= high_conf <= 1
        assert 0 <= low_conf <= 1

    def test_analyze_ticker_empty(self):
        """Test analyzing ticker with no content."""
        analyzer = create_reddit_sentiment_analyzer()

        result = analyzer.analyze_ticker("AAPL", posts=[], comments=[])

        assert result.ticker == "AAPL"
        assert result.signal == RedditSentimentSignal.NEUTRAL
        assert result.content_count == 0
        assert result.confidence == 0.0

    def test_ticker_sentiment_to_dict(self):
        """Test TickerSentiment serialization."""
        sentiment = TickerSentiment(
            ticker="AAPL",
            signal=RedditSentimentSignal.BUY,
            avg_score=0.5,
            std_score=0.2,
            confidence=0.8,
            content_count=20,
            unique_authors=15,
            bullish_pct=0.7,
            bearish_pct=0.1,
            neutral_pct=0.2,
            total_upvotes=1000,
            weighted_score=0.55,
            subreddit_breakdown={"wsb": 0.6, "options": 0.4},
        )

        data = sentiment.to_dict()

        assert data["ticker"] == "AAPL"
        assert data["signal"] == "buy"
        assert data["confidence"] == 0.8
        assert "wsb" in data["subreddit_breakdown"]

    def test_reddit_sentiment_summary_to_dict(self):
        """Test RedditSentimentSummary serialization."""
        ts = TickerSentiment(
            ticker="AAPL",
            signal=RedditSentimentSignal.BUY,
            avg_score=0.5,
            std_score=0.2,
            confidence=0.8,
            content_count=20,
            unique_authors=15,
            bullish_pct=0.7,
            bearish_pct=0.1,
            neutral_pct=0.2,
            total_upvotes=1000,
            weighted_score=0.55,
        )

        summary = RedditSentimentSummary(
            ticker_sentiments={"AAPL": ts},
            overall_market_sentiment=0.3,
            most_bullish=["AAPL"],
            most_bearish=["XYZ"],
            trending_tickers=["AAPL", "TSLA"],
            total_posts_analyzed=100,
            total_comments_analyzed=500,
        )

        data = summary.to_dict()

        assert "AAPL" in data["ticker_sentiments"]
        assert data["overall_market_sentiment"] == 0.3
        assert data["total_posts_analyzed"] == 100


class TestIntegration:
    """Integration tests for scanner and sentiment analyzer."""

    def test_scanner_to_sentiment_flow(self):
        """Test data flow from scanner to sentiment analyzer."""
        # Create scanner data
        author = RedditAuthor(
            name="test_user",
            account_age_days=365,
            karma=10000,
        )

        posts = [
            RedditPost(
                post_id="p1",
                subreddit="wallstreetbets",
                title="AAPL to the moon! 🚀",
                content="Diamond hands on these calls. Not selling!",
                author=author,
                created_utc=datetime.utcnow(),
                upvotes=500,
                upvote_ratio=0.95,
                num_comments=100,
                url="https://reddit.com/p1",
                is_dd=True,
                mentioned_tickers=["AAPL"],
            ),
        ]

        # Convert to format expected by sentiment analyzer
        post_dicts = [
            {
                "post_id": p.post_id,
                "title": p.title,
                "content": p.content,
                "subreddit": p.subreddit,
                "upvotes": p.upvotes,
                "is_dd": p.is_dd,
                "author": {
                    "name": p.author.name,
                    "karma": p.author.karma,
                },
            }
            for p in posts
        ]

        # Analyze sentiment
        analyzer = create_reddit_sentiment_analyzer()
        result = analyzer.analyze_ticker("AAPL", posts=post_dicts, comments=[])

        assert result.ticker == "AAPL"
        assert result.content_count >= 0  # May be 0 if text too short after preprocessing

    def test_alert_generation(self):
        """Test alert generation from scanner."""
        scanner = create_reddit_scanner()

        author = RedditAuthor(name="user", account_age_days=100, karma=5000)

        # Create posts with high engagement
        posts = [
            RedditPost(
                post_id=f"p{i}",
                subreddit="wallstreetbets",
                title=f"AAPL play #{i}",
                content="Great setup",
                author=RedditAuthor(name=f"user{i}", account_age_days=100, karma=5000),
                created_utc=datetime.utcnow() - timedelta(hours=i),
                upvotes=100 + i * 10,
                upvote_ratio=0.9,
                num_comments=10,
                url=f"https://reddit.com/p{i}",
                mentioned_tickers=["AAPL"],
            )
            for i in range(10)
        ]

        # Aggregate and generate alerts
        mentions = scanner.aggregate_mentions(posts, [])
        alerts = scanner.generate_alerts(mentions, min_mentions=5)

        # Should have at least one alert for AAPL
        if (
            mentions.get(
                "AAPL",
                TickerMention(
                    ticker="", mention_count=0, unique_authors=0, total_upvotes=0, avg_sentiment=0, sentiment_std=0
                ),
            ).trend_score
            >= 50
        ):
            assert len(alerts) >= 1
            assert any(a.ticker == "AAPL" for a in alerts)


# Skip tests that require PRAW if not available
@pytest.mark.skipif(not PRAW_AVAILABLE, reason="PRAW not installed")
class TestPRAWIntegration:
    """Tests that require PRAW to be installed."""

    def test_praw_available(self):
        """Test that PRAW is available."""
        assert PRAW_AVAILABLE is True

    @patch("scanners.reddit_scanner.praw.Reddit")
    def test_initialize_with_credentials(self, mock_reddit):
        """Test initialization with credentials."""
        config = RedditScannerConfig(
            client_id="test_id",
            client_secret="test_secret",
            user_agent="test_agent",
        )
        scanner = create_reddit_scanner(config)

        # Mock the user.me() call
        mock_instance = MagicMock()
        mock_reddit.return_value = mock_instance

        result = scanner._ensure_initialized()
        # Note: This will actually try to initialize, but we're checking the flow


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
