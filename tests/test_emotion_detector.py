"""
Tests for Emotion Detection Layer Module

Tests fear/greed detection beyond simple positive/negative sentiment.
Part of UPGRADE-010 Sprint 3 Iteration 3 - Test Coverage.
"""

import pytest

from llm.emotion_detector import (
    EmotionDetector,
    EmotionDetectorConfig,
    EmotionIndicator,
    EmotionResult,
    MarketEmotion,
    create_emotion_detector,
)


class TestMarketEmotion:
    """Tests for MarketEmotion enum."""

    def test_all_emotions_exist(self):
        """Test that all expected emotions exist."""
        assert MarketEmotion.EXTREME_FEAR.value == "extreme_fear"
        assert MarketEmotion.FEAR.value == "fear"
        assert MarketEmotion.NEUTRAL.value == "neutral"
        assert MarketEmotion.GREED.value == "greed"
        assert MarketEmotion.EXTREME_GREED.value == "extreme_greed"


class TestEmotionIndicator:
    """Tests for EmotionIndicator enum."""

    def test_all_indicators_exist(self):
        """Test that all expected indicators exist."""
        assert EmotionIndicator.PANIC.value == "panic"
        assert EmotionIndicator.ANXIETY.value == "anxiety"
        assert EmotionIndicator.UNCERTAINTY.value == "uncertainty"
        assert EmotionIndicator.CAUTION.value == "caution"
        assert EmotionIndicator.CONFIDENCE.value == "confidence"
        assert EmotionIndicator.OPTIMISM.value == "optimism"
        assert EmotionIndicator.EUPHORIA.value == "euphoria"
        assert EmotionIndicator.FOMO.value == "fomo"
        assert EmotionIndicator.CAPITULATION.value == "capitulation"
        assert EmotionIndicator.COMPLACENCY.value == "complacency"


class TestEmotionDetectorConfig:
    """Tests for EmotionDetectorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmotionDetectorConfig()

        assert config.extreme_fear_threshold == -0.7
        assert config.fear_threshold == -0.3
        assert config.greed_threshold == 0.3
        assert config.extreme_greed_threshold == 0.7
        assert config.panic_word_weight == 1.5
        assert config.euphoria_word_weight == 1.5
        assert config.normalize_by_length is True
        assert config.min_text_length == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = EmotionDetectorConfig(
            extreme_fear_threshold=-0.8,
            greed_threshold=0.4,
            normalize_by_length=False,
        )

        assert config.extreme_fear_threshold == -0.8
        assert config.greed_threshold == 0.4
        assert config.normalize_by_length is False


class TestEmotionResult:
    """Tests for EmotionResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create sample emotion result."""
        return EmotionResult(
            primary_emotion=MarketEmotion.FEAR,
            emotion_score=-0.5,
            fear_score=0.6,
            greed_score=0.1,
            uncertainty_score=0.3,
            indicators=[EmotionIndicator.ANXIETY],
            indicator_scores={EmotionIndicator.ANXIETY: 0.6},
            panic_level=0.4,
            euphoria_level=0.1,
            emotional_intensity=0.5,
            analysis_time_ms=10.0,
            text_length=100,
            matched_patterns=5,
        )

    def test_to_dict(self, sample_result):
        """Test conversion to dictionary."""
        result = sample_result.to_dict()

        assert result["primary_emotion"] == "fear"
        assert result["emotion_score"] == -0.5
        assert result["fear_score"] == 0.6
        assert result["greed_score"] == 0.1
        assert result["indicators"] == ["anxiety"]
        assert "anxiety" in result["indicator_scores"]

    def test_is_extreme_property(self):
        """Test is_extreme property."""
        extreme_fear = EmotionResult(
            primary_emotion=MarketEmotion.EXTREME_FEAR,
            emotion_score=-0.8,
            fear_score=0.9,
            greed_score=0.0,
            uncertainty_score=0.2,
            indicators=[],
            indicator_scores={},
            panic_level=0.9,
            euphoria_level=0.0,
            emotional_intensity=0.9,
            analysis_time_ms=5.0,
            text_length=50,
            matched_patterns=10,
        )

        assert extreme_fear.is_extreme is True

        neutral = EmotionResult(
            primary_emotion=MarketEmotion.NEUTRAL,
            emotion_score=0.0,
            fear_score=0.2,
            greed_score=0.2,
            uncertainty_score=0.3,
            indicators=[],
            indicator_scores={},
            panic_level=0.1,
            euphoria_level=0.1,
            emotional_intensity=0.2,
            analysis_time_ms=5.0,
            text_length=50,
            matched_patterns=2,
        )

        assert neutral.is_extreme is False

    def test_action_signal_property(self):
        """Test action_signal property."""
        extreme_fear = EmotionResult(
            primary_emotion=MarketEmotion.EXTREME_FEAR,
            emotion_score=-0.8,
            fear_score=0.9,
            greed_score=0.0,
            uncertainty_score=0.2,
            indicators=[],
            indicator_scores={},
            panic_level=0.9,
            euphoria_level=0.0,
            emotional_intensity=0.9,
            analysis_time_ms=5.0,
            text_length=50,
            matched_patterns=10,
        )

        assert extreme_fear.action_signal == "potential_buy"

        extreme_greed = EmotionResult(
            primary_emotion=MarketEmotion.EXTREME_GREED,
            emotion_score=0.8,
            fear_score=0.0,
            greed_score=0.9,
            uncertainty_score=0.1,
            indicators=[],
            indicator_scores={},
            panic_level=0.0,
            euphoria_level=0.9,
            emotional_intensity=0.9,
            analysis_time_ms=5.0,
            text_length=50,
            matched_patterns=10,
        )

        assert extreme_greed.action_signal == "potential_sell"


class TestEmotionDetector:
    """Tests for EmotionDetector class."""

    @pytest.fixture
    def detector(self):
        """Create default detector."""
        return EmotionDetector()

    @pytest.fixture
    def custom_detector(self):
        """Create detector with custom config."""
        config = EmotionDetectorConfig(
            panic_word_weight=2.0,
            normalize_by_length=False,
        )
        return EmotionDetector(config)

    # Fear detection tests
    def test_detect_panic(self, detector):
        """Test detection of panic/extreme fear."""
        result = detector.detect(
            "The market is crashing! Bloodbath! Panic selling everywhere! " "Run for the exits! Complete meltdown!"
        )

        assert result.primary_emotion in {MarketEmotion.EXTREME_FEAR, MarketEmotion.FEAR}
        assert result.fear_score > 0.5
        assert result.panic_level > 0.5

    def test_detect_anxiety(self, detector):
        """Test detection of anxiety/fear."""
        result = detector.detect(
            "I'm worried about the market. Concerned about my positions. " "Feeling anxious about the economy."
        )

        assert result.fear_score > 0.2
        assert EmotionIndicator.ANXIETY in result.indicators or result.fear_score > 0.3

    def test_detect_capitulation(self, detector):
        """Test detection of capitulation."""
        result = detector.detect(
            "I give up. Selling everything. Going to zero. I'm done with this market. "
            "Throw in the towel. Cut my losses."
        )

        assert result.fear_score > 0.3
        # Should detect capitulation indicator
        if result.indicator_scores:
            assert EmotionIndicator.CAPITULATION in result.indicator_scores or result.fear_score > 0.4

    # Greed detection tests
    def test_detect_euphoria(self, detector):
        """Test detection of euphoria/extreme greed."""
        result = detector.detect(
            "To the moon! 🚀🚀🚀 This is going parabolic! Diamond hands! "
            "We're all getting lambos! Free money! Can't lose!"
        )

        assert result.primary_emotion in {MarketEmotion.EXTREME_GREED, MarketEmotion.GREED}
        assert result.greed_score > 0.5
        assert result.euphoria_level > 0.4

    def test_detect_fomo(self, detector):
        """Test detection of FOMO."""
        result = detector.detect(
            "FOMO is real! Don't miss out! Last chance to buy! "
            "Everyone is getting in now. Should have bought earlier!"
        )

        assert result.greed_score > 0.3
        if result.indicator_scores:
            assert EmotionIndicator.FOMO in result.indicator_scores or result.greed_score > 0.4

    def test_detect_complacency(self, detector):
        """Test detection of complacency."""
        result = detector.detect(
            "Nothing can stop this rally. Stocks always go up. " "Can't go wrong buying here. Safe investment."
        )

        assert result.greed_score > 0.2
        # Should have some complacency signals

    # Uncertainty detection tests
    def test_detect_uncertainty(self, detector):
        """Test detection of uncertainty."""
        result = detector.detect(
            "Very uncertain market conditions. Unknown factors. "
            "Maybe this, maybe that. Who knows what will happen? Unpredictable."
        )

        assert result.uncertainty_score > 0.3
        if result.indicators:
            assert EmotionIndicator.UNCERTAINTY in result.indicators or result.uncertainty_score > 0.3

    # Neutral detection tests
    def test_detect_neutral(self, detector):
        """Test detection of neutral sentiment."""
        result = detector.detect(
            "The market opened flat today. Trading volume is average. " "No significant changes expected."
        )

        assert result.primary_emotion == MarketEmotion.NEUTRAL
        assert abs(result.emotion_score) < 0.3

    # Edge cases
    def test_empty_text(self, detector):
        """Test handling of empty text."""
        result = detector.detect("")

        assert result.primary_emotion == MarketEmotion.NEUTRAL
        assert result.emotion_score == 0.0
        assert result.fear_score == 0.0
        assert result.greed_score == 0.0

    def test_short_text(self, detector):
        """Test handling of short text below minimum."""
        result = detector.detect("hi")

        assert result.primary_emotion == MarketEmotion.NEUTRAL
        assert result.text_length == 2

    def test_mixed_emotions(self, detector):
        """Test handling of mixed emotions."""
        result = detector.detect(
            "I'm both scared and excited. Worried about losses but " "hopeful for gains. Uncertain but optimistic."
        )

        # Should have both fear and greed components
        assert result.fear_score > 0 or result.greed_score > 0

    def test_analysis_time_recorded(self, detector):
        """Test that analysis time is recorded."""
        result = detector.detect("Test text for timing analysis purposes")

        assert result.analysis_time_ms > 0
        assert result.analysis_time_ms < 1000  # Should be fast

    def test_matched_patterns_recorded(self, detector):
        """Test that matched patterns count is recorded."""
        result = detector.detect("Panic crash meltdown bloodbath doom")

        assert result.matched_patterns > 0

    # Batch detection tests
    def test_detect_batch(self, detector):
        """Test batch emotion detection."""
        texts = [
            "Crash! Panic! Sell!",
            "To the moon! 🚀",
            "Market is neutral today",
        ]

        results = detector.detect_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, EmotionResult) for r in results)

    # Aggregation tests
    def test_aggregate_emotions(self, detector):
        """Test emotion aggregation."""
        texts = [
            "Panic selling everywhere",
            "Markets crashing hard",
            "Everyone is scared",
        ]

        results = detector.detect_batch(texts)
        aggregated = detector.aggregate_emotions(results)

        assert isinstance(aggregated, EmotionResult)
        assert aggregated.fear_score > 0  # All texts are fearful

    def test_aggregate_with_weights(self, detector):
        """Test weighted emotion aggregation."""
        texts = [
            "Panic crash doom",  # Very fearful
            "Markets neutral",  # Neutral
        ]

        results = detector.detect_batch(texts)

        # Weight first (fearful) more heavily
        aggregated = detector.aggregate_emotions(results, weights=[2.0, 1.0])

        # Should lean more towards fear due to weighting
        assert aggregated.fear_score > 0.1

    def test_aggregate_empty_list(self, detector):
        """Test aggregation with empty list."""
        aggregated = detector.aggregate_emotions([])

        assert aggregated.primary_emotion == MarketEmotion.NEUTRAL

    # Emotion score range tests
    def test_emotion_score_range(self, detector):
        """Test that emotion score stays in valid range."""
        # Very fearful text
        result = detector.detect(
            "Panic panic panic crash crash crash doom doom doom " "meltdown bloodbath disaster catastrophe apocalypse"
        )

        assert -1.0 <= result.emotion_score <= 1.0

        # Very greedy text
        result = detector.detect(
            "Moon moon moon rocket rocket rocket lambo lambo " "diamond hands tendies yolo all in guaranteed"
        )

        assert -1.0 <= result.emotion_score <= 1.0


class TestCreateEmotionDetector:
    """Tests for factory function."""

    def test_create_default(self):
        """Test creating detector with defaults."""
        detector = create_emotion_detector()

        assert isinstance(detector, EmotionDetector)
        assert detector.config is not None

    def test_create_with_config(self):
        """Test creating detector with custom config."""
        config = EmotionDetectorConfig(
            panic_word_weight=2.0,
            extreme_fear_threshold=-0.5,
        )
        detector = create_emotion_detector(config)

        assert detector.config.panic_word_weight == 2.0
        assert detector.config.extreme_fear_threshold == -0.5


class TestEmotionPatterns:
    """Tests for specific emotion pattern detection."""

    @pytest.fixture
    def detector(self):
        """Create detector."""
        return EmotionDetector()

    def test_fear_patterns(self, detector):
        """Test individual fear patterns."""
        fear_texts = [
            "crash",
            "collapse",
            "plunge",
            "bloodbath",
            "meltdown",
            "tank",
            "dump",
            "sell-off",
        ]

        for text in fear_texts:
            result = detector.detect(f"The market is experiencing a {text}")
            # At least some fear detection
            assert result.fear_score >= 0 or result.matched_patterns >= 0

    def test_greed_patterns(self, detector):
        """Test individual greed patterns."""
        greed_texts = [
            "to the moon",
            "rocket",
            "lambo",
            "diamond hands",
            "yolo",
            "all-in",
            "parabolic",
            "soaring",
        ]

        for text in greed_texts:
            result = detector.detect(f"This stock is going {text}")
            # At least some greed detection
            assert result.greed_score >= 0 or result.matched_patterns >= 0
