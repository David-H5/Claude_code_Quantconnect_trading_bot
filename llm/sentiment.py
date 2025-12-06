"""
Sentiment Analysis Module

Provides financial sentiment analysis using FinBERT and other models.
"""

import logging
import time
from typing import Any

from .base import BaseSentimentAnalyzer, Sentiment, SentimentResult


logger = logging.getLogger(__name__)


class FinBERTSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Financial sentiment analyzer using FinBERT.

    FinBERT is fine-tuned on financial text and provides better
    accuracy for financial news and SEC filings.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize FinBERT analyzer.

        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of the model."""
        if self._initialized:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            model_name = self.config.get("model", "ProsusAI/finbert")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model,
                tokenizer=self._tokenizer,
                return_all_scores=True,
            )
            self._initialized = True
        except ImportError:
            raise ImportError("transformers package required. Install with: pip install transformers torch")

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult with classification
        """
        start_time = time.time()
        self._ensure_initialized()

        # Truncate text to model's max length
        text = text[:512]

        results = self._pipeline(text)[0]

        # Extract scores
        scores = {r["label"].lower(): r["score"] for r in results}
        positive = scores.get("positive", 0)
        negative = scores.get("negative", 0)
        neutral = scores.get("neutral", 0)

        # Determine sentiment
        if positive > negative and positive > neutral:
            if positive > 0.7:
                sentiment = Sentiment.VERY_BULLISH
            else:
                sentiment = Sentiment.BULLISH
            confidence = positive
        elif negative > positive and negative > neutral:
            if negative > 0.7:
                sentiment = Sentiment.VERY_BEARISH
            else:
                sentiment = Sentiment.BEARISH
            confidence = negative
        else:
            sentiment = Sentiment.NEUTRAL
            confidence = neutral

        # Calculate overall score (-1 to 1)
        score = positive - negative

        processing_time = (time.time() - start_time) * 1000

        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            score=score,
            provider="finbert",
            processing_time_ms=processing_time,
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """
        Analyze sentiment of multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentResult
        """
        return [self.analyze(text) for text in texts]


class SimpleSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Simple rule-based sentiment analyzer.

    Uses keyword matching for basic sentiment detection.
    Fallback when FinBERT is not available.
    """

    BULLISH_WORDS = {
        "surge",
        "soar",
        "rally",
        "gain",
        "jump",
        "rise",
        "up",
        "bullish",
        "strong",
        "beat",
        "exceed",
        "record",
        "high",
        "growth",
        "profit",
        "upgrade",
        "buy",
        "outperform",
        "positive",
        "optimistic",
        "boom",
    }

    BEARISH_WORDS = {
        "fall",
        "drop",
        "plunge",
        "decline",
        "slide",
        "down",
        "bearish",
        "weak",
        "miss",
        "below",
        "low",
        "loss",
        "crash",
        "downgrade",
        "sell",
        "underperform",
        "negative",
        "pessimistic",
        "bust",
        "slump",
    }

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment using keyword matching."""
        start_time = time.time()

        words = text.lower().split()
        bullish_count = sum(1 for w in words if w in self.BULLISH_WORDS)
        bearish_count = sum(1 for w in words if w in self.BEARISH_WORDS)

        total = bullish_count + bearish_count
        if total == 0:
            sentiment = Sentiment.NEUTRAL
            confidence = 0.5
            score = 0.0
        else:
            score = (bullish_count - bearish_count) / total
            confidence = abs(score)

            if score > 0.5:
                sentiment = Sentiment.VERY_BULLISH
            elif score > 0:
                sentiment = Sentiment.BULLISH
            elif score < -0.5:
                sentiment = Sentiment.VERY_BEARISH
            elif score < 0:
                sentiment = Sentiment.BEARISH
            else:
                sentiment = Sentiment.NEUTRAL

        processing_time = (time.time() - start_time) * 1000

        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            score=score,
            provider="simple",
            processing_time_ms=processing_time,
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment of multiple texts."""
        return [self.analyze(text) for text in texts]


def create_sentiment_analyzer(use_finbert: bool = True, config: dict[str, Any] | None = None) -> BaseSentimentAnalyzer:
    """
    Create appropriate sentiment analyzer.

    Args:
        use_finbert: Whether to use FinBERT (if available)
        config: Optional configuration

    Returns:
        Sentiment analyzer instance
    """
    if use_finbert:
        try:
            analyzer = FinBERTSentimentAnalyzer(config)
            # Test if it can initialize
            analyzer._ensure_initialized()
            return analyzer
        except ImportError as e:
            logger.info("FinBERT not available (missing dependencies): %s", e)
        except RuntimeError as e:
            logger.warning("FinBERT initialization failed: %s", e)
        except Exception as e:
            logger.error("Unexpected error initializing FinBERT: %s", e, exc_info=True)

    logger.info("Using SimpleSentimentAnalyzer as fallback")
    return SimpleSentimentAnalyzer()


__all__ = [
    "FinBERTSentimentAnalyzer",
    "SimpleSentimentAnalyzer",
    "create_sentiment_analyzer",
]
