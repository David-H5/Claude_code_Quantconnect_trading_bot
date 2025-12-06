"""
Emotion Detection Layer Module

Detects fear/greed and other market emotions beyond simple positive/negative.
Part of UPGRADE-010 Sprint 3 Expansion - Intelligence & Data Sources.
"""

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class MarketEmotion(Enum):
    """Primary market emotions."""

    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


class EmotionIndicator(Enum):
    """Specific emotion indicators detected in text."""

    PANIC = "panic"
    ANXIETY = "anxiety"
    UNCERTAINTY = "uncertainty"
    CAUTION = "caution"
    CONFIDENCE = "confidence"
    OPTIMISM = "optimism"
    EUPHORIA = "euphoria"
    FOMO = "fomo"  # Fear of missing out
    CAPITULATION = "capitulation"
    DENIAL = "denial"
    HOPE = "hope"
    COMPLACENCY = "complacency"


@dataclass
class EmotionResult:
    """Result of emotion detection analysis."""

    # Primary emotion
    primary_emotion: MarketEmotion
    emotion_score: float  # -1 (extreme fear) to +1 (extreme greed)

    # Component scores
    fear_score: float  # 0-1
    greed_score: float  # 0-1
    uncertainty_score: float  # 0-1

    # Specific indicators detected
    indicators: list[EmotionIndicator]
    indicator_scores: dict[EmotionIndicator, float]

    # Panic/euphoria levels (0-1)
    panic_level: float
    euphoria_level: float

    # Urgency derived from emotional intensity
    emotional_intensity: float  # 0-1, how strong the emotion is

    # Analysis metadata
    analysis_time_ms: float
    text_length: int
    matched_patterns: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_emotion": self.primary_emotion.value,
            "emotion_score": self.emotion_score,
            "fear_score": self.fear_score,
            "greed_score": self.greed_score,
            "uncertainty_score": self.uncertainty_score,
            "indicators": [i.value for i in self.indicators],
            "indicator_scores": {k.value: v for k, v in self.indicator_scores.items()},
            "panic_level": self.panic_level,
            "euphoria_level": self.euphoria_level,
            "emotional_intensity": self.emotional_intensity,
            "analysis_time_ms": self.analysis_time_ms,
            "text_length": self.text_length,
            "matched_patterns": self.matched_patterns,
        }

    @property
    def is_extreme(self) -> bool:
        """Check if emotion is at extreme level."""
        return self.primary_emotion in {
            MarketEmotion.EXTREME_FEAR,
            MarketEmotion.EXTREME_GREED,
        }

    @property
    def action_signal(self) -> str:
        """Get contrarian action signal based on emotion."""
        if self.primary_emotion == MarketEmotion.EXTREME_FEAR:
            return "potential_buy"  # Contrarian
        elif self.primary_emotion == MarketEmotion.EXTREME_GREED:
            return "potential_sell"  # Contrarian
        return "hold"


@dataclass
class EmotionDetectorConfig:
    """Configuration for emotion detector."""

    # Thresholds
    extreme_fear_threshold: float = -0.7
    fear_threshold: float = -0.3
    greed_threshold: float = 0.3
    extreme_greed_threshold: float = 0.7

    # Weights for different pattern categories
    panic_word_weight: float = 1.5
    euphoria_word_weight: float = 1.5
    hedge_word_weight: float = 0.8

    # Normalization
    normalize_by_length: bool = True
    min_text_length: int = 10


class EmotionDetector:
    """
    Detects market emotions in text beyond simple positive/negative sentiment.

    Identifies:
    - Fear/greed spectrum
    - Panic and euphoria indicators
    - FOMO, capitulation, complacency
    - Emotional intensity for urgency assessment
    """

    # Fear/panic patterns
    FEAR_PATTERNS: dict[str, float] = {
        r"\bpanic\b": 1.0,
        r"\bcrash(ing|ed)?\b": 0.9,
        r"\bcollaps(e|ing|ed)\b": 0.9,
        r"\bplunge(d|s|ing)?\b": 0.8,
        r"\bbloodbath\b": 1.0,
        r"\bfreefall\b": 0.9,
        r"\bmeltdown\b": 0.9,
        r"\btank(ing|ed|s)?\b": 0.7,
        r"\bdump(ing|ed)?\b": 0.7,
        r"\bsell[-\s]?off\b": 0.6,
        r"\bfear\b": 0.5,
        r"\bworr(y|ied|ies)\b": 0.4,
        r"\bconcern(ed|s)?\b": 0.4,
        r"\banxi(ous|ety)\b": 0.5,
        r"\bscared\b": 0.6,
        r"\bterrified\b": 0.8,
        r"\brun for the exits\b": 0.9,
        r"\bget out now\b": 0.9,
        r"\bdisaster\b": 0.7,
        r"\bcatastroph(e|ic)\b": 0.8,
        r"\bapocalyp(se|tic)\b": 0.9,
        r"\bdoom(ed)?\b": 0.7,
        r"\brecession\b": 0.5,
        r"\bdepression\b": 0.6,
        r"\bbear market\b": 0.5,
        r"\bdefault\b": 0.6,
        r"\bbankrupt(cy)?\b": 0.7,
        r"\binsolven(t|cy)\b": 0.7,
    }

    # Greed/euphoria patterns
    GREED_PATTERNS: dict[str, float] = {
        r"\bto the moon\b": 1.0,
        r"\bmooning\b": 0.9,
        r"\brocket(ing)?\b": 0.8,
        r"\b🚀+\b": 0.8,
        r"\blambo\b": 0.9,
        r"\btendies\b": 0.8,
        r"\bdiamond hands?\b": 0.7,
        r"\bhodl\b": 0.6,
        r"\bfomo\b": 0.8,
        r"\ball[- ]?in\b": 0.7,
        r"\byolo\b": 0.8,
        r"\bget rich\b": 0.7,
        r"\bfree money\b": 0.8,
        r"\bcan'?t lose\b": 0.9,
        r"\bguaranteed\b": 0.7,
        r"\bsure thing\b": 0.8,
        r"\beasy money\b": 0.8,
        r"\bprint(ing)? money\b": 0.7,
        r"\beuphori(a|c)\b": 0.9,
        r"\bbull(ish)? (run|market)\b": 0.5,
        r"\bexcite(d|ment)\b": 0.4,
        r"\boptimis(m|tic)\b": 0.4,
        r"\bconfiden(t|ce)\b": 0.3,
        r"\bsoar(ing|ed|s)?\b": 0.6,
        r"\bsurg(e|ing|ed)\b": 0.5,
        r"\bskyrocket\b": 0.7,
        r"\bexplod(e|ing)\b": 0.6,
        r"\bparabolic\b": 0.8,
        r"\bmania\b": 0.8,
        r"\bbubble\b": 0.6,  # Can be fear or greed context
    }

    # Uncertainty/hedge patterns
    UNCERTAINTY_PATTERNS: dict[str, float] = {
        r"\buncertain(ty)?\b": 0.6,
        r"\bunknown\b": 0.5,
        r"\bunpredict(able)?\b": 0.6,
        r"\bvolati(le|lity)\b": 0.5,
        r"\bconfus(ed|ing|ion)\b": 0.5,
        r"\bmixed signals?\b": 0.6,
        r"\bunclear\b": 0.5,
        r"\bmight\b": 0.3,
        r"\bmaybe\b": 0.3,
        r"\bperhaps\b": 0.3,
        r"\bcould\b": 0.2,
        r"\bpossibly\b": 0.3,
        r"\buncertain\b": 0.6,
        r"\bwait and see\b": 0.5,
        r"\bsidelines?\b": 0.4,
        r"\bcauti(on|ous)\b": 0.4,
        r"\bhedg(e|ing)\b": 0.4,
        r"\brisk(y|s)?\b": 0.4,
    }

    # Capitulation patterns (giving up)
    CAPITULATION_PATTERNS: dict[str, float] = {
        r"\bgive up\b": 0.8,
        r"\bsurrender\b": 0.8,
        r"\bcapitulat(e|ion)\b": 1.0,
        r"\bthrow in the towel\b": 0.9,
        r"\bcut (my|your|our) losses\b": 0.7,
        r"\bselling everything\b": 0.9,
        r"\bgoing to zero\b": 0.8,
        r"\bi'?m out\b": 0.6,
        r"\bdone with\b": 0.5,
        r"\bnever (again|recover)\b": 0.7,
    }

    # FOMO patterns
    FOMO_PATTERNS: dict[str, float] = {
        r"\bfomo\b": 1.0,
        r"\bmissing out\b": 0.8,
        r"\blast chance\b": 0.7,
        r"\bdon'?t miss\b": 0.7,
        r"\bbefore it'?s too late\b": 0.8,
        r"\bjump(ing)? in\b": 0.5,
        r"\bgetting in now\b": 0.6,
        r"\bwish i (had )?bought\b": 0.7,
        r"\bshould have bought\b": 0.7,
        r"\beveryone (is )?(buying|getting)\b": 0.6,
    }

    # Complacency patterns
    COMPLACENCY_PATTERNS: dict[str, float] = {
        r"\bnothing (can|will) stop\b": 0.8,
        r"\balways (goes|go) up\b": 0.9,
        r"\bcan'?t go wrong\b": 0.8,
        r"\bno way (it )?(goes )?down\b": 0.8,
        r"\bsafe (bet|investment)\b": 0.6,
        r"\bworry free\b": 0.7,
        r"\bset and forget\b": 0.5,
        r"\beasy\b": 0.3,
        r"\brelax(ed)?\b": 0.3,
    }

    def __init__(self, config: EmotionDetectorConfig | None = None):
        """
        Initialize emotion detector.

        Args:
            config: Detection configuration
        """
        self.config = config or EmotionDetectorConfig()

        # Compile patterns
        self._fear_patterns = self._compile_patterns(self.FEAR_PATTERNS)
        self._greed_patterns = self._compile_patterns(self.GREED_PATTERNS)
        self._uncertainty_patterns = self._compile_patterns(self.UNCERTAINTY_PATTERNS)
        self._capitulation_patterns = self._compile_patterns(self.CAPITULATION_PATTERNS)
        self._fomo_patterns = self._compile_patterns(self.FOMO_PATTERNS)
        self._complacency_patterns = self._compile_patterns(self.COMPLACENCY_PATTERNS)

    def _compile_patterns(
        self,
        patterns: dict[str, float],
    ) -> list[tuple[re.Pattern, float]]:
        """Compile regex patterns with weights."""
        return [(re.compile(pattern, re.IGNORECASE), weight) for pattern, weight in patterns.items()]

    def detect(self, text: str) -> EmotionResult:
        """
        Detect emotions in text.

        Args:
            text: Text to analyze

        Returns:
            EmotionResult with detected emotions
        """
        start_time = time.time()

        if len(text) < self.config.min_text_length:
            return self._create_neutral_result(text, start_time)

        text_lower = text.lower()

        # Calculate component scores
        fear_raw, fear_matches = self._calculate_pattern_score(text_lower, self._fear_patterns)
        greed_raw, greed_matches = self._calculate_pattern_score(text_lower, self._greed_patterns)
        uncertainty_raw, uncertainty_matches = self._calculate_pattern_score(text_lower, self._uncertainty_patterns)

        # Calculate specific indicator scores
        capitulation_score, cap_matches = self._calculate_pattern_score(text_lower, self._capitulation_patterns)
        fomo_score, fomo_matches = self._calculate_pattern_score(text_lower, self._fomo_patterns)
        complacency_score, comp_matches = self._calculate_pattern_score(text_lower, self._complacency_patterns)

        total_matches = fear_matches + greed_matches + uncertainty_matches + cap_matches + fomo_matches + comp_matches

        # Normalize scores
        if self.config.normalize_by_length:
            word_count = max(1, len(text.split()))
            normalization_factor = min(1.0, word_count / 100)
        else:
            normalization_factor = 1.0

        fear_score = min(1.0, fear_raw * normalization_factor * self.config.panic_word_weight)
        greed_score = min(1.0, greed_raw * normalization_factor * self.config.euphoria_word_weight)
        uncertainty_score = min(1.0, uncertainty_raw * normalization_factor * self.config.hedge_word_weight)

        # Calculate composite emotion score (-1 to +1)
        # Negative = fear, Positive = greed
        if fear_score + greed_score > 0:
            emotion_score = (greed_score - fear_score) / (fear_score + greed_score + 0.1)
        else:
            emotion_score = 0.0

        # Adjust for uncertainty (pushes toward neutral)
        emotion_score *= 1.0 - uncertainty_score * 0.5

        # Determine primary emotion
        primary_emotion = self._score_to_emotion(emotion_score)

        # Calculate panic and euphoria levels
        panic_level = min(1.0, fear_score + capitulation_score * 0.5)
        euphoria_level = min(1.0, greed_score + fomo_score * 0.3 + complacency_score * 0.2)

        # Detect specific indicators
        indicators, indicator_scores = self._detect_indicators(
            fear_score=fear_score,
            greed_score=greed_score,
            uncertainty_score=uncertainty_score,
            capitulation_score=capitulation_score,
            fomo_score=fomo_score,
            complacency_score=complacency_score,
        )

        # Calculate emotional intensity
        emotional_intensity = max(panic_level, euphoria_level, abs(emotion_score))

        analysis_time_ms = (time.time() - start_time) * 1000

        return EmotionResult(
            primary_emotion=primary_emotion,
            emotion_score=emotion_score,
            fear_score=fear_score,
            greed_score=greed_score,
            uncertainty_score=uncertainty_score,
            indicators=indicators,
            indicator_scores=indicator_scores,
            panic_level=panic_level,
            euphoria_level=euphoria_level,
            emotional_intensity=emotional_intensity,
            analysis_time_ms=analysis_time_ms,
            text_length=len(text),
            matched_patterns=total_matches,
        )

    def _calculate_pattern_score(
        self,
        text: str,
        patterns: list[tuple[re.Pattern, float]],
    ) -> tuple[float, int]:
        """Calculate score for a set of patterns."""
        total_score = 0.0
        match_count = 0

        for pattern, weight in patterns:
            matches = pattern.findall(text)
            if matches:
                match_count += len(matches)
                total_score += weight * len(matches)

        return total_score, match_count

    def _score_to_emotion(self, score: float) -> MarketEmotion:
        """Convert emotion score to MarketEmotion."""
        if score <= self.config.extreme_fear_threshold:
            return MarketEmotion.EXTREME_FEAR
        elif score <= self.config.fear_threshold:
            return MarketEmotion.FEAR
        elif score >= self.config.extreme_greed_threshold:
            return MarketEmotion.EXTREME_GREED
        elif score >= self.config.greed_threshold:
            return MarketEmotion.GREED
        else:
            return MarketEmotion.NEUTRAL

    def _detect_indicators(
        self,
        fear_score: float,
        greed_score: float,
        uncertainty_score: float,
        capitulation_score: float,
        fomo_score: float,
        complacency_score: float,
    ) -> tuple[list[EmotionIndicator], dict[EmotionIndicator, float]]:
        """Detect specific emotion indicators."""
        indicators = []
        scores: dict[EmotionIndicator, float] = {}

        # Panic
        if fear_score > 0.7:
            indicators.append(EmotionIndicator.PANIC)
            scores[EmotionIndicator.PANIC] = fear_score

        # Anxiety
        if 0.3 < fear_score <= 0.7:
            indicators.append(EmotionIndicator.ANXIETY)
            scores[EmotionIndicator.ANXIETY] = fear_score

        # Uncertainty
        if uncertainty_score > 0.4:
            indicators.append(EmotionIndicator.UNCERTAINTY)
            scores[EmotionIndicator.UNCERTAINTY] = uncertainty_score

        # Caution
        if 0.2 < uncertainty_score <= 0.4:
            indicators.append(EmotionIndicator.CAUTION)
            scores[EmotionIndicator.CAUTION] = uncertainty_score

        # Confidence
        if 0.3 < greed_score <= 0.6 and fear_score < 0.2:
            indicators.append(EmotionIndicator.CONFIDENCE)
            scores[EmotionIndicator.CONFIDENCE] = greed_score

        # Optimism
        if 0.2 < greed_score <= 0.5:
            indicators.append(EmotionIndicator.OPTIMISM)
            scores[EmotionIndicator.OPTIMISM] = greed_score

        # Euphoria
        if greed_score > 0.7:
            indicators.append(EmotionIndicator.EUPHORIA)
            scores[EmotionIndicator.EUPHORIA] = greed_score

        # FOMO
        if fomo_score > 0.3:
            indicators.append(EmotionIndicator.FOMO)
            scores[EmotionIndicator.FOMO] = fomo_score

        # Capitulation
        if capitulation_score > 0.3:
            indicators.append(EmotionIndicator.CAPITULATION)
            scores[EmotionIndicator.CAPITULATION] = capitulation_score

        # Complacency
        if complacency_score > 0.3:
            indicators.append(EmotionIndicator.COMPLACENCY)
            scores[EmotionIndicator.COMPLACENCY] = complacency_score

        # Hope (low fear, some greed, during fearful market)
        if 0.1 < greed_score < 0.4 and fear_score > 0.3:
            indicators.append(EmotionIndicator.HOPE)
            scores[EmotionIndicator.HOPE] = greed_score

        return indicators, scores

    def _create_neutral_result(
        self,
        text: str,
        start_time: float,
    ) -> EmotionResult:
        """Create neutral result for short/empty text."""
        return EmotionResult(
            primary_emotion=MarketEmotion.NEUTRAL,
            emotion_score=0.0,
            fear_score=0.0,
            greed_score=0.0,
            uncertainty_score=0.0,
            indicators=[],
            indicator_scores={},
            panic_level=0.0,
            euphoria_level=0.0,
            emotional_intensity=0.0,
            analysis_time_ms=(time.time() - start_time) * 1000,
            text_length=len(text),
            matched_patterns=0,
        )

    def detect_batch(self, texts: list[str]) -> list[EmotionResult]:
        """
        Detect emotions in multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of EmotionResults
        """
        return [self.detect(text) for text in texts]

    def aggregate_emotions(
        self,
        results: list[EmotionResult],
        weights: list[float] | None = None,
    ) -> EmotionResult:
        """
        Aggregate multiple emotion results.

        Args:
            results: List of EmotionResults
            weights: Optional weights for each result

        Returns:
            Aggregated EmotionResult
        """
        if not results:
            return self._create_neutral_result("", time.time())

        if weights is None:
            weights = [1.0] * len(results)

        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1.0

        # Weighted averages
        avg_emotion_score = sum(r.emotion_score * w for r, w in zip(results, weights)) / total_weight
        avg_fear = sum(r.fear_score * w for r, w in zip(results, weights)) / total_weight
        avg_greed = sum(r.greed_score * w for r, w in zip(results, weights)) / total_weight
        avg_uncertainty = sum(r.uncertainty_score * w for r, w in zip(results, weights)) / total_weight
        avg_panic = sum(r.panic_level * w for r, w in zip(results, weights)) / total_weight
        avg_euphoria = sum(r.euphoria_level * w for r, w in zip(results, weights)) / total_weight
        avg_intensity = sum(r.emotional_intensity * w for r, w in zip(results, weights)) / total_weight

        # Aggregate indicators
        all_indicators: dict[EmotionIndicator, float] = {}
        for result in results:
            for indicator, score in result.indicator_scores.items():
                if indicator not in all_indicators:
                    all_indicators[indicator] = 0.0
                all_indicators[indicator] += score

        # Normalize indicator scores
        if results:
            all_indicators = {k: v / len(results) for k, v in all_indicators.items()}

        return EmotionResult(
            primary_emotion=self._score_to_emotion(avg_emotion_score),
            emotion_score=avg_emotion_score,
            fear_score=avg_fear,
            greed_score=avg_greed,
            uncertainty_score=avg_uncertainty,
            indicators=list(all_indicators.keys()),
            indicator_scores=all_indicators,
            panic_level=avg_panic,
            euphoria_level=avg_euphoria,
            emotional_intensity=avg_intensity,
            analysis_time_ms=sum(r.analysis_time_ms for r in results),
            text_length=sum(r.text_length for r in results),
            matched_patterns=sum(r.matched_patterns for r in results),
        )


def create_emotion_detector(
    config: EmotionDetectorConfig | None = None,
) -> EmotionDetector:
    """
    Factory function to create an emotion detector.

    Args:
        config: Optional configuration

    Returns:
        Configured EmotionDetector instance
    """
    return EmotionDetector(config)
