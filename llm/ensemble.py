"""
LLM Ensemble Module

Combines outputs from multiple LLM providers using weighted averaging
or voting strategies for more robust predictions.

UPGRADE-014 Enhancements (December 2025):
- Dynamic weight adjustment based on provider performance
- Boosting-style weight updates
- Performance tracking and calibration
- Research: Ensemble of medium LLMs beats single large model by 18.6% RMSE

Research Sources:
- Sentiment Trading with LLMs (ScienceDirect 2024)
- LLM Ensemble Strategies (arXiv 2025)
- Generating Effective Ensembles (arXiv Feb 2024)
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .base import AnalysisResult, BaseLLMProvider, NewsItem, Sentiment, SentimentResult
from .providers import create_provider
from .sentiment import create_sentiment_analyzer


logger = logging.getLogger(__name__)


# ==============================================================================
# UPGRADE-014: Performance Tracking for Dynamic Weighting
# ==============================================================================


@dataclass
class ProviderPerformance:
    """Track provider performance for dynamic weighting (UPGRADE-014)."""

    provider_name: str
    total_predictions: int = 0
    correct_predictions: int = 0
    total_error: float = 0.0
    weighted_accuracy: float = 0.5  # Running weighted accuracy
    calibration_score: float = 1.0  # How well-calibrated confidence is
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def accuracy(self) -> float:
        """Get prediction accuracy."""
        if self.total_predictions == 0:
            return 0.5
        return self.correct_predictions / self.total_predictions

    @property
    def avg_error(self) -> float:
        """Get average prediction error."""
        if self.total_predictions == 0:
            return 0.5
        return self.total_error / self.total_predictions

    def update_accuracy(self, was_correct: bool, decay: float = 0.95) -> None:
        """Update weighted accuracy with exponential decay."""
        self.total_predictions += 1
        if was_correct:
            self.correct_predictions += 1

        # Exponential moving average for weighted accuracy
        actual = 1.0 if was_correct else 0.0
        self.weighted_accuracy = decay * self.weighted_accuracy + (1 - decay) * actual
        self.last_updated = datetime.now()

    def update_error(self, predicted: float, actual: float, decay: float = 0.95) -> None:
        """Update with prediction error (for regression-style updates)."""
        error = abs(predicted - actual)
        self.total_error += error
        self.total_predictions += 1
        self.last_updated = datetime.now()

    def update_calibration(self, confidence: float, was_correct: bool) -> None:
        """Update calibration score based on confidence vs outcome."""
        # Perfect calibration: high confidence = high accuracy
        expected_correct = confidence
        actual_correct = 1.0 if was_correct else 0.0
        calibration_error = abs(expected_correct - actual_correct)

        # Exponential moving average
        self.calibration_score = 0.95 * self.calibration_score + 0.05 * (1 - calibration_error)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider_name": self.provider_name,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "accuracy": self.accuracy,
            "weighted_accuracy": self.weighted_accuracy,
            "avg_error": self.avg_error,
            "calibration_score": self.calibration_score,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class DynamicWeightConfig:
    """Configuration for dynamic weighting (UPGRADE-014)."""

    # Weight adjustment parameters
    learning_rate: float = 0.1  # How fast weights adjust
    min_weight: float = 0.05  # Minimum weight per provider
    max_weight: float = 0.6  # Maximum weight per provider
    decay_factor: float = 0.95  # Exponential decay for running averages

    # Boosting parameters
    enable_boosting: bool = True
    boost_correct_factor: float = 1.1  # Multiply weight on correct prediction
    penalize_incorrect_factor: float = 0.9  # Multiply weight on incorrect

    # Rebalancing parameters
    rebalance_interval_hours: int = 24
    min_samples_for_adjustment: int = 10


@dataclass
class EnsembleResult:
    """Result from ensemble analysis."""

    sentiment: SentimentResult
    individual_results: list[SentimentResult] = field(default_factory=list)
    agreement_score: float = 0.0  # How much providers agree
    timestamp: datetime = field(default_factory=datetime.now)
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction_id": self.prediction_id,
            "sentiment": self.sentiment.to_dict(),
            "individual_results": [r.to_dict() for r in self.individual_results],
            "agreement_score": self.agreement_score,
            "timestamp": self.timestamp.isoformat(),
        }


class LLMEnsemble:
    """
    Ensemble of LLM providers for robust analysis.

    Combines FinBERT (fast, specialized) with GPT-4o and Claude
    using weighted averaging for sentiment and voting for signals.

    UPGRADE-014 Features:
    - Dynamic weight adjustment based on provider performance
    - Boosting-style weight updates after feedback
    - Performance tracking per provider
    - Calibration monitoring
    """

    def __init__(
        self,
        config: dict[str, Any],
        dynamic_weight_config: DynamicWeightConfig | None = None,
    ):
        """
        Initialize ensemble with configuration.

        Args:
            config: LLM configuration from settings.json
            dynamic_weight_config: Optional dynamic weighting config (UPGRADE-014)
        """
        self.config = config
        self.providers: dict[str, BaseLLMProvider] = {}
        self.weights: dict[str, float] = config.get("weights", {})
        self.strategy = config.get("ensemble_strategy", "weighted_average")

        # UPGRADE-014: Dynamic weighting
        self.dynamic_config = dynamic_weight_config or DynamicWeightConfig()
        self.enable_dynamic_weights = config.get("enable_dynamic_weights", True)
        self._performance: dict[str, ProviderPerformance] = {}
        self._base_weights: dict[str, float] = dict(self.weights)  # Original weights
        self._last_rebalance: datetime = datetime.now()
        self._prediction_history: list[dict[str, Any]] = []
        self._max_history = 500
        self._lock = threading.Lock()

        # Initialize providers
        self._init_providers()

        # Initialize sentiment analyzer (FinBERT or fallback)
        finbert_config = config.get("providers", {}).get("finbert", {})
        self.sentiment_analyzer = create_sentiment_analyzer(
            use_finbert=finbert_config.get("local", True),
            config=finbert_config,
        )

        # Initialize performance tracking for all providers
        self._init_performance_tracking()

    def _init_providers(self) -> None:
        """Initialize available LLM providers."""
        providers_config = self.config.get("providers", {})

        for name, provider_config in providers_config.items():
            if name == "finbert":
                continue  # Handled separately

            provider = create_provider(name, provider_config)
            if provider and provider.is_available:
                self.providers[name] = provider

    def _init_performance_tracking(self) -> None:
        """Initialize performance tracking for all providers (UPGRADE-014)."""
        # Track FinBERT
        self._performance["finbert"] = ProviderPerformance(provider_name="finbert")

        # Track other providers
        for name in self.providers:
            self._performance[name] = ProviderPerformance(provider_name=name)

        # Set initial weights if not configured
        if not self.weights:
            n_providers = len(self._performance)
            for name in self._performance:
                self.weights[name] = 1.0 / n_providers
                self._base_weights[name] = 1.0 / n_providers

    # =========================================================================
    # UPGRADE-014: Dynamic Weighting Methods
    # =========================================================================

    def record_prediction_outcome(
        self,
        prediction_id: str,
        actual_sentiment: float,
        actual_direction_correct: bool,
    ) -> None:
        """
        Record the actual outcome for a prediction (UPGRADE-014).

        Used to update provider weights based on accuracy.

        Args:
            prediction_id: ID from a previous EnsembleResult
            actual_sentiment: Actual sentiment outcome (-1 to 1)
            actual_direction_correct: Whether direction (bull/bear) was correct
        """
        with self._lock:
            # Find prediction in history
            prediction = None
            for p in self._prediction_history:
                if p.get("id") == prediction_id:
                    prediction = p
                    break

            if prediction is None:
                logger.warning(f"Prediction {prediction_id} not found in history")
                return

            # Update each provider's performance
            for result in prediction.get("individual_results", []):
                provider_name = result.get("provider", "unknown")
                if provider_name not in self._performance:
                    continue

                perf = self._performance[provider_name]
                predicted_score = result.get("score", 0)
                confidence = result.get("confidence", 0.5)

                # Update accuracy
                perf.update_accuracy(actual_direction_correct)

                # Update error
                perf.update_error(predicted_score, actual_sentiment)

                # Update calibration
                perf.update_calibration(confidence, actual_direction_correct)

            # Apply boosting if enabled
            if self.dynamic_config.enable_boosting:
                self._apply_boosting(prediction, actual_direction_correct)

            # Check if rebalancing is needed
            self._check_rebalance()

    def _apply_boosting(
        self,
        prediction: dict[str, Any],
        was_correct: bool,
    ) -> None:
        """Apply boosting-style weight adjustment (UPGRADE-014)."""
        for result in prediction.get("individual_results", []):
            provider_name = result.get("provider", "unknown")
            if provider_name not in self.weights:
                continue

            predicted_direction = result.get("score", 0) > 0

            # Check if this provider was correct
            provider_correct = predicted_direction == was_correct

            if provider_correct:
                self.weights[provider_name] *= self.dynamic_config.boost_correct_factor
            else:
                self.weights[provider_name] *= self.dynamic_config.penalize_incorrect_factor

        # Normalize weights and apply bounds
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0 and apply bounds."""
        total = sum(self.weights.values())
        if total == 0:
            # Reset to uniform
            for name in self.weights:
                self.weights[name] = 1.0 / len(self.weights)
            return

        # Apply bounds and normalize
        for name in self.weights:
            self.weights[name] = max(
                self.dynamic_config.min_weight,
                min(self.dynamic_config.max_weight, self.weights[name] / total),
            )

        # Re-normalize after bounds
        total = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total

    def _check_rebalance(self) -> None:
        """Check if weights should be rebalanced based on performance."""
        hours_since = (datetime.now() - self._last_rebalance).total_seconds() / 3600

        if hours_since < self.dynamic_config.rebalance_interval_hours:
            return

        # Check if we have enough samples
        total_samples = sum(p.total_predictions for p in self._performance.values())
        if total_samples < self.dynamic_config.min_samples_for_adjustment:
            return

        self._rebalance_weights()

    def _rebalance_weights(self) -> None:
        """Rebalance weights based on accumulated performance (UPGRADE-014)."""
        # Calculate performance-based weights
        performance_weights = {}
        total_perf = 0.0

        for name, perf in self._performance.items():
            if name not in self.weights:
                continue

            # Combined score: weighted accuracy * calibration
            score = perf.weighted_accuracy * perf.calibration_score
            performance_weights[name] = score
            total_perf += score

        if total_perf == 0:
            return

        # Blend performance weights with current weights
        learning_rate = self.dynamic_config.learning_rate
        for name in self.weights:
            if name in performance_weights:
                perf_weight = performance_weights[name] / total_perf

                # Exponential moving average update
                self.weights[name] = (1 - learning_rate) * self.weights[name] + learning_rate * perf_weight

        self._normalize_weights()
        self._last_rebalance = datetime.now()

        logger.info(f"Rebalanced weights: {self.weights}")

    def get_provider_performance(self) -> dict[str, dict[str, Any]]:
        """Get performance metrics for all providers (UPGRADE-014)."""
        return {name: perf.to_dict() for name, perf in self._performance.items()}

    def get_dynamic_weights(self) -> dict[str, float]:
        """Get current dynamic weights."""
        return dict(self.weights)

    def reset_weights(self) -> None:
        """Reset weights to base configuration (UPGRADE-014)."""
        self.weights = dict(self._base_weights)
        for perf in self._performance.values():
            perf.total_predictions = 0
            perf.correct_predictions = 0
            perf.total_error = 0.0
            perf.weighted_accuracy = 0.5
            perf.calibration_score = 1.0

    def set_weight(self, provider_name: str, weight: float) -> None:
        """Manually set weight for a provider."""
        if provider_name in self.weights:
            self.weights[provider_name] = weight
            self._normalize_weights()

    def analyze_sentiment(self, text: str) -> EnsembleResult:
        """
        Analyze sentiment using all available providers.

        Args:
            text: Text to analyze

        Returns:
            EnsembleResult with combined sentiment and prediction_id for feedback
        """
        results: list[SentimentResult] = []

        # Always use FinBERT first (fastest)
        try:
            finbert_result = self.sentiment_analyzer.analyze(text)
            results.append(finbert_result)
        except (ValueError, RuntimeError) as e:
            logger.warning("FinBERT analysis failed: %s", e)
        except Exception as e:
            logger.error("Unexpected error in FinBERT analysis: %s", e, exc_info=True)

        # Query other providers
        for name, provider in self.providers.items():
            try:
                result = provider.analyze_sentiment(text)
                results.append(result)
            except (ValueError, ConnectionError, TimeoutError) as e:
                logger.warning("Provider %s sentiment analysis failed: %s", name, e)
                continue
            except Exception as e:
                logger.error("Unexpected error from provider %s: %s", name, e, exc_info=True)
                continue

        if not results:
            # Fallback result
            return EnsembleResult(
                sentiment=SentimentResult(
                    sentiment=Sentiment.NEUTRAL,
                    confidence=0.0,
                    score=0.0,
                    provider="ensemble",
                ),
                individual_results=[],
                agreement_score=0.0,
            )

        # Combine results based on strategy
        combined = self._combine_sentiment_results(results)

        # Calculate agreement score
        agreement = self._calculate_agreement(results)

        # Generate prediction ID for tracking
        prediction_id = str(uuid.uuid4())

        # Create ensemble result
        ensemble_result = EnsembleResult(
            sentiment=combined,
            individual_results=results,
            agreement_score=agreement,
            prediction_id=prediction_id,
        )

        # UPGRADE-014: Record prediction for dynamic weighting feedback
        if self.enable_dynamic_weights:
            self._record_prediction(prediction_id, text, results, combined)

        return ensemble_result

    def _record_prediction(
        self,
        prediction_id: str,
        text: str,
        individual_results: list[SentimentResult],
        combined: SentimentResult,
    ) -> None:
        """Record prediction for later feedback (UPGRADE-014)."""
        with self._lock:
            prediction_record = {
                "id": prediction_id,
                "timestamp": datetime.now().isoformat(),
                "text_hash": hash(text[:100]),  # Hash for privacy
                "combined_score": combined.score,
                "combined_confidence": combined.confidence,
                "individual_results": [
                    {
                        "provider": r.provider,
                        "score": r.score,
                        "confidence": r.confidence,
                        "sentiment": r.sentiment.name,
                    }
                    for r in individual_results
                ],
            }

            self._prediction_history.append(prediction_record)

            # Trim history if too large
            if len(self._prediction_history) > self._max_history:
                self._prediction_history = self._prediction_history[-self._max_history :]

    def _combine_sentiment_results(self, results: list[SentimentResult]) -> SentimentResult:
        """Combine multiple sentiment results."""
        if self.strategy == "weighted_average":
            return self._weighted_average(results)
        elif self.strategy == "majority_vote":
            return self._majority_vote(results)
        else:
            return self._weighted_average(results)

    def _weighted_average(self, results: list[SentimentResult]) -> SentimentResult:
        """Combine using weighted average of scores."""
        total_weight = 0.0
        weighted_score = 0.0
        weighted_confidence = 0.0

        for result in results:
            weight = self.weights.get(result.provider, 1.0 / len(results))
            weighted_score += result.score * weight * result.confidence
            weighted_confidence += result.confidence * weight
            total_weight += weight

        if total_weight > 0:
            final_score = weighted_score / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_score = 0.0
            final_confidence = 0.0

        # Determine sentiment from score
        if final_score > 0.5:
            sentiment = Sentiment.VERY_BULLISH
        elif final_score > 0.1:
            sentiment = Sentiment.BULLISH
        elif final_score < -0.5:
            sentiment = Sentiment.VERY_BEARISH
        elif final_score < -0.1:
            sentiment = Sentiment.BEARISH
        else:
            sentiment = Sentiment.NEUTRAL

        return SentimentResult(
            sentiment=sentiment,
            confidence=final_confidence,
            score=final_score,
            provider="ensemble",
        )

    def _majority_vote(self, results: list[SentimentResult]) -> SentimentResult:
        """Combine using majority voting."""
        votes: dict[Sentiment, float] = {}

        for result in results:
            weight = self.weights.get(result.provider, 1.0)
            votes[result.sentiment] = votes.get(result.sentiment, 0) + weight

        if not votes:
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                confidence=0.0,
                score=0.0,
                provider="ensemble",
            )

        # Get majority sentiment
        majority_sentiment = max(votes, key=lambda k: votes[k])
        total_votes = sum(votes.values())
        confidence = votes[majority_sentiment] / total_votes if total_votes > 0 else 0

        # Calculate average score for the majority
        matching_results = [r for r in results if r.sentiment == majority_sentiment]
        avg_score = sum(r.score for r in matching_results) / len(matching_results)

        return SentimentResult(
            sentiment=majority_sentiment,
            confidence=confidence,
            score=avg_score,
            provider="ensemble",
        )

    def _calculate_agreement(self, results: list[SentimentResult]) -> float:
        """Calculate how much providers agree (0-1)."""
        if len(results) < 2:
            return 1.0

        # Calculate variance in scores
        scores: list[float] = [r.score for r in results]
        mean_score: float = sum(scores) / len(scores)
        variance: float = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        # Convert variance to agreement (lower variance = higher agreement)
        # Max variance for scores in [-1, 1] is 1.0
        agreement: float = 1.0 - min(variance, 1.0)

        return agreement

    def analyze_news(self, news_items: list[NewsItem]) -> AnalysisResult:
        """
        Analyze news using available providers.

        Uses the first available provider (Claude preferred for deep analysis).
        """
        # Prefer Claude for news analysis (better reasoning)
        preferred_order = ["anthropic", "openai"]

        for provider_name in preferred_order:
            if provider_name in self.providers:
                try:
                    return self.providers[provider_name].analyze_news(news_items)
                except (ValueError, ConnectionError, TimeoutError) as e:
                    logger.warning("Provider %s news analysis failed: %s", provider_name, e)
                    continue
                except Exception as e:
                    logger.error(
                        "Unexpected error from provider %s during news analysis: %s",
                        provider_name,
                        e,
                        exc_info=True,
                    )
                    continue

        # Fallback result
        return AnalysisResult(
            summary="No LLM providers available for news analysis",
            key_points=[],
            trading_signals=[],
            risk_factors=["No LLM analysis available"],
            confidence=0.0,
            provider="none",
        )

    def analyze_option_chain(self, symbol: str, chain_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Analyze option chain for underpriced options.

        Aggregates recommendations from multiple providers.
        """
        all_recommendations: list[dict[str, Any]] = []

        for name, provider in self.providers.items():
            try:
                recommendations = provider.analyze_option_chain(symbol, chain_data)
                for rec in recommendations:
                    rec["provider"] = name
                all_recommendations.extend(recommendations)
            except (ValueError, ConnectionError, TimeoutError) as e:
                logger.warning("Provider %s option chain analysis failed: %s", name, e)
                continue
            except Exception as e:
                logger.error(
                    "Unexpected error from provider %s during option analysis: %s",
                    name,
                    e,
                    exc_info=True,
                )
                continue

        # Sort by confidence and underpriced percentage
        all_recommendations.sort(
            key=lambda x: (x.get("confidence", 0) * x.get("underpriced_pct", 0)),
            reverse=True,
        )

        return all_recommendations

    def get_provider_status(self) -> dict[str, bool]:
        """Get availability status of all providers."""
        status = {"finbert": True}  # Always available (with fallback)

        for name, provider in self.providers.items():
            status[name] = provider.is_available

        return status


def create_ensemble(config: dict[str, Any] | None = None) -> LLMEnsemble:
    """
    Create LLM ensemble from configuration.

    Args:
        config: LLM configuration (from settings.json llm_integration section)

    Returns:
        Configured LLMEnsemble instance
    """
    if config is None:
        # Default configuration
        config = {
            "enabled": True,
            "providers": {
                "finbert": {"local": True},
            },
            "weights": {"finbert": 1.0},
            "ensemble_strategy": "weighted_average",
        }

    return LLMEnsemble(config)


__all__ = [
    "DynamicWeightConfig",
    "EnsembleResult",
    "LLMEnsemble",
    "ProviderPerformance",
    "create_ensemble",
]
