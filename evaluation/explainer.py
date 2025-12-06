"""
SHAP Decision Explainer (UPGRADE-010 Sprint 1)

Provides explainable AI (XAI) capabilities for trading model decisions.
Integrates SHAP and LIME libraries for feature importance and explanation.

Features:
- SHAP explanations for tree-based models
- LIME explanations for any model
- Feature importance ranking
- Regulatory-compliant audit trail export
- Decision explanation visualization (text-based)

Part of UPGRADE-010: Advanced AI Features
Phase: 11 (Explainable AI)

QuantConnect Compatible: Yes
Dependencies: shap>=0.42.0, lime>=0.2.0 (optional)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


# Optional imports with fallbacks
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

try:
    import lime
    import lime.lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime = None

logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """Type of explanation method."""

    SHAP = "shap"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"
    PERMUTATION = "permutation"


class ModelType(Enum):
    """Type of model being explained."""

    TREE = "tree"  # Decision tree, Random Forest, XGBoost, etc.
    LINEAR = "linear"  # Linear/Logistic regression
    NEURAL_NETWORK = "neural_network"
    KERNEL = "kernel"  # SVM, other kernel methods
    UNKNOWN = "unknown"


@dataclass
class FeatureContribution:
    """Single feature's contribution to a prediction."""

    feature_name: str
    feature_value: float
    contribution: float  # SHAP value or similar
    rank: int  # 1 = most important
    direction: str = "positive"  # "positive" or "negative"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "feature_value": self.feature_value,
            "contribution": self.contribution,
            "rank": self.rank,
            "direction": self.direction,
        }


@dataclass
class Explanation:
    """Complete explanation for a model prediction."""

    explanation_id: str
    timestamp: datetime
    explanation_type: ExplanationType
    model_type: ModelType
    prediction: float
    base_value: float  # Expected value / intercept
    feature_contributions: list[FeatureContribution]
    total_contribution: float
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def top_features(self) -> list[FeatureContribution]:
        """Get top 5 most important features."""
        return sorted(self.feature_contributions, key=lambda x: abs(x.contribution), reverse=True)[:5]

    @property
    def positive_contributors(self) -> list[FeatureContribution]:
        """Get features pushing prediction higher."""
        return [f for f in self.feature_contributions if f.contribution > 0]

    @property
    def negative_contributors(self) -> list[FeatureContribution]:
        """Get features pushing prediction lower."""
        return [f for f in self.feature_contributions if f.contribution < 0]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "explanation_id": self.explanation_id,
            "timestamp": self.timestamp.isoformat(),
            "explanation_type": self.explanation_type.value,
            "model_type": self.model_type.value,
            "prediction": self.prediction,
            "base_value": self.base_value,
            "feature_contributions": [f.to_dict() for f in self.feature_contributions],
            "total_contribution": self.total_contribution,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    def to_text(self) -> str:
        """Generate human-readable explanation text."""
        lines = [
            f"=== Explanation ({self.explanation_type.value.upper()}) ===",
            f"Prediction: {self.prediction:.4f}",
            f"Base Value: {self.base_value:.4f}",
            "",
            "Top Contributing Features:",
        ]

        for fc in self.top_features:
            direction = "↑" if fc.contribution > 0 else "↓"
            lines.append(
                f"  {fc.rank}. {fc.feature_name}: {fc.contribution:+.4f} {direction}" f" (value={fc.feature_value:.4f})"
            )

        lines.append("")
        lines.append(f"Total Contribution: {self.total_contribution:+.4f}")

        return "\n".join(lines)


@dataclass
class ExplainerConfig:
    """Configuration for explainer."""

    # SHAP settings
    shap_background_samples: int = 100
    shap_max_evals: int = 500

    # LIME settings
    lime_num_features: int = 10
    lime_num_samples: int = 5000

    # General settings
    top_k_features: int = 10
    cache_explanations: bool = True
    auto_detect_model_type: bool = True


class BaseExplainer:
    """Base class for model explainers."""

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        config: ExplainerConfig | None = None,
    ):
        """
        Initialize explainer.

        Args:
            model: Trained model to explain
            feature_names: Names of input features
            config: Explainer configuration
        """
        self.model = model
        self.feature_names = feature_names
        self.config = config or ExplainerConfig()

        # Sprint 1.5: Enhanced caching
        self._explanation_cache: dict[str, Explanation] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._max_cache_size: int = 1000  # Prevent unbounded growth

    def explain(
        self,
        instance: np.ndarray,
        prediction: float | None = None,
    ) -> Explanation:
        """
        Generate explanation for a single prediction.

        Args:
            instance: Input features (1D array)
            prediction: Pre-computed prediction (optional)

        Returns:
            Explanation object
        """
        raise NotImplementedError

    def explain_batch(
        self,
        instances: np.ndarray,
    ) -> list[Explanation]:
        """
        Generate explanations for multiple predictions.

        Args:
            instances: Input features (2D array)

        Returns:
            List of Explanation objects
        """
        return [self.explain(instance) for instance in instances]

    def _generate_explanation_id(self, instance: np.ndarray) -> str:
        """Generate unique explanation ID."""
        import hashlib

        content = f"{datetime.utcnow().isoformat()}:{instance.tobytes()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # =========================================================================
    # Sprint 1.5: Caching Methods
    # =========================================================================

    def _generate_cache_key(self, instance: np.ndarray) -> str:
        """
        Generate cache key from instance features.

        Sprint 1.5: Cache key is based on feature values, not timestamp,
        so identical inputs return cached results.
        """
        import hashlib

        # Round to avoid floating point precision issues
        rounded = np.round(instance.flatten(), decimals=6)
        content = rounded.tobytes()
        return hashlib.sha256(content).hexdigest()[:32]

    def _get_cached(self, cache_key: str) -> Explanation | None:
        """
        Get explanation from cache if available.

        Sprint 1.5: Returns cached explanation or None.
        """
        if not self.config.cache_explanations:
            return None

        if cache_key in self._explanation_cache:
            self._cache_hits += 1
            return self._explanation_cache[cache_key]

        self._cache_misses += 1
        return None

    def _cache_explanation(self, cache_key: str, explanation: Explanation) -> None:
        """
        Store explanation in cache.

        Sprint 1.5: Implements LRU-style eviction when cache is full.
        """
        if not self.config.cache_explanations:
            return

        # Evict oldest entries if cache is full
        if len(self._explanation_cache) >= self._max_cache_size:
            # Remove first 10% of entries (simple FIFO eviction)
            keys_to_remove = list(self._explanation_cache.keys())[: self._max_cache_size // 10]
            for key in keys_to_remove:
                del self._explanation_cache[key]

        self._explanation_cache[cache_key] = explanation

    def get_cache_statistics(self) -> dict[str, Any]:
        """
        Get cache performance statistics.

        Sprint 1.5: Returns cache hit rate and size metrics.
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_enabled": self.config.cache_explanations,
            "cache_size": len(self._explanation_cache),
            "max_cache_size": self._max_cache_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }

    def clear_cache(self) -> int:
        """
        Clear the explanation cache.

        Sprint 1.5: Returns number of entries cleared.
        """
        count = len(self._explanation_cache)
        self._explanation_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        return count

    # =========================================================================
    # End Sprint 1.5 Caching Methods
    # =========================================================================


class SHAPExplainer(BaseExplainer):
    """
    SHAP-based explainer for tree and general models.

    Uses TreeExplainer for tree-based models (fast) and
    KernelExplainer for general models (slower but universal).
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        background_data: np.ndarray | None = None,
        model_type: ModelType = ModelType.UNKNOWN,
        config: ExplainerConfig | None = None,
    ):
        super().__init__(model, feature_names, config)

        if not SHAP_AVAILABLE:
            raise ImportError("shap package not installed. Install with: pip install shap")

        self.model_type = model_type
        self.background_data = background_data
        self._shap_explainer: Any | None = None

        # Initialize SHAP explainer based on model type
        self._init_shap_explainer()

    def _init_shap_explainer(self) -> None:
        """Initialize appropriate SHAP explainer."""
        if self.model_type == ModelType.TREE:
            try:
                self._shap_explainer = shap.TreeExplainer(self.model)
                logger.info("Initialized SHAP TreeExplainer")
            except Exception as e:
                logger.warning(f"TreeExplainer failed, falling back to Kernel: {e}")
                self._init_kernel_explainer()
        else:
            self._init_kernel_explainer()

    def _init_kernel_explainer(self) -> None:
        """Initialize SHAP KernelExplainer."""
        if self.background_data is None:
            logger.warning("No background data provided for KernelExplainer")
            return

        try:
            # Use a subset of background data
            n_samples = min(self.config.shap_background_samples, len(self.background_data))
            background = shap.sample(self.background_data, n_samples)

            self._shap_explainer = shap.KernelExplainer(
                self.model.predict if hasattr(self.model, "predict") else self.model,
                background,
            )
            logger.info("Initialized SHAP KernelExplainer")
        except Exception as e:
            logger.error(f"Failed to initialize KernelExplainer: {e}")

    def explain(
        self,
        instance: np.ndarray,
        prediction: float | None = None,
    ) -> Explanation:
        """Generate SHAP explanation for instance."""
        if self._shap_explainer is None:
            raise RuntimeError("SHAP explainer not initialized")

        # Ensure 2D array
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        # Sprint 1.5: Check cache first
        cache_key = self._generate_cache_key(instance)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Get SHAP values
        try:
            shap_values = self._shap_explainer.shap_values(instance)
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            raise

        # Handle multi-output case (take first output)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = shap_values.flatten()

        # Get base value
        if hasattr(self._shap_explainer, "expected_value"):
            base_value = self._shap_explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
            else:
                base_value = float(base_value)
        else:
            base_value = 0.0

        # Get prediction if not provided
        if prediction is None:
            if hasattr(self.model, "predict"):
                prediction = float(self.model.predict(instance)[0])
            else:
                prediction = base_value + sum(shap_values)

        # Create feature contributions
        contributions = []
        for name, value, shap_val in zip(self.feature_names, instance.flatten(), shap_values):
            contributions.append(
                FeatureContribution(
                    feature_name=name,
                    feature_value=float(value),
                    contribution=float(shap_val),
                    rank=0,  # Will be set after sorting
                    direction="positive" if shap_val > 0 else "negative",
                )
            )

        # Sort and assign ranks
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        for i, c in enumerate(contributions):
            c.rank = i + 1

        explanation = Explanation(
            explanation_id=self._generate_explanation_id(instance),
            timestamp=datetime.utcnow(),
            explanation_type=ExplanationType.SHAP,
            model_type=self.model_type,
            prediction=prediction,
            base_value=base_value,
            feature_contributions=contributions,
            total_contribution=sum(shap_values),
        )

        # Sprint 1.5: Cache the result
        self._cache_explanation(cache_key, explanation)

        return explanation


class LIMEExplainer(BaseExplainer):
    """
    LIME-based explainer for any model type.

    Creates local linear approximations to explain predictions.
    Works with any model that has a predict method.
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        training_data: np.ndarray,
        mode: str = "regression",  # "regression" or "classification"
        config: ExplainerConfig | None = None,
    ):
        super().__init__(model, feature_names, config)

        if not LIME_AVAILABLE:
            raise ImportError("lime package not installed. Install with: pip install lime")

        self.mode = mode
        self.training_data = training_data

        # Initialize LIME explainer
        self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            mode=mode,
            verbose=False,
        )

    def explain(
        self,
        instance: np.ndarray,
        prediction: float | None = None,
    ) -> Explanation:
        """Generate LIME explanation for instance."""
        # Ensure 1D array
        if instance.ndim > 1:
            instance = instance.flatten()

        # Sprint 1.5: Check cache first
        cache_key = self._generate_cache_key(instance)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Get predict function
        predict_fn = self.model.predict if hasattr(self.model, "predict") else self.model

        # For classification, we need predict_proba
        if self.mode == "classification" and hasattr(self.model, "predict_proba"):
            predict_fn = self.model.predict_proba

        # Generate LIME explanation
        try:
            lime_exp = self._lime_explainer.explain_instance(
                instance,
                predict_fn,
                num_features=self.config.lime_num_features,
                num_samples=self.config.lime_num_samples,
            )
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            raise

        # Get prediction if not provided
        if prediction is None:
            if hasattr(self.model, "predict"):
                pred = self.model.predict(instance.reshape(1, -1))
                prediction = float(pred[0])
            else:
                prediction = 0.0

        # Extract feature contributions
        contributions = []
        lime_map = dict(lime_exp.as_list())

        for i, name in enumerate(self.feature_names):
            # LIME returns feature ranges, match by feature name prefix
            contrib = 0.0
            for lime_key, lime_val in lime_map.items():
                if name in lime_key:
                    contrib = lime_val
                    break

            contributions.append(
                FeatureContribution(
                    feature_name=name,
                    feature_value=float(instance[i]) if i < len(instance) else 0.0,
                    contribution=contrib,
                    rank=0,
                    direction="positive" if contrib > 0 else "negative",
                )
            )

        # Sort and assign ranks
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        for i, c in enumerate(contributions):
            c.rank = i + 1

        # Get intercept as base value
        base_value = lime_exp.intercept[0] if hasattr(lime_exp, "intercept") else 0.0

        explanation = Explanation(
            explanation_id=self._generate_explanation_id(instance),
            timestamp=datetime.utcnow(),
            explanation_type=ExplanationType.LIME,
            model_type=ModelType.UNKNOWN,
            prediction=prediction,
            base_value=base_value,
            feature_contributions=contributions,
            total_contribution=sum(c.contribution for c in contributions),
        )

        # Sprint 1.5: Cache the result
        self._cache_explanation(cache_key, explanation)

        return explanation


class FeatureImportanceExplainer(BaseExplainer):
    """
    Simple feature importance explainer.

    Uses model's built-in feature importances if available,
    otherwise computes permutation importance.
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        config: ExplainerConfig | None = None,
    ):
        super().__init__(model, feature_names, config)

        # Get feature importances from model if available
        self._importances: np.ndarray | None = None

        if hasattr(model, "feature_importances_"):
            self._importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            self._importances = np.abs(model.coef_).flatten()

    def explain(
        self,
        instance: np.ndarray,
        prediction: float | None = None,
    ) -> Explanation:
        """Generate feature importance explanation."""
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        # Sprint 1.5: Check cache first
        cache_key = self._generate_cache_key(instance)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Get prediction
        if prediction is None and hasattr(self.model, "predict"):
            prediction = float(self.model.predict(instance)[0])
        else:
            prediction = prediction or 0.0

        # Create contributions based on importance * feature_value
        contributions = []

        if self._importances is not None:
            for i, (name, importance) in enumerate(zip(self.feature_names, self._importances)):
                value = float(instance.flatten()[i]) if i < instance.size else 0.0
                # Contribution = importance weighted by normalized value
                contrib = float(importance)

                contributions.append(
                    FeatureContribution(
                        feature_name=name,
                        feature_value=value,
                        contribution=contrib,
                        rank=0,
                        direction="positive",  # Importance is always positive
                    )
                )
        else:
            # No importances available, use equal weights
            for i, name in enumerate(self.feature_names):
                value = float(instance.flatten()[i]) if i < instance.size else 0.0
                contributions.append(
                    FeatureContribution(
                        feature_name=name,
                        feature_value=value,
                        contribution=1.0 / len(self.feature_names),
                        rank=0,
                        direction="positive",
                    )
                )

        # Sort and assign ranks
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        for i, c in enumerate(contributions):
            c.rank = i + 1

        explanation = Explanation(
            explanation_id=self._generate_explanation_id(instance),
            timestamp=datetime.utcnow(),
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            model_type=ModelType.UNKNOWN,
            prediction=prediction,
            base_value=0.0,
            feature_contributions=contributions,
            total_contribution=sum(c.contribution for c in contributions),
        )

        # Sprint 1.5: Cache the result
        self._cache_explanation(cache_key, explanation)

        return explanation


class ExplainerFactory:
    """Factory for creating appropriate explainer based on model type."""

    @staticmethod
    def create(
        model: Any,
        feature_names: list[str],
        explanation_type: ExplanationType = ExplanationType.SHAP,
        training_data: np.ndarray | None = None,
        model_type: ModelType = ModelType.UNKNOWN,
        config: ExplainerConfig | None = None,
    ) -> BaseExplainer:
        """
        Create appropriate explainer for model.

        Args:
            model: Trained model to explain
            feature_names: Names of input features
            explanation_type: Type of explanation method
            training_data: Training data for background (needed for LIME/Kernel SHAP)
            model_type: Type of model
            config: Explainer configuration

        Returns:
            Appropriate explainer instance
        """
        # Auto-detect model type if needed
        if model_type == ModelType.UNKNOWN:
            model_type = ExplainerFactory._detect_model_type(model)

        if explanation_type == ExplanationType.SHAP:
            if not SHAP_AVAILABLE:
                logger.warning("SHAP not available, falling back to feature importance")
                return FeatureImportanceExplainer(model, feature_names, config)

            return SHAPExplainer(
                model=model,
                feature_names=feature_names,
                background_data=training_data,
                model_type=model_type,
                config=config,
            )

        elif explanation_type == ExplanationType.LIME:
            if not LIME_AVAILABLE:
                logger.warning("LIME not available, falling back to feature importance")
                return FeatureImportanceExplainer(model, feature_names, config)

            if training_data is None:
                raise ValueError("LIME requires training_data")

            return LIMEExplainer(
                model=model,
                feature_names=feature_names,
                training_data=training_data,
                config=config,
            )

        elif explanation_type == ExplanationType.FEATURE_IMPORTANCE:
            return FeatureImportanceExplainer(model, feature_names, config)

        else:
            raise ValueError(f"Unknown explanation type: {explanation_type}")

    @staticmethod
    def _detect_model_type(model: Any) -> ModelType:
        """Auto-detect model type from model object."""
        model_class = type(model).__name__.lower()

        tree_keywords = ["tree", "forest", "xgb", "lgb", "catboost", "gradient"]
        linear_keywords = ["linear", "logistic", "ridge", "lasso", "elastic"]
        nn_keywords = ["neural", "mlp", "keras", "torch", "tensorflow"]

        for keyword in tree_keywords:
            if keyword in model_class:
                return ModelType.TREE

        for keyword in linear_keywords:
            if keyword in model_class:
                return ModelType.LINEAR

        for keyword in nn_keywords:
            if keyword in model_class:
                return ModelType.NEURAL_NETWORK

        return ModelType.UNKNOWN


class ExplanationLogger:
    """Logger for explanations with audit trail export."""

    def __init__(
        self,
        storage_dir: str = "explanations",
        auto_persist: bool = True,
    ):
        self.storage_dir = storage_dir
        self.auto_persist = auto_persist
        self._explanations: list[Explanation] = []

        if auto_persist:
            os.makedirs(storage_dir, exist_ok=True)

    def log(self, explanation: Explanation) -> None:
        """Log an explanation."""
        self._explanations.append(explanation)

        if self.auto_persist:
            self._persist(explanation)

    def _persist(self, explanation: Explanation) -> None:
        """Persist explanation to disk."""
        filepath = os.path.join(self.storage_dir, f"{explanation.explanation_id}.json")
        with open(filepath, "w") as f:
            json.dump(explanation.to_dict(), f, indent=2)

    def export_audit_trail(
        self,
        filepath: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        """
        Export explanations as regulatory audit trail.

        Args:
            filepath: Output file path
            start_time: Filter start time
            end_time: Filter end time

        Returns:
            Number of explanations exported
        """
        explanations = self._explanations

        if start_time:
            explanations = [e for e in explanations if e.timestamp >= start_time]
        if end_time:
            explanations = [e for e in explanations if e.timestamp <= end_time]

        audit_trail = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_explanations": len(explanations),
            "shap_available": SHAP_AVAILABLE,
            "lime_available": LIME_AVAILABLE,
            "entries": [e.to_dict() for e in explanations],
        }

        with open(filepath, "w") as f:
            json.dump(audit_trail, f, indent=2)

        return len(explanations)

    def get_explanations(
        self,
        limit: int = 100,
        explanation_type: ExplanationType | None = None,
    ) -> list[Explanation]:
        """Get recent explanations."""
        explanations = self._explanations

        if explanation_type:
            explanations = [e for e in explanations if e.explanation_type == explanation_type]

        return explanations[-limit:]


def create_shap_explainer(
    model: Any,
    feature_names: list[str],
    background_data: np.ndarray | None = None,
    model_type: ModelType = ModelType.UNKNOWN,
) -> SHAPExplainer:
    """Factory function to create SHAP explainer."""
    return SHAPExplainer(
        model=model,
        feature_names=feature_names,
        background_data=background_data,
        model_type=model_type,
    )


def create_lime_explainer(
    model: Any,
    feature_names: list[str],
    training_data: np.ndarray,
    mode: str = "regression",
) -> LIMEExplainer:
    """Factory function to create LIME explainer."""
    return LIMEExplainer(
        model=model,
        feature_names=feature_names,
        training_data=training_data,
        mode=mode,
    )


def create_explainer(
    model: Any,
    feature_names: list[str],
    explanation_type: ExplanationType = ExplanationType.SHAP,
    training_data: np.ndarray | None = None,
) -> BaseExplainer:
    """Factory function to create appropriate explainer."""
    return ExplainerFactory.create(
        model=model,
        feature_names=feature_names,
        explanation_type=explanation_type,
        training_data=training_data,
    )
