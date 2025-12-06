"""
ML Fill Probability Model

Machine learning model for predicting option spread fill probability.
Enhances rule-based fill_predictor.py with learned patterns from
historical fill data.

Part of UPGRADE-010 Sprint 4: Risk & Execution.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from models.exceptions import DataValidationError


logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available ML model types."""

    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"


class MarketRegime(Enum):
    """Market volatility regime."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class FillFeatures:
    """
    Feature vector for ML fill prediction.

    Contains all features used to predict fill probability.
    """

    spread_bps: float  # Bid-ask spread in basis points
    volume_ratio: float  # Order size vs avg daily volume
    time_of_day: float  # Normalized 0-1 (9:30=0, 16:00=1)
    day_of_week: int  # 0-4 (Mon-Fri)
    volatility_rank: float  # Current IV percentile (0-1)
    delta: float  # Option delta (-1 to 1)
    days_to_expiry: int  # DTE
    underlying_move: float  # Underlying % move today
    vix_level: float  # VIX value
    market_regime: MarketRegime = MarketRegime.NORMAL
    num_legs: int = 1  # Number of legs in spread
    order_placement: float = 0.5  # 0=bid, 0.5=mid, 1=ask

    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for model input."""
        regime_map = {
            MarketRegime.LOW: 0,
            MarketRegime.NORMAL: 1,
            MarketRegime.HIGH: 2,
            MarketRegime.EXTREME: 3,
        }

        return np.array(
            [
                self.spread_bps,
                self.volume_ratio,
                self.time_of_day,
                self.day_of_week,
                self.volatility_rank,
                self.delta,
                self.days_to_expiry,
                self.underlying_move,
                self.vix_level,
                regime_map.get(self.market_regime, 1),
                self.num_legs,
                self.order_placement,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spread_bps": self.spread_bps,
            "volume_ratio": self.volume_ratio,
            "time_of_day": self.time_of_day,
            "day_of_week": self.day_of_week,
            "volatility_rank": self.volatility_rank,
            "delta": self.delta,
            "days_to_expiry": self.days_to_expiry,
            "underlying_move": self.underlying_move,
            "vix_level": self.vix_level,
            "market_regime": self.market_regime.value,
            "num_legs": self.num_legs,
            "order_placement": self.order_placement,
        }


@dataclass
class TrainingRecord:
    """Historical fill record for training."""

    features: FillFeatures
    filled: bool  # True if filled, False otherwise
    fill_time_seconds: float | None = None  # Time to fill if filled
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingResult:
    """Result of model training."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    num_samples: int
    num_features: int
    cross_val_scores: list[float]
    training_time_seconds: float
    feature_importance: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "num_samples": self.num_samples,
            "num_features": self.num_features,
            "cross_val_mean": np.mean(self.cross_val_scores),
            "cross_val_std": np.std(self.cross_val_scores),
            "training_time_seconds": self.training_time_seconds,
            "feature_importance": self.feature_importance,
        }


@dataclass
class MLFillPrediction:
    """ML model fill prediction result."""

    fill_probability: float  # 0-1
    confidence: float  # Model confidence in prediction
    expected_fill_time: float  # Expected seconds to fill
    model_version: str
    features_used: dict[str, float]
    top_contributing_features: list[tuple[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fill_probability": self.fill_probability,
            "confidence": self.confidence,
            "expected_fill_time": self.expected_fill_time,
            "model_version": self.model_version,
            "features_used": self.features_used,
            "top_features": [{"feature": f, "importance": i} for f, i in self.top_contributing_features],
        }


class FillMLModel:
    """
    Machine learning model for fill probability prediction.

    Uses gradient boosting by default for best performance on
    tabular data with mixed feature types.
    """

    FEATURE_NAMES = [
        "spread_bps",
        "volume_ratio",
        "time_of_day",
        "day_of_week",
        "volatility_rank",
        "delta",
        "days_to_expiry",
        "underlying_move",
        "vix_level",
        "market_regime",
        "num_legs",
        "order_placement",
    ]

    def __init__(
        self,
        model_type: ModelType = ModelType.GRADIENT_BOOSTING,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize ML fill model.

        Args:
            model_type: Type of ML model to use
            n_estimators: Number of trees (for ensemble methods)
            max_depth: Maximum tree depth
            learning_rate: Learning rate (for gradient boosting)
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model = None
        self.scaler = None
        self.is_trained = False
        self.version = "1.0.0"
        self._feature_importance: dict[str, float] = {}

        # Initialize model based on type
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the underlying ML model."""
        # Use simple implementations that don't require sklearn
        # In production, would use sklearn.ensemble.GradientBoostingClassifier
        self.model = SimpleGradientBoosting(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
        )
        self.scaler = SimpleStandardScaler()

    def train(
        self,
        records: list[TrainingRecord],
        validation_split: float = 0.2,
        n_folds: int = 5,
    ) -> TrainingResult:
        """
        Train the model on historical fill data.

        Args:
            records: List of historical fill records
            validation_split: Fraction for validation set
            n_folds: Number of cross-validation folds

        Returns:
            TrainingResult with metrics
        """
        import time

        start_time = time.time()

        if len(records) < 10:
            raise DataValidationError(
                field="records",
                value=len(records),
                reason="Need at least 10 records for training",
            )

        # Prepare data
        X = np.array([r.features.to_array() for r in records])
        y = np.array([1 if r.filled else 0 for r in records])

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_val)
        y_prob = self.model.predict_proba(X_val)

        # Calculate metrics
        accuracy = np.mean(y_pred == y_val)
        precision = self._calculate_precision(y_val, y_pred)
        recall = self._calculate_recall(y_val, y_pred)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        auc = self._calculate_auc(y_val, y_prob)

        # Cross-validation scores (simplified)
        cv_scores = self._cross_validate(X_scaled, y, n_folds)

        # Feature importance
        self._feature_importance = dict(zip(self.FEATURE_NAMES, self.model.feature_importances_))

        training_time = time.time() - start_time

        return TrainingResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc,
            num_samples=len(records),
            num_features=len(self.FEATURE_NAMES),
            cross_val_scores=cv_scores,
            training_time_seconds=training_time,
            feature_importance=self._feature_importance,
        )

    def predict(self, features: FillFeatures) -> MLFillPrediction:
        """
        Predict fill probability for given features.

        Args:
            features: Feature vector for prediction

        Returns:
            MLFillPrediction with probability and confidence
        """
        if not self.is_trained:
            # Return default prediction if not trained
            return self._default_prediction(features)

        # Prepare features
        X = features.to_array().reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Get prediction and probability
        prob = self.model.predict_proba(X_scaled)[0]
        fill_probability = float(prob)

        # Confidence based on distance from 0.5
        confidence = abs(fill_probability - 0.5) * 2  # 0-1 scale

        # Expected fill time (estimated)
        if fill_probability > 0.5:
            # Higher probability = faster expected fill
            expected_fill_time = (1 - fill_probability) * 30  # 0-30 seconds
        else:
            expected_fill_time = 30 + (0.5 - fill_probability) * 60  # 30-60+ seconds

        # Top contributing features
        top_features = sorted(self._feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        return MLFillPrediction(
            fill_probability=fill_probability,
            confidence=confidence,
            expected_fill_time=expected_fill_time,
            model_version=self.version,
            features_used=features.to_dict(),
            top_contributing_features=top_features,
        )

    def _default_prediction(self, features: FillFeatures) -> MLFillPrediction:
        """Return rule-based prediction when model not trained."""
        # Simple rule-based fallback
        prob = 0.5

        # Adjust for spread
        if features.spread_bps < 10:
            prob += 0.15
        elif features.spread_bps > 50:
            prob -= 0.15

        # Adjust for order placement
        prob += (features.order_placement - 0.5) * 0.3

        # Adjust for legs
        prob -= (features.num_legs - 1) * 0.1

        prob = max(0.1, min(0.9, prob))

        return MLFillPrediction(
            fill_probability=prob,
            confidence=0.3,  # Low confidence for default
            expected_fill_time=15.0,
            model_version="default",
            features_used=features.to_dict(),
            top_contributing_features=[],
        )

    def feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            return {}
        return self._feature_importance.copy()

    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision."""
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0

    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall."""
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / actual_positives if actual_positives > 0 else 0.0

    def _calculate_auc(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate AUC-ROC (simplified)."""
        # Simplified AUC calculation
        sorted_indices = np.argsort(y_prob)[::-1]
        y_sorted = y_true[sorted_indices]

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        # Count concordant pairs
        auc = 0.0
        cum_pos = 0
        for i, y in enumerate(y_sorted):
            if y == 1:
                cum_pos += 1
            else:
                auc += cum_pos

        return auc / (n_pos * n_neg)

    def _cross_validate(self, X: np.ndarray, y: np.ndarray, n_folds: int) -> list[float]:
        """Simple cross-validation."""
        fold_size = len(X) // n_folds
        scores = []

        for i in range(n_folds):
            start = i * fold_size
            end = start + fold_size

            X_val = X[start:end]
            y_val = y[start:end]
            X_train = np.concatenate([X[:start], X[end:]])
            y_train = np.concatenate([y[:start], y[end:]])

            # Train temporary model
            temp_model = SimpleGradientBoosting(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
            )
            temp_model.fit(X_train, y_train)

            # Evaluate
            y_pred = temp_model.predict(X_val)
            accuracy = np.mean(y_pred == y_val)
            scores.append(accuracy)

        return scores

    def save(self, path: str) -> None:
        """Save model to file."""
        import json

        model_data = {
            "version": self.version,
            "model_type": self.model_type.value,
            "feature_importance": self._feature_importance,
            "is_trained": self.is_trained,
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
            },
        }

        with open(path, "w") as f:
            json.dump(model_data, f, indent=2)

    def load(self, path: str) -> None:
        """Load model from file."""
        import json

        with open(path) as f:
            model_data = json.load(f)

        self.version = model_data.get("version", "1.0.0")
        self._feature_importance = model_data.get("feature_importance", {})
        self.is_trained = model_data.get("is_trained", False)


class SimpleStandardScaler:
    """Simple standard scaler implementation."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "SimpleStandardScaler":
        """Fit scaler to data."""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1  # Avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data."""
        if self.mean_ is None:
            return X
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)


class SimpleGradientBoosting:
    """
    Simplified gradient boosting implementation.

    For production use, replace with sklearn.ensemble.GradientBoostingClassifier.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees: list[SimpleDecisionStump] = []
        self.feature_importances_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleGradientBoosting":
        """Fit the model."""
        n_samples, n_features = X.shape
        self.feature_importances_ = np.zeros(n_features)

        # Initialize with average
        self.initial_prediction = np.mean(y)

        # Current predictions
        predictions = np.full(n_samples, self.initial_prediction)

        for _ in range(self.n_estimators):
            # Calculate residuals
            residuals = y - self._sigmoid(predictions)

            # Fit tree to residuals
            tree = SimpleDecisionStump(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update predictions
            tree_pred = tree.predict(X)
            predictions += self.learning_rate * tree_pred

            # Update feature importance
            self.feature_importances_[tree.best_feature] += abs(tree.best_threshold)

        # Normalize feature importance
        total = np.sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ /= total

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        predictions = np.full(len(X), self.initial_prediction)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        return self._sigmoid(predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class SimpleDecisionStump:
    """Simple decision stump for gradient boosting."""

    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth
        self.best_feature: int = 0
        self.best_threshold: float = 0.0
        self.left_value: float = 0.0
        self.right_value: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleDecisionStump":
        """Fit the stump."""
        n_samples, n_features = X.shape
        best_mse = float("inf")

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_mean = np.mean(y[left_mask])
                right_mean = np.mean(y[right_mask])

                # Calculate MSE
                mse = np.sum((y[left_mask] - left_mean) ** 2) + np.sum((y[right_mask] - right_mean) ** 2)

                if mse < best_mse:
                    best_mse = mse
                    self.best_feature = feature
                    self.best_threshold = threshold
                    self.left_value = left_mean
                    self.right_value = right_mean

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions = np.where(
            X[:, self.best_feature] <= self.best_threshold,
            self.left_value,
            self.right_value,
        )
        return predictions


def create_fill_ml_model(
    model_type: ModelType = ModelType.GRADIENT_BOOSTING,
) -> FillMLModel:
    """Factory function to create fill ML model."""
    return FillMLModel(model_type=model_type)


__all__ = [
    "FillFeatures",
    "FillMLModel",
    "MLFillPrediction",
    "MarketRegime",
    "ModelType",
    "TrainingRecord",
    "TrainingResult",
    "create_fill_ml_model",
]
