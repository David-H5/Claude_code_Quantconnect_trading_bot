"""
Tests for ML Fill Probability Model

Tests the machine learning fill prediction module.
Part of UPGRADE-010 Sprint 4 - Test Coverage.
"""

import numpy as np
import pytest

from execution.fill_ml_model import (
    FillFeatures,
    FillMLModel,
    MarketRegime,
    MLFillPrediction,
    ModelType,
    TrainingRecord,
    TrainingResult,
    create_fill_ml_model,
)
from models.exceptions.data import DataValidationError


class TestFillFeatures:
    """Tests for FillFeatures dataclass."""

    def test_default_values(self):
        """Test default feature values."""
        features = FillFeatures(
            spread_bps=15.0,
            volume_ratio=0.1,
            time_of_day=0.5,
            day_of_week=2,
            volatility_rank=0.5,
            delta=0.3,
            days_to_expiry=30,
            underlying_move=0.01,
            vix_level=20.0,
        )

        assert features.spread_bps == 15.0
        assert features.market_regime == MarketRegime.NORMAL
        assert features.num_legs == 1
        assert features.order_placement == 0.5

    def test_to_array(self):
        """Test conversion to numpy array."""
        features = FillFeatures(
            spread_bps=10.0,
            volume_ratio=0.2,
            time_of_day=0.3,
            day_of_week=1,
            volatility_rank=0.4,
            delta=0.25,
            days_to_expiry=45,
            underlying_move=0.005,
            vix_level=18.0,
            market_regime=MarketRegime.HIGH,
            num_legs=2,
            order_placement=0.6,
        )

        arr = features.to_array()

        assert isinstance(arr, np.ndarray)
        assert len(arr) == 12
        assert arr[0] == 10.0  # spread_bps
        assert arr[9] == 2  # MarketRegime.HIGH = 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        features = FillFeatures(
            spread_bps=20.0,
            volume_ratio=0.15,
            time_of_day=0.7,
            day_of_week=4,
            volatility_rank=0.8,
            delta=-0.35,
            days_to_expiry=60,
            underlying_move=-0.02,
            vix_level=25.0,
        )

        d = features.to_dict()

        assert d["spread_bps"] == 20.0
        assert d["volatility_rank"] == 0.8
        assert d["delta"] == -0.35
        assert d["market_regime"] == "normal"


class TestFillMLModel:
    """Tests for FillMLModel class."""

    @pytest.fixture
    def model(self):
        """Create default model."""
        return FillMLModel()

    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        return FillFeatures(
            spread_bps=15.0,
            volume_ratio=0.1,
            time_of_day=0.5,
            day_of_week=2,
            volatility_rank=0.5,
            delta=0.3,
            days_to_expiry=30,
            underlying_move=0.01,
            vix_level=20.0,
        )

    @pytest.fixture
    def training_data(self):
        """Create sample training data."""
        records = []
        for i in range(50):
            filled = i % 3 != 0  # 2/3 fill rate
            features = FillFeatures(
                spread_bps=10.0 + i * 2,
                volume_ratio=0.1 + i * 0.01,
                time_of_day=0.3 + (i % 10) * 0.05,
                day_of_week=i % 5,
                volatility_rank=0.3 + (i % 5) * 0.1,
                delta=0.2 + (i % 3) * 0.1,
                days_to_expiry=30 + i,
                underlying_move=0.001 * i,
                vix_level=15.0 + i * 0.2,
            )
            records.append(
                TrainingRecord(
                    features=features,
                    filled=filled,
                    fill_time_seconds=2.0 + i * 0.1 if filled else None,
                )
            )
        return records

    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.model_type == ModelType.GRADIENT_BOOSTING
        assert model.n_estimators == 100
        assert model.is_trained is False

    def test_default_prediction_untrained(self, model, sample_features):
        """Test prediction on untrained model."""
        prediction = model.predict(sample_features)

        assert isinstance(prediction, MLFillPrediction)
        assert 0 <= prediction.fill_probability <= 1
        assert prediction.model_version == "default"
        assert prediction.confidence == 0.3  # Low for untrained

    def test_prediction_spread_adjustment(self, model):
        """Test that prediction adjusts for spread."""
        tight_spread = FillFeatures(
            spread_bps=5.0,
            volume_ratio=0.1,
            time_of_day=0.5,
            day_of_week=2,
            volatility_rank=0.5,
            delta=0.3,
            days_to_expiry=30,
            underlying_move=0.01,
            vix_level=20.0,
        )

        wide_spread = FillFeatures(
            spread_bps=100.0,
            volume_ratio=0.1,
            time_of_day=0.5,
            day_of_week=2,
            volatility_rank=0.5,
            delta=0.3,
            days_to_expiry=30,
            underlying_move=0.01,
            vix_level=20.0,
        )

        tight_pred = model.predict(tight_spread)
        wide_pred = model.predict(wide_spread)

        # Tight spreads should have higher fill probability
        assert tight_pred.fill_probability > wide_pred.fill_probability

    def test_prediction_legs_adjustment(self, model):
        """Test that prediction adjusts for number of legs."""
        single_leg = FillFeatures(
            spread_bps=20.0,
            volume_ratio=0.1,
            time_of_day=0.5,
            day_of_week=2,
            volatility_rank=0.5,
            delta=0.3,
            days_to_expiry=30,
            underlying_move=0.01,
            vix_level=20.0,
            num_legs=1,
        )

        four_legs = FillFeatures(
            spread_bps=20.0,
            volume_ratio=0.1,
            time_of_day=0.5,
            day_of_week=2,
            volatility_rank=0.5,
            delta=0.3,
            days_to_expiry=30,
            underlying_move=0.01,
            vix_level=20.0,
            num_legs=4,
        )

        single_pred = model.predict(single_leg)
        four_pred = model.predict(four_legs)

        # More legs should lower fill probability
        assert single_pred.fill_probability > four_pred.fill_probability

    def test_train_model(self, model, training_data):
        """Test model training."""
        result = model.train(training_data)

        assert isinstance(result, TrainingResult)
        assert model.is_trained is True
        assert result.num_samples == 50
        assert result.num_features == 12
        assert 0 <= result.accuracy <= 1
        assert result.training_time_seconds > 0

    def test_train_insufficient_data(self, model):
        """Test training with insufficient data."""
        few_records = [
            TrainingRecord(
                features=FillFeatures(
                    spread_bps=10.0,
                    volume_ratio=0.1,
                    time_of_day=0.5,
                    day_of_week=2,
                    volatility_rank=0.5,
                    delta=0.3,
                    days_to_expiry=30,
                    underlying_move=0.01,
                    vix_level=20.0,
                ),
                filled=True,
            )
            for _ in range(5)
        ]

        with pytest.raises(DataValidationError, match="at least 10 records"):
            model.train(few_records)

    def test_prediction_after_training(self, model, training_data, sample_features):
        """Test prediction after training."""
        model.train(training_data)

        prediction = model.predict(sample_features)

        assert isinstance(prediction, MLFillPrediction)
        assert prediction.model_version == model.version
        assert prediction.confidence > 0.3  # Higher after training

    def test_feature_importance(self, model, training_data):
        """Test feature importance retrieval."""
        # Before training
        importance = model.feature_importance()
        assert importance == {}

        # After training
        model.train(training_data)
        importance = model.feature_importance()

        assert len(importance) == 12
        assert all(isinstance(v, float) for v in importance.values())

    def test_cross_validation(self, model, training_data):
        """Test cross-validation during training."""
        result = model.train(training_data, n_folds=3)

        assert len(result.cross_val_scores) == 3
        assert all(0 <= s <= 1 for s in result.cross_val_scores)

    def test_prediction_to_dict(self, model, sample_features):
        """Test prediction serialization."""
        prediction = model.predict(sample_features)
        d = prediction.to_dict()

        assert "fill_probability" in d
        assert "confidence" in d
        assert "expected_fill_time" in d
        assert "features_used" in d


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TrainingResult(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            auc_roc=0.90,
            num_samples=100,
            num_features=12,
            cross_val_scores=[0.83, 0.85, 0.87],
            training_time_seconds=1.5,
            feature_importance={"spread_bps": 0.25, "volume_ratio": 0.15},
        )

        d = result.to_dict()

        assert d["accuracy"] == 0.85
        assert d["auc_roc"] == 0.90
        assert d["cross_val_mean"] == pytest.approx(0.85, rel=0.01)
        assert d["training_time_seconds"] == 1.5


class TestModelType:
    """Tests for ModelType enum."""

    def test_model_types_exist(self):
        """Test that all model types exist."""
        assert ModelType.GRADIENT_BOOSTING.value == "gradient_boosting"
        assert ModelType.RANDOM_FOREST.value == "random_forest"
        assert ModelType.LOGISTIC_REGRESSION.value == "logistic_regression"


class TestMarketRegime:
    """Tests for MarketRegime enum."""

    def test_regimes_exist(self):
        """Test that all regimes exist."""
        assert MarketRegime.LOW.value == "low"
        assert MarketRegime.NORMAL.value == "normal"
        assert MarketRegime.HIGH.value == "high"
        assert MarketRegime.EXTREME.value == "extreme"


class TestCreateFillMlModel:
    """Tests for factory function."""

    def test_create_default(self):
        """Test creating model with defaults."""
        model = create_fill_ml_model()

        assert isinstance(model, FillMLModel)
        assert model.model_type == ModelType.GRADIENT_BOOSTING

    def test_create_with_type(self):
        """Test creating model with specific type."""
        model = create_fill_ml_model(model_type=ModelType.RANDOM_FOREST)

        assert model.model_type == ModelType.RANDOM_FOREST
