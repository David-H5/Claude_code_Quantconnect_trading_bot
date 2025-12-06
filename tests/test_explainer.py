"""
Tests for SHAP Decision Explainer (UPGRADE-010 Sprint 1)

Tests explanation generation using SHAP, LIME, and feature importance.
"""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pytest

from evaluation.explainer import (
    LIME_AVAILABLE,
    SHAP_AVAILABLE,
    ExplainerConfig,
    ExplainerFactory,
    Explanation,
    ExplanationLogger,
    ExplanationType,
    FeatureContribution,
    FeatureImportanceExplainer,
    LIMEExplainer,
    ModelType,
    SHAPExplainer,
    create_explainer,
    create_lime_explainer,
    create_shap_explainer,
)


class TestExplanationType:
    """Tests for ExplanationType enum."""

    @pytest.mark.unit
    def test_all_types_exist(self):
        """Test all explanation types exist."""
        assert ExplanationType.SHAP is not None
        assert ExplanationType.LIME is not None
        assert ExplanationType.FEATURE_IMPORTANCE is not None
        assert ExplanationType.PERMUTATION is not None

    @pytest.mark.unit
    def test_type_values(self):
        """Test explanation type values."""
        assert ExplanationType.SHAP.value == "shap"
        assert ExplanationType.LIME.value == "lime"


class TestModelType:
    """Tests for ModelType enum."""

    @pytest.mark.unit
    def test_all_types_exist(self):
        """Test all model types exist."""
        assert ModelType.TREE is not None
        assert ModelType.LINEAR is not None
        assert ModelType.NEURAL_NETWORK is not None
        assert ModelType.KERNEL is not None
        assert ModelType.UNKNOWN is not None


class TestFeatureContribution:
    """Tests for FeatureContribution dataclass."""

    @pytest.mark.unit
    def test_contribution_creation(self):
        """Test creating a feature contribution."""
        fc = FeatureContribution(
            feature_name="rsi",
            feature_value=35.0,
            contribution=0.15,
            rank=1,
            direction="positive",
        )

        assert fc.feature_name == "rsi"
        assert fc.feature_value == 35.0
        assert fc.contribution == 0.15
        assert fc.rank == 1

    @pytest.mark.unit
    def test_to_dict(self):
        """Test serialization to dict."""
        fc = FeatureContribution(
            feature_name="volume",
            feature_value=1000000,
            contribution=-0.05,
            rank=2,
            direction="negative",
        )

        data = fc.to_dict()

        assert data["feature_name"] == "volume"
        assert data["contribution"] == -0.05
        assert data["direction"] == "negative"


class TestExplanation:
    """Tests for Explanation dataclass."""

    @pytest.fixture
    def sample_explanation(self) -> Explanation:
        """Create a sample explanation."""
        contributions = [
            FeatureContribution("rsi", 35.0, 0.15, 1, "positive"),
            FeatureContribution("volume", 1000000, -0.05, 2, "negative"),
            FeatureContribution("macd", 0.5, 0.10, 3, "positive"),
        ]

        return Explanation(
            explanation_id="test123",
            timestamp=datetime.utcnow(),
            explanation_type=ExplanationType.SHAP,
            model_type=ModelType.TREE,
            prediction=0.75,
            base_value=0.5,
            feature_contributions=contributions,
            total_contribution=0.20,
            confidence=0.85,
        )

    @pytest.mark.unit
    def test_explanation_creation(self, sample_explanation):
        """Test creating an explanation."""
        assert sample_explanation.explanation_id == "test123"
        assert sample_explanation.prediction == 0.75
        assert len(sample_explanation.feature_contributions) == 3

    @pytest.mark.unit
    def test_top_features(self, sample_explanation):
        """Test getting top features."""
        top = sample_explanation.top_features

        assert len(top) == 3
        assert top[0].feature_name == "rsi"  # Highest absolute contribution

    @pytest.mark.unit
    def test_positive_contributors(self, sample_explanation):
        """Test getting positive contributors."""
        positive = sample_explanation.positive_contributors

        assert len(positive) == 2
        assert all(c.contribution > 0 for c in positive)

    @pytest.mark.unit
    def test_negative_contributors(self, sample_explanation):
        """Test getting negative contributors."""
        negative = sample_explanation.negative_contributors

        assert len(negative) == 1
        assert negative[0].feature_name == "volume"

    @pytest.mark.unit
    def test_to_dict(self, sample_explanation):
        """Test serialization to dict."""
        data = sample_explanation.to_dict()

        assert data["explanation_id"] == "test123"
        assert data["prediction"] == 0.75
        assert len(data["feature_contributions"]) == 3

    @pytest.mark.unit
    def test_to_text(self, sample_explanation):
        """Test text generation."""
        text = sample_explanation.to_text()

        assert "Explanation" in text
        assert "Prediction: 0.75" in text
        assert "rsi" in text


class TestExplainerConfig:
    """Tests for ExplainerConfig."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default configuration."""
        config = ExplainerConfig()

        assert config.shap_background_samples == 100
        assert config.lime_num_features == 10
        assert config.top_k_features == 10

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom configuration."""
        config = ExplainerConfig(
            shap_background_samples=50,
            lime_num_features=5,
        )

        assert config.shap_background_samples == 50
        assert config.lime_num_features == 5


class TestFeatureImportanceExplainer:
    """Tests for FeatureImportanceExplainer (always available)."""

    @pytest.fixture
    def mock_model_with_importance(self):
        """Create mock model with feature importances."""
        model = Mock()
        model.feature_importances_ = np.array([0.3, 0.5, 0.2])
        model.predict = Mock(return_value=np.array([0.75]))
        return model

    @pytest.fixture
    def mock_model_without_importance(self):
        """Create mock model without feature importances."""
        # Use spec=[] to prevent Mock from auto-creating attributes
        # This ensures hasattr() returns False for undefined attributes
        model = Mock(spec=[])
        model.predict = Mock(return_value=np.array([0.75]))
        return model

    @pytest.mark.unit
    def test_explainer_creation(self, mock_model_with_importance):
        """Test creating feature importance explainer."""
        explainer = FeatureImportanceExplainer(
            model=mock_model_with_importance,
            feature_names=["rsi", "volume", "macd"],
        )

        assert explainer._importances is not None
        assert len(explainer._importances) == 3

    @pytest.mark.unit
    def test_explain_with_importance(self, mock_model_with_importance):
        """Test explaining with model that has importances."""
        explainer = FeatureImportanceExplainer(
            model=mock_model_with_importance,
            feature_names=["rsi", "volume", "macd"],
        )

        instance = np.array([35.0, 1000000.0, 0.5])
        explanation = explainer.explain(instance)

        assert explanation.explanation_type == ExplanationType.FEATURE_IMPORTANCE
        assert len(explanation.feature_contributions) == 3
        assert explanation.feature_contributions[0].rank == 1

    @pytest.mark.unit
    def test_explain_without_importance(self, mock_model_without_importance):
        """Test explaining with model that lacks importances."""
        explainer = FeatureImportanceExplainer(
            model=mock_model_without_importance,
            feature_names=["rsi", "volume", "macd"],
        )

        instance = np.array([35.0, 1000000.0, 0.5])
        explanation = explainer.explain(instance)

        # Should use equal weights
        assert len(explanation.feature_contributions) == 3


class TestExplainerFactory:
    """Tests for ExplainerFactory."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.feature_importances_ = np.array([0.3, 0.5, 0.2])
        model.predict = Mock(return_value=np.array([0.75]))
        return model

    @pytest.mark.unit
    def test_create_feature_importance(self, mock_model):
        """Test creating feature importance explainer."""
        explainer = ExplainerFactory.create(
            model=mock_model,
            feature_names=["rsi", "volume", "macd"],
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
        )

        assert isinstance(explainer, FeatureImportanceExplainer)

    @pytest.mark.unit
    def test_detect_tree_model(self):
        """Test auto-detecting tree model type."""
        # Mock a tree-based model
        model = Mock()
        model.__class__.__name__ = "RandomForestClassifier"

        model_type = ExplainerFactory._detect_model_type(model)

        assert model_type == ModelType.TREE

    @pytest.mark.unit
    def test_detect_linear_model(self):
        """Test auto-detecting linear model type."""
        model = Mock()
        model.__class__.__name__ = "LinearRegression"

        model_type = ExplainerFactory._detect_model_type(model)

        assert model_type == ModelType.LINEAR

    @pytest.mark.unit
    def test_detect_unknown_model(self):
        """Test auto-detecting unknown model type."""
        model = Mock()
        model.__class__.__name__ = "MyCustomModel"

        model_type = ExplainerFactory._detect_model_type(model)

        assert model_type == ModelType.UNKNOWN

    @pytest.mark.unit
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="shap not installed")
    def test_create_shap_explainer(self, mock_model):
        """Test creating SHAP explainer."""
        training_data = np.random.randn(100, 3)

        explainer = ExplainerFactory.create(
            model=mock_model,
            feature_names=["rsi", "volume", "macd"],
            explanation_type=ExplanationType.SHAP,
            training_data=training_data,
        )

        assert isinstance(explainer, SHAPExplainer)

    @pytest.mark.unit
    @pytest.mark.skipif(not LIME_AVAILABLE, reason="lime not installed")
    def test_create_lime_explainer(self, mock_model):
        """Test creating LIME explainer."""
        training_data = np.random.randn(100, 3)

        explainer = ExplainerFactory.create(
            model=mock_model,
            feature_names=["rsi", "volume", "macd"],
            explanation_type=ExplanationType.LIME,
            training_data=training_data,
        )

        assert isinstance(explainer, LIMEExplainer)


class TestExplanationLogger:
    """Tests for ExplanationLogger."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_explanation(self) -> Explanation:
        """Create sample explanation."""
        return Explanation(
            explanation_id="test123",
            timestamp=datetime.utcnow(),
            explanation_type=ExplanationType.SHAP,
            model_type=ModelType.TREE,
            prediction=0.75,
            base_value=0.5,
            feature_contributions=[
                FeatureContribution("rsi", 35.0, 0.15, 1, "positive"),
            ],
            total_contribution=0.15,
        )

    @pytest.mark.unit
    def test_logger_creation(self, temp_dir):
        """Test creating logger."""
        logger = ExplanationLogger(storage_dir=temp_dir)

        assert logger.storage_dir == temp_dir
        assert len(logger._explanations) == 0

    @pytest.mark.unit
    def test_log_explanation(self, temp_dir, sample_explanation):
        """Test logging an explanation."""
        logger = ExplanationLogger(storage_dir=temp_dir)
        logger.log(sample_explanation)

        assert len(logger._explanations) == 1

        # Check file was created
        expected_file = os.path.join(temp_dir, f"{sample_explanation.explanation_id}.json")
        assert os.path.exists(expected_file)

    @pytest.mark.unit
    def test_export_audit_trail(self, temp_dir, sample_explanation):
        """Test exporting audit trail."""
        logger = ExplanationLogger(storage_dir=temp_dir)
        logger.log(sample_explanation)

        export_path = os.path.join(temp_dir, "audit.json")
        count = logger.export_audit_trail(export_path)

        assert count == 1
        assert os.path.exists(export_path)

        with open(export_path) as f:
            data = json.load(f)

        assert data["total_explanations"] == 1
        assert len(data["entries"]) == 1

    @pytest.mark.unit
    def test_get_explanations(self, temp_dir, sample_explanation):
        """Test getting explanations."""
        logger = ExplanationLogger(storage_dir=temp_dir)
        logger.log(sample_explanation)

        explanations = logger.get_explanations()

        assert len(explanations) == 1
        assert explanations[0].explanation_id == "test123"

    @pytest.mark.unit
    def test_get_explanations_filtered(self, temp_dir):
        """Test getting explanations filtered by type."""
        logger = ExplanationLogger(storage_dir=temp_dir)

        # Log SHAP explanation
        shap_exp = Explanation(
            explanation_id="shap1",
            timestamp=datetime.utcnow(),
            explanation_type=ExplanationType.SHAP,
            model_type=ModelType.TREE,
            prediction=0.75,
            base_value=0.5,
            feature_contributions=[],
            total_contribution=0.0,
        )
        logger.log(shap_exp)

        # Log LIME explanation
        lime_exp = Explanation(
            explanation_id="lime1",
            timestamp=datetime.utcnow(),
            explanation_type=ExplanationType.LIME,
            model_type=ModelType.UNKNOWN,
            prediction=0.8,
            base_value=0.5,
            feature_contributions=[],
            total_contribution=0.0,
        )
        logger.log(lime_exp)

        # Filter by type
        shap_only = logger.get_explanations(explanation_type=ExplanationType.SHAP)

        assert len(shap_only) == 1
        assert shap_only[0].explanation_id == "shap1"


class TestCreateExplainerFunctions:
    """Tests for factory functions."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.feature_importances_ = np.array([0.3, 0.5, 0.2])
        model.predict = Mock(return_value=np.array([0.75]))
        return model

    @pytest.mark.unit
    def test_create_explainer(self, mock_model):
        """Test create_explainer factory."""
        explainer = create_explainer(
            model=mock_model,
            feature_names=["rsi", "volume", "macd"],
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
        )

        assert explainer is not None

    @pytest.mark.unit
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="shap not installed")
    def test_create_shap_explainer_factory(self, mock_model):
        """Test create_shap_explainer factory."""
        explainer = create_shap_explainer(
            model=mock_model,
            feature_names=["rsi", "volume", "macd"],
            model_type=ModelType.TREE,
        )

        assert isinstance(explainer, SHAPExplainer)

    @pytest.mark.unit
    @pytest.mark.skipif(not LIME_AVAILABLE, reason="lime not installed")
    def test_create_lime_explainer_factory(self, mock_model):
        """Test create_lime_explainer factory."""
        training_data = np.random.randn(100, 3)

        explainer = create_lime_explainer(
            model=mock_model,
            feature_names=["rsi", "volume", "macd"],
            training_data=training_data,
        )

        assert isinstance(explainer, LIMEExplainer)


class TestLibraryAvailability:
    """Tests for library availability checks."""

    @pytest.mark.unit
    def test_shap_availability_flag(self):
        """Test SHAP availability flag exists."""
        # SHAP_AVAILABLE should be True or False
        assert isinstance(SHAP_AVAILABLE, bool)

    @pytest.mark.unit
    def test_lime_availability_flag(self):
        """Test LIME availability flag exists."""
        # LIME_AVAILABLE should be True or False
        assert isinstance(LIME_AVAILABLE, bool)
