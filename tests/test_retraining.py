"""
Tests for Model Retraining Pipeline (UPGRADE-010 Sprint 2)

Tests drift detection, retraining triggers, and model validation.
"""

import tempfile
from datetime import datetime

import pytest

from llm.retraining import (
    DriftMonitor,
    DriftResult,
    DriftSeverity,
    ModelRecord,
    ModelStatus,
    RetrainingConfig,
    RetrainingPipeline,
    RetrainJob,
    RetrainStatus,
    ValidationResult,
    create_retraining_pipeline,
)
from models.exceptions.data import DataMissingError


class TestDriftSeverity:
    """Tests for DriftSeverity enum."""

    @pytest.mark.unit
    def test_severity_levels_exist(self):
        """Test all severity levels exist."""
        assert DriftSeverity.NONE is not None
        assert DriftSeverity.LOW is not None
        assert DriftSeverity.MODERATE is not None
        assert DriftSeverity.HIGH is not None
        assert DriftSeverity.CRITICAL is not None


class TestRetrainStatus:
    """Tests for RetrainStatus enum."""

    @pytest.mark.unit
    def test_status_values_exist(self):
        """Test all status values exist."""
        assert RetrainStatus.PENDING is not None
        assert RetrainStatus.RUNNING is not None
        assert RetrainStatus.COMPLETED is not None
        assert RetrainStatus.FAILED is not None
        assert RetrainStatus.ROLLED_BACK is not None


class TestDriftMonitor:
    """Tests for DriftMonitor."""

    @pytest.fixture
    def monitor(self) -> DriftMonitor:
        """Create monitor for testing."""
        return DriftMonitor(psi_threshold=0.25, num_bins=10, min_samples=10)

    @pytest.mark.unit
    def test_monitor_creation(self, monitor):
        """Test monitor creation."""
        assert monitor is not None
        assert monitor.psi_threshold == 0.25
        assert monitor.num_bins == 10

    @pytest.mark.unit
    def test_calculate_psi_identical(self, monitor):
        """Test PSI for identical distributions."""
        data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 10

        psi = monitor.calculate_psi(data, data.copy())

        # Identical distributions should have PSI near 0
        assert psi < 0.01

    @pytest.mark.unit
    def test_calculate_psi_different(self, monitor):
        """Test PSI for different distributions."""
        reference = [0.1, 0.2, 0.3, 0.4, 0.5] * 20  # Uniform-ish
        current = [0.8, 0.85, 0.9, 0.95, 1.0] * 20  # Shifted high

        psi = monitor.calculate_psi(reference, current)

        # Different distributions should have higher PSI
        assert psi > 0.1

    @pytest.mark.unit
    def test_calculate_psi_insufficient_samples(self, monitor):
        """Test PSI with insufficient samples."""
        reference = [0.1, 0.2, 0.3]  # Too few
        current = [0.4, 0.5, 0.6]

        psi = monitor.calculate_psi(reference, current)

        # Should return 0 for insufficient samples
        assert psi == 0.0

    @pytest.mark.unit
    def test_calculate_psi_constant_values(self, monitor):
        """Test PSI with constant values."""
        data = [0.5] * 20

        psi = monitor.calculate_psi(data, data)

        # Constant values should return 0
        assert psi == 0.0

    @pytest.mark.unit
    def test_check_drift_no_drift(self, monitor):
        """Test drift check with no drift."""
        reference = {"feature1": [0.1 + i * 0.01 for i in range(100)]}
        current = {"feature1": [0.1 + i * 0.01 for i in range(100)]}

        result = monitor.check_drift("test_model", reference, current)

        assert result.drift_detected is False
        assert result.severity == DriftSeverity.NONE

    @pytest.mark.unit
    def test_check_drift_detected(self, monitor):
        """Test drift check with drift detected."""
        reference = {"feature1": [0.1 + i * 0.005 for i in range(100)]}
        current = {"feature1": [0.6 + i * 0.005 for i in range(100)]}  # Shifted

        result = monitor.check_drift("test_model", reference, current)

        assert result.drift_detected is True
        assert result.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]

    @pytest.mark.unit
    def test_check_drift_multiple_features(self, monitor):
        """Test drift check with multiple features."""
        reference = {
            "feature1": [0.1 + i * 0.01 for i in range(100)],
            "feature2": [0.5 + i * 0.005 for i in range(100)],
        }
        current = {
            "feature1": [0.1 + i * 0.01 for i in range(100)],  # No drift
            "feature2": [0.9 + i * 0.005 for i in range(100)],  # Drift
        }

        result = monitor.check_drift("test_model", reference, current)

        assert "feature1" in result.feature_drift
        assert "feature2" in result.feature_drift


class TestDriftResult:
    """Tests for DriftResult dataclass."""

    @pytest.mark.unit
    def test_drift_result_creation(self):
        """Test creating drift result."""
        result = DriftResult(
            model_name="test_model",
            drift_detected=True,
            psi_score=0.35,
            severity=DriftSeverity.HIGH,
            feature_drift={"feature1": 0.35},
            timestamp=datetime.utcnow(),
            recommendation="Schedule retraining soon",
        )

        assert result.model_name == "test_model"
        assert result.drift_detected is True
        assert result.psi_score == 0.35

    @pytest.mark.unit
    def test_drift_result_to_dict(self):
        """Test serialization."""
        result = DriftResult(
            model_name="test_model",
            drift_detected=True,
            psi_score=0.35,
            severity=DriftSeverity.HIGH,
            feature_drift={"feature1": 0.35},
            timestamp=datetime.utcnow(),
            recommendation="Retrain",
        )

        data = result.to_dict()

        assert data["model_name"] == "test_model"
        assert data["severity"] == "high"


class TestRetrainJob:
    """Tests for RetrainJob dataclass."""

    @pytest.mark.unit
    def test_retrain_job_creation(self):
        """Test creating retrain job."""
        job = RetrainJob(
            job_id="job_001",
            model_name="test_model",
            status=RetrainStatus.PENDING,
            triggered_at=datetime.utcnow(),
            trigger_reason="Drift detected",
        )

        assert job.job_id == "job_001"
        assert job.status == RetrainStatus.PENDING

    @pytest.mark.unit
    def test_duration_seconds(self):
        """Test duration calculation."""
        started = datetime.utcnow()
        completed = datetime.utcnow()

        job = RetrainJob(
            job_id="job_001",
            model_name="test_model",
            status=RetrainStatus.COMPLETED,
            triggered_at=started,
            trigger_reason="Test",
            started_at=started,
            completed_at=completed,
        )

        assert job.duration_seconds >= 0

    @pytest.mark.unit
    def test_duration_no_completion(self):
        """Test duration with no completion."""
        job = RetrainJob(
            job_id="job_001",
            model_name="test_model",
            status=RetrainStatus.PENDING,
            triggered_at=datetime.utcnow(),
            trigger_reason="Test",
        )

        assert job.duration_seconds == 0.0


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    @pytest.mark.unit
    def test_validation_result_creation(self):
        """Test creating validation result."""
        result = ValidationResult(
            model_name="test_model",
            old_metrics={"accuracy": 0.8},
            new_metrics={"accuracy": 0.85},
            is_improved=True,
            improvement_pct=6.25,
            validation_timestamp=datetime.utcnow(),
            test_samples=1000,
            passed_checks=["Accuracy improved"],
            failed_checks=[],
        )

        assert result.is_improved is True
        assert result.improvement_pct == 6.25

    @pytest.mark.unit
    def test_validation_result_to_dict(self):
        """Test serialization."""
        result = ValidationResult(
            model_name="test_model",
            old_metrics={"accuracy": 0.8},
            new_metrics={"accuracy": 0.85},
            is_improved=True,
            improvement_pct=6.25,
            validation_timestamp=datetime.utcnow(),
            test_samples=1000,
        )

        data = result.to_dict()

        assert data["model_name"] == "test_model"
        assert data["is_improved"] is True


class TestModelRecord:
    """Tests for ModelRecord dataclass."""

    @pytest.mark.unit
    def test_model_record_creation(self):
        """Test creating model record."""
        record = ModelRecord(
            model_name="sentiment_model",
            status=ModelStatus.ACTIVE,
            version="1.0.0",
            deployed_at=datetime.utcnow(),
        )

        assert record.model_name == "sentiment_model"
        assert record.status == ModelStatus.ACTIVE


class TestRetrainingConfig:
    """Tests for RetrainingConfig dataclass."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default configuration."""
        config = RetrainingConfig()

        assert config.psi_threshold == 0.25
        assert config.min_improvement_pct == 2.0
        assert config.auto_retrain_on_drift is False

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom configuration."""
        config = RetrainingConfig(
            psi_threshold=0.3,
            min_improvement_pct=5.0,
            auto_retrain_on_drift=True,
        )

        assert config.psi_threshold == 0.3
        assert config.min_improvement_pct == 5.0
        assert config.auto_retrain_on_drift is True


class TestRetrainingPipeline:
    """Tests for RetrainingPipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def pipeline(self, temp_dir) -> RetrainingPipeline:
        """Create pipeline for testing."""
        return RetrainingPipeline(storage_dir=temp_dir)

    @pytest.mark.unit
    def test_pipeline_creation(self, pipeline):
        """Test pipeline creation."""
        assert pipeline is not None
        assert pipeline.drift_monitor is not None
        assert pipeline.config is not None

    @pytest.mark.unit
    def test_register_model(self, pipeline):
        """Test model registration."""
        record = pipeline.register_model(
            "sentiment_model",
            version="1.0.0",
        )

        assert record.model_name == "sentiment_model"
        assert record.version == "1.0.0"
        assert record.status == ModelStatus.ACTIVE

    @pytest.mark.unit
    def test_register_model_idempotent(self, pipeline):
        """Test registering same model returns existing."""
        record1 = pipeline.register_model("test_model")
        record2 = pipeline.register_model("test_model")

        assert record1 is record2

    @pytest.mark.unit
    def test_get_model(self, pipeline):
        """Test getting model record."""
        pipeline.register_model("test_model")

        record = pipeline.get_model("test_model")
        assert record is not None
        assert record.model_name == "test_model"

    @pytest.mark.unit
    def test_get_model_not_found(self, pipeline):
        """Test getting non-existent model."""
        record = pipeline.get_model("unknown")
        assert record is None

    @pytest.mark.unit
    def test_check_drift(self, pipeline):
        """Test drift checking."""
        pipeline.register_model("test_model")

        reference = {"feature1": [0.1 + i * 0.01 for i in range(100)]}
        current = {"feature1": [0.1 + i * 0.01 for i in range(100)]}

        result = pipeline.check_drift("test_model", reference, current)

        assert result is not None
        assert result.model_name == "test_model"

    @pytest.mark.unit
    def test_trigger_retrain(self, pipeline):
        """Test triggering retraining."""
        pipeline.register_model("test_model")

        job = pipeline.trigger_retrain("test_model", reason="Manual test")

        assert job is not None
        assert job.model_name == "test_model"
        assert job.status == RetrainStatus.PENDING
        assert "Manual test" in job.trigger_reason

    @pytest.mark.unit
    def test_trigger_retrain_updates_model_status(self, pipeline):
        """Test that triggering retrain updates model status."""
        pipeline.register_model("test_model")
        pipeline.trigger_retrain("test_model")

        record = pipeline.get_model("test_model")
        assert record.status == ModelStatus.RETRAINING

    @pytest.mark.unit
    def test_run_retrain_job(self, pipeline):
        """Test running retrain job."""
        pipeline.register_model("test_model")
        job = pipeline.trigger_retrain("test_model")

        updated_job = pipeline.run_retrain_job(job.job_id, training_data=[])

        assert updated_job.status == RetrainStatus.VALIDATING
        assert updated_job.started_at is not None

    @pytest.mark.unit
    def test_run_retrain_job_not_found(self, pipeline):
        """Test running non-existent job."""
        with pytest.raises(DataMissingError, match="job_id"):
            pipeline.run_retrain_job("unknown_job", [])

    @pytest.mark.unit
    def test_validate_model_improved(self, pipeline):
        """Test model validation with improvement."""
        result = pipeline.validate_model(
            model_name="test_model",
            new_model=object(),
            old_model=object(),
            test_data=[1, 2, 3],
        )

        # Using placeholder metrics: new is better
        assert result.new_metrics["accuracy"] > result.old_metrics["accuracy"]

    @pytest.mark.unit
    def test_deploy_model(self, pipeline):
        """Test model deployment."""
        pipeline.register_model("test_model", version="1.0.0")

        # Stage a model
        pipeline._staged_models["test_model"] = object()

        success = pipeline.deploy_model("test_model")

        assert success is True
        record = pipeline.get_model("test_model")
        assert record.version == "1.0.1"  # Auto-incremented
        assert record.status == ModelStatus.ACTIVE

    @pytest.mark.unit
    def test_deploy_model_custom_version(self, pipeline):
        """Test deployment with custom version."""
        pipeline.register_model("test_model", version="1.0.0")
        pipeline._staged_models["test_model"] = object()

        pipeline.deploy_model("test_model", new_version="2.0.0")

        record = pipeline.get_model("test_model")
        assert record.version == "2.0.0"

    @pytest.mark.unit
    def test_deploy_model_no_staged(self, pipeline):
        """Test deployment with no staged model."""
        pipeline.register_model("test_model")

        success = pipeline.deploy_model("test_model")

        assert success is False

    @pytest.mark.unit
    def test_rollback(self, pipeline):
        """Test model rollback."""
        pipeline.register_model("test_model")
        job = pipeline.trigger_retrain("test_model")
        pipeline._staged_models["test_model"] = object()

        success = pipeline.rollback("test_model")

        assert success is True
        assert "test_model" not in pipeline._staged_models
        record = pipeline.get_model("test_model")
        assert record.status == ModelStatus.ACTIVE

        # Check job status
        job = pipeline.get_job(job.job_id)
        assert job.status == RetrainStatus.ROLLED_BACK

    @pytest.mark.unit
    def test_get_job(self, pipeline):
        """Test getting job by ID."""
        pipeline.register_model("test_model")
        job = pipeline.trigger_retrain("test_model")

        retrieved = pipeline.get_job(job.job_id)

        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    @pytest.mark.unit
    def test_list_jobs(self, pipeline):
        """Test listing jobs."""
        pipeline.register_model("model1")
        pipeline.register_model("model2")
        pipeline.trigger_retrain("model1")
        pipeline.trigger_retrain("model2")
        pipeline.trigger_retrain("model1")

        # All jobs
        all_jobs = pipeline.list_jobs()
        assert len(all_jobs) == 3

        # Filter by model
        model1_jobs = pipeline.list_jobs(model_name="model1")
        assert len(model1_jobs) == 2

    @pytest.mark.unit
    def test_list_jobs_by_status(self, pipeline):
        """Test listing jobs by status."""
        pipeline.register_model("test_model")
        job = pipeline.trigger_retrain("test_model")
        pipeline.run_retrain_job(job.job_id, [])

        pending_jobs = pipeline.list_jobs(status=RetrainStatus.PENDING)
        validating_jobs = pipeline.list_jobs(status=RetrainStatus.VALIDATING)

        assert len(pending_jobs) == 0
        assert len(validating_jobs) == 1

    @pytest.mark.unit
    def test_get_statistics(self, pipeline):
        """Test getting statistics."""
        pipeline.register_model("test_model")
        pipeline.trigger_retrain("test_model")

        stats = pipeline.get_statistics()

        assert stats["total_models"] == 1
        assert stats["total_jobs"] == 1

    @pytest.mark.unit
    def test_statistics_empty(self, pipeline):
        """Test statistics with no models."""
        stats = pipeline.get_statistics()

        assert stats["total_models"] == 0
        assert stats["total_jobs"] == 0


class TestCreateRetrainingPipeline:
    """Tests for factory function."""

    @pytest.mark.unit
    def test_create_with_defaults(self):
        """Test factory with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = create_retraining_pipeline(storage_dir=tmpdir)

            assert pipeline is not None
            assert pipeline.config.psi_threshold == 0.25

    @pytest.mark.unit
    def test_create_with_custom_settings(self):
        """Test factory with custom settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = create_retraining_pipeline(
                psi_threshold=0.3,
                min_improvement_pct=5.0,
                auto_retrain=True,
                storage_dir=tmpdir,
            )

            assert pipeline.config.psi_threshold == 0.3
            assert pipeline.config.min_improvement_pct == 5.0
            assert pipeline.config.auto_retrain_on_drift is True


class TestAutoRetrain:
    """Tests for auto-retraining functionality."""

    @pytest.mark.unit
    def test_auto_retrain_on_drift(self):
        """Test auto-retraining triggers on drift."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RetrainingConfig(
                psi_threshold=0.1,  # Low threshold
                auto_retrain_on_drift=True,
            )
            pipeline = RetrainingPipeline(
                config=config,
                storage_dir=tmpdir,
            )

            pipeline.register_model("test_model")

            # Create significant drift
            reference = {"feature1": [0.1 + i * 0.005 for i in range(100)]}
            current = {"feature1": [0.8 + i * 0.005 for i in range(100)]}

            result = pipeline.check_drift("test_model", reference, current)

            # Should have triggered a retrain job
            jobs = pipeline.list_jobs(model_name="test_model")
            assert len(jobs) >= 1 if result.drift_detected else len(jobs) == 0


class TestRetrainingIntegration:
    """Integration tests for retraining pipeline."""

    @pytest.mark.unit
    def test_full_retraining_workflow(self):
        """Test complete retraining workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = create_retraining_pipeline(storage_dir=tmpdir)

            # Register model
            pipeline.register_model("sentiment_model", version="1.0.0")

            # Check for drift
            reference = {"predictions": [0.5 + i * 0.01 for i in range(100)]}
            current = {"predictions": [0.7 + i * 0.01 for i in range(100)]}

            drift_result = pipeline.check_drift(
                "sentiment_model",
                reference,
                current,
            )

            # Trigger retraining if drift detected
            if drift_result.drift_detected:
                job = pipeline.trigger_retrain(
                    "sentiment_model",
                    drift_result=drift_result,
                )

                # Run training
                pipeline.run_retrain_job(job.job_id, [])

                # Validate
                validation = pipeline.validate_model(
                    "sentiment_model",
                    new_model=object(),
                    old_model=object(),
                    test_data=[1, 2, 3],
                )

                if validation.is_improved:
                    # Must pass the model object for deployment
                    pipeline.deploy_model("sentiment_model", model=object())
                else:
                    pipeline.rollback("sentiment_model")

            # Verify final state
            record = pipeline.get_model("sentiment_model")
            assert record.status == ModelStatus.ACTIVE
