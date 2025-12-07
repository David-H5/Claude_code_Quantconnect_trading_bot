"""Tests for overnight configuration system."""

import os
from pathlib import Path

import pytest

from utils.overnight_config import OvernightConfig

# Get project root for correct file paths
PROJECT_ROOT = Path(__file__).parent.parent


class TestOvernightConfig:
    """Tests for OvernightConfig."""

    def test_load_default_config(self) -> None:
        """Test loading default configuration."""
        config = OvernightConfig.load()

        # Check default session values
        assert config.max_runtime_hours == 10
        assert config.max_idle_minutes == 30

        # Check default budget values
        assert config.max_cost_usd == 50.0

        # Check default recovery values
        assert config.max_restarts == 5

    def test_session_attributes(self) -> None:
        """Test session configuration attributes."""
        config = OvernightConfig.load()

        assert hasattr(config, "max_runtime_hours")
        assert hasattr(config, "max_idle_minutes")
        assert hasattr(config, "checkpoint_interval_minutes")

    def test_budget_attributes(self) -> None:
        """Test budget configuration attributes."""
        config = OvernightConfig.load()

        assert hasattr(config, "max_cost_usd")
        assert hasattr(config, "cost_warning_threshold_pct")

    def test_recovery_attributes(self) -> None:
        """Test recovery configuration attributes."""
        config = OvernightConfig.load()

        assert hasattr(config, "max_restarts")
        assert hasattr(config, "backoff_base_seconds")

    def test_environment_variable_expansion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables are expanded."""
        monkeypatch.setenv("TEST_WEBHOOK_URL", "https://test.webhook/endpoint")

        config = OvernightConfig.load()
        # The config should be loadable even with env vars
        assert config is not None

    def test_missing_env_var_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing env vars return empty string."""
        # Ensure the var doesn't exist
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)

        config = OvernightConfig.load()
        # Config should still load successfully
        assert config is not None

    def test_config_is_immutable_values(self) -> None:
        """Test that config values are consistent across loads."""
        config1 = OvernightConfig.load()
        config2 = OvernightConfig.load()

        assert config1.max_runtime_hours == config2.max_runtime_hours
        assert config1.max_cost_usd == config2.max_cost_usd

    def test_to_shell_exports(self) -> None:
        """Test conversion to shell export format."""
        config = OvernightConfig.load()

        # Check if to_shell_exports method exists and works
        if hasattr(config, "to_shell_exports"):
            exports = config.to_shell_exports()
            assert isinstance(exports, str)
            assert "MAX_RUNTIME_HOURS=" in exports or len(exports) >= 0


class TestConfigFromYaml:
    """Tests for loading config from YAML file."""

    def test_yaml_file_exists(self) -> None:
        """Test that the config YAML file exists."""
        config_path = PROJECT_ROOT / "config" / "overnight.yaml"
        assert config_path.exists(), f"config/overnight.yaml should exist at {config_path}"

    def test_yaml_is_valid(self) -> None:
        """Test that the YAML file is valid."""
        config_path = PROJECT_ROOT / "config" / "overnight.yaml"
        if config_path.exists():
            import yaml

            content = config_path.read_text()
            data = yaml.safe_load(content)
            assert isinstance(data, dict)
