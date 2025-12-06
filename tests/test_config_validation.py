"""
Tests for Configuration Validation (UPGRADE-011)

Tests configuration loading, validation, and typed config access.
"""

import json

import pytest

from config import (
    APIConfig,
    ConfigManager,
    ConfigValidationError,
    LoggingConfig,
    PerformanceConfig,
    ProfitTakingConfig,
    RiskConfig,
    get_config,
    reload_config,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def valid_config():
    """Create a minimal valid configuration."""
    return {
        "brokerage": {"provider": "schwab", "paper_trading": True},
        "risk_management": {
            "max_position_size_pct": 0.25,
            "max_daily_loss_pct": 0.03,
            "max_drawdown_pct": 0.10,
            "max_risk_per_trade_pct": 0.02,
            "max_consecutive_losses": 5,
            "require_human_reset": True,
        },
        "profit_taking": {
            "enabled": True,
            "thresholds": [
                {"gain_pct": 1.00, "sell_pct": 0.50},
                {"gain_pct": 2.00, "sell_pct": 0.25},
            ],
            "trailing_stop_enabled": True,
            "trailing_stop_pct": 0.25,
        },
        "order_execution": {
            "cancel_replace_enabled": True,
            "max_bid_increase_pct": 1.00,
            "cancel_after_seconds": 30,
        },
        "options_scanner": {
            "enabled": True,
            "target_delta_range": [0.20, 0.40],
        },
        "movement_scanner": {
            "enabled": True,
            "min_movement_pct": 0.02,
        },
        "structured_logging": {
            "enabled": True,
            "level": "INFO",
            "file_path": "logs/test.jsonl",
        },
        "performance_tracker": {
            "enabled": True,
            "starting_equity": 100000.0,
        },
        "api_server": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 8080,
        },
    }


@pytest.fixture
def config_file(valid_config, tmp_path):
    """Create a temporary config file."""
    config_path = tmp_path / "settings.json"
    with open(config_path, "w") as f:
        json.dump(valid_config, f)
    return config_path


@pytest.fixture
def config_manager(config_file):
    """Create a ConfigManager with valid config."""
    return ConfigManager(config_file)


# ==============================================================================
# Test ConfigManager Basic Operations
# ==============================================================================


class TestConfigManagerBasics:
    """Test basic ConfigManager operations."""

    def test_load_config(self, config_manager):
        """Test config loads successfully."""
        assert config_manager is not None

    def test_get_value(self, config_manager):
        """Test getting values by dot notation."""
        assert config_manager.get("brokerage.provider") == "schwab"
        assert config_manager.get("risk_management.max_daily_loss_pct") == 0.03

    def test_get_with_default(self, config_manager):
        """Test getting values with default."""
        assert config_manager.get("nonexistent.key", "default") == "default"

    def test_update_value(self, config_manager):
        """Test updating values."""
        config_manager.update("risk_management.max_daily_loss_pct", 0.05)
        assert config_manager.get("risk_management.max_daily_loss_pct") == 0.05

    def test_file_not_found(self, tmp_path):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            ConfigManager(tmp_path / "nonexistent.json")


# ==============================================================================
# Test New Config Classes (UPGRADE-011)
# ==============================================================================


class TestLoggingConfig:
    """Test LoggingConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        cfg = LoggingConfig()
        assert cfg.enabled is True
        assert cfg.level == "INFO"
        assert cfg.console_enabled is True
        assert cfg.file_enabled is True
        assert cfg.max_file_size_mb == 50

    def test_get_logging_config(self, config_manager):
        """Test get_logging_config method."""
        cfg = config_manager.get_logging_config()
        assert isinstance(cfg, LoggingConfig)
        assert cfg.enabled is True
        assert cfg.level == "INFO"
        assert cfg.file_path == "logs/test.jsonl"


class TestPerformanceConfig:
    """Test PerformanceConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        cfg = PerformanceConfig()
        assert cfg.enabled is True
        assert cfg.starting_equity == 100000.0
        assert cfg.track_sessions is True
        assert "daily" in cfg.session_types

    def test_get_performance_config(self, config_manager):
        """Test get_performance_config method."""
        cfg = config_manager.get_performance_config()
        assert isinstance(cfg, PerformanceConfig)
        assert cfg.enabled is True
        assert cfg.starting_equity == 100000.0


class TestAPIConfig:
    """Test APIConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        cfg = APIConfig()
        assert cfg.enabled is True
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8080
        assert cfg.websocket_enabled is True

    def test_get_api_config(self, config_manager):
        """Test get_api_config method."""
        cfg = config_manager.get_api_config()
        assert isinstance(cfg, APIConfig)
        assert cfg.enabled is True
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8080


# ==============================================================================
# Test Existing Config Classes
# ==============================================================================


class TestRiskConfig:
    """Test RiskConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        cfg = RiskConfig()
        assert cfg.max_position_size_pct == 0.25
        assert cfg.max_daily_loss_pct == 0.03

    def test_get_risk_config(self, config_manager):
        """Test get_risk_config method."""
        cfg = config_manager.get_risk_config()
        assert isinstance(cfg, RiskConfig)
        assert cfg.max_daily_loss_pct == 0.03


class TestProfitTakingConfig:
    """Test ProfitTakingConfig dataclass."""

    def test_get_profit_taking_config(self, config_manager):
        """Test get_profit_taking_config method."""
        cfg = config_manager.get_profit_taking_config()
        assert isinstance(cfg, ProfitTakingConfig)
        assert cfg.enabled is True
        assert len(cfg.thresholds) == 2


# ==============================================================================
# Test Configuration Validation
# ==============================================================================


class TestConfigValidation:
    """Test configuration validation (UPGRADE-011)."""

    def test_valid_config(self, config_manager):
        """Test valid config passes validation."""
        errors = config_manager.validate()
        assert len(errors) == 0
        assert config_manager.is_valid()

    def test_invalid_daily_loss_limit(self, valid_config, tmp_path):
        """Test validation catches excessive daily loss limit."""
        valid_config["risk_management"]["max_daily_loss_pct"] = 0.15
        config_path = tmp_path / "settings.json"
        with open(config_path, "w") as f:
            json.dump(valid_config, f)

        mgr = ConfigManager(config_path)
        errors = mgr.validate()
        assert len(errors) >= 1
        assert any("daily_loss" in e.field.lower() for e in errors)

    def test_invalid_drawdown_limit(self, valid_config, tmp_path):
        """Test validation catches excessive drawdown limit."""
        valid_config["risk_management"]["max_drawdown_pct"] = 0.30
        config_path = tmp_path / "settings.json"
        with open(config_path, "w") as f:
            json.dump(valid_config, f)

        mgr = ConfigManager(config_path)
        errors = mgr.validate()
        assert len(errors) >= 1
        assert any("drawdown" in e.field.lower() for e in errors)

    def test_invalid_profit_threshold(self, valid_config, tmp_path):
        """Test validation catches invalid profit threshold."""
        valid_config["profit_taking"]["thresholds"][0]["gain_pct"] = -0.5
        config_path = tmp_path / "settings.json"
        with open(config_path, "w") as f:
            json.dump(valid_config, f)

        mgr = ConfigManager(config_path)
        errors = mgr.validate()
        assert len(errors) >= 1
        assert any("gain_pct" in e.field for e in errors)

    def test_invalid_sell_percentage(self, valid_config, tmp_path):
        """Test validation catches invalid sell percentage."""
        valid_config["profit_taking"]["thresholds"][0]["sell_pct"] = 1.5
        config_path = tmp_path / "settings.json"
        with open(config_path, "w") as f:
            json.dump(valid_config, f)

        mgr = ConfigManager(config_path)
        errors = mgr.validate()
        assert len(errors) >= 1
        assert any("sell_pct" in e.field for e in errors)

    def test_invalid_log_level(self, valid_config, tmp_path):
        """Test validation catches invalid log level."""
        valid_config["structured_logging"]["level"] = "INVALID"
        config_path = tmp_path / "settings.json"
        with open(config_path, "w") as f:
            json.dump(valid_config, f)

        mgr = ConfigManager(config_path)
        errors = mgr.validate()
        assert len(errors) >= 1
        assert any("level" in e.field for e in errors)

    def test_invalid_api_port(self, valid_config, tmp_path):
        """Test validation catches invalid API port."""
        valid_config["api_server"]["port"] = 70000
        config_path = tmp_path / "settings.json"
        with open(config_path, "w") as f:
            json.dump(valid_config, f)

        mgr = ConfigManager(config_path)
        errors = mgr.validate()
        assert len(errors) >= 1
        assert any("port" in e.field for e in errors)

    def test_invalid_starting_equity(self, valid_config, tmp_path):
        """Test validation catches invalid starting equity."""
        valid_config["performance_tracker"]["starting_equity"] = -1000
        config_path = tmp_path / "settings.json"
        with open(config_path, "w") as f:
            json.dump(valid_config, f)

        mgr = ConfigManager(config_path)
        errors = mgr.validate()
        assert len(errors) >= 1
        assert any("starting_equity" in e.field for e in errors)


# ==============================================================================
# Test ConfigValidationError
# ==============================================================================


class TestConfigValidationError:
    """Test ConfigValidationError dataclass."""

    def test_error_creation(self):
        """Test creating validation error."""
        error = ConfigValidationError(
            field="risk.max_loss",
            message="Value too high",
            value=0.15,
        )
        assert error.field == "risk.max_loss"
        assert error.message == "Value too high"
        assert error.value == 0.15

    def test_error_without_value(self):
        """Test error without value."""
        error = ConfigValidationError(
            field="api.port",
            message="Invalid port",
        )
        assert error.value is None


# ==============================================================================
# Test Environment Variable Substitution
# ==============================================================================


class TestEnvVarSubstitution:
    """Test environment variable substitution."""

    def test_env_var_substitution(self, tmp_path, monkeypatch):
        """Test ${VAR} substitution."""
        monkeypatch.setenv("TEST_API_KEY", "secret123")

        config = {
            "api_key": "${TEST_API_KEY}",
            "other": "value",
        }
        config_path = tmp_path / "settings.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        mgr = ConfigManager(config_path)
        assert mgr.get("api_key") == "secret123"
        assert mgr.get("other") == "value"

    def test_missing_env_var(self, tmp_path):
        """Test missing env var tracking."""
        config = {
            "api_key": "${NONEXISTENT_VAR}",
        }
        config_path = tmp_path / "settings.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        mgr = ConfigManager(config_path)
        missing = mgr.get_missing_env_vars()
        assert "NONEXISTENT_VAR" in missing


# ==============================================================================
# Test Global Config Functions
# ==============================================================================


class TestGlobalConfigFunctions:
    """Test get_config and reload_config functions."""

    def test_get_config_singleton(self):
        """Test get_config returns same instance."""
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reload_config(self, config_file):
        """Test reload_config creates new instance."""
        old_cfg = get_config()
        new_cfg = reload_config(config_file)
        assert new_cfg is not old_cfg


# ==============================================================================
# Test Config Save
# ==============================================================================


class TestConfigSave:
    """Test configuration saving."""

    def test_save_config(self, config_manager, config_file):
        """Test saving config changes."""
        config_manager.update("brokerage.paper_trading", False)
        config_manager.save()

        # Reload and verify
        mgr2 = ConfigManager(config_file)
        assert mgr2.get("brokerage.paper_trading") is False
