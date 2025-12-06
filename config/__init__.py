"""
Configuration Management Module

Layer: 1 (Infrastructure)
May import from: Layer 0 (utils)
May be imported by: Layers 2-4

Provides centralized configuration loading, validation, and access
for all trading bot components.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


logger = logging.getLogger(__name__)

# Environment variables that are required for live trading
# (warn on missing, but don't fail - allows local development)
REQUIRED_SECRETS_LIVE = {
    "QC_USER_ID",
    "QC_API_TOKEN",
}

# Environment variables that cause errors if missing when referenced
CRITICAL_SECRETS = {
    "SCHWAB_CLIENT_ID",
    "SCHWAB_CLIENT_SECRET",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
}


@dataclass
class ProfitThreshold:
    """Single profit-taking threshold configuration."""

    gain_pct: float
    sell_pct: float
    description: str = ""


@dataclass
class RiskConfig:
    """Risk management configuration."""

    max_position_size_pct: float = 0.25
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    max_risk_per_trade_pct: float = 0.02
    max_consecutive_losses: int = 5
    require_human_reset: bool = True


@dataclass
class ProfitTakingConfig:
    """Profit-taking configuration."""

    enabled: bool = True
    thresholds: list[ProfitThreshold] = field(default_factory=list)
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 0.25


@dataclass
class OrderExecutionConfig:
    """Order execution and cancel/replace configuration."""

    cancel_replace_enabled: bool = True
    max_bid_increase_pct: float = 1.00
    bid_increment_pct: float = 0.10
    cancel_after_seconds: int = 30
    max_cancel_replace_attempts: int = 10
    use_mid_price: bool = True


@dataclass
class OptionsScannerConfig:
    """Options scanner configuration."""

    enabled: bool = True
    target_delta_range: tuple = (0.20, 0.40)
    max_days_to_expiry: int = 45
    min_days_to_expiry: int = 7
    min_open_interest: int = 100
    min_volume: int = 50
    iv_percentile_threshold: float = 0.30
    underpriced_threshold: float = 0.10


@dataclass
class MovementScannerConfig:
    """Movement scanner configuration."""

    enabled: bool = True
    min_movement_pct: float = 0.02
    max_movement_pct: float = 0.04
    scan_interval_seconds: int = 60
    require_news_corroboration: bool = True
    volume_surge_threshold: float = 2.0


@dataclass
class LoggingConfig:
    """Structured logging configuration (UPGRADE-011)."""

    enabled: bool = True
    level: str = "INFO"
    console_enabled: bool = True
    file_enabled: bool = True
    file_path: str = "logs/trading_bot.jsonl"
    max_file_size_mb: int = 50
    backup_count: int = 10
    compress_rotated: bool = True
    object_store_enabled: bool = False
    object_store_buffer_size: int = 100
    object_store_flush_interval: int = 60


@dataclass
class PerformanceConfig:
    """Performance tracker configuration (UPGRADE-011)."""

    enabled: bool = True
    starting_equity: float = 100000.0
    track_sessions: bool = True
    session_types: list[str] = field(default_factory=lambda: ["daily", "weekly", "monthly"])
    persist_to_object_store: bool = True
    persist_interval_minutes: int = 15
    max_trade_history: int = 10000


@dataclass
class APIConfig:
    """REST API server configuration (UPGRADE-011)."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8080
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    websocket_enabled: bool = True
    rate_limit_per_minute: int = 100
    api_key_required: bool = False
    api_key: str = ""


@dataclass
class AlertingConfig:
    """Alerting system configuration (UPGRADE-011 expansion)."""

    enabled: bool = True
    # Alert channels
    console_alerts: bool = True
    email_enabled: bool = False
    email_recipients: list[str] = field(default_factory=list)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    discord_enabled: bool = False
    discord_webhook_url: str = ""
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    # Alert thresholds
    min_severity: str = "WARNING"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    rate_limit_per_minute: int = 10
    aggregate_similar: bool = True
    aggregation_window_seconds: int = 60
    # Alert categories
    alert_on_circuit_breaker: bool = True
    alert_on_large_loss: bool = True
    large_loss_threshold_pct: float = 0.02
    alert_on_error_spike: bool = True
    error_spike_threshold: int = 10
    alert_on_service_degradation: bool = True


@dataclass
class ErrorHandlingConfig:
    """Error handling configuration (UPGRADE-011 expansion)."""

    enabled: bool = True
    max_history: int = 1000
    # Retry settings
    default_max_retries: int = 3
    default_base_delay: float = 1.0
    default_max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter_enabled: bool = True
    # Recovery settings
    auto_recovery_enabled: bool = True
    recovery_check_interval_seconds: int = 30
    max_recovery_attempts: int = 3
    # Service health
    health_check_interval_seconds: int = 60
    degraded_threshold_errors: int = 5
    failed_threshold_errors: int = 10
    recovery_success_threshold: int = 3


@dataclass
class ConfigValidationError:
    """Configuration validation error."""

    field: str
    message: str
    value: Any = None


class ConfigManager:
    """
    Centralized configuration manager.

    Loads settings from JSON, validates them, and provides typed access.
    Supports environment variable substitution for secrets.
    """

    def __init__(self, config_path: Path | None = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to settings.json (default: config/settings.json)
        """
        if config_path is None:
            config_path = Path(__file__).parent / "settings.json"

        self.config_path = Path(config_path)
        self._raw_config: dict[str, Any] = {}
        self._missing_env_vars: set[str] = set()
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path) as f:
            self._raw_config = json.load(f)

        # Substitute environment variables
        self._substitute_env_vars(self._raw_config)

    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute ${VAR} patterns with environment variables."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = self._substitute_env_vars(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = self._substitute_env_vars(item)
        elif isinstance(obj, str):
            # Match ${VAR_NAME} pattern
            pattern = r"\$\{([^}]+)\}"
            matches = re.findall(pattern, obj)
            for var_name in matches:
                env_value = os.environ.get(var_name, "")
                if not env_value:
                    self._missing_env_vars.add(var_name)
                    if var_name in REQUIRED_SECRETS_LIVE:
                        logger.warning(
                            "Required environment variable not set: %s. " "This is needed for live trading.",
                            var_name,
                        )
                    elif var_name in CRITICAL_SECRETS:
                        logger.warning(
                            "Environment variable %s not set. " "Related functionality may not work.",
                            var_name,
                        )
                    else:
                        logger.debug("Optional environment variable not set: %s", var_name)
                obj = obj.replace(f"${{{var_name}}}", env_value)
            return obj
        return obj

    def get_missing_env_vars(self) -> set[str]:
        """Get set of environment variables that were referenced but not set."""
        return self._missing_env_vars.copy()

    def validate_for_live_trading(self) -> bool:
        """
        Check if all required environment variables are set for live trading.

        Returns:
            True if all required vars are set, False otherwise
        """
        missing_required = self._missing_env_vars & REQUIRED_SECRETS_LIVE
        if missing_required:
            logger.error(
                "Cannot start live trading. Missing required environment variables: %s",
                missing_required,
            )
            return False
        return True

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., 'risk_management.max_daily_loss_pct')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._raw_config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_risk_config(self) -> RiskConfig:
        """Get risk management configuration."""
        risk = self._raw_config.get("risk_management", {})
        return RiskConfig(
            max_position_size_pct=risk.get("max_position_size_pct", 0.25),
            max_daily_loss_pct=risk.get("max_daily_loss_pct", 0.03),
            max_drawdown_pct=risk.get("max_drawdown_pct", 0.10),
            max_risk_per_trade_pct=risk.get("max_risk_per_trade_pct", 0.02),
            max_consecutive_losses=risk.get("max_consecutive_losses", 5),
            require_human_reset=risk.get("require_human_reset", True),
        )

    def get_profit_taking_config(self) -> ProfitTakingConfig:
        """Get profit-taking configuration."""
        pt = self._raw_config.get("profit_taking", {})
        thresholds = [
            ProfitThreshold(
                gain_pct=t.get("gain_pct", 0),
                sell_pct=t.get("sell_pct", 0),
                description=t.get("description", ""),
            )
            for t in pt.get("thresholds", [])
        ]
        return ProfitTakingConfig(
            enabled=pt.get("enabled", True),
            thresholds=thresholds,
            trailing_stop_enabled=pt.get("trailing_stop_enabled", True),
            trailing_stop_pct=pt.get("trailing_stop_pct", 0.25),
        )

    def get_order_execution_config(self) -> OrderExecutionConfig:
        """Get order execution configuration."""
        oe = self._raw_config.get("order_execution", {})
        return OrderExecutionConfig(
            cancel_replace_enabled=oe.get("cancel_replace_enabled", True),
            max_bid_increase_pct=oe.get("max_bid_increase_pct", 1.00),
            bid_increment_pct=oe.get("bid_increment_pct", 0.10),
            cancel_after_seconds=oe.get("cancel_after_seconds", 30),
            max_cancel_replace_attempts=oe.get("max_cancel_replace_attempts", 10),
            use_mid_price=oe.get("use_mid_price", True),
        )

    def get_options_scanner_config(self) -> OptionsScannerConfig:
        """Get options scanner configuration."""
        os_cfg = self._raw_config.get("options_scanner", {})
        delta_range = os_cfg.get("target_delta_range", [0.20, 0.40])
        return OptionsScannerConfig(
            enabled=os_cfg.get("enabled", True),
            target_delta_range=tuple(delta_range),
            max_days_to_expiry=os_cfg.get("max_days_to_expiry", 45),
            min_days_to_expiry=os_cfg.get("min_days_to_expiry", 7),
            min_open_interest=os_cfg.get("min_open_interest", 100),
            min_volume=os_cfg.get("min_volume", 50),
            iv_percentile_threshold=os_cfg.get("iv_percentile_threshold", 0.30),
            underpriced_threshold=os_cfg.get("underpriced_threshold", 0.10),
        )

    def get_movement_scanner_config(self) -> MovementScannerConfig:
        """Get movement scanner configuration."""
        ms = self._raw_config.get("movement_scanner", {})
        return MovementScannerConfig(
            enabled=ms.get("enabled", True),
            min_movement_pct=ms.get("min_movement_pct", 0.02),
            max_movement_pct=ms.get("max_movement_pct", 0.04),
            scan_interval_seconds=ms.get("scan_interval_seconds", 60),
            require_news_corroboration=ms.get("require_news_corroboration", True),
            volume_surge_threshold=ms.get("volume_surge_threshold", 2.0),
        )

    def get_indicator_config(self, indicator: str) -> dict[str, Any]:
        """Get configuration for a specific technical indicator."""
        indicators = self._raw_config.get("technical_indicators", {})
        return indicators.get(indicator, {})

    def get_llm_config(self, provider: str | None = None) -> dict[str, Any]:
        """Get LLM configuration, optionally for a specific provider."""
        llm = self._raw_config.get("llm_integration", {})
        if provider:
            return llm.get("providers", {}).get(provider, {})
        return llm

    def get_logging_config(self) -> LoggingConfig:
        """Get structured logging configuration (UPGRADE-011)."""
        log_cfg = self._raw_config.get("structured_logging", {})
        return LoggingConfig(
            enabled=log_cfg.get("enabled", True),
            level=log_cfg.get("level", "INFO"),
            console_enabled=log_cfg.get("console_enabled", True),
            file_enabled=log_cfg.get("file_enabled", True),
            file_path=log_cfg.get("file_path", "logs/trading_bot.jsonl"),
            max_file_size_mb=log_cfg.get("max_file_size_mb", 50),
            backup_count=log_cfg.get("backup_count", 10),
            compress_rotated=log_cfg.get("compress_rotated", True),
            object_store_enabled=log_cfg.get("object_store_enabled", False),
            object_store_buffer_size=log_cfg.get("object_store_buffer_size", 100),
            object_store_flush_interval=log_cfg.get("object_store_flush_interval", 60),
        )

    def get_performance_config(self) -> PerformanceConfig:
        """Get performance tracker configuration (UPGRADE-011)."""
        perf_cfg = self._raw_config.get("performance_tracker", {})
        return PerformanceConfig(
            enabled=perf_cfg.get("enabled", True),
            starting_equity=perf_cfg.get("starting_equity", 100000.0),
            track_sessions=perf_cfg.get("track_sessions", True),
            session_types=perf_cfg.get("session_types", ["daily", "weekly", "monthly"]),
            persist_to_object_store=perf_cfg.get("persist_to_object_store", True),
            persist_interval_minutes=perf_cfg.get("persist_interval_minutes", 15),
            max_trade_history=perf_cfg.get("max_trade_history", 10000),
        )

    def get_api_config(self) -> APIConfig:
        """Get REST API server configuration (UPGRADE-011)."""
        api_cfg = self._raw_config.get("api_server", {})
        return APIConfig(
            enabled=api_cfg.get("enabled", True),
            host=api_cfg.get("host", "127.0.0.1"),
            port=api_cfg.get("port", 8080),
            cors_origins=api_cfg.get("cors_origins", ["http://localhost:3000"]),
            websocket_enabled=api_cfg.get("websocket_enabled", True),
            rate_limit_per_minute=api_cfg.get("rate_limit_per_minute", 100),
            api_key_required=api_cfg.get("api_key_required", False),
            api_key=api_cfg.get("api_key", ""),
        )

    def get_alerting_config(self) -> AlertingConfig:
        """Get alerting system configuration (UPGRADE-011 expansion)."""
        alert_cfg = self._raw_config.get("alerting", {})
        return AlertingConfig(
            enabled=alert_cfg.get("enabled", True),
            console_alerts=alert_cfg.get("console_alerts", True),
            email_enabled=alert_cfg.get("email_enabled", False),
            email_recipients=alert_cfg.get("email_recipients", []),
            smtp_host=alert_cfg.get("smtp_host", ""),
            smtp_port=alert_cfg.get("smtp_port", 587),
            smtp_username=alert_cfg.get("smtp_username", ""),
            smtp_password=alert_cfg.get("smtp_password", ""),
            discord_enabled=alert_cfg.get("discord_enabled", False),
            discord_webhook_url=alert_cfg.get("discord_webhook_url", ""),
            slack_enabled=alert_cfg.get("slack_enabled", False),
            slack_webhook_url=alert_cfg.get("slack_webhook_url", ""),
            min_severity=alert_cfg.get("min_severity", "WARNING"),
            rate_limit_per_minute=alert_cfg.get("rate_limit_per_minute", 10),
            aggregate_similar=alert_cfg.get("aggregate_similar", True),
            aggregation_window_seconds=alert_cfg.get("aggregation_window_seconds", 60),
            alert_on_circuit_breaker=alert_cfg.get("alert_on_circuit_breaker", True),
            alert_on_large_loss=alert_cfg.get("alert_on_large_loss", True),
            large_loss_threshold_pct=alert_cfg.get("large_loss_threshold_pct", 0.02),
            alert_on_error_spike=alert_cfg.get("alert_on_error_spike", True),
            error_spike_threshold=alert_cfg.get("error_spike_threshold", 10),
            alert_on_service_degradation=alert_cfg.get("alert_on_service_degradation", True),
        )

    def get_error_handling_config(self) -> ErrorHandlingConfig:
        """Get error handling configuration (UPGRADE-011 expansion)."""
        err_cfg = self._raw_config.get("error_handling", {})
        return ErrorHandlingConfig(
            enabled=err_cfg.get("enabled", True),
            max_history=err_cfg.get("max_history", 1000),
            default_max_retries=err_cfg.get("default_max_retries", 3),
            default_base_delay=err_cfg.get("default_base_delay", 1.0),
            default_max_delay=err_cfg.get("default_max_delay", 60.0),
            exponential_backoff=err_cfg.get("exponential_backoff", True),
            jitter_enabled=err_cfg.get("jitter_enabled", True),
            auto_recovery_enabled=err_cfg.get("auto_recovery_enabled", True),
            recovery_check_interval_seconds=err_cfg.get("recovery_check_interval_seconds", 30),
            max_recovery_attempts=err_cfg.get("max_recovery_attempts", 3),
            health_check_interval_seconds=err_cfg.get("health_check_interval_seconds", 60),
            degraded_threshold_errors=err_cfg.get("degraded_threshold_errors", 5),
            failed_threshold_errors=err_cfg.get("failed_threshold_errors", 10),
            recovery_success_threshold=err_cfg.get("recovery_success_threshold", 3),
        )

    def validate(self) -> list[ConfigValidationError]:
        """
        Validate configuration values (UPGRADE-011).

        Returns:
            List of validation errors (empty if valid)
        """
        errors: list[ConfigValidationError] = []

        # Validate risk management
        risk = self._raw_config.get("risk_management", {})
        if risk.get("max_daily_loss_pct", 0) > 0.10:
            errors.append(
                ConfigValidationError(
                    field="risk_management.max_daily_loss_pct",
                    message="Daily loss limit > 10% is extremely risky",
                    value=risk.get("max_daily_loss_pct"),
                )
            )
        if risk.get("max_drawdown_pct", 0) > 0.25:
            errors.append(
                ConfigValidationError(
                    field="risk_management.max_drawdown_pct",
                    message="Drawdown limit > 25% is extremely risky",
                    value=risk.get("max_drawdown_pct"),
                )
            )

        # Validate profit taking
        pt = self._raw_config.get("profit_taking", {})
        thresholds = pt.get("thresholds", [])
        for i, t in enumerate(thresholds):
            if t.get("gain_pct", 0) <= 0:
                errors.append(
                    ConfigValidationError(
                        field=f"profit_taking.thresholds[{i}].gain_pct",
                        message="Gain percentage must be positive",
                        value=t.get("gain_pct"),
                    )
                )
            if not 0 < t.get("sell_pct", 0) <= 1:
                errors.append(
                    ConfigValidationError(
                        field=f"profit_taking.thresholds[{i}].sell_pct",
                        message="Sell percentage must be between 0 and 1",
                        value=t.get("sell_pct"),
                    )
                )

        # Validate logging config
        log_cfg = self._raw_config.get("structured_logging", {})
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        level = log_cfg.get("level", "INFO").upper()
        if level not in valid_levels:
            errors.append(
                ConfigValidationError(
                    field="structured_logging.level",
                    message=f"Invalid log level. Must be one of: {valid_levels}",
                    value=level,
                )
            )

        # Validate API config
        api_cfg = self._raw_config.get("api_server", {})
        port = api_cfg.get("port", 8080)
        if not 1 <= port <= 65535:
            errors.append(
                ConfigValidationError(
                    field="api_server.port",
                    message="Port must be between 1 and 65535",
                    value=port,
                )
            )

        # Validate performance tracker
        perf_cfg = self._raw_config.get("performance_tracker", {})
        equity = perf_cfg.get("starting_equity", 100000)
        if equity <= 0:
            errors.append(
                ConfigValidationError(
                    field="performance_tracker.starting_equity",
                    message="Starting equity must be positive",
                    value=equity,
                )
            )

        # Validate alerting config (UPGRADE-011 expansion)
        alert_cfg = self._raw_config.get("alerting", {})
        alert_severity = alert_cfg.get("min_severity", "WARNING").upper()
        if alert_severity not in valid_levels:
            errors.append(
                ConfigValidationError(
                    field="alerting.min_severity",
                    message=f"Invalid severity. Must be one of: {valid_levels}",
                    value=alert_severity,
                )
            )
        if alert_cfg.get("rate_limit_per_minute", 10) < 1:
            errors.append(
                ConfigValidationError(
                    field="alerting.rate_limit_per_minute",
                    message="Rate limit must be at least 1 per minute",
                    value=alert_cfg.get("rate_limit_per_minute"),
                )
            )
        if alert_cfg.get("large_loss_threshold_pct", 0.02) > 0.10:
            errors.append(
                ConfigValidationError(
                    field="alerting.large_loss_threshold_pct",
                    message="Large loss threshold > 10% may miss critical alerts",
                    value=alert_cfg.get("large_loss_threshold_pct"),
                )
            )

        # Validate error handling config (UPGRADE-011 expansion)
        err_cfg = self._raw_config.get("error_handling", {})
        if err_cfg.get("default_max_retries", 3) < 0:
            errors.append(
                ConfigValidationError(
                    field="error_handling.default_max_retries",
                    message="Max retries cannot be negative",
                    value=err_cfg.get("default_max_retries"),
                )
            )
        if err_cfg.get("default_base_delay", 1.0) <= 0:
            errors.append(
                ConfigValidationError(
                    field="error_handling.default_base_delay",
                    message="Base delay must be positive",
                    value=err_cfg.get("default_base_delay"),
                )
            )
        if err_cfg.get("max_history", 1000) < 100:
            errors.append(
                ConfigValidationError(
                    field="error_handling.max_history",
                    message="Error history should be at least 100 for debugging",
                    value=err_cfg.get("max_history"),
                )
            )

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0

    def save(self) -> None:
        """Save current configuration to file."""
        with open(self.config_path, "w") as f:
            json.dump(self._raw_config, f, indent=2)

    def update(self, key: str, value: Any) -> None:
        """
        Update a configuration value.

        Args:
            key: Dot-notation key
            value: New value
        """
        keys = key.split(".")
        obj = self._raw_config

        for k in keys[:-1]:
            if k not in obj:
                obj[k] = {}
            obj = obj[k]

        obj[keys[-1]] = value


# Global config instance
_config: ConfigManager | None = None


def get_config() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config
    if _config is None:
        _config = ConfigManager()
    return _config


def reload_config(config_path: Path | None = None) -> ConfigManager:
    """Reload configuration from file."""
    global _config
    _config = ConfigManager(config_path)
    return _config


__all__ = [
    "ConfigManager",
    "RiskConfig",
    "ProfitTakingConfig",
    "OrderExecutionConfig",
    "OptionsScannerConfig",
    "MovementScannerConfig",
    "ProfitThreshold",
    # UPGRADE-011: New config classes
    "LoggingConfig",
    "PerformanceConfig",
    "APIConfig",
    "ConfigValidationError",
    # UPGRADE-011 Expansion: Alerting and Error Handling
    "AlertingConfig",
    "ErrorHandlingConfig",
    # Functions
    "get_config",
    "reload_config",
]
