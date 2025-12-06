"""
Centralized Overnight Configuration

Loads configuration from config/overnight.yaml with environment variable expansion.
Part of OVERNIGHT-002 refactoring based on docs/OVERNIGHT_SYSTEM_ANALYSIS.md

Usage:
    from utils.overnight_config import OvernightConfig, get_overnight_config

    config = get_overnight_config()
    max_runtime = config.max_runtime_hours
    discord_url = config.discord_webhook
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# Try to import yaml, fall back gracefully
try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.warning("PyYAML not installed, using default configuration")


@dataclass
class OvernightConfig:
    """
    Validated overnight configuration.

    All settings with sensible defaults that can be overridden via:
    1. config/overnight.yaml file
    2. Environment variables (highest priority)
    """

    # Session settings
    max_runtime_hours: float = 10.0
    max_idle_minutes: float = 30.0
    checkpoint_interval_minutes: float = 15.0
    max_continuation_attempts: int = 20

    # Budget settings
    max_cost_usd: float = 50.0
    cost_warning_threshold_pct: float = 80.0
    cost_critical_threshold_pct: float = 95.0

    # Recovery settings
    max_restarts: int = 5
    backoff_base_seconds: int = 30
    backoff_max_seconds: int = 600
    backoff_jitter_pct: int = 20

    # Enforcement settings
    continuous_mode: bool = True
    ric_mode: str = "SUGGESTED"  # ENFORCED | SUGGESTED | DISABLED
    min_completion_pct: int = 100
    require_p0: bool = True
    require_p1: bool = True
    require_p2: bool = False

    # Notification settings
    discord_webhook: str | None = None
    slack_webhook: str | None = None
    idle_warning_pct: float = 70.0
    enable_session_start: bool = True
    enable_checkpoints: bool = True
    enable_completion: bool = True

    # Watchdog settings
    watchdog_check_interval: int = 30
    memory_warning_pct: int = 80
    memory_critical_pct: int = 90
    cpu_warning_pct: int = 80
    min_checkpoint_interval_minutes: int = 15

    # Logging settings
    log_level: str = "INFO"
    log_format: str = "json"
    max_log_entries: int = 500
    max_progress_notes: int = 50

    # Path settings
    state_file: str = "logs/overnight_state.json"
    progress_file: str = "claude-progress.txt"
    session_notes: str = "claude-session-notes.md"
    ric_state: str = ".claude/state/ric.json"
    hook_logs: str = ".claude/logs/hook_activity.json"

    @classmethod
    def load(cls, config_path: Path | str | None = None) -> OvernightConfig:
        """
        Load configuration from YAML file with environment variable expansion.

        Args:
            config_path: Path to config file, defaults to config/overnight.yaml

        Returns:
            Loaded and validated configuration
        """
        if config_path is None:
            config_path = Path("config/overnight.yaml")
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            logger.info(f"Config file {config_path} not found, using defaults")
            return cls._apply_env_overrides(cls())

        if not HAS_YAML:
            logger.warning("PyYAML not installed, using defaults")
            return cls._apply_env_overrides(cls())

        try:
            with open(config_path) as f:
                raw_data = yaml.safe_load(f)

            if raw_data is None:
                return cls._apply_env_overrides(cls())

            # Expand environment variables
            expanded_data = cls._expand_env_vars(raw_data)

            # Flatten nested structure
            flat_data = cls._flatten_config(expanded_data)

            # Create config with loaded values
            config = cls._from_flat_dict(flat_data)

            # Apply any additional env overrides
            return cls._apply_env_overrides(config)

        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return cls._apply_env_overrides(cls())

    @classmethod
    def _expand_env_vars(cls, data: Any) -> Any:
        """Recursively expand ${VAR} patterns in config values."""
        if isinstance(data, dict):
            return {k: cls._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls._expand_env_vars(item) for item in data]
        elif isinstance(data, str):
            # Match ${VAR} or ${VAR:-default}
            pattern = r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}"

            def replacer(match):
                var_name = match.group(1)
                default = match.group(3)
                value = os.environ.get(var_name)
                if value is not None:
                    return value
                if default is not None:
                    return default
                return ""  # Empty string for undefined vars without default

            return re.sub(pattern, replacer, data)
        return data

    @classmethod
    def _flatten_config(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Flatten nested config structure to flat dict."""
        flat = {}

        # Session settings
        if "session" in data:
            session = data["session"]
            flat["max_runtime_hours"] = session.get("max_runtime_hours")
            flat["max_idle_minutes"] = session.get("max_idle_minutes")
            flat["checkpoint_interval_minutes"] = session.get("checkpoint_interval_minutes")
            flat["max_continuation_attempts"] = session.get("max_continuation_attempts")

        # Budget settings
        if "budget" in data:
            budget = data["budget"]
            flat["max_cost_usd"] = budget.get("max_cost_usd")
            flat["cost_warning_threshold_pct"] = budget.get("warning_threshold_pct")
            flat["cost_critical_threshold_pct"] = budget.get("critical_threshold_pct")

        # Recovery settings
        if "recovery" in data:
            recovery = data["recovery"]
            flat["max_restarts"] = recovery.get("max_restarts")
            flat["backoff_base_seconds"] = recovery.get("backoff_base_seconds")
            flat["backoff_max_seconds"] = recovery.get("backoff_max_seconds")
            flat["backoff_jitter_pct"] = recovery.get("backoff_jitter_pct")

        # Enforcement settings
        if "enforcement" in data:
            enforcement = data["enforcement"]
            flat["continuous_mode"] = enforcement.get("continuous_mode")
            flat["ric_mode"] = enforcement.get("ric_mode")
            flat["min_completion_pct"] = enforcement.get("min_completion_pct")
            flat["require_p0"] = enforcement.get("require_p0")
            flat["require_p1"] = enforcement.get("require_p1")
            flat["require_p2"] = enforcement.get("require_p2")

        # Notification settings
        if "notifications" in data:
            notif = data["notifications"]
            flat["discord_webhook"] = notif.get("discord_webhook") or None
            flat["slack_webhook"] = notif.get("slack_webhook") or None
            flat["idle_warning_pct"] = notif.get("idle_warning_pct")
            flat["enable_session_start"] = notif.get("enable_session_start")
            flat["enable_checkpoints"] = notif.get("enable_checkpoints")
            flat["enable_completion"] = notif.get("enable_completion")

        # Watchdog settings
        if "watchdog" in data:
            watchdog = data["watchdog"]
            flat["watchdog_check_interval"] = watchdog.get("check_interval_seconds")
            flat["memory_warning_pct"] = watchdog.get("memory_warning_pct")
            flat["memory_critical_pct"] = watchdog.get("memory_critical_pct")
            flat["cpu_warning_pct"] = watchdog.get("cpu_warning_pct")
            flat["min_checkpoint_interval_minutes"] = watchdog.get("min_checkpoint_interval_minutes")

        # Logging settings
        if "logging" in data:
            logging_cfg = data["logging"]
            flat["log_level"] = logging_cfg.get("level")
            flat["log_format"] = logging_cfg.get("format")
            flat["max_log_entries"] = logging_cfg.get("max_entries")
            flat["max_progress_notes"] = logging_cfg.get("max_progress_notes")

        # Path settings
        if "paths" in data:
            paths = data["paths"]
            flat["state_file"] = paths.get("state_file")
            flat["progress_file"] = paths.get("progress_file")
            flat["session_notes"] = paths.get("session_notes")
            flat["ric_state"] = paths.get("ric_state")
            flat["hook_logs"] = paths.get("hook_logs")

        # Filter out None values
        return {k: v for k, v in flat.items() if v is not None}

    @classmethod
    def _from_flat_dict(cls, data: dict[str, Any]) -> OvernightConfig:
        """Create config from flat dictionary, ignoring unknown keys."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def _apply_env_overrides(cls, config: OvernightConfig) -> OvernightConfig:
        """Apply environment variable overrides (highest priority)."""
        # Direct env var overrides
        env_mappings = {
            "OVERNIGHT_MAX_RUNTIME_HOURS": ("max_runtime_hours", float),
            "OVERNIGHT_MAX_IDLE_MINUTES": ("max_idle_minutes", float),
            "OVERNIGHT_MAX_COST_USD": ("max_cost_usd", float),
            "OVERNIGHT_MAX_RESTARTS": ("max_restarts", int),
            "OVERNIGHT_CONTINUOUS_MODE": ("continuous_mode", lambda x: x.lower() == "true"),
            "OVERNIGHT_RIC_MODE": ("ric_mode", str),
            "DISCORD_WEBHOOK_URL": ("discord_webhook", str),
            "SLACK_WEBHOOK_URL": ("slack_webhook", str),
        }

        for env_var, (field_name, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                try:
                    setattr(config, field_name, converter(value))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid value for {env_var}: {value}")

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        from dataclasses import asdict

        return asdict(self)

    def validate(self) -> list[str]:
        """Validate configuration, returning list of warnings."""
        warnings = []

        if self.max_runtime_hours <= 0:
            warnings.append("max_runtime_hours must be positive")

        if self.max_cost_usd <= 0:
            warnings.append("max_cost_usd must be positive")

        if self.cost_warning_threshold_pct >= self.cost_critical_threshold_pct:
            warnings.append("cost_warning_threshold_pct should be less than critical")

        if self.ric_mode not in ("ENFORCED", "SUGGESTED", "DISABLED"):
            warnings.append(f"Invalid ric_mode: {self.ric_mode}")

        return warnings


# Global instance cache
_config_cache: OvernightConfig | None = None


def get_overnight_config(reload: bool = False) -> OvernightConfig:
    """
    Get the overnight configuration (cached).

    Args:
        reload: Force reload from file

    Returns:
        Overnight configuration
    """
    global _config_cache
    if _config_cache is None or reload:
        _config_cache = OvernightConfig.load()
    return _config_cache


def reset_config_cache() -> None:
    """Reset the configuration cache (for testing)."""
    global _config_cache
    _config_cache = None
