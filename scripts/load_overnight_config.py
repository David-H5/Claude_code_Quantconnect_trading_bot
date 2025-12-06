#!/usr/bin/env python3
"""
Load overnight config and export as environment variables.

Part of P1-2 integration from REMEDIATION_PLAN.md

Usage in bash:
    eval $(python3 scripts/load_overnight_config.py)

This script loads config/overnight.yaml via OvernightConfig and
outputs key values as shell variable assignments that can be sourced.
"""

import sys
from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.overnight_config import OvernightConfig


def main():
    """Load config and print shell variable assignments."""
    config = OvernightConfig.load()

    # Export key session values
    print(f"MAX_RUNTIME_HOURS={config.max_runtime_hours}")
    print(f"MAX_IDLE_MINUTES={config.max_idle_minutes}")
    print(f"CHECKPOINT_INTERVAL_MINUTES={config.checkpoint_interval_minutes}")
    print(f"MAX_CONTINUATION_ATTEMPTS={config.max_continuation_attempts}")

    # Export budget values
    print(f"MAX_COST_USD={config.max_cost_usd}")
    print(f"COST_WARNING_THRESHOLD_PCT={config.cost_warning_threshold_pct}")
    print(f"COST_CRITICAL_THRESHOLD_PCT={config.cost_critical_threshold_pct}")

    # Export recovery values
    print(f"MAX_RESTARTS={config.max_restarts}")
    print(f"BACKOFF_BASE_SECONDS={config.backoff_base_seconds}")
    print(f"BACKOFF_MAX_SECONDS={config.backoff_max_seconds}")
    print(f"BACKOFF_JITTER_PCT={config.backoff_jitter_pct}")

    # Export enforcement values (convert booleans to 0/1 for bash)
    print(f"CONTINUOUS_MODE={'1' if config.continuous_mode else '0'}")
    print(f"RIC_MODE={config.ric_mode}")
    print(f"MIN_COMPLETION_PCT={config.min_completion_pct}")
    print(f"REQUIRE_P0={'1' if config.require_p0 else '0'}")
    print(f"REQUIRE_P1={'1' if config.require_p1 else '0'}")
    print(f"REQUIRE_P2={'1' if config.require_p2 else '0'}")

    # Export notification values (escape special chars for bash)
    if config.discord_webhook:
        # Quote the URL to handle special characters
        print(f"DISCORD_WEBHOOK='{config.discord_webhook}'")
    if config.slack_webhook:
        print(f"SLACK_WEBHOOK='{config.slack_webhook}'")

    # Export watchdog values
    print(f"WATCHDOG_CHECK_INTERVAL={config.watchdog_check_interval}")
    print(f"MEMORY_WARNING_PCT={config.memory_warning_pct}")
    print(f"MEMORY_CRITICAL_PCT={config.memory_critical_pct}")
    print(f"MIN_CHECKPOINT_INTERVAL_MINUTES={config.min_checkpoint_interval_minutes}")

    # Export path values
    print(f"STATE_FILE='{config.state_file}'")
    print(f"PROGRESS_FILE='{config.progress_file}'")
    print(f"SESSION_NOTES='{config.session_notes}'")


if __name__ == "__main__":
    main()
