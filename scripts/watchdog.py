#!/usr/bin/env python3
"""
External watchdog process for autonomous Claude Code sessions.

This script runs as a separate process and monitors Claude Code sessions for:
- Maximum runtime exceeded
- Idle time exceeded (no activity)
- Cost budget exceeded
- Missing checkpoint commits

It can terminate the Claude Code process if safety limits are breached.

Usage:
    python scripts/watchdog.py [--config config/watchdog.json] [--test]

Configuration (config/watchdog.json):
{
    "max_runtime_hours": 10,
    "max_idle_minutes": 30,
    "max_cost_usd": 50.0,
    "checkpoint_interval_minutes": 15,
    "log_file": "logs/watchdog.log",
    "progress_file": "claude-progress.txt",
    "alert_email": null,
    "slack_webhook": null
}
"""

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


try:
    import psutil
except ImportError:
    print("Error: psutil is required. Install with: pip install psutil")
    sys.exit(1)

# Import notification module (optional but recommended)
# Try both import paths to support running as script or module
HAS_NOTIFICATIONS = False
try:
    from notify import (
        notify_checkpoint,
        notify_completion,
        notify_context_warning,
        notify_cost_alert,
        notify_error,
        notify_idle_warning,
        notify_recovery_attempt,
        notify_recovery_result,
        notify_session_start,
        notify_stuck_detected,
        notify_warning,
    )

    HAS_NOTIFICATIONS = True
except ImportError:
    try:
        # Try scripts.notify when running from project root
        from scripts.notify import (
            notify_checkpoint,
            notify_completion,
            notify_context_warning,
            notify_cost_alert,
            notify_error,
            notify_idle_warning,
            notify_recovery_attempt,
            notify_recovery_result,
            notify_session_start,
            notify_stuck_detected,
            notify_warning,
        )

        HAS_NOTIFICATIONS = True
    except ImportError:
        pass  # HAS_NOTIFICATIONS stays False

# Define stub functions if notify module not available
if not HAS_NOTIFICATIONS:

    def notify_session_start(*args, **kwargs):
        pass

    def notify_checkpoint(*args, **kwargs):
        pass

    def notify_warning(*args, **kwargs):
        pass

    def notify_error(*args, **kwargs):
        pass

    def notify_completion(*args, **kwargs):
        pass

    def notify_cost_alert(*args, **kwargs):
        pass

    def notify_context_warning(*args, **kwargs):
        pass

    def notify_idle_warning(*args, **kwargs):
        pass

    def notify_stuck_detected(*args, **kwargs):
        pass

    def notify_recovery_attempt(*args, **kwargs):
        pass

    def notify_recovery_result(*args, **kwargs):
        pass


@dataclass
class WatchdogConfig:
    """Watchdog configuration parameters."""

    max_runtime_hours: float = 10.0
    max_idle_minutes: float = 30.0
    max_cost_usd: float = 50.0
    checkpoint_interval_minutes: float = 15.0
    log_file: str = "logs/watchdog.log"
    progress_file: str = "claude-progress.txt"
    budget_file: str = "logs/budget.json"
    alert_email: str | None = None
    slack_webhook: str | None = None
    check_interval_seconds: float = 30.0
    claude_process_name: str = "claude"

    @classmethod
    def from_file(cls, path: Path) -> "WatchdogConfig":
        """Load configuration from JSON file."""
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        return cls()


class Watchdog:
    """
    External watchdog for Claude Code sessions.

    Monitors for safety limit violations and can terminate sessions.
    """

    def __init__(self, config: WatchdogConfig):
        self.config = config
        self.start_time = datetime.now()
        self.last_activity_time = datetime.now()
        self.last_checkpoint_time = datetime.now()
        self.running = False
        self.claude_pid: int | None = None
        self._setup_logging()
        self._setup_signals()

    def _setup_logging(self):
        """Configure logging."""
        log_dir = Path(self.config.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("watchdog")

    def _setup_signals(self):
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def find_claude_process(self) -> psutil.Process | None:
        """Find the Claude Code process."""
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                name = proc.info["name"] or ""
                cmdline = proc.info["cmdline"] or []

                # Check for claude process
                if self.config.claude_process_name in name.lower():
                    return proc

                # Check command line for claude
                cmdline_str = " ".join(cmdline).lower()
                if self.config.claude_process_name in cmdline_str:
                    return proc

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return None

    def check_runtime(self) -> tuple[bool, str]:
        """Check if maximum runtime exceeded."""
        elapsed = datetime.now() - self.start_time
        max_runtime = timedelta(hours=self.config.max_runtime_hours)

        if elapsed > max_runtime:
            return (
                False,
                f"Maximum runtime exceeded: {elapsed} > {max_runtime}",
            )
        return True, f"Runtime OK: {elapsed} / {max_runtime}"

    def check_idle_time(self) -> tuple[bool, str]:
        """Check if session is idle too long."""
        # Check progress file modification time
        progress_path = Path(self.config.progress_file)
        if progress_path.exists():
            mtime = datetime.fromtimestamp(progress_path.stat().st_mtime)
            self.last_activity_time = max(self.last_activity_time, mtime)

        idle_time = datetime.now() - self.last_activity_time
        max_idle = timedelta(minutes=self.config.max_idle_minutes)

        if idle_time > max_idle:
            return (
                False,
                f"Session idle too long: {idle_time} > {max_idle}",
            )
        return True, f"Idle time OK: {idle_time} / {max_idle}"

    def check_cost(self) -> tuple[bool, str]:
        """Check if cost budget exceeded."""
        budget_path = Path(self.config.budget_file)
        if not budget_path.exists():
            return True, "No budget file found, assuming OK"

        try:
            with open(budget_path) as f:
                budget = json.load(f)

            spent = budget.get("spent_today", 0.0)
            limit = self.config.max_cost_usd

            if spent >= limit:
                return False, f"Cost budget exceeded: ${spent:.2f} >= ${limit:.2f}"

            return True, f"Cost OK: ${spent:.2f} / ${limit:.2f}"
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Could not parse budget file: {e}")
            return True, "Budget file unreadable, assuming OK"

    def check_checkpoints(self) -> tuple[bool, str]:
        """Check if checkpoint commits are happening."""
        interval = timedelta(minutes=self.config.checkpoint_interval_minutes)
        since_checkpoint = datetime.now() - self.last_checkpoint_time

        # Check for recent git commits
        try:
            result = subprocess.run(
                [
                    "git",
                    "log",
                    "--since",
                    f"{int(self.config.checkpoint_interval_minutes)} minutes ago",
                    "--oneline",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                self.last_checkpoint_time = datetime.now()
                return True, "Recent checkpoint found"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        if since_checkpoint > interval * 2:  # Allow some grace period
            return (
                False,
                f"No checkpoint in {since_checkpoint}, expected every {interval}",
            )
        return True, f"Checkpoint OK: {since_checkpoint} since last"

    def check_process_health(self) -> tuple[bool, str]:
        """Check if Claude Code process is healthy."""
        proc = self.find_claude_process()
        if proc is None:
            return False, "Claude Code process not found"

        try:
            status = proc.status()
            if status in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                return False, f"Claude Code process is {status}"

            # Check CPU usage (very high might indicate stuck)
            cpu = proc.cpu_percent(interval=1)
            if cpu > 95:
                return True, f"High CPU usage: {cpu}% (may be processing)"

            # Update PID
            self.claude_pid = proc.pid
            return True, f"Process healthy (PID: {proc.pid}, CPU: {cpu}%)"

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            return False, f"Could not check process: {e}"

    def trigger_checkpoint(self) -> tuple[bool, str]:
        """Trigger a checkpoint commit if interval has passed.

        This actively creates checkpoints rather than just checking for them.
        Uses scripts/checkpoint.sh periodic command.
        """
        interval = timedelta(minutes=self.config.checkpoint_interval_minutes)
        since_checkpoint = datetime.now() - self.last_checkpoint_time

        if since_checkpoint < interval:
            return True, f"Checkpoint not due yet ({since_checkpoint} < {interval})"

        # Try to trigger checkpoint
        try:
            result = subprocess.run(
                ["bash", "scripts/checkpoint.sh", "periodic", str(int(self.config.checkpoint_interval_minutes))],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=Path(__file__).parent.parent,  # Project root
            )
            if result.returncode == 0:
                self.last_checkpoint_time = datetime.now()
                self.logger.info("Checkpoint triggered successfully")
                notify_checkpoint("Periodic", f"Auto-checkpoint at {datetime.now().strftime('%H:%M')}")
                return True, "Checkpoint created"
            else:
                self.logger.warning(f"Checkpoint failed: {result.stderr[:200]}")
                return True, f"Checkpoint attempt failed (non-critical): {result.stderr[:100]}"
        except subprocess.TimeoutExpired:
            self.logger.warning("Checkpoint script timed out")
            return True, "Checkpoint timed out (non-critical)"
        except FileNotFoundError:
            self.logger.debug("checkpoint.sh not found")
            return True, "Checkpoint script not available"
        except Exception as e:
            self.logger.warning(f"Checkpoint error: {e}")
            return True, f"Checkpoint error (non-critical): {e}"

    def terminate_session(self, reason: str):
        """Terminate the Claude Code session."""
        self.logger.warning(f"Terminating session: {reason}")

        # Send error notification via notify module
        notify_error(reason)

        # Try graceful termination first
        proc = self.find_claude_process()
        if proc:
            try:
                self.logger.info(f"Sending SIGTERM to PID {proc.pid}")
                proc.terminate()

                # Wait for graceful shutdown
                try:
                    proc.wait(timeout=10)
                    self.logger.info("Process terminated gracefully")
                except psutil.TimeoutExpired:
                    self.logger.warning("Graceful shutdown timeout, sending SIGKILL")
                    proc.kill()

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                self.logger.error(f"Could not terminate process: {e}")

        # Send alert via legacy method if configured (backwards compatibility)
        self._send_alert(reason)

        # Update progress file with termination reason
        self._update_progress_file(reason)

        # Calculate runtime for completion notification
        runtime = datetime.now() - self.start_time
        runtime_str = f"{runtime.total_seconds() / 3600:.1f} hours"
        task_desc = getattr(self, "task_description", "Unknown task")
        notify_completion(
            f"Session terminated: {reason}\n\nTask: {task_desc}",
            success=False,
            stats={"runtime": runtime_str},
        )

    def _send_alert(self, message: str):
        """Send alert notification."""
        if self.config.slack_webhook:
            try:
                import urllib.request

                data = json.dumps({"text": f"🚨 Watchdog Alert: {message}"}).encode()
                req = urllib.request.Request(
                    self.config.slack_webhook,
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=5)
                self.logger.info("Slack alert sent")
            except Exception as e:
                self.logger.error(f"Failed to send Slack alert: {e}")

        if self.config.alert_email:
            self.logger.info(f"Email alert would be sent to {self.config.alert_email}")
            # Email implementation would go here

    def _update_progress_file(self, reason: str):
        """Update progress file with watchdog status."""
        progress_path = Path(self.config.progress_file)
        try:
            content = ""
            if progress_path.exists():
                content = progress_path.read_text()

            timestamp = datetime.now().isoformat()
            addition = f"\n\n## Watchdog Termination\n- Time: {timestamp}\n- Reason: {reason}\n"

            progress_path.write_text(content + addition)
        except Exception as e:
            self.logger.error(f"Could not update progress file: {e}")

    def check_ric_status(self) -> tuple[bool, str]:
        """Check RIC Loop status for monitoring purposes."""
        ric_state_file = Path(".claude/state/ric.json")
        if not ric_state_file.exists():
            return True, "No active RIC session"

        try:
            with open(ric_state_file) as f:
                state = json.load(f)

            if not state.get("upgrade_id"):
                return True, "No active RIC session"

            phase_names = {
                0: "Research",
                1: "Upgrade Path",
                2: "Checklist",
                3: "Coding",
                4: "Double-Check",
                5: "Introspection",
                6: "Metacognition",
                7: "Integration",
            }

            iteration = state.get("current_iteration", 1)
            max_iter = state.get("max_iterations", 5)
            phase = state.get("current_phase", 0)
            phase_name = phase_names.get(phase, "Unknown")

            insights = state.get("insights", [])
            p0_open = len([i for i in insights if i.get("priority") == "P0" and i.get("status") != "resolved"])
            p1_open = len([i for i in insights if i.get("priority") == "P1" and i.get("status") != "resolved"])
            p2_open = len([i for i in insights if i.get("priority") == "P2" and i.get("status") != "resolved"])

            status_msg = (
                f"RIC Active: Iter {iteration}/{max_iter}, Phase {phase} ({phase_name}), "
                f"Open: P0={p0_open} P1={p1_open} P2={p2_open}"
            )

            # Log RIC status for visibility
            self.logger.info(status_msg)

            return True, status_msg

        except (json.JSONDecodeError, OSError, KeyError) as e:
            return True, f"Could not read RIC state: {e}"

    def run_checks(self) -> tuple[bool, list[str]]:
        """Run all safety checks and maintenance tasks."""
        checks = [
            ("Runtime", self.check_runtime),
            ("Idle Time", self.check_idle_time),
            ("Cost", self.check_cost),
            ("Checkpoints", self.check_checkpoints),
            ("Checkpoint Trigger", self.trigger_checkpoint),  # Active checkpoint creation
            ("Process Health", self.check_process_health),
            ("RIC Status", self.check_ric_status),
        ]

        all_ok = True
        messages = []

        for name, check_func in checks:
            try:
                ok, message = check_func()
                status = "✓" if ok else "✗"
                messages.append(f"{status} {name}: {message}")
                if not ok:
                    all_ok = False
                    self.logger.warning(f"{name} check failed: {message}")
            except Exception as e:
                messages.append(f"? {name}: Error - {e}")
                self.logger.error(f"{name} check error: {e}")

        return all_ok, messages

    def start(self, task_description: str = "Autonomous development session"):
        """Start the watchdog monitoring loop."""
        self.running = True
        self.task_description = task_description
        self.logger.info("Watchdog started")
        self.logger.info(f"Max runtime: {self.config.max_runtime_hours} hours")
        self.logger.info(f"Max idle: {self.config.max_idle_minutes} minutes")
        self.logger.info(f"Max cost: ${self.config.max_cost_usd}")
        self.logger.info(f"Check interval: {self.config.check_interval_seconds}s")

        # Send session start notification
        if HAS_NOTIFICATIONS:
            config_dict = {
                "max_runtime_hours": self.config.max_runtime_hours,
                "max_cost_usd": self.config.max_cost_usd,
                "checkpoint_interval_minutes": self.config.checkpoint_interval_minutes,
            }
            notify_session_start(task_description, config_dict)

        # Track for idle warnings (only warn once per threshold)
        idle_warning_sent = False
        cost_warning_threshold = 0.8  # Warn at 80%
        cost_warning_sent = False

        while self.running:
            try:
                all_ok, messages = self.run_checks()

                # Log status periodically
                self.logger.debug("\n".join(messages))

                # Check for idle warning (before termination)
                idle_time = datetime.now() - self.last_activity_time
                idle_minutes = idle_time.total_seconds() / 60
                idle_warning_threshold = self.config.max_idle_minutes * 0.7
                if idle_minutes > idle_warning_threshold and not idle_warning_sent:
                    notify_idle_warning(idle_minutes, self.config.max_idle_minutes)
                    idle_warning_sent = True

                # Check for cost warning
                budget_path = Path(self.config.budget_file)
                if budget_path.exists() and not cost_warning_sent:
                    try:
                        with open(budget_path) as f:
                            budget = json.load(f)
                        spent = budget.get("spent_today", 0.0)
                        if spent >= self.config.max_cost_usd * cost_warning_threshold:
                            notify_cost_alert(spent, self.config.max_cost_usd)
                            cost_warning_sent = True
                    except (json.JSONDecodeError, KeyError):
                        pass

                if not all_ok:
                    # Find the first failure
                    failure = next((m for m in messages if m.startswith("✗")), "Unknown")
                    self.terminate_session(failure)
                    break

                time.sleep(self.config.check_interval_seconds)

            except Exception as e:
                self.logger.error(f"Watchdog error: {e}")
                notify_warning(f"Watchdog error: {e}")
                time.sleep(self.config.check_interval_seconds)

        self.logger.info("Watchdog stopped")

    def test(self):
        """Run a single check cycle for testing."""
        self.logger.info("Running test checks...")
        all_ok, messages = self.run_checks()

        print("\nWatchdog Test Results:")
        print("-" * 40)
        for message in messages:
            print(message)
        print("-" * 40)
        print(f"Overall: {'PASS' if all_ok else 'FAIL'}")

        return all_ok


def main():
    parser = argparse.ArgumentParser(description="Watchdog for autonomous Claude Code sessions")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/watchdog.json"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run single check cycle and exit",
    )
    parser.add_argument(
        "--max-runtime",
        type=float,
        help="Override max runtime hours",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        help="Override max cost USD",
    )

    args = parser.parse_args()

    # Load configuration
    config = WatchdogConfig.from_file(args.config)

    # Apply overrides
    if args.max_runtime:
        config.max_runtime_hours = args.max_runtime
    if args.max_cost:
        config.max_cost_usd = args.max_cost

    # Create and run watchdog
    watchdog = Watchdog(config)

    if args.test:
        success = watchdog.test()
        sys.exit(0 if success else 1)
    else:
        watchdog.start()


if __name__ == "__main__":
    main()
