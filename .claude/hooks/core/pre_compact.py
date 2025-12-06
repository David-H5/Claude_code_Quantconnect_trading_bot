#!/usr/bin/env python3
"""
PreCompact hook for Claude Code transcript backup.

This hook runs before context compaction (when approaching ~200K tokens) and:
1. Backs up the full conversation transcript
2. Creates a checkpoint commit with current progress
3. Sends a notification about context compaction
4. Logs context usage statistics

Referenced by .claude/settings.json PreCompact hook configuration.

Context compaction occurs when:
- Conversation exceeds ~70% of 200K token limit (~140K tokens)
- System needs to summarize older context to continue

This hook preserves the full conversation before summarization.
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from notify import notify_context_warning

    HAS_NOTIFICATIONS = True
except ImportError:
    HAS_NOTIFICATIONS = False

# Configuration
BACKUP_DIR = Path("logs/transcripts")
MAX_BACKUPS = 10  # Keep last 10 transcript backups


def get_transcript_from_hook_input() -> Path | None:
    """Get transcript path from Claude hook input (preferred method)."""
    tool_input = os.environ.get("CLAUDE_TOOL_INPUT", "")
    if tool_input:
        try:
            data = json.loads(tool_input)
            transcript_path = data.get("transcript_path")
            if transcript_path:
                path = Path(transcript_path)
                if path.exists():
                    return path
        except json.JSONDecodeError:
            pass
    return None


def get_transcript_path() -> Path | None:
    """Find the current session transcript file.

    First tries to get path from hook input (CLAUDE_TOOL_INPUT),
    then falls back to searching ~/.claude/ directory.
    """
    # Preferred: Get from hook input
    transcript = get_transcript_from_hook_input()
    if transcript:
        return transcript

    # Fallback: Search ~/.claude/ directory
    claude_dir = Path.home() / ".claude"

    if not claude_dir.exists():
        return None

    # Look for session files (most recent first)
    session_files = sorted(
        claude_dir.glob("*.jsonl"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if session_files:
        return session_files[0]

    # Also check for transcript directory
    transcript_dir = claude_dir / "transcripts"
    if transcript_dir.exists():
        transcript_files = sorted(
            transcript_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if transcript_files:
            return transcript_files[0]

    return None


def backup_transcript():
    """Backup the current conversation transcript."""
    transcript = get_transcript_path()

    if transcript is None:
        print("No transcript file found to backup", file=sys.stderr)
        return None

    # Create backup directory
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Generate backup filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_name = f"transcript-{timestamp}{transcript.suffix}"
    backup_path = BACKUP_DIR / backup_name

    try:
        shutil.copy2(transcript, backup_path)
        print(f"Transcript backed up to: {backup_path}")

        # Calculate size
        size_kb = backup_path.stat().st_size / 1024
        print(f"Transcript size: {size_kb:.1f} KB")

        return backup_path
    except Exception as e:
        print(f"Failed to backup transcript: {e}", file=sys.stderr)
        return None


def cleanup_old_backups():
    """Remove old transcript backups, keeping only MAX_BACKUPS."""
    if not BACKUP_DIR.exists():
        return

    backups = sorted(
        BACKUP_DIR.glob("transcript-*"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    # Remove excess backups
    for old_backup in backups[MAX_BACKUPS:]:
        try:
            old_backup.unlink()
            print(f"Removed old backup: {old_backup.name}")
        except Exception as e:
            print(f"Could not remove {old_backup}: {e}", file=sys.stderr)


def create_checkpoint():
    """Create a checkpoint commit before compaction."""
    try:
        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Stage all changes
            subprocess.run(["git", "add", "-A"], timeout=5)

            # Create checkpoint commit
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            commit_msg = f"checkpoint: Pre-compaction checkpoint at {timestamp}"
            result = subprocess.run(
                ["git", "commit", "-m", commit_msg, "--no-verify"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                print("Pre-compaction checkpoint created")
                return True
    except Exception as e:
        print(f"Could not create checkpoint: {e}", file=sys.stderr)
    return False


def update_progress_file():
    """Note context compaction in progress file."""
    progress_file = Path("claude-progress.txt")
    try:
        content = ""
        if progress_file.exists():
            content = progress_file.read_text()

        timestamp = datetime.now().isoformat()
        addition = (
            f"\n\n## Context Compaction\n- Time: {timestamp}\n- Note: Full transcript backed up before summarization\n"
        )

        progress_file.write_text(content + addition)
    except Exception as e:
        print(f"Could not update progress file: {e}", file=sys.stderr)


def estimate_context_usage() -> dict:
    """Estimate current context usage."""
    stats = {
        "timestamp": datetime.now().isoformat(),
        "estimated_tokens": "unknown",
        "compaction_reason": "approaching_limit",
    }

    # Try to estimate from transcript size
    transcript = get_transcript_path()
    if transcript and transcript.exists():
        size_bytes = transcript.stat().st_size
        # Rough estimate: ~4 chars per token
        estimated_tokens = size_bytes // 4
        stats["estimated_tokens"] = estimated_tokens
        stats["transcript_size_kb"] = round(size_bytes / 1024, 1)

    return stats


def log_compaction_event(backup_path: Path | None, stats: dict):
    """Log compaction event to history file."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    log_file = logs_dir / "compaction-history.jsonl"
    event = {
        **stats,
        "backup_path": str(backup_path) if backup_path else None,
        "checkpoint_created": backup_path is not None,
    }

    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"Could not log compaction event: {e}", file=sys.stderr)


def save_session_state():
    """Save session state for recovery after compaction."""
    try:
        # Import session state manager
        from session_state_manager import SessionStateManager

        manager = SessionStateManager()
        manager.record_compaction()

        # Create recovery context file
        recovery_summary = manager.get_recovery_summary()
        recovery_file = Path("claude-recovery-context.md")
        recovery_file.write_text(recovery_summary)

        print(f"Recovery context saved to: {recovery_file}")
        return True
    except ImportError:
        print("Session state manager not available, skipping state save")
        return False
    except Exception as e:
        print(f"Could not save session state: {e}", file=sys.stderr)
        return False


def main():
    """Main PreCompact hook execution."""
    print("Context compaction imminent, backing up transcript...")

    # Estimate context usage
    stats = estimate_context_usage()
    print(f"Estimated context: {stats.get('estimated_tokens', 'unknown')} tokens")

    # Backup transcript
    backup_path = backup_transcript()

    # Cleanup old backups
    cleanup_old_backups()

    # Save session state for recovery (NEW)
    save_session_state()

    # Create checkpoint commit
    create_checkpoint()

    # Update progress file
    update_progress_file()

    # Log compaction event
    log_compaction_event(backup_path, stats)

    # Send notification
    if HAS_NOTIFICATIONS:
        context_info = f"{stats.get('estimated_tokens', 'unknown')} tokens"
        notify_context_warning(
            current_pct=70,  # Approximate - compaction typically at 70%
            message=f"Context compaction starting. Transcript backed up. Est. {context_info}",
        )
    else:
        print("Context compaction logged (no notification webhooks configured)")

    print("Pre-compaction backup complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
