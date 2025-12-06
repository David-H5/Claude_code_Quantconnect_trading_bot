#!/usr/bin/env python3
"""
PreToolUse hook to protect sensitive files from Claude Code access (v4.3).

v4.3 CHANGES:
- Security-critical files (.env, credentials): STILL blocked (safety)
- Other protected patterns: WARN and LOG but allow (smooth workflow)
- All blocked/skipped attempts are logged for review

This hook is called before Edit, Write, and Read operations.
Exit code 2 blocks the operation (only for security-critical).
Exit code 0 allows the operation.

Usage in .claude/settings.json:
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Edit|Write|Read",
      "hooks": [{
        "type": "command",
        "command": "python3 .claude/hooks/protect_files.py"
      }]
    }]
  }
}

Environment variables passed by Claude Code:
- TOOL_NAME: The tool being called (Edit, Write, Read)
- TOOL_INPUT: JSON string of the tool input parameters
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


# Files that should NEVER be accessed (SECURITY CRITICAL - blocks)
SECURITY_CRITICAL = [
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "config/credentials.json",
    "config/api_keys.json",
    "secrets.json",
    ".claude/settings.local.json",  # Local overrides may contain secrets
]

# Patterns that indicate sensitive files (WARN only, don't block)
SENSITIVE_PATTERNS = [
    "credential",
    "secret",
    "api_key",
    "apikey",
    "password",
    "token",
    ".pem",
    ".key",
    "private",
]

# Directories to warn about (don't block, just log)
PROTECTED_DIRS = [
    ".git/",  # Prevent git corruption
    "node_modules/",  # No need to read/write these
    "__pycache__/",
    ".venv/",
    "venv/",
]

# Files that should only be read, never written (warn only)
READ_ONLY_FILES = [
    "LICENSE",
    ".gitignore",
    ".pre-commit-config.yaml",
]

# Log file for security events
SECURITY_LOG_FILE = Path(".claude/security_log.json")


def normalize_path(file_path: str) -> str:
    """Normalize path for consistent comparison."""
    return file_path.replace("\\", "/").lower()


def log_security_event(event_type: str, file_path: str, tool_name: str, reason: str, action: str) -> None:
    """Log security events for review."""
    try:
        log = []
        if SECURITY_LOG_FILE.exists():
            log = json.loads(SECURITY_LOG_FILE.read_text())

        log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "file": file_path,
                "tool": tool_name,
                "reason": reason,
                "action": action,
            }
        )

        # Keep last 200 events
        log = log[-200:]
        SECURITY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        SECURITY_LOG_FILE.write_text(json.dumps(log, indent=2))
    except (OSError, json.JSONDecodeError):
        pass  # Don't fail on logging errors


def check_security_critical(file_path: str) -> tuple[bool, str]:
    """Check if file is security-critical (blocks)."""
    normalized = normalize_path(file_path)

    for protected in SECURITY_CRITICAL:
        if normalized.endswith(normalize_path(protected)):
            return True, f"security-critical: {protected}"

    return False, ""


def check_sensitive_pattern(file_path: str) -> tuple[bool, str]:
    """Check if file matches sensitive patterns (warns)."""
    normalized = normalize_path(file_path)

    for pattern in SENSITIVE_PATTERNS:
        if pattern in normalized:
            return True, f"contains sensitive pattern: {pattern}"

    return False, ""


def check_protected_dir(file_path: str) -> tuple[bool, str]:
    """Check if file is in protected directory (warns)."""
    normalized = normalize_path(file_path)

    for protected_dir in PROTECTED_DIRS:
        if normalize_path(protected_dir) in normalized:
            return True, f"in protected directory: {protected_dir}"

    return False, ""


def check_read_only(file_path: str, tool_name: str) -> tuple[bool, str]:
    """Check if file is read-only for write operations (warns)."""
    if tool_name not in ("Edit", "Write"):
        return False, ""

    original_name = Path(file_path).name.lower()

    for read_only in READ_ONLY_FILES:
        if original_name == read_only.lower():
            return True, f"read-only file: {read_only}"

    return False, ""


def get_file_path_from_input(tool_input: dict[str, str], tool_name: str) -> str:
    """Extract file path from tool input based on tool type."""
    if tool_name == "Read" or tool_name == "Edit" or tool_name == "Write":
        return tool_input.get("file_path", "")
    return ""


def main():
    """Main hook function."""
    # Get environment variables
    tool_name = os.environ.get("TOOL_NAME", "")
    tool_input_str = os.environ.get("TOOL_INPUT", "{}")

    # If no tool info, allow (fail open for non-file operations)
    if not tool_name:
        sys.exit(0)

    # Parse tool input
    try:
        tool_input = json.loads(tool_input_str)
    except json.JSONDecodeError:
        # Can't parse input, allow operation
        sys.exit(0)

    # Get file path from input
    file_path = get_file_path_from_input(tool_input, tool_name)

    # If no file path, allow
    if not file_path:
        sys.exit(0)

    # Check 1: Security-critical files (BLOCKS)
    is_critical, critical_reason = check_security_critical(file_path)
    if is_critical:
        print(f"🔒 BLOCKED: {tool_name} on security-critical file", file=sys.stderr)
        print(f"   File: {file_path}", file=sys.stderr)
        print(f"   Reason: {critical_reason}", file=sys.stderr)
        log_security_event("blocked", file_path, tool_name, critical_reason, "blocked")
        sys.exit(2)  # Block security-critical files

    # Check 2: Sensitive patterns (WARNS, logs, allows)
    is_sensitive, sensitive_reason = check_sensitive_pattern(file_path)
    if is_sensitive:
        print(f"⚠️ WARNING: {tool_name} on sensitive file (allowed)", file=sys.stderr)
        print(f"   File: {file_path}", file=sys.stderr)
        print(f"   Reason: {sensitive_reason}", file=sys.stderr)
        log_security_event("warning", file_path, tool_name, sensitive_reason, "allowed")
        # Continue - don't block

    # Check 3: Protected directories (WARNS, logs, allows)
    is_protected_dir, dir_reason = check_protected_dir(file_path)
    if is_protected_dir:
        print(f"⚠️ WARNING: {tool_name} in protected directory (allowed)", file=sys.stderr)
        print(f"   File: {file_path}", file=sys.stderr)
        print(f"   Reason: {dir_reason}", file=sys.stderr)
        log_security_event("warning", file_path, tool_name, dir_reason, "allowed")
        # Continue - don't block

    # Check 4: Read-only files (WARNS, logs, allows)
    is_read_only, ro_reason = check_read_only(file_path, tool_name)
    if is_read_only:
        print(f"⚠️ WARNING: {tool_name} on read-only file (allowed)", file=sys.stderr)
        print(f"   File: {file_path}", file=sys.stderr)
        print(f"   Reason: {ro_reason}", file=sys.stderr)
        log_security_event("warning", file_path, tool_name, ro_reason, "allowed")
        # Continue - don't block

    # Allow the operation
    sys.exit(0)


if __name__ == "__main__":
    main()
