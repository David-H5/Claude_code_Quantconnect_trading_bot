#!/usr/bin/env python3
"""
Research Auto-Persist Hook (v4.3)

This hook automatically persists research findings during sessions.
NEVER blocks - always auto-creates/appends to docs to prevent data loss.

Triggers on: PostToolUse for WebSearch, WebFetch
Purpose: Auto-persist research to prevent loss during context compaction

v4.3 ENHANCEMENT (Dec 2025):
- Auto-persist instead of just reminders
- Never blocks, always auto-creates docs
- Tracks research per upgrade/category
- Appends to existing docs
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path


# Track research activity within session
RESEARCH_TRACKER_FILE = Path("/tmp/claude_research_tracker.json")

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Auto-persist threshold (persist after N web operations)
AUTO_PERSIST_THRESHOLD = 3


def load_tracker() -> dict:
    """Load research tracker state."""
    if RESEARCH_TRACKER_FILE.exists():
        try:
            return json.loads(RESEARCH_TRACKER_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "web_operations": 0,
        "last_persist": None,
        "session_start": datetime.now().isoformat(),
        "searches": [],
        "urls": [],
        "current_upgrade": None,
        "current_topic": None,
    }


def save_tracker(tracker: dict) -> None:
    """Save research tracker state."""
    try:
        RESEARCH_TRACKER_FILE.write_text(json.dumps(tracker, indent=2))
    except OSError:
        pass


def detect_upgrade_context() -> tuple[str, str]:
    """Detect current upgrade and topic from RIC state or progress file."""
    # Try RIC state first
    ric_state_file = PROJECT_ROOT / ".claude" / "state" / "ric.json"
    if ric_state_file.exists():
        try:
            state = json.loads(ric_state_file.read_text())
            if state.get("upgrade_id"):
                topic = state.get("topic", "RESEARCH")
                return state["upgrade_id"], topic
        except (OSError, json.JSONDecodeError):
            pass

    # Try progress file
    progress_file = PROJECT_ROOT / "claude-progress.txt"
    if progress_file.exists():
        try:
            content = progress_file.read_text()
            upgrade_match = re.search(r"UPGRADE-(\d{3})", content)
            if upgrade_match:
                return f"UPGRADE-{upgrade_match.group(1)}", "RESEARCH"
        except OSError:
            pass

    return "UPGRADE-UNKNOWN", "RESEARCH"


def get_or_create_research_doc(upgrade: str, topic: str) -> Path:
    """Get existing research doc or create new one. Never fails."""
    research_dir = PROJECT_ROOT / "docs" / "research"
    research_dir.mkdir(parents=True, exist_ok=True)

    # Look for existing doc for this upgrade
    pattern = f"{upgrade}*RESEARCH.md"
    existing_docs = list(research_dir.glob(pattern))

    if existing_docs:
        # Use most recently modified
        return max(existing_docs, key=lambda p: p.stat().st_mtime)

    # Create new doc
    safe_topic = re.sub(r"[^A-Z0-9-]", "-", topic.upper())
    doc_path = research_dir / f"{upgrade}-{safe_topic}-RESEARCH.md"

    if not doc_path.exists():
        template = f"""# {upgrade} {topic} Research

**Status**: AUTO-CREATED - Research in progress
**Created**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Mode**: Auto-persist enabled

## Research Overview

This document is automatically maintained during the research process.
Findings are auto-persisted to prevent loss during context compaction.

## Research Log

<!-- Auto-persisted research entries will be added below -->

"""
        try:
            doc_path.write_text(template)
        except OSError:
            pass

    return doc_path


def format_research_entry(tracker: dict) -> str:
    """Format research entry for appending to doc."""
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    searches = tracker.get("searches", [])
    urls = tracker.get("urls", [])

    entry = f"""
---

### Research Entry - {timestamp}

**Search Queries**:
"""
    for query in searches:
        entry += f'- "{query}"\n'

    if urls:
        entry += "\n**URLs Discovered**:\n"
        for url in urls:
            entry += f"- {url}\n"

    entry += f"""
**Search Date**: {timestamp}
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.

"""
    return entry


def auto_persist_research(tracker: dict) -> tuple[bool, str]:
    """
    Auto-persist research findings. NEVER blocks.

    Returns:
        (persisted, message)
    """
    if not tracker.get("searches"):
        return False, ""

    upgrade, topic = detect_upgrade_context()
    tracker["current_upgrade"] = upgrade
    tracker["current_topic"] = topic

    try:
        doc_path = get_or_create_research_doc(upgrade, topic)
        entry = format_research_entry(tracker)

        # Append to doc
        with open(doc_path, "a") as f:
            f.write(entry)

        # Reset tracker after persist
        search_count = len(tracker.get("searches", []))
        tracker["searches"] = []
        tracker["urls"] = []
        tracker["web_operations"] = 0
        tracker["last_persist"] = datetime.now().isoformat()
        save_tracker(tracker)

        message = f"""
📝 **AUTO-PERSISTED RESEARCH** ({search_count} searches saved)

**Saved to**: `{doc_path.name}`
**Timestamp**: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}

**TIP**: Review the auto-persisted entry and add:
- Publication dates to sources: `(Published: Month Year)`
- Key discoveries and implementation notes
"""
        return True, message

    except OSError as e:
        # Never fail - just log
        return False, f"⚠️ Auto-persist warning (will retry): {e}"


def main():
    """Main hook entry point."""
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    # Load tracker
    tracker = load_tracker()

    # Track WebSearch operations
    if tool_name == "WebSearch":
        tracker["web_operations"] = tracker.get("web_operations", 0) + 1

        query = tool_input.get("query", "")
        if query:
            if "searches" not in tracker:
                tracker["searches"] = []
            if query not in tracker["searches"]:
                tracker["searches"].append(query)

        save_tracker(tracker)

        # Check if we should auto-persist
        if tracker["web_operations"] >= AUTO_PERSIST_THRESHOLD:
            persisted, message = auto_persist_research(tracker)
            if persisted:
                print(message)
            else:
                # Show progress toward auto-persist
                print(f"📊 Research tracked ({tracker['web_operations']}/{AUTO_PERSIST_THRESHOLD} before auto-persist)")

    # Track WebFetch operations
    elif tool_name == "WebFetch":
        tracker["web_operations"] = tracker.get("web_operations", 0) + 1

        url = tool_input.get("url", "")
        if url:
            if "urls" not in tracker:
                tracker["urls"] = []
            if url not in tracker["urls"]:
                tracker["urls"].append(url)

        save_tracker(tracker)

        # Check if we should auto-persist
        if tracker["web_operations"] >= AUTO_PERSIST_THRESHOLD:
            persisted, message = auto_persist_research(tracker)
            if persisted:
                print(message)

    # Track documentation writes (reset counter when user writes to research)
    elif tool_name in ["Write", "Edit"]:
        file_path = tool_input.get("file_path", "")
        is_research_doc = (
            "docs/research/" in file_path or file_path.startswith("docs/research/") or "/docs/research/" in file_path
        ) and file_path.endswith(".md")

        if is_research_doc:
            # User manually wrote to research - reset auto-persist tracker
            tracker["last_persist"] = datetime.now().isoformat()
            tracker["web_operations"] = 0
            tracker["searches"] = []
            tracker["urls"] = []
            save_tracker(tracker)

            filename = Path(file_path).name
            print(f"✅ Research documented: {filename}")

    sys.exit(0)


if __name__ == "__main__":
    main()
