#!/usr/bin/env python3
"""
Trading Context Loader SessionStart Hook

Loads trading context and state at session start.
Provides market status, portfolio summary, risk alerts, and RIC Loop state.

UPGRADE-015 Phase 4: Hook System Implementation
UPGRADE-016: Added RIC Loop state loading
UPGRADE-017: Added upgrade doc reading at session start (v4.3 Doc Enforcement)

Usage:
    Called as SessionStart hook when new session begins.
    Prints trading context summary to stderr.

Context Loaded:
    - Market status (open/closed)
    - Portfolio summary
    - Active orders
    - Risk alerts
    - Daily P&L
    - RIC Loop state (iteration, phase, insights)
    - Upgrade documentation context (v4.3)
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path


# State files
STATE_DIR = Path("/home/dshooter/projects/Claude_code_Quantconnect_trading_bot")
PROGRESS_FILE = STATE_DIR / "claude-progress.txt"
RIC_PROGRESS_FILE = STATE_DIR / "ric-progress.md"
RIC_STATE_FILE = STATE_DIR / ".claude" / "state" / "ric.json"
SESSION_NOTES = STATE_DIR / "claude-session-notes.md"
TRADE_LOG = STATE_DIR / "logs" / "trade_log.jsonl"
RESEARCH_DIR = STATE_DIR / "docs" / "research"


def get_market_status() -> dict:
    """Get mock market status."""
    now = datetime.utcnow()
    hour = now.hour

    # Simple market hours check (UTC)
    is_market_hours = 14 <= hour < 21  # 9:30 AM - 4:00 PM ET in UTC

    return {
        "status": "open" if is_market_hours else "closed",
        "time": now.strftime("%H:%M:%S UTC"),
        "date": now.strftime("%Y-%m-%d"),
    }


def get_recent_trades() -> list:
    """Get recent trades from log file."""
    if not TRADE_LOG.exists():
        return []

    trades = []
    try:
        with open(TRADE_LOG) as f:
            lines = f.readlines()[-10:]  # Last 10 entries
            for line in lines:
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass

    return trades


def get_current_task() -> str:
    """Get current task from progress file."""
    if not PROGRESS_FILE.exists():
        return "No progress file found"

    try:
        with open(PROGRESS_FILE) as f:
            content = f.read()

        # Find first uncompleted task
        for line in content.split("\n"):
            if "- [ ]" in line:
                return line.strip().replace("- [ ]", "").strip()

        return "All tasks complete"
    except OSError:
        return "Could not read progress file"


def check_risk_alerts() -> list:
    """Check for any risk alerts."""
    alerts = []

    # Check if risk state file exists
    risk_state = Path("/tmp/trading_risk_state.json")
    if risk_state.exists():
        try:
            with open(risk_state) as f:
                state = json.load(f)

            # Check order count
            if state.get("order_count", 0) > 80:
                alerts.append(f"WARNING: {state['order_count']}/100 daily orders used")

            # Check volume
            if state.get("total_volume", 0) > 400000:
                alerts.append(f"WARNING: ${state['total_volume']:,.0f}/$500k daily volume used")

        except (json.JSONDecodeError, OSError):
            pass

    return alerts


def get_ric_state() -> dict:
    """Get RIC Loop state if active."""
    ric_state = {
        "active": False,
        "upgrade_id": "",
        "iteration": 0,
        "max_iterations": 5,
        "phase": 0,
        "phase_name": "",
        "open_insights": {"P0": 0, "P1": 0, "P2": 0},
        "can_exit": False,
        "exit_reason": "",
    }

    if not RIC_STATE_FILE.exists():
        return ric_state

    try:
        with open(RIC_STATE_FILE) as f:
            state = json.load(f)

        ric_state["active"] = bool(state.get("upgrade_id"))
        ric_state["upgrade_id"] = state.get("upgrade_id", "")
        ric_state["iteration"] = state.get("current_iteration", 1)
        ric_state["max_iterations"] = state.get("max_iterations", 5)
        ric_state["phase"] = state.get("current_phase", 0)

        # Phase names mapping
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
        ric_state["phase_name"] = phase_names.get(ric_state["phase"], "Unknown")

        # Count open insights
        insights = state.get("insights", [])
        for insight in insights:
            if isinstance(insight, dict):
                if insight.get("status") != "resolved":
                    priority = insight.get("priority", "P1")
                    if priority in ric_state["open_insights"]:
                        ric_state["open_insights"][priority] += 1

        # Check exit eligibility
        min_iterations = state.get("min_iterations", 3)
        p0_open = ric_state["open_insights"]["P0"]
        p1_open = ric_state["open_insights"]["P1"]
        p2_open = ric_state["open_insights"]["P2"]

        if ric_state["iteration"] < min_iterations:
            ric_state["can_exit"] = False
            ric_state["exit_reason"] = f"Min {min_iterations} iterations required"
        elif p0_open > 0:
            ric_state["can_exit"] = False
            ric_state["exit_reason"] = f"{p0_open} P0 insights open"
        elif p1_open > 0:
            ric_state["can_exit"] = False
            ric_state["exit_reason"] = f"{p1_open} P1 insights open"
        elif p2_open > 0:
            ric_state["can_exit"] = False
            ric_state["exit_reason"] = f"{p2_open} P2 insights open"
        else:
            ric_state["can_exit"] = True
            ric_state["exit_reason"] = "All criteria met"

    except (json.JSONDecodeError, OSError, KeyError):
        pass

    return ric_state


def find_upgrade_docs(upgrade_id: str) -> dict:
    """
    Find all documentation related to an upgrade.

    Returns:
        Dict with 'main', 'research', 'categories' keys containing doc paths.
    """
    docs = {
        "main": None,  # UPGRADE-XXX.md
        "research": [],  # UPGRADE-XXX-*-RESEARCH.md files
        "categories": [],  # UPGRADE-XXX-CATN-* files
        "summary": None,  # UPGRADE-XXX-SUMMARY.md
    }

    if not RESEARCH_DIR.exists():
        return docs

    # Pattern matching for different doc types
    upgrade_num = upgrade_id.replace("UPGRADE-", "")

    for doc_file in RESEARCH_DIR.glob(f"UPGRADE-{upgrade_num}*.md"):
        name = doc_file.name

        if name == f"UPGRADE-{upgrade_num}.md":
            docs["main"] = doc_file
        elif name == f"UPGRADE-{upgrade_num}-SUMMARY.md":
            docs["summary"] = doc_file
        elif "CAT" in name:
            docs["categories"].append(doc_file)
        elif name.endswith("-RESEARCH.md"):
            docs["research"].append(doc_file)

    return docs


def get_doc_summary(doc_path: Path, max_lines: int = 30) -> str | None:
    """
    Read and summarize a documentation file.

    Extracts key sections: Overview, Current Status, Key Discoveries.
    """
    if not doc_path or not doc_path.exists():
        return None

    try:
        content = doc_path.read_text()
        lines = content.split("\n")

        # Extract key sections
        summary_lines = []
        in_relevant_section = False
        section_count = 0

        for line in lines[: max_lines * 2]:  # Check more lines to find sections
            # Check for relevant section headers
            if re.match(
                r"^#{1,3}\s*(Overview|Current Status|Key Discoveries|Implementation Status)", line, re.IGNORECASE
            ):
                in_relevant_section = True
                summary_lines.append(line)
                section_count += 1
            elif re.match(r"^#{1,3}\s", line) and in_relevant_section:
                in_relevant_section = False
            elif in_relevant_section:
                summary_lines.append(line)
                if len(summary_lines) >= max_lines:
                    break

        if summary_lines:
            return "\n".join(summary_lines[:max_lines])
        else:
            # Just return the first N lines if no sections found
            return "\n".join(lines[: min(10, max_lines)])

    except OSError:
        return None


def check_doc_staleness(doc_path: Path, max_hours: int = 4) -> tuple[bool, str]:
    """
    Check if a document is stale (not updated recently).

    Returns:
        (is_stale, message)
    """
    if not doc_path or not doc_path.exists():
        return True, "Document does not exist"

    try:
        mtime = doc_path.stat().st_mtime
        age_hours = (datetime.now().timestamp() - mtime) / 3600

        if age_hours > max_hours:
            return True, f"Last updated {age_hours:.1f} hours ago (> {max_hours}h threshold)"
        return False, f"Updated {age_hours:.1f} hours ago"

    except OSError:
        return True, "Could not check modification time"


def get_upgrade_doc_context(upgrade_id: str) -> dict:
    """
    Get complete upgrade documentation context for session start.

    Returns:
        Dict with doc status, summaries, staleness warnings, and action items.
    """
    context = {
        "has_docs": False,
        "main_doc": None,
        "research_docs": [],
        "category_docs": [],
        "summary_doc": None,
        "warnings": [],
        "summaries": {},
    }

    if not upgrade_id:
        return context

    docs = find_upgrade_docs(upgrade_id)

    # Check main upgrade doc
    if docs["main"]:
        context["main_doc"] = str(docs["main"])
        context["has_docs"] = True
        summary = get_doc_summary(docs["main"])
        if summary:
            context["summaries"]["main"] = summary
        is_stale, msg = check_doc_staleness(docs["main"])
        if is_stale:
            context["warnings"].append(f"Main doc stale: {msg}")
    else:
        context["warnings"].append(f"⚠️ No main upgrade doc found: {upgrade_id}.md")

    # Check research docs
    for research_doc in docs["research"]:
        context["research_docs"].append(str(research_doc))
        context["has_docs"] = True

    # Check category docs
    for cat_doc in docs["categories"]:
        context["category_docs"].append(str(cat_doc))
        context["has_docs"] = True
        # Check category doc staleness
        is_stale, msg = check_doc_staleness(cat_doc, max_hours=8)
        if is_stale:
            context["warnings"].append(f"Category doc {cat_doc.name}: {msg}")

    # Check summary doc
    if docs["summary"]:
        context["summary_doc"] = str(docs["summary"])
        summary = get_doc_summary(docs["summary"])
        if summary:
            context["summaries"]["summary"] = summary

    return context


def format_context_summary() -> str:
    """Format the trading context summary."""
    market = get_market_status()
    recent_trades = get_recent_trades()
    current_task = get_current_task()
    risk_alerts = check_risk_alerts()
    ric_state = get_ric_state()

    lines = [
        "=" * 60,
        "TRADING SESSION CONTEXT",
        "=" * 60,
        f"Time: {market['date']} {market['time']}",
        f"Market Status: {market['status'].upper()}",
        "",
    ]

    # RIC Loop status (if active)
    if ric_state["active"]:
        lines.append("-" * 60)
        lines.append("RIC LOOP ACTIVE")
        lines.append("-" * 60)
        lines.append(f"Upgrade: {ric_state['upgrade_id']}")
        lines.append(f"Iteration: {ric_state['iteration']}/{ric_state['max_iterations']}")
        lines.append(f"Phase: {ric_state['phase']} - {ric_state['phase_name']}")
        lines.append(
            f"Open Insights: P0={ric_state['open_insights']['P0']} P1={ric_state['open_insights']['P1']} P2={ric_state['open_insights']['P2']}"
        )
        lines.append(f"Exit Status: {'ALLOWED' if ric_state['can_exit'] else 'BLOCKED'} ({ric_state['exit_reason']})")
        lines.append("")
        lines.append(
            f">>> RESUME AT: [ITERATION {ric_state['iteration']}/{ric_state['max_iterations']}] === PHASE {ric_state['phase']}: {ric_state['phase_name'].upper()} ==="
        )
        lines.append("")

        # v4.3 Doc Enforcement: Load upgrade documentation context
        doc_context = get_upgrade_doc_context(ric_state["upgrade_id"])

        if doc_context["has_docs"]:
            lines.append("-" * 60)
            lines.append("UPGRADE DOCUMENTATION")
            lines.append("-" * 60)

            if doc_context["main_doc"]:
                lines.append(f"Main Doc: {doc_context['main_doc']}")
            if doc_context["research_docs"]:
                lines.append(f"Research Docs: {len(doc_context['research_docs'])} files")
            if doc_context["category_docs"]:
                lines.append(f"Category Docs: {len(doc_context['category_docs'])} files")
                for cat_doc in doc_context["category_docs"][:5]:  # Show first 5
                    lines.append(f"  - {Path(cat_doc).name}")

            # Show doc warnings (stale, missing, etc.)
            if doc_context["warnings"]:
                lines.append("")
                lines.append("DOC WARNINGS:")
                for warning in doc_context["warnings"][:3]:  # Show first 3 warnings
                    lines.append(f"  ⚠️ {warning}")

            # Show summary if available
            if "main" in doc_context["summaries"]:
                lines.append("")
                lines.append(">>> CONTEXT SUMMARY (from upgrade doc):")
                for line in doc_context["summaries"]["main"].split("\n")[:10]:  # First 10 lines
                    lines.append(f"  {line}")

            lines.append("")
            lines.append(">>> ACTION: Read upgrade docs before resuming work:")
            if doc_context["main_doc"]:
                lines.append(f"   Read: {doc_context['main_doc']}")
            lines.append("")

        elif ric_state["active"]:
            # No docs found but RIC is active - CRITICAL warning
            lines.append("-" * 60)
            lines.append("⚠️ NO UPGRADE DOCUMENTATION FOUND")
            lines.append("-" * 60)
            lines.append(f"Expected: docs/research/{ric_state['upgrade_id']}-*-RESEARCH.md")
            lines.append("")
            lines.append(">>> ACTION REQUIRED: Create upgrade documentation before continuing!")
            lines.append(f"   Create: docs/research/{ric_state['upgrade_id']}-TOPIC-RESEARCH.md")
            lines.append("")

    else:
        lines.append("RIC Loop: Not active (use /ric-start for complex tasks)")
        lines.append("")

    lines.append(f"Current Task: {current_task}")
    lines.append("")

    if recent_trades:
        lines.append(f"Recent Activity: {len(recent_trades)} logged events")
        last_trade = recent_trades[-1]
        lines.append(f"  Last: {last_trade.get('tool_name', 'unknown')} at {last_trade.get('timestamp', 'unknown')}")
    else:
        lines.append("Recent Activity: No trading activity logged")

    if risk_alerts:
        lines.append("")
        lines.append("RISK ALERTS:")
        for alert in risk_alerts:
            lines.append(f"  {alert}")

    lines.append("")
    lines.append("MCP Servers: market-data, broker, portfolio, backtest")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    """Main entry point for hook."""
    try:
        summary = format_context_summary()
        print(summary, file=sys.stderr)
    except Exception as e:
        print(f"Trading context load error: {e}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
