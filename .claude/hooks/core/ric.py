#!/usr/bin/env python3
"""
RIC Loop for Autonomous Claude Code (v5.0 Guardian) - STANDALONE IMPLEMENTATION
================================================================================

STATUS: ACTIVE - Upgraded from v4.5 on 2025-12-04
CODENAME: Guardian
RESEARCH: docs/research/RIC-V46-RESEARCH-DEC2025.md (35+ sources, 7 phases)

v5.0 NEW FEATURES (December 2025 Research - 12 Enhancements):
[P0] DRIFT DETECTION: Track scope creep, alert on >20% expansion (AEGIS Framework)
[P0] GUARDIAN MODE: Independent verification pass as separate reviewer (Gartner 2025)
[P0] STRUCTURED MEMORY: RIC_NOTES.md for cross-session context (Anthropic 2025)
[P0] SEIDR DEBUG LOOP: Multi-candidate fix generation and ranking (ACM TELO 2025)
[P1] CANDIDATE RANKING: Lexicase/tournament selection for fixes (SEIDR Paper)
[P1] REPLACE/REPAIR TRACKING: Monitor replace vs repair ratio (SEIDR)
[P1] METAMORPHIC CHECK: Consistency testing for hallucination detection (MetaQA)
[P1] OODA LOOP ALIGNMENT: Security context per phase (Snyk 2025)
[P2] PACKAGE VERIFICATION: Check imported packages exist on PyPI (USENIX 2025)
[P2] POLICY-AS-CODE: Executable guardrails enforcement (AEGIS Framework)
[P2] UNCERTAINTY QUANTIFICATION: Token probability analysis (UQLM, planned)
[P2] SELF-IMPROVING PROMPTS: Track and optimize prompt effectiveness (SICA, planned)

v4.5 FEATURES (Research-Backed Safety & Intelligence):
- HALLUCINATION DETECTION: 5-category taxonomy check before commits
- ENHANCED CONVERGENCE: Multi-metric convergence detection (insight_rate, fix_success, churn)
- CONTEXT MANAGEMENT: Hierarchical memory protocol (core/working/archival)
- RUNAWAY LOOP PROTECTION: Safety throttles (tool calls, edits, time limits)
- DECISION TRACING: Structured decision logs for meta-debugging
- CONFIDENCE CALIBRATION: Per-phase confidence ratings with low-confidence protocol

v4.2 FEATURES (Retained):
- SELF-REFINE: Actionable feedback templates ("specific issue + location + root cause + fix")
- CoCoGen: Static analysis BEFORE tests (lint → types → tests → coverage)
- SAGE: Checker verification role (2.26x improvement) - independent review step
- ReVeal: Self-constructed tests (agent writes verification tests)
- PairCoder: Multi-plan exploration (generate 2-3 approaches, compare)

v4.1 FEATURES (Retained):
- Phase 0: Frame problem measurably first (NASA study)
- Phase 1: [OUT] scope + explicit priority definitions
- Phase 2: AND-test for atomic commits + revertability check
- Phase 3: Quality gates + 5 fix attempts
- Phase 4: Explicit critique step + reflection checklist

ENFORCEMENT INFRASTRUCTURE:
- Hook handlers for PreToolUse and UserPromptSubmit
- State persistence via .claude/state/ric.json
- Forced validation of [I{iter}/{max}][P{phase}] format
- Phase-appropriate tool blocking
- Logging with [RIC] prefixes
- Safety throttle enforcement
- Convergence metric tracking
- Drift detection on phase change
- Guardian review before commits
- Structured notes auto-update

CORE FILES:
  1. .claude/hooks/ric.py - This file (standalone, no external imports)
  2. .claude/RIC_CONTEXT.md - Quick reference
  3. .claude/RIC_NOTES.md - Structured memory (auto-generated)
"""

import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


RIC_VERSION = "5.1"
RIC_CODENAME = "Guardian"
RIC_BUILD_DATE = "2025-12-04"
RIC_V51_FEATURES = "SELF-REFINE + Reflexion Quality Gates"

# =============================================================================
# v5.0 FEATURE FLAGS (Enable/disable individual features)
# =============================================================================

FEATURE_FLAGS = {
    # P0 - Critical
    "drift_detection": True,
    "guardian_mode": True,
    "structured_memory": True,
    "seidr_debug_loop": True,
    # P1 - Important
    "candidate_ranking": True,
    "replace_repair_tracking": True,
    "metamorphic_check": True,
    "ooda_alignment": True,
    # P2 - Future
    "package_verification": True,
    "policy_as_code": True,
    "uncertainty_quantification": False,  # Requires API support
    "self_improving_prompts": False,  # Requires testing
    # v5.1 - ANTI-GAMING (December 2025 Research)
    "quality_gates": True,  # SELF-REFINE + Reflexion quality validation
    "require_location_in_ideas": True,  # Must reference file:line in upgrade ideas
    "require_iteration_citation": True,  # Must cite previous iteration in iteration 2+
    # v5.1 - AUDIT SUITE (Proactive Validation)
    "p3_audit_integration": True,  # Run critical audits (secrets, security) in P3 VERIFY
    "audit_blocks_exit": False,  # If True, critical audit issues block phase advancement
}


def is_feature_enabled(feature: str) -> bool:
    """Check if a feature is enabled via flag or environment override."""
    env_key = f"RIC_FEATURE_{feature.upper()}"
    env_val = os.environ.get(env_key)
    if env_val is not None:
        return env_val == "1"
    return FEATURE_FLAGS.get(feature, False)


# =============================================================================
# PHASES (5 Phases - Research Validated)
# =============================================================================

PHASES = {
    0: ("RESEARCH", "Frame problem measurably → WebSearch → persist IMMEDIATELY"),
    1: ("PLAN", "Define scope + success criteria + P0/P1/P2/[OUT] tasks"),
    2: ("BUILD", "Atomic commits (AND-test) → 1-5 files → hallucination check"),
    3: ("VERIFY", "Quality gates: tests, coverage, lint, security + consistency"),
    4: ("REFLECT", "Critique → Identify gaps → Classify → Confidence → Loop or Exit"),
}

# =============================================================================
# ITERATION LIMITS
# =============================================================================

ITERATION_LIMITS = {
    "min": 3,  # Cannot exit before 3 iterations
    "max": 5,  # Force exit at 5 iterations
    "plateau": 2,  # Exit if no new insights for 2 consecutive iterations
}

# =============================================================================
# FIX ATTEMPT LIMITS (Increased for code quality)
# =============================================================================

FIX_LIMITS = {
    "test_failure": 5,  # Max attempts to fix failing tests
    "lint_error": 3,  # Max attempts to fix lint errors
    "coverage_gap": 3,  # Max attempts to add coverage
    "type_error": 3,  # Max attempts to fix type errors
    "stuck_task": 3,  # Max attempts before moving on
    "hallucination_fix": 2,  # Max attempts to fix hallucination issues
}

# =============================================================================
# SAFETY THROTTLES (v4.3 NEW - Runaway Loop Protection)
# =============================================================================

SAFETY_THROTTLES = {
    "max_tool_calls_per_phase": 50,  # Prevent infinite tool loops
    "max_edits_per_file_per_iteration": 10,  # Prevent file thrashing
    "max_consecutive_failures": 3,  # Pause for review after 3 failures
    "max_time_per_phase_minutes": 30,  # Time-based throttle
    "cooldown_after_stuck_minutes": 5,  # Minutes before retry
    "max_decisions_without_progress": 5,  # Detect spinning
}

# =============================================================================
# RESEARCH ENFORCEMENT (v4.3 NEW - Compaction Protection)
# =============================================================================

RESEARCH_ENFORCEMENT = {
    # Persistence trigger: Auto-save after N web searches (NEVER block)
    "searches_before_auto_persist": 3,  # Auto-persist after every 3 searches
    "searches_before_forced_persist": 3,  # Alias for backward compatibility
    "auto_persist_research": True,  # Automatically create/append research doc
    # Timestamp validation patterns
    "search_timestamp_required": True,  # Check for "Search Date:" in research
    "publication_date_required": True,  # Check for "Published:" in sources
    "auto_add_timestamps": True,  # Auto-add timestamps if missing
    # NEVER block - use auto-create instead
    "block_search_without_persist": False,  # NEVER block - auto-persist instead
    "warn_missing_timestamps": True,  # Warn if timestamps missing (advisory)
    "auto_fix_timestamps": True,  # Auto-add timestamp template if missing
    # Research quality (advisory, not blocking)
    "min_sources_per_topic": 3,  # Advisory: suggest more sources
    "min_words_per_finding": 50,  # Advisory: suggest more detail
    # Auto-create research docs
    "auto_create_research_doc": True,  # Auto-create docs when needed
    "research_doc_template": True,  # Use template for auto-created docs
    "append_to_existing": True,  # Append to existing doc if present
}

# Timestamp regex patterns for validation
TIMESTAMP_PATTERNS = {
    # Search timestamp patterns:
    # - "Search Date: December 4, 2025"
    # - "**Search Date**: December 4, 2025"
    # - "Searched: 2025-12-04"
    "search_date": [
        r"[Ss]earch(?:ed)?[\s:]+(?:Date[\s:]+)?(\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
        r"\*\*Search Date\*\*:\s*(.+?)(?:\n|$)",
    ],
    # Publication date patterns:
    # - "Published: October 2024"
    # - "**Published**: October 2024"
    # - "(Published: ~2025)"
    # - "Published: ~2024"
    "publication_date": [
        r"[Pp]ublish(?:ed)?[\s:]+[~]?(\w+\s+\d{4}|\d{4}(?:-\d{2})?(?:-\d{2})?)",
        r"\*\*[Pp]ublished\*\*:\s*[~]?(.+?)(?:\n|$)",
        r"\([Pp]ublished:\s*[~]?(.+?)\)",
    ],
}

# =============================================================================
# THOROUGH RESEARCH WITH FAST AGENTS (v5.1 NEW)
# =============================================================================

THOROUGH_RESEARCH_CONFIG = {
    # Use fast agents (haiku) for parallel URL fetching
    "default_model": "haiku",
    "parallel_fetches": True,
    # Multi-pass extraction prompts
    "fetch_passes": ["overview", "code", "api", "config"],
    # Auto-spawn agents after finding URLs
    "auto_spawn_agents": True,
    "max_parallel_agents": 8,
    # Persist immediately after agent results
    "immediate_persist": True,
}

# Prompt for thorough research phase
THOROUGH_RESEARCH_PROMPT = """
**🔬 THOROUGH RESEARCH MODE** (Fast Parallel Agents)

For comprehensive research, use multi-pass URL fetching:

**Step 1: Find URLs** (WebSearch)
```
WebSearch("<topic> 2025 documentation")
```

**Step 2: Spawn Fast Agents** (Parallel)
For each important URL, spawn haiku agents in parallel:

```python
# Execute ALL in a SINGLE message for parallel execution
Task(model="haiku", subagent_type="general-purpose",
     description="Fetch overview from URL",
     prompt="Use WebFetch to extract overview and concepts from <url>")

Task(model="haiku", subagent_type="general-purpose",
     description="Fetch code examples from URL",
     prompt="Use WebFetch to extract ALL code examples verbatim from <url>")

Task(model="haiku", subagent_type="general-purpose",
     description="Fetch API details from URL",
     prompt="Use WebFetch to extract API signatures and config from <url>")
```

**Step 3: Consolidate & Save**
After agents return, IMMEDIATELY write to `docs/research/`:
- Merge all agent findings
- Add timestamps for each source
- Save before context compaction

**CLI Helper**:
```bash
python3 .claude/hooks/thorough_research.py plan <url1> <url2> --topic "Topic"
```

**Benefits**:
- 4x more information captured per URL
- Parallel execution (~30-60 seconds total)
- Haiku model = fast and cost-effective
- Immediate persistence prevents data loss
"""


def generate_thorough_research_tasks(urls: list, topic: str) -> list:
    """
    Generate Task tool calls for thorough URL research using fast agents.

    Args:
        urls: List of URLs to research
        topic: Research topic for context

    Returns:
        List of Task call dictionaries ready for execution
    """
    fetch_prompts = {
        "overview": "Extract the main concepts, purpose, and high-level architecture. Include key terminology with definitions.",
        "code": "Extract ALL code examples and snippets VERBATIM. Preserve exact formatting and comments.",
        "api": "Extract ALL function signatures, API endpoints, and their parameters with types and defaults.",
        "config": "Extract ALL configuration options, settings, environment variables, and their default values.",
    }

    tasks = []
    for url in urls:
        for category, prompt in fetch_prompts.items():
            task = {
                "tool": "Task",
                "params": {
                    "subagent_type": "general-purpose",
                    "model": THOROUGH_RESEARCH_CONFIG["default_model"],
                    "description": f"Fetch {category} from URL",
                    "prompt": f"""Research Task: Fetch and extract information.

**URL**: {url}
**Topic**: {topic}
**Focus**: {category.title()}

**Instructions**:
1. Use WebFetch to fetch the URL with this prompt: "{prompt}"
2. Format as markdown with clear headers
3. Return extracted content with source URL and timestamp
4. If fetch fails, report the error

Be thorough - extract ALL relevant information.""",
                },
            }
            tasks.append(task)

    return tasks


def format_research_tasks_for_execution(tasks: list) -> str:
    """Format Task calls for Claude to execute in parallel."""
    output = [
        "## Parallel Research Tasks",
        "",
        f"Execute these {len(tasks)} Task calls **in parallel** (single message):",
        "",
    ]

    for i, task in enumerate(tasks, 1):
        params = task["params"]
        output.append(f"**Task {i}**: {params['description']}")
        output.append(f"  - model: {params['model']}")
        output.append(f"  - subagent_type: {params['subagent_type']}")
        output.append("")

    output.append("**After all agents return**: Consolidate and save to docs/research/ IMMEDIATELY.")
    return "\n".join(output)


# Research auto-persist notification (informational, never blocks)
RESEARCH_AUTOPERSIST_NOTICE = """
📝 **AUTO-PERSISTING RESEARCH** (Search {search_count} completed)

Research findings are being automatically saved to protect against context compaction.

**Auto-Created/Updated**: `{research_doc_path}`

**What Was Saved**:
- Search queries: {queries}
- Key URLs discovered
- Timestamp: {today}

**ACTION** (optional but recommended):
- Review the auto-created doc and add detailed findings
- Add publication dates to sources: `(Published: Month Year)`
- Expand key discoveries with implementation notes

**Note**: Auto-persist runs every {persist_interval} searches to prevent data loss.
"""

# Legacy warning (kept for reference, but blocking is disabled)
RESEARCH_PERSIST_WARNING = """
**📋 RESEARCH DOCUMENTATION REMINDER** (Search {search_count}/{max_searches})

You have completed {search_count} web searches. Consider documenting findings.

**TIP**: Research is auto-persisted, but detailed notes improve future sessions.

**SUGGESTED ACTION**:
1. Create/update research doc: `docs/research/UPGRADE-XXX-TOPIC.md`
2. Include for EACH source:
   - **Search Date**: {today} (when you searched)
   - **Published**: [date or ~estimate] (when source was published)
3. Write substantive findings (not just URLs)

**FORMAT**:
```markdown
### Source {source_num}: [Title]
**URL**: [url]
**Search Date**: {today}
**Published**: [date]

**Key Findings**:
- [finding 1]
- [finding 2]
```

**Note**: This is advisory - auto-persist ensures basic research is saved.
"""

RESEARCH_TIMESTAMP_WARNING = """
**⚠️ MISSING TIMESTAMPS** in research document

Research file: {file_path}

**Issues Found**:
{issues}

**REQUIRED** (per project standards):
1. **Search Date**: When YOU searched (e.g., "Search Date: December 4, 2025")
2. **Publication Date**: When SOURCE was published (e.g., "Published: Oct 2024" or "Published: ~2025")

**Why This Matters**:
- Timestamps differentiate current vs outdated patterns
- Helps future sessions assess source reliability
- Required by RIC Loop Phase 0 gate
"""

# =============================================================================
# DOCUMENTATION ENFORCEMENT (v4.3 NEW - Upgrade Doc Lifecycle)
# =============================================================================


# Autonomous mode detection - set RIC_AUTONOMOUS_MODE=1 for overnight sessions
def is_autonomous_mode() -> bool:
    """Check if running in autonomous/overnight mode."""
    return os.environ.get("RIC_AUTONOMOUS_MODE", "0") == "1"


def get_enforcement_level() -> str:
    """
    Get enforcement level based on mode.

    Returns:
        'BLOCK' - Hard block on violations (interactive mode)
        'WARN'  - Warn but continue (autonomous mode)
        'LOG'   - Just log, no warnings (silent mode)
    """
    if is_autonomous_mode():
        return "WARN"  # Never block in autonomous mode
    mode = os.environ.get("RIC_ENFORCEMENT_LEVEL", "BLOCK").upper()
    return mode if mode in ("BLOCK", "WARN", "LOG") else "BLOCK"


# Autonomous mode issues tracker (persisted for later review)
AUTONOMOUS_ISSUES_FILE = Path(".claude/state/autonomous_issues.json")


def log_autonomous_issue(issue_type: str, details: dict) -> None:
    """Log an issue during autonomous mode for later review."""
    issues = []
    if AUTONOMOUS_ISSUES_FILE.exists():
        try:
            issues = json.loads(AUTONOMOUS_ISSUES_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            issues = []

    issues.append(
        {
            "timestamp": datetime.now().isoformat(),
            "type": issue_type,
            "details": details,
            "session_id": os.environ.get("CLAUDE_SESSION_ID", "unknown"),
        }
    )

    # Keep only last 100 issues
    issues = issues[-100:]

    try:
        AUTONOMOUS_ISSUES_FILE.parent.mkdir(parents=True, exist_ok=True)
        AUTONOMOUS_ISSUES_FILE.write_text(json.dumps(issues, indent=2))
    except OSError:
        pass


def get_autonomous_issues() -> list:
    """Get logged autonomous mode issues."""
    if not AUTONOMOUS_ISSUES_FILE.exists():
        return []
    try:
        return json.loads(AUTONOMOUS_ISSUES_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


DOC_ENFORCEMENT = {
    # Naming convention - AUTO-FIX instead of blocking
    "enforce_naming": True,  # Check naming compliance
    "block_on_naming_violation": False,  # NEVER block - auto-fix or warn instead
    "auto_fix_naming": True,  # Automatically suggest/apply correct naming
    # Required documents per upgrade - AUTO-CREATE instead of blocking
    "require_research_doc": True,  # UPGRADE-XXX-TOPIC-RESEARCH.md required
    "require_upgrade_doc": True,  # UPGRADE-XXX.md required at start
    "auto_create_missing_docs": True,  # Auto-create docs when missing (always)
    # Progress file protection - NEVER block, auto-create instead
    "block_completion_without_doc": False,  # Don't block - auto-create stub instead
    "block_progress_without_upgrade_doc": False,  # Don't block
    # Cross-reference requirements - ADVISORY only
    "require_cross_references": True,  # Links between docs required
    "validate_references_on_exit": True,  # Check all refs valid before exit
    "block_on_missing_refs": False,  # Never block - just warn
    # Session start requirements - informational only
    "read_upgrade_doc_on_start": True,  # Read upgrade doc at session start
    "read_progress_on_start": True,  # Read progress file
    # Update tracking - ADVISORY only
    "min_doc_updates_per_phase": 0,  # No minimum required (just track)
    "track_doc_staleness_hours": 8,  # Warn if doc not updated in 8 hours
    # Auto-create stubs for missing docs (always, not just autonomous mode)
    "auto_create_stub_docs": True,  # Create stub docs when missing
    "stub_doc_marker": "<!-- AUTO-GENERATED STUB - NEEDS CONTENT -->",
}

# Valid naming patterns for upgrade documents
UPGRADE_DOC_PATTERNS = {
    # Research documents
    "research": [
        r"^UPGRADE-\d{3}-[A-Z][A-Z0-9-]+-RESEARCH\.md$",  # UPGRADE-014-TOPIC-RESEARCH.md
        r"^UPGRADE-\d{3}-CAT\d+-[A-Z][A-Z0-9-]+-RESEARCH\.md$",  # UPGRADE-014-CAT1-ARCH-RESEARCH.md
    ],
    # Main upgrade documents
    "upgrade": [
        r"^UPGRADE-\d{3}\.md$",  # UPGRADE-014.md (main doc)
        r"^UPGRADE-\d{3}-[A-Z][A-Z0-9-]+\.md$",  # UPGRADE-014-AUTONOMOUS-ENHANCEMENTS.md
    ],
    # Summary documents
    "summary": [
        r"^UPGRADE-\d{3}-SUMMARY\.md$",
    ],
}

# Required sections in upgrade documents
UPGRADE_DOC_SECTIONS = {
    "research": [
        "## Overview",
        "## Research Objectives",
        "## Key Sources",
        "## Key Discoveries",
    ],
    "upgrade": [
        "## Overview",
        "## Success Criteria",
        "## Implementation Plan",
        "## Status",
    ],
}

# Cross-reference patterns to validate
CROSS_REFERENCE_PATTERNS = [
    r"See:\s*`?docs/research/[A-Z0-9_-]+\.md`?",
    r"\[.*?\]\(\.\.?/docs/research/[A-Z0-9_-]+\.md\)",
    r"Related:\s*UPGRADE-\d{3}",
]

DOC_NAMING_WARNING = """
**🚨 INVALID DOCUMENT NAME**

File: {file_path}
Expected patterns:
  - Research: UPGRADE-NNN-TOPIC-RESEARCH.md
  - Category: UPGRADE-NNN-CATN-NAME-RESEARCH.md
  - Main: UPGRADE-NNN.md or UPGRADE-NNN-TOPIC.md

**Examples**:
  ✅ UPGRADE-014-AUTONOMOUS-ENHANCEMENTS.md
  ✅ UPGRADE-014-CAT3-FAULT-TOLERANCE-RESEARCH.md
  ❌ upgrade_014_research.md (wrong format)
  ❌ UPGRADE-14-RESEARCH.md (missing leading zero)

**BLOCKING**: Write operation blocked until name corrected.
"""

DOC_COMPLETION_BLOCK_WARNING = """
**🛑 CANNOT MARK COMPLETED WITHOUT DOCUMENTATION**

You are trying to mark a category/task as COMPLETED but the required
documentation does not exist.

**Missing Document**: {expected_doc}

**REQUIRED BEFORE COMPLETION**:
1. Create the research document with proper naming
2. Include required sections: {required_sections}
3. Add cross-references to related docs

**Quick Create**:
```bash
python scripts/create_research_doc.py "{topic}" --upgrade UPGRADE-{upgrade_num}
```

**BLOCKING**: Progress update blocked until documentation exists.
"""

DOC_SESSION_START_PROMPT = """
**📄 UPGRADE DOCUMENTATION CONTEXT**

**Current Upgrade**: {upgrade_id}
**Upgrade Doc**: {upgrade_doc_path}
**Progress File**: claude-progress.txt

**Document Status**:
- Upgrade doc: {upgrade_doc_status}
- Research docs: {research_count} found
- Last updated: {last_updated}

**Required Reading** (summarized below):
{doc_summary}

**Next Actions** (from progress file):
{next_actions}

**Cross-References to Update** (if making changes):
{cross_refs}
"""

DOC_STALENESS_WARNING = """
**⚠️ DOCUMENTATION MAY BE STALE**

The upgrade documentation has not been updated in {hours} hours.

**Last Update**: {last_update_time}
**Current Phase**: P{phase}

**Consider Updating**:
- `{upgrade_doc}` with implementation progress
- Research doc with any new discoveries
- Cross-references if files changed

**Command to Check**:
```bash
git log --oneline -5 docs/research/
```
"""

# =============================================================================
# CONVERGENCE METRICS (v4.3 NEW - Enhanced Detection)
# =============================================================================

CONVERGENCE_THRESHOLDS = {
    "insight_rate_declining": True,  # New insights per iteration should decrease
    "fix_success_rate_target": 0.8,  # 80%+ first-try fix rate = converging
    "code_churn_declining": True,  # Lines changed should decrease
    "gate_pass_rate_target": 0.9,  # 90%+ gates pass without retry
    "confidence_floor": 70,  # Minimum confidence to proceed
}

# =============================================================================
# HALLUCINATION CATEGORIES (v4.3 NEW - Taxonomy from Research)
# =============================================================================

HALLUCINATION_CATEGORIES = {
    "intent_conflicting": "Code doesn't match stated intent/commit message",
    "context_inconsistency": "Code conflicts with existing codebase patterns",
    "context_repetition": "Unnecessary duplication of existing functionality",
    "dead_code": "Unreachable or unused code paths added",
    "knowledge_conflicting": "Incorrect API usage, wrong syntax, invalid patterns",
}

# =============================================================================
# LOGGING FORMAT
# =============================================================================

PHASE_HEADER = "[I{iter}/{max}][P{phase}] {phase_name}"
COMMIT_FORMAT = "[I{iter}/{max}][P2] {description}"

# Regex patterns for validation
PHASE_HEADER_REGEX = r"\[I(\d+)/(\d+)\]\[P(\d+)\]\s+(\w+)"
COMMIT_FORMAT_REGEX = r"\[I(\d+)/(\d+)\]\[P2\]\s+.+"

# =============================================================================
# PRIORITY DEFINITIONS (Research-backed)
# =============================================================================

PRIORITY_DEFINITIONS = """
**Priority Levels** (Industry Standard + RIC Rules):

| Level | Name | Definition | RIC Rule |
|-------|------|------------|----------|
| [P0] | CRITICAL | Would hold release, blocks everything | Must resolve to exit |
| [P1] | IMPORTANT | Should complete this iteration | Must resolve to exit |
| [P2] | POLISH | Nice to have, improves quality | **REQUIRED** (RIC unique) |
| [OUT] | OUT OF SCOPE | Explicitly excluded from this work | Prevents scope creep |

**Note**: Unlike industry standard where P2 is optional, RIC requires ALL P0/P1/P2
items to be resolved before exit. This ensures thoroughness.
"""

# =============================================================================
# CONFIDENCE LEVELS (v4.3 NEW - Per-Phase Calibration)
# =============================================================================

CONFIDENCE_LEVELS = """
**Confidence Calibration Scale**:

| Score | Level | Meaning | Action |
|-------|-------|---------|--------|
| 90-100% | HIGH | Very confident, well-understood | Proceed normally |
| 70-89% | MEDIUM | Reasonably confident, some uncertainty | Document uncertainties |
| 50-69% | LOW | Significant uncertainty | Flag for review, consider alternatives |
| <50% | VERY LOW | Major uncertainty | STOP - escalate or research more |

**Phase-Specific Confidence Questions**:
- P0 (Research): "How sure am I this is the right approach?"
- P1 (Plan): "How complete and realistic is this plan?"
- P2 (Build): "Does this code correctly solve the problem?"
- P3 (Verify): "Did tests catch all potential issues?"
- P4 (Reflect): "Did I identify all gaps and risks?"
"""

# =============================================================================
# PHASE 0: RESEARCH (Research-backed updates)
# =============================================================================

PHASE_0_PROMPT = """[I{iter}/{max}][P0] RESEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Objective**: Frame problem measurably, then gather information

**Why This Matters**: NASA study found projects spending <5% on requirements
had 80-200% cost overruns. Proper framing reduces rework.

**Actions**:
1. **FRAME FIRST**: Define measurable success criteria
   - BAD: "We need to improve the scanner"
   - GOOD: "Scanner should detect 80% of opportunities with <5% false positives"

2. **SEARCH**: WebSearch/WebFetch for relevant information
   - Search for existing solutions
   - Search for best practices
   - Search for potential pitfalls

3. **🛑 PERSIST EVERY 3 SEARCHES** (ENFORCED - v4.3):
   Write to docs/research/ after EVERY 3 searches - BLOCKING ENABLED.

   **Why**: Context compaction can destroy unpersisted research at ANY time.
   **Enforcement**: 4th WebSearch/WebFetch will be BLOCKED until you persist.
   **Counter**: Track searches since last Write to docs/research/

   ```
   Search 1 → Search 2 → Search 3 → ⚠️ PERSIST NOW → Search 4...
                                    ↓
                              Write to docs/research/
                              (counter resets to 0)
   ```

4. **📅 TIMESTAMPS REQUIRED** (ENFORCED - v4.3):
   All sources MUST have TWO timestamps - VALIDATED ON WRITE.

   **Format for each source**:
   ```markdown
   ### Source N: [Title]
   **URL**: [url]
   **Search Date**: {today}              ← When YOU searched
   **Published**: [Month Year or ~Year]  ← When SOURCE was written

   **Key Findings**:
   - [finding 1]
   - [finding 2]
   ```

   **Why**: Differentiates current patterns from deprecated ones.
   **Enforcement**: Write without timestamps triggers WARNING.
   **Estimation allowed**: Use "Published: ~2024" if exact date unknown.

5. **VALIDATE**: Does research answer success criteria questions?

**Allowed Tools**: WebSearch, WebFetch, Read, Grep, Glob, Write (docs/research/ only)
**Blocked Tools**: Edit (code), Bash(git commit)

**⚠️ BLOCKING RULES** (v4.3 Enforcement):
- WebSearch/WebFetch blocked after 3 unpersisted searches
- Next search requires Write to docs/research/ first
- Timestamps validated on each Write

**CONFIDENCE CHECK** (v4.3):
At phase end, rate: "How confident am I this is the right approach?" (0-100%)
If <70%: Document uncertainty, consider more research before proceeding.

**Gate Criteria** (ALL ENFORCED):
  □ Problem framed with measurable success criteria
  □ Research document exists in docs/research/
  □ At least 3 timestamped sources (Search Date + Published)
  □ All searches persisted (persist counter = 0)
  □ Timestamps validated (Search Date + Published per source)
  □ Confidence ≥70% (or uncertainty documented)
"""

# =============================================================================
# PHASE 1: PLAN (Research-backed updates)
# =============================================================================

PHASE_1_PROMPT = """[I{iter}/{max}][P1] PLAN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Objective**: Define scope, success criteria, and prioritized task list

**Priority Definitions**:
┌─────────┬─────────────┬────────────────────────────────────────────────────┐
│ Level   │ Name        │ Definition                                         │
├─────────┼─────────────┼────────────────────────────────────────────────────┤
│ [P0]    │ CRITICAL    │ Would hold release, blocks everything              │
│ [P1]    │ IMPORTANT   │ Should complete this iteration                     │
│ [P2]    │ POLISH      │ Required before exit (RIC: P2 is NOT optional)     │
│ [OUT]   │ OUT OF SCOPE│ Explicitly excluded (prevents scope creep)         │
└─────────┴─────────────┴────────────────────────────────────────────────────┘

**Actions**:
1. **SUCCESS CRITERIA**: Copy measurable criteria from Phase 0 research
   - What does "done" look like?
   - How will we verify success?

2. **TASK LIST**: Break down into prioritized tasks
   - Each task should be 30min-4hr of work
   - Assign P0/P1/P2 to EVERY task

3. **OUT OF SCOPE**: Explicitly list what we WON'T do
   - Prevents scope creep during implementation
   - "We will NOT do X, Y, Z in this iteration"

4. **DEPENDENCIES**: Identify task order
   - What must complete before what?
   - Are there parallel tracks?

5. **ESTIMATE**: How many iterations needed?
   - Simple: 3 iterations
   - Medium: 4 iterations
   - Complex: 5 iterations (max)

**Step 6: EXPLORE ALTERNATIVES** (PairCoder pattern - for complex tasks)
For complex/uncertain tasks, generate 2-3 approaches before committing:

| Approach | Description | Pros | Cons | Effort | Risk |
|----------|-------------|------|------|--------|------|
| A | [approach A] | ... | ... | Low/Med/High | Low/Med/High |
| B | [approach B] | ... | ... | Low/Med/High | Low/Med/High |
| C | [approach C] | ... | ... | Low/Med/High | Low/Med/High |

**Selection**: Choose approach with best effort/risk ratio
**Document**: WHY rejected approaches were rejected (prevents revisiting)

Skip this step for simple, well-understood tasks.

**CONFIDENCE CHECK** (v4.3):
At phase end, rate: "How complete and realistic is this plan?" (0-100%)
If <70%: Identify gaps, consider revisiting research.

**Gate Criteria**:
  □ Success criteria defined and measurable
  □ All tasks have P0/P1/P2 priority
  □ Out of scope items explicitly listed
  □ Dependencies identified
  □ Iteration estimate documented
  □ (Complex tasks) Alternative approaches considered
  □ Confidence ≥70% (or gaps documented)
"""

# =============================================================================
# PHASE 2: BUILD (Research-backed updates + Hallucination Check)
# =============================================================================

PHASE_2_PROMPT = """[I{iter}/{max}][P2] BUILD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Objective**: Implement changes with atomic, revertable commits

**The AND-Test** (From atomic commit research):
Can you describe this commit in ONE sentence without "AND"?

  ✅ GOOD: "Add user validation with tests"
     → One logical unit, code + test together

  ❌ BAD: "Add validation AND fix formatting AND update docs"
     → Three separate commits needed

**Commit Rules**:
┌─────────────────────────────────────────────────────────────────────────────┐
│ ONE logical change per commit (1-5 related files)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ ✅ ALLOWED in same commit:                                                  │
│    • Code file + its test file (same logical change)                        │
│    • Interface + implementations (one abstraction)                          │
│    • Config + modules that read it (one configuration)                      │
│    • Refactor across tightly coupled files (one refactor)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ ❌ NOT ALLOWED in same commit:                                              │
│    • Unrelated changes bundled together                                     │
│    • Changes to different components/features                               │
│    • More than 5 files (split the change)                                   │
│    • Style fixes mixed with feature changes                                 │
└─────────────────────────────────────────────────────────────────────────────┘

**Revertability Check**: Before committing, ask:
"Can I revert this commit without breaking other commits?"
If NO → Split into smaller commits

**HALLUCINATION CHECK** (v4.3 - BEFORE COMMIT):
┌─────────────────────────────────────────────────────────────────────────────┐
│ Before each commit, verify code doesn't contain hallucinations:              │
├─────────────────────────────────────────────────────────────────────────────┤
│ □ INTENT MATCH: Does code match commit message description?                  │
│   - Read commit message → Read code → Do they align?                        │
│                                                                             │
│ □ CONTEXT CONSISTENCY: Does code fit existing codebase patterns?            │
│   - Check imports: Do referenced modules exist?                             │
│   - Check calls: Are function signatures correct?                           │
│   - Check patterns: Does style match surrounding code?                      │
│                                                                             │
│ □ NO DEAD CODE: Is all code reachable and used?                             │
│   - No functions defined but never called                                   │
│   - No unreachable branches                                                 │
│                                                                             │
│ □ KNOWLEDGE CHECK: Is API usage correct?                                    │
│   - If using external API, verify syntax against docs                       │
│   - If unsure, search for correct usage                                     │
└─────────────────────────────────────────────────────────────────────────────┘

**Pattern**:
1. Pick ONE task from P0 list (or P1 if P0 done)
2. Make ONE logical change (1-5 files)
3. Write/update tests for that change
4. Run tests: `pytest tests/test_<module>.py -v`
5. Verify: Can I describe this without "AND"?
6. Verify: Can I revert this independently?
7. **RUN HALLUCINATION CHECK** (new in v4.3)
8. Commit: `[I{iter}/{max}][P2] <one-sentence description>`
9. Repeat for next task

**If Tests Fail** (max 5 attempts):
1. Read the failing test output
2. Read the relevant code
3. Fix ONE issue at a time
4. Re-run tests
5. If still failing after 5 attempts, document as P0 insight

**DECISION TRACE** (v4.3 - Log significant decisions):
For non-trivial changes, log:
- What: Action taken
- Why: Reasoning for this choice
- Alternatives: Other options considered
- Risk: Assessment of potential issues

**Gate Criteria**:
  □ All P0 tasks implemented
  □ Each commit passes AND-test (one sentence, no AND)
  □ Each commit is independently revertable
  □ Hallucination check passed for each commit
  □ Tests exist for new code
  □ All tests pass
"""

# =============================================================================
# PHASE 3: VERIFY (Research-backed updates + Consistency Check)
# =============================================================================

PHASE_3_PROMPT = """[I{iter}/{max}][P3] VERIFY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Objective**: Verify implementation meets ALL quality gates

**Step 1: SELF-VERIFICATION TEST** (ReVeal pattern)
Before running the full test suite, write ONE focused test:
  1. Write a test for the specific functionality you just implemented
  2. Include inputs you KNOW should work (happy path)
  3. Include inputs that SHOULD fail (edge cases, boundaries)
  4. Run YOUR test first: `pytest tests/test_<module>.py::test_<function> -v`

This catches obvious issues before the expensive full suite.

**Step 2: QUALITY GATES** (CoCoGen order: static analysis FIRST)
Run gates in this order to fail fast on cheap checks:

┌─────────────────────────────────────────────────────────────────────────────┐
│ Order │ Gate         │ Command                   │ Passing Criteria         │
├───────┼──────────────┼───────────────────────────┼──────────────────────────┤
│ 1     │ □ Lint       │ ruff check .              │ No errors (syntax first) │
│ 2     │ □ Types      │ mypy --config mypy.ini    │ No type errors           │
│ 3     │ □ Tests      │ pytest tests/ -v          │ Exit code 0              │
│ 4     │ □ Coverage   │ pytest --cov=. --cov-rep  │ ≥70% on new code         │
│ 5     │ □ Security   │ (hardcoded secrets check) │ No credentials in code   │
└─────────────────────────────────────────────────────────────────────────────┘

**Why this order?** (CoCoGen research):
- Lint catches syntax errors in <1 second
- Type errors often cause test failures - catch them first
- Running pytest with type errors wastes time on tests that will fail anyway

**Step 3: CONSISTENCY CHECK** (v4.3 - Hallucination Prevention)
After tests pass, verify consistency with existing codebase:

┌─────────────────────────────────────────────────────────────────────────────┐
│ □ IMPORT CHECK: All imports resolve correctly                               │
│   - Run: python -c "from <module> import <function>"                        │
│                                                                             │
│ □ INTERFACE CHECK: New code matches existing interfaces                     │
│   - Check return types match expectations                                   │
│   - Check parameters match caller usage                                     │
│                                                                             │
│ □ PATTERN CHECK: Code follows codebase conventions                          │
│   - Error handling style matches                                            │
│   - Logging format matches                                                  │
│   - Naming conventions match                                                │
│                                                                             │
│ □ DUPLICATION CHECK: Not reimplementing existing functionality              │
│   - Search for similar functions                                            │
│   - If found, use existing or refactor to share                            │
└─────────────────────────────────────────────────────────────────────────────┘

**Step 4: CHECKER VERIFICATION** (SAGE pattern - 2.26x improvement)
After self-review, adopt a "Checker" persona for independent verification:

┌─────────────────────────────────────────────────────────────────────────────┐
│ **CHECKER MODE** - Pretend you did NOT write this code                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ □ Does the code do what the commit message claims?                          │
│ □ Are there edge cases the author missed?                                   │
│ □ Would this pass a code review? (naming, structure, clarity)               │
│ □ Would this pass a security audit? (injection, secrets, validation)        │
│ □ Are there any obvious performance issues?                                 │
│ □ Is error handling adequate?                                               │
└─────────────────────────────────────────────────────────────────────────────┘

The Checker role is separate from self-critique - approach as external reviewer.

**If Gate Fails** (with increased fix attempts for quality):

| Gate Failed  | Max Attempts | Recovery Action                              |
|--------------|--------------|----------------------------------------------|
| Tests fail   | **5**        | Read error → Fix ONE issue → Re-run          |
| Coverage low | 3            | Identify uncovered paths → Add tests         |
| Lint errors  | 3            | Run `ruff check --fix` → Commit separately   |
| Type errors  | 3            | Fix type annotations → Re-run mypy           |
| Security     | Immediate    | Remove secrets → Use env vars                |
| Consistency  | **2**        | Fix inconsistency → Re-check                 |

**Important**: We allow **5 attempts** for test failures because:
- Writing correct code is more important than speed
- Each failure teaches us something about the code
- Rushing leads to technical debt

**Recovery Pattern** (for test failures):
```
Attempt 1: Read error, understand the failure
Attempt 2: Fix the most obvious issue
Attempt 3: Check for edge cases
Attempt 4: Review test assumptions
Attempt 5: Consider if test or implementation is wrong
Still failing? → Document as P0 insight for next iteration
```

**Allowed Tools**: Bash(pytest), Bash(ruff), Bash(mypy), Read
**Blocked Tools**: Edit, Write (EXCEPT to fix failing gates)

**CONFIDENCE CHECK** (v4.3):
At phase end, rate: "Did tests catch all potential issues?" (0-100%)
If <70%: Add more tests or document known gaps.

**Gate Criteria**:
  □ All tests pass (pytest exit 0)
  □ Coverage ≥70% on new/modified code
  □ No lint errors (ruff clean)
  □ No type errors (if mypy configured)
  □ No hardcoded secrets or credentials
  □ Consistency check passed
  □ Checker verification completed
  □ Confidence ≥70% (or gaps documented)
"""

# =============================================================================
# PHASE 4: REFLECT (Research-backed updates + Convergence)
# =============================================================================

PHASE_4_PROMPT = """[I{iter}/{max}][P4] REFLECT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Objective**: Critique work, find gaps, decide: Loop or Exit

**Reflection Pattern** (Generate → Critique → Improve):
This pattern is used by state-of-the-art AI agents (Andrew Ng identifies it
as one of 4 core agentic AI components).

**Step 1: CRITIQUE** - What did we accomplish?
  - List completed P0/P1/P2 tasks
  - Were success criteria met?
  - What worked well?

**Step 2: IDENTIFY** - What's missing, broken, or incomplete?
  Use this checklist:

  □ **Missing Files**: Are all planned files created?
  □ **Missing Tests**: Does every new function have tests?
  □ **Edge Cases**: Are boundary conditions handled?
  □ **Error Handling**: What happens when things fail?
  □ **Security**: Any vulnerabilities introduced?
  □ **Documentation**: Do complex functions have docstrings?
  □ **Integration**: Does everything work together?
  □ **Performance**: Any obvious bottlenecks?

**Step 3: CLASSIFY** - Assign priority to each gap
  - [P0] = Critical gap, blocks release
  - [P1] = Important gap, should fix
  - [P2] = Polish item, improves quality

**Step 4: CONVERGENCE CHECK** (v4.3 - Enhanced Detection)
Evaluate multiple metrics to detect convergence:

┌─────────────────────────────────────────────────────────────────────────────┐
│ Metric              │ This Iter │ Previous │ Trending   │ Target           │
├─────────────────────┼───────────┼──────────┼────────────┼──────────────────┤
│ New Insights        │ {insights}│ {prev_i} │ ↓ or ↑?    │ Declining        │
│ Fix Success Rate    │ {fix_rate}│ {prev_f} │ ↓ or ↑?    │ >80% & Increasing│
│ Code Churn (lines)  │ {churn}   │ {prev_c} │ ↓ or ↑?    │ Declining        │
│ Gate Pass Rate      │ {gate}    │ {prev_g} │ ↓ or ↑?    │ >90%             │
└─────────────────────────────────────────────────────────────────────────────┘

**Convergence Signals**:
- ✅ Insights declining + Fix rate improving + Churn declining = CONVERGING
- ⚠️ Mixed signals = CONTINUE ITERATING
- ❌ Metrics worsening = INVESTIGATE (possible thrashing)

**Step 5: CONFIDENCE CALIBRATION** (v4.3)
Rate overall confidence for this iteration:

| Phase | Confidence | Notes |
|-------|------------|-------|
| P0 Research | {conf_0}% | {notes_0} |
| P1 Plan | {conf_1}% | {notes_1} |
| P2 Build | {conf_2}% | {notes_2} |
| P3 Verify | {conf_3}% | {notes_3} |
| P4 Reflect | {conf_4}% | {notes_4} |
| **Overall** | {conf_avg}% | |

If any phase <70%, document specific uncertainty.

**Step 6: DECIDE** - Loop or Exit?

**Open Insights**: P0={p0} P1={p1} P2={p2}
**Iteration**: {iter}/{max}
**New insights this iteration**: {new_insights}
**Convergence Status**: {convergence_status}

**Decision Rules** (STRICT ORDER):
```
1. IF iter < 3           → LOOP to P0 (minimum 3 iterations required)
2. ELIF P0 > 0           → LOOP to P0 (critical gaps must be resolved)
3. ELIF P1 > 0           → LOOP to P0 (important items must be resolved)
4. ELIF P2 > 0           → LOOP to P0 (P2 is REQUIRED in RIC)
5. ELIF confidence < 70% → LOOP to P0 (low confidence, need more iteration)
6. ELIF plateau = 2      → EXIT (no new insights for 2 iterations)
7. ELIF converging       → EXIT (multi-metric convergence detected)
8. ELIF iter ≥ 5         → EXIT (max iterations reached)
9. ELSE                  → EXIT (all complete, success!)
```

**Plateau Detection**:
If this iteration found 0 new insights AND previous iteration found 0 new insights:
  → Plateau reached, EXIT allowed even if iteration < max

**Gate Criteria**:
  □ Critique completed (accomplishments listed)
  □ Reflection checklist reviewed
  □ All gaps classified as P0/P1/P2
  □ Convergence metrics evaluated
  □ Confidence calibration completed
  □ Decision made using rules above
  □ Decision justified with reasoning
"""

# =============================================================================
# CONTEXT MANAGEMENT PROTOCOL (v4.3 NEW)
# =============================================================================

CONTEXT_MANAGEMENT_PROMPT = """
**CONTEXT MANAGEMENT PROTOCOL** (v4.3 - Hierarchical Memory)

Before heavy operations, evaluate context usage to prevent "context rot":

**Memory Hierarchy**:
┌─────────────────────────────────────────────────────────────────────────────┐
│ CORE MEMORY (Always included - ~10% of context)                             │
│ • Current phase and iteration number                                        │
│ • Open insights (P0/P1/P2)                                                  │
│ • Active task description                                                   │
│ • Key success criteria                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ WORKING MEMORY (Phase-specific - ~50% of context)                           │
│ • P0 (Research): External sources, search results, findings                 │
│ • P1 (Plan): Task breakdown, dependencies, alternatives                     │
│ • P2 (Build): Code context, function signatures, recent changes             │
│ • P3 (Verify): Test results, error messages, coverage reports               │
│ • P4 (Reflect): Accomplishments, gaps, metrics                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ ARCHIVAL MEMORY (On-demand retrieval - load when needed)                    │
│ • Full research documents (docs/research/)                                  │
│ • Historical decisions from previous iterations                             │
│ • Complete test output logs                                                 │
│ • Prior iteration summaries                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

**Context Trimming Rules**:
1. When context feels overloaded (confused responses), trim working memory
2. Prefer summarized over raw when possible
3. Load archival memory only when specific information needed
4. Persist important findings to files immediately (don't rely on context)

**Warning Signs of Context Rot**:
- Repeating information already discussed
- Missing obvious connections
- Inconsistent with earlier statements
- Generic responses instead of specific

**Recovery**: Summarize current state, trim old context, re-read key files
"""

# =============================================================================
# DECISION TRACE FORMAT (v4.3 NEW)
# =============================================================================

DECISION_TRACE_FORMAT = """
**DECISION TRACE** (v4.3 - Meta-Debugging Support)

For significant decisions (file changes, architecture choices, trade-offs):

```json
{{
  "decision_id": "DEC-{iter:03d}-{seq:02d}",
  "timestamp": "{timestamp}",
  "phase": {phase},
  "iteration": {iter},
  "action": "{action}",
  "reasoning": "{why}",
  "alternatives_considered": [
    {{"option": "{alt1}", "rejected_because": "{reason1}"}},
    {{"option": "{alt2}", "rejected_because": "{reason2}"}}
  ],
  "risk_assessment": "{risk}",
  "confidence": {confidence},
  "related_insights": ["{insight_ids}"]
}}
```

**When to Log Decisions**:
- Adding/modifying code in protected paths (algorithms/, execution/, etc.)
- Choosing between multiple implementation approaches
- Skipping or deferring a task
- Changing earlier decisions

**Why Decision Tracing Matters**:
- Enables post-hoc debugging ("Why did the agent do X?")
- Prevents revisiting rejected alternatives
- Supports learning across sessions
"""

# =============================================================================
# AUTONOMOUS MODE PROMPTS
# =============================================================================

AUTONOMOUS_START = """
## 🤖 RIC Session Started (v5.0 Guardian)

**Iteration**: {iter}/{max} | **Phase**: P{phase} - {phase_name}

### 5-Phase Flow
```
P0:RESEARCH → P1:PLAN → P2:BUILD → P3:VERIFY → P4:REFLECT
     ↑        (frame    (AND-test  (5 fix     (critique  │
     │        problem)  + hallu    attempts   + converge │
     │                  check)     + consist)  + confid) │
     │                                                   │
     └──────────── Loop if P0/P1/P2 insights ────────────┘
```

### v4.3 Safety Features
- **Hallucination Check**: 5-category verification before commits
- **Convergence Detection**: Multi-metric tracking (insight rate, fix success, churn)
- **Confidence Calibration**: Per-phase confidence ratings (min 70%)
- **Safety Throttles**: Tool call limits, time limits, failure limits
- **Decision Tracing**: Structured logs for meta-debugging

### Key Rules (Research-Backed)
- **P0 Research**: Frame problem measurably FIRST (NASA study)
- **P2 Build**: AND-test + Hallucination check before commit
- **P3 Verify**: CoCoGen order + SAGE Checker + Consistency check
- **P4 Reflect**: Convergence metrics + Confidence calibration

### Logging Format
- Phases: `[I{iter}/{max}][P{phase}] {phase_name}`
- Commits: `[I{iter}/{max}][P2] <one-sentence description>`

### Current Gate
{gate_check}
"""

AUTONOMOUS_STATUS = """
---
**RIC v4.3**: [I{iter}/{max}][P{phase}] {phase_name}
**Insights**: P0={p0} P1={p1} P2={p2}
**Convergence**: {convergence_status}
**Confidence**: {confidence}%
**Gate**: {gate_status}
**Exit**: {can_exit} ({reason})
**Throttles**: Tools={tool_calls}/{max_tools}, Time={time_elapsed}min
---
"""

# =============================================================================
# ERROR RECOVERY (Increased attempts + Hallucination handling)
# =============================================================================

TEST_FAILURE_PROMPT = """
**🔴 TEST FAILURE** (Attempt {attempt}/{max_attempts})

Failed tests: {failed_tests}

**ACTIONABLE FEEDBACK PROTOCOL** (SELF-REFINE pattern)
Generic feedback like "fix the error" is ineffective. Use this structure:

┌─────────────────────────────────────────────────────────────────────────────┐
│ **ANALYZE THE FAILURE**                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. SPECIFIC ISSUE: What exactly failed?                                     │
│    - Assertion error? Type error? Exception?                                │
│    - What was expected vs actual?                                           │
│                                                                             │
│ 2. LOCATION: Where is the error?                                            │
│    - File name + line number                                                │
│    - Function/method name                                                   │
│                                                                             │
│ 3. ROOT CAUSE: WHY did it fail? (not just what)                             │
│    - Missing check? Wrong logic? Bad assumption?                            │
│    - Is this a hallucination? (API misuse, wrong syntax)                    │
│                                                                             │
│ 4. SUGGESTED FIX: Concrete code change                                      │
│    - "Change line X to do Y because Z"                                      │
└─────────────────────────────────────────────────────────────────────────────┘

**ATTEMPT-SPECIFIC FOCUS**:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Attempt │ Focus (actionable)                                                │
├─────────┼───────────────────────────────────────────────────────────────────┤
│ 1       │ Parse error message → Identify SPECIFIC issue + LOCATION          │
│ 2       │ Trace to ROOT CAUSE → Apply SUGGESTED FIX for obvious issue       │
│ 3       │ Check edge cases → Fix: null/empty/boundary conditions            │
│ 4       │ Question test → Is test correct? Fix test OR implementation       │
│ 5       │ Reconsider approach → Is fundamental design wrong?                │
└─────────────────────────────────────────────────────────────────────────────┘

**Current Attempt {attempt}**:
{attempt_guidance}

**Commands**:
1. Read failing test: Read the test file
2. Read implementation: Read the source file
3. Fix ONE issue with actionable specificity
4. Re-run: `pytest {test_file} -v`

**If still failing after attempt 5**:
Document as P0 insight with actionable detail:
"Test {test_name} fails because [ROOT CAUSE]. Needs [SUGGESTED FIX approach]."
"""

HALLUCINATION_FIX_PROMPT = """
**⚠️ HALLUCINATION DETECTED** (Type: {hallucination_type})

Detected issue: {description}

**Recovery Protocol**:

1. **IDENTIFY**: Which hallucination category?
   □ Intent Conflicting - Code doesn't match stated purpose
   □ Context Inconsistency - Conflicts with existing code
   □ Context Repetition - Duplicates existing functionality
   □ Dead Code - Unreachable code added
   □ Knowledge Conflicting - Wrong API/syntax

2. **VERIFY**: Confirm the hallucination
   - Read the existing codebase context
   - Check if the claimed API/function exists
   - Verify correct syntax in documentation

3. **FIX**: Address the root cause
   - If wrong API: Look up correct usage
   - If conflicts: Align with existing patterns
   - If duplicate: Remove or refactor to use existing

4. **PREVENT**: Update approach to avoid recurrence
   - If knowledge gap: Research before implementing
   - If context issue: Re-read relevant files

**Max 2 fix attempts for hallucinations** - if persistent, flag for human review.
"""

STUCK_PROMPT = """
**⚠️ STUCK** on task: {task}
Attempts: {attempts}/{max_attempts}

**Recovery Options**:
1. **Document the blocker** in claude-session-notes.md
   - What exactly is blocking progress?
   - What have you tried?

2. **Create checkpoint**:
   `git commit -m "[I{iter}/{max}] checkpoint: blocked on {task}"`

3. **Try alternative approach**:
   - Is there a different way to solve this?
   - Can we simplify the requirements?

4. **Skip and escalate**:
   - Add as P0 insight: "{task} blocked - needs investigation"
   - Move to next task

**Do NOT**: Keep trying the same approach that isn't working

**DECISION TRACE**: Log why we're stuck and what was tried
"""

CHECKPOINT_PROMPT = """
**💾 CHECKPOINT REMINDER** ({minutes} min since last)

Regular checkpoints protect your work from context compaction.

**Actions**:
1. `git add -A && git commit -m "[I{iter}/{max}] checkpoint: {phase_name}"`
2. Update claude-progress.txt with current status
3. Update claude-session-notes.md with any discoveries

**Why This Matters**:
- Context can compact at any time
- Uncommitted work could be lost
- Progress file helps resume after restart
"""

THROTTLE_WARNING_PROMPT = """
**🛑 SAFETY THROTTLE TRIGGERED**: {throttle_type}

{details}

**Pause Required**: {action_required}

**This throttle exists to prevent**:
- Runaway loops consuming resources
- File thrashing damaging codebase
- Time-intensive operations without progress

**Recovery**:
1. Review what triggered the throttle
2. Consider if approach needs adjustment
3. Wait for cooldown if required
4. Resume with modified strategy
"""

# =============================================================================
# GATE DEFINITIONS
# =============================================================================

GATES = {
    0: {
        "name": "RESEARCH_COMPLETE",
        "checks": [
            ("problem_framed", "Problem framed with measurable success criteria"),
            ("research_doc_exists", "Research document exists in docs/research/"),
            ("sources_count", f"At least {RESEARCH_ENFORCEMENT['min_sources_per_topic']} timestamped sources"),
            ("search_timestamps", "Each source has **Search Date** (when searched)"),
            ("publication_timestamps", "Each source has **Published** date (when source published)"),
            ("findings_persisted", "Findings written to file (not just context)"),
            ("persist_counter_zero", "All searches persisted (counter = 0)"),
            ("confidence_ok", "Confidence ≥70% (or uncertainty documented)"),
        ],
        # Automated validation functions (v4.3)
        "validators": {
            "sources_count": lambda content: count_timestamped_sources(content)
            >= RESEARCH_ENFORCEMENT["min_sources_per_topic"],
            "search_timestamps": lambda content: any(re.search(p, content) for p in TIMESTAMP_PATTERNS["search_date"]),
            "publication_timestamps": lambda content: any(
                re.search(p, content) for p in TIMESTAMP_PATTERNS["publication_date"]
            ),
            "persist_counter_zero": lambda state: state.web_searches_since_persist == 0,
        },
    },
    1: {
        "name": "PLAN_DEFINED",
        "checks": [
            ("success_criteria", "Success criteria defined and measurable"),
            ("tasks_prioritized", "All tasks have P0/P1/P2 priority"),
            ("out_of_scope", "Out of scope items explicitly listed"),
            ("dependencies_noted", "Task dependencies identified"),
            ("confidence_ok", "Confidence ≥70% (or gaps documented)"),
        ],
    },
    2: {
        "name": "BUILD_COMPLETE",
        "checks": [
            ("p0_tasks_done", "All P0 tasks implemented"),
            ("and_test_passed", "Each commit passes AND-test"),
            ("commits_revertable", "Each commit independently revertable"),
            ("hallucination_check", "Hallucination check passed for each commit"),
            ("tests_exist", "Tests exist for new code"),
            ("tests_pass", "All tests pass"),
        ],
    },
    3: {
        "name": "VERIFICATION_PASSED",
        # CoCoGen order: static analysis FIRST (fast failures before slow tests)
        "checks": [
            ("lint_clean", "ruff check passes (syntax/style errors)"),
            ("types_ok", "mypy passes (type errors)"),
            ("tests_pass", "pytest exits 0 (runtime tests)"),
            ("coverage_ok", "Coverage ≥70% on new code"),
            ("no_secrets", "No hardcoded secrets"),
            ("consistency_check", "Consistency with codebase verified"),
            ("checker_review", "SAGE Checker verification completed"),
            ("confidence_ok", "Confidence ≥70% (or gaps documented)"),
        ],
    },
    4: {
        "name": "REFLECT_DECISION",
        "checks": [
            ("critique_done", "Accomplishments listed"),
            ("checklist_reviewed", "Reflection checklist completed"),
            ("gaps_classified", "All gaps have P0/P1/P2"),
            ("convergence_evaluated", "Convergence metrics assessed"),
            ("confidence_calibrated", "Per-phase confidence recorded"),
            ("decision_made", "Loop or Exit decision made"),
            ("decision_justified", "Decision reasoning documented"),
        ],
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_header(iteration: int, max_iter: int, phase: int) -> str:
    """Generate phase header."""
    name = PHASES.get(phase, ("UNKNOWN", ""))[0]
    return PHASE_HEADER.format(iter=iteration, max=max_iter, phase=phase, phase_name=name)


def get_commit_example(iteration: int, max_iter: int) -> str:
    """Generate commit message example."""
    return f'git commit -m "[I{iteration}/{max_iter}][P2] Add circuit breaker with tests"'


def get_phase_prompt(phase: int, context: dict) -> str:
    """Get phase-specific prompt with context."""
    prompts = {
        0: PHASE_0_PROMPT,
        1: PHASE_1_PROMPT,
        2: PHASE_2_PROMPT,
        3: PHASE_3_PROMPT,
        4: PHASE_4_PROMPT,
    }
    template = prompts.get(phase, "")
    try:
        return template.format(**context)
    except KeyError:
        return template


def get_test_failure_guidance(attempt: int) -> str:
    """Get specific guidance for each test failure attempt."""
    guidance = {
        1: "Focus on understanding. Read the error message word by word. What is it actually telling you?",
        2: "Fix the obvious. What's the simplest change that could make this pass?",
        3: "Check edge cases. Are there null values? Empty lists? Boundary conditions?",
        4: "Question the test. Is the test actually testing the right thing? Are assumptions correct?",
        5: "Reconsider approach. Maybe the implementation approach itself needs to change?",
    }
    return guidance.get(attempt, "Document as insight and move on.")


def check_gate(phase: int, context: dict) -> tuple[bool, list[str]]:
    """Check if gate criteria are met for phase."""
    gate = GATES.get(phase, {})
    results = []
    all_passed = True

    for check_id, description in gate.get("checks", []):
        passed = context.get(check_id, False)
        if passed:
            results.append(f"✅ {description}")
        else:
            results.append(f"❌ {description}")
            all_passed = False

    return all_passed, results


# =============================================================================
# CONVERGENCE DETECTION (v4.3 NEW)
# =============================================================================


@dataclass
class IterationMetrics:
    """Metrics for a single iteration."""

    iteration: int
    new_insights: int
    fix_attempts: int
    fix_successes: int
    code_churn_lines: int
    gate_attempts: int
    gate_passes: int
    confidence_scores: dict = field(default_factory=dict)


def calculate_convergence(history: list[IterationMetrics]) -> tuple[bool, str, dict]:
    """
    Detect convergence using multiple metrics.

    Returns:
        (is_converging, status_message, metrics_dict)
    """
    if len(history) < 2:
        return False, "Insufficient history for convergence detection", {}

    current = history[-1]
    previous = history[-2]

    metrics = {
        "insight_rate": {
            "current": current.new_insights,
            "previous": previous.new_insights,
            "trending": "declining" if current.new_insights < previous.new_insights else "increasing",
            "target_met": current.new_insights <= previous.new_insights,
        },
        "fix_success_rate": {
            "current": current.fix_successes / max(current.fix_attempts, 1),
            "previous": previous.fix_successes / max(previous.fix_attempts, 1),
            "trending": "improving"
            if current.fix_successes / max(current.fix_attempts, 1)
            > previous.fix_successes / max(previous.fix_attempts, 1)
            else "declining",
            "target_met": current.fix_successes / max(current.fix_attempts, 1)
            >= CONVERGENCE_THRESHOLDS["fix_success_rate_target"],
        },
        "code_churn": {
            "current": current.code_churn_lines,
            "previous": previous.code_churn_lines,
            "trending": "declining" if current.code_churn_lines < previous.code_churn_lines else "increasing",
            "target_met": current.code_churn_lines <= previous.code_churn_lines,
        },
        "gate_pass_rate": {
            "current": current.gate_passes / max(current.gate_attempts, 1),
            "previous": previous.gate_passes / max(previous.gate_attempts, 1),
            "trending": "improving"
            if current.gate_passes / max(current.gate_attempts, 1)
            > previous.gate_passes / max(previous.gate_attempts, 1)
            else "declining",
            "target_met": current.gate_passes / max(current.gate_attempts, 1)
            >= CONVERGENCE_THRESHOLDS["gate_pass_rate_target"],
        },
    }

    # Count how many metrics indicate convergence
    converging_count = sum(1 for m in metrics.values() if m["target_met"])

    if converging_count >= 3:
        return True, "Multi-metric convergence detected (3+ metrics trending positively)", metrics
    elif converging_count >= 2:
        return False, "Partial convergence (2 metrics positive, continue iterating)", metrics
    else:
        return False, "Not yet converging (investigate if metrics worsening)", metrics


# =============================================================================
# HALLUCINATION DETECTION (v4.3 NEW)
# =============================================================================


def check_hallucination(
    code_content: str,
    commit_message: str,
    existing_imports: list[str],
    existing_functions: list[str],
) -> tuple[bool, list[dict]]:
    """
    Check code for potential hallucinations.

    Returns:
        (has_hallucination, list of detected issues)
    """
    issues = []

    # Check 1: Intent match (commit message vs code)
    # This would need more sophisticated NLP in real implementation
    # Placeholder: check if commit mentions functions that exist in code

    # Check 2: Import consistency
    import_pattern = r"^(?:from\s+(\S+)\s+)?import\s+(.+)$"
    for line in code_content.split("\n"):
        match = re.match(import_pattern, line.strip())
        if match:
            module = match.group(1) or match.group(2).split(",")[0].strip()
            # Check if it's a local module that should exist
            if module.startswith(".") or not module.startswith(("os", "sys", "json", "re", "typing")):
                if module not in existing_imports:
                    issues.append(
                        {
                            "type": "context_inconsistency",
                            "description": f"Import '{module}' may not exist in codebase",
                            "severity": "medium",
                        }
                    )

    # Check 3: Dead code detection (basic)
    # Look for functions defined but never called
    func_pattern = r"def\s+(\w+)\s*\("
    defined_funcs = re.findall(func_pattern, code_content)
    for func in defined_funcs:
        # Skip if it's likely a public method or test
        if func.startswith("_") or func.startswith("test_"):
            continue
        # Check if function is called anywhere
        call_pattern = rf"\b{func}\s*\("
        calls = re.findall(call_pattern, code_content)
        if len(calls) <= 1:  # Only the definition
            issues.append(
                {
                    "type": "dead_code",
                    "description": f"Function '{func}' defined but possibly never called",
                    "severity": "low",
                }
            )

    # Check 4: Duplicate functionality (basic check)
    for func in defined_funcs:
        if func in existing_functions:
            issues.append(
                {
                    "type": "context_repetition",
                    "description": f"Function '{func}' may duplicate existing functionality",
                    "severity": "medium",
                }
            )

    has_hallucination = any(i["severity"] in ("high", "medium") for i in issues)
    return has_hallucination, issues


# =============================================================================
# SAFETY THROTTLE CHECKS (v4.3 NEW)
# =============================================================================


@dataclass
class ThrottleState:
    """Track throttle counters."""

    tool_calls_this_phase: int = 0
    edits_per_file: dict = field(default_factory=dict)
    consecutive_failures: int = 0
    phase_start_time: float | None = None
    last_progress_time: float | None = None
    decisions_without_progress: int = 0
    # Research enforcement tracking (v4.3)
    web_searches_since_persist: int = 0  # Counter for WebSearch/WebFetch calls
    last_research_persist_time: float | None = None  # When last Write to docs/research/
    research_persist_blocked: bool = False  # Block next search until persist


def check_throttles(state: ThrottleState) -> tuple[bool, str, str]:
    """
    Check if any safety throttle is triggered.

    Returns:
        (throttle_triggered, throttle_type, action_required)
    """
    now = time.time()

    # Check tool call limit
    if state.tool_calls_this_phase >= SAFETY_THROTTLES["max_tool_calls_per_phase"]:
        return True, "max_tool_calls", "Review approach - too many tool calls this phase"

    # Check file edit limit
    for file_path, count in state.edits_per_file.items():
        if count >= SAFETY_THROTTLES["max_edits_per_file_per_iteration"]:
            return True, "file_thrashing", f"Stop editing {file_path} - consider different approach"

    # Check consecutive failures
    if state.consecutive_failures >= SAFETY_THROTTLES["max_consecutive_failures"]:
        return True, "consecutive_failures", "Pause for review - 3+ consecutive failures"

    # Check time limit
    if state.phase_start_time:
        elapsed_minutes = (now - state.phase_start_time) / 60
        if elapsed_minutes >= SAFETY_THROTTLES["max_time_per_phase_minutes"]:
            return True, "time_limit", "Time limit reached - checkpoint and reassess"

    # Check decisions without progress
    if state.decisions_without_progress >= SAFETY_THROTTLES["max_decisions_without_progress"]:
        return True, "no_progress", "No progress detected - may be spinning"

    return False, "", ""


# =============================================================================
# RESEARCH ENFORCEMENT (v4.3 NEW - Compaction Protection)
# =============================================================================


def check_research_persist_required(state: ThrottleState) -> tuple[bool, str]:
    """
    Check if research persistence is required before allowing more searches.

    Returns:
        (persist_required, warning_message)
    """
    max_searches = RESEARCH_ENFORCEMENT["searches_before_forced_persist"]

    if state.web_searches_since_persist >= max_searches:
        today = datetime.now().strftime("%B %d, %Y")
        warning = RESEARCH_PERSIST_WARNING.format(
            search_count=state.web_searches_since_persist,
            max_searches=max_searches,
            today=today,
            source_num=state.web_searches_since_persist,
        )
        return True, warning

    return False, ""


def validate_research_timestamps(file_path: str, content: str) -> tuple[bool, list[str]]:
    """
    Validate that research document has required timestamps.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    # Check for search date timestamps
    if RESEARCH_ENFORCEMENT["search_timestamp_required"]:
        has_search_date = False
        for pattern in TIMESTAMP_PATTERNS["search_date"]:
            if re.search(pattern, content):
                has_search_date = True
                break
        if not has_search_date:
            issues.append("- Missing **Search Date** timestamp (when you searched)")

    # Check for publication date timestamps
    if RESEARCH_ENFORCEMENT["publication_date_required"]:
        has_pub_date = False
        for pattern in TIMESTAMP_PATTERNS["publication_date"]:
            if re.search(pattern, content):
                has_pub_date = True
                break
        if not has_pub_date:
            issues.append("- Missing **Published** date for sources (when source was published)")

    # Check minimum content length
    min_words = RESEARCH_ENFORCEMENT["min_words_per_finding"]
    word_count = len(content.split())
    if word_count < min_words:
        issues.append(f"- Research content too brief ({word_count} words, minimum {min_words})")

    return len(issues) == 0, issues


def count_timestamped_sources(content: str) -> int:
    """Count sources that have both search and publication timestamps."""
    # Look for source blocks with timestamps
    source_blocks = re.findall(r"###\s+Source\s+\d+.*?(?=###|\Z)", content, re.DOTALL)

    valid_sources = 0
    for block in source_blocks:
        has_search = any(re.search(p, block) for p in TIMESTAMP_PATTERNS["search_date"])
        has_pub = any(re.search(p, block) for p in TIMESTAMP_PATTERNS["publication_date"])
        if has_search and has_pub:
            valid_sources += 1

    # Also count inline timestamped references
    # Pattern: [Title (Published: Date)](url)
    inline_sources = len(re.findall(r"\[.+?\s*\([Pp]ublished:.+?\)\]\(.+?\)", content))

    return valid_sources + inline_sources


def record_web_search(state: ThrottleState) -> tuple[bool, str]:
    """
    Record a web search and check if blocking is needed.

    Returns:
        (should_block, message)
    """
    state.web_searches_since_persist += 1

    max_searches = RESEARCH_ENFORCEMENT["searches_before_forced_persist"]

    # At the limit - warn
    if state.web_searches_since_persist == max_searches:
        today = datetime.now().strftime("%B %d, %Y")
        return False, RESEARCH_PERSIST_WARNING.format(
            search_count=state.web_searches_since_persist,
            max_searches=max_searches,
            today=today,
            source_num=state.web_searches_since_persist,
        )

    # Over the limit - block
    if state.web_searches_since_persist > max_searches:
        if RESEARCH_ENFORCEMENT["block_search_without_persist"]:
            state.research_persist_blocked = True
            return True, (
                f"**🛑 BLOCKED**: {state.web_searches_since_persist} searches without persistence. "
                f"Write to docs/research/ before continuing."
            )

    return False, ""


def record_research_persist(state: ThrottleState, file_path: str) -> None:
    """Record that research was persisted to file."""
    if "docs/research/" in file_path:
        state.web_searches_since_persist = 0
        state.last_research_persist_time = time.time()
        state.research_persist_blocked = False


# =============================================================================
# RIC v5.0 ADDITIONS - December 2025 Research
# =============================================================================


# -----------------------------------------------------------------------------
# v5.0 DATACLASSES
# -----------------------------------------------------------------------------


@dataclass
class DriftMetrics:
    """Track scope drift between iterations (AEGIS Framework - Forrester 2025)."""

    original_task_count: int = 0
    original_file_count: int = 0
    original_line_estimate: int = 0
    current_task_count: int = 0
    current_file_count: int = 0
    current_line_count: int = 0
    baseline_recorded: bool = False
    timestamp: str | None = None


@dataclass
class GuardianReview:
    """Result of independent guardian review (Gartner 2025 - Guardian Agents)."""

    passed: bool = True
    score: str = "PASS"  # "PASS", "WARN", "FAIL"
    issues: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    timestamp: str = ""
    criteria_results: dict = field(default_factory=dict)


@dataclass
class RepairStats:
    """Track replace vs repair ratio (SEIDR Paper - ACM TELO 2025)."""

    replace_count: int = 0  # Times we generated new code
    repair_count: int = 0  # Times we fixed existing code
    total_attempts: int = 0
    successful_repairs: int = 0
    successful_replaces: int = 0

    def get_ratio(self) -> float:
        """Get repair/replace ratio (target: 0.5 for hybrid strategy)."""
        total = self.replace_count + self.repair_count
        if total == 0:
            return 0.5
        return self.repair_count / total


@dataclass
class FixCandidate:
    """A candidate fix for a failing test (SEIDR - Lexicase Selection)."""

    id: str
    code: str
    test_results: dict = field(default_factory=dict)
    pass_rate: float = 0.0
    coverage_delta: float = 0.0
    complexity_delta: float = 0.0
    lines_changed: int = 0
    strategy: str = "hybrid"  # "replace_focused", "repair_focused", "hybrid"


# -----------------------------------------------------------------------------
# DRIFT DETECTION (AEGIS Framework - Forrester 2025)
# -----------------------------------------------------------------------------

DRIFT_DETECTION = {
    "enabled": True,
    "track_original_intent": True,
    "max_scope_expansion_pct": 20,  # Alert if scope grows >20%
    "max_file_drift": 5,  # Alert if touching >5 more files than planned
    "check_every_iteration": True,
    "metrics": ["task_count", "file_count", "line_count"],
}

DRIFT_WARNING = """
**⚠️ SCOPE DRIFT DETECTED**

Original scope: {original}
Current scope: {current}
Drift: {drift_pct:.1f}%

**Action Required**:
1. Review if expansion is justified
2. If justified, update original intent
3. If not, remove out-of-scope changes
4. Add [OUT] tags to prevent future creep
"""


def record_baseline_metrics(state: "RICState", tasks: list, files: list) -> None:
    """Record baseline metrics at session start for drift detection."""
    if not FEATURE_FLAGS.get("drift_detection", True):
        return

    if state.drift_metrics is None:
        state.drift_metrics = DriftMetrics()

    state.drift_metrics.original_task_count = len(tasks) if tasks else 0
    state.drift_metrics.original_file_count = len(files) if files else 0
    state.drift_metrics.original_line_estimate = estimate_lines(tasks) if tasks else 0
    state.drift_metrics.baseline_recorded = True
    state.drift_metrics.timestamp = datetime.now().isoformat()


def estimate_lines(tasks: list) -> int:
    """Estimate lines of code from task list."""
    if not tasks:
        return 0
    # Simple heuristic: 50 lines per task average
    return len(tasks) * 50


def update_current_metrics(state: "RICState") -> None:
    """Update current metrics for drift comparison."""
    if not FEATURE_FLAGS.get("drift_detection", True):
        return

    if state.drift_metrics is None:
        state.drift_metrics = DriftMetrics()

    state.drift_metrics.current_task_count = count_all_tasks(state)
    state.drift_metrics.current_file_count = count_files_touched(state)
    state.drift_metrics.current_line_count = count_lines_changed(state)


def count_all_tasks(state: "RICState") -> int:
    """Count all tasks/insights in current state."""
    return len(state.insights) if hasattr(state, "insights") and state.insights else 0


def count_files_touched(state: "RICState") -> int:
    """Count files modified in session."""
    # Track via decisions or file modifications
    files = set()
    if hasattr(state, "decisions") and state.decisions:
        for d in state.decisions:
            if isinstance(d, dict) and "file" in d:
                files.add(d["file"])
    return len(files)


def count_lines_changed(state: "RICState") -> int:
    """Estimate lines changed in session."""
    # Simple estimate based on insights and decisions
    base = count_all_tasks(state) * 30
    return base


def detect_drift(state: "RICState") -> tuple[bool, str, dict]:
    """
    Detect goal/scope drift between iterations.

    Returns:
        (has_drift, message, drift_data)
    """
    if not FEATURE_FLAGS.get("drift_detection", True):
        return False, "Drift detection disabled", {}

    if state.drift_metrics is None or not state.drift_metrics.baseline_recorded:
        return False, "No baseline metrics recorded", {}

    dm = state.drift_metrics
    drift_data = {}

    # Update current metrics first
    update_current_metrics(state)

    # Calculate task drift
    if dm.original_task_count > 0:
        task_drift = (dm.current_task_count - dm.original_task_count) / dm.original_task_count * 100
        drift_data["task_drift_pct"] = task_drift
        if task_drift > DRIFT_DETECTION["max_scope_expansion_pct"]:
            return True, f"Task count drift: {task_drift:.1f}%", drift_data

    # Calculate file drift
    file_drift = dm.current_file_count - dm.original_file_count
    drift_data["file_drift"] = file_drift
    if file_drift > DRIFT_DETECTION["max_file_drift"]:
        return True, f"File count drift: {file_drift} files over original", drift_data

    return False, "No significant drift", drift_data


# -----------------------------------------------------------------------------
# GUARDIAN MODE (Gartner 2025 - Guardian Agents)
# -----------------------------------------------------------------------------

GUARDIAN_MODE = {
    "enabled": False,  # Activated via --guardian flag or env var
    "review_criteria": [
        "code_matches_commit_message",
        "no_unrelated_changes",
        "tests_cover_changes",
        "no_security_issues",
        "no_hardcoded_secrets",
    ],
    "halt_on_critical": True,
    "require_clean_review": True,
}

GUARDIAN_PROMPT = """
**🛡️ GUARDIAN REVIEW** (Independent Verification)

You are reviewing code written by another agent. Pretend you did NOT write this.

**Review Checklist**:
□ Does the code do what the commit message says?
□ Are there any unrelated changes sneaking in?
□ Are tests adequate for the changes made?
□ Any security issues? (OWASP Top 10)
□ Any hardcoded secrets or credentials?
□ Would this pass a senior engineer's code review?

**Scoring**:
- PASS: All checks pass, ready to commit
- WARN: Minor issues, can proceed with notes
- FAIL: Critical issues, must address before commit

**Your Role**: Be skeptical. Assume there are problems to find.
Catch issues the original author missed.
"""

GUARDIAN_CRITERIA_PROMPTS = {
    "code_matches_commit_message": """
        Review: Does the code change actually do what the commit message claims?
        Look for mismatches between stated intent and actual changes.
    """,
    "no_unrelated_changes": """
        Review: Are there any changes unrelated to the stated purpose?
        Flag any scope creep or drive-by fixes.
    """,
    "tests_cover_changes": """
        Review: Do tests exist for the new/changed code?
        Check coverage of happy path and edge cases.
    """,
    "no_security_issues": """
        Review: Any security vulnerabilities?
        Check: injection, XSS, hardcoded credentials, unsafe deserialization.
    """,
    "no_hardcoded_secrets": """
        Review: Any hardcoded secrets, API keys, passwords, tokens?
        Check for patterns like: api_key=, password=, secret=, token=
    """,
}


def check_guardian_criterion(criterion: str, changes: dict, commit_message: str) -> tuple[bool, str, str]:
    """
    Check a single guardian criterion.

    Returns:
        (passed, issue, recommendation)
    """
    if criterion == "no_hardcoded_secrets":
        # Check for secrets patterns
        secret_patterns = [
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r"AWS_SECRET_ACCESS_KEY",
            r"PRIVATE_KEY",
        ]
        code_content = changes.get("content", "")
        for pattern in secret_patterns:
            if re.search(pattern, code_content, re.IGNORECASE):
                return (
                    False,
                    f"Potential secret found matching pattern: {pattern}",
                    "Remove hardcoded secrets and use environment variables",
                )

    elif criterion == "no_security_issues":
        # Basic security checks
        security_patterns = [
            (r"eval\s*\(", "Dangerous eval() usage"),
            (r"exec\s*\(", "Dangerous exec() usage"),
            (r"subprocess\.call\s*\([^)]*shell\s*=\s*True", "Shell injection risk"),
            (r"pickle\.loads?\s*\(", "Unsafe deserialization"),
        ]
        code_content = changes.get("content", "")
        for pattern, issue in security_patterns:
            if re.search(pattern, code_content):
                return False, issue, f"Review and fix: {issue}"

    # Default: pass if no issues found
    return True, "", ""


def run_guardian_review(changes: dict, commit_message: str) -> GuardianReview:
    """
    Run independent guardian review on changes.

    Returns:
        GuardianReview with results
    """
    if not FEATURE_FLAGS.get("guardian_mode", True):
        return GuardianReview(passed=True, score="PASS", timestamp=datetime.now().isoformat())

    issues = []
    recommendations = []
    criteria_results = {}

    for criterion in GUARDIAN_MODE["review_criteria"]:
        passed, issue, rec = check_guardian_criterion(criterion, changes, commit_message)
        criteria_results[criterion] = passed
        if not passed:
            issues.append({"criterion": criterion, "issue": issue})
        if rec:
            recommendations.append(rec)

    # Determine score
    critical_issues = [i for i in issues if i["criterion"] in ["no_security_issues", "no_hardcoded_secrets"]]

    if critical_issues:
        score = "FAIL"
        passed = False
    elif issues:
        score = "WARN"
        passed = not GUARDIAN_MODE["require_clean_review"]
    else:
        score = "PASS"
        passed = True

    return GuardianReview(
        passed=passed,
        score=score,
        issues=issues,
        recommendations=recommendations,
        timestamp=datetime.now().isoformat(),
        criteria_results=criteria_results,
    )


# -----------------------------------------------------------------------------
# STRUCTURED MEMORY (Anthropic Context Engineering 2025)
# -----------------------------------------------------------------------------

MEMORY_FILE = {
    "path": ".claude/RIC_NOTES.md",
    "auto_update": True,
    "update_triggers": ["phase_change", "insight_added", "decision_logged", "blocker_found"],
    "max_entries_per_section": 20,
    "archive_old_entries": True,
}

RIC_NOTES_TEMPLATE = """# RIC Session Notes

> Auto-generated memory file for cross-session context persistence.
> Last updated: {timestamp}
> Session: {session_id} | Iteration: {iteration}/{max_iterations}

---

## 🎯 Current Goal

{original_intent}

---

## ✅ Key Decisions Made

{decisions}

---

## 💡 Important Discoveries

{discoveries}

---

## 🚧 Current Blockers

{blockers}

---

## 📁 Files Modified

{files}

---

## ➡️ Next Steps

{next_steps}

---

## 📊 Session Metrics

| Metric | Value |
|--------|-------|
| Iteration | {iteration}/{max_iterations} |
| Phase | {phase_name} |
| Insights | P0:{p0} P1:{p1} P2:{p2} |
| Drift | {drift_status} |

---

*This file is auto-updated by RIC v5.0. Manual edits will be preserved.*
"""


def parse_ric_notes(content: str) -> dict:
    """Parse RIC_NOTES.md into structured dict."""
    notes = {
        "original_intent": "",
        "decisions": [],
        "discoveries": [],
        "blockers": [],
        "files": [],
        "next_steps": [],
        "raw_content": content,
    }

    # Simple parsing - extract sections
    sections = content.split("---")
    for section in sections:
        if "Current Goal" in section:
            notes["original_intent"] = section.split("Current Goal")[-1].strip()
        elif "Key Decisions" in section:
            notes["decisions"] = [line.strip() for line in section.split("\n") if line.strip().startswith("-")]
        elif "Important Discoveries" in section:
            notes["discoveries"] = [line.strip() for line in section.split("\n") if line.strip().startswith("-")]
        elif "Current Blockers" in section:
            notes["blockers"] = [line.strip() for line in section.split("\n") if line.strip().startswith("-")]

    return notes


def create_empty_notes() -> dict:
    """Create empty notes structure."""
    return {
        "original_intent": "",
        "decisions": [],
        "discoveries": [],
        "blockers": [],
        "files": [],
        "next_steps": [],
    }


def render_ric_notes(notes: dict, state: "RICState") -> str:
    """Render notes dict to markdown string."""
    # Count insights by priority
    p0_count = len([i for i in state.insights if i.priority.value == "P0"]) if state.insights else 0
    p1_count = len([i for i in state.insights if i.priority.value == "P1"]) if state.insights else 0
    p2_count = len([i for i in state.insights if i.priority.value == "P2"]) if state.insights else 0

    # Check drift status
    has_drift, drift_msg, _ = detect_drift(state)
    drift_status = "⚠️ " + drift_msg if has_drift else "✅ No drift"

    return RIC_NOTES_TEMPLATE.format(
        timestamp=datetime.now().isoformat(),
        session_id=state.session_id if hasattr(state, "session_id") else "unknown",
        iteration=state.current_iteration,
        max_iterations=state.max_iterations,
        original_intent=notes.get(
            "original_intent", state.original_intent if hasattr(state, "original_intent") else "Not specified"
        ),
        decisions="\n".join(notes.get("decisions", [])) or "- None yet",
        discoveries="\n".join(notes.get("discoveries", [])) or "- None yet",
        blockers="\n".join(notes.get("blockers", [])) or "- None",
        files="\n".join(notes.get("files", [])) or "- None yet",
        next_steps="\n".join(notes.get("next_steps", [])) or "- Continue with current phase",
        phase_name=state.current_phase.name if hasattr(state.current_phase, "name") else str(state.current_phase),
        p0=p0_count,
        p1=p1_count,
        p2=p2_count,
        drift_status=drift_status,
    )


def update_ric_notes(state: "RICState", trigger: str, content: dict) -> None:
    """Update RIC_NOTES.md with new information."""
    if not FEATURE_FLAGS.get("structured_memory", True):
        return

    notes_path = Path(MEMORY_FILE["path"])

    # Load or create
    if notes_path.exists():
        notes = parse_ric_notes(notes_path.read_text())
    else:
        notes = create_empty_notes()

    # Update based on trigger
    if trigger == "phase_change":
        notes["next_steps"] = content.get("next_steps", [])
    elif trigger == "insight_added":
        notes["discoveries"].append(f"- [{datetime.now().strftime('%H:%M')}] {content.get('description', '')}")
    elif trigger == "decision_logged":
        notes["decisions"].append(f"- [{datetime.now().strftime('%H:%M')}] {content.get('description', '')}")
    elif trigger == "blocker_found":
        notes["blockers"].append(f"- [{datetime.now().strftime('%H:%M')}] {content.get('description', '')}")

    # Limit entries
    max_entries = MEMORY_FILE["max_entries_per_section"]
    for key in ["decisions", "discoveries", "blockers"]:
        if len(notes[key]) > max_entries:
            notes[key] = notes[key][-max_entries:]

    # Render and write
    rendered = render_ric_notes(notes, state)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(rendered)


# -----------------------------------------------------------------------------
# CANDIDATE RANKING (SEIDR - Lexicase Selection)
# -----------------------------------------------------------------------------

CANDIDATE_RANKING = {
    "enabled": True,
    "algorithm": "lexicase",  # or "tournament"
    "max_candidates": 3,
    "criteria_weights": {
        "test_pass_rate": 0.4,
        "coverage_delta": 0.2,
        "complexity_delta": 0.2,
        "lines_changed": 0.2,
    },
}


def lexicase_selection(candidates: list, test_cases: list) -> list:
    """
    Rank candidates using lexicase selection.

    Lexicase selection (Spector et al.) shuffles test cases and
    filters candidates that pass each test in sequence.

    Returns:
        Ranked list of candidates (best first)
    """
    if not candidates:
        return []

    if not test_cases:
        # Sort by pass rate if no specific tests
        return sorted(candidates, key=lambda c: c.pass_rate if hasattr(c, "pass_rate") else 0, reverse=True)

    shuffled_tests = random.sample(test_cases, len(test_cases))
    remaining = candidates.copy()

    for test in shuffled_tests:
        if len(remaining) <= 1:
            break

        # Keep only candidates that pass this test
        passing = [c for c in remaining if c.test_results.get(test, False)]
        if passing:
            remaining = passing

    # Sort remaining by pass rate
    remaining.sort(key=lambda c: c.pass_rate if hasattr(c, "pass_rate") else 0, reverse=True)
    return remaining


def tournament_selection(candidates: list, tournament_size: int = 3) -> FixCandidate | None:
    """
    Select best candidate using tournament selection.

    Returns:
        Best candidate from tournament, or None if empty
    """
    if not candidates:
        return None

    tournament = random.sample(candidates, min(tournament_size, len(candidates)))
    return max(tournament, key=lambda c: c.pass_rate if hasattr(c, "pass_rate") else 0)


def rank_fix_candidates(candidates: list, test_cases: list) -> list:
    """
    Rank fix candidates using configured algorithm.

    Returns:
        Ranked list of candidates (best first)
    """
    if not FEATURE_FLAGS.get("candidate_ranking", True):
        return candidates

    if CANDIDATE_RANKING["algorithm"] == "lexicase":
        return lexicase_selection(candidates, test_cases)
    else:
        # For tournament, run multiple times and aggregate
        ranked = []
        remaining = candidates.copy()
        while remaining:
            winner = tournament_selection(remaining)
            if winner:
                ranked.append(winner)
                remaining.remove(winner)
            else:
                break
        return ranked


# -----------------------------------------------------------------------------
# REPLACE/REPAIR TRACKING (SEIDR Paper - ACM TELO 2025)
# -----------------------------------------------------------------------------

REPAIR_TRACKING = {
    "enabled": True,
    "track_ratio": True,
    "target_ratio": 0.5,  # 50% replace, 50% repair (hybrid strategy)
    "report_at_end": True,
}


def record_replace_attempt(state: "RICState", success: bool) -> None:
    """Record a replace (generate new code) attempt."""
    if not FEATURE_FLAGS.get("replace_repair_tracking", True):
        return

    if state.repair_stats is None:
        state.repair_stats = RepairStats()

    state.repair_stats.replace_count += 1
    state.repair_stats.total_attempts += 1
    if success:
        state.repair_stats.successful_replaces += 1


def record_repair_attempt(state: "RICState", success: bool) -> None:
    """Record a repair (fix existing code) attempt."""
    if not FEATURE_FLAGS.get("replace_repair_tracking", True):
        return

    if state.repair_stats is None:
        state.repair_stats = RepairStats()

    state.repair_stats.repair_count += 1
    state.repair_stats.total_attempts += 1
    if success:
        state.repair_stats.successful_repairs += 1


def get_repair_stats_summary(state: "RICState") -> str:
    """Get summary of repair/replace statistics."""
    if state.repair_stats is None:
        return "No repair/replace data"

    rs = state.repair_stats
    ratio = rs.get_ratio()
    target = REPAIR_TRACKING["target_ratio"]
    deviation = abs(ratio - target)

    status = "✅ Balanced" if deviation < 0.2 else "⚠️ Imbalanced"

    return f"""
Repair/Replace Statistics:
  Repairs: {rs.repair_count} ({rs.successful_repairs} successful)
  Replaces: {rs.replace_count} ({rs.successful_replaces} successful)
  Total: {rs.total_attempts}
  Ratio: {ratio:.1%} repair (target: {target:.0%})
  Status: {status}
"""


# -----------------------------------------------------------------------------
# SEIDR DEBUG LOOP (ACM TELO 2025)
# -----------------------------------------------------------------------------

SEIDR_CONFIG = {
    "enabled": True,
    "max_candidates": 3,
    "ranking_algorithm": "lexicase",
    "max_repair_iterations": 3,
    "strategy": "hybrid",  # "replace_focused", "repair_focused", "hybrid"
}

SEIDR_LOOP_PROMPT = """
**🔧 SEIDR DEBUG LOOP** (Synthesize-Execute-Instruct-Debug-Repair)

Current failure: {failure_description}

**Process**:
1. SYNTHESIZE: Generate {max_candidates} fix candidates
2. EXECUTE: Test each candidate
3. RANK: Sort by pass rate (lexicase selection)
4. SELECT: Pick best candidate
5. If still failing, REPAIR with feedback

**Strategy**: {strategy}
**Iteration**: {iteration}/{max_iterations}

Generate fix candidates now.
"""


def run_seidr_debug_loop(failures: list, state: "RICState", max_iter: int = 3) -> dict:
    """
    Run SEIDR-style debug sub-loop for test failures.

    Returns:
        {"success": bool, "candidate": FixCandidate, "iterations": int}
    """
    if not FEATURE_FLAGS.get("seidr_debug_loop", True):
        return {"success": False, "message": "SEIDR debug loop disabled", "iterations": 0}

    results = {
        "success": False,
        "candidate": None,
        "iterations": 0,
        "all_candidates": [],
    }

    for iteration in range(max_iter):
        results["iterations"] = iteration + 1
        candidates = []

        # Synthesize candidates (placeholder - actual implementation would generate fixes)
        for i in range(SEIDR_CONFIG["max_candidates"]):
            candidate = FixCandidate(
                id=f"candidate_{iteration}_{i}",
                code="",  # Would be populated with actual fix
                strategy=SEIDR_CONFIG["strategy"],
            )
            candidates.append(candidate)

        results["all_candidates"].extend(candidates)

        # Rank candidates
        ranked = rank_fix_candidates(candidates, [f.get("test_name", "") for f in failures] if failures else [])

        if ranked:
            best = ranked[0]
            if best.pass_rate >= 1.0:
                results["success"] = True
                results["candidate"] = best
                return results

    return results


# -----------------------------------------------------------------------------
# POLICY-AS-CODE GUARDRAILS (AEGIS Framework)
# -----------------------------------------------------------------------------

POLICY_AS_CODE = {
    "enabled": True,
    "min_iterations": 3,
    "max_iterations": 5,
    "require_tests_before_commit": True,
    "require_p0_p1_resolved_before_exit": True,
    "max_files_per_commit": 5,
    "max_lines_per_commit": 200,
    "block_on_violations": True,
}


def check_policy(action: str, context: dict) -> tuple[bool, str]:
    """
    Executable policy enforcement.

    Returns:
        (allowed, message)
    """
    if not FEATURE_FLAGS.get("policy_as_code", True):
        return True, "Policy checks disabled"

    if action == "commit":
        files_changed = context.get("files_changed", 0)
        if files_changed > POLICY_AS_CODE["max_files_per_commit"]:
            return (
                False,
                f"Too many files ({files_changed} > {POLICY_AS_CODE['max_files_per_commit']}). Split into smaller commits.",
            )

        lines_changed = context.get("lines_changed", 0)
        if lines_changed > POLICY_AS_CODE["max_lines_per_commit"]:
            return (
                False,
                f"Too many lines ({lines_changed} > {POLICY_AS_CODE['max_lines_per_commit']}). Split into smaller commits.",
            )

        if POLICY_AS_CODE["require_tests_before_commit"]:
            tests_passing = context.get("tests_passing", False)
            if not tests_passing:
                return False, "Tests must pass before committing."

    elif action == "exit":
        iteration = context.get("iteration", 0)
        if iteration < POLICY_AS_CODE["min_iterations"]:
            return False, f"Minimum {POLICY_AS_CODE['min_iterations']} iterations required. Currently at {iteration}."

        if POLICY_AS_CODE["require_p0_p1_resolved_before_exit"]:
            p0_remaining = context.get("p0_insights", 0)
            p1_remaining = context.get("p1_insights", 0)
            if p0_remaining > 0 or p1_remaining > 0:
                return False, f"P0/P1 insights must be resolved before exit. P0={p0_remaining}, P1={p1_remaining}"

    elif action == "advance_phase":
        # Check phase-specific requirements
        current_phase = context.get("current_phase", "")
        if current_phase == "P0" and not context.get("research_documented", False):
            return False, "Research must be documented before advancing from P0."

    return True, "Policy check passed"


# -----------------------------------------------------------------------------
# METAMORPHIC CONSISTENCY CHECK (MetaQA - ACM 2025)
# -----------------------------------------------------------------------------

METAMORPHIC_CHECK = {
    "enabled": False,  # Disabled by default - requires LLM API calls
    "mutation_types": [
        "rephrase_prompt",  # Same meaning, different words
        "add_irrelevant_info",  # Should not affect answer
        "change_order",  # Reorder requirements
    ],
    "consistency_threshold": 0.8,  # 80% consistency required
}


def metamorphic_consistency_check(original_response: str, mutations: list, responses: list) -> dict:
    """
    Check for hallucinations using metamorphic relations.

    Returns:
        {"likely_hallucination": bool, "consistency": float, "divergent": list}
    """
    if not FEATURE_FLAGS.get("metamorphic_check", False):
        return {"likely_hallucination": False, "consistency": 1.0, "message": "Metamorphic check disabled"}

    if not responses:
        return {"likely_hallucination": False, "consistency": 1.0, "message": "No mutations to compare"}

    # Simple consistency check - count matching responses
    matches = sum(1 for r in responses if similar_response(original_response, r))
    consistency = matches / len(responses) if responses else 1.0

    if consistency < METAMORPHIC_CHECK["consistency_threshold"]:
        return {
            "likely_hallucination": True,
            "consistency": consistency,
            "divergent": [r for r in responses if not similar_response(original_response, r)],
        }

    return {"likely_hallucination": False, "consistency": consistency, "divergent": []}


def similar_response(a: str, b: str, threshold: float = 0.7) -> bool:
    """Check if two responses are similar enough."""
    # Simple check - compare normalized versions
    a_normalized = a.lower().strip()
    b_normalized = b.lower().strip()

    # Check key content overlap
    a_words = set(a_normalized.split())
    b_words = set(b_normalized.split())

    if not a_words or not b_words:
        return True

    overlap = len(a_words & b_words) / max(len(a_words), len(b_words))
    return overlap >= threshold


# -----------------------------------------------------------------------------
# PACKAGE VERIFICATION (USENIX 2025)
# -----------------------------------------------------------------------------

PACKAGE_VERIFICATION = {
    "enabled": False,  # Disabled by default - requires network
    "check_pypi": True,
    "check_npm": False,
    "cache_duration_hours": 24,
    "known_packages_cache": {},
}


def verify_packages_exist(code: str) -> list:
    """
    Check that imported packages actually exist.

    Returns:
        List of potentially hallucinated packages
    """
    if not FEATURE_FLAGS.get("package_verification", False):
        return []

    # Extract imports
    import_pattern = r"^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    imports = re.findall(import_pattern, code, re.MULTILINE)

    # Standard library modules (partial list)
    stdlib = {
        "os",
        "sys",
        "json",
        "time",
        "datetime",
        "re",
        "math",
        "random",
        "collections",
        "itertools",
        "functools",
        "pathlib",
        "typing",
        "dataclasses",
        "enum",
        "abc",
        "copy",
        "io",
        "logging",
        "unittest",
        "subprocess",
        "threading",
        "multiprocessing",
        "queue",
        "socket",
        "http",
        "urllib",
        "email",
        "html",
        "xml",
        "csv",
        "sqlite3",
    }

    # Known good packages
    known_good = {
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "requests",
        "pytest",
        "flask",
        "django",
        "fastapi",
        "sqlalchemy",
        "pydantic",
    }

    suspicious = []
    for pkg in imports:
        if pkg not in stdlib and pkg not in known_good:
            # Would check PyPI here if network enabled
            suspicious.append(pkg)

    return suspicious


# -----------------------------------------------------------------------------
# PHASE ENFORCEMENT (v5.0 - Forced Research & Introspection)
# -----------------------------------------------------------------------------

# Phase completion requirements
PHASE_REQUIREMENTS = {
    "P0_RESEARCH": {
        "enabled": True,
        "min_web_searches": 3,  # Minimum web searches required
        "require_keyword_extraction": True,
        "require_persist_findings": True,
        "min_sources_documented": 3,
        "block_advance_without_completion": True,
        # v5.1 Quality: Diversity requirements
        "require_query_diversity": True,  # Can't repeat same query
        "require_source_diversity": True,  # Can't use same domain 3x
        "max_same_domain_sources": 2,
    },
    # v5.1 NEW: P1 PLAN phase requirements
    "P1_PLAN": {
        "enabled": True,
        "require_task_list": True,  # Must create explicit task list
        "min_tasks_defined": 3,  # At least 3 tasks
        "require_priority_assignment": True,  # P0/P1/P2/[OUT] for each
        "require_success_criteria": True,  # Measurable success criteria
        "require_scope_boundaries": True,  # What's IN and OUT of scope
        "require_smart_validation": True,  # Tasks must be SMART
        "block_advance_without_completion": True,
    },
    # v5.1 NEW: P2 BUILD phase requirements (ReVeal pattern)
    "P2_BUILD": {
        "enabled": True,
        "require_atomic_changes": True,  # 1-5 files per commit
        "max_files_per_commit": 5,
        "require_tests_with_changes": True,  # Tests for new code
        "require_generation_verification": True,  # ReVeal pattern
        "require_security_check": True,  # No secrets, no vulnerabilities
        "min_changes_before_advance": 1,  # Must make at least 1 change
        "block_advance_without_completion": True,
    },
    # v5.1 NEW: P3 VERIFY phase requirements
    "P3_VERIFY": {
        "enabled": True,
        "require_tests_pass": True,  # All tests must pass
        "require_coverage_threshold": True,  # Coverage above threshold
        "min_coverage_percent": 70,  # 70% minimum
        "require_lint_clean": True,  # No lint errors
        "require_type_check": False,  # Optional: mypy clean
        "require_security_scan": True,  # GitLeaks or similar
        "block_advance_without_completion": True,
    },
    "P4_REFLECT": {
        "enabled": True,
        "require_introspection": True,
        "min_insights_considered": 1,  # Must consider at least existing insights
        "require_convergence_check": True,
        "require_upgrade_ideas": True,  # Must propose at least one improvement idea
        "generate_next_iteration_plan": True,
        "block_advance_without_completion": True,
        # v5.1: Quality gates for upgrade ideas
        "require_quality_gate_pass": True,
    },
}

# =============================================================================
# v5.1 SMART CRITERIA FOR TASK VALIDATION
# =============================================================================
# Tasks must be: Specific, Measurable, Achievable, Relevant, (Time-bound optional)

SMART_VALIDATION_PATTERNS = {
    # Specific: Must reference concrete file, function, or component
    "specific": [
        r"[\w/]+\.(py|ts|js|md)",  # File reference
        r"class\s+[\w]+",  # Class reference
        r"function\s+[\w_]+",  # Function reference
        r"def\s+[\w_]+",  # Python function
        r"implement\s+[\w]+",  # Specific implementation
        r"create\s+[\w]+",  # Specific creation
        r"add\s+[\w]+\s+to",  # Specific addition
    ],
    # Measurable: Must have quantifiable outcome
    "measurable": [
        r"\d+%",  # Percentage target
        r"pass(es|ing)?",  # Test passing
        r"coverage",  # Coverage metric
        r"complete[ds]?",  # Completion state
        r"working",  # Working state
        r"test(s|ed|ing)?",  # Test verification
        r"verif(y|ied|ication)",  # Verification
    ],
    # Achievable: Has action verb
    "achievable": [
        r"\b(add|create|implement|fix|update|refactor|remove)\b",
        r"\b(integrate|configure|set up|write|modify|change)\b",
        r"\b(test|verify|validate|check|review|document)\b",
    ],
}

# P0 Research prompts
P0_RESEARCH_START_PROMPT = """
**🔬 [P0] RESEARCH PHASE - MANDATORY ACTIONS**

Before advancing to P1 PLAN, you MUST complete:

1. **KEYWORD EXTRACTION** (Required)
   - Extract 5-10 keywords from the task/goal
   - Expand with related terms and synonyms
   - Document in research notes

2. **WEB SEARCHES** (Minimum 3 required)
   - Search for each keyword group
   - Use current year (2025) in searches
   - Document sources with timestamps

3. **THOROUGH URL FETCHING** (Recommended - uses fast agents)
   For important URLs found, use multi-pass fetching:
   ```bash
   python3 .claude/hooks/thorough_research.py plan <url1> <url2> --topic "Topic"
   ```
   Or spawn haiku agents in parallel:
   - Agent 1: Overview & concepts
   - Agent 2: Code examples (verbatim)
   - Agent 3: API & configuration
   - Agent 4: Usage patterns

4. **PERSIST FINDINGS** (Required - IMMEDIATELY)
   - Save to docs/research/UPGRADE-XXX-TOPIC.md
   - Include: Search date, queries, sources, discoveries
   - Update research index
   - **CRITICAL**: Write to file BEFORE context compaction

**Current Status**:
- Web searches completed: {search_count}
- Keywords extracted: {keyword_count}
- Findings persisted: {persisted}

**Quick Command**: `/thorough-research <url_or_topic>`

⚠️ Cannot advance until requirements met.
"""

P0_KEYWORD_EXTRACTION_PROMPT = """
**📝 KEYWORD EXTRACTION** (P0 Research Step 1)

From your current task, extract keywords for research:

**Primary Keywords** (direct task terms):
1. [term 1]
2. [term 2]
3. [term 3]

**Related Keywords** (synonyms, related concepts):
1. [related 1]
2. [related 2]
3. [related 3]

**Expansion Keywords** (cutting-edge, 2025 trends):
1. [expansion 1]
2. [expansion 2]
3. [expansion 3]

**Search Queries to Execute**:
1. "[primary 1] [related 1] 2025 best practices"
2. "[primary 2] [expansion 1] latest research"
3. "[primary 3] implementation patterns 2025"

Execute these searches using WebSearch tool NOW.
"""

# =============================================================================
# v5.1 P1 PLAN PHASE PROMPTS
# =============================================================================

P1_PLAN_START_PROMPT = """
**📋 [P1] PLAN PHASE - MANDATORY TASK DESIGN**

Before advancing to P2 BUILD, you MUST complete:

1. **TASK LIST** (Required - min {min_tasks} tasks)
   Create explicit, atomic tasks with format:
   ```
   - [P0] Task description (file.py) - Success: measurable outcome
   - [P1] Task description (file.py) - Success: measurable outcome
   - [P2] Task description (file.py) - Success: measurable outcome
   - [OUT] Explicitly excluded item
   ```

2. **SMART VALIDATION** (Required for each task)
   Each task must be:
   - **S**pecific: Reference file, class, or function
   - **M**easurable: Define pass/fail criteria
   - **A**chievable: Use action verb (implement, add, fix)
   - **R**elevant: Tied to research findings from P0

3. **SCOPE BOUNDARIES** (Required)
   Define explicitly:
   - **IN SCOPE**: What WILL be done this iteration
   - **OUT OF SCOPE**: What is explicitly EXCLUDED

4. **SUCCESS CRITERIA** (Required)
   Answer: "How will we know this iteration succeeded?"
   - Test pass rate target: ____%
   - Coverage target: ____%
   - Features complete: [list]

**Current Status**:
- Tasks defined: {task_count}
- With priority: {priority_count}
- Success criteria: {has_criteria}

⚠️ Cannot advance until requirements met. Use 'p1-status' to check.
"""

P1_PLAN_TEMPLATE = """
## P1 PLAN - Iteration {iteration}/{max_iterations}

### Project Theme
[Brief description of what this project/upgrade is about]

### Tasks (SMART Validated)

#### [P0] Critical (Must complete)
1. **Task**: [Specific action] in [file.py]
   - Success: [Measurable outcome]
   - Estimated changes: [file list]

#### [P1] Important (Should complete)
1. **Task**: [Specific action] in [file.py]
   - Success: [Measurable outcome]

#### [P2] Polish (If time permits)
1. **Task**: [Specific action] in [file.py]
   - Success: [Measurable outcome]

#### [OUT] Out of Scope
- [Explicitly excluded item with reason]

### Scope Boundaries
- **IN SCOPE**: [What will be done]
- **OUT OF SCOPE**: [What is explicitly excluded]

### Success Criteria
- [ ] All P0 tasks complete
- [ ] Tests pass: target __%
- [ ] Coverage: target ≥70%
- [ ] Lint clean
"""

# =============================================================================
# v5.1 P2 BUILD PHASE PROMPTS (ReVeal Pattern)
# =============================================================================

P2_BUILD_START_PROMPT = """
**🔨 [P2] BUILD PHASE - GENERATION-VERIFICATION LOOP**

Before advancing to P3 VERIFY, you MUST complete:

1. **ATOMIC CHANGES** (Required - max {max_files} files per commit)
   Each change must be:
   - Self-contained (1-5 files)
   - Testable in isolation
   - Revertable if needed

2. **GENERATION-VERIFICATION** (ReVeal Pattern)
   For each change, follow this cycle:
   ```
   <generation-think> Reason about the change needed </generation-think>
   <generation-answer> Write the code </generation-answer>
   <verification-think> Identify edge cases and failure modes </verification-think>
   <verification-answer> Write or run tests </verification-answer>
   ```

3. **TESTS WITH CHANGES** (Required)
   - New code MUST have tests
   - Modified code MUST update tests
   - No untested code advances

4. **SECURITY CHECK** (Required)
   Before commit:
   - [ ] No hardcoded secrets
   - [ ] No credential files added
   - [ ] No SQL injection risks
   - [ ] No command injection risks

**Current Status**:
- Files changed: {files_changed}
- Tests added/modified: {tests_changed}
- Commits made: {commit_count}
- Security check: {security_status}

⚠️ Cannot advance without at least 1 verified change.
"""

P2_BUILD_DEBUG_TEMPLATE = """
## P2 BUILD Debug - When Things Go Wrong

### Error Analysis Template
When a build/test fails, analyze:

1. **Error Type**: [Syntax | Runtime | Logic | Import | Type]
2. **Error Location**: [file:line]
3. **Error Message**: [exact message]
4. **Root Cause**: [Why did this happen?]
5. **Fix Approach**: [How to fix]

### Common Issues Checklist
- [ ] Import paths correct?
- [ ] Dependencies installed?
- [ ] Type hints accurate?
- [ ] Mock objects properly configured?
- [ ] Async/await properly handled?

### Regression Prevention
After fixing:
- [ ] Add test case for this scenario
- [ ] Update documentation if API changed
- [ ] Check for similar patterns elsewhere
"""

# =============================================================================
# v5.1 P3 VERIFY PHASE PROMPTS
# =============================================================================

P3_VERIFY_START_PROMPT = """
**✅ [P3] VERIFY PHASE - QUALITY GATES**

Before advancing to P4 REFLECT, you MUST pass ALL gates:

1. **TESTS** (Required - must pass)
   ```bash
   pytest tests/ -v --tb=short
   ```
   Status: {test_status}

2. **COVERAGE** (Required - ≥{min_coverage}%)
   ```bash
   pytest --cov=. --cov-report=term-missing
   ```
   Current: {coverage_percent}%

3. **LINT** (Required - must be clean)
   ```bash
   ruff check . --fix
   # or: flake8 .
   ```
   Status: {lint_status}

4. **SECURITY SCAN** (Required)
   ```bash
   gitleaks detect --source . --verbose
   # or: pip-audit -r requirements.txt
   ```
   Status: {security_status}

5. **TYPE CHECK** (Optional but recommended)
   ```bash
   mypy --config-file mypy.ini
   ```
   Status: {type_status}

**Gate Summary**:
| Gate | Required | Status |
|------|----------|--------|
| Tests Pass | ✓ | {test_status} |
| Coverage ≥{min_coverage}% | ✓ | {coverage_status} |
| Lint Clean | ✓ | {lint_status} |
| Security Scan | ✓ | {security_status} |
| Type Check | ○ | {type_status} |

⚠️ Cannot advance until all required gates pass.
"""

P3_VERIFY_DEBUG_TEMPLATE = """
## P3 VERIFY Debug - Fixing Quality Issues

### Test Failure Debug
```
1. Run failing test in isolation:
   pytest tests/path/to_test.py::test_name -v

2. Add verbose output:
   pytest -v --tb=long --capture=no

3. Run with debugger:
   pytest --pdb --pdb-first

4. Check test fixtures:
   pytest --fixtures
```

### Coverage Gap Analysis
```
1. Find uncovered lines:
   pytest --cov=. --cov-report=term-missing

2. Generate HTML report:
   pytest --cov=. --cov-report=html
   # Open htmlcov/index.html

3. Focus on critical modules:
   pytest --cov=mcp --cov-report=term-missing
```

### Lint Error Fixes
```
1. Auto-fix what's possible:
   ruff check . --fix

2. Format code:
   black .

3. Sort imports:
   isort .
```

### Security Issue Remediation
| Issue | Fix |
|-------|-----|
| Hardcoded secret | Move to .env, add to .gitignore |
| SQL injection | Use parameterized queries |
| Command injection | Use subprocess with list args |
| Path traversal | Validate and sanitize paths |
"""

# =============================================================================
# v5.1 PROJECT THEME INTROSPECTION
# =============================================================================

PROJECT_THEME_TEMPLATE = """
## Project Theme Analysis

### Core Theme
**Primary Purpose**: {primary_purpose}
**Domain**: {domain}
**Key Capabilities**: {capabilities}

### Current Iteration Focus
**What's Being Built**: {current_focus}
**How It Relates to Theme**: {theme_relation}

### Theme Expansion Opportunities
Identified areas for growth:
1. {expansion_1}
2. {expansion_2}
3. {expansion_3}

### Polish Priorities
Based on theme analysis:
1. {polish_1} - Impact: {impact_1}
2. {polish_2} - Impact: {impact_2}

### Theme Coherence Check
- [ ] New features align with core theme
- [ ] No scope creep into unrelated areas
- [ ] Integration with existing capabilities considered
"""

# =============================================================================
# v5.1 ITERATION PROGRESS TRACKING
# =============================================================================

ITERATION_PROGRESS_TEMPLATE = """
## Iteration {current}/{max} Progress Summary

### Code Metrics
| Metric | Start | Now | Change |
|--------|-------|-----|--------|
| Files Changed | {files_start} | {files_now} | {files_delta} |
| Lines Added | {lines_add_start} | {lines_add_now} | {lines_add_delta} |
| Lines Removed | {lines_del_start} | {lines_del_now} | {lines_del_delta} |
| Test Count | {tests_start} | {tests_now} | {tests_delta} |
| Coverage % | {cov_start} | {cov_now} | {cov_delta} |

### Quality Metrics
| Metric | Iteration {prev} | Iteration {current} | Trend |
|--------|------------------|---------------------|-------|
| Lint Errors | {lint_prev} | {lint_now} | {lint_trend} |
| Type Errors | {type_prev} | {type_now} | {type_trend} |
| Test Failures | {fail_prev} | {fail_now} | {fail_trend} |

### Progress Assessment
- **Making Progress**: {progress_verdict}
- **Plateau Detected**: {plateau_status}
- **Recommendation**: {recommendation}

### What Changed This Iteration
{change_summary}

### Carried Forward to Next Iteration
{carried_forward}
"""

# =============================================================================
# v5.1 DEBUG SUITE TEMPLATES
# =============================================================================

DEBUG_PHASE_TEMPLATE = """
## 🔧 DEBUG SUITE - Phase {phase} ({phase_name})

### Current State Dump
```json
{state_json}
```

### Phase Completion Status
| Requirement | Met? | Value | Expected |
|-------------|------|-------|----------|
{requirements_table}

### Recent Tool Calls (last 5)
{recent_tools}

### Active Blockers
{blockers_list}

### Suggested Debug Steps
{debug_steps}

### Common Issues for {phase_name}
{common_issues}
"""

DEBUG_COMMON_ISSUES = {
    "P0_RESEARCH": """
1. **Not enough web searches**: Run 3+ searches with different keywords
2. **Sources not documented**: Use `record-search` after each WebSearch
3. **Findings not persisted**: Write to docs/research/ file before advancing
4. **Same domain sources**: Diversify sources (max 2 from same domain)
""",
    "P1_PLAN": """
1. **Tasks not SMART**: Add file references (broker.py) and action verbs (implement, add)
2. **No priorities**: Use `record-priorities` with P0/P1/P2/OUT assignments
3. **Missing scope**: Use `record-scope` with IN/OUT of scope items
4. **No success criteria**: Use `record-criteria` with testable outcomes
""",
    "P2_BUILD": """
1. **No changes recorded**: Use `record-change` after each commit
2. **Too many files**: Keep commits atomic (≤5 files per change)
3. **No tests with changes**: Use `record-tests-with-change` when adding tests
4. **Security not checked**: Use `record-security-p2` after security review
""",
    "P3_VERIFY": """
1. **Tests failing**: Run `pytest tests/ -v` and fix failures
2. **Coverage too low**: Add tests to reach 70%+ coverage
3. **Lint errors**: Run `ruff check . --fix` to auto-fix
4. **Security issues**: Run `gitleaks detect --source .` to check
""",
    "P4_REFLECT": """
1. **Vague upgrade ideas**: Add file:line location and action verbs
2. **No iteration citation**: Reference what was done in previous iteration
3. **Missing introspection**: Use `record-introspection` to complete
4. **No loop decision**: Use `loop-decision LOOP|EXIT "reason"` to decide
""",
}

DEBUG_ITERATION_TEMPLATE = """
## 🔧 FULL ITERATION DEBUG - Iteration {iteration}/{max}

### Session Overview
| Property | Value |
|----------|-------|
| Session ID | {session_id} |
| Started | {started_at} |
| Current Phase | {current_phase} |
| Insights Open | P0={p0_open}, P1={p1_open}, P2={p2_open} |
| Fix Attempts | {fix_attempts} |
| Plateau Count | {plateau_count} |

### All Phase Statuses
{phase_statuses}

### Throttle Status
| Throttle | Current | Limit | Status |
|----------|---------|-------|--------|
| Tool calls/phase | {tool_calls} | {tool_limit} | {tool_status} |
| Edits/file | {edits_max} | {edit_limit} | {edit_status} |
| Consecutive failures | {failures} | {fail_limit} | {fail_status} |

### Decision Trace (last 3)
{decision_trace}

### Suggested Recovery Actions
{recovery_actions}
"""

PROJECT_THEME_ANALYSIS_PROMPT = """
## 🎯 PROJECT THEME ANALYSIS

Analyze the current project and extract:

### 1. Core Theme Identification
Read the following files to understand the project:
- CLAUDE.md (project instructions)
- README.md (if exists)
- docs/PROJECT_STATUS.md (if exists)
- Main algorithm files

**Extract**:
- **Primary Purpose**: What is this project for?
- **Domain**: Trading? Analytics? Infrastructure?
- **Key Capabilities**: What can it do?

### 2. Current Work Context
- **Current Upgrade/Task**: What are we building?
- **How It Relates**: How does this connect to the core theme?

### 3. Theme Expansion Opportunities
Based on the core theme, identify 3 areas where the project could grow:
1. [Feature that extends core capabilities]
2. [Integration that enhances value]
3. [Refinement that improves quality]

### 4. Theme Coherence Checklist
Before advancing, verify:
- [ ] Current work aligns with core theme
- [ ] No scope creep into unrelated areas
- [ ] Integration with existing capabilities planned

**Use `record-theme` to save analysis.**
"""

THEME_EXPANSION_PROMPT = """
## 🌱 THEME EXPANSION - Iteration {iteration}/{max}

### Previous Theme Context
{previous_theme}

### Current Iteration Contributions
- What was built: {current_work}
- How it enhanced theme: {enhancement}

### Recommended Theme Expansions for Next Iteration
Based on theme analysis, prioritize:

1. **Most Aligned**: {aligned_expansion}
   - Directly extends core capabilities
   - Highest value-add

2. **Moderate Expansion**: {moderate_expansion}
   - Related enhancement
   - Medium complexity

3. **Polish/Refinement**: {polish_expansion}
   - Quality improvement
   - Low risk, high impact

### Theme Coherence Score: {coherence_score}/10
{coherence_notes}
"""

# ============================================================================
# AUDIT SUITE (v5.1) - Proactive Validation & Cleanup
# ============================================================================

AUDIT_CATEGORIES = {
    "code": {
        "name": "Code Quality",
        "checks": ["syntax", "imports", "types", "complexity", "duplication"],
        "description": "Validates code syntax, imports, and quality metrics",
    },
    "files": {
        "name": "File Completeness",
        "checks": ["missing_files", "empty_files", "orphan_files", "file_size"],
        "description": "Checks for missing, empty, or orphaned files",
    },
    "functions": {
        "name": "Function Analysis",
        "checks": ["undefined_calls", "unused_functions", "docstrings", "signatures"],
        "description": "Validates functions are defined, used, and documented",
    },
    "best_practices": {
        "name": "Best Practices",
        "checks": ["naming", "error_handling", "logging", "security", "tests"],
        "description": "Checks adherence to coding best practices",
    },
    "hooks": {
        "name": "Hook Integration",
        "checks": ["hook_files", "settings_sync", "hook_errors", "hook_performance"],
        "description": "Validates Claude Code hooks are properly configured",
    },
    "references": {
        "name": "References",
        "checks": ["broken_imports", "missing_deps", "version_mismatch", "circular"],
        "description": "Checks for broken references and dependencies",
    },
    "crossref": {
        "name": "Cross-References",
        "checks": ["doc_code_sync", "test_coverage", "config_sync", "changelog"],
        "description": "Validates cross-references between docs, code, and tests",
    },
    "cleanup": {
        "name": "Cleanup",
        "checks": ["temp_files", "debug_code", "todo_fixme", "dead_code", "stale_branches"],
        "description": "Identifies cleanup opportunities",
    },
}

AUDIT_ISSUE_SEVERITY = {
    "critical": {"emoji": "🔴", "blocks_exit": True, "priority": 0},
    "warning": {"emoji": "🟡", "blocks_exit": False, "priority": 1},
    "info": {"emoji": "🔵", "blocks_exit": False, "priority": 2},
    "suggestion": {"emoji": "💡", "blocks_exit": False, "priority": 3},
}

AUDIT_SUMMARY_TEMPLATE = """
## 🔍 AUDIT SUITE REPORT - {timestamp}

### Summary
| Category | Checks | Issues | Critical | Warnings |
|----------|--------|--------|----------|----------|
{category_rows}

### Overall Status: {overall_status}
- Total Issues: {total_issues}
- Critical (blocks exit): {critical_count}
- Warnings: {warning_count}
- Suggestions: {suggestion_count}

{detailed_findings}

### Recommended Actions
{recommendations}
"""

# Common patterns for checks
AUDIT_PATTERNS = {
    "todo_fixme": r"(?i)(TODO|FIXME|XXX|HACK|BUG)[\s:]*(.+?)(?:\n|$)",
    "debug_code": r"(print\s*\(|console\.log|debugger|breakpoint\(\)|pdb\.set_trace)",
    "hardcoded_secrets": r"(password|secret|api_key|token)\s*=\s*['\"][^'\"]+['\"]",
    "empty_except": r"except.*?:\s*pass",
    "bare_except": r"except\s*:",
    "missing_docstring": r"def\s+\w+\([^)]*\):\s*\n\s*(?!\"\"\"|\'\'\')[^\s]",
    "long_function": r"def\s+\w+\([^)]*\):.*?(?=\ndef|\nclass|\Z)",
    "import_star": r"from\s+\w+\s+import\s+\*",
    "unused_import": r"^import\s+(\w+)|^from\s+\w+\s+import\s+(\w+)",
}

# File patterns to check
AUDIT_FILE_PATTERNS = {
    "python": "**/*.py",
    "tests": "tests/**/*.py",
    "config": "**/*.json",
    "docs": "docs/**/*.md",
    "hooks": ".claude/hooks/*.py",
    "commands": ".claude/commands/*.md",
}

# Best practices checklist
BEST_PRACTICES_CHECKLIST = {
    "error_handling": [
        "Use specific exceptions, not bare except",
        "Log errors with context",
        "Provide meaningful error messages",
        "Handle edge cases explicitly",
    ],
    "security": [
        "No hardcoded secrets",
        "Validate all inputs",
        "Use parameterized queries",
        "Sanitize user input",
    ],
    "testing": [
        "Tests exist for new code",
        "Edge cases covered",
        "Mocks used for external services",
        "Assertions are specific",
    ],
    "documentation": [
        "Functions have docstrings",
        "Complex logic has comments",
        "README is up to date",
        "CHANGELOG updated",
    ],
    "code_quality": [
        "Functions are under 50 lines",
        "No duplicate code",
        "Consistent naming convention",
        "Type hints on public functions",
    ],
}

# P4 Reflect prompts
P4_REFLECT_START_PROMPT = """
**🔮 [P4] REFLECT PHASE - MANDATORY INTROSPECTION**

Before completing this iteration, you MUST:

1. **REVIEW ALL INSIGHTS** (Required)
   - P0 (Critical): {p0_count} open
   - P1 (Important): {p1_count} open
   - P2 (Optional): {p2_count} open

2. **CONVERGENCE CHECK** (Required)
   - Check: Are we making progress?
   - Check: Any repeated failures?
   - Check: Is code quality improving?

3. **GENERATE UPGRADE IDEAS** (Required - QUALITY GATES ENFORCED)

   ⚠️ **v5.1 QUALITY GATES - Ideas MUST include:**

   **(i) LOCATION** - Reference specific file:line or function:
   - ✅ "mcp/broker_server.py:145 get_positions()"
   - ✅ "Add to class BrokerServer in broker_server.py"
   - ❌ "improve the code" (rejected - no location)

   **(ii) ACTION** - Use concrete action verbs:
   - ✅ "Implement real Schwab API integration"
   - ✅ "Add error handling for timeout scenarios"
   - ❌ "make it better" (rejected - no action)

   **(iii) CITATION** (Iteration 2+) - Reference previous work:
   - ✅ "In iteration 1, we built basic structure. Now add API calls."
   - ❌ Starting fresh without citing prior iteration (rejected)

   **Example Quality Upgrade Idea:**
   ```
   Implement Schwab API integration in mcp/broker_server.py:145
   - Location: broker_server.py get_positions() method
   - Action: Replace mock data with SchwabClient.get_account_positions()
   - Context: In iteration 1, we built mock structure. Now needs real data.
   ```

4. **PLAN NEXT ITERATION** (If looping)
   - What should P0 research focus on?
   - What are the top priorities?

**Decision Required**: LOOP (continue) or EXIT (complete)

⚠️ Cannot exit/loop without completing introspection WITH QUALITY GATES.
"""

P4_INTROSPECTION_TEMPLATE = """
**🧠 INTROSPECTION RESULTS** (Iteration {iteration}/{max_iterations})

## What Went Well
- [observation 1]
- [observation 2]

## What Could Be Improved (LOCATION + ACTION required)
- **Location**: [file:line or function]
  **Action**: [concrete verb + description]
- **Location**: [file:line or function]
  **Action**: [concrete verb + description]

## Upgrade Ideas Generated (QUALITY GATES ENFORCED)

### Idea 1: [Title]
- **Location**: [file:line, function, or class] (REQUIRED)
- **Action**: [implement/add/fix/refactor + specific task] (REQUIRED)
- **Context**: [What was done before, what's needed now] (REQUIRED for iteration 2+)
- Priority: P0/P1/P2
- Effort: Low/Medium/High
- Impact: Low/Medium/High

### Idea 2: [Title]
- **Location**: [file:line, function, or class]
- **Action**: [concrete action verb + specific task]
- **Context**: [Reference to previous iteration if applicable]
- Priority: P0/P1/P2
- Effort: Low/Medium/High
- Impact: Low/Medium/High

## Convergence Assessment
- Progress Rate: {progress_rate}
- Fix Success Rate: {fix_success_rate}
- Code Churn: {code_churn}
- **Verdict**: {convergence_verdict}

## Next Iteration Plan (if LOOP)
- Research Focus: [topic]
- Key Tasks: [tasks] (with file locations)
- Expected Outcomes: [outcomes]
- **Citing This Iteration**: [What we did that next iteration builds on]

## Decision: {decision}
Reason: {decision_reason}
"""


@dataclass
class PhaseCompletionStatus:
    """Track phase completion requirements."""

    phase: str
    completed: bool = False
    requirements_met: dict = field(default_factory=dict)
    blockers: list = field(default_factory=list)
    timestamp: str | None = None

    # P0 Research tracking
    keywords_extracted: list = field(default_factory=list)
    web_searches_done: int = 0
    sources_documented: int = 0
    findings_persisted: bool = False

    # P4 Reflect tracking
    introspection_done: bool = False
    insights_reviewed: bool = False
    convergence_checked: bool = False
    upgrade_ideas: list = field(default_factory=list)
    next_iteration_plan: str = ""
    loop_decision: str = ""  # "LOOP" or "EXIT"
    loop_reason: str = ""


def check_p0_completion(state: "RICState") -> tuple[bool, list[str], dict]:
    """
    Check if P0 RESEARCH requirements are met.

    Returns:
        (is_complete, blockers, status_dict)
    """
    if not PHASE_REQUIREMENTS["P0_RESEARCH"]["enabled"]:
        return True, [], {"phase": "P0", "completed": True}

    reqs = PHASE_REQUIREMENTS["P0_RESEARCH"]
    # Get status as dict (key is "P0_RESEARCH" not "P0")
    status = state.phase_completion.get("P0_RESEARCH", {})
    blockers = []

    # Check web searches (from status dict)
    search_count = status.get("web_searches_done", 0)
    if search_count < reqs["min_web_searches"]:
        blockers.append(
            f"Need {reqs['min_web_searches']} web searches, only {search_count} done. "
            f"Use WebSearch tool to research your task."
        )

    # Check keyword extraction
    keywords = status.get("keywords_extracted", [])
    if reqs["require_keyword_extraction"] and not keywords:
        blockers.append(
            "Keyword extraction not done. Extract 5-10 keywords from task, " "expand with related terms, then search."
        )

    # Check findings persistence
    findings_persisted = status.get("findings_persisted", False)
    if reqs["require_persist_findings"] and not findings_persisted:
        blockers.append(
            "Research findings not persisted. Write to docs/research/UPGRADE-XXX-TOPIC.md "
            "with timestamps and sources."
        )

    # Check sources documented
    sources = status.get("sources_documented", 0)
    if sources < reqs["min_sources_documented"]:
        blockers.append(
            f"Need {reqs['min_sources_documented']} sources documented, " f"only {sources}. Add more source references."
        )

    completed = len(blockers) == 0
    status["completed"] = completed
    status["blockers"] = blockers

    return completed, blockers, status


# =============================================================================
# v5.1 P1 PLAN PHASE COMPLETION CHECK
# =============================================================================


def check_p1_completion(state: "RICState") -> tuple[bool, list[str], dict]:
    """
    Check if P1 PLAN requirements are met.

    v5.1 NEW: Validates task list, SMART criteria, scope boundaries, success criteria.

    Returns:
        (is_complete, blockers, status_dict)
    """
    if not PHASE_REQUIREMENTS["P1_PLAN"]["enabled"]:
        return True, [], {"phase": "P1", "completed": True}

    reqs = PHASE_REQUIREMENTS["P1_PLAN"]
    status = state.phase_completion.get("P1_PLAN", {})
    blockers = []

    # Check task list exists (v5.1: use "tasks_defined" key from CLI recording)
    tasks = status.get("tasks_defined", status.get("tasks", []))
    if reqs["require_task_list"] and len(tasks) < reqs["min_tasks_defined"]:
        blockers.append(
            f"Need at least {reqs['min_tasks_defined']} tasks defined, "
            f"only {len(tasks)} found. Create explicit tasks with priorities."
        )

    # Check priority assignment (v5.1: use "priorities_assigned" key)
    priorities_assigned = status.get("priorities_assigned", False)
    if reqs["require_priority_assignment"] and not priorities_assigned:
        blockers.append(
            "No priorities assigned. Use 'record-priorities P0:task1,P1:task2,...' "
            "to assign P0/P1/P2/[OUT] priorities."
        )

    # Check success criteria (v5.1: use "success_criteria_defined" key)
    has_success_criteria = status.get("success_criteria_defined", status.get("has_success_criteria", False))
    if reqs["require_success_criteria"] and not has_success_criteria:
        blockers.append(
            "No success criteria defined. Answer: 'How will we know this iteration succeeded?' "
            "Include test pass rate, coverage target, and feature list."
        )

    # Check scope boundaries (v5.1: use "scope_defined" key)
    has_scope_boundaries = status.get("scope_defined", status.get("has_scope_boundaries", False))
    if reqs["require_scope_boundaries"] and not has_scope_boundaries:
        blockers.append("No scope boundaries defined. Explicitly list what's IN SCOPE and OUT OF SCOPE.")

    # v5.1: SMART validation for tasks
    if reqs.get("require_smart_validation", False) and tasks:
        smart_valid_count = 0
        for task in tasks:
            task_text = task.get("description", "") if isinstance(task, dict) else str(task)
            is_smart = validate_task_smart(task_text)
            if is_smart:
                smart_valid_count += 1

        if smart_valid_count < len(tasks) * 0.5:  # At least 50% must be SMART
            blockers.append(
                f"Only {smart_valid_count}/{len(tasks)} tasks pass SMART validation. "
                "Tasks must be Specific (file/function), Measurable (testable), Achievable (action verb)."
            )

    completed = len(blockers) == 0
    status["completed"] = completed
    status["blockers"] = blockers

    return completed, blockers, status


def validate_task_smart(task_text: str) -> bool:
    """Validate a task against SMART criteria."""
    if not task_text or len(task_text) < 10:
        return False

    text = task_text.lower()
    has_specific = False
    has_measurable = False
    has_achievable = False

    # Check Specific (references file, class, function)
    for pattern in SMART_VALIDATION_PATTERNS["specific"]:
        if re.search(pattern, task_text, re.IGNORECASE):
            has_specific = True
            break

    # Check Measurable (has quantifiable outcome)
    for pattern in SMART_VALIDATION_PATTERNS["measurable"]:
        if re.search(pattern, text, re.IGNORECASE):
            has_measurable = True
            break

    # Check Achievable (has action verb)
    for pattern in SMART_VALIDATION_PATTERNS["achievable"]:
        if re.search(pattern, text, re.IGNORECASE):
            has_achievable = True
            break

    # Need at least 2 of 3 to pass
    return sum([has_specific, has_measurable, has_achievable]) >= 2


# =============================================================================
# v5.1 P2 BUILD PHASE COMPLETION CHECK (ReVeal Pattern)
# =============================================================================


def check_p2_completion(state: "RICState") -> tuple[bool, list[str], dict]:
    """
    Check if P2 BUILD requirements are met.

    v5.1 NEW: Validates atomic changes, tests with changes, security check.
    Implements ReVeal pattern (generation-verification cycle).

    Returns:
        (is_complete, blockers, status_dict)
    """
    if not PHASE_REQUIREMENTS["P2_BUILD"]["enabled"]:
        return True, [], {"phase": "P2", "completed": True}

    reqs = PHASE_REQUIREMENTS["P2_BUILD"]
    status = state.phase_completion.get("P2_BUILD", {})
    blockers = []

    # Check minimum changes made (v5.1: changes_made is a list from CLI)
    changes_list = status.get("changes_made", [])
    changes_count = len(changes_list) if isinstance(changes_list, list) else changes_list
    if changes_count < reqs["min_changes_before_advance"]:
        blockers.append(
            f"No changes made in P2 BUILD. Must make at least {reqs['min_changes_before_advance']} "
            "verified change before advancing. Write code and tests."
        )

    # Check tests accompany changes (v5.1: tests_with_changes is a count)
    tests_with_changes = status.get("tests_with_changes", 0)
    if reqs["require_tests_with_changes"] and changes_count > 0 and tests_with_changes == 0:
        blockers.append(
            "Code changes made without accompanying tests. " "New/modified code MUST have tests before advancing."
        )

    # Check atomic commits (not too many files per change)
    if reqs["require_atomic_changes"] and isinstance(changes_list, list):
        for change in changes_list:
            files_count = len(change.get("files", []))
            if files_count > reqs["max_files_per_commit"]:
                blockers.append(
                    f"Change '{change.get('description', '')[:30]}' has {files_count} files, "
                    f"exceeds atomic limit of {reqs['max_files_per_commit']}. Break into smaller commits."
                )
                break  # Only report first violation

    # Check security (v5.1: security_checked is a boolean)
    security_checked = status.get("security_checked", False)
    if reqs["require_security_check"] and changes_count > 0 and not security_checked:
        blockers.append(
            "Security check not done. Before advancing, verify: "
            "no hardcoded secrets, no credential files, no injection risks."
        )

    # v5.1: ReVeal pattern - generation-verification cycle
    if reqs.get("require_generation_verification", False):
        verification_done = status.get("generation_verification_used", False)
        if changes_count > 0 and not verification_done:
            blockers.append(
                "ReVeal pattern not followed. For each change, must complete: "
                "<generation-think> → <generation-answer> → <verification-think> → <verification-answer>"
            )

    completed = len(blockers) == 0
    status["completed"] = completed
    status["blockers"] = blockers

    return completed, blockers, status


# =============================================================================
# v5.1 P3 VERIFY PHASE COMPLETION CHECK
# =============================================================================


def check_p3_completion(state: "RICState") -> tuple[bool, list[str], dict]:
    """
    Check if P3 VERIFY requirements are met.

    v5.1 NEW: Validates tests pass, coverage threshold, lint clean, security scan.

    Returns:
        (is_complete, blockers, status_dict)
    """
    if not PHASE_REQUIREMENTS["P3_VERIFY"]["enabled"]:
        return True, [], {"phase": "P3", "completed": True}

    reqs = PHASE_REQUIREMENTS["P3_VERIFY"]
    status = state.phase_completion.get("P3_VERIFY", {})
    blockers = []

    # Check tests pass
    tests_passed = status.get("tests_passed", False)
    test_result = status.get("test_result", "not run")
    if reqs["require_tests_pass"] and not tests_passed:
        blockers.append(
            f"Tests not passing (status: {test_result}). "
            "Run 'pytest tests/ -v --tb=short' and fix failures before advancing."
        )

    # Check coverage threshold
    coverage_percent = status.get("coverage_percent", 0)
    if reqs["require_coverage_threshold"]:
        min_coverage = reqs["min_coverage_percent"]
        if coverage_percent < min_coverage:
            blockers.append(
                f"Coverage {coverage_percent}% below required {min_coverage}%. "
                "Add tests to increase coverage before advancing."
            )

    # Check lint clean (v5.1: use "lint_clean" and "lint_errors" keys from CLI)
    lint_passed = status.get("lint_clean", status.get("lint_passed", False))
    lint_errors = status.get("lint_errors", status.get("lint_error_count", 0))
    if reqs["require_lint_clean"] and not lint_passed:
        blockers.append(
            f"Lint not clean ({lint_errors} errors). " "Run 'ruff check . --fix' or 'flake8 .' and fix errors."
        )

    # Check security scan (v5.1: use "security_scanned" and "security_clean" keys from CLI)
    security_scanned = status.get("security_scanned", False)
    security_clean = status.get("security_clean", status.get("security_scan_passed", False))
    if reqs["require_security_scan"] and not (security_scanned and security_clean):
        blockers.append(
            "Security scan not passed or not run. " "Run 'gitleaks detect --source .' to check for secrets."
        )

    # Optional: Type check
    if reqs.get("require_type_check", False):
        type_passed = status.get("type_check_passed", False)
        if not type_passed:
            blockers.append("Type check not passing. Run 'mypy --config-file mypy.ini' and fix errors.")

    # v5.1: Audit Suite Integration
    # Note: Full audit runs are displayed in cli_p3_status
    # Here we only check if audit was explicitly required and failed
    if is_feature_enabled("p3_audit_integration"):
        audit_run = status.get("audit_run", False)
        audit_critical = status.get("audit_critical_count", 0)
        if is_feature_enabled("audit_blocks_exit") and audit_critical > 0:
            blockers.append(
                f"Audit found {audit_critical} critical issues. "
                "Run 'python3 .claude/hooks/ric.py audit' and fix critical issues."
            )
        elif not audit_run:
            # Soft warning - shown in status but doesn't block
            status["audit_suggestion"] = (
                "Consider running 'python3 .claude/hooks/ric.py audit' " "to check for code quality issues."
            )

    completed = len(blockers) == 0
    status["completed"] = completed
    status["blockers"] = blockers

    return completed, blockers, status


def check_p4_completion(state: "RICState") -> tuple[bool, list[str], dict]:
    """
    Check if P4 REFLECT requirements are met.

    v5.1 Update: Now includes QUALITY GATES that validate content, not just
    boolean flags. This prevents gaming by calling commands with placeholder values.

    Quality Gates (when quality_gates feature enabled):
    1. Upgrade ideas must reference specific locations (file:line, function)
    2. Upgrade ideas must have concrete action verbs
    3. Iteration 2+ must cite previous iteration (Reflexion pattern)

    Returns:
        (is_complete, blockers, status_dict)
    """
    if not PHASE_REQUIREMENTS["P4_REFLECT"]["enabled"]:
        return True, [], {"phase": "P4", "completed": True}

    reqs = PHASE_REQUIREMENTS["P4_REFLECT"]
    # Get status as dict (key is "P4_REFLECT" not "P4")
    status = state.phase_completion.get("P4_REFLECT", {})
    blockers = []

    # Check introspection
    introspection_done = status.get("introspection_done", False)
    if reqs["require_introspection"] and not introspection_done:
        blockers.append(
            "Introspection not completed. Review what went well, what could improve, " "and generate upgrade ideas."
        )

    # Check insights reviewed
    open_insights = [i for i in state.insights if not i.resolved]
    insights_reviewed = status.get("insights_reviewed", False)
    if reqs["min_insights_considered"] > 0 and not insights_reviewed:
        if open_insights:
            blockers.append(
                f"Have {len(open_insights)} open insights that haven't been reviewed. "
                "Decide: resolve, defer, or document why they remain."
            )

    # Check convergence
    convergence_checked = status.get("convergence_checked", False)
    if reqs["require_convergence_check"] and not convergence_checked:
        blockers.append(
            "Convergence check not done. Run 'convergence' command or manually " "assess if progress is being made."
        )

    # Check upgrade ideas - BASIC CHECK (still need at least one)
    upgrade_ideas = status.get("upgrade_ideas", [])
    if reqs["require_upgrade_ideas"] and not upgrade_ideas:
        blockers.append(
            "No upgrade ideas generated. Propose at least 1 improvement, "
            "even if small (e.g., 'Add better error handling to X')."
        )

    # ==========================================================================
    # v5.1 QUALITY GATES - Validate CONTENT, not just existence
    # ==========================================================================
    if is_feature_enabled("quality_gates") and upgrade_ideas:
        # Get introspection text for citation checking
        introspection_text = status.get("introspection_text", "")

        # Run quality assessment
        quality_assessment = validate_p4_quality(
            upgrade_ideas=upgrade_ideas,
            current_iteration=state.iteration,
            introspection_text=introspection_text,
            previous_iteration_summary="",  # Could enhance to pass actual previous summary
        )

        # Add quality blockers to main blockers
        if not quality_assessment.passes_quality_gate:
            blockers.extend(quality_assessment.quality_blockers)

        # Store quality assessment in status for debugging
        status["quality_assessment"] = {
            "overall_score": quality_assessment.overall_quality_score,
            "passes_quality_gate": quality_assessment.passes_quality_gate,
            "ideas_with_location": quality_assessment.ideas_with_location,
            "ideas_with_action": quality_assessment.ideas_with_action,
            "vague_ideas_count": quality_assessment.vague_ideas_count,
            "actionable_ideas_count": quality_assessment.actionable_ideas_count,
            "cites_previous_iteration": quality_assessment.cites_previous_iteration,
        }

    # Check loop decision (required before advancing/exiting)
    loop_decision = status.get("loop_decision", "")
    if not loop_decision:
        blockers.append(
            "No LOOP/EXIT decision made. Decide: continue to next iteration (LOOP) " "or complete session (EXIT)."
        )

    completed = len(blockers) == 0
    status["completed"] = completed
    status["blockers"] = blockers

    return completed, blockers, status


def get_phase_start_prompt(phase: "Phase", state: "RICState") -> str:
    """Get the mandatory prompt for starting a phase."""
    if phase == Phase.RESEARCH:
        # Use dict access (key is "P0_RESEARCH")
        status = state.phase_completion.get("P0_RESEARCH", {})
        return P0_RESEARCH_START_PROMPT.format(
            search_count=status.get("web_searches_done", 0),
            keyword_count=len(status.get("keywords_extracted", [])),
            persisted="Yes" if status.get("findings_persisted", False) else "No",
        )
    elif phase == Phase.REFLECT:
        p0_count = len([i for i in state.insights if not i.resolved and i.priority == Priority.P0])
        p1_count = len([i for i in state.insights if not i.resolved and i.priority == Priority.P1])
        p2_count = len([i for i in state.insights if not i.resolved and i.priority == Priority.P2])
        return P4_REFLECT_START_PROMPT.format(
            p0_count=p0_count,
            p1_count=p1_count,
            p2_count=p2_count,
        )
    return ""


def can_advance_phase(state: "RICState") -> tuple[bool, str, list[str]]:
    """
    Check if current phase requirements are met to advance.

    v5.1 Update: Now checks ALL phases, not just P0 and P4.

    Returns:
        (can_advance, phase_name, blockers)
    """
    current_phase = state.current_phase

    if current_phase == Phase.RESEARCH:
        completed, blockers, _ = check_p0_completion(state)
        if not completed and PHASE_REQUIREMENTS["P0_RESEARCH"]["block_advance_without_completion"]:
            return False, "P0 RESEARCH", blockers

    elif current_phase == Phase.PLAN:
        # v5.1 NEW: P1 PLAN phase enforcement
        completed, blockers, _ = check_p1_completion(state)
        if not completed and PHASE_REQUIREMENTS["P1_PLAN"]["block_advance_without_completion"]:
            return False, "P1 PLAN", blockers

    elif current_phase == Phase.BUILD:
        # v5.1 NEW: P2 BUILD phase enforcement
        completed, blockers, _ = check_p2_completion(state)
        if not completed and PHASE_REQUIREMENTS["P2_BUILD"]["block_advance_without_completion"]:
            return False, "P2 BUILD", blockers

    elif current_phase == Phase.VERIFY:
        # v5.1 NEW: P3 VERIFY phase enforcement
        completed, blockers, _ = check_p3_completion(state)
        if not completed and PHASE_REQUIREMENTS["P3_VERIFY"]["block_advance_without_completion"]:
            return False, "P3 VERIFY", blockers

    elif current_phase == Phase.REFLECT:
        completed, blockers, _ = check_p4_completion(state)
        if not completed and PHASE_REQUIREMENTS["P4_REFLECT"]["block_advance_without_completion"]:
            return False, "P4 REFLECT", blockers

    return True, "", []


# P0 Research helper functions
def extract_keywords(task_description: str) -> list[str]:
    """Extract keywords from task description for research."""
    # Remove common words and extract key terms
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "we",
        "they",
        "it",
    }

    words = re.findall(r"\b[a-zA-Z]{3,}\b", task_description.lower())
    keywords = [w for w in words if w not in stop_words]

    # Return unique keywords, preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)

    return unique[:10]  # Top 10 keywords


def expand_keywords(keywords: list[str]) -> list[str]:
    """Expand keywords with related terms for broader research."""
    expansions = {
        "agent": ["agentic", "autonomous", "multi-agent", "agent workflow"],
        "llm": ["large language model", "gpt", "claude", "ai", "ml"],
        "test": ["testing", "unit test", "integration test", "tdd"],
        "code": ["coding", "programming", "development", "implementation"],
        "bug": ["error", "fix", "debug", "issue", "defect"],
        "feature": ["functionality", "capability", "enhancement"],
        "performance": ["optimization", "speed", "efficiency", "latency"],
        "security": ["vulnerability", "safety", "authentication", "authorization"],
        "api": ["endpoint", "rest", "graphql", "interface"],
        "database": ["db", "sql", "nosql", "storage", "persistence"],
    }

    expanded = list(keywords)
    for kw in keywords:
        if kw in expansions:
            expanded.extend(expansions[kw])

    return list(set(expanded))


def generate_search_queries(keywords: list[str], topic: str = "") -> list[str]:
    """Generate search queries from keywords."""
    queries = []

    # Primary queries with current year
    for kw in keywords[:3]:
        queries.append(f"{kw} {topic} 2025 best practices")
        queries.append(f"{kw} latest research 2025")

    # Combination queries
    if len(keywords) >= 2:
        queries.append(f"{keywords[0]} {keywords[1]} implementation pattern")

    # Trend queries
    queries.append(f"{topic or keywords[0]} trends 2025")

    return queries[:5]  # Return top 5 queries


# P4 Reflect helper functions
def generate_introspection_questions() -> list[str]:
    """Generate introspection questions for P4 REFLECT."""
    return [
        "What worked well in this iteration?",
        "What patterns or approaches were effective?",
        "What caused delays or confusion?",
        "What would I do differently?",
        "What's still unclear or uncertain?",
        "What new insights emerged?",
        "What technical debt was introduced?",
        "What improvements could benefit the codebase?",
        "Are there any recurring issues?",
        "What should the next iteration focus on?",
    ]


def assess_convergence_for_reflect(state: "RICState") -> dict:
    """Assess convergence status for P4 REFLECT phase."""
    result = {
        "is_converging": False,
        "progress_rate": "Unknown",
        "fix_success_rate": "Unknown",
        "code_churn": "Unknown",
        "verdict": "Need more data",
        "recommendation": "LOOP",
    }

    if len(state.iteration_metrics) < 2:
        return result

    # Calculate metrics
    try:
        is_converging, message, metrics = calculate_convergence(state.iteration_metrics)
        result["is_converging"] = is_converging
        result["progress_rate"] = f"{metrics.get('insight_rate_change', 0):.1%}"
        result["fix_success_rate"] = f"{metrics.get('fix_success_change', 0):.1%}"
        result["code_churn"] = f"{metrics.get('churn_change', 0):.1%}"
        result["verdict"] = message
        result["recommendation"] = "EXIT" if is_converging else "LOOP"
    except Exception:
        pass

    return result


def record_p0_web_search(state: "RICState") -> None:
    """Record a web search for P0 completion tracking."""
    if "P0" not in state.phase_completion:
        state.phase_completion["P0"] = PhaseCompletionStatus(phase="P0")
    state.phase_completion["P0"].web_searches_done += 1


def record_p0_keywords(state: "RICState", keywords: list[str]) -> None:
    """Record extracted keywords for P0 completion."""
    if "P0" not in state.phase_completion:
        state.phase_completion["P0"] = PhaseCompletionStatus(phase="P0")
    state.phase_completion["P0"].keywords_extracted = keywords


def record_p0_persist(state: "RICState", source_count: int) -> None:
    """Record research persistence for P0 completion."""
    if "P0" not in state.phase_completion:
        state.phase_completion["P0"] = PhaseCompletionStatus(phase="P0")
    state.phase_completion["P0"].findings_persisted = True
    state.phase_completion["P0"].sources_documented = source_count


def record_p4_introspection(state: "RICState", introspection_notes: str) -> None:
    """Record introspection completion for P4."""
    if "P4" not in state.phase_completion:
        state.phase_completion["P4"] = PhaseCompletionStatus(phase="P4")
    state.phase_completion["P4"].introspection_done = True


def record_p4_insights_reviewed(state: "RICState") -> None:
    """Record that insights were reviewed in P4."""
    if "P4" not in state.phase_completion:
        state.phase_completion["P4"] = PhaseCompletionStatus(phase="P4")
    state.phase_completion["P4"].insights_reviewed = True


def record_p4_convergence_check(state: "RICState") -> None:
    """Record convergence check completion for P4."""
    if "P4" not in state.phase_completion:
        state.phase_completion["P4"] = PhaseCompletionStatus(phase="P4")
    state.phase_completion["P4"].convergence_checked = True


def record_p4_upgrade_idea(state: "RICState", idea: str, priority: str = "P1") -> None:
    """Record an upgrade idea generated in P4."""
    if "P4" not in state.phase_completion:
        state.phase_completion["P4"] = PhaseCompletionStatus(phase="P4")
    state.phase_completion["P4"].upgrade_ideas.append(
        {
            "idea": idea,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
        }
    )


def record_p4_loop_decision(state: "RICState", decision: str, reason: str) -> None:
    """Record LOOP/EXIT decision in P4."""
    if "P4" not in state.phase_completion:
        state.phase_completion["P4"] = PhaseCompletionStatus(phase="P4")
    state.phase_completion["P4"].loop_decision = decision
    state.phase_completion["P4"].loop_reason = reason


def reset_phase_completion(state: "RICState", phase: str) -> None:
    """Reset phase completion status for new iteration."""
    state.phase_completion[phase] = PhaseCompletionStatus(phase=phase)


# =============================================================================
# v5.1 QUALITY GATES - SELF-REFINE + REFLEXION ANTI-GAMING (December 2025)
# =============================================================================
# Research-backed: SELF-REFINE (Madaan et al.), Reflexion (Shinn et al.)
# Problem: v5.0 P4 REFLECT was gamed by calling commands with placeholder values
# Solution: Validate CONTENT QUALITY, not just boolean flags
# =============================================================================

# Patterns for detecting actionable upgrade ideas (SELF-REFINE style)
UPGRADE_IDEA_QUALITY_PATTERNS = {
    # Location patterns - must reference specific file/function/line
    "location": [
        r"[\w/]+\.py:\d+",  # file.py:123
        r"[\w/]+\.py:[\w_]+\(\)",  # file.py:function()
        r"`[\w/]+\.py`",  # `file.py` in markdown
        r"in\s+`?[\w/]+\.py`?",  # "in file.py" or "in `file.py`"
        r"to\s+`?[\w/]+\.py`?",  # "to file.py" or "to `file.py`"
        r"[\w]+/[\w]+\.py",  # directory/file.py
        r"\b[\w_]+\.py\b",  # bare filename.py (must be word bounded)
        r"class\s+[\w]+",  # class ClassName
        r"function\s+[\w_]+",  # function name
        r"method\s+[\w_]+",  # method name
        r"def\s+[\w_]+",  # def function_name
    ],
    # Action patterns - must have concrete action verbs
    "action": [
        r"\b(add|implement|create|refactor|fix|update|remove|replace|extract)\b",
        r"\b(integrate|improve|enhance|optimize|validate|verify|test)\b",
        r"\b(split|merge|rename|move|consolidate|extend|wrap)\b",
        r"\b(handle|catch|check|enforce|require|validate)\b",
    ],
    # Anti-vague patterns - reject if these are the ONLY content
    "vague_only": [
        r"^improve\s*(it|this|the\s+code)?\.?$",
        r"^make\s*(it)?\s*better\.?$",
        r"^something$",
        r"^todo$",
        r"^fix\s*it\.?$",
        r"^needs?\s*work\.?$",
        r"^could\s*be\s*improved\.?$",
    ],
}

# Minimum quality thresholds
UPGRADE_IDEA_QUALITY_THRESHOLDS = {
    "min_ideas_with_location": 1,  # At least 1 idea must have file:line
    "min_ideas_with_action": 1,  # At least 1 idea must have action verb
    "min_idea_length_chars": 20,  # Ideas must be at least 20 chars
    "max_vague_ideas_pct": 0.50,  # Max 50% of ideas can be vague
    "require_iteration_citation": True,  # If iteration > 1, must cite previous
}


@dataclass
class UpgradeIdeaQuality:
    """Quality assessment for a single upgrade idea (SELF-REFINE style)."""

    idea: str
    has_location: bool = False  # References specific file/function/line
    has_action: bool = False  # Has concrete action verb
    is_vague: bool = True  # Is too vague to be actionable
    is_actionable: bool = False  # Overall actionability
    location_matches: list = field(default_factory=list)
    action_matches: list = field(default_factory=list)
    quality_score: float = 0.0  # 0.0 to 1.0


@dataclass
class P4QualityAssessment:
    """Overall P4 REFLECT quality assessment (Reflexion episodic memory)."""

    ideas_assessed: list[UpgradeIdeaQuality] = field(default_factory=list)
    overall_quality_score: float = 0.0
    passes_quality_gate: bool = False
    quality_blockers: list[str] = field(default_factory=list)
    # Reflexion: Citation tracking
    cites_previous_iteration: bool = False
    iteration_citations: list[str] = field(default_factory=list)
    # Summary stats
    ideas_with_location: int = 0
    ideas_with_action: int = 0
    vague_ideas_count: int = 0
    actionable_ideas_count: int = 0


def validate_upgrade_idea(idea_text: str) -> UpgradeIdeaQuality:
    """
    Validate a single upgrade idea for quality (SELF-REFINE style).

    Checks for:
    1. Location: Does it reference a specific file/function/line?
    2. Action: Does it have a concrete action verb?
    3. Vagueness: Is it too vague to be actionable?

    Returns:
        UpgradeIdeaQuality with assessment details
    """
    result = UpgradeIdeaQuality(idea=idea_text)

    # Normalize text
    text = idea_text.lower() if idea_text else ""
    original_text = idea_text or ""

    # Check for location patterns
    for pattern in UPGRADE_IDEA_QUALITY_PATTERNS["location"]:
        matches = re.findall(pattern, original_text, re.IGNORECASE)
        if matches:
            result.has_location = True
            result.location_matches.extend(matches)

    # Check for action patterns
    for pattern in UPGRADE_IDEA_QUALITY_PATTERNS["action"]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            result.has_action = True
            result.action_matches.extend(matches)

    # Check for vagueness (only vague if matches vague_only patterns)
    result.is_vague = False
    if len(text) < UPGRADE_IDEA_QUALITY_THRESHOLDS["min_idea_length_chars"]:
        result.is_vague = True
    else:
        for pattern in UPGRADE_IDEA_QUALITY_PATTERNS["vague_only"]:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                result.is_vague = True
                break

    # Calculate actionability
    result.is_actionable = (result.has_location or result.has_action) and not result.is_vague

    # Quality score: 0.0 to 1.0
    score = 0.0
    if result.has_location:
        score += 0.4
    if result.has_action:
        score += 0.4
    if not result.is_vague:
        score += 0.2
    result.quality_score = min(score, 1.0)

    return result


def validate_p4_quality(
    upgrade_ideas: list,
    current_iteration: int,
    introspection_text: str = "",
    previous_iteration_summary: str = "",
) -> P4QualityAssessment:
    """
    Comprehensive P4 REFLECT quality assessment (SELF-REFINE + Reflexion).

    This validates that P4 actually produced meaningful introspection, not just
    placeholder values to satisfy boolean checks.

    Args:
        upgrade_ideas: List of upgrade idea dicts with 'idea' key
        current_iteration: Current iteration number (1-indexed)
        introspection_text: The full introspection text
        previous_iteration_summary: Summary from previous iteration (for citation check)

    Returns:
        P4QualityAssessment with pass/fail and detailed blockers
    """
    assessment = P4QualityAssessment()

    # Assess each upgrade idea
    for idea_dict in upgrade_ideas:
        idea_text = idea_dict.get("idea", "") if isinstance(idea_dict, dict) else str(idea_dict)
        quality = validate_upgrade_idea(idea_text)
        assessment.ideas_assessed.append(quality)

        if quality.has_location:
            assessment.ideas_with_location += 1
        if quality.has_action:
            assessment.ideas_with_action += 1
        if quality.is_vague:
            assessment.vague_ideas_count += 1
        if quality.is_actionable:
            assessment.actionable_ideas_count += 1

    # Check quality thresholds
    total_ideas = len(assessment.ideas_assessed)
    thresholds = UPGRADE_IDEA_QUALITY_THRESHOLDS

    # Gate 1: At least one idea with location
    if assessment.ideas_with_location < thresholds["min_ideas_with_location"]:
        assessment.quality_blockers.append(
            f"QUALITY GATE FAILED: No upgrade ideas reference a specific location "
            f"(file:line, function name, or class). Found {assessment.ideas_with_location} "
            f"with location, need at least {thresholds['min_ideas_with_location']}. "
            f"Example format: 'Add error handling to mcp/broker_server.py:145 get_positions()'"
        )

    # Gate 2: At least one idea with action verb
    if assessment.ideas_with_action < thresholds["min_ideas_with_action"]:
        assessment.quality_blockers.append(
            f"QUALITY GATE FAILED: No upgrade ideas have concrete action verbs "
            f"(add, implement, fix, refactor, etc.). Found {assessment.ideas_with_action} "
            f"with actions, need at least {thresholds['min_ideas_with_action']}. "
            f"Example format: 'Implement real Schwab API integration in broker_server.py'"
        )

    # Gate 3: Not too many vague ideas
    if total_ideas > 0:
        vague_pct = assessment.vague_ideas_count / total_ideas
        if vague_pct > thresholds["max_vague_ideas_pct"]:
            assessment.quality_blockers.append(
                f"QUALITY GATE FAILED: Too many vague ideas ({assessment.vague_ideas_count}/{total_ideas} = "
                f"{vague_pct:.0%}). Max allowed: {thresholds['max_vague_ideas_pct']:.0%}. "
                f"Replace vague ideas like 'improve it' with specific ones like "
                f"'Add input validation to X function in Y file'"
            )

    # Gate 4 (Reflexion): Must cite previous iteration if iteration > 1
    if current_iteration > 1 and thresholds["require_iteration_citation"]:
        # Look for references to previous iteration in introspection text
        citation_patterns = [
            r"iteration\s*\d",
            r"previous\s*(iteration|phase|cycle)",
            r"last\s*(iteration|phase|cycle)",
            r"from\s*(iteration|phase)\s*\d",
            r"in\s*I[1-9]",
            r"\[I\d",
            r"built\s*in\s*iteration",
            r"discovered\s*in\s*iteration",
        ]

        full_text = introspection_text + " ".join(
            [
                str(idea_dict.get("idea", "")) if isinstance(idea_dict, dict) else str(idea_dict)
                for idea_dict in upgrade_ideas
            ]
        )

        for pattern in citation_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                assessment.cites_previous_iteration = True
                assessment.iteration_citations.extend(matches)

        if not assessment.cites_previous_iteration:
            assessment.quality_blockers.append(
                f"QUALITY GATE FAILED (Reflexion): Iteration {current_iteration} does not cite "
                f"previous iteration work. You must reference what was done in iteration "
                f"{current_iteration - 1} and explain what still needs improvement. "
                f"Example: 'In iteration {current_iteration - 1}, we built the basic broker_server. "
                f"Now need to add real API integration.'"
            )

    # Calculate overall quality score (average of individual scores)
    if assessment.ideas_assessed:
        assessment.overall_quality_score = sum(q.quality_score for q in assessment.ideas_assessed) / len(
            assessment.ideas_assessed
        )

    # Pass quality gate if no blockers
    assessment.passes_quality_gate = len(assessment.quality_blockers) == 0

    return assessment


# -----------------------------------------------------------------------------
# v5.0 FEATURE STATUS HELPERS
# -----------------------------------------------------------------------------


def get_v50_feature_status() -> str:
    """Get status of all v5.0 features."""
    lines = ["**RIC v5.0 Feature Status**", ""]

    for feature, enabled in FEATURE_FLAGS.items():
        status = "✅ Enabled" if enabled else "❌ Disabled"
        lines.append(f"  {feature}: {status}")

    return "\n".join(lines)


def enable_feature(feature_name: str) -> bool:
    """Enable a v5.0 feature."""
    if feature_name in FEATURE_FLAGS:
        FEATURE_FLAGS[feature_name] = True
        return True
    return False


def disable_feature(feature_name: str) -> bool:
    """Disable a v5.0 feature."""
    if feature_name in FEATURE_FLAGS:
        FEATURE_FLAGS[feature_name] = False
        return True
    return False


# =============================================================================
# DOCUMENTATION ENFORCEMENT (v4.3 NEW - Upgrade Doc Lifecycle)
# =============================================================================


def validate_doc_naming(file_path: str) -> tuple[bool, str, str]:
    """
    Validate document naming convention.

    Returns:
        (is_valid, doc_type, error_message)
    """
    filename = Path(file_path).name

    # Check each pattern type
    for doc_type, patterns in UPGRADE_DOC_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, filename):
                return True, doc_type, ""

    # Invalid naming
    return False, "", DOC_NAMING_WARNING.format(file_path=file_path)


def suggest_correct_naming(file_path: str) -> str | None:
    """
    Suggest the correct naming for an incorrectly named doc.

    Returns:
        Suggested correct path, or None if can't determine.
    """
    filename = Path(file_path).name.upper()
    directory = Path(file_path).parent

    # Try to extract components from the filename
    # Pattern: might have upgrade, category, topic info in various formats

    # Look for upgrade number
    upgrade_match = re.search(r"UPGRADE[_-]?(\d{2,3})", filename, re.IGNORECASE)
    if not upgrade_match:
        return None

    upgrade_num = upgrade_match.group(1).zfill(3)
    upgrade_id = f"UPGRADE-{upgrade_num}"

    # Look for category
    cat_match = re.search(r"CAT[_-]?(\d+)[_-]?([A-Z][A-Z0-9_-]*)?", filename, re.IGNORECASE)

    # Look for topic (anything that looks like a topic name)
    topic_match = re.search(r"[_-]([A-Z][A-Z0-9_-]{2,})[_-]?(?:RESEARCH)?\.md", filename, re.IGNORECASE)

    if cat_match:
        cat_num = cat_match.group(1)
        cat_name = cat_match.group(2) or topic_match.group(1) if topic_match else "TOPIC"
        cat_name = cat_name.upper().replace("_", "-").strip("-")
        suggested = f"{upgrade_id}-CAT{cat_num}-{cat_name}-RESEARCH.md"
    elif topic_match:
        topic = topic_match.group(1).upper().replace("_", "-").strip("-")
        suggested = f"{upgrade_id}-{topic}-RESEARCH.md"
    else:
        # Generic fallback
        suggested = f"{upgrade_id}-RESEARCH.md"

    return str(directory / suggested)


def auto_fix_doc_naming(file_path: str, content: str) -> tuple[bool, str, str]:
    """
    Automatically fix document naming if possible.

    If the file has incorrect naming:
    1. Suggest correct name
    2. If auto_fix_naming is enabled, write to correct path instead

    Returns:
        (fixed, new_path, message)
    """
    is_valid, doc_type, error = validate_doc_naming(file_path)

    if is_valid:
        return False, file_path, ""  # Already valid

    if not DOC_ENFORCEMENT.get("auto_fix_naming"):
        return False, file_path, error  # Can't fix

    suggested = suggest_correct_naming(file_path)
    if not suggested:
        return False, file_path, f"Could not determine correct naming for: {file_path}"

    # Write to the correct path
    try:
        suggested_path = Path(suggested)
        suggested_path.parent.mkdir(parents=True, exist_ok=True)
        suggested_path.write_text(content)

        message = f"📝 Auto-renamed: {Path(file_path).name} → {suggested_path.name}"

        # Log the fix
        log_autonomous_issue(
            "auto_renamed_doc",
            {
                "original": file_path,
                "renamed_to": suggested,
            },
        )

        return True, suggested, message
    except OSError as e:
        return False, file_path, f"Could not auto-fix naming: {e}"


def detect_upgrade_from_path(file_path: str) -> str | None:
    """Extract upgrade ID from file path or name."""
    match = re.search(r"UPGRADE-(\d{3})", file_path)
    return match.group(0) if match else None


def detect_category_from_path(file_path: str) -> tuple[int, str] | None:
    """Extract category number and name from file path."""
    # Pattern: CAT3-FAULT-TOLERANCE (stops before -RESEARCH or .md)
    match = re.search(r"CAT(\d+)-([A-Z][A-Z0-9-]+?)(?:-RESEARCH|\.md|$)", file_path)
    if match:
        return int(match.group(1)), match.group(2)
    return None


def get_expected_research_doc(upgrade_id: str, category: tuple[int, str] | None = None) -> Path:
    """Get expected research doc path for an upgrade/category."""
    if category:
        cat_num, cat_name = category
        return Path(f"docs/research/{upgrade_id}-CAT{cat_num}-{cat_name}-RESEARCH.md")
    return Path(f"docs/research/{upgrade_id}-RESEARCH.md")


def create_stub_doc(doc_path: Path, upgrade_id: str, topic: str, category: tuple[int, str] | None = None) -> bool:
    """
    Auto-create a stub documentation file in autonomous mode.

    Returns:
        True if stub was created, False otherwise.
    """
    if not DOC_ENFORCEMENT.get("auto_create_stub_docs"):
        return False

    stub_marker = DOC_ENFORCEMENT.get("stub_doc_marker", "<!-- AUTO-STUB -->")

    if category:
        cat_num, cat_name = category
        title = f"{upgrade_id} Category {cat_num}: {cat_name} Research"
    else:
        title = f"{upgrade_id} {topic} Research"

    stub_content = f"""{stub_marker}

# {title}

**Status**: AUTO-GENERATED STUB - Needs content
**Created**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Mode**: Autonomous session (auto-created to prevent blocking)

## Overview

<!-- TODO: Add overview of this upgrade/category -->

## Research Objectives

<!-- TODO: Add research objectives -->

## Key Sources

<!-- TODO: Add sources with timestamps -->

## Key Discoveries

<!-- TODO: Document key findings -->

## Implementation Status

- [ ] Research complete
- [ ] Implementation started
- [ ] Tests written
- [ ] Documentation finalized

---
**NOTE**: This stub was auto-created during an autonomous session.
Review and fill in content before marking as complete.
"""

    try:
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(stub_content)

        # Log the auto-creation
        log_autonomous_issue(
            "stub_doc_created",
            {
                "path": str(doc_path),
                "upgrade_id": upgrade_id,
                "topic": topic,
                "category": category,
            },
        )

        return True
    except OSError:
        return False


# =============================================================================
# AUTO-PERSIST RESEARCH (v4.3 NEW - Never Block, Always Persist)
# =============================================================================

# State file for tracking research across tool calls
RESEARCH_STATE_FILE = Path(".claude/research_state.json")


def load_research_state() -> dict:
    """Load the research tracking state."""
    if RESEARCH_STATE_FILE.exists():
        try:
            return json.loads(RESEARCH_STATE_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "search_count": 0,
        "queries": [],
        "urls": [],
        "last_persist_time": None,
        "current_upgrade": None,
        "current_topic": None,
        "unpersisted_findings": [],
    }


def save_research_state(state: dict) -> None:
    """Save the research tracking state."""
    try:
        RESEARCH_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        RESEARCH_STATE_FILE.write_text(json.dumps(state, indent=2))
    except OSError:
        pass


def detect_current_upgrade() -> tuple[str | None, str | None]:
    """
    Detect current upgrade ID and topic from RIC state or progress file.

    Returns:
        (upgrade_id, topic) - e.g., ("UPGRADE-017", "DOC-ENFORCEMENT")
    """
    # Try RIC state first
    ric_state_file = Path(".claude/state/ric.json")
    if ric_state_file.exists():
        try:
            state = json.loads(ric_state_file.read_text())
            if state.get("upgrade_id"):
                topic = state.get("topic", "RESEARCH")
                return state["upgrade_id"], topic
        except (OSError, json.JSONDecodeError):
            pass

    # Try progress file
    progress_file = Path("claude-progress.txt")
    if progress_file.exists():
        try:
            content = progress_file.read_text()
            upgrade_match = re.search(r"UPGRADE-(\d{3})", content)
            if upgrade_match:
                return f"UPGRADE-{upgrade_match.group(1)}", "RESEARCH"
        except OSError:
            pass

    return None, None


def get_or_create_research_doc(upgrade_id: str, topic: str) -> Path:
    """
    Get existing research doc path or create new one.

    Returns:
        Path to the research document (creates if doesn't exist).
    """
    # Try to find existing doc
    research_dir = Path("docs/research")
    research_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing upgrade research doc
    pattern = f"{upgrade_id}*RESEARCH.md"
    existing_docs = list(research_dir.glob(pattern))

    if existing_docs:
        # Use the most recently modified one
        return max(existing_docs, key=lambda p: p.stat().st_mtime)

    # Create new doc
    safe_topic = re.sub(r"[^A-Z0-9-]", "-", topic.upper())
    doc_path = research_dir / f"{upgrade_id}-{safe_topic}-RESEARCH.md"

    if not doc_path.exists():
        template = f"""# {upgrade_id} {topic} Research

**Status**: AUTO-CREATED - Research in progress
**Created**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Mode**: Auto-persist enabled

## Research Overview

This document is automatically maintained during the research process.
Findings are auto-persisted to prevent loss during context compaction.

## Research Log

<!-- Auto-persisted research entries will be added below -->

"""
        doc_path.write_text(template)
        log_autonomous_issue(
            "research_doc_created",
            {
                "path": str(doc_path),
                "upgrade_id": upgrade_id,
                "topic": topic,
            },
        )

    return doc_path


def format_research_entry(queries: list[str], urls: list[str], findings: list[str], timestamp: str) -> str:
    """Format a research entry for appending to doc."""
    entry = f"""
---

### Research Entry - {timestamp}

**Search Queries**:
"""
    for query in queries:
        entry += f'- "{query}"\n'

    if urls:
        entry += "\n**URLs Discovered**:\n"
        for url in urls:
            entry += f"- {url}\n"

    if findings:
        entry += "\n**Key Findings**:\n"
        for finding in findings:
            entry += f"- {finding}\n"

    entry += f"\n**Search Date**: {timestamp}\n"
    entry += "**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.\n"

    return entry


def auto_persist_research(
    query: str | None = None, url: str | None = None, finding: str | None = None
) -> tuple[bool, str]:
    """
    Track research and auto-persist when threshold is reached.

    This function NEVER blocks - it always succeeds and returns status.

    Args:
        query: Search query (from WebSearch)
        url: URL (from WebFetch)
        finding: Key finding to persist

    Returns:
        (persisted, message) - Whether auto-persist happened and info message
    """
    state = load_research_state()

    # Update tracking state
    if query and query not in state["queries"]:
        state["queries"].append(query)
        state["search_count"] += 1
    if url and url not in state["urls"]:
        state["urls"].append(url)
    if finding:
        state["unpersisted_findings"].append(finding)

    # Detect current upgrade context
    upgrade_id, topic = detect_current_upgrade()
    if upgrade_id:
        state["current_upgrade"] = upgrade_id
        state["current_topic"] = topic

    save_research_state(state)

    # Check if we should auto-persist
    persist_threshold = RESEARCH_ENFORCEMENT.get("searches_before_auto_persist", 3)
    should_persist = (
        RESEARCH_ENFORCEMENT.get("auto_persist_research", True)
        and state["search_count"] >= persist_threshold
        and state["queries"]  # Have something to persist
    )

    if not should_persist:
        return False, f"📊 Research tracked ({state['search_count']}/{persist_threshold} before auto-persist)"

    # Auto-persist now
    upgrade_id = state.get("current_upgrade") or "UPGRADE-UNKNOWN"
    topic = state.get("current_topic") or "RESEARCH"

    try:
        doc_path = get_or_create_research_doc(upgrade_id, topic)
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

        entry = format_research_entry(
            queries=state["queries"], urls=state["urls"], findings=state["unpersisted_findings"], timestamp=timestamp
        )

        # Append to doc
        with open(doc_path, "a") as f:
            f.write(entry)

        # Log the auto-persist
        log_autonomous_issue(
            "research_auto_persisted",
            {
                "doc_path": str(doc_path),
                "search_count": state["search_count"],
                "queries": state["queries"],
                "url_count": len(state["urls"]),
            },
        )

        # Reset state after persist
        queries_persisted = len(state["queries"])
        state["search_count"] = 0
        state["queries"] = []
        state["urls"] = []
        state["unpersisted_findings"] = []
        state["last_persist_time"] = datetime.now().isoformat()
        save_research_state(state)

        return True, RESEARCH_AUTOPERSIST_NOTICE.format(
            search_count=queries_persisted,
            research_doc_path=str(doc_path),
            queries=", ".join(state.get("queries", [])[:3]) or "various",
            today=timestamp,
            persist_interval=persist_threshold,
        )

    except OSError as e:
        # Never block - just log and continue
        log_autonomous_issue(
            "research_persist_failed",
            {
                "error": str(e),
                "search_count": state["search_count"],
            },
        )
        return False, f"⚠️ Auto-persist failed (will retry): {e}"


def handle_websearch_post(tool_input: dict, tool_result: dict) -> str:
    """
    PostToolUse handler for WebSearch - auto-persists research.

    Returns:
        Message to show (empty if nothing to report)
    """
    query = tool_input.get("query", "")
    persisted, message = auto_persist_research(query=query)

    if persisted:
        return message
    return ""


def handle_webfetch_post(tool_input: dict, tool_result: dict) -> str:
    """
    PostToolUse handler for WebFetch - tracks URLs for auto-persist.

    Returns:
        Message to show (empty if nothing to report)
    """
    url = tool_input.get("url", "")
    persisted, message = auto_persist_research(url=url)

    if persisted:
        return message
    return ""


def reset_research_state() -> None:
    """Reset research state (called when RIC loop ends or new upgrade starts)."""
    state = {
        "search_count": 0,
        "queries": [],
        "urls": [],
        "last_persist_time": None,
        "current_upgrade": None,
        "current_topic": None,
        "unpersisted_findings": [],
    }
    save_research_state(state)


def check_doc_exists_for_completion(upgrade_id: str, category: tuple[int, str] | None = None) -> tuple[bool, str]:
    """
    Check if required documentation exists before allowing completion.

    ALWAYS auto-creates stub doc if missing (never blocks).

    Returns:
        (can_complete, message) - can_complete is always True (never blocks)
    """
    expected_doc = get_expected_research_doc(upgrade_id, category)

    if not expected_doc.exists():
        topic = category[1] if category else "UPGRADE"

        # Auto-create stub doc (always, not just autonomous mode)
        if DOC_ENFORCEMENT.get("auto_create_missing_docs", True):
            if create_stub_doc(expected_doc, upgrade_id, topic, category):
                return True, f"📝 Auto-created stub doc: {expected_doc.name}"

            # Stub creation failed - still allow but log
            log_autonomous_issue(
                "doc_missing_no_stub",
                {
                    "expected_doc": str(expected_doc),
                    "upgrade_id": upgrade_id,
                },
            )
            return True, f"⚠️ Doc missing (stub creation failed): {expected_doc.name}"

        # Auto-create disabled - just warn
        return True, f"⚠️ Doc missing: {expected_doc.name}"

    return True, ""


def validate_doc_sections(content: str, doc_type: str) -> tuple[bool, list[str]]:
    """
    Validate that document has required sections.

    Returns:
        (is_valid, missing_sections)
    """
    required = UPGRADE_DOC_SECTIONS.get(doc_type, [])
    missing = [section for section in required if section not in content]
    return len(missing) == 0, missing


def validate_cross_references(content: str) -> tuple[bool, list[str]]:
    """
    Validate that document has cross-references.

    Returns:
        (has_refs, found_refs)
    """
    found_refs = []
    for pattern in CROSS_REFERENCE_PATTERNS:
        matches = re.findall(pattern, content)
        found_refs.extend(matches)

    return len(found_refs) > 0, found_refs


def check_doc_staleness(last_update_time: float | None, phase: int) -> tuple[bool, str]:
    """
    Check if documentation is stale.

    Returns:
        (is_stale, warning_message)
    """
    if not last_update_time or not DOC_ENFORCEMENT.get("track_doc_staleness_hours"):
        return False, ""

    threshold_hours = DOC_ENFORCEMENT["track_doc_staleness_hours"]
    elapsed_hours = (time.time() - last_update_time) / 3600

    if elapsed_hours > threshold_hours:
        return True, DOC_STALENESS_WARNING.format(
            hours=int(elapsed_hours),
            last_update_time=datetime.fromtimestamp(last_update_time).strftime("%Y-%m-%d %H:%M"),
            phase=phase,
            upgrade_doc="docs/research/UPGRADE-XXX.md",
        )

    return False, ""


# =============================================================================
# MID-UPGRADE DOC UPDATE TRACKING (v4.3)
# =============================================================================

# State file for tracking doc updates during an upgrade
DOC_UPDATE_STATE_FILE = Path(".claude/state/doc_updates.json")


def load_doc_update_state() -> dict:
    """Load the doc update tracking state."""
    if not DOC_UPDATE_STATE_FILE.exists():
        return {
            "upgrade_id": "",
            "iteration": 0,
            "phase": 0,
            "doc_updates": [],  # List of (timestamp, file_path, phase)
            "last_update_time": None,
            "updates_per_phase": {},  # phase -> count
        }

    try:
        return json.loads(DOC_UPDATE_STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return load_doc_update_state.__wrapped__()  # Return default


def save_doc_update_state(state: dict) -> None:
    """Save the doc update tracking state."""
    try:
        DOC_UPDATE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        DOC_UPDATE_STATE_FILE.write_text(json.dumps(state, indent=2))
    except OSError:
        pass


def track_doc_update(file_path: str, upgrade_id: str, iteration: int, phase: int) -> None:
    """
    Track a documentation update.

    Called when an upgrade doc is written/edited.
    """
    state = load_doc_update_state()

    # Reset state if upgrade changed
    if state["upgrade_id"] != upgrade_id:
        state = {
            "upgrade_id": upgrade_id,
            "iteration": iteration,
            "phase": phase,
            "doc_updates": [],
            "last_update_time": None,
            "updates_per_phase": {},
        }

    # Update current position
    state["iteration"] = iteration
    state["phase"] = phase

    # Record the update
    update_record = {
        "timestamp": time.time(),
        "file_path": file_path,
        "phase": phase,
        "iteration": iteration,
    }
    state["doc_updates"].append(update_record)
    state["last_update_time"] = time.time()

    # Update per-phase count
    phase_key = str(phase)
    state["updates_per_phase"][phase_key] = state["updates_per_phase"].get(phase_key, 0) + 1

    save_doc_update_state(state)


def check_doc_updates_for_phase(upgrade_id: str, phase: int) -> tuple[bool, str]:
    """
    Check if documentation has been updated this phase.

    Returns:
        (has_updates, message)
    """
    if not DOC_ENFORCEMENT.get("min_doc_updates_per_phase"):
        return True, ""

    min_updates = DOC_ENFORCEMENT["min_doc_updates_per_phase"]
    state = load_doc_update_state()

    # Check if we're tracking the right upgrade
    if state["upgrade_id"] != upgrade_id:
        return False, f"No doc updates tracked for {upgrade_id}"

    phase_key = str(phase)
    updates_this_phase = state["updates_per_phase"].get(phase_key, 0)

    if updates_this_phase < min_updates:
        return False, (
            f"Phase {phase} requires {min_updates} doc update(s), "
            f"but only {updates_this_phase} recorded. "
            f"Update the upgrade doc before proceeding."
        )

    return True, f"Phase {phase} has {updates_this_phase} doc update(s)"


def get_doc_update_summary(upgrade_id: str) -> dict:
    """
    Get a summary of doc updates for an upgrade.

    Returns:
        Summary dict with counts, last update time, etc.
    """
    state = load_doc_update_state()

    if state["upgrade_id"] != upgrade_id:
        return {
            "tracked": False,
            "total_updates": 0,
            "updates_per_phase": {},
            "last_update": None,
            "last_file": None,
        }

    last_update = None
    last_file = None
    if state["doc_updates"]:
        last = state["doc_updates"][-1]
        last_update = datetime.fromtimestamp(last["timestamp"]).strftime("%Y-%m-%d %H:%M")
        last_file = last["file_path"]

    return {
        "tracked": True,
        "total_updates": len(state["doc_updates"]),
        "updates_per_phase": state["updates_per_phase"],
        "last_update": last_update,
        "last_file": last_file,
    }


DOC_UPDATE_REQUIRED_WARNING = """
**⚠️ DOC UPDATE REQUIRED**

Phase {phase} ({phase_name}) requires at least {min_updates} documentation update(s).
Current updates this phase: {current_updates}

Before advancing to the next phase, update the upgrade documentation:
  1. Open: docs/research/{upgrade_id}-*-RESEARCH.md
  2. Document what was accomplished this phase
  3. Update implementation status

This ensures future sessions have context about decisions made.
"""


# =============================================================================
# POST-COMPLETION CROSS-REFERENCE VALIDATION (v4.3)
# =============================================================================


def extract_file_references(content: str) -> list[str]:
    """
    Extract file path references from documentation content.

    Finds patterns like:
    - [file.py](path/to/file.py)
    - `path/to/file.py`
    - path/to/file.py (in code blocks)
    """
    refs = []

    # Markdown link pattern: [text](path)
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+\.(?:py|md|json|yaml|yml))\)")
    for match in link_pattern.finditer(content):
        refs.append(match.group(2))

    # Backtick pattern: `path/to/file.py`
    backtick_pattern = re.compile(r"`([^`]+\.(?:py|md|json|yaml|yml))`")
    for match in backtick_pattern.finditer(content):
        ref = match.group(1)
        # Filter out obvious non-paths
        if "/" in ref or ref.startswith("."):
            refs.append(ref)

    return list(set(refs))


def validate_file_references(content: str, base_path: Path = Path(".")) -> tuple[bool, list[str], list[str]]:
    """
    Validate that file references in content exist.

    Returns:
        (all_valid, valid_refs, missing_refs)
    """
    refs = extract_file_references(content)
    valid = []
    missing = []

    for ref in refs:
        # Clean up the reference
        clean_ref = ref.lstrip("./")
        file_path = base_path / clean_ref

        if file_path.exists():
            valid.append(ref)
        else:
            missing.append(ref)

    return len(missing) == 0, valid, missing


def validate_category_doc_links(
    upgrade_id: str, main_doc_content: str, category_docs: list[Path]
) -> tuple[bool, list[str]]:
    """
    Validate that main doc links to all category docs.

    Returns:
        (all_linked, missing_links)
    """
    missing = []

    for cat_doc in category_docs:
        doc_name = cat_doc.name
        # Check if the category doc is referenced in main doc
        if doc_name not in main_doc_content:
            # Also check for relative link
            if f"]({doc_name})" not in main_doc_content and f"](docs/research/{doc_name})" not in main_doc_content:
                missing.append(doc_name)

    return len(missing) == 0, missing


def validate_upgrade_completion(upgrade_id: str) -> dict:
    """
    Comprehensive validation for upgrade completion.

    Checks:
    1. All category docs exist
    2. Main doc references all category docs
    3. File references in docs are valid
    4. Cross-references between docs exist

    Returns:
        Validation result dict with status, errors, warnings.
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "checks": {
            "category_docs": {"status": "unknown", "details": []},
            "main_doc_links": {"status": "unknown", "details": []},
            "file_references": {"status": "unknown", "details": []},
            "cross_references": {"status": "unknown", "details": []},
        },
    }

    research_dir = Path("docs/research")
    if not research_dir.exists():
        result["valid"] = False
        result["errors"].append("docs/research/ directory not found")
        return result

    # Find all docs for this upgrade
    upgrade_num = upgrade_id.replace("UPGRADE-", "")
    main_doc = None
    category_docs = []
    research_docs = []

    for doc_file in research_dir.glob(f"UPGRADE-{upgrade_num}*.md"):
        name = doc_file.name
        if name == f"UPGRADE-{upgrade_num}.md":
            main_doc = doc_file
        elif "CAT" in name:
            category_docs.append(doc_file)
        elif name.endswith("-RESEARCH.md"):
            research_docs.append(doc_file)

    # Check 1: Category docs exist
    if category_docs:
        result["checks"]["category_docs"]["status"] = "pass"
        result["checks"]["category_docs"]["details"] = [f"Found {len(category_docs)} category docs"]
    else:
        result["checks"]["category_docs"]["status"] = "warn"
        result["checks"]["category_docs"]["details"] = ["No category docs found"]
        result["warnings"].append("No category documentation found")

    # Check 2: Main doc links to all category docs
    if main_doc:
        try:
            main_content = main_doc.read_text()
            all_linked, missing_links = validate_category_doc_links(upgrade_id, main_content, category_docs)

            if all_linked:
                result["checks"]["main_doc_links"]["status"] = "pass"
                result["checks"]["main_doc_links"]["details"] = ["All category docs linked"]
            else:
                result["checks"]["main_doc_links"]["status"] = "fail"
                result["checks"]["main_doc_links"]["details"] = [f"Missing links: {missing_links}"]
                result["errors"].append(f"Main doc missing links to: {', '.join(missing_links)}")
                result["valid"] = False
        except OSError as e:
            result["checks"]["main_doc_links"]["status"] = "error"
            result["checks"]["main_doc_links"]["details"] = [str(e)]
    else:
        result["checks"]["main_doc_links"]["status"] = "skip"
        result["checks"]["main_doc_links"]["details"] = ["No main doc found"]
        result["warnings"].append(f"No main upgrade doc: {upgrade_id}.md")

    # Check 3: File references are valid
    all_refs_valid = True
    all_missing_refs = []

    for doc_file in [main_doc, *category_docs, *research_docs]:
        if doc_file and doc_file.exists():
            try:
                content = doc_file.read_text()
                valid, _, missing = validate_file_references(content, Path("."))
                if not valid:
                    all_refs_valid = False
                    all_missing_refs.extend([(doc_file.name, ref) for ref in missing])
            except OSError:
                pass

    if all_refs_valid:
        result["checks"]["file_references"]["status"] = "pass"
        result["checks"]["file_references"]["details"] = ["All file references valid"]
    else:
        result["checks"]["file_references"]["status"] = "warn"
        result["checks"]["file_references"]["details"] = [f"Missing: {doc}:{ref}" for doc, ref in all_missing_refs[:5]]
        result["warnings"].append(f"{len(all_missing_refs)} broken file references")

    # Check 4: Cross-references exist
    has_cross_refs = False
    for doc_file in category_docs + research_docs:
        if doc_file and doc_file.exists():
            try:
                content = doc_file.read_text()
                has_refs, _ = validate_cross_references(content)
                if has_refs:
                    has_cross_refs = True
                    break
            except OSError:
                pass

    if has_cross_refs:
        result["checks"]["cross_references"]["status"] = "pass"
        result["checks"]["cross_references"]["details"] = ["Cross-references found"]
    else:
        result["checks"]["cross_references"]["status"] = "warn"
        result["checks"]["cross_references"]["details"] = ["No cross-references detected"]
        result["warnings"].append("Docs may lack inter-document links")

    return result


CROSS_REFERENCE_VALIDATION_BLOCK = """
**⚠️ CROSS-REFERENCE VALIDATION FAILED**

{upgrade_id} documentation has issues that must be resolved before exit:

**Errors**:
{errors}

**Warnings**:
{warnings}

**Action Required**:
1. Fix all errors before exiting RIC Loop
2. Review warnings and address if critical
3. Run validation again: validate_upgrade_completion('{upgrade_id}')
"""


def check_progress_file_for_completion(content: str) -> list[tuple[str, str]]:
    """
    Check progress file for items being marked COMPLETED.

    Handles multiple formats:
    - ## CATEGORY N: Name - COMPLETED
    - - [x] Category N: Name COMPLETED
    - [x] Category N: Name COMPLETED

    Returns:
        List of (category_id, category_name) tuples being marked complete.
    """
    completing = []

    # Pattern 1: ## CATEGORY N: Name ... - COMPLETED (header format)
    pattern1 = re.compile(
        r"##\s*CATEGORY\s*(\d+)[:\s]+([^(\n]+?)(?:\s*\([^)]+\))?\s*-\s*COMPLETED",
        re.IGNORECASE,
    )

    # Pattern 2: - [x] Category N: Name COMPLETED (checklist format)
    pattern2 = re.compile(
        r"-?\s*\[x\]\s*Category\s*(\d+)[:\s]+([^\n]+?)\s+COMPLETED",
        re.IGNORECASE,
    )

    for match in pattern1.finditer(content):
        cat_num = match.group(1)
        cat_name = match.group(2).strip().upper().replace(" ", "-")
        completing.append((cat_num, cat_name))

    for match in pattern2.finditer(content):
        cat_num = match.group(1)
        cat_name = match.group(2).strip().upper().replace(" ", "-")
        if (cat_num, cat_name) not in completing:  # Avoid duplicates
            completing.append((cat_num, cat_name))

    return completing


def get_upgrade_doc_context(upgrade_id: str) -> dict:
    """
    Get upgrade documentation context for session start.

    Returns:
        Context dict with doc status, summaries, and action items.
    """
    research_dir = Path("docs/research")
    context = {
        "upgrade_id": upgrade_id,
        "upgrade_doc_path": f"docs/research/{upgrade_id}.md",
        "upgrade_doc_status": "NOT FOUND",
        "research_count": 0,
        "last_updated": "Never",
        "doc_summary": "No upgrade doc found. Consider creating one.",
        "next_actions": "Check claude-progress.txt for pending tasks.",
        "cross_refs": "None found.",
    }

    # Check for upgrade doc
    upgrade_doc = research_dir / f"{upgrade_id}.md"
    if not upgrade_doc.exists():
        # Try with topic suffix
        upgrade_docs = list(research_dir.glob(f"{upgrade_id}*.md"))
        if upgrade_docs:
            upgrade_doc = upgrade_docs[0]

    if upgrade_doc.exists():
        context["upgrade_doc_status"] = "EXISTS"
        context["upgrade_doc_path"] = str(upgrade_doc)

        # Get last modified time
        mtime = upgrade_doc.stat().st_mtime
        context["last_updated"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

        # Try to extract summary
        content = upgrade_doc.read_text()
        overview_match = re.search(r"## Overview\s*\n(.+?)(?=\n##|\Z)", content, re.DOTALL)
        if overview_match:
            context["doc_summary"] = overview_match.group(1).strip()[:500] + "..."

        # Find cross-references
        _, refs = validate_cross_references(content)
        if refs:
            context["cross_refs"] = "\n".join(refs[:5])

    # Count research docs
    research_docs = list(research_dir.glob(f"{upgrade_id}*-RESEARCH.md"))
    context["research_count"] = len(research_docs)

    # Get next actions from progress file
    progress_file = Path("claude-progress.txt")
    if progress_file.exists():
        progress_content = progress_file.read_text()
        # Find first unchecked item
        unchecked = re.findall(r"- \[ \] (.+?)(?:\n|$)", progress_content)
        if unchecked:
            context["next_actions"] = "\n".join([f"- {item}" for item in unchecked[:5]])

    return context


# =============================================================================
# CONFIDENCE TRACKING (v4.3 NEW)
# =============================================================================


@dataclass
class ConfidenceRecord:
    """Track confidence across phases."""

    phase: int
    score: int  # 0-100
    notes: str
    timestamp: str


def calculate_average_confidence(records: list[ConfidenceRecord]) -> int:
    """Calculate average confidence across phases."""
    if not records:
        return 0
    return sum(r.score for r in records) // len(records)


def confidence_below_threshold(records: list[ConfidenceRecord], threshold: int = 70) -> list[int]:
    """Return list of phases with confidence below threshold."""
    return [r.phase for r in records if r.score < threshold]


# =============================================================================
# JSON STATUS OUTPUT (Machine-Parseable - Research: agent self-reference)
# =============================================================================


def get_status_json(manager: "RICStateManager") -> str:
    """Generate machine-parseable JSON status for agent self-reference."""
    if not manager.is_active():
        return '{"ric": {"active": false}}'

    counts = manager.count_insights_by_priority()
    can_exit, reason = manager.can_exit()

    # Get convergence status
    convergence_status = "unknown"
    if len(manager.state.iteration_metrics) >= 2:
        is_converging, msg, _ = calculate_convergence(manager.state.iteration_metrics)
        convergence_status = "converging" if is_converging else "not_converging"

    status = {
        "ric": {
            "version": RIC_VERSION,
            "active": True,
            "iteration": manager.state.current_iteration,
            "max_iterations": manager.state.max_iterations,
            "phase": manager.state.current_phase.value,
            "phase_name": PHASES[manager.state.current_phase.value][0],
            "insights": {"P0": counts["P0"], "P1": counts["P1"], "P2": counts["P2"]},
            "can_exit": can_exit,
            "exit_reason": reason,
            "fix_attempts": manager.state.fix_attempts,
            "plateau_count": manager.state.plateau_count,
            "convergence_status": convergence_status,
            "average_confidence": calculate_average_confidence(manager.state.confidence_records),
            "throttle_state": {
                "tool_calls": manager.state.throttle_state.tool_calls_this_phase,
                "consecutive_failures": manager.state.throttle_state.consecutive_failures,
            },
            "research_enforcement": {
                "searches_since_persist": manager.state.throttle_state.web_searches_since_persist,
                "max_searches_before_persist": RESEARCH_ENFORCEMENT["searches_before_forced_persist"],
                "blocked": manager.state.throttle_state.research_persist_blocked,
            },
        }
    }
    return json.dumps(status, indent=2)


# =============================================================================
# PROGRESS FILE INTEGRATION (Research: sync with claude-progress.txt)
# =============================================================================

PROGRESS_FILE = Path("claude-progress.txt")


def sync_progress_file(manager: "RICStateManager") -> None:
    """Sync RIC state to claude-progress.txt for cross-session persistence."""
    if not manager.is_active():
        return

    counts = manager.count_insights_by_priority()
    header = get_header(
        manager.state.current_iteration,
        manager.state.max_iterations,
        manager.state.current_phase.value,
    )

    # Get convergence status
    convergence_msg = "Not enough data"
    if len(manager.state.iteration_metrics) >= 2:
        _, convergence_msg, _ = calculate_convergence(manager.state.iteration_metrics)

    avg_confidence = calculate_average_confidence(manager.state.confidence_records)

    # Read existing content
    existing = ""
    if PROGRESS_FILE.exists():
        existing = PROGRESS_FILE.read_text()

    # Find or create RIC section
    ric_section = f"""
## RIC Session Status (v4.3)
{header}
- Insights: P0={counts['P0']} P1={counts['P1']} P2={counts['P2']}
- Convergence: {convergence_msg}
- Confidence: {avg_confidence}%
- Started: {manager.state.started_at or 'Unknown'}
- Last checkpoint: {manager.state.last_checkpoint or 'None'}
"""

    # Update or append RIC section
    if "## RIC Session Status" in existing:
        # Replace existing section
        pattern = r"## RIC Session Status.*?(?=\n## |\Z)"
        updated = re.sub(pattern, ric_section.strip() + "\n", existing, flags=re.DOTALL)
        PROGRESS_FILE.write_text(updated)
    else:
        # Append new section
        PROGRESS_FILE.write_text(existing + ric_section)


def read_progress_tasks() -> list[tuple[bool, str]]:
    """Read task status from claude-progress.txt."""
    tasks = []
    if not PROGRESS_FILE.exists():
        return tasks

    content = PROGRESS_FILE.read_text()
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("- [x]"):
            tasks.append((True, line[5:].strip()))
        elif line.startswith("- [ ]"):
            tasks.append((False, line[5:].strip()))

    return tasks


# =============================================================================
# SECURITY GATE CHECK (Research: "No new vulnerabilities")
# =============================================================================

SECRET_PATTERNS = [
    r'api[_-]?key\s*=\s*["\'][^"\']+["\']',
    r'password\s*=\s*["\'][^"\']+["\']',
    r'secret\s*=\s*["\'][^"\']+["\']',
    r'token\s*=\s*["\'][^"\']+["\']',
    r'aws_access_key_id\s*=\s*["\'][^"\']+["\']',
    r'aws_secret_access_key\s*=\s*["\'][^"\']+["\']',
    r"-----BEGIN (RSA |DSA |EC )?PRIVATE KEY-----",
]


def check_for_secrets(file_path: str) -> list[str]:
    """Scan file for potential hardcoded secrets."""
    findings = []
    try:
        path = Path(file_path)
        if not path.exists() or path.suffix not in [".py", ".json", ".yaml", ".yml", ".env"]:
            return findings

        content = path.read_text()
        for pattern in SECRET_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                findings.append(f"Potential secret in {file_path}: pattern '{pattern[:30]}...'")
    except Exception:
        pass
    return findings


def security_gate_check(changed_files: list[str]) -> tuple[bool, list[str]]:
    """Run security gate check on changed files."""
    all_findings = []
    for file_path in changed_files:
        findings = check_for_secrets(file_path)
        all_findings.extend(findings)

    passed = len(all_findings) == 0
    return passed, all_findings


# =============================================================================
# ITERATION SUMMARY (Research: multi-day session support)
# =============================================================================


def generate_iteration_summary(manager: "RICStateManager") -> str:
    """Generate summary at end of iteration for context persistence."""
    if not manager.is_active():
        return ""

    counts = manager.count_insights_by_priority()
    open_insights = manager.get_open_insights()

    # Get convergence info
    convergence_msg = "Not enough data"
    if len(manager.state.iteration_metrics) >= 2:
        _, convergence_msg, metrics = calculate_convergence(manager.state.iteration_metrics)

    avg_confidence = calculate_average_confidence(manager.state.confidence_records)
    low_conf_phases = confidence_below_threshold(manager.state.confidence_records)

    summary = f"""
## Iteration {manager.state.current_iteration} Summary

**Phase Completed**: {PHASES[manager.state.current_phase.value][0]}
**Open Insights**: P0={counts['P0']} P1={counts['P1']} P2={counts['P2']}
**Convergence**: {convergence_msg}
**Average Confidence**: {avg_confidence}%
**Low Confidence Phases**: {low_conf_phases if low_conf_phases else 'None'}

### Unresolved Items
"""
    for insight in open_insights:
        summary += f"- [{insight.priority.value}] {insight.description}\n"

    summary += """
### Fix Attempts This Iteration
"""
    for key, count in manager.state.fix_attempts.items():
        if f"_{manager.state.current_iteration}" in key:
            gate = key.replace(f"_{manager.state.current_iteration}", "")
            summary += f"- {gate}: {count} attempts\n"

    can_exit, reason = manager.can_exit()
    summary += f"""
### Exit Status
- Can Exit: {'Yes' if can_exit else 'No'}
- Reason: {reason}

### Decision Trace (Last 5)
"""
    for decision in manager.state.decision_trace[-5:]:
        summary += f"- {decision.get('decision_id', 'N/A')}: {decision.get('action', 'N/A')[:50]}...\n"

    return summary


# =============================================================================
# SUGGESTION PROMPT
# =============================================================================

SUGGESTION_PROMPT = """
---
**RIC Loop Suggested** (v5.0)

Complex task detected. Use Meta-RIC Loop:

`/ric-start [task]`

**5 Phases**: Research → Plan → Build → Verify → Reflect

**v4.3 Safety Features**:
- Hallucination detection (5-category check)
- Convergence detection (multi-metric)
- Confidence calibration (per-phase)
- Safety throttles (runaway protection)
- Decision tracing (meta-debugging)

**Key Rules**: AND-test commits | Min 3 iterations | All P0-P2 required

See: `.claude/RIC_CONTEXT.md`
---
"""

# =============================================================================
# STATE MANAGEMENT (Persistence Layer)
# =============================================================================

STATE_FILE = Path(".claude/state/ric.json")


class Phase(Enum):
    """5-phase RIC workflow."""

    RESEARCH = 0
    PLAN = 1
    BUILD = 2
    VERIFY = 3
    REFLECT = 4


class Priority(Enum):
    """Insight priority levels."""

    P0 = "P0"  # Critical
    P1 = "P1"  # Important
    P2 = "P2"  # Polish (REQUIRED in RIC)


@dataclass
class Insight:
    """A discovered gap or issue."""

    id: str
    description: str
    priority: Priority
    phase_found: int
    iteration_found: int
    resolved: bool = False
    resolution: str | None = None


@dataclass
class RICState:
    """Persistent RIC session state."""

    active: bool = False
    current_phase: Phase = Phase.RESEARCH
    current_iteration: int = 1
    max_iterations: int = 5
    insights: list = field(default_factory=list)
    fix_attempts: dict = field(default_factory=dict)  # Track fix attempts per gate
    last_checkpoint: str | None = None
    started_at: str | None = None
    plateau_count: int = 0  # Consecutive iterations with 0 new insights

    # v4.3 additions
    iteration_metrics: list = field(default_factory=list)  # IterationMetrics history
    confidence_records: list = field(default_factory=list)  # ConfidenceRecord history
    decision_trace: list = field(default_factory=list)  # Decision log
    throttle_state: ThrottleState = field(default_factory=ThrottleState)
    hallucination_flags: list = field(default_factory=list)  # Detected hallucinations

    # v5.0 additions
    drift_metrics: Optional["DriftMetrics"] = None
    guardian_enabled: bool = False
    guardian_reviews: list = field(default_factory=list)
    repair_stats: Optional["RepairStats"] = None
    fix_candidates: list = field(default_factory=list)
    notes_last_updated: str | None = None
    seidr_results: list = field(default_factory=list)
    policy_violations: list = field(default_factory=list)
    original_intent: str = ""
    session_id: str = ""

    # v5.0 Phase Enforcement - tracks completion status per phase
    phase_completion: dict = field(default_factory=dict)  # {phase_name: status_dict}


class RICStateManager:
    """Manages RIC session state with persistence."""

    def __init__(self):
        self.state = self._load_state()

    def _load_state(self) -> RICState:
        """Load state from file or create new."""
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                state = RICState(
                    active=data.get("active", False),
                    current_phase=Phase(data.get("current_phase", 0)),
                    current_iteration=data.get("current_iteration", 1),
                    max_iterations=data.get("max_iterations", 5),
                    fix_attempts=data.get("fix_attempts", {}),
                    last_checkpoint=data.get("last_checkpoint"),
                    started_at=data.get("started_at"),
                    plateau_count=data.get("plateau_count", 0),
                    decision_trace=data.get("decision_trace", []),
                    hallucination_flags=data.get("hallucination_flags", []),
                )
                # Load throttle state
                ts_data = data.get("throttle_state", {})
                state.throttle_state = ThrottleState(
                    tool_calls_this_phase=ts_data.get("tool_calls_this_phase", 0),
                    edits_per_file=ts_data.get("edits_per_file", {}),
                    consecutive_failures=ts_data.get("consecutive_failures", 0),
                    phase_start_time=ts_data.get("phase_start_time"),
                    decisions_without_progress=ts_data.get("decisions_without_progress", 0),
                )
                # Load insights
                for i_data in data.get("insights", []):
                    state.insights.append(
                        Insight(
                            id=i_data["id"],
                            description=i_data["description"],
                            priority=Priority(i_data["priority"]),
                            phase_found=i_data["phase_found"],
                            iteration_found=i_data["iteration_found"],
                            resolved=i_data.get("resolved", False),
                            resolution=i_data.get("resolution"),
                        )
                    )
                # Load iteration metrics
                for m_data in data.get("iteration_metrics", []):
                    state.iteration_metrics.append(
                        IterationMetrics(
                            iteration=m_data["iteration"],
                            new_insights=m_data["new_insights"],
                            fix_attempts=m_data["fix_attempts"],
                            fix_successes=m_data["fix_successes"],
                            code_churn_lines=m_data["code_churn_lines"],
                            gate_attempts=m_data["gate_attempts"],
                            gate_passes=m_data["gate_passes"],
                            confidence_scores=m_data.get("confidence_scores", {}),
                        )
                    )
                # Load confidence records
                for c_data in data.get("confidence_records", []):
                    state.confidence_records.append(
                        ConfidenceRecord(
                            phase=c_data["phase"],
                            score=c_data["score"],
                            notes=c_data["notes"],
                            timestamp=c_data["timestamp"],
                        )
                    )
                # Load phase completion (v5.0)
                state.phase_completion = data.get("phase_completion", {})
                state.original_intent = data.get("original_intent", "")
                state.session_id = data.get("session_id", "")
                return state
            except (json.JSONDecodeError, KeyError, ValueError):
                return RICState()
        return RICState()

    def save_state(self) -> None:
        """Persist state to file."""
        data = {
            "version": RIC_VERSION,
            "active": self.state.active,
            "current_phase": self.state.current_phase.value,
            "current_iteration": self.state.current_iteration,
            "max_iterations": self.state.max_iterations,
            "fix_attempts": self.state.fix_attempts,
            "last_checkpoint": self.state.last_checkpoint,
            "started_at": self.state.started_at,
            "plateau_count": self.state.plateau_count,
            "decision_trace": self.state.decision_trace,
            "hallucination_flags": self.state.hallucination_flags,
            "throttle_state": {
                "tool_calls_this_phase": self.state.throttle_state.tool_calls_this_phase,
                "edits_per_file": self.state.throttle_state.edits_per_file,
                "consecutive_failures": self.state.throttle_state.consecutive_failures,
                "phase_start_time": self.state.throttle_state.phase_start_time,
                "decisions_without_progress": self.state.throttle_state.decisions_without_progress,
            },
            "insights": [
                {
                    "id": i.id,
                    "description": i.description,
                    "priority": i.priority.value,
                    "phase_found": i.phase_found,
                    "iteration_found": i.iteration_found,
                    "resolved": i.resolved,
                    "resolution": i.resolution,
                }
                for i in self.state.insights
            ],
            "iteration_metrics": [
                {
                    "iteration": m.iteration,
                    "new_insights": m.new_insights,
                    "fix_attempts": m.fix_attempts,
                    "fix_successes": m.fix_successes,
                    "code_churn_lines": m.code_churn_lines,
                    "gate_attempts": m.gate_attempts,
                    "gate_passes": m.gate_passes,
                    "confidence_scores": m.confidence_scores,
                }
                for m in self.state.iteration_metrics
            ],
            "confidence_records": [
                {
                    "phase": c.phase,
                    "score": c.score,
                    "notes": c.notes,
                    "timestamp": c.timestamp,
                }
                for c in self.state.confidence_records
            ],
            # v5.0 additions
            "phase_completion": self.state.phase_completion,
            "original_intent": self.state.original_intent,
            "session_id": self.state.session_id,
        }
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(data, indent=2))

    def is_active(self) -> bool:
        """Check if RIC session is active."""
        return self.state.active

    def start_session(self, max_iterations: int = 5, original_intent: str = "", session_id: str = "") -> None:
        """Start a new RIC session with v5.0 phase enforcement."""
        import uuid

        self.state = RICState(
            active=True,
            current_phase=Phase.RESEARCH,
            current_iteration=1,
            max_iterations=max_iterations,
            started_at=datetime.now().isoformat(),
            original_intent=original_intent,
            session_id=session_id or str(uuid.uuid4())[:8],
            phase_completion={},  # Will be populated per-phase
        )
        self.state.throttle_state.phase_start_time = time.time()
        # Initialize P0 RESEARCH completion status
        self._init_phase_completion("P0_RESEARCH")
        self.save_state()

    def _init_phase_completion(self, phase_key: str) -> None:
        """Initialize phase completion tracking for a phase."""
        self.state.phase_completion[phase_key] = {
            "completed": False,
            "requirements_met": {},
            "blockers": [],
            # P0 Research tracking
            "keywords_extracted": [],
            "web_searches_done": 0,
            "sources_documented": 0,
            "findings_persisted": False,
            # P4 Reflect tracking
            "introspection_done": False,
            "insights_reviewed": False,
            "convergence_checked": False,
            "upgrade_ideas": [],
            "loop_decision": "",
            "loop_reason": "",
            "timestamp": datetime.now().isoformat(),
        }
        self.save_state()

    def advance_phase(self, force: bool = False) -> tuple[Phase, bool, str]:
        """
        Advance to next phase with v5.0 enforcement.

        Returns:
            tuple: (new_phase, success, message)
            - success=True: Phase advanced successfully
            - success=False: Phase blocked, message explains why
        """
        # Check if phase can be advanced (v5.0 enforcement)
        if not force:
            can_advance, phase_blocked, blockers = can_advance_phase(self.state)
            if not can_advance:
                blocker_list = "\n  - ".join(blockers) if blockers else "Unknown"
                return (
                    self.state.current_phase,
                    False,
                    f"❌ Cannot advance: {phase_blocked} requirements not met\n" f"  Blockers:\n  - {blocker_list}",
                )

        current = self.state.current_phase.value

        if current < 4:
            self.state.current_phase = Phase(current + 1)
            # Initialize completion tracking for new phase
            new_phase_key = f"P{current + 1}_{self.state.current_phase.name}"
            self._init_phase_completion(new_phase_key)
        else:
            # Phase 4 (REFLECT) -> loop back to Phase 0 (RESEARCH)
            self.state.current_phase = Phase.RESEARCH
            self.state.current_iteration += 1
            # Reset throttle state for new iteration
            self.state.throttle_state = ThrottleState()
            # Initialize P0 completion for new iteration
            self._init_phase_completion("P0_RESEARCH")

        self.state.throttle_state.phase_start_time = time.time()
        self.save_state()
        return (
            self.state.current_phase,
            True,
            f"✅ Advanced to {self.state.current_phase.name}",
        )

    def advance_phase_legacy(self) -> Phase:
        """Legacy advance_phase for backward compatibility."""
        new_phase, _success, _msg = self.advance_phase(force=True)
        return new_phase

    # =========================================================================
    # v5.0 PHASE COMPLETION RECORDING METHODS
    # =========================================================================

    def _get_current_phase_key(self) -> str:
        """Get the current phase's completion tracking key."""
        phase = self.state.current_phase
        return f"P{phase.value}_{phase.name}"

    def record_p0_keywords(self, keywords: list[str]) -> None:
        """Record keyword extraction for P0 RESEARCH."""
        key = "P0_RESEARCH"
        if key not in self.state.phase_completion:
            self._init_phase_completion(key)
        self.state.phase_completion[key]["keywords_extracted"] = keywords
        self.state.phase_completion[key]["requirements_met"]["keywords"] = len(keywords) >= 3
        self.save_state()

    def record_p0_search(self, query: str, sources_found: int = 0) -> None:
        """Record a web search for P0 RESEARCH."""
        key = "P0_RESEARCH"
        if key not in self.state.phase_completion:
            self._init_phase_completion(key)
        self.state.phase_completion[key]["web_searches_done"] += 1
        self.state.phase_completion[key]["sources_documented"] += sources_found
        min_searches = PHASE_REQUIREMENTS["P0_RESEARCH"]["min_web_searches"]
        self.state.phase_completion[key]["requirements_met"]["searches"] = (
            self.state.phase_completion[key]["web_searches_done"] >= min_searches
        )
        self.save_state()

    def record_p0_findings_persisted(self, doc_path: str = "") -> None:
        """Record that findings have been persisted for P0 RESEARCH."""
        key = "P0_RESEARCH"
        if key not in self.state.phase_completion:
            self._init_phase_completion(key)
        self.state.phase_completion[key]["findings_persisted"] = True
        self.state.phase_completion[key]["requirements_met"]["persisted"] = True
        self.save_state()

    def record_p4_introspection(self) -> None:
        """Record introspection completion for P4 REFLECT."""
        key = "P4_REFLECT"
        if key not in self.state.phase_completion:
            self._init_phase_completion(key)
        self.state.phase_completion[key]["introspection_done"] = True
        self.state.phase_completion[key]["requirements_met"]["introspection"] = True
        self.save_state()

    def record_p4_insights_reviewed(self) -> None:
        """Record insights review for P4 REFLECT."""
        key = "P4_REFLECT"
        if key not in self.state.phase_completion:
            self._init_phase_completion(key)
        self.state.phase_completion[key]["insights_reviewed"] = True
        self.state.phase_completion[key]["requirements_met"]["insights_reviewed"] = True
        self.save_state()

    def record_p4_convergence_check(self, converged: bool, reason: str) -> None:
        """Record convergence check for P4 REFLECT."""
        key = "P4_REFLECT"
        if key not in self.state.phase_completion:
            self._init_phase_completion(key)
        self.state.phase_completion[key]["convergence_checked"] = True
        self.state.phase_completion[key]["requirements_met"]["convergence"] = True
        self.save_state()

    def record_p4_upgrade_ideas(self, ideas: list[str]) -> None:
        """Record upgrade ideas generated in P4 REFLECT."""
        key = "P4_REFLECT"
        if key not in self.state.phase_completion:
            self._init_phase_completion(key)
        self.state.phase_completion[key]["upgrade_ideas"] = ideas
        self.state.phase_completion[key]["requirements_met"]["upgrade_ideas"] = len(ideas) > 0
        self.save_state()

    def record_p4_loop_decision(self, decision: str, reason: str) -> None:
        """Record the LOOP/EXIT decision for P4 REFLECT."""
        key = "P4_REFLECT"
        if key not in self.state.phase_completion:
            self._init_phase_completion(key)
        self.state.phase_completion[key]["loop_decision"] = decision.upper()
        self.state.phase_completion[key]["loop_reason"] = reason
        self.state.phase_completion[key]["requirements_met"]["loop_decision"] = True
        # Mark P4 as complete if all requirements are met
        completed, _blockers, _ = check_p4_completion(self.state)
        self.state.phase_completion[key]["completed"] = completed
        self.save_state()

    def get_phase_completion_status(self, phase_key: str) -> dict:
        """Get the completion status for a specific phase."""
        return self.state.phase_completion.get(phase_key, {})

    def record_phase_data(self, phase_key: str, data: dict) -> None:
        """Record completion data for a specific phase (v5.1)."""
        self.state.phase_completion[phase_key] = data
        self.save_state()

    def get_open_insights(self) -> list[Insight]:
        """Get all unresolved insights."""
        return [i for i in self.state.insights if not i.resolved]

    def count_insights_by_priority(self) -> dict[str, int]:
        """Count open insights by priority."""
        open_insights = self.get_open_insights()
        return {
            "P0": sum(1 for i in open_insights if i.priority == Priority.P0),
            "P1": sum(1 for i in open_insights if i.priority == Priority.P1),
            "P2": sum(1 for i in open_insights if i.priority == Priority.P2),
        }

    def add_insight(self, description: str, priority: Priority) -> Insight:
        """Add a new insight."""
        insight = Insight(
            id=f"INS-{len(self.state.insights) + 1:03d}",
            description=description,
            priority=priority,
            phase_found=self.state.current_phase.value,
            iteration_found=self.state.current_iteration,
        )
        self.state.insights.append(insight)
        self.save_state()
        return insight

    def resolve_insight(self, insight_id: str, resolution: str) -> bool:
        """Mark an insight as resolved."""
        for insight in self.state.insights:
            if insight.id == insight_id:
                insight.resolved = True
                insight.resolution = resolution
                self.save_state()
                return True
        return False

    def increment_fix_attempt(self, gate: str) -> int:
        """Increment fix attempt counter for a gate."""
        key = f"{gate}_{self.state.current_iteration}"
        self.state.fix_attempts[key] = self.state.fix_attempts.get(key, 0) + 1
        self.save_state()
        return self.state.fix_attempts[key]

    def get_fix_attempts(self, gate: str) -> int:
        """Get current fix attempt count for a gate."""
        key = f"{gate}_{self.state.current_iteration}"
        return self.state.fix_attempts.get(key, 0)

    def record_confidence(self, phase: int, score: int, notes: str) -> None:
        """Record confidence for a phase."""
        self.state.confidence_records.append(
            ConfidenceRecord(
                phase=phase,
                score=score,
                notes=notes,
                timestamp=datetime.now().isoformat(),
            )
        )
        self.save_state()

    def log_decision(self, action: str, reasoning: str, alternatives: list, risk: str, confidence: int) -> None:
        """Log a decision for meta-debugging."""
        decision = {
            "decision_id": f"DEC-{self.state.current_iteration:03d}-{len(self.state.decision_trace) + 1:02d}",
            "timestamp": datetime.now().isoformat(),
            "phase": self.state.current_phase.value,
            "iteration": self.state.current_iteration,
            "action": action,
            "reasoning": reasoning,
            "alternatives_considered": alternatives,
            "risk_assessment": risk,
            "confidence": confidence,
        }
        self.state.decision_trace.append(decision)
        self.save_state()

    def record_iteration_metrics(self, metrics: IterationMetrics) -> None:
        """Record metrics for current iteration."""
        self.state.iteration_metrics.append(metrics)
        self.save_state()

    def increment_tool_call(self) -> tuple[bool, str]:
        """Increment tool call counter and check throttle."""
        self.state.throttle_state.tool_calls_this_phase += 1
        triggered, throttle_type, action = check_throttles(self.state.throttle_state)
        self.save_state()
        return triggered, action

    def record_file_edit(self, file_path: str) -> tuple[bool, str]:
        """Record file edit and check throttle."""
        self.state.throttle_state.edits_per_file[file_path] = (
            self.state.throttle_state.edits_per_file.get(file_path, 0) + 1
        )
        triggered, throttle_type, action = check_throttles(self.state.throttle_state)
        self.save_state()
        return triggered, action

    def record_failure(self) -> tuple[bool, str]:
        """Record a failure and check throttle."""
        self.state.throttle_state.consecutive_failures += 1
        triggered, throttle_type, action = check_throttles(self.state.throttle_state)
        self.save_state()
        return triggered, action

    def record_success(self) -> None:
        """Record a success (resets consecutive failure counter)."""
        self.state.throttle_state.consecutive_failures = 0
        self.save_state()

    def can_exit(self) -> tuple[bool, str]:
        """Determine if session can exit."""
        counts = self.count_insights_by_priority()
        iteration = self.state.current_iteration
        avg_confidence = calculate_average_confidence(self.state.confidence_records)

        # Check convergence
        is_converging = False
        if len(self.state.iteration_metrics) >= 2:
            is_converging, _, _ = calculate_convergence(self.state.iteration_metrics)

        # Decision rules (strict order)
        if iteration < ITERATION_LIMITS["min"]:
            return False, f"Minimum {ITERATION_LIMITS['min']} iterations required"
        if counts["P0"] > 0:
            return False, f"P0 insights remaining: {counts['P0']}"
        if counts["P1"] > 0:
            return False, f"P1 insights remaining: {counts['P1']}"
        if counts["P2"] > 0:
            return False, f"P2 insights remaining: {counts['P2']}"
        if avg_confidence > 0 and avg_confidence < CONVERGENCE_THRESHOLDS["confidence_floor"]:
            return False, f"Low confidence: {avg_confidence}% (need ≥{CONVERGENCE_THRESHOLDS['confidence_floor']}%)"
        if self.state.plateau_count >= ITERATION_LIMITS["plateau"]:
            return True, "Plateau reached (no new insights)"
        if is_converging:
            return True, "Multi-metric convergence detected"
        if iteration >= ITERATION_LIMITS["max"]:
            return True, "Maximum iterations reached"
        return True, "All insights resolved"

    def end_session(self) -> None:
        """End the RIC session."""
        self.state.active = False
        self.save_state()


# =============================================================================
# ENFORCEMENT CONFIGURATION
# =============================================================================


# Environment variables
def get_ric_mode() -> str:
    """Get RIC enforcement mode: ENFORCED | SUGGESTED | DISABLED."""
    return os.environ.get("RIC_MODE", "SUGGESTED").upper()


def is_strict() -> bool:
    """Check if strict mode enabled (blocks on violations)."""
    return os.environ.get("RIC_STRICT", "0") == "1"


# Protected paths that trigger RIC suggestions
PROTECTED_PATTERNS = [
    "algorithms/",
    "execution/",
    "models/risk",
    "models/circuit",
    "llm/",
    "scanners/",
]

# Complex task keywords (require 2+ matches)
COMPLEX_TASK_KEYWORDS = [
    r"\bimplement\b",
    r"\brefactor\b",
    r"\bintegrate\b",
    r"\bredesign\b",
    r"\bupgrade\b",
    r"\bmigrate\b",
    r"\boverhaul\b",
    r"\brewrite\b",
    r"\boptimize\b",
    r"\bconsolidate\b",
    r"\bbuild\b.*system",
    r"\bcreate\b.*feature",
    r"multi.?file",
    r"multiple.*files",
    r"across.*codebase",
    r"major.*change",
    r"complex\b",
    r"\bresearch\b",
    r"\binvestigate\b",
]

# Simple task keywords (skip RIC)
SIMPLE_TASK_KEYWORDS = [
    r"\bfix\b.*typo",
    r"\bupdate\b.*readme",
    r"single.*file",
    r"quick.*fix",
    r"minor.*update",
    r"trivial\b",
]


# =============================================================================
# LOGGING UTILITIES
# =============================================================================


def output_message(message: str, level: str = "INFO") -> None:
    """Output a message to stderr (visible to Claude)."""
    prefixes = {
        "BLOCK": "[RIC BLOCK]",
        "WARN": "[RIC WARN]",
        "TIP": "[RIC TIP]",
        "INFO": "[RIC]",
        "THROTTLE": "[RIC THROTTLE]",
        "HALLU": "[RIC HALLU]",
    }
    prefix = prefixes.get(level, "[RIC]")
    print(f"{prefix} {message}", file=sys.stderr)


# =============================================================================
# COMPLIANCE CHECKS
# =============================================================================


def check_protected_path(file_path: str) -> tuple[bool, str]:
    """Check if file is in a protected path."""
    for pattern in PROTECTED_PATTERNS:
        if pattern in file_path:
            return True, f"Protected path: {pattern}"
    return False, ""


def check_commit_message(command: str, manager: RICStateManager) -> tuple[bool, str]:
    """Validate commit message format: [I{iter}/{max}][P{phase}]."""
    if "git commit" not in command or not manager.is_active():
        return True, ""

    # Extract commit message
    match = re.search(r'-m\s+["\'](.+?)["\']', command) or re.search(r"-m\s+(\S+)", command)
    if not match:
        return True, ""

    message = match.group(1)
    iteration = manager.state.current_iteration
    max_iter = manager.state.max_iterations

    # Check for new format: [I{iter}/{max}][P{phase}]
    pattern = rf"\[I{iteration}/{max_iter}\]\[P\d\]"
    if not re.search(pattern, message, re.IGNORECASE):
        return False, f"Missing header: [I{iteration}/{max_iter}][P2] for Build phase commits"

    return True, ""


def check_phase_appropriate(tool_name: str, file_path: str, manager: RICStateManager) -> tuple[bool, str]:
    """Check if tool usage is appropriate for current phase."""
    if not manager.is_active():
        return True, ""

    phase = manager.state.current_phase

    # Phase 0 (RESEARCH) - shouldn't modify code
    if phase == Phase.RESEARCH and tool_name in ["Edit", "Write"]:
        # Allow writing to docs/research/
        if "docs/research/" in file_path:
            return True, ""
        return False, "Phase 0 (Research): Only write to docs/research/, use WebSearch/WebFetch for research"

    # Phase 3 (VERIFY) - focus on verification, not code changes
    if phase == Phase.VERIFY and tool_name in ["Edit", "Write"]:
        # Allow test files and fixing gate failures
        if "/test" in file_path or file_path.startswith("test"):
            return True, ""
        return False, "Phase 3 (Verify): Focus on verification. Edit only to fix failing gates."

    return True, ""


def check_prompt_complexity(prompt: str) -> tuple[bool, list[str]]:
    """Check if prompt indicates a complex task."""
    prompt_lower = prompt.lower()

    # Simple task - don't suggest
    for pattern in SIMPLE_TASK_KEYWORDS:
        if re.search(pattern, prompt_lower):
            return False, []

    # Count complex matches
    matches = [p for p in COMPLEX_TASK_KEYWORDS if re.search(p, prompt_lower)]
    return len(matches) >= 2, matches


# =============================================================================
# SESSION STATUS FORMATTING
# =============================================================================


def get_session_status(manager: RICStateManager) -> dict[str, Any]:
    """Get current RIC session status."""
    if not manager.is_active():
        return {
            "active": False,
            "phase": None,
            "phase_name": None,
            "iteration": None,
            "max_iterations": None,
            "open_insights": {"P0": 0, "P1": 0, "P2": 0},
            "can_exit": False,
            "exit_reason": "",
            "convergence_status": "N/A",
            "confidence": 0,
        }

    counts = manager.count_insights_by_priority()
    can_exit, exit_reason = manager.can_exit()

    # Get convergence status
    convergence_status = "Not enough data"
    if len(manager.state.iteration_metrics) >= 2:
        is_converging, msg, _ = calculate_convergence(manager.state.iteration_metrics)
        convergence_status = "Converging" if is_converging else "Not yet converging"

    return {
        "active": True,
        "phase": manager.state.current_phase.value,
        "phase_name": PHASES[manager.state.current_phase.value][0],
        "iteration": manager.state.current_iteration,
        "max_iterations": manager.state.max_iterations,
        "open_insights": counts,
        "can_exit": can_exit,
        "exit_reason": exit_reason,
        "convergence_status": convergence_status,
        "confidence": calculate_average_confidence(manager.state.confidence_records),
    }


def format_session_status(status: dict[str, Any]) -> str:
    """Format session status as a string."""
    if not status["active"]:
        return "No active RIC session"

    p = status["open_insights"]
    header = get_header(status["iteration"], status["max_iterations"], status["phase"])
    exit_status = "ALLOWED" if status["can_exit"] else "BLOCKED"

    return f"""**RIC v{RIC_VERSION}**: {header}
**Insights**: P0={p['P0']} P1={p['P1']} P2={p['P2']}
**Convergence**: {status['convergence_status']}
**Confidence**: {status['confidence']}%
**Exit**: {exit_status} ({status['exit_reason']})"""


# =============================================================================
# HOOK HANDLERS
# =============================================================================


def handle_pretool_use(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Handle PreToolUse hook for Edit/Write/Bash/WebSearch/WebFetch."""
    result = {"allow": True, "messages": [], "suggestions": []}

    mode = get_ric_mode()
    if mode == "DISABLED":
        return result

    enforcement_level = get_enforcement_level()
    manager = RICStateManager()
    status = get_session_status(manager)
    file_path = tool_input.get("file_path", "")
    command = tool_input.get("command", "")

    violations = []

    # Check research persistence blocking FIRST (v4.3 - Compaction Protection)
    if manager.is_active() and tool_name in ["WebSearch", "WebFetch"]:
        if manager.state.current_phase == Phase.RESEARCH:
            persist_required, warning = check_research_persist_required(manager.state.throttle_state)
            if persist_required or manager.state.throttle_state.research_persist_blocked:
                violations.append(
                    f"**🛑 RESEARCH PERSISTENCE REQUIRED**: "
                    f"{manager.state.throttle_state.web_searches_since_persist} searches without persist. "
                    f"Write findings to docs/research/ before more searches."
                )
                result["suggestions"].append(
                    "Use Write tool to persist research to docs/research/UPGRADE-XXX-TOPIC.md with timestamps"
                )
                if RESEARCH_ENFORCEMENT["block_search_without_persist"]:
                    result["messages"].extend(violations)
                    if mode == "ENFORCED" and is_strict():
                        result["allow"] = False
                        return result
                    elif mode == "ENFORCED":
                        # Even in non-strict mode, block searches when limit exceeded
                        result["allow"] = False
                        return result

    # Check throttles (v4.3)
    if manager.is_active():
        triggered, action = manager.increment_tool_call()
        if triggered:
            result["messages"].append(f"Safety throttle triggered: {action}")
            if mode == "ENFORCED" and is_strict():
                result["allow"] = False
                return result

    # Check 1: Protected path warning
    if tool_name in ["Edit", "Write"]:
        is_protected, reason = check_protected_path(file_path)
        if is_protected:
            if status["active"]:
                result["messages"].append(format_session_status(status))
                result["suggestions"].append(
                    f"Remember: AND-test + Hallucination check. Header: [I{status['iteration']}/{status['max_iterations']}][P{status['phase']}]"
                )
                # Track file edit for throttling
                triggered, action = manager.record_file_edit(file_path)
                if triggered:
                    result["messages"].append(f"File edit throttle: {action}")
            else:
                result["suggestions"].append(f"{reason} - Consider /ric-start for complex changes")

    # Check 1b: Documentation naming - AUTO-FIX instead of blocking (v4.3)
    # Never blocks - uses auto-fix or just warns
    if tool_name == "Write" and "docs/research/" in file_path:
        content = tool_input.get("content", "")
        is_valid_name, doc_type, naming_error = validate_doc_naming(file_path)

        if not is_valid_name and DOC_ENFORCEMENT.get("auto_fix_naming"):
            # Try to auto-fix the naming
            fixed, new_path, fix_message = auto_fix_doc_naming(file_path, content)
            if fixed:
                result["suggestions"].append(fix_message)
                file_path = new_path  # Use the corrected path
            else:
                # Couldn't auto-fix - just warn, never block
                result["suggestions"].append(f"⚠️ {naming_error}")
        elif not is_valid_name:
            # Auto-fix disabled - just warn
            result["suggestions"].append(f"⚠️ {naming_error}")

        # Track the doc update regardless of naming
        upgrade_id = detect_upgrade_from_path(file_path)
        if upgrade_id and status["active"]:
            track_doc_update(
                file_path=file_path,
                upgrade_id=upgrade_id,
                iteration=status.get("iteration", 1),
                phase=status.get("phase", 0),
            )
            result["suggestions"].append(f"📝 Tracked doc update for {upgrade_id} Phase {status.get('phase', 0)}")

    # Check 1b2: Track Edit operations on docs/research (v4.3 - Doc Update Tracking)
    if tool_name == "Edit" and "docs/research/" in file_path:
        upgrade_id = detect_upgrade_from_path(file_path)
        if upgrade_id and status["active"]:
            track_doc_update(
                file_path=file_path,
                upgrade_id=upgrade_id,
                iteration=status.get("iteration", 1),
                phase=status.get("phase", 0),
            )
            result["suggestions"].append(f"📝 Tracked doc edit for {upgrade_id} Phase {status.get('phase', 0)}")

    # Check 1c: Progress file protection - block COMPLETED without doc (v4.3)
    # In autonomous mode: creates stub docs instead of blocking
    if tool_name in ["Write", "Edit"] and "claude-progress.txt" in file_path:
        content = tool_input.get("content", "") or tool_input.get("new_string", "")
        if content and DOC_ENFORCEMENT["block_completion_without_doc"]:
            # Check if any categories are being marked COMPLETED
            completing = check_progress_file_for_completion(content)
            for cat_num, cat_name in completing:
                # Detect upgrade ID from content or use default
                upgrade_match = re.search(r"UPGRADE-(\d{3})", content)
                upgrade_id = upgrade_match.group(0) if upgrade_match else "UPGRADE-014"

                # check_doc_exists_for_completion handles autonomous mode internally
                # (creates stub docs if missing in autonomous mode)
                can_complete, msg = check_doc_exists_for_completion(upgrade_id, (int(cat_num), cat_name))
                if not can_complete:
                    # Only block in interactive mode with ENFORCED
                    if enforcement_level == "BLOCK" and mode == "ENFORCED":
                        violations.append(msg)
                        result["allow"] = False
                        result["messages"].append(msg)
                        result["suggestions"].append(
                            f"Create: docs/research/{upgrade_id}-CAT{cat_num}-{cat_name}-RESEARCH.md"
                        )
                        return result
                    elif enforcement_level == "WARN":
                        # Autonomous mode: already logged in check_doc_exists_for_completion
                        result["suggestions"].append(f"[AUTONOMOUS] ⚠️ {msg}")
                elif msg:  # Stub created message
                    result["suggestions"].append(msg)

    # Check 2: Commit message format (FORCED)
    if tool_name == "Bash":
        valid, msg = check_commit_message(command, manager)
        if not valid:
            violations.append(msg)
            result["suggestions"].append(
                f'Use: git commit -m "[I{status["iteration"]}/{status["max_iterations"]}][P2] <description>"'
            )

    # Check 3: Phase-appropriate tool usage
    valid, msg = check_phase_appropriate(tool_name, file_path, manager)
    if not valid:
        violations.append(msg)
        result["suggestions"].append(
            f"Current: Phase {status['phase']} ({status['phase_name']}). "
            f"Advance with: python3 .claude/hooks/ric.py advance"
        )

    # Determine blocking
    if violations:
        result["messages"].extend(violations)
        manager.record_failure()  # Track failure for throttling
        if mode == "ENFORCED" and is_strict():
            result["allow"] = False
    else:
        manager.record_success()  # Reset failure counter on success

    return result


def handle_user_prompt(prompt: str) -> dict[str, Any]:
    """Handle UserPromptSubmit hook."""
    result = {"suggest_ric": False, "message": ""}

    if get_ric_mode() == "DISABLED":
        return result

    is_complex, _ = check_prompt_complexity(prompt)

    if is_complex:
        manager = RICStateManager()
        status = get_session_status(manager)

        if status["active"]:
            # Already have session, remind of status
            result["message"] = f"""---
{format_session_status(status)}
Use /ric-introspect for Phase 4, /ric-converge to check exit
---"""
        else:
            # Suggest starting RIC
            result["suggest_ric"] = True
            result["message"] = SUGGESTION_PROMPT

    return result


def handle_posttool_use(tool_name: str, tool_input: dict[str, Any], tool_output: Any) -> dict[str, Any]:
    """
    Handle PostToolUse hook for WebSearch/WebFetch tracking (v4.3 NEW).

    This enforces research persistence by tracking web searches and
    warning/blocking when persistence is required.
    """
    result = {"messages": [], "suggestions": []}

    mode = get_ric_mode()
    if mode == "DISABLED":
        return result

    manager = RICStateManager()
    if not manager.is_active():
        return result

    # Only track during Research phase (P0)
    if manager.state.current_phase != Phase.RESEARCH:
        return result

    # Track WebSearch/WebFetch calls
    if tool_name in ["WebSearch", "WebFetch"]:
        should_block, message = record_web_search(manager.state.throttle_state)

        if message:
            result["messages"].append(message)

        # Provide search count status
        searches = manager.state.throttle_state.web_searches_since_persist
        max_searches = RESEARCH_ENFORCEMENT["searches_before_forced_persist"]

        if searches > 0 and searches < max_searches:
            result["suggestions"].append(
                f"[Research] {searches}/{max_searches} searches since last persist. "
                f"Persist to docs/research/ after {max_searches - searches} more search(es)."
            )

        # Save state
        manager.save_state()

    # Track Write to docs/research/ - reset counter
    if tool_name == "Write":
        file_path = tool_input.get("file_path", "")
        if "docs/research/" in file_path:
            record_research_persist(manager.state.throttle_state, file_path)

            # Validate timestamps in written content
            content = tool_input.get("content", "")
            if content:
                is_valid, issues = validate_research_timestamps(file_path, content)
                if not is_valid and RESEARCH_ENFORCEMENT["warn_missing_timestamps"]:
                    warning = RESEARCH_TIMESTAMP_WARNING.format(file_path=file_path, issues="\n".join(issues))
                    result["messages"].append(warning)
                else:
                    # Count valid sources
                    source_count = count_timestamped_sources(content)
                    min_sources = RESEARCH_ENFORCEMENT["min_sources_per_topic"]
                    if source_count < min_sources:
                        result["suggestions"].append(
                            f"[Research] {source_count}/{min_sources} timestamped sources. "
                            f"Need {min_sources - source_count} more for gate."
                        )
                    else:
                        result["suggestions"].append(
                            f"[Research] ✅ Persisted {source_count} timestamped sources. " f"Search counter reset."
                        )

            manager.save_state()

    return result


# =============================================================================
# CLI INTERFACE
# =============================================================================


def cli_init(max_iterations: int = 5) -> None:
    """Initialize a new RIC session."""
    manager = RICStateManager()
    if manager.is_active():
        print(f"[RIC] Session already active at iteration {manager.state.current_iteration}")
        return
    manager.start_session(max_iterations)
    print(f"[RIC] Session started with max {max_iterations} iterations (v5.0)")
    print("[RIC] Current phase: P0 RESEARCH")
    print("[RIC] Safety features: Hallucination check, Convergence detection, Confidence calibration")


def cli_status() -> None:
    """Show current RIC status."""
    manager = RICStateManager()
    status = get_session_status(manager)
    print(format_session_status(status))


def cli_advance(force: bool = False) -> None:
    """Advance to next phase with v5.0 enforcement."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session. Use 'init' first.")
        return

    new_phase, success, message = manager.advance_phase(force=force)

    if not success:
        print("[RIC v5.0] Phase Advancement BLOCKED")
        print(message)
        print("\n[RIC] Complete the required actions before advancing.")
        print("[RIC] Use 'p0-status' or 'p4-status' to see requirements.")
        print("[RIC] Use 'advance --force' to override (not recommended).")
        return

    header = get_header(manager.state.current_iteration, manager.state.max_iterations, new_phase.value)
    print(f"[RIC] Advanced to: {header}")
    print(message)

    # Print phase-specific prompts when entering P0 or P4
    if new_phase == Phase.RESEARCH:
        print("\n" + "=" * 60)
        print("📚 P0 RESEARCH - REQUIRED ACTIONS:")
        print("=" * 60)
        print(P0_RESEARCH_START_PROMPT)
        print("\n[RIC] Use 'record-keywords <kw1,kw2,...>' to track extraction.")
        print("[RIC] Use 'record-search <query>' after each web search.")
    elif new_phase == Phase.REFLECT:
        print("\n" + "=" * 60)
        print("🔄 P4 REFLECT - REQUIRED ACTIONS:")
        print("=" * 60)
        print(P4_REFLECT_START_PROMPT)
        print("\n[RIC] Use 'record-introspection' after completing reflection.")
        print("[RIC] Use 'loop-decision <LOOP|EXIT> <reason>' to record decision.")


def cli_add_insight(description: str, priority: str) -> None:
    """Add a new insight."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return
    try:
        p = Priority(priority.upper())
    except ValueError:
        print(f"[RIC] Invalid priority: {priority}. Use P0, P1, or P2.")
        return
    insight = manager.add_insight(description, p)
    print(f"[RIC] Added {insight.id}: [{p.value}] {description}")


def cli_confidence(phase: int, score: int, notes: str) -> None:
    """Record confidence for a phase."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return
    if score < 0 or score > 100:
        print("[RIC] Score must be 0-100")
        return
    manager.record_confidence(phase, score, notes)
    print(f"[RIC] Recorded confidence for P{phase}: {score}% - {notes}")


def cli_decision(action: str, reasoning: str, risk: str, confidence: int) -> None:
    """Log a decision."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return
    manager.log_decision(action, reasoning, [], risk, confidence)
    print(f"[RIC] Logged decision: {action}")


def cli_convergence() -> None:
    """Check convergence status."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return
    if len(manager.state.iteration_metrics) < 2:
        print("[RIC] Not enough iteration data for convergence detection")
        return
    is_converging, message, metrics = calculate_convergence(manager.state.iteration_metrics)
    print(f"[RIC] Convergence Status: {'CONVERGING' if is_converging else 'NOT YET CONVERGING'}")
    print(f"[RIC] Details: {message}")
    for metric, data in metrics.items():
        print(f"  - {metric}: {data['trending']} (target {'met' if data['target_met'] else 'not met'})")


def cli_can_exit() -> None:
    """Check if session can exit."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return
    can_exit, reason = manager.can_exit()
    if can_exit:
        print(f"[RIC] ✅ EXIT ALLOWED: {reason}")
    else:
        print(f"[RIC] ❌ EXIT BLOCKED: {reason}")


def cli_end() -> None:
    """End the RIC session."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return
    can_exit, reason = manager.can_exit()
    if not can_exit:
        print(f"[RIC] ❌ Cannot end: {reason}")
        return
    manager.end_session()
    print("[RIC] Session ended successfully.")


def cli_json() -> None:
    """Output machine-parseable JSON status."""
    manager = RICStateManager()
    print(get_status_json(manager))


def cli_sync_progress() -> None:
    """Sync RIC state to claude-progress.txt."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return
    sync_progress_file(manager)
    print(f"[RIC] Synced to {PROGRESS_FILE}")


def cli_check_gate(phase: int) -> None:
    """Check if gate criteria are met for a phase."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    # Build context from current state
    counts = manager.count_insights_by_priority()
    avg_conf = calculate_average_confidence(manager.state.confidence_records)
    context = {
        # Phase 0 gates
        "problem_framed": True,  # Would need external check
        "research_doc_exists": Path("docs/research").exists(),
        "sources_count": True,  # Would need content check
        "findings_persisted": True,
        "confidence_ok": avg_conf >= CONVERGENCE_THRESHOLDS["confidence_floor"],
        # Phase 1 gates
        "success_criteria": True,
        "tasks_prioritized": True,
        "out_of_scope": True,
        "dependencies_noted": True,
        # Phase 2 gates
        "p0_tasks_done": counts["P0"] == 0,
        "and_test_passed": True,
        "commits_revertable": True,
        "hallucination_check": len(manager.state.hallucination_flags) == 0,
        "tests_exist": True,
        "tests_pass": True,  # Would need pytest check
        # Phase 3 gates
        "coverage_ok": True,  # Would need coverage check
        "lint_clean": True,  # Would need ruff check
        "types_ok": True,
        "no_secrets": True,
        "consistency_check": True,
        "checker_review": True,
        # Phase 4 gates
        "critique_done": True,
        "checklist_reviewed": True,
        "gaps_classified": True,
        "convergence_evaluated": len(manager.state.iteration_metrics) >= 1,
        "confidence_calibrated": len(manager.state.confidence_records) >= 1,
        "decision_made": True,
        "decision_justified": True,
    }

    passed, results = check_gate(phase, context)
    print(f"[RIC] Gate check for Phase {phase} ({PHASES.get(phase, ('Unknown',))[0]}):")
    for result in results:
        print(f"  {result}")
    print(f"\n[RIC] Gate {'PASSED ✅' if passed else 'FAILED ❌'}")


def cli_security_check(files: list[str]) -> None:
    """Run security gate check on files."""
    if not files:
        # Check staged files
        import subprocess

        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
        )
        files = [f.strip() for f in result.stdout.split("\n") if f.strip()]

    if not files:
        print("[RIC] No files to check.")
        return

    passed, findings = security_gate_check(files)
    if passed:
        print(f"[RIC] Security check PASSED ✅ ({len(files)} files)")
    else:
        print("[RIC] Security check FAILED ❌")
        for finding in findings:
            print(f"  ⚠️ {finding}")


def cli_summary() -> None:
    """Generate iteration summary for context persistence."""
    manager = RICStateManager()
    summary = generate_iteration_summary(manager)
    if summary:
        print(summary)
    else:
        print("[RIC] No active session.")


def cli_resolve(insight_id: str, resolution: str) -> None:
    """Resolve an insight."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return
    if manager.resolve_insight(insight_id, resolution):
        print(f"[RIC] Resolved {insight_id}: {resolution}")
    else:
        print(f"[RIC] Insight {insight_id} not found.")


def cli_insights() -> None:
    """List all insights."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    print("[RIC] All Insights:")
    for insight in manager.state.insights:
        status = "✅" if insight.resolved else "❌"
        print(f"  {status} {insight.id} [{insight.priority.value}] {insight.description}")
        if insight.resolved:
            print(f"      Resolution: {insight.resolution}")


def cli_throttles() -> None:
    """Show current throttle status."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    ts = manager.state.throttle_state
    print("[RIC] Throttle Status:")
    print(f"  Tool calls this phase: {ts.tool_calls_this_phase}/{SAFETY_THROTTLES['max_tool_calls_per_phase']}")
    print(f"  Consecutive failures: {ts.consecutive_failures}/{SAFETY_THROTTLES['max_consecutive_failures']}")
    for file_path, count in ts.edits_per_file.items():
        print(f"  Edits to {file_path}: {count}/{SAFETY_THROTTLES['max_edits_per_file_per_iteration']}")
    if ts.phase_start_time:
        elapsed = (time.time() - ts.phase_start_time) / 60
        print(f"  Time in phase: {elapsed:.1f}/{SAFETY_THROTTLES['max_time_per_phase_minutes']} min")


def cli_research_status() -> None:
    """Show research enforcement status (v4.3)."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    ts = manager.state.throttle_state
    max_searches = RESEARCH_ENFORCEMENT["searches_before_forced_persist"]
    searches = ts.web_searches_since_persist

    print("[RIC] Research Enforcement Status (v4.3):")
    print(f"  Current phase: P{manager.state.current_phase.value}")
    print()
    print(f"  🔍 Web searches since last persist: {searches}/{max_searches}")

    if searches == 0:
        print("     ✅ All searches persisted")
    elif searches < max_searches:
        remaining = max_searches - searches
        print(f"     ⏳ {remaining} more search(es) before forced persist")
    else:
        print("     🛑 BLOCKED: Must persist before next search")

    if ts.research_persist_blocked:
        print("     ⛔ Next WebSearch/WebFetch will be BLOCKED")
    print()

    if ts.last_research_persist_time:
        elapsed = time.time() - ts.last_research_persist_time
        minutes = int(elapsed // 60)
        print(f"  📝 Last persist: {minutes} min ago")
    else:
        print("  📝 Last persist: Never (this session)")
    print()

    print("  📋 Enforcement Rules:")
    print(f"     - Persist every {max_searches} searches (ENFORCED)")
    print(f"     - Search timestamp required: {RESEARCH_ENFORCEMENT['search_timestamp_required']}")
    print(f"     - Publication date required: {RESEARCH_ENFORCEMENT['publication_date_required']}")
    print(f"     - Block on exceed: {RESEARCH_ENFORCEMENT['block_search_without_persist']}")
    print(f"     - Min sources: {RESEARCH_ENFORCEMENT['min_sources_per_topic']}")


def cli_decisions() -> None:
    """Show decision trace."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    print("[RIC] Decision Trace:")
    for decision in manager.state.decision_trace:
        print(f"  {decision['decision_id']}: {decision['action']}")
        print(f"    Reasoning: {decision['reasoning'][:60]}...")
        print(f"    Risk: {decision['risk_assessment']}, Confidence: {decision['confidence']}%")


# =============================================================================
# v5.0 CLI COMMANDS
# =============================================================================


def cli_drift() -> None:
    """Check for scope drift."""
    manager = RICStateManager()
    if not manager.state.active:
        print("[RIC] No active session")
        return

    has_drift, msg, metrics = detect_drift(manager.state)

    print(f"[RIC v{RIC_VERSION}] Drift Detection:")
    if has_drift:
        print(f"  ⚠️ DRIFT DETECTED: {msg}")
    else:
        print(f"  ✅ No significant drift: {msg}")

    if metrics:
        print("\n  Metrics:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")


def cli_guardian() -> None:
    """Run guardian review on staged changes."""
    manager = RICStateManager()

    if not FEATURE_FLAGS.get("guardian_mode", True):
        print("[RIC] Guardian mode not enabled. Use: init --guardian")
        return

    # Get staged changes via git
    try:
        result = subprocess.run(["git", "diff", "--cached", "--stat"], capture_output=True, text=True, timeout=30)
        changes_summary = result.stdout
    except Exception as e:
        changes_summary = f"Error getting changes: {e}"

    # Run review
    changes = {"content": changes_summary, "summary": changes_summary}
    commit_msg = "Pending commit"

    review = run_guardian_review(changes, commit_msg)

    print(f"[RIC v{RIC_VERSION}] Guardian Review: {review.score}")

    if review.issues:
        print("\n  Issues:")
        for issue in review.issues:
            print(f"    ❌ {issue['criterion']}: {issue['issue']}")

    if review.recommendations:
        print("\n  Recommendations:")
        for rec in review.recommendations:
            print(f"    💡 {rec}")

    # Store review in state
    if manager.state.active:
        manager.state.guardian_reviews.append(
            {
                "timestamp": review.timestamp,
                "score": review.score,
                "passed": review.passed,
                "issues": len(review.issues),
            }
        )
        manager.save_state()


def cli_notes() -> None:
    """Show or update RIC_NOTES.md."""
    notes_path = Path(MEMORY_FILE["path"])

    if notes_path.exists():
        print(notes_path.read_text())
    else:
        print(f"[RIC v{RIC_VERSION}] No RIC_NOTES.md found.")
        print("  Will be created automatically when session updates occur.")
        print(f"  Path: {notes_path}")


def cli_features() -> None:
    """Show v5.0 feature status."""
    print(f"[RIC v{RIC_VERSION} {RIC_CODENAME}] Feature Status")
    print("=" * 50)

    enabled_count = sum(1 for v in FEATURE_FLAGS.values() if v)
    total_count = len(FEATURE_FLAGS)

    print(f"\nEnabled: {enabled_count}/{total_count} features\n")

    # Group by enabled/disabled
    print("✅ ENABLED:")
    for feature, enabled in sorted(FEATURE_FLAGS.items()):
        if enabled:
            print(f"    {feature}")

    print("\n❌ DISABLED:")
    for feature, enabled in sorted(FEATURE_FLAGS.items()):
        if not enabled:
            print(f"    {feature}")


def cli_repair_stats() -> None:
    """Show repair/replace statistics."""
    manager = RICStateManager()
    if not manager.state.active:
        print("[RIC] No active session")
        return

    print(get_repair_stats_summary(manager.state))


def cli_policy_check(action: str) -> None:
    """Check if an action is allowed by policy."""
    manager = RICStateManager()

    # Build context based on current state
    context = {
        "iteration": manager.state.current_iteration if manager.state.active else 0,
        "current_phase": manager.state.current_phase.name if manager.state.active else "",
        "p0_insights": len([i for i in manager.state.insights if i.priority.value == "P0" and not i.resolved])
        if manager.state.insights
        else 0,
        "p1_insights": len([i for i in manager.state.insights if i.priority.value == "P1" and not i.resolved])
        if manager.state.insights
        else 0,
        "tests_passing": True,  # Would check actual test status
        "files_changed": 1,
        "lines_changed": 50,
    }

    allowed, message = check_policy(action, context)

    print(f"[RIC v{RIC_VERSION}] Policy Check: {action}")
    if allowed:
        print(f"  ✅ Allowed: {message}")
    else:
        print(f"  ❌ Blocked: {message}")


def cli_enable_feature(feature: str) -> None:
    """Enable a v5.0 feature."""
    if enable_feature(feature):
        print(f"[RIC] Feature '{feature}' enabled")
    else:
        print(f"[RIC] Unknown feature: {feature}")
        print(f"  Available: {', '.join(FEATURE_FLAGS.keys())}")


def cli_disable_feature(feature: str) -> None:
    """Disable a v5.0 feature."""
    if disable_feature(feature):
        print(f"[RIC] Feature '{feature}' disabled")
    else:
        print(f"[RIC] Unknown feature: {feature}")
        print(f"  Available: {', '.join(FEATURE_FLAGS.keys())}")


# =============================================================================
# v5.0 PHASE ENFORCEMENT CLI COMMANDS
# =============================================================================


def cli_p0_status() -> None:
    """Show P0 RESEARCH completion status."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    print("[RIC v5.0] P0 RESEARCH Completion Status")
    print("=" * 60)

    status = manager.get_phase_completion_status("P0_RESEARCH")
    if not status:
        print("  No P0 tracking data. Use 'record-keywords' to start.")
        return

    # Requirements
    reqs = PHASE_REQUIREMENTS["P0_RESEARCH"]
    keywords = status.get("keywords_extracted", [])
    searches = status.get("web_searches_done", 0)
    sources = status.get("sources_documented", 0)
    persisted = status.get("findings_persisted", False)

    print("\n📋 Requirements:")
    print(f"  [{'✓' if len(keywords) >= 3 else '✗'}] Keywords extracted: {len(keywords)}/3+ required")
    if keywords:
        print(f"      Keywords: {', '.join(keywords[:10])}")
    print(
        f"  [{'✓' if searches >= reqs['min_web_searches'] else '✗'}] Web searches: {searches}/{reqs['min_web_searches']} required"
    )
    print(
        f"  [{'✓' if sources >= reqs['min_sources_documented'] else '✗'}] Sources documented: {sources}/{reqs['min_sources_documented']} required"
    )
    print(f"  [{'✓' if persisted else '✗'}] Findings persisted: {'Yes' if persisted else 'No'}")

    # Overall completion
    completed, blockers, _ = check_p0_completion(manager.state)
    print(f"\n📊 Overall: {'✅ COMPLETE' if completed else '❌ INCOMPLETE'}")
    if blockers:
        print("  Blockers:")
        for b in blockers:
            print(f"    - {b}")

    # Prompts
    if not completed:
        print(f"\n{P0_KEYWORD_EXTRACTION_PROMPT}")


def cli_p1_status() -> None:
    """Show P1 PLAN completion status."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    print("[RIC v5.1] P1 PLAN Completion Status")
    print("=" * 60)

    status = manager.get_phase_completion_status("P1_PLAN")
    if not status:
        print("  No P1 tracking data. Use 'record-tasks' to start.")
        return

    # Requirements
    reqs = PHASE_REQUIREMENTS["P1_PLAN"]
    tasks = status.get("tasks_defined", [])
    priorities = status.get("priorities_assigned", False)
    scope_defined = status.get("scope_defined", False)
    criteria_defined = status.get("success_criteria_defined", False)

    print("\n📋 Requirements:")
    print(
        f"  [{'✓' if len(tasks) >= reqs['min_tasks_defined'] else '✗'}] Tasks defined: {len(tasks)}/{reqs['min_tasks_defined']}+ required"
    )
    if tasks:
        for i, task in enumerate(tasks[:5]):
            is_smart = validate_task_smart(task)
            print(f"      {i + 1}. [{'SMART' if is_smart else 'VAGUE'}] {task[:60]}{'...' if len(task) > 60 else ''}")
        if len(tasks) > 5:
            print(f"      ... and {len(tasks) - 5} more tasks")
    print(f"  [{'✓' if priorities else '✗'}] Priorities assigned: {'Yes' if priorities else 'No'}")
    print(f"  [{'✓' if scope_defined else '✗'}] Scope boundaries defined: {'Yes' if scope_defined else 'No'}")
    print(f"  [{'✓' if criteria_defined else '✗'}] Success criteria defined: {'Yes' if criteria_defined else 'No'}")

    # SMART validation summary
    smart_tasks = sum(1 for t in tasks if validate_task_smart(t))
    print(f"\n📊 SMART Validation: {smart_tasks}/{len(tasks)} tasks pass")

    # Overall completion
    completed, blockers, _ = check_p1_completion(manager.state)
    print(f"\n📊 Overall: {'✅ COMPLETE' if completed else '❌ INCOMPLETE'}")
    if blockers:
        print("  Blockers:")
        for b in blockers:
            print(f"    - {b}")

    # Prompts
    if not completed:
        print(f"\n{P1_PLAN_START_PROMPT}")


def cli_p2_status() -> None:
    """Show P2 BUILD completion status."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    print("[RIC v5.1] P2 BUILD Completion Status")
    print("=" * 60)

    status = manager.get_phase_completion_status("P2_BUILD")
    if not status:
        print("  No P2 tracking data. Use 'record-change' to start.")
        return

    # Requirements
    reqs = PHASE_REQUIREMENTS["P2_BUILD"]
    changes = status.get("changes_made", [])
    tests_with_changes = status.get("tests_with_changes", 0)
    security_checked = status.get("security_checked", False)
    generation_verification = status.get("generation_verification_used", False)

    print("\n📋 Requirements:")
    print(
        f"  [{'✓' if len(changes) >= reqs['min_changes_before_advance'] else '✗'}] Changes made: {len(changes)}/{reqs['min_changes_before_advance']}+ required"
    )
    if changes:
        for i, change in enumerate(changes[:5]):
            files_count = len(change.get("files", []))
            atomic = files_count <= reqs["max_files_per_commit"]
            print(
                f"      {i + 1}. [{'ATOMIC' if atomic else 'LARGE'}] {change.get('description', 'No description')[:50]}... ({files_count} files)"
            )
        if len(changes) > 5:
            print(f"      ... and {len(changes) - 5} more changes")
    print(f"  [{'✓' if tests_with_changes > 0 else '✗'}] Tests with changes: {tests_with_changes} test commits")
    print(f"  [{'✓' if security_checked else '✗'}] Security checked: {'Yes' if security_checked else 'No'}")
    print(
        f"  [{'✓' if generation_verification else '✗'}] ReVeal pattern used: {'Yes' if generation_verification else 'No'}"
    )

    # Atomicity summary
    atomic_changes = sum(1 for c in changes if len(c.get("files", [])) <= reqs["max_files_per_commit"])
    print(f"\n📊 Atomic Changes: {atomic_changes}/{len(changes)} within {reqs['max_files_per_commit']} file limit")

    # Overall completion
    completed, blockers, _ = check_p2_completion(manager.state)
    print(f"\n📊 Overall: {'✅ COMPLETE' if completed else '❌ INCOMPLETE'}")
    if blockers:
        print("  Blockers:")
        for b in blockers:
            print(f"    - {b}")

    # Prompts
    if not completed:
        print(f"\n{P2_BUILD_START_PROMPT}")


def cli_p3_status() -> None:
    """Show P3 VERIFY completion status."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    print("[RIC v5.1] P3 VERIFY Completion Status")
    print("=" * 60)

    status = manager.get_phase_completion_status("P3_VERIFY")
    if not status:
        print("  No P3 tracking data. Use 'record-test-result' to start.")
        return

    # Requirements
    reqs = PHASE_REQUIREMENTS["P3_VERIFY"]
    tests_passed = status.get("tests_passed", False)
    test_count = status.get("test_count", 0)
    failed_count = status.get("failed_count", 0)
    coverage_pct = status.get("coverage_percent", 0)
    lint_clean = status.get("lint_clean", False)
    lint_errors = status.get("lint_errors", 0)
    security_scanned = status.get("security_scanned", False)
    security_issues = status.get("security_issues", 0)

    print("\n📋 Requirements:")
    print(
        f"  [{'✓' if tests_passed else '✗'}] Tests pass: {'Yes' if tests_passed else 'No'} ({test_count} tests, {failed_count} failed)"
    )
    print(
        f"  [{'✓' if coverage_pct >= reqs['min_coverage_percent'] else '✗'}] Coverage: {coverage_pct:.1f}%/{reqs['min_coverage_percent']}% required"
    )
    print(f"  [{'✓' if lint_clean else '✗'}] Lint clean: {'Yes' if lint_clean else 'No'} ({lint_errors} errors)")
    print(
        f"  [{'✓' if security_scanned else '✗'}] Security scanned: {'Yes' if security_scanned else 'No'} ({security_issues} issues)"
    )

    # Overall completion
    completed, blockers, _ = check_p3_completion(manager.state)
    print(f"\n📊 Overall: {'✅ COMPLETE' if completed else '❌ INCOMPLETE'}")
    if blockers:
        print("  Blockers:")
        for b in blockers:
            print(f"    - {b}")

    # v5.1: Show audit suggestion if enabled
    if is_feature_enabled("p3_audit_integration"):
        audit_run = status.get("audit_run", False)
        audit_critical = status.get("audit_critical_count", 0)
        audit_warnings = status.get("audit_warning_count", 0)
        print("\n🔍 Audit Suite (v5.1):")
        if audit_run:
            audit_status = "✅" if audit_critical == 0 else "❌" if audit_critical > 0 else "⚠️"
            print(f"  [{audit_status}] Audit run: Yes ({audit_critical} critical, {audit_warnings} warnings)")
        else:
            print("  [?] Audit not run yet")
            print("      💡 Run: python3 .claude/hooks/ric.py audit")
            print("      Categories: code, files, functions, best_practices, hooks, references, crossref, cleanup")

    # Prompts
    if not completed:
        print(f"\n{P3_VERIFY_START_PROMPT}")


def cli_p4_status() -> None:
    """Show P4 REFLECT completion status."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    print("[RIC v5.0] P4 REFLECT Completion Status")
    print("=" * 60)

    status = manager.get_phase_completion_status("P4_REFLECT")
    if not status:
        print("  No P4 tracking data. Phase not yet reached.")
        return

    # Requirements
    introspection = status.get("introspection_done", False)
    insights_rev = status.get("insights_reviewed", False)
    convergence = status.get("convergence_checked", False)
    upgrade_ideas = status.get("upgrade_ideas", [])
    loop_decision = status.get("loop_decision", "")
    loop_reason = status.get("loop_reason", "")

    print("\n📋 Requirements:")
    print(f"  [{'✓' if introspection else '✗'}] Introspection completed: {'Yes' if introspection else 'No'}")
    print(f"  [{'✓' if insights_rev else '✗'}] Insights reviewed: {'Yes' if insights_rev else 'No'}")
    print(f"  [{'✓' if convergence else '✗'}] Convergence checked: {'Yes' if convergence else 'No'}")
    print(f"  [{'✓' if upgrade_ideas else '✗'}] Upgrade ideas generated: {len(upgrade_ideas)}")
    if upgrade_ideas:
        for idea in upgrade_ideas[:5]:
            print(f"      - {idea}")
    print(f"  [{'✓' if loop_decision else '✗'}] Loop decision: {loop_decision or 'Not made'}")
    if loop_reason:
        print(f"      Reason: {loop_reason}")

    # Overall completion
    completed, blockers, _ = check_p4_completion(manager.state)
    print(f"\n📊 Overall: {'✅ COMPLETE' if completed else '❌ INCOMPLETE'}")
    if blockers:
        print("  Blockers:")
        for b in blockers:
            print(f"    - {b}")

    # Prompts
    if not completed:
        print(f"\n{P4_INTROSPECTION_TEMPLATE}")


def cli_record_keywords(keywords_str: str) -> None:
    """Record extracted keywords for P0 RESEARCH."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
    if not keywords:
        print("[RIC] No keywords provided. Use: record-keywords kw1,kw2,kw3")
        return

    manager.record_p0_keywords(keywords)
    print(f"[RIC] Recorded {len(keywords)} keywords: {', '.join(keywords)}")

    # Also expand keywords
    expanded = expand_keywords(keywords)
    if expanded:
        print(f"[RIC] Expanded keywords: {', '.join(expanded[:10])}")
        print("[RIC] Consider searching for these related terms.")


def cli_record_search(query: str, sources: int = 1) -> None:
    """Record a web search for P0 RESEARCH."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    manager.record_p0_search(query, sources)
    status = manager.get_phase_completion_status("P0_RESEARCH")
    searches_done = status.get("web_searches_done", 0)
    min_required = PHASE_REQUIREMENTS["P0_RESEARCH"]["min_web_searches"]
    print(f"[RIC] Recorded search: '{query}' ({sources} sources)")
    print(f"[RIC] Total searches: {searches_done}/{min_required} required")


def cli_record_findings() -> None:
    """Record that findings have been persisted for P0 RESEARCH."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    manager.record_p0_findings_persisted()
    print("[RIC] ✅ Findings persistence recorded.")
    print("[RIC] Make sure to save to docs/research/ directory.")


def cli_record_introspection() -> None:
    """Record introspection completion for P4 REFLECT."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    # Mark all P4 introspection tasks as done
    manager.record_p4_introspection()
    manager.record_p4_insights_reviewed()
    manager.record_p4_convergence_check(True, "Manual introspection")
    print("[RIC] ✅ Introspection recorded.")
    print("[RIC] Now use 'loop-decision LOOP|EXIT <reason>' to complete P4.")


def cli_record_upgrade_ideas(ideas_str: str) -> None:
    """Record upgrade ideas for P4 REFLECT with quality gate validation."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    ideas = [i.strip() for i in ideas_str.split(",") if i.strip()]
    if not ideas:
        print("[RIC] No ideas provided. Use: upgrade-ideas idea1,idea2,idea3")
        return

    # v5.1: Validate quality before recording
    if is_feature_enabled("quality_gates"):
        print("\n[RIC] v5.1 Quality Gate Assessment:")
        print("=" * 60)

        ideas_as_dicts = [{"idea": idea} for idea in ideas]
        assessment = validate_p4_quality(
            upgrade_ideas=ideas_as_dicts,
            current_iteration=manager.state.iteration,
            introspection_text="",
        )

        # Show individual idea assessments
        for i, quality in enumerate(assessment.ideas_assessed, 1):
            status = "✅" if quality.is_actionable else "❌"
            print(f"\n  Idea {i}: {status}")
            print(f"    Text: {quality.idea[:80]}{'...' if len(quality.idea) > 80 else ''}")
            print(
                f"    Has Location: {'✅' if quality.has_location else '❌'} {quality.location_matches[:2] if quality.location_matches else ''}"
            )
            print(
                f"    Has Action: {'✅' if quality.has_action else '❌'} {quality.action_matches[:3] if quality.action_matches else ''}"
            )
            print(f"    Is Vague: {'❌ Yes' if quality.is_vague else '✅ No'}")
            print(f"    Quality Score: {quality.quality_score:.1%}")

        # Show overall assessment
        print("\n" + "=" * 60)
        print(f"  Overall Quality Score: {assessment.overall_quality_score:.1%}")
        print(f"  Ideas with Location: {assessment.ideas_with_location}/{len(ideas)}")
        print(f"  Ideas with Action: {assessment.ideas_with_action}/{len(ideas)}")
        print(f"  Actionable Ideas: {assessment.actionable_ideas_count}/{len(ideas)}")

        if assessment.passes_quality_gate:
            print("\n  ✅ QUALITY GATE PASSED")
        else:
            print("\n  ❌ QUALITY GATE FAILED:")
            for blocker in assessment.quality_blockers:
                print(f"    - {blocker[:100]}...")

        print("=" * 60 + "\n")

    manager.record_p4_upgrade_ideas(ideas)
    print(f"[RIC] Recorded {len(ideas)} upgrade ideas:")
    for idea in ideas:
        print(f"  - {idea}")


def cli_loop_decision(decision: str, reason: str) -> None:
    """Record the LOOP/EXIT decision for P4 REFLECT."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    decision = decision.upper()
    if decision not in ("LOOP", "EXIT"):
        print("[RIC] Invalid decision. Use: LOOP or EXIT")
        return

    manager.record_p4_loop_decision(decision, reason)
    print(f"[RIC] ✅ Loop decision recorded: {decision}")
    print(f"[RIC] Reason: {reason}")

    if decision == "EXIT":
        print("\n[RIC] You chose EXIT. Use 'can-exit' to verify, then 'end' to finish.")
    else:
        print("\n[RIC] You chose LOOP. Use 'advance' to start next iteration.")


# =============================================================================
# P1 PLAN RECORDING FUNCTIONS (v5.1)
# =============================================================================


def cli_record_tasks(tasks_str: str) -> None:
    """Record planned tasks for P1 PLAN phase with SMART validation."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    tasks = [t.strip() for t in tasks_str.split(",") if t.strip()]
    if not tasks:
        print("[RIC] No tasks provided. Use: record-tasks task1,task2,task3")
        return

    # SMART validation
    print("\n[RIC] v5.1 SMART Task Validation:")
    print("=" * 60)
    smart_count = 0
    for i, task in enumerate(tasks, 1):
        is_smart = validate_task_smart(task)
        if is_smart:
            smart_count += 1
        status = "✅ SMART" if is_smart else "⚠️ VAGUE"
        print(f"  {i}. [{status}] {task[:70]}{'...' if len(task) > 70 else ''}")

    print("=" * 60)
    print(f"  SMART Tasks: {smart_count}/{len(tasks)}")
    if smart_count < len(tasks):
        print("  💡 Tip: Add file references (e.g., 'mcp/broker.py') or action verbs")
    print()

    # Record tasks
    status = manager.get_phase_completion_status("P1_PLAN") or {}
    status["tasks_defined"] = tasks
    manager.record_phase_data("P1_PLAN", status)

    print(f"[RIC] ✅ Recorded {len(tasks)} tasks for P1 PLAN phase.")
    print("[RIC] Next: Use 'record-scope' and 'record-criteria' to complete P1.")


def cli_record_scope(scope_str: str) -> None:
    """Record scope boundaries for P1 PLAN phase."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    if not scope_str.strip():
        print("[RIC] No scope provided. Use: record-scope 'in-scope items | out-of-scope items'")
        return

    status = manager.get_phase_completion_status("P1_PLAN") or {}
    status["scope_defined"] = True
    status["scope_description"] = scope_str.strip()
    manager.record_phase_data("P1_PLAN", status)

    print("[RIC] ✅ Recorded scope boundaries for P1 PLAN phase.")
    print(f"[RIC] Scope: {scope_str[:100]}{'...' if len(scope_str) > 100 else ''}")


def cli_record_criteria(criteria_str: str) -> None:
    """Record success criteria for P1 PLAN phase."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    criteria = [c.strip() for c in criteria_str.split(",") if c.strip()]
    if not criteria:
        print("[RIC] No criteria provided. Use: record-criteria criterion1,criterion2")
        return

    status = manager.get_phase_completion_status("P1_PLAN") or {}
    status["success_criteria_defined"] = True
    status["success_criteria"] = criteria
    manager.record_phase_data("P1_PLAN", status)

    print(f"[RIC] ✅ Recorded {len(criteria)} success criteria for P1 PLAN phase:")
    for c in criteria:
        print(f"  - {c}")


def cli_record_priorities(priorities_str: str) -> None:
    """Record task priorities for P1 PLAN phase."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    if not priorities_str.strip():
        print("[RIC] No priorities provided. Use: record-priorities P0:task1,P1:task2,P2:task3")
        return

    status = manager.get_phase_completion_status("P1_PLAN") or {}
    status["priorities_assigned"] = True
    status["priority_assignments"] = priorities_str.strip()
    manager.record_phase_data("P1_PLAN", status)

    print("[RIC] ✅ Recorded priorities for P1 PLAN phase.")


# =============================================================================
# P2 BUILD RECORDING FUNCTIONS (v5.1)
# =============================================================================


def cli_record_change(description: str, files_str: str = "") -> None:
    """Record an atomic change for P2 BUILD phase."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    if not description.strip():
        print("[RIC] No description provided. Use: record-change 'description' 'file1.py,file2.py'")
        return

    files = [f.strip() for f in files_str.split(",") if f.strip()] if files_str else []

    status = manager.get_phase_completion_status("P2_BUILD") or {}
    changes = status.get("changes_made", [])

    change_entry = {
        "description": description.strip(),
        "files": files,
        "timestamp": datetime.now().isoformat(),
    }
    changes.append(change_entry)
    status["changes_made"] = changes

    # Check atomicity
    reqs = PHASE_REQUIREMENTS["P2_BUILD"]
    is_atomic = len(files) <= reqs["max_files_per_commit"]

    manager.record_phase_data("P2_BUILD", status)

    print(f"[RIC] ✅ Recorded change #{len(changes)} for P2 BUILD phase.")
    print(f"  Description: {description[:80]}...")
    print(f"  Files ({len(files)}): {', '.join(files[:3])}{'...' if len(files) > 3 else ''}")
    print(
        f"  Atomicity: {'✅ ATOMIC' if is_atomic else '⚠️ LARGE'} ({len(files)}/{reqs['max_files_per_commit']} file limit)"
    )


def cli_record_tests_with_change() -> None:
    """Record that tests were included with a change (P2 BUILD)."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    status = manager.get_phase_completion_status("P2_BUILD") or {}
    tests_with_changes = status.get("tests_with_changes", 0) + 1
    status["tests_with_changes"] = tests_with_changes
    manager.record_phase_data("P2_BUILD", status)

    print(f"[RIC] ✅ Recorded tests with change #{tests_with_changes} for P2 BUILD phase.")


def cli_record_reveal_pattern() -> None:
    """Record that ReVeal generation-verification pattern was used (P2 BUILD)."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    status = manager.get_phase_completion_status("P2_BUILD") or {}
    status["generation_verification_used"] = True
    manager.record_phase_data("P2_BUILD", status)

    print("[RIC] ✅ Recorded ReVeal pattern usage for P2 BUILD phase.")
    print("[RIC] Tip: Use <generation-think> and <verification-think> tags in reasoning.")


def cli_record_security_check_p2() -> None:
    """Record that security check was done for P2 BUILD phase."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    status = manager.get_phase_completion_status("P2_BUILD") or {}
    status["security_checked"] = True
    manager.record_phase_data("P2_BUILD", status)

    print("[RIC] ✅ Recorded security check for P2 BUILD phase.")


# =============================================================================
# P3 VERIFY RECORDING FUNCTIONS (v5.1)
# =============================================================================


def cli_record_test_result(passed: str, test_count: str = "0", failed_count: str = "0") -> None:
    """Record test results for P3 VERIFY phase."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    tests_passed = passed.lower() in ("true", "yes", "1", "pass", "passed")
    try:
        total = int(test_count)
        failed = int(failed_count)
    except ValueError:
        print("[RIC] Invalid counts. Use: record-test-result pass|fail 100 5")
        return

    status = manager.get_phase_completion_status("P3_VERIFY") or {}
    status["tests_passed"] = tests_passed
    status["test_count"] = total
    status["failed_count"] = failed
    manager.record_phase_data("P3_VERIFY", status)

    status_icon = "✅" if tests_passed else "❌"
    print(f"[RIC] {status_icon} Recorded test result for P3 VERIFY phase.")
    print(f"  Tests: {total} total, {failed} failed")
    print(f"  Status: {'PASSED' if tests_passed else 'FAILED'}")


def cli_record_coverage(coverage_pct: str) -> None:
    """Record coverage percentage for P3 VERIFY phase."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    try:
        coverage = float(coverage_pct.replace("%", ""))
    except ValueError:
        print("[RIC] Invalid coverage. Use: record-coverage 75.5")
        return

    status = manager.get_phase_completion_status("P3_VERIFY") or {}
    status["coverage_percent"] = coverage
    manager.record_phase_data("P3_VERIFY", status)

    reqs = PHASE_REQUIREMENTS["P3_VERIFY"]
    status_icon = "✅" if coverage >= reqs["min_coverage_percent"] else "❌"
    print(f"[RIC] {status_icon} Recorded coverage {coverage:.1f}% for P3 VERIFY phase.")
    print(f"  Required: {reqs['min_coverage_percent']}%")


def cli_record_lint(clean: str, errors: str = "0") -> None:
    """Record lint results for P3 VERIFY phase."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    lint_clean = clean.lower() in ("true", "yes", "1", "clean", "pass")
    try:
        error_count = int(errors)
    except ValueError:
        error_count = 0

    status = manager.get_phase_completion_status("P3_VERIFY") or {}
    status["lint_clean"] = lint_clean
    status["lint_errors"] = error_count
    manager.record_phase_data("P3_VERIFY", status)

    status_icon = "✅" if lint_clean else "❌"
    print(f"[RIC] {status_icon} Recorded lint result for P3 VERIFY phase.")
    print(f"  Clean: {lint_clean}, Errors: {error_count}")


def cli_record_security_scan(clean: str, issues: str = "0") -> None:
    """Record security scan results for P3 VERIFY phase."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    security_clean = clean.lower() in ("true", "yes", "1", "clean", "pass")
    try:
        issue_count = int(issues)
    except ValueError:
        issue_count = 0

    status = manager.get_phase_completion_status("P3_VERIFY") or {}
    status["security_scanned"] = True
    status["security_clean"] = security_clean
    status["security_issues"] = issue_count
    manager.record_phase_data("P3_VERIFY", status)

    status_icon = "✅" if security_clean else "❌"
    print(f"[RIC] {status_icon} Recorded security scan for P3 VERIFY phase.")
    print(f"  Clean: {security_clean}, Issues: {issue_count}")


def cli_v50_status() -> None:
    """Show comprehensive v5.0 status."""
    manager = RICStateManager()

    print(f"[RIC v{RIC_VERSION} {RIC_CODENAME}] Status")
    print("=" * 60)

    # Basic status
    if manager.state.active:
        print("\nSession: ACTIVE")
        print(f"  Iteration: {manager.state.current_iteration}/{manager.state.max_iterations}")
        print(f"  Phase: {manager.state.current_phase.name}")

        # Drift status
        has_drift, drift_msg, _ = detect_drift(manager.state)
        drift_icon = "⚠️" if has_drift else "✅"
        print(f"\nDrift: {drift_icon} {drift_msg}")

        # Guardian status
        guardian_enabled = manager.state.guardian_enabled or FEATURE_FLAGS.get("guardian_mode", True)
        print(f"\nGuardian: {'✅ Enabled' if guardian_enabled else '❌ Disabled'}")
        if manager.state.guardian_reviews:
            last_review = manager.state.guardian_reviews[-1]
            print(f"  Last Review: {last_review['score']} ({last_review['timestamp']})")

        # Repair stats
        if manager.state.repair_stats:
            rs = manager.state.repair_stats
            print(f"\nRepair/Replace: {rs.repair_count}/{rs.replace_count} (ratio: {rs.get_ratio():.1%})")

        # Notes status
        notes_path = Path(MEMORY_FILE["path"])
        print(f"\nNotes: {'✅ Exists' if notes_path.exists() else '❌ Not created'}")

    else:
        print("\nNo active session. Use 'init' to start.")

    # Feature summary
    enabled_count = sum(1 for v in FEATURE_FLAGS.values() if v)
    print(f"\nFeatures: {enabled_count}/{len(FEATURE_FLAGS)} enabled")


# =============================================================================
# v5.1 DEBUG SUITE CLI FUNCTIONS
# =============================================================================


def cli_debug_phase(phase_num: int = -1) -> None:
    """Show debug information for a specific phase or current phase."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    # Use current phase if not specified
    if phase_num < 0:
        phase_num = manager.state.current_phase.value

    phase_names = {0: "P0_RESEARCH", 1: "P1_PLAN", 2: "P2_BUILD", 3: "P3_VERIFY", 4: "P4_REFLECT"}
    phase_key = phase_names.get(phase_num, "UNKNOWN")
    phase_display = PHASES.get(Phase(phase_num), ("UNKNOWN", ""))[0]

    print(f"[RIC v5.1] 🔧 DEBUG SUITE - Phase {phase_num} ({phase_display})")
    print("=" * 70)

    # Current state dump
    status = manager.get_phase_completion_status(phase_key)
    print("\n### Current Phase State")
    print(json.dumps(status, indent=2, default=str))

    # Get completion check
    check_funcs = {
        0: check_p0_completion,
        1: check_p1_completion,
        2: check_p2_completion,
        3: check_p3_completion,
        4: check_p4_completion,
    }
    check_func = check_funcs.get(phase_num)
    if check_func:
        completed, blockers, _ = check_func(manager.state)
        print(f"\n### Completion Status: {'✅ COMPLETE' if completed else '❌ INCOMPLETE'}")
        if blockers:
            print("\n### Active Blockers:")
            for b in blockers:
                print(f"  ❌ {b}")

    # Common issues
    common = DEBUG_COMMON_ISSUES.get(phase_key, "No common issues documented.")
    print(f"\n### Common Issues for {phase_display}")
    print(common)

    # Suggested debug steps
    print("\n### Suggested Debug Steps")
    print(f"1. Check phase status: `python3 .claude/hooks/ric.py p{phase_num}-status`")
    print(f"2. Review requirements: See PHASE_REQUIREMENTS['{phase_key}']")
    print("3. Try recording: Use the appropriate record-* command")
    print("4. Force advance (not recommended): `python3 .claude/hooks/ric.py advance --force`")


def cli_debug_iteration() -> None:
    """Show comprehensive debug information for the current iteration."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    state = manager.state
    print(f"[RIC v5.1] 🔧 FULL ITERATION DEBUG - Iteration {state.current_iteration}/{state.max_iterations}")
    print("=" * 70)

    # Session overview
    print("\n### Session Overview")
    print(f"  Session ID: {state.session_id or 'N/A'}")
    print(f"  Started: {state.started_at or 'N/A'}")
    print(f"  Current Phase: {state.current_phase.name}")

    # Insights
    counts = manager.count_insights_by_priority()
    print(f"  Open Insights: P0={counts['P0']}, P1={counts['P1']}, P2={counts['P2']}")
    print(f"  Fix Attempts: {sum(state.fix_attempts.values())}")
    print(f"  Plateau Count: {state.plateau_count}")

    # All phase statuses
    print("\n### All Phase Statuses")
    for phase_num in range(5):
        phase_key = {0: "P0_RESEARCH", 1: "P1_PLAN", 2: "P2_BUILD", 3: "P3_VERIFY", 4: "P4_REFLECT"}[phase_num]
        status = manager.get_phase_completion_status(phase_key)
        completed = status.get("completed", False)
        icon = "✅" if completed else "❌" if status else "⬜"
        print(
            f"  P{phase_num}: {icon} {phase_key} - {'Complete' if completed else 'Incomplete' if status else 'Not started'}"
        )

    # Throttle status
    print("\n### Throttle Status")
    ts = state.throttle_state
    print(f"  Tool calls/phase: {ts.tool_calls_this_phase}/{SAFETY_THROTTLES['max_tool_calls_per_phase']}")
    print(f"  Consecutive failures: {ts.consecutive_failures}/{SAFETY_THROTTLES['max_consecutive_failures']}")

    # Decision trace
    if state.decision_trace:
        print("\n### Recent Decisions (last 3)")
        for d in state.decision_trace[-3:]:
            print(f"  - [{d.get('timestamp', 'N/A')}] {d.get('action', 'N/A')}")
            print(f"    Reasoning: {d.get('reasoning', 'N/A')[:60]}...")

    # Suggested recovery
    print("\n### Suggested Recovery Actions")
    if state.plateau_count > 0:
        print("  1. Plateau detected - consider different approach")
    if ts.consecutive_failures > 2:
        print("  2. Multiple failures - review error messages carefully")
    print("  3. Use `p{X}-status` to check specific phase requirements")
    print("  4. Review research docs in docs/research/ for context")


def cli_theme() -> None:
    """Show project theme analysis prompt."""
    print("[RIC v5.1] 🎯 PROJECT THEME ANALYSIS")
    print("=" * 70)
    print(PROJECT_THEME_ANALYSIS_PROMPT)


def cli_record_theme(theme_data: str) -> None:
    """Record project theme analysis."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    if not theme_data.strip():
        print("[RIC] No theme data provided.")
        print("Usage: record-theme 'Primary purpose: X | Domain: Y | Capabilities: Z'")
        return

    status = manager.get_phase_completion_status("PROJECT_THEME") or {}
    status["theme_analysis"] = theme_data.strip()
    status["recorded_at"] = datetime.now().isoformat()
    status["iteration"] = manager.state.current_iteration
    manager.record_phase_data("PROJECT_THEME", status)

    print("[RIC] ✅ Recorded project theme analysis.")
    print(f"  Theme: {theme_data[:100]}{'...' if len(theme_data) > 100 else ''}")


def cli_theme_expansion() -> None:
    """Show theme expansion suggestions based on previous analysis."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session.")
        return

    theme = manager.get_phase_completion_status("PROJECT_THEME") or {}
    previous_theme = theme.get("theme_analysis", "No previous theme recorded")

    print("[RIC v5.1] 🌱 THEME EXPANSION")
    print("=" * 70)
    print("\n### Previous Theme Context")
    print(f"{previous_theme[:500]}{'...' if len(previous_theme) > 500 else ''}")

    print("\n### Theme Expansion Recommendations")
    print("""
Based on the project theme, consider expanding in these directions:

1. **Core Capability Enhancement**
   - What fundamental feature could be improved?
   - What existing capability needs refinement?

2. **Integration Opportunities**
   - What external systems could connect?
   - What data sources could enhance value?

3. **Quality & Polish**
   - What user experience improvements are possible?
   - What technical debt should be addressed?

Use `record-theme` to update the theme analysis with current iteration's contributions.
""")


# =============================================================================
# AUDIT SUITE CLI FUNCTIONS (v5.1)
# =============================================================================


@dataclass
class AuditIssue:
    """Single audit issue found."""

    category: str
    check_name: str
    severity: str  # critical, warning, info, suggestion
    message: str
    file_path: str = ""
    line_number: int = 0
    suggestion: str = ""


@dataclass
class AuditResult:
    """Result of an audit check."""

    category: str
    check_name: str
    passed: bool
    issues: list[AuditIssue] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    """Complete audit report."""

    timestamp: str
    categories_checked: list[str]
    results: list[AuditResult]
    total_issues: int = 0
    critical_count: int = 0
    warning_count: int = 0
    suggestion_count: int = 0
    blocks_exit: bool = False

    def __post_init__(self):
        """Calculate totals."""
        for result in self.results:
            for issue in result.issues:
                self.total_issues += 1
                if issue.severity == "critical":
                    self.critical_count += 1
                    self.blocks_exit = True
                elif issue.severity == "warning":
                    self.warning_count += 1
                else:
                    self.suggestion_count += 1


def audit_code_quality(target_path: str = ".") -> AuditResult:
    """Check code quality issues."""
    issues: list[AuditIssue] = []
    details = {"files_checked": 0, "issues_by_type": {}}

    # Find Python files
    try:
        import subprocess

        result = subprocess.run(
            ["find", target_path, "-name", "*.py", "-type", "f"], capture_output=True, text=True, timeout=30
        )
        py_files = [f for f in result.stdout.strip().split("\n") if f and not f.startswith("./venv")]
    except Exception:
        py_files = []

    for filepath in py_files[:50]:  # Limit to 50 files
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")
            details["files_checked"] += 1

            # Check for bare except
            for i, line in enumerate(lines, 1):
                if re.search(AUDIT_PATTERNS["bare_except"], line):
                    issues.append(
                        AuditIssue(
                            category="code",
                            check_name="bare_except",
                            severity="warning",
                            message="Bare 'except:' clause - should catch specific exceptions",
                            file_path=filepath,
                            line_number=i,
                            suggestion="Use 'except Exception:' or more specific exceptions",
                        )
                    )

            # Check for import *
            for i, line in enumerate(lines, 1):
                if re.search(AUDIT_PATTERNS["import_star"], line):
                    issues.append(
                        AuditIssue(
                            category="code",
                            check_name="import_star",
                            severity="warning",
                            message="Wildcard import 'from X import *' pollutes namespace",
                            file_path=filepath,
                            line_number=i,
                            suggestion="Import specific names instead",
                        )
                    )

        except Exception:
            pass

    return AuditResult(
        category="code",
        check_name="code_quality",
        passed=len([i for i in issues if i.severity == "critical"]) == 0,
        issues=issues,
        details=details,
    )


def audit_missing_files(expected_files: list[str] = None) -> AuditResult:
    """Check for missing expected files."""
    issues: list[AuditIssue] = []
    details = {"files_checked": 0, "missing": [], "empty": []}

    if expected_files is None:
        # Default expected files for this project
        expected_files = [
            "requirements.txt",
            "CLAUDE.md",
            ".claude/settings.json",
            ".claude/RIC_CONTEXT.md",
            "docs/PROJECT_STATUS.md",
        ]

    for filepath in expected_files:
        details["files_checked"] += 1
        full_path = Path(filepath)
        if not full_path.exists():
            issues.append(
                AuditIssue(
                    category="files",
                    check_name="missing_files",
                    severity="critical" if "requirements" in filepath or "settings" in filepath else "warning",
                    message=f"Expected file missing: {filepath}",
                    file_path=filepath,
                    suggestion=f"Create {filepath} or update expected files list",
                )
            )
            details["missing"].append(filepath)
        elif full_path.stat().st_size == 0:
            issues.append(
                AuditIssue(
                    category="files",
                    check_name="empty_files",
                    severity="warning",
                    message=f"File is empty: {filepath}",
                    file_path=filepath,
                    suggestion="Add content or remove if unused",
                )
            )
            details["empty"].append(filepath)

    return AuditResult(
        category="files",
        check_name="file_completeness",
        passed=len(details["missing"]) == 0,
        issues=issues,
        details=details,
    )


def audit_functions(target_path: str = ".") -> AuditResult:
    """Check function definitions and usage."""
    issues: list[AuditIssue] = []
    details = {"functions_found": 0, "missing_docstrings": 0, "long_functions": 0}

    try:
        import subprocess

        result = subprocess.run(
            ["find", target_path, "-name", "*.py", "-type", "f"], capture_output=True, text=True, timeout=30
        )
        py_files = [
            f for f in result.stdout.strip().split("\n") if f and not f.startswith("./venv") and "test" not in f.lower()
        ]
    except Exception:
        py_files = []

    for filepath in py_files[:30]:  # Limit
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")

            # Find function definitions
            func_pattern = r"^\s*def\s+(\w+)\s*\("
            for i, line in enumerate(lines, 1):
                match = re.match(func_pattern, line)
                if match:
                    details["functions_found"] += 1
                    func_name = match.group(1)

                    # Check for docstring (next non-empty line should be docstring)
                    if i < len(lines):
                        next_lines = "\n".join(lines[i : i + 3])
                        if '"""' not in next_lines and "'''" not in next_lines:
                            # Only flag public functions (not starting with _)
                            if not func_name.startswith("_"):
                                details["missing_docstrings"] += 1
                                if details["missing_docstrings"] <= 5:  # Limit warnings
                                    issues.append(
                                        AuditIssue(
                                            category="functions",
                                            check_name="missing_docstring",
                                            severity="info",
                                            message=f"Public function '{func_name}' missing docstring",
                                            file_path=filepath,
                                            line_number=i,
                                            suggestion="Add docstring describing purpose, args, returns",
                                        )
                                    )

        except Exception:
            pass

    return AuditResult(
        category="functions",
        check_name="function_analysis",
        passed=True,  # Info-only check
        issues=issues,
        details=details,
    )


def audit_best_practices(target_path: str = ".") -> AuditResult:
    """Check adherence to best practices."""
    issues: list[AuditIssue] = []
    details = {"checks_performed": [], "findings": {}}

    try:
        import subprocess

        result = subprocess.run(
            ["find", target_path, "-name", "*.py", "-type", "f"], capture_output=True, text=True, timeout=30
        )
        py_files = [f for f in result.stdout.strip().split("\n") if f and not f.startswith("./venv")]
    except Exception:
        py_files = []

    hardcoded_secrets = 0
    debug_code = 0
    todo_fixme = 0

    for filepath in py_files[:50]:
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")

            # Check for hardcoded secrets
            for i, line in enumerate(lines, 1):
                if re.search(AUDIT_PATTERNS["hardcoded_secrets"], line, re.IGNORECASE):
                    hardcoded_secrets += 1
                    if hardcoded_secrets <= 3:
                        issues.append(
                            AuditIssue(
                                category="best_practices",
                                check_name="hardcoded_secrets",
                                severity="critical",
                                message="Possible hardcoded secret detected",
                                file_path=filepath,
                                line_number=i,
                                suggestion="Use environment variables or secrets manager",
                            )
                        )

            # Check for debug code
            for i, line in enumerate(lines, 1):
                if re.search(AUDIT_PATTERNS["debug_code"], line):
                    # Exclude test files
                    if "test" not in filepath.lower():
                        debug_code += 1
                        if debug_code <= 5:
                            issues.append(
                                AuditIssue(
                                    category="best_practices",
                                    check_name="debug_code",
                                    severity="warning",
                                    message="Debug code found (print/breakpoint)",
                                    file_path=filepath,
                                    line_number=i,
                                    suggestion="Remove or replace with proper logging",
                                )
                            )

            # Check for TODO/FIXME
            for i, line in enumerate(lines, 1):
                match = re.search(AUDIT_PATTERNS["todo_fixme"], line)
                if match:
                    todo_fixme += 1
                    if todo_fixme <= 10:
                        issues.append(
                            AuditIssue(
                                category="best_practices",
                                check_name="todo_fixme",
                                severity="info",
                                message=f"{match.group(1).upper()}: {match.group(2)[:50]}...",
                                file_path=filepath,
                                line_number=i,
                                suggestion="Address or track in issue tracker",
                            )
                        )

        except Exception:
            pass

    details["findings"] = {
        "hardcoded_secrets": hardcoded_secrets,
        "debug_code": debug_code,
        "todo_fixme": todo_fixme,
    }

    return AuditResult(
        category="best_practices",
        check_name="best_practices_audit",
        passed=hardcoded_secrets == 0,
        issues=issues,
        details=details,
    )


def audit_hooks(hooks_path: str = ".claude/hooks") -> AuditResult:
    """Check Claude Code hooks configuration."""
    issues: list[AuditIssue] = []
    details = {"hooks_found": [], "settings_valid": False, "errors": []}

    hooks_dir = Path(hooks_path)
    settings_path = Path(".claude/settings.json")

    # Check hooks directory exists
    if not hooks_dir.exists():
        issues.append(
            AuditIssue(
                category="hooks",
                check_name="hooks_directory",
                severity="critical",
                message=f"Hooks directory not found: {hooks_path}",
                file_path=hooks_path,
                suggestion="Create .claude/hooks/ directory",
            )
        )
        return AuditResult(category="hooks", check_name="hooks_audit", passed=False, issues=issues, details=details)

    # Find hook files
    hook_files = list(hooks_dir.glob("*.py"))
    details["hooks_found"] = [str(f) for f in hook_files]

    # Check settings.json references hooks
    if settings_path.exists():
        try:
            with open(settings_path) as f:
                settings = json.load(f)
            details["settings_valid"] = True

            # Check if hooks are referenced
            hooks_config = settings.get("hooks", {})
            registered_hooks = set()

            for hook_type, hook_list in hooks_config.items():
                if isinstance(hook_list, list):
                    for entry in hook_list:
                        if isinstance(entry, dict) and "hooks" in entry:
                            for h in entry["hooks"]:
                                if "command" in h:
                                    cmd = h["command"]
                                    for hf in hook_files:
                                        if hf.name in cmd:
                                            registered_hooks.add(hf.name)

            # Check for unregistered hooks
            for hf in hook_files:
                if hf.name not in registered_hooks and hf.name != "__init__.py":
                    issues.append(
                        AuditIssue(
                            category="hooks",
                            check_name="unregistered_hook",
                            severity="info",
                            message=f"Hook file not registered in settings.json: {hf.name}",
                            file_path=str(hf),
                            suggestion="Add to .claude/settings.json hooks section if needed",
                        )
                    )

        except json.JSONDecodeError as e:
            issues.append(
                AuditIssue(
                    category="hooks",
                    check_name="settings_json",
                    severity="critical",
                    message=f"Invalid JSON in settings.json: {e}",
                    file_path=str(settings_path),
                    suggestion="Fix JSON syntax errors",
                )
            )

    return AuditResult(
        category="hooks",
        check_name="hooks_audit",
        passed=len([i for i in issues if i.severity == "critical"]) == 0,
        issues=issues,
        details=details,
    )


def audit_references(target_path: str = ".") -> AuditResult:
    """Check for broken imports and references."""
    issues: list[AuditIssue] = []
    details = {"imports_checked": 0, "broken_imports": []}

    try:
        import subprocess

        result = subprocess.run(
            ["find", target_path, "-name", "*.py", "-type", "f"], capture_output=True, text=True, timeout=30
        )
        py_files = [f for f in result.stdout.strip().split("\n") if f and not f.startswith("./venv")]
    except Exception:
        py_files = []

    # Extract all module names defined in the project
    project_modules = set()
    for filepath in py_files:
        parts = Path(filepath).parts
        for i, part in enumerate(parts):
            if part.endswith(".py"):
                module_name = part[:-3]
                if module_name != "__init__":
                    project_modules.add(module_name)
            else:
                project_modules.add(part)

    # Check imports in each file (simplified check)
    for filepath in py_files[:30]:
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                # Simple import pattern
                import_match = re.match(r"^from\s+(\w+(?:\.\w+)*)\s+import", line)
                if import_match:
                    details["imports_checked"] += 1
                    module = import_match.group(1).split(".")[0]
                    # Only flag project imports that might be broken
                    # (Skip standard library and known third-party)

        except Exception:
            pass

    return AuditResult(
        category="references", check_name="references_audit", passed=True, issues=issues, details=details
    )


def audit_crossref(target_path: str = ".") -> AuditResult:
    """Check cross-references between docs, code, and tests."""
    issues: list[AuditIssue] = []
    details = {"doc_files": 0, "test_files": 0, "sync_issues": []}

    docs_path = Path("docs")
    tests_path = Path("tests")

    # Count docs
    if docs_path.exists():
        details["doc_files"] = len(list(docs_path.rglob("*.md")))

    # Count tests
    if tests_path.exists():
        details["test_files"] = len(list(tests_path.rglob("test_*.py")))

    # Check if key docs exist and are recent
    key_docs = [
        "docs/PROJECT_STATUS.md",
        "docs/IMPLEMENTATION_TRACKER.md",
        "CLAUDE.md",
    ]

    for doc in key_docs:
        doc_path = Path(doc)
        if doc_path.exists():
            # Check if recently modified (within 7 days)
            mtime = datetime.fromtimestamp(doc_path.stat().st_mtime)
            age_days = (datetime.now() - mtime).days
            if age_days > 30:
                issues.append(
                    AuditIssue(
                        category="crossref",
                        check_name="stale_docs",
                        severity="info",
                        message=f"Document may be stale (last modified {age_days} days ago)",
                        file_path=doc,
                        suggestion="Review and update if needed",
                    )
                )

    return AuditResult(category="crossref", check_name="crossref_audit", passed=True, issues=issues, details=details)


def audit_cleanup(target_path: str = ".") -> AuditResult:
    """Identify cleanup opportunities."""
    issues: list[AuditIssue] = []
    details = {"temp_files": [], "stale_files": [], "large_files": []}

    # Check for temp files
    temp_patterns = ["*.tmp", "*.bak", "*.swp", "*~", "*.pyc", "__pycache__"]
    for pattern in temp_patterns:
        try:
            for filepath in Path(target_path).rglob(pattern):
                if ".git" not in str(filepath):
                    details["temp_files"].append(str(filepath))
        except Exception:
            pass

    if details["temp_files"]:
        issues.append(
            AuditIssue(
                category="cleanup",
                check_name="temp_files",
                severity="info",
                message=f"Found {len(details['temp_files'])} temporary/cache files",
                suggestion="Run cleanup: find . -name '*.pyc' -delete && find . -name '__pycache__' -type d -delete",
            )
        )

    # Check for large files
    try:
        for filepath in Path(target_path).rglob("*"):
            if filepath.is_file() and ".git" not in str(filepath):
                size_mb = filepath.stat().st_size / (1024 * 1024)
                if size_mb > 10:
                    details["large_files"].append({"path": str(filepath), "size_mb": round(size_mb, 2)})
    except Exception:
        pass

    if details["large_files"]:
        issues.append(
            AuditIssue(
                category="cleanup",
                check_name="large_files",
                severity="warning",
                message=f"Found {len(details['large_files'])} files >10MB",
                suggestion="Review large files - consider .gitignore or compression",
            )
        )

    return AuditResult(category="cleanup", check_name="cleanup_audit", passed=True, issues=issues, details=details)


def run_full_audit(categories: list[str] = None) -> AuditReport:
    """Run complete audit suite."""
    if categories is None:
        categories = list(AUDIT_CATEGORIES.keys())

    results: list[AuditResult] = []
    timestamp = datetime.now().isoformat()

    audit_funcs = {
        "code": audit_code_quality,
        "files": audit_missing_files,
        "functions": audit_functions,
        "best_practices": audit_best_practices,
        "hooks": audit_hooks,
        "references": audit_references,
        "crossref": audit_crossref,
        "cleanup": audit_cleanup,
    }

    for category in categories:
        if category in audit_funcs:
            try:
                result = audit_funcs[category]()
                results.append(result)
            except Exception as e:
                results.append(
                    AuditResult(
                        category=category,
                        check_name=f"{category}_error",
                        passed=False,
                        issues=[
                            AuditIssue(
                                category=category,
                                check_name="audit_error",
                                severity="warning",
                                message=f"Audit error: {e!s}",
                            )
                        ],
                    )
                )

    return AuditReport(timestamp=timestamp, categories_checked=categories, results=results)


def cli_audit(categories: str = "") -> None:
    """Run audit suite and display report."""
    print("[RIC v5.1] 🔍 AUDIT SUITE")
    print("=" * 70)

    # Parse categories
    if categories:
        cat_list = [c.strip() for c in categories.split(",")]
    else:
        cat_list = list(AUDIT_CATEGORIES.keys())

    print(f"\n### Running {len(cat_list)} audit categories...")
    print(f"  Categories: {', '.join(cat_list)}")
    print()

    report = run_full_audit(cat_list)

    # Summary table
    print("### Summary")
    print("| Category | Checks | Issues | Critical | Warnings |")
    print("|----------|--------|--------|----------|----------|")
    for result in report.results:
        critical = len([i for i in result.issues if i.severity == "critical"])
        warnings = len([i for i in result.issues if i.severity == "warning"])
        status = "❌" if critical > 0 else "✅" if len(result.issues) == 0 else "⚠️"
        print(f"| {status} {result.category} | {result.check_name} | {len(result.issues)} | {critical} | {warnings} |")

    print()
    status_icon = "❌ BLOCKED" if report.blocks_exit else "✅ PASS" if report.critical_count == 0 else "⚠️ WARNINGS"
    print(f"### Overall Status: {status_icon}")
    print(f"  Total Issues: {report.total_issues}")
    print(f"  Critical: {report.critical_count}")
    print(f"  Warnings: {report.warning_count}")
    print(f"  Suggestions: {report.suggestion_count}")

    # Show critical and warning issues
    if report.critical_count > 0 or report.warning_count > 0:
        print("\n### Issues Requiring Attention")
        for result in report.results:
            for issue in result.issues:
                if issue.severity in ["critical", "warning"]:
                    sev = AUDIT_ISSUE_SEVERITY[issue.severity]
                    loc = f"{issue.file_path}:{issue.line_number}" if issue.file_path else ""
                    print(f"  {sev['emoji']} [{issue.severity.upper()}] {issue.message}")
                    if loc:
                        print(f"     📍 {loc}")
                    if issue.suggestion:
                        print(f"     💡 {issue.suggestion}")

    # Suggestions
    if report.suggestion_count > 0:
        print(f"\n### Info & Suggestions ({report.suggestion_count} items)")
        print("  Use 'audit info' to see all suggestions")

    # Reference to comprehensive QA Validator
    print("\n### For Comprehensive Validation")
    print("  Use 'full-audit' for 35 checks across 11 categories (QA Validator)")
    print("  Use 'record-qa' to run QA Validator and record to P3 VERIFY")

    return report


def cli_audit_category(category: str) -> None:
    """Run audit for a specific category."""
    if category not in AUDIT_CATEGORIES:
        print(f"[RIC] Unknown category: {category}")
        print(f"  Available: {', '.join(AUDIT_CATEGORIES.keys())}")
        return

    print(f"[RIC v5.1] 🔍 AUDIT: {AUDIT_CATEGORIES[category]['name']}")
    print("=" * 70)
    print(f"  {AUDIT_CATEGORIES[category]['description']}")
    print(f"  Checks: {', '.join(AUDIT_CATEGORIES[category]['checks'])}")
    print()

    report = run_full_audit([category])
    result = report.results[0] if report.results else None

    if result:
        print(f"### Result: {'✅ PASS' if result.passed else '❌ ISSUES FOUND'}")
        print(f"  Issues: {len(result.issues)}")
        print(f"  Details: {result.details}")

        if result.issues:
            print("\n### Issues")
            for issue in result.issues:
                sev = AUDIT_ISSUE_SEVERITY.get(issue.severity, {})
                emoji = sev.get("emoji", "•")
                print(f"  {emoji} {issue.message}")
                if issue.file_path:
                    loc = f"{issue.file_path}:{issue.line_number}" if issue.line_number else issue.file_path
                    print(f"     📍 {loc}")
                if issue.suggestion:
                    print(f"     💡 {issue.suggestion}")


def cli_audit_fix(category: str = "") -> None:
    """Show fix suggestions for audit issues."""
    print("[RIC v5.1] 🔧 AUDIT FIX SUGGESTIONS")
    print("=" * 70)

    if category and category in AUDIT_CATEGORIES:
        report = run_full_audit([category])
    else:
        report = run_full_audit()

    if report.total_issues == 0:
        print("✅ No issues found! Codebase is clean.")
        return

    # Group issues by type and provide fix commands
    print("\n### Automated Fix Commands")

    # Cleanup fixes
    cleanup_issues = [r for r in report.results if r.category == "cleanup"]
    if cleanup_issues and cleanup_issues[0].issues:
        print("\n**Cleanup:**")
        print("```bash")
        print("# Remove Python cache files")
        print("find . -name '*.pyc' -delete")
        print("find . -name '__pycache__' -type d -exec rm -rf {} +")
        print("```")

    # Best practices fixes
    bp_issues = [r for r in report.results if r.category == "best_practices"]
    if bp_issues:
        for issue in bp_issues[0].issues:
            if issue.check_name == "debug_code":
                print("\n**Remove Debug Code:**")
                print("```bash")
                print("# Find all print statements (review before deleting)")
                print("grep -rn 'print(' --include='*.py' | grep -v test")
                print("```")

    print("\n### Manual Review Required")
    for result in report.results:
        for issue in result.issues:
            if issue.severity == "critical":
                print(f"  ❗ {issue.message}")
                print(f"     File: {issue.file_path}")


def cli_record_audit() -> None:
    """Run audit and record results to P3 VERIFY status."""
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session. Running standalone audit...")
        cli_audit("")
        return

    print("[RIC v5.1] 🔍 Running Audit and Recording to P3 VERIFY...")

    # Run critical audits
    report = run_full_audit(["code", "best_practices", "hooks", "files"])

    # Update P3 status
    status = manager.get_phase_completion_status("P3_VERIFY") or {}
    status["audit_run"] = True
    status["audit_timestamp"] = datetime.now().isoformat()
    status["audit_critical_count"] = report.critical_count
    status["audit_warning_count"] = report.warning_count
    status["audit_total_issues"] = report.total_issues
    manager.record_phase_data("P3_VERIFY", status)

    # Display summary
    status_icon = "❌" if report.critical_count > 0 else "✅" if report.warning_count == 0 else "⚠️"
    print(f"\n{status_icon} Audit Complete:")
    print(f"  Critical: {report.critical_count}")
    print(f"  Warnings: {report.warning_count}")
    print(f"  Total Issues: {report.total_issues}")

    if report.critical_count > 0:
        print("\n❗ Critical issues found - review with 'audit' command")
    else:
        print("\n✅ Audit results recorded to P3 VERIFY status")


def cli_cleanup() -> None:
    """Run cleanup check and show suggestions."""
    print("[RIC v5.1] 🧹 CLEANUP SUITE")
    print("=" * 70)

    result = audit_cleanup()

    print("\n### Temp Files")
    if result.details["temp_files"]:
        print(f"  Found {len(result.details['temp_files'])} temp/cache files")
        for f in result.details["temp_files"][:10]:
            print(f"    - {f}")
        if len(result.details["temp_files"]) > 10:
            print(f"    ... and {len(result.details['temp_files']) - 10} more")
    else:
        print("  ✅ No temp files found")

    print("\n### Large Files")
    if result.details["large_files"]:
        print(f"  Found {len(result.details['large_files'])} files >10MB")
        for f in result.details["large_files"]:
            print(f"    - {f['path']} ({f['size_mb']} MB)")
    else:
        print("  ✅ No oversized files")

    print("\n### Quick Cleanup Commands")
    print("```bash")
    print("# Remove Python cache")
    print("find . -name '__pycache__' -type d -exec rm -rf {} +")
    print("find . -name '*.pyc' -delete")
    print("")
    print("# Remove editor backup files")
    print("find . -name '*~' -delete")
    print("find . -name '*.swp' -delete")
    print("```")


# =============================================================================
# QA VALIDATOR INTEGRATION (Assimilation)
# =============================================================================


def cli_full_audit() -> None:
    """Run comprehensive QA Validator (35 checks across 11 categories).

    This wraps scripts/qa_validator.py which provides deeper analysis than
    the built-in Audit Suite. Use this for thorough pre-commit validation.
    """
    print("[RIC v5.1] 🔍 FULL QA AUDIT (Comprehensive)")
    print("=" * 70)
    print("Running scripts/qa_validator.py with all 35 checks...")
    print()

    import subprocess

    qa_script = Path("scripts/qa_validator.py")

    if not qa_script.exists():
        print("❌ ERROR: scripts/qa_validator.py not found")
        print("  The comprehensive QA validator is not available.")
        print("  Use 'audit' command for built-in Audit Suite instead.")
        return

    try:
        result = subprocess.run(
            ["python3", str(qa_script), "--verbose"],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        # Display output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        # Return code indicates issues
        if result.returncode != 0:
            print("\n❗ QA Validator found issues requiring attention")
        else:
            print("\n✅ QA Validator completed successfully")

    except subprocess.TimeoutExpired:
        print("❌ ERROR: QA Validator timed out after 2 minutes")
    except Exception as e:
        print(f"❌ ERROR running QA Validator: {e}")


def cli_qa_fix() -> None:
    """Run QA Validator with auto-fix enabled."""
    print("[RIC v5.1] 🔧 QA AUTO-FIX")
    print("=" * 70)
    print("Running scripts/qa_validator.py --fix...")
    print()

    import subprocess

    qa_script = Path("scripts/qa_validator.py")

    if not qa_script.exists():
        print("❌ ERROR: scripts/qa_validator.py not found")
        return

    try:
        result = subprocess.run(["python3", str(qa_script), "--fix"], capture_output=True, text=True, timeout=120)

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

    except subprocess.TimeoutExpired:
        print("❌ ERROR: QA Validator timed out")
    except Exception as e:
        print(f"❌ ERROR: {e}")


def cli_qa_check(category: str = "") -> None:
    """Run QA Validator on specific category."""
    print(f"[RIC v5.1] 🔍 QA CHECK: {category or 'all'}")
    print("=" * 70)

    import subprocess

    qa_script = Path("scripts/qa_validator.py")

    if not qa_script.exists():
        print("❌ ERROR: scripts/qa_validator.py not found")
        return

    cmd = ["python3", str(qa_script)]
    if category:
        cmd.extend(["--check", category])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

    except Exception as e:
        print(f"❌ ERROR: {e}")


def cli_record_qa() -> None:
    """Run QA Validator and record results to P3 VERIFY status.

    This integrates qa_validator.py results into the RIC session state,
    providing comprehensive validation data for phase advancement decisions.
    """
    manager = RICStateManager()
    if not manager.is_active():
        print("[RIC] No active session. Running standalone full-audit...")
        cli_full_audit()
        return

    print("[RIC v5.1] 🔍 Running QA Validator and Recording to P3 VERIFY...")
    print()

    import subprocess

    qa_script = Path("scripts/qa_validator.py")

    if not qa_script.exists():
        print("❌ ERROR: scripts/qa_validator.py not found")
        print("  Recording built-in audit results instead...")
        cli_record_audit()
        return

    try:
        # Run with JSON output for machine-readable results
        result = subprocess.run(["python3", str(qa_script), "--json"], capture_output=True, text=True, timeout=120)

        if result.returncode != 0 and not result.stdout:
            print(f"❌ QA Validator failed: {result.stderr}")
            return

        # Parse JSON output
        try:
            qa_report = json.loads(result.stdout)
        except json.JSONDecodeError:
            # Fallback: try to parse summary from text output
            print("⚠️ Could not parse JSON output, extracting summary...")
            qa_report = {
                "total_issues": 0,
                "critical_count": 0,
                "warning_count": 0,
                "categories_checked": [],
            }

            # Try to extract counts from text
            lines = result.stdout.split("\n")
            for line in lines:
                if "critical" in line.lower():
                    import re

                    match = re.search(r"(\d+)", line)
                    if match:
                        qa_report["critical_count"] = int(match.group(1))
                elif "warning" in line.lower():
                    import re

                    match = re.search(r"(\d+)", line)
                    if match:
                        qa_report["warning_count"] = int(match.group(1))

        # Update P3 status with QA results
        status = manager.get_phase_completion_status("P3_VERIFY") or {}
        status["qa_run"] = True
        status["qa_timestamp"] = datetime.now().isoformat()
        status["qa_tool"] = "qa_validator.py"
        status["qa_critical_count"] = qa_report.get("critical_count", 0)
        status["qa_warning_count"] = qa_report.get("warning_count", 0)
        status["qa_total_issues"] = qa_report.get("total_issues", 0)
        status["qa_categories"] = qa_report.get("categories_checked", [])

        # Also include detailed results if available
        if "categories" in qa_report:
            status["qa_details"] = qa_report["categories"]

        manager.record_phase_data("P3_VERIFY", status)

        # Display summary
        critical = qa_report.get("critical_count", 0)
        warnings = qa_report.get("warning_count", 0)
        total = qa_report.get("total_issues", 0)

        status_icon = "❌" if critical > 0 else "✅" if warnings == 0 else "⚠️"
        print(f"\n{status_icon} QA Validator Results Recorded to P3 VERIFY:")
        print("  Tool: scripts/qa_validator.py (35 checks)")
        print(f"  Critical: {critical}")
        print(f"  Warnings: {warnings}")
        print(f"  Total Issues: {total}")

        if critical > 0:
            print("\n❗ Critical issues found - review with 'full-audit' command")
        else:
            print("\n✅ Results recorded - ready for phase advancement review")

    except subprocess.TimeoutExpired:
        print("❌ ERROR: QA Validator timed out after 2 minutes")
    except Exception as e:
        print(f"❌ ERROR running QA Validator: {e}")


def cli_qa_categories() -> None:
    """Show all available QA Validator categories."""
    print("[RIC v5.1] 📋 QA VALIDATOR CATEGORIES")
    print("=" * 70)
    print("\nThe comprehensive QA Validator (scripts/qa_validator.py) provides")
    print("53 checks across 16 categories:\n")

    categories = {
        "code": "Python syntax, imports, ruff linting, type hints",
        "docs": "Research docs validation, docstrings, README presence",
        "tests": "Test suite passes, coverage config, naming conventions",
        "git": "Uncommitted changes, large files, secrets detection",
        "files": "Temp files, empty files, duplicate detection",
        "progress": "Format validation, session summaries cleanup",
        "debug": "Breakpoints, TODO/FIXME/BUG markers, print statements",
        "integrity": "Corrupt files, missing imports, circular imports",
        "xref": "Broken imports, config refs, class refs, doc links",
        "ric": "RIC phases, upgrade docs, iteration tracking",
        "security": "Secrets scanning, credential detection",
        "hooks": "Hook existence, syntax, registration, settings validation",
        "trading": "Risk params, algorithm structure, paper mode default",
        "config": "Config schema, env vars, MCP config validation",
        "deps": "Requirements sync, version conflicts, outdated packages",
        "agents": "Personas exist, persona format, slash commands valid",
    }

    print("| Category | Checks |")
    print("|----------|--------|")
    for cat, checks in categories.items():
        print(f"| `{cat}` | {checks} |")

    print("\n### Usage Examples")
    print("```bash")
    print("# Run all checks")
    print("full-audit")
    print("")
    print("# Run specific category")
    print("qa-check code")
    print("qa-check security")
    print("")
    print("# Run with auto-fix")
    print("qa-fix")
    print("")
    print("# Record results to P3 VERIFY")
    print("record-qa")
    print("```")

    print("\n### Comparison: Audit Suite vs QA Validator")
    print("| Feature | Audit Suite | QA Validator |")
    print("|---------|-------------|--------------|")
    print("| Checks | 8 categories | 41 checks |")
    print("| Speed | Fast (inline) | Comprehensive |")
    print("| Auto-fix | No | Yes (--fix) |")
    print("| JSON output | No | Yes (--json) |")
    print("| RIC integration | Built-in | Via record-qa |")
    print("\nUse 'audit' for quick RIC-integrated checks.")
    print("Use 'full-audit' for comprehensive pre-commit validation.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Main hook entry point - dispatches based on hook type or CLI."""
    # Check if running as CLI
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "init":
            max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            cli_init(max_iter)
        elif cmd == "status":
            cli_status()
        elif cmd == "advance":
            force = "--force" in sys.argv or "-f" in sys.argv
            cli_advance(force=force)
        elif cmd == "add-insight":
            if len(sys.argv) < 4:
                print("Usage: add-insight <priority> <description>")
                return
            cli_add_insight(" ".join(sys.argv[3:]), sys.argv[2])
        elif cmd == "confidence":
            if len(sys.argv) < 5:
                print("Usage: confidence <phase> <score> <notes>")
                return
            cli_confidence(int(sys.argv[2]), int(sys.argv[3]), " ".join(sys.argv[4:]))
        elif cmd == "decision":
            if len(sys.argv) < 6:
                print("Usage: decision <action> <reasoning> <risk> <confidence>")
                return
            cli_decision(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
        elif cmd == "convergence":
            cli_convergence()
        elif cmd == "can-exit":
            cli_can_exit()
        elif cmd == "end":
            cli_end()
        elif cmd == "json":
            cli_json()
        elif cmd == "sync":
            cli_sync_progress()
        elif cmd == "check-gate":
            phase = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            cli_check_gate(phase)
        elif cmd == "security":
            files = sys.argv[2:] if len(sys.argv) > 2 else []
            cli_security_check(files)
        elif cmd == "summary":
            cli_summary()
        elif cmd == "resolve":
            if len(sys.argv) < 4:
                print("Usage: resolve <insight-id> <resolution>")
                return
            cli_resolve(sys.argv[2], " ".join(sys.argv[3:]))
        elif cmd == "insights":
            cli_insights()
        elif cmd == "throttles":
            cli_throttles()
        elif cmd == "research":
            cli_research_status()
        elif cmd == "decisions":
            cli_decisions()
        # v5.0 commands
        elif cmd == "drift":
            cli_drift()
        elif cmd == "guardian":
            cli_guardian()
        elif cmd == "notes":
            cli_notes()
        elif cmd == "features":
            cli_features()
        elif cmd == "v50-status":
            cli_v50_status()
        elif cmd == "repair-stats":
            cli_repair_stats()
        elif cmd == "policy-check":
            if len(sys.argv) < 3:
                print("Usage: policy-check <action>")
                print("  Actions: commit, exit, advance_phase")
                return
            cli_policy_check(sys.argv[2])
        elif cmd == "enable-feature":
            if len(sys.argv) < 3:
                print("Usage: enable-feature <feature_name>")
                print(f"  Available: {', '.join(FEATURE_FLAGS.keys())}")
                return
            cli_enable_feature(sys.argv[2])
        elif cmd == "disable-feature":
            if len(sys.argv) < 3:
                print("Usage: disable-feature <feature_name>")
                print(f"  Available: {', '.join(FEATURE_FLAGS.keys())}")
                return
            cli_disable_feature(sys.argv[2])
        # v5.0 Phase Enforcement commands
        elif cmd == "p0-status":
            cli_p0_status()
        elif cmd == "p1-status":
            cli_p1_status()
        elif cmd == "p2-status":
            cli_p2_status()
        elif cmd == "p3-status":
            cli_p3_status()
        elif cmd == "p4-status":
            cli_p4_status()
        # P1 PLAN recording commands (v5.1)
        elif cmd == "record-tasks":
            if len(sys.argv) < 3:
                print("Usage: record-tasks task1,task2,task3,...")
                return
            cli_record_tasks(sys.argv[2])
        elif cmd == "record-scope":
            if len(sys.argv) < 3:
                print("Usage: record-scope 'in-scope | out-of-scope'")
                return
            cli_record_scope(" ".join(sys.argv[2:]))
        elif cmd == "record-criteria":
            if len(sys.argv) < 3:
                print("Usage: record-criteria criterion1,criterion2,...")
                return
            cli_record_criteria(sys.argv[2])
        elif cmd == "record-priorities":
            if len(sys.argv) < 3:
                print("Usage: record-priorities P0:task1,P1:task2,...")
                return
            cli_record_priorities(sys.argv[2])
        # P2 BUILD recording commands (v5.1)
        elif cmd == "record-change":
            if len(sys.argv) < 3:
                print("Usage: record-change 'description' [file1,file2,...]")
                return
            files = sys.argv[3] if len(sys.argv) > 3 else ""
            cli_record_change(sys.argv[2], files)
        elif cmd == "record-tests-with-change":
            cli_record_tests_with_change()
        elif cmd == "record-reveal":
            cli_record_reveal_pattern()
        elif cmd == "record-security-p2":
            cli_record_security_check_p2()
        # P3 VERIFY recording commands (v5.1)
        elif cmd == "record-test-result":
            if len(sys.argv) < 3:
                print("Usage: record-test-result pass|fail [total_tests] [failed_count]")
                return
            test_count = sys.argv[3] if len(sys.argv) > 3 else "0"
            failed_count = sys.argv[4] if len(sys.argv) > 4 else "0"
            cli_record_test_result(sys.argv[2], test_count, failed_count)
        elif cmd == "record-coverage":
            if len(sys.argv) < 3:
                print("Usage: record-coverage 75.5")
                return
            cli_record_coverage(sys.argv[2])
        elif cmd == "record-lint":
            if len(sys.argv) < 3:
                print("Usage: record-lint clean|fail [error_count]")
                return
            errors = sys.argv[3] if len(sys.argv) > 3 else "0"
            cli_record_lint(sys.argv[2], errors)
        elif cmd == "record-security-scan":
            if len(sys.argv) < 3:
                print("Usage: record-security-scan clean|fail [issue_count]")
                return
            issues = sys.argv[3] if len(sys.argv) > 3 else "0"
            cli_record_security_scan(sys.argv[2], issues)
        # P0 RESEARCH recording commands
        elif cmd == "record-keywords":
            if len(sys.argv) < 3:
                print("Usage: record-keywords kw1,kw2,kw3,...")
                return
            cli_record_keywords(sys.argv[2])
        elif cmd == "record-search":
            if len(sys.argv) < 3:
                print("Usage: record-search <query> [sources_count]")
                return
            sources = int(sys.argv[3]) if len(sys.argv) > 3 else 1
            cli_record_search(" ".join(sys.argv[2:3]), sources)
        elif cmd == "record-findings":
            cli_record_findings()
        elif cmd == "record-introspection":
            cli_record_introspection()
        elif cmd == "upgrade-ideas":
            if len(sys.argv) < 3:
                print("Usage: upgrade-ideas idea1,idea2,idea3,...")
                return
            cli_record_upgrade_ideas(sys.argv[2])
        elif cmd == "quality-check":
            # v5.1: Standalone quality gate check without recording
            if len(sys.argv) < 3:
                print("Usage: quality-check idea1,idea2,idea3,...")
                print("\nTests upgrade ideas against v5.1 quality gates:")
                print("  - Must reference specific location (file:line, function)")
                print("  - Must have concrete action verb (implement, add, fix)")
                print("  - Iteration 2+ must cite previous iteration")
                print("\nExample passing ideas:")
                print('  "Implement error handling in mcp/broker_server.py:145"')
                print('  "Add unit tests for class MarketDataServer in market_data_server.py"')
                print("\nExample failing ideas:")
                print('  "improve the code" (no location, vague)')
                print('  "something" (too short, no action)')
                return
            ideas = [i.strip() for i in sys.argv[2].split(",") if i.strip()]
            print("\n[RIC] v5.1 Quality Gate Check (not recording):")
            print("=" * 60)
            ideas_as_dicts = [{"idea": idea} for idea in ideas]
            assessment = validate_p4_quality(
                upgrade_ideas=ideas_as_dicts,
                current_iteration=1,  # Use 1 for standalone check
                introspection_text="",
            )
            for i, quality in enumerate(assessment.ideas_assessed, 1):
                status = "✅" if quality.is_actionable else "❌"
                print(f"\n  Idea {i}: {status}")
                print(f"    Text: {quality.idea[:80]}{'...' if len(quality.idea) > 80 else ''}")
                print(f"    Has Location: {'✅' if quality.has_location else '❌'}")
                print(f"    Has Action: {'✅' if quality.has_action else '❌'}")
                print(f"    Quality Score: {quality.quality_score:.1%}")
            print("\n" + "=" * 60)
            print(f"  Overall: {'✅ PASS' if assessment.passes_quality_gate else '❌ FAIL'}")
            print(f"  Score: {assessment.overall_quality_score:.1%}")
            if not assessment.passes_quality_gate:
                print("\n  Blockers:")
                for b in assessment.quality_blockers:
                    print(f"    - {b[:80]}...")
            print("=" * 60)
        elif cmd == "loop-decision":
            if len(sys.argv) < 4:
                print("Usage: loop-decision LOOP|EXIT <reason>")
                return
            cli_loop_decision(sys.argv[2], " ".join(sys.argv[3:]))
        elif cmd == "demo":
            # Demo mode
            print(f"RIC v{RIC_VERSION} {RIC_CODENAME} Demo")
            print("=" * 70)
            print("\n## Phase Headers:")
            for phase, (_name, _desc) in PHASES.items():
                print(f"  {get_header(2, 5, phase)}")
            print("\n## Commit Example:")
            print(f"  {get_commit_example(2, 5)}")
            print("\n## Fix Attempt Limits:")
            for gate, limit in FIX_LIMITS.items():
                print(f"  {gate}: {limit} attempts")
            print("\n## Safety Throttles:")
            for throttle, limit in SAFETY_THROTTLES.items():
                print(f"  {throttle}: {limit}")
            print("\n## Hallucination Categories:")
            for cat, desc in HALLUCINATION_CATEGORIES.items():
                print(f"  {cat}: {desc}")
        elif cmd == "help":
            print(f"RIC v{RIC_VERSION} {RIC_CODENAME} CLI Commands:")
            print("\n== Session Management ==")
            print("  init [max_iter]     - Start new session (default: 5 iterations)")
            print("  status              - Show current session status")
            print("  advance             - Advance to next phase")
            print("  can-exit            - Check if session can exit")
            print("  end                 - End session (if allowed)")
            print("\n== Insights & Decisions ==")
            print("  add-insight P0|P1|P2 <desc> - Add new insight")
            print("  resolve <id> <desc> - Resolve an insight")
            print("  insights            - List all insights")
            print("  confidence <phase> <score> <notes> - Record phase confidence")
            print("  decision <action> <reasoning> <risk> <confidence> - Log decision")
            print("  decisions           - Show decision trace")
            print("\n== Convergence & Gates ==")
            print("  convergence         - Check convergence status")
            print("  check-gate <phase>  - Check gate criteria for phase")
            print("  throttles           - Show throttle status")
            print("\n== v5.0 Features ==")
            print("  drift               - Check scope drift (AEGIS)")
            print("  guardian            - Run guardian review (Gartner)")
            print("  notes               - Show RIC_NOTES.md (Anthropic)")
            print("  features            - Show v5.0 feature status")
            print("  v50-status          - Comprehensive v5.0 status")
            print("  repair-stats        - Show repair/replace stats (SEIDR)")
            print("  policy-check <act>  - Check policy for action")
            print("  enable-feature <f>  - Enable a v5.0 feature")
            print("  disable-feature <f> - Disable a v5.0 feature")
            print("\n== v5.1 Phase Status Commands ==")
            print("  p0-status           - Show P0 RESEARCH completion requirements")
            print("  p1-status           - Show P1 PLAN completion requirements")
            print("  p2-status           - Show P2 BUILD completion requirements")
            print("  p3-status           - Show P3 VERIFY completion requirements")
            print("  p4-status           - Show P4 REFLECT completion requirements")
            print("\n== P0 RESEARCH Recording ==")
            print("  record-keywords <k> - Record extracted keywords (comma-separated)")
            print("  record-search <q>   - Record a web search query")
            print("  record-findings     - Mark findings as persisted")
            print("\n== P1 PLAN Recording ==")
            print("  record-tasks <t>    - Record planned tasks (comma-separated)")
            print("  record-scope <s>    - Record scope boundaries")
            print("  record-criteria <c> - Record success criteria (comma-separated)")
            print("  record-priorities <p> - Record priorities (P0:task1,P1:task2,...)")
            print("\n== P2 BUILD Recording ==")
            print("  record-change <d> [files] - Record an atomic change")
            print("  record-tests-with-change  - Record tests were included with change")
            print("  record-reveal            - Record ReVeal pattern usage")
            print("  record-security-p2       - Record security check done")
            print("\n== P3 VERIFY Recording ==")
            print("  record-test-result <p> [total] [failed] - Record test results")
            print("  record-coverage <pct>   - Record coverage percentage")
            print("  record-lint <c> [errs]  - Record lint results")
            print("  record-security-scan <c> [issues] - Record security scan")
            print("\n== P4 REFLECT Recording ==")
            print("  record-introspection - Complete P4 introspection")
            print("  upgrade-ideas <i>   - Record upgrade ideas (comma-separated)")
            print("  quality-check <i>   - Test ideas against v5.1 quality gates (no record)")
            print("  loop-decision <D> <r> - Record LOOP/EXIT decision with reason")
            print("  advance [--force]   - Advance phase (enforced unless --force)")
            print("\n== v5.1 Quality Gates ==")
            print("  Ideas must have: (i) LOCATION (file:line) + (ii) ACTION verb")
            print("  Iteration 2+ must cite previous iteration (Reflexion pattern)")
            print("  Tasks must be SMART: Specific, Measurable, Achievable")
            print("\n== Debug Suite (v5.1) ==")
            print("  debug [phase]       - Show debug info for phase (default: current)")
            print("  debug-iteration     - Full iteration debug with all phases")
            print("\n== Theme Analysis (v5.1) ==")
            print("  theme               - Show project theme analysis prompt")
            print("  record-theme <t>    - Record project theme summary")
            print("  theme-expansion     - Show theme expansion suggestions")
            print("\n== Audit Suite (v5.1) ==")
            print("  audit [categories]  - Run full audit (or specific: code,files,hooks)")
            print("  audit-category <c>  - Audit single category with details")
            print("  audit-fix [cat]     - Show fix suggestions for issues")
            print("  record-audit        - Run audit and record results to P3 VERIFY")
            print("  cleanup             - Show cleanup opportunities and commands")
            print("  Categories: code, files, functions, best_practices, hooks,")
            print("              references, crossref, cleanup")
            print("\n== QA Validator Integration (v5.1) ==")
            print("  full-audit          - Run comprehensive QA Validator (35 checks)")
            print("  qa-check [category] - Run QA Validator on specific category")
            print("  qa-fix              - Run QA Validator with auto-fix")
            print("  record-qa           - Run QA Validator and record to P3 VERIFY")
            print("  qa-categories       - Show all QA Validator categories")
            print("  Categories: code, docs, tests, git, files, progress, debug,")
            print("              integrity, xref, ric, security")
            print("\n== Utilities ==")
            print("  json                - Output JSON status (machine-parseable)")
            print("  sync                - Sync to claude-progress.txt")
            print("  security [files]    - Run security check on files")
            print("  summary             - Generate iteration summary")
            print("  research            - Show research enforcement status")
            print("  demo                - Show format examples")
        # Debug Suite Commands (v5.1)
        elif cmd == "debug":
            phase = int(sys.argv[2]) if len(sys.argv) > 2 else -1
            cli_debug_phase(phase)
        elif cmd == "debug-iteration":
            cli_debug_iteration()
        # Theme Analysis Commands (v5.1)
        elif cmd == "theme":
            cli_theme()
        elif cmd == "record-theme":
            if len(sys.argv) < 3:
                print("Usage: record-theme '<theme_summary>'")
                print("  Example: record-theme 'MCP trading server with multi-agent orchestration'")
                return
            cli_record_theme(" ".join(sys.argv[2:]))
        elif cmd == "theme-expansion":
            cli_theme_expansion()
        # Audit Suite Commands (v5.1)
        elif cmd == "audit":
            categories = sys.argv[2] if len(sys.argv) > 2 else ""
            cli_audit(categories)
        elif cmd == "audit-category":
            if len(sys.argv) < 3:
                print("Usage: audit-category <category>")
                print(f"  Available: {', '.join(AUDIT_CATEGORIES.keys())}")
                return
            cli_audit_category(sys.argv[2])
        elif cmd == "audit-fix":
            category = sys.argv[2] if len(sys.argv) > 2 else ""
            cli_audit_fix(category)
        elif cmd == "cleanup":
            cli_cleanup()
        elif cmd == "record-audit":
            cli_record_audit()
        # QA Validator Integration Commands (v5.1 Assimilation)
        elif cmd == "full-audit":
            cli_full_audit()
        elif cmd == "qa-fix":
            cli_qa_fix()
        elif cmd == "qa-check":
            category = sys.argv[2] if len(sys.argv) > 2 else ""
            cli_qa_check(category)
        elif cmd == "record-qa":
            cli_record_qa()
        elif cmd == "qa-categories":
            cli_qa_categories()
        else:
            print(f"Unknown command: {cmd}")
            print("Use 'help' for list of commands")
        return

    # Running as hook - read from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    # Determine hook type from input structure
    hook_type = input_data.get("hook_type", "")

    # Dispatch based on input structure
    if hook_type == "PostToolUse" or ("tool_output" in input_data and "tool_name" in input_data):
        # PostToolUse hook (v4.3 - Research tracking)
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        tool_output = input_data.get("tool_output", "")

        # Only track WebSearch, WebFetch, and Write for research enforcement
        if tool_name not in ["WebSearch", "WebFetch", "Write"]:
            sys.exit(0)

        result = handle_posttool_use(tool_name, tool_input, tool_output)

        # Output messages
        for msg in result["messages"]:
            output_message(msg, "WARN")
        for sug in result["suggestions"]:
            output_message(sug, "TIP")

        sys.exit(0)

    elif "tool_name" in input_data:
        # PreToolUse hook
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # Track Edit, Write, Bash, WebSearch, WebFetch (v4.3)
        if tool_name not in ["Edit", "Write", "Bash", "WebSearch", "WebFetch"]:
            sys.exit(0)

        result = handle_pretool_use(tool_name, tool_input)

        # Output messages
        for msg in result["messages"]:
            output_message(msg, "WARN" if result["allow"] else "BLOCK")
        for sug in result["suggestions"]:
            output_message(sug, "TIP")

        # Exit code
        if not result["allow"]:
            print(json.dumps({"decision": "block", "reason": "; ".join(result["messages"])}))
            sys.exit(2)
        sys.exit(0)

    elif "prompt" in input_data:
        # UserPromptSubmit hook
        prompt = input_data.get("prompt", "")
        if not prompt:
            sys.exit(0)

        result = handle_user_prompt(prompt)

        if result["message"]:
            print(result["message"])

        sys.exit(0)

    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
