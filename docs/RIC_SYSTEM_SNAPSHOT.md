# RIC System Snapshot (v5.1 Guardian)

**Date**: December 5, 2025
**Purpose**: Consolidated RIC system structure reference

> **Current Implementation**: `.claude/hooks/ric_v50.py`
>
> **Quick Reference**: `.claude/RIC_CONTEXT.md`

## Overview

The RIC (Research-Implement-Converge) Loop system v5.1 "Guardian" provides structured workflow management for complex development tasks. This document shows the consolidated system structure.

---

## Final File Structure

### Core Files (ACTIVE)

| File | Lines | Purpose |
|------|-------|---------|
| `.claude/hooks/ric_v50.py` | ~6700 | **MAIN** - Standalone RIC v5.1 implementation |
| `.claude/RIC_CONTEXT.md` | ~150 | Quick reference for workflow |
| `.claude/ric_state.json` | auto | State persistence |
| `.claude/RIC_NOTES.md` | auto | Structured memory (auto-generated) |

### Support Files

| File | Purpose |
|------|---------|
| `.claude/hooks/research_saver.py` | Research document creator |
| `.claude/hooks/agent_orchestrator.py` | Multi-agent orchestration |
| `.claude/hooks/session_stop_v2.py` | Continuous mode exit handling |

### Slash Commands

| Command | File | Description |
|---------|------|-------------|
| `/ric-start` | `.claude/commands/ric-start.md` | Initialize RIC session |
| `/ric-research` | `.claude/commands/ric-research.md` | Phase 0 research protocol |
| `/ric-introspect` | `.claude/commands/ric-introspect.md` | Phase 4 introspection |
| `/ric-converge` | `.claude/commands/ric-converge.md` | Convergence check |

### Documentation

| File | Description |
|------|-------------|
| `docs/development/RIC_LOOP_ENFORCEMENT.md` | System documentation |
| `docs/development/ENHANCED_RIC_WORKFLOW.md` | Workflow details |
| `docs/research/RIC-V46-RESEARCH-DEC2025.md` | Source research (35+ sources) |
| `docs/research/RIC-V51-QUALITY-GATES-RESEARCH.md` | Latest research |
| `docs/research/UPGRADE-016-RIC-V50.md` | Implementation status |

---

## Deprecated Files (Historical Reference)

All deprecated files are in `.claude/hooks/deprecated/`:

| File | Reason |
|------|--------|
| `ric_v45.py` | Superseded by ric_v50.py |
| `ric_v50_dev.py` | Development complete, merged |
| `ric_state_manager.py` | Merged into ric_v50.py |
| `ric_prompts.py` | Merged into ric_v50.py |
| `ric_hooks.py` | Merged into ric_v50.py |
| `ric_enforcer.py` | Merged into ric_v50.py |
| `enforce_ric_compliance.py` | Merged into ric_v50.py |

Additionally in `.claude/deprecated/`:

| File | Reason |
|------|--------|
| `RIC_ENFORCEMENT.md` | Outdated, references deprecated files |

---

## Settings.json Hook Configuration

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Edit|Write|Bash",
      "hooks": [{
        "type": "command",
        "command": "python3 .claude/hooks/ric_v50.py",
        "statusMessage": "RIC v5.0 compliance check",
        "timeout": 10
      }]
    }],
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "python3 .claude/hooks/ric_v50.py",
        "statusMessage": "RIC v5.0 enforcement",
        "timeout": 10
      }]
    }],
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "python3 .claude/hooks/ric_v50.py status 2>/dev/null || true",
        "timeout": 5
      }]
    }]
  }
}
```

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v5.1 | 2025-12-05 | Quality gates, SELF-REFINE, Reflexion |
| v5.0 | 2025-12-04 | Guardian mode, drift detection, SEIDR |
| v4.5 | 2025-12-04 | Hallucination detection, convergence |
| v4.2 | 2025-12-03 | SELF-REFINE, CoCoGen, SAGE patterns |
| v3.0 | 2025-12-02 | 7-phase workflow (deprecated) |

---

## CLI Quick Reference

```bash
# Initialize
python3 .claude/hooks/ric_v50.py init 5

# Status
python3 .claude/hooks/ric_v50.py status
python3 .claude/hooks/ric_v50.py json

# Navigation
python3 .claude/hooks/ric_v50.py advance
python3 .claude/hooks/ric_v50.py can-exit

# Insights
python3 .claude/hooks/ric_v50.py add-insight P0 "desc"
python3 .claude/hooks/ric_v50.py resolve INS-001 "fix"
python3 .claude/hooks/ric_v50.py insights

# Safety
python3 .claude/hooks/ric_v50.py convergence
python3 .claude/hooks/ric_v50.py throttles
python3 .claude/hooks/ric_v50.py security

# Help
python3 .claude/hooks/ric_v50.py help
```

---

## Related Documentation

- [RIC Quick Reference](../.claude/RIC_CONTEXT.md)
- [RIC Enforcement](development/RIC_LOOP_ENFORCEMENT.md)
- [RIC Workflow](development/ENHANCED_RIC_WORKFLOW.md)
