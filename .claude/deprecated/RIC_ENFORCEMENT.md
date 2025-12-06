# RIC Auto-Enforcement System (v1.0)

> **⚠️ DEPRECATED**: This document describes an outdated enforcement system.
>
> **Superseded by**: RIC v5.1 Guardian (`.claude/hooks/ric_v50.py`)
> **Date**: December 5, 2025
> **Reason**: All enforcement features merged into standalone ric_v50.py
> **See Instead**: `.claude/RIC_CONTEXT.md` for current quick reference

---

## Overview (HISTORICAL)

The RIC Auto-Enforcement System ensures Claude completes ALL RIC workflow steps without skipping, skimping, or forgetting. It uses **AUTO-INJECTION** instead of blocking or warnings.

## Philosophy

| Approach | Behavior | Result |
|----------|----------|--------|
| ~~Blocking~~ | Stops session completely | ❌ Bad - interrupts autonomous work |
| ~~Warnings~~ | Can be ignored/skipped | ❌ Bad - steps get missed |
| **Auto-Injection** | Forces required actions | ✅ Good - guarantees completion |

## How It Works

```
User Prompt
    │
    ▼
┌─────────────────────────────────────┐
│   Enforcement Hook Checks:          │
│   1. Any required actions pending?  │
│   2. Phase gate criteria met?       │
│   3. Skip attempt detected?         │
│   4. Loop back required?            │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│   AUTO-INJECT into Claude's context:│
│   - MANDATORY ACTION prompts        │
│   - Specific steps to complete      │
│   - Verification instructions       │
└─────────────────────────────────────┘
    │
    ▼
Claude receives injected context
and MUST complete required actions
before proceeding
```

## Components

### 1. RIC Enforcer (`ric_enforcer.py`)

The main enforcement engine that:
- Tracks checklist items per phase
- Validates gate completion
- Forces loops when exit criteria not met
- Generates auto-inject prompts

**CLI Commands:**

```bash
# Initialize for a new upgrade
python3 .claude/hooks/ric_enforcer.py init --upgrade-id UPGRADE-XXX

# Check status
python3 .claude/hooks/ric_enforcer.py status

# View required actions
python3 .claude/hooks/ric_enforcer.py required

# Validate phase completion
python3 .claude/hooks/ric_enforcer.py validate

# Check if loop required
python3 .claude/hooks/ric_enforcer.py loop-check

# Full report
python3 .claude/hooks/ric_enforcer.py report

# Mark item complete
python3 .claude/hooks/ric_enforcer.py complete --description "research" --evidence "3 searches done"
```

### 2. Enforcement Hook (`enforce_ric_compliance.py`)

UserPromptSubmit hook that:
- Checks for pending requirements
- Detects phase transitions
- Catches skip attempts
- Injects mandatory actions

### 3. Persistent Tracking

State files:
- `.claude/enforcement_state.json` - Checklist and completion tracking
- `.claude/ric_state.json` - RIC Loop iteration/phase state
- `.claude/logs/enforcement_injections.json` - Log of injected actions

## Phase Requirements

Each phase has MANDATORY items that MUST be completed:

| Phase | Requirements |
|-------|-------------|
| **0. Research** | 3+ timestamped web searches, Research doc created |
| **1. Upgrade Path** | Target state defined, Success criteria listed |
| **2. Checklist** | P0/P1/P2 tasks classified |
| **3. Coding** | ONE component at a time, Tests per component, Checkpoint commit |
| **4. Double-Check** | All tests pass, Coverage > 70% |
| **5. Introspection** | Gap analysis complete, Debt/bugs documented |
| **6. Metacognition** | All insights classified P0/P1/P2 |
| **7. Integration** | Exit criteria validated |

## Enforcement Behavior

### Required Actions

When phase requirements are incomplete, Claude receives:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ⚠️  MANDATORY ACTIONS REQUIRED  ⚠️                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  The RIC Enforcement System has detected incomplete requirements.            ║
║  You MUST complete these actions BEFORE proceeding with other work.          ║
╚══════════════════════════════════════════════════════════════════════════════╝

### REQUIRED ACTION 1 [P0] - Phase 0

**Task**: Conduct minimum 3 web searches with timestamps

**Exact Steps**:
Execute WebSearch for '{topic}' and document findings with timestamps

**Completion Verification**:
3+ timestamped searches in research doc
```

### Phase Gate Enforcement

When attempting to advance without completing requirements:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚧 PHASE GATE - REQUIREMENTS INCOMPLETE 🚧                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Cannot advance to next phase. Complete these items first:                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

**Current Phase**: 0 - Research

**Missing Requirements** (2):
  - [P0] Conduct minimum 3 web searches with timestamps
  - [P0] Create/update research document
```

### Loop Enforcement

When trying to exit before minimum iterations:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔄 LOOP TO PHASE 0 REQUIRED 🔄                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Cannot proceed or exit. Must loop back to Phase 0 (Research).               ║
╚══════════════════════════════════════════════════════════════════════════════╝

**Reason**: Minimum 3 iterations required (current: 1)
```

### Skip Detection

When Claude attempts to skip required work:

```
⚠️ **SKIP ATTEMPT DETECTED** - This has been logged.

RIC Enforcement does not allow skipping required items.
The item will remain on your checklist until completed.
```

## Configuration

### Enforcement Levels

Set via environment variable:

```bash
# Full enforcement (default) - all checks, all injections
export RIC_ENFORCEMENT=FULL

# Light enforcement - only critical checks
export RIC_ENFORCEMENT=LIGHT

# Disabled - no enforcement
export RIC_ENFORCEMENT=DISABLED
```

### Minimum Requirements

In `ric_enforcer.py`:

```python
MINIMUM_REQUIREMENTS = {
    "min_iterations": 3,           # Must complete 3 full loops
    "min_research_searches": 3,    # 3 web searches minimum
    "min_test_coverage": 70,       # 70% coverage required
    "min_components_per_phase3": 1, # At least 1 component per coding phase
}
```

## Integration with Hooks

### Settings.json Configuration

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 .claude/hooks/ric_enforcer.py status 2>/dev/null || true",
            "statusMessage": "Checking RIC enforcement status"
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 .claude/hooks/enforce_ric_compliance.py",
            "statusMessage": "Checking RIC compliance enforcement"
          }
        ]
      }
    ]
  }
}
```

## Workflow Example

### Starting a New Upgrade

```bash
# Initialize RIC state
python3 .claude/hooks/ric_state_manager.py init --upgrade-id UPGRADE-016 --title "New Feature"

# Initialize enforcement tracking
python3 .claude/hooks/ric_enforcer.py init --upgrade-id UPGRADE-016
```

### During Development

1. **Each user prompt** triggers the enforcement hook
2. Hook checks for incomplete requirements
3. If incomplete, Claude receives MANDATORY action prompts
4. Claude MUST complete before proceeding

### Completing Items

Items are marked complete through:

1. **Evidence-based**: When Claude provides evidence (commit hash, test output, etc.)
2. **Manual**: `python3 .claude/hooks/ric_enforcer.py complete --description "..." --evidence "..."`
3. **Automatic**: Some checks can auto-detect completion (file exists, tests pass, etc.)

### Phase Transitions

When Claude says "next phase" or "continue":

1. Enforcement validates all current phase items are complete
2. If not complete, injects specific missing requirements
3. Claude completes requirements
4. Phase advances

### Exit Prevention

At Phase 7 (Integration):

1. Enforcement checks minimum iterations (3)
2. Checks all P0/P1/P2 insights resolved
3. If not met, forces loop to Phase 0
4. Counts forced loops for transparency

## Logs and Debugging

### View Enforcement Log

```bash
cat .claude/logs/enforcement_injections.json | python3 -m json.tool | tail -50
```

### View Enforcement State

```bash
cat .claude/enforcement_state.json | python3 -m json.tool
```

### Full Report

```bash
python3 .claude/hooks/ric_enforcer.py report
```

## Key Benefits

1. **No Session Interruption**: Never blocks, only injects
2. **Persistent Tracking**: Survives restarts, context compaction
3. **Forced Completion**: Cannot skip required items
4. **Iteration Enforcement**: Guarantees minimum 3 loops
5. **Transparent**: Logs all injections and skip attempts
6. **Evidence-Based**: Tracks completion evidence

## Troubleshooting

### Enforcement Not Triggering

1. Check `RIC_ENFORCEMENT` env var is not "DISABLED"
2. Verify hooks are configured in `.claude/settings.json`
3. Check enforcement state file exists

### False Positives

1. Mark item complete manually with evidence
2. Adjust phase requirements in `PHASE_REQUIREMENTS` dict

### Too Many Injections

1. Use `RIC_ENFORCEMENT=LIGHT` for reduced checks
2. Complete items promptly to clear backlog
