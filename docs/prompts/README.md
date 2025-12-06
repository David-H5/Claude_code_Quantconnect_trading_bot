# Prompt Framework

**Current Version**: v6 (December 2025)

## Overview

This directory contains the prompt engineering framework for Claude Code agent interactions in the QuantConnect Trading Bot project.

## Current Framework

See [FRAMEWORK.md](FRAMEWORK.md) for the current prompt framework documentation.

## Version History

| Version | Date | Status | Description |
|---------|------|--------|-------------|
| [v6](versions/v6_framework.md) | Dec 2025 | **Current** | RIC Loop v5.1 Guardian with multi-agent orchestration |
| [v5](versions/v5_framework.md) | Dec 2025 | Previous | Enhanced RIC Loop with convergence detection |
| [v4](versions/v4_framework.md) | Dec 2025 | Legacy | Initial structured RIC Loop |

## Key Features (v6)

- **RIC Loop v5.1 Guardian**: 5-phase workflow with strict sequential execution
- **Multi-Agent Orchestration**: Intelligent agent selection and coordination
- **Research Enforcement**: Automatic timestamping and source validation
- **Convergence Detection**: Multi-metric tracking for iteration control
- **Safety Throttles**: Tool call limits, time limits, failure limits

## Related Documentation

- [RIC Context Quick Reference](../../.claude/RIC_CONTEXT.md)
- [CLAUDE.md Instructions](../../CLAUDE.md)
- [Agent Orchestrator](../../.claude/hooks/agents/agent_orchestrator.py)

## Directory Structure

```
docs/prompts/
├── README.md           # This file
├── FRAMEWORK.md        # Current framework (copy of latest version)
└── versions/
    ├── v4_framework.md # Legacy version
    ├── v5_framework.md # Previous version
    └── v6_framework.md # Current version
```

## Updating the Framework

When creating a new prompt framework version:

1. Create `versions/v{N}_framework.md` with the new framework
2. Copy it to `FRAMEWORK.md` as the current version
3. Update the version history table in this README
4. Update CLAUDE.md if there are workflow changes
