# CLAUDE.md Restructuring Master Plan

## Executive Summary

**Current State**: 3,173 lines | 60+ sections | 149 subsections
**Best Practice**: <300 lines | 5-7 sections | Ideally ~60-100 lines
**Verdict**: Current file is **50x larger** than recommended

---

## Analysis 1: Current CLAUDE.md Structure

### Metrics

| Metric | Current | Best Practice | Deviation |
|--------|---------|---------------|-----------|
| Total Lines | 3,173 | <300 | **10.6x over** |
| Major Sections (##) | 60+ | 5-7 | **10x over** |
| Subsections (###) | 149 | 10-15 | **10x over** |
| Sub-subsections (####) | 16 | 0-5 | 3x over |

### Content Categories Found

| Category | Lines (est.) | Should Be In CLAUDE.md? |
|----------|--------------|------------------------|
| RIC Loop Workflow | ~175 | Partial (summary only) |
| TAP Protocol | ~80 | Yes (core rule) |
| Project Overview | ~15 | Yes |
| Directory Structure | ~80 | Reference only |
| Key Commands | ~90 | Yes (condensed) |
| Safety Guidelines | ~55 | Yes |
| QuantConnect Essentials | ~200 | No → separate doc |
| Options Trading Patterns | ~155 | No → separate doc |
| Agent Personas | ~45 | No → separate doc |
| Multi-Agent Orchestration | ~160 | Reference only |
| LLM Dashboard | ~115 | No → separate doc |
| Research Documentation | ~280 | No → separate doc |
| RCA/Postmortem Templates | ~80 | No → separate doc |
| Execution Quality Metrics | ~60 | No → separate doc |
| Docker/MCP Config | ~50 | Reference only |
| Historical/Changelog | ~100+ | No → remove |
| Scattered Notes | ~200+ | No → consolidate |

### Critical Issues Identified

1. **Instruction Overload**: Frontier LLMs can reliably follow ~150-200 instructions. Claude Code's system prompt already uses ~50, leaving ~100-150. Current CLAUDE.md has **300+ instructions**.

2. **Task-Specific Content**: Contains content that's only relevant to specific tasks (trading patterns, backtest parameters), causing distraction during unrelated work.

3. **Outdated Content**: Contains historical notes, completed sprints, and changelog entries that should be removed.

4. **Code Duplication**: Contains code snippets that will become outdated. Should use `file:line` references instead.

5. **Linter's Job**: Contains code style guidelines that deterministic tools (ruff, black, mypy) should handle.

6. **Missing Progressive Disclosure**: Instead of referencing separate docs, everything is inline.

---

## Analysis 2: Best Practices Research

### Sources Consulted

1. [Anthropic Official Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
2. [HumanLayer: Writing a Good CLAUDE.md](https://www.humanlayer.dev/blog/writing-a-good-claude-md)
3. [Notes on CLAUDE.md Structure](https://callmephilip.com/posts/notes-on-claude-md-structure-and-best-practices/)
4. [Claude Code Complete Guide](https://www.siddharthbharath.com/claude-code-the-complete-guide/)

### Key Recommendations

#### From Anthropic (Official)

> "Keep CLAUDE.md files concise and human-readable. Your files become part of Claude's prompts, so they should be refined like any frequently used prompt."

> "A common mistake is adding extensive content without iterating on its effectiveness."

#### From HumanLayer

> "< 300 lines is best, and shorter is even better. Our root CLAUDE.md file is less than sixty lines."

> "Content must be universally applicable across sessions. Avoid task-specific instructions."

> "Use Progressive Disclosure: Rather than stuffing all information into CLAUDE.md, create separate markdown files and reference them."

#### From Community Best Practices

> "Never send an LLM to do a linter's job. Use deterministic tools for code style."

> "Prefer pointers over copies. Use file:line references rather than code snippets."

> "The Claude Code system includes: 'this context may or may not be relevant to your tasks. You should not respond...unless highly relevant.' Non-universal instructions get uniformly ignored."

### Recommended Structure (WHY, WHAT, HOW)

```markdown
# Project Name

## Overview (WHY)
Brief project purpose

## Tech Stack (WHAT)
- Language/Framework versions
- Key dependencies

## Project Structure (WHAT)
Brief directory overview with references to detailed docs

## Commands (HOW)
Essential commands only

## Workflow (HOW)
Core development workflow rules

## References
Links to detailed documentation
```

---

## Comparison: Current vs Best Practice

| Aspect | Current | Best Practice | Gap |
|--------|---------|---------------|-----|
| **Length** | 3,173 lines | <300 lines | 2,873 lines to remove |
| **Focus** | Everything | Universal instructions | Remove task-specific |
| **Code Snippets** | 50+ inline | References only | Move to docs |
| **Style Rules** | Extensive | None (use linters) | Delete entirely |
| **Historical** | Included | None | Delete entirely |
| **Templates** | Inline | Separate files | Move to .claude/templates/ |
| **Trading Docs** | Inline | Separate files | Move to docs/ |
| **Agent Docs** | Inline | Separate files | Move to docs/ |

---

## Master Restructuring Plan

### Phase 1: Extract Content to Separate Files

| Current Section | New Location | Action |
|-----------------|--------------|--------|
| QuantConnect Essentials | `docs/QUANTCONNECT_GUIDE.md` | Move |
| Options Trading Patterns | `docs/OPTIONS_PATTERNS.md` | Move |
| Agent Personas | `docs/AGENT_PERSONAS.md` | Move |
| Multi-Agent Orchestration | `docs/AGENT_ORCHESTRATION.md` | Move (detailed) |
| LLM Dashboard | `docs/LLM_DASHBOARD.md` | Move |
| Research Documentation | `docs/RESEARCH_GUIDE.md` | Move |
| RCA Templates | `.claude/templates/rca_template.md` | Move |
| Postmortem Templates | `.claude/templates/postmortem_template.md` | Move |
| Execution Metrics | `docs/EXECUTION_METRICS.md` | Move |
| Strategy Documentation | `docs/STRATEGY_GUIDE.md` | Move |
| Historical/Changelog | Delete or `docs/CHANGELOG.md` | Remove |
| RIC Loop Details | `.claude/RIC_CONTEXT.md` | Already exists, reference |

### Phase 2: Create New CLAUDE.md Structure

**Target: ~150-200 lines**

```markdown
# Claude Code Instructions - QuantConnect Trading Bot

## Quick Reference
- Purpose: Semi-autonomous options trading bot
- Platform: QuantConnect/LEAN + Charles Schwab
- Language: Python 3.10+

## Critical Rules (TAP Protocol)
LOG → CHECK → TEST → INTEGRATE → VERIFY → LOG RESULT
[Condensed 10-line version]

## Essential Commands
[10 most-used commands only]

## Directory Overview
[5-line summary with reference to docs/ARCHITECTURE.md]

## Development Workflow
- Use RIC Loop for complex tasks (@.claude/RIC_CONTEXT.md)
- Run tests before commits: pytest tests/ -v
- Never deploy untested code to live trading

## Safety Rules
- Circuit breaker integration required
- All trades require human approval in production
- See docs/SAFETY_GUIDE.md for details

## References
- RIC Loop: @.claude/RIC_CONTEXT.md
- Architecture: @docs/ARCHITECTURE.md
- Trading Patterns: @docs/OPTIONS_PATTERNS.md
- Agent System: @docs/AGENT_ORCHESTRATION.md
```

### Phase 3: Create Agent Documentation Directory

```
docs/
├── CLAUDE_QUICK_REF.md       # Ultra-condensed reference
├── ARCHITECTURE.md           # Directory structure, modules
├── QUANTCONNECT_GUIDE.md     # QuantConnect essentials
├── OPTIONS_PATTERNS.md       # Trading patterns
├── AGENT_ORCHESTRATION.md    # Multi-agent system
├── AGENT_PERSONAS.md         # Agent descriptions
├── SAFETY_GUIDE.md           # Trading safety
├── EXECUTION_METRICS.md      # Quality metrics
├── STRATEGY_GUIDE.md         # Strategy documentation
├── RESEARCH_GUIDE.md         # Research process
└── LLM_DASHBOARD.md          # Dashboard usage
```

### Phase 4: Delete/Archive Obsolete Content

| Content Type | Action |
|--------------|--------|
| Sprint-specific notes | Delete |
| Completed task lists | Delete |
| Historical decisions | Archive to `docs/DECISIONS_LOG.md` |
| Changelog entries | Archive to `docs/CHANGELOG.md` |
| Inline code examples | Replace with file references |

---

## New CLAUDE.md Template (Target: ~180 lines)

```markdown
# Claude Code Instructions - QuantConnect Trading Bot

## Overview
Semi-autonomous options trading bot with LLM-powered analysis.
Platform: QuantConnect/LEAN with Charles Schwab brokerage.
Language: Python 3.10+ | Testing: pytest (70% coverage)

## Critical: TAP Protocol (Thorough Action Protocol)

Every change MUST follow: LOG → CHECK → TEST → INTEGRATE → VERIFY → LOG

1. Log what you're starting: `[TAP] Starting: <action>`
2. Check prerequisites exist
3. Implement the change
4. Run tests: `pytest <test_file> -v`
5. Verify integration: `python3 -c "from X import Y"`
6. Log result: `[TAP] Complete: <summary>`

**Never skip tests. Never create without integrating. Never assume - verify.**

## Essential Commands

| Command | Description |
|---------|-------------|
| `pytest tests/ -v` | Run all tests |
| `ruff check .` | Lint code |
| `mypy .` | Type check |
| `/ric-start <task>` | Start RIC Loop for complex tasks |
| `/agents auto <task>` | Auto-route to best agents |

## Project Structure

```
algorithms/     # QuantConnect trading algorithms
llm/agents/     # LLM trading agents (primary)
execution/      # Order execution (profit-taking, smart-exec)
scanners/       # Market scanners (options, movement)
models/         # Risk models, circuit breaker
tests/          # Unit and integration tests
```

See: @docs/ARCHITECTURE.md for full structure

## Development Workflow

### For Complex Tasks (multi-file, features, refactoring)
Use RIC Loop: Research → Plan → Build → Verify → Reflect
See: @.claude/RIC_CONTEXT.md

### For Simple Tasks (single file, bug fixes)
1. Make change
2. Run relevant tests
3. Verify imports work
4. Commit with clear message

## Safety Rules

- **NEVER** deploy untested code to live trading
- **ALWAYS** verify circuit breaker integration
- **REQUIRE** human approval for production trades
- **TEST** in paper trading before live

See: @docs/SAFETY_GUIDE.md

## Key References

| Topic | File |
|-------|------|
| RIC Loop | @.claude/RIC_CONTEXT.md |
| Architecture | @docs/ARCHITECTURE.md |
| Trading Patterns | @docs/OPTIONS_PATTERNS.md |
| Agent System | @docs/AGENT_ORCHESTRATION.md |
| Safety | @docs/SAFETY_GUIDE.md |
| QuantConnect | @docs/QUANTCONNECT_GUIDE.md |

## Code Quality

Use deterministic tools (not CLAUDE.md instructions):
- Formatting: `ruff format .`
- Linting: `ruff check . --fix`
- Types: `mypy .`

## Common Pitfalls

- Don't modify live algorithm without backtest
- Don't skip paper trading validation
- Don't commit without running tests
- Don't ignore circuit breaker warnings
```

---

## Migration Checklist

### Step 1: Create New Documentation Files
- [ ] `docs/ARCHITECTURE.md` (from Directory Structure section)
- [ ] `docs/QUANTCONNECT_GUIDE.md` (from QuantConnect Essentials)
- [ ] `docs/OPTIONS_PATTERNS.md` (from Options Trading Patterns)
- [ ] `docs/AGENT_ORCHESTRATION.md` (from Multi-Agent sections)
- [ ] `docs/AGENT_PERSONAS.md` (from Agent Personas)
- [ ] `docs/SAFETY_GUIDE.md` (from Safety sections)
- [ ] `docs/EXECUTION_METRICS.md` (from Execution Quality)
- [ ] `docs/RESEARCH_GUIDE.md` (from Research Documentation)
- [ ] `docs/LLM_DASHBOARD.md` (from Dashboard sections)
- [ ] `.claude/templates/rca_template.md` (from RCA section)
- [ ] `.claude/templates/postmortem_template.md` (from Postmortem section)

### Step 2: Archive Historical Content
- [ ] Create `docs/CHANGELOG.md` with historical entries
- [ ] Create `docs/DECISIONS_LOG.md` with key decisions

### Step 3: Rewrite CLAUDE.md
- [ ] Backup current: `cp CLAUDE.md CLAUDE.md.backup`
- [ ] Write new condensed version (~180 lines)
- [ ] Add `@` references to extracted docs
- [ ] Test that Claude can still access needed info

### Step 4: Verify
- [ ] Claude can find project info
- [ ] Claude follows TAP protocol
- [ ] Claude knows where to find detailed docs
- [ ] Commands still work
- [ ] No critical information lost

---

## Expected Outcomes

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Lines | 3,173 | ~180 | **94% reduction** |
| Sections | 60+ | 8 | **87% reduction** |
| Load Time | Slow | Fast | Significant |
| Instruction Clarity | Low | High | Major |
| Universality | Low | High | Major |
| Maintainability | Poor | Good | Major |

---

## Timeline

| Phase | Duration | Priority |
|-------|----------|----------|
| Phase 1: Extract to docs | 1-2 hours | High |
| Phase 2: New CLAUDE.md | 30 min | High |
| Phase 3: Agent docs | 1 hour | Medium |
| Phase 4: Cleanup | 30 min | Medium |
| Verification | 30 min | High |

**Total Estimated Time**: 3-4 hours

---

## References

- [Anthropic: Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [HumanLayer: Writing a Good CLAUDE.md](https://www.humanlayer.dev/blog/writing-a-good-claude-md)
- [Notes on CLAUDE.md Structure](https://callmephilip.com/posts/notes-on-claude-md-structure-and-best-practices/)
- [Claude Code Complete Guide](https://www.siddharthbharath.com/claude-code-the-complete-guide/)
