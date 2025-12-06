# Multi-Agent Orchestration Suite Research

## 📋 Research Overview

**Date**: December 4, 2025
**Scope**: Design and implementation of intelligent multi-agent coordination system
**Focus**: Spawning, managing, and coordinating multiple Claude agents with model selection
**Result**: Complete orchestration suite with 10 slash commands and 1000+ line engine

---

## 🎯 Research Objectives

1. Enable parallel execution of multiple Claude agents
2. Implement intelligent model selection based on task complexity
3. Create workflow patterns (parallel, sequential, consensus)
4. Build simple command interface for humans and autonomous agents
5. Optimize for cost-effectiveness (haiku vs sonnet vs opus)

---

## 📊 Key Discoveries

### Claude Code Task Tool Capabilities

**Search Date**: December 4, 2025

The Task tool natively supports:

- **Model Selection**: `model` parameter accepts "haiku", "sonnet", "opus"
- **Agent Types**: `subagent_type` for specialized behavior
  - `Explore` - Fast codebase exploration
  - `Plan` - Architecture and planning
  - `general-purpose` - Complex multi-step tasks
  - `claude-code-guide` - Documentation queries
- **Parallel Execution**: Multiple Task calls in single response run simultaneously

### Model Characteristics

| Model | Speed | Cost | Best For |
|-------|-------|------|----------|
| **haiku** | ~5s | $0.25/$1.25 per MTok | Search, grep, simple checks |
| **sonnet** | ~15s | $3/$15 per MTok | Implementation, review |
| **opus** | ~30s | $15/$75 per MTok | Architecture, critical decisions |

### Task Complexity Classification

Developed keyword-based classification system:

| Complexity | Keywords | Model |
|------------|----------|-------|
| TRIVIAL | find, search, grep, list | haiku |
| SIMPLE | read, summarize, explain | haiku |
| MODERATE | implement, fix, refactor | sonnet |
| COMPLEX | design, architect, optimize | sonnet |
| CRITICAL | security, trading, production | opus |

---

## 🔧 Implementation

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Orchestrator Suite                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   /agents    │    │ Orchestrator │    │    Config    │       │
│  │   Commands   │───▶│    Engine    │◀───│    JSON      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         ▼                   ▼                   ▼               │
│  ┌────────────┐      ┌────────────┐      ┌────────────┐         │
│  │   Model    │      │  Workflow  │      │   Agent    │         │
│  │  Selector  │      │  Patterns  │      │ Templates  │         │
│  └────────────┘      └────────────┘      └────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Workflow Patterns

1. **Parallel** - All agents run simultaneously
   - Best for: Independent searches, multi-perspective review
   - Example: 4-agent code review (security + types + tests + architecture)

2. **Sequential** - Output chains to next agent
   - Best for: Implementation pipelines
   - Example: Research → Plan → Implement → Review

3. **Consensus** - Multiple agents vote independently
   - Best for: Critical decisions requiring multiple perspectives
   - Threshold: 66% (2/3 majority) for approval

4. **Hierarchical** - Manager coordinates workers
   - Best for: Complex coordinated reviews
   - Example: Manager assigns and aggregates specialist findings

### Agent Templates

**Search Agents** (haiku):

- `CodeFinder` - Search source code patterns
- `DocFinder` - Search documentation
- `TestFinder` - Search test files

**Review Agents** (haiku/sonnet):

- `SecurityScanner` - OWASP vulnerability scanning
- `TypeChecker` - Type hint validation
- `TestAnalyzer` - Coverage analysis
- `Architect` - Architecture review

**Trading Agents** (sonnet):

- `RiskReviewer` - Circuit breaker, position limits
- `ExecutionReviewer` - Order execution logic
- `BacktestReviewer` - Look-ahead bias detection

**Deep Analysis** (opus):

- `DeepArchitect` - Comprehensive architecture analysis
- `CriticalReviewer` - Production code review

---

## 📁 Files Created

### Core Engine

| File | Lines | Purpose |
|------|-------|---------|
| `.claude/hooks/agent_orchestrator.py` | ~1000 | Main orchestration engine |
| `.claude/agent_config.json` | ~50 | Configuration and preferences |
| `.claude/hooks/multi_agent.py` | ~300 | Utility patterns (legacy) |

### Slash Commands

| Command File | Purpose |
|--------------|---------|
| `.claude/commands/agents.md` | Master command hub |
| `.claude/commands/agent-auto.md` | Intelligent auto-routing |
| `.claude/commands/agent-quick.md` | Single fast agent |
| `.claude/commands/agent-swarm.md` | 8-agent parallel exploration |
| `.claude/commands/agent-consensus.md` | Multi-agent voting |
| `.claude/commands/agent-implement.md` | Implementation pipeline |
| `.claude/commands/agent-compare.md` | Option comparison |
| `.claude/commands/agent-status.md` | System statistics |
| `.claude/commands/parallel-review.md` | 4-agent code review |
| `.claude/commands/multi-search.md` | 3-agent search |
| `.claude/commands/trading-review.md` | Trading safety review |

---

## 📖 Usage Guide

### Quick Start

```bash
# Intelligent auto-selection (recommended)
/agents auto <any task description>

# Specific patterns
/agent-quick <simple task>       # Single haiku agent
/agent-swarm <topic>             # 8-agent exploration
/agent-consensus <decision>      # 3-agent voting
/agent-implement <feature>       # Full pipeline
/agent-compare A vs B            # Compare options
```

### Examples

```bash
# Search tasks → haiku agents
/agents auto find all error handling code
/agents auto where is authentication implemented

# Review tasks → mixed agents
/agents auto review the execution module
/agents auto check trading safety in algorithms/

# Design tasks → sonnet/opus
/agents auto design caching strategy
/agents auto should we refactor the scanner

# Implementation → sequential pipeline
/agent-implement add IV rank filter to options scanner
```

### CLI Usage

```bash
# Show help
python3 .claude/hooks/agent_orchestrator.py help

# Auto-select for task
python3 .claude/hooks/agent_orchestrator.py auto "find auth code"

# List resources
python3 .claude/hooks/agent_orchestrator.py list

# Generate Task calls
python3 .claude/hooks/agent_orchestrator.py generate code_review

# Show statistics
python3 .claude/hooks/agent_orchestrator.py status
```

---

## 💡 Cost Optimization

### Strategy

1. **Default to haiku** for all search and simple tasks
2. **Use sonnet** only for implementation and complex review
3. **Reserve opus** for architecture and critical decisions
4. **Batch searches** - spawn 3-8 haiku agents in parallel

### Cost Comparison (per 1M tokens)

| Pattern | Models | Input Cost | Output Cost |
|---------|--------|------------|-------------|
| 3-agent search | 3x haiku | $0.75 | $3.75 |
| 4-agent review | 3 haiku + 1 sonnet | $3.75 | $16.25 |
| 3-agent consensus | 3x sonnet | $9.00 | $45.00 |
| Deep architecture | 1x opus | $15.00 | $75.00 |

---

## 🔄 Integration with Autonomous Sessions

The orchestration suite is designed for use by both humans and autonomous Claude agents:

1. **Slash commands** work in any Claude Code session
2. **CLI interface** can be called from hooks or scripts
3. **State tracking** persists across sessions
4. **JSON output** available for programmatic use

### Autonomous Usage Pattern

```python
# In overnight session, Claude can call:
/agents auto complete the remaining implementation tasks

# Or for specific patterns:
/agent-swarm the authentication module
/agent-consensus should we deploy to production
```

---

## 📝 Change Log

| Date | Change | Impact |
|------|--------|--------|
| 2025-12-04 | Initial implementation | Full orchestration suite |
| 2025-12-04 | Added 10 slash commands | Easy human/agent access |
| 2025-12-04 | Updated CLAUDE.md | Comprehensive documentation |

---

## 🔗 Related Documentation

- [CLAUDE.md - Multi-Agent Section](../../CLAUDE.md#multi-agent-orchestration-suite)
- [Agent Personas](../../.claude/agents/)
- [Autonomous Agents Guide](../autonomous-agents/README.md)
