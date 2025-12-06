# UPGRADE-018: Agent Swarm Cost-Optimized Enhancement

## Overview

**Date**: December 5, 2025
**Status**: Planning
**Priority**: High
**Estimated Effort**: 8-12 hours

### Summary

Enhance the existing Agent Orchestration Suite (UPGRADE-017) with cost-optimized quick agents, RIC workflow integration, and auto-persistence for research findings. Focus on using Haiku for simple tasks and reserving Opus for critical decisions only.

---

## Problem Statement

### Current State

The agent orchestration system (`.claude/hooks/agent_orchestrator.py`) has:
- 14 agent templates
- 6 workflow templates
- Model selection based on complexity
- State tracking and statistics

### Gaps Identified

| Gap | Impact | Priority |
|-----|--------|----------|
| No dedicated web research agent | Can't auto-scrape and save to docs | P0 |
| No simple file operations agents | Manual text copying required | P1 |
| No RIC phase integration | Agents disconnected from RIC workflow | P0 |
| No auto-persistence for research | Web findings not saved | P1 |
| Some agents over-modeled | Using Sonnet where Haiku works | P1 |
| No cost tracking | Can't monitor actual spend | P2 |

### User Requirements

> "I want to keep Claude Opus for important thinking and coding and upgrading, but I want quick agents for file searching, text copying, online search scraping and saving to research documents, and anything else that doesn't require high intelligence."

---

## Solution Design

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    UPGRADE-018 Enhancement Layers                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Layer 1: Quick Haiku Agents (New)                                  │
│  ├── web_researcher       - Web search + auto-save to docs          │
│  ├── text_extractor       - Extract/copy text from files            │
│  ├── grep_agent           - Fast pattern search                     │
│  ├── file_lister          - List files by pattern                   │
│  └── research_saver       - Save findings to docs/research/         │
│                                                                      │
│  Layer 2: RIC Integration (New)                                     │
│  ├── ric_research         - P0 Research workflow                    │
│  ├── ric_verify           - P3 Verify workflow                      │
│  └── /ric-agents command  - Suggest agents per RIC phase            │
│                                                                      │
│  Layer 3: Auto-Persistence (New)                                    │
│  ├── ResearchPersister    - Auto-save web research                  │
│  └── Integration hooks    - Connect to WebSearch tool               │
│                                                                      │
│  Layer 4: Cost Optimization (Enhanced)                              │
│  ├── CostTracker          - Track estimated costs                   │
│  └── Model enforcement    - Haiku-first policy                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Model Selection Policy

| Task Type | Model | Cost/MTok | Examples |
|-----------|-------|-----------|----------|
| File Search | haiku | $0.25 | Grep, Glob, find files |
| Text Copy | haiku | $0.25 | Extract, format text |
| Web Research | haiku | $0.25 | Fetch URLs, parse content |
| Code Review | haiku | $0.25 | Security scan, type check |
| Implementation | sonnet | $3.00 | Write/modify code |
| Architecture | sonnet | $3.00 | Design decisions |
| Critical Review | opus | $15.00 | Production trading code |
| Deep Analysis | opus | $15.00 | Major refactoring |

**Cost-Saving Rules:**
1. Default to Haiku for all Explore tasks
2. Use Sonnet only for code-writing tasks
3. Reserve Opus for architecture/critical decisions (max 1 per workflow)
4. Batch searches: spawn 3-8 Haiku agents vs 1 Sonnet

---

## Implementation Tasks

### Phase 1: Quick Haiku Agents (4 tasks)

- [ ] **1.1** Add `web_researcher` agent template
- [ ] **1.2** Add `text_extractor` agent template
- [ ] **1.3** Add `grep_agent` agent template
- [ ] **1.4** Add `file_lister` agent template

### Phase 2: RIC Integration (4 tasks)

- [ ] **2.1** Add `ric_research` workflow (P0)
- [ ] **2.2** Add `ric_verify` workflow (P3)
- [ ] **2.3** Create `/ric-agents` command
- [ ] **2.4** Update RIC hooks to suggest agents

### Phase 3: Auto-Persistence (3 tasks)

- [ ] **3.1** Create `ResearchPersister` class
- [ ] **3.2** Add save_web_research() method
- [ ] **3.3** Integrate with research document format

### Phase 4: Cost Tracking (3 tasks)

- [ ] **4.1** Create `CostTracker` class
- [ ] **4.2** Add cost estimates to AgentResult
- [ ] **4.3** Add cost summary to workflow output

### Phase 5: Documentation & Testing (3 tasks)

- [ ] **5.1** Update agent command documentation
- [ ] **5.2** Fix persona files (add Role/Expertise)
- [ ] **5.3** Add tests for new agents

---

## Detailed Specifications

### New Agent Templates

```python
# Add to AGENT_TEMPLATES in agent_orchestrator.py

"web_researcher": AgentSpec(
    name="WebResearcher",
    role="Search web and format findings for docs",
    model=Model.HAIKU,
    agent_type=AgentType.EXPLORE,
    prompt_template="""Research online: {query}

Search for current information (2024-2025).
Return findings formatted for docs/research/:

### Research Entry - [Current Date]
**Query**: {query}
**Sources**:
- [Title](URL) (Published: Date)
**Key Findings**:
- Finding 1
- Finding 2
**Applied**: How this applies to our project""",
    tags=["research", "web", "fast"],
),

"text_extractor": AgentSpec(
    name="TextExtractor",
    role="Extract and format text from files",
    model=Model.HAIKU,
    agent_type=AgentType.EXPLORE,
    prompt_template="""Extract from: {target}

Task: {task}

Return extracted text formatted for copying.
Include file path and line numbers if relevant.""",
    tags=["extract", "copy", "fast"],
),

"grep_agent": AgentSpec(
    name="GrepAgent",
    role="Fast pattern search across codebase",
    model=Model.HAIKU,
    agent_type=AgentType.EXPLORE,
    prompt_template="""Find all: {pattern}

Use Grep tool with output_mode="content".
Return matches as file:line references.
Max 50 results, sorted by relevance.""",
    tags=["search", "grep", "fast"],
),

"file_lister": AgentSpec(
    name="FileLister",
    role="List files matching patterns",
    model=Model.HAIKU,
    agent_type=AgentType.EXPLORE,
    prompt_template="""Find files: {pattern}

Use Glob tool.
Return sorted list grouped by directory.
Include file count per directory.""",
    tags=["files", "list", "fast"],
),

"research_saver": AgentSpec(
    name="ResearchSaver",
    role="Save research findings to documentation",
    model=Model.HAIKU,
    agent_type=AgentType.GENERAL,
    prompt_template="""Save research to docs/research/:

Findings:
{findings}

Format as markdown with:
- Timestamped entry header
- Source URLs with publication dates
- Key findings as bullets
- Applied section""",
    tags=["docs", "save", "fast"],
),
```

### RIC Integration Workflows

```python
# Add to WORKFLOW_TEMPLATES

"ric_research": WorkflowSpec(
    name="RIC Phase 0 Research",
    description="Parallel research for RIC P0 RESEARCH phase",
    pattern=WorkflowPattern.PARALLEL,
    agents=[
        AgentSpec(
            name="WebResearcher",
            role="Search web for best practices",
            model=Model.HAIKU,
            agent_type=AgentType.EXPLORE,
            prompt_template="Research: {topic} best practices 2025. Return formatted findings.",
        ),
        AgentSpec(
            name="CodebaseSearcher",
            role="Find existing implementations",
            model=Model.HAIKU,
            agent_type=AgentType.EXPLORE,
            prompt_template="Find existing code for: {topic}. Return file:line references.",
        ),
        AgentSpec(
            name="DocSearcher",
            role="Search project documentation",
            model=Model.HAIKU,
            agent_type=AgentType.EXPLORE,
            prompt_template="Find docs about: {topic} in docs/, CLAUDE.md, README files.",
        ),
    ],
    tags=["ric", "research", "p0"],
),

"ric_verify": WorkflowSpec(
    name="RIC Phase 3 Verify",
    description="Verification workflow for RIC P3 VERIFY phase",
    pattern=WorkflowPattern.PARALLEL,
    agents=[
        AGENT_TEMPLATES["test_analyzer"],
        AGENT_TEMPLATES["type_checker"],
        AGENT_TEMPLATES["security_scanner"],
    ],
    tags=["ric", "verify", "p3"],
),
```

### ResearchPersister Class

```python
class ResearchPersister:
    """Auto-save research findings to docs/research/."""

    RESEARCH_DIR = PROJECT_ROOT / "docs" / "research"
    DEFAULT_DOC = "UPGRADE-015-MCP-SERVER-RESEARCH.md"

    @classmethod
    def save_entry(cls, query: str, findings: str,
                   sources: List[str] = None,
                   doc_name: str = None) -> bool:
        """Save a research entry to documentation.

        Args:
            query: The search query used
            findings: Key findings to save
            sources: List of source URLs
            doc_name: Target document (default: MCP research doc)

        Returns:
            True if saved successfully
        """
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        doc_path = cls.RESEARCH_DIR / (doc_name or cls.DEFAULT_DOC)

        entry = f"""

---

### Research Entry - {timestamp}

**Search Queries**:
- "{query}"

"""
        if sources:
            entry += "**Key Sources**:\n"
            for s in sources:
                entry += f"- {s}\n"
            entry += "\n"

        entry += f"""**Key Findings**:
{findings}

**Search Date**: {timestamp}
"""

        try:
            with open(doc_path, "a") as f:
                f.write(entry)
            return True
        except Exception:
            return False
```

### Cost Tracker

```python
@dataclass
class CostEstimate:
    """Cost estimate for agent execution."""
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float

    @classmethod
    def estimate(cls, model: Model, prompt_len: int = 1000,
                 output_len: int = 2000) -> "CostEstimate":
        """Estimate cost based on model and token counts."""
        # Approximate costs per million tokens
        COSTS = {
            Model.HAIKU: {"input": 0.25, "output": 1.25},
            Model.SONNET: {"input": 3.0, "output": 15.0},
            Model.OPUS: {"input": 15.0, "output": 75.0},
        }

        rates = COSTS.get(model, COSTS[Model.SONNET])
        cost = (prompt_len * rates["input"] +
                output_len * rates["output"]) / 1_000_000

        return cls(
            model=model.value,
            input_tokens=prompt_len,
            output_tokens=output_len,
            cost_usd=cost,
        )
```

---

## RIC Phase Integration Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RIC Phase → Agent Recommendations                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  P0 RESEARCH ───────────────────────────────────────────────────────│
│  │                                                                   │
│  ├─► /agent-swarm {topic}     ← 8 haiku agents explore codebase     │
│  ├─► web_researcher (haiku)   ← Search web, format for docs         │
│  ├─► doc_finder (haiku)       ← Find existing documentation         │
│  └─► research_saver (haiku)   ← Save findings to docs/research/     │
│                                                                      │
│  P1 PLAN ───────────────────────────────────────────────────────────│
│  │                                                                   │
│  └─► architect (sonnet)       ← Design implementation plan          │
│                                                                      │
│  P2 BUILD ──────────────────────────────────────────────────────────│
│  │                                                                   │
│  ├─► implementer (sonnet)     ← Write code                          │
│  └─► refactorer (sonnet)      ← Improve code quality                │
│                                                                      │
│  P3 VERIFY ─────────────────────────────────────────────────────────│
│  │                                                                   │
│  ├─► /parallel-review         ← 4 haiku agents check code           │
│  ├─► test_analyzer (haiku)    ← Check test coverage                 │
│  ├─► type_checker (haiku)     ← Validate types                      │
│  └─► security_scanner (haiku) ← Security vulnerabilities            │
│                                                                      │
│  P4 REFLECT ────────────────────────────────────────────────────────│
│  │                                                                   │
│  ├─► /agent-consensus         ← 3 sonnet agents vote on quality     │
│  └─► deep_architect (opus)    ← Only if complex decisions needed    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Search task model | 100% Haiku | Audit agent logs |
| Research auto-saved | >80% entries | Count doc entries |
| Opus usage | <5% of tasks | Cost tracker |
| Agent success rate | >90% | State statistics |
| RIC integration | All 5 phases | Command coverage |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Haiku quality insufficient | Medium | Fall back to Sonnet for complex searches |
| Auto-save overwrites | High | Append-only mode, timestamped entries |
| Cost tracking inaccurate | Low | Use estimates, validate quarterly |
| RIC workflow coupling | Medium | Keep agents independent, workflows composable |

---

## Dependencies

- UPGRADE-017: Multi-Agent Orchestration (existing agent system)
- UPGRADE-016: RIC v5.0 (RIC phase system)
- `.claude/hooks/agent_orchestrator.py` (main implementation file)

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-05 | Initial document created | Claude |
