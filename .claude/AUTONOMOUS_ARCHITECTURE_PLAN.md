# Autonomous Development Architecture Plan

**Created:** 2025-12-05
**Status:** Ready for Review
**Estimated Effort:** 6-10 weeks
**Prerequisites:** REFACTOR_PLAN.md Complete ✅, NEXT_REFACTOR_PLAN.md Phases 1-5 Complete ✅ (Phase 6 Deferred)

---

## Research Sources

This plan is based on industry best practices from:

- [Claude Code: Best practices for agentic coding](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Autonomous Code Documentation: Enterprise Solutions](https://www.augmentcode.com/guides/autonomous-code-documentation)
- [Enabling Claude Code to work more autonomously](https://www.anthropic.com/news/enabling-claude-code-to-work-more-autonomously)
- [Agentic Coding: 6 Best Practices](https://aiagent.marktechpost.com/post/agentic-coding-6-best-practices-you-need-to-know)
- [Architectural Design Patterns for HFT Bots](https://medium.com/@halljames9963/architectural-design-patterns-for-high-frequency-algo-trading-bots-c84f5083d704)
- [From Stale Docs to Living Architecture: Automating ADRs](https://medium.com/@iraj.hedayati/from-stale-docs-to-living-architecture-automating-adrs-with-github-llm-e80bb066b4b6)
- [Google Cloud: Design Patterns for Agentic AI Systems](https://cloud.google.com/architecture/choose-design-pattern-agentic-ai-system)
- [Architecture Decision Records](https://adr.github.io/)
- [Claude Code Hooks Mastery](https://github.com/disler/claude-code-hooks-mastery)

---

## Executive Summary

### Current State

| Metric | Current | Target |
|--------|---------|--------|
| Test Coverage | 34% | 70%+ |
| Docstring Coverage | 60-70% | 95%+ |
| TODO/FIXME Count | 11,554 | <100 |
| Auto-Generated Docs | None | 80% coverage |
| ADR Count | 7 | 15+ |
| Agent Explainability | Minimal | Full audit trail |

### Vision

Transform the codebase into a **self-documenting, self-improving autonomous development system** where:

1. **Code documents itself** - Docstrings auto-generated, ADRs auto-created from PRs
2. **System monitors itself** - Agent performance tracked, anomalies detected, alerts escalated
3. **Architecture evolves** - Decisions recorded, patterns extracted, improvements suggested
4. **Quality enforced** - Tests auto-generated, coverage tracked, regressions blocked

---

## Architecture Overview

### Self-Documenting Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS DOCUMENTATION LAYER                        │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │  Auto-Docstring │  │  ADR Generator  │  │  Architecture  │            │
│  │    Generator    │  │  (from PRs)     │  │  Diagrammer    │            │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘            │
│           │                   │                   │                     │
│           └───────────────────┼───────────────────┘                     │
│                               │                                         │
│                    ┌──────────▼──────────┐                             │
│                    │   Living CLAUDE.md   │                             │
│                    │   (Auto-Updated)     │                             │
│                    └──────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                    SELF-IMPROVEMENT LAYER                               │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │ Agent Evaluator │  │  Prompt        │  │  Test          │            │
│  │ (Performance)   │  │  Optimizer     │  │  Generator     │            │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘            │
│           │                   │                   │                     │
│           └───────────────────┼───────────────────┘                     │
│                               │                                         │
│                    ┌──────────▼──────────┐                             │
│                    │   Feedback Loop      │                             │
│                    │   (Weekly Cycle)     │                             │
│                    └──────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                    OBSERVABILITY LAYER                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │ Decision Audit  │  │  Anomaly       │  │  Performance   │            │
│  │ Trail           │  │  Detector      │  │  Leaderboard   │            │
│  └────────────────┘  └────────────────┘  └────────────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Self-Documenting Code Infrastructure

**Goal:** Achieve 95% docstring coverage with auto-generation

**Duration:** 1-2 weeks

### 1.1 Auto-Docstring Generator Hook

Create hook that generates docstrings for functions missing them:

**File:** `.claude/hooks/formatting/auto_docstring.py`

```python
#!/usr/bin/env python3
"""
Auto-Docstring Generator Hook

Runs PostToolUse on Edit|Write operations.
Uses Claude to generate Google-style docstrings for functions without them.
"""
import ast
import json
import sys
from pathlib import Path

def find_undocumented_functions(filepath: str) -> list[dict]:
    """Find functions without docstrings."""
    with open(filepath) as f:
        tree = ast.parse(f.read())

    undocumented = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not ast.get_docstring(node):
                undocumented.append({
                    "name": node.name,
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "returns": ast.unparse(node.returns) if node.returns else None
                })
    return undocumented

def main():
    input_data = json.load(sys.stdin)
    tool_input = input_data.get("tool_input", {})
    filepath = tool_input.get("file_path", "")

    if not filepath.endswith(".py"):
        sys.exit(0)

    undocumented = find_undocumented_functions(filepath)

    if undocumented:
        # Output suggestion for Claude to add docstrings
        print(json.dumps({
            "result": "continue",
            "message": f"[Auto-Doc] {len(undocumented)} functions need docstrings in {Path(filepath).name}",
            "data": {"undocumented": undocumented[:5]}  # Limit to 5
        }))

    sys.exit(0)

if __name__ == "__main__":
    main()
```

### 1.2 Docstring Validation Pre-Commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
  - repo: https://github.com/pycqa/pydocstyle
    rev: 6.3.0
    hooks:
      - id: pydocstyle
        args: ["--convention=google", "--add-ignore=D100,D104"]
        files: ^(llm|execution|models|observability)/.*\.py$
        exclude: ^tests/
```

### 1.3 Docstring Coverage Tracking

**File:** `scripts/docstring_coverage.py`

```python
#!/usr/bin/env python3
"""Track docstring coverage and report trends."""
import ast
import json
from pathlib import Path
from datetime import datetime

def calculate_coverage(directory: str) -> dict:
    """Calculate docstring coverage for a directory."""
    total_functions = 0
    documented_functions = 0

    for py_file in Path(directory).rglob("*.py"):
        if "test" in str(py_file) or "__pycache__" in str(py_file):
            continue

        with open(py_file) as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_functions += 1
                if ast.get_docstring(node):
                    documented_functions += 1

    coverage = (documented_functions / total_functions * 100) if total_functions else 0

    return {
        "total": total_functions,
        "documented": documented_functions,
        "coverage": round(coverage, 2),
        "timestamp": datetime.now().isoformat()
    }

def main():
    modules = ["llm", "execution", "models", "observability", "evaluation"]
    results = {mod: calculate_coverage(mod) for mod in modules}

    # Save to state
    state_file = Path(".claude/state/docstring_coverage.json")
    state_file.parent.mkdir(exist_ok=True)

    history = []
    if state_file.exists():
        history = json.loads(state_file.read_text())

    history.append({"date": datetime.now().isoformat(), "results": results})
    history = history[-30:]  # Keep 30 days

    state_file.write_text(json.dumps(history, indent=2))

    # Print summary
    total_coverage = sum(r["coverage"] for r in results.values()) / len(results)
    print(f"Overall docstring coverage: {total_coverage:.1f}%")
    for mod, data in results.items():
        print(f"  {mod}: {data['coverage']:.1f}% ({data['documented']}/{data['total']})")

if __name__ == "__main__":
    main()
```

### 1.4 Commit Phase 1

```bash
git add -A
git commit -m "feat(docs): Add auto-docstring generation infrastructure

- Created auto_docstring.py hook for PostToolUse
- Added pydocstyle to pre-commit hooks
- Created docstring_coverage.py tracking script
- Target: 95% docstring coverage

Implements Phase 1 of AUTONOMOUS_ARCHITECTURE_PLAN.md

$(cat <<'EOF'
Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Phase 2: Living Documentation System

**Goal:** Auto-generate and maintain architecture documentation

**Duration:** 1-2 weeks

### 2.1 Architecture Diagram Generator

**File:** `scripts/generate_architecture_diagrams.py`

```python
#!/usr/bin/env python3
"""
Generate Mermaid architecture diagrams from code analysis.

Produces:
- Agent hierarchy diagram
- Module dependency graph
- Data flow diagram
- Decision audit flow
"""
import ast
import json
from pathlib import Path
from typing import Dict, List, Set

def extract_agent_hierarchy() -> str:
    """Extract agent classes and generate hierarchy diagram."""
    agents_dir = Path("llm/agents")
    agents = []

    for py_file in agents_dir.glob("*.py"):
        with open(py_file) as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [ast.unparse(base) for base in node.bases]
                if any("Agent" in b for b in bases):
                    agents.append({
                        "name": node.name,
                        "bases": bases,
                        "file": py_file.stem
                    })

    # Generate Mermaid
    mermaid = ["```mermaid", "classDiagram"]
    mermaid.append("    class TradingAgent {")
    mermaid.append("        +analyze()")
    mermaid.append("        +think()")
    mermaid.append("        +act()")
    mermaid.append("    }")

    for agent in agents:
        mermaid.append(f"    TradingAgent <|-- {agent['name']}")

    mermaid.append("```")
    return "\n".join(mermaid)

def generate_module_dependencies() -> str:
    """Generate module dependency graph."""
    layers = {
        "Layer 4": ["algorithms", "api", "ui"],
        "Layer 3": ["execution", "llm", "scanners"],
        "Layer 2": ["models", "compliance"],
        "Layer 1": ["observability", "config"],
        "Layer 0": ["utils"]
    }

    mermaid = ["```mermaid", "flowchart TB"]
    mermaid.append("    subgraph Layer4[\"Layer 4: Applications\"]")
    for mod in layers["Layer 4"]:
        mermaid.append(f"        {mod}")
    mermaid.append("    end")

    mermaid.append("    subgraph Layer3[\"Layer 3: Domain Logic\"]")
    for mod in layers["Layer 3"]:
        mermaid.append(f"        {mod}")
    mermaid.append("    end")

    mermaid.append("    subgraph Layer2[\"Layer 2: Core Models\"]")
    for mod in layers["Layer 2"]:
        mermaid.append(f"        {mod}")
    mermaid.append("    end")

    mermaid.append("    subgraph Layer1[\"Layer 1: Infrastructure\"]")
    for mod in layers["Layer 1"]:
        mermaid.append(f"        {mod}")
    mermaid.append("    end")

    mermaid.append("    subgraph Layer0[\"Layer 0: Utilities\"]")
    for mod in layers["Layer 0"]:
        mermaid.append(f"        {mod}")
    mermaid.append("    end")

    # Add dependencies
    mermaid.append("    Layer4 --> Layer3")
    mermaid.append("    Layer3 --> Layer2")
    mermaid.append("    Layer2 --> Layer1")
    mermaid.append("    Layer1 --> Layer0")

    mermaid.append("```")
    return "\n".join(mermaid)

def main():
    output_dir = Path("docs/architecture/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate agent hierarchy
    agent_diagram = extract_agent_hierarchy()
    (output_dir / "agent_hierarchy.md").write_text(
        f"# Agent Hierarchy\n\n{agent_diagram}"
    )

    # Generate module dependencies
    deps_diagram = generate_module_dependencies()
    (output_dir / "module_dependencies.md").write_text(
        f"# Module Dependencies\n\n{deps_diagram}"
    )

    print(f"Generated diagrams in {output_dir}")

if __name__ == "__main__":
    main()
```

### 2.2 Auto-ADR Generator

**File:** `scripts/generate_adr_from_pr.py`

```python
#!/usr/bin/env python3
"""
Auto-generate Architecture Decision Records from PR diffs.

Based on: https://medium.com/@iraj.hedayati/from-stale-docs-to-living-architecture-automating-adrs-with-github-llm-e80bb066b4b6
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

ADR_TEMPLATE = """# ADR-{number}: {title}

**Date:** {date}
**Status:** {status}
**Deciders:** {deciders}

## Context

{context}

## Decision

{decision}

## Consequences

### Positive
{positive_consequences}

### Negative
{negative_consequences}

## Related

- PR: {pr_link}
- Files Changed: {files_changed}
"""

def get_next_adr_number() -> int:
    """Get the next ADR number."""
    adr_dir = Path("docs/adr")
    existing = list(adr_dir.glob("ADR-*.md"))
    if not existing:
        return 1

    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split("-")[1].split("_")[0])
            numbers.append(num)
        except (IndexError, ValueError):
            continue

    return max(numbers) + 1 if numbers else 1

def generate_adr_prompt(diff: str, pr_title: str, pr_description: str) -> str:
    """Generate prompt for LLM to create ADR."""
    return f"""Based on this PR, generate an Architecture Decision Record.

PR Title: {pr_title}
PR Description: {pr_description}

Diff (truncated):
{diff[:5000]}

Generate an ADR with:
1. Clear title summarizing the architectural decision
2. Context explaining why this change was needed
3. The decision made and how it addresses the context
4. Positive and negative consequences

Format as JSON with keys: title, context, decision, positive_consequences, negative_consequences
"""

def create_adr(
    title: str,
    context: str,
    decision: str,
    positive: str,
    negative: str,
    pr_link: str = "",
    files_changed: int = 0
) -> Path:
    """Create a new ADR file."""
    number = get_next_adr_number()
    filename = f"ADR-{number:04d}_{title.lower().replace(' ', '_')[:30]}.md"

    content = ADR_TEMPLATE.format(
        number=f"{number:04d}",
        title=title,
        date=datetime.now().strftime("%Y-%m-%d"),
        status="Proposed",
        deciders="Claude Code (auto-generated)",
        context=context,
        decision=decision,
        positive_consequences=positive,
        negative_consequences=negative,
        pr_link=pr_link or "N/A",
        files_changed=files_changed
    )

    adr_path = Path("docs/adr") / filename
    adr_path.write_text(content)

    return adr_path

def main():
    # Example: Generate ADR from last commit
    diff = subprocess.check_output(
        ["git", "diff", "HEAD~1..HEAD"],
        text=True
    )

    log = subprocess.check_output(
        ["git", "log", "-1", "--format=%s%n%n%b"],
        text=True
    )

    lines = log.strip().split("\n")
    pr_title = lines[0] if lines else "Untitled"
    pr_description = "\n".join(lines[1:]) if len(lines) > 1 else ""

    # For now, create a stub - in production, call LLM
    print(f"Would generate ADR for: {pr_title}")
    print(f"Diff size: {len(diff)} chars")
    print(f"Description: {pr_description[:200]}...")

if __name__ == "__main__":
    main()
```

### 2.3 Living CLAUDE.md Generator

**File:** `scripts/update_claude_md.py`

```python
#!/usr/bin/env python3
"""
Auto-update CLAUDE.md with current project metrics.

Updates sections:
- Module statistics (count, LOC)
- Test coverage
- Agent registry
- Command summary
- Recent changes
"""
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime

def get_module_stats() -> str:
    """Get module statistics."""
    modules = ["llm", "execution", "models", "observability", "evaluation", "algorithms"]
    stats = []

    for mod in modules:
        mod_path = Path(mod)
        if not mod_path.exists():
            continue

        py_files = list(mod_path.rglob("*.py"))
        loc = sum(len(f.read_text().splitlines()) for f in py_files if f.is_file())

        stats.append(f"| {mod}/ | {len(py_files)} | {loc:,} |")

    header = "| Module | Files | LOC |\n|--------|-------|-----|\n"
    return header + "\n".join(stats)

def get_test_coverage() -> str:
    """Get test coverage from pytest-cov."""
    try:
        result = subprocess.run(
            ["pytest", "--cov=.", "--cov-report=json", "-q", "--no-header"],
            capture_output=True,
            text=True,
            timeout=120
        )

        cov_json = Path("coverage.json")
        if cov_json.exists():
            data = json.loads(cov_json.read_text())
            total = data.get("totals", {}).get("percent_covered", 0)
            return f"**Test Coverage:** {total:.1f}%"
    except Exception:
        pass

    return "**Test Coverage:** Run `pytest --cov` to calculate"

def get_agent_count() -> str:
    """Count registered agents."""
    agents_dir = Path("llm/agents")
    count = len(list(agents_dir.glob("*.py"))) - 2  # Exclude __init__ and base
    return f"**Agent Count:** {count} specialized agents"

def get_command_count() -> str:
    """Count Claude Code commands."""
    commands_dir = Path(".claude/commands")
    count = len(list(commands_dir.glob("*.md")))
    return f"**Claude Commands:** {count} slash commands available"

def get_recent_changes() -> str:
    """Get recent git commits."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            capture_output=True,
            text=True
        )
        commits = result.stdout.strip().split("\n")
        return "**Recent Changes:**\n" + "\n".join(f"- `{c}`" for c in commits)
    except Exception:
        return "**Recent Changes:** Git not available"

def update_claude_md():
    """Update the auto-generated section of CLAUDE.md."""
    claude_md = Path("CLAUDE.md")
    content = claude_md.read_text()

    # Find or create auto-generated section
    marker_start = "<!-- AUTO-GENERATED-START -->"
    marker_end = "<!-- AUTO-GENERATED-END -->"

    auto_content = f"""
{marker_start}

## Auto-Generated Project Metrics

*Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}*

### Module Statistics

{get_module_stats()}

### Quality Metrics

{get_test_coverage()}
{get_agent_count()}
{get_command_count()}

{get_recent_changes()}

{marker_end}
"""

    if marker_start in content:
        # Replace existing
        pattern = f"{re.escape(marker_start)}.*?{re.escape(marker_end)}"
        content = re.sub(pattern, auto_content.strip(), content, flags=re.DOTALL)
    else:
        # Append
        content += "\n\n" + auto_content

    claude_md.write_text(content)
    print("Updated CLAUDE.md with current metrics")

if __name__ == "__main__":
    update_claude_md()
```

### 2.4 Commit Phase 2

```bash
git add -A
git commit -m "feat(docs): Add living documentation system

- Created generate_architecture_diagrams.py for Mermaid diagrams
- Created generate_adr_from_pr.py for auto-ADR generation
- Created update_claude_md.py for living metrics
- Architecture diagrams auto-generated from code

Implements Phase 2 of AUTONOMOUS_ARCHITECTURE_PLAN.md

$(cat <<'EOF'
Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Phase 3: Agent Decision Explainability

**Goal:** Full audit trail for agent decisions

**Duration:** 1-2 weeks

### 3.1 Explanation Report Type

**File:** `llm/agents/explanation.py`

```python
"""Agent decision explainability framework."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

class ConfidenceLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class SignalContribution:
    """A signal that contributed to the decision."""
    source: str  # e.g., "technical_analyst", "sentiment", "news"
    signal_type: str  # e.g., "bullish", "bearish", "neutral"
    weight: float  # 0.0 to 1.0
    reasoning: str
    confidence: ConfidenceLevel
    data_points: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class AlternativeConsidered:
    """An alternative that was considered but not chosen."""
    action: str
    expected_outcome: str
    reason_rejected: str
    confidence_if_chosen: float

@dataclass
class RiskAssessment:
    """Risk assessment for the decision."""
    max_loss_estimate: float
    probability_of_loss: float
    risk_reward_ratio: float
    position_size_justification: str
    stop_loss_reasoning: str

@dataclass
class ExplanationReport:
    """
    Comprehensive explanation of an agent decision.

    Provides full transparency for audit, debugging, and regulatory compliance.
    """
    # Metadata
    agent_name: str
    decision_id: str
    timestamp: datetime

    # The decision
    action: str  # "buy", "sell", "hold", "close"
    symbol: str
    quantity: Optional[int] = None
    price_target: Optional[float] = None

    # Reasoning
    reasoning_chain: List[str] = field(default_factory=list)
    contributing_signals: List[SignalContribution] = field(default_factory=list)

    # Confidence
    overall_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)

    # Alternatives
    alternatives_considered: List[AlternativeConsidered] = field(default_factory=list)

    # Risk
    risk_assessment: Optional[RiskAssessment] = None

    # Context
    market_context: Dict[str, Any] = field(default_factory=dict)
    position_context: Dict[str, Any] = field(default_factory=dict)

    def to_human_readable(self) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"## Decision Explanation: {self.action.upper()} {self.symbol}",
            f"**Agent:** {self.agent_name}",
            f"**Time:** {self.timestamp.isoformat()}",
            f"**Confidence:** {self.overall_confidence.value}",
            "",
            "### Reasoning Chain",
        ]

        for i, step in enumerate(self.reasoning_chain, 1):
            lines.append(f"{i}. {step}")

        lines.extend([
            "",
            "### Contributing Signals",
        ])

        for signal in self.contributing_signals:
            lines.append(
                f"- **{signal.source}** ({signal.signal_type}): "
                f"{signal.reasoning} [weight: {signal.weight:.2f}]"
            )

        if self.alternatives_considered:
            lines.extend([
                "",
                "### Alternatives Considered",
            ])
            for alt in self.alternatives_considered:
                lines.append(f"- **{alt.action}**: {alt.reason_rejected}")

        if self.risk_assessment:
            lines.extend([
                "",
                "### Risk Assessment",
                f"- Max Loss Estimate: ${self.risk_assessment.max_loss_estimate:.2f}",
                f"- Probability of Loss: {self.risk_assessment.probability_of_loss:.1%}",
                f"- Risk/Reward: {self.risk_assessment.risk_reward_ratio:.2f}",
            ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_name": self.agent_name,
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "price_target": self.price_target,
            "reasoning_chain": self.reasoning_chain,
            "contributing_signals": [
                {
                    "source": s.source,
                    "signal_type": s.signal_type,
                    "weight": s.weight,
                    "reasoning": s.reasoning,
                    "confidence": s.confidence.value
                }
                for s in self.contributing_signals
            ],
            "overall_confidence": self.overall_confidence.value,
            "confidence_breakdown": self.confidence_breakdown,
            "alternatives_considered": [
                {
                    "action": a.action,
                    "expected_outcome": a.expected_outcome,
                    "reason_rejected": a.reason_rejected
                }
                for a in self.alternatives_considered
            ],
            "risk_assessment": {
                "max_loss_estimate": self.risk_assessment.max_loss_estimate,
                "probability_of_loss": self.risk_assessment.probability_of_loss,
                "risk_reward_ratio": self.risk_assessment.risk_reward_ratio
            } if self.risk_assessment else None
        }
```

### 3.2 Decision Audit API

**File:** `api/decision_audit.py`

```python
"""REST API for decision audit trail."""
from fastapi import APIRouter, Query
from typing import List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

router = APIRouter(prefix="/audit", tags=["Decision Audit"])

DECISION_LOG_DIR = Path(".claude/state/decisions")

@router.get("/decisions")
async def get_decisions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=1000),
    agent: Optional[str] = None,
    symbol: Optional[str] = None,
    action: Optional[str] = None,
    since: Optional[datetime] = None
) -> List[dict]:
    """
    Query decision audit trail.

    Args:
        skip: Number of records to skip
        limit: Maximum records to return
        agent: Filter by agent name
        symbol: Filter by symbol
        action: Filter by action type
        since: Only return decisions after this datetime

    Returns:
        List of decision records with explanations
    """
    decisions = []

    for log_file in sorted(DECISION_LOG_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(log_file.read_text())

            # Apply filters
            if agent and data.get("agent_name") != agent:
                continue
            if symbol and data.get("symbol") != symbol:
                continue
            if action and data.get("action") != action:
                continue
            if since:
                decision_time = datetime.fromisoformat(data.get("timestamp", ""))
                if decision_time < since:
                    continue

            decisions.append(data)

            if len(decisions) >= skip + limit:
                break
        except (json.JSONDecodeError, KeyError):
            continue

    return decisions[skip:skip + limit]

@router.get("/decisions/{decision_id}")
async def get_decision(decision_id: str) -> dict:
    """Get a specific decision by ID."""
    log_file = DECISION_LOG_DIR / f"{decision_id}.json"

    if not log_file.exists():
        return {"error": "Decision not found"}

    return json.loads(log_file.read_text())

@router.get("/agents/performance")
async def get_agent_performance(
    days: int = Query(30, ge=1, le=365)
) -> dict:
    """
    Get agent performance metrics over time.

    Returns accuracy, latency, and decision counts by agent.
    """
    since = datetime.now() - timedelta(days=days)

    performance = {}

    for log_file in DECISION_LOG_DIR.glob("*.json"):
        try:
            data = json.loads(log_file.read_text())
            decision_time = datetime.fromisoformat(data.get("timestamp", ""))

            if decision_time < since:
                continue

            agent = data.get("agent_name", "unknown")
            if agent not in performance:
                performance[agent] = {
                    "decision_count": 0,
                    "actions": {},
                    "avg_confidence": 0,
                    "confidence_sum": 0
                }

            performance[agent]["decision_count"] += 1

            action = data.get("action", "unknown")
            performance[agent]["actions"][action] = \
                performance[agent]["actions"].get(action, 0) + 1

            # Track confidence
            conf_map = {"very_low": 0.1, "low": 0.3, "medium": 0.5, "high": 0.7, "very_high": 0.9}
            conf = conf_map.get(data.get("overall_confidence", "medium"), 0.5)
            performance[agent]["confidence_sum"] += conf
        except Exception:
            continue

    # Calculate averages
    for agent, stats in performance.items():
        if stats["decision_count"] > 0:
            stats["avg_confidence"] = stats["confidence_sum"] / stats["decision_count"]
        del stats["confidence_sum"]

    return performance
```

### 3.3 Commit Phase 3

```bash
git add -A
git commit -m "feat(agents): Add decision explainability framework

- Created ExplanationReport with full audit trail
- Added SignalContribution and RiskAssessment types
- Created decision_audit.py REST API
- Supports regulatory compliance and debugging

Implements Phase 3 of AUTONOMOUS_ARCHITECTURE_PLAN.md

$(cat <<'EOF'
Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Phase 4: Self-Improvement Feedback Loop

**Goal:** Automated agent improvement from performance data

**Duration:** 1-2 weeks

### 4.1 Weekly Agent Evaluation Pipeline

**File:** `scripts/weekly_agent_evaluation.py`

```python
#!/usr/bin/env python3
"""
Weekly Agent Evaluation Pipeline

Runs every Sunday to:
1. Evaluate all agents over past week
2. Identify top/bottom performers
3. Extract weakness categories
4. Generate prompt improvements
5. Store results for A/B testing
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class AgentEvaluation:
    agent_name: str
    decision_count: int
    accuracy: float  # Correct predictions / total
    avg_confidence: float
    weaknesses: List[str]
    strengths: List[str]
    improvement_suggestions: List[str]

def evaluate_agent(agent_name: str, decisions: List[dict]) -> AgentEvaluation:
    """Evaluate a single agent's performance."""
    if not decisions:
        return AgentEvaluation(
            agent_name=agent_name,
            decision_count=0,
            accuracy=0.0,
            avg_confidence=0.0,
            weaknesses=["No decisions to evaluate"],
            strengths=[],
            improvement_suggestions=[]
        )

    # Calculate metrics
    conf_map = {"very_low": 0.1, "low": 0.3, "medium": 0.5, "high": 0.7, "very_high": 0.9}
    confidences = [conf_map.get(d.get("overall_confidence", "medium"), 0.5) for d in decisions]
    avg_conf = sum(confidences) / len(confidences)

    # Identify patterns (simplified - would use actual outcome data)
    action_counts = {}
    for d in decisions:
        action = d.get("action", "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1

    # Determine weaknesses/strengths
    weaknesses = []
    strengths = []

    if avg_conf < 0.4:
        weaknesses.append("Low confidence in decisions")
    elif avg_conf > 0.7:
        strengths.append("High confidence in decisions")

    if len(action_counts) == 1:
        weaknesses.append(f"Only makes {list(action_counts.keys())[0]} decisions - lacks diversity")

    # Generate improvement suggestions
    suggestions = []
    if "Low confidence" in str(weaknesses):
        suggestions.append("Add more data sources to increase signal strength")
    if "lacks diversity" in str(weaknesses):
        suggestions.append("Review decision thresholds - may be too conservative")

    return AgentEvaluation(
        agent_name=agent_name,
        decision_count=len(decisions),
        accuracy=0.5,  # Would calculate from actual outcomes
        avg_confidence=avg_conf,
        weaknesses=weaknesses,
        strengths=strengths,
        improvement_suggestions=suggestions
    )

def run_weekly_evaluation() -> Dict[str, AgentEvaluation]:
    """Run evaluation for all agents."""
    decision_dir = Path(".claude/state/decisions")
    since = datetime.now() - timedelta(days=7)

    # Group decisions by agent
    agent_decisions: Dict[str, List[dict]] = {}

    for log_file in decision_dir.glob("*.json"):
        try:
            data = json.loads(log_file.read_text())
            decision_time = datetime.fromisoformat(data.get("timestamp", ""))

            if decision_time < since:
                continue

            agent = data.get("agent_name", "unknown")
            if agent not in agent_decisions:
                agent_decisions[agent] = []
            agent_decisions[agent].append(data)
        except Exception:
            continue

    # Evaluate each agent
    evaluations = {}
    for agent, decisions in agent_decisions.items():
        evaluations[agent] = evaluate_agent(agent, decisions)

    # Save results
    output_dir = Path(".claude/state/evaluations")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"weekly_{datetime.now().strftime('%Y%m%d')}.json"
    output_file.write_text(json.dumps({
        agent: {
            "decision_count": e.decision_count,
            "accuracy": e.accuracy,
            "avg_confidence": e.avg_confidence,
            "weaknesses": e.weaknesses,
            "strengths": e.strengths,
            "improvement_suggestions": e.improvement_suggestions
        }
        for agent, e in evaluations.items()
    }, indent=2))

    return evaluations

def main():
    print("Running weekly agent evaluation...")
    evaluations = run_weekly_evaluation()

    print(f"\nEvaluated {len(evaluations)} agents:")
    for agent, eval in evaluations.items():
        print(f"\n{agent}:")
        print(f"  Decisions: {eval.decision_count}")
        print(f"  Avg Confidence: {eval.avg_confidence:.2f}")
        if eval.weaknesses:
            print(f"  Weaknesses: {', '.join(eval.weaknesses)}")
        if eval.improvement_suggestions:
            print(f"  Suggestions: {', '.join(eval.improvement_suggestions)}")

if __name__ == "__main__":
    main()
```

### 4.2 Prompt Version Management

**File:** `llm/prompt_versions/manager.py`

```python
"""Prompt version management for A/B testing."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import hashlib

PROMPT_DIR = Path("llm/prompt_versions")

def get_prompt_version(agent_name: str, variant: str = "default") -> Dict:
    """Get a specific prompt version for an agent."""
    prompt_file = PROMPT_DIR / agent_name / f"{variant}.json"

    if not prompt_file.exists():
        return {"error": f"Prompt version not found: {agent_name}/{variant}"}

    return json.loads(prompt_file.read_text())

def save_prompt_version(
    agent_name: str,
    prompt_content: str,
    metadata: Optional[Dict] = None
) -> str:
    """Save a new prompt version."""
    agent_dir = PROMPT_DIR / agent_name
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Generate version hash
    version_hash = hashlib.md5(prompt_content.encode()).hexdigest()[:8]

    prompt_data = {
        "version": version_hash,
        "created": datetime.now().isoformat(),
        "content": prompt_content,
        "metadata": metadata or {}
    }

    # Save as new version
    version_file = agent_dir / f"v_{version_hash}.json"
    version_file.write_text(json.dumps(prompt_data, indent=2))

    # Update default pointer
    default_file = agent_dir / "default.json"
    default_file.write_text(json.dumps({"current": version_hash}))

    return version_hash

def get_prompt_history(agent_name: str) -> list:
    """Get version history for an agent's prompts."""
    agent_dir = PROMPT_DIR / agent_name

    if not agent_dir.exists():
        return []

    versions = []
    for version_file in agent_dir.glob("v_*.json"):
        data = json.loads(version_file.read_text())
        versions.append({
            "version": data["version"],
            "created": data["created"],
            "metadata": data.get("metadata", {})
        })

    return sorted(versions, key=lambda x: x["created"], reverse=True)
```

### 4.3 Commit Phase 4

```bash
git add -A
git commit -m "feat(agents): Add self-improvement feedback loop

- Created weekly_agent_evaluation.py pipeline
- Added prompt version management for A/B testing
- Tracks weaknesses and generates improvement suggestions
- Supports continuous agent optimization

Implements Phase 4 of AUTONOMOUS_ARCHITECTURE_PLAN.md

$(cat <<'EOF'
Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Phase 5: Technical Debt Management

**Goal:** Systematic TODO/FIXME tracking and resolution

**Duration:** 1 week

### 5.1 TODO Scanner and Tracker

**File:** `scripts/scan_todos.py`

```python
#!/usr/bin/env python3
"""
Scan codebase for TODO/FIXME items and generate tracking report.

Categories:
- P0: Critical (blocks functionality)
- P1: Important (should fix soon)
- P2: Polish (nice to have)
"""
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

TODO_PATTERN = re.compile(
    r'#\s*(TODO|FIXME|XXX|HACK|BUG)[\s:]*(.+)',
    re.IGNORECASE
)

PRIORITY_KEYWORDS = {
    "P0": ["critical", "blocking", "urgent", "security", "crash"],
    "P1": ["important", "should", "need", "bug", "fix"],
    "P2": ["polish", "refactor", "cleanup", "nice", "maybe"]
}

def classify_priority(text: str) -> str:
    """Classify TODO priority based on keywords."""
    text_lower = text.lower()

    for priority, keywords in PRIORITY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return priority

    return "P2"  # Default to lowest priority

def scan_file(filepath: Path) -> List[Dict]:
    """Scan a file for TODOs."""
    todos = []

    try:
        content = filepath.read_text()
    except Exception:
        return []

    for i, line in enumerate(content.splitlines(), 1):
        match = TODO_PATTERN.search(line)
        if match:
            todo_type = match.group(1).upper()
            text = match.group(2).strip()

            todos.append({
                "file": str(filepath),
                "line": i,
                "type": todo_type,
                "text": text,
                "priority": classify_priority(text)
            })

    return todos

def scan_codebase(root: Path = Path(".")) -> Dict[str, List[Dict]]:
    """Scan entire codebase for TODOs."""
    todos = {"P0": [], "P1": [], "P2": []}

    for py_file in root.rglob("*.py"):
        if any(part.startswith(".") for part in py_file.parts):
            continue
        if "venv" in str(py_file) or "__pycache__" in str(py_file):
            continue

        file_todos = scan_file(py_file)
        for todo in file_todos:
            todos[todo["priority"]].append(todo)

    return todos

def generate_report(todos: Dict[str, List[Dict]]) -> str:
    """Generate markdown report."""
    report = [
        "# Technical Debt Report",
        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        "## Summary\n",
        f"| Priority | Count |",
        f"|----------|-------|",
    ]

    total = 0
    for priority in ["P0", "P1", "P2"]:
        count = len(todos[priority])
        total += count
        report.append(f"| {priority} | {count} |")

    report.append(f"| **Total** | **{total}** |")

    for priority in ["P0", "P1", "P2"]:
        if todos[priority]:
            report.extend([
                f"\n## {priority} Items\n",
            ])

            for todo in todos[priority][:20]:  # Limit to 20 per priority
                report.append(
                    f"- [ ] `{todo['file']}:{todo['line']}` - "
                    f"**{todo['type']}**: {todo['text']}"
                )

            if len(todos[priority]) > 20:
                report.append(f"\n*... and {len(todos[priority]) - 20} more*")

    return "\n".join(report)

def main():
    todos = scan_codebase()

    # Save to state
    state_file = Path(".claude/state/todos.json")
    state_file.parent.mkdir(exist_ok=True)
    state_file.write_text(json.dumps(todos, indent=2))

    # Generate report
    report = generate_report(todos)
    report_file = Path("docs/TECHNICAL_DEBT.md")
    report_file.write_text(report)

    # Print summary
    total = sum(len(v) for v in todos.values())
    print(f"Found {total} TODOs:")
    for priority in ["P0", "P1", "P2"]:
        print(f"  {priority}: {len(todos[priority])}")

    print(f"\nReport saved to {report_file}")

if __name__ == "__main__":
    main()
```

### 5.2 Pre-Commit TODO Gate

Add to `.pre-commit-config.yaml`:

```yaml
  - repo: local
    hooks:
      - id: todo-scanner
        name: Scan TODOs
        entry: python scripts/scan_todos.py
        language: python
        pass_filenames: false
        stages: [pre-commit]
```

### 5.3 Commit Phase 5

```bash
git add -A
git commit -m "feat(quality): Add technical debt management

- Created scan_todos.py for TODO tracking
- Auto-generates docs/TECHNICAL_DEBT.md
- Classifies by priority (P0/P1/P2)
- Added pre-commit hook for tracking

Implements Phase 5 of AUTONOMOUS_ARCHITECTURE_PLAN.md

$(cat <<'EOF'
Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Phase 6: Observability Enhancement

**Goal:** Comprehensive system health monitoring

**Duration:** 1-2 weeks

### 6.1 Agent Performance Leaderboard

**File:** `observability/agent_leaderboard.py`

```python
"""Real-time agent performance leaderboard."""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class AgentRanking:
    agent_name: str
    rank: int
    score: float  # Composite score 0-100
    accuracy: float
    avg_confidence: float
    decision_count: int
    trend: str  # "up", "down", "stable"
    rank_change: int

def calculate_composite_score(
    accuracy: float,
    confidence: float,
    decision_count: int
) -> float:
    """Calculate composite performance score."""
    # Weighted components
    accuracy_weight = 0.5
    confidence_weight = 0.3
    activity_weight = 0.2

    # Normalize decision count (assume 100 decisions/week is max)
    activity_score = min(decision_count / 100, 1.0)

    score = (
        accuracy * accuracy_weight +
        confidence * confidence_weight +
        activity_score * activity_weight
    ) * 100

    return round(score, 2)

def get_leaderboard(days: int = 30) -> List[AgentRanking]:
    """Generate agent performance leaderboard."""
    decision_dir = Path(".claude/state/decisions")
    since = datetime.now() - timedelta(days=days)

    # Collect metrics by agent
    agent_metrics: Dict[str, Dict] = {}

    for log_file in decision_dir.glob("*.json"):
        try:
            data = json.loads(log_file.read_text())
            decision_time = datetime.fromisoformat(data.get("timestamp", ""))

            if decision_time < since:
                continue

            agent = data.get("agent_name", "unknown")
            if agent not in agent_metrics:
                agent_metrics[agent] = {
                    "decisions": 0,
                    "confidence_sum": 0
                }

            agent_metrics[agent]["decisions"] += 1

            conf_map = {"very_low": 0.1, "low": 0.3, "medium": 0.5, "high": 0.7, "very_high": 0.9}
            conf = conf_map.get(data.get("overall_confidence", "medium"), 0.5)
            agent_metrics[agent]["confidence_sum"] += conf
        except Exception:
            continue

    # Calculate scores and rank
    rankings = []
    for agent, metrics in agent_metrics.items():
        avg_conf = metrics["confidence_sum"] / max(metrics["decisions"], 1)
        score = calculate_composite_score(
            accuracy=0.6,  # Would come from actual outcomes
            confidence=avg_conf,
            decision_count=metrics["decisions"]
        )
        rankings.append({
            "agent": agent,
            "score": score,
            "confidence": avg_conf,
            "decisions": metrics["decisions"]
        })

    # Sort by score
    rankings.sort(key=lambda x: x["score"], reverse=True)

    # Load previous rankings for trend
    prev_file = Path(".claude/state/leaderboard_prev.json")
    prev_rankings = {}
    if prev_file.exists():
        prev_data = json.loads(prev_file.read_text())
        prev_rankings = {r["agent"]: r["rank"] for r in prev_data}

    # Build ranking objects
    result = []
    for i, r in enumerate(rankings, 1):
        prev_rank = prev_rankings.get(r["agent"], i)
        rank_change = prev_rank - i

        trend = "stable"
        if rank_change > 0:
            trend = "up"
        elif rank_change < 0:
            trend = "down"

        result.append(AgentRanking(
            agent_name=r["agent"],
            rank=i,
            score=r["score"],
            accuracy=0.6,  # Placeholder
            avg_confidence=r["confidence"],
            decision_count=r["decisions"],
            trend=trend,
            rank_change=abs(rank_change)
        ))

    # Save current rankings
    save_data = [{"agent": r.agent_name, "rank": r.rank} for r in result]
    prev_file.write_text(json.dumps(save_data))

    return result

def print_leaderboard():
    """Print formatted leaderboard."""
    rankings = get_leaderboard()

    print("\n🏆 Agent Performance Leaderboard\n")
    print(f"{'Rank':<6}{'Agent':<25}{'Score':<10}{'Decisions':<12}{'Trend':<10}")
    print("-" * 63)

    for r in rankings:
        trend_icon = "📈" if r.trend == "up" else "📉" if r.trend == "down" else "➡️"
        print(f"{r.rank:<6}{r.agent_name:<25}{r.score:<10.1f}{r.decision_count:<12}{trend_icon} {r.trend}")

if __name__ == "__main__":
    print_leaderboard()
```

### 6.2 Anomaly Detection System

**File:** `observability/anomaly_detector.py`

```python
"""Anomaly detection for agent behavior."""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Anomaly:
    metric_name: str
    current_value: float
    baseline_value: float
    deviation: float  # Number of std devs from baseline
    severity: AlertSeverity
    agent: Optional[str]
    timestamp: datetime
    message: str

THRESHOLDS = {
    "confidence_drop": {
        "warning": 0.15,  # 15% drop
        "error": 0.25,    # 25% drop
        "critical": 0.40  # 40% drop
    },
    "decision_rate_change": {
        "warning": 0.30,  # 30% change
        "error": 0.50,    # 50% change
        "critical": 0.70  # 70% change
    }
}

def calculate_baseline(values: List[float]) -> Tuple[float, float]:
    """Calculate baseline mean and std dev."""
    if len(values) < 3:
        return 0.5, 0.1  # Default baseline

    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.1

    return mean, max(stdev, 0.01)  # Prevent zero stdev

def detect_confidence_anomaly(
    agent: str,
    current: float,
    history: List[float]
) -> Optional[Anomaly]:
    """Detect confidence anomalies."""
    baseline, stdev = calculate_baseline(history)

    deviation = (baseline - current) / stdev if stdev > 0 else 0
    drop_pct = (baseline - current) / baseline if baseline > 0 else 0

    if drop_pct < THRESHOLDS["confidence_drop"]["warning"]:
        return None

    severity = AlertSeverity.WARNING
    if drop_pct >= THRESHOLDS["confidence_drop"]["critical"]:
        severity = AlertSeverity.CRITICAL
    elif drop_pct >= THRESHOLDS["confidence_drop"]["error"]:
        severity = AlertSeverity.ERROR

    return Anomaly(
        metric_name="confidence",
        current_value=current,
        baseline_value=baseline,
        deviation=deviation,
        severity=severity,
        agent=agent,
        timestamp=datetime.now(),
        message=f"Agent {agent} confidence dropped {drop_pct:.1%} from baseline"
    )

def run_anomaly_detection() -> List[Anomaly]:
    """Run anomaly detection on all agents."""
    anomalies = []

    decision_dir = Path(".claude/state/decisions")
    if not decision_dir.exists():
        return anomalies

    # Group decisions by agent
    agent_history: Dict[str, List[float]] = {}
    agent_recent: Dict[str, List[float]] = {}

    cutoff_recent = datetime.now() - timedelta(hours=24)
    cutoff_history = datetime.now() - timedelta(days=7)

    for log_file in decision_dir.glob("*.json"):
        try:
            data = json.loads(log_file.read_text())
            decision_time = datetime.fromisoformat(data.get("timestamp", ""))
            agent = data.get("agent_name", "unknown")

            conf_map = {"very_low": 0.1, "low": 0.3, "medium": 0.5, "high": 0.7, "very_high": 0.9}
            conf = conf_map.get(data.get("overall_confidence", "medium"), 0.5)

            if decision_time >= cutoff_recent:
                if agent not in agent_recent:
                    agent_recent[agent] = []
                agent_recent[agent].append(conf)
            elif decision_time >= cutoff_history:
                if agent not in agent_history:
                    agent_history[agent] = []
                agent_history[agent].append(conf)
        except Exception:
            continue

    # Detect anomalies
    for agent, recent in agent_recent.items():
        history = agent_history.get(agent, [])
        if history:
            current_avg = statistics.mean(recent)
            anomaly = detect_confidence_anomaly(agent, current_avg, history)
            if anomaly:
                anomalies.append(anomaly)

    return anomalies

def main():
    anomalies = run_anomaly_detection()

    if not anomalies:
        print("✅ No anomalies detected")
        return

    print(f"⚠️ Detected {len(anomalies)} anomalies:\n")

    for a in anomalies:
        icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}
        print(f"{icon.get(a.severity.value, '❓')} [{a.severity.value.upper()}] {a.message}")
        print(f"   Current: {a.current_value:.2f} | Baseline: {a.baseline_value:.2f}")
        print()

if __name__ == "__main__":
    main()
```

### 6.3 Commit Phase 6

```bash
git add -A
git commit -m "feat(observability): Add comprehensive monitoring

- Created agent_leaderboard.py with rankings
- Added anomaly_detector.py for behavior monitoring
- Tracks confidence drops and decision rate changes
- Alerts on significant deviations

Implements Phase 6 of AUTONOMOUS_ARCHITECTURE_PLAN.md

$(cat <<'EOF'
Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Summary: Implementation Roadmap

### Phase Overview

| Phase | Focus | Duration | Key Deliverables |
|-------|-------|----------|------------------|
| 1 | Self-Documenting Code | 1-2 weeks | Auto-docstring hook, coverage tracking |
| 2 | Living Documentation | 1-2 weeks | ADR generator, architecture diagrams, living CLAUDE.md |
| 3 | Decision Explainability | 1-2 weeks | ExplanationReport, audit API |
| 4 | Self-Improvement | 1-2 weeks | Weekly evaluation, prompt versioning |
| 5 | Tech Debt Management | 1 week | TODO scanner, TECHNICAL_DEBT.md |
| 6 | Observability | 1-2 weeks | Leaderboard, anomaly detection |

### Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Docstring Coverage | 60-70% | 95%+ | `scripts/docstring_coverage.py` |
| TODO Count | 11,554 | <100 | `scripts/scan_todos.py` |
| ADR Count | 7 | 15+ | Count files in `docs/adr/` |
| Decision Explainability | Minimal | 100% | All decisions have ExplanationReport |
| Anomaly Detection | None | Active | Alerts on deviations |

### Quick Start

```bash
# Execute phase by phase:
"Read .claude/AUTONOMOUS_ARCHITECTURE_PLAN.md and execute Phase 1"
```

---

## Appendix: New Files Created

```
.claude/hooks/formatting/auto_docstring.py
scripts/docstring_coverage.py
scripts/generate_architecture_diagrams.py
scripts/generate_adr_from_pr.py
scripts/update_claude_md.py
scripts/weekly_agent_evaluation.py
scripts/scan_todos.py
llm/agents/explanation.py
llm/prompt_versions/manager.py
api/decision_audit.py
observability/agent_leaderboard.py
observability/anomaly_detector.py
docs/architecture/generated/ (auto-generated)
docs/TECHNICAL_DEBT.md (auto-generated)
```

---

## Notes

1. **Execute phases sequentially** - Each builds on previous infrastructure
2. **Commit after each phase** - Easy rollback if issues arise
3. **Run tests after each phase** - Ensure no regressions
4. **Monitor metrics** - Track improvement over time
5. **Iterate** - Adjust thresholds and parameters based on real-world usage
