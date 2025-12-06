#!/usr/bin/env python3
"""
Generate Mermaid architecture diagrams from code analysis.

Produces:
- Agent hierarchy diagram
- Module dependency graph (layer architecture)
- Data flow diagram
- Hook system diagram

Usage:
    python scripts/generate_architecture_diagrams.py [--output-dir DIR]

Outputs diagrams to docs/architecture/generated/ by default.
"""

import ast
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def extract_agent_hierarchy() -> str:
    """
    Extract agent classes and generate hierarchy diagram.

    Returns:
        Mermaid diagram string showing agent class hierarchy.
    """
    agents_dir = Path("llm/agents")
    if not agents_dir.exists():
        return '```mermaid\nclassDiagram\n    note "No agents directory found"\n```'

    agents: list[dict[str, Any]] = []

    for py_file in agents_dir.glob("*.py"):
        if py_file.name.startswith("__"):
            continue

        try:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except (OSError, SyntaxError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = []
                for base in node.bases:
                    try:
                        bases.append(ast.unparse(base))
                    except Exception:
                        bases.append("?")

                # Only include classes that inherit from something with "Agent"
                if any("Agent" in b or "Base" in b for b in bases):
                    # Extract methods
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                            if not item.name.startswith("_"):
                                methods.append(item.name)

                    agents.append(
                        {
                            "name": node.name,
                            "bases": bases,
                            "file": py_file.stem,
                            "methods": methods[:5],  # Limit to 5 methods
                        }
                    )

    # Generate Mermaid class diagram
    lines = [
        "```mermaid",
        "classDiagram",
        "    direction TB",
        "",
        "    class TradingAgent {",
        "        <<abstract>>",
        "        +analyze()",
        "        +think()",
        "        +act()",
        "    }",
        "",
    ]

    # Add agent classes
    for agent in agents:
        lines.append(f"    class {agent['name']} {{")
        for method in agent["methods"]:
            lines.append(f"        +{method}()")
        lines.append("    }")
        lines.append("")

    # Add inheritance relationships
    for agent in agents:
        for base in agent["bases"]:
            # Clean up base name
            base_clean = base.split(".")[-1].replace("'", "").replace('"', "")
            if base_clean and base_clean != "object":
                lines.append(f"    {base_clean} <|-- {agent['name']}")

    lines.append("```")
    return "\n".join(lines)


def generate_module_dependencies() -> str:
    """
    Generate module dependency graph based on layer architecture.

    Returns:
        Mermaid flowchart showing module layers and dependencies.
    """
    # Layer definitions from REFACTOR_PLAN.md
    layers = {
        "Layer 4 - Applications": ["algorithms", "api", "ui"],
        "Layer 3 - Domain Logic": ["execution", "llm", "scanners", "indicators"],
        "Layer 2 - Core Models": ["models", "compliance", "evaluation"],
        "Layer 1 - Infrastructure": ["observability", "config"],
        "Layer 0 - Utilities": ["utils"],
    }

    lines = [
        "```mermaid",
        "flowchart TB",
        "    subgraph Legend",
        "        direction LR",
        "        L4[Layer 4: Apps]",
        "        L3[Layer 3: Domain]",
        "        L2[Layer 2: Core]",
        "        L1[Layer 1: Infra]",
        "        L0[Layer 0: Utils]",
        "    end",
        "",
    ]

    # Add layer subgraphs
    for layer_name, modules in layers.items():
        layer_id = layer_name.split(" - ")[0].replace(" ", "")
        layer_short = layer_name.split(" - ")[1] if " - " in layer_name else layer_name

        lines.append(f'    subgraph {layer_id}["{layer_short}"]')
        for mod in modules:
            if Path(mod).exists():
                lines.append(f"        {mod}[{mod}/]")
            else:
                lines.append(f"        {mod}[{mod}/ ❌]")
        lines.append("    end")
        lines.append("")

    # Add layer dependencies
    lines.extend(
        [
            "    Layer4 --> Layer3",
            "    Layer3 --> Layer2",
            "    Layer2 --> Layer1",
            "    Layer1 --> Layer0",
            "",
            "    %% Styling",
            "    classDef layer4 fill:#e1f5fe,stroke:#01579b",
            "    classDef layer3 fill:#f3e5f5,stroke:#4a148c",
            "    classDef layer2 fill:#fff3e0,stroke:#e65100",
            "    classDef layer1 fill:#e8f5e9,stroke:#1b5e20",
            "    classDef layer0 fill:#fafafa,stroke:#212121",
            "```",
        ]
    )

    return "\n".join(lines)


def generate_hook_system_diagram() -> str:
    """
    Generate diagram of Claude Code hook system.

    Returns:
        Mermaid flowchart showing hook triggers and handlers.
    """
    hooks_dir = Path(".claude/hooks")
    if not hooks_dir.exists():
        return "```mermaid\nflowchart LR\n    note[No hooks directory]\n```"

    # Categorize hooks
    hook_categories = {
        "core": [],
        "validation": [],
        "research": [],
        "trading": [],
        "formatting": [],
        "agents": [],
    }

    for category_dir in hooks_dir.iterdir():
        if category_dir.is_dir() and not category_dir.name.startswith("_"):
            category = category_dir.name
            if category in hook_categories:
                for hook_file in category_dir.glob("*.py"):
                    if not hook_file.name.startswith("__"):
                        hook_categories[category].append(hook_file.stem)

    lines = [
        "```mermaid",
        "flowchart LR",
        "    subgraph Triggers",
        "        PreToolUse[PreToolUse]",
        "        PostToolUse[PostToolUse]",
        "        SessionStart[SessionStart]",
        "        Stop[Stop]",
        "        PreCompact[PreCompact]",
        "    end",
        "",
        "    subgraph Hooks",
    ]

    # Add hook categories
    for category, hooks in hook_categories.items():
        if hooks:
            lines.append(f"        subgraph {category.title()}")
            for hook in hooks[:5]:  # Limit to 5 per category
                lines.append(f"            {hook}[{hook}]")
            if len(hooks) > 5:
                lines.append(f"            more_{category}[+{len(hooks) - 5} more]")
            lines.append("        end")

    lines.extend(
        [
            "    end",
            "",
            "    PreToolUse --> core",
            "    PostToolUse --> formatting",
            "    PostToolUse --> validation",
            "    SessionStart --> core",
            "    Stop --> core",
            "```",
        ]
    )

    return "\n".join(lines)


def generate_data_flow_diagram() -> str:
    """
    Generate data flow diagram for trading system.

    Returns:
        Mermaid flowchart showing data flow through system components.
    """
    lines = [
        "```mermaid",
        "flowchart TD",
        "    subgraph Input",
        "        Market[Market Data]",
        "        News[News Feed]",
        "        User[User Commands]",
        "    end",
        "",
        "    subgraph Analysis",
        "        Scanner[Movement Scanner]",
        "        Options[Options Scanner]",
        "        LLM[LLM Ensemble]",
        "        Technical[Technical Indicators]",
        "    end",
        "",
        "    subgraph Decision",
        "        Agents[Trading Agents]",
        "        Risk[Risk Manager]",
        "        CircuitBreaker[Circuit Breaker]",
        "    end",
        "",
        "    subgraph Execution",
        "        Orders[Order Manager]",
        "        Fills[Fill Tracker]",
        "        Positions[Position Manager]",
        "    end",
        "",
        "    subgraph Output",
        "        Broker[Broker API]",
        "        Dashboard[UI Dashboard]",
        "        Logs[Audit Logs]",
        "    end",
        "",
        "    Market --> Scanner",
        "    Market --> Options",
        "    Market --> Technical",
        "    News --> LLM",
        "    User --> Agents",
        "",
        "    Scanner --> Agents",
        "    Options --> Agents",
        "    LLM --> Agents",
        "    Technical --> Agents",
        "",
        "    Agents --> Risk",
        "    Risk --> CircuitBreaker",
        "    CircuitBreaker --> Orders",
        "",
        "    Orders --> Broker",
        "    Orders --> Fills",
        "    Fills --> Positions",
        "",
        "    Positions --> Dashboard",
        "    Orders --> Logs",
        "```",
    ]

    return "\n".join(lines)


def main() -> int:
    """
    Generate all architecture diagrams.

    Returns:
        Exit code (0 for success).
    """
    # Parse output directory
    output_dir = Path("docs/architecture/generated")
    if "--output-dir" in sys.argv:
        idx = sys.argv.index("--output-dir")
        if idx + 1 < len(sys.argv):
            output_dir = Path(sys.argv[idx + 1])

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Generate agent hierarchy
    agent_diagram = extract_agent_hierarchy()
    agent_content = f"""# Agent Hierarchy

*Auto-generated: {timestamp}*

This diagram shows the inheritance hierarchy of trading agents.

{agent_diagram}

## Notes

- Agents inherit from base classes in `llm/agents/base.py`
- Each agent specializes in a specific analysis type
- Methods shown are public interface methods
"""
    (output_dir / "agent_hierarchy.md").write_text(agent_content)
    print(f"Generated: {output_dir}/agent_hierarchy.md")

    # Generate module dependencies
    deps_diagram = generate_module_dependencies()
    deps_content = f"""# Module Dependencies

*Auto-generated: {timestamp}*

This diagram shows the layer architecture and module dependencies.

{deps_diagram}

## Layer Rules

- **Layer 4 (Applications)**: Can import from all lower layers
- **Layer 3 (Domain Logic)**: Can import from Layer 2, 1, 0
- **Layer 2 (Core Models)**: Can import from Layer 1, 0
- **Layer 1 (Infrastructure)**: Can import from Layer 0 only
- **Layer 0 (Utilities)**: No internal dependencies

Violations are detected by `scripts/check_layer_violations.py`.
"""
    (output_dir / "module_dependencies.md").write_text(deps_content)
    print(f"Generated: {output_dir}/module_dependencies.md")

    # Generate hook system diagram
    hook_diagram = generate_hook_system_diagram()
    hook_content = f"""# Claude Code Hook System

*Auto-generated: {timestamp}*

This diagram shows the Claude Code hook system architecture.

{hook_diagram}

## Hook Categories

| Category | Purpose |
|----------|---------|
| **core** | RIC Loop, file protection, session management |
| **validation** | Algorithm, research, and QA validation |
| **research** | Research tracking and documentation |
| **trading** | Trading risk and execution hooks |
| **formatting** | Code formatting and cross-references |
| **agents** | Agent orchestration hooks |

Configuration: `.claude/settings.json`
"""
    (output_dir / "hook_system.md").write_text(hook_content)
    print(f"Generated: {output_dir}/hook_system.md")

    # Generate data flow diagram
    flow_diagram = generate_data_flow_diagram()
    flow_content = f"""# Trading System Data Flow

*Auto-generated: {timestamp}*

This diagram shows how data flows through the trading system.

{flow_diagram}

## Components

### Input Layer
- **Market Data**: Real-time price, volume, options chains
- **News Feed**: Financial news for sentiment analysis
- **User Commands**: Manual trading overrides

### Analysis Layer
- **Scanners**: Detect trading opportunities
- **LLM Ensemble**: Sentiment analysis with multiple models
- **Technical**: RSI, MACD, Bollinger, etc.

### Decision Layer
- **Trading Agents**: Coordinate analysis and generate signals
- **Risk Manager**: Position sizing and exposure limits
- **Circuit Breaker**: Emergency halt on excessive losses

### Execution Layer
- **Order Manager**: Order lifecycle management
- **Fill Tracker**: Execution quality tracking

### Output Layer
- **Broker API**: Charles Schwab integration
- **Dashboard**: Real-time monitoring UI
- **Audit Logs**: Compliance and debugging
"""
    (output_dir / "data_flow.md").write_text(flow_content)
    print(f"Generated: {output_dir}/data_flow.md")

    # Generate index
    index_content = f"""# Generated Architecture Diagrams

*Auto-generated: {timestamp}*

This directory contains auto-generated architecture diagrams.
Regenerate with: `python scripts/generate_architecture_diagrams.py`

## Available Diagrams

| Diagram | Description |
|---------|-------------|
| [Agent Hierarchy](agent_hierarchy.md) | Trading agent class inheritance |
| [Module Dependencies](module_dependencies.md) | Layer architecture |
| [Hook System](hook_system.md) | Claude Code hooks |
| [Data Flow](data_flow.md) | Trading system data flow |

## Regeneration

These diagrams are generated from code analysis. To update:

```bash
python scripts/generate_architecture_diagrams.py
```

To use a custom output directory:

```bash
python scripts/generate_architecture_diagrams.py --output-dir path/to/dir
```
"""
    (output_dir / "README.md").write_text(index_content)
    print(f"Generated: {output_dir}/README.md")

    print(f"\nAll diagrams generated in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
