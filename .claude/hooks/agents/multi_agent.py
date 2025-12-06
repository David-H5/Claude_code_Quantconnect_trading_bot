#!/usr/bin/env python3
"""
Multi-Agent Orchestration System for Claude Code

This module provides utilities for spawning and coordinating multiple Claude agents
with different models (haiku/sonnet/opus) for parallel task execution.

Usage:
    # From Claude Code, use the Task tool directly with these patterns:

    # Quick parallel search (use haiku for speed)
    Task(subagent_type="Explore", model="haiku", prompt="Find X")
    Task(subagent_type="Explore", model="haiku", prompt="Find Y")

    # Complex analysis (use sonnet or opus)
    Task(subagent_type="Plan", model="sonnet", prompt="Design architecture for X")

This module provides:
1. Agent pattern definitions for common workflows
2. Prompt templates for specialized tasks
3. Result aggregation utilities
"""

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class AgentModel(Enum):
    """Available Claude models for agent tasks."""

    HAIKU = "haiku"  # Fast, cheap - use for searches, simple tasks
    SONNET = "sonnet"  # Balanced - use for most development tasks
    OPUS = "opus"  # Deep reasoning - use for architecture, complex analysis


class AgentType(Enum):
    """Built-in Claude Code agent types."""

    GENERAL = "general-purpose"
    EXPLORE = "Explore"
    PLAN = "Plan"
    CLAUDE_GUIDE = "claude-code-guide"


@dataclass
class AgentTask:
    """Definition of a task for an agent."""

    name: str
    description: str
    prompt: str
    agent_type: AgentType = AgentType.EXPLORE
    model: AgentModel = AgentModel.HAIKU
    priority: int = 1  # 1=highest
    depends_on: list[str] = field(default_factory=list)

    def to_task_params(self) -> dict[str, str]:
        """Convert to Task tool parameters."""
        return {
            "description": self.description,
            "prompt": self.prompt,
            "subagent_type": self.agent_type.value,
            "model": self.model.value,
        }


# =============================================================================
# Pre-defined Multi-Agent Patterns
# =============================================================================

PARALLEL_CODE_REVIEW = [
    AgentTask(
        name="security_scan",
        description="Security vulnerability scan",
        prompt="""Scan the following files for security vulnerabilities:
- SQL injection
- Command injection
- Hardcoded credentials
- Path traversal
- OWASP Top 10 issues

Report findings with file:line references and severity (CRITICAL/HIGH/MEDIUM/LOW).
{files}""",
        agent_type=AgentType.EXPLORE,
        model=AgentModel.HAIKU,
    ),
    AgentTask(
        name="type_check",
        description="Type hint validation",
        prompt="""Check for type hint completeness and correctness:
- Missing type hints on functions
- Incorrect type annotations
- Optional vs None handling
- Generic type issues

Report findings with file:line references.
{files}""",
        agent_type=AgentType.EXPLORE,
        model=AgentModel.HAIKU,
    ),
    AgentTask(
        name="test_coverage",
        description="Test coverage analysis",
        prompt="""Analyze test coverage for the following code:
- Identify untested functions
- Check for missing edge cases
- Evaluate test quality
- Suggest additional tests needed

{files}""",
        agent_type=AgentType.EXPLORE,
        model=AgentModel.HAIKU,
    ),
    AgentTask(
        name="architecture_review",
        description="Architecture review",
        prompt="""Review architecture and design patterns:
- SOLID principles adherence
- Design pattern usage
- Dependency management
- Coupling and cohesion

Provide specific recommendations.
{files}""",
        agent_type=AgentType.PLAN,
        model=AgentModel.SONNET,
    ),
]

PARALLEL_RESEARCH = [
    AgentTask(
        name="codebase_search",
        description="Search codebase for patterns",
        prompt="""Search the codebase for:
{query}

Find all relevant files, functions, and usage patterns.
Return file paths and key code snippets.""",
        agent_type=AgentType.EXPLORE,
        model=AgentModel.HAIKU,
    ),
    AgentTask(
        name="documentation_search",
        description="Search documentation",
        prompt="""Search project documentation for:
{query}

Check:
- docs/ directory
- README files
- Code comments and docstrings
- CLAUDE.md instructions

Return relevant documentation snippets.""",
        agent_type=AgentType.EXPLORE,
        model=AgentModel.HAIKU,
    ),
    AgentTask(
        name="test_search",
        description="Search test files",
        prompt="""Search test files for:
{query}

Find:
- Related test cases
- Test patterns used
- Fixtures and mocks
- Test coverage for the topic

Return test file paths and relevant test code.""",
        agent_type=AgentType.EXPLORE,
        model=AgentModel.HAIKU,
    ),
]

PARALLEL_TRADING_REVIEW = [
    AgentTask(
        name="risk_check",
        description="Trading risk analysis",
        prompt="""Review trading code for risk management:
- Circuit breaker integration
- Position limits enforcement
- Stop loss implementation
- Max drawdown protection
- Daily loss limits

Flag any code that bypasses safety checks.
{files}""",
        agent_type=AgentType.EXPLORE,
        model=AgentModel.SONNET,
    ),
    AgentTask(
        name="execution_check",
        description="Order execution review",
        prompt="""Review order execution logic:
- Order validation before submission
- Fill handling correctness
- Cancel/replace logic
- Slippage handling
- Duplicate order prevention

{files}""",
        agent_type=AgentType.EXPLORE,
        model=AgentModel.HAIKU,
    ),
    AgentTask(
        name="backtest_check",
        description="Look-ahead bias check",
        prompt="""Check for backtesting issues:
- Look-ahead bias (using future data)
- Survivorship bias
- Unrealistic fill assumptions
- Missing transaction costs
- Data snooping

{files}""",
        agent_type=AgentType.EXPLORE,
        model=AgentModel.HAIKU,
    ),
]


# =============================================================================
# Agent Result Aggregation
# =============================================================================


@dataclass
class AgentResult:
    """Result from an agent task."""

    task_name: str
    model_used: str
    result: str
    execution_time_ms: float
    success: bool
    error: str | None = None


def aggregate_results(results: list[AgentResult]) -> dict[str, Any]:
    """Aggregate results from multiple agents."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    return {
        "total_tasks": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "total_time_ms": sum(r.execution_time_ms for r in results),
        "results_by_task": {r.task_name: r.result for r in successful},
        "errors": {r.task_name: r.error for r in failed},
    }


# =============================================================================
# Prompt Templates for Specialized Agents
# =============================================================================

AGENT_PROMPTS = {
    "senior_engineer": """You are a senior software engineer reviewing code.
Focus on:
- Code quality and best practices
- Performance implications
- Maintainability concerns
- Error handling completeness
- Type safety

Be direct and technical. Reference specific line numbers.""",
    "risk_reviewer": """You are a trading risk specialist.
Your ONLY concern is trading safety:
- Position limits and sizing
- Circuit breaker integration
- Risk per trade limits
- Drawdown protection
- Order validation

Flag ANY code that could lead to financial loss.""",
    "qa_engineer": """You are a QA engineer focused on testing.
Evaluate:
- Test coverage completeness
- Edge case handling
- Integration test needs
- Mock/fixture quality
- Test reliability

Suggest specific test cases to add.""",
    "strategy_dev": """You are an options trading strategy developer.
Focus on:
- Greeks calculations correctness
- IV analysis accuracy
- Spread construction logic
- Entry/exit timing
- Position management

Reference QuantConnect patterns and best practices.""",
}


def get_persona_prompt(persona: str) -> str | None:
    """Get the system prompt for a persona."""
    return AGENT_PROMPTS.get(persona)


def load_persona_from_file(persona_name: str) -> str | None:
    """Load persona from .claude/agents/ directory."""
    agents_dir = Path(__file__).parent.parent / "agents"
    persona_file = agents_dir / f"{persona_name}.md"

    if persona_file.exists():
        return persona_file.read_text()
    return None


# =============================================================================
# CLI Interface
# =============================================================================


def print_usage():
    """Print usage information."""
    print("""
Multi-Agent Orchestration System
================================

This module provides patterns for parallel agent execution in Claude Code.

USAGE IN CLAUDE CODE:

1. Parallel Code Review (4 agents):
   Use Task tool 4 times in ONE message with different focuses:
   - Task(model="haiku", prompt="Security scan...")
   - Task(model="haiku", prompt="Type check...")
   - Task(model="haiku", prompt="Test coverage...")
   - Task(model="sonnet", prompt="Architecture review...")

2. Parallel Research (3 agents):
   - Task(model="haiku", prompt="Search codebase for X")
   - Task(model="haiku", prompt="Search docs for X")
   - Task(model="haiku", prompt="Search tests for X")

3. Trading Code Review (3 agents):
   - Task(model="sonnet", prompt="Risk management check")
   - Task(model="haiku", prompt="Execution logic check")
   - Task(model="haiku", prompt="Look-ahead bias check")

MODEL SELECTION GUIDE:

| Model  | Speed   | Cost   | Use For                          |
|--------|---------|--------|----------------------------------|
| haiku  | Fastest | Lowest | File search, grep, simple checks |
| sonnet | Medium  | Medium | Code review, implementation      |
| opus   | Slowest | Highest| Architecture, deep analysis      |

AVAILABLE PATTERNS:

- PARALLEL_CODE_REVIEW: 4-agent comprehensive code review
- PARALLEL_RESEARCH: 3-agent codebase research
- PARALLEL_TRADING_REVIEW: 3-agent trading safety review

Run with --patterns to see pattern details.
Run with --personas to see available personas.
""")


def print_patterns():
    """Print available multi-agent patterns."""
    patterns = {
        "PARALLEL_CODE_REVIEW": PARALLEL_CODE_REVIEW,
        "PARALLEL_RESEARCH": PARALLEL_RESEARCH,
        "PARALLEL_TRADING_REVIEW": PARALLEL_TRADING_REVIEW,
    }

    print("\nAvailable Multi-Agent Patterns:")
    print("=" * 60)

    for name, tasks in patterns.items():
        print(f"\n{name}:")
        print("-" * 40)
        for task in tasks:
            print(f"  • {task.name}: {task.description}")
            print(f"    Model: {task.model.value}, Type: {task.agent_type.value}")


def print_personas():
    """Print available agent personas."""
    print("\nBuilt-in Personas:")
    print("=" * 40)
    for name in AGENT_PROMPTS:
        print(f"  • {name}")

    agents_dir = Path(__file__).parent.parent / "agents"
    if agents_dir.exists():
        print("\nFile-based Personas (.claude/agents/):")
        print("=" * 40)
        for f in agents_dir.glob("*.md"):
            print(f"  • {f.stem}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--patterns":
            print_patterns()
        elif sys.argv[1] == "--personas":
            print_personas()
        elif sys.argv[1] == "--help":
            print_usage()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print_usage()
    else:
        print_usage()
