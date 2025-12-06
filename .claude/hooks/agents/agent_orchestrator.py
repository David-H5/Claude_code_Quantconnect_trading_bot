#!/usr/bin/env python3
"""
Agent Orchestrator - Intelligent Multi-Agent Coordination Suite

A comprehensive system for spawning, managing, and coordinating multiple Claude agents
with intelligent model selection, workflow patterns, and autonomous capabilities.

Features:
- Intelligent model selection based on task complexity
- Pre-built workflow patterns (parallel, sequential, pipeline, consensus)
- Agent templates for common development tasks
- State tracking and result aggregation
- Simple CLI interface for human and autonomous invocation

Usage:
    # From command line
    python agent_orchestrator.py run <workflow> [args]
    python agent_orchestrator.py auto "<task description>"
    python agent_orchestrator.py status
    python agent_orchestrator.py help

    # From Claude Code
    Use /agents command with subcommands
"""

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
AGENT_STATE_FILE = PROJECT_ROOT / ".claude" / "state" / "agent_state.json"
AGENT_CONFIG_FILE = PROJECT_ROOT / ".claude" / "config" / "agents.json"
AGENT_LOGS_DIR = PROJECT_ROOT / ".claude" / "agent_logs"


class Model(Enum):
    """Available Claude models with characteristics."""

    HAIKU = "haiku"  # Fast, cheap - searches, simple checks
    SONNET = "sonnet"  # Balanced - implementation, review
    OPUS = "opus"  # Deep - architecture, complex reasoning


class AgentType(Enum):
    """Built-in Claude Code agent types."""

    EXPLORE = "Explore"
    PLAN = "Plan"
    GENERAL = "general-purpose"
    GUIDE = "claude-code-guide"


class WorkflowPattern(Enum):
    """Coordination patterns for multi-agent workflows."""

    PARALLEL = "parallel"  # All agents run simultaneously
    SEQUENTIAL = "sequential"  # Agents run one after another
    PIPELINE = "pipeline"  # Output of one feeds into next
    CONSENSUS = "consensus"  # Multiple agents vote on decision
    HIERARCHICAL = "hierarchical"  # Manager agent coordinates workers
    SWARM = "swarm"  # Self-organizing agents


class TaskComplexity(Enum):
    """Task complexity levels for model selection."""

    TRIVIAL = 1  # Simple search, grep
    SIMPLE = 2  # Single-file analysis
    MODERATE = 3  # Multi-file changes
    COMPLEX = 4  # Architecture decisions
    CRITICAL = 5  # High-stakes decisions


# =============================================================================
# Retry & Fallback Configuration (UPGRADE-017-MEDIUM Phase 2)
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for agent retry behavior."""

    max_retries: int = 3
    initial_delay_ms: int = 1000
    max_delay_ms: int = 30000
    exponential_base: float = 2.0
    jitter_pct: float = 0.25  # ±25% randomization
    retryable_errors: list[str] = field(
        default_factory=lambda: [
            "timeout",
            "rate_limit",
            "server_error",
            "connection_error",
            "context_length",
            "overloaded",
        ]
    )


def calculate_backoff(attempt: int, config: RetryConfig) -> int:
    """Calculate delay with exponential backoff and jitter.

    Args:
        attempt: Current attempt number (0-based)
        config: RetryConfig with backoff parameters

    Returns:
        Delay in milliseconds
    """
    import random

    delay = min(config.initial_delay_ms * (config.exponential_base**attempt), config.max_delay_ms)
    jitter = delay * config.jitter_pct * (2 * random.random() - 1)
    return int(delay + jitter)


class RetryableAgent:
    """Wrapper for agents with retry capability."""

    def __init__(self, agent: "AgentSpec", config: RetryConfig = None):
        self.agent = agent
        self.config = config or RetryConfig()
        self.attempts = 0
        self.last_error: str | None = None

    def should_retry(self, error: str) -> bool:
        """Check if the error is retryable and attempts remain."""
        return self.attempts < self.config.max_retries and any(e in error.lower() for e in self.config.retryable_errors)

    def get_retry_prompt(self) -> str:
        """Get prompt with retry context if applicable."""
        if self.attempts == 0:
            return self.agent.prompt_template
        return f"""RETRY ATTEMPT {self.attempts + 1}/{self.config.max_retries}
Previous error: {self.last_error}

{self.agent.prompt_template}"""

    def record_attempt(self, error: str = None):
        """Record an attempt and optional error."""
        self.attempts += 1
        if error:
            self.last_error = error

    def reset(self):
        """Reset retry state."""
        self.attempts = 0
        self.last_error = None


class FallbackRouter:
    """Route to fallback agents when primary agent fails."""

    def __init__(self, config_path: Path = None):
        self.config_path = config_path or AGENT_CONFIG_FILE
        self.fallback_chains: dict[str, list[str]] = {}
        self._load_config()

    def _load_config(self):
        """Load fallback configuration from file."""
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text())
                self.fallback_chains = data.get("fallbacks", {})
            except (json.JSONDecodeError, KeyError):
                self.fallback_chains = self._default_fallbacks()
        else:
            self.fallback_chains = self._default_fallbacks()

    def _default_fallbacks(self) -> dict[str, list[str]]:
        """Default fallback mappings."""
        return {
            "security_scanner": ["type_checker", "code_finder"],
            "architect": ["implementer"],
            "risk_reviewer": ["execution_reviewer", "backtest_reviewer"],
            "deep_architect": ["architect", "implementer"],
            "web_researcher": ["doc_finder", "code_finder"],
            "critical_reviewer": ["architect", "security_scanner"],
        }

    def get_fallback(self, failed_agent: str, attempt: int) -> str | None:
        """Get next fallback agent for a failed agent.

        Args:
            failed_agent: Name of the agent that failed
            attempt: Fallback attempt number (0-based)

        Returns:
            Name of fallback agent or None if exhausted
        """
        chain = self.fallback_chains.get(failed_agent, [])
        if attempt < len(chain):
            return chain[attempt]
        return None

    def has_fallback(self, agent: str) -> bool:
        """Check if agent has fallback options."""
        return agent in self.fallback_chains and len(self.fallback_chains[agent]) > 0

    def get_all_fallbacks(self, agent: str) -> list[str]:
        """Get all fallback options for an agent."""
        return self.fallback_chains.get(agent, [])


# Generic fallback agents for when specific fallbacks fail
FALLBACK_AGENTS: dict[str, "AgentSpec"] = {}  # Populated after AgentSpec definition


# =============================================================================
# Circuit Breaker & Graceful Degradation (UPGRADE-017-MEDIUM Phase 3)
# =============================================================================


class AgentCircuitBreaker:
    """Prevent cascading failures by opening circuit after repeated failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit open, requests blocked (fail fast)
    - HALF-OPEN: Testing if system recovered (allow one request)
    """

    def __init__(self, failure_threshold: int = 3, reset_timeout_s: int = 300):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            reset_timeout_s: Seconds before attempting half-open state
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.failures: dict[str, int] = {}
        self.open_circuits: dict[str, datetime] = {}
        self._load_state()

    def _state_file(self) -> Path:
        """Get state file path."""
        return PROJECT_ROOT / ".claude" / "state" / "circuit_breaker.json"

    def _load_state(self):
        """Load circuit breaker state from file."""
        state_file = self._state_file()
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
                self.failures = state.get("failures", {})
                for agent, ts in state.get("open_circuits", {}).items():
                    if ts:
                        self.open_circuits[agent] = datetime.fromisoformat(ts)
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_state(self):
        """Save circuit breaker state to file."""
        state_file = self._state_file()
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "failures": self.failures,
            "open_circuits": {agent: ts.isoformat() if ts else None for agent, ts in self.open_circuits.items()},
            "updated_at": datetime.now().isoformat(),
        }
        state_file.write_text(json.dumps(state, indent=2))

    def record_failure(self, agent_name: str):
        """Record a failure for an agent."""
        self.failures[agent_name] = self.failures.get(agent_name, 0) + 1
        if self.failures[agent_name] >= self.failure_threshold:
            self.open_circuits[agent_name] = datetime.now()
        self._save_state()

    def record_success(self, agent_name: str):
        """Record success, resetting failure count."""
        self.failures[agent_name] = 0
        if agent_name in self.open_circuits:
            del self.open_circuits[agent_name]
        self._save_state()

    def is_open(self, agent_name: str) -> bool:
        """Check if circuit is open for an agent."""
        if agent_name not in self.open_circuits:
            return False

        elapsed = (datetime.now() - self.open_circuits[agent_name]).total_seconds()
        if elapsed >= self.reset_timeout_s:
            # Transition to half-open: allow one attempt
            del self.open_circuits[agent_name]
            self._save_state()
            return False
        return True

    def can_execute(self, agent_name: str) -> bool:
        """Check if agent can be executed (circuit not open)."""
        return not self.is_open(agent_name)

    def get_status(self) -> dict[str, str]:
        """Get circuit breaker status for all agents."""
        status = {}
        all_agents = set(list(self.failures.keys()) + list(self.open_circuits.keys()))

        for agent in all_agents:
            if self.is_open(agent):
                remaining = self.reset_timeout_s - (datetime.now() - self.open_circuits[agent]).total_seconds()
                status[agent] = f"OPEN ({int(remaining)}s remaining)"
            elif self.failures.get(agent, 0) > 0:
                status[agent] = f"DEGRADED ({self.failures[agent]}/{self.failure_threshold})"
            else:
                status[agent] = "CLOSED"
        return status

    def reset(self, agent_name: str = None):
        """Reset circuit breaker state.

        Args:
            agent_name: Specific agent to reset, or None for all
        """
        if agent_name:
            self.failures.pop(agent_name, None)
            self.open_circuits.pop(agent_name, None)
        else:
            self.failures.clear()
            self.open_circuits.clear()
        self._save_state()


@dataclass
class PartialResult:
    """Result that may be partial due to some agent failures."""

    workflow_name: str
    total_agents: int
    completed_agents: int
    failed_agents: int
    results: dict[str, "AgentResult"]
    failures: dict[str, str]  # agent_name -> error message
    degradation_level: str  # "full", "partial", "minimal"
    usable: bool  # Whether results are actionable

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_agents == 0:
            return 0.0
        return self.completed_agents / self.total_agents


def aggregate_partial_results(workflow_name: str, results: list["AgentResult"]) -> PartialResult:
    """Aggregate results even with partial failures.

    Args:
        workflow_name: Name of the workflow
        results: List of AgentResult objects

    Returns:
        PartialResult with aggregated data
    """
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    # Determine degradation level based on success rate
    success_rate = len(successful) / len(results) if results else 0
    if success_rate >= 0.75:
        level = "full"
    elif success_rate >= 0.5:
        level = "partial"
    else:
        level = "minimal"

    return PartialResult(
        workflow_name=workflow_name,
        total_agents=len(results),
        completed_agents=len(successful),
        failed_agents=len(failed),
        results={r.agent_name: r for r in successful},
        failures={r.agent_name: r.error or "Unknown error" for r in failed},
        degradation_level=level,
        usable=success_rate >= 0.25,  # At least 25% success = usable
    )


def format_partial_result(result: PartialResult) -> str:
    """Format partial result for display.

    Args:
        result: PartialResult to format

    Returns:
        Formatted string for display
    """
    status_icon = {"full": "✅", "partial": "⚠️", "minimal": "❌"}.get(result.degradation_level, "?")

    output = f"""## Results ({result.degradation_level.title()} - {result.completed_agents}/{result.total_agents} agents)
{status_icon} Degradation Level: {result.degradation_level}
"""

    if result.results:
        output += f"\n✅ Completed: {', '.join(result.results.keys())}\n"

    if result.failures:
        output += f"\n❌ Failed: {', '.join(f'{k} ({v})' for k, v in result.failures.items())}\n"

    if not result.usable:
        output += "\n⚠️ WARNING: Results may not be usable (success rate < 25%)\n"

    return output


def select_agent_with_circuit_breaker(
    requested_agent: str, circuit_breaker: AgentCircuitBreaker, fallback_router: FallbackRouter
) -> str | None:
    """Select agent, falling back if circuit is open.

    Args:
        requested_agent: Name of requested agent
        circuit_breaker: CircuitBreaker instance
        fallback_router: FallbackRouter instance

    Returns:
        Agent name to use, or None if all options exhausted
    """
    if circuit_breaker.can_execute(requested_agent):
        return requested_agent

    # Circuit is open, try fallbacks
    for attempt in range(3):
        fallback = fallback_router.get_fallback(requested_agent, attempt)
        if fallback and circuit_breaker.can_execute(fallback):
            return fallback

    # All fallbacks exhausted or open
    return None


def get_timeout(agent: "AgentSpec", config: dict[str, Any] = None) -> int:
    """Select appropriate timeout based on agent type and model.

    Args:
        agent: AgentSpec to get timeout for
        config: Optional config dict (loaded from file if not provided)

    Returns:
        Timeout in milliseconds
    """
    if config is None:
        if AGENT_CONFIG_FILE.exists():
            try:
                config = json.loads(AGENT_CONFIG_FILE.read_text())
            except json.JSONDecodeError:
                config = {}
        else:
            config = {}

    timeouts = config.get("timeouts", {})

    # Check model-specific timeout
    model_timeout = timeouts.get("by_model", {}).get(agent.model.value)

    # Check type-specific timeout (use first tag)
    type_timeout = None
    if agent.tags:
        type_timeout = timeouts.get("by_type", {}).get(agent.tags[0])

    # Select minimum of available timeouts
    candidates = [t for t in [model_timeout, type_timeout] if t]
    if candidates:
        return min(candidates)

    return timeouts.get("default_ms", agent.timeout_ms)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AgentSpec:
    """Specification for a single agent."""

    name: str
    role: str
    model: Model = Model.HAIKU
    agent_type: AgentType = AgentType.EXPLORE
    prompt_template: str = ""
    priority: int = 1
    timeout_ms: int = 120000
    depends_on: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_task_call(self, context: dict[str, Any] = None) -> dict[str, str]:
        """Generate Task tool parameters."""
        prompt = self.prompt_template
        if context:
            for key, value in context.items():
                prompt = prompt.replace(f"{{{key}}}", str(value))

        return {
            "description": f"{self.name}: {self.role}",
            "prompt": prompt,
            "subagent_type": self.agent_type.value,
            "model": self.model.value,
        }


@dataclass
class WorkflowSpec:
    """Specification for a multi-agent workflow."""

    name: str
    description: str
    pattern: WorkflowPattern
    agents: list[AgentSpec]
    auto_aggregate: bool = True
    min_consensus: float = 0.6  # For consensus pattern
    max_iterations: int = 3  # For iterative patterns
    tags: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result from an agent execution."""

    agent_name: str
    model: str
    success: bool
    output: str
    execution_time_ms: float
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class WorkflowResult:
    """Aggregated results from a workflow execution."""

    workflow_name: str
    pattern: str
    total_agents: int
    successful: int
    failed: int
    total_time_ms: float
    results: list[AgentResult] = field(default_factory=list)
    aggregated_output: str = ""
    consensus_score: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Intelligent Model Selector
# =============================================================================


class ModelSelector:
    """Intelligently select the best model for a task."""

    # Keywords that suggest different complexity levels
    COMPLEXITY_KEYWORDS = {
        TaskComplexity.TRIVIAL: ["find", "search", "grep", "list", "count", "check if"],
        TaskComplexity.SIMPLE: ["read", "analyze file", "summarize", "explain", "what is"],
        TaskComplexity.MODERATE: ["implement", "fix", "refactor", "update", "modify", "add feature"],
        TaskComplexity.COMPLEX: ["design", "architect", "optimize", "migrate", "rewrite", "plan"],
        TaskComplexity.CRITICAL: ["security", "risk", "trading", "production", "deploy", "financial"],
    }

    # Model recommendations by complexity
    MODEL_MAP = {
        TaskComplexity.TRIVIAL: Model.HAIKU,
        TaskComplexity.SIMPLE: Model.HAIKU,
        TaskComplexity.MODERATE: Model.SONNET,
        TaskComplexity.COMPLEX: Model.SONNET,
        TaskComplexity.CRITICAL: Model.OPUS,
    }

    @classmethod
    def analyze_complexity(cls, task_description: str) -> TaskComplexity:
        """Analyze task description to determine complexity."""
        task_lower = task_description.lower()

        # Check for keywords in order of priority (critical first)
        for complexity in reversed(list(TaskComplexity)):
            keywords = cls.COMPLEXITY_KEYWORDS.get(complexity, [])
            for keyword in keywords:
                if keyword in task_lower:
                    return complexity

        return TaskComplexity.MODERATE  # Default

    @classmethod
    def select_model(cls, task_description: str, override_critical: bool = False) -> Model:
        """Select the best model for a task."""
        complexity = cls.analyze_complexity(task_description)

        # Allow override for cost savings on non-critical tasks
        if override_critical and complexity == TaskComplexity.CRITICAL:
            complexity = TaskComplexity.COMPLEX

        return cls.MODEL_MAP.get(complexity, Model.SONNET)

    @classmethod
    def recommend_agent_type(cls, task_description: str) -> AgentType:
        """Recommend the best agent type for a task."""
        task_lower = task_description.lower()

        if any(kw in task_lower for kw in ["find", "search", "grep", "where", "list"]):
            return AgentType.EXPLORE
        elif any(kw in task_lower for kw in ["plan", "design", "architect", "strategy"]):
            return AgentType.PLAN
        elif any(kw in task_lower for kw in ["claude code", "how to", "documentation"]):
            return AgentType.GUIDE
        else:
            return AgentType.GENERAL


# =============================================================================
# Pre-Built Agent Templates
# =============================================================================

AGENT_TEMPLATES: dict[str, AgentSpec] = {
    # Search Agents (Haiku - fast)
    "code_finder": AgentSpec(
        name="CodeFinder",
        role="Find code patterns and implementations",
        model=Model.HAIKU,
        agent_type=AgentType.EXPLORE,
        prompt_template="""Find all occurrences of: {query}

Search in:
- Source code files (*.py)
- Configuration files
- Test files

Return:
- File paths with line numbers
- Code snippets (max 20 lines each)
- Usage patterns found""",
        tags=["search", "fast"],
    ),
    "doc_finder": AgentSpec(
        name="DocFinder",
        role="Find documentation and comments",
        model=Model.HAIKU,
        agent_type=AgentType.EXPLORE,
        prompt_template="""Search documentation for: {query}

Look in:
- docs/ directory
- README files
- CLAUDE.md
- Docstrings and comments

Return relevant documentation excerpts with file locations.""",
        tags=["search", "docs", "fast"],
    ),
    "test_finder": AgentSpec(
        name="TestFinder",
        role="Find related tests",
        model=Model.HAIKU,
        agent_type=AgentType.EXPLORE,
        prompt_template="""Find tests related to: {query}

Search in:
- tests/unit/
- tests/integration/
- tests/regression/
- conftest.py fixtures

Return test cases, fixtures, and coverage information.""",
        tags=["search", "tests", "fast"],
    ),
    # Analysis Agents (Haiku/Sonnet)
    "security_scanner": AgentSpec(
        name="SecurityScanner",
        role="Scan for security vulnerabilities",
        model=Model.HAIKU,
        agent_type=AgentType.EXPLORE,
        prompt_template="""Scan for security vulnerabilities in: {target}

Check for:
- SQL injection, command injection
- Hardcoded credentials/secrets
- Path traversal
- OWASP Top 10 issues
- Unsafe deserialization

Report with severity (CRITICAL/HIGH/MEDIUM/LOW) and file:line references.""",
        tags=["security", "review"],
    ),
    "type_checker": AgentSpec(
        name="TypeChecker",
        role="Check type hint completeness",
        model=Model.HAIKU,
        agent_type=AgentType.EXPLORE,
        prompt_template="""Check type hints in: {target}

Verify:
- All functions have type hints
- Return types specified
- Optional vs None handling
- Generic types used correctly

Report missing/incorrect type hints with file:line references.""",
        tags=["quality", "types", "fast"],
    ),
    "test_analyzer": AgentSpec(
        name="TestAnalyzer",
        role="Analyze test coverage",
        model=Model.HAIKU,
        agent_type=AgentType.EXPLORE,
        prompt_template="""Analyze test coverage for: {target}

Find:
- Functions without tests
- Missing edge cases
- Integration test gaps
- Suggested tests to add

Reference the tests/ directory structure.""",
        tags=["quality", "tests"],
    ),
    # Review Agents (Sonnet)
    "architect": AgentSpec(
        name="Architect",
        role="Review architecture and design",
        model=Model.SONNET,
        agent_type=AgentType.PLAN,
        prompt_template="""Review architecture of: {target}

Evaluate:
- SOLID principles adherence
- Design pattern appropriateness
- Coupling and cohesion
- Dependency management
- Scalability considerations

Provide specific, actionable recommendations.""",
        tags=["architecture", "review"],
    ),
    "risk_reviewer": AgentSpec(
        name="RiskReviewer",
        role="Review trading risk management",
        model=Model.SONNET,
        agent_type=AgentType.EXPLORE,
        prompt_template="""Review trading code for risk management: {target}

CRITICAL CHECKS:
1. Circuit breaker integration
2. Position limits (max 25%)
3. Daily loss limits (max 3%)
4. Max drawdown protection (10%)
5. Order validation
6. Stop losses

FLAG code that bypasses safety checks.
Reference models/circuit_breaker.py patterns.""",
        tags=["trading", "risk", "critical"],
    ),
    "execution_reviewer": AgentSpec(
        name="ExecutionReviewer",
        role="Review order execution logic",
        model=Model.HAIKU,
        agent_type=AgentType.EXPLORE,
        prompt_template="""Review order execution in: {target}

Check:
1. Order validation before submission
2. Fill handling (partial, complete)
3. Cancel/replace logic
4. Duplicate prevention
5. Rate limiting

Reference execution/smart_execution.py patterns.""",
        tags=["trading", "execution"],
    ),
    "backtest_reviewer": AgentSpec(
        name="BacktestReviewer",
        role="Check for backtesting issues",
        model=Model.HAIKU,
        agent_type=AgentType.EXPLORE,
        prompt_template="""Check backtesting integrity of: {target}

LOOK FOR:
- Look-ahead bias (using future data)
- Survivorship bias
- Unrealistic fill assumptions
- Missing transaction costs
- Data snooping

Reference correct patterns in algorithms/.""",
        tags=["trading", "backtest"],
    ),
    # Implementation Agents (Sonnet)
    "implementer": AgentSpec(
        name="Implementer",
        role="Implement features and fixes",
        model=Model.SONNET,
        agent_type=AgentType.GENERAL,
        prompt_template="""Implement: {task}

Requirements:
{requirements}

Follow project coding standards:
- Type hints on all functions
- Google-style docstrings
- Defensive error handling
- Tests for new code

Reference existing patterns in the codebase.""",
        tags=["implementation"],
    ),
    "refactorer": AgentSpec(
        name="Refactorer",
        role="Refactor code for quality",
        model=Model.SONNET,
        agent_type=AgentType.GENERAL,
        prompt_template="""Refactor: {target}

Goals:
{goals}

Maintain:
- All existing tests passing
- Same public API (unless specified)
- Documentation updated

Apply SOLID principles and project patterns.""",
        tags=["refactor", "quality"],
    ),
    # Deep Analysis Agents (Opus)
    "deep_architect": AgentSpec(
        name="DeepArchitect",
        role="Deep architectural analysis",
        model=Model.OPUS,
        agent_type=AgentType.PLAN,
        prompt_template="""Perform deep architectural analysis of: {target}

Analyze:
1. Current architecture strengths/weaknesses
2. Scalability bottlenecks
3. Technical debt assessment
4. Migration strategies
5. Long-term maintainability

Provide comprehensive recommendations with trade-offs.""",
        tags=["architecture", "deep", "opus"],
    ),
    "critical_reviewer": AgentSpec(
        name="CriticalReviewer",
        role="Critical review for production code",
        model=Model.OPUS,
        agent_type=AgentType.GENERAL,
        prompt_template="""Perform critical review of: {target}

This is production/trading code. Check:
1. All safety mechanisms present
2. Error handling comprehensive
3. Edge cases covered
4. Performance acceptable
5. Security vulnerabilities
6. Regulatory compliance considerations

This review must be thorough - financial risk involved.""",
        tags=["critical", "production", "opus"],
    ),
    # =========================================================================
    # Quick Haiku Agents (UPGRADE-017-MEDIUM Phase 1)
    # =========================================================================
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
}


# =============================================================================
# Pre-Built Workflows
# =============================================================================

WORKFLOW_TEMPLATES: dict[str, WorkflowSpec] = {
    # Parallel Workflows
    "code_review": WorkflowSpec(
        name="Comprehensive Code Review",
        description="4-agent parallel code review covering security, types, tests, and architecture",
        pattern=WorkflowPattern.PARALLEL,
        agents=[
            AGENT_TEMPLATES["security_scanner"],
            AGENT_TEMPLATES["type_checker"],
            AGENT_TEMPLATES["test_analyzer"],
            AGENT_TEMPLATES["architect"],
        ],
        tags=["review", "quality"],
    ),
    "multi_search": WorkflowSpec(
        name="Multi-Source Search",
        description="3-agent parallel search across code, docs, and tests",
        pattern=WorkflowPattern.PARALLEL,
        agents=[
            AGENT_TEMPLATES["code_finder"],
            AGENT_TEMPLATES["doc_finder"],
            AGENT_TEMPLATES["test_finder"],
        ],
        tags=["search", "fast"],
    ),
    "trading_review": WorkflowSpec(
        name="Trading Safety Review",
        description="3-agent review focused on trading safety",
        pattern=WorkflowPattern.PARALLEL,
        agents=[
            AGENT_TEMPLATES["risk_reviewer"],
            AGENT_TEMPLATES["execution_reviewer"],
            AGENT_TEMPLATES["backtest_reviewer"],
        ],
        tags=["trading", "safety", "critical"],
    ),
    # Sequential Workflows
    "implementation_pipeline": WorkflowSpec(
        name="Implementation Pipeline",
        description="Sequential: Research → Plan → Implement → Review",
        pattern=WorkflowPattern.SEQUENTIAL,
        agents=[
            AgentSpec(
                name="Researcher",
                role="Research existing patterns",
                model=Model.HAIKU,
                agent_type=AgentType.EXPLORE,
                prompt_template="Research existing code patterns for: {task}",
            ),
            AgentSpec(
                name="Planner",
                role="Create implementation plan",
                model=Model.SONNET,
                agent_type=AgentType.PLAN,
                prompt_template="Create implementation plan for: {task}\n\nBased on research:\n{prev_output}",
            ),
            AGENT_TEMPLATES["implementer"],
            AgentSpec(
                name="Reviewer",
                role="Review implementation",
                model=Model.SONNET,
                agent_type=AgentType.GENERAL,
                prompt_template="Review implementation for: {task}\n\nCode:\n{prev_output}",
            ),
        ],
        tags=["implementation", "full-cycle"],
    ),
    # Consensus Workflows
    "critical_decision": WorkflowSpec(
        name="Critical Decision Consensus",
        description="3 agents vote on critical decisions",
        pattern=WorkflowPattern.CONSENSUS,
        min_consensus=0.66,
        agents=[
            AgentSpec(
                name="Analyst1",
                role="First analyst perspective",
                model=Model.SONNET,
                agent_type=AgentType.GENERAL,
                prompt_template="""Analyze this decision: {decision}

Consider:
- Benefits and risks
- Alternative approaches
- Implementation complexity

Vote: APPROVE, REJECT, or NEEDS_MORE_INFO
Provide reasoning.""",
            ),
            AgentSpec(
                name="Analyst2",
                role="Second analyst perspective",
                model=Model.SONNET,
                agent_type=AgentType.GENERAL,
                prompt_template="""Independently analyze: {decision}

Consider:
- Technical feasibility
- Maintenance burden
- Team impact

Vote: APPROVE, REJECT, or NEEDS_MORE_INFO
Provide reasoning.""",
            ),
            AgentSpec(
                name="RiskAnalyst",
                role="Risk-focused analyst",
                model=Model.SONNET,
                agent_type=AgentType.GENERAL,
                prompt_template="""Risk analysis for: {decision}

Focus on:
- What could go wrong
- Mitigation strategies
- Worst case scenarios

Vote: APPROVE, REJECT, or NEEDS_MORE_INFO
Provide reasoning.""",
            ),
        ],
        tags=["decision", "consensus", "critical"],
    ),
    # Hierarchical Workflows
    "managed_review": WorkflowSpec(
        name="Manager-Coordinated Review",
        description="Manager agent coordinates specialist workers",
        pattern=WorkflowPattern.HIERARCHICAL,
        agents=[
            AgentSpec(
                name="Manager",
                role="Coordinate review and aggregate findings",
                model=Model.SONNET,
                agent_type=AgentType.PLAN,
                prompt_template="""You are the review manager for: {target}

Coordinate these specialists and aggregate their findings:
1. Security specialist
2. Quality specialist
3. Architecture specialist

Provide a unified report with prioritized action items.""",
            ),
            AGENT_TEMPLATES["security_scanner"],
            AGENT_TEMPLATES["type_checker"],
            AGENT_TEMPLATES["architect"],
        ],
        tags=["review", "managed"],
    ),
    # =========================================================================
    # RIC Integration Workflows (UPGRADE-017-MEDIUM Phase 4)
    # =========================================================================
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
                prompt_template="""Research: {topic} best practices 2025.

Search for current information and industry standards.
Return formatted findings suitable for docs/research/.""",
                tags=["research", "web"],
            ),
            AgentSpec(
                name="CodebaseSearcher",
                role="Find existing implementations",
                model=Model.HAIKU,
                agent_type=AgentType.EXPLORE,
                prompt_template="""Find existing code for: {topic}.

Search in:
- algorithms/
- execution/
- models/
- scanners/

Return file:line references and code snippets.""",
                tags=["search", "code"],
            ),
            AgentSpec(
                name="DocSearcher",
                role="Search project documentation",
                model=Model.HAIKU,
                agent_type=AgentType.EXPLORE,
                prompt_template="""Find docs about: {topic}.

Search in:
- docs/
- CLAUDE.md
- README files
- Docstrings

Return relevant documentation excerpts.""",
                tags=["search", "docs"],
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
}


# =============================================================================
# State Management
# =============================================================================


class AgentState:
    """Manage persistent state for agent orchestration."""

    def __init__(self, state_file: Path = AGENT_STATE_FILE):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        """Load state from file."""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except json.JSONDecodeError:
                return self._default_state()
        return self._default_state()

    def _default_state(self) -> dict[str, Any]:
        """Return default state structure."""
        return {
            "active_workflows": [],
            "completed_workflows": [],
            "agent_stats": {},
            "last_run": None,
            "total_runs": 0,
        }

    def save(self):
        """Save state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def record_workflow(self, result: WorkflowResult):
        """Record a completed workflow."""
        record = {
            "name": result.workflow_name,
            "pattern": result.pattern,
            "agents": result.total_agents,
            "successful": result.successful,
            "failed": result.failed,
            "time_ms": result.total_time_ms,
            "timestamp": result.timestamp,
        }

        self.state["completed_workflows"].append(record)
        self.state["last_run"] = result.timestamp
        self.state["total_runs"] += 1

        # Keep only last 100 workflows
        if len(self.state["completed_workflows"]) > 100:
            self.state["completed_workflows"] = self.state["completed_workflows"][-100:]

        self.save()

    def get_stats(self) -> dict[str, Any]:
        """Get aggregated statistics."""
        completed = self.state["completed_workflows"]
        if not completed:
            return {"total_runs": 0, "avg_success_rate": 0, "avg_time_ms": 0}

        total_agents = sum(w["agents"] for w in completed)
        total_successful = sum(w["successful"] for w in completed)
        total_time = sum(w["time_ms"] for w in completed)

        return {
            "total_runs": len(completed),
            "total_agents_spawned": total_agents,
            "avg_success_rate": total_successful / total_agents if total_agents else 0,
            "avg_time_ms": total_time / len(completed),
            "last_run": self.state["last_run"],
        }


# =============================================================================
# Cost & Token Tracking (UPGRADE-017-MEDIUM Phase 5)
# =============================================================================


@dataclass
class CostEstimate:
    """Cost estimate for agent execution."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float

    # Pricing per million tokens (December 2025)
    COSTS = {
        "haiku": {"input": 0.25, "output": 1.25},
        "sonnet": {"input": 3.0, "output": 15.0},
        "opus": {"input": 15.0, "output": 75.0},
    }

    @classmethod
    def estimate(cls, model: str, prompt_len: int = 1000, output_len: int = 2000) -> "CostEstimate":
        """Estimate cost based on model and token counts.

        Args:
            model: Model name (haiku, sonnet, opus)
            prompt_len: Estimated input tokens
            output_len: Estimated output tokens

        Returns:
            CostEstimate with calculated cost
        """
        rates = cls.COSTS.get(model, cls.COSTS["sonnet"])
        cost = (prompt_len * rates["input"] + output_len * rates["output"]) / 1_000_000

        return cls(
            model=model,
            input_tokens=prompt_len,
            output_tokens=output_len,
            cost_usd=cost,
        )


class TokenTracker:
    """Track token usage across agents and models."""

    def __init__(self):
        self.usage: dict[str, dict[str, int]] = {}  # model -> {input, output}
        self.by_agent: dict[str, dict[str, int]] = {}  # agent -> {input, output}
        self.session_start = datetime.now()

    def record(self, agent_name: str, model: str, tokens_in: int, tokens_out: int):
        """Record token usage for an agent execution.

        Args:
            agent_name: Name of the agent
            model: Model used (haiku, sonnet, opus)
            tokens_in: Input tokens used
            tokens_out: Output tokens used
        """
        # By model
        if model not in self.usage:
            self.usage[model] = {"input": 0, "output": 0}
        self.usage[model]["input"] += tokens_in
        self.usage[model]["output"] += tokens_out

        # By agent
        if agent_name not in self.by_agent:
            self.by_agent[agent_name] = {"input": 0, "output": 0}
        self.by_agent[agent_name]["input"] += tokens_in
        self.by_agent[agent_name]["output"] += tokens_out

    def get_total_cost(self) -> float:
        """Calculate total cost based on model pricing.

        Returns:
            Total estimated cost in USD
        """
        total = 0.0
        for model, tokens in self.usage.items():
            rates = CostEstimate.COSTS.get(model, CostEstimate.COSTS["sonnet"])
            total += (tokens["input"] * rates["input"] + tokens["output"] * rates["output"]) / 1_000_000
        return total

    def get_report(self) -> str:
        """Generate token usage report.

        Returns:
            Formatted markdown report
        """
        lines = ["## Token Usage Report", ""]
        lines.append(f"Session started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("### By Model")
        lines.append("| Model | Input | Output | Est. Cost |")
        lines.append("|-------|-------|--------|-----------|")

        for model, tokens in self.usage.items():
            rates = CostEstimate.COSTS.get(model, CostEstimate.COSTS["sonnet"])
            cost = (tokens["input"] * rates["input"] + tokens["output"] * rates["output"]) / 1_000_000
            lines.append(f"| {model} | {tokens['input']:,} | {tokens['output']:,} | ${cost:.4f} |")

        lines.append(f"\n**Total Estimated Cost**: ${self.get_total_cost():.4f}")

        if self.by_agent:
            lines.append("\n### By Agent")
            lines.append("| Agent | Input | Output |")
            lines.append("|-------|-------|--------|")
            for agent, tokens in sorted(self.by_agent.items()):
                lines.append(f"| {agent} | {tokens['input']:,} | {tokens['output']:,} |")

        return "\n".join(lines)


def estimate_workflow_cost(workflow: WorkflowSpec) -> CostEstimate:
    """Estimate total cost for a workflow.

    Args:
        workflow: WorkflowSpec to estimate

    Returns:
        CostEstimate with total workflow cost
    """
    total_input = 0
    total_output = 0

    for agent in workflow.agents:
        # Estimate based on prompt template length (~4 chars per token)
        prompt_tokens = len(agent.prompt_template) // 4
        output_tokens = 2000  # Default estimate

        # Scale by model (opus tends to be more verbose)
        if agent.model == Model.OPUS:
            output_tokens *= 2

        total_input += prompt_tokens
        total_output += output_tokens

    # Use most expensive model in workflow for conservative estimate
    models = [a.model.value for a in workflow.agents]
    if "opus" in models:
        avg_model = "opus"
    elif "sonnet" in models:
        avg_model = "sonnet"
    else:
        avg_model = "haiku"

    return CostEstimate.estimate(avg_model, total_input, total_output)


# =============================================================================
# Execution Tracing (UPGRADE-017-MEDIUM Phase 6)
# =============================================================================


@dataclass
class TraceSpan:
    """A single span in an execution trace."""

    span_id: str
    agent_name: str
    started_at: datetime
    ended_at: datetime | None = None
    status: str = "running"  # "running", "success", "failed", "timeout"
    duration_ms: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None


@dataclass
class ExecutionTrace:
    """Complete execution trace for a workflow."""

    trace_id: str
    workflow_name: str
    started_at: datetime
    ended_at: datetime | None = None
    spans: list[TraceSpan] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        """Get total trace duration in milliseconds."""
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds() * 1000
        return None


class Tracer:
    """Trace workflow execution for debugging and observability."""

    def __init__(self, storage_dir: Path = None):
        """Initialize tracer.

        Args:
            storage_dir: Directory to store traces
        """
        self.storage_dir = storage_dir or PROJECT_ROOT / ".claude" / "traces"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.current_trace: ExecutionTrace | None = None

    def start_trace(self, workflow_name: str) -> str:
        """Start a new execution trace.

        Args:
            workflow_name: Name of the workflow being traced

        Returns:
            Trace ID
        """
        trace_id = f"{workflow_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_trace = ExecutionTrace(
            trace_id=trace_id,
            workflow_name=workflow_name,
            started_at=datetime.now(),
        )
        return trace_id

    def start_span(self, agent_name: str) -> str:
        """Start a new span within the current trace.

        Args:
            agent_name: Name of the agent being traced

        Returns:
            Span ID
        """
        if not self.current_trace:
            raise ValueError("No active trace. Call start_trace first.")

        span_id = f"{agent_name}_{len(self.current_trace.spans)}"
        span = TraceSpan(
            span_id=span_id,
            agent_name=agent_name,
            started_at=datetime.now(),
        )
        self.current_trace.spans.append(span)
        return span_id

    def end_span(self, span_id: str, status: str, tokens_in: int = 0, tokens_out: int = 0, error: str = None):
        """End a span with results.

        Args:
            span_id: ID of the span to end
            status: Final status (success, failed, timeout)
            tokens_in: Input tokens used
            tokens_out: Output tokens used
            error: Error message if failed
        """
        if not self.current_trace:
            return

        for span in self.current_trace.spans:
            if span.span_id == span_id:
                span.ended_at = datetime.now()
                span.status = status
                span.duration_ms = (span.ended_at - span.started_at).total_seconds() * 1000
                span.tokens_in = tokens_in
                span.tokens_out = tokens_out
                span.error = error
                break

    def end_trace(self):
        """End the current trace and save to storage."""
        if self.current_trace:
            self.current_trace.ended_at = datetime.now()
            self._save_trace()
            trace_id = self.current_trace.trace_id
            self.current_trace = None
            return trace_id
        return None

    def _save_trace(self):
        """Save the current trace to storage."""
        if not self.current_trace:
            return

        filepath = self.storage_dir / f"{self.current_trace.trace_id}.json"

        # Convert to dict for JSON serialization
        trace_dict = {
            "trace_id": self.current_trace.trace_id,
            "workflow_name": self.current_trace.workflow_name,
            "started_at": self.current_trace.started_at.isoformat(),
            "ended_at": self.current_trace.ended_at.isoformat() if self.current_trace.ended_at else None,
            "duration_ms": self.current_trace.duration_ms,
            "spans": [
                {
                    "span_id": s.span_id,
                    "agent_name": s.agent_name,
                    "started_at": s.started_at.isoformat(),
                    "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                    "tokens_in": s.tokens_in,
                    "tokens_out": s.tokens_out,
                    "error": s.error,
                }
                for s in self.current_trace.spans
            ],
        }

        filepath.write_text(json.dumps(trace_dict, indent=2))

    def list_traces(self, limit: int = 10) -> list[dict[str, Any]]:
        """List recent traces.

        Args:
            limit: Maximum number of traces to return

        Returns:
            List of trace summaries
        """
        traces = sorted(self.storage_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        summaries = []
        for trace_file in traces[:limit]:
            try:
                trace = json.loads(trace_file.read_text())
                summaries.append(
                    {
                        "trace_id": trace["trace_id"],
                        "workflow": trace["workflow_name"],
                        "duration_ms": trace.get("duration_ms"),
                        "spans": len(trace.get("spans", [])),
                        "status": "success"
                        if all(s.get("status") == "success" for s in trace.get("spans", []))
                        else "partial",
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue

        return summaries

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Get a specific trace by ID.

        Args:
            trace_id: ID of the trace to retrieve

        Returns:
            Trace data or None if not found
        """
        filepath = self.storage_dir / f"{trace_id}.json"
        if filepath.exists():
            return json.loads(filepath.read_text())
        return None


def format_trace(trace: dict[str, Any]) -> str:
    """Format a trace for display.

    Args:
        trace: Trace dictionary

    Returns:
        Formatted string for display
    """
    lines = [
        f"## Trace: {trace['trace_id']}",
        f"Workflow: {trace['workflow_name']}",
        f"Duration: {trace.get('duration_ms', 0):.0f}ms",
        "",
        "### Spans",
        "| Agent | Status | Duration | Tokens |",
        "|-------|--------|----------|--------|",
    ]

    for span in trace.get("spans", []):
        status_icon = {"success": "✅", "failed": "❌", "timeout": "⏱️"}.get(span["status"], "?")
        tokens = f"{span.get('tokens_in', 0)}/{span.get('tokens_out', 0)}"
        lines.append(f"| {span['agent_name']} | {status_icon} | {span.get('duration_ms', 0):.0f}ms | {tokens} |")

    if any(s.get("error") for s in trace.get("spans", [])):
        lines.append("\n### Errors")
        for span in trace.get("spans", []):
            if span.get("error"):
                lines.append(f"- **{span['agent_name']}**: {span['error']}")

    return "\n".join(lines)


# =============================================================================
# Auto-Persistence (UPGRADE-017-MEDIUM Phase 7)
# =============================================================================


class ResearchPersister:
    """Auto-save research findings to docs/research/."""

    RESEARCH_DIR = PROJECT_ROOT / "docs" / "research"

    @classmethod
    def save_entry(cls, query: str, findings: str, sources: list[str] = None, doc_name: str = None) -> bool:
        """Save a research entry to documentation.

        Args:
            query: The search query used
            findings: Key findings to save
            sources: List of source URLs
            doc_name: Target document (default: auto-generated)

        Returns:
            True if saved successfully
        """
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

        # Auto-generate doc name if not provided
        if not doc_name:
            date_str = datetime.now().strftime("%Y%m%d")
            doc_name = f"RESEARCH-{date_str}.md"

        doc_path = cls.RESEARCH_DIR / doc_name

        # Create research dir if needed
        cls.RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

        entry = f"""

---

### Research Entry - {timestamp}

**Search Query**: "{query}"

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
            # Append to existing or create new
            mode = "a" if doc_path.exists() else "w"
            if mode == "w":
                # Add header for new file
                header = f"# Research Log - {datetime.now().strftime('%B %Y')}\n\n"
                header += "Auto-generated research entries from web searches.\n"
                entry = header + entry

            with open(doc_path, mode) as f:
                f.write(entry)
            return True
        except Exception as e:
            print(f"Failed to save research: {e}")
            return False


def save_web_research(query: str, findings: str, sources: list[str] = None, upgrade: str = None) -> bool:
    """Convenience function to save web research findings.

    Args:
        query: Search query used
        findings: Key findings in markdown format
        sources: List of source URLs with optional dates
        upgrade: Optional UPGRADE-XXX name to save to specific doc

    Returns:
        True if saved successfully
    """
    doc_name = None
    if upgrade:
        doc_name = f"{upgrade}-RESEARCH.md"

    return ResearchPersister.save_entry(query=query, findings=findings, sources=sources, doc_name=doc_name)


def process_web_researcher_result(result: AgentResult) -> bool:
    """Process web researcher result and auto-save if successful.

    Args:
        result: AgentResult from a web researcher agent

    Returns:
        True if findings were saved
    """
    if not result.success:
        return False

    output = result.output

    # Extract query
    query_match = re.search(r"\*\*Query\*\*:\s*(.+)", output)
    query = query_match.group(1) if query_match else "Unknown query"

    # Extract sources
    sources = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", output)
    source_list = [f"[{title}]({url})" for title, url in sources]

    # Extract findings
    findings_match = re.search(r"\*\*Key Findings\*\*:\s*(.+?)(?:\*\*|$)", output, re.DOTALL)
    findings = findings_match.group(1).strip() if findings_match else output

    return save_web_research(query, findings, source_list)


# =============================================================================
# RIC Phase Detection (UPGRADE-017-MEDIUM Phase 4)
# =============================================================================


def detect_ric_phase(context: dict[str, Any] = None) -> str | None:
    """Detect current RIC phase from context or progress file.

    Args:
        context: Optional context dict with phase info

    Returns:
        Phase name (RESEARCH, PLAN, BUILD, VERIFY, REFLECT) or None
    """
    # Check context first
    if context and "ric_phase" in context:
        return context["ric_phase"]

    # Check progress file
    progress_file = PROJECT_ROOT / "claude-progress.txt"
    if progress_file.exists():
        content = progress_file.read_text()

        # Try format 1: [P0], [P1], etc.
        phase_match = re.search(r"\[P(\d)\]", content)
        if phase_match:
            phase_num = int(phase_match.group(1))
            phases = ["RESEARCH", "PLAN", "BUILD", "VERIFY", "REFLECT"]
            if 0 <= phase_num < len(phases):
                return phases[phase_num]

        # Try format 2: "Phase: P2 BUILD" or "- Phase: P0 RESEARCH"
        phase_match = re.search(r"Phase:\s*P(\d)\s*(\w+)", content, re.IGNORECASE)
        if phase_match:
            # Use the phase name directly if available
            phase_name = phase_match.group(2).upper()
            if phase_name in ["RESEARCH", "PLAN", "BUILD", "VERIFY", "REFLECT"]:
                return phase_name
            # Fall back to number
            phase_num = int(phase_match.group(1))
            phases = ["RESEARCH", "PLAN", "BUILD", "VERIFY", "REFLECT"]
            if 0 <= phase_num < len(phases):
                return phases[phase_num]
    return None


def get_ric_recommended_agents(phase: str) -> list[str]:
    """Get recommended agents for a RIC phase.

    Args:
        phase: RIC phase name (RESEARCH, PLAN, BUILD, VERIFY, REFLECT)

    Returns:
        List of recommended agent names
    """
    recommendations = {
        "RESEARCH": ["web_researcher", "doc_finder", "code_finder"],
        "PLAN": ["architect"],
        "BUILD": ["implementer", "refactorer"],
        "VERIFY": ["test_analyzer", "type_checker", "security_scanner"],
        "REFLECT": ["deep_architect"],
    }
    return recommendations.get(phase.upper(), [])


def get_ric_recommended_workflow(phase: str) -> str | None:
    """Get recommended workflow for a RIC phase.

    Args:
        phase: RIC phase name

    Returns:
        Workflow name or None
    """
    workflows = {
        "RESEARCH": "ric_research",
        "VERIFY": "ric_verify",
    }
    return workflows.get(phase.upper())


# =============================================================================
# Workflow Generator (Creates Task tool calls)
# =============================================================================


class WorkflowGenerator:
    """Generate Claude Code Task tool calls for workflows."""

    @staticmethod
    def generate_parallel(workflow: WorkflowSpec, context: dict[str, Any]) -> str:
        """Generate parallel Task calls."""
        calls = []
        for agent in workflow.agents:
            params = agent.to_task_call(context)
            calls.append(f"""Task(
    subagent_type="{params['subagent_type']}",
    model="{params['model']}",
    description="{params['description']}",
    prompt=\"\"\"{params['prompt']}\"\"\"
)""")

        return "\n\n".join(calls)

    @staticmethod
    def generate_sequential(workflow: WorkflowSpec, context: dict[str, Any]) -> str:
        """Generate sequential Task calls with dependency chain."""
        instructions = [
            f"# {workflow.name} - Sequential Workflow",
            f"# {workflow.description}",
            "",
            "# Execute these agents IN ORDER, passing output to next:",
            "",
        ]

        for i, agent in enumerate(workflow.agents, 1):
            params = agent.to_task_call(context)
            instructions.append(f"# Step {i}: {agent.name}")
            instructions.append(f"""Task(
    subagent_type="{params['subagent_type']}",
    model="{params['model']}",
    description="{params['description']}",
    prompt=\"\"\"{params['prompt']}\"\"\"
)""")
            if i < len(workflow.agents):
                instructions.append("# Wait for result, then pass to next agent as {{prev_output}}")
            instructions.append("")

        return "\n".join(instructions)

    @staticmethod
    def generate_consensus(workflow: WorkflowSpec, context: dict[str, Any]) -> str:
        """Generate consensus workflow calls."""
        instructions = [
            f"# {workflow.name} - Consensus Workflow",
            f"# {workflow.description}",
            f"# Minimum consensus required: {workflow.min_consensus:.0%}",
            "",
            "# Execute ALL agents in PARALLEL:",
            "",
        ]

        calls = []
        for agent in workflow.agents:
            params = agent.to_task_call(context)
            calls.append(f"""Task(
    subagent_type="{params['subagent_type']}",
    model="{params['model']}",
    description="{params['description']}",
    prompt=\"\"\"{params['prompt']}\"\"\"
)""")

        instructions.append("\n\n".join(calls))
        instructions.append("")
        instructions.append("# After all complete, tally votes:")
        instructions.append("# - APPROVE votes / Total = consensus score")
        instructions.append(f"# - If score >= {workflow.min_consensus:.0%}: APPROVED")
        instructions.append(f"# - If score < {workflow.min_consensus:.0%}: REJECTED")

        return "\n".join(instructions)


# =============================================================================
# Auto-Agent Selector
# =============================================================================


class AutoAgentSelector:
    """Automatically select agents and workflows based on task description."""

    # Task patterns and their recommended workflows/agents
    TASK_PATTERNS = [
        # Reviews
        (r"review|check|audit|analyze", ["code_review", "trading_review"]),
        # Searches
        (r"find|search|where|locate|grep", ["multi_search"]),
        # Trading specific
        (r"trading|risk|order|execution|backtest", ["trading_review"]),
        # Implementation
        (r"implement|create|build|add|develop", ["implementation_pipeline"]),
        # Decisions
        (r"should|decide|choose|evaluate|compare", ["critical_decision"]),
    ]

    @classmethod
    def select_workflow(cls, task: str) -> str | None:
        """Select best workflow for a task."""
        task_lower = task.lower()

        for pattern, workflows in cls.TASK_PATTERNS:
            if re.search(pattern, task_lower):
                # Return first matching workflow
                return workflows[0]

        return None

    @classmethod
    def select_agents(cls, task: str, max_agents: int = 5) -> list[AgentSpec]:
        """Select best agents for a task."""
        task_lower = task.lower()
        selected = []

        # Score each agent template
        scores = []
        for name, agent in AGENT_TEMPLATES.items():
            score = 0
            # Check role match
            for word in task_lower.split():
                if word in agent.role.lower():
                    score += 2
                if word in " ".join(agent.tags):
                    score += 1

            if score > 0:
                scores.append((score, name, agent))

        # Sort by score and take top agents
        scores.sort(reverse=True, key=lambda x: x[0])
        selected = [agent for _, _, agent in scores[:max_agents]]

        # If no matches, return default search agents
        if not selected:
            selected = [
                AGENT_TEMPLATES["code_finder"],
                AGENT_TEMPLATES["doc_finder"],
            ]

        return selected

    @classmethod
    def create_auto_workflow(cls, task: str) -> tuple[str, WorkflowSpec]:
        """Create an automatic workflow for any task."""
        # Try to find a matching template
        workflow_name = cls.select_workflow(task)
        if workflow_name and workflow_name in WORKFLOW_TEMPLATES:
            return workflow_name, WORKFLOW_TEMPLATES[workflow_name]

        # Create custom workflow with auto-selected agents
        agents = cls.select_agents(task)
        model = ModelSelector.select_model(task)

        # Add a coordinator agent for complex tasks
        if model in [Model.SONNET, Model.OPUS]:
            coordinator = AgentSpec(
                name="Coordinator",
                role="Coordinate and synthesize results",
                model=model,
                agent_type=AgentType.GENERAL,
                prompt_template=f"""Coordinate task: {task}

Synthesize results from other agents and provide:
1. Key findings
2. Recommended actions
3. Priority items

Format as actionable summary.""",
            )
            agents.append(coordinator)

        custom_workflow = WorkflowSpec(
            name=f"Auto: {task[:50]}...",
            description=f"Automatically generated workflow for: {task}",
            pattern=WorkflowPattern.PARALLEL,
            agents=agents,
            tags=["auto-generated"],
        )

        return "auto", custom_workflow


# =============================================================================
# CLI Interface
# =============================================================================


def print_help():
    """Print comprehensive help."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║       AGENT ORCHESTRATOR v1.5 - Multi-Agent Coordination Suite                ║
║                     (UPGRADE-017-MEDIUM Enhanced)                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  COMMANDS:                                                                   ║
║                                                                              ║
║    run <workflow> [context]   Run a pre-built workflow                       ║
║    auto "<task>"              Auto-select agents for any task                ║
║    list                       List all workflows and agents                  ║
║    agents                     List all agent templates                       ║
║    workflows                  List all workflow templates                    ║
║    status                     Show stats + circuit breaker + RIC phase       ║
║    generate <workflow>        Generate Task tool calls for workflow          ║
║    trace [id|--recent N]      View execution traces                          ║
║    ric-phase                  Show RIC phase and recommendations             ║
║    help                       Show this help message                         ║
║                                                                              ║
║  WORKFLOWS:                                                                  ║
║                                                                              ║
║    code_review       4-agent parallel: security, types, tests, architecture  ║
║    multi_search      3-agent parallel: code, docs, tests search              ║
║    trading_review    3-agent parallel: risk, execution, backtest             ║
║    critical_decision 3-agent consensus voting                                ║
║    implementation_pipeline  Sequential: research → plan → implement → review ║
║    managed_review    Hierarchical: manager coordinates specialists           ║
║    ric_research      RIC P0: web + codebase + docs search (haiku)            ║
║    ric_verify        RIC P3: tests + types + security (haiku)                ║
║                                                                              ║
║  NEW AGENTS (v1.5):                                                          ║
║                                                                              ║
║    web_researcher    Search web, format for docs (haiku)                     ║
║    text_extractor    Extract text from files (haiku)                         ║
║    grep_agent        Fast pattern search (haiku)                             ║
║    file_lister       List files by pattern (haiku)                           ║
║    research_saver    Save findings to docs/ (haiku)                          ║
║                                                                              ║
║  RELIABILITY FEATURES (v1.5):                                                ║
║                                                                              ║
║    - Retry with exponential backoff + jitter                                 ║
║    - Fallback routing when agents fail                                       ║
║    - Circuit breaker to prevent cascading failures                           ║
║    - Graceful degradation with partial results                               ║
║    - Execution tracing for debugging                                         ║
║    - Cost tracking per model/agent                                           ║
║    - Auto-persistence of research findings                                   ║
║                                                                              ║
║  EXAMPLES:                                                                   ║
║                                                                              ║
║    python agent_orchestrator.py run code_review target=algorithms/           ║
║    python agent_orchestrator.py auto "find all circuit breaker usage"        ║
║    python agent_orchestrator.py status                                       ║
║    python agent_orchestrator.py trace --recent 5                             ║
║    python agent_orchestrator.py ric-phase                                    ║
║                                                                              ║
║  MODEL SELECTION:                                                            ║
║                                                                              ║
║    haiku  - Fast/cheap: searches, simple checks ($0.25/MTok)                 ║
║    sonnet - Balanced: implementation, review ($3/MTok)                       ║
║    opus   - Deep: architecture, critical decisions ($15/MTok)                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def list_workflows():
    """List all workflow templates."""
    print("\n=== WORKFLOW TEMPLATES ===\n")
    for name, workflow in WORKFLOW_TEMPLATES.items():
        print(f"  {name}")
        print(f"    Description: {workflow.description}")
        print(f"    Pattern: {workflow.pattern.value}")
        print(f"    Agents: {len(workflow.agents)}")
        print(f"    Tags: {', '.join(workflow.tags)}")
        print()


def list_agents():
    """List all agent templates."""
    print("\n=== AGENT TEMPLATES ===\n")
    for name, agent in AGENT_TEMPLATES.items():
        print(f"  {name}")
        print(f"    Role: {agent.role}")
        print(f"    Model: {agent.model.value}")
        print(f"    Type: {agent.agent_type.value}")
        print(f"    Tags: {', '.join(agent.tags)}")
        print()


def show_status():
    """Show orchestration statistics including circuit breaker status."""
    state = AgentState()
    stats = state.get_stats()

    print("\n=== AGENT ORCHESTRATOR STATUS ===\n")
    print(f"  Total Runs: {stats.get('total_runs', 0)}")
    print(f"  Total Agents Spawned: {stats.get('total_agents_spawned', 0)}")
    print(f"  Average Success Rate: {stats.get('avg_success_rate', 0):.1%}")
    print(f"  Average Time: {stats.get('avg_time_ms', 0):.0f}ms")
    print(f"  Last Run: {stats.get('last_run') or 'Never'}")
    print()

    # Circuit breaker status
    cb = AgentCircuitBreaker()
    cb_status = cb.get_status()
    if cb_status:
        print("=== CIRCUIT BREAKER STATUS ===\n")
        print("  | Agent | Status |")
        print("  |-------|--------|")
        for agent, status in cb_status.items():
            print(f"  | {agent} | {status} |")
        print()

    # RIC phase detection
    ric_phase = detect_ric_phase()
    if ric_phase:
        print(f"=== RIC PHASE: {ric_phase} ===\n")
        recommended = get_ric_recommended_agents(ric_phase)
        print(f"  Recommended agents: {', '.join(recommended)}")
        workflow = get_ric_recommended_workflow(ric_phase)
        if workflow:
            print(f"  Recommended workflow: {workflow}")
        print()


def show_traces(args: list[str]):
    """Show execution traces."""
    tracer = Tracer()

    if not args or args[0] == "--recent":
        limit = int(args[1]) if len(args) > 1 else 5
        traces = tracer.list_traces(limit=limit)

        if not traces:
            print("\nNo traces found")
            return

        print(f"\n## Recent Traces (last {limit})\n")
        print("| Status | Trace ID | Workflow | Duration | Agents |")
        print("|--------|----------|----------|----------|--------|")

        for t in traces:
            status = "✅" if t["status"] == "success" else "⚠️"
            duration = t.get("duration_ms", 0) or 0
            print(f"| {status} | {t['trace_id']} | {t['workflow']} | {duration:.0f}ms | {t['spans']} |")
    else:
        # View specific trace
        trace = tracer.get_trace(args[0])
        if trace:
            print(format_trace(trace))
        else:
            print(f"\nTrace not found: {args[0]}")


def show_ric_phase():
    """Show current RIC phase and recommendations."""
    phase = detect_ric_phase()

    if not phase:
        print("\nNo RIC phase detected")
        print("Check claude-progress.txt for [P0], [P1], etc.")
        return

    print(f"\n=== Current RIC Phase: {phase} ===\n")

    recommended_agents = get_ric_recommended_agents(phase)
    print("Recommended Agents:")
    for agent_name in recommended_agents:
        if agent_name in AGENT_TEMPLATES:
            agent = AGENT_TEMPLATES[agent_name]
            print(f"  - {agent_name} ({agent.model.value}): {agent.role}")
        else:
            print(f"  - {agent_name}")

    workflow = get_ric_recommended_workflow(phase)
    if workflow:
        print(f"\nRecommended Workflow: {workflow}")
        print(f"  Run with: /agents run {workflow}")


def generate_workflow(workflow_name: str, context: dict[str, Any] = None):
    """Generate Task tool calls for a workflow."""
    if workflow_name not in WORKFLOW_TEMPLATES:
        print(f"Unknown workflow: {workflow_name}")
        print(f"Available: {', '.join(WORKFLOW_TEMPLATES.keys())}")
        return

    workflow = WORKFLOW_TEMPLATES[workflow_name]
    context = context or {"target": "<files>", "query": "<search>"}

    print(f"\n=== GENERATED TASK CALLS: {workflow_name} ===\n")

    if workflow.pattern == WorkflowPattern.PARALLEL:
        print(WorkflowGenerator.generate_parallel(workflow, context))
    elif workflow.pattern == WorkflowPattern.SEQUENTIAL:
        print(WorkflowGenerator.generate_sequential(workflow, context))
    elif workflow.pattern == WorkflowPattern.CONSENSUS:
        print(WorkflowGenerator.generate_consensus(workflow, context))
    else:
        print(WorkflowGenerator.generate_parallel(workflow, context))


def auto_select(task: str):
    """Auto-select agents for a task."""
    print(f"\n=== AUTO-SELECTING FOR: {task} ===\n")

    # Analyze task
    complexity = ModelSelector.analyze_complexity(task)
    recommended_model = ModelSelector.select_model(task)
    recommended_type = ModelSelector.recommend_agent_type(task)

    print(f"  Task Complexity: {complexity.name}")
    print(f"  Recommended Model: {recommended_model.value}")
    print(f"  Recommended Agent Type: {recommended_type.value}")
    print()

    # Get workflow or create custom
    workflow_name, workflow = AutoAgentSelector.create_auto_workflow(task)

    print(f"  Selected Workflow: {workflow_name}")
    print(f"  Pattern: {workflow.pattern.value}")
    print(f"  Agents: {len(workflow.agents)}")
    print()

    for agent in workflow.agents:
        print(f"    - {agent.name} ({agent.model.value}): {agent.role}")

    print("\n=== GENERATED TASK CALLS ===\n")
    context = {"task": task, "target": "<files>", "query": task}
    print(WorkflowGenerator.generate_parallel(workflow, context))


# =============================================================================
# Auto-Dispatch for Easy Tasks (UPGRADE-017-MEDIUM Enhancement)
# =============================================================================
# These functions make it easy to automatically spawn agent swarms for common tasks.
# Call these directly when you detect a task that benefits from parallel agents.


def quick_search(query: str) -> list[dict[str, Any]]:
    """Generate 3 parallel haiku search agents for a query.

    Returns list of Task tool call specs ready to execute.
    Use this for any "find X" or "search for X" task.

    Example:
        tasks = quick_search("error handling code")
        # Returns 3 Task specs for code_finder, doc_finder, test_finder
    """
    agents = ["code_finder", "doc_finder", "test_finder"]
    return [
        {
            "subagent_type": "Explore",
            "model": "haiku",
            "description": f"{AGENT_TEMPLATES[a].name}: {AGENT_TEMPLATES[a].role}",
            "prompt": AGENT_TEMPLATES[a].prompt_template.format(query=query, pattern=query, target=query),
        }
        for a in agents
        if a in AGENT_TEMPLATES
    ]


def quick_review(target: str) -> list[dict[str, Any]]:
    """Generate 4 parallel review agents for code review.

    Returns list of Task tool call specs ready to execute.
    Use this for any code review or quality check task.

    Example:
        tasks = quick_review("algorithms/")
        # Returns 4 Task specs for security, types, tests, architecture
    """
    agents = [
        ("security_scanner", "haiku"),
        ("type_checker", "haiku"),
        ("test_analyzer", "haiku"),
        ("architect", "sonnet"),
    ]
    return [
        {
            "subagent_type": AGENT_TEMPLATES[a].agent_type.value if a in AGENT_TEMPLATES else "Explore",
            "model": model,
            "description": f"{AGENT_TEMPLATES[a].name}: {AGENT_TEMPLATES[a].role}" if a in AGENT_TEMPLATES else a,
            "prompt": AGENT_TEMPLATES[a].prompt_template.format(target=target, files=target, query=target)
            if a in AGENT_TEMPLATES
            else f"Review: {target}",
        }
        for a, model in agents
        if a in AGENT_TEMPLATES
    ]


def quick_ric(phase: str, topic: str) -> list[dict[str, Any]]:
    """Generate agents for current RIC phase.

    Returns list of Task tool call specs for the given RIC phase.

    Example:
        tasks = quick_ric("RESEARCH", "circuit breaker patterns")
        # Returns 3 haiku agents for web, code, and doc search
    """
    agents = get_ric_recommended_agents(phase)
    return [
        {
            "subagent_type": AGENT_TEMPLATES[a].agent_type.value if a in AGENT_TEMPLATES else "Explore",
            "model": AGENT_TEMPLATES[a].model.value if a in AGENT_TEMPLATES else "haiku",
            "description": f"{AGENT_TEMPLATES[a].name}: {AGENT_TEMPLATES[a].role}" if a in AGENT_TEMPLATES else a,
            "prompt": AGENT_TEMPLATES[a].prompt_template.format(topic=topic, query=topic, target=topic)
            if a in AGENT_TEMPLATES
            else f"Analyze: {topic}",
        }
        for a in agents
        if a in AGENT_TEMPLATES
    ]


def auto_dispatch(task: str) -> dict[str, Any]:
    """Automatically analyze task and return optimal agent configuration.

    This is the main entry point for automatic agent swarms.
    Returns a dict with 'pattern', 'agents', and 'task_calls'.

    Example:
        result = auto_dispatch("find all error handling")
        # Returns: {
        #   'pattern': 'search',
        #   'agents': 3,
        #   'task_calls': [...],  # Ready to execute
        #   'estimated_cost': 0.01
        # }
    """
    task_lower = task.lower()

    # Pattern detection
    if any(kw in task_lower for kw in ["find", "search", "where", "locate", "grep"]):
        pattern = "search"
        task_calls = quick_search(task)

    elif any(kw in task_lower for kw in ["review", "check", "audit", "analyze", "scan"]):
        pattern = "review"
        task_calls = quick_review(task)

    elif any(kw in task_lower for kw in ["test", "verify", "validate"]):
        pattern = "verify"
        task_calls = quick_ric("VERIFY", task)

    elif any(kw in task_lower for kw in ["research", "learn", "explore", "investigate"]):
        pattern = "research"
        task_calls = quick_ric("RESEARCH", task)

    else:
        # Default: use auto selector
        pattern = "auto"
        _, workflow = AutoAgentSelector.create_auto_workflow(task)
        task_calls = [
            {
                "subagent_type": a.agent_type.value,
                "model": a.model.value,
                "description": f"{a.name}: {a.role}",
                "prompt": a.prompt_template.format(task=task, query=task, target=task),
            }
            for a in workflow.agents
        ]

    # Estimate cost
    total_cost = sum(CostEstimate.estimate(t["model"], 500, 1500).cost_usd for t in task_calls)

    return {
        "pattern": pattern,
        "agents": len(task_calls),
        "task_calls": task_calls,
        "estimated_cost": total_cost,
    }


def should_auto_swarm(task: str) -> bool:
    """Check if a task should trigger automatic agent swarm.

    Returns True for tasks that benefit from parallel agents:
    - Search/find tasks
    - Review/audit tasks
    - Research tasks
    - Multi-file analysis
    """
    task_lower = task.lower()
    triggers = [
        "find all",
        "search for",
        "where is",
        "locate",
        "review",
        "audit",
        "check all",
        "analyze",
        "research",
        "investigate",
        "explore",
        "scan",
        "grep",
        "look for",
    ]
    return any(t in task_lower for t in triggers)


def format_task_calls(task_calls: list[dict[str, Any]]) -> str:
    """Format task calls for display or execution.

    Returns a string representation of Task tool calls.
    """
    output = []
    for tc in task_calls:
        output.append(f"""Task(
    subagent_type="{tc['subagent_type']}",
    model="{tc['model']}",
    description="{tc['description']}",
    prompt=\"\"\"{tc['prompt'][:200]}...\"\"\"
)""")
    return "\n\n".join(output)


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()

    if command == "help":
        print_help()
    elif command == "list":
        list_workflows()
        list_agents()
    elif command == "workflows":
        list_workflows()
    elif command == "agents":
        list_agents()
    elif command == "status":
        show_status()
    elif command == "generate":
        if len(sys.argv) < 3:
            print("Usage: generate <workflow_name>")
            return
        context = {}
        if len(sys.argv) > 3:
            # Parse key=value pairs
            for arg in sys.argv[3:]:
                if "=" in arg:
                    k, v = arg.split("=", 1)
                    context[k] = v
        generate_workflow(sys.argv[2], context)
    elif command == "auto":
        if len(sys.argv) < 3:
            print('Usage: auto "<task description>"')
            return
        auto_select(" ".join(sys.argv[2:]))
    elif command == "run":
        if len(sys.argv) < 3:
            print("Usage: run <workflow_name> [key=value ...]")
            return
        # Parse context
        context = {}
        for arg in sys.argv[3:]:
            if "=" in arg:
                k, v = arg.split("=", 1)
                context[k] = v
        generate_workflow(sys.argv[2], context)
    elif command == "trace":
        show_traces(sys.argv[2:])
    elif command == "ric-phase":
        show_ric_phase()
    elif command == "circuit-breaker" or command == "cb":
        cb = AgentCircuitBreaker()
        if len(sys.argv) > 2 and sys.argv[2] == "reset":
            agent = sys.argv[3] if len(sys.argv) > 3 else None
            cb.reset(agent)
            print(f"Circuit breaker reset for: {agent or 'all agents'}")
        else:
            status = cb.get_status()
            if status:
                print("\n=== CIRCUIT BREAKER STATUS ===\n")
                print("| Agent | Status |")
                print("|-------|--------|")
                for agent, st in status.items():
                    print(f"| {agent} | {st} |")
            else:
                print("\nAll circuits closed (no failures recorded)")
    else:
        print(f"Unknown command: {command}")
        print_help()


if __name__ == "__main__":
    main()
