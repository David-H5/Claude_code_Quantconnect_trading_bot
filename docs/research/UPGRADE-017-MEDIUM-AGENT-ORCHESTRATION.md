# UPGRADE-017-MEDIUM: Agent Orchestration Enhancement (Practical Edition)

## Overview

**Date**: December 5, 2025
**Status**: COMPLETE
**Priority**: P0 (Critical for autonomous development)
**Estimated Effort**: 8-10 hours (overnight session)
**Actual Effort**: ~4 hours

### Summary

Practical enhancement of the agent orchestration suite combining:
- **Quick haiku agents** for file search, web research, text extraction
- **Reliability features** (retry, fallback, circuit breaker, graceful degradation)
- **RIC workflow integration** with phase-specific agent recommendations
- **Cost tracking** and model optimization (Haiku-first policy)
- **Auto-persistence** for research findings

This is a merged "best of both" from UPGRADE-017 (infrastructure) and UPGRADE-018 (user needs), skipping over-engineered features.

---

## What's Included vs Skipped

### ✅ Included (High Value, Practical)

| Source | Feature | Tasks | Value |
|--------|---------|-------|-------|
| 018 | Quick Haiku Agents | 5 | User's explicit request |
| 018 | RIC Integration | 4 | Workflow connection |
| 018 | Auto-Persistence | 3 | Save research automatically |
| 018 | Cost Tracking | 4 | Monitor spend |
| 017 | Retry + Backoff | 4 | Reliability |
| 017 | Fallback Routing | 4 | Graceful failures |
| 017 | Graceful Degradation | 3 | Partial results work |
| 017 | Agent Circuit Breaker | 4 | Prevent runaway failures |
| 017 | Execution Tracing | 4 | Debug visibility |
| 017 | Token Tracking | 3 | Cost visibility |
| **Total** | | **38** | |

### ❌ Skipped (Over-Engineering)

| Feature | Why Skip |
|---------|----------|
| Swarm topologies (mesh/ring/star) | Parallel works fine |
| Plugin 3-tier architecture | Complexity without benefit |
| Agent-to-agent communication | Not needed yet |
| Memory persistence classes | Session notes file works |
| Checkpoint/resume system | Git checkpointing exists |
| Working/Session memory | YAGNI |
| Conditional branching | YAGNI |
| Cyclic workflows | YAGNI |
| Scatter-gather pattern | YAGNI |
| Human escalation system | Can add later |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Agent Orchestrator v1.5 (UPGRADE-017-MEDIUM)          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Layer 1: Quick Agents (NEW)                                            │
│  ├── web_researcher (haiku)     - Web search + auto-save to docs        │
│  ├── text_extractor (haiku)     - Extract/copy text from files          │
│  ├── grep_agent (haiku)         - Fast pattern search                   │
│  ├── file_lister (haiku)        - List files by pattern                 │
│  └── research_saver (haiku)     - Save findings to docs/research/       │
│                                                                          │
│  Layer 2: Reliability (NEW)                                             │
│  ├── RetryConfig                - Exponential backoff with jitter       │
│  ├── FallbackRouter             - Route to backup agents on failure     │
│  ├── AgentCircuitBreaker        - Prevent cascading failures            │
│  └── PartialResult              - Return usable results on partial fail │
│                                                                          │
│  Layer 3: RIC Integration (NEW)                                         │
│  ├── ric_research workflow      - P0 Research phase agents              │
│  ├── ric_verify workflow        - P3 Verify phase agents                │
│  └── /ric-agents command        - Suggest agents per RIC phase          │
│                                                                          │
│  Layer 4: Observability (NEW)                                           │
│  ├── CostTracker                - Estimate and track costs              │
│  ├── TokenTracker               - Track token usage by model            │
│  └── ExecutionTracer            - Trace workflow execution              │
│                                                                          │
│  Layer 5: Auto-Persistence (NEW)                                        │
│  └── ResearchPersister          - Auto-save web research to docs/       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Checklist

### Phase 1: Quick Haiku Agents (5 tasks) - ~1 hour

Add to `AGENT_TEMPLATES` in `.claude/hooks/agent_orchestrator.py`:

- [ ] **1.1** Add `web_researcher` agent template
  ```python
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
  ```

- [ ] **1.2** Add `text_extractor` agent template
  ```python
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
  ```

- [ ] **1.3** Add `grep_agent` agent template
  ```python
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
  ```

- [ ] **1.4** Add `file_lister` agent template
  ```python
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
  ```

- [ ] **1.5** Add `research_saver` agent template
  ```python
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

---

### Phase 2: Reliability - Retry & Fallback (8 tasks) - ~2 hours

- [ ] **2.1** Create `RetryConfig` dataclass
  ```python
  @dataclass
  class RetryConfig:
      max_retries: int = 3
      initial_delay_ms: int = 1000
      max_delay_ms: int = 30000
      exponential_base: float = 2.0
      jitter_pct: float = 0.25  # ±25% randomization
      retryable_errors: List[str] = field(default_factory=lambda: [
          "timeout", "rate_limit", "server_error", "connection_error"
      ])
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **2.2** Implement `calculate_backoff()` function
  ```python
  def calculate_backoff(attempt: int, config: RetryConfig) -> int:
      """Calculate delay with exponential backoff and jitter."""
      import random
      delay = min(
          config.initial_delay_ms * (config.exponential_base ** attempt),
          config.max_delay_ms
      )
      jitter = delay * config.jitter_pct * (2 * random.random() - 1)
      return int(delay + jitter)
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **2.3** Create `RetryableAgent` wrapper
  ```python
  class RetryableAgent:
      def __init__(self, agent: AgentSpec, config: RetryConfig = None):
          self.agent = agent
          self.config = config or RetryConfig()
          self.attempts = 0
          self.last_error = None

      def should_retry(self, error: str) -> bool:
          return (
              self.attempts < self.config.max_retries and
              any(e in error.lower() for e in self.config.retryable_errors)
          )

      def get_retry_prompt(self) -> str:
          """Add retry context to prompt."""
          if self.attempts == 0:
              return self.agent.prompt_template
          return f"""RETRY ATTEMPT {self.attempts + 1}/{self.config.max_retries}
  Previous error: {self.last_error}

  {self.agent.prompt_template}"""
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **2.4** Define fallback mappings in config
  ```json
  {
    "fallbacks": {
      "security_scanner": ["type_checker", "code_finder"],
      "architect": ["senior_reviewer", "implementer"],
      "risk_reviewer": ["execution_reviewer", "backtest_reviewer"],
      "deep_architect": ["architect", "implementer"],
      "web_researcher": ["doc_finder", "code_finder"]
    }
  }
  ```
  **File**: `.claude/agent_config.json`

- [ ] **2.5** Create `FallbackRouter` class
  ```python
  class FallbackRouter:
      def __init__(self, config_path: Path = None):
          self.config_path = config_path or PROJECT_ROOT / ".claude" / "agent_config.json"
          self._load_config()

      def _load_config(self):
          if self.config_path.exists():
              data = json.loads(self.config_path.read_text())
              self.fallback_chains = data.get("fallbacks", {})
          else:
              self.fallback_chains = {}

      def get_fallback(self, failed_agent: str, attempt: int) -> Optional[str]:
          """Get next fallback agent for a failed agent."""
          chain = self.fallback_chains.get(failed_agent, [])
          if attempt < len(chain):
              return chain[attempt]
          return None

      def has_fallback(self, agent: str) -> bool:
          return agent in self.fallback_chains
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **2.6** Add generic fallback agents
  ```python
  FALLBACK_AGENTS = {
      "generic_searcher": AgentSpec(
          name="GenericSearcher",
          role="Fallback search agent",
          model=Model.HAIKU,
          agent_type=AgentType.EXPLORE,
          prompt_template="Search for: {query}\nReturn any relevant findings.",
          tags=["fallback", "search"],
      ),
      "generic_reviewer": AgentSpec(
          name="GenericReviewer",
          role="Fallback review agent",
          model=Model.SONNET,
          agent_type=AgentType.PLAN,
          prompt_template="Review: {target}\nProvide general feedback.",
          tags=["fallback", "review"],
      ),
  }
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **2.7** Add timeout configuration to agent_config.json
  ```json
  {
    "timeouts": {
      "default_ms": 120000,
      "by_model": {
        "haiku": 60000,
        "sonnet": 180000,
        "opus": 300000
      },
      "by_type": {
        "search": 60000,
        "review": 180000,
        "implementation": 300000
      }
    }
  }
  ```
  **File**: `.claude/agent_config.json`

- [ ] **2.8** Implement `get_timeout()` function
  ```python
  def get_timeout(agent: AgentSpec) -> int:
      """Select appropriate timeout based on agent type and model."""
      config = load_agent_config()
      timeouts = config.get("timeouts", {})

      model_timeout = timeouts.get("by_model", {}).get(agent.model.value)
      type_timeout = None
      if agent.tags:
          type_timeout = timeouts.get("by_type", {}).get(agent.tags[0])

      candidates = [t for t in [model_timeout, type_timeout] if t]
      return min(candidates) if candidates else timeouts.get("default_ms", 120000)
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

---

### Phase 3: Reliability - Circuit Breaker & Degradation (7 tasks) - ~1.5 hours

- [ ] **3.1** Create `AgentCircuitBreaker` class
  ```python
  class AgentCircuitBreaker:
      """Prevent cascading failures by opening circuit after repeated failures."""

      def __init__(self, failure_threshold: int = 3, reset_timeout_s: int = 300):
          self.failure_threshold = failure_threshold
          self.reset_timeout_s = reset_timeout_s
          self.failures: Dict[str, int] = {}
          self.open_circuits: Dict[str, datetime] = {}

      def record_failure(self, agent_name: str):
          self.failures[agent_name] = self.failures.get(agent_name, 0) + 1
          if self.failures[agent_name] >= self.failure_threshold:
              self.open_circuits[agent_name] = datetime.now()

      def is_open(self, agent_name: str) -> bool:
          if agent_name not in self.open_circuits:
              return False
          elapsed = (datetime.now() - self.open_circuits[agent_name]).seconds
          if elapsed >= self.reset_timeout_s:
              # Half-open: allow one attempt
              del self.open_circuits[agent_name]
              return False
          return True

      def record_success(self, agent_name: str):
          self.failures[agent_name] = 0
          if agent_name in self.open_circuits:
              del self.open_circuits[agent_name]

      def get_status(self) -> Dict[str, str]:
          """Get circuit breaker status for all agents."""
          status = {}
          for agent in set(list(self.failures.keys()) + list(self.open_circuits.keys())):
              if self.is_open(agent):
                  status[agent] = "OPEN"
              elif self.failures.get(agent, 0) > 0:
                  status[agent] = f"HALF-OPEN ({self.failures[agent]}/{self.failure_threshold})"
              else:
                  status[agent] = "CLOSED"
          return status
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **3.2** Add circuit breaker state persistence
  ```python
  def save_circuit_state(self):
      """Save circuit breaker state to file."""
      state_file = PROJECT_ROOT / ".claude" / "agent_state.json"
      state = {
          "circuit_breakers": {
              agent: {
                  "failures": self.failures.get(agent, 0),
                  "open_since": self.open_circuits.get(agent, "").isoformat() if agent in self.open_circuits else None
              }
              for agent in set(list(self.failures.keys()) + list(self.open_circuits.keys()))
          }
      }
      state_file.write_text(json.dumps(state, indent=2))

  def load_circuit_state(self):
      """Load circuit breaker state from file."""
      state_file = PROJECT_ROOT / ".claude" / "agent_state.json"
      if state_file.exists():
          state = json.loads(state_file.read_text())
          for agent, data in state.get("circuit_breakers", {}).items():
              self.failures[agent] = data.get("failures", 0)
              if data.get("open_since"):
                  self.open_circuits[agent] = datetime.fromisoformat(data["open_since"])
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **3.3** Create `PartialResult` dataclass
  ```python
  @dataclass
  class PartialResult:
      """Result that may be partial due to some agent failures."""
      workflow_name: str
      total_agents: int
      completed_agents: int
      failed_agents: int
      results: Dict[str, AgentResult]
      failures: Dict[str, str]
      degradation_level: str  # "full", "partial", "minimal"
      usable: bool  # Whether results are actionable

      @property
      def success_rate(self) -> float:
          if self.total_agents == 0:
              return 0.0
          return self.completed_agents / self.total_agents
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **3.4** Implement `aggregate_partial_results()` function
  ```python
  def aggregate_partial_results(
      workflow_name: str,
      results: List[AgentResult]
  ) -> PartialResult:
      """Aggregate results even with partial failures."""
      successful = [r for r in results if r.success]
      failed = [r for r in results if not r.success]

      # Determine degradation level
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
          failures={r.agent_name: r.error for r in failed},
          degradation_level=level,
          usable=success_rate >= 0.25,  # At least 25% success = usable
      )
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **3.5** Add degradation indicators to output format
  ```python
  def format_partial_result(result: PartialResult) -> str:
      """Format partial result for display."""
      status_icon = {"full": "✅", "partial": "⚠️", "minimal": "❌"}.get(result.degradation_level, "?")

      output = f"""## Results ({result.degradation_level.title()} - {result.completed_agents}/{result.total_agents} agents)
  {status_icon} Degradation Level: {result.degradation_level}
  """

      if result.results:
          output += f"✅ Completed: {', '.join(result.results.keys())}\n"

      if result.failures:
          output += f"❌ Failed: {', '.join(f'{k} ({v})' for k, v in result.failures.items())}\n"

      if not result.usable:
          output += "\n⚠️ WARNING: Results may not be usable (success rate < 25%)\n"

      return output
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **3.6** Integrate circuit breaker into agent selection
  ```python
  def select_agent_with_circuit_breaker(
      requested_agent: str,
      circuit_breaker: AgentCircuitBreaker,
      fallback_router: FallbackRouter
  ) -> str:
      """Select agent, falling back if circuit is open."""
      if not circuit_breaker.is_open(requested_agent):
          return requested_agent

      # Circuit is open, try fallbacks
      for attempt in range(3):
          fallback = fallback_router.get_fallback(requested_agent, attempt)
          if fallback and not circuit_breaker.is_open(fallback):
              return fallback

      # All fallbacks exhausted or open
      return None
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **3.7** Add circuit breaker status to `/agent-status` command
  ```markdown
  ## Circuit Breaker Status
  | Agent | Status | Failures |
  |-------|--------|----------|
  | {agent} | {status} | {failures}/{threshold} |
  ```
  **File**: `.claude/commands/agent-status.md`

---

### Phase 4: RIC Integration (4 tasks) - ~1 hour

- [ ] **4.1** Add `ric_research` workflow template
  ```python
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
              tags=["research", "web"],
          ),
          AgentSpec(
              name="CodebaseSearcher",
              role="Find existing implementations",
              model=Model.HAIKU,
              agent_type=AgentType.EXPLORE,
              prompt_template="Find existing code for: {topic}. Return file:line references.",
              tags=["search", "code"],
          ),
          AgentSpec(
              name="DocSearcher",
              role="Search project documentation",
              model=Model.HAIKU,
              agent_type=AgentType.EXPLORE,
              prompt_template="Find docs about: {topic} in docs/, CLAUDE.md, README files.",
              tags=["search", "docs"],
          ),
      ],
      tags=["ric", "research", "p0"],
  ),
  ```
  **File**: `.claude/hooks/agent_orchestrator.py` (add to WORKFLOW_TEMPLATES)

- [ ] **4.2** Add `ric_verify` workflow template
  ```python
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
  **File**: `.claude/hooks/agent_orchestrator.py` (add to WORKFLOW_TEMPLATES)

- [ ] **4.3** Create `/ric-agents` command
  ```markdown
  # RIC Agents - Phase-Specific Agent Recommendations

  Suggest optimal agents for the current RIC phase.

  ## Usage
  /ric-agents [phase]

  ## Phase Recommendations

  ### P0 RESEARCH
  - `/agent-swarm {topic}` - 8 haiku agents explore codebase
  - `web_researcher` (haiku) - Search web, format for docs
  - `doc_finder` (haiku) - Find existing documentation
  - `research_saver` (haiku) - Save findings to docs/research/

  ### P1 PLAN
  - `architect` (sonnet) - Design implementation plan

  ### P2 BUILD
  - `implementer` (sonnet) - Write code
  - `refactorer` (sonnet) - Improve code quality

  ### P3 VERIFY
  - `/parallel-review` - 4 haiku agents check code
  - `test_analyzer` (haiku) - Check test coverage
  - `type_checker` (haiku) - Validate types
  - `security_scanner` (haiku) - Security vulnerabilities

  ### P4 REFLECT
  - `/agent-consensus` - 3 sonnet agents vote on quality
  - `deep_architect` (opus) - Only if complex decisions needed
  ```
  **File**: `.claude/commands/ric-agents.md` (new file)

- [ ] **4.4** Add RIC phase detection to auto-routing
  ```python
  def detect_ric_phase(context: Dict[str, Any]) -> Optional[str]:
      """Detect current RIC phase from context."""
      # Check progress file for current phase
      progress_file = PROJECT_ROOT / "claude-progress.txt"
      if progress_file.exists():
          content = progress_file.read_text()
          phase_match = re.search(r"\[P(\d)\]", content)
          if phase_match:
              phase_num = int(phase_match.group(1))
              return ["RESEARCH", "PLAN", "BUILD", "VERIFY", "REFLECT"][phase_num]
      return None

  def get_ric_recommended_agents(phase: str) -> List[str]:
      """Get recommended agents for a RIC phase."""
      recommendations = {
          "RESEARCH": ["web_researcher", "doc_finder", "code_finder"],
          "PLAN": ["architect"],
          "BUILD": ["implementer", "refactorer"],
          "VERIFY": ["test_analyzer", "type_checker", "security_scanner"],
          "REFLECT": ["deep_architect"],
      }
      return recommendations.get(phase, [])
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

---

### Phase 5: Cost & Token Tracking (4 tasks) - ~1 hour

- [ ] **5.1** Create `CostEstimate` dataclass
  ```python
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
      def estimate(cls, model: str, prompt_len: int = 1000,
                   output_len: int = 2000) -> "CostEstimate":
          """Estimate cost based on model and token counts."""
          rates = cls.COSTS.get(model, cls.COSTS["sonnet"])
          cost = (prompt_len * rates["input"] +
                  output_len * rates["output"]) / 1_000_000

          return cls(
              model=model,
              input_tokens=prompt_len,
              output_tokens=output_len,
              cost_usd=cost,
          )
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **5.2** Create `TokenTracker` class
  ```python
  class TokenTracker:
      """Track token usage across agents and models."""

      def __init__(self):
          self.usage: Dict[str, Dict[str, int]] = {}  # model -> {input, output}
          self.by_agent: Dict[str, Dict[str, int]] = {}  # agent -> {input, output}

      def record(self, agent_name: str, model: str,
                 tokens_in: int, tokens_out: int):
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
          """Calculate total cost based on model pricing."""
          total = 0.0
          for model, tokens in self.usage.items():
              rates = CostEstimate.COSTS.get(model, CostEstimate.COSTS["sonnet"])
              total += (tokens["input"] * rates["input"] +
                        tokens["output"] * rates["output"]) / 1_000_000
          return total

      def get_report(self) -> str:
          """Generate token usage report."""
          lines = ["## Token Usage Report", ""]
          lines.append("### By Model")
          lines.append("| Model | Input | Output | Est. Cost |")
          lines.append("|-------|-------|--------|-----------|")

          for model, tokens in self.usage.items():
              rates = CostEstimate.COSTS.get(model, CostEstimate.COSTS["sonnet"])
              cost = (tokens["input"] * rates["input"] +
                      tokens["output"] * rates["output"]) / 1_000_000
              lines.append(f"| {model} | {tokens['input']:,} | {tokens['output']:,} | ${cost:.4f} |")

          lines.append(f"\n**Total Estimated Cost**: ${self.get_total_cost():.4f}")
          return "\n".join(lines)
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **5.3** Add cost estimation to workflow generation
  ```python
  def estimate_workflow_cost(workflow: WorkflowSpec) -> CostEstimate:
      """Estimate total cost for a workflow."""
      total_input = 0
      total_output = 0

      for agent in workflow.agents:
          # Estimate based on prompt template length
          prompt_tokens = len(agent.prompt_template) // 4  # ~4 chars per token
          output_tokens = 2000  # Default estimate

          # Scale by model
          if agent.model == Model.OPUS:
              output_tokens *= 2  # Opus tends to be more verbose

          total_input += prompt_tokens
          total_output += output_tokens

      # Use average model for cost calculation
      avg_model = workflow.agents[0].model.value if workflow.agents else "sonnet"
      return CostEstimate.estimate(avg_model, total_input, total_output)
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **5.4** Add cost summary to workflow output
  ```python
  def format_workflow_summary(
      workflow_name: str,
      results: List[AgentResult],
      cost: CostEstimate
  ) -> str:
      """Format workflow summary with cost information."""
      return f"""
  ## Workflow Complete: {workflow_name}

  ### Agents Run
  {len(results)} agents executed

  ### Cost Summary
  - Model: {cost.model}
  - Input tokens: ~{cost.input_tokens:,}
  - Output tokens: ~{cost.output_tokens:,}
  - **Estimated cost**: ${cost.cost_usd:.4f}

  ### Results
  {chr(10).join(f'- {r.agent_name}: {"✅" if r.success else "❌"}' for r in results)}
  """
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

---

### Phase 6: Execution Tracing (4 tasks) - ~1 hour

- [ ] **6.1** Create `TraceSpan` and `ExecutionTrace` dataclasses
  ```python
  @dataclass
  class TraceSpan:
      span_id: str
      agent_name: str
      started_at: datetime
      ended_at: Optional[datetime] = None
      status: str = "running"  # "running", "success", "failed", "timeout"
      duration_ms: Optional[float] = None
      tokens_in: Optional[int] = None
      tokens_out: Optional[int] = None
      error: Optional[str] = None

  @dataclass
  class ExecutionTrace:
      trace_id: str
      workflow_name: str
      started_at: datetime
      ended_at: Optional[datetime] = None
      spans: List[TraceSpan] = field(default_factory=list)

      @property
      def duration_ms(self) -> Optional[float]:
          if self.ended_at:
              return (self.ended_at - self.started_at).total_seconds() * 1000
          return None
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **6.2** Create `Tracer` class
  ```python
  class Tracer:
      """Trace workflow execution for debugging and observability."""

      def __init__(self, storage_dir: Path = None):
          self.storage_dir = storage_dir or PROJECT_ROOT / ".claude" / "traces"
          self.storage_dir.mkdir(exist_ok=True)
          self.current_trace: Optional[ExecutionTrace] = None

      def start_trace(self, workflow_name: str) -> str:
          trace_id = f"{workflow_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
          self.current_trace = ExecutionTrace(
              trace_id=trace_id,
              workflow_name=workflow_name,
              started_at=datetime.now(),
          )
          return trace_id

      def start_span(self, agent_name: str) -> str:
          span_id = f"{agent_name}_{len(self.current_trace.spans)}"
          span = TraceSpan(
              span_id=span_id,
              agent_name=agent_name,
              started_at=datetime.now(),
          )
          self.current_trace.spans.append(span)
          return span_id

      def end_span(self, span_id: str, status: str,
                   tokens_in: int = 0, tokens_out: int = 0,
                   error: str = None):
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
          if self.current_trace:
              self.current_trace.ended_at = datetime.now()
              self.save_trace()

      def save_trace(self):
          if self.current_trace:
              filepath = self.storage_dir / f"{self.current_trace.trace_id}.json"
              filepath.write_text(json.dumps(asdict(self.current_trace), default=str, indent=2))
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **6.3** Add trace viewing CLI command
  ```python
  def cli_trace(args: List[str]):
      """View execution traces."""
      traces_dir = PROJECT_ROOT / ".claude" / "traces"

      if not traces_dir.exists():
          print("No traces found")
          return

      if args and args[0] != "--recent":
          # View specific trace
          trace_file = traces_dir / f"{args[0]}.json"
          if trace_file.exists():
              trace = json.loads(trace_file.read_text())
              print(format_trace(trace))
          else:
              print(f"Trace not found: {args[0]}")
      else:
          # List recent traces
          limit = int(args[1]) if len(args) > 1 else 5
          traces = sorted(traces_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

          print(f"\n## Recent Traces (last {limit})\n")
          for trace_file in traces[:limit]:
              trace = json.loads(trace_file.read_text())
              status = "✅" if all(s["status"] == "success" for s in trace["spans"]) else "❌"
              duration = trace.get("duration_ms", 0) or 0
              print(f"{status} {trace['trace_id']} - {len(trace['spans'])} agents - {duration:.0f}ms")
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **6.4** Create `/agent-trace` command
  ```markdown
  # Agent Trace - View Execution Traces

  View detailed execution traces for debugging agent workflows.

  ## Usage
  /agent-trace [trace_id]
  /agent-trace --recent [count]

  ## Examples
  ```bash
  # List recent traces
  /agent-trace --recent 5

  # View specific trace
  /agent-trace code_review_20251205_143022
  ```

  ## Instructions
  Run: `python3 .claude/hooks/agent_orchestrator.py trace $ARGUMENTS`
  ```
  **File**: `.claude/commands/agent-trace.md` (new file)

---

### Phase 7: Auto-Persistence (3 tasks) - ~30 min

- [ ] **7.1** Create `ResearchPersister` class
  ```python
  class ResearchPersister:
      """Auto-save research findings to docs/research/."""

      RESEARCH_DIR = PROJECT_ROOT / "docs" / "research"

      @classmethod
      def save_entry(cls, query: str, findings: str,
                     sources: List[str] = None,
                     doc_name: str = None) -> bool:
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
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **7.2** Add `save_web_research()` convenience method
  ```python
  def save_web_research(query: str, findings: str,
                        sources: List[str] = None,
                        upgrade: str = None) -> bool:
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

      return ResearchPersister.save_entry(
          query=query,
          findings=findings,
          sources=sources,
          doc_name=doc_name
      )
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **7.3** Integrate with web_researcher agent output
  ```python
  def process_web_researcher_result(result: AgentResult) -> bool:
      """Process web researcher result and auto-save if successful."""
      if not result.success:
          return False

      # Parse result for query, findings, sources
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
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

---

## Model Selection Policy

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
| Retry success rate | >80% recoveries | Circuit breaker logs |

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `.claude/commands/ric-agents.md` | RIC phase agent recommendations |
| `.claude/commands/agent-trace.md` | Trace viewing command |
| `.claude/traces/` | Execution trace storage |

### Modified Files

| File | Changes |
|------|---------|
| `.claude/hooks/agent_orchestrator.py` | All new classes and functions |
| `.claude/agent_config.json` | Fallbacks, timeouts config |
| `.claude/agent_state.json` | Circuit breaker state |
| `.claude/commands/agent-status.md` | Add circuit breaker status |

---

## Testing Checklist

- [ ] Quick agents spawn correctly with haiku model
- [ ] Retry logic triggers on timeout/error
- [ ] Fallback routing activates when primary agent fails
- [ ] Circuit breaker opens after threshold failures
- [ ] Partial results aggregate correctly
- [ ] Cost tracking produces accurate estimates
- [ ] Execution traces saved and readable
- [ ] Research auto-persists to docs/research/
- [ ] RIC workflows map to correct agents

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-05 | Created merged UPGRADE-017-MEDIUM | Claude |
| 2025-12-05 | Combined best of UPGRADE-017 and UPGRADE-018 | Claude |
