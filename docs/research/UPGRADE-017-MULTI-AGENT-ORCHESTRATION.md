# UPGRADE-017: Multi-Agent Orchestration Enhancement

## 📋 Overview

**Version**: 1.0.0
**Created**: December 4, 2025
**Status**: Planning
**Estimated Effort**: 4-6 weeks
**Priority**: P0 (Critical for autonomous development)

This upgrade guide provides a comprehensive roadmap for enhancing the multi-agent orchestration suite based on research from Anthropic, Claude-Flow, LangGraph, wshobson/agents, and industry best practices.

---

## 🎯 Objectives

1. **Reliability**: Implement error handling, retries, fallbacks, and circuit breakers
2. **Intelligence**: Add memory persistence, context sharing, and smart delegation
3. **Scalability**: Support swarm topologies, dynamic agent spawning, and parallel patterns
4. **Observability**: Add tracing, token tracking, and performance metrics
5. **Extensibility**: Plugin architecture with three-tier loading

---

## 📚 Reference Frameworks

### Framework 1: Claude-Flow (ruvnet)

**Repository**: https://github.com/ruvnet/claude-flow
**Key Features**: 64-agent swarm, MCP integration, multiple topologies

```bash
# Installation
npx claude-flow@alpha init --force
claude mcp add claude-flow
npx claude-flow@alpha mcp start

# Swarm initialization
npx claude-flow@alpha swarm init --topology mesh --max-agents 5
npx claude-flow@alpha swarm spawn researcher "analyze API patterns"
```

**Architecture Insights**:
- Supports hierarchical, mesh, ring, star topologies
- Stream-json chaining for real-time agent-to-agent communication
- 87 MCP tools in `mcp__claude-flow__` namespace
- Agent types: coordinator, researcher, coder, analyst, architect, tester, reviewer

### Framework 2: wshobson/agents

**Repository**: https://github.com/wshobson/agents
**Key Features**: 85 agents, 63 plugins, three-tier architecture

**Three-Tier Loading Pattern**:
```
Tier 1: Metadata (always loaded)
  - Name, activation criteria
  - ~50 tokens per plugin

Tier 2: Instructions (loaded when activated)
  - Core guidance, commands
  - ~500 tokens per plugin

Tier 3: Resources (loaded on demand)
  - Examples, templates, detailed docs
  - ~2000+ tokens per plugin
```

**Plugin Categories**:
- `backend-development` - API design, TDD
- `frontend-mobile-development` - React/React Native
- `full-stack-orchestration` - Multi-agent coordination
- `security-compliance` - OWASP, vulnerability scanning
- `data-ai-ml` - ML pipelines, model training

### Framework 3: LangGraph

**Repository**: https://github.com/langchain-ai/langgraph
**Key Features**: State machines, checkpointing, conditional edges

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Define state
class AgentState(TypedDict):
    messages: list
    current_agent: str
    task_status: str

# Build graph
graph = StateGraph(AgentState)
graph.add_node("researcher", researcher_node)
graph.add_node("implementer", implementer_node)
graph.add_node("reviewer", reviewer_node)

# Conditional routing
graph.add_conditional_edges(
    "researcher",
    route_after_research,
    {"implement": "implementer", "done": END}
)

# Compile with checkpointing
app = graph.compile(checkpointer=MemorySaver())
```

### Framework 4: Anthropic Multi-Agent Research System

**Source**: https://www.anthropic.com/engineering/multi-agent-research-system

**Key Patterns**:
1. **Orchestrator-Worker**: Lead agent coordinates, spawns subagents
2. **Scaling Rules**: Embed complexity-based effort rules in prompts
3. **Task Boundaries**: Clear objective, output format, tools, scope per agent

---

## 🏗️ Architecture Enhancements

### Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Current Agent Orchestrator                       │
├─────────────────────────────────────────────────────────────────┤
│  /agents command → ModelSelector → AgentTemplates → Task calls  │
│                                                                  │
│  Limitations:                                                    │
│  - No error handling/retry                                       │
│  - No memory persistence                                         │
│  - No agent-to-agent communication                               │
│  - No checkpointing                                              │
│  - Single topology (parallel only)                               │
└─────────────────────────────────────────────────────────────────┘
```

### Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Enhanced Agent Orchestrator v2.0                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Command    │  │  Orchestrator │  │    Memory    │  │   Plugin     │ │
│  │   Router     │──│    Engine     │──│    Manager   │──│   Loader     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │
│         │                 │                 │                 │          │
│         │          ┌──────┴──────┐          │                 │          │
│         │          │             │          │                 │          │
│         ▼          ▼             ▼          ▼                 ▼          │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────┐ │
│  │   Model    │ │  Workflow  │ │   Error    │ │  Context   │ │ Agent  │ │
│  │  Selector  │ │  Executor  │ │  Handler   │ │  Sharing   │ │Registry│ │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────┘ │
│         │              │              │              │              │    │
│         │              │              │              │              │    │
│         ▼              ▼              ▼              ▼              ▼    │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                         Agent Execution Layer                        ││
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       ││
│  │  │ Agent 1 │ │ Agent 2 │ │ Agent 3 │ │ Agent N │ │Fallback │       ││
│  │  │ (haiku) │ │ (haiku) │ │(sonnet) │ │ (opus)  │ │ Agent   │       ││
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                         Persistence Layer                            ││
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   ││
│  │  │   Working   │ │  Session    │ │  Long-term  │ │ Checkpoint  │   ││
│  │  │   Memory    │ │  Memory     │ │  Memory     │ │   Store     │   ││
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✅ Implementation Checklist

### Phase 1: Error Handling & Reliability (Week 1)

#### 1.1 Structured Handoff Schema

**Insight**: "Most agent failures are orchestration and context-transfer issues" - Skywork AI

- [ ] **1.1.1** Create `AgentHandoff` dataclass with required fields
  ```python
  @dataclass
  class AgentHandoff:
      task_id: str
      objective: str  # Clear goal
      output_format: str  # Expected output structure
      tools_allowed: List[str]  # Scoped tool access
      context: Dict[str, Any]  # Relevant context only
      boundaries: TaskBoundaries  # What to do/not do
      timeout_ms: int
      retry_config: RetryConfig
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`
  **Lines**: Add after `AgentSpec` class (~line 100)

- [ ] **1.1.2** Create `TaskBoundaries` dataclass
  ```python
  @dataclass
  class TaskBoundaries:
      must_do: List[str]  # Required actions
      must_not_do: List[str]  # Prohibited actions
      scope_includes: List[str]  # In scope
      scope_excludes: List[str]  # Out of scope
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **1.1.3** Create `OutputSchema` for each agent type
  ```python
  OUTPUT_SCHEMAS = {
      "search": {
          "type": "object",
          "required": ["files_found", "code_snippets", "summary"],
          "properties": {...}
      },
      "review": {
          "type": "object",
          "required": ["findings", "severity", "recommendations"],
          "properties": {...}
      }
  }
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **1.1.4** Update `AgentSpec.to_task_call()` to use handoff schema
  **File**: `.claude/hooks/agent_orchestrator.py`
  **Method**: `AgentSpec.to_task_call()`

- [ ] **1.1.5** Add handoff validation before agent spawn
  **File**: `.claude/hooks/agent_orchestrator.py`
  **New method**: `validate_handoff(handoff: AgentHandoff) -> ValidationResult`

#### 1.2 Per-Agent Timeout Handling

**Insight**: "With timeout controls, swarms can gracefully handle agent failures" - Claude-Flow

- [ ] **1.2.1** Add `TimeoutConfig` dataclass
  ```python
  @dataclass
  class TimeoutConfig:
      default_ms: int = 120000
      search_ms: int = 60000
      review_ms: int = 180000
      implementation_ms: int = 300000
      critical_ms: int = 600000
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **1.2.2** Add timeout configuration to `agent_config.json`
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

- [ ] **1.2.3** Implement timeout selection logic
  ```python
  def get_timeout(agent: AgentSpec) -> int:
      """Select appropriate timeout based on agent type and model."""
      config = load_config()
      model_timeout = config["timeouts"]["by_model"].get(agent.model.value)
      type_timeout = config["timeouts"]["by_type"].get(agent.tags[0] if agent.tags else None)
      return min(filter(None, [model_timeout, type_timeout, config["timeouts"]["default_ms"]]))
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **1.2.4** Add timeout to generated Task calls
  **File**: `.claude/hooks/agent_orchestrator.py`
  **Method**: `WorkflowGenerator.generate_parallel()`

#### 1.3 Retry with Exponential Backoff

**Insight**: "Implement retry or fallback just for that step, without restarting the entire process" - GoCodeo

- [ ] **1.3.1** Create `RetryConfig` dataclass
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

- [ ] **1.3.2** Implement `calculate_backoff()` function
  ```python
  def calculate_backoff(attempt: int, config: RetryConfig) -> int:
      """Calculate delay with exponential backoff and jitter."""
      delay = min(
          config.initial_delay_ms * (config.exponential_base ** attempt),
          config.max_delay_ms
      )
      jitter = delay * config.jitter_pct * (2 * random.random() - 1)
      return int(delay + jitter)
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **1.3.3** Create `RetryableAgent` wrapper class
  ```python
  class RetryableAgent:
      def __init__(self, agent: AgentSpec, config: RetryConfig):
          self.agent = agent
          self.config = config
          self.attempts = 0
          self.last_error = None

      def should_retry(self, error: str) -> bool:
          return (
              self.attempts < self.config.max_retries and
              any(e in error.lower() for e in self.config.retryable_errors)
          )
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **1.3.4** Add retry instructions to generated Task calls
  **File**: `.claude/commands/agents.md`
  **Section**: Add retry guidance in instructions

#### 1.4 Fallback Agent Routing

**Insight**: "Timeout-based fallbacks automatically route requests to backup agents" - AI Agent Frameworks

- [ ] **1.4.1** Define fallback mappings in config
  ```json
  {
    "fallbacks": {
      "security_scanner": ["type_checker", "code_finder"],
      "architect": ["senior_reviewer", "implementer"],
      "risk_reviewer": ["execution_reviewer", "backtest_reviewer"],
      "deep_architect": ["architect", "implementer"]
    }
  }
  ```
  **File**: `.claude/agent_config.json`

- [ ] **1.4.2** Create `FallbackRouter` class
  ```python
  class FallbackRouter:
      def __init__(self, config: Dict[str, List[str]]):
          self.fallback_chains = config

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

- [ ] **1.4.3** Integrate fallback routing into workflow executor
  **File**: `.claude/hooks/agent_orchestrator.py`
  **Method**: Update `WorkflowGenerator` methods

- [ ] **1.4.4** Add fallback agent templates
  ```python
  FALLBACK_AGENTS = {
      "generic_searcher": AgentSpec(
          name="GenericSearcher",
          role="Fallback search agent",
          model=Model.HAIKU,
          prompt_template="Search for: {query}\nReturn any relevant findings.",
      ),
      "generic_reviewer": AgentSpec(
          name="GenericReviewer",
          role="Fallback review agent",
          model=Model.SONNET,
          prompt_template="Review: {target}\nProvide general feedback.",
      ),
  }
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

#### 1.5 Graceful Degradation

**Insight**: "Return partial results on partial failure instead of all-or-nothing"

- [ ] **1.5.1** Create `PartialResult` dataclass
  ```python
  @dataclass
  class PartialResult:
      workflow_name: str
      total_agents: int
      completed_agents: int
      failed_agents: int
      results: Dict[str, AgentResult]
      failures: Dict[str, str]
      degradation_level: str  # "full", "partial", "minimal"
      usable: bool  # Whether results are actionable
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **1.5.2** Implement `aggregate_partial_results()` function
  ```python
  def aggregate_partial_results(results: List[AgentResult]) -> PartialResult:
      """Aggregate results even with partial failures."""
      successful = [r for r in results if r.success]
      failed = [r for r in results if not r.success]

      # Determine degradation level
      success_rate = len(successful) / len(results)
      if success_rate >= 0.75:
          level = "full"
      elif success_rate >= 0.5:
          level = "partial"
      else:
          level = "minimal"

      return PartialResult(
          completed_agents=len(successful),
          failed_agents=len(failed),
          degradation_level=level,
          usable=success_rate >= 0.25,  # At least 25% success
          ...
      )
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **1.5.3** Update result aggregation in commands
  **File**: `.claude/commands/agents.md`
  **Section**: Update "After Agents Complete" to handle partial results

- [ ] **1.5.4** Add degradation indicators to output
  ```markdown
  ## Results (Partial - 3/4 agents succeeded)
  ⚠️ Degradation Level: partial
  ✅ Completed: security_scanner, type_checker, test_analyzer
  ❌ Failed: architect (timeout)
  ```
  **File**: `.claude/commands/agents.md`

#### 1.6 Circuit Breaker Pattern

**Insight**: "Consider circuit breaker patterns for agent dependencies" - Azure Architecture

- [ ] **1.6.1** Create `CircuitBreaker` class
  ```python
  class AgentCircuitBreaker:
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
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **1.6.2** Add circuit breaker state to `agent_state.json`
  ```json
  {
    "circuit_breakers": {
      "architect": {"failures": 2, "last_failure": "2025-12-04T10:30:00"},
      "deep_architect": {"failures": 3, "open_since": "2025-12-04T10:35:00"}
    }
  }
  ```
  **File**: `.claude/agent_state.json`

- [ ] **1.6.3** Integrate circuit breaker into agent selection
  **File**: `.claude/hooks/agent_orchestrator.py`
  **Method**: Update `AutoAgentSelector.select_agents()`

- [ ] **1.6.4** Add circuit breaker status to `/agent-status`
  **File**: `.claude/commands/agent-status.md`

---

### Phase 2: Memory & State Management (Week 2)

#### 2.1 Working Memory

**Insight**: "Working memory holds short-lived, high-urgency context for the current task"

- [ ] **2.1.1** Create `WorkingMemory` class
  ```python
  class WorkingMemory:
      def __init__(self):
          self.current_task: Optional[str] = None
          self.active_agents: List[str] = []
          self.intermediate_results: Dict[str, Any] = {}
          self.context_stack: List[Dict[str, Any]] = []
          self.created_at: datetime = datetime.now()

      def push_context(self, context: Dict[str, Any]):
          self.context_stack.append(context)

      def pop_context(self) -> Optional[Dict[str, Any]]:
          return self.context_stack.pop() if self.context_stack else None

      def add_result(self, agent_name: str, result: Any):
          self.intermediate_results[agent_name] = result

      def get_relevant_context(self, agent_name: str) -> Dict[str, Any]:
          """Get context relevant to a specific agent."""
          # Filter intermediate results based on agent dependencies
          ...
  ```
  **File**: `.claude/hooks/agent_memory.py` (new file)

- [ ] **2.1.2** Create memory manager initialization
  ```python
  class MemoryManager:
      def __init__(self):
          self.working = WorkingMemory()
          self.session = SessionMemory()
          self.persistent = PersistentMemory()

      def start_workflow(self, task: str):
          self.working = WorkingMemory()
          self.working.current_task = task

      def end_workflow(self):
          # Promote important findings to session memory
          self.session.add_from_working(self.working)
          self.working = WorkingMemory()
  ```
  **File**: `.claude/hooks/agent_memory.py`

- [ ] **2.1.3** Integrate working memory into orchestrator
  **File**: `.claude/hooks/agent_orchestrator.py`
  **Import**: `from .agent_memory import MemoryManager`

- [ ] **2.1.4** Pass working memory context to agents
  **File**: `.claude/hooks/agent_orchestrator.py`
  **Method**: Update `AgentSpec.to_task_call()`

#### 2.2 Session Memory

**Insight**: "Short-term memory preserves conversation or session context across minutes or hours"

- [ ] **2.2.1** Create `SessionMemory` class
  ```python
  class SessionMemory:
      def __init__(self, max_entries: int = 100):
          self.max_entries = max_entries
          self.entries: List[MemoryEntry] = []
          self.session_id: str = str(uuid.uuid4())
          self.started_at: datetime = datetime.now()

      def add(self, entry: MemoryEntry):
          self.entries.append(entry)
          if len(self.entries) > self.max_entries:
              self._prune_old_entries()

      def search(self, query: str, limit: int = 5) -> List[MemoryEntry]:
          """Search session memory for relevant entries."""
          # Simple keyword matching (can be enhanced with embeddings)
          ...

      def get_summary(self) -> str:
          """Generate summary of session memory for context injection."""
          ...
  ```
  **File**: `.claude/hooks/agent_memory.py`

- [ ] **2.2.2** Create `MemoryEntry` dataclass
  ```python
  @dataclass
  class MemoryEntry:
      id: str
      timestamp: datetime
      entry_type: str  # "discovery", "decision", "result", "error"
      content: str
      agent_source: Optional[str] = None
      importance: float = 0.5  # 0-1 scale
      tags: List[str] = field(default_factory=list)
  ```
  **File**: `.claude/hooks/agent_memory.py`

- [ ] **2.2.3** Auto-capture important discoveries
  ```python
  def auto_capture(self, agent_result: AgentResult):
      """Automatically capture important findings from agent results."""
      if self._is_important(agent_result):
          self.add(MemoryEntry(
              entry_type="discovery",
              content=self._extract_key_findings(agent_result),
              agent_source=agent_result.agent_name,
              importance=self._calculate_importance(agent_result),
          ))
  ```
  **File**: `.claude/hooks/agent_memory.py`

- [ ] **2.2.4** Persist session memory to file
  **File**: `.claude/session_memory.json`
  **Method**: Add `save()` and `load()` methods to `SessionMemory`

#### 2.3 Cross-Agent Context Sharing

**Insight**: "If you're an agent-builder, ensure every action is informed by context from other parts of the system"

- [ ] **2.3.1** Create `SharedContext` class
  ```python
  class SharedContext:
      def __init__(self):
          self.global_context: Dict[str, Any] = {}
          self.agent_contexts: Dict[str, Dict[str, Any]] = {}
          self.locks: Dict[str, threading.Lock] = {}

      def set_global(self, key: str, value: Any):
          self.global_context[key] = value

      def get_global(self, key: str) -> Any:
          return self.global_context.get(key)

      def set_for_agent(self, agent_name: str, key: str, value: Any):
          if agent_name not in self.agent_contexts:
              self.agent_contexts[agent_name] = {}
          self.agent_contexts[agent_name][key] = value

      def get_context_for_agent(self, agent_name: str) -> Dict[str, Any]:
          """Get combined global + agent-specific context."""
          return {
              **self.global_context,
              **self.agent_contexts.get(agent_name, {})
          }
  ```
  **File**: `.claude/hooks/agent_memory.py`

- [ ] **2.3.2** Implement context passing between sequential agents
  ```python
  def pass_context(self, from_agent: str, to_agent: str, keys: List[str]):
      """Pass specific context from one agent to another."""
      from_ctx = self.agent_contexts.get(from_agent, {})
      for key in keys:
          if key in from_ctx:
              self.set_for_agent(to_agent, key, from_ctx[key])
  ```
  **File**: `.claude/hooks/agent_memory.py`

- [ ] **2.3.3** Add context injection to sequential workflows
  **File**: `.claude/hooks/agent_orchestrator.py`
  **Method**: Update `WorkflowGenerator.generate_sequential()`

- [ ] **2.3.4** Implement context summarization for large contexts
  ```python
  def summarize_context(self, context: Dict[str, Any], max_tokens: int = 500) -> str:
      """Summarize context to fit within token limits."""
      # Prioritize recent, high-importance entries
      ...
  ```
  **File**: `.claude/hooks/agent_memory.py`

#### 2.4 Checkpoint & Resume

**Insight**: "Save checkpoints after each successfully completed step, allow plan rewinding to the most recent checkpoint"

- [ ] **2.4.1** Create `Checkpoint` dataclass
  ```python
  @dataclass
  class Checkpoint:
      id: str
      workflow_name: str
      step_index: int
      timestamp: datetime
      state: Dict[str, Any]
      completed_agents: List[str]
      pending_agents: List[str]
      intermediate_results: Dict[str, Any]
      memory_snapshot: Dict[str, Any]
  ```
  **File**: `.claude/hooks/agent_checkpoint.py` (new file)

- [ ] **2.4.2** Create `CheckpointManager` class
  ```python
  class CheckpointManager:
      def __init__(self, storage_dir: Path):
          self.storage_dir = storage_dir
          self.storage_dir.mkdir(exist_ok=True)

      def save(self, checkpoint: Checkpoint):
          filepath = self.storage_dir / f"{checkpoint.id}.json"
          filepath.write_text(json.dumps(asdict(checkpoint), default=str))

      def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
          filepath = self.storage_dir / f"{checkpoint_id}.json"
          if filepath.exists():
              data = json.loads(filepath.read_text())
              return Checkpoint(**data)
          return None

      def get_latest(self, workflow_name: str) -> Optional[Checkpoint]:
          """Get most recent checkpoint for a workflow."""
          ...

      def resume_from(self, checkpoint: Checkpoint) -> WorkflowState:
          """Resume workflow execution from checkpoint."""
          ...
  ```
  **File**: `.claude/hooks/agent_checkpoint.py`

- [ ] **2.4.3** Auto-checkpoint after each agent completes
  ```python
  def checkpoint_after_agent(
      self,
      workflow: WorkflowSpec,
      completed_agent: AgentSpec,
      result: AgentResult
  ):
      checkpoint = Checkpoint(
          id=f"{workflow.name}_{completed_agent.name}_{datetime.now().timestamp()}",
          workflow_name=workflow.name,
          step_index=workflow.agents.index(completed_agent),
          completed_agents=[a.name for a in workflow.agents[:step_index+1]],
          pending_agents=[a.name for a in workflow.agents[step_index+1:]],
          intermediate_results={completed_agent.name: result},
          ...
      )
      self.save(checkpoint)
  ```
  **File**: `.claude/hooks/agent_checkpoint.py`

- [ ] **2.4.4** Add resume command
  ```bash
  /agent-resume <checkpoint_id>
  ```
  **File**: `.claude/commands/agent-resume.md` (new file)

- [ ] **2.4.5** Store checkpoints directory
  **Directory**: `.claude/checkpoints/`

---

### Phase 3: Swarm Intelligence (Week 3)

#### 3.1 Topology Options

**Insight**: Claude-Flow supports "hierarchical, mesh, ring, star" topologies

- [ ] **3.1.1** Create `Topology` enum and classes
  ```python
  class Topology(Enum):
      PARALLEL = "parallel"  # All agents independent
      HIERARCHICAL = "hierarchical"  # Manager coordinates workers
      MESH = "mesh"  # All agents can communicate
      RING = "ring"  # Sequential circular communication
      STAR = "star"  # Central coordinator, spoke agents
      PIPELINE = "pipeline"  # Linear sequential

  @dataclass
  class TopologyConfig:
      topology: Topology
      coordinator: Optional[str] = None  # For hierarchical/star
      communication_edges: List[Tuple[str, str]] = field(default_factory=list)
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **3.1.2** Implement `HierarchicalTopology`
  ```python
  class HierarchicalTopology:
      def __init__(self, coordinator: AgentSpec, workers: List[AgentSpec]):
          self.coordinator = coordinator
          self.workers = workers

      def generate_workflow(self, context: Dict[str, Any]) -> str:
          """Generate hierarchical workflow with coordinator managing workers."""
          # 1. Coordinator analyzes task and creates subtasks
          # 2. Workers execute subtasks in parallel
          # 3. Coordinator aggregates results
          ...
  ```
  **File**: `.claude/hooks/agent_topologies.py` (new file)

- [ ] **3.1.3** Implement `MeshTopology`
  ```python
  class MeshTopology:
      def __init__(self, agents: List[AgentSpec]):
          self.agents = agents
          self.adjacency = self._build_full_mesh()

      def _build_full_mesh(self) -> Dict[str, List[str]]:
          """Every agent can communicate with every other agent."""
          return {
              a.name: [b.name for b in self.agents if b.name != a.name]
              for a in self.agents
          }
  ```
  **File**: `.claude/hooks/agent_topologies.py`

- [ ] **3.1.4** Add topology selection to workflows
  ```python
  @dataclass
  class WorkflowSpec:
      # ... existing fields ...
      topology: Topology = Topology.PARALLEL
      topology_config: Optional[TopologyConfig] = None
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **3.1.5** Add topology option to `/agents` command
  ```bash
  /agents run code_review --topology hierarchical
  ```
  **File**: `.claude/commands/agents.md`

#### 3.2 Dynamic Agent Spawning

**Insight**: "Spawn subagents to explore different aspects simultaneously based on task complexity"

- [ ] **3.2.1** Create `DynamicSpawner` class
  ```python
  class DynamicSpawner:
      def __init__(self, max_agents: int = 10):
          self.max_agents = max_agents
          self.active_agents: List[str] = []

      def estimate_agent_count(self, task: str, complexity: TaskComplexity) -> int:
          """Estimate optimal number of agents for a task."""
          base_count = {
              TaskComplexity.TRIVIAL: 1,
              TaskComplexity.SIMPLE: 2,
              TaskComplexity.MODERATE: 3,
              TaskComplexity.COMPLEX: 5,
              TaskComplexity.CRITICAL: 7,
          }.get(complexity, 3)
          return min(base_count, self.max_agents)

      def spawn_additional(self, reason: str) -> Optional[AgentSpec]:
          """Spawn additional agent if needed and capacity allows."""
          if len(self.active_agents) < self.max_agents:
              # Select appropriate agent based on reason
              ...
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **3.2.2** Add spawning triggers
  ```python
  SPAWN_TRIGGERS = {
      "large_codebase": lambda ctx: ctx.get("file_count", 0) > 100,
      "multiple_languages": lambda ctx: len(ctx.get("languages", [])) > 2,
      "high_complexity": lambda ctx: ctx.get("complexity_score", 0) > 0.7,
      "conflicting_results": lambda ctx: ctx.get("agreement_score", 1.0) < 0.5,
  }
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **3.2.3** Implement spawn-on-demand logic
  **File**: `.claude/hooks/agent_orchestrator.py`
  **Method**: Add to workflow executor

#### 3.3 Agent-to-Agent Communication

**Insight**: "Stream-json chaining for real-time agent-to-agent communication" - Claude-Flow

- [ ] **3.3.1** Create `AgentMessage` protocol
  ```python
  @dataclass
  class AgentMessage:
      from_agent: str
      to_agent: str
      message_type: str  # "request", "response", "broadcast", "query"
      content: Dict[str, Any]
      timestamp: datetime
      correlation_id: Optional[str] = None
  ```
  **File**: `.claude/hooks/agent_communication.py` (new file)

- [ ] **3.3.2** Create `MessageBus` for agent communication
  ```python
  class MessageBus:
      def __init__(self):
          self.messages: List[AgentMessage] = []
          self.subscribers: Dict[str, List[Callable]] = {}

      def publish(self, message: AgentMessage):
          self.messages.append(message)
          for subscriber in self.subscribers.get(message.to_agent, []):
              subscriber(message)

      def subscribe(self, agent_name: str, handler: Callable):
          if agent_name not in self.subscribers:
              self.subscribers[agent_name] = []
          self.subscribers[agent_name].append(handler)

      def get_messages_for(self, agent_name: str) -> List[AgentMessage]:
          return [m for m in self.messages if m.to_agent == agent_name]
  ```
  **File**: `.claude/hooks/agent_communication.py`

- [ ] **3.3.3** Add communication context to agent prompts
  ```python
  def inject_communication_context(self, agent: AgentSpec, bus: MessageBus) -> str:
      """Add relevant messages to agent's prompt."""
      messages = bus.get_messages_for(agent.name)
      if not messages:
          return ""

      return f"""
## Messages from Other Agents
{self._format_messages(messages)}

You may respond to these messages or incorporate their findings.
"""
  ```
  **File**: `.claude/hooks/agent_communication.py`

#### 3.4 Collective Intelligence Aggregation

**Insight**: "Weighted consensus based on confidence scores"

- [ ] **3.4.1** Create `ConsensusAggregator` class
  ```python
  class ConsensusAggregator:
      def __init__(self, method: str = "weighted_vote"):
          self.method = method

      def aggregate(self, results: List[AgentResult]) -> AggregatedResult:
          if self.method == "weighted_vote":
              return self._weighted_vote(results)
          elif self.method == "confidence_weighted":
              return self._confidence_weighted(results)
          elif self.method == "majority":
              return self._majority_vote(results)
          elif self.method == "unanimous":
              return self._unanimous(results)

      def _weighted_vote(self, results: List[AgentResult]) -> AggregatedResult:
          """Weight votes by agent expertise and confidence."""
          weights = {
              Model.OPUS: 3.0,
              Model.SONNET: 2.0,
              Model.HAIKU: 1.0,
          }
          ...

      def _confidence_weighted(self, results: List[AgentResult]) -> AggregatedResult:
          """Weight by reported confidence scores."""
          ...
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **3.4.2** Add confidence extraction from agent results
  ```python
  def extract_confidence(self, result: AgentResult) -> float:
      """Extract confidence score from agent output."""
      # Look for patterns like "Confidence: 85%" or "## Confidence: 0.85"
      patterns = [
          r"confidence:\s*(\d+)%",
          r"confidence:\s*(0?\.\d+)",
          r"certainty:\s*(\d+)%",
      ]
      for pattern in patterns:
          match = re.search(pattern, result.output.lower())
          if match:
              value = float(match.group(1))
              return value / 100 if value > 1 else value
      return 0.5  # Default confidence
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **3.4.3** Create aggregation templates for different result types
  ```python
  AGGREGATION_TEMPLATES = {
      "search": {
          "strategy": "union",  # Combine all findings
          "dedup": True,
          "sort_by": "relevance",
      },
      "review": {
          "strategy": "severity_ranked",  # Prioritize by severity
          "dedup": True,
          "group_by": "category",
      },
      "consensus": {
          "strategy": "weighted_vote",
          "threshold": 0.66,
          "require_reasoning": True,
      },
  }
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

---

### Phase 4: Plugin Architecture (Week 4)

#### 4.1 Three-Tier Loading

**Insight**: "Metadata → Instructions → Resources for token efficiency" - wshobson/agents

- [ ] **4.1.1** Create plugin directory structure
  ```
  .claude/plugins/
  ├── trading-safety/
  │   ├── metadata.json      # Tier 1: Always loaded (~50 tokens)
  │   ├── instructions.md    # Tier 2: Loaded when activated (~500 tokens)
  │   └── resources/         # Tier 3: Loaded on demand
  │       ├── examples.md
  │       └── templates.md
  ├── code-review/
  │   ├── metadata.json
  │   ├── instructions.md
  │   └── resources/
  └── ...
  ```
  **Directory**: `.claude/plugins/`

- [ ] **4.1.2** Create `PluginMetadata` schema
  ```json
  {
    "name": "trading-safety",
    "version": "1.0.0",
    "description": "Trading safety and risk management agents",
    "activation_keywords": ["trading", "risk", "order", "position"],
    "agents": ["risk_reviewer", "execution_reviewer", "backtest_reviewer"],
    "dependencies": [],
    "tier1_tokens": 45
  }
  ```
  **File**: `.claude/plugins/trading-safety/metadata.json`

- [ ] **4.1.3** Create `PluginLoader` class
  ```python
  class PluginLoader:
      def __init__(self, plugins_dir: Path):
          self.plugins_dir = plugins_dir
          self.loaded_metadata: Dict[str, PluginMetadata] = {}
          self.loaded_instructions: Dict[str, str] = {}
          self.loaded_resources: Dict[str, Dict[str, str]] = {}

      def load_tier1(self):
          """Load all plugin metadata (always loaded)."""
          for plugin_dir in self.plugins_dir.iterdir():
              if plugin_dir.is_dir():
                  metadata_file = plugin_dir / "metadata.json"
                  if metadata_file.exists():
                      self.loaded_metadata[plugin_dir.name] = json.loads(
                          metadata_file.read_text()
                      )

      def load_tier2(self, plugin_name: str):
          """Load instructions when plugin is activated."""
          if plugin_name not in self.loaded_instructions:
              instructions_file = self.plugins_dir / plugin_name / "instructions.md"
              if instructions_file.exists():
                  self.loaded_instructions[plugin_name] = instructions_file.read_text()

      def load_tier3(self, plugin_name: str, resource_name: str):
          """Load specific resource on demand."""
          if plugin_name not in self.loaded_resources:
              self.loaded_resources[plugin_name] = {}

          resource_file = self.plugins_dir / plugin_name / "resources" / f"{resource_name}.md"
          if resource_file.exists():
              self.loaded_resources[plugin_name][resource_name] = resource_file.read_text()
  ```
  **File**: `.claude/hooks/agent_plugins.py` (new file)

- [ ] **4.1.4** Integrate plugin loader into orchestrator
  **File**: `.claude/hooks/agent_orchestrator.py`
  **Import**: `from .agent_plugins import PluginLoader`

#### 4.2 Progressive Disclosure

**Insight**: "Load details on demand to minimize token usage"

- [ ] **4.2.1** Create progressive loading triggers
  ```python
  PROGRESSIVE_TRIGGERS = {
      "need_examples": ["example", "show me", "how to", "demonstrate"],
      "need_templates": ["template", "format", "structure", "schema"],
      "need_detailed_docs": ["explain", "documentation", "reference", "details"],
  }

  def should_load_resource(self, prompt: str, resource_type: str) -> bool:
      """Determine if a resource should be loaded based on prompt."""
      triggers = PROGRESSIVE_TRIGGERS.get(resource_type, [])
      return any(trigger in prompt.lower() for trigger in triggers)
  ```
  **File**: `.claude/hooks/agent_plugins.py`

- [ ] **4.2.2** Implement lazy loading in agent prompts
  ```python
  def get_agent_prompt(self, agent: AgentSpec, context: Dict[str, Any]) -> str:
      """Build agent prompt with progressive disclosure."""
      prompt = agent.prompt_template

      # Always include instructions
      if agent.plugin:
          self.plugin_loader.load_tier2(agent.plugin)
          prompt = self.plugin_loader.loaded_instructions[agent.plugin] + "\n\n" + prompt

      # Conditionally include resources
      if self.should_load_resource(context.get("user_query", ""), "need_examples"):
          self.plugin_loader.load_tier3(agent.plugin, "examples")
          prompt += f"\n\n## Examples\n{self.plugin_loader.loaded_resources[agent.plugin]['examples']}"

      return prompt
  ```
  **File**: `.claude/hooks/agent_plugins.py`

#### 4.3 Custom Agent Registration

- [ ] **4.3.1** Create agent registration CLI
  ```bash
  python3 .claude/hooks/agent_orchestrator.py register-agent \
      --name "custom_scanner" \
      --role "Custom code scanner" \
      --model haiku \
      --type Explore \
      --prompt-file ./my_scanner_prompt.txt \
      --tags security,custom
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`
  **Method**: Add `register_agent()` CLI command

- [ ] **4.3.2** Create agent registration schema
  ```json
  {
    "custom_agents": {
      "custom_scanner": {
        "name": "CustomScanner",
        "role": "Custom code scanner",
        "model": "haiku",
        "agent_type": "Explore",
        "prompt_template": "...",
        "tags": ["security", "custom"],
        "created_at": "2025-12-04T10:00:00",
        "version": "1.0.0"
      }
    }
  }
  ```
  **File**: `.claude/agent_config.json`

- [ ] **4.3.3** Add custom agent loading to orchestrator
  **File**: `.claude/hooks/agent_orchestrator.py`
  **Method**: Update `AGENT_TEMPLATES` initialization

---

### Phase 5: Observability (Week 5)

#### 5.1 Execution Tracing

- [ ] **5.1.1** Create `ExecutionTrace` dataclass
  ```python
  @dataclass
  class ExecutionTrace:
      trace_id: str
      workflow_name: str
      started_at: datetime
      ended_at: Optional[datetime] = None
      spans: List[TraceSpan] = field(default_factory=list)

  @dataclass
  class TraceSpan:
      span_id: str
      agent_name: str
      started_at: datetime
      ended_at: Optional[datetime]
      status: str  # "running", "success", "failed", "timeout"
      duration_ms: Optional[float] = None
      tokens_in: Optional[int] = None
      tokens_out: Optional[int] = None
      error: Optional[str] = None
  ```
  **File**: `.claude/hooks/agent_tracing.py` (new file)

- [ ] **5.1.2** Create `Tracer` class
  ```python
  class Tracer:
      def __init__(self, storage_dir: Path):
          self.storage_dir = storage_dir
          self.current_trace: Optional[ExecutionTrace] = None

      def start_trace(self, workflow_name: str) -> str:
          trace_id = str(uuid.uuid4())
          self.current_trace = ExecutionTrace(
              trace_id=trace_id,
              workflow_name=workflow_name,
              started_at=datetime.now(),
          )
          return trace_id

      def start_span(self, agent_name: str) -> str:
          span = TraceSpan(
              span_id=str(uuid.uuid4()),
              agent_name=agent_name,
              started_at=datetime.now(),
              status="running",
          )
          self.current_trace.spans.append(span)
          return span.span_id

      def end_span(self, span_id: str, status: str, tokens_in: int = 0, tokens_out: int = 0):
          for span in self.current_trace.spans:
              if span.span_id == span_id:
                  span.ended_at = datetime.now()
                  span.status = status
                  span.duration_ms = (span.ended_at - span.started_at).total_seconds() * 1000
                  span.tokens_in = tokens_in
                  span.tokens_out = tokens_out
                  break

      def save_trace(self):
          filepath = self.storage_dir / f"{self.current_trace.trace_id}.json"
          filepath.write_text(json.dumps(asdict(self.current_trace), default=str))
  ```
  **File**: `.claude/hooks/agent_tracing.py`

- [ ] **5.1.3** Integrate tracing into workflow execution
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **5.1.4** Add trace viewing command
  ```bash
  /agent-trace <trace_id>
  /agent-trace --recent 5
  ```
  **File**: `.claude/commands/agent-trace.md` (new file)

#### 5.2 Token Usage Tracking

- [ ] **5.2.1** Create `TokenTracker` class
  ```python
  class TokenTracker:
      def __init__(self):
          self.usage: Dict[str, TokenUsage] = {}

      def record(self, agent_name: str, model: str, tokens_in: int, tokens_out: int):
          if agent_name not in self.usage:
              self.usage[agent_name] = TokenUsage()
          self.usage[agent_name].add(model, tokens_in, tokens_out)

      def get_total_cost(self) -> float:
          """Calculate total cost based on model pricing."""
          PRICING = {
              "haiku": {"input": 0.25, "output": 1.25},  # per MTok
              "sonnet": {"input": 3.0, "output": 15.0},
              "opus": {"input": 15.0, "output": 75.0},
          }
          ...

      def get_report(self) -> str:
          """Generate token usage report."""
          ...
  ```
  **File**: `.claude/hooks/agent_tracing.py`

- [ ] **5.2.2** Estimate tokens before execution
  ```python
  def estimate_tokens(self, prompt: str) -> int:
      """Estimate token count for a prompt."""
      # Rough estimate: 4 chars per token
      return len(prompt) // 4
  ```
  **File**: `.claude/hooks/agent_tracing.py`

- [ ] **5.2.3** Add token tracking to `/agent-status`
  **File**: `.claude/commands/agent-status.md`

#### 5.3 Performance Metrics Dashboard

- [ ] **5.3.1** Create `MetricsCollector` class
  ```python
  class MetricsCollector:
      def __init__(self):
          self.metrics: Dict[str, List[float]] = {
              "workflow_duration_ms": [],
              "agent_success_rate": [],
              "tokens_per_workflow": [],
              "retries_per_workflow": [],
          }

      def record(self, metric: str, value: float):
          self.metrics[metric].append(value)

      def get_percentiles(self, metric: str) -> Dict[str, float]:
          values = sorted(self.metrics[metric])
          return {
              "p50": values[len(values) // 2],
              "p90": values[int(len(values) * 0.9)],
              "p99": values[int(len(values) * 0.99)],
          }

      def get_dashboard(self) -> str:
          """Generate metrics dashboard."""
          ...
  ```
  **File**: `.claude/hooks/agent_tracing.py`

- [ ] **5.3.2** Add metrics output format
  ```markdown
  ## Agent Orchestrator Metrics

  ### Performance (Last 24h)
  | Metric | P50 | P90 | P99 |
  |--------|-----|-----|-----|
  | Workflow Duration | 45s | 120s | 300s |
  | Agent Success Rate | 95% | 85% | 70% |
  | Tokens per Workflow | 5K | 15K | 50K |

  ### Cost Analysis
  | Model | Invocations | Tokens | Cost |
  |-------|-------------|--------|------|
  | haiku | 150 | 50K | $0.08 |
  | sonnet | 45 | 100K | $0.45 |
  | opus | 5 | 25K | $0.56 |

  ### Error Analysis
  | Error Type | Count | Rate |
  |------------|-------|------|
  | Timeout | 5 | 3% |
  | Rate Limit | 2 | 1% |
  | Other | 1 | 0.5% |
  ```
  **File**: `.claude/commands/agent-metrics.md` (new file)

---

### Phase 6: Advanced Patterns (Week 6)

#### 6.1 Conditional Branching

- [ ] **6.1.1** Create `ConditionalEdge` class
  ```python
  @dataclass
  class ConditionalEdge:
      from_agent: str
      condition: Callable[[AgentResult], str]  # Returns next agent name
      branches: Dict[str, str]  # condition_result -> agent_name

  # Example usage
  def route_after_review(result: AgentResult) -> str:
      if "CRITICAL" in result.output:
          return "security_deep_dive"
      elif "needs_refactor" in result.output:
          return "refactorer"
      else:
          return "done"
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **6.1.2** Add conditional workflow support
  ```python
  @dataclass
  class WorkflowSpec:
      # ... existing fields ...
      conditional_edges: List[ConditionalEdge] = field(default_factory=list)
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **6.1.3** Implement conditional executor
  **File**: `.claude/hooks/agent_orchestrator.py`
  **Method**: Add `execute_conditional_workflow()`

#### 6.2 Cyclic Workflows (Feedback Loops)

- [ ] **6.2.1** Create `CyclicWorkflow` class
  ```python
  class CyclicWorkflow:
      def __init__(self, max_iterations: int = 5):
          self.max_iterations = max_iterations
          self.iteration = 0
          self.convergence_threshold = 0.9

      def should_continue(self, results: List[AgentResult]) -> bool:
          """Determine if another iteration is needed."""
          if self.iteration >= self.max_iterations:
              return False

          # Check for convergence
          quality_score = self.evaluate_quality(results)
          return quality_score < self.convergence_threshold

      def evaluate_quality(self, results: List[AgentResult]) -> float:
          """Evaluate if results meet quality threshold."""
          ...
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **6.2.2** Add cyclic workflow pattern
  ```
  implement → review → fix (if issues) → review → fix → ... → done
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

#### 6.3 Scatter-Gather Pattern

- [ ] **6.3.1** Create `ScatterGather` class
  ```python
  class ScatterGather:
      def __init__(self, scatter_count: int, gather_strategy: str = "merge"):
          self.scatter_count = scatter_count
          self.gather_strategy = gather_strategy

      def scatter(self, task: str, context: Dict[str, Any]) -> List[AgentSpec]:
          """Split task into subtasks for parallel execution."""
          subtasks = self.decompose_task(task, self.scatter_count)
          return [
              AgentSpec(
                  name=f"scatter_worker_{i}",
                  role=f"Process subtask {i}",
                  prompt_template=subtask,
                  model=Model.HAIKU,
              )
              for i, subtask in enumerate(subtasks)
          ]

      def gather(self, results: List[AgentResult]) -> AggregatedResult:
          """Consolidate results from scattered workers."""
          if self.gather_strategy == "merge":
              return self._merge_results(results)
          elif self.gather_strategy == "vote":
              return self._vote_results(results)
          elif self.gather_strategy == "best":
              return self._select_best(results)
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **6.3.2** Add scatter-gather workflow template
  **File**: `.claude/hooks/agent_orchestrator.py`
  **Add to**: `WORKFLOW_TEMPLATES`

#### 6.4 Human Escalation

- [ ] **6.4.1** Create `EscalationManager` class
  ```python
  class EscalationManager:
      def __init__(self, max_agent_attempts: int = 3):
          self.max_agent_attempts = max_agent_attempts
          self.escalation_log: List[EscalationEvent] = []

      def should_escalate(self, agent_name: str, failures: int) -> bool:
          return failures >= self.max_agent_attempts

      def escalate(self, context: EscalationContext) -> str:
          """Generate escalation message for human review."""
          return f"""
## Human Escalation Required

**Task**: {context.task}
**Reason**: {context.reason}
**Attempts**: {context.attempts}

### Agent Failures
{self._format_failures(context.failures)}

### Recommended Actions
1. Review the task requirements
2. Check if additional context is needed
3. Decide whether to retry, modify, or abort

**Reply with**: `retry`, `modify <instructions>`, or `abort`
"""
  ```
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **6.4.2** Add escalation triggers to error handler
  **File**: `.claude/hooks/agent_orchestrator.py`

- [ ] **6.4.3** Create `/agent-escalations` command
  **File**: `.claude/commands/agent-escalations.md` (new file)

---

## 📊 Progress Tracking

### Summary

| Phase | Tasks | Priority | Status |
|-------|-------|----------|--------|
| Phase 1: Error Handling | 24 | P0 | ⬜ Not Started |
| Phase 2: Memory | 16 | P0 | ⬜ Not Started |
| Phase 3: Swarm | 15 | P1 | ⬜ Not Started |
| Phase 4: Plugins | 11 | P1 | ⬜ Not Started |
| Phase 5: Observability | 11 | P2 | ⬜ Not Started |
| Phase 6: Advanced | 10 | P2 | ⬜ Not Started |
| **Total** | **87** | | |

### Completion Checklist

- [ ] Phase 1 Complete (0/24)
- [ ] Phase 2 Complete (0/16)
- [ ] Phase 3 Complete (0/15)
- [ ] Phase 4 Complete (0/11)
- [ ] Phase 5 Complete (0/11)
- [ ] Phase 6 Complete (0/10)
- [ ] All Tests Passing
- [ ] Documentation Updated
- [ ] CLAUDE.md Updated
- [ ] Research Doc Created

---

## 📁 New Files to Create

| File | Purpose |
|------|---------|
| `.claude/hooks/agent_memory.py` | Memory management (working, session, persistent) |
| `.claude/hooks/agent_checkpoint.py` | Checkpoint/resume functionality |
| `.claude/hooks/agent_topologies.py` | Topology implementations |
| `.claude/hooks/agent_communication.py` | Agent-to-agent messaging |
| `.claude/hooks/agent_plugins.py` | Plugin loader and management |
| `.claude/hooks/agent_tracing.py` | Execution tracing and metrics |
| `.claude/commands/agent-resume.md` | Resume from checkpoint |
| `.claude/commands/agent-trace.md` | View execution traces |
| `.claude/commands/agent-metrics.md` | Performance dashboard |
| `.claude/commands/agent-escalations.md` | Human escalation management |
| `.claude/plugins/` | Plugin directory structure |
| `.claude/checkpoints/` | Checkpoint storage |
| `.claude/traces/` | Trace storage |

---

## 🔗 References

- [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Claude-Flow](https://github.com/ruvnet/claude-flow)
- [wshobson/agents](https://github.com/wshobson/agents)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Best Practices for Multi-Agent Orchestration](https://skywork.ai/blog/ai-agent-orchestration-best-practices-handoffs/)
- [Why Multi-Agent Systems Need Memory Engineering](https://www.mongodb.com/company/blog/technical/why-multi-agent-systems-need-memory-engineering)
- [Error Recovery and Fallback Strategies](https://www.gocodeo.com/post/error-recovery-and-fallback-strategies-in-ai-agent-development)

---

## 📝 Change Log

| Date | Change | Impact |
|------|--------|--------|
| 2025-12-04 | Initial upgrade guide created | 87 tasks across 6 phases |
