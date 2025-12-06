# Tool Comparison Tables

**Version**: 1.1.0 | **Last Updated**: November 28, 2025

Comprehensive comparison of tools, frameworks, and services for autonomous Claude Code development.

---

## Table of Contents

1. [November 2025 Release Comparison](#november-2025-release-comparison) ⭐ NEW
2. [Sandbox Technologies](#sandbox-technologies)
3. [Agent Frameworks](#agent-frameworks)
4. [AI Coding Agents](#ai-coding-agents)
5. [Trading MCP Servers](#trading-mcp-servers)
6. [Durable Execution Engines](#durable-execution-engines)
7. [Claude Code Plan Comparison](#claude-code-plan-comparison)
8. [IDE Integration Options](#ide-integration-options)

---

## November 2025 Release Comparison

### Executive Summary: Should You Prioritize New Releases?

**Yes, for most use cases.** November 2025 brought major GA releases that significantly simplify autonomous agent workflows. The table below shows what changed:

| Tool | Before Nov 2025 | November 2025 Release | Impact |
|------|-----------------|----------------------|--------|
| **Claude Code** | Manual permissions per command | Native sandbox (84% fewer prompts) | **High** - Essential for overnight sessions |
| **Cursor** | Synchronous agent mode | Background agents (parallel, remote) | **Medium** - Good for multi-task workflows |
| **Kiro** | Preview (limited features) | GA with CLI, property-based testing | **High** - Spec-driven approach is mature |
| **GitHub Copilot** | Basic agent mode | Isolated subagents (background PRs) | **Medium** - Good GitHub integration |
| **OpenHands** | Early versions | CodeAct 2.1 (53% SWE-bench) | **Medium** - Best open-source option |
| **Docker Sandbox** | Container only | MicroVM roadmap announced | **Low** - Current container approach works |

### Priority Ranking for QuantConnect Trading Bot

Based on project requirements (overnight autonomous sessions, Python/QuantConnect, trading safety):

| Rank | Tool | Why Prioritize | Release Date |
|------|------|---------------|--------------|
| **1** | **Claude Code Sandbox** | Native sandbox reduces prompts 84%, essential for overnight automation | Nov 2025 |
| **2** | **Claude Code on Web** | Kick off sessions from browser, GitHub integration | Nov 12, 2025 |
| **3** | **Kiro GA** | Spec-driven ensures requirements are met before coding | Nov 17, 2025 |
| **4** | **GitHub Copilot Subagents** | Background PR generation, good for CI/CD integration | Nov 18, 2025 |
| **5** | Cursor 2.0 Background Agents | Parallel task execution, but not CLI-native | Oct 29, 2025 |

### Detailed November 2025 Release Analysis

#### Claude Code Sandbox (November 2025) ⭐ TOP PRIORITY

**Source**: [Anthropic Engineering Blog](https://www.anthropic.com/engineering/claude-code-sandboxing)

| Feature | Details |
|---------|---------|
| **Permission Reduction** | 84% fewer prompts in internal testing |
| **Isolation Method** | Linux bubblewrap / macOS seatbelt |
| **Network** | Proxy through Unix domain socket |
| **Filesystem** | Read/write CWD only, system files blocked |
| **Activation** | `/sandbox` command or `--sandbox` flag |
| **Open Source** | Sandboxing code available for custom integration |

**Key Benefits for Overnight Sessions**:
- Runs autonomously without constant permission approvals
- Safe execution of tests, builds, git operations
- Immediate notification if agent tries to escape sandbox
- Combined with watchdog = production-ready overnight loop

```bash
# Start overnight session with sandbox
claude --sandbox -p "Implement feature X" --output-format stream-json
```

#### Claude Code on Web (November 12, 2025)

**Source**: [Anthropic News](https://www.anthropic.com/news/claude-code-on-the-web)

| Feature | Details |
|---------|---------|
| **Access** | Browser-based, no terminal needed |
| **GitHub** | Direct repository connection |
| **Sandbox** | Every task runs in isolated environment |
| **Git** | Secure proxy for authorized repos only |

**Use Case**: Start autonomous tasks from mobile/tablet while away from workstation.

#### Kiro GA (November 17, 2025) ⭐ HIGH PRIORITY

**Source**: [Kiro Blog](https://kiro.dev/blog/general-availability/)

| Feature | Details |
|---------|---------|
| **Users** | 250,000+ developers since preview |
| **Spec Files** | requirements.md, design.md, tasks.md |
| **Property-Based Testing** | Tests code against spec properties |
| **Checkpointing** | Progress preservation built-in |
| **CLI** | Terminal-based agents now supported |
| **Team Plans** | Centralized management |
| **Hooks** | User prompts triggered by file changes |
| **Steering Files** | Persistent knowledge in markdown |

**Pricing**:
| Tier | Cost | Credits/Month |
|------|------|---------------|
| Free | $0 | 50 |
| Pro | $20/user | 1,000 |
| Pro+ | $40 | 2,000 |
| Power | $200 | 10,000 |

**Why Consider for Trading Bot**: Spec-driven development ensures trading logic is explicitly specified before implementation. Property-based testing validates code matches spec - critical for financial applications.

**Real-World Result**: Rackspace completed 52 weeks of modernization work in 3 weeks using Kiro.

#### Cursor 2.0 Background Agents (October 29, 2025)

**Source**: [Cursor Blog](https://cursor.com/blog)

| Feature | Details |
|---------|---------|
| **Execution** | Remote, parallel background agents |
| **Output** | Auto-generated pull requests with summaries |
| **Access** | Desktop, tablet, or mobile browser |
| **Autonomy Controls** | Terminal policies (Off/Auto/Turbo), review policies |
| **Multi-Agent** | Different models for planning vs execution |
| **Productivity** | 40% reduction in context switching (reported) |

**Autonomy Levels**:
- **Terminal Execution**: Off / Auto / Turbo
- **Review Policy**: Always proceed / Agent decides / Request review

```python
# Cursor 2.0 separates planning from execution
# Example: Assign GPT-4 for planning, Claude for execution
```

#### GitHub Copilot Isolated Subagents (November 18, 2025)

**Source**: [GitHub Blog](https://github.blog/news-insights/product-news/github-copilot-the-agent-awakens/)

| Feature | Details |
|---------|---------|
| **Execution** | GitHub Actions-powered environment |
| **Output** | Draft PRs with commit history |
| **IDEs** | JetBrains, Eclipse, Xcode (preview) |
| **Background** | Runs without blocking main chat |
| **Autonomy** | Self-corrects from lint/test errors |
| **Availability** | Pro, Pro+, Business, Enterprise plans |

**Best For**: Teams already using GitHub extensively; excellent for background PR generation while coding.

#### OpenHands CodeAct 2.1

**Source**: [OpenHands Blog](https://openhands.dev/blog/openhands-codeact-21-an-open-state-of-the-art-software-development-agent)

| Metric | Score |
|--------|-------|
| **SWE-Bench Verified** | 53% (SOTA at release) |
| **SWE-Bench Lite** | 41.7% |
| **GitHub Stars** | 61,000+ |
| **License** | MIT (commercial-friendly) |

**Key Improvements in 2.1**:
- Function calling for precise specification
- Claude 3.5 model integration
- Improved directory traversal (fewer stuck loops)

### Comparison: Old vs New Approach

| Aspect | Pre-November 2025 | November 2025+ |
|--------|-------------------|----------------|
| **Permission Handling** | Approve each command | Sandbox auto-approves safe commands |
| **Session Start** | Terminal only | Browser or terminal |
| **Spec Management** | Manual documentation | Kiro auto-generates spec files |
| **Background Work** | Single synchronous session | Parallel background agents |
| **PR Generation** | Manual git commands | Auto-generated PRs with summaries |
| **Progress Tracking** | Custom scripts | Built-in checkpointing |
| **Overnight Safety** | Custom watchdog only | Sandbox + watchdog + checkpoints |

### Recommended Stack for QuantConnect Project (Updated)

| Component | November 2025 Choice | Previous Choice | Why Switch |
|-----------|---------------------|-----------------|------------|
| **Primary Agent** | Claude Code + Sandbox | Claude Code + Docker | Native sandbox, 84% fewer prompts |
| **Spec Management** | Kiro (optional) | Manual CLAUDE.md | Property-based testing for trading logic |
| **Background Tasks** | Claude Code checkpoints | Custom progress file | Built-in, more reliable |
| **CI/CD Integration** | GitHub Copilot subagents | Manual PR creation | Automated background PRs |
| **Sandbox** | Native Claude sandbox | Docker Sandbox | Better integration, easier setup |
| **Watchdog** | Keep custom watchdog | Custom watchdog | Still needed for budget/time limits |

### Migration Path

1. **Immediate**: Enable Claude Code sandbox (`/sandbox` or `--sandbox`)
2. **Week 1**: Test overnight session with sandbox + existing watchdog
3. **Week 2**: Evaluate Kiro for spec-driven strategy development
4. **Week 3**: Integrate GitHub Copilot for automated PR generation
5. **Ongoing**: Keep custom watchdog for budget tracking and circuit breaker

---

## Sandbox Technologies

### Security Isolation Levels

| Technology | Isolation Type | Kernel Sharing | Escape Risk | Boot Time | Memory Overhead |
|------------|---------------|----------------|-------------|-----------|-----------------|
| Full VM (VMware/Hyper-V) | Complete | None | Minimal | 10-60s | 512MB+ |
| Firecracker (E2B) | MicroVM | Minimal | Very Low | <125ms | 5MB |
| libkrun (Microsandbox) | MicroVM | Minimal | Very Low | <50ms | 3MB |
| gVisor | User-space kernel | Intercepted | Low | Instant | 50MB |
| Kata Containers | MicroVM | Minimal | Very Low | <1s | 20MB |
| Docker (default) | Container | Shared | Medium | Instant | Minimal |
| WSL2 | Lightweight VM | Shared | Medium | 2-5s | 1GB |

### Sandbox Feature Comparison

| Feature | Docker Sandbox | Microsandbox | E2B | K8s Agent Sandbox |
|---------|---------------|--------------|-----|-------------------|
| **Availability** | GA (Docker 4.50) | GA | GA | Preview (Nov 2025) |
| **Pricing** | Free (Docker Desktop) | Self-hosted (free) | $0.10/hr | Enterprise |
| **Network Isolation** | Configurable | Full | Full | Full |
| **Filesystem Isolation** | Volumes | Snapshot | Snapshot | PV/PVC |
| **GPU Support** | Yes | Limited | Yes | Yes |
| **Persistence** | Volume mounts | Snapshots | Cloud storage | PVC |
| **Max Session** | Unlimited | Unlimited | 24 hours | Unlimited |
| **Claude Integration** | Native | MCP adapter | MCP server | MCP adapter |
| **Best For** | Development | Self-hosted prod | Cloud agents | Enterprise |

### Docker + E2B Partnership (November 2025)

E2B sandboxes now include direct access to **270+ MCP tools** via Docker's MCP Gateway ([source](https://www.docker.com/blog/docker-e2b-building-the-future-of-trusted-ai/)):

```python
from e2b_code_interpreter import CodeInterpreter

# Every E2B sandbox includes Docker MCP Gateway access
with CodeInterpreter(mcp_tools=["github", "perplexity", "browserbase"]) as sandbox:
    # Agent can safely use 270+ vetted MCP tools
    result = sandbox.notebook.exec_cell("...")
```

**Benefits**: Pre-vetted tools, unified interface, type-safe access in sandboxed environment.

### Recommendation Matrix

| Use Case | Recommended | Alternative |
|----------|-------------|-------------|
| Local development | Docker Sandbox | Microsandbox |
| CI/CD pipelines | Docker Sandbox | gVisor |
| Cloud production | E2B | Firecracker |
| Enterprise/regulated | K8s Agent Sandbox | Full VM |
| Self-hosted production | Microsandbox | Kata Containers |
| Maximum security | Full VM | Firecracker |
| **Parallel multi-feature work** | **Container Use (Dagger)** | K8s Agent Sandbox |

### Container Use by Dagger (Parallel Agents)

**NEW** - Best for running multiple agents on different features simultaneously.

```bash
# Install
brew install dagger/tap/container-use

# Or on all platforms
curl -fsSL https://raw.githubusercontent.com/dagger/container-use/main/install.sh | bash
```

```json
{
  "mcpServers": {
    "container-use": {
      "command": "container-use",
      "args": ["stdio"]
    }
  }
}
```

**Key Features**:
- Each agent gets isolated container + isolated Git worktree
- Real-time visibility into command history and logs
- Direct intervention - drop into any agent's terminal
- Standard git workflow - `git checkout <branch_name>` to review work

**How It Works**:
1. **Git Worktree Management**: Each agent triggers automatic creation of a new Git worktree
2. **Containerized Isolation**: Dagger-managed containers with isolated resources
3. **Automatic Port Assignment**: No multi-agent port conflicts

**Status**: Early development, known issues being refined.

### Microsandbox Details (Self-Hosted MicroVMs)

**Key advantages** over Firecracker ([source](https://github.com/microsandbox/microsandbox)):
- Works with Apple HV (M-series Macs) in addition to KVM
- Native MCP server integration - works directly with Claude Code
- Apache 2.0 license - commercial-friendly
- Sub-200ms startup times
- Each sandbox gets its own virtual machine with dedicated kernel

```bash
# Installation via Cargo
cargo install microsandbox-server

# Or via script
curl -fsSL https://get.microsandbox.dev | sh

# Start with MCP server
microsandbox serve --mcp
```

**Platform Requirements**:
- Linux: KVM support required
- macOS: Apple Silicon only (M1/M2/M3/M4), Intel not supported

### Kubernetes Agent Sandbox (KubeCon November 2025)

K8s-native sandbox primitive under SIG Apps ([source](https://cloud.google.com/blog/topics/kubernetes)):

**Key Resources**:
- `Sandbox` - Core resource for isolated agent execution
- `SandboxTemplate` - Blueprint for resource limits, base image, security policies
- `SandboxClaim` - Transactional resource for frameworks (LangChain, ADK) to request environments

**WarmPools Feature**: Pre-warmed pods deliver sub-second cold starts (90% improvement).

```yaml
# Example SandboxTemplate
apiVersion: agent-sandbox.sigs.k8s.io/v1alpha1
kind: SandboxTemplate
metadata:
  name: trading-bot-sandbox
spec:
  runtime: gvisor  # or kata-containers
  resources:
    limits:
      memory: "4Gi"
      cpu: "2"
  warmPool:
    minReady: 3
    maxSize: 10
```

### Daytona Detailed Benchmarks

Daytona ($7M seed funding) provides enterprise-grade sandboxes ([source](https://github.com/daytonaio/daytona)):

| Operation | Time |
|-----------|------|
| Creation | 71ms |
| Execution | 67ms |
| Cleanup | 59ms |
| **Total** | **197ms** |

**Default Resources**: 1 vCPU, 1GB RAM, 3GiB disk
**Auto-stop**: After 15 minutes of inactivity

### Cloud Sandbox Platforms Comparison

| Platform | Technology | GPU Support | Boot Time | Pricing | Best For |
|----------|------------|-------------|-----------|---------|----------|
| **E2B** | Firecracker MicroVM | Yes | <125ms | $0.10/hr | Cloud-native agents |
| **Microsandbox** | libkrun MicroVM | Limited | <50ms | Self-hosted (free) | Self-hosted production |
| **Modal** | gVisor + custom | Yes (A100/H100) | <1s | $0.50/hr compute | GPU-intensive workloads |
| **Daytona** | Container + MicroVM | No | 197ms total | $0.08/hr | Enterprise dev environments |
| **Northflank** | gVisor | Yes | ~1s | $0.03/hr base | Multi-cloud deployments |
| **Docker Sandbox** | Containers | Yes | Instant | Free (Docker Desktop) | Local development |

### Modal Sandbox Details

Modal uses gVisor for secure code execution with GPU support ([source](https://modal.com/docs/guide/sandbox)):

```python
import modal

app = modal.App()

@app.function(gpu="A100")
def run_in_sandbox():
    # Secure execution with GPU access
    sb = modal.Sandbox.create(
        app=app,
        image=modal.Image.debian_slim().pip_install("torch"),
        timeout=300,
        cpu=2,
        memory=8192
    )
    result = sb.exec("python", "-c", "import torch; print(torch.cuda.is_available())")
    return result

# Benefits: gVisor isolation + GPU support in same sandbox
```

**Key Features**:
- gVisor kernel-level isolation
- GPU support (A100, H100, T4)
- Automatic scaling
- Pay-per-second billing

### Northflank Comparison Points

From their comparison with Modal ([source](https://northflank.com/compare/modal-vs-northflank)):
- More flexible multi-cloud deployment
- Better for persistent services (vs Modal's serverless focus)
- Lower base pricing for always-on workloads
- Native Kubernetes under the hood

---

## Agent Frameworks

### Framework Comparison

| Framework | Language | Architecture | State Mgmt | Learning Curve | Maturity |
|-----------|----------|--------------|------------|----------------|----------|
| Claude Agent SDK | Python | Native | Built-in | Low | GA |
| LangGraph | Python | Graph-based | Checkpoints | Medium | GA |
| CrewAI | Python | Role-based | Process | Low | GA |
| AutoGen | Python | Multi-agent | Memory | Medium | GA |
| Semantic Kernel | C#/Python | Plugin-based | Planners | Medium | GA |
| Temporal.io | Multi | Workflow | Durable | High | GA |
| Microsoft Agent Framework | Python | Unified | Distributed | Medium | Preview |
| **SmolAgents** | Python | Code Agents | Minimal | **Very Low** | GA |

### Feature Matrix

| Feature | Claude SDK | LangGraph | CrewAI | AutoGen | Temporal | SmolAgents |
|---------|------------|-----------|--------|---------|----------|------------|
| **Native Claude** | Yes | Adapter | Adapter | Adapter | Adapter | Adapter |
| **Multi-model** | No | Yes | Yes | Yes | Yes | Yes (LiteLLM) |
| **Human-in-loop** | Yes | Yes | Yes | Yes | Yes | Yes |
| **Tool calling** | Native | Native | Native | Native | Via activities | Code-based |
| **Streaming** | Yes | Yes | Yes | Yes | No | Yes |
| **Async support** | Yes | Yes | Yes | Yes | Yes | Yes |
| **State persistence** | Manual | Checkpoints | Process | Memory | Durable | Minimal |
| **Retry logic** | Manual | Built-in | Built-in | Built-in | Advanced | Manual |
| **Observability** | Basic | LangSmith | Basic | Logging | Dashboard | Basic |
| **Cost tracking** | Basic | LangSmith | Manual | Manual | Manual | Manual |
| **Sandbox support** | Docker | No | No | Docker | No | E2B/Modal/Docker |

### Framework Recommendations by Use Case

| Use Case | Primary | Alternative | Why |
|----------|---------|-------------|-----|
| Simple automation | Claude Agent SDK | CrewAI | Native integration, lowest overhead |
| Complex workflows | LangGraph | Temporal | State machines, checkpointing |
| Multi-agent debate | AutoGen | CrewAI | Designed for agent collaboration |
| Production systems | Temporal.io | LangGraph | Fault tolerance, durability |
| Role-based tasks | CrewAI | AutoGen | Intuitive role definitions |
| Enterprise scale | Temporal.io | Microsoft Agent Framework | Battle-tested, scalable |
| **Rapid prototyping** | **SmolAgents** | CrewAI | Minimal code, code-first approach |
| **Sandboxed execution** | SmolAgents | AutoGen | Built-in E2B/Modal/Docker support |
| **Overnight loops** | continuous-claude | claude-flow | PR-based iteration with CI |
| **Spec-driven** | Kiro | claude-code-workflow | Requirements → design → tasks |

### Agentic Design Patterns

Key patterns for autonomous coding workflows ([source](https://weaviate.io/blog/what-are-agentic-workflows)):

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Reflection** | Agent evaluates own outputs before finalizing | Self-improvement, error detection |
| **Tool Use** | Dynamic interaction with external resources | APIs, databases, web search |
| **Semantic Routing** | Route tasks to best model/tool based on intent | Cost optimization, specialization |
| **Planning** | Decompose complex goals into subtasks | Multi-step implementations |
| **ReAct** | Reason + Act in alternating steps | Iterative problem solving |
| **Multi-Agent** | Specialized agents collaborate | Complex projects, parallel work |

### Agent Framework Updates (November 2025)

#### AutoGen v0.4 (Async Event-Driven)

Complete async, event-driven redesign ([source](https://microsoft.github.io/autogen/0.4.1/reference/python/autogen_ext.code_executors.docker.html)):

```python
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

# DockerCommandLineCodeExecutor is now the default for MagenticOne
async with DockerCommandLineCodeExecutor(
    image="python:3.11-slim",
    timeout=60,
    auto_remove=True,
    work_dir="/workspace"
) as executor:
    code_executor_agent = CodeExecutorAgent(
        "code_executor",
        code_executor=executor
    )
```

Install: `pip install autogen-ext[docker]`

#### CrewAI v1.6.0 (MCP Support)

Comprehensive MCP support with MCPServerAdapter ([source](https://docs.crewai.com/en/mcp/overview)):

```python
from crewai import Agent
from crewai.tools import MCPServerAdapter

# Simple approach
agent = Agent(
    role="Trading Bot Developer",
    mcps=["quantconnect-mcp", "filesystem-mcp"]
)

# Advanced MCPServerAdapter approach
qc_mcp = MCPServerAdapter(
    server="quantconnect/mcp-server",
    transport="stdio"
)

# Supports: stdio, SSE, streamable HTTP transports
```

Install: `pip install crewai-tools[mcp]`

**Note**: crewAI-tools repository was archived November 10, 2025.

**Performance Benchmark**: Independent testing shows CrewAI executes **5.76x faster** than LangGraph for equivalent multi-agent workflows ([source](https://github.com/crewAIInc/crewAI), [benchmark](https://aiagentinsider.com/crewai-vs-langgraph/)). This performance advantage comes from:
- Optimized agent communication patterns
- Reduced overhead in role-based task delegation
- More efficient memory management between agents

#### LangGraph v0.2.31+ (Interrupt Function)

New `interrupt` function replaces static breakpoints ([source](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/)):

```python
from langgraph.types import interrupt, Command

def human_review_node(state):
    # Pause for human approval
    is_approved = interrupt({
        "question": "Approve this backtest configuration?",
        "config": state["backtest_config"]
    })

    if is_approved:
        return Command(goto="run_backtest")
    else:
        return Command(goto="revise_config")

# Resume after interrupt
graph.invoke(Command(resume="Your response here"), thread)
```

**Requirements**: Checkpointing must be enabled (in-memory saver).

---

## AI Coding Agents

### Agent Comparison (November 2025)

| Agent | Model | IDE | Autonomous | Sandbox | Pricing |
|-------|-------|-----|------------|---------|---------|
| Claude Code | Claude 4 | CLI/VSCode | Yes | Docker | Usage-based |
| Cursor | Multi | Custom IDE | Yes (Agent mode) | No | $20-40/mo |
| Windsurf | Claude/GPT | Custom IDE | Yes | No | $15-60/mo |
| Cline | Multi | VSCode | Yes | No | Usage-based |
| Aider | Multi | CLI | Yes | No | Usage-based |
| Kiro | Claude | Custom IDE | Yes (specs) | Yes | Free-$39/mo |
| Google Antigravity | Gemini | Custom IDE | Yes | Yes | Preview |
| GitHub Copilot Agent | Multi | VSCode/GitHub | Yes (background) | GitHub Actions | $10-39/mo |
| **OpenCode** | Multi (75+) | CLI/TUI | Yes | No | Free/OSS |
| **OpenAI Codex** | GPT-4.1 | Cloud/ChatGPT | Yes | Isolated containers | ChatGPT Plus |
| **SmolAgents** | Multi | Python lib | Yes | E2B/Modal/Docker | Free/OSS |
| **OpenHands** | Multi | Web/CLI | Yes | Docker | Free/OSS |
| **Refact.ai** | Multi | IDE plugin | Yes | On-premise | Free-Enterprise |
| **Qodo** | Multi | IDE plugin | Yes (multi-agent) | No | Free-Enterprise |
| **Devin** | Custom | Web | Yes | Cloud sandbox | Enterprise |

### Detailed Feature Comparison

| Feature | Claude Code | Cursor | Windsurf | Cline | Kiro |
|---------|-------------|--------|----------|-------|------|
| **Extended thinking** | Yes (ultrathink) | No | No | Yes | Yes |
| **Context window** | 200K | 128K | 128K | 200K | 200K |
| **Multi-file editing** | Yes | Yes | Yes | Yes | Yes |
| **Git integration** | Native | Native | Native | Native | Native |
| **MCP support** | Yes | Limited | Limited | Yes | Preview |
| **Headless mode** | Yes | No | No | Yes | Yes |
| **Memory/RAG** | Manual | Codebase | Codebase | Yes | Specs |
| **Background tasks** | Yes | No | Flows | Yes | Yes |
| **Self-correction** | Yes | Limited | Yes | Yes | Yes |
| **Cost visibility** | Yes | Limited | Yes | Yes | Yes |

### Agent Autonomy Levels

| Agent | Autonomy Level | Human Gates | Max Unattended |
|-------|---------------|-------------|----------------|
| Claude Code | High | Configurable | Hours (with watchdog) |
| Cursor | Medium | Per-change | Minutes |
| Windsurf | High (Flows) | Checkpoints | Hours |
| Cline | High | Configurable | Hours |
| Aider | Medium | Per-commit | Minutes |
| Kiro | Very High (Specs) | Spec approval | Hours |
| Google Antigravity | Very High | Agent loops | Hours |

### New Agents (November 2025)

| Agent | Release | Key Innovation | Status |
|-------|---------|----------------|--------|
| Kiro (AWS) | Nov 17, 2025 | Spec-driven development | GA |
| Google Antigravity | Nov 18, 2025 | Agent-first IDE, multi-model | Preview |
| Cline v3.25 | Nov 13, 2025 | Deep Planning, Focus Chain, Auto Compact (5M tokens) | GA |
| Microsoft Agent Framework | Oct 1, 2025 | Unified AutoGen + Semantic Kernel | Preview |

### Kiro Details (AWS - GA November 17, 2025)

**250,000+ developers** have adopted Kiro since preview release.

**Spec-Driven Development** produces three specification files:
- `requirements.md` - User stories in EARS format
- `design.md` - Technical architecture, components, data models
- `tasks.md` - Checklist of coding tasks

**New GA Features**:
- Property-based testing for spec correctness
- Checkpointing for progress preservation
- Kiro CLI for terminal-based agents
- Team plans with centralized management

**Pricing**:
| Tier | Cost | Credits/Month |
|------|------|---------------|
| Free | $0 | 50 |
| Pro | $20/user | 1,000 |
| Pro+ | $40 | 2,000 |
| Power | $200 | 10,000 |

**Real-World Results**: Rackspace completed 52 weeks of modernization in 3 weeks using Kiro.

### OpenCode Details (Terminal TUI)

Open-source AI coding agent built for the terminal ([source](https://opencode.ai/)):

**Stats**: 30,000+ GitHub stars, 300,000+ monthly users

**Key Features**:
- **Native TUI**: Responsive, themeable terminal interface
- **LSP Enabled**: Automatically loads right LSPs for LLM
- **Multi-Session**: Start multiple agents in parallel on same project
- **75+ LLM Providers**: Via Models.dev, including local models
- **Claude Pro Support**: Login with Anthropic for Pro/Max usage

**Installation**:
```bash
# Via curl
curl -fsSL https://opencode.ai/install | bash

# Via npm
npm i -g opencode-ai@latest

# Via Homebrew
brew install opencode
```

**Built-in Agents**:
- `build` - Default, full access for development
- `plan` - Read-only for analysis/exploration

### OpenAI Codex Details (Cloud Agent)

Cloud-based AI coding agent integrated into ChatGPT ([source](https://openai.com/codex/)):

**Security Architecture**:
- Executes in **isolated OpenAI-managed containers**
- **Internet access disabled by default** during task execution
- Cannot access external websites, APIs, or services
- Setup phase has internet access for dependencies

**Sandbox Options**:
| Environment | Isolation Method |
|-------------|------------------|
| Codex Cloud | Isolated OpenAI containers |
| Codex CLI (macOS) | Seatbelt policies |
| Codex CLI (Linux) | seccomp + landlock |

**Internet Access** (if enabled):
- Can configure domain allowlist
- HTTP method restrictions
- **Warning**: Exposes to prompt injection, exfiltration risks

**Availability**: ChatGPT Plus users (released June 2025)

### SmolAgents Details (HuggingFace Code Agents)

Minimalist Python library for agents that write actions in code ([source](https://huggingface.co/docs/smolagents)):

**Philosophy**: Logic fits in ~1000 lines of code; abstractions kept minimal.

**Code Agents vs Tool-Calling**:
- Traditional agents: Generate function calls sequentially
- **Code Agents**: Write Python code blocks to execute entire plans at once

```python
from smolagents import CodeAgent, HfApiModel

# Create agent with any model
agent = CodeAgent(
    tools=[],
    model=HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct"),
    max_steps=10
)

# Run task
result = agent.run("Calculate the factorial of 10")
```

**Sandbox Support**:
- E2B (Firecracker microVMs)
- Modal (gVisor containers)
- Docker
- Pyodide + Deno (WebAssembly)

**Model Support**: Any LLM via LiteLLM - local transformers, ollama, OpenAI, Anthropic, etc.

**Input Types**: Text, vision, video, audio

**Educational**: [DeepLearning.AI course](https://www.deeplearning.ai/short-courses/building-code-agents-with-hugging-face-smolagents/) by Thomas Wolf (HF co-founder)

### GitHub Copilot Coding Agent (GA October 2025)

GitHub's asynchronous background agent ([docs](https://docs.github.com/en/copilot/concepts/coding-agent)):

**How It Works**:
1. Assign GitHub issue to Copilot (or delegate from chat)
2. Agent works in isolated GitHub Actions environment
3. Pushes commits to draft PR as it works
4. You review and merge when complete

**Key Features**:
- Background execution while you do other work
- Excels at low-to-medium complexity in well-tested codebases
- Custom agents for specialized workflows (frontend reviewer, test generator, security auditor)
- Linear integration for issue management

**Pricing**: 1 premium request per model request (included in Pro/Pro+/Business/Enterprise)

### OpenHands Details (Formerly OpenDevin)

Open-source platform for AI software developers ([source](https://arxiv.org/abs/2407.16741)):

**Stats**: 61,000+ GitHub stars, most popular open-source AI agent

**Key Features**:
- Devin-like experience
- Docker-based execution
- Web UI and headless mode
- Multi-model support

```bash
# Quick start
docker run -it --rm \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -p 3000:3000 \
  ghcr.io/all-hands-ai/openhands:latest
```

**Performance**: Resolves ~50% of SWE-bench Verified issues

### Refact.ai Details

Open-source, privacy-focused AI coding agent ([source](https://refact.ai/)):

**Key Features**:
- **On-premise deployment** - keep code private
- **Self-hosting** - run on your hardware
- **SWE-bench leader** - 74.4% on Verified (using Claude Sonnet 4)
- 2000 free "coins"/month

```bash
# Self-hosted installation
docker run -d --gpus all -p 8008:8008 smallcloud/refact
```

### Qodo Details (Multi-Agent System)

Beyond autocomplete - orchestrates across workflows ([source](https://www.qodo.ai/)):

**Built-in Agents**:
- **Qodo Gen** - Code generation
- **Qodo Merge** - PR analysis
- **Qodo Cover** - Test coverage

**Key Capability**: Proposed schema migration PR, validated against staging, auto-tagged Jira ticket - full workflow without human intervention.

### Devin AI Details (by Cognition)

First AI software engineer ([source](https://cognition.ai/blog/introducing-devin)):

**Architecture**: Shell + code editor + browser in sandboxed compute environment

**Performance**:
- 13.86% on original SWE-bench (vs 1.96% previous SOTA at launch)
- Can plan and execute tasks requiring thousands of decisions

**Controversy**: Criticized for failing Upwork project demonstrations

---

## Trading MCP Servers

### Server Comparison

| Server | Type | Data | Trading | Auth | Pricing |
|--------|------|------|---------|------|---------|
| QuantConnect | Official | Full | Yes | API Key | Free tier |
| Alpaca | Official | Full | Yes | OAuth/API | Free tier |
| Alpha Vantage | Official | Full | No | API Key | Free tier |
| Finviz | Community | Screening | No | API Key | Elite ($40/mo) |
| Financial Datasets | Community | Historical | No | API Key | Varies |
| Yahoo Finance | Community | Limited | No | None | Free |

### Feature Matrix

| Feature | QuantConnect | Alpaca | Alpha Vantage | Finviz |
|---------|--------------|--------|---------------|--------|
| **Real-time quotes** | Yes | Yes | 1-min delay | 15-min delay |
| **Historical data** | Yes | Yes | Yes | Limited |
| **Options data** | Yes | Yes | No | Screening |
| **Order execution** | Yes | Yes | No | No |
| **Backtesting** | Yes | No | No | No |
| **Paper trading** | Yes | Yes | No | No |
| **Live trading** | Yes | Yes | No | No |
| **Fundamental data** | Yes | Limited | Yes | Yes |
| **SEC filings** | No | No | No | Yes |
| **Insider trading** | No | No | No | Yes |
| **Technical indicators** | Yes | No | Yes | Limited |
| **Screening** | Limited | No | No | Yes |

### MCP Server Installation

| Server | Install Command | Config Location |
|--------|-----------------|-----------------|
| QuantConnect | `docker pull quantconnect/mcp-server` | `.mcp.json` |
| Alpaca | `npx -y @anthropic/alpaca-mcp` | `.mcp.json` |
| Alpha Vantage | `npx -y @anthropic/alphavantage-mcp` | `.mcp.json` |
| Finviz | `uvx finviz-mcp` | `.mcp.json` |
| Financial Datasets | `pip install financial-datasets-mcp` | `.mcp.json` |

### Recommended Stack for This Project

| Purpose | Primary | Backup |
|---------|---------|--------|
| Backtesting | QuantConnect MCP | Local LEAN |
| Live trading | QuantConnect MCP | Alpaca |
| Market data | Alpha Vantage | Yahoo Finance |
| Screening | Finviz | Manual scan |
| Fundamentals | Alpha Vantage | QuantConnect |

---

## Durable Execution Engines

### Engine Comparison

| Engine | Language | Hosting | Pricing | Learning Curve | Production Users |
|--------|----------|---------|---------|----------------|------------------|
| Temporal.io | Multi | Cloud/Self | Cloud: $/action | High | Replit, OpenAI |
| Inngest | TypeScript | Cloud/Self | Free tier | Medium | Vercel |
| Trigger.dev | TypeScript | Cloud/Self | Free tier | Medium | - |
| AWS Step Functions | JSON/YAML | AWS | $/transition | Medium | Enterprise |
| Prefect | Python | Cloud/Self | Free tier | Medium | Data teams |
| Dagster | Python | Cloud/Self | Free tier | Medium | Data teams |

### Feature Comparison

| Feature | Temporal | Inngest | Step Functions | Prefect |
|---------|----------|---------|----------------|---------|
| **Retry policies** | Advanced | Advanced | Basic | Advanced |
| **State durability** | Complete | Complete | Complete | Partial |
| **Human approval** | Signals | Events | Manual | Sensors |
| **Long-running** | Yes (years) | Yes (days) | Yes (1 year) | Limited |
| **Visibility** | Excellent | Good | Good | Good |
| **Debugging** | Timeline replay | Logs | X-Ray | Logs |
| **Versioning** | Native | Events | Manual | Native |
| **Python SDK** | Yes | Limited | Boto3 | Native |

### Recommendations

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Agent orchestration | Temporal.io | Best durability, proven at scale |
| Serverless workflows | Inngest | Easy setup, good free tier |
| AWS-native | Step Functions | Native integration |
| Data pipelines | Prefect | Python-native, great UI |
| TypeScript projects | Trigger.dev | Modern DX |

---

## Claude Code Plan Comparison

### Plan Tiers (Verified November 2025)

| Feature | Pro | Max 5x | Max 20x |
|---------|-----|--------|---------|
| **Monthly cost** | $20 | $100 | $200 |
| **5-hour prompts** | 10-40 | 50-200 | 200-800 |
| **Weekly Sonnet 4** | 40-80 hrs | 140-280 hrs | 240-480 hrs |
| **Weekly Opus 4** | None | 15-35 hrs | 24-40 hrs |
| **Extended thinking** | Yes | Yes | Yes |
| **Priority access** | No | Yes | Yes |
| **Background tasks** | Limited | Yes | Yes |
| **Opus → Sonnet switch** | N/A | At 20% usage | At 50% usage |

### Value Analysis

| Plan | Sonnet $/hr | Opus $/hr | Value Rating |
|------|-------------|-----------|--------------|
| Pro | $0.25-0.50 | N/A | Good for light use |
| Max 5x | $0.36-0.71 | $2.86-6.67 | **Best value for Opus** |
| Max 20x | $0.42-0.83 | $5.00-8.33 | Best for heavy Sonnet |

**Important**: Max 20x costs 2x Max 5x but provides only ~1.7x Sonnet hours and ~1.3x Opus hours. **Max 5x offers better dollar-per-Opus-hour value.**

### Plan Selection Guide

| Usage Pattern | Recommended Plan | Rationale |
|---------------|------------------|-----------|
| < 2 hrs/day coding | Pro | Sufficient quota |
| 2-6 hrs/day coding | Max 5x | 5x more capacity |
| Overnight sessions | Max 20x | Maximum weekly hours |
| Weekend marathons | Max 20x | Burst capacity |
| Team development | API (usage-based) | Better cost control |

---

## IDE Integration Options

### IDE Comparison for Claude Code

| IDE | Claude Code Support | Extension | Sandbox | MCP |
|-----|---------------------|-----------|---------|-----|
| VSCode | Native | Yes | Docker | Yes |
| Cursor | Claude API | Built-in | No | Limited |
| Windsurf | Claude API | Built-in | No | Limited |
| JetBrains | Plugin | Third-party | No | Limited |
| Neovim | CLI integration | Third-party | Docker | Yes |
| Terminal | Native | N/A | Docker | Yes |

### VSCode Extension Features

| Feature | Claude Code Extension | Copilot | Cline |
|---------|----------------------|---------|-------|
| **Inline completion** | No | Yes | No |
| **Chat interface** | Yes | Yes | Yes |
| **Multi-file edit** | Yes | Limited | Yes |
| **Terminal access** | Yes | No | Yes |
| **Git integration** | Yes | Yes | Yes |
| **Custom commands** | Yes | No | Yes |
| **MCP servers** | Yes | No | Yes |
| **Background tasks** | Yes | No | Yes |

---

## Summary Recommendations

### For This Trading Bot Project

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Plan** | Max 5x or Max 20x | Max 5x better Opus value; Max 20x for heavy Sonnet use |
| **Sandbox** | Docker Sandbox | Native integration, good security |
| **Parallel Work** | Container Use (Dagger) | Isolated Git worktrees per agent |
| **Framework** | Claude Agent SDK + Temporal | Native + durability |
| **Trading MCP** | QuantConnect | Full lifecycle support |
| **Data MCP** | Alpha Vantage + Finviz | Comprehensive coverage |
| **Watchdog** | Custom Python | Project-specific needs |
| **IDE** | VSCode + Terminal | Best Claude Code support |

### Implementation Priority

1. **Immediate**: Docker Sandbox + Watchdog
2. **Short-term**: QuantConnect MCP + Alpha Vantage MCP
3. **Medium-term**: Temporal workflows
4. **Long-term**: Multi-agent architecture

---

## Related Documentation

- [README.md](README.md) - Main guide
- [INSTALLATION.md](INSTALLATION.md) - Setup instructions
- [TODO.md](TODO.md) - Implementation checklist
