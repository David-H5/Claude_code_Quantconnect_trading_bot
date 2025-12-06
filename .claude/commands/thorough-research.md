# Thorough Research Command

Perform thorough multi-pass web research using fast parallel agents.

## Usage

```
/thorough-research <url_or_topic> [options]
```

## Arguments

- `url_or_topic`: A URL to research OR a topic for web search
- `--quick`: Quick mode (overview only, fewer agents)
- `--urls <url1> <url2>`: Additional URLs to research in parallel

## How It Works

1. **Web Search** (if topic provided): Search for relevant URLs
2. **Multi-Pass Fetching**: Each URL is fetched multiple times with different extraction prompts:
   - Overview & Concepts
   - Code Examples (verbatim)
   - API & Signatures
   - Configuration & Parameters
   - Usage Examples & Patterns
3. **Parallel Agents**: Uses fast haiku agents for parallel fetching
4. **Immediate Persistence**: Results saved to docs/research/ immediately

## Process

When you invoke this command, I will:

1. Generate a research plan with fast agent (haiku) Task calls
2. Spawn agents in parallel to fetch different aspects
3. Consolidate results into a research document
4. Save to `docs/research/` with proper naming

## Examples

```bash
# Research a specific URL thoroughly
/thorough-research https://www.quantconnect.com/docs/v2/writing-algorithms/securities/asset-classes/equity-options/greeks

# Research a topic (will search first)
/thorough-research "QuantConnect iron condor implementation 2025"

# Quick overview only
/thorough-research https://example.com --quick

# Multiple URLs in parallel
/thorough-research https://url1.com --urls https://url2.com https://url3.com
```

---

$ARGUMENTS

## Thorough Research Execution

I'll now perform thorough multi-pass research using fast parallel agents.

**Input**: `$ARGUMENTS`

### Step 1: Analyze Input

First, let me determine if this is a URL or a topic that needs web search.

### Step 2: Generate Research Plan

Using the thorough research helper to plan multi-pass fetching:

```bash
python3 .claude/hooks/thorough_research.py plan "$ARGUMENTS" --topic "research"
```

### Step 3: Execute Parallel Agents

I will spawn multiple haiku agents in parallel, each fetching a different aspect:
- Agent 1: Overview & Concepts
- Agent 2: Code Examples (verbatim)
- Agent 3: API & Configuration
- Agent 4: Usage Patterns

### Step 4: Consolidate & Save

After all agents return, I will:
1. Merge findings into a comprehensive document
2. Save to `docs/research/` with timestamp
3. Update the research index

### Instructions for Claude

1. If `$ARGUMENTS` is a URL, use `thorough_research.py` to generate Task calls
2. If `$ARGUMENTS` is a topic, first do WebSearch to find relevant URLs
3. Spawn ALL Task calls in a SINGLE message for parallel execution
4. Use model="haiku" for all fetch agents (fast and cost-effective)
5. After agents return, consolidate findings and save to file IMMEDIATELY
6. Never hold research only in context - always persist to file

**IMPORTANT**: Execute Task calls in parallel (single message with multiple Task tool uses) to minimize total time.
