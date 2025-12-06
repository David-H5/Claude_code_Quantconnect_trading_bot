# UPGRADE-012.2: ACE Reflector & Semantic Routing

## Research Overview

**Date**: December 3, 2025
**Scope**: Implement ACE Reflector module and hybrid semantic+keyword routing
**Status**: ✅ Complete - P0 and P1 items implemented
**Parent**: [UPGRADE-012 Hierarchical Prompts](UPGRADE-012-HIERARCHICAL-PROMPTS.md)
**Predecessor**: [UPGRADE-012.1 Depth+Width Classification](UPGRADE-012.1-DEPTH-WIDTH-CLASSIFICATION.md)

---

## Phase 0: Research Summary

**Search Date**: December 3, 2025 at ~10:30 AM EST

### Research Topic 1: Multi-Agent Task Complexity

**Search Query**: "multi-agent task complexity depth width classification 2025 arXiv"

**Key Sources**:
1. [On the Importance of Task Complexity in Evaluating LLM-Based Multi-Agent Systems (Published: Oct 2025)](https://arxiv.org/abs/2510.04311)
2. [DeepWideSearch Benchmark (Published: Oct 2025)](https://arxiv.org/pdf/2510.20168)
3. [Multi-Agent Deep Research with M-GRPO (Published: Nov 2025)](https://arxiv.org/html/2511.13288)

**Key Findings**:
- Depth × Width framework confirmed: benefits of multi-agent systems increase with both dimensions
- Depth (reasoning chain length) has more pronounced effect than width
- DeepWideSearch benchmark tests "high width, high depth" tasks requiring extensive information and deep reasoning
- Current agents face substantial limitations with depth+width tasks

**Applied**: Already implemented depth+width indicators in UPGRADE-012.1

---

### Research Topic 2: ACE (Agentic Context Engineering) Framework

**Search Query**: "ACE framework autonomous coding evaluation outcome-based learning 2025"

**Key Sources**:
1. [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models (Published: Oct 2025)](https://arxiv.org/abs/2510.04618)
2. [Stanford's ACE Framework - InfoQ Analysis (Published: Oct 2025)](https://www.infoq.com/news/2025/10/agentic-context-eng/)
3. [ACE-Bench: Benchmarking Agentic Coding (Published: 2025)](https://openreview.net/forum?id=41xrZ3uGuI)

**Key Findings**:

The ACE framework has three core components:

| Component | Purpose | Key Function |
|-----------|---------|--------------|
| **Generator** | Task execution | Produces reasoning traces and outputs, explores strategies |
| **Reflector** | Outcome analysis | Analyzes successes/failures, extracts lessons and heuristics |
| **Curator** | Playbook maintenance | Applies controlled, incremental "delta edits" to context |

**Performance Results**:
- +10.6% improvement on agent benchmarks
- +8.6% improvement on finance domain
- Significantly reduced adaptation latency and rollout cost
- Achieved parity with top production agents using smaller open-source models

**Key Innovation**: Prevents "context collapse" (where detail is lost through repeated rewriting) by using structured, incremental updates that preserve detailed knowledge.

**Applied**: Will implement Reflector module for session outcome pattern extraction

---

### Research Topic 3: Hybrid Semantic + Keyword Routing

**Search Query**: "sentence embeddings task classification semantic fallback LLM routing 2025"

**Key Sources**:
1. [LLM Semantic Router - Red Hat (Published: May 2025)](https://developers.redhat.com/articles/2025/05/20/llm-semantic-router-intelligent-request-routing)
2. [Multi-LLM Routing Strategies - AWS (Published: 2025)](https://aws.amazon.com/blogs/machine-learning/multi-llm-routing-strategies-for-generative-ai-applications-on-aws/)
3. [vLLM Semantic Router (Published: Sep 2025)](https://blog.vllm.ai/2025/09/11/semantic-router.html)
4. [Signal-Decision Architecture (Published: Nov 2025)](https://blog.vllm.ai/2025/11/19/signal-decision.html)

**Key Findings - Signal-Decision Architecture**:

Three complementary signal types for routing:

| Signal Type | Technique | Advantages |
|-------------|-----------|------------|
| **Keyword** | Regex pattern matching | Zero ML overhead, human-interpretable |
| **Embedding** | Cosine similarity with sentence-transformers | Handles typos, abbreviations, fuzzy matching |
| **Domain** | MMLU-trained classification | Dataset-driven domain expertise |

**Hybrid Routing Pattern**:
```
User Query → Keyword Filter (fast, obvious cases)
           ↓ (no match)
           → Semantic Router (embedding similarity)
           ↓ (low confidence)
           → LLM Fallback (catch-all)
```

**Configuration Patterns**:
- Pre-computed embeddings for candidate phrases (offline)
- Runtime query embedding with lightweight models (sentence-transformers)
- Cosine similarity with configurable thresholds
- AND/OR boolean operators for combining signals
- Priority-based selection when multiple decisions match

**Applied**: Will implement hybrid keyword+semantic routing for task classification fallback

---

### Research Topic 4: Session Outcome Pattern Extraction

**Search Query**: "session outcome pattern extraction automated learning agent improvement 2025"

**Key Sources**:
1. [Self-Learning AI Agents - Beam.ai (Published: 2025)](https://beam.ai/agentic-insights/self-learning-ai-agents-transforming-automation-with-continuous-improvement)
2. [LLM-Based Agents for Tool Learning Survey (Published: 2025)](https://link.springer.com/article/10.1007/s41019-025-00296-9)
3. [KDD 2025 Workshop on Agentic Evaluation](https://kdd-eval-workshop.github.io/genai-evaluation-kdd2025/)

**Key Findings**:
- Self-learning agents "improve on their own by spotting patterns, learning from mistakes"
- Task mining captures not just steps taken, but **decision logic** behind successful outcomes
- QA-driven metrics with reinforcement learning enable scalable self-improvement
- Contextual Experience Replay enables self-improvement through outcome analysis

**Applied**: Will implement pattern extraction from session-outcomes.jsonl

---

## Problem Statement

**Current State (v1.1)**:
- Two-dimensional complexity scoring (depth × width) ✅
- Session outcome logging to JSONL ✅
- No automatic learning from outcomes
- Keyword-only task classification (may miss nuanced tasks)

**Gap**:
- ACE research shows +10.6% improvement through outcome-based learning
- Keyword matching misses tasks like "implement authentication" (complex but no depth/width keywords)
- No fallback for low-confidence keyword matches

**Target State (v1.2)**:
- ACE Reflector module extracts patterns from session outcomes
- Hybrid keyword+semantic routing with cascading fallback
- Auto-generated recommendations for improving task classification

---

## Phase 1: Upgrade Path

### Success Criteria

1. [x] Reflector module analyzes session outcomes and extracts failure patterns
2. [x] Semantic embeddings computed for task descriptions
3. [x] Hybrid routing: keyword → semantic → fallback
4. [x] Pattern recommendations logged for review
5. [ ] Classification accuracy improved (measured via feedback field) - needs real data
6. [x] All existing tests still pass

### Scope

**In Scope**:
- Create `scripts/ace_reflector.py` with pattern extraction
- Add semantic embedding computation to `scripts/select_prompts.py`
- Implement hybrid routing with confidence thresholds
- Create recommendations output from Reflector

**Out of Scope**:
- Full Curator module (auto-updating task-router.yaml) - defer to v1.3
- LLM-based complexity estimation - defer to v1.3
- Real-time adaptation during sessions - defer to v1.3

---

## Phase 2: Implementation Checklist

### P0 - Critical

- [x] Create `scripts/ace_reflector.py` with pattern extraction
- [x] Add sentence-transformers embedding support to select_prompts.py
- [x] Implement semantic similarity fallback routing
- [x] Create `config/semantic-routes.yaml` for embedding candidates

### P1 - Important

- [x] Implement confidence thresholds for cascading fallback
- [x] Add pattern recommendation output to Reflector
- [x] Update run_overnight.sh to call Reflector on session end
- [x] Add unit tests for new modules (tests/test_ace_reflector.py, tests/test_semantic_router.py)

### P2 - Nice to Have

- [x] Add visualization for semantic similarity scores (updated visualize_complexity)
- [ ] Create dashboard for pattern recommendations
- [ ] Add batch analysis mode for historical sessions

---

## Design: ACE Reflector Module

### Architecture

```
Session Outcomes (JSONL)
        ↓
   ┌────────────┐
   │ Reflector  │ ← Analyzes patterns
   └────────────┘
        ↓
   Pattern Analysis:
   - Misclassified tasks (classification_accurate=false)
   - Partial/failed sessions by complexity level
   - Domain-specific failure patterns
   - Keyword gaps (complex tasks with low scores)
        ↓
   ┌────────────┐
   │  Output    │ → Recommendations JSON
   └────────────┘
```

### Pattern Extraction Rules

| Pattern | Detection | Recommendation |
|---------|-----------|----------------|
| **Misclassification** | `classification_accurate=false` | Add keywords from task description |
| **Complexity Mismatch** | L1 tasks taking > 2 hours | Increase score threshold |
| **Domain Miss** | High failure rate in domain | Add domain-specific patterns |
| **Keyword Gap** | Complex task, low keyword score | Extract new depth/width indicators |

### Reflector Output Schema

```json
{
  "analysis_date": "2025-12-03T12:00:00Z",
  "sessions_analyzed": 50,
  "patterns_found": [
    {
      "pattern_type": "keyword_gap",
      "confidence": 0.85,
      "description": "Tasks containing 'authentication' frequently misclassified as L1",
      "recommendation": {
        "action": "add_l3_pattern",
        "pattern": "authentication|auth|login|oauth",
        "weight": 3,
        "justification": "Found in 5 sessions marked as misclassified"
      },
      "supporting_evidence": [
        {"session_id": "20251201-...", "task": "implement authentication..."}
      ]
    }
  ],
  "overall_metrics": {
    "classification_accuracy": 0.82,
    "success_rate_by_level": {"L1": 0.95, "L2": 0.78, "L3": 0.65}
  }
}
```

---

## Design: Hybrid Semantic Routing

### Architecture

```
Task Description
        ↓
   ┌─────────────────┐
   │ Keyword Matcher │ → High confidence? → Use keyword result
   └─────────────────┘
        ↓ (low confidence)
   ┌─────────────────┐
   │ Semantic Router │ → Above threshold? → Use semantic result
   └─────────────────┘
        ↓ (below threshold)
   ┌─────────────────┐
   │ Default (L2)    │ → Moderate complexity fallback
   └─────────────────┘
```

### Semantic Route Configuration

```yaml
# config/semantic-routes.yaml
version: "1.0.0"

embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
similarity_threshold: 0.75
fallback_level: "L1_moderate"

# Candidate phrases for each complexity level
complexity_candidates:
  L1_complex:
    - "implement new authentication system"
    - "refactor entire module architecture"
    - "create multi-step workflow"
    - "integrate multiple external services"
    - "design database schema with migrations"

  L1_moderate:
    - "add new feature to existing module"
    - "create tests for functionality"
    - "update API endpoint"
    - "fix bug in component"

  L1_simple:
    - "fix typo in documentation"
    - "update configuration value"
    - "rename variable"
    - "add comment to code"

domain_candidates:
  algorithm:
    - "trading strategy implementation"
    - "backtest performance analysis"
    - "options pricing model"

  llm:
    - "sentiment analysis prompt"
    - "agent ensemble configuration"
    - "LLM provider integration"

  infrastructure:
    - "deployment pipeline setup"
    - "docker container configuration"
    - "CI/CD workflow"
```

### Implementation Approach

```python
# Pseudocode for hybrid routing
def route_task(task: str) -> RoutingDecision:
    # Step 1: Keyword matching (fast)
    keyword_result = keyword_classify(task)

    if keyword_result.confidence >= 0.8:
        return keyword_result

    # Step 2: Semantic similarity (if keyword confidence low)
    task_embedding = embed(task)
    semantic_result = semantic_classify(task_embedding, candidates)

    if semantic_result.similarity >= THRESHOLD:
        # Combine keyword and semantic signals
        return combine_signals(keyword_result, semantic_result)

    # Step 3: Fallback to moderate complexity
    return RoutingDecision(
        complexity_level="L1_moderate",
        confidence="low",
        reasoning=["Fallback: low confidence from both keyword and semantic"]
    )
```

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/ace_reflector.py` | CREATE | Pattern extraction from session outcomes |
| `scripts/select_prompts.py` | MODIFY | Add semantic embedding support |
| `config/semantic-routes.yaml` | CREATE | Semantic routing configuration |
| `scripts/run_overnight.sh` | MODIFY | Call Reflector on session end |
| `tests/test_ace_reflector.py` | CREATE | Unit tests for Reflector |
| `tests/test_semantic_router.py` | CREATE | Unit tests for semantic routing |

---

## Technical Considerations

### Embedding Model Selection

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| all-MiniLM-L6-v2 | 22MB | Fast | Good for short text |
| all-mpnet-base-v2 | 438MB | Medium | Higher quality |
| bge-small-en-v1.5 | 130MB | Fast | State-of-art for size |

**Recommendation**: Start with `all-MiniLM-L6-v2` for low latency, upgrade if needed.

### Dependency Considerations

New dependencies needed:
- `sentence-transformers>=2.2.0`
- `torch` (CPU-only for inference)

Keep embedding computation optional - fall back to keyword-only if dependencies unavailable.

---

## Changelog

| Date | Change |
|------|--------|
| 2025-12-03 | Initial UPGRADE-012.2 research document created |
| 2025-12-03 | Phase 0 research completed with 4 topic areas |
| 2025-12-03 | Phase 1 upgrade path defined |
| 2025-12-03 | Phase 2 implementation checklist created |
| 2025-12-03 | P0: Created ace_reflector.py with pattern extraction and analysis |
| 2025-12-03 | P0: Added SemanticRouter class to select_prompts.py |
| 2025-12-03 | P0: Implemented hybrid keyword+semantic routing with fallback |
| 2025-12-03 | P0: Created config/semantic-routes.yaml with complexity/domain candidates |
| 2025-12-03 | P1: Added confidence thresholds for cascading fallback |
| 2025-12-03 | P1: Integrated ACE Reflector with run_overnight.sh cleanup |
| 2025-12-03 | P2: Updated visualization to show keyword/semantic confidence |
| 2025-12-03 | **P0+P1 COMPLETE** - UPGRADE-012.2 v1.2 implemented |
| 2025-12-03 | P1: Added unit tests (test_ace_reflector.py, test_semantic_router.py) |
| 2025-12-03 | P1: Added sentence-transformers to requirements.txt |
| 2025-12-03 | **ALL P0+P1 COMPLETE** - UPGRADE-012.2 finalized via RIC Loop |
