#!/usr/bin/env python3
"""
Task Router for Hierarchical Prompt System (UPGRADE-012).

This script analyzes a task description and selects appropriate prompts based on:
1. Override prefixes (SIMPLE:, MODERATE:, COMPLEX:, DOMAIN:xxx:)
2. Complexity classification (rule-based keyword matching)
3. Domain detection (keyword matching)

Usage:
    python scripts/select_prompts.py "Implement feature X"
    python scripts/select_prompts.py "COMPLEX: Major refactoring"
    python scripts/select_prompts.py --json "Add tests for module Y"

Output:
    Combined prompt text (default) or JSON routing decision (--json)

Design Principles (from research):
1. Start Simple - L1 is default, escalate only when needed (Anthropic)
2. Minimal Context - High-signal tokens only (Anthropic)
3. Rule-Based Routing - Keyword matching for v1 (Patronus AI)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml


# Optional: sentence-transformers for semantic routing (v1.2)
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False


@dataclass
class RoutingDecision:
    """Result of task routing analysis."""

    task_description: str
    original_task: str
    complexity_level: str  # L1_simple, L1_moderate, L1_complex
    complexity_score: int
    depth_score: float  # v1.1: Sequential reasoning chain length
    width_score: float  # v1.1: Capability diversity
    complexity_reasoning: list[str]
    domain: str
    domain_reasoning: list[str]
    override_used: bool
    override_type: str | None  # complexity, domain, or None
    prompts_selected: list[str]
    safety_checks: list[str]
    # v1.2: Semantic routing fields
    keyword_confidence: float = 1.0  # Confidence from keyword matching
    semantic_confidence: float = 0.0  # Confidence from semantic similarity
    routing_method: str = "keyword"  # keyword, semantic, hybrid, fallback
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_description": self.task_description,
            "original_task": self.original_task,
            "complexity_level": self.complexity_level,
            "complexity_score": self.complexity_score,
            "depth_score": self.depth_score,
            "width_score": self.width_score,
            "complexity_reasoning": self.complexity_reasoning,
            "domain": self.domain,
            "domain_reasoning": self.domain_reasoning,
            "override_used": self.override_used,
            "override_type": self.override_type,
            "prompts_selected": self.prompts_selected,
            "safety_checks": self.safety_checks,
            "keyword_confidence": self.keyword_confidence,
            "semantic_confidence": self.semantic_confidence,
            "routing_method": self.routing_method,
            "timestamp": self.timestamp,
        }


class SemanticRouter:
    """Semantic similarity-based routing using sentence embeddings (v1.2).

    Implements the embedding signal from Signal-Decision Architecture:
    https://blog.vllm.ai/2025/11/19/signal-decision.html

    Falls back gracefully if sentence-transformers is not installed.
    """

    def __init__(self, config_path: Path | None = None):
        """Initialize semantic router.

        Args:
            config_path: Path to semantic-routes.yaml config.
        """
        self.available = SEMANTIC_AVAILABLE
        self.model = None
        self.config = {}
        self.candidate_embeddings: dict[str, dict] = {}

        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "semantic-routes.yaml"

        self.config_path = config_path
        if config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f) or {}

    def _ensure_model(self) -> bool:
        """Load the embedding model if not already loaded.

        Returns:
            True if model is available, False otherwise.
        """
        if not self.available:
            return False

        if self.model is None:
            model_name = self.config.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.available = False
                return False

        return True

    def _get_candidate_embeddings(self, category: str) -> dict[str, list[float]]:
        """Get or compute embeddings for candidate phrases.

        Args:
            category: "complexity" or "domain"

        Returns:
            Dictionary mapping level/domain to list of embeddings.
        """
        if category in self.candidate_embeddings:
            return self.candidate_embeddings[category]

        if not self._ensure_model():
            return {}

        candidates_config = self.config.get(f"{category}_candidates", {})
        embeddings = {}

        for level, level_info in candidates_config.items():
            examples = level_info.get("examples", [])
            if examples:
                # Compute embeddings for all examples
                level_embeddings = self.model.encode(examples)
                embeddings[level] = level_embeddings

        self.candidate_embeddings[category] = embeddings
        return embeddings

    def classify_complexity(self, task: str) -> tuple[str, float]:
        """Classify task complexity using semantic similarity.

        Args:
            task: Task description to classify.

        Returns:
            Tuple of (complexity_level, similarity_score).
        """
        if not self._ensure_model():
            return "L1_moderate", 0.0

        # Get task embedding
        task_embedding = self.model.encode([task])[0]

        # Get candidate embeddings
        candidates = self._get_candidate_embeddings("complexity")
        if not candidates:
            return "L1_moderate", 0.0

        best_level = "L1_moderate"
        best_similarity = 0.0

        for level, level_embeddings in candidates.items():
            # Compute cosine similarity with all examples
            similarities = np.dot(level_embeddings, task_embedding) / (
                np.linalg.norm(level_embeddings, axis=1) * np.linalg.norm(task_embedding)
            )
            max_similarity = float(np.max(similarities))

            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_level = level

        return best_level, best_similarity

    def classify_domain(self, task: str) -> tuple[str, float]:
        """Classify task domain using semantic similarity.

        Args:
            task: Task description to classify.

        Returns:
            Tuple of (domain, similarity_score).
        """
        if not self._ensure_model():
            return "general", 0.0

        # Get task embedding
        task_embedding = self.model.encode([task])[0]

        # Get candidate embeddings
        candidates = self._get_candidate_embeddings("domain")
        if not candidates:
            return "general", 0.0

        best_domain = "general"
        best_similarity = 0.0

        for domain, domain_embeddings in candidates.items():
            similarities = np.dot(domain_embeddings, task_embedding) / (
                np.linalg.norm(domain_embeddings, axis=1) * np.linalg.norm(task_embedding)
            )
            max_similarity = float(np.max(similarities))

            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_domain = domain

        return best_domain, best_similarity


class TaskRouter:
    """Routes tasks to appropriate prompts based on complexity and domain.

    Supports hybrid keyword+semantic routing (v1.2):
    1. Keyword matching (fast, rule-based)
    2. Semantic similarity (fallback for low-confidence keyword matches)
    3. Default fallback to moderate complexity
    """

    def __init__(
        self,
        config_path: Path | None = None,
        enable_semantic: bool = True,
    ):
        """Initialize router with configuration.

        Args:
            config_path: Path to task-router.yaml. Defaults to config/task-router.yaml.
            enable_semantic: Enable semantic fallback routing (v1.2).
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "task-router.yaml"

        self.config_path = config_path
        self.config = self._load_config()
        self.project_root = Path(__file__).parent.parent

        # v1.2: Initialize semantic router if enabled
        self.enable_semantic = enable_semantic and SEMANTIC_AVAILABLE
        self.semantic_router: SemanticRouter | None = None
        if self.enable_semantic:
            semantic_config = self.project_root / "config" / "semantic-routes.yaml"
            self.semantic_router = SemanticRouter(semantic_config)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def route(self, task_description: str) -> RoutingDecision:
        """Analyze task and determine routing.

        Args:
            task_description: The task to analyze.

        Returns:
            RoutingDecision with complexity, domain, and prompt selection.
        """
        original_task = task_description
        override_used = False
        override_type = None
        forced_level = None
        forced_domain = None

        # Step 1: Check for overrides
        task_description, forced_level, override_type = self._check_complexity_override(task_description)
        if forced_level:
            override_used = True

        task_description, forced_domain, domain_override = self._check_domain_override(task_description)
        if forced_domain:
            override_used = True
            override_type = override_type or "domain"

        # v1.2: Initialize routing confidence tracking
        keyword_confidence = 1.0
        semantic_confidence = 0.0
        routing_method = "keyword"

        # Step 2: Classify complexity (if not overridden)
        if forced_level:
            complexity_level = forced_level
            complexity_score = 0
            depth_score = 0.0
            width_score = 0.0
            complexity_reasoning = [f"Override: forced to {forced_level}"]
            routing_method = "override"
        else:
            complexity_level, complexity_score, depth_score, width_score, complexity_reasoning = (
                self._classify_complexity(task_description)
            )

            # v1.2: Calculate keyword confidence based on score
            # High score = high confidence, low score = may need semantic fallback
            complexity_config = self.config.get("complexity", {})
            l3_threshold = complexity_config.get("L3_triggers", {}).get("threshold", 5)
            l2_threshold = complexity_config.get("L2_triggers", {}).get("threshold", 2)

            if complexity_score >= l3_threshold:
                keyword_confidence = min(0.5 + complexity_score * 0.1, 1.0)
            elif complexity_score >= l2_threshold:
                keyword_confidence = 0.5 + (complexity_score - l2_threshold) * 0.1
            else:
                keyword_confidence = 0.3 + complexity_score * 0.1

            # v1.2: Use semantic fallback if keyword confidence is low
            if self.semantic_router and keyword_confidence < 0.7:
                semantic_level, semantic_confidence = self.semantic_router.classify_complexity(task_description)

                # Thresholds from semantic-routes.yaml
                semantic_threshold = 0.70
                if self.semantic_router.config:
                    semantic_threshold = self.semantic_router.config.get("thresholds", {}).get(
                        "semantic_similarity_threshold", 0.70
                    )

                if semantic_confidence >= semantic_threshold:
                    # Semantic confidence high enough to consider
                    if semantic_confidence > keyword_confidence:
                        # Semantic wins - use it
                        complexity_level = semantic_level
                        complexity_reasoning.append(
                            f"Semantic override: {semantic_level} "
                            f"(similarity={semantic_confidence:.2f} > keyword={keyword_confidence:.2f})"
                        )
                        routing_method = "semantic"
                    else:
                        # Combine signals
                        complexity_reasoning.append(
                            f"Semantic confirmed: {semantic_level} " f"(similarity={semantic_confidence:.2f})"
                        )
                        routing_method = "hybrid"
                elif keyword_confidence < 0.5:
                    # Both low confidence - use fallback
                    fallback_level = "L1_moderate"
                    if self.semantic_router.config:
                        fallback_level = self.semantic_router.config.get("thresholds", {}).get(
                            "fallback_level", "L1_moderate"
                        )
                    complexity_level = fallback_level
                    complexity_reasoning.append(
                        f"Fallback to {fallback_level}: low confidence "
                        f"(keyword={keyword_confidence:.2f}, semantic={semantic_confidence:.2f})"
                    )
                    routing_method = "fallback"

        # Step 3: Detect domain (if not overridden)
        if forced_domain:
            domain = forced_domain
            domain_reasoning = [f"Override: forced to {forced_domain}"]
        else:
            domain, domain_reasoning = self._detect_domain(task_description)

            # v1.2: Try semantic domain detection if keyword detection returned general
            if domain == "general" and self.semantic_router:
                semantic_domain, domain_similarity = self.semantic_router.classify_domain(task_description)
                if domain_similarity >= 0.65 and semantic_domain != "general":
                    domain = semantic_domain
                    domain_reasoning.append(
                        f"Semantic domain: {semantic_domain} " f"(similarity={domain_similarity:.2f})"
                    )

        # Step 4: Select prompts
        prompts_selected = self._select_prompts(complexity_level, domain)

        # Step 5: Gather safety checks
        safety_checks = self._get_safety_checks(domain)

        return RoutingDecision(
            task_description=task_description.strip(),
            original_task=original_task,
            complexity_level=complexity_level,
            complexity_score=complexity_score,
            depth_score=depth_score,
            width_score=width_score,
            complexity_reasoning=complexity_reasoning,
            domain=domain,
            domain_reasoning=domain_reasoning,
            override_used=override_used,
            override_type=override_type,
            prompts_selected=prompts_selected,
            safety_checks=safety_checks,
            keyword_confidence=keyword_confidence,
            semantic_confidence=semantic_confidence,
            routing_method=routing_method,
        )

    def _check_complexity_override(self, task: str) -> tuple[str, str | None, str | None]:
        """Check for complexity override prefix.

        Returns:
            Tuple of (cleaned_task, forced_level, override_type)
        """
        overrides = self.config.get("overrides", {}).get("complexity", [])

        for override in overrides:
            pattern = override["pattern"]
            if re.match(pattern, task, re.IGNORECASE):
                level = override["level"]
                if override.get("strip_prefix", False):
                    # Remove the prefix
                    task = re.sub(pattern, "", task, flags=re.IGNORECASE).strip()
                return task, level, "complexity"

        return task, None, None

    def _check_domain_override(self, task: str) -> tuple[str, str | None, str | None]:
        """Check for domain override prefix.

        Returns:
            Tuple of (cleaned_task, forced_domain, override_type)
        """
        overrides = self.config.get("overrides", {}).get("domain", [])

        for override in overrides:
            pattern = override["pattern"]
            match = re.search(pattern, task, re.IGNORECASE)
            if match:
                # Extract domain name from capture group
                domain = match.group(1) if match.groups() else None
                if domain and override.get("strip_prefix", False):
                    # Remove the prefix
                    task = re.sub(f"DOMAIN:{domain}:", "", task, flags=re.IGNORECASE).strip()
                return task, domain, "domain"

        return task, None, None

    def _classify_complexity(self, task: str) -> tuple[str, int, float, float, list[str]]:
        """Classify task complexity using rule-based patterns with depth+width (v1.1).

        Returns:
            Tuple of (complexity_level, score, depth_score, width_score, reasoning)
        """
        task_lower = task.lower()
        base_score = 0
        depth_raw = 0
        width_raw = 0
        reasoning = []

        complexity_config = self.config.get("complexity", {})

        # Check L3 triggers
        l3_config = complexity_config.get("L3_triggers", {})
        l3_threshold = l3_config.get("threshold", 5)
        for pattern_info in l3_config.get("patterns", []):
            pattern = pattern_info["pattern"]
            weight = pattern_info["weight"]
            if re.search(pattern, task_lower):
                base_score += weight
                reasoning.append(f"+{weight}: matched L3 pattern '{pattern}'")

        # Check L2 triggers
        l2_config = complexity_config.get("L2_triggers", {})
        l2_threshold = l2_config.get("threshold", 2)
        for pattern_info in l2_config.get("patterns", []):
            pattern = pattern_info["pattern"]
            weight = pattern_info["weight"]
            if re.search(pattern, task_lower):
                base_score += weight
                reasoning.append(f"+{weight}: matched L2 pattern '{pattern}'")

        # Check L1 indicators (negative weight reduces score)
        l1_config = complexity_config.get("L1_indicators", {})
        for pattern_info in l1_config.get("patterns", []):
            pattern = pattern_info["pattern"]
            weight = pattern_info["weight"]
            if re.search(pattern, task_lower):
                base_score += weight
                reasoning.append(f"{weight}: matched L1 pattern '{pattern}'")

        # v1.1: Calculate depth score (sequential reasoning chain)
        depth_config = self.config.get("depth_indicators", {})
        depth_multiplier = depth_config.get("multiplier", 1.5)
        for pattern_info in depth_config.get("patterns", []):
            pattern = pattern_info["pattern"]
            weight = pattern_info["weight"]
            if re.search(pattern, task_lower):
                depth_raw += weight
                reasoning.append(f"+{weight} depth: matched '{pattern}'")

        # v1.1: Calculate width score (capability diversity)
        width_config = self.config.get("width_indicators", {})
        width_multiplier = width_config.get("multiplier", 1.2)
        for pattern_info in width_config.get("patterns", []):
            pattern = pattern_info["pattern"]
            weight = pattern_info["weight"]
            if re.search(pattern, task_lower):
                width_raw += weight
                reasoning.append(f"+{weight} width: matched '{pattern}'")

        # Apply multipliers to get final dimensional scores
        depth_score = depth_raw * depth_multiplier
        width_score = width_raw * width_multiplier

        # Combined score: base + depth contribution + width contribution
        # Research shows depth has more pronounced effect
        combined_score = base_score + depth_score + width_score
        final_score = int(round(combined_score))

        # Log dimensional analysis
        if depth_raw > 0 or width_raw > 0:
            reasoning.append(
                f"Dimensions: depth={depth_raw}×{depth_multiplier}={depth_score:.1f}, "
                f"width={width_raw}×{width_multiplier}={width_score:.1f}"
            )

        # Determine level based on combined score
        if final_score >= l3_threshold:
            level = "L1_complex"
            reasoning.append(f"Final: L3 (Complex) - combined score {final_score} >= {l3_threshold}")
        elif final_score >= l2_threshold:
            level = "L1_moderate"
            reasoning.append(f"Final: L2 (Moderate) - combined score {final_score} >= {l2_threshold}")
        else:
            level = "L1_simple"
            reasoning.append(f"Final: L1 (Simple) - combined score {final_score} < {l2_threshold}")

        return level, final_score, depth_score, width_score, reasoning

    def _detect_domain(self, task: str) -> tuple[str, list[str]]:
        """Detect task domain using keyword matching.

        Returns:
            Tuple of (domain, reasoning)
        """
        task_lower = task.lower()
        domains_config = self.config.get("domains", {})
        reasoning = []

        domain_scores: dict[str, int] = {}

        for domain_name, domain_info in domains_config.items():
            if domain_name == "general":
                continue  # Skip default domain

            keywords = domain_info.get("keywords", [])
            for pattern in keywords:
                if re.search(pattern, task_lower):
                    domain_scores[domain_name] = domain_scores.get(domain_name, 0) + 1
                    reasoning.append(f"Domain '{domain_name}': matched '{pattern}'")

        if domain_scores:
            # Return domain with highest match count
            best_domain = max(domain_scores, key=domain_scores.get)
            reasoning.append(f"Selected domain: {best_domain}")
            return best_domain, reasoning
        else:
            reasoning.append("No domain keywords matched, using 'general'")
            return "general", reasoning

    def _select_prompts(self, complexity_level: str, domain: str) -> list[str]:
        """Select prompt files based on complexity and domain.

        Returns:
            List of prompt file paths.
        """
        prompts = []

        # Add complexity prompt
        prompts_config = self.config.get("prompts", {}).get("complexity", {})
        complexity_prompt = prompts_config.get(complexity_level)
        if complexity_prompt:
            prompts.append(complexity_prompt)

        # Add domain prompt
        domains_config = self.config.get("domains", {})
        domain_info = domains_config.get(domain, {})
        domain_prompt = domain_info.get("prompt_file")
        if domain_prompt:
            prompts.append(domain_prompt)

        return prompts

    def _get_safety_checks(self, domain: str) -> list[str]:
        """Get safety checks for the detected domain.

        Returns:
            List of safety check descriptions.
        """
        domains_config = self.config.get("domains", {})
        domain_info = domains_config.get(domain, {})
        return domain_info.get("safety_checks", [])

    def assemble_prompt(self, decision: RoutingDecision, task: str) -> str:
        """Assemble the final prompt from selected templates.

        Args:
            decision: Routing decision with selected prompts.
            task: Original task description.

        Returns:
            Combined prompt text.
        """
        parts = []

        # Load and combine prompt templates
        for prompt_path in decision.prompts_selected:
            full_path = self.project_root / prompt_path
            if full_path.exists():
                content = full_path.read_text()
                parts.append(content)
            else:
                parts.append(f"<!-- Prompt not found: {prompt_path} -->")

        # Add routing metadata as comment
        routing_info = f"""
<!-- Task Routing Decision (UPGRADE-012 v1.2)
Complexity: {decision.complexity_level} (combined score: {decision.complexity_score})
Dimensions: depth={decision.depth_score:.1f}, width={decision.width_score:.1f}
Domain: {decision.domain}
Override: {decision.override_type if decision.override_used else 'none'}
Routing: method={decision.routing_method}, keyword={decision.keyword_confidence:.0%}, semantic={decision.semantic_confidence:.0%}
Prompts: {', '.join(decision.prompts_selected)}
-->
"""
        parts.insert(0, routing_info)

        # Add task description
        parts.append(f"\n## Your Task\n\n{task}\n")

        # Add safety checks if any
        if decision.safety_checks:
            safety_section = "\n## Safety Checks Required\n\n"
            for check in decision.safety_checks:
                safety_section += f"- [ ] {check}\n"
            parts.append(safety_section)

        return "\n".join(parts)

    def log_decision(self, decision: RoutingDecision) -> None:
        """Log routing decision to file and session notes.

        Args:
            decision: The routing decision to log.
        """
        logging_config = self.config.get("logging", {})
        if not logging_config.get("enabled", True):
            return

        # Log to JSONL file
        log_file = logging_config.get("log_file", "logs/task-routing.jsonl")
        log_path = self.project_root / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "a") as f:
            f.write(json.dumps(decision.to_dict()) + "\n")

        # Log to session notes if enabled
        if logging_config.get("log_to_session_notes", True):
            notes_path = self.project_root / "claude-session-notes.md"
            if notes_path.exists():
                content = notes_path.read_text()
                # Add routing info to Technical Notes section
                routing_note = f"\n- Task routed: {decision.complexity_level}, domain={decision.domain}"
                if "## Technical Notes" in content:
                    content = content.replace(
                        "## Technical Notes",
                        f"## Technical Notes{routing_note}",
                    )
                    notes_path.write_text(content)


def visualize_complexity(decision: RoutingDecision) -> str:
    """Generate ASCII visualization of depth/width scores.

    Args:
        decision: The routing decision to visualize.

    Returns:
        ASCII art visualization string.
    """
    # Normalize scores to 0-10 scale for visualization
    max_display = 10
    depth_bars = min(int(decision.depth_score), max_display)
    width_bars = min(int(decision.width_score), max_display)
    score_bars = min(decision.complexity_score, max_display)
    kw_conf_bars = min(int(decision.keyword_confidence * 10), max_display)
    sem_conf_bars = min(int(decision.semantic_confidence * 10), max_display)

    # Create bar charts
    depth_bar = "█" * depth_bars + "░" * (max_display - depth_bars)
    width_bar = "█" * width_bars + "░" * (max_display - width_bars)
    score_bar = "█" * score_bars + "░" * (max_display - score_bars)
    kw_conf_bar = "█" * kw_conf_bars + "░" * (max_display - kw_conf_bars)
    sem_conf_bar = "█" * sem_conf_bars + "░" * (max_display - sem_conf_bars)

    # Level indicator
    level_map = {
        "L1_simple": ("🟢", "SIMPLE"),
        "L1_moderate": ("🟡", "MODERATE"),
        "L1_complex": ("🔴", "COMPLEX"),
    }
    emoji, level_name = level_map.get(decision.complexity_level, ("⚪", "UNKNOWN"))

    # Routing method indicator
    method_map = {
        "keyword": "🔤 Keyword",
        "semantic": "🧠 Semantic",
        "hybrid": "🔀 Hybrid",
        "fallback": "⚠️ Fallback",
        "override": "🎯 Override",
    }
    method_display = method_map.get(decision.routing_method, decision.routing_method)

    output = f"""
┌─────────────────────────────────────────────────────────┐
│  TASK COMPLEXITY ANALYSIS (UPGRADE-012 v1.2)            │
├─────────────────────────────────────────────────────────┤
│  Task: {decision.task_description[:45]:<45} │
├─────────────────────────────────────────────────────────┤
│  Depth Score:  [{depth_bar}] {decision.depth_score:>5.1f}          │
│  Width Score:  [{width_bar}] {decision.width_score:>5.1f}          │
│  Total Score:  [{score_bar}] {decision.complexity_score:>5}          │
├─────────────────────────────────────────────────────────┤
│  Keyword Conf: [{kw_conf_bar}] {decision.keyword_confidence:>5.0%}          │
│  Semantic Conf:[{sem_conf_bar}] {decision.semantic_confidence:>5.0%}          │
├─────────────────────────────────────────────────────────┤
│  Result: {emoji} {level_name:<10} │ Domain: {decision.domain:<15} │
│  Method: {method_display:<14} │                          │
└─────────────────────────────────────────────────────────┘
"""
    return output


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Route task to appropriate prompts based on complexity and domain.")
    parser.add_argument("task", help="Task description to analyze")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output routing decision as JSON instead of combined prompt",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Show visual depth/width analysis chart",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to task-router.yaml config file",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable logging of routing decision",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic fallback routing (v1.2)",
    )

    args = parser.parse_args()

    try:
        router = TaskRouter(
            config_path=args.config,
            enable_semantic=not args.no_semantic,
        )
        decision = router.route(args.task)

        if not args.no_log:
            router.log_decision(decision)

        if args.visualize:
            print(visualize_complexity(decision))
        elif args.json:
            print(json.dumps(decision.to_dict(), indent=2))
        else:
            prompt = router.assemble_prompt(decision, args.task)
            print(prompt)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error routing task: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
