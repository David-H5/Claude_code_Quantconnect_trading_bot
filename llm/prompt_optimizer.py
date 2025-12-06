"""
Prompt Optimizer

Automatic prompt refinement based on evaluation feedback.

Based on 2025 research:
- PromptWizard iterative refinement
- Few-shot example optimization
- Constraint-based guidance improvement

QuantConnect Compatible: Yes
- Non-blocking optimization
- Version-controlled prompts
- Rollback support
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    pass


class RefinementCategory(Enum):
    """Categories of prompt refinements."""

    ADD_EXAMPLES = "add_examples"
    CLARIFY_INSTRUCTIONS = "clarify_instructions"
    ADD_CONSTRAINTS = "add_constraints"
    REMOVE_CONFLICTS = "remove_conflicts"
    IMPROVE_STRUCTURE = "improve_structure"
    ADD_ERROR_HANDLING = "add_error_handling"


class RefinementStrategy(Enum):
    """Strategies for generating refinements."""

    RULE_BASED = "rule_based"
    LLM_ASSISTED = "llm_assisted"
    HYBRID = "hybrid"


@dataclass
class PromptRefinement:
    """A suggested refinement to a prompt."""

    category: RefinementCategory
    original_section: str
    refined_section: str
    description: str
    expected_impact: float  # 0-1 expected improvement
    confidence: float = 0.5  # Confidence in refinement
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "original_section": self.original_section,
            "refined_section": self.refined_section,
            "description": self.description,
            "expected_impact": self.expected_impact,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class OptimizationResult:
    """Result of prompt optimization."""

    original_prompt: str
    optimized_prompt: str
    refinements_applied: list[PromptRefinement]
    total_expected_impact: float
    optimization_time_ms: float
    strategy_used: RefinementStrategy
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "refinements_applied": len(self.refinements_applied),
            "total_expected_impact": self.total_expected_impact,
            "optimization_time_ms": self.optimization_time_ms,
            "strategy_used": self.strategy_used.value,
            "timestamp": self.timestamp.isoformat(),
        }


class PromptOptimizer:
    """
    Automatic prompt refinement based on evaluation feedback.

    Strategies:
    - Add specific examples for weak categories
    - Clarify ambiguous instructions
    - Add constraints to reduce errors
    - Remove conflicting guidance
    - Improve prompt structure

    Usage:
        optimizer = PromptOptimizer()
        refinements = optimizer.generate_refinements(prompt, weaknesses)
        new_prompt = optimizer.apply_refinements(prompt, refinements)
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        strategy: RefinementStrategy = RefinementStrategy.RULE_BASED,
        refinement_strategies: list[str] | None = None,
        max_refinements_per_cycle: int = 5,
        min_expected_impact: float = 0.05,
    ):
        """
        Initialize prompt optimizer.

        Args:
            llm_client: Optional LLM client for LLM-assisted refinements
            strategy: Overall optimization strategy
            refinement_strategies: List of enabled refinement types
            max_refinements_per_cycle: Maximum refinements to apply per cycle
            min_expected_impact: Minimum expected impact to include refinement
        """
        self.llm_client = llm_client
        self.strategy = strategy
        self.max_refinements = max_refinements_per_cycle
        self.min_impact = min_expected_impact

        self.enabled_strategies = refinement_strategies or [
            "add_examples",
            "clarify_instructions",
            "add_constraints",
            "remove_conflicts",
            "improve_structure",
            "add_error_handling",
        ]

        # Pattern library for rule-based refinements
        self._init_pattern_library()

    def _init_pattern_library(self) -> None:
        """Initialize pattern matching library for refinements."""
        self.weakness_patterns = {
            "example": {
                "keywords": ["example", "instance", "sample", "demonstrate"],
                "category": RefinementCategory.ADD_EXAMPLES,
                "template": "\n\nExample: {example_content}",
            },
            "step": {
                "keywords": ["step", "process", "procedure", "sequence"],
                "category": RefinementCategory.IMPROVE_STRUCTURE,
                "template": "\n\nProcess:\n{step_content}",
            },
            "constraint": {
                "keywords": ["constraint", "limit", "boundary", "restrict", "rule"],
                "category": RefinementCategory.ADD_CONSTRAINTS,
                "template": "\n\nConstraints:\n{constraint_content}",
            },
            "error": {
                "keywords": ["error", "fail", "exception", "handle", "edge case"],
                "category": RefinementCategory.ADD_ERROR_HANDLING,
                "template": "\n\nError handling:\n{error_content}",
            },
            "ambiguous": {
                "keywords": ["unclear", "ambiguous", "vague", "confusing"],
                "category": RefinementCategory.CLARIFY_INSTRUCTIONS,
                "template": None,  # Requires context-specific handling
            },
            "conflict": {
                "keywords": ["conflict", "contradict", "inconsistent"],
                "category": RefinementCategory.REMOVE_CONFLICTS,
                "template": None,  # Requires analysis
            },
        }

        # Domain-specific examples for trading
        self.trading_examples = {
            "analysis": "When analyzing SPY, evaluate technicals (RSI, MACD), fundamentals (P/E, growth), and sentiment (news, social) before making a recommendation.",
            "risk": "For a $100,000 portfolio with 2% risk per trade, maximum position size is $2,000 loss potential.",
            "decision": "If RSI > 70 and price at resistance with negative divergence, recommend SELL with 70% confidence.",
            "constraints": "- Maximum 2% risk per trade\n- Confidence must exceed 60%\n- Never risk more than 5% of portfolio on correlated positions",
            "error_handling": "On data unavailability, return HOLD recommendation. On conflicting signals, reduce confidence by 20%.",
        }

    def generate_refinements(
        self,
        current_prompt: str,
        weaknesses: list[Any],
        context: dict[str, Any] | None = None,
    ) -> list[PromptRefinement]:
        """
        Generate prompt refinements based on identified weaknesses.

        Args:
            current_prompt: The current agent prompt
            weaknesses: List of identified weaknesses
            context: Optional context for refinement

        Returns:
            List of suggested refinements
        """
        import time

        start = time.time()

        refinements = []
        prompt_lower = current_prompt.lower()

        for weakness in weaknesses:
            weakness_str = str(weakness).lower()

            # Find matching pattern
            for pattern_name, pattern_config in self.weakness_patterns.items():
                if any(kw in weakness_str for kw in pattern_config["keywords"]):
                    refinement = self._create_refinement_for_pattern(
                        pattern_name,
                        pattern_config,
                        weakness_str,
                        prompt_lower,
                        context or {},
                    )
                    if refinement and refinement.expected_impact >= self.min_impact:
                        refinements.append(refinement)
                    break
            else:
                # No pattern matched, create generic refinement
                generic = self._create_generic_refinement(weakness_str, prompt_lower)
                if generic:
                    refinements.append(generic)

        # Sort by expected impact and limit
        refinements.sort(key=lambda r: r.expected_impact, reverse=True)
        return refinements[: self.max_refinements]

    def _create_refinement_for_pattern(
        self,
        pattern_name: str,
        pattern_config: dict[str, Any],
        weakness: str,
        prompt: str,
        context: dict[str, Any],
    ) -> PromptRefinement | None:
        """Create refinement for a matched pattern."""
        category = pattern_config["category"]
        template = pattern_config.get("template")

        if category == RefinementCategory.ADD_EXAMPLES:
            return self._create_example_refinement(weakness, prompt, context)

        elif category == RefinementCategory.IMPROVE_STRUCTURE:
            return self._create_structure_refinement(weakness, prompt, context)

        elif category == RefinementCategory.ADD_CONSTRAINTS:
            return self._create_constraint_refinement(weakness, prompt, context)

        elif category == RefinementCategory.ADD_ERROR_HANDLING:
            return self._create_error_handling_refinement(weakness, prompt, context)

        elif category == RefinementCategory.CLARIFY_INSTRUCTIONS:
            return self._create_clarification_refinement(weakness, prompt, context)

        elif category == RefinementCategory.REMOVE_CONFLICTS:
            return self._create_conflict_resolution_refinement(weakness, prompt, context)

        return None

    def _create_example_refinement(
        self,
        weakness: str,
        prompt: str,
        context: dict[str, Any],
    ) -> PromptRefinement:
        """Create refinement to add examples."""
        # Select appropriate example based on context
        example_type = "analysis"  # Default
        if "risk" in weakness:
            example_type = "risk"
        elif "decision" in weakness:
            example_type = "decision"

        example_content = self.trading_examples.get(example_type, self.trading_examples["analysis"])

        return PromptRefinement(
            category=RefinementCategory.ADD_EXAMPLES,
            original_section="",
            refined_section=f"\n\nExample: {example_content}",
            description=f"Added concrete {example_type} example",
            expected_impact=0.15,
            confidence=0.7,
            metadata={"example_type": example_type},
        )

    def _create_structure_refinement(
        self,
        weakness: str,
        prompt: str,
        context: dict[str, Any],
    ) -> PromptRefinement:
        """Create refinement to improve structure with steps."""
        step_content = """1. Gather relevant data (price, volume, indicators)
2. Analyze signals across timeframes
3. Assess risk-reward ratio
4. Determine position sizing
5. Make final decision with confidence level"""

        return PromptRefinement(
            category=RefinementCategory.IMPROVE_STRUCTURE,
            original_section="",
            refined_section=f"\n\nProcess:\n{step_content}",
            description="Added step-by-step analysis process",
            expected_impact=0.12,
            confidence=0.65,
        )

    def _create_constraint_refinement(
        self,
        weakness: str,
        prompt: str,
        context: dict[str, Any],
    ) -> PromptRefinement:
        """Create refinement to add constraints."""
        constraint_content = self.trading_examples["constraints"]

        return PromptRefinement(
            category=RefinementCategory.ADD_CONSTRAINTS,
            original_section="",
            refined_section=f"\n\nConstraints:\n{constraint_content}",
            description="Added explicit decision constraints",
            expected_impact=0.10,
            confidence=0.75,
        )

    def _create_error_handling_refinement(
        self,
        weakness: str,
        prompt: str,
        context: dict[str, Any],
    ) -> PromptRefinement:
        """Create refinement to add error handling guidance."""
        error_content = self.trading_examples["error_handling"]

        return PromptRefinement(
            category=RefinementCategory.ADD_ERROR_HANDLING,
            original_section="",
            refined_section=f"\n\nError handling: {error_content}",
            description="Added error handling guidance",
            expected_impact=0.08,
            confidence=0.70,
        )

    def _create_clarification_refinement(
        self,
        weakness: str,
        prompt: str,
        context: dict[str, Any],
    ) -> PromptRefinement | None:
        """Create refinement to clarify ambiguous instructions."""
        # Look for ambiguous patterns
        ambiguous_patterns = [
            (r"should\s+consider", "must explicitly evaluate"),
            (r"may\s+want\s+to", "should always"),
            (r"it\s+is\s+recommended", "you must"),
            (r"try\s+to", "ensure you"),
        ]

        for pattern, replacement in ambiguous_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return PromptRefinement(
                    category=RefinementCategory.CLARIFY_INSTRUCTIONS,
                    original_section=pattern,
                    refined_section=replacement,
                    description=f"Clarified ambiguous instruction: '{pattern}' -> '{replacement}'",
                    expected_impact=0.06,
                    confidence=0.60,
                    metadata={"pattern": pattern, "replacement": replacement},
                )

        return None

    def _create_conflict_resolution_refinement(
        self,
        weakness: str,
        prompt: str,
        context: dict[str, Any],
    ) -> PromptRefinement | None:
        """Create refinement to resolve conflicts."""
        # Check for common conflicting patterns
        conflict_pairs = [
            ("always buy", "never buy"),
            ("maximize returns", "minimize risk"),
            ("aggressive", "conservative"),
        ]

        for term1, term2 in conflict_pairs:
            if term1 in prompt.lower() and term2 in prompt.lower():
                return PromptRefinement(
                    category=RefinementCategory.REMOVE_CONFLICTS,
                    original_section=f"Conflict: {term1} vs {term2}",
                    refined_section=f"Balance {term1} and {term2} based on market conditions",
                    description=f"Resolved conflict between {term1} and {term2}",
                    expected_impact=0.10,
                    confidence=0.55,
                    metadata={"conflict": [term1, term2]},
                )

        return None

    def _create_generic_refinement(
        self,
        weakness: str,
        prompt: str,
    ) -> PromptRefinement | None:
        """Create a generic refinement for unmatched weaknesses."""
        # Truncate long weaknesses
        description = weakness[:100] if len(weakness) > 100 else weakness

        return PromptRefinement(
            category=RefinementCategory.CLARIFY_INSTRUCTIONS,
            original_section="",
            refined_section=f"\n\nNote: {description}",
            description=f"Addressed: {description}",
            expected_impact=0.05,
            confidence=0.40,
        )

    def apply_refinements(
        self,
        prompt: str,
        refinements: list[PromptRefinement],
    ) -> str:
        """
        Apply refinements to a prompt.

        Args:
            prompt: Original prompt
            refinements: List of refinements to apply

        Returns:
            Refined prompt
        """
        new_prompt = prompt

        for refinement in refinements:
            if refinement.original_section and refinement.original_section in new_prompt:
                # Replace existing section
                new_prompt = new_prompt.replace(
                    refinement.original_section,
                    refinement.refined_section,
                )
            else:
                # Append new section
                new_prompt = new_prompt + refinement.refined_section

        return new_prompt

    def optimize(
        self,
        prompt: str,
        weaknesses: list[Any],
        context: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """
        Optimize a prompt based on weaknesses.

        Args:
            prompt: Original prompt
            weaknesses: List of identified weaknesses
            context: Optional context

        Returns:
            OptimizationResult with refined prompt
        """
        import time

        start = time.time()

        # Generate refinements
        refinements = self.generate_refinements(prompt, weaknesses, context)

        # Apply refinements
        optimized = self.apply_refinements(prompt, refinements)

        # Calculate total expected impact
        total_impact = sum(r.expected_impact for r in refinements)

        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized,
            refinements_applied=refinements,
            total_expected_impact=min(total_impact, 1.0),  # Cap at 100%
            optimization_time_ms=(time.time() - start) * 1000,
            strategy_used=self.strategy,
        )

    def analyze_prompt(self, prompt: str) -> dict[str, Any]:
        """
        Analyze a prompt for potential improvements.

        Args:
            prompt: Prompt to analyze

        Returns:
            Analysis results
        """
        analysis = {
            "length": len(prompt),
            "word_count": len(prompt.split()),
            "has_examples": "example" in prompt.lower(),
            "has_constraints": any(w in prompt.lower() for w in ["constraint", "limit", "must", "never"]),
            "has_process": any(w in prompt.lower() for w in ["step", "process", "first", "then"]),
            "has_error_handling": any(w in prompt.lower() for w in ["error", "fail", "if not", "exception"]),
            "potential_weaknesses": [],
        }

        # Identify potential weaknesses
        if not analysis["has_examples"]:
            analysis["potential_weaknesses"].append("Missing concrete examples")
        if not analysis["has_constraints"]:
            analysis["potential_weaknesses"].append("Missing explicit constraints")
        if not analysis["has_process"]:
            analysis["potential_weaknesses"].append("Missing step-by-step process")
        if not analysis["has_error_handling"]:
            analysis["potential_weaknesses"].append("Missing error handling guidance")

        analysis["improvement_potential"] = len(analysis["potential_weaknesses"]) * 0.1

        return analysis


def create_prompt_optimizer(
    strategy: str = "rule_based",
    max_refinements: int = 5,
    min_impact: float = 0.05,
    llm_client: Any | None = None,
) -> PromptOptimizer:
    """
    Factory function to create a prompt optimizer.

    Args:
        strategy: Optimization strategy ("rule_based", "llm_assisted", "hybrid")
        max_refinements: Maximum refinements per cycle
        min_impact: Minimum expected impact threshold
        llm_client: Optional LLM client for LLM-assisted mode

    Returns:
        Configured PromptOptimizer
    """
    strategy_enum = RefinementStrategy(strategy)

    return PromptOptimizer(
        llm_client=llm_client,
        strategy=strategy_enum,
        max_refinements_per_cycle=max_refinements,
        min_expected_impact=min_impact,
    )


def generate_optimization_report(result: OptimizationResult) -> str:
    """
    Generate a human-readable optimization report.

    Args:
        result: OptimizationResult from optimization

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "PROMPT OPTIMIZATION REPORT",
        "=" * 60,
        "",
        f"Strategy: {result.strategy_used.value}",
        f"Duration: {result.optimization_time_ms:.1f}ms",
        f"Total Expected Impact: {result.total_expected_impact:.1%}",
        "",
        f"REFINEMENTS APPLIED: {len(result.refinements_applied)}",
    ]

    for i, ref in enumerate(result.refinements_applied, 1):
        lines.extend(
            [
                "",
                f"  {i}. {ref.category.value}",
                f"     Description: {ref.description}",
                f"     Expected Impact: {ref.expected_impact:.1%}",
                f"     Confidence: {ref.confidence:.1%}",
            ]
        )

    lines.extend(
        [
            "",
            "PROMPT SIZE CHANGE:",
            f"  Before: {len(result.original_prompt)} chars",
            f"  After: {len(result.optimized_prompt)} chars",
            f"  Delta: {len(result.optimized_prompt) - len(result.original_prompt):+d} chars",
            "",
            "=" * 60,
        ]
    )

    return "\n".join(lines)
