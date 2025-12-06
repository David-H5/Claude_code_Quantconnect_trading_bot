"""
Cost Optimization for LLM Operations.

Implements strategies to reduce LLM costs while maintaining quality:
- Model cascading: Route simple queries to cheaper models
- Response caching: Cache responses with semantic similarity matching
- Budget controls: Monthly caps with alerts at 60/80/100%
- Context pruning: Reduce token usage by 40-50%

Usage:
    from llm.cost_optimization import CostOptimizer, create_cost_optimizer

    optimizer = create_cost_optimizer(monthly_budget=100.0)

    # Get optimal model for request
    model = optimizer.select_model(query, complexity_score=0.3)

    # Check cache before calling LLM
    cached = optimizer.get_cached_response(query)
    if cached:
        return cached

    # After LLM call, cache and track costs
    optimizer.cache_response(query, response)
    optimizer.track_cost(model="gpt-4", tokens_in=1000, tokens_out=500)
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar


class ModelTier(Enum):
    """Model pricing tiers."""

    BUDGET = "budget"  # Cheapest (e.g., GPT-3.5, Claude Haiku)
    STANDARD = "standard"  # Mid-tier (e.g., GPT-4-turbo, Claude Sonnet)
    PREMIUM = "premium"  # Most capable (e.g., GPT-4, Claude Opus)


class AlertLevel(Enum):
    """Budget alert levels."""

    NORMAL = "normal"
    WARNING = "warning"  # 60% of budget
    CRITICAL = "critical"  # 80% of budget
    EXCEEDED = "exceeded"  # 100% of budget


@dataclass
class ModelConfig:
    """Configuration for a model tier."""

    tier: ModelTier
    model_name: str
    cost_per_1k_input: float  # Cost per 1000 input tokens
    cost_per_1k_output: float  # Cost per 1000 output tokens
    max_context: int  # Maximum context window
    complexity_threshold: float  # Max complexity score for this tier


@dataclass
class CostRecord:
    """Record of a single LLM cost."""

    timestamp: float
    model: str
    tokens_in: int
    tokens_out: int
    cost: float
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Cached response entry."""

    query_hash: str
    query: str
    response: str
    created_at: float
    access_count: int = 0
    last_accessed: float | None = None
    model: str = ""
    tokens_saved: int = 0


@dataclass
class BudgetStatus:
    """Current budget status."""

    monthly_budget: float
    spent_this_month: float
    remaining: float
    utilization_pct: float
    alert_level: AlertLevel
    days_remaining: int
    projected_spend: float


class CostOptimizer:
    """
    Optimizes LLM costs through model selection, caching, and budget controls.

    Strategies:
    1. Model Cascading: Route by complexity (simple → budget, complex → premium)
    2. Response Caching: Semantic similarity matching for cache hits
    3. Budget Controls: Alerts at 60/80/100% of monthly budget
    4. Context Pruning: Reduce redundant tokens
    """

    # Default model configurations
    DEFAULT_MODELS: ClassVar[dict[ModelTier, ModelConfig]] = {
        ModelTier.BUDGET: ModelConfig(
            tier=ModelTier.BUDGET,
            model_name="claude-3-haiku-20240307",
            cost_per_1k_input=0.00025,
            cost_per_1k_output=0.00125,
            max_context=200000,
            complexity_threshold=0.3,
        ),
        ModelTier.STANDARD: ModelConfig(
            tier=ModelTier.STANDARD,
            model_name="claude-sonnet-4-20250514",
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            max_context=200000,
            complexity_threshold=0.7,
        ),
        ModelTier.PREMIUM: ModelConfig(
            tier=ModelTier.PREMIUM,
            model_name="claude-opus-4-20250514",
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            max_context=200000,
            complexity_threshold=1.0,
        ),
    }

    def __init__(
        self,
        monthly_budget: float = 100.0,
        cache_dir: str = "logs/llm_cache",
        cost_log_dir: str = "logs/llm_costs",
        cache_ttl: int = 86400,  # 24 hours
        cache_capacity: int = 1000,
        similarity_threshold: float = 0.92,
        model_configs: dict[ModelTier, ModelConfig] | None = None,
    ):
        """
        Initialize cost optimizer.

        Args:
            monthly_budget: Monthly spending limit in USD
            cache_dir: Directory for response cache
            cost_log_dir: Directory for cost logs
            cache_ttl: Cache time-to-live in seconds
            cache_capacity: Maximum cache entries
            similarity_threshold: Minimum similarity for cache hits
            model_configs: Custom model configurations
        """
        self.monthly_budget = monthly_budget
        self.cache_dir = Path(cache_dir)
        self.cost_log_dir = Path(cost_log_dir)
        self.cache_ttl = cache_ttl
        self.cache_capacity = cache_capacity
        self.similarity_threshold = similarity_threshold

        # Model configurations
        self.models = model_configs or self.DEFAULT_MODELS

        # Initialize directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cost_log_dir.mkdir(parents=True, exist_ok=True)

        # Load state
        self._cache: dict[str, CacheEntry] = {}
        self._cost_records: list[CostRecord] = []
        self._load_cache()
        self._load_costs()

    # ========================================================================
    # Model Selection (Cascading)
    # ========================================================================

    def select_model(
        self,
        query: str,
        complexity_score: float | None = None,
        require_premium: bool = False,
        context_length: int = 0,
    ) -> ModelConfig:
        """
        Select optimal model based on query complexity and constraints.

        Args:
            query: The user query
            complexity_score: Pre-computed complexity (0-1), auto-computed if None
            require_premium: Force premium model selection
            context_length: Required context window size

        Returns:
            ModelConfig for the selected model
        """
        if require_premium:
            return self.models[ModelTier.PREMIUM]

        # Auto-compute complexity if not provided
        if complexity_score is None:
            complexity_score = self._estimate_complexity(query)

        # Check context requirements
        for tier in [ModelTier.BUDGET, ModelTier.STANDARD, ModelTier.PREMIUM]:
            config = self.models[tier]
            if context_length > config.max_context:
                continue
            if complexity_score <= config.complexity_threshold:
                return config

        # Default to premium if nothing else fits
        return self.models[ModelTier.PREMIUM]

    def _estimate_complexity(self, query: str) -> float:
        """
        Estimate query complexity based on heuristics.

        Factors:
        - Length: Longer queries often need more reasoning
        - Keywords: Certain words indicate complex tasks
        - Structure: Multi-part questions, comparisons

        Returns:
            Complexity score between 0 and 1
        """
        score = 0.0

        # Length factor (normalized to 0-0.3)
        length_score = min(len(query) / 1000, 0.3)
        score += length_score

        # Complexity keywords
        complex_keywords = [
            "analyze",
            "compare",
            "explain",
            "synthesize",
            "evaluate",
            "architecture",
            "design",
            "refactor",
            "optimize",
            "debug",
            "complex",
            "comprehensive",
            "detailed",
            "thorough",
        ]
        simple_keywords = [
            "what is",
            "define",
            "list",
            "simple",
            "quick",
            "brief",
            "short",
            "yes or no",
            "true or false",
        ]

        query_lower = query.lower()
        complex_count = sum(1 for kw in complex_keywords if kw in query_lower)
        simple_count = sum(1 for kw in simple_keywords if kw in query_lower)

        # Keyword factor (0-0.4)
        keyword_score = min(complex_count * 0.1, 0.4) - min(simple_count * 0.1, 0.2)
        score += max(keyword_score, 0)

        # Multi-part question factor (0-0.2)
        question_marks = query.count("?")
        if question_marks > 1:
            score += min(question_marks * 0.05, 0.2)

        # Code/technical factor (0-0.1)
        if "```" in query or "def " in query or "class " in query:
            score += 0.1

        return min(score, 1.0)

    # ========================================================================
    # Response Caching
    # ========================================================================

    def get_cached_response(self, query: str) -> str | None:
        """
        Get cached response for a query.

        First checks exact match, then semantic similarity.

        Args:
            query: The user query

        Returns:
            Cached response if found, None otherwise
        """
        self._cleanup_expired_cache()

        # Exact match
        query_hash = self._hash_query(query)
        if query_hash in self._cache:
            entry = self._cache[query_hash]
            entry.access_count += 1
            entry.last_accessed = time.time()
            return entry.response

        # Semantic similarity (simplified - just check for similar prefixes/structure)
        # In production, use embedding-based similarity
        for _hash_key, entry in self._cache.items():
            if self._simple_similarity(query, entry.query) >= self.similarity_threshold:
                entry.access_count += 1
                entry.last_accessed = time.time()
                return entry.response

        return None

    def cache_response(
        self,
        query: str,
        response: str,
        model: str = "",
        tokens_used: int = 0,
    ) -> None:
        """
        Cache a response for future use.

        Args:
            query: The user query
            response: The LLM response
            model: Model that generated the response
            tokens_used: Tokens used for this response
        """
        # Evict if at capacity
        if len(self._cache) >= self.cache_capacity:
            self._evict_lru_cache()

        query_hash = self._hash_query(query)
        self._cache[query_hash] = CacheEntry(
            query_hash=query_hash,
            query=query,
            response=response,
            created_at=time.time(),
            model=model,
            tokens_saved=tokens_used,
        )
        self._save_cache()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_tokens_saved = sum(e.tokens_saved * e.access_count for e in self._cache.values())
        total_hits = sum(e.access_count for e in self._cache.values())

        return {
            "entries": len(self._cache),
            "capacity": self.cache_capacity,
            "utilization": len(self._cache) / self.cache_capacity,
            "total_hits": total_hits,
            "tokens_saved": total_tokens_saved,
            "estimated_savings": self._estimate_cache_savings(total_tokens_saved),
        }

    def _hash_query(self, query: str) -> str:
        """Create hash for query."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def _simple_similarity(self, query1: str, query2: str) -> float:
        """
        Simple similarity metric (Jaccard-like).

        In production, replace with embedding-based similarity.
        """
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _cleanup_expired_cache(self) -> None:
        """Remove expired cache entries."""
        now = time.time()
        expired = [k for k, v in self._cache.items() if now - v.created_at > self.cache_ttl]
        for key in expired:
            del self._cache[key]

    def _evict_lru_cache(self) -> None:
        """Evict least recently used cache entry."""
        if not self._cache:
            return

        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed or self._cache[k].created_at,
        )
        del self._cache[oldest_key]

    def _estimate_cache_savings(self, tokens_saved: int) -> float:
        """Estimate cost savings from cached tokens."""
        # Use standard tier pricing as baseline
        config = self.models[ModelTier.STANDARD]
        avg_cost = (config.cost_per_1k_input + config.cost_per_1k_output) / 2
        return (tokens_saved / 1000) * avg_cost

    # ========================================================================
    # Budget Controls
    # ========================================================================

    def track_cost(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CostRecord:
        """
        Track cost of an LLM request.

        Args:
            model: Model name used
            tokens_in: Input tokens
            tokens_out: Output tokens
            request_id: Optional request identifier
            metadata: Additional metadata

        Returns:
            CostRecord with calculated cost
        """
        cost = self._calculate_cost(model, tokens_in, tokens_out)

        record = CostRecord(
            timestamp=time.time(),
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            request_id=request_id,
            metadata=metadata or {},
        )

        self._cost_records.append(record)
        self._save_costs()

        # Check budget alerts
        status = self.get_budget_status()
        if status.alert_level != AlertLevel.NORMAL:
            self._trigger_budget_alert(status)

        return record

    def get_budget_status(self) -> BudgetStatus:
        """Get current budget status."""
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Sum costs for current month
        spent = sum(r.cost for r in self._cost_records if r.timestamp >= month_start.timestamp())

        remaining = max(self.monthly_budget - spent, 0)
        utilization = spent / self.monthly_budget if self.monthly_budget > 0 else 0

        # Determine alert level
        if utilization >= 1.0:
            alert_level = AlertLevel.EXCEEDED
        elif utilization >= 0.8:
            alert_level = AlertLevel.CRITICAL
        elif utilization >= 0.6:
            alert_level = AlertLevel.WARNING
        else:
            alert_level = AlertLevel.NORMAL

        # Project end-of-month spend
        days_elapsed = (now - month_start).days + 1
        days_in_month = 30  # Simplified
        days_remaining = max(days_in_month - days_elapsed, 1)
        daily_rate = spent / days_elapsed if days_elapsed > 0 else 0
        projected_spend = spent + (daily_rate * days_remaining)

        return BudgetStatus(
            monthly_budget=self.monthly_budget,
            spent_this_month=spent,
            remaining=remaining,
            utilization_pct=utilization * 100,
            alert_level=alert_level,
            days_remaining=days_remaining,
            projected_spend=projected_spend,
        )

    def _calculate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Calculate cost for a request."""
        # Find matching model config
        for config in self.models.values():
            if config.model_name == model or model in config.model_name:
                return (tokens_in / 1000 * config.cost_per_1k_input) + (tokens_out / 1000 * config.cost_per_1k_output)

        # Default to standard tier if unknown
        config = self.models[ModelTier.STANDARD]
        return (tokens_in / 1000 * config.cost_per_1k_input) + (tokens_out / 1000 * config.cost_per_1k_output)

    def _trigger_budget_alert(self, status: BudgetStatus) -> None:
        """Trigger budget alert (log for now, could send notifications)."""
        alert_file = self.cost_log_dir / "budget_alerts.log"
        with open(alert_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} - {status.alert_level.value}: "
                f"{status.utilization_pct:.1f}% of ${status.monthly_budget:.2f} used\n"
            )

    # ========================================================================
    # Context Pruning
    # ========================================================================

    def prune_context(
        self,
        context: str,
        target_reduction: float = 0.4,
        preserve_start: int = 500,
        preserve_end: int = 500,
    ) -> str:
        """
        Prune context to reduce token usage.

        Strategies:
        - Remove redundant whitespace
        - Compress repeated patterns
        - Preserve start and end (U-shaped attention)

        Args:
            context: Original context string
            target_reduction: Target reduction percentage (0.4 = 40%)
            preserve_start: Characters to preserve at start
            preserve_end: Characters to preserve at end

        Returns:
            Pruned context string
        """
        if not context:
            return context

        original_length = len(context)
        target_length = int(original_length * (1 - target_reduction))

        # Step 1: Normalize whitespace
        pruned = " ".join(context.split())

        # Step 2: If still too long, truncate middle (preserve start/end)
        if len(pruned) > target_length:
            if preserve_start + preserve_end >= target_length:
                # Can't preserve both, prioritize start
                pruned = pruned[:target_length]
            else:
                start = pruned[:preserve_start]
                end = pruned[-preserve_end:]
                middle_budget = target_length - preserve_start - preserve_end
                middle_start = preserve_start
                middle_end = len(pruned) - preserve_end

                if middle_budget > 0 and middle_end > middle_start:
                    # Take from middle
                    middle = pruned[middle_start : middle_start + middle_budget]
                    pruned = start + "\n...[truncated]...\n" + middle + "\n...[truncated]...\n" + end
                else:
                    pruned = start + "\n...[truncated]...\n" + end

        return pruned

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses simple heuristic: ~4 chars per token.
        """
        return len(text) // 4

    # ========================================================================
    # Persistence
    # ========================================================================

    def _save_cache(self) -> None:
        """Save cache to disk."""
        cache_file = self.cache_dir / "response_cache.json"
        data = {k: self._cache_entry_to_dict(v) for k, v in self._cache.items()}
        cache_file.write_text(json.dumps(data, indent=2))

    def _load_cache(self) -> None:
        """Load cache from disk."""
        cache_file = self.cache_dir / "response_cache.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                self._cache = {k: self._dict_to_cache_entry(v) for k, v in data.items()}
            except (json.JSONDecodeError, KeyError):
                self._cache = {}

    def _cache_entry_to_dict(self, entry: CacheEntry) -> dict[str, Any]:
        """Convert cache entry to dict."""
        return {
            "query_hash": entry.query_hash,
            "query": entry.query,
            "response": entry.response,
            "created_at": entry.created_at,
            "access_count": entry.access_count,
            "last_accessed": entry.last_accessed,
            "model": entry.model,
            "tokens_saved": entry.tokens_saved,
        }

    def _dict_to_cache_entry(self, data: dict[str, Any]) -> CacheEntry:
        """Convert dict to cache entry."""
        return CacheEntry(
            query_hash=data["query_hash"],
            query=data["query"],
            response=data["response"],
            created_at=data["created_at"],
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed"),
            model=data.get("model", ""),
            tokens_saved=data.get("tokens_saved", 0),
        )

    def _save_costs(self) -> None:
        """Save cost records to disk."""
        # Keep only last 30 days
        cutoff = time.time() - (30 * 86400)
        self._cost_records = [r for r in self._cost_records if r.timestamp > cutoff]

        cost_file = self.cost_log_dir / "cost_records.json"
        data = [self._cost_record_to_dict(r) for r in self._cost_records]
        cost_file.write_text(json.dumps(data, indent=2))

    def _load_costs(self) -> None:
        """Load cost records from disk."""
        cost_file = self.cost_log_dir / "cost_records.json"
        if cost_file.exists():
            try:
                data = json.loads(cost_file.read_text())
                self._cost_records = [self._dict_to_cost_record(d) for d in data]
            except (json.JSONDecodeError, KeyError):
                self._cost_records = []

    def _cost_record_to_dict(self, record: CostRecord) -> dict[str, Any]:
        """Convert cost record to dict."""
        return {
            "timestamp": record.timestamp,
            "model": record.model,
            "tokens_in": record.tokens_in,
            "tokens_out": record.tokens_out,
            "cost": record.cost,
            "request_id": record.request_id,
            "metadata": record.metadata,
        }

    def _dict_to_cost_record(self, data: dict[str, Any]) -> CostRecord:
        """Convert dict to cost record."""
        return CostRecord(
            timestamp=data["timestamp"],
            model=data["model"],
            tokens_in=data["tokens_in"],
            tokens_out=data["tokens_out"],
            cost=data["cost"],
            request_id=data.get("request_id"),
            metadata=data.get("metadata", {}),
        )

    def get_cost_report(self, days: int = 30) -> dict[str, Any]:
        """
        Generate cost report for specified period.

        Args:
            days: Number of days to include

        Returns:
            Report with cost breakdown
        """
        cutoff = time.time() - (days * 86400)
        records = [r for r in self._cost_records if r.timestamp > cutoff]

        if not records:
            return {
                "period_days": days,
                "total_cost": 0.0,
                "total_requests": 0,
                "by_model": {},
                "daily_average": 0.0,
            }

        total_cost = sum(r.cost for r in records)
        by_model: dict[str, float] = {}
        for r in records:
            by_model[r.model] = by_model.get(r.model, 0) + r.cost

        return {
            "period_days": days,
            "total_cost": total_cost,
            "total_requests": len(records),
            "total_tokens_in": sum(r.tokens_in for r in records),
            "total_tokens_out": sum(r.tokens_out for r in records),
            "by_model": by_model,
            "daily_average": total_cost / days,
            "budget_status": self.get_budget_status().__dict__,
            "cache_stats": self.get_cache_stats(),
        }


# ============================================================================
# Convenience Functions
# ============================================================================


def create_cost_optimizer(
    monthly_budget: float = 100.0,
    cache_dir: str = "logs/llm_cache",
    cost_log_dir: str = "logs/llm_costs",
) -> CostOptimizer:
    """Create a cost optimizer with default settings."""
    return CostOptimizer(
        monthly_budget=monthly_budget,
        cache_dir=cache_dir,
        cost_log_dir=cost_log_dir,
    )


if __name__ == "__main__":
    # Demo usage
    optimizer = create_cost_optimizer(monthly_budget=50.0)

    # Test model selection
    simple_query = "What is Python?"
    complex_query = (
        "Analyze and compare the architectural patterns of microservices vs monolith, provide comprehensive evaluation"
    )

    simple_model = optimizer.select_model(simple_query)
    complex_model = optimizer.select_model(complex_query)

    print(f"Simple query → {simple_model.tier.value} ({simple_model.model_name})")
    print(f"Complex query → {complex_model.tier.value} ({complex_model.model_name})")

    # Test caching
    optimizer.cache_response("What is Python?", "Python is a programming language.", tokens_used=50)
    cached = optimizer.get_cached_response("What is Python?")
    print(f"\nCached response: {cached}")

    # Test cost tracking
    optimizer.track_cost(model="claude-sonnet-4-20250514", tokens_in=1000, tokens_out=500)
    optimizer.track_cost(model="claude-3-haiku-20240307", tokens_in=500, tokens_out=200)

    # Print status
    status = optimizer.get_budget_status()
    print(f"\nBudget Status: {status.utilization_pct:.1f}% used, ${status.remaining:.2f} remaining")

    # Print report
    report = optimizer.get_cost_report(days=7)
    print(f"\n7-Day Report: ${report['total_cost']:.4f} total, {report['total_requests']} requests")
