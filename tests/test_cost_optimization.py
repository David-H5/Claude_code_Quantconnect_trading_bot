"""Tests for cost optimization module."""

import tempfile
import time

import pytest

from llm.cost_optimization import (
    AlertLevel,
    CacheEntry,
    CostOptimizer,
    CostRecord,
    ModelConfig,
    ModelTier,
    create_cost_optimizer,
)


class TestModelTier:
    """Tests for ModelTier enum."""

    def test_tier_values(self):
        """Test tier enum values."""
        assert ModelTier.BUDGET.value == "budget"
        assert ModelTier.STANDARD.value == "standard"
        assert ModelTier.PREMIUM.value == "premium"


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_alert_values(self):
        """Test alert enum values."""
        assert AlertLevel.NORMAL.value == "normal"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.EXCEEDED.value == "exceeded"


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_config_creation(self):
        """Test creating model config."""
        config = ModelConfig(
            tier=ModelTier.STANDARD,
            model_name="test-model",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            max_context=100000,
            complexity_threshold=0.5,
        )
        assert config.tier == ModelTier.STANDARD
        assert config.model_name == "test-model"
        assert config.cost_per_1k_input == 0.001


class TestCostOptimizer:
    """Tests for CostOptimizer class."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for cache and logs."""
        with (
            tempfile.TemporaryDirectory() as cache_dir,
            tempfile.TemporaryDirectory() as cost_dir,
        ):
            yield cache_dir, cost_dir

    @pytest.fixture
    def optimizer(self, temp_dirs):
        """Create optimizer instance."""
        cache_dir, cost_dir = temp_dirs
        return CostOptimizer(
            monthly_budget=100.0,
            cache_dir=cache_dir,
            cost_log_dir=cost_dir,
            cache_capacity=10,
        )

    # Model Selection Tests
    def test_select_model_simple_query(self, optimizer):
        """Test simple queries get budget model."""
        model = optimizer.select_model("What is Python?")
        assert model.tier == ModelTier.BUDGET

    def test_select_model_complex_query(self, optimizer):
        """Test complex queries get appropriate model."""
        query = "Analyze and compare the architectural patterns of microservices versus monolith, provide comprehensive evaluation with detailed examples"
        model = optimizer.select_model(query)
        assert model.tier in [ModelTier.STANDARD, ModelTier.PREMIUM]

    def test_select_model_with_complexity_score(self, optimizer):
        """Test model selection with explicit complexity score."""
        # Low complexity
        model = optimizer.select_model("any query", complexity_score=0.1)
        assert model.tier == ModelTier.BUDGET

        # Medium complexity
        model = optimizer.select_model("any query", complexity_score=0.5)
        assert model.tier == ModelTier.STANDARD

        # High complexity
        model = optimizer.select_model("any query", complexity_score=0.9)
        assert model.tier == ModelTier.PREMIUM

    def test_select_model_require_premium(self, optimizer):
        """Test forcing premium model."""
        model = optimizer.select_model("simple query", require_premium=True)
        assert model.tier == ModelTier.PREMIUM

    def test_estimate_complexity_simple(self, optimizer):
        """Test complexity estimation for simple queries."""
        score = optimizer._estimate_complexity("What is a list?")
        assert score < 0.3

    def test_estimate_complexity_complex(self, optimizer):
        """Test complexity estimation for complex queries."""
        query = "Analyze and compare the comprehensive architectural design patterns"
        score = optimizer._estimate_complexity(query)
        assert score > 0.3

    def test_estimate_complexity_code(self, optimizer):
        """Test complexity for code-related queries."""
        query = "Here is some code:\n```python\ndef foo(): pass\n```"
        score = optimizer._estimate_complexity(query)
        # Code adds complexity
        assert score >= 0.1

    # Caching Tests
    def test_cache_response(self, optimizer):
        """Test caching a response."""
        optimizer.cache_response("test query", "test response", tokens_used=100)
        assert len(optimizer._cache) == 1

    def test_get_cached_response_exact_match(self, optimizer):
        """Test retrieving exact cached response."""
        optimizer.cache_response("test query", "test response")
        cached = optimizer.get_cached_response("test query")
        assert cached == "test response"

    def test_get_cached_response_miss(self, optimizer):
        """Test cache miss."""
        cached = optimizer.get_cached_response("nonexistent query")
        assert cached is None

    def test_cache_capacity_eviction(self, optimizer):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        for i in range(10):
            optimizer.cache_response(f"query {i}", f"response {i}")

        # Access first few to make them recently used
        optimizer.get_cached_response("query 0")
        optimizer.get_cached_response("query 1")

        # Add one more - should evict least recently used
        optimizer.cache_response("query 10", "response 10")

        # query 0 and 1 should still be there
        assert optimizer.get_cached_response("query 0") is not None
        assert optimizer.get_cached_response("query 1") is not None

    def test_cache_stats(self, optimizer):
        """Test cache statistics."""
        optimizer.cache_response("q1", "r1", tokens_used=50)
        optimizer.cache_response("q2", "r2", tokens_used=100)
        optimizer.get_cached_response("q1")  # Hit

        stats = optimizer.get_cache_stats()
        assert stats["entries"] == 2
        assert stats["total_hits"] >= 1

    def test_cache_persistence(self, temp_dirs):
        """Test cache persists to disk."""
        cache_dir, cost_dir = temp_dirs

        # Create optimizer and cache
        opt1 = CostOptimizer(cache_dir=cache_dir, cost_log_dir=cost_dir)
        opt1.cache_response("persistent query", "persistent response")

        # Create new optimizer - should load cache
        opt2 = CostOptimizer(cache_dir=cache_dir, cost_log_dir=cost_dir)
        cached = opt2.get_cached_response("persistent query")
        assert cached == "persistent response"

    # Budget Control Tests
    def test_track_cost(self, optimizer):
        """Test cost tracking."""
        record = optimizer.track_cost(
            model="claude-sonnet-4-20250514",
            tokens_in=1000,
            tokens_out=500,
        )
        assert record.tokens_in == 1000
        assert record.tokens_out == 500
        assert record.cost > 0

    def test_budget_status_normal(self, optimizer):
        """Test budget status under threshold."""
        # Track small cost
        optimizer.track_cost(model="claude-3-haiku-20240307", tokens_in=100, tokens_out=50)

        status = optimizer.get_budget_status()
        assert status.alert_level == AlertLevel.NORMAL
        assert status.utilization_pct < 60

    def test_budget_status_warning(self, temp_dirs):
        """Test budget warning at 60%."""
        cache_dir, cost_dir = temp_dirs
        optimizer = CostOptimizer(
            monthly_budget=1.0,  # Small budget
            cache_dir=cache_dir,
            cost_log_dir=cost_dir,
        )

        # Spend 65% of budget
        optimizer.track_cost(
            model="claude-opus-4-20250514",
            tokens_in=5000,  # Enough to exceed 60%
            tokens_out=2000,
        )

        status = optimizer.get_budget_status()
        assert status.utilization_pct >= 60

    def test_cost_calculation(self, optimizer):
        """Test cost calculation."""
        cost = optimizer._calculate_cost(
            model="claude-sonnet-4-20250514",
            tokens_in=1000,
            tokens_out=1000,
        )
        # Should be: (1000/1000 * 0.003) + (1000/1000 * 0.015) = 0.018
        assert abs(cost - 0.018) < 0.001

    def test_cost_report(self, optimizer):
        """Test cost report generation."""
        optimizer.track_cost(model="test-model", tokens_in=1000, tokens_out=500)
        optimizer.track_cost(model="test-model", tokens_in=500, tokens_out=250)

        report = optimizer.get_cost_report(days=7)
        assert report["total_requests"] == 2
        assert report["total_tokens_in"] == 1500
        assert report["total_tokens_out"] == 750

    # Context Pruning Tests
    def test_prune_context_whitespace(self, optimizer):
        """Test pruning removes extra whitespace."""
        context = "This    has   extra    spaces"
        pruned = optimizer.prune_context(context, target_reduction=0.0)
        assert "    " not in pruned

    def test_prune_context_reduction(self, optimizer):
        """Test context is reduced."""
        context = "A" * 1000
        pruned = optimizer.prune_context(context, target_reduction=0.4)
        assert len(pruned) < len(context)

    def test_prune_context_preserves_ends(self, optimizer):
        """Test pruning preserves start and end."""
        context = "START" + "M" * 500 + "END"
        pruned = optimizer.prune_context(
            context,
            target_reduction=0.5,
            preserve_start=5,
            preserve_end=3,
        )
        assert pruned.startswith("START")
        assert pruned.endswith("END")

    def test_prune_context_empty(self, optimizer):
        """Test pruning empty context."""
        pruned = optimizer.prune_context("")
        assert pruned == ""

    def test_estimate_tokens(self, optimizer):
        """Test token estimation."""
        text = "a" * 400  # 400 chars ≈ 100 tokens
        tokens = optimizer.estimate_tokens(text)
        assert tokens == 100


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_entry_creation(self):
        """Test cache entry creation."""
        entry = CacheEntry(
            query_hash="abc123",
            query="test",
            response="result",
            created_at=time.time(),
        )
        assert entry.query == "test"
        assert entry.access_count == 0


class TestCostRecord:
    """Tests for CostRecord dataclass."""

    def test_record_creation(self):
        """Test cost record creation."""
        record = CostRecord(
            timestamp=time.time(),
            model="test-model",
            tokens_in=100,
            tokens_out=50,
            cost=0.01,
        )
        assert record.model == "test-model"
        assert record.cost == 0.01


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_create_cost_optimizer(self, tmp_path):
        """Test creating optimizer with convenience function."""
        optimizer = create_cost_optimizer(
            monthly_budget=50.0,
            cache_dir=str(tmp_path / "cache"),
            cost_log_dir=str(tmp_path / "costs"),
        )
        assert optimizer.monthly_budget == 50.0


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def optimizer(self, tmp_path):
        """Create optimizer instance."""
        return CostOptimizer(
            monthly_budget=100.0,
            cache_dir=str(tmp_path / "cache"),
            cost_log_dir=str(tmp_path / "costs"),
        )

    def test_zero_budget(self, tmp_path):
        """Test with zero budget."""
        optimizer = CostOptimizer(
            monthly_budget=0.0,
            cache_dir=str(tmp_path / "cache"),
            cost_log_dir=str(tmp_path / "costs"),
        )
        status = optimizer.get_budget_status()
        # Should handle division by zero
        assert status.utilization_pct == 0 or status.alert_level == AlertLevel.EXCEEDED

    def test_empty_query(self, optimizer):
        """Test with empty query."""
        model = optimizer.select_model("")
        assert model is not None  # Should not crash

    def test_very_long_query(self, optimizer):
        """Test with very long query."""
        query = "analyze " * 1000
        score = optimizer._estimate_complexity(query)
        assert 0 <= score <= 1

    def test_unicode_query(self, optimizer):
        """Test with unicode query."""
        query = "分析这个代码 🎉"
        optimizer.cache_response(query, "response")
        cached = optimizer.get_cached_response(query)
        assert cached == "response"

    def test_corrupted_cache_recovery(self, tmp_path):
        """Test recovery from corrupted cache file."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "response_cache.json").write_text("{ invalid json }")

        # Should recover gracefully
        optimizer = CostOptimizer(
            cache_dir=str(cache_dir),
            cost_log_dir=str(tmp_path / "costs"),
        )
        assert len(optimizer._cache) == 0

    def test_corrupted_costs_recovery(self, tmp_path):
        """Test recovery from corrupted cost file."""
        cost_dir = tmp_path / "costs"
        cost_dir.mkdir()
        (cost_dir / "cost_records.json").write_text("[ invalid ]")

        # Should recover gracefully
        optimizer = CostOptimizer(
            cache_dir=str(tmp_path / "cache"),
            cost_log_dir=str(cost_dir),
        )
        assert len(optimizer._cost_records) == 0
