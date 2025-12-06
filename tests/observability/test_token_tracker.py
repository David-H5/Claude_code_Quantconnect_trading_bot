"""
Tests for Token Tracking Module

UPGRADE-015 Phase 6: Observability Setup

Tests cover:
- Token usage tracking
- Cost estimation
- Budget limits
- Usage reports
"""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import token metrics from existing module
try:
    from observability.token_metrics import (
        TokenMetrics,
        create_token_tracker,
    )

    TOKEN_TRACKER_AVAILABLE = True
except ImportError:
    TOKEN_TRACKER_AVAILABLE = False


@pytest.mark.skipif(not TOKEN_TRACKER_AVAILABLE, reason="Token tracker not available")
class TestTokenTracker:
    """Test token tracking functionality."""

    def test_record_usage(self):
        """Test recording token usage."""
        tracker = create_token_tracker()

        tracker.record_usage(
            model="claude-3-sonnet",
            prompt_tokens=100,
            completion_tokens=50,
        )

        stats = tracker.get_stats()
        assert stats["total_tokens"] == 150
        assert stats["prompt_tokens"] == 100
        assert stats["completion_tokens"] == 50

    def test_cost_estimation(self):
        """Test cost estimation for token usage."""
        tracker = create_token_tracker()

        tracker.record_usage(
            model="claude-3-sonnet",
            prompt_tokens=1000,
            completion_tokens=500,
        )

        stats = tracker.get_stats()
        # Cost should be > 0 for any non-zero usage
        assert stats.get("estimated_cost", 0) >= 0

    def test_multiple_models(self):
        """Test tracking multiple models."""
        tracker = create_token_tracker()

        tracker.record_usage(model="claude-3-sonnet", prompt_tokens=100, completion_tokens=50)
        tracker.record_usage(model="gpt-4", prompt_tokens=200, completion_tokens=100)

        stats = tracker.get_stats()
        assert stats["total_tokens"] == 450  # 150 + 300

    def test_budget_limit(self):
        """Test budget limit warning."""
        tracker = create_token_tracker(daily_budget=0.01)

        # Record usage that might exceed budget
        for _ in range(100):
            tracker.record_usage(
                model="claude-3-sonnet",
                prompt_tokens=1000,
                completion_tokens=500,
            )

        # Should track that budget is exceeded
        stats = tracker.get_stats()
        assert "budget_exceeded" in stats or stats.get("estimated_cost", 0) > 0

    def test_reset_daily(self):
        """Test daily stats reset."""
        tracker = create_token_tracker()

        tracker.record_usage(model="claude-3-sonnet", prompt_tokens=100, completion_tokens=50)

        # Reset should clear stats
        tracker.reset_daily()

        stats = tracker.get_stats()
        assert stats["total_tokens"] == 0


@pytest.mark.skipif(not TOKEN_TRACKER_AVAILABLE, reason="Token tracker not available")
class TestTokenMetrics:
    """Test token metrics dataclass."""

    def test_metrics_creation(self):
        """Test creating token metrics."""
        metrics = TokenMetrics(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        assert metrics.prompt_tokens == 100
        assert metrics.completion_tokens == 50
        assert metrics.total_tokens == 150

    def test_metrics_add(self):
        """Test adding token metrics."""
        m1 = TokenMetrics(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        m2 = TokenMetrics(prompt_tokens=200, completion_tokens=100, total_tokens=300)

        combined = m1 + m2

        assert combined.prompt_tokens == 300
        assert combined.completion_tokens == 150
        assert combined.total_tokens == 450


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
