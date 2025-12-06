"""Unit tests for Semantic Router (UPGRADE-012.2).

Tests the hybrid keyword+semantic routing capabilities in select_prompts.py.
Semantic routing is optional - tests handle the case when sentence-transformers
is not installed.
"""

from pathlib import Path

import pytest
import yaml

# Import the modules under test
from scripts.select_prompts import (
    SEMANTIC_AVAILABLE,
    RoutingDecision,
    SemanticRouter,
    TaskRouter,
    visualize_complexity,
)


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_to_dict_basic(self):
        """Test basic dictionary conversion."""
        decision = RoutingDecision(
            task_description="Test task",
            original_task="Test task",
            complexity_level="L1_moderate",
            complexity_score=3,
            depth_score=1.5,
            width_score=1.2,
            complexity_reasoning=["Matched pattern X"],
            domain="general",
            domain_reasoning=["No domain matched"],
            override_used=False,
            override_type=None,
            prompts_selected=["prompts/complexity/L1_moderate.md"],
            safety_checks=[],
            keyword_confidence=0.7,
            semantic_confidence=0.0,
            routing_method="keyword",
        )

        result = decision.to_dict()

        assert result["complexity_level"] == "L1_moderate"
        assert result["complexity_score"] == 3
        assert result["depth_score"] == 1.5
        assert result["width_score"] == 1.2
        assert result["keyword_confidence"] == 0.7
        assert result["routing_method"] == "keyword"

    def test_to_dict_with_override(self):
        """Test dictionary conversion with override."""
        decision = RoutingDecision(
            task_description="Test task",
            original_task="COMPLEX: Test task",
            complexity_level="L1_complex",
            complexity_score=0,
            depth_score=0.0,
            width_score=0.0,
            complexity_reasoning=["Override: forced to L1_complex"],
            domain="general",
            domain_reasoning=[],
            override_used=True,
            override_type="complexity",
            prompts_selected=[],
            safety_checks=[],
        )

        result = decision.to_dict()

        assert result["override_used"] is True
        assert result["override_type"] == "complexity"


class TestSemanticRouter:
    """Tests for SemanticRouter class."""

    @pytest.fixture
    def semantic_config(self, tmp_path: Path) -> Path:
        """Create a temporary semantic routes config."""
        config = {
            "version": "1.0.0",
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "thresholds": {
                "keyword_confidence_threshold": 0.8,
                "semantic_similarity_threshold": 0.70,
                "fallback_level": "L1_moderate",
            },
            "complexity_candidates": {
                "L1_complex": {
                    "description": "Complex tasks",
                    "examples": [
                        "implement authentication system",
                        "refactor entire module",
                    ],
                },
                "L1_moderate": {
                    "description": "Moderate tasks",
                    "examples": [
                        "add feature to existing module",
                        "fix bug in component",
                    ],
                },
                "L1_simple": {
                    "description": "Simple tasks",
                    "examples": [
                        "fix typo",
                        "update config",
                    ],
                },
            },
            "domain_candidates": {
                "algorithm": {
                    "description": "Trading",
                    "examples": ["trading strategy", "backtest"],
                },
                "llm": {
                    "description": "LLM",
                    "examples": ["sentiment analysis", "agent prompt"],
                },
            },
        }

        config_path = tmp_path / "semantic-routes.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    def test_init_loads_config(self, semantic_config: Path):
        """Test that SemanticRouter loads config file."""
        router = SemanticRouter(config_path=semantic_config)

        assert router.config is not None
        assert "complexity_candidates" in router.config
        assert "domain_candidates" in router.config

    def test_init_with_missing_config(self, tmp_path: Path):
        """Test SemanticRouter with non-existent config."""
        missing_path = tmp_path / "missing.yaml"
        router = SemanticRouter(config_path=missing_path)

        assert router.config == {}

    @pytest.mark.skipif(
        not SEMANTIC_AVAILABLE,
        reason="sentence-transformers not installed",
    )
    def test_classify_complexity_with_model(self, semantic_config: Path):
        """Test complexity classification with sentence-transformers."""
        router = SemanticRouter(config_path=semantic_config)

        level, similarity = router.classify_complexity("implement new authentication with OAuth")

        assert level in ["L1_complex", "L1_moderate", "L1_simple"]
        assert 0.0 <= similarity <= 1.0

    def test_classify_complexity_without_model(self, semantic_config: Path):
        """Test fallback when model not available."""
        router = SemanticRouter(config_path=semantic_config)
        # Force model unavailable
        router.available = False

        level, similarity = router.classify_complexity("test task")

        assert level == "L1_moderate"
        assert similarity == 0.0

    @pytest.mark.skipif(
        not SEMANTIC_AVAILABLE,
        reason="sentence-transformers not installed",
    )
    def test_classify_domain_with_model(self, semantic_config: Path):
        """Test domain classification with sentence-transformers."""
        router = SemanticRouter(config_path=semantic_config)

        domain, similarity = router.classify_domain("implement trading strategy")

        assert domain in ["algorithm", "llm", "general"]
        assert 0.0 <= similarity <= 1.0

    def test_classify_domain_without_model(self, semantic_config: Path):
        """Test domain fallback when model not available."""
        router = SemanticRouter(config_path=semantic_config)
        router.available = False

        domain, similarity = router.classify_domain("test task")

        assert domain == "general"
        assert similarity == 0.0


class TestTaskRouter:
    """Tests for TaskRouter class."""

    @pytest.fixture
    def router_config(self, tmp_path: Path) -> Path:
        """Create a temporary task router config."""
        config = {
            "version": "1.2.0",
            "overrides": {
                "complexity": [
                    {
                        "pattern": r"^SIMPLE:\s*",
                        "level": "L1_simple",
                        "strip_prefix": True,
                    },
                    {
                        "pattern": r"^COMPLEX:\s*",
                        "level": "L1_complex",
                        "strip_prefix": True,
                    },
                ],
                "domain": [
                    {
                        "pattern": r"DOMAIN:(\w+):",
                        "strip_prefix": True,
                    },
                ],
            },
            "complexity": {
                "L3_triggers": {
                    "threshold": 5,
                    "patterns": [
                        {"pattern": "refactor", "weight": 3},
                        {"pattern": "implement", "weight": 2},
                        {"pattern": "authentication", "weight": 2},
                    ],
                },
                "L2_triggers": {
                    "threshold": 2,
                    "patterns": [
                        {"pattern": "add.*feature", "weight": 2},
                        {"pattern": "fix.*bug", "weight": 1},
                    ],
                },
                "L1_indicators": {
                    "patterns": [
                        {"pattern": "fix.*typo", "weight": -2},
                        {"pattern": "update.*config", "weight": -1},
                    ],
                },
            },
            "depth_indicators": {
                "multiplier": 1.5,
                "patterns": [
                    {"pattern": "multi.?step", "weight": 2},
                    {"pattern": "chain", "weight": 1},
                ],
            },
            "width_indicators": {
                "multiplier": 1.2,
                "patterns": [
                    {"pattern": "multiple.*files", "weight": 2},
                    {"pattern": "cross.?component", "weight": 1},
                ],
            },
            "domains": {
                "algorithm": {
                    "keywords": ["trading", "backtest", "strategy"],
                    "prompt_file": "prompts/domain/algorithm.md",
                    "safety_checks": ["Validate risk parameters"],
                },
                "general": {
                    "keywords": [],
                    "prompt_file": None,
                    "safety_checks": [],
                },
            },
            "prompts": {
                "complexity": {
                    "L1_simple": "prompts/complexity/L1_simple.md",
                    "L1_moderate": "prompts/complexity/L1_moderate.md",
                    "L1_complex": "prompts/complexity/L1_complex.md",
                },
            },
            "logging": {
                "enabled": False,
            },
        }

        config_path = tmp_path / "task-router.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    def test_route_simple_task(self, router_config: Path):
        """Test routing a simple task."""
        router = TaskRouter(config_path=router_config, enable_semantic=False)
        decision = router.route("fix typo in readme")

        assert decision.complexity_level == "L1_simple"
        assert decision.override_used is False

    def test_route_complex_task(self, router_config: Path):
        """Test routing a complex task."""
        router = TaskRouter(config_path=router_config, enable_semantic=False)
        decision = router.route("refactor authentication system")

        assert decision.complexity_level == "L1_complex"
        assert decision.complexity_score >= 5

    def test_route_with_complexity_override(self, router_config: Path):
        """Test complexity override prefix."""
        router = TaskRouter(config_path=router_config, enable_semantic=False)
        decision = router.route("COMPLEX: simple task")

        assert decision.complexity_level == "L1_complex"
        assert decision.override_used is True
        assert decision.override_type == "complexity"
        assert "simple task" in decision.task_description

    def test_route_detects_domain(self, router_config: Path):
        """Test domain detection."""
        router = TaskRouter(config_path=router_config, enable_semantic=False)
        decision = router.route("implement trading strategy")

        assert decision.domain == "algorithm"
        assert "Validate risk parameters" in decision.safety_checks

    def test_route_calculates_depth_width(self, router_config: Path):
        """Test depth and width score calculation."""
        router = TaskRouter(config_path=router_config, enable_semantic=False)
        decision = router.route("implement multi-step workflow across multiple files")

        # Should have depth score from "multi-step" and width score from "multiple files"
        assert decision.depth_score > 0
        assert decision.width_score > 0

    def test_route_keyword_confidence(self, router_config: Path):
        """Test keyword confidence calculation."""
        router = TaskRouter(config_path=router_config, enable_semantic=False)

        # High score = high confidence
        high_decision = router.route("refactor implement authentication")
        assert high_decision.keyword_confidence > 0.7

        # Low score = lower confidence
        low_decision = router.route("do something")
        assert low_decision.keyword_confidence < high_decision.keyword_confidence


class TestVisualizeComplexity:
    """Tests for visualization function."""

    def test_visualize_basic(self):
        """Test basic visualization output."""
        decision = RoutingDecision(
            task_description="Test task for visualization",
            original_task="Test task for visualization",
            complexity_level="L1_moderate",
            complexity_score=5,
            depth_score=2.0,
            width_score=1.5,
            complexity_reasoning=[],
            domain="general",
            domain_reasoning=[],
            override_used=False,
            override_type=None,
            prompts_selected=[],
            safety_checks=[],
            keyword_confidence=0.75,
            semantic_confidence=0.0,
            routing_method="keyword",
        )

        output = visualize_complexity(decision)

        assert "TASK COMPLEXITY ANALYSIS" in output
        assert "MODERATE" in output
        assert "Depth Score" in output
        assert "Width Score" in output
        assert "Keyword Conf" in output

    def test_visualize_all_levels(self):
        """Test visualization for each complexity level."""
        levels = [
            ("L1_simple", "SIMPLE", "🟢"),
            ("L1_moderate", "MODERATE", "🟡"),
            ("L1_complex", "COMPLEX", "🔴"),
        ]

        for level, expected_name, expected_emoji in levels:
            decision = RoutingDecision(
                task_description="Test",
                original_task="Test",
                complexity_level=level,
                complexity_score=0,
                depth_score=0.0,
                width_score=0.0,
                complexity_reasoning=[],
                domain="general",
                domain_reasoning=[],
                override_used=False,
                override_type=None,
                prompts_selected=[],
                safety_checks=[],
            )

            output = visualize_complexity(decision)
            assert expected_name in output
            assert expected_emoji in output

    def test_visualize_routing_methods(self):
        """Test visualization shows routing method."""
        methods = [
            ("keyword", "Keyword"),
            ("semantic", "Semantic"),
            ("hybrid", "Hybrid"),
            ("fallback", "Fallback"),
            ("override", "Override"),
        ]

        for method, expected_display in methods:
            decision = RoutingDecision(
                task_description="Test",
                original_task="Test",
                complexity_level="L1_moderate",
                complexity_score=0,
                depth_score=0.0,
                width_score=0.0,
                complexity_reasoning=[],
                domain="general",
                domain_reasoning=[],
                override_used=False,
                override_type=None,
                prompts_selected=[],
                safety_checks=[],
                routing_method=method,
            )

            output = visualize_complexity(decision)
            assert expected_display in output
