"""Unit tests for ACE Reflector module (UPGRADE-012.2).

Tests the pattern extraction and analysis capabilities of the ACE Reflector,
which implements the Reflector component from Stanford's ACE framework.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scripts.ace_reflector import (
    ACEReflector,
    PatternRecommendation,
    ReflectorAnalysis,
)


class TestPatternRecommendation:
    """Tests for PatternRecommendation dataclass."""

    def test_to_dict_basic(self):
        """Test basic dictionary conversion."""
        pattern = PatternRecommendation(
            pattern_type="keyword_gap",
            confidence=0.85,
            description="Test pattern",
            action="add_l3_pattern",
            pattern="test",
            weight=2,
            justification="Testing",
            supporting_sessions=["session1", "session2"],
        )

        result = pattern.to_dict()

        assert result["pattern_type"] == "keyword_gap"
        assert result["confidence"] == 0.85
        assert result["description"] == "Test pattern"
        assert result["recommendation"]["action"] == "add_l3_pattern"
        assert result["recommendation"]["pattern"] == "test"
        assert result["recommendation"]["weight"] == 2
        assert len(result["supporting_evidence"]) == 2

    def test_to_dict_limits_evidence(self):
        """Test that supporting evidence is limited to 5 entries."""
        pattern = PatternRecommendation(
            pattern_type="keyword_gap",
            confidence=0.8,
            description="Test",
            action="add_l3_pattern",
            pattern="test",
            weight=1,
            justification="Test",
            supporting_sessions=[f"session{i}" for i in range(10)],
        )

        result = pattern.to_dict()
        assert len(result["supporting_evidence"]) == 5


class TestReflectorAnalysis:
    """Tests for ReflectorAnalysis dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        analysis = ReflectorAnalysis(
            analysis_date="2025-12-03T12:00:00",
            sessions_analyzed=10,
            date_range={"start": "2025-12-01", "end": "2025-12-03"},
            patterns_found=[],
            overall_metrics={"success_rate": 0.8},
            top_keywords_in_failures=[("test", 5), ("bug", 3)],
        )

        result = analysis.to_dict()

        assert result["sessions_analyzed"] == 10
        assert result["overall_metrics"]["success_rate"] == 0.8
        assert len(result["top_keywords_in_failures"]) == 2
        assert result["top_keywords_in_failures"][0]["keyword"] == "test"


class TestACEReflector:
    """Tests for ACEReflector class."""

    @pytest.fixture
    def reflector(self, tmp_path: Path) -> ACEReflector:
        """Create a Reflector with a temporary log path."""
        log_path = tmp_path / "session-outcomes.jsonl"
        return ACEReflector(log_path=log_path)

    @pytest.fixture
    def sample_sessions(self) -> list[dict]:
        """Create sample session data for testing."""
        base_time = datetime.now()
        return [
            {
                "session_id": "session1",
                "timestamp": (base_time - timedelta(hours=1)).isoformat(),
                "task_description": "Implement authentication system",
                "routing_decision": {"complexity_level": "L1_simple"},
                "outcome": {"status": "success", "duration_minutes": 30},
                "feedback": {"classification_accurate": True},
            },
            {
                "session_id": "session2",
                "timestamp": (base_time - timedelta(hours=2)).isoformat(),
                "task_description": "Add authentication feature",
                "routing_decision": {"complexity_level": "L1_simple"},
                "outcome": {"status": "failed", "duration_minutes": 120},
                "feedback": {"classification_accurate": False},
            },
            {
                "session_id": "session3",
                "timestamp": (base_time - timedelta(hours=3)).isoformat(),
                "task_description": "Fix authentication bug",
                "routing_decision": {"complexity_level": "L1_simple"},
                "outcome": {"status": "partial", "duration_minutes": 90},
                "feedback": {"classification_accurate": False},
            },
            {
                "session_id": "session4",
                "timestamp": base_time.isoformat(),
                "task_description": "Update readme documentation",
                "routing_decision": {"complexity_level": "L1_simple"},
                "outcome": {"status": "success", "duration_minutes": 10},
                "feedback": {"classification_accurate": True},
            },
        ]

    def test_load_sessions_empty(self, reflector: ACEReflector):
        """Test loading from non-existent file returns empty list."""
        sessions = reflector.load_sessions()
        assert sessions == []

    def test_load_sessions_with_data(self, reflector: ACEReflector, sample_sessions: list[dict]):
        """Test loading sessions from JSONL file."""
        # Write sample data
        with open(reflector.log_path, "w") as f:
            for session in sample_sessions:
                f.write(json.dumps(session) + "\n")

        sessions = reflector.load_sessions()
        assert len(sessions) == 4

    def test_load_sessions_with_days_filter(self, reflector: ACEReflector, sample_sessions: list[dict]):
        """Test loading sessions with days filter."""
        # Write sample data
        with open(reflector.log_path, "w") as f:
            for session in sample_sessions:
                f.write(json.dumps(session) + "\n")

        # All sessions are within 1 day
        sessions = reflector.load_sessions(days=1)
        assert len(sessions) == 4

    def test_extract_keywords_basic(self, reflector: ACEReflector):
        """Test keyword extraction from text."""
        text = "Implement new authentication system with OAuth"
        keywords = reflector.extract_keywords(text)

        assert "implement" in keywords
        assert "authentication" in keywords
        assert "system" in keywords
        assert "oauth" in keywords
        assert "new" in keywords  # "new" is a meaningful word, not a stop word
        # Stop words should be filtered
        assert "with" not in keywords

    def test_extract_keywords_filters_short_words(self, reflector: ACEReflector):
        """Test that short words are filtered."""
        text = "a to in at it do fix the bug"
        keywords = reflector.extract_keywords(text)

        # "fix" and "bug" should be included (3+ chars, not stop words)
        assert "fix" in keywords
        assert "bug" in keywords
        # Short words should be filtered
        assert "to" not in keywords
        assert "in" not in keywords

    def test_analyze_empty(self, reflector: ACEReflector):
        """Test analyzing empty session log."""
        analysis = reflector.analyze()

        assert analysis.sessions_analyzed == 0
        assert analysis.patterns_found == []
        assert analysis.overall_metrics == {}

    def test_analyze_with_sessions(self, reflector: ACEReflector, sample_sessions: list[dict]):
        """Test analyzing sessions with data."""
        # Write sample data
        with open(reflector.log_path, "w") as f:
            for session in sample_sessions:
                f.write(json.dumps(session) + "\n")

        analysis = reflector.analyze()

        assert analysis.sessions_analyzed == 4
        assert "success_rate" in analysis.overall_metrics
        assert "failure_rate" in analysis.overall_metrics
        assert "misclassification_rate" in analysis.overall_metrics

    def test_analyze_detects_misclassification_patterns(self, reflector: ACEReflector, sample_sessions: list[dict]):
        """Test that analysis detects keyword patterns in misclassified tasks."""
        # Write sample data
        with open(reflector.log_path, "w") as f:
            for session in sample_sessions:
                f.write(json.dumps(session) + "\n")

        analysis = reflector.analyze()

        # Should find "authentication" as a common keyword in misclassified tasks
        keyword_patterns = [p for p in analysis.patterns_found if p.pattern_type == "keyword_gap"]
        keywords_found = [p.pattern for p in keyword_patterns]

        assert "authentication" in keywords_found

    def test_analyze_calculates_success_rate(self, reflector: ACEReflector, sample_sessions: list[dict]):
        """Test that success rate is calculated correctly."""
        # Write sample data
        with open(reflector.log_path, "w") as f:
            for session in sample_sessions:
                f.write(json.dumps(session) + "\n")

        analysis = reflector.analyze()

        # 2 successes out of 4 = 50%
        assert analysis.overall_metrics["success_rate"] == 0.5

    def test_generate_report(self, reflector: ACEReflector, sample_sessions: list[dict]):
        """Test report generation."""
        # Write sample data
        with open(reflector.log_path, "w") as f:
            for session in sample_sessions:
                f.write(json.dumps(session) + "\n")

        analysis = reflector.analyze()
        report = reflector.generate_report(analysis)

        assert "ACE Reflector Analysis Report" in report
        assert "Sessions Analyzed" in report
        assert "Overall Metrics" in report


class TestACEReflectorPatternDetection:
    """Tests for specific pattern detection scenarios."""

    @pytest.fixture
    def reflector(self, tmp_path: Path) -> ACEReflector:
        """Create a Reflector with a temporary log path."""
        log_path = tmp_path / "session-outcomes.jsonl"
        return ACEReflector(log_path=log_path)

    def test_detect_long_running_simple_tasks(self, reflector: ACEReflector):
        """Test detection of L1 tasks that take too long."""
        sessions = [
            {
                "session_id": f"session{i}",
                "timestamp": datetime.now().isoformat(),
                "task_description": f"Implement feature {i} with integration",
                "routing_decision": {"complexity_level": "L1_simple"},
                "outcome": {"status": "success", "duration_minutes": 120},
                "feedback": {"classification_accurate": False},
            }
            for i in range(3)
        ]

        with open(reflector.log_path, "w") as f:
            for session in sessions:
                f.write(json.dumps(session) + "\n")

        analysis = reflector.analyze()

        # Should detect complexity mismatch pattern
        mismatch_patterns = [p for p in analysis.patterns_found if p.pattern_type == "complexity_mismatch"]
        assert len(mismatch_patterns) > 0

    def test_detect_low_success_rate_level(self, reflector: ACEReflector):
        """Test detection of complexity levels with low success rates."""
        sessions = [
            {
                "session_id": f"session{i}",
                "timestamp": datetime.now().isoformat(),
                "task_description": f"Task {i}",
                "routing_decision": {"complexity_level": "L1_moderate"},
                "outcome": {"status": "failed" if i < 4 else "success", "duration_minutes": 60},
                "feedback": {},
            }
            for i in range(6)
        ]

        with open(reflector.log_path, "w") as f:
            for session in sessions:
                f.write(json.dumps(session) + "\n")

        analysis = reflector.analyze()

        # Should detect domain_miss pattern for low success rate
        domain_patterns = [p for p in analysis.patterns_found if p.pattern_type == "domain_miss"]
        # 4 failed out of 6 = 33% success rate (< 50% threshold)
        assert len(domain_patterns) > 0
