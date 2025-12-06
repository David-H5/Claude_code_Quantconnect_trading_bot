"""
Prompt Registry and Versioning System

Manages prompt templates for all trading agents with version control and A/B testing.

Features:
- Version control for prompts
- A/B testing framework
- Performance tracking
- Easy prompt iteration

QuantConnect Compatible: Yes
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class AgentRole(Enum):
    """Agent roles in the trading system."""

    SUPERVISOR = "supervisor"
    TECHNICAL_ANALYST = "technical_analyst"
    SENTIMENT_ANALYST = "sentiment_analyst"
    NEWS_ANALYST = "news_analyst"
    FUNDAMENTALS_ANALYST = "fundamentals_analyst"
    VOLATILITY_ANALYST = "volatility_analyst"
    BULL_RESEARCHER = "bull_researcher"
    BEAR_RESEARCHER = "bear_researcher"
    MARKET_REGIME_ANALYST = "market_regime_analyst"
    CONSERVATIVE_TRADER = "conservative_trader"
    MODERATE_TRADER = "moderate_trader"
    AGGRESSIVE_TRADER = "aggressive_trader"
    POSITION_RISK_MANAGER = "position_risk_manager"
    PORTFOLIO_RISK_MANAGER = "portfolio_risk_manager"
    CIRCUIT_BREAKER_MANAGER = "circuit_breaker_manager"


@dataclass
class PromptMetrics:
    """Performance metrics for a prompt version."""

    total_uses: int = 0
    successful_responses: int = 0
    failed_responses: int = 0
    avg_response_time_ms: float = 0.0
    avg_confidence: float = 0.0
    accuracy: float | None = None  # Requires ground truth
    user_feedback_score: float | None = None  # 1-5 rating


@dataclass
class PromptVersion:
    """A versioned prompt template."""

    role: AgentRole
    version: str  # e.g., "v1.0", "v1.1", "v2.0"
    template: str  # Prompt text with {variable} placeholders
    model: str  # Claude model: "opus-4", "sonnet-4", "haiku"
    temperature: float  # 0.0-1.0
    max_tokens: int
    created_date: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    description: str = ""
    changelog: str = ""
    metrics: PromptMetrics = field(default_factory=PromptMetrics)
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["role"] = self.role.value
        data["created_date"] = self.created_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptVersion":
        """Create from dictionary."""
        data = data.copy()
        data["role"] = AgentRole(data["role"])
        data["created_date"] = datetime.fromisoformat(data["created_date"])
        data["metrics"] = PromptMetrics(**data.get("metrics", {}))
        return cls(**data)


class PromptRegistry:
    """
    Central registry for all prompt templates.

    Features:
    - Store multiple versions per role
    - Track performance metrics
    - A/B testing support
    - Easy rollback to previous versions
    """

    def __init__(self, storage_path: Path | None = None):
        """
        Initialize prompt registry.

        Args:
            storage_path: Path to store prompt versions (JSON file)
        """
        self.storage_path = storage_path or Path("llm/prompts/registry.json")
        self.prompts: dict[str, list[PromptVersion]] = {}  # role -> versions
        self.active_versions: dict[str, str] = {}  # role -> active version
        self.load()

    def register(
        self,
        role: AgentRole,
        template: str,
        version: str,
        model: str = "sonnet-4",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        description: str = "",
        changelog: str = "",
        created_by: str = "system",
    ) -> PromptVersion:
        """
        Register a new prompt version.

        Args:
            role: Agent role this prompt is for
            template: Prompt text with {variable} placeholders
            version: Version string (e.g., "v1.0")
            model: Claude model ("opus-4", "sonnet-4", "haiku")
            temperature: Sampling temperature
            max_tokens: Max response tokens
            description: Description of this prompt
            changelog: What changed from previous version
            created_by: Who created this version

        Returns:
            PromptVersion instance
        """
        prompt = PromptVersion(
            role=role,
            version=version,
            template=template,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            description=description,
            changelog=changelog,
            created_by=created_by,
        )

        role_key = role.value
        if role_key not in self.prompts:
            self.prompts[role_key] = []

        self.prompts[role_key].append(prompt)

        # Set as active if first version or explicitly marked
        if role_key not in self.active_versions:
            self.active_versions[role_key] = version

        self.save()
        return prompt

    def get_prompt(self, role: AgentRole, version: str = "active") -> PromptVersion | None:
        """
        Get a specific prompt version.

        Args:
            role: Agent role
            version: Version string or "active" for current active version

        Returns:
            PromptVersion or None if not found
        """
        role_key = role.value

        if role_key not in self.prompts:
            return None

        if version == "active":
            version = self.active_versions.get(role_key)
            if not version:
                return None

        for prompt in self.prompts[role_key]:
            if prompt.version == version and prompt.active:
                return prompt

        return None

    def set_active(self, role: AgentRole, version: str) -> bool:
        """
        Set a version as the active version for a role.

        Args:
            role: Agent role
            version: Version to activate

        Returns:
            True if successful, False if version not found
        """
        role_key = role.value

        if role_key not in self.prompts:
            return False

        # Check version exists
        found = False
        for prompt in self.prompts[role_key]:
            if prompt.version == version:
                found = True
                break

        if not found:
            return False

        self.active_versions[role_key] = version
        self.save()
        return True

    def get_all_versions(self, role: AgentRole) -> list[PromptVersion]:
        """Get all versions for a role."""
        role_key = role.value
        return self.prompts.get(role_key, [])

    def record_usage(
        self,
        role: AgentRole,
        version: str,
        success: bool,
        response_time_ms: float,
        confidence: float | None = None,
    ) -> None:
        """
        Record usage metrics for a prompt.

        Args:
            role: Agent role
            version: Prompt version used
            success: Whether the response was successful
            response_time_ms: Response time in milliseconds
            confidence: Agent's confidence score (0-1)
        """
        prompt = self.get_prompt(role, version)
        if not prompt:
            return

        metrics = prompt.metrics
        metrics.total_uses += 1

        if success:
            metrics.successful_responses += 1
        else:
            metrics.failed_responses += 1

        # Update running averages
        n = metrics.total_uses
        metrics.avg_response_time_ms = (metrics.avg_response_time_ms * (n - 1) + response_time_ms) / n

        if confidence is not None:
            if metrics.avg_confidence == 0:
                metrics.avg_confidence = confidence
            else:
                metrics.avg_confidence = (metrics.avg_confidence * (n - 1) + confidence) / n

        self.save()

    def record_accuracy(self, role: AgentRole, version: str, correct: bool) -> None:
        """
        Record accuracy for a prompt (requires ground truth).

        Args:
            role: Agent role
            version: Prompt version
            correct: Whether the agent's decision was correct
        """
        prompt = self.get_prompt(role, version)
        if not prompt:
            return

        metrics = prompt.metrics
        if metrics.accuracy is None:
            metrics.accuracy = 1.0 if correct else 0.0
        else:
            n = metrics.total_uses
            metrics.accuracy = (metrics.accuracy * (n - 1) + (1.0 if correct else 0.0)) / n

        self.save()

    def get_best_version(self, role: AgentRole, metric: str = "accuracy") -> PromptVersion | None:
        """
        Get the best-performing version for a role based on a metric.

        Args:
            role: Agent role
            metric: Metric to optimize ("accuracy", "avg_confidence", "avg_response_time_ms")

        Returns:
            Best PromptVersion or None
        """
        versions = self.get_all_versions(role)
        if not versions:
            return None

        # Filter versions with enough data (at least 10 uses)
        valid_versions = [v for v in versions if v.metrics.total_uses >= 10]
        if not valid_versions:
            return versions[0]  # Return latest if not enough data

        if metric == "accuracy":
            # Higher is better
            return max(valid_versions, key=lambda v: v.metrics.accuracy or 0)
        elif metric == "avg_confidence":
            # Higher is better
            return max(valid_versions, key=lambda v: v.metrics.avg_confidence)
        elif metric == "avg_response_time_ms":
            # Lower is better
            return min(valid_versions, key=lambda v: v.metrics.avg_response_time_ms)
        else:
            return None

    def compare_versions(self, role: AgentRole, version1: str, version2: str) -> dict[str, Any]:
        """
        Compare two prompt versions.

        Args:
            role: Agent role
            version1: First version
            version2: Second version

        Returns:
            Comparison dictionary
        """
        prompt1 = self.get_prompt(role, version1)
        prompt2 = self.get_prompt(role, version2)

        if not prompt1 or not prompt2:
            return {"error": "One or both versions not found"}

        return {
            "version1": version1,
            "version2": version2,
            "metrics_comparison": {
                "accuracy": {
                    "v1": prompt1.metrics.accuracy,
                    "v2": prompt2.metrics.accuracy,
                    "winner": "v1" if (prompt1.metrics.accuracy or 0) > (prompt2.metrics.accuracy or 0) else "v2",
                },
                "avg_confidence": {
                    "v1": prompt1.metrics.avg_confidence,
                    "v2": prompt2.metrics.avg_confidence,
                    "winner": "v1" if prompt1.metrics.avg_confidence > prompt2.metrics.avg_confidence else "v2",
                },
                "avg_response_time_ms": {
                    "v1": prompt1.metrics.avg_response_time_ms,
                    "v2": prompt2.metrics.avg_response_time_ms,
                    "winner": "v1"
                    if prompt1.metrics.avg_response_time_ms < prompt2.metrics.avg_response_time_ms
                    else "v2",
                },
                "total_uses": {
                    "v1": prompt1.metrics.total_uses,
                    "v2": prompt2.metrics.total_uses,
                },
            },
        }

    def save(self) -> None:
        """Save registry to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "prompts": {role: [p.to_dict() for p in versions] for role, versions in self.prompts.items()},
            "active_versions": self.active_versions,
        }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load registry from disk."""
        if not self.storage_path.exists():
            return

        with open(self.storage_path) as f:
            data = json.load(f)

        self.prompts = {
            role: [PromptVersion.from_dict(p) for p in versions] for role, versions in data.get("prompts", {}).items()
        }
        self.active_versions = data.get("active_versions", {})


# Global registry instance
_registry: PromptRegistry | None = None


def get_registry() -> PromptRegistry:
    """Get the global prompt registry."""
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry


def register_prompt(role: AgentRole, template: str, version: str, **kwargs) -> PromptVersion:
    """Convenience function to register a prompt."""
    return get_registry().register(role, template, version, **kwargs)


def get_prompt(role: AgentRole, version: str = "active") -> PromptVersion | None:
    """Convenience function to get a prompt."""
    return get_registry().get_prompt(role, version)
