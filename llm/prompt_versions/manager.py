"""Prompt version management for A/B testing.

Provides infrastructure for versioning agent prompts, tracking their
performance, and supporting rollback when new versions underperform.

Usage:
    from llm.prompt_versions.manager import PromptVersionManager

    manager = PromptVersionManager()

    # Save a new version
    version_hash = manager.save_version(
        agent_name="technical_analyst",
        prompt_content="You are a technical analyst...",
        metadata={"author": "system", "reason": "Improved clarity"}
    )

    # Get current version
    current = manager.get_current("technical_analyst")
    print(current.content)

    # Compare versions
    comparison = manager.compare_versions(
        "technical_analyst",
        version_a="abc123",
        version_b="def456"
    )

    # Rollback if needed
    manager.rollback("technical_analyst", "abc123")
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


PROMPT_DIR = Path("llm/prompt_versions/data")


@dataclass
class PromptVersion:
    """A versioned prompt with metadata.

    Attributes:
        version_hash: Short hash identifying this version.
        content: The actual prompt text.
        created: When this version was created.
        metadata: Additional version metadata.
        performance: Performance metrics for this version.
    """

    version_hash: str
    content: str
    created: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version_hash,
            "content": self.content,
            "created": self.created.isoformat(),
            "metadata": self.metadata,
            "performance": self.performance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptVersion":
        """Create from dictionary."""
        return cls(
            version_hash=data["version"],
            content=data["content"],
            created=datetime.fromisoformat(data["created"]),
            metadata=data.get("metadata", {}),
            performance=data.get("performance", {}),
        )


@dataclass
class VersionComparison:
    """Comparison between two prompt versions.

    Attributes:
        version_a: First version hash.
        version_b: Second version hash.
        winner: Which version performed better (or None if inconclusive).
        metrics: Detailed metric comparison.
        recommendation: Action recommendation.
    """

    version_a: str
    version_b: str
    winner: str | None
    metrics: dict[str, dict[str, float]]
    recommendation: str


class PromptVersionManager:
    """Manages prompt versions for agents.

    Provides version control, A/B testing support, and rollback
    capability for agent prompts.
    """

    def __init__(self, base_dir: Path | None = None):
        """Initialize the version manager.

        Args:
            base_dir: Base directory for storing versions.
        """
        self.base_dir = base_dir or PROMPT_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _agent_dir(self, agent_name: str) -> Path:
        """Get the directory for an agent's versions."""
        agent_dir = self.base_dir / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        return agent_dir

    def _generate_hash(self, content: str) -> str:
        """Generate a short hash for content."""
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def save_version(
        self,
        agent_name: str,
        prompt_content: str,
        metadata: dict[str, Any] | None = None,
        set_as_current: bool = True,
    ) -> str:
        """Save a new prompt version.

        Args:
            agent_name: Name of the agent.
            prompt_content: The prompt text.
            metadata: Optional version metadata.
            set_as_current: Whether to set this as the current version.

        Returns:
            Version hash for the new version.
        """
        agent_dir = self._agent_dir(agent_name)
        version_hash = self._generate_hash(prompt_content)

        version = PromptVersion(
            version_hash=version_hash,
            content=prompt_content,
            created=datetime.now(),
            metadata=metadata or {},
        )

        # Save version file
        version_file = agent_dir / f"v_{version_hash}.json"
        version_file.write_text(json.dumps(version.to_dict(), indent=2))

        # Update current pointer if requested
        if set_as_current:
            self._set_current(agent_name, version_hash)

        return version_hash

    def _set_current(self, agent_name: str, version_hash: str) -> None:
        """Set the current version for an agent."""
        agent_dir = self._agent_dir(agent_name)
        current_file = agent_dir / "current.json"
        current_file.write_text(
            json.dumps(
                {
                    "current": version_hash,
                    "updated": datetime.now().isoformat(),
                }
            )
        )

    def get_current(self, agent_name: str) -> PromptVersion | None:
        """Get the current prompt version for an agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            Current PromptVersion or None if not found.
        """
        agent_dir = self._agent_dir(agent_name)
        current_file = agent_dir / "current.json"

        if not current_file.exists():
            return None

        try:
            current_data = json.loads(current_file.read_text())
            version_hash = current_data.get("current")
            if not version_hash:
                return None

            return self.get_version(agent_name, version_hash)
        except (json.JSONDecodeError, OSError):
            return None

    def get_version(self, agent_name: str, version_hash: str) -> PromptVersion | None:
        """Get a specific prompt version.

        Args:
            agent_name: Name of the agent.
            version_hash: Version hash to retrieve.

        Returns:
            PromptVersion or None if not found.
        """
        agent_dir = self._agent_dir(agent_name)
        version_file = agent_dir / f"v_{version_hash}.json"

        if not version_file.exists():
            return None

        try:
            data = json.loads(version_file.read_text())
            return PromptVersion.from_dict(data)
        except (json.JSONDecodeError, OSError):
            return None

    def get_history(self, agent_name: str, limit: int = 10) -> list[PromptVersion]:
        """Get version history for an agent.

        Args:
            agent_name: Name of the agent.
            limit: Maximum versions to return.

        Returns:
            List of PromptVersion objects, newest first.
        """
        agent_dir = self._agent_dir(agent_name)

        if not agent_dir.exists():
            return []

        versions = []
        for version_file in agent_dir.glob("v_*.json"):
            try:
                data = json.loads(version_file.read_text())
                versions.append(PromptVersion.from_dict(data))
            except (json.JSONDecodeError, OSError):
                continue

        # Sort by creation date, newest first
        versions.sort(key=lambda v: v.created, reverse=True)

        return versions[:limit]

    def update_performance(self, agent_name: str, version_hash: str, metrics: dict[str, float]) -> None:
        """Update performance metrics for a version.

        Args:
            agent_name: Name of the agent.
            version_hash: Version to update.
            metrics: Performance metrics to record.
        """
        version = self.get_version(agent_name, version_hash)
        if not version:
            return

        # Merge new metrics
        version.performance.update(metrics)

        # Save updated version
        agent_dir = self._agent_dir(agent_name)
        version_file = agent_dir / f"v_{version_hash}.json"
        version_file.write_text(json.dumps(version.to_dict(), indent=2))

    def compare_versions(self, agent_name: str, version_a: str, version_b: str) -> VersionComparison:
        """Compare two prompt versions based on performance.

        Args:
            agent_name: Name of the agent.
            version_a: First version hash.
            version_b: Second version hash.

        Returns:
            VersionComparison with analysis.
        """
        v_a = self.get_version(agent_name, version_a)
        v_b = self.get_version(agent_name, version_b)

        if not v_a or not v_b:
            return VersionComparison(
                version_a=version_a,
                version_b=version_b,
                winner=None,
                metrics={},
                recommendation="Cannot compare - one or both versions not found",
            )

        # Compare metrics
        metrics: dict[str, dict[str, float]] = {}
        all_metric_names = set(v_a.performance.keys()) | set(v_b.performance.keys())

        a_wins = 0
        b_wins = 0

        for metric in all_metric_names:
            a_val = v_a.performance.get(metric, 0.0)
            b_val = v_b.performance.get(metric, 0.0)

            metrics[metric] = {"version_a": a_val, "version_b": b_val}

            # Higher is better for most metrics
            if a_val > b_val:
                a_wins += 1
            elif b_val > a_val:
                b_wins += 1

        # Determine winner
        winner = None
        if a_wins > b_wins:
            winner = version_a
            recommendation = f"Version {version_a} outperforms on {a_wins} metrics"
        elif b_wins > a_wins:
            winner = version_b
            recommendation = f"Version {version_b} outperforms on {b_wins} metrics"
        else:
            recommendation = "No clear winner - consider other factors"

        return VersionComparison(
            version_a=version_a,
            version_b=version_b,
            winner=winner,
            metrics=metrics,
            recommendation=recommendation,
        )

    def rollback(self, agent_name: str, version_hash: str) -> bool:
        """Rollback to a previous version.

        Args:
            agent_name: Name of the agent.
            version_hash: Version to rollback to.

        Returns:
            True if successful, False if version not found.
        """
        version = self.get_version(agent_name, version_hash)
        if not version:
            return False

        self._set_current(agent_name, version_hash)
        return True

    def list_agents(self) -> list[str]:
        """List all agents with stored versions.

        Returns:
            List of agent names.
        """
        if not self.base_dir.exists():
            return []

        agents = []
        for agent_dir in self.base_dir.iterdir():
            if agent_dir.is_dir() and not agent_dir.name.startswith("_"):
                agents.append(agent_dir.name)

        return sorted(agents)


# Convenience functions for simple usage


def get_prompt_version(agent_name: str, variant: str = "current") -> dict[str, Any] | None:
    """Get a specific prompt version for an agent.

    Args:
        agent_name: Name of the agent.
        variant: Version hash or "current" for current version.

    Returns:
        Version data dictionary or None.
    """
    manager = PromptVersionManager()

    if variant == "current":
        version = manager.get_current(agent_name)
    else:
        version = manager.get_version(agent_name, variant)

    return version.to_dict() if version else None


def save_prompt_version(
    agent_name: str,
    prompt_content: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save a new prompt version.

    Args:
        agent_name: Name of the agent.
        prompt_content: The prompt text.
        metadata: Optional version metadata.

    Returns:
        Version hash.
    """
    manager = PromptVersionManager()
    return manager.save_version(agent_name, prompt_content, metadata)


def get_prompt_history(agent_name: str) -> list[dict[str, Any]]:
    """Get version history for an agent.

    Args:
        agent_name: Name of the agent.

    Returns:
        List of version info dictionaries.
    """
    manager = PromptVersionManager()
    versions = manager.get_history(agent_name)

    return [
        {
            "version": v.version_hash,
            "created": v.created.isoformat(),
            "metadata": v.metadata,
        }
        for v in versions
    ]
