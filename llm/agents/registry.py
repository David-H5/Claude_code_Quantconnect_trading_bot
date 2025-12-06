"""
Agent Registry - Micro-Agent Pattern Implementation

Centralized registry for agent discovery, capability tracking, and health monitoring.
Implements the micro-agent pattern where each agent has:
- PURPOSE: Immutable role/responsibility
- CAPABILITIES: What the agent can do
- TASK: Current assignment (mutable)

UPGRADE-014 Category 1: Architecture Enhancements

QuantConnect Compatible: Yes
- No blocking operations
- Thread-safe singleton
- Defensive error handling
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING, Any, Optional

from models.exceptions import AgentConfigurationError


if TYPE_CHECKING:
    from llm.agents.base import AgentRole, TradingAgent

logger = logging.getLogger(__name__)


class AgentHealth(Enum):
    """Agent health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Slow but functional
    UNHEALTHY = "unhealthy"  # Failing
    UNKNOWN = "unknown"  # Not checked yet


class CapabilityCategory(Enum):
    """Categories of agent capabilities."""

    ANALYSIS = "analysis"
    TRADING = "trading"
    RISK = "risk"
    ORCHESTRATION = "orchestration"
    RESEARCH = "research"
    SENTIMENT = "sentiment"


@dataclass
class AgentCapability:
    """
    Describes what an agent can do.

    Attributes:
        name: Capability name (e.g., "technical_analysis")
        description: Human-readable description
        category: Capability category
        tools: List of tool names this capability uses
        input_schema: Expected input format
        output_schema: Expected output format
    """

    name: str
    description: str
    category: CapabilityCategory
    tools: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "tools": self.tools,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }


@dataclass
class AgentRegistration:
    """
    Registration record for an agent.

    Attributes:
        agent: The registered agent instance
        capabilities: List of agent's capabilities
        health: Current health status
        registered_at: Registration timestamp
        last_health_check: Last health check timestamp
        metadata: Additional registration metadata
    """

    agent: "TradingAgent"
    capabilities: list[AgentCapability]
    health: AgentHealth = AgentHealth.UNKNOWN
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_health_check: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.agent.name

    @property
    def role(self) -> "AgentRole":
        """Get agent role."""
        return self.agent.role

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without agent instance)."""
        return {
            "name": self.name,
            "role": self.role.value,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "health": self.health.value,
            "registered_at": self.registered_at.isoformat(),
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "metadata": self.metadata,
        }


class AgentRegistry:
    """
    Centralized registry for micro-agents.

    Provides:
    - Agent registration and discovery
    - Capability-based routing
    - Health monitoring
    - Dynamic agent loading

    Thread-safe singleton implementation.

    Usage:
        registry = AgentRegistry()

        # Register agent
        registry.register(
            agent=my_agent,
            capabilities=[
                AgentCapability(
                    name="technical_analysis",
                    description="Analyze price charts and patterns",
                    category=CapabilityCategory.ANALYSIS,
                ),
            ],
        )

        # Discover agents
        analysts = registry.discover(category=CapabilityCategory.ANALYSIS)

        # Check health
        health = registry.health_check("my_agent")
    """

    def __init__(self):
        """Initialize empty registry."""
        self._agents: dict[str, AgentRegistration] = {}
        self._lock = Lock()
        self._capability_index: dict[str, set[str]] = {}  # capability -> agent names
        self._role_index: dict[str, set[str]] = {}  # role -> agent names
        self._category_index: dict[CapabilityCategory, set[str]] = {}

    def register(
        self,
        agent: "TradingAgent",
        capabilities: list[AgentCapability],
        metadata: dict[str, Any] | None = None,
    ) -> AgentRegistration:
        """
        Register an agent with its capabilities.

        Args:
            agent: The agent instance to register
            capabilities: List of agent's capabilities
            metadata: Optional metadata

        Returns:
            AgentRegistration record

        Raises:
            AgentConfigurationError: If agent with same name already registered
        """
        with self._lock:
            if agent.name in self._agents:
                raise AgentConfigurationError(
                    agent_name=agent.name,
                    config_key="registration",
                    reason="Agent already registered",
                )

            registration = AgentRegistration(
                agent=agent,
                capabilities=capabilities,
                metadata=metadata or {},
            )

            self._agents[agent.name] = registration

            # Update indices
            role_key = agent.role.value
            if role_key not in self._role_index:
                self._role_index[role_key] = set()
            self._role_index[role_key].add(agent.name)

            for cap in capabilities:
                if cap.name not in self._capability_index:
                    self._capability_index[cap.name] = set()
                self._capability_index[cap.name].add(agent.name)

                if cap.category not in self._category_index:
                    self._category_index[cap.category] = set()
                self._category_index[cap.category].add(agent.name)

            logger.info(f"Registered agent '{agent.name}' with {len(capabilities)} capabilities")
            return registration

    def unregister(self, agent_name: str) -> bool:
        """
        Remove an agent from the registry.

        Args:
            agent_name: Name of agent to remove

        Returns:
            True if agent was removed, False if not found
        """
        with self._lock:
            if agent_name not in self._agents:
                return False

            registration = self._agents.pop(agent_name)

            # Update indices
            role_key = registration.role.value
            if role_key in self._role_index:
                self._role_index[role_key].discard(agent_name)

            for cap in registration.capabilities:
                if cap.name in self._capability_index:
                    self._capability_index[cap.name].discard(agent_name)
                if cap.category in self._category_index:
                    self._category_index[cap.category].discard(agent_name)

            logger.info(f"Unregistered agent '{agent_name}'")
            return True

    def get(self, agent_name: str) -> Optional["TradingAgent"]:
        """
        Get agent by name.

        Args:
            agent_name: Name of agent

        Returns:
            Agent instance or None if not found
        """
        with self._lock:
            registration = self._agents.get(agent_name)
            return registration.agent if registration else None

    def get_registration(self, agent_name: str) -> AgentRegistration | None:
        """
        Get full registration record by name.

        Args:
            agent_name: Name of agent

        Returns:
            AgentRegistration or None if not found
        """
        with self._lock:
            return self._agents.get(agent_name)

    def discover(
        self,
        role: Optional["AgentRole"] = None,
        capability: str | None = None,
        category: CapabilityCategory | None = None,
        health: AgentHealth | None = None,
    ) -> list["TradingAgent"]:
        """
        Discover agents matching criteria.

        Args:
            role: Filter by agent role
            capability: Filter by capability name
            category: Filter by capability category
            health: Filter by health status

        Returns:
            List of matching agents
        """
        with self._lock:
            candidates = set(self._agents.keys())

            # Filter by role
            if role is not None:
                role_matches = self._role_index.get(role.value, set())
                candidates = candidates.intersection(role_matches)

            # Filter by capability
            if capability is not None:
                cap_matches = self._capability_index.get(capability, set())
                candidates = candidates.intersection(cap_matches)

            # Filter by category
            if category is not None:
                cat_matches = self._category_index.get(category, set())
                candidates = candidates.intersection(cat_matches)

            # Filter by health
            if health is not None:
                candidates = {name for name in candidates if self._agents[name].health == health}

            return [self._agents[name].agent for name in candidates]

    def health_check(
        self,
        agent_name: str | None = None,
        check_function: Callable[["TradingAgent"], AgentHealth] | None = None,
    ) -> dict[str, AgentHealth]:
        """
        Check health of agent(s).

        Args:
            agent_name: Specific agent to check, or None for all
            check_function: Custom health check function

        Returns:
            Dictionary of agent name -> health status
        """
        with self._lock:
            if agent_name:
                agents_to_check = [agent_name] if agent_name in self._agents else []
            else:
                agents_to_check = list(self._agents.keys())

            results = {}
            for name in agents_to_check:
                registration = self._agents[name]

                if check_function:
                    try:
                        health = check_function(registration.agent)
                    except Exception as e:
                        logger.warning(f"Health check failed for '{name}': {e}")
                        health = AgentHealth.UNHEALTHY
                else:
                    # Default check: verify agent has required attributes
                    try:
                        if hasattr(registration.agent, "analyze") and callable(registration.agent.analyze):
                            health = AgentHealth.HEALTHY
                        else:
                            health = AgentHealth.DEGRADED
                    except Exception:
                        health = AgentHealth.UNHEALTHY

                registration.health = health
                registration.last_health_check = datetime.utcnow()
                results[name] = health

            return results

    def list_all(self) -> list[AgentRegistration]:
        """
        List all registered agents.

        Returns:
            List of all registrations
        """
        with self._lock:
            return list(self._agents.values())

    def list_capabilities(self) -> list[str]:
        """
        List all registered capabilities.

        Returns:
            List of capability names
        """
        with self._lock:
            return list(self._capability_index.keys())

    def list_by_category(self) -> dict[str, list[str]]:
        """
        List agents grouped by category.

        Returns:
            Dictionary of category -> agent names
        """
        with self._lock:
            return {cat.value: list(names) for cat, names in self._category_index.items()}

    def count(self) -> int:
        """Get number of registered agents."""
        with self._lock:
            return len(self._agents)

    def clear(self) -> None:
        """Clear all registrations (use with caution)."""
        with self._lock:
            self._agents.clear()
            self._capability_index.clear()
            self._role_index.clear()
            self._category_index.clear()
            logger.info("Cleared agent registry")

    def to_dict(self) -> dict[str, Any]:
        """
        Export registry state as dictionary.

        Returns:
            Dictionary representation of registry
        """
        with self._lock:
            return {
                "agents": {name: reg.to_dict() for name, reg in self._agents.items()},
                "capabilities": list(self._capability_index.keys()),
                "roles": list(self._role_index.keys()),
                "categories": [cat.value for cat in self._category_index.keys()],
                "count": len(self._agents),
            }


# Singleton instance
_global_registry: AgentRegistry | None = None
_registry_lock = Lock()


def get_global_registry() -> AgentRegistry:
    """
    Get the global agent registry singleton.

    Returns:
        Global AgentRegistry instance
    """
    global _global_registry

    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = AgentRegistry()

    return _global_registry


def register_agent(
    capabilities: list[AgentCapability],
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator to auto-register agent class on instantiation.

    Usage:
        @register_agent(
            capabilities=[
                AgentCapability(
                    name="technical_analysis",
                    description="Analyze charts",
                    category=CapabilityCategory.ANALYSIS,
                ),
            ],
        )
        class MyAnalyst(TradingAgent):
            ...

    Args:
        capabilities: Agent capabilities
        metadata: Optional metadata

    Returns:
        Decorator function
    """

    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Auto-register after initialization
            try:
                registry = get_global_registry()
                registry.register(
                    agent=self,
                    capabilities=capabilities,
                    metadata=metadata,
                )
            except AgentConfigurationError:
                # Already registered (e.g., singleton pattern)
                pass

        cls.__init__ = new_init
        return cls

    return decorator


# Pre-defined capabilities for common agent types
TECHNICAL_ANALYSIS_CAPABILITY = AgentCapability(
    name="technical_analysis",
    description="Analyze price charts, patterns, and technical indicators",
    category=CapabilityCategory.ANALYSIS,
    tools=["get_price_data", "calculate_indicators", "identify_patterns"],
)

SENTIMENT_ANALYSIS_CAPABILITY = AgentCapability(
    name="sentiment_analysis",
    description="Analyze market sentiment from news and social media",
    category=CapabilityCategory.SENTIMENT,
    tools=["get_news", "analyze_sentiment", "aggregate_signals"],
)

TRADING_EXECUTION_CAPABILITY = AgentCapability(
    name="trading_execution",
    description="Execute trades and manage orders",
    category=CapabilityCategory.TRADING,
    tools=["place_order", "modify_order", "cancel_order"],
)

RISK_ASSESSMENT_CAPABILITY = AgentCapability(
    name="risk_assessment",
    description="Assess portfolio and position risk",
    category=CapabilityCategory.RISK,
    tools=["calculate_var", "check_limits", "analyze_exposure"],
)

ORCHESTRATION_CAPABILITY = AgentCapability(
    name="orchestration",
    description="Coordinate multiple agents and make final decisions",
    category=CapabilityCategory.ORCHESTRATION,
    tools=["gather_analyses", "synthesize_views", "make_decision"],
)


def create_registry() -> AgentRegistry:
    """
    Factory function to create a new registry instance.

    Use this when you need a separate registry (e.g., for testing).
    For most use cases, prefer get_global_registry().

    Returns:
        New AgentRegistry instance
    """
    return AgentRegistry()
