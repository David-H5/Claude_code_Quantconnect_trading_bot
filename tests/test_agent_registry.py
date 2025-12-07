"""
Tests for Agent Registry

UPGRADE-014 Category 1: Architecture Enhancements
Tests for micro-agent pattern implementation with registry and discovery.
"""

from unittest.mock import patch

import pytest

from llm.agents.base import AgentResponse, AgentRole, TradingAgent
from models.exceptions.agent import AgentConfigurationError
from llm.agents.registry import (
    ORCHESTRATION_CAPABILITY,
    RISK_ASSESSMENT_CAPABILITY,
    SENTIMENT_ANALYSIS_CAPABILITY,
    TECHNICAL_ANALYSIS_CAPABILITY,
    AgentCapability,
    AgentHealth,
    AgentRegistration,
    CapabilityCategory,
    create_registry,
    get_global_registry,
    register_agent,
)


class MockAgent(TradingAgent):
    """Mock agent for testing."""

    def __init__(self, name: str, role: AgentRole):
        super().__init__(
            name=name,
            role=role,
            system_prompt="Test agent",
            tools=[],
            max_iterations=3,
            timeout_ms=1000.0,
        )

    def analyze(self, query: str, context: dict) -> AgentResponse:
        """Mock analyze method."""
        return AgentResponse(
            agent_name=self.name,
            agent_role=self.role,
            query=query,
            thoughts=[],
            final_answer="Mock analysis",
            confidence=0.8,
            tools_used=[],
            execution_time_ms=100.0,
            success=True,
        )


@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    return create_registry()


@pytest.fixture
def mock_analyst():
    """Create a mock analyst agent."""
    return MockAgent("test_analyst", AgentRole.TECHNICAL_ANALYST)


@pytest.fixture
def mock_trader():
    """Create a mock trader agent."""
    return MockAgent("test_trader", AgentRole.CONSERVATIVE_TRADER)


@pytest.fixture
def mock_risk_manager():
    """Create a mock risk manager agent."""
    return MockAgent("test_risk_manager", AgentRole.POSITION_RISK_MANAGER)


class TestAgentCapability:
    """Tests for AgentCapability dataclass."""

    def test_capability_creation(self):
        """Test creating a capability."""
        cap = AgentCapability(
            name="test_cap",
            description="Test capability",
            category=CapabilityCategory.ANALYSIS,
            tools=["tool1", "tool2"],
        )

        assert cap.name == "test_cap"
        assert cap.category == CapabilityCategory.ANALYSIS
        assert len(cap.tools) == 2

    def test_capability_to_dict(self):
        """Test capability serialization."""
        cap = AgentCapability(
            name="test_cap",
            description="Test",
            category=CapabilityCategory.TRADING,
        )

        d = cap.to_dict()
        assert d["name"] == "test_cap"
        assert d["category"] == "trading"

    def test_predefined_capabilities(self):
        """Test predefined capability constants."""
        assert TECHNICAL_ANALYSIS_CAPABILITY.category == CapabilityCategory.ANALYSIS
        assert SENTIMENT_ANALYSIS_CAPABILITY.category == CapabilityCategory.SENTIMENT
        assert RISK_ASSESSMENT_CAPABILITY.category == CapabilityCategory.RISK
        assert ORCHESTRATION_CAPABILITY.category == CapabilityCategory.ORCHESTRATION


class TestAgentRegistration:
    """Tests for AgentRegistration dataclass."""

    def test_registration_properties(self, mock_analyst):
        """Test registration properties."""
        reg = AgentRegistration(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )

        assert reg.name == "test_analyst"
        assert reg.role == AgentRole.TECHNICAL_ANALYST
        assert reg.health == AgentHealth.UNKNOWN

    def test_registration_to_dict(self, mock_analyst):
        """Test registration serialization."""
        reg = AgentRegistration(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
            metadata={"version": "1.0"},
        )

        d = reg.to_dict()
        assert d["name"] == "test_analyst"
        assert d["role"] == "technical_analyst"
        assert len(d["capabilities"]) == 1
        assert d["metadata"]["version"] == "1.0"


class TestAgentRegistry:
    """Tests for AgentRegistry class."""

    def test_register_agent(self, registry, mock_analyst):
        """Test agent registration."""
        reg = registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )

        assert reg.name == "test_analyst"
        assert registry.count() == 1

    def test_register_duplicate_raises(self, registry, mock_analyst):
        """Test that duplicate registration raises error."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )

        with pytest.raises(AgentConfigurationError, match="already registered"):
            registry.register(
                agent=mock_analyst,
                capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
            )

    def test_unregister_agent(self, registry, mock_analyst):
        """Test agent unregistration."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )

        assert registry.unregister("test_analyst") is True
        assert registry.count() == 0

    def test_unregister_nonexistent(self, registry):
        """Test unregistering nonexistent agent."""
        assert registry.unregister("nonexistent") is False

    def test_get_agent(self, registry, mock_analyst):
        """Test getting agent by name."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )

        agent = registry.get("test_analyst")
        assert agent is mock_analyst

    def test_get_nonexistent(self, registry):
        """Test getting nonexistent agent."""
        assert registry.get("nonexistent") is None

    def test_get_registration(self, registry, mock_analyst):
        """Test getting full registration."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
            metadata={"test": True},
        )

        reg = registry.get_registration("test_analyst")
        assert reg is not None
        assert reg.metadata["test"] is True


class TestAgentDiscovery:
    """Tests for agent discovery functionality."""

    def test_discover_by_role(self, registry, mock_analyst, mock_trader):
        """Test discovering agents by role."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )
        registry.register(
            agent=mock_trader,
            capabilities=[
                AgentCapability(
                    name="trading",
                    description="Execute trades",
                    category=CapabilityCategory.TRADING,
                )
            ],
        )

        analysts = registry.discover(role=AgentRole.TECHNICAL_ANALYST)
        assert len(analysts) == 1
        assert analysts[0].name == "test_analyst"

    def test_discover_by_capability(self, registry, mock_analyst, mock_trader):
        """Test discovering agents by capability."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )
        registry.register(
            agent=mock_trader,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],  # Both have this
        )

        agents = registry.discover(capability="technical_analysis")
        assert len(agents) == 2

    def test_discover_by_category(self, registry, mock_analyst, mock_risk_manager):
        """Test discovering agents by category."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )
        registry.register(
            agent=mock_risk_manager,
            capabilities=[RISK_ASSESSMENT_CAPABILITY],
        )

        risk_agents = registry.discover(category=CapabilityCategory.RISK)
        assert len(risk_agents) == 1
        assert risk_agents[0].name == "test_risk_manager"

    def test_discover_by_health(self, registry, mock_analyst, mock_trader):
        """Test discovering agents by health status."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )
        registry.register(
            agent=mock_trader,
            capabilities=[
                AgentCapability(
                    name="trading",
                    description="Trade",
                    category=CapabilityCategory.TRADING,
                )
            ],
        )

        # Set health status
        registry.get_registration("test_analyst").health = AgentHealth.HEALTHY
        registry.get_registration("test_trader").health = AgentHealth.DEGRADED

        healthy = registry.discover(health=AgentHealth.HEALTHY)
        assert len(healthy) == 1
        assert healthy[0].name == "test_analyst"

    def test_discover_combined_filters(self, registry, mock_analyst, mock_trader):
        """Test discovery with multiple filters."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )
        registry.register(
            agent=mock_trader,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )

        # Only analyst matches role AND capability
        agents = registry.discover(
            role=AgentRole.TECHNICAL_ANALYST,
            capability="technical_analysis",
        )
        assert len(agents) == 1
        assert agents[0].name == "test_analyst"

    def test_discover_no_matches(self, registry, mock_analyst):
        """Test discovery with no matching agents."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )

        # No risk agents registered
        agents = registry.discover(category=CapabilityCategory.RISK)
        assert len(agents) == 0


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_default_health_check(self, registry, mock_analyst):
        """Test default health check logic."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )

        results = registry.health_check("test_analyst")
        assert results["test_analyst"] == AgentHealth.HEALTHY

    def test_health_check_all(self, registry, mock_analyst, mock_trader):
        """Test checking health of all agents."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )
        registry.register(
            agent=mock_trader,
            capabilities=[
                AgentCapability(
                    name="trading",
                    description="Trade",
                    category=CapabilityCategory.TRADING,
                )
            ],
        )

        results = registry.health_check()
        assert len(results) == 2
        assert all(h == AgentHealth.HEALTHY for h in results.values())

    def test_custom_health_check(self, registry, mock_analyst):
        """Test custom health check function."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )

        def custom_check(agent):
            return AgentHealth.DEGRADED

        results = registry.health_check("test_analyst", check_function=custom_check)
        assert results["test_analyst"] == AgentHealth.DEGRADED

    def test_health_check_updates_registration(self, registry, mock_analyst):
        """Test that health check updates registration record."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )

        registry.health_check("test_analyst")

        reg = registry.get_registration("test_analyst")
        assert reg.health == AgentHealth.HEALTHY
        assert reg.last_health_check is not None

    def test_health_check_nonexistent(self, registry):
        """Test health check on nonexistent agent."""
        results = registry.health_check("nonexistent")
        assert len(results) == 0


class TestRegistryUtilities:
    """Tests for utility methods."""

    def test_list_all(self, registry, mock_analyst, mock_trader):
        """Test listing all registrations."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )
        registry.register(
            agent=mock_trader,
            capabilities=[
                AgentCapability(
                    name="trading",
                    description="Trade",
                    category=CapabilityCategory.TRADING,
                )
            ],
        )

        all_agents = registry.list_all()
        assert len(all_agents) == 2

    def test_list_capabilities(self, registry, mock_analyst):
        """Test listing capabilities."""
        registry.register(
            agent=mock_analyst,
            capabilities=[
                TECHNICAL_ANALYSIS_CAPABILITY,
                SENTIMENT_ANALYSIS_CAPABILITY,
            ],
        )

        caps = registry.list_capabilities()
        assert "technical_analysis" in caps
        assert "sentiment_analysis" in caps

    def test_list_by_category(self, registry, mock_analyst, mock_risk_manager):
        """Test listing agents by category."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )
        registry.register(
            agent=mock_risk_manager,
            capabilities=[RISK_ASSESSMENT_CAPABILITY],
        )

        by_cat = registry.list_by_category()
        assert "analysis" in by_cat
        assert "risk" in by_cat
        assert "test_analyst" in by_cat["analysis"]

    def test_clear(self, registry, mock_analyst):
        """Test clearing registry."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )

        registry.clear()
        assert registry.count() == 0
        assert len(registry.list_capabilities()) == 0

    def test_to_dict(self, registry, mock_analyst):
        """Test registry serialization."""
        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )

        d = registry.to_dict()
        assert "agents" in d
        assert "test_analyst" in d["agents"]
        assert d["count"] == 1


class TestGlobalRegistry:
    """Tests for global registry singleton."""

    def test_global_registry_singleton(self):
        """Test that global registry is a singleton."""
        with patch("llm.agents.registry._global_registry", None):
            reg1 = get_global_registry()
            reg2 = get_global_registry()
            assert reg1 is reg2

    def test_create_registry_returns_new(self):
        """Test that create_registry returns new instances."""
        reg1 = create_registry()
        reg2 = create_registry()
        assert reg1 is not reg2


class TestRegisterDecorator:
    """Tests for @register_agent decorator."""

    def test_auto_registration(self):
        """Test automatic agent registration via decorator."""
        # Create a fresh registry for this test
        test_registry = create_registry()

        with patch("llm.agents.registry.get_global_registry", return_value=test_registry):

            @register_agent(
                capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
                metadata={"version": "1.0"},
            )
            class DecoratedAgent(TradingAgent):
                def __init__(self):
                    super().__init__(
                        name="decorated_agent",
                        role=AgentRole.ANALYST,
                        system_prompt="Decorated",
                    )

                def analyze(self, query, context):
                    return None

            # Instantiate the agent
            agent = DecoratedAgent()

            # Should be automatically registered
            assert test_registry.count() == 1
            assert test_registry.get("decorated_agent") is agent


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_registration(self, registry):
        """Test concurrent agent registration."""
        import threading

        errors = []

        def register_agent(name):
            try:
                agent = MockAgent(name, AgentRole.ANALYST)
                registry.register(
                    agent=agent,
                    capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_agent, args=(f"agent_{i}",)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All registrations should succeed
        assert len(errors) == 0
        assert registry.count() == 10

    def test_concurrent_discovery(self, registry, mock_analyst):
        """Test concurrent agent discovery."""
        import threading

        registry.register(
            agent=mock_analyst,
            capabilities=[TECHNICAL_ANALYSIS_CAPABILITY],
        )

        results = []

        def discover():
            agents = registry.discover(category=CapabilityCategory.ANALYSIS)
            results.append(len(agents))

        threads = [threading.Thread(target=discover) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All discoveries should return same result
        assert all(r == 1 for r in results)
