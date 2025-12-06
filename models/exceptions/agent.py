"""LLM Agent exceptions."""

from .base import TradingError


class AgentError(TradingError):
    """Base class for agent errors."""

    pass


class AgentTimeoutError(AgentError):
    """Agent execution timed out."""

    def __init__(self, agent_name: str, timeout_ms: int):
        super().__init__(
            f"Agent '{agent_name}' timed out after {timeout_ms}ms",
            recoverable=True,
        )
        self.agent_name = agent_name
        self.timeout_ms = timeout_ms
        self.with_context(agent_name=agent_name)


class AgentRateLimitError(AgentError):
    """LLM API rate limit hit."""

    def __init__(
        self,
        agent_name: str,
        retry_after_seconds: int | None = None,
    ):
        msg = f"Agent '{agent_name}' hit rate limit"
        if retry_after_seconds:
            msg += f", retry after {retry_after_seconds}s"
        super().__init__(msg, recoverable=True)
        self.agent_name = agent_name
        self.retry_after_seconds = retry_after_seconds
        self.with_context(agent_name=agent_name)


class ConsensusFailedError(AgentError):
    """Multi-agent consensus failed."""

    def __init__(
        self,
        agents: list[str],
        reason: str,
        votes: dict[str, str] | None = None,
    ):
        super().__init__(
            f"Consensus failed among {len(agents)} agents: {reason}",
            recoverable=True,
        )
        self.agents = agents
        self.reason = reason
        self.votes = votes or {}


class AgentHallucinationError(AgentError):
    """Agent produced invalid/hallucinated output."""

    def __init__(self, agent_name: str, field: str, invalid_value: str):
        super().__init__(
            f"Agent '{agent_name}' produced invalid {field}: {invalid_value}",
            recoverable=True,
        )
        self.agent_name = agent_name
        self.field = field
        self.invalid_value = invalid_value
        self.with_context(agent_name=agent_name)


class PromptVersionError(AgentError):
    """Prompt version not found or invalid."""

    def __init__(self, agent_role: str, version: str):
        super().__init__(
            f"Prompt version '{version}' not found for {agent_role}",
            recoverable=False,
        )
        self.agent_role = agent_role
        self.version = version


class AgentConfigurationError(AgentError):
    """Agent configuration is invalid."""

    def __init__(self, agent_name: str, config_key: str, reason: str):
        super().__init__(
            f"Agent '{agent_name}' configuration error for '{config_key}': {reason}",
            recoverable=False,
        )
        self.agent_name = agent_name
        self.config_key = config_key
        self.reason = reason
        self.with_context(agent_name=agent_name)


class AgentCommunicationError(AgentError):
    """Error communicating with LLM provider."""

    def __init__(self, agent_name: str, provider: str, reason: str):
        super().__init__(
            f"Agent '{agent_name}' failed to communicate with {provider}: {reason}",
            recoverable=True,
        )
        self.agent_name = agent_name
        self.provider = provider
        self.reason = reason
        self.with_context(agent_name=agent_name)


class DebateResolutionError(AgentError):
    """Bull/Bear debate failed to reach resolution."""

    def __init__(
        self,
        bull_confidence: float,
        bear_confidence: float,
        rounds_completed: int,
    ):
        super().__init__(
            f"Debate failed after {rounds_completed} rounds: "
            f"Bull={bull_confidence:.0%}, Bear={bear_confidence:.0%}",
            recoverable=True,
        )
        self.bull_confidence = bull_confidence
        self.bear_confidence = bear_confidence
        self.rounds_completed = rounds_completed


__all__ = [
    "AgentCommunicationError",
    "AgentConfigurationError",
    "AgentError",
    "AgentHallucinationError",
    "AgentRateLimitError",
    "AgentTimeoutError",
    "ConsensusFailedError",
    "DebateResolutionError",
    "PromptVersionError",
]
