"""
LLM Trading Agents

Multi-agent trading system with specialized roles:
- Supervisor: Orchestrates team decisions
- Technical Analyst: Chart analysis and patterns
- Sentiment Analyst: News + social + FinBERT sentiment
- Traders: Strategy design (Conservative, Moderate, Aggressive)
- Risk Managers: Position and portfolio risk checks

QuantConnect Compatible: Yes
"""

from llm.agents.base import (
    AgentResponse,
    AgentRole,
    AgentThought,
    LoopMetrics,
    TerminationReason,
    ThoughtType,
    TradingAgent,
)
from llm.agents.bear_researcher import (
    BearResearcher,
    create_bear_researcher,
)
from llm.agents.bull_researcher import (
    BullResearcher,
    create_bull_researcher,
)

# Bull/Bear Debate Mechanism (UPGRADE-004 - December 2025)
from llm.agents.debate_mechanism import (
    BullBearDebate,
    DebateConfig,
    DebateOutcome,
    DebateResult,
    DebateRound,
    DebateTrigger,
    create_debate_mechanism,
    generate_debate_report,
)

# Multi-Agent Consensus (UPGRADE-014 Feature 6: Multi-Agent Architecture)
from llm.agents.multi_agent_consensus import (
    AgentOpinion,
    AgentType,
    ConsensusConfig,
    ConsensusResult,
    ConsensusSignal,
    MultiAgentConsensus,
    create_multi_agent_consensus,
    opinion_from_agent_response,
)

# News Analyst (UPGRADE-014 Feature 6: Multi-Agent Architecture)
from llm.agents.news_analyst import (
    NewsAnalysis,
    NewsAnalyst,
    NewsAnalystResult,
    NewsEventType,
    NewsImpactLevel,
    NewsTimeRelevance,
    create_news_analyst,
    create_safe_news_analyst,
)

# Observability Agent (UPGRADE-014 Category 2: Observability & Debugging)
from llm.agents.observability_agent import (
    AgentMetricsSnapshot,
    Alert,
    AlertSeverity,
    AlertType,
    MonitoringThresholds,
    ObservabilityAgent,
    create_observability_agent,
)

# Agent Registry (UPGRADE-014 Category 1: Architecture Enhancements)
from llm.agents.registry import (
    ORCHESTRATION_CAPABILITY,
    RISK_ASSESSMENT_CAPABILITY,
    SENTIMENT_ANALYSIS_CAPABILITY,
    TECHNICAL_ANALYSIS_CAPABILITY,
    TRADING_EXECUTION_CAPABILITY,
    AgentCapability,
    AgentHealth,
    AgentRegistration,
    AgentRegistry,
    CapabilityCategory,
    create_registry,
    get_global_registry,
    register_agent,
)
from llm.agents.risk_managers import (
    PositionRiskManager,
    create_position_risk_manager,
    create_safe_position_risk_manager,
)

# Safe Agent Wrapper with Circuit Breaker (Phase 1 - December 2025)
from llm.agents.safe_agent_wrapper import (
    AuditRecord,
    RiskTier,
    RiskTierConfig,
    SafeAgentWrapper,
    SafetyCheckResult,
    wrap_agent_with_safety,
)
from llm.agents.sentiment_analyst import (
    SentimentAnalyst,
    create_safe_sentiment_analyst,
    create_sentiment_analyst,
)
from llm.agents.supervisor import (
    DebateTriggerReason,
    SupervisorAgent,
    create_safe_supervisor_agent,
    create_supervisor_agent,
    create_supervisor_with_debate,
)
from llm.agents.technical_analyst import (
    TechnicalAnalyst,
    create_safe_technical_analyst,
    create_technical_analyst,
)
from llm.agents.traders import (
    ConservativeTrader,
    create_conservative_trader,
    create_safe_conservative_trader,
)


__all__ = [
    # Base classes
    "TradingAgent",
    "AgentRole",
    "AgentResponse",
    "AgentThought",
    "ThoughtType",
    # ReAct loop enhancements (UPGRADE-014 Category 1)
    "TerminationReason",
    "LoopMetrics",
    # Supervisor
    "SupervisorAgent",
    "DebateTriggerReason",
    "create_supervisor_agent",
    "create_supervisor_with_debate",
    "create_safe_supervisor_agent",
    # Analysts
    "TechnicalAnalyst",
    "create_technical_analyst",
    "create_safe_technical_analyst",
    "SentimentAnalyst",
    "create_sentiment_analyst",
    "create_safe_sentiment_analyst",
    # Traders
    "ConservativeTrader",
    "create_conservative_trader",
    "create_safe_conservative_trader",
    # Risk Managers
    "PositionRiskManager",
    "create_position_risk_manager",
    "create_safe_position_risk_manager",
    # Safe Agent Wrapper with Circuit Breaker (Phase 1 - December 2025)
    "RiskTier",
    "RiskTierConfig",
    "SafetyCheckResult",
    "AuditRecord",
    "SafeAgentWrapper",
    "wrap_agent_with_safety",
    # Bull/Bear Debate Mechanism (UPGRADE-004 - December 2025)
    "BullBearDebate",
    "DebateConfig",
    "DebateResult",
    "DebateRound",
    "DebateOutcome",
    "DebateTrigger",
    "create_debate_mechanism",
    "generate_debate_report",
    "BullResearcher",
    "create_bull_researcher",
    "BearResearcher",
    "create_bear_researcher",
    # News Analyst (UPGRADE-014 Feature 6: Multi-Agent Architecture)
    "NewsAnalyst",
    "NewsEventType",
    "NewsImpactLevel",
    "NewsTimeRelevance",
    "NewsAnalysis",
    "NewsAnalystResult",
    "create_news_analyst",
    "create_safe_news_analyst",
    # Multi-Agent Consensus (UPGRADE-014 Feature 6: Multi-Agent Architecture)
    "MultiAgentConsensus",
    "ConsensusSignal",
    "ConsensusConfig",
    "ConsensusResult",
    "AgentOpinion",
    "AgentType",
    "create_multi_agent_consensus",
    "opinion_from_agent_response",
    # Agent Registry (UPGRADE-014 Category 1: Architecture Enhancements)
    "AgentRegistry",
    "AgentRegistration",
    "AgentCapability",
    "AgentHealth",
    "CapabilityCategory",
    "get_global_registry",
    "register_agent",
    "create_registry",
    "TECHNICAL_ANALYSIS_CAPABILITY",
    "SENTIMENT_ANALYSIS_CAPABILITY",
    "TRADING_EXECUTION_CAPABILITY",
    "RISK_ASSESSMENT_CAPABILITY",
    "ORCHESTRATION_CAPABILITY",
    # Observability Agent (UPGRADE-014 Category 2: Observability & Debugging)
    "ObservabilityAgent",
    "Alert",
    "AlertSeverity",
    "AlertType",
    "MonitoringThresholds",
    "AgentMetricsSnapshot",
    "create_observability_agent",
]
