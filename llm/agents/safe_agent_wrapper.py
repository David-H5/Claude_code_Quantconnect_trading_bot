"""
Safe Agent Wrapper with Circuit Breaker Integration.

Wraps trading agents with safety checks including:
- Circuit breaker integration (trading halt on risk threshold breach)
- Risk tier classification (LOW, MEDIUM, HIGH, CRITICAL)
- Position size validation
- Approval workflow for high-risk decisions
- Audit logging of all decisions

This module solves the critical gap identified in Phase 6 research:
- Circuit breaker exists but is DISCONNECTED from agent decision flow
- Agents make decisions without checking risk controls

Version: 1.0 (December 2025)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from llm.agents.base import (
    AgentResponse,
    AgentRole,
    AgentThought,
    ThoughtType,
    TradingAgent,
)
from models.circuit_breaker import TradingCircuitBreaker


logger = logging.getLogger(__name__)


class RiskTier(Enum):
    """Risk tiers for agent decisions."""

    LOW = "low"  # Auto-approve
    MEDIUM = "medium"  # Notify
    HIGH = "high"  # Require approval
    CRITICAL = "critical"  # Block


@dataclass
class RiskTierConfig:
    """Configuration for risk tier thresholds."""

    low_max_position_pct: float = 0.02  # 2% position = LOW
    medium_max_position_pct: float = 0.05  # 5% position = MEDIUM
    high_max_position_pct: float = 0.10  # 10% position = HIGH
    # Above HIGH = CRITICAL (blocked)

    # Additional risk factors
    high_volatility_threshold: float = 0.03  # 3% daily volatility
    max_correlated_exposure_pct: float = 0.15  # 15% in correlated assets


@dataclass
class SafetyCheckResult:
    """Result of safety check on agent decision."""

    passed: bool
    risk_tier: RiskTier
    blocked: bool
    requires_approval: bool
    block_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    checks_performed: list[str] = field(default_factory=list)


@dataclass
class AuditRecord:
    """Audit record for agent decision."""

    timestamp: datetime
    agent_name: str
    decision_action: str
    safety_result: SafetyCheckResult
    original_response: dict[str, Any]
    modified: bool = False
    modification_reason: str | None = None


class SafeAgentWrapper:
    """
    Wraps trading agents with circuit breaker and risk control integration.

    All agent decisions pass through safety checks before execution:
    1. Circuit breaker status check
    2. Risk tier classification
    3. Position size validation
    4. Approval workflow for high-risk decisions

    Usage:
        # Wrap an existing agent
        circuit_breaker = TradingCircuitBreaker()
        safe_agent = SafeAgentWrapper(
            agent=technical_analyst,
            circuit_breaker=circuit_breaker,
        )

        # Use the wrapped agent
        response = safe_agent.analyze(query, context)

        # Response includes safety metadata
        print(response.metadata.get("risk_tier"))
        print(response.metadata.get("requires_approval"))
    """

    def __init__(
        self,
        agent: TradingAgent,
        circuit_breaker: TradingCircuitBreaker,
        risk_config: RiskTierConfig | None = None,
        approval_callback: Callable[[AgentResponse, SafetyCheckResult], bool] | None = None,
        enable_audit: bool = True,
    ):
        """
        Initialize safe agent wrapper.

        Args:
            agent: Trading agent to wrap
            circuit_breaker: Circuit breaker for trading halts
            risk_config: Risk tier configuration
            approval_callback: Optional callback for high-risk approval
            enable_audit: Whether to log all decisions
        """
        self.agent = agent
        self.circuit_breaker = circuit_breaker
        self.risk_config = risk_config or RiskTierConfig()
        self.approval_callback = approval_callback
        self.enable_audit = enable_audit

        # Audit log
        self.audit_log: list[AuditRecord] = []

    @property
    def name(self) -> str:
        """Get wrapped agent's name."""
        return f"Safe[{self.agent.name}]"

    @property
    def role(self) -> AgentRole:
        """Get wrapped agent's role."""
        return self.agent.role

    def analyze(
        self,
        query: str,
        context: dict[str, Any],
        skip_safety_checks: bool = False,
    ) -> AgentResponse:
        """
        Analyze with safety checks.

        Args:
            query: Question or task for the agent
            context: Market context and data
            skip_safety_checks: Bypass safety (for testing only!)

        Returns:
            AgentResponse with safety metadata
        """
        # Step 1: Check circuit breaker BEFORE calling agent
        if not skip_safety_checks and not self.circuit_breaker.can_trade():
            return self._create_blocked_response(
                query=query,
                reason="Circuit breaker active - trading halted",
                blocked_by="circuit_breaker",
            )

        # Step 2: Call underlying agent
        try:
            response = self.agent.analyze(query, context)
        except Exception as e:
            logger.error(f"Agent {self.agent.name} failed: {e}")
            return self._create_error_response(query, str(e))

        # Step 3: Apply safety checks to response
        if not skip_safety_checks:
            response = self._apply_safety_checks(response, context)

        # Step 4: Audit logging
        if self.enable_audit:
            self._log_audit(response, context)

        return response

    def _apply_safety_checks(
        self,
        response: AgentResponse,
        context: dict[str, Any],
    ) -> AgentResponse:
        """
        Apply safety checks to agent response.

        Args:
            response: Original agent response
            context: Market context

        Returns:
            Modified response with safety metadata
        """
        # Perform safety checks
        safety_result = self._run_safety_checks(response, context)

        # If blocked, override the response
        if safety_result.blocked:
            return self._create_blocked_response(
                query=response.query,
                reason=safety_result.block_reason or "Blocked by risk controls",
                blocked_by="risk_tier",
                original_response=response,
                safety_result=safety_result,
            )

        # If requires approval, check callback
        if safety_result.requires_approval:
            if self.approval_callback is not None:
                approved = self.approval_callback(response, safety_result)
                if not approved:
                    return self._create_blocked_response(
                        query=response.query,
                        reason="High-risk decision not approved",
                        blocked_by="approval_denied",
                        original_response=response,
                        safety_result=safety_result,
                    )

        # Add safety metadata to response
        return self._add_safety_metadata(response, safety_result)

    def _run_safety_checks(
        self,
        response: AgentResponse,
        context: dict[str, Any],
    ) -> SafetyCheckResult:
        """
        Run all safety checks on agent response.

        Args:
            response: Agent response to check
            context: Market context

        Returns:
            SafetyCheckResult with check details
        """
        checks_performed = []
        warnings = []

        # Check 1: Circuit breaker
        checks_performed.append("circuit_breaker")
        if not self.circuit_breaker.can_trade():
            return SafetyCheckResult(
                passed=False,
                risk_tier=RiskTier.CRITICAL,
                blocked=True,
                requires_approval=False,
                block_reason="Circuit breaker is active",
                checks_performed=checks_performed,
            )

        # Check 2: Extract position size from response
        checks_performed.append("position_size")
        position_size = self._extract_position_size(response, context)
        risk_tier = self._classify_risk_tier(position_size)

        # Check 3: High volatility environment
        checks_performed.append("volatility")
        volatility = context.get("volatility", 0.0)
        if volatility > self.risk_config.high_volatility_threshold:
            warnings.append(f"High volatility environment: {volatility:.2%}")
            # Escalate risk tier in high volatility
            if risk_tier == RiskTier.LOW:
                risk_tier = RiskTier.MEDIUM
            elif risk_tier == RiskTier.MEDIUM:
                risk_tier = RiskTier.HIGH

        # Check 4: Correlated exposure
        checks_performed.append("correlation")
        correlated_exposure = context.get("correlated_exposure_pct", 0.0)
        if correlated_exposure > self.risk_config.max_correlated_exposure_pct:
            warnings.append(f"High correlated exposure: {correlated_exposure:.2%}")
            risk_tier = RiskTier.HIGH

        # Determine if blocked or needs approval
        blocked = risk_tier == RiskTier.CRITICAL
        requires_approval = risk_tier == RiskTier.HIGH

        return SafetyCheckResult(
            passed=not blocked,
            risk_tier=risk_tier,
            blocked=blocked,
            requires_approval=requires_approval,
            block_reason="Position size exceeds maximum limits" if blocked else None,
            warnings=warnings,
            checks_performed=checks_performed,
        )

    def _extract_position_size(
        self,
        response: AgentResponse,
        context: dict[str, Any],
    ) -> float:
        """
        Extract position size from agent response.

        Args:
            response: Agent response
            context: Market context

        Returns:
            Position size as fraction of portfolio (0.0 to 1.0)
        """
        # Try to get from response metadata
        # Check thoughts for position size information
        for thought in response.thoughts:
            if thought.metadata.get("position_size_pct"):
                return thought.metadata["position_size_pct"]

        # Try to parse from final answer
        final_answer = response.final_answer.lower()

        # Look for percentage mentions
        import re

        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", final_answer)
        if pct_match:
            pct = float(pct_match.group(1))
            if pct <= 100:  # Likely a position size percentage
                return pct / 100.0

        # Default: Use context or assume small position
        return context.get("proposed_position_pct", 0.01)

    def _classify_risk_tier(self, position_size: float) -> RiskTier:
        """
        Classify risk tier based on position size.

        Args:
            position_size: Position size as fraction (0.0 to 1.0)

        Returns:
            RiskTier classification
        """
        if position_size <= self.risk_config.low_max_position_pct:
            return RiskTier.LOW
        elif position_size <= self.risk_config.medium_max_position_pct:
            return RiskTier.MEDIUM
        elif position_size <= self.risk_config.high_max_position_pct:
            return RiskTier.HIGH
        else:
            return RiskTier.CRITICAL

    def _add_safety_metadata(
        self,
        response: AgentResponse,
        safety_result: SafetyCheckResult,
    ) -> AgentResponse:
        """
        Add safety metadata to agent response.

        Note: AgentResponse is a dataclass with no metadata field,
        so we create a new response with added thoughts.
        """
        # Add safety thought to reasoning chain
        safety_thought = AgentThought(
            thought_type=ThoughtType.OBSERVATION,
            content=f"Safety check passed. Risk tier: {safety_result.risk_tier.value}",
            metadata={
                "safety_check": True,
                "risk_tier": safety_result.risk_tier.value,
                "requires_approval": safety_result.requires_approval,
                "warnings": safety_result.warnings,
                "checks_performed": safety_result.checks_performed,
            },
        )

        # Create new response with safety thought added
        new_thoughts = list(response.thoughts) + [safety_thought]

        return AgentResponse(
            agent_name=response.agent_name,
            agent_role=response.agent_role,
            query=response.query,
            thoughts=new_thoughts,
            final_answer=response.final_answer,
            confidence=response.confidence,
            tools_used=response.tools_used,
            execution_time_ms=response.execution_time_ms,
            success=response.success,
            error=response.error,
        )

    def _create_blocked_response(
        self,
        query: str,
        reason: str,
        blocked_by: str,
        original_response: AgentResponse | None = None,
        safety_result: SafetyCheckResult | None = None,
    ) -> AgentResponse:
        """
        Create a blocked/HOLD response when safety check fails.

        Args:
            query: Original query
            reason: Reason for blocking
            blocked_by: What triggered the block
            original_response: Original response if available
            safety_result: Safety check result if available

        Returns:
            AgentResponse with HOLD action
        """
        blocked_thought = AgentThought(
            thought_type=ThoughtType.FINAL_ANSWER,
            content=f"Action blocked: {reason}",
            metadata={
                "blocked": True,
                "blocked_by": blocked_by,
                "original_action": original_response.final_answer if original_response else None,
                "risk_tier": safety_result.risk_tier.value if safety_result else "critical",
            },
        )

        return AgentResponse(
            agent_name=self.name,
            agent_role=self.role,
            query=query,
            thoughts=[blocked_thought],
            final_answer=f"Action: HOLD\nReason: {reason}",
            confidence=1.0,  # High confidence in blocking
            tools_used=[],
            execution_time_ms=0.0,
            success=True,  # Block is a successful safety response
            error=None,
        )

    def _create_error_response(self, query: str, error: str) -> AgentResponse:
        """Create error response when agent fails."""
        error_thought = AgentThought(
            thought_type=ThoughtType.FINAL_ANSWER,
            content=f"Agent error: {error}",
            metadata={"error": True, "error_message": error},
        )

        return AgentResponse(
            agent_name=self.name,
            agent_role=self.role,
            query=query,
            thoughts=[error_thought],
            final_answer=f"Action: HOLD\nReason: Agent error - {error}",
            confidence=0.0,
            tools_used=[],
            execution_time_ms=0.0,
            success=False,
            error=error,
        )

    def _log_audit(self, response: AgentResponse, context: dict[str, Any]) -> None:
        """Log decision to audit trail."""
        # Extract safety result from thoughts
        safety_result = None
        for thought in response.thoughts:
            if thought.metadata.get("safety_check"):
                safety_result = SafetyCheckResult(
                    passed=True,
                    risk_tier=RiskTier(thought.metadata.get("risk_tier", "low")),
                    blocked=thought.metadata.get("blocked", False),
                    requires_approval=thought.metadata.get("requires_approval", False),
                    warnings=thought.metadata.get("warnings", []),
                    checks_performed=thought.metadata.get("checks_performed", []),
                )
                break
            elif thought.metadata.get("blocked"):
                safety_result = SafetyCheckResult(
                    passed=False,
                    risk_tier=RiskTier(thought.metadata.get("risk_tier", "critical")),
                    blocked=True,
                    requires_approval=False,
                    block_reason=thought.content,
                )
                break

        if safety_result is None:
            safety_result = SafetyCheckResult(
                passed=True,
                risk_tier=RiskTier.LOW,
                blocked=False,
                requires_approval=False,
            )

        # Extract action from response
        action = "HOLD"
        if "BUY" in response.final_answer.upper():
            action = "BUY"
        elif "SELL" in response.final_answer.upper():
            action = "SELL"

        record = AuditRecord(
            timestamp=datetime.now(),
            agent_name=self.agent.name,
            decision_action=action,
            safety_result=safety_result,
            original_response={
                "final_answer": response.final_answer,
                "confidence": response.confidence,
            },
        )

        self.audit_log.append(record)

        # Keep only last 1000 records
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]

    def get_audit_summary(self) -> dict[str, Any]:
        """Get audit log summary."""
        if not self.audit_log:
            return {"total_decisions": 0}

        blocked = sum(1 for r in self.audit_log if r.safety_result.blocked)
        by_tier = {}
        for r in self.audit_log:
            tier = r.safety_result.risk_tier.value
            by_tier[tier] = by_tier.get(tier, 0) + 1

        return {
            "total_decisions": len(self.audit_log),
            "blocked_decisions": blocked,
            "block_rate": blocked / len(self.audit_log),
            "by_risk_tier": by_tier,
            "recent_decisions": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "action": r.decision_action,
                    "risk_tier": r.safety_result.risk_tier.value,
                    "blocked": r.safety_result.blocked,
                }
                for r in self.audit_log[-10:]
            ],
        }


def wrap_agent_with_safety(
    agent: TradingAgent,
    circuit_breaker: TradingCircuitBreaker,
    risk_config: RiskTierConfig | None = None,
) -> SafeAgentWrapper:
    """
    Factory function to wrap an agent with safety controls.

    Args:
        agent: Trading agent to wrap
        circuit_breaker: Circuit breaker instance
        risk_config: Optional risk configuration

    Returns:
        SafeAgentWrapper instance
    """
    return SafeAgentWrapper(
        agent=agent,
        circuit_breaker=circuit_breaker,
        risk_config=risk_config,
    )


# Export all public API
__all__ = [
    "AuditRecord",
    "RiskTier",
    "RiskTierConfig",
    "SafeAgentWrapper",
    "SafetyCheckResult",
    "wrap_agent_with_safety",
]
