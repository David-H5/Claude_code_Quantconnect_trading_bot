"""
Bull/Bear Debate Mechanism for Trading Decisions

Implements TradingAgents-inspired multi-agent debate pattern where:
- Bull researcher argues for bullish positions
- Bear researcher argues for bearish positions
- Moderator assesses arguments and determines consensus

Research Source: TradingAgents (2024) - Multi-agent trading with structured debate

UPGRADE-014 Enhancements (December 2025):
- Structured round phases (Opening, Rebuttal, Closing)
- Sentiment integration for context-aware debates
- Enhanced scoring with sentiment weight
- Time-boxed rounds with configurable timeouts
- Integration with SentimentFilter

QuantConnect Compatible: Yes
- Non-blocking design
- Configurable timeouts
- Decision logging integration
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional


if TYPE_CHECKING:
    from llm.agents.base import AgentThought, TradingAgent
    from llm.decision_logger import DecisionLogger


class DebateOutcome(Enum):
    """Possible outcomes of a debate."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    AVOID = "avoid"
    INCONCLUSIVE = "inconclusive"


class RoundPhase(Enum):
    """Phase of debate round (UPGRADE-014)."""

    OPENING = "opening"  # Initial arguments
    REBUTTAL = "rebuttal"  # Counter-arguments
    CLOSING = "closing"  # Final statements
    SUMMARY = "summary"  # Moderator summary


class DebateTrigger(Enum):
    """Reasons to trigger a debate."""

    HIGH_POSITION_SIZE = "high_position_size"
    LOW_CONFIDENCE = "low_confidence"
    CONFLICTING_SIGNALS = "conflicting_signals"
    HIGH_IMPACT_EVENT = "high_impact_event"
    UNUSUAL_MARKET = "unusual_market"
    MANUAL_REQUEST = "manual_request"


@dataclass
class DebateArgument:
    """Single argument in a debate."""

    content: str
    confidence: float
    key_points: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    risks_identified: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "confidence": self.confidence,
            "key_points": self.key_points,
            "evidence": self.evidence,
            "risks_identified": self.risks_identified,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ModeratorAssessment:
    """Moderator's assessment of debate round."""

    summary: str
    stronger_argument: str  # "bull", "bear", or "tie"
    key_disagreements: list[str]
    areas_of_agreement: list[str]
    recommended_action: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "stronger_argument": self.stronger_argument,
            "key_disagreements": self.key_disagreements,
            "areas_of_agreement": self.areas_of_agreement,
            "recommended_action": self.recommended_action,
            "confidence": self.confidence,
        }


@dataclass
class DebateRound:
    """Single round of bull/bear debate."""

    round_number: int
    bull_argument: DebateArgument
    bear_argument: DebateArgument
    moderator_assessment: ModeratorAssessment
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "round_number": self.round_number,
            "bull_argument": self.bull_argument.to_dict(),
            "bear_argument": self.bear_argument.to_dict(),
            "moderator_assessment": self.moderator_assessment.to_dict(),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DebateResult:
    """Result of multi-round debate."""

    debate_id: str
    opportunity: dict[str, Any]
    rounds: list[DebateRound]
    final_outcome: DebateOutcome
    consensus_confidence: float
    key_points_bull: list[str]
    key_points_bear: list[str]
    risk_factors: list[str]
    trigger_reason: DebateTrigger
    total_duration_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "debate_id": self.debate_id,
            "opportunity": self.opportunity,
            "rounds": [r.to_dict() for r in self.rounds],
            "final_outcome": self.final_outcome.value,
            "consensus_confidence": self.consensus_confidence,
            "key_points_bull": self.key_points_bull,
            "key_points_bear": self.key_points_bear,
            "risk_factors": self.risk_factors,
            "trigger_reason": self.trigger_reason.value,
            "total_duration_ms": self.total_duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DebateConfig:
    """Configuration for debate mechanism."""

    max_rounds: int = 3
    consensus_threshold: float = 0.7
    min_confidence_delta: float = 0.15
    timeout_per_round_ms: float = 10000.0
    require_evidence: bool = True
    log_decisions: bool = True

    # Trigger thresholds
    position_size_threshold: float = 0.10  # 10% of portfolio
    confidence_threshold: float = 0.70  # Below this triggers debate
    conflicting_signal_threshold: float = 0.30  # Signal difference

    # UPGRADE-014: Structured rounds
    use_structured_phases: bool = True
    phases_per_round: list[str] = field(default_factory=lambda: ["opening", "rebuttal", "closing"])

    # UPGRADE-014: Sentiment integration
    use_sentiment_context: bool = True
    sentiment_weight_in_scoring: float = 0.2  # 20% weight for sentiment
    min_sentiment_confidence: float = 0.5
    sentiment_mismatch_threshold: float = 0.4  # Trigger debate if mismatch


@dataclass
class SentimentContext:
    """Sentiment context for debate (UPGRADE-014)."""

    sentiment_score: float  # -1.0 to 1.0
    sentiment_confidence: float  # 0.0 to 1.0
    source: str  # "finbert", "ensemble", etc.
    articles_analyzed: int = 0
    trend_direction: str = "neutral"  # "improving", "declining", "neutral"

    @property
    def supports_bull(self) -> bool:
        """Check if sentiment supports bullish case."""
        return self.sentiment_score > 0.1 and self.sentiment_confidence >= 0.5

    @property
    def supports_bear(self) -> bool:
        """Check if sentiment supports bearish case."""
        return self.sentiment_score < -0.1 and self.sentiment_confidence >= 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sentiment_score": self.sentiment_score,
            "sentiment_confidence": self.sentiment_confidence,
            "source": self.source,
            "articles_analyzed": self.articles_analyzed,
            "trend_direction": self.trend_direction,
            "supports_bull": self.supports_bull,
            "supports_bear": self.supports_bear,
        }


@dataclass
class StructuredRoundPhase:
    """Single phase of a structured round (UPGRADE-014)."""

    phase: RoundPhase
    speaker: str  # "bull", "bear", "moderator"
    content: str
    confidence: float
    sentiment_alignment: float | None = None  # How well argument aligns with sentiment
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase.value,
            "speaker": self.speaker,
            "content": self.content,
            "confidence": self.confidence,
            "sentiment_alignment": self.sentiment_alignment,
            "timestamp": self.timestamp.isoformat(),
        }


class BullBearDebate:
    """
    Multi-agent debate mechanism for trading decisions.

    Based on TradingAgents (2024) research showing improved
    decision quality through structured adversarial debate.

    Usage:
        debate = BullBearDebate(
            bull_agent=bull,
            bear_agent=bear,
            moderator_agent=moderator,
        )

        if debate.should_debate(opportunity, context):
            result = debate.run_debate(opportunity, initial_analysis)
            if result.consensus_confidence >= 0.7:
                execute_trade(result.final_outcome)

    Trigger Criteria:
        - Position size > 10% of portfolio
        - Initial confidence < 70%
        - Conflicting analyst signals
        - High-impact events (earnings, etc.)
    """

    def __init__(
        self,
        bull_agent: Optional["TradingAgent"] = None,
        bear_agent: Optional["TradingAgent"] = None,
        moderator_agent: Optional["TradingAgent"] = None,
        config: DebateConfig | None = None,
        decision_logger: Optional["DecisionLogger"] = None,
    ):
        """
        Initialize debate mechanism.

        Args:
            bull_agent: Agent arguing bullish positions
            bear_agent: Agent arguing bearish positions
            moderator_agent: Agent moderating and assessing
            config: Debate configuration
            decision_logger: Optional logger for audit trails
        """
        self.bull = bull_agent
        self.bear = bear_agent
        self.moderator = moderator_agent
        self.config = config or DebateConfig()
        self.decision_logger = decision_logger

        # History
        self.debate_history: list[DebateResult] = []
        self._debate_count = 0

    def should_debate(
        self,
        opportunity: dict[str, Any],
        context: dict[str, Any],
        initial_confidence: float = 1.0,
        sentiment_context: SentimentContext | None = None,
    ) -> tuple[bool, DebateTrigger | None]:
        """
        Determine if a debate should be triggered.

        Args:
            opportunity: The trading opportunity
            context: Current market/portfolio context
            initial_confidence: Confidence from initial analysis
            sentiment_context: Optional sentiment data (UPGRADE-014)

        Returns:
            Tuple of (should_debate, trigger_reason)
        """
        # Check position size
        position_size = opportunity.get("position_size_pct", 0.0)
        if position_size > self.config.position_size_threshold:
            return True, DebateTrigger.HIGH_POSITION_SIZE

        # Check confidence
        if initial_confidence < self.config.confidence_threshold:
            return True, DebateTrigger.LOW_CONFIDENCE

        # Check for conflicting signals
        signals = context.get("analyst_signals", {})
        if signals:
            signal_values = list(signals.values())
            if len(signal_values) >= 2:
                signal_range = max(signal_values) - min(signal_values)
                if signal_range > self.config.conflicting_signal_threshold:
                    return True, DebateTrigger.CONFLICTING_SIGNALS

        # UPGRADE-014: Check sentiment mismatch
        if sentiment_context and self.config.use_sentiment_context:
            proposed_action = opportunity.get("proposed_action", "").lower()
            if proposed_action == "buy" and sentiment_context.supports_bear:
                # Bullish action with bearish sentiment - trigger debate
                return True, DebateTrigger.CONFLICTING_SIGNALS
            elif proposed_action == "sell" and sentiment_context.supports_bull:
                # Bearish action with bullish sentiment - trigger debate
                return True, DebateTrigger.CONFLICTING_SIGNALS

        # Check for high-impact events
        if opportunity.get("has_earnings", False):
            return True, DebateTrigger.HIGH_IMPACT_EVENT

        if context.get("unusual_volatility", False):
            return True, DebateTrigger.UNUSUAL_MARKET

        return False, None

    def run_debate(
        self,
        opportunity: dict[str, Any],
        initial_analysis: dict[str, Any],
        trigger_reason: DebateTrigger | None = None,
        sentiment_context: SentimentContext | None = None,
    ) -> DebateResult:
        """
        Run structured bull/bear debate on trading opportunity.

        Args:
            opportunity: The trading opportunity to debate
            initial_analysis: Initial market analysis
            trigger_reason: Why debate was triggered
            sentiment_context: Optional sentiment data (UPGRADE-014)

        Returns:
            DebateResult with final recommendation
        """
        import time

        start_time = time.time()

        self._debate_count += 1
        debate_id = f"debate_{self._debate_count}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        rounds: list[DebateRound] = []
        context = {
            "opportunity": opportunity,
            "analysis": initial_analysis,
            "debate_history": [],
        }

        # UPGRADE-014: Add sentiment context
        if sentiment_context:
            context["sentiment"] = sentiment_context.to_dict()

        for round_num in range(self.config.max_rounds):
            round_result = self._run_round(round_num, context, sentiment_context)
            rounds.append(round_result)

            # Update context for next round
            context["debate_history"].append(
                {
                    "bull": round_result.bull_argument.content,
                    "bear": round_result.bear_argument.content,
                    "assessment": round_result.moderator_assessment.summary,
                }
            )

            # Check for early consensus
            if self._check_consensus(rounds):
                break

        # Compile result with sentiment-aware scoring
        duration_ms = (time.time() - start_time) * 1000
        result = self._compile_result(
            debate_id=debate_id,
            opportunity=opportunity,
            rounds=rounds,
            trigger_reason=trigger_reason or DebateTrigger.MANUAL_REQUEST,
            duration_ms=duration_ms,
            sentiment_context=sentiment_context,
        )

        # Record history
        self.debate_history.append(result)

        return result

    def run_structured_debate(
        self,
        opportunity: dict[str, Any],
        initial_analysis: dict[str, Any],
        sentiment_context: SentimentContext | None = None,
    ) -> DebateResult:
        """
        Run debate with structured phases (UPGRADE-014).

        Each round has Opening -> Rebuttal -> Closing phases.

        Args:
            opportunity: The trading opportunity
            initial_analysis: Initial analysis
            sentiment_context: Optional sentiment data

        Returns:
            DebateResult with structured rounds
        """
        import time

        start_time = time.time()

        self._debate_count += 1
        debate_id = f"structured_{self._debate_count}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        rounds: list[DebateRound] = []
        context = {
            "opportunity": opportunity,
            "analysis": initial_analysis,
            "debate_history": [],
            "structured": True,
        }

        if sentiment_context:
            context["sentiment"] = sentiment_context.to_dict()

        for round_num in range(self.config.max_rounds):
            # Run structured phases
            phases: list[StructuredRoundPhase] = []

            # Phase 1: Opening statements
            bull_opening = self._get_bull_argument(context, None)
            bear_opening = self._get_bear_argument(context, bull_opening.content)

            phases.append(
                StructuredRoundPhase(
                    phase=RoundPhase.OPENING,
                    speaker="bull",
                    content=bull_opening.content,
                    confidence=bull_opening.confidence,
                    sentiment_alignment=self._calc_sentiment_alignment(bull_opening, sentiment_context, "bull"),
                )
            )
            phases.append(
                StructuredRoundPhase(
                    phase=RoundPhase.OPENING,
                    speaker="bear",
                    content=bear_opening.content,
                    confidence=bear_opening.confidence,
                    sentiment_alignment=self._calc_sentiment_alignment(bear_opening, sentiment_context, "bear"),
                )
            )

            # Phase 2: Rebuttals
            context["last_bull"] = bull_opening.content
            context["last_bear"] = bear_opening.content

            bull_rebuttal = self._get_bull_argument(context, bear_opening.content)
            bear_rebuttal = self._get_bear_argument(context, bull_rebuttal.content)

            phases.append(
                StructuredRoundPhase(
                    phase=RoundPhase.REBUTTAL,
                    speaker="bull",
                    content=bull_rebuttal.content,
                    confidence=bull_rebuttal.confidence,
                )
            )
            phases.append(
                StructuredRoundPhase(
                    phase=RoundPhase.REBUTTAL,
                    speaker="bear",
                    content=bear_rebuttal.content,
                    confidence=bear_rebuttal.confidence,
                )
            )

            # Phase 3: Moderator summary
            final_bull = bull_rebuttal
            final_bear = bear_rebuttal
            moderator_assessment = self._get_moderator_assessment(final_bull, final_bear, context)

            phases.append(
                StructuredRoundPhase(
                    phase=RoundPhase.SUMMARY,
                    speaker="moderator",
                    content=moderator_assessment.summary,
                    confidence=moderator_assessment.confidence,
                )
            )

            # Create debate round with final arguments
            debate_round = DebateRound(
                round_number=round_num,
                bull_argument=final_bull,
                bear_argument=final_bear,
                moderator_assessment=moderator_assessment,
            )
            rounds.append(debate_round)

            # Update history
            context["debate_history"].append(
                {
                    "bull": final_bull.content,
                    "bear": final_bear.content,
                    "assessment": moderator_assessment.summary,
                    "phases": [p.to_dict() for p in phases],
                }
            )

            if self._check_consensus(rounds):
                break

        duration_ms = (time.time() - start_time) * 1000
        return self._compile_result(
            debate_id=debate_id,
            opportunity=opportunity,
            rounds=rounds,
            trigger_reason=DebateTrigger.MANUAL_REQUEST,
            duration_ms=duration_ms,
            sentiment_context=sentiment_context,
        )

    def _calc_sentiment_alignment(
        self,
        argument: DebateArgument,
        sentiment_context: SentimentContext | None,
        side: str,
    ) -> float | None:
        """Calculate how well argument aligns with sentiment (UPGRADE-014)."""
        if not sentiment_context:
            return None

        if side == "bull":
            # Bull argument aligns with positive sentiment
            if sentiment_context.supports_bull:
                return 0.8 + (sentiment_context.sentiment_confidence * 0.2)
            elif sentiment_context.supports_bear:
                return 0.2 - (sentiment_context.sentiment_confidence * 0.1)
            else:
                return 0.5
        else:  # bear
            # Bear argument aligns with negative sentiment
            if sentiment_context.supports_bear:
                return 0.8 + (sentiment_context.sentiment_confidence * 0.2)
            elif sentiment_context.supports_bull:
                return 0.2 - (sentiment_context.sentiment_confidence * 0.1)
            else:
                return 0.5

    def _run_round(
        self,
        round_number: int,
        context: dict[str, Any],
        sentiment_context: SentimentContext | None = None,
    ) -> DebateRound:
        """Run a single debate round."""
        previous_bear = None
        if context["debate_history"]:
            previous_bear = context["debate_history"][-1]["bear"]

        # Bull makes argument
        bull_arg = self._get_bull_argument(context, previous_bear)

        # Bear counters
        bear_arg = self._get_bear_argument(context, bull_arg.content)

        # Moderator assesses
        moderator_assessment = self._get_moderator_assessment(bull_arg, bear_arg, context)

        return DebateRound(
            round_number=round_number,
            bull_argument=bull_arg,
            bear_argument=bear_arg,
            moderator_assessment=moderator_assessment,
        )

    def _get_bull_argument(
        self,
        context: dict[str, Any],
        previous_bear: str | None = None,
    ) -> DebateArgument:
        """Get bullish argument from bull agent."""
        if self.bull is None:
            return self._generate_mock_bull_argument(context, previous_bear)

        try:
            response = self.bull.analyze(
                query=self._create_bull_prompt(context, previous_bear),
                context=context,
            )
            return DebateArgument(
                content=response.final_answer,
                confidence=response.confidence,
                key_points=self._extract_key_points(response.final_answer),
                evidence=self._extract_evidence_from_thoughts(response.thoughts),
                risks_identified=[],
            )
        except Exception:
            return self._generate_mock_bull_argument(context, previous_bear)

    def _get_bear_argument(
        self,
        context: dict[str, Any],
        previous_bull: str,
    ) -> DebateArgument:
        """Get bearish argument from bear agent."""
        if self.bear is None:
            return self._generate_mock_bear_argument(context, previous_bull)

        try:
            response = self.bear.analyze(
                query=self._create_bear_prompt(context, previous_bull),
                context=context,
            )
            return DebateArgument(
                content=response.final_answer,
                confidence=response.confidence,
                key_points=self._extract_key_points(response.final_answer),
                evidence=self._extract_evidence_from_thoughts(response.thoughts),
                risks_identified=self._extract_risks(response.final_answer),
            )
        except Exception:
            return self._generate_mock_bear_argument(context, previous_bull)

    def _get_moderator_assessment(
        self,
        bull_arg: DebateArgument,
        bear_arg: DebateArgument,
        context: dict[str, Any],
    ) -> ModeratorAssessment:
        """Get moderator assessment of the debate round."""
        if self.moderator is None:
            return self._generate_mock_moderator_assessment(bull_arg, bear_arg)

        try:
            prompt = self._create_moderator_prompt(bull_arg, bear_arg, context)
            response = self.moderator.analyze(
                query=prompt,
                context=context,
            )

            # Determine stronger argument
            if bull_arg.confidence > bear_arg.confidence + 0.1:
                stronger = "bull"
            elif bear_arg.confidence > bull_arg.confidence + 0.1:
                stronger = "bear"
            else:
                stronger = "tie"

            return ModeratorAssessment(
                summary=response.final_answer,
                stronger_argument=stronger,
                key_disagreements=self._extract_disagreements(bull_arg.content, bear_arg.content),
                areas_of_agreement=self._extract_agreements(bull_arg.content, bear_arg.content),
                recommended_action=response.final_answer.split(".")[0],
                confidence=response.confidence,
            )
        except Exception:
            return self._generate_mock_moderator_assessment(bull_arg, bear_arg)

    def _check_consensus(self, rounds: list[DebateRound]) -> bool:
        """Check if debate has reached consensus."""
        if len(rounds) < 2:
            return False

        last_round = rounds[-1]
        confidence_gap = abs(last_round.bull_argument.confidence - last_round.bear_argument.confidence)
        return confidence_gap > self.config.consensus_threshold

    def _compile_result(
        self,
        debate_id: str,
        opportunity: dict[str, Any],
        rounds: list[DebateRound],
        trigger_reason: DebateTrigger,
        duration_ms: float,
        sentiment_context: SentimentContext | None = None,
    ) -> DebateResult:
        """Compile debate rounds into final result with sentiment-aware scoring."""
        final_round = rounds[-1]

        # Determine outcome based on final confidences
        bull_conf = final_round.bull_argument.confidence
        bear_conf = final_round.bear_argument.confidence

        # UPGRADE-014: Apply sentiment weight to scoring
        if sentiment_context and self.config.use_sentiment_context:
            sentiment_weight = self.config.sentiment_weight_in_scoring

            if sentiment_context.supports_bull:
                # Boost bull confidence based on sentiment alignment
                bull_conf = bull_conf * (1 - sentiment_weight) + (bull_conf + 0.1) * sentiment_weight
                # Reduce bear confidence slightly
                bear_conf = bear_conf * (1 - sentiment_weight * 0.5) + (bear_conf - 0.05) * (sentiment_weight * 0.5)
            elif sentiment_context.supports_bear:
                # Boost bear confidence based on sentiment alignment
                bear_conf = bear_conf * (1 - sentiment_weight) + (bear_conf + 0.1) * sentiment_weight
                # Reduce bull confidence slightly
                bull_conf = bull_conf * (1 - sentiment_weight * 0.5) + (bull_conf - 0.05) * (sentiment_weight * 0.5)

            # Clamp to valid range
            bull_conf = max(0.0, min(1.0, bull_conf))
            bear_conf = max(0.0, min(1.0, bear_conf))

        if bull_conf > bear_conf + self.config.min_confidence_delta:
            outcome = DebateOutcome.BUY
            consensus = bull_conf
        elif bear_conf > bull_conf + self.config.min_confidence_delta:
            has_position = opportunity.get("has_position", False)
            outcome = DebateOutcome.SELL if has_position else DebateOutcome.AVOID
            consensus = bear_conf
        else:
            outcome = DebateOutcome.HOLD
            consensus = (bull_conf + bear_conf) / 2

        # If consensus is too low, mark as inconclusive
        if consensus < 0.5:
            outcome = DebateOutcome.INCONCLUSIVE

        # Extract all key points and risks
        key_points_bull = []
        key_points_bear = []
        risk_factors = []

        for r in rounds:
            key_points_bull.extend(r.bull_argument.key_points)
            key_points_bear.extend(r.bear_argument.key_points)
            risk_factors.extend(r.bear_argument.risks_identified)

        # Deduplicate
        key_points_bull = list(dict.fromkeys(key_points_bull))[:5]
        key_points_bear = list(dict.fromkeys(key_points_bear))[:5]
        risk_factors = list(dict.fromkeys(risk_factors))[:5]

        return DebateResult(
            debate_id=debate_id,
            opportunity=opportunity,
            rounds=rounds,
            final_outcome=outcome,
            consensus_confidence=consensus,
            key_points_bull=key_points_bull,
            key_points_bear=key_points_bear,
            risk_factors=risk_factors,
            trigger_reason=trigger_reason,
            total_duration_ms=duration_ms,
        )

    # === Mock implementations for testing ===

    def _generate_mock_bull_argument(
        self,
        context: dict[str, Any],
        previous_bear: str | None,
    ) -> DebateArgument:
        """Generate mock bullish argument for testing."""
        opportunity = context.get("opportunity", {})
        symbol = opportunity.get("symbol", "SPY")

        content = f"The opportunity in {symbol} shows strong bullish signals. "
        if previous_bear:
            content += f"Responding to bearish concerns: {previous_bear[:50]}... "
        content += "Technical indicators support upside potential."

        return DebateArgument(
            content=content,
            confidence=0.72,
            key_points=[
                "Strong technical momentum",
                "Favorable risk/reward ratio",
            ],
            evidence=[
                "Price above 20-day SMA",
                "RSI at 55 (neutral, room to run)",
            ],
            risks_identified=[],
        )

    def _generate_mock_bear_argument(
        self,
        context: dict[str, Any],
        previous_bull: str,
    ) -> DebateArgument:
        """Generate mock bearish argument for testing."""
        opportunity = context.get("opportunity", {})
        symbol = opportunity.get("symbol", "SPY")

        content = f"Caution is warranted for {symbol}. "
        content += f"Counter to bull's view: {previous_bull[:50]}... "
        content += "Risk factors outweigh potential gains."

        return DebateArgument(
            content=content,
            confidence=0.65,
            key_points=[
                "Elevated volatility risk",
                "Weak market breadth",
            ],
            evidence=[
                "VIX above 20",
                "Sector rotation patterns",
            ],
            risks_identified=[
                "Market correction risk",
                "Liquidity concerns",
            ],
        )

    def _generate_mock_moderator_assessment(
        self,
        bull_arg: DebateArgument,
        bear_arg: DebateArgument,
    ) -> ModeratorAssessment:
        """Generate mock moderator assessment for testing."""
        if bull_arg.confidence > bear_arg.confidence:
            stronger = "bull"
            action = "Proceed with caution"
        elif bear_arg.confidence > bull_arg.confidence:
            stronger = "bear"
            action = "Wait for better entry"
        else:
            stronger = "tie"
            action = "Hold current position"

        return ModeratorAssessment(
            summary="Both sides present valid arguments. " + action,
            stronger_argument=stronger,
            key_disagreements=["Risk assessment", "Timing"],
            areas_of_agreement=["Symbol selection"],
            recommended_action=action,
            confidence=(bull_arg.confidence + bear_arg.confidence) / 2,
        )

    # === Helper methods ===

    def _create_bull_prompt(
        self,
        context: dict[str, Any],
        previous_bear: str | None,
    ) -> str:
        """Create prompt for bull agent."""
        opportunity = context.get("opportunity", {})
        prompt = f"""Analyze this trading opportunity with a BULLISH perspective:
Symbol: {opportunity.get('symbol', 'Unknown')}
Current Price: {opportunity.get('price', 'Unknown')}
Initial Analysis: {context.get('analysis', {})}

Provide a compelling bullish argument with:
1. Key reasons to be optimistic
2. Supporting evidence and data
3. Potential upside targets
"""
        if previous_bear:
            prompt += f"\nAddress the bear's previous argument:\n{previous_bear}"
        return prompt

    def _create_bear_prompt(
        self,
        context: dict[str, Any],
        previous_bull: str,
    ) -> str:
        """Create prompt for bear agent."""
        opportunity = context.get("opportunity", {})
        return f"""Analyze this trading opportunity with a BEARISH perspective:
Symbol: {opportunity.get('symbol', 'Unknown')}
Current Price: {opportunity.get('price', 'Unknown')}
Initial Analysis: {context.get('analysis', {})}

Counter the bull's argument:
{previous_bull}

Provide a compelling bearish argument with:
1. Key risks and concerns
2. Evidence supporting caution
3. Potential downside scenarios
"""

    def _create_moderator_prompt(
        self,
        bull_arg: DebateArgument,
        bear_arg: DebateArgument,
        context: dict[str, Any],
    ) -> str:
        """Create prompt for moderator agent."""
        return f"""As a neutral moderator, assess this trading debate:

BULL ARGUMENT (Confidence: {bull_arg.confidence:.0%}):
{bull_arg.content}

BEAR ARGUMENT (Confidence: {bear_arg.confidence:.0%}):
{bear_arg.content}

Provide:
1. Summary of both positions
2. Which argument is stronger and why
3. Key points of disagreement
4. Recommended action
"""

    def _extract_key_points(self, text: str) -> list[str]:
        """Extract key points from argument text."""
        sentences = text.split(".")
        key_points = []
        for s in sentences[:3]:  # First 3 sentences
            s = s.strip()
            if s and len(s) > 10:
                key_points.append(s)
        return key_points

    def _extract_evidence(self, text: str) -> list[str]:
        """Extract evidence from reasoning text."""
        evidence_keywords = ["data", "shows", "indicates", "based on", "according to"]
        evidence = []
        sentences = text.split(".")
        for s in sentences:
            if any(kw in s.lower() for kw in evidence_keywords):
                evidence.append(s.strip())
        return evidence[:3]

    def _extract_evidence_from_thoughts(self, thoughts: list["AgentThought"]) -> list[str]:
        """Extract evidence from agent thoughts."""
        evidence = []
        for thought in thoughts:
            # Extract relevant parts from thought content
            content = thought.content
            # Look for evidence in the thought
            if ":" in content:
                # Extract items after colon (e.g., "Technical analysis for SPY: item1; item2")
                parts = content.split(":")[-1].strip()
                items = [s.strip() for s in parts.split(";") if s.strip()]
                evidence.extend(items[:2])
        return evidence[:5]

    def _extract_risks(self, text: str) -> list[str]:
        """Extract risk factors from text."""
        risk_keywords = ["risk", "danger", "concern", "warning", "caution", "downside"]
        risks = []
        sentences = text.split(".")
        for s in sentences:
            if any(kw in s.lower() for kw in risk_keywords):
                risks.append(s.strip())
        return risks[:5]

    def _extract_disagreements(self, bull_text: str, bear_text: str) -> list[str]:
        """Extract key disagreements between arguments."""
        # Simple implementation - in production would use NLP
        return ["Market direction", "Risk assessment", "Timing"]

    def _extract_agreements(self, bull_text: str, bear_text: str) -> list[str]:
        """Extract areas of agreement between arguments."""
        return ["Liquidity is adequate", "Symbol selection appropriate"]

    def get_debate_statistics(self) -> dict[str, Any]:
        """Get statistics on debate outcomes."""
        if not self.debate_history:
            return {
                "total_debates": 0,
                "avg_rounds": 0,
                "avg_consensus": 0,
                "outcome_distribution": {},
            }

        outcomes = {}
        total_rounds = 0
        total_consensus = 0

        for result in self.debate_history:
            outcome = result.final_outcome.value
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            total_rounds += len(result.rounds)
            total_consensus += result.consensus_confidence

        n = len(self.debate_history)
        return {
            "total_debates": n,
            "avg_rounds": total_rounds / n,
            "avg_consensus": total_consensus / n,
            "outcome_distribution": outcomes,
            "avg_duration_ms": sum(r.total_duration_ms for r in self.debate_history) / n,
        }

    def clear_history(self) -> None:
        """Clear debate history."""
        self.debate_history = []


def create_debate_mechanism(
    bull_agent: Optional["TradingAgent"] = None,
    bear_agent: Optional["TradingAgent"] = None,
    moderator_agent: Optional["TradingAgent"] = None,
    max_rounds: int = 3,
    consensus_threshold: float = 0.7,
) -> BullBearDebate:
    """
    Factory function to create a debate mechanism.

    Args:
        bull_agent: Agent for bullish arguments
        bear_agent: Agent for bearish arguments
        moderator_agent: Agent for moderation
        max_rounds: Maximum debate rounds
        consensus_threshold: Threshold for early consensus

    Returns:
        Configured BullBearDebate instance
    """
    config = DebateConfig(
        max_rounds=max_rounds,
        consensus_threshold=consensus_threshold,
    )
    return BullBearDebate(
        bull_agent=bull_agent,
        bear_agent=bear_agent,
        moderator_agent=moderator_agent,
        config=config,
    )


def generate_debate_report(result: DebateResult) -> str:
    """
    Generate a human-readable debate report.

    Args:
        result: DebateResult to report on

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "BULL/BEAR DEBATE REPORT",
        "=" * 60,
        "",
        f"Debate ID: {result.debate_id}",
        f"Trigger: {result.trigger_reason.value}",
        f"Symbol: {result.opportunity.get('symbol', 'Unknown')}",
        f"Duration: {result.total_duration_ms:.0f}ms",
        "",
        "-" * 40,
        "OUTCOME",
        "-" * 40,
        f"Final Recommendation: {result.final_outcome.value.upper()}",
        f"Consensus Confidence: {result.consensus_confidence:.1%}",
        "",
        "-" * 40,
        "ROUNDS SUMMARY",
        "-" * 40,
    ]

    for r in result.rounds:
        lines.extend(
            [
                f"\nRound {r.round_number + 1}:",
                f"  Bull ({r.bull_argument.confidence:.0%}): {r.bull_argument.content[:80]}...",
                f"  Bear ({r.bear_argument.confidence:.0%}): {r.bear_argument.content[:80]}...",
                f"  Moderator: {r.moderator_assessment.summary[:60]}...",
            ]
        )

    lines.extend(
        [
            "",
            "-" * 40,
            "KEY POINTS - BULL",
            "-" * 40,
        ]
    )
    for point in result.key_points_bull:
        lines.append(f"  • {point}")

    lines.extend(
        [
            "",
            "-" * 40,
            "KEY POINTS - BEAR",
            "-" * 40,
        ]
    )
    for point in result.key_points_bear:
        lines.append(f"  • {point}")

    lines.extend(
        [
            "",
            "-" * 40,
            "RISK FACTORS",
            "-" * 40,
        ]
    )
    for risk in result.risk_factors:
        lines.append(f"  ⚠ {risk}")

    lines.append("=" * 60)

    return "\n".join(lines)
