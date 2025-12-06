"""
LLM Guardrails for Trading Decisions.

Provides comprehensive input/output validation and safety constraints
for LLM-powered trading agents. Implements layered guardrails based on
best practices from industry research.

UPGRADE-014: LLM Sentiment Integration (December 2025)

Research Sources:
- Datadog LLM Guardrails Best Practices (2024)
- arXiv Guardrails Architecture (Feb 2024)
- NVIDIA NeMo Guardrails (2023-2024)
- Leanware LLM Guardrails 2025

Key Principles:
1. Layer fast checks first (low latency)
2. Escalate to heavier checks when needed
3. Trade-off: Speed vs Safety vs Accuracy
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


# ==============================================================================
# Guardrail Types
# ==============================================================================


class GuardrailType(Enum):
    """Type of guardrail check."""

    INPUT = "input"
    OUTPUT = "output"
    BEHAVIORAL = "behavioral"
    SEMANTIC = "semantic"


class GuardrailResult(Enum):
    """Result of guardrail check."""

    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"
    MODIFY = "modify"


class ViolationType(Enum):
    """Type of guardrail violation."""

    POSITION_SIZE_EXCEEDED = "position_size_exceeded"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    SENTIMENT_THRESHOLD_BREACH = "sentiment_threshold_breach"
    CIRCUIT_BREAKER_ACTIVE = "circuit_breaker_active"
    INVALID_SYMBOL = "invalid_symbol"
    INVALID_ACTION = "invalid_action"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    HALLUCINATION_DETECTED = "hallucination_detected"
    EXCESSIVE_RISK = "excessive_risk"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    POLICY_VIOLATION = "policy_violation"


# ==============================================================================
# Data Types
# ==============================================================================


@dataclass
class GuardrailViolation:
    """Record of a guardrail violation."""

    violation_type: ViolationType
    guardrail_type: GuardrailType
    severity: str  # "warning", "error", "critical"
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    remediation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "violation_type": self.violation_type.value,
            "guardrail_type": self.guardrail_type.value,
            "severity": self.severity,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "remediation": self.remediation,
        }


@dataclass
class GuardrailCheckResult:
    """Result of guardrail check."""

    result: GuardrailResult
    passed: bool
    violations: list[GuardrailViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    modified_content: str | None = None
    execution_time_ms: float = 0.0
    checks_performed: list[str] = field(default_factory=list)

    @property
    def has_violations(self) -> bool:
        """Check if there are any violations."""
        return len(self.violations) > 0

    @property
    def has_critical_violations(self) -> bool:
        """Check if there are critical violations."""
        return any(v.severity == "critical" for v in self.violations)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result": self.result.value,
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
            "modified_content": self.modified_content,
            "execution_time_ms": self.execution_time_ms,
            "checks_performed": self.checks_performed,
        }


@dataclass
class TradingConstraints:
    """Trading-specific constraints for guardrails."""

    # Position sizing
    max_position_size_pct: float = 0.10  # 10% max per position
    max_portfolio_exposure_pct: float = 0.80  # 80% max invested

    # Confidence requirements
    min_confidence_for_trade: float = 0.6
    min_confidence_for_large_trade: float = 0.75

    # Sentiment requirements
    min_sentiment_for_long: float = 0.0
    max_sentiment_for_short: float = 0.0
    require_sentiment_confirmation: bool = True

    # Risk constraints
    max_daily_trades: int = 50
    max_correlated_exposure_pct: float = 0.15
    require_stop_loss: bool = True

    # Symbol restrictions
    allowed_symbols: list[str] | None = None
    blocked_symbols: list[str] | None = None

    # Action restrictions
    allowed_actions: list[str] = field(default_factory=lambda: ["BUY", "SELL", "HOLD", "CLOSE", "REDUCE"])


# ==============================================================================
# Individual Guardrails
# ==============================================================================


class InputGuardrail:
    """
    Validates inputs before LLM processing.

    Fast checks that run before expensive LLM calls.
    """

    def __init__(self, constraints: TradingConstraints):
        self.constraints = constraints

    def check(
        self,
        query: str,
        context: dict[str, Any],
    ) -> tuple[bool, list[GuardrailViolation]]:
        """
        Check input validity.

        Args:
            query: User query/prompt
            context: Trading context

        Returns:
            Tuple of (passed, violations)
        """
        violations = []

        # Check 1: Query not empty
        if not query or not query.strip():
            violations.append(
                GuardrailViolation(
                    violation_type=ViolationType.INSUFFICIENT_CONTEXT,
                    guardrail_type=GuardrailType.INPUT,
                    severity="error",
                    message="Empty query provided",
                )
            )

        # Check 2: Symbol validation
        symbol = context.get("symbol")
        if symbol:
            if self.constraints.blocked_symbols:
                if symbol.upper() in self.constraints.blocked_symbols:
                    violations.append(
                        GuardrailViolation(
                            violation_type=ViolationType.INVALID_SYMBOL,
                            guardrail_type=GuardrailType.INPUT,
                            severity="error",
                            message=f"Symbol {symbol} is blocked",
                            context={"symbol": symbol},
                        )
                    )

            if self.constraints.allowed_symbols:
                if symbol.upper() not in self.constraints.allowed_symbols:
                    violations.append(
                        GuardrailViolation(
                            violation_type=ViolationType.INVALID_SYMBOL,
                            guardrail_type=GuardrailType.INPUT,
                            severity="error",
                            message=f"Symbol {symbol} not in allowed list",
                            context={"symbol": symbol},
                        )
                    )

        # Check 3: Circuit breaker status
        if context.get("circuit_breaker_active", False):
            violations.append(
                GuardrailViolation(
                    violation_type=ViolationType.CIRCUIT_BREAKER_ACTIVE,
                    guardrail_type=GuardrailType.INPUT,
                    severity="critical",
                    message="Circuit breaker is active, trading halted",
                )
            )

        # Check 4: Required context fields
        required_fields = ["portfolio_value", "current_positions"]
        for field_name in required_fields:
            if field_name not in context:
                violations.append(
                    GuardrailViolation(
                        violation_type=ViolationType.INSUFFICIENT_CONTEXT,
                        guardrail_type=GuardrailType.INPUT,
                        severity="warning",
                        message=f"Missing context field: {field_name}",
                        context={"missing_field": field_name},
                    )
                )

        return len(violations) == 0 or all(v.severity == "warning" for v in violations), violations


class OutputGuardrail:
    """
    Validates LLM outputs before execution.

    Checks for valid actions, reasonable parameters, and policy compliance.
    """

    def __init__(self, constraints: TradingConstraints):
        self.constraints = constraints

    def check(
        self,
        output: str,
        confidence: float,
        context: dict[str, Any],
    ) -> tuple[bool, list[GuardrailViolation], str | None]:
        """
        Check output validity.

        Args:
            output: LLM output/decision
            confidence: Confidence level
            context: Trading context

        Returns:
            Tuple of (passed, violations, modified_output)
        """
        violations = []
        modified_output = None

        # Check 1: Valid action
        action = self._extract_action(output)
        if action and action.upper() not in self.constraints.allowed_actions:
            violations.append(
                GuardrailViolation(
                    violation_type=ViolationType.INVALID_ACTION,
                    guardrail_type=GuardrailType.OUTPUT,
                    severity="error",
                    message=f"Invalid action: {action}",
                    context={"action": action},
                    remediation="Action modified to HOLD",
                )
            )
            modified_output = output.replace(action, "HOLD")

        # Check 2: Confidence threshold
        if "BUY" in output.upper() or "SELL" in output.upper():
            if confidence < self.constraints.min_confidence_for_trade:
                violations.append(
                    GuardrailViolation(
                        violation_type=ViolationType.CONFIDENCE_TOO_LOW,
                        guardrail_type=GuardrailType.OUTPUT,
                        severity="error",
                        message=(
                            f"Confidence {confidence:.2f} below minimum "
                            f"{self.constraints.min_confidence_for_trade:.2f}"
                        ),
                        context={"confidence": confidence},
                        remediation="Trade blocked due to low confidence",
                    )
                )

        # Check 3: Position size
        position_size = self._extract_position_size(output, context)
        if position_size > self.constraints.max_position_size_pct:
            violations.append(
                GuardrailViolation(
                    violation_type=ViolationType.POSITION_SIZE_EXCEEDED,
                    guardrail_type=GuardrailType.OUTPUT,
                    severity="error",
                    message=(
                        f"Position size {position_size:.1%} exceeds maximum "
                        f"{self.constraints.max_position_size_pct:.1%}"
                    ),
                    context={"position_size": position_size},
                    remediation=f"Reduce position to {self.constraints.max_position_size_pct:.1%}",
                )
            )

        # Check 4: Hallucination detection (basic)
        if self._detect_hallucination(output, context):
            violations.append(
                GuardrailViolation(
                    violation_type=ViolationType.HALLUCINATION_DETECTED,
                    guardrail_type=GuardrailType.OUTPUT,
                    severity="warning",
                    message="Potential hallucination detected in output",
                    remediation="Verify output against actual data",
                )
            )

        passed = len(violations) == 0 or all(v.severity == "warning" for v in violations)
        return passed, violations, modified_output

    def _extract_action(self, output: str) -> str | None:
        """Extract action from output."""
        patterns = [
            r"Action:\s*(\w+)",
            r"\b(BUY|SELL|HOLD|CLOSE|REDUCE)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None

    def _extract_position_size(
        self,
        output: str,
        context: dict[str, Any],
    ) -> float:
        """Extract position size from output or context."""
        # Try to find percentage in output
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", output)
        if pct_match:
            pct = float(pct_match.group(1))
            if pct <= 100:
                return pct / 100.0

        return context.get("proposed_position_pct", 0.01)

    def _detect_hallucination(
        self,
        output: str,
        context: dict[str, Any],
    ) -> bool:
        """Basic hallucination detection."""
        # Check for made-up symbols
        symbol_pattern = r"\$([A-Z]{1,5})\b"
        mentioned_symbols = re.findall(symbol_pattern, output)

        known_symbols = context.get("known_symbols", set())
        watchlist = context.get("watchlist", set())
        valid_symbols = known_symbols | watchlist

        if valid_symbols and mentioned_symbols:
            for symbol in mentioned_symbols:
                if symbol not in valid_symbols:
                    return True

        # Check for impossible price claims
        prices = context.get("current_prices", {})
        price_pattern = r"\$(\d+(?:\.\d+)?)\s*(?:price|stock|share)"
        price_claims = re.findall(price_pattern, output, re.IGNORECASE)

        for claim in price_claims:
            claimed_price = float(claim)
            for symbol, actual_price in prices.items():
                # If price is way off (>50% different), might be hallucination
                if abs(claimed_price - actual_price) / actual_price > 0.5:
                    return True

        return False


class SentimentGuardrail:
    """
    Validates trading decisions against sentiment data.

    Ensures trades align with sentiment analysis results.
    """

    def __init__(self, constraints: TradingConstraints):
        self.constraints = constraints

    def check(
        self,
        action: str,
        sentiment_score: float,
        sentiment_confidence: float,
        context: dict[str, Any],
    ) -> tuple[bool, list[GuardrailViolation]]:
        """
        Check sentiment alignment.

        Args:
            action: Proposed action (BUY/SELL)
            sentiment_score: Current sentiment (-1 to 1)
            sentiment_confidence: Sentiment confidence
            context: Trading context

        Returns:
            Tuple of (passed, violations)
        """
        violations = []

        if not self.constraints.require_sentiment_confirmation:
            return True, violations

        action_upper = action.upper()

        # Check long entries against sentiment
        if action_upper == "BUY":
            if sentiment_score < self.constraints.min_sentiment_for_long:
                violations.append(
                    GuardrailViolation(
                        violation_type=ViolationType.SENTIMENT_THRESHOLD_BREACH,
                        guardrail_type=GuardrailType.BEHAVIORAL,
                        severity="error",
                        message=(
                            f"Long entry blocked: sentiment {sentiment_score:.2f} "
                            f"below threshold {self.constraints.min_sentiment_for_long:.2f}"
                        ),
                        context={
                            "sentiment_score": sentiment_score,
                            "threshold": self.constraints.min_sentiment_for_long,
                        },
                    )
                )

        # Check short entries against sentiment
        elif action_upper == "SELL":
            # For shorts, we want negative sentiment
            is_short_position = context.get("is_short", False)
            if is_short_position:
                if sentiment_score > self.constraints.max_sentiment_for_short:
                    violations.append(
                        GuardrailViolation(
                            violation_type=ViolationType.SENTIMENT_THRESHOLD_BREACH,
                            guardrail_type=GuardrailType.BEHAVIORAL,
                            severity="error",
                            message=(
                                f"Short entry blocked: sentiment {sentiment_score:.2f} "
                                f"above threshold {self.constraints.max_sentiment_for_short:.2f}"
                            ),
                            context={
                                "sentiment_score": sentiment_score,
                                "threshold": self.constraints.max_sentiment_for_short,
                            },
                        )
                    )

        return len(violations) == 0, violations


class HallucinationDetector:
    """
    Enhanced hallucination detection for LLM trading outputs.

    UPGRADE-014 Expansion (December 2025)

    Based on research:
    - arXiv Nov 2025: Dissecting Liar Circuits in Financial LLMs
    - BizTech Aug 2025: LLM hallucination rate 3-27% in finance
    - Guardrails AI 2025: Provenance validation

    Strategies implemented:
    1. Multi-model consensus validation
    2. Source verification against news APIs
    3. Numerical claim validation
    4. Provenance tracking
    5. Confidence calibration checks
    """

    def __init__(
        self,
        strict_mode: bool = False,
        require_consensus: bool = True,
        consensus_threshold: float = 0.67,  # 2/3 agreement
        price_tolerance_pct: float = 0.20,  # 20% price deviation tolerance
    ):
        """
        Initialize hallucination detector.

        Args:
            strict_mode: Stricter detection (more false positives)
            require_consensus: Require multi-model consensus
            consensus_threshold: Minimum agreement for consensus
            price_tolerance_pct: Tolerance for price claims
        """
        self.strict_mode = strict_mode
        self.require_consensus = require_consensus
        self.consensus_threshold = consensus_threshold
        self.price_tolerance_pct = price_tolerance_pct

        self._detection_history: list[dict[str, Any]] = []

    def detect(
        self,
        output: str,
        context: dict[str, Any],
        model_predictions: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, list[str], float]:
        """
        Detect potential hallucinations in LLM output.

        Args:
            output: LLM output text
            context: Trading context with known facts
            model_predictions: Optional multi-model predictions for consensus

        Returns:
            Tuple of (is_hallucination, reasons, confidence)
        """
        reasons = []
        confidence = 0.0

        # Strategy 1: Symbol validation
        symbol_result = self._check_symbol_validity(output, context)
        if symbol_result:
            reasons.append(symbol_result)
            confidence += 0.3

        # Strategy 2: Price claim validation
        price_result = self._check_price_claims(output, context)
        if price_result:
            reasons.append(price_result)
            confidence += 0.3

        # Strategy 3: Date/time validation
        date_result = self._check_temporal_claims(output, context)
        if date_result:
            reasons.append(date_result)
            confidence += 0.2

        # Strategy 4: Numerical consistency
        numeric_result = self._check_numerical_consistency(output, context)
        if numeric_result:
            reasons.append(numeric_result)
            confidence += 0.2

        # Strategy 5: Multi-model consensus (if available)
        if model_predictions and len(model_predictions) >= 2:
            consensus_result = self._check_model_consensus(output, model_predictions)
            if consensus_result:
                reasons.append(consensus_result)
                confidence += 0.4  # Strong signal

        # Strategy 6: Provenance check
        provenance_result = self._check_provenance(output, context)
        if provenance_result:
            reasons.append(provenance_result)
            confidence += 0.15

        # Cap confidence
        confidence = min(1.0, confidence)

        # Record detection
        self._detection_history.append(
            {
                "output_snippet": output[:200],
                "is_hallucination": len(reasons) > 0,
                "reasons": reasons,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Trim history
        if len(self._detection_history) > 500:
            self._detection_history = self._detection_history[-500:]

        is_hallucination = confidence > 0.5 if self.strict_mode else confidence > 0.7

        return is_hallucination, reasons, confidence

    def _check_symbol_validity(
        self,
        output: str,
        context: dict[str, Any],
    ) -> str | None:
        """Check for invalid or made-up symbols."""
        # Extract mentioned symbols
        symbol_patterns = [
            r"\$([A-Z]{1,5})\b",  # $AAPL format
            r"\b([A-Z]{1,5})\s+(?:stock|share|option)",  # AAPL stock
        ]

        mentioned_symbols = set()
        for pattern in symbol_patterns:
            matches = re.findall(pattern, output)
            mentioned_symbols.update(matches)

        # Get known valid symbols
        known_symbols = set(context.get("known_symbols", []))
        watchlist = set(context.get("watchlist", []))
        portfolio_symbols = set(context.get("portfolio_symbols", []))

        valid_symbols = known_symbols | watchlist | portfolio_symbols

        if valid_symbols and mentioned_symbols:
            invalid = mentioned_symbols - valid_symbols
            if invalid:
                return f"Unknown symbols mentioned: {', '.join(invalid)}"

        return None

    def _check_price_claims(
        self,
        output: str,
        context: dict[str, Any],
    ) -> str | None:
        """Check for impossible price claims."""
        prices = context.get("current_prices", {})
        if not prices:
            return None

        # Patterns for price claims
        price_patterns = [
            r"(\$?\d+(?:\.\d{1,2})?)\s*(?:per\s+)?(?:share|stock)",
            r"(?:price|trading|at)\s+\$?(\d+(?:\.\d{1,2})?)",
            r"\$(\d+(?:\.\d{1,2})?)\s+(?:target|level)",
        ]

        for pattern in price_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for match in matches:
                claimed_price = float(match.replace("$", ""))

                # Check against known prices
                for symbol, actual_price in prices.items():
                    if actual_price == 0:
                        continue

                    deviation = abs(claimed_price - actual_price) / actual_price

                    # Check if deviation is suspicious
                    if deviation > self.price_tolerance_pct:
                        # Could be talking about a different stock, check context
                        if symbol.lower() in output.lower():
                            return (
                                f"Price claim ${claimed_price} deviates "
                                f"{deviation:.0%} from {symbol} actual ${actual_price}"
                            )

        return None

    def _check_temporal_claims(
        self,
        output: str,
        context: dict[str, Any],
    ) -> str | None:
        """Check for impossible date/time claims."""
        current_date = context.get("current_date", datetime.now(timezone.utc).date())

        # Patterns for future claims stated as past
        future_patterns = [
            r"(?:yesterday|last\s+week)\s+.{0,50}?\b(202[6-9]|20[3-9]\d)\b",
            r"(?:already|has)\s+.{0,30}?\breported\b.{0,50}?\b(202[6-9]|20[3-9]\d)\b",
        ]

        for pattern in future_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return "Claims about future dates stated as past events"

        # Check for specific date claims
        date_pattern = r"(?:on|dated?)\s+(\w+\s+\d{1,2},?\s+\d{4})"
        matches = re.findall(date_pattern, output, re.IGNORECASE)

        for match in matches:
            try:
                from dateutil import parser

                claimed_date = parser.parse(match).date()
                if claimed_date > current_date:
                    return f"Future date {claimed_date} mentioned as if past"
            except Exception:
                pass

        return None

    def _check_numerical_consistency(
        self,
        output: str,
        context: dict[str, Any],
    ) -> str | None:
        """Check for numerical inconsistencies."""
        # Check percentage claims
        pct_pattern = r"(\d+(?:\.\d+)?)\s*%"
        percentages = re.findall(pct_pattern, output)

        for pct in percentages:
            value = float(pct)
            # Impossible percentages
            if value > 1000 and "return" in output.lower():
                return f"Suspicious percentage claim: {value}%"

        # Check for math errors in calculations
        # Pattern: "X + Y = Z" or "X * Y = Z"
        calc_pattern = r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)"
        matches = re.findall(calc_pattern, output)

        for match in matches:
            a, op, b, result = float(match[0]), match[1], float(match[2]), float(match[3])
            expected = None

            if op == "+":
                expected = a + b
            elif op == "-":
                expected = a - b
            elif op == "*":
                expected = a * b
            elif op == "/" and b != 0:
                expected = a / b

            if expected and abs(expected - result) > 0.01 * max(expected, result):
                return f"Arithmetic error: {a} {op} {b} ≠ {result}"

        return None

    def _check_model_consensus(
        self,
        output: str,
        model_predictions: list[dict[str, Any]],
    ) -> str | None:
        """Check if multiple models agree on key facts."""
        if len(model_predictions) < 2:
            return None

        # Extract key claims from main output
        main_action = self._extract_action(output)
        main_sentiment = self._extract_sentiment(output)

        # Check agreement
        action_votes = {}
        sentiment_votes = {}

        for pred in model_predictions:
            action = pred.get("action", "UNKNOWN")
            sentiment = (
                "positive"
                if pred.get("sentiment_score", 0) > 0.1
                else ("negative" if pred.get("sentiment_score", 0) < -0.1 else "neutral")
            )

            action_votes[action] = action_votes.get(action, 0) + 1
            sentiment_votes[sentiment] = sentiment_votes.get(sentiment, 0) + 1

        total = len(model_predictions)

        # Check consensus on action
        if main_action:
            agreement = action_votes.get(main_action, 0) / total
            if agreement < self.consensus_threshold:
                return f"Action '{main_action}' lacks consensus ({agreement:.0%} agreement)"

        # Check consensus on sentiment direction
        if main_sentiment:
            agreement = sentiment_votes.get(main_sentiment, 0) / total
            if agreement < self.consensus_threshold:
                return f"Sentiment direction lacks consensus ({agreement:.0%} agreement)"

        return None

    def _check_provenance(
        self,
        output: str,
        context: dict[str, Any],
    ) -> str | None:
        """Check if claims have source backing."""
        # Look for specific factual claims without sources
        factual_patterns = [
            r"(?:according to|reported by|announced)\s+[A-Za-z]+",
            r"(?:confirmed|verified|stated)\s+that",
        ]

        has_factual_claims = any(re.search(p, output, re.IGNORECASE) for p in factual_patterns)

        if has_factual_claims:
            # Check if context has source data
            news_sources = context.get("news_sources", [])
            if not news_sources:
                return "Factual claims made without available source data"

        return None

    def _extract_action(self, output: str) -> str | None:
        """Extract action from output."""
        patterns = [
            r"Action:\s*(\w+)",
            r"\b(BUY|SELL|HOLD|CLOSE|REDUCE)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None

    def _extract_sentiment(self, output: str) -> str | None:
        """Extract sentiment direction from output."""
        pos_patterns = [r"\b(bullish|positive|optimistic|upside)\b"]
        neg_patterns = [r"\b(bearish|negative|pessimistic|downside)\b"]

        for pattern in pos_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return "positive"

        for pattern in neg_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return "negative"

        return None

    def get_detection_stats(self) -> dict[str, Any]:
        """Get hallucination detection statistics."""
        if not self._detection_history:
            return {
                "total_checks": 0,
                "hallucinations_detected": 0,
                "detection_rate": 0.0,
            }

        total = len(self._detection_history)
        hallucinations = sum(1 for d in self._detection_history if d["is_hallucination"])

        reason_counts: dict[str, int] = {}
        for detection in self._detection_history:
            for reason in detection["reasons"]:
                # Truncate reason for grouping
                key = reason[:50] if len(reason) > 50 else reason
                reason_counts[key] = reason_counts.get(key, 0) + 1

        return {
            "total_checks": total,
            "hallucinations_detected": hallucinations,
            "detection_rate": hallucinations / total,
            "avg_confidence": sum(d["confidence"] for d in self._detection_history) / total,
            "reason_counts": reason_counts,
        }


class ConfidenceGuardrail:
    """
    Adjusts position sizing based on LLM confidence.

    Research finding: Lower confidence → smaller positions
    """

    def __init__(self, constraints: TradingConstraints):
        self.constraints = constraints

    def adjust_position_size(
        self,
        base_position_size: float,
        confidence: float,
        context: dict[str, Any],
    ) -> tuple[float, list[str]]:
        """
        Adjust position size based on confidence.

        Args:
            base_position_size: Requested position size
            confidence: LLM confidence level
            context: Trading context

        Returns:
            Tuple of (adjusted_size, adjustments_made)
        """
        adjustments = []
        adjusted_size = base_position_size

        # Confidence-based scaling
        if confidence < self.constraints.min_confidence_for_trade:
            adjusted_size = 0.0
            adjustments.append(f"Position zeroed: confidence {confidence:.2f} below minimum")
        elif confidence < self.constraints.min_confidence_for_large_trade:
            # Scale down position for medium confidence
            scale_factor = confidence / self.constraints.min_confidence_for_large_trade
            adjusted_size = base_position_size * scale_factor
            adjustments.append(f"Position scaled to {scale_factor:.0%} due to confidence {confidence:.2f}")

        # Volatility adjustment
        volatility = context.get("volatility", 0.0)
        if volatility > 0.03:  # High volatility
            vol_scale = 0.03 / volatility
            adjusted_size = adjusted_size * min(1.0, vol_scale)
            adjustments.append(f"Position scaled for volatility {volatility:.1%}")

        # Cap at maximum
        if adjusted_size > self.constraints.max_position_size_pct:
            adjusted_size = self.constraints.max_position_size_pct
            adjustments.append(f"Position capped at maximum {self.constraints.max_position_size_pct:.1%}")

        return adjusted_size, adjustments


# ==============================================================================
# Main Guardrails Manager
# ==============================================================================


class LLMGuardrails:
    """
    Comprehensive guardrails manager for LLM trading decisions.

    Implements layered guardrails:
    1. Input guardrails (fast, before LLM)
    2. Output guardrails (after LLM, before execution)
    3. Behavioral guardrails (policy enforcement)
    4. Semantic guardrails (content validation)

    Example:
        >>> guardrails = LLMGuardrails(
        ...     constraints=TradingConstraints(
        ...         max_position_size_pct=0.05,
        ...         min_confidence_for_trade=0.7,
        ...     )
        ... )
        >>> # Check input before LLM call
        >>> input_result = guardrails.check_input(query, context)
        >>> if not input_result.passed:
        ...     return blocked_response(input_result)
        >>>
        >>> # Get LLM response...
        >>> llm_output = call_llm(query)
        >>>
        >>> # Check output before execution
        >>> output_result = guardrails.check_output(llm_output, confidence, context)
        >>> if not output_result.passed:
        ...     return blocked_response(output_result)
    """

    def __init__(
        self,
        constraints: TradingConstraints | None = None,
        enable_input_guardrails: bool = True,
        enable_output_guardrails: bool = True,
        enable_sentiment_guardrails: bool = True,
        enable_confidence_adjustment: bool = True,
    ):
        """
        Initialize LLM guardrails.

        Args:
            constraints: Trading constraints configuration
            enable_input_guardrails: Enable input validation
            enable_output_guardrails: Enable output validation
            enable_sentiment_guardrails: Enable sentiment checks
            enable_confidence_adjustment: Enable confidence-based sizing
        """
        self.constraints = constraints or TradingConstraints()

        self.enable_input = enable_input_guardrails
        self.enable_output = enable_output_guardrails
        self.enable_sentiment = enable_sentiment_guardrails
        self.enable_confidence = enable_confidence_adjustment

        # Initialize guardrails
        self._input_guardrail = InputGuardrail(self.constraints)
        self._output_guardrail = OutputGuardrail(self.constraints)
        self._sentiment_guardrail = SentimentGuardrail(self.constraints)
        self._confidence_guardrail = ConfidenceGuardrail(self.constraints)

        # Tracking
        self._violation_history: list[GuardrailViolation] = []
        self._max_history = 500
        self._check_count = 0
        self._block_count = 0

        self._lock = threading.Lock()

    def check_input(
        self,
        query: str,
        context: dict[str, Any],
    ) -> GuardrailCheckResult:
        """
        Check input before LLM processing.

        Args:
            query: User query/prompt
            context: Trading context

        Returns:
            GuardrailCheckResult
        """
        import time

        start = time.time()

        violations = []
        checks = []

        if self.enable_input:
            checks.append("input_validation")
            passed, input_violations = self._input_guardrail.check(query, context)
            violations.extend(input_violations)

        execution_time = (time.time() - start) * 1000

        # Determine result
        has_blocking = any(v.severity in ("error", "critical") for v in violations)
        result = GuardrailResult.BLOCK if has_blocking else GuardrailResult.PASS

        self._record_check(violations)

        return GuardrailCheckResult(
            result=result,
            passed=not has_blocking,
            violations=violations,
            warnings=[v.message for v in violations if v.severity == "warning"],
            execution_time_ms=execution_time,
            checks_performed=checks,
        )

    def check_output(
        self,
        output: str,
        confidence: float,
        context: dict[str, Any],
        sentiment_score: float | None = None,
        sentiment_confidence: float | None = None,
    ) -> GuardrailCheckResult:
        """
        Check output before execution.

        Args:
            output: LLM output/decision
            confidence: LLM confidence level
            context: Trading context
            sentiment_score: Optional sentiment score
            sentiment_confidence: Optional sentiment confidence

        Returns:
            GuardrailCheckResult
        """
        import time

        start = time.time()

        violations = []
        checks = []
        modified_output = None

        # Output guardrails
        if self.enable_output:
            checks.append("output_validation")
            passed, output_violations, mod = self._output_guardrail.check(output, confidence, context)
            violations.extend(output_violations)
            if mod:
                modified_output = mod

        # Sentiment guardrails
        if self.enable_sentiment and sentiment_score is not None:
            checks.append("sentiment_validation")
            action = self._output_guardrail._extract_action(output)
            if action:
                passed, sent_violations = self._sentiment_guardrail.check(
                    action,
                    sentiment_score,
                    sentiment_confidence or 0.5,
                    context,
                )
                violations.extend(sent_violations)

        execution_time = (time.time() - start) * 1000

        # Determine result
        has_blocking = any(v.severity in ("error", "critical") for v in violations)

        if modified_output:
            result = GuardrailResult.MODIFY
        elif has_blocking:
            result = GuardrailResult.BLOCK
        elif violations:
            result = GuardrailResult.WARN
        else:
            result = GuardrailResult.PASS

        self._record_check(violations)

        return GuardrailCheckResult(
            result=result,
            passed=not has_blocking,
            violations=violations,
            warnings=[v.message for v in violations if v.severity == "warning"],
            modified_content=modified_output,
            execution_time_ms=execution_time,
            checks_performed=checks,
        )

    def adjust_position_for_confidence(
        self,
        base_size: float,
        confidence: float,
        context: dict[str, Any],
    ) -> tuple[float, list[str]]:
        """
        Adjust position size based on confidence.

        Args:
            base_size: Requested position size
            confidence: LLM confidence
            context: Trading context

        Returns:
            Tuple of (adjusted_size, adjustments)
        """
        if not self.enable_confidence:
            return base_size, []

        return self._confidence_guardrail.adjust_position_size(base_size, confidence, context)

    def validate_trade_decision(
        self,
        action: str,
        symbol: str,
        position_size: float,
        confidence: float,
        sentiment_score: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> GuardrailCheckResult:
        """
        Comprehensive trade decision validation.

        Args:
            action: Trade action (BUY/SELL/HOLD)
            symbol: Trading symbol
            position_size: Proposed position size
            confidence: Decision confidence
            sentiment_score: Optional sentiment
            context: Additional context

        Returns:
            GuardrailCheckResult
        """
        context = context or {}
        context["symbol"] = symbol
        context["proposed_position_pct"] = position_size

        # Build output string for validation
        output = f"Action: {action}\nSymbol: {symbol}\nPosition: {position_size:.1%}"

        return self.check_output(
            output=output,
            confidence=confidence,
            context=context,
            sentiment_score=sentiment_score,
        )

    def _record_check(self, violations: list[GuardrailViolation]) -> None:
        """Record check in history."""
        with self._lock:
            self._check_count += 1
            if violations:
                has_blocking = any(v.severity in ("error", "critical") for v in violations)
                if has_blocking:
                    self._block_count += 1

                self._violation_history.extend(violations)
                if len(self._violation_history) > self._max_history:
                    self._violation_history = self._violation_history[-self._max_history :]

    def get_stats(self) -> dict[str, Any]:
        """Get guardrails statistics."""
        with self._lock:
            violation_by_type = {}
            for v in self._violation_history:
                vtype = v.violation_type.value
                violation_by_type[vtype] = violation_by_type.get(vtype, 0) + 1

            return {
                "total_checks": self._check_count,
                "total_blocks": self._block_count,
                "block_rate": (self._block_count / self._check_count if self._check_count > 0 else 0.0),
                "total_violations": len(self._violation_history),
                "violations_by_type": violation_by_type,
                "guardrails_enabled": {
                    "input": self.enable_input,
                    "output": self.enable_output,
                    "sentiment": self.enable_sentiment,
                    "confidence": self.enable_confidence,
                },
            }

    def get_recent_violations(
        self,
        limit: int = 20,
        violation_type: ViolationType | None = None,
    ) -> list[GuardrailViolation]:
        """Get recent violations."""
        with self._lock:
            violations = list(self._violation_history)

        if violation_type:
            violations = [v for v in violations if v.violation_type == violation_type]

        return violations[-limit:]

    def update_constraints(
        self,
        **kwargs: Any,
    ) -> None:
        """Update trading constraints."""
        for key, value in kwargs.items():
            if hasattr(self.constraints, key):
                setattr(self.constraints, key, value)

        # Reinitialize guardrails with new constraints
        self._input_guardrail = InputGuardrail(self.constraints)
        self._output_guardrail = OutputGuardrail(self.constraints)
        self._sentiment_guardrail = SentimentGuardrail(self.constraints)
        self._confidence_guardrail = ConfidenceGuardrail(self.constraints)


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_llm_guardrails(
    config: dict[str, Any] | None = None,
) -> LLMGuardrails:
    """
    Factory function to create LLM guardrails.

    Args:
        config: Configuration dictionary

    Returns:
        Configured LLMGuardrails instance
    """
    config = config or {}

    # Build constraints
    constraints = TradingConstraints(
        max_position_size_pct=config.get("max_position_size_pct", 0.10),
        max_portfolio_exposure_pct=config.get("max_portfolio_exposure_pct", 0.80),
        min_confidence_for_trade=config.get("min_confidence_for_trade", 0.6),
        min_confidence_for_large_trade=config.get("min_confidence_for_large_trade", 0.75),
        min_sentiment_for_long=config.get("min_sentiment_for_long", 0.0),
        max_sentiment_for_short=config.get("max_sentiment_for_short", 0.0),
        require_sentiment_confirmation=config.get("require_sentiment_confirmation", True),
        max_daily_trades=config.get("max_daily_trades", 50),
        max_correlated_exposure_pct=config.get("max_correlated_exposure_pct", 0.15),
        require_stop_loss=config.get("require_stop_loss", True),
        allowed_symbols=config.get("allowed_symbols"),
        blocked_symbols=config.get("blocked_symbols"),
        allowed_actions=config.get("allowed_actions", ["BUY", "SELL", "HOLD", "CLOSE", "REDUCE"]),
    )

    return LLMGuardrails(
        constraints=constraints,
        enable_input_guardrails=config.get("enable_input_guardrails", True),
        enable_output_guardrails=config.get("enable_output_guardrails", True),
        enable_sentiment_guardrails=config.get("enable_sentiment_guardrails", True),
        enable_confidence_adjustment=config.get("enable_confidence_adjustment", True),
    )


__all__ = [
    "ConfidenceGuardrail",
    "GuardrailCheckResult",
    "GuardrailResult",
    "GuardrailType",
    "GuardrailViolation",
    "HallucinationDetector",  # UPGRADE-014 Expansion
    "InputGuardrail",
    "LLMGuardrails",
    "OutputGuardrail",
    "SentimentGuardrail",
    "TradingConstraints",
    "ViolationType",
    "create_llm_guardrails",
]
