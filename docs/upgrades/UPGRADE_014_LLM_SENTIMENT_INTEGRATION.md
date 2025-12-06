# Upgrade Path: LLM Sentiment Integration

**Upgrade ID**: UPGRADE-014
**Iteration**: 1
**Date**: December 1, 2025
**Status**: ✅ Complete

---

## Target State

Integrate LLM-powered sentiment analysis into trading decisions for:

1. **Sentiment Entry Filter**: Block trades when sentiment is unfavorable
2. **News-Driven Alerts**: Alert on high-impact news events
3. **Circuit Breaker Integration**: Trigger halts on severe negative sentiment
4. **Confidence-Based Position Sizing**: Adjust risk based on LLM confidence
5. **Ensemble Enhancement**: Improve weighted voting accuracy
6. **Multi-Agent Debate**: Enhance bull/bear debate mechanism

---

## Scope

### Included

- Create `llm/sentiment_filter.py` for entry filtering
- Create `llm/news_alert_manager.py` for news-driven alerts
- Enhance `llm/agents/debate_mechanism.py` with structured rounds
- Create `llm/agents/llm_guardrails.py` for trading constraints
- Integrate sentiment with `models/circuit_breaker.py`
- Enhance `llm/ensemble.py` with dynamic weighting
- Create comprehensive tests
- Update configuration for sentiment thresholds

### Excluded

- Real-time social media integration (P2, requires Twitter API)
- Custom LLM fine-tuning (P2, requires training infrastructure)
- GEX/DEX options flow analysis (P2, requires specialized data)

---

## Research Summary

See: [LLM_SENTIMENT_RESEARCH.md](../research/LLM_SENTIMENT_RESEARCH.md)

### Key Findings Applied

| Finding | Source | Application |
|---------|--------|-------------|
| Multi-agent debate improves decisions | TradingAgents (2024) | Enhance debate_mechanism.py |
| FinBERT + aggregation adds alpha | arXiv (2023) | Add sentiment aggregation |
| Confidence-based position sizing | AI Trading Experiments (2025) | Risk adjustment in guardrails |
| Ensemble of medium LLMs > single large | arXiv (2025) | Dynamic weight adjustment |
| Sentiment as entry filter | QuantInsti (2024) | Create sentiment_filter.py |
| News-driven circuit breaker | Industry practice | Circuit breaker integration |

---

## Success Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Sentiment filter created | File exists | `llm/sentiment_filter.py` |
| News alert manager created | File exists | `llm/news_alert_manager.py` |
| LLM guardrails created | File exists | `llm/agents/llm_guardrails.py` |
| Circuit breaker integration | Feature works | Sentiment triggers halt |
| Tests created | Test count | >= 30 test cases |
| Debate mechanism enhanced | Structured rounds | Configurable |

---

## Dependencies

- [x] UPGRADE-009 Structured Logging complete
- [x] UPGRADE-012 Error Handling complete
- [x] UPGRADE-013 Monitoring & Alerting complete
- [x] Existing llm/ module with FinBERT, ensemble, providers
- [x] Existing debate_mechanism.py

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLM latency affects execution | Medium | Medium | Async processing, caching |
| False positive sentiment blocks | Medium | Low | Configurable thresholds |
| API rate limits | Low | Medium | Rate limiting, fallback |
| Model disagreement | Medium | Low | Ensemble voting, human override |

---

## Estimated Effort

- Sentiment Filter Core: 1.5 hours
- News Alert Manager: 1 hour
- LLM Guardrails: 1.5 hours
- Circuit Breaker Integration: 0.5 hours
- Debate Mechanism Enhancement: 1 hour
- Ensemble Enhancement: 0.5 hours
- Tests: 1.5 hours
- **Total**: ~7.5 hours

---

## Phase 2: Task Checklist

### Core Infrastructure (T1-T4)

| ID | Task | Est. | Depends | Priority |
|----|------|------|---------|----------|
| T1 | Create `llm/sentiment_filter.py` | 45m | - | P0 |
| T2 | Create `llm/news_alert_manager.py` | 30m | T1 | P0 |
| T3 | Create `llm/agents/llm_guardrails.py` | 45m | - | P0 |
| T4 | Enhance `llm/agents/debate_mechanism.py` | 30m | - | P1 |

### Integration (T5-T7)

| ID | Task | Est. | Depends | Priority |
|----|------|------|---------|----------|
| T5 | Integrate sentiment with circuit breaker | 20m | T1 | P0 |
| T6 | Enhance `llm/ensemble.py` weighting | 20m | - | P1 |
| T7 | Add configuration for thresholds | 15m | T1-T3 | P0 |

### Testing (T8-T9)

| ID | Task | Est. | Depends | Priority |
|----|------|------|---------|----------|
| T8 | Create `tests/test_sentiment_filter.py` | 45m | T1-T7 | P0 |
| T9 | Update llm/__init__.py exports | 10m | T1-T3 | P0 |

---

## Phase 3: Implementation

### T1: Sentiment Filter

```python
# llm/sentiment_filter.py
"""
Sentiment-based entry filter for trading decisions.

Uses ensemble sentiment analysis to determine if market conditions
are favorable for trade entry. Blocks entries when sentiment is
unfavorable to reduce false positives.

UPGRADE-014: LLM Sentiment Integration (December 2025)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from enum import Enum

class FilterDecision(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_REVIEW = "require_review"

@dataclass
class SentimentSignal:
    """Sentiment analysis result for filtering."""

    symbol: str
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    source: str  # "finbert", "ensemble", etc.
    timestamp: datetime
    articles_analyzed: int = 0

    @property
    def is_bullish(self) -> bool:
        return self.sentiment_score > 0.1

    @property
    def is_bearish(self) -> bool:
        return self.sentiment_score < -0.1

    @property
    def is_neutral(self) -> bool:
        return -0.1 <= self.sentiment_score <= 0.1

class SentimentFilter:
    """
    Filter trade entries based on sentiment analysis.

    Uses configurable thresholds to allow, block, or flag
    trades based on current sentiment conditions.
    """

    def __init__(
        self,
        min_sentiment_for_long: float = 0.0,
        max_sentiment_for_short: float = 0.0,
        min_confidence: float = 0.5,
        lookback_hours: int = 24,
        require_positive_trend: bool = False,
    ):
        self.min_sentiment_for_long = min_sentiment_for_long
        self.max_sentiment_for_short = max_sentiment_for_short
        self.min_confidence = min_confidence
        self.lookback_hours = lookback_hours
        self.require_positive_trend = require_positive_trend

        self._signal_history: Dict[str, List[SentimentSignal]] = {}

    def check_entry(
        self,
        symbol: str,
        direction: str,  # "long" or "short"
        current_signal: SentimentSignal,
    ) -> FilterDecision:
        """
        Check if trade entry should be allowed.

        Args:
            symbol: Trading symbol
            direction: "long" or "short"
            current_signal: Current sentiment analysis

        Returns:
            FilterDecision indicating allow/block/review
        """
        # Low confidence -> require review
        if current_signal.confidence < self.min_confidence:
            return FilterDecision.REQUIRE_REVIEW

        # Long entries need positive sentiment
        if direction == "long":
            if current_signal.sentiment_score < self.min_sentiment_for_long:
                return FilterDecision.BLOCK

        # Short entries need negative sentiment
        elif direction == "short":
            if current_signal.sentiment_score > self.max_sentiment_for_short:
                return FilterDecision.BLOCK

        # Check trend if required
        if self.require_positive_trend:
            if not self._check_trend(symbol, direction):
                return FilterDecision.REQUIRE_REVIEW

        return FilterDecision.ALLOW

    def _check_trend(self, symbol: str, direction: str) -> bool:
        """Check if sentiment trend supports direction."""
        history = self._signal_history.get(symbol, [])
        if len(history) < 2:
            return True  # Not enough data

        recent = history[-3:]  # Last 3 signals
        if direction == "long":
            return all(s.sentiment_score > 0 for s in recent)
        else:
            return all(s.sentiment_score < 0 for s in recent)
```

### T2: News Alert Manager

```python
# llm/news_alert_manager.py
"""
News-driven alert manager for trading decisions.

Monitors news sentiment and generates alerts for significant
events that may require trading action.

UPGRADE-014: LLM Sentiment Integration (December 2025)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional
from enum import Enum

class NewsImpact(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class NewsAlert:
    """Alert generated from news analysis."""

    symbol: str
    headline: str
    sentiment_score: float
    impact: NewsImpact
    source: str
    timestamp: datetime
    summary: str = ""
    action_recommended: str = ""

    @property
    def is_actionable(self) -> bool:
        return self.impact in (NewsImpact.HIGH, NewsImpact.CRITICAL)

class NewsAlertManager:
    """
    Manage news-driven alerts and actions.

    Analyzes incoming news, classifies impact, and triggers
    appropriate alerts or trading actions.
    """

    def __init__(
        self,
        alerting_service: Optional[Any] = None,
        circuit_breaker: Optional[Any] = None,
        critical_threshold: float = -0.7,
        high_impact_threshold: float = -0.5,
    ):
        self.alerting_service = alerting_service
        self.circuit_breaker = circuit_breaker
        self.critical_threshold = critical_threshold
        self.high_impact_threshold = high_impact_threshold

        self._alert_history: List[NewsAlert] = []
        self._listeners: List[Callable[[NewsAlert], None]] = []

    def process_news(
        self,
        symbol: str,
        headline: str,
        content: str,
        sentiment_score: float,
        source: str,
    ) -> Optional[NewsAlert]:
        """
        Process news item and generate alert if warranted.

        Args:
            symbol: Trading symbol
            headline: News headline
            content: Full news content
            sentiment_score: FinBERT/ensemble score
            source: News source

        Returns:
            NewsAlert if generated, None otherwise
        """
        impact = self._classify_impact(sentiment_score, headline)

        if impact == NewsImpact.LOW:
            return None

        alert = NewsAlert(
            symbol=symbol,
            headline=headline,
            sentiment_score=sentiment_score,
            impact=impact,
            source=source,
            timestamp=datetime.now(timezone.utc),
            action_recommended=self._recommend_action(impact, sentiment_score),
        )

        self._alert_history.append(alert)
        self._notify_listeners(alert)

        # Critical alerts may trigger circuit breaker
        if impact == NewsImpact.CRITICAL and self.circuit_breaker:
            self._trigger_circuit_breaker(alert)

        return alert

    def _classify_impact(
        self, sentiment_score: float, headline: str
    ) -> NewsImpact:
        """Classify news impact level."""
        # Check for critical keywords
        critical_keywords = ["bankrupt", "fraud", "investigation", "crash"]
        if any(kw in headline.lower() for kw in critical_keywords):
            if sentiment_score < self.critical_threshold:
                return NewsImpact.CRITICAL

        # Score-based classification
        if sentiment_score < self.critical_threshold:
            return NewsImpact.CRITICAL
        elif sentiment_score < self.high_impact_threshold:
            return NewsImpact.HIGH
        elif abs(sentiment_score) > 0.3:
            return NewsImpact.MEDIUM
        else:
            return NewsImpact.LOW

    def _recommend_action(
        self, impact: NewsImpact, sentiment_score: float
    ) -> str:
        """Recommend trading action based on impact."""
        if impact == NewsImpact.CRITICAL:
            return "HALT_TRADING"
        elif impact == NewsImpact.HIGH:
            return "REDUCE_EXPOSURE" if sentiment_score < 0 else "REVIEW_POSITIONS"
        elif impact == NewsImpact.MEDIUM:
            return "MONITOR_CLOSELY"
        else:
            return "NO_ACTION"
```

### T3: LLM Guardrails

```python
# llm/agents/llm_guardrails.py
"""
Trading-specific guardrails for LLM-powered decisions.

Implements safety constraints to prevent LLM decisions from
exceeding risk limits or violating trading rules.

UPGRADE-014: LLM Sentiment Integration (December 2025)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum

class GuardrailViolation(Enum):
    NONE = "none"
    POSITION_SIZE = "position_size_exceeded"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    SENTIMENT_MISMATCH = "sentiment_mismatch"
    RISK_LIMIT = "risk_limit_exceeded"
    DAILY_LOSS = "daily_loss_limit"
    HUMAN_REQUIRED = "human_approval_required"

@dataclass
class GuardrailCheck:
    """Result of guardrail check."""

    passed: bool
    violation: GuardrailViolation
    message: str
    suggested_adjustment: Optional[Dict[str, Any]] = None

class TradingGuardrails:
    """
    Enforce trading constraints on LLM decisions.

    Validates LLM-suggested trades against risk limits,
    confidence thresholds, and sentiment alignment.
    """

    def __init__(
        self,
        max_position_size_pct: float = 0.10,
        min_confidence_for_trade: float = 0.6,
        min_confidence_for_full_size: float = 0.8,
        require_sentiment_alignment: bool = True,
        max_daily_loss_pct: float = 0.03,
    ):
        self.max_position_size_pct = max_position_size_pct
        self.min_confidence_for_trade = min_confidence_for_trade
        self.min_confidence_for_full_size = min_confidence_for_full_size
        self.require_sentiment_alignment = require_sentiment_alignment
        self.max_daily_loss_pct = max_daily_loss_pct

    def check_trade(
        self,
        direction: str,
        position_size_pct: float,
        confidence: float,
        sentiment_score: float,
        current_daily_loss_pct: float = 0.0,
    ) -> GuardrailCheck:
        """
        Validate proposed trade against guardrails.

        Args:
            direction: "long" or "short"
            position_size_pct: Position size as % of portfolio
            confidence: LLM confidence (0-1)
            sentiment_score: Sentiment score (-1 to 1)
            current_daily_loss_pct: Current daily loss

        Returns:
            GuardrailCheck with pass/fail and adjustments
        """
        # Check daily loss limit
        if current_daily_loss_pct >= self.max_daily_loss_pct:
            return GuardrailCheck(
                passed=False,
                violation=GuardrailViolation.DAILY_LOSS,
                message="Daily loss limit reached",
            )

        # Check minimum confidence
        if confidence < self.min_confidence_for_trade:
            return GuardrailCheck(
                passed=False,
                violation=GuardrailViolation.CONFIDENCE_TOO_LOW,
                message=f"Confidence {confidence:.2f} below minimum {self.min_confidence_for_trade}",
            )

        # Check sentiment alignment
        if self.require_sentiment_alignment:
            if direction == "long" and sentiment_score < 0:
                return GuardrailCheck(
                    passed=False,
                    violation=GuardrailViolation.SENTIMENT_MISMATCH,
                    message="Long trade blocked: negative sentiment",
                )
            if direction == "short" and sentiment_score > 0:
                return GuardrailCheck(
                    passed=False,
                    violation=GuardrailViolation.SENTIMENT_MISMATCH,
                    message="Short trade blocked: positive sentiment",
                )

        # Adjust position size based on confidence
        adjusted_size = self._adjust_position_size(position_size_pct, confidence)

        if adjusted_size > self.max_position_size_pct:
            return GuardrailCheck(
                passed=False,
                violation=GuardrailViolation.POSITION_SIZE,
                message=f"Position size {adjusted_size:.1%} exceeds max {self.max_position_size_pct:.1%}",
                suggested_adjustment={"position_size_pct": self.max_position_size_pct},
            )

        return GuardrailCheck(
            passed=True,
            violation=GuardrailViolation.NONE,
            message="Trade approved",
            suggested_adjustment={"position_size_pct": adjusted_size} if adjusted_size != position_size_pct else None,
        )

    def _adjust_position_size(
        self, requested_size: float, confidence: float
    ) -> float:
        """Adjust position size based on confidence."""
        if confidence >= self.min_confidence_for_full_size:
            return requested_size

        # Scale down position for lower confidence
        confidence_ratio = confidence / self.min_confidence_for_full_size
        return requested_size * confidence_ratio
```

---

## Phase 4: Double-Check

**Date**: December 1, 2025
**Checked By**: Claude Code Agent

### Implementation Progress

| Task | Status | Notes |
|------|--------|-------|
| T1: Sentiment Filter | ✅ Complete | `llm/sentiment_filter.py` (191 lines) |
| T2: News Alert Manager | ✅ Complete | `llm/news_alert_manager.py` (300 lines) |
| T3: LLM Guardrails | ✅ Complete | `llm/agents/llm_guardrails.py` (294 lines) |
| T4: Debate Enhancement | ✅ Complete | Structured rounds in `debate_mechanism.py` |
| T5: Circuit Breaker Integration | ✅ Complete | Sentiment triggers in `circuit_breaker.py` |
| T6: Ensemble Enhancement | ✅ Complete | Dynamic weighting in `ensemble.py` |
| T7: Configuration | ✅ Complete | `config/settings.json` updated |
| T8: Tests | ✅ Complete | 49 tests in `tests/test_llm_sentiment.py` |
| T9: Exports | ✅ Complete | `llm/__init__.py` updated |
| T10: Algorithm Integration | ✅ Complete | `algorithms/hybrid_options_bot.py` integrated |

### Success Criteria Verification

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Sentiment filter created | File exists | `llm/sentiment_filter.py` | ✅ Pass |
| News alert manager created | File exists | `llm/news_alert_manager.py` | ✅ Pass |
| LLM guardrails created | File exists | `llm/agents/llm_guardrails.py` | ✅ Pass |
| Circuit breaker integration | Feature works | Integrated with triggers | ✅ Pass |
| Tests created | >= 30 | 49 tests passing | ✅ Pass |
| Algorithm integration | Sentiment in bot | `hybrid_options_bot.py` | ✅ Pass |
| Persistence | Object Store | `SentimentPersistence` class | ✅ Pass |

---

## Phase 5: Introspection Report

**Report Date**: December 1, 2025

### What Worked Well

1. **TradingAgents Research**: The TradingAgents multi-agent framework research provided excellent foundation for the debate mechanism and ensemble design.
2. **Modular Design**: Creating separate modules for filter, alerts, and guardrails allowed clean separation of concerns.
3. **Comprehensive Testing**: 49 test cases covered all edge cases and integration scenarios.
4. **Configuration-Driven**: All thresholds and settings are configurable via settings.json.

### Challenges Encountered

1. **Parameter Name Consistency**: Initial test failures due to mismatched parameter names between config and class signatures (e.g., `max_position_size` vs `max_position_size_pct`).
2. **Method Signatures**: LLMGuardrails `validate_trade_decision` required careful alignment with actual implementation.
3. **Enum Values**: Some NewsEventType values didn't exist as expected, required verification against actual implementation.

### Improvements Made During Implementation

1. Added `SentimentPersistence` class for storing sentiment history in Object Store.
2. Added `SENTIMENT_DATA` storage category with 0.2GB quota allocation.
3. Created comprehensive research notebook demonstrating all features.
4. Integrated sentiment checks into `hybrid_options_bot.py` main trading loop.

### Lessons Learned

1. **Test First**: Reading actual implementation before writing tests prevents parameter mismatch issues.
2. **Factory Functions**: Using `create_*` factory functions simplifies configuration and testing.
3. **Dynamic Weighting**: Ensemble weights should be persisted to Object Store for cross-session learning.

---

## Phase 6: Convergence Decision

**Decision**: ✅ CONVERGED

**Rationale**: All success criteria met with 49 passing tests, full algorithm integration, and persistence layer added.

**Next Steps**:
1. Add sentiment widgets to UI dashboard (deferred to UI sprint)
2. Connect to live news feeds for production use
3. Monitor dynamic weight adjustments in paper trading
4. Fine-tune thresholds based on backtest results

---

## Usage Examples

### Basic Sentiment Analysis

```python
from llm import SimpleSentimentAnalyzer

analyzer = SimpleSentimentAnalyzer()
result = analyzer.analyze("Apple reports record earnings")
print(f"Sentiment: {result.score:+.2f}, Confidence: {result.confidence:.2f}")
```

### Sentiment Filter for Trade Entry

```python
from llm import create_sentiment_filter, SentimentSignal
from datetime import datetime

# Create filter
filter = create_sentiment_filter(
    min_sentiment_for_long=0.0,
    max_sentiment_for_short=0.0,
    min_confidence=0.5,
)

# Add signal
signal = SentimentSignal(
    symbol="AAPL",
    score=0.5,
    confidence=0.8,
    source="ensemble",
    timestamp=datetime.now(),
)
filter.add_signal(signal)

# Check entry
result = filter.check_entry("AAPL", "long")
if result.decision == FilterDecision.ALLOW:
    print("Trade allowed")
else:
    print(f"Trade blocked: {result.reason}")
```

### LLM Guardrails Validation

```python
from llm import create_llm_guardrails, TradingConstraints

constraints = TradingConstraints(
    max_position_size_pct=0.25,
    min_confidence_for_trade=0.5,
    blocked_symbols=["GME", "AMC"],
    max_daily_trades=50,
)

guardrails = create_llm_guardrails(constraints=constraints)

result = guardrails.validate_trade_decision(
    action="buy",
    symbol="AAPL",
    position_size=0.15,
    confidence=0.8,
    sentiment_score=0.5,
)

if result.passed:
    print("Trade approved")
else:
    for v in result.violations:
        print(f"Violation: {v.rule} - {v.message}")
```

### Sentiment Persistence

```python
from utils.object_store import create_sentiment_persistence

# Create persistence (requires ObjectStoreManager)
persistence = create_sentiment_persistence(object_store_manager)

# Save sentiment history
persistence.save_sentiment_history("AAPL", [0.5, 0.3, 0.4])

# Load history
history = persistence.load_sentiment_history("AAPL")

# Get stats
stats = persistence.get_sentiment_stats()
print(f"Total symbols: {stats['history_count']}")
```

### Algorithm Integration

The `HybridOptionsBot` automatically integrates sentiment:

```python
# In hybrid_options_bot.py Initialize():
self._setup_sentiment_components()

# In OnData():
self._update_sentiment_signals(slice)  # Updates every 5 minutes

# Before trading:
result = self._validate_sentiment_for_trade(symbol, direction)
if result.decision == FilterDecision.BLOCK:
    return  # Skip trade
```

---

## Expansion Features (Introspection Round 2)

### Bug Fixes

1. **SentimentFilter.add_signal()**: Added public method for external callers
2. **SentimentFilter.check_entry()**: Made `current_signal` optional (uses history if not provided)

### New Features

1. **Sentiment Decay (Time-Weighted Analysis)**:
   - `get_weighted_sentiment(symbol, decay_rate=0.9)` - Exponential decay weighting
   - Recent signals weighted more heavily than older ones
   - Configurable decay rate and max age

2. **Bulk Sentiment Analysis**:
   - `get_bulk_sentiment(symbols, decay_rate)` - Analyze multiple symbols at once

3. **QuantConnect News Integration**:
   - Tiingo News data integration in `_get_recent_news()`
   - News caching to minimize API calls
   - Graceful fallback when news unavailable

4. **Sentiment Persistence on Shutdown**:
   - `_persist_sentiment_data()` in `OnEndOfAlgorithm`
   - Saves sentiment history for cross-session learning
   - Persists LLM provider performance for dynamic weighting

### Usage Example: Sentiment Decay

```python
# Get time-weighted sentiment (recent signals dominate)
result = sentiment_filter.get_weighted_sentiment("AAPL", decay_rate=0.9)
print(f"Weighted Score: {result['weighted_score']:+.2f}")
print(f"Trend: {result['trend']:+.2f}")
print(f"Is Bullish: {result['is_bullish']}")

# Analyze multiple symbols
bulk = sentiment_filter.get_bulk_sentiment(["AAPL", "TSLA", "MSFT"])
for symbol, data in bulk.items():
    print(f"{symbol}: {data['weighted_score']:+.2f}")
```

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-01 | Research completed (7 phases) |
| 2025-12-01 | Upgrade path created |
| 2025-12-01 | All core components implemented (T1-T9) |
| 2025-12-01 | 49 tests created and passing |
| 2025-12-01 | Algorithm integration completed (T10) |
| 2025-12-01 | SentimentPersistence added to Object Store |
| 2025-12-01 | Research notebook created |
| 2025-12-01 | Documentation updated with usage examples |
| 2025-12-01 | **Introspection Round 2**: Fixed add_signal bug |
| 2025-12-01 | **Introspection Round 2**: Made check_entry signal optional |
| 2025-12-01 | **Introspection Round 2**: Added sentiment decay feature |
| 2025-12-01 | **Introspection Round 2**: Added bulk sentiment analysis |
| 2025-12-01 | **Introspection Round 2**: Added QuantConnect Tiingo news integration |
| 2025-12-01 | **Introspection Round 2**: Added OnEndOfAlgorithm sentiment persistence |
| 2025-12-01 | **Introspection Round 2**: Added 17 new test cases |

---

## Related Documents

- [UPGRADE-013](UPGRADE_013_MONITORING_ALERTING.md) - Alerting Service
- [UPGRADE-012](UPGRADE_012_ERROR_HANDLING.md) - Error Handling
- [Research Document](../research/LLM_SENTIMENT_RESEARCH.md) - Full research
- [Research Notebook](../../research/sentiment_analysis.ipynb) - Interactive examples
- [Roadmap](../ROADMAP.md) - Phase 2 Week 3 tasks
