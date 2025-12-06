"""
Tests for Multi-Agent Architecture (UPGRADE-014 Feature 6)

Tests:
- NewsAnalyst agent
- MultiAgentConsensus mechanism
- Agent opinion aggregation
"""

from datetime import datetime, timezone

# Base imports
from llm.agents.base import AgentRole

# Multi-Agent Consensus imports
from llm.agents.multi_agent_consensus import (
    AgentOpinion,
    AgentType,
    ConsensusResult,
    ConsensusSignal,
    MultiAgentConsensus,
    create_multi_agent_consensus,
    opinion_from_agent_response,
)

# News Analyst imports
from llm.agents.news_analyst import (
    SOURCE_RELIABILITY,
    NewsAnalysis,
    NewsAnalyst,
    NewsAnalystResult,
    NewsEventType,
    NewsImpactLevel,
    NewsTimeRelevance,
    create_news_analyst,
)


# ============================================================================
# NewsAnalyst Tests
# ============================================================================


class TestNewsEventType:
    """Test NewsEventType enum."""

    def test_event_type_values(self):
        """Test all event type values exist."""
        assert NewsEventType.EARNINGS.value == "earnings"
        assert NewsEventType.MERGER_ACQUISITION.value == "merger_acquisition"
        assert NewsEventType.REGULATORY.value == "regulatory"
        assert NewsEventType.GENERAL.value == "general"

    def test_event_type_count(self):
        """Test expected number of event types."""
        assert len(NewsEventType) == 13


class TestNewsImpactLevel:
    """Test NewsImpactLevel enum."""

    def test_impact_level_values(self):
        """Test all impact level values."""
        assert NewsImpactLevel.CRITICAL.value == "critical"
        assert NewsImpactLevel.HIGH.value == "high"
        assert NewsImpactLevel.MEDIUM.value == "medium"
        assert NewsImpactLevel.LOW.value == "low"
        assert NewsImpactLevel.NEUTRAL.value == "neutral"


class TestNewsTimeRelevance:
    """Test NewsTimeRelevance enum."""

    def test_time_relevance_values(self):
        """Test all time relevance values."""
        assert NewsTimeRelevance.BREAKING.value == "breaking"
        assert NewsTimeRelevance.TODAY.value == "today"
        assert NewsTimeRelevance.RECENT.value == "recent"
        assert NewsTimeRelevance.OLD.value == "old"
        assert NewsTimeRelevance.STALE.value == "stale"


class TestNewsAnalysis:
    """Test NewsAnalysis dataclass."""

    def test_news_analysis_creation(self):
        """Test creating NewsAnalysis instance."""
        analysis = NewsAnalysis(
            headline="Apple beats earnings expectations",
            event_type=NewsEventType.EARNINGS,
            impact_level=NewsImpactLevel.HIGH,
            time_relevance=NewsTimeRelevance.TODAY,
            sentiment_score=0.7,
            sentiment_direction="bullish",
            confidence=0.85,
            key_entities=["Apple", "Tim Cook"],
            affected_symbols=["AAPL"],
            source_reliability=0.9,
            trading_implications=["Positive momentum expected"],
        )

        assert analysis.headline == "Apple beats earnings expectations"
        assert analysis.event_type == NewsEventType.EARNINGS
        assert analysis.sentiment_score == 0.7
        assert analysis.confidence == 0.85

    def test_news_analysis_to_dict(self):
        """Test NewsAnalysis serialization."""
        analysis = NewsAnalysis(
            headline="Test headline",
            event_type=NewsEventType.GENERAL,
            impact_level=NewsImpactLevel.LOW,
            time_relevance=NewsTimeRelevance.RECENT,
            sentiment_score=0.0,
            sentiment_direction="neutral",
            confidence=0.5,
            key_entities=[],
            affected_symbols=["TEST"],
            source_reliability=0.5,
            trading_implications=[],
        )

        result = analysis.to_dict()

        assert result["headline"] == "Test headline"
        assert result["event_type"] == "general"
        assert result["impact_level"] == "low"
        assert "timestamp" in result


class TestNewsAnalystResult:
    """Test NewsAnalystResult dataclass."""

    def test_result_creation(self):
        """Test creating NewsAnalystResult."""
        result = NewsAnalystResult(
            symbol="AAPL",
            analyses=[],
            overall_sentiment=0.5,
            overall_impact=NewsImpactLevel.MEDIUM,
            actionable=True,
            recommendation="Consider buying",
            key_risks=["Market volatility"],
            key_catalysts=["Earnings beat"],
        )

        assert result.symbol == "AAPL"
        assert result.overall_sentiment == 0.5
        assert result.actionable is True

    def test_result_to_dict(self):
        """Test NewsAnalystResult serialization."""
        result = NewsAnalystResult(
            symbol="TEST",
            analyses=[],
            overall_sentiment=0.0,
            overall_impact=NewsImpactLevel.NEUTRAL,
            actionable=False,
            recommendation="Hold",
            key_risks=[],
            key_catalysts=[],
        )

        data = result.to_dict()

        assert data["symbol"] == "TEST"
        assert data["overall_impact"] == "neutral"
        assert "timestamp" in data


class TestNewsAnalyst:
    """Test NewsAnalyst agent."""

    def test_analyst_initialization(self):
        """Test NewsAnalyst initialization without LLM client."""
        analyst = NewsAnalyst()

        assert analyst.name == "NewsAnalyst"
        assert analyst.role == AgentRole.NEWS_ANALYST
        assert analyst.source_reliability == SOURCE_RELIABILITY

    def test_analyst_with_custom_reliability(self):
        """Test NewsAnalyst with custom source reliability."""
        custom_reliability = {"custom_source": 0.99}
        analyst = NewsAnalyst(source_reliability=custom_reliability)

        assert analyst.source_reliability["custom_source"] == 0.99

    def test_analyze_basic(self):
        """Test basic analysis without news."""
        analyst = NewsAnalyst()

        result = analyst.analyze(
            query="Analyze AAPL news",
            context={"symbol": "AAPL"},
        )

        assert result.agent_name == "NewsAnalyst"
        assert result.success is True
        assert result.confidence >= 0.0

    def test_analyze_with_news_articles(self):
        """Test analysis with news articles."""
        analyst = NewsAnalyst()

        context = {
            "symbol": "AAPL",
            "news_articles": [
                {
                    "headline": "Apple reports record earnings, beats estimates",
                    "content": "Apple Inc. reported quarterly earnings that exceeded analyst expectations.",
                    "source": "reuters",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "headline": "Apple announces new iPhone launch",
                    "content": "Apple will unveil new iPhone models next month.",
                    "source": "bloomberg",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ],
        }

        result = analyst.analyze(
            query="Analyze AAPL news sentiment",
            context=context,
        )

        assert result.success is True
        assert "AAPL" in result.final_answer

    def test_classify_event_type_earnings(self):
        """Test event type classification for earnings."""
        analyst = NewsAnalyst()

        event_type = analyst._classify_event_type(
            headline="Apple reports Q3 earnings beat",
            content="Revenue exceeded expectations, EPS up 15%",
        )

        assert event_type == NewsEventType.EARNINGS

    def test_classify_event_type_merger(self):
        """Test event type classification for M&A."""
        analyst = NewsAnalyst()

        event_type = analyst._classify_event_type(
            headline="Microsoft announces acquisition of gaming company",
            content="The takeover deal valued at $10 billion",
        )

        assert event_type == NewsEventType.MERGER_ACQUISITION

    def test_assess_impact_critical(self):
        """Test critical impact assessment."""
        analyst = NewsAnalyst()

        impact = analyst._assess_impact_level(
            event_type=NewsEventType.REGULATORY,
            headline="FDA approves breakthrough cancer drug",
            content="Major breakthrough in cancer treatment",
        )

        assert impact == NewsImpactLevel.CRITICAL

    def test_assess_impact_high(self):
        """Test high impact assessment."""
        analyst = NewsAnalyst()

        impact = analyst._assess_impact_level(
            event_type=NewsEventType.EARNINGS,
            headline="Company beats earnings",
            content="Quarterly results exceeded expectations",
        )

        assert impact == NewsImpactLevel.HIGH

    def test_calculate_sentiment_bullish(self):
        """Test bullish sentiment calculation."""
        analyst = NewsAnalyst()

        score, direction = analyst._calculate_sentiment(
            headline="Stock surges on record profits",
            content="Company reports strong growth and beats expectations",
        )

        assert score > 0.2
        assert direction == "bullish"

    def test_calculate_sentiment_bearish(self):
        """Test bearish sentiment calculation."""
        analyst = NewsAnalyst()

        score, direction = analyst._calculate_sentiment(
            headline="Stock plunges on weak guidance",
            content="Company warns of decline and slowdown in sales",
        )

        assert score < -0.2
        assert direction == "bearish"

    def test_source_reliability_known(self):
        """Test source reliability for known sources."""
        analyst = NewsAnalyst()

        assert analyst._get_source_reliability("bloomberg") == 0.95
        assert analyst._get_source_reliability("reuters") == 0.95
        assert analyst._get_source_reliability("sec_filing") == 0.99

    def test_source_reliability_unknown(self):
        """Test source reliability for unknown sources."""
        analyst = NewsAnalyst()

        # "random_blog_123" matches "blog" (0.35) due to partial matching
        reliability = analyst._get_source_reliability("random_blog_123")
        assert reliability == 0.35  # Matches "blog"

        # Truly unknown source gets default
        reliability = analyst._get_source_reliability("xyzabc_news")
        assert reliability == 0.50  # Default for unknown


class TestCreateNewsAnalyst:
    """Test factory functions for NewsAnalyst."""

    def test_create_news_analyst_default(self):
        """Test creating NewsAnalyst with defaults."""
        analyst = create_news_analyst()

        assert isinstance(analyst, NewsAnalyst)
        assert analyst.name == "NewsAnalyst"

    def test_create_news_analyst_custom_reliability(self):
        """Test creating NewsAnalyst with custom reliability."""
        custom = {"my_source": 0.8}
        analyst = create_news_analyst(source_reliability=custom)

        assert analyst.source_reliability["my_source"] == 0.8


# ============================================================================
# MultiAgentConsensus Tests
# ============================================================================


class TestConsensusSignal:
    """Test ConsensusSignal enum."""

    def test_signal_values(self):
        """Test all signal values exist."""
        assert ConsensusSignal.STRONG_BUY.value == "strong_buy"
        assert ConsensusSignal.BUY.value == "buy"
        assert ConsensusSignal.HOLD.value == "hold"
        assert ConsensusSignal.SELL.value == "sell"
        assert ConsensusSignal.STRONG_SELL.value == "strong_sell"
        assert ConsensusSignal.CONFLICTED.value == "conflicted"


class TestAgentType:
    """Test AgentType enum."""

    def test_agent_type_values(self):
        """Test agent type values."""
        assert AgentType.SENTIMENT.value == "sentiment"
        assert AgentType.TECHNICAL.value == "technical"
        assert AgentType.NEWS.value == "news"
        assert AgentType.RISK.value == "risk"


class TestAgentOpinion:
    """Test AgentOpinion dataclass."""

    def test_opinion_creation(self):
        """Test creating AgentOpinion."""
        opinion = AgentOpinion(
            agent_type=AgentType.SENTIMENT,
            agent_name="SentimentAnalyst",
            signal_score=0.7,
            confidence=0.85,
            reasoning="Strong bullish sentiment",
            key_factors=["Positive news"],
            risk_factors=["High volatility"],
        )

        assert opinion.signal_score == 0.7
        assert opinion.confidence == 0.85
        assert opinion.agent_type == AgentType.SENTIMENT

    def test_opinion_to_dict(self):
        """Test AgentOpinion serialization."""
        opinion = AgentOpinion(
            agent_type=AgentType.TECHNICAL,
            agent_name="TechAnalyst",
            signal_score=0.3,
            confidence=0.6,
            reasoning="Price above support",
        )

        data = opinion.to_dict()

        assert data["agent_type"] == "technical"
        assert data["signal_score"] == 0.3
        assert "timestamp" in data

    def test_opinion_from_dict(self):
        """Test AgentOpinion deserialization."""
        data = {
            "agent_type": "sentiment",
            "agent_name": "Test",
            "signal_score": 0.5,
            "confidence": 0.7,
            "reasoning": "Test reasoning",
        }

        opinion = AgentOpinion.from_dict(data)

        assert opinion.agent_type == AgentType.SENTIMENT
        assert opinion.signal_score == 0.5


class TestConsensusResult:
    """Test ConsensusResult dataclass."""

    def test_result_creation(self):
        """Test creating ConsensusResult."""
        result = ConsensusResult(
            symbol="AAPL",
            signal=ConsensusSignal.BUY,
            consensus_score=0.6,
            confidence=0.8,
            agreement_score=0.85,
            participating_agents=3,
            agent_opinions=[],
            bullish_agents=["Agent1", "Agent2"],
            bearish_agents=[],
            neutral_agents=["Agent3"],
            key_bull_factors=["Strong momentum"],
            key_bear_factors=[],
            risk_factors=["Market volatility"],
            recommendation="Consider long entry",
            requires_debate=False,
        )

        assert result.signal == ConsensusSignal.BUY
        assert result.consensus_score == 0.6
        assert result.agreement_score == 0.85

    def test_result_to_dict(self):
        """Test ConsensusResult serialization."""
        result = ConsensusResult(
            symbol="TEST",
            signal=ConsensusSignal.HOLD,
            consensus_score=0.0,
            confidence=0.5,
            agreement_score=0.6,
            participating_agents=2,
            agent_opinions=[],
            bullish_agents=[],
            bearish_agents=[],
            neutral_agents=[],
            key_bull_factors=[],
            key_bear_factors=[],
            risk_factors=[],
            recommendation="Hold",
            requires_debate=False,
        )

        data = result.to_dict()

        assert data["symbol"] == "TEST"
        assert data["signal"] == "hold"


class TestMultiAgentConsensus:
    """Test MultiAgentConsensus class."""

    def test_consensus_initialization(self):
        """Test MultiAgentConsensus initialization."""
        consensus = MultiAgentConsensus()

        assert len(consensus.opinions) == 0
        assert len(consensus.weights) > 0

    def test_add_opinion(self):
        """Test adding single opinion."""
        consensus = MultiAgentConsensus()

        opinion = AgentOpinion(
            agent_type=AgentType.SENTIMENT,
            agent_name="Test",
            signal_score=0.5,
            confidence=0.7,
            reasoning="Test",
        )

        consensus.add_opinion(opinion)

        assert len(consensus.opinions) == 1

    def test_add_multiple_opinions(self):
        """Test adding multiple opinions."""
        consensus = MultiAgentConsensus()

        opinions = [
            AgentOpinion(
                agent_type=AgentType.SENTIMENT,
                agent_name="Sentiment",
                signal_score=0.6,
                confidence=0.8,
                reasoning="Bullish",
            ),
            AgentOpinion(
                agent_type=AgentType.TECHNICAL,
                agent_name="Technical",
                signal_score=0.4,
                confidence=0.7,
                reasoning="Above support",
            ),
        ]

        consensus.add_opinions(opinions)

        assert len(consensus.opinions) == 2

    def test_clear_opinions(self):
        """Test clearing opinions."""
        consensus = MultiAgentConsensus()
        consensus.add_opinion(
            AgentOpinion(
                agent_type=AgentType.SENTIMENT,
                agent_name="Test",
                signal_score=0.5,
                confidence=0.7,
                reasoning="Test",
            )
        )

        consensus.clear_opinions()

        assert len(consensus.opinions) == 0

    def test_calculate_consensus_insufficient_data(self):
        """Test consensus with insufficient opinions."""
        consensus = MultiAgentConsensus()

        result = consensus.calculate_consensus("AAPL")

        assert result.signal == ConsensusSignal.HOLD
        assert "Insufficient" in result.recommendation

    def test_calculate_consensus_bullish(self):
        """Test bullish consensus calculation."""
        consensus = MultiAgentConsensus()

        consensus.add_opinions(
            [
                AgentOpinion(
                    agent_type=AgentType.SENTIMENT,
                    agent_name="Sentiment",
                    signal_score=0.7,
                    confidence=0.8,
                    reasoning="Very bullish",
                ),
                AgentOpinion(
                    agent_type=AgentType.TECHNICAL,
                    agent_name="Technical",
                    signal_score=0.6,
                    confidence=0.85,
                    reasoning="Strong momentum",
                ),
                AgentOpinion(
                    agent_type=AgentType.NEWS,
                    agent_name="News",
                    signal_score=0.5,
                    confidence=0.7,
                    reasoning="Positive news",
                ),
            ]
        )

        result = consensus.calculate_consensus("AAPL")

        assert result.consensus_score > 0.3
        assert result.signal in [ConsensusSignal.BUY, ConsensusSignal.STRONG_BUY, ConsensusSignal.WEAK_BUY]

    def test_calculate_consensus_bearish(self):
        """Test bearish consensus calculation."""
        consensus = MultiAgentConsensus()

        consensus.add_opinions(
            [
                AgentOpinion(
                    agent_type=AgentType.SENTIMENT,
                    agent_name="Sentiment",
                    signal_score=-0.6,
                    confidence=0.8,
                    reasoning="Bearish sentiment",
                ),
                AgentOpinion(
                    agent_type=AgentType.TECHNICAL,
                    agent_name="Technical",
                    signal_score=-0.5,
                    confidence=0.7,
                    reasoning="Below support",
                ),
            ]
        )

        result = consensus.calculate_consensus("AAPL")

        assert result.consensus_score < -0.2
        assert result.signal in [ConsensusSignal.SELL, ConsensusSignal.STRONG_SELL, ConsensusSignal.WEAK_SELL]

    def test_calculate_consensus_conflicted(self):
        """Test conflicted consensus when agents disagree."""
        consensus = MultiAgentConsensus()

        consensus.add_opinions(
            [
                AgentOpinion(
                    agent_type=AgentType.SENTIMENT,
                    agent_name="Sentiment",
                    signal_score=0.8,
                    confidence=0.9,
                    reasoning="Very bullish",
                ),
                AgentOpinion(
                    agent_type=AgentType.TECHNICAL,
                    agent_name="Technical",
                    signal_score=-0.7,
                    confidence=0.85,
                    reasoning="Very bearish",
                ),
            ]
        )

        result = consensus.calculate_consensus("AAPL")

        # High disagreement should result in conflicted or low agreement
        assert result.agreement_score < 0.7

    def test_agreement_score_high_when_aligned(self):
        """Test high agreement when agents are aligned."""
        consensus = MultiAgentConsensus()

        consensus.add_opinions(
            [
                AgentOpinion(
                    agent_type=AgentType.SENTIMENT,
                    agent_name="Agent1",
                    signal_score=0.6,
                    confidence=0.8,
                    reasoning="Bullish",
                ),
                AgentOpinion(
                    agent_type=AgentType.TECHNICAL,
                    agent_name="Agent2",
                    signal_score=0.55,
                    confidence=0.75,
                    reasoning="Bullish",
                ),
                AgentOpinion(
                    agent_type=AgentType.NEWS,
                    agent_name="Agent3",
                    signal_score=0.65,
                    confidence=0.7,
                    reasoning="Bullish",
                ),
            ]
        )

        result = consensus.calculate_consensus("AAPL")

        assert result.agreement_score > 0.7

    def test_set_custom_weights(self):
        """Test setting custom weights."""
        consensus = MultiAgentConsensus()

        custom_weights = {
            AgentType.SENTIMENT: 0.5,
            AgentType.TECHNICAL: 0.5,
        }

        consensus.set_weights(custom_weights)

        assert consensus.weights[AgentType.SENTIMENT] == 0.5

    def test_adjust_weights_for_high_volatility(self):
        """Test weight adjustment for high volatility regime."""
        consensus = MultiAgentConsensus()

        original_sentiment = consensus.weights.get(AgentType.SENTIMENT, 0.25)

        consensus.adjust_weights_for_regime(is_high_volatility=True)

        # Sentiment weight should be reduced in high vol
        assert consensus.weights.get(AgentType.SENTIMENT, 0) < original_sentiment

    def test_statistics(self):
        """Test consensus statistics tracking."""
        consensus = MultiAgentConsensus()

        # Add opinions and calculate
        consensus.add_opinions(
            [
                AgentOpinion(
                    agent_type=AgentType.SENTIMENT,
                    agent_name="Test1",
                    signal_score=0.5,
                    confidence=0.7,
                    reasoning="Test",
                ),
                AgentOpinion(
                    agent_type=AgentType.TECHNICAL,
                    agent_name="Test2",
                    signal_score=0.4,
                    confidence=0.6,
                    reasoning="Test",
                ),
            ]
        )

        consensus.calculate_consensus("AAPL")

        stats = consensus.get_statistics()

        assert stats["total_calculations"] == 1
        assert stats["avg_agreement"] > 0
        assert stats["avg_participating_agents"] == 2


class TestCreateMultiAgentConsensus:
    """Test factory functions."""

    def test_create_consensus_default(self):
        """Test creating consensus with defaults."""
        consensus = create_multi_agent_consensus()

        assert isinstance(consensus, MultiAgentConsensus)
        assert consensus.config.strong_signal_threshold == 0.7

    def test_create_consensus_custom(self):
        """Test creating consensus with custom config."""
        consensus = create_multi_agent_consensus(
            strong_signal_threshold=0.8,
            debate_trigger_threshold=0.5,
        )

        assert consensus.config.strong_signal_threshold == 0.8
        assert consensus.config.debate_trigger_threshold == 0.5

    def test_create_consensus_custom_weights(self):
        """Test creating consensus with custom weights."""
        weights = {AgentType.SENTIMENT: 0.8, AgentType.TECHNICAL: 0.2}
        consensus = create_multi_agent_consensus(custom_weights=weights)

        assert consensus.weights[AgentType.SENTIMENT] == 0.8


class TestOpinionFromAgentResponse:
    """Test opinion extraction from agent responses."""

    def test_opinion_from_mock_response(self):
        """Test extracting opinion from mock response."""

        class MockResponse:
            agent_name = "MockAgent"
            confidence = 0.75
            final_answer = '{"signal_score": 0.6, "reasoning": "Test"}'

        opinion = opinion_from_agent_response(MockResponse(), AgentType.SENTIMENT)

        assert opinion is not None
        assert opinion.signal_score == 0.6
        assert opinion.confidence == 0.75

    def test_opinion_from_invalid_response(self):
        """Test handling invalid response."""

        class MockResponse:
            agent_name = "Mock"
            confidence = 0.5
            final_answer = "not json"

        opinion = opinion_from_agent_response(MockResponse(), AgentType.TECHNICAL)

        # Should still create opinion with defaults
        assert opinion is not None
        assert opinion.signal_score == 0.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestMultiAgentIntegration:
    """Integration tests for multi-agent system."""

    def test_news_analyst_with_consensus(self):
        """Test integrating NewsAnalyst with consensus."""
        # Create news analyst
        news_analyst = create_news_analyst()

        # Analyze news
        news_result = news_analyst.analyze(
            query="Analyze AAPL news",
            context={
                "symbol": "AAPL",
                "news_articles": [
                    {
                        "headline": "Apple reports strong earnings",
                        "content": "Beat expectations",
                        "source": "reuters",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            },
        )

        # Create consensus
        consensus = create_multi_agent_consensus()

        # Convert to opinion
        opinion = opinion_from_agent_response(news_result, AgentType.NEWS)

        if opinion:
            consensus.add_opinion(opinion)

            # Add another opinion for consensus
            consensus.add_opinion(
                AgentOpinion(
                    agent_type=AgentType.TECHNICAL,
                    agent_name="TechnicalAnalyst",
                    signal_score=0.4,
                    confidence=0.7,
                    reasoning="Above moving average",
                )
            )

            result = consensus.calculate_consensus("AAPL")

            assert result.participating_agents >= 2

    def test_full_multi_agent_workflow(self):
        """Test complete multi-agent workflow."""
        consensus = create_multi_agent_consensus()

        # Simulate multiple agent opinions
        consensus.add_opinions(
            [
                AgentOpinion(
                    agent_type=AgentType.SENTIMENT,
                    agent_name="SentimentAnalyst",
                    signal_score=0.6,
                    confidence=0.85,
                    reasoning="FinBERT shows positive sentiment",
                    key_factors=["Positive news coverage"],
                ),
                AgentOpinion(
                    agent_type=AgentType.TECHNICAL,
                    agent_name="TechnicalAnalyst",
                    signal_score=0.4,
                    confidence=0.8,
                    reasoning="Price above 20-day SMA",
                    key_factors=["Bullish trend"],
                ),
                AgentOpinion(
                    agent_type=AgentType.NEWS,
                    agent_name="NewsAnalyst",
                    signal_score=0.5,
                    confidence=0.75,
                    reasoning="Recent earnings beat",
                    key_factors=["Earnings surprise"],
                ),
                AgentOpinion(
                    agent_type=AgentType.RISK,
                    agent_name="RiskManager",
                    signal_score=0.2,
                    confidence=0.7,
                    reasoning="Acceptable risk levels",
                    risk_factors=["Market volatility"],
                ),
            ]
        )

        result = consensus.calculate_consensus("AAPL")

        # Should have consensus with good agreement
        assert result.participating_agents == 4
        assert result.agreement_score > 0.5
        assert result.signal in [
            ConsensusSignal.BUY,
            ConsensusSignal.WEAK_BUY,
            ConsensusSignal.HOLD,
        ]
        assert len(result.bullish_agents) > 0


class TestAgentRoleIntegration:
    """Test AgentRole integration."""

    def test_news_analyst_role(self):
        """Test NEWS_ANALYST role exists."""
        assert AgentRole.NEWS_ANALYST.value == "news_analyst"

    def test_all_roles_exist(self):
        """Test all required roles exist."""
        assert hasattr(AgentRole, "SENTIMENT_ANALYST")
        assert hasattr(AgentRole, "TECHNICAL_ANALYST")
        assert hasattr(AgentRole, "NEWS_ANALYST")
        assert hasattr(AgentRole, "SUPERVISOR")
