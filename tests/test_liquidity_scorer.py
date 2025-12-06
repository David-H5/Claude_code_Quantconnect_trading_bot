"""
Tests for Liquidity Scorer (Sprint 4 Expansion)

Tests option liquidity scoring for execution quality.
Part of UPGRADE-010 Sprint 4 Expansion.
"""

import pytest

from execution.liquidity_scorer import (
    ChainLiquiditySummary,
    LiquidityConfig,
    LiquidityRating,
    LiquidityScore,
    LiquidityScorer,
    OptionLiquidityData,
    create_liquidity_scorer,
)


class TestOptionLiquidityData:
    """Tests for OptionLiquidityData dataclass."""

    def test_creation(self):
        """Test data creation."""
        data = OptionLiquidityData(
            symbol="SPY_241220C450",
            bid=5.00,
            ask=5.10,
            volume=500,
            open_interest=2000,
        )

        assert data.bid == 5.00
        assert data.ask == 5.10

    def test_mid_price(self):
        """Test mid price calculation."""
        data = OptionLiquidityData(
            symbol="SPY_241220C450",
            bid=5.00,
            ask=5.20,
            volume=500,
            open_interest=2000,
        )

        assert data.mid_price == 5.10

    def test_spread(self):
        """Test absolute spread calculation."""
        data = OptionLiquidityData(
            symbol="SPY_241220C450",
            bid=5.00,
            ask=5.20,
            volume=500,
            open_interest=2000,
        )

        assert data.spread == pytest.approx(0.20)

    def test_spread_pct(self):
        """Test spread percentage calculation."""
        data = OptionLiquidityData(
            symbol="SPY_241220C450",
            bid=5.00,
            ask=5.20,
            volume=500,
            open_interest=2000,
        )

        # Spread = 0.20, Mid = 5.10
        assert data.spread_pct == pytest.approx(0.20 / 5.10, rel=0.01)

    def test_moneyness(self):
        """Test moneyness calculation."""
        data = OptionLiquidityData(
            symbol="SPY_241220C450",
            bid=5.00,
            ask=5.10,
            volume=500,
            open_interest=2000,
            underlying_price=450.0,
            strike=450.0,
        )

        assert data.moneyness == 1.0  # ATM

    def test_zero_ask(self):
        """Test handling of zero ask price."""
        data = OptionLiquidityData(
            symbol="SPY_241220C450",
            bid=5.00,
            ask=0.0,
            volume=500,
            open_interest=2000,
        )

        assert data.mid_price == 5.00
        assert data.spread == 0.0


class TestLiquidityConfig:
    """Tests for LiquidityConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LiquidityConfig()

        assert config.excellent_spread_pct == 0.02
        assert config.spread_weight == 0.50
        assert config.volume_weight == 0.30
        assert config.oi_weight == 0.20

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = LiquidityConfig()
        d = config.to_dict()

        assert d["excellent_spread_pct"] == 0.02
        assert "min_tradeable_score" in d


class TestLiquidityScore:
    """Tests for LiquidityScore dataclass."""

    def test_creation(self):
        """Test score creation."""
        score = LiquidityScore(
            score=85.0,
            rating=LiquidityRating.EXCELLENT,
            bid_ask_score=90.0,
            volume_score=80.0,
            oi_score=85.0,
            is_tradeable=True,
            estimated_slippage_bps=2.0,
            recommendation="Trade at will",
        )

        assert score.score == 85.0
        assert score.rating == LiquidityRating.EXCELLENT
        assert score.is_tradeable is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = LiquidityScore(
            score=70.0,
            rating=LiquidityRating.GOOD,
            bid_ask_score=75.0,
            volume_score=70.0,
            oi_score=65.0,
            is_tradeable=True,
            estimated_slippage_bps=3.0,
            recommendation="Standard execution",
        )

        d = score.to_dict()

        assert d["score"] == 70.0
        assert d["rating"] == "good"
        assert "timestamp" in d


class TestLiquidityScorer:
    """Tests for LiquidityScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create default scorer."""
        return LiquidityScorer()

    @pytest.fixture
    def excellent_contract(self):
        """Create contract with excellent liquidity."""
        return OptionLiquidityData(
            symbol="SPY_241220C450",
            bid=5.00,
            ask=5.05,  # 1% spread
            volume=2000,
            open_interest=10000,
        )

    @pytest.fixture
    def poor_contract(self):
        """Create contract with poor liquidity."""
        return OptionLiquidityData(
            symbol="XYZ_241220C50",
            bid=1.00,
            ask=1.30,  # 30% spread
            volume=5,
            open_interest=20,
        )

    @pytest.fixture
    def sample_chain(self):
        """Create sample option chain."""
        return [
            OptionLiquidityData(
                symbol="SPY_241220C445",
                bid=10.00,
                ask=10.10,
                volume=1500,
                open_interest=8000,
                strike=445.0,
                is_call=True,
            ),
            OptionLiquidityData(
                symbol="SPY_241220C450",
                bid=5.00,
                ask=5.08,
                volume=2000,
                open_interest=12000,
                strike=450.0,
                is_call=True,
            ),
            OptionLiquidityData(
                symbol="SPY_241220C455",
                bid=2.00,
                ask=2.10,
                volume=800,
                open_interest=5000,
                strike=455.0,
                is_call=True,
            ),
            OptionLiquidityData(
                symbol="SPY_241220P445",
                bid=2.50,
                ask=2.60,
                volume=1000,
                open_interest=6000,
                strike=445.0,
                is_call=False,
            ),
            OptionLiquidityData(
                symbol="SPY_241220P450",
                bid=5.20,
                ask=5.35,
                volume=1200,
                open_interest=9000,
                strike=450.0,
                is_call=False,
            ),
        ]

    def test_initialization(self, scorer):
        """Test scorer initialization."""
        assert scorer.config is not None
        assert scorer.config.spread_weight == 0.50

    def test_score_excellent_contract(self, scorer, excellent_contract):
        """Test scoring excellent liquidity contract."""
        score = scorer.score_contract(excellent_contract)

        assert score.rating == LiquidityRating.EXCELLENT
        assert score.score >= 85
        assert score.is_tradeable is True

    def test_score_poor_contract(self, scorer, poor_contract):
        """Test scoring poor liquidity contract."""
        score = scorer.score_contract(poor_contract)

        assert score.rating in [LiquidityRating.POOR, LiquidityRating.AVOID]
        assert score.is_tradeable is False

    def test_spread_score_excellent(self, scorer):
        """Test spread scoring for excellent spread."""
        # 1% spread is excellent
        score = scorer._score_spread(0.01)
        assert score == 100.0

    def test_spread_score_good(self, scorer):
        """Test spread scoring for good spread."""
        # 4% spread is good
        score = scorer._score_spread(0.04)
        assert 80 <= score <= 100

    def test_spread_score_poor(self, scorer):
        """Test spread scoring for poor spread."""
        # 15% spread is poor
        score = scorer._score_spread(0.15)
        assert score < 50

    def test_volume_score_excellent(self, scorer):
        """Test volume scoring for excellent volume."""
        score = scorer._score_volume(2000)
        assert score == 100.0

    def test_volume_score_low(self, scorer):
        """Test volume scoring for low volume."""
        score = scorer._score_volume(50)
        assert score < 50

    def test_oi_score_excellent(self, scorer):
        """Test OI scoring for excellent open interest."""
        score = scorer._score_open_interest(10000)
        assert score == 100.0

    def test_oi_score_low(self, scorer):
        """Test OI scoring for low open interest."""
        score = scorer._score_open_interest(100)
        assert score < 50

    def test_get_rating_excellent(self, scorer):
        """Test rating for excellent score."""
        rating = scorer._get_rating(90)
        assert rating == LiquidityRating.EXCELLENT

    def test_get_rating_good(self, scorer):
        """Test rating for good score."""
        rating = scorer._get_rating(75)
        assert rating == LiquidityRating.GOOD

    def test_get_rating_acceptable(self, scorer):
        """Test rating for acceptable score."""
        rating = scorer._get_rating(55)
        assert rating == LiquidityRating.ACCEPTABLE

    def test_get_rating_poor(self, scorer):
        """Test rating for poor score."""
        rating = scorer._get_rating(35)
        assert rating == LiquidityRating.POOR

    def test_get_rating_avoid(self, scorer):
        """Test rating for avoid score."""
        rating = scorer._get_rating(20)
        assert rating == LiquidityRating.AVOID

    def test_score_chain(self, scorer, sample_chain):
        """Test scoring entire option chain."""
        scores = scorer.score_chain(sample_chain, underlying="SPY")

        assert len(scores) == 5
        assert all(isinstance(s, LiquidityScore) for s in scores.values())

    def test_filter_liquid_contracts(self, scorer, sample_chain):
        """Test filtering liquid contracts."""
        # Add a poor contract
        sample_chain.append(
            OptionLiquidityData(
                symbol="SPY_241220C500",
                bid=0.05,
                ask=0.20,  # 300% spread
                volume=2,
                open_interest=10,
                strike=500.0,
                is_call=True,
            )
        )

        liquid = scorer.filter_liquid_contracts(sample_chain)

        # Should filter out the poor contract
        assert len(liquid) <= len(sample_chain)
        symbols = [c.symbol for c in liquid]
        assert "SPY_241220C500" not in symbols

    def test_filter_by_min_rating(self, scorer, sample_chain):
        """Test filtering by minimum rating."""
        liquid = scorer.filter_liquid_contracts(
            sample_chain,
            min_rating=LiquidityRating.GOOD,
        )

        for contract in liquid:
            score = scorer.score_contract(contract)
            assert score.rating in [LiquidityRating.EXCELLENT, LiquidityRating.GOOD]

    def test_get_chain_summary(self, scorer, sample_chain):
        """Test chain summary generation."""
        summary = scorer.get_chain_summary(sample_chain, underlying="SPY")

        assert summary.underlying == "SPY"
        assert summary.total_contracts == 5
        assert summary.avg_score > 0

    def test_get_chain_summary_empty(self, scorer):
        """Test chain summary with empty chain."""
        summary = scorer.get_chain_summary([], underlying="SPY")

        assert summary.total_contracts == 0
        assert summary.avg_score == 0.0

    def test_get_best_contracts(self, scorer, sample_chain):
        """Test getting best contracts."""
        best = scorer.get_best_contracts(sample_chain, n=3)

        assert len(best) == 3
        # Should be sorted by score descending
        scores = [s.score for _, s in best]
        assert scores == sorted(scores, reverse=True)

    def test_get_best_contracts_calls_only(self, scorer, sample_chain):
        """Test getting best call contracts only."""
        best = scorer.get_best_contracts(sample_chain, n=5, calls_only=True)

        for contract, _ in best:
            assert contract.is_call is True

    def test_get_best_contracts_puts_only(self, scorer, sample_chain):
        """Test getting best put contracts only."""
        best = scorer.get_best_contracts(sample_chain, n=5, puts_only=True)

        for contract, _ in best:
            assert contract.is_call is False

    def test_estimated_slippage(self, scorer, excellent_contract):
        """Test slippage estimation."""
        score = scorer.score_contract(excellent_contract)

        # Should estimate slippage based on spread
        assert score.estimated_slippage_bps > 0
        assert score.estimated_slippage_bps < 100  # Less than 1%


class TestChainLiquiditySummary:
    """Tests for ChainLiquiditySummary dataclass."""

    def test_creation(self):
        """Test summary creation."""
        summary = ChainLiquiditySummary(
            underlying="SPY",
            total_contracts=100,
            tradeable_contracts=85,
            excellent_count=20,
            good_count=40,
            acceptable_count=25,
            poor_count=10,
            avoid_count=5,
            avg_score=72.5,
        )

        assert summary.underlying == "SPY"
        assert summary.total_contracts == 100

    def test_to_dict(self):
        """Test conversion to dictionary."""
        summary = ChainLiquiditySummary(
            underlying="QQQ",
            total_contracts=50,
            tradeable_contracts=40,
            excellent_count=10,
            good_count=20,
            acceptable_count=10,
            poor_count=5,
            avoid_count=5,
            avg_score=68.0,
        )

        d = summary.to_dict()

        assert d["underlying"] == "QQQ"
        assert "timestamp" in d


class TestCreateLiquidityScorer:
    """Tests for factory function."""

    def test_create_default(self):
        """Test creating with defaults."""
        scorer = create_liquidity_scorer()

        assert isinstance(scorer, LiquidityScorer)
        assert scorer.config.spread_weight == 0.50

    def test_create_with_custom_weights(self):
        """Test creating with custom weights."""
        scorer = create_liquidity_scorer(
            spread_weight=0.60,
            volume_weight=0.25,
        )

        assert scorer.config.spread_weight == 0.60
        assert scorer.config.volume_weight == 0.25
        assert scorer.config.oi_weight == pytest.approx(0.15)

    def test_create_with_custom_min_score(self):
        """Test creating with custom minimum score."""
        scorer = create_liquidity_scorer(min_score=50.0)

        assert scorer.config.min_tradeable_score == 50.0


class TestLiquidityScorerEdgeCases:
    """Edge case tests."""

    def test_zero_volume(self):
        """Test contract with zero volume."""
        scorer = LiquidityScorer()
        data = OptionLiquidityData(
            symbol="TEST",
            bid=1.00,
            ask=1.05,
            volume=0,
            open_interest=100,
        )

        score = scorer.score_contract(data)

        assert score.volume_score == 0.0
        assert score.is_tradeable is False

    def test_zero_open_interest(self):
        """Test contract with zero open interest."""
        scorer = LiquidityScorer()
        data = OptionLiquidityData(
            symbol="TEST",
            bid=1.00,
            ask=1.05,
            volume=100,
            open_interest=0,
        )

        score = scorer.score_contract(data)

        assert score.oi_score == 0.0

    def test_very_wide_spread(self):
        """Test contract with very wide spread."""
        scorer = LiquidityScorer()
        data = OptionLiquidityData(
            symbol="TEST",
            bid=0.10,
            ask=0.50,  # 400% spread
            volume=100,
            open_interest=100,
        )

        score = scorer.score_contract(data)

        assert score.bid_ask_score < 20
        assert score.rating == LiquidityRating.AVOID

    def test_recommendation_text(self):
        """Test recommendation text generation."""
        scorer = LiquidityScorer()

        # Excellent
        data_excellent = OptionLiquidityData(symbol="TEST", bid=5.00, ask=5.05, volume=2000, open_interest=10000)
        score = scorer.score_contract(data_excellent)
        assert "excellent" in score.recommendation.lower() or "trade at will" in score.recommendation.lower()

        # Poor
        data_poor = OptionLiquidityData(symbol="TEST", bid=1.00, ask=1.50, volume=5, open_interest=20)
        score = scorer.score_contract(data_poor)
        assert "caution" in score.recommendation.lower() or "avoid" in score.recommendation.lower()
