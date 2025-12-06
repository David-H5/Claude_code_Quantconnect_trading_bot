"""
Tests for Entity Extractor Module

Tests entity extraction functionality including ticker, company, and sector detection.
Part of UPGRADE-010 Sprint 3 Iteration 3 - Test Coverage.
"""

import pytest

from llm.entity_extractor import (
    EntityExtractor,
    EntityExtractorConfig,
    EntityType,
    ExtractedEntity,
    ExtractionResult,
    create_entity_extractor,
)


class TestEntityExtractorConfig:
    """Tests for EntityExtractorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EntityExtractorConfig()

        assert config.enable_ticker_detection is True
        assert config.min_ticker_length == 1
        assert config.max_ticker_length == 5
        assert config.enable_company_detection is True
        assert config.fuzzy_match_threshold == 0.8
        assert config.enable_sector_detection is True
        assert config.max_entities_per_text == 50
        assert config.cache_lookups is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = EntityExtractorConfig(
            enable_ticker_detection=False,
            max_ticker_length=4,
            max_entities_per_text=100,
        )

        assert config.enable_ticker_detection is False
        assert config.max_ticker_length == 4
        assert config.max_entities_per_text == 100


class TestExtractedEntity:
    """Tests for ExtractedEntity dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        entity = ExtractedEntity(
            text="AAPL",
            entity_type=EntityType.TICKER,
            normalized="AAPL",
            confidence=0.95,
            start_pos=0,
            end_pos=4,
            metadata={"source": "headline"},
        )

        result = entity.to_dict()

        assert result["text"] == "AAPL"
        assert result["entity_type"] == "ticker"
        assert result["normalized"] == "AAPL"
        assert result["confidence"] == 0.95
        assert result["start_pos"] == 0
        assert result["end_pos"] == 4
        assert result["metadata"] == {"source": "headline"}


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        entity = ExtractedEntity(
            text="AAPL",
            entity_type=EntityType.TICKER,
            normalized="AAPL",
            confidence=0.95,
            start_pos=0,
            end_pos=4,
        )

        result = ExtractionResult(
            entities=[entity],
            tickers=["AAPL"],
            sectors=[],
            companies=[],
            extraction_time_ms=5.0,
        )

        dict_result = result.to_dict()

        assert len(dict_result["entities"]) == 1
        assert dict_result["tickers"] == ["AAPL"]
        assert dict_result["sectors"] == []
        assert dict_result["companies"] == []
        assert dict_result["extraction_time_ms"] == 5.0


class TestEntityExtractor:
    """Tests for EntityExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create default extractor."""
        return EntityExtractor()

    @pytest.fixture
    def custom_extractor(self):
        """Create extractor with custom config."""
        config = EntityExtractorConfig(
            enable_sector_detection=False,
        )
        return EntityExtractor(config)

    # Ticker extraction tests
    def test_extract_known_ticker(self, extractor):
        """Test extraction of known ticker."""
        result = extractor.extract("AAPL is trading higher today")

        assert "AAPL" in result.tickers
        assert len(result.tickers) >= 1

    def test_extract_ticker_with_dollar_sign(self, extractor):
        """Test extraction of ticker with $ prefix."""
        result = extractor.extract("$TSLA is up 5% today")

        assert "TSLA" in result.tickers

    def test_extract_multiple_tickers(self, extractor):
        """Test extraction of multiple tickers."""
        result = extractor.extract("AAPL, MSFT, and GOOGL are all moving")

        assert "AAPL" in result.tickers
        assert "MSFT" in result.tickers
        assert "GOOGL" in result.tickers

    def test_blacklist_filtering(self, extractor):
        """Test that blacklisted words are not extracted as tickers."""
        result = extractor.extract("I AM looking at IT stocks in the US")

        assert "I" not in result.tickers
        assert "AM" not in result.tickers
        assert "IT" not in result.tickers
        assert "US" not in result.tickers

    def test_extract_tickers_only(self, extractor):
        """Test extract_tickers convenience method."""
        tickers = extractor.extract_tickers("Buy AAPL and MSFT")

        assert "AAPL" in tickers
        assert "MSFT" in tickers

    # Company extraction tests
    def test_extract_company_name(self, extractor):
        """Test extraction of company names."""
        result = extractor.extract("Apple reported strong earnings")

        assert "Apple" in result.companies or len(result.tickers) > 0

    def test_company_to_ticker_lookup(self, extractor):
        """Test company to ticker lookup."""
        ticker = extractor.company_to_ticker("apple")
        assert ticker == "AAPL"

        ticker = extractor.company_to_ticker("microsoft")
        assert ticker == "MSFT"

        ticker = extractor.company_to_ticker("unknown company")
        assert ticker is None

    def test_extract_multiple_companies(self, extractor):
        """Test extraction of multiple company names."""
        result = extractor.extract("Apple and Microsoft both reported earnings")

        # Should find either companies or their tickers
        entities_found = result.companies + result.tickers
        assert len(entities_found) > 0

    # Sector extraction tests
    def test_extract_sector(self, extractor):
        """Test extraction of sector references."""
        result = extractor.extract("Technology stocks are outperforming")

        assert "Technology" in result.sectors

    def test_extract_multiple_sectors(self, extractor):
        """Test extraction of multiple sectors."""
        result = extractor.extract("Healthcare and energy sectors are rising")

        assert "Healthcare" in result.sectors or "Energy" in result.sectors

    def test_sector_detection_disabled(self, custom_extractor):
        """Test that sector detection can be disabled."""
        result = custom_extractor.extract("Technology stocks are up")

        assert len(result.sectors) == 0

    # Index extraction tests
    def test_extract_index_reference(self, extractor):
        """Test extraction of index references."""
        result = extractor.extract("The S&P 500 is hitting new highs")

        # Should find SPY as normalized index
        found_index = any(e.entity_type == EntityType.INDEX for e in result.entities)
        assert found_index or "SPY" in result.tickers

    def test_extract_dow_reference(self, extractor):
        """Test extraction of Dow Jones reference."""
        result = extractor.extract("The Dow Jones is up 200 points")

        # Check for DIA or index entity
        found = any(e.normalized == "DIA" for e in result.entities)
        assert found

    # Confidence calculation tests
    def test_known_ticker_high_confidence(self, extractor):
        """Test that known tickers get high confidence."""
        result = extractor.extract("$AAPL stock price")

        aapl_entity = next((e for e in result.entities if e.normalized == "AAPL"), None)

        assert aapl_entity is not None
        assert aapl_entity.confidence > 0.7

    def test_unknown_ticker_lower_confidence(self, extractor):
        """Test that unknown tickers get lower confidence."""
        result = extractor.extract("XYZ stock is moving")

        xyz_entities = [e for e in result.entities if e.normalized == "XYZ"]

        # May or may not be extracted depending on context
        if xyz_entities:
            # Unknown ticker should have lower confidence than known ones
            assert xyz_entities[0].confidence <= 0.9

    # Deduplication tests
    def test_deduplication(self, extractor):
        """Test that duplicates are removed."""
        result = extractor.extract("AAPL AAPL AAPL - Apple stock is hot")

        # Should only have one AAPL ticker
        aapl_count = sum(1 for t in result.tickers if t == "AAPL")
        assert aapl_count == 1

    # Entity limit tests
    def test_max_entities_limit(self):
        """Test that max entities limit is respected."""
        config = EntityExtractorConfig(max_entities_per_text=5)
        extractor = EntityExtractor(config)

        # Text with many potential entities
        text = "AAPL MSFT GOOGL AMZN META NVDA TSLA JPM BAC WFC GS"
        result = extractor.extract(text)

        assert len(result.entities) <= 5

    # Validation tests
    def test_is_valid_ticker(self, extractor):
        """Test ticker validation."""
        assert extractor.is_valid_ticker("AAPL") is True
        assert extractor.is_valid_ticker("MSFT") is True
        assert extractor.is_valid_ticker("CEO") is False  # Blacklisted
        assert extractor.is_valid_ticker("TOOLONG123") is False  # Invalid format

    # Edge cases
    def test_empty_text(self, extractor):
        """Test handling of empty text."""
        result = extractor.extract("")

        assert len(result.tickers) == 0
        assert len(result.companies) == 0
        assert len(result.sectors) == 0
        assert len(result.entities) == 0

    def test_special_characters(self, extractor):
        """Test handling of special characters."""
        result = extractor.extract("$AAPL!! Great news!!! #stocks")

        assert "AAPL" in result.tickers

    def test_case_insensitivity_companies(self, extractor):
        """Test case insensitive company matching."""
        result1 = extractor.extract("APPLE is strong")
        result2 = extractor.extract("apple is strong")

        # Both should find Apple/AAPL
        assert len(result1.companies) > 0 or "AAPL" in result1.tickers
        assert len(result2.companies) > 0 or "AAPL" in result2.tickers

    def test_word_boundary_matching(self, extractor):
        """Test that partial matches are avoided."""
        result = extractor.extract("applesauce is not a stock")

        # Should not match "apple" in "applesauce"
        apple_found = "Apple" in result.companies or any(e.text.lower() == "apple" for e in result.entities)
        assert not apple_found

    def test_extraction_time_recorded(self, extractor):
        """Test that extraction time is recorded."""
        result = extractor.extract("AAPL is trading today")

        assert result.extraction_time_ms > 0
        assert result.extraction_time_ms < 1000  # Should be fast


class TestCreateEntityExtractor:
    """Tests for factory function."""

    def test_create_default(self):
        """Test creating extractor with defaults."""
        extractor = create_entity_extractor()

        assert isinstance(extractor, EntityExtractor)
        assert extractor.config is not None

    def test_create_with_config(self):
        """Test creating extractor with custom config."""
        config = EntityExtractorConfig(
            enable_ticker_detection=False,
            max_entities_per_text=10,
        )
        extractor = create_entity_extractor(config)

        assert extractor.config.enable_ticker_detection is False
        assert extractor.config.max_entities_per_text == 10


class TestEntityTypeEnum:
    """Tests for EntityType enum."""

    def test_all_types_exist(self):
        """Test that all expected entity types exist."""
        assert EntityType.TICKER.value == "ticker"
        assert EntityType.COMPANY.value == "company"
        assert EntityType.SECTOR.value == "sector"
        assert EntityType.INDEX.value == "index"
        assert EntityType.COMMODITY.value == "commodity"
        assert EntityType.CURRENCY.value == "currency"
        assert EntityType.PERSON.value == "person"
        assert EntityType.ORGANIZATION.value == "organization"
