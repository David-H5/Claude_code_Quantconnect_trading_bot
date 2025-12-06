"""
Entity Extractor Module

Extracts financial entities (symbols, sectors, companies) from text.
Part of UPGRADE-010 Sprint 3 Expansion - Intelligence & Data Sources.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of financial entities."""

    TICKER = "ticker"
    COMPANY = "company"
    SECTOR = "sector"
    INDEX = "index"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    PERSON = "person"
    ORGANIZATION = "organization"


@dataclass
class ExtractedEntity:
    """An entity extracted from text."""

    text: str
    entity_type: EntityType
    normalized: str  # Normalized form (e.g., ticker symbol)
    confidence: float
    start_pos: int
    end_pos: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "entity_type": self.entity_type.value,
            "normalized": self.normalized,
            "confidence": self.confidence,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "metadata": self.metadata,
        }


@dataclass
class ExtractionResult:
    """Result of entity extraction."""

    entities: list[ExtractedEntity]
    tickers: list[str]
    sectors: list[str]
    companies: list[str]
    extraction_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "tickers": self.tickers,
            "sectors": self.sectors,
            "companies": self.companies,
            "extraction_time_ms": self.extraction_time_ms,
        }


@dataclass
class EntityExtractorConfig:
    """Configuration for entity extraction."""

    # Ticker detection
    enable_ticker_detection: bool = True
    min_ticker_length: int = 1
    max_ticker_length: int = 5

    # Company detection
    enable_company_detection: bool = True
    fuzzy_match_threshold: float = 0.8

    # Sector detection
    enable_sector_detection: bool = True

    # Performance
    max_entities_per_text: int = 50
    cache_lookups: bool = True


class EntityExtractor:
    """
    Extracts financial entities from text.

    Uses pattern matching, dictionary lookup, and optional NER
    for comprehensive entity extraction.
    """

    # Common ticker pattern: 1-5 uppercase letters, optionally with $ prefix
    TICKER_PATTERN = re.compile(r"\$?([A-Z]{1,5})\b")

    # Ticker blacklist - common words that look like tickers
    TICKER_BLACKLIST: set[str] = {
        "I",
        "A",
        "AM",
        "PM",
        "US",
        "UK",
        "EU",
        "CEO",
        "CFO",
        "COO",
        "CTO",
        "IPO",
        "ETF",
        "GDP",
        "FBI",
        "CIA",
        "FDA",
        "SEC",
        "FED",
        "NYSE",
        "NASDAQ",
        "DOW",
        "IT",
        "AI",
        "ML",
        "API",
        "THE",
        "AND",
        "FOR",
        "ARE",
        "BUT",
        "NOT",
        "YOU",
        "ALL",
        "CAN",
        "HER",
        "WAS",
        "ONE",
        "OUR",
        "OUT",
        "HAS",
        "HIS",
        "HOW",
        "MAN",
        "NEW",
        "NOW",
        "OLD",
        "SEE",
        "WAY",
        "WHO",
        "BOY",
        "DID",
        "ITS",
        "LET",
        "PUT",
        "SAY",
        "SHE",
        "TOO",
        "USE",
        "EPS",
        "PE",
        "PB",
        "ROE",
        "ROA",
        "YOY",
        "QOQ",
        "MOM",
        "YTD",
        "ATH",
        "ATL",
        "EOD",
        "EOW",
        "EOM",
    }

    # Well-known tickers (always valid)
    KNOWN_TICKERS: set[str] = {
        "AAPL",
        "MSFT",
        "GOOGL",
        "GOOG",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "C",
        "V",
        "MA",
        "PYPL",
        "SQ",
        "DIS",
        "NFLX",
        "CMCSA",
        "T",
        "VZ",
        "TMUS",
        "CRM",
        "ORCL",
        "IBM",
        "INTC",
        "AMD",
        "QCOM",
        "AVGO",
        "TXN",
        "MU",
        "AMAT",
        "LRCX",
        "KLAC",
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
        "VOO",
        "BND",
        "TLT",
        "GLD",
        "XLF",
        "XLK",
        "XLE",
        "XLV",
        "XLI",
        "XLY",
        "XLP",
        "XLB",
        "XLU",
        "BA",
        "CAT",
        "MMM",
        "HON",
        "GE",
        "UNP",
        "UPS",
        "FDX",
        "LMT",
        "RTX",
        "JNJ",
        "PFE",
        "MRK",
        "ABBV",
        "BMY",
        "LLY",
        "UNH",
        "CVS",
        "CI",
        "KO",
        "PEP",
        "MCD",
        "SBUX",
        "NKE",
        "TGT",
        "WMT",
        "COST",
        "HD",
        "LOW",
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "OXY",
        "PSX",
        "VLO",
        "MPC",
        "HES",
        "BRK.A",
        "BRK.B",
        "BLK",
        "SCHW",
        "SPGI",
        "MCO",
        "ICE",
        "CME",
    }

    # Company name to ticker mapping
    COMPANY_TO_TICKER: dict[str, str] = {
        "apple": "AAPL",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "amazon": "AMZN",
        "meta": "META",
        "facebook": "META",
        "nvidia": "NVDA",
        "tesla": "TSLA",
        "jpmorgan": "JPM",
        "jp morgan": "JPM",
        "bank of america": "BAC",
        "wells fargo": "WFC",
        "goldman sachs": "GS",
        "morgan stanley": "MS",
        "disney": "DIS",
        "netflix": "NFLX",
        "salesforce": "CRM",
        "oracle": "ORCL",
        "intel": "INTC",
        "amd": "AMD",
        "qualcomm": "QCOM",
        "broadcom": "AVGO",
        "boeing": "BA",
        "caterpillar": "CAT",
        "johnson & johnson": "JNJ",
        "j&j": "JNJ",
        "pfizer": "PFE",
        "merck": "MRK",
        "coca-cola": "KO",
        "coke": "KO",
        "pepsi": "PEP",
        "pepsico": "PEP",
        "mcdonald's": "MCD",
        "mcdonalds": "MCD",
        "starbucks": "SBUX",
        "nike": "NKE",
        "walmart": "WMT",
        "target": "TGT",
        "costco": "COST",
        "home depot": "HD",
        "exxon": "XOM",
        "exxonmobil": "XOM",
        "chevron": "CVX",
        "berkshire": "BRK.B",
        "berkshire hathaway": "BRK.B",
        "visa": "V",
        "mastercard": "MA",
        "paypal": "PYPL",
        "uber": "UBER",
        "lyft": "LYFT",
        "airbnb": "ABNB",
        "zoom": "ZM",
        "palantir": "PLTR",
        "snowflake": "SNOW",
        "coinbase": "COIN",
        "robinhood": "HOOD",
    }

    # Sector keywords
    SECTOR_KEYWORDS: dict[str, str] = {
        "technology": "Technology",
        "tech": "Technology",
        "software": "Technology",
        "semiconductor": "Technology",
        "chip": "Technology",
        "chips": "Technology",
        "financial": "Financials",
        "finance": "Financials",
        "bank": "Financials",
        "banking": "Financials",
        "insurance": "Financials",
        "healthcare": "Healthcare",
        "health": "Healthcare",
        "pharma": "Healthcare",
        "pharmaceutical": "Healthcare",
        "biotech": "Healthcare",
        "medical": "Healthcare",
        "energy": "Energy",
        "oil": "Energy",
        "gas": "Energy",
        "petroleum": "Energy",
        "consumer": "Consumer",
        "retail": "Consumer",
        "restaurant": "Consumer",
        "industrial": "Industrials",
        "manufacturing": "Industrials",
        "aerospace": "Industrials",
        "defense": "Industrials",
        "utility": "Utilities",
        "utilities": "Utilities",
        "electric": "Utilities",
        "material": "Materials",
        "materials": "Materials",
        "mining": "Materials",
        "chemical": "Materials",
        "real estate": "Real Estate",
        "reit": "Real Estate",
        "property": "Real Estate",
        "communication": "Communication Services",
        "media": "Communication Services",
        "telecom": "Communication Services",
    }

    # Index names
    INDEX_NAMES: dict[str, str] = {
        "s&p 500": "SPY",
        "s&p500": "SPY",
        "sp500": "SPY",
        "spy": "SPY",
        "dow jones": "DIA",
        "dow": "DIA",
        "djia": "DIA",
        "nasdaq": "QQQ",
        "nasdaq 100": "QQQ",
        "qqq": "QQQ",
        "russell 2000": "IWM",
        "russell": "IWM",
        "iwm": "IWM",
        "vix": "VIX",
        "volatility index": "VIX",
    }

    def __init__(self, config: EntityExtractorConfig | None = None):
        """
        Initialize entity extractor.

        Args:
            config: Extraction configuration
        """
        self.config = config or EntityExtractorConfig()
        self._entity_cache: dict[str, list[ExtractedEntity]] = {}

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract all financial entities from text.

        Args:
            text: Text to analyze

        Returns:
            ExtractionResult with all extracted entities
        """
        import time

        start_time = time.time()

        entities: list[ExtractedEntity] = []

        # Extract tickers
        if self.config.enable_ticker_detection:
            ticker_entities = self._extract_tickers(text)
            entities.extend(ticker_entities)

        # Extract company names
        if self.config.enable_company_detection:
            company_entities = self._extract_companies(text)
            entities.extend(company_entities)

        # Extract sectors
        if self.config.enable_sector_detection:
            sector_entities = self._extract_sectors(text)
            entities.extend(sector_entities)

        # Extract indices
        index_entities = self._extract_indices(text)
        entities.extend(index_entities)

        # Deduplicate and limit
        entities = self._deduplicate_entities(entities)
        if len(entities) > self.config.max_entities_per_text:
            entities = entities[: self.config.max_entities_per_text]

        # Extract unique values
        tickers = list(set(e.normalized for e in entities if e.entity_type == EntityType.TICKER))
        sectors = list(set(e.normalized for e in entities if e.entity_type == EntityType.SECTOR))
        companies = list(set(e.normalized for e in entities if e.entity_type == EntityType.COMPANY))

        extraction_time_ms = (time.time() - start_time) * 1000

        return ExtractionResult(
            entities=entities,
            tickers=tickers,
            sectors=sectors,
            companies=companies,
            extraction_time_ms=extraction_time_ms,
        )

    def extract_tickers(self, text: str) -> list[str]:
        """
        Extract just ticker symbols from text.

        Args:
            text: Text to analyze

        Returns:
            List of ticker symbols
        """
        result = self.extract(text)
        return result.tickers

    def _extract_tickers(self, text: str) -> list[ExtractedEntity]:
        """Extract ticker symbols from text."""
        entities = []

        for match in self.TICKER_PATTERN.finditer(text):
            ticker = match.group(1)

            # Skip blacklisted words
            if ticker in self.TICKER_BLACKLIST:
                continue

            # Check length constraints
            if not (self.config.min_ticker_length <= len(ticker) <= self.config.max_ticker_length):
                continue

            # Calculate confidence
            confidence = self._calculate_ticker_confidence(ticker, text, match)

            if confidence > 0.3:  # Minimum threshold
                entities.append(
                    ExtractedEntity(
                        text=match.group(0),
                        entity_type=EntityType.TICKER,
                        normalized=ticker,
                        confidence=confidence,
                        start_pos=match.start(),
                        end_pos=match.end(),
                    )
                )

        return entities

    def _calculate_ticker_confidence(
        self,
        ticker: str,
        text: str,
        match: re.Match,
    ) -> float:
        """Calculate confidence that a match is a valid ticker."""
        confidence = 0.5  # Base confidence

        # Known ticker bonus
        if ticker in self.KNOWN_TICKERS:
            confidence += 0.4

        # Dollar sign prefix bonus
        if match.group(0).startswith("$"):
            confidence += 0.2

        # Surrounded by financial context
        context_start = max(0, match.start() - 50)
        context_end = min(len(text), match.end() + 50)
        context = text[context_start:context_end].lower()

        financial_terms = ["stock", "share", "price", "trading", "buy", "sell", "option", "call", "put"]
        for term in financial_terms:
            if term in context:
                confidence += 0.1
                break

        return min(1.0, confidence)

    def _extract_companies(self, text: str) -> list[ExtractedEntity]:
        """Extract company names from text."""
        entities = []
        text_lower = text.lower()

        for company, ticker in self.COMPANY_TO_TICKER.items():
            # Find all occurrences
            start = 0
            while True:
                pos = text_lower.find(company, start)
                if pos == -1:
                    break

                # Check word boundaries
                before_ok = pos == 0 or not text_lower[pos - 1].isalnum()
                after_ok = pos + len(company) >= len(text_lower) or not text_lower[pos + len(company)].isalnum()

                if before_ok and after_ok:
                    entities.append(
                        ExtractedEntity(
                            text=text[pos : pos + len(company)],
                            entity_type=EntityType.COMPANY,
                            normalized=company.title(),
                            confidence=0.9,
                            start_pos=pos,
                            end_pos=pos + len(company),
                            metadata={"ticker": ticker},
                        )
                    )

                start = pos + 1

        return entities

    def _extract_sectors(self, text: str) -> list[ExtractedEntity]:
        """Extract sector references from text."""
        entities = []
        text_lower = text.lower()

        for keyword, sector in self.SECTOR_KEYWORDS.items():
            start = 0
            while True:
                pos = text_lower.find(keyword, start)
                if pos == -1:
                    break

                # Check word boundaries
                before_ok = pos == 0 or not text_lower[pos - 1].isalnum()
                after_ok = pos + len(keyword) >= len(text_lower) or not text_lower[pos + len(keyword)].isalnum()

                if before_ok and after_ok:
                    entities.append(
                        ExtractedEntity(
                            text=text[pos : pos + len(keyword)],
                            entity_type=EntityType.SECTOR,
                            normalized=sector,
                            confidence=0.85,
                            start_pos=pos,
                            end_pos=pos + len(keyword),
                        )
                    )

                start = pos + 1

        return entities

    def _extract_indices(self, text: str) -> list[ExtractedEntity]:
        """Extract market index references from text."""
        entities = []
        text_lower = text.lower()

        for name, symbol in self.INDEX_NAMES.items():
            pos = text_lower.find(name)
            if pos != -1:
                entities.append(
                    ExtractedEntity(
                        text=text[pos : pos + len(name)],
                        entity_type=EntityType.INDEX,
                        normalized=symbol,
                        confidence=0.95,
                        start_pos=pos,
                        end_pos=pos + len(name),
                    )
                )

        return entities

    def _deduplicate_entities(
        self,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        """Remove duplicate entities, keeping highest confidence."""
        # Group by (normalized, entity_type)
        grouped: dict[tuple[str, EntityType], list[ExtractedEntity]] = {}

        for entity in entities:
            key = (entity.normalized, entity.entity_type)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(entity)

        # Keep highest confidence for each group
        result = []
        for group in grouped.values():
            best = max(group, key=lambda e: e.confidence)
            result.append(best)

        # Sort by position
        result.sort(key=lambda e: e.start_pos)

        return result

    def company_to_ticker(self, company: str) -> str | None:
        """
        Look up ticker symbol for a company name.

        Args:
            company: Company name

        Returns:
            Ticker symbol or None
        """
        return self.COMPANY_TO_TICKER.get(company.lower())

    def is_valid_ticker(self, ticker: str) -> bool:
        """
        Check if a string is likely a valid ticker.

        Args:
            ticker: Potential ticker symbol

        Returns:
            True if likely valid
        """
        ticker = ticker.upper()

        # Check blacklist
        if ticker in self.TICKER_BLACKLIST:
            return False

        # Check known tickers
        if ticker in self.KNOWN_TICKERS:
            return True

        # Check format
        if not re.match(r"^[A-Z]{1,5}$", ticker):
            return False

        return True


def create_entity_extractor(
    config: EntityExtractorConfig | None = None,
) -> EntityExtractor:
    """
    Factory function to create an entity extractor.

    Args:
        config: Optional configuration

    Returns:
        Configured EntityExtractor instance
    """
    return EntityExtractor(config)
