"""
Earnings Call Analyzer Module

Analyzes earnings call transcripts for sentiment, tone shifts, and red flags.
Part of UPGRADE-010 Sprint 3 Expansion - Intelligence & Data Sources.
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class TranscriptSection(Enum):
    """Sections of an earnings call transcript."""

    OPERATOR = "operator"
    PREPARED_REMARKS = "prepared_remarks"
    QA_SESSION = "qa_session"
    CLOSING = "closing"
    UNKNOWN = "unknown"


class ToneCategory(Enum):
    """Management tone categories."""

    CONFIDENT = "confident"
    CAUTIOUS = "cautious"
    DEFENSIVE = "defensive"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    EVASIVE = "evasive"
    NEUTRAL = "neutral"


class RedFlagType(Enum):
    """Types of red flags detected in earnings calls."""

    GUIDANCE_CUT = "guidance_cut"
    HEDGING_LANGUAGE = "hedging_language"
    BLAME_EXTERNAL = "blame_external"
    EVASIVE_ANSWER = "evasive_answer"
    TONE_SHIFT = "tone_shift"
    CFO_DEPARTURE = "cfo_departure"
    ACCOUNTING_CHANGE = "accounting_change"
    ONE_TIME_ITEMS = "one_time_items"
    EXECUTIVE_SELLING = "executive_selling"
    UNUSUAL_METRICS = "unusual_metrics"


@dataclass
class SectionAnalysis:
    """Analysis of a transcript section."""

    section_type: TranscriptSection
    text: str
    start_position: int
    end_position: int
    word_count: int
    sentiment_score: float  # -1 to +1
    tone: ToneCategory
    key_phrases: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "section_type": self.section_type.value,
            "text_preview": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "word_count": self.word_count,
            "sentiment_score": self.sentiment_score,
            "tone": self.tone.value,
            "key_phrases": self.key_phrases,
        }


@dataclass
class RedFlag:
    """A red flag detected in the earnings call."""

    flag_type: RedFlagType
    description: str
    severity: float  # 0-1
    evidence: str
    position: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "flag_type": self.flag_type.value,
            "description": self.description,
            "severity": self.severity,
            "evidence": self.evidence[:200],
            "position": self.position,
        }


@dataclass
class EarningsCallResult:
    """Complete analysis of an earnings call."""

    # Identification
    ticker: str | None
    quarter: str | None
    call_date: datetime | None

    # Section analyses
    sections: list[SectionAnalysis]
    prepared_sentiment: float
    qa_sentiment: float
    sentiment_delta: float  # Change from prepared to Q&A

    # Tone analysis
    management_tone: ToneCategory
    tone_consistency: float  # 0-1, how consistent is the tone

    # Red flags
    red_flags: list[RedFlag]
    total_red_flag_severity: float

    # Key metrics
    guidance_direction: str  # "raised", "lowered", "maintained", "removed", "unknown"
    key_numbers: dict[str, str]  # Extracted numbers like revenue, EPS

    # Analysis metadata
    total_word_count: int
    analysis_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "quarter": self.quarter,
            "call_date": self.call_date.isoformat() if self.call_date else None,
            "sections": [s.to_dict() for s in self.sections],
            "prepared_sentiment": self.prepared_sentiment,
            "qa_sentiment": self.qa_sentiment,
            "sentiment_delta": self.sentiment_delta,
            "management_tone": self.management_tone.value,
            "tone_consistency": self.tone_consistency,
            "red_flags": [f.to_dict() for f in self.red_flags],
            "total_red_flag_severity": self.total_red_flag_severity,
            "guidance_direction": self.guidance_direction,
            "key_numbers": self.key_numbers,
            "total_word_count": self.total_word_count,
            "analysis_time_ms": self.analysis_time_ms,
        }

    @property
    def overall_signal(self) -> str:
        """Get overall trading signal from analysis."""
        # High red flags = bearish
        if self.total_red_flag_severity > 0.7:
            return "bearish"

        # Negative sentiment delta (worse in Q&A) = bearish
        if self.sentiment_delta < -0.3:
            return "bearish"

        # Lowered guidance = bearish
        if self.guidance_direction == "lowered":
            return "bearish"

        # Positive signals
        if self.guidance_direction == "raised" and self.prepared_sentiment > 0.3:
            return "bullish"

        if self.sentiment_delta > 0.2 and self.qa_sentiment > 0.3:
            return "bullish"

        return "neutral"


@dataclass
class EarningsAnalyzerConfig:
    """Configuration for earnings analyzer."""

    # Sentiment analysis
    use_finbert: bool = True
    sentiment_chunk_size: int = 500

    # Red flag detection
    red_flag_sensitivity: float = 0.5

    # Section parsing
    min_section_words: int = 50


class EarningsAnalyzer:
    """
    Analyzes earnings call transcripts.

    Provides:
    - Section parsing (prepared remarks vs Q&A)
    - Management tone analysis
    - Sentiment delta detection
    - Red flag identification
    - Key number extraction
    """

    # Section markers
    PREPARED_MARKERS = [
        r"prepared remarks",
        r"opening remarks",
        r"let me begin",
        r"i'?d like to (start|begin)",
        r"good (morning|afternoon|evening)",
    ]

    QA_MARKERS = [
        r"q&a|q ?and ?a",
        r"question[- ]and[- ]answer",
        r"open (it )?up for questions",
        r"take (your )?questions",
        r"first question",
        r"operator.{0,50}question",
    ]

    # Tone patterns
    CONFIDENT_PATTERNS = [
        r"\bwill\b",
        r"\bcommit(ted)?\b",
        r"\bguarantee\b",
        r"\bstrong(ly)?\b",
        r"\bexcellent\b",
        r"\boutstanding\b",
        r"\bexceed(ed)?\b",
        r"\brecord\b",
        r"\bbest[- ]in[- ]class\b",
    ]

    CAUTIOUS_PATTERNS = [
        r"\bmay\b",
        r"\bmight\b",
        r"\bcould\b",
        r"\bpossibly\b",
        r"\buncertain(ty)?\b",
        r"\bvolatil(e|ity)\b",
        r"\bchalleng(e|ing)\b",
        r"\bheadwind\b",
        r"\brisk\b",
    ]

    EVASIVE_PATTERNS = [
        r"\bi (don'?t|can'?t) (comment|speculate|predict)\b",
        r"\bit'?s too early\b",
        r"\bpremature to\b",
        r"\bnot (going to|gonna) (get into|discuss)\b",
        r"\bwe'?ll see\b",
        r"\btime will tell\b",
        r"\bi'?m not sure\b",
        r"\bthat'?s a good question\b",
    ]

    DEFENSIVE_PATTERNS = [
        r"\blet me (be )?clear\b",
        r"\bi want to emphasize\b",
        r"\bto be fair\b",
        r"\bin fairness\b",
        r"\byou have to understand\b",
        r"\bit'?s important to note\b",
        r"\bcontext\b",
    ]

    # Hedging language (red flag indicator)
    HEDGING_PATTERNS = [
        r"\bexcluding\b",
        r"\badjusted\b",
        r"\bnon[- ]?gaap\b",
        r"\bpro[- ]?forma\b",
        r"\bone[- ]?time\b",
        r"\bextraordinary\b",
        r"\bnormaliz(e|ed|ing)\b",
        r"\bstripping out\b",
    ]

    # External blame patterns
    BLAME_PATTERNS = [
        r"\bmacro (environment|conditions)\b",
        r"\bheadwinds?\b",
        r"\bsupply chain\b",
        r"\blabor (market|shortage)\b",
        r"\binflation(ary)?\b",
        r"\binterest rate\b",
        r"\bforeign exchange|fx\b",
        r"\bregulatory\b",
        r"\bweather\b",
        r"\bseasonality\b",
    ]

    # Guidance patterns
    GUIDANCE_RAISED = [
        r"rais(e|ed|ing) (our )?guidance",
        r"increas(e|ed|ing) (our )?outlook",
        r"upward revision",
        r"above (our )?prior",
        r"beat.{0,20}guidance",
        r"exceed.{0,20}expectations",
    ]

    GUIDANCE_LOWERED = [
        r"lower(ed|ing) (our )?guidance",
        r"reduc(e|ed|ing) (our )?outlook",
        r"downward revision",
        r"below (our )?prior",
        r"miss.{0,20}guidance",
        r"disappoint",
        r"revise down",
        r"cut.{0,20}forecast",
    ]

    def __init__(
        self,
        config: EarningsAnalyzerConfig | None = None,
        sentiment_analyzer: Any | None = None,
    ):
        """
        Initialize earnings analyzer.

        Args:
            config: Analyzer configuration
            sentiment_analyzer: Optional sentiment analyzer (FinBERT)
        """
        self.config = config or EarningsAnalyzerConfig()
        self._sentiment_analyzer = sentiment_analyzer

        # Compile patterns
        self._prepared_markers = [re.compile(p, re.IGNORECASE) for p in self.PREPARED_MARKERS]
        self._qa_markers = [re.compile(p, re.IGNORECASE) for p in self.QA_MARKERS]

    def analyze(
        self,
        transcript: str,
        ticker: str | None = None,
        quarter: str | None = None,
        call_date: datetime | None = None,
    ) -> EarningsCallResult:
        """
        Analyze an earnings call transcript.

        Args:
            transcript: Full transcript text
            ticker: Stock ticker symbol
            quarter: Quarter identifier (e.g., "Q3 2025")
            call_date: Date of the earnings call

        Returns:
            EarningsCallResult with complete analysis
        """
        start_time = time.time()

        # Parse sections
        sections = self._parse_sections(transcript)

        # Analyze each section
        analyzed_sections = []
        for section in sections:
            analysis = self._analyze_section(section)
            analyzed_sections.append(analysis)

        # Calculate section sentiments
        prepared_sections = [s for s in analyzed_sections if s.section_type == TranscriptSection.PREPARED_REMARKS]
        qa_sections = [s for s in analyzed_sections if s.section_type == TranscriptSection.QA_SESSION]

        prepared_sentiment = self._average_sentiment(prepared_sections)
        qa_sentiment = self._average_sentiment(qa_sections)
        sentiment_delta = qa_sentiment - prepared_sentiment

        # Determine management tone
        management_tone = self._determine_overall_tone(analyzed_sections)
        tone_consistency = self._calculate_tone_consistency(analyzed_sections)

        # Detect red flags
        red_flags = self._detect_red_flags(transcript, sentiment_delta)
        total_severity = sum(f.severity for f in red_flags)

        # Determine guidance direction
        guidance_direction = self._detect_guidance_direction(transcript)

        # Extract key numbers
        key_numbers = self._extract_key_numbers(transcript)

        analysis_time_ms = (time.time() - start_time) * 1000

        return EarningsCallResult(
            ticker=ticker,
            quarter=quarter,
            call_date=call_date,
            sections=analyzed_sections,
            prepared_sentiment=prepared_sentiment,
            qa_sentiment=qa_sentiment,
            sentiment_delta=sentiment_delta,
            management_tone=management_tone,
            tone_consistency=tone_consistency,
            red_flags=red_flags,
            total_red_flag_severity=min(1.0, total_severity),
            guidance_direction=guidance_direction,
            key_numbers=key_numbers,
            total_word_count=len(transcript.split()),
            analysis_time_ms=analysis_time_ms,
        )

    def _parse_sections(
        self,
        transcript: str,
    ) -> list[tuple[TranscriptSection, str, int, int]]:
        """Parse transcript into sections."""
        sections = []

        # Find Q&A start
        qa_start = None
        for pattern in self._qa_markers:
            match = pattern.search(transcript)
            if match:
                if qa_start is None or match.start() < qa_start:
                    qa_start = match.start()

        if qa_start:
            # Prepared remarks before Q&A
            if qa_start > self.config.min_section_words:
                sections.append(
                    (
                        TranscriptSection.PREPARED_REMARKS,
                        transcript[:qa_start],
                        0,
                        qa_start,
                    )
                )

            # Q&A section
            sections.append(
                (
                    TranscriptSection.QA_SESSION,
                    transcript[qa_start:],
                    qa_start,
                    len(transcript),
                )
            )
        else:
            # No clear Q&A marker, treat all as prepared remarks
            sections.append(
                (
                    TranscriptSection.PREPARED_REMARKS,
                    transcript,
                    0,
                    len(transcript),
                )
            )

        return sections

    def _analyze_section(
        self,
        section: tuple[TranscriptSection, str, int, int],
    ) -> SectionAnalysis:
        """Analyze a single section."""
        section_type, text, start, end = section

        # Calculate sentiment
        sentiment_score = self._calculate_sentiment(text)

        # Determine tone
        tone = self._determine_tone(text)

        # Extract key phrases
        key_phrases = self._extract_key_phrases(text)

        return SectionAnalysis(
            section_type=section_type,
            text=text,
            start_position=start,
            end_position=end,
            word_count=len(text.split()),
            sentiment_score=sentiment_score,
            tone=tone,
            key_phrases=key_phrases,
        )

    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score for text."""
        if self._sentiment_analyzer and self.config.use_finbert:
            try:
                # Analyze in chunks if text is long
                chunks = self._chunk_text(text, self.config.sentiment_chunk_size)
                scores = []

                for chunk in chunks:
                    result = self._sentiment_analyzer.analyze(chunk)
                    # Convert to -1 to +1 scale
                    if hasattr(result, "signal"):
                        if result.signal.name == "BULLISH":
                            scores.append(result.confidence)
                        elif result.signal.name == "BEARISH":
                            scores.append(-result.confidence)
                        else:
                            scores.append(0.0)
                    else:
                        scores.append(0.0)

                return sum(scores) / len(scores) if scores else 0.0

            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")

        # Fallback: simple pattern-based
        return self._simple_sentiment(text)

    def _simple_sentiment(self, text: str) -> float:
        """Simple pattern-based sentiment."""
        text_lower = text.lower()

        positive_words = ["strong", "growth", "exceed", "beat", "record", "improvement"]
        negative_words = ["weak", "decline", "miss", "below", "challenge", "headwind"]

        positive_count = sum(1 for w in positive_words if w in text_lower)
        negative_count = sum(1 for w in negative_words if w in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        return (positive_count - negative_count) / total

    def _chunk_text(self, text: str, chunk_size: int) -> list[str]:
        """Split text into chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)

        return chunks if chunks else [text]

    def _determine_tone(self, text: str) -> ToneCategory:
        """Determine tone category for text."""
        text_lower = text.lower()

        scores = {
            ToneCategory.CONFIDENT: self._count_patterns(text_lower, self.CONFIDENT_PATTERNS),
            ToneCategory.CAUTIOUS: self._count_patterns(text_lower, self.CAUTIOUS_PATTERNS),
            ToneCategory.EVASIVE: self._count_patterns(text_lower, self.EVASIVE_PATTERNS),
            ToneCategory.DEFENSIVE: self._count_patterns(text_lower, self.DEFENSIVE_PATTERNS),
        }

        if not any(scores.values()):
            return ToneCategory.NEUTRAL

        max_tone = max(scores.keys(), key=lambda t: scores[t])

        # Determine optimistic/pessimistic based on sentiment
        if max_tone == ToneCategory.CONFIDENT:
            return ToneCategory.OPTIMISTIC if scores[max_tone] > 3 else ToneCategory.CONFIDENT
        elif max_tone == ToneCategory.CAUTIOUS:
            return ToneCategory.PESSIMISTIC if scores[max_tone] > 3 else ToneCategory.CAUTIOUS

        return max_tone

    def _count_patterns(self, text: str, patterns: list[str]) -> int:
        """Count pattern matches in text."""
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count

    def _extract_key_phrases(self, text: str) -> list[str]:
        """Extract key phrases from text."""
        phrases = []

        # Guidance phrases
        guidance_patterns = [
            r"guidance of \$?[\d.,]+",
            r"expect(s|ing)? (revenue|earnings|eps) of \$?[\d.,]+",
            r"full[- ]year outlook",
            r"q\d outlook",
        ]

        for pattern in guidance_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            phrases.extend(matches[:3])

        # Numeric milestones
        milestone_pattern = r"record [\w\s]+ of \$?[\d.,]+"
        milestones = re.findall(milestone_pattern, text, re.IGNORECASE)
        phrases.extend(milestones[:3])

        return phrases[:10]

    def _average_sentiment(self, sections: list[SectionAnalysis]) -> float:
        """Calculate average sentiment across sections."""
        if not sections:
            return 0.0

        total_words = sum(s.word_count for s in sections)
        if total_words == 0:
            return 0.0

        weighted_sum = sum(s.sentiment_score * s.word_count for s in sections)

        return weighted_sum / total_words

    def _determine_overall_tone(
        self,
        sections: list[SectionAnalysis],
    ) -> ToneCategory:
        """Determine overall management tone."""
        if not sections:
            return ToneCategory.NEUTRAL

        tone_counts: dict[ToneCategory, int] = {}
        for section in sections:
            tone_counts[section.tone] = tone_counts.get(section.tone, 0) + section.word_count

        if not tone_counts:
            return ToneCategory.NEUTRAL

        return max(tone_counts.keys(), key=lambda t: tone_counts[t])

    def _calculate_tone_consistency(
        self,
        sections: list[SectionAnalysis],
    ) -> float:
        """Calculate how consistent the tone is across sections."""
        if len(sections) < 2:
            return 1.0

        tones = [s.tone for s in sections]
        main_tone = max(set(tones), key=tones.count)
        matches = sum(1 for t in tones if t == main_tone)

        return matches / len(tones)

    def _detect_red_flags(
        self,
        transcript: str,
        sentiment_delta: float,
    ) -> list[RedFlag]:
        """Detect red flags in transcript."""
        red_flags = []
        text_lower = transcript.lower()

        # Check hedging language
        hedging_count = self._count_patterns(text_lower, self.HEDGING_PATTERNS)
        if hedging_count > 5:
            red_flags.append(
                RedFlag(
                    flag_type=RedFlagType.HEDGING_LANGUAGE,
                    description=f"Excessive hedging language ({hedging_count} instances)",
                    severity=min(0.3 + hedging_count * 0.05, 0.8),
                    evidence="Multiple uses of 'adjusted', 'excluding', 'non-GAAP'",
                    position=0,
                )
            )

        # Check external blame
        blame_count = self._count_patterns(text_lower, self.BLAME_PATTERNS)
        if blame_count > 4:
            red_flags.append(
                RedFlag(
                    flag_type=RedFlagType.BLAME_EXTERNAL,
                    description=f"Frequent external blame ({blame_count} instances)",
                    severity=min(0.2 + blame_count * 0.05, 0.6),
                    evidence="Multiple references to macro/supply chain/headwinds",
                    position=0,
                )
            )

        # Check evasive language
        evasive_count = self._count_patterns(text_lower, self.EVASIVE_PATTERNS)
        if evasive_count > 3:
            red_flags.append(
                RedFlag(
                    flag_type=RedFlagType.EVASIVE_ANSWER,
                    description=f"Evasive responses detected ({evasive_count} instances)",
                    severity=min(0.3 + evasive_count * 0.1, 0.8),
                    evidence="Uses of 'can't comment', 'too early', 'premature'",
                    position=0,
                )
            )

        # Tone shift (negative sentiment delta)
        if sentiment_delta < -0.2:
            red_flags.append(
                RedFlag(
                    flag_type=RedFlagType.TONE_SHIFT,
                    description=f"Negative tone shift in Q&A (delta: {sentiment_delta:.2f})",
                    severity=min(abs(sentiment_delta), 0.8),
                    evidence="Sentiment declined from prepared remarks to Q&A",
                    position=0,
                )
            )

        # Check for guidance cut language
        lowered_count = self._count_patterns(text_lower, self.GUIDANCE_LOWERED)
        if lowered_count > 0:
            red_flags.append(
                RedFlag(
                    flag_type=RedFlagType.GUIDANCE_CUT,
                    description="Guidance lowered or removed",
                    severity=0.7,
                    evidence="References to lowered guidance or outlook",
                    position=0,
                )
            )

        # Sort by severity
        red_flags.sort(key=lambda f: f.severity, reverse=True)

        return red_flags

    def _detect_guidance_direction(self, transcript: str) -> str:
        """Detect guidance direction from transcript."""
        text_lower = transcript.lower()

        raised_count = self._count_patterns(text_lower, self.GUIDANCE_RAISED)
        lowered_count = self._count_patterns(text_lower, self.GUIDANCE_LOWERED)

        if raised_count > lowered_count and raised_count > 0:
            return "raised"
        elif lowered_count > raised_count and lowered_count > 0:
            return "lowered"
        elif "maintain" in text_lower or "reiterat" in text_lower:
            return "maintained"
        elif "remov" in text_lower or "withdraw" in text_lower:
            return "removed"

        return "unknown"

    def _extract_key_numbers(self, transcript: str) -> dict[str, str]:
        """Extract key financial numbers from transcript."""
        numbers = {}

        # Revenue pattern
        rev_match = re.search(r"revenue (?:of |was )?\$?([\d.,]+)\s*(billion|million|b|m)?", transcript, re.IGNORECASE)
        if rev_match:
            numbers["revenue"] = rev_match.group(0)

        # EPS pattern
        eps_match = re.search(r"(?:eps|earnings per share) (?:of |was )?\$?([\d.]+)", transcript, re.IGNORECASE)
        if eps_match:
            numbers["eps"] = eps_match.group(0)

        # Margin pattern
        margin_match = re.search(
            r"(?:gross |operating |profit )?margin (?:of |was )?([\d.]+)%?", transcript, re.IGNORECASE
        )
        if margin_match:
            numbers["margin"] = margin_match.group(0)

        # Growth pattern
        growth_match = re.search(r"(?:revenue |sales )?growth (?:of |was )?([\d.]+)%", transcript, re.IGNORECASE)
        if growth_match:
            numbers["growth"] = growth_match.group(0)

        return numbers


def create_earnings_analyzer(
    config: EarningsAnalyzerConfig | None = None,
    sentiment_analyzer: Any | None = None,
) -> EarningsAnalyzer:
    """
    Factory function to create an earnings analyzer.

    Args:
        config: Optional configuration
        sentiment_analyzer: Optional sentiment analyzer

    Returns:
        Configured EarningsAnalyzer instance
    """
    return EarningsAnalyzer(config, sentiment_analyzer)
