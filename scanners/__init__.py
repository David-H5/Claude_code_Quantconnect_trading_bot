"""
Scanners Module

Layer: 3 (Domain Logic)
May import from: Layers 0-2 (utils, observability, config, models, compliance)
May be imported by: Layer 4 (algorithms, api, ui)

Provides market scanning capabilities for:
- Underpriced options detection
- Significant price movements with news corroboration
- Reddit sentiment scanning (UPGRADE-010 Sprint 3)
"""

from .movement_scanner import (
    MovementAlert,
    MovementDirection,
    MovementScanner,
    PriceData,
    create_movement_scanner,
)
from .options_scanner import (
    OptionContract,
    OptionsScanner,
    OptionType,
    UnderpricedOption,
    create_options_scanner,
)
from .reddit_scanner import (
    PRAW_AVAILABLE,
    ContentType,
    RedditAuthor,
    RedditComment,
    RedditPost,
    RedditScanner,
    RedditScannerConfig,
    RedditSentimentAlert,
    SubredditCategory,
    TickerMention,
    create_reddit_scanner,
)

# Unusual Activity Scanner (UPGRADE-010 Sprint 4 - December 2025)
from .unusual_activity_scanner import (
    ActivityHistory,
    ActivityType,
    DirectionBias,
    FlowAnalysis,
    OptionActivityData,
    UnusualActivityAlert,
    UnusualActivityConfig,
    UnusualActivityScanner,
    UrgencyLevel,
    create_unusual_activity_scanner,
)


__all__ = [
    # Options scanner
    "OptionType",
    "OptionContract",
    "UnderpricedOption",
    "OptionsScanner",
    "create_options_scanner",
    # Movement scanner
    "MovementDirection",
    "PriceData",
    "MovementAlert",
    "MovementScanner",
    "create_movement_scanner",
    # Reddit scanner (UPGRADE-010 Sprint 3)
    "PRAW_AVAILABLE",
    "ContentType",
    "SubredditCategory",
    "RedditScannerConfig",
    "RedditAuthor",
    "RedditPost",
    "RedditComment",
    "TickerMention",
    "RedditSentimentAlert",
    "RedditScanner",
    "create_reddit_scanner",
    # Unusual Activity Scanner (UPGRADE-010 Sprint 4)
    "ActivityHistory",
    "ActivityType",
    "DirectionBias",
    "FlowAnalysis",
    "OptionActivityData",
    "UnusualActivityAlert",
    "UnusualActivityConfig",
    "UnusualActivityScanner",
    "UrgencyLevel",
    "create_unusual_activity_scanner",
]
