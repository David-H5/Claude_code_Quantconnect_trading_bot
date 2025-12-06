"""
LLM Integration Module

Layer: 3 (Domain Logic)
May import from: Layers 0-2 (utils, observability, config, models, compliance)
May be imported by: Layer 4 (algorithms, api, ui)

Provides AI-powered analysis capabilities including:
- Sentiment analysis (FinBERT, GPT-4o, Claude)
- News analysis and alerts
- Option chain analysis
- Ensemble predictions
- Sentiment-based entry filtering (UPGRADE-014)
- LLM guardrails for trading safety (UPGRADE-014)
"""

# LLM Guardrails (UPGRADE-014 - December 2025)
from .agents.llm_guardrails import (
    GuardrailCheckResult,
    GuardrailViolation,
    HallucinationDetector,  # UPGRADE-014 Expansion
    LLMGuardrails,
    TradingConstraints,
    create_llm_guardrails,
)
from .base import (
    AnalysisResult,
    BaseLLMProvider,
    BaseSentimentAnalyzer,
    NewsItem,
    Sentiment,
    SentimentResult,
)

# Bot Detector (UPGRADE-010 Sprint 3 - December 2025)
from .bot_detector import (
    BotConfidence,
    BotDetectionResult,
    BotDetector,
    BotDetectorConfig,
    BotIndicator,
    BotIndicatorResult,
    CoordinatedCampaign,
    create_bot_detector,
)

# Earnings Analyzer (UPGRADE-010 Sprint 3 Expansion - December 2025)
from .earnings_analyzer import (
    EarningsAnalyzer,
    EarningsAnalyzerConfig,
    EarningsCallResult,
    RedFlag,
    RedFlagType,
    SectionAnalysis,
    ToneCategory,
    TranscriptSection,
    create_earnings_analyzer,
)

# Emotion Detector (UPGRADE-010 Sprint 3 Expansion - December 2025)
from .emotion_detector import (
    EmotionDetector,
    EmotionDetectorConfig,
    EmotionIndicator,
    EmotionResult,
    MarketEmotion,
    create_emotion_detector,
)
from .ensemble import (
    DynamicWeightConfig,
    EnsembleResult,
    LLMEnsemble,
    ProviderPerformance,
    create_ensemble,
)

# Entity Extractor (UPGRADE-010 Sprint 3 Expansion - December 2025)
from .entity_extractor import (
    EntityExtractor,
    EntityExtractorConfig,
    EntityType,
    ExtractedEntity,
    ExtractionResult,
    create_entity_extractor,
)

# Dual LLM Router (UPGRADE-010 Sprint 2 - December 2025)
from .model_router import (
    CostReport,
    CostTracker,
    LLMRouter,
    ModelConfig,
    ModelTier,
    RouteDecision,
    TaskClassifier,
    TaskType,
    UsageRecord,
    create_router,
)

# News Alert Manager (UPGRADE-014 - December 2025)
from .news_alert_manager import (
    NewsAlertConfig,
    NewsAlertManager,
    NewsEvent,
    NewsEventType,
    NewsImpact,
    create_news_alert_manager,
)
from .news_analyzer import NewsAlert, NewsAnalyzer, create_news_analyzer

# News Processor (UPGRADE-010 Sprint 3 Expansion - December 2025)
from .news_processor import (
    NewsEventType,
    NewsProcessor,
    NewsProcessorConfig,
    NewsUrgency,
    ProcessedNewsEvent,
    create_news_processor,
)

# PPO Weight Optimizer (UPGRADE-014 Expansion - Feature 8 - December 2025)
from .ppo_weight_optimizer import (
    Experience,
    ExperienceBuffer,
    PPOConfig,
    PPOWeightOptimizer,
    RewardType,
    SimpleNeuralNetwork,
    TradeOutcome,
    TradingRewardCalculator,
    ValueNetwork,
    WeightState,
    create_adaptive_weight_optimizer,
    create_ppo_optimizer,
)

# Prompt Optimizer (UPGRADE-005 - December 2025)
from .prompt_optimizer import (
    OptimizationResult,
    PromptOptimizer,
    PromptRefinement,
    RefinementCategory,
    RefinementStrategy,
    create_prompt_optimizer,
    generate_optimization_report,
)
from .providers import AnthropicProvider, OpenAIProvider, create_provider

# Reasoning Logger (UPGRADE-010 Sprint 1 - December 2025)
from .reasoning_logger import (
    ChainStatus,
    ReasoningChain,
    ReasoningLogger,
    SearchResult,
    create_reasoning_logger,
)

# Reddit Sentiment (UPGRADE-010 Sprint 3 - December 2025)
from .reddit_sentiment import (
    ContentSentiment,
    RedditSentimentAnalyzer,
    RedditSentimentConfig,
    RedditSentimentSignal,
    RedditSentimentSummary,
    TickerSentiment,
    create_reddit_sentiment_analyzer,
)

# Model Retraining Pipeline (UPGRADE-010 Sprint 2 - December 2025)
from .retraining import (
    DriftMonitor,
    DriftResult,
    DriftSeverity,
    ModelRecord,
    ModelStatus,
    ModelTrainer,
    RetrainingConfig,
    RetrainingPipeline,
    RetrainJob,
    RetrainStatus,
    ValidationResult,
    create_retraining_pipeline,
)

# Self-Evolving Agent (UPGRADE-005 - December 2025)
from .self_evolving_agent import (
    ConvergenceReason,
    EvolutionCycle,
    EvolutionResult,
    PromptVersion,
    SelfEvolvingAgent,
    create_self_evolving_agent,
    generate_evolution_report,
)
from .sentiment import (
    FinBERTSentimentAnalyzer,
    SimpleSentimentAnalyzer,
    create_sentiment_analyzer,
)

# Sentiment Filter (UPGRADE-014 - December 2025)
from .sentiment_filter import (
    DEFAULT_REGIME_CONFIGS,
    FilterDecision,
    FilterReason,
    FilterResult,
    # Regime-Adaptive Weighting (UPGRADE-014 Expansion - December 2025)
    MarketRegime,
    MomentumSignal,
    # Confidence-Based Position Sizing (UPGRADE-014 Expansion)
    PositionSizeResult,
    RegimeConfig,
    RegimeDetector,
    RegimeState,
    # Sentiment Momentum & Mean Reversion (UPGRADE-014 Expansion - Feature 7)
    SentimentExtreme,
    SentimentFilter,
    SentimentMomentumTracker,
    SentimentSignal,
    # Soft Voting Ensemble (UPGRADE-014 Expansion)
    VotingResult,
    calculate_confidence_position_size,
    create_momentum_tracker,
    create_sentiment_filter,
    create_signal_from_ensemble,
    hard_vote_ensemble,
    logit_to_score,
    soft_vote_ensemble,
    weighted_soft_vote_ensemble,
)

# Signal Aggregator (UPGRADE-010 Sprint 3 Expansion - December 2025)
from .signal_aggregator import (
    AggregatedAction,
    AggregatedSignal,
    SignalAggregator,
    SignalAggregatorConfig,
    SignalDirection,
    SignalSource,
    SourceSignal,
    create_signal_aggregator,
)


__all__ = [
    # Base classes
    "Sentiment",
    "SentimentResult",
    "NewsItem",
    "AnalysisResult",
    "BaseLLMProvider",
    "BaseSentimentAnalyzer",
    # Sentiment analyzers
    "FinBERTSentimentAnalyzer",
    "SimpleSentimentAnalyzer",
    "create_sentiment_analyzer",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "create_provider",
    # Ensemble
    "LLMEnsemble",
    "EnsembleResult",
    "ProviderPerformance",
    "DynamicWeightConfig",
    "create_ensemble",
    # News analyzer
    "NewsAnalyzer",
    "NewsAlert",
    "create_news_analyzer",
    # Self-Evolving Agent (UPGRADE-005 - December 2025)
    "SelfEvolvingAgent",
    "EvolutionCycle",
    "EvolutionResult",
    "PromptVersion",
    "ConvergenceReason",
    "create_self_evolving_agent",
    "generate_evolution_report",
    # Prompt Optimizer (UPGRADE-005 - December 2025)
    "PromptOptimizer",
    "PromptRefinement",
    "OptimizationResult",
    "RefinementCategory",
    "RefinementStrategy",
    "create_prompt_optimizer",
    "generate_optimization_report",
    # Sentiment Filter (UPGRADE-014 - December 2025)
    "SentimentFilter",
    "SentimentSignal",
    "FilterResult",
    "FilterDecision",
    "FilterReason",
    "create_sentiment_filter",
    "create_signal_from_ensemble",
    # Regime-Adaptive Weighting (UPGRADE-014 Expansion - December 2025)
    "MarketRegime",
    "RegimeConfig",
    "RegimeState",
    "RegimeDetector",
    "DEFAULT_REGIME_CONFIGS",
    # Confidence-Based Position Sizing (UPGRADE-014 Expansion)
    "PositionSizeResult",
    "calculate_confidence_position_size",
    "logit_to_score",
    # Soft Voting Ensemble (UPGRADE-014 Expansion)
    "VotingResult",
    "soft_vote_ensemble",
    "hard_vote_ensemble",
    "weighted_soft_vote_ensemble",
    # Sentiment Momentum & Mean Reversion (UPGRADE-014 Expansion - Feature 7)
    "SentimentExtreme",
    "MomentumSignal",
    "SentimentMomentumTracker",
    "create_momentum_tracker",
    # News Alert Manager (UPGRADE-014 - December 2025)
    "NewsAlertManager",
    "NewsEvent",
    "NewsImpact",
    "NewsEventType",
    "NewsAlertConfig",
    "create_news_alert_manager",
    # LLM Guardrails (UPGRADE-014 - December 2025)
    "LLMGuardrails",
    "TradingConstraints",
    "GuardrailCheckResult",
    "GuardrailViolation",
    "HallucinationDetector",  # UPGRADE-014 Expansion
    "create_llm_guardrails",
    # PPO Weight Optimizer (UPGRADE-014 Expansion - Feature 8 - December 2025)
    "PPOWeightOptimizer",
    "PPOConfig",
    "WeightState",
    "Experience",
    "TradeOutcome",
    "RewardType",
    "ExperienceBuffer",
    "TradingRewardCalculator",
    "SimpleNeuralNetwork",
    "ValueNetwork",
    "create_ppo_optimizer",
    "create_adaptive_weight_optimizer",
    # Reasoning Logger (UPGRADE-010 Sprint 1 - December 2025)
    "ReasoningChain",
    "ReasoningLogger",
    "ChainStatus",
    "SearchResult",
    "create_reasoning_logger",
    # Dual LLM Router (UPGRADE-010 Sprint 2 - December 2025)
    "TaskType",
    "ModelTier",
    "ModelConfig",
    "RouteDecision",
    "UsageRecord",
    "CostReport",
    "TaskClassifier",
    "CostTracker",
    "LLMRouter",
    "create_router",
    # Model Retraining Pipeline (UPGRADE-010 Sprint 2 - December 2025)
    "DriftSeverity",
    "RetrainStatus",
    "ModelStatus",
    "DriftResult",
    "RetrainJob",
    "ValidationResult",
    "ModelRecord",
    "DriftMonitor",
    "ModelTrainer",
    "RetrainingConfig",
    "RetrainingPipeline",
    "create_retraining_pipeline",
    # Reddit Sentiment (UPGRADE-010 Sprint 3 - December 2025)
    "RedditSentimentSignal",
    "RedditSentimentConfig",
    "ContentSentiment",
    "TickerSentiment",
    "RedditSentimentSummary",
    "RedditSentimentAnalyzer",
    "create_reddit_sentiment_analyzer",
    # Bot Detector (UPGRADE-010 Sprint 3 - December 2025)
    "BotIndicator",
    "BotConfidence",
    "BotDetectorConfig",
    "BotIndicatorResult",
    "BotDetectionResult",
    "CoordinatedCampaign",
    "BotDetector",
    "create_bot_detector",
    # Entity Extractor (UPGRADE-010 Sprint 3 Expansion - December 2025)
    "EntityType",
    "ExtractedEntity",
    "ExtractionResult",
    "EntityExtractorConfig",
    "EntityExtractor",
    "create_entity_extractor",
    # News Processor (UPGRADE-010 Sprint 3 Expansion - December 2025)
    "NewsEventType",
    "NewsUrgency",
    "ProcessedNewsEvent",
    "NewsProcessorConfig",
    "NewsProcessor",
    "create_news_processor",
    # Emotion Detector (UPGRADE-010 Sprint 3 Expansion - December 2025)
    "MarketEmotion",
    "EmotionIndicator",
    "EmotionResult",
    "EmotionDetectorConfig",
    "EmotionDetector",
    "create_emotion_detector",
    # Earnings Analyzer (UPGRADE-010 Sprint 3 Expansion - December 2025)
    "TranscriptSection",
    "ToneCategory",
    "RedFlagType",
    "SectionAnalysis",
    "RedFlag",
    "EarningsCallResult",
    "EarningsAnalyzerConfig",
    "EarningsAnalyzer",
    "create_earnings_analyzer",
    # Signal Aggregator (UPGRADE-010 Sprint 3 Expansion - December 2025)
    "SignalSource",
    "SignalDirection",
    "AggregatedAction",
    "SourceSignal",
    "AggregatedSignal",
    "SignalAggregatorConfig",
    "SignalAggregator",
    "create_signal_aggregator",
]
