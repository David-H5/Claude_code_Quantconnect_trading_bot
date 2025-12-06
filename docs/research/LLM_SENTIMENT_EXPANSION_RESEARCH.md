---
title: "LLM Sentiment Expansion Research"
topic: llm
related_upgrades: [UPGRADE-014]
related_docs: []
tags: [llm, agents]
created: 2025-12-01
updated: 2025-12-02
---

# UPGRADE-014 LLM Sentiment Integration Expansion Research

## 📋 Research Overview

**Research Date**: December 1, 2025
**Scope**: Advanced LLM sentiment analysis techniques for algorithmic trading
**Focus**: Multi-model ensembles, confidence-based position sizing, hallucination guardrails, adaptive weighting
**Result**: Comprehensive expansion plan with 8 new features identified

---

## 🎯 Research Objectives

1. Identify state-of-the-art LLM sentiment techniques for trading (2024-2025)
2. Research multi-agent trading frameworks and architectures
3. Evaluate real-time sentiment APIs and data sources
4. Investigate hallucination detection and trading guardrails
5. Explore adaptive sentiment weighting based on market regimes
6. Study confidence-based position sizing strategies
7. Assess QuantConnect-specific integrations (Tiingo, Brain Sentiment)

---

## 📊 Research Phases

### Phase 1: LLM Sentiment Trading Strategies 2025

**Search Date**: December 1, 2025 at 10:15 AM EST
**Search Queries**:
- "LLM sentiment analysis trading strategies 2025"
- "FinBERT GPT-4 sentiment trading comparison"

**Key Sources**:

1. [FinDPO: Financial Sentiment Analysis for Algorithmic Trading (Published: 2025)](https://arxiv.org/abs/2507.18417)
2. [How Sentiment Indicators Improve Algorithmic Trading Performance (Published: 2025)](https://journals.sagepub.com/doi/full/10.1177/21582440251369559)
3. [Real-time Gold Trading Using Sentiment and Time-Series Forecasting (Published: 2025)](https://www.sciencedirect.com/science/article/pii/S277266222500089X)
4. [QuantStart Sentiment Analysis Trading Strategy (Updated: ~2024)](https://www.quantstart.com/articles/sentiment-analysis-trading-strategy-via-sentdex-data-in-qstrader/)

**Key Discoveries**:

| Finding | Source | Impact |
|---------|--------|--------|
| FinDPO achieves 67% annual returns with Sharpe 2.0 | arXiv 2507.18417 | High - New SOTA approach |
| FinDPO outperforms fine-tuned models by 11% average | arXiv 2507.18417 | High - Consider implementation |
| Novel 'logit-to-score' conversion for continuous sentiment | arXiv 2507.18417 | High - Enables better position sizing |
| Sentiment-augmented strategies reduce volatility | SAGE Journal 2025 | Medium - Validates current approach |
| CNN Fear & Greed Index integration improves returns | SAGE Journal 2025 | Medium - Additional signal source |

**Applied**: Informs confidence-to-position-size conversion logic

---

### Phase 2: Multi-Agent Trading Systems

**Search Date**: December 1, 2025 at 10:20 AM EST
**Search Queries**:
- "multi-agent LLM trading systems 2025"
- "TradingAgents AI framework"

**Key Sources**:

1. [TradingAgents: Multi-Agent LLM Financial Trading Framework (Published: Dec 2024)](https://arxiv.org/abs/2412.20138)
2. [Survey of Financial AI: Architectures and Advances (Published: Nov 2024)](https://arxiv.org/html/2411.12747v1)

**Key Discoveries**:

| Finding | Source | Impact |
|---------|--------|--------|
| TradingAgents uses 7 distinct specialized roles | arXiv 2412.20138 | High - Architecture pattern |
| Roles: Fundamentals, Sentiment, News, Technical, Researcher, Trader, Risk Manager | arXiv 2412.20138 | High - Role specialization |
| Multi-agent debate/discussion improves decision quality | arXiv 2412.20138 | Medium - Ensemble enhancement |
| Agent collaboration reduces individual LLM hallucinations | arXiv 2412.20138 | High - Safety improvement |

**TradingAgents Architecture (7 Roles)**:
```
┌─────────────────────────────────────────────────────────────┐
│                    TRADING AGENTS FRAMEWORK                  │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ Fundamentals  │  │   Sentiment   │  │     News      │   │
│  │    Analyst    │  │    Analyst    │  │    Analyst    │   │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘   │
│          │                  │                  │            │
│  ┌───────┴───────┐  ┌───────┴───────┐                      │
│  │   Technical   │  │   Research    │                      │
│  │    Analyst    │  │    Analyst    │                      │
│  └───────┬───────┘  └───────┬───────┘                      │
│          │                  │                              │
│          └────────┬─────────┘                              │
│                   ▼                                         │
│          ┌───────────────┐                                 │
│          │    Trader     │◄──── Trading Decisions          │
│          └───────┬───────┘                                 │
│                  │                                          │
│                  ▼                                          │
│          ┌───────────────┐                                 │
│          │ Risk Manager  │◄──── Position Sizing & Limits   │
│          └───────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

**Applied**: Informs multi-agent sentiment architecture expansion

---

### Phase 3: Real-Time Sentiment APIs

**Search Date**: December 1, 2025 at 10:25 AM EST
**Search Queries**:
- "real-time news sentiment API trading 2025"
- "financial sentiment data providers API"

**Key Sources**:

1. [Alpha Vantage API Documentation (Updated: 2025)](https://www.alphavantage.co/)
2. [Finnhub API - Real-Time Stock APIs (Updated: 2025)](https://finnhub.io/)
3. [EODHD Financial Data APIs (Updated: 2025)](https://eodhd.com/)
4. [Polygon.io Market Data Platform (Updated: 2025)](https://polygon.io/)
5. [QuantConnect Brain Sentiment Indicator (Updated: ~2024)](https://www.quantconnect.com/datasets/brain-sentiment-indicator)

**Key Discoveries**:

| Provider | Sentiment Features | Cost | Real-Time | Notes |
|----------|-------------------|------|-----------|-------|
| Alpha Vantage | News sentiment, ticker scores | Free tier available | Yes | Good for prototyping |
| Finnhub | Company news, sentiment scores | Free tier + paid | Yes | Comprehensive coverage |
| EODHD | Financial news sentiment | Paid | Yes | Historical depth |
| Polygon.io | News with sentiment | Paid | Yes | Low latency |
| Brain (via QC) | Institutional sentiment | QC subscription | Yes | Integrated with LEAN |
| Tiingo (via QC) | News feed, NLP ready | QC subscription | Yes | Best for QuantConnect |

**Applied**: Tiingo integration via QuantConnect identified as primary path

---

### Phase 4: Hallucination Detection & Trading Guardrails

**Search Date**: December 1, 2025 at 10:30 AM EST
**Search Queries**:
- "LLM hallucination detection trading guardrails financial AI"
- "financial AI safety guardrails 2025"

**Key Sources**:

1. [Dissecting the Ledger: Locating Liar Circuits in Financial LLMs (Published: Nov 2025)](https://arxiv.org/html/2511.21756)
2. [Guardrails for LLMs in Banking (Published: ~2024)](https://www.getdynamiq.ai/post/guardrails-for-llms-in-banking-essential-measures-for-secure-ai-use)
3. [Detecting & Addressing LLM Hallucinations in Finance (Published: ~2024)](https://www.packtpub.com/qa-fr/learning/how-to-tutorials/detecting-addressing-llm-hallucinations-in-finance)
4. [LLM Hallucinations: Implications for Financial Institutions (Published: Aug 2025)](https://biztechmagazine.com/article/2025/08/llm-hallucinations-what-are-implications-financial-institutions)
5. [Guardrails AI Provenance Validators (Updated: 2025)](https://www.guardrailsai.com/blog/reduce-ai-hallucinations-provenance-guardrails)

**Key Discoveries**:

| Finding | Source | Impact |
|---------|--------|--------|
| LLMs hallucinate 3-27% of the time in financial contexts | BizTech 2025 | Critical - Must detect |
| Multi-model consensus greatly reduces hallucination risk | Guardrails AI | High - Ensemble validation |
| Mechanistic "Liar Circuits" identified at Layer 46 in GPT-2 XL | arXiv 2511.21756 | Research - Future detection |
| 98% accuracy detecting arithmetic hallucinations via probing | arXiv 2511.21756 | High - Potential integration |
| RAG grounding reduces hallucinations significantly | Multiple sources | High - Already have partial |

**Hallucination Mitigation Strategies**:
```python
class HallucinationGuardrails:
    """Multi-layer hallucination detection for trading decisions."""

    strategies = {
        "multi_model_consensus": {
            "description": "Require agreement across multiple LLMs",
            "effectiveness": "High - different models unlikely to hallucinate identically",
            "implementation": "Compare outputs from FinBERT, GPT-4, Claude"
        },
        "source_verification": {
            "description": "Validate claims against authoritative sources",
            "effectiveness": "High - catches factual errors",
            "implementation": "Cross-reference with SEC filings, news APIs"
        },
        "confidence_thresholding": {
            "description": "Reject low-confidence predictions",
            "effectiveness": "Medium - some hallucinations have high confidence",
            "implementation": "Require confidence > 0.7 for trading signals"
        },
        "provenance_tracking": {
            "description": "Ensure outputs derive from provided sources",
            "effectiveness": "High - prevents fabricated information",
            "implementation": "Use RAG with citation requirements"
        },
        "numerical_validation": {
            "description": "Verify arithmetic and numerical claims",
            "effectiveness": "High - catches calculation errors",
            "implementation": "Independent calculation verification"
        }
    }
```

**Applied**: Informs LLM Guardrails enhancement for trading safety

---

### Phase 5: Multi-Model Ensemble Voting

**Search Date**: December 1, 2025 at 10:35 AM EST
**Search Queries**:
- "multi-model ensemble sentiment voting finance 2024 2025"
- "LLM ensemble trading predictions"

**Key Sources**:

1. [Multi-Model Ensemble-HMM Voting Framework for Market Regime Detection (Published: 2025)](https://www.aimspress.com/article/doi/10.3934/DSFE.2025019?viewType=HTML)
2. [Soft Voting Ensemble Learning for Multimodal Sentiment (Published: 2022)](https://link.springer.com/article/10.1007/s00521-022-07451-7)
3. [LLM Uncertainty and Variability in Sentiment Analysis (Published: 2025)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1609097/full)
4. [Generating Effective Ensembles for Sentiment Analysis (Published: Feb 2024)](https://arxiv.org/html/2402.16700v1)
5. [Financial Sentiment Analysis: Techniques and Applications (Published: 2024)](https://dl.acm.org/doi/10.1145/3649451)

**Key Discoveries**:

| Finding | Source | Impact |
|---------|--------|--------|
| Ensemble-HMM hybrid improves regime classification robustness | AIMS 2025 | High - Market regime awareness |
| RoBERTa + LSTM + BiLSTM ensemble shows strong generalization | Springer 2022 | Medium - Architecture pattern |
| LLMs unstable in finance - lexicon models more stable | Frontiers 2025 | High - Hybrid approach needed |
| Ensemble averaging reduces variability significantly | Frontiers 2025 | High - Confirms current approach |
| Soft voting outperforms hard voting in complex domains | Multiple sources | Medium - Implementation detail |

**Ensemble Voting Strategies**:
```python
class EnsembleVotingStrategies:
    """Voting strategies for multi-model sentiment ensemble."""

    STRATEGIES = {
        "hard_voting": {
            "description": "Majority vote on discrete labels",
            "formula": "mode([model1_label, model2_label, ...])",
            "best_for": "Clear sentiment signals",
            "weakness": "Loses confidence information"
        },
        "soft_voting": {
            "description": "Average probabilities then classify",
            "formula": "argmax(mean([model1_probs, model2_probs, ...]))",
            "best_for": "Nuanced sentiment, uncertain cases",
            "strength": "Preserves confidence information"
        },
        "weighted_soft_voting": {
            "description": "Weighted average based on model performance",
            "formula": "argmax(sum(weight_i * model_i_probs))",
            "best_for": "When model accuracies differ significantly",
            "strength": "Leverages best-performing models"
        },
        "confidence_weighted": {
            "description": "Weight by each model's self-reported confidence",
            "formula": "sum(confidence_i * prediction_i) / sum(confidence_i)",
            "best_for": "Models with calibrated confidence",
            "strength": "Adaptive per-prediction weighting"
        },
        "regime_adaptive": {
            "description": "Adjust weights based on market regime",
            "formula": "weights = f(volatility, trend, regime_state)",
            "best_for": "Varying market conditions",
            "strength": "Adapts to market environment"
        }
    }
```

**Applied**: Informs enhanced ensemble voting implementation

---

### Phase 6: Confidence-Based Position Sizing

**Search Date**: December 1, 2025 at 10:40 AM EST
**Search Queries**:
- "sentiment confidence position sizing algorithmic trading 2025"
- "AI confidence position sizing trading"

**Key Sources**:

1. [FinDPO: Logit-to-Score Conversion for Position Sizing (Published: 2025)](https://arxiv.org/abs/2507.18417)
2. [CMC Markets: Market Sentiment Analysis (Updated: 2025)](https://www.cmcmarkets.com/en/technical-analysis/market-sentiment-analysis)
3. [QuantStart: Sentiment Analysis Trading Strategy (Updated: ~2024)](https://www.quantstart.com/articles/sentiment-analysis-trading-strategy-via-sentdex-data-in-qstrader/)

**Key Discoveries**:

| Finding | Source | Impact |
|---------|--------|--------|
| Logit-to-score conversion enables continuous ranking | FinDPO 2025 | High - Better position sizing |
| Scale into positions gradually based on confidence | CMC Markets | High - Risk management |
| Larger deviations warrant larger positions (mean reversion) | CMC Markets | Medium - Strategy specific |
| Integration should be gradual with small sizes first | Multiple | Medium - Best practice |
| AI platforms rank opportunities by confidence level | Multiple | High - Already implemented |

**Position Sizing Formula**:
```python
def confidence_adjusted_position_size(
    base_size: float,
    sentiment_confidence: float,
    ensemble_agreement: float,
    volatility_regime: str = "normal"
) -> float:
    """
    Calculate position size adjusted for sentiment confidence.

    Args:
        base_size: Base position size (e.g., 0.02 = 2% of portfolio)
        sentiment_confidence: Model confidence [0, 1]
        ensemble_agreement: Agreement between models [0, 1]
        volatility_regime: Market volatility state

    Returns:
        Adjusted position size
    """
    # Confidence multiplier (0.5x to 1.5x)
    confidence_mult = 0.5 + sentiment_confidence

    # Agreement multiplier (0.5x to 1.5x)
    agreement_mult = 0.5 + ensemble_agreement

    # Volatility adjustment
    vol_mult = {
        "low": 1.2,      # Increase size in calm markets
        "normal": 1.0,   # Standard size
        "high": 0.6,     # Reduce size in volatile markets
        "extreme": 0.3   # Minimal size in extreme volatility
    }.get(volatility_regime, 1.0)

    # Final position size with caps
    adjusted = base_size * confidence_mult * agreement_mult * vol_mult

    # Cap at 2x base size maximum
    return min(adjusted, base_size * 2.0)
```

**Applied**: Informs confidence-weighted position sizing feature

---

### Phase 7: Adaptive Sentiment Weighting & Market Regimes

**Search Date**: December 1, 2025 at 10:45 AM EST
**Search Queries**:
- "adaptive sentiment weighting market volatility regime trading"
- "market regime sentiment trading strategy"

**Key Sources**:

1. [S&P 500 Volatility Forecasting with Regime-Switching (Published: Oct 2025)](https://arxiv.org/html/2510.03236v1)
2. [AI-Driven Real-Time Algorithmic Trading System (Published: ~2024)](https://anjikeesari.com/resources/publications/algo-trading/algo-trading/)
3. [Adaptive Alpha Weighting with PPO (Published: Sep 2025)](https://arxiv.org/html/2509.01393v1)
4. [Sentiment Analysis for Market Volatility Prediction (Published: 2022)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2022.836809/full)
5. [Enhancing Trading Performance Through Sentiment Analysis (Published: 2025)](https://arxiv.org/html/2507.09739v1)

**Key Discoveries**:

| Finding | Source | Impact |
|---------|--------|--------|
| Dual-memory HAR model captures volatility patterns + sentiment | arXiv Oct 2025 | High - Architecture pattern |
| HAR-style lags on VIX improve regime detection | arXiv Oct 2025 | High - Feature engineering |
| PPO-optimized alpha weighting outperforms equal-weighted | arXiv Sep 2025 | High - Dynamic weighting |
| Sentiment as filter not predictor works better in volatile markets | Multiple 2025 | High - Confirms our approach |
| Combined sentiment + technical beats either alone | arXiv Jul 2025 | High - Validates hybrid |
| 67% regime detection accuracy with adaptive weighting | Anjikeesari 2024 | Medium - Benchmark |

**Regime-Adaptive Sentiment Configuration**:
```python
class RegimeAdaptiveSentiment:
    """Adjust sentiment weights based on market regime."""

    REGIME_CONFIGS = {
        "bull_trending": {
            "sentiment_weight": 0.7,      # High weight on sentiment
            "technical_weight": 0.3,      # Lower technical weight
            "min_confidence": 0.5,        # Lower threshold
            "position_mult": 1.2,         # Slightly larger positions
            "notes": "Sentiment drives momentum in trends"
        },
        "bear_trending": {
            "sentiment_weight": 0.6,      # Moderate sentiment weight
            "technical_weight": 0.4,      # Higher technical weight
            "min_confidence": 0.6,        # Higher threshold
            "position_mult": 0.8,         # Smaller positions
            "notes": "Be more cautious, use technicals for timing"
        },
        "high_volatility": {
            "sentiment_weight": 0.4,      # Lower sentiment weight
            "technical_weight": 0.6,      # Higher technical weight
            "min_confidence": 0.75,       # Very high threshold
            "position_mult": 0.5,         # Much smaller positions
            "notes": "Sentiment less reliable in chaos"
        },
        "low_volatility": {
            "sentiment_weight": 0.5,      # Balanced weights
            "technical_weight": 0.5,      # Balanced weights
            "min_confidence": 0.55,       # Moderate threshold
            "position_mult": 1.0,         # Standard positions
            "notes": "Standard operation in calm markets"
        },
        "mean_reverting": {
            "sentiment_weight": 0.3,      # Lower sentiment weight
            "technical_weight": 0.7,      # Higher technical weight
            "min_confidence": 0.65,       # Higher threshold
            "position_mult": 0.9,         # Slightly smaller
            "notes": "Technical levels more important"
        }
    }
```

**Applied**: Informs regime-adaptive sentiment weighting system

---

### Phase 8: QuantConnect Tiingo Integration

**Search Date**: December 1, 2025 at 10:50 AM EST
**Search Queries**:
- "QuantConnect Tiingo news data sentiment integration Python"
- "QuantConnect sentiment analysis bootcamp"

**Key Sources**:

1. [QuantConnect Tiingo News Feed Documentation (Updated: ~2024)](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/tiingo/tiingo-news-feed)
2. [QuantConnect Tiingo News Feed Dataset (Updated: ~2024)](https://www.quantconnect.com/data/tiingo-news-feed)
3. [Tiingo NLP Sentiment Competition Example (Published: ~2020)](https://www.quantconnect.com/forum/discussion/6695/competition-algorithm-example-tiingo-nlp-sentiment/)
4. [Tiingo Sentiment Mean Strategy (Published: ~2021)](https://www.quantconnect.com/forum/discussion/8743/using-tiingo-news-to-create-a-sentiment-mean-strategy/)
5. [QuantConnect Bootcamp Tiingo Example (Published: ~2023)](https://github.com/ruthrootz/quantconnect-bootcamp/blob/main/1-11-tiingo-sentiment-analysis-on-stocks.py)

**Key Discoveries**:

| Finding | Source | Impact |
|---------|--------|--------|
| Tiingo covers 10,000+ US Equities since Jan 2014 | QC Docs | High - Comprehensive coverage |
| Second-frequency news delivery | QC Docs | High - Real-time capable |
| 120+ news providers integrated | QC Docs | High - Broad coverage |
| Key properties: Title, Description, Tags, Source, Symbols | QC Docs | High - Rich data |
| Linked data - auto-tied to underlying equity | QC Docs | High - Easy integration |
| Free for Alpha Streams, paid API for live trading | QC Docs | Medium - Cost consideration |

**Tiingo Integration Pattern**:
```python
from AlgorithmImports import *
from QuantConnect.Data.Custom.Tiingo import TiingoNews

class TiingoSentimentIntegration(QCAlgorithm):
    """Pattern for integrating Tiingo news with LLM sentiment."""

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)

        # Add equity
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol

        # Add Tiingo news (linked to equity)
        self.AddData(TiingoNews, self.spy)

        # Sentiment word dictionary
        self.sentiment_words = {
            "bullish": 1.0, "upgrade": 0.8, "beat": 0.6,
            "growth": 0.5, "strong": 0.4, "positive": 0.3,
            "bearish": -1.0, "downgrade": -0.8, "miss": -0.6,
            "decline": -0.5, "weak": -0.4, "negative": -0.3
        }

        # News cache for LLM analysis
        self.news_cache = {}

    def OnData(self, data: Slice):
        # Process Tiingo news
        if data.ContainsKey(TiingoNews):
            for news in data.Get(TiingoNews).Values:
                self._process_news(news)

    def _process_news(self, news: TiingoNews):
        """Process news for sentiment analysis."""
        # Quick word-based sentiment
        text = f"{news.Title} {news.Description}".lower()
        quick_score = sum(
            score for word, score in self.sentiment_words.items()
            if word in text
        )

        # Cache for LLM analysis
        for symbol in news.Symbols:
            if symbol not in self.news_cache:
                self.news_cache[symbol] = []
            self.news_cache[symbol].append({
                "title": news.Title,
                "description": news.Description,
                "tags": list(news.Tags),
                "source": news.Source,
                "time": news.Time,
                "quick_sentiment": quick_score
            })

        # Log significant news
        if abs(quick_score) > 0.5:
            self.Debug(f"News [{news.Source}]: {news.Title} (score: {quick_score:.2f})")
```

**Applied**: Direct implementation pattern for Tiingo integration

---

## 🔑 Critical Discoveries Summary

### Highest Impact Findings

| # | Discovery | Source | Implementation Priority |
|---|-----------|--------|------------------------|
| 1 | FinDPO logit-to-score for continuous sentiment ranking | arXiv 2507.18417 | P1 - High |
| 2 | Multi-model consensus reduces hallucinations significantly | Multiple 2025 | P1 - High |
| 3 | TradingAgents 7-role architecture pattern | arXiv 2412.20138 | P2 - Medium |
| 4 | Regime-adaptive sentiment weighting improves performance | arXiv Oct 2025 | P1 - High |
| 5 | Sentiment as filter (not predictor) works best in volatile markets | Multiple 2025 | P1 - High |
| 6 | LLMs hallucinate 3-27% in finance - must detect | BizTech 2025 | P1 - Critical |
| 7 | Tiingo provides second-frequency news for 10,000+ equities | QC Docs | P1 - High |
| 8 | PPO-optimized alpha weighting outperforms static weights | arXiv Sep 2025 | P3 - Low |

### Research Validation

Our current UPGRADE-014 implementation aligns well with 2025 research:

| Current Feature | Research Validation |
|-----------------|---------------------|
| Multi-model ensemble (FinBERT + GPT-4 + Claude) | ✅ Confirmed as best practice |
| Sentiment filter (not predictor) | ✅ Validated for volatile markets |
| Confidence-based filtering | ✅ Supports position sizing |
| Time-weighted decay | ✅ Aligns with regime-switching models |
| LLM Guardrails | ✅ Critical given 3-27% hallucination rate |

---

## 📈 Expansion Plan: 8 New Features

Based on research findings, the following features are recommended for UPGRADE-014 expansion:

### Feature 1: Regime-Adaptive Sentiment Weighting (P1)
**Source**: arXiv Oct 2025, Multiple
**Implementation**:
- Detect market regime (bull/bear/volatile/calm)
- Adjust sentiment weight vs technical weight dynamically
- Adjust position sizing based on regime
- Adjust confidence thresholds based on regime

### Feature 2: Enhanced Hallucination Detection (P1)
**Source**: arXiv Nov 2025, BizTech 2025
**Implementation**:
- Multi-model consensus validation
- Source verification against news APIs
- Numerical claim validation
- Provenance tracking for citations
- Confidence calibration checks

### Feature 3: Tiingo Deep Integration (P1)
**Source**: QuantConnect Docs
**Implementation**:
- Real-time news processing from 120+ sources
- Automatic symbol linking
- News caching for LLM analysis
- Quick sentiment scoring + LLM deep analysis
- Historical sentiment patterns

### Feature 4: Confidence-to-Position Sizing (P1)
**Source**: FinDPO 2025, CMC Markets
**Implementation**:
- Logit-to-score conversion for continuous ranking
- Confidence multiplier (0.5x to 1.5x)
- Agreement multiplier (0.5x to 1.5x)
- Volatility regime adjustment
- Maximum position caps

### Feature 5: Soft Voting Ensemble (P1)
**Source**: Frontiers 2025, Springer 2022
**Implementation**:
- Average probabilities across models
- Weighted soft voting based on model performance
- Confidence-weighted voting option
- Regime-adaptive weight adjustment

### Feature 6: Multi-Agent Architecture (P2)
**Source**: TradingAgents (arXiv Dec 2024)
**Implementation**:
- Specialized sentiment analyst agent
- News analyst agent
- Technical analyst agent
- Risk management agent
- Agent collaboration/debate mechanism

### Feature 7: Sentiment Momentum & Mean Reversion (P2)
**Source**: QuantConnect Forum, SAGE 2025
**Implementation**:
- Track sentiment momentum (rate of change)
- Detect sentiment extremes for mean reversion
- Sentiment divergence from price signals
- Cross-symbol sentiment correlation

### Feature 8: PPO-Optimized Weighting (P3)
**Source**: arXiv Sep 2025
**Implementation**:
- Reinforcement learning for weight optimization
- Dynamic alpha weighting based on performance
- Continuous learning from trading outcomes
- Paper trading validation required

---

## ✅ Implementation Status

**Implementation Date**: December 1-2, 2025
**Status**: 7/8 Features Complete (Feature 3 requires QC subscription)

### Implemented Features

| Feature | Status | Location | Tests |
|---------|--------|----------|-------|
| Feature 1: Regime-Adaptive Sentiment Weighting | ✅ Complete | `llm/sentiment_filter.py` | ✅ 8 tests |
| Feature 2: Enhanced Hallucination Detection | ✅ Complete | `llm/agents/llm_guardrails.py` | ✅ 6 tests |
| Feature 4: Confidence-to-Position Sizing | ✅ Complete | `llm/sentiment_filter.py` | ✅ 4 tests |
| Feature 5: Soft Voting Ensemble | ✅ Complete | `llm/sentiment_filter.py` | ✅ 5 tests |
| Feature 6: Multi-Agent Architecture | ✅ Complete | `llm/agents/` | ✅ 50 tests |
| Feature 7: Sentiment Momentum & Mean Reversion | ✅ Complete | `llm/sentiment_filter.py` | ✅ 20 tests |
| Feature 8: PPO-Optimized Weighting | ✅ Complete | `llm/ppo_weight_optimizer.py` | ✅ 59 tests |

### Pending Features

| Feature | Priority | Status | Notes |
|---------|----------|--------|-------|
| Feature 3: Tiingo Deep Integration | P1 | 🔜 Pending | Requires QC subscription |

### New Classes & Functions Added

```python
# llm/sentiment_filter.py - Regime Detection
class MarketRegime(Enum):
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MEAN_REVERTING = "mean_reverting"

@dataclass
class RegimeConfig:
    sentiment_weight: float
    technical_weight: float
    min_confidence: float
    position_multiplier: float

@dataclass
class RegimeState:
    regime: MarketRegime
    volatility_percentile: float
    trend_strength: float
    config: RegimeConfig

class RegimeDetector:
    def update(self, current_vol: float, daily_return: float) -> None
    def get_current_regime(self) -> RegimeState
    def get_sentiment_weight(self) -> float
    def get_position_multiplier(self) -> float

# llm/sentiment_filter.py - Position Sizing
@dataclass
class PositionSizeResult:
    adjusted_size: float
    confidence_mult: float
    agreement_mult: float
    regime_mult: float
    final_mult: float

def logit_to_score(logit: float) -> float
def calculate_confidence_position_size(...) -> PositionSizeResult

# llm/sentiment_filter.py - Voting Ensemble
@dataclass
class VotingResult:
    final_sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    vote_details: Dict[str, Any]
    method: str

def soft_vote_ensemble(...) -> VotingResult
def hard_vote_ensemble(...) -> VotingResult
def weighted_soft_vote_ensemble(...) -> VotingResult

# llm/agents/llm_guardrails.py - Hallucination Detection
class HallucinationDetector:
    def detect(
        self,
        output: str,
        context: Dict[str, Any],
        model_predictions: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, List[str], float]

# llm/sentiment_filter.py - Sentiment Momentum & Mean Reversion
class SentimentExtreme(Enum):
    EXTREME_BULLISH = "extreme_bullish"
    EXTREME_BEARISH = "extreme_bearish"
    NORMAL = "normal"

@dataclass
class MomentumSignal:
    symbol: str
    current_score: float
    momentum: float
    acceleration: float
    extreme: SentimentExtreme
    mean_reversion_signal: bool
    divergence_from_price: Optional[float]
    lookback_periods: int
    timestamp: datetime

class SentimentMomentumTracker:
    def update_sentiment(self, symbol: str, sentiment_score: float, timestamp: Optional[datetime] = None) -> None
    def update_price(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> None
    def get_momentum_signal(self, symbol: str) -> Optional[MomentumSignal]
    def get_all_momentum_signals(self) -> Dict[str, MomentumSignal]
    def get_extreme_symbols(self) -> Dict[str, SentimentExtreme]
    def get_mean_reversion_candidates(self) -> List[str]
    def get_divergence_signals(self, min_divergence: float = 0.3) -> Dict[str, float]
    def get_stats(self) -> Dict[str, Any]

def create_momentum_tracker(...) -> SentimentMomentumTracker

# llm/agents/news_analyst.py - News Analyst Agent (Feature 6)
class NewsEventType(Enum):
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    MERGER_ACQUISITION = "merger_acquisition"
    REGULATORY = "regulatory"
    # ... and more

class NewsImpactLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEUTRAL = "neutral"

class NewsTimeRelevance(Enum):
    BREAKING = "breaking"
    TODAY = "today"
    RECENT = "recent"
    OLD = "old"
    STALE = "stale"

@dataclass
class NewsAnalysis:
    headline: str
    event_type: NewsEventType
    impact_level: NewsImpactLevel
    time_relevance: NewsTimeRelevance
    sentiment_score: float
    sentiment_direction: str
    confidence: float
    key_entities: List[str]
    affected_symbols: List[str]
    source_reliability: float
    trading_implications: List[str]

class NewsAnalyst(TradingAgent):
    def analyze(self, query: str, context: Dict[str, Any]) -> AgentResponse

def create_news_analyst(...) -> NewsAnalyst
def create_safe_news_analyst(...) -> SafeAgentWrapper

# llm/agents/multi_agent_consensus.py - Multi-Agent Consensus (Feature 6)
class ConsensusSignal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    HOLD = "hold"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    CONFLICTED = "conflicted"

class AgentType(Enum):
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    NEWS = "news"
    RISK = "risk"
    FUNDAMENTAL = "fundamental"

@dataclass
class AgentOpinion:
    agent_type: AgentType
    agent_name: str
    signal_score: float
    confidence: float
    reasoning: str
    key_factors: List[str]
    risk_factors: List[str]

@dataclass
class ConsensusResult:
    symbol: str
    signal: ConsensusSignal
    consensus_score: float
    confidence: float
    agreement_score: float
    participating_agents: int
    agent_opinions: List[AgentOpinion]
    recommendation: str
    requires_debate: bool

class MultiAgentConsensus:
    def add_opinion(self, opinion: AgentOpinion) -> None
    def calculate_consensus(self, symbol: str) -> ConsensusResult
    def adjust_weights_for_regime(self, is_high_volatility: bool) -> None
    def get_statistics(self) -> Dict[str, Any]

def create_multi_agent_consensus(...) -> MultiAgentConsensus
def opinion_from_agent_response(response, agent_type) -> AgentOpinion

# llm/ppo_weight_optimizer.py - PPO Weight Optimization (Feature 8)
class RewardType(Enum):
    SHARPE = "sharpe"
    RETURNS = "returns"
    ACCURACY = "accuracy"
    COMBINED = "combined"

@dataclass
class WeightState:
    volatility_percentile: float
    trend_strength: float
    is_high_vol: bool
    is_trending: bool
    recent_sharpe: float
    recent_accuracy: float
    recent_returns: float
    current_weights: List[float]
    model_confidences: List[float]
    def to_vector(self) -> List[float]
    @classmethod
    def default(cls, num_models: int = 3) -> "WeightState"

@dataclass
class Experience:
    state: WeightState
    action: List[float]
    reward: float
    next_state: Optional[WeightState]
    done: bool
    log_prob: float
    value: float

@dataclass
class TradeOutcome:
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    predicted_direction: str
    actual_direction: Optional[str]
    pnl_pct: Optional[float]
    weights_used: List[float]
    model_predictions: List[Dict[str, Any]]

@dataclass
class PPOConfig:
    learning_rate: float = 0.001
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    batch_size: int = 32
    buffer_size: int = 1000
    min_weight: float = 0.05
    max_weight: float = 0.90
    reward_type: RewardType = RewardType.COMBINED

class SimpleNeuralNetwork:
    """Lightweight numpy-based feedforward neural network."""
    def forward(self, x: List[float]) -> List[float]
    def get_parameters(self) -> List[float]
    def set_parameters(self, params: List[float]) -> None

class ValueNetwork(SimpleNeuralNetwork):
    """Value function network for PPO."""
    def forward(self, x: List[float]) -> float

class ExperienceBuffer:
    """Experience replay buffer for PPO training."""
    def add(self, experience: Experience) -> None
    def sample(self, batch_size: int) -> List[Experience]
    def get_all(self) -> List[Experience]
    def clear(self) -> None

class TradingRewardCalculator:
    """Calculate rewards from trading outcomes."""
    def calculate_reward(self, outcome: TradeOutcome) -> float
    def reset(self) -> None

class PPOWeightOptimizer:
    """PPO-based weight optimizer for sentiment ensemble."""
    def get_optimal_weights(self, state: WeightState) -> List[float]
    def record_outcome(self, state, weights_used, outcome) -> float
    def train(self) -> Dict[str, float]
    def should_train(self) -> bool
    def get_statistics(self) -> Dict[str, Any]
    def save_state(self) -> Dict[str, Any]
    def load_state(self, state: Dict[str, Any]) -> None
    def reset(self) -> None

def create_ppo_optimizer(...) -> PPOWeightOptimizer
def create_adaptive_weight_optimizer(...) -> PPOWeightOptimizer
```

### Test Coverage

```text
tests/test_llm_sentiment.py
├── TestRegimeDetector (8 tests) ✅
│   ├── test_regime_detector_initialization
│   ├── test_detect_high_volatility_regime
│   ├── test_detect_low_volatility_regime
│   ├── test_detect_bull_trending_regime
│   ├── test_detect_bear_trending_regime
│   ├── test_regime_state_to_dict
│   ├── test_get_sentiment_weight
│   └── test_get_position_multiplier
├── TestConfidencePositionSizing (4 tests) ✅
│   ├── test_base_size_calculation
│   ├── test_high_confidence_increases_size
│   ├── test_low_confidence_decreases_size
│   └── test_max_size_capped
├── TestLogitToScore (2 tests) ✅
│   ├── test_logit_to_score_zero
│   └── test_logit_to_score_positive_negative
├── TestSoftVotingEnsemble (3 tests) ✅
│   ├── test_soft_vote_unanimous_bullish
│   ├── test_soft_vote_mixed_signals
│   └── test_soft_vote_with_weights
├── TestHardVotingEnsemble (2 tests) ✅
│   ├── test_hard_vote_majority_bullish
│   └── test_hard_vote_tie_defaults_neutral
├── TestHallucinationDetector (6 tests) ✅
│   ├── test_hallucination_detector_initialization
│   ├── test_detect_invalid_symbol
│   ├── test_detect_price_claim_discrepancy
│   ├── test_detect_arithmetic_error
│   ├── test_detect_model_consensus_mismatch
│   └── test_clean_output_passes
├── TestVotingResultSerialization (1 test) ✅
├── TestDefaultRegimeConfigs (1 test) ✅
├── TestSentimentMomentumTracker (15 tests) ✅
│   ├── test_tracker_initialization
│   ├── test_update_sentiment
│   ├── test_get_momentum_signal_insufficient_data
│   ├── test_get_momentum_signal_with_data
│   ├── test_momentum_calculation
│   ├── test_acceleration_calculation
│   ├── test_extreme_detection_bullish
│   ├── test_extreme_detection_bearish
│   ├── test_normal_sentiment
│   ├── test_mean_reversion_signal
│   ├── test_price_divergence
│   ├── test_get_extreme_symbols
│   ├── test_get_mean_reversion_candidates
│   ├── test_get_divergence_signals
│   └── test_get_stats
├── TestMomentumSignal (1 test) ✅
│   └── test_momentum_signal_dataclass
├── TestCreateMomentumTracker (2 tests) ✅
│   ├── test_create_momentum_tracker_default
│   └── test_create_momentum_tracker_custom
├── TestSentimentExtremeEnum (1 test) ✅
│   └── test_sentiment_extreme_values
└── TestMomentumImports (1 test) ✅
    └── test_momentum_imports

tests/test_multi_agent.py
├── TestNewsEventType (2 tests) ✅
├── TestNewsImpactLevel (1 test) ✅
├── TestNewsTimeRelevance (1 test) ✅
├── TestNewsAnalysis (2 tests) ✅
├── TestNewsAnalystResult (2 tests) ✅
├── TestNewsAnalyst (12 tests) ✅
│   ├── test_analyst_initialization
│   ├── test_analyst_with_custom_reliability
│   ├── test_analyze_basic
│   ├── test_analyze_with_news_articles
│   ├── test_classify_event_type_earnings
│   ├── test_classify_event_type_merger
│   ├── test_assess_impact_critical
│   ├── test_assess_impact_high
│   ├── test_calculate_sentiment_bullish
│   ├── test_calculate_sentiment_bearish
│   ├── test_source_reliability_known
│   └── test_source_reliability_unknown
├── TestCreateNewsAnalyst (2 tests) ✅
├── TestConsensusSignal (1 test) ✅
├── TestAgentType (1 test) ✅
├── TestAgentOpinion (3 tests) ✅
├── TestConsensusResult (2 tests) ✅
├── TestMultiAgentConsensus (11 tests) ✅
│   ├── test_consensus_initialization
│   ├── test_add_opinion
│   ├── test_add_multiple_opinions
│   ├── test_clear_opinions
│   ├── test_calculate_consensus_insufficient_data
│   ├── test_calculate_consensus_bullish
│   ├── test_calculate_consensus_bearish
│   ├── test_calculate_consensus_conflicted
│   ├── test_agreement_score_high_when_aligned
│   ├── test_set_custom_weights
│   └── test_adjust_weights_for_high_volatility
│   └── test_statistics
├── TestCreateMultiAgentConsensus (3 tests) ✅
├── TestOpinionFromAgentResponse (2 tests) ✅
├── TestMultiAgentIntegration (2 tests) ✅
└── TestAgentRoleIntegration (2 tests) ✅

tests/test_ppo_optimizer.py
├── TestRewardType (2 tests) ✅
├── TestWeightState (6 tests) ✅
├── TestExperience (2 tests) ✅
├── TestTradeOutcome (2 tests) ✅
├── TestPPOConfig (4 tests) ✅
├── TestSimpleNeuralNetwork (5 tests) ✅
├── TestValueNetwork (2 tests) ✅
├── TestExperienceBuffer (7 tests) ✅
├── TestTradingRewardCalculator (9 tests) ✅
├── TestPPOWeightOptimizer (12 tests) ✅
├── TestCreatePPOOptimizer (2 tests) ✅
├── TestCreateAdaptiveWeightOptimizer (3 tests) ✅
└── TestPPOIntegration (3 tests) ✅
```

**Total Tests**: 227 passed ✅ (118 sentiment + 50 multi-agent + 59 PPO optimizer)

---

## 💾 Research Deliverables

| Deliverable | Status | Location |
|-------------|--------|----------|
| Research Document | ✅ Complete | This file |
| Phase 1-8 Searches | ✅ Complete | Documented above |
| Architecture Diagrams | ✅ Included | ASCII diagrams |
| Code Patterns | ✅ Included | Python examples |
| Implementation Priorities | ✅ Complete | P1/P2/P3 rankings |
| Expansion Plan | ✅ Complete | 8 features identified |
| Feature Implementation | ✅ 7/8 Complete | Features 1, 2, 4, 5, 6, 7, 8 |
| Test Suite | ✅ Complete | 227 tests passing |

---

## 📝 Change Log

| Date | Change | Impact |
|------|--------|--------|
| 2025-12-01 | Initial research document created | Comprehensive expansion plan |
| 2025-12-01 | 8 research phases completed | All key areas covered |
| 2025-12-01 | 8 expansion features identified | Clear implementation roadmap |
| 2025-12-01 | Features 1, 2, 4, 5 implemented | 4/8 expansion features complete |
| 2025-12-01 | 98 tests passing | Full test coverage for new features |
| 2025-12-01 | Implementation status added | Documentation updated |
| 2025-12-01 | Feature 7 (Sentiment Momentum) implemented | 5/8 expansion features complete |
| 2025-12-01 | 118 tests passing | 20 new momentum tests added |
| 2025-12-01 | Feature 6 (Multi-Agent Architecture) implemented | 6/8 expansion features complete |
| 2025-12-01 | NewsAnalyst agent created | Event classification, impact assessment |
| 2025-12-01 | MultiAgentConsensus mechanism created | Weighted consensus aggregation |
| 2025-12-01 | 168 tests passing | 50 new multi-agent tests added |
| 2025-12-02 | Feature 8 (PPO-Optimized Weighting) implemented | 7/8 expansion features complete |
| 2025-12-02 | PPOWeightOptimizer created | Lightweight numpy-based PPO for ensemble weight optimization |
| 2025-12-02 | Trading reward calculators added | Sharpe, returns, accuracy, and combined reward functions |
| 2025-12-02 | Experience buffer for RL training | Batch training with GAE and evolution strategies updates |
| 2025-12-02 | 227 tests passing | 59 new PPO optimizer tests added |

---

## 🔗 References

### Academic Papers
- [FinDPO: Financial Sentiment Analysis through Preference Optimization](https://arxiv.org/abs/2507.18417)
- [TradingAgents: Multi-Agent LLM Framework](https://arxiv.org/abs/2412.20138)
- [Dissecting the Ledger: Liar Circuits in Financial LLMs](https://arxiv.org/html/2511.21756)
- [S&P 500 Volatility Forecasting with Regime-Switching](https://arxiv.org/html/2510.03236v1)
- [Adaptive Alpha Weighting with PPO](https://arxiv.org/html/2509.01393v1)
- [LLM Uncertainty in Sentiment Analysis](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1609097/full)

### Industry Sources
- [QuantConnect Tiingo Documentation](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/tiingo/tiingo-news-feed)
- [Guardrails AI](https://www.guardrailsai.com/)
- [CMC Markets Sentiment Analysis](https://www.cmcmarkets.com/en/technical-analysis/market-sentiment-analysis)
- [BizTech LLM Hallucinations](https://biztechmagazine.com/article/2025/08/llm-hallucinations-what-are-implications-financial-institutions)

### QuantConnect Resources
- [Tiingo News Feed Dataset](https://www.quantconnect.com/data/tiingo-news-feed)
- [Tiingo NLP Sentiment Competition](https://www.quantconnect.com/forum/discussion/6695/competition-algorithm-example-tiingo-nlp-sentiment/)
- [Sentiment Mean Strategy Forum](https://www.quantconnect.com/forum/discussion/8743/using-tiingo-news-to-create-a-sentiment-mean-strategy/)
