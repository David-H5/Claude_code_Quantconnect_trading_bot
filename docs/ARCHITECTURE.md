# Project Architecture

This document describes the directory structure and key modules of the QuantConnect Trading Bot.

## Directory Structure

### Claude Code Infrastructure (.claude/)

```
.claude/
├── hooks/                  # Claude Code hooks (organized by category)
│   ├── core/              # Critical hooks (RIC, session, protection)
│   │   ├── ric.py         # RIC Loop v5.1 Guardian (~6700 lines)
│   │   ├── session_stop.py
│   │   ├── pre_compact.py
│   │   ├── protect_files.py
│   │   └── hook_utils.py
│   ├── validation/        # Code and document validators
│   ├── research/          # Research tracking hooks
│   ├── trading/           # Trading-specific hooks (risk_validator.py)
│   ├── formatting/        # Code formatting hooks
│   └── agents/            # Agent orchestration hooks
├── state/                 # Runtime state files
│   ├── ric.json           # RIC Loop state
│   ├── doc_updates.json   # Documentation tracking
│   └── circuit_breaker.json
├── config/                # Configuration files
│   └── agents.json        # Agent configuration
├── commands/              # Slash commands
├── templates/             # Document templates
├── settings.json          # Claude Code settings
└── registry.json          # Hook/script registry
```

### Trading Infrastructure

```
algorithms/     # Trading algorithms (QuantConnect compatible)
├── options_trading_bot.py  # Main algorithm with Schwab integration

config/         # Configuration management
├── settings.json           # All adjustable parameters
├── __init__.py            # ConfigManager class

llm/            # LLM integration
├── agents/                # LLM trading agents (primary)
├── base.py                # Base classes (Sentiment, NewsItem)
├── sentiment.py           # FinBERT + Simple analyzers
├── providers.py           # OpenAI + Anthropic providers
├── ensemble.py            # Weighted ensemble predictions
└── news_analyzer.py       # News fetching and analysis

scanners/       # Market scanners
├── options_scanner.py     # Underpriced options detection
├── movement_scanner.py    # Price movement + news correlation

execution/      # Order execution
├── profit_taking.py       # Graduated profit-taking model
├── smart_execution.py     # Cancel/replace execution model

indicators/     # Technical indicators
├── volatility_bands.py    # Keltner, Bollinger
├── technical_alpha.py     # VWAP, RSI, MACD, CCI, OBV, Ichimoku

ui/             # PySide6 dashboard
├── widgets.py             # Reusable UI components
├── dashboard.py           # Main trading dashboard

models/         # Risk models
├── risk_manager.py        # Position sizing, limits
├── circuit_breaker.py     # Trading halt safety

agents_deprecated/  # DEPRECATED: Use llm/agents/ instead

utils/          # Helper functions
├── overnight_state.py     # Unified state manager (file locking)
├── overnight_config.py    # Session configuration loader
└── progress_parser.py     # Progress file parser

scripts/        # Utility scripts
├── health_check.py        # HTTP health endpoints
├── auto-resume.sh         # Session auto-resume
└── run_overnight.sh       # Overnight session runner

observability/  # Logging and monitoring (CANONICAL LOCATIONS)
├── logging/
│   ├── structured.py      # StructuredLogger (use this, not utils/)
│   └── audit.py           # AuditLogger (use this, not compliance/)
└── monitoring/
    └── system/
        ├── health.py      # SystemMonitor (use this, not utils/)
        └── resource.py    # ResourceMonitor (use this, not utils/)

tests/          # Unit and integration tests
docs/prompts/   # Prompt framework (versioned)
.backups/       # Automatic backups (DO NOT DELETE)
```

## Key Files

| File | Purpose |
|------|---------|
| `.claude/hooks/core/ric.py` | Main RIC v5.1 Guardian (~6700 lines) |
| `.claude/hooks/research/research_saver.py` | Research/upgrade document creator (~1200 lines) |
| `.claude/hooks/agents/agent_orchestrator.py` | Multi-agent orchestration engine (~1000 lines) |
| `.claude/RIC_CONTEXT.md` | Quick reference |
| `.claude/state/ric.json` | Session state (auto-managed) |
| `.claude/templates/` | Document templates (upgrade, research, insight, guide) |

## Core Capabilities

1. **LLM Integration**: FinBERT + GPT-4o + Claude ensemble for sentiment analysis
2. **Options Scanner**: Underpriced options detection using Greeks and IV analysis
3. **Movement Scanner**: 2-4% movers with news corroboration
4. **Profit-Taking**: Graduated selling at +100%, +200%, +400%, +1000%
5. **Smart Execution**: Auto cancel/replace with configurable max bid increase
6. **Technical Analysis**: VWAP, RSI, MACD, CCI, Bollinger, OBV, Ichimoku

## Module Dependencies

```
algorithms/
    └── uses: llm/, scanners/, execution/, models/, indicators/

llm/agents/
    └── uses: llm/base.py, llm/providers.py

execution/
    └── uses: models/risk_manager.py, models/circuit_breaker.py

ui/
    └── uses: config/, evaluation/, llm/agents/
```

## Configuration System

The codebase uses TWO separate configuration systems for different purposes:

### Trading Configuration (`config/`)

All trading parameters are configurable via `config/settings.json`:

```python
from config import get_config, ConfigManager

config = get_config()

# Access nested values
max_daily_loss = config.get("risk_management.max_daily_loss_pct")

# Get typed config objects
profit_config = config.get_profit_taking_config()
```

| Section | Description |
|---------|-------------|
| `brokerage` | Schwab API credentials and settings |
| `risk_management` | Position limits, daily loss, drawdown |
| `profit_taking` | Threshold levels for selling winners |
| `order_execution` | Cancel/replace settings |
| `options_scanner` | Delta range, DTE, IV thresholds |
| `movement_scanner` | Movement %, volume surge |
| `llm_integration` | Provider configs, ensemble weights |
| `technical_indicators` | Indicator parameters |
| `ui` | Dashboard theme and layout |

### Overnight/Session Configuration (`config/overnight.yaml`)

Claude Code session settings are in `config/overnight.yaml`:

```python
from utils.overnight_config import get_overnight_config

config = get_overnight_config()
max_runtime = config.max_runtime_hours
discord_url = config.discord_webhook
```

| Setting | Description |
|---------|-------------|
| `max_runtime_hours` | Maximum session duration |
| `max_continuations` | Max auto-continuations |
| `max_restarts` | Max crash restarts |
| `discord_webhook` | Notification URL |
| `ric_mode` | RIC Loop enforcement level |
| `progress_file` | Task tracking file |

## Branch Strategy

| Branch | Purpose | Protection |
|--------|---------|------------|
| `main` | Production algorithms | Requires PR from develop |
| `develop` | Integration testing | Requires PR approval |
| `feature/*` | New features | None |
| `bugfix/*` | Bug fixes | None |
| `hotfix/*` | Urgent fixes | Can PR to main |
