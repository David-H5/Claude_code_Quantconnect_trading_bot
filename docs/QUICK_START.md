# Quick Start Guide

Get up and running with the QuantConnect Options Trading Bot in under 30 minutes.

**Last Updated**: November 30, 2025

---

## 🚀 30-Second Overview

This project is a **semi-autonomous options trading bot** that:
- Trades options autonomously using 37+ QuantConnect strategies
- Accepts manual orders via UI widgets
- Uses LLM sentiment analysis to filter trades
- Manages positions automatically with profit-taking and stop-loss

**Current Status**: All modules complete (541 tests passing), needs integration

**Next Step**: Create main algorithm to integrate all modules

---

## 📋 Prerequisites

### Required
- **Python 3.10+** installed
- **Git** for version control
- **QuantConnect Account** (free tier OK for backtesting)
- **Code editor** (VS Code recommended)

### Optional
- **Charles Schwab Account** (for live/paper trading)
- **OpenAI API Key** (for GPT-4 sentiment)
- **Anthropic API Key** (for Claude sentiment)

---

## 🏃 Quick Setup (5 Minutes)

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd Claude_code_Quantconnect_trading_bot
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .  # Install project in editable mode
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys (optional for basic testing)
```

### 5. Run Tests
```bash
pytest tests/ -v
```

**Expected**: 541 tests passing ✅

---

## 📊 Project Structure (What's Where)

```
Claude_code_Quantconnect_trading_bot/
├── algorithms/          # QuantConnect algorithms
│   ├── options_trading_bot.py      # Original main algorithm
│   └── hybrid_options_bot.py       # ⚠️ TO CREATE (main integration)
│
├── execution/          # Order execution modules (✅ Complete)
│   ├── option_strategies_executor.py   # 37+ autonomous strategies
│   ├── manual_legs_executor.py         # Two-part spread execution
│   ├── bot_managed_positions.py        # Auto profit-taking
│   └── recurring_order_manager.py      # Scheduled orders
│
├── api/                # API for UI integration (✅ Complete)
│   ├── order_queue_api.py              # Order queue logic
│   └── rest_server.py                  # ⚠️ TO CREATE (HTTP server)
│
├── ui/                 # PySide6 UI widgets (✅ Complete)
│   ├── strategy_selector.py            # 37+ strategy dropdown
│   ├── custom_leg_builder.py           # Custom spread builder
│   └── position_tracker.py             # Position tracking widget
│
├── models/             # Risk and analysis models
│   ├── risk_manager.py                 # Position sizing, limits
│   ├── circuit_breaker.py              # Safety halt mechanism
│   └── enhanced_volatility.py          # IV analysis
│
├── indicators/         # Technical indicators
│   └── technical_alpha.py              # VWAP, RSI, MACD, etc.
│
├── llm/                # LLM integration (✅ Complete, unused)
│   ├── sentiment.py                    # FinBERT sentiment
│   ├── providers.py                    # OpenAI, Anthropic
│   └── ensemble.py                     # Multi-model ensemble
│
├── scanners/           # Market scanners
│   ├── options_scanner.py              # Underpriced options
│   └── movement_scanner.py             # Price movement + news
│
├── tests/              # Test suite (541 tests)
├── config/             # Configuration
│   └── settings.json                   # All adjustable parameters
│
└── docs/               # Documentation (YOU ARE HERE!)
    ├── README.md                       # Main documentation index
    ├── PROJECT_STATUS.md               # Current status dashboard
    ├── IMPLEMENTATION_TRACKER.md       # Sprint task tracking
    ├── ROADMAP.md                      # Strategic roadmap
    └── QUICK_START.md                  # This file
```

---

## 🎯 What You Should Do Next

### For New Developers

**Day 1**: Understand the codebase
1. Read [Project Status](PROJECT_STATUS.md) - Understand current state
2. Read [Architecture Overview](architecture/README.md) - Understand design
3. Browse module READMEs to understand each component

**Day 2-3**: Make your first contribution
4. Pick a task from [Implementation Tracker](IMPLEMENTATION_TRACKER.md)
5. Read [Development Guide](development/README.md)
6. Create a branch and start coding

### For Claude Code Agent

**Immediate**: Start integration work
1. Read [Implementation Tracker](IMPLEMENTATION_TRACKER.md) - Current sprint
2. Start with Task 1: Create main hybrid algorithm
3. Follow [Next Steps Guide](NEXT_STEPS_GUIDE.md) for detailed instructions

### For Project Managers

**Review**: Check project health
1. Read [Project Status](PROJECT_STATUS.md) - Metrics and KPIs
2. Read [Roadmap](ROADMAP.md) - Timeline and milestones
3. Check [Implementation Tracker](IMPLEMENTATION_TRACKER.md) - Sprint progress

---

## 🧪 Running Tests

### All Tests
```bash
pytest tests/ -v
```

### Specific Module
```bash
pytest tests/test_bot_managed_positions.py -v
```

### With Coverage
```bash
pytest tests/ --cov=execution --cov-report=html
open htmlcov/index.html
```

### Integration Tests Only
```bash
pytest tests/test_integration.py -v
```

---

## 🔧 Configuration

### Main Configuration File

**Location**: `config/settings.json`

**Key Sections**:
- `brokerage`: Charles Schwab credentials
- `risk_management`: Position limits, daily loss
- `profit_taking`: Profit thresholds
- `option_strategies_executor`: Autonomous strategy settings
- `bot_managed_positions`: Auto profit-taking config

**Example**:
```json
{
  "risk_management": {
    "max_open_positions": 5,
    "max_position_size_pct": 0.25,
    "max_daily_loss_pct": 0.03
  },
  "option_strategies_executor": {
    "enabled": true,
    "iv_rank_threshold_iron_condor": 50,
    "min_dte": 30,
    "max_dte": 60
  }
}
```

**See**: [Configuration Guide](development/CONFIGURATION_GUIDE.md) (to be created)

---

## 🚀 Running a Backtest (Coming Soon)

> **Note**: Main algorithm integration is pending (Task 1)

### Once Integration Complete:

1. **Open QuantConnect**
   - Go to https://www.quantconnect.com
   - Login to your account

2. **Create New Project**
   - Algorithm Language: Python
   - Copy algorithm code

3. **Configure Backtest**
   ```python
   SetStartDate(2024, 11, 1)
   SetEndDate(2024, 11, 30)
   SetCash(100000)
   ```

4. **Run Backtest**
   - Click "Backtest"
   - Wait for results
   - Review metrics

**See**: [Backtesting Guide](development/BACKTESTING_GUIDE.md) (to be created)

---

## 💻 Development Workflow

### Creating a New Feature

1. **Create Branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Write Code**
   - Follow [Coding Standards](development/CODING_STANDARDS.md)
   - Add type hints
   - Write docstrings

3. **Write Tests**
   - Create test file: `tests/test_my_feature.py`
   - Aim for >70% coverage

4. **Run Tests**
   ```bash
   pytest tests/test_my_feature.py -v
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add my feature"
   git push origin feature/my-feature
   ```

6. **Create Pull Request**
   - Review checklist
   - Request review

---

## 🤖 Claude Code Setup

### For Autonomous Development

1. **Review Instructions**
   - Read `CLAUDE.md` in repository root
   - Understand project context

2. **Check Current Sprint**
   - Review [Implementation Tracker](IMPLEMENTATION_TRACKER.md)
   - Identify next task

3. **Start Working**
   - Use Task tool for complex work
   - Update todo list with TodoWrite
   - Mark tasks complete as you go

4. **Documentation**
   - Update relevant docs as code changes
   - Cross-reference related documents

---

## 📚 Key Documentation

### Essential Reading (Priority Order)

1. [Project Status](PROJECT_STATUS.md) - Where we are
2. [Implementation Tracker](IMPLEMENTATION_TRACKER.md) - What's next
3. [Architecture Overview](architecture/README.md) - How it works
4. [Development Guide](development/README.md) - How to code
5. [Roadmap](ROADMAP.md) - Where we're going

### Reference Documentation

- [API Reference](api/README.md) - Code API docs
- [Strategy Docs](strategies/README.md) - Trading strategies
- [QuantConnect Reference](quantconnect/README.md) - Platform docs
- [Infrastructure](infrastructure/README.md) - Setup guides

---

## ❓ Common Questions

### Q: Can I run the trading bot right now?
**A**: Not yet. The modules are complete but not integrated. You need to create the main algorithm first ([Task 1](IMPLEMENTATION_TRACKER.md#task-1-create-main-hybrid-algorithm)).

### Q: How do I add a new trading strategy?
**A**: See [Adding Strategies Guide](development/ADDING_STRATEGIES.md) (to be created). For now, check `execution/option_strategies_executor.py`.

### Q: Where are the configuration settings?
**A**: `config/settings.json` contains all adjustable parameters.

### Q: How do I run tests?
**A**: `pytest tests/ -v` runs all tests. See [Testing Guide](development/TESTING_GUIDE.md).

### Q: Where do I find QuantConnect documentation?
**A**: [docs/quantconnect/README.md](quantconnect/README.md) has curated reference docs.

### Q: Can I trade live right now?
**A**: No. You must complete: Integration → Backtesting → Paper Trading → Human Approval → Live.

---

## 🆘 Getting Help

### Documentation
- **Main Index**: [docs/README.md](README.md)
- **Search**: Use `grep -r "your term" docs/`

### Issues
- Check existing GitHub Issues
- Create new issue with `bug` or `question` label

### Community
- QuantConnect Forum: https://www.quantconnect.com/forum
- LEAN GitHub: https://github.com/QuantConnect/Lean

---

## ✅ Quick Health Check

Run this command to verify your setup:

```bash
# Test environment
python3 --version          # Should be 3.10+
pytest --version           # Should be installed

# Test imports
python3 -c "from execution import create_bot_position_manager; print('✅ Modules OK')"

# Run quick tests
pytest tests/ -x --lf      # Run last failed, stop on first failure
```

**All passing?** You're ready to start developing! 🎉

---

## 🎯 Next Steps

Choose your path:

### 👨‍💻 I'm a Developer
→ Read [Development Guide](development/README.md)
→ Check [Implementation Tracker](IMPLEMENTATION_TRACKER.md)
→ Pick a task and start coding

### 🤖 I'm Claude Code
→ Read [Implementation Tracker](IMPLEMENTATION_TRACKER.md)
→ Start with Task 1 (main algorithm)
→ Follow [Next Steps Guide](NEXT_STEPS_GUIDE.md)

### 📊 I'm a Project Manager
→ Review [Project Status](PROJECT_STATUS.md)
→ Check [Roadmap](ROADMAP.md)
→ Monitor progress

### 📚 I'm a Researcher
→ Browse [Strategy Docs](strategies/README.md)
→ Check [Research Notes](research/README.md)
→ Review backtest results (when available)

---

**Welcome to the project!** 🚀

**Questions?** See [Documentation Index](README.md) for all resources.

**Last Updated**: November 30, 2025
