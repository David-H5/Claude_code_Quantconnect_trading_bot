# QuantConnect Trading Bot

A Python-based algorithmic trading project built for QuantConnect's Lean platform, designed to be developed semi-autonomously with Claude Code.

## Overview

This project contains custom trading algorithms for the QuantConnect platform. The codebase is structured to enable efficient development with AI assistance through Claude Code.

## Project Structure

```
.
├── .claude/                 # Claude Code configuration
│   ├── settings.json       # Permissions and model settings
│   └── commands/           # Custom slash commands
├── .github/                # GitHub Actions workflows
│   ├── workflows/
│   │   ├── tests.yml       # CI testing pipeline
│   │   ├── code-quality.yml # Linting and formatting
│   │   ├── deploy.yml      # Deploy to QuantConnect
│   │   └── backtest.yml    # Run cloud backtests
│   └── ISSUE_TEMPLATE/     # Issue templates
├── algorithms/             # Trading algorithm implementations
│   ├── basic_buy_hold.py   # Buy-and-hold baseline
│   └── simple_momentum.py  # RSI momentum strategy
├── data/                   # Custom data sources
│   └── custom_data.py      # Sentiment & economic data loaders
├── indicators/             # Custom technical indicators
│   └── volatility_bands.py # Volatility bands & Keltner channels
├── models/                 # Risk and portfolio models
│   └── risk_manager.py     # Position sizing & risk controls
├── research/               # Jupyter notebooks
│   └── strategy_exploration.ipynb
├── tests/                  # Unit and integration tests
├── utils/                  # Utility functions
│   └── calculations.py     # Performance metrics & position sizing
├── config/                 # Configuration files
├── CLAUDE.md              # Claude Code project instructions
├── .clinerules            # Development guidelines
└── requirements.txt       # Python dependencies
```

## Features

### Algorithms

- **OptionsTradingBot**: Full-featured options trading with Greeks-based filtering, LLM sentiment analysis, and smart execution
- **Two-Part Spread Strategy**: Leg into net-credit butterflies/iron condors with intelligent order management
- **Wheel Strategy**: Options income generation (reference implementation)

### Options Trading Capabilities

- **Greeks-Based Universe Filtering**: Pre-filter options by Delta, Gamma, Theta, Vega, IV before analysis
- **OptionStrategies Factory Methods**: 37+ pre-built multi-leg strategies (butterflies, condors, spreads)
- **ComboOrders**: Atomic multi-leg execution with net pricing (Charles Schwab compatible)
- **Immediate Greeks Access**: IV-based Greeks available instantly (no warmup required, PR #6720)
- **ThetaPerDay**: Interactive Brokers-compatible daily time decay

### Custom Indicators

- **Technical Alpha**: VWAP, RSI, MACD, CCI, Bollinger Bands, OBV, Ichimoku
- **Volatility Bands**: ATR-based dynamic support/resistance, Keltner Channels
- **Enhanced Volatility**: IV Rank, IV Percentile, volatility regime detection
- **Volatility Surface**: Multi-dimensional IV analysis across strikes and expirations

### Risk Management

- **Circuit Breaker**: Trading halt safety with human reset requirement
- **Risk Manager**: Position sizing, daily loss limits, drawdown controls
- **Portfolio Hedging**: Delta/gamma hedging for option portfolios
- **Multi-Leg Greeks**: Aggregate position Greeks tracking
- **PnL Attribution**: Breakdown P&L by Greek contributions

### LLM Integration

- **Ensemble Analysis**: FinBERT + GPT-4o + Claude for sentiment
- **News Correlation**: Movement validation with news corroboration
- **Weighted Voting**: Configurable ensemble weights

### Smart Execution

- **Cancel/Replace**: Auto-cancel unfilled orders with bid improvement
- **Profit Taking**: Graduated selling at +100%, +200%, +400%, +1000%
- **Fill Predictor**: Track and optimize fill rates
- **Spread Analysis**: Quality scoring for wide bid-ask spreads

## Getting Started

### Prerequisites

- Python 3.8+
- QuantConnect LEAN CLI (optional, for local backtesting)
- Git

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Claude_code_Quantconnect_trading_bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your QuantConnect credentials
```

5. (Optional) Log in to LEAN CLI:
```bash
lean login
```

## Usage

### Claude Code Commands

Use these slash commands for common tasks:

| Command | Description |
|---------|-------------|
| `/new-algorithm <description>` | Create a new trading algorithm |
| `/backtest <algorithm>` | Run a backtest |
| `/run-tests [pattern]` | Run test suite |
| `/analyze-strategy <algorithm>` | Review strategy code |
| `/lint [target]` | Run code quality checks |

### Creating a New Algorithm

1. Create a new Python file in the `algorithms/` directory
2. Inherit from `QCAlgorithm` base class
3. Implement `Initialize()` and `OnData()` methods
4. Add your trading logic

```python
from AlgorithmImports import *

class MyAlgorithm(QCAlgorithm):
    def Initialize(self) -> None:
        # Python API uses snake_case for methods
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        self.symbol = self.add_equity("SPY", Resolution.Daily).Symbol

    def OnData(self, data: Slice) -> None:
        # ContainsKey is a framework method (PascalCase exception)
        if not data.ContainsKey(self.symbol):
            return
        # Trading logic here
```

### Running Backtests

**On QuantConnect Cloud:**
- Upload your algorithm to QuantConnect.com
- Configure backtest parameters
- Run and analyze results

**Locally with LEAN:**
```bash
lean backtest "algorithms/your_algorithm.py"
```

**Via GitHub Actions:**
- Go to Actions tab
- Select "Run Backtest" workflow
- Enter algorithm name and date range

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=algorithms --cov=indicators --cov=models --cov=utils --cov-report=term-missing
```

Run specific test categories:
```bash
pytest tests/ -m unit       # Unit tests only
pytest tests/ -m integration # Integration tests only
```

## GitHub Integration

### Workflows

- **tests.yml**: Runs on every push/PR - executes tests across Python 3.8-3.10
- **code-quality.yml**: Checks Black formatting and Pylint score
- **deploy.yml**: Deploys changed algorithms to QuantConnect on push to main
- **backtest.yml**: Manual workflow to run cloud backtests

### Required Secrets

Add these secrets to your GitHub repository:
- `QC_USER_ID`: Your QuantConnect user ID
- `QC_API_TOKEN`: Your QuantConnect API token

## Development with Claude Code

This project is optimized for development with Claude Code. Key features:

- **CLAUDE.md**: Project-specific instructions and quick reference
- **.clinerules**: Comprehensive QuantConnect development guidelines
- **Custom Commands**: Slash commands for common tasks
- **Modular Design**: Algorithms, indicators, and utilities are separated
- **Comprehensive Tests**: Mocked QuantConnect environment for testing

### Working with Claude Code

When requesting changes:
- Be specific about the strategy logic you want to implement
- Reference existing patterns in the codebase
- Ask for explanations of QuantConnect API methods if needed

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## Resources

### Official QuantConnect Resources

- [QuantConnect Documentation](https://www.quantconnect.com/docs)
- [LEAN GitHub Repository](https://github.com/QuantConnect/Lean)
- [QuantConnect Community](https://www.quantconnect.com/forum)

### Project Documentation

**QuantConnect Research (November 2025):**

- [Quick Reference Guide](docs/QUICK_REFERENCE.md) - Common patterns and code snippets
- [Phase 2: Integration Research](docs/research/PHASE_2_INTEGRATION_RESEARCH.md) - Schwab integration, ComboOrders, Greeks (39KB)
- [Phase 3: Advanced Features](docs/research/PHASE3_ADVANCED_FEATURES_RESEARCH.md) - Multi-leg strategies, risk management (53KB)
- [Integration Summary](docs/research/INTEGRATION_SUMMARY.md) - Phase 2 executive summary
- [Research Summary](docs/research/RESEARCH_SUMMARY.md) - Phase 3 executive summary

**Core Documentation:**

- [QuantConnect Options Trading](docs/quantconnect/OPTIONS_TRADING.md) - **Updated with PR #6720, OptionStrategies, Greeks filtering**
- [QuantConnect GitHub Resources](docs/development/QUANTCONNECT_GITHUB_GUIDE.md) - LEAN architecture, patterns, examples
- [Development Best Practices](docs/development/BEST_PRACTICES.md) - Trading safety and risk management
- [Coding Standards](docs/development/CODING_STANDARDS.md) - Style guide and conventions
- [Infrastructure Setup](docs/infrastructure/SETUP_SUMMARY.md) - Compute nodes and Object Store
- [Strategy Documentation](docs/strategies/README.md) - Two-part spread strategy details

### Claude Code

- [Claude Code Documentation](https://docs.anthropic.com/claude-code)
- [Autonomous Agents Guide](docs/autonomous-agents/README.md) - Extended development sessions

## License

MIT License - See LICENSE file for details

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.
