# Documentation Index

**Project**: QuantConnect Semi-Autonomous Options Trading Bot
**Last Updated**: November 30, 2025
**Documentation Version**: 2.0

---

## 📊 Quick Links

| Document | Purpose | Audience |
|----------|---------|----------|
| [📍 Project Status](PROJECT_STATUS.md) | Current progress, metrics, next steps | All |
| [🗺️ Roadmap](ROADMAP.md) | Strategic direction, phases, timeline | All |
| [🎯 Implementation Tracker](IMPLEMENTATION_TRACKER.md) | Detailed task tracking | Developers |
| [🚀 Quick Start](QUICK_START.md) | Get up and running fast | New developers |
| [📚 API Reference](api/README.md) | Code API documentation | Developers |

---

## 🎯 For Different Audiences

### 👤 New to the Project?
Start here in this order:
1. [Project Status](PROJECT_STATUS.md) - Understand current state
2. [Quick Start](QUICK_START.md) - Set up development environment
3. [Architecture Overview](architecture/README.md) - Understand system design
4. [Contributing Guide](CONTRIBUTING.md) - How to contribute

### 💻 Developers
- [Development Guide](development/README.md) - Standards, practices, workflows
- [Implementation Tracker](IMPLEMENTATION_TRACKER.md) - Current sprint tasks
- [Testing Guide](development/TESTING_GUIDE.md) - How to run and write tests
- [API Reference](api/README.md) - Code documentation

### 🤖 Claude Code Agent
- [Claude Instructions](../CLAUDE.md) - Main instructions for autonomous development
- [Implementation Tracker](IMPLEMENTATION_TRACKER.md) - What to work on next
- [Autonomous Agent Guide](autonomous-agents/README.md) - Overnight sessions

### 📈 Project Managers
- [Project Status](PROJECT_STATUS.md) - High-level overview
- [Roadmap](ROADMAP.md) - Timeline and milestones
- [Implementation Tracker](IMPLEMENTATION_TRACKER.md) - Detailed progress

### 📖 Strategy Researchers
- [Strategy Documentation](strategies/README.md) - All trading strategies
- [Research Notes](research/README.md) - Analysis and findings

---

## 📁 Documentation Structure

```
docs/
├── README.md                           # This file - main index
├── PROJECT_STATUS.md                   # 📊 Current state, metrics, next steps
├── ROADMAP.md                          # 🗺️ Strategic roadmap (links to ../ROADMAP.md)
├── IMPLEMENTATION_TRACKER.md           # 🎯 Detailed task tracking
├── QUICK_START.md                      # 🚀 Get started fast
├── CONTRIBUTING.md                     # 🤝 How to contribute
│
├── architecture/                       # System architecture
│   ├── README.md                       # Architecture overview
│   ├── HYBRID_ARCHITECTURE.md          # Hybrid options trading system
│   ├── OPTIONS_ORDER_ARCHITECTURE.md   # Order flow architecture
│   └── SYSTEM_DIAGRAMS.md              # Visual architecture diagrams
│
├── development/                        # Development guides
│   ├── README.md                       # Development overview
│   ├── ENHANCED_RIC_WORKFLOW.md        # ⭐⭐ 7-phase Meta-RIC Loop v2.2 (CURRENT)
│   ├── UPGRADE_LOOP_WORKFLOW.md        # 6-phase workflow (DEPRECATED)
│   ├── BEST_PRACTICES.md               # Trading safety, risk management
│   ├── CODING_STANDARDS.md             # Code style, type hints
│   ├── TESTING_GUIDE.md                # Test strategy and execution
│   └── QUANTCONNECT_GITHUB_GUIDE.md    # QuantConnect patterns
│
├── strategies/                         # Trading strategies
│   ├── README.md                       # Strategy overview
│   ├── TWO_PART_SPREAD_STRATEGY.md     # Main 2-part spread strategy
│   └── ARBITRAGE_EXECUTOR.md           # Arbitrage execution details
│
├── infrastructure/                     # Infrastructure setup
│   ├── README.md                       # Infrastructure overview
│   ├── COMPUTE_NODES.md                # Node selection guide
│   ├── OBJECT_STORE.md                 # Object Store usage
│   ├── DATA_SUBSCRIPTIONS.md           # Data feed setup
│   └── SETUP_SUMMARY.md                # Complete setup guide
│
├── autonomous-agents/                  # Autonomous development
│   ├── README.md                       # Autonomous agent guide
│   ├── INSTALLATION.md                 # Setup instructions
│   ├── COMPARISON.md                   # Tool comparisons
│   └── TODO.md                         # Implementation checklist
│
├── quantconnect/                       # QuantConnect reference
│   ├── README.md                       # QuantConnect overview
│   ├── PYTHON_API_REFERENCE.md         # Python API guide
│   ├── OPTIONS_TRADING.md              # Options-specific patterns
│   └── ... (21 total reference docs)
│
├── research/                           # Research & analysis
│   ├── README.md                       # Research overview
│   └── ... (analysis documents)
│
└── api/                                # API documentation
    └── README.md                       # API reference

Root level:
../ROADMAP.md                           # Main project roadmap
../CLAUDE.md                            # Claude Code instructions
../CONTRIBUTING.md                      # Contribution guidelines
```

---

## 📚 Documentation by Category

### 🎯 Project Management

| Document | Description | Status |
|----------|-------------|--------|
| [Project Status](PROJECT_STATUS.md) | Current progress, metrics, KPIs | ✅ Active |
| [Roadmap](ROADMAP.md) | Strategic phases and timeline | ✅ Active |
| [Implementation Tracker](IMPLEMENTATION_TRACKER.md) | Sprint-level task tracking | ✅ Active |
| [HYBRID_IMPLEMENTATION_PROGRESS](architecture/HYBRID_IMPLEMENTATION_PROGRESS.md) | Hybrid architecture progress | ✅ Complete |

### 🏗️ Architecture

| Document | Description | Status |
|----------|-------------|--------|
| [Architecture Overview](architecture/README.md) | System design overview | ✅ Current |
| [Hybrid Architecture](architecture/HYBRID_ARCHITECTURE.md) | Autonomous + Manual hybrid system | ✅ Current |
| [Options Order Architecture](architecture/OPTIONS_ORDER_ARCHITECTURE.md) | Order flow design | ✅ Current |
| [System Diagrams](architecture/SYSTEM_DIAGRAMS.md) | Visual architecture | 📝 Planned |

### 💻 Development

| Document | Description | Status |
|----------|-------------|--------|
| [Development Guide](development/README.md) | Development overview | ✅ Current |
| [**Enhanced RIC Workflow**](development/ENHANCED_RIC_WORKFLOW.md) | **7-phase Meta-RIC Loop v2.2** | ⭐⭐ CURRENT |
| [Upgrade Loop Workflow](development/UPGRADE_LOOP_WORKFLOW.md) | Legacy 6-phase workflow | ⚠️ DEPRECATED |
| [Best Practices](development/BEST_PRACTICES.md) | Trading safety, risk management | ✅ Current |
| [Coding Standards](development/CODING_STANDARDS.md) | Style guide, conventions | ✅ Current |
| [Testing Guide](development/TESTING_GUIDE.md) | Test strategy | 📝 To Create |
| [QuantConnect GitHub Guide](development/QUANTCONNECT_GITHUB_GUIDE.md) | QC patterns from source | ✅ Current |

### 📈 Trading Strategies

| Document | Description | Status |
|----------|-------------|--------|
| [Strategy Overview](strategies/README.md) | All strategies index | ✅ Current |
| [Two-Part Spread Strategy](strategies/TWO_PART_SPREAD_STRATEGY.md) | Primary strategy | ✅ Current |
| [Arbitrage Executor](strategies/ARBITRAGE_EXECUTOR.md) | Arbitrage execution | ✅ Current |

### 🔧 Infrastructure

| Document | Description | Status |
|----------|-------------|--------|
| [Infrastructure Overview](infrastructure/README.md) | Setup overview | 📝 To Create |
| [Compute Nodes](infrastructure/COMPUTE_NODES.md) | Node selection guide | ✅ Current |
| [Object Store](infrastructure/OBJECT_STORE.md) | Persistence guide | ✅ Current |
| [Data Subscriptions](infrastructure/DATA_SUBSCRIPTIONS.md) | Data feed setup | ✅ Current |
| [Setup Summary](infrastructure/SETUP_SUMMARY.md) | Complete setup | ✅ Current |

### 🤖 Autonomous Development

| Document | Description | Status |
|----------|-------------|--------|
| [Autonomous Agents](autonomous-agents/README.md) | Main agent guide | ✅ Current |
| [Installation](autonomous-agents/INSTALLATION.md) | Setup instructions | ✅ Current |
| [Tool Comparison](autonomous-agents/COMPARISON.md) | Framework comparison | ✅ Current |

### 📖 Reference

| Document | Description | Status |
|----------|-------------|--------|
| [QuantConnect Reference](quantconnect/README.md) | QC documentation | ✅ Current |
| [Python API Reference](quantconnect/PYTHON_API_REFERENCE.md) | Python API guide | ✅ Current |
| [Options Trading](quantconnect/OPTIONS_TRADING.md) | Options patterns | ✅ Current |

### 📋 Architectural Decision Records (ADRs)

| Document | Description | Status |
|----------|-------------|--------|
| [ADR Index](adr/README.md) | All architectural decisions | ⭐ NEW |
| [ADR-0001](adr/ADR-0001-use-quantconnect-lean.md) | Use QuantConnect LEAN | ✅ Accepted |
| [ADR-0002](adr/ADR-0002-charles-schwab-brokerage.md) | Charles Schwab Brokerage | ✅ Accepted |
| [ADR-0003](adr/ADR-0003-llm-ensemble-approach.md) | LLM Ensemble Approach | ✅ Accepted |
| [ADR-0004](adr/ADR-0004-hybrid-architecture.md) | Hybrid Architecture | ✅ Accepted |
| [ADR-0005](adr/ADR-0005-circuit-breaker-pattern.md) | Circuit Breaker Pattern | ✅ Accepted |
| [ADR-0006](adr/ADR-0006-two-part-spread-strategy.md) | Two-Part Spread Strategy | ✅ Accepted |
| [ADR-0007](adr/ADR-0007-upgrade-loop-workflow.md) | Upgrade Loop Workflow | ✅ Accepted |

### 🔒 Processes & Safety

| Document | Description | Status |
|----------|-------------|--------|
| [Root Cause Analysis](processes/ROOT_CAUSE_ANALYSIS.md) | RCA process and 5 Whys method | ⭐ NEW |
| [RCA Template](processes/rca-template.md) | Template for incident analysis | ⭐ NEW |
| [Incident Log](incidents/README.md) | Incident tracking and history | ⭐ NEW |

### 🧪 Testing & Quality

| Document | Description | Status |
|----------|-------------|--------|
| [Regression Tests](../tests/regression/) | Historical bug and edge case tests | ⭐ NEW |
| [Pre-Trade Validator](../execution/pre_trade_validator.py) | Position limit enforcement | ⭐ NEW |
| [Monte Carlo Tests](../tests/test_monte_carlo.py) | Volatility regime stress testing | ⭐ NEW |

### 📋 Upgrade Paths

| Document | Description | Status |
|----------|-------------|--------|
| [UPGRADE-001](upgrades/UPGRADE_001_FOUNDATION.md) | Foundation Infrastructure | ✅ Complete |
| [UPGRADE-002](upgrades/UPGRADE_002_TESTING_SAFETY.md) | Testing & Safety | ✅ Complete |

---

## 🔄 Documentation Workflow

### When to Update Documentation

| Trigger | Documents to Update |
|---------|---------------------|
| Sprint starts | [Implementation Tracker](IMPLEMENTATION_TRACKER.md), [Project Status](PROJECT_STATUS.md) |
| Task completed | [Implementation Tracker](IMPLEMENTATION_TRACKER.md), relevant module docs |
| Architecture changes | [Architecture docs](architecture/), [System Diagrams](architecture/SYSTEM_DIAGRAMS.md) |
| New feature | [Roadmap](ROADMAP.md), strategy docs, API docs |
| Bug fixes | [CHANGELOG.md](../CHANGELOG.md), module docs |
| Major milestone | [Project Status](PROJECT_STATUS.md), [Roadmap](ROADMAP.md) |

### Documentation Standards

1. **File Format**: Markdown (.md)
2. **Line Length**: 100 characters max
3. **Heading Style**: ATX style (`#`, `##`, etc.)
4. **Links**: Always use relative paths
5. **Dates**: Format as `YYYY-MM-DD`
6. **Status Indicators**: ✅ Complete, ⏳ In Progress, 📝 Planned, ❌ Blocked

### Cross-Referencing

All documents should link to related documents:
- **See Also** section at bottom
- Inline links to relevant docs
- Backlinks where appropriate

---

## 📊 Documentation Health Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Documentation Coverage | > 90% | ~85% |
| Outdated Docs (> 30 days) | < 10% | ~15% |
| Broken Links | 0 | TBD |
| Missing Cross-References | < 5% | ~20% |

---

## 🆘 Need Help?

- **Can't find something?** Check [Project Status](PROJECT_STATUS.md) for overview
- **New developer?** Start with [Quick Start](QUICK_START.md)
- **Want to contribute?** Read [Contributing Guide](CONTRIBUTING.md)
- **Complex feature to implement?** Use [Enhanced RIC Workflow](development/ENHANCED_RIC_WORKFLOW.md) or `/ric-start` ⭐⭐
- **Questions about architecture?** See [Architecture Overview](architecture/README.md)
- **Looking for specific API?** Check [API Reference](api/README.md)

---

## 📝 Recent Updates

| Date | Document | Change |
|------|----------|--------|
| 2025-12-01 | **ADR System** | **Created 7 Architectural Decision Records** ⭐ |
| 2025-12-01 | Pre-commit Config | Migrated to Ruff (200x faster) + GitLeaks |
| 2025-12-01 | CI Workflow | Added mypy gate + 70% coverage enforcement |
| 2025-12-01 | CLAUDE.md | Added pre-deployment checklist |
| 2025-12-03 | **Enhanced RIC Workflow** | **Upgraded to Meta-RIC Loop v2.2 (7 phases, insight-driven)** |
| 2025-12-02 | Enhanced RIC Workflow | Initial RIC Loop v1.0 (8 phases, score-based) |
| 2025-12-01 | Upgrade Loop Workflow | Created 6-phase iterative development workflow (now deprecated) |
| 2025-12-01 | Implementation Tracker | Added loop verification checklist |
| 2025-11-30 | Documentation Index | Created centralized index |
| 2025-11-30 | Implementation Tracker | Consolidated all progress tracking |
| 2025-11-30 | Project Status | Added current state dashboard |
| 2025-11-30 | Hybrid Architecture | Marked as 100% complete |

---

## 🔗 External Resources

- [QuantConnect Documentation](https://www.quantconnect.com/docs)
- [LEAN Engine GitHub](https://github.com/QuantConnect/Lean)
- [Claude Code Documentation](https://docs.anthropic.com/claude-code)
- [Charles Schwab API](https://developer.schwab.com/)

---

**Last Reviewed**: December 1, 2025
**Next Review**: December 8, 2025
**Maintained By**: Claude Code Agent + Human Review
