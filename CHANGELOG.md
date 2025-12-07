# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **BaseOptionsBot** base class ([algorithms/base_options_bot.py](algorithms/base_options_bot.py))
  - Shared initialization: config, risk management, circuit breaker, monitoring
  - Template Method Pattern with `_setup_strategy_specific()` hook
  - Common lifecycle methods: `OnEndOfAlgorithm()`
- **CheckIntervals** class for standardized timing constants
  - `STRATEGY = 300` (5 minutes)
  - `RECURRING = 3600` (1 hour)
  - `SENTIMENT = 300` (5 minutes)
  - `RESOURCE = 30` (30 seconds)
  - `CIRCUIT_BREAKER_LOG = 3600` (1 hour)
  - `NEWS_CACHE = 1800` (30 minutes)
- **LogPrefix** class in [utils/log_handlers.py](utils/log_handlers.py)
  - Standardized prefixes: `OK:`, `WARN:`, `ERR:`, `INFO:`
  - Additional: `ALERT:`, `TRADE:`, `CIRCUIT:`, `RESOURCE:`, `SENTIMENT:`
- Generic `_load_dataclass_config()` method in ConfigManager

### Changed
- **HybridOptionsBot** now extends BaseOptionsBot
  - Removed ~80 lines of duplicated initialization code
  - Overrides `_setup_basic()` for custom dates/cash
  - Implements `_setup_strategy_specific()` for LLM and hybrid components
- **OptionsTradingBot** now extends BaseOptionsBot
  - Removed duplicated config/risk/monitoring setup
  - Cleaner inheritance hierarchy
- Callback type hints improved:
  - `_on_profit_take_order(order: ProfitTakeOrder)`
  - `_on_order_event(order: ExecutionOrder, action: str)`
- Condensed initialization logging (11 lines -> 4-5 summary lines)
- Tests updated to reflect new architecture

### Fixed
- **P0-1 Critical**: Shared timestamp bug in HybridOptionsBot
  - `_should_check_strategies()` and `_should_check_recurring()` now use separate timestamps
  - Previously shared `_last_check_time` caused incorrect 300s vs 3600s intervals
- Circuit breaker callback signature: `details: dict` (was `urgency: str`)

### Removed
- Dead scheduled methods: `_scheduled_strategy_check`, `_scheduled_recurring_check`
- Commented butterfly code from OptionsTradingBot
- Unused type imports: `Dict`, `List`, `Set` from config/__init__.py
- Redundant imports in HybridOptionsBot (now inherited from base class)

### Documentation
- Added cross-reference in CLAUDE.md for RIC Loop sections
- Documented QuantConnectOptionsBot alias class purpose
- Clarified warmup comments (Greeks use IV, no warmup needed)

---

## FIX_GUIDE.md Phases Completed

| Phase | Items | Status |
|-------|-------|--------|
| Phase 1 (Critical) | P0-1 timestamp bug | Complete |
| Phase 2 (Architecture) | P1-1, P1-2, P1-3 refactoring | Complete |
| Phase 3 (Cleanup) | P2-1 through P2-5 | Complete |
| Phase 4 (Polish) | P3-1 through P3-5 | Complete |

All 19 algorithm tests passing.
