# QA Validation

Run comprehensive quality assurance checks on the codebase (53 checks across 16 categories).

## Usage

```
/qa-validate [options]
```

## Options

- `--fix` - Auto-fix issues where possible
- `--check CATEGORY` - Run specific category (see categories below)
- `--verbose` - Show all issues including info level
- `--upgrade UPGRADE-XXX` - Focus on specific upgrade
- `--json` - Output as JSON for automation

## Check Categories

| Category | Checks | Purpose |
|----------|--------|---------|
| `code` | syntax, imports, ruff, types | Code quality |
| `docs` | research_docs, docstrings, readme | Documentation |
| `tests` | tests_pass, coverage, naming | Test validation |
| `git` | uncommitted, large_files, secrets | Git hygiene |
| `files` | temp_files, empty, duplicates | File cleanliness |
| `progress` | format, session_summaries | Progress tracking |
| `debug` | breakpoints, todo, incomplete | Debug artifacts |
| `integrity` | corrupt, imports, circular | Code integrity |
| `xref` | broken_imports, config, links | Cross-references |
| `ric` | phases, upgrade_docs, iterations | RIC compliance |
| `security` | bandit, complexity | Security vulnerabilities |
| `hooks` | exist, syntax, registration, settings | Claude Code hooks |
| `trading` | risk_params, algorithm_structure, paper_mode | Trading safety |
| `config` | config_schema, env_vars, mcp_config | Configuration validation |
| `deps` | requirements_sync, version_conflicts, outdated | Dependency management |
| `agents` | personas_exist, persona_format, commands_valid | Agent personas/commands |

## All 53 Checks

### Code Quality (4)
- `python_syntax` - Syntax errors
- `python_imports` - Import validation
- `ruff_lint` - Linting (E, F rules)
- `type_hints` - Type hint coverage

### Documentation (3)
- `research_docs` - Research doc validation
- `docstrings` - Docstring presence
- `readme_exists` - README files

### Tests (3)
- `tests_pass` - Test suite passes
- `test_coverage` - Coverage config
- `test_naming` - File naming

### Git Hygiene (3)
- `uncommitted_changes` - Modified files
- `large_files` - Files >10MB
- `secrets_check` - Potential secrets

### File Cleanliness (3)
- `temp_files` - .pyc, .swp, etc.
- `empty_files` - Empty Python files
- `duplicate_files` - Duplicate names

### Progress Tracking (2)
- `progress_format` - Required sections
- `session_summaries` - Cleanup needed

### Debug Diagnostics (5)
- `debug_statements` - pdb/ipdb
- `todo_fixme` - TODO/FIXME/BUG
- `print_statements` - Excessive prints
- `breakpoints` - breakpoint() calls
- `incomplete_code` - NotImplementedError

### Code Integrity (5)
- `corrupt_files` - Null bytes, truncation
- `missing_imports` - Undefined names
- `circular_imports` - Star imports
- `orphan_functions` - Unused code
- `init_files` - Missing __init__.py

### Cross-References (4)
- `broken_imports` - Bad import paths
- `config_refs` - Invalid config keys
- `class_refs` - Unknown type hints
- `doc_links` - Broken doc links

### RIC Compliance (3)
- `ric_phases` - Phase coverage
- `upgrade_docs` - Doc completeness
- `iteration_tracking` - Iteration markers

### Security (2)
- `security_bandit` - Bandit security scanner
- `security_complexity` - Code complexity analysis

### Hooks Validation (4)
- `hooks_exist` - Required hooks present
- `hooks_syntax` - Hook files compile
- `hooks_registration` - Hooks in settings.json
- `hooks_settings` - Best practices and security

### Trading Safety (3)
- `risk_params` - Risk parameters in config (max_position_size, max_daily_loss_pct, max_drawdown)
- `algorithm_structure` - QuantConnect algorithms have Initialize/OnData methods
- `paper_mode_default` - Paper trading mode is default (no hardcoded live mode)

### Configuration Validation (3)
- `config_schema` - settings.json has expected sections
- `env_vars` - .env.example exists with required variables
- `mcp_config` - .mcp.json is valid with mcpServers

### Dependencies Management (3)
- `requirements_sync` - requirements.txt and pyproject.toml in sync
- `version_conflicts` - No version conflicts in dependencies
- `outdated_packages` - Security-critical packages checked

### Agent Personas/Commands (3)
- `personas_exist` - Required agent personas exist (.claude/agents/)
- `persona_format` - Personas have Role/Expertise/Responsibilities sections
- `commands_valid` - Slash commands have content and usage sections

## Auto-Fix Capabilities

The `--fix` flag automatically:
- Cleans temp files (.pyc, __pycache__)
- Removes accumulated session summaries
- Runs `ruff check --fix`

## RIC Loop Integration

Use at different RIC phases:

| Phase | Command |
|-------|---------|
| Phase 4 (Double-Check) | `--check debug --check integrity` |
| Phase 5 (Introspection) | `--check xref --verbose` |
| Phase 6 (Metacognition) | `--check ric` |
| Phase 7 (Exit) | `--verbose` (all checks) |
| Pre-Deploy | `--check trading --check config` |

## Examples

```bash
# Run all 53 checks
/qa-validate

# Run with auto-fix
/qa-validate --fix

# Check only code quality
/qa-validate --check code

# Debug and integrity
/qa-validate --check debug

# Focus on upgrade
/qa-validate --upgrade UPGRADE-014

# Verbose output
/qa-validate --verbose

# Trading safety checks
/qa-validate --check trading

# Configuration validation
/qa-validate --check config

# Dependency checks
/qa-validate --check deps

# Agent personas/commands
/qa-validate --check agents
```

---

Run the QA validator now:

```bash
python3 scripts/qa_validator.py $ARGUMENTS
```
