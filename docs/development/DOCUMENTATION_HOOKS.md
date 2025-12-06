# Documentation Workflow Hooks & Scripts

**Version**: 1.0
**Created**: December 2, 2025
**Purpose**: Automated enforcement of research documentation standards

---

## Overview

This project uses Claude Code hooks to automatically enforce documentation standards. These hooks integrate with the RIC (Research-Introspection-Convergence) workflow to ensure all research is properly documented.

```
┌─────────────────────────────────────────────────────────────────┐
│                  DOCUMENTATION WORKFLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  WebSearch/WebFetch                                              │
│       │                                                          │
│       ▼                                                          │
│  document_research.py ──▶ Tracks searches, reminds after 3+     │
│       │                                                          │
│       ▼                                                          │
│  User creates research doc                                       │
│       │                                                          │
│       ▼                                                          │
│  validate_research.py ──▶ Validates naming & frontmatter        │
│       │                                                          │
│       ▼                                                          │
│  /ric-converge ──▶ Full validation before RIC exit              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Hooks

### 1. validate_research.py

**Location**: `.claude/hooks/validate_research.py`
**Trigger**: PostToolUse on Edit/Write operations
**Purpose**: Validates research documents after creation/modification

**What It Checks**:

| Check | Type | Description |
|-------|------|-------------|
| Naming convention | ERROR | Must match `[TOPIC]_RESEARCH.md` pattern |
| YAML frontmatter | ERROR | Must have frontmatter block |
| Required fields | WARNING | title, topic, tags, created |
| Valid topic | WARNING | Must be valid category |
| Search timestamps | WARNING | Must include `**Search Date**:` |
| Source dates | WARNING | Must include `(Published:` or `(Updated:` |
| Word count | WARNING | Minimum 500 words for research docs |

**Output Example**:

```
Research Doc Validation Failed: TEST_RESEARCH.md
  ERROR: Missing YAML frontmatter
  WARNING: No search timestamps found
  TIP: Add frontmatter with: title, topic, tags, created fields

Run: python scripts/validate_research_docs.py --fix for auto-fixes
See: docs/research/NAMING_CONVENTION.md for rules
```

**Configuration**: Configured in `.claude/settings.json`:

```json
{
  "PostToolUse": [
    {
      "matcher": "Edit|Write",
      "hooks": [
        {
          "type": "command",
          "command": "python3 .claude/hooks/validate_research.py",
          "statusMessage": "Validating research documentation"
        }
      ]
    }
  ]
}
```

---

### 2. document_research.py

**Location**: `.claude/hooks/document_research.py`
**Trigger**: PostToolUse on WebSearch/WebFetch operations
**Purpose**: Tracks web searches and reminds to document after 3+ searches

**How It Works**:

1. Tracks web operations in `/tmp/claude_research_tracker.json`
2. After 3+ searches without documentation, shows reminder
3. Resets counter when a research doc is written
4. Reminds with proper documentation format

**Reminder Threshold**: 3 web operations (configurable in script)

**Tracker State**:

```json
{
  "web_operations": 3,
  "last_doc_write": "2025-12-02T10:00:00",
  "session_start": "2025-12-02T09:00:00"
}
```

**Reset Triggers**:

- Writing to `docs/research/*.md`
- Editing `docs/research/*.md`

---

### 3. ric_hooks.py (Unified RIC Hook)

**Location**: `.claude/hooks/ric_hooks.py`
**Triggers**:
- PreToolUse on Edit/Write/Bash operations
- UserPromptSubmit for prompt complexity detection

**Purpose**: Unified RIC loop enforcement and suggestions

**Protected Paths**:

- `algorithms/`
- `execution/`
- `models/risk`
- `models/circuit`
- `llm/`
- `scanners/`

When modifying files in these paths, the hook reminds to use the RIC loop.
Also suggests RIC for complex prompts (multi-file, refactoring, etc.).

---

## Scripts

### 1. validate_research_docs.py

**Location**: `scripts/validate_research_docs.py`
**Purpose**: Full validation of all research documents

**Usage**:

```bash
# Basic validation
python scripts/validate_research_docs.py

# Verbose output
python scripts/validate_research_docs.py --verbose

# Auto-fix issues
python scripts/validate_research_docs.py --fix

# JSON output for automation
python scripts/validate_research_docs.py --json
```

**Validation Checks**:

| Check | Description |
|-------|-------------|
| `naming_convention` | Filename matches valid patterns |
| `frontmatter_presence` | Has YAML frontmatter |
| `frontmatter_fields` | Has required fields |
| `frontmatter_topic` | Topic is valid category |
| `cross_reference` | Internal links resolve |
| `orphaned_document` | Has incoming links |
| `upgrade_index_coverage` | Upgrade refs in index |
| `content_quality` | Minimum word count |
| `duplicate_links` | No excessive duplication |

**Exit Codes**:

- `0`: All checks passed
- `1`: Validation errors found
- `2`: Script error

---

### 2. create_research_doc.py

**Location**: `scripts/create_research_doc.py`
**Purpose**: Create new research documents with proper templates

**Usage**:

```bash
# Create research document
python scripts/create_research_doc.py "Evaluation Framework" --topic evaluation

# With upgrade reference
python scripts/create_research_doc.py "LLM Sentiment" --topic llm --upgrade UPGRADE-014

# Create summary document
python scripts/create_research_doc.py "Integration" --topic integration --type summary

# Create upgrade guide
python scripts/create_research_doc.py "Agent Integration" --topic agents --type guide
```

**Valid Topics**:

- `quantconnect`
- `evaluation`
- `llm`
- `workflow`
- `prompts`
- `autonomous`
- `agents`
- `sentiment`
- `integration`
- `general`

**Document Types**:

| Type | Naming Pattern | Template |
|------|----------------|----------|
| `research` | `[TOPIC]_RESEARCH.md` | Full research template |
| `summary` | `[TOPIC]_SUMMARY.md` | Executive summary |
| `guide` | `[TOPIC]_UPGRADE_GUIDE.md` | Implementation guide |

---

### 3. update_research_index.py

**Location**: `scripts/update_research_index.py`
**Purpose**: Auto-update cross-references in index files

**Usage**:

```bash
# Full update
python scripts/update_research_index.py

# Check only (no modifications)
python scripts/update_research_index.py --check

# Update after rename
python scripts/update_research_index.py --rename OLD_NAME.md NEW_NAME.md
```

**Updates**:

- `docs/research/README.md` - Quick reference tables
- Cross-references in all research docs

---

## Slash Commands

### /validate-docs

Run documentation validation with guidance for fixing issues.

### /ric-research

Start Phase 0 research with documentation protocol:

1. Initial search with timestamps
2. Keyword expansion
3. Expanded searches
4. Synthesize findings
5. Create document with `create_research_doc.py`
6. Validate with `validate_research_docs.py`

### /ric-converge

Phase 8 convergence check including **mandatory** documentation validation before exit.

---

## Integration with RIC Loop

The documentation workflow is integrated with the Enhanced RIC Loop:

| RIC Phase | Documentation Action |
|-----------|---------------------|
| Phase 0 (Research) | Create research doc, timestamp sources |
| Phase 3 (Coding) | Document implementation decisions |
| Phase 5 (Introspection) | Check for missing documentation |
| Phase 8 (Converge) | **Mandatory** validation before exit |

---

## Troubleshooting

### Hook Not Running

Check `.claude/settings.json` has the hook configured:

```bash
cat .claude/settings.json | grep -A5 "validate_research"
```

### Validation Failing

1. Check naming convention: `[TOPIC]_RESEARCH.md`
2. Add YAML frontmatter:
   ```yaml
   ---
   title: "Topic Research"
   topic: llm
   tags: [tag1, tag2]
   created: 2025-12-02
   ---
   ```
3. Add timestamps:
   ```markdown
   **Search Date**: December 2, 2025 at 10:00 AM EST
   [Source (Published: Nov 2025)](URL)
   ```

### Reminder Not Resetting

Ensure you're writing to `docs/research/*.md` with a `.md` extension.

Check tracker state:

```bash
cat /tmp/claude_research_tracker.json
```

---

## Related Documents

- [NAMING_CONVENTION.md](../research/NAMING_CONVENTION.md) - Naming rules
- [UPGRADE_INDEX.md](../research/UPGRADE_INDEX.md) - Upgrade mapping
- [ENHANCED_RIC_WORKFLOW.md](ENHANCED_RIC_WORKFLOW.md) - RIC workflow
- [CLAUDE.md](../../CLAUDE.md) - Main instructions

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-02 | Initial documentation created |
| 2025-12-02 | Fixed path handling bug in document_research.py |
