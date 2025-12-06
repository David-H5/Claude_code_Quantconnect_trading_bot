# Research Document Naming Convention

**Version**: 2.0
**Created**: December 2, 2025
**Updated**: December 3, 2025
**Based On**: [IT Glue Best Practices](https://www.itglue.com/naming-conventions/), [DRY Principle](https://en.wikipedia.org/wiki/Don't_repeat_yourself)

---

## Core Principle: Single Source of Truth (SSOT)

> "Every piece of knowledge must have a single, unambiguous, authoritative representation within a system."
> — *The Pragmatic Programmer*

**Key Rules:**

1. **Name by SUBJECT, not by upgrade number** (for single-topic upgrades)
2. **Use category naming for multi-category upgrades** (e.g., UPGRADE-014-CAT3)
3. **No duplication** - use cross-references instead
4. **One document per topic** - aggregate research, don't fragment
5. **Cross-reference via index** - map upgrades to documents
6. **MANDATORY**: Create category docs BEFORE marking implementation complete

---

## Naming Convention

### Primary Format

```
[TOPIC]_RESEARCH.md                    # Single-topic research documents
[TOPIC]_SUMMARY.md                     # Executive summaries
UPGRADE-NNN-[TOPIC].md                 # Implementation checklists only
UPGRADE-NNN-CATN-[TOPIC]-RESEARCH.md   # Multi-category upgrade research
[TOPIC]_UPGRADE_GUIDE.md               # Implementation guides
```

### Examples

| Document Type | Naming Pattern | Example |
|---------------|----------------|---------|
| Research (Single-topic) | `[TOPIC]_RESEARCH.md` | `EVALUATION_FRAMEWORK_RESEARCH.md` |
| Research (Multi-category) | `UPGRADE-NNN-CATN-[TOPIC]-RESEARCH.md` | `UPGRADE-014-CAT3-FAULT-TOLERANCE-RESEARCH.md` |
| Summary | `[TOPIC]_SUMMARY.md` | `INTEGRATION_SUMMARY.md` |
| Upgrade Checklist | `UPGRADE-NNN-[TOPIC].md` | `UPGRADE-008-ENHANCED-RIC-LOOP.md` |
| Implementation Guide | `[TOPIC]_UPGRADE_GUIDE.md` | `EVALUATION_UPGRADE_GUIDE.md` |

### Multi-Category Upgrade Naming Rules

For upgrades with multiple categories (like UPGRADE-014 with 12 categories):

```
UPGRADE-014-AUTONOMOUS-AGENT-ENHANCEMENTS.md      # Main document (overview)
UPGRADE-014-CAT1-ARCHITECTURE-RESEARCH.md         # Category 1 research
UPGRADE-014-CAT2-OBSERVABILITY-RESEARCH.md        # Category 2 research
UPGRADE-014-CAT3-FAULT-TOLERANCE-RESEARCH.md      # Category 3 research
...
```

**Pattern**: `UPGRADE-NNN-CATN-[CATEGORY-NAME]-RESEARCH.md`

| Component | Description | Example |
|-----------|-------------|---------|
| `UPGRADE-NNN` | Upgrade number | `UPGRADE-014` |
| `CATN` | Category number | `CAT3` |
| `[CATEGORY-NAME]` | Descriptive name from main doc | `FAULT-TOLERANCE` |
| `-RESEARCH.md` | Document type suffix | Required |

### Topic Naming Rules

1. **Use SCREAMING_SNAKE_CASE** for consistency
2. **Be descriptive** - prefer `EVALUATION_FRAMEWORK` over `EVAL`
3. **Include domain** when ambiguous - `QC_INTEGRATION` vs `API_INTEGRATION`
4. **Avoid dates in filename** - use YAML frontmatter or changelog instead

---

## Document Structure with Metadata

Every research document MUST include YAML frontmatter:

```markdown
---
title: "Evaluation Framework Research"
topic: evaluation
related_upgrades: [UPGRADE-010, UPGRADE-015]
related_docs:
  - AUTONOMOUS_AGENT_UPGRADE_GUIDE.md
  - EVALUATION_UPGRADE_GUIDE.md
tags: [evaluation, testing, stockbench, classic]
created: 2025-12-01
updated: 2025-12-02
---

# Evaluation Framework Research

...content...
```

### Required Frontmatter Fields

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | Human-readable title |
| `topic` | string | Primary topic category |
| `related_upgrades` | list | Upgrade numbers this research supports |
| `related_docs` | list | Cross-referenced documents |
| `tags` | list | Searchable keywords |
| `created` | date | Creation date |
| `updated` | date | Last update date |

---

## Topic Categories

Use these standardized topic categories:

| Category | Prefix | Example Documents |
|----------|--------|-------------------|
| QuantConnect | `QC_` | `QC_INTEGRATION_RESEARCH.md` |
| Evaluation | `EVALUATION_` | `EVALUATION_FRAMEWORK_RESEARCH.md` |
| LLM/Agents | `LLM_` | `LLM_SENTIMENT_RESEARCH.md` |
| Workflow | `WORKFLOW_` | `WORKFLOW_MANAGEMENT_RESEARCH.md` |
| Prompts | `PROMPT_` | `PROMPT_ENHANCEMENTS_RESEARCH.md` |
| Autonomous | `AUTONOMOUS_` | `AUTONOMOUS_WORKFLOW_RESEARCH.md` |

---

## Cross-Referencing Rules

### DO: Use Links Instead of Duplication

```markdown
For sentiment analysis integration, see [LLM Sentiment Research](LLM_SENTIMENT_RESEARCH.md).
```

### DON'T: Copy Content Between Documents

```markdown
<!-- BAD: Duplicating content -->
## Sentiment Analysis (copied from LLM_SENTIMENT_RESEARCH.md)
...100 lines of duplicated content...
```

### Upgrade-to-Research Mapping

Instead of naming documents by upgrade number, maintain a mapping in `UPGRADE_INDEX.md`:

```markdown
| Upgrade | Primary Research | Related Docs |
|---------|------------------|--------------|
| UPGRADE-008 | AUTONOMOUS_WORKFLOW_RESEARCH.md | ENHANCED_RIC_WORKFLOW.md |
| UPGRADE-014 | LLM_SENTIMENT_RESEARCH.md | LLM_SENTIMENT_EXPANSION_RESEARCH.md |
```

---

## Migration Guide

### Converting Upgrade-Named Documents

If you have a document named `UPGRADE_014_LLM_SENTIMENT_RESEARCH.md`:

1. **Rename** to `LLM_SENTIMENT_RESEARCH.md`
2. **Add frontmatter** with `related_upgrades: [UPGRADE-014]`
3. **Update UPGRADE_INDEX.md** to map UPGRADE-014 to this document
4. **Update cross-references** in other documents

### Exception: Implementation Checklists

Upgrade checklists (task lists for implementing an upgrade) MAY keep the upgrade number:
- `UPGRADE-008-ENHANCED-RIC-LOOP.md` - checklist of tasks
- `UPGRADE-009-WORKFLOW-ENHANCEMENTS.md` - checklist of tasks

These are ACTION documents, not KNOWLEDGE documents.

---

## Validation & Enforcement

### Validation Scripts

Run the validation script before committing:

```bash
python scripts/validate_research_docs.py
```

The script checks:

- [ ] Naming convention compliance
- [ ] Required frontmatter fields
- [ ] Broken cross-references
- [ ] Orphaned documents (no incoming links)
- [ ] Duplicate content detection
- [ ] **Multi-category upgrade completeness** (NEW)

### Automated Enforcement Hooks

The following hooks enforce documentation requirements:

| Hook | Trigger | Purpose |
|------|---------|---------|
| `validate_category_docs.py` | Pre-commit | Validates category docs exist before commit |
| `document_research.py` | PostToolUse (WebSearch) | Reminds to document after 3+ searches |
| `check_progress_docs.py` | Progress file update | Validates docs exist before marking complete |

### Enforcement Rules for Multi-Category Upgrades

**CRITICAL**: For upgrades with multiple categories (e.g., UPGRADE-014):

1. **Before marking a category COMPLETE** in `claude-progress.txt`:
   - The corresponding research document MUST exist
   - Example: Before `[x] Category 3: Fault Tolerance`, ensure `UPGRADE-014-CAT3-FAULT-TOLERANCE-RESEARCH.md` exists

2. **During implementation**:
   - Create the category research doc FIRST
   - Document discoveries as you implement
   - Link to implementation files in the research doc

3. **For autonomous sessions**:
   - The `check_progress_docs.py` hook validates completeness
   - Session stop hook checks documentation status
   - Missing docs block session completion

### Creating Category Documents

Use the template generator:

```bash
python scripts/create_category_doc.py UPGRADE-014 3 "Fault Tolerance"
# Creates: docs/research/UPGRADE-014-CAT3-FAULT-TOLERANCE-RESEARCH.md
```

Or manually create with the template from `docs/research/templates/category_research_template.md`

---

## Related Documents

- [UPGRADE_INDEX.md](UPGRADE_INDEX.md) - Upgrade to research mapping
- [README.md](README.md) - Research index with topical organization
- [../development/ENHANCED_RIC_WORKFLOW.md](../development/ENHANCED_RIC_WORKFLOW.md) - RIC workflow integration

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-02 | Initial naming convention created |
