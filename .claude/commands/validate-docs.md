# Validate Research Documentation

Run documentation validation checks and fix any issues found.

## Run Validation

Execute the validation script:

```bash
python scripts/validate_research_docs.py --verbose
```

## If Issues Found

### Naming Convention Violations

If documents have incorrect names (e.g., `UPGRADE_NNN_` prefix):

1. Rename using proper convention: `[TOPIC]_RESEARCH.md`
2. Update cross-references:
   ```bash
   python scripts/update_research_index.py --rename OLD_NAME.md NEW_NAME.md
   ```
3. Update frontmatter to include `related_upgrades: [UPGRADE-NNN]`

### Missing Frontmatter

Add YAML frontmatter to research documents:

```yaml
---
title: "Topic Research"
topic: [quantconnect|evaluation|llm|workflow|prompts|autonomous|agents|sentiment|integration|general]
related_upgrades: [UPGRADE-NNN]
related_docs: []
tags: [tag1, tag2]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

Or use auto-fix:
```bash
python scripts/validate_research_docs.py --fix
```

### Missing Cross-References

Update index files:
```bash
python scripts/update_research_index.py
```

### Broken Links

Check for and fix broken links in documents by verifying the linked files exist.

## Quick Reference

| Script | Purpose |
|--------|---------|
| `python scripts/validate_research_docs.py` | Full validation |
| `python scripts/validate_research_docs.py --fix` | Auto-fix where possible |
| `python scripts/validate_research_docs.py --json` | JSON output for automation |
| `python scripts/update_research_index.py` | Update cross-references |
| `python scripts/create_research_doc.py "Title" --topic X` | Create new doc |

## Validation Checklist

- [ ] All research docs named `[TOPIC]_RESEARCH.md` (not `UPGRADE_NNN_...`)
- [ ] YAML frontmatter present with required fields
- [ ] Cross-references in README.md updated
- [ ] UPGRADE_INDEX.md updated (if upgrade-related)
- [ ] No broken internal links
- [ ] Sources have timestamps (search date + publication date)
