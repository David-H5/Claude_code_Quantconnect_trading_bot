# Create Documentation Combo

Create a full documentation set: Research + Insight + Upgrade documents.

**Arguments**: `$ARGUMENTS` (project/feature name)

## Instructions

1. Parse the name from: `$ARGUMENTS`
2. Create three linked documents:
   - Research doc: `{NAME}-RESEARCH.md`
   - Insight doc: `{NAME}-INSIGHT.md`
   - Upgrade doc: `UPGRADE-{NNN}-{NAME}.md`
3. All docs are saved to `docs/research/`

## Quick Generation

Run this command to create all files:

```bash
python3 .claude/hooks/research_saver.py create combo "$ARGUMENTS"
```

## What Gets Created

| Document | Purpose | Location |
|----------|---------|----------|
| Research | Background research with timestamped sources | `docs/research/{NAME}-RESEARCH.md` |
| Insight | Key discoveries and learnings | `docs/insights/{NAME}-INSIGHT.md` |
| Upgrade | Implementation checklist with phases | `docs/research/UPGRADE-{NNN}-{NAME}.md` |

## Cross-References

All three documents include links to each other in their "Related" sections.

## Example Usage

```
/create-docs-combo "Options Scanner Enhancement"
/create-docs-combo "Multi-Agent Orchestration"
```

After running, you'll have:
- `docs/research/OPTIONS-SCANNER-ENHANCEMENT-RESEARCH.md`
- `docs/insights/OPTIONS-SCANNER-ENHANCEMENT-INSIGHT.md`
- `docs/research/UPGRADE-{NNN}-OPTIONS-SCANNER-ENHANCEMENT.md`
