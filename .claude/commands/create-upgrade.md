# Create Upgrade Document

Create a new upgrade checklist with phases and task tracking.

**Arguments**: `$ARGUMENTS` (title for the upgrade)

## Instructions

1. Parse the title from: `$ARGUMENTS`
2. Get the next available upgrade number
3. Generate filename: `UPGRADE-{NNN}-{TITLE}.md`
4. Use the streamlined upgrade template
5. Save to `docs/research/`

## Template Location

The upgrade template is at `.claude/templates/upgrade_template.md`

## Quick Generation

Run this command to create the file:

```bash
python3 .claude/hooks/research_saver.py create upgrade "$ARGUMENTS"
```

## Template Features

- **Agent**: Claude Code header
- **Mandatory timestamping**: Research phase timestamps
- **Clear task tracking**: ⬜ 🔄 ✅ ⏸️ status symbols
- **Phased implementation**: Foundation → Core → Testing → Docs
- **Progress tracking table**
- **Definition of Done checklists**
- **Rollback plan**

## Example Usage

```
/create-upgrade "Multi-Agent Orchestration"
/create-upgrade "Options Scanner Enhancement"
```

After running, the document will be at:
`docs/research/UPGRADE-{NNN}-{TITLE}.md`

The upgrade number is automatically assigned based on existing upgrades.
