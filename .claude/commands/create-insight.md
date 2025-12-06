# Create Insight Document

Create a new insight document to capture a discovery or learning.

**Arguments**: `$ARGUMENTS` (title for the insight)

## Instructions

1. Parse the title from: `$ARGUMENTS`
2. Generate filename: `INSIGHT-{DATE}-{TITLE}.md`
3. Use the streamlined insight template
4. Save to `docs/insights/` (create if needed)

## Template Location

The insight template is at `.claude/templates/insight_template.md`

## Quick Generation

Run this command to create the file:

```bash
python3 .claude/hooks/research_saver.py create insight "$ARGUMENTS"
```

## Template Features

- **Agent**: Claude Code header
- **Mandatory timestamping**: Discovery timestamp and source publication dates
- **One-Line Summary**: Quick capture of the insight
- **Evidence table**: Sources with publication dates
- **Confidence assessment**: Source reliability and reproducibility ratings
- **Application section**: How to apply, where to apply, code examples
- **Caveats**: When NOT to apply

## Example Usage

```
/create-insight "Greeks Use IV Not Historical Vol"
/create-insight "ComboOrders Supported on Schwab"
```

After running, the document will be at:
`docs/insights/INSIGHT-{DATE}-{TITLE}.md`

Use insights to capture important discoveries that should be preserved for future reference.
