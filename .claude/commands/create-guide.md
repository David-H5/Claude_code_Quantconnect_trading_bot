# Create Guide Document

Create a new how-to guide with steps and examples.

**Arguments**: `$ARGUMENTS` (title for the guide)

## Instructions

1. Parse the title from: `$ARGUMENTS`
2. Generate filename: `{TITLE}-GUIDE.md`
3. Use the streamlined guide template
4. Save to `docs/guides/` (create if needed)

## Template Location

The guide template is at `.claude/templates/guide_template.md`

## Quick Generation

Run this command to create the file:

```bash
python3 .claude/hooks/research_saver.py create guide "$ARGUMENTS"
```

## Template Features

- **Agent**: Claude Code header
- **Difficulty rating**: Beginner | Intermediate | Advanced
- **Quick Start section**: Get running in minutes
- **Detailed Steps**: Step-by-step with expected output
- **Examples**: Basic and advanced use cases
- **Troubleshooting**: Common issues and fixes
- **Configuration reference table**

## Example Usage

```
/create-guide "Setting Up Paper Trading"
/create-guide "Using the Options Scanner"
```

After running, the document will be at:
`docs/guides/{TITLE}-GUIDE.md`

Fill in the steps, examples, and troubleshooting as you document the feature.
