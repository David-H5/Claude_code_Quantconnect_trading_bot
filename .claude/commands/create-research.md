# Create Research Document

Create a new research document with timestamped sources.

**Arguments**: `$ARGUMENTS` (topic/title for the research)

## Instructions

1. Parse the topic from: `$ARGUMENTS`
2. Generate a research document filename: `{TOPIC}-RESEARCH.md`
3. Use the streamlined research template with mandatory timestamping
4. Save to `docs/research/`

## Template Location

The research template is at `.claude/templates/research_template.md`

## Quick Generation

Run this command to create the file:

```bash
python3 .claude/hooks/research_saver.py create research "$ARGUMENTS"
```

## Template Features

- **Agent**: Claude Code header
- **Mandatory timestamping**: Search timestamps and source publication dates
- **TIMESTAMPING RULES** block for compliance
- **Streamlined sections**: Overview, Research Questions, Phases, Discoveries, Best Practices, Anti-Patterns, Deliverables

## Example Usage

```
/create-research "QuantConnect Greeks API"
/create-research "AI Agent Evaluation Methods"
```

After running, the document will be at:
`docs/research/{TOPIC}-RESEARCH.md`

Fill in the research phases as you conduct web searches, ensuring all timestamps and source dates are recorded.
