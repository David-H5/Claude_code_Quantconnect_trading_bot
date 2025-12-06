# Create Documentation

Create a new documentation file from a template.

**Arguments**: `$ARGUMENTS` (format: `<type> "<title>"`)

## Available Document Types

| Type | Command | Description |
|------|---------|-------------|
| `research` | `/create-research "Topic"` | Research doc with timestamped sources |
| `upgrade` | `/create-upgrade "Title"` | Upgrade checklist with phases |
| `guide` | `/create-guide "Title"` | How-to guide with examples |
| `insight` | `/create-insight "Title"` | Discovery/insight documentation |
| `combo` | `/create-docs-combo "Name"` | **Full set**: Research + Insight + Upgrade |

## Quick Usage

```bash
/create-doc research "QuantConnect API"
/create-doc upgrade "New Feature"
/create-doc guide "Setup Instructions"
/create-doc insight "Important Discovery"
/create-docs-combo "My Project"
```

## CLI Alternative

```bash
# Create any document type
python3 .claude/hooks/research_saver.py create <type> "<title>"

# Examples
python3 .claude/hooks/research_saver.py create research "API Research"
python3 .claude/hooks/research_saver.py create upgrade "Feature X"
python3 .claude/hooks/research_saver.py create guide "Setup Guide"
python3 .claude/hooks/research_saver.py create insight "Key Finding"
python3 .claude/hooks/research_saver.py create combo "Project Name"  # Creates all 3

# List existing documents
python3 .claude/hooks/research_saver.py list

# Get next upgrade number
python3 .claude/hooks/research_saver.py next

# Validate a document
python3 .claude/hooks/research_saver.py validate docs/research/FILE.md upgrade
```

## Template Locations

All templates are in `.claude/templates/`:

- `research_template.md` - Research documentation
- `upgrade_template.md` - Upgrade checklists
- `guide_template.md` - How-to guides
- `insight_template.md` - Insights/discoveries

## Natural Language Detection

The system automatically detects when you ask to create or update documentation:

### Create Keywords

| You Say | Detected As |
|---------|-------------|
| "create research doc" | Research |
| "research document for X" | Research |
| "create upgrade guide" | Upgrade |
| "implementation checklist" | Upgrade |
| "create a guide for" | Guide |
| "how-to guide" | Guide |
| "document this insight" | Insight |
| "create insight" | Insight |
| **"create documents"** | **Combo (all 3)** |
| **"make documents"** | **Combo (all 3)** |
| "full documentation" | Combo (all 3) |
| "project documentation" | Combo (all 3) |

### Update Keywords

| You Say | Action |
|---------|--------|
| **"update research doc"** | Shows recent research docs to update |
| **"update insight doc"** | Shows recent insight docs to update |
| **"update upgrade guide"** | Shows recent upgrade docs to update |
| **"update guide"** / **"update guides"** | Shows recent guides to update |
| **"update document"** / **"update documents"** | Shows ALL recent docs across types |

When update is detected, you'll see:
- List of recent files (sorted by modification time)
- Instructions for proper updates (timestamps, changelog)
- Template-specific requirements

When detected, you'll see a suggestion with the appropriate command.
