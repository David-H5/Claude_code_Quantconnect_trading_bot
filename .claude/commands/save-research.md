# Save Research - Intelligent Documentation Saver

Save research, insights, ideas, and upgrade guides to appropriate locations with proper naming conventions.

## Arguments
- `$ARGUMENTS`: Document type and title (e.g., "upgrade Multi-Agent Memory", "research LLM Patterns", "insight Context Optimization")

## Usage Examples

```bash
# Create a new upgrade guide (auto-assigns next available number)
/save-research upgrade "Multi-Agent Memory System"

# Create a research document
/save-research research "LLM Prompt Patterns"

# Create an insight document (auto-dated)
/save-research insight "Context Window Optimization"

# Create a how-to guide
/save-research guide "Agent Orchestration"

# Check what number is next
/save-research next

# List existing documents
/save-research list

# List only upgrades
/save-research list upgrade

# Check if a name is available
/save-research check "UPGRADE-018-SOMETHING.md"
```

## Document Types

| Type | Pattern | Directory | Purpose |
|------|---------|-----------|---------|
| `upgrade` | `UPGRADE-NNN-NAME.md` | `docs/research/` | Implementation checklists with phases and tasks |
| `research` | `NAME-RESEARCH.md` | `docs/research/` | Research findings and discoveries |
| `category` | `UPGRADE-NNN-CATN-NAME-RESEARCH.md` | `docs/research/` | Category-specific research for upgrades |
| `insight` | `INSIGHT-YYYY-MM-DD-NAME.md` | `docs/insights/` | Quick insights and discoveries |
| `guide` | `NAME-GUIDE.md` | `docs/guides/` | How-to guides and tutorials |

## Instructions

Based on the arguments provided, execute the appropriate action:

### For Document Creation

1. Parse the document type from arguments (upgrade, research, insight, guide)
2. Extract the title/name from the remaining arguments
3. Run the research saver tool:

```bash
python3 .claude/hooks/research_saver.py create --type TYPE --name "TITLE"
```

For upgrades with specific number:
```bash
python3 .claude/hooks/research_saver.py create --type upgrade --name "TITLE" --number 18
```

With additional options:
```bash
python3 .claude/hooks/research_saver.py create --type upgrade --name "TITLE" --priority P0 --effort "3-4 weeks"
```

### For Listing Documents

```bash
python3 .claude/hooks/research_saver.py list
python3 .claude/hooks/research_saver.py list --type upgrade
```

### For Checking Names

```bash
python3 .claude/hooks/research_saver.py check "FILENAME.md" --type upgrade
```

### For Getting Next Number

```bash
python3 .claude/hooks/research_saver.py next
```

### For Viewing Templates

```bash
python3 .claude/hooks/research_saver.py template upgrade
python3 .claude/hooks/research_saver.py template research
```

## After Document Creation

1. **Open the created file** and fill in the template placeholders
2. **Update the research index** at `docs/research/README.md` if applicable
3. **Link related documents** in the Related Documentation section
4. **Update CLAUDE.md** if the document introduces new patterns or commands

## Template Features

### Upgrade Template Includes:
- Overview with version, status, priority, effort
- Objectives with success criteria
- Research & References section with source tables
- Architecture diagrams (current vs target)
- Phased implementation checklist with:
  - Numbered tasks (1.1.1, 1.1.2, etc.)
  - Code examples per task
  - File paths and line estimates
  - Complexity ratings
  - Dependencies
- Files to create/modify tables
- Testing strategy with test scenarios
- Risk assessment matrix
- Progress tracking with completion checklist
- Design decisions documentation
- Migration guide
- Change log

### Research Template Includes:
- Research overview with scope and focus
- Research objectives
- Phased research with timestamped searches
- Source citations with publication dates
- Key discoveries with impact ratings
- Deliverables table
- Change log

### Insight Template Includes:
- Summary with category and impact
- Context and background
- Key finding with supporting evidence
- Application steps with examples
- Caveats and related work
- Follow-up actions

### Guide Template Includes:
- Prerequisites checklist
- Quick start steps
- Detailed usage sections
- Examples with expected output
- Troubleshooting section
- References

## Automatic Features

The tool automatically:
- Assigns the next available upgrade number if not specified
- Validates filenames against naming conventions
- Checks for conflicts with existing files
- Creates necessary directories
- Generates comprehensive templates
- Suggests proper filenames from titles

## CLI Reference

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RESEARCH SAVER - Documentation Tool                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  COMMANDS:                                                                   ║
║    create    Create a new document from template                             ║
║    check     Check if a filename is valid and available                      ║
║    list      List existing documents                                         ║
║    next      Get next available upgrade number                               ║
║    template  Print a template to stdout                                      ║
║    validate  Validate a document has required sections                       ║
║    help      Show this help message                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
```
