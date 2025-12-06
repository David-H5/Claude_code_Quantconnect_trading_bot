# Documentation Domain Context

You are working on **documentation, research docs, or README** files.

## Documentation Structure

```text
docs/
├── README.md              # Documentation index
├── PROJECT_STATUS.md      # Current state dashboard
├── IMPLEMENTATION_TRACKER.md  # Sprint tasks
├── architecture/          # Architecture docs
├── autonomous-agents/     # Autonomous guide
├── development/           # Best practices, standards
├── infrastructure/        # Setup, compute nodes
├── processes/            # RCA, postmortem
├── research/             # Research documents
└── strategies/           # Trading strategies
```

## Research Document Naming

**Pattern**: `[TOPIC]_RESEARCH.md` (by subject, not upgrade number)

**Required Frontmatter**:

```markdown
# [Topic] Research

**Search Date**: December 3, 2025 at 10:30 AM EST
**Scope**: [What was researched]
**Focus**: [Specific areas]
**Result**: [Summary]
```

## Timestamping Requirements

**All research must include**:

1. **Search timestamp**: When search was conducted
2. **Source publication date**: When source was published

```markdown
**Key Sources**:
1. [Source Title (Published: Aug 2024)](URL)
2. [Source Title (Updated: Nov 2025)](URL)
```

## Validation

```bash
# Validate research docs
python scripts/validate_research_docs.py

# Create new research doc
python scripts/create_research_doc.py "Title" --topic X
```

## Key Files to Update

| When changing... | Update... |
|-----------------|-----------|
| Workflow | CLAUDE.md, ENHANCED_RIC_WORKFLOW.md |
| Architecture | ADR in docs/adr/ |
| Strategy | docs/strategies/ |
| Research | docs/research/README.md |

## Before Committing

- [ ] Naming convention followed
- [ ] Timestamps included
- [ ] Cross-references updated
- [ ] Frontmatter valid
