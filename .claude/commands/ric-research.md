# RIC Loop Phase 0: Research Mode (v3.1)

You are executing Phase 0 (Research) of the Meta-RIC Loop v3.0.

## Research Topic

$ARGUMENTS

## FIRST: Log Phase Header

**Output immediately**:
```
[ITERATION X/5] === PHASE 0: RESEARCH ===
```

**Check current state**:
```bash
python3 .claude/hooks/ric_state_manager.py status
```

## MANDATORY: Follow Research Protocol

### Step 1: Initial Search

Conduct an online search on the topic. Document:

```markdown
### Phase 0: Research

**Search Date**: [YYYY-MM-DD at HH:MM TZ]
**Initial Query**: "[exact search query]"

**Key Sources Found**:
1. [Source Title (Published: Month Year)](URL)
   - Key insight: [summary]
   - Confidence: [HIGH/MEDIUM/LOW]
```

### Step 2: Keyword Expansion

Extract keywords from initial results:
- Technical terms discovered
- Related concepts
- Alternative terminology

### Step 3: Expanded Search (2-3 Additional Searches)

Using expanded keywords, conduct additional searches:

```markdown
**Search Date**: [YYYY-MM-DD at HH:MM TZ]
**Expanded Query**: "[new search query with keywords]"
**Keywords Used**: [term1, term2, term3]

**Additional Sources**:
1. [Source (Published: Date)](URL)
   - Key insight: [summary]
```

### Step 4: Synthesize Findings

Combine findings into actionable ideas:

```markdown
## Synthesized Ideas

Based on research, the following approaches are recommended:

1. **[Idea 1]**: Combines [Source A] + [Source B]
   - Pros: [list]
   - Cons: [list]
   - Confidence: [%]

2. **[Idea 2]**: Extends [Source C]
   - Pros: [list]
   - Cons: [list]
   - Confidence: [%]

## Contradictions Found
- [Source X] says [A], but [Source Y] says [B]
- Resolution: [how to handle]

## Research Gaps
- [What still needs investigation]
```

### Step 5: Save Research Document

**Use the automated document generator:**

```bash
python scripts/create_research_doc.py "Topic Name" --topic [category] --upgrade UPGRADE-NNN
```

**Valid topic categories**: quantconnect, evaluation, llm, workflow, prompts, autonomous, agents, sentiment, integration, general

**Naming Convention** (see `docs/research/NAMING_CONVENTION.md`):

- Name by SUBJECT, not upgrade number: `[TOPIC]_RESEARCH.md`
- Add upgrade numbers to frontmatter, not filename
- Update `docs/research/UPGRADE_INDEX.md` to map upgrades to research

**Manual creation**: If not using the script, save to `docs/research/[TOPIC]_RESEARCH.md` with YAML frontmatter:

```yaml
---
title: "Topic Research"
topic: [category]
related_upgrades: [UPGRADE-NNN]
tags: [tag1, tag2]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

### Step 6: Validate and Cross-Reference

Run validation before proceeding:

```bash
python scripts/validate_research_docs.py
```

Update cross-references:

1. Add to `docs/research/README.md` quick reference table
2. Update `docs/research/UPGRADE_INDEX.md` if upgrade-related
3. Link from related documents

## CRITICAL: Timestamping Requirements

**Every source MUST have**:
1. **Search Date**: When YOU searched (e.g., "December 2, 2025 at 9:30 AM EST")
2. **Publication Date**: When source was published (e.g., "(Published: Feb 2025)")

If publication date unknown, estimate: `(Published: ~2025)` or `(Published: Unknown)`

## Gate: Proceed to Phase 1 When

- [ ] At least 2 search iterations completed
- [ ] All sources have timestamps
- [ ] Keywords expanded from initial findings
- [ ] Ideas synthesized (not just listed)
- [ ] Research document saved with proper naming (`[TOPIC]_RESEARCH.md`)
- [ ] YAML frontmatter added with topic and tags
- [ ] Validation passed (`python scripts/validate_research_docs.py`)
- [ ] Cross-references updated (README.md, UPGRADE_INDEX.md if applicable)

## Next Step

After completing research:

1. **Advance to Phase 1**:
   ```bash
   python3 .claude/hooks/ric_state_manager.py advance
   ```

2. **Log Phase 1 Header**:
   ```
   [ITERATION X/5] === PHASE 1: UPGRADE PATH ===
   ```

3. Continue to Phase 1 (Upgrade Path) to define target state and success criteria.
