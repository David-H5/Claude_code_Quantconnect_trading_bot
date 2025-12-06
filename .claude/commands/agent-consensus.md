# Agent Consensus - Multi-Agent Decision Making

Get multiple independent perspectives on a decision or question.

## Arguments
- `$ARGUMENTS`: Decision or question to evaluate

## Examples
```
/agent-consensus should we refactor the execution module
/agent-consensus is this trading strategy safe for production
/agent-consensus which caching approach should we use
/agent-consensus should we add async support to the scanners
```

## Instructions

Spawn 3 independent analyst agents to vote on the decision.

### Consensus Configuration

Each agent analyzes independently from a different perspective:

```python
# Analyst 1: Technical Feasibility
Task(
    subagent_type="general-purpose",
    model="sonnet",
    description="Technical analysis",
    prompt="""Analyze this decision from a TECHNICAL perspective: $ARGUMENTS

Consider:
1. Technical complexity
2. Implementation effort
3. System impact
4. Performance implications
5. Maintenance burden

After analysis, provide your vote:
- **APPROVE**: Technically sound, should proceed
- **REJECT**: Technical issues, should not proceed
- **NEEDS_MORE_INFO**: Cannot decide without more information

Format:
## Technical Analysis
[Your analysis]

## Vote: [APPROVE/REJECT/NEEDS_MORE_INFO]
## Confidence: [0-100]%
## Key Reasoning: [One sentence]"""
)

# Analyst 2: Risk Assessment
Task(
    subagent_type="general-purpose",
    model="sonnet",
    description="Risk analysis",
    prompt="""Analyze this decision from a RISK perspective: $ARGUMENTS

Consider:
1. What could go wrong?
2. Worst case scenarios
3. Mitigation strategies available
4. Reversibility of decision
5. Impact on trading safety

After analysis, provide your vote:
- **APPROVE**: Risks are acceptable/manageable
- **REJECT**: Risks are too high
- **NEEDS_MORE_INFO**: Cannot assess risks without more information

Format:
## Risk Analysis
[Your analysis]

## Vote: [APPROVE/REJECT/NEEDS_MORE_INFO]
## Confidence: [0-100]%
## Key Reasoning: [One sentence]"""
)

# Analyst 3: Strategic Value
Task(
    subagent_type="general-purpose",
    model="sonnet",
    description="Strategic analysis",
    prompt="""Analyze this decision from a STRATEGIC perspective: $ARGUMENTS

Consider:
1. Does this align with project goals?
2. What value does it add?
3. Opportunity cost of doing vs not doing
4. Long-term implications
5. Priority relative to other work

After analysis, provide your vote:
- **APPROVE**: Strategically valuable, should proceed
- **REJECT**: Not worth the investment
- **NEEDS_MORE_INFO**: Cannot assess value without more information

Format:
## Strategic Analysis
[Your analysis]

## Vote: [APPROVE/REJECT/NEEDS_MORE_INFO]
## Confidence: [0-100]%
## Key Reasoning: [One sentence]"""
)
```

### Consensus Calculation

After all agents respond:

1. **Count Votes**:
   - APPROVE = +1
   - REJECT = -1
   - NEEDS_MORE_INFO = 0

2. **Calculate Consensus Score**:
   - Score = (APPROVE count) / (Total non-abstain votes)
   - Threshold = 66% (2/3 majority)

3. **Determine Outcome**:
   - Score >= 66%: **APPROVED** (proceed recommended)
   - Score < 66% but > 33%: **CONTESTED** (discuss further)
   - Score <= 33%: **REJECTED** (do not proceed)
   - Multiple NEEDS_MORE_INFO: **INCONCLUSIVE** (gather more info)

### Result Format

```markdown
## Consensus Decision: $ARGUMENTS

### Voting Results
| Analyst | Vote | Confidence | Key Reasoning |
|---------|------|------------|---------------|
| Technical | {vote} | {conf}% | {reason} |
| Risk | {vote} | {conf}% | {reason} |
| Strategic | {vote} | {conf}% | {reason} |

### Consensus Score: {X}% ({APPROVED/REJECTED/CONTESTED/INCONCLUSIVE})

### Analysis Summary

**Technical Perspective:**
{Summary of technical analysis}

**Risk Perspective:**
{Summary of risk analysis}

**Strategic Perspective:**
{Summary of strategic analysis}

### Recommendation
{Final recommendation based on consensus}

### If Proceeding, Consider:
- {Key consideration 1}
- {Key consideration 2}
- {Key consideration 3}

### If Not Proceeding, Alternative:
- {Alternative approach}
```
