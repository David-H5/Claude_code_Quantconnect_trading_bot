# Agent Personas

Specialized agent personas for different development tasks. Use these with the Task tool for focused expertise.

## Available Personas

| Persona | File | Use Case |
|---------|------|----------|
| **Senior Engineer** | `.claude/agents/senior-engineer.md` | Complex implementation, architecture, code quality |
| **Risk Reviewer** | `.claude/agents/risk-reviewer.md` | Trading safety review, risk validation, compliance |
| **Strategy Dev** | `.claude/agents/strategy-dev.md` | Strategy design, options expertise, backtest analysis |
| **Code Reviewer** | `.claude/agents/code-reviewer.md` | PR reviews, security audit, code quality checks |
| **QA Engineer** | `.claude/agents/qa-engineer.md` | Test design, coverage analysis, quality gates |
| **Researcher** | `.claude/agents/researcher.md` | Technical research, market analysis, documentation |
| **Backtest Analyst** | `.claude/agents/backtest-analyst.md` | Performance analysis, overfitting detection |

## Usage Examples

```bash
# Use senior engineer for complex implementation
# In Claude Code, invoke via Task tool with appropriate prompt

# Code review before PR
# Use code-reviewer persona for security and quality checks

# Risk assessment for trading changes
# Use risk-reviewer persona for safety validation
```

## Persona Selection Guide

| Task Type | Recommended Persona |
|-----------|---------------------|
| New feature implementation | senior-engineer |
| Trading code changes | risk-reviewer -> senior-engineer |
| Strategy development | strategy-dev |
| Pre-commit review | code-reviewer |
| Test creation | qa-engineer |
| Technical research | researcher |
| Backtest evaluation | backtest-analyst |

## When to Use Multiple Personas

For complex tasks, combine personas in sequence:

1. **Research Phase**: Use `researcher` to investigate approach
2. **Implementation Phase**: Use `senior-engineer` to build
3. **Review Phase**: Use `code-reviewer` + `risk-reviewer` for validation
4. **Analysis Phase**: Use `backtest-analyst` to evaluate results

## Creating Custom Personas

Add new personas to `.claude/agents/` with:

```markdown
# Persona Name

## Role
Brief description of the persona's expertise.

## Primary Focus
- Key responsibility 1
- Key responsibility 2

## Guidelines
Specific instructions for this persona.

## Output Format
Expected output structure.
```
