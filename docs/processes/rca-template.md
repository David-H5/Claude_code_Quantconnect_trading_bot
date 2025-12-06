# Root Cause Analysis Report

**Incident ID**: INC-XXX
**Date**: YYYY-MM-DD
**Severity**: P0/P1/P2/P3
**Status**: Draft | In Review | Complete
**Author**: [Name]
**Reviewer**: [Name]

---

## 1. Incident Summary

### What Happened

[Brief description of the incident in 2-3 sentences]

### Impact

| Metric | Value |
|--------|-------|
| Duration | X hours |
| Financial Impact | $X or N/A |
| Users/Trades Affected | X |
| Data Loss | Yes/No |

### Detection

| Question | Answer |
|----------|--------|
| How was it detected? | [Monitoring/User report/Testing] |
| Time to detection | X minutes |
| Who detected it? | [Person/System] |

---

## 2. Timeline

| Time (UTC) | Event |
|------------|-------|
| HH:MM | [First related event] |
| HH:MM | [Incident began] |
| HH:MM | [Incident detected] |
| HH:MM | [Investigation started] |
| HH:MM | [Root cause identified] |
| HH:MM | [Fix deployed] |
| HH:MM | [Incident resolved] |

---

## 3. Root Cause Analysis

### 5 Whys

| Level | Question | Answer |
|-------|----------|--------|
| Problem | [What was the problem?] | - |
| Why #1 | Why did this happen? | [Answer] |
| Why #2 | Why did that happen? | [Answer] |
| Why #3 | Why? | [Answer] |
| Why #4 | Why? | [Answer] |
| Why #5 | Why? | [Answer] |

### Root Cause

[Clear statement of the root cause]

### Root Cause Category

- [ ] People (Training, Communication, Manual Error)
- [ ] Process (Missing Steps, No Review, Unclear Procedure)
- [ ] Technology (Code Bug, Infrastructure, Dependencies)
- [ ] External (API, Market Conditions, Third-Party)

---

## 4. Contributing Factors

| Factor | Description | Impact |
|--------|-------------|--------|
| [Factor 1] | [Description] | [High/Medium/Low] |
| [Factor 2] | [Description] | [High/Medium/Low] |

---

## 5. Corrective Actions

### Immediate Actions (Done)

| Action | Owner | Status | Date |
|--------|-------|--------|------|
| [Emergency fix deployed] | [Name] | Done | YYYY-MM-DD |
| [Monitoring added] | [Name] | Done | YYYY-MM-DD |

### Short-Term Actions (This Sprint)

| Action | Owner | Status | Due Date |
|--------|-------|--------|----------|
| [Add regression test] | [Name] | Todo | YYYY-MM-DD |
| [Update documentation] | [Name] | Todo | YYYY-MM-DD |
| [Code review] | [Name] | Todo | YYYY-MM-DD |

### Long-Term Actions (Backlog)

| Action | Priority | Target |
|--------|----------|--------|
| [Architectural improvement] | P1 | Q1 2026 |
| [Process change] | P2 | Q1 2026 |

---

## 6. Prevention Measures

### Technical

- [ ] Regression test added: `tests/regression/test_INCXXX.py`
- [ ] Monitoring/alerting improved
- [ ] Code review checklist updated
- [ ] Automated validation added

### Process

- [ ] Documentation updated
- [ ] Runbook created/updated
- [ ] Training provided
- [ ] Review process enhanced

---

## 7. Regression Test

### Test Location

`tests/regression/test_historical_bugs.py::test_INCXXX_[description]`

### Test Description

```python
def test_INCXXX_short_description():
    """
    Regression test for INC-XXX: [Title]

    Root Cause: [Brief root cause]
    Fix: [Brief fix description]
    """
    # Test implementation
    pass
```

### Test Verified

- [ ] Test fails before fix
- [ ] Test passes after fix
- [ ] Test added to CI pipeline

---

## 8. Lessons Learned

### What Went Well

- [Positive aspect 1]
- [Positive aspect 2]

### What Could Be Improved

- [Improvement area 1]
- [Improvement area 2]

### Action Items for Future Prevention

- [Systemic improvement 1]
- [Systemic improvement 2]

---

## 9. References

| Reference | Link |
|-----------|------|
| Related PR | [#XXX](link) |
| Related Issue | [#XXX](link) |
| Related ADR | [ADR-XXXX](link) |
| Monitoring Dashboard | [Link](link) |
| Logs | [Link](link) |

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Author | [Name] | YYYY-MM-DD | [ ] |
| Reviewer | [Name] | YYYY-MM-DD | [ ] |
| Tech Lead | [Name] | YYYY-MM-DD | [ ] |

---

## Change Log

| Date | Change |
|------|--------|
| YYYY-MM-DD | Initial RCA created |
| YYYY-MM-DD | [Update description] |
