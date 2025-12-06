# Root Cause Analysis Process

**Version**: 1.0
**Created**: December 1, 2025
**Status**: Active

---

## Purpose

Root Cause Analysis (RCA) is a systematic approach to identify the underlying causes of problems or incidents. This document defines when and how to perform RCA for the QuantConnect Trading Bot project.

---

## When to Perform RCA

### Required (Mandatory)

| Trigger | Description |
|---------|-------------|
| Production Incident | Any incident affecting live trading |
| Financial Loss | Any bug causing unintended financial loss |
| Critical Bug | P0/P1 bugs found in testing or production |
| Security Vulnerability | Any security issue discovered |
| Data Loss/Corruption | Any loss of trading data or state |

### Recommended (Optional)

| Trigger | Description |
|---------|-------------|
| Near Miss | Bugs caught just before deployment |
| Performance Degradation | >20% slowdown in trading operations |
| Integration Failures | Third-party API failures with impact |
| Repeated Issues | Same bug type occurring 3+ times |

---

## The 5 Whys Method

The 5 Whys technique helps drill down to root causes by asking "Why?" iteratively.

### Process

1. **State the Problem**: Clearly describe what happened
2. **Ask "Why?"**: Identify the immediate cause
3. **Ask "Why?" Again**: Identify the cause of the immediate cause
4. **Continue 3+ More Times**: Keep asking until you reach the root cause
5. **Identify Corrective Action**: Define how to prevent recurrence

### Example

| Level | Question | Answer |
|-------|----------|--------|
| Problem | Order executed at wrong price | - |
| Why #1 | Why wrong price? | Stale price data was used |
| Why #2 | Why stale data? | Price update handler wasn't called |
| Why #3 | Why not called? | WebSocket reconnection failed silently |
| Why #4 | Why silent failure? | No error logging for reconnection failures |
| Why #5 | Why no logging? | Reconnection logic was added hastily without tests |
| **Root Cause** | Missing test coverage for WebSocket reconnection error paths |
| **Fix** | Add reconnection tests and error logging |

---

## Fishbone Diagram (Ishikawa)

For complex issues, use a Fishbone diagram to categorize potential causes:

```
                    ┌─────────────────┐
     ┌──────────────┤    PROBLEM      │
     │              └────────┬────────┘
     │                       │
┌────┴────┐  ┌───────┐  ┌───┴───┐  ┌────────┐
│ PEOPLE  │  │PROCESS│  │TECH   │  │EXTERNAL│
└─┬───┬───┘  └──┬──┬─┘  └─┬──┬──┘  └──┬──┬──┘
  │   │         │  │      │  │        │  │
  │   │         │  │      │  │        │  │
Training        │  │    Code │      API  │
   │        Review │      │  │    Latency│
 Error        │  │    Tests  │        │
 Handling  Process│      │  Data   Market
              │      Memory  │    Conditions
            Steps      │   Cache
                      │
                   Libraries
```

### Categories

| Category | Examples |
|----------|----------|
| **People** | Training gaps, manual errors, communication |
| **Process** | Missing steps, unclear procedures, no review |
| **Technology** | Code bugs, infrastructure, dependencies |
| **External** | APIs, market conditions, regulations |

---

## RCA Template

Use the [RCA Template](rca-template.md) for all root cause analyses.

### Required Sections

1. **Incident Summary** - What happened, when, impact
2. **Timeline** - Detailed sequence of events
3. **5 Whys Analysis** - Root cause identification
4. **Contributing Factors** - Secondary causes
5. **Corrective Actions** - Immediate and long-term fixes
6. **Prevention Measures** - How to prevent recurrence
7. **Regression Test** - Link to test preventing recurrence

---

## Post-RCA Actions

After completing an RCA, the following actions are **required**:

### 1. Create Regression Test

Every RCA must result in at least one regression test:

```python
# tests/regression/test_historical_bugs.py

def test_INC001_websocket_reconnection_logging():
    """
    Regression test for INC-001: Silent WebSocket reconnection failure.

    Root Cause: Missing error logging for WebSocket reconnection.
    Fix: Added error logging and reconnection retry logic.
    RCA: docs/incidents/INC-001.md
    """
    # Test that reconnection failures are logged
    with mock_websocket_failure():
        connector = WebSocketConnector()
        connector.connect()

        # Should log error on failure
        assert "reconnection failed" in caplog.text

        # Should retry
        assert connector.retry_count > 0
```

### 2. Update Documentation

- Update relevant ADRs if architectural changes made
- Update CLAUDE.md if process changes needed
- Add to incident log (`docs/incidents/README.md`)

### 3. Track Metrics

Add to defect metrics in PROJECT_STATUS.md:

| Metric | What to Track |
|--------|---------------|
| MTTR | Mean time to resolve |
| Root Cause Category | People/Process/Tech/External |
| Recurrence | Has this type of bug occurred before? |

### 4. Review in Retrospective

Schedule RCA review in next team retrospective:

- Were corrective actions effective?
- Are there systemic issues?
- Should processes change?

---

## RCA Storage

All RCA documents are stored in:

```
docs/incidents/
├── README.md          # Incident index
├── template.md        # Incident report template
├── INC-001.md         # Individual incident RCAs
├── INC-002.md
└── ...
```

---

## Incident Severity Levels

| Level | Description | RCA Required | Timeline |
|-------|-------------|--------------|----------|
| P0 | Critical - Trading halted | Yes | Within 24h |
| P1 | High - Financial impact | Yes | Within 48h |
| P2 | Medium - Degraded service | Recommended | Within 1 week |
| P3 | Low - Minor issue | Optional | Best effort |

---

## Escalation Path

```
Developer → Tech Lead → Risk Manager → Human Approval Gate
    ↓           ↓            ↓               ↓
 Fix Bug    Review RCA   Approve Fix    Resume Trading
```

---

## Tools and Resources

| Tool | Purpose |
|------|---------|
| Git blame | Identify when/who introduced code |
| Git bisect | Find exact commit causing regression |
| Logging | Review application logs |
| Metrics | Check performance dashboards |
| Tests | Verify fix with new tests |

---

## Best Practices

1. **Blameless Culture**: Focus on systems, not individuals
2. **Document Everything**: Thorough timeline and analysis
3. **Share Learnings**: Distribute RCA findings to team
4. **Follow Through**: Verify corrective actions are implemented
5. **Track Patterns**: Look for recurring root causes
6. **Time-Box**: Complete RCA within severity timeline

---

## Change Log

| Date | Version | Change |
|------|---------|--------|
| 2025-12-01 | 1.0 | Initial RCA process created |
