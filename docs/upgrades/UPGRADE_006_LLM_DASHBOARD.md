# Upgrade Path: LLM Dashboard Integration

**Upgrade ID**: UPGRADE-006
**Iteration**: 1
**Date**: December 1, 2025
**Status**: ✅ Complete

---

## Target State

Integrate LLM agent capabilities into the existing UI dashboard:

1. **Agent Metrics Widget**: Real-time performance metrics for trading agents
2. **Debate Visualization**: View Bull/Bear debate rounds and outcomes
3. **Evolution Monitor**: Track self-evolution progress and prompt versions
4. **Decision Log Viewer**: Browse and analyze agent decisions

---

## Scope

### Included

- Create `ui/agent_metrics_widget.py` for agent performance display
- Create `ui/debate_viewer.py` for debate visualization
- Create `ui/evolution_monitor.py` for evolution tracking
- Create `ui/decision_log_viewer.py` for decision browsing
- Update `ui/dashboard.py` to integrate new widgets
- Create tests for all new components
- Update CLAUDE.md with dashboard usage

### Excluded

- Real-time streaming updates (P2, defer to UPGRADE-007)
- External notification integration (P3, defer)
- Multi-user dashboard support (P3, defer)

---

## Success Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Agent metrics widget created | File exists | `ui/agent_metrics_widget.py` |
| Debate viewer created | File exists | `ui/debate_viewer.py` |
| Evolution monitor created | File exists | `ui/evolution_monitor.py` |
| Decision log viewer created | File exists | `ui/decision_log_viewer.py` |
| Widgets tested | Test count | ≥ 20 test cases |
| Dashboard updated | Integration complete | New tabs/panels added |
| CLAUDE.md updated | Sections added | Dashboard usage guide |

---

## Dependencies

- [x] UPGRADE-004 complete (Agent Metrics)
- [x] UPGRADE-005 complete (Self-Evolving Agents)
- [x] UI framework exists (`ui/dashboard.py`)
- [x] PySide6 available (existing dependency)

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PySide6 compatibility issues | Low | Medium | Use standard widgets |
| Performance with large datasets | Medium | Medium | Implement pagination |
| Dashboard layout complexity | Low | Low | Use tabbed interface |

---

## Estimated Effort

- Agent Metrics Widget: 2 hours
- Debate Viewer: 2 hours
- Evolution Monitor: 1.5 hours
- Decision Log Viewer: 1.5 hours
- Dashboard Integration: 1 hour
- Tests: 2 hours
- Documentation: 0.5 hour
- **Total**: ~10.5 hours

---

## Phase 2: Task Checklist

### Widget Components (T1-T4)

| ID | Task | Est. | Depends | Priority |
|----|------|------|---------|----------|
| T1 | Create `ui/agent_metrics_widget.py` | 60m | - | P0 |
| T2 | Create `ui/debate_viewer.py` | 60m | - | P0 |
| T3 | Create `ui/evolution_monitor.py` | 45m | - | P0 |
| T4 | Create `ui/decision_log_viewer.py` | 45m | - | P0 |

### Integration (T5-T6)

| ID | Task | Est. | Depends | Priority |
|----|------|------|---------|----------|
| T5 | Update `ui/dashboard.py` with new widgets | 45m | T1-T4 | P0 |
| T6 | Create `tests/test_llm_dashboard.py` | 60m | T1-T5 | P0 |

### Documentation (T7-T8)

| ID | Task | Est. | Depends | Priority |
|----|------|------|---------|----------|
| T7 | Update `ui/__init__.py` exports | 15m | T1-T4 | P0 |
| T8 | Update CLAUDE.md with dashboard guide | 30m | T5 | P0 |

---

## Phase 3: Implementation

### T1: Agent Metrics Widget

```python
# ui/agent_metrics_widget.py
class AgentMetricsWidget(QWidget):
    """Widget displaying agent performance metrics."""

    def __init__(self, metrics_tracker: AgentMetricsTracker):
        super().__init__()
        self.tracker = metrics_tracker
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Agent selector
        self.agent_combo = QComboBox()
        layout.addWidget(self.agent_combo)

        # Metrics display
        self.accuracy_label = QLabel("Accuracy: --")
        self.confidence_label = QLabel("Avg Confidence: --")
        self.calibration_label = QLabel("Calibration Error: --")
        self.decisions_label = QLabel("Total Decisions: --")

        layout.addWidget(self.accuracy_label)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.calibration_label)
        layout.addWidget(self.decisions_label)

        # Chart (if matplotlib available)
        self.chart = MetricsChart()
        layout.addWidget(self.chart)

    def refresh(self):
        """Refresh metrics display."""
        agent = self.agent_combo.currentText()
        if agent:
            metrics = self.tracker.get_metrics(agent)
            self._update_display(metrics)
```

### T2: Debate Viewer

```python
# ui/debate_viewer.py
class DebateViewer(QWidget):
    """Widget for viewing Bull/Bear debate sessions."""

    def __init__(self, debate_mechanism: BullBearDebate):
        super().__init__()
        self.debate = debate_mechanism
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Debate selector
        self.debate_combo = QComboBox()
        layout.addWidget(QLabel("Select Debate:"))
        layout.addWidget(self.debate_combo)

        # Rounds display (splitter for bull/bear)
        splitter = QSplitter(Qt.Horizontal)

        self.bull_panel = ArgumentPanel("Bull", "#4CAF50")
        self.bear_panel = ArgumentPanel("Bear", "#F44336")

        splitter.addWidget(self.bull_panel)
        splitter.addWidget(self.bear_panel)

        layout.addWidget(splitter)

        # Outcome display
        self.outcome_label = QLabel("Outcome: --")
        self.consensus_label = QLabel("Consensus: --")
        layout.addWidget(self.outcome_label)
        layout.addWidget(self.consensus_label)

    def load_debate(self, debate_id: str):
        """Load and display a specific debate."""
        ...
```

### T3: Evolution Monitor

```python
# ui/evolution_monitor.py
class EvolutionMonitor(QWidget):
    """Widget for monitoring agent evolution progress."""

    def __init__(self, evolving_agents: List[SelfEvolvingAgent]):
        super().__init__()
        self.agents = evolving_agents
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Agent selector
        self.agent_combo = QComboBox()
        layout.addWidget(self.agent_combo)

        # Evolution status
        status_group = QGroupBox("Evolution Status")
        status_layout = QFormLayout(status_group)

        self.cycles_label = QLabel("0")
        self.score_label = QLabel("0.0%")
        self.improvement_label = QLabel("+0.0%")
        self.convergence_label = QLabel("--")

        status_layout.addRow("Cycles:", self.cycles_label)
        status_layout.addRow("Current Score:", self.score_label)
        status_layout.addRow("Total Improvement:", self.improvement_label)
        status_layout.addRow("Convergence:", self.convergence_label)

        layout.addWidget(status_group)

        # Prompt versions table
        self.versions_table = QTableWidget()
        self.versions_table.setColumnCount(3)
        self.versions_table.setHorizontalHeaderLabels(["Version", "Score", "Timestamp"])
        layout.addWidget(self.versions_table)
```

### T4: Decision Log Viewer

```python
# ui/decision_log_viewer.py
class DecisionLogViewer(QWidget):
    """Widget for browsing agent decision history."""

    def __init__(self, decision_logger: DecisionLogger):
        super().__init__()
        self.logger = decision_logger
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Filters
        filter_layout = QHBoxLayout()

        self.agent_filter = QComboBox()
        self.agent_filter.addItem("All Agents")

        self.outcome_filter = QComboBox()
        self.outcome_filter.addItems(["All", "Correct", "Incorrect", "Pending"])

        self.date_filter = QDateEdit()
        self.date_filter.setDate(QDate.currentDate())

        filter_layout.addWidget(QLabel("Agent:"))
        filter_layout.addWidget(self.agent_filter)
        filter_layout.addWidget(QLabel("Outcome:"))
        filter_layout.addWidget(self.outcome_filter)
        filter_layout.addWidget(QLabel("Date:"))
        filter_layout.addWidget(self.date_filter)

        layout.addLayout(filter_layout)

        # Decision table
        self.decisions_table = QTableWidget()
        self.decisions_table.setColumnCount(6)
        self.decisions_table.setHorizontalHeaderLabels([
            "Timestamp", "Agent", "Decision", "Confidence", "Outcome", "Details"
        ])
        layout.addWidget(self.decisions_table)

        # Decision detail panel
        self.detail_panel = DecisionDetailPanel()
        layout.addWidget(self.detail_panel)
```

---

## When to Access Dashboard

Add to CLAUDE.md:

```markdown
### LLM Dashboard Usage

Access agent analytics through the dashboard:

```python
from ui import (
    AgentMetricsWidget,
    DebateViewer,
    EvolutionMonitor,
    DecisionLogViewer,
    create_llm_dashboard,
)

# Create integrated LLM dashboard
dashboard = create_llm_dashboard(
    metrics_tracker=tracker,
    debate_mechanism=debate,
    evolving_agents=[agent1, agent2],
    decision_logger=logger,
)

# Or add widgets individually to existing dashboard
main_dashboard.add_tab(AgentMetricsWidget(tracker), "Agent Metrics")
main_dashboard.add_tab(DebateViewer(debate), "Debates")
main_dashboard.add_tab(EvolutionMonitor([agent1]), "Evolution")
main_dashboard.add_tab(DecisionLogViewer(logger), "Decisions")
```

---

## Phase 4: Double-Check Report

**Date**: 2025-12-01
**Checked By**: Claude Code Agent

### Success Criteria Verification

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Agent metrics widget created | File exists | `ui/agent_metrics_widget.py` (438 lines) | ✅ |
| Debate viewer created | File exists | `ui/debate_viewer.py` (405 lines) | ✅ |
| Evolution monitor created | File exists | `ui/evolution_monitor.py` (395 lines) | ✅ |
| Decision log viewer created | File exists | `ui/decision_log_viewer.py` (485 lines) | ✅ |
| Widgets tested | ≥ 20 test cases | 33 test cases in `test_llm_dashboard.py` | ✅ (exceeded) |
| Dashboard updated | Integration complete | New docks + menu items added | ✅ |
| CLAUDE.md updated | Sections added | LLM Dashboard section with usage guide | ✅ |

### All Components Implemented

- `ui/agent_metrics_widget.py`: AgentMetricsWidget, MetricLabel, MetricsChartWidget
- `ui/debate_viewer.py`: DebateViewer, ArgumentPanel, ModeratorPanel
- `ui/evolution_monitor.py`: EvolutionMonitor, EvolutionProgressWidget, CycleHistoryWidget
- `ui/decision_log_viewer.py`: DecisionLogViewer, DecisionDetailPanel
- `ui/dashboard.py`: Updated with LLM docks, View menu, set_llm_components()
- `ui/__init__.py`: All new exports added
- `tests/test_llm_dashboard.py`: 33 comprehensive test cases

---

## Phase 5: Introspection Report

**Date**: 2025-12-01

### Code Quality Improvements

| Improvement | Priority | Effort | Impact |
|-------------|----------|--------|--------|
| Add matplotlib charts for trend visualization | P2 | Medium | High |
| Add export functionality for decision logs | P2 | Low | Medium |
| Add real-time streaming updates | P2 | High | Medium |

### Feature Extensions

| Feature | Priority | Effort | Value |
|---------|----------|--------|-------|
| Dashboard state persistence | P2 | Medium | Medium |
| Custom widget layouts | P2 | Medium | Medium |
| External notification integration | P3 | High | Medium |
| Multi-user dashboard support | P3 | High | Low |

### Developer Experience

| Enhancement | Priority | Effort |
|-------------|----------|--------|
| Add widget theming customization | P2 | Medium |
| Add keyboard shortcuts for panels | P2 | Low |
| Add widget documentation tooltips | P3 | Low |

### Lessons Learned

1. **What worked:** Modular widget design allows independent testing
2. **What worked:** Factory functions simplify widget creation
3. **Key insight:** Stub classes enable graceful degradation without PySide6
4. **Key insight:** Dock-based layout provides flexible panel arrangement

### Recommended Next Steps

1. Add matplotlib integration for enhanced charting
2. Implement dashboard state save/restore
3. Add real-time websocket updates for live data

---

## Phase 6: Convergence Decision

**Date**: 2025-12-01

### Summary

- Tasks Completed: 8/8 (T1-T8 all complete)
- All success criteria met
- 33 test cases created (exceeds 20 target)
- CLAUDE.md updated with comprehensive usage guide

### Convergence Status

- [x] Core success criteria met (all 4 widgets created)
- [x] Test coverage exceeds target (33 vs 20 minimum)
- [x] Exports updated (`ui/__init__.py`)
- [x] CLAUDE.md updated with usage guide
- [x] Dashboard integration complete

### Decision

- [ ] **CONTINUE LOOP** - More work needed
- [x] **EXIT LOOP** - Convergence achieved
- [ ] **PAUSE** - Waiting for external dependency

---

## Final Status

**Status**: ✅ Complete (Converged)

All LLM Dashboard Integration has been implemented:

1. **Agent Metrics Widget**: Performance display with charts and color coding
2. **Debate Viewer**: Bull/Bear argument visualization with round navigation
3. **Evolution Monitor**: Progress tracking with cycle history and prompt versions
4. **Decision Log Viewer**: Filterable decision browser with detail panel
5. **Dashboard Integration**: Dockable panels via View menu
6. **Tests**: 33 test cases covering all functionality
7. **Exports**: All widgets available via `ui` module
8. **Documentation**: CLAUDE.md updated with comprehensive guide

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-01 | Upgrade path created |
| 2025-12-01 | Phase 3 implementation complete (T1-T8) |
| 2025-12-01 | Phase 4 double-check complete (33 tests) |
| 2025-12-01 | Phase 5 introspection complete |
| 2025-12-01 | **Convergence achieved** - All criteria met |

---

## Related Documents

- [UPGRADE-004](UPGRADE_004_MULTI_AGENT_DEBATE.md) - Agent Metrics (dependency)
- [UPGRADE-005](UPGRADE_005_SELF_EVOLVING_AGENTS.md) - Self-Evolving Agents (dependency)
- [UI Dashboard](../../ui/dashboard.py) - Existing dashboard
- [CLAUDE.md](../../CLAUDE.md) - Main documentation
