# Claude Code Hook System

*Auto-generated: 2025-12-05 19:44*

This diagram shows the Claude Code hook system architecture.

```mermaid
flowchart LR
    subgraph Triggers
        PreToolUse[PreToolUse]
        PostToolUse[PostToolUse]
        SessionStart[SessionStart]
        Stop[Stop]
        PreCompact[PreCompact]
    end

    subgraph Hooks
        subgraph Core
            ric[ric]
            protect_files[protect_files]
            pre_compact[pre_compact]
            session_stop[session_stop]
            hook_utils[hook_utils]
        end
        subgraph Validation
            algo_change_guard[algo_change_guard]
            validate_research[validate_research]
            qa_auto_check[qa_auto_check]
            validate_algorithm[validate_algorithm]
            validate_category_docs[validate_category_docs]
        end
        subgraph Research
            research_saver[research_saver]
            thorough_research[thorough_research]
            document_research[document_research]
            auto_research_trigger[auto_research_trigger]
        end
        subgraph Trading
            log_trade[log_trade]
            parse_backtest[parse_backtest]
            load_trading_context[load_trading_context]
            risk_validator[risk_validator]
        end
        subgraph Formatting
            auto_docstring[auto_docstring]
            update_cross_refs[update_cross_refs]
            auto_format[auto_format]
            template_detector[template_detector]
        end
        subgraph Agents
            agent_orchestrator[agent_orchestrator]
            multi_agent[multi_agent]
        end
    end

    PreToolUse --> core
    PostToolUse --> formatting
    PostToolUse --> validation
    SessionStart --> core
    Stop --> core
```

## Hook Categories

| Category | Purpose |
|----------|---------|
| **core** | RIC Loop, file protection, session management |
| **validation** | Algorithm, research, and QA validation |
| **research** | Research tracking and documentation |
| **trading** | Trading risk and execution hooks |
| **formatting** | Code formatting and cross-references |
| **agents** | Agent orchestration hooks |

Configuration: `.claude/settings.json`
