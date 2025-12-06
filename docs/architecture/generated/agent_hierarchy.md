# Agent Hierarchy

*Auto-generated: 2025-12-05 19:44*

This diagram shows the inheritance hierarchy of trading agents.

```mermaid
classDiagram
    direction TB

    class TradingAgent {
        <<abstract>>
        +analyze()
        +think()
        +act()
    }

    class TechnicalAnalyst {
        +analyze()
    }

    class BullResearcher {
        +analyze()
        +argue()
        +get_recent_signals()
        +clear_signals()
    }

    class SupervisorAgent {
        +analyze()
        +should_debate()
        +analyze_with_debate()
        +get_debate_history()
        +clear_debate_history()
    }

    class NewsAnalyst {
        +analyze()
    }

    class SentimentAnalyst {
        +analyze()
    }

    class ObservabilityAgent {
        +analyze()
        +update_token_usage()
        +start_monitoring()
        +stop_monitoring()
        +get_active_alerts()
    }

    class PositionRiskManager {
        +analyze()
    }

    class ConservativeTrader {
        +analyze()
    }

    class BearResearcher {
        +analyze()
        +argue()
        +get_recent_risks()
        +clear_risks()
    }

    TradingAgent <|-- TechnicalAnalyst
    TradingAgent <|-- BullResearcher
    TradingAgent <|-- SupervisorAgent
    TradingAgent <|-- NewsAnalyst
    TradingAgent <|-- SentimentAnalyst
    TradingAgent <|-- ObservabilityAgent
    TradingAgent <|-- PositionRiskManager
    TradingAgent <|-- ConservativeTrader
    TradingAgent <|-- BearResearcher
```

## Notes

- Agents inherit from base classes in `llm/agents/base.py`
- Each agent specializes in a specific analysis type
- Methods shown are public interface methods
