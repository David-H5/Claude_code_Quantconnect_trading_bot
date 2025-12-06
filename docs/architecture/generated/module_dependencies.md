# Module Dependencies

*Auto-generated: 2025-12-05 19:44*

This diagram shows the layer architecture and module dependencies.

```mermaid
flowchart TB
    subgraph Legend
        direction LR
        L4[Layer 4: Apps]
        L3[Layer 3: Domain]
        L2[Layer 2: Core]
        L1[Layer 1: Infra]
        L0[Layer 0: Utils]
    end

    subgraph Layer4["Applications"]
        algorithms[algorithms/]
        api[api/]
        ui[ui/]
    end

    subgraph Layer3["Domain Logic"]
        execution[execution/]
        llm[llm/]
        scanners[scanners/]
        indicators[indicators/]
    end

    subgraph Layer2["Core Models"]
        models[models/]
        compliance[compliance/]
        evaluation[evaluation/]
    end

    subgraph Layer1["Infrastructure"]
        observability[observability/]
        config[config/]
    end

    subgraph Layer0["Utilities"]
        utils[utils/]
    end

    Layer4 --> Layer3
    Layer3 --> Layer2
    Layer2 --> Layer1
    Layer1 --> Layer0

    %% Styling
    classDef layer4 fill:#e1f5fe,stroke:#01579b
    classDef layer3 fill:#f3e5f5,stroke:#4a148c
    classDef layer2 fill:#fff3e0,stroke:#e65100
    classDef layer1 fill:#e8f5e9,stroke:#1b5e20
    classDef layer0 fill:#fafafa,stroke:#212121
```

## Layer Rules

- **Layer 4 (Applications)**: Can import from all lower layers
- **Layer 3 (Domain Logic)**: Can import from Layer 2, 1, 0
- **Layer 2 (Core Models)**: Can import from Layer 1, 0
- **Layer 1 (Infrastructure)**: Can import from Layer 0 only
- **Layer 0 (Utilities)**: No internal dependencies

Violations are detected by `scripts/check_layer_violations.py`.
