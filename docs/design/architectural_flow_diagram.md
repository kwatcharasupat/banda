graph TD
    A[Hydra Config] --> B{Component Registries};
    B --> B1[MODELS_REGISTRY];
    B --> B2[DATASETS_REGISTRY];
    B --> B3[LOSSES_REGISTRY];
    B --> B4[METRICS_REGISTRY];
    B --> B5[OPTIMIZERS_REGISTRY];
    B --> B6[QUERY_MODELS_REGISTRY];
    B --> B7[COLLATE_FUNCTIONS_REGISTRY];

    C[train.py] --> D[SeparationTask (LightningModule)];
    D --> E[Model (e.g., Separator)];
    D --> F[LossHandler];
    D --> G[MetricHandler];
    D --> H[Optimizer];
    D --> I[DataModule];

    E -- Retrieves --> B1;
    F -- Retrieves --> B3;
    G -- Retrieves --> B4;
    H -- Retrieves --> B5;
    I -- Retrieves --> B2;
    I -- Retrieves --> B7;
    E -- Retrieves --> B6;

    subgraph Component Definitions
        J[NewModel.py] -- Registers --> B1;
        K[NewDataset.py] -- Registers --> B2;
        L[NewLoss.py] -- Registers --> B3;
        M[NewMetric.py] -- Registers --> B4;
        N[NewOptimizer.py] -- Registers --> B5;
        O[NewQueryModel.py] -- Registers --> B6;
        P[NewCollateFn.py] -- Registers --> B7;
    end

    subgraph Pydantic Data Flow
        Q[Dataset.__getitem__] --> R[Pydantic Batch Model];
        S[Collate Function] --> R;
        R --> T[Model Input];
        R --> U[Loss Function Input];
    end

    style B1 fill:#f9f,stroke:#333,stroke-width:2px
    style B2 fill:#f9f,stroke:#333,stroke-width:2px
    style B3 fill:#f9f,stroke:#333,stroke-width:2px
    style B4 fill:#f9f,stroke:#333,stroke-width:2px
    style B5 fill:#f9f,stroke:#333,stroke-width:2px
    style B6 fill:#f9f,stroke:#333,stroke-width:2px
    style B7 fill:#f9f,stroke:#333,stroke-width:2px
    style J fill:#ccf,stroke:#333,stroke-width:2px
    style K fill:#ccf,stroke:#333,stroke-width:2px
    style L fill:#ccf,stroke:#333,stroke-width:2px
    style M fill:#ccf,stroke:#333,stroke-width:2px
    style N fill:#ccf,stroke:#333,stroke-width:2px
    style O fill:#ccf,stroke:#333,stroke-width:2px
    style P fill:#ccf,stroke:#333,stroke-width:2px
    style Q fill:#e0f7fa,stroke:#333,stroke-width:2px
    style R fill:#e0f7fa,stroke:#333,stroke-width:2px
    style S fill:#e0f7fa,stroke:#333,stroke-width:2px
    style T fill:#e0f7fa,stroke:#333,stroke-width:2px
    style U fill:#e0f7fa,stroke:#333,stroke-width:2px