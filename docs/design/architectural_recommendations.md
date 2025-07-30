# 2. Architectural Recommendations

To achieve high extensibility, `banda` will adopt the following architectural principles and patterns:

### Core Principle: Dependency Inversion & Inversion of Control

The system will emphasize **Dependency Inversion**, where high-level modules do not depend on low-level modules, but both depend on abstractions. This will be achieved through:
*   **Abstract Base Classes (ABCs):** Defining clear interfaces for all swappable components (models, datasets, losses, etc.).
*   **Inversion of Control (IoC):** Components will be instantiated and managed by a central mechanism (the proposed Registry Pattern) rather than being created directly by their consumers.

### Registry Pattern for Component Discovery

A **Registry Pattern** will be the cornerstone of `banda`'s extensibility. This pattern provides a centralized mechanism for components to register themselves and be discovered and instantiated dynamically based on a unique identifier (e.g., a string name).

*   **Concept:** A generic `Registry` class will maintain a mapping of string identifiers to component classes or factory functions.
*   **Implementation:** Dedicated registries will be created for each major component type:
    *   `MODELS_REGISTRY`
    *   `DATASETS_REGISTRY`
    *   `LOSSES_REGISTRY`
    *   `METRICS_REGISTRY`
    *   `OPTIMIZERS_REGISTRY`
    *   `QUERY_MODELS_REGISTRY`
    *   `COLLATE_FUNCTIONS_REGISTRY` (for dynamic data collators)
*   **Registration Mechanism:** Components will register themselves using a decorator-based approach. For example:
    ```python
    from banda.utils.registry import MODELS_REGISTRY

    @MODELS_REGISTRY.register("my_new_model")
    class MyNewModel(BaseModel):
        # ... implementation ...
    ```
*   **Retrieval:** The main application logic (e.g., `train.py`, `SeparationTask`) will retrieve component classes from these registries using their registered names, as specified in the Hydra configuration. This replaces direct imports and hardcoded instantiations.

### Strategy Pattern for Dynamic Behavior

The **Strategy Pattern** will be applied where different algorithms or behaviors need to be swapped at runtime. This is particularly relevant for:
*   **Loss Function Composition:** Combining multiple loss functions (ee.g., weighted sums, conditional losses).
*   **Data Augmentation Pipelines:** Dynamically selecting and composing different augmentation strategies.