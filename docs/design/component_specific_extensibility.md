# 3. Component-Specific Extensibility

### 3.1. Models

*   **Interfaces (ABCs):**
    *   `banda.models.base.BaseModel`: Base class for all models.
    *   `banda.models.components.base.BaseEncoder`: Interface for model encoders.
    *   `banda.models.components.base.BaseDecoder`: Interface for model decoders.
    *   `banda.models.components.base.BaseMaskingHead`: Interface for mask estimation heads.
    *   `banda.models.components.base.BaseTimeFrequencyModel`: Interface for time-frequency processing modules.
    *   `banda.models.query_models.base.BaseQueryModel`: Interface for query encoding models.
*   **Registration:** All new model implementations (e.g., `Separator`, `Bandit`) and their sub-components (encoders, decoders, masking heads, TF models) will register themselves with the `MODELS_REGISTRY` or dedicated sub-registries.
*   **Dynamic Instantiation:** The `Separator` class (or any orchestrating model) will dynamically load its sub-components from the registry based on configuration. This eliminates the hardcoded `if/elif` logic seen in `coda`'s `BaseBandit` and `banda`'s current `Separator`.
*   **Query Models:** `SeparationTask` will dynamically load and integrate different query models (e.g., `Passt`, `Querier`) based on the batch type and configuration, leveraging the `QUERY_MODELS_REGISTRY`.

### 3.2. Data Handling

*   **Dataset Registration:** New dataset implementations (inheriting from a `banda.data.datasets.base.BaseDataset` ABC) will register with a `DATASETS_REGISTRY`.
*   **Dynamic DataModule:** `SourceSeparationDataModule` will be refactored to dynamically load dataset classes from the `DATASETS_REGISTRY` based on configuration. This ensures flexibility beyond specific datasets like MUSDB18HQ and allows for easy integration of new data sources.
*   **Flexible Collate Functions:** A mechanism will be introduced for `_collate_fn` to dynamically select or compose appropriate collate logic based on the specific dataset or batch type, potentially using a `COLLATE_FUNCTIONS_REGISTRY`. This will allow for custom batching strategies for different data modalities or model inputs.

### 3.3. Losses

*   **Loss Function Interfaces:** Define an ABC for `banda.losses.base.BaseLoss` (inheriting from `torch.nn.Module`).
*   **Loss Registration:** Individual loss functions (e.g., `L1Loss`, `MultiResolutionSTFTLoss`, `TimeDomainLoss`) will register with a `LOSSES_REGISTRY`.
*   **Dynamic Loss Handler:** The `banda.losses.separation_loss_handler.SeparationLossHandler` will be enhanced to dynamically compose and apply multiple loss functions (e.g., weighted sums, conditional losses) based on configuration, retrieving them from the `LOSSES_REGISTRY`. This will replace the current manual instantiation in `SeparationTask.from_config` and provide a more flexible loss aggregation mechanism than `coda`'s `BaseMultiLossHandler`.

### 3.4. Metrics

*   **Metric Interfaces:** Define an ABC for `banda.metrics.base.BaseMetric` (inheriting from `torchmetrics.Metric`).
*   **Metric Registration:** New metrics (e.g., `SNR`, `SDR`, `PredictedDecibels`) will register with a `METRICS_REGISTRY`.
*   **Dynamic MetricHandler:** `banda.metrics.metric_handler.MetricHandler` will dynamically load and manage metrics from the `METRICS_REGISTRY`, allowing for easy addition of new evaluation criteria and flexible reporting.

### 3.5. Optimizers

*   **Optimizer Registration:** While Hydra's `_target_` is generally effective for optimizers, for consistency and if more complex custom optimizer logic is introduced, optimizers could also register with an `OPTIMIZERS_REGISTRY`. For now, Hydra's `_target_` will be retained for standard PyTorch optimizers.

### 3.6. Pydantic Data Models

To further standardize data handling and ensure strict type checking and validation, Pydantic models will be introduced for all core data structures, particularly for batch types.

*   **Batch Type Standardization:**
    *   New Pydantic models will be defined in `src/banda/data/batch_types.py` to represent the structure of data batches passed between components (e.g., `AudioBatch`, `SpectrogramBatch`, `MetadataBatch`).
    *   These models will enforce the expected types and shapes of data, improving data integrity and making debugging easier.

*   **Example Pydantic Batch Model:**
    ```python
    # Example: src/banda/data/batch_types.py
    from pydantic import BaseModel
    import torch

    class AudioBatch(BaseModel):
        audio: torch.Tensor  # Shape: (batch_size, channels, samples)
        metadata: dict       # Arbitrary metadata

        class Config:
            arbitrary_types_allowed = True # Allow torch.Tensor
    ```

*   **Integration with Data Loading/Processing:**
    *   The `__getitem__` methods within dataset classes (e.g., in `src/banda/data/datasets/`) will be updated to return instances of the appropriate Pydantic batch models.
    *   The `collate_fn` in `src/banda/data/datamodule.py` (or any custom collate functions registered via `COLLATE_FUNCTIONS_REGISTRY`) will be modified to assemble individual samples into batches that conform to the defined Pydantic batch models.
    *   Components consuming these batches (e.g., models, loss functions) will be updated to expect and utilize these Pydantic-validated data structures.