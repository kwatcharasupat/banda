# Overlap-Add Inference Strategy

This document outlines the refined plan for implementing a flexible and efficient overlap-add inference strategy within the `banda` project. This strategy is designed to handle various models and scenarios, including single-stem, multi-stem, single-mixture, and multi-query inference, while minimizing CPU-GPU data transfer.

## Phase 1: Core Overlap-Add Utility Module (`src/banda/utils/inference.py`)

### `OverlapAddProcessor` Class

This class will encapsulate the core logic for audio chunking, windowing, and reconstruction using the overlap-add method.

#### Constructor (`__init__`)

*   **Parameters**: `chunk_size_seconds` (float), `hop_size_seconds` (float), `fs` (int, sampling rate).
*   **Calculations**:
    *   Convert `chunk_size_seconds` and `hop_size_seconds` into sample counts (`chunk_size_samples`, `hop_size_samples`).
    *   Calculate `overlap_samples = chunk_size_samples - hop_size_samples`.
*   **Windowing**:
    *   Pre-compute a Hann window of `chunk_size_samples`.
    *   The window will be scaled by `scaler = chunk_size_samples / (2 * hop_size_samples)` to ensure proper amplitude reconstruction after overlap-add.
*   **Storage**: Store these calculated parameters and the window as instance variables.

#### `_chunk_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, int, int]`

*   **Purpose**: Prepares the audio for chunking.
*   **Input**: `audio` (a single mixture tensor of shape `(channels, samples)`).
*   **Padding**:
    *   Apply symmetric padding to the input audio to ensure that all parts of the original signal are covered by chunks.
    *   The padding strategy will be similar to `coda`'s: `2 * overlap_samples` at the beginning and `2 * overlap_samples + pad` at the end. This ensures `F.unfold` operates correctly at boundaries.
*   **Unfolding**: Use `torch.nn.functional.unfold` to extract overlapping chunks from the padded audio.
*   **Reshaping**: Reshape the unfolded chunks into `(n_chunks, channels, chunk_size_samples)`.
*   **Returns**: The `chunked_audio` tensor, the `padded_length` of the audio, and the `original_length` of the audio.

#### `_reconstruct_audio(self, processed_chunks: Dict[str, torch.Tensor], padded_length: int, original_length: int) -> Dict[str, torch.Tensor]`

*   **Purpose**: Reconstructs the full audio from processed chunks using overlap-add.
*   **Input**: `processed_chunks` (a dictionary where keys are stem names and values are tensors of processed chunks for that stem, shape `(total_processed_chunks, channels, chunk_size_samples)`), `padded_length`, `original_length`.
*   **Window Application**: Apply the pre-computed Hann window (scaled appropriately) to each `processed_chunk`.
*   **Folding**: Use `torch.nn.functional.fold` to combine the windowed, overlapping chunks back into a single audio tensor for each stem.
*   **Cropping**: Crop the reconstructed audio to remove the padding, restoring it to its `original_length`.
*   **Returns**: A dictionary where keys are stem names and values are the full-length `reconstructed_audio` tensors of shape `(channels, original_samples)`.

#### `process(self, model: torch.nn.Module, audio: torch.Tensor, batch_size: int, device: torch.device, **model_kwargs) -> Dict[str, torch.Tensor]`

*   **Purpose**: Main public method for performing overlap-add inference on a given model and audio.
*   **Inputs**:
    *   `model` (the source separation model, expected to have a `forward` method that accepts `(batch_size, channels, samples)` and returns a dictionary of separated stems).
    *   `audio` (the input mixture tensor of shape `(channels, samples)`).
    *   `batch_size` (for processing chunks in mini-batches).
    *   `device` (e.g., `cpu`, `cuda`, `mps`).
    *   `**model_kwargs`: Allows passing additional arguments to the model's `forward` method (e.g., `query` for multi-query scenarios).
*   **Device Management**: Ensure the `model` is moved to the specified `device`.
*   **Chunking**: Call `_chunk_audio` to obtain the chunks.
*   **Batch Processing**: Iterate through the `chunked_audio` in mini-batches (defined by `batch_size`). For each mini-batch:
    *   Move the mini-batch of chunks to the `device`.
    *   Pass the mini-batch through the `model.forward()` method.
    *   **Crucial**: The `model.forward` method is expected to return a `Dict[str, torch.Tensor]` where keys are stem names and values are the separated audio chunks for that stem.
    *   Collect these separated chunks, maintaining their stem identity (e.g., using a `defaultdict(list)` to accumulate chunks for each stem).
*   **Reconstruction**: After all chunks are processed, for each stem, concatenate its collected chunks and call `_reconstruct_audio`.
*   **Returns**: A dictionary where keys are stem names (e.g., "vocals", "bass") and values are the full-length separated audio tensors.
*   **Optimization**: This method will use `torch.inference_mode()` and `torch.no_grad()` for efficient inference and minimize CPU-GPU transfers by keeping all chunk processing on the specified `device`.

## Phase 2: Integration into `SeparationTask` (`src/banda/tasks/separation.py`)

1.  **Update `__init__`**:
    *   Add a new parameter `inference_config: DictConfig` to the `SeparationTask` constructor.
    *   Instantiate `OverlapAddProcessor` using the parameters from `inference_config` (e.g., `chunk_size_seconds`, `hop_size_seconds`, `fs`).
    *   Store the `OverlapAddProcessor` instance as `self.overlap_add_processor`.
2.  **Update `from_config`**:
    *   Ensure that the `inference` section from the main configuration is passed to the `SeparationTask` constructor.
3.  **Implement `test_step`**:
    *   **Adapt to Full-Length Audio:** The `test_dataloader` will provide full-length audio.
    *   **Call `OverlapAddProcessor.process`**:
        *   Extract `full_length_mixture_audio` from the batch.
        *   Determine `model_kwargs` based on the batch structure (e.g., if `batch` contains `queries` for multi-query inference). This is where the `SeparationTask` will handle the "single-mixture multi-query" case by passing the relevant queries to the `OverlapAddProcessor.process` method, which then passes them to the model.
        *   Call `separated_audio = self.overlap_add_processor.process(self.model, full_length_mixture_audio, self.batch_size, self.device, **model_kwargs)`.
    *   **Loss and Metric Calculation**: Adapt existing loss and metric calculation to work with the `separated_audio` dictionary (full-length separated stems).
    *   **Robust Error Handling (OOM)**: Implement a retry mechanism with reduced `batch_size` for `OverlapAddProcessor.process` in case of CUDA OOM errors, similar to the `coda` example. This will involve a `try-except` block and recursively calling `test_step` with a smaller `batch_size`.
    *   **Result Saving**: Implement saving of test results (e.g., metrics, separated audio) to a designated output directory, potentially in JSON format for metrics and WAV for audio, similar to `coda`'s `save_to_audio` and JSON logging.

## Phase 3: Configuration (`src/banda/configs/train_bandit_musdb18hq_test.yaml`)

1.  **Add `inference` section:**
    *   Define the parameters for the `OverlapAddProcessor` under a new top-level `inference` section in the YAML configuration.
    *   **Example**:
        ```yaml
        inference:
          chunk_size_seconds: 6.0 # Duration of each chunk for inference
          hop_size_seconds: 3.0   # Overlap between chunks
          batch_size: 4           # Number of chunks to process simultaneously
        ```
    *   Ensure `fs` (sampling rate) is accessible, either by passing it directly or by referencing it from another part of the config (e.g., `model.fs`).

## Phase 4: DataModule Adaptation for Testing (`src/banda/data/datamodule.py`)

1.  **`test_dataloader` for Full Audio:**
    *   Modify `SourceSeparationDataset` to load full-length audio for the "test" split. This might involve a new `DatasetConfig` for testing that doesn't use `RandomChunkingTransform` for `premix_transform`.
    *   Adjust the `collate_fn` to handle full-length audio for the test dataloader. It should return `FixedStemSeparationBatch` where `mixture` and `sources` are full-length tensors, not chunks.

## Overall Flow Diagram (Updated):

```mermaid
graph TD
    A[Start Training Script] --> B{Hydra Config Load};
    B --> C[Instantiate DataModule];
    C --> D[Instantiate Model, Loss, Metrics];
    D --> E[Instantiate Lightning Task];
    E --> F[Instantiate Trainer with Callbacks];
    F --> G[Add ModelCheckpoint Callback];
    F --> H[Add WandbLogger Callback];
    H --> I[Wandb: Log Config, System Metrics];
    G --> J[Trainer.fit (Training Loop)];
    J --> K{Training Step};
    K --> L[Calculate Loss];
    L --> M[Calculate SNR Metric];
    M --> N[Log Train Loss & SNR (Prog Bar)];
    J --> O{Validation Step};
    O --> P[Calculate Loss];
    P --> Q[Update Val Metrics (SNR)];
    Q --> R[Log Val Loss & Metrics];
    J --> S[Trainer.test (Testing Loop)];
    S --> T{Test Dataloader (Full Audio)};
    T --> U{Test Step (Overlap-Add)};
    U --> V[OverlapAddProcessor.process];
    V --> W[Model.forward (on chunks)];
    W --> X[OverlapAddProcessor.reconstruct];
    X --> Y[Calculate Test Loss & Metrics];
    Y --> Z[Log Test Metrics];
    Z --> AA[Save Best Model Checkpoint];
    AA --> BB[Wandb: Log Model Artifacts];
    BB --> CC[End Training];