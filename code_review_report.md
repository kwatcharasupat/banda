# Codebase Review Report for Banda

This report details findings from a comprehensive review of the `banda` codebase, focusing on inconsistencies, inefficiencies, and code duplication. For each identified issue, a clear description, relevant code snippets, and proposed refactoring strategies are provided.

## 1. Code Duplication

Significant code duplication exists across core model implementations, particularly within the `__init__`, `forward`, and `from_config` methods of separator models. This redundancy complicates maintenance, understanding, and extensibility.

### 1.1. Duplication in Separator Model Initialization and Forward Pass

**Issue Description:**
The `Bandit`, `Banquet`, `Separator`, and `QuerySeparator` models share substantial boilerplate code for setting up STFT parameters, bandsplit modules, time-frequency models, and mask estimation. Their `forward` passes also contain identical or near-identical logic for spectrogram processing, mask application, and audio reconstruction.

**Relevant Files and Code Snippets:**

*   **`src/banda/models/core_models/bandit_separator.py`**
    *   Mask Estimation Instantiation: [`src/banda/models/core_models/bandit_separator.py:49-56`](src/banda/models/core_models/bandit_separator.py:49-56)
    *   Mask Application & ISTFT: [`src/banda/models/core_models/bandit_separator.py:86-109`](src/banda/models/core_models/bandit_separator.py:86-109)
*   **`src/banda/models/core_models/banquet_separator.py`**
    *   Mask Estimation Instantiation: [`src/banda/models/core_models/banquet_separator.py:78-80`](src/banda/models/core_models/banquet_separator.py:78-80)
    *   Mask Application & ISTFT: [`src/banda/models/core_models/banquet_separator.py:120-140`](src/banda/models/core_models/banquet_separator.py:120-140)
*   **`src/banda/models/core_models/separator.py`**
    *   Initialization of Sub-modules: [`src/banda/models/core_models/separator.py:80-143`](src/banda/models/core_models/separator.py:80-143)
    *   Spectrogram Processing & Mask Application: [`src/banda/models/core_models/separator.py:183-258`](src/banda/models/core_models/separator.py:183-258)
    *   `from_config` Method: [`src/banda/models/core_models/separator.py:281-313`](src/banda/models/core_models/separator.py:281-313)
*   **`src/banda/models/core_models/query_separator.py`**
    *   Initialization of Sub-modules: [`src/banda/models/core_models/query_separator.py:92-164`](src/banda/models/core_models/query_separator.py:92-164)
    *   Spectrogram Processing & Mask Application: [`src/banda/models/core_models/query_separator.py:185-231`](src/banda/models/core_models/query_separator.py:185-231)
    *   `from_config` Method: [`src/banda/models/core_models/query_separator.py:262-284`](src/banda/models/core_models/query_separator.py:262-284)

**Proposed Refactoring Strategy:**

1.  **Unified `BaseSeparator`:** Create a new abstract base class, e.g., `src/banda/models/core_models/base_separator.py`, that inherits from `BaseModel` and `SpectralComponent`. This class will encapsulate all common initialization logic (STFT parameters, bandsplit module, TF model, mask estimation setup) and common forward pass steps.
2.  **Abstract Common Processing Steps:**
    *   Move the `_process_spectrogram`, `_process_bandsplit`, and `_reconstruct_audio_from_masked_spec` methods to the new `BaseSeparator` and ensure they are generic enough.
    *   Introduce a new protected method, e.g., `_apply_masks_and_reconstruct(self, mixture_signal, masks)`, in `BaseSeparator` to centralize the mask application and ISTFT logic.
3.  **Abstract Query Integration:** Define an abstract method in `BaseSeparator` (e.g., `_process_features(self, band_features, query_input)`) that fixed-stem models implement as an identity operation and query-based models implement with their specific query conditioning logic.
4.  **Centralize `from_config` Logic:** Implement a `_from_config_common(cls, cfg)` class method in `BaseSeparator` to handle common parameter extraction and sub-config instantiation. Subclasses will call this method and add their specific parameters.

### 1.2. Duplication of `TorchInputAudioDict`

**Issue Description:**
The `TorchInputAudioDict` Pydantic model is redundantly defined in two separate files. This can lead to inconsistencies if one definition is updated and the other is not, and makes code harder to navigate.

**Relevant Files and Code Snippets:**

*   [`src/banda/data/batch_types.py:20-31`](src/banda/data/batch_types.py:20-31)
*   [`src/banda/data/types.py:24-35`](src/banda/data/types.py:24-35)

**Proposed Refactoring Strategy:**

1.  **Single Source of Truth:** Remove the duplicate definition from `src/banda/data/types.py`.
2.  **Consistent Import:** Ensure all modules that require `TorchInputAudioDict` import it solely from `src/banda/data/batch_types.py`.

## 2. Inconsistencies

Several inconsistencies were found in naming conventions, parameter handling, and logging practices, leading to reduced clarity and potential for errors.

### 2.1. Naming Convention and Parameter Handling for TF Models

**Issue Description:**
There's a mismatch between the Pydantic model's type hint for `tf_model` in `common_configs.py` (a `Union` of specific TF model configs) and its actual usage in `Bandit.from_config` and `Banquet.from_config`, where a `DictConfig` is passed directly.

**Relevant Files and Code Snippets:**

*   [`src/banda/models/common_components/configs/common_configs.py:68`](src/banda/models/common_components/configs/common_configs.py:68)
*   [`src/banda/models/core_models/bandit_separator.py:155`](src/banda/models/core_models/bandit_separator.py:155)
*   [`src/banda/models/core_models/banquet_separator.py:186`](src/banda/models/core_models/banquet_separator.py:186)

**Proposed Refactoring Strategy:**

1.  **Consistent Instantiation:** Standardize the instantiation of TF models. Either:
    *   Modify the Pydantic configs to accept `DictConfig` directly for `tf_model` and handle instantiation within the model's `__init__` or `from_config` method using `hydra.utils.instantiate`.
    *   Or, ensure that the `tf_model` field in the Pydantic config is always instantiated into the correct Pydantic model type *before* being passed to the separator model. The current approach of passing `DictConfig` and then using `hydra.utils.get_class` is functional but less type-safe.

### 2.2. Hardcoded `drop_dc_band` in `Separator`

**Issue Description:**
The `drop_dc_band` parameter is hardcoded to `True` in `Separator.__init__`, making it less flexible compared to `Bandit` and `Banquet` where it's configurable.

**Relevant Files and Code Snippets:**

*   [`src/banda/models/core_models/separator.py:74`](src/banda/models/core_models/separator.py:74)

**Proposed Refactoring Strategy:**

1.  **Make Configurable:** Add `drop_dc_band` to `SeparatorConfig` and ensure it's passed consistently throughout the `Separator` model's initialization and `from_config` method, similar to `Bandit` and `Banquet`.

### 2.3. Inconsistent Logging Configuration

**Issue Description:**
Logging configuration is fragmented across modules. `src/banda/train.py` uses `hydra.utils.log` and calls `configure_logging`, while other modules directly use `structlog.get_logger(__name__)` and set levels individually. This leads to inconsistent logging behavior and makes global log level management difficult.

**Relevant Files and Code Snippets:**

*   [`src/banda/train.py:72-77`](src/banda/train.py:72-77)
*   [`src/banda/models/common_components/bandsplit/band_split_module.py:15`](src/banda/models/common_components/bandsplit/band_split_module.py:15)
*   [`src/banda/models/common_components/time_frequency_models/tf_models.py:15`](src/banda/models/common_components/time_frequency_models/tf_models.py:15)
*   [`src/banda/models/common_components/mask_estimation/mask_estimation_modules.py:22`](src/banda/models/common_components/mask_estimation/mask_estimation_modules.py:22)

**Proposed Refactoring Strategy:**

1.  **Unified Logging Setup:** Centralize all logging configuration in `src/banda/utils/logging.py`.
2.  **Consistent Logger Usage:** Ensure all modules import and use the logger consistently without setting individual levels. The `configure_logging` function should be called once at the application entry point (e.g., in `src/banda/train.py`).

### 2.4. Inconsistent `BandsplitSpecification` Initialization

**Issue Description:**
The `BandsplitSpecification` base class and its subclasses (e.g., `FixedBandsplitSpecs`, `VocalBandsplitSpecification`) have inconsistent `__init__` signatures regarding the `config` parameter. Some take `config: DictConfig` while others don't, leading to potential confusion and errors when instantiating via `get_bandsplit_specs_factory`.

**Relevant Files and Code Snippets:**

*   [`src/banda/models/common_components/spectral_components/spectral_base.py:185`](src/banda/models/common_components/spectral_components/spectral_base.py:185) (`BandsplitSpecification` does not take `config`)
*   [`src/banda/models/common_components/spectral_components/spectral_base.py:262`](src/banda/models/common_components/spectral_components/spectral_base.py:262) (`FixedBandsplitSpecs` takes `config`)
*   [`src/banda/models/common_components/spectral_components/spectral_base.py:305`](src/banda/models/common_components/spectral_components/spectral_base.py:305) (`VocalBandsplitSpecification` takes `config`)

**Proposed Refactoring Strategy:**

1.  **Standardize `BandsplitSpecification` `__init__`:** Decide whether `BandsplitSpecification` and its direct subclasses should always receive a `config` object. If not, ensure that `FixedBandsplitSpecs` and its children handle the `config` parameter explicitly and consistently, perhaps by passing only the necessary parts of the config to the `super().__init__` call.
2.  **Refine `get_bandsplit_specs_factory`:** Adjust the `get_bandsplit_specs_factory` function to handle the `config` parameter more robustly, ensuring it's only passed to constructors that expect it.

## 3. Inefficiencies

Some areas could lead to inefficiencies, particularly in data handling and processing, which are crucial for a research repository dealing with large audio datasets.

### 3.1. Hardcoded `query_features` in `_collate_fn`

**Issue Description:**
In `src/banda/data/datamodule.py`, the `query_features` for the dummy query in `_collate_fn` is hardcoded to `128`. This is brittle and not scalable, as the actual `query_features` should be determined by the model's configuration.

**Relevant Files and Code Snippets:**

*   [`src/banda/data/datamodule.py:226`](src/banda/data/datamodule.py:226)

**Proposed Refactoring Strategy:**

1.  **Dynamic `query_features`:** Pass the `query_features` from the model configuration to the `SourceSeparationDataModule` during its instantiation. The `_collate_fn` can then access this value from `self`.

### 3.2. Repeated `os.path.expandvars` Calls

**Issue Description:**
In dataset connectors (e.g., `MUSDB18Connector`, `MedleyDBConnector`, `MoisesDBConnector`), `os.path.expandvars` is called multiple times on `data_root` and `stem_path`. While not a major performance bottleneck, it's redundant and can be optimized.

**Relevant Files and Code Snippets:**

*   [`src/banda/data/datasets/musdb18hq.py:96`](src/banda/data/datasets/musdb18hq.py:96)
*   [`src/banda/data/datasets/medleydb.py:88`](src/banda/data/datasets/medleydb.py:88)
*   [`src/banda/data/datasets/moisesdb.py:88`](src/banda/data/datasets/moisesdb.py:88)

**Proposed Refactoring Strategy:**

1.  **Single Expansion:** Expand environment variables for `data_root` once during the initialization of `DatasetConnector` subclasses, and store the resolved path.

### 3.3. Suboptimal Multiple Inheritance with `nn.Module`

**Issue Description:**
In `src/banda/core/interfaces.py`, `BaseQueryModel` explicitly comments out `super().__init__()` to prevent multiple `nn.Module.__init__` calls. This indicates a potential issue with Python's Method Resolution Order (MRO) when using multiple inheritance with `nn.Module`. While it prevents errors, it's a sign of a design that could be cleaner and more robust.

**Relevant Files and Code Snippets:**

*   [`src/banda/core/interfaces.py:159-161`](src/banda/core/interfaces.py:159-161)

**Proposed Refactoring Strategy:**

1.  **Review Class Hierarchy:** Re-evaluate the class hierarchy. If `BaseQueryModel` is intended to be a mixin, consider if it truly needs to inherit from `nn.Module`. A common pattern is to have a single base class that inherits from `nn.Module`, and other mixins provide additional functionality without inheriting from `nn.Module` themselves. This ensures a clear MRO and proper initialization of `nn.Module`.

## 4. Configuration Structure Review

The configuration system, primarily managed by Hydra and OmegaConf, is generally well-structured. However, some areas can be improved for clarity and robustness.

### 4.1. Overly Complex `defaults` in `config.yaml`

**Issue Description:**
The `defaults` section in `src/banda/configs/config.yaml` uses `_self_` and imports other config groups. While functional, it can become complex to trace the full configuration inheritance, especially with nested `defaults`.

**Relevant Files and Code Snippets:**

*   [`src/banda/configs/config.yaml:4`](src/banda/configs/config.yaml:4)
*   [`src/banda/configs/train_bandit_musdb18hq_test.yaml:3-8`](src/banda/configs/train_bandit_musdb18hq_test.yaml:3-8)

**Proposed Refactoring Strategy:**

1.  **Simplify Defaults:** Consider flattening the `defaults` hierarchy where possible or using more explicit imports to make the configuration composition clearer. For example, instead of `_self_` and then importing other groups, explicitly list all top-level defaults.

### 4.2. Hardcoded Paths in `train_bandit_musdb18hq_test.yaml`

**Issue Description:**
The `data_root` paths in `src/banda/configs/train_bandit_musdb18hq_test.yaml` are hardcoded absolute paths, which makes the configuration non-portable across different environments.

**Relevant Files and Code Snippets:**

*   [`src/banda/configs/train_bandit_musdb18hq_test.yaml:66`](src/banda/configs/train_bandit_musdb18hq_test.yaml:66)
*   [`src/banda/configs/train_bandit_musdb18hq_test.yaml:81`](src/banda/configs/train_bandit_musdb18hq_test.yaml:81)
*   [`src/banda/configs/train_bandit_musdb18hq_test.yaml:95`](src/banda/configs/train_bandit_musdb18hq_test.yaml:95)

**Proposed Refactoring Strategy:**

1.  **Use Environment Variables or Relative Paths:** Consistently use environment variables (e.g., `${oc.env:DATA_ROOT}`) or paths relative to `project_root` for data directories, as is done in `src/banda/configs/data/default.yaml`.

### 4.3. Redundant `mask_estim` Configuration in `train_bandit_musdb18hq_test.yaml`

**Issue Description:**
The `mask_estim` section at the end of `src/banda/configs/train_bandit_musdb18hq_test.yaml` seems to be an override for `in_channel` that might be better placed within the `model/mask_estim` config group itself or handled more dynamically.

**Relevant Files and Code Snippets:**

*   [`src/banda/configs/train_bandit_musdb18hq_test.yaml:164-165`](src/banda/configs/train_bandit_musdb18hq_test.yaml:164-165)

**Proposed Refactoring Strategy:**

1.  **Consolidate Configuration:** Ensure that all parameters related to `mask_estim` are defined within the `model/mask_estim` config group (`src/banda/configs/model/mask_estim/mask_estim_default.yaml`). If `in_channel` needs to be dynamic, it should be passed through the model's `from_config` method or derived from other model parameters.

## 5. General Design Improvements

*   **Pydantic Model Enforcement:** Continue to enforce and leverage Pydantic models for all configurations. Ensure that `DictConfig` objects are consistently converted to Pydantic models at the earliest appropriate stage to benefit from type validation and IDE support.
*   **Clearer Data Flow with `AudioSignal`:** The introduction of `AudioSignal` in `batch_types.py` is a positive step towards encapsulating audio data and its representations. Ensure all model inputs and outputs consistently use `AudioSignal` where appropriate to improve type safety and code readability.
*   **Consistent Error Handling:** Maintain the current practice of raising `ValueError` for invalid states (e.g., NaNs in tensors) and use `structlog` for detailed logging of these errors.
*   **Deprecation Management:** Address deprecated parameters and warnings (e.g., `use_freq_weights` in `mask_estimation_modules.py`) by removing them from the code and configurations.

This report provides a comprehensive overview of areas for improvement within the `banda` codebase. Addressing these points will significantly enhance the maintainability, extensibility, and reproducibility of the research repository.