# Banda: Advanced Source Separation Framework

Banda is a cutting-edge source separation framework designed for both fixed-stem and query-based audio separation tasks. It leverages advanced deep learning models, including Transformer and Mamba architectures, combined with sophisticated spectral processing techniques. Banda is built with a strong emphasis on modularity, configurability, and extensibility, making it a versatile tool for audio research and development.

## Description

Banda provides a comprehensive solution for separating individual audio sources from a mixed audio signal. Its core features include:

-   **Flexible Model Architectures**: Supports both fixed-stem separation (e.g., separating vocals, drums, bass, other) and query-based separation, where a query (audio, embedding, or class label) guides the separation process.
-   **Modular Design**: Built with a highly modular structure, allowing for easy interchangeability of components such as STFT configurations, bandsplit specifications, time-frequency models, and mask estimation modules.
-   **Advanced Time-Frequency Models**: Integrates state-of-the-art Transformer and Mamba models for processing time-frequency representations, enabling powerful feature extraction and context modeling.
-   **Configurable Spectral Processing**: Features a robust bandsplit mechanism that allows for various frequency band definitions (fixed, perceptual) and flexible handling of DC bands and channel configurations.
-   **Hydra-based Configuration**: Utilizes Hydra for a clean and efficient configuration management system, enabling dynamic instantiation of models, datasets, and losses directly from YAML files.
-   **Pydantic for Type Safety**: Employs Pydantic models for configuration schemas, ensuring type safety and validation across all configurable components.
-   **Comprehensive Loss Functions**: Includes a variety of loss functions, including multi-resolution STFT losses, time-domain losses, and specialized losses like L1SNR and dBRMS regularization, managed by an extensible `SeparationLossHandler`.
-   **Weights & Biases Integration**: Seamlessly integrates with Weights & Biases (wandb) for experiment tracking, visualization, and reproducibility.

## Features

-   **Fixed-Stem Separation**: Separate predefined sources (e.g., vocals, drums, bass, other) from a mixture.
-   **Query-Based Separation**: Condition the separation on an input query (audio, embedding, or class label).
-   **STFT/ISTFT Module**: Efficient and configurable Short-Time Fourier Transform and Inverse STFT operations.
-   **Bandsplit Module**: Divides the spectrogram into multiple frequency bands for specialized processing.
    -   Supports fixed bands (e.g., vocal, bass, drums, other).
    -   Supports perceptual bands (e.g., Musical, Mel, Bark, ERB, Triangular Bark, Mini Bark).
-   **Time-Frequency (TF) Models**:
    -   **Transformer**: Attention-based models for capturing long-range dependencies.
    -   **Mamba**: State-space models for efficient sequence modeling.
    -   **RNN/GRU**: Recurrent neural networks for sequential processing.
-   **Mask Estimation**: Predicts masks for each source in each frequency band.
    -   Supports both overlapping and non-overlapping band mask estimation.
-   **Loss Handler**: Manages and applies multiple loss functions for training.
-   **Data Modules**: Standardized data loading and batching using PyTorch Lightning DataModules.
-   **Logging**: Structured logging with `structlog` for better debugging and monitoring.

## Installation

To set up Banda, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/banda.git
    cd banda
    ```

2.  **Create and activate a Conda environment (recommended)**:
    ```bash
    conda create -n banda python=3.9
    conda activate banda
    ```

3.  **Install PyTorch**:
    Follow the instructions on the official PyTorch website to install the correct version for your system and CUDA/MPS setup. For example, for CUDA 11.8:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    For Apple Silicon (MPS):
    ```bash
    pip install torch torchvision torchaudio
    ```

4.  **Install `mambapy` (if using Mamba models)**:
    ```bash
    pip install mambapy
    ```

5.  **Install other dependencies**:
    ```bash
    pip install -e .
    ```

## Usage

### Training a Model

Banda uses Hydra for configuration. You can train a model using the `train.py` script and specify configurations via command-line arguments.

Example training command for a Bandit model on MUSDB18HQ:

```bash
python src/banda/train.py \
    model=bandit_separator \
    data=musdb18hq \
    trainer=default \
    logger=wandb \
    +experiment=train_bandit_musdb18hq_test \
    hydra.run.dir=outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}_${experiment.name}
```

**Important Configuration Notes**:

-   **`data_root`**: Ensure your `data_root` path in the dataset configuration (e.g., `src/banda/configs/data/musdb18hq.yaml`) is correctly set. You can use environment variables for flexibility:
    ```yaml
    # src/banda/configs/data/musdb18hq.yaml
    data_root: "${oc.env:MUSDB18HQ_DATA_ROOT,${project_root}/data/musdb18hq/intermediates/npz}"
    ```
    Then, set the environment variable:
    ```bash
    export MUSDB18HQ_DATA_ROOT="/path/to/your/musdb18hq/data"
    ```
    or place your data in `${project_root}/data/musdb18hq/intermediates/npz`.

-   **MPS Fallback (for Apple Silicon)**: If you encounter issues with PyTorch MPS, you might need to set the `PYTORCH_ENABLE_MPS_FALLBACK` environment variable:
    ```bash
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    ```

### Evaluation

(Details on evaluation script and usage will be added here.)

### Inference

(Details on inference script and usage will be added here.)

## Configuration

Banda's configuration is managed by Hydra. Key configuration files are located in `src/banda/configs/`:

-   `config.yaml`: Main configuration entry point.
-   `model/`: Model-specific configurations (e.g., `bandit_separator.yaml`, `banquet_separator.yaml`).
-   `data/`: Dataset configurations (e.g., `musdb18hq.yaml`).
-   `trainer/`: PyTorch Lightning Trainer configurations.
-   `logger/`: Logger configurations (e.g., `wandb.yaml`).
-   `experiment/`: Predefined experiment configurations that compose other configs.

You can override any parameter from the command line:
`python src/banda/train.py model.stft.n_fft=4096`

## Project Structure

```
banda/
├── docs/                     # Project documentation (design, architecture, etc.)
├── outputs/                  # Training outputs, logs, and checkpoints
├── scripts/                  # Utility scripts (e.g., data preprocessing)
├── src/
│   ├── banda/
│   │   ├── configs/          # Hydra configuration files
│   │   │   ├── data/
│   │   │   ├── model/
│   │   │   │   ├── bandsplit/
│   │   │   │   ├── mask_estim/
│   │   │   │   └── tf_model/
│   │   │   ├── trainer/
│   │   │   └── ...
│   │   ├── core/             # Core interfaces and base classes
│   │   ├── data/             # Data loading, batching, and dataset implementations
│   │   │   ├── batch_types.py
│   │   │   └── datasets/
│   │   ├── losses/           # Loss functions and SeparationLossHandler
│   │   ├── models/           # Model definitions
│   │   │   ├── common_components/ # Reusable model components (STFT, bandsplit, TF models, mask estimation)
│   │   │   ├── core_models/     # Main separator models (BaseSeparator, Bandit, Banquet)
│   │   │   └── query_models/    # Query-specific model components
│   │   ├── utils/            # Utility functions (logging, registry)
│   │   └── train.py          # Main training script
│   └── tests/                # Unit and integration tests
├── .gitignore
├── pyproject.toml            # Project metadata and dependencies
├── README.md                 # Project overview (this file)
└── CONTRIBUTING.md           # Guidelines for contributing
```

## Contributing

We welcome contributions to Banda! Please see `CONTRIBUTING.md` for guidelines on how to contribute.

## License

This project is dual-licensed:

1.  **GNU Affero General Public License v3.0 (AGPLv3)** for academic and non-commercial research use. For details, see [https://www.gnu.org/licenses/agpl-3.0.en.html](https://www.gnu.org/licenses/agpl-3.0.en.html)
2.  **Commercial License** for all other uses. Contact kwatcharasupat \[at] ieee.org for commercial licensing.

## Acknowledgements

(Add acknowledgements here, e.g., for datasets, libraries, or research papers that inspired the project.)