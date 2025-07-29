# Bandsplit System Documentation

This document describes the bandsplit system implemented in the `banda` project, which is crucial for spectral processing in source separation tasks. The system provides a flexible way to define and manage frequency bands for various models, including the Bandit model.

## Core Concepts

The bandsplit system is built around the `BandsplitSpecification` abstract base class, which defines the interface for all bandsplit types. Concrete implementations derive from this base class, providing specific logic for generating frequency bands.

### `BandsplitSpecification`

- **Purpose**: Defines the common interface for all bandsplit specifications.
- **Key Methods**:
    - `index_to_hertz(index: int)`: Converts a frequency bin index to its corresponding frequency in Hertz.
    - `hertz_to_index(hz: float, round: bool = True)`: Converts a frequency in Hertz to its corresponding frequency bin index.
    - `get_band_specs_with_bandwidth(start_index, end_index, bandwidth_hz)`: A helper method to generate sub-bands within a given range based on a specified bandwidth.
    - `get_band_specs() -> List[Tuple[int, int]]`: **Abstract method** that must be implemented by subclasses to return a list of frequency band tuples, where each tuple `(start_index, end_index)` represents the start and end frequency bin indices for a band.
    - `_validate_band_specs(band_specs: List[Tuple[int, int]])`: Validates the generated band specifications, checking for potential issues like very narrow bands between much larger ones.

## Types of Bandsplit Specifications

The system supports several types of bandsplit specifications, categorized into fixed and perceptual bands.

### Fixed Bandsplit Specifications

These specifications define frequency bands based on predefined, fixed frequency ranges, often derived from domain-specific knowledge (e.g., vocal, bass, drum, other instrument ranges).

#### `FixedBandsplitSpecs`

- **Purpose**: A base class for fixed bandsplit specifications. It reads band definitions from a configuration (typically a YAML file).
- **Configuration**: Expects a `DictConfig` object containing a `bands` key, which is a list of dictionaries. Each dictionary defines a band with `name`, `start_hz`, `end_hz`, and `bandwidth_hz` (optional).
- **Subclasses**:
    - `VocalBandsplitSpecification`: Defines bands for vocal frequencies. Supports different versions of vocal band definitions.
    - `BassBandsplitSpecification`: Defines bands for bass frequencies.
    - `DrumBandsplitSpecification`: Defines bands for drum frequencies.
    - `OtherBandsplitSpecification`: Defines bands for other instrument frequencies.

**Example YAML Configuration for Fixed Bands (e.g., `vocal.yaml`):**

```yaml
# src/banda/configs/bandsplit/fixed_bands/vocal.yaml
version7:
  bands:
    - name: "V1"
      start_hz: 0
      end_hz: 100
      bandwidth_hz: 10
    - name: "V2"
      start_hz: 100
      end_hz: 200
      bandwidth_hz: 20
    # ... more bands
```

### Perceptual Bandsplit Specifications

These specifications define frequency bands based on human auditory perception, often using non-linear scales.

#### `PerceptualBandsplitSpecification`

- **Purpose**: A base class for perceptual bandsplit specifications. It takes a filterbank function as input to generate the bands.
- **Key Attributes**:
    - `filterbank`: A `torch.Tensor` representing the filterbank used to derive the bands.
    - `freq_weights`: A list of `torch.Tensor` objects, where each tensor contains the frequency weights for a specific band.
- **Subclasses**:
    - `MusicalBandsplitSpecification`: Bands based on a musical scale.
    - `MelBandsplitSpecification`: Bands based on the Mel scale.
    - `BarkBandsplitSpecification`: Bands based on the Bark scale.
    - `TriangularBarkBandsplitSpecification`: Bands based on a triangular Bark filterbank.
    - `MiniBarkBandsplitSpecification`: A variation of Bark bands with a thresholding mechanism.
    - `EquivalentRectangularBandsplitSpecification`: Bands based on the Equivalent Rectangular Band (ERB) scale.

## Bandsplit Factory

The `get_bandsplit_specs_factory` function serves as a central factory for instantiating different `BandsplitSpecification` objects based on the provided `bandsplit_type`.

```python
def get_bandsplit_specs_factory(
    bandsplit_type: str,
    n_fft: int,
    fs: int,
    n_bands: Optional[int] = None,
    version: Optional[str] = None,
    fixed_bands_config: Optional[DictConfig] = None
) -> List[Tuple[int, int]]:
    """
    Factory function to create and return band specifications based on type.
    """
    # ... implementation details ...
```

- **`bandsplit_type`**: A string indicating the type of bandsplit (e.g., "vocal", "bass", "musical", "mel").
- **`n_fft`**: The number of FFT points.
- **`fs`**: The sampling rate.
- **`n_bands`**: (Optional) The number of bands for perceptual bandsplit types.
- **`version`**: (Optional) The version of the fixed bandsplit (e.g., "7" for vocal bands).
- **`fixed_bands_config`**: (Optional) A `DictConfig` object containing the configuration for fixed bandsplit types (loaded from YAML).

## Usage in Hydra Configuration

To use the bandsplit system in your Hydra configurations, you can leverage the `get_bandsplit_specs_factory` function with the `_target_` key.

**Example (`train_bandit_musdb18hq_test.yaml`):**

```yaml
model:
  # ... other model parameters ...
  band_specs:
    _target_: banda.utils.spectral.get_bandsplit_specs_factory
    bandsplit_type: "vocal" # Or "bass", "drum", "other", "musical", "mel", etc.
    n_fft: ${model.n_fft}
    fs: ${model.fs}
    # For fixed bandsplit types (vocal, bass, drum, other):
    fixed_bands_config:
      _target_: omegaconf.OmegaConf.load
      _args_:
        - ${project_root}/src/banda/configs/bandsplit/fixed_bands/vocal.yaml
    # For vocal bands, specify the version:
    version: "7"
    # For perceptual bandsplit types (musical, mel, bark, tribark, erb, minibark):
    # n_bands: 64 # Example for musical bandsplit
```

This setup allows for flexible and dynamic configuration of bandsplit specifications directly within your Hydra YAML files, enabling easy experimentation with different frequency band definitions.