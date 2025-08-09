import torch
import matplotlib.pyplot as plt
import numpy as np
import hydra
from omegaconf import OmegaConf

from banda.models.common_components.spectral_components.spectral_base import (
    BandsplitSpecification,
)
from banda.models.common_components.configs.bandsplit_configs._bandsplit_models import (
    MusicalBandsplitSpecsConfig,
    FixedBandsplitSpecsConfig,
    VocalBandsplitSpecsConfig,
    MelBandsplitSpecsConfig,
    TriangularBarkBandsplitSpecsConfig,
    EquivalentRectangularBandsplitSpecsConfig,
    BandsplitType,
    BANDSPLIT_CONFIG_MAP, # Import BANDSPLIT_CONFIG_MAP
)
from banda.models.common_components.spectral_components.bandsplit_registry import bandsplit_registry


def plot_bandsplit_assignment(bandsplit_type: BandsplitType, nfft: int, fs: int, n_bands: int, drop_dc_band: bool = False):
    """
    Generates and plots the frequency bin assignment for a given BandsplitSpecification type.
    """
    print(f"Generating plot for BandsplitType: {bandsplit_type.value}, nfft={nfft}, fs={fs}, n_bands={n_bands}, drop_dc_band={drop_dc_band}")

    # Determine the correct config type based on bandsplit_type using BANDSPLIT_CONFIG_MAP
    config_class = BANDSPLIT_CONFIG_MAP.get(bandsplit_type)
    if config_class is None:
        raise ValueError(f"No config class found for BandsplitType: {bandsplit_type}")

    if config_class == VocalBandsplitSpecsConfig:
        config_instance = config_class(version="7")
    elif config_class == FixedBandsplitSpecsConfig:
        # For FixedBandsplitSpecs, we need to provide some dummy bands
        config_instance = config_class(bands=[
            {"start_hz": 20.0, "end_hz": 500.0, "bandwidth_hz": None},
            {"start_hz": 500.0, "end_hz": 2000.0, "bandwidth_hz": None},
            {"start_hz": 2000.0, "end_hz": fs / 2, "bandwidth_hz": None},
        ])
    else: # Perceptual bands
        config_instance = config_class(n_bands=n_bands)
    
    # Instantiate the BandsplitSpecification using the registry
    bandsplit_specs = bandsplit_registry.get(bandsplit_type)(
        nfft=nfft,
        fs=fs,
        config=config_instance,
        drop_dc_band=drop_dc_band,
    )

    band_specs = bandsplit_specs.get_band_specs()

    # Prepare data for plotting
    max_freq_bin = (nfft // 2 + 1) - (1 if drop_dc_band else 0)
    bin_assignments = np.full(max_freq_bin, -1, dtype=int) # -1 for unassigned

    for i, (start_idx, end_idx) in enumerate(band_specs):
        for j in range(start_idx, end_idx):
            if j < max_freq_bin:
                bin_assignments[j] = i

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Create a colormap for bands
    cmap = plt.colormaps.get_cmap('tab20')
    
    for i in range(len(band_specs)):
        band_bins = np.where(bin_assignments == i)[0]
        if len(band_bins) > 0:
            plt.scatter(band_bins, np.full_like(band_bins, i), color=cmap(i), label=f'Band {i}', s=10)

    # Plot the boundaries of each band
    for i, (start_idx, end_idx) in enumerate(band_specs):
        plt.axvspan(start_idx - 0.5, end_idx - 0.5, color=cmap(i), alpha=0.1, lw=0)
        plt.text((start_idx + end_idx) / 2 - 0.5, i + 0.2, f'Band {i}', ha='center', va='bottom', fontsize=8, color=cmap(i))


    plt.title(f'Frequency Bin Assignment for {bandsplit_type.value} (n_bands={len(band_specs)}, drop_dc_band={drop_dc_band})')
    plt.xlabel('Frequency Bin Index')
    plt.ylabel('Band Index')
    plt.yticks(range(len(band_specs)))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([-0.5, max_freq_bin - 0.5])
    plt.ylim([-0.5, len(band_specs) - 0.5 + 1])
    plt.tight_layout()
    plt.savefig(f'bandsplit_assignment_{bandsplit_type.value}_n{nfft}_b{len(band_specs)}_dc{drop_dc_band}.png')
    plt.close()
    print(f"Plot saved as bandsplit_assignment_{bandsplit_type.value}_n{nfft}_b{len(band_specs)}_dc{drop_dc_band}.png")

if __name__ == "__main__":
    N_FFT = 2048
    FS = 44100
    
    # Focus on music, bark, and mel bands as requested
    bands_to_test = {
        # BandsplitType.MUSIC: [48, 64],
        # BandsplitType.TRIBARK: [48, 64],
        # BandsplitType.MEL: [48, 64],
        BandsplitType.ERB: [48, 64],
    }

    for bs_type, n_bands_options in bands_to_test.items():
        for n_bands in n_bands_options:
            plot_bandsplit_assignment(bs_type, N_FFT, FS, n_bands, drop_dc_band=True)