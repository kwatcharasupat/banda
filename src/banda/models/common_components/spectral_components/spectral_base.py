#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

from typing import Any, Dict, Optional, Tuple, Union, List
import torch
from torch import nn
from pydantic import BaseModel, Field, ConfigDict
import warnings

# from banda.utils.registry import MODELS_REGISTRY # Removed as per previous instructions

class STFTConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    target_: str = Field(alias="_target_") # Changed _target_ to target_
    n_fft: int
    hop_length: int
    win_length: Optional[int] = None
    window: str = "hann"
    center: bool = True
    pad_mode: str = "reflect"
    freeze_parameters: bool = True

class SpectralComponent(nn.Module):
    """
    Base class for spectral processing components.
    """
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: Optional[int] = None,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        freeze_parameters: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.freeze_parameters = freeze_parameters

        # Create the window function
        self.register_buffer("window_fn", torch.hann_window(self.win_length), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: STFTConfig) -> "SpectralComponent":
        """
        Instantiates SpectralComponent from an STFTConfig Pydantic model.
        """
        return cls(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            window=config.window,
            center=config.center,
            pad_mode=config.pad_mode,
            freeze_parameters=config.freeze_parameters,
        )

class BandsplitSpecification(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    nfft: int
    fs: int
    drop_dc_band: bool = False

    def hertz_to_index(self, hertz: float) -> int:
        return int(hertz / (self.fs / self.nfft))

    def index_to_hertz(self, index: int) -> float:
        return index * (self.fs / self.nfft)

    @property
    def max_index(self) -> int:
        return self.nfft // 2 + 1

    def get_band_specs(self) -> List[Tuple[int, int]]:
        raise NotImplementedError

    def _validate_band_specs(self, bands: List[Tuple[int, int]]) -> None:
        # Basic validation: ensure bands are within bounds and sorted
        for start, end in bands:
            if not (0 <= start <= end < self.max_index):
                raise ValueError(f"Band indices out of bounds: ({start}, {end}) for max_index {self.max_index}")
        
        # Check for overlaps (simple check for now)
        sorted_bands = sorted(bands)
        for i in range(len(sorted_bands) - 1):
            if sorted_bands[i][1] > sorted_bands[i+1][0]:
                warnings.warn(f"Overlapping bands detected: {sorted_bands[i]} and {sorted_bands[i+1]}")

    def get_band_specs_with_bandwidth(self, start_index: int, end_index: int, bandwidth_hz: float) -> List[Tuple[int, int]]:
        bands = []
        current_index = start_index
        bandwidth_index = self.hertz_to_index(bandwidth_hz)

        while current_index < end_index:
            band_end_index = min(current_index + bandwidth_index, end_index)
            bands.append((current_index, band_end_index))
            current_index = band_end_index
        return bands

# Helper functions for frequency conversions and filterbank creation
def hz_to_bark(hz: torch.Tensor) -> torch.Tensor:
    return 26.81 * hz / (1960 + hz) - 0.53

def bark_to_hz(bark: torch.Tensor) -> torch.Tensor:
    return 1960 * (bark + 0.53) / (26.81 - (bark + 0.53))

def hz_to_midi(hz: torch.Tensor) -> torch.Tensor:
    return 12 * torch.log2(hz / 440) + 69

def midi_to_hz(midi: torch.Tensor) -> torch.Tensor:
    return 440 * torch.pow(2, (midi - 69) / 12)

def hz_to_erb(hz: torch.Tensor) -> torch.Tensor:
    return 21.4 * torch.log10(1 + hz / 229)

def _create_triangular_filterbank(all_freqs: torch.Tensor, f_pts: torch.Tensor) -> torch.Tensor:
    """
    Creates a triangular filterbank.
    Args:
        all_freqs (torch.Tensor): All frequency bins (Hz) for the spectrogram.
        f_pts (torch.Tensor): Center frequencies of the triangular filters.
    Returns:
        torch.Tensor: Filterbank matrix of shape (n_bands, n_freq_bins).
    """
    n_bands = len(f_pts) - 2 # Number of bands is number of points - 2 (for start and end)
    n_freq_bins = len(all_freqs)
    filterbank = torch.zeros((n_bands, n_freq_bins))

    for i in range(n_bands):
        left_f = f_pts[i]
        center_f = f_pts[i+1]
        right_f = f_pts[i+2]

        # Create the triangular filter
        for j, freq in enumerate(all_freqs):
            if left_f <= freq <= center_f:
                filterbank[i, j] = (freq - left_f) / (center_f - left_f)
            elif center_f < freq <= right_f:
                filterbank[i, j] = (right_f - freq) / (right_f - center_f)
            else:
                filterbank[i, j] = 0.0
    
    # Ensure the first bin of the first band is active if it covers 0 Hz
    if n_bands > 0 and f_pts[0] == 0.0:
        filterbank[0, 0] = 1.0

    return filterbank

# No model_rebuild() calls needed here