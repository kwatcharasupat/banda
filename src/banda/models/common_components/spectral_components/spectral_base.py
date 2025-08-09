import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta
import torchaudio.functional as taF # Import torchaudio.functional

from omegaconf import DictConfig, OmegaConf # Import OmegaConf
import structlog
import logging

logger = structlog.get_logger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)

from banda.models.common_components.configs.bandsplit_configs._bandsplit_models import (
    BandDefinitionConfig,
    FixedBandsplitSpecsConfig,
    VocalBandsplitSpecsConfig,
    PerceptualBandsplitSpecsConfig,
    MusicalBandsplitSpecsConfig,
    MelBandsplitSpecsConfig,
    BarkBandsplitSpecsConfig,
    TriangularBarkBandsplitSpecsConfig,
    MiniBarkBandsplitSpecsConfig,
    EquivalentRectangularBandsplitSpecsConfig, # Added this import
    BandsplitConfig,
    BandsplitType, # Import BandsplitType from here
)


# Add these helper functions if not already defined
def hz_to_bark(hz: float) -> float:
    return 6 * torch.as_tensor(hz / 600).atanh()

def hz_to_erb(hz: float) -> float:
    return 21.4 * torch.log10(1 + 0.00437 * hz)

class SpectralComponent(nn.Module):
    """
    A module for performing Short-Time Fourier Transform (STFT) and Inverse STFT (ISTFT).
    """
    def __init__(
            self,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None, # Set to None for complex spectrogram
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
            **kwargs,
    ) -> None:
        super().__init__()

        # Ensure power is None for complex spectrograms
        assert power is None, "power must be None for complex spectrograms."

        # Get window function from torch
        window_function = torch.__dict__[window_fn]

        self.stft = ta.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad_mode=pad_mode,
            pad=0, # Padding is handled by center=True
            window_fn=window_function,
            wkwargs=wkwargs,
            power=power,
            normalized=normalized,
            center=center,
            onesided=onesided,
        )

        self.istft = ta.transforms.InverseSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad_mode=pad_mode,
            pad=0, # Padding is handled by center=True
            window_fn=window_function,
            wkwargs=wkwargs,
            normalized=normalized,
            center=center,
            onesided=onesided,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for STFT.

        Args:
            x (torch.Tensor): Input audio waveform. Shape: (batch_size, channels, samples)

        Returns:
            torch.Tensor: Complex spectrogram. Shape: (batch_size, channels, freq_bins, time_frames, 2) # UPDATED DOCSTRING
        """
        # Ensure audio is float32
        x = x.to(torch.float32)

        # torchaudio.Spectrogram expects (..., time)
        # Our input is (batch, channels, samples)
        # We need to reshape to (batch * channels, samples) for STFT, then reshape back
        batch_size, num_channels, num_samples = x.shape
        x = x.view(batch_size * num_channels, num_samples)
        
        complex_spec = self.stft(x) # (batch * channels, freq_bins, time_frames) - complex tensor

        # Reshape back to (batch, channels, freq_bins, time_frames)
        complex_spec = complex_spec.view(batch_size, num_channels, complex_spec.shape[-2], complex_spec.shape[-1])

        # Convert complex tensor to real tensor with last dim for real/imag
        # Shape: (batch_size, channels, freq_bins, time_frames, 2)
        real_spec = torch.view_as_real(complex_spec)
        return real_spec

    def inverse(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """
        Inverse pass for ISTFT.

        Args:
            spec (torch.Tensor): Real spectrogram with last dim for real/imag. Shape: (batch_size, channels, freq_bins, time_frames, 2) # UPDATED DOCSTRING
            length (int): Original audio length for ISTFT.

        Returns:
            torch.Tensor: Reconstructed audio waveform. Shape: (batch_size, channels, samples)
        """
        batch_size, num_channels, freq_bins, time_frames, _ = spec.shape # Added _ for last dim
        
        # Convert real tensor (with last dim for real/imag) to complex tensor
        complex_spec = torch.view_as_complex(spec)

        # Reshape complex_spec for ISTFT: (batch_size * num_channels, freq_bins, time_frames)
        complex_spec_reshaped = complex_spec.view(batch_size * num_channels, freq_bins, time_frames)
        
        audio_reshaped = self.istft(complex_spec_reshaped, length=length) # (batch_size * num_channels, samples)
        
        # Reshape back to (batch, channels, samples)
        audio = audio_reshaped.view(batch_size, num_channels, audio_reshaped.shape[-1])
        return audio

# Custom helper functions for frequency and MIDI conversion
def hz_to_midi(frequencies: torch.Tensor) -> torch.Tensor:
    """
    Converts frequencies (Hz) to MIDI note numbers.
    Formula: 69 + 12 * log2(frequencies / 440)
    Handles zero or negative frequencies by clamping to a small positive value.
    """
    # Clamp frequencies to a small positive value to avoid log(0) or log(negative)
    min_freq = 1e-6 # A very small positive frequency
    clamped_frequencies = torch.clamp(frequencies, min=min_freq)
    return 69 + 12 * torch.log2(clamped_frequencies / 440.0)

def midi_to_hz(midi_notes: torch.Tensor) -> torch.Tensor:
    """
    Converts MIDI note numbers to frequencies (Hz).
    Formula: 440 * 2^((midi_notes - 69) / 12)
    """
    return 440.0 * torch.pow(2, (midi_notes - 69) / 12.0)

# Helper function from torchaudio.functional.functional._create_triangular_filterbank
# This is a private function, so we'll re-implement it here.
def _create_triangular_filterbank(all_freqs: torch.Tensor, f_pts: torch.Tensor) -> torch.Tensor:
    """
    Creates a triangular filterbank.
    Args:
        all_freqs (torch.Tensor): All frequency bins.
        f_pts (torch.Tensor): Filterbank center frequencies.
    Returns:
        torch.Tensor: Triangular filterbank.
    """
    # filterbank: (n_mels, n_freqs)
    n_mels = f_pts.shape[0] - 2
    n_freqs = all_freqs.shape[0]
    filterbank = torch.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        left_f = f_pts[i]
        center_f = f_pts[i + 1]
        right_f = f_pts[i + 2]

        # Left slope
        for j in range(n_freqs):
            if left_f <= all_freqs[j] <= center_f:
                filterbank[i, j] = (all_freqs[j] - left_f) / (center_f - left_f)
            elif center_f < all_freqs[j] <= right_f:
                filterbank[i, j] = (right_f - all_freqs[j]) / (right_f - center_f)
    return filterbank


class BandsplitSpecification(ABC):
    def __init__(self, nfft: int, fs: int, drop_dc_band: bool = False) -> None:
        self.fs = fs
        self.nfft = nfft
        self.nyquist = fs / 2
        self.drop_dc_band = drop_dc_band
        # max_index is the number of frequency bins relevant for bandsplit
        # If DC band is dropped, we have one less bin
        self.max_index = (nfft // 2 + 1) - (1 if drop_dc_band else 0)

    @staticmethod
    def index_to_hertz(index: int, nfft: int, fs: int, drop_dc_band: bool):
        # If DC band is dropped, index 0 now corresponds to original index 1
        adjusted_index = index + (1 if drop_dc_band else 0)
        return adjusted_index * fs / nfft

    def hertz_to_index(self, hz: float, round: bool = True):
        # Calculate original index
        original_index = hz * self.nfft / self.fs

        if round:
            original_index = int(torch.round(torch.tensor(original_index)).item())
        
        # If DC band is dropped, and the original index is not 0 (DC), then subtract 1
        if self.drop_dc_band and original_index > 0:
            return original_index - 1
        elif self.drop_dc_band and original_index == 0:
            # If the frequency is 0 Hz (DC), and we are dropping the DC band,
            # this frequency should not be mapped to an index in the processed spectrum.
            # This case should ideally be avoided by ensuring f_min is not 0 when drop_dc_band is True.
            raise ValueError("Attempted to convert DC frequency to index when DC band is dropped.")
        return original_index

    def get_band_specs_with_bandwidth(
            self,
            start_index,
            end_index,
            bandwidth_hz
            ):
        band_specs = []
        lower = start_index
        while lower < end_index:
            upper = int(torch.floor(torch.tensor(lower + self.hertz_to_index(bandwidth_hz, round=False))).item())
            upper = min(upper, end_index)

            band_specs.append((lower, upper))
            lower = upper

        return band_specs

    @abstractmethod
    def get_band_specs(self) -> List[Tuple[int, int]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_overlapping(self) -> bool:
        """
        Returns True if the bandsplit specification defines overlapping bands, False otherwise.
        """
        raise NotImplementedError

    def _validate_band_specs(self, band_specs: List[Tuple[int, int]]) -> None:
        """
        Validates the generated band specifications, including checking for
        bands with only one or two bins between much larger bands.
        """
        # Implement the safeguard logic here
        # Check for bands with only one or two bins between much larger bands
        # For now, a basic check, can be refined later.
        for i in range(1, len(band_specs) - 1):
            prev_band_width = band_specs[i][1] - band_specs[i][0]
            current_band_width = band_specs[i+1][1] - band_specs[i+1][0]
            if prev_band_width > 0 and current_band_width > 0:
                ratio = max(prev_band_width, current_band_width) / min(prev_band_width, current_band_width)
                if ratio > 5 and (band_specs[i][1] - band_specs[i][0] <= 2 or band_specs[i+1][1] - band_specs[i+1][0] <= 2):
                    logger.warning(f"Potential problematic band size detected between {band_specs[i]} and {band_specs[i+1]}")


class FixedBandsplitSpecs(BandsplitSpecification):
    def __init__(self, nfft: int, fs: int, config: FixedBandsplitSpecsConfig, drop_dc_band: bool = False) -> None:
        super().__init__(nfft=nfft, fs=fs, drop_dc_band=drop_dc_band)
        self.config = config
        self.bands_config = config.bands # Expecting a list of band definitions

    def get_band_specs(self) -> List[Tuple[int, int]]:
        bands = []
        for band_def in self.bands_config:
            start_hz = band_def.start_hz
            end_hz = band_def.end_hz
            bandwidth_hz = band_def.bandwidth_hz

            start_index = self.hertz_to_index(start_hz)
            end_index = self.hertz_to_index(end_hz) if end_hz is not None else self.max_index

            if bandwidth_hz is None:
                # Treat as one subband
                bands.append((start_index, end_index))
            else:
                # Split by bandwidth
                bands.extend(self.get_band_specs_with_bandwidth(start_index, end_index, bandwidth_hz))
        
        self._validate_band_specs(bands)
        return bands

    @property
    def is_overlapping(self) -> bool:
        return False # Fixed bands are typically not overlapping


class VocalBandsplitSpecification(FixedBandsplitSpecs):
    def __init__(self, nfft: int, fs: int, config: VocalBandsplitSpecsConfig, version: str = "7", drop_dc_band: bool = False, require_no_overlap: bool = False) -> None: # Added require_no_overlap
        # Load bands from YAML file
        script_dir = os.path.dirname(__file__)
        yaml_path = os.path.join(script_dir, "..", "configs", "bandsplit_configs", "vocal_bands.yaml")
        vocal_bands_config_data = OmegaConf.load(yaml_path)
        
        # Dynamically select the version's bands from the loaded config
        version_config_data = vocal_bands_config_data.get(f"version{version}")
        if version_config_data is None:
            raise ValueError(f"Configuration for Vocal Bandsplit version {version} not found in {yaml_path}.")
        
        # Convert DictConfig to list of BandDefinitionConfig
        bands_data = [BandDefinitionConfig(**band) for band in version_config_data.bands]

        # Adjust start_hz of the first band if drop_dc_band is True and start_hz is 0.0
        if drop_dc_band and bands_data[0].start_hz == 0.0:
            bands_data[0].start_hz = BandsplitSpecification.index_to_hertz(0, nfft=nfft, fs=fs, drop_dc_band=True)
            logger.info(f"Adjusted first band start_hz to {bands_data[0].start_hz} due to drop_dc_band=True.")

        # Create a FixedBandsplitSpecsConfig with the dynamically loaded bands
        fixed_bands_config = FixedBandsplitSpecsConfig(bands=bands_data)
        
        # Pass this new config to the superclass
        super().__init__(nfft=nfft, fs=fs, config=fixed_bands_config, drop_dc_band=drop_dc_band)
        self.version = version
        self.require_no_overlap = require_no_overlap # Store the parameter

    def get_band_specs(self) -> List[Tuple[int, int]]:
        # This logic is now handled in __init__ by creating a FixedBandsplitSpecsConfig
        # So, we can just call the superclass method directly.
        bands = super().get_band_specs()
        return bands


class OtherBandsplitSpecification(FixedBandsplitSpecs):
    def __init__(self, nfft: int, fs: int, config: FixedBandsplitSpecsConfig, drop_dc_band: bool = False) -> None:
        super().__init__(nfft=nfft, fs=fs, config=config, drop_dc_band=drop_dc_band)

    def get_band_specs(self) -> List[Tuple[int, int]]:
        # Other bandsplit will directly use the bands defined in its config
        bands = super().get_band_specs()
        return bands


class BassBandsplitSpecification(FixedBandsplitSpecs):
    def __init__(self, nfft: int, fs: int, config: FixedBandsplitSpecsConfig, drop_dc_band: bool = False) -> None:
        super().__init__(nfft=nfft, fs=fs, config=config, drop_dc_band=drop_dc_band)

    def get_band_specs(self) -> List[Tuple[int, int]]:
        # Bass bandsplit will directly use the bands defined in its config
        bands = super().get_band_specs()
        return bands


class DrumBandsplitSpecification(FixedBandsplitSpecs):
    def __init__(self, nfft: int, fs: int, config: FixedBandsplitSpecsConfig, drop_dc_band: bool = False) -> None:
        super().__init__(nfft=nfft, fs=fs, config=config, drop_dc_band=drop_dc_band)

    def get_band_specs(self) -> List[Tuple[int, int]]:
        # Drum bandsplit will directly use the bands defined in its config
        bands = super().get_band_specs()
        return bands
    @abstractmethod
    def _get_fbank_fn(self) -> Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]:
        raise NotImplementedError


class PerceptualBandsplitSpecification(BandsplitSpecification):
    def __init__(
            self,
            nfft: int,
            fs: int,
            config: PerceptualBandsplitSpecsConfig,
            drop_dc_band: bool = False,
            require_no_overlap: bool = False # Add this parameter
    ) -> None:
        super().__init__(nfft=nfft, fs=fs, drop_dc_band=drop_dc_band)
        self.config = config # Store the config
        self.n_bands = config.n_bands
        f_min = config.f_min
        f_max = config.f_max
        self.require_no_overlap = require_no_overlap # Store the parameter
        fbank_fn = self._get_fbank_fn() # This will be implemented in subclasses

        if f_max is None:
            f_max = fs / 2

        # Adjust f_min based on drop_dc_band
        if self.drop_dc_band and f_min < BandsplitSpecification.index_to_hertz(0, nfft=self.nfft, fs=self.fs, drop_dc_band=self.drop_dc_band):
            f_min = BandsplitSpecification.index_to_hertz(0, nfft=self.nfft, fs=self.fs, drop_dc_band=self.drop_dc_band)

        # Construct all_freqs for the filterbank based on the new spectrum
        # This will be nfft // 2 if drop_dc_band is True, otherwise nfft // 2 + 1
        n_freqs_for_fbank = self.max_index 
        all_freqs_for_fbank = torch.linspace(self.index_to_hertz(0, nfft=self.nfft, fs=self.fs, drop_dc_band=self.drop_dc_band), self.index_to_hertz(n_freqs_for_fbank - 1, nfft=self.nfft, fs=self.fs, drop_dc_band=self.drop_dc_band), n_freqs_for_fbank)

        # Calculate MIDI points for the musical bands based on the full frequency range
        low_midi = hz_to_midi(torch.tensor(f_min)).item()
        high_midi = hz_to_midi(torch.tensor(f_max)).item()
        midi_points = torch.linspace(low_midi, high_midi, self.n_bands + 2) # Use self.n_bands

        # Convert MIDI points back to Hz
        hz_points = midi_to_hz(midi_points)

        self.filterbank = fbank_fn(
                all_freqs_for_fbank, hz_points, self.n_bands # Use self.n_bands
        )

        # Explicitly ensure the first band covers the first frequency bin if DC band is dropped
        if self.drop_dc_band:
            # Ensure the first band starts at index 0 (which is original bin 1)
            # and has some weight at that bin.
            if self.filterbank.shape[1] > 0: # Check if there are frequency bins
                self.filterbank[0, 0] = 1.0 # Force the first band to cover the first bin

        weight_per_bin = torch.sum(
            self.filterbank,
            dim=0,
            keepdim=True
            )  # (1, n_freqs)
        # Handle division by zero for bins with no weight
        epsilon = 1e-10 # Define epsilon here
        normalized_mel_fb = torch.where(weight_per_bin > 0, self.filterbank / (weight_per_bin + epsilon), torch.zeros_like(self.filterbank)) # (n_mels, n_freqs)

        band_specs = []
        
        # Track covered bins to ensure full coverage
        covered_bins = set()

        for i in range(self.n_bands):
            active_bins = torch.nonzero(self.filterbank[i, :]).squeeze()
            if active_bins.dim() == 0: # Handle case where squeeze results in a 0-dim tensor
                active_bins = active_bins.unsqueeze(0)
            active_bins = active_bins.tolist()

            if len(active_bins) == 0:
                logger.warning(f"PerceptualBandsplitSpecification: Band {i} has no active bins. Skipping this band.")
                continue
            
            start_index = max(0, active_bins[0]) # Ensure start_index is not negative
            end_index = min(active_bins[-1] + 1, self.max_index) # Cap end_index at self.max_index
            
            # If require_no_overlap is True, ensure bands do not overlap
            if self.require_no_overlap and band_specs and start_index < band_specs[-1][1]:
                logger.warning(f"Adjusting overlapping band {start_index}-{end_index} to {band_specs[-1][1]}-{end_index} due to require_no_overlap=True.")
                start_index = band_specs[-1][1] # Start current band from end of previous band

            if start_index >= end_index:
                logger.warning(f"PerceptualBandsplitSpecification: Calculated band ({start_index}, {end_index}) has zero or negative bandwidth. Skipping.")
                continue

            band_specs.append((start_index, end_index))
            
            covered_bins.update(range(start_index, end_index))

        # After generating all bands, check for any gaps and add them as single-bin bands
        expected_bins_range = set(range(self.max_index))
        # Removed the problematic line: if self.drop_dc_band: expected_bins_range = set(range(1, self.max_index + 1)) # Adjust for dropped DC band

        uncovered_bins = sorted(list(expected_bins_range - covered_bins))
        
        for bin_idx in uncovered_bins:
            # Add single-bin bands for any uncovered bins
            band_specs.append((bin_idx, min(bin_idx + 1, self.max_index))) # Cap bin_idx + 1 at self.max_index

        # Sort band_specs by start_index to maintain order
        band_specs.sort(key=lambda x: x[0])

        self.band_specs = band_specs

    def get_band_specs(self) -> List[Tuple[int, int]]:
        self._validate_band_specs(self.band_specs)
        total_bandwidth = sum([fend - fstart for fstart, fend in self.band_specs])
        logger.debug(f"PerceptualBandsplitSpecification: Total bandwidth of band_specs: {total_bandwidth}")
        return self.band_specs

    @property
    def is_overlapping(self) -> bool:
        return not self.require_no_overlap # Now determined by the parameter

    def get_freq_weights(self) -> List[torch.Tensor]:
        return self.freq_weights


def musical_filterbank(all_freqs: torch.Tensor, hz_points: torch.Tensor, n_bands: int, scale="constant"): # Modified signature
    # Create the triangular filterbank
    fb = _create_triangular_filterbank(all_freqs, hz_points)

    # Normalize the filterbank so that each frequency bin sums to 1 across bands
    # This ensures that the sum of masks for each frequency bin is 1, preventing
    # amplitude changes due to the bandsplit.
    # Add a small epsilon to avoid division by zero for bins with no weight
    epsilon = 1e-10
    fb = fb / (fb.sum(axis=0, keepdims=True) + epsilon)

    return fb


class MusicalBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(
            self,
            nfft: int,
            fs: int,
            config: MusicalBandsplitSpecsConfig, # Changed to use config
            drop_dc_band: bool = False,
            require_no_overlap: bool = False # Add this parameter
    ) -> None:
        super().__init__(nfft=nfft, fs=fs, config=config, drop_dc_band=drop_dc_band, require_no_overlap=require_no_overlap)

    def _get_fbank_fn(self) -> Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]:
        return musical_filterbank


def mel_filterbank(all_freqs: torch.Tensor, hz_points: torch.Tensor, n_bands: int, fs: int): # Modified signature to include fs
    # Implement melscale_fbanks using torchaudio.functional
    # n_freqs is the number of frequency bins in the spectrogram, which is all_freqs.shape[0]
    n_freqs = all_freqs.shape[0]
    
    # f_min and f_max are derived from hz_points
    f_min = hz_points[0].item()
    f_max = hz_points[-1].item()

    fb = taF.melscale_fbanks(
                n_mels=n_bands,
                sample_rate=fs,
                f_min=f_min,
                f_max=f_max,
                n_freqs=n_freqs, # Changed n_fft to n_freqs
                norm="slaney", # Common normalization for Mel filterbanks
                mel_scale="htk", # Common mel scale for Mel filterbanks
        ).T # Transpose to get (n_mels, n_freqs)

    # Ensure that the sum of the filterbank weights for each frequency bin is 1
    # This ensures that the sum of masks for each frequency bin is 1, preventing
    # amplitude changes due to the bandsplit.
    # Add a small epsilon to avoid division by zero for bins with no weight
    epsilon = 1e-10
    fb = fb / (fb.sum(axis=0, keepdims=True) + epsilon)

    return fb


class MelBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(
            self,
            nfft: int,
            fs: int,
            config: MelBandsplitSpecsConfig, # Changed to use config
            drop_dc_band: bool = False,
            require_no_overlap: bool = False # Add this parameter
    ) -> None:
        super().__init__(nfft=nfft, fs=fs, config=config, drop_dc_band=drop_dc_band, require_no_overlap=require_no_overlap)

    def _get_fbank_fn(self) -> Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]:
        # Return a lambda to pass fs to mel_filterbank
        return lambda all_freqs, hz_points, n_bands: mel_filterbank(all_freqs, hz_points, n_bands, self.fs)


def bark_filterbank(all_freqs: torch.Tensor, hz_points: torch.Tensor, n_bands: int): # Modified signature
    # Re-implement bark_filter_banks if bark_fbanks is not available or desired
    # For now, keeping the original as it's commented out anyway.
    # If this becomes an issue, a custom implementation will be needed.
    # nfft = 2 * (n_freqs -1)
    # fb, _ = bark_fbanks.bark_filter_banks(
    #         nfilts=n_bands,
    #         nfft=nfft,
    #         fs=fs,
    #         low_freq=f_min,
    #         high_freq=f_max,
    #         scale="constant"
    # )
    # return torch.as_as_tensor(fb)
    raise NotImplementedError("Bark filterbank not implemented without bark_fbanks")


class BarkBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(
            self,
            nfft: int,
            fs: int,
            config: BarkBandsplitSpecsConfig, # Changed to use config
            drop_dc_band: bool = False,
            require_no_overlap: bool = False # Add this parameter
    ) -> None:
        super().__init__(nfft=nfft, fs=fs, config=config, drop_dc_band=drop_dc_band, require_no_overlap=require_no_overlap)

    def _get_fbank_fn(self) -> Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]:
        return bark_filterbank


def triangular_bark_filterbank(all_freqs: torch.Tensor, hz_points: torch.Tensor, n_bands: int): # Modified signature

    # calculate mel freq bins
    m_min = hz_to_bark(all_freqs.min()).item()
    m_max = hz_to_bark(all_freqs.max()).item()

    m_pts = torch.linspace(m_min, m_max, n_bands + 2)
    f_pts = 600 * torch.sinh(m_pts / 6)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    fb = fb.T


    first_active_band = torch.nonzero(torch.sum(fb, dim=-1))[0, 0]
    first_active_bin = torch.nonzero(fb[first_active_band, :])[0, 0]

    fb[first_active_band, :first_active_bin] = 1.0

    return fb


class TriangularBarkBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(
            self,
            nfft: int,
            fs: int,
            config: TriangularBarkBandsplitSpecsConfig,
            drop_dc_band: bool = False,
            require_no_overlap: bool = False # Add this parameter
    ) -> None:
        super().__init__(nfft=nfft, fs=fs, config=config, drop_dc_band=drop_dc_band, require_no_overlap=require_no_overlap)

    def _get_fbank_fn(self) -> Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]:
        return triangular_bark_filterbank


def minibark_filterbank(all_freqs: torch.Tensor, hz_points: torch.Tensor, n_bands: int): # Modified signature
    fb = bark_filterbank(
            all_freqs,
            hz_points,
            n_bands
    )

    fb[fb < torch.sqrt(torch.tensor(0.5))] = 0.0 # Convert to torch.sqrt and torch.tensor

    return fb


class MiniBarkBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(
            self,
            nfft: int,
            fs: int,
            config: MiniBarkBandsplitSpecsConfig, # Changed to use config
            drop_dc_band: bool = False,
            require_no_overlap: bool = False # Add this parameter
    ) -> None:
        super().__init__(nfft=nfft, fs=fs, config=config, drop_dc_band=drop_dc_band, require_no_overlap=require_no_overlap)

    def _get_fbank_fn(self) -> Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]:
        return minibark_filterbank


def erb_filterbank(all_freqs: torch.Tensor, hz_points: torch.Tensor, n_bands: int) -> torch.Tensor: # Modified signature
    # freq bins
    A = (1000 * torch.log(torch.tensor(10))) / (24.7 * 4.37) # Convert to torch.log and torch.tensor

    # calculate mel freq bins
    m_min = hz_to_erb(all_freqs.min()).item()
    m_max = hz_to_erb(all_freqs.max()).item()

    m_pts = torch.linspace(m_min, m_max, n_bands + 2)
    f_pts = (torch.pow(torch.tensor(10), (m_pts / A)) - 1)/ 0.00437 # Convert to torch.pow and torch.tensor

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    fb = fb.T


    first_active_band = torch.nonzero(torch.sum(fb, dim=-1))[0, 0]
    first_active_bin = torch.nonzero(fb[first_active_band, :])[0, 0]

    fb[first_active_band, :first_active_bin] = 1.0
    
    # Ensure that the sum of the filterbank weights for each frequency bin is 1
    # This ensures that the sum of masks for each frequency bin is 1, preventing
    # amplitude changes due to the bandsplit.
    # Add a small epsilon to avoid division by zero for bins with no weight
    epsilon = 1e-10
    fb = fb / (fb.sum(axis=0, keepdims=True) + epsilon)

    return fb


class EquivalentRectangularBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(
            self,
            nfft: int,
            fs: int,
            config: EquivalentRectangularBandsplitSpecsConfig, # Changed to use config
            drop_dc_band: bool = False,
            require_no_overlap: bool = False # Add this parameter
    ) -> None:
        super().__init__(nfft=nfft, fs=fs, config=config, drop_dc_band=drop_dc_band, require_no_overlap=require_no_overlap)

    def _get_fbank_fn(self) -> Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]:
        return erb_filterbank


# Bandsplit Factory Function
def get_bandsplit_specs_factory(
    bandsplit_type: BandsplitType, # Use Enum type hint
    n_fft: int,
    fs: int,
    config: BandsplitConfig, # Now takes a Pydantic config
    drop_dc_band: bool = False, # New parameter
    require_no_overlap: bool = False # Add this parameter
) -> BandsplitSpecification: # Return type is now BandsplitSpecification
    """
    Factory function to create and return band specifications based on type.
    """
    bandsplit_map = {
        BandsplitType.VOCAL: VocalBandsplitSpecification,
        BandsplitType.VOCAL_V7: VocalBandsplitSpecification, # Added VOCAL_V7
        BandsplitType.BASS: BassBandsplitSpecification,
        BandsplitType.DRUM: DrumBandsplitSpecification,
        BandsplitType.OTHER: OtherBandsplitSpecification,
        BandsplitType.MUSICAL: MusicalBandsplitSpecification,
        BandsplitType.MUSIC: MusicalBandsplitSpecification,
        BandsplitType.MEL: MelBandsplitSpecification,
        BandsplitType.BARK: BarkBandsplitSpecification,
        BandsplitType.TRIBARK: TriangularBarkBandsplitSpecification,
        BandsplitType.ERB: EquivalentRectangularBandsplitSpecification,
        BandsplitType.MINIBARK: MiniBarkBandsplitSpecification,
    }
    bandsplit_type = BandsplitType(bandsplit_type)

    bandsplit_class = bandsplit_map.get(bandsplit_type)
    if bandsplit_class is None:
        raise ValueError(f"Unknown bandsplit_type: {bandsplit_type}")

    # The config object is now directly passed and is already a Pydantic model
    # The bandsplit_class constructor will handle the specific config type
    specs = bandsplit_class(nfft=n_fft, fs=fs, config=config, drop_dc_band=drop_dc_band, require_no_overlap=require_no_overlap)

    return specs