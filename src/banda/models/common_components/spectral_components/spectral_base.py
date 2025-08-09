import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta
import torchaudio.functional as taF

from omegaconf import DictConfig, OmegaConf
import structlog
import logging

logger = structlog.get_logger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)

from banda.models.common_components.configs.bandsplit_configs._bandsplit_models import (
    BandsplitConfig,
    BandsplitType,
    FixedBandsplitSpecsConfig,
    VocalBandsplitSpecsConfig,
    PerceptualBandsplitSpecsConfig,
    MusicalBandsplitSpecsConfig,
    MelBandsplitSpecsConfig,
    TriangularBarkBandsplitSpecsConfig,
    EquivalentRectangularBandsplitSpecsConfig,
)

from banda.models.common_components.spectral_components.bandsplit_registry import bandsplit_registry


# Add these helper functions if not already defined
def hz_to_bark(hz: torch.Tensor) -> torch.Tensor:
    """
    Converts frequency (Hz) to Bark scale.
    Using the formula: 7 * asinh(hz / 650)
    """
    logger.debug(f"hz_to_bark: Input hz: {hz}")
    bark = 7 * torch.asinh(hz / 650.0)
    logger.debug(f"hz_to_bark: Output bark: {bark}")
    return bark

def bark_to_hz(bark: torch.Tensor) -> torch.Tensor:
    """
    Converts Bark scale to frequency (Hz).
    Using the formula: 650 * exp(bark / 7)
    """
    logger.debug(f"bark_to_hz: Input bark: {bark}")
    hz = 650 * torch.sinh(bark / 7.0)
    logger.debug(f"bark_to_hz: Output hz: {hz}")
    return hz

def hz_to_erb(hz: torch.Tensor) -> torch.Tensor:
    """
    Converts frequency (Hz) to ERB scale.
    Using the formula: 21.4 * log10(1 + 0.00437 * hz)
    """
    logger.debug(f"hz_to_erb: Input hz: {hz}")
    erb = 21.4 * torch.log10(1 + 0.00437 * hz)
    logger.debug(f"hz_to_erb: Output erb: {erb}")
    return erb

def erb_to_hz(erb: torch.Tensor) -> torch.Tensor:
    """
    Converts ERB scale to frequency (Hz).
    Using the formula: 650 * (exp(erb / 21.4) - 1) / 0.00437
    """
    logger.debug(f"erb_to_hz: Input erb: {erb}")
    hz = 650 * (torch.exp(erb / 21.4) - 1) / 0.00437
    logger.debug(f"erb_to_hz: Output hz: {hz}")
    return hz

def hz_to_midi(frequencies: torch.Tensor) -> torch.Tensor:
    """
    Converts frequencies (Hz) to MIDI note numbers.
    Formula: 69 + 12 * log2(frequencies / 440)
    Handles zero or negative frequencies by clamping to a small positive value.
    """
    logger.debug(f"hz_to_midi: Input frequencies: {frequencies}")
    min_freq = 1e-6
    clamped_frequencies = torch.clamp(frequencies, min=min_freq)
    midi = 69 + 12 * torch.log2(clamped_frequencies / 440.0)
    logger.debug(f"hz_to_midi: Output midi: {midi}")
    return midi

def midi_to_hz(midi_notes: torch.Tensor) -> torch.Tensor:
    """
    Converts MIDI note numbers to frequencies (Hz).
    Formula: 440 * 2^((midi_notes - 69) / 12)
    """
    logger.debug(f"midi_to_hz: Input midi_notes: {midi_notes}")
    hz = 440.0 * torch.pow(2, (midi_notes - 69) / 12.0)
    logger.debug(f"midi_to_hz: Output hz: {hz}")
    return hz

def mel_scale_to_hz(mel_points: torch.Tensor) -> torch.Tensor:
    """
    Converts Mel scale points to Hz.
    """
    logger.debug(f"mel_scale_to_hz: Input mel_points: {mel_points}")
    hz = 700 * (torch.exp(mel_points / 1127) - 1)
    logger.debug(f"mel_scale_to_hz: Output hz: {hz}")
    return hz

def hz_to_mel_scale(hz: torch.Tensor) -> torch.Tensor:
    """
    Converts Hz to Mel scale points.
    """
    logger.debug(f"hz_to_mel_scale: Input hz: {hz}")
    mel = 1127 * torch.log(1 + hz / 700)
    logger.debug(f"hz_to_mel_scale: Output mel: {mel}")
    return mel

def _create_triangular_filterbank(all_freqs: torch.Tensor, f_pts: torch.Tensor) -> torch.Tensor:
    """
    Creates a triangular filterbank.
    Args:
        all_freqs (torch.Tensor): All frequency bins.
        f_pts (torch.Tensor): Filterbank center frequencies.
    Returns:
        torch.Tensor: Triangular filterbank.
    """
    logger.debug(f"_create_triangular_filterbank: all_freqs min: {all_freqs.min()}, max: {all_freqs.max()}, shape: {all_freqs.shape}")
    logger.debug(f"_create_triangular_filterbank: f_pts: {f_pts}")

    n_mels = f_pts.shape[0] - 2
    n_freqs = all_freqs.shape[0]
    filterbank = torch.zeros((n_mels, n_freqs))
    epsilon = 1e-9 # Small epsilon for floating point comparisons and division

    for i in range(n_mels):
        left_f = f_pts[i]
        center_f = f_pts[i + 1]
        right_f = f_pts[i + 2]
        logger.debug(f"_create_triangular_filterbank: Band {i}: left_f={left_f}, center_f={center_f}, right_f={right_f}")

        for j in range(n_freqs):
            freq_val = all_freqs[j]
            val = 0.0
            
            if left_f <= freq_val <= center_f:
                if torch.isclose(center_f, left_f, atol=epsilon): # Handle zero width segment
                    val = 1.0 if torch.isclose(freq_val, center_f, atol=epsilon) else 0.0
                else:
                    val = (freq_val - left_f) / (center_f - left_f)
            elif center_f < freq_val <= right_f:
                if torch.isclose(right_f, center_f, atol=epsilon): # Handle zero width segment
                    val = 1.0 if torch.isclose(freq_val, center_f, atol=epsilon) else 0.0
                else:
                    val = (right_f - freq_val) / (right_f - center_f)
            
            filterbank[i, j] = val
            if val > 0: # Log only when a value is assigned
                logger.debug(f"_create_triangular_filterbank: Band {i}, Bin {j}: freq_val={freq_val}, assigned {val}")

    # Ensure the first band covers from the lowest frequency bin
    # Find the first frequency bin that is covered by any band
    first_covered_bin = -1
    for i in range(n_freqs):
        if torch.sum(filterbank[:, i]) > 0:
            first_covered_bin = i
            break
    
    if first_covered_bin > 0:
        # If the first band (index 0) exists, extend it to cover from bin 0
        if n_mels > 0:
            filterbank[0, :first_covered_bin] = 1.0
            logger.debug(f"_create_triangular_filterbank: Extended first band (0) to cover bins 0 to {first_covered_bin-1}")

    # Ensure the last band covers up to the highest frequency bin
    # Find the last frequency bin that is covered by any band
    last_covered_bin = -1
    for i in range(n_freqs - 1, -1, -1):
        if torch.sum(filterbank[:, i]) > 0:
            last_covered_bin = i
            break
    
    if last_covered_bin < n_freqs - 1:
        # If the last band (index n_mels-1) exists, extend it to cover up to the last bin
        if n_mels > 0:
            filterbank[n_mels - 1, last_covered_bin + 1:] = 1.0
            logger.debug(f"_create_triangular_filterbank: Extended last band ({n_mels-1}) to cover bins {last_covered_bin+1} to {n_freqs-1}")


    logger.debug(f"_create_triangular_filterbank: Final filterbank shape: {filterbank.shape}, sum: {filterbank.sum()}")
    return filterbank


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


# Bandsplit Factory Function
def get_bandsplit_specs_factory(
    bandsplit_type: BandsplitType,
    n_fft: int,
    fs: int,
    config: BandsplitConfig,
    drop_dc_band: bool = False,
) -> BandsplitSpecification:
    """
    Factory function to create and return band specifications based on type.
    """
    bandsplit_type = BandsplitType(bandsplit_type)

    bandsplit_class = bandsplit_registry.get(bandsplit_type)
    if bandsplit_class is None:
        raise ValueError(f"Unknown bandsplit_type: {bandsplit_type}")

    # The config object is now directly passed and is already a Pydantic model
    # The bandsplit_class constructor will handle the specific config type
    specs = bandsplit_class(nfft=n_fft, fs=fs, config=config, drop_dc_band=drop_dc_band)

    return specs