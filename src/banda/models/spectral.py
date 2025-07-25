#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

from typing import Dict, Optional

import torch
import torchaudio as ta
from torch import nn


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
            torch.Tensor: Complex spectrogram. Shape: (batch_size, channels, freq_bins, time_frames, 2)
        """
        # torchaudio.Spectrogram expects (..., time)
        # Our input is (batch, channels, samples)
        # We need to reshape to (batch * channels, samples) for STFT, then reshape back
        batch_size, num_channels, num_samples = x.shape
        x = x.view(batch_size * num_channels, num_samples)
        
        spec = self.stft(x) # (batch * channels, freq_bins, time_frames, 2)
        
        # Reshape back to (batch, channels, freq_bins, time_frames, 2)
        spec = spec.view(batch_size, num_channels, spec.shape[-3], spec.shape[-2], spec.shape[-1])
        return spec

    def inverse(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """
        Inverse pass for ISTFT.

        Args:
            spec (torch.Tensor): Complex spectrogram. Shape: (batch_size, channels, freq_bins, time_frames, 2)
            length (int): Original length of the audio waveform.

        Returns:
            torch.Tensor: Reconstructed audio waveform. Shape: (batch_size, channels, samples)
        """
        batch_size, num_channels, freq_bins, time_frames, _ = spec.shape
        
        # Reshape to (batch * channels, freq_bins, time_frames, 2) for ISTFT
        spec = spec.view(batch_size * num_channels, freq_bins, time_frames, 2)
        
        audio = self.istft(spec, length=length) # (batch * channels, samples)
        
        # Reshape back to (batch, channels, samples)
        audio = audio.view(batch_size, num_channels, audio.shape[-1])
        return audio