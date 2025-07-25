#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

import torch
import torch.nn as nn
import torchaudio.transforms as T
from typing import List, Tuple, Dict, Optional


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss, combining L1 loss on magnitude spectrograms
    and optionally on phase spectrograms.
    """
    def __init__(
        self,
        n_ffts: List[int] = [2048, 512],
        hop_lengths: List[int] = [512, 128],
        win_lengths: List[int] = [2048, 512],
        window_fn: str = "hann_window",
        wkwargs: Optional[Dict] = None,
        mag_weight: float = 1.0,
        phase_weight: float = 0.0,
        log_scale: bool = False,
    ) -> None:
        """
        Args:
            n_ffts (List[int]): List of n_fft values for each STFT resolution.
            hop_lengths (List[int]): List of hop_length values for each STFT resolution.
            win_lengths (List[int]): List of win_length values for each STFT resolution.
            window_fn (str): Name of the window function to use (e.g., "hann_window").
            wkwargs (Optional[Dict]): Keyword arguments for the window function.
            mag_weight (float): Weight for the magnitude spectrogram loss.
            phase_weight (float): Weight for the phase spectrogram loss.
            log_scale (bool): If True, apply log scale to magnitude spectrograms before computing loss.
        """
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths), \
            "n_ffts, hop_lengths, and win_lengths must have the same number of elements."

        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths
        self.mag_weight = mag_weight
        self.phase_weight = phase_weight
        self.log_scale = log_scale

        window = torch.__dict__[window_fn]
        self.stfts = nn.ModuleList()
        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
            self.stfts.append(
                T.Spectrogram(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window_fn=window,
                    wkwargs=wkwargs,
                    return_complex=True, # Get complex spectrogram for magnitude and phase
                )
            )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the multi-resolution STFT loss.

        Args:
            prediction (torch.Tensor): Predicted audio waveform. Shape: (batch, channels, samples)
            target (torch.Tensor): Ground truth audio waveform. Shape: (batch, channels, samples)

        Returns:
            torch.Tensor: The calculated multi-resolution STFT loss.
        """
        total_loss = 0.0
        num_resolutions = len(self.stfts)

        for i, stft_layer in enumerate(self.stfts):
            pred_spec = stft_layer(prediction)
            target_spec = stft_layer(target)

            # Magnitude loss
            pred_mag = pred_spec.abs()
            target_mag = target_spec.abs()

            if self.log_scale:
                pred_mag = torch.log1p(pred_mag)
                target_mag = torch.log1p(target_mag)

            mag_loss = F.l1_loss(pred_mag, target_mag) * self.mag_weight
            total_loss += mag_loss

            # Phase loss (optional)
            if self.phase_weight > 0:
                pred_phase = torch.atan2(pred_spec.imag, pred_spec.real)
                target_phase = torch.atan2(target_spec.imag, target_spec.real)
                phase_loss = F.l1_loss(pred_phase, target_phase) * self.phase_weight
                total_loss += phase_loss
        
        # Normalize by the number of resolutions to ensure equal weighting
        return total_loss / num_resolutions
