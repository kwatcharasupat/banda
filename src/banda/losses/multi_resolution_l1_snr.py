#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.


import torch
import torch.nn as nn
from typing import List, Dict, Optional
from omegaconf import DictConfig

from banda.losses.single_resolution_stft import SingleResolutionSTFTLoss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss, combining L1 loss on real and imaginary spectrograms
    across multiple STFT resolutions.
    """
    def __init__(
        self,
        n_ffts: List[int] = [2048, 512],
        hop_lengths: List[int] = [512, 128],
        win_lengths: List[int] = [2048, 512],
        window_fn: str = "hann_window",
        wkwargs: Optional[Dict] = None,
        # Removed p, scale_invariant, take_log, eps as they are handled by calculate_l1_snr
        # Removed reduction as it will be handled by SeparationLossHandler
    ) -> None:
        """
        Args:
            n_ffts (List[int]): List of n_fft values for each STFT resolution.
            hop_lengths (List[int]): List of hop_length values for each STFT resolution.
            win_lengths (List[int]): List of win_length values for each STFT resolution.
            window_fn (str): Name of the window function to use (e.g., "hann_window").
            wkwargs (Optional[Dict]): Keyword arguments for the window function.
        """
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths), \
            "n_ffts, hop_lengths, and win_lengths must have the same number of elements."
 
        self.stft_losses = nn.ModuleList()
        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
            self.stft_losses.append(
                SingleResolutionSTFTLoss(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window_fn=window_fn,
                    wkwargs=wkwargs,
                    # Removed p, scale_invariant, take_log, eps, reduction
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
        for stft_loss_layer in self.stft_losses:
            total_loss += stft_loss_layer(prediction, target)
        
        # Normalize by the number of resolutions to ensure equal weighting
        return total_loss / len(self.stft_losses)

    @classmethod
    def from_config(cls, config: DictConfig):
        """
        Instantiates a MultiResolutionSTFTLoss from a DictConfig.
        """
        n_ffts = config.get("n_ffts", [2048, 512])
        hop_lengths = config.get("hop_lengths", [512, 128])
        win_lengths = config.get("win_lengths", [2048, 512])
        window_fn = config.get("window_fn", "hann_window")
        wkwargs = config.get("wkwargs", None)
        
        return cls(
            n_ffts=n_ffts,
            hop_lengths=hop_lengths,
            win_lengths=win_lengths,
            window_fn=window_fn,
            wkwargs=wkwargs,
        )
