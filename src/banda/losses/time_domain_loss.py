import torch
import torch.nn as nn
from typing import Optional, Dict
from omegaconf import DictConfig

from banda.losses.l1_snr_utils import calculate_l1_snr


class TimeDomainLoss(nn.Module):
    """
    Computes a time-domain loss (L1 or L2) between predicted and target audio waveforms.
    """
    def __init__(
        self,
        # Removed p, reduction, eps as they are handled by calculate_l1_snr
    ) -> None:
        """
        Args:
            # Removed p, reduction, eps as they are handled by calculate_l1_snr
        """
        super().__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the time-domain loss.

        Args:
            prediction (torch.Tensor): Predicted audio waveform. Shape: (batch, channels, samples)
            target (torch.Tensor): Ground truth audio waveform. Shape: (batch, channels, samples)

        Returns:
            torch.Tensor: The calculated time-domain loss.
        """
        # Calculate L1 SNR using the utility function
        loss = calculate_l1_snr(
            prediction,
            target,
            scale_invariant=False, # Time-domain loss is typically not scale-invariant
            take_log=True,
            eps=1e-8,
        )

        return loss

    @classmethod
    def from_config(cls, config: DictConfig):
        """
        Instantiates a TimeDomainLoss from a DictConfig.
        """
        # Removed p, reduction, eps from config
        return cls()