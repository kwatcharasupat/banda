import torch
import torch.nn as nn
from typing import Optional, Dict
from omegaconf import DictConfig

from banda.utils.registry import LOSSES_REGISTRY
from banda.losses.l1_snr_utils import calculate_l1_snr


from banda.core.interfaces import BaseLoss
@LOSSES_REGISTRY.register("time_domain_loss")

class TimeDomainLoss(BaseLoss):
    """
    Computes a time-domain loss between predicted and target audio waveforms.

    This loss function uses a Signal-to-Noise P-Norm Ratio (L1-SNR) in the time domain.
    """
    def __init__(
        self,
    ) -> None:
        """
        Initializes the TimeDomainLoss.
        """
        super().__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the time-domain loss.

        Args:
            prediction (torch.Tensor): Predicted audio waveform.
                Shape: (batch, channels, samples)
            target (torch.Tensor): Ground truth audio waveform.
                Shape: (batch, channels, samples)

        Returns:
            torch.Tensor: The calculated time-domain loss.
        """
        # Calculate L1 SNR using the utility function
        loss: torch.Tensor = calculate_l1_snr(
            prediction,
            target,
            scale_invariant=False, # Time-domain loss is typically not scale-invariant
            take_log=True,
            eps=1e-8,
        )

        return loss

    @classmethod
    def from_config(cls, config: DictConfig) -> "TimeDomainLoss":
        """
        Instantiates a TimeDomainLoss from a DictConfig.

        Args:
            config (DictConfig): A DictConfig object containing the loss configuration.

        Returns:
            TimeDomainLoss: An instance of the TimeDomainLoss.
        """
        return cls()