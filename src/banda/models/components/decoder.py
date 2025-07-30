#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

import torch
import torch.nn as nn
from typing import Dict

from banda.utils.registry import MODELS_REGISTRY
from banda.core.interfaces import BaseDecoder


@MODELS_REGISTRY.register("decoder")
class Decoder(BaseDecoder):
    """
    A generic decoder module for source separation.

    This module takes encoded features and applies masks to reconstruct separated audio
    using a 1D convolutional transpose layer followed by a ReLU activation.
    """

    def __init__(self, input_features: int, output_channels: int) -> None:
        """
        Initializes the Decoder module.

        Args:
            input_features (int): Number of input features/channels from the encoder.
            output_channels (int): Number of output audio channels (e.g., 2 for stereo).
        """
        super().__init__()
        self.conv_transpose: nn.ConvTranspose1d = nn.ConvTranspose1d(input_features, output_channels, kernel_size=16, stride=8)
        self.relu: nn.ReLU = nn.ReLU()

    def forward(self, encoded_features: torch.Tensor, masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the decoder.

        Args:
            encoded_features (torch.Tensor): Encoded feature tensor from the encoder.
                Shape: (batch_size, input_features, encoded_samples)
            masks (Dict[str, torch.Tensor]): Dictionary of masks for each source.
                Each mask tensor has shape: (batch_size, input_features, encoded_samples)

        Returns:
            Dict[str, torch.Tensor]: Dictionary of separated source audio tensors.
                Each tensor has shape: (batch_size, output_channels, samples)
        """
        separated_sources: Dict[str, torch.Tensor] = {}
        for source_name, mask in masks.items():
            # Apply mask to encoded features
            masked_features: torch.Tensor = encoded_features * mask # Shape: (batch_size, input_features, encoded_samples)
            # Decode masked features
            separated_audio: torch.Tensor = self.relu(self.conv_transpose(masked_features)) # Shape: (batch_size, output_channels, samples)
            separated_sources[source_name] = separated_audio
        return separated_sources