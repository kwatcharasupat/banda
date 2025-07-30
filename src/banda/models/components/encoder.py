#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

import torch
import torch.nn as nn

from banda.utils.registry import MODELS_REGISTRY
from banda.core.interfaces import BaseEncoder


@MODELS_REGISTRY.register("encoder")
class Encoder(BaseEncoder):
    """
    A generic encoder module for source separation.

    This module takes raw audio and transforms it into a higher-level representation
    suitable for separation using a 1D convolutional layer followed by a ReLU activation.
    """

    def __init__(self, input_channels: int, output_features: int) -> None:
        """
        Initializes the Encoder module.

        Args:
            input_channels (int): Number of input audio channels (e.g., 2 for stereo).
            output_features (int): Number of output features/channels after encoding.
        """
        super().__init__()
        self.conv: nn.Conv1d = nn.Conv1d(input_channels, output_features, kernel_size=16, stride=8)
        self.relu: nn.ReLU = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input audio tensor.
                Shape: (batch_size, channels, samples)

        Returns:
            torch.Tensor: Encoded feature tensor.
                Shape: (batch_size, output_features, encoded_samples)
        """
        return self.relu(self.conv(x))