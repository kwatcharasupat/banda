#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

import torch
import torch.nn as nn
from typing import Dict, List


class MaskingHead(nn.Module):
    """
    A generic masking head module for source separation.

    This module takes encoded features and predicts masks for each source.
    """

    def __init__(self, input_features: int, num_sources: int, source_names: List[str]) -> None:
        """
        Args:
            input_features (int): Number of input features/channels from the encoder.
            num_sources (int): Number of sources to separate.
            source_names (List[str]): List of names for each source (e.g., ["vocals", "bass", ...]).
        """
        super().__init__()
        self.num_sources = num_sources
        self.source_names = source_names
        self.mask_prediction_layers = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv1d(input_features, input_features, kernel_size=1),
                nn.Sigmoid() # Sigmoid for masks between 0 and 1
            ) for name in source_names
        })

    def forward(self, encoded_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the masking head.

        Args:
            encoded_features (torch.Tensor): Encoded feature tensor from the encoder.
                                             Shape: (batch_size, input_features, encoded_samples)

        Returns:
            Dict[str, torch.Tensor]: Dictionary of predicted masks for each source.
                                     Each mask tensor has shape: (batch_size, input_features, encoded_samples)
        """
        masks = {}
        for name, layer in self.mask_prediction_layers.items():
            masks[name] = layer(encoded_features)
        return masks