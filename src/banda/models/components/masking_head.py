#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

import torch
import torch.nn as nn
from typing import Dict, List

from banda.utils.registry import MODELS_REGISTRY
from banda.core.interfaces import BaseMaskingHead


@MODELS_REGISTRY.register("masking_head")
class MaskingHead(BaseMaskingHead):
    """
    A generic masking head module for source separation.

    This module takes encoded features (e.g., from an encoder or time-frequency model)
    and predicts a mask for each individual source. The masks are typically applied
    to the mixture spectrogram to estimate the separated source spectrograms.
    """

    def __init__(self, input_features: int, num_sources: int, source_names: List[str]) -> None:
        """
        Initializes the MaskingHead module.

        Args:
            input_features (int): Number of input features/channels from the preceding module
                                  (e.g., encoder output features or embedding dimension).
            num_sources (int): The total number of sources to separate.
            source_names (List[str]): A list of strings, where each string is the name
                                      of a source (e.g., ["vocals", "bass", "drums", "other"]).
        """
        super().__init__()
        self.num_sources: int = num_sources
        self.source_names: List[str] = source_names
        self.mask_prediction_layers: nn.ModuleDict = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv1d(input_features, input_features, kernel_size=1),
                nn.Sigmoid() # Sigmoid for masks between 0 and 1
            ) for name in source_names
        })

    def forward(self, encoded_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the masking head.

        Args:
            encoded_features (torch.Tensor): Encoded feature tensor from the preceding module.
                Shape: (batch_size, input_features, sequence_length)

        Returns:
            Dict[str, torch.Tensor]: A dictionary where keys are source names and values
                                     are the predicted masks for each source.
                Each mask tensor has shape: (batch_size, input_features, sequence_length)
        """
        masks: Dict[str, torch.Tensor] = {}
        for name, layer in self.mask_prediction_layers.items():
            masks[name] = layer(encoded_features) # Shape: (batch_size, input_features, sequence_length)
        return masks