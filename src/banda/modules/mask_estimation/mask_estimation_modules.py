#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from pydantic import BaseModel, Field, ConfigDict

# from banda.utils.registry import MODELS_REGISTRY # Removed as per previous instructions

class MaskEstimationConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    target_: str = Field(alias="_target_") # Changed _target_ to target_
    emb_dim: int
    output_channels: int = 2 # For magnitude and phase masks

class MaskEstimationModule(nn.Module):
    """
    Estimates a mask from the processed time-frequency features.
    """
    def __init__(self, emb_dim: int, output_channels: int = 2):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_channels = output_channels
        self.linear = nn.Linear(emb_dim, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor from the time-frequency model.
                              Shape: (batch, n_bands, n_time, emb_dim)
        Returns:
            torch.Tensor: Estimated mask. Shape: (batch, output_channels, n_bands, n_time)
        """
        # Apply linear layer to the last dimension (emb_dim)
        # Output shape: (batch, n_bands, n_time, output_channels)
        mask = self.linear(x)
        
        # Permute to (batch, output_channels, n_bands, n_time)
        mask = mask.permute(0, 3, 1, 2)
        return mask

    @classmethod
    def from_config(cls, config: MaskEstimationConfig) -> "MaskEstimationModule":
        """
        Instantiates MaskEstimationModule from a MaskEstimationConfig Pydantic model.
        """
        return cls(
            emb_dim=config.emb_dim,
            output_channels=config.output_channels,
        )

# No model_rebuild() calls needed here