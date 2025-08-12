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

class BandsplitModuleConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    target_: str = Field(alias="_target_") # Changed _target_ to target_
    n_bands: int
    emb_dim: int
    input_channels: int = 2 # For magnitude and phase

class BandsplitModule(nn.Module):
    """
    Splits the input spectrogram into multiple frequency bands and applies a linear
    projection to each band.
    """
    def __init__(self, n_bands: int, emb_dim: int, input_channels: int = 2):
        super().__init__()
        self.n_bands = n_bands
        self.emb_dim = emb_dim
        self.input_channels = input_channels

        self.linear_projections = nn.ModuleList(
            [nn.Linear(input_channels, emb_dim) for _ in range(n_bands)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input spectrogram. Shape: (batch, channels, freq, time)
        Returns:
            torch.Tensor: Output tensor with shape (batch, n_bands, time, emb_dim)
        """
        batch_size, channels, n_freq, n_time = x.shape
        
        # Ensure n_freq is divisible by n_bands
        if n_freq % self.n_bands != 0:
            raise ValueError(f"Number of frequency bins ({n_freq}) must be divisible by n_bands ({self.n_bands})")
        
        band_size = n_freq // self.n_bands
        
        outputs = []
        for i in range(self.n_bands):
            # Extract band
            band = x[:, :, i * band_size : (i + 1) * band_size, :]
            
            # Reshape band for linear projection: (batch, band_size, time, channels)
            # Then flatten band_size and channels for linear layer: (batch, time, band_size * channels)
            # Or, apply linear layer to (batch, time, channels) for each freq bin, then average/pool
            # Let's assume the linear projection is applied per-frequency-bin, then averaged.
            # A simpler approach is to apply linear projection to the channels dimension directly.
            
            # Current approach: linear projection on the channel dimension (input_channels -> emb_dim)
            # This means the linear layer operates on (batch, freq_bin, time, channels)
            # We need to reshape to (batch * freq_bin * time, channels) for linear layer
            # Or, apply it to (batch, channels, freq, time) directly if it's a 1x1 conv
            
            # Let's assume the linear projection is applied to the last dimension (channels)
            # Reshape band to (batch * band_size * n_time, channels)
            band_reshaped = band.permute(0, 2, 3, 1).contiguous().view(-1, channels)
            
            projected_band = self.linear_projections[i](band_reshaped)
            
            # Reshape back to (batch, band_size, n_time, emb_dim)
            projected_band = projected_band.view(batch_size, band_size, n_time, self.emb_dim)
            
            # Average or pool across the frequency dimension within the band
            # For simplicity, let's average across the frequency bins within the band
            outputs.append(projected_band.mean(dim=1, keepdim=True)) # Shape: (batch, 1, n_time, emb_dim)
        
        # Concatenate outputs along the band dimension
        output_tensor = torch.cat(outputs, dim=1) # Shape: (batch, n_bands, n_time, emb_dim)
        return output_tensor

    @classmethod
    def from_config(cls, config: BandsplitModuleConfig) -> "BandsplitModule":
        """
        Instantiates BandsplitModule from a BandsplitModuleConfig Pydantic model.
        """
        return cls(
            n_bands=config.n_bands,
            emb_dim=config.emb_dim,
            input_channels=config.input_channels,
        )

# No model_rebuild() calls needed here
