#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from banda.utils.registry import MODELS_REGISTRY
from banda.core.interfaces import BaseTimeFrequencyModel


@MODELS_REGISTRY.register("seq_band_modelling")
class SeqBandModellingModule(BaseTimeFrequencyModel):
    """
    Sequential Band Modelling Module using RNNs.

    This module processes features across frequency bands and time using a series of
    recurrent neural networks (RNNs) with residual connections.
    """
    def __init__(
            self,
            n_modules: int,
            emb_dim: int,
            rnn_dim: int,
            bidirectional: bool = True,
            rnn_type: str = "LSTM",
    ) -> None:
        """
        Initializes the SeqBandModellingModule.

        Args:
            n_modules (int): The number of RNN modules to stack.
            emb_dim (int): The input and output embedding dimension for each RNN.
            rnn_dim (int): The hidden dimension of the RNNs.
            bidirectional (bool): If True, the RNNs will be bidirectional.
            rnn_type (str): The type of RNN to use ("LSTM" or "GRU").
        """
        super().__init__()
        self.n_modules: int = n_modules
        self.emb_dim: int = emb_dim
        self.rnn_dim: int = rnn_dim
        self.bidirectional: bool = bidirectional
        self.rnn_type: str = rnn_type

        rnn_cls: Type[nn.Module] = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnns: nn.ModuleList = nn.ModuleList(
            [
                rnn_cls(
                    input_size=emb_dim,
                    hidden_size=rnn_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
                for _ in range(n_modules)
            ]
        )
        self.fcs: nn.ModuleList = nn.ModuleList(
            [
                nn.Linear(
                    in_features=rnn_dim * 2 if bidirectional else rnn_dim,
                    out_features=emb_dim,
                )
                for _ in range(n_modules)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SeqBandModellingModule.

        Args:
            x (torch.Tensor): Input tensor from BandSplitModule.
                Shape: (batch, n_bands, n_time, emb_dim)

        Returns:
            torch.Tensor: Processed tensor.
                Shape: (batch, n_bands, n_time, emb_dim)
        """
        batch: int
        n_bands: int
        n_time: int
        emb_dim: int
        batch, n_bands, n_time, emb_dim = x.shape
        
        # Reshape for RNN: (batch * n_bands, n_time, emb_dim)
        x_reshaped: torch.Tensor = x.view(batch * n_bands, n_time, emb_dim)

        for i in range(self.n_modules):
            rnn_out, _ = self.rnns[i](x_reshaped) # Shape: (batch * n_bands, n_time, rnn_dim * (1 or 2))
            x_reshaped = x_reshaped + self.fcs[i](rnn_out) # Residual connection

        # Reshape back: (batch, n_bands, n_time, emb_dim)
        x_out: torch.Tensor = x_reshaped.view(batch, n_bands, n_time, emb_dim)
        return x_out


@MODELS_REGISTRY.register("transformer_time_freq")
class TransformerTimeFreqModule(BaseTimeFrequencyModel):
    """
    Transformer-based Time-Frequency Modelling Module.

    This module applies a Transformer Encoder to process features across time
    for each frequency band.
    """
    def __init__(
            self,
            n_modules: int,
            emb_dim: int,
            rnn_dim: int, # This parameter is not used in Transformer, but kept for compatibility
            bidirectional: bool = True, # This parameter is not used in Transformer, but kept for compatibility
            dropout: float = 0.0,
    ) -> None:
        """
        Initializes the TransformerTimeFreqModule.

        Args:
            n_modules (int): The number of Transformer Encoder layers to stack.
            emb_dim (int): The input and output embedding dimension for the Transformer.
            rnn_dim (int): Placeholder for compatibility, not used in Transformer.
            bidirectional (bool): Placeholder for compatibility, not used in Transformer.
            dropout (float): Dropout rate for the Transformer layers.
        """
        super().__init__()
        self.n_modules: int = n_modules
        self.emb_dim: int = emb_dim
        self.dropout_layer: nn.Dropout = nn.Dropout(dropout)

        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=8, # Example number of heads
            dim_feedforward=emb_dim * 4, # Example feedforward dimension
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder: nn.TransformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=n_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TransformerTimeFreqModule.

        Args:
            x (torch.Tensor): Input tensor from BandSplitModule.
                Shape: (batch, n_bands, n_time, emb_dim)

        Returns:
            torch.Tensor: Processed tensor.
                Shape: (batch, n_bands, n_time, emb_dim)
        """
        batch: int
        n_bands: int
        n_time: int
        emb_dim: int
        batch, n_bands, n_time, emb_dim = x.shape
        
        # Reshape for Transformer: (batch * n_bands, n_time, emb_dim)
        x_reshaped: torch.Tensor = x.view(batch * n_bands, n_time, emb_dim)

        x_transformed: torch.Tensor = self.transformer_encoder(x_reshaped)
        x_transformed = self.dropout_layer(x_transformed)

        # Reshape back: (batch, n_bands, n_time, emb_dim)
        x_out: torch.Tensor = x_transformed.view(batch, n_bands, n_time, emb_dim)
        return x_out


@MODELS_REGISTRY.register("conv_time_freq")
class ConvolutionalTimeFreqModule(BaseTimeFrequencyModel):
    """
    Convolutional Time-Frequency Modelling Module.

    This module applies 2D convolutional blocks with residual connections
    to process features across frequency bands and time.
    """
    def __init__(
            self,
            n_modules: int,
            emb_dim: int,
            rnn_dim: int, # Not used, but kept for compatibility
            bidirectional: bool = True, # Not used, but kept for compatibility
            dropout: float = 0.0,
    ) -> None:
        """
        Initializes the ConvolutionalTimeFreqModule.

        Args:
            n_modules (int): The number of convolutional blocks to stack.
            emb_dim (int): The input and output embedding dimension for the convolutional layers.
            rnn_dim (int): Placeholder for compatibility, not used in Convolutional module.
            bidirectional (bool): Placeholder for compatibility, not used in Convolutional module.
            dropout (float): Dropout rate after convolutional blocks.
        """
        super().__init__()
        self.n_modules: int = n_modules
        self.emb_dim: int = emb_dim
        self.dropout_layer: nn.Dropout = nn.Dropout(dropout)

        self.conv_blocks: nn.ModuleList = nn.ModuleList()
        for _ in range(n_modules):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim, kernel_size=(3, 3), padding=(1, 1)),
                    nn.BatchNorm2d(emb_dim),
                    nn.ReLU(),
                    nn.Conv2d(emb_dim, emb_dim, kernel_size=(3, 3), padding=(1, 1)),
                    nn.BatchNorm2d(emb_dim),
                    nn.ReLU(),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvolutionalTimeFreqModule.

        Args:
            x (torch.Tensor): Input tensor from BandSplitModule.
                Shape: (batch, n_bands, n_time, emb_dim)

        Returns:
            torch.Tensor: Processed tensor.
                Shape: (batch, n_bands, n_time, emb_dim)
        """
        # Permute to (batch, emb_dim, n_bands, n_time) for Conv2d
        x_permuted: torch.Tensor = x.permute(0, 3, 1, 2) # Shape: (batch, emb_dim, n_bands, n_time)

        for conv_block in self.conv_blocks:
            x_permuted = x_permuted + conv_block(x_permuted) # Residual connection

        x_permuted = self.dropout_layer(x_permuted)

        # Permute back to (batch, n_bands, n_time, emb_dim)
        x_out: torch.Tensor = x_permuted.permute(0, 2, 3, 1)
        return x_out