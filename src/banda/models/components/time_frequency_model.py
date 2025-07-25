#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


class SeqBandModellingModule(nn.Module):
    """
    Sequential Band Modelling Module using RNNs.
    Processes features across frequency bands and time.
    """
    def __init__(
            self,
            n_modules: int,
            emb_dim: int,
            rnn_dim: int,
            bidirectional: bool = True,
            rnn_type: str = "LSTM",
    ) -> None:
        super().__init__()
        self.n_modules = n_modules
        self.emb_dim = emb_dim
        self.rnn_dim = rnn_dim
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnns = nn.ModuleList(
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
        self.fcs = nn.ModuleList(
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
            torch.Tensor: Processed tensor. Shape: (batch, n_bands, n_time, emb_dim)
        """
        batch, n_bands, n_time, emb_dim = x.shape
        
        # Reshape for RNN: (batch * n_bands, n_time, emb_dim)
        x = x.view(batch * n_bands, n_time, emb_dim)

        for i in range(self.n_modules):
            rnn_out, _ = self.rnns[i](x)
            x = x + self.fcs[i](rnn_out) # Residual connection

        # Reshape back: (batch, n_bands, n_time, emb_dim)
        x = x.view(batch, n_bands, n_time, emb_dim)
        return x


class TransformerTimeFreqModule(nn.Module):
    """
    Transformer-based Time-Frequency Modelling Module.
    """
    def __init__(
            self,
            n_modules: int,
            emb_dim: int,
            rnn_dim: int, # This parameter is not used in Transformer, but kept for compatibility
            bidirectional: bool = True, # This parameter is not used in Transformer, but kept for compatibility
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_modules = n_modules
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=8, # Example number of heads
            dim_feedforward=emb_dim * 4, # Example feedforward dimension
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TransformerTimeFreqModule.

        Args:
            x (torch.Tensor): Input tensor from BandSplitModule.
                              Shape: (batch, n_bands, n_time, emb_dim)

        Returns:
            torch.Tensor: Processed tensor. Shape: (batch, n_bands, n_time, emb_dim)
        """
        batch, n_bands, n_time, emb_dim = x.shape
        
        # Reshape for Transformer: (batch * n_bands, n_time, emb_dim)
        x = x.view(batch * n_bands, n_time, emb_dim)

        x = self.transformer_encoder(x)
        x = self.dropout(x)

        # Reshape back: (batch, n_bands, n_time, emb_dim)
        x = x.view(batch, n_bands, n_time, emb_dim)
        return x


class ConvolutionalTimeFreqModule(nn.Module):
    """
    Convolutional Time-Frequency Modelling Module.
    """
    def __init__(
            self,
            n_modules: int,
            emb_dim: int,
            rnn_dim: int, # Not used, but kept for compatibility
            bidirectional: bool = True, # Not used, but kept for compatibility
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_modules = n_modules
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout)

        self.conv_blocks = nn.ModuleList()
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
            torch.Tensor: Processed tensor. Shape: (batch, n_bands, n_time, emb_dim)
        """
        # Permute to (batch, emb_dim, n_bands, n_time) for Conv2d
        x = x.permute(0, 3, 1, 2) # (batch, emb_dim, n_bands, n_time)

        for conv_block in self.conv_blocks:
            x = x + conv_block(x) # Residual connection

        x = self.dropout(x)

        # Permute back to (batch, n_bands, n_time, emb_dim)
        x = x.permute(0, 2, 3, 1)
        return x