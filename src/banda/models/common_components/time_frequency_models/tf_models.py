#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

from typing import Dict, Optional, Type
import structlog
import logging
import warnings
import math

logger = structlog.get_logger(__name__)
logging.getLogger(__name__).setLevel(logging.INFO) # Changed to INFO

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torch.nn.modules import rnn

from mambapy.mamba import Mamba # Import Mamba
from mambapy.vim import VMamba # Import VMamba for Vision Mamba

from banda.utils.registry import MODELS_REGISTRY
from banda.core.interfaces import BaseTimeFrequencyModel
from banda.models.common_components.positional_embeddings import RotaryPositionalEmbedding2D # Import the new module


class ResidualRNN(nn.Module):
    def __init__(
            self,
            emb_dim: int,
            rnn_dim: int,
            bidirectional: bool = True,
            rnn_type: str = "LSTM",
            use_layer_norm: bool = True,
            dropout: float = 0.0, # Added dropout parameter
    ) -> None:
        super().__init__()

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm = nn.LayerNorm(emb_dim, eps=1e-2)
        else:
            self.norm = nn.LayerNorm(emb_dim, eps=1e-2) # Fallback to LayerNorm

        self.rnn = rnn.__dict__[rnn_type](
                input_size=emb_dim,
                hidden_size=rnn_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
        )

        self.fc = weight_norm(nn.Linear(
                in_features=rnn_dim * (2 if bidirectional else 1),
                out_features=emb_dim
        ))
        self.dropout_layer = nn.Dropout(dropout) # Added dropout layer

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z is expected to be (N, L, H) where N = batch * n_bands, L = n_time, H = emb_dim
        z0 = torch.clone(z)

        if self.use_layer_norm:
            z = self.norm(z)
        else:
            z = torch.permute(z, (0, 2, 1)) # (N, H, L)
            z = self.norm(z) # This will be LayerNorm due to fallback in __init__
            z = torch.permute(z, (0, 2, 1)) # (N, L, H)

        rnn_out, _ = self.rnn(z.contiguous())
        fc_out = self.fc(rnn_out)
        fc_out = self.dropout_layer(fc_out) # Apply dropout

        z = fc_out + z0 # Residual connection

        return z


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
            use_layer_norm: bool = True,
            tf_dropout: float = 0.0, # Added tf_dropout parameter
    ) -> None:
        """
        Initializes the SeqBandModellingModule.
        Args:
            n_modules (int): The number of RNN modules to stack.
            emb_dim (int): The input and output embedding dimension for each RNN.
            rnn_dim (int): The hidden dimension of the RNNs.
            bidirectional (bool): If True, the RNNs will be bidirectional.
            rnn_type (str): The type of RNN to use ("LSTM" or "GRU").
            use_layer_norm (bool): If True, apply LayerNorm within ResidualRNN.
            tf_dropout (float): Dropout rate to apply after each ResidualRNN module.
        """
        super().__init__()
        self.n_modules: int = n_modules
        self.emb_dim: int = emb_dim
        self.rnn_dim: int = rnn_dim
        self.bidirectional: bool = bidirectional
        self.rnn_type: str = rnn_type
        self.use_layer_norm: bool = use_layer_norm
        self.tf_dropout: float = tf_dropout # Store dropout for potential use

        # Replace self.rnns and self.fcs with a single ModuleList of ResidualRNNs
        self.seq_modules: nn.ModuleList = nn.ModuleList(
            [
                ResidualRNN(
                    emb_dim=emb_dim,
                    rnn_dim=rnn_dim,
                    bidirectional=bidirectional,
                    rnn_type=rnn_type,
                    use_layer_norm=use_layer_norm,
                    dropout=tf_dropout, # Pass tf_dropout to ResidualRNN
                )
                for _ in range(n_modules)
            ]
        )
        self.final_dropout_layer = nn.Dropout(tf_dropout) # Final dropout layer

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
        logger.info(f"SeqBandModellingModule: x_reshaped shape: {x_reshaped.shape}, min: {x_reshaped.min().item()}, max: {x_reshaped.max().item()}, mean: {x_reshaped.mean().item()}") # Changed to INFO
        if torch.isnan(x_reshaped).any():
            logger.error("SeqBandModellingModule: NaN detected in x_reshaped before RNN", mean_val=x_reshaped.mean().item())
            raise ValueError("NaN in x_reshaped in SeqBandModellingModule")

        # Iterate through the new seq_modules (ResidualRNNs)
        for i in range(self.n_modules):
            # Pass x_reshaped directly to ResidualRNN
            x_reshaped = self.seq_modules[i](x_reshaped)
            logger.info(f"SeqBandModellingModule: x_reshaped shape after ResidualRNN module {i}: {x_reshaped.shape}, min: {x_reshaped.min().item()}, max: {x_reshaped.max().item()}, mean: {x_reshaped.mean().item()}") # Changed to INFO
            if torch.isnan(x_reshaped).any():
                logger.error(f"SeqBandModellingModule: NaN detected in x_reshaped after ResidualRNN module {i}", mean_val=x_reshaped.mean().item())
                raise ValueError(f"NaN in x_reshaped after ResidualRNN module {i}")
            # No need for separate fc and residual connection here, as ResidualRNN handles it.
        
        x_reshaped = self.final_dropout_layer(x_reshaped) # Apply final dropout

        # Reshape back: (batch, n_bands, n_time, emb_dim)
        x_out: torch.Tensor = x_reshaped.view(batch, n_bands, n_time, emb_dim)
        logger.info(f"SeqBandModellingModule: x_out shape before return: {x_out.shape}, min: {x_out.min().item()}, max: {x_out.max().item()}, mean: {x_out.mean().item()}") # Changed to INFO
        if torch.isnan(x_out).any():
            logger.error("SeqBandModellingModule: NaN detected in x_out before return", mean_val=x_out.mean().item())
            raise ValueError("NaN in x_out in SeqBandModellingModule")
        return x_out


@MODELS_REGISTRY.register("transformer_time_freq")
class TransformerTimeFreqModule(BaseTimeFrequencyModel):
    """
    Transformer-based Time-Frequency Modelling Module (ViT-like).

    This module applies a Transformer Encoder to process features across flattened
    time-frequency patches, incorporating Rotary Positional Embeddings.
    """
    def __init__(
            self,
            n_modules: int,
            emb_dim: int,
            dropout: float = 0.0,
            nhead: int = 8,
            dim_feedforward: int = 2048,
            max_seq_len_bands: int = 256, # Max bands for RoPE
            max_seq_len_time: int = 512,  # Max time for RoPE
    ) -> None:
        """
        Initializes the TransformerTimeFreqModule.
        Args:
            n_modules (int): The number of Transformer Encoder layers to stack.
            emb_dim (int): The input and output embedding dimension for the Transformer.
            dropout (float): Dropout rate for the Transformer layers.
            nhead (int): The number of attention heads.
            dim_feedforward (int): The dimension of the feedforward network model.
            max_seq_len_bands (int): Maximum sequence length for bands for RoPE.
            max_seq_len_time (int): Maximum sequence length for time for RoPE.
        """
        super().__init__()
        self.n_modules: int = n_modules
        self.emb_dim: int = emb_dim
        self.dropout_layer: nn.Dropout = nn.Dropout(dropout)

        # Rotary Positional Embedding
        self.rope = RotaryPositionalEmbedding2D(emb_dim, max_seq_len_bands, max_seq_len_time)

        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            emb_dim,
            nhead=nhead, # Configurable
            dim_feedforward=dim_feedforward, # Configurable
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

        # Reshape for transformer: (batch, n_bands * n_time, emb_dim)
        x_reshaped: torch.Tensor = x.view(batch, n_bands * n_time, emb_dim)

        # Apply Rotary Positional Embedding
        x_encoded = self.rope(x_reshaped, n_bands, n_time) # Pass n_bands and n_time as integers

        # Pass through Transformer
        x_transformed: torch.Tensor = self.transformer_encoder(x_encoded)
        x_transformed = self.dropout_layer(x_transformed)

        # Reshape back to original time-frequency grid
        x_out = x_transformed.view(batch, n_bands, n_time, emb_dim)
        
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
            dropout: float = 0.0,
    ) -> None:
        """
        Initializes the ConvolutionalTimeFreqModule.
        Args:
            n_modules (int): The number of convolutional blocks to stack.
            emb_dim (int): The input and output embedding dimension for the convolutional layers.
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
                    nn.Conv2d(emb_dim, emb_dim, kernel_size=(3, 1), padding=(1, 0)), # Changed kernel_size and padding
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


@MODELS_REGISTRY.register("mamba_time_freq")
class MambaTimeFreqModule(BaseTimeFrequencyModel):
    """
    Mamba-based Time-Frequency Modelling Module.

    This module applies Mamba layers to process features across flattened
    time-frequency patches, incorporating Rotary Positional Embeddings.
    """
    def __init__(
            self,
            n_modules: int,
            emb_dim: int,
            dropout: float = 0.0,
            max_seq_len_bands: int = 256, # Max bands for RoPE
            max_seq_len_time: int = 512,  # Max time for RoPE
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
    ) -> None:
        """
        Initializes the MambaTimeFreqModule.
        Args:
            n_modules (int): The number of Mamba layers to stack.
            emb_dim (int): The input and output embedding dimension for the Mamba layers.
            dropout (float): Dropout rate for the Mamba layers.
            max_seq_len_bands (int): Maximum sequence length for bands for RoPE.
            max_seq_len_time (int): Maximum sequence length for time for RoPE.
            d_state (int): The state dimension for the Mamba layer.
            d_conv (int): The convolutional kernel size for the Mamba layer.
            expand (int): The expansion factor for the Mamba layer.
        """
        super().__init__()
        self.n_modules: int = n_modules
        self.emb_dim: int = emb_dim
        self.dropout_layer: nn.Dropout = nn.Dropout(dropout)

        # Rotary Positional Embedding
        self.rope = RotaryPositionalEmbedding2D(emb_dim, max_seq_len_bands, max_seq_len_time)

        self.mamba_blocks: nn.ModuleList = nn.ModuleList()
        for _ in range(n_modules):
            self.mamba_blocks.append(
                Mamba(
                    emb_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MambaTimeFreqModule.
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

        # Reshape for Mamba: (batch, n_bands * n_time, emb_dim)
        x_reshaped: torch.Tensor = x.view(batch, n_bands * n_time, emb_dim)

        # Apply Rotary Positional Embedding
        x_encoded = self.rope(x_reshaped, n_bands, n_time) # Pass n_bands and n_time as integers

        for mamba_block in self.mamba_blocks:
            x_encoded = mamba_block(x_encoded)
        
        x_encoded = self.dropout_layer(x_encoded)

        # Reshape back to original time-frequency grid
        x_out = x_encoded.view(batch, n_bands, n_time, emb_dim)
        
        return x_out