from typing import Callable, List, Literal

import torch
from torch import nn
from torch.nn.modules import rnn

import torch.backends.cuda

from torch.nn.utils.parametrizations import weight_norm


from torch.utils.checkpoint import checkpoint_sequential

from banda.modules.utils import Transpose


RNNType = Literal["RNN", "GRU", "LSTM"]

class TimeFrequencyModellingModule(nn.Module):
    """
    Base class for time-frequency modelling modules.
    """

    def __init__(self) -> None:
        """
        Args:
            _verbose (bool): Whether to enable verbose logging. Default: False.
        """
        super().__init__()


class ResidualRNN(nn.Module):
    """
    Residual RNN module with optional layer normalization and weight normalization.

    Args:
        emb_dim (int): Input embedding dimension.
        rnn_dim (int): Hidden dimension of the RNN.
        bidirectional (bool): Whether the RNN is bidirectional. Default: True.
        rnn_type (RNNType): Type of RNN to use ("LSTM", "GRU", or "RNN"). Default: "LSTM".
        use_layer_norm (bool): Whether to apply layer normalization. Default: True.
        use_weight_norm (bool): Whether to apply weight normalization. Default: True.
        _verbose (bool): Whether to enable verbose logging. Default: False.
    """

    def __init__(
        self,
        *,
        emb_dim: int,
        rnn_dim: int,
        bidirectional: bool = True,
        rnn_type: RNNType = "GRU",
       
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(emb_dim)

        self.rnn: nn.Module = rnn.__dict__[rnn_type](
            input_size=emb_dim,
            hidden_size=rnn_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )


        self.fc: nn.Module = weight_norm(
            nn.Linear(
                in_features=rnn_dim * (2 if bidirectional else 1), out_features=emb_dim
            )
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ResidualRNN module.

        Args:
            z (torch.Tensor): Input tensor. Shape: (batch, n_uncrossed, n_across, emb_dim)

        Returns:
            torch.Tensor: Output tensor. Shape: (batch, n_uncrossed, n_across, emb_dim)
        """
        # z = (batch, n_uncrossed, n_across, emb_dim)

        z0: torch.Tensor = torch.clone(z)

        z = self.norm(z)  # (batch, n_uncrossed, n_across, emb_dim)
        
        batch, n_uncrossed, n_across, emb_dim = z.shape

        z = torch.reshape(z, (batch * n_uncrossed, n_across, emb_dim))

        z = self.rnn(z.contiguous())[0]
        # (batch * n_uncrossed, n_across, dir_rnn_dim)

        z = torch.reshape(z, (batch, n_uncrossed, n_across, -1))
        # (batch, n_uncrossed, n_across, dir_rnn_dim)

        z = self.fc(z)  # (batch, n_uncrossed, n_across, emb_dim)

        z = z + z0

        return z


class RNNSeqBandModellingModule(TimeFrequencyModellingModule):
    """
    Sequential band modelling module using ResidualRNN.

    Args:
        n_modules (int): Number of ResidualRNN modules. Default: 12.
        emb_dim (int): Input embedding dimension.
        rnn_dim (int): Hidden dimension of the RNN.
        bidirectional (bool): Whether the RNN is bidirectional. Default: True.
        rnn_type (RNNType): Type of RNN to use ("LSTM", "GRU", or "RNN"). Default: "LSTM".
        parallel_mode (bool): Whether to enable parallel mode. Default: False.
        _verbose (bool): Whether to enable verbose logging. Default: False.
    """

    def __init__(
        self,
        *,
        n_modules: int = 8,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: RNNType = "GRU",
       
    ) -> None:
        super().__init__()

        self.n_modules: int = n_modules

        seqband: List[nn.Module] = []
        for _ in range(2 * n_modules):
            seqband += [
                ResidualRNN(
                    emb_dim=emb_dim,
                    rnn_dim=rnn_dim,
                    bidirectional=bidirectional,
                    rnn_type=rnn_type,
                ),
                Transpose(dim0=1, dim1=2),
            ]

        self.seqband = nn.Sequential(*seqband)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SeqBandModellingModule.

        Args:
            z (torch.Tensor): Input tensor. Shape: (batch, n_bands, n_time, emb_dim)

        Returns:
            torch.Tensor: Output tensor. Shape: (batch, n_bands, n_time, emb_dim)
        """

        z = checkpoint_sequential(self.seqband, self.n_modules, z, use_reentrant=False)
        return z  # (batch, n_bands, n_time, emb_dim)
