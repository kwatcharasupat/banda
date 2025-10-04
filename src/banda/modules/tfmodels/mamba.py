from typing import List
from omegaconf import DictConfig
import torch

from banda.modules.tfmodels.tfmodel import TimeFrequencyModellingModule
from torch import nn

from banda.modules.utils import Transpose

from torch.utils.checkpoint import checkpoint_sequential


from torch.nn.utils.parametrizations import weight_norm
from mamba_ssm import Mamba2


class MambaParams(DictConfig):
    n_modules: int = 6
    emb_dim: int = 128


class MambaTFModel(TimeFrequencyModellingModule):
    def __init__(
        self,
        *,
        config: DictConfig,
    ) -> None:
        super().__init__(config=config)

        seqband: List[nn.Module] = []
        for _ in range(2 * self.config.n_modules):
            seqband += [
                ResidualMamba(
                    emb_dim=self.config.emb_dim,
                    hidden_dim=self.config.hidden_dim,
                    n_heads=self.config.n_heads,
                    dropout=self.config.dropout,
                    max_seq_len=self.config.max_seq_len,
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

        z = checkpoint_sequential(
            self.seqband, self.config.n_modules, z, use_reentrant=False
        )
        return z  # (batch, n_bands, n_time, emb_dim)


class NormFF(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.combined = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Sequential(
                weight_norm(nn.Linear(emb_dim, emb_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
            ),
            nn.Sequential(
                weight_norm(nn.Linear(emb_dim, emb_dim)),
                nn.Dropout(dropout),
            ),
        )

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for NormFF.

        Args:
            xb (torch.Tensor): Subband spectrogram. Shape: (batch, n_time, in_chan * bw * 2)

        Returns:
            torch.Tensor: Subband embedding. Shape: (batch, n_time, emb_dim)
        """
        return xb + self.combined(xb)


class ResidualMamba(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        hidden_dim: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.mamba: nn.Module = MambaBlock(
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.fc: nn.Module = NormFF(emb_dim=emb_dim, dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.mamba(z)
        z = self.fc(z)
        return z


class MambaBlock(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.ff = weight_norm(nn.Linear(emb_dim, emb_dim))
        self.hidden_dim = hidden_dim

        self.mamba = Mamba2(
            d_model=emb_dim,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch, n_uncrossed, n_across, emb_dim = z.shape

        z_ = self.norm(z)
        # (batch, n_uncrossed, n_across, emb_dim)
        z_ = z_.reshape(-1, n_across, emb_dim)
        # print("Before Mamba:", z_.shape)
        z_ = self.mamba(z_)
        # print("After Mamba:", z_.shape)
        z_ = self.ff(z_)
        z_ = z_.reshape(batch, n_uncrossed, n_across, emb_dim)

        z = z + z_

        return z
