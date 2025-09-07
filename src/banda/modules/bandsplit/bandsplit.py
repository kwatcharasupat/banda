from typing import List, Tuple

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .band_specs import (
    band_widths_from_specs,
    check_no_gap,
    check_no_overlap,
    check_nonzero_bandwidth,
)

from torch.utils.checkpoint import checkpoint_sequential


class NormFC(nn.Module):
    """
    Fully connected layer with optional weight normalization and layer normalization.

    Args:
        emb_dim (int): Output embedding dimension.
        bandwidth (int): Bandwidth of the subband.
        in_channels (int): Number of input channels.
        use_weight_norm (bool, optional): Whether to apply weight normalization. Defaults to True.
        _verbose (bool): Verbose mode.
    """

    def __init__(
        self,
        *,
        emb_dim: int,
        bandwidth: int,
        in_channels: int,
    ) -> None:
        super().__init__()

        reim: int = 2

        norm_in: int = in_channels * bandwidth * reim
        fc_in: int = bandwidth * reim * in_channels

        self.combined = nn.Sequential(
            nn.LayerNorm(norm_in),
            weight_norm(nn.Linear(fc_in, emb_dim)),
        )

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for NormFC.

        Args:
            xb (torch.Tensor): Subband spectrogram. Shape: (batch, n_time, in_chan * bw * 2)

        Returns:
            torch.Tensor: Subband embedding. Shape: (batch, n_time, emb_dim)
        """
        return checkpoint_sequential(self.combined, 1, xb, use_reentrant=False)


class BandSplitModule(nn.Module):
    """
    Module for splitting input spectrogram into subbands and processing each subband.

    Args:
        band_specs (List[Tuple[float, float]]): List of frequency band specifications as (start, end).
        emb_dim (int): Output embedding dimension for each subband.
        in_channels (int): Number of input channels.
        require_no_overlap (bool, optional): Ensure no overlap between bands. Defaults to False.
        require_no_gap (bool, optional): Ensure no gaps between bands. Defaults to True.
        _verbose (bool): Verbose mode.
    """

    def __init__(
        self,
        *,
        band_specs: List[Tuple[float, float]],
        emb_dim: int,
        in_channels: int,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
    ) -> None:
        super().__init__()

        check_nonzero_bandwidth(band_specs=band_specs)

        if require_no_gap:
            check_no_gap(band_specs=band_specs)

        if require_no_overlap:
            check_no_overlap(band_specs=band_specs)

        self.band_specs: List[Tuple[float, float]] = band_specs
        self.band_widths: List[int] = band_widths_from_specs(band_specs=band_specs)
        self.n_bands: int = len(band_specs)
        self.emb_dim: int = emb_dim

        self.norm_fc_modules = nn.ModuleList(
            [
                NormFC(
                    emb_dim=emb_dim,
                    bandwidth=bw,
                    in_channels=in_channels,
                )
                for bw in self.band_widths
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for BandSplitModule.

        Args:
            x (torch.Tensor): Input spectrogram. Shape: (batch, in_chan, n_freq, n_time)

        Returns:
            torch.Tensor: Embedding tensor. Shape: (batch, n_bands, n_time, emb_dim)
        """

        batch, _, _, n_time = x.shape

        z: torch.Tensor = torch.zeros(
            size=(batch, self.n_bands, n_time, self.emb_dim), device=x.device
        )

        x = torch.permute(x, (0, 3, 1, 2)).contiguous()
        # (batch, in_chan, n_freq, n_time) -> (batch, n_time, in_chan, n_freq)

        for i, nfm in enumerate(self.norm_fc_modules):
            fstart, fend = self.band_specs[i]

            xb: torch.Tensor = x[:, :, :, int(fstart) : int(fend)]
            # (batch, n_time, in_chan, bw)

            xb = torch.view_as_real(xb)
            # (batch, n_time, in_chan, bw, 2)

            xb = torch.reshape(xb, (batch, n_time, -1))
            # (batch, n_time, in_chan * bw * 2)

            z[:, i, :, :] = nfm(xb)
            # (batch, n_time, emb_dim)

        return z
