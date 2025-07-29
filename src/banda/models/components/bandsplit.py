#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#


from typing import List, Tuple

import torch
from torch import nn

from banda.models.utils import (
    band_widths_from_specs,
    check_no_overlap,
    check_nonzero_bandwidth,
    check_all_bins_covered, # Import the new function
)
from banda.models.spectral import BandsplitSpecification # Import BandsplitSpecification (now in models folder)


class NormFC(nn.Module):
    """
    A normalized fully connected layer used within the BandSplitModule.
    """
    def __init__(
            self,
            emb_dim: int,
            bandwidth: int,
            in_channel: int,
            normalize_channel_independently: bool = False,
            treat_channel_as_feature: bool = True,
    ) -> None:
        super().__init__()

        self.treat_channel_as_feature = treat_channel_as_feature

        if normalize_channel_independently:
            raise NotImplementedError("Channel-independent normalization is not yet implemented.")

        reim = 2 # Real and Imaginary parts of complex numbers

        self.norm = nn.LayerNorm(in_channel * bandwidth * reim)

        fc_in = bandwidth * reim

        if treat_channel_as_feature:
            fc_in *= in_channel
        else:
            assert emb_dim % in_channel == 0, "Embedding dimension must be divisible by input channels if not treating channel as feature."
            emb_dim = emb_dim // in_channel

        self.fc = nn.Linear(fc_in, emb_dim)

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NormFC module.

        Args:
            xb (torch.Tensor): Input tensor. Shape: (batch, n_time, in_chan, reim * band_width)

        Returns:
            torch.Tensor: Output tensor. Shape: (batch, n_time, emb_dim)
        """
        batch, n_time, in_chan, ribw = xb.shape
        xb = self.norm(xb.reshape(batch, n_time, in_chan * ribw))
        # (batch, n_time, in_chan * reim * band_width)

        if not self.treat_channel_as_feature:
            xb = xb.reshape(batch, n_time, in_chan, ribw)
            # (batch, n_time, in_chan, reim * band_width)

        zb = self.fc(xb)
        # (batch, n_time, emb_dim)
        # OR
        # (batch, n_time, in_chan, emb_dim_per_chan)

        if not self.treat_channel_as_feature:
            batch, n_time, in_chan, emb_dim_per_chan = zb.shape
            # (batch, n_time, in_chan, emb_dim_per_chan)
            zb = zb.reshape((batch, n_time, in_chan * emb_dim_per_chan))

        return zb  # (batch, n_time, emb_dim)


class BandSplitModule(nn.Module):
    """
    Splits a complex spectrogram into multiple frequency bands and processes each band.
    """
    def __init__(
            self,
            band_spec_obj: BandsplitSpecification, # Changed to BandsplitSpecification object
            emb_dim: int,
            in_channel: int,
            require_no_overlap: bool = False,
            require_no_gap: bool = True,
            normalize_channel_independently: bool = False,
            treat_channel_as_feature: bool = True,
    ) -> None:
        """
        Args:
            band_spec_obj (BandsplitSpecification): The BandsplitSpecification object.
            emb_dim (int): Embedding dimension for each band's output.
            in_channel (int): Number of input channels (e.g., 2 for stereo audio).
            require_no_overlap (bool): If True, raises an error if bands overlap.
            require_no_gap (bool): If True, raises an error if there are gaps between bands.
            normalize_channel_independently (bool): If True, normalizes channels independently.
                                                    (Not implemented yet)
            treat_channel_as_feature (bool): If True, treats channels as part of the feature dimension.
        """
        super().__init__()

        self.band_spec_obj = band_spec_obj
        band_specs = self.band_spec_obj.get_band_specs()

        check_nonzero_bandwidth(band_specs)

        if require_no_gap:
            # For overlapping bands, we check if all bins are covered.
            # For non-overlapping bands, we check for strict contiguity.
            if self.band_spec_obj.is_overlapping:
                check_all_bins_covered(band_specs, self.band_spec_obj.max_index)
            else:
                # This check is for strictly non-overlapping, contiguous bands.
                # It ensures no gaps and no overlaps.
                # The original check_no_gap is implicitly handled by check_no_overlap
                # and the way non-overlapping bands are generated.
                # For now, we'll keep check_no_overlap for non-overlapping bands.
                check_no_overlap(band_specs) # This also implies no gaps for non-overlapping.

        if require_no_overlap and self.band_spec_obj.is_overlapping:
            raise ValueError("Cannot require no overlap for an overlapping bandsplit specification.")
        elif require_no_overlap and not self.band_spec_obj.is_overlapping:
            check_no_overlap(band_specs) # Explicitly check for no overlap for non-overlapping bands.


        self.band_specs = band_specs
        # list of [fstart, fend) in index.
        # Note that fend is exclusive.
        self.band_widths = band_widths_from_specs(band_specs)
        self.n_bands = len(band_specs)
        self.emb_dim = emb_dim

        self.norm_fc_modules = nn.ModuleList(
            [
                NormFC(
                    emb_dim=emb_dim,
                    bandwidth=bw,
                    in_channel=in_channel,
                    normalize_channel_independently=normalize_channel_independently,
                    treat_channel_as_feature=treat_channel_as_feature,
                )
                for bw in self.band_widths
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BandSplitModule.

        Args:
            x (torch.Tensor): Complex spectrogram. Shape: (batch, in_chan, n_freq, n_time, 2)
                              (Note: The input is expected to be real-view of complex spectrogram)

        Returns:
            torch.Tensor: Processed band features. Shape: (batch, n_bands, n_time, emb_dim)
        """
        
        batch, doubled_in_chan, n_freq, n_time = x.shape
        
        # Assuming doubled_in_chan is original_channels * 2
        original_in_chan = doubled_in_chan // 2
        reim = 2 # Real and Imaginary parts

        # Reshape x from (batch, original_channels * 2, n_freq, n_time)
        # to (batch, original_channels, reim, n_freq, n_time)
        x_reshaped = x.reshape(batch, original_in_chan, reim, n_freq, n_time)
        
        # Permute to (batch, n_time, original_channels, reim, n_freq)
        xr = torch.permute(x_reshaped, (0, 4, 1, 2, 3))

        z = torch.zeros(
            size=(batch, self.n_bands, n_time, self.emb_dim),
            device=x.device,
            dtype=x.dtype # Use the same dtype as input for consistency
        )

        for i, nfm in enumerate(self.norm_fc_modules):
            fstart, fend = self.band_specs[i]
            xb = xr[..., fstart:fend]
            # (batch, n_time, in_chan, reim, band_width)
            xb = torch.reshape(xb, (batch, n_time, original_in_chan, -1))
            # (batch, n_time, in_chan, reim * band_width)
            z[:, i, :, :] = nfm(xb.contiguous())

        return z