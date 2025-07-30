#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

from banda.utils.registry import MODELS_REGISTRY

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

    This module applies Layer Normalization followed by a linear transformation
    to frequency band features. It can optionally treat channels as part of the
    feature dimension.
    """
    def __init__(
            self,
            emb_dim: int,
            bandwidth: int,
            in_channel: int,
            normalize_channel_independently: bool = False,
            treat_channel_as_feature: bool = True,
    ) -> None:
        """
        Initializes the NormFC module.

        Args:
            emb_dim (int): The output embedding dimension.
            bandwidth (int): The frequency bandwidth of the input.
            in_channel (int): The number of input channels.
            normalize_channel_independently (bool): If True, normalizes channels independently.
                                                    (Currently not implemented).
            treat_channel_as_feature (bool): If True, treats channels as part of the feature dimension
                                             for the linear layer.
        """
        super().__init__()

        self.treat_channel_as_feature: bool = treat_channel_as_feature

        if normalize_channel_independently:
            raise NotImplementedError("Channel-independent normalization is not yet implemented.")

        reim: int = 2 # Real and Imaginary parts of complex numbers

        self.norm: nn.LayerNorm = nn.LayerNorm(in_channel * bandwidth * reim)

        fc_in: int = bandwidth * reim

        if treat_channel_as_feature:
            fc_in *= in_channel
        else:
            assert emb_dim % in_channel == 0, "Embedding dimension must be divisible by input channels if not treating channel as feature."
            emb_dim = emb_dim // in_channel

        self.fc: nn.Linear = nn.Linear(fc_in, emb_dim)

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NormFC module.

        Args:
            xb (torch.Tensor): Input tensor containing frequency band data.
                Shape: (batch, n_time, in_chan, reim * band_width)

        Returns:
            torch.Tensor: Output tensor after normalization and linear transformation.
                Shape: (batch, n_time, emb_dim)
        """
        batch: int
        n_time: int
        in_chan: int
        ribw: int
        batch, n_time, in_chan, ribw = xb.shape
        xb_reshaped: torch.Tensor = xb.reshape(batch, n_time, in_chan * ribw) # Shape: (batch, n_time, in_chan * reim * band_width)
        xb_normalized: torch.Tensor = self.norm(xb_reshaped)

        if not self.treat_channel_as_feature:
            xb_normalized = xb_normalized.reshape(batch, n_time, in_chan, ribw) # Shape: (batch, n_time, in_chan, reim * band_width)

        zb: torch.Tensor = self.fc(xb_normalized)
        # Shape: (batch, n_time, emb_dim)
        # OR (if not treat_channel_as_feature): (batch, n_time, in_chan, emb_dim_per_chan)

        if not self.treat_channel_as_feature:
            batch, n_time, in_chan, emb_dim_per_chan = zb.shape
            zb = zb.reshape((batch, n_time, in_chan * emb_dim_per_chan))

        return zb  # Shape: (batch, n_time, emb_dim)


@MODELS_REGISTRY.register("bandsplit_module")
class BandSplitModule(nn.Module):
    """
    Splits a complex spectrogram into multiple frequency bands and processes each band.

    This module takes a complex spectrogram, extracts features from defined frequency
    bands, and processes them using `NormFC` layers. It supports checks for band
    overlap and gaps.
    """
    def __init__(
            self,
            band_spec_obj: BandsplitSpecification,
            emb_dim: int,
            in_channel: int,
            require_no_overlap: bool = False,
            require_no_gap: bool = True,
            normalize_channel_independently: bool = False,
            treat_channel_as_feature: bool = True,
    ) -> None:
        """
        Initializes the BandSplitModule.

        Args:
            band_spec_obj (BandsplitSpecification): The BandsplitSpecification object
                                                    defining the frequency bands.
            emb_dim (int): Embedding dimension for each band's output.
            in_channel (int): Number of input channels (e.g., 2 for stereo audio).
            require_no_overlap (bool): If True, raises an error if bands overlap.
            require_no_gap (bool): If True, raises an error if there are gaps between bands.
            normalize_channel_independently (bool): If True, normalizes channels independently.
                                                    (Currently not implemented).
            treat_channel_as_feature (bool): If True, treats channels as part of the feature dimension.

        Raises:
            ValueError: If `normalize_channel_independently` is True (not implemented).
            ValueError: If `require_no_overlap` is True for an overlapping band specification.
        """
        super().__init__()

        self.band_spec_obj: BandsplitSpecification = band_spec_obj
        band_specs: List[Tuple[int, int]] = self.band_spec_obj.get_band_specs()

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


        self.band_specs: List[Tuple[int, int]] = band_specs
        # list of [fstart, fend) in index.
        # Note that fend is exclusive.
        self.band_widths: List[int] = band_widths_from_specs(band_specs)
        self.n_bands: int = len(band_specs)
        self.emb_dim: int = emb_dim

        self.norm_fc_modules: nn.ModuleList = nn.ModuleList(
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
            x (torch.Tensor): Input complex spectrogram (real-view).
                Shape: (batch, in_chan * 2, n_freq, n_time)
                (where `in_chan * 2` represents concatenated real and imaginary parts
                of the original `in_chan` complex channels).

        Returns:
            torch.Tensor: Processed band features.
                Shape: (batch, n_bands, n_time, emb_dim)
        """
        
        batch: int
        doubled_in_chan: int
        n_freq: int
        n_time: int
        batch, doubled_in_chan, n_freq, n_time = x.shape
        
        # Assuming doubled_in_chan is original_channels * 2
        original_in_chan: int = doubled_in_chan // 2
        reim: int = 2 # Real and Imaginary parts
        
        # Reshape x from (batch, original_channels * 2, n_freq, n_time)
        # to (batch, original_channels, reim, n_freq, n_time)
        x_reshaped: torch.Tensor = x.reshape(batch, original_in_chan, reim, n_freq, n_time)
        
        # Permute to (batch, n_time, original_channels, reim, n_freq)
        xr: torch.Tensor = torch.permute(x_reshaped, (0, 4, 1, 2, 3))

        z: torch.Tensor = torch.zeros(
            size=(batch, self.n_bands, n_time, self.emb_dim),
            device=x.device,
            dtype=x.dtype # Use the same dtype as input for consistency
        )

        for i, nfm in enumerate(self.norm_fc_modules):
            fstart: int
            fend: int
            fstart, fend = self.band_specs[i]
            xb: torch.Tensor = xr[..., fstart:fend] # Shape: (batch, n_time, in_chan, reim, band_width)
            xb = torch.reshape(xb, (batch, n_time, original_in_chan, -1)) # Shape: (batch, n_time, in_chan, reim * band_width)
            z[:, i, :, :] = nfm(xb.contiguous())

        return z