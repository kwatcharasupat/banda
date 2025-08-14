#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.


from banda.utils.registry import MODELS_REGISTRY

from typing import List, Tuple
import structlog
import logging

logger = structlog.get_logger(__name__)
logging.getLogger(__name__).setLevel(logging.INFO) # Changed to INFO

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from omegaconf import DictConfig, OmegaConf
import hydra.utils

from banda.models.common_components.spectral_components.spectral_base import BandsplitSpecification, get_bandsplit_specs_factory
from banda.models.common_components.configs.common_configs import BandsplitModuleConfig, STFTConfig


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

        # Ensure bandwidth is not zero before creating LayerNorm
        if bandwidth == 0:
            raise ValueError(f"NormFC received bandwidth of 0. This is not allowed. emb_dim={emb_dim}, in_channel={in_channel}")

        self.norm: nn.LayerNorm = nn.LayerNorm(in_channel * bandwidth * reim, eps=1e-5) # Changed eps to 1e-5

        fc_in: int = bandwidth * reim

        if treat_channel_as_feature:
            fc_in *= in_channel
        else:
            assert emb_dim % in_channel == 0, "Embedding dimension must be divisible by input channels if not treating channel as feature."
            emb_dim = emb_dim // in_channel

        self.fc: nn.Linear = weight_norm(nn.Linear(fc_in, emb_dim))

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
        xb_reshaped: torch.Tensor = xb.reshape(batch, n_time, in_chan * ribw)
        logger.debug(f"NormFC: xb_reshaped shape: {xb_reshaped.shape}, min: {xb_reshaped.min().item()}, max: {xb_reshaped.max().item()}, mean: {xb_reshaped.mean().item()}")
        if torch.isnan(xb_reshaped).any():
            logger.error("NormFC: NaN detected in xb_reshaped", mean_val=xb_reshaped.mean().item())
            raise ValueError("NaN in xb_reshaped in NormFC")
        
        xb_normalized: torch.Tensor = self.norm(xb_reshaped)
        logger.debug(f"NormFC: xb_normalized shape: {xb_normalized.shape}, min: {xb_normalized.min().item()}, max: {xb_normalized.max().item()}, mean: {xb_normalized.mean().item()}")
        if torch.isnan(xb_normalized).any():
            logger.error("NormFC: NaN detected in xb_normalized", mean_val=xb_normalized.mean().item())
            raise ValueError("NaN in xb_normalized in NormFC")

        if not self.treat_channel_as_feature:
            xb_normalized = xb_normalized.reshape(batch, n_time, in_chan, ribw) # Shape: (batch, n_time, in_chan, reim * band_width)

        zb: torch.Tensor = self.fc(xb_normalized)
        logger.debug(f"NormFC: zb shape after fc layer: {zb.shape}, min: {zb.min().item()}, max: {zb.max().item()}, mean: {zb.mean().item()}")
        if torch.isnan(zb).any():
            logger.error("NormFC: NaN detected in zb after fc layer", mean_val=zb.mean().item())
            raise ValueError("NaN in zb after fc layer")
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
            # Removed n_time_frames as it was a misnomer for n_freq_bins
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

        self.band_widths: List[int] = []
        for fstart, fend in band_specs:
            bw = fend - fstart
            logger.debug(f"BandSplitModule: Processing band ({fstart}, {fend}) with bandwidth {bw}")
            if bw <= 0: # Changed from bw == 0 to bw <= 0 to catch negative or zero bandwidths
                logger.error(f"BandSplitModule: Invalid bandwidth detected for band ({fstart}, {fend}). Bandwidth: {bw}")
                raise ValueError("Bandwidth must be positive.")
            self.band_widths.append(bw)

        if require_no_gap:
            # For overlapping bands, we check if all bins are covered.
            # For non-overlapping bands, we check for strict contiguity.
            if self.band_spec_obj.is_overlapping:
                # Check if all bins are covered
                all_bins = set()
                for fstart, fend in band_specs:
                    all_bins.update(range(fstart, fend))
                expected_bins = set(range(self.band_spec_obj.max_index))
                # Removed the problematic line: if self.band_spec_obj.drop_dc_band: expected_bins = set(range(1, self.band_spec_obj.max_index + 1))
                if not expected_bins.issubset(all_bins):
                    raise ValueError("Not all frequency bins are covered by the bandsplit specification.")
            else:
                # This check is for strictly non-overlapping, contiguous bands.
                # It ensures no gaps and no overlaps.
                # The original check_no_gap is implicitly handled by check_no_overlap
                # and the way non-overlapping bands are generated.
                # For now, we'll keep check_no_overlap for non-overlapping bands.
                # Check for no overlap
                for i in range(len(band_specs) - 1):
                    if band_specs[i][1] > band_specs[i+1][0]:
                        raise ValueError("Bands are overlapping.") # This also implies no gaps for non-overlapping.

        if require_no_overlap and self.band_spec_obj.is_overlapping:
            raise ValueError("Cannot require no overlap for an overlapping bandsplit specification.")
        elif require_no_overlap and not self.band_spec_obj.is_overlapping:
            # Check for no overlap
                for i in range(len(band_specs) - 1):
                    if band_specs[i][1] > band_specs[i+1][0]:
                        raise ValueError("Bands are overlapping.") # Explicitly check for no overlap for non-overlapping bands.


        self.band_specs: List[Tuple[int, int]] = band_specs
        # list of [fstart, fend) in index.
        # Note that fend is exclusive.
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
        logger.debug(f"BandSplitModule: x_reshaped shape: {x_reshaped.shape}, min: {x_reshaped.min().item()}, max: {x_reshaped.max().item()}, mean: {x_reshaped.mean().item()}")
        if torch.isnan(x_reshaped).any():
            logger.error("BandSplitModule: NaN detected in x_reshaped", mean_val=x_reshaped.mean().item())
            raise ValueError("NaN in x_reshaped in BandSplitModule")
        
        # Permute to (batch, n_time, original_channels, reim, n_freq)
        xr: torch.Tensor = torch.permute(x_reshaped, (0, 4, 1, 2, 3))
        logger.debug(f"BandSplitModule: xr shape after permute: {xr.shape}, min: {xr.min().item()}, max: {xr.max().item()}, mean: {xr.mean().item()}")
        if torch.isnan(xr).any():
            logger.error("BandSplitModule: NaN detected in xr after permute", mean_val=xr.mean().item())
            raise ValueError("NaN in xr in BandSplitModule")

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
            logger.debug(f"BandSplitModule: xb shape before reshape for band {i}: {xb.shape}, min: {xb.min().item()}, max: {xb.max().item()}, mean: {xb.mean().item()}")
            xb = torch.reshape(xb, (batch, n_time, original_in_chan, -1)) # Shape: (batch, n_time, in_chan, reim * band_width)
            logger.debug(f"BandSplitModule: xb shape after reshape for band {i}: {xb.shape}, min: {xb.min().item()}, max: {xb.max().item()}, mean: {xb.mean().item()}")
            z[:, i, :, :] = nfm(xb.contiguous())
            if torch.isnan(z[:, i, :, :]).any():
                logger.error(f"BandSplitModule: NaN detected in z after NormFC for band {i}", mean_val=z[:, i, :, :].mean().item())
                raise ValueError(f"NaN in z for band {i} in BandSplitModule")

        return z

    @classmethod
    def from_config(cls, cfg: BandsplitModuleConfig, stft_config: STFTConfig, emb_dim: int, in_channel: int, drop_dc_band: bool) -> "BandSplitModule":
        """
        Instantiates a BandSplitModule from a BandsplitModuleConfig.

        Args:
            cfg (BandsplitModuleConfig): A BandsplitModuleConfig object containing the bandsplit configuration.
            stft_config (STFTConfig): The STFT configuration, needed for band specification.
            emb_dim (int): Embedding dimension for each band's output.
            in_channel (int): Number of input channels (e.g., 2 for stereo audio).
            drop_dc_band (bool): Whether the DC band is dropped from the spectrogram.

        Returns:
            BandSplitModule: An instance of the BandSplitModule.
        """
        # The band_spec_obj is already instantiated within the BandsplitModuleConfig
        band_spec_obj: BandsplitSpecification = cfg.band_specs

        # Calculate n_freq_bins correctly
        n_freq_bins = stft_config.n_fft // 2 + 1
        if drop_dc_band:
            n_freq_bins -= 1
        
        # Ensure that the band_spec_obj's max_index matches the calculated n_freq_bins
        # This is a sanity check to ensure consistency between STFT and band specification
        if band_spec_obj.max_index != n_freq_bins:
            raise ValueError(f"Mismatch between band_spec_obj.max_index ({band_spec_obj.max_index}) and calculated n_freq_bins ({n_freq_bins}).")

        return cls(
            band_spec_obj=band_spec_obj,
            emb_dim=emb_dim,
            in_channel=in_channel,
            require_no_overlap=cfg.require_no_overlap,
            require_no_gap=cfg.require_no_gap,
            normalize_channel_independently=cfg.normalize_channel_independently,
            treat_channel_as_feature=cfg.treat_channel_as_feature,
        )