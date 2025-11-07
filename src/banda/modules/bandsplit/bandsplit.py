import math
from typing import Callable, Dict, List, Literal, Tuple

import torch
from torch import nn

from banda.data.item import SourceSeparationBatch
from banda.modules.bandsplit.base import BaseRegisteredBandsplitModule, NormFC

from .band_specs import (
    MusicalBandsplitSpecification,
)



class BaseBandsplitModule(BaseRegisteredBandsplitModule):
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
        config: Dict,
    ) -> None:
        super().__init__(config=config)

        # band_specs: List[Tuple[float, float]] = config.get("band_specs", None)
        emb_dim: int = config["emb_dim"]
        self.in_channels: int = config["in_channels"]
        self.n_bands: int = config["n_bands"]
        self.n_fft : int = config["n_fft"]
        self.emb_dim: int = emb_dim

        self.band_specs, self.band_selectors = self._make_band_definitions()
        self.band_widths: List[int] = self.band_selectors.sum(dim=1).tolist() # number of freq bins in each band
        

        # check non-zero bandwidth
        assert torch.all(torch.tensor(self.band_widths) > 0), "All bands must have non-zero bandwidth"

        require_no_gap: bool = config.get("require_no_gap", True)
        if require_no_gap:
            print(torch.nonzero(torch.sum(self.band_selectors, dim=0) == 0))
            assert torch.all(torch.sum(self.band_selectors, dim=0) > 0), "There are gaps between bands"

        require_no_overlap: bool = config.get("require_no_overlap", False)
        if require_no_overlap:
            assert torch.all(torch.sum(self.band_selectors, dim=1) <= 1), "There is overlap between bands"

        self._initialize_norm_fc_modules()

    def _initialize_norm_fc_modules(self) -> None:


        self.norm_fc_modules = nn.ModuleList(
            [
                NormFC(
                    emb_dim=self.emb_dim,
                    bandwidth=bw,
                    in_channels=self.in_channels,
                )
                for bw in self.band_widths
            ]
        )

    def _make_band_definitions(self,):
        raise NotImplementedError

    def forward(self, x: torch.Tensor, batch: SourceSeparationBatch | None = None) -> torch.Tensor:
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

        with torch.no_grad():
            x = torch.permute(x, (0, 3, 1, 2)).contiguous()
        # (batch, in_chan, n_freq, n_time) -> (batch, n_time, in_chan, n_freq)

        for i, nfm in enumerate(self.norm_fc_modules):
            with torch.no_grad():
                fselector = self.band_selectors[i]  # (n_freq,)

                xb: torch.Tensor = x[:, :, :, fselector]
                # (batch, n_time, in_chan, bw)

                xb = torch.view_as_real(xb)
                # (batch, n_time, in_chan, bw, 2)

                xb = torch.reshape(xb, (batch, n_time, -1))
                # (batch, n_time, in_chan * bw * 2)

            z[:, i, :, :] = nfm(xb)
            # (batch, n_time, emb_dim)

        return z

class FilterbankBandsplitModule(BaseBandsplitModule):

    def __init__(self, *, config):
        super().__init__(config=config)

    def _make_band_definitions(
        self,
    ) -> List[Tuple[float, float]]:
        
        band_specs = self._get_filterbank_band_specs()
        band_selectors = self._band_specs_to_band_selectors(band_specs=band_specs)
        self.band_selectors_for_decoders = band_selectors.clone()

        return band_specs, band_selectors
    
    def _band_specs_to_band_selectors(
        self,
        *,
        band_specs: List[Tuple[float, float]],
    ) -> torch.Tensor:
        band_selectors = torch.zeros(
            (self.n_bands, self.n_fft // 2 + 1), dtype=torch.bool
        )
        for i, (fstart, fend) in enumerate(band_specs):
            band_selectors[i, int(fstart):int(fend)] = 1

        return band_selectors
    
    def _get_filterbank_band_specs(self):
        band_specs = MusicalBandsplitSpecification(
            n_fft=self.config["n_fft"],
            n_bands=self.config["n_bands"],
            fs=self.config["fs"],
            fb_kwargs=self.config.get("fb_kwargs", {})
        ).get_band_specs()
        return band_specs

class FilterbankBandsplitModuleWithDrop(FilterbankBandsplitModule):

    def __init__(self, *, config):
        config["require_no_gap"] = False
        super().__init__(config=config)

    def _make_band_definitions(
        self,
    ) -> List[Tuple[float, float]]:

        band_specs = self._get_filterbank_band_specs()
        undropped_band_selectors = self._band_specs_to_band_selectors(band_specs=band_specs)
        self.band_selectors_for_decoders = undropped_band_selectors.clone()
        # introduce drop rate
        drop_rate: float = self.config.get("drop_rate", 0.0)
        new_band_specs = []
        for fstart, fend in band_specs:
            bandwidth = fend - fstart
            if bandwidth <= 2:
                new_band_specs.append((fstart, fend))
                continue

            reduced_bandwidth = math.floor(bandwidth * (1 - drop_rate))
            fmid = (fstart + fend) / 2

            new_fstart = math.ceil(fmid - reduced_bandwidth / 2)
            new_fend = math.floor(fmid + reduced_bandwidth / 2)

            new_fstart = max(new_fstart, fstart)
            new_fend = max(new_fstart + 1, new_fend)
            new_fend = min(new_fend, fend)

            assert new_fend - new_fstart <= bandwidth
            assert new_fstart >= fstart
            assert new_fend <= fend
            assert new_fend - new_fstart > 0

            new_band_specs.append((new_fstart, new_fend))

        band_selectors = self._band_specs_to_band_selectors(band_specs=new_band_specs)

        return band_specs, band_selectors


class HarmonicBandsplitModule(BaseBandsplitModule):
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
        config: Dict,
    ) -> None:
        super().__init__(config=config)

        self.band_selectors_for_decoders = self.band_selectors.clone()

    def _make_band_definitions(self, *, n_bands: int, n_fft: int):
        band_selectors = torch.zeros(
            (n_bands, n_fft // 2 + 1), dtype=torch.bool
        )
        for i in range(n_bands):
            harmonic_idx = i + 1
            band_selectors[i, harmonic_idx::harmonic_idx] = 1

        return None, band_selectors
