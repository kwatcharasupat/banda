from collections.abc import Collection
import math
from typing import Callable, Dict, List, Literal, Optional, Tuple

from librosa import note_to_hz
import numpy as np
import torch
from torch import nn

from banda.data.item import SourceSeparationBatch
from banda.modules.bandsplit.base import NormFC, BaseRegisteredBandsplitModule

from librosa import filters
from torchaudio.transforms import Spectrogram

from scipy import fft


def nextpow2(x: np.ndarray) -> np.ndarray:
    return np.power(2, np.ceil(np.log2(x))).astype(int)


class CQTish(nn.Module):
    def __init__(
        self,
        *,
        n_bands: int,
        bins_per_band: int,
        hop_length: int,
        fs: int,
        n_fft: int,  # for decoder, not used in encoder
        fmin_hz: float | str = "C0",
    ) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.fs = fs
        self.decoder_n_fft = n_fft

        if isinstance(fmin_hz, str):
            fmin_hz = note_to_hz(fmin_hz)
        self.fmin_hz = fmin_hz

        # get the number of bins needed to reach nyquist
        self.n_bins = int(np.ceil(self.n_bands * bins_per_band))
        self.bins_per_band = bins_per_band
        fmax_hz = fs / 2
        center_freq_hz = fmin_hz * np.power(
            fmax_hz / fmin_hz, np.arange(self.n_bins) / (self.n_bins)
        )
        # print("CQTish center frequencies:", center_freq_hz)
        # print("Max frequency:", center_freq_hz[-1], "Nyquist frequency:", fs / 2)

        _, lengths = filters.wavelet(
            freqs=center_freq_hz,
            sr=fs,
        )
        self.lengths = nextpow2(np.array(lengths)).astype(int)

        # half of fft sizes cannot be smaller than hop length
        self.lengths = np.maximum(self.lengths, hop_length * 2)

        self.unique_lengths = np.sort(np.unique(self.lengths)).astype(int)[::-1]

        # for each length, compute the cqt filters
        cqt_kernels = {}
        for n_fft in self.unique_lengths:
            wavelets, wavelet_lengths = filters.wavelet(
                freqs=center_freq_hz[self.lengths == n_fft],
                sr=fs,
            )
            wavelets = wavelets * wavelet_lengths[:, None] / n_fft
            wavelets_fft = fft.fft(wavelets, n=n_fft, axis=1)[:, : n_fft // 2 + 1]
            # (n_bins_for_length, n_fft)
            cqt_kernels[n_fft] = torch.tensor(
                wavelets_fft, dtype=torch.complex64
            )  # (n_bins_for_length, n_fft//2 + 1)

        self.stfts = nn.ModuleList(
            [
                Spectrogram(
                    n_fft=int(n_fft),
                    win_length=int(n_fft),
                    hop_length=int(hop_length),
                    power=None,
                    center=True,
                    pad_mode="constant",
                    normalized=True,
                    onesided=True,
                )
                for n_fft in self.unique_lengths
            ]
        )

        self.cqt_kernels = nn.ParameterList(
            [
                nn.Parameter(cqt_kernels[length], requires_grad=False)
                for length in self.unique_lengths
            ]
        )

        self._cqt_kernels_to_band_specs(center_freq_hz)

    def _cqt_kernels_to_band_specs(
        self, center_freq_hz: np.ndarray
    ) -> List[Tuple[float, float]]:
        center_freq_hz = center_freq_hz.reshape(self.n_bands, self.bins_per_band)
        # print("CQTish center frequencies reshaped:", center_freq_hz)

        center_freq_log = np.log2(center_freq_hz)
        # midpoint
        midpoints_log = (
            center_freq_log[:-1, -1] + center_freq_log[1:, 0]
        ) / 2  # (n_bands - 1, )
        midpoints_hz = np.power(2.0, midpoints_log)  # (n_bands - 1, )

        band_edges_hz = np.stack(
            [
                np.concatenate(([0.0], midpoints_hz)),
                np.concatenate((midpoints_hz, [self.fs / 2])),
            ],
            axis=1,
        )

        band_edges_idx = np.round(band_edges_hz / self.fs * self.decoder_n_fft)
        band_edges_idx[:, 1] = np.maximum(
            band_edges_idx[:, 1], band_edges_idx[:, 0] + 1
        )
        band_edges_idx[0, 0] = 0  # ensure first band starts at 0
        band_edges_idx[-1, -1] = (
            self.decoder_n_fft // 2 + 1
        )  # ensure last band ends at nyquist


        band_selectors = np.zeros(
            (self.n_bands, self.decoder_n_fft // 2 + 1), dtype=bool
        )
        for i in range(self.n_bands):
            start_idx = int(band_edges_idx[i, 0])
            end_idx = int(band_edges_idx[i, 1])
            band_selectors[i, start_idx:end_idx] = 1

        self.band_specs = band_edges_idx.astype(int).tolist()
        # print("CQTish band specs (in FFT bin indices):", self.band_specs)
        self.band_selectors_for_decoders = torch.tensor(
            band_selectors, dtype=torch.bool, requires_grad=False
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_chan, n_time)

        batch, in_chan, n_time = x.shape
        C_list = []
        for i, n_fft in enumerate(self.unique_lengths):
            stft = self.stfts[i]
            cqt_kernel = self.cqt_kernels[i]  # (n_bins_for_length, n_fft//2 + 1)
            X = stft(x)  # (batch, in_chan, n_fft//2 + 1, n_frames)
            C = torch.einsum(
                "bcft,kf->bckt", X, cqt_kernel
            )  # (batch, in_chan, n_bins_for_length, n_frames)
            C_list.append(C)

        C = torch.cat(C_list, dim=2)  # (batch, in_chan, n_bins, n_frames)
        # print("CQTish output shape:", C.shape)
        return C


class CQTBandsplitModule(BaseRegisteredBandsplitModule):
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
        self.bins_per_band: int = config["bins_per_band"]
        self.n_fft: int = config["n_fft"]
        self.hop_length: int = config["hop_length"]
        self.emb_dim: int = emb_dim
        self.fs: int = config["fs"]

        self.cqt = CQTish(
            n_bands=self.n_bands,
            bins_per_band=self.bins_per_band,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fs=self.fs,
            fmin_hz=config.get("fmin_hz", "C0"),
        )
        self.band_selectors = self.cqt.band_selectors_for_decoders
        self.band_selectors_for_decoders = self.cqt.band_selectors_for_decoders

        require_no_gap: bool = config.get("require_no_gap", True)
        if require_no_gap:
            assert torch.all(torch.sum(self.band_selectors, dim=0) > 0), (
                "There are gaps between bands at indices "+ str(torch.nonzero(torch.sum(self.band_selectors, dim=0) == 0))
            )

        require_no_overlap: bool = config.get("require_no_overlap", False)
        if require_no_overlap:
            assert torch.all(torch.sum(self.band_selectors, dim=1) <= 1), (
                "There is overlap between bands"
            )

        self._initialize_norm_fc_modules()

    def _initialize_norm_fc_modules(self) -> None:
        self.norm_fc_modules = nn.ModuleList(
            [
                NormFC(
                    emb_dim=self.emb_dim,
                    bandwidth=self.bins_per_band,
                    in_channels=self.in_channels,
                )
                for _ in range(self.n_bands)
            ]
        )

    def forward(self, _: torch.Tensor, batch: SourceSeparationBatch) -> torch.Tensor:
        """
        Forward pass for BandSplitModule.

        Args:
            x (torch.Tensor): Input audio signal. Shape: (batch, in_chan, n_time)
        Returns:
            torch.Tensor: Embedding tensor. Shape: (batch, n_bands, n_time, emb_dim)
        """

        with torch.no_grad():
            audio = batch.mixture["audio"]  # (batch, in_chan, n_time)
            batch, _, _ = audio.shape
            X = self.cqt(audio)  # (batch, in_chan, n_bins, n_frames)
            _, _, _, n_time = X.shape
            Xc = torch.reshape(
                X, (batch, self.in_channels, self.n_bands, self.bins_per_band, -1)
            )
            Xc = torch.permute(Xc, (0, 2, 4, 1, 3)).contiguous()
            # (batch, n_bands, n_frames, in_chan, bins_per_band)
            Xc = torch.view_as_real(Xc)
            # print(Xc.shape)

        z: torch.Tensor = torch.zeros(
            size=(batch, self.n_bands, n_time, self.emb_dim), device=audio.device
        )

        for i, nfm in enumerate(self.norm_fc_modules):
            with torch.no_grad():
                xb: torch.Tensor = Xc[:, i, ...]
                # (batch, n_frames, in_chan, bins_per_band, 2)
                xb = torch.reshape(xb, (batch, n_time, -1 ))
                # (batch, n_time, in_chan * bw * 2)
            z[:, i, :, :] = nfm(xb)
            # (batch, n_time, emb_dim)

        return z


if __name__ == "__main__":
    cqt = CQTish(
        n_bands=64, bins_per_band=8, fs=44100, fmin_hz="C0", hop_length=512, n_fft=4096
    )

    dummy_audio = torch.randn(3, 2, 44100 * 5)  # (batch, in_chan, n_time)
    cqt_output = cqt(dummy_audio)
