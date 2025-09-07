import os
from abc import abstractmethod
from typing import Callable, List, Literal, Tuple, Optional

import numpy as np
import torch
from torchaudio import functional as taF

from banda.modules.bandsplit.filterbanks import (
    bark_filterbank,
    erb_filterbank,
    minibark_filterbank,
    musical_filterbank,
    triangular_bark_filterbank,
)


def band_widths_from_specs(*, band_specs: List[Tuple[float, float]]) -> List[float]:
    """
    Calculate the bandwidths from band specifications.

    Args:
        band_specs (List[Tuple[float, float]]): List of tuples where each tuple contains
            the start (inclusive) and end (exclusive) frequencies of a band.

    Returns:
        List[float]: List of bandwidths for each band.
    """
    return [e - i for i, e in band_specs]


def check_nonzero_bandwidth(*, band_specs: List[Tuple[float, float]]) -> None:
    """
    Ensure that all bands have non-zero bandwidth.

    Args:
        band_specs (List[Tuple[float, float]]): List of tuples where each tuple contains
            the start (inclusive) and end (exclusive) frequencies of a band.

    Raises:
        ValueError: If any band has zero or negative bandwidth.
    """
    for start, end in band_specs:
        if end - start <= 0:
            raise ValueError(
                f"Band with start {start} and end {end} has zero or negative bandwidth."
            )


def check_no_overlap(*, band_specs: List[Tuple[float, float]]) -> None:
    """
    Ensure that bands do not overlap.

    Args:
        band_specs (List[Tuple[float, float]]): List of tuples where each tuple contains
            the start (inclusive) and end (exclusive) frequencies of a band.

    Raises:
        ValueError: If any two bands overlap.
    """
    for i in range(len(band_specs) - 1):
        _, end = band_specs[i]
        start, _ = band_specs[i + 1]
        if end > start:
            raise ValueError(
                f"Band with end {end} overlaps with band starting at {start}."
            )


def check_no_gap(*, band_specs: List[Tuple[float, float]]) -> None:
    """
    Ensure that there are no gaps between bands.

    Args:
        band_specs (List[Tuple[float, float]]): List of tuples where each tuple contains
            the start (inclusive) and end (exclusive) frequencies of a band.

    Raises:
        ValueError: If there are gaps between bands.
    """
    for i in range(len(band_specs) - 1):
        _, end = band_specs[i]
        start, _ = band_specs[i + 1]
        if end < start:
            raise ValueError(
                f"Gap detected between band ending at {end} and band starting at {start}."
            )


class BandsplitSpecification:
    """
    Base class for defining band split specifications.

    Args:
        n_fft (int): Number of FFT points.
        fs (int): Sampling frequency.
    """

    def __init__(self, *, n_fft: int, fs: int) -> None:
        self.fs = fs
        self.n_fft = n_fft
        self.nyquist = fs / 2
        self.max_index = n_fft // 2 + 1

    def index_to_hertz(self, index: int) -> float:
        """
        Convert an index to its corresponding frequency in Hertz.

        Args:
            index (int): Frequency index.

        Returns:
            float: Frequency in Hertz.
        """
        return index * self.fs / self.n_fft

    def hertz_to_index(
        self, *, hz: float, round: Literal["floor", "ceil", "nearest"] = "nearest"
    ) -> int:
        """
        Convert a frequency in Hertz to its corresponding index.

        Args:
            hz (float): Frequency in Hertz.
            round (Literal["floor", "ceil", "nearest"], optional): Method to round the index. Defaults to "nearest".

        Returns:
            int: Frequency index.
        """
        index = hz * self.n_fft / self.fs

        if round == "floor":
            index = int(np.floor(index))
        elif round == "ceil":
            index = int(np.ceil(index))
        else:
            index = int(np.round(index))

        return index

    def get_band_specs_with_bandwidth(
        self, *, start_index: int, end_index: int, bandwidth_hz: float
    ) -> List[Tuple[int, int]]:
        """
        Generate band specifications with a fixed bandwidth.

        Args:
            start_index (int): Starting index.
            end_index (int): Ending index.
            bandwidth_hz (float): Bandwidth in Hertz.

        Returns:
            List[Tuple[int, int]]: List of band specifications as (start, end) indices.
        """
        band_specs = []
        lower = start_index

        while lower < end_index:
            upper = int(lower + self.hertz_to_index(hz=bandwidth_hz, round="nearest"))
            upper = min(upper, end_index)

            band_specs.append((lower, upper))
            lower = upper

        return band_specs

    @abstractmethod
    def get_band_specs(self) -> List[Tuple[int, int]]:
        """
        Abstract method to get band specifications.

        Returns:
            List[Tuple[int, int]]: List of band specifications as [start, end) indices.
        """
        raise NotImplementedError


class HandcraftedBandsplitSpecification(BandsplitSpecification):
    """
    Handcrafted band split specification with predefined frequency splits.

    Args:
        n_fft (int): Number of FFT points.
        fs (int): Sampling frequency.
    """

    def __init__(self, *, n_fft: int, fs: int) -> None:
        super().__init__(n_fft=n_fft, fs=fs)

        self.split500 = self.hertz_to_index(hz=500)
        self.split1k = self.hertz_to_index(hz=1000)
        self.split2k = self.hertz_to_index(hz=2000)
        self.split4k = self.hertz_to_index(hz=4000)
        self.split8k = self.hertz_to_index(hz=8000)
        self.split16k = self.hertz_to_index(hz=16000)
        self.split20k = self.hertz_to_index(hz=20000)

        self.above20k = [(self.split20k, self.max_index)]
        self.above16k = [(self.split16k, self.split20k)] + self.above20k


class VocalBandsplitSpecification(BandsplitSpecification):
    """
    Band split specification for vocal frequencies.

    Args:
        n_fft (int): Number of FFT points.
        fs (int): Sampling frequency.
        version (str, optional): Version of the band split. Defaults to "7".
    """

    def __init__(self, *, n_fft: int, fs: int, version: str = "7") -> None:
        super().__init__(n_fft=n_fft, fs=fs)

        self.version = version

    def get_band_specs(self) -> List[Tuple[int, int]]:
        """
        Get band specifications for the vocal band split.

        Returns:
            List[Tuple[int, int]]: List of band specifications as (start, end) indices.
        """
        return getattr(self, f"version{self.version}")()

    @property
    def version1(self) -> List[Tuple[int, int]]:
        """
        Version 1 of the vocal band split.

        Returns:
            List[Tuple[int, int]]: List of band specifications as (start, end) indices.
        """
        return self.get_band_specs_with_bandwidth(
            start_index=0, end_index=self.max_index, bandwidth_hz=1000
        )

    def version2(self) -> List[Tuple[int, int]]:
        """
        Version 2 of the vocal band split.

        Returns:
            List[Tuple[int, int]]: List of band specifications as (start, end) indices.
        """
        below16k = self.get_band_specs_with_bandwidth(
            start_index=0, end_index=self.split16k, bandwidth_hz=1000
        )
        below20k = self.get_band_specs_with_bandwidth(
            start_index=self.split16k, end_index=self.split20k, bandwidth_hz=2000
        )

        return below16k + below20k + self.above20k

    def version3(self) -> List[Tuple[int, int]]:
        """
        Version 3 of the vocal band split.

        Returns:
            List[Tuple[int, int]]: List of band specifications as (start, end) indices.
        """
        below8k = self.get_band_specs_with_bandwidth(
            start_index=0, end_index=self.split8k, bandwidth_hz=1000
        )
        below16k = self.get_band_specs_with_bandwidth(
            start_index=self.split8k, end_index=self.split16k, bandwidth_hz=2000
        )

        return below8k + below16k + self.above16k

    def version4(self) -> List[Tuple[int, int]]:
        """
        Version 4 of the vocal band split.

        Returns:
            List[Tuple[int, int]]: List of band specifications as (start, end) indices.
        """
        below1k = self.get_band_specs_with_bandwidth(
            start_index=0, end_index=self.split1k, bandwidth_hz=100
        )
        below8k = self.get_band_specs_with_bandwidth(
            start_index=self.split1k, end_index=self.split8k, bandwidth_hz=1000
        )
        below16k = self.get_band_specs_with_bandwidth(
            start_index=self.split8k, end_index=self.split16k, bandwidth_hz=2000
        )

        return below1k + below8k + below16k + self.above16k

    def version5(self) -> List[Tuple[int, int]]:
        """
        Version 5 of the vocal band split.

        Returns:
            List[Tuple[int, int]]: List of band specifications as (start, end) indices.
        """
        below1k = self.get_band_specs_with_bandwidth(
            start_index=0, end_index=self.split1k, bandwidth_hz=100
        )
        below16k = self.get_band_specs_with_bandwidth(
            start_index=self.split1k, end_index=self.split16k, bandwidth_hz=1000
        )
        below20k = self.get_band_specs_with_bandwidth(
            start_index=self.split16k, end_index=self.split20k, bandwidth_hz=2000
        )
        return below1k + below16k + below20k + self.above20k

    def version6(self) -> List[Tuple[int, int]]:
        """
        Version 6 of the vocal band split.

        Returns:
            List[Tuple[int, int]]: List of band specifications as (start, end) indices.
        """
        below1k = self.get_band_specs_with_bandwidth(
            start_index=0, end_index=self.split1k, bandwidth_hz=100
        )
        below4k = self.get_band_specs_with_bandwidth(
            start_index=self.split1k, end_index=self.split4k, bandwidth_hz=500
        )
        below8k = self.get_band_specs_with_bandwidth(
            start_index=self.split4k, end_index=self.split8k, bandwidth_hz=1000
        )
        below16k = self.get_band_specs_with_bandwidth(
            start_index=self.split8k, end_index=self.split16k, bandwidth_hz=2000
        )
        return below1k + below4k + below8k + below16k + self.above16k

    def version7(self) -> List[Tuple[int, int]]:
        """
        Version 7 of the vocal band split.

        Returns:
            List[Tuple[int, int]]: List of band specifications as (start, end) indices.
        """
        below1k = self.get_band_specs_with_bandwidth(
            start_index=0, end_index=self.split1k, bandwidth_hz=100
        )
        below4k = self.get_band_specs_with_bandwidth(
            start_index=self.split1k, end_index=self.split4k, bandwidth_hz=250
        )
        below8k = self.get_band_specs_with_bandwidth(
            start_index=self.split4k, end_index=self.split8k, bandwidth_hz=500
        )
        below16k = self.get_band_specs_with_bandwidth(
            start_index=self.split8k, end_index=self.split16k, bandwidth_hz=1000
        )
        below20k = self.get_band_specs_with_bandwidth(
            start_index=self.split16k, end_index=self.split20k, bandwidth_hz=2000
        )
        return below1k + below4k + below8k + below16k + below20k + self.above20k


class OtherBandsplitSpecification(VocalBandsplitSpecification):
    """
    Band split specification for other frequencies.

    Args:
        n_fft (int): Number of FFT points.
        fs (int): Sampling frequency.
    """

    def __init__(self, *, n_fft: int, fs: int) -> None:
        super().__init__(n_fft=n_fft, fs=fs, version="7")


class BassBandsplitSpecification(BandsplitSpecification):
    """
    Band split specification for bass frequencies.

    Args:
        n_fft (int): Number of FFT points.
        fs (int): Sampling frequency.
    """

    def __init__(self, *, n_fft: int, fs: int) -> None:
        super().__init__(n_fft=n_fft, fs=fs)

    def get_band_specs(self) -> List[Tuple[int, int]]:
        """
        Get band specifications for the bass band split.

        Returns:
            List[Tuple[int, int]]: List of band specifications as (start, end) indices.
        """
        below500 = self.get_band_specs_with_bandwidth(
            start_index=0, end_index=self.split500, bandwidth_hz=50
        )
        below1k = self.get_band_specs_with_bandwidth(
            start_index=self.split500, end_index=self.split1k, bandwidth_hz=100
        )
        below4k = self.get_band_specs_with_bandwidth(
            start_index=self.split1k, end_index=self.split4k, bandwidth_hz=500
        )
        below8k = self.get_band_specs_with_bandwidth(
            start_index=self.split4k, end_index=self.split8k, bandwidth_hz=1000
        )
        below16k = self.get_band_specs_with_bandwidth(
            start_index=self.split8k, end_index=self.split16k, bandwidth_hz=2000
        )
        above16k = [(self.split16k, self.max_index)]

        return below500 + below1k + below4k + below8k + below16k + above16k


class DrumBandsplitSpecification(BandsplitSpecification):
    """
    Band split specification for drum frequencies.

    Args:
        n_fft (int): Number of FFT points.
        fs (int): Sampling frequency.
    """

    def __init__(self, *, n_fft: int, fs: int) -> None:
        super().__init__(n_fft=n_fft, fs=fs)

    def get_band_specs(self) -> List[Tuple[int, int]]:
        """
        Get band specifications for the drum band split.

        Returns:
            List[Tuple[int, int]]: List of band specifications as (start, end) indices.
        """
        below1k = self.get_band_specs_with_bandwidth(
            start_index=0, end_index=self.split1k, bandwidth_hz=50
        )
        below2k = self.get_band_specs_with_bandwidth(
            start_index=self.split1k, end_index=self.split2k, bandwidth_hz=100
        )
        below4k = self.get_band_specs_with_bandwidth(
            start_index=self.split2k, end_index=self.split4k, bandwidth_hz=250
        )
        below8k = self.get_band_specs_with_bandwidth(
            start_index=self.split4k, end_index=self.split8k, bandwidth_hz=500
        )
        below16k = self.get_band_specs_with_bandwidth(
            start_index=self.split8k, end_index=self.split16k, bandwidth_hz=1000
        )
        above16k = [(self.split16k, self.max_index)]

        return below1k + below2k + below4k + below8k + below16k + above16k


class PerceptualBandsplitSpecification(BandsplitSpecification):
    """
    Band split specification based on perceptual filterbanks.

    Args:
        n_fft (int): Number of FFT points.
        fs (int): Sampling frequency.
        fbank_fn (Callable): Function to generate the filterbank.
        n_bands (int): Number of bands.
        f_min (float, optional): Minimum frequency. Defaults to 0.0.
        f_max (float, optional): Maximum frequency. Defaults to None.
    """

    def __init__(
        self,
        *,
        n_fft: int,
        fs: int,
        fbank_fn: Callable[[int, int, float, float, int], torch.Tensor],
        n_bands: int,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ) -> None:
        super().__init__(n_fft=n_fft, fs=fs)
        self.n_bands = n_bands
        if f_max is None:
            f_max = fs / 2

        self.filterbank = fbank_fn(
            n_bands=n_bands, fs=fs, f_min=f_min, f_max=f_max, n_freqs=self.max_index
        )

        weight_per_bin = torch.sum(self.filterbank, dim=0, keepdim=True)  # (1, n_freqs)
        normalized_mel_fb = self.filterbank / weight_per_bin  # (n_mels, n_freqs)

        freq_weights = []
        band_specs = []
        for i in range(self.n_bands):
            active_bins = torch.nonzero(self.filterbank[i, :]).squeeze().tolist()
            if isinstance(active_bins, int):
                active_bins = (active_bins, active_bins)
            if len(active_bins) == 0:
                continue
            start_index = active_bins[0]
            end_index = active_bins[-1] + 1
            band_specs.append((start_index, end_index))
            freq_weights.append(normalized_mel_fb[i, start_index:end_index])

        self.freq_weights = freq_weights
        self.band_specs = band_specs

    def get_band_specs(self) -> List[Tuple[int, int]]:
        """
        Get band specifications for the perceptual band split.

        Returns:
            List[Tuple[int, int]]: List of band specifications as (start, end) indices.
        """
        return self.band_specs

    def get_freq_weights(self) -> List[torch.Tensor]:
        """
        Get frequency weights for the perceptual band split.

        Returns:
            List[torch.Tensor]: List of frequency weight tensors.
        """
        return self.freq_weights

    def save_to_file(self, *, dir_path: str) -> None:
        """
        Save the band split specification to a file.

        Args:
            dir_path (str): Directory path to save the file.
        """
        os.makedirs(dir_path, exist_ok=True)

        import pickle

        with open(os.path.join(dir_path, "mel_bandsplit_spec.pkl"), "wb") as f:
            pickle.dump(
                {
                    "band_specs": self.band_specs,
                    "freq_weights": self.freq_weights,
                    "filterbank": self.filterbank,
                },
                f,
            )


def mel_filterbank(
    *, n_bands: int, fs: int, f_min: float, f_max: float, n_freqs: int
) -> torch.Tensor:
    """
    Generate a Mel filterbank.

    Args:
        n_bands (int): Number of Mel bands.
        fs (int): Sampling frequency.
        f_min (float): Minimum frequency.
        f_max (float): Maximum frequency.
        n_freqs (int): Number of frequency bins.

    Returns:
        torch.Tensor: Mel filterbank tensor.
    """
    fb = taF.melscale_fbanks(
        n_mels=n_bands,
        sample_rate=fs,
        f_min=f_min,
        f_max=f_max,
        n_freqs=n_freqs,
    ).T

    fb[0, 0] = 1.0

    return fb


class MelBandsplitSpecification(PerceptualBandsplitSpecification):
    """
    Band split specification based on Mel filterbanks.

    Args:
        n_fft (int): Number of FFT points.
        fs (int): Sampling frequency.
        n_bands (int): Number of bands.
        f_min (float, optional): Minimum frequency. Defaults to 0.0.
        f_max (float, optional): Maximum frequency. Defaults to None.
    """

    def __init__(
        self,
        *,
        n_fft: int,
        fs: int,
        n_bands: int,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ) -> None:
        super().__init__(
            fbank_fn=mel_filterbank,
            n_fft=n_fft,
            fs=fs,
            n_bands=n_bands,
            f_min=f_min,
            f_max=f_max,
        )


class MusicalBandsplitSpecification(PerceptualBandsplitSpecification):
    """
    Band split specification based on musical filterbanks.

    Args:
        n_fft (int): Number of FFT points.
        fs (int): Sampling frequency.
        n_bands (int): Number of bands.
        f_min (float, optional): Minimum frequency. Defaults to 0.0.
        f_max (float, optional): Maximum frequency. Defaults to None.
    """

    def __init__(
        self,
        *,
        n_fft: int,
        fs: int,
        n_bands: int,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ) -> None:
        super().__init__(
            fbank_fn=musical_filterbank,
            n_fft=n_fft,
            fs=fs,
            n_bands=n_bands,
            f_min=f_min,
            f_max=f_max,
        )


class BarkBandsplitSpecification(PerceptualBandsplitSpecification):
    """
    Band split specification based on Bark filterbanks.

    Args:
        n_fft (int): Number of FFT points.
        fs (int): Sampling frequency.
        n_bands (int): Number of bands.
        f_min (float, optional): Minimum frequency. Defaults to 0.0.
        f_max (float, optional): Maximum frequency. Defaults to None.
    """

    def __init__(
        self,
        *,
        n_fft: int,
        fs: int,
        n_bands: int,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ) -> None:
        super().__init__(
            fbank_fn=bark_filterbank,
            n_fft=n_fft,
            fs=fs,
            n_bands=n_bands,
            f_min=f_min,
            f_max=f_max,
        )


class TriangularBarkBandsplitSpecification(PerceptualBandsplitSpecification):
    """
    Band split specification based on triangular Bark filterbanks.

    Args:
        n_fft (int): Number of FFT points.
        fs (int): Sampling frequency.
        n_bands (int): Number of bands.
        f_min (float, optional): Minimum frequency. Defaults to 0.0.
        f_max (float, optional): Maximum frequency. Defaults to None.
    """

    def __init__(
        self,
        *,
        n_fft: int,
        fs: int,
        n_bands: int,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ) -> None:
        super().__init__(
            fbank_fn=triangular_bark_filterbank,
            n_fft=n_fft,
            fs=fs,
            n_bands=n_bands,
            f_min=f_min,
            f_max=f_max,
        )


class MiniBarkBandsplitSpecification(PerceptualBandsplitSpecification):
    """
    Band split specification based on mini Bark filterbanks.

    Args:
        n_fft (int): Number of FFT points.
        fs (int): Sampling frequency.
        n_bands (int): Number of bands.
        f_min (float, optional): Minimum frequency. Defaults to 0.0.
        f_max (float, optional): Maximum frequency. Defaults to None.
    """

    def __init__(
        self,
        *,
        n_fft: int,
        fs: int,
        n_bands: int,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ) -> None:
        super().__init__(
            fbank_fn=minibark_filterbank,
            n_fft=n_fft,
            fs=fs,
            n_bands=n_bands,
            f_min=f_min,
            f_max=f_max,
        )


class EquivalentRectangularBandsplitSpecification(PerceptualBandsplitSpecification):
    """
    Band split specification based on Equivalent Rectangular Bandwidth (ERB) filterbanks.

    Args:
        n_fft (int): Number of FFT points.
        fs (int): Sampling frequency.
        n_bands (int): Number of bands.
        f_min (float, optional): Minimum frequency. Defaults to 0.0.
        f_max (float, optional): Maximum frequency. Defaults to None.
    """

    def __init__(
        self,
        *,
        n_fft: int,
        fs: int,
        n_bands: int,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ) -> None:
        super().__init__(
            fbank_fn=erb_filterbank,
            n_fft=n_fft,
            fs=fs,
            n_bands=n_bands,
            f_min=f_min,
            f_max=f_max,
        )
