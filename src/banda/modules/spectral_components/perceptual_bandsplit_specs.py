import torch
import torch.nn.functional as F
import torchaudio.functional as taF
from typing import Any, Callable, Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import structlog

logger = structlog.get_logger(__name__)

from banda.models.common_components.configs.bandsplit_configs._bandsplit_models import (
    PerceptualBandsplitSpecsConfig,
    MusicalBandsplitSpecsConfig,
    MelBandsplitSpecsConfig,
    TriangularBarkBandsplitSpecsConfig,
    EquivalentRectangularBandsplitSpecsConfig,
    BandsplitType,
)
from banda.models.common_components.spectral_components.spectral_base import (
    BandsplitSpecification,
    bark_to_hz,
    hz_to_midi,
    midi_to_hz,
    hz_to_bark,
    hz_to_erb,
    _create_triangular_filterbank,
)
from banda.models.common_components.spectral_components.bandsplit_registry import bandsplit_registry

class PerceptualBandsplitSpecification(BandsplitSpecification):
    def __init__(
            self,
            nfft: int,
            fs: int,
            config: PerceptualBandsplitSpecsConfig,
            drop_dc_band: bool = False,
    ) -> None:
        # Initialize BandsplitSpecification with drop_dc_band=False for initial filterbank creation
        # We will handle dropping the DC band manually after filterbank creation
        super().__init__(nfft=nfft, fs=fs, drop_dc_band=False) 
        self.config = config
        self.n_bands = config.n_bands
        self._drop_dc_band_actual = drop_dc_band # Store the actual drop_dc_band setting

        # f_min and f_max from config are not used for band definition,
        # as bands should cover the entire spectrogram.
        # They might be used for other purposes or removed later.
        self.f_min_config = config.f_min if config.f_min is not None else 0.0
        self.f_max_config = config.f_max if config.f_max is not None else fs / 2

        logger.debug(f"PerceptualBandsplitSpecification: Initial f_min_config: {self.f_min_config}, f_max_config: {self.f_max_config}")

        # Define the actual frequency range for band generation based on spectrogram
        # Always use 0 Hz to Nyquist for initial filterbank creation
        self.min_freq_for_bands = 0.0
        self.max_freq_for_bands = self.nyquist
        
        logger.debug(f"PerceptualBandsplitSpecification: Actual band generation range: min_freq={self.min_freq_for_bands}, max_freq={self.max_freq_for_bands}")

        hz_points = self._get_hz_points()
        logger.debug(f"PerceptualBandsplitSpecification: Generated hz_points: {hz_points}")

        # Always create filterbank for nfft // 2 + 1 frequency bins (including DC)
        n_freqs_for_fbank = nfft // 2 + 1
        all_freqs_for_fbank = torch.arange(0, nfft//2 + 1) * (fs / nfft)
        logger.debug(f"PerceptualBandsplitSpecification: n_freqs_for_fbank: {n_freqs_for_fbank}, all_freqs_for_fbank shape: {all_freqs_for_fbank.shape}")

        fbank_fn = self._get_fbank_fn()
        self.filterbank = fbank_fn(all_freqs_for_fbank, hz_points, self.n_bands, n_freqs_for_fbank, self.fs)

        # If drop_dc_band is true, remove the DC band from the filterbank
        if self._drop_dc_band_actual:
            self.filterbank = self.filterbank[:, 1:] # Remove the first column (DC band)
            # Update max_index to reflect the dropped DC band
            self.max_index = nfft // 2
            logger.debug(f"PerceptualBandsplitSpecification: Dropped DC band. New filterbank shape: {self.filterbank.shape}, new max_index: {self.max_index}")


        epsilon = 1e-10
        self.filterbank = self.filterbank / (self.filterbank.sum(axis=0, keepdims=True) + epsilon)

        logger.debug(
            f"PerceptualBandsplitSpecification: Filterbank shape: {self.filterbank.shape}, max_index: {self.max_index}, epsilon: {epsilon}"
        )

        band_specs = []
        for i in range(self.n_bands):
            
            # find the first and last index with nonzero values in the filterbank
            fbank = self.filterbank[i, :]

            active_bins = torch.nonzero(fbank).flatten()
            if len(active_bins) == 0:
                logger.debug(f"PerceptualBandsplitSpecification: Band {i} has no active bins after filterbank creation. Skipping this band.")
                continue
            start_index = active_bins[0]
            end_index = active_bins[-1]

            start_index = max(0, start_index)
            end_index = min(end_index + 1, self.max_index)
            
            if start_index >= end_index:
                logger.warning(f"PerceptualBandsplitSpecification: Calculated band ({start_index}, {end_index}) has zero or negative bandwidth. Skipping.")
                continue
            band_specs.append((start_index, end_index))

        logger.debug(f"Bandspecs: {len(band_specs)}")

        band_specs.sort(key=lambda x: x[0])
        self.band_specs = band_specs

        # Explicit validation check for the number of bands
        if len(self.band_specs) != self.n_bands:
            raise ValueError(f"Expected {self.n_bands} bands, but got {len(self.band_specs)} bands after processing. This indicates an issue with band generation or merging.")


    def get_band_specs(self) -> List[Tuple[int, int]]:
        self._validate_band_specs(self.band_specs)
        total_bandwidth = sum([fend - fstart for fstart, fend in self.band_specs])
        logger.debug(f"PerceptualBandsplitSpecification: Total bandwidth of band_specs: {total_bandwidth}")
        return self.band_specs

    @property
    def is_overlapping(self) -> bool:
        return True

    def get_freq_weights(self) -> torch.Tensor:
        """
        Returns the filterbank weights for each frequency bin.
        Shape: (n_bands, n_freq_bins)
        """
        return self.filterbank


    @abstractmethod
    def _get_fbank_fn(self) -> Callable[[torch.Tensor, torch.Tensor, int, int, int], torch.Tensor]: # Added fs to signature
        raise NotImplementedError

    @abstractmethod
    def _get_hz_points(self) -> torch.Tensor:
        raise NotImplementedError


def musical_filterbank(n_bands: int, fs: int, f_min: float, f_max: float, n_freqs: int): # Corrected signature
    """
    Creates a musical filterbank using the bandwidth method.
    """
    nfft = 2 * (n_freqs - 1) # n_freqs here is nfft // 2 + 1
    df = fs / nfft
    
    # f_max and f_min are now passed from config
    f_min_for_octaves = fs / nfft # Lowest non-zero frequency for octave calculation

    n_octaves = torch.log2(torch.tensor(f_max / f_min_for_octaves))
    n_octaves_per_band = n_octaves / n_bands
    bandwidth_mult = torch.pow(2.0, n_octaves_per_band)

    low_midi = torch.max(torch.tensor(0.0), hz_to_midi(torch.tensor(f_min))).item()
    high_midi = hz_to_midi(torch.tensor(f_max)).item()
    midi_points = torch.linspace(low_midi, high_midi, n_bands) # n_bands points
    hz_pts = midi_to_hz(midi_points)

    low_pts = hz_pts / bandwidth_mult
    high_pts = hz_pts * bandwidth_mult # Corrected: high_pts = hz_pts * bandwidth_mult

    low_bins = torch.floor(low_pts / df).int()
    high_bins = torch.ceil(high_pts / df).int()

    fb = torch.zeros((n_bands, n_freqs))

    for i in range(n_bands):
        # Ensure indices are within bounds
        start_idx = torch.clamp(low_bins[i], 0, n_freqs - 1)
        end_idx = torch.clamp(high_bins[i], 0, n_freqs - 1)
        fb[i, start_idx:end_idx+1] = 1.0

    # Explicitly cover from 0 Hz for the first band
    if n_bands > 0:
        fb[0, :low_bins[0]] = 1.0
    
    # Explicitly cover up to Nyquist for the last band
    if n_bands > 0:
        fb[n_bands - 1, high_bins[-1]+1:] = 1.0

    return fb


class MusicalBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(
            self,
            nfft: int,
            fs: int,
            config: MusicalBandsplitSpecsConfig,
            drop_dc_band: bool = False,
    ) -> None:
        super().__init__(nfft=nfft, fs=fs, config=config, drop_dc_band=drop_dc_band)
        # Removed f_min adjustment, as bands should cover the entire spectrogram.

    def _get_fbank_fn(self) -> Callable[[torch.Tensor, torch.Tensor, int, int, int], torch.Tensor]:
        # The musical_filterbank from utils.py does not use all_freqs or hz_points directly.
        # It uses n_bands, fs, f_min, f_max, n_freqs.
        # We need to adapt the signature.
        def fbank_wrapper(all_freqs: torch.Tensor, hz_points: torch.Tensor, n_bands: int, n_freqs: int, fs: int):
            # Pass the relevant parameters to the musical_filterbank
            return musical_filterbank(n_bands, fs, self.f_min_config, self.f_max_config, n_freqs)
        return fbank_wrapper

    def _get_hz_points(self) -> torch.Tensor:
        # This method is not directly used by the musical_filterbank,
        # but is required by the abstract base class. Return a dummy tensor.
        return torch.empty(0)


def mel_scale_to_hz(mel_points: torch.Tensor) -> torch.Tensor:
    """
    Converts Mel scale points to Hz.
    """
    return 700 * (torch.exp(mel_points / 1127) - 1)

def hz_to_mel_scale(hz: torch.Tensor) -> torch.Tensor:
    """
    Converts Hz to Mel scale points.
    """
    return 1127 * torch.log(1 + hz / 700)

def torchaudio_mel_filterbank(all_freqs: torch.Tensor, hz_points: torch.Tensor, n_bands: int, n_freqs: int, fs: int):
    """
    Creates a Mel filterbank using torchaudio.functional.melscale_fbanks.
    """
    # The hz_points and all_freqs are not directly used by taF.melscale_fbanks,
    # as it generates its own frequency points based on n_mels, sample_rate, f_min, f_max, n_freqs.
    # We need to ensure f_min and f_max passed to taF.melscale_fbanks are consistent with the overall band generation range.
    
    # Always use 0 Hz and Nyquist for torchaudio's f_min and f_max
    f_min_for_ta = 0.0
    f_max_for_ta = fs / 2

    fb = taF.melscale_fbanks(
                n_mels=n_bands,
                sample_rate=fs, # Use fs from BandsplitSpecification
                f_min=f_min_for_ta,
                f_max=f_max_for_ta,
                n_freqs=n_freqs, # This is now nfft // 2 + 1
        ).T

    # Ensure the first bin of the first band is active, as seen in utils.py
    if fb.shape[0] > 0 and fb.shape[1] > 0:
        fb[0, 0] = 1.0

    return fb


class MelBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(
            self,
            nfft: int,
            fs: int,
            config: MelBandsplitSpecsConfig,
            drop_dc_band: bool = False,
    ) -> None:
        super().__init__(nfft=nfft, fs=fs, config=config, drop_dc_band=drop_dc_band)

    def _get_fbank_fn(self) -> Callable[[torch.Tensor, torch.Tensor, int, int, int], torch.Tensor]:
        return torchaudio_mel_filterbank

    def _get_hz_points(self) -> torch.Tensor:
        # For torchaudio_mel_filterbank, hz_points are not directly used for filterbank creation,
        # but they are still part of the PerceptualBandsplitSpecification interface.
        # We can generate them based on the Mel scale for consistency, even if not directly used by the fbank_fn.
        mel_min = hz_to_mel_scale(torch.tensor(self.min_freq_for_bands)).item()
        mel_max = hz_to_mel_scale(torch.tensor(self.max_freq_for_bands)).item()
        logger.debug(f"MelBandsplitSpecification: mel_min: {mel_min}, mel_max: {mel_max}")
        mel_pts = torch.linspace(mel_min, mel_max, self.n_bands + 2)
        logger.debug(f"MelBandsplitSpecification: mel_pts: {mel_pts}")
        hz_points = mel_scale_to_hz(mel_pts)
        logger.debug(f"MelBandsplitSpecification: hz_points: {hz_points}") # Added debug log
        return hz_points


def triangular_bark_filterbank(all_freqs: torch.Tensor, hz_points: torch.Tensor, n_bands: int, n_freqs: int, fs: int):
    logger.debug(f"triangular_bark_filterbank: all_freqs min: {all_freqs.min()}, max: {all_freqs.max()}, shape: {all_freqs.shape}")
    logger.debug(f"triangular_bark_filterbank: hz_points min: {hz_points.min()}, max: {hz_points.max()}, shape: {hz_points.shape}")
    logger.debug(f"triangular_bark_filterbank: n_bands: {n_bands}")

    m_min = hz_to_bark(torch.tensor(0.0))
    m_max = hz_to_bark(torch.tensor(fs/2.0))

    logger.debug(f"triangular_bark_filterbank: m_min: {m_min}, m_max: {m_max}")

    m_pts = torch.linspace(m_min, m_max, n_bands + 2)
    f_pts = bark_to_hz(m_pts)

    logger.debug(f"triangular_bark_filterbank: m_pts: {m_pts}")
    logger.debug(f"triangular_bark_filterbank: f_pts: {f_pts}")

    fb = _create_triangular_filterbank(all_freqs, f_pts)
    logger.debug(f"triangular_bark_filterbank: initial filterbank shape: {fb.shape}, sum: {fb.sum()}")

    return fb


class TriangularBarkBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(
            self,
            nfft: int,
            fs: int,
            config: TriangularBarkBandsplitSpecsConfig,
            drop_dc_band: bool = False,
    ) -> None:
        super().__init__(nfft=nfft, fs=fs, config=config, drop_dc_band=drop_dc_band)

    def _get_fbank_fn(self) -> Callable[[torch.Tensor, torch.Tensor, int, int, int], torch.Tensor]:
        return triangular_bark_filterbank

    def _get_hz_points(self) -> torch.Tensor:
        logger.debug(f"TriangularBarkBandsplitSpecification: f_min: {self.f_min_config}, f_max: {self.f_max_config}")
        m_min = hz_to_bark(torch.tensor(self.min_freq_for_bands)).item()
        m_max = hz_to_bark(torch.tensor(self.max_freq_for_bands)).item()
        m_pts = torch.linspace(m_min, m_max, self.n_bands + 2)
        f_pts = bark_to_hz(m_pts)
        print(f_pts)
        logger.debug(f"TriangularBarkBandsplitSpecification: generated f_pts: {f_pts}")
        return f_pts


def erb_filterbank(all_freqs: torch.Tensor, hz_points: torch.Tensor, n_bands: int, n_freqs: int, fs: int) -> torch.Tensor:
    logger.debug(f"erb_filterbank: all_freqs min: {all_freqs.min()}, max: {all_freqs.max()}, shape: {all_freqs.shape}")
    logger.debug(f"erb_filterbank: hz_points min: {hz_points.min()}, max: {hz_points.max()}, shape: {hz_points.shape}")
    logger.debug(f"erb_filterbank: n_bands: {n_bands}")

    A = (1000 * torch.log(torch.tensor(10))) / (24.7 * 4.37)

    m_min = hz_to_erb(all_freqs.min()).item()
    m_max = hz_to_erb(all_freqs.max()).item()

    logger.debug(f"erb_filterbank: m_min: {m_min}, m_max: {m_max}")

    m_pts = torch.linspace(m_min, m_max, n_bands + 2)
    f_pts = (torch.pow(torch.tensor(10), (m_pts / A)) - 1)/ 0.00437
    logger.debug(f"erb_filterbank: m_pts: {m_pts}")
    logger.debug(f"erb_pts: {f_pts}")

    fb = _create_triangular_filterbank(all_freqs, f_pts)
    logger.debug(f"erb_filterbank: initial filterbank shape: {fb.shape}, sum: {fb.sum()}")
    
    # The logic to extend the first and last bands is now in _create_triangular_filterbank
    # No need to duplicate it here.

    return fb


class EquivalentRectangularBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(
            self,
            nfft: int,
            fs: int,
            config: EquivalentRectangularBandsplitSpecsConfig,
            drop_dc_band: bool = False,
    ) -> None:
        super().__init__(nfft=nfft, fs=fs, config=config, drop_dc_band=drop_dc_band)

    def _get_fbank_fn(self) -> Callable[[torch.Tensor, torch.Tensor, int, int, int], torch.Tensor]:
        return erb_filterbank

    def _get_hz_points(self) -> torch.Tensor:
        logger.debug(f"EquivalentRectangularBandsplitSpecification: f_min: {self.f_min_config}, f_max: {self.f_max_config}")
        A = (1000 * torch.log(torch.tensor(10))) / (24.7 * 4.37)
        m_min = hz_to_erb(torch.tensor(self.min_freq_for_bands)).item()
        m_max = hz_to_erb(torch.tensor(self.max_freq_for_bands)).item()
        m_pts = torch.linspace(m_min, m_max, self.n_bands + 2)
        f_pts = (torch.pow(torch.tensor(10), (m_pts / A)) - 1)/ 0.00437
        logger.debug(f"EquivalentRectangularBandsplitSpecification: generated f_pts: {f_pts}")
        return f_pts

def register_perceptual_bandsplit_specs():
    bandsplit_registry.register(BandsplitType.MUSICAL, MusicalBandsplitSpecification)
    bandsplit_registry.register(BandsplitType.MUSIC, MusicalBandsplitSpecification)
    bandsplit_registry.register(BandsplitType.MEL, MelBandsplitSpecification)
    bandsplit_registry.register(BandsplitType.TRIBARK, TriangularBarkBandsplitSpecification)
    bandsplit_registry.register(BandsplitType.ERB, EquivalentRectangularBandsplitSpecification)