import numpy as np
from spafe.fbanks import bark_fbanks
from spafe.utils.converters import hz2bark, hz2erb
import torch
from librosa import hz_to_midi, midi_to_hz
from torchaudio.functional.functional import _create_triangular_filterbank


def musical_filterbank(
    *,
    n_bands: int,
    fs: int,
    f_min: float,
    f_max: float,
    n_freqs: int,
    octave_per_band_mult: float = 1.0,
) -> torch.Tensor:
    """
    Generate a musical filterbank.

    Args:
        n_bands (int): Number of musical bands.
        fs (int): Sampling frequency.
        f_min (float): Minimum frequency.
        f_max (float): Maximum frequency.
        n_freqs (int): Number of frequency bins.

    Returns:
        torch.Tensor: Filterbank tensor of shape (n_bands, n_freqs).
    """
    nfft = 2 * (n_freqs - 1)
    df = fs / nfft
    f_max = f_max or fs / 2
    f_min = f_min or fs / nfft

    assert octave_per_band_mult >= 0.5, "octave_per_band_mult must be >= 0.5"

    n_octaves = np.log2(f_max / f_min)
    n_octaves_per_band = n_octaves / n_bands
    bandwidth_mult = np.power(2.0, n_octaves_per_band * octave_per_band_mult)

    low_midi = max(0, hz_to_midi(f_min))
    high_midi = hz_to_midi(f_max)
    midi_points = np.linspace(low_midi, high_midi, n_bands)
    hz_pts = midi_to_hz(midi_points)

    low_pts = hz_pts / bandwidth_mult
    high_pts = hz_pts * bandwidth_mult

    low_bins = np.floor(low_pts / df).astype(int)
    high_bins = np.ceil(high_pts / df).astype(int)

    fb = np.zeros((n_bands, n_freqs))

    for i in range(n_bands):
        fb[i, low_bins[i] : high_bins[i] + 1] = 1.0

    fb[0, : low_bins[0]] = 1.0
    fb[-1, high_bins[-1] + 1 :] = 1.0

    return torch.as_tensor(fb)


def bark_filterbank(
    *, n_bands: int, fs: int, f_min: float, f_max: float, n_freqs: int
) -> torch.Tensor:
    """
    Generate a Bark filterbank.

    Args:
        n_bands (int): Number of Bark bands.
        fs (int): Sampling frequency.
        f_min (float): Minimum frequency.
        f_max (float): Maximum frequency.
        n_freqs (int): Number of frequency bins.

    Returns:
        torch.Tensor: Filterbank tensor of shape (n_bands, n_freqs).
    """
    nfft = 2 * (n_freqs - 1)
    fb, _ = bark_fbanks.bark_filter_banks(
        nfilts=n_bands,
        nfft=nfft,
        fs=fs,
        low_freq=f_min,
        high_freq=f_max,
        scale="constant",
    )

    return torch.as_tensor(fb)


def triangular_bark_filterbank(
    *, n_bands: int, fs: int, f_min: float, f_max: float, n_freqs: int
) -> torch.Tensor:
    """
    Generate a triangular Bark filterbank.

    Args:
        n_bands (int): Number of Bark bands.
        fs (int): Sampling frequency.
        f_min (float): Minimum frequency.
        f_max (float): Maximum frequency.
        n_freqs (int): Number of frequency bins.

    Returns:
        torch.Tensor: Filterbank tensor of shape (n_bands, n_freqs).
    """
    all_freqs = torch.linspace(0, fs // 2, n_freqs)

    # calculate mel freq bins
    m_min = hz2bark(f_min)
    m_max = hz2bark(f_max)

    m_pts = torch.linspace(m_min, m_max, n_bands + 2)
    f_pts = 600 * torch.sinh(m_pts / 6)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    fb = fb.T

    first_active_band = torch.nonzero(torch.sum(fb, dim=-1))[0, 0]
    first_active_bin = torch.nonzero(fb[first_active_band, :])[0, 0]

    fb[first_active_band, :first_active_bin] = 1.0

    return fb


def minibark_filterbank(
    *, n_bands: int, fs: int, f_min: float, f_max: float, n_freqs: int
) -> torch.Tensor:
    """
    Generate a mini Bark filterbank.

    Args:
        n_bands (int): Number of Bark bands.
        fs (int): Sampling frequency.
        f_min (float): Minimum frequency.
        f_max (float): Maximum frequency.
        n_freqs (int): Number of frequency bins.

    Returns:
        torch.Tensor: Filterbank tensor of shape (n_bands, n_freqs).
    """
    fb = bark_filterbank(
        n_bands=n_bands, fs=fs, f_min=f_min, f_max=f_max, n_freqs=n_freqs
    )

    fb[fb < np.sqrt(0.5)] = 0.0

    return fb


def erb_filterbank(
    *, n_bands: int, fs: int, f_min: float, f_max: float, n_freqs: int
) -> torch.Tensor:
    """
    Generate an Equivalent Rectangular Bandwidth (ERB) filterbank.

    Args:
        n_bands (int): Number of ERB bands.
        fs (int): Sampling frequency.
        f_min (float): Minimum frequency.
        f_max (float): Maximum frequency.
        n_freqs (int): Number of frequency bins.

    Returns:
        torch.Tensor: Filterbank tensor of shape (n_bands, n_freqs).
    """
    # freq bins
    A = (1000 * np.log(10)) / (24.7 * 4.37)
    all_freqs = torch.linspace(0, fs // 2, n_freqs)

    # calculate mel freq bins
    m_min = hz2erb(f_min)
    m_max = hz2erb(f_max)

    m_pts = torch.linspace(m_min, m_max, n_bands + 2)
    f_pts = (torch.pow(10, (m_pts / A)) - 1) / 0.00437

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    fb = fb.T

    first_active_band = torch.nonzero(torch.sum(fb, dim=-1))[0, 0]
    first_active_bin = torch.nonzero(fb[first_active_band, :])[0, 0]

    fb[first_active_band, :first_active_bin] = 1.0

    return fb
