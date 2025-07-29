#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#


import torch
from typing import List, Tuple

def band_widths_from_specs(band_specs: List[Tuple[float, float]]) -> List[int]:
    """
    Calculates the bandwidths from a list of band specifications.

    Args:
        band_specs (List[Tuple[float, float]]): A list of tuples, where each tuple
                                                 represents a frequency band [fstart, fend).

    Returns:
        List[int]: A list of bandwidths (fend - fstart) for each band.
    """
    return [int(fend - fstart) for fstart, fend in band_specs]

def check_no_gap(band_specs: List[Tuple[float, float]]) -> None:
    """
    Checks if there are no gaps between consecutive frequency bands.
    This function is deprecated and replaced by check_all_bins_covered.
    """
    # This function is no longer used for its original purpose.
    # It will be replaced by a more comprehensive check.
    pass

def check_no_overlap(band_specs: List[Tuple[float, float]]) -> None:
    """
    Checks if there is no overlap between consecutive frequency bands.

    Args:
        band_specs (List[Tuple[float, float]]): A list of tuples, where each tuple
                                                 represents a frequency band [fstart, fend).

    Raises:
        ValueError: If there is an overlap between bands.
    """
    for i in range(len(band_specs) - 1):
        if band_specs[i][1] > band_specs[i+1][0]:
            raise ValueError(f"Overlap detected between band {i} ({band_specs[i]}) and band {i+1} ({band_specs[i+1]})")

def check_nonzero_bandwidth(band_specs: List[Tuple[float, float]]) -> None:
    """
    Checks if all frequency bands have a non-zero bandwidth.

    Args:
        band_specs (List[Tuple[float, float]]): A list of tuples, where each tuple
                                                 represents a frequency band [fstart, fend).

    Raises:
        ValueError: If any band has a zero or negative bandwidth.
    """
    for i, (fstart, fend) in enumerate(band_specs):
        if fend - fstart <= 0:
            raise ValueError(f"Band {i} has non-positive bandwidth: [{fstart}, {fend})")

def check_all_bins_covered(band_specs: List[Tuple[float, float]], n_freq: int) -> None:
    """
    Checks if all frequency bins from 0 to n_freq are covered by at least one band.

    Args:
        band_specs (List[Tuple[float, float]]): A list of tuples, where each tuple
                                                 represents a frequency band [fstart, fend).
        n_freq (int): The total number of frequency bins (e.g., n_fft // 2 + 1).

    Raises:
        ValueError: If any frequency bin is not covered by any band.
    """
    covered_bins = torch.zeros(n_freq, dtype=torch.bool)
    for fstart, fend in band_specs:
        start_idx = int(fstart)
        end_idx = int(fend)
        covered_bins[start_idx:end_idx] = True
    
    if not torch.all(covered_bins):
        uncovered_indices = torch.nonzero(~covered_bins).squeeze().tolist()
        raise ValueError(f"Not all frequency bins are covered. Uncovered bins: {uncovered_indices}")

# Common stem definitions
VDBO_STEMS = ["vocals", "bass", "drums", "other"]
DME_STEMS = ["drums", "music", "effects"]