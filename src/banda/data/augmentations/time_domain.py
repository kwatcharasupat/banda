#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

import numpy as np
import torch
from typing import Optional

from banda.data.augmentations.base import PreMixTransform, PostMixTransform
from banda.data.batch_types import TorchInputAudioDict
from banda.data.types import Identifier, NumPySourceDict


class Gain(PreMixTransform):
    """
    Applies a random gain to the audio.
    """

    def __init__(self, min_gain_db: float = -6.0, max_gain_db: float = 6.0) -> None:
        """
        Args:
            min_gain_db (float): Minimum gain in dB.
            max_gain_db (float): Maximum gain in dB.
        """
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

    def __call__(self, *, sources: NumPySourceDict, identifier: Identifier) -> NumPySourceDict:
        """
        Apply random gain to each source.

        Args:
            sources (NumPySourceDict): Dictionary containing source data.
            identifier (Identifier): Identifier for the source data.

        Returns:
            NumPySourceDict: Transformed source data with applied gain.
        """
        gain_db = torch.empty(1).uniform_(self.min_gain_db, self.max_gain_db).item()
        gain_linear = 10**(gain_db / 20.0)
        return {key: source * gain_linear for key, source in sources.items()}


class AddBackgroundNoise(PreMixTransform):
    """
    Adds background noise to the mixture.
    """

    def __init__(self, noise_level_db: float = -30.0) -> None:
        """
        Args:
            noise_level_db (float): Desired noise level in dB.
        """
        self.noise_level_db = noise_level_db

    def __call__(self, *, sources: NumPySourceDict, identifier: Identifier) -> NumPySourceDict:
        """
        Add background noise to the mixture. This transform assumes a 'mixture' key exists
        or will be computed later. For simplicity, this example adds noise to all sources.
        In a real scenario, you might add noise only to the mixture or specific sources.

        Args:
            sources (NumPySourceDict): Dictionary containing source data.
            identifier (Identifier): Identifier for the source data.

        Returns:
            NumPySourceDict: Transformed source data with added noise.
        """
        # This is a simplified example. In a real scenario, you'd load actual noise.
        # Here, we generate random noise.
        if not sources:
            return sources

        # Get shape from first source
        first_source_key = next(iter(sources))
        shape = sources[first_source_key].shape
        
        noise = np.random.randn(*shape).astype(sources[first_source_key].dtype)
        
        # Normalize noise to desired level
        noise_rms = np.sqrt(np.mean(noise**2))
        target_noise_rms = 10**(self.noise_level_db / 20.0)
        if noise_rms > 0:
            noise = noise * (target_noise_rms / noise_rms)

        return {key: source + noise for key, source in sources.items()}


class NormalizeAudio(PostMixTransform):
    """
    Normalizes the audio in the TorchInputAudioDict to a target RMS level.
    """

    def __init__(self, target_rms_db: float = -20.0) -> None:
        """
        Args:
            target_rms_db (float): Target RMS level in dB.
        """
        self.target_rms_db = target_rms_db

    def __call__(self, *, audio_dict: TorchInputAudioDict, identifier: Identifier) -> TorchInputAudioDict:
        """
        Normalize the mixture and sources in the audio dictionary.

        Args:
            audio_dict (TorchInputAudioDict): Dictionary containing audio data.
            identifier (Identifier): Identifier for the audio data.

        Returns:
            TorchInputAudioDict: Transformed audio data with normalized levels.
        """
        target_rms_linear = 10**(self.target_rms_db / 20.0)

        def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
            if tensor is None:
                return None
            rms = torch.sqrt(torch.mean(tensor**2))
            if rms > 1e-8:  # Avoid division by zero
                return tensor * (target_rms_linear / rms)
            return tensor

        normalized_mixture = _normalize_tensor(audio_dict.mixture)
        normalized_sources = {
            key: _normalize_tensor(source) for key, source in audio_dict.sources.items()
        }

        return TorchInputAudioDict(mixture=normalized_mixture, sources=normalized_sources)